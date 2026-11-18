//! Attention: the ICL attention bodies on tensor cores, and attention over 16 keys.

#![expect(
    unsafe_code,
    reason = "shared memory, tensor-core intrinsics, and unchecked indexing; each unsafe block states its invariant"
)]
#![expect(
    clippy::no_mangle_with_rust_abi,
    reason = "cuda-oxide exports kernels and device functions with #[no_mangle] and the Rust ABI"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "a kernel's parameters are its launch ABI, which device bodies take on"
)]
#![expect(clippy::too_many_lines, reason = "a kernel body is one loop nest, kept whole")]
#![expect(clippy::many_single_char_names, reason = "the math's names: q, k, v, m, n, k")]
#![expect(
    clippy::similar_names,
    reason = "paired high and low parts such as w_hi and w_lo"
)]
#![expect(
    clippy::needless_range_loop,
    reason = "#[unroll] loops index register arrays in step"
)]
#![expect(
    clippy::cast_ptr_alignment,
    reason = "vector loads at offsets aligned by construction"
)]
#![expect(clippy::inline_always, reason = "device helpers inline into their kernels")]
#![allow(clippy::wildcard_imports, reason = "the other ops, through ops/mod.rs")]

use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::convert::{cvt_f16x2_f32, cvt_f32x2_f16x2};
use cuda_device::vector::F32x4;
use cuda_device::wmma::{ldmatrix_x4, ldmatrix_x4_trans, mma_m16n8k16_f16_f16, mma_m16n8k16_f32_f16};
use cuda_device::{DisjointSlice, SharedArray, device, thread, warp};

use super::*;

/// Attention head width.
pub(crate) const D: usize = 64;

/// u32 index of 16-byte chunk `chunk` (0..8) of row `row` in a 64-wide FP16
/// tile: chunks are XOR-swizzled by the row so that `ldmatrix` reads of 8
/// rows at one column hit 8 different bank groups.
#[inline(always)]
pub(crate) fn swizzled(row: usize, chunk: usize) -> usize {
    row * 32 + ((chunk ^ (row & 7)) * 4)
}

/// FlashAttention-2 style partial attention with tensor cores. `KW` keys per
/// warp per 64-key stage: 64 when the four warps own different 16-query row
/// groups (`DECODE` false, 64 queries per block), 16 when all four share one
/// 16-row group and split the stage's keys (`DECODE` true).
///
/// Keys and values are FP16, copied asynchronously, or with `FP8` E4M3 bytes,
/// loaded into registers a stage ahead and converted to FP16 in shared memory;
/// their scales are folded into the queries and outputs outside. With `HALF`,
/// both products accumulate in FP16 within a stage (twice the
/// FP32-accumulator rate) and are added to FP32 once per stage.
#[device]
pub(crate) fn fast_body<const KW: usize, const DECODE: bool, const HALF: bool, const FP8: bool>(
    q: &[u16],
    k: &[u8],
    v: &[u8],
    part: &mut DisjointSlice<f32>,
    lse: &mut DisjointSlice<f32>,
    qp: u32,
    np: u32,
    n_valid: u32,
    chunk_tiles: u32,
    splits: u32,
    scale_log2: f32,
) {
    static mut QS: SharedArray<u32, { 64 * 32 }, 128> = SharedArray::UNINIT;
    static mut KS: SharedArray<u32, { 2 * 64 * 32 }, 128> = SharedArray::UNINIT;
    static mut VS: SharedArray<u32, { 2 * 64 * 32 }, 128> = SharedArray::UNINIT;
    let nt_count = KW / 8; // 8-key n-tiles of scores per warp (8 or 2)
    let rows_per_block: usize = if DECODE { 16 } else { 64 };

    let tid = thread::threadIdx_x() as usize;
    let warp_id = tid / 32;
    let lane = tid % 32;
    // Query blocks on x, so blocks that read the same keys and values run
    // together and share them in L2.
    let block = thread::blockIdx_y() as usize;
    let (bh, chunk) = (block / splits as usize, block % splits as usize);
    let q0 = thread::blockIdx_x() as usize * rows_per_block;
    let (qp, np) = (qp as usize, np as usize);
    let first_tile = chunk * chunk_tiles as usize;
    // SAFETY: `QS`, `KS`, and `VS` are this function's own shared statics, reached only through
    // these pointers; each kernel calls `fast_body` once, and the indices below stay in the arrays'
    // sizes, ordered by the barriers.
    let qs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(QS)) };
    // SAFETY: as above.
    let ks = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KS)) };
    // SAFETY: as above.
    let vs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VS)) };

    // Q tile, then K/V stage 0, as two async-copy groups.
    // SAFETY: `row < rows_per_block` and `q0 + rows_per_block <= qp` (`qp` is 16 for the one-block
    // decode grid, else a multiple of 64: `padded_query_rows`), and `bh < pairs` by the grid, so
    // the source is 16 aligned bytes inside `q` `[pairs, qp, 64]`; the destination `swizzled(row,
    // c) + 4 <= 64 * 32`, 16-byte aligned. QS is read only after `cp_async_wait_group` and a
    // barrier.
    unsafe {
        let mut i = tid;
        while i < rows_per_block * 8 {
            let (row, c) = (i / 8, i % 8);
            let src = q.as_ptr().add((bh * qp + q0 + row) * 64 + c * 8).cast::<u32>();
            cp_async_cg_16(qs.add(swizzled(row, c)), src);
            i += 128;
        }
        cp_async_commit_group();
    }
    // FP16: two async 16-byte copies per thread and tensor.
    // SAFETY: `tile < np / 64` (see `tiles` below) and `bh < pairs`, so the 16 bytes at byte `off`
    // lie in the FP16 `[pairs, np, 64]` caches, 16-byte aligned; `dst + 4 <= (buf + 1) * 2048` with
    // `buf < 2`. The stage's previous readers passed the barrier that ends their stage, and it is
    // read only after a wait and a barrier.
    let load_stage = |tile: usize, buf: usize| unsafe {
        let base = (bh * np + tile * 64) * 64;
        let mut i = tid;
        while i < 64 * 8 {
            let (row, c) = (i / 8, i % 8);
            let off = 2 * (base + row * 64 + c * 8);
            let dst = buf * 64 * 32 + swizzled(row, c);
            cp_async_cg_16(ks.add(dst), k.as_ptr().add(off).cast::<u32>());
            cp_async_cg_16(vs.add(dst), v.as_ptr().add(off).cast::<u32>());
            i += 128;
        }
        cp_async_commit_group();
    };
    // FP8: two 16-byte chunks of 16 keys' or values' dims per thread and
    // tensor, `[K0, K1, V0, V1]`, as chunk index i = tid + 128 j: row i / 4,
    // dims (i % 4) * 16 on.
    let fetch8 = |tile: usize| -> [[u32; 4]; 4] {
        let base = (bh * np + tile * 64) * 64;
        let mut out = [[0u32; 4]; 4];
        #[unroll]
        for j in 0..2 {
            let i = tid + 128 * j;
            let off = base + (i / 4) * 64 + (i % 4) * 16;
            // SAFETY: `i < 256`: row `i / 4 < 64` of tile `tile < np / 64`, bytes `(i % 4) *
            // 16..+16` of 64, so 16 aligned bytes inside the FP8 `[pairs, np, 64]` caches.
            unsafe {
                out[j] = (*k.as_ptr().add(off).cast::<F32x4>()).0.map(f32::to_bits);
                out[2 + j] = (*v.as_ptr().add(off).cast::<F32x4>()).0.map(f32::to_bits);
            }
        }
        out
    };
    // Chunk t of `[K0, K1, V0, V1]`: plain index loops, so every index is a
    // constant once unrolled and the chunks stay in registers.
    let store8 = |bytes: [[u32; 4]; 4], buf: usize| {
        #[unroll]
        for t in 0..4 {
            let shared = if t < 2 { ks } else { vs };
            let i = tid + 128 * (t % 2);
            let (row, c) = (i / 4, (i % 4) * 2);
            #[unroll]
            for half in 0..2 {
                let at = buf * 64 * 32 + swizzled(row, c + half);
                #[unroll]
                for w in 0..2 {
                    let word = bytes[t][half * 2 + w];
                    #[expect(clippy::cast_possible_truncation, reason = "The word's two halves")]
                    let (low, high) = (word as u16, (word >> 16) as u16);
                    // SAFETY: `buf < 2`, `row < 64`, chunk `c + half < 8`, and `2 * w + 1 < 4`:
                    // inside the 16-byte chunk at `at`, within `KS` or `VS`; each thread writes its
                    // own words, into the stage no thread reads before the next barrier.
                    unsafe {
                        *shared.add(at + 2 * w) = e4m3x2_to_f16x2(low);
                        *shared.add(at + 2 * w + 1) = e4m3x2_to_f16x2(high);
                    }
                }
            }
        }
    };
    let mut ahead = [[0u32; 4]; 4];
    if FP8 {
        store8(fetch8(first_tile), 0);
    } else {
        load_stage(first_tile, 0);
    }

    let row_base = if DECODE { 0 } else { warp_id * 16 };
    let key_base = if DECODE { warp_id * 16 } else { 0 };
    let r = lane / 4; // fragment rows r and r + 8
    let col = (lane % 4) * 2; // fragment columns col and col + 1
    let mut qf = [[0u32; 4]; 4]; // Q A-fragments for the four 16-dim k-steps
    let mut acc = [[0.0f32; 4]; 8]; // output: eight 8-dim n-tiles
    let (mut m0, mut m1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let (mut l0, mut l1) = (0.0f32, 0.0f32);

    // The cache holds np / 64 tiles, so the last chunk can be shorter.
    let tiles = (chunk_tiles as usize).min(np / 64 - first_tile);
    for t in 0..tiles {
        let buf = t % 2;
        if FP8 {
            // The next stage's bytes load during this stage; its buffer was
            // last read before the previous stage's closing barrier.
            if t + 1 < tiles {
                ahead = fetch8(first_tile + t + 1);
            }
            if t == 0 {
                // SAFETY: waits for this thread's Q copy; the barrier after it publishes Q and the
                // stage-0 stores.
                unsafe { cp_async_wait_group(0) };
                thread::sync_threads();
            }
        } else {
            if t + 1 < tiles {
                load_stage(first_tile + t + 1, (t + 1) % 2);
            } else {
                // SAFETY: commits an empty group, so that the wait below always leaves only the
                // next stage's group pending.
                unsafe { cp_async_commit_group() };
            }
            // SAFETY: leaves only the newest group (the next stage) pending: Q and this stage's
            // copies are complete, and the barrier after it publishes them.
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();
        }
        if t == 0 {
            #[unroll]
            for kstep in 0..4 {
                let row = row_base + lane % 16;
                let c = kstep * 2 + lane / 16;
                // SAFETY: row `row_base + lane % 16` is below the 16 (decode) or 64 Q rows copied,
                // chunk `c < 8`: an aligned 16-byte row of `QS`, published by the barrier above;
                // the whole warp executes it.
                qf[kstep] = unsafe { ldmatrix_x4(qs.add(swizzled(row, c))) };
            }
        }

        // Scores S = Q Kᵀ for this warp's KW keys.
        let mut s = [[0.0f32; 4]; 8]; // only the first `nt_count` are used
        let mut s16 = [[0u32; 2]; 8];
        #[unroll]
        for kstep in 0..4 {
            #[unroll]
            for pair in 0..nt_count / 2 {
                let row = key_base + pair * 16 + (lane / 16) * 8 + lane % 8;
                let c = kstep * 2 + (lane / 8) % 2;
                // SAFETY: row `key_base + pair * 16 + 15 < key_base + KW <= 64` and chunk `c < 8`:
                // an aligned 16-byte row of stage `buf` of `KS`, published by the barrier above;
                // the whole warp executes it.
                let b = unsafe { ldmatrix_x4(ks.add(buf * 64 * 32 + swizzled(row, c))) };
                if HALF {
                    // SAFETY: every lane runs these m16n8k16 products on its `ldmatrix` fragments:
                    // the loop bounds are compile-time or block-uniform, and no lane has exited.
                    s16[2 * pair] = unsafe { mma_m16n8k16_f16_f16(s16[2 * pair], qf[kstep], [b[0], b[1]]) };
                    // SAFETY: as above.
                    s16[2 * pair + 1] =
                        unsafe { mma_m16n8k16_f16_f16(s16[2 * pair + 1], qf[kstep], [b[2], b[3]]) };
                } else {
                    // SAFETY: as above.
                    s[2 * pair] = unsafe { mma_m16n8k16_f32_f16(s[2 * pair], qf[kstep], [b[0], b[1]]) };
                    // SAFETY: as above.
                    s[2 * pair + 1] =
                        unsafe { mma_m16n8k16_f32_f16(s[2 * pair + 1], qf[kstep], [b[2], b[3]]) };
                }
            }
        }
        if HALF {
            #[unroll]
            for nt in 0..nt_count {
                let (a0, a1) = cvt_f32x2_f16x2(s16[nt][0]);
                let (a2, a3) = cvt_f32x2_f16x2(s16[nt][1]);
                s[nt] = [a0, a1, a2, a3];
            }
        }

        // Scale into the exp2 domain, mask keys at or past N, row maxima.
        let key0 = (first_tile + t) * 64 + key_base;
        let (mut max0, mut max1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        #[unroll]
        for nt in 0..nt_count {
            #[unroll]
            for j in 0..2 {
                let valid = key0 + nt * 8 + col + j < n_valid as usize;
                s[nt][j] = if valid {
                    s[nt][j] * scale_log2
                } else {
                    f32::NEG_INFINITY
                };
                s[nt][2 + j] = if valid {
                    s[nt][2 + j] * scale_log2
                } else {
                    f32::NEG_INFINITY
                };
                max0 = max0.max(s[nt][j]);
                max1 = max1.max(s[nt][2 + j]);
            }
        }
        max0 = max0.max(warp::shuffle_xor_f32(max0, 1));
        max1 = max1.max(warp::shuffle_xor_f32(max1, 1));
        max0 = max0.max(warp::shuffle_xor_f32(max0, 2));
        max1 = max1.max(warp::shuffle_xor_f32(max1, 2));
        let new0 = m0.max(max0);
        let new1 = m1.max(max1);
        // A row with no valid key yet keeps zero weight instead of NaN.
        let base0 = if new0 == f32::NEG_INFINITY { 0.0 } else { new0 };
        let base1 = if new1 == f32::NEG_INFINITY { 0.0 } else { new1 };
        let alpha0 = (m0 - base0).exp2();
        let alpha1 = (m1 - base1).exp2();
        let (mut sum0, mut sum1) = (0.0f32, 0.0f32);
        #[unroll]
        for nt in 0..nt_count {
            #[unroll]
            for j in 0..2 {
                s[nt][j] = (s[nt][j] - base0).exp2();
                s[nt][2 + j] = (s[nt][2 + j] - base1).exp2();
                sum0 += s[nt][j];
                sum1 += s[nt][2 + j];
            }
        }
        sum0 += warp::shuffle_xor_f32(sum0, 1);
        sum1 += warp::shuffle_xor_f32(sum1, 1);
        sum0 += warp::shuffle_xor_f32(sum0, 2);
        sum1 += warp::shuffle_xor_f32(sum1, 2);
        l0 = l0 * alpha0 + sum0;
        l1 = l1 * alpha1 + sum1;
        m0 = new0;
        m1 = new1;
        #[unroll]
        for nt in 0..8 {
            acc[nt][0] *= alpha0;
            acc[nt][1] *= alpha0;
            acc[nt][2] *= alpha1;
            acc[nt][3] *= alpha1;
        }

        // O += P V: the score accumulators become the A operand directly.
        let mut pv = [[0u32; 2]; 8];
        #[unroll]
        for kk in 0..KW / 16 {
            let a = [
                cvt_f16x2_f32(s[2 * kk][0], s[2 * kk][1]),
                cvt_f16x2_f32(s[2 * kk][2], s[2 * kk][3]),
                cvt_f16x2_f32(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                cvt_f16x2_f32(s[2 * kk + 1][2], s[2 * kk + 1][3]),
            ];
            #[unroll]
            for dp in 0..4 {
                let row = key_base + kk * 16 + ((lane / 8) % 2) * 8 + lane % 8;
                let c = dp * 2 + lane / 16;
                // SAFETY: row `key_base + kk * 16 + 15 < 64` and chunk `c < 8`: an aligned 16-byte
                // row of stage `buf` of `VS`, published by the barrier above; the whole warp
                // executes it.
                let b = unsafe { ldmatrix_x4_trans(vs.add(buf * 64 * 32 + swizzled(row, c))) };
                if HALF {
                    // SAFETY: full-warp products, as for the scores.
                    pv[2 * dp] = unsafe { mma_m16n8k16_f16_f16(pv[2 * dp], a, [b[0], b[1]]) };
                    // SAFETY: as above.
                    pv[2 * dp + 1] = unsafe { mma_m16n8k16_f16_f16(pv[2 * dp + 1], a, [b[2], b[3]]) };
                } else {
                    // SAFETY: as above.
                    acc[2 * dp] = unsafe { mma_m16n8k16_f32_f16(acc[2 * dp], a, [b[0], b[1]]) };
                    // SAFETY: as above.
                    acc[2 * dp + 1] = unsafe { mma_m16n8k16_f32_f16(acc[2 * dp + 1], a, [b[2], b[3]]) };
                }
            }
        }
        if HALF {
            #[unroll]
            for nt in 0..8 {
                let (c0, c1) = cvt_f32x2_f16x2(pv[nt][0]);
                let (c2, c3) = cvt_f32x2_f16x2(pv[nt][1]);
                acc[nt] = [acc[nt][0] + c0, acc[nt][1] + c1, acc[nt][2] + c2, acc[nt][3] + c3];
            }
        }
        if FP8 && t + 1 < tiles {
            store8(ahead, (t + 1) % 2);
        }
        thread::sync_threads();
    }

    let out_row = (bh * splits as usize + chunk) * qp + q0;
    if !DECODE {
        // Each warp owns its 16 rows: normalize and write.
        let (row0, row1) = (out_row + row_base + r, out_row + row_base + r + 8);
        #[unroll]
        for nt in 0..8 {
            #[unroll]
            for j in 0..2 {
                let dim = nt * 8 + col + j;
                // SAFETY: `row_base + r + 8 < 64` and `q0 + 64 <= qp`, so both rows lie in `part`
                // `[pairs * splits, qp, 64]` (the workspace reserves chunks times padded rows);
                // each (row, dim) belongs to one lane of one warp of one block.
                unsafe {
                    *part.get_unchecked_mut(row0 * 64 + dim) = acc[nt][j] / l0;
                    *part.get_unchecked_mut(row1 * 64 + dim) = acc[nt][2 + j] / l1;
                }
            }
        }
        if lane.is_multiple_of(4) {
            // SAFETY: the same rows, in `lse` `[pairs * splits, qp]`; only lane `4 * r` writes
            // them.
            unsafe {
                *lse.get_unchecked_mut(row0) = m0 + l0.log2();
                *lse.get_unchecked_mut(row1) = m1 + l1.log2();
            }
        }
        return;
    }

    // DECODE: combine the four warps' partial results through shared memory.
    let accs = ks.cast::<f32>(); // 4 warps x 16 rows x 64 dims
    let stats = qs.cast::<f32>(); // m and l: 4 warps x 16 rows each
    // SAFETY: `accs` reuses `KS` (4 * 16 * 64 = 2 * 64 * 32 words) and `stats` the first 128 words
    // of `QS`; the key loop's closing barrier retired their readers, and no copy is pending (only
    // an empty group). Each (warp, row, dim) and stat slot belongs to one lane.
    unsafe {
        #[unroll]
        for nt in 0..8 {
            #[unroll]
            for j in 0..2 {
                let dim = nt * 8 + col + j;
                *accs.add((warp_id * 16 + r) * 64 + dim) = acc[nt][j];
                *accs.add((warp_id * 16 + r + 8) * 64 + dim) = acc[nt][2 + j];
            }
        }
        if lane.is_multiple_of(4) {
            *stats.add(warp_id * 16 + r) = m0;
            *stats.add(warp_id * 16 + r + 8) = m1;
            *stats.add(64 + warp_id * 16 + r) = l0;
            *stats.add(64 + warp_id * 16 + r + 8) = l1;
        }
    }
    thread::sync_threads();
    let mut e = tid;
    while e < 16 * 64 {
        let (row, dim) = (e / 64, e % 64);
        // SAFETY: `row < 16`, `w < 4`: inside `accs` and `stats`, published by the barrier above.
        // The decode grid has `qp = 16` and `q0 = 0`, so `out_row + row` is a row of `part` and
        // `lse`; each (row, dim) is one thread's, and only `dim == 0` writes `lse`.
        unsafe {
            let mut m = f32::NEG_INFINITY;
            #[unroll]
            for w in 0..4 {
                m = m.max(*stats.add(w * 16 + row));
            }
            let (mut total, mut o) = (0.0f32, 0.0f32);
            #[unroll]
            for w in 0..4 {
                let weight = (*stats.add(w * 16 + row) - m).exp2();
                total += *stats.add(64 + w * 16 + row) * weight;
                o += *accs.add((w * 16 + row) * 64 + dim) * weight;
            }
            *part.get_unchecked_mut((out_row + row) * 64 + dim) = o / total;
            if dim == 0 {
                *lse.get_unchecked_mut(out_row + row) = m + total.log2();
            }
        }
        e += 128;
    }
}

/// Softmax attention with 16-dimensional heads on tensor cores. Sequence `s`
/// (grid z) and head (grid y) select keys and values at
/// `(s / inner) * kv_outer + (s % inner) * kv_inner + head * kv_head + t * kv_tok`
/// past `k_base` and `v_base` (the column caches and token buffers fit this), queries at
/// `s * q_seq + i * q_tok + head * 16`, outputs at `s * o_seq + i * o_tok + head * 16`.
/// One warp per 16 queries of a (sequence, head), four warps per block sharing each 64-key chunk of K and V, which
/// the block converts to FP16 once in shared memory; an online softmax in the
/// exp2 domain. With `SPLIT`, every product is split into high and low FP16
/// parts as in [`linear16_body`].
/// Grid: (ceil(lq / 64), 8, sequences); block: 128 threads.
#[device]
pub(crate) fn attention16_tc_body<const SPLIT: bool>(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut DisjointSlice<f32>,
    lq: u32,
    lk: u32,
    inner: u32,
    q_seq: u32,
    q_tok: u32,
    k_base: u32,
    v_base: u32,
    kv_outer: u32,
    kv_inner: u32,
    kv_tok: u32,
    kv_head: u32,
    o_seq: u32,
    o_tok: u32,
    scale: f32,
) {
    // K: [64 keys][16 dims] FP16, 12 u32 per key; Vᵀ: [16 dims][64 keys], 36
    // u32 per dim. The padding keeps fragment reads free of bank conflicts.
    static mut KH: SharedArray<u32, { 64 * 12 }, 16> = SharedArray::UNINIT;
    static mut KL: SharedArray<u32, { 64 * 12 }, 16> = SharedArray::UNINIT;
    static mut VH: SharedArray<u32, { 16 * 36 }, 16> = SharedArray::UNINIT;
    static mut VL: SharedArray<u32, { 16 * 36 }, 16> = SharedArray::UNINIT;
    // SAFETY: `KH`, `KL`, `VH`, and `VL` are this function's own shared statics, reached only
    // through these pointers; the indices below stay in their sizes, between the key loop's
    // barriers.
    let kh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KH)) };
    // SAFETY: as above.
    let kl = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KL)) };
    // SAFETY: as above.
    let vh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VH)) };
    // SAFETY: as above.
    let vl = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VL)) };
    let tid = thread::threadIdx_x() as usize;
    let (warp_id, lane) = (tid / 32, tid % 32);
    let (lr, lc) = (lane / 4, (lane % 4) * 2);
    let query0 = thread::blockIdx_x() as usize * 64 + warp_id * 16;
    let head = thread::blockIdx_y() as usize;
    let seq = thread::blockIdx_z() as usize;
    let (lq, lk, inner) = (lq as usize, lk as usize, inner as usize);
    let kv0 = (seq / inner) * kv_outer as usize + (seq % inner) * kv_inner as usize + head * kv_head as usize;
    let kv_tok = kv_tok as usize;
    let (qp, kp, vp) = (q.as_ptr(), &k[k_base as usize..], &v[v_base as usize..]);
    let (kp, vp) = (kp.as_ptr(), vp.as_ptr());

    // Q as the A fragment: rows lr and lr + 8, dims lc, lc + 1, lc + 8, lc + 9.
    let q_at = |row: usize, dim: usize| -> (f32, f32) {
        let query = query0 + row;
        if query < lq {
            let at = seq * q_seq as usize + query * q_tok as usize + head * 16 + dim;
            // SAFETY: `query < lq` and `dim + 1 < 16`: the launchers' `SmallAttention` layouts
            // place each sequence's queries at `seq * q_seq + query * q_tok`, with the head's 16
            // values inside `q`.
            unsafe { (*qp.add(at), *qp.add(at + 1)) }
        } else {
            (0.0, 0.0)
        }
    };
    let quads = [
        q_at(lr, lc),
        q_at(lr + 8, lc),
        q_at(lr, lc + 8),
        q_at(lr + 8, lc + 8),
    ];
    let mut qa = [0u32; 4];
    let mut qal = [0u32; 4];
    #[unroll]
    for i in 0..4 {
        qa[i] = pack_f16(quads[i].0, quads[i].1);
        if SPLIT {
            qal[i] = pack_lo(quads[i].0, quads[i].1);
        }
    }
    let scale_log2 = scale * core::f32::consts::LOG2_E;

    let mut acc = [[0.0f32; 4]; 2];
    let mut corr = [[0.0f32; 4]; 2];
    let (mut m0, mut m1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let (mut l0, mut l1) = (0.0f32, 0.0f32);
    let mut chunk = 0;
    while chunk < lk {
        // Each thread converts a key-pair's two dims of K and V.
        thread::sync_threads();
        #[unroll]
        for i in 0..4 {
            let index = tid + i * 128; // 32 key pairs x 16 dims
            let (pair, dim) = (index / 16, index % 16);
            let key = chunk + pair * 2;
            let at = kv0 + key * kv_tok + dim;
            let (k0, v0) = if key < lk {
                // SAFETY: `key < lk` and `dim < 16`: the launchers' layouts place key `key` of this
                // (sequence, head) at `kv0 + key * kv_tok` past `k_base` and `v_base`, 16 values
                // inside `k` and `v`.
                unsafe { (*kp.add(at), *vp.add(at)) }
            } else {
                (0.0, 0.0)
            };
            let (k1, v1) = if key + 1 < lk {
                // SAFETY: as above, for key `key + 1 < lk`.
                unsafe { (*kp.add(at + kv_tok), *vp.add(at + kv_tok)) }
            } else {
                (0.0, 0.0)
            };
            // SAFETY: `pair < 32`, `dim < 16`: half-word `(2 * pair + 1) * 24 + dim < 64 * 24` of
            // `KH` and `KL`, word `dim * 36 + pair < 16 * 36` of `VH` and `VL`; each thread writes
            // its own slots after the barrier that retired the previous chunk's readers.
            unsafe {
                // K is stored by key, so a key pair writes two half-words.
                let kh16 = kh.cast::<u16>();
                *kh16.add((pair * 2) * 24 + dim) = (k0 as f16).to_bits();
                *kh16.add((pair * 2 + 1) * 24 + dim) = (k1 as f16).to_bits();
                *vh.add(dim * 36 + pair) = pack_f16(v0, v1);
                if SPLIT {
                    let kl16 = kl.cast::<u16>();
                    *kl16.add((pair * 2) * 24 + dim) =
                        (((k0 - (k0 as f16) as f32) * LO_SCALE) as f16).to_bits();
                    *kl16.add((pair * 2 + 1) * 24 + dim) =
                        (((k1 - (k1 as f16) as f32) * LO_SCALE) as f16).to_bits();
                    *vl.add(dim * 36 + pair) = pack_lo(v0, v1);
                }
            }
        }
        thread::sync_threads();
        if query0 < lq {
            let mut step = 0;
            while step < 2 && chunk + step * 32 < lk {
                let key0 = chunk + step * 32;
                // Scores for 32 keys: four 8-key n-tiles.
                let mut s = [[0.0f32; 4]; 4];
                #[unroll]
                for nt in 0..4 {
                    let at = (step * 32 + nt * 8 + lr) * 12 + lc / 2;
                    // SAFETY: row `step * 32 + nt * 8 + lr < 64` and word `lc / 2 + 4 < 12`: inside
                    // `KH`, published by the barrier above.
                    let kb = unsafe { [*kh.add(at), *kh.add(at + 4)] };
                    let mut c = [0.0f32; 4];
                    // SAFETY: a full-warp product: `query0` is uniform within the warp, and so are
                    // the step bounds.
                    c = unsafe { mma_m16n8k16_f32_f16(c, qa, kb) };
                    if SPLIT {
                        // SAFETY: as for `kb`, inside `KL`.
                        let kbl = unsafe { [*kl.add(at), *kl.add(at + 4)] };
                        let cl = add_correction([0.0; 4], qa, qal, kb, kbl);
                        #[unroll]
                        for e in 0..4 {
                            c[e] += cl[e] * (1.0 / LO_SCALE);
                        }
                    }
                    #[unroll]
                    for e in 0..4 {
                        let valid = key0 + nt * 8 + lc + e % 2 < lk;
                        s[nt][e] = if valid {
                            c[e] * scale_log2
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                }
                let (mut max0, mut max1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                #[unroll]
                for nt in 0..4 {
                    max0 = max0.max(s[nt][0]).max(s[nt][1]);
                    max1 = max1.max(s[nt][2]).max(s[nt][3]);
                }
                max0 = max0.max(warp::shuffle_xor_f32(max0, 1));
                max1 = max1.max(warp::shuffle_xor_f32(max1, 1));
                max0 = max0.max(warp::shuffle_xor_f32(max0, 2));
                max1 = max1.max(warp::shuffle_xor_f32(max1, 2));
                let new0 = m0.max(max0);
                let new1 = m1.max(max1);
                let alpha0 = (m0 - new0).exp2();
                let alpha1 = (m1 - new1).exp2();
                let (mut sum0, mut sum1) = (0.0f32, 0.0f32);
                #[unroll]
                for nt in 0..4 {
                    s[nt][0] = (s[nt][0] - new0).exp2();
                    s[nt][1] = (s[nt][1] - new0).exp2();
                    s[nt][2] = (s[nt][2] - new1).exp2();
                    s[nt][3] = (s[nt][3] - new1).exp2();
                    sum0 += s[nt][0] + s[nt][1];
                    sum1 += s[nt][2] + s[nt][3];
                }
                sum0 += warp::shuffle_xor_f32(sum0, 1);
                sum1 += warp::shuffle_xor_f32(sum1, 1);
                sum0 += warp::shuffle_xor_f32(sum0, 2);
                sum1 += warp::shuffle_xor_f32(sum1, 2);
                l0 = l0 * alpha0 + sum0;
                l1 = l1 * alpha1 + sum1;
                m0 = new0;
                m1 = new1;
                #[unroll]
                for nt in 0..2 {
                    acc[nt][0] *= alpha0;
                    acc[nt][1] *= alpha0;
                    acc[nt][2] *= alpha1;
                    acc[nt][3] *= alpha1;
                    if SPLIT {
                        corr[nt][0] *= alpha0;
                        corr[nt][1] *= alpha0;
                        corr[nt][2] *= alpha1;
                        corr[nt][3] *= alpha1;
                    }
                }

                // O += P V over two 16-key steps.
                #[unroll]
                for kk in 0..2 {
                    let (s0, s1) = (s[2 * kk], s[2 * kk + 1]);
                    let pa = [
                        pack_f16(s0[0], s0[1]),
                        pack_f16(s0[2], s0[3]),
                        pack_f16(s1[0], s1[1]),
                        pack_f16(s1[2], s1[3]),
                    ];
                    let pal = if SPLIT {
                        [
                            pack_lo(s0[0], s0[1]),
                            pack_lo(s0[2], s0[3]),
                            pack_lo(s1[0], s1[1]),
                            pack_lo(s1[2], s1[3]),
                        ]
                    } else {
                        [0u32; 4]
                    };
                    #[unroll]
                    for nt in 0..2 {
                        let at = (nt * 8 + lr) * 36 + (step * 32 + kk * 16 + lc) / 2;
                        // SAFETY: row `nt * 8 + lr < 16` and word `(step * 32 + kk * 16 + lc) / 2 +
                        // 4 < 36`: inside `VH`, published by the barrier above.
                        let vb = unsafe { [*vh.add(at), *vh.add(at + 4)] };
                        // SAFETY: a full-warp product, as for the scores.
                        acc[nt] = unsafe { mma_m16n8k16_f32_f16(acc[nt], pa, vb) };
                        if SPLIT {
                            // SAFETY: as for `vb`, inside `VL`.
                            let vbl = unsafe { [*vl.add(at), *vl.add(at + 4)] };
                            corr[nt] = add_correction(corr[nt], pa, pal, vb, vbl);
                        }
                    }
                }
                step += 1;
            }
        }
        chunk += 64;
    }

    if query0 >= lq {
        return;
    }
    #[unroll]
    for nt in 0..2 {
        #[unroll]
        for e in 0..4 {
            let query = query0 + lr + (e / 2) * 8;
            if query < lq {
                let mut value = acc[nt][e];
                if SPLIT {
                    value += corr[nt][e] * (1.0 / LO_SCALE);
                }
                let total = if e < 2 { l0 } else { l1 };
                let at = seq * o_seq as usize + query * o_tok as usize + head * 16 + nt * 8 + lc + e % 2;
                // SAFETY: `query < lq` and dim `nt * 8 + lc + e % 2 < 16`: the launchers' layouts
                // (`o_tok = 128`, `o_seq = lq * 128`) give each (sequence, query, head, dim) its
                // own slot in `out`, written by one lane.
                unsafe { *out.get_unchecked_mut(at) = value / total };
            }
        }
    }
}

/// Softmax attention of one 16-wide `query` to keys `[from, to)` at `kp`
/// and `vp` (`stride` values apart, 16-byte aligned), with scale 1 / 4 in
/// the exp2 domain and an online softmax key by key: the normalized output
/// into `acc`, which starts at zero.
///
/// # Safety
///
/// `kp` and `vp` must be 16-byte aligned and `stride` a multiple of 4, and for every key in
/// `[from, to)` the 16 values at `kp + key * stride` and at `vp + key * stride` must be readable
/// and not written during the call.
#[device]
pub(crate) unsafe fn attend16_f32(
    query: &[f32; 16],
    (kp, vp, stride): (*const f32, *const f32, usize),
    (from, to): (usize, usize),
    acc: &mut [f32; 16],
) {
    let scale_log2 = 0.25 * core::f32::consts::LOG2_E; // 1 / sqrt(16)
    let (mut max, mut total) = (f32::NEG_INFINITY, 0.0f32);
    let mut key = from;
    while key < to {
        let mut dot = 0.0f32;
        #[unroll]
        for c in 0..4 {
            // SAFETY: `key` is in `[from, to)` and `c * 4 + 4 <= 16`: by this function's contract,
            // 16 aligned readable bytes at `kp`.
            let k = unsafe { *kp.add(key * stride + c * 4).cast::<F32x4>() }.0;
            #[unroll]
            for e in 0..4 {
                dot += query[c * 4 + e] * k[e];
            }
        }
        let score = dot * scale_log2;
        if score > max {
            let alpha = (max - score).exp2();
            total *= alpha;
            #[unroll]
            for d in 0..16 {
                acc[d] *= alpha;
            }
            max = score;
        }
        let p = (score - max).exp2();
        total += p;
        #[unroll]
        for c in 0..4 {
            // SAFETY: as for the key, at `vp`.
            let v = unsafe { *vp.add(key * stride + c * 4).cast::<F32x4>() }.0;
            #[unroll]
            for e in 0..4 {
                acc[c * 4 + e] += p * v[e];
            }
        }
        key += 1;
    }
    #[unroll]
    for d in 0..16 {
        acc[d] /= total;
    }
}

/// The largest of `x` over the 16 lanes of a half warp.
#[inline(always)]
pub(crate) fn max16(x: f32) -> f32 {
    let x = x.max(warp::shuffle_xor_f32(x, 1));
    let x = x.max(warp::shuffle_xor_f32(x, 2));
    let x = x.max(warp::shuffle_xor_f32(x, 4));
    x.max(warp::shuffle_xor_f32(x, 8))
}

/// The sum of `x` over the 16 lanes of a half warp.
#[inline(always)]
pub(crate) fn sum16(x: f32) -> f32 {
    let x = x + warp::shuffle_xor_f32(x, 1);
    let x = x + warp::shuffle_xor_f32(x, 2);
    let x = x + warp::shuffle_xor_f32(x, 4);
    x + warp::shuffle_xor_f32(x, 8)
}
