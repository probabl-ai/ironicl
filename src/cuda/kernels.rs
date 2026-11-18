//! Ironicl's CUDA kernels, compiled to PTX by cuda-oxide's codegen backend
//! and embedded in the library, once per target in `IRONICL_CUDA_ARCH` (see
//! `build.rs`). A target's capabilities set `ironicl_cap_*` cfgs:
//! `ironicl_cap_fp8_cvt` (Ada and later) converts FP8 in hardware.
//!
//! ICL attention: every (estimator, head) pair attends with Q query rows to N
//! cached training rows, in Ironicl's `[estimators x heads, N, 64]` cache
//! layout. The attention kernels write a normalized partial output and its
//! log-sum-exp per chunk of N; `icl_attention_merge` combines the chunks.
//!
//! Each tensor-core family (`_half`, `_split`) has an `_f32` counterpart on
//! the CUDA cores for `optimize="safe"`, which never touches FP16.

use cuda_host::cuda_module;

#[cuda_module]
#[allow(
    unsafe_code, // Shared memory, tensor-core intrinsics, and unchecked indexing.
    clippy::no_mangle_with_rust_abi, // How cuda-oxide's `#[kernel]` exports entries.
    clippy::too_many_arguments, // A kernel's parameters are its launch ABI.
    clippy::too_many_lines,
    clippy::many_single_char_names, // The math's names: q, k, v, m, n, k.
    clippy::similar_names, // Paired high and low parts: `w_hi`, `w_lo`.
    clippy::needless_range_loop, // `#[unroll]` loops over register-array indices.
    clippy::inline_always, // Device helpers inline into their kernels.
    clippy::cast_ptr_alignment, // Vector loads at offsets aligned by construction.
    clippy::items_after_statements // `#[launch_bounds]` expands ahead of a kernel's shared statics.
)]
pub(super) mod device {
    use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
    #[cfg(ironicl_cap_fp8_cvt)]
    use cuda_device::convert::cvt_rn_f16x2_e4m3x2;
    use cuda_device::convert::{cvt_f16x2_f32, cvt_f32x2_f16x2};
    #[cfg(not(ironicl_cap_fp8_cvt))]
    use cuda_device::f16x2::fma_f16x2;
    use cuda_device::vector::F32x4;
    use cuda_device::wmma::{ldmatrix_x4, ldmatrix_x4_trans, mma_m16n8k16_f16_f16, mma_m16n8k16_f32_f16};
    use cuda_device::{DisjointSlice, SharedArray, device, kernel, launch_bounds, thread, warp};

    /// Attention head width.
    const D: usize = 64;

    /// u32 index of 16-byte chunk `chunk` (0..8) of row `row` in a 64-wide FP16
    /// tile: chunks are XOR-swizzled by the row so that `ldmatrix` reads of 8
    /// rows at one column hit 8 different bank groups.
    #[inline(always)]
    fn swizzled(row: usize, chunk: usize) -> usize {
        row * 32 + ((chunk ^ (row & 7)) * 4)
    }

    /// Two FP8 E4M3 values, the first in the low byte, as packed FP16. Ada and
    /// later convert in hardware; before them, each byte's exponent and mantissa
    /// shift into FP16's place, which scales the value by 2^-8 (the exponent biases
    /// differ by 8), and an exact FP16 multiply by 256 undoes that. E4M3
    /// subnormals land on FP16 values exactly, so both ways give the same result.
    #[inline(always)]
    fn e4m3x2_to_f16x2(pair: u16) -> u32 {
        #[cfg(ironicl_cap_fp8_cvt)]
        {
            cvt_rn_f16x2_e4m3x2(pair)
        }
        #[cfg(not(ironicl_cap_fp8_cvt))]
        {
            let word = u32::from(pair);
            let spread = (word & 0xff) | ((word & 0xff00) << 8); // bytes at bits 0 and 16
            let bits = ((spread & 0x007f_007f) << 7) | ((spread & 0x0080_0080) << 8);
            fma_f16x2(bits, 0x5c00_5c00, 0) // 256.0 in each lane
        }
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
    fn fast_body<const KW: usize, const DECODE: bool, const HALF: bool, const FP8: bool>(
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
        let qs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(QS)) };
        let ks = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KS)) };
        let vs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VS)) };

        // Q tile, then K/V stage 0, as two async-copy groups.
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
                        #[allow(clippy::cast_possible_truncation)] // The word's two halves.
                        let (low, high) = (word as u16, (word >> 16) as u16);
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
                    unsafe { cp_async_wait_group(0) };
                    thread::sync_threads();
                }
            } else {
                if t + 1 < tiles {
                    load_stage(first_tile + t + 1, (t + 1) % 2);
                } else {
                    unsafe { cp_async_commit_group() };
                }
                unsafe { cp_async_wait_group(1) };
                thread::sync_threads();
            }
            if t == 0 {
                #[unroll]
                for kstep in 0..4 {
                    let row = row_base + lane % 16;
                    let c = kstep * 2 + lane / 16;
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
                    let b = unsafe { ldmatrix_x4(ks.add(buf * 64 * 32 + swizzled(row, c))) };
                    if HALF {
                        s16[2 * pair] =
                            unsafe { mma_m16n8k16_f16_f16(s16[2 * pair], qf[kstep], [b[0], b[1]]) };
                        s16[2 * pair + 1] =
                            unsafe { mma_m16n8k16_f16_f16(s16[2 * pair + 1], qf[kstep], [b[2], b[3]]) };
                    } else {
                        s[2 * pair] = unsafe { mma_m16n8k16_f32_f16(s[2 * pair], qf[kstep], [b[0], b[1]]) };
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
                    let b = unsafe { ldmatrix_x4_trans(vs.add(buf * 64 * 32 + swizzled(row, c))) };
                    if HALF {
                        pv[2 * dp] = unsafe { mma_m16n8k16_f16_f16(pv[2 * dp], a, [b[0], b[1]]) };
                        pv[2 * dp + 1] = unsafe { mma_m16n8k16_f16_f16(pv[2 * dp + 1], a, [b[2], b[3]]) };
                    } else {
                        acc[2 * dp] = unsafe { mma_m16n8k16_f32_f16(acc[2 * dp], a, [b[0], b[1]]) };
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
                    unsafe {
                        *part.get_unchecked_mut(row0 * 64 + dim) = acc[nt][j] / l0;
                        *part.get_unchecked_mut(row1 * 64 + dim) = acc[nt][2 + j] / l1;
                    }
                }
            }
            if lane.is_multiple_of(4) {
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

    /// Tensor-core partial attention over FP16 keys and values, 64 queries per
    /// block (wide batches).
    #[kernel]
    pub fn icl_attention_wide(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<64, false, false, false>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// Tensor-core partial attention over FP16 keys and values, 16 queries per
    /// block, keys split over warps.
    #[kernel]
    pub fn icl_attention_decode(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<16, true, false, false>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// [`icl_attention_wide`] over FP8 keys and values, accumulating in FP16, for
    /// `optimize="fast"`.
    #[kernel]
    pub fn icl_attention_wide_fp8(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<64, false, true, true>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// [`icl_attention_decode`] over FP8 keys and values, for `optimize="fast"`.
    #[kernel]
    pub fn icl_attention_decode_fp8(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<16, true, false, true>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// Combines the chunks' partial outputs by their log2-domain log-sum-exp:
    /// one thread per output element of `[BH, Qp, 64]`.
    #[kernel]
    pub fn icl_attention_merge(
        part: &[f32],
        lse: &[f32],
        mut out: DisjointSlice<f32>,
        bh_count: u32,
        qp: u32,
        splits: u32,
    ) {
        let gid = thread::index_1d();
        let i = gid.get();
        let (qp, splits) = (qp as usize, splits as usize);
        if i >= bh_count as usize * qp * D {
            return;
        }
        let (bh, rest) = (i / (qp * D), i % (qp * D));
        let (q, d) = (rest / D, rest % D);
        let mut max = f32::NEG_INFINITY;
        for s in 0..splits {
            max = max.max(lse[(bh * splits + s) * qp + q]);
        }
        let (mut total, mut acc) = (0.0f32, 0.0f32);
        for s in 0..splits {
            let row = (bh * splits + s) * qp + q;
            let w = (lse[row] - max).exp2();
            total += w;
            acc += w * part[row * D + d];
        }
        if let Some(slot) = out.get_mut(gid) {
            *slot = acc / total;
        }
    }

    // ---------------------------------------------------------------------------
    // Dense stages in FP32, for products too small or narrow for the tensor-core
    // kernels.
    // ---------------------------------------------------------------------------

    /// `LayerNorm` over `width` values (at most 512) of each of `rows` rows (row
    /// strides `ldx`, `ldy`), epsilon 1e-5, population variance, written as
    /// [`store_activation`] does with `format`. One warp per row; each lane loads
    /// its values once, all loads in flight together, and keeps them in
    /// registers. 256 threads per block.
    #[kernel]
    pub fn layer_norm(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut y: DisjointSlice<f32>,
        mut y_hi: DisjointSlice<u16>,
        mut y_lo: DisjointSlice<u16>,
        rows: u32,
        width: u32,
        ldx: u32,
        ldy: u32,
        format: u32,
    ) {
        let global =
            thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize;
        let (row, lane) = (global / 32, global % 32);
        let (width, ldx, ldy) = (width as usize, ldx as usize, ldy as usize);
        if row >= rows as usize {
            return;
        }
        let mut values = [0.0f32; 16];
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0..16 {
            let i = lane + 32 * j;
            if i < width {
                values[j] = unsafe { *x.get_unchecked(row * ldx + i) };
                sum += values[j];
            }
        }
        #[allow(clippy::cast_precision_loss)] // At most 512.
        let count = width as f32;
        let mean = warp::reduce_sum_f32(sum) / count;
        let mut squares = 0.0f32;
        #[unroll]
        for j in 0..16 {
            if lane + 32 * j < width {
                let d = values[j] - mean;
                squares += d * d;
            }
        }
        let inverse = 1.0 / libm::sqrtf(warp::reduce_sum_f32(squares) / count + 1e-5);
        #[unroll]
        for j in 0..16 {
            let i = lane + 32 * j;
            if i < width {
                let value = (values[j] - mean) * inverse * gamma[i] + beta[i];
                store_activation((&mut y, &mut y_hi, &mut y_lo), format, row * ldy + i, value);
            }
        }
    }

    /// Store an activation `value` at `at` as the product reading it next takes
    /// it: FP32 in `y` (`format` 0), FP16 in `hi` (1), or FP16 high and scaled low
    /// parts in `hi` and `lo` (2).
    #[inline(always)]
    fn store_activation(
        (y, hi, lo): (
            &mut DisjointSlice<f32>,
            &mut DisjointSlice<u16>,
            &mut DisjointSlice<u16>,
        ),
        format: u32,
        at: usize,
        value: f32,
    ) {
        unsafe {
            if format == 0 {
                *y.get_unchecked_mut(at) = value;
            } else {
                let high = value as f16;
                *hi.get_unchecked_mut(at) = high.to_bits();
                if format == 2 {
                    *lo.get_unchecked_mut(at) = (((value - high as f32) * LO_SCALE) as f16).to_bits();
                }
            }
        }
    }

    /// `x wᵀ` for 64 x 64 output tiles over one split of k, double-buffered in
    /// 16-wide K steps with 16-byte loads; 256 threads with 4 x 4 outputs each.
    /// With one split it applies the epilogue like [`linear`] and writes `y`;
    /// otherwise split `s` (grid z) covers k in `[s * k_chunk, (s + 1) * k_chunk)`
    /// and writes its raw sums to `partial[s, m, n]` for [`linear_reduce`].
    /// Needs `k`, `k_chunk`, and `ldx` to be multiples of 4 (`k_chunk` of 16) and
    /// 16-byte aligned `x` and `w`. Grid: (ceil(n / 64), ceil(m / 64), splits).
    #[kernel]
    pub fn linear64(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        mut partial: DisjointSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
        ldx: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
        k_chunk: u32,
    ) {
        static mut AS: SharedArray<f32, { 2 * 16 * 64 }, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, { 2 * 16 * 64 }, 16> = SharedArray::UNINIT;
        let a_tile = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        let b_tile = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BS)) };
        let tid = thread::threadIdx_x() as usize;
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let (ldx, ldy, ldr) = (ldx as usize, ldy as usize, ldr as usize);
        let row0 = thread::blockIdx_y() as usize * 64;
        let col0 = thread::blockIdx_x() as usize * 64;
        let split = thread::blockIdx_z() as usize;
        let splits = thread::gridDim_z() as usize;
        let k_begin = split * k_chunk as usize;
        let k_end = if k_begin + (k_chunk as usize) < k {
            k_begin + k_chunk as usize
        } else {
            k
        };
        let (ty, tx) = (tid / 16, tid % 16);
        let (load_row, load_k) = (tid / 4, (tid % 4) * 4);
        let (xp, wp) = (x.as_ptr(), w.as_ptr());
        let fetch = |k0: usize| -> (F32x4, F32x4) {
            let gk = k0 + load_k;
            let (gr, gc) = (row0 + load_row, col0 + load_row);
            let a = if gr < m && gk < k_end {
                unsafe { *xp.add(gr * ldx + gk).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            let bv = if gc < n && gk < k_end {
                unsafe { *wp.add(gc * k + gk).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            (a, bv)
        };
        let store = |buf: usize, (a, bv): (F32x4, F32x4)| {
            #[unroll]
            for i in 0..4 {
                unsafe {
                    *a_tile.add(buf * 1024 + (load_k + i) * 64 + load_row) = a.0[i];
                    *b_tile.add(buf * 1024 + (load_k + i) * 64 + load_row) = bv.0[i];
                }
            }
        };
        let mut acc = [[0.0f32; 4]; 4];
        store(0, fetch(k_begin));
        thread::sync_threads();
        let steps = (k_end - k_begin).div_ceil(16);
        let mut step = 0;
        while step < steps {
            let buf = step % 2;
            let next = if step + 1 < steps {
                fetch(k_begin + (step + 1) * 16)
            } else {
                (F32x4::splat(0.0), F32x4::splat(0.0))
            };
            #[unroll]
            for kk in 0..16 {
                let base = buf * 1024 + kk * 64;
                let a = unsafe { *(a_tile.add(base + ty * 4) as *const F32x4) };
                let bv = unsafe { *(b_tile.add(base + tx * 4) as *const F32x4) };
                #[unroll]
                for i in 0..4 {
                    #[unroll]
                    for j in 0..4 {
                        acc[i][j] += a.0[i] * bv.0[j];
                    }
                }
            }
            if step + 1 < steps {
                store(1 - buf, next);
            }
            thread::sync_threads();
            step += 1;
        }
        #[unroll]
        for i in 0..4 {
            let row = row0 + ty * 4 + i;
            if row < m {
                #[unroll]
                for j in 0..4 {
                    let col = col0 + tx * 4 + j;
                    if col < n {
                        if splits > 1 {
                            unsafe { *partial.get_unchecked_mut((split * m + row) * n + col) = acc[i][j] };
                        } else {
                            let mut value = acc[i][j] + b[col];
                            if mode & 1 != 0 {
                                value = 0.5
                                    * value
                                    * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2));
                            }
                            if mode & 2 != 0 {
                                value += r[row * ldr + col];
                            }
                            unsafe { *y.get_unchecked_mut(row * ldy + col) = value };
                        }
                    }
                }
            }
        }
    }

    /// Sum [`linear64`]'s split partials `[splits, m, n]` and apply the epilogue:
    /// bias, then GELU if `mode` bit 1, then the residual `r` if bit 2.
    #[kernel]
    pub fn linear_reduce(
        partial: &[f32],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        m: u32,
        n: u32,
        splits: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
    ) {
        let i = thread::index_1d().get();
        let (m, n) = (m as usize, n as usize);
        if i >= m * n {
            return;
        }
        let (row, col) = (i / n, i % n);
        let mut value = b[col];
        let mut s = 0;
        while s < splits as usize {
            value += partial[s * m * n + i];
            s += 1;
        }
        if mode & 1 != 0 {
            value = 0.5 * value * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2));
        }
        if mode & 2 != 0 {
            value += r[row * ldr as usize + col];
        }
        unsafe { *y.get_unchecked_mut(row * ldy as usize + col) = value };
    }

    /// The column stage's input layer from raw query rows: for ensemble `e`,
    /// feature `f`, and query row `i` of rows `x[e, start + i, :]` (`[B, total, H]`),
    /// groups the features `f + 1`, `f + 2`, and `f + 4` (mod H) and writes
    /// `w · group + b` as `out[(e * H + f) * rows + i, 0..128]`. Sets
    /// `missing[start + i]` to 1 when a group of the row is the all-missing marker
    /// (-100), which only the CPU column stage handles. One thread per output value.
    #[kernel]
    pub fn column_input(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32>,
        mut missing: DisjointSlice<f32>,
        ensembles: u32,
        features: u32,
        total: u32,
        start: u32,
        rows: u32,
    ) {
        let i = thread::index_1d().get();
        let (h, rows) = (features as usize, rows as usize);
        if i >= ensembles as usize * h * rows * 128 {
            return;
        }
        let (c, rest) = (i % 128, i / 128);
        let (row, column) = (rest % rows, rest / rows);
        let (e, f) = (column / h, column % h);
        let base = (e * total as usize + start as usize + row) * h;
        let group = [
            x[base + (f + 1) % h],
            x[base + (f + 2) % h],
            x[base + (f + 4) % h],
        ];
        #[allow(clippy::float_cmp)] // The all-missing marker is exactly -100.
        let marked = group[0] == -100.0 && group[1] == -100.0 && group[2] == -100.0;
        if c == 0 && marked {
            unsafe { *missing.get_unchecked_mut(start as usize + row) = 1.0 };
        }
        let value = group[0] * w[c * 3] + group[1] * w[c * 3 + 1] + group[2] * w[c * 3 + 2] + b[c];
        unsafe { *out.get_unchecked_mut(i) = value };
    }

    /// Row-stage sequences `[B * Q, 4 + H, 128]`: the four CLS tokens, then each
    /// feature's column embedding of rows `start..start + Q` of `[B * H, total, 128]`.
    #[kernel]
    pub fn build_rows(
        columns: &[f32],
        cls: &[f32],
        mut rows: DisjointSlice<f32>,
        ensembles: u32,
        features: u32,
        queries: u32,
        total: u32,
        start: u32,
    ) {
        let i = thread::index_1d().get();
        let (h, q) = (features as usize, queries as usize);
        let len = h + 4;
        if i >= ensembles as usize * q * len * 128 {
            return;
        }
        let (c, token, seq) = (i % 128, (i / 128) % len, i / (128 * len));
        let value = if token < 4 {
            cls[token * 128 + c]
        } else {
            let (b, query) = (seq / q, seq % q);
            columns[((b * h + token - 4) * total as usize + start as usize + query) * 128 + c]
        };
        unsafe { *rows.get_unchecked_mut(i) = value };
    }

    /// The CLS tokens of each `[len, 128]` sequence, as `[seqs * 4, 128]`.
    #[kernel]
    pub fn gather_cls(rows: &[f32], mut cls: DisjointSlice<f32>, seqs: u32, len: u32) {
        let i = thread::index_1d().get();
        if i >= seqs as usize * 4 * 128 {
            return;
        }
        let (c, token, seq) = (i % 128, (i / 128) % 4, i / 512);
        unsafe { *cls.get_unchecked_mut(i) = rows[(seq * len as usize + token) * 128 + c] };
    }

    /// `QASSMax` and packing for ICL attention: for token `t = b * Q + q` and head
    /// `j`, `q * base[j] * (1 + tanh(modulation)) * key_scale[b * 8 + j]` at
    /// `[(b * 8 + j) * qp + q, 64]`, in FP32 into `packed` (`format` 0) or in
    /// FP16 into `packed_hi` (1); `key_scale` is the FP8 keys' scale per dim, or
    /// ones.
    #[kernel]
    pub fn qass_pack(
        query: &[f32],
        modulation: &[f32],
        base: &[f32],
        key_scale: &[f32],
        mut packed: DisjointSlice<f32>,
        mut packed_hi: DisjointSlice<u16>,
        ensembles: u32,
        queries: u32,
        qp: u32,
        format: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let gate = 1.0 + modulation[(token * 8 + head) * 64 + d].tanh();
        let value = query[i] * base[head * 64 + d] * gate * key_scale[(b * 8 + head) * 64 + d];
        let at = ((b * 8 + head) * qp as usize + row) * 64 + d;
        unsafe {
            if format == 0 {
                *packed.get_unchecked_mut(at) = value;
            } else {
                *packed_hi.get_unchecked_mut(at) = (value as f16).to_bits();
            }
        }
    }

    /// Attention outputs `[(b * 8 + j) * qp + q, 64]` back to tokens `[B * Q, 512]`,
    /// times `value_scale[b * 8 + j]`, the FP8 values' scale per dim, or ones;
    /// written as [`store_activation`] does with `format`.
    #[kernel]
    pub fn heads_to_tokens(
        attended: &[f32],
        value_scale: &[f32],
        mut tokens: DisjointSlice<f32>,
        mut tokens_hi: DisjointSlice<u16>,
        mut tokens_lo: DisjointSlice<u16>,
        ensembles: u32,
        queries: u32,
        qp: u32,
        format: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let pair = b * 8 + head;
        let value = attended[(pair * qp as usize + row) * 64 + d] * value_scale[pair * 64 + d];
        store_activation((&mut tokens, &mut tokens_hi, &mut tokens_lo), format, i, value);
    }

    // ---------------------------------------------------------------------------
    // Building a context: the training rows' pass, which keeps keys and values.
    // ---------------------------------------------------------------------------

    /// The column stage's input layer for training columns `first..first +
    /// columns` of `[B * H]`, over all `rows` training rows `x[B, rows, H]`,
    /// plus each estimator's label embeddings `labels[B, rows, 128]`, into
    /// `out[columns, rows, 128]`. The caller has checked for the all-missing
    /// marker. One thread per output value.
    #[kernel]
    pub fn column_train(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        labels: &[f32],
        mut out: DisjointSlice<f32>,
        features: u32,
        rows: u32,
        first: u32,
        columns: u32,
    ) {
        let i = thread::index_1d().get();
        let (h, rows) = (features as usize, rows as usize);
        if i >= columns as usize * rows * 128 {
            return;
        }
        let (c, rest) = (i % 128, i / 128);
        let (row, column) = (rest % rows, first as usize + rest / rows);
        let (e, f) = (column / h, column % h);
        let base = (e * rows + row) * h;
        let group = [
            x[base + (f + 1) % h],
            x[base + (f + 2) % h],
            x[base + (f + 4) % h],
        ];
        let value = group[0] * w[c * 3] + group[1] * w[c * 3 + 1] + group[2] * w[c * 3 + 2] + b[c];
        unsafe { *out.get_unchecked_mut(i) = value + labels[(e * rows + row) * 128 + c] };
    }

    /// `out[i] = source[i % period]` for `len` values.
    #[kernel]
    pub fn repeat(source: &[f32], mut out: DisjointSlice<f32>, period: u32, len: u32) {
        let i = thread::index_1d().get();
        if i < len as usize {
            unsafe { *out.get_unchecked_mut(i) = source[i % period as usize] };
        }
    }

    /// `out[offset + i] = source[i]` for `len` values.
    #[kernel]
    pub fn copy_into(source: &[f32], mut out: DisjointSlice<f32>, offset: u32, len: u32) {
        let i = thread::index_1d().get();
        if i < len as usize {
            unsafe { *out.get_unchecked_mut(offset as usize + i) = source[i] };
        }
    }

    /// A column block's inducing-point keys and values from the in-projection
    /// `kv[columns * 128, 384]` (queries, keys, values) into `[columns, 8, 128,
    /// 16]`, which [`column_block_body`] reads with one feature and no reserved
    /// streams. One thread per key value.
    #[kernel]
    pub fn column_kv(kv: &[f32], mut keys: DisjointSlice<f32>, mut values: DisjointSlice<f32>, columns: u32) {
        let i = thread::index_1d().get();
        if i >= columns as usize * 128 * 128 {
            return;
        }
        let (d, head, point, column) = (i % 16, (i / 16) % 8, (i / 128) % 128, i / (128 * 128));
        let from = (column * 128 + point) * 384 + head * 16 + d;
        let to = (column * 8 + head) * 128 * 16 + point * 16 + d;
        unsafe {
            *keys.get_unchecked_mut(to) = kv[from + 128];
            *values.get_unchecked_mut(to) = kv[from + 256];
        }
    }

    /// Rows `[B, q, width]` of `x` plus rows `start..start + q` of
    /// `labels[B, total, width]`, in place.
    #[kernel]
    pub fn add_labels(
        mut x: DisjointSlice<f32>,
        labels: &[f32],
        queries: u32,
        total: u32,
        start: u32,
        width: u32,
        len: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let (q, width) = (queries as usize, width as usize);
        let (c, token) = (i % width, i / width);
        let (e, row) = (token / q, token % q);
        let label = labels[(e * total as usize + start as usize + row) * width + c];
        unsafe { *x.get_unchecked_mut(i) += label };
    }

    /// An ICL block's keys and values for training rows `start..start + q`,
    /// from the in-projection `kv[B * q, 1024]` (keys, then values), into the
    /// FP16 caches `[B * 8, np, 64]`. One thread per key value.
    #[kernel]
    pub fn icl_kv(
        kv: &[f32],
        mut keys: DisjointSlice<u16>,
        mut values: DisjointSlice<u16>,
        ensembles: u32,
        queries: u32,
        start: u32,
        np: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let to = ((b * 8 + head) * np as usize + start as usize + row) * 64 + d;
        unsafe {
            *keys.get_unchecked_mut(to) = (kv[token * 1024 + head * 64 + d] as f16).to_bits();
            *values.get_unchecked_mut(to) = (kv[token * 1024 + 512 + head * 64 + d] as f16).to_bits();
        }
    }

    /// [`icl_kv`] into FP32 caches, for `optimize="safe"`.
    #[kernel]
    pub fn icl_kv_f32(
        kv: &[f32],
        mut keys: DisjointSlice<f32>,
        mut values: DisjointSlice<f32>,
        ensembles: u32,
        queries: u32,
        start: u32,
        np: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let to = ((b * 8 + head) * np as usize + start as usize + row) * 64 + d;
        unsafe {
            *keys.get_unchecked_mut(to) = kv[token * 1024 + head * 64 + d];
            *values.get_unchecked_mut(to) = kv[token * 1024 + 512 + head * 64 + d];
        }
    }

    /// `value` as FP8 E4M3 bits, rounded to nearest even and saturated at
    /// ±448, as the host's `cache::e4m3` does.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Subnormal steps: 0..=8.
    fn e4m3(value: f32) -> u32 {
        let sign = (value.to_bits() >> 24) & 0x80;
        let magnitude = value.abs().min(448.0);
        if magnitude < 1.0 / 64.0 {
            // Subnormals are multiples of 2^-9: adding and taking away 2^23
            // rounds them to the nearest even integer.
            let steps = (magnitude * 512.0 + 8_388_608.0) - 8_388_608.0;
            return sign | steps as u32;
        }
        let bits = magnitude.to_bits();
        let mut exponent = ((bits >> 23) & 0xff).cast_signed() - 127;
        let (mut mantissa, rest) = ((bits >> 20) & 7, bits & 0xf_ffff);
        if rest > 0x8_0000 || (rest == 0x8_0000 && mantissa & 1 == 1) {
            mantissa += 1;
        }
        if mantissa == 8 {
            mantissa = 0;
            exponent += 1;
        }
        // 448 is exponent 8, mantissa 6; mantissa 7 there is NaN.
        if exponent > 8 || (exponent == 8 && mantissa == 7) {
            return sign | 0x7e;
        }
        sign | ((exponent + 7).cast_unsigned() << 3) | mantissa
    }

    /// The largest magnitude of each (pair, dim) of FP16 `values` `[pairs,
    /// stride, 64]` over a chunk of 256 of their first `n` rows, into
    /// `peaks[(pair * chunks + chunk) * 64 + dim]`. One thread per peak.
    #[kernel]
    pub fn fp8_peaks(
        values: &[u16],
        mut peaks: DisjointSlice<f32>,
        stride: u32,
        n: u32,
        chunks: u32,
        len: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let (dim, rest) = (i % 64, i / 64);
        let (chunk, pair) = (rest % chunks as usize, rest / chunks as usize);
        let start = chunk * 256;
        let end = if start + 256 < n as usize {
            start + 256
        } else {
            n as usize
        };
        let mut peak = 0.0f32;
        for row in start..end {
            let value = f16::from_bits(values[(pair * stride as usize + row) * 64 + dim]) as f32;
            peak = peak.max(value.abs());
        }
        unsafe { *peaks.get_unchecked_mut(i) = peak };
    }

    /// Each (pair, dim)'s FP8 scale from its chunks' peaks: the scale that
    /// maps its largest magnitude to E4M3's largest, 448, or 1 for zeros.
    /// One thread per scale.
    #[kernel]
    pub fn fp8_scales(peaks: &[f32], mut scales: DisjointSlice<f32>, chunks: u32, len: u32) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let (dim, pair, chunks) = (i % 64, i / 64, chunks as usize);
        let mut peak = 0.0f32;
        for chunk in 0..chunks {
            peak = peak.max(peaks[(pair * chunks + chunk) * 64 + dim]);
        }
        unsafe { *scales.get_unchecked_mut(i) = if peak > 0.0 { peak / 448.0 } else { 1.0 } };
    }

    /// The first `n` rows of FP16 `values` `[pairs, stride, 64]` divided by
    /// their scales `[pairs, 64]` as FP8 E4M3 bytes `[pairs, np, 64]`, two to
    /// a `u16`; rows from `n` on are zero. One thread per pair of bytes.
    #[kernel]
    pub fn fp8_encode(
        values: &[u16],
        scales: &[f32],
        mut packed: DisjointSlice<u16>,
        stride: u32,
        np: u32,
        n: u32,
        len: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let np = np as usize;
        let (dim, row, pair) = ((i % 32) * 2, (i / 32) % np, i / (32 * np));
        let mut bytes = 0u32;
        if row < n as usize {
            let at = (pair * stride as usize + row) * 64 + dim;
            let low = f16::from_bits(values[at]) as f32 / scales[pair * 64 + dim];
            let high = f16::from_bits(values[at + 1]) as f32 / scales[pair * 64 + dim + 1];
            bytes = e4m3(low) | (e4m3(high) << 8);
        }
        #[allow(clippy::cast_possible_truncation)] // Two bytes.
        let bytes = bytes as u16;
        unsafe { *packed.get_unchecked_mut(i) = bytes };
    }

    // ---------------------------------------------------------------------------
    // Dense layers on tensor cores.
    // ---------------------------------------------------------------------------

    /// Scale of the low FP16 part of a split value: `v ≈ hi + lo / LO_SCALE`.
    const LO_SCALE: f32 = 2048.0;

    /// u32 index of 16-byte chunk `chunk` (0..4) of row `row` in a 32-wide FP16
    /// tile, swizzled so that `ldmatrix` reads of 8 rows at one chunk hit 8
    /// different bank groups.
    #[inline(always)]
    fn swizzled32(row: usize, chunk: usize) -> usize {
        row * 16 + ((chunk ^ ((row >> 1) & 3)) * 4)
    }

    /// Two values as packed FP16, the first in the low half.
    #[inline(always)]
    fn pack_f16(a: f32, b: f32) -> u32 {
        u32::from((a as f16).to_bits()) | (u32::from((b as f16).to_bits()) << 16)
    }

    /// Low parts of two values: `(v - f16(v)) * LO_SCALE` as packed FP16.
    #[inline(always)]
    fn pack_lo(a: f32, b: f32) -> u32 {
        let (ha, hb) = ((a as f16) as f32, (b as f16) as f32);
        pack_f16((a - ha) * LO_SCALE, (b - hb) * LO_SCALE)
    }

    /// The split-precision correction `a_hi b_lo + a_lo b_hi` (low parts scaled by
    /// [`LO_SCALE`]) for one 16-wide K step, added to `corr`. It accumulates in
    /// FP16, which runs at twice the FP32-accumulator rate: one step's sum is
    /// far from FP16's range, and its rounding error is about 2^-11 of a term
    /// that is itself divided by `LO_SCALE`.
    #[inline(always)]
    fn add_correction(
        corr: [f32; 4],
        a_hi: [u32; 4],
        a_lo: [u32; 4],
        b_hi: [u32; 2],
        b_lo: [u32; 2],
    ) -> [f32; 4] {
        let mut c = unsafe { mma_m16n8k16_f16_f16([0; 2], a_hi, b_lo) };
        c = unsafe { mma_m16n8k16_f16_f16(c, a_lo, b_hi) };
        let (c0, c1) = cvt_f32x2_f16x2(c[0]);
        let (c2, c3) = cvt_f32x2_f16x2(c[1]);
        [corr[0] + c0, corr[1] + c1, corr[2] + c2, corr[3] + c3]
    }

    // ---------------------------------------------------------------------------
    // Fused 128-wide transformer block tail.
    // ---------------------------------------------------------------------------

    /// u32 index of the FP16 pair at (`row`, even `col`) in a tile with `stride`
    /// u32 per row, 16-byte chunks XOR-swizzled by the row's low three bits.
    #[inline(always)]
    fn tile_at(stride: usize, row: usize, col: usize) -> usize {
        row * stride + (((col / 8) ^ (row & 7)) * 4) + (col % 8) / 2
    }

    /// Store an (even, odd) column pair as FP16, and its low parts with `SPLIT`.
    #[inline(always)]
    fn store_pair<const SPLIT: bool>(hi: *mut u32, lo: *mut u32, at: usize, a: f32, b: f32) {
        unsafe {
            *hi.add(at) = pack_f16(a, b);
            if SPLIT {
                *lo.add(at) = pack_lo(a, b);
            }
        }
    }

    /// One warp's `[16, NT * 8]` product of a shared FP16 tile (rows from `row0`,
    /// `stride` u32 per row, `ks` 16-wide K steps) with a weight in fragment order
    /// (`[n / 8, k / 16, 32 lanes, 2]` u32), n-tiles from `nt0`.
    #[device]
    fn warp_product<const SPLIT: bool, const NT: usize>(
        a_hi: *mut u32,
        a_lo: *mut u32,
        stride: usize,
        row0: usize,
        ks: usize,
        w_hi: &[u32],
        w_lo: &[u32],
        nt0: usize,
        lane: usize,
        acc: &mut [[f32; 4]; NT],
        corr: &mut [[f32; 4]; NT],
    ) {
        let row = row0 + lane % 16;
        let mut step = 0;
        while step < ks {
            let at = row * stride + (((step * 2 + lane / 16) ^ (row & 7)) * 4);
            let fa = unsafe { ldmatrix_x4(a_hi.add(at)) };
            let fal = if SPLIT {
                unsafe { ldmatrix_x4(a_lo.add(at)) }
            } else {
                [0u32; 4]
            };
            #[unroll]
            for nt in 0..NT {
                let index = (((nt0 + nt) * ks + step) * 32 + lane) * 2;
                let fb = unsafe { [*w_hi.get_unchecked(index), *w_hi.get_unchecked(index + 1)] };
                acc[nt] = unsafe { mma_m16n8k16_f32_f16(acc[nt], fa, fb) };
                if SPLIT {
                    let fbl = unsafe { [*w_lo.get_unchecked(index), *w_lo.get_unchecked(index + 1)] };
                    corr[nt] = add_correction(corr[nt], fa, fal, fb, fbl);
                }
            }
            step += 1;
        }
    }

    #[inline(always)]
    fn gelu(value: f32) -> f32 {
        0.5 * value * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2))
    }

    /// The rest of a pre-norm block with width 128, for 32 rows per block:
    /// `mid = state + attended Woᵀ + bo`, then
    /// `state = mid + gelu(norm2(mid) W1ᵀ + b1) W2ᵀ + b2`, in place. Weights are
    /// FP16 in fragment order (see [`warp_product`]), with low parts for `SPLIT`.
    /// 8 warps: two 16-row halves by four column quarters.
    /// Grid: ceil(m / 32); block: 256 threads.
    #[device]
    fn block_tail_body<const SPLIT: bool>(
        attended: &[f32],
        state: &mut DisjointSlice<f32>,
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        m: u32,
    ) {
        // A: 32 x 128 FP16 (64 u32 per row); H: 32 x 256 FP16 (128 u32 per row).
        static mut AH: SharedArray<u32, { 32 * 64 }, 128> = SharedArray::UNINIT;
        static mut AL: SharedArray<u32, { 32 * 64 }, 128> = SharedArray::UNINIT;
        static mut HH: SharedArray<u32, { 32 * 128 }, 128> = SharedArray::UNINIT;
        static mut HL: SharedArray<u32, { 32 * 128 }, 128> = SharedArray::UNINIT;
        let ah = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)) };
        let al = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AL)) };
        let hh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HH)) };
        let hl = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HL)) };
        let tid = thread::threadIdx_x() as usize;
        let m = m as usize;
        let row0 = thread::blockIdx_x() as usize * 32;

        // The attention output tile into A.
        let ap = attended.as_ptr();
        #[unroll]
        for i in 0..4 {
            let index = tid + i * 256;
            let (row, col) = (index / 32, (index % 32) * 4);
            let v = if row0 + row < m {
                unsafe { *ap.add((row0 + row) * 128 + col).cast::<F32x4>() }.0
            } else {
                [0.0; 4]
            };
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col), v[0], v[1]);
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col + 2), v[2], v[3]);
        }
        thread::sync_threads();
        block_tail_core::<SPLIT>(
            ah, al, hh, hl, state, row0, m, wo_hi, wo_lo, bo, gamma, beta, w1_hi, w1_lo, b1, w2_hi, w2_lo, b2,
        );
    }

    /// [`block_tail_body`] once the attention output tile of rows
    /// `[row0, row0 + 32)` is in A (FP16 parts, 64 u32 per row); rows from `m`
    /// on are not written. H (128 u32 per row) is scratch.
    #[device]
    fn block_tail_core<const SPLIT: bool>(
        ah: *mut u32,
        al: *mut u32,
        hh: *mut u32,
        hl: *mut u32,
        state: &mut DisjointSlice<f32>,
        row0: usize,
        m: usize,
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
    ) {
        let tid = thread::threadIdx_x() as usize;
        let (warp_id, lane) = (tid / 32, tid % 32);
        let (wm, wn) = (warp_id / 4, warp_id % 4);
        let (lr, lc) = (lane / 4, (lane % 4) * 2);

        // mid = state + attended Woᵀ + bo, kept in registers in fragment layout.
        let mut acc = [[0.0f32; 4]; 4];
        let mut corr = [[0.0f32; 4]; 4];
        warp_product::<SPLIT, 4>(
            ah,
            al,
            64,
            wm * 16,
            8,
            wo_hi,
            wo_lo,
            wn * 4,
            lane,
            &mut acc,
            &mut corr,
        );
        let mut mid = [[0.0f32; 4]; 4];
        let mut sums = [0.0f32; 2];
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for e in 0..4 {
                let row = row0 + wm * 16 + lr + (e / 2) * 8;
                let col = wn * 32 + nt * 8 + lc + e % 2;
                if row < m {
                    let mut value =
                        acc[nt][e] + bo[col] + unsafe { *state.get_unchecked_mut(row * 128 + col) };
                    if SPLIT {
                        value += corr[nt][e] * (1.0 / LO_SCALE);
                    }
                    mid[nt][e] = value;
                }
                sums[e / 2] += mid[nt][e];
            }
        }

        // norm2 over each row's 128 values: quads, then the four column warps.
        let stats = hh.cast::<f32>(); // [2][32 rows][4 quarters], before H is used
        #[unroll]
        for half in 0..2 {
            sums[half] += warp::shuffle_xor_f32(sums[half], 1);
            sums[half] += warp::shuffle_xor_f32(sums[half], 2);
        }
        if lane % 4 == 0 {
            unsafe {
                *stats.add((wm * 16 + lr) * 4 + wn) = sums[0];
                *stats.add((wm * 16 + lr + 8) * 4 + wn) = sums[1];
            }
        }
        thread::sync_threads();
        let mut mean = [0.0f32; 2];
        #[unroll]
        for half in 0..2 {
            let row = wm * 16 + lr + half * 8;
            unsafe {
                mean[half] = (*stats.add(row * 4)
                    + *stats.add(row * 4 + 1)
                    + *stats.add(row * 4 + 2)
                    + *stats.add(row * 4 + 3))
                    * (1.0 / 128.0);
            }
        }
        let mut squares = [0.0f32; 2];
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for e in 0..4 {
                let d = mid[nt][e] - mean[e / 2];
                squares[e / 2] += d * d;
            }
        }
        #[unroll]
        for half in 0..2 {
            squares[half] += warp::shuffle_xor_f32(squares[half], 1);
            squares[half] += warp::shuffle_xor_f32(squares[half], 2);
        }
        if lane % 4 == 0 {
            unsafe {
                *stats.add(128 + (wm * 16 + lr) * 4 + wn) = squares[0];
                *stats.add(128 + (wm * 16 + lr + 8) * 4 + wn) = squares[1];
            }
        }
        thread::sync_threads();
        let mut inverse = [0.0f32; 2];
        #[unroll]
        for half in 0..2 {
            let row = 128 + (wm * 16 + lr + half * 8) * 4;
            let variance =
                unsafe { *stats.add(row) + *stats.add(row + 1) + *stats.add(row + 2) + *stats.add(row + 3) }
                    * (1.0 / 128.0);
            inverse[half] = 1.0 / libm::sqrtf(variance + 1e-5);
        }
        // Every warp has finished reading A, so the normalized tile replaces it.
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for half in 0..2 {
                let row = wm * 16 + lr + half * 8;
                let col = wn * 32 + nt * 8 + lc;
                let a = (mid[nt][half * 2] - mean[half]) * inverse[half] * gamma[col] + beta[col];
                let b = (mid[nt][half * 2 + 1] - mean[half]) * inverse[half] * gamma[col + 1] + beta[col + 1];
                store_pair::<SPLIT>(ah, al, tile_at(64, row, col), a, b);
            }
        }
        thread::sync_threads();

        // H = gelu(normed W1ᵀ + b1): each warp 16 rows x 64 columns, in two
        // passes of 32 to keep registers low.
        #[unroll]
        for pass in 0..2 {
            let mut acc2 = [[0.0f32; 4]; 4];
            let mut corr2 = [[0.0f32; 4]; 4];
            warp_product::<SPLIT, 4>(
                ah,
                al,
                64,
                wm * 16,
                8,
                w1_hi,
                w1_lo,
                wn * 8 + pass * 4,
                lane,
                &mut acc2,
                &mut corr2,
            );
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for half in 0..2 {
                    let row = wm * 16 + lr + half * 8;
                    let col = wn * 64 + pass * 32 + nt * 8 + lc;
                    let mut a = acc2[nt][half * 2] + b1[col];
                    let mut b = acc2[nt][half * 2 + 1] + b1[col + 1];
                    if SPLIT {
                        a += corr2[nt][half * 2] * (1.0 / LO_SCALE);
                        b += corr2[nt][half * 2 + 1] * (1.0 / LO_SCALE);
                    }
                    store_pair::<SPLIT>(hh, hl, tile_at(128, row, col), gelu(a), gelu(b));
                }
            }
        }
        thread::sync_threads();

        // state = mid + H W2ᵀ + b2.
        let mut acc3 = [[0.0f32; 4]; 4];
        let mut corr3 = [[0.0f32; 4]; 4];
        warp_product::<SPLIT, 4>(
            hh,
            hl,
            128,
            wm * 16,
            16,
            w2_hi,
            w2_lo,
            wn * 4,
            lane,
            &mut acc3,
            &mut corr3,
        );
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for e in 0..4 {
                let row = row0 + wm * 16 + lr + (e / 2) * 8;
                let col = wn * 32 + nt * 8 + lc + e % 2;
                if row < m {
                    let mut value = acc3[nt][e] + b2[col] + mid[nt][e];
                    if SPLIT {
                        value += corr3[nt][e] * (1.0 / LO_SCALE);
                    }
                    unsafe { *state.get_unchecked_mut(row * 128 + col) = value };
                }
            }
        }
    }

    /// [`block_tail_body`] with one FP16 product per layer.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_tail_half(
        attended: &[f32],
        mut state: DisjointSlice<f32>,
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        m: u32,
    ) {
        block_tail_body::<false>(
            attended, &mut state, wo_hi, wo_lo, bo, gamma, beta, w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, m,
        );
    }

    /// [`block_tail_body`] with split, near-FP32 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_tail_split(
        attended: &[f32],
        mut state: DisjointSlice<f32>,
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        m: u32,
    ) {
        block_tail_body::<true>(
            attended, &mut state, wo_hi, wo_lo, bo, gamma, beta, w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, m,
        );
    }

    /// Rows `[row0, row0 + 32)` of `x[.., 128]` through `LayerNorm` (epsilon 1e-5)
    /// into A as FP16 parts (64 u32 per row); rows from `m` on become zeros. Each
    /// of 8 warps normalizes four rows, four values per lane.
    #[device]
    fn load_normed<const SPLIT: bool>(
        xp: *const f32,
        gamma: &[f32],
        beta: &[f32],
        ah: *mut u32,
        al: *mut u32,
        row0: usize,
        m: usize,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let (warp_id, lane) = (tid / 32, tid % 32);
        #[unroll]
        for i in 0..4 {
            let row = warp_id * 4 + i;
            let col = lane * 4;
            let v = if row0 + row < m {
                unsafe { *xp.add((row0 + row) * 128 + col).cast::<F32x4>() }.0
            } else {
                [0.0; 4]
            };
            let mean = warp::reduce_sum_f32(v[0] + v[1] + v[2] + v[3]) * (1.0 / 128.0);
            let d = [v[0] - mean, v[1] - mean, v[2] - mean, v[3] - mean];
            let variance =
                warp::reduce_sum_f32(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]) * (1.0 / 128.0);
            let inverse = 1.0 / libm::sqrtf(variance + 1e-5);
            let y = [
                d[0] * inverse * gamma[col] + beta[col],
                d[1] * inverse * gamma[col + 1] + beta[col + 1],
                d[2] * inverse * gamma[col + 2] + beta[col + 2],
                d[3] * inverse * gamma[col + 3] + beta[col + 3],
            ];
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col), y[0], y[1]);
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col + 2), y[2], y[3]);
        }
    }

    /// The start of a pre-norm block with width 128, for 32 rows per block:
    /// `out = norm1(x) Wᵀ + b` for `n` (128 or 384) outputs, then, when `len` is
    /// not zero, `RoPE` on the first 256 outputs (queries and keys) by each row's
    /// position `row % len`: for dims `d < 8` of each 16-wide head,
    /// `x[d] = x[d] cos[t][d] + x[d + 8] sin[t][d]` and
    /// `x[d + 8] = x[d + 8] cos[t][d + 8] + x[d] sin[t][d + 8]`, with the tables
    /// holding `[-sin, sin]`. Weights as in [`block_tail_body`].
    /// Grid: ceil(m / 32); block: 256 threads.
    #[device]
    fn block_head_body<const SPLIT: bool>(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w_hi: &[u32],
        w_lo: &[u32],
        bias: &[f32],
        cos: &[f32],
        sin: &[f32],
        out: &mut DisjointSlice<f32>,
        m: u32,
        n: u32,
        len: u32,
    ) {
        static mut AH: SharedArray<u32, { 32 * 64 }, 128> = SharedArray::UNINIT;
        static mut AL: SharedArray<u32, { 32 * 64 }, 128> = SharedArray::UNINIT;
        let ah = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)) };
        let al = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AL)) };
        let tid = thread::threadIdx_x() as usize;
        let (warp_id, lane) = (tid / 32, tid % 32);
        let (m, n, len) = (m as usize, n as usize, len as usize);
        let row0 = thread::blockIdx_x() as usize * 32;

        load_normed::<SPLIT>(x.as_ptr(), gamma, beta, ah, al, row0, m);
        thread::sync_threads();

        // Each warp: 16 rows x n / 4 columns, 32 at a time (two RoPE heads).
        let (wm, wn) = (warp_id / 4, warp_id % 4);
        let (lr, lc) = (lane / 4, (lane % 4) * 2);
        let quarter = n / 4;
        let mut pass = 0;
        while pass < quarter / 32 {
            let col0 = wn * quarter + pass * 32;
            let mut acc = [[0.0f32; 4]; 4];
            let mut corr = [[0.0f32; 4]; 4];
            warp_product::<SPLIT, 4>(
                ah,
                al,
                64,
                wm * 16,
                8,
                w_hi,
                w_lo,
                col0 / 8,
                lane,
                &mut acc,
                &mut corr,
            );
            let mut value = [[0.0f32; 4]; 4];
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for e in 0..4 {
                    let col = col0 + nt * 8 + lc + e % 2;
                    value[nt][e] = acc[nt][e] + bias[col];
                    if SPLIT {
                        value[nt][e] += corr[nt][e] * (1.0 / LO_SCALE);
                    }
                }
            }
            if len != 0 && col0 < 256 {
                // n-tiles 2h and 2h + 1 hold dims d and d + 8 of head h.
                #[unroll]
                for head in 0..2 {
                    #[unroll]
                    for e in 0..4 {
                        let t = (row0 + wm * 16 + lr + (e / 2) * 8) % len;
                        let d = lc + e % 2;
                        let (a, b) = (value[2 * head][e], value[2 * head + 1][e]);
                        value[2 * head][e] = a * cos[t * 16 + d] + b * sin[t * 16 + d];
                        value[2 * head + 1][e] = b * cos[t * 16 + d + 8] + a * sin[t * 16 + d + 8];
                    }
                }
            }
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for e in 0..4 {
                    let row = row0 + wm * 16 + lr + (e / 2) * 8;
                    let col = col0 + nt * 8 + lc + e % 2;
                    if row < m {
                        unsafe { *out.get_unchecked_mut(row * n + col) = value[nt][e] };
                    }
                }
            }
            pass += 1;
        }
    }

    /// [`block_head_body`] with one FP16 product.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_head_half(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w_hi: &[u32],
        w_lo: &[u32],
        bias: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut out: DisjointSlice<f32>,
        m: u32,
        n: u32,
        len: u32,
    ) {
        block_head_body::<false>(x, gamma, beta, w_hi, w_lo, bias, cos, sin, &mut out, m, n, len);
    }

    /// [`block_head_body`] with split, near-FP32 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_head_split(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w_hi: &[u32],
        w_lo: &[u32],
        bias: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut out: DisjointSlice<f32>,
        m: u32,
        n: u32,
        len: u32,
    ) {
        block_head_body::<true>(x, gamma, beta, w_hi, w_lo, bias, cos, sin, &mut out, m, n, len);
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
    fn attention16_tc_body<const SPLIT: bool>(
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
        let kh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KH)) };
        let kl = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KL)) };
        let vh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VH)) };
        let vl = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VL)) };
        let tid = thread::threadIdx_x() as usize;
        let (warp_id, lane) = (tid / 32, tid % 32);
        let (lr, lc) = (lane / 4, (lane % 4) * 2);
        let query0 = thread::blockIdx_x() as usize * 64 + warp_id * 16;
        let head = thread::blockIdx_y() as usize;
        let seq = thread::blockIdx_z() as usize;
        let (lq, lk, inner) = (lq as usize, lk as usize, inner as usize);
        let kv0 =
            (seq / inner) * kv_outer as usize + (seq % inner) * kv_inner as usize + head * kv_head as usize;
        let kv_tok = kv_tok as usize;
        let (qp, kp, vp) = (q.as_ptr(), &k[k_base as usize..], &v[v_base as usize..]);
        let (kp, vp) = (kp.as_ptr(), vp.as_ptr());

        // Q as the A fragment: rows lr and lr + 8, dims lc, lc + 1, lc + 8, lc + 9.
        let q_at = |row: usize, dim: usize| -> (f32, f32) {
            let query = query0 + row;
            if query < lq {
                let at = seq * q_seq as usize + query * q_tok as usize + head * 16 + dim;
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
                    unsafe { (*kp.add(at), *vp.add(at)) }
                } else {
                    (0.0, 0.0)
                };
                let (k1, v1) = if key + 1 < lk {
                    unsafe { (*kp.add(at + kv_tok), *vp.add(at + kv_tok)) }
                } else {
                    (0.0, 0.0)
                };
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
                        let kb = unsafe { [*kh.add(at), *kh.add(at + 4)] };
                        let mut c = [0.0f32; 4];
                        c = unsafe { mma_m16n8k16_f32_f16(c, qa, kb) };
                        if SPLIT {
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
                            let vb = unsafe { [*vh.add(at), *vh.add(at + 4)] };
                            acc[nt] = unsafe { mma_m16n8k16_f32_f16(acc[nt], pa, vb) };
                            if SPLIT {
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
                    unsafe { *out.get_unchecked_mut(at) = value / total };
                }
            }
        }
    }

    /// [`attention16_tc_body`] with FP16 products.
    #[kernel]
    pub fn attention16_half(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
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
        attention16_tc_body::<false>(
            q, k, v, &mut out, lq, lk, inner, q_seq, q_tok, k_base, v_base, kv_outer, kv_inner, kv_tok,
            kv_head, o_seq, o_tok, scale,
        );
    }

    /// [`attention16_tc_body`] with split, near-FP32 products.
    #[kernel]
    pub fn attention16_split(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
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
        attention16_tc_body::<true>(
            q, k, v, &mut out, lq, lk, inner, q_seq, q_tok, k_base, v_base, kv_outer, kv_inner, kv_tok,
            kv_head, o_seq, o_tok, scale,
        );
    }

    /// A whole column block for 32 query rows of one column (estimator `e`,
    /// feature `f`): `norm1`, the query projection (the first 128 rows of the
    /// fused in-projection, in fragment order), attention of each 16-wide head to
    /// the column's 128 cached inducing points (cache stream `f + reserved` of
    /// `e`, `[B, H + reserved, 8, 128, 16]` FP32: 4 reserved streams for a
    /// context's cache, none for a build's), then [`block_tail_core`], in place on
    /// `state` `[B * H, q, 128]`: each row is read before its block writes it. Weights and precision as in [`block_tail_body`].
    /// Grid: (ceil(q / 32), B * H); block: 256 threads.
    #[device]
    fn column_block_body<const SPLIT: bool>(
        state: &mut DisjointSlice<f32>,
        gamma1: &[f32],
        beta1: &[f32],
        wq_hi: &[u32],
        wq_lo: &[u32],
        bq: &[f32],
        keys: &[f32],
        values: &[f32],
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma2: &[f32],
        beta2: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        features: u32,
        queries: u32,
        reserved: u32,
    ) {
        static mut AH: SharedArray<u32, { 32 * 64 }, 128> = SharedArray::UNINIT;
        static mut AL: SharedArray<u32, { 32 * 64 }, 128> = SharedArray::UNINIT;
        static mut HH: SharedArray<u32, { 32 * 128 }, 128> = SharedArray::UNINIT;
        static mut HL: SharedArray<u32, { 32 * 128 }, 128> = SharedArray::UNINIT;
        let ah = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)) };
        let hh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HH)) };
        // FP16 never touches the low parts, so it leaves their buffers unused.
        let (al, hl) = if SPLIT {
            unsafe {
                (
                    SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AL)),
                    SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HL)),
                )
            }
        } else {
            (ah, hh)
        };
        let tid = thread::threadIdx_x() as usize;
        let (warp_id, lane) = (tid / 32, tid % 32);
        let (wm, wn) = (warp_id / 4, warp_id % 4);
        let (lr, lc) = (lane / 4, (lane % 4) * 2);
        let (h, q) = (features as usize, queries as usize);
        let column = thread::blockIdx_y() as usize;
        let tile = thread::blockIdx_x() as usize * 32;
        let row0 = column * q + tile;
        let m = row0 + if q - tile < 32 { q - tile } else { 32 };

        // norm1 into A, then Q = A Wqᵀ + bq into the first half of H.
        load_normed::<SPLIT>(state.as_mut_ptr(), gamma1, beta1, ah, al, row0, m);
        thread::sync_threads();
        let mut acc = [[0.0f32; 4]; 4];
        let mut corr = [[0.0f32; 4]; 4];
        warp_product::<SPLIT, 4>(
            ah,
            al,
            64,
            wm * 16,
            8,
            wq_hi,
            wq_lo,
            wn * 4,
            lane,
            &mut acc,
            &mut corr,
        );
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for half in 0..2 {
                let row = wm * 16 + lr + half * 8;
                let col = wn * 32 + nt * 8 + lc;
                let mut a = acc[nt][half * 2] + bq[col];
                let mut b = acc[nt][half * 2 + 1] + bq[col + 1];
                if SPLIT {
                    a += corr[nt][half * 2] * (1.0 / LO_SCALE);
                    b += corr[nt][half * 2 + 1] * (1.0 / LO_SCALE);
                }
                store_pair::<SPLIT>(hh, hl, tile_at(64, row, col), a, b);
            }
        }
        thread::sync_threads();

        // Attention: warp j is head j, for both 16-row halves of the tile.
        let head = warp_id;
        let (e, f) = (column / h, column % h);
        let reserved = reserved as usize;
        let kv0 = (e * (h + reserved) + f + reserved) * (8 * 128 * 16) + head * 128 * 16;
        let (kp, vp) = (keys.as_ptr(), values.as_ptr());
        let mut qa = [[0u32; 4]; 2];
        let mut qal = [[0u32; 4]; 2];
        #[unroll]
        for mt in 0..2 {
            let row = mt * 16 + lane % 16;
            let at = row * 64 + (((head * 2 + lane / 16) ^ (row & 7)) * 4);
            qa[mt] = unsafe { ldmatrix_x4(hh.add(at)) };
            if SPLIT {
                qal[mt] = unsafe { ldmatrix_x4(hl.add(at)) };
            }
        }
        let scale_log2 = 0.25 * core::f32::consts::LOG2_E; // 1 / sqrt(16)
        let mut out = [[[0.0f32; 4]; 2]; 2];
        let mut out_corr = [[[0.0f32; 4]; 2]; 2];
        let mut row_max = [[f32::NEG_INFINITY; 2]; 2];
        let mut row_sum = [[0.0f32; 2]; 2];
        let mut key0 = 0;
        while key0 < 128 {
            // Scores for 32 keys: four 8-key n-tiles, K shared by both halves.
            let mut s = [[[0.0f32; 4]; 4]; 2];
            #[unroll]
            for nt in 0..4 {
                let at = kv0 + (key0 + nt * 8 + lr) * 16 + lc;
                let (k0, k1, k8, k9) =
                    unsafe { (*kp.add(at), *kp.add(at + 1), *kp.add(at + 8), *kp.add(at + 9)) };
                let kb = [pack_f16(k0, k1), pack_f16(k8, k9)];
                let kbl = if SPLIT {
                    [pack_lo(k0, k1), pack_lo(k8, k9)]
                } else {
                    [0u32; 2]
                };
                #[unroll]
                for mt in 0..2 {
                    let mut c = unsafe { mma_m16n8k16_f32_f16([0.0; 4], qa[mt], kb) };
                    if SPLIT {
                        let cl = add_correction([0.0; 4], qa[mt], qal[mt], kb, kbl);
                        #[unroll]
                        for i in 0..4 {
                            c[i] += cl[i] * (1.0 / LO_SCALE);
                        }
                    }
                    #[unroll]
                    for i in 0..4 {
                        s[mt][nt][i] = c[i] * scale_log2;
                    }
                }
            }
            // Online softmax per half: rows lr and lr + 8.
            #[unroll]
            for mt in 0..2 {
                #[unroll]
                for half in 0..2 {
                    let mut max = f32::NEG_INFINITY;
                    #[unroll]
                    for nt in 0..4 {
                        max = max.max(s[mt][nt][half * 2]).max(s[mt][nt][half * 2 + 1]);
                    }
                    max = max.max(warp::shuffle_xor_f32(max, 1));
                    max = max.max(warp::shuffle_xor_f32(max, 2));
                    let new = row_max[mt][half].max(max);
                    let alpha = (row_max[mt][half] - new).exp2();
                    let mut sum = 0.0f32;
                    #[unroll]
                    for nt in 0..4 {
                        s[mt][nt][half * 2] = (s[mt][nt][half * 2] - new).exp2();
                        s[mt][nt][half * 2 + 1] = (s[mt][nt][half * 2 + 1] - new).exp2();
                        sum += s[mt][nt][half * 2] + s[mt][nt][half * 2 + 1];
                    }
                    sum += warp::shuffle_xor_f32(sum, 1);
                    sum += warp::shuffle_xor_f32(sum, 2);
                    row_sum[mt][half] = row_sum[mt][half] * alpha + sum;
                    row_max[mt][half] = new;
                    #[unroll]
                    for nt in 0..2 {
                        out[mt][nt][half * 2] *= alpha;
                        out[mt][nt][half * 2 + 1] *= alpha;
                        if SPLIT {
                            out_corr[mt][nt][half * 2] *= alpha;
                            out_corr[mt][nt][half * 2 + 1] *= alpha;
                        }
                    }
                }
            }
            // O += P V over two 16-key steps, V shared by both halves.
            #[unroll]
            for kk in 0..2 {
                #[unroll]
                for nt in 0..2 {
                    let at = kv0 + (key0 + kk * 16 + lc) * 16 + nt * 8 + lr;
                    let (v0, v1, v8, v9) = unsafe {
                        (
                            *vp.add(at),
                            *vp.add(at + 16),
                            *vp.add(at + 128),
                            *vp.add(at + 144),
                        )
                    };
                    let vb = [pack_f16(v0, v1), pack_f16(v8, v9)];
                    let vbl = if SPLIT {
                        [pack_lo(v0, v1), pack_lo(v8, v9)]
                    } else {
                        [0u32; 2]
                    };
                    #[unroll]
                    for mt in 0..2 {
                        let (s0, s1) = (s[mt][2 * kk], s[mt][2 * kk + 1]);
                        let pa = [
                            pack_f16(s0[0], s0[1]),
                            pack_f16(s0[2], s0[3]),
                            pack_f16(s1[0], s1[1]),
                            pack_f16(s1[2], s1[3]),
                        ];
                        out[mt][nt] = unsafe { mma_m16n8k16_f32_f16(out[mt][nt], pa, vb) };
                        if SPLIT {
                            let pal = [
                                pack_lo(s0[0], s0[1]),
                                pack_lo(s0[2], s0[3]),
                                pack_lo(s1[0], s1[1]),
                                pack_lo(s1[2], s1[3]),
                            ];
                            out_corr[mt][nt] = add_correction(out_corr[mt][nt], pa, pal, vb, vbl);
                        }
                    }
                }
            }
            key0 += 32;
        }
        // The attention output replaces the normalized rows in A.
        #[unroll]
        for mt in 0..2 {
            #[unroll]
            for nt in 0..2 {
                #[unroll]
                for half in 0..2 {
                    let row = mt * 16 + lr + half * 8;
                    let col = head * 16 + nt * 8 + lc;
                    let mut a = out[mt][nt][half * 2];
                    let mut b = out[mt][nt][half * 2 + 1];
                    if SPLIT {
                        a += out_corr[mt][nt][half * 2] * (1.0 / LO_SCALE);
                        b += out_corr[mt][nt][half * 2 + 1] * (1.0 / LO_SCALE);
                    }
                    let total = row_sum[mt][half];
                    store_pair::<SPLIT>(ah, al, tile_at(64, row, col), a / total, b / total);
                }
            }
        }
        thread::sync_threads();
        block_tail_core::<SPLIT>(
            ah, al, hh, hl, state, row0, m, wo_hi, wo_lo, bo, gamma2, beta2, w1_hi, w1_lo, b1, w2_hi, w2_lo,
            b2,
        );
    }

    /// [`column_block_body`] with FP16 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn column_block_half(
        mut state: DisjointSlice<f32>,
        gamma1: &[f32],
        beta1: &[f32],
        wq_hi: &[u32],
        wq_lo: &[u32],
        bq: &[f32],
        keys: &[f32],
        values: &[f32],
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma2: &[f32],
        beta2: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        features: u32,
        queries: u32,
        reserved: u32,
    ) {
        column_block_body::<false>(
            &mut state, gamma1, beta1, wq_hi, wq_lo, bq, keys, values, wo_hi, wo_lo, bo, gamma2, beta2,
            w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, features, queries, reserved,
        );
    }

    /// [`column_block_body`] with split, near-FP32 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn column_block_split(
        mut state: DisjointSlice<f32>,
        gamma1: &[f32],
        beta1: &[f32],
        wq_hi: &[u32],
        wq_lo: &[u32],
        bq: &[f32],
        keys: &[f32],
        values: &[f32],
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma2: &[f32],
        beta2: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        features: u32,
        queries: u32,
        reserved: u32,
    ) {
        column_block_body::<true>(
            &mut state, gamma1, beta1, wq_hi, wq_lo, bq, keys, values, wo_hi, wo_lo, bo, gamma2, beta2,
            w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, features, queries, reserved,
        );
    }

    /// `y = act(x wᵀ + b) (+ r)` on tensor cores from FP16 operands, accumulating
    /// in FP32: 64 x 64 outputs per block of 4 warps, each 32 x 32, in 32-wide K
    /// steps. Both operands are copied asynchronously through a three-stage
    /// pipeline; each product's fragments load one 16-wide sub-step ahead, across
    /// stages too, so the warps rarely wait on shared memory. Every per-thread
    /// pointer and offset is computed once.
    ///
    /// `x` is `[m, ldx]` and `w` `[n, k]`. With `SPLIT`, both also carry a low
    /// FP16 part (`x_lo`, `w_lo`, scaled by [`LO_SCALE`]) and each product is
    /// `hi·hi + (hi·lo + lo·hi) / LO_SCALE`, close to FP32, the corrections
    /// accumulating in FP16 per K step ([`add_correction`]). `mode` bit 1 applies
    /// GELU, bit 2 adds the residual `r` (`ldr`), bit 4 writes `y` in FP32, and
    /// bit 8 writes the next product's operand to `y_hi` (and `y_lo` with
    /// `SPLIT`), all with row stride `ldy`. The shared tiles `a_hi`, `a_lo`,
    /// `b_hi`, `b_lo` hold three stages of 64 x 16 `u32` each.
    /// Needs `k` a multiple of 32 and 16-byte aligned rows.
    /// Grid: (ceil(n / 64), ceil(m / 64)); block: 128 threads.
    #[device]
    fn linear16_body<const SPLIT: bool>(
        (a_hi, a_lo, b_hi, b_lo): (*mut u32, *mut u32, *mut u32, *mut u32),
        (x, x_lo, w, w_lo): (&[u16], &[u16], &[u16], &[u16]),
        (b, r): (&[f32], &[f32]),
        (y, y_hi, y_lo): (
            &mut DisjointSlice<f32>,
            &mut DisjointSlice<u16>,
            &mut DisjointSlice<u16>,
        ),
        (m, n, k): (u32, u32, u32),
        (ldx, ldy, ldr, mode): (u32, u32, u32, u32),
    ) {
        const STAGES: usize = 3;
        /// Copy groups left in flight when a stage must be on hand.
        #[allow(clippy::cast_possible_truncation)] // Three stages.
        const PENDING: u32 = STAGES as u32 - 2;
        const STAGE: usize = 64 * 16; // u32 per operand tile: 64 rows x 32 FP16
        let tid = thread::threadIdx_x() as usize;
        let (warp_id, lane) = (tid / 32, tid % 32);
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let (ldx, ldy, ldr) = (ldx as usize, ldy as usize, ldr as usize);
        let row0 = thread::blockIdx_y() as usize * 64;
        let col0 = thread::blockIdx_x() as usize * 64;
        let steps = k / 32;

        // Copies: two 16-byte chunks per thread and operand, rows tid / 4 and
        // tid / 4 + 32 at chunk tid % 4. Rows past m or n repeat the last one;
        // their outputs are not written. Pointers and offsets are fixed per thread.
        let chunk = tid % 4;
        let mut a_src = [(0usize, 0usize); 2]; // (row offset, shared offset)
        let mut b_src = [0usize; 2];
        #[unroll]
        for j in 0..2 {
            let row = tid / 4 + 32 * j;
            let a_row = if row0 + row < m { row0 + row } else { m - 1 };
            let b_row = if col0 + row < n { col0 + row } else { n - 1 };
            a_src[j] = (a_row * ldx + chunk * 8, swizzled32(row, chunk));
            b_src[j] = b_row * k + chunk * 8;
        }
        let load = |step: usize, stage: usize| unsafe {
            #[unroll]
            for j in 0..2 {
                let at = stage * STAGE + a_src[j].1;
                let (xa, wa) = (a_src[j].0 + step * 32, b_src[j] + step * 32);
                cp_async_cg_16(a_hi.add(at), x.as_ptr().add(xa).cast::<u32>());
                cp_async_cg_16(b_hi.add(at), w.as_ptr().add(wa).cast::<u32>());
                if SPLIT {
                    cp_async_cg_16(a_lo.add(at), x_lo.as_ptr().add(xa).cast::<u32>());
                    cp_async_cg_16(b_lo.add(at), w_lo.as_ptr().add(wa).cast::<u32>());
                }
            }
        };
        // Fragment offsets within a stage, per 16-wide K sub-step.
        let (warp_m, warp_n) = (warp_id / 2, warp_id % 2);
        let mut a_frag = [[0usize; 2]; 2]; // [kstep][mt]
        let mut b_frag = [[0usize; 2]; 2]; // [kstep][pair]
        #[unroll]
        for kstep in 0..2 {
            #[unroll]
            for t in 0..2 {
                a_frag[kstep][t] = swizzled32(warp_m * 32 + t * 16 + lane % 16, kstep * 2 + lane / 16);
                b_frag[kstep][t] = swizzled32(
                    warp_n * 32 + t * 16 + (lane / 16) * 8 + lane % 8,
                    kstep * 2 + (lane / 8) % 2,
                );
            }
        }

        for s in 0..STAGES - 1 {
            if s < steps {
                load(s, s);
            }
            unsafe { cp_async_commit_group() };
        }
        // One sub-step's fragments: A and B, high parts then low parts.
        let fragments = |stage: usize, kstep: usize| -> [[[u32; 4]; 2]; 4] {
            let base = stage * STAGE;
            let mut f = [[[0u32; 4]; 2]; 4];
            #[unroll]
            for i in 0..2 {
                f[0][i] = unsafe { ldmatrix_x4(a_hi.add(base + a_frag[kstep][i])) };
                f[1][i] = unsafe { ldmatrix_x4(b_hi.add(base + b_frag[kstep][i])) };
                if SPLIT {
                    f[2][i] = unsafe { ldmatrix_x4(a_lo.add(base + a_frag[kstep][i])) };
                    f[3][i] = unsafe { ldmatrix_x4(b_lo.add(base + b_frag[kstep][i])) };
                }
            }
            f
        };
        let mut acc = [[[0.0f32; 4]; 4]; 2];
        let mut corr = [[[0.0f32; 4]; 4]; 2];
        let product = |acc: &mut [[[f32; 4]; 4]; 2], corr: &mut [[[f32; 4]; 4]; 2], f: [[[u32; 4]; 2]; 4]| {
            #[unroll]
            for pair in 0..2 {
                #[unroll]
                for mt in 0..2 {
                    #[unroll]
                    for half in 0..2 {
                        let nt = pair * 2 + half;
                        let bh = [f[1][pair][half * 2], f[1][pair][half * 2 + 1]];
                        acc[mt][nt] = unsafe { mma_m16n8k16_f32_f16(acc[mt][nt], f[0][mt], bh) };
                        if SPLIT {
                            let bl = [f[3][pair][half * 2], f[3][pair][half * 2 + 1]];
                            corr[mt][nt] = add_correction(corr[mt][nt], f[0][mt], f[2][mt], bh, bl);
                        }
                    }
                }
            }
        };
        // Stage `read` holds step t; step t + STAGES - 1 copies into `write`. Each
        // product's fragments load one sub-step ahead, the next stage's first ones
        // after its wait and barrier, which also retire the stage the next copy
        // overwrites.
        unsafe { cp_async_wait_group(PENDING) };
        thread::sync_threads();
        let (mut read, mut write) = (0usize, STAGES - 1);
        let mut first = fragments(0, 0);
        for t in 0..steps {
            if t + STAGES - 1 < steps {
                load(t + STAGES - 1, write);
            }
            unsafe { cp_async_commit_group() };
            let second = fragments(read, 1);
            product(&mut acc, &mut corr, first);
            read = if read + 1 == STAGES { 0 } else { read + 1 };
            write = if write + 1 == STAGES { 0 } else { write + 1 };
            if t + 1 < steps {
                unsafe { cp_async_wait_group(PENDING) };
                thread::sync_threads();
                first = fragments(read, 0);
            }
            product(&mut acc, &mut corr, second);
        }

        // Each thread's 8 output columns: their bias once.
        let (r_lane, c_lane) = (lane / 4, (lane % 4) * 2);
        let mut bias = [[0.0f32; 2]; 4];
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for e in 0..2 {
                let col = col0 + warp_n * 32 + nt * 8 + c_lane + e;
                if col < n {
                    bias[nt][e] = unsafe { *b.get_unchecked(col) };
                }
            }
        }
        let next = if SPLIT { 2 } else { 1 };
        #[unroll]
        for mt in 0..2 {
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for e in 0..4 {
                    let row = row0 + warp_m * 32 + mt * 16 + r_lane + (e / 2) * 8;
                    let col = col0 + warp_n * 32 + nt * 8 + c_lane + e % 2;
                    if row < m && col < n {
                        let mut value = acc[mt][nt][e];
                        if SPLIT {
                            value += corr[mt][nt][e] * (1.0 / LO_SCALE);
                        }
                        value += bias[nt][e % 2];
                        if mode & 1 != 0 {
                            value =
                                0.5 * value * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2));
                        }
                        if mode & 2 != 0 {
                            value += unsafe { *r.get_unchecked(row * ldr + col) };
                        }
                        let at = row * ldy + col;
                        if mode & 4 != 0 {
                            unsafe { *y.get_unchecked_mut(at) = value };
                        }
                        if mode & 8 != 0 {
                            store_activation((&mut *y, &mut *y_hi, &mut *y_lo), next, at, value);
                        }
                    }
                }
            }
        }
    }

    /// [`linear16_body`] with FP16 operands, for `optimize="fast"`.
    #[kernel]
    pub fn linear16(
        x: &[u16],
        x_lo: &[u16],
        w: &[u16],
        w_lo: &[u16],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        mut y_hi: DisjointSlice<u16>,
        mut y_lo: DisjointSlice<u16>,
        m: u32,
        n: u32,
        k: u32,
        ldx: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
    ) {
        static mut AS: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        let bt = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BS)) };
        linear16_body::<false>(
            (a, a, bt, bt),
            (x, x_lo, w, w_lo),
            (b, r),
            (&mut y, &mut y_hi, &mut y_lo),
            (m, n, k),
            (ldx, ldy, ldr, mode),
        );
    }

    /// [`linear16_body`] with split, near-FP32 operands, for `optimize="auto"`.
    #[kernel]
    pub fn linear16_split(
        x: &[u16],
        x_lo: &[u16],
        w: &[u16],
        w_lo: &[u16],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        mut y_hi: DisjointSlice<u16>,
        mut y_lo: DisjointSlice<u16>,
        m: u32,
        n: u32,
        k: u32,
        ldx: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
    ) {
        static mut AH: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut AL: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut BH: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut BL: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        let tiles = unsafe {
            (
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)),
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AL)),
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BH)),
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BL)),
            )
        };
        linear16_body::<true>(
            tiles,
            (x, x_lo, w, w_lo),
            (b, r),
            (&mut y, &mut y_hi, &mut y_lo),
            (m, n, k),
            (ldx, ldy, ldr, mode),
        );
    }

    // ---------------------------------------------------------------------------
    // FP32 on the CUDA cores, for `optimize="safe"`: the counterparts of the
    // tensor-core block, attention, and ICL kernels, every product, sum, and
    // softmax in FP32. Weights are FP32 `[n, k]` in row order.
    // ---------------------------------------------------------------------------

    /// `acc[i][j] += Σ_t a[(r0 + i) * lda + t] w[(c0 + j) * ldw + t]` over
    /// `t < k` (a multiple of 4): four rows of a shared tile by four weight
    /// rows, in 16-byte loads. Row starts are 16-byte aligned.
    #[device]
    fn product4x4(
        (a, lda, r0): (*const f32, usize, usize),
        (w, ldw, c0): (*const f32, usize, usize),
        k: usize,
        acc: &mut [[f32; 4]; 4],
    ) {
        let mut t = 0;
        while t < k {
            let mut av = [[0.0f32; 4]; 4];
            let mut wv = [[0.0f32; 4]; 4];
            #[unroll]
            for i in 0..4 {
                av[i] = unsafe { *a.add((r0 + i) * lda + t).cast::<F32x4>() }.0;
                wv[i] = unsafe { *w.add((c0 + i) * ldw + t).cast::<F32x4>() }.0;
            }
            #[unroll]
            for i in 0..4 {
                #[unroll]
                for j in 0..4 {
                    #[unroll]
                    for e in 0..4 {
                        acc[i][j] += av[i][e] * wv[j][e];
                    }
                }
            }
            t += 4;
        }
    }

    /// Rows `[row0, row0 + 32)` of `x[.., 128]` through `LayerNorm` (epsilon
    /// 1e-5) into the FP32 tile `a` `[32, 128]`; rows from `m` on read as zeros.
    /// Each of 8 warps normalizes four rows, four values per lane.
    #[device]
    fn load_normed_f32(xp: *const f32, gamma: &[f32], beta: &[f32], a: *mut f32, row0: usize, m: usize) {
        let tid = thread::threadIdx_x() as usize;
        let (warp_id, lane) = (tid / 32, tid % 32);
        #[unroll]
        for i in 0..4 {
            let row = warp_id * 4 + i;
            let col = lane * 4;
            let v = if row0 + row < m {
                unsafe { *xp.add((row0 + row) * 128 + col).cast::<F32x4>() }.0
            } else {
                [0.0; 4]
            };
            let y = normed4(v, gamma, beta, col);
            unsafe { *a.add(row * 128 + col).cast::<F32x4>() = F32x4(y) };
        }
    }

    /// One lane's four values `v` of a 128-wide row (columns `col..col + 4`;
    /// the warp holds the row) through `LayerNorm`, epsilon 1e-5.
    #[inline(always)]
    fn normed4(v: [f32; 4], gamma: &[f32], beta: &[f32], col: usize) -> [f32; 4] {
        let mean = warp::reduce_sum_f32(v[0] + v[1] + v[2] + v[3]) * (1.0 / 128.0);
        let d = [v[0] - mean, v[1] - mean, v[2] - mean, v[3] - mean];
        let variance =
            warp::reduce_sum_f32(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]) * (1.0 / 128.0);
        let inverse = 1.0 / libm::sqrtf(variance + 1e-5);
        [
            d[0] * inverse * gamma[col] + beta[col],
            d[1] * inverse * gamma[col + 1] + beta[col + 1],
            d[2] * inverse * gamma[col + 2] + beta[col + 2],
            d[3] * inverse * gamma[col + 3] + beta[col + 3],
        ]
    }

    /// [`block_head_body`] in FP32: `out = norm1(x) Wᵀ + b` for `n` (128 or
    /// 384) outputs, then `RoPE` on the first 256 by each row's position
    /// `row % len` when `len` is not zero. Thread `(warp, lane)` computes rows
    /// `4 warp..` by columns `4 lane..` of each 128-column pass; a head's dims
    /// `d` and `d + 8` sit two lanes apart.
    /// Grid: ceil(m / 32); block: 256 threads.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_head_f32(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w: &[f32],
        bias: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut out: DisjointSlice<f32>,
        m: u32,
        n: u32,
        len: u32,
    ) {
        static mut AS: SharedArray<f32, { 32 * 128 }, 16> = SharedArray::UNINIT;
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        let tid = thread::threadIdx_x() as usize;
        let (r0, lane) = ((tid / 32) * 4, tid % 32);
        let (m, n, len) = (m as usize, n as usize, len as usize);
        let row0 = thread::blockIdx_x() as usize * 32;
        load_normed_f32(x.as_ptr(), gamma, beta, a, row0, m);
        thread::sync_threads();
        let mut pass = 0;
        while pass < n / 128 {
            let c0 = pass * 128 + lane * 4;
            let mut acc = [[0.0f32; 4]; 4];
            product4x4((a.cast_const(), 128, r0), (w.as_ptr(), 128, c0), 128, &mut acc);
            #[unroll]
            for i in 0..4 {
                #[unroll]
                for j in 0..4 {
                    acc[i][j] += bias[c0 + j];
                }
            }
            if len != 0 && pass < 2 {
                // With the tables holding [-sin, sin], both halves of a head
                // take `x cos + partner sin`.
                #[unroll]
                for i in 0..4 {
                    let t = (row0 + r0 + i) % len;
                    #[unroll]
                    for j in 0..4 {
                        let partner = warp::shuffle_xor_f32(acc[i][j], 2);
                        let d = (c0 + j) % 16;
                        acc[i][j] = acc[i][j] * cos[t * 16 + d] + partner * sin[t * 16 + d];
                    }
                }
            }
            #[unroll]
            for i in 0..4 {
                let row = row0 + r0 + i;
                if row < m {
                    #[unroll]
                    for j in 0..4 {
                        unsafe { *out.get_unchecked_mut(row * n + c0 + j) = acc[i][j] };
                    }
                }
            }
            pass += 1;
        }
    }

    /// [`block_tail_core`] in FP32, once the attention output of rows `[row0,
    /// row0 + 32)` is in `a` `[32, 128]`; rows from `m` on are not written.
    /// `h` `[32, 256]` is scratch. Warp `w` owns rows `4w..4w + 4` throughout,
    /// lane `l` columns `4l..4l + 4` of each 128-column pass, so `mid` stays
    /// in registers and `norm2` reduces within the warp.
    #[device]
    fn block_tail_core_f32(
        a: *mut f32,
        h: *mut f32,
        state: &mut DisjointSlice<f32>,
        (row0, m): (usize, usize),
        (wo, bo): (&[f32], &[f32]),
        (gamma, beta): (&[f32], &[f32]),
        (w1, b1): (&[f32], &[f32]),
        (w2, b2): (&[f32], &[f32]),
    ) {
        let tid = thread::threadIdx_x() as usize;
        let (r0, c0) = ((tid / 32) * 4, (tid % 32) * 4);
        // mid = state + attended Woᵀ + bo.
        let mut mid = [[0.0f32; 4]; 4];
        product4x4((a.cast_const(), 128, r0), (wo.as_ptr(), 128, c0), 128, &mut mid);
        #[unroll]
        for i in 0..4 {
            let row = row0 + r0 + i;
            #[unroll]
            for j in 0..4 {
                let residual = if row < m {
                    unsafe { *state.get_unchecked_mut(row * 128 + c0 + j) }
                } else {
                    0.0
                };
                mid[i][j] += bo[c0 + j] + residual;
            }
        }
        // norm2(mid) replaces the attention output in A.
        thread::sync_threads();
        #[unroll]
        for i in 0..4 {
            let y = normed4(mid[i], gamma, beta, c0);
            unsafe { *a.add((r0 + i) * 128 + c0).cast::<F32x4>() = F32x4(y) };
        }
        thread::sync_threads();
        // gelu(norm2(mid) W1ᵀ + b1) into H, two 128-column passes.
        #[unroll]
        for pass in 0..2 {
            let c = pass * 128 + c0;
            let mut acc = [[0.0f32; 4]; 4];
            product4x4((a.cast_const(), 128, r0), (w1.as_ptr(), 128, c), 128, &mut acc);
            #[unroll]
            for i in 0..4 {
                let y = [
                    gelu(acc[i][0] + b1[c]),
                    gelu(acc[i][1] + b1[c + 1]),
                    gelu(acc[i][2] + b1[c + 2]),
                    gelu(acc[i][3] + b1[c + 3]),
                ];
                unsafe { *h.add((r0 + i) * 256 + c).cast::<F32x4>() = F32x4(y) };
            }
        }
        thread::sync_threads();
        // state = mid + H W2ᵀ + b2.
        let mut out = [[0.0f32; 4]; 4];
        product4x4((h.cast_const(), 256, r0), (w2.as_ptr(), 256, c0), 256, &mut out);
        #[unroll]
        for i in 0..4 {
            let row = row0 + r0 + i;
            if row < m {
                #[unroll]
                for j in 0..4 {
                    unsafe {
                        *state.get_unchecked_mut(row * 128 + c0 + j) = mid[i][j] + out[i][j] + b2[c0 + j];
                    }
                }
            }
        }
    }

    /// [`block_tail_body`] in FP32 (see [`block_tail_core_f32`]).
    /// Grid: ceil(m / 32); block: 256 threads.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_tail_f32(
        attended: &[f32],
        mut state: DisjointSlice<f32>,
        wo: &[f32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        m: u32,
    ) {
        static mut AS: SharedArray<f32, { 32 * 128 }, 16> = SharedArray::UNINIT;
        static mut HS: SharedArray<f32, { 32 * 256 }, 16> = SharedArray::UNINIT;
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        let h = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HS)) };
        let tid = thread::threadIdx_x() as usize;
        let m = m as usize;
        let row0 = thread::blockIdx_x() as usize * 32;
        let ap = attended.as_ptr();
        #[unroll]
        for i in 0..4 {
            let index = tid + i * 256;
            let (row, col) = (index / 32, (index % 32) * 4);
            let v = if row0 + row < m {
                unsafe { *ap.add((row0 + row) * 128 + col).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            unsafe { *a.add(row * 128 + col).cast::<F32x4>() = v };
        }
        thread::sync_threads();
        block_tail_core_f32(
            a,
            h,
            &mut state,
            (row0, m),
            (wo, bo),
            (gamma, beta),
            (w1, b1),
            (w2, b2),
        );
    }

    /// [`column_block_body`] in FP32: `norm1`, the query projection (the first
    /// 128 rows of the in-projection), attention of each 16-wide head to the
    /// column's 128 cached inducing points, then [`block_tail_core_f32`], in
    /// place on `state` `[B * H, q, 128]`. Thread `t` attends for row `t / 8`
    /// and head `t % 8`.
    /// Grid: (ceil(q / 32), B * H); block: 256 threads.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn column_block_f32(
        mut state: DisjointSlice<f32>,
        gamma1: &[f32],
        beta1: &[f32],
        wq: &[f32],
        bq: &[f32],
        keys: &[f32],
        values: &[f32],
        wo: &[f32],
        bo: &[f32],
        gamma2: &[f32],
        beta2: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        features: u32,
        queries: u32,
        reserved: u32,
    ) {
        static mut AS: SharedArray<f32, { 32 * 128 }, 16> = SharedArray::UNINIT;
        static mut HS: SharedArray<f32, { 32 * 256 }, 16> = SharedArray::UNINIT;
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        let h = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HS)) };
        let tid = thread::threadIdx_x() as usize;
        let (r0, c0) = ((tid / 32) * 4, (tid % 32) * 4);
        let (features, q) = (features as usize, queries as usize);
        let column = thread::blockIdx_y() as usize;
        let tile = thread::blockIdx_x() as usize * 32;
        let row0 = column * q + tile;
        let m = row0 + if q - tile < 32 { q - tile } else { 32 };

        // norm1 into A, then Q = A Wqᵀ + bq into the first 128 columns of H.
        load_normed_f32(state.as_mut_ptr(), gamma1, beta1, a, row0, m);
        thread::sync_threads();
        let mut acc = [[0.0f32; 4]; 4];
        product4x4((a.cast_const(), 128, r0), (wq.as_ptr(), 128, c0), 128, &mut acc);
        #[unroll]
        for i in 0..4 {
            let y = [
                acc[i][0] + bq[c0],
                acc[i][1] + bq[c0 + 1],
                acc[i][2] + bq[c0 + 2],
                acc[i][3] + bq[c0 + 3],
            ];
            unsafe { *h.add((r0 + i) * 256 + c0).cast::<F32x4>() = F32x4(y) };
        }
        thread::sync_threads();

        // Attention to the column's cache stream `f + reserved` of estimator `e`.
        let (row, head) = (tid / 8, tid % 8);
        let (e, f) = (column / features, column % features);
        let kv0 =
            (e * (features + reserved as usize) + f + reserved as usize) * (8 * 128 * 16) + head * 128 * 16;
        let mut query = [0.0f32; 16];
        #[unroll]
        for c in 0..4 {
            let v = unsafe { *h.add(row * 256 + head * 16 + c * 4).cast::<F32x4>() }.0;
            #[unroll]
            for t in 0..4 {
                query[c * 4 + t] = v[t];
            }
        }
        let mut out = [0.0f32; 16];
        attend16_f32(
            &query,
            (
                keys.as_ptr().wrapping_add(kv0),
                values.as_ptr().wrapping_add(kv0),
                16,
            ),
            (0, 128),
            &mut out,
        );
        // The attention output replaces the normalized rows in A.
        #[unroll]
        for c in 0..4 {
            let y = [out[c * 4], out[c * 4 + 1], out[c * 4 + 2], out[c * 4 + 3]];
            unsafe { *a.add(row * 128 + head * 16 + c * 4).cast::<F32x4>() = F32x4(y) };
        }
        thread::sync_threads();
        block_tail_core_f32(
            a,
            h,
            &mut state,
            (row0, m),
            (wo, bo),
            (gamma2, beta2),
            (w1, b1),
            (w2, b2),
        );
    }

    /// Softmax attention of one 16-wide `query` to keys `[from, to)` at `kp`
    /// and `vp` (`stride` values apart, 16-byte aligned), with scale 1 / 4 in
    /// the exp2 domain and an online softmax key by key: the normalized output
    /// into `acc`, which starts at zero.
    #[device]
    fn attend16_f32(
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
    fn max16(x: f32) -> f32 {
        let x = x.max(warp::shuffle_xor_f32(x, 1));
        let x = x.max(warp::shuffle_xor_f32(x, 2));
        let x = x.max(warp::shuffle_xor_f32(x, 4));
        x.max(warp::shuffle_xor_f32(x, 8))
    }

    /// The sum of `x` over the 16 lanes of a half warp.
    #[inline(always)]
    fn sum16(x: f32) -> f32 {
        let x = x + warp::shuffle_xor_f32(x, 1);
        let x = x + warp::shuffle_xor_f32(x, 2);
        let x = x + warp::shuffle_xor_f32(x, 4);
        x + warp::shuffle_xor_f32(x, 8)
    }

    /// [`attention16_tc_body`] in FP32, with its layout: one thread per query
    /// of a (sequence, head), each query against every key in turn.
    /// Grid: (ceil(lq / 64), 8, sequences); block: 64 threads.
    #[kernel]
    pub fn attention16_f32(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
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
    ) {
        let query_row = thread::blockIdx_x() as usize * 64 + thread::threadIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let seq = thread::blockIdx_z() as usize;
        let inner = inner as usize;
        if query_row >= lq as usize {
            return;
        }
        let kv0 =
            (seq / inner) * kv_outer as usize + (seq % inner) * kv_inner as usize + head * kv_head as usize;
        let mut query = [0.0f32; 16];
        let at = seq * q_seq as usize + query_row * q_tok as usize + head * 16;
        #[unroll]
        for d in 0..16 {
            query[d] = q[at + d];
        }
        let (kp, vp) = (
            k.as_ptr().wrapping_add(k_base as usize + kv0),
            v.as_ptr().wrapping_add(v_base as usize + kv0),
        );
        let mut result = [0.0f32; 16];
        attend16_f32(&query, (kp, vp, kv_tok as usize), (0, lk as usize), &mut result);
        let at = seq * o_seq as usize + query_row * o_tok as usize + head * 16;
        #[unroll]
        for d in 0..16 {
            unsafe { *out.get_unchecked_mut(at + d) = result[d] };
        }
    }

    /// Partial ICL attention in FP32 on the CUDA cores over FP32 keys and
    /// values (bytes of `[pairs, np, 64]`), with [`fast_body`]'s split of the
    /// keys into chunks and its outputs: 32 queries per block, keys in 32-key
    /// steps through shared memory. Thread `(y, x)` = `(t / 16, t % 16)`
    /// scores rows `y + 8i` against keys `x + 16j` and accumulates dims
    /// `4x..4x + 4` of the same rows.
    /// Grid: (ceil(qp / 32), pairs * splits); block: 128 threads.
    #[kernel]
    #[launch_bounds(128)]
    pub fn icl_attention_f32(
        q: &[f32],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        // Q and K rows padded to 68 values, so that rows four apart and keys
        // one apart start in different banks.
        static mut QS: SharedArray<f32, { 32 * 68 }, 16> = SharedArray::UNINIT;
        static mut KS: SharedArray<f32, { 32 * 68 }, 16> = SharedArray::UNINIT;
        static mut VS: SharedArray<f32, { 32 * 64 }, 16> = SharedArray::UNINIT;
        static mut PS: SharedArray<f32, { 32 * 33 }, 16> = SharedArray::UNINIT;
        let qs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(QS)) };
        let ks = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KS)) };
        let vs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VS)) };
        let ps = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(PS)) };
        let tid = thread::threadIdx_x() as usize;
        let (ty, tx) = (tid / 16, tid % 16);
        let block = thread::blockIdx_y() as usize;
        let (bh, chunk) = (block / splits as usize, block % splits as usize);
        let q0 = thread::blockIdx_x() as usize * 32;
        let (qp, np) = (qp as usize, np as usize);
        let first_tile = chunk * chunk_tiles as usize;
        let (kp, vp) = (k.as_ptr().cast::<f32>(), v.as_ptr().cast::<f32>());

        // The block's 32 query rows; rows from qp on are zeros.
        #[unroll]
        for i in 0..4 {
            let index = tid + i * 128;
            let (row, c) = (index / 16, (index % 16) * 4);
            let value = if q0 + row < qp {
                unsafe { *q.as_ptr().add((bh * qp + q0 + row) * 64 + c).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            unsafe { *qs.add(row * 68 + c).cast::<F32x4>() = value };
        }

        let mut acc = [[0.0f32; 4]; 4];
        let mut max = [f32::NEG_INFINITY; 4];
        let mut total = [0.0f32; 4];
        // The cache holds np / 64 tiles, so the last chunk can be shorter.
        let tiles = (chunk_tiles as usize).min(np / 64 - first_tile);
        for step in 0..2 * tiles {
            let key0 = first_tile * 64 + step * 32;
            // The previous step is done reading K, V, and P.
            thread::sync_threads();
            #[unroll]
            for i in 0..4 {
                let index = tid + i * 128;
                let (key, c) = (index / 16, (index % 16) * 4);
                let at = (bh * np + key0 + key) * 64 + c;
                unsafe {
                    *ks.add(key * 68 + c).cast::<F32x4>() = *kp.add(at).cast::<F32x4>();
                    *vs.add(key * 64 + c).cast::<F32x4>() = *vp.add(at).cast::<F32x4>();
                }
            }
            thread::sync_threads();

            // Scores, masked past N, in the exp2 domain.
            let mut s = [[0.0f32; 2]; 4];
            let mut c = 0;
            while c < 64 {
                let mut qv = [[0.0f32; 4]; 4];
                #[unroll]
                for i in 0..4 {
                    qv[i] = unsafe { *qs.add((ty + 8 * i) * 68 + c).cast::<F32x4>() }.0;
                }
                #[unroll]
                for j in 0..2 {
                    let kv = unsafe { *ks.add((tx + 16 * j) * 68 + c).cast::<F32x4>() }.0;
                    #[unroll]
                    for i in 0..4 {
                        #[unroll]
                        for e in 0..4 {
                            s[i][j] += qv[i][e] * kv[e];
                        }
                    }
                }
                c += 4;
            }
            #[unroll]
            for i in 0..4 {
                #[unroll]
                for j in 0..2 {
                    let valid = key0 + tx + 16 * j < n_valid as usize;
                    s[i][j] = if valid {
                        s[i][j] * scale_log2
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                // Online softmax over the 16 lanes sharing the row.
                let new = max[i].max(max16(s[i][0].max(s[i][1])));
                // A row with no valid key yet keeps zero weight instead of NaN.
                let base = if new == f32::NEG_INFINITY { 0.0 } else { new };
                let alpha = (max[i] - base).exp2();
                let (p0, p1) = ((s[i][0] - base).exp2(), (s[i][1] - base).exp2());
                total[i] = total[i] * alpha + sum16(p0 + p1);
                max[i] = new;
                #[unroll]
                for e in 0..4 {
                    acc[i][e] *= alpha;
                }
                unsafe {
                    *ps.add((ty + 8 * i) * 33 + tx) = p0;
                    *ps.add((ty + 8 * i) * 33 + tx + 16) = p1;
                }
            }
            thread::sync_threads();

            // O += P V.
            for key in 0..32 {
                let value = unsafe { *vs.add(key * 64 + tx * 4).cast::<F32x4>() }.0;
                #[unroll]
                for i in 0..4 {
                    let p = unsafe { *ps.add((ty + 8 * i) * 33 + key) };
                    #[unroll]
                    for e in 0..4 {
                        acc[i][e] += p * value[e];
                    }
                }
            }
        }

        let out_row = (bh * splits as usize + chunk) * qp + q0;
        #[unroll]
        for i in 0..4 {
            let row = ty + 8 * i;
            if q0 + row < qp {
                #[unroll]
                for e in 0..4 {
                    unsafe {
                        *part.get_unchecked_mut((out_row + row) * 64 + tx * 4 + e) = acc[i][e] / total[i];
                    }
                }
                if tx == 0 {
                    unsafe { *lse.get_unchecked_mut(out_row + row) = max[i] + total[i].log2() };
                }
            }
        }
    }
}
