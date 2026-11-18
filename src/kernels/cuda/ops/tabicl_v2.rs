//! `TabICLv2`'s fused kernels: the 128-wide transformer block head and tail, and the column block.

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
#![allow(clippy::wildcard_imports, reason = "the other ops, through ops/mod.rs")]

use cuda_device::vector::F32x4;
use cuda_device::wmma::{ldmatrix_x4, mma_m16n8k16_f32_f16};
use cuda_device::{DisjointSlice, SharedArray, device, thread, warp};

use super::*;

/// The rest of a pre-norm block with width 128, for 32 rows per block:
/// `mid = state + attended Woᵀ + bo`, then
/// `state = mid + gelu(norm2(mid) W1ᵀ + b1) W2ᵀ + b2`, in place. Weights are
/// FP16 in fragment order (see [`warp_product`]), with low parts for `SPLIT`.
/// 8 warps: two 16-row halves by four column quarters.
/// Grid: ceil(m / 32); block: 256 threads.
#[device]
pub(crate) fn block_tail_body<const SPLIT: bool>(
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
    // SAFETY: `AH`, `AL`, `HH`, and `HL` are this function's own shared statics, reached only
    // through these pointers, here and in [`block_tail_core`], which keep indices in their sizes
    // and accesses between barriers.
    let ah = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)) };
    // SAFETY: as above.
    let al = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AL)) };
    // SAFETY: as above.
    let hh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HH)) };
    // SAFETY: as above.
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
            // SAFETY: `row0 + row < m` and `col + 4 <= 128`: 16 aligned bytes inside `attended`
            // `[m, 128]`.
            unsafe { *ap.add((row0 + row) * 128 + col).cast::<F32x4>() }.0
        } else {
            [0.0; 4]
        };
        // SAFETY: `row < 32` (`index < 1024`) and `col + 3 < 128`: both `tile_at(64, ..)` words lie
        // in `AH` and `AL` (`32 * 64` u32); each thread writes its own pairs, read only after the
        // barrier below.
        unsafe {
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col), v[0], v[1]);
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col + 2), v[2], v[3]);
        }
    }
    thread::sync_threads();
    // SAFETY: all 256 threads of block `row0 / 32` call it (grid ceil(m / 32)); `AH`, `AL` hold the
    // tile, published by the barrier above, `HH`, `HL` are `32 * 128` u32. The launcher passes
    // `state` `[m, 128]` and the fixed 128- and 256-wide fragment-order weights.
    unsafe {
        block_tail_core::<SPLIT>(
            ah, al, hh, hl, state, row0, m, wo_hi, wo_lo, bo, gamma, beta, w1_hi, w1_lo, b1, w2_hi, w2_lo, b2,
        );
    }
}

/// [`block_tail_body`] once the attention output tile of rows
/// `[row0, row0 + 32)` is in A (FP16 parts, 64 u32 per row); rows from `m`
/// on are not written. H (128 u32 per row) is scratch.
///
/// # Safety
///
/// All 256 threads of the block must call it. `ah` and `hh` (and distinct `al`, `hl` with `SPLIT`)
/// point to 16-byte aligned shared tiles of `32 * 64` and `32 * 128` u32, whose earlier accesses a
/// barrier ends, A holding the tile so written. `state` holds `[m, 128]` and rows
/// `[row0, min(row0 + 32, m))` are this block's alone. In fragment order, `wo_hi` holds 16 n-tiles
/// of 8 K steps, `w1_hi` 32 of 8, and `w2_hi` 16 of 16, and their low parts likewise with `SPLIT`.
#[device]
pub(crate) unsafe fn block_tail_core<const SPLIT: bool>(
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
    // SAFETY: by this function's contract every warp runs it with A published, `32 * 64` u32 (8
    // steps, stride 64, rows `wm * 16 + 16 <= 32`), and `wo` holds `wn * 4 + 4 <= 16` n-tiles.
    unsafe {
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
    }
    let mut mid = [[0.0f32; 4]; 4];
    let mut sums = [0.0f32; 2];
    #[unroll]
    for nt in 0..4 {
        #[unroll]
        for e in 0..4 {
            let row = row0 + wm * 16 + lr + (e / 2) * 8;
            let col = wn * 32 + nt * 8 + lc + e % 2;
            if row < m {
                // SAFETY: `row < m`: by this function's contract, inside `state` `[m, 128]` and in
                // this block's rows; this lane alone touches (row, col).
                let mut value = acc[nt][e] + bo[col] + unsafe { *state.get_unchecked_mut(row * 128 + col) };
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
        // SAFETY: slots `row * 4 + wn < 128` of the contract's H tile, which holds no H yet and
        // whose earlier accesses a barrier ended; one lane per (row, quarter).
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
        // SAFETY: slots below 128, published by the barrier above.
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
        // SAFETY: slots `128 + row * 4 + wn < 256`, apart from the means other warps may still
        // read; one lane per (row, quarter).
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
        // SAFETY: slots below 256, published by the barrier above.
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
            // SAFETY: `row < 32` and even `col < 128`: a word of A (`32 * 64` u32), whose last
            // reads (`warp_product` above) precede the barriers since; one lane per pair, read
            // after the barrier below.
            unsafe { store_pair::<SPLIT>(ah, al, tile_at(64, row, col), a, b) };
        }
    }
    thread::sync_threads();

    // H = gelu(normed W1ᵀ + b1): each warp 16 rows x 64 columns, in two
    // passes of 32 to keep registers low.
    #[unroll]
    for pass in 0..2 {
        let mut acc2 = [[0.0f32; 4]; 4];
        let mut corr2 = [[0.0f32; 4]; 4];
        // SAFETY: every warp runs it on the normalized A, published by the barrier above (8 steps,
        // stride 64, rows below 32); `w1` holds `wn * 8 + pass * 4 + 4 <= 32` n-tiles by this
        // function's contract.
        unsafe {
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
        }
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
                // SAFETY: `row < 32` and even `col < 256`: a word of H (`32 * 128` u32), whose
                // `stats` reads precede the last barrier; one lane per pair, read after the
                // barrier below.
                unsafe { store_pair::<SPLIT>(hh, hl, tile_at(128, row, col), gelu(a), gelu(b)) };
            }
        }
    }
    thread::sync_threads();

    // state = mid + H W2ᵀ + b2.
    let mut acc3 = [[0.0f32; 4]; 4];
    let mut corr3 = [[0.0f32; 4]; 4];
    // SAFETY: every warp runs it on H, published by the barrier above (16 steps, stride 128, rows
    // below 32); `w2` holds `wn * 4 + 4 <= 16` n-tiles of 16 steps by this function's contract.
    unsafe {
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
    }
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
                // SAFETY: `row < m`: inside the contract's `state` `[m, 128]`, the (row, col) this
                // lane alone read above.
                unsafe { *state.get_unchecked_mut(row * 128 + col) = value };
            }
        }
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
pub(crate) fn block_head_body<const SPLIT: bool>(
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
    // SAFETY: `AH` and `AL` are this function's own shared statics, reached only through these
    // pointers by `load_normed` and `warp_product`, between barriers.
    let ah = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)) };
    // SAFETY: as above.
    let al = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AL)) };
    let tid = thread::threadIdx_x() as usize;
    let (warp_id, lane) = (tid / 32, tid % 32);
    let (m, n, len) = (m as usize, n as usize, len as usize);
    let row0 = thread::blockIdx_x() as usize * 32;

    // SAFETY: all 256 threads call it; the launcher passes `x` `[m, 128]` (16-byte aligned rows),
    // `AH` and `AL` are `32 * 64` u32, and the barrier below ends the stores.
    unsafe { load_normed::<SPLIT>(x.as_ptr(), gamma, beta, ah, al, row0, m) };
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
        // SAFETY: every warp runs it (the loop bound is uniform) on A, published by the barrier
        // above (8 steps, stride 64, rows below 32); `col0 / 8 + 4 <= n / 8` n-tiles, which the
        // launcher's in-projection `[n / 8, 8, 32, 2]` u32 holds.
        unsafe {
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
        }
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
                    // SAFETY: `row < m` and `col < col0 + 32 <= (wn + 1) * quarter <= n`: inside
                    // `out` `[m, n]`; each (row, col) belongs to one lane of one block.
                    unsafe { *out.get_unchecked_mut(row * n + col) = value[nt][e] };
                }
            }
        }
        pass += 1;
    }
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
pub(crate) fn column_block_body<const SPLIT: bool>(
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
    // SAFETY: `AH` and `HH` (and `AL` and `HL` below) are this function's own shared statics,
    // reached only through these pointers, here and in [`block_tail_core`], in bounds and between
    // barriers.
    let ah = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)) };
    // SAFETY: as above.
    let hh = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HH)) };
    // FP16 never touches the low parts, so it leaves their buffers unused.
    let (al, hl) = if SPLIT {
        // SAFETY: as above.
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
    // SAFETY: all 256 threads call it; rows `[row0, m)` with `m <= (column + 1) * q` lie in the
    // launcher's `state` `[B * H * q, 128]`, which this block writes only later, after barriers.
    // `AH` (and `AL` with `SPLIT`) are `32 * 64` u32, and the barrier below ends the stores.
    unsafe { load_normed::<SPLIT>(state.as_mut_ptr(), gamma1, beta1, ah, al, row0, m) };
    thread::sync_threads();
    let mut acc = [[0.0f32; 4]; 4];
    let mut corr = [[0.0f32; 4]; 4];
    // SAFETY: every warp runs it on A, published by the barrier above (8 steps, stride 64, rows
    // below 32); `wq` (the in-projection's first 128 rows) holds `wn * 4 + 4 <= 16` n-tiles.
    unsafe {
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
    }
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
            // SAFETY: `row < 32` and even `col < 128`: a word of the Q tile in `HH` (and `HL`),
            // untouched before; one lane per pair, read after the barrier below.
            unsafe { store_pair::<SPLIT>(hh, hl, tile_at(64, row, col), a, b) };
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
        // SAFETY: row `mt * 16 + lane % 16 < 32` at chunk `(head * 2 + lane / 16) ^ (row & 7) <
        // 16`: an aligned 16-byte run of the Q tile in `HH` (64 u32 per row), published by the
        // barrier above; the whole warp executes it.
        qa[mt] = unsafe { ldmatrix_x4(hh.add(at)) };
        if SPLIT {
            // SAFETY: as above, in `HL`.
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
            // SAFETY: `at + 9 <= kv0 + 127 * 16 + 15`: key `key0 + nt * 8 + lr < 128` of stream `e
            // * (H + reserved) + f + reserved` of `B * (H + reserved)`, which the launcher's caches
            // `[B, H + reserved, 8, 128, 16]` hold (checked by `debug_assert!` only).
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
                // SAFETY: a full-warp product; the key loop's bounds are uniform.
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
                // SAFETY: `at + 144 <= kv0 + 118 * 16 + 15 + 144 = kv0 + 2047`: inside the stream,
                // as for the keys.
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
                    // SAFETY: a full-warp product, as above.
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
                // SAFETY: `row < 32` and even `col = head * 16 + nt * 8 + lc < 128`: a word of A,
                // whose last reads (`warp_product`) precede an earlier barrier; one lane per pair,
                // read after the barrier below.
                unsafe { store_pair::<SPLIT>(ah, al, tile_at(64, row, col), a / total, b / total) };
            }
        }
    }
    thread::sync_threads();
    // SAFETY: all 256 threads call it with A published by the barrier above and `HH` last read
    // before it; `AL`, `HL` are distinct with `SPLIT`. Block (tile, column) alone owns rows `[row0,
    // m)` of `state` `[B * H * q, 128]`; the launcher passes the fixed fragment-order layers.
    unsafe {
        block_tail_core::<SPLIT>(
            ah, al, hh, hl, state, row0, m, wo_hi, wo_lo, bo, gamma2, beta2, w1_hi, w1_lo, b1, w2_hi, w2_lo,
            b2,
        );
    }
}

/// [`block_tail_core`] in FP32, once the attention output of rows `[row0,
/// row0 + 32)` is in `a` `[32, 128]`; rows from `m` on are not written.
/// `h` `[32, 256]` is scratch. Warp `w` owns rows `4w..4w + 4` throughout,
/// lane `l` columns `4l..4l + 4` of each 128-column pass, so `mid` stays
/// in registers and `norm2` reduces within the warp.
///
/// # Safety
///
/// All 256 threads of the block must call it. `a` and `h` point to 16-byte aligned shared `[32,
/// 128]` and `[32, 256]` tiles whose earlier accesses a barrier ends, `a` holding the tile so
/// written. `state` holds `[m, 128]` and rows `[row0, min(row0 + 32, m))` are this block's alone.
/// `wo` holds `[128, 128]`, `w1` `[256, 128]`, and `w2` `[128, 256]`, 16-byte aligned.
#[device]
pub(crate) unsafe fn block_tail_core_f32(
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
    // SAFETY: by this function's contract, A `[32, 128]` is published and `wo` is `[128, 128]`:
    // rows `r0 + 4 <= 32` and `c0 + 4 <= 128`, 128 values each, 16-byte aligned.
    unsafe { product4x4((a.cast_const(), 128, r0), (wo.as_ptr(), 128, c0), 128, &mut mid) };
    #[unroll]
    for i in 0..4 {
        let row = row0 + r0 + i;
        #[unroll]
        for j in 0..4 {
            let residual = if row < m {
                // SAFETY: `row < m`: by this function's contract, inside `state` `[m, 128]` and in
                // this block's rows; this thread alone reads and later writes the (row, column).
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
        // SAFETY: `r0 + i < 32` and `c0 + 4 <= 128`: inside the contract's `a` `[32, 128]`; every
        // thread finished reading A in `product4x4` before the barrier above, and each slot is this
        // thread's.
        unsafe { *a.add((r0 + i) * 128 + c0).cast::<F32x4>() = F32x4(y) };
    }
    thread::sync_threads();
    // gelu(norm2(mid) W1ᵀ + b1) into H, two 128-column passes.
    #[unroll]
    for pass in 0..2 {
        let c = pass * 128 + c0;
        let mut acc = [[0.0f32; 4]; 4];
        // SAFETY: the normalized A is published by the barrier above and `w1` is `[256, 128]` by
        // this function's contract: rows `r0 + 4 <= 32` and `c + 4 <= 256`, 16-byte aligned.
        unsafe { product4x4((a.cast_const(), 128, r0), (w1.as_ptr(), 128, c), 128, &mut acc) };
        #[unroll]
        for i in 0..4 {
            let y = [
                gelu(acc[i][0] + b1[c]),
                gelu(acc[i][1] + b1[c + 1]),
                gelu(acc[i][2] + b1[c + 2]),
                gelu(acc[i][3] + b1[c + 3]),
            ];
            // SAFETY: `r0 + i < 32` and `c + 4 <= 256`: inside `h` `[32, 256]`, 16-byte aligned;
            // each slot is this thread's, read after the barrier below.
            unsafe { *h.add((r0 + i) * 256 + c).cast::<F32x4>() = F32x4(y) };
        }
    }
    thread::sync_threads();
    // state = mid + H W2ᵀ + b2.
    let mut out = [[0.0f32; 4]; 4];
    // SAFETY: H `[32, 256]` is published by the barrier above and `w2` is `[128, 256]` by this
    // function's contract: rows `r0 + 4 <= 32` and `c0 + 4 <= 128`, 256 values each, aligned.
    unsafe { product4x4((h.cast_const(), 256, r0), (w2.as_ptr(), 256, c0), 256, &mut out) };
    #[unroll]
    for i in 0..4 {
        let row = row0 + r0 + i;
        if row < m {
            #[unroll]
            for j in 0..4 {
                // SAFETY: `row < m`: as for the residual read above, this thread's own (row,
                // column).
                unsafe {
                    *state.get_unchecked_mut(row * 128 + c0 + j) = mid[i][j] + out[i][j] + b2[c0 + j];
                }
            }
        }
    }
}
