//! Dense products: tensor-core tiles in FP16 and split FP16, and FP32 tiles on the CUDA cores.

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
    clippy::needless_range_loop,
    reason = "#[unroll] loops index register arrays in step"
)]
#![expect(
    clippy::cast_ptr_alignment,
    reason = "vector loads at offsets aligned by construction"
)]
#![expect(clippy::inline_always, reason = "device helpers inline into their kernels")]

use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::convert::cvt_f32x2_f16x2;
use cuda_device::vector::F32x4;
use cuda_device::wmma::{ldmatrix_x4, mma_m16n8k16_f16_f16, mma_m16n8k16_f32_f16};
use cuda_device::{DisjointSlice, device, thread, warp};

/// Store an activation `value` at `at` as the product reading it next takes
/// it: FP32 in `y` (`format` 0), FP16 in `hi` (1), or FP16 high and scaled low
/// parts in `hi` and `lo` (2).
///
/// # Safety
///
/// `at` must be in bounds of each buffer `format` selects (`y` for 0, `hi` for 1, `hi` and `lo`
/// for 2), and no other thread may access element `at` of those buffers during the launch.
#[inline(always)]
pub(crate) unsafe fn store_activation(
    (y, hi, lo): (
        &mut DisjointSlice<f32>,
        &mut DisjointSlice<u16>,
        &mut DisjointSlice<u16>,
    ),
    format: u32,
    at: usize,
    value: f32,
) {
    // SAFETY: by this function's contract, `at` is in bounds of the buffers `format` selects and
    // this thread's alone.
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

/// Scale of the low FP16 part of a split value: `v ≈ hi + lo / LO_SCALE`.
pub(crate) const LO_SCALE: f32 = 2048.0;

/// u32 index of 16-byte chunk `chunk` (0..4) of row `row` in a 32-wide FP16
/// tile, swizzled so that `ldmatrix` reads of 8 rows at one chunk hit 8
/// different bank groups.
#[inline(always)]
pub(crate) fn swizzled32(row: usize, chunk: usize) -> usize {
    row * 16 + ((chunk ^ ((row >> 1) & 3)) * 4)
}

/// Two values as packed FP16, the first in the low half.
#[inline(always)]
pub(crate) fn pack_f16(a: f32, b: f32) -> u32 {
    u32::from((a as f16).to_bits()) | (u32::from((b as f16).to_bits()) << 16)
}

/// Low parts of two values: `(v - f16(v)) * LO_SCALE` as packed FP16.
#[inline(always)]
pub(crate) fn pack_lo(a: f32, b: f32) -> u32 {
    let (ha, hb) = ((a as f16) as f32, (b as f16) as f32);
    pack_f16((a - ha) * LO_SCALE, (b - hb) * LO_SCALE)
}

/// The split-precision correction `a_hi b_lo + a_lo b_hi` (low parts scaled by
/// [`LO_SCALE`]) for one 16-wide K step, added to `corr`. It accumulates in
/// FP16, which runs at twice the FP32-accumulator rate: one step's sum is
/// far from FP16's range, and its rounding error is about 2^-11 of a term
/// that is itself divided by `LO_SCALE`.
#[inline(always)]
pub(crate) fn add_correction(
    corr: [f32; 4],
    a_hi: [u32; 4],
    a_lo: [u32; 4],
    b_hi: [u32; 2],
    b_lo: [u32; 2],
) -> [f32; 4] {
    // SAFETY: a full-warp product on the caller's fragments: every caller runs it on all 32 lanes
    // under warp-uniform control flow.
    let mut c = unsafe { mma_m16n8k16_f16_f16([0; 2], a_hi, b_lo) };
    // SAFETY: as above.
    c = unsafe { mma_m16n8k16_f16_f16(c, a_lo, b_hi) };
    let (c0, c1) = cvt_f32x2_f16x2(c[0]);
    let (c2, c3) = cvt_f32x2_f16x2(c[1]);
    [corr[0] + c0, corr[1] + c1, corr[2] + c2, corr[3] + c3]
}

/// u32 index of the FP16 pair at (`row`, even `col`) in a tile with `stride`
/// u32 per row, 16-byte chunks XOR-swizzled by the row's low three bits.
#[inline(always)]
pub(crate) fn tile_at(stride: usize, row: usize, col: usize) -> usize {
    row * stride + (((col / 8) ^ (row & 7)) * 4) + (col % 8) / 2
}

/// Store an (even, odd) column pair as FP16, and its low parts with `SPLIT`.
///
/// # Safety
///
/// `hi` (and `lo` with `SPLIT`) must be valid for writing the aligned `u32` at offset `at`, and no
/// other thread may access that word between the barriers around this store.
#[inline(always)]
pub(crate) unsafe fn store_pair<const SPLIT: bool>(hi: *mut u32, lo: *mut u32, at: usize, a: f32, b: f32) {
    // SAFETY: by this function's contract, word `at` of `hi` (and `lo` with `SPLIT`) is writable
    // and this thread's until the next barrier.
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
///
/// # Safety
///
/// The whole warp must call it under warp-uniform control flow, with `lane` its lane index. `ks` is
/// 8 or 16, `stride = 8 * ks`, and `a_hi` (and `a_lo` with `SPLIT`) point to 16-byte aligned shared
/// tiles holding rows `row0..row0 + 16`, written before a barrier and not written until the next.
/// `w_hi` (and `w_lo` with `SPLIT`) hold at least `(nt0 + NT) * ks * 64` u32.
#[device]
pub(crate) unsafe fn warp_product<const SPLIT: bool, const NT: usize>(
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
        // SAFETY: by this function's contract, row `row0 + lane % 16` and chunk `(step * 2 + lane /
        // 16) ^ (row & 7) < 2 * ks` address an aligned 16-byte run of the published tile
        // (`stride = 8 * ks` u32 per row); the whole warp executes it.
        let fa = unsafe { ldmatrix_x4(a_hi.add(at)) };
        let fal = if SPLIT {
            // SAFETY: as above, in `a_lo`.
            unsafe { ldmatrix_x4(a_lo.add(at)) }
        } else {
            [0u32; 4]
        };
        #[unroll]
        for nt in 0..NT {
            let index = (((nt0 + nt) * ks + step) * 32 + lane) * 2;
            // SAFETY: `index + 1 < (nt0 + NT) * ks * 64`, which by this function's contract `w_hi`
            // holds.
            let fb = unsafe { [*w_hi.get_unchecked(index), *w_hi.get_unchecked(index + 1)] };
            // SAFETY: a full-warp product on the fragments above; by this function's contract the
            // whole warp runs it, and the loop bounds are warp-uniform.
            acc[nt] = unsafe { mma_m16n8k16_f32_f16(acc[nt], fa, fb) };
            if SPLIT {
                // SAFETY: as for `fb`, in `w_lo`, which `SPLIT` callers pass in the same shape.
                let fbl = unsafe { [*w_lo.get_unchecked(index), *w_lo.get_unchecked(index + 1)] };
                corr[nt] = add_correction(corr[nt], fa, fal, fb, fbl);
            }
        }
        step += 1;
    }
}

#[inline(always)]
pub(crate) fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2))
}

/// Rows `[row0, row0 + 32)` of `x[.., 128]` through `LayerNorm` (epsilon 1e-5)
/// into A as FP16 parts (64 u32 per row); rows from `m` on become zeros. Each
/// of 8 warps normalizes four rows, four values per lane.
///
/// # Safety
///
/// All 256 threads of the block must call it. `xp` must be 16-byte aligned and valid for reading
/// rows `[row0, m)` of `[.., 128]`. `ah` (and `al` with `SPLIT`) must point to shared tiles of
/// `32 * 64` u32 that no other thread accesses until the barrier after this call.
#[device]
pub(crate) unsafe fn load_normed<const SPLIT: bool>(
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
            // SAFETY: `row0 + row < m` and `col + 4 <= 128`: by this function's contract, 16
            // aligned readable bytes of `x`.
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
        // SAFETY: `row = warp_id * 4 + i < 32` (256 threads) and `col + 3 < 128`, so both
        // `tile_at(64, ..)` words lie in the `32 * 64` tiles of this function's contract, which
        // no other thread touches before the caller's barrier; each lane writes its own pairs.
        unsafe {
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col), y[0], y[1]);
            store_pair::<SPLIT>(ah, al, tile_at(64, row, col + 2), y[2], y[3]);
        }
    }
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
///
/// # Safety
///
/// Every thread of a launch with that grid and block must call it, with `m, n >= 1`. `a_hi` and
/// `b_hi` (and `a_lo`, `b_lo` with `SPLIT`) point to distinct 16-byte aligned shared tiles of
/// `3 * 64 * 16` u32 that nothing else touches. `x` (`x_lo`) holds `[m, ldx = k]` and `w` (`w_lo`)
/// `[n, k]`, 16-byte aligned with `k` a multiple of 32; `b` holds `n`. With `mode & 2` `r` holds
/// `[m, ldr]`, with `mode & 4` `y`, and with `mode & 8` `y_hi` (`y_lo` with `SPLIT`), `[m, ldy]`,
/// with `n <= ldr, ldy`.
#[device]
pub(crate) unsafe fn linear16_body<const SPLIT: bool>(
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
    #[expect(clippy::cast_possible_truncation, reason = "Three stages")]
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
    // SAFETY: by this function's contract, rows are clamped below `m` and `n` (both at least 1)
    // and `step < k / 32`, so each 16 bytes lie in `x` `[m, ldx = k]`, `w` `[n, k]`, and with
    // `SPLIT` their low parts; `k` a multiple of 32 keeps them 16-byte aligned. `stage * STAGE +
    // swizzled32(row < 64, chunk < 4) + 4 <= 3 * STAGE`, the contract's tile size. A stage is
    // overwritten only after the barrier that follows its last `ldmatrix`.
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
        // SAFETY: commits this thread's copies of the prologue stage (or an empty group), one group
        // per stage for the waits below.
        unsafe { cp_async_commit_group() };
    }
    // One sub-step's fragments: A and B, high parts then low parts.
    let fragments = |stage: usize, kstep: usize| -> [[[u32; 4]; 2]; 4] {
        let base = stage * STAGE;
        let mut f = [[[0u32; 4]; 2]; 4];
        #[unroll]
        for i in 0..2 {
            // SAFETY: fragment rows below 64 at chunks below 4 of `stage < 3`: aligned 16-byte runs
            // inside the tiles, whose copies the last wait and barrier completed; the whole warp
            // executes it.
            f[0][i] = unsafe { ldmatrix_x4(a_hi.add(base + a_frag[kstep][i])) };
            // SAFETY: as above.
            f[1][i] = unsafe { ldmatrix_x4(b_hi.add(base + b_frag[kstep][i])) };
            if SPLIT {
                // SAFETY: as above.
                f[2][i] = unsafe { ldmatrix_x4(a_lo.add(base + a_frag[kstep][i])) };
                // SAFETY: as above.
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
                    // SAFETY: a full-warp product on `ldmatrix` fragments; all loop bounds are
                    // block-uniform.
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
    // SAFETY: `PENDING` is a constant: stage 0's group completes, and the barrier after publishes
    // it.
    unsafe { cp_async_wait_group(PENDING) };
    thread::sync_threads();
    let (mut read, mut write) = (0usize, STAGES - 1);
    let mut first = fragments(0, 0);
    for t in 0..steps {
        if t + STAGES - 1 < steps {
            load(t + STAGES - 1, write);
        }
        // SAFETY: commits this step's copies (or an empty group), so that each wait counts one
        // group per stage.
        unsafe { cp_async_commit_group() };
        let second = fragments(read, 1);
        product(&mut acc, &mut corr, first);
        read = if read + 1 == STAGES { 0 } else { read + 1 };
        write = if write + 1 == STAGES { 0 } else { write + 1 };
        if t + 1 < steps {
            // SAFETY: as above: the group of step `t + 1` completes, and the barrier after
            // publishes it.
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
                // SAFETY: `col < n`, and by this function's contract `b` holds `n` biases.
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
                        value = 0.5 * value * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2));
                    }
                    if mode & 2 != 0 {
                        // SAFETY: `row < m`, `col < n <= ldr`; with `mode & 2` this function's
                        // contract has `r` hold `[m, ldr]`.
                        value += unsafe { *r.get_unchecked(row * ldr + col) };
                    }
                    let at = row * ldy + col;
                    if mode & 4 != 0 {
                        // SAFETY: `row < m` and `col < n <= ldy`; with `mode & 4` this function's
                        // contract has `y` hold `[m, ldy]`. Each (row, col) belongs to one lane of
                        // one block.
                        unsafe { *y.get_unchecked_mut(at) = value };
                    }
                    if mode & 8 != 0 {
                        // SAFETY: `at = row * ldy + col` with `row < m`, `col < n <= ldy`: inside
                        // `y_hi` (and `y_lo`, which `next = 2` selects with `SPLIT`) `[m, ldy]` by
                        // this function's contract; each (row, col) is one lane's of one block.
                        unsafe { store_activation((&mut *y, &mut *y_hi, &mut *y_lo), next, at, value) };
                    }
                }
            }
        }
    }
}

/// `acc[i][j] += Σ_t a[(r0 + i) * lda + t] w[(c0 + j) * ldw + t]` over
/// `t < k` (a multiple of 4): four rows of a shared tile by four weight
/// rows, in 16-byte loads. Row starts are 16-byte aligned.
///
/// # Safety
///
/// `a` and `w` must be 16-byte aligned, `lda` and `ldw` multiples of 4, and `k` a multiple of 4
/// with `k <= lda, ldw`. The first `k` values of rows `r0..r0 + 4` of `a` and `c0..c0 + 4` of `w`
/// must be readable and, in shared memory, written before a barrier and not written until the next.
#[device]
pub(crate) unsafe fn product4x4(
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
            // SAFETY: by this function's contract, row `r0 + i` of `a` is readable for `t + 4 <= k`
            // values, and `lda`, `t`, and the base keep the 16-byte load aligned.
            av[i] = unsafe { *a.add((r0 + i) * lda + t).cast::<F32x4>() }.0;
            // SAFETY: as above, for `w`.
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
///
/// # Safety
///
/// All 256 threads of the block must call it. `xp` must be 16-byte aligned and valid for reading
/// rows `[row0, m)` of `[.., 128]`. `a` must point to a 16-byte aligned shared `[32, 128]` tile
/// that no other thread accesses until the barrier after this call.
#[device]
pub(crate) unsafe fn load_normed_f32(
    xp: *const f32,
    gamma: &[f32],
    beta: &[f32],
    a: *mut f32,
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
            // SAFETY: `row0 + row < m` and `col + 4 <= 128`: by this function's contract, 16
            // aligned readable bytes of `x`.
            unsafe { *xp.add((row0 + row) * 128 + col).cast::<F32x4>() }.0
        } else {
            [0.0; 4]
        };
        let y = normed4(v, gamma, beta, col);
        // SAFETY: `row = warp_id * 4 + i < 32` (256 threads) and `col + 4 <= 128`: inside the
        // contract's `a` `[32, 128]`, 16-byte aligned; each lane writes its own four values.
        unsafe { *a.add(row * 128 + col).cast::<F32x4>() = F32x4(y) };
    }
}

/// One lane's four values `v` of a 128-wide row (columns `col..col + 4`;
/// the warp holds the row) through `LayerNorm`, epsilon 1e-5.
#[inline(always)]
pub(crate) fn normed4(v: [f32; 4], gamma: &[f32], beta: &[f32], col: usize) -> [f32; 4] {
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
