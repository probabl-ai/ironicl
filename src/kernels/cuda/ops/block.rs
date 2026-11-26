//! Generic pieces of a transformer block that any model's kernels chain: row
//! norms, and queries, keys, or values packed into heads for attention.

#![expect(
    unsafe_code,
    reason = "unchecked stores into disjoint output slots; each unsafe block states its invariant"
)]
#![expect(
    clippy::no_mangle_with_rust_abi,
    reason = "cuda-oxide exports kernels and device functions with #[no_mangle] and the Rust ABI"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "a kernel's parameters are its launch ABI, which device bodies take on"
)]
#![expect(
    clippy::needless_range_loop,
    reason = "#[unroll] loops index register arrays in step"
)]

use cuda_device::vector::F32x4;
use cuda_device::{DisjointSlice, device, thread, warp};

use super::{pack_f16, store_activation};

/// Dense `RMSNorm` directly to packed FP16 for tensor-core products. Each lane owns four
/// adjacent values per vector, so both its FP32 read and its FP16 write are coalesced.
#[allow(
    clippy::cast_ptr_alignment,
    reason = "Device allocations and four-value row offsets are vector aligned"
)]
#[device]
pub(crate) fn rms_half<const V: usize>(
    x: &[f32],
    gamma: &[f32],
    hi: &mut DisjointSlice<u16>,
    rows: u32,
    eps: f32,
) {
    let tid = thread::index_1d().get();
    let (row, lane) = (tid / 32, tid % 32);
    if row >= rows as usize {
        return;
    }
    let mut values = [[0.0_f32; 4]; V];
    let mut sum = 0.0;
    #[unroll]
    for j in 0..V {
        let at = (row * 32 * V + 32 * j + lane) * 4;
        // SAFETY: width is 128 * V; this aligned float4 lies inside the row.
        values[j] = unsafe { (*x.as_ptr().add(at).cast::<F32x4>()).0 };
        #[unroll]
        for c in 0..4 {
            sum += values[j][c] * values[j][c];
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "Widths at most 1024 are exactly representable"
    )]
    let inv = cuda_device::float::rsqrt_approx_f32(warp::reduce_sum_f32(sum) / (128 * V) as f32 + eps);
    #[unroll]
    for j in 0..V {
        let col = (32 * j + lane) * 4;
        // SAFETY: gamma and hi have width 128 * V; each aligned pack belongs to one lane.
        unsafe {
            let g = (*gamma.as_ptr().add(col).cast::<F32x4>()).0;
            let mut v = values[j];
            #[unroll]
            for c in 0..4 {
                v[c] = v[c] * inv * g[c];
            }
            *hi.as_mut_ptr().add(row * 128 * V + col).cast::<u64>() =
                u64::from(pack_f16(v[0], v[1])) | (u64::from(pack_f16(v[2], v[3])) << 32);
        }
    }
}

/// A norm over `width` values (at most `32 * SLOTS`) of each of `rows` rows
/// (row strides `ldx`, `ldy`), written as [`store_activation`] does with
/// `format`: `LayerNorm` with `gamma` and `beta` (population variance), or with
/// `RMS` division by the root mean square, then `gamma`. One warp per row;
/// each lane loads its values once and keeps them in registers. Grid: one warp
/// per row, 256 threads per block.
#[device]
pub(crate) fn norm_rows<const SLOTS: usize, const RMS: bool>(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    y: &mut DisjointSlice<f32>,
    y_hi: &mut DisjointSlice<u16>,
    y_lo: &mut DisjointSlice<u16>,
    rows: u32,
    width: u32,
    ldx: u32,
    ldy: u32,
    format: u32,
    eps: f32,
) {
    let global =
        thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize;
    let (row, lane) = (global / 32, global % 32);
    let (width, ldx, ldy) = (width as usize, ldx as usize, ldy as usize);
    if row >= rows as usize {
        return;
    }
    let mut values = [0.0f32; SLOTS];
    let mut sum = 0.0f32;
    #[unroll]
    for j in 0..SLOTS {
        let i = lane + 32 * j;
        if i < width {
            // SAFETY: `row < rows` (returned above, warp-uniformly) and `i < width <= ldx`; the
            // launcher passes `x` as `[rows, ldx]`.
            values[j] = unsafe { *x.get_unchecked(row * ldx + i) };
            sum += if RMS { values[j] * values[j] } else { values[j] };
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "At most 32 * SLOTS")]
    let count = width as f32;
    let (mean, inverse) = if RMS {
        (
            0.0,
            1.0 / cuda_device::float::sqrt_rn_f32(warp::reduce_sum_f32(sum) / count + eps),
        )
    } else {
        let mean = warp::reduce_sum_f32(sum) / count;
        let mut squares = 0.0f32;
        #[unroll]
        for j in 0..SLOTS {
            if lane + 32 * j < width {
                let d = values[j] - mean;
                squares += d * d;
            }
        }
        (
            mean,
            1.0 / cuda_device::float::sqrt_rn_f32(warp::reduce_sum_f32(squares) / count + eps),
        )
    };
    #[unroll]
    for j in 0..SLOTS {
        let i = lane + 32 * j;
        if i < width {
            let value = if RMS {
                values[j] * inverse * gamma[i]
            } else {
                (values[j] - mean) * inverse * gamma[i] + beta[i]
            };
            // SAFETY: `row < rows` and `i < width <= ldy`; the launcher sizes the buffers
            // `format` names to `[rows, ldy]`, and each (row, i) is one lane's.
            unsafe { store_activation((y, y_hi, y_lo), format, row * ldy + i, value) };
        }
    }
}

/// Token rows `x[sequences * len, heads * dim]` packed for attention as
/// `[sequences * heads, padded, 64]`: for each (token, head), `RoPE` at the
/// token's position in its sequence when `frequencies` (`dim / 2` of them) is
/// not empty, division by the head's root mean square plus 1e-6 with `norm`,
/// then `x * base[head, d] * (1 + tanh(gate[token, head, d])) *
/// key_scale[pair, d]`, each factor only when its buffer is not empty, in FP32
/// into `packed` (`format` 0) or FP16 into `packed_hi` (1). Dims from `dim` to
/// 64, and rows from `len` to `padded`, are zero. One warp per (row, head) of
/// the padded rows; lane `p` holds dims `p` and `p + dim / 2`. Grid: one warp
/// per (padded row, head), 256 threads per block.
#[device]
pub(crate) fn pack_heads_body(
    x: &[f32],
    gate: &[f32],
    base: &[f32],
    key_scale: &[f32],
    frequencies: &[f32],
    packed: &mut DisjointSlice<f32>,
    packed_hi: &mut DisjointSlice<u16>,
    sequences: u32,
    len: u32,
    padded: u32,
    heads: u32,
    dim: u32,
    norm: u32,
    format: u32,
) {
    let global =
        thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize;
    let (item, lane) = (global / 32, global % 32);
    let (len, padded, heads, dim) = (len as usize, padded as usize, heads as usize, dim as usize);
    // Warp-uniform: every lane of a warp has the same item.
    if item >= sequences as usize * padded * heads {
        return;
    }
    let (head, slot) = (item % heads, item / heads);
    let (sequence, row) = (slot / padded, slot % padded);
    let pair = sequence * heads + head;
    let at = (pair * padded + row) * 64;
    if row >= len {
        // A padding row: zeros, so the buffer needs no clearing first.
        // SAFETY: `pair < sequences * heads` and `row < padded`: `at + lane` and `at + lane + 32`
        // lie in `[sequences * heads, padded, 64]` and are this lane's alone.
        unsafe {
            store_packed(packed, packed_hi, format, at + lane, 0.0);
            store_packed(packed, packed_hi, format, at + lane + 32, 0.0);
        }
        return;
    }
    let token = sequence * len + row;
    let half = dim / 2;
    let active = lane < half;
    let dims = [lane, lane + half];
    let mut values = [0.0f32; 2];
    if active {
        #[unroll]
        for k in 0..2 {
            values[k] = x[token * heads * dim + head * dim + dims[k]];
        }
        if !frequencies.is_empty() {
            #[allow(clippy::cast_precision_loss, reason = "Positions are small")]
            let angle = row as f32 * frequencies[lane];
            let (sin, cos) = angle.sin_cos();
            values = [
                values[0] * cos - values[1] * sin,
                values[1] * cos + values[0] * sin,
            ];
        }
    }
    if norm != 0 {
        #[allow(clippy::cast_precision_loss, reason = "Head dims are at most 64")]
        let scale = 1.0
            / cuda_device::float::sqrt_rn_f32(
                warp::reduce_sum_f32(values[0] * values[0] + values[1] * values[1]) / dim as f32 + 1e-6,
            );
        values = [values[0] * scale, values[1] * scale];
    }
    if active {
        #[unroll]
        for k in 0..2 {
            let d = dims[k];
            let mut value = values[k];
            if !base.is_empty() {
                value *= base[head * dim + d];
            }
            if !gate.is_empty() {
                value *= 1.0 + gate[(token * heads + head) * dim + d].tanh();
            }
            if !key_scale.is_empty() {
                value *= key_scale[pair * 64 + d];
            }
            // SAFETY: `pair < sequences * heads`, `row < len <= padded`, and `d < dim <= 64`: `at +
            // d` lies in `[sequences * heads, padded, 64]` and is this lane's alone. `format`
            // selects the buffer the launcher sized; the other may be empty and is not touched.
            unsafe { store_packed(packed, packed_hi, format, at + d, value) };
        }
    }
    let mut d = dim + lane;
    while d < 64 {
        // SAFETY: as above, for the zero padding `dim <= d < 64`, one lane per dim.
        unsafe { store_packed(packed, packed_hi, format, at + d, 0.0) };
        d += 32;
    }
}

/// `value` into slot `at` of `packed` in FP32 (`format` 0) or of `packed_hi`
/// in FP16.
///
/// # Safety
///
/// `at` lies in the buffer `format` selects, and no other thread writes it.
#[device]
unsafe fn store_packed(
    packed: &mut DisjointSlice<f32>,
    packed_hi: &mut DisjointSlice<u16>,
    format: u32,
    at: usize,
    value: f32,
) {
    // SAFETY: the caller's contract.
    unsafe {
        if format == 0 {
            *packed.get_unchecked_mut(at) = value;
        } else {
            *packed_hi.get_unchecked_mut(at) = (value as f16).to_bits();
        }
    }
}
