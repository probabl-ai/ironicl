//! `AArch64` kernels: NEON, and Accelerate on macOS.

#![expect(
    unsafe_code,
    reason = "NEON intrinsics, raw pointer loads, and Accelerate FFI; each unsafe block states its invariant"
)]

use crate::kernels::cpu::*;

/// Longest source that `small_attention` handles; one query row's scores live on the stack.
#[cfg(target_arch = "aarch64")]
pub(crate) const SMALL_ATTENTION_MAX_SOURCES: usize = 128;

/// Exact head-major attention for short sources, such as the row stage's
/// 26-token sequences. It avoids the tiled kernel's per-pair calls. Returns
/// false when the shape is not supported.
#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
pub(crate) fn small_attention(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if source_rows == 0 || source_rows > SMALL_ATTENTION_MAX_SOURCES || !head_dim.is_multiple_of(4) {
        return false;
    }
    assert!(query.len() >= heads * query_rows * head_dim && output.len() >= heads * query_rows * head_dim);
    assert!(key.len() >= heads * source_rows * head_dim && value.len() >= heads * source_rows * head_dim);
    // SAFETY: NEON is a baseline feature of the AArch64 targets this crate builds for. The
    // assertions above give the slice lengths `small_attention_neon` requires, the early return
    // gives `0 < source_rows <= SMALL_ATTENTION_MAX_SOURCES` and `head_dim % 4 == 0`.
    unsafe {
        small_attention_neon(
            query,
            query_rows,
            key,
            value,
            source_rows,
            heads,
            head_dim,
            output,
        );
    }
    true
}

/// # Safety
///
/// The CPU must support NEON. `query` and `output` must hold at least
/// `heads * query_rows * head_dim` values, `key` and `value` at least
/// `heads * source_rows * head_dim`, and `head_dim` must be a multiple of 4.
/// `source_rows > SMALL_ATTENTION_MAX_SOURCES` panics rather than overflowing.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "head_dim is at most a few thousand, exact in f32"
)]
pub(crate) unsafe fn small_attention_neon(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    let scale = (head_dim as f32).sqrt().recip();
    let mut scores = [0.0_f32; SMALL_ATTENTION_MAX_SOURCES];
    // SAFETY: by the caller's length contract, query/output row `(head * query_rows + row)` and
    // key/value row `head * source_rows + source` lie in bounds, and each 4-lane access at
    // `column < head_dim` (step 4, `head_dim % 4 == 0`) stays inside its row.
    unsafe {
        for head in 0..heads {
            let key = key.as_ptr().add(head * source_rows * head_dim);
            let value = value.as_ptr().add(head * source_rows * head_dim);
            for row in 0..query_rows {
                let offset = (head * query_rows + row) * head_dim;
                let query = query.as_ptr().add(offset);
                let output = output.as_mut_ptr().add(offset);
                let mut maximum = f32::NEG_INFINITY;
                for (source, score) in scores[..source_rows].iter_mut().enumerate() {
                    let key = key.add(source * head_dim);
                    let mut dot = vdupq_n_f32(0.0);
                    for column in (0..head_dim).step_by(4) {
                        dot = vfmaq_f32(dot, vld1q_f32(query.add(column)), vld1q_f32(key.add(column)));
                    }
                    *score = vaddvq_f32(dot) * scale;
                    maximum = maximum.max(*score);
                }
                let mut sum = 0.0_f32;
                for score in &mut scores[..source_rows] {
                    *score = (*score - maximum).exp();
                    sum += *score;
                }
                for column in (0..head_dim).step_by(4) {
                    vst1q_f32(output.add(column), vdupq_n_f32(0.0));
                }
                for (source, score) in scores[..source_rows].iter().enumerate() {
                    let weight = vdupq_n_f32(score / sum);
                    let value = value.add(source * head_dim);
                    for column in (0..head_dim).step_by(4) {
                        let target = output.add(column);
                        vst1q_f32(
                            target,
                            vfmaq_f32(vld1q_f32(target), weight, vld1q_f32(value.add(column))),
                        );
                    }
                }
            }
        }
    }
}

/// Smallest `query_rows * source_rows` per head that goes to Accelerate on Apple
/// Silicon. Thresholds from 1024 to 16384 scores measured the same on an M1 Max.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const SGEMM_MIN_ATTENTION_SCORES: usize = 4096;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod accelerate {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        fn vvexpf(output: *mut f32, input: *const f32, count: *const i32);
        fn vDSP_maxv(input: *const f32, stride: isize, output: *mut f32, count: usize);
        fn vDSP_vsadd(
            input: *const f32,
            stride: isize,
            scalar: *const f32,
            output: *mut f32,
            output_stride: isize,
            count: usize,
        );
        fn vDSP_sve(input: *const f32, stride: isize, output: *mut f32, count: usize);
    }

    /// Query rows per score tile, which bounds scratch to `ATTENTION_TILE * source_rows`.
    /// Tiles from 16 to 1024 rows measured slower on an M1 Max: smaller tiles
    /// lose `sgemm` efficiency faster than they gain cache locality.
    const ATTENTION_TILE: usize = 256;

    /// Head-major attention: `softmax(q k^T / sqrt(d)) v` for each head, with
    /// `query` and `output` `[heads, q_rows, d]` and `key`, `value` `[heads, source_rows, d]`.
    /// Returns false when a dimension does not fit the BLAS integer type.
    #[expect(
        clippy::too_many_arguments,
        reason = "a kernel takes its buffers and their shape as plain arguments"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim is at most a few thousand, exact in f32"
    )]
    pub(crate) fn attention(
        query: &[f32],
        q_rows: usize,
        key: &[f32],
        value: &[f32],
        source_rows: usize,
        heads: usize,
        head_dim: usize,
        output: &mut [f32],
    ) -> bool {
        let (Ok(source), Ok(dim), Ok(_)) = (
            i32::try_from(source_rows),
            i32::try_from(head_dim),
            i32::try_from(q_rows.min(ATTENTION_TILE) * source_rows),
        ) else {
            return false;
        };
        assert!(query.len() >= heads * q_rows * head_dim && output.len() >= heads * q_rows * head_dim);
        assert!(key.len() >= heads * source_rows * head_dim && value.len() >= heads * source_rows * head_dim);
        let scale = (head_dim as f32).sqrt().recip();
        let mut scores = vec![0.0_f32; q_rows.min(ATTENTION_TILE) * source_rows];
        let mut sums = vec![0.0_f32; q_rows.min(ATTENTION_TILE)];
        for head in 0..heads {
            let query = &query[head * q_rows * head_dim..];
            let key = &key[head * source_rows * head_dim..];
            let value = &value[head * source_rows * head_dim..];
            let output = &mut output[head * q_rows * head_dim..];
            for start in (0..q_rows).step_by(ATTENTION_TILE) {
                let rows = (q_rows - start).min(ATTENTION_TILE);
                let scores = &mut scores[..rows * source_rows];
                let tile = i32::try_from(rows).expect("tile rows fit i32");
                // SAFETY: row-major `A = query[start..start + rows, dim]` (lda = dim), `B^T` with
                // `B = key[source, dim]` (ldb = dim), `C = scores[rows, source]` (ldc = source).
                // The assertions give `(start + rows) * dim <= q_rows * dim` and `source * dim`
                // within this head's slices; `scores` holds exactly `rows * source`.
                unsafe {
                    cblas_sgemm(
                        ROW_MAJOR,
                        NO_TRANS,
                        TRANS,
                        tile,
                        source,
                        dim,
                        scale,
                        query[start * head_dim..].as_ptr(),
                        dim,
                        key.as_ptr(),
                        dim,
                        0.0,
                        scores.as_mut_ptr(),
                        source,
                    );
                }
                // Unnormalized softmax; the output rows are divided by the sums below,
                // which touches `head_dim` values per row instead of `source_rows`.
                for (row, sum) in scores.chunks_exact_mut(source_rows).zip(&mut sums) {
                    let row_ptr = row.as_mut_ptr();
                    let mut negated_maximum = 0.0_f32;
                    // SAFETY: `row` holds exactly `source_rows` values (`source` as i32 for
                    // vForce); these vDSP and vForce calls read and write only that range, in
                    // place (both allow it), and the scalar out-pointers are live locals.
                    unsafe {
                        vDSP_maxv(row_ptr, 1, &raw mut negated_maximum, source_rows);
                        negated_maximum = -negated_maximum;
                        vDSP_vsadd(row_ptr, 1, &raw const negated_maximum, row_ptr, 1, source_rows);
                        vvexpf(row_ptr, row_ptr, &raw const source);
                        vDSP_sve(row_ptr, 1, sum, source_rows);
                    }
                }
                // SAFETY: `A = scores[rows, source]` (lda = source), `B = value[source, dim]`
                // (ldb = dim), `C = output[start..start + rows, dim]` (ldc = dim); all within
                // the slices by the assertions above, as for the first product.
                unsafe {
                    cblas_sgemm(
                        ROW_MAJOR,
                        NO_TRANS,
                        NO_TRANS,
                        tile,
                        dim,
                        source,
                        1.0,
                        scores.as_ptr(),
                        source,
                        value.as_ptr(),
                        dim,
                        0.0,
                        output[start * head_dim..].as_mut_ptr(),
                        dim,
                    );
                }
                let output = &mut output[start * head_dim..][..rows * head_dim];
                for (row, sum) in output.chunks_exact_mut(head_dim).zip(&sums) {
                    let scale = sum.recip();
                    for value in row {
                        *value *= scale;
                    }
                }
            }
        }
        true
    }

    /// `output[rows, out] = input[rows, in] * weight[out, in]^T + bias`.
    /// Returns false when a dimension does not fit the BLAS integer type.
    pub(crate) fn affine(
        input: &[f32],
        rows: usize,
        input_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output_dim: usize,
        output: &mut [f32],
    ) -> bool {
        let (Ok(m), Ok(n), Ok(k)) = (
            i32::try_from(rows),
            i32::try_from(output_dim),
            i32::try_from(input_dim),
        ) else {
            return false;
        };
        assert!(input.len() >= rows * input_dim && weight.len() >= output_dim * input_dim);
        assert!(output.len() >= rows * output_dim);
        match bias {
            Some(bias) => {
                for row in output[..rows * output_dim].chunks_exact_mut(output_dim) {
                    row.copy_from_slice(&bias[..output_dim]);
                }
            }
            None => output[..rows * output_dim].fill(0.0),
        }
        // SAFETY: row-major `A = input[m, k]` (lda = k), `B^T` with `B = weight[n, k]`
        // (ldb = k), `C = output[m, n]` (ldc = n); the assertions above give each slice at
        // least those lengths, and `m`, `n`, `k` are the checked i32 forms of the dimensions.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                m,
                n,
                k,
                1.0,
                input.as_ptr(),
                k,
                weight.as_ptr(),
                k,
                1.0,
                output.as_mut_ptr(),
                n,
            );
        }
        true
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn attention_dot_for_target(backend: Isa, left: &[f32], right: &[f32]) -> Option<f32> {
    if right.len() < left.len() {
        return None;
    }
    // SAFETY: NEON is baseline on AArch64 (and `Isa::Neon` was selected), and
    // `right.len() >= left.len()` was checked above.
    (backend == Isa::Neon).then(|| unsafe { attention_dot_neon(left, right) })
}

/// # Safety
///
/// The CPU must support NEON, and `right.len() >= left.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn attention_dot_neon(left: &[f32], right: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
    let end = left.len() / 4 * 4;
    let mut accumulator = vdupq_n_f32(0.0);
    // SAFETY: `offset + 4 <= end <= left.len() <= right.len()` (caller contract).
    unsafe {
        for offset in (0..end).step_by(4) {
            accumulator = vfmaq_f32(
                accumulator,
                vld1q_f32(left.as_ptr().add(offset)),
                vld1q_f32(right.as_ptr().add(offset)),
            );
        }
    }
    let mut sum = vaddvq_f32(accumulator);
    for offset in end..left.len() {
        sum = left[offset].mul_add(right[offset], sum);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn attention_value_fma_for_target(
    backend: Isa,
    weight: f32,
    value: &[f32],
    output: &mut [f32],
) -> bool {
    if output.len() < value.len() {
        return false;
    }
    if backend == Isa::Neon {
        // SAFETY: NEON is baseline on AArch64, and `output.len() >= value.len()` was checked
        // above.
        unsafe { attention_value_fma_neon(weight, value, output) }
        true
    } else {
        false
    }
}

/// # Safety
///
/// The CPU must support NEON, and `output.len() >= value.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn attention_value_fma_neon(weight: f32, value: &[f32], output: &mut [f32]) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    let end = value.len() / 4 * 4;
    // SAFETY: `offset + 4 <= end <= value.len() <= output.len()` (caller contract).
    unsafe {
        let weight = vdupq_n_f32(weight);
        for offset in (0..end).step_by(4) {
            let value = vld1q_f32(value.as_ptr().add(offset));
            let current = vld1q_f32(output.as_ptr().add(offset));
            vst1q_f32(output.as_mut_ptr().add(offset), vfmaq_f32(current, value, weight));
        }
    }
    for offset in end..value.len() {
        output[offset] = weight.mul_add(value[offset], output[offset]);
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn gelu_for_target(input: &[f32], output: &mut [f32]) -> bool {
    if output.len() < input.len() {
        return false;
    }
    // SAFETY: NEON is baseline on AArch64, and `output.len() >= input.len()` was checked above.
    unsafe { gelu_neon(input, output) }
    true
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn layer_norm_for_target(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> bool {
    let fits = rows
        .checked_mul(width)
        .is_some_and(|len| input.len() >= len && output.len() >= len)
        && weight.len() >= width
        && bias.is_none_or(|bias| bias.len() >= width);
    if !width.is_multiple_of(16) || !fits {
        return false;
    }
    // SAFETY: NEON is baseline on AArch64; `width % 16 == 0` and every length were checked
    // above.
    unsafe { layer_norm_neon(input, rows, width, weight, bias, output) }
    true
}

/// NEON `LayerNorm` with the same two-pass statistics as the AVX-512 kernel.
///
/// # Safety
///
/// The CPU must support NEON, `width % 16 == 0`, `input` and `output` hold at least
/// `rows * width` values, and `weight` and `bias` (when present) at least `width`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[expect(
    clippy::cast_precision_loss,
    reason = "width is at most a few thousand, exact in f32"
)]
pub(crate) unsafe fn layer_norm_neon(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    use core::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
        vsubq_f32,
    };
    // Four accumulators break the add dependency chain; the caller checked width % 16 == 0.
    /// # Safety
    ///
    /// The CPU must support NEON.
    unsafe fn reduce(accumulators: [float32x4_t; 4]) -> f32 {
        // SAFETY: register-only NEON arithmetic; NEON is baseline on AArch64.
        unsafe {
            vaddvq_f32(vaddq_f32(
                vaddq_f32(accumulators[0], accumulators[1]),
                vaddq_f32(accumulators[2], accumulators[3]),
            ))
        }
    }
    // SAFETY: row `row < rows` starts at `row * width`, inside `input` and `output`; the
    // 16-wide statistics loops and 4-wide output loop cover `column + 16 <= width` and
    // `column + 4 <= width` (`width % 16 == 0`), and `weight`/`bias` hold `width` values.
    unsafe {
        for row in 0..rows {
            let input = input.as_ptr().add(row * width);
            let output = output.as_mut_ptr().add(row * width);
            let mut sums = [vdupq_n_f32(0.0); 4];
            for column in (0..width).step_by(16) {
                for (lane, sum) in sums.iter_mut().enumerate() {
                    *sum = vaddq_f32(*sum, vld1q_f32(input.add(column + lane * 4)));
                }
            }
            let mean = reduce(sums) / width as f32;
            let mean_vector = vdupq_n_f32(mean);
            let mut squares = [vdupq_n_f32(0.0); 4];
            for column in (0..width).step_by(16) {
                for (lane, square) in squares.iter_mut().enumerate() {
                    let centered = vsubq_f32(vld1q_f32(input.add(column + lane * 4)), mean_vector);
                    *square = vfmaq_f32(*square, centered, centered);
                }
            }
            let scale = vdupq_n_f32(
                (reduce(squares) / width as f32 + crate::kernels::reference::LAYER_NORM_EPSILON)
                    .sqrt()
                    .recip(),
            );
            for column in (0..width).step_by(4) {
                let normalized = vmulq_f32(vsubq_f32(vld1q_f32(input.add(column)), mean_vector), scale);
                let mut result = vmulq_f32(normalized, vld1q_f32(weight.as_ptr().add(column)));
                if let Some(bias) = bias {
                    result = vaddq_f32(result, vld1q_f32(bias.as_ptr().add(column)));
                }
                vst1q_f32(output.add(column), result);
            }
        }
    }
}

// The explicit vector loads and stores keep contiguous activation data on the SIMD path.
// The approximation is evaluated lane-wise to retain the scalar approximation's bit behavior.
/// # Safety
///
/// The CPU must support NEON, and `output.len() >= input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gelu_neon(input: &[f32], output: &mut [f32]) {
    use core::arch::aarch64::{vld1q_f32, vst1q_f32};
    let end = input.len() / 4 * 4;
    // SAFETY: `offset + 4 <= end <= input.len() <= output.len()` (caller contract); the
    // transmutes are between `float32x4_t` and `[f32; 4]`, both 16 bytes of plain `f32`.
    unsafe {
        for offset in (0..end).step_by(4) {
            let values = core::mem::transmute::<core::arch::aarch64::float32x4_t, [f32; 4]>(vld1q_f32(
                input.as_ptr().add(offset),
            ));
            let result = values.map(fast_gelu);
            vst1q_f32(
                output.as_mut_ptr().add(offset),
                core::mem::transmute::<[f32; 4], core::arch::aarch64::float32x4_t>(result),
            );
        }
    }
    for offset in end..input.len() {
        output[offset] = fast_gelu(input[offset]);
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn affine_for_target(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    let fits = |count: Option<usize>, len: usize| count.is_some_and(|count| len >= count);
    if !fits(rows.checked_mul(input_dim), input.len())
        || !fits(output_dim.checked_mul(input_dim), weight.len())
        || !fits(rows.checked_mul(output_dim), output.len())
        || bias.is_some_and(|bias| bias.len() < output_dim)
    {
        return false;
    }
    #[cfg(target_os = "macos")]
    if affine_uses_sgemm(rows) && accelerate::affine(input, rows, input_dim, weight, bias, output_dim, output)
    {
        return true;
    }
    // SAFETY: NEON is baseline on AArch64, including Apple Silicon, and every length was
    // above.
    unsafe {
        affine_neon(input, rows, input_dim, weight, bias, output_dim, output);
    }
    true
}

/// # Safety
///
/// The CPU must support NEON. `input` must hold at least `rows * input_dim` values and
/// `weight` at least `output_dim * input_dim`. (`output` and `bias` are index-checked.)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "register-blocked kernel: accumulators and loads named by row and column"
)]
pub(crate) unsafe fn affine_neon(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    // SAFETY: the row loops keep `row + 1 < rows` (and `row + ROWS <= rows` for the helpers),
    // the channel loops `channel + 3 < output_dim` or `channel < output_dim`, and column reads
    // `column + 4 <= vector_end` or `column < input_dim`, so every input and weight read is
    // inside the caller-guaranteed `rows * input_dim` and `output_dim * input_dim` values.
    unsafe {
        let vector_end = input_dim / 4 * 4;
        let mut row = 0;
        while row + 4 < rows {
            affine_neon_rows::<5>(input, row, input_dim, weight, bias, output_dim, output);
            row += 5;
        }
        while row + 3 < rows {
            affine_neon_rows::<4>(input, row, input_dim, weight, bias, output_dim, output);
            row += 4;
        }
        while row + 1 < rows {
            let input0 = input.as_ptr().add(row * input_dim);
            let input1 = input0.add(input_dim);
            let mut channel = 0;
            while channel + 3 < output_dim {
                let weight0 = weight.as_ptr().add(channel * input_dim);
                let weight1 = weight0.add(input_dim);
                let weight2 = weight1.add(input_dim);
                let weight3 = weight2.add(input_dim);
                let mut acc00 = vdupq_n_f32(0.0);
                let mut acc01 = vdupq_n_f32(0.0);
                let mut acc02 = vdupq_n_f32(0.0);
                let mut acc03 = vdupq_n_f32(0.0);
                let mut acc10 = vdupq_n_f32(0.0);
                let mut acc11 = vdupq_n_f32(0.0);
                let mut acc12 = vdupq_n_f32(0.0);
                let mut acc13 = vdupq_n_f32(0.0);
                let mut column = 0;
                while column < vector_end {
                    let inputs0 = vld1q_f32(input0.add(column));
                    let inputs1 = vld1q_f32(input1.add(column));
                    let weights0 = vld1q_f32(weight0.add(column));
                    let weights1 = vld1q_f32(weight1.add(column));
                    let weights2 = vld1q_f32(weight2.add(column));
                    let weights3 = vld1q_f32(weight3.add(column));
                    acc00 = vfmaq_f32(acc00, weights0, inputs0);
                    acc01 = vfmaq_f32(acc01, weights1, inputs0);
                    acc02 = vfmaq_f32(acc02, weights2, inputs0);
                    acc03 = vfmaq_f32(acc03, weights3, inputs0);
                    acc10 = vfmaq_f32(acc10, weights0, inputs1);
                    acc11 = vfmaq_f32(acc11, weights1, inputs1);
                    acc12 = vfmaq_f32(acc12, weights2, inputs1);
                    acc13 = vfmaq_f32(acc13, weights3, inputs1);
                    column += 4;
                }
                let mut sums0 = [
                    vaddvq_f32(acc00),
                    vaddvq_f32(acc01),
                    vaddvq_f32(acc02),
                    vaddvq_f32(acc03),
                ];
                let mut sums1 = [
                    vaddvq_f32(acc10),
                    vaddvq_f32(acc11),
                    vaddvq_f32(acc12),
                    vaddvq_f32(acc13),
                ];
                while column < input_dim {
                    for lane in 0..4 {
                        let value = *weight0.add(lane * input_dim + column);
                        sums0[lane] = value.mul_add(*input0.add(column), sums0[lane]);
                        sums1[lane] = value.mul_add(*input1.add(column), sums1[lane]);
                    }
                    column += 1;
                }
                for lane in 0..4 {
                    let offset = channel + lane;
                    output[row * output_dim + offset] =
                        bias.map_or(sums0[lane], |values| sums0[lane] + values[offset]);
                    output[(row + 1) * output_dim + offset] =
                        bias.map_or(sums1[lane], |values| sums1[lane] + values[offset]);
                }
                channel += 4;
            }
            while channel < output_dim {
                let weight_row = weight.as_ptr().add(channel * input_dim);
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut column = 0;
                while column < vector_end {
                    let weights = vld1q_f32(weight_row.add(column));
                    acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input0.add(column)));
                    acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input1.add(column)));
                    column += 4;
                }
                let mut sum0 = vaddvq_f32(acc0);
                let mut sum1 = vaddvq_f32(acc1);
                while column < input_dim {
                    let value = *weight_row.add(column);
                    sum0 = value.mul_add(*input0.add(column), sum0);
                    sum1 = value.mul_add(*input1.add(column), sum1);
                    column += 1;
                }
                output[row * output_dim + channel] = bias.map_or(sum0, |values| sum0 + values[channel]);
                output[(row + 1) * output_dim + channel] = bias.map_or(sum1, |values| sum1 + values[channel]);
                channel += 1;
            }
            row += 2;
        }
        if row < rows {
            affine_neon_single_row(input, row, input_dim, weight, bias, output_dim, output);
        }
    }
}

/// # Safety
///
/// As [`affine_neon`], and `row + ROWS <= rows` so `input` holds rows `row..row + ROWS`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[expect(
    clippy::needless_range_loop,
    reason = "the loop indexes the accumulator and output arrays in step"
)]
pub(crate) unsafe fn affine_neon_rows<const ROWS: usize>(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    // SAFETY: input rows `row..row + ROWS` and weight rows `channel..channel + 4` (while
    // `channel + 3 < output_dim`) or `channel < output_dim` are in bounds by the contract;
    // column reads stay below `vector_end` (4-wide) or `input_dim` (scalar).
    unsafe {
        let inputs: [*const f32; ROWS] =
            std::array::from_fn(|index| input.as_ptr().add((row + index) * input_dim));
        let vector_end = input_dim / 4 * 4;
        let mut channel = 0;
        while channel + 3 < output_dim {
            let weights = [
                weight.as_ptr().add(channel * input_dim),
                weight.as_ptr().add((channel + 1) * input_dim),
                weight.as_ptr().add((channel + 2) * input_dim),
                weight.as_ptr().add((channel + 3) * input_dim),
            ];
            let mut accumulators = [[vdupq_n_f32(0.0); 4]; ROWS];
            let mut column = 0;
            while column < vector_end {
                let input_vectors: [core::arch::aarch64::float32x4_t; ROWS] =
                    std::array::from_fn(|index| vld1q_f32(inputs[index].add(column)));
                let weight_vectors = [
                    vld1q_f32(weights[0].add(column)),
                    vld1q_f32(weights[1].add(column)),
                    vld1q_f32(weights[2].add(column)),
                    vld1q_f32(weights[3].add(column)),
                ];
                for input_index in 0..ROWS {
                    for weight_index in 0..4 {
                        accumulators[input_index][weight_index] = vfmaq_f32(
                            accumulators[input_index][weight_index],
                            weight_vectors[weight_index],
                            input_vectors[input_index],
                        );
                    }
                }
                column += 4;
            }
            let mut sums = [[0.0; 4]; ROWS];
            for input_index in 0..ROWS {
                for weight_index in 0..4 {
                    sums[input_index][weight_index] = vaddvq_f32(accumulators[input_index][weight_index]);
                }
            }
            while column < input_dim {
                for input_index in 0..ROWS {
                    for weight_index in 0..4 {
                        sums[input_index][weight_index] = (*weights[weight_index].add(column))
                            .mul_add(*inputs[input_index].add(column), sums[input_index][weight_index]);
                    }
                }
                column += 1;
            }
            for input_index in 0..ROWS {
                for weight_index in 0..4 {
                    let offset = channel + weight_index;
                    output[(row + input_index) * output_dim + offset] = bias
                        .map_or(sums[input_index][weight_index], |values| {
                            sums[input_index][weight_index] + values[offset]
                        });
                }
            }
            channel += 4;
        }
        while channel < output_dim {
            let weight_row = weight.as_ptr().add(channel * input_dim);
            let mut accumulators = [vdupq_n_f32(0.0); ROWS];
            let mut column = 0;
            while column < vector_end {
                let weight_vector = vld1q_f32(weight_row.add(column));
                for input_index in 0..ROWS {
                    accumulators[input_index] = vfmaq_f32(
                        accumulators[input_index],
                        weight_vector,
                        vld1q_f32(inputs[input_index].add(column)),
                    );
                }
                column += 4;
            }
            let mut sums = [0.0; ROWS];
            for input_index in 0..ROWS {
                sums[input_index] = vaddvq_f32(accumulators[input_index]);
            }
            while column < input_dim {
                let value = *weight_row.add(column);
                for input_index in 0..ROWS {
                    sums[input_index] = value.mul_add(*inputs[input_index].add(column), sums[input_index]);
                }
                column += 1;
            }
            for input_index in 0..ROWS {
                output[(row + input_index) * output_dim + channel] =
                    bias.map_or(sums[input_index], |values| sums[input_index] + values[channel]);
            }
            channel += 1;
        }
    }
}

/// # Safety
///
/// As [`affine_neon`], and `row < rows` so `input` holds row `row`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[expect(
    clippy::needless_range_loop,
    reason = "the loop indexes the accumulator and output arrays in step"
)]
pub(crate) unsafe fn affine_neon_single_row(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    // SAFETY: as in `affine_neon_rows` with one input row: weight rows stay below
    // `output_dim` and columns below `vector_end` (4-wide) or `input_dim` (scalar).
    unsafe {
        let input_row = input.as_ptr().add(row * input_dim);
        let vector_end = input_dim / 4 * 4;
        let mut channel = 0;
        while channel + 3 < output_dim {
            let weight0 = weight.as_ptr().add(channel * input_dim);
            let weight1 = weight0.add(input_dim);
            let weight2 = weight1.add(input_dim);
            let weight3 = weight2.add(input_dim);
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut column = 0;
            while column < vector_end {
                let inputs = vld1q_f32(input_row.add(column));
                acc0 = vfmaq_f32(acc0, vld1q_f32(weight0.add(column)), inputs);
                acc1 = vfmaq_f32(acc1, vld1q_f32(weight1.add(column)), inputs);
                acc2 = vfmaq_f32(acc2, vld1q_f32(weight2.add(column)), inputs);
                acc3 = vfmaq_f32(acc3, vld1q_f32(weight3.add(column)), inputs);
                column += 4;
            }
            let mut sums = [
                vaddvq_f32(acc0),
                vaddvq_f32(acc1),
                vaddvq_f32(acc2),
                vaddvq_f32(acc3),
            ];
            while column < input_dim {
                for lane in 0..4 {
                    sums[lane] =
                        (*weight0.add(lane * input_dim + column)).mul_add(*input_row.add(column), sums[lane]);
                }
                column += 1;
            }
            for lane in 0..4 {
                let offset = channel + lane;
                output[row * output_dim + offset] =
                    bias.map_or(sums[lane], |values| sums[lane] + values[offset]);
            }
            channel += 4;
        }
        while channel < output_dim {
            let weight_row = weight.as_ptr().add(channel * input_dim);
            let mut accumulator = vdupq_n_f32(0.0);
            let mut column = 0;
            while column < vector_end {
                accumulator = vfmaq_f32(
                    accumulator,
                    vld1q_f32(weight_row.add(column)),
                    vld1q_f32(input_row.add(column)),
                );
                column += 4;
            }
            let mut sum = vaddvq_f32(accumulator);
            while column < input_dim {
                sum = (*weight_row.add(column)).mul_add(*input_row.add(column), sum);
                column += 1;
            }
            output[row * output_dim + channel] = bias.map_or(sum, |values| sum + values[channel]);
            channel += 1;
        }
    }
}
