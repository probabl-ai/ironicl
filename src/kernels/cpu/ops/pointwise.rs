//! Elementwise kernels: GELU, QASS modulation, BF16 conversion, fast exp/erf/tanh.

#![allow(
    dead_code,
    unsafe_code,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::missing_transmute_annotations,
    clippy::wildcard_imports // Sibling items differ per target.
)]

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::kernels::cpu::*;

pub(crate) const SQRT_2: f32 = core::f32::consts::SQRT_2;

pub(crate) fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    u16::try_from(bits.wrapping_add(rounding_bias) >> 16).expect("rounded BF16 bits fit u16")
}

/// Whether the CPU has native BF16 support (`AVX512_BF16`). Gates storing a
/// static context's ICL values in BF16 under `optimize="fast"`.
pub(crate) fn bf16_storage_supported() -> bool {
    bf16_storage_supported_for_target()
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) const fn bf16_storage_supported_for_target() -> bool {
    false
}

pub(crate) fn bf16_bits_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// Apply the native GELU approximation when the host supports it.
///
/// The caller validates that `input` and `output` have equal lengths.
pub(crate) fn gelu_f32(input: &[f32], output: &mut [f32]) -> bool {
    gelu_for_target(input, output)
}

/// Apply `query * base_scale * (1 + tanh(mlp_output))` natively.
///
/// The caller validates the head-major slice lengths and dimensions.
pub(crate) fn qass_modulate_f32(
    queries: &[f32],
    base_scale: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    qass_modulate_for_target(queries, base_scale, heads, rows, head_dim, output)
}

/// Expand BF16 storage values into FP32 tile scratch.
pub(crate) fn bf16_to_f32(input: &[u16], output: &mut [f32]) -> bool {
    if input.len() != output.len() {
        return false;
    }
    bf16_to_f32_for_target(input, output)
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn bf16_to_f32_for_target(_input: &[u16], _output: &mut [f32]) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn convert_f32_to_bf16(_input: &[f32], _output: &mut [u16]) -> bool {
    false
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn bounded_exp(value: f32) -> f32 {
    let value = value.clamp(-40.0, 20.0);
    let exponent = (value * core::f32::consts::LOG2_E).round() as i32;
    let remainder = (-exponent as f32).mul_add(core::f32::consts::LN_2, value);
    let polynomial = remainder.mul_add(
        remainder.mul_add(
            remainder.mul_add(
                remainder.mul_add(
                    remainder.mul_add(
                        remainder.mul_add(remainder.mul_add(1.0 / 5040.0, 1.0 / 720.0), 1.0 / 120.0),
                        1.0 / 24.0,
                    ),
                    1.0 / 6.0,
                ),
                0.5,
            ),
            1.0,
        ),
        1.0,
    );
    polynomial * f32::from_bits(u32::try_from(exponent + 127).expect("bounded exponent is positive") << 23)
}

#[inline]
pub(crate) fn fast_erf(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude >= 6.0 {
        return value.signum();
    }
    if magnitude < 1e-4 {
        return value * core::f32::consts::FRAC_2_SQRT_PI;
    }
    let t = (0.327_591_1_f32.mul_add(magnitude, 1.0)).recip();
    let polynomial = 1.061_405_4_f32
        .mul_add(t, -1.453_152_1)
        .mul_add(t, 1.421_413_8)
        .mul_add(t, -0.284_496_72)
        .mul_add(t, 0.254_829_6)
        * t;
    value.signum() * (1.0 - polynomial * bounded_exp(-magnitude * magnitude))
}

#[inline]
pub(crate) fn fast_tanh(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude >= 9.0 {
        return value.signum();
    }
    let result = if magnitude <= 0.5 {
        let square = magnitude * magnitude;
        magnitude
            * square.mul_add(
                square.mul_add(
                    square.mul_add(
                        square.mul_add(square.mul_add(-0.008_863_236, 0.021_869_488), -0.053_968_254),
                        0.133_333_34,
                    ),
                    -1.0 / 3.0,
                ),
                1.0,
            )
    } else {
        let exponential = bounded_exp(2.0 * magnitude);
        (exponential - 1.0) / (exponential + 1.0)
    };
    value.signum() * result
}

#[inline]
pub(crate) fn fast_gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + fast_erf(value / SQRT_2))
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn gelu_for_target(_input: &[f32], _output: &mut [f32]) -> bool {
    false
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn qass_modulate_for_target(
    _queries: &[f32],
    _base_scale: &[f32],
    _heads: usize,
    _rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}
