//! FP8 E4M3 conversion, in hardware with `ironicl_cap_fp8_cvt`.

#![allow(
    unsafe_code, // Shared memory, tensor-core intrinsics, and unchecked indexing.
    clippy::no_mangle_with_rust_abi, // How cuda-oxide's `#[kernel]` exports entries.
    clippy::too_many_arguments, // A kernel's parameters are its launch ABI.
    clippy::too_many_lines,
    clippy::many_single_char_names, // The math's names: q, k, v, m, n, k.
    clippy::similar_names, // Paired high and low parts: `w_hi`, `w_lo`.
    clippy::needless_range_loop, // `#[unroll]` loops over register-array indices.
    clippy::inline_always, // Device helpers inline into their kernels.
    clippy::cast_ptr_alignment, // Vector loads at offsets aligned by construction.
    clippy::items_after_statements, // `#[launch_bounds]` expands ahead of a kernel's shared statics.
)]

#[cfg(ironicl_cap_fp8_cvt)]
use cuda_device::convert::cvt_rn_f16x2_e4m3x2;
#[cfg(not(ironicl_cap_fp8_cvt))]
use cuda_device::f16x2::fma_f16x2;

/// Two FP8 E4M3 values, the first in the low byte, as packed FP16. Ada and
/// later convert in hardware; before them, each byte's exponent and mantissa
/// shift into FP16's place, which scales the value by 2^-8 (the exponent biases
/// differ by 8), and an exact FP16 multiply by 256 undoes that. E4M3
/// subnormals land on FP16 values exactly, so both ways give the same result.
#[inline(always)]
pub(crate) fn e4m3x2_to_f16x2(pair: u16) -> u32 {
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

/// `value` as FP8 E4M3 bits, rounded to nearest even and saturated at
/// ±448, as the host's `cache::e4m3` does.
#[inline(always)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Subnormal steps: 0..=8.
pub(crate) fn e4m3(value: f32) -> u32 {
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
