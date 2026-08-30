//! Scene-referred 32-bit float sample helpers (the `GrayF32Le` /
//! `RgbF32Le` / `RgbaF32Le` / `GbrpF32Le` / `GbrapF32Le` family of
//! oxideav-core 0.1.35).
//!
//! # Semantics
//!
//! Float samples are IEEE 754 binary32 little-endian words carrying
//! **linear light** with no integer full-scale: `1.0` is the nominal
//! diffuse-white anchor, values above `1.0` (speculars) and below `0.0`
//! (out-of-gamut excursions) are legal and are preserved verbatim by
//! every float → float move. Alpha is straight (non-premultiplied) with
//! nominal range `[0, 1]`.
//!
//! # Integer ↔ float rule
//!
//! The integer RGB / gray / GBR formats carry no transfer
//! characteristic of their own in the core format definitions, so the
//! hop between them and the float family is a **pure normalisation** —
//! no OETF / EOTF is applied in either direction (callers that need one
//! run [`crate::transfer`] on the float samples):
//!
//! - integer → float: `f = code / (2^bits − 1)`, so `0` → `0.0` and
//!   full-scale → exactly `1.0`;
//! - float → integer: `code = round(clamp(f, 0, 1) · (2^bits − 1))`
//!   (round half away from zero); NaN quantises to `0`, `+∞` to
//!   full-scale, `−∞` to `0`. Values outside `[0, 1]` are therefore
//!   **saturated** — the integer formats have no headroom to carry
//!   them — which is the one lossy step in the family.
//!
//! An integer → float → integer round trip at the same depth is exact
//! for every code (the normalisation is strictly monotonic and the
//! quantisation rounds to nearest); float → integer → float is exact
//! for the `2^bits` representable values.

use crate::yuv::YuvMatrix;

/// Read the little-endian binary32 sample at index `i` of `buf`.
#[inline]
pub fn read_f32le(buf: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]])
}

/// Write `v` as the little-endian binary32 sample at index `i` of `buf`.
#[inline]
pub fn write_f32le(buf: &mut [u8], i: usize, v: f32) {
    buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// Normalise an unsigned integer code of `bits` significant bits to
/// the float domain: `code / (2^bits − 1)`.
#[inline]
pub fn unorm_to_f32(code: u32, bits: u32) -> f32 {
    let full = ((1u64 << bits) - 1) as f32;
    code as f32 / full
}

/// Quantise a float sample to an unsigned integer code of `bits`
/// significant bits: saturate to `[0, 1]`, scale by `2^bits − 1`, round
/// to nearest (half away from zero). NaN maps to `0`.
#[inline]
pub fn f32_to_unorm(v: f32, bits: u32) -> u32 {
    let full = ((1u64 << bits) - 1) as f32;
    // `v.clamp(0.0, 1.0)` propagates NaN; the explicit comparisons
    // fold NaN into the low clamp instead.
    let c = if v >= 1.0 {
        1.0
    } else if v > 0.0 {
        v
    } else {
        0.0
    };
    (c * full).round() as u32
}

/// Linear-light luminance of an `(r, g, b)` triple under the primaries
/// selected by `matrix` (`Y = Kr·R + (1 − Kr − Kb)·G + Kb·B`). The
/// luma coefficients of the BT-series matrices are defined for linear
/// components, so on the float family this projection is exact rather
/// than the gamma-domain approximation the 8-bit `Gray8` rows apply.
///
/// A neutral triple (`r == g == b`, infinities included) returns its
/// value bit-exactly instead of picking up the rounding of three
/// weights that only sum to one in exact arithmetic.
#[inline]
pub fn luminance_linear(r: f32, g: f32, b: f32, matrix: YuvMatrix) -> f32 {
    if r == g && g == b {
        return g;
    }
    let kg = 1.0 - matrix.kr - matrix.kb;
    matrix.kr * r + kg * g + matrix.kb * b
}

/// Convert `count` byte samples (`bits == 8`) to binary32 LE samples.
pub fn plane_u8_to_f32le(src: &[u8], dst: &mut [u8], count: usize) {
    for (i, &code) in src.iter().enumerate().take(count) {
        write_f32le(dst, i, unorm_to_f32(code as u32, 8));
    }
}

/// Convert `count` LE 16-bit words carrying `bits` significant low
/// bits to binary32 LE samples.
pub fn plane_le16_to_f32le(src: &[u8], dst: &mut [u8], count: usize, bits: u32) {
    let mask = ((1u32 << bits) - 1) as u16;
    for i in 0..count {
        let code = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]) & mask;
        write_f32le(dst, i, unorm_to_f32(code as u32, bits));
    }
}

/// Quantise `count` binary32 LE samples to bytes (8 significant bits).
pub fn plane_f32le_to_u8(src: &[u8], dst: &mut [u8], count: usize) {
    for (i, out) in dst.iter_mut().enumerate().take(count) {
        *out = f32_to_unorm(read_f32le(src, i), 8) as u8;
    }
}

/// Quantise `count` binary32 LE samples to LE 16-bit words carrying
/// `bits` significant low bits.
pub fn plane_f32le_to_le16(src: &[u8], dst: &mut [u8], count: usize, bits: u32) {
    for i in 0..count {
        let code = f32_to_unorm(read_f32le(src, i), bits) as u16;
        dst[i * 2..i * 2 + 2].copy_from_slice(&code.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantise_rules() {
        assert_eq!(f32_to_unorm(0.0, 8), 0);
        assert_eq!(f32_to_unorm(1.0, 8), 255);
        assert_eq!(f32_to_unorm(1.5, 8), 255);
        assert_eq!(f32_to_unorm(-0.25, 8), 0);
        assert_eq!(f32_to_unorm(f32::NAN, 16), 0);
        assert_eq!(f32_to_unorm(f32::INFINITY, 16), 65535);
        assert_eq!(f32_to_unorm(f32::NEG_INFINITY, 10), 0);
        assert_eq!(f32_to_unorm(0.5, 8), 128); // 127.5 rounds away from zero
        assert_eq!(f32_to_unorm(0.5, 16), 32768);
    }

    #[test]
    fn every_code_round_trips_at_every_depth() {
        for bits in [8u32, 10, 12, 14, 16] {
            for code in 0..(1u32 << bits) {
                assert_eq!(f32_to_unorm(unorm_to_f32(code, bits), bits), code);
            }
        }
    }

    #[test]
    fn luminance_weights_sum_to_one() {
        for m in [YuvMatrix::BT601, YuvMatrix::BT709, YuvMatrix::BT2020] {
            // Neutral triples are returned bit-exactly.
            assert_eq!(luminance_linear(1.0, 1.0, 1.0, m), 1.0);
            assert_eq!(luminance_linear(0.5, 0.5, 0.5, m), 0.5);
            assert_eq!(luminance_linear(2.0, 2.0, 2.0, m), 2.0);
            let inf = f32::INFINITY;
            assert_eq!(luminance_linear(inf, inf, inf, m), inf);
            // Coloured: the Kr / Kg / Kb row.
            let y = luminance_linear(1.0, 0.0, 0.0, m);
            assert!((y - m.kr).abs() < 1e-6);
            let y = luminance_linear(0.0, 0.0, 1.0, m);
            assert!((y - m.kb).abs() < 1e-6);
            let y = luminance_linear(0.0, 1.0, 0.0, m);
            assert!((y - (1.0 - m.kr - m.kb)).abs() < 1e-6);
        }
    }
}
