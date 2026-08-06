//! CMYK ↔ RGB conversion.
//!
//! CMYK is the four-component "print" colour space — Cyan, Magenta,
//! Yellow, blacK. oxideav's [`oxideav_core::PixelFormat::Cmyk`] is the
//! "regular" encoding: each component is an 8-bit ink amount where
//! `0` means no ink (white for all four) and `255` means full ink.
//! Bytes are packed `C, M, Y, K` per pixel.
//!
//! # Formula
//!
//! The conversion to RGB is the standard uncalibrated device-CMYK
//! approximation (no ICC profile):
//!
//! ```text
//! R = (255 - C) · (255 - K) / 255
//! G = (255 - M) · (255 - K) / 255
//! B = (255 - Y) · (255 - K) / 255
//! ```
//!
//! Inverse (RGB → CMYK): compute `K = 255 - max(R, G, B)`, then
//! `C = (255 - R - K) · 255 / (255 - K)` (and similarly for M, Y),
//! with the degenerate case `K = 255` → `C = M = Y = 0`. Round-tripping
//! RGB → CMYK → RGB is lossless at 8-bit precision for every input.
//!
//! # Inverted-ink convention (`CmykInverted`)
//!
//! Some 4-component scans store ink coverage INVERTED on the wire
//! (`0` = full ink, `255` = no ink — see
//! [`oxideav_core::PixelFormat::CmykInverted`]). Because the inversion
//! is a per-byte complement, the inverted formula folds it away:
//!
//! ```text
//! R = C' · K' / 255        (C' = stored cyan byte, K' = stored black byte)
//! ```
//!
//! and the two conventions interconvert losslessly by complementing
//! every byte ([`cmyk_complement`], a self-inverse bijection). The
//! `*_inverted` entry points below produce byte-identical results to
//! complement-then-regular (pinned in the test suite).
//!
//! # What this is NOT
//!
//! * **Not ICC-calibrated.** Real print workflows require an ICC
//!   profile (CMYK device → CIELAB → RGB device) to reproduce colours
//!   accurately. This module is the "screen-preview" approximation.
//! * **No YCCK.** JPEGs whose adobe-convention APP14 segment signals a
//!   YCbCrK colour transform need that transform undone before this
//!   module is applied. That belongs in the JPEG decoder, not here.

/// Convert packed CMYK to packed RGB24.
///
/// Processes `pixels` pixels: reads `4 * pixels` input bytes, writes
/// `3 * pixels` output bytes. Panics in debug if the slices are
/// shorter than that.
pub fn cmyk_to_rgb24(src: &[u8], dst: &mut [u8], pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 3);
    for i in 0..pixels {
        let c = src[i * 4] as u32;
        let m = src[i * 4 + 1] as u32;
        let y = src[i * 4 + 2] as u32;
        let k = src[i * 4 + 3] as u32;
        let kc = 255 - k;
        // (255 - C) * (255 - K) / 255, done with a / 255 fast-path via
        // ((v * 0x8081) >> 23) style division. We do an explicit / 255
        // here; the compiler folds it to a multiply + shift.
        dst[i * 3] = (((255 - c) * kc) / 255) as u8;
        dst[i * 3 + 1] = (((255 - m) * kc) / 255) as u8;
        dst[i * 3 + 2] = (((255 - y) * kc) / 255) as u8;
    }
}

/// Convert packed CMYK to packed RGBA (opaque alpha of 255).
pub fn cmyk_to_rgba(src: &[u8], dst: &mut [u8], pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 4);
    for i in 0..pixels {
        let c = src[i * 4] as u32;
        let m = src[i * 4 + 1] as u32;
        let y = src[i * 4 + 2] as u32;
        let k = src[i * 4 + 3] as u32;
        let kc = 255 - k;
        dst[i * 4] = (((255 - c) * kc) / 255) as u8;
        dst[i * 4 + 1] = (((255 - m) * kc) / 255) as u8;
        dst[i * 4 + 2] = (((255 - y) * kc) / 255) as u8;
        dst[i * 4 + 3] = 255;
    }
}

/// Convert packed RGB24 to packed CMYK.
pub fn rgb24_to_cmyk(src: &[u8], dst: &mut [u8], pixels: usize) {
    debug_assert!(src.len() >= pixels * 3 && dst.len() >= pixels * 4);
    for i in 0..pixels {
        let r = src[i * 3] as u32;
        let g = src[i * 3 + 1] as u32;
        let b = src[i * 3 + 2] as u32;
        let k = 255 - r.max(g).max(b);
        if k == 255 {
            // Pure black: C / M / Y are indeterminate — zero them.
            dst[i * 4] = 0;
            dst[i * 4 + 1] = 0;
            dst[i * 4 + 2] = 0;
            dst[i * 4 + 3] = 255;
            continue;
        }
        let denom = 255 - k;
        dst[i * 4] = (((255 - r - k) * 255) / denom) as u8;
        dst[i * 4 + 1] = (((255 - g - k) * 255) / denom) as u8;
        dst[i * 4 + 2] = (((255 - b - k) * 255) / denom) as u8;
        dst[i * 4 + 3] = k as u8;
    }
}

/// Convert packed RGBA to packed CMYK (discards the alpha channel).
pub fn rgba_to_cmyk(src: &[u8], dst: &mut [u8], pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 4);
    for i in 0..pixels {
        let r = src[i * 4] as u32;
        let g = src[i * 4 + 1] as u32;
        let b = src[i * 4 + 2] as u32;
        let k = 255 - r.max(g).max(b);
        if k == 255 {
            dst[i * 4] = 0;
            dst[i * 4 + 1] = 0;
            dst[i * 4 + 2] = 0;
            dst[i * 4 + 3] = 255;
            continue;
        }
        let denom = 255 - k;
        dst[i * 4] = (((255 - r - k) * 255) / denom) as u8;
        dst[i * 4 + 1] = (((255 - g - k) * 255) / denom) as u8;
        dst[i * 4 + 2] = (((255 - b - k) * 255) / denom) as u8;
        dst[i * 4 + 3] = k as u8;
    }
}

// ---------------------------------------------------------------------
// Inverted-ink convention (`CmykInverted`): stored byte = 255 − ink.

/// Complement every byte of a packed 4-component buffer — the exact,
/// self-inverse bijection between the regular and inverted CMYK
/// conventions (`Cmyk` ↔ `CmykInverted`). Also serves alpha-less
/// 4-byte layouts generally; here it is only registered for the CMYK
/// pair.
pub fn cmyk_complement(src: &[u8], dst: &mut [u8], pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 4);
    for (d, &s) in dst[..pixels * 4].iter_mut().zip(src[..pixels * 4].iter()) {
        *d = !s;
    }
}

/// Convert packed inverted CMYK to packed RGB24. With the complement
/// folded in, the formula is `R = C' · K' / 255` (and likewise for
/// G / B) — byte-identical to complementing first and running
/// [`cmyk_to_rgb24`].
pub fn cmyk_inverted_to_rgb24(src: &[u8], dst: &mut [u8], pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 3);
    for i in 0..pixels {
        let c = src[i * 4] as u32;
        let m = src[i * 4 + 1] as u32;
        let y = src[i * 4 + 2] as u32;
        let k = src[i * 4 + 3] as u32;
        dst[i * 3] = ((c * k) / 255) as u8;
        dst[i * 3 + 1] = ((m * k) / 255) as u8;
        dst[i * 3 + 2] = ((y * k) / 255) as u8;
    }
}

/// Convert packed inverted CMYK to packed RGBA (opaque alpha of 255).
pub fn cmyk_inverted_to_rgba(src: &[u8], dst: &mut [u8], pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 4);
    for i in 0..pixels {
        let c = src[i * 4] as u32;
        let m = src[i * 4 + 1] as u32;
        let y = src[i * 4 + 2] as u32;
        let k = src[i * 4 + 3] as u32;
        dst[i * 4] = ((c * k) / 255) as u8;
        dst[i * 4 + 1] = ((m * k) / 255) as u8;
        dst[i * 4 + 2] = ((y * k) / 255) as u8;
        dst[i * 4 + 3] = 255;
    }
}

/// Convert packed RGB24 to packed inverted CMYK: the regular
/// separation followed by the byte complement, so
/// RGB → inverted-CMYK → RGB round-trips losslessly at 8-bit
/// precision exactly like the regular convention.
pub fn rgb24_to_cmyk_inverted(src: &[u8], dst: &mut [u8], pixels: usize) {
    rgb24_to_cmyk(src, dst, pixels);
    for b in dst[..pixels * 4].iter_mut() {
        *b = !*b;
    }
}

/// Convert packed RGBA to packed inverted CMYK (discards the alpha
/// channel).
pub fn rgba_to_cmyk_inverted(src: &[u8], dst: &mut [u8], pixels: usize) {
    rgba_to_cmyk(src, dst, pixels);
    for b in dst[..pixels * 4].iter_mut() {
        *b = !*b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_black() {
        // CMYK (0, 0, 0, 255) → RGB (0, 0, 0).
        let src = [0u8, 0, 0, 255];
        let mut dst = [0u8; 3];
        cmyk_to_rgb24(&src, &mut dst, 1);
        assert_eq!(dst, [0, 0, 0]);
    }

    #[test]
    fn pure_white() {
        // CMYK (0, 0, 0, 0) → RGB (255, 255, 255).
        let src = [0u8, 0, 0, 0];
        let mut dst = [0u8; 3];
        cmyk_to_rgb24(&src, &mut dst, 1);
        assert_eq!(dst, [255, 255, 255]);
    }

    #[test]
    fn pure_cyan() {
        // CMYK (255, 0, 0, 0) → RGB (0, 255, 255).
        let src = [255u8, 0, 0, 0];
        let mut dst = [0u8; 3];
        cmyk_to_rgb24(&src, &mut dst, 1);
        assert_eq!(dst, [0, 255, 255]);
    }

    #[test]
    fn pure_magenta() {
        let src = [0u8, 255, 0, 0];
        let mut dst = [0u8; 3];
        cmyk_to_rgb24(&src, &mut dst, 1);
        assert_eq!(dst, [255, 0, 255]);
    }

    #[test]
    fn pure_yellow() {
        let src = [0u8, 0, 255, 0];
        let mut dst = [0u8; 3];
        cmyk_to_rgb24(&src, &mut dst, 1);
        assert_eq!(dst, [255, 255, 0]);
    }

    #[test]
    fn rgb_to_cmyk_basics() {
        // White → (0, 0, 0, 0).
        let src = [255u8, 255, 255];
        let mut dst = [0u8; 4];
        rgb24_to_cmyk(&src, &mut dst, 1);
        assert_eq!(dst, [0, 0, 0, 0]);

        // Black → (0, 0, 0, 255) per the degenerate-branch rule.
        let src = [0u8, 0, 0];
        let mut dst = [0u8; 4];
        rgb24_to_cmyk(&src, &mut dst, 1);
        assert_eq!(dst, [0, 0, 0, 255]);

        // Pure red (255, 0, 0) → (0, 255, 255, 0).
        let src = [255u8, 0, 0];
        let mut dst = [0u8; 4];
        rgb24_to_cmyk(&src, &mut dst, 1);
        assert_eq!(dst, [0, 255, 255, 0]);
    }

    #[test]
    fn roundtrip_rgb_cmyk_rgb() {
        // Sweep a handful of saturated and mixed colours; every pixel
        // should round-trip losslessly at 8-bit precision.
        let colours = [
            [0u8, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
            [50, 200, 100],
            [17, 34, 51],
            [240, 15, 5],
        ];
        for rgb in &colours {
            let mut cmyk = [0u8; 4];
            rgb24_to_cmyk(rgb, &mut cmyk, 1);
            let mut back = [0u8; 3];
            cmyk_to_rgb24(&cmyk, &mut back, 1);
            assert_eq!(&back, rgb, "round-trip failed for {rgb:?} via {cmyk:?}");
        }
    }

    #[test]
    fn rgba_variant_matches_rgb24() {
        let src_rgba = [200u8, 100, 50, 255];
        let src_rgb = [200u8, 100, 50];
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        rgba_to_cmyk(&src_rgba, &mut a, 1);
        rgb24_to_cmyk(&src_rgb, &mut b, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn cmyk_to_rgba_always_opaque() {
        let src = [40u8, 0, 100, 30];
        let mut rgba = [0u8; 4];
        cmyk_to_rgba(&src, &mut rgba, 1);
        assert_eq!(rgba[3], 255);
    }

    #[test]
    fn complement_is_self_inverse_bijection() {
        let src: Vec<u8> = (0..=255u8).chain(0..=255).collect();
        let px = src.len() / 4;
        let mut inv = vec![0u8; src.len()];
        cmyk_complement(&src, &mut inv, px);
        for (a, b) in src.iter().zip(inv.iter()) {
            assert_eq!(*a, !*b);
        }
        let mut back = vec![0u8; src.len()];
        cmyk_complement(&inv, &mut back, px);
        assert_eq!(back, src);
    }

    #[test]
    fn inverted_decode_equals_complement_then_regular() {
        // Sweep a spread of inverted-CMYK quads: the folded-in formula
        // must be byte-identical to complementing first and running
        // the regular decode.
        for seed in 0..64u32 {
            let q = [
                (seed * 37 + 1) as u8,
                (seed * 91 + 5) as u8,
                (seed * 53 + 11) as u8,
                (seed * 17 + 3) as u8,
            ];
            let mut direct = [0u8; 3];
            cmyk_inverted_to_rgb24(&q, &mut direct, 1);
            let comp: Vec<u8> = q.iter().map(|&b| !b).collect();
            let mut staged = [0u8; 3];
            cmyk_to_rgb24(&comp, &mut staged, 1);
            assert_eq!(direct, staged.as_slice(), "quad {q:?}");
            // RGBA variant shares the colour math and is opaque.
            let mut rgba = [0u8; 4];
            cmyk_inverted_to_rgba(&q, &mut rgba, 1);
            assert_eq!(&rgba[..3], &direct);
            assert_eq!(rgba[3], 255);
        }
    }

    #[test]
    fn inverted_anchor_values() {
        // Inverted white: no ink stored as all-255 → RGB white.
        let mut dst = [0u8; 3];
        cmyk_inverted_to_rgb24(&[255, 255, 255, 255], &mut dst, 1);
        assert_eq!(dst, [255, 255, 255]);
        // Inverted full black ink: K' = 0 → RGB black.
        cmyk_inverted_to_rgb24(&[255, 255, 255, 0], &mut dst, 1);
        assert_eq!(dst, [0, 0, 0]);
        // Inverted pure cyan: C' = 0, K' = 255 → (0, 255, 255).
        cmyk_inverted_to_rgb24(&[0, 255, 255, 255], &mut dst, 1);
        assert_eq!(dst, [0, 255, 255]);
    }

    #[test]
    fn roundtrip_rgb_inverted_cmyk_rgb() {
        // Same lossless-at-8-bit guarantee as the regular convention —
        // the complement is a bijection, so it cannot cost precision.
        let colours = [
            [0u8, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
            [50, 200, 100],
            [17, 34, 51],
            [240, 15, 5],
        ];
        for rgb in &colours {
            let mut q = [0u8; 4];
            rgb24_to_cmyk_inverted(rgb, &mut q, 1);
            let mut back = [0u8; 3];
            cmyk_inverted_to_rgb24(&q, &mut back, 1);
            assert_eq!(&back, rgb, "round-trip failed for {rgb:?} via {q:?}");
        }
        // RGBA separation matches the RGB24 one.
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        rgba_to_cmyk_inverted(&[200, 100, 50, 77], &mut a, 1);
        rgb24_to_cmyk_inverted(&[200, 100, 50], &mut b, 1);
        assert_eq!(a, b);
    }
}
