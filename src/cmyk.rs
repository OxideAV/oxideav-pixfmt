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
//! The conversion to RGB is an uncalibrated device-CMYK approximation
//! (no ICC profile) that matches the formula used by libjpeg,
//! ImageMagick's default, and most non-Adobe encoders:
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
//! # What this is NOT
//!
//! * **Not ICC-calibrated.** Real print workflows require an ICC
//!   profile (CMYK device → CIELAB → RGB device) to reproduce colours
//!   accurately. This module is the "screen-preview" approximation.
//! * **Not the Adobe/Photoshop inverted CMYK.** Photoshop stores JPEG
//!   CMYK with the components INVERTED (`255` = no ink); those files
//!   need a separate `CmykInverted` path that this module does not
//!   provide yet.
//! * **No YCCK.** JPEGs tagged with the Adobe APP14 marker often carry
//!   a YCbCrK colour transform that needs to be undone before this
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
}
