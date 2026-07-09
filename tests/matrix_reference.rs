//! Colour-matrix correctness against an independent f64 reference model.
//!
//! The crate's hot paths run in Q15 fixed point. These tests rebuild the
//! Y'CbCr construction in plain f64 directly from the k-coefficient
//! definitions (BT.601-7 / BT.709-6 / BT.2020-2 Table 4 NCL, staged in
//! `docs/video/signal-metadata/`) and the 8-bit quantisation rules
//! (limited range: Y' = 16 + 219·E'_Y, C' = 128 + 224·E'_C; full range:
//! the identity 0..=255 mapping), then require the shipped fixed-point
//! converters to match within ±1 code in every channel over a dense RGB
//! / YCbCr sweep — for all six `ColorSpace` variants, encode and decode.
//!
//! On top of the property sweep, classic anchor codes for the BT.601 and
//! BT.709 limited-range primaries are pinned as literals.

use oxideav_pixfmt::yuv::{rgb_to_yuv, yuv_to_rgb, YuvMatrix};

struct RefMatrix {
    kr: f64,
    kb: f64,
    limited: bool,
}

impl RefMatrix {
    fn of(m: &YuvMatrix) -> Self {
        Self {
            kr: m.kr as f64,
            kb: m.kb as f64,
            limited: m.limited,
        }
    }

    /// f64 encode: 8-bit R'G'B' → 8-bit Y'CbCr codes (rounded, clamped).
    fn encode(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let kr = self.kr;
        let kb = self.kb;
        let kg = 1.0 - kr - kb;
        let rf = r as f64 / 255.0;
        let gf = g as f64 / 255.0;
        let bf = b as f64 / 255.0;
        let ey = kr * rf + kg * gf + kb * bf; // E'_Y in 0..=1
        let ecb = (bf - ey) / (2.0 * (1.0 - kb)); // E'_CB in -0.5..=0.5
        let ecr = (rf - ey) / (2.0 * (1.0 - kr));
        let (y, cb, cr) = if self.limited {
            (16.0 + 219.0 * ey, 128.0 + 224.0 * ecb, 128.0 + 224.0 * ecr)
        } else {
            (255.0 * ey, 128.0 + 255.0 * ecb, 128.0 + 255.0 * ecr)
        };
        (round_clamp(y), round_clamp(cb), round_clamp(cr))
    }

    /// f64 decode: 8-bit Y'CbCr codes → 8-bit R'G'B' (rounded, clamped).
    fn decode(&self, y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
        let kr = self.kr;
        let kb = self.kb;
        let kg = 1.0 - kr - kb;
        let (ey, ecb, ecr) = if self.limited {
            (
                (y as f64 - 16.0) / 219.0,
                (cb as f64 - 128.0) / 224.0,
                (cr as f64 - 128.0) / 224.0,
            )
        } else {
            (
                y as f64 / 255.0,
                (cb as f64 - 128.0) / 255.0,
                (cr as f64 - 128.0) / 255.0,
            )
        };
        let rf = ey + 2.0 * (1.0 - kr) * ecr;
        let bf = ey + 2.0 * (1.0 - kb) * ecb;
        let gf = ey - (2.0 * kr * (1.0 - kr) / kg) * ecr - (2.0 * kb * (1.0 - kb) / kg) * ecb;
        (
            round_clamp(rf * 255.0),
            round_clamp(gf * 255.0),
            round_clamp(bf * 255.0),
        )
    }
}

fn round_clamp(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn all_matrices() -> Vec<(&'static str, YuvMatrix)> {
    vec![
        ("BT.601 limited", YuvMatrix::BT601),
        ("BT.601 full", YuvMatrix::BT601.with_range(false)),
        ("BT.709 limited", YuvMatrix::BT709),
        ("BT.709 full", YuvMatrix::BT709.with_range(false)),
        ("BT.2020 limited", YuvMatrix::BT2020),
        ("BT.2020 full", YuvMatrix::BT2020.with_range(false)),
    ]
}

// Keep the sweep light under miri (the interpreter is ~3 orders of
// magnitude slower); native runs use a dense grid.
#[cfg(miri)]
const STEP: usize = 85;
#[cfg(not(miri))]
const STEP: usize = 15;

/// Fixed-point encode matches the f64 model within ±1 code everywhere.
#[test]
fn encode_matches_f64_reference_within_one() {
    for (name, m) in all_matrices() {
        let refm = RefMatrix::of(&m);
        for r in (0..=255).step_by(STEP) {
            for g in (0..=255).step_by(STEP) {
                for b in (0..=255).step_by(STEP) {
                    let (r, g, b) = (r as u8, g as u8, b as u8);
                    let got = rgb_to_yuv(r, g, b, m);
                    let want = refm.encode(r, g, b);
                    for (i, (gv, wv)) in [(got.0, want.0), (got.1, want.1), (got.2, want.2)]
                        .iter()
                        .enumerate()
                    {
                        assert!(
                            (*gv as i32 - *wv as i32).abs() <= 1,
                            "{name} encode({r},{g},{b}) ch{i}: fixed {gv} vs f64 {wv}"
                        );
                    }
                }
            }
        }
    }
}

/// Fixed-point decode matches the f64 model within ±1 code everywhere
/// (including out-of-range codes, which both sides clamp).
#[test]
fn decode_matches_f64_reference_within_one() {
    for (name, m) in all_matrices() {
        let refm = RefMatrix::of(&m);
        for y in (0..=255).step_by(STEP) {
            for cb in (0..=255).step_by(STEP) {
                for cr in (0..=255).step_by(STEP) {
                    let (y, cb, cr) = (y as u8, cb as u8, cr as u8);
                    let got = yuv_to_rgb(y, cb, cr, m);
                    let want = refm.decode(y, cb, cr);
                    for (i, (gv, wv)) in [(got.0, want.0), (got.1, want.1), (got.2, want.2)]
                        .iter()
                        .enumerate()
                    {
                        assert!(
                            (*gv as i32 - *wv as i32).abs() <= 1,
                            "{name} decode({y},{cb},{cr}) ch{i}: fixed {gv} vs f64 {wv}"
                        );
                    }
                }
            }
        }
    }
}

/// Classic BT.601 limited-range primary codes, hand-derived from the
/// §2.5 construction (kr = 0.299, kb = 0.114; Y' = 16 + 219·E'_Y,
/// C' = 128 + 224·E'_C):
///   red   (255, 0, 0) → ( 81,  90, 240)   [E'_CR = 0.5 exactly → 240]
///   green (0, 255, 0) → (145,  54,  34)
///   blue  (0, 0, 255) → ( 41, 240, 110)   [E'_CB = 0.5 exactly → 240]
/// plus the range rails: black → (16, 128, 128), white → (235, 128, 128).
#[test]
fn bt601_limited_primary_anchors() {
    let m = YuvMatrix::BT601;
    assert_eq!(rgb_to_yuv(0, 0, 0, m), (16, 128, 128));
    assert_eq!(rgb_to_yuv(255, 255, 255, m), (235, 128, 128));
    assert_eq!(rgb_to_yuv(255, 0, 0, m), (81, 90, 240));
    assert_eq!(rgb_to_yuv(0, 255, 0, m), (145, 54, 34));
    assert_eq!(rgb_to_yuv(0, 0, 255, m), (41, 240, 110));
}

/// BT.709 limited-range primary codes from the same construction with
/// kr = 0.2126, kb = 0.0722:
///   red   → ( 63, 102, 240)
///   green → (173,  42,  26)
///   blue  → ( 32, 240, 118)
#[test]
fn bt709_limited_primary_anchors() {
    let m = YuvMatrix::BT709;
    assert_eq!(rgb_to_yuv(0, 0, 0, m), (16, 128, 128));
    assert_eq!(rgb_to_yuv(255, 255, 255, m), (235, 128, 128));
    assert_eq!(rgb_to_yuv(255, 0, 0, m), (63, 102, 240));
    assert_eq!(rgb_to_yuv(0, 255, 0, m), (173, 42, 26));
    assert_eq!(rgb_to_yuv(0, 0, 255, m), (32, 240, 118));
}

/// Full-range rails: black → (0, 128, 128), white → (255, 128, 128) for
/// every primaries choice, and gray codes carry straight through
/// (Y' = v, chroma neutral) since the luma weights sum to one.
#[test]
fn full_range_rails_and_gray_identity() {
    for (name, m) in all_matrices() {
        let m = m.with_range(false);
        assert_eq!(rgb_to_yuv(0, 0, 0, m), (0, 128, 128), "{name}");
        assert_eq!(rgb_to_yuv(255, 255, 255, m), (255, 128, 128), "{name}");
        for v in (0..=255).step_by(5) {
            let v = v as u8;
            assert_eq!(rgb_to_yuv(v, v, v, m), (v, 128, 128), "{name} gray {v}");
            assert_eq!(yuv_to_rgb(v, 128, 128, m), (v, v, v), "{name} gray {v}");
        }
    }
}

/// Encode → decode is the identity within ±2 codes on the sweep for
/// every limited-range space (each direction rounds once), except where
/// the encoder itself clipped (saturated full-range chroma) — those
/// points are excluded by construction here since limited-range codes
/// never clip for in-gamut RGB.
#[test]
fn limited_roundtrip_within_two() {
    for (name, m) in [
        ("BT.601", YuvMatrix::BT601),
        ("BT.709", YuvMatrix::BT709),
        ("BT.2020", YuvMatrix::BT2020),
    ] {
        for r in (0..=255).step_by(STEP) {
            for g in (0..=255).step_by(STEP) {
                for b in (0..=255).step_by(STEP) {
                    let (r, g, b) = (r as u8, g as u8, b as u8);
                    let (y, cb, cr) = rgb_to_yuv(r, g, b, m);
                    let (r2, g2, b2) = yuv_to_rgb(y, cb, cr, m);
                    for (a, c) in [(r, r2), (g, g2), (b, b2)] {
                        assert!(
                            (a as i32 - c as i32).abs() <= 2,
                            "{name} roundtrip ({r},{g},{b}) → ({y},{cb},{cr}) → ({r2},{g2},{b2})"
                        );
                    }
                }
            }
        }
    }
}
