//! `colorspace::ColorModel` round trips (the issue #15 acceptance: ±1
//! LSB at 16 bits with float storage of the model values), the
//! ISO 32000-2 CMYK rules, and spot values from the staged formulas.

use oxideav_pixfmt::colorspace::{
    ColorModel, LabWhite, LumaWeights, cmyk_to_rgb, delta_e2000, linear_to_srgb, rgb_to_cmyk,
    rgb_to_cmyk_with, srgb_to_linear,
};

fn models() -> Vec<ColorModel> {
    let mut v = vec![
        ColorModel::Srgb,
        ColorModel::LinearSrgb,
        ColorModel::XyzD65,
        ColorModel::XyzD50,
        ColorModel::XyY,
        ColorModel::Lab(LabWhite::D65),
        ColorModel::Lab(LabWhite::D50),
        ColorModel::Lch(LabWhite::D65),
        ColorModel::Lch(LabWhite::D50),
        ColorModel::Oklab,
        ColorModel::Oklch,
        ColorModel::Hsl,
        ColorModel::Hwb,
        ColorModel::Cmy,
    ];
    for w in [LumaWeights::BT601, LumaWeights::BT709, LumaWeights::BT2020] {
        v.push(ColorModel::YPbPr(w));
        v.push(ColorModel::YCbCr(w));
    }
    v
}

fn probes() -> Vec<[u16; 3]> {
    let mut v = Vec::new();
    // Miri interprets every float op; keep its sweep small.
    let (steps, random) = if cfg!(miri) { (2u32, 16) } else { (16, 20000) };
    let lv: Vec<u16> = (0..=steps).map(|k| (k * 65535 / steps) as u16).collect();
    for &r in &lv {
        for &g in &lv {
            for &b in &lv {
                v.push([r, g, b]);
            }
        }
    }
    let mut s: u32 = 0x9E37_79B9;
    for _ in 0..random {
        let mut px = [0u16; 3];
        for c in &mut px {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *c = (s >> 16) as u16;
        }
        v.push(px);
    }
    // Near-black and near-white, where the curves bend.
    for k in 0..if cfg!(miri) { 4u16 } else { 64 } {
        v.push([k, k * 3, 65535 - k]);
        v.push([65535 - k, k, 0]);
    }
    v
}

#[test]
fn every_model_round_trips_within_one_lsb_at_16_bits() {
    for m in models() {
        let mut worst = 0i32;
        for px in probes() {
            let rgb = px.map(|c| c as f64 / 65535.0);
            // Imagick's Q16-HDRI keeps the model values as floats.
            let stored = m.from_srgb(rgb).map(|v| v as f32 as f64);
            let back = m.to_srgb(stored);
            for c in 0..3 {
                let q = (back[c].clamp(0.0, 1.0) * 65535.0).round() as i32;
                worst = worst.max((q - px[c] as i32).abs());
            }
        }
        assert!(worst <= 1, "{m:?}: {worst} LSB");
    }
}

#[test]
fn gray_models_are_luma() {
    let rgb = [0.2, 0.4, 0.8];
    let g = ColorModel::Gray(LumaWeights::BT709).from_srgb(rgb);
    assert!((g[0] - (0.2126 * 0.2 + 0.7152 * 0.4 + 0.0722 * 0.8)).abs() < 1e-12);
    let lg = ColorModel::LinearGray(LumaWeights::BT709).from_srgb(rgb);
    let lin = rgb.map(srgb_to_linear);
    assert!((lg[0] - (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])).abs() < 1e-12);
    // ISO 32000-2 §10.4.2.2 device gray.
    assert!(
        (LumaWeights::PDF_DEVICE_GRAY.luma(rgb) - (0.3 * 0.2 + 0.59 * 0.4 + 0.11 * 0.8)).abs()
            < 1e-12
    );
    let back = ColorModel::LinearGray(LumaWeights::BT709).to_srgb(lg);
    assert!((back[0] - linear_to_srgb(lg[0])).abs() < 1e-12);
}

#[test]
fn pdf_device_cmyk_rules() {
    // §10.4.2.4 EXAMPLE: 0.2 red, 0.7 green, 0.4 blue → 0.8 / 0.3 / 0.6.
    let cmy = ColorModel::Cmy.from_srgb([0.2, 0.7, 0.4]);
    for (got, want) in cmy.iter().zip([0.8, 0.3, 0.6]) {
        assert!((got - want).abs() < 1e-12);
    }
    // Full BG / UCR: k = min(c, m, y) = 0.3.
    let k = rgb_to_cmyk([0.2, 0.7, 0.4]);
    for (got, want) in k.iter().zip([0.5, 0.0, 0.3, 0.3]) {
        assert!((got - want).abs() < 1e-12, "{k:?}");
    }
    // §10.4.2.5 inverts full UCR exactly.
    let back = cmyk_to_rgb(k);
    for (got, want) in back.iter().zip([0.2, 0.7, 0.4]) {
        assert!((got - want).abs() < 1e-12);
    }
    // No black generation / no undercolour removal = CMY with K = 0.
    let none = rgb_to_cmyk_with([0.2, 0.7, 0.4], &|_| 0.0, &|_| 0.0);
    assert_eq!(none[3], 0.0);
    assert!((none[0] - 0.8).abs() < 1e-12);
    // Half GCR; results clamp to [0, 1].
    let half = rgb_to_cmyk_with([0.0, 0.0, 0.0], &|k| k * 0.5, &|k| k * 0.5);
    for (got, want) in half.iter().zip([0.5, 0.5, 0.5, 0.5]) {
        assert!((got - want).abs() < 1e-12);
    }
}

#[test]
fn css_color_4_reference_points() {
    // HSL / HWB of pure red and of a mid colour (CSS Color 4 §7 / §8).
    assert_eq!(ColorModel::Hsl.from_srgb([1.0, 0.0, 0.0]), [0.0, 1.0, 0.5]);
    let h = ColorModel::Hsl.from_srgb([0.2, 0.4, 0.8]);
    assert!((h[0] - 220.0).abs() < 1e-9 && (h[1] - 0.6).abs() < 1e-9 && (h[2] - 0.5).abs() < 1e-9);
    let w = ColorModel::Hwb.from_srgb([0.2, 0.4, 0.8]);
    assert!((w[0] - 220.0).abs() < 1e-9 && (w[1] - 0.2).abs() < 1e-9 && (w[2] - 0.2).abs() < 1e-9);
    // HWB with W + B ≥ 1 is gray W / (W + B).
    assert_eq!(ColorModel::Hwb.to_srgb([123.0, 0.6, 0.6]), [0.5, 0.5, 0.5]);
    // White is L = 100 / Oklab L = 1 with zero chroma.
    for m in [
        ColorModel::Lab(LabWhite::D65),
        ColorModel::Lab(LabWhite::D50),
    ] {
        let l = m.from_srgb([1.0, 1.0, 1.0]);
        assert!(
            (l[0] - 100.0).abs() < 1e-9 && l[1].abs() < 1e-9 && l[2].abs() < 1e-9,
            "{m:?} {l:?}"
        );
    }
    let ok = ColorModel::Oklab.from_srgb([1.0, 1.0, 1.0]);
    assert!(
        (ok[0] - 1.0).abs() < 1e-6 && ok[1].abs() < 1e-6 && ok[2].abs() < 1e-6,
        "{ok:?}"
    );
    // xyY of white is the D65 chromaticity.
    let xyy = ColorModel::XyY.from_srgb([1.0, 1.0, 1.0]);
    assert!(
        (xyy[0] - 0.3127).abs() < 1e-9
            && (xyy[1] - 0.3290).abs() < 1e-9
            && (xyy[2] - 1.0).abs() < 1e-9
    );
}

#[test]
fn ciede2000_basic_properties() {
    let a = [50.0, 2.6772, -79.7751];
    assert_eq!(delta_e2000(a, a), 0.0);
    // Pure lightness difference on the neutral axis: ΔL / S_L with
    // L̄ = 55 → S_L = 1 + 0.015·25 / √45.
    let d = delta_e2000([50.0, 0.0, 0.0], [60.0, 0.0, 0.0]);
    let sl = 1.0 + 0.015 * 25.0 / 45f64.sqrt();
    assert!((d - 10.0 / sl).abs() < 1e-12);
    // Symmetric in its arguments for these inputs.
    let b = [55.0, 10.0, 20.0];
    assert!((delta_e2000(a, b) - delta_e2000(b, a)).abs() < 1e-9);
}
