//! `colorspace::ColorModel` round trips (the issue #15 acceptance: ±1
//! LSB at 16 bits with float storage of the model values), the
//! ISO 32000-2 CMYK rules, and spot values from the staged formulas.

use oxideav_pixfmt::colorspace::{
    CineonLog, ColorModel, ConeResponse, D65, LabWhite, LumaWeights, cmyk_to_rgb, delta_e2000,
    jzazbz_to_xyz_abs, linear_to_srgb, luv_to_xyz, rgb_to_cmyk, rgb_to_cmyk_with, srgb_to_linear,
    xyz_abs_to_jzazbz, xyz_to_luv,
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
        ColorModel::Hsv,
        ColorModel::Hsi,
        ColorModel::Yiq,
        ColorModel::Yuv,
        ColorModel::YDbDr,
        ColorModel::Ohta,
        ColorModel::Lms,
        ColorModel::LmsOf(ConeResponse::Bradford),
        ColorModel::LmsOf(ConeResponse::HuntPointerEstevez),
        ColorModel::ScRgb,
        ColorModel::Luv(LabWhite::D65),
        ColorModel::Luv(LabWhite::D50),
        ColorModel::Lchuv(LabWhite::D65),
        ColorModel::Lchuv(LabWhite::D50),
        ColorModel::HclAb(LabWhite::D65),
        ColorModel::HclUv(LabWhite::D65),
        ColorModel::Jzazbz(ColorModel::JZ_SDR_WHITE),
        ColorModel::Jzazbz(ColorModel::JZ_HDR_REFERENCE_WHITE),
        ColorModel::JzCzhz(ColorModel::JZ_SDR_WHITE),
        ColorModel::PhotoYcc,
        ColorModel::Log(CineonLog::new()),
    ];
    for w in [LumaWeights::BT601, LumaWeights::BT709, LumaWeights::BT2020] {
        v.push(ColorModel::YPbPr(w));
        v.push(ColorModel::YCbCr(w));
        v.push(ColorModel::YCbCrLimited(w));
        v.push(ColorModel::Hcl(w));
        v.push(ColorModel::Hclp(w));
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
    // Not `== 0.0`: Miri perturbs transcendental results by a few ULP.
    assert!(delta_e2000(a, a).abs() < 1e-12);
    // Pure lightness difference on the neutral axis: ΔL / S_L with
    // L̄ = 55 → S_L = 1 + 0.015·25 / √45.
    let d = delta_e2000([50.0, 0.0, 0.0], [60.0, 0.0, 0.0]);
    let sl = 1.0 + 0.015 * 25.0 / 45f64.sqrt();
    assert!((d - 10.0 / sl).abs() < 1e-12);
    // Symmetric in its arguments for these inputs.
    let b = [55.0, 10.0, 20.0];
    assert!((delta_e2000(a, b) - delta_e2000(b, a)).abs() < 1e-9);
}

#[test]
fn hclp_pulls_out_of_gamut_values_toward_grey() {
    let w = LumaWeights::BT601;
    // In gamut: identical to HCL.
    let v = ColorModel::Hcl(w).from_srgb([0.2, 0.5, 0.7]);
    assert_eq!(
        ColorModel::Hclp(w).to_srgb(v),
        ColorModel::Hcl(w).to_srgb(v)
    );
    // Out of gamut: hue and luma kept, chroma reduced, inside 0..1.
    for hcl in [[0.0, 1.0, 0.9], [120.0, 1.0, 0.1], [250.0, 0.9, 0.5]] {
        let rgb = ColorModel::Hclp(w).to_srgb(hcl);
        assert!(rgb.iter().all(|c| (0.0..=1.0).contains(c)), "{rgb:?}");
        let back = ColorModel::Hcl(w).from_srgb(rgb);
        assert!((back[2] - hcl[2]).abs() < 1e-12, "luma {back:?}");
        assert!(back[1] <= hcl[1] + 1e-12, "chroma {back:?}");
        let dh = (back[0] - hcl[0]).rem_euclid(360.0);
        assert!(dh.min(360.0 - dh) < 1e-9, "hue {back:?}");
        // At least one component sits on the gamut boundary.
        assert!(rgb.iter().any(|&c| c == 0.0 || c == 1.0), "{rgb:?}");
    }
}

#[test]
fn hue_model_reference_points() {
    // HSV of pure colours and greys.
    assert_eq!(ColorModel::Hsv.from_srgb([1.0, 0.0, 0.0]), [0.0, 1.0, 1.0]);
    assert_eq!(ColorModel::Hsv.from_srgb([0.5, 0.5, 0.5]), [0.0, 0.0, 0.5]);
    let hsv = ColorModel::Hsv.from_srgb([0.2, 0.4, 0.6]);
    assert!((hsv[0] - 210.0).abs() < 1e-9 && (hsv[1] - 2.0 / 3.0).abs() < 1e-12);
    // HSI: the primaries sit 120° apart with I = 1/3.
    for (rgb, h) in [
        ([1.0, 0.0, 0.0], 0.0),
        ([0.0, 1.0, 0.0], 120.0),
        ([0.0, 0.0, 1.0], 240.0),
    ] {
        let v = ColorModel::Hsi.from_srgb(rgb);
        assert!(
            (v[0] - h).abs() < 1e-9 && (v[1] - 1.0).abs() < 1e-12,
            "{v:?}"
        );
        assert!((v[2] - 1.0 / 3.0).abs() < 1e-12);
    }
    // Ohta: I1 is the mean, I2 / I3 vanish on greys.
    let o = ColorModel::Ohta.from_srgb([0.3, 0.3, 0.3]);
    assert!((o[0] - 0.3).abs() < 1e-12 && o[1].abs() < 1e-12 && o[2].abs() < 1e-12);
    // Y′IQ / Y′UV / Y′DbDr: greys carry no chroma.
    for m in [ColorModel::Yiq, ColorModel::Yuv, ColorModel::YDbDr] {
        let v = m.from_srgb([0.7, 0.7, 0.7]);
        assert!(
            (v[0] - 0.7).abs() < 1e-6 && v[1].abs() < 2e-3 && v[2].abs() < 2e-3,
            "{m:?} {v:?}"
        );
    }
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
}

const WHITE: [f64; 3] = [1.0, 1.0, 1.0];
const BLACK: [f64; 3] = [0.0, 0.0, 0.0];
const RED: [f64; 3] = [1.0, 0.0, 0.0];
const BLUE: [f64; 3] = [0.0, 0.0, 1.0];

#[test]
fn cieluv_reference_points() {
    let m = ColorModel::Luv(LabWhite::D65);
    assert!(close(m.from_srgb(WHITE), [100.0, 0.0, 0.0], 1e-9));
    assert!(close(m.from_srgb(BLACK), [0.0, 0.0, 0.0], 1e-12));
    // sRGB red: Y = 0.2126 → L* = 116·0.2126^(1/3) − 16; u′ = 4x / (−2x
    // + 12y + 3) at the red primary (0.64, 0.33), against the white's.
    let l = 116.0 * 0.212_639f64.cbrt() - 16.0;
    let ucs = |x: f64, y: f64| {
        let d = -2.0 * x + 12.0 * y + 3.0;
        [4.0 * x / d, 9.0 * y / d]
    };
    let (r, w) = (ucs(0.64, 0.33), ucs(D65.x, D65.y));
    let want = [l, 13.0 * l * (r[0] - w[0]), 13.0 * l * (r[1] - w[1])];
    assert!(
        close(m.from_srgb(RED), want, 1e-3),
        "{:?}",
        m.from_srgb(RED)
    );
    assert!((want[1] - 175.0).abs() < 0.1 && (want[2] - 37.76).abs() < 0.1);
    // Mid grey keeps the white's chromaticity: u* = v* = 0.
    let g = m.from_srgb([0.5, 0.5, 0.5]);
    assert!((g[0] - 53.389).abs() < 1e-3 && g[1].abs() < 1e-9 && g[2].abs() < 1e-9);
    // Polar form of red.
    let lch = ColorModel::Lchuv(LabWhite::D65).from_srgb(RED);
    assert!((lch[1] - want[1].hypot(want[2])).abs() < 1e-3);
    assert!((lch[2] - want[2].atan2(want[1]).to_degrees()).abs() < 1e-3);
    let hcl = ColorModel::HclUv(LabWhite::D65).from_srgb(RED);
    assert_eq!([hcl[2], hcl[1], hcl[0]], lch);
    // Direct XYZ round trip, including black.
    let w = D65.to_xyz();
    for xyz in [[0.2, 0.3, 0.1], [0.9, 1.0, 1.05], [0.0, 0.0, 0.0]] {
        assert!(close(luv_to_xyz(xyz_to_luv(xyz, w), w), xyz, 1e-12));
    }
}

#[test]
fn jzazbz_reference_points() {
    // D65 white at 100 cd/m²: neutral (az, bz within 3e-4 of zero — the
    // published matrices carry seven digits) and Jz ≈ 0.1672.
    let w = D65.to_xyz().map(|c| c * 100.0);
    let j = xyz_abs_to_jzazbz(w);
    assert!((j[0] - 0.1672).abs() < 2e-4, "{j:?}");
    assert!(j[1].abs() < 3e-4 && j[2].abs() < 3e-4, "{j:?}");
    // Black is the origin exactly (that is what d0 is for).
    let k = xyz_abs_to_jzazbz([0.0; 3]);
    assert!(k.iter().all(|v| v.abs() < 1e-12), "{k:?}");
    // The ceiling of the encoding: 10 000 cd/m² white gives Jz just
    // under 1.
    let top = xyz_abs_to_jzazbz(D65.to_xyz().map(|c| c * 10000.0));
    assert!(top[0] < 1.0 && top[0] > 0.98, "{top:?}");
    for xyz in [[20.0, 30.0, 10.0], [95.0, 100.0, 108.0], [0.5, 0.2, 3.0]] {
        let back = jzazbz_to_xyz_abs(xyz_abs_to_jzazbz(xyz));
        assert!(close(back, xyz, 1e-7), "{back:?}");
    }
    // Lightness is monotonic along the grey axis.
    let m = ColorModel::Jzazbz(ColorModel::JZ_HDR_REFERENCE_WHITE);
    let mut last = -1.0;
    for k in 0..=64 {
        let v = m.from_srgb([k as f64 / 64.0; 3])[0];
        assert!(v > last);
        last = v;
    }
}

#[test]
fn broadcast_colour_difference_reference_points() {
    // Narrow-range Y′CbCr: the 8-bit rails of H.273.
    let m = ColorModel::YCbCrLimited(LumaWeights::BT601);
    let code = |v: [f64; 3]| v.map(|c| c * 255.0);
    assert!(close(code(m.from_srgb(WHITE)), [235.0, 128.0, 128.0], 1e-9));
    assert!(close(code(m.from_srgb(BLACK)), [16.0, 128.0, 128.0], 1e-9));
    let red = code(m.from_srgb(RED));
    let want = [16.0 + 219.0 * 0.299, 128.0 - 112.0 * 0.299 / 0.886, 240.0];
    assert!(close(red, want, 1e-9), "{red:?}");
    let blue = code(ColorModel::YCbCrLimited(LumaWeights::BT2020).from_srgb(BLUE));
    assert!(close(
        blue,
        [
            16.0 + 219.0 * 0.0593,
            240.0,
            128.0 - 112.0 * 0.0593 / 0.7373
        ],
        1e-9
    ));
    // PAL Y′UV: blue reaches U = 0.436, red V = 0.615.
    let yuv = ColorModel::Yuv;
    assert!(close(
        yuv.from_srgb(BLUE),
        [0.114, 0.436, -0.615 * 0.114 / 0.701],
        1e-12
    ));
    assert!(close(
        yuv.from_srgb(RED),
        [0.299, -0.436 * 0.299 / 0.886, 0.615],
        1e-12
    ));
    assert!(close(yuv.from_srgb(WHITE), [1.0, 0.0, 0.0], 1e-12));
    // The published three-decimal Y′UV rows.
    let g = yuv.from_srgb([0.0, 1.0, 0.0]);
    assert!(close(g, [0.587, -0.289, -0.515], 5e-4), "{g:?}");
    // NTSC Y′IQ: the published four-decimal matrix.
    let yiq = ColorModel::Yiq;
    assert!(close(yiq.from_srgb(RED), [0.299, 0.5959, 0.2115], 1e-4));
    assert!(close(
        yiq.from_srgb([0.0, 1.0, 0.0]),
        [0.587, -0.2746, -0.5227],
        1e-4
    ));
    assert!(close(yiq.from_srgb(BLUE), [0.114, -0.3213, 0.3112], 1e-4));
    assert!(close(yiq.from_srgb([0.5; 3]), [0.5, 0.0, 0.0], 1e-12));
    // SECAM Y′DbDr: Db and Dr span ±1.333.
    let d = ColorModel::YDbDr;
    assert!(close(
        d.from_srgb(BLUE),
        [0.114, 1.505 * 0.886, 1.902 * 0.114],
        1e-12
    ));
    assert!(close(
        d.from_srgb(RED),
        [0.299, -1.505 * 0.299, -1.902 * 0.701],
        1e-12
    ));
    assert!((1.505f64 * 0.886 - 1.333).abs() < 1e-3 && (1.902f64 * 0.701 - 1.333).abs() < 1e-3);
}

#[test]
fn photoycc_reference_points() {
    let m = ColorModel::PhotoYcc;
    let code = |v: [f64; 3]| v.map(|c| c * 255.0);
    // Neutral colours sit on the chroma offsets 156 / 137; white is
    // 255 / 1.402 ≈ 181.9.
    assert!(close(
        code(m.from_srgb(WHITE)),
        [255.0 / 1.402, 156.0, 137.0],
        1e-9
    ));
    assert!(close(code(m.from_srgb(BLACK)), [0.0, 156.0, 137.0], 1e-9));
    let red = code(m.from_srgb(RED));
    let want = [
        255.0 / 1.402 * 0.299,
        156.0 - 111.40 * 0.299,
        137.0 + 135.64 * 0.701,
    ];
    assert!(close(red, want, 1e-9), "{red:?}");
    // Mid grey: sRGB 0.5 is linear 0.21404, BT.709-encoded.
    let lin = srgb_to_linear(0.5);
    let v = 1.099 * lin.powf(0.45) - 0.099;
    assert!(close(
        code(m.from_srgb([0.5; 3])),
        [255.0 / 1.402 * v, 156.0, 137.0],
        1e-9
    ));
}

#[test]
fn cineon_log_reference_points() {
    let p = CineonLog::new();
    assert!((p.encode(1.0) - 685.0).abs() < 1e-9);
    assert!((p.encode(0.0) - 95.0).abs() < 1e-9);
    // 300 codes per decade: ten times the white exposure is +300 codes
    // on the offset-free scale.
    let o = 10f64.powf(-590.0 / 300.0);
    let lin18 = 0.18;
    let want = 685.0 + 300.0 * (lin18 * (1.0 - o) + o).log10();
    assert!((p.encode(lin18) - want).abs() < 1e-9);
    assert!((want - 470.0).abs() < 3.0, "{want}");
    for l in [0.0, 0.001, 0.18, 1.0, 4.0] {
        assert!((p.decode(p.encode(l)) - l).abs() < 1e-12);
    }
    let m = ColorModel::Log(p);
    assert!(close(m.from_srgb(WHITE), [685.0 / 1023.0; 3], 1e-12));
    assert!(close(m.from_srgb(BLACK), [95.0 / 1023.0; 3], 1e-12));
    // Exposure below film base clamps to code 0.
    assert_eq!(p.encode(-1.0), p.encode(-2.0));
    assert!(p.encode(-1.0).abs() < 1e-9);
}

#[test]
fn cone_response_reference_points() {
    // CAT02 and Bradford rows sum to one (to the four published
    // decimals): the equal-energy white maps to L = M = S = 1.
    for c in [ConeResponse::Cat02, ConeResponse::Bradford] {
        for row in c.matrix() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 2e-4);
        }
    }
    // Hunt–Pointer–Estevez is normalised to D65: sRGB white is (1, 1, 1).
    let w = ColorModel::LmsOf(ConeResponse::HuntPointerEstevez).from_srgb(WHITE);
    assert!(close(w, [1.0, 1.0, 1.0], 2e-4), "{w:?}");
    assert_eq!(
        ColorModel::Lms.from_srgb([0.2, 0.6, 0.9]),
        ColorModel::LmsOf(ConeResponse::Cat02).from_srgb([0.2, 0.6, 0.9])
    );
    // scRGB is linear sRGB, values outside 0..1 kept.
    let v = ColorModel::ScRgb.from_srgb([1.5, -0.25, 0.5]);
    assert!(v[0] > 1.0 && v[1] < 0.0);
    assert!(close(
        ColorModel::ScRgb.to_srgb(v),
        [1.5, -0.25, 0.5],
        1e-12
    ));
}
