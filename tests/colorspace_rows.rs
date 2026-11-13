//! `colorspace::ModelConverter`: buffer conversions against the
//! per-colour `ColorModel` functions, round trips over the gamut, the
//! out-of-gamut policy, the integer encodings and thread parity.

use oxideav_core::ExecutionContext;
use oxideav_pixfmt::colorspace::{
    CineonLog, ColorModel, ConeResponse, GamutPolicy, LabWhite, LumaWeights, ModelConverter,
    linear_to_srgb, srgb_to_linear,
};

fn models() -> Vec<ColorModel> {
    let w = LumaWeights::BT709;
    vec![
        ColorModel::Srgb,
        ColorModel::LinearSrgb,
        ColorModel::ScRgb,
        ColorModel::XyzD65,
        ColorModel::XyzD50,
        ColorModel::XyY,
        ColorModel::Lab(LabWhite::D65),
        ColorModel::Lab(LabWhite::D50),
        ColorModel::Lch(LabWhite::D50),
        ColorModel::HclAb(LabWhite::D65),
        ColorModel::Luv(LabWhite::D65),
        ColorModel::Lchuv(LabWhite::D50),
        ColorModel::HclUv(LabWhite::D65),
        ColorModel::Oklab,
        ColorModel::Oklch,
        ColorModel::Jzazbz(ColorModel::JZ_SDR_WHITE),
        ColorModel::JzCzhz(ColorModel::JZ_HDR_REFERENCE_WHITE),
        ColorModel::Hsl,
        ColorModel::Hwb,
        ColorModel::Hsv,
        ColorModel::Hsi,
        ColorModel::Hcl(w),
        ColorModel::Hclp(w),
        ColorModel::Cmy,
        ColorModel::YPbPr(LumaWeights::BT601),
        ColorModel::YCbCr(w),
        ColorModel::YCbCrLimited(LumaWeights::BT2020),
        ColorModel::Yiq,
        ColorModel::Yuv,
        ColorModel::YDbDr,
        ColorModel::Ohta,
        ColorModel::Lms,
        ColorModel::LmsOf(ConeResponse::HuntPointerEstevez),
        ColorModel::PhotoYcc,
        ColorModel::Log(CineonLog::new()),
    ]
}

fn gamut_probes() -> Vec<[u16; 3]> {
    let (steps, random) = if cfg!(miri) { (1u32, 8) } else { (12, 6000) };
    let lv: Vec<u16> = (0..=steps).map(|k| (k * 65535 / steps) as u16).collect();
    let mut v = Vec::new();
    for &r in &lv {
        for &g in &lv {
            for &b in &lv {
                v.push([r, g, b]);
            }
        }
    }
    let mut s: u32 = 0x1234_5678;
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
    for k in 0..if cfg!(miri) { 2u16 } else { 48 } {
        v.push([k, k * 5, 65535 - k]);
        v.push([65535 - k, k, k / 2]);
    }
    v
}

#[test]
fn buffers_agree_with_the_per_colour_functions() {
    let probes: Vec<f64> = gamut_probes()
        .iter()
        .step_by(if cfg!(miri) { 1 } else { 7 })
        .flat_map(|p| p.map(|c| c as f64 / 65535.0))
        .collect();
    let all = models();
    for (i, &a) in all.iter().enumerate() {
        // Every model against sRGB both ways, and a rotating partner.
        for b in [ColorModel::Srgb, all[(i * 7 + 3) % all.len()]] {
            let from_srgb = ModelConverter::new(ColorModel::Srgb, a);
            let mut va = vec![0f64; probes.len()];
            from_srgb.convert_f64(&probes, &mut va).unwrap();
            let conv = ModelConverter::new(a, b);
            let mut vb = vec![0f64; probes.len()];
            conv.convert_f64(&va, &mut vb).unwrap();
            for (x, y) in va
                .as_chunks::<3>()
                .0
                .iter()
                .zip(vb.as_chunks::<3>().0.iter())
            {
                let x = *x;
                let want = a.convert(b, x);
                // A hue is circular and meaningless at zero chroma, so
                // hue-bearing results are compared as the colours they
                // stand for.
                let hue = b.component_ranges().contains(&(0.0, 360.0));
                let (got, want) = if hue {
                    (b.to_srgb(*y), b.to_srgb(want))
                } else {
                    (*y, want)
                };
                for c in 0..3 {
                    let tol = 1e-9 * (1.0 + want[c].abs());
                    assert!(
                        (got[c] - want[c]).abs() <= tol,
                        "{a:?} → {b:?}: {x:?} gave {got:?}, want {want:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn f32_round_trips_stay_within_one_lsb_at_16_bits() {
    let probes = gamut_probes();
    let src: Vec<f32> = probes
        .iter()
        .flat_map(|p| p.map(|c| c as f32 / 65535.0))
        .collect();
    for m in models() {
        if matches!(m, ColorModel::Hclp(_)) {
            continue; // same forward as Hcl; its inverse pulls into gamut
        }
        let mut mid = vec![0f32; src.len()];
        let mut back = vec![0f32; src.len()];
        ModelConverter::new(ColorModel::Srgb, m)
            .convert_f32(&src, &mut mid)
            .unwrap();
        ModelConverter::new(m, ColorModel::Srgb)
            .convert_f32(&mid, &mut back)
            .unwrap();
        let mut worst = 0i32;
        for (p, b) in probes.iter().zip(back.as_chunks::<3>().0.iter()) {
            for c in 0..3 {
                let q = (b[c].clamp(0.0, 1.0) as f64 * 65535.0).round() as i32;
                worst = worst.max((q - p[c] as i32).abs());
            }
        }
        assert!(worst <= 1, "{m:?}: {worst} LSB");
    }
}

#[test]
fn out_of_gamut_values_are_preserved_or_clamped() {
    // A saturated CIELAB green far outside sRGB, and a brighter-than-
    // white grey.
    let lab = [80.0f64, -120.0, 90.0, 120.0, 0.0, 0.0];
    let m = ColorModel::Lab(LabWhite::D50);
    let keep = ModelConverter::new(m, ColorModel::Srgb);
    let mut rgb = [0f64; 6];
    keep.convert_f64(&lab, &mut rgb).unwrap();
    assert!(rgb[0] < 0.0 && rgb[2] < 0.0, "{rgb:?}");
    assert!(rgb[3] > 1.0);
    // Preserved values survive the trip back.
    let mut again = [0f64; 6];
    ModelConverter::new(ColorModel::Srgb, m)
        .convert_f64(&rgb, &mut again)
        .unwrap();
    for (a, b) in lab.iter().zip(&again) {
        assert!((a - b).abs() < 1e-9, "{again:?}");
    }
    // Clamped values are inside 0..1 and equal the clamp of the
    // preserved ones.
    let clamp = keep.clone().with_gamut(GamutPolicy::Clamp);
    let mut c = [0f64; 6];
    clamp.convert_f64(&lab, &mut c).unwrap();
    for (k, p) in c.iter().zip(&rgb) {
        assert!((0.0..=1.0).contains(k));
        assert!((k - p.clamp(0.0, 1.0)).abs() < 1e-12);
    }
    // Clamping also applies between two non-sRGB models: the linear
    // value is limited before it enters the destination.
    let lin = ModelConverter::new(m, ColorModel::LinearSrgb).with_gamut(GamutPolicy::Clamp);
    let mut l = [0f64; 6];
    lin.convert_f64(&lab, &mut l).unwrap();
    assert!(l.iter().all(|v| (0.0..=1.0).contains(v)));
    assert_eq!(&l[3..], &[1.0, 1.0, 1.0]);
    // Integer outputs saturate under either policy.
    let lab16 = ModelConverter::new(m, ColorModel::Srgb);
    let ranges = m.component_ranges();
    let code = |v: f64, c: usize| {
        ((v - ranges[c].0) / (ranges[c].1 - ranges[c].0) * 65535.0).round() as u16
    };
    let src = [code(80.0, 0), code(-120.0, 1), code(90.0, 2)];
    let mut out = [0u16; 3];
    lab16.convert_u16(&src, &mut out).unwrap();
    assert_eq!(out[0], 0);
    assert_eq!(out[2], 0);
    assert!(out[1] > 60000);
}

#[test]
fn integer_paths_follow_the_component_ranges() {
    // 8-bit sRGB → narrow-range BT.601 Y′CbCr: the H.273 rails.
    let c = ModelConverter::new(
        ColorModel::Srgb,
        ColorModel::YCbCrLimited(LumaWeights::BT601),
    );
    let src = [255u8, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255];
    let mut dst = [0u8; 12];
    c.convert_u8(&src, &mut dst).unwrap();
    assert_eq!(
        dst,
        [235, 128, 128, 16, 128, 128, 81, 90, 240, 41, 240, 110]
    );
    let back = ModelConverter::new(
        ColorModel::YCbCrLimited(LumaWeights::BT601),
        ColorModel::Srgb,
    );
    let mut rgb = [0u8; 12];
    back.convert_u8(&dst, &mut rgb).unwrap();
    for (a, b) in src.iter().zip(&rgb) {
        assert!((*a as i32 - *b as i32).abs() <= 1, "{rgb:?}");
    }
    // 16-bit sRGB ↔ linear: the table-backed paths against the exact
    // formula, every code.
    let n = if cfg!(miri) { 64 } else { 65536 };
    let ramp: Vec<u16> = (0..n as u64)
        .flat_map(|k| {
            let v = (k * 65535 / (n as u64 - 1)) as u16;
            [v, v, v]
        })
        .collect();
    let mut lin = vec![0u16; ramp.len()];
    ModelConverter::new(ColorModel::Srgb, ColorModel::LinearSrgb)
        .convert_u16(&ramp, &mut lin)
        .unwrap();
    let mut enc = vec![0u16; ramp.len()];
    ModelConverter::new(ColorModel::LinearSrgb, ColorModel::Srgb)
        .convert_u16(&ramp, &mut enc)
        .unwrap();
    for (i, v) in ramp.iter().enumerate() {
        let x = *v as f64 / 65535.0;
        let want_lin = (srgb_to_linear(x) * 65535.0).round() as i32;
        assert!((lin[i] as i32 - want_lin).abs() <= 1, "linearise {v}");
        let want_enc = (linear_to_srgb(x) * 65535.0).round() as i32;
        assert!((enc[i] as i32 - want_enc).abs() <= 1, "encode {v}");
    }
    // CIELAB at 8 bits: L* 0..100 over 0..255, a* = code − 128.
    let lab = ModelConverter::new(ColorModel::Srgb, ColorModel::Lab(LabWhite::D50));
    let mut out = [0u8; 6];
    lab.convert_u8(&[255, 255, 255, 0, 0, 0], &mut out).unwrap();
    assert_eq!(out, [255, 128, 128, 0, 128, 128]);
    // scRGB at 16 bits: black 4096, white 12288 (IEC 61966-2-2).
    let sc = ModelConverter::new(ColorModel::Srgb, ColorModel::ScRgb);
    let mut out = [0u16; 6];
    sc.convert_u16(&[0, 0, 0, 65535, 65535, 65535], &mut out)
        .unwrap();
    assert_eq!(out, [4096, 4096, 4096, 12288, 12288, 12288]);
}

#[test]
fn integer_srgb_round_trip_through_16_bit_linear_models() {
    // sRGB 8-bit → 16-bit-precision models and back is lossless when
    // the intermediate is held in f32; through the integer encodings an
    // 8-bit sRGB value survives a 16-bit linear intermediate.
    let n = if cfg!(miri) { 8 } else { 256 };
    let src8: Vec<u8> = (0..n)
        .flat_map(|k| [k as u8, (255 - k) as u8, 77])
        .collect();
    let src16: Vec<u16> = src8.iter().map(|&v| v as u16 * 257).collect();
    for m in [
        ColorModel::LinearSrgb,
        ColorModel::XyzD65,
        ColorModel::Oklab,
    ] {
        let mut mid = vec![0u16; src16.len()];
        let mut back = vec![0u16; src16.len()];
        ModelConverter::new(ColorModel::Srgb, m)
            .convert_u16(&src16, &mut mid)
            .unwrap();
        ModelConverter::new(m, ColorModel::Srgb)
            .convert_u16(&mid, &mut back)
            .unwrap();
        for (a, b) in src8.iter().zip(&back) {
            // 12.92 codes of sRGB per linear code at the toe: within
            // 13 16-bit codes, i.e. the same 8-bit value.
            assert_eq!(((*b as u32 + 128) / 257) as u8, *a, "{m:?}");
        }
    }
}

#[test]
fn threads_and_extra_channels_do_not_change_the_colour() {
    let px = if cfg!(miri) { 64 } else { 40_000 };
    let mut s: u32 = 0xCAFE_F00D;
    let rgba: Vec<u16> = (0..px * 4)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u16
        })
        .collect();
    for (a, b) in [
        (ColorModel::Srgb, ColorModel::Oklch),
        (ColorModel::Srgb, ColorModel::YCbCr(LumaWeights::BT709)),
        (ColorModel::Lab(LabWhite::D50), ColorModel::Srgb),
        (ColorModel::XyzD65, ColorModel::Srgb),
    ] {
        let serial = ModelConverter::new(a, b).with_channels(4);
        let mut one = vec![0u16; rgba.len()];
        serial.convert_u16(&rgba, &mut one).unwrap();
        for t in [2, 3, 8] {
            let par = serial
                .clone()
                .with_execution(&ExecutionContext::with_threads(t));
            let mut many = vec![0u16; rgba.len()];
            par.convert_u16(&rgba, &mut many).unwrap();
            assert_eq!(one, many, "{a:?} → {b:?} on {t} threads");
        }
        // Alpha is carried; the colour equals the 3-channel result.
        let rgb: Vec<u16> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let mut three = vec![0u16; rgb.len()];
        ModelConverter::new(a, b)
            .convert_u16(&rgb, &mut three)
            .unwrap();
        for ((o, i), t) in one
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba.as_chunks::<4>().0.iter())
            .zip(three.as_chunks::<3>().0.iter())
        {
            assert_eq!(o[3], i[3]);
            assert_eq!(&o[..3], &t[..]);
        }
    }
    // Float buffers too.
    let f: Vec<f32> = rgba.iter().map(|&v| v as f32 / 65535.0).collect();
    let c = ModelConverter::new(ColorModel::Srgb, ColorModel::Jzazbz(203.0)).with_channels(4);
    let mut one = vec![0f32; f.len()];
    let mut many = vec![0f32; f.len()];
    c.convert_f32(&f, &mut one).unwrap();
    c.clone()
        .with_threads(5)
        .convert_f32(&f, &mut many)
        .unwrap();
    assert_eq!(one, many);
}

#[test]
fn mismatched_buffers_are_rejected() {
    let c = ModelConverter::new(ColorModel::Srgb, ColorModel::Hsl);
    assert!(c.convert_f32(&[0.0; 4], &mut [0.0; 4]).is_err());
    assert!(c.convert_u8(&[0; 6], &mut [0; 3]).is_err());
    assert!(c.convert_u16(&[], &mut []).is_ok());
    let same = ModelConverter::new(ColorModel::Hsl, ColorModel::Hsl);
    let mut out = [0f32; 3];
    same.convert_f32(&[725.0, 2.0, -1.0], &mut out).unwrap();
    assert_eq!(out, [725.0, 2.0, -1.0]);
    assert_eq!(same.from_model(), ColorModel::Hsl);
    assert_eq!(c.to_model(), ColorModel::Hsl);
    assert_eq!(c.convert_pixel([1.0, 0.0, 0.0]), [0.0, 1.0, 0.5]);
}
