//! Black-box comparison of `colorspace::ColorModel` with `magick
//! -colorspace X` (ImageMagick 7 Q16-HDRI, 32-bit float raw I/O). The
//! channel scalings ImageMagick stores (e.g. `L*/100`, `a*/255 + 0.5`,
//! hue/360) were read off its output; the formulas are the staged ones
//! our code implements. Skips when `magick` is not on `PATH`.
#![cfg(not(miri))]

use std::process::Command;

use oxideav_pixfmt::colorspace::{ColorModel, LabWhite, LumaWeights};

/// The luma weights ImageMagick's HCL / YIQ / YUV rows use (measured).
const MAGICK_LUMA: LumaWeights = LumaWeights::new(0.298839, 0.114350);

fn magick_available() -> bool {
    Command::new("magick")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn probes() -> Vec<[f32; 3]> {
    let mut v = Vec::new();
    for r in 0..6 {
        for g in 0..6 {
            for b in 0..6 {
                v.push([r as f32 / 5.0, g as f32 / 5.0, b as f32 / 5.0]);
            }
        }
    }
    v
}

fn magick_colorspace(px: &[[f32; 3]], cs: &str) -> Vec<[f64; 3]> {
    magick_run(px, cs, false)
}

/// Pixels holding `cs` values → `magick -colorspace sRGB`.
fn magick_to_srgb(px: &[[f32; 3]], cs: &str) -> Vec<[f64; 3]> {
    magick_run(px, cs, true)
}

fn magick_run(px: &[[f32; 3]], cs: &str, inverse: bool) -> Vec<[f64; 3]> {
    let (tag_in, target) = if inverse { (cs, "sRGB") } else { ("sRGB", cs) };
    let dir = std::env::temp_dir();
    let i = dir.join(format!(
        "oxideav-pixfmt-cs-{}-{cs}-{inverse}.in",
        std::process::id()
    ));
    let o = dir.join(format!(
        "oxideav-pixfmt-cs-{}-{cs}-{inverse}.out",
        std::process::id()
    ));
    let bytes: Vec<u8> = px.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&i, bytes).unwrap();
    let st = Command::new("magick")
        .args(["-size", &format!("{}x1", px.len()), "-depth", "32"])
        .args(["-define", "quantum:format=floating-point", "-endian", "LSB"])
        .arg(format!("rgb:{}", i.display()))
        .args(["-set", "colorspace", tag_in, "-colorspace", target])
        .args(["-set", "colorspace", "sRGB"])
        .args([
            "-depth",
            "32",
            "-define",
            "quantum:format=floating-point",
            "-endian",
            "LSB",
        ])
        .arg(format!("rgb:{}", o.display()))
        .status()
        .unwrap();
    assert!(st.success());
    let d = std::fs::read(&o).unwrap();
    let _ = std::fs::remove_file(&i);
    let _ = std::fs::remove_file(&o);
    let f: Vec<f64> = d
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c) as f64)
        .collect();
    f.as_chunks::<3>().0.to_vec()
}

#[test]
fn models_match_magick_colorspace() {
    if !magick_available() {
        eprintln!("magick not found; skipping");
        return;
    }
    let px = probes();
    // (ImageMagick name, our model, scaling of ours into IM's storage,
    // tolerance, compare hue channel?)
    type Scale = fn([f64; 3]) -> [f64; 3];
    let cases: [(&str, ColorModel, Scale, f64); 25] = [
        ("RGB", ColorModel::LinearSrgb, |v| v, 1e-5),
        ("Gray", ColorModel::Gray(LumaWeights::BT709), |v| v, 1e-4),
        (
            "LinearGray",
            ColorModel::LinearGray(LumaWeights::BT709),
            |v| v,
            1e-4,
        ),
        ("XYZ", ColorModel::XyzD65, |v| v, 2e-4),
        ("xyY", ColorModel::XyY, |v| v, 2e-4),
        (
            "Lab",
            ColorModel::Lab(LabWhite::D65),
            |v| [v[0] / 100.0, v[1] / 255.0 + 0.5, v[2] / 255.0 + 0.5],
            2e-4,
        ),
        (
            "Oklab",
            ColorModel::Oklab,
            |v| [v[0], v[1] + 0.5, v[2] + 0.5],
            1e-4,
        ),
        (
            "LCHab",
            ColorModel::Lch(LabWhite::D65),
            |v| [v[0] / 100.0, v[1] / 255.0 + 0.5, v[2] / 360.0],
            2e-4,
        ),
        // ImageMagick stores the OkLCh hue in radians.
        (
            "Oklch",
            ColorModel::Oklch,
            |v| [v[0], v[1], v[2].to_radians()],
            1e-4,
        ),
        ("HWB", ColorModel::Hwb, |v| [v[0] / 360.0, v[1], v[2]], 1e-5),
        ("HSL", ColorModel::Hsl, |v| [v[0] / 360.0, v[1], v[2]], 1e-5),
        ("CMY", ColorModel::Cmy, |v| v, 1e-6),
        (
            "Rec709YCbCr",
            ColorModel::YCbCr(LumaWeights::BT709),
            |v| v,
            2e-4,
        ),
        // ImageMagick's BT.601 luma weights differ from BT.601's
        // 0.299 / 0.114 by ~1.6e-4.
        (
            "Rec601YCbCr",
            ColorModel::YCbCr(LumaWeights::BT601),
            |v| v,
            5e-4,
        ),
        (
            "YPbPr",
            ColorModel::YPbPr(LumaWeights::BT601),
            |v| [v[0], v[1] + 0.5, v[2] + 0.5],
            5e-4,
        ),
        ("HSV", ColorModel::Hsv, |v| [v[0] / 360.0, v[1], v[2]], 1e-5),
        ("HSB", ColorModel::Hsv, |v| [v[0] / 360.0, v[1], v[2]], 1e-5),
        ("HSI", ColorModel::Hsi, |v| [v[0] / 360.0, v[1], v[2]], 1e-5),
        (
            "HCL",
            ColorModel::Hcl(MAGICK_LUMA),
            |v| [v[0] / 360.0, v[1], v[2]],
            1e-5,
        ),
        (
            "HCLp",
            ColorModel::Hclp(MAGICK_LUMA),
            |v| [v[0] / 360.0, v[1], v[2]],
            1e-5,
        ),
        (
            "YIQ",
            ColorModel::Yiq,
            |v| [v[0], v[1] + 0.5, v[2] + 0.5],
            1e-5,
        ),
        (
            "YUV",
            ColorModel::Yuv,
            |v| [v[0], v[1] + 0.5, v[2] + 0.5],
            1e-5,
        ),
        (
            "YDbDr",
            ColorModel::YDbDr,
            |v| [v[0], v[1] + 0.5, v[2] + 0.5],
            1e-5,
        ),
        // ImageMagick offsets I2 / I3 by 32768 / 65535 and rounds the
        // I1 weights to five decimals.
        (
            "OHTA",
            ColorModel::Ohta,
            |v| [v[0], v[1] + 0.5, v[2] + 0.5],
            3e-5,
        ),
        ("LMS", ColorModel::Lms, |v| v, 2e-4),
    ];
    for (name, model, scale, tol) in cases {
        let theirs = magick_colorspace(&px, name);
        let mut worst: f64 = 0.0;
        for (p, t) in px.iter().zip(&theirs) {
            // xyY of black is undefined; we report the white's
            // chromaticity, ImageMagick (0, 0).
            if name == "xyY" && p.iter().all(|&v| v == 0.0) {
                continue;
            }
            let ours = scale(model.from_srgb(p.map(|v| v as f64)));
            for c in 0..3 {
                // Skip the hue of achromatic colours (powerless; a
                // near-zero chroma makes it numerically arbitrary).
                let achromatic = p[0] == p[1] && p[1] == p[2];
                let hue_slot = match name {
                    "HSL" | "HWB" | "HSV" | "HSB" | "HSI" | "HCL" | "HCLp" => 0,
                    "LCHab" | "Oklch" => 2,
                    _ => 3,
                };
                if c == hue_slot && achromatic {
                    continue;
                }
                let mut d = (ours[c] - t[c]).abs();
                if c == hue_slot && name != "Oklch" {
                    // Hues are circular: 0 and 1 (= 360°) coincide.
                    d = d.min((1.0 - d).abs());
                }
                worst = worst.max(d);
            }
        }
        eprintln!("{name}: max |Δ| {worst:.2e}");
        assert!(worst < tol, "{name}: {worst}");
    }
}

#[test]
fn inverse_models_match_magick() {
    if !magick_available() {
        eprintln!("magick not found; skipping");
        return;
    }
    // Model values (in ImageMagick's storage scaling) on a grid; both
    // sides convert them back to sRGB.
    let mut grid = Vec::new();
    for a in 0..5 {
        for b in 0..5 {
            for c in 0..5 {
                grid.push([
                    0.1 + 0.2 * a as f32,
                    0.1 + 0.2 * b as f32,
                    0.1 + 0.2 * c as f32,
                ]);
            }
        }
    }
    // ImageMagick's way back from Y′IQ / Y′UV / Y′DbDr / OHTA uses its
    // own rounded inverse coefficients; ours is the exact inverse of
    // the forward matrix (so our round trips stay within ±1 LSB at 16
    // bits), which puts the two up to ~2.5e-4 apart.
    type Unscale = fn([f64; 3]) -> [f64; 3];
    let cases: [(&str, ColorModel, Unscale, f64); 10] = [
        ("HSV", ColorModel::Hsv, |v| [v[0] * 360.0, v[1], v[2]], 1e-5),
        ("HSI", ColorModel::Hsi, |v| [v[0] * 360.0, v[1], v[2]], 1e-5),
        (
            "HCL",
            ColorModel::Hcl(MAGICK_LUMA),
            |v| [v[0] * 360.0, v[1], v[2]],
            1e-5,
        ),
        ("HSL", ColorModel::Hsl, |v| [v[0] * 360.0, v[1], v[2]], 1e-5),
        (
            "HCLp",
            ColorModel::Hclp(MAGICK_LUMA),
            |v| [v[0] * 360.0, v[1], v[2]],
            1e-5,
        ),
        (
            "YIQ",
            ColorModel::Yiq,
            |v| [v[0], v[1] - 0.5, v[2] - 0.5],
            3e-4,
        ),
        (
            "YUV",
            ColorModel::Yuv,
            |v| [v[0], v[1] - 0.5, v[2] - 0.5],
            3e-4,
        ),
        (
            "YDbDr",
            ColorModel::YDbDr,
            |v| [v[0], v[1] - 0.5, v[2] - 0.5],
            3e-4,
        ),
        (
            "OHTA",
            ColorModel::Ohta,
            |v| [v[0], v[1] - 0.5, v[2] - 0.5],
            3e-5,
        ),
        ("LMS", ColorModel::Lms, |v| v, 1e-3),
    ];
    for (name, model, unscale, tol) in cases {
        let theirs = magick_to_srgb(&grid, name);
        let (mut worst, mut n): (f64, usize) = (0.0, 0);
        for (g, t) in grid.iter().zip(&theirs) {
            let ours = model.to_srgb(unscale(g.map(|v| v as f64)));
            // Compare in gamut only: out-of-gamut results may be
            // clipped on the way out.
            let inside = |v: &[f64; 3]| v.iter().all(|c| (0.0..=1.0).contains(c));
            if !inside(&ours) || !inside(t) {
                continue;
            }
            n += 1;
            for c in 0..3 {
                worst = worst.max((ours[c] - t[c]).abs());
            }
        }
        eprintln!("{name} inverse: max |Δ| {worst:.2e} over {n} in-gamut values");
        assert!(n > 10, "{name}: only {n} in-gamut values");
        assert!(worst < tol, "{name}: {worst}");
    }
}
