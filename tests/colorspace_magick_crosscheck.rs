//! Black-box comparison of `colorspace::ColorModel` with `magick
//! -colorspace X` (ImageMagick 7 Q16-HDRI, 32-bit float raw I/O). The
//! channel scalings ImageMagick stores (e.g. `L*/100`, `a*/255 + 0.5`,
//! hue/360) were read off its output; the formulas are the staged ones
//! our code implements. Skips when `magick` is not on `PATH`.
#![cfg(not(miri))]

use std::process::Command;

use oxideav_pixfmt::colorspace::{ColorModel, LabWhite, LumaWeights};

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
    let dir = std::env::temp_dir();
    let i = dir.join(format!("oxideav-pixfmt-cs-{}-{cs}.in", std::process::id()));
    let o = dir.join(format!("oxideav-pixfmt-cs-{}-{cs}.out", std::process::id()));
    let bytes: Vec<u8> = px.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&i, bytes).unwrap();
    let st = Command::new("magick")
        .args(["-size", &format!("{}x1", px.len()), "-depth", "32"])
        .args(["-define", "quantum:format=floating-point", "-endian", "LSB"])
        .arg(format!("rgb:{}", i.display()))
        .args([
            "-set",
            "colorspace",
            "sRGB",
            "-colorspace",
            cs,
            "-set",
            "colorspace",
            "sRGB",
        ])
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
    let cases: [(&str, ColorModel, Scale, f64); 15] = [
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
                    "HSL" | "HWB" => 0,
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
