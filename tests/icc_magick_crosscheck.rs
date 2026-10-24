//! Black-box cross-validation of `icc::IccTransform` against the
//! `magick` binary (ImageMagick 7, Q16-HDRI) used strictly as an opaque
//! CLI validator: 16-bit raw pixels go in with `-profile a.icc -profile
//! b.icc`, raw pixels come out, and the two results must agree within
//! CIEDE2000 < 1 (CSS Color 4 §20.2 formula, our own code), measured on
//! the destination profile's own PCSLAB.
//!
//! Skips when `magick` is not on `PATH`. The macOS system profiles are
//! used when present; the synthesized profiles always are.
#![cfg(all(feature = "icc", not(miri)))]

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_pixfmt::colorspace::delta_e2000;
use oxideav_pixfmt::icc::{DataColorSpace, IccProfile, IccTransform, RenderingIntent};

fn magick_available() -> bool {
    Command::new("magick")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn scratch(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "oxideav-pixfmt-icc-{}-{}-{tag}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

fn raw_kind(cs: DataColorSpace) -> &'static str {
    match cs {
        DataColorSpace::Rgb => "rgb",
        DataColorSpace::Cmyk => "cmyk",
        DataColorSpace::Gray => "gray",
        _ => panic!("no raw format for {cs:?}"),
    }
}

/// Run `magick` on 16-bit big-endian samples.
fn magick_profile(
    input: &[u16],
    src_cs: DataColorSpace,
    dst_cs: DataColorSpace,
    a: &Path,
    b: &Path,
    intent: Option<&str>,
) -> Vec<u16> {
    let ch = src_cs.channels().unwrap();
    let px = input.len() / ch;
    let inp = scratch("in.raw");
    let out = scratch("out.raw");
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_be_bytes()).collect();
    std::fs::write(&inp, bytes).unwrap();
    let mut cmd = Command::new("magick");
    cmd.args([
        "-size",
        &format!("{px}x1"),
        "-depth",
        "16",
        "-endian",
        "MSB",
    ])
    .arg(format!("{}:{}", raw_kind(src_cs), inp.display()))
    .arg("-profile")
    .arg(a);
    if let Some(i) = intent {
        cmd.args(["-intent", i]);
    }
    cmd.arg("-profile")
        .arg(b)
        .args(["-depth", "16", "-endian", "MSB"])
        .arg(format!("{}:{}", raw_kind(dst_cs), out.display()));
    let st = cmd.status().expect("spawn magick");
    assert!(st.success(), "magick failed");
    let o = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&out);
    o.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_be_bytes(*c))
        .collect()
}

/// Deterministic probe set: a coarse grid plus pseudo-random points.
fn probes(ch: usize) -> Vec<u16> {
    let mut v = Vec::new();
    let levels: &[u16] = if ch == 4 {
        &[0, 16384, 40000, 65535]
    } else {
        &[0, 8192, 21000, 32768, 47000, 58000, 65535]
    };
    let mut idx = vec![0usize; ch];
    loop {
        for &i in &idx {
            v.push(levels[i]);
        }
        let mut d = 0;
        loop {
            if d == ch {
                break;
            }
            idx[d] += 1;
            if idx[d] < levels.len() {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == ch {
            break;
        }
    }
    let mut s: u32 = 0x1234_5678;
    for _ in 0..200 * ch {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v.push((s >> 16) as u16);
    }
    v
}

struct Stats {
    max: f64,
    mean: f64,
}

fn compare(
    src: &IccProfile,
    dst: &IccProfile,
    intent: RenderingIntent,
    a: &Path,
    b: &Path,
) -> Stats {
    let src_cs = src.header().color_space;
    let dst_cs = dst.header().color_space;
    let input = probes(src_cs.channels().unwrap());
    let iname = match intent {
        RenderingIntent::Perceptual => "Perceptual",
        RenderingIntent::RelativeColorimetric => "Relative",
        RenderingIntent::Saturation => "Saturation",
        RenderingIntent::AbsoluteColorimetric => "Absolute",
    };
    let theirs = magick_profile(&input, src_cs, dst_cs, a, b, Some(iname));
    let t = IccTransform::new(src, dst, intent).expect("transform");
    let mut ours = vec![0u16; theirs.len()];
    t.transform_u16(&input, &mut ours).unwrap();
    // Measure both results in the destination profile's PCSLAB.
    let lab = IccProfile::lab_d50();
    let to_lab = IccTransform::new(dst, &lab, RenderingIntent::RelativeColorimetric).unwrap();
    let oc = dst_cs.channels().unwrap();
    let mut max: f64 = 0.0;
    let mut sum = 0.0;
    let n = ours.len() / oc;
    for p in 0..n {
        let mut l1 = [0.0; 3];
        let mut l2 = [0.0; 3];
        let o: Vec<f64> = ours[p * oc..p * oc + oc]
            .iter()
            .map(|&v| v as f64 / 65535.0)
            .collect();
        let m: Vec<f64> = theirs[p * oc..p * oc + oc]
            .iter()
            .map(|&v| v as f64 / 65535.0)
            .collect();
        to_lab.transform_pixel(&o, &mut l1);
        to_lab.transform_pixel(&m, &mut l2);
        let dec = |l: [f64; 3]| [l[0] * 100.0, l[1] * 255.0 - 128.0, l[2] * 255.0 - 128.0];
        let de = delta_e2000(dec(l1), dec(l2));
        if de > max {
            max = de;
            if std::env::var("ICC_XCHECK_DEBUG").is_ok() {
                let ic = src_cs.channels().unwrap();
                eprintln!(
                    "  worst so far {de:.3}: in {:?} ours {:?} magick {:?}",
                    &input[p * ic..p * ic + ic],
                    &ours[p * oc..p * oc + oc],
                    &theirs[p * oc..p * oc + oc]
                );
            }
        }
        sum += de;
    }
    Stats {
        max,
        mean: sum / n as f64,
    }
}

fn write_tmp(name: &str, p: &IccProfile) -> PathBuf {
    let f = scratch(name);
    std::fs::write(&f, p.as_bytes()).unwrap();
    f
}

#[test]
fn synthesized_matrix_trc_pairs_match_magick() {
    if !magick_available() {
        eprintln!("magick not found; skipping");
        return;
    }
    let profiles = [
        ("srgb", IccProfile::srgb()),
        ("p3", IccProfile::display_p3()),
        ("adobe", IccProfile::adobe_rgb()),
        ("2020", IccProfile::rec2020()),
        ("prophoto", IccProfile::prophoto()),
    ];
    let paths: Vec<PathBuf> = profiles
        .iter()
        .map(|(n, p)| write_tmp(&format!("{n}.icc"), p))
        .collect();
    for (i, (na, a)) in profiles.iter().enumerate() {
        for (j, (nb, b)) in profiles.iter().enumerate() {
            if i == j {
                continue;
            }
            for intent in [
                RenderingIntent::RelativeColorimetric,
                RenderingIntent::Perceptual,
            ] {
                let s = compare(a, b, intent, &paths[i], &paths[j]);
                eprintln!(
                    "{na} -> {nb} {intent:?}: max {:.4} mean {:.4}",
                    s.max, s.mean
                );
                assert!(s.max < 1.0, "{na} -> {nb} {intent:?}: ΔE2000 max {}", s.max);
            }
        }
    }
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

const SYS: &str = "/System/Library/ColorSync/Profiles";

#[test]
fn system_profiles_match_magick() {
    if !magick_available() || !Path::new(SYS).exists() {
        eprintln!("magick or system profiles not found; skipping");
        return;
    }
    let names = [
        "sRGB Profile.icc",
        "Display P3.icc",
        "AdobeRGB1998.icc",
        "Generic RGB Profile.icc",
        "ITU-2020.icc",
        "ITU-709.icc",
        "ROMM RGB.icc",
        "Generic CMYK Profile.icc",
        "Generic Gray Profile.icc",
        "Generic Gray Gamma 2.2 Profile.icc",
    ];
    let mut loaded = Vec::new();
    for n in names {
        let p = Path::new(SYS).join(n);
        let Ok(bytes) = std::fs::read(&p) else {
            continue;
        };
        let prof = IccProfile::parse(&bytes).unwrap_or_else(|e| panic!("{n}: {e}"));
        loaded.push((n, p, prof));
    }
    let mut worst: f64 = 0.0;
    let mut worst_lut: f64 = 0.0;
    for (na, pa, a) in &loaded {
        for (nb, pb, b) in &loaded {
            if na == nb {
                continue;
            }
            // A CMYK source runs Generic CMYK's 9-point 4-D `mft1`
            // CLUT: on the grid nodes the two implementations agree to
            // one code (see `cmyk_clut_nodes_match_magick`); between
            // nodes ICC.1's "linear interpolation" leaves the simplex
            // scheme open, and the 9-point grid lets the two schemes
            // part by up to ~1.2 ΔE2000 in deep shadows.
            let lut_src = a.header().color_space == DataColorSpace::Cmyk;
            for intent in [
                RenderingIntent::Perceptual,
                RenderingIntent::RelativeColorimetric,
                RenderingIntent::Saturation,
                RenderingIntent::AbsoluteColorimetric,
            ] {
                let s = compare(a, b, intent, pa, pb);
                eprintln!(
                    "{na} -> {nb} {intent:?}: max {:.4} mean {:.4}",
                    s.max, s.mean
                );
                assert!(s.mean < 0.1, "{na} -> {nb} {intent:?}: mean {}", s.mean);
                if lut_src {
                    worst_lut = worst_lut.max(s.max);
                } else {
                    worst = worst.max(s.max);
                }
            }
        }
    }
    eprintln!("worst {worst}, worst with a CMYK CLUT source {worst_lut}");
    assert!(worst < 1.0, "{worst}");
    assert!(worst_lut < 1.25, "{worst_lut}");
}

/// CMYK inputs sitting exactly on Generic CMYK's 9-point CLUT grid:
/// with no interpolation in play, our result and `magick`'s must agree
/// to within a couple of 16-bit codes (and a few hundredths of ΔE2000
/// in the clipped deep-shadow corner).
#[test]
fn cmyk_clut_nodes_match_magick() {
    let cmyk = Path::new(SYS).join("Generic CMYK Profile.icc");
    let romm = Path::new(SYS).join("ROMM RGB.icc");
    if !magick_available() || !cmyk.exists() || !romm.exists() {
        eprintln!("magick or system profiles not found; skipping");
        return;
    }
    let a = IccProfile::parse(&std::fs::read(&cmyk).unwrap()).unwrap();
    let b = IccProfile::parse(&std::fs::read(&romm).unwrap()).unwrap();
    let mut input = Vec::new();
    for c in 0..9 {
        for m in 0..9 {
            for (y, k) in [(2, 6), (6, 7), (8, 8), (0, 0)] {
                for v in [c, m, y, k] {
                    input.push((v as f64 / 8.0 * 65535.0).round() as u16);
                }
            }
        }
    }
    let theirs = magick_profile(
        &input,
        DataColorSpace::Cmyk,
        DataColorSpace::Rgb,
        &cmyk,
        &romm,
        Some("Relative"),
    );
    let t = IccTransform::new(&a, &b, RenderingIntent::RelativeColorimetric).unwrap();
    let mut ours = vec![0u16; theirs.len()];
    t.transform_u16(&input, &mut ours).unwrap();
    let lab = IccProfile::lab_d50();
    let meas = IccTransform::new(&b, &lab, RenderingIntent::RelativeColorimetric).unwrap();
    let to_lab = |px: &[u16; 3]| {
        let d = px.map(|v| v as f64 / 65535.0);
        let mut l = [0.0; 3];
        meas.transform_pixel(&d, &mut l);
        [l[0] * 100.0, l[1] * 255.0 - 128.0, l[2] * 255.0 - 128.0]
    };
    let mut worst_de: f64 = 0.0;
    let mut worst_code = 0;
    for (o, m) in ours
        .as_chunks::<3>()
        .0
        .iter()
        .zip(theirs.as_chunks::<3>().0)
    {
        worst_de = worst_de.max(delta_e2000(to_lab(o), to_lab(m)));
        // Codes compared away from the clipped deep-shadow corner (both
        // implementations clip negative linear light there, at slightly
        // different points).
        if o.iter().chain(m).all(|&v| v > 1024) {
            for c in 0..3 {
                worst_code = worst_code.max((o[c] as i32 - m[c] as i32).abs());
            }
        }
    }
    eprintln!("CLUT nodes: max ΔE2000 {worst_de:.4}, max code difference {worst_code}");
    assert!(worst_de < 0.05, "{worst_de}");
    assert!(
        worst_code <= 2,
        "max code difference on CLUT nodes: {worst_code}"
    );
}
