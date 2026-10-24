//! LUT-based profiles synthesized from a known model — `mft2`
//! (lut16Type), `mft1` (lut8Type), `mAB ` / `mBA ` (lutAToBType /
//! lutBToAType with every processing element) — checked three ways:
//!
//! 1. against the analytic matrix/TRC model they were sampled from
//!    (our own transform, no external tool);
//! 2. round trips device → PCS → device;
//! 3. (when `magick` is on `PATH`) against `magick -profile a -profile
//!    b`, CIEDE2000 < 1.
#![cfg(all(feature = "icc", not(miri)))]

use std::path::PathBuf;
use std::process::Command;

use oxideav_pixfmt::colorspace::delta_e2000;
use oxideav_pixfmt::icc::{
    Curve, DataColorSpace, IccProfile, IccProfileBuilder, IccTransform, ProfileClass,
    RenderingIntent, lab_to_xyz, xyz_to_lab,
};

fn sig(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

fn s15(v: &mut Vec<u8>, x: f64) {
    v.extend_from_slice(&((x * 65536.0).round() as i32).to_be_bytes());
}

fn pad4(v: &mut Vec<u8>) {
    while !v.len().is_multiple_of(4) {
        v.push(0);
    }
}

/// Model: sRGB-shaped matrix/TRC from the synthesized profile.
struct Model {
    m: [[f64; 3]; 3],
    inv: [[f64; 3]; 3],
    trc: Curve,
}

fn model() -> Model {
    let p = IccProfile::srgb();
    let m = p.colorant_matrix().unwrap();
    // 3×3 inverse by cofactors.
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let inv = [
        [
            c(1, 1, 2, 2) / det,
            -c(0, 1, 2, 2) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            -c(1, 0, 2, 2) / det,
            c(0, 0, 2, 2) / det,
            -c(0, 0, 1, 2) / det,
        ],
        [
            c(1, 0, 2, 1) / det,
            -c(0, 0, 2, 1) / det,
            c(0, 0, 1, 1) / det,
        ],
    ];
    Model {
        m,
        inv,
        trc: Curve::srgb(),
    }
}

fn mv(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Lab normalisation for the v4 encoding (Tables 12/13) or the lut16
/// legacy one (Tables 42/43).
fn lab_norm(lab: [f64; 3], legacy: bool) -> [f64; 3] {
    let s = if legacy { 65280.0 / 65535.0 } else { 1.0 };
    [
        lab[0] / 100.0 * s,
        (lab[1] + 128.0) / 255.0 * s,
        (lab[2] + 128.0) / 255.0 * s,
    ]
}

fn lab_denorm(n: [f64; 3], legacy: bool) -> [f64; 3] {
    let s = if legacy { 65535.0 / 65280.0 } else { 1.0 };
    [
        n[0] * s * 100.0,
        n[1] * s * 255.0 - 128.0,
        n[2] * s * 255.0 - 128.0,
    ]
}

/// Sample `f` on an n-dimensional grid (first input slowest) into
/// 16-bit (or 8-bit) big-endian codes.
fn sample_grid(
    dims: &[usize],
    out: usize,
    bytes: usize,
    f: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Vec<u8> {
    let total: usize = dims.iter().product();
    let mut v = Vec::with_capacity(total * out * bytes);
    let mut idx = vec![0usize; dims.len()];
    for _ in 0..total {
        let x: Vec<f64> = idx
            .iter()
            .zip(dims)
            .map(|(&i, &g)| i as f64 / (g - 1) as f64)
            .collect();
        for y in f(&x) {
            let y = y.clamp(0.0, 1.0);
            if bytes == 2 {
                v.extend_from_slice(&((y * 65535.0).round() as u16).to_be_bytes());
            } else {
                v.push((y * 255.0).round() as u8);
            }
        }
        for d in (0..dims.len()).rev() {
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    v
}

fn curve_seq(v: &mut Vec<u8>, curves: &[Curve]) {
    for c in curves {
        v.extend_from_slice(&IccProfileBuilder::curve_tag(c));
        pad4(v);
    }
}

/// M curves, 3×3 matrix and offsets of an `mAB ` / `mBA `.
type MatrixElement<'a> = (&'a [Curve], [[f64; 3]; 3], [f64; 3]);

/// An `mAB ` with A curves → CLUT → M curves → matrix → B curves.
#[allow(clippy::too_many_arguments)]
fn mab(
    a_to_b: bool,
    in_ch: usize,
    out_ch: usize,
    a: &[Curve],
    grid: &[usize],
    prec: usize,
    clut: &dyn Fn(&[f64]) -> Vec<f64>,
    m: Option<MatrixElement<'_>>,
    b: &[Curve],
) -> Vec<u8> {
    let mut v = if a_to_b { b"mAB " } else { b"mBA " }.to_vec();
    v.extend_from_slice(&[0; 4]);
    v.extend_from_slice(&[in_ch as u8, out_ch as u8, 0, 0]);
    v.extend_from_slice(&[0; 20]); // offsets patched below
    let mut offs = [0u32; 5]; // B, matrix, M, CLUT, A
    offs[0] = v.len() as u32;
    curve_seq(&mut v, b);
    if let Some((mc, mat, off)) = m {
        offs[1] = v.len() as u32;
        for r in mat {
            for x in r {
                s15(&mut v, x);
            }
        }
        for x in off {
            s15(&mut v, x);
        }
        offs[2] = v.len() as u32;
        curve_seq(&mut v, mc);
    }
    offs[3] = v.len() as u32;
    let mut g16 = [0u8; 16];
    for (i, g) in grid.iter().enumerate() {
        g16[i] = *g as u8;
    }
    v.extend_from_slice(&g16);
    v.extend_from_slice(&[prec as u8, 0, 0, 0]);
    let n_out = out_ch;
    v.extend_from_slice(&sample_grid(grid, n_out, prec, clut));
    pad4(&mut v);
    offs[4] = v.len() as u32;
    curve_seq(&mut v, a);
    for (k, o) in offs.iter().enumerate() {
        v[12 + 4 * k..16 + 4 * k].copy_from_slice(&o.to_be_bytes());
    }
    v
}

/// An `mft2` / `mft1` with input tables, CLUT and output tables.
fn mft(
    sixteen: bool,
    in_ch: usize,
    out_ch: usize,
    in_tables: &dyn Fn(f64) -> f64,
    g: usize,
    clut: &dyn Fn(&[f64]) -> Vec<f64>,
    out_tables: &dyn Fn(f64) -> f64,
) -> Vec<u8> {
    let mut v = if sixteen { b"mft2" } else { b"mft1" }.to_vec();
    v.extend_from_slice(&[0; 4]);
    v.extend_from_slice(&[in_ch as u8, out_ch as u8, g as u8, 0]);
    for r in 0..3 {
        for c in 0..3 {
            s15(&mut v, if r == c { 1.0 } else { 0.0 });
        }
    }
    let (n, w) = if sixteen {
        (1024usize, 2usize)
    } else {
        (256, 1)
    };
    if sixteen {
        v.extend_from_slice(&(n as u16).to_be_bytes());
        v.extend_from_slice(&(n as u16).to_be_bytes());
    }
    let table = |v: &mut Vec<u8>, f: &dyn Fn(f64) -> f64| {
        for k in 0..n {
            let y = f(k as f64 / (n - 1) as f64).clamp(0.0, 1.0);
            if w == 2 {
                v.extend_from_slice(&((y * 65535.0).round() as u16).to_be_bytes());
            } else {
                v.push((y * 255.0).round() as u8);
            }
        }
    };
    for _ in 0..in_ch {
        table(&mut v, in_tables);
    }
    v.extend_from_slice(&sample_grid(&vec![g; in_ch], out_ch, w, clut));
    for _ in 0..out_ch {
        table(&mut v, out_tables);
    }
    v
}

fn rgb_lut_profile(kind: &str) -> IccProfile {
    // Device RGB → (A curves: u = cbrt(TRC(d))) → CLUT(u → Lab) → PCS.
    // The cube-root shaping keeps the CLUT close to linear in Lab, so a
    // 17-point grid represents the model well.
    let md = model();
    let srgb = md.trc.clone();
    let legacy = kind == "mft2";
    let fwd_m = md.m;
    let a2b_clut = move |x: &[f64]| -> Vec<f64> {
        let lin = [x[0].powi(3), x[1].powi(3), x[2].powi(3)];
        lab_norm(xyz_to_lab(mv(&fwd_m, lin)), legacy).to_vec()
    };
    let inv_m = md.inv;
    // PCS Lab → CLUT(Lab → cbrt(linear RGB)) → (A curves: TRC⁻¹(u³)).
    let b2a_clut = move |x: &[f64]| -> Vec<f64> {
        let lab = lab_denorm([x[0], x[1], x[2]], legacy);
        let lin = mv(&inv_m, lab_to_xyz(lab));
        lin.iter().map(|v| v.clamp(0.0, 1.0).cbrt()).collect()
    };
    let s1 = srgb.clone();
    let shape = move |d: f64| s1.eval(d).cbrt();
    let s2 = srgb.clone();
    let unshape = move |u: f64| s2.eval_inverse(u.powi(3));
    let tab = |f: &dyn Fn(f64) -> f64| -> Curve {
        Curve::Sampled((0..4096).map(|k| f(k as f64 / 4095.0)).collect())
    };
    let shape3 = [tab(&shape), tab(&shape), tab(&shape)];
    let unshape3 = [tab(&unshape), tab(&unshape), tab(&unshape)];
    let id = [Curve::Identity, Curve::Identity, Curve::Identity];
    let ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let (a2b, b2a) = match kind {
        "mAB" => (
            mab(
                true,
                3,
                3,
                &shape3,
                &[17, 17, 17],
                2,
                &a2b_clut,
                Some((&id, ident, [0.0; 3])),
                &id,
            ),
            mab(
                false,
                3,
                3,
                &unshape3,
                &[33, 33, 33],
                2,
                &b2a_clut,
                Some((&id, ident, [0.0; 3])),
                &id,
            ),
        ),
        "mAB8" => (
            mab(true, 3, 3, &shape3, &[17, 17, 17], 1, &a2b_clut, None, &id),
            mab(
                false,
                3,
                3,
                &unshape3,
                &[33, 33, 33],
                2,
                &b2a_clut,
                None,
                &id,
            ),
        ),
        _ => (
            mft(true, 3, 3, &shape, 17, &a2b_clut, &|y| y),
            mft(true, 3, 3, &|x| x, 33, &b2a_clut, &unshape),
        ),
    };
    IccProfileBuilder::new(
        ProfileClass::Display,
        DataColorSpace::Rgb,
        DataColorSpace::Lab,
    )
    .with_text(sig(b"desc"), &format!("synth {kind}"))
    .with_text(sig(b"cprt"), "No copyright, use freely")
    .with_xyz(sig(b"wtpt"), [0.9642, 1.0, 0.8249])
    .with_tag(sig(b"A2B0"), a2b)
    .with_tag(sig(b"B2A0"), b2a)
    .build_profile()
    .expect("synthesized LUT profile parses")
}

/// A CMYK output profile from the ISO 32000-2 §10.4.2 device formulas:
/// CMYK → RGB = 1 − min(1, C + K) …, RGB interpreted as sRGB.
fn cmyk_lut_profile() -> IccProfile {
    let md = model();
    let trc = md.trc.clone();
    let fm = md.m;
    let a2b_clut = move |x: &[f64]| -> Vec<f64> {
        let rgb = [
            1.0 - (x[0] + x[3]).min(1.0),
            1.0 - (x[1] + x[3]).min(1.0),
            1.0 - (x[2] + x[3]).min(1.0),
        ];
        let lin = [trc.eval(rgb[0]), trc.eval(rgb[1]), trc.eval(rgb[2])];
        lab_norm(xyz_to_lab(mv(&fm, lin)), false).to_vec()
    };
    let trc2 = md.trc.clone();
    let im = md.inv;
    let b2a_clut = move |x: &[f64]| -> Vec<f64> {
        let lab = lab_denorm([x[0], x[1], x[2]], false);
        let lin = mv(&im, lab_to_xyz(lab));
        let rgb: Vec<f64> = lin
            .iter()
            .map(|v| trc2.eval_inverse(v.clamp(0.0, 1.0)))
            .collect();
        // §10.4.2.4 with BG(k) = k, UCR(k) = k.
        let (c, m, y) = (1.0 - rgb[0], 1.0 - rgb[1], 1.0 - rgb[2]);
        let k = c.min(m).min(y);
        vec![c - k, m - k, y - k, k]
    };
    let id3 = [Curve::Identity, Curve::Identity, Curve::Identity];
    let id4 = [
        Curve::Identity,
        Curve::Identity,
        Curve::Identity,
        Curve::Identity,
    ];
    let a2b = mab(
        true,
        4,
        3,
        &id4,
        &[17, 17, 17, 17],
        2,
        &a2b_clut,
        None,
        &id3,
    );
    let b2a = mab(false, 3, 4, &id4, &[33, 33, 33], 2, &b2a_clut, None, &id3);
    IccProfileBuilder::new(
        ProfileClass::Output,
        DataColorSpace::Cmyk,
        DataColorSpace::Lab,
    )
    .with_text(sig(b"desc"), "synth CMYK")
    .with_text(sig(b"cprt"), "No copyright, use freely")
    .with_xyz(sig(b"wtpt"), [0.9642, 1.0, 0.8249])
    .with_tag(sig(b"A2B0"), a2b.clone())
    .with_tag(sig(b"A2B1"), a2b.clone())
    .with_tag(sig(b"A2B2"), a2b)
    .with_tag(sig(b"B2A0"), b2a.clone())
    .with_tag(sig(b"B2A1"), b2a.clone())
    .with_tag(sig(b"B2A2"), b2a)
    .build_profile()
    .expect("synthesized CMYK profile parses")
}

fn probe_rgb() -> Vec<u16> {
    let mut v = Vec::new();
    let lv = [0u16, 4000, 13107, 26214, 39321, 52428, 60000, 65535];
    for &r in &lv {
        for &g in &lv {
            for &b in &lv {
                v.extend_from_slice(&[r, g, b]);
            }
        }
    }
    v
}

fn lab_of(t: &IccTransform, d: &[u16]) -> [f64; 3] {
    let x: Vec<f64> = d.iter().map(|&c| c as f64 / 65535.0).collect();
    let mut o = [0.0; 3];
    t.transform_pixel(&x, &mut o);
    [o[0] * 100.0, o[1] * 255.0 - 128.0, o[2] * 255.0 - 128.0]
}

#[test]
fn lut_profiles_agree_with_the_matrix_model() {
    let srgb = IccProfile::srgb();
    let lab = IccProfile::lab_d50();
    let ref_lab = IccTransform::new(&srgb, &lab, RenderingIntent::Perceptual).unwrap();
    for kind in ["mAB", "mAB8", "mft2"] {
        let p = rgb_lut_profile(kind);
        let to_lab = IccTransform::new(&p, &lab, RenderingIntent::Perceptual).unwrap();
        let input = probe_rgb();
        let mut worst: f64 = 0.0;
        for px in input.as_chunks::<3>().0.iter() {
            let a = lab_of(&ref_lab, px);
            let b = lab_of(&to_lab, px);
            worst = worst.max(delta_e2000(a, b));
        }
        eprintln!("{kind}: AToB vs matrix model max ΔE2000 {worst:.4}");
        // The residual is the model's own CLUT sampling error (17³ grid).
        assert!(worst < 1.0, "{kind}: {worst}");
        // Round trip through the profile's own BToA.
        let rt = IccTransform::new(&p, &p, RenderingIntent::Perceptual).unwrap();
        let mut out = vec![0u16; input.len()];
        rt.transform_u16(&input, &mut out).unwrap();
        let mut worst_rt: f64 = 0.0;
        for (i, o) in input
            .as_chunks::<3>()
            .0
            .iter()
            .zip(out.as_chunks::<3>().0.iter())
        {
            worst_rt = worst_rt.max(delta_e2000(lab_of(&ref_lab, i), lab_of(&ref_lab, o)));
        }
        eprintln!("{kind}: device round trip max ΔE2000 {worst_rt:.4}");
        // Worst cases are dark saturated colours next to the gamut
        // boundary, where the BToA CLUT samples clamp.
        assert!(worst_rt < 2.5, "{kind} round trip: {worst_rt}");
    }
}

#[test]
fn cmyk_lut_profile_round_trips() {
    let cmyk = cmyk_lut_profile();
    let srgb = IccProfile::srgb();
    let lab = IccProfile::lab_d50();
    let ref_lab = IccTransform::new(&srgb, &lab, RenderingIntent::RelativeColorimetric).unwrap();
    let to_cmyk = IccTransform::new(&srgb, &cmyk, RenderingIntent::RelativeColorimetric).unwrap();
    let back = IccTransform::new(&cmyk, &srgb, RenderingIntent::RelativeColorimetric).unwrap();
    assert_eq!(to_cmyk.output_channels(), 4);
    let input = probe_rgb();
    let mut k = vec![0u16; input.len() / 3 * 4];
    to_cmyk.transform_u16(&input, &mut k).unwrap();
    let mut rgb = vec![0u16; input.len()];
    back.transform_u16(&k, &mut rgb).unwrap();
    let mut worst: f64 = 0.0;
    for (i, o) in input
        .as_chunks::<3>()
        .0
        .iter()
        .zip(rgb.as_chunks::<3>().0.iter())
    {
        worst = worst.max(delta_e2000(lab_of(&ref_lab, i), lab_of(&ref_lab, o)));
    }
    eprintln!("sRGB → CMYK(mAB/mBA) → sRGB max ΔE2000 {worst:.4}");
    // Two CLUT samplings of a model with kinks (min(1, C + K) and
    // K = min(C, M, Y)) bound how close the round trip can get.
    assert!(worst < 2.5, "{worst}");
    // Pure white maps to (almost) no ink and pure black to (almost) K
    // only (BG = UCR = k); the PCS white / black sit between CLUT nodes
    // on the a*/b* axes, so a few hundred codes of interpolation spill
    // from out-of-gamut neighbours remain.
    let mut px = [0u16; 4];
    to_cmyk
        .transform_u16(&[65535, 65535, 65535], &mut px)
        .unwrap();
    assert!(px.iter().all(|&v| v < 600), "{px:?}");
    to_cmyk.transform_u16(&[0, 0, 0], &mut px).unwrap();
    assert!(
        px[3] > 64000 && px[0] < 600 && px[1] < 600 && px[2] < 600,
        "{px:?}"
    );
}

fn magick_available() -> bool {
    Command::new("magick")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("oxideav-pixfmt-iccl-{}-{name}", std::process::id()))
}

fn magick(input: &[u16], src: (&str, &PathBuf), dst: (&str, &PathBuf), px: usize) -> Vec<u16> {
    let i = tmp(&format!("{}-{}.in", src.0, dst.0));
    let o = tmp(&format!("{}-{}.out", src.0, dst.0));
    std::fs::write(
        &i,
        input
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    let st = Command::new("magick")
        .args([
            "-size",
            &format!("{px}x1"),
            "-depth",
            "16",
            "-endian",
            "MSB",
        ])
        .arg(format!("{}:{}", src.0, i.display()))
        .arg("-profile")
        .arg(src.1)
        .args(["-intent", "Relative", "-profile"])
        .arg(dst.1)
        .args(["-depth", "16", "-endian", "MSB"])
        .arg(format!("{}:{}", dst.0, o.display()))
        .status()
        .unwrap();
    assert!(st.success());
    let b = std::fs::read(&o).unwrap();
    let _ = std::fs::remove_file(&i);
    let _ = std::fs::remove_file(&o);
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_be_bytes(*c))
        .collect()
}

#[test]
fn lut_profiles_match_magick() {
    if !magick_available() {
        eprintln!("magick not found; skipping");
        return;
    }
    let srgb = IccProfile::srgb();
    let lab = IccProfile::lab_d50();
    let ps = tmp("srgb.icc");
    std::fs::write(&ps, srgb.as_bytes()).unwrap();
    let input = probe_rgb();
    let px = input.len() / 3;
    for kind in ["mAB", "mAB8", "mft2"] {
        let p = rgb_lut_profile(kind);
        let pp = tmp(&format!("{kind}.icc"));
        std::fs::write(&pp, p.as_bytes()).unwrap();
        for (a, pa, b, pb) in [(&srgb, &ps, &p, &pp), (&p, &pp, &srgb, &ps)] {
            let theirs = magick(&input, ("rgb", pa), ("rgb", pb), px);
            let t = IccTransform::new(a, b, RenderingIntent::RelativeColorimetric).unwrap();
            let mut ours = vec![0u16; input.len()];
            t.transform_u16(&input, &mut ours).unwrap();
            let meas = IccTransform::new(b, &lab, RenderingIntent::RelativeColorimetric).unwrap();
            let mut worst: f64 = 0.0;
            for (o, m) in ours
                .as_chunks::<3>()
                .0
                .iter()
                .zip(theirs.as_chunks::<3>().0.iter())
            {
                worst = worst.max(delta_e2000(lab_of(&meas, o), lab_of(&meas, m)));
            }
            eprintln!(
                "{kind} ({}): max ΔE2000 vs magick {worst:.4}",
                if std::ptr::eq(a, &srgb) {
                    "into"
                } else {
                    "out of"
                }
            );
            assert!(worst < 1.0, "{kind}: {worst}");
        }
        let _ = std::fs::remove_file(&pp);
    }
    // CMYK.
    let c = cmyk_lut_profile();
    let pc = tmp("cmyk.icc");
    std::fs::write(&pc, c.as_bytes()).unwrap();
    let theirs = magick(&input, ("rgb", &ps), ("cmyk", &pc), px);
    let t = IccTransform::new(&srgb, &c, RenderingIntent::RelativeColorimetric).unwrap();
    let mut ours = vec![0u16; px * 4];
    t.transform_u16(&input, &mut ours).unwrap();
    let meas = IccTransform::new(&c, &lab, RenderingIntent::RelativeColorimetric).unwrap();
    let mut worst: f64 = 0.0;
    for (o, m) in ours
        .as_chunks::<4>()
        .0
        .iter()
        .zip(theirs.as_chunks::<4>().0.iter())
    {
        worst = worst.max(delta_e2000(lab_of(&meas, o), lab_of(&meas, m)));
    }
    eprintln!("sRGB → synth CMYK: max ΔE2000 vs magick {worst:.4}");
    assert!(worst < 1.0, "cmyk: {worst}");
    let _ = std::fs::remove_file(&pc);
    let _ = std::fs::remove_file(&ps);
}
