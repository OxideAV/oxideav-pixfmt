//! `multiProcessElementsType` (`DToBx` / `BToDx`, ICC.1:2022 §10.16)
//! evaluation: synthesized float profiles against the matrix/TRC model
//! they encode, the §8.10.2 precedence and fallback, the unclipped float
//! range, DeviceLink `DToB0`, and the DToB3 / BToD3 absolute tags.
#![cfg(feature = "icc")]

use oxideav_pixfmt::icc::{
    DataColorSpace, IccProfile, IccProfileBuilder, IccTransform, ProfileClass, RenderingIntent,
};

fn sig(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

fn pad4(v: &mut Vec<u8>) {
    while !v.len().is_multiple_of(4) {
        v.push(0);
    }
}

fn put_f32(v: &mut Vec<u8>, x: f64) {
    v.extend_from_slice(&(x as f32).to_be_bytes());
}

/// Container with a positions table (Table 55 / Table 57 share the
/// layout: 12-byte head, then `n` (offset, size) pairs, then data).
fn positioned(head: &[u8], items: &[Vec<u8>]) -> Vec<u8> {
    let mut v = head.to_vec();
    let mut off = head.len() + 8 * items.len();
    let mut data = Vec::new();
    for it in items {
        v.extend_from_slice(&(off as u32).to_be_bytes());
        v.extend_from_slice(&(it.len() as u32).to_be_bytes());
        let mut it = it.clone();
        pad4(&mut it);
        off += it.len();
        data.extend_from_slice(&it);
    }
    v.extend_from_slice(&data);
    v
}

fn mpet(in_ch: u16, out_ch: u16, elements: &[Vec<u8>]) -> Vec<u8> {
    let mut head = b"mpet\0\0\0\0".to_vec();
    head.extend_from_slice(&in_ch.to_be_bytes());
    head.extend_from_slice(&out_ch.to_be_bytes());
    head.extend_from_slice(&(elements.len() as u32).to_be_bytes());
    positioned(&head, elements)
}

fn cvst(curves: &[Vec<u8>]) -> Vec<u8> {
    let n = curves.len() as u16;
    let mut head = b"cvst\0\0\0\0".to_vec();
    head.extend_from_slice(&n.to_be_bytes());
    head.extend_from_slice(&n.to_be_bytes());
    positioned(&head, curves)
}

fn curf(breaks: &[f64], segments: &[Vec<u8>]) -> Vec<u8> {
    let mut v = b"curf\0\0\0\0".to_vec();
    v.extend_from_slice(&(segments.len() as u16).to_be_bytes());
    v.extend_from_slice(&[0, 0]);
    for &b in breaks {
        put_f32(&mut v, b);
    }
    for s in segments {
        v.extend_from_slice(s);
    }
    v
}

fn parf(function: u16, params: &[f64]) -> Vec<u8> {
    let mut v = b"parf\0\0\0\0".to_vec();
    v.extend_from_slice(&function.to_be_bytes());
    v.extend_from_slice(&[0, 0]);
    for &p in params {
        put_f32(&mut v, p);
    }
    v
}

fn samf(values: &[f64]) -> Vec<u8> {
    let mut v = b"samf\0\0\0\0".to_vec();
    v.extend_from_slice(&(values.len() as u32).to_be_bytes());
    for &x in values {
        put_f32(&mut v, x);
    }
    v
}

fn matf(rows: &[Vec<f64>], offset: &[f64]) -> Vec<u8> {
    let (q, p) = (rows.len() as u16, rows[0].len() as u16);
    let mut v = b"matf\0\0\0\0".to_vec();
    v.extend_from_slice(&p.to_be_bytes());
    v.extend_from_slice(&q.to_be_bytes());
    for r in rows {
        for &e in r {
            put_f32(&mut v, e);
        }
    }
    for &o in offset {
        put_f32(&mut v, o);
    }
    v
}

fn clut(grid: &[u8], q: u16, data: &[f64]) -> Vec<u8> {
    let mut v = b"clut\0\0\0\0".to_vec();
    v.extend_from_slice(&(grid.len() as u16).to_be_bytes());
    v.extend_from_slice(&q.to_be_bytes());
    let mut g = [0u8; 16];
    g[..grid.len()].copy_from_slice(grid);
    v.extend_from_slice(&g);
    for &x in data {
        put_f32(&mut v, x);
    }
    v
}

/// The sRGB decode curve as two formula segments split at 0.04045.
fn srgb_decode_curf() -> Vec<u8> {
    curf(
        &[0.04045],
        &[
            parf(0, &[1.0, 1.0 / 12.92, 0.0, 0.0]),
            parf(0, &[2.4, 1.0 / 1.055, 0.055 / 1.055, 0.0]),
        ],
    )
}

/// The sRGB encode curve: `12.92·X` up to 0.0031308, then
/// `(1.055^2.4·X)^(1/2.4) − 0.055`.
fn srgb_encode_curf() -> Vec<u8> {
    curf(
        &[0.0031308],
        &[
            parf(0, &[1.0, 12.92, 0.0, 0.0]),
            parf(0, &[1.0 / 2.4, 1.055f64.powf(2.4), 0.0, -0.055]),
        ],
    )
}

fn rows(m: [[f64; 3]; 3]) -> Vec<Vec<f64>> {
    m.iter().map(|r| r.to_vec()).collect()
}

fn invert(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = |r: usize, k: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
        m[r1][k1] * m[r2][k2] - m[r1][k2] * m[r2][k1]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    let mut inv = [[0.0; 3]; 3];
    for (r, row) in inv.iter_mut().enumerate() {
        for (k, v) in row.iter_mut().enumerate() {
            *v = c(k, r) / det;
        }
    }
    inv
}

/// A display profile whose DToB0 / BToD0 encode the sRGB matrix/TRC
/// model in float, next to a deliberately *wrong* integer model (gray
/// TRC colorants of the identity) so the test can tell which one runs.
fn float_srgb(extra: &[(u32, Vec<u8>)]) -> IccProfile {
    let m = IccProfile::srgb().colorant_matrix().unwrap();
    let to_pcs = mpet(
        3,
        3,
        &[
            cvst(&vec![srgb_decode_curf(); 3]),
            matf(&rows(m), &[0.0; 3]),
        ],
    );
    let from_pcs = mpet(
        3,
        3,
        &[
            matf(&rows(invert(m)), &[0.0; 3]),
            cvst(&vec![srgb_encode_curf(); 3]),
        ],
    );
    let mut b = IccProfileBuilder::new(
        ProfileClass::Display,
        DataColorSpace::Rgb,
        DataColorSpace::Xyz,
    )
    .with_xyz(sig(b"wtpt"), [0.9642, 1.0, 0.8249])
    .with_xyz(sig(b"rXYZ"), [1.0, 0.0, 0.0])
    .with_xyz(sig(b"gXYZ"), [0.0, 1.0, 0.0])
    .with_xyz(sig(b"bXYZ"), [0.0, 0.0, 1.0])
    .with_curve(sig(b"rTRC"), &oxideav_pixfmt::icc::Curve::Identity)
    .with_curve(sig(b"gTRC"), &oxideav_pixfmt::icc::Curve::Identity)
    .with_curve(sig(b"bTRC"), &oxideav_pixfmt::icc::Curve::Identity)
    .with_tag(sig(b"D2B0"), to_pcs)
    .with_tag(sig(b"B2D0"), from_pcs);
    for (s, d) in extra {
        b = b.with_tag(*s, d.clone());
    }
    b.build_profile().unwrap()
}

fn probes() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for r in 0..5 {
        for g in 0..5 {
            for b in 0..5 {
                v.push([r as f64 / 4.0, g as f64 / 4.0, b as f64 / 4.0]);
            }
        }
    }
    v
}

fn max_diff(a: &IccTransform, b: &IccTransform) -> f64 {
    let mut worst: f64 = 0.0;
    for p in probes() {
        let (mut x, mut y) = ([0.0; 3], [0.0; 3]);
        a.transform_pixel(&p, &mut x);
        b.transform_pixel(&p, &mut y);
        for c in 0..3 {
            // The float tags do not clip (P3 colours outside sRGB come
            // out negative or above 1); compare what a device can show.
            worst = worst.max((x[c].clamp(0.0, 1.0) - y[c].clamp(0.0, 1.0)).abs());
        }
    }
    worst
}

#[test]
fn dtob_and_btod_take_precedence_over_matrix_trc() {
    let f = float_srgb(&[]);
    assert!(f.has_float_to_pcs(RenderingIntent::Perceptual));
    assert!(f.has_float_from_pcs(RenderingIntent::Perceptual));
    // §8.10.2 a): only the intent's own float tag counts.
    assert!(!f.has_float_to_pcs(RenderingIntent::RelativeColorimetric));
    let p3 = IccProfile::display_p3();
    let srgb = IccProfile::srgb();
    let pi = RenderingIntent::Perceptual;
    let ours = IccTransform::new(&f, &p3, pi).unwrap();
    let reference = IccTransform::new(&srgb, &p3, pi).unwrap();
    let d = max_diff(&ours, &reference);
    assert!(d < 2e-5, "DToB0 vs sRGB matrix/TRC: {d}");
    let back = IccTransform::new(&p3, &f, pi).unwrap();
    let reference = IccTransform::new(&p3, &srgb, pi).unwrap();
    let d = max_diff(&back, &reference);
    assert!(d < 2e-5, "BToD0 vs sRGB matrix/TRC: {d}");
    // Relative colorimetric has no DToB1, so the (identity) matrix/TRC
    // model runs: clearly different.
    let rel = IccTransform::new(&f, &p3, RenderingIntent::RelativeColorimetric).unwrap();
    let reference = IccTransform::new(&srgb, &p3, RenderingIntent::RelativeColorimetric).unwrap();
    assert!(max_diff(&rel, &reference) > 0.05);
}

#[test]
fn float_tags_keep_values_outside_the_unit_range() {
    let f = float_srgb(&[]);
    let t = IccTransform::new(&f, &f, RenderingIntent::Perceptual).unwrap();
    // §10.16.2.1: no clipping between elements; the f32 / pixel paths
    // keep extended values.
    for px in [[1.25, -0.1, 0.5], [-0.5, 0.0, 2.0], [0.3, 0.6, 0.9]] {
        let mut o = [0.0; 3];
        t.transform_pixel(&px, &mut o);
        for c in 0..3 {
            assert!((o[c] - px[c]).abs() < 1e-5, "{px:?} → {o:?}");
        }
    }
    let src = [1.25f32, -0.1, 0.5];
    let mut dst = [0f32; 3];
    t.transform_f32(&src, &mut dst).unwrap();
    assert!((dst[0] - 1.25).abs() < 1e-5 && (dst[1] + 0.1).abs() < 1e-5);
    // The integer paths clamp at quantisation.
    let mut d16 = [0u16; 3];
    t.transform_u16(&[65535, 0, 32768], &mut d16).unwrap();
    assert_eq!(d16[0], 65535);
    assert_eq!(d16[1], 0);
    assert!((d16[2] as i32 - 32768).abs() <= 1);
}

#[test]
fn unknown_element_falls_back_to_the_integer_tags() {
    let mut bad = mpet(3, 3, &[b"xyzw\0\0\0\0\0\x03\0\x03".to_vec()]);
    bad.extend_from_slice(&[0; 4]);
    let p = IccProfileBuilder::srgb_like_with(&[(sig(b"D2B0"), bad.clone())]);
    assert!(!p.has_float_to_pcs(RenderingIntent::Perceptual));
    let t = IccTransform::new(&p, &IccProfile::srgb(), RenderingIntent::Perceptual).unwrap();
    let mut o = [0.0; 3];
    t.transform_pixel(&[0.2, 0.4, 0.6], &mut o);
    for (got, want) in o.iter().zip([0.2, 0.4, 0.6]) {
        assert!((got - want).abs() < 1e-6);
    }
    // A truncated tag is ignored the same way.
    let p = IccProfileBuilder::srgb_like_with(&[(sig(b"D2B0"), bad[..10].to_vec())]);
    assert!(!p.has_float_to_pcs(RenderingIntent::Perceptual));
}

/// Helper trait: an sRGB matrix/TRC profile plus extra raw tags.
trait SrgbLike {
    fn srgb_like_with(extra: &[(u32, Vec<u8>)]) -> IccProfile;
}

impl SrgbLike for IccProfileBuilder {
    fn srgb_like_with(extra: &[(u32, Vec<u8>)]) -> IccProfile {
        let base = IccProfile::srgb();
        let mut b = IccProfileBuilder::new(
            ProfileClass::Display,
            DataColorSpace::Rgb,
            DataColorSpace::Xyz,
        );
        for t in base.tags() {
            b = b.with_tag(t.signature, base.tag_data(t.signature).unwrap().to_vec());
        }
        for (s, d) in extra {
            b = b.with_tag(*s, d.clone());
        }
        b.build_profile().unwrap()
    }
}

/// sRGB with a sampled decode-curve middle, a float CLUT and the
/// pass-through elements in its DToB1.
fn sampled_clut_profile() -> IccProfile {
    // sRGB decode with the middle of the curve sampled: formula below 0,
    // 256 samples over (0, 1], formula above 1.
    let samples: Vec<f64> = (1..=256)
        .map(|j| {
            let x: f64 = j as f64 / 256.0;
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let sampled = curf(
        &[0.0, 1.0],
        &[
            parf(0, &[1.0, 1.0 / 12.92, 0.0, 0.0]),
            samf(&samples),
            parf(0, &[2.4, 1.0 / 1.055, 0.055 / 1.055, 0.0]),
        ],
    );
    // A 2×2×2 float CLUT that swaps R and B (exact under multilinear
    // interpolation), undone by the matrix.
    let mut data = Vec::new();
    for r in 0..2 {
        for g in 0..2 {
            for b in 0..2 {
                data.extend_from_slice(&[b as f64, g as f64, r as f64]);
            }
        }
    }
    let m = IccProfile::srgb().colorant_matrix().unwrap();
    let swapped: Vec<Vec<f64>> = m.iter().map(|r| vec![r[2], r[1], r[0]]).collect();
    let to_pcs = mpet(
        3,
        3,
        &[
            cvst(&vec![sampled; 3]),
            b"bACS\0\0\0\0\0\x03\0\x03\0\0\0\0".to_vec(),
            clut(&[2, 2, 2], 3, &data),
            matf(&swapped, &[0.0; 3]),
            b"eACS\0\0\0\0\0\x03\0\x03\0\0\0\0".to_vec(),
        ],
    );
    IccProfileBuilder::srgb_like_with(&[(sig(b"D2B1"), to_pcs)])
}

#[test]
fn sampled_segments_and_float_clut() {
    let p = sampled_clut_profile();
    assert!(p.has_float_to_pcs(RenderingIntent::RelativeColorimetric));
    let rel = RenderingIntent::RelativeColorimetric;
    let ours = IccTransform::new(&p, &IccProfile::display_p3(), rel).unwrap();
    let reference = IccTransform::new(&IccProfile::srgb(), &IccProfile::display_p3(), rel).unwrap();
    let d = max_diff(&ours, &reference);
    // Linear interpolation over 256 samples of the decode curve.
    assert!(d < 5e-5, "sampled curve + CLUT: {d}");
}

#[test]
fn absolute_intent_uses_dtob3_without_rescaling() {
    // A print-like media white: ICC-absolute through the relative tags
    // rescales by it, DToB3 is taken as absolute already.
    let m = IccProfile::srgb().colorant_matrix().unwrap();
    let d2b = mpet(
        3,
        3,
        &[
            cvst(&vec![srgb_decode_curf(); 3]),
            matf(&rows(m), &[0.0; 3]),
        ],
    );
    let base = IccProfileBuilder::srgb_like_with(&[]);
    let mut b = IccProfileBuilder::new(
        ProfileClass::Output,
        DataColorSpace::Rgb,
        DataColorSpace::Xyz,
    );
    for t in base.tags() {
        if t.signature != sig(b"wtpt") {
            b = b.with_tag(t.signature, base.tag_data(t.signature).unwrap().to_vec());
        }
    }
    let b = b.with_xyz(sig(b"wtpt"), [0.8, 0.85, 0.7]);
    let without = b.clone().build_profile().unwrap();
    let with = b.with_tag(sig(b"D2B3"), d2b).build_profile().unwrap();
    assert!(with.has_float_to_pcs(RenderingIntent::AbsoluteColorimetric));
    let srgb = IccProfile::srgb();
    let abs = RenderingIntent::AbsoluteColorimetric;
    let rel = RenderingIntent::RelativeColorimetric;
    let t3 = IccTransform::new(&with, &srgb, abs).unwrap();
    let t_rel = IccTransform::new(&without, &srgb, rel).unwrap();
    let t_abs = IccTransform::new(&without, &srgb, abs).unwrap();
    // DToB3 carries the same model as the relative tags and sRGB's media
    // white is the PCS white, so no rescaling happens at all.
    assert!(max_diff(&t3, &t_rel) < 2e-5);
    // Without DToB3 the media white rescales the result.
    assert!(max_diff(&t_abs, &t_rel) > 0.05);
}

#[test]
fn device_link_prefers_dtob0() {
    // RGB → CMY link: the float DToB0 computes 1 − v, the integer AToB0
    // (identity B curves) would pass values through unchanged.
    let neg = mpet(
        3,
        3,
        &[matf(
            &[
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -1.0, 0.0],
                vec![0.0, 0.0, -1.0],
            ],
            &[1.0, 1.0, 1.0],
        )],
    );
    let id = oxideav_pixfmt::icc::Curve::Identity;
    let a2b = IccProfileBuilder::b_curves_lut_tag(true, &[id.clone(), id.clone(), id]);
    let builder = IccProfileBuilder::new(
        ProfileClass::DeviceLink,
        DataColorSpace::Rgb,
        DataColorSpace::Cmy,
    )
    .with_tag(sig(b"A2B0"), a2b);
    let plain = builder.clone().build_profile().unwrap();
    let float = builder.with_tag(sig(b"D2B0"), neg).build_profile().unwrap();
    let mut o = [0.0; 3];
    IccTransform::device_link(&plain)
        .unwrap()
        .transform_pixel(&[0.2, 0.5, 0.9], &mut o);
    assert!(
        (o[0] - 0.2).abs() < 1e-6 && (o[2] - 0.9).abs() < 1e-6,
        "{o:?}"
    );
    IccTransform::device_link(&float)
        .unwrap()
        .transform_pixel(&[0.2, 0.5, 0.9], &mut o);
    assert!(
        (o[0] - 0.8).abs() < 1e-6 && (o[2] - 0.1).abs() < 1e-6,
        "{o:?}"
    );
}

#[test]
fn lab_pcs_float_tags_use_actual_lab_values() {
    // A Lab-PCS "profile" whose float tags are the identity (device =
    // L*/100, (a*, b*)/256 + 0.5 would be the integer encoding; the
    // float encoding is the actual L*, a*, b*, §6.3.4.2).
    let to_pcs = mpet(
        3,
        3,
        &[matf(
            &[
                vec![100.0, 0.0, 0.0],
                vec![0.0, 255.0, 0.0],
                vec![0.0, 0.0, 255.0],
            ],
            &[0.0, -127.5, -127.5],
        )],
    );
    let from_pcs = mpet(
        3,
        3,
        &[matf(
            &[
                vec![0.01, 0.0, 0.0],
                vec![0.0, 1.0 / 255.0, 0.0],
                vec![0.0, 0.0, 1.0 / 255.0],
            ],
            &[0.0, 0.5, 0.5],
        )],
    );
    let p = IccProfileBuilder::new(
        ProfileClass::ColorSpace,
        DataColorSpace::Rgb,
        DataColorSpace::Lab,
    )
    .with_xyz(sig(b"wtpt"), [0.9642, 1.0, 0.8249])
    .with_tag(sig(b"D2B0"), to_pcs)
    .with_tag(sig(b"B2D0"), from_pcs)
    .build_profile()
    .unwrap();
    let lab = IccProfile::lab_d50();
    let t = IccTransform::new(&p, &lab, RenderingIntent::Perceptual).unwrap();
    let back = IccTransform::new(&lab, &p, RenderingIntent::Perceptual).unwrap();
    for px in [[0.5, 0.5, 0.5], [0.7, 0.6, 0.3], [0.2, 0.45, 0.55]] {
        let (mut l, mut o) = ([0.0; 3], [0.0; 3]);
        t.transform_pixel(&px, &mut l);
        back.transform_pixel(&l, &mut o);
        for c in 0..3 {
            assert!((o[c] - px[c]).abs() < 1e-5, "{px:?} → {l:?} → {o:?}");
        }
    }
}

#[cfg(not(miri))]
fn magick_available() -> bool {
    std::process::Command::new("magick")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `magick -size Nx1 rgb:in -profile a -profile b rgb:out` on 16-bit
/// big-endian samples.
#[cfg(not(miri))]
fn magick_rgb(input: &[u16], a: &IccProfile, b: &IccProfile, intent: &str) -> Vec<u16> {
    let dir = std::env::temp_dir();
    let tag = format!("oxideav-pixfmt-mpet-{}-{intent}", std::process::id());
    let (pa, pb) = (
        dir.join(format!("{tag}-a.icc")),
        dir.join(format!("{tag}-b.icc")),
    );
    let (pi, po) = (
        dir.join(format!("{tag}.in")),
        dir.join(format!("{tag}.out")),
    );
    std::fs::write(&pa, a.as_bytes()).unwrap();
    std::fs::write(&pb, b.as_bytes()).unwrap();
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_be_bytes()).collect();
    std::fs::write(&pi, bytes).unwrap();
    let st = std::process::Command::new("magick")
        .args(["-size", &format!("{}x1", input.len() / 3)])
        .args(["-depth", "16", "-endian", "MSB"])
        .arg(format!("rgb:{}", pi.display()))
        .arg("-profile")
        .arg(&pa)
        .args(["-intent", intent])
        .arg("-profile")
        .arg(&pb)
        .args(["-depth", "16", "-endian", "MSB"])
        .arg(format!("rgb:{}", po.display()))
        .status()
        .unwrap();
    assert!(st.success());
    let out = std::fs::read(&po).unwrap();
    for p in [pa, pb, pi, po] {
        let _ = std::fs::remove_file(p);
    }
    out.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_be_bytes(*c))
        .collect()
}

#[cfg(not(miri))]
#[test]
fn float_profile_matches_magick() {
    if !magick_available() {
        eprintln!("magick not found; skipping");
        return;
    }
    let f = float_srgb(&[]);
    let p3 = IccProfile::display_p3();
    let input: Vec<u16> = probes()
        .iter()
        .flatten()
        .map(|v| (v * 65535.0).round() as u16)
        .collect();
    let s = sampled_clut_profile();
    let cases = [
        (&f, &p3, "Perceptual", RenderingIntent::Perceptual),
        (&p3, &f, "Perceptual", RenderingIntent::Perceptual),
        (&s, &p3, "Relative", RenderingIntent::RelativeColorimetric),
    ];
    for (a, b, name, intent) in cases {
        let theirs = magick_rgb(&input, a, b, name);
        let t = IccTransform::new(a, b, intent).unwrap();
        let mut ours = vec![0u16; input.len()];
        t.transform_u16(&input, &mut ours).unwrap();
        let worst = ours
            .iter()
            .zip(&theirs)
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap();
        eprintln!("mpet vs magick ({name}): max {worst} codes of 65535");
        assert!(worst <= 2, "{worst}");
    }
}
