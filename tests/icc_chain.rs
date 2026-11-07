//! `IccTransform::chain`: more than two profiles, Abstract profiles
//! (ICC.1:2022 §8.8, §8.10.3) and device profiles in the middle.
#![cfg(feature = "icc")]

use oxideav_pixfmt::icc::{
    Curve, DataColorSpace, IccError, IccProfile, IccProfileBuilder, IccTransform, ProfileClass,
    RenderingIntent, TransformOptions,
};

fn sig(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

const INTENTS: [RenderingIntent; 4] = [
    RenderingIntent::Perceptual,
    RenderingIntent::RelativeColorimetric,
    RenderingIntent::Saturation,
    RenderingIntent::AbsoluteColorimetric,
];

fn opts(i: RenderingIntent) -> TransformOptions {
    TransformOptions::new().with_intent(i)
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

fn run(t: &IccTransform, px: [f64; 3]) -> [f64; 3] {
    let mut o = [0.0; 3];
    t.transform_pixel(&px, &mut o);
    o
}

/// An Abstract profile `Lab → Lab` whose AToB0 is an `mAB ` with only B
/// curves (one per channel).
fn abstract_lab(curves: [Curve; 3]) -> IccProfile {
    IccProfileBuilder::new(
        ProfileClass::Abstract,
        DataColorSpace::Lab,
        DataColorSpace::Lab,
    )
    .with_xyz(sig(b"wtpt"), [0.9642, 1.0, 0.8249])
    .with_tag(
        sig(b"A2B0"),
        IccProfileBuilder::b_curves_lut_tag(true, &curves),
    )
    .build_profile()
    .unwrap()
}

#[test]
fn two_profile_chain_is_the_plain_transform() {
    let (a, b) = (IccProfile::srgb(), IccProfile::display_p3());
    for i in INTENTS {
        let c = IccTransform::chain(&[&a, &b], opts(i)).unwrap();
        let t = IccTransform::new(&a, &b, i).unwrap();
        for p in probes() {
            assert_eq!(run(&c, p), run(&t, p), "{i:?} {p:?}");
        }
    }
}

#[test]
fn chain_shape_errors() {
    let a = IccProfile::srgb();
    let abs = abstract_lab([Curve::Identity, Curve::Identity, Curve::Identity]);
    let o = TransformOptions::new();
    assert!(matches!(
        IccTransform::chain(&[&a], o),
        Err(IccError::Unsupported(_))
    ));
    assert!(matches!(
        IccTransform::chain(&[], o),
        Err(IccError::Unsupported(_))
    ));
    assert!(matches!(
        IccTransform::chain(&[&abs, &a], o),
        Err(IccError::Unsupported(_))
    ));
    assert!(matches!(
        IccTransform::chain(&[&a, &abs], o),
        Err(IccError::Unsupported(_))
    ));
}

#[test]
fn identity_abstract_profile_changes_nothing() {
    let (a, b) = (IccProfile::srgb(), IccProfile::display_p3());
    let abs = abstract_lab([Curve::Identity, Curve::Identity, Curve::Identity]);
    for i in [
        RenderingIntent::Perceptual,
        RenderingIntent::RelativeColorimetric,
    ] {
        let c = IccTransform::chain(&[&a, &abs, &b], opts(i)).unwrap();
        let t = IccTransform::new(&a, &b, i).unwrap();
        for p in probes() {
            let (x, y) = (run(&c, p), run(&t, p));
            for k in 0..3 {
                assert!((x[k] - y[k]).abs() < 1e-6, "{p:?}: {x:?} vs {y:?}");
            }
        }
    }
}

#[test]
fn abstract_lab_curve_reshapes_lightness_only() {
    // L* (encoded L/100) through x^0.75 (exact in u8Fixed8); a*, b*
    // untouched.
    let abs = abstract_lab([Curve::Gamma(0.75), Curve::Identity, Curve::Identity]);
    let lab = IccProfile::lab_d50();
    let c = IccTransform::chain(&[&lab, &abs, &lab], opts(RenderingIntent::Perceptual)).unwrap();
    for (l, a, b) in [(50.0, 20.0, -30.0), (10.0, 0.0, 0.0), (85.0, -40.0, 60.0)] {
        let enc = [l / 100.0, (a + 128.0) / 255.0, (b + 128.0) / 255.0];
        let o = run(&c, enc);
        let want = (l / 100.0f64).powf(0.75);
        assert!((o[0] - want).abs() < 1e-6, "L {o:?}");
        assert!((o[1] - enc[1]).abs() < 1e-6 && (o[2] - enc[2]).abs() < 1e-6);
    }
}

#[test]
fn float_abstract_profile_in_xyz() {
    // An XYZ → XYZ Abstract whose DToB0 halves the light (a float
    // `matf`; there is no AToB0 at all).
    let mut mpet = b"mpet\0\0\0\0\0\x03\0\x03\0\0\0\x01".to_vec();
    mpet.extend_from_slice(&24u32.to_be_bytes());
    let mut matf = b"matf\0\0\0\0\0\x03\0\x03".to_vec();
    for r in 0..3 {
        for c in 0..3 {
            matf.extend_from_slice(&(if r == c { 0.5f32 } else { 0.0 }).to_be_bytes());
        }
    }
    matf.extend_from_slice(&[0; 12]);
    mpet.extend_from_slice(&(matf.len() as u32).to_be_bytes());
    mpet.extend_from_slice(&matf);
    let abs = IccProfileBuilder::new(
        ProfileClass::Abstract,
        DataColorSpace::Xyz,
        DataColorSpace::Xyz,
    )
    .with_xyz(sig(b"wtpt"), [0.9642, 1.0, 0.8249])
    .with_tag(sig(b"D2B0"), mpet)
    .build_profile()
    .unwrap();
    let s = IccProfile::srgb();
    let c = IccTransform::chain(&[&s, &abs, &s], opts(RenderingIntent::Perceptual)).unwrap();
    let dec = |v: f64| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let enc = |v: f64| {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };
    for p in probes() {
        let o = run(&c, p);
        for k in 0..3 {
            let want = enc(0.5 * dec(p[k]));
            assert!((o[k] - want).abs() < 2e-4, "{p:?}: {o:?}");
        }
    }
}

#[test]
fn device_profile_in_the_middle() {
    // sRGB → P3 → sRGB: every sRGB colour fits in P3, so the round trip
    // is the identity. (The probes keep off the sRGB gamut edges: the
    // black → blue edge is P3's own, where s15Fixed16 colorant rounding
    // dips a channel just below zero and the clip shows.)
    let (s, p3) = (IccProfile::srgb(), IccProfile::display_p3());
    for i in INTENTS {
        let c = IccTransform::chain(&[&s, &p3, &s], opts(i)).unwrap();
        for p in probes().into_iter().map(|p| p.map(|v| 0.05 + 0.9 * v)) {
            let o = run(&c, p);
            for k in 0..3 {
                assert!((o[k] - p[k]).abs() < 1e-5, "{i:?} {p:?}: {o:?}");
            }
        }
    }
    // Through a gray profile every colour comes out neutral, with the
    // luminance it went in with.
    let gray = IccProfile::gray(&Curve::srgb());
    let c = IccTransform::chain(&[&s, &gray, &s], opts(RenderingIntent::Perceptual)).unwrap();
    let to_xyz =
        IccTransform::new(&s, &IccProfile::lab_d50(), RenderingIntent::Perceptual).unwrap();
    for p in probes() {
        let o = run(&c, p);
        assert!(
            (o[0] - o[1]).abs() < 1e-5 && (o[1] - o[2]).abs() < 1e-5,
            "{o:?}"
        );
        let (l_in, l_out) = (run(&to_xyz, p)[0], run(&to_xyz, o)[0]);
        assert!((l_in - l_out).abs() < 1e-4, "{p:?}: L {l_in} → {l_out}");
    }
}

#[test]
fn absolute_chain_rescales_into_and_out_of_the_middle_profile() {
    // A print-like sRGB with a dim media white: absolute into it scales
    // up by D50 / W, out of it back down by W / D50 — mid-tone colours
    // come back unchanged, while a direct absolute transform into it
    // differs from the relative one.
    let base = IccProfile::srgb();
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
    let print = b
        .with_xyz(sig(b"wtpt"), [0.88, 0.9, 0.74])
        .build_profile()
        .unwrap();
    let abs = opts(RenderingIntent::AbsoluteColorimetric);
    let c = IccTransform::chain(&[&base, &print, &base], abs).unwrap();
    for p in [[0.3, 0.4, 0.5], [0.5, 0.5, 0.5], [0.2, 0.6, 0.3]] {
        let o = run(&c, p);
        for k in 0..3 {
            assert!((o[k] - p[k]).abs() < 1e-5, "{p:?}: {o:?}");
        }
    }
    let direct = IccTransform::with_options(&base, &print, abs).unwrap();
    let rel = IccTransform::new(&base, &print, RenderingIntent::RelativeColorimetric).unwrap();
    let (x, y) = (run(&direct, [0.5; 3]), run(&rel, [0.5; 3]));
    assert!((x[0] - y[0]).abs() > 0.01, "{x:?} {y:?}");
}

#[cfg(not(miri))]
#[test]
fn three_profile_chain_matches_magick() {
    use std::process::Command;
    let ok = Command::new("magick")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("magick not found; skipping");
        return;
    }
    let profiles = [
        IccProfile::srgb(),
        IccProfile::rec2020(),
        IccProfile::adobe_rgb(),
    ];
    let dir = std::env::temp_dir();
    let tag = format!("oxideav-pixfmt-chain-{}", std::process::id());
    let paths: Vec<_> = (0..3)
        .map(|k| {
            let p = dir.join(format!("{tag}-{k}.icc"));
            std::fs::write(&p, profiles[k].as_bytes()).unwrap();
            p
        })
        .collect();
    let input: Vec<u16> = probes()
        .iter()
        .flatten()
        .map(|v| (v * 65535.0).round() as u16)
        .collect();
    let (pi, po) = (
        dir.join(format!("{tag}.in")),
        dir.join(format!("{tag}.out")),
    );
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_be_bytes()).collect();
    std::fs::write(&pi, bytes).unwrap();
    let mut cmd = Command::new("magick");
    cmd.args(["-size", &format!("{}x1", input.len() / 3)])
        .args(["-depth", "16", "-endian", "MSB"])
        .arg(format!("rgb:{}", pi.display()))
        .args(["-intent", "Relative"]);
    for p in &paths {
        cmd.arg("-profile").arg(p);
    }
    let st = cmd
        .args(["-depth", "16", "-endian", "MSB"])
        .arg(format!("rgb:{}", po.display()))
        .status()
        .unwrap();
    assert!(st.success());
    let theirs: Vec<u16> = std::fs::read(&po)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_be_bytes(*c))
        .collect();
    for p in paths.iter().chain([&pi, &po]) {
        let _ = std::fs::remove_file(p);
    }
    let refs: Vec<&IccProfile> = profiles.iter().collect();
    let t = IccTransform::chain(&refs, opts(RenderingIntent::RelativeColorimetric)).unwrap();
    let mut ours = vec![0u16; input.len()];
    t.transform_u16(&input, &mut ours).unwrap();
    let worst = ours
        .iter()
        .zip(&theirs)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap();
    eprintln!("sRGB → BT.2020 → Adobe RGB chain vs magick: max {worst} codes");
    assert!(worst <= 8, "{worst}");
}
