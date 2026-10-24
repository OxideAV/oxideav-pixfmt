//! ICC profile parsing, writing and matrix/TRC / grayTRC transforms
//! against values computed independently from the staged references
//! (ICC.1:2022, CSS Color 4 §19 sample code).
#![cfg(feature = "icc")]

use oxideav_pixfmt::colorspace::delta_e2000;
use oxideav_pixfmt::icc::{
    Curve, DataColorSpace, IccError, IccProfile, IccProfileBuilder, IccTransform, PCS_WHITE,
    ProfileClass, RenderingIntent, xyz_to_lab,
};

fn sig(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

#[test]
fn header_and_tags_round_trip_through_the_writer() {
    let p = IccProfile::srgb();
    let h = p.header();
    assert_eq!(h.version, (4, 4, 0));
    assert_eq!(h.class, ProfileClass::Display);
    assert_eq!(h.color_space, DataColorSpace::Rgb);
    assert_eq!(h.pcs, DataColorSpace::Xyz);
    assert_eq!(h.size as usize, p.as_bytes().len());
    assert_eq!(&p.as_bytes()[36..40], b"acsp");
    assert!(p.is_matrix_shaper());
    assert_eq!(p.description().as_deref(), Some("sRGB (synthesized)"));
    assert_eq!(p.copyright().as_deref(), Some("No copyright, use freely"));
    // §7.3.4: every tag offset is 4-byte aligned; shared TRC data.
    for t in p.tags() {
        assert_eq!(t.offset % 4, 0);
    }
    let r = p
        .tags()
        .iter()
        .find(|t| t.signature == sig(b"rTRC"))
        .unwrap();
    let b = p
        .tags()
        .iter()
        .find(|t| t.signature == sig(b"bTRC"))
        .unwrap();
    assert_eq!(r.offset, b.offset);
    // Header illuminant is the D50 PCS white (§7.2.16) within s15.16.
    for (got, want) in h.illuminant.iter().zip(PCS_WHITE) {
        assert!((got - want).abs() < 1.0 / 65536.0);
    }
    // Re-parse the bytes: identical decode.
    let q = IccProfile::parse(p.as_bytes()).unwrap();
    assert_eq!(q.colorant_matrix(), p.colorant_matrix());
    assert_eq!(q.rgb_trc(), p.rgb_trc());
}

#[test]
fn malformed_profiles_are_errors_not_panics() {
    let good = IccProfile::srgb().as_bytes().to_vec();
    assert_eq!(
        IccProfile::parse(&good[..100]).unwrap_err(),
        IccError::Truncated
    );
    let mut bad = good.clone();
    bad[36] = b'x';
    assert_eq!(IccProfile::parse(&bad).unwrap_err(), IccError::BadSignature);
    // Declared size beyond the buffer.
    let mut big = good.clone();
    big[0..4].copy_from_slice(&(good.len() as u32 + 4).to_be_bytes());
    assert_eq!(IccProfile::parse(&big).unwrap_err(), IccError::Truncated);
    // A tag pointing past the end.
    let mut out = good.clone();
    out[132 + 4..132 + 8].copy_from_slice(&(good.len() as u32).to_be_bytes());
    assert!(IccProfile::parse(&out).is_err());
    // Every truncation point either parses or errors.
    for n in 0..good.len() {
        let mut t = good[..n].to_vec();
        if n >= 4 {
            t[0..4].copy_from_slice(&(n as u32).to_be_bytes());
        }
        let _ = IccProfile::parse(&t);
    }
    // A CLUT whose grid would need terabytes is refused up front.
    let mut lut = b"mAB \0\0\0\0".to_vec();
    lut.extend_from_slice(&[8, 3, 0, 0]);
    lut.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0]);
    lut.extend_from_slice(&[255; 8]);
    lut.extend_from_slice(&[0; 8]);
    lut.extend_from_slice(&[2, 0, 0, 0]);
    let p = IccProfileBuilder::new(
        ProfileClass::Output,
        DataColorSpace::Color(8),
        DataColorSpace::Lab,
    )
    .with_tag(sig(b"A2B0"), lut)
    .build();
    assert!(matches!(IccProfile::parse(&p), Err(IccError::Malformed(_))));
}

/// CSS Color 4 §19: `lin_sRGB_to_XYZ` then the Bradford `D65_to_D50`.
fn css_srgb_to_xyz_d50(rgb: [f64; 3]) -> [f64; 3] {
    let lin = |v: f64| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let l = [lin(rgb[0]), lin(rgb[1]), lin(rgb[2])];
    let m = [
        [506752.0 / 1228815.0, 87881.0 / 245763.0, 12673.0 / 70218.0],
        [87098.0 / 409605.0, 175762.0 / 245763.0, 12673.0 / 175545.0],
        [7918.0 / 409605.0, 87881.0 / 737289.0, 1001167.0 / 1053270.0],
    ];
    let a = [
        [
            1.0479297925449969,
            0.022946870601609652,
            -0.05019226628920524,
        ],
        [
            0.02962780877005599,
            0.9904344267538799,
            -0.017073799063418826,
        ],
        [
            -0.009243040646204504,
            0.015055191490298152,
            0.7518742814281371,
        ],
    ];
    let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    };
    mv(&a, mv(&m, l))
}

#[test]
fn srgb_profile_agrees_with_css_color_4() {
    let p = IccProfile::srgb();
    let lab = IccProfile::lab_d50();
    let t = IccTransform::new(&p, &lab, RenderingIntent::RelativeColorimetric).unwrap();
    let mut worst: f64 = 0.0;
    for r in 0..=8 {
        for g in 0..=8 {
            for b in 0..=8 {
                let rgb = [r as f64 / 8.0, g as f64 / 8.0, b as f64 / 8.0];
                let mut o = [0.0; 3];
                t.transform_pixel(&rgb, &mut o);
                let ours = [o[0] * 100.0, o[1] * 255.0 - 128.0, o[2] * 255.0 - 128.0];
                // CSS works relative to its own D50 (x 0.3457, y 0.3585);
                // the ICC PCS white is 0.9642 / 1 / 0.8249, so compare in
                // Lab relative to each one's white.
                let css_white = [0.3457 / 0.3585, 1.0, (1.0 - 0.3457 - 0.3585) / 0.3585];
                let x = css_srgb_to_xyz_d50(rgb);
                let rel = [
                    x[0] / css_white[0] * PCS_WHITE[0],
                    x[1],
                    x[2] / css_white[2] * PCS_WHITE[2],
                ];
                worst = worst.max(delta_e2000(ours, xyz_to_lab(rel)));
            }
        }
    }
    // s15Fixed16 colorants and two slightly different D50s.
    assert!(worst < 0.05, "{worst}");
}

#[test]
fn matrix_trc_round_trips_at_16_bits() {
    for p in [
        IccProfile::srgb(),
        IccProfile::display_p3(),
        IccProfile::adobe_rgb(),
        IccProfile::rec2020(),
        IccProfile::prophoto(),
    ] {
        let t = IccTransform::new(&p, &p, RenderingIntent::Perceptual).unwrap();
        let mut input = Vec::new();
        for k in 0..4096u32 {
            let v = (k * 16 + k / 256) as u16;
            input.extend_from_slice(&[v, 65535 - v, v / 3]);
        }
        let mut out = vec![0u16; input.len()];
        t.transform_u16(&input, &mut out).unwrap();
        for (i, o) in input.iter().zip(&out) {
            assert!(
                (*i as i32 - *o as i32).abs() <= 1,
                "{:?} {i} -> {o}",
                p.description()
            );
        }
    }
}

#[test]
fn gray_trc_model() {
    // Annex F.2: connection = grayTRC(device) × PCS white. The gamma is
    // stored as u8Fixed8 (§10.6), so 2.2 becomes 563 / 256.
    let g = IccProfile::gray(&Curve::Gamma(563.0 / 256.0));
    let lab = IccProfile::lab_d50();
    let t = IccTransform::new(&g, &lab, RenderingIntent::Perceptual).unwrap();
    let mut o = [0.0; 3];
    t.transform_pixel(&[0.5], &mut o);
    let y = 0.5f64.powf(563.0 / 256.0);
    let expect = xyz_to_lab([y * PCS_WHITE[0], y, y * PCS_WHITE[2]]);
    assert!((o[0] * 100.0 - expect[0]).abs() < 1e-9);
    assert!((o[1] * 255.0 - 128.0).abs() < 1e-9 && (o[2] * 255.0 - 128.0).abs() < 1e-9);
    // Gray → sRGB gives neutral RGB; sRGB → gray inverts it.
    let s = IccProfile::srgb();
    let to = IccTransform::new(&g, &s, RenderingIntent::Perceptual).unwrap();
    let back = IccTransform::new(&s, &g, RenderingIntent::Perceptual).unwrap();
    for v in (0..=65535u32).step_by(257) {
        let d = v as f64 / 65535.0;
        let mut rgb = [0.0; 3];
        to.transform_pixel(&[d], &mut rgb);
        assert!(
            (rgb[0] - rgb[1]).abs() < 1e-4 && (rgb[2] - rgb[1]).abs() < 1e-4,
            "{rgb:?}"
        );
        let mut gray = [0.0];
        back.transform_pixel(&rgb, &mut gray);
        assert!(
            (gray[0] - d).abs() < 0.5 / 65535.0,
            "{v} -> {rgb:?} -> {gray:?}"
        );
    }
}

#[test]
fn absolute_intent_scales_by_media_white() {
    // An output-class gray profile whose media white is a paper at 80 %
    // of the PCS white (equations 1–6 of §6.2.3).
    let paper = [0.9642 * 0.8, 0.8, 0.8249 * 0.8];
    let p = IccProfileBuilder::new(
        ProfileClass::Output,
        DataColorSpace::Gray,
        DataColorSpace::Xyz,
    )
    .with_xyz(sig(b"wtpt"), paper)
    .with_curve(sig(b"kTRC"), &Curve::Identity)
    .build_profile()
    .unwrap();
    let g = IccProfile::gray(&Curve::Identity);
    let rel = IccTransform::new(&g, &p, RenderingIntent::RelativeColorimetric).unwrap();
    let abs = IccTransform::new(&g, &p, RenderingIntent::AbsoluteColorimetric).unwrap();
    let mut o = [0.0];
    rel.transform_pixel(&[0.5], &mut o);
    assert!((o[0] - 0.5).abs() < 1e-6);
    // Display source: media white = PCS white (§9.2.36), so absolute
    // 0.5 lands at 0.5 / 0.8 relative to the paper.
    abs.transform_pixel(&[0.5], &mut o);
    assert!((o[0] - 0.625).abs() < 1e-4, "{}", o[0]);
    // Paper white is not reachable: clipped to the device maximum.
    abs.transform_pixel(&[1.0], &mut o);
    assert!((o[0] - 1.0).abs() < 1e-9);
}

#[test]
fn buffer_length_checks() {
    let t = IccTransform::new(
        &IccProfile::srgb(),
        &IccProfile::gray(&Curve::srgb()),
        RenderingIntent::Perceptual,
    )
    .unwrap();
    assert_eq!(t.input_channels(), 3);
    assert_eq!(t.output_channels(), 1);
    let mut out = [0u8; 2];
    assert_eq!(
        t.transform_u8(&[1, 2, 3, 4], &mut out),
        Err(IccError::BufferSize)
    );
    assert_eq!(
        t.transform_u8(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &mut out),
        Err(IccError::BufferSize)
    );
    t.transform_u8(&[255, 255, 255, 0, 0, 0], &mut out).unwrap();
    assert_eq!(out, [255, 0]);
    let mut f = [0f32; 2];
    t.transform_f32(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], &mut f)
        .unwrap();
    assert!((f[0] - 1.0).abs() < 1e-6 && f[1].abs() < 1e-6);
}

#[test]
fn threads_and_tabulated_curves_do_not_change_results() {
    use oxideav_pixfmt::icc::TransformOptions;
    let s = IccProfile::srgb();
    let d = IccProfile::display_p3();
    let serial = IccTransform::new(&s, &d, RenderingIntent::Perceptual).unwrap();
    let par = IccTransform::with_options(
        &s,
        &d,
        TransformOptions::new()
            .with_intent(RenderingIntent::Perceptual)
            .with_threads(4),
    )
    .unwrap();
    // ≥ 2^15 pixels: the u16 path tabulates the input curves.
    let n = 40_000usize;
    let src: Vec<u16> = (0..n * 3)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 16) as u16)
        .collect();
    let mut a = vec![0u16; n * 3];
    let mut b = vec![0u16; n * 3];
    serial.transform_u16(&src, &mut a).unwrap();
    par.transform_u16(&src, &mut b).unwrap();
    assert_eq!(a, b);
    // Per-pixel reference (no tables).
    for p in (0..n).step_by(997) {
        let i: Vec<f64> = src[p * 3..p * 3 + 3]
            .iter()
            .map(|&v| v as f64 / 65535.0)
            .collect();
        let mut o = [0.0; 3];
        serial.transform_pixel(&i, &mut o);
        for c in 0..3 {
            assert_eq!(
                a[p * 3 + c],
                (o[c].clamp(0.0, 1.0) * 65535.0).round() as u16
            );
        }
    }
    let src8: Vec<u8> = src.iter().map(|&v| (v >> 8) as u8).collect();
    let mut a8 = vec![0u8; n * 3];
    let mut b8 = vec![0u8; n * 3];
    serial.transform_u8(&src8, &mut a8).unwrap();
    par.transform_u8(&src8, &mut b8).unwrap();
    assert_eq!(a8, b8);
    // A short buffer (no tables) agrees with the long one.
    let mut c8 = vec![0u8; 30];
    serial.transform_u8(&src8[..30], &mut c8).unwrap();
    assert_eq!(&c8[..], &a8[..30]);
}

#[test]
fn frame_transform_carries_alpha_and_replaces_the_icc_blob() {
    use oxideav_core::{BlobKind, MetadataBlob, PixelFormat, VideoFrame, VideoPlane};
    use oxideav_pixfmt::FrameInfo;
    use oxideav_pixfmt::icc::{frame_profile, transform_frame};
    let s = IccProfile::srgb();
    let g = IccProfile::gray(&Curve::srgb());
    // 2×2 RGBA with a padded stride and an embedded sRGB profile.
    let px: [[u8; 4]; 4] = [
        [255, 255, 255, 10],
        [0, 0, 0, 20],
        [128, 128, 128, 30],
        [255, 0, 0, 40],
    ];
    let stride = 12;
    let mut data = vec![0xAAu8; stride * 2];
    for (k, p) in px.iter().enumerate() {
        let o = (k / 2) * stride + (k % 2) * 4;
        data[o..o + 4].copy_from_slice(p);
    }
    let mut f = VideoFrame {
        pts: Some(7),
        planes: vec![VideoPlane { stride, data }],
    };
    f.set_blobs(vec![
        MetadataBlob::new(BlobKind::ICC, s.as_bytes().to_vec()),
        MetadataBlob::new(BlobKind::EXIF, vec![1, 2, 3]),
    ]);
    let src_profile = frame_profile(&f).unwrap().unwrap();
    let t = IccTransform::new(&src_profile, &g, RenderingIntent::RelativeColorimetric).unwrap();
    let out = transform_frame(
        &t,
        &f,
        FrameInfo::new(PixelFormat::Rgba, 2, 2),
        PixelFormat::Ya8,
        Some(&g),
    )
    .unwrap();
    assert_eq!(out.pts, Some(7));
    let plane = &out.image_planes()[0];
    assert_eq!(plane.stride, 4);
    // White → 255, black → 0, neutral 128 stays 128 (same sRGB curve);
    // alpha carried.
    assert_eq!(&plane.data[0..2], &[255, 10]);
    assert_eq!(&plane.data[2..4], &[0, 20]);
    assert!((plane.data[4] as i32 - 128).abs() <= 1 && plane.data[5] == 30);
    assert_eq!(plane.data[7], 40);
    assert_eq!(out.blob(&BlobKind::ICC).unwrap().data, g.as_bytes());
    assert_eq!(out.blob(&BlobKind::EXIF).unwrap().data, vec![1, 2, 3]);
    // Gray8 → Rgba64Le synthesises opaque alpha.
    let back = IccTransform::new(&g, &s, RenderingIntent::RelativeColorimetric).unwrap();
    let gray = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 1,
            data: vec![255],
        }],
    };
    let o = transform_frame(
        &back,
        &gray,
        FrameInfo::new(PixelFormat::Gray8, 1, 1),
        PixelFormat::Rgba64Le,
        None,
    )
    .unwrap();
    assert_eq!(o.image_planes()[0].data, vec![255; 8]);
    // Channel mismatch is an error, not a panic.
    assert!(
        transform_frame(
            &back,
            &gray,
            FrameInfo::new(PixelFormat::Rgb24, 1, 1),
            PixelFormat::Rgb24,
            None
        )
        .is_err()
    );
    // Short plane.
    assert!(
        transform_frame(
            &t,
            &gray,
            FrameInfo::new(PixelFormat::Rgba, 4, 4),
            PixelFormat::Gray8,
            None
        )
        .is_err()
    );
}
