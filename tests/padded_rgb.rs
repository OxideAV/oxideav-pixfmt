//! The padded 32-bit RGB layouts of oxideav-core 0.1.41 (`Rgbx` /
//! `Bgrx` / `Xrgb` / `Xbgr`): the padding byte is written as 0 and
//! never read, the alpha twins get opaque alpha, every hop to the
//! single-plane 8-bit family / `Gray8` / `Pal8` / the planar YUV(A)
//! family is direct and byte-identical to the hand-staged route, odd
//! geometry on padded strides, and thread parity.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, convert_with, staged_pivot, supports_direct, ConvertContext, ConvertOptions,
    FormatInfo, FrameInfo, Palette,
};

const PADDED: [PixelFormat; 4] = [
    PixelFormat::Rgbx,
    PixelFormat::Bgrx,
    PixelFormat::Xrgb,
    PixelFormat::Xbgr,
];
const ALPHA: [PixelFormat; 4] = [
    PixelFormat::Rgba,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
];
const THREE: [PixelFormat; 2] = [PixelFormat::Rgb24, PixelFormat::Bgr24];
const WORDS: [PixelFormat; 8] = [
    PixelFormat::Rgb555Le,
    PixelFormat::Rgb555Be,
    PixelFormat::Bgr555Le,
    PixelFormat::Bgr555Be,
    PixelFormat::Rgb565Le,
    PixelFormat::Rgb565Be,
    PixelFormat::Bgr565Le,
    PixelFormat::Bgr565Be,
];
/// A spread of planar family members: 8-bit, alpha-bearing, deep,
/// every chroma siting.
const PLANAR: [PixelFormat; 8] = [
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::Yuv440P,
    PixelFormat::Yuva420P,
    PixelFormat::Yuva444P12Le,
    PixelFormat::Yuv420P10Le,
    PixelFormat::Yuv444P16Le,
];

fn tight(data: Vec<u8>, stride: usize) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

fn conv(src: &VideoFrame, sf: PixelFormat, df: PixelFormat, w: u32, h: u32) -> VideoFrame {
    convert(
        src,
        FrameInfo::new(sf, w, h),
        df,
        &ConvertOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{sf:?} → {df:?}: {e:?}"))
}

fn rng(seed: &mut u32) -> u8 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 24) as u8
}

fn random_bytes(n: usize, seed: &mut u32) -> Vec<u8> {
    (0..n).map(|_| rng(seed)).collect()
}

/// Byte positions of R, G, B and the fourth lane (alpha or padding).
fn pos4(f: PixelFormat) -> [usize; 4] {
    match f {
        PixelFormat::Rgba | PixelFormat::Rgbx => [0, 1, 2, 3],
        PixelFormat::Bgra | PixelFormat::Bgrx => [2, 1, 0, 3],
        PixelFormat::Argb | PixelFormat::Xrgb => [1, 2, 3, 0],
        PixelFormat::Abgr | PixelFormat::Xbgr => [3, 2, 1, 0],
        other => panic!("{other:?} is not a 4-byte order"),
    }
}

fn pos3(f: PixelFormat) -> [usize; 3] {
    match f {
        PixelFormat::Rgb24 => [0, 1, 2],
        PixelFormat::Bgr24 => [2, 1, 0],
        other => panic!("{other:?} is not a 3-byte order"),
    }
}

/// The alpha twin of a padded layout.
fn twin(f: PixelFormat) -> PixelFormat {
    match f {
        PixelFormat::Rgbx => PixelFormat::Rgba,
        PixelFormat::Bgrx => PixelFormat::Bgra,
        PixelFormat::Xrgb => PixelFormat::Argb,
        PixelFormat::Xbgr => PixelFormat::Abgr,
        other => panic!("{other:?} is not padded"),
    }
}

fn planes_eq(a: &VideoFrame, b: &VideoFrame, what: &str) {
    assert_eq!(a.planes.len(), b.planes.len(), "{what}: plane count");
    for (i, (pa, pb)) in a.planes.iter().zip(&b.planes).enumerate() {
        assert_eq!(pa.data, pb.data, "{what}: plane {i}");
    }
}

#[test]
fn the_brief_pairs_are_direct() {
    for &pf in &PADDED {
        for &other in ALPHA
            .iter()
            .chain(&THREE)
            .chain(&PADDED)
            .chain(&WORDS)
            .chain(&PLANAR)
            .chain(&[PixelFormat::Gray8, PixelFormat::Pal8])
        {
            if other == pf {
                continue;
            }
            assert!(supports_direct(pf, other), "{pf:?} → {other:?} not direct");
            assert!(supports_direct(other, pf), "{other:?} → {pf:?} not direct");
        }
        // Layouts outside the planar family stage through 8-bit RGB.
        for other in [
            PixelFormat::Nv12,
            PixelFormat::Yuyv422,
            PixelFormat::YuvJ420P,
            PixelFormat::Rgb48Le,
        ] {
            let p = staged_pivot(pf, other).unwrap_or_else(|| panic!("{pf:?} → {other:?} direct?"));
            assert!(
                matches!(p, PixelFormat::Rgba | PixelFormat::Rgb24),
                "{pf:?} → {other:?} via {p:?}"
            );
        }
    }
}

#[test]
fn format_info_and_core_geometry_agree() {
    for &f in &PADDED {
        let info = FormatInfo::of(f);
        assert_eq!((info.bit_depth, info.planes), (8, 1), "{f:?}");
        assert!(
            !info.has_alpha && !info.is_planar && !info.is_palette,
            "{f:?}"
        );
        assert_eq!(f.plane_row_bytes(0, 5), Some(20), "{f:?}");
        assert!(f.is_packed_rgb() && !f.has_alpha(), "{f:?}");
    }
}

#[test]
fn padding_is_written_zero_and_never_read() {
    let (w, h) = if cfg!(miri) { (5u32, 2u32) } else { (37, 11) };
    let mut seed = 0x0000_0101u32;
    let n = w as usize * h as usize;
    for &pf in &PADDED {
        let [pr, pg, pb, px] = pos4(pf);
        // From every 4-byte layout (alpha or padded): colour carried,
        // the fourth lane of the source ignored, 0 written.
        for &sf in ALPHA.iter().chain(&PADDED) {
            if sf == pf {
                continue;
            }
            let [sr, sg, sb, sa] = pos4(sf);
            let src = tight(random_bytes(n * 4, &mut seed), w as usize * 4);
            let out = conv(&src, sf, pf, w, h);
            for (d, s) in out.planes[0]
                .data
                .chunks_exact(4)
                .zip(src.planes[0].data.chunks_exact(4))
            {
                assert_eq!(
                    [d[pr], d[pg], d[pb], d[px]],
                    [s[sr], s[sg], s[sb], 0],
                    "{sf:?} → {pf:?}"
                );
            }
            // Changing the source's fourth lane changes nothing.
            let mut junk = src.clone();
            for p in junk.planes[0].data.chunks_exact_mut(4) {
                p[sa] = rng(&mut seed);
            }
            let again = conv(&junk, sf, pf, w, h);
            assert_eq!(
                again.planes[0].data, out.planes[0].data,
                "{sf:?} → {pf:?} reads its fourth lane"
            );
        }
        // From the 3-byte orders.
        for &sf in &THREE {
            let [sr, sg, sb] = pos3(sf);
            let src = tight(random_bytes(n * 3, &mut seed), w as usize * 3);
            let out = conv(&src, sf, pf, w, h);
            for (d, s) in out.planes[0]
                .data
                .chunks_exact(4)
                .zip(src.planes[0].data.chunks_exact(3))
            {
                assert_eq!(
                    [d[pr], d[pg], d[pb], d[px]],
                    [s[sr], s[sg], s[sb], 0],
                    "{sf:?} → {pf:?}"
                );
            }
        }
        // To the alpha twin and every other alpha order: opaque alpha
        // from a junk-filled padding lane.
        let src = tight(random_bytes(n * 4, &mut seed), w as usize * 4);
        for &df in &ALPHA {
            let [dr, dg, db, da] = pos4(df);
            let out = conv(&src, pf, df, w, h);
            for (d, s) in out.planes[0]
                .data
                .chunks_exact(4)
                .zip(src.planes[0].data.chunks_exact(4))
            {
                assert_eq!(
                    [d[dr], d[dg], d[db], d[da]],
                    [s[pr], s[pg], s[pb], 255],
                    "{pf:?} → {df:?}"
                );
            }
        }
        // To the 3-byte orders: the padding lane dropped.
        for &df in &THREE {
            let [dr, dg, db] = pos3(df);
            let out = conv(&src, pf, df, w, h);
            for (d, s) in out.planes[0]
                .data
                .chunks_exact(3)
                .zip(src.planes[0].data.chunks_exact(4))
            {
                assert_eq!(
                    [d[dr], d[dg], d[db]],
                    [s[pr], s[pg], s[pb]],
                    "{pf:?} → {df:?}"
                );
            }
        }
        // Twin round trip: Rgba → Rgbx → Rgba is the source with alpha
        // forced opaque.
        let a = twin(pf);
        let [_, _, _, aa] = pos4(a);
        let rgba = tight(random_bytes(n * 4, &mut seed), w as usize * 4);
        let back = conv(&conv(&rgba, a, pf, w, h), pf, a, w, h);
        let mut expect = rgba.planes[0].data.clone();
        for p in expect.chunks_exact_mut(4) {
            p[aa] = 255;
        }
        assert_eq!(back.planes[0].data, expect, "{a:?} → {pf:?} → {a:?}");
    }
}

#[test]
fn gray8_and_pal8_hops() {
    let (w, h) = if cfg!(miri) { (4u32, 2u32) } else { (16, 5) };
    let n = w as usize * h as usize;
    let mut seed = 0x6a6a_0101u32;
    let gray = tight(random_bytes(n, &mut seed), w as usize);
    let colors: Vec<[u8; 4]> = (0..=255u16)
        .map(|_| {
            [
                rng(&mut seed),
                rng(&mut seed),
                rng(&mut seed),
                rng(&mut seed),
            ]
        })
        .collect();
    let opts = ConvertOptions {
        palette: Some(Palette {
            colors: colors.clone(),
        }),
        ..Default::default()
    };
    let pal = tight(random_bytes(n, &mut seed), w as usize);
    for &pf in &PADDED {
        let [pr, pg, pb, px] = pos4(pf);
        let out = conv(&gray, PixelFormat::Gray8, pf, w, h);
        for (d, &g) in out.planes[0].data.chunks_exact(4).zip(&gray.planes[0].data) {
            assert_eq!([d[pr], d[pg], d[pb], d[px]], [g, g, g, 0], "Gray8 → {pf:?}");
        }
        // Padded → Gray8 is the Rgb24 route's luminance projection.
        let src = tight(random_bytes(n * 4, &mut seed), w as usize * 4);
        let direct = conv(&src, pf, PixelFormat::Gray8, w, h);
        let via = conv(
            &conv(&src, pf, PixelFormat::Rgb24, w, h),
            PixelFormat::Rgb24,
            PixelFormat::Gray8,
            w,
            h,
        );
        assert_eq!(direct.planes[0].data, via.planes[0].data, "{pf:?} → Gray8");
        // Pal8 → padded: the palette colour, alpha column dropped, 0 padding.
        let out =
            convert(&pal, FrameInfo::new(PixelFormat::Pal8, w, h), pf, &opts).expect("Pal8 →");
        for (d, &i) in out.planes[0].data.chunks_exact(4).zip(&pal.planes[0].data) {
            let c = colors[i as usize];
            assert_eq!(
                [d[pr], d[pg], d[pb], d[px]],
                [c[0], c[1], c[2], 0],
                "Pal8 → {pf:?}"
            );
        }
        // Padded → Pal8 is the Rgb24 route's quantiser, table attached.
        let direct =
            convert(&src, FrameInfo::new(pf, w, h), PixelFormat::Pal8, &opts).expect("→ Pal8");
        let rgb = conv(&src, pf, PixelFormat::Rgb24, w, h);
        let via = convert(
            &rgb,
            FrameInfo::new(PixelFormat::Rgb24, w, h),
            PixelFormat::Pal8,
            &opts,
        )
        .expect("Rgb24 → Pal8");
        assert_eq!(direct.planes[0].data, via.planes[0].data, "{pf:?} → Pal8");
        assert!(direct.palette().is_some());
    }
}

/// A structurally valid planar family frame (deep words masked to the
/// member's significant bits).
fn planar_frame(f: PixelFormat, w: u32, h: u32, seed: &mut u32) -> VideoFrame {
    let info = FormatInfo::of(f);
    let planes = (0..f.plane_count())
        .map(|p| {
            let row = f.plane_row_bytes(p, w).expect("row bytes");
            let (_, ph) = f.plane_dimensions(p, w, h).expect("dims");
            let mut data = random_bytes(row * ph as usize, seed);
            if info.bit_depth > 8 && info.bit_depth < 16 {
                let mask = ((1u32 << info.bit_depth) - 1) as u16;
                for c in data.chunks_exact_mut(2) {
                    let v = u16::from_le_bytes([c[0], c[1]]) & mask;
                    c.copy_from_slice(&v.to_le_bytes());
                }
            }
            VideoPlane { stride: row, data }
        })
        .collect();
    VideoFrame { pts: None, planes }
}

#[test]
fn planar_family_hops_equal_the_rgb24_route() {
    let sizes: &[(u32, u32)] = if cfg!(miri) {
        &[(4, 2)]
    } else {
        &[(8, 6), (7, 5), (1, 1)]
    };
    let mut seed = 0x9a9a_0101u32;
    for &(w, h) in sizes {
        for &yf in &PLANAR {
            let yuv = planar_frame(yf, w, h, &mut seed);
            let rgb = conv(&yuv, yf, PixelFormat::Rgb24, w, h);
            for &pf in &PADDED {
                let [pr, pg, pb, px] = pos4(pf);
                let direct = conv(&yuv, yf, pf, w, h);
                for (d, s) in direct.planes[0]
                    .data
                    .chunks_exact(4)
                    .zip(rgb.planes[0].data.chunks_exact(3))
                {
                    assert_eq!(
                        [d[pr], d[pg], d[pb], d[px]],
                        [s[0], s[1], s[2], 0],
                        "{yf:?} → {pf:?} {w}×{h}"
                    );
                }
                // And back: padded → member equals Rgb24 → member
                // (opaque alpha on the alpha-bearing members).
                let src = tight(
                    random_bytes(w as usize * h as usize * 4, &mut seed),
                    w as usize * 4,
                );
                let direct = conv(&src, pf, yf, w, h);
                let mid = conv(&src, pf, PixelFormat::Rgb24, w, h);
                let manual = conv(&mid, PixelFormat::Rgb24, yf, w, h);
                planes_eq(&direct, &manual, &format!("{pf:?} → {yf:?} {w}×{h}"));
                if yf.has_alpha() {
                    let a = &direct.planes[3].data;
                    let full = if FormatInfo::of(yf).bit_depth > 8 {
                        let m = ((1u32 << FormatInfo::of(yf).bit_depth) - 1) as u16;
                        a.chunks_exact(2)
                            .all(|c| u16::from_le_bytes([c[0], c[1]]) == m)
                    } else {
                        a.iter().all(|&b| b == 255)
                    };
                    assert!(full, "{pf:?} → {yf:?}: alpha plane not opaque");
                }
            }
        }
    }
}

#[test]
fn staged_routes_equal_the_manual_two_step() {
    let (w, h) = if cfg!(miri) { (4u32, 2u32) } else { (16, 6) };
    let mut seed = 0xabcd_0101u32;
    for &pf in &PADDED {
        let src = tight(
            random_bytes(w as usize * h as usize * 4, &mut seed),
            w as usize * 4,
        );
        for df in [
            PixelFormat::Nv12,
            PixelFormat::Yuyv422,
            PixelFormat::Rgb48Le,
            PixelFormat::Gbrp8,
            PixelFormat::Cmyk,
            PixelFormat::Ya8,
        ] {
            let pivot = staged_pivot(pf, df).unwrap_or_else(|| panic!("{pf:?} → {df:?} direct?"));
            let staged = conv(&src, pf, df, w, h);
            let manual = conv(&conv(&src, pf, pivot, w, h), pivot, df, w, h);
            planes_eq(&staged, &manual, &format!("{pf:?} → {pivot:?} → {df:?}"));
            let pivot = staged_pivot(df, pf).unwrap_or_else(|| panic!("{df:?} → {pf:?} direct?"));
            let back = conv(&staged, df, pf, w, h);
            let manual = conv(&conv(&staged, df, pivot, w, h), pivot, pf, w, h);
            planes_eq(&back, &manual, &format!("{df:?} → {pivot:?} → {pf:?}"));
            // Whatever the route, the padding lane comes out 0.
            let px = pos4(pf)[3];
            assert!(
                back.planes[0].data.chunks_exact(4).all(|p| p[px] == 0),
                "{df:?} → {pf:?} padding"
            );
        }
    }
}

#[test]
fn odd_geometry_on_padded_strides() {
    let sizes: &[(u32, u32)] = if cfg!(miri) {
        &[(3, 2)]
    } else {
        &[(1, 1), (3, 1), (1, 3), (5, 3), (7, 2), (3, 7)]
    };
    let pads: &[usize] = if cfg!(miri) { &[1] } else { &[0, 1, 2, 5] };
    let mut seed = 0x0dd0_0101u32;
    let targets: Vec<PixelFormat> = ALPHA
        .iter()
        .chain(&THREE)
        .chain(&PADDED)
        .chain(&WORDS)
        .chain(&PLANAR)
        .chain(&[PixelFormat::Gray8])
        .copied()
        .collect();
    for &(w, h) in sizes {
        for &pad in pads {
            for &pf in &PADDED {
                let stride = w as usize * 4 + pad;
                let data = random_bytes(stride * h as usize, &mut seed);
                let padded = tight(data.clone(), stride);
                let mut td = Vec::new();
                for row in 0..h as usize {
                    td.extend_from_slice(&data[row * stride..row * stride + w as usize * 4]);
                }
                let tight_src = tight(td, w as usize * 4);
                for &df in &targets {
                    if df == pf {
                        continue;
                    }
                    let a = conv(&padded, pf, df, w, h);
                    let b = conv(&tight_src, pf, df, w, h);
                    planes_eq(&a, &b, &format!("{pf:?} → {df:?} {w}×{h} pad {pad}"));
                    let back = conv(&a, df, pf, w, h);
                    assert_eq!(
                        back.planes[0].data.len(),
                        w as usize * h as usize * 4,
                        "{df:?} → {pf:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn short_planes_are_errors_not_panics() {
    let short = tight(vec![0u8; 7], 16);
    for &df in ALPHA
        .iter()
        .chain(&THREE)
        .chain(&WORDS)
        .chain(&PLANAR)
        .chain(&[PixelFormat::Gray8])
    {
        let r = convert(
            &short,
            FrameInfo::new(PixelFormat::Bgrx, 4, 2),
            df,
            &ConvertOptions::default(),
        );
        assert!(r.is_err(), "Bgrx → {df:?} on a short plane");
    }
}

#[test]
fn thread_budget_does_not_change_the_bytes() {
    let (w, h) = if cfg!(miri) { (7u32, 5u32) } else { (64, 37) };
    let mut seed = 0x7777_0101u32;
    let padded = tight(
        random_bytes(w as usize * h as usize * 4, &mut seed),
        w as usize * 4,
    );
    let yuv = planar_frame(PixelFormat::Yuv420P, w, h, &mut seed);
    let opts = ConvertOptions::default();
    let serial = ConvertContext::new();
    let banded = ConvertContext::new().with_execution(ExecutionContext::with_threads(4));
    let pairs: &[(&VideoFrame, PixelFormat, PixelFormat)] = &[
        (&padded, PixelFormat::Bgrx, PixelFormat::Yuv420P),
        (&padded, PixelFormat::Xrgb, PixelFormat::Yuva444P),
        (&padded, PixelFormat::Rgbx, PixelFormat::Bgra),
        (&padded, PixelFormat::Xbgr, PixelFormat::Rgb24),
        (&padded, PixelFormat::Rgbx, PixelFormat::Rgb565Le),
        (&yuv, PixelFormat::Yuv420P, PixelFormat::Bgrx),
        (&yuv, PixelFormat::Yuv420P, PixelFormat::Xbgr),
    ];
    for &(src, sf, df) in pairs {
        let info = FrameInfo::new(sf, w, h);
        let a = convert_with(src, info, df, &opts, &serial).expect("serial");
        let b = convert_with(src, info, df, &opts, &banded).expect("banded");
        planes_eq(&a, &b, &format!("{sf:?} → {df:?}"));
    }
}
