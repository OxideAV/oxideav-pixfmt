//! Yuva422P / Yuva444P conversion suite — the 4:2:2 and 4:4:4 members
//! of the planar YUV + full-resolution-alpha family that complete
//! `Yuva420P`.
//!
//! The invariants mirror the established `Yuva420P` suite in
//! `tests/conversions.rs`: the YUV planes behave exactly like the
//! alpha-less sibling's, while the trailing alpha plane is a verbatim
//! byte carrier — it must survive every alpha-capable route bit-exact,
//! be dropped on alpha-less targets, and be synthesised opaque when the
//! source has no alpha.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, supports, ConvertOptions, FrameInfo};

/// Deterministic planar YUVA frame: Y/A at full resolution, U/V on the
/// (wsub, hsub) chroma grid.
fn synth_yuva(fmt: PixelFormat, w: usize, h: usize, wsub: usize, hsub: usize) -> VideoFrame {
    let (cw, ch) = (w / wsub, h / hsub);
    let mut yp = Vec::with_capacity(w * h);
    let mut ap = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            yp.push(((i * 11 + j * 23) & 0xFF) as u8);
            ap.push(((i * 7 + j * 19 + 13) & 0xFF) as u8);
        }
    }
    let mut up = Vec::with_capacity(cw * ch);
    let mut vp = Vec::with_capacity(cw * ch);
    for j in 0..ch {
        for i in 0..cw {
            up.push(((i * 5 + j * 17) & 0xFF) as u8);
            vp.push(((i * 3 + j * 29 + 41) & 0xFF) as u8);
        }
    }
    let _ = fmt;
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
            VideoPlane {
                stride: w,
                data: ap,
            },
        ],
    }
}

/// Deterministic packed RGBA frame.
fn synth_rgba(w: usize, h: usize) -> VideoFrame {
    let mut data = Vec::with_capacity(w * h * 4);
    for j in 0..h {
        for i in 0..w {
            data.push(((i * 13 + j * 7) & 0xFF) as u8);
            data.push(((i * 31 + j * 3 + 9) & 0xFF) as u8);
            data.push(((i * 5 + j * 41 + 77) & 0xFF) as u8);
            data.push(((i * 17 + j * 29 + 100) & 0xFF) as u8);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data,
        }],
    }
}

const CASES: &[(PixelFormat, PixelFormat, usize, usize)] = &[
    // (alpha-less sibling, yuva format, wsub, hsub)
    (PixelFormat::Yuv422P, PixelFormat::Yuva422P, 2, 1),
    (PixelFormat::Yuv444P, PixelFormat::Yuva444P, 1, 1),
];

/// Yuv{422,444}P → Yuva sibling: Y/U/V byte-for-byte + opaque alpha;
/// the reverse drops the plane and round-trips bit-exact.
#[test]
fn promote_and_drop_roundtrip_all_grids() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(yuv_fmt, yuva_fmt, wsub, hsub) in CASES {
        let yuva_src = synth_yuva(yuva_fmt, w, h, wsub, hsub);
        // Build the alpha-less sibling from the same first three planes.
        let yuv_src = VideoFrame {
            pts: None,
            planes: yuva_src.planes[..3].to_vec(),
        };
        let info = FrameInfo::new(yuv_fmt, w as u32, h as u32);
        let promoted = convert(&yuv_src, info, yuva_fmt, &opts).expect("yuv → yuva");
        assert_eq!(promoted.planes.len(), 4, "{yuva_fmt:?}");
        for p in 0..3 {
            assert_eq!(promoted.planes[p].data, yuv_src.planes[p].data);
        }
        assert_eq!(promoted.planes[3].data.len(), w * h);
        assert!(promoted.planes[3].data.iter().all(|&a| a == 0xFF));

        let yuva_info = FrameInfo::new(yuva_fmt, w as u32, h as u32);
        let dropped = convert(&yuva_src, yuva_info, yuv_fmt, &opts).expect("yuva → yuv");
        assert_eq!(dropped.planes.len(), 3);
        for p in 0..3 {
            assert_eq!(dropped.planes[p].data, yuva_src.planes[p].data);
        }
    }
}

/// Yuva{422,444}P → Rgba carries the alpha plane bit-exact into the 4th
/// byte, and the Rgb24 output's colour channels match the Rgba ones.
#[test]
fn to_rgb_alpha_bit_exact_and_channels_agree() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(_, yuva_fmt, wsub, hsub) in CASES {
        let src = synth_yuva(yuva_fmt, w, h, wsub, hsub);
        let info = FrameInfo::new(yuva_fmt, w as u32, h as u32);
        let rgba = convert(&src, info, PixelFormat::Rgba, &opts).expect("yuva → rgba");
        let rgb = convert(&src, info, PixelFormat::Rgb24, &opts).expect("yuva → rgb24");
        for p in 0..w * h {
            assert_eq!(
                rgba.planes[0].data[p * 4 + 3],
                src.planes[3].data[p],
                "{yuva_fmt:?} alpha at {p}"
            );
            for c in 0..3 {
                assert_eq!(
                    rgb.planes[0].data[p * 3 + c],
                    rgba.planes[0].data[p * 4 + c],
                    "{yuva_fmt:?} channel {c} at {p}"
                );
            }
        }
    }
}

/// Yuva{422,444}P YUV planes decode identically to the alpha-less
/// sibling: the RGB output must be byte-identical to converting the
/// same three planes as Yuv{422,444}P.
#[test]
fn colour_math_matches_alpha_less_sibling() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(yuv_fmt, yuva_fmt, wsub, hsub) in CASES {
        let yuva_src = synth_yuva(yuva_fmt, w, h, wsub, hsub);
        let yuv_src = VideoFrame {
            pts: None,
            planes: yuva_src.planes[..3].to_vec(),
        };
        let via_yuva = convert(
            &yuva_src,
            FrameInfo::new(yuva_fmt, w as u32, h as u32),
            PixelFormat::Rgb24,
            &opts,
        )
        .expect("yuva → rgb");
        let via_yuv = convert(
            &yuv_src,
            FrameInfo::new(yuv_fmt, w as u32, h as u32),
            PixelFormat::Rgb24,
            &opts,
        )
        .expect("yuv → rgb");
        assert_eq!(
            via_yuva.planes[0].data, via_yuv.planes[0].data,
            "{yuva_fmt:?} colour math must match {yuv_fmt:?}"
        );
    }
}

/// Rgba → Yuva{422,444}P splits alpha out bit-exact; Rgb24 synthesises
/// an opaque plane.
#[test]
fn from_rgb_alpha_split_and_opaque_synthesis() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    let rgba = synth_rgba(w, h);
    let rgba_info = FrameInfo::new(PixelFormat::Rgba, w as u32, h as u32);
    for &(_, yuva_fmt, _, _) in CASES {
        let yuva = convert(&rgba, rgba_info, yuva_fmt, &opts).expect("rgba → yuva");
        assert_eq!(yuva.planes.len(), 4);
        for p in 0..w * h {
            assert_eq!(
                yuva.planes[3].data[p],
                rgba.planes[0].data[p * 4 + 3],
                "{yuva_fmt:?} alpha at {p}"
            );
        }
        // Rgb24 source (alpha-less): plane synthesised opaque.
        let rgb = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w * 3,
                data: rgba.planes[0]
                    .data
                    .chunks_exact(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect(),
            }],
        };
        let rgb_info = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
        let yuva = convert(&rgb, rgb_info, yuva_fmt, &opts).expect("rgb → yuva");
        assert!(yuva.planes[3].data.iter().all(|&a| a == 0xFF));
    }
}

/// 4:4:4 carries chroma at full resolution, so the
/// Rgba → Yuva444P → Rgba round-trip is a pure double-rounded matrix
/// trip: every colour channel within ±2 codes and alpha bit-exact.
#[test]
fn rgba_via_yuva444_roundtrips_within_two() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 16usize);
    let rgba = synth_rgba(w, h);
    let rgba_info = FrameInfo::new(PixelFormat::Rgba, w as u32, h as u32);
    let yuva = convert(&rgba, rgba_info, PixelFormat::Yuva444P, &opts).expect("rgba → yuva444");
    let back = convert(
        &yuva,
        FrameInfo::new(PixelFormat::Yuva444P, w as u32, h as u32),
        PixelFormat::Rgba,
        &opts,
    )
    .expect("yuva444 → rgba");
    for p in 0..w * h {
        assert_eq!(
            back.planes[0].data[p * 4 + 3],
            rgba.planes[0].data[p * 4 + 3],
            "alpha at {p}"
        );
        for c in 0..3 {
            let a = rgba.planes[0].data[p * 4 + c] as i32;
            let b = back.planes[0].data[p * 4 + c] as i32;
            assert!(
                (a - b).abs() <= 2,
                "channel {c} at {p}: {a} vs {b} (limited-range matrix rounds twice)"
            );
        }
    }
}

/// Chroma resample inside the Yuva family: luma and alpha are copied
/// byte-for-byte across all six ordered pairs, and the resampled chroma
/// equals what the alpha-less sibling conversion produces on the same
/// planes (shared primitives — no divergent math).
#[test]
fn yuva_resample_luma_alpha_exact_chroma_matches_sibling() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    let grids = [
        (PixelFormat::Yuva420P, PixelFormat::Yuv420P, 2usize, 2usize),
        (PixelFormat::Yuva422P, PixelFormat::Yuv422P, 2, 1),
        (PixelFormat::Yuva444P, PixelFormat::Yuv444P, 1, 1),
    ];
    for &(src_yuva, src_yuv, swsub, shsub) in &grids {
        for &(dst_yuva, dst_yuv, _, _) in &grids {
            if src_yuva == dst_yuva {
                continue;
            }
            let yuva_src = synth_yuva(src_yuva, w, h, swsub, shsub);
            let yuv_src = VideoFrame {
                pts: None,
                planes: yuva_src.planes[..3].to_vec(),
            };
            let got = convert(
                &yuva_src,
                FrameInfo::new(src_yuva, w as u32, h as u32),
                dst_yuva,
                &opts,
            )
            .unwrap_or_else(|e| panic!("{src_yuva:?} → {dst_yuva:?}: {e:?}"));
            assert_eq!(got.planes.len(), 4);
            assert_eq!(got.planes[0].data, yuva_src.planes[0].data, "luma copy");
            assert_eq!(got.planes[3].data, yuva_src.planes[3].data, "alpha copy");
            let sibling = convert(
                &yuv_src,
                FrameInfo::new(src_yuv, w as u32, h as u32),
                dst_yuv,
                &opts,
            )
            .expect("sibling resample");
            assert_eq!(got.planes[1].data, sibling.planes[1].data, "U plane");
            assert_eq!(got.planes[2].data, sibling.planes[2].data, "V plane");
        }
    }
}

/// Luma extraction: Yuva{422,444}P → Gray8 matches the alpha-less
/// sibling's output byte-for-byte (limited → full range rescale, chroma
/// and alpha dropped).
#[test]
fn to_gray8_matches_sibling_luma_extraction() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(yuv_fmt, yuva_fmt, wsub, hsub) in CASES {
        let yuva_src = synth_yuva(yuva_fmt, w, h, wsub, hsub);
        let yuv_src = VideoFrame {
            pts: None,
            planes: yuva_src.planes[..3].to_vec(),
        };
        let a = convert(
            &yuva_src,
            FrameInfo::new(yuva_fmt, w as u32, h as u32),
            PixelFormat::Gray8,
            &opts,
        )
        .expect("yuva → gray");
        let b = convert(
            &yuv_src,
            FrameInfo::new(yuv_fmt, w as u32, h as u32),
            PixelFormat::Gray8,
            &opts,
        )
        .expect("yuv → gray");
        assert_eq!(a.planes[0].data, b.planes[0].data, "{yuva_fmt:?}");
        assert_eq!(a.planes.len(), 1);
    }
}

/// Alpha survives staged routes on the new members: Yuva422P → Bgra and
/// Yuva444P → Bgra pivot through Rgba (alpha-capable pivot first) and
/// keep the plane bit-exact.
#[test]
fn alpha_survives_staged_routes() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(_, yuva_fmt, wsub, hsub) in CASES {
        let src = synth_yuva(yuva_fmt, w, h, wsub, hsub);
        let out = convert(
            &src,
            FrameInfo::new(yuva_fmt, w as u32, h as u32),
            PixelFormat::Bgra,
            &opts,
        )
        .expect("yuva → bgra");
        let alpha_out: Vec<u8> = out.planes[0].data.chunks(4).map(|c| c[3]).collect();
        assert_eq!(alpha_out, src.planes[3].data, "{yuva_fmt:?}");
    }
}

/// Reachability: every ordered pair inside the Yuva family, and between
/// each Yuva member and the packed-alpha RGB world, resolves.
#[test]
fn yuva_family_reachability() {
    use PixelFormat as P;
    const YUVA: &[PixelFormat] = &[P::Yuva420P, P::Yuva422P, P::Yuva444P];
    for &a in YUVA {
        for &b in YUVA {
            assert!(supports(a, b), "{a:?} → {b:?}");
        }
        for &b in &[
            P::Rgba,
            P::Bgra,
            P::Argb,
            P::Abgr,
            P::Rgb24,
            P::Gray8,
            P::Yuv420P,
            P::Yuv422P,
            P::Yuv444P,
        ] {
            assert!(supports(a, b), "{a:?} → {b:?}");
            assert!(supports(b, a), "{b:?} → {a:?}");
        }
    }
}
