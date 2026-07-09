//! Gray8 ↔ YUV-family conversions through `convert()`.
//!
//! These paths are pure luma-plane operations: YUV → Gray8 extracts the
//! luma plane (rescaling limited → full range for the `Yuv*` / `Nv*` /
//! `Yuva*` families, copying verbatim for `YuvJ*`), and Gray8 → YUV
//! synthesises neutral (128) chroma. No colour matrix is involved.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

fn gray_frame(w: usize, h: usize, data: Vec<u8>) -> VideoFrame {
    assert_eq!(data.len(), w * h);
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride: w, data }],
    }
}

fn planar_yuv(w: usize, h: usize, wsub: usize, hsub: usize, y: Vec<u8>, c: u8) -> VideoFrame {
    let cw = w / wsub;
    let ch = h / hsub;
    assert_eq!(y.len(), w * h);
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: vec![c; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![c; cw * ch],
            },
        ],
    }
}

const LIMITED_PLANAR: [(PixelFormat, usize, usize); 4] = [
    (PixelFormat::Yuv420P, 2, 2),
    (PixelFormat::Yuv422P, 2, 1),
    (PixelFormat::Yuv444P, 1, 1),
    (PixelFormat::Yuv411P, 4, 1),
];

const FULL_PLANAR: [(PixelFormat, usize, usize); 3] = [
    (PixelFormat::YuvJ420P, 2, 2),
    (PixelFormat::YuvJ422P, 2, 1),
    (PixelFormat::YuvJ444P, 1, 1),
];

/// Limited-range luma rails map onto the full-range Gray8 rails:
/// Y'=16 → 0, Y'=235 → 255 (and clamped beyond).
#[test]
fn limited_yuv_to_gray_range_anchors() {
    for (fmt, wsub, hsub) in LIMITED_PLANAR {
        let w = 8usize;
        let h = 8usize;
        for (y_in, want) in [(16u8, 0u8), (235, 255), (0, 0), (255, 255)] {
            let src = planar_yuv(w, h, wsub, hsub, vec![y_in; w * h], 128);
            let info = FrameInfo::new(fmt, w as u32, h as u32);
            let out = convert(&src, info, PixelFormat::Gray8, &ConvertOptions::default())
                .expect("yuv → gray");
            assert!(
                out.planes[0].data.iter().all(|&g| g == want),
                "fmt={fmt:?} y={y_in}: got {} want {want}",
                out.planes[0].data[0]
            );
        }
    }
}

/// Full-range `YuvJ*` luma is copied into Gray8 verbatim.
#[test]
fn yuvj_to_gray_is_luma_copy() {
    for (fmt, wsub, hsub) in FULL_PLANAR {
        let w = 16usize;
        let h = 4usize;
        let y: Vec<u8> = (0..w * h).map(|i| ((i * 5) % 256) as u8).collect();
        let src = planar_yuv(w, h, wsub, hsub, y.clone(), 77);
        let info = FrameInfo::new(fmt, w as u32, h as u32);
        let out = convert(&src, info, PixelFormat::Gray8, &ConvertOptions::default())
            .expect("yuvJ → gray");
        assert_eq!(out.planes[0].data, y, "fmt={fmt:?}");
    }
}

/// Gray8 → YuvJ* is likewise a verbatim luma copy with neutral chroma,
/// and the round-trip through the J family is bit-exact.
#[test]
fn gray_yuvj_roundtrip_exact() {
    for (fmt, _, _) in FULL_PLANAR {
        let w = 8usize;
        let h = 4usize;
        let g: Vec<u8> = (0..w * h).map(|i| ((i * 9 + 3) % 256) as u8).collect();
        let src = gray_frame(w, h, g.clone());
        let info = FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32);
        let j = convert(&src, info, fmt, &ConvertOptions::default()).expect("gray → J");
        assert_eq!(j.planes[0].data, g, "fmt={fmt:?}: luma not copied");
        assert!(j.planes[1].data.iter().all(|&c| c == 128));
        assert!(j.planes[2].data.iter().all(|&c| c == 128));
        let back = convert(
            &j,
            FrameInfo::new(fmt, w as u32, h as u32),
            PixelFormat::Gray8,
            &ConvertOptions::default(),
        )
        .expect("J → gray");
        assert_eq!(back.planes[0].data, g, "fmt={fmt:?}: round-trip not exact");
    }
}

/// Gray8 → limited Yuv* → Gray8 round-trips within ±1 (the 255→219
/// squeeze merges neighbouring codes; the rescale back rounds).
#[test]
fn gray_limited_yuv_roundtrip_within_one() {
    for (fmt, _, _) in LIMITED_PLANAR {
        let w = 16usize;
        let h = 16usize;
        let g: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let src = gray_frame(w, h, g.clone());
        let info = FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32);
        let yuv = convert(&src, info, fmt, &ConvertOptions::default()).expect("gray → yuv");
        // Luma must sit in the limited range.
        assert!(yuv.planes[0].data.iter().all(|&y| (16..=235).contains(&y)));
        assert!(yuv.planes[1].data.iter().all(|&c| c == 128));
        let back = convert(
            &yuv,
            FrameInfo::new(fmt, w as u32, h as u32),
            PixelFormat::Gray8,
            &ConvertOptions::default(),
        )
        .expect("yuv → gray");
        for (a, b) in g.iter().zip(back.planes[0].data.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 1, "fmt={fmt:?}: {a} → {b}");
        }
    }
}

/// Gray8 → Yuv444P → Rgb24 must agree with the direct Gray8 → Rgb24
/// broadcast within ±1 (one limited-range round-trip of quantisation).
#[test]
fn gray_via_yuv_matches_direct_broadcast() {
    let w = 16usize;
    let h = 16usize;
    let g: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
    let src = gray_frame(w, h, g.clone());
    let info = FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32);
    let o = ConvertOptions::default();

    let direct = convert(&src, info, PixelFormat::Rgb24, &o).expect("gray → rgb");
    let yuv = convert(&src, info, PixelFormat::Yuv444P, &o).expect("gray → yuv");
    let via = convert(
        &yuv,
        FrameInfo::new(PixelFormat::Yuv444P, w as u32, h as u32),
        PixelFormat::Rgb24,
        &o,
    )
    .expect("yuv → rgb");

    let mut max_diff = 0i32;
    for (a, b) in direct.planes[0].data.iter().zip(via.planes[0].data.iter()) {
        max_diff = max_diff.max((*a as i32 - *b as i32).abs());
    }
    assert!(max_diff <= 1, "max diff {max_diff}");
}

/// NV12 / NV21 → Gray8 matches Yuv420P → Gray8 (same luma plane), and
/// Gray8 → NV12 / NV21 produce identical bytes (all-neutral chroma).
#[test]
fn nv_gray_paths() {
    let w = 8usize;
    let h = 8usize;
    let y: Vec<u8> = (0..w * h).map(|i| ((i * 3 + 20) % 256) as u8).collect();
    let o = ConvertOptions::default();

    let nv = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w,
                data: y.clone(),
            },
            VideoPlane {
                stride: w,
                data: vec![90u8; (w / 2) * (h / 2) * 2],
            },
        ],
    };
    let planar = planar_yuv(w, h, 2, 2, y.clone(), 90);
    let g_from_nv12 = convert(
        &nv,
        FrameInfo::new(PixelFormat::Nv12, w as u32, h as u32),
        PixelFormat::Gray8,
        &o,
    )
    .expect("nv12 → gray");
    let g_from_nv21 = convert(
        &nv,
        FrameInfo::new(PixelFormat::Nv21, w as u32, h as u32),
        PixelFormat::Gray8,
        &o,
    )
    .expect("nv21 → gray");
    let g_from_planar = convert(
        &planar,
        FrameInfo::new(PixelFormat::Yuv420P, w as u32, h as u32),
        PixelFormat::Gray8,
        &o,
    )
    .expect("420 → gray");
    assert_eq!(g_from_nv12.planes[0].data, g_from_planar.planes[0].data);
    assert_eq!(g_from_nv21.planes[0].data, g_from_planar.planes[0].data);

    let g: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
    let gsrc = gray_frame(w, h, g);
    let ginfo = FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32);
    let nv12 = convert(&gsrc, ginfo, PixelFormat::Nv12, &o).expect("gray → nv12");
    let nv21 = convert(&gsrc, ginfo, PixelFormat::Nv21, &o).expect("gray → nv21");
    assert_eq!(nv12.planes.len(), 2);
    assert_eq!(nv12.planes[0].data, nv21.planes[0].data);
    assert_eq!(nv12.planes[1].data, nv21.planes[1].data);
    assert!(nv12.planes[1].data.iter().all(|&c| c == 128));
}

/// Yuva420P → Gray8 extracts luma and drops both chroma and alpha.
#[test]
fn yuva_to_gray_drops_alpha() {
    let w = 4usize;
    let h = 4usize;
    let mut f = planar_yuv(w, h, 2, 2, vec![128u8; w * h], 128);
    f.planes.push(VideoPlane {
        stride: w,
        data: vec![7u8; w * h],
    });
    let out = convert(
        &f,
        FrameInfo::new(PixelFormat::Yuva420P, w as u32, h as u32),
        PixelFormat::Gray8,
        &ConvertOptions::default(),
    )
    .expect("yuva → gray");
    assert_eq!(out.planes.len(), 1);
    // (128 - 16) * 255/219 rounds to 130.
    assert!(out.planes[0].data.iter().all(|&g| g == 130));
}

/// Luma extraction never touches chroma, so odd dimensions are accepted
/// on subsampled sources; the reverse (Gray8 → subsampled YUV) must
/// reject geometry whose chroma grid is unrepresentable.
#[test]
fn gray_odd_dimension_contract() {
    let o = ConvertOptions::default();
    // 3×3 claimed 4:2:0 → Gray8: allowed (luma-only read).
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: 3,
                data: vec![100u8; 9],
            },
            VideoPlane {
                stride: 1,
                data: vec![128u8; 1],
            },
            VideoPlane {
                stride: 1,
                data: vec![128u8; 1],
            },
        ],
    };
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv420P, 3, 3),
        PixelFormat::Gray8,
        &o,
    )
    .expect("odd-dim luma extraction");
    assert_eq!(out.planes[0].data.len(), 9);

    // Gray8 3×3 → 4:2:0 (planar or NV): rejected.
    let g = gray_frame(3, 3, vec![55u8; 9]);
    let ginfo = FrameInfo::new(PixelFormat::Gray8, 3, 3);
    assert!(convert(&g, ginfo, PixelFormat::Yuv420P, &o).is_err());
    assert!(convert(&g, ginfo, PixelFormat::Nv12, &o).is_err());
    // ...but 4:4:4 has no grid constraint.
    assert!(convert(&g, ginfo, PixelFormat::Yuv444P, &o).is_ok());
    assert!(convert(&g, ginfo, PixelFormat::YuvJ444P, &o).is_ok());
}
