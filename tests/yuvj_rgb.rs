//! Direct full-range `YuvJ*` ↔ packed RGB conversions through `convert()`.
//!
//! The `YuvJ420P` / `YuvJ422P` / `YuvJ444P` families carry full-range
//! (0..=255) samples by definition, so their RGB paths must use the
//! full-range matrix regardless of the range half of
//! `ConvertOptions::color_space` — the option only selects the primaries
//! (BT.601 / BT.709 / BT.2020). These tests pin that contract and check
//! the direct path against the previously-shipped two-step staging
//! (`YuvJ* → Yuv* (range rescale) → RGB`).

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ColorSpace, ConvertOptions, FrameInfo};

fn opts(cs: ColorSpace) -> ConvertOptions {
    ConvertOptions {
        color_space: cs,
        ..Default::default()
    }
}

/// Build a planar YUV frame with the given plane dimensions and constant
/// (y, u, v) sample values.
fn flat_yuv(w: usize, h: usize, wsub: usize, hsub: usize, y: u8, u: u8, v: u8) -> VideoFrame {
    let cw = w / wsub;
    let ch = h / hsub;
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w,
                data: vec![y; w * h],
            },
            VideoPlane {
                stride: cw,
                data: vec![u; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![v; cw * ch],
            },
        ],
    }
}

fn rgb_frame(w: usize, h: usize, rgb: &[u8]) -> VideoFrame {
    assert_eq!(rgb.len(), w * h * 3);
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 3,
            data: rgb.to_vec(),
        }],
    }
}

const J_FORMATS: [(PixelFormat, usize, usize); 3] = [
    (PixelFormat::YuvJ420P, 2, 2),
    (PixelFormat::YuvJ422P, 2, 1),
    (PixelFormat::YuvJ444P, 1, 1),
];

/// Full-range neutral gray is an identity point of every matrix: for any
/// primaries, Y' = 128 with centred chroma must decode to exactly
/// (128, 128, 128) — no 16..235 rescale may sneak in (limited range
/// would produce ~130).
#[test]
fn yuvj_gray_decodes_identically_for_all_primaries() {
    for (fmt, wsub, hsub) in J_FORMATS {
        for cs in [
            ColorSpace::Bt601Limited,
            ColorSpace::Bt709Limited,
            ColorSpace::Bt2020Limited,
            ColorSpace::Bt601Full,
        ] {
            let src = flat_yuv(8, 8, wsub, hsub, 128, 128, 128);
            let info = FrameInfo::new(fmt, 8, 8);
            let out = convert(&src, info, PixelFormat::Rgb24, &opts(cs)).expect("YuvJ → Rgb24");
            for px in out.planes[0].data.chunks(3) {
                assert_eq!(px, [128, 128, 128], "fmt={fmt:?} cs={cs:?}");
            }
        }
    }
}

/// Full-range black (Y'=0) and white (Y'=255) hit the RGB rails exactly.
#[test]
fn yuvj_black_white_full_scale() {
    for (fmt, wsub, hsub) in J_FORMATS {
        let info = FrameInfo::new(fmt, 8, 8);
        let black = flat_yuv(8, 8, wsub, hsub, 0, 128, 128);
        let out = convert(
            &black,
            info,
            PixelFormat::Rgb24,
            &opts(ColorSpace::Bt709Limited),
        )
        .expect("black");
        assert!(out.planes[0].data.iter().all(|&b| b == 0), "fmt={fmt:?}");

        let white = flat_yuv(8, 8, wsub, hsub, 255, 128, 128);
        let out = convert(
            &white,
            info,
            PixelFormat::Rgb24,
            &opts(ColorSpace::Bt709Limited),
        )
        .expect("white");
        assert!(out.planes[0].data.iter().all(|&b| b == 255), "fmt={fmt:?}");
    }
}

/// RGB white / black / neutral encode into full-range YUV rails: white →
/// Y'=255, black → Y'=0, and both carry centred (128) chroma. A limited
/// matrix would emit 235 / 16 instead.
#[test]
fn rgb_to_yuvj_reaches_full_scale() {
    for (fmt, _, _) in J_FORMATS {
        let w = 8usize;
        let h = 8usize;
        let info = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
        for (rgb, want_y) in [([255u8, 255, 255], 255u8), ([0, 0, 0], 0)] {
            let data: Vec<u8> = rgb.iter().copied().cycle().take(w * h * 3).collect();
            let src = rgb_frame(w, h, &data);
            let out = convert(&src, info, fmt, &opts(ColorSpace::Bt601Limited)).expect("rgb → J");
            assert!(
                out.planes[0].data.iter().all(|&y| y == want_y),
                "fmt={fmt:?} rgb={rgb:?}: luma {} != {want_y}",
                out.planes[0].data[0]
            );
            assert!(out.planes[1].data.iter().all(|&c| c == 128), "fmt={fmt:?}");
            assert!(out.planes[2].data.iter().all(|&c| c == 128), "fmt={fmt:?}");
        }
    }
}

/// The direct YuvJ → RGB path must agree with the pre-existing two-step
/// staging (YuvJ → Yuv range rescale, then limited-range Yuv → RGB)
/// within the tolerance the 8-bit range rescale itself costs (±3: the
/// 255→219 luma / 255→224 chroma squeeze merges up to two full-range
/// codes per limited code, and the matrix then rescales the residual).
#[test]
fn direct_yuvj_matches_two_step_staging() {
    for (jfmt, wsub, hsub) in J_FORMATS {
        let limited = match jfmt {
            PixelFormat::YuvJ420P => PixelFormat::Yuv420P,
            PixelFormat::YuvJ422P => PixelFormat::Yuv422P,
            _ => PixelFormat::Yuv444P,
        };
        let w = 16usize;
        let h = 8usize;
        // Gradient planes so the comparison sweeps the transfer curve.
        let mut src = flat_yuv(w, h, wsub, hsub, 0, 0, 0);
        for (i, s) in src.planes[0].data.iter_mut().enumerate() {
            *s = ((i * 7) % 256) as u8;
        }
        for p in 1..=2 {
            for (i, s) in src.planes[p].data.iter_mut().enumerate() {
                *s = ((i * 11 + p * 40) % 256) as u8;
            }
        }
        let info = FrameInfo::new(jfmt, w as u32, h as u32);
        let o = opts(ColorSpace::Bt601Limited);

        let direct = convert(&src, info, PixelFormat::Rgb24, &o).expect("direct");
        let staged_yuv = convert(&src, info, limited, &o).expect("stage 1");
        let staged = convert(
            &staged_yuv,
            FrameInfo::new(limited, w as u32, h as u32),
            PixelFormat::Rgb24,
            &o,
        )
        .expect("stage 2");

        let mut max_diff = 0i32;
        for (a, b) in direct.planes[0]
            .data
            .iter()
            .zip(staged.planes[0].data.iter())
        {
            max_diff = max_diff.max((*a as i32 - *b as i32).abs());
        }
        assert!(
            max_diff <= 3,
            "fmt={jfmt:?}: direct vs staged max diff {max_diff}"
        );
    }
}

/// Full-range 4:4:4 round-trip RGB → YuvJ444P → RGB is near-lossless
/// (±2 from the two fixed-point matrix applications; no subsampling and
/// no range squeeze on this path).
#[test]
fn rgb_yuvj444_roundtrip_tight() {
    let w = 32usize;
    let h = 8usize;
    let mut rgb = vec![0u8; w * h * 3];
    for (i, s) in rgb.iter_mut().enumerate() {
        *s = ((i * 13 + 5) % 256) as u8;
    }
    let src = rgb_frame(w, h, &rgb);
    let info = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
    for cs in [
        ColorSpace::Bt601Limited,
        ColorSpace::Bt709Limited,
        ColorSpace::Bt2020Limited,
    ] {
        let o = opts(cs);
        let j = convert(&src, info, PixelFormat::YuvJ444P, &o).expect("encode");
        let back = convert(
            &j,
            FrameInfo::new(PixelFormat::YuvJ444P, w as u32, h as u32),
            PixelFormat::Rgb24,
            &o,
        )
        .expect("decode");
        let mut max_diff = 0i32;
        for (a, b) in rgb.iter().zip(back.planes[0].data.iter()) {
            max_diff = max_diff.max((*a as i32 - *b as i32).abs());
        }
        assert!(max_diff <= 2, "cs={cs:?}: round-trip max diff {max_diff}");
    }
}

/// Rgba → YuvJ* consumes the alpha byte; YuvJ* → Rgba emits opaque alpha.
#[test]
fn yuvj_rgba_paths() {
    let (fmt, wsub, hsub) = (PixelFormat::YuvJ420P, 2, 2);
    let src = flat_yuv(8, 8, wsub, hsub, 200, 100, 60);
    let info = FrameInfo::new(fmt, 8, 8);
    let o = opts(ColorSpace::Bt601Limited);
    let rgba = convert(&src, info, PixelFormat::Rgba, &o).expect("J → Rgba");
    assert_eq!(rgba.planes[0].data.len(), 8 * 8 * 4);
    for px in rgba.planes[0].data.chunks(4) {
        assert_eq!(px[3], 255);
    }
    // And back: Rgba → YuvJ420P must match Rgb24 → YuvJ420P of the same colour.
    let rgb24 = convert(&src, info, PixelFormat::Rgb24, &o).expect("J → Rgb24");
    let j_from_rgba =
        convert(&rgba, FrameInfo::new(PixelFormat::Rgba, 8, 8), fmt, &o).expect("Rgba → J");
    let j_from_rgb =
        convert(&rgb24, FrameInfo::new(PixelFormat::Rgb24, 8, 8), fmt, &o).expect("Rgb24 → J");
    for p in 0..3 {
        assert_eq!(j_from_rgba.planes[p].data, j_from_rgb.planes[p].data);
    }
}

/// Odd dimensions on subsampled J layouts reject exactly like the
/// limited-range families do.
#[test]
fn yuvj_odd_dimensions_reject() {
    let src = flat_yuv(4, 4, 2, 2, 128, 128, 128);
    let o = opts(ColorSpace::Bt601Limited);
    // Claim 3×3 on a 4:2:0 J layout: must be Error::Invalid, not a panic.
    let info = FrameInfo::new(PixelFormat::YuvJ420P, 3, 3);
    assert!(convert(&src, info, PixelFormat::Rgb24, &o).is_err());
    // RGB → J with odd dims likewise.
    let rgb = rgb_frame(3, 3, &[100u8; 27]);
    let info = FrameInfo::new(PixelFormat::Rgb24, 3, 3);
    assert!(convert(&rgb, info, PixelFormat::YuvJ420P, &o).is_err());
}
