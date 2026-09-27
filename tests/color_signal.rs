//! Colour-signal handling: the source frame's `VideoFrame::color_signal`
//! record and the `ConvertContext::signal` override select the YUV range
//! and the H.273 matrix; an unsignalled conversion is byte-identical to
//! the historical behaviour.
//!
//! Oracles are the crate's public scalar Q15 primitives
//! (`yuv::yuv_to_rgb` / `yuv::rgb_to_yuv`) evaluated with the matrix
//! weights from H.273 Table 4 and the range from H.273 §8.3
//! (`VideoFullRangeFlag`), per pixel.

use oxideav_core::{
    ColorRange, ColorSignal, MatrixCoefficients, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_pixfmt::yuv::{self, YuvMatrix};
use oxideav_pixfmt::{
    convert, convert_with, ColorSpace, ConvertContext, ConvertOptions, FrameInfo,
};

fn ramp(n: usize, mul: usize, add: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * mul + add) % 256) as u8).collect()
}

fn yuv_frame(w: usize, h: usize, wsub: usize, hsub: usize, alpha: bool) -> VideoFrame {
    let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
    let mut planes = vec![
        VideoPlane {
            stride: w,
            data: ramp(w * h, 7, 3),
        },
        VideoPlane {
            stride: cw,
            data: ramp(cw * ch, 13, 40),
        },
        VideoPlane {
            stride: cw,
            data: ramp(cw * ch, 29, 90),
        },
    ];
    if alpha {
        planes.push(VideoPlane {
            stride: w,
            data: ramp(w * h, 3, 1),
        });
    }
    VideoFrame { pts: None, planes }
}

fn oracle_420(frame: &VideoFrame, w: usize, h: usize, m: YuvMatrix, bpp: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let mut out = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let ci = (row / 2) * cw + col / 2;
            let (r, g, b) = yuv::yuv_to_rgb(
                frame.planes[0].data[row * w + col],
                frame.planes[1].data[ci],
                frame.planes[2].data[ci],
                m,
            );
            out.extend_from_slice(&[r, g, b]);
            if bpp == 4 {
                out.push(frame.planes.get(3).map_or(255, |a| a.data[row * w + col]));
            }
        }
    }
    out
}

fn m(kr: f32, kb: f32, limited: bool) -> YuvMatrix {
    YuvMatrix { kr, kb, limited }
}

#[test]
fn unsignalled_frames_convert_exactly_as_before() {
    let (w, h) = (10, 6);
    let f = yuv_frame(w, h, 2, 2, true);
    let info = FrameInfo::new(PixelFormat::Yuva420P, w as u32, h as u32);
    let o = ConvertOptions {
        color_space: ColorSpace::Bt709Limited,
        ..Default::default()
    };
    let plain = convert(&f, info, PixelFormat::Rgba, &o).unwrap();
    assert_eq!(
        plain.planes[0].data,
        oracle_420(&f, w, h, YuvMatrix::BT709, 4)
    );
    // A fully unspecified record changes nothing either.
    let tagged = f.clone().with_color_signal(ColorSignal::unspecified());
    let again = convert(&tagged, info, PixelFormat::Rgba, &o).unwrap();
    assert_eq!(again.planes[0].data, plain.planes[0].data);
}

#[test]
fn frame_signal_selects_range_and_matrix() {
    let (w, h) = (9, 5);
    let base = yuv_frame(w, h, 2, 2, true);
    let info = FrameInfo::new(PixelFormat::Yuva420P, w as u32, h as u32);
    // Options say BT.601; the frame says full-range BT.709 — the frame
    // wins on both halves.
    let o = ConvertOptions::default();
    for (code, kr, kb) in [
        (MatrixCoefficients::BT709, 0.2126f32, 0.0722f32),
        (MatrixCoefficients::BT470_SYSTEM_BG, 0.299, 0.114),
        (MatrixCoefficients::BT601_525, 0.299, 0.114),
        (MatrixCoefficients::BT2020_NCL, 0.2627, 0.0593),
        (MatrixCoefficients::FCC, 0.30, 0.11),
        (MatrixCoefficients::SMPTE_ST240, 0.212, 0.087),
    ] {
        for (range, limited) in [(ColorRange::Full, false), (ColorRange::Limited, true)] {
            let sig = ColorSignal::unspecified()
                .with_range(range)
                .with_matrix(code);
            let f = base.clone().with_color_signal(sig);
            let got = convert(&f, info, PixelFormat::Rgba, &o).unwrap();
            assert_eq!(
                got.planes[0].data,
                oracle_420(&base, w, h, m(kr, kb, limited), 4),
                "matrix {code:?} range {range:?}"
            );
        }
    }
}

#[test]
fn context_override_beats_the_frame_record() {
    let (w, h) = (8, 4);
    let base = yuv_frame(w, h, 2, 2, false);
    let info = FrameInfo::new(PixelFormat::Yuv420P, w as u32, h as u32);
    let f = base.clone().with_color_signal(
        ColorSignal::unspecified()
            .with_range(ColorRange::Limited)
            .with_matrix(MatrixCoefficients::BT601_525),
    );
    let ctx = ConvertContext::new().with_range(ColorRange::Full);
    let got = convert_with(
        &f,
        info,
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
        &ctx,
    )
    .unwrap();
    // Range from the context, matrix still from the frame.
    assert_eq!(
        got.planes[0].data,
        oracle_420(&base, w, h, m(0.299, 0.114, false), 3)
    );
}

/// The HEIF case the r462 matrix flagged: full range on layouts with no
/// `YuvJ*` label — deep 4:2:0 to 16-bit RGB through the deep matrix.
#[test]
fn full_range_deep_420_reaches_the_deep_matrix() {
    let (w, h) = (6usize, 4usize);
    let (cw, ch) = (3usize, 2usize);
    let word = |v: u16| v.to_le_bytes();
    // Full-range 10-bit white (1023) with neutral chroma (512).
    let mk = |n: usize, v: u16| VideoPlane {
        stride: 0,
        data: (0..n).flat_map(|_| word(v)).collect(),
    };
    let mut planes = vec![mk(w * h, 1023), mk(cw * ch, 512), mk(cw * ch, 512)];
    planes[0].stride = w * 2;
    planes[1].stride = cw * 2;
    planes[2].stride = cw * 2;
    let f = VideoFrame { pts: None, planes };
    let info = FrameInfo::new(PixelFormat::Yuv420P10Le, w as u32, h as u32);
    let o = ConvertOptions::default();
    // Unsignalled: 1023 is above limited-range white → clips to 65535
    // anyway; use a mid-grey to see the range matter.
    let mut grey = f.clone();
    for p in grey.planes[0].data.chunks_exact_mut(2) {
        p.copy_from_slice(&word(512));
    }
    let limited = convert(&grey, info, PixelFormat::Rgb48Le, &o).unwrap();
    let full_sig = grey
        .clone()
        .with_color_signal(ColorSignal::unspecified().with_range(ColorRange::Full));
    let full = convert(&full_sig, info, PixelFormat::Rgb48Le, &o).unwrap();
    let px = |fr: &VideoFrame| u16::from_le_bytes([fr.planes[0].data[0], fr.planes[0].data[1]]);
    // Limited: (512 − 64) / 876 of full scale ≈ 0.5114 → ~33516.
    // Full: 512 / 1023 ≈ 0.5005 → ~32800. The two must differ and sit
    // near those values (±0.5 %).
    let (l, fl) = (px(&limited) as f64, px(&full) as f64);
    assert!((l / 65535.0 - 448.0 / 876.0).abs() < 0.005, "limited {l}");
    assert!((fl / 65535.0 - 512.0 / 1023.0).abs() < 0.005, "full {fl}");
    // Full-range white decodes to (near) full-scale white. The deep
    // path reaches the 16-bit matrix through the crate's MSB-replicating
    // widen, which maps the 10-bit neutral chroma 512 to 32800 rather
    // than the exact 16-bit neutral 32768 — a residual of a few tens of
    // codes on G, within 0.1 % of full scale.
    let white = f.with_color_signal(ColorSignal::unspecified().with_range(ColorRange::Full));
    let out = convert(&white, info, PixelFormat::Rgba64Le, &o).unwrap();
    for px in out.planes[0].data.chunks_exact(8) {
        for c in 0..3 {
            let v = u16::from_le_bytes([px[c * 2], px[c * 2 + 1]]);
            assert!(v >= 65535 - 64, "channel {c} = {v}");
        }
        assert_eq!(&px[6..8], &[0xFF, 0xFF]);
    }
}

#[test]
fn identity_matrix_routes_planes_as_gbr() {
    let (w, h) = (5usize, 3usize);
    let f = yuv_frame(w, h, 1, 1, false);
    let info = FrameInfo::new(PixelFormat::Yuv444P, w as u32, h as u32);
    let sig = ColorSignal::unspecified()
        .with_range(ColorRange::Full)
        .with_matrix(MatrixCoefficients::IDENTITY);
    let tagged = f.clone().with_color_signal(sig);
    let rgb = convert(
        &tagged,
        info,
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
    )
    .unwrap();
    for i in 0..w * h {
        // Y = G, Cb = B, Cr = R (H.273 equations 48–50).
        assert_eq!(
            &rgb.planes[0].data[i * 3..i * 3 + 3],
            &[
                f.planes[2].data[i],
                f.planes[0].data[i],
                f.planes[1].data[i]
            ]
        );
    }
    // Encode back through the context override: exact inverse.
    let rgb_info = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
    let back = convert_with(
        &rgb,
        rgb_info,
        PixelFormat::Yuv444P,
        &ConvertOptions::default(),
        &ConvertContext::new().with_signal(sig),
    )
    .unwrap();
    for p in 0..3 {
        assert_eq!(back.planes[p].data, f.planes[p].data, "plane {p}");
    }
    // Limited-range identity: every component scaled like luma.
    let lim = ColorSignal::unspecified()
        .with_range(ColorRange::Limited)
        .with_matrix(MatrixCoefficients::IDENTITY);
    let mut black_white = yuv_frame(2, 1, 1, 1, false);
    for p in 0..3 {
        black_white.planes[p].data = vec![16, 235];
    }
    let bw = convert(
        &black_white.with_color_signal(lim),
        FrameInfo::new(PixelFormat::Yuv444P, 2, 1),
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert_eq!(bw.planes[0].data, vec![0, 0, 0, 255, 255, 255]);
}

#[test]
fn identity_matrix_on_deep_alpha_layouts() {
    let (w, h) = (4usize, 2usize);
    let words = |vals: &[u16]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let g: Vec<u16> = (0..8).map(|i| i * 100).collect();
    let b: Vec<u16> = (0..8).map(|i| 1023 - i * 50).collect();
    let r: Vec<u16> = (0..8).map(|i| 300 + i * 10).collect();
    let a: Vec<u16> = (0..8).map(|i| i * 120).collect();
    let plane = |v: &[u16]| VideoPlane {
        stride: w * 2,
        data: words(v),
    };
    let f = VideoFrame {
        pts: None,
        planes: vec![plane(&g), plane(&b), plane(&r), plane(&a)],
    }
    .with_color_signal(
        ColorSignal::unspecified()
            .with_range(ColorRange::Full)
            .with_matrix(MatrixCoefficients::IDENTITY),
    );
    let info = FrameInfo::new(PixelFormat::Yuva444P10Le, w as u32, h as u32);
    let out = convert(&f, info, PixelFormat::Gbrap10Le, &ConvertOptions::default()).unwrap();
    assert_eq!(out.planes[0].data, words(&g));
    assert_eq!(out.planes[1].data, words(&b));
    assert_eq!(out.planes[2].data, words(&r));
    assert_eq!(out.planes[3].data, words(&a));
}

#[test]
fn rgb_source_signal_does_not_leak_into_the_yuv_side() {
    // An sRGB-tagged RGB frame (matrix = identity) encodes to YUV with
    // the options' matrix, not as identity.
    let (w, h) = (4usize, 2usize);
    let rgb = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 3,
            data: ramp(w * h * 3, 17, 5),
        }],
    };
    let info = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
    let o = ConvertOptions::default();
    let plain = convert(&rgb, info, PixelFormat::Yuv444P, &o).unwrap();
    let tagged = rgb.clone().with_color_signal(ColorSignal::srgb());
    let got = convert(&tagged, info, PixelFormat::Yuv444P, &o).unwrap();
    for p in 0..3 {
        assert_eq!(got.planes[p].data, plain.planes[p].data);
    }
}

#[test]
fn unimplemented_matrices_reject_cleanly() {
    let f = yuv_frame(4, 2, 2, 2, false);
    let info = FrameInfo::new(PixelFormat::Yuv420P, 4, 2);
    for code in [
        MatrixCoefficients::YCGCO,
        MatrixCoefficients::BT2020_CL,
        MatrixCoefficients::ICTCP,
        MatrixCoefficients::new(200),
    ] {
        let tagged = f
            .clone()
            .with_color_signal(ColorSignal::unspecified().with_matrix(code));
        assert!(
            convert(
                &tagged,
                info,
                PixelFormat::Rgb24,
                &ConvertOptions::default()
            )
            .is_err(),
            "{code:?}"
        );
        // A carriage move ignores the matrix.
        assert!(convert(
            &tagged,
            info,
            PixelFormat::Yuv444P,
            &ConvertOptions::default()
        )
        .is_ok());
    }
}

#[test]
fn range_signal_turns_the_j_rescale_into_a_copy() {
    // Yuv420P samples signalled full range → YuvJ420P is a plain copy.
    let f = yuv_frame(6, 4, 2, 2, false);
    let info = FrameInfo::new(PixelFormat::Yuv420P, 6, 4);
    let tagged = f
        .clone()
        .with_color_signal(ColorSignal::unspecified().with_range(ColorRange::Full));
    let j = convert(
        &tagged,
        info,
        PixelFormat::YuvJ420P,
        &ConvertOptions::default(),
    )
    .unwrap();
    for p in 0..3 {
        assert_eq!(j.planes[p].data, f.planes[p].data);
    }
    // Unsignalled, the historical limited → full rescale applies.
    let rescaled = convert(&f, info, PixelFormat::YuvJ420P, &ConvertOptions::default()).unwrap();
    assert_ne!(rescaled.planes[0].data, f.planes[0].data);
}
