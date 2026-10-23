//! `Pal8` expansion honours the palette-alpha side-channel (core
//! 0.1.40): a transparent entry expands to alpha 0 on the `Rgba*`
//! rows, every entry the record does not cover is opaque, the `Rgb*`
//! rows drop alpha (documented rule), and the `Pal8` writer attaches
//! the record when its `Palette` carries non-opaque entries.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{ConvertOptions, FrameInfo, Palette, convert};

fn pal8_frame(w: usize, h: usize, indices: Vec<u8>) -> VideoFrame {
    assert_eq!(indices.len(), w * h);
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w,
            data: indices,
        }],
    }
}

fn four_entry_palette() -> Vec<u8> {
    // 0 red, 1 green, 2 blue, 3 white.
    vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]
}

#[test]
fn transparent_entry_expands_to_alpha_zero() {
    let (w, h) = (4usize, 2usize);
    let src = pal8_frame(w, h, vec![0, 1, 2, 3, 3, 2, 1, 0])
        .with_palette(four_entry_palette())
        // Entry 1 transparent, entry 0 half — the record stops at
        // entry 1; entries 2 and 3 are uncovered (opaque).
        .with_palette_alpha(vec![128, 0]);
    let info = FrameInfo::new(PixelFormat::Pal8, w as u32, h as u32);
    let rgba = convert(&src, info, PixelFormat::Rgba, &ConvertOptions::default()).unwrap();
    let px = &rgba.planes[0].data;
    let want_a = [128u8, 0, 255, 255, 255, 255, 0, 128];
    for (i, &a) in want_a.iter().enumerate() {
        assert_eq!(px[i * 4 + 3], a, "pixel {i} alpha");
    }
    assert_eq!(&px[0..3], &[255, 0, 0]);
    assert_eq!(&px[4..7], &[0, 255, 0]);
    // The alpha-first / swapped orders ride the Rgba pivot and keep it.
    for (fmt, a_at) in [
        (PixelFormat::Bgra, 3usize),
        (PixelFormat::Argb, 0),
        (PixelFormat::Abgr, 0),
    ] {
        let out = convert(&src, info, fmt, &ConvertOptions::default()).unwrap();
        for (i, &a) in want_a.iter().enumerate() {
            assert_eq!(out.planes[0].data[i * 4 + a_at], a, "{fmt:?} pixel {i}");
        }
    }
    // `Rgb*` rows drop alpha: colour only.
    let rgb = convert(&src, info, PixelFormat::Rgb24, &ConvertOptions::default()).unwrap();
    assert_eq!(&rgb.planes[0].data[0..6], &[255, 0, 0, 0, 255, 0]);
    let bgr = convert(&src, info, PixelFormat::Bgr24, &ConvertOptions::default()).unwrap();
    assert_eq!(&bgr.planes[0].data[0..6], &[0, 0, 255, 0, 255, 0]);
}

#[test]
fn no_record_or_malformed_record_is_opaque() {
    let (w, h) = (4usize, 1usize);
    let info = FrameInfo::new(PixelFormat::Pal8, w as u32, h as u32);
    let plain = pal8_frame(w, h, vec![0, 1, 2, 3]).with_palette(four_entry_palette());
    let rgba = convert(&plain, info, PixelFormat::Rgba, &ConvertOptions::default()).unwrap();
    assert!(
        rgba.planes[0]
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[3] == 255)
    );
    // A record longer than the palette is malformed: core reads it as
    // absent, so every entry is opaque.
    let bad = pal8_frame(w, h, vec![0, 1, 2, 3])
        .with_palette(four_entry_palette())
        .with_palette_alpha(vec![0, 0, 0, 0, 0]);
    let rgba = convert(&bad, info, PixelFormat::Rgba, &ConvertOptions::default()).unwrap();
    assert!(
        rgba.planes[0]
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[3] == 255)
    );
    // An options-bundle palette (no side-channel) keeps its own alpha
    // column, as before.
    let opts = ConvertOptions {
        palette: Some(Palette {
            colors: vec![[255, 0, 0, 0], [0, 255, 0, 255]],
        }),
        ..Default::default()
    };
    let bare = pal8_frame(2, 1, vec![0, 1]);
    let rgba = convert(
        &bare,
        FrameInfo::new(PixelFormat::Pal8, 2, 1),
        PixelFormat::Rgba,
        &opts,
    )
    .unwrap();
    assert_eq!(rgba.planes[0].data, vec![255, 0, 0, 0, 0, 255, 0, 255]);
}

#[test]
fn writer_attaches_the_alpha_record_and_round_trips() {
    let palette = Palette {
        colors: vec![
            [0, 0, 0, 0],
            [255, 0, 0, 255],
            [0, 255, 0, 64],
            [0, 0, 255, 255],
        ],
    };
    let opts = ConvertOptions {
        palette: Some(palette),
        ..Default::default()
    };
    let (w, h) = (4u32, 1u32);
    let rgba = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 16,
            data: vec![0, 0, 0, 0, 255, 0, 0, 255, 0, 255, 0, 64, 0, 0, 255, 255],
        }],
    };
    let pal = convert(
        &rgba,
        FrameInfo::new(PixelFormat::Rgba, w, h),
        PixelFormat::Pal8,
        &opts,
    )
    .unwrap();
    assert_eq!(pal.planes[0].data, vec![0, 1, 2, 3]);
    // The record covers entries up to the last non-opaque one.
    assert_eq!(pal.palette_alpha(), Some(&[0u8, 255, 64][..]));
    assert_eq!(pal.palette_rgba(0), Some([0, 0, 0, 0]));
    assert_eq!(pal.palette_rgba(3), Some([0, 0, 255, 255]));
    // Self-describing: expanding without the options bundle reproduces
    // the alpha column.
    let back = convert(
        &pal,
        FrameInfo::new(PixelFormat::Pal8, w, h),
        PixelFormat::Rgba,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert_eq!(back.planes[0].data, rgba.planes[0].data);
    // An all-opaque palette attaches no record.
    let opaque = ConvertOptions {
        palette: Some(Palette {
            colors: vec![[0, 0, 0, 255], [255, 255, 255, 255]],
        }),
        ..Default::default()
    };
    let pal = convert(
        &rgba,
        FrameInfo::new(PixelFormat::Rgba, w, h),
        PixelFormat::Pal8,
        &opaque,
    )
    .unwrap();
    assert_eq!(pal.palette_alpha(), None);
    assert!(pal.palette().is_some());
}
