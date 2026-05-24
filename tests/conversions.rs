//! Exact-roundtrip tests for the RGB-family swizzles and bit-depth
//! conversions. Every pair tested here must be lossless.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

fn synth_rgba(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
            data.push(((x + y) * 5) as u8);
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        },
        FrameInfo::new(PixelFormat::Rgba, w, h),
    )
}

fn synth_rgb24(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        },
        FrameInfo::new(PixelFormat::Rgb24, w, h),
    )
}

#[test]
fn rgb_family_4byte_roundtrips() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(32, 16);
    for fmt in [PixelFormat::Bgra, PixelFormat::Argb, PixelFormat::Abgr] {
        let stage = convert(&src, src_info, fmt, &opts).expect("swizzle");
        let stage_info = FrameInfo::new(fmt, src_info.width, src_info.height);
        let back = convert(&stage, stage_info, PixelFormat::Rgba, &opts).expect("swizzle back");
        assert_eq!(back.planes[0].data, src.planes[0].data, "roundtrip {fmt:?}");
    }
}

#[test]
fn rgb_family_3byte_roundtrips() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    let bgr = convert(&src, src_info, PixelFormat::Bgr24, &opts).unwrap();
    let bgr_info = FrameInfo::new(PixelFormat::Bgr24, src_info.width, src_info.height);
    let back = convert(&bgr, bgr_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgb24_to_rgba_and_back_preserves_colour() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(16, 8);
    let rgba = convert(&src, src_info, PixelFormat::Rgba, &opts).unwrap();
    let rgba_info = FrameInfo::new(PixelFormat::Rgba, src_info.width, src_info.height);
    let back = convert(&rgba, rgba_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgb48_rgb24_roundtrip() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(16, 8);
    let deep = convert(&src, src_info, PixelFormat::Rgb48Le, &opts).unwrap();
    let deep_info = FrameInfo::new(PixelFormat::Rgb48Le, src_info.width, src_info.height);
    let back = convert(&deep, deep_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgba64_rgba_roundtrip() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(16, 8);
    let deep = convert(&src, src_info, PixelFormat::Rgba64Le, &opts).unwrap();
    let deep_info = FrameInfo::new(PixelFormat::Rgba64Le, src_info.width, src_info.height);
    let back = convert(&deep, deep_info, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn gray8_gray16_roundtrip() {
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = Vec::with_capacity((w * h) as usize);
    for i in 0..(w * h) {
        data.push((i * 5) as u8);
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize,
            data,
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Gray8, w, h);
    let deep = convert(&src, src_info, PixelFormat::Gray16Le, &opts).unwrap();
    let deep_info = FrameInfo::new(PixelFormat::Gray16Le, w, h);
    let back = convert(&deep, deep_info, PixelFormat::Gray8, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn mono_black_gray8_roundtrip() {
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h) as usize];
    for (i, b) in data.iter_mut().enumerate() {
        *b = if i % 2 == 0 { 255 } else { 0 };
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize,
            data: data.clone(),
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Gray8, w, h);
    let mono = convert(&src, src_info, PixelFormat::MonoBlack, &opts).unwrap();
    let mono_info = FrameInfo::new(PixelFormat::MonoBlack, w, h);
    let back = convert(&mono, mono_info, PixelFormat::Gray8, &opts).unwrap();
    assert_eq!(back.planes[0].data, data);
}

#[test]
fn swizzle_all_four_byte_pairs() {
    // Every 4-byte ↔ 4-byte pair must roundtrip exactly.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(32, 16);
    let formats = [
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    for a in formats {
        for b in formats {
            if a == b {
                continue;
            }
            let frame_a = convert(&src, src_info, a, &opts).unwrap();
            let info_a = FrameInfo::new(a, src_info.width, src_info.height);
            let frame_b = convert(&frame_a, info_a, b, &opts).unwrap();
            let info_b = FrameInfo::new(b, src_info.width, src_info.height);
            let frame_back = convert(&frame_b, info_b, a, &opts).unwrap();
            assert_eq!(
                frame_a.planes[0].data, frame_back.planes[0].data,
                "a=Rgba stage={a:?} then {b:?}"
            );
        }
    }
}

#[test]
fn cmyk_roundtrip_via_rgb24() {
    // Rgb24 → Cmyk → Rgb24 is lossless at 8-bit precision by
    // construction of the formulas in the `cmyk` module.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(16, 8);
    let cmyk = convert(&src, src_info, PixelFormat::Cmyk, &opts).unwrap();
    assert_eq!(cmyk.planes[0].data.len(), 16 * 8 * 4);
    let cmyk_info = FrameInfo::new(PixelFormat::Cmyk, src_info.width, src_info.height);
    let back = convert(&cmyk, cmyk_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn cmyk_roundtrip_via_rgba() {
    // Rgba → Cmyk → Rgba. Alpha is dropped by Cmyk then restored
    // as opaque 255 on the way back, so the data matches only when
    // the source alpha was 255 to begin with.
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
            data.push(255);
        }
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 4) as usize,
            data,
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Rgba, w, h);
    let cmyk = convert(&src, src_info, PixelFormat::Cmyk, &opts).unwrap();
    let cmyk_info = FrameInfo::new(PixelFormat::Cmyk, w, h);
    let back = convert(&cmyk, cmyk_info, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}
