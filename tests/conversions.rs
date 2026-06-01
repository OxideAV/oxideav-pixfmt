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

// -------- Ya8 (grey + alpha, 2 bytes/pixel) --------

fn synth_ya8(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let mut data = Vec::with_capacity((w * h * 2) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8); // Y
            data.push((x * 5 + y * 11) as u8); // A
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 2) as usize,
                data,
            }],
        },
        FrameInfo::new(PixelFormat::Ya8, w, h),
    )
}

#[test]
fn ya8_gray8_roundtrip_via_alpha_drop() {
    // Ya8 → Gray8 drops alpha; Gray8 → Ya8 restores alpha as 255.
    // Equality holds only when the source alpha was already 255.
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = Vec::with_capacity((w * h * 2) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8); // Y
            data.push(255); // A = opaque
        }
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 2) as usize,
            data,
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Ya8, w, h);
    let gray = convert(&src, src_info, PixelFormat::Gray8, &opts).unwrap();
    let gray_info = FrameInfo::new(PixelFormat::Gray8, w, h);
    let back = convert(&gray, gray_info, PixelFormat::Ya8, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn ya8_to_rgba_preserves_alpha() {
    // Ya8 → Rgba: luma is broadcast to R = G = B; alpha is carried
    // through bit-exact. Going back via rgba_to_ya8 must reproduce the
    // original because R = G = B → mean = R = Y.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_ya8(16, 8);
    let rgba = convert(&src, src_info, PixelFormat::Rgba, &opts).unwrap();
    // Spot-check the broadcast.
    for px in 0..(16 * 8) {
        let y = src.planes[0].data[px * 2];
        let a = src.planes[0].data[px * 2 + 1];
        assert_eq!(rgba.planes[0].data[px * 4], y);
        assert_eq!(rgba.planes[0].data[px * 4 + 1], y);
        assert_eq!(rgba.planes[0].data[px * 4 + 2], y);
        assert_eq!(rgba.planes[0].data[px * 4 + 3], a);
    }
    let rgba_info = FrameInfo::new(PixelFormat::Rgba, src_info.width, src_info.height);
    let back = convert(&rgba, rgba_info, PixelFormat::Ya8, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn ya8_to_rgb24_drops_alpha_and_broadcasts() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_ya8(16, 8);
    let rgb = convert(&src, src_info, PixelFormat::Rgb24, &opts).unwrap();
    for px in 0..(16 * 8) {
        let y = src.planes[0].data[px * 2];
        assert_eq!(rgb.planes[0].data[px * 3], y);
        assert_eq!(rgb.planes[0].data[px * 3 + 1], y);
        assert_eq!(rgb.planes[0].data[px * 3 + 2], y);
    }
    // Rgb24 → Ya8 derives Y as mean(R, G, B) = Y (since R = G = B)
    // and sets A = 255.
    let rgb_info = FrameInfo::new(PixelFormat::Rgb24, src_info.width, src_info.height);
    let back = convert(&rgb, rgb_info, PixelFormat::Ya8, &opts).unwrap();
    for px in 0..(16 * 8) {
        assert_eq!(back.planes[0].data[px * 2], src.planes[0].data[px * 2]);
        assert_eq!(back.planes[0].data[px * 2 + 1], 255);
    }
}

#[test]
fn rgb24_to_ya8_luma_is_rounded_mean() {
    // Verify the rounded-mean formula: y = (r + g + b + 1) / 3.
    let opts = ConvertOptions::default();
    let w = 4u32;
    let h = 1u32;
    // Pick (r, g, b) triples with deterministic means.
    let data: Vec<u8> = vec![
        0, 0, 0, // mean 0
        255, 255, 255, // mean 255
        10, 20, 30, // mean (60+1)/3 = 20
        100, 200, 0, // mean (300+1)/3 = 100
    ];
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 3) as usize,
            data,
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Rgb24, w, h);
    let ya = convert(&src, src_info, PixelFormat::Ya8, &opts).unwrap();
    assert_eq!(ya.planes[0].data, vec![0, 255, 255, 255, 20, 255, 100, 255]);
}
