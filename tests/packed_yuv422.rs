//! Packed 4:2:2 (YUYV / UYVY) ↔ planar Yuv422P and ↔ RGB.
//!
//! YUYV (Y0 U Y1 V) and UYVY (U Y0 V Y1) are byte-permutations of the
//! same logical 4:2:2 layout, so:
//!
//! 1. packed → planar → packed must be bit-exact (no information loss).
//! 2. YUYV ↔ UYVY is a pure shuffle and must be bit-exact in both
//!    directions.
//! 3. RGB → packed → RGB must hit the same PSNR floor as the planar
//!    Yuv422P path, since the chroma is the same.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

fn make_frame_single(
    format: PixelFormat,
    w: u32,
    h: u32,
    data: Vec<u8>,
    stride: usize,
) -> (VideoFrame, FrameInfo) {
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data }],
        },
        FrameInfo::new(format, w, h),
    )
}

fn make_planar_yuv422p(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let wu = w as usize;
    let hu = h as usize;
    let cw = wu / 2;
    let mut yp = vec![0u8; wu * hu];
    let mut up = vec![0u8; cw * hu];
    let mut vp = vec![0u8; cw * hu];
    // Asymmetric pattern so a swapped luma/chroma byte trips the test.
    for r in 0..hu {
        for c in 0..wu {
            yp[r * wu + c] = ((r * 7 + c * 13 + 17) & 0xff) as u8;
        }
        for c in 0..cw {
            up[r * cw + c] = ((r * 3 + c * 5 + 47) & 0xff) as u8;
            vp[r * cw + c] = ((r * 11 + c * 19 + 137) & 0xff) as u8;
        }
    }
    let frame = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: wu,
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
        ],
    };
    (frame, FrameInfo::new(PixelFormat::Yuv422P, w, h))
}

#[test]
fn yuyv_planar_roundtrip_is_bit_exact() {
    let opts = ConvertOptions::default();
    let (src, info) = make_planar_yuv422p(16, 8);
    let packed = convert(&src, info, PixelFormat::Yuyv422, &opts).unwrap();
    let packed_info = FrameInfo::new(PixelFormat::Yuyv422, info.width, info.height);
    assert_eq!(packed.planes[0].data.len(), 16 * 8 * 2);
    let back = convert(&packed, packed_info, PixelFormat::Yuv422P, &opts).unwrap();
    for i in 0..3 {
        assert_eq!(
            src.planes[i].data, back.planes[i].data,
            "plane {i} round-trip mismatch (Yuv422P → Yuyv422 → Yuv422P)",
        );
    }
}

#[test]
fn uyvy_planar_roundtrip_is_bit_exact() {
    let opts = ConvertOptions::default();
    let (src, info) = make_planar_yuv422p(16, 8);
    let packed = convert(&src, info, PixelFormat::Uyvy422, &opts).unwrap();
    let packed_info = FrameInfo::new(PixelFormat::Uyvy422, info.width, info.height);
    let back = convert(&packed, packed_info, PixelFormat::Yuv422P, &opts).unwrap();
    for i in 0..3 {
        assert_eq!(
            src.planes[i].data, back.planes[i].data,
            "plane {i} round-trip mismatch (Yuv422P → Uyvy422 → Yuv422P)",
        );
    }
}

#[test]
fn yuyv_uyvy_swap_is_involutive() {
    let opts = ConvertOptions::default();
    // Start from a known YUYV stream.
    let (planar, planar_info) = make_planar_yuv422p(16, 8);
    let yuyv = convert(&planar, planar_info, PixelFormat::Yuyv422, &opts).unwrap();
    let yuyv_info = FrameInfo::new(PixelFormat::Yuyv422, 16, 8);
    let uyvy = convert(&yuyv, yuyv_info, PixelFormat::Uyvy422, &opts).unwrap();
    let uyvy_info = FrameInfo::new(PixelFormat::Uyvy422, 16, 8);
    let yuyv2 = convert(&uyvy, uyvy_info, PixelFormat::Yuyv422, &opts).unwrap();
    assert_eq!(
        yuyv.planes[0].data, yuyv2.planes[0].data,
        "Yuyv → Uyvy → Yuyv must reproduce the original packed bytes",
    );
}

#[test]
fn yuyv_byte_layout_matches_spec() {
    // Build a known YUYV packed quad: (Y0=10, U=20, Y1=30, V=40) for
    // a 2×1 frame. Round-tripping through Yuv422P must put each byte
    // back in the same position.
    let bytes = vec![10u8, 20, 30, 40];
    let (src, info) = make_frame_single(PixelFormat::Yuyv422, 2, 1, bytes.clone(), 4);
    let opts = ConvertOptions::default();
    let planar = convert(&src, info, PixelFormat::Yuv422P, &opts).unwrap();
    assert_eq!(planar.planes[0].data, vec![10, 30], "Y plane Y0|Y1");
    assert_eq!(planar.planes[1].data, vec![20], "U plane");
    assert_eq!(planar.planes[2].data, vec![40], "V plane");
    let planar_info = FrameInfo::new(PixelFormat::Yuv422P, 2, 1);
    let back = convert(&planar, planar_info, PixelFormat::Yuyv422, &opts).unwrap();
    assert_eq!(back.planes[0].data, bytes);
}

#[test]
fn uyvy_byte_layout_matches_spec() {
    // (U=20, Y0=10, V=40, Y1=30) for a 2×1 UYVY frame.
    let bytes = vec![20u8, 10, 40, 30];
    let (src, info) = make_frame_single(PixelFormat::Uyvy422, 2, 1, bytes.clone(), 4);
    let opts = ConvertOptions::default();
    let planar = convert(&src, info, PixelFormat::Yuv422P, &opts).unwrap();
    assert_eq!(planar.planes[0].data, vec![10, 30]);
    assert_eq!(planar.planes[1].data, vec![20]);
    assert_eq!(planar.planes[2].data, vec![40]);
    let planar_info = FrameInfo::new(PixelFormat::Yuv422P, 2, 1);
    let back = convert(&planar, planar_info, PixelFormat::Uyvy422, &opts).unwrap();
    assert_eq!(back.planes[0].data, bytes);
}

#[test]
fn odd_width_packed422_rejected() {
    // A 3×1 frame has no valid YUYV/UYVY representation. The
    // converter must reject rather than silently truncate.
    let opts = ConvertOptions::default();
    let bytes = vec![0u8; 4]; // dummy; conversion must error before reading
    let (_dummy, info) = make_frame_single(PixelFormat::Yuv422P, 3, 1, bytes.clone(), 3);
    // Need 3 planes to match the planar source shape; fudge minimal valid frames.
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: 3,
                data: vec![0; 3],
            },
            VideoPlane {
                stride: 2,
                data: vec![0; 2],
            },
            VideoPlane {
                stride: 2,
                data: vec![0; 2],
            },
        ],
    };
    let err = convert(&src, info, PixelFormat::Yuyv422, &opts).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("even width"),
        "expected even-width refusal, got: {msg}"
    );
}

#[test]
fn yuyv_to_rgb24_matches_planar_route() {
    // Going packed → RGB and planar → RGB on the same logical YUV
    // content must produce identical pixels (the packed path is just
    // a deinterleave-and-call).
    let opts = ConvertOptions::default();
    let (planar, planar_info) = make_planar_yuv422p(32, 16);
    let yuyv = convert(&planar, planar_info, PixelFormat::Yuyv422, &opts).unwrap();
    let yuyv_info = FrameInfo::new(PixelFormat::Yuyv422, 32, 16);

    let direct = convert(&planar, planar_info, PixelFormat::Rgb24, &opts).unwrap();
    let via_packed = convert(&yuyv, yuyv_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(
        direct.planes[0].data, via_packed.planes[0].data,
        "packed YUYV → Rgb24 must match planar Yuv422P → Rgb24 byte-for-byte",
    );
}

#[test]
fn uyvy_to_rgba_alpha_is_255() {
    let opts = ConvertOptions::default();
    let (planar, planar_info) = make_planar_yuv422p(16, 8);
    let uyvy = convert(&planar, planar_info, PixelFormat::Uyvy422, &opts).unwrap();
    let uyvy_info = FrameInfo::new(PixelFormat::Uyvy422, 16, 8);
    let rgba = convert(&uyvy, uyvy_info, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(rgba.planes[0].data.len(), 16 * 8 * 4);
    for px in 0..16 * 8 {
        assert_eq!(rgba.planes[0].data[px * 4 + 3], 255, "alpha at px {px}");
    }
}

#[test]
fn rgb_to_yuyv_psnr_floor() {
    // Same expectation as the planar Yuv422P encode: chroma subsampled
    // by 2 horizontally → PSNR > 30 dB on a smooth gradient is healthy.
    let w = 64;
    let h = 48;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            rgb.push(((x * 255) / (w - 1)) as u8);
            rgb.push(((y * 255) / (h - 1)) as u8);
            rgb.push((((x + y) * 255) / (w + h - 2)) as u8);
        }
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 3,
            data: rgb.clone(),
        }],
    };
    let info = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
    let opts = ConvertOptions::default();
    let yuyv = convert(&src, info, PixelFormat::Yuyv422, &opts).unwrap();
    let yuyv_info = FrameInfo::new(PixelFormat::Yuyv422, w as u32, h as u32);
    let back = convert(&yuyv, yuyv_info, PixelFormat::Rgb24, &opts).unwrap();
    // PSNR.
    let a = &rgb;
    let b = &back.planes[0].data;
    assert_eq!(a.len(), b.len());
    let mut sq = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sq += d * d;
    }
    let mse = sq / a.len() as f64;
    let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
    println!("rgb→yuyv→rgb psnr = {psnr:.2}");
    assert!(psnr > 30.0, "rgb→yuyv→rgb PSNR too low: {psnr}");
}

#[test]
fn rgba_in_alpha_stripped() {
    // RGBA → YUYV should ignore the alpha (it's not part of the
    // packed representation).
    let w = 16usize;
    let h = 8usize;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        rgba.push(100);
        rgba.push(150);
        rgba.push(200);
        rgba.push(33); // arbitrary alpha
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data: rgba,
        }],
    };
    let info = FrameInfo::new(PixelFormat::Rgba, w as u32, h as u32);
    let opts = ConvertOptions::default();
    // Should not panic; the encoded YUYV must match the equivalent Rgb24.
    let mut rgb24 = Vec::with_capacity(w * h * 3);
    for _ in 0..w * h {
        rgb24.push(100);
        rgb24.push(150);
        rgb24.push(200);
    }
    let src_rgb = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 3,
            data: rgb24,
        }],
    };
    let info_rgb = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
    let yuyv_a = convert(&src, info, PixelFormat::Yuyv422, &opts).unwrap();
    let yuyv_b = convert(&src_rgb, info_rgb, PixelFormat::Yuyv422, &opts).unwrap();
    assert_eq!(yuyv_a.planes[0].data, yuyv_b.planes[0].data);
}
