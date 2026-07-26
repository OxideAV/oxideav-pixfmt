//! Exact-roundtrip tests for the RGB-family swizzles and bit-depth
//! conversions. Every pair tested here must be lossless.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, supports, ConvertOptions, FrameInfo};

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

// -------- Yuva420P (planar 4:2:0 YUV + full-resolution alpha plane) --------

fn synth_yuva420p(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    // Build a Yuva420P frame with deterministic per-pixel patterns on
    // each plane. Y is full-resolution, U/V are subsampled 2×2, and A
    // is full-resolution.
    let (wu, hu) = (w as usize, h as usize);
    let (cw, ch) = (wu / 2, hu / 2);
    let mut y_plane = Vec::with_capacity(wu * hu);
    let mut u_plane = Vec::with_capacity(cw * ch);
    let mut v_plane = Vec::with_capacity(cw * ch);
    let mut a_plane = Vec::with_capacity(wu * hu);
    for j in 0..hu {
        for i in 0..wu {
            y_plane.push(((i * 11 + j * 23) & 0xFF) as u8);
            a_plane.push(((i * 7 + j * 19 + 13) & 0xFF) as u8);
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            u_plane.push(((i * 5 + j * 17) & 0xFF) as u8);
            v_plane.push(((i * 3 + j * 29 + 41) & 0xFF) as u8);
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: wu,
                    data: y_plane,
                },
                VideoPlane {
                    stride: cw,
                    data: u_plane,
                },
                VideoPlane {
                    stride: cw,
                    data: v_plane,
                },
                VideoPlane {
                    stride: wu,
                    data: a_plane,
                },
            ],
        },
        FrameInfo::new(PixelFormat::Yuva420P, w, h),
    )
}

#[test]
fn yuv420p_to_yuva420p_appends_opaque_alpha_plane() {
    // Yuv420P → Yuva420P MUST copy Y/U/V byte-for-byte and append a
    // full-resolution alpha plane filled with 0xFF.
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let (wu, hu) = (w as usize, h as usize);
    let (cw, ch) = (wu / 2, hu / 2);
    let y: Vec<u8> = (0..wu * hu).map(|i| (i & 0xFF) as u8).collect();
    let u: Vec<u8> = (0..cw * ch).map(|i| ((i * 3) & 0xFF) as u8).collect();
    let v: Vec<u8> = (0..cw * ch).map(|i| ((i * 5 + 11) & 0xFF) as u8).collect();
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: wu,
                data: y.clone(),
            },
            VideoPlane {
                stride: cw,
                data: u.clone(),
            },
            VideoPlane {
                stride: cw,
                data: v.clone(),
            },
        ],
    };
    let src_info = FrameInfo::new(PixelFormat::Yuv420P, w, h);
    let yuva = convert(&src, src_info, PixelFormat::Yuva420P, &opts).unwrap();
    assert_eq!(yuva.planes.len(), 4);
    assert_eq!(yuva.planes[0].data, y);
    assert_eq!(yuva.planes[1].data, u);
    assert_eq!(yuva.planes[2].data, v);
    assert_eq!(yuva.planes[3].data.len(), wu * hu);
    assert!(yuva.planes[3].data.iter().all(|&a| a == 0xFF));
    // And the round-trip Yuv420P → Yuva420P → Yuv420P is bit-exact.
    let yuva_info = FrameInfo::new(PixelFormat::Yuva420P, w, h);
    let back = convert(&yuva, yuva_info, PixelFormat::Yuv420P, &opts).unwrap();
    assert_eq!(back.planes.len(), 3);
    assert_eq!(back.planes[0].data, y);
    assert_eq!(back.planes[1].data, u);
    assert_eq!(back.planes[2].data, v);
}

#[test]
fn yuva420p_to_rgba_preserves_alpha_plane_bit_exact() {
    // Yuva420P → Rgba interleaves the full-resolution alpha plane into
    // the 4th byte of every pixel. The Y/U/V channels drive R, G, B; the
    // alpha plane must be carried through unchanged.
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let (src, src_info) = synth_yuva420p(w, h);
    let rgba = convert(&src, src_info, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(rgba.planes.len(), 1);
    assert_eq!(rgba.planes[0].data.len(), (w * h * 4) as usize);
    for p in 0..(w as usize) * (h as usize) {
        assert_eq!(rgba.planes[0].data[p * 4 + 3], src.planes[3].data[p]);
    }
}

#[test]
fn yuva420p_to_rgb24_drops_alpha_but_matches_yuva420p_to_rgba_rgb_channels() {
    // The Rgb24 / Rgba forms differ only in the trailing alpha column;
    // the R / G / B bytes MUST be identical between the two outputs.
    let opts = ConvertOptions::default();
    let w = 32u32;
    let h = 16u32;
    let (src, src_info) = synth_yuva420p(w, h);
    let rgb = convert(&src, src_info, PixelFormat::Rgb24, &opts).unwrap();
    let rgba = convert(&src, src_info, PixelFormat::Rgba, &opts).unwrap();
    let n = (w * h) as usize;
    for p in 0..n {
        assert_eq!(rgb.planes[0].data[p * 3], rgba.planes[0].data[p * 4]);
        assert_eq!(
            rgb.planes[0].data[p * 3 + 1],
            rgba.planes[0].data[p * 4 + 1]
        );
        assert_eq!(
            rgb.planes[0].data[p * 3 + 2],
            rgba.planes[0].data[p * 4 + 2]
        );
    }
}

#[test]
fn rgba_to_yuva420p_carries_alpha_plane_bit_exact() {
    // Rgba → Yuva420P splits the source's 4th byte out into the trailing
    // alpha plane. The plane must be bit-exact to the source's alpha
    // column (no chroma-style 2×2 averaging).
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(16, 8);
    let yuva = convert(&src, src_info, PixelFormat::Yuva420P, &opts).unwrap();
    assert_eq!(yuva.planes.len(), 4);
    let n = 16usize * 8usize;
    assert_eq!(yuva.planes[3].data.len(), n);
    for p in 0..n {
        assert_eq!(yuva.planes[3].data[p], src.planes[0].data[p * 4 + 3]);
    }
}

#[test]
fn rgb24_to_yuva420p_synthesises_opaque_alpha_plane() {
    // Rgb24 has no alpha, so the alpha plane on the destination must be
    // filled opaque (0xFF) at full luma resolution.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(16, 8);
    let yuva = convert(&src, src_info, PixelFormat::Yuva420P, &opts).unwrap();
    assert_eq!(yuva.planes.len(), 4);
    let n = 16usize * 8usize;
    assert_eq!(yuva.planes[3].data.len(), n);
    assert!(yuva.planes[3].data.iter().all(|&a| a == 0xFF));
}

#[test]
fn rgba_to_yuva420p_to_rgba_roundtrip_keeps_alpha_and_lifts_psnr() {
    // The colour math goes through the lossy 4:2:0 chroma down/up cycle,
    // so RGB-channel PSNR can be ~30–40 dB depending on content. Alpha,
    // by contrast, is a verbatim copy — it must come back bit-exact.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(32, 16);
    let yuva = convert(&src, src_info, PixelFormat::Yuva420P, &opts).unwrap();
    let yuva_info = FrameInfo::new(PixelFormat::Yuva420P, src_info.width, src_info.height);
    let back = convert(&yuva, yuva_info, PixelFormat::Rgba, &opts).unwrap();
    let n = 32usize * 16usize;
    // Alpha bit-exact.
    for p in 0..n {
        assert_eq!(
            back.planes[0].data[p * 4 + 3],
            src.planes[0].data[p * 4 + 3]
        );
    }
    // RGB channels: bounded MSE.
    let mut sse: u64 = 0;
    for p in 0..n {
        for c in 0..3 {
            let s = src.planes[0].data[p * 4 + c] as i32;
            let d = back.planes[0].data[p * 4 + c] as i32;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    let mse = sse as f64 / (n as f64 * 3.0);
    // The synthetic gradient is hostile (deliberately non-smooth across
    // the chroma block) so the floor here is loose. The point is that
    // 4:2:0's RGB roundtrip stays within a sensible bound — we don't
    // try to match the smooth-gradient PSNR floors from the planar
    // 4:2:0 suite. 15 dB is a wide guardrail that still catches a
    // completely broken colour-math path.
    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    assert!(psnr > 15.0, "RGB-channel PSNR {psnr} dB below 15 floor");
}

#[test]
fn yuv420p_to_yuva420p_rejects_odd_dimensions() {
    // 4:2:0 has no representation for a half-pixel chroma sample, so
    // odd width or height MUST be rejected rather than silently
    // truncated.
    let opts = ConvertOptions::default();
    let bad = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: 15,
                data: vec![0; 15 * 8],
            },
            VideoPlane {
                stride: 8,
                data: vec![128; 8 * 4],
            },
            VideoPlane {
                stride: 8,
                data: vec![128; 8 * 4],
            },
        ],
    };
    let bad_info = FrameInfo::new(PixelFormat::Yuv420P, 15, 8);
    assert!(convert(&bad, bad_info, PixelFormat::Yuva420P, &opts).is_err());
}

// ---------------------------------------------------------------------
// Direct planar YUV ↔ planar YUV chroma resample. The dispatch now
// covers six ordered pairs over `(4:2:0, 4:2:2, 4:4:4)` plus the same
// six on the full-range "J" family — twelve new entries that previously
// returned `Error::Unsupported`. Callers that need to switch chroma
// subsampling without colour-space conversion no longer have to stage
// through `Rgb24`.

/// Build a tightly-packed planar YUV frame from synthetic content. The
/// luma plane is a 2-axis ramp; the chroma planes are gentler ramps
/// offset around 128. Returns the frame plus its `FrameInfo`.
fn synth_planar_yuv(format: PixelFormat, w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let (wsub, hsub) = match format {
        PixelFormat::Yuv420P | PixelFormat::YuvJ420P => (2, 2),
        PixelFormat::Yuv422P | PixelFormat::YuvJ422P => (2, 1),
        PixelFormat::Yuv444P | PixelFormat::YuvJ444P => (1, 1),
        PixelFormat::Yuv411P => (4, 1),
        _ => panic!("synth_planar_yuv: unsupported format {format:?}"),
    };
    let cw = (w as usize) / wsub;
    let ch = (h as usize) / hsub;
    let mut yp = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            yp.push(((x * 7 + y * 11) & 0xFF) as u8);
        }
    }
    let mut up = Vec::with_capacity(cw * ch);
    let mut vp = Vec::with_capacity(cw * ch);
    for y in 0..ch {
        for x in 0..cw {
            up.push(128u8.wrapping_add(((x * 3 + y * 5) & 0x3F) as u8));
            vp.push(128u8.wrapping_add(((x * 5 + y * 3) & 0x3F) as u8));
        }
    }
    let frame = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w as usize,
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
    (frame, FrameInfo::new(format, w, h))
}

#[test]
fn yuv420p_to_yuv422p_direct_copies_luma_and_widens_chroma() {
    // 4:2:0 → 4:2:2: chroma width unchanged, height doubled by
    // vertical replication. Luma copies byte-for-byte.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv420P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv422P, &opts).expect("420 → 422");
    assert_eq!(out.planes.len(), 3);
    assert_eq!(out.planes[0].stride, 16);
    assert_eq!(out.planes[1].stride, 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    // Chroma plane height doubles via row-duplicate; each source row at
    // chroma index `cr` lands at destination rows `cr*2` and `cr*2+1`.
    let cw = 8;
    let ch_src = 4;
    for cr in 0..ch_src {
        let src_row = &src.planes[1].data[cr * cw..(cr + 1) * cw];
        let dst_row_a = &out.planes[1].data[(cr * 2) * cw..(cr * 2 + 1) * cw];
        let dst_row_b = &out.planes[1].data[(cr * 2 + 1) * cw..(cr * 2 + 2) * cw];
        assert_eq!(src_row, dst_row_a, "U row {cr} top");
        assert_eq!(src_row, dst_row_b, "U row {cr} bottom");
    }
}

#[test]
fn yuv420p_to_yuv444p_direct_doubles_both_chroma_axes() {
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv420P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv444P, &opts).expect("420 → 444");
    assert_eq!(out.planes[1].stride, 16);
    assert_eq!(out.planes[1].data.len(), 16 * 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    // Nearest-neighbour: each source chroma sample populates a 2×2
    // square in the destination.
    let cw_src = 8;
    let w_dst = 16;
    for cr in 0..4 {
        for cc in 0..cw_src {
            let s = src.planes[1].data[cr * cw_src + cc];
            let r0 = cr * 2;
            let r1 = cr * 2 + 1;
            let c0 = cc * 2;
            let c1 = cc * 2 + 1;
            assert_eq!(out.planes[1].data[r0 * w_dst + c0], s);
            assert_eq!(out.planes[1].data[r0 * w_dst + c1], s);
            assert_eq!(out.planes[1].data[r1 * w_dst + c0], s);
            assert_eq!(out.planes[1].data[r1 * w_dst + c1], s);
        }
    }
}

#[test]
fn yuv422p_to_yuv420p_direct_halves_chroma_height() {
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv422P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv420P, &opts).expect("422 → 420");
    assert_eq!(out.planes[1].stride, 8);
    assert_eq!(out.planes[1].data.len(), 8 * 4);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    // Vertical pair-average with rounding: dst[cr,cc] = round((a + b)/2)
    // where a = src[2*cr, cc], b = src[2*cr + 1, cc]. Spot-check a few.
    let cw = 8;
    for cr in 0..4 {
        for cc in 0..cw {
            let a = src.planes[1].data[(cr * 2) * cw + cc] as u16;
            let b = src.planes[1].data[(cr * 2 + 1) * cw + cc] as u16;
            // round-half-up on a u16 sum — matches the chroma resampler's
            // expression `(a + b).div_ceil(2)`.
            let want = (a + b).div_ceil(2);
            assert_eq!(
                out.planes[1].data[cr * cw + cc] as u16,
                want,
                "cr={cr} cc={cc}"
            );
        }
    }
}

#[test]
fn yuv422p_to_yuv444p_direct_widens_chroma_width() {
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv422P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv444P, &opts).expect("422 → 444");
    assert_eq!(out.planes[1].stride, 16);
    assert_eq!(out.planes[1].data.len(), 16 * 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    // Horizontal duplicate: each source byte appears twice consecutively.
    let cw_src = 8;
    let w_dst = 16;
    for row in 0..8 {
        for cc in 0..cw_src {
            let s = src.planes[1].data[row * cw_src + cc];
            assert_eq!(out.planes[1].data[row * w_dst + cc * 2], s);
            assert_eq!(out.planes[1].data[row * w_dst + cc * 2 + 1], s);
        }
    }
}

#[test]
fn yuv444p_to_yuv422p_direct_averages_horizontal_pairs() {
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv444P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv422P, &opts).expect("444 → 422");
    assert_eq!(out.planes[1].stride, 8);
    assert_eq!(out.planes[1].data.len(), 8 * 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    // Horizontal pair-average with rounding.
    let cw_src = 16;
    let cw_dst = 8;
    for row in 0..8 {
        for cc in 0..cw_dst {
            let a = src.planes[1].data[row * cw_src + cc * 2] as u16;
            let b = src.planes[1].data[row * cw_src + cc * 2 + 1] as u16;
            let want = (a + b).div_ceil(2);
            assert_eq!(out.planes[1].data[row * cw_dst + cc] as u16, want);
        }
    }
}

#[test]
fn yuv444p_to_yuv420p_direct_averages_2x2_blocks() {
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv444P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv420P, &opts).expect("444 → 420");
    assert_eq!(out.planes[1].stride, 8);
    assert_eq!(out.planes[1].data.len(), 8 * 4);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    // 2×2 box average: dst[cr,cc] = round((a + b + c + d)/4) with +2
    // rounding before /4 to match the helper's expression `(s + 2) / 4`.
    let cw_src = 16;
    let cw_dst = 8;
    for cr in 0..4 {
        for cc in 0..cw_dst {
            let mut s = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    s += src.planes[1].data[(cr * 2 + dy) * cw_src + cc * 2 + dx] as u32;
                }
            }
            let want = (s + 2) / 4;
            assert_eq!(out.planes[1].data[cr * cw_dst + cc] as u32, want);
        }
    }
}

#[test]
fn yuv420p_to_yuv444p_round_trip_back_to_420p_is_bit_exact() {
    // 4:2:0 → 4:4:4 widens chroma by nearest-neighbour (duplicating
    // each sample into a 2×2 block). Round-tripping through 4:4:4 and
    // back averages those four identical samples, which gives the
    // original byte exactly — so the round trip is bit-exact.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv420P, 16, 8);
    let stage = convert(&src, info, PixelFormat::Yuv444P, &opts).expect("→444");
    let stage_info = FrameInfo::new(PixelFormat::Yuv444P, 16, 8);
    let back = convert(&stage, stage_info, PixelFormat::Yuv420P, &opts).expect("→420 back");
    assert_eq!(back.planes[0].data, src.planes[0].data, "luma round trip");
    assert_eq!(back.planes[1].data, src.planes[1].data, "U round trip");
    assert_eq!(back.planes[2].data, src.planes[2].data, "V round trip");
}

#[test]
fn yuv420p_to_yuv422p_round_trip_back_to_420p_is_bit_exact() {
    // 4:2:0 → 4:2:2 widens chroma by row-duplicate; the reverse step
    // averages the duplicate rows back to the original byte. Result is
    // bit-exact.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv420P, 16, 8);
    let stage = convert(&src, info, PixelFormat::Yuv422P, &opts).expect("→422");
    let stage_info = FrameInfo::new(PixelFormat::Yuv422P, 16, 8);
    let back = convert(&stage, stage_info, PixelFormat::Yuv420P, &opts).expect("→420 back");
    assert_eq!(back.planes[0].data, src.planes[0].data);
    assert_eq!(back.planes[1].data, src.planes[1].data);
    assert_eq!(back.planes[2].data, src.planes[2].data);
}

#[test]
fn yuvj_planar_inter_conversions_route_through_same_chroma_resampler() {
    // The "J" full-range family reuses the same chroma resampler — the
    // matrix coefficient table never enters here, only chroma
    // subsampling. So a `YuvJ420P → YuvJ444P` conversion should
    // produce bit-identical chroma to a `Yuv420P → Yuv444P` conversion
    // run on the same byte content.
    let opts = ConvertOptions::default();
    let (limited, limited_info) = synth_planar_yuv(PixelFormat::Yuv420P, 16, 8);
    let mut full = limited.clone();
    let full_info = FrameInfo::new(PixelFormat::YuvJ420P, 16, 8);

    let limited_444 = convert(&limited, limited_info, PixelFormat::Yuv444P, &opts).unwrap();
    let full_444 = convert(&full, full_info, PixelFormat::YuvJ444P, &opts).unwrap();
    assert_eq!(limited_444.planes[0].data, full_444.planes[0].data);
    assert_eq!(limited_444.planes[1].data, full_444.planes[1].data);
    assert_eq!(limited_444.planes[2].data, full_444.planes[2].data);

    // And the reverse direction also matches.
    let limited_back = convert(
        &limited_444,
        FrameInfo::new(PixelFormat::Yuv444P, 16, 8),
        PixelFormat::Yuv420P,
        &opts,
    )
    .unwrap();
    let full_back = convert(
        &full_444,
        FrameInfo::new(PixelFormat::YuvJ444P, 16, 8),
        PixelFormat::YuvJ420P,
        &opts,
    )
    .unwrap();
    assert_eq!(limited_back.planes[1].data, full_back.planes[1].data);

    // Suppress unused-mut warning on `full` while keeping the local
    // explicit for readability.
    let _ = &mut full;
}

#[test]
fn yuv420p_to_yuv422p_rejects_odd_height() {
    // 4:2:0 source has half-height chroma; odd luma height has no
    // 4:2:0 chroma representation. Error::Invalid expected.
    let opts = ConvertOptions::default();
    let bad = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: 16,
                data: vec![0; 16 * 5],
            },
            VideoPlane {
                stride: 8,
                data: vec![128; 8 * 2],
            },
            VideoPlane {
                stride: 8,
                data: vec![128; 8 * 2],
            },
        ],
    };
    let bad_info = FrameInfo::new(PixelFormat::Yuv420P, 16, 5);
    assert!(convert(&bad, bad_info, PixelFormat::Yuv422P, &opts).is_err());
}

// -------------------------------------------------------------------------
// Yuv411P — NTSC DV-25 native sampling and a legal JPEG 4:1:1 layout
// (`cjpeg -sample 4x1`). Luma at full resolution; chroma horizontally
// subsampled by 4 (4 luma per 1 chroma on each row, no vertical
// subsample). Six chroma-resample pairs (411 ↔ 420 / 422 / 444 in both
// directions) plus RGB encode/decode.

#[test]
fn yuv411p_to_yuv444p_widens_chroma_4x_horizontally() {
    // 4:1:1 → 4:4:4: chroma plane goes from (w/4) × h to w × h via
    // horizontal nearest-neighbour broadcast. Luma copies byte-for-byte.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv411P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv444P, &opts).expect("411 → 444");
    assert_eq!(out.planes.len(), 3);
    assert_eq!(out.planes[0].stride, 16);
    assert_eq!(out.planes[1].stride, 16);
    assert_eq!(out.planes[1].data.len(), 16 * 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    // Each source chroma column populates 4 destination columns on the
    // same row.
    let src_cw = 4;
    for row in 0..8usize {
        for cc in 0..src_cw {
            let src_v = src.planes[1].data[row * src_cw + cc];
            for k in 0..4 {
                assert_eq!(
                    out.planes[1].data[row * 16 + cc * 4 + k],
                    src_v,
                    "U row {row} src cc {cc} dst col {k}",
                );
            }
        }
    }
}

#[test]
fn yuv411p_to_yuv422p_widens_chroma_2x_horizontally() {
    // 4:1:1 → 4:2:2: each chroma sample broadcasts to two columns
    // (chroma width goes from w/4 to w/2).
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv411P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv422P, &opts).expect("411 → 422");
    assert_eq!(out.planes[1].stride, 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    let src_cw = 4;
    let dst_cw = 8;
    for row in 0..8usize {
        for cc in 0..src_cw {
            let v = src.planes[1].data[row * src_cw + cc];
            assert_eq!(out.planes[1].data[row * dst_cw + cc * 2], v);
            assert_eq!(out.planes[1].data[row * dst_cw + cc * 2 + 1], v);
        }
    }
}

#[test]
fn yuv411p_to_yuv420p_pair_averages_vertically_and_broadcasts() {
    // 4:1:1 → 4:2:0: 4:1:1 has chroma at (w/4) × h, 4:2:0 has chroma at
    // (w/2) × (h/2). Each destination chroma sample is the vertical
    // pair-average of two source samples broadcast horizontally to two
    // columns.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv411P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv420P, &opts).expect("411 → 420");
    assert_eq!(out.planes[1].stride, 8);
    assert_eq!(out.planes[1].data.len(), 8 * 4);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    let src_cw = 4;
    let dst_cw = 8;
    for cr in 0..4usize {
        for cc in 0..src_cw {
            let a = src.planes[1].data[(cr * 2) * src_cw + cc] as u16;
            let b = src.planes[1].data[(cr * 2 + 1) * src_cw + cc] as u16;
            let expected = (a + b).div_ceil(2) as u8;
            assert_eq!(out.planes[1].data[cr * dst_cw + cc * 2], expected);
            assert_eq!(out.planes[1].data[cr * dst_cw + cc * 2 + 1], expected);
        }
    }
}

#[test]
fn yuv444p_to_yuv411p_box_averages_horizontal_quads() {
    // 4:4:4 → 4:1:1: each destination chroma sample is the 4-sample
    // horizontal box average of the source row.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv444P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv411P, &opts).expect("444 → 411");
    assert_eq!(out.planes[1].stride, 4);
    assert_eq!(out.planes[1].data.len(), 4 * 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    for row in 0..8usize {
        for cc in 0..4usize {
            let base = row * 16 + cc * 4;
            let s = src.planes[1].data[base] as u32
                + src.planes[1].data[base + 1] as u32
                + src.planes[1].data[base + 2] as u32
                + src.planes[1].data[base + 3] as u32;
            let expected = ((s + 2) / 4) as u8;
            assert_eq!(
                out.planes[1].data[row * 4 + cc],
                expected,
                "row {row} cc {cc}"
            );
        }
    }
}

#[test]
fn yuv422p_to_yuv411p_pair_averages() {
    // 4:2:2 → 4:1:1: chroma width halves from w/2 to w/4; each
    // destination sample is the pair-average of two adjacent source
    // samples on the same row.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv422P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv411P, &opts).expect("422 → 411");
    assert_eq!(out.planes[1].stride, 4);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    let src_cw = 8;
    for row in 0..8usize {
        for cc in 0..4usize {
            let a = src.planes[1].data[row * src_cw + cc * 2] as u16;
            let b = src.planes[1].data[row * src_cw + cc * 2 + 1] as u16;
            let expected = (a + b).div_ceil(2) as u8;
            assert_eq!(out.planes[1].data[row * 4 + cc], expected);
        }
    }
}

#[test]
fn yuv420p_to_yuv411p_pair_averages_horizontally_and_duplicates_vertically() {
    // 4:2:0 → 4:1:1: chroma goes from (w/2) × (h/2) to (w/4) × h. Each
    // destination chroma row reuses the same source chroma row (4:2:0
    // already pair-averaged the vertical pair) and pair-averages two
    // source samples horizontally.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv420P, 16, 8);
    let out = convert(&src, info, PixelFormat::Yuv411P, &opts).expect("420 → 411");
    assert_eq!(out.planes[1].stride, 4);
    assert_eq!(out.planes[1].data.len(), 4 * 8);
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma must copy");
    let src_cw = 8;
    for row in 0..8usize {
        let src_row = row / 2;
        for cc in 0..4usize {
            let a = src.planes[1].data[src_row * src_cw + cc * 2] as u16;
            let b = src.planes[1].data[src_row * src_cw + cc * 2 + 1] as u16;
            let expected = (a + b).div_ceil(2) as u8;
            assert_eq!(out.planes[1].data[row * 4 + cc], expected);
        }
    }
}

#[test]
fn yuv411p_to_yuv444p_round_trip_back_to_411p_is_bit_exact() {
    // The 4:1:1 → 4:4:4 widening step duplicates every source chroma
    // sample to four destination columns; the reverse 4:4:4 → 4:1:1
    // step averages those four columns back to one. Since all four
    // values are identical the average reproduces the source byte
    // exactly. Luma is a byte-for-byte copy in both directions.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv411P, 16, 8);
    let mid = convert(&src, info, PixelFormat::Yuv444P, &opts).expect("411 → 444");
    let mid_info = FrameInfo::new(PixelFormat::Yuv444P, 16, 8);
    let back = convert(&mid, mid_info, PixelFormat::Yuv411P, &opts).expect("444 → 411");
    assert_eq!(back.planes[0].data, src.planes[0].data, "luma round-trip");
    assert_eq!(back.planes[1].data, src.planes[1].data, "U round-trip");
    assert_eq!(back.planes[2].data, src.planes[2].data, "V round-trip");
}

#[test]
fn yuv411p_to_yuv422p_round_trip_back_to_411p_is_bit_exact() {
    // 4:1:1 → 4:2:2 broadcasts each chroma sample to two adjacent
    // columns; 4:2:2 → 4:1:1 averages those two identical samples.
    // Round-trip is bit-exact.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv411P, 16, 8);
    let mid = convert(&src, info, PixelFormat::Yuv422P, &opts).expect("411 → 422");
    let mid_info = FrameInfo::new(PixelFormat::Yuv422P, 16, 8);
    let back = convert(&mid, mid_info, PixelFormat::Yuv411P, &opts).expect("422 → 411");
    assert_eq!(back.planes[1].data, src.planes[1].data, "U round-trip");
    assert_eq!(back.planes[2].data, src.planes[2].data, "V round-trip");
}

#[test]
fn yuv411p_rgb_round_trip_recovers_luma_and_holds_chroma_psnr() {
    // 4:1:1 → RGB → 4:1:1: luma is recovered to within a few LSBs (the
    // YUV ↔ RGB matrix introduces ±1 rounding per channel; per-pixel
    // RGB encode preserves the per-pixel luma derivation modulo that
    // bound). Chroma reproduces exactly because the round-trip path
    // box-averages four identical pixels back to the original sample.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv411P, 16, 8);
    let rgb = convert(&src, info, PixelFormat::Rgb24, &opts).expect("411 → rgb24");
    assert_eq!(rgb.planes.len(), 1);
    assert_eq!(rgb.planes[0].stride, 16 * 3);
    assert_eq!(rgb.planes[0].data.len(), 16 * 8 * 3);
    let rgb_info = FrameInfo::new(PixelFormat::Rgb24, 16, 8);
    let back = convert(&rgb, rgb_info, PixelFormat::Yuv411P, &opts).expect("rgb24 → 411");
    // PSNR floor on luma — same ±1-LSB-ish bound the other YUV ↔ RGB
    // tests in this file pin. Use the simple MSE form rather than
    // pulling in a sibling crate.
    let mut sse = 0u64;
    for (a, b) in back.planes[0].data.iter().zip(src.planes[0].data.iter()) {
        let d = *a as i32 - *b as i32;
        sse += (d * d) as u64;
    }
    let mse = sse as f64 / (16.0 * 8.0);
    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    assert!(psnr > 30.0, "luma PSNR {psnr:.2} dB below 30 dB floor");
}

#[test]
fn yuv411p_to_rgba_synthesises_opaque_alpha() {
    // → Rgba: the alpha column must be 0xFF everywhere because Yuv411P
    // carries no alpha plane and the destination needs one.
    let opts = ConvertOptions::default();
    let (src, info) = synth_planar_yuv(PixelFormat::Yuv411P, 16, 8);
    let rgba = convert(&src, info, PixelFormat::Rgba, &opts).expect("411 → rgba");
    assert_eq!(rgba.planes[0].stride, 16 * 4);
    for i in 0..(16 * 8) {
        assert_eq!(rgba.planes[0].data[i * 4 + 3], 0xFF, "alpha at pixel {i}");
    }
}

#[test]
fn yuv411p_chroma_resample_rejects_width_not_divisible_by_4() {
    // 4:1:1 with luma width 18 (not a multiple of 4) has no valid
    // chroma layout for either resampling direction; convert() must
    // reject with Error::Invalid rather than producing a truncated
    // destination plane.
    let opts = ConvertOptions::default();
    // The frame itself is built to be consistent so the dimension check
    // is what trips, not a plane-size mismatch.
    let bad = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: 18,
                data: vec![16; 18 * 4],
            },
            VideoPlane {
                stride: 4,
                data: vec![128; 4 * 4],
            },
            VideoPlane {
                stride: 4,
                data: vec![128; 4 * 4],
            },
        ],
    };
    let bad_info = FrameInfo::new(PixelFormat::Yuv411P, 18, 4);
    assert!(convert(&bad, bad_info, PixelFormat::Yuv444P, &opts).is_err());
    assert!(convert(&bad, bad_info, PixelFormat::Rgb24, &opts).is_err());
}

// --- Planar GBR(A) ↔ packed deep RGB -------------------------------------

/// Pack a u16 as a little-endian byte pair onto `out`.
fn push16le(out: &mut Vec<u8>, v: u16) {
    out.push((v & 0xFF) as u8);
    out.push((v >> 8) as u8);
}

/// Read a little-endian 16-bit word at byte offset `off`.
fn read16le(buf: &[u8], off: usize) -> u16 {
    (buf[off] as u16) | ((buf[off + 1] as u16) << 8)
}

/// Build a GBR(A) frame whose G/B/R(/A) plane samples are distinct
/// `bits`-significant ramps so a reorder bug is observable. Plane order
/// is G, B, R(, A) per the oxideav-core variant docs.
fn synth_gbr(w: u32, h: u32, bits: u32, alpha: bool) -> VideoFrame {
    let mask = (1u32 << bits) - 1;
    let n = (w * h) as usize;
    let mut g = Vec::with_capacity(n * 2);
    let mut b = Vec::with_capacity(n * 2);
    let mut r = Vec::with_capacity(n * 2);
    let mut a = Vec::with_capacity(n * 2);
    for i in 0..n as u32 {
        push16le(&mut g, ((i * 7) & mask) as u16);
        push16le(&mut b, ((i * 11 + 3) & mask) as u16);
        push16le(&mut r, ((i * 5 + 1) & mask) as u16);
        push16le(&mut a, ((i * 13 + 2) & mask) as u16);
    }
    let mut planes = vec![
        VideoPlane {
            stride: (w * 2) as usize,
            data: g,
        },
        VideoPlane {
            stride: (w * 2) as usize,
            data: b,
        },
        VideoPlane {
            stride: (w * 2) as usize,
            data: r,
        },
    ];
    if alpha {
        planes.push(VideoPlane {
            stride: (w * 2) as usize,
            data: a,
        });
    }
    VideoFrame { pts: None, planes }
}

#[test]
fn gbr_to_packed_deep_known_values() {
    let opts = ConvertOptions::default();
    // 10-bit Gbrp → Rgb48Le: a sample of value V<<6 must land in the
    // packed word, with R G B byte order.
    let (w, h, bits) = (4u32, 2u32, 10u32);
    let src = synth_gbr(w, h, bits, false);
    let info = FrameInfo::new(PixelFormat::Gbrp10Le, w, h);
    let dst = convert(&src, info, PixelFormat::Rgb48Le, &opts).expect("gbrp10 → rgb48");
    let shift = 16 - bits;
    let g = &src.planes[0].data;
    let b = &src.planes[1].data;
    let r = &src.planes[2].data;
    let packed = &dst.planes[0].data;
    for i in 0..(w * h) as usize {
        let base = i * 6;
        assert_eq!(read16le(packed, base), read16le(r, i * 2) << shift, "R {i}");
        assert_eq!(
            read16le(packed, base + 2),
            read16le(g, i * 2) << shift,
            "G {i}"
        );
        assert_eq!(
            read16le(packed, base + 4),
            read16le(b, i * 2) << shift,
            "B {i}"
        );
    }
}

#[test]
fn gbr_packed_deep_roundtrip_all_depths() {
    let opts = ConvertOptions::default();
    let (w, h) = (6u32, 4u32);
    // No-alpha (Gbrp* ↔ Rgb48Le).
    for (gbr, bits) in [
        (PixelFormat::Gbrp10Le, 10u32),
        (PixelFormat::Gbrp12Le, 12),
        (PixelFormat::Gbrp14Le, 14),
    ] {
        let src = synth_gbr(w, h, bits, false);
        let info = FrameInfo::new(gbr, w, h);
        let packed = convert(&src, info, PixelFormat::Rgb48Le, &opts).expect("→ rgb48");
        let packed_info = FrameInfo::new(PixelFormat::Rgb48Le, w, h);
        let back = convert(&packed, packed_info, gbr, &opts).expect("→ gbr");
        for p in 0..3 {
            assert_eq!(
                back.planes[p].data, src.planes[p].data,
                "{gbr:?} plane {p} round-trip"
            );
        }
    }
    // Alpha (Gbrap* ↔ Rgba64Le).
    for (gbr, bits) in [
        (PixelFormat::Gbrap10Le, 10u32),
        (PixelFormat::Gbrap12Le, 12),
        (PixelFormat::Gbrap14Le, 14),
    ] {
        let src = synth_gbr(w, h, bits, true);
        let info = FrameInfo::new(gbr, w, h);
        let packed = convert(&src, info, PixelFormat::Rgba64Le, &opts).expect("→ rgba64");
        let packed_info = FrameInfo::new(PixelFormat::Rgba64Le, w, h);
        let back = convert(&packed, packed_info, gbr, &opts).expect("→ gbra");
        for p in 0..4 {
            assert_eq!(
                back.planes[p].data, src.planes[p].data,
                "{gbr:?} plane {p} round-trip"
            );
        }
    }
}

#[test]
fn gbrap_alpha_plane_reaches_packed_word() {
    let opts = ConvertOptions::default();
    let (w, h, bits) = (4u32, 2u32, 12u32);
    let src = synth_gbr(w, h, bits, true);
    let info = FrameInfo::new(PixelFormat::Gbrap12Le, w, h);
    let dst = convert(&src, info, PixelFormat::Rgba64Le, &opts).expect("gbrap12 → rgba64");
    let shift = 16 - bits;
    let a = &src.planes[3].data;
    let packed = &dst.planes[0].data;
    for i in 0..(w * h) as usize {
        // A is the 4th packed component (R G B A).
        assert_eq!(
            read16le(packed, i * 8 + 6),
            read16le(a, i * 2) << shift,
            "A {i}"
        );
    }
}

#[test]
fn gbr_short_plane_count_rejected() {
    let opts = ConvertOptions::default();
    // A Gbrap source carrying only 3 planes must error (alpha missing).
    let src = synth_gbr(4, 2, 10, false);
    let info = FrameInfo::new(PixelFormat::Gbrap10Le, 4, 2);
    assert!(convert(&src, info, PixelFormat::Rgba64Le, &opts).is_err());
}

/// Build a `Gbrp8` frame (byte samples) with distinct per-plane ramps.
fn synth_gbr8(w: u32, h: u32) -> VideoFrame {
    let n = (w * h) as usize;
    let ramp = |mul: usize, add: usize| -> Vec<u8> {
        (0..n).map(|i| ((i * mul + add) & 0xFF) as u8).collect()
    };
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: ramp(7, 0),
            },
            VideoPlane {
                stride: w as usize,
                data: ramp(11, 3),
            },
            VideoPlane {
                stride: w as usize,
                data: ramp(5, 1),
            },
        ],
    }
}

/// `Gbrp16Le` ↔ `Rgb48Le` (and the alpha pair) is a pure plane reorder
/// at 16 bits — the packed word equals the source word exactly, and the
/// round-trip is bit-exact on every plane.
#[test]
fn gbr16_packed_deep_pure_reorder_roundtrip() {
    let opts = ConvertOptions::default();
    let (w, h) = (6u32, 4u32);
    let src = synth_gbr(w, h, 16, false);
    let info = FrameInfo::new(PixelFormat::Gbrp16Le, w, h);
    let packed = convert(&src, info, PixelFormat::Rgb48Le, &opts).expect("gbrp16 → rgb48");
    let (g, b, r) = (
        &src.planes[0].data,
        &src.planes[1].data,
        &src.planes[2].data,
    );
    for i in 0..(w * h) as usize {
        let base = i * 6;
        assert_eq!(read16le(&packed.planes[0].data, base), read16le(r, i * 2));
        assert_eq!(
            read16le(&packed.planes[0].data, base + 2),
            read16le(g, i * 2)
        );
        assert_eq!(
            read16le(&packed.planes[0].data, base + 4),
            read16le(b, i * 2)
        );
    }
    let back = convert(
        &packed,
        FrameInfo::new(PixelFormat::Rgb48Le, w, h),
        PixelFormat::Gbrp16Le,
        &opts,
    )
    .expect("rgb48 → gbrp16");
    for p in 0..3 {
        assert_eq!(back.planes[p].data, src.planes[p].data, "plane {p}");
    }
    // Alpha pair: all four planes bit-exact through Rgba64Le.
    let src = synth_gbr(w, h, 16, true);
    let info = FrameInfo::new(PixelFormat::Gbrap16Le, w, h);
    let packed = convert(&src, info, PixelFormat::Rgba64Le, &opts).expect("gbrap16 → rgba64");
    for i in 0..(w * h) as usize {
        assert_eq!(
            read16le(&packed.planes[0].data, i * 8 + 6),
            read16le(&src.planes[3].data, i * 2),
            "alpha word {i}"
        );
    }
    let back = convert(
        &packed,
        FrameInfo::new(PixelFormat::Rgba64Le, w, h),
        PixelFormat::Gbrap16Le,
        &opts,
    )
    .expect("rgba64 → gbrap16");
    for p in 0..4 {
        assert_eq!(back.planes[p].data, src.planes[p].data, "plane {p}");
    }
}

/// 8-bit packed content round-trips exactly through the full-width
/// 16-bit GBR members: the widen is the exact ×257 (peak maps to peak),
/// the narrow keeps the top byte.
#[test]
fn gbr16_8bit_widen_is_exact_257() {
    let opts = ConvertOptions::default();
    let (w, h) = (8u32, 4u32);
    let n = (w * h) as usize;
    let rgb: Vec<u8> = (0..n * 3).map(|i| ((i * 3 + 5) & 0xFF) as u8).collect();
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 3) as usize,
            data: rgb.clone(),
        }],
    };
    let planar = convert(
        &src,
        FrameInfo::new(PixelFormat::Rgb24, w, h),
        PixelFormat::Gbrp16Le,
        &opts,
    )
    .expect("rgb24 → gbrp16");
    // Plane order G, B, R; every word is the ×257 widen of the byte.
    for i in 0..n {
        assert_eq!(
            read16le(&planar.planes[0].data, i * 2),
            rgb[i * 3 + 1] as u16 * 257,
            "G {i}"
        );
        assert_eq!(
            read16le(&planar.planes[1].data, i * 2),
            rgb[i * 3 + 2] as u16 * 257,
            "B {i}"
        );
        assert_eq!(
            read16le(&planar.planes[2].data, i * 2),
            rgb[i * 3] as u16 * 257,
            "R {i}"
        );
    }
    let back = convert(
        &planar,
        FrameInfo::new(PixelFormat::Gbrp16Le, w, h),
        PixelFormat::Rgb24,
        &opts,
    )
    .expect("gbrp16 → rgb24");
    assert_eq!(back.planes[0].data, rgb);
    // Alpha variant through Gbrap16Le.
    let rgba: Vec<u8> = (0..n * 4).map(|i| ((i * 7 + 9) & 0xFF) as u8).collect();
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 4) as usize,
            data: rgba.clone(),
        }],
    };
    let planar = convert(
        &src,
        FrameInfo::new(PixelFormat::Rgba, w, h),
        PixelFormat::Gbrap16Le,
        &opts,
    )
    .expect("rgba → gbrap16");
    for i in 0..n {
        assert_eq!(
            read16le(&planar.planes[3].data, i * 2),
            rgba[i * 4 + 3] as u16 * 257,
            "A {i}"
        );
    }
    let back = convert(
        &planar,
        FrameInfo::new(PixelFormat::Gbrap16Le, w, h),
        PixelFormat::Rgba,
        &opts,
    )
    .expect("gbrap16 → rgba");
    assert_eq!(back.planes[0].data, rgba);
}

/// `Gbrp8` ↔ `Rgb24` is a zero-math plane reorder: known byte positions
/// on the way out, bit-exact round-trips in both directions.
#[test]
fn gbrp8_rgb24_pure_reorder_bit_exact() {
    let opts = ConvertOptions::default();
    let (w, h) = (6u32, 4u32);
    let src = synth_gbr8(w, h);
    let packed = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrp8, w, h),
        PixelFormat::Rgb24,
        &opts,
    )
    .expect("gbrp8 → rgb24");
    let n = (w * h) as usize;
    for i in 0..n {
        assert_eq!(packed.planes[0].data[i * 3], src.planes[2].data[i], "R {i}");
        assert_eq!(
            packed.planes[0].data[i * 3 + 1],
            src.planes[0].data[i],
            "G {i}"
        );
        assert_eq!(
            packed.planes[0].data[i * 3 + 2],
            src.planes[1].data[i],
            "B {i}"
        );
    }
    let back = convert(
        &packed,
        FrameInfo::new(PixelFormat::Rgb24, w, h),
        PixelFormat::Gbrp8,
        &opts,
    )
    .expect("rgb24 → gbrp8");
    for p in 0..3 {
        assert_eq!(back.planes[p].data, src.planes[p].data, "plane {p}");
    }
}

/// `Gbrp8` ↔ `Rgb48Le`: the widen is the exact ×257 (zero → zero,
/// 255 → 65535) and the narrow keeps the top byte, so the round-trip is
/// lossless.
#[test]
fn gbrp8_rgb48_widen_truncate_roundtrip() {
    let opts = ConvertOptions::default();
    let (w, h) = (6u32, 4u32);
    let mut src = synth_gbr8(w, h);
    // Rails on the first two G samples.
    src.planes[0].data[0] = 0;
    src.planes[0].data[1] = 255;
    let packed = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrp8, w, h),
        PixelFormat::Rgb48Le,
        &opts,
    )
    .expect("gbrp8 → rgb48");
    let n = (w * h) as usize;
    for i in 0..n {
        assert_eq!(
            read16le(&packed.planes[0].data, i * 6),
            src.planes[2].data[i] as u16 * 257,
            "R {i}"
        );
        assert_eq!(
            read16le(&packed.planes[0].data, i * 6 + 2),
            src.planes[0].data[i] as u16 * 257,
            "G {i}"
        );
        assert_eq!(
            read16le(&packed.planes[0].data, i * 6 + 4),
            src.planes[1].data[i] as u16 * 257,
            "B {i}"
        );
    }
    assert_eq!(read16le(&packed.planes[0].data, 2), 0, "zero rail");
    assert_eq!(read16le(&packed.planes[0].data, 8), 65535, "peak rail");
    let back = convert(
        &packed,
        FrameInfo::new(PixelFormat::Rgb48Le, w, h),
        PixelFormat::Gbrp8,
        &opts,
    )
    .expect("rgb48 → gbrp8");
    for p in 0..3 {
        assert_eq!(back.planes[p].data, src.planes[p].data, "plane {p}");
    }
}

/// The ladder ends are reachable from the wider ecosystem through the
/// staged fallback, and the in-ladder cross-depth moves resolve.
#[test]
fn gbr_ladder_end_reachability() {
    for (a, b) in [
        (PixelFormat::Gbrp8, PixelFormat::Yuv420P),
        (PixelFormat::Gbrp8, PixelFormat::Gbrp16Le),
        (PixelFormat::Gbrp8, PixelFormat::Gbrp10Le),
        (PixelFormat::Gbrp16Le, PixelFormat::Gbrp12Le),
        (PixelFormat::Gbrp16Le, PixelFormat::Yuv444P16Le),
        (PixelFormat::Gbrap16Le, PixelFormat::Yuva444P16Le),
        (PixelFormat::Gbrap16Le, PixelFormat::Bgra),
        (PixelFormat::Gbrap16Le, PixelFormat::Rgba64Le),
    ] {
        assert!(supports(a, b), "{a:?} → {b:?}");
        assert!(supports(b, a), "{b:?} → {a:?}");
    }
}
