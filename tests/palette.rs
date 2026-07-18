//! Palette generation + Pal8 quantise → dequantise tests.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, generate_palette, ConvertOptions, Dither, FrameInfo, PaletteGenOptions,
    PaletteStrategy,
};

fn deterministic_rgba(w: u32, h: u32, seed: u32) -> (VideoFrame, FrameInfo) {
    // Cheap xorshift for repeatability without a random crate.
    let mut state = seed | 1;
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data.push((state & 0xff) as u8);
        data.push(((state >> 8) & 0xff) as u8);
        data.push(((state >> 16) & 0xff) as u8);
        data.push(((state >> 24) & 0xff) as u8);
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

fn gradient_rgb24(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / (w - 1).max(1)) as u8;
            let g = ((y * 255) / (h - 1).max(1)) as u8;
            let b = (((x + y) * 255) / ((w + h) - 2).max(1)) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
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

fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sq = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sq += d * d;
    }
    if sq == 0.0 {
        return f64::INFINITY;
    }
    let mse = sq / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

// Median-cut / octree palette generation over random frames is pure
// pointer-free arithmetic but costs O(pixels · iterations); under the
// miri interpreter the full-size frames below run for hours (the CI
// miri job has hung in this file at its 6 h timeout since it was
// introduced). Shrinking the corpus under cfg(miri) follows the same
// convention as tests/yuv_simd_parity.rs — a build-time toggle, NOT
// `#[ignore]`; every test still runs and asserts under miri.

#[test]
fn generate_palette_stays_under_256() {
    // 24×24 random pixels still produce > 256 candidate colours, so the
    // ≤ 256 cap assertion keeps its teeth under miri.
    #[cfg(miri)]
    let dim = 24;
    #[cfg(not(miri))]
    let dim = 256;
    let (frame, info) = deterministic_rgba(dim, dim, 0xDEADBEEF);
    let opts = PaletteGenOptions::default();
    let palette = generate_palette(&[(&frame, info)], &opts).unwrap();
    assert!(
        palette.colors.len() <= 256,
        "got {} colours",
        palette.colors.len()
    );
    assert!(!palette.colors.is_empty());
}

#[test]
fn octree_palette_respects_max_and_roundtrips() {
    #[cfg(miri)]
    let dim = 16;
    #[cfg(not(miri))]
    let dim = 64;
    let (src, src_info) = gradient_rgb24(dim, dim);
    let palette = generate_palette(
        &[(&src, src_info)],
        &PaletteGenOptions {
            strategy: PaletteStrategy::Octree,
            max_colors: 64,
            transparency: None,
        },
    )
    .unwrap();
    assert!(
        !palette.colors.is_empty(),
        "octree must emit at least one colour"
    );
    assert!(
        palette.colors.len() <= 64,
        "octree over-produced: {} > 64",
        palette.colors.len()
    );

    let opts = ConvertOptions {
        dither: Dither::FloydSteinberg,
        palette: Some(palette.clone()),
        color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
    };
    let pal8 = convert(&src, src_info, PixelFormat::Pal8, &opts).unwrap();
    let pal8_info = FrameInfo::new(PixelFormat::Pal8, src_info.width, src_info.height);
    let back = convert(
        &pal8,
        pal8_info,
        PixelFormat::Rgb24,
        &ConvertOptions {
            dither: Dither::None,
            palette: Some(palette),
            color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
        },
    )
    .unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    assert!(psnr > 24.0, "octree roundtrip psnr {psnr} below 24 dB");
}

#[test]
fn octree_palette_small_max_caps_output() {
    #[cfg(miri)]
    let dim = 16;
    #[cfg(not(miri))]
    let dim = 128;
    let (frame, info) = deterministic_rgba(dim, dim, 0xCAFEF00D);
    let palette = generate_palette(
        &[(&frame, info)],
        &PaletteGenOptions {
            strategy: PaletteStrategy::Octree,
            max_colors: 16,
            transparency: None,
        },
    )
    .unwrap();
    assert!(palette.colors.len() <= 16);
    assert!(!palette.colors.is_empty());
}

#[test]
fn uniform_palette_has_256_entries() {
    // The uniform strategy's entry count is input-independent; the frame
    // only feeds the API signature, so it can be tiny under miri.
    #[cfg(miri)]
    let dim = 8;
    #[cfg(not(miri))]
    let dim = 64;
    let (frame, info) = deterministic_rgba(dim, dim, 0xB16B00B5);
    let opts = PaletteGenOptions {
        strategy: PaletteStrategy::Uniform,
        max_colors: 255, // u8 max
        transparency: None,
    };
    let palette = generate_palette(&[(&frame, info)], &opts).unwrap();
    assert_eq!(palette.colors.len(), 255);
}

#[test]
fn pal8_roundtrip_exceeds_24_db() {
    #[cfg(miri)]
    let dim = 16;
    #[cfg(not(miri))]
    let dim = 64;
    let (src, src_info) = gradient_rgb24(dim, dim);
    let palette = generate_palette(
        &[(&src, src_info)],
        &PaletteGenOptions {
            strategy: PaletteStrategy::MedianCut,
            max_colors: 64,
            transparency: None,
        },
    )
    .unwrap();

    let opts = ConvertOptions {
        dither: Dither::FloydSteinberg,
        palette: Some(palette.clone()),
        color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
    };

    let pal8 = convert(&src, src_info, PixelFormat::Pal8, &opts).unwrap();
    let pal8_info = FrameInfo::new(PixelFormat::Pal8, src_info.width, src_info.height);
    let back = convert(
        &pal8,
        pal8_info,
        PixelFormat::Rgb24,
        &ConvertOptions {
            dither: Dither::None,
            palette: Some(palette),
            color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
        },
    )
    .unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("pal8 Floyd-Steinberg psnr = {psnr:.2}");
    assert!(psnr > 24.0, "pal8 psnr {psnr} below 24 dB");
}

#[test]
fn pal8_decode_missing_palette_errors() {
    let (src, src_info) = gradient_rgb24(8, 4);
    let palette = generate_palette(
        &[(&src, src_info)],
        &PaletteGenOptions {
            strategy: PaletteStrategy::MedianCut,
            max_colors: 16,
            transparency: None,
        },
    )
    .unwrap();
    let opts = ConvertOptions {
        dither: Dither::None,
        palette: Some(palette),
        color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
    };
    let mut pal8 = convert(&src, src_info, PixelFormat::Pal8, &opts).unwrap();
    let pal8_info = FrameInfo::new(PixelFormat::Pal8, src_info.width, src_info.height);
    // RGB → Pal8 attaches the colour table as the frame's palette
    // side-channel, so decoding the frame it produced succeeds even
    // without ConvertOptions.palette — the frame is self-describing.
    let bare = ConvertOptions::default();
    assert!(
        convert(&pal8, pal8_info, PixelFormat::Rgb24, &bare).is_ok(),
        "side-channel palette must satisfy Pal8 → RGB"
    );
    // Strip the side-channel: with no attached palette AND no options
    // palette, the conversion has no colour table at all — must fail.
    assert!(
        pal8.take_palette().is_some(),
        "encode must attach a palette"
    );
    let res = convert(&pal8, pal8_info, PixelFormat::Rgb24, &bare);
    assert!(res.is_err(), "palette omission must error");
}

#[test]
fn pal8_side_channel_takes_precedence_and_roundtrips() {
    use oxideav_core::{VideoFrame, VideoPlane};

    // A 16 × 1 index ramp with an identity-gray attached palette:
    // entry i = (i * 16, i * 16, i * 16).
    let w = 16u32;
    let indices: Vec<u8> = (0..16u8).collect();
    let side: Vec<u8> = (0..16u8).flat_map(|i| [i * 16, i * 16, i * 16]).collect();
    let frame = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize,
            data: indices.clone(),
        }],
    }
    .with_palette(side);
    let info = FrameInfo::new(PixelFormat::Pal8, w, 1);

    // Options carry a DIFFERENT (all-red) palette: the frame-attached
    // table must win — the frame's own colours are ground truth.
    let red = oxideav_pixfmt::Palette {
        colors: (0..16).map(|_| [255, 0, 0, 255]).collect(),
    };
    let opts = ConvertOptions {
        palette: Some(red),
        ..Default::default()
    };
    let rgb = convert(&frame, info, PixelFormat::Rgb24, &opts).unwrap();
    for (i, px) in rgb.planes[0].data.chunks_exact(3).enumerate() {
        let want = (i as u8) * 16;
        assert_eq!(px, [want, want, want], "index {i}");
    }
    // Rgba expansion synthesises opaque alpha from the 3-byte entries.
    let rgba = convert(&frame, info, PixelFormat::Rgba, &ConvertOptions::default()).unwrap();
    for (i, px) in rgba.planes[0].data.chunks_exact(4).enumerate() {
        let want = (i as u8) * 16;
        assert_eq!(px, [want, want, want, 255], "index {i}");
    }
}

#[test]
fn rgb_to_pal8_attaches_matching_side_channel() {
    let (src, src_info) = gradient_rgb24(16, 8);
    let palette = generate_palette(
        &[(&src, src_info)],
        &PaletteGenOptions {
            strategy: PaletteStrategy::MedianCut,
            max_colors: 32,
            transparency: None,
        },
    )
    .unwrap();
    let opts = ConvertOptions {
        dither: Dither::None,
        palette: Some(palette.clone()),
        color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
    };
    let pal8 = convert(&src, src_info, PixelFormat::Pal8, &opts).unwrap();
    // The attached side-channel mirrors the quantisation palette's RGB
    // columns entry-for-entry (alpha is not representable there).
    let side = pal8.palette().expect("side-channel attached");
    assert_eq!(side.len(), palette.colors.len() * 3);
    for (entry, c) in side.chunks_exact(3).zip(palette.colors.iter()) {
        assert_eq!(entry, &c[..3]);
    }
    // The index plane itself is unchanged by the attachment.
    assert_eq!(pal8.image_planes().len(), 1);
    assert_eq!(
        pal8.image_planes()[0].data.len(),
        (src_info.width * src_info.height) as usize
    );
    // Self-describing round-trip: decode with a default options bundle
    // must equal decoding with the explicit palette.
    let pal8_info = FrameInfo::new(PixelFormat::Pal8, src_info.width, src_info.height);
    let via_side = convert(
        &pal8,
        pal8_info,
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
    )
    .unwrap();
    let via_opts = convert(&pal8, pal8_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(via_side.planes[0].data, via_opts.planes[0].data);
}
