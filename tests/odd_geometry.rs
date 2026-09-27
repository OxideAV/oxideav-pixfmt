//! Odd-geometry closure: with the rounded-up chroma grid (oxideav-core
//! `PixelFormat::plane_dimensions`, T.81 A.1.1), every ordered pair
//! `supports()` reports converts at odd sizes — not merely "does not
//! panic" — and every output plane has at least the geometry core
//! describes for the destination. The only rejections allowed are on
//! the two layouts whose grid pins a width: `Yuv411P` (multiple of 4)
//! and the packed `Yuyv422` / `Uyvy422` pairs (even width) — and even
//! those still succeed on hops that touch only luma.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, convert_with, supports, ConvertContext, ConvertOptions, FormatInfo, FrameInfo, Palette,
};

/// Every PixelFormat variant (kept in discriminant order).
const ALL_FORMATS: &[PixelFormat] = &[
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Gray8,
    PixelFormat::Pal8,
    PixelFormat::Bgr24,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray16Le,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Yuv420P10Le,
    PixelFormat::Yuv422P10Le,
    PixelFormat::Yuv444P10Le,
    PixelFormat::Yuv420P12Le,
    PixelFormat::Yuv422P12Le,
    PixelFormat::Yuv444P12Le,
    PixelFormat::YuvJ420P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ444P,
    PixelFormat::Nv12,
    PixelFormat::Nv21,
    PixelFormat::Ya8,
    PixelFormat::Yuva420P,
    PixelFormat::MonoBlack,
    PixelFormat::MonoWhite,
    PixelFormat::Yuyv422,
    PixelFormat::Uyvy422,
    PixelFormat::Cmyk,
    PixelFormat::Yuv411P,
    PixelFormat::Gbrp10Le,
    PixelFormat::Gbrap10Le,
    PixelFormat::Gbrp12Le,
    PixelFormat::Gbrap12Le,
    PixelFormat::Gbrp14Le,
    PixelFormat::Gbrap14Le,
    PixelFormat::Yuv420P16Le,
    PixelFormat::Yuv422P16Le,
    PixelFormat::Yuv444P16Le,
    PixelFormat::Yuva422P,
    PixelFormat::Yuva444P,
    PixelFormat::Yuva422P10Le,
    PixelFormat::Yuva422P12Le,
    PixelFormat::Yuva444P10Le,
    PixelFormat::Yuva444P12Le,
    PixelFormat::Yuva422P16Le,
    PixelFormat::Yuva444P16Le,
    PixelFormat::Gbrp8,
    PixelFormat::Gbrp16Le,
    PixelFormat::Gbrap16Le,
    PixelFormat::Yuva420P10Le,
    PixelFormat::Yuva420P12Le,
    PixelFormat::Yuva420P16Le,
    PixelFormat::Gbrap8,
    PixelFormat::Ya16Le,
    PixelFormat::CmykInverted,
    PixelFormat::Yuv440P,
    PixelFormat::Yuv440P10Le,
    PixelFormat::Yuv440P12Le,
    PixelFormat::Yuv440P16Le,
    PixelFormat::GrayF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::RgbaF32Le,
    PixelFormat::GbrpF32Le,
    PixelFormat::GbrapF32Le,
];

/// Build a frame for `fmt` at `w × h` on the geometry core reports for
/// each plane, with pseudo-random content (deep words masked to the
/// format's significant bits).
fn build(fmt: PixelFormat, w: u32, h: u32, seed: &mut u32) -> VideoFrame {
    let info = FormatInfo::of(fmt);
    let deep_mask = if info.bit_depth > 8 && info.bit_depth < 16 && !fmt.is_float() {
        Some(((1u32 << info.bit_depth) - 1) as u16)
    } else {
        None
    };
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let row = fmt.plane_row_bytes(p, w).expect("row bytes");
            let (_, ph) = fmt.plane_dimensions(p, w, h).expect("plane dims");
            let mut data = vec![0u8; row * ph as usize];
            for b in data.iter_mut() {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (*seed >> 24) as u8;
            }
            if let Some(mask) = deep_mask {
                for c in data.chunks_exact_mut(2) {
                    let v = u16::from_le_bytes([c[0], c[1]]) & mask;
                    c.copy_from_slice(&v.to_le_bytes());
                }
            }
            VideoPlane { stride: row, data }
        })
        .collect();
    VideoFrame { pts: None, planes }
}

/// Pairs whose geometry rule pins the width.
fn width_pinned(f: PixelFormat, w: u32) -> bool {
    match f {
        PixelFormat::Yuv411P => w % 4 != 0,
        PixelFormat::Yuyv422 | PixelFormat::Uyvy422 => w % 2 != 0,
        _ => false,
    }
}

fn opts() -> ConvertOptions {
    let colors: Vec<[u8; 4]> = (0..=255u16)
        .map(|i| [i as u8, i as u8, i as u8, 255])
        .collect();
    ConvertOptions {
        palette: Some(Palette { colors }),
        ..Default::default()
    }
}

#[test]
fn every_supported_pair_converts_at_odd_sizes() {
    let sizes: &[(u32, u32)] = if cfg!(miri) {
        &[(3, 5)]
    } else {
        &[(1, 1), (3, 5), (7, 3), (5, 2), (2, 7)]
    };
    let src_stride = if cfg!(miri) { 7 } else { 1 };
    let o = opts();
    let mut seed = 0x0463_0DD5u32;
    let mut converted = 0usize;
    for &(w, h) in sizes {
        for &src_fmt in ALL_FORMATS.iter().step_by(src_stride) {
            let frame = build(src_fmt, w, h, &mut seed);
            let info = FrameInfo::new(src_fmt, w, h);
            for &dst_fmt in ALL_FORMATS {
                if src_fmt == dst_fmt || !supports(src_fmt, dst_fmt) {
                    continue;
                }
                let got = convert(&frame, info, dst_fmt, &o);
                let out = match got {
                    Ok(out) => out,
                    // Only a width-pinned layout may refuse (luma-only
                    // hops such as 4:1:1 → Gray8 still succeed).
                    Err(_) if width_pinned(src_fmt, w) || width_pinned(dst_fmt, w) => continue,
                    Err(e) => panic!("{src_fmt:?} → {dst_fmt:?} at {w}x{h}: {e:?}"),
                };
                assert!(out.planes.len() >= dst_fmt.plane_count());
                for p in 0..dst_fmt.plane_count() {
                    let row = dst_fmt.plane_row_bytes(p, w).unwrap();
                    let (_, ph) = dst_fmt.plane_dimensions(p, w, h).unwrap();
                    let pl = &out.planes[p];
                    assert!(
                        pl.stride >= row
                            && pl.data.len() >= pl.stride * (ph as usize - 1) + row,
                        "{src_fmt:?} → {dst_fmt:?} at {w}x{h}: plane {p} is {} bytes / stride {} (need {row} x {ph})",
                        pl.data.len(),
                        pl.stride
                    );
                }
                converted += 1;
            }
        }
    }
    assert!(converted > 0);
}

/// The same odd-size sweep through the threaded entry point agrees
/// with the serial one byte for byte on the YUV ↔ RGB rows.
#[test]
fn threaded_odd_sizes_match_serial() {
    let o = opts();
    let mut seed = 0x0463_7777u32;
    let (w, h) = if cfg!(miri) { (5, 37) } else { (7, 97) };
    let ctx = ConvertContext::new().with_execution(ExecutionContext::with_threads(3));
    for src_fmt in [
        PixelFormat::Yuv420P,
        PixelFormat::Yuva420P,
        PixelFormat::Yuv422P10Le,
        PixelFormat::Yuva420P12Le,
        PixelFormat::Nv12,
        PixelFormat::YuvJ420P,
        PixelFormat::Yuv440P,
    ] {
        let frame = build(src_fmt, w, h, &mut seed);
        let info = FrameInfo::new(src_fmt, w, h);
        for dst in [
            PixelFormat::Rgb24,
            PixelFormat::Rgba,
            PixelFormat::Rgb48Le,
            PixelFormat::Rgba64Le,
        ] {
            let a = convert(&frame, info, dst, &o).unwrap();
            let b = convert_with(&frame, info, dst, &o, &ctx).unwrap();
            assert!(
                a.planes[0].data == b.planes[0].data,
                "{src_fmt:?} → {dst:?}"
            );
        }
    }
}
