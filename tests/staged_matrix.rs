//! Coverage-matrix and correctness tests for the single-pivot staged
//! conversion fallback in `convert()`.
//!
//! When no direct `(src, dst)` table entry exists, `convert()` now tries
//! one intermediate hop (YUV pivots first for YUV → YUV moves so no
//! colour matrix enters the path; RGB pivots first otherwise). These
//! tests build a structurally-valid frame for every `PixelFormat`
//! variant, sweep the full cross product, pin the coverage floor, and
//! spot-check that staged results equal the equivalent hand-staged
//! two-step conversion.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, supports, supports_direct, ConvertOptions, FormatInfo, FrameInfo, Palette,
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
];

fn plane(rows: usize, row_bytes: usize, seed: &mut u32) -> VideoPlane {
    let mut data = vec![0u8; rows * row_bytes];
    for b in data.iter_mut() {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (*seed >> 24) as u8;
    }
    VideoPlane {
        stride: row_bytes,
        data,
    }
}

/// Build a structurally-valid 8×8 frame for `fmt` with pseudo-random
/// content. Deep (16-bit-word) planes are masked to the format's
/// significant bits so samples are in-range.
fn build_frame(fmt: PixelFormat, w: usize, h: usize) -> VideoFrame {
    let info = FormatInfo::of(fmt);
    let mut seed = 0x1234_5678u32 ^ (fmt as u32);
    let planes: Vec<VideoPlane> = match fmt {
        PixelFormat::Rgb24 | PixelFormat::Bgr24 => vec![plane(h, w * 3, &mut seed)],
        PixelFormat::Rgba
        | PixelFormat::Bgra
        | PixelFormat::Argb
        | PixelFormat::Abgr
        | PixelFormat::Cmyk
        | PixelFormat::CmykInverted => vec![plane(h, w * 4, &mut seed)],
        // Packed 16-bit grey + alpha: one interleaved plane of LE16
        // word pairs, all 16 bits significant (no masking needed).
        PixelFormat::Ya16Le => vec![plane(h, w * 4, &mut seed)],
        PixelFormat::Rgb48Le => vec![plane(h, w * 6, &mut seed)],
        PixelFormat::Rgba64Le => vec![plane(h, w * 8, &mut seed)],
        PixelFormat::Gray8 | PixelFormat::Pal8 => vec![plane(h, w, &mut seed)],
        PixelFormat::Ya8 => vec![plane(h, w * 2, &mut seed)],
        PixelFormat::Gray16Le | PixelFormat::Gray10Le | PixelFormat::Gray12Le => {
            let mut p = plane(h, w * 2, &mut seed);
            mask_le16(&mut p.data, info.bit_depth as u32);
            vec![p]
        }
        PixelFormat::MonoBlack | PixelFormat::MonoWhite => {
            vec![plane(h, w.div_ceil(8), &mut seed)]
        }
        PixelFormat::Yuyv422 | PixelFormat::Uyvy422 => vec![plane(h, w * 2, &mut seed)],
        PixelFormat::Nv12 | PixelFormat::Nv21 => {
            vec![plane(h, w, &mut seed), plane(h / 2, (w / 2) * 2, &mut seed)]
        }
        PixelFormat::Gbrp10Le
        | PixelFormat::Gbrp12Le
        | PixelFormat::Gbrp14Le
        | PixelFormat::Gbrp16Le
        | PixelFormat::Gbrap10Le
        | PixelFormat::Gbrap12Le
        | PixelFormat::Gbrap14Le
        | PixelFormat::Gbrap16Le => {
            let n = if info.has_alpha { 4 } else { 3 };
            (0..n)
                .map(|_| {
                    let mut p = plane(h, w * 2, &mut seed);
                    mask_le16(&mut p.data, info.bit_depth as u32);
                    p
                })
                .collect()
        }
        _ => {
            // Planar YUV (8- or 16-bit storage), incl. Yuva420P.
            let sb = if info.bit_depth > 8 { 2 } else { 1 };
            let cw = w / info.chroma_w_sub as usize;
            let ch = h / info.chroma_h_sub as usize;
            let mut planes = vec![
                plane(h, w * sb, &mut seed),
                plane(ch, cw * sb, &mut seed),
                plane(ch, cw * sb, &mut seed),
            ];
            if sb == 2 {
                for p in planes.iter_mut() {
                    mask_le16(&mut p.data, info.bit_depth as u32);
                }
            }
            if info.has_alpha {
                planes.push(plane(h, w * sb, &mut seed));
            }
            planes
        }
    };
    VideoFrame { pts: None, planes }
}

fn mask_le16(data: &mut [u8], bits: u32) {
    let mask = ((1u32 << bits) - 1) as u16;
    for c in data.chunks_mut(2) {
        let v = ((c[0] as u16) | ((c[1] as u16) << 8)) & mask;
        c[0] = (v & 0xFF) as u8;
        c[1] = (v >> 8) as u8;
    }
}

fn test_opts() -> ConvertOptions {
    let colors: Vec<[u8; 4]> = (0..=255u16)
        .map(|i| [i as u8, i as u8, i as u8, 255])
        .collect();
    ConvertOptions {
        palette: Some(Palette { colors }),
        ..Default::default()
    }
}

/// Sweep the full format × format matrix on an 8×8 frame: every pair
/// that `supports()` claims must convert without error, every claimed-
/// unsupported pair must return `Err`, and the overall coverage floor
/// is pinned so a regression that silently drops routes fails loudly.
#[test]
fn coverage_matrix_matches_supports() {
    // The full 61 × 60 sweep converts every reachable pair; under the
    // miri interpreter that is many minutes of work, so shrink twice:
    // 4 × 4 frames (the smallest size every subsampling grid accepts —
    // 4:1:1 needs width divisible by 4) and a deterministic subset of
    // source formats. The UB-detection value is per-converter, which the
    // subset still reaches through the dst sweep; the exhaustive
    // pairing + coverage floors run natively.
    let dim: u32 = if cfg!(miri) { 4 } else { 8 };
    let src_stride = if cfg!(miri) { 5 } else { 1 };
    let opts = test_opts();
    let mut ok_pairs = 0usize;
    let mut direct_pairs = 0usize;
    let mut unsupported = Vec::new();
    for &src_fmt in ALL_FORMATS.iter().step_by(src_stride) {
        let frame = build_frame(src_fmt, dim as usize, dim as usize);
        let info = FrameInfo::new(src_fmt, dim, dim);
        for &dst_fmt in ALL_FORMATS {
            if src_fmt == dst_fmt {
                continue;
            }
            let claimed = supports(src_fmt, dst_fmt);
            let got = convert(&frame, info, dst_fmt, &opts);
            assert_eq!(
                claimed,
                got.is_ok(),
                "{src_fmt:?} → {dst_fmt:?}: supports()={claimed} but convert()={:?}",
                got.as_ref().err()
            );
            if claimed {
                ok_pairs += 1;
            } else {
                unsupported.push((src_fmt, dst_fmt));
            }
            if supports_direct(src_fmt, dst_fmt) {
                direct_pairs += 1;
            }
        }
    }
    // Coverage after the round-438 matrix closure (core 0.1.34 formats
    // + the Gray8-hub rows + the direct Rgb48Le ↔ Rgba64Le pair + the
    // full-precision deep-matrix tier): 976
    // direct and — for the first time — ALL 61 × 60 = 3660 ordered
    // pairs reachable. The direct floor may only go UP; the total is
    // pinned exact (any regression means a route was dropped). The
    // floors only hold for the full native sweep — the miri run covers
    // a source subset.
    if src_stride == 1 {
        eprintln!(
            "coverage: direct={direct_pairs} total={ok_pairs} of {}",
            ALL_FORMATS.len() * (ALL_FORMATS.len() - 1)
        );
        assert!(
            direct_pairs >= 976,
            "direct coverage regressed: {direct_pairs}"
        );
        assert_eq!(
            ok_pairs,
            ALL_FORMATS.len() * (ALL_FORMATS.len() - 1),
            "matrix no longer fully closed (unsupported sample: {:?})",
            &unsupported[..unsupported.len().min(8)]
        );
    }
}

/// A staged conversion must produce byte-identical output to manually
/// converting through the same pivot.
#[test]
fn staged_equals_manual_two_step() {
    let opts = test_opts();
    // Bgra → Yuv420P stages through Rgba (RGB pivots first, alpha kept).
    let bgra = build_frame(PixelFormat::Bgra, 8, 8);
    let staged = convert(
        &bgra,
        FrameInfo::new(PixelFormat::Bgra, 8, 8),
        PixelFormat::Yuv420P,
        &opts,
    )
    .expect("staged");
    let mid = convert(
        &bgra,
        FrameInfo::new(PixelFormat::Bgra, 8, 8),
        PixelFormat::Rgba,
        &opts,
    )
    .expect("leg 1");
    let manual = convert(
        &mid,
        FrameInfo::new(PixelFormat::Rgba, 8, 8),
        PixelFormat::Yuv420P,
        &opts,
    )
    .expect("leg 2");
    for p in 0..3 {
        assert_eq!(staged.planes[p].data, manual.planes[p].data, "plane {p}");
    }
}

/// YUV → YUV staged routes stay matrix-free: Yuyv422 → Yuv420P goes
/// through planar 4:2:2, so the luma plane survives byte-exact.
#[test]
fn packed422_to_420_keeps_luma_exact() {
    let opts = test_opts();
    let yuyv = build_frame(PixelFormat::Yuyv422, 8, 8);
    let out = convert(
        &yuyv,
        FrameInfo::new(PixelFormat::Yuyv422, 8, 8),
        PixelFormat::Yuv420P,
        &opts,
    )
    .expect("yuyv → 420");
    // Luma = every even byte of the packed source.
    let want_luma: Vec<u8> = yuyv.planes[0].data.chunks(2).map(|c| c[0]).collect();
    assert_eq!(out.planes[0].data, want_luma);
    assert_eq!(out.planes.len(), 3);
    assert_eq!(out.planes[1].data.len(), 4 * 4);
}

/// Alpha survives staged routes when both endpoints carry it:
/// Yuva420P → Bgra keeps the alpha plane bit-exact (pivot is Rgba).
#[test]
fn alpha_survives_staged_route() {
    let opts = test_opts();
    let yuva = build_frame(PixelFormat::Yuva420P, 8, 8);
    let out = convert(
        &yuva,
        FrameInfo::new(PixelFormat::Yuva420P, 8, 8),
        PixelFormat::Bgra,
        &opts,
    )
    .expect("yuva → bgra");
    let alpha_in = &yuva.planes[3].data;
    let alpha_out: Vec<u8> = out.planes[0].data.chunks(4).map(|c| c[3]).collect();
    assert_eq!(&alpha_out, alpha_in);
}

/// Deep GBR cross-depth moves stage through the deep packed pivot and
/// stay lossless when widening: Gbrp10Le → Gbrp12Le → Gbrp10Le is exact.
#[test]
fn gbr_cross_depth_via_deep_pivot_roundtrips() {
    let opts = test_opts();
    let src = build_frame(PixelFormat::Gbrp10Le, 8, 8);
    let up = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrp10Le, 8, 8),
        PixelFormat::Gbrp12Le,
        &opts,
    )
    .expect("10 → 12");
    let back = convert(
        &up,
        FrameInfo::new(PixelFormat::Gbrp12Le, 8, 8),
        PixelFormat::Gbrp10Le,
        &opts,
    )
    .expect("12 → 10");
    for p in 0..3 {
        assert_eq!(src.planes[p].data, back.planes[p].data, "plane {p}");
    }
}

/// 8-bit packed RGB round-trips exactly through every planar GBR depth
/// (widen = MSB replication, narrow = truncation), and the staged
/// fallback now reaches GBR from the YUV world via the Rgb24 pivot.
#[test]
fn gbr_8bit_roundtrips_and_yuv_reachability() {
    let opts = test_opts();
    let rgb = build_frame(PixelFormat::Rgb24, 8, 8);
    let info = FrameInfo::new(PixelFormat::Rgb24, 8, 8);
    for gbr in [
        PixelFormat::Gbrp10Le,
        PixelFormat::Gbrp12Le,
        PixelFormat::Gbrp14Le,
    ] {
        let planar = convert(&rgb, info, gbr, &opts).expect("rgb → gbr");
        assert_eq!(planar.planes.len(), 3);
        let back = convert(
            &planar,
            FrameInfo::new(gbr, 8, 8),
            PixelFormat::Rgb24,
            &opts,
        )
        .expect("gbr → rgb");
        assert_eq!(back.planes[0].data, rgb.planes[0].data, "{gbr:?}");
    }
    // Alpha variant keeps the alpha channel bit-exact.
    let rgba = build_frame(PixelFormat::Rgba, 8, 8);
    let planar = convert(
        &rgba,
        FrameInfo::new(PixelFormat::Rgba, 8, 8),
        PixelFormat::Gbrap12Le,
        &opts,
    )
    .expect("rgba → gbrap");
    let back = convert(
        &planar,
        FrameInfo::new(PixelFormat::Gbrap12Le, 8, 8),
        PixelFormat::Rgba,
        &opts,
    )
    .expect("gbrap → rgba");
    assert_eq!(back.planes[0].data, rgba.planes[0].data);
    // YUV ↔ GBR routes exist now (staged through Rgb24 / Rgba).
    assert!(supports(PixelFormat::Yuv420P, PixelFormat::Gbrp10Le));
    assert!(supports(PixelFormat::Gbrp14Le, PixelFormat::Yuv444P));
    assert!(supports(PixelFormat::Yuva420P, PixelFormat::Gbrap10Le));
}

/// Mono → colour formats now resolve through the Gray8 pivot.
#[test]
fn mono_reaches_colour_targets() {
    let opts = test_opts();
    let mono = build_frame(PixelFormat::MonoBlack, 8, 8);
    let info = FrameInfo::new(PixelFormat::MonoBlack, 8, 8);
    for dst in [
        PixelFormat::Rgba,
        PixelFormat::Bgr24,
        PixelFormat::Yuv444P,
        PixelFormat::MonoWhite,
    ] {
        let out = convert(&mono, info, dst, &opts);
        assert!(out.is_ok(), "MonoBlack → {dst:?}: {:?}", out.err());
    }
    // MonoBlack → MonoWhite must invert every pixel (both stage through
    // Gray8, whose threshold logic is symmetric).
    let white = convert(&mono, info, PixelFormat::MonoWhite, &opts).expect("mono swap");
    for (a, b) in mono.planes[0].data.iter().zip(white.planes[0].data.iter()) {
        assert_eq!(*a, !*b, "bit sense must flip");
    }
}

/// The new direct Rgb24/Rgba → Gray8 projection: r = g = b inputs map to
/// themselves exactly, and the round-trip through the Gray8 broadcast is
/// the identity on gray content.
#[test]
fn rgb_to_gray_identity_on_gray_content() {
    let w = 16usize;
    let h = 16usize;
    let opts = ConvertOptions::default();
    let mut rgb = vec![0u8; w * h * 3];
    for i in 0..w * h {
        let v = (i % 256) as u8;
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 3,
            data: rgb,
        }],
    };
    let gray = convert(
        &src,
        FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32),
        PixelFormat::Gray8,
        &opts,
    )
    .expect("rgb → gray");
    for (i, g) in gray.planes[0].data.iter().enumerate() {
        assert_eq!(*g as usize, i % 256);
    }
}

/// supports() reflexivity and a few negative pins (routes that would
/// need more than one pivot are correctly reported unsupported).
#[test]
fn supports_contract() {
    for &f in ALL_FORMATS {
        assert!(supports(f, f));
        assert!(supports_direct(f, f));
    }
    // Bgra → Yuv420P is staged, not direct.
    assert!(supports(PixelFormat::Bgra, PixelFormat::Yuv420P));
    assert!(!supports_direct(PixelFormat::Bgra, PixelFormat::Yuv420P));
}

/// Every round-438 closure row must be byte-identical to the manual
/// two-step route it replaced (where one existed) — the rows are
/// compositions of the same kernels, so any drift is a bug.
#[test]
fn closure_rows_match_manual_two_step() {
    let opts = test_opts();
    let dim = 8u32;
    // (src, dst, pivot) triples: direct result == src → pivot → dst.
    let cases = [
        (PixelFormat::Bgra, PixelFormat::Gray8, PixelFormat::Rgba),
        (PixelFormat::Bgr24, PixelFormat::Gray8, PixelFormat::Rgb24),
        (PixelFormat::Argb, PixelFormat::Gray8, PixelFormat::Rgba),
        (PixelFormat::Abgr, PixelFormat::Gray8, PixelFormat::Rgba),
        (PixelFormat::Rgb48Le, PixelFormat::Gray8, PixelFormat::Rgb24),
        (PixelFormat::Rgba64Le, PixelFormat::Gray8, PixelFormat::Rgba),
        (PixelFormat::Cmyk, PixelFormat::Gray8, PixelFormat::Rgb24),
        (
            PixelFormat::CmykInverted,
            PixelFormat::Gray8,
            PixelFormat::Rgb24,
        ),
        (PixelFormat::Pal8, PixelFormat::Gray8, PixelFormat::Rgb24),
        (
            PixelFormat::Yuyv422,
            PixelFormat::Gray8,
            PixelFormat::Yuv422P,
        ),
        (
            PixelFormat::Uyvy422,
            PixelFormat::Gray8,
            PixelFormat::Yuv422P,
        ),
        (PixelFormat::Gray8, PixelFormat::Cmyk, PixelFormat::Rgb24),
        (
            PixelFormat::Gray8,
            PixelFormat::CmykInverted,
            PixelFormat::Rgb24,
        ),
        (PixelFormat::Gray8, PixelFormat::Pal8, PixelFormat::Rgb24),
        (
            PixelFormat::Gray8,
            PixelFormat::Yuyv422,
            PixelFormat::Yuv422P,
        ),
        (
            PixelFormat::Gray8,
            PixelFormat::Uyvy422,
            PixelFormat::Yuv422P,
        ),
        (PixelFormat::Gray8, PixelFormat::Rgb48Le, PixelFormat::Rgb24),
        (PixelFormat::Gray8, PixelFormat::Rgba64Le, PixelFormat::Rgba),
    ];
    for (src_fmt, dst_fmt, pivot) in cases {
        assert!(
            supports_direct(src_fmt, dst_fmt),
            "{src_fmt:?} → {dst_fmt:?}"
        );
        let frame = build_frame(src_fmt, dim as usize, dim as usize);
        let info = FrameInfo::new(src_fmt, dim, dim);
        let direct = convert(&frame, info, dst_fmt, &opts).expect("direct");
        let mid = convert(&frame, info, pivot, &opts).expect("leg 1");
        let manual = convert(&mid, FrameInfo::new(pivot, dim, dim), dst_fmt, &opts).expect("leg 2");
        assert_eq!(
            direct.planes.len(),
            manual.planes.len(),
            "{src_fmt:?} → {dst_fmt:?}"
        );
        for p in 0..direct.planes.len() {
            assert_eq!(
                direct.planes[p].data, manual.planes[p].data,
                "{src_fmt:?} → {dst_fmt:?} plane {p}"
            );
        }
    }
}

/// The deep-packed pair converts directly and exactly: colour words
/// verbatim both ways, opaque 65535 alpha synthesised on the way up.
#[test]
fn rgb48_rgba64_direct_exact() {
    let opts = test_opts();
    let src = build_frame(PixelFormat::Rgb48Le, 8, 8);
    let info = FrameInfo::new(PixelFormat::Rgb48Le, 8, 8);
    assert!(supports_direct(PixelFormat::Rgb48Le, PixelFormat::Rgba64Le));
    assert!(supports_direct(PixelFormat::Rgba64Le, PixelFormat::Rgb48Le));
    let up = convert(&src, info, PixelFormat::Rgba64Le, &opts).expect("up");
    for i in 0..64usize {
        assert_eq!(
            &up.planes[0].data[i * 8..i * 8 + 6],
            &src.planes[0].data[i * 6..i * 6 + 6],
            "colour words {i}"
        );
        assert_eq!(&up.planes[0].data[i * 8 + 6..i * 8 + 8], &[0xFF, 0xFF]);
    }
    let back = convert(
        &up,
        FrameInfo::new(PixelFormat::Rgba64Le, 8, 8),
        PixelFormat::Rgb48Le,
        &opts,
    )
    .expect("down");
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

/// The previously-unreachable island pairs now convert end to end.
#[test]
fn closed_island_pairs_convert() {
    let opts = test_opts();
    for (src_fmt, dst_fmt) in [
        (PixelFormat::Cmyk, PixelFormat::MonoBlack),
        (PixelFormat::CmykInverted, PixelFormat::Gray16Le),
        (PixelFormat::MonoWhite, PixelFormat::Rgba64Le),
        (PixelFormat::MonoBlack, PixelFormat::Pal8),
        (PixelFormat::Gray12Le, PixelFormat::Yuyv422),
        (PixelFormat::Uyvy422, PixelFormat::Gray10Le),
        (PixelFormat::Rgb48Le, PixelFormat::MonoWhite),
        (PixelFormat::Bgra, PixelFormat::Gray12Le),
        (PixelFormat::Pal8, PixelFormat::Gray16Le),
        (PixelFormat::Gray16Le, PixelFormat::CmykInverted),
    ] {
        let frame = build_frame(src_fmt, 8, 8);
        let info = FrameInfo::new(src_fmt, 8, 8);
        let out = convert(&frame, info, dst_fmt, &opts);
        assert!(out.is_ok(), "{src_fmt:?} → {dst_fmt:?}: {:?}", out.err());
    }
}

/// Gray content survives the new deep-packed broadcast exactly: the
/// ×257 widen projects back to the original byte.
#[test]
fn gray_deep_packed_roundtrip_identity() {
    let opts = test_opts();
    let gray = build_frame(PixelFormat::Gray8, 8, 8);
    let info = FrameInfo::new(PixelFormat::Gray8, 8, 8);
    for deep in [PixelFormat::Rgb48Le, PixelFormat::Rgba64Le] {
        let up = convert(&gray, info, deep, &opts).expect("up");
        let back =
            convert(&up, FrameInfo::new(deep, 8, 8), PixelFormat::Gray8, &opts).expect("down");
        assert_eq!(back.planes[0].data, gray.planes[0].data, "{deep:?}");
    }
}
