//! The 4:4:0 planar family (`Yuv440P` / `Yuv440P10Le` / `Yuv440P12Le`
//! / `Yuv440P16Le`, oxideav-core 0.1.35): full-width, half-height
//! chroma. Every conversion involving the family is pinned against the
//! equivalent route through 4:4:4, where the only new arithmetic is a
//! vertical pair-average (down) or row broadcast (up).

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, supports, supports_direct, ChromaSubsampling, ConvertOptions, FormatInfo, FrameInfo,
};

const FAMILY_440: [PixelFormat; 4] = [
    PixelFormat::Yuv440P,
    PixelFormat::Yuv440P10Le,
    PixelFormat::Yuv440P12Le,
    PixelFormat::Yuv440P16Le,
];

fn opts() -> ConvertOptions {
    ConvertOptions::default()
}

/// Deterministic sample generator: `plane` selects a different pattern
/// per plane, `bits` bounds the value.
fn sample(plane: usize, x: usize, y: usize, bits: u32) -> u32 {
    let full = (1u32 << bits) - 1;
    let v = (x * 37 + y * 91 + plane * 53 + (x * y) % 17) as u32 * 977;
    // Keep away from the extremes so ±1 rounding never clips.
    16 + v % (full - 32)
}

/// Build a tightly-packed planar YUV frame of `fmt` with the generator
/// above; `pad` extra stride bytes per row exercise the gather path.
fn build_yuv(fmt: PixelFormat, w: usize, h: usize, pad: usize) -> VideoFrame {
    let info = FormatInfo::of(fmt);
    let bits = info.bit_depth as u32;
    let sb = if bits > 8 { 2 } else { 1 };
    let cw = w / info.chroma_w_sub as usize;
    let ch = h / info.chroma_h_sub as usize;
    let mk = |plane: usize, pw: usize, ph: usize| {
        let stride = pw * sb + pad;
        let mut data = vec![0u8; stride * ph];
        for y in 0..ph {
            for x in 0..pw {
                let v = sample(plane, x, y, bits);
                if sb == 2 {
                    data[y * stride + x * 2] = (v & 0xFF) as u8;
                    data[y * stride + x * 2 + 1] = (v >> 8) as u8;
                } else {
                    data[y * stride + x] = v as u8;
                }
            }
        }
        VideoPlane { stride, data }
    };
    VideoFrame {
        pts: None,
        planes: vec![mk(0, w, h), mk(1, cw, ch), mk(2, cw, ch)],
    }
}

/// Strip stride padding so two frames can be compared plane-by-plane.
fn tight(frame: &VideoFrame, fmt: PixelFormat, w: usize, h: usize) -> Vec<Vec<u8>> {
    let info = FormatInfo::of(fmt);
    let sb = if info.bit_depth > 8 { 2 } else { 1 };
    let cw = w / info.chroma_w_sub as usize;
    let ch = h / info.chroma_h_sub as usize;
    frame
        .planes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (pw, ph) = if i == 0 || i == 3 { (w, h) } else { (cw, ch) };
            let mut out = Vec::with_capacity(pw * sb * ph);
            for y in 0..ph {
                out.extend_from_slice(&p.data[y * p.stride..y * p.stride + pw * sb]);
            }
            out
        })
        .collect()
}

fn conv(frame: &VideoFrame, src: PixelFormat, dst: PixelFormat, w: usize, h: usize) -> VideoFrame {
    convert(frame, FrameInfo::new(src, w as u32, h as u32), dst, &opts())
        .unwrap_or_else(|e| panic!("{src:?} → {dst:?}: {e:?}"))
}

#[test]
fn format_info_geometry() {
    for (fmt, bits) in FAMILY_440.iter().zip([8u8, 10, 12, 16]) {
        let info = FormatInfo::of(*fmt);
        assert_eq!(info.bit_depth, bits, "{fmt:?}");
        assert_eq!(info.planes, 3);
        assert_eq!((info.chroma_w_sub, info.chroma_h_sub), (1, 2));
        assert!(info.is_planar && !info.has_alpha && !info.is_palette);
        assert_eq!(info.chroma_subsampling(), ChromaSubsampling::C440);
        assert!(info.is_chroma_subsampled());
        // Agrees with the core geometry helpers.
        assert_eq!(fmt.chroma_subsampling(), Some((0, 1)));
        assert_eq!(fmt.plane_dimensions(1, 6, 4), Some((6, 2)));
    }
}

#[test]
fn family_pairs_are_direct() {
    let others = [
        PixelFormat::Yuv420P,
        PixelFormat::Yuv422P,
        PixelFormat::Yuv444P,
        PixelFormat::Yuv420P10Le,
        PixelFormat::Yuv444P12Le,
        PixelFormat::Yuv422P16Le,
        PixelFormat::Yuva420P,
        PixelFormat::Yuva444P16Le,
        PixelFormat::Rgb24,
        PixelFormat::Rgba,
        PixelFormat::Gray8,
    ];
    for &f in &FAMILY_440 {
        for &o in &others {
            assert!(supports_direct(f, o), "{f:?} → {o:?} should be direct");
            assert!(supports_direct(o, f), "{o:?} → {f:?} should be direct");
        }
        for &g in &FAMILY_440 {
            assert!(supports_direct(f, g), "{f:?} → {g:?}");
        }
    }
    assert!(supports_direct(
        PixelFormat::Yuv440P16Le,
        PixelFormat::Rgb48Le
    ));
    assert!(supports_direct(
        PixelFormat::Rgb48Le,
        PixelFormat::Yuv440P16Le
    ));
    assert!(supports_direct(PixelFormat::Yuv411P, PixelFormat::Yuv440P));
    assert!(supports_direct(PixelFormat::Yuv440P, PixelFormat::Yuv411P));
    // Staged but closed: full-range, semi-planar, packed 4:2:2, deep RGB.
    for o in [
        PixelFormat::YuvJ420P,
        PixelFormat::Nv12,
        PixelFormat::Yuyv422,
        PixelFormat::Rgba64Le,
        PixelFormat::Gbrp10Le,
        PixelFormat::Pal8,
        PixelFormat::MonoBlack,
    ] {
        for &f in &FAMILY_440 {
            assert!(supports(f, o), "{f:?} → {o:?}");
            assert!(supports(o, f), "{o:?} → {f:?}");
        }
    }
}

/// 4:4:4 → 4:4:0 is a vertical pair-average, 4:4:0 → 4:4:4 a row
/// broadcast — so chroma that is constant over each row pair survives
/// the round trip exactly, at every depth, and luma is untouched.
#[test]
fn round_trip_444_440_444_is_exact_on_row_paired_chroma() {
    let (w, h) = (7, 6);
    for (f440, f444) in [
        (PixelFormat::Yuv440P, PixelFormat::Yuv444P),
        (PixelFormat::Yuv440P10Le, PixelFormat::Yuv444P10Le),
        (PixelFormat::Yuv440P12Le, PixelFormat::Yuv444P12Le),
        (PixelFormat::Yuv440P16Le, PixelFormat::Yuv444P16Le),
    ] {
        let src = build_yuv(f440, w, h, 3);
        let up = conv(&src, f440, f444, w, h);
        let back = conv(&up, f444, f440, w, h);
        assert_eq!(
            tight(&back, f440, w, h),
            tight(&src, f440, w, h),
            "{f440:?}"
        );
        // The broadcast really is a broadcast: rows 2k and 2k+1 match.
        let up_t = tight(&up, f444, w, h);
        let sb = if FormatInfo::of(f444).bit_depth > 8 {
            2
        } else {
            1
        };
        for (plane, data) in up_t.iter().enumerate().skip(1) {
            for pair in 0..h / 2 {
                let a = &data[pair * 2 * w * sb..(pair * 2 + 1) * w * sb];
                let b = &data[(pair * 2 + 1) * w * sb..(pair * 2 + 2) * w * sb];
                assert_eq!(a, b, "{f444:?} plane {plane} row pair {pair}");
            }
        }
    }
}

/// Every siting move touching 4:4:0 equals the two-step route through
/// 4:4:4 (the composition is what the resampler executes, and this pins
/// that no other arithmetic sneaks in).
#[test]
fn siting_moves_match_444_composition() {
    let (w, h) = (8, 6);
    let sitings = [
        (PixelFormat::Yuv420P, PixelFormat::Yuv444P),
        (PixelFormat::Yuv422P, PixelFormat::Yuv444P),
        (PixelFormat::Yuv411P, PixelFormat::Yuv444P),
        (PixelFormat::Yuv420P16Le, PixelFormat::Yuv444P16Le),
        (PixelFormat::Yuv422P16Le, PixelFormat::Yuv444P16Le),
    ];
    for (other, pivot) in sitings {
        let f440 = if FormatInfo::of(other).bit_depth > 8 {
            PixelFormat::Yuv440P16Le
        } else {
            PixelFormat::Yuv440P
        };
        // 4:4:0 → other
        let src = build_yuv(f440, w, h, 0);
        let direct = conv(&src, f440, other, w, h);
        let via = conv(&conv(&src, f440, pivot, w, h), pivot, other, w, h);
        assert_eq!(
            tight(&direct, other, w, h),
            tight(&via, other, w, h),
            "{f440:?} → {other:?}"
        );
        // other → 4:4:0
        let src = build_yuv(other, w, h, 2);
        let direct = conv(&src, other, f440, w, h);
        let via = conv(&conv(&src, other, pivot, w, h), pivot, f440, w, h);
        assert_eq!(
            tight(&direct, f440, w, h),
            tight(&via, f440, w, h),
            "{other:?} → {f440:?}"
        );
    }
}

/// Odd widths are legal on 4:4:0 (chroma is full width); odd heights
/// have no tight representation on a half-height chroma plane and are
/// rejected cleanly — the same rule the crate applies to odd widths on
/// 4:2:2 — never a panic or an out-of-bounds read.
#[test]
fn geometry_rules() {
    let opts = opts();
    // Odd width, even height: fine in both directions.
    let src = build_yuv(PixelFormat::Yuv440P, 5, 4, 1);
    let rgb = conv(&src, PixelFormat::Yuv440P, PixelFormat::Rgb24, 5, 4);
    assert_eq!(rgb.planes[0].data.len(), 5 * 4 * 3);
    let back = conv(&rgb, PixelFormat::Rgb24, PixelFormat::Yuv440P, 5, 4);
    assert_eq!(back.planes[1].data.len(), 5 * 2);
    let p444 = conv(&src, PixelFormat::Yuv440P, PixelFormat::Yuv444P, 5, 4);
    assert_eq!(p444.planes[1].data.len(), 5 * 4);
    // Odd height: rejected, not panicked.
    let src = build_yuv(PixelFormat::Yuv440P, 6, 5, 0);
    let info = FrameInfo::new(PixelFormat::Yuv440P, 6, 5);
    for dst in [
        PixelFormat::Yuv444P,
        PixelFormat::Yuv420P,
        PixelFormat::Rgb24,
        PixelFormat::Yuv440P10Le,
    ] {
        assert!(convert(&src, info, dst, &opts).is_err(), "→ {dst:?}");
    }
    let rgb = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 18,
            data: vec![0x80; 18 * 5],
        }],
    };
    let info = FrameInfo::new(PixelFormat::Rgb24, 6, 5);
    assert!(convert(&rgb, info, PixelFormat::Yuv440P, &opts).is_err());
    // Luma-only extraction ignores the chroma grid, so it still works.
    let src = build_yuv(PixelFormat::Yuv440P, 6, 5, 0);
    let info = FrameInfo::new(PixelFormat::Yuv440P, 6, 5);
    assert!(convert(&src, info, PixelFormat::Gray8, &opts).is_ok());
}

/// Padded strides on every plane produce the same output as the tight
/// frame.
#[test]
fn stride_padding_is_transparent() {
    let (w, h) = (6, 4);
    for &f in &FAMILY_440 {
        let tight_src = build_yuv(f, w, h, 0);
        let padded = build_yuv(f, w, h, 5);
        for dst in [
            PixelFormat::Yuv444P,
            PixelFormat::Rgb24,
            PixelFormat::Yuv420P10Le,
        ] {
            let a = conv(&tight_src, f, dst, w, h);
            let b = conv(&padded, f, dst, w, h);
            assert_eq!(
                tight(&a, dst, w, h),
                tight(&b, dst, w, h),
                "{f:?} → {dst:?}"
            );
        }
    }
}

/// The depth ladder inside the family is the crate-wide exact
/// widen / truncate: 8 → 10 → 12 → 16 → 8 round-trips bit-exact.
#[test]
fn depth_ladder_round_trips() {
    let (w, h) = (5, 4);
    let src = build_yuv(PixelFormat::Yuv440P, w, h, 0);
    let a = conv(&src, PixelFormat::Yuv440P, PixelFormat::Yuv440P10Le, w, h);
    let b = conv(&a, PixelFormat::Yuv440P10Le, PixelFormat::Yuv440P12Le, w, h);
    let c = conv(&b, PixelFormat::Yuv440P12Le, PixelFormat::Yuv440P16Le, w, h);
    let back = conv(&c, PixelFormat::Yuv440P16Le, PixelFormat::Yuv440P, w, h);
    assert_eq!(
        tight(&back, PixelFormat::Yuv440P, w, h),
        tight(&src, PixelFormat::Yuv440P, w, h)
    );
    // 8 → 16 is the exact ×257 mapping.
    let wide = conv(&src, PixelFormat::Yuv440P, PixelFormat::Yuv440P16Le, w, h);
    let s = tight(&src, PixelFormat::Yuv440P, w, h);
    let t = tight(&wide, PixelFormat::Yuv440P16Le, w, h);
    for p in 0..3 {
        for (i, &v) in s[p].iter().enumerate() {
            let got = u16::from_le_bytes([t[p][i * 2], t[p][i * 2 + 1]]);
            assert_eq!(got, v as u16 * 257);
        }
    }
}

/// RGB interop at 8 bits: decode equals the 4:4:4 decode of the
/// row-broadcast frame; encode equals the 4:4:4 encode followed by the
/// vertical pair-average. Alpha is synthesised opaque / dropped.
#[test]
fn rgb_interop_matches_444_route() {
    let (w, h) = (6, 4);
    for &f in &FAMILY_440 {
        let f444 = match f {
            PixelFormat::Yuv440P => PixelFormat::Yuv444P,
            PixelFormat::Yuv440P10Le => PixelFormat::Yuv444P10Le,
            PixelFormat::Yuv440P12Le => PixelFormat::Yuv444P12Le,
            _ => PixelFormat::Yuv444P16Le,
        };
        let src = build_yuv(f, w, h, 0);
        for rgb in [PixelFormat::Rgb24, PixelFormat::Rgba] {
            let direct = conv(&src, f, rgb, w, h);
            let via = conv(&conv(&src, f, f444, w, h), f444, rgb, w, h);
            assert_eq!(direct.planes[0].data, via.planes[0].data, "{f:?} → {rgb:?}");
            if rgb == PixelFormat::Rgba {
                assert!(direct.planes[0].data.chunks(4).all(|p| p[3] == 255));
            }
            // Encode: the family encodes through the 8-bit kernel and
            // widens afterwards (crate policy, identical to the 4:2:0 /
            // 4:2:2 members), so the pin is the 8-bit 4:4:4 encode, the
            // 8-bit vertical pair-average, then the exact widen.
            let back = conv(&direct, rgb, f, w, h);
            let via8 = conv(
                &conv(&direct, rgb, PixelFormat::Yuv444P, w, h),
                PixelFormat::Yuv444P,
                PixelFormat::Yuv440P,
                w,
                h,
            );
            let via = conv(&via8, PixelFormat::Yuv440P, f, w, h);
            assert_eq!(
                tight(&back, f, w, h),
                tight(&via, f, w, h),
                "{rgb:?} → {f:?}"
            );
        }
    }
}

/// Deep matrix: `Yuv440P16Le` ↔ `Rgb48Le` is the 16-bit 4:4:4 kernel
/// around a 16-bit vertical resample — pinned to the explicit route.
#[test]
fn deep_matrix_matches_444_route() {
    let (w, h) = (6, 4);
    let src = build_yuv(PixelFormat::Yuv440P16Le, w, h, 2);
    let direct = conv(&src, PixelFormat::Yuv440P16Le, PixelFormat::Rgb48Le, w, h);
    let via = conv(
        &conv(
            &src,
            PixelFormat::Yuv440P16Le,
            PixelFormat::Yuv444P16Le,
            w,
            h,
        ),
        PixelFormat::Yuv444P16Le,
        PixelFormat::Rgb48Le,
        w,
        h,
    );
    assert_eq!(direct.planes[0].data, via.planes[0].data);
    let back = conv(
        &direct,
        PixelFormat::Rgb48Le,
        PixelFormat::Yuv440P16Le,
        w,
        h,
    );
    let via = conv(
        &conv(
            &direct,
            PixelFormat::Rgb48Le,
            PixelFormat::Yuv444P16Le,
            w,
            h,
        ),
        PixelFormat::Yuv444P16Le,
        PixelFormat::Yuv440P16Le,
        w,
        h,
    );
    assert_eq!(
        tight(&back, PixelFormat::Yuv440P16Le, w, h),
        tight(&via, PixelFormat::Yuv440P16Le, w, h)
    );
    // The 10/12-bit members reach deep RGB through the exact widen to
    // the 16-bit tier (staged), never through an 8-bit hop.
    let src10 = build_yuv(PixelFormat::Yuv440P10Le, w, h, 0);
    let deep = conv(&src10, PixelFormat::Yuv440P10Le, PixelFormat::Rgb48Le, w, h);
    let via = conv(
        &conv(
            &src10,
            PixelFormat::Yuv440P10Le,
            PixelFormat::Yuv440P16Le,
            w,
            h,
        ),
        PixelFormat::Yuv440P16Le,
        PixelFormat::Rgb48Le,
        w,
        h,
    );
    assert_eq!(deep.planes[0].data, via.planes[0].data);
}

/// Gray8 interop: luma extraction / neutral-chroma synthesis, same as
/// every other family member; the synthesised chroma plane has the
/// 4:4:0 geometry.
#[test]
fn gray_interop() {
    let (w, h) = (5, 4);
    let src = build_yuv(PixelFormat::Yuv440P, w, h, 0);
    let g = conv(&src, PixelFormat::Yuv440P, PixelFormat::Gray8, w, h);
    let g444 = conv(
        &conv(&src, PixelFormat::Yuv440P, PixelFormat::Yuv444P, w, h),
        PixelFormat::Yuv444P,
        PixelFormat::Gray8,
        w,
        h,
    );
    assert_eq!(g.planes[0].data, g444.planes[0].data);
    for &f in &FAMILY_440 {
        let back = conv(&g, PixelFormat::Gray8, f, w, h);
        let sb = if FormatInfo::of(f).bit_depth > 8 {
            2
        } else {
            1
        };
        assert_eq!(back.planes[1].data.len(), w * (h / 2) * sb);
        assert_eq!(back.planes[2].data.len(), w * (h / 2) * sb);
        let bits = FormatInfo::of(f).bit_depth as u32;
        let mid = 1u32 << (bits - 1);
        for c in back.planes[1].data.chunks(sb) {
            let v = if sb == 2 {
                u16::from_le_bytes([c[0], c[1]]) as u32
            } else {
                c[0] as u32
            };
            assert_eq!(v, mid, "{f:?}");
        }
    }
}
