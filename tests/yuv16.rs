//! 16-bit planar YUV (`Yuv420P16Le` / `Yuv422P16Le` / `Yuv444P16Le`)
//! correctness suite.
//!
//! The 16-bit trio stores the same three-plane LE-word layout as the
//! 10/12-bit variants but with ALL 16 bits of every word significant
//! (full-scale 65535). Three properties are pinned here, each against an
//! independent model rather than the implementation's own arithmetic:
//!
//! 1. **Depth ladder** — 8/10/12 → 16 widening tracks the ideal
//!    real-valued rescale `v · (2¹⁶ − 1) / (2ⁿ − 1)` (exactly for
//!    8-bit, within one code otherwise), is monotonic, maps zero to
//!    zero and peak to peak, and round-trips losslessly through the
//!    truncating narrow.
//! 2. **Chroma resample** — the direct 16-bit subsample moves match a
//!    plain f64 round-half-up box/pair-average model sample-for-sample,
//!    and luma is copied word-for-word.
//! 3. **Staged fidelity** — deep YUV staged routes pivot through the
//!    16-bit tier, so e.g. `Yuv420P16Le → Yuv422P10Le` keeps the top 10
//!    bits of every luma word instead of quantising through an 8-bit
//!    intermediate.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::yuv::{
    depth_down_le16_plane, depth_rescale_le16_plane, depth_up_8_to_le16_plane,
};
use oxideav_pixfmt::{convert, supports, supports_direct, ConvertOptions, FrameInfo};

fn rd16(buf: &[u8], i: usize) -> u16 {
    (buf[i * 2] as u16) | ((buf[i * 2 + 1] as u16) << 8)
}

fn wr16(buf: &mut [u8], i: usize, v: u16) {
    buf[i * 2] = (v & 0xFF) as u8;
    buf[i * 2 + 1] = (v >> 8) as u8;
}

// -------------------------------------------------------------------------
// 1. Depth-ladder reference model.

/// Ideal real-valued depth rescale, rounded to nearest.
fn ideal_rescale(v: u32, src_bits: u32, dst_bits: u32) -> u32 {
    let smax = (1u64 << src_bits) - 1;
    let dmax = (1u64 << dst_bits) - 1;
    ((v as f64) * (dmax as f64) / (smax as f64)).round() as u32
}

/// 8 → 16 widening is EXACTLY the ideal rescale: 65535 / 255 = 257 with
/// no remainder, and MSB replication with an 8-bit period is ×257.
#[test]
fn widen_8_to_16_is_exact_ideal_rescale() {
    let src: Vec<u8> = (0..=255u8).collect();
    let mut dst = vec![0u8; 256 * 2];
    depth_up_8_to_le16_plane(&src, &mut dst, 256, 16);
    for (v, i) in (0..=255u32).zip(0..256) {
        let got = rd16(&dst, i) as u32;
        assert_eq!(got, ideal_rescale(v, 8, 16), "v = {v}");
        assert_eq!(got, v * 257, "v = {v}");
    }
}

/// 10/12 → 16 widening tracks the ideal rescale within one output code,
/// is strictly monotonic, and maps the rails exactly (0 → 0,
/// full-scale → 65535).
#[test]
fn widen_10_12_to_16_tracks_ideal_within_one() {
    for src_bits in [10u32, 12u32] {
        let n = 1usize << src_bits;
        let mut src = vec![0u8; n * 2];
        for v in 0..n {
            wr16(&mut src, v, v as u16);
        }
        let mut dst = vec![0u8; n * 2];
        depth_rescale_le16_plane(&src, &mut dst, n, src_bits, 16);
        let mut prev: i64 = -1;
        for v in 0..n {
            let got = rd16(&dst, v) as i64;
            let want = ideal_rescale(v as u32, src_bits, 16) as i64;
            assert!(
                (got - want).abs() <= 1,
                "{src_bits}-bit v = {v}: got {got}, ideal {want}"
            );
            assert!(got > prev, "{src_bits}-bit v = {v}: not monotonic");
            prev = got;
        }
        assert_eq!(rd16(&dst, 0), 0);
        assert_eq!(
            rd16(&dst, n - 1),
            65535,
            "{src_bits}-bit peak must hit peak"
        );
    }
}

/// Narrow-of-widen is the identity for every source depth feeding the
/// 16-bit rung (the truncation drops exactly the replicated fill).
#[test]
fn depth_16_roundtrips_are_lossless() {
    // 8 → 16 → 8.
    let src8: Vec<u8> = (0..=255u8).collect();
    let mut wide = vec![0u8; 256 * 2];
    depth_up_8_to_le16_plane(&src8, &mut wide, 256, 16);
    let mut back8 = vec![0u8; 256];
    depth_down_le16_plane(&wide, &mut back8, 256, 16);
    assert_eq!(back8, src8);
    // 10 → 16 → 10 and 12 → 16 → 12.
    for bits in [10u32, 12u32] {
        let n = 1usize << bits;
        let mut src = vec![0u8; n * 2];
        for v in 0..n {
            wr16(&mut src, v, v as u16);
        }
        let mut wide = vec![0u8; n * 2];
        depth_rescale_le16_plane(&src, &mut wide, n, bits, 16);
        let mut back = vec![0u8; n * 2];
        depth_rescale_le16_plane(&wide, &mut back, n, 16, bits);
        assert_eq!(back, src, "{bits} → 16 → {bits} must be lossless");
    }
}

/// 16 → 8 narrowing keeps the high byte of every word (truncation, per
/// the crate-wide no-dither depth policy).
#[test]
fn narrow_16_to_8_keeps_high_byte() {
    let samples: &[u16] = &[0, 1, 255, 256, 257, 0x1234, 0x7FFF, 0x8000, 0xFF00, 0xFFFF];
    let mut src = vec![0u8; samples.len() * 2];
    for (i, &v) in samples.iter().enumerate() {
        wr16(&mut src, i, v);
    }
    let mut dst = vec![0u8; samples.len()];
    depth_down_le16_plane(&src, &mut dst, samples.len(), 16);
    for (i, &v) in samples.iter().enumerate() {
        assert_eq!(dst[i], (v >> 8) as u8, "sample {v:#06x}");
    }
}

// -------------------------------------------------------------------------
// Frame builders.

/// Deterministic 16-bit planar frame exercising the full word range
/// (including values above any 12-bit ceiling — all 16 bits are
/// significant on this family).
fn synth16(fmt: PixelFormat, w: usize, h: usize, wsub: usize, hsub: usize) -> VideoFrame {
    let cw = w / wsub;
    let ch = h / hsub;
    let mut yp = vec![0u8; w * h * 2];
    let mut up = vec![0u8; cw * ch * 2];
    let mut vp = vec![0u8; cw * ch * 2];
    for y in 0..h {
        for x in 0..w {
            wr16(&mut yp, y * w + x, ((x * 9973 + y * 31337) % 65536) as u16);
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            wr16(
                &mut up,
                y * cw + x,
                ((x * 12345 + y * 54321) % 65536) as u16,
            );
            wr16(
                &mut vp,
                y * cw + x,
                ((x * 41999 + y * 27644 + 7) % 65536) as u16,
            );
        }
    }
    let _ = fmt;
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: up,
            },
            VideoPlane {
                stride: cw * 2,
                data: vp,
            },
        ],
    }
}

// -------------------------------------------------------------------------
// 2. Chroma-resample reference model (f64 round-half-up averages).

fn model_pair_avg(a: u16, b: u16) -> u16 {
    ((a as f64 + b as f64) / 2.0).round() as u16
}

fn model_box_avg(s: [u16; 4]) -> u16 {
    (s.iter().map(|&v| v as f64).sum::<f64>() / 4.0).round() as u16
}

/// `Yuv444P16Le → Yuv422P16Le`: luma word-copied, chroma equals the
/// horizontal pair-average model at every sample.
#[test]
fn resample16_444_to_422_matches_model() {
    let (w, h) = (16usize, 8usize);
    let src = synth16(PixelFormat::Yuv444P16Le, w, h, 1, 1);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv444P16Le, w as u32, h as u32),
        PixelFormat::Yuv422P16Le,
        &ConvertOptions::default(),
    )
    .expect("444P16 → 422P16");
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma word copy");
    let cw = w / 2;
    for plane in [1usize, 2] {
        for row in 0..h {
            for cc in 0..cw {
                let a = rd16(&src.planes[plane].data, row * w + cc * 2);
                let b = rd16(&src.planes[plane].data, row * w + cc * 2 + 1);
                let got = rd16(&out.planes[plane].data, row * cw + cc);
                assert_eq!(
                    got,
                    model_pair_avg(a, b),
                    "plane {plane} row {row} col {cc}"
                );
            }
        }
    }
}

/// `Yuv444P16Le → Yuv420P16Le`: chroma equals the 2×2 round-half-up box
/// average model.
#[test]
fn resample16_444_to_420_matches_model() {
    let (w, h) = (16usize, 8usize);
    let src = synth16(PixelFormat::Yuv444P16Le, w, h, 1, 1);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv444P16Le, w as u32, h as u32),
        PixelFormat::Yuv420P16Le,
        &ConvertOptions::default(),
    )
    .expect("444P16 → 420P16");
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma word copy");
    let (cw, ch) = (w / 2, h / 2);
    for plane in [1usize, 2] {
        for cr in 0..ch {
            for cc in 0..cw {
                let s = [
                    rd16(&src.planes[plane].data, (cr * 2) * w + cc * 2),
                    rd16(&src.planes[plane].data, (cr * 2) * w + cc * 2 + 1),
                    rd16(&src.planes[plane].data, (cr * 2 + 1) * w + cc * 2),
                    rd16(&src.planes[plane].data, (cr * 2 + 1) * w + cc * 2 + 1),
                ];
                let got = rd16(&out.planes[plane].data, cr * cw + cc);
                assert_eq!(got, model_box_avg(s), "plane {plane} ({cr},{cc})");
            }
        }
    }
}

/// `Yuv422P16Le → Yuv420P16Le`: vertical pair-average model.
#[test]
fn resample16_422_to_420_matches_model() {
    let (w, h) = (16usize, 8usize);
    let src = synth16(PixelFormat::Yuv422P16Le, w, h, 2, 1);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv422P16Le, w as u32, h as u32),
        PixelFormat::Yuv420P16Le,
        &ConvertOptions::default(),
    )
    .expect("422P16 → 420P16");
    assert_eq!(out.planes[0].data, src.planes[0].data, "luma word copy");
    let (cw, ch) = (w / 2, h / 2);
    for plane in [1usize, 2] {
        for cr in 0..ch {
            for cc in 0..cw {
                let a = rd16(&src.planes[plane].data, (cr * 2) * cw + cc);
                let b = rd16(&src.planes[plane].data, (cr * 2 + 1) * cw + cc);
                let got = rd16(&out.planes[plane].data, cr * cw + cc);
                assert_eq!(got, model_pair_avg(a, b), "plane {plane} ({cr},{cc})");
            }
        }
    }
}

/// Expanders are nearest-sample duplicates, so shrink-of-expand is the
/// identity: 422 → 444 → 422 and 420 → 444 / 422 → 420 round-trip every
/// plane word-exactly.
#[test]
fn resample16_expand_then_shrink_is_identity() {
    let opts = ConvertOptions::default();
    let (w, h) = (16u32, 8u32);
    let cases = [
        (
            PixelFormat::Yuv422P16Le,
            2usize,
            1usize,
            PixelFormat::Yuv444P16Le,
        ),
        (PixelFormat::Yuv420P16Le, 2, 2, PixelFormat::Yuv444P16Le),
        (PixelFormat::Yuv420P16Le, 2, 2, PixelFormat::Yuv422P16Le),
    ];
    for (fmt, wsub, hsub, via) in cases {
        let src = synth16(fmt, w as usize, h as usize, wsub, hsub);
        let up = convert(&src, FrameInfo::new(fmt, w, h), via, &opts).expect("expand");
        let back = convert(&up, FrameInfo::new(via, w, h), fmt, &opts).expect("shrink");
        for p in 0..3 {
            assert_eq!(
                back.planes[p].data, src.planes[p].data,
                "{fmt:?} → {via:?} → {fmt:?} plane {p}"
            );
        }
    }
}

// -------------------------------------------------------------------------
// 3. Depth ladder through convert() + staged fidelity.

/// Full-frame 8 ↔ 16 round-trip through convert() is lossless on all
/// three subsamplings, and the widen hits the exact ×257 mapping.
#[test]
fn convert_depth_16_roundtrip_all_subsamplings() {
    let opts = ConvertOptions::default();
    let (w, h) = (16u32, 8u32);
    let cases = [
        (
            PixelFormat::Yuv420P,
            PixelFormat::Yuv420P16Le,
            2usize,
            2usize,
        ),
        (PixelFormat::Yuv422P, PixelFormat::Yuv422P16Le, 2, 1),
        (PixelFormat::Yuv444P, PixelFormat::Yuv444P16Le, 1, 1),
    ];
    for (fmt8, fmt16, wsub, hsub) in cases {
        let (wu, hu) = (w as usize, h as usize);
        let (cw, ch) = (wu / wsub, hu / hsub);
        let mk = |mul: usize, add: usize, n: usize| -> Vec<u8> {
            (0..n).map(|i| ((i * mul + add) & 0xFF) as u8).collect()
        };
        let src = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: wu,
                    data: mk(13, 7, wu * hu),
                },
                VideoPlane {
                    stride: cw,
                    data: mk(5, 3, cw * ch),
                },
                VideoPlane {
                    stride: cw,
                    data: mk(29, 17, cw * ch),
                },
            ],
        };
        let deep = convert(&src, FrameInfo::new(fmt8, w, h), fmt16, &opts).expect("8 → 16");
        // Exact ×257 widening on every plane.
        for p in 0..3 {
            let n = src.planes[p].data.len();
            for i in 0..n {
                let got = rd16(&deep.planes[p].data, i) as u32;
                assert_eq!(got, src.planes[p].data[i] as u32 * 257);
            }
        }
        let back = convert(&deep, FrameInfo::new(fmt16, w, h), fmt8, &opts).expect("16 → 8");
        for p in 0..3 {
            assert_eq!(
                back.planes[p].data, src.planes[p].data,
                "{fmt8:?} plane {p}"
            );
        }
    }
}

/// 10/12 ↔ 16 through convert(): lossless round-trip and peak-to-peak
/// mapping on the way up.
#[test]
fn convert_cross_depth_16_roundtrips() {
    let opts = ConvertOptions::default();
    let (w, h) = (8u32, 8u32);
    let cases = [
        (PixelFormat::Yuv444P10Le, PixelFormat::Yuv444P16Le, 10u32),
        (PixelFormat::Yuv444P12Le, PixelFormat::Yuv444P16Le, 12u32),
    ];
    for (fmt_lo, fmt16, bits) in cases {
        let n = (w * h) as usize;
        let mask = (1u32 << bits) - 1;
        let mut plane_bytes = vec![0u8; n * 2];
        for i in 0..n {
            // Cover the rails and a spread of interior codes.
            let v = match i {
                0 => 0,
                1 => mask,
                _ => (i as u32 * 40503) & mask,
            };
            wr16(&mut plane_bytes, i, v as u16);
        }
        let src = VideoFrame {
            pts: None,
            planes: (0..3)
                .map(|_| VideoPlane {
                    stride: w as usize * 2,
                    data: plane_bytes.clone(),
                })
                .collect(),
        };
        let deep = convert(&src, FrameInfo::new(fmt_lo, w, h), fmt16, &opts).expect("lo → 16");
        // Rails: 0 → 0, full-scale → 65535.
        assert_eq!(rd16(&deep.planes[0].data, 0), 0);
        assert_eq!(rd16(&deep.planes[0].data, 1), 65535);
        let back = convert(&deep, FrameInfo::new(fmt16, w, h), fmt_lo, &opts).expect("16 → lo");
        for p in 0..3 {
            assert_eq!(
                back.planes[p].data, src.planes[p].data,
                "{fmt_lo:?} plane {p}"
            );
        }
    }
}

/// Cross-depth + cross-subsampling pairs involving the 16-bit trio all
/// resolve (directly or via one pivot), and the deep staged routes keep
/// full input precision on luma: `Yuv420P16Le → Yuv422P10Le` pivots
/// through `Yuv422P16Le`, so every output luma word is exactly the top
/// 10 bits of the input word (an 8-bit pivot would zero the low 2).
#[test]
fn staged_deep_pivot_preserves_luma_precision() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    let src = synth16(PixelFormat::Yuv420P16Le, w, h, 2, 2);
    let info = FrameInfo::new(PixelFormat::Yuv420P16Le, w as u32, h as u32);

    assert!(supports(PixelFormat::Yuv420P16Le, PixelFormat::Yuv422P10Le));
    assert!(!supports_direct(
        PixelFormat::Yuv420P16Le,
        PixelFormat::Yuv422P10Le
    ));
    let out = convert(&src, info, PixelFormat::Yuv422P10Le, &opts).expect("420P16 → 422P10");
    for i in 0..w * h {
        let want = rd16(&src.planes[0].data, i) >> 6; // top 10 of 16
        assert_eq!(rd16(&out.planes[0].data, i), want, "luma sample {i}");
    }

    // Same reasoning down to the 8-bit sibling at a different
    // subsampling: luma must be exactly the high byte.
    let out8 = convert(&src, info, PixelFormat::Yuv422P, &opts).expect("420P16 → 422P");
    for i in 0..w * h {
        let want = (rd16(&src.planes[0].data, i) >> 8) as u8;
        assert_eq!(out8.planes[0].data[i], want, "luma sample {i}");
    }
}

/// Every ordered pair between the 16-bit trio and the full planar YUV
/// depth family (8 / 10 / 12 / 16 at 4:2:0 / 4:2:2 / 4:4:4) resolves —
/// directly or through one staged pivot.
#[test]
fn family_closure_for_16bit_formats() {
    use PixelFormat as P;
    const FAMILY: &[PixelFormat] = &[
        P::Yuv420P,
        P::Yuv422P,
        P::Yuv444P,
        P::Yuv420P10Le,
        P::Yuv422P10Le,
        P::Yuv444P10Le,
        P::Yuv420P12Le,
        P::Yuv422P12Le,
        P::Yuv444P12Le,
        P::Yuv420P16Le,
        P::Yuv422P16Le,
        P::Yuv444P16Le,
    ];
    const TRIO: &[PixelFormat] = &[P::Yuv420P16Le, P::Yuv422P16Le, P::Yuv444P16Le];
    for &a in TRIO {
        for &b in FAMILY {
            assert!(supports(a, b), "{a:?} → {b:?} must resolve");
            assert!(supports(b, a), "{b:?} → {a:?} must resolve");
        }
        // And the RGB / gray world stays reachable.
        for &b in &[P::Rgb24, P::Rgba, P::Gray8] {
            assert!(supports(a, b), "{a:?} → {b:?} must resolve");
            assert!(supports(b, a), "{b:?} → {a:?} must resolve");
        }
    }
}
