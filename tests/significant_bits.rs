//! Per-plane significant-bits side-channel policy suite.
//!
//! oxideav-core 0.1.31 lets a `VideoFrame` carry one LSB-anchored byte
//! per image plane naming that plane's *significant* depth (e.g.
//! `[12, 10, 10]` for a wavelet codec's 12-bit luma + 10-bit chroma on
//! a `Yuv444P12Le` or `Yuv444P16Le` surface). The `convert()` policy
//! under test:
//!
//! - marked planes convert at their recorded depth: they are normalised
//!   to the surface's nominal depth by the crate-wide MSB-replicating
//!   widen before dispatch, so a record-carrying frame converts exactly
//!   like the equivalent nominal-depth frame;
//! - outputs are always nominal-depth and never carry a record
//!   (passthrough `src == dst` is the only exception);
//! - invalid values (0, or greater than the surface's nominal depth)
//!   reject with `Error::Invalid`; extra record bytes are ignored;
//! - a record on `Pal8` is ignored (indices are not magnitudes) and
//!   composes with the palette side-channel.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

/// MSB-replicating widen mirroring the documented policy (pattern
/// repetition into the freed low bits).
fn widen(v: u32, from: u32, to: u32) -> u32 {
    let mut out = v << (to - from);
    let mut fill = to - from;
    while fill > 0 {
        let take = fill.min(from);
        out |= (v >> (from - take)) << (fill - take);
        fill -= take;
    }
    out
}

fn rd16(buf: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]])
}

fn le16_plane(vals: impl Iterator<Item = u16>) -> Vec<u8> {
    vals.flat_map(|v| v.to_le_bytes()).collect()
}

/// A deterministic 3-plane 4:4:4 LE16 frame whose plane `k` samples are
/// masked to `bits[k]` significant bits.
fn synth_444_le16(w: usize, h: usize, bits: [u32; 3]) -> VideoFrame {
    let plane = |mul: u16, add: u16, b: u32| -> Vec<u8> {
        let mask = ((1u32 << b) - 1) as u16;
        le16_plane((0..w * h).map(move |i| (i as u16).wrapping_mul(mul).wrapping_add(add) & mask))
    };
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: plane(2311, 17, bits[0]),
            },
            VideoPlane {
                stride: w * 2,
                data: plane(929, 3, bits[1]),
            },
            VideoPlane {
                stride: w * 2,
                data: plane(1597, 41, bits[2]),
            },
        ],
    }
}

/// The ruling case: a `Yuv444P16Le` surface marked `[12, 10, 10]`
/// converts as 12-bit luma + 10-bit chroma, not as full-range 16 — the
/// 8-bit target sees the top 8 of each plane's *significant* bits.
#[test]
fn p16_mixed_record_to_8bit_reference() {
    let (w, h) = (8usize, 8usize);
    let src = synth_444_le16(w, h, [12, 10, 10]).with_significant_bits(vec![12, 10, 10]);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv444P16Le, w as u32, h as u32),
        PixelFormat::Yuv444P,
        &ConvertOptions::default(),
    )
    .expect("p16 + record → 8-bit");
    for i in 0..w * h {
        // Widen 12 → 16 then truncate 16 → 8 keeps the top 8 of the
        // 12 significant bits: v >> 4. Without the record the same
        // frame would produce (v16 >> 8) = v >> 4 only when v had its
        // top four storage bits zero — i.e. always 0..=15 here, so the
        // record genuinely changes the output scale.
        assert_eq!(
            out.planes[0].data[i],
            (rd16(&src.planes[0].data, i) >> 4) as u8,
            "luma {i}"
        );
        assert_eq!(
            out.planes[1].data[i],
            (rd16(&src.planes[1].data, i) >> 2) as u8,
            "U {i}"
        );
        assert_eq!(
            out.planes[2].data[i],
            (rd16(&src.planes[2].data, i) >> 2) as u8,
            "V {i}"
        );
    }
    // Output carries no side-channel record.
    assert_eq!(out.significant_bits(), None);
    assert_eq!(out.image_plane_count(), out.planes.len());
}

/// Same mixed record on the `Yuv444P12Le` surface (12-bit luma is
/// already nominal there; only the chroma is refined) → both the 8-bit
/// and 16-bit targets match the widen-then-convert reference.
#[test]
fn p12_mixed_record_to_8_and_16_bit_reference() {
    let (w, h) = (8usize, 8usize);
    let src = synth_444_le16(w, h, [12, 10, 10]).with_significant_bits(vec![12, 10, 10]);
    let info = FrameInfo::new(PixelFormat::Yuv444P12Le, w as u32, h as u32);
    let opts = ConvertOptions::default();

    let out8 = convert(&src, info, PixelFormat::Yuv444P, &opts).expect("→ 8-bit");
    for i in 0..w * h {
        assert_eq!(
            out8.planes[0].data[i],
            (rd16(&src.planes[0].data, i) >> 4) as u8,
            "luma {i}"
        );
        // Chroma: widen 10 → 12 (nominal), then truncate 12 → 8 —
        // algebraically the top 8 of the 10 significant bits.
        assert_eq!(
            out8.planes[1].data[i],
            (rd16(&src.planes[1].data, i) >> 2) as u8,
            "U {i}"
        );
    }

    let out16 = convert(&src, info, PixelFormat::Yuv444P16Le, &opts).expect("→ 16-bit");
    for i in 0..w * h {
        assert_eq!(
            rd16(&out16.planes[0].data, i) as u32,
            widen(rd16(&src.planes[0].data, i) as u32, 12, 16),
            "luma {i}"
        );
        // Chroma: widen 10 → 12 at normalisation, then 12 → 16 in the
        // depth ladder (compositional, as documented).
        let v = rd16(&src.planes[1].data, i) as u32;
        assert_eq!(
            rd16(&out16.planes[1].data, i) as u32,
            widen(widen(v, 10, 12), 12, 16),
            "U {i}"
        );
    }
}

/// The equivalence that defines the policy: a record-carrying frame
/// converts byte-identically to the frame whose marked planes were
/// materialised (widened) to the nominal depth up front — across a
/// depth-ladder move, an RGB decode, and a staged route.
#[test]
fn record_equals_materialised_frame_everywhere() {
    let (w, h) = (8usize, 8usize);
    let carried = synth_444_le16(w, h, [12, 10, 10]).with_significant_bits(vec![12, 10, 10]);
    // Hand-materialised twin: same samples widened to 16 bits.
    let materialised = VideoFrame {
        pts: None,
        planes: carried
            .image_planes()
            .iter()
            .enumerate()
            .map(|(p, plane)| VideoPlane {
                stride: plane.stride,
                data: le16_plane(plane.data.chunks_exact(2).map(|c| {
                    let v = u16::from_le_bytes([c[0], c[1]]) as u32;
                    widen(v, if p == 0 { 12 } else { 10 }, 16) as u16
                })),
            })
            .collect(),
    };
    let info = FrameInfo::new(PixelFormat::Yuv444P16Le, w as u32, h as u32);
    let opts = ConvertOptions::default();
    for dst in [
        PixelFormat::Yuv444P,     // depth ladder
        PixelFormat::Yuv420P12Le, // computed family op (resample + narrow)
        PixelFormat::Rgba,        // matrix decode
        PixelFormat::Bgra,        // staged route (Rgba pivot)
        PixelFormat::Gray8,       // luma extraction
    ] {
        let a = convert(&carried, info, dst, &opts).expect("record-carrying");
        let b = convert(&materialised, info, dst, &opts).expect("materialised");
        assert_eq!(a.planes.len(), b.planes.len(), "{dst:?}");
        for p in 0..a.planes.len() {
            assert_eq!(a.planes[p].data, b.planes[p].data, "{dst:?} plane {p}");
        }
    }
}

/// 8-bit storage surfaces honour sub-8 records: a `Gray8` frame marked
/// `[4]` converts to `Gray16Le` as 4-bit content (full-scale 15 maps to
/// full-scale 65535).
#[test]
fn sub_8bit_record_on_byte_surface() {
    let (w, h) = (8usize, 4usize);
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w,
            data: (0..w * h).map(|i| (i % 16) as u8).collect(),
        }],
    }
    .with_significant_bits(vec![4]);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32),
        PixelFormat::Gray16Le,
        &ConvertOptions::default(),
    )
    .expect("gray8[4] → gray16");
    for i in 0..w * h {
        let v4 = (src.planes[0].data[i] & 0xF) as u32;
        let v8 = widen(v4, 4, 8);
        // Gray8 → Gray16Le is the exact ×257 widen of the normalised
        // 8-bit value.
        assert_eq!(rd16(&out.planes[0].data, i) as u32, v8 * 257, "sample {i}");
    }
    // Rails: 0 stays 0, 4-bit full-scale hits 16-bit full-scale.
    assert_eq!(rd16(&out.planes[0].data, 0), 0);
    assert_eq!(rd16(&out.planes[0].data, 15), 65535);
}

/// Hostile records: zero bits and bits above the surface's nominal
/// depth must reject with an error (never panic); extra bytes beyond
/// the image-plane count are ignored.
#[test]
fn hostile_records_reject_or_are_ignored() {
    let (w, h) = (8usize, 8usize);
    let opts = ConvertOptions::default();
    let info = FrameInfo::new(PixelFormat::Yuv444P12Le, w as u32, h as u32);

    // Zero significant bits: invalid.
    let src = synth_444_le16(w, h, [12, 12, 12]).with_significant_bits(vec![12, 0, 12]);
    assert!(convert(&src, info, PixelFormat::Yuv444P, &opts).is_err());

    // More bits than the surface's nominal depth: invalid (a record
    // may only refine the depth downward).
    let src = synth_444_le16(w, h, [12, 12, 12]).with_significant_bits(vec![14, 12, 12]);
    assert!(convert(&src, info, PixelFormat::Yuv444P, &opts).is_err());

    // Sub-storage but above nominal on a partial-depth surface: a
    // 10-bit surface may not claim 12 significant bits even though the
    // storage word could hold them.
    let src10 = synth_444_le16(w, h, [10, 10, 10]).with_significant_bits(vec![12, 10, 10]);
    let info10 = FrameInfo::new(PixelFormat::Yuv444P10Le, w as u32, h as u32);
    assert!(convert(&src10, info10, PixelFormat::Yuv444P, &opts).is_err());

    // 9 significant bits on an 8-bit surface: invalid.
    let gray = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w,
            data: vec![0x40; w * h],
        }],
    }
    .with_significant_bits(vec![9]);
    let ginfo = FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32);
    assert!(convert(&gray, ginfo, PixelFormat::Gray16Le, &opts).is_err());

    // Extra record bytes beyond the image-plane count are ignored, even
    // hostile ones.
    let src = synth_444_le16(w, h, [12, 12, 12]).with_significant_bits(vec![12, 12, 12, 0, 99]);
    let out = convert(&src, info, PixelFormat::Yuv444P, &opts).expect("extra bytes ignored");
    let bare = synth_444_le16(w, h, [12, 12, 12]);
    let want = convert(&bare, info, PixelFormat::Yuv444P, &opts).expect("no record");
    for p in 0..3 {
        assert_eq!(out.planes[p].data, want.planes[p].data, "plane {p}");
    }
}

/// An all-nominal record is a no-op: output matches the recordless
/// conversion byte-for-byte. A record shorter than the plane count
/// leaves the uncovered planes at nominal depth.
#[test]
fn nominal_and_short_records() {
    let (w, h) = (8usize, 8usize);
    let opts = ConvertOptions::default();
    let info = FrameInfo::new(PixelFormat::Yuv444P12Le, w as u32, h as u32);
    let bare = synth_444_le16(w, h, [12, 10, 12]);
    let want_nominal = convert(&bare, info, PixelFormat::Yuv444P, &opts).expect("bare");

    let all_nominal = bare.clone().with_significant_bits(vec![12, 12, 12]);
    let out = convert(&all_nominal, info, PixelFormat::Yuv444P, &opts).expect("nominal record");
    for p in 0..3 {
        assert_eq!(out.planes[p].data, want_nominal.planes[p].data);
    }

    // Short record [12, 10]: U converts as 10-bit, V stays nominal.
    let short = bare.clone().with_significant_bits(vec![12, 10]);
    let out = convert(&short, info, PixelFormat::Yuv444P, &opts).expect("short record");
    for i in 0..w * h {
        assert_eq!(
            out.planes[1].data[i],
            (rd16(&bare.planes[1].data, i) >> 2) as u8,
            "U (covered, 10-bit) {i}"
        );
        assert_eq!(
            out.planes[2].data[i],
            (rd16(&bare.planes[2].data, i) >> 4) as u8,
            "V (uncovered, nominal) {i}"
        );
    }
}

/// Pal8 composition: a significant-bits record on a palette frame is
/// ignored — the expansion equals the record-less one — and the palette
/// side-channel is honoured as usual alongside it.
#[test]
fn pal8_record_is_ignored_and_composes_with_palette() {
    let (w, h) = (8usize, 4usize);
    let opts = ConvertOptions::default();
    let pal: Vec<u8> = (0..=255u16)
        .flat_map(|i| {
            [
                i as u8,
                (i as u8).wrapping_mul(3),
                (i as u8).wrapping_mul(7),
            ]
        })
        .collect();
    let indices: Vec<u8> = (0..w * h).map(|i| (i * 11 % 256) as u8).collect();
    let bare = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w,
            data: indices.clone(),
        }],
    }
    .with_palette(pal.clone());
    let carried = bare.clone().with_significant_bits(vec![4]);
    // Both side-channels attached, in composition.
    assert!(carried.palette().is_some());
    assert!(carried.significant_bits().is_some());
    assert_eq!(carried.image_plane_count(), 1);

    let info = FrameInfo::new(PixelFormat::Pal8, w as u32, h as u32);
    let a = convert(&carried, info, PixelFormat::Rgba, &opts).expect("with record");
    let b = convert(&bare, info, PixelFormat::Rgba, &opts).expect("without record");
    assert_eq!(a.planes[0].data, b.planes[0].data);
}

/// Passthrough (`src == dst`) is the documented exception: the frame is
/// untouched, so its record still describes it and rides along.
#[test]
fn passthrough_keeps_record() {
    let (w, h) = (8usize, 8usize);
    let src = synth_444_le16(w, h, [12, 10, 10]).with_significant_bits(vec![12, 10, 10]);
    let info = FrameInfo::new(PixelFormat::Yuv444P16Le, w as u32, h as u32);
    let out = convert(
        &src,
        info,
        PixelFormat::Yuv444P16Le,
        &ConvertOptions::default(),
    )
    .expect("passthrough");
    assert_eq!(out.significant_bits(), Some(&[12u8, 10, 10][..]));
    assert_eq!(out.image_plane_count(), 3);
}

/// The record changes staged routes too: a deep Yuva frame with
/// refined alpha depth carries the *significant* alpha across the Rgba
/// pivot (top 8 of the recorded depth, not of the storage depth).
#[test]
fn record_applies_to_alpha_plane_on_staged_route() {
    let (w, h) = (8usize, 8usize);
    // Yuva444P16Le surface where alpha genuinely holds 10-bit samples.
    let mask10 = ((1u32 << 10) - 1) as u16;
    let mut planes: Vec<VideoPlane> = (0..3)
        .map(|p| VideoPlane {
            stride: w * 2,
            data: le16_plane((0..w * h).map(move |i| ((i * 257 + p * 71) % 65536) as u16)),
        })
        .collect();
    let alpha10 = le16_plane((0..w * h).map(|i| (i as u16).wrapping_mul(613) & mask10));
    planes.push(VideoPlane {
        stride: w * 2,
        data: alpha10.clone(),
    });
    let src = VideoFrame { pts: None, planes }.with_significant_bits(vec![16, 16, 16, 10]);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva444P16Le, w as u32, h as u32),
        PixelFormat::Bgra,
        &ConvertOptions::default(),
    )
    .expect("yuva + record → bgra");
    for i in 0..w * h {
        let a10 = rd16(&alpha10, i) as u32;
        let want = (widen(a10, 10, 16) >> 8) as u8;
        assert_eq!(out.planes[0].data[i * 4 + 3], want, "alpha {i}");
    }
}

/// A record on a `Gbrp16Le` surface normalises the marked plane before
/// the plane-reorder hop: plane 2 (R) marked 12-bit reaches `Rgb48Le`
/// as its MSB-replicated 16-bit widen, while unmarked full-width
/// planes ride across untouched.
#[test]
fn record_on_gbr16_surface_normalises_before_reorder() {
    let (w, h) = (8usize, 4usize);
    let mask12 = ((1u32 << 12) - 1) as u16;
    let g = le16_plane((0..w * h).map(|i| (i as u16).wrapping_mul(2311).wrapping_add(17)));
    let b = le16_plane((0..w * h).map(|i| (i as u16).wrapping_mul(929).wrapping_add(3)));
    let r = le16_plane((0..w * h).map(|i| (i as u16).wrapping_mul(1597) & mask12));
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: g.clone(),
            },
            VideoPlane {
                stride: w * 2,
                data: b.clone(),
            },
            VideoPlane {
                stride: w * 2,
                data: r.clone(),
            },
        ],
    }
    .with_significant_bits(vec![16, 16, 12]);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrp16Le, w as u32, h as u32),
        PixelFormat::Rgb48Le,
        &ConvertOptions::default(),
    )
    .expect("gbrp16 + record → rgb48");
    for i in 0..w * h {
        assert_eq!(
            rd16(&out.planes[0].data, i * 3),
            widen(rd16(&r, i) as u32, 12, 16) as u16,
            "R word {i} must be the 12 → 16 widen"
        );
        assert_eq!(rd16(&out.planes[0].data, i * 3 + 1), rd16(&g, i), "G {i}");
        assert_eq!(rd16(&out.planes[0].data, i * 3 + 2), rd16(&b, i), "B {i}");
    }
    assert!(out.significant_bits().is_none());
}

/// Byte-sample surfaces validate the record against their 8-bit
/// nominal depth: 9 significant bits on `Gbrp8` is impossible and
/// rejects, while a legal sub-8 record widens the marked plane.
#[test]
fn record_on_gbrp8_validates_and_normalises() {
    let (w, h) = (4usize, 2usize);
    let n = w * h;
    let mk = |mul: usize, mask: usize| -> VideoPlane {
        VideoPlane {
            stride: w,
            data: (0..n).map(|i| ((i * mul) & mask) as u8).collect(),
        }
    };
    let frame = VideoFrame {
        pts: None,
        planes: vec![mk(7, 0xFF), mk(11, 0xFF), mk(5, 0x3F)],
    };
    // 9 > nominal 8 → Error::Invalid.
    let bad = frame.clone().with_significant_bits(vec![8, 8, 9]);
    assert!(convert(
        &bad,
        FrameInfo::new(PixelFormat::Gbrp8, w as u32, h as u32),
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
    )
    .is_err());
    // R plane genuinely 6-bit → widened to 8 before the reorder.
    let marked = frame.clone().with_significant_bits(vec![8, 8, 6]);
    let out = convert(
        &marked,
        FrameInfo::new(PixelFormat::Gbrp8, w as u32, h as u32),
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
    )
    .expect("gbrp8 + record → rgb24");
    for i in 0..n {
        let r6 = frame.planes[2].data[i] as u32;
        assert_eq!(
            out.planes[0].data[i * 3],
            widen(r6, 6, 8) as u8,
            "R byte {i} must be the 6 → 8 widen"
        );
        assert_eq!(out.planes[0].data[i * 3 + 1], frame.planes[0].data[i]);
        assert_eq!(out.planes[0].data[i * 3 + 2], frame.planes[1].data[i]);
    }
}

/// The record covers the full-resolution alpha plane of the deep 4:2:0
/// Yuva trio (plane index 3): an 8-bit-marked alpha on a
/// `Yuva420P10Le` surface reaches `Rgba` as exactly the original 8-bit
/// codes (widen to 10, then take the top 8 — the identity).
#[test]
fn record_on_deep_420_yuva_alpha_plane() {
    let (w, h) = (8usize, 8usize);
    let mask10 = ((1u32 << 10) - 1) as u16;
    let (cw, ch) = (w / 2, h / 2);
    let yp =
        le16_plane((0..w * h).map(|i| (i as u16).wrapping_mul(2311).wrapping_add(17) & mask10));
    let up = le16_plane((0..cw * ch).map(|i| (i as u16).wrapping_mul(929) & mask10));
    let vp = le16_plane((0..cw * ch).map(|i| (i as u16).wrapping_mul(1597) & mask10));
    let a8: Vec<u16> = (0..w * h)
        .map(|i| (i as u16).wrapping_mul(613) & 0xFF)
        .collect();
    let ap = le16_plane(a8.iter().copied());
    let src = VideoFrame {
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
            VideoPlane {
                stride: w * 2,
                data: ap,
            },
        ],
    }
    .with_significant_bits(vec![10, 10, 10, 8]);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva420P10Le, w as u32, h as u32),
        PixelFormat::Rgba,
        &ConvertOptions::default(),
    )
    .expect("yuva420p10 + record → rgba");
    for (i, &a) in a8.iter().enumerate() {
        assert_eq!(out.planes[0].data[i * 4 + 3], a as u8, "alpha {i}");
    }
    assert!(out.significant_bits().is_none());
}

/// `Ya16Le` (core 0.1.34) is a single interleaved plane, so one record
/// byte covers BOTH the luma and alpha words: a 12-bit-marked frame
/// converts exactly like the materialised 12 → 16 widen, and invalid
/// records (0, above nominal) reject.
#[test]
fn record_on_ya16le_interleaved_plane() {
    let (w, h) = (6usize, 2usize);
    let n = w * h;
    let mask12 = ((1u32 << 12) - 1) as u16;
    let words: Vec<u16> = (0..n * 2)
        .map(|i| (i as u16).wrapping_mul(2741).wrapping_add(9) & mask12)
        .collect();
    let raw = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data: le16_plane(words.iter().copied()),
        }],
    };
    let marked = raw.clone().with_significant_bits(vec![12]);
    let materialised = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data: le16_plane(words.iter().map(|&v| widen(v as u32, 12, 16) as u16)),
        }],
    };
    let info = FrameInfo::new(PixelFormat::Ya16Le, w as u32, h as u32);
    for dst in [
        PixelFormat::Ya8,
        PixelFormat::Gray16Le,
        PixelFormat::Rgba64Le,
        PixelFormat::Rgba,
    ] {
        let a = convert(&marked, info, dst, &ConvertOptions::default()).expect("marked");
        let b =
            convert(&materialised, info, dst, &ConvertOptions::default()).expect("materialised");
        assert_eq!(a.planes.len(), b.planes.len(), "{dst:?}");
        for p in 0..a.planes.len() {
            assert_eq!(a.planes[p].data, b.planes[p].data, "{dst:?} plane {p}");
        }
        assert!(a.significant_bits().is_none());
    }
    // Invalid records reject.
    for bad in [vec![0u8], vec![17u8]] {
        let f = raw.clone().with_significant_bits(bad);
        assert!(convert(&f, info, PixelFormat::Ya8, &ConvertOptions::default()).is_err());
    }
}

/// `Gbrap8` (core 0.1.34): the record covers the byte alpha plane
/// (index 3) — a 6-bit-marked alpha reaches `Rgba` as the 6 → 8 widen
/// while the colour planes pass through untouched; above-nominal
/// records reject.
#[test]
fn record_on_gbrap8_alpha_plane() {
    let (w, h) = (4usize, 2usize);
    let n = w * h;
    let mk = |mul: usize, mask: usize| -> VideoPlane {
        VideoPlane {
            stride: w,
            data: (0..n).map(|i| ((i * mul) & mask) as u8).collect(),
        }
    };
    let frame = VideoFrame {
        pts: None,
        planes: vec![mk(7, 0xFF), mk(11, 0xFF), mk(5, 0xFF), mk(13, 0x3F)],
    };
    let marked = frame.clone().with_significant_bits(vec![8, 8, 8, 6]);
    let info = FrameInfo::new(PixelFormat::Gbrap8, w as u32, h as u32);
    let out = convert(&marked, info, PixelFormat::Rgba, &ConvertOptions::default())
        .expect("gbrap8 + record → rgba");
    for i in 0..n {
        assert_eq!(out.planes[0].data[i * 4], frame.planes[2].data[i], "R {i}");
        assert_eq!(
            out.planes[0].data[i * 4 + 1],
            frame.planes[0].data[i],
            "G {i}"
        );
        assert_eq!(
            out.planes[0].data[i * 4 + 2],
            frame.planes[1].data[i],
            "B {i}"
        );
        assert_eq!(
            out.planes[0].data[i * 4 + 3],
            widen(frame.planes[3].data[i] as u32, 6, 8) as u8,
            "alpha byte {i} must be the 6 → 8 widen"
        );
    }
    assert!(out.significant_bits().is_none());
    let bad = frame.clone().with_significant_bits(vec![8, 8, 8, 9]);
    assert!(convert(&bad, info, PixelFormat::Rgba, &ConvertOptions::default()).is_err());
}

/// `CmykInverted` (core 0.1.34): a sub-8 record on the packed plane
/// widens every component byte before the decode — identical to the
/// materialised frame — and the complement row sees nominal bytes.
#[test]
fn record_on_cmyk_inverted_packed_plane() {
    let (w, h) = (4usize, 2usize);
    let n = w * h;
    let raw: Vec<u8> = (0..n * 4).map(|i| ((i * 19 + 2) & 0x0F) as u8).collect();
    let frame = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data: raw.clone(),
        }],
    };
    let marked = frame.clone().with_significant_bits(vec![4]);
    let materialised = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data: raw.iter().map(|&v| widen(v as u32, 4, 8) as u8).collect(),
        }],
    };
    let info = FrameInfo::new(PixelFormat::CmykInverted, w as u32, h as u32);
    for dst in [PixelFormat::Rgb24, PixelFormat::Cmyk, PixelFormat::Gray8] {
        let a = convert(&marked, info, dst, &ConvertOptions::default()).expect("marked");
        let b =
            convert(&materialised, info, dst, &ConvertOptions::default()).expect("materialised");
        assert_eq!(a.planes[0].data, b.planes[0].data, "{dst:?}");
        assert!(a.significant_bits().is_none());
    }
    let bad = frame.clone().with_significant_bits(vec![9]);
    assert!(convert(&bad, info, PixelFormat::Rgb24, &ConvertOptions::default()).is_err());
}
