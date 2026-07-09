//! Cross-depth storage-width conversions through `convert()`:
//! planar YUV 10 ↔ 12 bit and the deep-grayscale ladder
//! (`Gray8` ↔ `Gray10Le` / `Gray12Le` ↔ `Gray16Le`).
//!
//! All of these are pure per-plane storage rescales: widening places the
//! value in the high bits and replicates its MSBs into the new low bits
//! (peak maps to peak), narrowing truncates the low bits — the exact
//! inverse, so widen → narrow round-trips are lossless.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

fn le16_plane(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.push((*v & 0xFF) as u8);
        out.push((*v >> 8) as u8);
    }
    out
}

fn read_le16(data: &[u8]) -> Vec<u16> {
    data.chunks(2)
        .map(|c| (c[0] as u16) | ((c[1] as u16) << 8))
        .collect()
}

fn yuv16_frame(w: usize, h: usize, wsub: usize, hsub: usize, seed: u16, bits: u32) -> VideoFrame {
    let cw = w / wsub;
    let ch = h / hsub;
    let mask = ((1u32 << bits) - 1) as u16;
    let plane = |n: usize, off: u16| -> Vec<u8> {
        let vals: Vec<u16> = (0..n)
            .map(|i| ((i as u16).wrapping_mul(37).wrapping_add(seed + off)) & mask)
            .collect();
        le16_plane(&vals)
    };
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: plane(w * h, 0),
            },
            VideoPlane {
                stride: cw * 2,
                data: plane(cw * ch, 101),
            },
            VideoPlane {
                stride: cw * 2,
                data: plane(cw * ch, 202),
            },
        ],
    }
}

const CROSS_DEPTH: [(PixelFormat, PixelFormat, usize, usize); 3] = [
    (PixelFormat::Yuv420P10Le, PixelFormat::Yuv420P12Le, 2, 2),
    (PixelFormat::Yuv422P10Le, PixelFormat::Yuv422P12Le, 2, 1),
    (PixelFormat::Yuv444P10Le, PixelFormat::Yuv444P12Le, 1, 1),
];

/// 10 → 12 → 10 is bit-exact for every plane, and peak maps to peak.
#[test]
fn yuv_10_12_roundtrip_exact() {
    for (f10, f12, wsub, hsub) in CROSS_DEPTH {
        let w = 8usize;
        let h = 8usize;
        let src = yuv16_frame(w, h, wsub, hsub, 5, 10);
        let o = ConvertOptions::default();
        let up = convert(&src, FrameInfo::new(f10, w as u32, h as u32), f12, &o).expect("10→12");
        let back = convert(&up, FrameInfo::new(f12, w as u32, h as u32), f10, &o).expect("12→10");
        for p in 0..3 {
            assert_eq!(
                src.planes[p].data, back.planes[p].data,
                "{f10:?} plane {p} round-trip"
            );
        }
    }
}

/// The 10 → 12 widening is the documented shift + MSB replication:
/// v12 = (v10 << 2) | (v10 >> 8). Spot-check rails and a mid code.
#[test]
fn yuv_10_to_12_value_map() {
    let (f10, f12, wsub, hsub) = CROSS_DEPTH[2];
    let w = 4usize;
    let h = 1usize;
    let vals10: [u16; 4] = [0, 0x3FF, 512, 100];
    // v12 = (v10 << 2) | (v10 >> 8): 512 → 2049, 100 → 400 (100 >> 8 = 0).
    let want12: [u16; 4] = [0, 0xFFF, (512 << 2) | (512 >> 8), 100 << 2];
    let cw = w / wsub;
    let ch = h / hsub;
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: le16_plane(&vals10),
            },
            VideoPlane {
                stride: cw * 2,
                data: le16_plane(&vals10[..cw * ch]),
            },
            VideoPlane {
                stride: cw * 2,
                data: le16_plane(&vals10[..cw * ch]),
            },
        ],
    };
    let out = convert(
        &src,
        FrameInfo::new(f10, w as u32, h as u32),
        f12,
        &ConvertOptions::default(),
    )
    .expect("10→12");
    assert_eq!(read_le16(&out.planes[0].data), want12);
}

/// 12 → 10 truncates the two low bits (12 → 10 → 12 differs only there).
#[test]
fn yuv_12_to_10_truncates() {
    let (f10, f12, _, _) = CROSS_DEPTH[2];
    let w = 4usize;
    let h = 1usize;
    let vals12: [u16; 4] = [0xFFF, 0xFFE, 0x001, 0x800];
    let want10: [u16; 4] = [0x3FF, 0x3FF, 0x000, 0x200];
    let plane = le16_plane(&vals12);
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: plane.clone(),
            },
            VideoPlane {
                stride: w * 2,
                data: plane.clone(),
            },
            VideoPlane {
                stride: w * 2,
                data: plane,
            },
        ],
    };
    let out = convert(
        &src,
        FrameInfo::new(f12, w as u32, h as u32),
        f10,
        &ConvertOptions::default(),
    )
    .expect("12→10");
    assert_eq!(read_le16(&out.planes[0].data), want10);
}

/// Odd dimensions on subsampled cross-depth layouts reject up front.
#[test]
fn yuv_cross_depth_odd_dims_reject() {
    let src = yuv16_frame(4, 4, 2, 2, 1, 10);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv420P10Le, 3, 3),
        PixelFormat::Yuv420P12Le,
        &ConvertOptions::default(),
    );
    assert!(out.is_err());
}

// ---------------------------------------------------------------------
// Deep grayscale ladder.

fn gray16_frame(w: usize, h: usize, vals: &[u16]) -> VideoFrame {
    assert_eq!(vals.len(), w * h);
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 2,
            data: le16_plane(vals),
        }],
    }
}

/// Every 8-bit value round-trips exactly through Gray10Le and Gray12Le.
#[test]
fn gray8_deep_roundtrip_exact() {
    let w = 16usize;
    let h = 16usize;
    let g: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w,
            data: g.clone(),
        }],
    };
    let o = ConvertOptions::default();
    for deep in [PixelFormat::Gray10Le, PixelFormat::Gray12Le] {
        let up = convert(
            &src,
            FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32),
            deep,
            &o,
        )
        .expect("8 → deep");
        // 0xFF must reach the deep peak (MSB replication, not zero-fill).
        let vals = read_le16(&up.planes[0].data);
        let bits = if deep == PixelFormat::Gray10Le {
            10
        } else {
            12
        };
        assert_eq!(vals[255], (1u16 << bits) - 1, "{deep:?} peak");
        let back = convert(
            &up,
            FrameInfo::new(deep, w as u32, h as u32),
            PixelFormat::Gray8,
            &o,
        )
        .expect("deep → 8");
        assert_eq!(back.planes[0].data, g, "{deep:?} round-trip");
    }
}

/// Gray10 → Gray16 → Gray10 and Gray12 → Gray16 → Gray12 are exact,
/// and 10 ↔ 12 behaves like the YUV cross-depth path.
#[test]
fn gray_deep_ladder_roundtrips() {
    let w = 32usize;
    let h = 4usize;
    let o = ConvertOptions::default();
    for (fmt, bits) in [(PixelFormat::Gray10Le, 10u32), (PixelFormat::Gray12Le, 12)] {
        let mask = ((1u32 << bits) - 1) as u16;
        let vals: Vec<u16> = (0..w * h)
            .map(|i| ((i as u16).wrapping_mul(53).wrapping_add(7)) & mask)
            .collect();
        let src = gray16_frame(w, h, &vals);
        let info = FrameInfo::new(fmt, w as u32, h as u32);
        let wide = convert(&src, info, PixelFormat::Gray16Le, &o).expect("→ 16");
        let back = convert(
            &wide,
            FrameInfo::new(PixelFormat::Gray16Le, w as u32, h as u32),
            fmt,
            &o,
        )
        .expect("16 →");
        assert_eq!(back.planes[0].data, src.planes[0].data, "{fmt:?} via 16");
    }
    // 10 → 12 → 10 exact.
    let vals: Vec<u16> = (0..w * h).map(|i| (i as u16 * 31 + 3) & 0x3FF).collect();
    let src = gray16_frame(w, h, &vals);
    let up = convert(
        &src,
        FrameInfo::new(PixelFormat::Gray10Le, w as u32, h as u32),
        PixelFormat::Gray12Le,
        &o,
    )
    .expect("10 → 12");
    let back = convert(
        &up,
        FrameInfo::new(PixelFormat::Gray12Le, w as u32, h as u32),
        PixelFormat::Gray10Le,
        &o,
    )
    .expect("12 → 10");
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

/// Gray16Le → Gray10/12Le keeps the top bits — consistent with the
/// long-standing Gray16Le → Gray8 high-byte behaviour.
#[test]
fn gray16_narrowing_keeps_top_bits() {
    let w = 4usize;
    let h = 1usize;
    let vals: [u16; 4] = [0xFFFF, 0x8000, 0x00FF, 0x0000];
    let src = gray16_frame(w, h, &vals);
    let o = ConvertOptions::default();
    let out10 = convert(
        &src,
        FrameInfo::new(PixelFormat::Gray16Le, w as u32, h as u32),
        PixelFormat::Gray10Le,
        &o,
    )
    .expect("16 → 10");
    assert_eq!(
        read_le16(&out10.planes[0].data),
        vec![0x3FF, 0x200, 0x0003, 0x0000]
    );
    let out12 = convert(
        &src,
        FrameInfo::new(PixelFormat::Gray16Le, w as u32, h as u32),
        PixelFormat::Gray12Le,
        &o,
    )
    .expect("16 → 12");
    assert_eq!(
        read_le16(&out12.planes[0].data),
        vec![0xFFF, 0x800, 0x000F, 0x0000]
    );
}
