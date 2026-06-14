//! Bit-depth conversion between high-precision planar YUV (10/12-bit,
//! 16-bit LE storage) and the 8-bit planar siblings.
//!
//! The conversion is pure per-plane bit-depth scaling: subsampling layout
//! and colour are untouched. Up-conversion places the 8-bit value in the
//! high bits and replicates its MSBs into the low slack; down-conversion
//! drops those low bits (truncation). Because the slack carries the
//! value's own high bits, the 8 → high → 8 round-trip is exact, which is
//! the core invariant exercised here.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

/// Build an 8-bit planar YUV frame for the given subsampling.
fn synth_yuv8(fmt: PixelFormat, w: usize, h: usize, wsub: usize, hsub: usize) -> VideoFrame {
    let cw = w / wsub;
    let ch = h / hsub;
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    for y in 0..h {
        for x in 0..w {
            yp[y * w + x] = (x * 13 + y * 7) as u8;
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            up[y * cw + x] = (x * 5 + y * 3) as u8;
            vp[y * cw + x] = (x * 29 + y * 17) as u8;
        }
    }
    let _ = fmt;
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w,
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
    }
}

const CASES: &[(PixelFormat, PixelFormat, PixelFormat, usize, usize)] = &[
    // (8-bit, 10-bit hi, 12-bit hi, wsub, hsub)
    (
        PixelFormat::Yuv420P,
        PixelFormat::Yuv420P10Le,
        PixelFormat::Yuv420P12Le,
        2,
        2,
    ),
    (
        PixelFormat::Yuv422P,
        PixelFormat::Yuv422P10Le,
        PixelFormat::Yuv422P12Le,
        2,
        1,
    ),
    (
        PixelFormat::Yuv444P,
        PixelFormat::Yuv444P10Le,
        PixelFormat::Yuv444P12Le,
        1,
        1,
    ),
];

#[test]
fn depth_up_down_roundtrip_is_exact() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(eight, ten, twelve, wsub, hsub) in CASES {
        for hi in [ten, twelve] {
            let src = synth_yuv8(eight, w, h, wsub, hsub);
            let src_info = FrameInfo::new(eight, w as u32, h as u32);
            let up = convert(&src, src_info, hi, &opts).expect("depth up");
            let up_info = FrameInfo::new(hi, w as u32, h as u32);
            let back = convert(&up, up_info, eight, &opts).expect("depth down");
            assert_eq!(src.planes.len(), back.planes.len());
            for p in 0..src.planes.len() {
                assert_eq!(
                    src.planes[p].data, back.planes[p].data,
                    "plane {p} round-trip {eight:?} -> {hi:?} -> {eight:?}"
                );
            }
        }
    }
}

#[test]
fn depth_up_word_layout_10bit() {
    // 8-bit value 0xAB = 0b1010_1011. 10-bit target: shift left 2
    // (0b1010_1011_00) and replicate the top 2 MSBs (0b10) into the freed
    // low bits, giving 0b1010_1011_10 = 0x2AE.
    let opts = ConvertOptions::default();
    let (w, h) = (2usize, 2usize);
    let mut src = synth_yuv8(PixelFormat::Yuv444P, w, h, 1, 1);
    for plane in src.planes.iter_mut() {
        for b in plane.data.iter_mut() {
            *b = 0xAB;
        }
    }
    let src_info = FrameInfo::new(PixelFormat::Yuv444P, w as u32, h as u32);
    let up = convert(&src, src_info, PixelFormat::Yuv444P10Le, &opts).expect("up");
    for plane in &up.planes {
        for word in plane.data.chunks_exact(2) {
            let v = u16::from_le_bytes([word[0], word[1]]);
            assert_eq!(v, 0x2AE, "10-bit packed word for input 0xAB");
        }
    }
}

#[test]
fn depth_up_word_layout_12bit() {
    // 8-bit value 0xAB = 0b1010_1011. 12-bit target: shift left 4 and
    // replicate the top 4 MSBs (0b1010) into the low nibble, giving
    // 0b1010_1011_1010 = 0xABA.
    let opts = ConvertOptions::default();
    let (w, h) = (2usize, 2usize);
    let mut src = synth_yuv8(PixelFormat::Yuv444P, w, h, 1, 1);
    for plane in src.planes.iter_mut() {
        for b in plane.data.iter_mut() {
            *b = 0xAB;
        }
    }
    let src_info = FrameInfo::new(PixelFormat::Yuv444P, w as u32, h as u32);
    let up = convert(&src, src_info, PixelFormat::Yuv444P12Le, &opts).expect("up");
    for plane in &up.planes {
        for word in plane.data.chunks_exact(2) {
            let v = u16::from_le_bytes([word[0], word[1]]);
            assert_eq!(v, 0xABA, "12-bit packed word for input 0xAB");
        }
    }
}

#[test]
fn depth_down_truncates_10bit() {
    // Construct a 10-bit source by hand and verify the >> 2 truncation
    // (top 8 of the 10 significant bits, low bits dropped):
    //   0x000 -> 0
    //   0x003 -> 0   (3 >> 2 = 0)
    //   0x004 -> 1   (4 >> 2 = 1)
    //   0x3FB -> 254 (1019 >> 2 = 254)
    //   0x3FF -> 255 (1023 >> 2 = 255, reaches the 8-bit ceiling)
    let opts = ConvertOptions::default();
    let inputs: [(u16, u8); 5] = [
        (0x000, 0),
        (0x003, 0),
        (0x004, 1),
        (0x3FB, 254),
        (0x3FF, 255),
    ];
    let w = inputs.len();
    let h = 1usize;
    let mut yp = Vec::new();
    for &(v, _) in &inputs {
        yp.extend_from_slice(&v.to_le_bytes());
    }
    // Chroma: 4:4:4 so full resolution; fill with a mid value.
    let up: Vec<u8> = (0..w).flat_map(|_| 0x200u16.to_le_bytes()).collect();
    let vp = up.clone();
    let frame = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: yp,
            },
            VideoPlane {
                stride: w * 2,
                data: up,
            },
            VideoPlane {
                stride: w * 2,
                data: vp,
            },
        ],
    };
    let info = FrameInfo::new(PixelFormat::Yuv444P10Le, w as u32, h as u32);
    let down = convert(&frame, info, PixelFormat::Yuv444P, &opts).expect("down");
    for (i, &(_, expect)) in inputs.iter().enumerate() {
        assert_eq!(down.planes[0].data[i], expect, "luma sample {i}");
    }
    // 0x200 (512) >> 2 = 128.
    for &c in &down.planes[1].data {
        assert_eq!(c, 128, "chroma mid value 0x200 -> 128");
    }
}

#[test]
fn depth_conversion_rejects_indivisible_dims() {
    // 4:2:0 needs even width and height. An odd width must error rather
    // than panic or silently truncate.
    let opts = ConvertOptions::default();
    let frame = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: 5,
                data: vec![0u8; 5 * 4],
            },
            VideoPlane {
                stride: 2,
                data: vec![0u8; 2 * 2],
            },
            VideoPlane {
                stride: 2,
                data: vec![0u8; 2 * 2],
            },
        ],
    };
    let info = FrameInfo::new(PixelFormat::Yuv420P, 5, 4);
    assert!(convert(&frame, info, PixelFormat::Yuv420P10Le, &opts).is_err());
}

#[test]
fn depth_up_emits_correct_plane_sizes() {
    // The high-bit destination must be exactly twice the byte size of the
    // 8-bit source per plane (16-bit storage), with subsampling intact.
    let opts = ConvertOptions::default();
    let (w, h) = (8usize, 8usize);
    let src = synth_yuv8(PixelFormat::Yuv420P, w, h, 2, 2);
    let info = FrameInfo::new(PixelFormat::Yuv420P, w as u32, h as u32);
    let up = convert(&src, info, PixelFormat::Yuv420P10Le, &opts).expect("up");
    assert_eq!(up.planes[0].data.len(), w * h * 2);
    assert_eq!(up.planes[1].data.len(), (w / 2) * (h / 2) * 2);
    assert_eq!(up.planes[2].data.len(), (w / 2) * (h / 2) * 2);
}
