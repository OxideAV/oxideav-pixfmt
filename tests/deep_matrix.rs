//! Full-precision deep-matrix suite: the 16-bit planar YUV(A) tier ↔
//! packed deep RGB rows must run the k-coefficient construction at
//! 16-bit precision (Q30 fixed point) — never narrowing to 8 bits —
//! and match an independent f64 model built straight from the same
//! construction to ±1 LSB at 16-bit scale.
//!
//! Range convention under test (limited): the n-bit digital
//! representation scales the 8-bit offsets and spans by 2^(n−8) — at
//! n = 16 luma is 4096 + [0, 56064] and chroma is centred on the exact
//! achromatic code 32768 with span 57344. Full-scale white therefore
//! encodes to Y = 60160 (235 × 256) exactly, black to 4096, and
//! r = g = b content to chroma exactly 32768.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, supports_direct, yuv::rgb48_pixel_to_yuv16, yuv::yuv16_pixel_to_rgb48, yuv::YuvMatrix,
    ColorSpace, ConvertOptions, FrameInfo,
};

fn rd16(buf: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]])
}

fn le16_plane(vals: impl Iterator<Item = u16>) -> Vec<u8> {
    vals.flat_map(|v| v.to_le_bytes()).collect()
}

const SPACES: [ColorSpace; 6] = [
    ColorSpace::Bt601Limited,
    ColorSpace::Bt601Full,
    ColorSpace::Bt709Limited,
    ColorSpace::Bt709Full,
    ColorSpace::Bt2020Limited,
    ColorSpace::Bt2020Full,
];

/// Independent f64 model of the deep encode, straight from the
/// k-coefficient construction (kr, kb pulled from the public
/// `YuvMatrix` constants).
fn model_encode(r: u16, g: u16, b: u16, m: YuvMatrix) -> (f64, f64, f64) {
    let kr = m.kr as f64;
    let kb = m.kb as f64;
    let kg = 1.0 - kr - kb;
    let (rf, gf, bf) = (r as f64, g as f64, b as f64);
    let y_full = kr * rf + kg * gf + kb * bf;
    let (ys, cs, y_off) = if m.limited {
        (56064.0 / 65535.0, 57344.0 / 65535.0, 4096.0)
    } else {
        (1.0, 1.0, 0.0)
    };
    let y = y_off + ys * y_full;
    let cb = 32768.0 + cs * (bf - y_full) / (2.0 * (1.0 - kb));
    let cr = 32768.0 + cs * (rf - y_full) / (2.0 * (1.0 - kr));
    (y, cb, cr)
}

/// Independent f64 model of the deep decode.
fn model_decode(y: u16, cb: u16, cr: u16, m: YuvMatrix) -> (f64, f64, f64) {
    let kr = m.kr as f64;
    let kb = m.kb as f64;
    let kg = 1.0 - kr - kb;
    let (ys, cs, y_off) = if m.limited {
        (65535.0 / 56064.0, 65535.0 / 57344.0, 4096.0)
    } else {
        (1.0, 1.0, 0.0)
    };
    let yv = (y as f64 - y_off) * ys;
    let cbv = cb as f64 - 32768.0;
    let crv = cr as f64 - 32768.0;
    let r = yv + 2.0 * (1.0 - kr) * cs * crv;
    let b = yv + 2.0 * (1.0 - kb) * cs * cbv;
    let g = yv - (2.0 * kr * (1.0 - kr) / kg) * cs * crv - (2.0 * kb * (1.0 - kb) / kg) * cs * cbv;
    (r, g, b)
}

fn assert_close(got: u16, want: f64, tag: &str) {
    let want_clamped = want.clamp(0.0, 65535.0);
    assert!(
        (got as f64 - want_clamped).abs() <= 1.0 + 1e-9,
        "{tag}: got {got}, model {want_clamped:.3}"
    );
}

/// The Q30 per-pixel kernels sit within ±1 LSB of the f64 model at
/// 16-bit precision across a deterministic sample sweep, all six
/// matrix variants, both directions.
#[test]
fn deep_kernels_match_f64_model() {
    for cs in SPACES {
        let m = YuvMatrix::from_color_space(cs);
        let mut v = 0x2468u32;
        for _ in 0..2000 {
            v = v.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (v >> 16) as u16;
            let g = (v & 0xFFFF) as u16;
            v = v.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (v >> 16) as u16;
            let (y, cb, cr) = rgb48_pixel_to_yuv16(r, g, b, m);
            let (my, mcb, mcr) = model_encode(r, g, b, m);
            assert_close(y, my, &format!("{cs:?} encode Y({r},{g},{b})"));
            assert_close(cb, mcb, &format!("{cs:?} encode Cb({r},{g},{b})"));
            assert_close(cr, mcr, &format!("{cs:?} encode Cr({r},{g},{b})"));
            // Decode sweep over the same pseudo-random codes.
            let (dr, dg, db) = yuv16_pixel_to_rgb48(y, cb, cr, m);
            let (mr, mg, mb) = model_decode(y, cb, cr, m);
            assert_close(dr, mr, &format!("{cs:?} decode R({y},{cb},{cr})"));
            assert_close(dg, mg, &format!("{cs:?} decode G({y},{cb},{cr})"));
            assert_close(db, mb, &format!("{cs:?} decode B({y},{cb},{cr})"));
        }
    }
}

/// Exact anchors from the n-bit digital representation: full-scale
/// white → (60160, 32768, 32768) limited, black → (4096, 32768,
/// 32768); full-range keeps white at 65535 and both conventions pin
/// r = g = b content to the exact achromatic chroma code.
#[test]
fn deep_anchor_codes() {
    for cs in [
        ColorSpace::Bt601Limited,
        ColorSpace::Bt709Limited,
        ColorSpace::Bt2020Limited,
    ] {
        let m = YuvMatrix::from_color_space(cs);
        assert_eq!(
            rgb48_pixel_to_yuv16(65535, 65535, 65535, m),
            (60160, 32768, 32768),
            "{cs:?} white"
        );
        assert_eq!(
            rgb48_pixel_to_yuv16(0, 0, 0, m),
            (4096, 32768, 32768),
            "{cs:?} black"
        );
        // Achromatic mid-grey: chroma pinned to 32768 exactly.
        let (_, cb, cr) = rgb48_pixel_to_yuv16(30000, 30000, 30000, m);
        assert_eq!((cb, cr), (32768, 32768), "{cs:?} grey chroma");
    }
    for cs in [
        ColorSpace::Bt601Full,
        ColorSpace::Bt709Full,
        ColorSpace::Bt2020Full,
    ] {
        let m = YuvMatrix::from_color_space(cs);
        assert_eq!(
            rgb48_pixel_to_yuv16(65535, 65535, 65535, m),
            (65535, 32768, 32768),
            "{cs:?} white"
        );
        assert_eq!(
            rgb48_pixel_to_yuv16(0, 0, 0, m),
            (0, 32768, 32768),
            "{cs:?} black"
        );
    }
}

/// Rgb48Le → Yuv444P16Le → Rgb48Le round-trips within ±2 LSB at
/// 16-bit scale (each direction rounds once; the limited-range
/// quantisation stretches decode error by 65535/56064).
#[test]
fn deep_444_roundtrip_within_2lsb_16bit() {
    let (w, h) = (16usize, 8usize);
    let n = w * h;
    let src_words: Vec<u16> = (0..n * 3)
        .map(|i| (i as u32).wrapping_mul(40507).wrapping_add(129) as u16)
        .collect();
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 6,
            data: le16_plane(src_words.iter().copied()),
        }],
    };
    for cs in SPACES {
        let opts = ConvertOptions {
            color_space: cs,
            ..Default::default()
        };
        let yuv = convert(
            &src,
            FrameInfo::new(PixelFormat::Rgb48Le, w as u32, h as u32),
            PixelFormat::Yuv444P16Le,
            &opts,
        )
        .expect("rgb48 → yuv444p16");
        assert_eq!(yuv.planes.len(), 3);
        let back = convert(
            &yuv,
            FrameInfo::new(PixelFormat::Yuv444P16Le, w as u32, h as u32),
            PixelFormat::Rgb48Le,
            &opts,
        )
        .expect("yuv444p16 → rgb48");
        let mut max_err = 0i32;
        for i in 0..n * 3 {
            let a = rd16(&src.planes[0].data, i) as i32;
            let b = rd16(&back.planes[0].data, i) as i32;
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err <= 2, "{cs:?}: max round-trip error {max_err}");
    }
}

/// The deep rows are direct (no staging), and the alpha word rides
/// bit-exact both ways on the Yuva16 pairs.
#[test]
fn deep_rows_direct_and_alpha_verbatim() {
    for (a, b) in [
        (PixelFormat::Yuv444P16Le, PixelFormat::Rgb48Le),
        (PixelFormat::Yuv422P16Le, PixelFormat::Rgb48Le),
        (PixelFormat::Yuv420P16Le, PixelFormat::Rgb48Le),
        (PixelFormat::Yuva444P16Le, PixelFormat::Rgba64Le),
        (PixelFormat::Yuva422P16Le, PixelFormat::Rgba64Le),
        (PixelFormat::Yuva420P16Le, PixelFormat::Rgba64Le),
    ] {
        assert!(supports_direct(a, b), "{a:?} → {b:?}");
        assert!(supports_direct(b, a), "{b:?} → {a:?}");
    }
    let (w, h) = (8usize, 4usize);
    let n = w * h;
    let y = le16_plane((0..n).map(|i| (i as u16).wrapping_mul(2311).wrapping_add(8000)));
    let u = le16_plane((0..n).map(|i| (i as u16).wrapping_mul(929).wrapping_add(20000)));
    let v = le16_plane((0..n).map(|i| (i as u16).wrapping_mul(1597).wrapping_add(30000)));
    let a = le16_plane((0..n).map(|i| (i as u16).wrapping_mul(40961).wrapping_add(3)));
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y,
            },
            VideoPlane {
                stride: w * 2,
                data: u,
            },
            VideoPlane {
                stride: w * 2,
                data: v,
            },
            VideoPlane {
                stride: w * 2,
                data: a.clone(),
            },
        ],
    };
    let opts = ConvertOptions::default();
    let packed = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva444P16Le, w as u32, h as u32),
        PixelFormat::Rgba64Le,
        &opts,
    )
    .expect("yuva444p16 → rgba64");
    for i in 0..n {
        assert_eq!(
            rd16(&packed.planes[0].data, i * 4 + 3),
            rd16(&a, i),
            "alpha word {i}"
        );
    }
    let back = convert(
        &packed,
        FrameInfo::new(PixelFormat::Rgba64Le, w as u32, h as u32),
        PixelFormat::Yuva444P16Le,
        &opts,
    )
    .expect("rgba64 → yuva444p16");
    assert_eq!(back.planes.len(), 4);
    assert_eq!(back.planes[3].data, a, "alpha plane must ride verbatim");
}

/// Subsampled deep rows must equal the hand-staged 16-bit route
/// (chroma resample at 16-bit, then the 4:4:4 deep matrix) — proof no
/// 8-bit hop hides on the path.
#[test]
fn deep_subsampled_equals_16bit_staging() {
    let (w, h) = (8usize, 8usize);
    let n = w * h;
    let cn = n / 4;
    let y = le16_plane((0..n).map(|i| (i as u16).wrapping_mul(2311).wrapping_add(8000)));
    let u = le16_plane((0..cn).map(|i| (i as u16).wrapping_mul(929).wrapping_add(20000)));
    let v = le16_plane((0..cn).map(|i| (i as u16).wrapping_mul(1597).wrapping_add(30000)));
    let src = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y,
            },
            VideoPlane { stride: w, data: u },
            VideoPlane { stride: w, data: v },
        ],
    };
    let opts = ConvertOptions::default();
    let direct = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv420P16Le, w as u32, h as u32),
        PixelFormat::Rgb48Le,
        &opts,
    )
    .expect("direct");
    let mid = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuv420P16Le, w as u32, h as u32),
        PixelFormat::Yuv444P16Le,
        &opts,
    )
    .expect("leg 1");
    let staged = convert(
        &mid,
        FrameInfo::new(PixelFormat::Yuv444P16Le, w as u32, h as u32),
        PixelFormat::Rgb48Le,
        &opts,
    )
    .expect("leg 2");
    assert_eq!(direct.planes[0].data, staged.planes[0].data);
}

/// The 10-bit family now reaches Rgb48Le at full precision through the
/// exact widen to the 16-bit tier: adjacent 10-bit luma codes must
/// produce distinct 16-bit RGB outputs (they collapsed to one value
/// when the old route quantised through an 8-bit pivot).
#[test]
fn deep_staging_upgrade_preserves_10bit_distinctions() {
    let (w, h) = (4usize, 4usize);
    let n = w * h;
    let mk = |luma: u16| -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w * 2,
                    data: le16_plane((0..n).map(|_| luma)),
                },
                VideoPlane {
                    stride: w * 2,
                    data: le16_plane((0..n).map(|_| 512)),
                },
                VideoPlane {
                    stride: w * 2,
                    data: le16_plane((0..n).map(|_| 512)),
                },
            ],
        }
    };
    let opts = ConvertOptions::default();
    let info = FrameInfo::new(PixelFormat::Yuv444P10Le, w as u32, h as u32);
    let a = convert(&mk(500), info, PixelFormat::Rgb48Le, &opts).expect("a");
    let b = convert(&mk(501), info, PixelFormat::Rgb48Le, &opts).expect("b");
    assert_ne!(
        a.planes[0].data, b.planes[0].data,
        "adjacent 10-bit luma codes must stay distinguishable in Rgb48Le"
    );
}
