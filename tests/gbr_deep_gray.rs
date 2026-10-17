//! Planar GBR(A) ↔ the deep grey words are direct (round 476): the
//! luma projection at 16 bits over ladder-widened planes, and the
//! grey word rescaled and broadcast. Also pins the alpha-less deep
//! pivot preference: when neither endpoint carries alpha, a staged
//! deep route pivots through `Rgb48Le` / `Yuv*P16Le`, never a
//! synthesised-alpha member, with the same bytes as before.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, convert_with, staged_pivot, supports_direct, ConvertContext, ConvertOptions,
    FormatInfo, FrameInfo,
};

fn le16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn rd16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn one_plane(stride: usize, data: Vec<u8>) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

fn same_planes(a: &VideoFrame, b: &VideoFrame) -> bool {
    a.planes.len() == b.planes.len()
        && a.planes
            .iter()
            .zip(b.planes.iter())
            .all(|(x, y)| x.stride == y.stride && x.data == y.data)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn u16(&mut self) -> u16 {
        (self.next() >> 48) as u16
    }
}

const GBR: &[(PixelFormat, u32, bool)] = &[
    (PixelFormat::Gbrp8, 8, false),
    (PixelFormat::Gbrap8, 8, true),
    (PixelFormat::Gbrp10Le, 10, false),
    (PixelFormat::Gbrap10Le, 10, true),
    (PixelFormat::Gbrp12Le, 12, false),
    (PixelFormat::Gbrap12Le, 12, true),
    (PixelFormat::Gbrp14Le, 14, false),
    (PixelFormat::Gbrap14Le, 14, true),
    (PixelFormat::Gbrp16Le, 16, false),
    (PixelFormat::Gbrap16Le, 16, true),
];

const DEEP_GRAY: &[(PixelFormat, u32)] = &[
    (PixelFormat::Gray10Le, 10),
    (PixelFormat::Gray12Le, 12),
    (PixelFormat::Gray16Le, 16),
];

fn gbr_frame(w: usize, h: usize, bits: u32, alpha: bool, seed: u64) -> VideoFrame {
    let n = w * h;
    let mut rng = Rng(seed | 1);
    let mask = ((1u32 << bits) - 1) as u16;
    let plane = |rng: &mut Rng| -> VideoPlane {
        if bits > 8 {
            let words: Vec<u16> = (0..n).map(|_| rng.u16() & mask).collect();
            VideoPlane {
                stride: w * 2,
                data: le16(&words),
            }
        } else {
            VideoPlane {
                stride: w,
                data: (0..n).map(|_| (rng.u16() & 0xFF) as u8).collect(),
            }
        }
    };
    let mut planes = vec![plane(&mut rng), plane(&mut rng), plane(&mut rng)];
    if alpha {
        planes.push(plane(&mut rng));
    }
    VideoFrame { pts: None, planes }
}

fn conv(src: &VideoFrame, sf: PixelFormat, w: usize, h: usize, df: PixelFormat) -> VideoFrame {
    convert(
        src,
        FrameInfo::new(sf, w as u32, h as u32),
        df,
        &ConvertOptions::default(),
    )
    .unwrap()
}

#[test]
fn gbr_and_deep_gray_are_direct() {
    for &(f, ..) in GBR {
        for &(g, _) in DEEP_GRAY {
            assert!(supports_direct(f, g), "{f:?} → {g:?}");
            assert!(supports_direct(g, f), "{g:?} → {f:?}");
        }
    }
}

/// GBR → deep grey equals the previous route (packed deep pivot + the
/// 16-bit projection) byte for byte; grey → GBR equals grey → packed
/// → GBR.
#[test]
fn gbr_deep_gray_matches_the_packed_route() {
    let (w, h) = if cfg!(miri) {
        (3usize, 2usize)
    } else {
        (13, 7)
    };
    for &(f, bits, alpha) in GBR {
        let src = gbr_frame(w, h, bits, alpha, bits as u64);
        let packed = if alpha {
            PixelFormat::Rgba64Le
        } else {
            PixelFormat::Rgb48Le
        };
        let mid = conv(&src, f, w, h, packed);
        for &(g, _) in DEEP_GRAY {
            let direct = conv(&src, f, w, h, g);
            let staged = conv(&mid, packed, w, h, g);
            assert_eq!(
                direct.planes[0].data, staged.planes[0].data,
                "{f:?} → {g:?}"
            );
            assert_eq!(direct.planes[0].stride, w * 2);
        }
    }
    for &(g, gbits) in DEEP_GRAY {
        let mask = ((1u32 << gbits) - 1) as u16;
        let mut rng = Rng(gbits as u64);
        let codes: Vec<u16> = (0..w * h).map(|_| rng.u16() & mask).collect();
        let src = one_plane(w * 2, le16(&codes));
        let mid = conv(&src, g, w, h, PixelFormat::Rgb48Le);
        for &(f, bits, alpha) in GBR {
            let direct = conv(&src, g, w, h, f);
            let staged = conv(&mid, PixelFormat::Rgb48Le, w, h, f);
            assert!(same_planes(&direct, &staged), "{g:?} → {f:?}");
            assert_eq!(direct.planes.len(), if alpha { 4 } else { 3 });
            let info = FormatInfo::of(f);
            assert_eq!(info.bit_depth as u32, bits);
            // Round trip: grey → GBR → grey is exact when the GBR depth
            // is at least the grey depth (projection of r = g = b).
            if bits >= gbits {
                let back = conv(&direct, f, w, h, g);
                assert_eq!(rd16(&back.planes[0].data), codes, "{g:?} → {f:?} → {g:?}");
            }
        }
    }
}

/// Every 16-bit code survives `Gray16Le → Gbrp16Le → Gray16Le`.
#[test]
fn gray16_gbrp16_ramp_is_exact() {
    let step = if cfg!(miri) { 257 } else { 1 };
    let mut ramp: Vec<u16> = (0..=u16::MAX).step_by(step).collect();
    let w = if step == 1 { 256 } else { 16 };
    let h = ramp.len().div_ceil(w);
    ramp.resize(w * h, 0);
    let src = one_plane(w * 2, le16(&ramp));
    for f in [PixelFormat::Gbrp16Le, PixelFormat::Gbrap16Le] {
        let gbr = conv(&src, PixelFormat::Gray16Le, w, h, f);
        for p in 0..3 {
            assert_eq!(rd16(&gbr.planes[p].data), ramp);
        }
        let back = conv(&gbr, f, w, h, PixelFormat::Gray16Le);
        assert_eq!(rd16(&back.planes[0].data), ramp, "{f:?}");
    }
}

#[test]
fn gbr_deep_gray_threaded_equals_serial() {
    let mt = ConvertContext::new().with_execution(ExecutionContext::with_threads(8));
    let opts = ConvertOptions::default();
    let sizes: &[(usize, usize)] = if cfg!(miri) {
        &[(2, 33)]
    } else {
        &[(1, 1), (5, 33), (70, 97)]
    };
    for &(w, h) in sizes {
        for &(f, bits, alpha) in GBR {
            let src = gbr_frame(w, h, bits, alpha, (w * 7 + h) as u64);
            let info = FrameInfo::new(f, w as u32, h as u32);
            for &(g, _) in DEEP_GRAY {
                let a = convert(&src, info, g, &opts).unwrap();
                let b = convert_with(&src, info, g, &opts, &mt).unwrap();
                assert!(same_planes(&a, &b), "{f:?} → {g:?} {w}×{h}");
                let ginfo = FrameInfo::new(g, w as u32, h as u32);
                let ia = convert(&a, ginfo, f, &opts).unwrap();
                let ib = convert_with(&a, ginfo, f, &opts, &mt).unwrap();
                assert!(same_planes(&ia, &ib), "{g:?} → {f:?} {w}×{h}");
            }
        }
    }
}

/// The planar GBR(A) ↔ packed deep rows (now banded) are byte-identical
/// serial vs threaded, alpha carried / synthesised / dropped.
#[test]
fn gbr_packed_deep_threaded_equals_serial() {
    use PixelFormat as P;
    let mt = ConvertContext::new().with_execution(ExecutionContext::with_threads(8));
    let opts = ConvertOptions::default();
    let sizes: &[(usize, usize)] = if cfg!(miri) {
        &[(2, 33)]
    } else {
        &[(1, 1), (3, 33), (64, 70), (130, 97)]
    };
    for &(w, h) in sizes {
        for (fi, &(f, bits, alpha)) in GBR.iter().enumerate() {
            if bits == 8 || (cfg!(miri) && fi % 3 != 0) {
                continue;
            }
            let src = gbr_frame(w, h, bits, alpha, (w * 3 + h) as u64);
            let info = FrameInfo::new(f, w as u32, h as u32);
            for packed in [P::Rgb48Le, P::Rgba64Le] {
                let a = convert(&src, info, packed, &opts).unwrap();
                let b = convert_with(&src, info, packed, &opts, &mt).unwrap();
                assert!(same_planes(&a, &b), "{f:?} → {packed:?} {w}×{h}");
                let pinfo = FrameInfo::new(packed, w as u32, h as u32);
                for &(back, bbits, _) in GBR {
                    if bbits == 8 {
                        continue;
                    }
                    let ia = convert(&a, pinfo, back, &opts).unwrap();
                    let ib = convert_with(&a, pinfo, back, &opts, &mt).unwrap();
                    assert!(same_planes(&ia, &ib), "{packed:?} → {back:?} {w}×{h}");
                }
            }
        }
    }
}

/// Alpha-less deep endpoints pivot through alpha-less deep members,
/// with the same bytes the alpha-first order produced.
#[test]
fn alpha_less_deep_pairs_pivot_without_alpha() {
    use PixelFormat as P;
    let (w, h) = if cfg!(miri) { (2usize, 2usize) } else { (6, 4) };
    let cases: &[(P, P, P)] = &[
        // (src, dst, the alpha pivot the old order picked)
        (P::Gbrp10Le, P::Yuv420P10Le, P::Rgba64Le),
        (P::Gbrp16Le, P::Yuv444P12Le, P::Rgba64Le),
        (P::Yuv420P10Le, P::Gbrp12Le, P::Yuva444P16Le),
        (P::Yuv422P12Le, P::Gbrp14Le, P::Yuva444P16Le),
        (P::Gbrp14Le, P::Yuv440P12Le, P::Rgba64Le),
    ];
    for &(sf, df, old) in cases {
        let pivot = staged_pivot(sf, df).unwrap_or_else(|| panic!("{sf:?} → {df:?} direct?"));
        assert!(
            !FormatInfo::of(pivot).has_alpha,
            "{sf:?} → {df:?} via {pivot:?}"
        );
        assert!(FormatInfo::of(pivot).bit_depth >= 10);
        let src = match sf {
            P::Gbrp10Le => gbr_frame(w, h, 10, false, 1),
            P::Gbrp16Le => gbr_frame(w, h, 16, false, 2),
            P::Gbrp14Le => gbr_frame(w, h, 14, false, 4),
            _ => {
                let g = gbr_frame(w, h, 16, false, 3);
                conv(&g, P::Gbrp16Le, w, h, sf)
            }
        };
        let direct = conv(&src, sf, w, h, df);
        let mid = conv(&src, sf, w, h, old);
        let staged = conv(&mid, old, w, h, df);
        assert!(
            same_planes(&direct, &staged),
            "{sf:?} → {df:?}: {pivot:?} vs {old:?}"
        );
    }
}
