//! Precision gates for the deep grey ↔ planar YUV(A) rows and the
//! packed-16 → planar family rows (round 476).
//!
//! * `Gray10Le` / `Gray12Le` / `Gray16Le` ↔ every deep (10 / 12 / 16-bit)
//!   planar family member is direct, through the luma plane only:
//!   the range move happens at the deeper of the two depths (H.273
//!   equations 27–29 scaled by 2^(n−8)), chroma is the exact neutral
//!   mid-code, alpha opaque full-scale;
//! * a full-range colour signal makes the luma move a verbatim copy
//!   (plus the ladder widen / narrow);
//! * `Rgb48Le` / `Rgba64Le` → every planar family member is direct
//!   and byte-identical to the historical staged route through the
//!   16-bit 4:4:4 tier;
//! * every new row is byte-identical serial vs threaded.

use oxideav_core::{ColorRange, ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    ConvertContext, ConvertOptions, FormatInfo, FrameInfo, convert, convert_with, staged_pivot,
    supports_direct,
};

fn le16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn rd16(data: &[u8]) -> Vec<u16> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn same_planes(a: &VideoFrame, b: &VideoFrame) -> bool {
    a.planes.len() == b.planes.len()
        && a.planes
            .iter()
            .zip(b.planes.iter())
            .all(|(x, y)| x.stride == y.stride && x.data == y.data)
}

fn one_plane(stride: usize, data: Vec<u8>) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
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

/// `(format, wsub, hsub, bits, alpha)` for every deep planar member.
const DEEP_FAMILY: &[(PixelFormat, usize, usize, u32, bool)] = &[
    (PixelFormat::Yuv420P10Le, 2, 2, 10, false),
    (PixelFormat::Yuv422P10Le, 2, 1, 10, false),
    (PixelFormat::Yuv444P10Le, 1, 1, 10, false),
    (PixelFormat::Yuv440P10Le, 1, 2, 10, false),
    (PixelFormat::Yuv420P12Le, 2, 2, 12, false),
    (PixelFormat::Yuv422P12Le, 2, 1, 12, false),
    (PixelFormat::Yuv444P12Le, 1, 1, 12, false),
    (PixelFormat::Yuv440P12Le, 1, 2, 12, false),
    (PixelFormat::Yuv420P16Le, 2, 2, 16, false),
    (PixelFormat::Yuv422P16Le, 2, 1, 16, false),
    (PixelFormat::Yuv444P16Le, 1, 1, 16, false),
    (PixelFormat::Yuv440P16Le, 1, 2, 16, false),
    (PixelFormat::Yuva420P10Le, 2, 2, 10, true),
    (PixelFormat::Yuva422P10Le, 2, 1, 10, true),
    (PixelFormat::Yuva444P10Le, 1, 1, 10, true),
    (PixelFormat::Yuva420P12Le, 2, 2, 12, true),
    (PixelFormat::Yuva422P12Le, 2, 1, 12, true),
    (PixelFormat::Yuva444P12Le, 1, 1, 12, true),
    (PixelFormat::Yuva420P16Le, 2, 2, 16, true),
    (PixelFormat::Yuva422P16Le, 2, 1, 16, true),
    (PixelFormat::Yuva444P16Le, 1, 1, 16, true),
];

const BYTE_FAMILY: &[(PixelFormat, usize, usize, bool)] = &[
    (PixelFormat::Yuv420P, 2, 2, false),
    (PixelFormat::Yuv422P, 2, 1, false),
    (PixelFormat::Yuv444P, 1, 1, false),
    (PixelFormat::Yuv440P, 1, 2, false),
    (PixelFormat::Yuva420P, 2, 2, true),
    (PixelFormat::Yuva422P, 2, 1, true),
    (PixelFormat::Yuva444P, 1, 1, true),
];

/// Under miri (every instruction interpreted) the member loops below
/// keep one representative per depth / siting class and a few pixels;
/// natively every member runs at the full sizes.
fn miri_thin(i: usize) -> bool {
    cfg!(miri) && !i.is_multiple_of(4)
}

const DEEP_GRAY: &[(PixelFormat, u32)] = &[
    (PixelFormat::Gray10Le, 10),
    (PixelFormat::Gray12Le, 12),
    (PixelFormat::Gray16Le, 16),
];

/// Ladder widen (MSB replication) / narrow (truncation).
fn rescale(v: u32, from: u32, to: u32) -> u32 {
    if to > from {
        (v << (to - from)) | (v >> (from - (to - from)))
    } else {
        v >> (from - to)
    }
}

/// f64 full → limited luma at `n` bits (H.273 eq. 27 scaled).
fn full_to_limited_ref(v: u32, n: u32) -> f64 {
    let s = (1u64 << (n - 8)) as f64;
    let max = ((1u64 << n) - 1) as f64;
    (219.0 * v as f64 / max + 16.0) * s
}

/// f64 limited → full luma at `n` bits (inverse of the above, clamped).
fn limited_to_full_ref(v: u32, n: u32) -> f64 {
    let s = (1u64 << (n - 8)) as f64;
    let max = ((1u64 << n) - 1) as f64;
    ((v as f64 - 16.0 * s) * max / (219.0 * s)).clamp(0.0, max)
}

fn family_frame(
    w: usize,
    h: usize,
    wsub: usize,
    hsub: usize,
    bits: u32,
    alpha: bool,
    seed: u64,
) -> VideoFrame {
    let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
    let mask = ((1u32 << bits) - 1) as u16;
    let mut rng = Rng(seed | 1);
    let mut plane = |n: usize| -> VideoPlane {
        let words: Vec<u16> = (0..n).map(|_| rng.u16() & mask).collect();
        VideoPlane {
            stride: 0,
            data: le16(&words),
        }
    };
    let mut y = plane(w * h);
    y.stride = w * 2;
    let mut u = plane(cw * ch);
    u.stride = cw * 2;
    let mut v = plane(cw * ch);
    v.stride = cw * 2;
    let mut planes = vec![y, u, v];
    if alpha {
        let mut a = plane(w * h);
        a.stride = w * 2;
        planes.push(a);
    }
    VideoFrame { pts: None, planes }
}

#[test]
fn deep_gray_and_deep_family_are_direct_both_ways() {
    for &(g, _) in DEEP_GRAY {
        for &(f, ..) in DEEP_FAMILY {
            assert!(supports_direct(g, f), "{g:?} → {f:?}");
            assert!(supports_direct(f, g), "{f:?} → {g:?}");
        }
        // The byte members are direct too (the luma move at the grey
        // word's depth, then the ladder narrow / after the ladder widen).
        for &(f, ..) in BYTE_FAMILY {
            assert!(supports_direct(g, f), "{g:?} → {f:?}");
            assert!(supports_direct(f, g), "{f:?} → {g:?}");
            assert_eq!(staged_pivot(g, f), None);
        }
    }
    for p in [PixelFormat::Rgb48Le, PixelFormat::Rgba64Le] {
        for &(f, ..) in DEEP_FAMILY {
            assert!(supports_direct(p, f), "{p:?} → {f:?}");
        }
        for &(f, ..) in BYTE_FAMILY {
            assert!(supports_direct(p, f), "{p:?} → {f:?}");
        }
    }
}

/// Grey → 8-bit YUV: the range move at the grey word's depth, then
/// truncation to bytes; chroma 128, alpha 255.
#[test]
fn deep_gray_to_byte_family_luma_range_at_depth() {
    let (w, h) = if cfg!(miri) { (3usize, 2usize) } else { (9, 6) };
    let n = w * h;
    for &(g, gbits) in DEEP_GRAY {
        let gmask = (1u32 << gbits) - 1;
        let mut rng = Rng(gbits as u64 + 100);
        let mut codes: Vec<u16> = vec![0, gmask as u16];
        codes.extend((2..n).map(|_| (rng.u16() as u32 & gmask) as u16));
        let src = one_plane(w * 2, le16(&codes));
        for &(f, wsub, hsub, alpha) in BYTE_FAMILY {
            let out = convert(
                &src,
                FrameInfo::new(g, w as u32, h as u32),
                f,
                &ConvertOptions::default(),
            )
            .unwrap();
            let y = &out.planes[0].data;
            assert_eq!(out.planes[0].stride, w);
            for i in 0..n {
                let want = full_to_limited_ref(codes[i] as u32, gbits);
                let got = ((y[i] as u32) << (gbits - 8)) as f64;
                let slack = ((1u32 << (gbits - 8)) - 1) as f64 + 1.0;
                assert!((got - want).abs() <= slack, "{g:?} → {f:?} pixel {i}");
            }
            assert_eq!(y[0], 16);
            assert_eq!(y[1], 235);
            let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
            for p in 1..=2 {
                assert_eq!(out.planes[p].stride, cw);
                assert!(
                    out.planes[p].data.len() == cw * ch
                        && out.planes[p].data.iter().all(|&c| c == 128)
                );
            }
            if alpha {
                assert!(out.planes[3].data.iter().all(|&a| a == 255));
            }
            // And back: the luma widened ×257, range-moved at the grey
            // depth's work width.
            let back = convert(
                &out,
                FrameInfo::new(f, w as u32, h as u32),
                g,
                &ConvertOptions::default(),
            )
            .unwrap();
            let gb = rd16(&back.planes[0].data);
            for i in 0..n {
                let v = rescale(y[i] as u32, 8, gbits);
                let want = limited_to_full_ref(v, gbits);
                assert!(
                    (gb[i] as f64 - want).abs() <= 1.0,
                    "{f:?} → {g:?} pixel {i}"
                );
            }
        }
    }
}

/// Grey → deep YUV: luma is the full → limited move at the deeper
/// depth (within 1 LSB of f64 there), then the ladder narrow; chroma
/// is the exact mid-code; alpha is opaque full-scale; the chroma
/// planes have core's geometry.
#[test]
fn deep_gray_to_deep_family_luma_range_at_depth() {
    let (w, h) = if cfg!(miri) {
        (3usize, 2usize)
    } else {
        (37, 11)
    };
    let n = w * h;
    for &(g, gbits) in DEEP_GRAY {
        let gmask = (1u32 << gbits) - 1;
        let mut rng = Rng(gbits as u64);
        // Rails plus random.
        let mut codes: Vec<u16> = vec![0, gmask as u16, (gmask / 2) as u16, 1, (gmask - 1) as u16];
        codes.extend((5..n).map(|_| (rng.u16() as u32 & gmask) as u16));
        let src = one_plane(w * 2, le16(&codes));
        for (fi, &(f, wsub, hsub, bits, alpha)) in DEEP_FAMILY.iter().enumerate() {
            if miri_thin(fi) {
                continue;
            }
            let out = convert(
                &src,
                FrameInfo::new(g, w as u32, h as u32),
                f,
                &ConvertOptions::default(),
            )
            .unwrap();
            let work = gbits.max(bits);
            let y = rd16(&out.planes[0].data);
            assert_eq!(out.planes[0].stride, w * 2);
            for i in 0..n {
                let v = rescale(codes[i] as u32, gbits, work);
                let want = full_to_limited_ref(v, work);
                let got = rescale(y[i] as u32, bits, work) as f64;
                // Narrowing after the move truncates the low bits: allow
                // their span on top of the 1 LSB at the work depth.
                let slack = ((1u32 << (work - bits)) - 1) as f64 + 1.0;
                assert!(
                    (got - want).abs() <= slack,
                    "{g:?} → {f:?} pixel {i}: code {} got {} want {want:.3} (work {work})",
                    codes[i],
                    y[i]
                );
                if bits == work {
                    assert!((got - want.round()).abs() <= 1.0);
                }
            }
            // Rails at the work depth: 0 → 16·2^(n−8), peak → 235·2^(n−8).
            let s = 1u32 << (bits - 8);
            assert_eq!(y[0] as u32, 16 * s, "{g:?} → {f:?} black rail");
            assert_eq!(y[1] as u32, 235 * s, "{g:?} → {f:?} white rail");
            let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
            for p in 1..=2 {
                assert_eq!(out.planes[p].stride, cw * 2);
                assert_eq!(out.planes[p].data.len(), cw * ch * 2);
                let mid = 1u16 << (bits - 1);
                assert!(
                    rd16(&out.planes[p].data).iter().all(|&c| c == mid),
                    "{g:?} → {f:?} chroma"
                );
            }
            assert_eq!(out.planes.len(), if alpha { 4 } else { 3 });
            if alpha {
                let full = ((1u32 << bits) - 1) as u16;
                assert!(rd16(&out.planes[3].data).iter().all(|&a| a == full));
            }
            let geom = FormatInfo::of(f);
            assert_eq!(geom.chroma_w_sub as usize, wsub);
            assert_eq!(geom.chroma_h_sub as usize, hsub);
        }
    }
}

/// Deep YUV → grey: the limited → full move at the deeper depth (within
/// 1 LSB of f64 there), then the ladder narrow. Chroma and alpha never
/// influence the result.
#[test]
fn deep_family_to_deep_gray_luma_range_at_depth() {
    let (w, h) = if cfg!(miri) {
        (3usize, 2usize)
    } else {
        (23, 9)
    };
    let n = w * h;
    for (fi, &(f, wsub, hsub, bits, alpha)) in DEEP_FAMILY.iter().enumerate() {
        if miri_thin(fi) {
            continue;
        }
        let src = family_frame(w, h, wsub, hsub, bits, alpha, bits as u64 * 7 + wsub as u64);
        let y_in = rd16(&src.planes[0].data);
        for &(g, gbits) in DEEP_GRAY {
            let out = convert(
                &src,
                FrameInfo::new(f, w as u32, h as u32),
                g,
                &ConvertOptions::default(),
            )
            .unwrap();
            assert_eq!(out.planes.len(), 1);
            assert_eq!(out.planes[0].stride, w * 2);
            let got = rd16(&out.planes[0].data);
            let work = gbits.max(bits);
            for i in 0..n {
                let v = rescale(y_in[i] as u32, bits, work);
                let want = limited_to_full_ref(v, work);
                let gw = rescale(got[i] as u32, gbits, work) as f64;
                let slack = ((1u32 << (work - gbits)) - 1) as f64 + 1.0;
                assert!(
                    (gw - want).abs() <= slack,
                    "{f:?} → {g:?} pixel {i}: code {} got {} want {want:.3}",
                    y_in[i],
                    got[i]
                );
            }
            // Chroma / alpha independence: scramble them, same output.
            let mut scrambled = src.clone();
            for p in scrambled.planes.iter_mut().skip(1) {
                for b in p.data.iter_mut() {
                    *b = !*b;
                }
            }
            let again = convert(
                &scrambled,
                FrameInfo::new(f, w as u32, h as u32),
                g,
                &ConvertOptions::default(),
            )
            .unwrap();
            assert_eq!(again.planes[0].data, out.planes[0].data);
        }
    }
}

/// Rails on the headline 16-bit pair: limited black / white at 16 bits
/// are 4096 / 60160 (BT-series n-bit representation), and the
/// 56064-code limited span maps onto the full 65536-code grey space
/// within 1 LSB and back within 1 LSB.
#[test]
fn gray16_yuv444p16_rails_and_round_trip() {
    // Full ramp natively; every 257th code (and the rails) under miri.
    let step = if cfg!(miri) { 257 } else { 1 };
    let mut ramp: Vec<u16> = (0..=u16::MAX).step_by(step).collect();
    let w = if step == 1 { 256u32 } else { 16 };
    let h = (ramp.len() as u32).div_ceil(w);
    ramp.resize((w * h) as usize, u16::MAX);
    let last = ramp.len() - 1;
    let src = one_plane(w as usize * 2, le16(&ramp));
    let yuv = convert(
        &src,
        FrameInfo::new(PixelFormat::Gray16Le, w, h),
        PixelFormat::Yuv444P16Le,
        &ConvertOptions::default(),
    )
    .unwrap();
    let y = rd16(&yuv.planes[0].data);
    assert_eq!(y[0], 4096);
    assert_eq!(y[last], 60160);
    assert!(y.windows(2).all(|p| p[0] <= p[1]), "monotonic");
    let back = convert(
        &yuv,
        FrameInfo::new(PixelFormat::Yuv444P16Le, w, h),
        PixelFormat::Gray16Le,
        &ConvertOptions::default(),
    )
    .unwrap();
    let g = rd16(&back.planes[0].data);
    for (i, (&a, &b)) in ramp.iter().zip(g.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "code {i}: {a} → {} → {b}",
            y[i]
        );
    }
    // Every limited code survives YUV → grey → YUV exactly (the full
    // space is wider than the limited span).
    let limited: Vec<u16> = (4096u16..=60160).step_by(step).collect();
    let lw = 256u32;
    let lh = (limited.len() as u32).div_ceil(lw);
    let mut padded = limited.clone();
    padded.resize((lw * lh) as usize, 4096);
    let yfr = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: lw as usize * 2,
                data: le16(&padded),
            },
            VideoPlane {
                stride: lw as usize * 2,
                data: le16(&vec![32768u16; (lw * lh) as usize]),
            },
            VideoPlane {
                stride: lw as usize * 2,
                data: le16(&vec![32768u16; (lw * lh) as usize]),
            },
        ],
    };
    let g = convert(
        &yfr,
        FrameInfo::new(PixelFormat::Yuv444P16Le, lw, lh),
        PixelFormat::Gray16Le,
        &ConvertOptions::default(),
    )
    .unwrap();
    let y2 = convert(
        &g,
        FrameInfo::new(PixelFormat::Gray16Le, lw, lh),
        PixelFormat::Yuv444P16Le,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert_eq!(rd16(&y2.planes[0].data), padded);
}

/// A full-range signal (context override, or the frame's own record on
/// the YUV side) turns the range move off: grey ↔ deep YUV luma is the
/// bare ladder move, and `Gray16Le ↔ Yuv*P16Le` is a verbatim copy.
#[test]
fn full_range_signal_makes_the_luma_move_verbatim() {
    let (w, h) = if cfg!(miri) {
        (2usize, 2usize)
    } else {
        (16, 8)
    };
    let n = w * h;
    let full = ConvertContext::new().with_range(ColorRange::Full);
    for &(g, gbits) in DEEP_GRAY {
        let mask = (1u32 << gbits) - 1;
        let codes: Vec<u16> = (0..n).map(|i| ((i * 2731) as u32 & mask) as u16).collect();
        let src = one_plane(w * 2, le16(&codes));
        for (fi, &(f, _, _, bits, _)) in DEEP_FAMILY.iter().enumerate() {
            if miri_thin(fi) {
                continue;
            }
            let out = convert_with(
                &src,
                FrameInfo::new(g, w as u32, h as u32),
                f,
                &ConvertOptions::default(),
                &full,
            )
            .unwrap();
            let y = rd16(&out.planes[0].data);
            for i in 0..n {
                assert_eq!(
                    y[i] as u32,
                    rescale(codes[i] as u32, gbits, bits),
                    "{g:?} → {f:?}"
                );
            }
            // Back through the frame's own record.
            let mut tagged = out.clone();
            tagged.set_color_signal(
                oxideav_core::ColorSignal::unspecified().with_range(ColorRange::Full),
            );
            let back = convert(
                &tagged,
                FrameInfo::new(f, w as u32, h as u32),
                g,
                &ConvertOptions::default(),
            )
            .unwrap();
            let gb = rd16(&back.planes[0].data);
            for i in 0..n {
                assert_eq!(
                    gb[i] as u32,
                    rescale(y[i] as u32, bits, gbits),
                    "{f:?} → {g:?}"
                );
            }
            if gbits == bits {
                assert_eq!(gb, codes);
            }
        }
    }
}

/// Packed 16-bit RGB(A) → every planar member equals the historical
/// two-step route (deep matrix to the 16-bit 4:4:4 tier, then the
/// planar-family engine), at even and odd sizes.
#[test]
fn packed16_to_family_matches_the_staged_route() {
    let sizes: &[(usize, usize)] = if cfg!(miri) {
        &[(3, 5), (1, 1)]
    } else {
        &[(8, 8), (3, 5), (7, 3), (1, 1), (17, 2)]
    };
    for &(w, h) in sizes {
        let n = w * h;
        for (src_fmt, comps) in [(PixelFormat::Rgb48Le, 3usize), (PixelFormat::Rgba64Le, 4)] {
            let mut rng = Rng((w * 31 + h) as u64 * comps as u64);
            let words: Vec<u16> = (0..n * comps).map(|_| rng.u16()).collect();
            let src = one_plane(w * comps * 2, le16(&words));
            let info = FrameInfo::new(src_fmt, w as u32, h as u32);
            let tier = if comps == 4 {
                PixelFormat::Yuva444P16Le
            } else {
                PixelFormat::Yuv444P16Le
            };
            let mid = convert(&src, info, tier, &ConvertOptions::default()).unwrap();
            let mid_info = FrameInfo::new(tier, w as u32, h as u32);
            let mut targets: Vec<PixelFormat> = DEEP_FAMILY.iter().map(|t| t.0).collect();
            targets.extend(BYTE_FAMILY.iter().map(|t| t.0));
            for (ti, dst) in targets.into_iter().enumerate() {
                if miri_thin(ti) {
                    continue;
                }
                let direct = convert(&src, info, dst, &ConvertOptions::default()).unwrap();
                let staged = convert(&mid, mid_info, dst, &ConvertOptions::default()).unwrap();
                assert_eq!(
                    direct.planes.len(),
                    staged.planes.len(),
                    "{src_fmt:?} → {dst:?}"
                );
                for (p, (a, b)) in direct.planes.iter().zip(staged.planes.iter()).enumerate() {
                    assert_eq!(a.stride, b.stride, "{src_fmt:?} → {dst:?} plane {p} stride");
                    assert_eq!(a.data, b.data, "{src_fmt:?} → {dst:?} plane {p} {w}×{h}");
                }
            }
        }
    }
}

/// Alpha crossing on the packed-16 → family rows: an `Rgba64Le` alpha
/// word lands in the alpha plane narrowed to the member's depth; an
/// `Rgb48Le` source synthesises opaque; a non-alpha member drops it.
#[test]
fn packed16_to_family_alpha_rules() {
    let (w, h) = if cfg!(miri) { (2usize, 2usize) } else { (6, 4) };
    let n = w * h;
    let mut rng = Rng(42);
    let words: Vec<u16> = (0..n * 4).map(|_| rng.u16()).collect();
    let rgba = one_plane(w * 8, le16(&words));
    let rgb = one_plane(
        w * 6,
        le16(
            &(0..n)
                .flat_map(|i| [words[i * 4], words[i * 4 + 1], words[i * 4 + 2]])
                .collect::<Vec<_>>(),
        ),
    );
    for (fi, &(f, _, _, bits, alpha)) in DEEP_FAMILY.iter().enumerate() {
        if miri_thin(fi) {
            continue;
        }
        let from_rgba = convert(
            &rgba,
            FrameInfo::new(PixelFormat::Rgba64Le, w as u32, h as u32),
            f,
            &ConvertOptions::default(),
        )
        .unwrap();
        let from_rgb = convert(
            &rgb,
            FrameInfo::new(PixelFormat::Rgb48Le, w as u32, h as u32),
            f,
            &ConvertOptions::default(),
        )
        .unwrap();
        for p in 0..3 {
            assert_eq!(
                from_rgba.planes[p].data, from_rgb.planes[p].data,
                "{f:?} colour plane {p}"
            );
        }
        if alpha {
            let a = rd16(&from_rgba.planes[3].data);
            for i in 0..n {
                assert_eq!(
                    a[i] as u32,
                    words[i * 4 + 3] as u32 >> (16 - bits),
                    "{f:?} alpha"
                );
            }
            let full = ((1u32 << bits) - 1) as u16;
            assert!(rd16(&from_rgb.planes[3].data).iter().all(|&a| a == full));
        } else {
            assert_eq!(from_rgba.planes.len(), 3);
            assert_eq!(from_rgb.planes.len(), 3);
        }
    }
}

/// Every new row (and the re-engined `Rgb48Le → Yuv*P16Le` table rows)
/// is byte-identical on one thread and on many, at sizes that split
/// into several bands and at sizes that do not.
#[test]
fn new_rows_threaded_equals_serial() {
    let mt = ConvertContext::new().with_execution(ExecutionContext::with_threads(8));
    let opts = ConvertOptions::default();
    let sizes: &[(usize, usize)] = if cfg!(miri) {
        &[(2, 33)]
    } else {
        &[(1, 1), (5, 3), (64, 33), (130, 97)]
    };
    for &(w, h) in sizes {
        let n = w * h;
        let mut rng = Rng((w * 1000 + h) as u64);
        for &(g, gbits) in DEEP_GRAY {
            let mask = (1u32 << gbits) - 1;
            let codes: Vec<u16> = (0..n).map(|_| (rng.u16() as u32 & mask) as u16).collect();
            let src = one_plane(w * 2, le16(&codes));
            let info = FrameInfo::new(g, w as u32, h as u32);
            let targets = DEEP_FAMILY
                .iter()
                .map(|t| t.0)
                .chain(BYTE_FAMILY.iter().map(|t| t.0));
            for (ti, f) in targets.enumerate() {
                if miri_thin(ti) {
                    continue;
                }
                let a = convert(&src, info, f, &opts).unwrap();
                let b = convert_with(&src, info, f, &opts, &mt).unwrap();
                assert!(same_planes(&a, &b), "{g:?} → {f:?} {w}×{h}");
                let ia = convert(&a, FrameInfo::new(f, w as u32, h as u32), g, &opts).unwrap();
                let ib =
                    convert_with(&a, FrameInfo::new(f, w as u32, h as u32), g, &opts, &mt).unwrap();
                assert!(same_planes(&ia, &ib), "{f:?} → {g:?} {w}×{h}");
            }
        }
        for (fi, &(f, wsub, hsub, bits, alpha)) in DEEP_FAMILY.iter().enumerate() {
            if miri_thin(fi) {
                continue;
            }
            let src = family_frame(w, h, wsub, hsub, bits, alpha, (w + h) as u64);
            let info = FrameInfo::new(f, w as u32, h as u32);
            for &(g, _) in DEEP_GRAY {
                let a = convert(&src, info, g, &opts).unwrap();
                let b = convert_with(&src, info, g, &opts, &mt).unwrap();
                assert!(same_planes(&a, &b), "{f:?} → {g:?} {w}×{h}");
            }
        }
        for (p, comps) in [(PixelFormat::Rgb48Le, 3usize), (PixelFormat::Rgba64Le, 4)] {
            let words: Vec<u16> = (0..n * comps).map(|_| rng.u16()).collect();
            let src = one_plane(w * comps * 2, le16(&words));
            let info = FrameInfo::new(p, w as u32, h as u32);
            let targets = DEEP_FAMILY
                .iter()
                .map(|t| t.0)
                .chain(BYTE_FAMILY.iter().map(|t| t.0));
            for (ti, f) in targets.enumerate() {
                if miri_thin(ti) {
                    continue;
                }
                let a = convert(&src, info, f, &opts).unwrap();
                let b = convert_with(&src, info, f, &opts, &mt).unwrap();
                assert!(same_planes(&a, &b), "{p:?} → {f:?} {w}×{h}");
            }
        }
    }
}

/// Short planes are `Error::Invalid`, never a panic.
#[test]
fn short_planes_are_invalid() {
    let src = one_plane(8, vec![0u8; 8]);
    for (sf, df) in [
        (PixelFormat::Gray16Le, PixelFormat::Yuv420P10Le),
        (PixelFormat::Yuv444P16Le, PixelFormat::Gray16Le),
        (PixelFormat::Rgb48Le, PixelFormat::Yuv420P12Le),
    ] {
        let r = convert(
            &src,
            FrameInfo::new(sf, 4, 4),
            df,
            &ConvertOptions::default(),
        );
        assert!(r.is_err(), "{sf:?} → {df:?}");
    }
    let no_planes = VideoFrame {
        pts: None,
        planes: vec![],
    };
    assert!(
        convert(
            &no_planes,
            FrameInfo::new(PixelFormat::Yuv444P16Le, 4, 4),
            PixelFormat::Gray16Le,
            &ConvertOptions::default()
        )
        .is_err()
    );
}
