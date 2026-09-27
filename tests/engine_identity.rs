//! Byte-identity gate for the row-band YUV ↔ RGB engine.
//!
//! Every 8-bit planar YUV(A) ↔ packed RGB(A) row of `convert()` now runs
//! through one engine (borrowed planes, fused RGBA interleave, optional
//! row-band threads). These tests pin its output, byte for byte, to an
//! independent per-pixel oracle built from the crate's public scalar
//! Q15 primitives — `yuv::yuv_to_rgb` / `yuv::rgb_to_yuv`, the exact
//! functions the historical whole-frame scalar path evaluated per pixel
//! — plus the documented chroma geometry:
//!
//! - decode reads chroma sample `(col / wsub, row / hsub)`;
//! - encode averages the per-pixel rounded chroma of each block,
//!   `(sum + 2) / 4` for 2×2 and `(sum + 1) / 2` for pairs, with
//!   positions past the picture edge replicating the last column / row.
//!
//! Geometries cover even sizes (the historical contract — so identity
//! here is identity with the pre-engine bytes), odd widths and heights
//! (1×1, 3×5, 7×3, …), row counts that straddle the engine's band
//! boundaries, stride padding, and every thread budget from 1 to 8.

use oxideav_core::{ColorRange, ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::yuv::{self, YuvMatrix};
use oxideav_pixfmt::{
    convert, convert_with, ColorSpace, ConvertContext, ConvertOptions, FrameInfo,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

fn plane(rng: &mut Rng, w_bytes: usize, h: usize, pad: usize) -> VideoPlane {
    let stride = w_bytes + pad;
    let mut data = vec![0xEEu8; stride * h];
    for row in 0..h {
        for b in &mut data[row * stride..row * stride + w_bytes] {
            *b = rng.byte();
        }
    }
    VideoPlane { stride, data }
}

fn tight(p: &VideoPlane, w_bytes: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w_bytes * h);
    for row in 0..h {
        out.extend_from_slice(&p.data[row * p.stride..row * p.stride + w_bytes]);
    }
    out
}

/// (format, wsub, hsub, alpha, full_range)
const PLANAR8: &[(PixelFormat, usize, usize, bool, bool)] = &[
    (PixelFormat::Yuv420P, 2, 2, false, false),
    (PixelFormat::Yuv422P, 2, 1, false, false),
    (PixelFormat::Yuv444P, 1, 1, false, false),
    (PixelFormat::YuvJ420P, 2, 2, false, true),
    (PixelFormat::YuvJ422P, 2, 1, false, true),
    (PixelFormat::YuvJ444P, 1, 1, false, true),
    (PixelFormat::Yuva420P, 2, 2, true, false),
    (PixelFormat::Yuva422P, 2, 1, true, false),
    (PixelFormat::Yuva444P, 1, 1, true, false),
    (PixelFormat::Yuv440P, 1, 2, false, false),
];

/// Geometry sweep. Under Miri (the UB job interprets every test) the
/// sweep keeps its odd, even and band-straddling representatives but
/// drops the larger pictures, the crate's usual `cfg(miri)` shrink.
const GEOMETRIES: &[(usize, usize)] = if cfg!(miri) {
    &[(1, 1), (3, 5), (7, 3), (16, 8), (5, 67)]
} else {
    &[
        (1, 1),
        (2, 2),
        (3, 5),
        (7, 3),
        (16, 8),
        (17, 9),
        (33, 65),
        (64, 70),
        (5, 131),
    ]
};

/// Thread budgets exercised per case (serial, and enough workers to
/// split every multi-band geometry above).
const THREADS: &[usize] = if cfg!(miri) { &[1, 3] } else { &[1, 2, 3, 8] };

fn source_frame(
    rng: &mut Rng,
    w: usize,
    h: usize,
    wsub: usize,
    hsub: usize,
    alpha: bool,
    pad: usize,
) -> VideoFrame {
    let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
    let mut planes = vec![
        plane(rng, w, h, pad),
        plane(rng, cw, ch, pad),
        plane(rng, cw, ch, pad),
    ];
    if alpha {
        planes.push(plane(rng, w, h, pad));
    }
    VideoFrame { pts: None, planes }
}

/// Per-pixel decode oracle.
#[allow(clippy::too_many_arguments)]
fn oracle_decode(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    a: Option<&[u8]>,
    w: usize,
    h: usize,
    wsub: usize,
    hsub: usize,
    bpp: usize,
    m: YuvMatrix,
) -> Vec<u8> {
    let cw = w.div_ceil(wsub);
    let mut out = Vec::with_capacity(w * h * bpp);
    for row in 0..h {
        for col in 0..w {
            let ci = (row / hsub) * cw + col / wsub;
            let (r, g, b) = yuv::yuv_to_rgb(y[row * w + col], u[ci], v[ci], m);
            out.extend_from_slice(&[r, g, b]);
            if bpp == 4 {
                out.push(a.map_or(255, |a| a[row * w + col]));
            }
        }
    }
    out
}

/// Per-pixel encode oracle: (Y, U, V, A-or-empty).
#[allow(clippy::type_complexity)]
fn oracle_encode(
    rgb: &[u8],
    bpp: usize,
    w: usize,
    h: usize,
    wsub: usize,
    hsub: usize,
    m: YuvMatrix,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let px = |row: usize, col: usize| {
        let o = (row * w + col) * bpp;
        yuv::rgb_to_yuv(rgb[o], rgb[o + 1], rgb[o + 2], m)
    };
    let mut y = Vec::with_capacity(w * h);
    let mut a = Vec::new();
    for row in 0..h {
        for col in 0..w {
            y.push(px(row, col).0);
            if bpp == 4 {
                a.push(rgb[(row * w + col) * 4 + 3]);
            }
        }
    }
    let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
    let mut u = Vec::with_capacity(cw * ch);
    let mut v = Vec::with_capacity(cw * ch);
    let n = (wsub * hsub) as u32;
    for cr in 0..ch {
        for cc in 0..cw {
            let (mut su, mut sv) = (0u32, 0u32);
            for dy in 0..hsub {
                for dx in 0..wsub {
                    let row = (cr * hsub + dy).min(h - 1);
                    let col = (cc * wsub + dx).min(w - 1);
                    let (_, pu, pv) = px(row, col);
                    su += pu as u32;
                    sv += pv as u32;
                }
            }
            u.push(((su + n / 2) / n) as u8);
            v.push(((sv + n / 2) / n) as u8);
        }
    }
    (y, u, v, a)
}

fn matrix_for(cs: ColorSpace, full_range: bool) -> YuvMatrix {
    YuvMatrix::from_color_space(cs).with_range(!full_range)
}

fn opts(cs: ColorSpace) -> ConvertOptions {
    ConvertOptions {
        color_space: cs,
        ..Default::default()
    }
}

#[test]
fn planar8_to_packed_matches_the_per_pixel_oracle() {
    let mut rng = Rng(0x5EED_0463);
    for &(fmt, wsub, hsub, alpha, full) in PLANAR8 {
        for &(w, h) in GEOMETRIES {
            for (pad, cs) in [
                (0usize, ColorSpace::Bt601Limited),
                (3, ColorSpace::Bt709Full),
            ] {
                let src = source_frame(&mut rng, w, h, wsub, hsub, alpha, pad);
                let info = FrameInfo::new(fmt, w as u32, h as u32);
                let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
                let y = tight(&src.planes[0], w, h);
                let u = tight(&src.planes[1], cw, ch);
                let v = tight(&src.planes[2], cw, ch);
                let a = alpha.then(|| tight(&src.planes[3], w, h));
                let m = matrix_for(cs, full);
                for (dst, bpp) in [(PixelFormat::Rgb24, 3), (PixelFormat::Rgba, 4)] {
                    let want = oracle_decode(&y, &u, &v, a.as_deref(), w, h, wsub, hsub, bpp, m);
                    for &threads in THREADS {
                        let ctx = ConvertContext::new().with_threads(threads);
                        let got = convert_with(&src, info, dst, &opts(cs), &ctx)
                            .unwrap_or_else(|e| panic!("{fmt:?} {w}x{h} → {dst:?}: {e:?}"));
                        assert_eq!(got.planes[0].stride, w * bpp);
                        assert!(
                            got.planes[0].data == want,
                            "{fmt:?} {w}x{h} pad {pad} {cs:?} → {dst:?} ×{threads}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn packed_to_planar8_matches_the_per_pixel_oracle() {
    let mut rng = Rng(0x0463_EC0D);
    for &(fmt, wsub, hsub, alpha, full) in PLANAR8 {
        for &(w, h) in GEOMETRIES {
            for (src_fmt, bpp, pad) in [
                (PixelFormat::Rgb24, 3usize, 0usize),
                (PixelFormat::Rgb24, 3, 5),
                (PixelFormat::Rgba, 4, 0),
                (PixelFormat::Rgba, 4, 2),
            ] {
                let cs = ColorSpace::Bt2020Limited;
                let src = VideoFrame {
                    pts: None,
                    planes: vec![plane(&mut rng, w * bpp, h, pad)],
                };
                let rgb = tight(&src.planes[0], w * bpp, h);
                let (wy, wu, wv, wa) =
                    oracle_encode(&rgb, bpp, w, h, wsub, hsub, matrix_for(cs, full));
                let info = FrameInfo::new(src_fmt, w as u32, h as u32);
                for &threads in THREADS {
                    let ctx = ConvertContext::new().with_threads(threads);
                    let got = convert_with(&src, info, fmt, &opts(cs), &ctx)
                        .unwrap_or_else(|e| panic!("{src_fmt:?} {w}x{h} → {fmt:?}: {e:?}"));
                    let tag = format!("{src_fmt:?} pad {pad} {w}x{h} → {fmt:?} ×{threads}");
                    assert!(got.planes[0].data == wy, "{tag}: Y");
                    assert!(got.planes[1].data == wu, "{tag}: U");
                    assert!(got.planes[2].data == wv, "{tag}: V");
                    if alpha {
                        let want_a = if bpp == 4 {
                            wa.clone()
                        } else {
                            vec![255u8; w * h]
                        };
                        assert!(got.planes[3].data == want_a, "{tag}: A");
                    }
                }
            }
        }
    }
}

#[test]
fn nv12_and_nv21_match_the_oracle_at_odd_geometry() {
    let mut rng = Rng(0x0463_0012);
    for &(w, h) in GEOMETRIES {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y = plane(&mut rng, w, h, 0);
        let uv = plane(&mut rng, cw * 2, ch, 0);
        let m = matrix_for(ColorSpace::Bt709Limited, false);
        for (fmt, nv12) in [(PixelFormat::Nv12, true), (PixelFormat::Nv21, false)] {
            let src = VideoFrame {
                pts: None,
                planes: vec![y.clone(), uv.clone()],
            };
            let (mut u, mut v) = (Vec::new(), Vec::new());
            for pair in uv.data.chunks_exact(2) {
                let (first, second) = (pair[0], pair[1]);
                if nv12 {
                    u.push(first);
                    v.push(second);
                } else {
                    v.push(first);
                    u.push(second);
                }
            }
            let want = oracle_decode(&y.data, &u, &v, None, w, h, 2, 2, 4, m);
            let got = convert(
                &src,
                FrameInfo::new(fmt, w as u32, h as u32),
                PixelFormat::Rgba,
                &opts(ColorSpace::Bt709Limited),
            )
            .expect("NV → Rgba");
            assert!(got.planes[0].data == want, "{fmt:?} {w}x{h}");
        }
    }
}

/// Deep 4:2:0 members narrow to 8 bits (truncation) and then decode
/// through the same engine — pinned against narrow-then-oracle.
#[test]
fn deep_420_to_packed_matches_narrow_then_oracle() {
    let mut rng = Rng(0x0463_0010);
    for (fmt, bits, alpha) in [
        (PixelFormat::Yuv420P10Le, 10u32, false),
        (PixelFormat::Yuv420P12Le, 12, false),
        (PixelFormat::Yuva420P10Le, 10, true),
    ] {
        for &(w, h) in GEOMETRIES {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let mut word_plane = |pw: usize, ph: usize| {
                let mut data = Vec::with_capacity(pw * ph * 2);
                for _ in 0..pw * ph {
                    let v = (rng.next() as u32) & ((1 << bits) - 1);
                    data.extend_from_slice(&(v as u16).to_le_bytes());
                }
                VideoPlane {
                    stride: pw * 2,
                    data,
                }
            };
            let mut planes = vec![word_plane(w, h), word_plane(cw, ch), word_plane(cw, ch)];
            if alpha {
                planes.push(word_plane(w, h));
            }
            let narrow = |p: &VideoPlane| -> Vec<u8> {
                p.data
                    .chunks_exact(2)
                    .map(|b| (u16::from_le_bytes([b[0], b[1]]) >> (bits - 8)) as u8)
                    .collect()
            };
            let (y, u, v) = (narrow(&planes[0]), narrow(&planes[1]), narrow(&planes[2]));
            let a = alpha.then(|| narrow(&planes[3]));
            let src = VideoFrame { pts: None, planes };
            let m = matrix_for(ColorSpace::Bt2020Limited, false);
            let want = oracle_decode(&y, &u, &v, a.as_deref(), w, h, 2, 2, 4, m);
            for threads in [1usize, 4] {
                let got = convert_with(
                    &src,
                    FrameInfo::new(fmt, w as u32, h as u32),
                    PixelFormat::Rgba,
                    &opts(ColorSpace::Bt2020Limited),
                    &ConvertContext::new().with_threads(threads),
                )
                .expect("deep 4:2:0 → Rgba");
                assert!(got.planes[0].data == want, "{fmt:?} {w}x{h} ×{threads}");
            }
        }
    }
}

/// The full 12 MP odd geometry the HEIF matrix exercises (4031×3023):
/// decode and encode at 4:2:0, serial and threaded, checked against
/// the oracle on the whole last row / column plus a sparse interior
/// sample (the dense geometry sweep above covers the interior rules).
/// Under Miri the same checks run on a 67×37 odd picture (two bands).
#[test]
fn twelve_megapixel_odd_geometry_matches_the_oracle() {
    let (w, h) = if cfg!(miri) {
        (67usize, 37usize)
    } else {
        (4031usize, 3023usize)
    };
    let mut rng = Rng(0x0463_4031);
    let src = source_frame(&mut rng, w, h, 2, 2, false, 0);
    let info = FrameInfo::new(PixelFormat::Yuv420P, w as u32, h as u32);
    let m = matrix_for(ColorSpace::Bt709Limited, false);
    let o = opts(ColorSpace::Bt709Limited);
    let serial = convert(&src, info, PixelFormat::Rgb24, &o).expect("serial");
    let threaded = convert_with(
        &src,
        info,
        PixelFormat::Rgb24,
        &o,
        &ConvertContext::new().with_execution(ExecutionContext::with_threads(8)),
    )
    .expect("threaded");
    assert!(serial.planes[0].data == threaded.planes[0].data);
    let (y, u, v) = (
        &src.planes[0].data,
        &src.planes[1].data,
        &src.planes[2].data,
    );
    let cw = w.div_ceil(2);
    let check = |row: usize, col: usize| {
        let ci = (row / 2) * cw + col / 2;
        let (r, g, b) = yuv::yuv_to_rgb(y[row * w + col], u[ci], v[ci], m);
        let o = (row * w + col) * 3;
        assert_eq!(
            &serial.planes[0].data[o..o + 3],
            &[r, g, b],
            "({col}, {row})"
        );
    };
    for col in 0..w {
        check(h - 1, col);
    }
    for row in 0..h {
        check(row, w - 1);
    }
    let (sy, sx) = if cfg!(miri) { (5, 7) } else { (97, 89) };
    for row in (0..h).step_by(sy) {
        for col in (0..w).step_by(sx) {
            check(row, col);
        }
    }

    // Encode the decoded picture back and compare with the oracle on the
    // edge chroma samples (the truncated blocks).
    let rgb_info = FrameInfo::new(PixelFormat::Rgb24, w as u32, h as u32);
    let enc = convert(&serial, rgb_info, PixelFormat::Yuv420P, &o).expect("encode");
    let enc_mt = convert_with(
        &serial,
        rgb_info,
        PixelFormat::Yuv420P,
        &o,
        &ConvertContext::new().with_threads(6),
    )
    .expect("encode threaded");
    for p in 0..3 {
        assert!(enc.planes[p].data == enc_mt.planes[p].data, "plane {p}");
    }
    let rgb = &serial.planes[0].data;
    let ch = h.div_ceil(2);
    let chroma_at = |cr: usize, cc: usize| {
        let (mut su, mut sv) = (0u32, 0u32);
        for dy in 0..2 {
            for dx in 0..2 {
                let row = (cr * 2 + dy).min(h - 1);
                let col = (cc * 2 + dx).min(w - 1);
                let o = (row * w + col) * 3;
                let (_, pu, pv) = yuv::rgb_to_yuv(rgb[o], rgb[o + 1], rgb[o + 2], m);
                su += pu as u32;
                sv += pv as u32;
            }
        }
        (((su + 2) / 4) as u8, ((sv + 2) / 4) as u8)
    };
    for cc in 0..cw {
        let (eu, ev) = chroma_at(ch - 1, cc);
        assert_eq!(enc.planes[1].data[(ch - 1) * cw + cc], eu);
        assert_eq!(enc.planes[2].data[(ch - 1) * cw + cc], ev);
    }
    for cr in 0..ch {
        let (eu, ev) = chroma_at(cr, cw - 1);
        assert_eq!(enc.planes[1].data[cr * cw + cw - 1], eu);
        assert_eq!(enc.planes[2].data[cr * cw + cw - 1], ev);
    }
}

/// A `ColorRange` override replaces the format's range on the YUV side
/// and is a no-op when it restates the format.
#[test]
fn range_override_selects_the_matrix_range() {
    let mut rng = Rng(0x0463_0F11);
    let (w, h) = (9usize, 7usize);
    let src = source_frame(&mut rng, w, h, 2, 2, true, 0);
    let info = FrameInfo::new(PixelFormat::Yuva420P, w as u32, h as u32);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (y, u, v, a) = (
        tight(&src.planes[0], w, h),
        tight(&src.planes[1], cw, ch),
        tight(&src.planes[2], cw, ch),
        tight(&src.planes[3], w, h),
    );
    for cs in [
        ColorSpace::Bt601Limited,
        ColorSpace::Bt709Limited,
        ColorSpace::Bt2020Limited,
    ] {
        let o = opts(cs);
        let base = convert(&src, info, PixelFormat::Rgba, &o).unwrap();
        let same = convert_with(
            &src,
            info,
            PixelFormat::Rgba,
            &o,
            &ConvertContext::new().with_range(ColorRange::Limited),
        )
        .unwrap();
        assert!(base.planes[0].data == same.planes[0].data);
        let full = convert_with(
            &src,
            info,
            PixelFormat::Rgba,
            &o,
            &ConvertContext::new().with_range(ColorRange::Full),
        )
        .unwrap();
        let want = oracle_decode(&y, &u, &v, Some(&a), w, h, 2, 2, 4, matrix_for(cs, true));
        assert!(full.planes[0].data == want, "{cs:?} full override");
        assert!(full.planes[0].data != base.planes[0].data);
    }
}

/// Every planar family member → `Rgb48Le` / `Rgba64Le` now runs as one
/// banded deep-matrix pass. Pin it to the two-step route it replaces —
/// the exact widen to the 16-bit 4:4:4 alpha tier (`Yuva444P16Le`), then
/// the deep matrix — at even, odd and band-straddling geometry and at
/// several thread budgets.
#[test]
fn family_to_deep_packed_matches_the_staged_route() {
    use oxideav_pixfmt::FormatInfo;
    let members = [
        PixelFormat::Yuv420P,
        PixelFormat::Yuva420P,
        PixelFormat::Yuv422P,
        PixelFormat::Yuv444P,
        PixelFormat::Yuv440P,
        PixelFormat::Yuv420P10Le,
        PixelFormat::Yuv422P10Le,
        PixelFormat::Yuv444P10Le,
        PixelFormat::Yuva420P10Le,
        PixelFormat::Yuva444P12Le,
        PixelFormat::Yuv420P12Le,
        PixelFormat::Yuv440P12Le,
        PixelFormat::Yuv420P16Le,
        PixelFormat::Yuva422P16Le,
        PixelFormat::Yuv444P16Le,
    ];
    let mut rng = Rng(0x0463_DEE9);
    for fmt in members {
        let info = FormatInfo::of(fmt);
        let bits = info.bit_depth as u32;
        let sb = if bits > 8 { 2 } else { 1 };
        let (wsub, hsub) = (info.chroma_w_sub as usize, info.chroma_h_sub as usize);
        let geoms: &[(usize, usize)] = if cfg!(miri) {
            &[(7, 5), (1, 1)]
        } else {
            &[(4, 4), (7, 5), (3, 67), (1, 1)]
        };
        for &(w, h) in geoms {
            let (cw, ch) = (w.div_ceil(wsub), h.div_ceil(hsub));
            let mut mk = |pw: usize, ph: usize| {
                let mut p = plane(&mut rng, pw * sb, ph, 1);
                if sb == 2 {
                    // Keep words in range for the declared depth, but
                    // leave one stray high bit on 10/12-bit so the mask
                    // is exercised too.
                    for row in 0..ph {
                        for x in 0..pw {
                            let o = row * p.stride + x * 2;
                            let v = u16::from_le_bytes([p.data[o], p.data[o + 1]]);
                            let v = if bits < 16 && x == 0 {
                                v
                            } else {
                                v & ((1u32 << bits) - 1) as u16
                            };
                            p.data[o..o + 2].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
                p
            };
            let mut planes = vec![mk(w, h), mk(cw, ch), mk(cw, ch)];
            if info.has_alpha {
                planes.push(mk(w, h));
            }
            let src = VideoFrame { pts: None, planes };
            let si = FrameInfo::new(fmt, w as u32, h as u32);
            let o = opts(ColorSpace::Bt709Limited);
            let mid = convert(&src, si, PixelFormat::Yuva444P16Le, &o).expect("to 16-bit tier");
            // The historical second leg, evaluated with the unchanged
            // public 4:4:4 deep kernel on the widened planes.
            let mut rgb48 = vec![0u8; w * h * 6];
            yuv::yuv444p16_to_rgb48(
                &mid.planes[0].data,
                &mid.planes[1].data,
                &mid.planes[2].data,
                &mut rgb48,
                w,
                h,
                YuvMatrix::BT709,
            );
            for dst in [PixelFormat::Rgb48Le, PixelFormat::Rgba64Le] {
                let staged: Vec<u8> = if dst == PixelFormat::Rgb48Le {
                    rgb48.clone()
                } else {
                    rgb48
                        .chunks_exact(6)
                        .zip(mid.planes[3].data.chunks_exact(2))
                        .flat_map(|(c, a)| c.iter().chain(a.iter()).copied().collect::<Vec<u8>>())
                        .collect()
                };
                for threads in [1usize, 4] {
                    let direct = convert_with(
                        &src,
                        si,
                        dst,
                        &o,
                        &ConvertContext::new().with_threads(threads),
                    )
                    .expect("direct");
                    assert!(
                        direct.planes[0].data == staged,
                        "{fmt:?} {w}x{h} → {dst:?} ×{threads}"
                    );
                }
            }
        }
    }
}
