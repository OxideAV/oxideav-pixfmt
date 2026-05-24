//! Exact-roundtrip tests for the RGB-family swizzles and bit-depth
//! conversions. Every pair tested here must be lossless.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

fn synth_rgba(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
            data.push(((x + y) * 5) as u8);
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        },
        FrameInfo::new(PixelFormat::Rgba, w, h),
    )
}

fn synth_rgb24(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        },
        FrameInfo::new(PixelFormat::Rgb24, w, h),
    )
}

#[test]
fn rgb_family_4byte_roundtrips() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(32, 16);
    for fmt in [PixelFormat::Bgra, PixelFormat::Argb, PixelFormat::Abgr] {
        let stage = convert(&src, src_info, fmt, &opts).expect("swizzle");
        let stage_info = FrameInfo::new(fmt, src_info.width, src_info.height);
        let back = convert(&stage, stage_info, PixelFormat::Rgba, &opts).expect("swizzle back");
        assert_eq!(back.planes[0].data, src.planes[0].data, "roundtrip {fmt:?}");
    }
}

#[test]
fn rgb_family_3byte_roundtrips() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    let bgr = convert(&src, src_info, PixelFormat::Bgr24, &opts).unwrap();
    let bgr_info = FrameInfo::new(PixelFormat::Bgr24, src_info.width, src_info.height);
    let back = convert(&bgr, bgr_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgb24_to_rgba_and_back_preserves_colour() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(16, 8);
    let rgba = convert(&src, src_info, PixelFormat::Rgba, &opts).unwrap();
    let rgba_info = FrameInfo::new(PixelFormat::Rgba, src_info.width, src_info.height);
    let back = convert(&rgba, rgba_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgb48_rgb24_roundtrip() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(16, 8);
    let deep = convert(&src, src_info, PixelFormat::Rgb48Le, &opts).unwrap();
    let deep_info = FrameInfo::new(PixelFormat::Rgb48Le, src_info.width, src_info.height);
    let back = convert(&deep, deep_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgba64_rgba_roundtrip() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(16, 8);
    let deep = convert(&src, src_info, PixelFormat::Rgba64Le, &opts).unwrap();
    let deep_info = FrameInfo::new(PixelFormat::Rgba64Le, src_info.width, src_info.height);
    let back = convert(&deep, deep_info, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn gray8_gray16_roundtrip() {
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = Vec::with_capacity((w * h) as usize);
    for i in 0..(w * h) {
        data.push((i * 5) as u8);
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize,
            data,
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Gray8, w, h);
    let deep = convert(&src, src_info, PixelFormat::Gray16Le, &opts).unwrap();
    let deep_info = FrameInfo::new(PixelFormat::Gray16Le, w, h);
    let back = convert(&deep, deep_info, PixelFormat::Gray8, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn mono_black_gray8_roundtrip() {
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h) as usize];
    for (i, b) in data.iter_mut().enumerate() {
        *b = if i % 2 == 0 { 255 } else { 0 };
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize,
            data: data.clone(),
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Gray8, w, h);
    let mono = convert(&src, src_info, PixelFormat::MonoBlack, &opts).unwrap();
    let mono_info = FrameInfo::new(PixelFormat::MonoBlack, w, h);
    let back = convert(&mono, mono_info, PixelFormat::Gray8, &opts).unwrap();
    assert_eq!(back.planes[0].data, data);
}

#[test]
fn swizzle_all_four_byte_pairs() {
    // Every 4-byte ↔ 4-byte pair must roundtrip exactly.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgba(32, 16);
    let formats = [
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    for a in formats {
        for b in formats {
            if a == b {
                continue;
            }
            let frame_a = convert(&src, src_info, a, &opts).unwrap();
            let info_a = FrameInfo::new(a, src_info.width, src_info.height);
            let frame_b = convert(&frame_a, info_a, b, &opts).unwrap();
            let info_b = FrameInfo::new(b, src_info.width, src_info.height);
            let frame_back = convert(&frame_b, info_b, a, &opts).unwrap();
            assert_eq!(
                frame_a.planes[0].data, frame_back.planes[0].data,
                "a=Rgba stage={a:?} then {b:?}"
            );
        }
    }
}

#[test]
fn cmyk_roundtrip_via_rgb24() {
    // Rgb24 → Cmyk → Rgb24 is lossless at 8-bit precision by
    // construction of the formulas in the `cmyk` module.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(16, 8);
    let cmyk = convert(&src, src_info, PixelFormat::Cmyk, &opts).unwrap();
    assert_eq!(cmyk.planes[0].data.len(), 16 * 8 * 4);
    let cmyk_info = FrameInfo::new(PixelFormat::Cmyk, src_info.width, src_info.height);
    let back = convert(&cmyk, cmyk_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn cmyk_roundtrip_via_rgba() {
    // Rgba → Cmyk → Rgba. Alpha is dropped by Cmyk then restored
    // as opaque 255 on the way back, so the data matches only when
    // the source alpha was 255 to begin with.
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
            data.push(255);
        }
    }
    let src = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 4) as usize,
            data,
        }],
    };
    let src_info = FrameInfo::new(PixelFormat::Rgba, w, h);
    let cmyk = convert(&src, src_info, PixelFormat::Cmyk, &opts).unwrap();
    let cmyk_info = FrameInfo::new(PixelFormat::Cmyk, w, h);
    let back = convert(&cmyk, cmyk_info, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

// ---------------------------------------------------------------------
// Property / randomised round-trip suite. Same crate has hand-picked
// anchor tests above; the helpers below hammer every supported pair
// with a self-contained xorshift PRNG (no extra dep) to assert
// panic-freedom + structural invariants the spec/contract demands.

use oxideav_pixfmt::{premultiply, unpremultiply, ColorSpace};

struct PropRng(u64);
impl PropRng {
    fn new(seed: u64) -> Self {
        PropRng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
    }
}

fn prop_rand_packed(rng: &mut PropRng, w: u32, h: u32, bpp: usize, pad: usize) -> VideoFrame {
    let stride = w as usize * bpp + pad;
    let mut data = vec![0u8; stride * h as usize];
    for row in 0..h as usize {
        for b in 0..w as usize * bpp {
            data[row * stride + b] = rng.byte();
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

fn prop_rand_planar_yuv(rng: &mut PropRng, w: u32, h: u32, wsub: usize, hsub: usize) -> VideoFrame {
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = (w / wsub, h / hsub);
    let mut mk = |n: usize| {
        let mut v = vec![0u8; n];
        for b in v.iter_mut() {
            *b = rng.byte();
        }
        v
    };
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w,
                data: mk(w * h),
            },
            VideoPlane {
                stride: cw,
                data: mk(cw * ch),
            },
            VideoPlane {
                stride: cw,
                data: mk(cw * ch),
            },
        ],
    }
}

fn prop_assert_tight_eq(
    src: &VideoFrame,
    back: &VideoFrame,
    w: u32,
    h: u32,
    bpp: usize,
    case: u32,
    mid: PixelFormat,
) {
    let rb = w as usize * bpp;
    let ss = src.planes[0].stride;
    assert_eq!(back.planes[0].stride, rb);
    for row in 0..h as usize {
        let s = &src.planes[0].data[row * ss..row * ss + rb];
        let b = &back.planes[0].data[row * rb..row * rb + rb];
        assert_eq!(s, b, "case {case} via {mid:?} row {row}");
    }
}

#[test]
fn prop_rgb_family_swizzle_roundtrips_exactly() {
    let formats4 = [
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    let formats3 = [PixelFormat::Rgb24, PixelFormat::Bgr24];
    let opts = ConvertOptions::default();
    let mut rng = PropRng::new(0x5152_5354);
    for case in 0..200u32 {
        let w = rng.range(16, 32);
        let h = rng.range(1, 13);
        let pad = (rng.next_u64() % 4) as usize;
        let src = prop_rand_packed(&mut rng, w, h, 4, pad);
        let si = FrameInfo::new(PixelFormat::Rgba, w, h);
        for &mid in &formats4 {
            if mid == PixelFormat::Rgba {
                continue;
            }
            let stage = convert(&src, si, mid, &opts).unwrap();
            let mi = FrameInfo::new(mid, w, h);
            let back = convert(&stage, mi, PixelFormat::Rgba, &opts).unwrap();
            prop_assert_tight_eq(&src, &back, w, h, 4, case, mid);
        }
        let src3 = prop_rand_packed(&mut rng, w, h, 3, pad);
        let si3 = FrameInfo::new(PixelFormat::Rgb24, w, h);
        for &mid in &formats3 {
            if mid == PixelFormat::Rgb24 {
                continue;
            }
            let stage = convert(&src3, si3, mid, &opts).unwrap();
            let mi = FrameInfo::new(mid, w, h);
            let back = convert(&stage, mi, PixelFormat::Rgb24, &opts).unwrap();
            prop_assert_tight_eq(&src3, &back, w, h, 3, case, mid);
        }
    }
}

#[test]
fn prop_rgb_3to4_promote_demote_roundtrips_exactly() {
    let opts = ConvertOptions::default();
    let dst4 = [
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    let mut rng = PropRng::new(0xA1B2_C3D4);
    for case in 0..150u32 {
        let w = rng.range(16, 36);
        let h = rng.range(1, 11);
        let pad = (rng.next_u64() % 5) as usize;
        let src = prop_rand_packed(&mut rng, w, h, 3, pad);
        let si = FrameInfo::new(PixelFormat::Rgb24, w, h);
        for &four in &dst4 {
            let up = convert(&src, si, four, &opts).unwrap();
            let ui = FrameInfo::new(four, w, h);
            let back = convert(&up, ui, PixelFormat::Rgb24, &opts).unwrap();
            prop_assert_tight_eq(&src, &back, w, h, 3, case, four);
        }
    }
}

#[test]
fn prop_bit_depth_promote_demote_roundtrips_exactly() {
    let opts = ConvertOptions::default();
    let mut rng = PropRng::new(0x0BAD_F00D);
    for case in 0..150u32 {
        let w = rng.range(16, 40);
        let h = rng.range(1, 9);
        let pad = (rng.next_u64() % 4) as usize;

        let s = prop_rand_packed(&mut rng, w, h, 3, pad);
        let si = FrameInfo::new(PixelFormat::Rgb24, w, h);
        let deep = convert(&s, si, PixelFormat::Rgb48Le, &opts).unwrap();
        let di = FrameInfo::new(PixelFormat::Rgb48Le, w, h);
        let back = convert(&deep, di, PixelFormat::Rgb24, &opts).unwrap();
        prop_assert_tight_eq(&s, &back, w, h, 3, case, PixelFormat::Rgb48Le);

        let s = prop_rand_packed(&mut rng, w, h, 4, pad);
        let si = FrameInfo::new(PixelFormat::Rgba, w, h);
        let deep = convert(&s, si, PixelFormat::Rgba64Le, &opts).unwrap();
        let di = FrameInfo::new(PixelFormat::Rgba64Le, w, h);
        let back = convert(&deep, di, PixelFormat::Rgba, &opts).unwrap();
        prop_assert_tight_eq(&s, &back, w, h, 4, case, PixelFormat::Rgba64Le);

        let s = prop_rand_packed(&mut rng, w, h, 1, pad);
        let si = FrameInfo::new(PixelFormat::Gray8, w, h);
        let deep = convert(&s, si, PixelFormat::Gray16Le, &opts).unwrap();
        let di = FrameInfo::new(PixelFormat::Gray16Le, w, h);
        let back = convert(&deep, di, PixelFormat::Gray8, &opts).unwrap();
        prop_assert_tight_eq(&s, &back, w, h, 1, case, PixelFormat::Gray16Le);
    }
}

#[test]
fn prop_nv_yuv420p_interleave_roundtrips_exactly() {
    let opts = ConvertOptions::default();
    let mut rng = PropRng::new(0x1234_ABCD);
    for case in 0..100u32 {
        let w = rng.range(8, 24) * 2;
        let h = rng.range(1, 12) * 2;
        let yuv = prop_rand_planar_yuv(&mut rng, w, h, 2, 2);
        let yi = FrameInfo::new(PixelFormat::Yuv420P, w, h);
        for &nv in &[PixelFormat::Nv12, PixelFormat::Nv21] {
            let inter = convert(&yuv, yi, nv, &opts).unwrap();
            let ii = FrameInfo::new(nv, w, h);
            let back = convert(&inter, ii, PixelFormat::Yuv420P, &opts).unwrap();
            for p in 0..3 {
                assert_eq!(
                    yuv.planes[p].data, back.planes[p].data,
                    "case {case} via {nv:?} plane {p}"
                );
            }
        }
    }
}

#[test]
fn prop_yuvj_yuv_range_rescale_is_idempotent() {
    let opts = ConvertOptions::default();
    let mut rng = PropRng::new(0x7777_3333);
    for case in 0..80u32 {
        let w = rng.range(8, 20) * 2;
        let h = rng.range(1, 10) * 2;
        let yuv = prop_rand_planar_yuv(&mut rng, w, h, 2, 2);
        let yi = FrameInfo::new(PixelFormat::Yuv420P, w, h);
        let j = convert(&yuv, yi, PixelFormat::YuvJ420P, &opts).unwrap();
        let ji = FrameInfo::new(PixelFormat::YuvJ420P, w, h);
        let back = convert(&j, ji, PixelFormat::Yuv420P, &opts).unwrap();
        let j2 = convert(&back, yi, PixelFormat::YuvJ420P, &opts).unwrap();
        for p in 0..3 {
            for (a, b) in j.planes[p].data.iter().zip(j2.planes[p].data.iter()) {
                assert!(
                    a.abs_diff(*b) <= 2,
                    "case {case} plane {p} rescale not idempotent: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn prop_no_panic_over_supported_pairs() {
    // Every entry in the conversion table, fed random in-spec buffers
    // (including non-tight strides), must return Ok or a clean Err —
    // never panic. Palette pairs go through tests/palette.rs (they
    // need a populated ConvertOptions.palette). Widths start at 16
    // so vectorised paths see a full vector ahead of the scalar tail.
    let opts = ConvertOptions::default();
    let mut rng = PropRng::new(0xDEAD_C0DE);
    let packed_pairs: &[(PixelFormat, usize, PixelFormat)] = &[
        (PixelFormat::Rgb24, 3, PixelFormat::Bgr24),
        (PixelFormat::Rgba, 4, PixelFormat::Bgra),
        (PixelFormat::Rgba, 4, PixelFormat::Argb),
        (PixelFormat::Rgb24, 3, PixelFormat::Rgba),
        (PixelFormat::Rgba, 4, PixelFormat::Rgb24),
        (PixelFormat::Rgb48Le, 6, PixelFormat::Rgb24),
        (PixelFormat::Rgba64Le, 8, PixelFormat::Rgba),
        (PixelFormat::Gray8, 1, PixelFormat::Rgb24),
        (PixelFormat::Gray8, 1, PixelFormat::Rgba),
        (PixelFormat::Gray16Le, 2, PixelFormat::Gray8),
        (PixelFormat::Cmyk, 4, PixelFormat::Rgb24),
        (PixelFormat::Cmyk, 4, PixelFormat::Rgba),
        (PixelFormat::Rgb24, 3, PixelFormat::Cmyk),
    ];
    for case in 0..60u32 {
        let w = rng.range(16, 40);
        let h = rng.range(1, 17);
        let pad = (rng.next_u64() % 6) as usize;
        for &(src_fmt, bpp, dst_fmt) in packed_pairs {
            let f = prop_rand_packed(&mut rng, w, h, bpp, pad);
            let fi = FrameInfo::new(src_fmt, w, h);
            let _ = convert(&f, fi, dst_fmt, &opts);
            let _ = case;
        }
    }
    for _ in 0..40u32 {
        let w = rng.range(8, 20) * 2;
        let h = rng.range(1, 10) * 2;
        for (fmt, wsub, hsub) in [
            (PixelFormat::Yuv420P, 2, 2),
            (PixelFormat::Yuv422P, 2, 1),
            (PixelFormat::Yuv444P, 1, 1),
        ] {
            let yuv = prop_rand_planar_yuv(&mut rng, w, h, wsub, hsub);
            let yi = FrameInfo::new(fmt, w, h);
            let _ = convert(&yuv, yi, PixelFormat::Rgb24, &opts);
            let _ = convert(&yuv, yi, PixelFormat::Rgba, &opts);
        }
        let rgb = prop_rand_packed(&mut rng, w, h, 3, 0);
        let ri = FrameInfo::new(PixelFormat::Rgb24, w, h);
        for dst in [
            PixelFormat::Yuv420P,
            PixelFormat::Yuv422P,
            PixelFormat::Yuv444P,
        ] {
            let _ = convert(&rgb, ri, dst, &opts);
        }
    }
}

#[test]
fn prop_rgb_to_yuv_odd_dims_error_not_panic() {
    let opts = ConvertOptions::default();
    let mut rng = PropRng::new(0x0DD0_0DD0);
    for _ in 0..60u32 {
        let w = rng.range(8, 20) * 2 - 1; // odd
        let h = rng.range(8, 20) * 2 - 1; // odd
        let rgb = prop_rand_packed(&mut rng, w, h, 3, 0);
        let ri = FrameInfo::new(PixelFormat::Rgb24, w, h);
        let res = convert(&rgb, ri, PixelFormat::Yuv420P, &opts);
        assert!(res.is_err(), "odd {w}x{h} -> Yuv420P should error");
    }
}

#[test]
fn prop_premultiply_unpremultiply_bounded_by_alpha() {
    let mut rng = PropRng::new(0xBEEF_CAFE);
    let mut worst_high = 0u8;
    for _ in 0..100_000u32 {
        let r = rng.byte();
        let g = rng.byte();
        let b = rng.byte();
        let a = rng.byte();
        let round = unpremultiply(premultiply([r, g, b, a]));
        assert_eq!(round[3], a, "alpha must survive exactly");
        if a == 0 {
            assert_eq!(round, [0, 0, 0, 0], "a=0 must clear colour");
            continue;
        }
        let bound = 255u32.div_ceil(a as u32) as u8;
        for (ch, &orig) in [r, g, b].iter().enumerate() {
            let diff = round[ch].abs_diff(orig);
            assert!(
                diff <= bound,
                "channel {ch} drift {diff} > bound {bound} for ({r},{g},{b},{a})"
            );
            if a >= 128 {
                worst_high = worst_high.max(diff);
            }
        }
        if a == 255 {
            assert_eq!(&round[..3], &[r, g, b], "a=255 must be exact");
        }
    }
    println!("premul roundtrip worst diff for a>=128 = {worst_high}");
    assert!(
        worst_high <= 1,
        "high-alpha roundtrip drifted by {worst_high}"
    );
}

/// Per-channel bound for one RGB→YUV→RGB hop through the Q15 matrices
/// at 4:4:4. Empirical worst across 300 000 random pixels × 3 matrices
/// × 2 ranges is 2 LSB; ceiling pinned at 3 (1 LSB headroom).
const PROP_YUV444_MAX_CH_ERR: u8 = 3;

#[test]
fn prop_rgb_yuv444_roundtrip_per_pixel_bounded() {
    use oxideav_pixfmt::yuv::{rgb_to_yuv, yuv_to_rgb, YuvMatrix};
    let mut rng = PropRng::new(0xFEED_BEEF);
    let mut worst = 0u8;
    for m in [YuvMatrix::BT601, YuvMatrix::BT709, YuvMatrix::BT2020] {
        for limited in [true, false] {
            let mat = m.with_range(limited);
            for _ in 0..50_000u32 {
                let r = rng.byte();
                let g = rng.byte();
                let b = rng.byte();
                let (y, cb, cr) = rgb_to_yuv(r, g, b, mat);
                let (r2, g2, b2) = yuv_to_rgb(y, cb, cr, mat);
                let er = r.abs_diff(r2);
                let eg = g.abs_diff(g2);
                let eb = b.abs_diff(b2);
                worst = worst.max(er).max(eg).max(eb);
                assert!(
                    er <= PROP_YUV444_MAX_CH_ERR
                        && eg <= PROP_YUV444_MAX_CH_ERR
                        && eb <= PROP_YUV444_MAX_CH_ERR,
                    "rgb({r},{g},{b}) -> yuv({y},{cb},{cr}) -> rgb({r2},{g2},{b2}); \
                     err=({er},{eg},{eb}) limited={limited}"
                );
            }
        }
    }
    println!(
        "yuv444 per-pixel worst per-channel error = {worst} (ceiling {PROP_YUV444_MAX_CH_ERR})"
    );
    assert!(worst <= PROP_YUV444_MAX_CH_ERR);
}

#[test]
fn prop_full_frame_yuv_roundtrip_psnr_floor() {
    fn gradient_rgb24(w: u32, h: u32) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(((x * 255) / (w - 1).max(1)) as u8);
                data.push(((y * 255) / (h - 1).max(1)) as u8);
                data.push((((x + y) * 255) / ((w + h) - 2).max(1)) as u8);
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        }
    }
    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut sq = 0.0f64;
        for i in 0..a.len() {
            let d = a[i] as f64 - b[i] as f64;
            sq += d * d;
        }
        if sq == 0.0 {
            return f64::INFINITY;
        }
        let mse = sq / a.len() as f64;
        10.0 * (255.0 * 255.0 / mse).log10()
    }
    let mut rng = PropRng::new(0xC0FF_EE11);
    for cs in [
        ColorSpace::Bt601Limited,
        ColorSpace::Bt601Full,
        ColorSpace::Bt709Limited,
        ColorSpace::Bt709Full,
        ColorSpace::Bt2020Limited,
        ColorSpace::Bt2020Full,
    ] {
        let opts = ConvertOptions {
            color_space: cs,
            ..Default::default()
        };
        for _ in 0..4u32 {
            let w = rng.range(8, 32) * 2;
            let h = rng.range(8, 24) * 2;
            let src = gradient_rgb24(w, h);
            let si = FrameInfo::new(PixelFormat::Rgb24, w, h);
            for (fmt, floor) in [
                (PixelFormat::Yuv444P, 36.0f64),
                (PixelFormat::Yuv422P, 32.0),
                (PixelFormat::Yuv420P, 29.0),
            ] {
                let yuv = convert(&src, si, fmt, &opts).unwrap();
                let yi = FrameInfo::new(fmt, w, h);
                let back = convert(&yuv, yi, PixelFormat::Rgb24, &opts).unwrap();
                let psnr = psnr(&src.planes[0].data, &back.planes[0].data);
                assert!(
                    psnr > floor,
                    "{cs:?} {fmt:?} {w}x{h} psnr {psnr:.2} below floor {floor}"
                );
            }
        }
    }
}
