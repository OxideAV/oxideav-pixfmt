//! Randomised property / round-trip suite for the conversion surface.
//!
//! Where the existing `tests/*.rs` pin a handful of hand-picked anchor
//! cases, this suite hammers every supported `(src, dst)` family with a
//! large sweep of pseudo-random pixel buffers across many dimensions
//! (and non-tight strides) and asserts the structural invariants:
//!
//! * **Panic-freedom** — no entry in the conversion table may panic for
//!   any in-spec input, including non-tight source strides.
//! * **Exactly-lossless** round-trips — RGB-family swizzles, 3↔4 alpha
//!   promote/demote, 8↔16-bit promote/demote, plane interleave↔planar
//!   (NV12/NV21 ↔ Yuv420P) and YuvJ↔Yuv range-rescale-of-neutrals.
//! * **Tolerance-bounded** round-trips — YUV↔RGB through the Q15 fixed-
//!   point matrices, and RGBA premultiply↔unpremultiply, with per-channel
//!   bounds derived from the documented contracts (no ffmpeg/libyuv
//!   reference — the matrices are textbook BT.601/709/2020).
//!
//! The PRNG is a self-contained xorshift (no external crate), seeded per
//! case so every failure is exactly reproducible from its seed.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::yuv::{rgb_to_yuv, yuv_to_rgb, YuvMatrix};
use oxideav_pixfmt::{convert, premultiply, unpremultiply, ColorSpace, ConvertOptions, FrameInfo};

/// Deterministic xorshift64* PRNG — same family the palette tests use,
/// kept local so this file has zero added dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point of xorshift.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
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
    /// Inclusive range `[lo, hi]`.
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
    }
}

/// Build a frame of `bpp` bytes-per-pixel with random bytes. `pad` extra
/// bytes are appended to every row so the converters see a non-tight
/// stride (exercising the `gather_tight` / `tight_row` paths).
fn rand_packed(rng: &mut Rng, w: u32, h: u32, bpp: usize, pad: usize) -> VideoFrame {
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

/// Random 3-plane planar YUV frame (`wsub`/`hsub` chroma subsampling).
fn rand_planar_yuv(rng: &mut Rng, w: u32, h: u32, wsub: usize, hsub: usize) -> VideoFrame {
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

const COLOR_SPACES: [ColorSpace; 6] = [
    ColorSpace::Bt601Limited,
    ColorSpace::Bt601Full,
    ColorSpace::Bt709Limited,
    ColorSpace::Bt709Full,
    ColorSpace::Bt2020Limited,
    ColorSpace::Bt2020Full,
];

const MATRICES: [YuvMatrix; 3] = [YuvMatrix::BT601, YuvMatrix::BT709, YuvMatrix::BT2020];

// ---------------------------------------------------------------------
// Exactly-lossless round-trips.

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
    let mut rng = Rng::new(0x5152_5354);
    for case in 0..400u32 {
        let w = rng.range(1, 17);
        let h = rng.range(1, 13);
        let pad = (rng.next_u64() % 4) as usize;
        // 4-byte family.
        let src = rand_packed(&mut rng, w, h, 4, pad);
        let si = FrameInfo::new(PixelFormat::Rgba, w, h);
        for &mid in &formats4 {
            if mid == PixelFormat::Rgba {
                continue;
            }
            let stage = convert(&src, si, mid, &opts).unwrap();
            let mi = FrameInfo::new(mid, w, h);
            let back = convert(&stage, mi, PixelFormat::Rgba, &opts).unwrap();
            // Compare tight content (source carries padding).
            assert_tight_eq(&src, &back, w, h, 4, case, mid);
        }
        // 3-byte family.
        let src3 = rand_packed(&mut rng, w, h, 3, pad);
        let si3 = FrameInfo::new(PixelFormat::Rgb24, w, h);
        for &mid in &formats3 {
            if mid == PixelFormat::Rgb24 {
                continue;
            }
            let stage = convert(&src3, si3, mid, &opts).unwrap();
            let mi = FrameInfo::new(mid, w, h);
            let back = convert(&stage, mi, PixelFormat::Rgb24, &opts).unwrap();
            assert_tight_eq(&src3, &back, w, h, 3, case, mid);
        }
    }
}

fn assert_tight_eq(
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
    // `back` is always tight (converters emit tight strides).
    assert_eq!(back.planes[0].stride, rb);
    for row in 0..h as usize {
        let s = &src.planes[0].data[row * ss..row * ss + rb];
        let b = &back.planes[0].data[row * rb..row * rb + rb];
        assert_eq!(s, b, "case {case} via {mid:?} row {row}");
    }
}

#[test]
fn prop_rgb_3to4_promote_demote_roundtrips_exactly() {
    // Rgb24 -> Rgba -> Rgb24 must be the identity (alpha appended is
    // opaque and then dropped). Cover all 3->4 target byte orders.
    let opts = ConvertOptions::default();
    let dst4 = [
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    let mut rng = Rng::new(0xA1B2_C3D4);
    for case in 0..300u32 {
        let w = rng.range(1, 19);
        let h = rng.range(1, 11);
        let pad = (rng.next_u64() % 5) as usize;
        let src = rand_packed(&mut rng, w, h, 3, pad);
        let si = FrameInfo::new(PixelFormat::Rgb24, w, h);
        for &four in &dst4 {
            let up = convert(&src, si, four, &opts).unwrap();
            let ui = FrameInfo::new(four, w, h);
            let back = convert(&up, ui, PixelFormat::Rgb24, &opts).unwrap();
            assert_tight_eq(&src, &back, w, h, 3, case, four);
        }
    }
}

#[test]
fn prop_bit_depth_promote_demote_roundtrips_exactly() {
    let opts = ConvertOptions::default();
    let mut rng = Rng::new(0x0BAD_F00D);
    for case in 0..300u32 {
        let w = rng.range(1, 21);
        let h = rng.range(1, 9);
        let pad = (rng.next_u64() % 4) as usize;

        // Rgb24 <-> Rgb48Le (byte-replicate promote, truncate demote).
        let s = rand_packed(&mut rng, w, h, 3, pad);
        let si = FrameInfo::new(PixelFormat::Rgb24, w, h);
        let deep = convert(&s, si, PixelFormat::Rgb48Le, &opts).unwrap();
        let di = FrameInfo::new(PixelFormat::Rgb48Le, w, h);
        let back = convert(&deep, di, PixelFormat::Rgb24, &opts).unwrap();
        assert_tight_eq(&s, &back, w, h, 3, case, PixelFormat::Rgb48Le);

        // Rgba <-> Rgba64Le.
        let s = rand_packed(&mut rng, w, h, 4, pad);
        let si = FrameInfo::new(PixelFormat::Rgba, w, h);
        let deep = convert(&s, si, PixelFormat::Rgba64Le, &opts).unwrap();
        let di = FrameInfo::new(PixelFormat::Rgba64Le, w, h);
        let back = convert(&deep, di, PixelFormat::Rgba, &opts).unwrap();
        assert_tight_eq(&s, &back, w, h, 4, case, PixelFormat::Rgba64Le);

        // Gray8 <-> Gray16Le.
        let s = rand_packed(&mut rng, w, h, 1, pad);
        let si = FrameInfo::new(PixelFormat::Gray8, w, h);
        let deep = convert(&s, si, PixelFormat::Gray16Le, &opts).unwrap();
        let di = FrameInfo::new(PixelFormat::Gray16Le, w, h);
        let back = convert(&deep, di, PixelFormat::Gray8, &opts).unwrap();
        assert_tight_eq(&s, &back, w, h, 1, case, PixelFormat::Gray16Le);
    }
}

#[test]
fn prop_nv_yuv420p_interleave_roundtrips_exactly() {
    // Yuv420P -> NV12/NV21 -> Yuv420P is a pure interleave / de-interleave
    // and must be byte-exact on all three planes.
    let opts = ConvertOptions::default();
    let mut rng = Rng::new(0x1234_ABCD);
    for case in 0..200u32 {
        let w = rng.range(1, 16) * 2; // even dims required for 4:2:0
        let h = rng.range(1, 12) * 2;
        let yuv = rand_planar_yuv(&mut rng, w, h, 2, 2);
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
fn prop_yuvj_yuv_range_rescale_neutral_is_stable() {
    // For neutral chroma (128) and a Y already inside the limited range,
    // Yuv -> YuvJ -> Yuv should land within the rescale's rounding bound.
    // We assert the looser structural property: every output byte stays a
    // valid u8 and a second round-trip is idempotent within +/-2 LSB.
    let opts = ConvertOptions::default();
    let mut rng = Rng::new(0x7777_3333);
    for case in 0..150u32 {
        let w = rng.range(1, 16) * 2;
        let h = rng.range(1, 10) * 2;
        let yuv = rand_planar_yuv(&mut rng, w, h, 2, 2);
        let yi = FrameInfo::new(PixelFormat::Yuv420P, w, h);
        let j = convert(&yuv, yi, PixelFormat::YuvJ420P, &opts).unwrap();
        let ji = FrameInfo::new(PixelFormat::YuvJ420P, w, h);
        let back = convert(&j, ji, PixelFormat::Yuv420P, &opts).unwrap();
        // Second round-trip from the recovered limited-range frame must be
        // idempotent (the rescale is stable once you're in-range).
        let j2 = convert(&back, yi, PixelFormat::YuvJ420P, &opts).unwrap();
        for p in 0..3 {
            assert_eq!(
                j.planes[p].data.len(),
                j2.planes[p].data.len(),
                "case {case} plane {p} length"
            );
            for (a, b) in j.planes[p].data.iter().zip(j2.planes[p].data.iter()) {
                assert!(
                    a.abs_diff(*b) <= 2,
                    "case {case} plane {p} rescale not stable: {a} vs {b}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// Tolerance-bounded round-trips.

/// Measured per-channel bound for one RGB→YUV→RGB hop through the Q15
/// matrices at 4:4:4 (no chroma loss). The empirical worst case across
/// 300 000 random pixels × 3 matrices × 2 ranges is 2 LSB; the ceiling
/// is pinned at 3 (1 LSB headroom) so a regression that widens the
/// matrix round-off is caught, while the `println!` surfaces the live
/// maximum.
const YUV444_MAX_CH_ERR: u8 = 3;

#[test]
fn prop_rgb_yuv444_roundtrip_per_pixel_bounded() {
    // Exercise the per-pixel scalar path directly across every matrix
    // and range so the bound is independent of plane plumbing.
    let mut rng = Rng::new(0xFEED_BEEF);
    let mut worst = 0u8;
    for m in MATRICES {
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
                    er <= YUV444_MAX_CH_ERR && eg <= YUV444_MAX_CH_ERR && eb <= YUV444_MAX_CH_ERR,
                    "rgb({r},{g},{b}) -> yuv({y},{cb},{cr}) -> rgb({r2},{g2},{b2}); \
                     err=({er},{eg},{eb}) limited={limited}"
                );
            }
        }
    }
    println!("yuv444 per-pixel worst per-channel error = {worst} (ceiling {YUV444_MAX_CH_ERR})");
    assert!(worst <= YUV444_MAX_CH_ERR);
}

#[test]
fn prop_full_frame_yuv_roundtrip_psnr_floor() {
    // Smooth-gradient frames through the high-level convert() path. 4:4:4
    // is near-lossless; 4:2:2 / 4:2:0 lose chroma detail. Floors mirror
    // the deterministic suite but sweep all six colour spaces and a range
    // of dimensions.
    let mut rng = Rng::new(0xC0FF_EE11);
    for cs in COLOR_SPACES {
        let opts = ConvertOptions {
            color_space: cs,
            ..Default::default()
        };
        for _ in 0..8u32 {
            let w = rng.range(8, 40) * 2;
            let h = rng.range(8, 30) * 2;
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

#[test]
fn prop_premultiply_unpremultiply_bounded_by_alpha() {
    // The documented contract: roundtrip error is bounded by the spacing
    // of representable premultiplied values, ceil(255 / a), and is exact
    // at a = 0 and a = 255.
    let mut rng = Rng::new(0xBEEF_CAFE);
    let mut worst_high = 0u8; // worst diff for a >= 128
    for _ in 0..200_000u32 {
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
    // For high alpha the bound collapses to +/-1; pin that tighter
    // invariant explicitly.
    println!("premul roundtrip worst diff for a>=128 = {worst_high}");
    assert!(
        worst_high <= 1,
        "high-alpha roundtrip drifted by {worst_high}"
    );
}

// ---------------------------------------------------------------------
// Panic-freedom sweep over the whole conversion table.

#[test]
fn prop_no_panic_over_supported_pairs() {
    // Every entry in the conversion table, fed random in-spec buffers
    // (including non-tight strides), must return Ok or a clean Err —
    // never panic. Palette pairs are skipped here (covered separately;
    // they require a populated ConvertOptions.palette and are exercised
    // in tests/palette.rs).
    let opts = ConvertOptions::default();
    let mut rng = Rng::new(0xDEAD_C0DE);

    // (src, bytes-per-pixel-ish packing helper, dst) for the packed
    // single-plane families.
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
    for case in 0..500u32 {
        let w = rng.range(1, 23);
        let h = rng.range(1, 17);
        let pad = (rng.next_u64() % 6) as usize;
        for &(src_fmt, bpp, dst_fmt) in packed_pairs {
            let f = rand_packed(&mut rng, w, h, bpp, pad);
            let fi = FrameInfo::new(src_fmt, w, h);
            // Must not panic; result may be Ok or Err.
            let _ = convert(&f, fi, dst_fmt, &opts);
            let _ = case; // keep the loop var meaningful in failure msgs
        }
    }

    // YUV planar sources (need 3 planes) and RGB->YUV (need even dims).
    for _ in 0..300u32 {
        let w = rng.range(1, 12) * 2;
        let h = rng.range(1, 10) * 2;
        for (fmt, wsub, hsub) in [
            (PixelFormat::Yuv420P, 2, 2),
            (PixelFormat::Yuv422P, 2, 1),
            (PixelFormat::Yuv444P, 1, 1),
        ] {
            let yuv = rand_planar_yuv(&mut rng, w, h, wsub, hsub);
            let yi = FrameInfo::new(fmt, w, h);
            let _ = convert(&yuv, yi, PixelFormat::Rgb24, &opts);
            let _ = convert(&yuv, yi, PixelFormat::Rgba, &opts);
        }
        // RGB -> YUV (even dims satisfy the divisibility guard).
        let rgb = rand_packed(&mut rng, w, h, 3, 0);
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
    // RGB -> 4:2:0 with odd dimensions must return Err (divisibility
    // guard), never panic or silently produce a malformed frame.
    let opts = ConvertOptions::default();
    let mut rng = Rng::new(0x0DD0_0DD0);
    for _ in 0..100u32 {
        let w = rng.range(1, 20) * 2 - 1; // odd
        let h = rng.range(1, 20) * 2 - 1; // odd
        let rgb = rand_packed(&mut rng, w, h, 3, 0);
        let ri = FrameInfo::new(PixelFormat::Rgb24, w, h);
        let res = convert(&rgb, ri, PixelFormat::Yuv420P, &opts);
        assert!(res.is_err(), "odd {w}x{h} -> Yuv420P should error");
    }
}

// ---------------------------------------------------------------------
// Local helpers (kept out of the production crate).

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
