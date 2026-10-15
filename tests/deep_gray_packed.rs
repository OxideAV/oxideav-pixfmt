//! Precision gates for the deep grey ↔ packed deep RGB rows (round
//! 476): `Gray10Le` / `Gray12Le` / `Gray16Le` ↔ `Rgb48Le` / `Rgba64Le`
//! and `Ya16Le` ↔ the 10 / 12-bit grey words.
//!
//! * a full 16-bit ramp survives `Gray16Le → Rgb48Le → Gray16Le`
//!   bit-exact;
//! * `Rgb48Le → Gray16Le` is the 16-bit Y' row of the selected matrix
//!   (BT.601 / BT.709 / BT.2020 through `ConvertOptions::color_space`,
//!   the same selection the 8-bit `Rgb24 → Gray8` row uses) within
//!   1 LSB of an f64 reference over a structured sweep and 10^6 random
//!   samples;
//! * the 10 / 12-bit ends follow the crate's ladder convention (MSB
//!   replication out, truncation in);
//! * the rows are direct (no pivot) and byte-identical serial vs
//!   threaded.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, convert_with, staged_pivot, supports_direct, ColorSpace, ConvertContext,
    ConvertOptions, FrameInfo,
};

fn le16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn rd16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn frame(stride: usize, data: Vec<u8>) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

fn conv(src: &VideoFrame, sf: PixelFormat, w: u32, h: u32, df: PixelFormat) -> VideoFrame {
    convert(
        src,
        FrameInfo::new(sf, w, h),
        df,
        &ConvertOptions::default(),
    )
    .unwrap()
}

/// Deterministic pseudo-random generator (xorshift64*).
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

#[test]
fn deep_gray_packed_rows_are_direct() {
    use PixelFormat as P;
    for g in [P::Gray10Le, P::Gray12Le, P::Gray16Le] {
        for rgb in [P::Rgb48Le, P::Rgba64Le] {
            assert!(supports_direct(g, rgb), "{g:?} → {rgb:?}");
            assert!(supports_direct(rgb, g), "{rgb:?} → {g:?}");
            assert_eq!(staged_pivot(g, rgb), None);
            assert_eq!(staged_pivot(rgb, g), None);
        }
        assert!(supports_direct(g, P::Ya16Le));
        assert!(supports_direct(P::Ya16Le, g));
    }
}

/// Every one of the 65536 grey codes survives the broadcast and the
/// projection, on every matrix.
#[test]
fn gray16_ramp_round_trips_through_rgb48_bit_exact() {
    let (w, h) = (256u32, 256u32);
    let ramp: Vec<u16> = (0..=u16::MAX).collect();
    let src = frame(w as usize * 2, le16(&ramp));
    for cs in [
        ColorSpace::Bt601Limited,
        ColorSpace::Bt709Full,
        ColorSpace::Bt2020Limited,
    ] {
        let opts = ConvertOptions {
            color_space: cs,
            ..Default::default()
        };
        for mid in [PixelFormat::Rgb48Le, PixelFormat::Rgba64Le] {
            let rgb = convert(
                &src,
                FrameInfo::new(PixelFormat::Gray16Le, w, h),
                mid,
                &opts,
            )
            .unwrap();
            let words = rd16(&rgb.planes[0].data);
            let comps = if mid == PixelFormat::Rgba64Le { 4 } else { 3 };
            for (i, &v) in ramp.iter().enumerate() {
                assert_eq!(&words[i * comps..i * comps + 3], &[v, v, v]);
                if comps == 4 {
                    assert_eq!(words[i * comps + 3], 0xFFFF);
                }
            }
            let back = convert(
                &rgb,
                FrameInfo::new(mid, w, h),
                PixelFormat::Gray16Le,
                &opts,
            )
            .unwrap();
            assert_eq!(rd16(&back.planes[0].data), ramp, "{cs:?} via {mid:?}");
        }
    }
}

fn kr_kb(cs: ColorSpace) -> (f64, f64) {
    match cs {
        ColorSpace::Bt601Limited | ColorSpace::Bt601Full => (0.299, 0.114),
        ColorSpace::Bt709Limited | ColorSpace::Bt709Full => (0.2126, 0.0722),
        ColorSpace::Bt2020Limited | ColorSpace::Bt2020Full => (0.2627, 0.0593),
    }
}

/// f64 reference: the full-range Y' row of the selected primaries at
/// 16-bit scale, rounded to nearest.
fn luma_ref(r: u16, g: u16, b: u16, cs: ColorSpace) -> f64 {
    let (kr, kb) = kr_kb(cs);
    let kg = 1.0 - kr - kb;
    kr * r as f64 + kg * g as f64 + kb * b as f64
}

fn check_luma(pixels: &[(u16, u16, u16)], cs: ColorSpace, alpha: bool) {
    let w = 1024u32;
    let h = (pixels.len() as u32).div_ceil(w);
    let n = (w * h) as usize;
    let comps = if alpha { 4 } else { 3 };
    let mut words = vec![0u16; n * comps];
    for (i, &(r, g, b)) in pixels.iter().enumerate() {
        words[i * comps] = r;
        words[i * comps + 1] = g;
        words[i * comps + 2] = b;
        if alpha {
            // Alpha must never leak into the projection.
            words[i * comps + 3] = (i as u16).wrapping_mul(40503);
        }
    }
    let sf = if alpha {
        PixelFormat::Rgba64Le
    } else {
        PixelFormat::Rgb48Le
    };
    let src = frame(w as usize * comps * 2, le16(&words));
    let opts = ConvertOptions {
        color_space: cs,
        ..Default::default()
    };
    let out = convert(&src, FrameInfo::new(sf, w, h), PixelFormat::Gray16Le, &opts).unwrap();
    let got = rd16(&out.planes[0].data);
    let mut max_err = 0.0f64;
    for (i, &(r, g, b)) in pixels.iter().enumerate() {
        let want = luma_ref(r, g, b, cs);
        let err = (got[i] as f64 - want).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err <= 1.0,
            "{sf:?} {cs:?} pixel {i} ({r}, {g}, {b}): got {} want {want:.4}",
            got[i]
        );
        // Exact rounding whenever the reference is clear of a half-code
        // boundary (the weights are f32, about 1e-3 at 16-bit scale).
        if (want - want.round()).abs() < 0.49 {
            assert_eq!(got[i] as f64, want.round(), "{sf:?} {cs:?} pixel {i}");
        }
    }
    assert!(max_err <= 1.0, "{sf:?} {cs:?} max error {max_err}");
}

/// 4096-sample structured sweep: the primaries / secondaries / grey
/// rails at every 16th code plus the 8-bit-aligned cube corners.
fn structured_sweep() -> Vec<(u16, u16, u16)> {
    let mut v = Vec::with_capacity(4096);
    let rails: [(u16, u16, u16); 7] = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ];
    for step in 0..=511u32 {
        let code = ((step * 65535) / 511) as u16;
        for &(r, g, b) in &rails {
            v.push((r * code, g * code, b * code));
        }
    }
    // Cube corners at 8-bit-aligned codes (×257) for the remainder.
    let mut k = 0u32;
    while v.len() < 4096 {
        let r = ((k * 37) % 256) as u16 * 257;
        let g = ((k * 91) % 256) as u16 * 257;
        let b = ((k * 173) % 256) as u16 * 257;
        v.push((r, g, b));
        k += 1;
    }
    v.truncate(4096);
    v
}

#[test]
fn rgb48_to_gray16_matches_f64_luma_structured() {
    let sweep = structured_sweep();
    for cs in [
        ColorSpace::Bt601Limited,
        ColorSpace::Bt709Limited,
        ColorSpace::Bt2020Limited,
    ] {
        check_luma(&sweep, cs, false);
        check_luma(&sweep, cs, true);
    }
}

#[test]
fn rgb48_to_gray16_matches_f64_luma_random_million() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let samples: Vec<(u16, u16, u16)> = (0..1_000_000)
        .map(|_| (rng.u16(), rng.u16(), rng.u16()))
        .collect();
    for cs in [
        ColorSpace::Bt601Full,
        ColorSpace::Bt709Full,
        ColorSpace::Bt2020Full,
    ] {
        check_luma(&samples, cs, false);
    }
}

/// The full-range / limited-range halves of `ColorSpace` select the
/// same Y' row: Gray16Le is a full-range space whatever the YUV
/// convention of the variant (the 8-bit rule, at depth).
#[test]
fn gray16_projection_ignores_the_range_half() {
    let (w, h) = (64u32, 64u32);
    let mut rng = Rng(7);
    let words: Vec<u16> = (0..(w * h * 3) as usize).map(|_| rng.u16()).collect();
    let src = frame(w as usize * 6, le16(&words));
    for (a, b) in [
        (ColorSpace::Bt601Limited, ColorSpace::Bt601Full),
        (ColorSpace::Bt709Limited, ColorSpace::Bt709Full),
        (ColorSpace::Bt2020Limited, ColorSpace::Bt2020Full),
    ] {
        let run = |cs| {
            let opts = ConvertOptions {
                color_space: cs,
                ..Default::default()
            };
            convert(
                &src,
                FrameInfo::new(PixelFormat::Rgb48Le, w, h),
                PixelFormat::Gray16Le,
                &opts,
            )
            .unwrap()
            .planes[0]
                .data
                .clone()
        };
        assert_eq!(run(a), run(b));
    }
}

/// `Gray10Le` / `Gray12Le` → packed: the word widened by MSB
/// replication (peak → 65535) and broadcast; packed → `Gray10Le` /
/// `Gray12Le`: the 16-bit projection truncated to the target width.
#[test]
fn ten_and_twelve_bit_ends_follow_the_ladder_convention() {
    let (w, h) = (64u32, 16u32);
    let n = (w * h) as usize;
    for (bits, fmt) in [(10u32, PixelFormat::Gray10Le), (12, PixelFormat::Gray12Le)] {
        let mask = (1u16 << bits) - 1;
        let codes: Vec<u16> = (0..n).map(|i| ((i * 977) as u16) & mask).collect();
        let src = frame(w as usize * 2, le16(&codes));
        let widen = |v: u16| -> u16 { (v << (16 - bits)) | (v >> (2 * bits - 16)) };
        for packed in [PixelFormat::Rgb48Le, PixelFormat::Rgba64Le] {
            let comps = if packed == PixelFormat::Rgba64Le {
                4
            } else {
                3
            };
            let out = conv(&src, fmt, w, h, packed);
            let words = rd16(&out.planes[0].data);
            for (i, &c) in codes.iter().enumerate() {
                let want = widen(c);
                assert_eq!(&words[i * comps..i * comps + 3], &[want, want, want]);
            }
            assert_eq!(widen(mask), 65535);
            // Back: exact (the projection of r = g = b is the identity
            // and truncation inverts the replication).
            let back = conv(&out, packed, w, h, fmt);
            assert_eq!(rd16(&back.planes[0].data), codes, "{fmt:?} via {packed:?}");
        }
        // Truncation rule on a non-grey source: Rgb48Le → GrayN equals
        // Rgb48Le → Gray16Le → GrayN.
        let mut rng = Rng(bits as u64);
        let rgb: Vec<u16> = (0..n * 3).map(|_| rng.u16()).collect();
        let rgb_frame = frame(w as usize * 6, le16(&rgb));
        let direct = conv(&rgb_frame, PixelFormat::Rgb48Le, w, h, fmt);
        let g16 = conv(
            &rgb_frame,
            PixelFormat::Rgb48Le,
            w,
            h,
            PixelFormat::Gray16Le,
        );
        let two_step = conv(&g16, PixelFormat::Gray16Le, w, h, fmt);
        assert_eq!(direct.planes[0].data, two_step.planes[0].data);
        let g16w = rd16(&g16.planes[0].data);
        let dw = rd16(&direct.planes[0].data);
        for i in 0..n {
            assert_eq!(dw[i], g16w[i] >> (16 - bits));
        }
    }
}

/// `Ya16Le` ↔ `Gray10Le` / `Gray12Le`: luma word narrowed / widened,
/// alpha dropped / opaque.
#[test]
fn ya16_and_deep_gray_words() {
    let (w, h) = (32u32, 8u32);
    let n = (w * h) as usize;
    let mut rng = Rng(99);
    let ya: Vec<u16> = (0..n * 2).map(|_| rng.u16()).collect();
    let src = frame(w as usize * 4, le16(&ya));
    for (bits, fmt) in [(10u32, PixelFormat::Gray10Le), (12, PixelFormat::Gray12Le)] {
        let out = conv(&src, PixelFormat::Ya16Le, w, h, fmt);
        let words = rd16(&out.planes[0].data);
        for i in 0..n {
            assert_eq!(words[i], ya[i * 2] >> (16 - bits));
        }
        let back = conv(&out, fmt, w, h, PixelFormat::Ya16Le);
        let bw = rd16(&back.planes[0].data);
        for i in 0..n {
            let v = words[i];
            assert_eq!(bw[i * 2], (v << (16 - bits)) | (v >> (2 * bits - 16)));
            assert_eq!(bw[i * 2 + 1], 0xFFFF);
        }
    }
}

/// The banded rows produce the same bytes on one thread and on many,
/// at sizes that split into several bands and at sizes that do not.
#[test]
fn deep_gray_packed_rows_threaded_equals_serial() {
    use PixelFormat as P;
    let mt = ConvertContext::new().with_execution(ExecutionContext::with_threads(8));
    let opts = ConvertOptions::default();
    for (w, h) in [(1u32, 1u32), (7, 3), (64, 33), (300, 200)] {
        let n = (w * h) as usize;
        let mut rng = Rng(w as u64 * 1000 + h as u64);
        for (sf, comps) in [
            (P::Gray10Le, 1usize),
            (P::Gray12Le, 1),
            (P::Gray16Le, 1),
            (P::Rgb48Le, 3),
            (P::Rgba64Le, 4),
        ] {
            let mask = match sf {
                P::Gray10Le => 0x3FF,
                P::Gray12Le => 0xFFF,
                _ => 0xFFFF,
            };
            let words: Vec<u16> = (0..n * comps).map(|_| rng.u16() & mask).collect();
            let src = frame(w as usize * comps * 2, le16(&words));
            let info = FrameInfo::new(sf, w, h);
            let targets: &[P] = if comps == 1 {
                &[P::Rgb48Le, P::Rgba64Le]
            } else {
                &[P::Gray10Le, P::Gray12Le, P::Gray16Le]
            };
            for &df in targets {
                let a = convert(&src, info, df, &opts).unwrap();
                let b = convert_with(&src, info, df, &opts, &mt).unwrap();
                assert_eq!(
                    a.planes[0].data, b.planes[0].data,
                    "{sf:?} → {df:?} {w}×{h}"
                );
                assert_eq!(a.planes[0].stride, b.planes[0].stride);
            }
        }
    }
}

/// Short planes are `Error::Invalid`, never a panic.
#[test]
fn short_planes_are_invalid() {
    let src = frame(8, vec![0u8; 8]);
    for (sf, df) in [
        (PixelFormat::Gray16Le, PixelFormat::Rgb48Le),
        (PixelFormat::Rgb48Le, PixelFormat::Gray16Le),
        (PixelFormat::Rgba64Le, PixelFormat::Gray10Le),
    ] {
        let r = convert(
            &src,
            FrameInfo::new(sf, 4, 4),
            df,
            &ConvertOptions::default(),
        );
        assert!(r.is_err(), "{sf:?} → {df:?}");
    }
}
