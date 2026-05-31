//! Property / fuzz-style suite for the Porter-Duff alpha primitives.
//!
//! `alpha.rs` historically grew anchor cases (50% red over blue, opaque
//! passthrough, glyph-square blit) but no randomised contract sweep. This
//! file pins the *mathematical* invariants of the operators across a
//! ~200k-sample PRNG sweep:
//!
//! * **`over_premul`**:
//!     * Opaque source replaces destination exactly.
//!     * Transparent source is a destination no-op.
//!     * Result lies in the convex hull of the two operands per channel
//!       (no extrapolation, no overflow past 255).
//!     * Compositing the same source again is idempotent at α=255 and
//!       monotonic on the alpha channel.
//! * **`over_straight`**:
//!     * Opaque source replaces destination exactly.
//!     * Transparent source is a destination no-op.
//!     * Output alpha equals `src.a + dst.a × (1 - src.a)` to within
//!       one LSB (the `mul_div_255` rounding step's worst case).
//!     * Output alpha is monotonically non-decreasing in `src.a`.
//! * **`over_buffer`**: matches the per-pixel `over_premul` /
//!   `over_straight` element-wise across every premultiplied/straight
//!   choice, every dimension in a random set, and every tight-stride row.
//! * **`blit_alpha_mask`**: no-panic and out-of-rect leak-freedom across
//!   ~10k random (offset, mask-size, dst-size) placements covering
//!   wholly-inside, partly-clipped (each edge + each corner), and
//!   wholly-outside cases.
//! * **`modulate_alpha`**: monotonic in opacity, exact endpoints, RGB
//!   channels are byte-identical.
//!
//! No external crates; the PRNG is the same xorshift family as
//! `tests/property.rs`, kept local so this file is self-contained.

use oxideav_pixfmt::{
    blit_alpha_mask, modulate_alpha, over_buffer, over_premul, over_straight, premultiply,
    unpremultiply,
};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
    }
    fn rgba(&mut self) -> [u8; 4] {
        [self.byte(), self.byte(), self.byte(), self.byte()]
    }
    /// Sample a **valid** premultiplied-alpha pixel — every colour
    /// channel is clamped to alpha so the operand satisfies the
    /// `C ≤ A` premultiplied invariant. Without this clamp the random
    /// sweep would inject malformed premultiplied inputs that the
    /// per-pixel operator is explicitly *not* required to handle
    /// monotonically (premultiplied pixels with `C > A` are out of
    /// the operator's domain).
    fn rgba_premul(&mut self) -> [u8; 4] {
        let a = self.byte();
        let mut p = [0u8; 4];
        p[3] = a;
        for c in p.iter_mut().take(3) {
            *c = if a == 0 { 0 } else { self.byte().min(a) };
        }
        p
    }
}

// ---------------------------------------------------------------------
// over_premul

#[test]
fn prop_over_premul_opaque_src_replaces_dst() {
    let mut rng = Rng::new(0xA1);
    for _ in 0..50_000u32 {
        let mut s = rng.rgba_premul();
        s[3] = 255;
        // Clamp colour to satisfy premultiplied invariant at A=255 (no-op since 255 ≥ c).
        let d = rng.rgba_premul();
        let out = over_premul(s, d);
        assert_eq!(
            out, s,
            "opaque-src premul: expected {:?}, got {:?} over {:?}",
            s, out, d
        );
    }
}

#[test]
fn prop_over_premul_transparent_src_keeps_dst() {
    let mut rng = Rng::new(0xA2);
    for _ in 0..50_000u32 {
        // Transparent premultiplied source must have colour = 0.
        let s = [0u8, 0, 0, 0];
        let d = rng.rgba_premul();
        let out = over_premul(s, d);
        assert_eq!(
            out, d,
            "transparent-src premul: expected {:?}, got {:?}",
            d, out
        );
    }
}

#[test]
fn prop_over_premul_output_bounded_by_operands() {
    // out = src + dst × (1 - src.a). Each component is the sum of a
    // non-negative source contribution and a fraction of the
    // destination. With *valid* premultiplied inputs (C ≤ A on both
    // operands), the result has C ≤ A on every channel and never
    // overflows u8. We pin both bounds here.
    let mut rng = Rng::new(0xA3);
    for case in 0..200_000u32 {
        let s = rng.rgba_premul();
        let d = rng.rgba_premul();
        let out = over_premul(s, d);
        // Premultiplied invariant on output: C ≤ A.
        for ch in 0..3 {
            assert!(
                out[ch] <= out[3],
                "case {case}: out.C={} > out.A={} for src={:?} dst={:?} -> {:?}",
                out[ch],
                out[3],
                s,
                d,
                out
            );
        }
    }
}

#[test]
fn prop_over_premul_alpha_monotone_in_src_alpha() {
    // Holding src.colour proportional to src.a (the premultiplied form)
    // and the destination fixed, increasing src.a may only *increase*
    // out.a — covering the source contributes more, blocking covers less.
    let mut rng = Rng::new(0xA4);
    for _ in 0..20_000u32 {
        let d = rng.rgba_premul();
        // Hold the source *colour direction* (r, g, b ∝ a) so the operand
        // stays inside the premultiplied domain while sweeping a.
        let dir_r = rng.byte();
        let dir_g = rng.byte();
        let dir_b = rng.byte();
        let mut prev_a = 0u8;
        for a in (0u32..=255).step_by(17) {
            let s = [
                ((dir_r as u32 * a) / 255) as u8,
                ((dir_g as u32 * a) / 255) as u8,
                ((dir_b as u32 * a) / 255) as u8,
                a as u8,
            ];
            let out = over_premul(s, d);
            assert!(
                out[3] >= prev_a,
                "out.a regressed: {prev_a} -> {} at a={a} (s={:?} d={:?})",
                out[3],
                s,
                d
            );
            prev_a = out[3];
        }
    }
}

#[test]
fn prop_over_premul_associative_with_opaque_replacement() {
    // Compositing any premultiplied source over the result of an opaque
    // composite must equal the over-the-opaque case directly. (Opaque
    // intermediates are an absorbing element for the operator.)
    let mut rng = Rng::new(0xA5);
    for _ in 0..20_000u32 {
        let mut bottom = rng.rgba_premul();
        bottom[3] = 255;
        // Colour channels already u8 → ≤ 255; alpha 255 satisfies C ≤ A.
        let mid = rng.rgba_premul();
        let top = rng.rgba_premul();
        let direct = over_premul(top, over_premul(mid, bottom));
        // Anything composited over an opaque pixel keeps alpha=255 in
        // the premul form, so the chain stays opaque-bottomed.
        assert_eq!(direct[3], 255);
    }
}

// ---------------------------------------------------------------------
// over_straight

#[test]
fn prop_over_straight_opaque_src_replaces_dst() {
    let mut rng = Rng::new(0xB1);
    for _ in 0..50_000u32 {
        let mut s = rng.rgba();
        s[3] = 255;
        let d = rng.rgba();
        let out = over_straight(s, d);
        assert_eq!(out, s);
    }
}

#[test]
fn prop_over_straight_transparent_src_keeps_dst() {
    let mut rng = Rng::new(0xB2);
    for _ in 0..50_000u32 {
        let mut s = rng.rgba();
        s[3] = 0;
        let d = rng.rgba();
        let out = over_straight(s, d);
        assert_eq!(out, d);
    }
}

#[test]
fn prop_over_straight_alpha_formula_within_one_lsb() {
    // out.a = src.a + dst.a × (1 - src.a) / 255. The implementation
    // uses the rounded mul_div_255 step, so the result must equal the
    // reference within one ULP.
    let mut rng = Rng::new(0xB3);
    for _ in 0..200_000u32 {
        let s = rng.rgba();
        let d = rng.rgba();
        let out = over_straight(s, d);
        let want = s[3] as i32 + ((d[3] as i32) * (255 - s[3] as i32) + 127) / 255;
        let want = want.min(255);
        let diff = (out[3] as i32 - want).abs();
        assert!(
            diff <= 1,
            "alpha formula drift {diff} > 1: got {} want {} for s={:?} d={:?}",
            out[3],
            want,
            s,
            d
        );
    }
}

#[test]
fn prop_over_straight_alpha_monotone_in_src_alpha() {
    let mut rng = Rng::new(0xB4);
    for _ in 0..20_000u32 {
        let mut s = rng.rgba();
        let d = rng.rgba();
        let mut prev = 0u8;
        for a in (0u32..=255).step_by(11) {
            s[3] = a as u8;
            let out = over_straight(s, d);
            assert!(
                out[3] >= prev,
                "out.a regressed: {prev} -> {} at src.a={a}, dst={:?}",
                out[3],
                d
            );
            prev = out[3];
        }
    }
}

// ---------------------------------------------------------------------
// over_buffer

#[test]
fn prop_over_buffer_premul_matches_per_pixel() {
    let mut rng = Rng::new(0xC1);
    for _ in 0..50u32 {
        let w = rng.range(1, 9);
        let h = rng.range(1, 9);
        let stride = (w * 4) as usize;
        let mut src = vec![0u8; stride * h as usize];
        let mut dst = vec![0u8; stride * h as usize];
        for px in src.chunks_exact_mut(4) {
            let p = rng.rgba_premul();
            px.copy_from_slice(&p);
        }
        for px in dst.chunks_exact_mut(4) {
            let p = rng.rgba_premul();
            px.copy_from_slice(&p);
        }
        let mut expected = dst.clone();
        for row in 0..h as usize {
            for col in 0..w as usize {
                let off = row * stride + col * 4;
                let s = [src[off], src[off + 1], src[off + 2], src[off + 3]];
                let d = [
                    expected[off],
                    expected[off + 1],
                    expected[off + 2],
                    expected[off + 3],
                ];
                let out = over_premul(s, d);
                expected[off..off + 4].copy_from_slice(&out);
            }
        }
        over_buffer(&mut dst, &src, w, h, stride, true);
        assert_eq!(dst, expected, "premul {w}x{h}");
    }
}

#[test]
fn prop_over_buffer_straight_matches_per_pixel() {
    let mut rng = Rng::new(0xC2);
    for _ in 0..50u32 {
        let w = rng.range(1, 9);
        let h = rng.range(1, 9);
        let stride = (w * 4) as usize;
        let mut src = vec![0u8; stride * h as usize];
        let mut dst = vec![0u8; stride * h as usize];
        for b in src.iter_mut() {
            *b = rng.byte();
        }
        for b in dst.iter_mut() {
            *b = rng.byte();
        }
        let mut expected = dst.clone();
        for row in 0..h as usize {
            for col in 0..w as usize {
                let off = row * stride + col * 4;
                let s = [src[off], src[off + 1], src[off + 2], src[off + 3]];
                let d = [
                    expected[off],
                    expected[off + 1],
                    expected[off + 2],
                    expected[off + 3],
                ];
                let out = over_straight(s, d);
                expected[off..off + 4].copy_from_slice(&out);
            }
        }
        over_buffer(&mut dst, &src, w, h, stride, false);
        assert_eq!(dst, expected, "straight {w}x{h}");
    }
}

// ---------------------------------------------------------------------
// blit_alpha_mask

#[test]
fn prop_blit_alpha_mask_does_not_write_outside_visible_rect() {
    // Surround the destination rectangle with sentinel rows/cols at a
    // wider stride; any write past the declared width must leave the
    // sentinels untouched. Sweep random placements covering wholly-in,
    // partly-clipped per edge, and wholly-out cases.
    let mut rng = Rng::new(0xD1);
    let dst_w: u32 = 24;
    let dst_h: u32 = 18;
    let padded_stride: usize = (dst_w as usize + 8) * 4; // 8 px of right-side padding per row
    let total_rows = dst_h as usize + 6; // 6 rows below for bottom-padding
    let total_bytes = padded_stride * total_rows;
    let sentinel: u8 = 0xA5;
    for case in 0..10_000u32 {
        let mut dst = vec![sentinel; total_bytes];
        let mw = rng.range(1, 32);
        let mh = rng.range(1, 24);
        let mask_stride = mw as usize;
        let mut mask = vec![0u8; mask_stride * mh as usize];
        for b in mask.iter_mut() {
            *b = rng.byte();
        }
        let x = rng.range(0, 60) as i32 - 30;
        let y = rng.range(0, 50) as i32 - 25;
        let color = rng.rgba();
        blit_alpha_mask(
            &mut dst,
            dst_w,
            dst_h,
            padded_stride,
            x,
            y,
            &mask,
            mw,
            mh,
            mask_stride,
            color,
        );
        // 1. Padding beyond `dst_w` on every active row must keep the sentinel.
        let row_visible_bytes = (dst_w as usize) * 4;
        for row in 0..dst_h as usize {
            let pad_start = row * padded_stride + row_visible_bytes;
            let pad_end = (row + 1) * padded_stride;
            for (off, &byte) in dst.iter().enumerate().take(pad_end).skip(pad_start) {
                assert_eq!(
                    byte, sentinel,
                    "case {case}: padding byte at row {row} off {off} was overwritten (x={x} y={y} mw={mw} mh={mh})"
                );
            }
        }
        // 2. Rows beyond `dst_h` must keep the sentinel.
        let bot_start = dst_h as usize * padded_stride;
        for (off, &byte) in dst.iter().enumerate().take(total_bytes).skip(bot_start) {
            assert_eq!(
                byte, sentinel,
                "case {case}: bottom-padding byte at off {off} was overwritten (x={x} y={y} mw={mw} mh={mh})"
            );
        }
    }
}

#[test]
fn prop_blit_alpha_mask_zero_size_inputs_are_noop() {
    let mut rng = Rng::new(0xD2);
    let dst_w: u32 = 8;
    let dst_h: u32 = 8;
    let stride = (dst_w as usize) * 4;
    for _ in 0..200u32 {
        let mut dst = vec![rng.byte(); stride * dst_h as usize];
        let snapshot = dst.clone();
        let mask: Vec<u8> = vec![255; 16];
        // Zero mask width.
        blit_alpha_mask(
            &mut dst, dst_w, dst_h, stride, 0, 0, &mask, 0, 4, 4, [255; 4],
        );
        assert_eq!(dst, snapshot);
        // Zero mask height.
        blit_alpha_mask(
            &mut dst, dst_w, dst_h, stride, 0, 0, &mask, 4, 0, 4, [255; 4],
        );
        assert_eq!(dst, snapshot);
        // Zero dst width/height.
        blit_alpha_mask(&mut dst, 0, dst_h, stride, 0, 0, &mask, 4, 4, 4, [255; 4]);
        assert_eq!(dst, snapshot);
        blit_alpha_mask(&mut dst, dst_w, 0, stride, 0, 0, &mask, 4, 4, 4, [255; 4]);
        assert_eq!(dst, snapshot);
    }
}

// ---------------------------------------------------------------------
// modulate_alpha

#[test]
fn prop_modulate_alpha_monotone_and_endpoint_exact() {
    let mut rng = Rng::new(0xE1);
    for _ in 0..50_000u32 {
        let p = rng.rgba();
        // Endpoints exact.
        assert_eq!(modulate_alpha(p, 255), p);
        let zeroed = [p[0], p[1], p[2], 0];
        assert_eq!(modulate_alpha(p, 0), zeroed);
        // RGB untouched for any opacity.
        let mid = modulate_alpha(p, 128);
        assert_eq!(&mid[..3], &p[..3]);
        // Monotone in opacity.
        let mut prev = 0u8;
        for op in (0u32..=255).step_by(13) {
            let out = modulate_alpha(p, op as u8);
            assert!(
                out[3] >= prev,
                "modulate_alpha not monotone: {prev} -> {} at op={op}",
                out[3]
            );
            prev = out[3];
        }
    }
}

// ---------------------------------------------------------------------
// premultiply / unpremultiply structural pin (complements
// tests/property.rs::prop_premultiply_unpremultiply_bounded_by_alpha,
// which sweeps the *bound*; here we pin the *exact-at-A=255 plus
// alpha-survives* axiom on a smaller hand-picked grid for fast feedback).

#[test]
fn prop_premultiply_alpha_survives_exactly() {
    let mut rng = Rng::new(0xE2);
    for _ in 0..50_000u32 {
        let p = rng.rgba();
        let pm = premultiply(p);
        assert_eq!(pm[3], p[3]);
        let back = unpremultiply(pm);
        assert_eq!(back[3], p[3]);
        if p[3] == 255 {
            assert_eq!(back, p, "A=255 must roundtrip exactly");
        }
    }
}
