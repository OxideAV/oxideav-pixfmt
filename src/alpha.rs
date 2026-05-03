//! Alpha-blending and Porter-Duff composite primitives for RGBA buffers.
//!
//! These are the small, well-defined building blocks that font renderers,
//! subtitle compositors and any future overlay pipeline reach for. The
//! maths is the standard Porter-Duff "over" operator from the 1984
//! "Compositing Digital Images" paper — nothing exotic, no third-party
//! library.
//!
//! # Premultiplied vs straight alpha
//!
//! Two flavours of [`over_premul`] / [`over_straight`] are provided:
//!
//! * [`over_premul`] expects both operands to carry **premultiplied**
//!   alpha (R, G, B already scaled by A). The composite reduces to
//!   `dst = src + dst × (1 - src.a)` — one multiply per channel.
//! * [`over_straight`] expects **straight** (non-premultiplied) alpha
//!   inputs and returns a straight-alpha result:
//!   `dst = src × src.a + dst × (1 - src.a)` after rebuilding the output
//!   alpha as `src.a + dst.a × (1 - src.a)`.
//!
//! Use the premultiplied path whenever you have the choice — it's faster
//! and avoids the divide that lurks in [`unpremultiply`].
//!
//! # Numerical accuracy
//!
//! All `u8 × u8 → u8` multiplications go through [`mul_div_255`],
//! which computes `round(a × b / 255)` bit-exactly for every pair of
//! u8 inputs using only integer arithmetic. The shift trick
//! `t = a × b + 128; (t + (t >> 8)) >> 8` is the standard fast form
//! (see e.g. Jim Blinn, "Three Wrongs Make a Right", 1995): it is
//! equivalent to `(a × b + 128) × 257 / 65536` and matches
//! `(a × b + 127) / 255` rounding to the nearest integer for every
//! `a, b ∈ [0, 255]`.

/// Bit-exact rounded byte multiply: `round(a × b / 255)` for all
/// `a, b ∈ [0, 255]`. Implemented as the standard fast shift trick
/// `(t + (t >> 8)) >> 8` with `t = a × b + 128` — one multiply, two
/// shifts, two adds, no division.
#[inline]
fn mul_div_255(a: u32, b: u32) -> u32 {
    let t = a * b + 128;
    (t + (t >> 8)) >> 8
}

/// Composite a premultiplied-alpha RGBA source pixel over a
/// premultiplied-alpha RGBA destination pixel using the Porter-Duff
/// "over" operator: `dst = src + dst × (1 - src.a)`.
#[inline]
pub fn over_premul(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let inv_a = 255 - src[3] as u32;
    [
        (src[0] as u32 + mul_div_255(dst[0] as u32, inv_a)).min(255) as u8,
        (src[1] as u32 + mul_div_255(dst[1] as u32, inv_a)).min(255) as u8,
        (src[2] as u32 + mul_div_255(dst[2] as u32, inv_a)).min(255) as u8,
        (src[3] as u32 + mul_div_255(dst[3] as u32, inv_a)).min(255) as u8,
    ]
}

/// Composite a straight-alpha RGBA source pixel over a straight-alpha
/// RGBA destination pixel: `dst = src × src.a + dst × (1 - src.a)`.
/// Result is straight-alpha.
///
/// The output alpha follows the standard Porter-Duff rebuild
/// `out.a = src.a + dst.a × (1 - src.a)`. When `out.a == 0` the colour
/// channels are forced to zero (the colour is meaningless at fully
/// transparent and a divide by zero would be undefined).
#[inline]
pub fn over_straight(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let sa = src[3] as u32;
    if sa == 0 {
        return dst; // Fully-transparent source: destination is unchanged exactly.
    }
    if sa == 255 {
        return src; // Fully-opaque source replaces destination exactly.
    }
    let da = dst[3] as u32;
    let inv_sa = 255 - sa;
    // Premultiplied numerator for each channel:
    //   num = src × src.a + dst × dst.a × (1 - src.a)
    // out.a = src.a + dst.a × (1 - src.a)
    // out.c = num / out.a (when out.a != 0, else 0)
    let out_a = sa + mul_div_255(da, inv_sa);
    if out_a == 0 {
        return [0, 0, 0, 0];
    }
    let dst_factor = mul_div_255(da, inv_sa);
    let mut out = [0u8; 4];
    for i in 0..3 {
        let num = mul_div_255(src[i] as u32, sa) + mul_div_255(dst[i] as u32, dst_factor);
        // num / out_a: integer divide is fine, both are in [0, 255].
        out[i] = ((num * 255 + (out_a >> 1)) / out_a).min(255) as u8;
    }
    out[3] = out_a.min(255) as u8;
    out
}

/// Premultiply a straight-alpha RGBA pixel:
/// `(R, G, B, A) → (R × A / 255, G × A / 255, B × A / 255, A)`.
#[inline]
pub fn premultiply(rgba: [u8; 4]) -> [u8; 4] {
    let a = rgba[3] as u32;
    [
        mul_div_255(rgba[0] as u32, a) as u8,
        mul_div_255(rgba[1] as u32, a) as u8,
        mul_div_255(rgba[2] as u32, a) as u8,
        rgba[3],
    ]
}

/// Inverse of [`premultiply`]. Maps `A = 0` to `(0, 0, 0, 0)` rather
/// than dividing by zero.
///
/// Note that premultiply→unpremultiply is **not** a perfect roundtrip
/// for non-opaque pixels — quantisation to u8 in the forward step
/// loses the low bits of each channel and the loss grows as `A`
/// shrinks. The roundtrip is exact for `A = 255` and for `A = 0`;
/// for high alpha (`A ≳ 128`) it is within `±1` per channel; at the
/// far end (`A = 1`) up to one bit of source colour can survive (the
/// remaining ~7 bits are unrecoverable). Premultiplied alpha is the
/// natural representation for compositing — going back to straight
/// alpha is intrinsically lossy and only meaningful at moderate-to-
/// high alphas.
#[inline]
pub fn unpremultiply(rgba: [u8; 4]) -> [u8; 4] {
    let a = rgba[3] as u32;
    if a == 0 {
        return [0, 0, 0, 0];
    }
    let half = a / 2;
    [
        ((rgba[0] as u32 * 255 + half) / a).min(255) as u8,
        ((rgba[1] as u32 * 255 + half) / a).min(255) as u8,
        ((rgba[2] as u32 * 255 + half) / a).min(255) as u8,
        rgba[3],
    ]
}

/// Multiply the alpha channel by `opacity` (0..=255). Per-pixel
/// input/output. Colour channels are untouched — this is meant for
/// straight-alpha pixels. For premultiplied pixels, also scale R/G/B
/// by `opacity / 255` (or call this and then [`premultiply`]).
#[inline]
pub fn modulate_alpha(rgba: [u8; 4], opacity: u8) -> [u8; 4] {
    let new_a = mul_div_255(rgba[3] as u32, opacity as u32) as u8;
    [rgba[0], rgba[1], rgba[2], new_a]
}

/// Blit a single-channel alpha mask × an RGBA colour over an RGBA
/// destination buffer.
///
/// * `mask`: row-major u8 grayscale `[0..=255]` of size
///   `mask_width × mask_height`.
/// * `dst_stride`: bytes per row of `dst` (typically `dst_width × 4`).
/// * `mask_stride`: bytes per row of `mask` (typically `mask_width`).
/// * `color`: straight-alpha RGBA — internally premultiplied with the
///   per-pixel mask alpha before the over-composite. The destination is
///   treated as **straight-alpha** RGBA and the result is straight-alpha
///   (this matches the way font renderers and subtitle compositors hand
///   off coloured glyph masks to a regular RGBA framebuffer).
///
/// `(x, y)` is the top-left destination coordinate at which to place the
/// mask. Coordinates are clipped to the destination buffer — any
/// portion of the mask that falls outside `[0, dst_width) × [0,
/// dst_height)` is silently skipped. A blit that is entirely outside
/// the buffer (or has a zero-sized mask) is a no-op.
#[allow(clippy::too_many_arguments)] // dst + mask are (buf, w, h, stride) tuples, by design
pub fn blit_alpha_mask(
    dst: &mut [u8],
    dst_width: u32,
    dst_height: u32,
    dst_stride: usize,
    x: i32,
    y: i32,
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    mask_stride: usize,
    color: [u8; 4],
) {
    if mask_width == 0 || mask_height == 0 || dst_width == 0 || dst_height == 0 {
        return;
    }
    let dst_w = dst_width as i64;
    let dst_h = dst_height as i64;
    let mw = mask_width as i64;
    let mh = mask_height as i64;
    let x = x as i64;
    let y = y as i64;

    // Compute the visible rectangle in destination coordinates.
    let dx0 = x.max(0);
    let dy0 = y.max(0);
    let dx1 = (x + mw).min(dst_w);
    let dy1 = (y + mh).min(dst_h);
    if dx0 >= dx1 || dy0 >= dy1 {
        return; // fully off-screen
    }

    // Corresponding offset into the mask.
    let mx0 = (dx0 - x) as usize;
    let my0 = (dy0 - y) as usize;
    let blit_w = (dx1 - dx0) as usize;
    let blit_h = (dy1 - dy0) as usize;

    let cr = color[0] as u32;
    let cg = color[1] as u32;
    let cb = color[2] as u32;
    let ca = color[3] as u32;

    for row in 0..blit_h {
        let dst_y = (dy0 as usize) + row;
        let m_y = my0 + row;
        let dst_row_off = dst_y * dst_stride + (dx0 as usize) * 4;
        let mask_row_off = m_y * mask_stride + mx0;
        let dst_row = &mut dst[dst_row_off..dst_row_off + blit_w * 4];
        let mask_row = &mask[mask_row_off..mask_row_off + blit_w];

        for col in 0..blit_w {
            let m = mask_row[col] as u32;
            if m == 0 {
                continue;
            }
            // Effective straight source alpha = color.a × mask / 255.
            let sa = mul_div_255(ca, m);
            if sa == 0 {
                continue;
            }
            let inv_sa = 255 - sa;
            // Premultiplied source colour (sa already includes mask):
            //   src_premul.c = color.c × sa / 255.
            // Then standard over-into-straight-dst:
            //   out.a = sa + dst.a × (1 - sa)
            //   num.c = color.c × sa + dst.c × dst.a × (1 - sa)
            //   out.c = num.c / out.a    (or 0 if out.a == 0)
            let dr = dst_row[col * 4] as u32;
            let dg = dst_row[col * 4 + 1] as u32;
            let db = dst_row[col * 4 + 2] as u32;
            let da = dst_row[col * 4 + 3] as u32;

            let out_a = sa + mul_div_255(da, inv_sa);
            if out_a == 0 {
                dst_row[col * 4] = 0;
                dst_row[col * 4 + 1] = 0;
                dst_row[col * 4 + 2] = 0;
                dst_row[col * 4 + 3] = 0;
                continue;
            }
            let dst_factor = mul_div_255(da, inv_sa);
            let half = out_a >> 1;
            let nr = mul_div_255(cr, sa) + mul_div_255(dr, dst_factor);
            let ng = mul_div_255(cg, sa) + mul_div_255(dg, dst_factor);
            let nb = mul_div_255(cb, sa) + mul_div_255(db, dst_factor);

            dst_row[col * 4] = ((nr * 255 + half) / out_a).min(255) as u8;
            dst_row[col * 4 + 1] = ((ng * 255 + half) / out_a).min(255) as u8;
            dst_row[col * 4 + 2] = ((nb * 255 + half) / out_a).min(255) as u8;
            dst_row[col * 4 + 3] = out_a.min(255) as u8;
        }
    }
}

/// Composite an entire RGBA source buffer over an RGBA destination
/// buffer in-place. Both buffers must have the same width, height, and
/// stride. Set `premultiplied = true` to use the premultiplied "over"
/// operator on both operands; `false` for straight-alpha inputs and
/// outputs.
pub fn over_buffer(
    dst: &mut [u8],
    src: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    premultiplied: bool,
) {
    if width == 0 || height == 0 {
        return;
    }
    let row_bytes = width as usize * 4;
    debug_assert!(stride >= row_bytes, "stride must accommodate width × 4");
    for row in 0..height as usize {
        let off = row * stride;
        let dst_row = &mut dst[off..off + row_bytes];
        let src_row = &src[off..off + row_bytes];
        for col in 0..width as usize {
            let s = [
                src_row[col * 4],
                src_row[col * 4 + 1],
                src_row[col * 4 + 2],
                src_row[col * 4 + 3],
            ];
            let d = [
                dst_row[col * 4],
                dst_row[col * 4 + 1],
                dst_row[col * 4 + 2],
                dst_row[col * 4 + 3],
            ];
            let out = if premultiplied {
                over_premul(s, d)
            } else {
                over_straight(s, d)
            };
            dst_row[col * 4] = out[0];
            dst_row[col * 4 + 1] = out[1];
            dst_row[col * 4 + 2] = out[2];
            dst_row[col * 4 + 3] = out[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_div_255_matches_reference() {
        for a in 0u32..=255 {
            for b in 0u32..=255 {
                let got = mul_div_255(a, b);
                let want = (a * b + 127) / 255;
                assert_eq!(got, want, "mismatch at a={a} b={b}");
            }
        }
    }

    #[test]
    fn over_premul_opaque_src_replaces_dst() {
        let src = [10, 20, 30, 255];
        let dst = [100, 100, 100, 255];
        assert_eq!(over_premul(src, dst), src);
    }

    #[test]
    fn over_premul_transparent_src_keeps_dst() {
        let src = [0, 0, 0, 0]; // fully transparent (premultiplied)
        let dst = [100, 110, 120, 200];
        assert_eq!(over_premul(src, dst), dst);
    }

    #[test]
    fn over_premul_half_alpha() {
        // 50% red premultiplied (R=128, A=128) over opaque blue.
        let src = [128, 0, 0, 128];
        let dst = [0, 0, 255, 255];
        let out = over_premul(src, dst);
        // dst contribution: dst × (1 - 128/255) = dst × 127/255.
        // Blue channel ≈ 255 * 127/255 ≈ 127, red ≈ 128 + 0 = 128.
        assert_eq!(out[0], 128);
        assert_eq!(out[1], 0);
        assert!((out[2] as i32 - 127).abs() <= 1, "blue was {}", out[2]);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn over_straight_opaque_src_replaces_dst() {
        let src = [200, 50, 25, 255];
        let dst = [10, 10, 10, 255];
        assert_eq!(over_straight(src, dst), src);
    }

    #[test]
    fn over_straight_transparent_src_keeps_dst() {
        let src = [255, 255, 255, 0];
        let dst = [40, 50, 60, 200];
        assert_eq!(over_straight(src, dst), dst);
    }

    #[test]
    fn over_straight_half_alpha_over_opaque() {
        // 50% red over opaque blue: result is straight-alpha purple, A=255.
        let src = [255, 0, 0, 128];
        let dst = [0, 0, 255, 255];
        let out = over_straight(src, dst);
        // out.a = 128 + 255 * 127 / 255 ≈ 255
        // out.r ≈ 255 * 128/255 ≈ 128
        // out.b ≈ 255 * 127/255 ≈ 127
        assert!((out[0] as i32 - 128).abs() <= 2, "r was {}", out[0]);
        assert_eq!(out[1], 0);
        assert!((out[2] as i32 - 127).abs() <= 2, "b was {}", out[2]);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn premultiply_opaque_is_identity() {
        let p = [200, 100, 50, 255];
        assert_eq!(premultiply(p), p);
    }

    #[test]
    fn premultiply_transparent_yields_zero_color() {
        let p = [200, 100, 50, 0];
        assert_eq!(premultiply(p), [0, 0, 0, 0]);
    }

    #[test]
    fn premul_unpremul_roundtrip_opaque() {
        let p = [200, 100, 50, 255];
        assert_eq!(unpremultiply(premultiply(p)), p);
    }

    #[test]
    fn premul_unpremul_roundtrip_transparent() {
        let p = [200, 100, 50, 0];
        // Forward gives (0,0,0,0); inverse stays at (0,0,0,0).
        assert_eq!(unpremultiply(premultiply(p)), [0, 0, 0, 0]);
    }

    #[test]
    fn premul_unpremul_roundtrip_high_alpha_within_one() {
        // Quantisation of u8 premul loses precision as alpha shrinks.
        // For high alpha (≥128) the roundtrip stays within ±1 per
        // channel; lower alphas are intrinsically lossy and not part
        // of the contract (see `unpremultiply` docstring).
        for a in (128u32..=255).step_by(17) {
            for r in (0u32..=255).step_by(13) {
                let p = [r as u8, (255 - r) as u8, (r ^ 0x5a) as u8, a as u8];
                let round = unpremultiply(premultiply(p));
                for ch in 0..3 {
                    let diff = (round[ch] as i32 - p[ch] as i32).abs();
                    assert!(
                        diff <= 1,
                        "channel {ch} drifted by {diff} for input {:?}",
                        p
                    );
                }
                assert_eq!(round[3], p[3]);
            }
        }
    }

    #[test]
    fn premul_unpremul_roundtrip_low_alpha_bounded_by_quantisation() {
        // At very low alpha there's only ~log2(A) bits of usable
        // colour precision; the roundtrip diff is bounded by
        // ceil(255 / A) which is the spacing between representable
        // premultiplied colour values.
        for a in 1u32..32 {
            let max_diff = 255u32.div_ceil(a) as i32;
            for r in (0u32..=255).step_by(7) {
                let p = [r as u8, 0, 0, a as u8];
                let round = unpremultiply(premultiply(p));
                let diff = (round[0] as i32 - p[0] as i32).abs();
                assert!(
                    diff <= max_diff,
                    "channel 0 drift {diff} > bound {max_diff} for input {:?}",
                    p
                );
                assert_eq!(round[3], p[3]);
            }
        }
    }

    #[test]
    fn modulate_alpha_full_is_noop() {
        let p = [10, 20, 30, 200];
        assert_eq!(modulate_alpha(p, 255), p);
    }

    #[test]
    fn modulate_alpha_zero_clears_alpha() {
        let p = [10, 20, 30, 200];
        assert_eq!(modulate_alpha(p, 0), [10, 20, 30, 0]);
    }

    #[test]
    fn modulate_alpha_half() {
        let p = [10, 20, 30, 200];
        let out = modulate_alpha(p, 128);
        // 200 * 128 / 255 ≈ 100
        assert!((out[3] as i32 - 100).abs() <= 1, "alpha was {}", out[3]);
        assert_eq!(&out[..3], &p[..3]);
    }

    #[test]
    fn blit_alpha_mask_full_white_8x8_onto_black() {
        let dst_w = 16u32;
        let dst_h = 16u32;
        let stride = (dst_w * 4) as usize;
        let mut dst = vec![0u8; stride * dst_h as usize];
        // Make destination fully opaque black so we can verify the
        // result without alpha math.
        for px in dst.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let mask_w = 8u32;
        let mask_h = 8u32;
        let mask = vec![255u8; (mask_w * mask_h) as usize];
        let color = [255, 255, 255, 255];
        blit_alpha_mask(
            &mut dst,
            dst_w,
            dst_h,
            stride,
            0,
            0,
            &mask,
            mask_w,
            mask_h,
            mask_w as usize,
            color,
        );
        // Top-left 8×8 must be white, rest must still be black.
        for y in 0..dst_h as usize {
            for x in 0..dst_w as usize {
                let off = y * stride + x * 4;
                let in_mask = x < 8 && y < 8;
                let want: [u8; 4] = if in_mask {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                assert_eq!(
                    [dst[off], dst[off + 1], dst[off + 2], dst[off + 3]],
                    want,
                    "pixel ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn blit_alpha_mask_clips_top_left() {
        let dst_w = 4u32;
        let dst_h = 4u32;
        let stride = (dst_w * 4) as usize;
        let mut dst = vec![0u8; stride * dst_h as usize];
        for px in dst.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let mask = vec![255u8; 16]; // 4×4
                                    // Place 4×4 mask at (-2, -2): only the bottom-right 2×2 is visible.
        blit_alpha_mask(
            &mut dst,
            dst_w,
            dst_h,
            stride,
            -2,
            -2,
            &mask,
            4,
            4,
            4,
            [255, 255, 255, 255],
        );
        for y in 0..4usize {
            for x in 0..4usize {
                let off = y * stride + x * 4;
                let in_visible = x < 2 && y < 2;
                let want: [u8; 4] = if in_visible {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                assert_eq!(
                    [dst[off], dst[off + 1], dst[off + 2], dst[off + 3]],
                    want,
                    "pixel ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn blit_alpha_mask_clips_bottom_right() {
        let dst_w = 4u32;
        let dst_h = 4u32;
        let stride = (dst_w * 4) as usize;
        let mut dst = vec![0u8; stride * dst_h as usize];
        for px in dst.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let mask = vec![255u8; 16];
        // Place 4×4 mask at (3, 3): only top-left 1×1 is visible.
        blit_alpha_mask(
            &mut dst,
            dst_w,
            dst_h,
            stride,
            3,
            3,
            &mask,
            4,
            4,
            4,
            [255, 255, 255, 255],
        );
        for y in 0..4usize {
            for x in 0..4usize {
                let off = y * stride + x * 4;
                let in_visible = x == 3 && y == 3;
                let want: [u8; 4] = if in_visible {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                assert_eq!(
                    [dst[off], dst[off + 1], dst[off + 2], dst[off + 3]],
                    want,
                    "pixel ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn blit_alpha_mask_fully_offscreen_is_noop() {
        let dst_w = 4u32;
        let dst_h = 4u32;
        let stride = (dst_w * 4) as usize;
        let mut dst = vec![42u8; stride * dst_h as usize];
        let snapshot = dst.clone();
        let mask = vec![255u8; 16];
        // Way past the buffer.
        blit_alpha_mask(
            &mut dst,
            dst_w,
            dst_h,
            stride,
            100,
            100,
            &mask,
            4,
            4,
            4,
            [255, 255, 255, 255],
        );
        assert_eq!(dst, snapshot);
        // Way before the buffer.
        blit_alpha_mask(
            &mut dst,
            dst_w,
            dst_h,
            stride,
            -100,
            -100,
            &mask,
            4,
            4,
            4,
            [255, 255, 255, 255],
        );
        assert_eq!(dst, snapshot);
    }

    #[test]
    fn over_buffer_premul_4x4_red_over_blue() {
        // 50% red premultiplied (R=128, A=128) over opaque blue.
        let w = 4u32;
        let h = 4u32;
        let stride = (w * 4) as usize;
        let mut dst = vec![0u8; stride * h as usize];
        let mut src = vec![0u8; stride * h as usize];
        for px in dst.chunks_exact_mut(4) {
            px[2] = 255;
            px[3] = 255;
        }
        for px in src.chunks_exact_mut(4) {
            px[0] = 128;
            px[3] = 128;
        }
        over_buffer(&mut dst, &src, w, h, stride, true);
        for px in dst.chunks_exact(4) {
            assert_eq!(px[0], 128);
            assert_eq!(px[1], 0);
            assert!((px[2] as i32 - 127).abs() <= 1, "blue was {}", px[2]);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn over_buffer_straight_4x4_red_over_blue() {
        let w = 4u32;
        let h = 4u32;
        let stride = (w * 4) as usize;
        let mut dst = vec![0u8; stride * h as usize];
        let mut src = vec![0u8; stride * h as usize];
        for px in dst.chunks_exact_mut(4) {
            px[2] = 255;
            px[3] = 255;
        }
        // Straight-alpha 50% red.
        for px in src.chunks_exact_mut(4) {
            px[0] = 255;
            px[3] = 128;
        }
        over_buffer(&mut dst, &src, w, h, stride, false);
        for px in dst.chunks_exact(4) {
            assert!((px[0] as i32 - 128).abs() <= 2, "red was {}", px[0]);
            assert_eq!(px[1], 0);
            assert!((px[2] as i32 - 127).abs() <= 2, "blue was {}", px[2]);
            assert_eq!(px[3], 255);
        }
    }
}
