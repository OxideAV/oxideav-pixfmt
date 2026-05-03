//! Integration tests for the `alpha` module: exercise the public API
//! through the crate root re-exports the same way downstream callers
//! (oxideav-scribe, future subtitle compositor) will.

use oxideav_pixfmt::{
    blit_alpha_mask, modulate_alpha, over_buffer, over_premul, over_straight, premultiply,
    unpremultiply,
};

#[test]
fn opaque_src_is_passthrough_for_both_modes() {
    let src = [123, 45, 67, 255];
    let dst = [200, 100, 50, 255];
    assert_eq!(over_premul(src, dst), src);
    assert_eq!(over_straight(src, dst), src);
}

#[test]
fn transparent_src_is_noop_for_both_modes() {
    let src_premul = [0, 0, 0, 0];
    let src_straight = [123, 45, 67, 0];
    let dst = [200, 100, 50, 200];
    assert_eq!(over_premul(src_premul, dst), dst);
    assert_eq!(over_straight(src_straight, dst), dst);
}

#[test]
fn premultiply_then_unpremultiply_is_close_to_identity() {
    // Roundtrip is exact at A=0/255 and stays within ±1 per channel
    // for high alphas (A ≥ 128). Lower alphas drift by up to
    // ~ceil(255/A) — see `unpremultiply` docstring.
    for &a in &[0u8, 128, 200, 254, 255] {
        for &c in &[0u8, 1, 50, 127, 200, 255] {
            let p = [c, c, c, a];
            let round = unpremultiply(premultiply(p));
            if a == 0 {
                assert_eq!(round, [0, 0, 0, 0]);
            } else {
                for ch in 0..3 {
                    assert!(
                        (round[ch] as i32 - p[ch] as i32).abs() <= 1,
                        "drift at ch={ch} for {:?} -> {:?}",
                        p,
                        round
                    );
                }
                assert_eq!(round[3], a);
            }
        }
    }
}

#[test]
fn modulate_alpha_endpoints() {
    let p = [42, 99, 144, 200];
    assert_eq!(modulate_alpha(p, 255), p);
    assert_eq!(modulate_alpha(p, 0), [42, 99, 144, 0]);
}

#[test]
fn blit_alpha_mask_8x8_white_on_16x16_black_yields_white_square() {
    let dst_w = 16u32;
    let dst_h = 16u32;
    let stride = (dst_w * 4) as usize;
    let mut dst = vec![0u8; stride * dst_h as usize];
    for px in dst.chunks_exact_mut(4) {
        px[3] = 255;
    }
    let mask = vec![255u8; 64];
    blit_alpha_mask(
        &mut dst,
        dst_w,
        dst_h,
        stride,
        0,
        0,
        &mask,
        8,
        8,
        8,
        [255, 255, 255, 255],
    );
    for y in 0..dst_h as usize {
        for x in 0..dst_w as usize {
            let off = y * stride + x * 4;
            let want: [u8; 4] = if x < 8 && y < 8 {
                [255, 255, 255, 255]
            } else {
                [0, 0, 0, 255]
            };
            assert_eq!([dst[off], dst[off + 1], dst[off + 2], dst[off + 3]], want);
        }
    }
}

#[test]
fn blit_alpha_mask_clipping_does_not_panic() {
    let dst_w = 8u32;
    let dst_h = 8u32;
    let stride = (dst_w * 4) as usize;
    let mut dst = vec![64u8; stride * dst_h as usize];
    let mask = vec![255u8; 16]; // 4×4
    for &(x, y) in &[
        (-1000i32, -1000i32),
        (1000, 1000),
        (-3, -3),
        (6, 6),
        (-2, 4),
        (5, -2),
    ] {
        blit_alpha_mask(
            &mut dst,
            dst_w,
            dst_h,
            stride,
            x,
            y,
            &mask,
            4,
            4,
            4,
            [10, 20, 30, 200],
        );
    }
}

#[test]
fn over_buffer_preserves_dimensions_and_blends_uniformly() {
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
    let first = [dst[0], dst[1], dst[2], dst[3]];
    for px in dst.chunks_exact(4) {
        assert_eq!([px[0], px[1], px[2], px[3]], first);
    }
}
