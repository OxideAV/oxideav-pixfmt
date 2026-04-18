//! Bit-exact comparison between the `rgb::swizzle*` vectorised paths
//! and a reference scalar implementation.
//!
//! The tests exercise every Rgba4/Rgb3 permutation pair, at widths that
//! span below the SIMD block size, at the SIMD block boundary, and
//! above it so the tail handling is covered.

use oxideav_pixfmt::rgb::{
    self, Rgb3, Rgba4, ABGR_POS, ARGB_POS, BGRA_POS, BGR_POS, RGBA_POS, RGB_POS,
};

fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        out.push((state >> 32) as u8);
    }
    out
}

fn reference_swizzle4(src: &[u8], src_pos: Rgba4, dst: &mut [u8], dst_pos: Rgba4, pixels: usize) {
    for i in 0..pixels {
        let s = i * 4;
        let d = i * 4;
        dst[d + dst_pos.r] = src[s + src_pos.r];
        dst[d + dst_pos.g] = src[s + src_pos.g];
        dst[d + dst_pos.b] = src[s + src_pos.b];
        dst[d + dst_pos.a] = src[s + src_pos.a];
    }
}

fn reference_swizzle3(src: &[u8], src_pos: Rgb3, dst: &mut [u8], dst_pos: Rgb3, pixels: usize) {
    for i in 0..pixels {
        let s = i * 3;
        let d = i * 3;
        dst[d + dst_pos.r] = src[s + src_pos.r];
        dst[d + dst_pos.g] = src[s + src_pos.g];
        dst[d + dst_pos.b] = src[s + src_pos.b];
    }
}

#[test]
fn swizzle4_matches_reference_for_every_pair() {
    let layouts = [RGBA_POS, BGRA_POS, ARGB_POS, ABGR_POS];
    // Sizes: below the 8-pixel AVX2 block, at the boundary, above with
    // an awkward tail.
    let widths = [1usize, 3, 7, 8, 9, 15, 16, 23, 31, 64, 129];
    for &w in &widths {
        let src = lcg_bytes(0x41 ^ w as u64, w * 4);
        for &sp in &layouts {
            for &dp in &layouts {
                let mut got = vec![0u8; w * 4];
                let mut want = vec![0u8; w * 4];
                rgb::swizzle4(&src, sp, &mut got, dp, w);
                reference_swizzle4(&src, sp, &mut want, dp, w);
                assert_eq!(
                    got,
                    want,
                    "swizzle4 w={w} src={:?} dst={:?}",
                    (sp.r, sp.g, sp.b, sp.a),
                    (dp.r, dp.g, dp.b, dp.a),
                );
            }
        }
    }
}

fn reference_rgb3_to_rgba4(
    src: &[u8],
    src_pos: Rgb3,
    dst: &mut [u8],
    dst_pos: Rgba4,
    pixels: usize,
) {
    for i in 0..pixels {
        let s = i * 3;
        let d = i * 4;
        dst[d + dst_pos.r] = src[s + src_pos.r];
        dst[d + dst_pos.g] = src[s + src_pos.g];
        dst[d + dst_pos.b] = src[s + src_pos.b];
        dst[d + dst_pos.a] = 255;
    }
}

fn reference_rgba4_to_rgb3(
    src: &[u8],
    src_pos: Rgba4,
    dst: &mut [u8],
    dst_pos: Rgb3,
    pixels: usize,
) {
    for i in 0..pixels {
        let s = i * 4;
        let d = i * 3;
        dst[d + dst_pos.r] = src[s + src_pos.r];
        dst[d + dst_pos.g] = src[s + src_pos.g];
        dst[d + dst_pos.b] = src[s + src_pos.b];
    }
}

#[test]
fn rgb3_to_rgba4_matches_reference() {
    let rgb3_layouts = [RGB_POS, BGR_POS];
    let rgba4_layouts = [RGBA_POS, BGRA_POS, ARGB_POS, ABGR_POS];
    let widths = [1usize, 3, 4, 5, 7, 8, 12, 16, 31, 64, 127];
    for &w in &widths {
        let src = lcg_bytes(0x33 ^ w as u64, w * 3);
        for &sp in &rgb3_layouts {
            for &dp in &rgba4_layouts {
                let mut got = vec![0u8; w * 4];
                let mut want = vec![0u8; w * 4];
                rgb::rgb3_to_rgba4(&src, sp, &mut got, dp, w);
                reference_rgb3_to_rgba4(&src, sp, &mut want, dp, w);
                assert_eq!(got, want, "rgb3→rgba4 w={w}");
            }
        }
    }
}

#[test]
fn rgba4_to_rgb3_matches_reference() {
    let rgba4_layouts = [RGBA_POS, BGRA_POS, ARGB_POS, ABGR_POS];
    let rgb3_layouts = [RGB_POS, BGR_POS];
    let widths = [1usize, 3, 4, 5, 7, 8, 12, 16, 31, 64, 127];
    for &w in &widths {
        let src = lcg_bytes(0x55 ^ w as u64, w * 4);
        for &sp in &rgba4_layouts {
            for &dp in &rgb3_layouts {
                let mut got = vec![0u8; w * 3];
                let mut want = vec![0u8; w * 3];
                rgb::rgba4_to_rgb3(&src, sp, &mut got, dp, w);
                reference_rgba4_to_rgb3(&src, sp, &mut want, dp, w);
                assert_eq!(got, want, "rgba4→rgb3 w={w}");
            }
        }
    }
}

#[test]
fn swizzle3_matches_reference_for_every_pair() {
    let layouts = [RGB_POS, BGR_POS];
    // swizzle3 processes 5 pixels per AVX2 iteration — exercise the
    // under-/at-/over-boundary cases.
    let widths = [1usize, 4, 5, 6, 9, 10, 11, 16, 31, 64, 127];
    for &w in &widths {
        let src = lcg_bytes(0x77 ^ w as u64, w * 3);
        for &sp in &layouts {
            for &dp in &layouts {
                let mut got = vec![0u8; w * 3];
                let mut want = vec![0u8; w * 3];
                rgb::swizzle3(&src, sp, &mut got, dp, w);
                reference_swizzle3(&src, sp, &mut want, dp, w);
                assert_eq!(
                    got,
                    want,
                    "swizzle3 w={w} src={:?} dst={:?}",
                    (sp.r, sp.g, sp.b),
                    (dp.r, dp.g, dp.b),
                );
            }
        }
    }
}
