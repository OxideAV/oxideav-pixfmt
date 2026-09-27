//! NEON (aarch64) implementations of the YUV ↔ RGB inner loops.
//!
//! Both directions are vectorised. Decode runs 8 pixels per step with
//! `vst3_u8` interleaving the RGB24 store; encode de-interleaves 16
//! RGB24 pixels per step with `vld3q_u8`, evaluates the Q15 matrix in
//! 32-bit lanes (`vmlal_n_s16`, exact — no 16-bit high-half
//! approximation), saturates to bytes exactly like the scalar clamp,
//! and forms the chroma box averages from the rounded per-pixel chroma
//! with pairwise adds and rounding narrows (`vpaddlq_u8` +
//! `vrshrn_n_u16`), which compute the scalar `(sum + 1) / 2` and
//! `(sum + 2) / 4` exactly. Every output byte equals the scalar
//! reference.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::yuv::{
    encode420_rows_from, encode422_row_from, encode444_row_from, yuv_to_rgb_fp, DecodeParams,
    EncodeParams, YuvMatrix, FP_HALF, FP_SHIFT,
};
use core::arch::aarch64::*;

const LANES: usize = 8;

#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_block_i32x8(
    y_lin: int32x4x2_t,
    cb: int32x4x2_t,
    cr: int32x4x2_t,
    d: &DecodeParams,
) -> (uint8x8_t, uint8x8_t, uint8x8_t) {
    let cr_r = vdupq_n_s32(d.cr_r);
    let cb_b = vdupq_n_s32(d.cb_b);
    let cg_cr = vdupq_n_s32(d.cg_cr);
    let cg_cb = vdupq_n_s32(d.cg_cb);
    let bias = vdupq_n_s32(FP_HALF);

    let r_lo = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(vaddq_s32(y_lin.0, vmulq_s32(cr_r, cr.0)), bias));
    let r_hi = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(vaddq_s32(y_lin.1, vmulq_s32(cr_r, cr.1)), bias));
    let b_lo = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(vaddq_s32(y_lin.0, vmulq_s32(cb_b, cb.0)), bias));
    let b_hi = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(vaddq_s32(y_lin.1, vmulq_s32(cb_b, cb.1)), bias));
    let g_lo = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(
        vsubq_s32(
            vsubq_s32(y_lin.0, vmulq_s32(cg_cr, cr.0)),
            vmulq_s32(cg_cb, cb.0),
        ),
        bias,
    ));
    let g_hi = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(
        vsubq_s32(
            vsubq_s32(y_lin.1, vmulq_s32(cg_cr, cr.1)),
            vmulq_s32(cg_cb, cb.1),
        ),
        bias,
    ));

    // Saturate i32 → u16 → u8.
    let r16 = vcombine_s16(vqmovn_s32(r_lo), vqmovn_s32(r_hi));
    let g16 = vcombine_s16(vqmovn_s32(g_lo), vqmovn_s32(g_hi));
    let b16 = vcombine_s16(vqmovn_s32(b_lo), vqmovn_s32(b_hi));
    (vqmovun_s16(r16), vqmovun_s16(g16), vqmovun_s16(b16))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load8_sub_i32(src: &[u8], off: i32) -> int32x4x2_t {
    let v = vld1_u8(src.as_ptr());
    let wide = vmovl_u8(v);
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(wide)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(wide)));
    let off_v = vdupq_n_s32(off);
    int32x4x2_t(vsubq_s32(lo, off_v), vsubq_s32(hi, off_v))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_chroma_4_broadcast(src: &[u8]) -> int32x4x2_t {
    // Load 4 chroma bytes and duplicate each to produce 8.
    let bytes = [
        src[0], src[0], src[1], src[1], src[2], src[2], src[3], src[3],
    ];
    let v = vld1_u8(bytes.as_ptr());
    let wide = vmovl_u8(v);
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(wide)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(wide)));
    let c128 = vdupq_n_s32(128);
    int32x4x2_t(vsubq_s32(lo, c128), vsubq_s32(hi, c128))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_rgb24_lane8(dst: &mut [u8], r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) {
    // NEON has `vst3_u8` which interleaves 3 u8x8 vectors — exactly what
    // we want for a 24-byte RGB24 store.
    vst3_u8(dst.as_mut_ptr(), uint8x8x3_t(r, g, b));
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv444_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let d = matrix.decode_params();
    let y_scale = vdupq_n_s32(d.y_scale);
    for row in 0..h {
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * w..row * w + w];
        let vrow = &vp[row * w..row * w + w];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let y = load8_sub_i32(&yrow[off..], d.y_off);
            let y_lin = int32x4x2_t(vmulq_s32(y.0, y_scale), vmulq_s32(y.1, y_scale));
            let cb = load8_sub_i32(&urow[off..], 128);
            let cr = load8_sub_i32(&vrow[off..], 128);
            let (rv, gv, bv) = decode_block_i32x8(y_lin, cb, cr, &d);
            store_rgb24_lane8(&mut drow[off * 3..off * 3 + LANES * 3], rv, gv, bv);
        }
        for col in (chunks * LANES)..w {
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[col], vrow[col], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv422_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w.div_ceil(2);
    let d = matrix.decode_params();
    let y_scale = vdupq_n_s32(d.y_scale);
    for row in 0..h {
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * cw..row * cw + cw];
        let vrow = &vp[row * cw..row * cw + cw];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let coff = off / 2;
            let y = load8_sub_i32(&yrow[off..], d.y_off);
            let y_lin = int32x4x2_t(vmulq_s32(y.0, y_scale), vmulq_s32(y.1, y_scale));
            let cb = load_chroma_4_broadcast(&urow[coff..]);
            let cr = load_chroma_4_broadcast(&vrow[coff..]);
            let (rv, gv, bv) = decode_block_i32x8(y_lin, cb, cr, &d);
            store_rgb24_lane8(&mut drow[off * 3..off * 3 + LANES * 3], rv, gv, bv);
        }
        for col in (chunks * LANES)..w {
            let cc = col / 2;
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[cc], vrow[cc], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv420_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w.div_ceil(2);
    let d = matrix.decode_params();
    let y_scale = vdupq_n_s32(d.y_scale);
    for row in 0..h {
        let cr = row / 2;
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[cr * cw..cr * cw + cw];
        let vrow = &vp[cr * cw..cr * cw + cw];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let coff = off / 2;
            let y = load8_sub_i32(&yrow[off..], d.y_off);
            let y_lin = int32x4x2_t(vmulq_s32(y.0, y_scale), vmulq_s32(y.1, y_scale));
            let cb = load_chroma_4_broadcast(&urow[coff..]);
            let cr_c = load_chroma_4_broadcast(&vrow[coff..]);
            let (rv, gv, bv) = decode_block_i32x8(y_lin, cb, cr_c, &d);
            store_rgb24_lane8(&mut drow[off * 3..off * 3 + LANES * 3], rv, gv, bv);
        }
        for col in (chunks * LANES)..w {
            let cc = col / 2;
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[cc], vrow[cc], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

// ---------------------------------------------------------------------
// Encode: RGB24 → planar YUV.

/// One Q15 dot product over 8 pixels: `(cr*r + cg*g + cb*b + bias) >> 15`
/// saturated to u8 — the scalar `rgb_to_yuv_fp` row for one component.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot8(
    r: int16x8_t,
    g: int16x8_t,
    b: int16x8_t,
    c: [i32; 3],
    bias: int32x4_t,
) -> uint8x8_t {
    let (cr, cg, cb) = (c[0] as i16, c[1] as i16, c[2] as i16);
    let mut lo = vmlal_n_s16(bias, vget_low_s16(r), cr);
    lo = vmlal_n_s16(lo, vget_low_s16(g), cg);
    lo = vmlal_n_s16(lo, vget_low_s16(b), cb);
    let mut hi = vmlal_n_s16(bias, vget_high_s16(r), cr);
    hi = vmlal_n_s16(hi, vget_high_s16(g), cg);
    hi = vmlal_n_s16(hi, vget_high_s16(b), cb);
    let lo = vshrq_n_s32::<FP_SHIFT>(lo);
    let hi = vshrq_n_s32::<FP_SHIFT>(hi);
    vqmovun_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)))
}

/// Encode 16 RGB24 pixels at `src` into per-pixel (Y, Cb, Cr) bytes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn encode16(src: *const u8, p: &EncodeParams) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    let px = vld3q_u8(src);
    let widen = |v: uint8x16_t| {
        (
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))),
        )
    };
    let (r0, r1) = widen(px.0);
    let (g0, g1) = widen(px.1);
    let (b0, b1) = widen(px.2);
    let yb = vdupq_n_s32(p.y_bias);
    let cb_bias = vdupq_n_s32(p.c_bias);
    let yc = [p.cy_r, p.cy_g, p.cy_b];
    let uc = [p.cb_r, p.cb_g, p.cb_b];
    let vc = [p.cr_r, p.cr_g, p.cr_b];
    let y = vcombine_u8(dot8(r0, g0, b0, yc, yb), dot8(r1, g1, b1, yc, yb));
    let u = vcombine_u8(dot8(r0, g0, b0, uc, cb_bias), dot8(r1, g1, b1, uc, cb_bias));
    let v = vcombine_u8(dot8(r0, g0, b0, vc, cb_bias), dot8(r1, g1, b1, vc, cb_bias));
    (y, u, v)
}

/// The Q15 encode coefficients all fit an i16 multiplicand (|c| < 2^15
/// for every matrix the crate builds); `vmlal_n_s16` relies on it.
#[inline]
fn coeffs_fit_i16(p: &EncodeParams) -> bool {
    [
        p.cy_r, p.cy_g, p.cy_b, p.cb_r, p.cb_g, p.cb_b, p.cr_r, p.cr_g, p.cr_b,
    ]
    .iter()
    .all(|&c| (i16::MIN as i32..=i16::MAX as i32).contains(&c))
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb24_to_yuv444(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let p = matrix.encode_params();
    let vec_ok = coeffs_fit_i16(&p);
    for row in 0..h {
        let srow = &src[row * w * 3..(row + 1) * w * 3];
        let yrow = &mut yp[row * w..(row + 1) * w];
        let urow = &mut up[row * w..(row + 1) * w];
        let vrow = &mut vp[row * w..(row + 1) * w];
        let blocks = if vec_ok { w / 16 } else { 0 };
        for blk in 0..blocks {
            let x = blk * 16;
            let (y, u, v) = encode16(srow.as_ptr().add(x * 3), &p);
            vst1q_u8(yrow.as_mut_ptr().add(x), y);
            vst1q_u8(urow.as_mut_ptr().add(x), u);
            vst1q_u8(vrow.as_mut_ptr().add(x), v);
        }
        encode444_row_from(srow, yrow, urow, vrow, w, blocks * 16, &p);
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb24_to_yuv422(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let p = matrix.encode_params();
    let cw = w.div_ceil(2);
    let vec_ok = coeffs_fit_i16(&p);
    for row in 0..h {
        let srow = &src[row * w * 3..(row + 1) * w * 3];
        let yrow = &mut yp[row * w..(row + 1) * w];
        let urow = &mut up[row * cw..(row + 1) * cw];
        let vrow = &mut vp[row * cw..(row + 1) * cw];
        let blocks = if vec_ok { w / 16 } else { 0 };
        for blk in 0..blocks {
            let x = blk * 16;
            let (y, u, v) = encode16(srow.as_ptr().add(x * 3), &p);
            vst1q_u8(yrow.as_mut_ptr().add(x), y);
            // (a + b + 1) >> 1 over adjacent pairs.
            vst1_u8(
                urow.as_mut_ptr().add(x / 2),
                vrshrn_n_u16::<1>(vpaddlq_u8(u)),
            );
            vst1_u8(
                vrow.as_mut_ptr().add(x / 2),
                vrshrn_n_u16::<1>(vpaddlq_u8(v)),
            );
        }
        encode422_row_from(srow, yrow, urow, vrow, w, blocks * 16, &p);
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb24_to_yuv420(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let p = matrix.encode_params();
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let vec_ok = coeffs_fit_i16(&p);
    for cr in 0..ch {
        let row_a = cr * 2;
        let sa = &src[row_a * w * 3..(row_a + 1) * w * 3];
        let urow = &mut up[cr * cw..(cr + 1) * cw];
        let vrow = &mut vp[cr * cw..(cr + 1) * cw];
        if row_a + 1 < h {
            let sb = &src[(row_a + 1) * w * 3..(row_a + 2) * w * 3];
            let (ya, yb) = yp[row_a * w..(row_a + 2) * w].split_at_mut(w);
            let blocks = if vec_ok { w / 16 } else { 0 };
            for blk in 0..blocks {
                let x = blk * 16;
                let (y0, u0, v0) = encode16(sa.as_ptr().add(x * 3), &p);
                let (y1, u1, v1) = encode16(sb.as_ptr().add(x * 3), &p);
                vst1q_u8(ya.as_mut_ptr().add(x), y0);
                vst1q_u8(yb.as_mut_ptr().add(x), y1);
                // (sum of the 2×2 block + 2) >> 2.
                let us = vaddq_u16(vpaddlq_u8(u0), vpaddlq_u8(u1));
                let vs = vaddq_u16(vpaddlq_u8(v0), vpaddlq_u8(v1));
                vst1_u8(urow.as_mut_ptr().add(x / 2), vrshrn_n_u16::<2>(us));
                vst1_u8(vrow.as_mut_ptr().add(x / 2), vrshrn_n_u16::<2>(vs));
            }
            encode420_rows_from(sa, Some((sb, yb)), ya, urow, vrow, w, blocks * 16, &p);
        } else {
            let ya = &mut yp[row_a * w..(row_a + 1) * w];
            let blocks = if vec_ok { w / 16 } else { 0 };
            for blk in 0..blocks {
                let x = blk * 16;
                let (y0, u0, v0) = encode16(sa.as_ptr().add(x * 3), &p);
                vst1q_u8(ya.as_mut_ptr().add(x), y0);
                // Odd height: the row is replicated into the block's
                // missing second row — (2·(a + b) + 2) >> 2.
                let us = vpaddlq_u8(u0);
                let vs = vpaddlq_u8(v0);
                vst1_u8(
                    urow.as_mut_ptr().add(x / 2),
                    vrshrn_n_u16::<2>(vaddq_u16(us, us)),
                );
                vst1_u8(
                    vrow.as_mut_ptr().add(x / 2),
                    vrshrn_n_u16::<2>(vaddq_u16(vs, vs)),
                );
            }
            encode420_rows_from(sa, None, ya, urow, vrow, w, blocks * 16, &p);
        }
    }
}
