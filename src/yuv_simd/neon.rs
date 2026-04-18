//! NEON (aarch64) implementations of the YUV ↔ RGB inner loops.
//!
//! Only the decode direction is vectorised; RGB24's 3-byte-packed stream
//! makes the encode path more awkward and we dispatch it to the scalar
//! fallback for now.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::yuv::{yuv_to_rgb_fp, DecodeParams, YuvMatrix, FP_HALF, FP_SHIFT};
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

    let r_lo = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(
        vaddq_s32(y_lin.0, vmulq_s32(cr_r, cr.0)),
        bias,
    ));
    let r_hi = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(
        vaddq_s32(y_lin.1, vmulq_s32(cr_r, cr.1)),
        bias,
    ));
    let b_lo = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(
        vaddq_s32(y_lin.0, vmulq_s32(cb_b, cb.0)),
        bias,
    ));
    let b_hi = vshrq_n_s32::<FP_SHIFT>(vaddq_s32(
        vaddq_s32(y_lin.1, vmulq_s32(cb_b, cb.1)),
        bias,
    ));
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
    (
        vqmovun_s16(r16),
        vqmovun_s16(g16),
        vqmovun_s16(b16),
    )
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
    vst3_u8(
        dst.as_mut_ptr(),
        uint8x8x3_t(r, g, b),
    );
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
    let cw = w / 2;
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
    let cw = w / 2;
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
