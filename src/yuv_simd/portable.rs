//! Portable SIMD (`std::simd`) YUV decode path. Requires nightly.
//!
//! Only `yuv420_to_rgb24` is vectorised — it's the hottest path in the
//! framework. The scalar fixed-point fallback handles everything else,
//! which keeps the nightly-only surface small and the stable-toolchain
//! scalar/AVX2/NEON codepaths unaffected.

#![allow(clippy::needless_range_loop)]

use crate::yuv::{yuv_to_rgb_fp, DecodeParams, YuvMatrix, FP_HALF, FP_SHIFT};
use core::simd::prelude::*;
use core::simd::{i32x8, u8x8};

const LANES: usize = 8;

#[inline]
fn decode_lane(
    y_bytes: [u8; LANES],
    cb_bytes: [u8; LANES],
    cr_bytes: [u8; LANES],
    d: &DecodeParams,
) -> ([u8; LANES], [u8; LANES], [u8; LANES]) {
    let y = u8x8::from_array(y_bytes).cast::<i32>();
    let cb = u8x8::from_array(cb_bytes).cast::<i32>();
    let cr = u8x8::from_array(cr_bytes).cast::<i32>();
    let y_off = i32x8::splat(d.y_off);
    let c128 = i32x8::splat(128);
    let y_scale = i32x8::splat(d.y_scale);
    let cr_r = i32x8::splat(d.cr_r);
    let cb_b = i32x8::splat(d.cb_b);
    let cg_cr = i32x8::splat(d.cg_cr);
    let cg_cb = i32x8::splat(d.cg_cb);
    let bias = i32x8::splat(FP_HALF);
    let shift = i32x8::splat(FP_SHIFT);
    let y_lin = (y - y_off) * y_scale;
    let cbv = cb - c128;
    let crv = cr - c128;
    let r = (y_lin + cr_r * crv + bias) >> shift;
    let b = (y_lin + cb_b * cbv + bias) >> shift;
    let g = (y_lin - cg_cr * crv - cg_cb * cbv + bias) >> shift;
    let clamp_vec = |v: i32x8| -> [u8; LANES] {
        let clamped = v.simd_clamp(i32x8::splat(0), i32x8::splat(255));
        let mut out = [0u8; LANES];
        let arr = clamped.to_array();
        for i in 0..LANES {
            out[i] = arr[i] as u8;
        }
        out
    };
    (clamp_vec(r), clamp_vec(g), clamp_vec(b))
}

pub(crate) fn yuv420_to_rgb24(
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
            let mut y_bytes = [0u8; LANES];
            let mut cb_bytes = [0u8; LANES];
            let mut cr_bytes = [0u8; LANES];
            y_bytes.copy_from_slice(&yrow[off..off + LANES]);
            for i in 0..LANES {
                let cc = i / 2;
                cb_bytes[i] = urow[coff + cc];
                cr_bytes[i] = vrow[coff + cc];
            }
            let (rs, gs, bs) = decode_lane(y_bytes, cb_bytes, cr_bytes, &d);
            for i in 0..LANES {
                drow[(off + i) * 3] = rs[i];
                drow[(off + i) * 3 + 1] = gs[i];
                drow[(off + i) * 3 + 2] = bs[i];
            }
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
