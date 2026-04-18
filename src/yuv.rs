//! YUV ↔ RGB conversions (BT.601 and BT.709, limited and full range),
//! planar chroma resampling (4:2:0 ↔ 4:2:2 ↔ 4:4:4), and NV12/NV21
//! ↔ Yuv420P bridging.
//!
//! The per-pixel math runs in signed fixed-point (Q15) integer arithmetic
//! so the hot loops avoid f32 conversion and give a clean target for the
//! SIMD vectorisation in [`crate::yuv_simd`]. Scalar results match the
//! historical f32 implementation within ±1 LSB after rounding.

use crate::convert::ColorSpace;
use crate::yuv_simd;

/// BT.601 / BT.709 weight pair. The integer matrix used by the
/// converters below is derived from these f32 values.
#[derive(Clone, Copy)]
pub struct YuvMatrix {
    pub kr: f32,
    pub kb: f32,
    pub limited: bool,
}

impl YuvMatrix {
    pub const BT601: Self = Self {
        kr: 0.299,
        kb: 0.114,
        limited: true,
    };
    pub const BT709: Self = Self {
        kr: 0.2126,
        kb: 0.0722,
        limited: true,
    };
    pub fn with_range(mut self, limited: bool) -> Self {
        self.limited = limited;
        self
    }

    pub fn from_color_space(cs: ColorSpace) -> Self {
        match cs {
            ColorSpace::Bt601Limited => Self::BT601.with_range(true),
            ColorSpace::Bt601Full => Self::BT601.with_range(false),
            ColorSpace::Bt709Limited => Self::BT709.with_range(true),
            ColorSpace::Bt709Full => Self::BT709.with_range(false),
        }
    }
}

// ---------------------------------------------------------------------
// Fixed-point matrices (Q15).
//
// Encode  (RGB → YUV): y = (cy_r*r + cy_g*g + cy_b*b + y_bias) >> SHIFT
// Decode  (YUV → RGB): r = y_lin + (cr_coeff * (cr-128)) >> SHIFT
//                      b = y_lin + (cb_coeff * (cb-128)) >> SHIFT
//                      g = y_lin - (cg_cr*(cr-128) + cg_cb*(cb-128)) >> SHIFT
// For limited range, y_lin = ((y-16) * y_scale) >> SHIFT; scaling into the
// same 0..255 target space. The pre-shift rounding bias is folded into the
// offset terms where it matters.

pub(crate) const FP_SHIFT: i32 = 15;
pub(crate) const FP_ONE: i32 = 1 << FP_SHIFT;
pub(crate) const FP_HALF: i32 = 1 << (FP_SHIFT - 1);

#[derive(Clone, Copy, Debug)]
pub(crate) struct EncodeParams {
    // Y = (cy_r*r + cy_g*g + cy_b*b + y_bias) >> SHIFT
    pub cy_r: i32,
    pub cy_g: i32,
    pub cy_b: i32,
    pub y_bias: i32,
    // Cb = (cb_r*r + cb_g*g + cb_b*b + c_bias) >> SHIFT
    pub cb_r: i32,
    pub cb_g: i32,
    pub cb_b: i32,
    // Cr = (cr_r*r + cr_g*g + cr_b*b + c_bias) >> SHIFT
    pub cr_r: i32,
    pub cr_g: i32,
    pub cr_b: i32,
    pub c_bias: i32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodeParams {
    // y_lin = y_scale * (y - y_off), >> SHIFT then added to chroma term.
    pub y_scale: i32,
    pub y_off: i32,
    pub cr_r: i32,
    pub cb_b: i32,
    pub cg_cr: i32, // always positive; subtracted
    pub cg_cb: i32, // always positive; subtracted
}

/// Q15 rounding: `round(f * FP_ONE)` as i32. Manual rounding for negatives.
fn q15(f: f32) -> i32 {
    let v = f * FP_ONE as f32;
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        -((-v + 0.5) as i32)
    }
}

impl YuvMatrix {
    pub(crate) fn encode_params(&self) -> EncodeParams {
        let kr = self.kr;
        let kb = self.kb;
        let kg = 1.0 - kr - kb;
        // Limited: Y_lim = 16 + 219/255 * (kr*R + kg*G + kb*B)
        //          C_lim = 128 + 224/255 * (C - y_full) / (2*(1-k))
        let (ys, cs, y_off, c_off) = if self.limited {
            (219.0 / 255.0, 224.0 / 255.0, 16.0, 128.0)
        } else {
            (1.0, 1.0, 0.0, 128.0)
        };
        let cy_r = q15(kr * ys);
        let cy_g = q15(kg * ys);
        let cy_b = q15(kb * ys);
        // Cb = cs/(2*(1-kb)) * (B - y_full) = cs/(2*(1-kb)) * (-kr*R - kg*G + (1-kb)*B)
        let cb_scale = cs / (2.0 * (1.0 - kb));
        let cb_r = q15(cb_scale * -kr);
        let cb_g = q15(cb_scale * -kg);
        let cb_b = q15(cb_scale * (1.0 - kb));
        // Cr = cs/(2*(1-kr)) * (R - y_full) = cs/(2*(1-kr)) * ((1-kr)*R - kg*G - kb*B)
        let cr_scale = cs / (2.0 * (1.0 - kr));
        let cr_r = q15(cr_scale * (1.0 - kr));
        let cr_g = q15(cr_scale * -kg);
        let cr_b = q15(cr_scale * -kb);
        // Biases: fold offset and rounding (+0.5 LSB) into the bias term.
        let y_bias = ((y_off * FP_ONE as f32).round() as i32) + FP_HALF;
        let c_bias = ((c_off * FP_ONE as f32).round() as i32) + FP_HALF;
        EncodeParams {
            cy_r,
            cy_g,
            cy_b,
            y_bias,
            cb_r,
            cb_g,
            cb_b,
            cr_r,
            cr_g,
            cr_b,
            c_bias,
        }
    }

    pub(crate) fn decode_params(&self) -> DecodeParams {
        let kr = self.kr;
        let kb = self.kb;
        let kg = 1.0 - kr - kb;
        if self.limited {
            // y_lin = (y - 16) * 255/219
            // chroma: (c - 128) * 255/224 * 2*(1-k) = (c-128) * factor
            let y_scale = q15(255.0 / 219.0);
            let cr_r = q15(2.0 * (1.0 - kr) * (255.0 / 224.0));
            let cb_b = q15(2.0 * (1.0 - kb) * (255.0 / 224.0));
            // g = y - kr/kg * (r - y) - kb/kg * (b - y)
            //   = y_lin - (kr/kg * cr_delta + kb/kg * cb_delta)
            // cr_delta = (cr-128) * 2*(1-kr) * 255/224
            // kr/kg * 2*(1-kr) = 2*kr*(1-kr)/kg
            let cg_cr = q15((2.0 * kr * (1.0 - kr) / kg) * (255.0 / 224.0));
            let cg_cb = q15((2.0 * kb * (1.0 - kb) / kg) * (255.0 / 224.0));
            DecodeParams {
                y_scale,
                y_off: 16,
                cr_r,
                cb_b,
                cg_cr,
                cg_cb,
            }
        } else {
            let y_scale = FP_ONE;
            let cr_r = q15(2.0 * (1.0 - kr));
            let cb_b = q15(2.0 * (1.0 - kb));
            let cg_cr = q15(2.0 * kr * (1.0 - kr) / kg);
            let cg_cb = q15(2.0 * kb * (1.0 - kb) / kg);
            DecodeParams {
                y_scale,
                y_off: 0,
                cr_r,
                cb_b,
                cg_cr,
                cg_cb,
            }
        }
    }
}

#[inline]
pub(crate) fn clamp_u8_i32(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

// ---------------------------------------------------------------------
// Per-pixel scalar paths.

/// Encode a single (R, G, B) pixel into (Y, U, V) per `matrix`.
pub fn rgb_to_yuv(r: u8, g: u8, b: u8, matrix: YuvMatrix) -> (u8, u8, u8) {
    let p = matrix.encode_params();
    rgb_to_yuv_fp(r, g, b, &p)
}

#[inline]
pub(crate) fn rgb_to_yuv_fp(r: u8, g: u8, b: u8, p: &EncodeParams) -> (u8, u8, u8) {
    let ri = r as i32;
    let gi = g as i32;
    let bi = b as i32;
    let y = (p.cy_r * ri + p.cy_g * gi + p.cy_b * bi + p.y_bias) >> FP_SHIFT;
    let cb = (p.cb_r * ri + p.cb_g * gi + p.cb_b * bi + p.c_bias) >> FP_SHIFT;
    let cr = (p.cr_r * ri + p.cr_g * gi + p.cr_b * bi + p.c_bias) >> FP_SHIFT;
    (clamp_u8_i32(y), clamp_u8_i32(cb), clamp_u8_i32(cr))
}

/// Decode a single (Y, U, V) pixel into (R, G, B).
pub fn yuv_to_rgb(y: u8, cb: u8, cr: u8, matrix: YuvMatrix) -> (u8, u8, u8) {
    let d = matrix.decode_params();
    yuv_to_rgb_fp(y, cb, cr, &d)
}

#[inline]
pub(crate) fn yuv_to_rgb_fp(y: u8, cb: u8, cr: u8, d: &DecodeParams) -> (u8, u8, u8) {
    let yv = (y as i32 - d.y_off) * d.y_scale;
    let cbv = cb as i32 - 128;
    let crv = cr as i32 - 128;
    let r = (yv + d.cr_r * crv + FP_HALF) >> FP_SHIFT;
    let b = (yv + d.cb_b * cbv + FP_HALF) >> FP_SHIFT;
    let g = (yv - d.cg_cr * crv - d.cg_cb * cbv + FP_HALF) >> FP_SHIFT;
    (clamp_u8_i32(r), clamp_u8_i32(g), clamp_u8_i32(b))
}

// ---------------------------------------------------------------------
// Scalar-fixed-point planar converters. These are the golden fallback;
// SIMD dispatch delegates to them when the CPU lacks vector support or
// the frame is too small to vectorise.

pub(crate) fn yuv444_to_rgb24_scalar(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let d = matrix.decode_params();
    for row in 0..h {
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * w..row * w + w];
        let vrow = &vp[row * w..row * w + w];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        for col in 0..w {
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[col], vrow[col], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

pub(crate) fn yuv422_to_rgb24_scalar(
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
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * cw..row * cw + cw];
        let vrow = &vp[row * cw..row * cw + cw];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        for col in 0..w {
            let cc = col >> 1;
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[cc], vrow[cc], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

pub(crate) fn yuv420_to_rgb24_scalar(
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
        let cr = row >> 1;
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[cr * cw..cr * cw + cw];
        let vrow = &vp[cr * cw..cr * cw + cw];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        for col in 0..w {
            let cc = col >> 1;
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[cc], vrow[cc], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

pub(crate) fn rgb24_to_yuv444_scalar(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let p = matrix.encode_params();
    for row in 0..h {
        for col in 0..w {
            let o = (row * w + col) * 3;
            let (y, u, v) = rgb_to_yuv_fp(src[o], src[o + 1], src[o + 2], &p);
            yp[row * w + col] = y;
            up[row * w + col] = u;
            vp[row * w + col] = v;
        }
    }
}

pub(crate) fn rgb24_to_yuv422_scalar(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w / 2;
    let p = matrix.encode_params();
    for row in 0..h {
        for col in 0..w {
            let o = (row * w + col) * 3;
            let (y, _u, _v) = rgb_to_yuv_fp(src[o], src[o + 1], src[o + 2], &p);
            yp[row * w + col] = y;
        }
        for cc in 0..cw {
            let mut cbs = 0i32;
            let mut crs = 0i32;
            for dx in 0..2 {
                let col = cc * 2 + dx;
                let o = (row * w + col) * 3;
                let (_y, u, v) = rgb_to_yuv_fp(src[o], src[o + 1], src[o + 2], &p);
                cbs += u as i32;
                crs += v as i32;
            }
            up[row * cw + cc] = ((cbs + 1) / 2) as u8;
            vp[row * cw + cc] = ((crs + 1) / 2) as u8;
        }
    }
}

pub(crate) fn rgb24_to_yuv420_scalar(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w / 2;
    let ch = h / 2;
    let p = matrix.encode_params();
    for row in 0..h {
        for col in 0..w {
            let o = (row * w + col) * 3;
            let (y, _u, _v) = rgb_to_yuv_fp(src[o], src[o + 1], src[o + 2], &p);
            yp[row * w + col] = y;
        }
    }
    for cr in 0..ch {
        for cc in 0..cw {
            let mut cbs = 0i32;
            let mut crs = 0i32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let row = cr * 2 + dy;
                    let col = cc * 2 + dx;
                    let o = (row * w + col) * 3;
                    let (_y, u, v) = rgb_to_yuv_fp(src[o], src[o + 1], src[o + 2], &p);
                    cbs += u as i32;
                    crs += v as i32;
                }
            }
            up[cr * cw + cc] = ((cbs + 2) / 4) as u8;
            vp[cr * cw + cc] = ((crs + 2) / 4) as u8;
        }
    }
}

// ---------------------------------------------------------------------
// Public dispatching entrypoints. The SIMD module picks the best path
// available at runtime (scalar / AVX2 / NEON / std::simd).

pub fn yuv444_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    debug_assert!(dst.len() >= w * h * 3);
    yuv_simd::yuv444_to_rgb24(yp, up, vp, dst, w, h, matrix);
}

pub fn yuv422_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    yuv_simd::yuv422_to_rgb24(yp, up, vp, dst, w, h, matrix);
}

pub fn yuv420_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    yuv_simd::yuv420_to_rgb24(yp, up, vp, dst, w, h, matrix);
}

pub fn rgb24_to_yuv444(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    yuv_simd::rgb24_to_yuv444(src, yp, up, vp, w, h, matrix);
}

pub fn rgb24_to_yuv422(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    yuv_simd::rgb24_to_yuv422(src, yp, up, vp, w, h, matrix);
}

pub fn rgb24_to_yuv420(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    yuv_simd::rgb24_to_yuv420(src, yp, up, vp, w, h, matrix);
}

// ---------------------------------------------------------------------
// Planar ↔ planar subsample conversions (kept scalar — cheap already).

pub fn chroma_444_to_422(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        for cc in 0..cw {
            let a = src[row * w + cc * 2] as u16;
            let b = src[row * w + cc * 2 + 1] as u16;
            dst[row * cw + cc] = (a + b).div_ceil(2) as u8;
        }
    }
}

pub fn chroma_422_to_444(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        for col in 0..w {
            let cc = col / 2;
            dst[row * w + col] = src[row * cw + cc];
        }
    }
}

pub fn chroma_444_to_420(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    let ch = h / 2;
    for cr in 0..ch {
        for cc in 0..cw {
            let mut s = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    s += src[(cr * 2 + dy) * w + cc * 2 + dx] as u32;
                }
            }
            dst[cr * cw + cc] = ((s + 2) / 4) as u8;
        }
    }
}

pub fn chroma_420_to_444(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        let cr = row / 2;
        for col in 0..w {
            let cc = col / 2;
            dst[row * w + col] = src[cr * cw + cc];
        }
    }
}

pub fn chroma_422_to_420(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    let ch = h / 2;
    for cr in 0..ch {
        for cc in 0..cw {
            let a = src[(cr * 2) * cw + cc] as u16;
            let b = src[(cr * 2 + 1) * cw + cc] as u16;
            dst[cr * cw + cc] = (a + b).div_ceil(2) as u8;
        }
    }
}

pub fn chroma_420_to_422(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        let cr = row / 2;
        for cc in 0..cw {
            dst[row * cw + cc] = src[cr * cw + cc];
        }
    }
}

// ---------------------------------------------------------------------
// NV12 / NV21 ↔ Yuv420P.

pub fn nv12_uv_split(uv: &[u8], up: &mut [u8], vp: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        up[i] = uv[i * 2];
        vp[i] = uv[i * 2 + 1];
    }
}

pub fn nv21_vu_split(vu: &[u8], up: &mut [u8], vp: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        vp[i] = vu[i * 2];
        up[i] = vu[i * 2 + 1];
    }
}

pub fn nv12_uv_merge(up: &[u8], vp: &[u8], uv: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        uv[i * 2] = up[i];
        uv[i * 2 + 1] = vp[i];
    }
}

pub fn nv21_vu_merge(up: &[u8], vp: &[u8], vu: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        vu[i * 2] = vp[i];
        vu[i * 2 + 1] = up[i];
    }
}

// ---------------------------------------------------------------------
// Full/limited range plane conversion for YuvJ* ↔ Yuv*.
// Fixed-point per-byte scaling avoids f32 in the hot loop.

pub fn limited_to_full_luma(plane: &mut [u8]) {
    // scale = 255/219 ≈ 1.16438
    const SCALE: i32 = ((255 * FP_ONE as i64) / 219) as i32;
    for b in plane.iter_mut() {
        let v = (*b as i32 - 16) * SCALE + FP_HALF;
        let v = v >> FP_SHIFT;
        *b = clamp_u8_i32(v);
    }
}

pub fn limited_to_full_chroma(plane: &mut [u8]) {
    // scale = 255/224
    const SCALE: i32 = ((255 * FP_ONE as i64) / 224) as i32;
    for b in plane.iter_mut() {
        let v = (*b as i32 - 128) * SCALE + (128 << FP_SHIFT) + FP_HALF;
        let v = v >> FP_SHIFT;
        *b = clamp_u8_i32(v);
    }
}

pub fn full_to_limited_luma(plane: &mut [u8]) {
    // scale = 219/255
    const SCALE: i32 = ((219 * FP_ONE as i64) / 255) as i32;
    for b in plane.iter_mut() {
        let v = (*b as i32) * SCALE + (16 << FP_SHIFT) + FP_HALF;
        let v = v >> FP_SHIFT;
        *b = clamp_u8_i32(v);
    }
}

pub fn full_to_limited_chroma(plane: &mut [u8]) {
    // scale = 224/255
    const SCALE: i32 = ((224 * FP_ONE as i64) / 255) as i32;
    for b in plane.iter_mut() {
        let v = (*b as i32 - 128) * SCALE + (128 << FP_SHIFT) + FP_HALF;
        let v = v >> FP_SHIFT;
        *b = clamp_u8_i32(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // BT.709 limited-range test vectors. These are the reference values
    // produced by the canonical f32 math, and the fixed-point path must
    // match within ±1 LSB (documented tolerance for int rounding).
    #[test]
    fn bt709_limited_known_vectors() {
        let m = YuvMatrix::BT709.with_range(true);
        // (R, G, B, expected Y, expected U, expected V) from f32 reference.
        let cases = [
            (0u8, 0u8, 0u8, 16u8, 128u8, 128u8),
            (255, 255, 255, 235, 128, 128),
            (255, 0, 0, 63, 102, 240),
            (0, 255, 0, 173, 42, 26),
            (0, 0, 255, 32, 240, 118),
            (128, 128, 128, 126, 128, 128),
        ];
        for (r, g, b, ey, eu, ev) in cases {
            let (y, u, v) = rgb_to_yuv(r, g, b, m);
            assert!(
                (y as i32 - ey as i32).abs() <= 1,
                "Y mismatch for ({r},{g},{b}): got {y}, want {ey}"
            );
            assert!(
                (u as i32 - eu as i32).abs() <= 1,
                "U mismatch for ({r},{g},{b}): got {u}, want {eu}"
            );
            assert!(
                (v as i32 - ev as i32).abs() <= 1,
                "V mismatch for ({r},{g},{b}): got {v}, want {ev}"
            );
        }
    }

    #[test]
    fn bt709_limited_decode_vectors() {
        let m = YuvMatrix::BT709.with_range(true);
        // Decoding the encoded values should round-trip to within ±2 LSB
        // for primary colours (combined encode+decode error budget).
        let rgbs = [
            (0u8, 0u8, 0u8),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
        ];
        for (r, g, b) in rgbs {
            let (y, u, v) = rgb_to_yuv(r, g, b, m);
            let (r2, g2, b2) = yuv_to_rgb(y, u, v, m);
            assert!(
                (r2 as i32 - r as i32).abs() <= 2,
                "R mismatch: ({r},{g},{b}) → ({r2},{g2},{b2})"
            );
            assert!(
                (g2 as i32 - g as i32).abs() <= 2,
                "G mismatch: ({r},{g},{b}) → ({r2},{g2},{b2})"
            );
            assert!(
                (b2 as i32 - b as i32).abs() <= 2,
                "B mismatch: ({r},{g},{b}) → ({r2},{g2},{b2})"
            );
        }
    }
}
