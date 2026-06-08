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
    /// BT.2020 non-constant luminance Y'CbCr coefficients per
    /// `docs/video/signal-metadata/R-REC-BT.2020-2-201510-I.pdf` Table 4
    /// (NCL column): `Y' = 0.2627 R' + 0.6780 G' + 0.0593 B'`. The Cb
    /// divisor 1.8814 = 2*(1 - 0.0593) and Cr divisor 1.4746 = 2*(1 - 0.2627)
    /// fall out of the standard k-coefficient construction shared with
    /// BT.709, so the Q15 matrices generated here are correct without
    /// additional special-casing. Also used (with identical NCL
    /// coefficients) for BT.2100 Y'C'BC'R signal format per
    /// `R-REC-BT.2100-3-202502-I.pdf` Table 6.
    pub const BT2020: Self = Self {
        kr: 0.2627,
        kb: 0.0593,
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
            ColorSpace::Bt2020Limited => Self::BT2020.with_range(true),
            ColorSpace::Bt2020Full => Self::BT2020.with_range(false),
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
        dup_bytes_2x(
            &src[row * cw..row * cw + cw],
            &mut dst[row * w..row * w + w],
            cw,
        );
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
        dup_bytes_2x(
            &src[cr * cw..cr * cw + cw],
            &mut dst[row * w..row * w + w],
            cw,
        );
    }
}

/// Copy `src[0..n]` into `dst[0..2n]`, duplicating every byte so
/// `dst[2i] == dst[2i+1] == src[i]`. Vectorised via AVX2 when
/// available.
#[inline]
fn dup_bytes_2x(src: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n * 2);
    if crate::simd_dispatch::has_avx2() {
        // SAFETY: path() guards AVX2 feature detection.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            dup_bytes_2x_avx2(src, dst, n);
            return;
        }
    }
    for i in 0..n {
        dst[i * 2] = src[i];
        dst[i * 2 + 1] = src[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dup_bytes_2x_avx2(src: &[u8], dst: &mut [u8], n: usize) {
    use core::arch::x86_64::*;
    // 16 source bytes → 32 dest bytes per iteration via two
    // _mm_unpack[lo|hi]_epi8(v, v) that interleave the vector with
    // itself. The unpack produces a duplicated-byte stream directly.
    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let v = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
        let lo = _mm_unpacklo_epi8(v, v);
        let hi = _mm_unpackhi_epi8(v, v);
        _mm_storeu_si128(dst.as_mut_ptr().add(off * 2) as *mut __m128i, lo);
        _mm_storeu_si128(dst.as_mut_ptr().add(off * 2 + 16) as *mut __m128i, hi);
    }
    let tail = chunks * 16;
    for i in tail..n {
        dst[i * 2] = src[i];
        dst[i * 2 + 1] = src[i];
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
// 4:1:1 chroma resamplers — horizontal subsample by 4 (one chroma sample
// per 4 luma samples on the same row). `Yuv411P` is the native NTSC
// DV-25 layout and a legal JPEG sampling pattern (luma h_samp=4,
// chroma h_samp=1). The vertical dimension is unchanged from 4:4:4 /
// 4:2:2, so the shrinkers box-average four horizontal samples and the
// expanders broadcast each chroma sample to four luma columns —
// horizontal mirrors of the 4:2:2 ↔ 4:4:4 helpers above.

/// 4:4:4 chroma → 4:1:1 chroma (horizontal 4-sample box average; vertical
/// dimension unchanged). `w` is the destination's full image width — the
/// source plane is `w × h`, the destination is `(w / 4) × h`. `w` must
/// be a multiple of 4 (4:1:1 has no representation for a 1-, 2-, or
/// 3-luma trailing column).
pub fn chroma_444_to_411(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 4;
    debug_assert_eq!(w, cw * 4, "chroma_444_to_411: width must be /4");
    for row in 0..h {
        for cc in 0..cw {
            let base = row * w + cc * 4;
            // +2 rounds to nearest on the four-sample sum (matches the
            // (a + b + 2) / 4 rounding already used by chroma_444_to_420).
            let s = src[base] as u32
                + src[base + 1] as u32
                + src[base + 2] as u32
                + src[base + 3] as u32;
            dst[row * cw + cc] = ((s + 2) / 4) as u8;
        }
    }
}

/// 4:1:1 chroma → 4:4:4 chroma (horizontal nearest, broadcasting each
/// chroma sample to four luma columns; vertical dimension unchanged).
/// `w` is the destination's full image width; the source plane is
/// `(w / 4) × h`, the destination is `w × h`.
pub fn chroma_411_to_444(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 4;
    debug_assert_eq!(w, cw * 4, "chroma_411_to_444: width must be /4");
    for row in 0..h {
        for cc in 0..cw {
            let v = src[row * cw + cc];
            let off = row * w + cc * 4;
            dst[off] = v;
            dst[off + 1] = v;
            dst[off + 2] = v;
            dst[off + 3] = v;
        }
    }
}

/// 4:2:2 chroma → 4:1:1 chroma (horizontal pair-average; vertical
/// unchanged). `w` is the image width; source is `(w / 2) × h`,
/// destination is `(w / 4) × h`.
pub fn chroma_422_to_411(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let src_cw = w / 2;
    let dst_cw = w / 4;
    debug_assert_eq!(w, dst_cw * 4, "chroma_422_to_411: width must be /4");
    for row in 0..h {
        for cc in 0..dst_cw {
            let base = row * src_cw + cc * 2;
            let s = src[base] as u16 + src[base + 1] as u16;
            dst[row * dst_cw + cc] = s.div_ceil(2) as u8;
        }
    }
}

/// 4:1:1 chroma → 4:2:2 chroma (horizontal pair-duplicate; vertical
/// unchanged). Source is `(w / 4) × h`, destination is `(w / 2) × h`.
pub fn chroma_411_to_422(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let src_cw = w / 4;
    let dst_cw = w / 2;
    debug_assert_eq!(w, src_cw * 4, "chroma_411_to_422: width must be /4");
    for row in 0..h {
        for cc in 0..src_cw {
            let v = src[row * src_cw + cc];
            let off = row * dst_cw + cc * 2;
            dst[off] = v;
            dst[off + 1] = v;
        }
    }
}

/// 4:2:0 chroma → 4:1:1 chroma. 4:2:0 is `(w / 2) × (h / 2)` and 4:1:1
/// is `(w / 4) × h`. Each destination row consumes the same source
/// chroma row (since 4:2:0 already pair-averaged the vertical pair, the
/// 4:1:1 row above and the 4:1:1 row below share the chroma value) and
/// horizontally pair-averages two source samples into one destination
/// sample. `w` and `h` must both be even (4:2:0 requirement) and `w`
/// must additionally be a multiple of 4.
pub fn chroma_420_to_411(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let src_cw = w / 2;
    let src_ch = h / 2;
    let dst_cw = w / 4;
    debug_assert_eq!(w, dst_cw * 4, "chroma_420_to_411: width must be /4");
    debug_assert_eq!(h, src_ch * 2, "chroma_420_to_411: height must be /2");
    for row in 0..h {
        let src_row = row / 2;
        for cc in 0..dst_cw {
            let base = src_row * src_cw + cc * 2;
            let s = src[base] as u16 + src[base + 1] as u16;
            dst[row * dst_cw + cc] = s.div_ceil(2) as u8;
        }
    }
}

/// 4:1:1 chroma → 4:2:0 chroma. 4:1:1 is `(w / 4) × h` and 4:2:0 is
/// `(w / 2) × (h / 2)`. Each destination chroma row is the vertical
/// average of two source rows; each destination sample is the horizontal
/// duplicate of the source sample broadcast to two columns.
pub fn chroma_411_to_420(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let src_cw = w / 4;
    let dst_cw = w / 2;
    let dst_ch = h / 2;
    debug_assert_eq!(w, src_cw * 4, "chroma_411_to_420: width must be /4");
    debug_assert_eq!(h, dst_ch * 2, "chroma_411_to_420: height must be /2");
    for cr in 0..dst_ch {
        for cc in 0..src_cw {
            let a = src[(cr * 2) * src_cw + cc] as u16;
            let b = src[(cr * 2 + 1) * src_cw + cc] as u16;
            let v = (a + b).div_ceil(2) as u8;
            // Broadcast vertical-averaged chroma sample to two destination
            // columns (4:2:0 has cw = src_cw * 2).
            let off = cr * dst_cw + cc * 2;
            dst[off] = v;
            dst[off + 1] = v;
        }
    }
}

// ---------------------------------------------------------------------
// NV12 / NV21 ↔ Yuv420P.

pub fn nv12_uv_split(uv: &[u8], up: &mut [u8], vp: &mut [u8], cw: usize, ch: usize) {
    deinterleave_u8_pair(uv, up, vp, cw * ch);
}

pub fn nv21_vu_split(vu: &[u8], up: &mut [u8], vp: &mut [u8], cw: usize, ch: usize) {
    // NV21 is VUVU...: the even byte is V, the odd byte is U. Same
    // deinterleave as NV12 with the two output slots swapped.
    deinterleave_u8_pair(vu, vp, up, cw * ch);
}

/// Split an interleaved byte-pair stream `src[0..2n]` into two output
/// planes `a[0..n]` (even bytes) and `b[0..n]` (odd bytes). Vectorised
/// via AVX2 when available.
#[inline]
fn deinterleave_u8_pair(src: &[u8], a: &mut [u8], b: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n * 2 && a.len() >= n && b.len() >= n);
    if crate::simd_dispatch::has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            deinterleave_u8_pair_avx2(src, a, b, n);
            return;
        }
    }
    for i in 0..n {
        a[i] = src[i * 2];
        b[i] = src[i * 2 + 1];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_u8_pair_avx2(src: &[u8], a: &mut [u8], b: &mut [u8], n: usize) {
    use core::arch::x86_64::*;
    // 32 source bytes per iteration (= 16 output bytes per stream).
    // Two pshufb on each __m128i pull the evens / odds apart.
    const MASK_EVEN: [u8; 16] = [
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ];
    const MASK_ODD: [u8; 16] = [
        1, 3, 5, 7, 9, 11, 13, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ];
    let m_even = _mm_loadu_si128(MASK_EVEN.as_ptr() as *const __m128i);
    let m_odd = _mm_loadu_si128(MASK_ODD.as_ptr() as *const __m128i);

    let chunks = n / 16;
    for c in 0..chunks {
        let soff = c * 32;
        let doff = c * 16;
        let v0 = _mm_loadu_si128(src.as_ptr().add(soff) as *const __m128i);
        let v1 = _mm_loadu_si128(src.as_ptr().add(soff + 16) as *const __m128i);
        // Low half of each: bytes 0..7 valid, rest zero.
        let a0 = _mm_shuffle_epi8(v0, m_even);
        let b0 = _mm_shuffle_epi8(v0, m_odd);
        let a1 = _mm_shuffle_epi8(v1, m_even);
        let b1 = _mm_shuffle_epi8(v1, m_odd);
        // Combine: pack the two halves into one 16-byte register.
        let a_comb = _mm_unpacklo_epi64(a0, a1);
        let b_comb = _mm_unpacklo_epi64(b0, b1);
        _mm_storeu_si128(a.as_mut_ptr().add(doff) as *mut __m128i, a_comb);
        _mm_storeu_si128(b.as_mut_ptr().add(doff) as *mut __m128i, b_comb);
    }
    let tail = chunks * 16;
    for i in tail..n {
        a[i] = src[i * 2];
        b[i] = src[i * 2 + 1];
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
// Packed 4:2:2 (YUYV / UYVY) ↔ planar Yuv422P.
//
// YUYV (Y0 U0 Y1 V0) and UYVY (U0 Y0 V0 Y1) are the two byte orderings
// used by every common 4:2:2 packed format. Each pair of horizontal
// pixels shares one (U, V) chroma sample, identical to Yuv422P planar
// chroma layout. Conversion is a pure shuffle — no colour math — so
// the routines below are byte-permutations only.
//
// `w` is the pixel width; it MUST be even (the format has no concept
// of an odd-width pair). `src` carries `w * h * 2` bytes (packed)
// and `dst` planes carry `w * h` for luma, `(w/2) * h` for each
// chroma plane. The caller is responsible for plane sizing; debug
// asserts pin the contract.

/// Split a packed YUYV (Y0 U0 Y1 V0) buffer into a Yuv422P planar
/// layout (Y full-res, U and V each `w/2 × h`). Pure deinterleave —
/// no colour math, no rounding.
pub fn yuyv422_to_yuv422p(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
) {
    debug_assert!(w % 2 == 0, "YUYV requires even width");
    let cw = w / 2;
    debug_assert!(src.len() >= w * h * 2);
    debug_assert!(yp.len() >= w * h);
    debug_assert!(up.len() >= cw * h);
    debug_assert!(vp.len() >= cw * h);
    for row in 0..h {
        let s = &src[row * w * 2..row * w * 2 + w * 2];
        let y_row = &mut yp[row * w..row * w + w];
        let u_row = &mut up[row * cw..row * cw + cw];
        let v_row = &mut vp[row * cw..row * cw + cw];
        for cc in 0..cw {
            // packed quad: Y0 U Y1 V
            y_row[cc * 2] = s[cc * 4];
            u_row[cc] = s[cc * 4 + 1];
            y_row[cc * 2 + 1] = s[cc * 4 + 2];
            v_row[cc] = s[cc * 4 + 3];
        }
    }
}

/// Split a packed UYVY (U0 Y0 V0 Y1) buffer into a Yuv422P planar
/// layout. Identical to [`yuyv422_to_yuv422p`] with luma and chroma
/// byte positions swapped within each quad.
pub fn uyvy422_to_yuv422p(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
) {
    debug_assert!(w % 2 == 0, "UYVY requires even width");
    let cw = w / 2;
    debug_assert!(src.len() >= w * h * 2);
    debug_assert!(yp.len() >= w * h);
    debug_assert!(up.len() >= cw * h);
    debug_assert!(vp.len() >= cw * h);
    for row in 0..h {
        let s = &src[row * w * 2..row * w * 2 + w * 2];
        let y_row = &mut yp[row * w..row * w + w];
        let u_row = &mut up[row * cw..row * cw + cw];
        let v_row = &mut vp[row * cw..row * cw + cw];
        for cc in 0..cw {
            // packed quad: U Y0 V Y1
            u_row[cc] = s[cc * 4];
            y_row[cc * 2] = s[cc * 4 + 1];
            v_row[cc] = s[cc * 4 + 2];
            y_row[cc * 2 + 1] = s[cc * 4 + 3];
        }
    }
}

/// Merge planar Yuv422P (`yp` full-res; `up`, `vp` each `w/2 × h`)
/// into a packed YUYV byte stream `dst` of length `w * h * 2`.
pub fn yuv422p_to_yuyv422(yp: &[u8], up: &[u8], vp: &[u8], dst: &mut [u8], w: usize, h: usize) {
    debug_assert!(w % 2 == 0, "YUYV requires even width");
    let cw = w / 2;
    debug_assert!(yp.len() >= w * h);
    debug_assert!(up.len() >= cw * h);
    debug_assert!(vp.len() >= cw * h);
    debug_assert!(dst.len() >= w * h * 2);
    for row in 0..h {
        let y_row = &yp[row * w..row * w + w];
        let u_row = &up[row * cw..row * cw + cw];
        let v_row = &vp[row * cw..row * cw + cw];
        let d = &mut dst[row * w * 2..row * w * 2 + w * 2];
        for cc in 0..cw {
            d[cc * 4] = y_row[cc * 2];
            d[cc * 4 + 1] = u_row[cc];
            d[cc * 4 + 2] = y_row[cc * 2 + 1];
            d[cc * 4 + 3] = v_row[cc];
        }
    }
}

/// Merge planar Yuv422P into a packed UYVY byte stream. Mirror of
/// [`yuv422p_to_yuyv422`] with luma and chroma byte positions swapped.
pub fn yuv422p_to_uyvy422(yp: &[u8], up: &[u8], vp: &[u8], dst: &mut [u8], w: usize, h: usize) {
    debug_assert!(w % 2 == 0, "UYVY requires even width");
    let cw = w / 2;
    debug_assert!(yp.len() >= w * h);
    debug_assert!(up.len() >= cw * h);
    debug_assert!(vp.len() >= cw * h);
    debug_assert!(dst.len() >= w * h * 2);
    for row in 0..h {
        let y_row = &yp[row * w..row * w + w];
        let u_row = &up[row * cw..row * cw + cw];
        let v_row = &vp[row * cw..row * cw + cw];
        let d = &mut dst[row * w * 2..row * w * 2 + w * 2];
        for cc in 0..cw {
            d[cc * 4] = u_row[cc];
            d[cc * 4 + 1] = y_row[cc * 2];
            d[cc * 4 + 2] = v_row[cc];
            d[cc * 4 + 3] = y_row[cc * 2 + 1];
        }
    }
}

/// In-place byte swap converting YUYV (Y0 U0 Y1 V0) to UYVY
/// (U0 Y0 V0 Y1) by exchanging the two bytes inside each pair. Works
/// in either direction (involutive on the four-byte quad).
pub fn yuyv_uyvy_swap(buf: &mut [u8]) {
    // YUYV quad = [Y0 U Y1 V]; UYVY quad = [U Y0 V Y1]:
    //   index 0 ↔ index 1  (Y0 ↔ U)
    //   index 2 ↔ index 3  (Y1 ↔ V)
    debug_assert!(
        buf.len() % 4 == 0,
        "packed 4:2:2 buffer must be a multiple of 4 bytes"
    );
    let mut i = 0;
    while i + 3 < buf.len() {
        buf.swap(i, i + 1);
        buf.swap(i + 2, i + 3);
        i += 4;
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
    fn yuyv_uyvy_swap_pins_byte_positions() {
        // YUYV quad = [Y0 U Y1 V]; swap gives UYVY = [U Y0 V Y1].
        let mut buf = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        yuyv_uyvy_swap(&mut buf);
        // Two quads:
        //   [10 20 30 40] → [20 10 40 30]
        //   [50 60 70 80] → [60 50 80 70]
        assert_eq!(buf, vec![20u8, 10, 40, 30, 60, 50, 80, 70]);
        // Involutive — second swap returns the original bytes.
        yuyv_uyvy_swap(&mut buf);
        assert_eq!(buf, vec![10u8, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn yuyv422_to_yuv422p_pins_byte_positions() {
        // 4×1 YUYV: two quads.
        // pixels (Y0=1,U=2,Y1=3,V=4) (Y0=5,U=6,Y1=7,V=8)
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut yp = [0u8; 4];
        let mut up = [0u8; 2];
        let mut vp = [0u8; 2];
        yuyv422_to_yuv422p(&src, &mut yp, &mut up, &mut vp, 4, 1);
        assert_eq!(yp, [1, 3, 5, 7]);
        assert_eq!(up, [2, 6]);
        assert_eq!(vp, [4, 8]);
        // Round-trip back to packed.
        let mut out = [0u8; 8];
        yuv422p_to_yuyv422(&yp, &up, &vp, &mut out, 4, 1);
        assert_eq!(out, src);
    }

    #[test]
    fn uyvy422_to_yuv422p_pins_byte_positions() {
        // 4×1 UYVY: two quads.
        // pixels (U=2,Y0=1,V=4,Y1=3) (U=6,Y0=5,V=8,Y1=7)
        let src = [2u8, 1, 4, 3, 6, 5, 8, 7];
        let mut yp = [0u8; 4];
        let mut up = [0u8; 2];
        let mut vp = [0u8; 2];
        uyvy422_to_yuv422p(&src, &mut yp, &mut up, &mut vp, 4, 1);
        assert_eq!(yp, [1, 3, 5, 7]);
        assert_eq!(up, [2, 6]);
        assert_eq!(vp, [4, 8]);
        // Round-trip back to packed.
        let mut out = [0u8; 8];
        yuv422p_to_uyvy422(&yp, &up, &vp, &mut out, 4, 1);
        assert_eq!(out, src);
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
