//! AVX2 implementations of the YUV ↔ RGB inner loops.
//!
//! Every routine here requires `avx2`; the caller (in `mod.rs`) only
//! dispatches after `is_x86_feature_detected!("avx2")`. The math is kept
//! bit-exact against the scalar fixed-point path where possible — i16
//! multiplications may differ in edge rounding, so the property test in
//! `tests/yuv_simd.rs` pins the per-pixel tolerance at ±1 LSB.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_range_loop)]

use crate::yuv::{
    rgb_to_yuv_fp, yuv_to_rgb_fp, DecodeParams, EncodeParams, YuvMatrix, FP_HALF, FP_SHIFT,
};
use core::arch::x86_64::*;

// Process 16 Y samples per iteration (one __m128i of luma → two
// __m256i of 16-bit intermediates). We widen to i32 lanes for the
// multiply-add so we stay within i16 range on overflow-prone mixes.
const LANES: usize = 16;

// ---------------------------------------------------------------------
// Helpers shared by the decoders.

/// Compute R/G/B for 16 pixels given matching 16-wide Y/U/V i32 vectors
/// (already offset: y = y_byte - y_off; cb = cbyte - 128; cr = crbyte - 128).
/// The rounding bias FP_HALF is already folded in via `y_lin_*`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_block_i32x16(
    y_lin_lo: __m256i,
    y_lin_hi: __m256i,
    cb_lo: __m256i,
    cb_hi: __m256i,
    cr_lo: __m256i,
    cr_hi: __m256i,
    d: &DecodeParams,
) -> (__m128i, __m128i, __m128i) {
    let cr_r = _mm256_set1_epi32(d.cr_r);
    let cb_b = _mm256_set1_epi32(d.cb_b);
    let cg_cr = _mm256_set1_epi32(d.cg_cr);
    let cg_cb = _mm256_set1_epi32(d.cg_cb);
    let bias = _mm256_set1_epi32(FP_HALF);

    // R = (y_lin + cr_r*cr + bias) >> SHIFT
    let r_lo = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(y_lin_lo, _mm256_mullo_epi32(cr_r, cr_lo)),
            bias,
        ),
        FP_SHIFT,
    );
    let r_hi = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(y_lin_hi, _mm256_mullo_epi32(cr_r, cr_hi)),
            bias,
        ),
        FP_SHIFT,
    );
    let b_lo = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(y_lin_lo, _mm256_mullo_epi32(cb_b, cb_lo)),
            bias,
        ),
        FP_SHIFT,
    );
    let b_hi = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(y_lin_hi, _mm256_mullo_epi32(cb_b, cb_hi)),
            bias,
        ),
        FP_SHIFT,
    );
    let g_lo = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_sub_epi32(y_lin_lo, _mm256_mullo_epi32(cg_cr, cr_lo)),
                _mm256_mullo_epi32(cg_cb, cb_lo),
            ),
            bias,
        ),
        FP_SHIFT,
    );
    let g_hi = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_sub_epi32(y_lin_hi, _mm256_mullo_epi32(cg_cr, cr_hi)),
                _mm256_mullo_epi32(cg_cb, cb_hi),
            ),
            bias,
        ),
        FP_SHIFT,
    );

    // Pack i32 → i16 (saturate in-lane, then cross-lane shuffle to get
    // a linear 16-element vector) → i16 → u8.
    let pack16 = |lo: __m256i, hi: __m256i| -> __m128i {
        let p = _mm256_packs_epi32(lo, hi);
        // packs_epi32 interleaves the two 128-bit lanes: [lo.lo, hi.lo, lo.hi, hi.hi]
        // Permute 64-bit quadwords so we get [lo.lo, lo.hi, hi.lo, hi.hi].
        let permuted = _mm256_permute4x64_epi64(p, 0b1101_1000);
        // Saturate-cast to u8 within the 256-bit register, then extract low.
        let packed_u8 = _mm256_packus_epi16(permuted, permuted);
        // packus interleaves again, so permute and grab the low 128.
        let perm = _mm256_permute4x64_epi64(packed_u8, 0b1101_1000);
        _mm256_castsi256_si128(perm)
    };
    let r = pack16(r_lo, r_hi);
    let g = pack16(g_lo, g_hi);
    let b = pack16(b_lo, b_hi);
    (r, g, b)
}

/// Scatter 16 RGB triples from three __m128i lanes (R, G, B) into 48
/// contiguous destination bytes. Uses a scalar spill — the AVX2 pshufb
/// tricks buy us a couple of percent at best on a per-block basis and
/// complicate correctness for a three-byte-stride write.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_rgb24_lane16(dst: &mut [u8], r: __m128i, g: __m128i, b: __m128i) {
    let mut rs = [0u8; 16];
    let mut gs = [0u8; 16];
    let mut bs = [0u8; 16];
    _mm_storeu_si128(rs.as_mut_ptr() as *mut __m128i, r);
    _mm_storeu_si128(gs.as_mut_ptr() as *mut __m128i, g);
    _mm_storeu_si128(bs.as_mut_ptr() as *mut __m128i, b);
    for i in 0..LANES {
        dst[i * 3] = rs[i];
        dst[i * 3 + 1] = gs[i];
        dst[i * 3 + 2] = bs[i];
    }
}

/// Load 16 consecutive bytes as a u8x16 vector, zero-extend to two
/// __m256i of 8 × i32, subtracting `off` from every lane (useful for
/// applying the y_off / 128 chroma centre-offset in one go).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load16_sub_i32(src: &[u8], off: i32) -> (__m256i, __m256i) {
    let v = _mm_loadu_si128(src.as_ptr() as *const __m128i);
    let lo8 = _mm256_cvtepu8_epi32(v);
    let hi8 = _mm256_cvtepu8_epi32(_mm_srli_si128(v, 8));
    let off_v = _mm256_set1_epi32(off);
    (
        _mm256_sub_epi32(lo8, off_v),
        _mm256_sub_epi32(hi8, off_v),
    )
}

// Expand 8 chroma bytes to a 16-wide vector where each sample is
// duplicated (cc[0]=cc[1], cc[2]=cc[3], ...), then subtract 128 and
// widen to two __m256i of i32. For 4:2:0 and 4:2:2 chroma.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_chroma_8_broadcast(src: &[u8]) -> (__m256i, __m256i) {
    // src has at least 8 bytes. Duplicate each byte to produce a 16-byte
    // pattern.
    let v = _mm_loadl_epi64(src.as_ptr() as *const __m128i);
    // pshuflb to duplicate each byte.
    let dup_mask = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup = _mm_shuffle_epi8(v, dup_mask);
    let lo = _mm256_cvtepu8_epi32(dup);
    let hi = _mm256_cvtepu8_epi32(_mm_srli_si128(dup, 8));
    let c128 = _mm256_set1_epi32(128);
    (_mm256_sub_epi32(lo, c128), _mm256_sub_epi32(hi, c128))
}

// ---------------------------------------------------------------------
// 4:4:4 → RGB24.

#[target_feature(enable = "avx2")]
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
    let y_off_v = d.y_off;
    let y_scale_v = _mm256_set1_epi32(d.y_scale);
    for row in 0..h {
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * w..row * w + w];
        let vrow = &vp[row * w..row * w + w];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let (y_lo, y_hi) = load16_sub_i32(&yrow[off..], y_off_v);
            let y_lin_lo = _mm256_mullo_epi32(y_lo, y_scale_v);
            let y_lin_hi = _mm256_mullo_epi32(y_hi, y_scale_v);
            let (cb_lo, cb_hi) = load16_sub_i32(&urow[off..], 128);
            let (cr_lo, cr_hi) = load16_sub_i32(&vrow[off..], 128);
            let (r, g, b) = decode_block_i32x16(
                y_lin_lo, y_lin_hi, cb_lo, cb_hi, cr_lo, cr_hi, &d,
            );
            store_rgb24_lane16(&mut drow[off * 3..off * 3 + LANES * 3], r, g, b);
        }
        // Tail in scalar.
        for col in (chunks * LANES)..w {
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[col], vrow[col], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

// ---------------------------------------------------------------------
// 4:2:2 → RGB24.

#[target_feature(enable = "avx2")]
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
    let y_off_v = d.y_off;
    let y_scale_v = _mm256_set1_epi32(d.y_scale);
    for row in 0..h {
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * cw..row * cw + cw];
        let vrow = &vp[row * cw..row * cw + cw];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let coff = off / 2;
            let (y_lo, y_hi) = load16_sub_i32(&yrow[off..], y_off_v);
            let y_lin_lo = _mm256_mullo_epi32(y_lo, y_scale_v);
            let y_lin_hi = _mm256_mullo_epi32(y_hi, y_scale_v);
            let (cb_lo, cb_hi) = load_chroma_8_broadcast(&urow[coff..]);
            let (cr_lo, cr_hi) = load_chroma_8_broadcast(&vrow[coff..]);
            let (r, g, b) = decode_block_i32x16(
                y_lin_lo, y_lin_hi, cb_lo, cb_hi, cr_lo, cr_hi, &d,
            );
            store_rgb24_lane16(&mut drow[off * 3..off * 3 + LANES * 3], r, g, b);
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
// 4:2:0 → RGB24.

#[target_feature(enable = "avx2")]
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
    let y_off_v = d.y_off;
    let y_scale_v = _mm256_set1_epi32(d.y_scale);
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
            let (y_lo, y_hi) = load16_sub_i32(&yrow[off..], y_off_v);
            let y_lin_lo = _mm256_mullo_epi32(y_lo, y_scale_v);
            let y_lin_hi = _mm256_mullo_epi32(y_hi, y_scale_v);
            let (cb_lo, cb_hi) = load_chroma_8_broadcast(&urow[coff..]);
            let (cr_lo2, cr_hi2) = load_chroma_8_broadcast(&vrow[coff..]);
            let (r, g, b) = decode_block_i32x16(
                y_lin_lo, y_lin_hi, cb_lo, cb_hi, cr_lo2, cr_hi2, &d,
            );
            store_rgb24_lane16(&mut drow[off * 3..off * 3 + LANES * 3], r, g, b);
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
// Encode: RGB24 → YUV planar.
//
// Loading RGB24's 3-byte-packed stream into lanes cleanly is painful;
// the straightforward route is a byte-level gather via `pshufb`. We use
// 32-byte loads and a pair of shuffle masks that extract 8 R/G/B i32
// values from the low half. Two such blocks per iteration = 16 pixels.

// Encode strategy: load 8 pixels via scalar → 8×i32 R/G/B, then one
// __m256i multiply-accumulate each for Y/Cb/Cr. Process 8 pixels per
// iteration. Still avoids the f32 division and clamp branch.

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn encode_block_i32x8(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    p: &EncodeParams,
) -> (__m128i, __m128i, __m128i) {
    let y_vec = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_set1_epi32(p.cy_r), r),
                    _mm256_mullo_epi32(_mm256_set1_epi32(p.cy_g), g),
                ),
                _mm256_mullo_epi32(_mm256_set1_epi32(p.cy_b), b),
            ),
            _mm256_set1_epi32(p.y_bias),
        ),
        FP_SHIFT,
    );
    let cb_vec = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_set1_epi32(p.cb_r), r),
                    _mm256_mullo_epi32(_mm256_set1_epi32(p.cb_g), g),
                ),
                _mm256_mullo_epi32(_mm256_set1_epi32(p.cb_b), b),
            ),
            _mm256_set1_epi32(p.c_bias),
        ),
        FP_SHIFT,
    );
    let cr_vec = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_set1_epi32(p.cr_r), r),
                    _mm256_mullo_epi32(_mm256_set1_epi32(p.cr_g), g),
                ),
                _mm256_mullo_epi32(_mm256_set1_epi32(p.cr_b), b),
            ),
            _mm256_set1_epi32(p.c_bias),
        ),
        FP_SHIFT,
    );
    // Pack i32 (8-wide) → i16 → u8 (8-wide).
    let pack8 = |v: __m256i| -> __m128i {
        let p16 = _mm256_packs_epi32(v, v);
        let perm = _mm256_permute4x64_epi64(p16, 0b1101_1000);
        let p8 = _mm256_packus_epi16(perm, perm);
        _mm256_castsi256_si128(p8)
    };
    (pack8(y_vec), pack8(cb_vec), pack8(cr_vec))
}

/// Load 8 consecutive RGB triples → 3 × __m256i of i32.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_rgb24_8(src: &[u8]) -> (__m256i, __m256i, __m256i) {
    let mut rs = [0i32; 8];
    let mut gs = [0i32; 8];
    let mut bs = [0i32; 8];
    for i in 0..8 {
        rs[i] = src[i * 3] as i32;
        gs[i] = src[i * 3 + 1] as i32;
        bs[i] = src[i * 3 + 2] as i32;
    }
    let r = _mm256_loadu_si256(rs.as_ptr() as *const __m256i);
    let g = _mm256_loadu_si256(gs.as_ptr() as *const __m256i);
    let b = _mm256_loadu_si256(bs.as_ptr() as *const __m256i);
    (r, g, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_u8_lane8(dst: &mut [u8], v: __m128i) {
    // Only the low 8 bytes are meaningful.
    let mut tmp = [0u8; 16];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v);
    dst[..8].copy_from_slice(&tmp[..8]);
}

#[target_feature(enable = "avx2")]
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
    for row in 0..h {
        let chunks = w / 8;
        let srow = &src[row * w * 3..row * w * 3 + w * 3];
        let yrow = &mut yp[row * w..row * w + w];
        let urow = &mut up[row * w..row * w + w];
        let vrow = &mut vp[row * w..row * w + w];
        for chunk in 0..chunks {
            let off = chunk * 8;
            let (r, g, b) = load_rgb24_8(&srow[off * 3..off * 3 + 24]);
            let (y, u, v) = encode_block_i32x8(r, g, b, &p);
            store_u8_lane8(&mut yrow[off..off + 8], y);
            store_u8_lane8(&mut urow[off..off + 8], u);
            store_u8_lane8(&mut vrow[off..off + 8], v);
        }
        for col in (chunks * 8)..w {
            let o = col * 3;
            let (y, u, v) = rgb_to_yuv_fp(srow[o], srow[o + 1], srow[o + 2], &p);
            yrow[col] = y;
            urow[col] = u;
            vrow[col] = v;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb24_to_yuv422(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    // Luma pass (vectorised); chroma pass keeps the scalar averaging
    // because the horizontal-add is a few adds per pair and would need
    // a cross-lane shuffle to justify any vector work.
    let p = matrix.encode_params();
    let cw = w / 2;
    for row in 0..h {
        let chunks = w / 8;
        let srow = &src[row * w * 3..row * w * 3 + w * 3];
        let yrow = &mut yp[row * w..row * w + w];
        for chunk in 0..chunks {
            let off = chunk * 8;
            let (r, g, b) = load_rgb24_8(&srow[off * 3..off * 3 + 24]);
            let (y, _u, _v) = encode_block_i32x8(r, g, b, &p);
            store_u8_lane8(&mut yrow[off..off + 8], y);
        }
        for col in (chunks * 8)..w {
            let o = col * 3;
            let (y, _u, _v) = rgb_to_yuv_fp(srow[o], srow[o + 1], srow[o + 2], &p);
            yrow[col] = y;
        }
        // Chroma averages (2 cols → 1).
        for cc in 0..cw {
            let mut cbs = 0i32;
            let mut crs = 0i32;
            for dx in 0..2 {
                let col = cc * 2 + dx;
                let o = col * 3;
                let (_y, u, v) = rgb_to_yuv_fp(srow[o], srow[o + 1], srow[o + 2], &p);
                cbs += u as i32;
                crs += v as i32;
            }
            up[row * cw + cc] = ((cbs + 1) / 2) as u8;
            vp[row * cw + cc] = ((crs + 1) / 2) as u8;
        }
    }
}

#[target_feature(enable = "avx2")]
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
    let cw = w / 2;
    let ch = h / 2;
    // Luma pass with AVX2.
    for row in 0..h {
        let chunks = w / 8;
        let srow = &src[row * w * 3..row * w * 3 + w * 3];
        let yrow = &mut yp[row * w..row * w + w];
        for chunk in 0..chunks {
            let off = chunk * 8;
            let (r, g, b) = load_rgb24_8(&srow[off * 3..off * 3 + 24]);
            let (y, _u, _v) = encode_block_i32x8(r, g, b, &p);
            store_u8_lane8(&mut yrow[off..off + 8], y);
        }
        for col in (chunks * 8)..w {
            let o = col * 3;
            let (y, _u, _v) = rgb_to_yuv_fp(srow[o], srow[o + 1], srow[o + 2], &p);
            yrow[col] = y;
        }
    }
    // Chroma pass: scalar (2×2 averaging is bandwidth-bound).
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

