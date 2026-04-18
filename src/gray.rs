//! Grayscale / mono conversions.
//!
//! The luma value is broadcast directly to RGB components when expanding
//! to a colour format — we do *not* apply any colour-space transfer
//! function here. For accurate luma from a colour source, call
//! [`crate::yuv::rgb_to_yuv`] and keep only the Y plane.

/// Gray8 → Rgb24 (broadcast the grey value to R, G, and B).
pub fn gray8_to_rgb24(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        let v = src[i];
        dst[i * 3] = v;
        dst[i * 3 + 1] = v;
        dst[i * 3 + 2] = v;
    }
}

/// Gray8 → Rgba (broadcast grey; alpha = 255).
pub fn gray8_to_rgba(src: &[u8], dst: &mut [u8], pixels: usize) {
    if crate::simd_dispatch::has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            gray8_to_rgba_avx2(src, dst, pixels);
            return;
        }
    }
    for i in 0..pixels {
        let v = src[i];
        dst[i * 4] = v;
        dst[i * 4 + 1] = v;
        dst[i * 4 + 2] = v;
        dst[i * 4 + 3] = 255;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gray8_to_rgba_avx2(src: &[u8], dst: &mut [u8], pixels: usize) {
    use core::arch::x86_64::*;
    // 4 pixels per __m128i iteration: load 4 bytes, pshufb to broadcast
    // each into a 4-byte slot, OR with a 0xFF alpha mask. Actually 16
    // pixels per __m256i is just as easy — process that.
    // 16 src bytes → 64 dst bytes per iteration. Use per-lane pshufb
    // where each lane handles 8 source bytes → 32 output bytes. That
    // doesn't fit; stick with 4-pixel __m128i → 16-byte store.
    //
    // Mask: byte j of the 16-byte output corresponds to pixel `j/4`;
    // the RGB lanes get source byte `j/4`, the alpha lane gets 0x80
    // (pshufb emits zero) and we OR in 0xFF.
    const SHUF: [u8; 16] = [0, 0, 0, 0x80, 1, 1, 1, 0x80, 2, 2, 2, 0x80, 3, 3, 3, 0x80];
    let shuf = _mm_loadu_si128(SHUF.as_ptr() as *const __m128i);
    let alpha = _mm_set1_epi32(0xFF00_0000u32 as i32);

    let chunks = pixels / 4;
    for c in 0..chunks {
        let s = _mm_cvtsi32_si128(core::ptr::read_unaligned(src.as_ptr().add(c * 4) as *const i32));
        let broadcast = _mm_shuffle_epi8(s, shuf);
        let with_alpha = _mm_or_si128(broadcast, alpha);
        _mm_storeu_si128(dst.as_mut_ptr().add(c * 16) as *mut __m128i, with_alpha);
    }
    let tail = chunks * 4;
    for i in tail..pixels {
        let v = src[i];
        dst[i * 4] = v;
        dst[i * 4 + 1] = v;
        dst[i * 4 + 2] = v;
        dst[i * 4 + 3] = 255;
    }
}

/// Gray16Le → Gray8 (keep the high byte of each LE u16 — simple
/// truncation; matches what a naïve >> 8 would produce).
pub fn gray16le_to_gray8(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        dst[i] = src[i * 2 + 1];
    }
}

/// Gray8 → Gray16Le (replicate byte into high and low halves so a
/// subsequent gray16 → gray8 round-trips to the original value).
pub fn gray8_to_gray16le(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        let b = src[i];
        dst[i * 2] = b;
        dst[i * 2 + 1] = b;
    }
}

/// 1 bit per pixel (MSB-first) → Gray8. `black_is_zero = true` means
/// MonoBlack (0 bit = 0, 1 bit = 255). `false` means MonoWhite (0 bit
/// = 255, 1 bit = 0). The row stride on the source side is the packed
/// byte width (w + 7) / 8.
pub fn mono_to_gray8(src: &[u8], dst: &mut [u8], w: usize, h: usize, black_is_zero: bool) {
    let stride = w.div_ceil(8);
    for row in 0..h {
        for col in 0..w {
            let byte = src[row * stride + col / 8];
            let bit = (byte >> (7 - (col & 7))) & 1;
            let g = if bit == 1 { 255u8 } else { 0u8 };
            dst[row * w + col] = if black_is_zero { g } else { 255 - g };
        }
    }
}

/// Gray8 → 1 bpp (MSB-first). A threshold of 128 decides bit value.
pub fn gray8_to_mono(src: &[u8], dst: &mut [u8], w: usize, h: usize, black_is_zero: bool) {
    let stride = w.div_ceil(8);
    for b in dst.iter_mut() {
        *b = 0;
    }
    for row in 0..h {
        for col in 0..w {
            let g = src[row * w + col];
            let bit_on = if black_is_zero { g >= 128 } else { g < 128 };
            if bit_on {
                let shift = 7 - (col & 7);
                dst[row * stride + col / 8] |= 1u8 << shift;
            }
        }
    }
}
