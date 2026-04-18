//! RGB / BGR family swizzles, plus bit-depth changes between packed
//! 8-bit and 16-bit representations.
//!
//! All functions in this module assume tightly packed input/output
//! (no stride padding). The caller is responsible for stripping stride
//! before handing buffers in and re-adding it afterwards.
//!
//! `swizzle4` / `swizzle3` compute a per-channel permutation from the
//! runtime `src_pos` / `dst_pos` indices and then dispatch to a
//! vectorised path (`swizzle_simd::*`) that rides on a single AVX2
//! `pshufb`. The scalar fallback lives right here.

mod swizzle_simd;

/// Component index into a 4-byte packed pixel. Used to describe where
/// R, G, B, and A live for each of the 4-channel formats.
#[derive(Clone, Copy)]
pub struct Rgba4 {
    pub r: usize,
    pub g: usize,
    pub b: usize,
    pub a: usize,
}

/// Byte positions for each 4-channel packed format.
pub const RGBA_POS: Rgba4 = Rgba4 {
    r: 0,
    g: 1,
    b: 2,
    a: 3,
};
pub const BGRA_POS: Rgba4 = Rgba4 {
    r: 2,
    g: 1,
    b: 0,
    a: 3,
};
pub const ARGB_POS: Rgba4 = Rgba4 {
    r: 1,
    g: 2,
    b: 3,
    a: 0,
};
pub const ABGR_POS: Rgba4 = Rgba4 {
    r: 3,
    g: 2,
    b: 1,
    a: 0,
};

/// Component index into a 3-byte packed pixel.
#[derive(Clone, Copy)]
pub struct Rgb3 {
    pub r: usize,
    pub g: usize,
    pub b: usize,
}

pub const RGB_POS: Rgb3 = Rgb3 { r: 0, g: 1, b: 2 };
pub const BGR_POS: Rgb3 = Rgb3 { r: 2, g: 1, b: 0 };

/// Swizzle a packed 3-byte pixel stream between RGB and BGR (or any
/// two Rgb3 layouts).
///
/// Uses a pre-computed 3-byte permutation so the compiler can lift the
/// position indirection out of the hot loop and auto-vectorise the
/// byte-shuffle.
pub fn swizzle3(src: &[u8], src_pos: Rgb3, dst: &mut [u8], dst_pos: Rgb3, pixels: usize) {
    debug_assert!(src.len() >= pixels * 3 && dst.len() >= pixels * 3);
    // perm[j] = source byte offset (within the 3-byte group) that goes
    // to destination byte j.
    let mut perm = [0u8; 3];
    perm[dst_pos.r] = src_pos.r as u8;
    perm[dst_pos.g] = src_pos.g as u8;
    perm[dst_pos.b] = src_pos.b as u8;
    swizzle_simd::swizzle3_perm(src, dst, pixels, perm);
}

/// Swizzle a packed 4-byte pixel stream between any two Rgba4 layouts.
///
/// Routes through an AVX2 `pshufb`-based path when the CPU supports it;
/// otherwise falls back to a specialised-permutation scalar loop.
pub fn swizzle4(src: &[u8], src_pos: Rgba4, dst: &mut [u8], dst_pos: Rgba4, pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 4);
    // perm[j] = source byte offset (within the 4-byte group) that goes
    // to destination byte j.
    let mut perm = [0u8; 4];
    perm[dst_pos.r] = src_pos.r as u8;
    perm[dst_pos.g] = src_pos.g as u8;
    perm[dst_pos.b] = src_pos.b as u8;
    perm[dst_pos.a] = src_pos.a as u8;
    swizzle_simd::swizzle4_perm(src, dst, pixels, perm);
}

/// Convert a 3-byte packed source to a 4-byte packed destination,
/// synthesising an opaque alpha (255).
pub fn rgb3_to_rgba4(src: &[u8], src_pos: Rgb3, dst: &mut [u8], dst_pos: Rgba4, pixels: usize) {
    // perm3[i] (0..3) = source byte within the 3-byte group for dst RGB
    // byte i; perm3[dst_pos.a] is set to 0xFF to mark "emit 255".
    let mut perm3 = [0xFFu8; 4];
    perm3[dst_pos.r] = src_pos.r as u8;
    perm3[dst_pos.g] = src_pos.g as u8;
    perm3[dst_pos.b] = src_pos.b as u8;
    swizzle_simd::rgb3_to_rgba4_perm(src, dst, pixels, perm3);
}

/// Drop the alpha channel, converting a 4-byte packed source to a
/// 3-byte packed destination.
pub fn rgba4_to_rgb3(src: &[u8], src_pos: Rgba4, dst: &mut [u8], dst_pos: Rgb3, pixels: usize) {
    // perm4[i] (0..3) = source byte within the 4-byte group for dst byte i.
    let mut perm4 = [0u8; 3];
    perm4[dst_pos.r] = src_pos.r as u8;
    perm4[dst_pos.g] = src_pos.g as u8;
    perm4[dst_pos.b] = src_pos.b as u8;
    swizzle_simd::rgba4_to_rgb3_perm(src, dst, pixels, perm4);
}

/// Rgb48Le → Rgb24 (drop low 8 bits, keep the high byte of each LE word).
pub fn rgb48_to_rgb24(src: &[u8], dst: &mut [u8], pixels: usize) {
    if crate::simd_dispatch::has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            rgb48_to_rgb24_avx2(src, dst, pixels);
            return;
        }
    }
    for i in 0..pixels {
        dst[i * 3] = src[i * 6 + 1];
        dst[i * 3 + 1] = src[i * 6 + 3];
        dst[i * 3 + 2] = src[i * 6 + 5];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb48_to_rgb24_avx2(src: &[u8], dst: &mut [u8], pixels: usize) {
    use core::arch::x86_64::*;
    // 2 pixels per __m128i iteration: 12 source bytes → 6 dst bytes.
    // That's too small — bump up to 4 pixels = 24 src bytes = one full
    // __m128i input, producing 12 dst bytes per iteration.
    //
    // For each pixel `p` in 0..4, we want dst[p*3 + c] = src[p*6 + c*2 + 1]
    // for c in 0..3. Source-byte offsets within the 16-byte input:
    //   pixel 0: 1, 3, 5
    //   pixel 1: 7, 9, 11
    //   pixel 2: 13, 15, (17 - out of lane)
    //   pixel 3: (19, 21, 23 - out of lane)
    // So 4 pixels won't fit in one 16-byte load. Use 2 pixels per
    // iteration = 12 src bytes → 6 dst bytes instead — still avoids
    // scalar-per-byte.
    const SHUF: [u8; 16] = [
        1, 3, 5, 7, 9, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ];
    let shuf = _mm_loadu_si128(SHUF.as_ptr() as *const __m128i);

    // Leave a 2-pixel tail so our 16-byte load never reads past end.
    let simd_pixels = pixels.saturating_sub(2);
    let chunks = simd_pixels / 2;
    for c in 0..chunks {
        let soff = c * 12;
        let doff = c * 6;
        let v = _mm_loadu_si128(src.as_ptr().add(soff) as *const __m128i);
        let out = _mm_shuffle_epi8(v, shuf);
        // Store 16 bytes; next iteration (or scalar tail) overwrites
        // the 10 zero bytes.
        _mm_storeu_si128(dst.as_mut_ptr().add(doff) as *mut __m128i, out);
    }
    let tail_start = chunks * 2;
    for i in tail_start..pixels {
        dst[i * 3] = src[i * 6 + 1];
        dst[i * 3 + 1] = src[i * 6 + 3];
        dst[i * 3 + 2] = src[i * 6 + 5];
    }
}

/// Rgb24 → Rgb48Le (left-shift 8 and replicate high byte into the low
/// byte for a proper scaling instead of losing bottom range).
pub fn rgb24_to_rgb48(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        for c in 0..3 {
            let b = src[i * 3 + c];
            // Replicate: value * 257 / 256 style — use (b << 8) | b.
            let v: u16 = (b as u16) << 8 | (b as u16);
            let off = i * 6 + c * 2;
            dst[off] = (v & 0xFF) as u8;
            dst[off + 1] = (v >> 8) as u8;
        }
    }
}

/// Rgba64Le → Rgba.
pub fn rgba64_to_rgba(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        // 8 bytes in, 4 bytes out; LE high byte = index 1,3,5,7.
        dst[i * 4] = src[i * 8 + 1];
        dst[i * 4 + 1] = src[i * 8 + 3];
        dst[i * 4 + 2] = src[i * 8 + 5];
        dst[i * 4 + 3] = src[i * 8 + 7];
    }
}

/// Rgba → Rgba64Le.
pub fn rgba_to_rgba64(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        for c in 0..4 {
            let b = src[i * 4 + c];
            let v: u16 = (b as u16) << 8 | (b as u16);
            let off = i * 8 + c * 2;
            dst[off] = (v & 0xFF) as u8;
            dst[off + 1] = (v >> 8) as u8;
        }
    }
}
