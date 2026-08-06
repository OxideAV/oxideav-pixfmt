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

    // Each SIMD iteration writes a full 16-byte register but only the
    // first 6 bytes are meaningful, then advances the dst cursor by 6
    // on the next iteration. The *final* SIMD iteration's 16-byte
    // store must fit entirely within `dst[..pixels*3]` — otherwise it
    // walks past the end of the destination slice and corrupts what
    // follows (the next row of the parent buffer, or for the final
    // row, the heap allocator headers right after the destination
    // Vec).
    //
    // Required: chunks*6 + 16 ≤ pixels*3, so chunks ≤ (pixels*3 - 16)/6.
    // Equivalently, the SIMD loop must leave at least
    // ceil(16/3) = 6 dst-pixels' worth of room (18 bytes) for the tail
    // to fall inside the slice. Reserve a 6-pixel tail.
    //
    // (The previous "2-pixel tail" reserve only protected the read
    // side; the write side over-ran the slice by 4 bytes on every
    // call. On x86_64 Linux CI this manifested as a glibc malloc
    // sysmalloc assertion when the corrupted bytes happened to be
    // allocator metadata.)
    let simd_pixels = pixels.saturating_sub(6);
    let chunks = simd_pixels / 2;
    for c in 0..chunks {
        let soff = c * 12;
        let doff = c * 6;
        let v = _mm_loadu_si128(src.as_ptr().add(soff) as *const __m128i);
        let out = _mm_shuffle_epi8(v, shuf);
        // Store 16 bytes; the next iteration's 6-byte advance and
        // scalar tail together overwrite the 10 dead bytes.
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

/// Rgb48Le → Rgba64Le: copy the three colour words verbatim and append
/// an opaque 65535 alpha word — bit-exact on colour, and
/// [`rgba64_to_rgb48`] is its exact inverse.
pub fn rgb48_to_rgba64(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        dst[i * 8..i * 8 + 6].copy_from_slice(&src[i * 6..i * 6 + 6]);
        dst[i * 8 + 6] = 0xFF;
        dst[i * 8 + 7] = 0xFF;
    }
}

/// Rgba64Le → Rgb48Le: copy the three colour words verbatim, dropping
/// the alpha word.
pub fn rgba64_to_rgb48(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        dst[i * 6..i * 6 + 6].copy_from_slice(&src[i * 8..i * 8 + 6]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `rgb48_to_rgb24` used to write 4 bytes past the end of
    /// `dst` because the SIMD path emitted a full 16-byte store every two
    /// pixels and only reserved a 2-pixel scalar tail (6 bytes). On
    /// x86_64 Linux CI the stomp tripped a glibc malloc sysmalloc
    /// assertion at allocator-metadata boundaries. We pin the contract
    /// here with a tight-fit `dst` whose backing Vec ends exactly at
    /// `pixels*3` — any write past the slice corrupts memory the heap
    /// is about to reuse.
    #[test]
    fn rgb48_to_rgb24_does_not_write_past_dst() {
        // Try every pixel count from 1..=64 so we cover both the
        // pre-SIMD short cases and the SIMD-with-tail case.
        for pixels in 1..=64 {
            let mut src = vec![0u8; pixels * 6];
            for (i, b) in src.iter_mut().enumerate() {
                *b = (i & 0xff) as u8;
            }
            let mut dst = vec![0u8; pixels * 3];
            rgb48_to_rgb24(&src, &mut dst, pixels);
            // Verify every dst byte matches the scalar reference.
            for i in 0..pixels {
                assert_eq!(dst[i * 3], src[i * 6 + 1], "px {i}/{pixels} R");
                assert_eq!(dst[i * 3 + 1], src[i * 6 + 3], "px {i}/{pixels} G");
                assert_eq!(dst[i * 3 + 2], src[i * 6 + 5], "px {i}/{pixels} B");
            }
        }
    }

    /// Hammer the SIMD path by allocating + freeing the dst Vec a few
    /// thousand times — if any earlier call corrupted heap metadata,
    /// glibc's malloc will trip a sysmalloc assertion on a subsequent
    /// free.
    #[test]
    fn rgb48_to_rgb24_alloc_churn_does_not_corrupt_heap() {
        for _ in 0..2000 {
            let pixels = 32;
            let src = vec![0xa5u8; pixels * 6];
            let mut dst = vec![0u8; pixels * 3];
            rgb48_to_rgb24(&src, &mut dst, pixels);
            drop(dst);
        }
    }
}
