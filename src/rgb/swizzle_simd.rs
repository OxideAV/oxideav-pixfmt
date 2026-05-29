//! Per-permutation swizzle implementations.
//!
//! The public [`crate::rgb::swizzle3`] / [`crate::rgb::swizzle4`] entry
//! points compute a small permutation table from their runtime
//! `src_pos` / `dst_pos` arguments, then delegate here. We keep the
//! vectorised paths behind cached runtime feature detection — the
//! fastest x86_64 path is a single `_mm256_shuffle_epi8` per block.

use core::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Path {
    Unknown = 0,
    Scalar = 1,
    Avx2 = 2,
}

static DISPATCH: AtomicU8 = AtomicU8::new(Path::Unknown as u8);

fn select_path() -> Path {
    if std::env::var_os("OXIDEAV_PIXFMT_FORCE_SCALAR").is_some() {
        return Path::Scalar;
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return Path::Avx2;
        }
    }
    Path::Scalar
}

fn path() -> Path {
    let v = DISPATCH.load(Ordering::Relaxed);
    if v != Path::Unknown as u8 {
        return if v == Path::Avx2 as u8 {
            Path::Avx2
        } else {
            Path::Scalar
        };
    }
    let p = select_path();
    DISPATCH.store(p as u8, Ordering::Relaxed);
    p
}

// -------------------------------------------------------------------------
// swizzle4 — 4-byte permutation per pixel.

pub(crate) fn swizzle4_perm(src: &[u8], dst: &mut [u8], pixels: usize, perm: [u8; 4]) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2_swizzle4(src, dst, pixels, perm) },
        _ => scalar_swizzle4(src, dst, pixels, perm),
    }
}

#[inline(always)]
fn scalar_swizzle4(src: &[u8], dst: &mut [u8], pixels: usize, perm: [u8; 4]) {
    // perm is bounded to [0, 3]; copy it into locals to drop bounds
    // checks on the per-pixel access.
    let p0 = (perm[0] & 3) as usize;
    let p1 = (perm[1] & 3) as usize;
    let p2 = (perm[2] & 3) as usize;
    let p3 = (perm[3] & 3) as usize;
    for i in 0..pixels {
        let s = i * 4;
        let d = i * 4;
        dst[d] = src[s + p0];
        dst[d + 1] = src[s + p1];
        dst[d + 2] = src[s + p2];
        dst[d + 3] = src[s + p3];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_swizzle4(src: &[u8], dst: &mut [u8], pixels: usize, perm: [u8; 4]) {
    use core::arch::x86_64::*;
    // Build a 32-byte shuffle mask: within each 16-byte lane, replicate
    // the 4-byte permutation across 4 pixels, offset by the pixel's base.
    let mut mask_bytes = [0u8; 32];
    for lane in 0..2 {
        for pix in 0..4 {
            let base = pix * 4;
            for j in 0..4 {
                mask_bytes[lane * 16 + base + j] = (base as u8) + perm[j];
            }
        }
    }
    let mask = _mm256_loadu_si256(mask_bytes.as_ptr() as *const __m256i);

    // 8 pixels per AVX2 iteration (32 bytes).
    let chunks = pixels / 8;
    let tail_start = chunks * 8;
    for c in 0..chunks {
        let off = c * 32;
        let v = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
        let out = _mm256_shuffle_epi8(v, mask);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, out);
    }
    if tail_start < pixels {
        scalar_swizzle4(
            &src[tail_start * 4..],
            &mut dst[tail_start * 4..],
            pixels - tail_start,
            perm,
        );
    }
}

// -------------------------------------------------------------------------
// swizzle3 — 3-byte permutation per pixel.
//
// A 3-byte stride doesn't align with 128/256-bit lane boundaries, so we
// can't express it as a single `pshufb`. We use a 15-byte (5-pixel)
// lane mask and pay a scalar tail. Five pixels per iteration keeps the
// mask constant and gets the byte-shuffle out of the inner loop.

pub(crate) fn swizzle3_perm(src: &[u8], dst: &mut [u8], pixels: usize, perm: [u8; 3]) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2_swizzle3(src, dst, pixels, perm) },
        _ => scalar_swizzle3(src, dst, pixels, perm),
    }
}

#[inline(always)]
fn scalar_swizzle3(src: &[u8], dst: &mut [u8], pixels: usize, perm: [u8; 3]) {
    let p0 = (perm[0] % 3) as usize;
    let p1 = (perm[1] % 3) as usize;
    let p2 = (perm[2] % 3) as usize;
    for i in 0..pixels {
        let s = i * 3;
        let d = i * 3;
        dst[d] = src[s + p0];
        dst[d + 1] = src[s + p1];
        dst[d + 2] = src[s + p2];
    }
}

// -------------------------------------------------------------------------
// rgb3 → rgba4 — promote (3→4 bytes, synthesise alpha).
//
// `perm3[i]` contains the source byte offset (within the 3-byte input
// group) for destination byte `i`, except at the alpha lane (`perm3[a] =
// 0xFF`), which is emitted as 255.

pub(crate) fn rgb3_to_rgba4_perm(src: &[u8], dst: &mut [u8], pixels: usize, perm3: [u8; 4]) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2_rgb3_to_rgba4(src, dst, pixels, perm3) },
        _ => scalar_rgb3_to_rgba4(src, dst, pixels, perm3),
    }
}

#[inline(always)]
fn scalar_rgb3_to_rgba4(src: &[u8], dst: &mut [u8], pixels: usize, perm3: [u8; 4]) {
    for i in 0..pixels {
        let s = i * 3;
        let d = i * 4;
        for j in 0..4 {
            let p = perm3[j];
            dst[d + j] = if p == 0xFF {
                255
            } else {
                src[s + (p & 3) as usize]
            };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_rgb3_to_rgba4(src: &[u8], dst: &mut [u8], pixels: usize, perm3: [u8; 4]) {
    use core::arch::x86_64::*;
    // Process 4 pixels per __m128i iteration: read 12 bytes (3 pixels
    // worth + 3 extra for the 4th), shuffle into a 16-byte register
    // with the alpha lane filled from a zeroed lane, then OR with 0xFF
    // at the alpha slot. We handle the 4-pixel block at a time from a
    // 16-byte load (which safely reads 4 extra bytes past the 3rd pixel
    // as long as we leave 4 pixels' tail for the scalar path).
    //
    // Mask entry j (within 16-byte output):
    //   pixel = j / 4, slot = j % 4, base = pixel * 3.
    //   if perm3[slot] == 0xFF: 0x80 (pshufb emits zero); we'll OR 0xFF later.
    //   else: base + perm3[slot].
    let mut mask_bytes = [0u8; 16];
    let mut alpha_or = [0u8; 16];
    for pix in 0..4 {
        let base = (pix * 3) as u8;
        for (slot, &p) in perm3.iter().enumerate() {
            let idx = pix * 4 + slot;
            if p == 0xFF {
                mask_bytes[idx] = 0x80;
                alpha_or[idx] = 0xFF;
            } else {
                mask_bytes[idx] = base + p;
            }
        }
    }
    let mask = _mm_loadu_si128(mask_bytes.as_ptr() as *const __m128i);
    let alpha = _mm_loadu_si128(alpha_or.as_ptr() as *const __m128i);

    // Advance 12 src bytes → 16 dst bytes per iter. Leave a tail of at
    // least 4 pixels for the scalar path so the 16-byte load doesn't
    // read past the end of src.
    let simd_blocks = if pixels >= 4 { (pixels - 4) / 4 } else { 0 };
    for c in 0..simd_blocks {
        let soff = c * 12;
        let doff = c * 16;
        let v = _mm_loadu_si128(src.as_ptr().add(soff) as *const __m128i);
        let shuffled = _mm_shuffle_epi8(v, mask);
        let out = _mm_or_si128(shuffled, alpha);
        _mm_storeu_si128(dst.as_mut_ptr().add(doff) as *mut __m128i, out);
    }
    let tail_start = simd_blocks * 4;
    scalar_rgb3_to_rgba4(
        &src[tail_start * 3..],
        &mut dst[tail_start * 4..],
        pixels - tail_start,
        perm3,
    );
}

// -------------------------------------------------------------------------
// rgba4 → rgb3 — demote (4→3 bytes, drop alpha).
//
// `perm4[i]` (for i in 0..3) contains the source byte offset within the
// 4-byte input group that becomes destination byte `i`.

pub(crate) fn rgba4_to_rgb3_perm(src: &[u8], dst: &mut [u8], pixels: usize, perm4: [u8; 3]) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2_rgba4_to_rgb3(src, dst, pixels, perm4) },
        _ => scalar_rgba4_to_rgb3(src, dst, pixels, perm4),
    }
}

#[inline(always)]
fn scalar_rgba4_to_rgb3(src: &[u8], dst: &mut [u8], pixels: usize, perm4: [u8; 3]) {
    let p0 = (perm4[0] & 3) as usize;
    let p1 = (perm4[1] & 3) as usize;
    let p2 = (perm4[2] & 3) as usize;
    for i in 0..pixels {
        let s = i * 4;
        let d = i * 3;
        dst[d] = src[s + p0];
        dst[d + 1] = src[s + p1];
        dst[d + 2] = src[s + p2];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba4_to_rgb3(src: &[u8], dst: &mut [u8], pixels: usize, perm4: [u8; 3]) {
    use core::arch::x86_64::*;
    // 4 pixels per iteration: read 16 bytes (exactly the pshufb source
    // range), shuffle into 12 consecutive dst bytes. Per-iteration we
    // store 16 bytes but advance dst by 12; the 4 trailing bytes get
    // overwritten by the next iteration. The scalar tail covers the
    // final 4 pixels so the last SIMD iteration's spill is guaranteed
    // to be overwritten.
    let mut mask_bytes = [0x80u8; 16];
    for pix in 0..4 {
        let sbase = (pix * 4) as u8;
        let dbase = pix * 3;
        for j in 0..3 {
            mask_bytes[dbase + j] = sbase + perm4[j];
        }
    }
    let mask = _mm_loadu_si128(mask_bytes.as_ptr() as *const __m128i);

    let simd_blocks = if pixels >= 4 { (pixels - 4) / 4 } else { 0 };
    for c in 0..simd_blocks {
        let soff = c * 16;
        let doff = c * 12;
        let v = _mm_loadu_si128(src.as_ptr().add(soff) as *const __m128i);
        let out = _mm_shuffle_epi8(v, mask);
        _mm_storeu_si128(dst.as_mut_ptr().add(doff) as *mut __m128i, out);
    }
    let tail_start = simd_blocks * 4;
    scalar_rgba4_to_rgb3(
        &src[tail_start * 4..],
        &mut dst[tail_start * 3..],
        pixels - tail_start,
        perm4,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_swizzle3(src: &[u8], dst: &mut [u8], pixels: usize, perm: [u8; 3]) {
    use core::arch::x86_64::*;
    // Each __m128i lane holds 5 whole pixels (15 bytes) plus one dead
    // byte at index 15. We build a 16-byte mask that picks the
    // permuted bytes for those 5 pixels and fills byte 15 with 0x80
    // (which makes `pshufb` emit a zero there — we overwrite it at the
    // store or just ignore it because we advance the dst by 15 bytes
    // each iteration).
    let mut mask_bytes = [0x80u8; 16];
    for pix in 0..5 {
        let base = pix * 3;
        for j in 0..3 {
            mask_bytes[base + j] = (base as u8) + perm[j];
        }
    }
    let mask = _mm_loadu_si128(mask_bytes.as_ptr() as *const __m128i);

    // We process 5 pixels per iteration. To emit exactly 15 bytes
    // without overwriting subsequent output with the zero at lane 15,
    // we use a masked store via _mm_maskmoveu_si128 — but that's
    // expensive. Simpler: write 16 bytes and rely on the next
    // iteration to overwrite the dead byte (which becomes the 16th
    // written byte of the previous iteration). Only the very last
    // iteration needs care; we stop the SIMD loop one block short and
    // let the scalar tail finish safely.
    let avail = pixels;
    let simd_blocks = if avail >= 5 { (avail - 5) / 5 } else { 0 };
    // simd_blocks iterations process 5 pixels each. Remaining pixels go
    // to the scalar path, which always includes at least the last
    // 5-pixel block so the 16th-byte spill is guaranteed to be
    // overwritten by the scalar store.
    for c in 0..simd_blocks {
        let soff = c * 15;
        let doff = c * 15;
        let v = _mm_loadu_si128(src.as_ptr().add(soff) as *const __m128i);
        let out = _mm_shuffle_epi8(v, mask);
        _mm_storeu_si128(dst.as_mut_ptr().add(doff) as *mut __m128i, out);
    }
    let tail_start = simd_blocks * 5;
    scalar_swizzle3(
        &src[tail_start * 3..],
        &mut dst[tail_start * 3..],
        pixels - tail_start,
        perm,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // These three SIMD paths share the over-store pattern that caused the
    // `rgb48_to_rgb24` glibc-abort regression (commit 2e821b7): each AVX2
    // iteration writes a full 16-byte register but advances the dst cursor
    // by fewer bytes (12 for rgba4→rgb3, 15 for swizzle3), relying on the
    // reserved scalar tail to overwrite the dead bytes and keep the final
    // iteration's 16-byte footprint inside `dst`. If a future edit shrinks
    // the `-4` / `-5` tail reserve, the last store walks past the slice and
    // corrupts adjacent heap — on x86_64 Linux that surfaces as a glibc
    // `sysmalloc` assertion abort, *not* a value mismatch, so a content
    // comparison against an oversized buffer would silently miss it. These
    // tests pin the write footprint with tightly-sized `dst` Vecs (backing
    // allocation ends exactly at the output length) plus an alloc-churn
    // loop so a reintroduced over-run trips the allocator deterministically.
    //
    // `swizzle4` / `rgb3_to_rgba4` advance the dst cursor by the same number
    // of bytes they store (16), so they can't over-run; they're covered here
    // anyway for completeness and to lock in the read-side tail reserve.

    fn ramp(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i & 0xff) as u8).collect()
    }

    #[test]
    fn swizzle3_does_not_write_past_dst() {
        // Identity 3-byte permutation; check value + tight write footprint
        // across every pixel count spanning the 5-pixel AVX2 block + tail.
        let perm = [0u8, 1, 2];
        for pixels in 1..=80 {
            let src = ramp(pixels * 3);
            let mut dst = vec![0u8; pixels * 3];
            swizzle3_perm(&src, &mut dst, pixels, perm);
            assert_eq!(dst, src[..pixels * 3], "swizzle3 pixels={pixels}");
        }
    }

    #[test]
    fn rgba4_to_rgb3_does_not_write_past_dst() {
        // perm4 picks src bytes 0,1,2 of each 4-byte group into the 3-byte
        // output. Exercise the 4-pixel AVX2 block + 4-pixel reserved tail.
        let perm4 = [0u8, 1, 2];
        for pixels in 1..=80 {
            let src = ramp(pixels * 4);
            let mut dst = vec![0u8; pixels * 3];
            rgba4_to_rgb3_perm(&src, &mut dst, pixels, perm4);
            for i in 0..pixels {
                assert_eq!(dst[i * 3], src[i * 4], "rgba4→rgb3 px {i}/{pixels}");
                assert_eq!(dst[i * 3 + 1], src[i * 4 + 1], "rgba4→rgb3 px {i}/{pixels}");
                assert_eq!(dst[i * 3 + 2], src[i * 4 + 2], "rgba4→rgb3 px {i}/{pixels}");
            }
        }
    }

    #[test]
    fn rgb3_to_rgba4_does_not_read_or_write_past_bounds() {
        // perm3: bytes 0,1,2 then a synthesised alpha (0xFF). Tight src AND
        // dst so the 16-byte SIMD load can't over-read past src either.
        let perm3 = [0u8, 1, 2, 0xFF];
        for pixels in 1..=80 {
            let src = ramp(pixels * 3);
            let mut dst = vec![0u8; pixels * 4];
            rgb3_to_rgba4_perm(&src, &mut dst, pixels, perm3);
            for i in 0..pixels {
                assert_eq!(dst[i * 4], src[i * 3], "rgb3→rgba4 px {i}/{pixels}");
                assert_eq!(dst[i * 4 + 1], src[i * 3 + 1], "rgb3→rgba4 px {i}/{pixels}");
                assert_eq!(dst[i * 4 + 2], src[i * 3 + 2], "rgb3→rgba4 px {i}/{pixels}");
                assert_eq!(dst[i * 4 + 3], 255, "rgb3→rgba4 alpha px {i}/{pixels}");
            }
        }
    }

    #[test]
    fn swizzle4_does_not_write_past_dst() {
        let perm = [0u8, 1, 2, 3];
        for pixels in 1..=80 {
            let src = ramp(pixels * 4);
            let mut dst = vec![0u8; pixels * 4];
            swizzle4_perm(&src, &mut dst, pixels, perm);
            assert_eq!(dst, src[..pixels * 4], "swizzle4 pixels={pixels}");
        }
    }

    // Hammer the over-store paths by allocating + freeing tight dst Vecs
    // thousands of times: a heap stomp from an earlier iteration trips
    // glibc's malloc on a later free even when the comparison above passed
    // (the spill may land in metadata the comparison never reads).
    #[test]
    fn swizzle_simd_alloc_churn_does_not_corrupt_heap() {
        for _ in 0..2000 {
            // swizzle3: 33 pixels = 6 full 5-pixel SIMD blocks + tail.
            let s3 = vec![0xa5u8; 33 * 3];
            let mut d3 = vec![0u8; 33 * 3];
            swizzle3_perm(&s3, &mut d3, 33, [0, 1, 2]);
            drop(d3);

            // rgba4→rgb3: 33 pixels exercises the 12-byte-advance store.
            let s4 = vec![0x5au8; 33 * 4];
            let mut d4 = vec![0u8; 33 * 3];
            rgba4_to_rgb3_perm(&s4, &mut d4, 33, [0, 1, 2]);
            drop(d4);
        }
    }
}
