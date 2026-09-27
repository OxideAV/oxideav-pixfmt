//! Row-band engine for the 8-bit planar YUV(A) ↔ packed RGB(A) hops.
//!
//! Every `convert()` row that moves between an 8-bit planar YUV family
//! member (`Yuv*`, `YuvJ*`, `Yuva*`, and the deep members once their
//! planes have been narrowed to bytes) and `Rgb24` / `Rgba` funnels
//! through the two functions here. They do three things the individual
//! table rows used to do separately and slowly:
//!
//! - **No whole-frame staging copies.** Tightly packed source planes
//!   are borrowed; only stride-padded planes are gathered. RGBA input
//!   is de-interleaved a few rows at a time into an L1-resident scratch
//!   instead of a whole-frame `Vec<u8>` built by `push`.
//! - **Fused RGBA output.** The RGB row is decoded into a scratch row
//!   and interleaved with the alpha plane (or an opaque byte) in one
//!   pass, instead of decoding a whole RGB24 frame and re-walking it
//!   pixel by pixel.
//! - **Row bands.** The picture is cut into `workers` bands of whole
//!   chroma rows and each band is converted independently; with
//!   `workers > 1` the bands run on scoped threads. The band split never
//!   changes a byte: every kernel is a per-row loop whose chroma row is
//!   `row / hsub`, so a band starting on a chroma-row boundary sees the
//!   same samples the whole-frame call would.
//!
//! The arithmetic itself is unchanged — the bands call the same
//! `crate::yuv` kernels (scalar / NEON / AVX2, all bit-identical to the
//! scalar Q15 reference) that the historical rows called on the whole
//! frame.
//!
//! # Odd dimensions
//!
//! Subsampled chroma planes hold `ceil(w / wsub)` × `ceil(h / hsub)`
//! samples (the component-dimension rule of ITU-T T.81 A.1.1, and the
//! `PixelFormat::plane_dimensions` contract in oxideav-core). Decoding
//! indexes chroma as `col / wsub`, `row / hsub`, so the trailing odd
//! luma column / row reads the last chroma sample. Encoding averages
//! the luma positions of each chroma block with the missing positions
//! replicating the last column / row, which makes decode ∘ encode
//! consistent at the edge and reproduces the even-dimension bytes
//! exactly when nothing is missing.

use std::borrow::Cow;

use oxideav_core::{Error, Result, VideoPlane};

use crate::yuv::{self, YuvMatrix};

/// Rows below which a band is not worth a thread of its own: bands are
/// sized to at least this many rows, so small frames stay serial even
/// under a large worker budget.
pub(crate) const MIN_BAND_ROWS: usize = 32;

/// Chroma plane dimensions for a `w × h` picture under `(wsub, hsub)`
/// subsampling: `ceil(w / wsub)` × `ceil(h / hsub)`.
#[inline]
pub(crate) fn chroma_dims(w: usize, h: usize, wsub: usize, hsub: usize) -> (usize, usize) {
    (w.div_ceil(wsub), h.div_ceil(hsub))
}

/// Borrow a plane's `w_bytes × h` payload tightly: a zero-copy borrow
/// when the stride already equals the row width, a gathered copy when
/// the plane carries padding. A plane too short for the geometry is an
/// `Error::Invalid` rather than a panic.
pub(crate) fn tight_plane(plane: &VideoPlane, w_bytes: usize, h: usize) -> Result<Cow<'_, [u8]>> {
    if h == 0 || w_bytes == 0 {
        return Ok(Cow::Borrowed(&[]));
    }
    if plane.stride < w_bytes {
        return Err(Error::invalid(format!(
            "pixfmt: plane stride {} shorter than its {w_bytes}-byte rows",
            plane.stride
        )));
    }
    // The last row only needs `w_bytes`, not a whole stride.
    let need = plane
        .stride
        .checked_mul(h - 1)
        .and_then(|v| v.checked_add(w_bytes))
        .ok_or_else(|| Error::invalid("pixfmt: plane geometry overflows usize"))?;
    if plane.data.len() < need {
        return Err(Error::invalid(format!(
            "pixfmt: plane holds {} bytes, geometry needs {need}",
            plane.data.len()
        )));
    }
    if plane.stride == w_bytes {
        return Ok(Cow::Borrowed(&plane.data[..need]));
    }
    let mut out = Vec::with_capacity(w_bytes * h);
    for row in 0..h {
        let off = row * plane.stride;
        out.extend_from_slice(&plane.data[off..off + w_bytes]);
    }
    Ok(Cow::Owned(out))
}

/// Split `h` rows into at most `workers` bands whose boundaries fall on
/// multiples of `hsub` (so each band owns whole chroma rows) and which
/// hold at least [`MIN_BAND_ROWS`] rows apart from the last. Returns
/// `(row0, row1)` half-open pairs covering `0..h` in order.
pub(crate) fn row_bands(h: usize, hsub: usize, workers: usize) -> Vec<(usize, usize)> {
    if h == 0 {
        return Vec::new();
    }
    let workers = workers.max(1);
    let mut per = h.div_ceil(workers).max(MIN_BAND_ROWS);
    // Round the band height up to a whole number of chroma rows.
    per = per.div_ceil(hsub) * hsub;
    let mut out = Vec::with_capacity(h.div_ceil(per));
    let mut r0 = 0;
    while r0 < h {
        let r1 = (r0 + per).min(h);
        out.push((r0, r1));
        r0 = r1;
    }
    out
}

/// Run `f` once per band. A single band (or a single worker) runs
/// inline on the caller's thread; otherwise the bands are dispatched on
/// scoped threads, one per band, and joined before returning. The
/// caller has already carved every band's output into disjoint `&mut`
/// slices, so the closure receives only its own band's state.
fn run_bands<T, F>(mut bands: Vec<T>, f: F)
where
    T: Send,
    F: Fn(T) + Sync,
{
    if bands.len() <= 1 {
        for b in bands {
            f(b);
        }
        return;
    }
    // Keep one band for this thread — spawning `n - 1` helpers.
    let last = bands.pop();
    std::thread::scope(|scope| {
        for b in bands {
            scope.spawn(|| f(b));
        }
        if let Some(b) = last {
            f(b);
        }
    });
}

/// Interleave a tight RGB24 row with an alpha row (or opaque alpha)
/// into an RGBA row of `w` pixels.
#[inline]
fn rgb_row_to_rgba(rgb: &[u8], alpha: Option<&[u8]>, dst: &mut [u8], w: usize) {
    debug_assert!(rgb.len() >= w * 3 && dst.len() >= w * 4);
    #[cfg(target_arch = "aarch64")]
    {
        if crate::yuv_simd::neon_enabled() {
            // SAFETY: NEON is baseline on aarch64 and the dispatch gate
            // only returns true when runtime detection passed; the
            // slices are length-checked in the callee.
            unsafe { rgb_row_to_rgba_neon(rgb, alpha, dst, w) };
            return;
        }
    }
    rgb_row_to_rgba_scalar(rgb, alpha, dst, w);
}

#[inline]
fn rgb_row_to_rgba_scalar(rgb: &[u8], alpha: Option<&[u8]>, dst: &mut [u8], w: usize) {
    let rgb = &rgb[..w * 3];
    let dst = &mut dst[..w * 4];
    match alpha {
        Some(a) => {
            let a = &a[..w];
            for ((d, s), &av) in dst
                .chunks_exact_mut(4)
                .zip(rgb.chunks_exact(3))
                .zip(a.iter())
            {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                d[3] = av;
            }
        }
        None => {
            for (d, s) in dst.chunks_exact_mut(4).zip(rgb.chunks_exact(3)) {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                d[3] = 255;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rgb_row_to_rgba_neon(rgb: &[u8], alpha: Option<&[u8]>, dst: &mut [u8], w: usize) {
    use core::arch::aarch64::*;
    let rgb = &rgb[..w * 3];
    let dst = &mut dst[..w * 4];
    let full = w / 16;
    let opaque = vdupq_n_u8(255);
    for i in 0..full {
        let s = rgb.as_ptr().add(i * 48);
        let d = dst.as_mut_ptr().add(i * 64);
        let px = vld3q_u8(s);
        let a = match alpha {
            Some(a) => vld1q_u8(a.as_ptr().add(i * 16)),
            None => opaque,
        };
        vst4q_u8(d, uint8x16x4_t(px.0, px.1, px.2, a));
    }
    let done = full * 16;
    if done < w {
        rgb_row_to_rgba_scalar(
            &rgb[done * 3..],
            alpha.map(|a| &a[done..]),
            &mut dst[done * 4..],
            w - done,
        );
    }
}

/// Split a tight RGBA row into a tight RGB24 row and an alpha row.
#[inline]
fn rgba_row_split(src: &[u8], rgb: &mut [u8], alpha: Option<&mut [u8]>, w: usize) {
    let src = &src[..w * 4];
    let rgb = &mut rgb[..w * 3];
    #[cfg(target_arch = "aarch64")]
    {
        if crate::yuv_simd::neon_enabled() {
            // SAFETY: see `rgb_row_to_rgba`.
            unsafe { rgba_row_split_neon(src, rgb, alpha, w) };
            return;
        }
    }
    rgba_row_split_scalar(src, rgb, alpha, w);
}

#[inline]
fn rgba_row_split_scalar(src: &[u8], rgb: &mut [u8], alpha: Option<&mut [u8]>, w: usize) {
    let src = &src[..w * 4];
    let rgb = &mut rgb[..w * 3];
    match alpha {
        Some(a) => {
            let a = &mut a[..w];
            for ((s, d), av) in src
                .chunks_exact(4)
                .zip(rgb.chunks_exact_mut(3))
                .zip(a.iter_mut())
            {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                *av = s[3];
            }
        }
        None => {
            for (s, d) in src.chunks_exact(4).zip(rgb.chunks_exact_mut(3)) {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rgba_row_split_neon(src: &[u8], rgb: &mut [u8], alpha: Option<&mut [u8]>, w: usize) {
    use core::arch::aarch64::*;
    let full = w / 16;
    let mut alpha = alpha;
    for i in 0..full {
        let px = vld4q_u8(src.as_ptr().add(i * 64));
        vst3q_u8(rgb.as_mut_ptr().add(i * 48), uint8x16x3_t(px.0, px.1, px.2));
        if let Some(a) = alpha.as_deref_mut() {
            vst1q_u8(a.as_mut_ptr().add(i * 16), px.3);
        }
    }
    let done = full * 16;
    if done < w {
        rgba_row_split_scalar(
            &src[done * 4..],
            &mut rgb[done * 3..],
            alpha.map(|a| &mut a[done..]),
            w - done,
        );
    }
}

/// Decode `h` rows of tight 8-bit planar YUV into a tight packed RGB
/// stream, one call per row band (`yp` / `up` / `vp` / `dst` are the
/// band's slices). `wsub ∈ {1, 2}`, `hsub ∈ {1, 2}` with `(1, 2)`
/// excluded — 4:4:0 and 4:1:1 sources are broadcast to 4:4:4 by the
/// caller first.
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_rows_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    wsub: usize,
    hsub: usize,
    matrix: YuvMatrix,
) {
    match (wsub, hsub) {
        (1, 1) => yuv::yuv444_to_rgb24(yp, up, vp, dst, w, h, matrix),
        (2, 1) => yuv::yuv422_to_rgb24(yp, up, vp, dst, w, h, matrix),
        (2, 2) => yuv::yuv420_to_rgb24(yp, up, vp, dst, w, h, matrix),
        _ => unreachable!("planar8: sitings other than 4:4:4 / 4:2:2 / 4:2:0 are staged first"),
    }
}

/// Tight 8-bit planar YUV(A) source for [`decode`].
pub(crate) struct Planar8<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    /// Full-resolution alpha plane, when the source carries one.
    pub a: Option<&'a [u8]>,
    pub w: usize,
    pub h: usize,
    pub wsub: usize,
    pub hsub: usize,
}

/// Planar 8-bit YUV(A) → tight packed `Rgb24` (`bpp == 3`) or `Rgba`
/// (`bpp == 4`; alpha from the source plane when present, else opaque)
/// under `matrix`, using up to `workers` row bands in parallel.
pub(crate) fn decode(src: &Planar8<'_>, bpp: usize, matrix: YuvMatrix, workers: usize) -> Vec<u8> {
    debug_assert!(bpp == 3 || bpp == 4);
    let (w, h) = (src.w, src.h);
    let mut out = vec![0u8; w * h * bpp];
    if w == 0 || h == 0 {
        return out;
    }
    // Sitings the kernels do not take natively are broadcast to 4:4:4
    // chroma first — the same staging the historical rows performed,
    // so the bytes are unchanged.
    let staged: Option<(Vec<u8>, Vec<u8>)> = match (src.wsub, src.hsub) {
        (1, 2) => {
            let mut u = vec![0u8; w * h];
            let mut v = vec![0u8; w * h];
            yuv::chroma_440_to_444(src.u, &mut u, w, h);
            yuv::chroma_440_to_444(src.v, &mut v, w, h);
            Some((u, v))
        }
        (4, 1) => {
            let mut u = vec![0u8; w * h];
            let mut v = vec![0u8; w * h];
            yuv::chroma_411_to_444(src.u, &mut u, w, h);
            yuv::chroma_411_to_444(src.v, &mut v, w, h);
            Some((u, v))
        }
        _ => None,
    };
    let (up, vp, wsub, hsub) = match &staged {
        Some((u, v)) => (u.as_slice(), v.as_slice(), 1, 1),
        None => (src.u, src.v, src.wsub, src.hsub),
    };
    let (cw, _) = chroma_dims(w, h, wsub, hsub);
    let row_bytes = w * bpp;

    // Carve the output into per-band slices up front.
    let bands = row_bands(h, hsub, workers);
    let mut jobs = Vec::with_capacity(bands.len());
    let mut rest: &mut [u8] = &mut out;
    for &(r0, r1) in &bands {
        let (band, tail) = rest.split_at_mut((r1 - r0) * row_bytes);
        rest = tail;
        jobs.push((r0, r1, band));
    }

    let yp = src.y;
    let ap = src.a;
    run_bands(jobs, |(r0, r1, band): (usize, usize, &mut [u8])| {
        let rows = r1 - r0;
        let cr0 = r0 / hsub;
        let cr1 = r1.div_ceil(hsub);
        let yb = &yp[r0 * w..r1 * w];
        let ub = &up[cr0 * cw..cr1 * cw];
        let vb = &vp[cr0 * cw..cr1 * cw];
        if bpp == 3 {
            decode_rows_rgb24(yb, ub, vb, band, w, rows, wsub, hsub, matrix);
            return;
        }
        // RGBA: decode one row at a time into an L1-resident scratch
        // row, then interleave with alpha.
        let mut scratch = vec![0u8; w * 3];
        for (i, drow) in band.chunks_exact_mut(w * 4).enumerate() {
            let row = r0 + i;
            let cr = row / hsub - cr0;
            let yrow = &yb[i * w..(i + 1) * w];
            let urow = &ub[cr * cw..(cr + 1) * cw];
            let vrow = &vb[cr * cw..(cr + 1) * cw];
            decode_rows_rgb24(yrow, urow, vrow, &mut scratch, w, 1, wsub, 1, matrix);
            let arow = ap.map(|a| &a[row * w..(row + 1) * w]);
            rgb_row_to_rgba(&scratch, arow, drow, w);
        }
    });
    out
}

/// Encode `rows` tight RGB24 rows into the band's Y / U / V slices.
#[inline]
#[allow(clippy::too_many_arguments)]
fn encode_rows_rgb24(
    rgb: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    rows: usize,
    wsub: usize,
    hsub: usize,
    matrix: YuvMatrix,
) {
    match (wsub, hsub) {
        (1, 1) => yuv::rgb24_to_yuv444(rgb, yp, up, vp, w, rows, matrix),
        (2, 1) => yuv::rgb24_to_yuv422(rgb, yp, up, vp, w, rows, matrix),
        (2, 2) => yuv::rgb24_to_yuv420(rgb, yp, up, vp, w, rows, matrix),
        _ => unreachable!("planar8: sitings other than 4:4:4 / 4:2:2 / 4:2:0 are staged after"),
    }
}

/// Packed RGB source for [`encode`]: `bpp ∈ {3, 4}` (`Rgb24` / `Rgba`),
/// rows `stride` bytes apart.
pub(crate) struct Packed8<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub bpp: usize,
    pub w: usize,
    pub h: usize,
}

/// One encode band: its row range and its slices of the Y / U / V (/ A)
/// output planes.
type EncodeJob<'a> = (
    usize,
    usize,
    &'a mut [u8],
    &'a mut [u8],
    &'a mut [u8],
    Option<&'a mut [u8]>,
);

/// Tight planar output of [`encode`].
pub(crate) struct Planar8Out {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    /// The source's alpha plane, split out when requested and present.
    pub a: Option<Vec<u8>>,
    pub cw: usize,
    pub ch: usize,
}

/// Packed `Rgb24` / `Rgba` → tight 8-bit planar YUV at `(wsub, hsub)`
/// ∈ {4:4:4, 4:2:2, 4:2:0} under `matrix`, using up to `workers` row
/// bands in parallel. `want_alpha` splits the RGBA source's fourth byte
/// into a full-resolution alpha plane (`None` for an `Rgb24` source —
/// the caller synthesises opaque).
pub(crate) fn encode(
    src: &Packed8<'_>,
    wsub: usize,
    hsub: usize,
    matrix: YuvMatrix,
    want_alpha: bool,
    workers: usize,
) -> Planar8Out {
    let (w, h) = (src.w, src.h);
    let (cw, ch) = chroma_dims(w, h, wsub, hsub);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    let split_alpha = want_alpha && src.bpp == 4;
    let mut a = if split_alpha {
        Some(vec![0u8; w * h])
    } else {
        None
    };
    if w == 0 || h == 0 {
        return Planar8Out { y, u, v, a, cw, ch };
    }

    let bands = row_bands(h, hsub, workers);
    let mut jobs: Vec<EncodeJob<'_>> = Vec::with_capacity(bands.len());
    {
        let mut yr: &mut [u8] = &mut y;
        let mut ur: &mut [u8] = &mut u;
        let mut vr: &mut [u8] = &mut v;
        let mut ar: Option<&mut [u8]> = a.as_deref_mut();
        for &(r0, r1) in &bands {
            let cr0 = r0 / hsub;
            let cr1 = r1.div_ceil(hsub);
            let (yb, yt) = yr.split_at_mut((r1 - r0) * w);
            yr = yt;
            let (ub, ut) = ur.split_at_mut((cr1 - cr0) * cw);
            ur = ut;
            let (vb, vt) = vr.split_at_mut((cr1 - cr0) * cw);
            vr = vt;
            let ab = match ar.take() {
                Some(rest) => {
                    let (ab, at) = rest.split_at_mut((r1 - r0) * w);
                    ar = Some(at);
                    Some(ab)
                }
                None => None,
            };
            jobs.push((r0, r1, yb, ub, vb, ab));
        }
    }

    let direct = src.bpp == 3 && src.stride == w * 3;
    let data = src.data;
    let stride = src.stride;
    let bpp = src.bpp;
    run_bands(jobs, |(r0, r1, yb, ub, vb, ab): EncodeJob<'_>| {
        let rows = r1 - r0;
        if direct {
            let band = &data[r0 * w * 3..r1 * w * 3];
            encode_rows_rgb24(band, yb, ub, vb, w, rows, wsub, hsub, matrix);
            if let Some(ab) = ab {
                ab.fill(0xFF);
            }
            return;
        }
        // Stride-padded or RGBA input: stage `hsub` rows at a time
        // through a small tight RGB24 scratch (and peel alpha off).
        let mut scratch = vec![0u8; w * 3 * hsub];
        let mut ab = ab;
        let mut r = 0;
        while r < rows {
            let n = hsub.min(rows - r);
            for i in 0..n {
                let row = r0 + r + i;
                let srow = &data[row * stride..row * stride + w * bpp];
                let drow = &mut scratch[i * w * 3..(i + 1) * w * 3];
                if bpp == 4 {
                    let arow = ab
                        .as_deref_mut()
                        .map(|ab| &mut ab[(r + i) * w..(r + i + 1) * w]);
                    rgba_row_split(srow, drow, arow, w);
                } else {
                    drow.copy_from_slice(srow);
                }
            }
            let cr = r / hsub;
            let crn = (r + n).div_ceil(hsub) - cr;
            encode_rows_rgb24(
                &scratch[..n * w * 3],
                &mut yb[r * w..(r + n) * w],
                &mut ub[cr * cw..(cr + crn) * cw],
                &mut vb[cr * cw..(cr + crn) * cw],
                w,
                n,
                wsub,
                hsub,
                matrix,
            );
            r += n;
        }
    });
    Planar8Out { y, u, v, a, cw, ch }
}

/// Narrow a tight plane of `count` LE16 words at `bits` to bytes (the
/// crate's truncating depth move, `yuv::depth_down_le16_plane`), split
/// into up to `workers` chunks that run on scoped threads.
pub(crate) fn narrow_to_8(src: &[u8], count: usize, bits: u32, workers: usize) -> Vec<u8> {
    let mut out = vec![0u8; count];
    let parts = workers.max(1).min(count.div_ceil(1 << 16)).max(1);
    let per = count.div_ceil(parts).max(1);
    let jobs: Vec<(usize, &mut [u8])> = out
        .chunks_mut(per)
        .enumerate()
        .map(|(i, c)| (i * per, c))
        .collect();
    run_bands(jobs, |(start, chunk): (usize, &mut [u8])| {
        let n = chunk.len();
        yuv::depth_down_le16_plane(&src[start * 2..(start + n) * 2], chunk, n, bits);
    });
    out
}

// ---------------------------------------------------------------------
// Deep planar YUV(A) → packed 16-bit RGB(A) (the Q30 deep matrix).

/// Tight planar YUV(A) source for [`decode_deep`]: samples are bytes
/// when `bits == 8`, else LE16 words with `bits` significant low bits.
pub(crate) struct PlanarDeep<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub a: Option<&'a [u8]>,
    pub w: usize,
    pub h: usize,
    pub wsub: usize,
    pub hsub: usize,
    pub bits: u32,
}

/// Widen a row of `n` samples at `bits` to 16 bits by the crate's
/// MSB-replicating rule (`×257` for bytes; `(v << d) | (v >> (bits − d))`
/// for words), masking stray high bits first — the exact mapping of
/// `yuv::depth_up_8_to_le16_plane` / `yuv::depth_rescale_le16_plane`.
#[inline]
fn widen_row16(src: &[u8], dst: &mut [u16], bits: u32) {
    if bits <= 8 {
        for (d, &v) in dst.iter_mut().zip(src.iter()) {
            *d = ((v as u16) << 8) | v as u16;
        }
    } else if bits >= 16 {
        for (d, b) in dst.iter_mut().zip(src.chunks_exact(2)) {
            *d = u16::from_le_bytes([b[0], b[1]]);
        }
    } else {
        let mask = (1u32 << bits) - 1;
        let sh = 16 - bits;
        let back = bits - sh;
        for (d, b) in dst.iter_mut().zip(src.chunks_exact(2)) {
            let v = u16::from_le_bytes([b[0], b[1]]) as u32 & mask;
            *d = ((v << sh) | (v >> back)) as u16;
        }
    }
}

/// Planar YUV(A) at any family depth → tight packed `Rgb48Le`
/// (`alpha_out == false`) or `Rgba64Le` through the full-precision Q30
/// deep matrix, row-banded like [`decode`]. Every sample is widened to
/// 16 bits (MSB replication), chroma is read nearest-neighbour at
/// `(col / wsub, row / hsub)`, and alpha is the widened source plane or
/// opaque 65535 — sample for sample what the historical route (widen to
/// the 16-bit 4:4:4 tier, upsample, decode) produced, without the three
/// whole-frame intermediates.
pub(crate) fn decode_deep(
    src: &PlanarDeep<'_>,
    alpha_out: bool,
    matrix: YuvMatrix,
    workers: usize,
) -> Vec<u8> {
    let (w, h) = (src.w, src.h);
    let bpp = if alpha_out { 8 } else { 6 };
    let mut out = vec![0u8; w * h * bpp];
    if w == 0 || h == 0 {
        return out;
    }
    let (cw, _) = chroma_dims(w, h, src.wsub, src.hsub);
    let d = matrix.decode_params16();
    let row_bytes = w * bpp;
    let bands = row_bands(h, src.hsub, workers);
    let mut jobs = Vec::with_capacity(bands.len());
    let mut rest: &mut [u8] = &mut out;
    for &(r0, r1) in &bands {
        let (band, tail) = rest.split_at_mut((r1 - r0) * row_bytes);
        rest = tail;
        jobs.push((r0, band));
    }
    let (wsub, hsub, bits) = (src.wsub, src.hsub, src.bits);
    let sb = if bits > 8 { 2 } else { 1 };
    run_bands(jobs, |(r0, band): (usize, &mut [u8])| {
        // Per-row 16-bit staging: luma (and alpha) every row, chroma
        // once per chroma row.
        let mut y16 = vec![0u16; w];
        let mut u16r = vec![0u16; cw];
        let mut v16r = vec![0u16; cw];
        let mut a16 = vec![u16::MAX; w];
        let mut staged_cr = usize::MAX;
        for (i, drow) in band.chunks_exact_mut(row_bytes).enumerate() {
            let row = r0 + i;
            widen_row16(&src.y[row * w * sb..(row + 1) * w * sb], &mut y16, bits);
            let cr = row / hsub;
            if cr != staged_cr {
                widen_row16(&src.u[cr * cw * sb..(cr + 1) * cw * sb], &mut u16r, bits);
                widen_row16(&src.v[cr * cw * sb..(cr + 1) * cw * sb], &mut v16r, bits);
                staged_cr = cr;
            }
            if alpha_out {
                if let Some(a) = src.a {
                    widen_row16(&a[row * w * sb..(row + 1) * w * sb], &mut a16, bits);
                }
            }
            for (col, px) in drow.chunks_exact_mut(bpp).enumerate() {
                let ci = col / wsub;
                let (r, g, b) = yuv::yuv16_to_rgb48_fp(y16[col], u16r[ci], v16r[ci], &d);
                px[0..2].copy_from_slice(&r.to_le_bytes());
                px[2..4].copy_from_slice(&g.to_le_bytes());
                px[4..6].copy_from_slice(&b.to_le_bytes());
                if alpha_out {
                    px[6..8].copy_from_slice(&a16[col].to_le_bytes());
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_rows_on_chroma_boundaries() {
        for h in [1usize, 2, 3, 31, 32, 33, 64, 65, 1080, 3023, 3024] {
            for workers in [1usize, 2, 3, 7, 12, 64] {
                let b = row_bands(h, 2, workers);
                assert_eq!(b.first().map(|x| x.0), Some(0));
                assert_eq!(b.last().map(|x| x.1), Some(h));
                assert!(b.len() <= workers.max(1));
                for (i, &(r0, r1)) in b.iter().enumerate() {
                    assert!(r1 > r0);
                    assert_eq!(r0 % 2, 0, "band {i} starts mid chroma row");
                    if i + 1 < b.len() {
                        assert_eq!(r1, b[i + 1].0);
                        assert!(r1 - r0 >= MIN_BAND_ROWS);
                    }
                }
            }
        }
        assert!(row_bands(0, 2, 4).is_empty());
    }

    #[test]
    fn chroma_dims_round_up() {
        assert_eq!(chroma_dims(1, 1, 2, 2), (1, 1));
        assert_eq!(chroma_dims(3, 5, 2, 2), (2, 3));
        assert_eq!(chroma_dims(7, 3, 2, 1), (4, 3));
        assert_eq!(chroma_dims(4032, 3024, 2, 2), (2016, 1512));
    }

    #[test]
    fn tight_plane_borrows_or_gathers() {
        let tight = VideoPlane {
            stride: 4,
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };
        assert!(matches!(
            tight_plane(&tight, 4, 2).unwrap(),
            Cow::Borrowed(_)
        ));
        let padded = VideoPlane {
            stride: 6,
            data: vec![1, 2, 3, 4, 0, 0, 5, 6, 7, 8],
        };
        let g = tight_plane(&padded, 4, 2).unwrap();
        assert_eq!(&*g, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(tight_plane(&padded, 4, 3).is_err());
        assert!(tight_plane(&padded, 8, 1).is_err());
    }

    #[test]
    fn rgba_interleave_and_split_round_trip() {
        for w in [0usize, 1, 5, 15, 16, 17, 33, 100] {
            let rgb: Vec<u8> = (0..w * 3).map(|i| (i * 7 + 3) as u8).collect();
            let alpha: Vec<u8> = (0..w).map(|i| (i * 13 + 1) as u8).collect();
            let mut rgba = vec![0u8; w * 4];
            rgb_row_to_rgba(&rgb, Some(&alpha), &mut rgba, w);
            for i in 0..w {
                assert_eq!(&rgba[i * 4..i * 4 + 3], &rgb[i * 3..i * 3 + 3]);
                assert_eq!(rgba[i * 4 + 3], alpha[i]);
            }
            let mut rgb2 = vec![0u8; w * 3];
            let mut a2 = vec![0u8; w];
            rgba_row_split(&rgba, &mut rgb2, Some(&mut a2), w);
            assert_eq!(rgb2, rgb);
            assert_eq!(a2, alpha);
            let mut opaque = vec![0u8; w * 4];
            rgb_row_to_rgba(&rgb, None, &mut opaque, w);
            assert!(opaque.chunks_exact(4).all(|p| p[3] == 255));
        }
    }
}
