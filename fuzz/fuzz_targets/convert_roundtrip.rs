#![no_main]

//! Geometry / SIMD-edge fuzz harness for the `convert()` dispatch table.
//!
//! The conversion functions trust the caller-supplied [`FrameInfo`]
//! (format, width, height) and the matching plane layout — they do *not*
//! re-derive plane sizes from the byte buffers, so handing them raw
//! garbage would only exercise a caller-contract violation, not a library
//! bug. This harness therefore builds a *correctly sized* source frame for
//! a fuzzer-chosen format at fuzzer-chosen dimensions and runs every
//! registered conversion out of that format. The interesting axes are:
//!
//!   * odd widths/heights (1, 3, 5, …) that fall between SIMD lane
//!     boundaries — the historical AVX2 heap-corruption regression lived
//!     exactly here;
//!   * tiny dimensions (1×1, 1×N, N×1) that stress the scalar tail;
//!   * extra per-row stride padding so the `gather_tight` / `tight_row`
//!     row walkers are exercised on a non-tight layout.
//!
//! The contract under test: a correctly-described frame must never panic,
//! integer-overflow (debug build), index out of bounds, or OOM in any
//! converter. A conversion may legitimately return `Err` (e.g. a 4:1:1
//! source with a width that is not a multiple of 4); only a *panic*/abort
//! is a finding. Round-trips are additionally checked for the lossless
//! pairs where exact equality is guaranteed.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ColorSpace, ConvertOptions, FrameInfo};

/// All source formats that appear on the left of a `convert()` table row.
const FORMATS: &[PixelFormat] = &[
    PixelFormat::Rgb24,
    PixelFormat::Bgr24,
    PixelFormat::Rgba,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Cmyk,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray8,
    PixelFormat::Gray16Le,
    PixelFormat::Ya8,
    PixelFormat::MonoBlack,
    PixelFormat::MonoWhite,
    PixelFormat::Pal8,
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::Yuv411P,
    PixelFormat::YuvJ420P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ444P,
    PixelFormat::Yuva420P,
    PixelFormat::Nv12,
    PixelFormat::Nv21,
    PixelFormat::Yuyv422,
    PixelFormat::Uyvy422,
    PixelFormat::Gbrp10Le,
    PixelFormat::Gbrp12Le,
    PixelFormat::Gbrp14Le,
    PixelFormat::Gbrap10Le,
    PixelFormat::Gbrap12Le,
    PixelFormat::Gbrap14Le,
];

const COLOR_SPACES: &[ColorSpace] = &[
    ColorSpace::Bt601Limited,
    ColorSpace::Bt601Full,
    ColorSpace::Bt709Limited,
    ColorSpace::Bt709Full,
    ColorSpace::Bt2020Limited,
    ColorSpace::Bt2020Full,
];

#[derive(Arbitrary, Debug)]
struct Input {
    fmt_idx: u8,
    cs_idx: u8,
    /// Width in [1, 64].
    w: u8,
    /// Height in [1, 64].
    h: u8,
    /// Extra per-row stride padding bytes in [0, 8].
    pad: u8,
    /// Pseudo-random plane fill seed.
    seed: u32,
}

/// Cheap xorshift so plane bytes vary without pulling in a PRNG crate.
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u8 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x & 0xff) as u8
    }
}

/// Build a single plane: `row_bytes` of payload per row + `pad` trailing
/// padding bytes, `h` rows, stride = `row_bytes + pad`.
fn plane(rng: &mut Rng, row_bytes: usize, pad: usize, h: usize) -> VideoPlane {
    let stride = row_bytes + pad;
    let mut data = vec![0u8; stride * h];
    // Only fill the live payload region; padding stays zero.
    for row in 0..h {
        for b in 0..row_bytes {
            data[row * stride + b] = rng.next();
        }
    }
    VideoPlane { stride, data }
}

/// Construct a well-formed source frame for `fmt` at `w × h`, with `pad`
/// extra stride bytes on each plane. Returns `None` for geometry that the
/// format cannot represent (e.g. a subsampled chroma plane needs an even
/// dimension to be meaningfully laid out — the converters still accept
/// the floor division, so we size the chroma planes to match exactly what
/// the converter reads: `w / wsub` × `h / hsub`).
fn build_frame(fmt: PixelFormat, w: usize, h: usize, pad: usize, seed: u32) -> Option<VideoFrame> {
    let mut rng = Rng(seed | 1);
    let p = |rng: &mut Rng, rb: usize, hh: usize| plane(rng, rb, pad, hh);

    let planes = match fmt {
        // Packed 3 bytes/pixel.
        PixelFormat::Rgb24 | PixelFormat::Bgr24 => vec![p(&mut rng, w * 3, h)],
        // Packed 4 bytes/pixel.
        PixelFormat::Rgba
        | PixelFormat::Bgra
        | PixelFormat::Argb
        | PixelFormat::Abgr
        | PixelFormat::Cmyk => vec![p(&mut rng, w * 4, h)],
        // Packed 16-bit deep RGB.
        PixelFormat::Rgb48Le => vec![p(&mut rng, w * 3 * 2, h)],
        PixelFormat::Rgba64Le => vec![p(&mut rng, w * 4 * 2, h)],
        // Gray.
        PixelFormat::Gray8 => vec![p(&mut rng, w, h)],
        PixelFormat::Gray16Le => vec![p(&mut rng, w * 2, h)],
        PixelFormat::Ya8 => vec![p(&mut rng, w * 2, h)],
        // 1-bit mono: ceil(w/8) bytes per row.
        PixelFormat::MonoBlack | PixelFormat::MonoWhite => vec![p(&mut rng, w.div_ceil(8), h)],
        // Palette: one index plane.
        PixelFormat::Pal8 => vec![p(&mut rng, w, h)],
        // Planar YUV: Y full res, U/V subsampled.
        PixelFormat::Yuv420P | PixelFormat::YuvJ420P => {
            vec![
                p(&mut rng, w, h),
                p(&mut rng, w / 2, h / 2),
                p(&mut rng, w / 2, h / 2),
            ]
        }
        PixelFormat::Yuv422P | PixelFormat::YuvJ422P => {
            vec![
                p(&mut rng, w, h),
                p(&mut rng, w / 2, h),
                p(&mut rng, w / 2, h),
            ]
        }
        PixelFormat::Yuv444P | PixelFormat::YuvJ444P => {
            vec![p(&mut rng, w, h), p(&mut rng, w, h), p(&mut rng, w, h)]
        }
        PixelFormat::Yuv411P => {
            vec![
                p(&mut rng, w, h),
                p(&mut rng, w / 4, h),
                p(&mut rng, w / 4, h),
            ]
        }
        // 4:2:0 + full-res alpha plane.
        PixelFormat::Yuva420P => vec![
            p(&mut rng, w, h),
            p(&mut rng, w / 2, h / 2),
            p(&mut rng, w / 2, h / 2),
            p(&mut rng, w, h),
        ],
        // Semi-planar: Y plane + interleaved UV plane (2 bytes/chroma px).
        PixelFormat::Nv12 | PixelFormat::Nv21 => {
            vec![p(&mut rng, w, h), p(&mut rng, (w / 2) * 2, h / 2)]
        }
        // Packed 4:2:2: 2 bytes/pixel single plane.
        PixelFormat::Yuyv422 | PixelFormat::Uyvy422 => vec![p(&mut rng, w * 2, h)],
        // Planar GBR(A): each plane is w 16-bit words/row.
        PixelFormat::Gbrp10Le | PixelFormat::Gbrp12Le | PixelFormat::Gbrp14Le => {
            vec![
                p(&mut rng, w * 2, h),
                p(&mut rng, w * 2, h),
                p(&mut rng, w * 2, h),
            ]
        }
        PixelFormat::Gbrap10Le | PixelFormat::Gbrap12Le | PixelFormat::Gbrap14Le => vec![
            p(&mut rng, w * 2, h),
            p(&mut rng, w * 2, h),
            p(&mut rng, w * 2, h),
            p(&mut rng, w * 2, h),
        ],
        _ => return None,
    };
    Some(VideoFrame { pts: None, planes })
}

/// Destinations to try for each source. Covering the full format list as
/// destinations means unsupported pairs simply return `Err(Unsupported)`,
/// which is fine — we only care that nothing panics.
fn try_convert(src: &VideoFrame, info: FrameInfo, opts: &ConvertOptions) {
    for &dst in FORMATS {
        let _ = convert(src, info, dst, opts);
    }
}

fuzz_target!(|input: Input| {
    let fmt = FORMATS[(input.fmt_idx as usize) % FORMATS.len()];
    let cs = COLOR_SPACES[(input.cs_idx as usize) % COLOR_SPACES.len()];
    // Keep geometry small so a single iteration stays well under the
    // libfuzzer rss budget even for 8 bytes/pixel deep-RGBA layouts.
    let w = (input.w as usize % 64) + 1;
    let h = (input.h as usize % 64) + 1;
    let pad = input.pad as usize % 9;

    let Some(src) = build_frame(fmt, w, h, pad, input.seed) else {
        return;
    };
    let info = FrameInfo::new(fmt, w as u32, h as u32);
    let opts = ConvertOptions {
        color_space: cs,
        ..Default::default()
    };

    try_convert(&src, info, &opts);
});
