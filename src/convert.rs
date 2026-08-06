//! High-level `convert()` entry point.
//!
//! Every supported conversion flows through [`convert`], which dispatches
//! on `(src_info.format, dst_format)` to the appropriate helper in
//! [`crate::rgb`], [`crate::yuv`], [`crate::gray`], [`crate::palette`],
//! or [`crate::pal8`]. Anything that isn't wired up yet returns
//! `Error::Unsupported`.
//!
//! Stream-level properties (pixel format, width, height) live on the
//! caller's [`oxideav_core::CodecParameters`], not on the [`VideoFrame`]
//! itself, so every entry point takes them as an explicit
//! [`FrameInfo`] argument alongside the frame.

use oxideav_core::{Error, PixelFormat, Result, VideoFrame, VideoPlane};

use crate::cmyk;
use crate::gray;
use crate::pal8;
use crate::palette::Palette;
use crate::rgb;
use crate::yuv::{self, YuvMatrix};

/// Stream-level metadata that used to live on `VideoFrame`. Threaded
/// through every conversion so the helpers know how to interpret the
/// raw plane bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameInfo {
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
}

impl FrameInfo {
    pub const fn new(format: PixelFormat, width: u32, height: u32) -> Self {
        Self {
            format,
            width,
            height,
        }
    }
}

/// Dither strategy selected when down-quantising to a palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Dither {
    #[default]
    None,
    Bayer8x8,
    FloydSteinberg,
}

/// YUV / RGB matrix selection.
///
/// BT.2020 variants implement the non-constant-luminance Y'CbCr matrix
/// from ITU-R BT.2020-2 Table 4 (kr=0.2627, kb=0.0593). The same
/// coefficients are reused by ITU-R BT.2100-3 Table 6 for HDR video.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorSpace {
    #[default]
    Bt601Limited,
    Bt601Full,
    Bt709Limited,
    Bt709Full,
    Bt2020Limited,
    Bt2020Full,
}

/// Options bundle passed to [`convert`].
#[derive(Clone, Debug, Default)]
pub struct ConvertOptions {
    pub dither: Dither,
    pub palette: Option<Palette>,
    pub color_space: ColorSpace,
}

/// Return `Some(src)` when the caller's destination format already
/// matches the source's format — useful to skip a pointless clone in
/// hot paths.
pub fn convert_in_place_if_same(
    src: &VideoFrame,
    src_info: FrameInfo,
    dst_format: PixelFormat,
) -> Option<&VideoFrame> {
    if src_info.format == dst_format {
        Some(src)
    } else {
        None
    }
}

/// Convert `src` to `dst_format`, producing a newly allocated frame.
///
/// Dispatch first looks for a direct `(src, dst)` entry in the coverage
/// table. When none exists, a **single-pivot staged conversion** is
/// attempted: the frame is converted to one intermediate format and
/// then to the destination, with pivot candidates tried in a fixed
/// fidelity-aware order (YUV pivots first when both endpoints are YUV
/// carriage so no colour matrix enters the path; RGB pivots first
/// otherwise, alpha-capable before alpha-less, deep before 8-bit where
/// it matters). Staged paths are exactly as correct as their two legs —
/// but they can round twice and, when the pivot is 8-bit, quantise a
/// deeper source; callers that care can check [`supports_direct`].
///
/// # Per-plane significant-bits side-channel
///
/// A source frame may carry the per-plane significant-bits record
/// defined by [`oxideav_core::VideoFrame::significant_bits`] (one
/// LSB-anchored byte per image plane — e.g. `[12, 10, 10]` for 12-bit
/// luma with 10-bit chroma on a `Yuv444P12Le` or `Yuv444P16Le`
/// surface). `convert()` honours it with one policy:
///
/// - **Input**: every covered plane is treated as having exactly its
///   recorded significant depth. Before dispatch, marked planes are
///   *normalised* to the surface format's nominal depth by the
///   crate-wide MSB-replicating widen (full-scale at `b` bits maps to
///   full-scale at the nominal depth), so a record-carrying frame
///   converts exactly like the equivalent frame whose samples were
///   materialised at the nominal depth up front.
/// - **Validation**: a record value of `0`, or one *greater than the
///   surface format's nominal depth* (a significant-bits record can
///   only refine a format's depth downward), rejects with
///   `Error::Invalid`. Record bytes beyond the image-plane count are
///   ignored; a shorter record leaves the uncovered planes at the
///   nominal depth, per the core semantics.
/// - **Output**: converted frames are always produced at the
///   destination format's nominal depth and never carry a
///   significant-bits record — a stale record is never propagated. The
///   `src == dst` passthrough clone is the one exception: the frame is
///   untouched, so its record still describes it and rides along.
/// - **`Pal8`**: palette indices are identifiers, not magnitudes, so a
///   significant-bits record on a `Pal8` source is meaningless for
///   conversion and is ignored (the palette side-channel composes with
///   it and is honoured as usual).
pub fn convert(
    src: &VideoFrame,
    src_info: FrameInfo,
    dst_format: PixelFormat,
    opts: &ConvertOptions,
) -> Result<VideoFrame> {
    if src_info.format == dst_format {
        return Ok(src.clone());
    }
    // Materialise any significant-bits side-channel to the surface
    // format's nominal depth (see the rustdoc above) so every converter
    // below sees plain nominal-depth planes.
    let normalized = normalize_significant_bits(src, src_info)?;
    let src = normalized.as_ref().unwrap_or(src);
    if let Some(op) = lookup_any(src_info.format, dst_format) {
        return op.apply(src, src_info, opts);
    }
    if let Some((first, pivot, second)) = lookup_staged(src_info.format, dst_format) {
        let mid = first.apply(src, src_info, opts)?;
        let mid_info = FrameInfo::new(pivot, src_info.width, src_info.height);
        return second.apply(&mid, mid_info, opts);
    }
    Err(Error::unsupported(format!(
        "pixfmt: conversion {:?} → {:?} not implemented",
        src_info.format, dst_format
    )))
}

/// Widen an LSB-anchored `from`-bit value to `to` bits by repeating its
/// bit pattern into the freed low bits (full MSB replication). Zero
/// maps to zero, full-scale maps to full-scale, and the mapping is
/// strictly monotonic — the same rule as the crate's depth ladder,
/// generalised to widths below 8 bits (where the fill can be wider than
/// the source and the pattern repeats more than once).
fn widen_bits(v: u32, from: u32, to: u32) -> u32 {
    let mut out = v << (to - from);
    let mut fill = to - from;
    while fill > 0 {
        let take = fill.min(from);
        out |= (v >> (from - take)) << (fill - take);
        fill -= take;
    }
    out
}

/// Materialise a source frame's per-plane significant-bits side-channel
/// (see the policy on [`convert`]): validate the record against the
/// surface format and, when any covered plane is marked shallower than
/// the format's nominal depth, return a copy with those planes widened
/// (MSB replication) to the nominal depth and no side-channel records.
/// Returns `Ok(None)` when no work is needed (no record, an all-nominal
/// record, or a `Pal8` source, whose record is ignored).
fn normalize_significant_bits(src: &VideoFrame, src_info: FrameInfo) -> Result<Option<VideoFrame>> {
    let Some(record) = src.significant_bits() else {
        return Ok(None);
    };
    let fmt = src_info.format;
    // Palette indices are identifiers, not magnitudes — a record on a
    // Pal8 frame carries no meaning for conversion.
    if fmt == PixelFormat::Pal8 {
        return Ok(None);
    }
    let nominal = crate::format_info::FormatInfo::of(fmt).bit_depth as u32;
    let plane_count = src.image_plane_count();
    // Validate the covered planes; bytes beyond the image-plane count
    // are ignored (the record is defined per image plane).
    let mut needs_work = false;
    for (i, &b) in record.iter().take(plane_count).enumerate() {
        let b = b as u32;
        if b == 0 || b > nominal {
            return Err(Error::invalid(format!(
                "pixfmt: significant-bits record byte {i} = {b} out of range 1..={nominal} for {fmt:?}"
            )));
        }
        if b < nominal {
            needs_work = true;
        }
    }
    if !needs_work {
        return Ok(None);
    }
    // `needs_work` implies nominal > 1, so this is a byte-sample or
    // LE16-word format (Mono's nominal of 1 pins every record value to
    // 1). Widen each marked plane in place on a copy; padding bytes
    // beyond the tight row width are widened too, harmlessly — no
    // converter reads them.
    let wide_words = nominal > 8;
    let mut planes = Vec::with_capacity(plane_count);
    for (i, plane) in src.image_planes().iter().enumerate() {
        let b = record.get(i).map(|&b| b as u32).unwrap_or(nominal);
        let mut data = plane.data.clone();
        if b < nominal {
            if wide_words {
                let mask = (1u32 << b) - 1;
                for word in data.chunks_exact_mut(2) {
                    let v = u16::from_le_bytes([word[0], word[1]]) as u32 & mask;
                    let out = widen_bits(v, b, nominal) as u16;
                    word.copy_from_slice(&out.to_le_bytes());
                }
            } else {
                let mask = (1u32 << b) - 1;
                for byte in data.iter_mut() {
                    let v = *byte as u32 & mask;
                    *byte = widen_bits(v, b, 8) as u8;
                }
            }
        }
        planes.push(VideoPlane {
            stride: plane.stride,
            data,
        });
    }
    Ok(Some(VideoFrame {
        pts: src.pts,
        planes,
    }))
}

/// True when `convert()` can carry out `src → dst` — directly or via a
/// single staged pivot. `src == dst` is always supported (passthrough).
pub fn supports(src: PixelFormat, dst: PixelFormat) -> bool {
    src == dst || lookup_any(src, dst).is_some() || lookup_staged(src, dst).is_some()
}

/// True when a *direct* (single-step) conversion exists for `src → dst`
/// — either an explicit coverage-table entry or a computed planar-family
/// op — i.e. `convert()` will not stage through an intermediate format.
/// `src == dst` counts as direct.
pub fn supports_direct(src: PixelFormat, dst: PixelFormat) -> bool {
    src == dst || lookup_any(src, dst).is_some()
}

/// Pivot preference for staged conversions. YUV carriage pivots keep
/// YUV → YUV moves free of any colour matrix; the RGB list is ordered
/// alpha-capable first (so alpha survives whenever both endpoints carry
/// it) and deep before nothing-deeper-available. `Gray8` rescues the
/// mono / gray ladders.
const YUV_PIVOTS: &[PixelFormat] = &[
    PixelFormat::Yuv444P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv420P,
];
/// YUV pivot order used when either endpoint carries more than 8
/// significant bits: the 16-bit planar tier comes first so a deep
/// YUV → YUV staged route (e.g. `Yuv420P16Le → Yuv422P`) resamples
/// chroma at full 16-bit precision and only quantises at the final leg,
/// instead of truncating to 8 bits before the resample.
const YUV_PIVOTS_DEEP: &[PixelFormat] = &[
    PixelFormat::Yuv444P16Le,
    PixelFormat::Yuv422P16Le,
    PixelFormat::Yuv420P16Le,
    PixelFormat::Yuv444P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv420P,
];
const RGB_PIVOTS: &[PixelFormat] = &[
    PixelFormat::Rgba,
    PixelFormat::Rgb24,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray8,
];
/// RGB pivot order used when either endpoint carries more than 8
/// significant bits: the deep packed pivots come first so a
/// deep → deep staged route (e.g. `Gbrp10Le → Gbrp12Le`) never
/// quantises through an 8-bit intermediate when a lossless deep hop
/// exists.
const RGB_PIVOTS_DEEP: &[PixelFormat] = &[
    PixelFormat::Rgba64Le,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba,
    PixelFormat::Rgb24,
    PixelFormat::Gray8,
];

/// True for formats whose samples are YUV carriage (planar, semi-planar
/// or packed, any range or depth) — used to prefer matrix-free pivots.
fn is_yuv_carriage(f: PixelFormat) -> bool {
    use PixelFormat as P;
    matches!(
        f,
        P::Yuv420P
            | P::Yuv422P
            | P::Yuv444P
            | P::Yuv411P
            | P::Yuv420P10Le
            | P::Yuv422P10Le
            | P::Yuv444P10Le
            | P::Yuv420P12Le
            | P::Yuv422P12Le
            | P::Yuv444P12Le
            | P::Yuv420P16Le
            | P::Yuv422P16Le
            | P::Yuv444P16Le
            | P::YuvJ420P
            | P::YuvJ422P
            | P::YuvJ444P
            | P::Nv12
            | P::Nv21
            | P::Yuva420P
            | P::Yuva422P
            | P::Yuva444P
            | P::Yuva422P10Le
            | P::Yuva422P12Le
            | P::Yuva422P16Le
            | P::Yuva444P10Le
            | P::Yuva444P12Le
            | P::Yuva444P16Le
            | P::Yuva420P10Le
            | P::Yuva420P12Le
            | P::Yuva420P16Le
            | P::Yuyv422
            | P::Uyvy422
    )
}

/// Layout descriptor for the *uniform planar YUV(A) family*: 3 planes
/// (Y, U, V) — or 4 with a full-resolution alpha plane — at 4:2:0 /
/// 4:2:2 / 4:4:4 chroma siting, every sample either one byte
/// (`bits == 8`) or a 16-bit LE word carrying `bits` significant low
/// bits (10 / 12 / 16), limited-range carriage.
///
/// Formats matching this shape are handled by a *computed* dispatch
/// tier (see [`lookup_computed`]) that generates the conversion
/// parametrically instead of enumerating hundreds of table rows: any
/// ordered pair inside the family (depth move + chroma resample +
/// alpha drop / carry / synthesis fused in one step), plus
/// `Rgb24` / `Rgba` / `Gray8` interop. Full-range `YuvJ*`, 4:1:1,
/// semi-planar NV and packed 4:2:2 layouts are deliberately *not* part
/// of the family — their extra semantics (range rescale, 4× siting,
/// interleaving) stay on the explicit table rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlanarYuv {
    /// Chroma horizontal subsample factor (1 or 2).
    wsub: usize,
    /// Chroma vertical subsample factor (1 or 2).
    hsub: usize,
    /// Significant bits per sample: 8 (byte storage) or 10 / 12 / 16
    /// (16-bit LE word storage).
    bits: u32,
    /// Whether a full-resolution alpha plane trails as plane 3, stored
    /// at the same depth as the colour planes.
    alpha: bool,
}

impl PlanarYuv {
    /// Bytes per sample word (1 for 8-bit storage, 2 for LE-16).
    const fn sample_bytes(&self) -> usize {
        if self.bits > 8 {
            2
        } else {
            1
        }
    }
}

/// The [`PlanarYuv`] descriptor for `f`, or `None` when `f` is not a
/// member of the uniform planar family.
fn planar_yuv_desc(f: PixelFormat) -> Option<PlanarYuv> {
    use PixelFormat as P;
    let (wsub, hsub, bits, alpha) = match f {
        P::Yuv420P => (2, 2, 8, false),
        P::Yuv422P => (2, 1, 8, false),
        P::Yuv444P => (1, 1, 8, false),
        P::Yuv420P10Le => (2, 2, 10, false),
        P::Yuv422P10Le => (2, 1, 10, false),
        P::Yuv444P10Le => (1, 1, 10, false),
        P::Yuv420P12Le => (2, 2, 12, false),
        P::Yuv422P12Le => (2, 1, 12, false),
        P::Yuv444P12Le => (1, 1, 12, false),
        P::Yuv420P16Le => (2, 2, 16, false),
        P::Yuv422P16Le => (2, 1, 16, false),
        P::Yuv444P16Le => (1, 1, 16, false),
        P::Yuva420P => (2, 2, 8, true),
        P::Yuva422P => (2, 1, 8, true),
        P::Yuva444P => (1, 1, 8, true),
        P::Yuva422P10Le => (2, 1, 10, true),
        P::Yuva422P12Le => (2, 1, 12, true),
        P::Yuva422P16Le => (2, 1, 16, true),
        P::Yuva444P10Le => (1, 1, 10, true),
        P::Yuva444P12Le => (1, 1, 12, true),
        P::Yuva444P16Le => (1, 1, 16, true),
        P::Yuva420P10Le => (2, 2, 10, true),
        P::Yuva420P12Le => (2, 2, 12, true),
        P::Yuva420P16Le => (2, 2, 16, true),
        _ => return None,
    };
    Some(PlanarYuv {
        wsub,
        hsub,
        bits,
        alpha,
    })
}

/// Computed dispatch tier: conversions derived from the [`PlanarYuv`]
/// family descriptors rather than enumerated as table rows. Consulted
/// only when [`lookup`] has no explicit entry, so every pair the table
/// names keeps its exact historical behaviour. A computed op is still a
/// *direct* (single-step) conversion for [`supports_direct`] purposes.
fn lookup_computed(src: PixelFormat, dst: PixelFormat) -> Option<ConvertOp> {
    use PixelFormat as P;
    match (planar_yuv_desc(src), planar_yuv_desc(dst)) {
        // Anywhere-to-anywhere inside the planar family: depth move,
        // chroma resample and alpha handling fused into one step (the
        // chroma pair is resampled at the deeper of the two depths).
        (Some(s), Some(d)) => Some(ConvertOp::PlanarFamily { src: s, dst: d }),
        // Family member → packed RGB / Gray8.
        (Some(s), None) => match dst {
            P::Rgba => Some(ConvertOp::PlanarFamilyToRgb {
                src: s,
                alpha: true,
            }),
            P::Rgb24 => Some(ConvertOp::PlanarFamilyToRgb {
                src: s,
                alpha: false,
            }),
            P::Gray8 => Some(ConvertOp::PlanarFamilyToGray { src: s }),
            _ => None,
        },
        // Packed RGB / Gray8 → family member.
        (None, Some(d)) => match src {
            P::Rgba => Some(ConvertOp::RgbToPlanarFamily {
                dst: d,
                alpha_in: true,
            }),
            P::Rgb24 => Some(ConvertOp::RgbToPlanarFamily {
                dst: d,
                alpha_in: false,
            }),
            P::Gray8 => Some(ConvertOp::GrayToPlanarFamily { dst: d }),
            _ => None,
        },
        (None, None) => None,
    }
}

/// Single-step lookup across both dispatch tiers: the explicit coverage
/// table first (exact historical behaviour), then the computed planar
/// family tier.
fn lookup_any(src: PixelFormat, dst: PixelFormat) -> Option<ConvertOp> {
    lookup(src, dst)
        .copied()
        .or_else(|| lookup_computed(src, dst))
}

/// Find a single-pivot staged route `src → pivot → dst` where both legs
/// are direct (table or computed) ops. Returns the two ops and the
/// pivot format.
fn lookup_staged(
    src: PixelFormat,
    dst: PixelFormat,
) -> Option<(ConvertOp, PixelFormat, ConvertOp)> {
    let yuv_first = is_yuv_carriage(src) && is_yuv_carriage(dst);
    let deep = crate::format_info::FormatInfo::of(src).bit_depth > 8
        || crate::format_info::FormatInfo::of(dst).bit_depth > 8;
    let rgb_pivots = if deep { RGB_PIVOTS_DEEP } else { RGB_PIVOTS };
    let yuv_pivots = if deep { YUV_PIVOTS_DEEP } else { YUV_PIVOTS };
    let (a, b) = if yuv_first {
        (yuv_pivots, rgb_pivots)
    } else {
        (rgb_pivots, yuv_pivots)
    };
    for &pivot in a.iter().chain(b.iter()) {
        if pivot == src || pivot == dst {
            continue;
        }
        if let (Some(first), Some(second)) = (lookup_any(src, pivot), lookup_any(pivot, dst)) {
            return Some((first, pivot, second));
        }
    }
    None
}

/// Coverage table — one entry per supported `(src, dst)` pair. The
/// associated [`ConvertOp`] captures any variant-specific parameters
/// (RGB byte positions, chroma subsampling, range direction, …) so the
/// dispatch below is a short match over ~20 arms instead of a 68-arm
/// cross-product.
#[rustfmt::skip]
const TABLE: &[(PixelFormat, PixelFormat, ConvertOp)] = {
    use ConvertOp::*;
    use PixelFormat as P;
    &[
        // RGB family: all-to-all packed swizzles.
        (P::Rgb24, P::Bgr24, Swizzle3 { src: rgb::RGB_POS, dst: rgb::BGR_POS }),
        (P::Bgr24, P::Rgb24, Swizzle3 { src: rgb::BGR_POS, dst: rgb::RGB_POS }),
        (P::Rgba, P::Bgra, Swizzle4 { src: rgb::RGBA_POS, dst: rgb::BGRA_POS }),
        (P::Bgra, P::Rgba, Swizzle4 { src: rgb::BGRA_POS, dst: rgb::RGBA_POS }),
        (P::Rgba, P::Argb, Swizzle4 { src: rgb::RGBA_POS, dst: rgb::ARGB_POS }),
        (P::Argb, P::Rgba, Swizzle4 { src: rgb::ARGB_POS, dst: rgb::RGBA_POS }),
        (P::Rgba, P::Abgr, Swizzle4 { src: rgb::RGBA_POS, dst: rgb::ABGR_POS }),
        (P::Abgr, P::Rgba, Swizzle4 { src: rgb::ABGR_POS, dst: rgb::RGBA_POS }),
        (P::Bgra, P::Argb, Swizzle4 { src: rgb::BGRA_POS, dst: rgb::ARGB_POS }),
        (P::Argb, P::Bgra, Swizzle4 { src: rgb::ARGB_POS, dst: rgb::BGRA_POS }),
        (P::Bgra, P::Abgr, Swizzle4 { src: rgb::BGRA_POS, dst: rgb::ABGR_POS }),
        (P::Abgr, P::Bgra, Swizzle4 { src: rgb::ABGR_POS, dst: rgb::BGRA_POS }),
        (P::Argb, P::Abgr, Swizzle4 { src: rgb::ARGB_POS, dst: rgb::ABGR_POS }),
        (P::Abgr, P::Argb, Swizzle4 { src: rgb::ABGR_POS, dst: rgb::ARGB_POS }),

        // 3 ↔ 4 promote (append opaque alpha) / demote (drop alpha).
        (P::Rgb24, P::Rgba, Promote3To4 { src: rgb::RGB_POS, dst: rgb::RGBA_POS }),
        (P::Rgb24, P::Bgra, Promote3To4 { src: rgb::RGB_POS, dst: rgb::BGRA_POS }),
        (P::Rgb24, P::Argb, Promote3To4 { src: rgb::RGB_POS, dst: rgb::ARGB_POS }),
        (P::Rgb24, P::Abgr, Promote3To4 { src: rgb::RGB_POS, dst: rgb::ABGR_POS }),
        (P::Bgr24, P::Rgba, Promote3To4 { src: rgb::BGR_POS, dst: rgb::RGBA_POS }),
        (P::Bgr24, P::Bgra, Promote3To4 { src: rgb::BGR_POS, dst: rgb::BGRA_POS }),
        (P::Bgr24, P::Argb, Promote3To4 { src: rgb::BGR_POS, dst: rgb::ARGB_POS }),
        (P::Bgr24, P::Abgr, Promote3To4 { src: rgb::BGR_POS, dst: rgb::ABGR_POS }),
        (P::Rgba, P::Rgb24, Demote4To3 { src: rgb::RGBA_POS, dst: rgb::RGB_POS }),
        (P::Rgba, P::Bgr24, Demote4To3 { src: rgb::RGBA_POS, dst: rgb::BGR_POS }),
        (P::Bgra, P::Rgb24, Demote4To3 { src: rgb::BGRA_POS, dst: rgb::RGB_POS }),
        (P::Bgra, P::Bgr24, Demote4To3 { src: rgb::BGRA_POS, dst: rgb::BGR_POS }),
        (P::Argb, P::Rgb24, Demote4To3 { src: rgb::ARGB_POS, dst: rgb::RGB_POS }),
        (P::Argb, P::Bgr24, Demote4To3 { src: rgb::ARGB_POS, dst: rgb::BGR_POS }),
        (P::Abgr, P::Rgb24, Demote4To3 { src: rgb::ABGR_POS, dst: rgb::RGB_POS }),
        (P::Abgr, P::Bgr24, Demote4To3 { src: rgb::ABGR_POS, dst: rgb::BGR_POS }),

        // Deeper packed RGB ↔ 8-bit.
        (P::Rgb48Le, P::Rgb24, Rgb48ToRgb24),
        (P::Rgb24, P::Rgb48Le, Rgb24ToRgb48),
        (P::Rgba64Le, P::Rgba, Rgba64ToRgba),
        (P::Rgba, P::Rgba64Le, RgbaToRgba64),

        // Gray ↔ RGB / Gray16 / Mono. A gray broadcast writes the same
        // value to every colour channel, so the Rgb24 emitter serves
        // Bgr24 verbatim and the Rgba emitter serves Bgra; the
        // alpha-first orders (Argb / Abgr) shift the opaque byte to the
        // front via `alpha_first`.
        (P::Gray8, P::Rgb24, Gray8ToPacked3),
        (P::Gray8, P::Bgr24, Gray8ToPacked3),
        (P::Gray8, P::Rgba, Gray8ToPacked4 { alpha_first: false }),
        (P::Gray8, P::Bgra, Gray8ToPacked4 { alpha_first: false }),
        (P::Gray8, P::Argb, Gray8ToPacked4 { alpha_first: true }),
        (P::Gray8, P::Abgr, Gray8ToPacked4 { alpha_first: true }),
        (P::Gray16Le, P::Gray8, Gray16ToGray8),
        (P::Gray8, P::Gray16Le, Gray8ToGray16),
        (P::MonoBlack, P::Gray8, MonoToGray { black_is_zero: true }),
        (P::MonoWhite, P::Gray8, MonoToGray { black_is_zero: false }),
        (P::Gray8, P::MonoBlack, GrayToMono { black_is_zero: true }),
        (P::Gray8, P::MonoWhite, GrayToMono { black_is_zero: false }),

        // Packed RGB → Gray8 luminance projection. Uses the Y' row of
        // the selected primaries at full range (Gray8 is a full-range
        // space); r = g = b inputs map to themselves exactly, so the
        // Gray8 → RGB broadcast round-trips gray content.
        (P::Rgb24, P::Gray8, RgbToGray { alpha_in: false }),
        (P::Rgba,  P::Gray8, RgbToGray { alpha_in: true }),

        // Ya8 (grey + alpha, 2 bytes/pixel) ↔ Gray8 / Rgb24 / Rgba.
        // Promote/demote helpers for icon / glyph / single-channel-mask
        // workflows that need to carry an alpha plane through the
        // pipeline without going through a full YUVA path.
        (P::Ya8,   P::Gray8, Ya8ToGray8),
        (P::Gray8, P::Ya8,   Gray8ToYa8),
        (P::Ya8,   P::Rgb24, Ya8ToRgb24),
        (P::Ya8,   P::Rgba,  Ya8ToRgba),
        (P::Rgb24, P::Ya8,   Rgb24ToYa8),
        (P::Rgba,  P::Ya8,   RgbaToYa8),

        // Ya16Le (packed 16-bit grey + alpha, core 0.1.34) — the deep
        // companion to Ya8. Both words are full-scale LE16 (the
        // Gray16Le convention). Depth moves follow the ladder rules:
        // high-byte truncation down, exact ×257 widen up (so the 8-bit
        // round-trips are lossless); the Gray16Le pair is bit-exact in
        // luma; the Rgba64Le pair is bit-exact out (broadcast + carry)
        // and recovers grey-on-alpha content exactly on the way back
        // (rounded-mean luma derivation, the Ya8 convention at depth).
        (P::Ya16Le,   P::Ya8,      Ya16ToYa8),
        (P::Ya8,      P::Ya16Le,   Ya8ToYa16),
        (P::Ya16Le,   P::Gray16Le, Ya16ToGray16),
        (P::Gray16Le, P::Ya16Le,   Gray16ToYa16),
        (P::Ya16Le,   P::Gray8,    Ya16ToGray8),
        (P::Gray8,    P::Ya16Le,   Gray8ToYa16),
        (P::Ya16Le,   P::Rgba64Le, Ya16ToRgba64),
        (P::Rgba64Le, P::Ya16Le,   Rgba64ToYa16),
        (P::Ya16Le,   P::Rgba,     Ya16ToPacked8 { alpha: true }),
        (P::Ya16Le,   P::Rgb24,    Ya16ToPacked8 { alpha: false }),
        (P::Rgba,     P::Ya16Le,   Packed8ToYa16 { alpha_in: true }),
        (P::Rgb24,    P::Ya16Le,   Packed8ToYa16 { alpha_in: false }),

        // YUV family → Gray8 (luma extraction) and Gray8 → YUV family
        // (neutral chroma synthesis). Both directions are pure luma-plane
        // operations: chroma is dropped on the way out and written as the
        // neutral code 128 on the way in, so no colour matrix applies —
        // only the range rescale between the limited-range `Yuv*` /
        // `Nv*` / `Yuva*` families (16..=235 luma) and the full-range
        // `Gray8` / `YuvJ*` sample space. The `Yuva*` → Gray8 rows also
        // drop the alpha plane.
        (P::Yuv420P,  P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Yuv422P,  P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Yuv444P,  P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Yuv411P,  P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Nv12,     P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Nv21,     P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Yuva420P, P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Yuva422P, P::Gray8, YuvLumaToGray { full_range: false }),
        (P::Yuva444P, P::Gray8, YuvLumaToGray { full_range: false }),
        (P::YuvJ420P, P::Gray8, YuvLumaToGray { full_range: true }),
        (P::YuvJ422P, P::Gray8, YuvLumaToGray { full_range: true }),
        (P::YuvJ444P, P::Gray8, YuvLumaToGray { full_range: true }),
        (P::Gray8, P::Yuv420P,  GrayToYuvPlanar { wsub: 2, hsub: 2, full_range: false }),
        (P::Gray8, P::Yuv422P,  GrayToYuvPlanar { wsub: 2, hsub: 1, full_range: false }),
        (P::Gray8, P::Yuv444P,  GrayToYuvPlanar { wsub: 1, hsub: 1, full_range: false }),
        (P::Gray8, P::Yuv411P,  GrayToYuvPlanar { wsub: 4, hsub: 1, full_range: false }),
        (P::Gray8, P::YuvJ420P, GrayToYuvPlanar { wsub: 2, hsub: 2, full_range: true }),
        (P::Gray8, P::YuvJ422P, GrayToYuvPlanar { wsub: 2, hsub: 1, full_range: true }),
        (P::Gray8, P::YuvJ444P, GrayToYuvPlanar { wsub: 1, hsub: 1, full_range: true }),
        // The NV interleaved chroma plane is all-neutral (128, 128), so
        // NV12 and NV21 receive byte-identical output.
        (P::Gray8, P::Nv12, GrayToNv),
        (P::Gray8, P::Nv21, GrayToNv),

        // YUV planar → packed RGB.
        (P::Yuv420P, P::Rgb24, YuvToRgb { wsub: 2, hsub: 2, alpha: false, full_range: false }),
        (P::Yuv422P, P::Rgb24, YuvToRgb { wsub: 2, hsub: 1, alpha: false, full_range: false }),
        (P::Yuv444P, P::Rgb24, YuvToRgb { wsub: 1, hsub: 1, alpha: false, full_range: false }),
        (P::Yuv420P, P::Rgba,  YuvToRgb { wsub: 2, hsub: 2, alpha: true, full_range: false }),
        (P::Yuv422P, P::Rgba,  YuvToRgb { wsub: 2, hsub: 1, alpha: true, full_range: false }),
        (P::Yuv444P, P::Rgba,  YuvToRgb { wsub: 1, hsub: 1, alpha: true, full_range: false }),

        // Packed RGB → YUV planar.
        (P::Rgb24, P::Yuv420P, RgbToYuv { wsub: 2, hsub: 2, alpha_in: false, full_range: false }),
        (P::Rgb24, P::Yuv422P, RgbToYuv { wsub: 2, hsub: 1, alpha_in: false, full_range: false }),
        (P::Rgb24, P::Yuv444P, RgbToYuv { wsub: 1, hsub: 1, alpha_in: false, full_range: false }),
        (P::Rgba,  P::Yuv420P, RgbToYuv { wsub: 2, hsub: 2, alpha_in: true, full_range: false }),
        (P::Rgba,  P::Yuv422P, RgbToYuv { wsub: 2, hsub: 1, alpha_in: true, full_range: false }),
        (P::Rgba,  P::Yuv444P, RgbToYuv { wsub: 1, hsub: 1, alpha_in: true, full_range: false }),

        // Full-range "J" YUV planar ↔ packed RGB, direct (no staging
        // through the limited-range sibling). The YuvJ* families carry
        // full-range samples by definition, so these rows pin the matrix
        // to full range regardless of the `ColorSpace` range half —
        // `opts.color_space` still selects the primaries (601/709/2020).
        (P::YuvJ420P, P::Rgb24, YuvToRgb { wsub: 2, hsub: 2, alpha: false, full_range: true }),
        (P::YuvJ422P, P::Rgb24, YuvToRgb { wsub: 2, hsub: 1, alpha: false, full_range: true }),
        (P::YuvJ444P, P::Rgb24, YuvToRgb { wsub: 1, hsub: 1, alpha: false, full_range: true }),
        (P::YuvJ420P, P::Rgba,  YuvToRgb { wsub: 2, hsub: 2, alpha: true, full_range: true }),
        (P::YuvJ422P, P::Rgba,  YuvToRgb { wsub: 2, hsub: 1, alpha: true, full_range: true }),
        (P::YuvJ444P, P::Rgba,  YuvToRgb { wsub: 1, hsub: 1, alpha: true, full_range: true }),
        (P::Rgb24, P::YuvJ420P, RgbToYuv { wsub: 2, hsub: 2, alpha_in: false, full_range: true }),
        (P::Rgb24, P::YuvJ422P, RgbToYuv { wsub: 2, hsub: 1, alpha_in: false, full_range: true }),
        (P::Rgb24, P::YuvJ444P, RgbToYuv { wsub: 1, hsub: 1, alpha_in: false, full_range: true }),
        (P::Rgba,  P::YuvJ420P, RgbToYuv { wsub: 2, hsub: 2, alpha_in: true, full_range: true }),
        (P::Rgba,  P::YuvJ422P, RgbToYuv { wsub: 2, hsub: 1, alpha_in: true, full_range: true }),
        (P::Rgba,  P::YuvJ444P, RgbToYuv { wsub: 1, hsub: 1, alpha_in: true, full_range: true }),

        // YuvJ* ↔ Yuv* (range rescale only — same planar layout).
        (P::YuvJ420P, P::Yuv420P, RescaleRange { wsub: 2, hsub: 2, to_full: false }),
        (P::YuvJ422P, P::Yuv422P, RescaleRange { wsub: 2, hsub: 1, to_full: false }),
        (P::YuvJ444P, P::Yuv444P, RescaleRange { wsub: 1, hsub: 1, to_full: false }),
        (P::Yuv420P, P::YuvJ420P, RescaleRange { wsub: 2, hsub: 2, to_full: true }),
        (P::Yuv422P, P::YuvJ422P, RescaleRange { wsub: 2, hsub: 1, to_full: true }),
        (P::Yuv444P, P::YuvJ444P, RescaleRange { wsub: 1, hsub: 1, to_full: true }),

        // NV12 / NV21 ↔ Yuv420P.
        (P::Nv12, P::Yuv420P, NvToYuv420p { is_nv12: true }),
        (P::Nv21, P::Yuv420P, NvToYuv420p { is_nv12: false }),
        (P::Yuv420P, P::Nv12, Yuv420pToNv { is_nv12: true }),
        (P::Yuv420P, P::Nv21, Yuv420pToNv { is_nv12: false }),
        // NV12 / NV21 ↔ RGB direct. Fused path: walk the interleaved
        // UV plane once per row pair, split into a transient (U, V)
        // plane pair, then run the proven planar 4:2:0 → RGB decoder.
        // Avoids the caller having to stage through Yuv420P.
        (P::Nv12, P::Rgb24, NvToRgb { is_nv12: true, alpha: false }),
        (P::Nv21, P::Rgb24, NvToRgb { is_nv12: false, alpha: false }),
        (P::Nv12, P::Rgba,  NvToRgb { is_nv12: true, alpha: true }),
        (P::Nv21, P::Rgba,  NvToRgb { is_nv12: false, alpha: true }),
        (P::Rgb24, P::Nv12, RgbToNv { is_nv12: true, alpha_in: false }),
        (P::Rgb24, P::Nv21, RgbToNv { is_nv12: false, alpha_in: false }),
        (P::Rgba,  P::Nv12, RgbToNv { is_nv12: true, alpha_in: true }),
        (P::Rgba,  P::Nv21, RgbToNv { is_nv12: false, alpha_in: true }),

        // Direct planar YUV ↔ planar YUV (chroma resample only — luma
        // copied byte-for-byte). Limited-range BT.x carriage where the
        // YUV ↔ YUV step never visits RGB. Six entries cover every
        // ordered pair on (4:2:0, 4:2:2, 4:4:4).
        (P::Yuv420P, P::Yuv422P, ChromaResample { src_wsub: 2, src_hsub: 2, dst_wsub: 2, dst_hsub: 1 }),
        (P::Yuv420P, P::Yuv444P, ChromaResample { src_wsub: 2, src_hsub: 2, dst_wsub: 1, dst_hsub: 1 }),
        (P::Yuv422P, P::Yuv420P, ChromaResample { src_wsub: 2, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::Yuv422P, P::Yuv444P, ChromaResample { src_wsub: 2, src_hsub: 1, dst_wsub: 1, dst_hsub: 1 }),
        (P::Yuv444P, P::Yuv420P, ChromaResample { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::Yuv444P, P::Yuv422P, ChromaResample { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 1 }),

        // Same six pairs on the full-range "J" planar family. The
        // matrix is the same as the limited-range op (no luma/chroma
        // rescale enters here — only chroma subsampling changes), so
        // these reuse `ChromaResample` directly.
        (P::YuvJ420P, P::YuvJ422P, ChromaResample { src_wsub: 2, src_hsub: 2, dst_wsub: 2, dst_hsub: 1 }),
        (P::YuvJ420P, P::YuvJ444P, ChromaResample { src_wsub: 2, src_hsub: 2, dst_wsub: 1, dst_hsub: 1 }),
        (P::YuvJ422P, P::YuvJ420P, ChromaResample { src_wsub: 2, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::YuvJ422P, P::YuvJ444P, ChromaResample { src_wsub: 2, src_hsub: 1, dst_wsub: 1, dst_hsub: 1 }),
        (P::YuvJ444P, P::YuvJ420P, ChromaResample { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::YuvJ444P, P::YuvJ422P, ChromaResample { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 1 }),

        // Packed 4:2:2 (YUYV / UYVY) ↔ planar Yuv422P. Pure deinterleave
        // / interleave; no colour math.
        (P::Yuyv422, P::Yuv422P, Packed422ToYuv422p { is_yuyv: true }),
        (P::Uyvy422, P::Yuv422P, Packed422ToYuv422p { is_yuyv: false }),
        (P::Yuv422P, P::Yuyv422, Yuv422pToPacked422 { is_yuyv: true }),
        (P::Yuv422P, P::Uyvy422, Yuv422pToPacked422 { is_yuyv: false }),
        // YUYV ↔ UYVY: two byte-swaps per quad.
        (P::Yuyv422, P::Uyvy422, Packed422Swap),
        (P::Uyvy422, P::Yuyv422, Packed422Swap),
        // Packed 4:2:2 ↔ RGB. Fused path that walks the packed buffer
        // once per row, deinterleaves on the fly, and feeds the result
        // into the same colour math used by the planar 4:2:2 route.
        (P::Yuyv422, P::Rgb24, Packed422ToRgb { is_yuyv: true, alpha: false }),
        (P::Uyvy422, P::Rgb24, Packed422ToRgb { is_yuyv: false, alpha: false }),
        (P::Yuyv422, P::Rgba,  Packed422ToRgb { is_yuyv: true, alpha: true }),
        (P::Uyvy422, P::Rgba,  Packed422ToRgb { is_yuyv: false, alpha: true }),
        (P::Rgb24, P::Yuyv422, RgbToPacked422 { is_yuyv: true, alpha_in: false }),
        (P::Rgb24, P::Uyvy422, RgbToPacked422 { is_yuyv: false, alpha_in: false }),
        (P::Rgba,  P::Yuyv422, RgbToPacked422 { is_yuyv: true, alpha_in: true }),
        (P::Rgba,  P::Uyvy422, RgbToPacked422 { is_yuyv: false, alpha_in: true }),

        // Palette.
        (P::Pal8, P::Rgb24, Pal8ToRgb { alpha: false }),
        (P::Pal8, P::Rgba,  Pal8ToRgb { alpha: true }),
        (P::Rgb24, P::Pal8, RgbToPal8 { alpha_in: false }),
        (P::Rgba,  P::Pal8, RgbToPal8 { alpha_in: true }),

        // CMYK ↔ RGB. Uncalibrated device-CMYK approximation; pure
        // bit-manipulation (no matrix / ColorSpace knob applies).
        (P::Cmyk,  P::Rgb24, CmykToRgb { alpha: false, inverted: false }),
        (P::Cmyk,  P::Rgba,  CmykToRgb { alpha: true, inverted: false }),
        (P::Rgb24, P::Cmyk,  RgbToCmyk { alpha_in: false, inverted: false }),
        (P::Rgba,  P::Cmyk,  RgbToCmyk { alpha_in: true, inverted: false }),
        // CmykInverted (core 0.1.34): the inverted-ink convention
        // (stored byte = 255 − ink). RGB interop folds the complement
        // into the formula — byte-identical to complement-then-regular;
        // the Cmyk ↔ CmykInverted pair is the exact self-inverse
        // per-byte complement (lossless both ways).
        (P::CmykInverted, P::Rgb24, CmykToRgb { alpha: false, inverted: true }),
        (P::CmykInverted, P::Rgba,  CmykToRgb { alpha: true, inverted: true }),
        (P::Rgb24, P::CmykInverted, RgbToCmyk { alpha_in: false, inverted: true }),
        (P::Rgba,  P::CmykInverted, RgbToCmyk { alpha_in: true, inverted: true }),
        (P::Cmyk,  P::CmykInverted, CmykComplement),
        (P::CmykInverted, P::Cmyk,  CmykComplement),

        // Yuv411P (4:1:1 planar — luma full res, chroma horizontally
        // subsampled by 4). Native NTSC DV-25 layout and a legal JPEG
        // sampling pattern (`cjpeg -sample 4x1`). Six chroma-resample
        // pairs (411 ↔ 420 / 422 / 444 in both directions) plus
        // RGB encode/decode under any `ColorSpace`. Width must be a
        // multiple of 4; odd-by-4 widths reject with `Error::Invalid`.
        (P::Yuv411P, P::Yuv444P, ChromaResample { src_wsub: 4, src_hsub: 1, dst_wsub: 1, dst_hsub: 1 }),
        (P::Yuv411P, P::Yuv422P, ChromaResample { src_wsub: 4, src_hsub: 1, dst_wsub: 2, dst_hsub: 1 }),
        (P::Yuv411P, P::Yuv420P, ChromaResample { src_wsub: 4, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::Yuv444P, P::Yuv411P, ChromaResample { src_wsub: 1, src_hsub: 1, dst_wsub: 4, dst_hsub: 1 }),
        (P::Yuv422P, P::Yuv411P, ChromaResample { src_wsub: 2, src_hsub: 1, dst_wsub: 4, dst_hsub: 1 }),
        (P::Yuv420P, P::Yuv411P, ChromaResample { src_wsub: 2, src_hsub: 2, dst_wsub: 4, dst_hsub: 1 }),
        // Yuv411P ↔ RGB. The 4:1:1 ↔ 4:4:4 chroma step stages through
        // a transient full-resolution chroma pair before calling the
        // proven scalar 4:4:4 ↔ RGB matrix; no new colour math.
        (P::Yuv411P, P::Rgb24, YuvToRgb { wsub: 4, hsub: 1, alpha: false, full_range: false }),
        (P::Yuv411P, P::Rgba,  YuvToRgb { wsub: 4, hsub: 1, alpha: true, full_range: false }),
        (P::Rgb24,   P::Yuv411P, RgbToYuv { wsub: 4, hsub: 1, alpha_in: false, full_range: false }),
        (P::Rgba,    P::Yuv411P, RgbToYuv { wsub: 4, hsub: 1, alpha_in: true, full_range: false }),

        // Yuva420P / Yuva422P / Yuva444P (planar YUV + a full-resolution
        // alpha plane appended as plane 3 — the alpha plane is `w × h`
        // regardless of the chroma subsampling). The YUV planes match
        // the alpha-less sibling byte-for-byte; conversions reuse the
        // proven planar YUV ↔ RGB paths and either drop, synthesise
        // opaque, or carry the alpha plane through unchanged.
        (P::Yuv420P,  P::Yuva420P, YuvToYuva { wsub: 2, hsub: 2 }),
        (P::Yuv422P,  P::Yuva422P, YuvToYuva { wsub: 2, hsub: 1 }),
        (P::Yuv444P,  P::Yuva444P, YuvToYuva { wsub: 1, hsub: 1 }),
        (P::Yuva420P, P::Yuv420P,  YuvaToYuv { wsub: 2, hsub: 2 }),
        (P::Yuva422P, P::Yuv422P,  YuvaToYuv { wsub: 2, hsub: 1 }),
        (P::Yuva444P, P::Yuv444P,  YuvaToYuv { wsub: 1, hsub: 1 }),
        (P::Yuva420P, P::Rgb24,    YuvaToRgb { wsub: 2, hsub: 2, alpha: false }),
        (P::Yuva422P, P::Rgb24,    YuvaToRgb { wsub: 2, hsub: 1, alpha: false }),
        (P::Yuva444P, P::Rgb24,    YuvaToRgb { wsub: 1, hsub: 1, alpha: false }),
        (P::Yuva420P, P::Rgba,     YuvaToRgb { wsub: 2, hsub: 2, alpha: true }),
        (P::Yuva422P, P::Rgba,     YuvaToRgb { wsub: 2, hsub: 1, alpha: true }),
        (P::Yuva444P, P::Rgba,     YuvaToRgb { wsub: 1, hsub: 1, alpha: true }),
        (P::Rgb24,    P::Yuva420P, RgbToYuva { wsub: 2, hsub: 2, alpha_in: false }),
        (P::Rgb24,    P::Yuva422P, RgbToYuva { wsub: 2, hsub: 1, alpha_in: false }),
        (P::Rgb24,    P::Yuva444P, RgbToYuva { wsub: 1, hsub: 1, alpha_in: false }),
        (P::Rgba,     P::Yuva420P, RgbToYuva { wsub: 2, hsub: 2, alpha_in: true }),
        (P::Rgba,     P::Yuva422P, RgbToYuva { wsub: 2, hsub: 1, alpha_in: true }),
        (P::Rgba,     P::Yuva444P, RgbToYuva { wsub: 1, hsub: 1, alpha_in: true }),
        // Alpha-preserving moves inside the Yuva family: luma and the
        // full-resolution alpha plane are copied byte-for-byte, only
        // the chroma pair is resampled (same primitives and rounding
        // as the alpha-less `ChromaResample` rows — no colour matrix,
        // alpha bit-exact).
        (P::Yuva420P, P::Yuva422P, YuvaChromaResample { src_wsub: 2, src_hsub: 2, dst_wsub: 2, dst_hsub: 1 }),
        (P::Yuva420P, P::Yuva444P, YuvaChromaResample { src_wsub: 2, src_hsub: 2, dst_wsub: 1, dst_hsub: 1 }),
        (P::Yuva422P, P::Yuva420P, YuvaChromaResample { src_wsub: 2, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::Yuva422P, P::Yuva444P, YuvaChromaResample { src_wsub: 2, src_hsub: 1, dst_wsub: 1, dst_hsub: 1 }),
        (P::Yuva444P, P::Yuva420P, YuvaChromaResample { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::Yuva444P, P::Yuva422P, YuvaChromaResample { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 1 }),

        // Planar GBR(A) ↔ packed deep RGB. GBR stores RGB as three (or
        // four with alpha) planes in G, B, R order, each sample a 16-bit
        // LE word carrying `bits` significant low bits (per the
        // oxideav-core `Gbrp*Le` / `Gbrap*Le` variant docs). The packed
        // `Rgb48Le` / `Rgba64Le` targets carry 16 significant bits in
        // R, G, B(, A) byte order. The conversion is a pure
        // deinterleave/interleave plus a `16 - bits` left-shift toward
        // the 16-bit container (and the reverse right-shift on the way
        // back) — no colour matrix. No `ColorSpace` knob applies.
        (P::Gbrp10Le,  P::Rgb48Le,  GbrToPackedDeep { bits: 10, alpha_in: false, alpha_out: false }),
        (P::Gbrp12Le,  P::Rgb48Le,  GbrToPackedDeep { bits: 12, alpha_in: false, alpha_out: false }),
        (P::Gbrp14Le,  P::Rgb48Le,  GbrToPackedDeep { bits: 14, alpha_in: false, alpha_out: false }),
        (P::Gbrap10Le, P::Rgba64Le, GbrToPackedDeep { bits: 10, alpha_in: true, alpha_out: true }),
        (P::Gbrap12Le, P::Rgba64Le, GbrToPackedDeep { bits: 12, alpha_in: true, alpha_out: true }),
        (P::Gbrap14Le, P::Rgba64Le, GbrToPackedDeep { bits: 14, alpha_in: true, alpha_out: true }),
        // Planar GBR(A) ↔ 8-bit packed RGB(A). Same plane reorder as the
        // deep rows with a bit-depth step folded in: narrowing keeps the
        // top 8 of the `bits` significant bits (truncation, matching the
        // depth ladder), widening replicates MSBs so 8-bit round-trips
        // are exact and peak maps to peak. These rows also make the GBR
        // families reachable from the whole 8-bit ecosystem (YUV, gray,
        // palette, …) through the staged fallback's Rgb24/Rgba pivots.
        (P::Gbrp10Le,  P::Rgb24, GbrToPacked8 { bits: 10, alpha: false }),
        (P::Gbrp12Le,  P::Rgb24, GbrToPacked8 { bits: 12, alpha: false }),
        (P::Gbrp14Le,  P::Rgb24, GbrToPacked8 { bits: 14, alpha: false }),
        (P::Gbrap10Le, P::Rgba,  GbrToPacked8 { bits: 10, alpha: true }),
        (P::Gbrap12Le, P::Rgba,  GbrToPacked8 { bits: 12, alpha: true }),
        (P::Gbrap14Le, P::Rgba,  GbrToPacked8 { bits: 14, alpha: true }),
        (P::Rgb24, P::Gbrp10Le,  Packed8ToGbr { bits: 10, alpha: false }),
        (P::Rgb24, P::Gbrp12Le,  Packed8ToGbr { bits: 12, alpha: false }),
        (P::Rgb24, P::Gbrp14Le,  Packed8ToGbr { bits: 14, alpha: false }),
        (P::Rgba,  P::Gbrap10Le, Packed8ToGbr { bits: 10, alpha: true }),
        (P::Rgba,  P::Gbrap12Le, Packed8ToGbr { bits: 12, alpha: true }),
        (P::Rgba,  P::Gbrap14Le, Packed8ToGbr { bits: 14, alpha: true }),

        (P::Rgb48Le,  P::Gbrp10Le,  PackedDeepToGbr { bits: 10, alpha_in: false, alpha_out: false }),
        (P::Rgb48Le,  P::Gbrp12Le,  PackedDeepToGbr { bits: 12, alpha_in: false, alpha_out: false }),
        (P::Rgb48Le,  P::Gbrp14Le,  PackedDeepToGbr { bits: 14, alpha_in: false, alpha_out: false }),
        (P::Rgba64Le, P::Gbrap10Le, PackedDeepToGbr { bits: 10, alpha_in: true, alpha_out: true }),
        (P::Rgba64Le, P::Gbrap12Le, PackedDeepToGbr { bits: 12, alpha_in: true, alpha_out: true }),
        (P::Rgba64Le, P::Gbrap14Le, PackedDeepToGbr { bits: 14, alpha_in: true, alpha_out: true }),

        // GBR(A) depth-ladder ends (core 0.1.33). The 16-bit members
        // reuse the shared GBR ops with `bits = 16`: the deep-packed
        // hop degenerates to a pure plane reorder (shift by 0 — exact
        // both ways), the 8-bit hops are the exact x257 widen /
        // top-byte truncation. `Gbrp8` stores byte samples, so it gets
        // dedicated byte-plane ops: Rgb24 interop is a zero-math plane
        // reorder (bit-exact, self-inverse), Rgb48Le interop widens
        // x257 / keeps the top byte, mirroring Rgb24 ↔ Rgb48Le.
        (P::Gbrp16Le,  P::Rgb48Le,  GbrToPackedDeep { bits: 16, alpha_in: false, alpha_out: false }),
        (P::Gbrap16Le, P::Rgba64Le, GbrToPackedDeep { bits: 16, alpha_in: true, alpha_out: true }),
        (P::Rgb48Le,  P::Gbrp16Le,  PackedDeepToGbr { bits: 16, alpha_in: false, alpha_out: false }),
        (P::Rgba64Le, P::Gbrap16Le, PackedDeepToGbr { bits: 16, alpha_in: true, alpha_out: true }),
        (P::Gbrp16Le,  P::Rgb24, GbrToPacked8 { bits: 16, alpha: false }),
        (P::Gbrap16Le, P::Rgba,  GbrToPacked8 { bits: 16, alpha: true }),
        (P::Rgb24, P::Gbrp16Le,  Packed8ToGbr { bits: 16, alpha: false }),
        (P::Rgba,  P::Gbrap16Le, Packed8ToGbr { bits: 16, alpha: true }),
        (P::Gbrp8,   P::Rgb24,   Gbr8ToPacked8 { alpha_in: false, alpha_out: false }),
        (P::Rgb24,   P::Gbrp8,   Packed8ToGbr8 { alpha_in: false, alpha_out: false }),
        (P::Gbrp8,   P::Rgb48Le, Gbr8ToPackedDeep { alpha_in: false, alpha_out: false }),
        (P::Rgb48Le, P::Gbrp8,   PackedDeepToGbr8 { alpha_in: false, alpha_out: false }),

        // Alpha-crossing deep-packed GBR rows: every GBR(A) member has a
        // direct hop to BOTH deep packed formats. Colour words keep the
        // family shift convention; a missing alpha is synthesised opaque
        // (full-scale 65535 on the packed side, `(1 << bits) - 1` on the
        // planar side) and a surplus alpha is dropped. These rows give
        // the staged fallback its alpha-crossing GBR pivot legs, closing
        // e.g. Gbrp8 -> Gbrap12Le and Gbrap16Le -> Gbrp10Le.
        (P::Gbrp10Le, P::Rgba64Le, GbrToPackedDeep { bits: 10, alpha_in: false, alpha_out: true }),
        (P::Gbrp12Le, P::Rgba64Le, GbrToPackedDeep { bits: 12, alpha_in: false, alpha_out: true }),
        (P::Gbrp14Le, P::Rgba64Le, GbrToPackedDeep { bits: 14, alpha_in: false, alpha_out: true }),
        (P::Gbrp16Le, P::Rgba64Le, GbrToPackedDeep { bits: 16, alpha_in: false, alpha_out: true }),
        (P::Gbrap10Le, P::Rgb48Le, GbrToPackedDeep { bits: 10, alpha_in: true, alpha_out: false }),
        (P::Gbrap12Le, P::Rgb48Le, GbrToPackedDeep { bits: 12, alpha_in: true, alpha_out: false }),
        (P::Gbrap14Le, P::Rgb48Le, GbrToPackedDeep { bits: 14, alpha_in: true, alpha_out: false }),
        (P::Gbrap16Le, P::Rgb48Le, GbrToPackedDeep { bits: 16, alpha_in: true, alpha_out: false }),
        (P::Rgba64Le, P::Gbrp10Le, PackedDeepToGbr { bits: 10, alpha_in: true, alpha_out: false }),
        (P::Rgba64Le, P::Gbrp12Le, PackedDeepToGbr { bits: 12, alpha_in: true, alpha_out: false }),
        (P::Rgba64Le, P::Gbrp14Le, PackedDeepToGbr { bits: 14, alpha_in: true, alpha_out: false }),
        (P::Rgba64Le, P::Gbrp16Le, PackedDeepToGbr { bits: 16, alpha_in: true, alpha_out: false }),
        (P::Rgb48Le, P::Gbrap10Le, PackedDeepToGbr { bits: 10, alpha_in: false, alpha_out: true }),
        (P::Rgb48Le, P::Gbrap12Le, PackedDeepToGbr { bits: 12, alpha_in: false, alpha_out: true }),
        (P::Rgb48Le, P::Gbrap14Le, PackedDeepToGbr { bits: 14, alpha_in: false, alpha_out: true }),
        (P::Rgb48Le, P::Gbrap16Le, PackedDeepToGbr { bits: 16, alpha_in: false, alpha_out: true }),
        (P::Gbrp8,    P::Rgba64Le, Gbr8ToPackedDeep { alpha_in: false, alpha_out: true }),
        (P::Rgba64Le, P::Gbrp8,    PackedDeepToGbr8 { alpha_in: true, alpha_out: false }),

        // Gbrap8 (core 0.1.34) — the byte-tier GBR + alpha member. The
        // packed-8 hops are pure plane reorders (bit-exact, and the
        // matched-alpha pair Gbrap8 ↔ Rgba is self-inverse); the deep
        // hops are the exact ×257 widen / top-byte truncation the rest
        // of the byte tier uses. Missing alpha is synthesised opaque
        // (255 on byte surfaces, 65535 on the packed-deep ones) and a
        // surplus alpha is dropped, matching the family convention.
        (P::Gbrap8, P::Rgba,     Gbr8ToPacked8 { alpha_in: true, alpha_out: true }),
        (P::Rgba,   P::Gbrap8,   Packed8ToGbr8 { alpha_in: true, alpha_out: true }),
        (P::Gbrap8, P::Rgb24,    Gbr8ToPacked8 { alpha_in: true, alpha_out: false }),
        (P::Rgb24,  P::Gbrap8,   Packed8ToGbr8 { alpha_in: false, alpha_out: true }),
        (P::Gbrp8,  P::Rgba,     Gbr8ToPacked8 { alpha_in: false, alpha_out: true }),
        (P::Rgba,   P::Gbrp8,    Packed8ToGbr8 { alpha_in: true, alpha_out: false }),
        (P::Gbrap8, P::Rgba64Le, Gbr8ToPackedDeep { alpha_in: true, alpha_out: true }),
        (P::Rgba64Le, P::Gbrap8, PackedDeepToGbr8 { alpha_in: true, alpha_out: true }),
        (P::Gbrap8, P::Rgb48Le,  Gbr8ToPackedDeep { alpha_in: true, alpha_out: false }),
        (P::Rgb48Le, P::Gbrap8,  PackedDeepToGbr8 { alpha_in: false, alpha_out: true }),
        // Alpha append / drop inside the byte tier (the same shape as
        // the YuvToYuva / YuvaToYuv rows): colour planes are carried
        // byte-for-byte, alpha is synthesised opaque or dropped.
        (P::Gbrp8,  P::Gbrap8,   Gbr8Alpha { add: true }),
        (P::Gbrap8, P::Gbrp8,    Gbr8Alpha { add: false }),

        // GBR(A) <-> Gray8 for the whole nine-member family: narrow to
        // 8 bits and project through the full-range Y' row of the
        // selected primaries on the way out (same kernel as the packed
        // RgbToGray rows, alpha dropped); broadcast the gray value into
        // G = B = R at depth on the way in (MSB-replicated widen, alpha
        // synthesised opaque). r = g = b content round-trips exactly,
        // and the Gray8 pivot now rescues the GBR <-> Mono and
        // GBR <-> deep-grayscale routes.
        (P::Gbrp8,     P::Gray8, GbrToGray { bits: 8,  alpha_in: false }),
        (P::Gbrap8,    P::Gray8, GbrToGray { bits: 8,  alpha_in: true }),
        (P::Gbrp10Le,  P::Gray8, GbrToGray { bits: 10, alpha_in: false }),
        (P::Gbrp12Le,  P::Gray8, GbrToGray { bits: 12, alpha_in: false }),
        (P::Gbrp14Le,  P::Gray8, GbrToGray { bits: 14, alpha_in: false }),
        (P::Gbrp16Le,  P::Gray8, GbrToGray { bits: 16, alpha_in: false }),
        (P::Gbrap10Le, P::Gray8, GbrToGray { bits: 10, alpha_in: true }),
        (P::Gbrap12Le, P::Gray8, GbrToGray { bits: 12, alpha_in: true }),
        (P::Gbrap14Le, P::Gray8, GbrToGray { bits: 14, alpha_in: true }),
        (P::Gbrap16Le, P::Gray8, GbrToGray { bits: 16, alpha_in: true }),
        (P::Gray8, P::Gbrp8,     GrayToGbr { bits: 8,  alpha_out: false }),
        (P::Gray8, P::Gbrap8,    GrayToGbr { bits: 8,  alpha_out: true }),
        (P::Gray8, P::Gbrp10Le,  GrayToGbr { bits: 10, alpha_out: false }),
        (P::Gray8, P::Gbrp12Le,  GrayToGbr { bits: 12, alpha_out: false }),
        (P::Gray8, P::Gbrp14Le,  GrayToGbr { bits: 14, alpha_out: false }),
        (P::Gray8, P::Gbrp16Le,  GrayToGbr { bits: 16, alpha_out: false }),
        (P::Gray8, P::Gbrap10Le, GrayToGbr { bits: 10, alpha_out: true }),
        (P::Gray8, P::Gbrap12Le, GrayToGbr { bits: 12, alpha_out: true }),
        (P::Gray8, P::Gbrap14Le, GrayToGbr { bits: 14, alpha_out: true }),
        (P::Gray8, P::Gbrap16Le, GrayToGbr { bits: 16, alpha_out: true }),

        // High-precision planar YUV (10/12-bit, 16-bit LE storage with the
        // value in the low `bits` bits) ↔ the 8-bit planar siblings. Pure
        // per-plane bit-depth scaling — luma and both chroma planes are
        // resized between 16-bit and 8-bit storage with no colour matrix
        // and no chroma resampling (the subsampling layout is preserved).
        // `wsub` / `hsub` describe the shared chroma division of both the
        // source and destination so the helper can size each plane.
        (P::Yuv420P10Le, P::Yuv420P, DepthDownYuv { wsub: 2, hsub: 2, bits: 10 }),
        (P::Yuv422P10Le, P::Yuv422P, DepthDownYuv { wsub: 2, hsub: 1, bits: 10 }),
        (P::Yuv444P10Le, P::Yuv444P, DepthDownYuv { wsub: 1, hsub: 1, bits: 10 }),
        (P::Yuv420P12Le, P::Yuv420P, DepthDownYuv { wsub: 2, hsub: 2, bits: 12 }),
        (P::Yuv422P12Le, P::Yuv422P, DepthDownYuv { wsub: 2, hsub: 1, bits: 12 }),
        (P::Yuv444P12Le, P::Yuv444P, DepthDownYuv { wsub: 1, hsub: 1, bits: 12 }),
        (P::Yuv420P, P::Yuv420P10Le, DepthUpYuv { wsub: 2, hsub: 2, bits: 10 }),
        (P::Yuv422P, P::Yuv422P10Le, DepthUpYuv { wsub: 2, hsub: 1, bits: 10 }),
        (P::Yuv444P, P::Yuv444P10Le, DepthUpYuv { wsub: 1, hsub: 1, bits: 10 }),
        (P::Yuv420P, P::Yuv420P12Le, DepthUpYuv { wsub: 2, hsub: 2, bits: 12 }),
        (P::Yuv422P, P::Yuv422P12Le, DepthUpYuv { wsub: 2, hsub: 1, bits: 12 }),
        (P::Yuv444P, P::Yuv444P12Le, DepthUpYuv { wsub: 1, hsub: 1, bits: 12 }),

        // Cross-depth planar YUV (10-bit ↔ 12-bit, same subsampling).
        // Pure per-plane storage-width rescale — widening replicates MSBs
        // into the new low bits (peak maps to peak, 10 → 12 → 10 is
        // exact); narrowing truncates. No colour math, no resampling.
        // Without these rows a 10 ↔ 12 move had to stage through the
        // 8-bit sibling and lose the low bits of both depths.
        (P::Yuv420P10Le, P::Yuv420P12Le, DepthRescaleYuv { wsub: 2, hsub: 2, src_bits: 10, dst_bits: 12 }),
        (P::Yuv422P10Le, P::Yuv422P12Le, DepthRescaleYuv { wsub: 2, hsub: 1, src_bits: 10, dst_bits: 12 }),
        (P::Yuv444P10Le, P::Yuv444P12Le, DepthRescaleYuv { wsub: 1, hsub: 1, src_bits: 10, dst_bits: 12 }),
        (P::Yuv420P12Le, P::Yuv420P10Le, DepthRescaleYuv { wsub: 2, hsub: 2, src_bits: 12, dst_bits: 10 }),
        (P::Yuv422P12Le, P::Yuv422P10Le, DepthRescaleYuv { wsub: 2, hsub: 1, src_bits: 12, dst_bits: 10 }),
        (P::Yuv444P12Le, P::Yuv444P10Le, DepthRescaleYuv { wsub: 1, hsub: 1, src_bits: 12, dst_bits: 10 }),

        // 16-bit planar YUV (`Yuv*P16Le`) — the full-width rung of the
        // depth ladder. Unlike the 10/12-bit variants every bit of the
        // LE word is significant (full-scale 65535), which the shared
        // primitives express as `bits = 16`: the mask covers the whole
        // word and the 8 → 16 widen becomes an exact ×257 (MSB
        // replication with an 8-bit period), so 8-bit content
        // round-trips losslessly and peak maps to peak. Narrowing
        // truncates (keeps the top bits) per the crate-wide
        // no-dither depth policy documented in [`crate::yuv`].
        // 16 ↔ 8 (same subsampling):
        (P::Yuv420P16Le, P::Yuv420P, DepthDownYuv { wsub: 2, hsub: 2, bits: 16 }),
        (P::Yuv422P16Le, P::Yuv422P, DepthDownYuv { wsub: 2, hsub: 1, bits: 16 }),
        (P::Yuv444P16Le, P::Yuv444P, DepthDownYuv { wsub: 1, hsub: 1, bits: 16 }),
        (P::Yuv420P, P::Yuv420P16Le, DepthUpYuv { wsub: 2, hsub: 2, bits: 16 }),
        (P::Yuv422P, P::Yuv422P16Le, DepthUpYuv { wsub: 2, hsub: 1, bits: 16 }),
        (P::Yuv444P, P::Yuv444P16Le, DepthUpYuv { wsub: 1, hsub: 1, bits: 16 }),
        // 16 ↔ 10 and 16 ↔ 12 (same subsampling): storage-width rescale,
        // exact inverses (widen replicates MSBs, narrow truncates).
        (P::Yuv420P10Le, P::Yuv420P16Le, DepthRescaleYuv { wsub: 2, hsub: 2, src_bits: 10, dst_bits: 16 }),
        (P::Yuv422P10Le, P::Yuv422P16Le, DepthRescaleYuv { wsub: 2, hsub: 1, src_bits: 10, dst_bits: 16 }),
        (P::Yuv444P10Le, P::Yuv444P16Le, DepthRescaleYuv { wsub: 1, hsub: 1, src_bits: 10, dst_bits: 16 }),
        (P::Yuv420P16Le, P::Yuv420P10Le, DepthRescaleYuv { wsub: 2, hsub: 2, src_bits: 16, dst_bits: 10 }),
        (P::Yuv422P16Le, P::Yuv422P10Le, DepthRescaleYuv { wsub: 2, hsub: 1, src_bits: 16, dst_bits: 10 }),
        (P::Yuv444P16Le, P::Yuv444P10Le, DepthRescaleYuv { wsub: 1, hsub: 1, src_bits: 16, dst_bits: 10 }),
        (P::Yuv420P12Le, P::Yuv420P16Le, DepthRescaleYuv { wsub: 2, hsub: 2, src_bits: 12, dst_bits: 16 }),
        (P::Yuv422P12Le, P::Yuv422P16Le, DepthRescaleYuv { wsub: 2, hsub: 1, src_bits: 12, dst_bits: 16 }),
        (P::Yuv444P12Le, P::Yuv444P16Le, DepthRescaleYuv { wsub: 1, hsub: 1, src_bits: 12, dst_bits: 16 }),
        (P::Yuv420P16Le, P::Yuv420P12Le, DepthRescaleYuv { wsub: 2, hsub: 2, src_bits: 16, dst_bits: 12 }),
        (P::Yuv422P16Le, P::Yuv422P12Le, DepthRescaleYuv { wsub: 2, hsub: 1, src_bits: 16, dst_bits: 12 }),
        (P::Yuv444P16Le, P::Yuv444P12Le, DepthRescaleYuv { wsub: 1, hsub: 1, src_bits: 16, dst_bits: 12 }),

        // Direct 16-bit chroma resample — the six ordered pairs over
        // (4:2:0, 4:2:2, 4:4:4) on the 16-bit family, mirroring the
        // 8-bit `ChromaResample` rows. Luma is copied word-for-word;
        // chroma is resampled at full 16-bit precision with the same
        // rounding conventions as the 8-bit helpers. These rows also
        // give the deep staged fallback its lossless-resample pivot
        // tier (see `YUV_PIVOTS_DEEP`).
        (P::Yuv420P16Le, P::Yuv422P16Le, ChromaResample16 { src_wsub: 2, src_hsub: 2, dst_wsub: 2, dst_hsub: 1 }),
        (P::Yuv420P16Le, P::Yuv444P16Le, ChromaResample16 { src_wsub: 2, src_hsub: 2, dst_wsub: 1, dst_hsub: 1 }),
        (P::Yuv422P16Le, P::Yuv420P16Le, ChromaResample16 { src_wsub: 2, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::Yuv422P16Le, P::Yuv444P16Le, ChromaResample16 { src_wsub: 2, src_hsub: 1, dst_wsub: 1, dst_hsub: 1 }),
        (P::Yuv444P16Le, P::Yuv420P16Le, ChromaResample16 { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 2 }),
        (P::Yuv444P16Le, P::Yuv422P16Le, ChromaResample16 { src_wsub: 1, src_hsub: 1, dst_wsub: 2, dst_hsub: 1 }),

        // Deep grayscale wiring — Gray10Le / Gray12Le previously had NO
        // conversion entries at all (the only PixelFormat variants with
        // zero coverage). Same per-plane primitives as the YUV depth
        // ladder: 8-bit endpoints round-trip exactly through any deeper
        // width, and 16-bit storage acts as the common widest rung.
        (P::Gray10Le, P::Gray8, GrayDepthDown8 { bits: 10 }),
        (P::Gray12Le, P::Gray8, GrayDepthDown8 { bits: 12 }),
        (P::Gray8, P::Gray10Le, GrayDepthUp8 { bits: 10 }),
        (P::Gray8, P::Gray12Le, GrayDepthUp8 { bits: 12 }),
        (P::Gray10Le, P::Gray12Le, GrayDepthRescale { src_bits: 10, dst_bits: 12 }),
        (P::Gray12Le, P::Gray10Le, GrayDepthRescale { src_bits: 12, dst_bits: 10 }),
        (P::Gray10Le, P::Gray16Le, GrayDepthRescale { src_bits: 10, dst_bits: 16 }),
        (P::Gray16Le, P::Gray10Le, GrayDepthRescale { src_bits: 16, dst_bits: 10 }),
        (P::Gray12Le, P::Gray16Le, GrayDepthRescale { src_bits: 12, dst_bits: 16 }),
        (P::Gray16Le, P::Gray12Le, GrayDepthRescale { src_bits: 16, dst_bits: 12 }),
    ]
};

fn lookup(src: PixelFormat, dst: PixelFormat) -> Option<&'static ConvertOp> {
    TABLE
        .iter()
        .find(|(s, d, _)| *s == src && *d == dst)
        .map(|(_, _, op)| op)
}

/// Dispatch descriptor for each coverage-table row. The variant
/// discriminates on the conversion family, and embedded fields carry
/// the variant-specific parameters (swizzle positions, chroma
/// subsampling, range direction, …).
#[derive(Clone, Copy)]
enum ConvertOp {
    Swizzle3 {
        src: rgb::Rgb3,
        dst: rgb::Rgb3,
    },
    Swizzle4 {
        src: rgb::Rgba4,
        dst: rgb::Rgba4,
    },
    Promote3To4 {
        src: rgb::Rgb3,
        dst: rgb::Rgba4,
    },
    Demote4To3 {
        src: rgb::Rgba4,
        dst: rgb::Rgb3,
    },
    Rgb48ToRgb24,
    Rgb24ToRgb48,
    Rgba64ToRgba,
    RgbaToRgba64,
    Gray8ToPacked3,
    /// Gray broadcast into a 4-byte packed pixel. `alpha_first: false`
    /// emits (g, g, g, 255) — Rgba and Bgra alike; `true` emits
    /// (255, g, g, g) for the alpha-first orders Argb / Abgr.
    Gray8ToPacked4 {
        alpha_first: bool,
    },
    Gray16ToGray8,
    Gray8ToGray16,
    MonoToGray {
        black_is_zero: bool,
    },
    GrayToMono {
        black_is_zero: bool,
    },
    /// Packed RGB(A) → `Gray8`: full-range luminance projection under
    /// the Y' row of the selected primaries. Alpha (if present) is
    /// dropped.
    RgbToGray {
        alpha_in: bool,
    },
    Ya8ToGray8,
    Gray8ToYa8,
    Ya8ToRgb24,
    Ya8ToRgba,
    Rgb24ToYa8,
    RgbaToYa8,
    /// `Ya16Le` → `Ya8`: high-byte truncation of both words.
    Ya16ToYa8,
    /// `Ya8` → `Ya16Le`: exact ×257 widen of both components (the
    /// inverse of [`Self::Ya16ToYa8`] on 8-bit content).
    Ya8ToYa16,
    /// `Ya16Le` → `Gray16Le`: luma word carried verbatim, alpha
    /// dropped.
    Ya16ToGray16,
    /// `Gray16Le` → `Ya16Le`: luma word carried verbatim, alpha
    /// synthesised opaque 65535.
    Gray16ToYa16,
    /// `Ya16Le` → `Gray8`: high byte of the luma word, alpha dropped.
    Ya16ToGray8,
    /// `Gray8` → `Ya16Le`: ×257 widen, alpha opaque 65535.
    Gray8ToYa16,
    /// `Ya16Le` → `Rgba64Le`: luma word broadcast into R, G, B; alpha
    /// word carried verbatim (bit-exact).
    Ya16ToRgba64,
    /// `Rgba64Le` → `Ya16Le`: rounded-mean luma derivation over the R,
    /// G, B words (the 16-bit analogue of the Ya8 rule); alpha word
    /// carried verbatim.
    Rgba64ToYa16,
    /// `Ya16Le` → packed 8-bit RGB: high-byte broadcast; the alpha
    /// high byte is carried (`Rgba`) or dropped (`Rgb24`).
    Ya16ToPacked8 {
        alpha: bool,
    },
    /// Packed 8-bit RGB → `Ya16Le`: rounded-mean luma then the exact
    /// ×257 widen; alpha widened from the source (`Rgba`) or opaque
    /// 65535 (`Rgb24`).
    Packed8ToYa16 {
        alpha_in: bool,
    },
    /// Any YUV-family source (planar, semi-planar NV, or planar +
    /// alpha) → `Gray8` by extracting the full-resolution luma plane.
    /// Chroma (and alpha, for `Yuva420P`) is dropped. `full_range`
    /// mirrors the source family: limited-range sources are rescaled
    /// 16..=235 → 0..=255; `YuvJ*` luma is copied verbatim. Only the
    /// luma plane is touched, so odd dimensions are fine even on
    /// subsampled sources.
    YuvLumaToGray {
        full_range: bool,
    },
    /// `Gray8` → planar YUV: the gray plane becomes luma (rescaled to
    /// limited range unless the destination is a full-range `YuvJ*`),
    /// and the chroma planes are synthesised at the neutral code 128.
    GrayToYuvPlanar {
        wsub: usize,
        hsub: usize,
        full_range: bool,
    },
    /// `Gray8` → NV12 / NV21: limited-range luma plus an all-neutral
    /// interleaved chroma plane (identical bytes for both NV orders).
    GrayToNv,
    YuvToRgb {
        wsub: usize,
        hsub: usize,
        alpha: bool,
        /// `true` for the full-range `YuvJ*` source families: the matrix
        /// range is a property of the *format*, so it overrides the range
        /// half of `ConvertOptions::color_space` (which still picks the
        /// primaries).
        full_range: bool,
    },
    RgbToYuv {
        wsub: usize,
        hsub: usize,
        alpha_in: bool,
        /// `true` when the destination is a full-range `YuvJ*` family.
        full_range: bool,
    },
    RescaleRange {
        wsub: usize,
        hsub: usize,
        to_full: bool,
    },
    /// Planar YUV → planar YUV with the luma plane copied byte-for-byte
    /// and the chroma planes resampled between two subsampling layouts.
    /// `src_wsub` / `src_hsub` describe the source's chroma division;
    /// `dst_wsub` / `dst_hsub` the destination's. Equal sub factors on
    /// both sides would be a no-op and are not registered.
    ChromaResample {
        src_wsub: usize,
        src_hsub: usize,
        dst_wsub: usize,
        dst_hsub: usize,
    },
    NvToYuv420p {
        is_nv12: bool,
    },
    Yuv420pToNv {
        is_nv12: bool,
    },
    NvToRgb {
        is_nv12: bool,
        /// `true` to emit RGBA, `false` to emit Rgb24.
        alpha: bool,
    },
    RgbToNv {
        is_nv12: bool,
        /// `true` if source is RGBA (alpha consumed by skipping the 4th byte).
        alpha_in: bool,
    },
    Packed422ToYuv422p {
        /// `true` for YUYV byte order, `false` for UYVY.
        is_yuyv: bool,
    },
    Yuv422pToPacked422 {
        is_yuyv: bool,
    },
    Packed422Swap,
    Packed422ToRgb {
        is_yuyv: bool,
        /// `true` to emit RGBA, `false` to emit Rgb24.
        alpha: bool,
    },
    RgbToPacked422 {
        is_yuyv: bool,
        /// `true` if source is RGBA (alpha consumed by skipping the 4th byte).
        alpha_in: bool,
    },
    Pal8ToRgb {
        alpha: bool,
    },
    RgbToPal8 {
        alpha_in: bool,
    },
    CmykToRgb {
        /// When true, output is RGBA (opaque alpha). When false, Rgb24.
        alpha: bool,
        /// When true, the source is the inverted-ink convention
        /// (`CmykInverted`, stored byte = 255 − ink).
        inverted: bool,
    },
    RgbToCmyk {
        /// When true, source is RGBA (alpha ignored). When false, Rgb24.
        alpha_in: bool,
        /// When true, the destination is `CmykInverted` (regular
        /// separation followed by the per-byte complement).
        inverted: bool,
    },
    /// `Cmyk` ↔ `CmykInverted`: complement every byte — an exact,
    /// self-inverse bijection serving both directions.
    CmykComplement,
    /// Alpha-less planar YUV (3 planes) → the `Yuva*` sibling with the
    /// same chroma grid (4 planes) by appending an opaque
    /// full-resolution alpha plane. `wsub` / `hsub` are the shared
    /// chroma division.
    YuvToYuva {
        wsub: usize,
        hsub: usize,
    },
    /// `Yuva*` → the alpha-less planar sibling by dropping the trailing
    /// full-resolution alpha plane.
    YuvaToYuv {
        wsub: usize,
        hsub: usize,
    },
    /// `Yuva420P` / `Yuva422P` / `Yuva444P` → packed RGB. The YUV math
    /// runs through the existing planar decoder for the format's chroma
    /// grid; the full-resolution alpha plane is either dropped
    /// (`alpha = false`, emit `Rgb24`) or interleaved into the output
    /// (`alpha = true`, emit `Rgba`).
    YuvaToRgb {
        wsub: usize,
        hsub: usize,
        alpha: bool,
    },
    /// Packed RGB → `Yuva420P` / `Yuva422P` / `Yuva444P`. The YUV math
    /// runs through the existing planar encoder; the alpha plane is
    /// either synthesised opaque (`alpha_in = false`, source is
    /// `Rgb24`) or split out of the input (`alpha_in = true`, source is
    /// `Rgba`).
    RgbToYuva {
        wsub: usize,
        hsub: usize,
        alpha_in: bool,
    },
    /// `Yuva*` → `Yuva*`: luma and the full-resolution alpha plane are
    /// copied byte-for-byte; the chroma pair is resampled between the
    /// two subsampling layouts with the same primitives as
    /// [`Self::ChromaResample`]. Alpha survives bit-exact.
    YuvaChromaResample {
        src_wsub: usize,
        src_hsub: usize,
        dst_wsub: usize,
        dst_hsub: usize,
    },
    /// 16-bit planar YUV → 16-bit planar YUV (`Yuv*P16Le` family): luma
    /// copied word-for-word, chroma resampled between the two
    /// subsampling layouts at full 16-bit precision via the
    /// `yuv::chroma16le_*` helpers (rounding mirrors the 8-bit
    /// [`Self::ChromaResample`] exactly).
    ChromaResample16 {
        src_wsub: usize,
        src_hsub: usize,
        dst_wsub: usize,
        dst_hsub: usize,
    },
    /// Planar GBR(A) → packed deep RGB (`Rgb48Le` / `Rgba64Le`). Reorders
    /// the G, B, R(, A) planes into packed R, G, B(, A) 16-bit words and
    /// left-shifts each `bits`-significant sample by `16 - bits` so the
    /// packed word uses the full 16-bit range. `alpha_in` names the
    /// source's 4-plane shape, `alpha_out` the packed `Rgba64Le` target;
    /// when they differ the alpha plane is dropped (`Gbrap* → Rgb48Le`)
    /// or synthesised opaque full-scale 65535 (`Gbrp* → Rgba64Le`).
    GbrToPackedDeep {
        bits: u8,
        alpha_in: bool,
        alpha_out: bool,
    },
    /// Packed deep RGB (`Rgb48Le` / `Rgba64Le`) → planar GBR(A). The
    /// inverse of [`Self::GbrToPackedDeep`]: splits the packed R, G, B(, A)
    /// words into G, B, R(, A) planes and right-shifts each by `16 - bits`
    /// back into the `bits`-significant low range. Mismatched alpha
    /// flags drop the packed alpha word (`Rgba64Le → Gbrp*`) or
    /// synthesise an opaque `(1 << bits) - 1` plane (`Rgb48Le → Gbrap*`).
    PackedDeepToGbr {
        bits: u8,
        alpha_in: bool,
        alpha_out: bool,
    },
    /// High-precision planar YUV (`Yuv*P10Le` / `Yuv*P12Le` /
    /// `Yuv*P16Le`, 16-bit LE storage) → the 8-bit planar sibling
    /// (`Yuv*P`). Each of the three planes is reduced from a
    /// `bits`-significant 16-bit word to 8 bits by truncation (keep the
    /// top 8 significant bits). `wsub` / `hsub` are the shared chroma
    /// division; subsampling is preserved (no chroma resample).
    DepthDownYuv {
        wsub: usize,
        hsub: usize,
        bits: u32,
    },
    /// 8-bit planar YUV (`Yuv*P`) → the high-precision planar sibling
    /// (`Yuv*P10Le` / `Yuv*P12Le` / `Yuv*P16Le`). The inverse of
    /// [`Self::DepthDownYuv`]: each plane is widened to a
    /// `bits`-significant 16-bit LE word with the 8-bit value in the
    /// high bits and its MSBs replicated into the low slack so the
    /// down-conversion round-trips exactly (for `bits = 16` the widen
    /// is an exact ×257 and full-scale 255 maps to 65535).
    DepthUpYuv {
        wsub: usize,
        hsub: usize,
        bits: u32,
    },
    /// Planar GBR(A) → 8-bit packed `Rgb24` / `Rgba`: plane reorder plus
    /// a `bits` → 8 narrowing (keep the top 8 significant bits).
    GbrToPacked8 {
        bits: u32,
        alpha: bool,
    },
    /// 8-bit packed `Rgb24` / `Rgba` → planar GBR(A): plane split plus an
    /// 8 → `bits` MSB-replicated widen (exact inverse of
    /// [`Self::GbrToPacked8`], so 8-bit content round-trips losslessly).
    Packed8ToGbr {
        bits: u32,
        alpha: bool,
    },
    /// Byte-tier planar GBR(A) (`Gbrp8` / `Gbrap8`) → packed `Rgb24` /
    /// `Rgba`: pure plane reorder, no depth math — bit-exact, and the
    /// matched-alpha pairs are self-inverse with
    /// [`Self::Packed8ToGbr8`]. When the alpha flags differ the alpha
    /// is synthesised opaque 255 (`alpha_out` without `alpha_in`) or
    /// dropped (`alpha_in` without `alpha_out`).
    Gbr8ToPacked8 {
        alpha_in: bool,
        alpha_out: bool,
    },
    /// Packed `Rgb24` / `Rgba` → byte-tier planar GBR(A): the inverse
    /// plane split of [`Self::Gbr8ToPacked8`], same alpha convention.
    Packed8ToGbr8 {
        alpha_in: bool,
        alpha_out: bool,
    },
    /// Byte-tier planar GBR(A) → packed deep RGB: plane reorder plus
    /// the exact ×257 widen into full-width 16-bit words (peak maps to
    /// peak), mirroring `Rgb24 → Rgb48Le`. A carried alpha
    /// (`alpha_in && alpha_out`) is widened like the colour bytes; a
    /// missing one is synthesised opaque 65535 and a surplus one is
    /// dropped.
    Gbr8ToPackedDeep {
        alpha_in: bool,
        alpha_out: bool,
    },
    /// Packed deep RGB (`Rgb48Le` / `Rgba64Le`) → byte-tier planar
    /// GBR(A): plane split keeping the top byte of each word
    /// (truncation — the exact inverse of the ×257 widen). Alpha is
    /// carried truncated, synthesised opaque 255, or dropped per the
    /// flag pair.
    PackedDeepToGbr8 {
        alpha_in: bool,
        alpha_out: bool,
    },
    /// `Gbrp8` ↔ `Gbrap8`: colour planes copied byte-for-byte; `add`
    /// appends an opaque 255 full-resolution alpha plane, `!add` drops
    /// plane 3 (the byte-tier mirror of [`Self::YuvToYuva`] /
    /// [`Self::YuvaToYuv`]).
    Gbr8Alpha {
        add: bool,
    },
    /// Planar GBR(A) → `Gray8`: narrow each colour plane to 8 bits
    /// (top-bits truncation, crate depth policy) and project through
    /// the full-range Y' row of the selected primaries — the same
    /// kernel as the packed [`Self::RgbToGray`] rows, so `r = g = b`
    /// content maps to itself exactly. Alpha (when present) is dropped.
    GbrToGray {
        bits: u32,
        alpha_in: bool,
    },
    /// `Gray8` → planar GBR(A): broadcast the gray value into the G, B
    /// and R planes at the family depth (MSB-replicated widen — peak
    /// maps to peak, and [`Self::GbrToGray`] recovers the original
    /// exactly). `alpha_out` appends an opaque full-scale alpha plane.
    GrayToGbr {
        bits: u32,
        alpha_out: bool,
    },
    /// Cross-depth planar YUV: rescale every plane between two
    /// `bits`-significant 16-bit LE storage widths (10 ↔ 12 ↔ 16) with
    /// the subsampling layout preserved. Widening replicates MSBs into
    /// the new low bits; narrowing truncates them (exact inverses).
    DepthRescaleYuv {
        wsub: usize,
        hsub: usize,
        src_bits: u32,
        dst_bits: u32,
    },
    /// Deep grayscale (`Gray10Le` / `Gray12Le`) → `Gray8`: keep the top
    /// 8 of the `bits` significant bits (same primitive as the YUV depth
    /// ladder).
    GrayDepthDown8 {
        bits: u32,
    },
    /// `Gray8` → deep grayscale: value in the high bits with MSB
    /// replication into the low slack, so `GrayDepthDown8` recovers the
    /// original exactly.
    GrayDepthUp8 {
        bits: u32,
    },
    /// Deep grayscale ↔ deep grayscale storage-width rescale
    /// (10 ↔ 12 ↔ 16), single plane.
    GrayDepthRescale {
        src_bits: u32,
        dst_bits: u32,
    },
    /// Computed tier (see [`lookup_computed`]): any ordered pair inside
    /// the uniform planar YUV(A) family. Fuses the depth move, the
    /// chroma resample (performed at the deeper of the two depths) and
    /// the alpha handling (carry with depth move / drop / synthesise
    /// opaque full-scale) into one step. Luma — and a carried alpha
    /// plane — never touch the resampler, so they survive bit-exact
    /// whenever the depths match.
    PlanarFamily {
        src: PlanarYuv,
        dst: PlanarYuv,
    },
    /// Computed tier: planar family member → packed `Rgb24` / `Rgba`.
    /// Deep planes are reduced to 8 bits (truncation, per the crate
    /// depth policy) before the proven 8-bit scalar/SIMD decode; alpha
    /// is carried (reduced to 8 bits), synthesised opaque, or dropped.
    PlanarFamilyToRgb {
        src: PlanarYuv,
        alpha: bool,
    },
    /// Computed tier: packed `Rgb24` / `Rgba` → planar family member.
    /// Encodes through the proven 8-bit path, then widens every plane
    /// (MSB replication) to the destination depth; alpha is split out
    /// of an `Rgba` source or synthesised opaque full-scale.
    RgbToPlanarFamily {
        dst: PlanarYuv,
        alpha_in: bool,
    },
    /// Computed tier: planar family member → `Gray8` luma extraction
    /// (deep luma truncated to 8 bits, then limited → full range
    /// rescale; chroma and alpha dropped).
    PlanarFamilyToGray {
        src: PlanarYuv,
    },
    /// Computed tier: `Gray8` → planar family member. The gray plane
    /// becomes luma (full → limited range, then widened to the family
    /// depth); chroma is synthesised at the exact neutral mid-code
    /// `1 << (bits - 1)` (512 at 10 bits, 32768 at 16 — not the
    /// widened 8-bit 128, which lands a couple of codes off neutral);
    /// alpha (when present) is synthesised opaque full-scale.
    GrayToPlanarFamily {
        dst: PlanarYuv,
    },
}

impl ConvertOp {
    fn apply(
        &self,
        src: &VideoFrame,
        src_info: FrameInfo,
        opts: &ConvertOptions,
    ) -> Result<VideoFrame> {
        // The plain `Yuv*` paths always use the limited-range matrix and
        // the full-range `YuvJ*` paths always use the full-range matrix —
        // range is a property of the pixel format, so only the primaries
        // half of `ConvertOptions::color_space` is honoured here. The
        // format-specific override happens in the YuvToRgb / RgbToYuv
        // arms below via their `full_range` field.
        let matrix = YuvMatrix::from_color_space(opts.color_space).with_range(true);
        match *self {
            Self::Swizzle3 { src: sp, dst: dp } => swizzle3(src, src_info, sp, dp),
            Self::Swizzle4 { src: sp, dst: dp } => swizzle4(src, src_info, sp, dp),
            Self::Promote3To4 { src: sp, dst: dp } => promote3_to_4(src, src_info, sp, dp),
            Self::Demote4To3 { src: sp, dst: dp } => demote4_to_3(src, src_info, sp, dp),
            Self::Rgb48ToRgb24 => do_rgb48_to_rgb24(src, src_info),
            Self::Rgb24ToRgb48 => do_rgb24_to_rgb48(src, src_info),
            Self::Rgba64ToRgba => do_rgba64_to_rgba(src, src_info),
            Self::RgbaToRgba64 => do_rgba_to_rgba64(src, src_info),
            Self::Gray8ToPacked3 => gray_to_packed3(src, src_info),
            Self::Gray8ToPacked4 { alpha_first } => gray_to_packed4(src, src_info, alpha_first),
            Self::Gray16ToGray8 => do_gray16_to_gray8(src, src_info),
            Self::Gray8ToGray16 => do_gray8_to_gray16(src, src_info),
            Self::MonoToGray { black_is_zero } => do_mono_to_gray(src, src_info, black_is_zero),
            Self::GrayToMono { black_is_zero } => do_gray_to_mono(src, src_info, black_is_zero),
            Self::RgbToGray { alpha_in } => {
                do_rgb_to_gray(src, src_info, matrix.with_range(false), alpha_in)
            }
            Self::Ya8ToGray8 => do_ya8_to_gray8(src, src_info),
            Self::Gray8ToYa8 => do_gray8_to_ya8(src, src_info),
            Self::Ya8ToRgb24 => do_ya8_to_rgb24(src, src_info),
            Self::Ya8ToRgba => do_ya8_to_rgba(src, src_info),
            Self::Rgb24ToYa8 => do_rgb24_to_ya8(src, src_info),
            Self::RgbaToYa8 => do_rgba_to_ya8(src, src_info),
            Self::Ya16ToYa8 => packed_map(src, src_info, 4, 2, gray::ya16le_to_ya8),
            Self::Ya8ToYa16 => packed_map(src, src_info, 2, 4, gray::ya8_to_ya16le),
            Self::Ya16ToGray16 => packed_map(src, src_info, 4, 2, gray::ya16le_to_gray16le),
            Self::Gray16ToYa16 => packed_map(src, src_info, 2, 4, gray::gray16le_to_ya16le),
            Self::Ya16ToGray8 => packed_map(src, src_info, 4, 1, gray::ya16le_to_gray8),
            Self::Gray8ToYa16 => packed_map(src, src_info, 1, 4, gray::gray8_to_ya16le),
            Self::Ya16ToRgba64 => packed_map(src, src_info, 4, 8, gray::ya16le_to_rgba64le),
            Self::Rgba64ToYa16 => packed_map(src, src_info, 8, 4, gray::rgba64le_to_ya16le),
            Self::Ya16ToPacked8 { alpha } => {
                if alpha {
                    packed_map(src, src_info, 4, 4, gray::ya16le_to_rgba)
                } else {
                    packed_map(src, src_info, 4, 3, gray::ya16le_to_rgb24)
                }
            }
            Self::Packed8ToYa16 { alpha_in } => {
                if alpha_in {
                    packed_map(src, src_info, 4, 4, gray::rgba_to_ya16le)
                } else {
                    packed_map(src, src_info, 3, 4, gray::rgb24_to_ya16le)
                }
            }
            Self::YuvLumaToGray { full_range } => do_yuv_luma_to_gray(src, src_info, full_range),
            Self::GrayToYuvPlanar {
                wsub,
                hsub,
                full_range,
            } => do_gray_to_yuv_planar(src, src_info, wsub, hsub, full_range),
            Self::GrayToNv => do_gray_to_nv(src, src_info),
            Self::YuvToRgb {
                wsub,
                hsub,
                alpha,
                full_range,
            } => {
                let m = matrix.with_range(!full_range);
                do_yuv_to_rgb(src, src_info, m, wsub, hsub, alpha)
            }
            Self::RgbToYuv {
                wsub,
                hsub,
                alpha_in,
                full_range,
            } => {
                let m = matrix.with_range(!full_range);
                do_rgb_to_yuv(src, src_info, m, wsub, hsub, alpha_in)
            }
            Self::RescaleRange {
                wsub,
                hsub,
                to_full,
            } => rescale_range(src, src_info, wsub, hsub, to_full),
            Self::ChromaResample {
                src_wsub,
                src_hsub,
                dst_wsub,
                dst_hsub,
            } => chroma_resample(src, src_info, src_wsub, src_hsub, dst_wsub, dst_hsub),
            Self::NvToYuv420p { is_nv12 } => nv_to_yuv420p(src, src_info, is_nv12),
            Self::Yuv420pToNv { is_nv12 } => yuv420p_to_nv(src, src_info, is_nv12),
            Self::NvToRgb { is_nv12, alpha } => nv_to_rgb(src, src_info, matrix, is_nv12, alpha),
            Self::RgbToNv { is_nv12, alpha_in } => {
                rgb_to_nv(src, src_info, matrix, is_nv12, alpha_in)
            }
            Self::Packed422ToYuv422p { is_yuyv } => packed422_to_yuv422p(src, src_info, is_yuyv),
            Self::Yuv422pToPacked422 { is_yuyv } => yuv422p_to_packed422(src, src_info, is_yuyv),
            Self::Packed422Swap => packed422_swap(src, src_info),
            Self::Packed422ToRgb { is_yuyv, alpha } => {
                packed422_to_rgb(src, src_info, matrix, is_yuyv, alpha)
            }
            Self::RgbToPacked422 { is_yuyv, alpha_in } => {
                rgb_to_packed422(src, src_info, matrix, is_yuyv, alpha_in)
            }
            Self::Pal8ToRgb { alpha } => pal8_to_rgb(src, src_info, opts, alpha),
            Self::RgbToPal8 { alpha_in } => rgb_to_pal8(src, src_info, opts, alpha_in),
            Self::CmykToRgb { alpha, inverted } => {
                let f = match (alpha, inverted) {
                    (false, false) => cmyk::cmyk_to_rgb24,
                    (true, false) => cmyk::cmyk_to_rgba,
                    (false, true) => cmyk::cmyk_inverted_to_rgb24,
                    (true, true) => cmyk::cmyk_inverted_to_rgba,
                };
                packed_map(src, src_info, 4, if alpha { 4 } else { 3 }, f)
            }
            Self::RgbToCmyk { alpha_in, inverted } => {
                let f = match (alpha_in, inverted) {
                    (false, false) => cmyk::rgb24_to_cmyk,
                    (true, false) => cmyk::rgba_to_cmyk,
                    (false, true) => cmyk::rgb24_to_cmyk_inverted,
                    (true, true) => cmyk::rgba_to_cmyk_inverted,
                };
                packed_map(src, src_info, if alpha_in { 4 } else { 3 }, 4, f)
            }
            Self::CmykComplement => packed_map(src, src_info, 4, 4, cmyk::cmyk_complement),
            Self::YuvToYuva { wsub, hsub } => do_yuv_to_yuva(src, src_info, wsub, hsub),
            Self::YuvaToYuv { wsub, hsub } => do_yuva_to_yuv(src, src_info, wsub, hsub),
            Self::YuvaToRgb { wsub, hsub, alpha } => {
                do_yuva_to_rgb(src, src_info, matrix, wsub, hsub, alpha)
            }
            Self::RgbToYuva {
                wsub,
                hsub,
                alpha_in,
            } => do_rgb_to_yuva(src, src_info, matrix, wsub, hsub, alpha_in),
            Self::YuvaChromaResample {
                src_wsub,
                src_hsub,
                dst_wsub,
                dst_hsub,
            } => yuva_chroma_resample(src, src_info, src_wsub, src_hsub, dst_wsub, dst_hsub),
            Self::ChromaResample16 {
                src_wsub,
                src_hsub,
                dst_wsub,
                dst_hsub,
            } => chroma_resample16(src, src_info, src_wsub, src_hsub, dst_wsub, dst_hsub),
            Self::GbrToPackedDeep {
                bits,
                alpha_in,
                alpha_out,
            } => do_gbr_to_packed_deep(src, src_info, bits, alpha_in, alpha_out),
            Self::PackedDeepToGbr {
                bits,
                alpha_in,
                alpha_out,
            } => do_packed_deep_to_gbr(src, src_info, bits, alpha_in, alpha_out),
            Self::Gbr8ToPackedDeep {
                alpha_in,
                alpha_out,
            } => do_gbr8_to_packed_deep(src, src_info, alpha_in, alpha_out),
            Self::PackedDeepToGbr8 {
                alpha_in,
                alpha_out,
            } => do_packed_deep_to_gbr8(src, src_info, alpha_in, alpha_out),
            Self::Gbr8Alpha { add } => do_gbr8_alpha(src, src_info, add),
            Self::GbrToGray { bits, alpha_in } => {
                // Gray8 is a full-range space — like the packed RgbToGray
                // rows, the projection always uses the full-range Y' row
                // (the ColorSpace knob still picks the primaries).
                do_gbr_to_gray(src, src_info, matrix.with_range(false), bits, alpha_in)
            }
            Self::GrayToGbr { bits, alpha_out } => do_gray_to_gbr(src, src_info, bits, alpha_out),
            Self::DepthDownYuv { wsub, hsub, bits } => {
                do_yuv_depth_down(src, src_info, wsub, hsub, bits)
            }
            Self::DepthUpYuv { wsub, hsub, bits } => {
                do_yuv_depth_up(src, src_info, wsub, hsub, bits)
            }
            Self::GbrToPacked8 { bits, alpha } => do_gbr_to_packed8(src, src_info, bits, alpha),
            Self::Packed8ToGbr { bits, alpha } => do_packed8_to_gbr(src, src_info, bits, alpha),
            Self::Gbr8ToPacked8 {
                alpha_in,
                alpha_out,
            } => do_gbr8_to_packed8(src, src_info, alpha_in, alpha_out),
            Self::Packed8ToGbr8 {
                alpha_in,
                alpha_out,
            } => do_packed8_to_gbr8(src, src_info, alpha_in, alpha_out),
            Self::DepthRescaleYuv {
                wsub,
                hsub,
                src_bits,
                dst_bits,
            } => do_yuv_depth_rescale(src, src_info, wsub, hsub, src_bits, dst_bits),
            Self::GrayDepthDown8 { bits } => do_gray_depth_down8(src, src_info, bits),
            Self::GrayDepthUp8 { bits } => do_gray_depth_up8(src, src_info, bits),
            Self::GrayDepthRescale { src_bits, dst_bits } => {
                do_gray_depth_rescale(src, src_info, src_bits, dst_bits)
            }
            Self::PlanarFamily { src: s, dst: d } => planar_family(src, src_info, s, d),
            Self::PlanarFamilyToRgb { src: s, alpha } => {
                planar_family_to_rgb(src, src_info, matrix, s, alpha)
            }
            Self::RgbToPlanarFamily { dst: d, alpha_in } => {
                rgb_to_planar_family(src, src_info, matrix, d, alpha_in)
            }
            Self::PlanarFamilyToGray { src: s } => planar_family_to_gray(src, src_info, s),
            Self::GrayToPlanarFamily { dst: d } => gray_to_planar_family(src, src_info, d),
        }
    }
}

// -------------------------------------------------------------------------
// Frame helpers.

fn make_frame(src: &VideoFrame, planes: Vec<VideoPlane>) -> VideoFrame {
    VideoFrame {
        pts: src.pts,
        planes,
    }
}

fn tight_row(src: &[u8], stride: usize, row: usize, row_bytes: usize) -> &[u8] {
    let off = row * stride;
    &src[off..off + row_bytes]
}

/// Row-wise map between two single-plane packed layouts: gather each
/// tight source row (`src_bpp` bytes/pixel) and let `f` emit the
/// corresponding destination row (`dst_bpp` bytes/pixel). The kernel
/// signature matches the low-level `(src, dst, pixels)` helpers in
/// [`crate::gray`] / [`crate::rgb`].
fn packed_map(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_bpp: usize,
    dst_bpp: usize,
    f: fn(&[u8], &mut [u8], usize),
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * dst_bpp];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * src_bpp);
        f(sr, &mut out[row * w * dst_bpp..(row + 1) * w * dst_bpp], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * dst_bpp,
            data: out,
        }],
    ))
}

fn gather_tight(src: &[u8], stride: usize, w_bytes: usize, h: usize) -> Vec<u8> {
    if stride == w_bytes {
        return src[..w_bytes * h].to_vec();
    }
    let mut out = Vec::with_capacity(w_bytes * h);
    for row in 0..h {
        out.extend_from_slice(tight_row(src, stride, row, w_bytes));
    }
    out
}

// -------------------------------------------------------------------------
// RGB family.

fn swizzle3(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_pos: rgb::Rgb3,
    dst_pos: rgb::Rgb3,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 3);
        rgb::swizzle3(
            sr,
            src_pos,
            &mut out[row * w * 3..row * w * 3 + w * 3],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

fn swizzle4(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_pos: rgb::Rgba4,
    dst_pos: rgb::Rgba4,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        rgb::swizzle4(
            sr,
            src_pos,
            &mut out[row * w * 4..row * w * 4 + w * 4],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn promote3_to_4(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_pos: rgb::Rgb3,
    dst_pos: rgb::Rgba4,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 3);
        rgb::rgb3_to_rgba4(
            sr,
            src_pos,
            &mut out[row * w * 4..row * w * 4 + w * 4],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn demote4_to_3(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_pos: rgb::Rgba4,
    dst_pos: rgb::Rgb3,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        rgb::rgba4_to_rgb3(
            sr,
            src_pos,
            &mut out[row * w * 3..row * w * 3 + w * 3],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

// -------------------------------------------------------------------------
// Deep RGB.

fn do_rgb48_to_rgb24(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 6);
        rgb::rgb48_to_rgb24(sr, &mut out[row * w * 3..row * w * 3 + w * 3], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

fn do_rgb24_to_rgb48(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 6];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 3);
        rgb::rgb24_to_rgb48(sr, &mut out[row * w * 6..row * w * 6 + w * 6], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 6,
            data: out,
        }],
    ))
}

fn do_rgba64_to_rgba(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 8);
        rgb::rgba64_to_rgba(sr, &mut out[row * w * 4..row * w * 4 + w * 4], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn do_rgba_to_rgba64(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 8];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        rgb::rgba_to_rgba64(sr, &mut out[row * w * 8..row * w * 8 + w * 8], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 8,
            data: out,
        }],
    ))
}

// -------------------------------------------------------------------------
// Gray / Mono.

fn gray_to_packed3(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        gray::gray8_to_rgb24(sr, &mut out[row * w * 3..row * w * 3 + w * 3], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

fn gray_to_packed4(src: &VideoFrame, src_info: FrameInfo, alpha_first: bool) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        let dr = &mut out[row * w * 4..row * w * 4 + w * 4];
        if alpha_first {
            for (i, &g) in sr.iter().enumerate().take(w) {
                dr[i * 4] = 255;
                dr[i * 4 + 1] = g;
                dr[i * 4 + 2] = g;
                dr[i * 4 + 3] = g;
            }
        } else {
            gray::gray8_to_rgba(sr, dr, w);
        }
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn do_gray16_to_gray8(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 2);
        gray::gray16le_to_gray8(sr, &mut out[row * w..row * w + w], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}

fn do_gray8_to_gray16(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 2];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        gray::gray8_to_gray16le(sr, &mut out[row * w * 2..row * w * 2 + w * 2], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: out,
        }],
    ))
}

fn do_mono_to_gray(
    src: &VideoFrame,
    src_info: FrameInfo,
    black_is_zero: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h];
    // Mono strides are often `(w + 7) / 8`, but honour the provided
    // stride if it differs.
    let src_stride = in_plane.stride;
    let compact = gather_mono_rows(&in_plane.data, src_stride, w.div_ceil(8), h);
    gray::mono_to_gray8(&compact, &mut out, w, h, black_is_zero);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}

fn do_gray_to_mono(
    src: &VideoFrame,
    src_info: FrameInfo,
    black_is_zero: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let packed_stride = w.div_ceil(8);
    let src_tight = gather_tight(&in_plane.data, in_plane.stride, w, h);
    let mut out = vec![0u8; packed_stride * h];
    gray::gray8_to_mono(&src_tight, &mut out, w, h, black_is_zero);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: packed_stride,
            data: out,
        }],
    ))
}

fn gather_mono_rows(src: &[u8], stride: usize, packed: usize, h: usize) -> Vec<u8> {
    if stride == packed {
        return src[..packed * h].to_vec();
    }
    let mut out = Vec::with_capacity(packed * h);
    for row in 0..h {
        out.extend_from_slice(&src[row * stride..row * stride + packed]);
    }
    out
}

/// Packed RGB(A) → Gray8 luminance projection (full-range Y' row of the
/// selected primaries; alpha dropped).
fn do_rgb_to_gray(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let rgb24: Vec<u8> = if alpha_in {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
            for i in 0..w {
                out.push(sr[i * 4]);
                out.push(sr[i * 4 + 1]);
                out.push(sr[i * 4 + 2]);
            }
        }
        out
    } else {
        gather_tight(&in_plane.data, in_plane.stride, w * 3, h)
    };
    let mut gray = vec![0u8; w * h];
    yuv::rgb24_to_gray8(&rgb24, &mut gray, w * h, matrix);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: gray,
        }],
    ))
}

// -------------------------------------------------------------------------
// Ya8 (grey + alpha, 2 bytes/pixel).

fn do_ya8_to_gray8(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 2);
        gray::ya8_to_gray8(sr, &mut out[row * w..row * w + w], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}

fn do_gray8_to_ya8(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 2];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        gray::gray8_to_ya8(sr, &mut out[row * w * 2..row * w * 2 + w * 2], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: out,
        }],
    ))
}

fn do_ya8_to_rgb24(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 2);
        gray::ya8_to_rgb24(sr, &mut out[row * w * 3..row * w * 3 + w * 3], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

fn do_ya8_to_rgba(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 2);
        gray::ya8_to_rgba(sr, &mut out[row * w * 4..row * w * 4 + w * 4], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn do_rgb24_to_ya8(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 2];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 3);
        gray::rgb24_to_ya8(sr, &mut out[row * w * 2..row * w * 2 + w * 2], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: out,
        }],
    ))
}

fn do_rgba_to_ya8(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 2];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        gray::rgba_to_ya8(sr, &mut out[row * w * 2..row * w * 2 + w * 2], w);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: out,
        }],
    ))
}

// -------------------------------------------------------------------------
// YUV ↔ Gray8 (luma extraction / neutral-chroma synthesis).

/// Any YUV-family source → `Gray8`: gather the full-resolution luma
/// plane, rescale limited → full range unless the source is a `YuvJ*`
/// family, and drop everything else. Chroma is never read, so this path
/// accepts odd dimensions even on subsampled layouts.
fn do_yuv_luma_to_gray(
    src: &VideoFrame,
    src_info: FrameInfo,
    full_range: bool,
) -> Result<VideoFrame> {
    if src.planes.is_empty() {
        return Err(Error::invalid("pixfmt: YUV source needs a luma plane"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let mut yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    if !full_range {
        yuv::limited_to_full_luma(&mut yp);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: yp,
        }],
    ))
}

/// `Gray8` → planar YUV: gray becomes luma (compressed to the limited
/// 16..=235 range unless the destination is full-range `YuvJ*`), chroma
/// planes are synthesised at the neutral code 128. Dimensions must
/// divide by the destination's chroma grid so the U/V planes are
/// representable.
fn do_gray_to_yuv_planar(
    src: &VideoFrame,
    src_info: FrameInfo,
    wsub: usize,
    hsub: usize,
    full_range: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: Gray8 → subsampled YUV requires dimensions divisible by the subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    let mut yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    if !full_range {
        yuv::full_to_limited_luma(&mut yp);
    }
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
        ],
    ))
}

/// `Gray8` → NV12 / NV21: limited-range luma + one interleaved chroma
/// plane holding the neutral code 128 in every byte — U and V are equal,
/// so NV12 and NV21 receive identical bytes.
fn do_gray_to_nv(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 || h % 2 != 0 {
        return Err(Error::invalid(
            "pixfmt: Gray8 → NV12/NV21 requires even width and height",
        ));
    }
    let cw = w / 2;
    let ch = h / 2;
    let mut yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    yuv::full_to_limited_luma(&mut yp);
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: vec![128u8; cw * ch * 2],
            },
        ],
    ))
}

// -------------------------------------------------------------------------
// YUV ↔ RGB.

fn do_yuv_to_rgb(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    wsub: usize,
    hsub: usize,
    alpha: bool,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid("pixfmt: YUV source needs 3 planes"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    // Subsampled-chroma layouts have no representation for a partial
    // chroma sample, so the luma dimensions must divide evenly by the
    // subsampling factors. Truncating (`w / wsub`) here would size the
    // U/V planes one sample short of what the decoder reads back for the
    // trailing odd luma column/row, indexing past the chroma plane.
    // Reject up front (mirrors the RGB → YUV guard) instead.
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUV → RGB requires dimensions divisible by chroma subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);

    let mut rgb_buf = vec![0u8; w * h * 3];
    match (wsub, hsub) {
        (1, 1) => yuv::yuv444_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 1) => yuv::yuv422_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 2) => yuv::yuv420_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        // 4:1:1 → RGB: upsample U / V from `(w/4) × h` to `w × h` by
        // horizontally broadcasting each chroma sample to the four luma
        // columns it covers, then run the proven 4:4:4 → RGB path on
        // the staged planes. Width must be a multiple of 4 — the
        // `chroma_411_*` helpers `debug_assert!` this; the public-API
        // guard is in `convert()`'s up-front dispatch where 4:1:1
        // sources reject odd luma columns with `Error::Invalid`.
        (4, 1) => {
            if w % 4 != 0 {
                return Err(Error::invalid(
                    "pixfmt: 4:1:1 YUV requires width divisible by 4",
                ));
            }
            let mut u444 = vec![0u8; w * h];
            let mut v444 = vec![0u8; w * h];
            yuv::chroma_411_to_444(&up, &mut u444, w, h);
            yuv::chroma_411_to_444(&vp, &mut v444, w, h);
            yuv::yuv444_to_rgb24(&yp, &u444, &v444, &mut rgb_buf, w, h, matrix);
        }
        _ => return Err(Error::unsupported("pixfmt: unsupported YUV subsampling")),
    }

    if !alpha {
        return Ok(make_frame(
            src,
            vec![VideoPlane {
                stride: w * 3,
                data: rgb_buf,
            }],
        ));
    }
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb_buf[i * 3];
        rgba[i * 4 + 1] = rgb_buf[i * 3 + 1];
        rgba[i * 4 + 2] = rgb_buf[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: rgba,
        }],
    ))
}

fn do_rgb_to_yuv(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    wsub: usize,
    hsub: usize,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: RGB → YUV requires dimensions divisible by subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;

    let in_plane = &src.planes[0];
    // Project to a tight RGB24 buffer.
    let rgb24: Vec<u8> = if alpha_in {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            let row_bytes = w * 4;
            let sr = tight_row(&in_plane.data, in_plane.stride, row, row_bytes);
            for i in 0..w {
                out.push(sr[i * 4]);
                out.push(sr[i * 4 + 1]);
                out.push(sr[i * 4 + 2]);
            }
        }
        out
    } else {
        gather_tight(&in_plane.data, in_plane.stride, w * 3, h)
    };

    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    match (wsub, hsub) {
        (1, 1) => yuv::rgb24_to_yuv444(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        (2, 1) => yuv::rgb24_to_yuv422(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        (2, 2) => yuv::rgb24_to_yuv420(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        // RGB → 4:1:1: encode to 4:4:4 first (luma byte-for-byte from
        // R/G/B, full-resolution chroma from the same per-pixel
        // R/G/B → Cb/Cr matrix), then horizontally box-average chroma
        // down to one sample per four luma columns. Matches what a
        // 4:1:1 JPEG encoder produces from `cjpeg -sample 4x1`.
        (4, 1) => {
            // Width-divisibility was already checked above (w % wsub).
            let mut u444 = vec![0u8; w * h];
            let mut v444 = vec![0u8; w * h];
            yuv::rgb24_to_yuv444(&rgb24, &mut yp, &mut u444, &mut v444, w, h, matrix);
            yuv::chroma_444_to_411(&u444, &mut up, w, h);
            yuv::chroma_444_to_411(&v444, &mut vp, w, h);
        }
        _ => return Err(Error::unsupported("pixfmt: unsupported YUV subsampling")),
    }
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

fn rescale_range(
    src: &VideoFrame,
    src_info: FrameInfo,
    wsub: usize,
    hsub: usize,
    to_full: bool,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid("pixfmt: YuvJ source needs 3 planes"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let cw = w / wsub;
    let ch = h / hsub;
    let mut yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let mut up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let mut vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);
    if to_full {
        yuv::limited_to_full_luma(&mut yp);
        yuv::limited_to_full_chroma(&mut up);
        yuv::limited_to_full_chroma(&mut vp);
    } else {
        yuv::full_to_limited_luma(&mut yp);
        yuv::full_to_limited_chroma(&mut up);
        yuv::full_to_limited_chroma(&mut vp);
    }
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

/// Planar YUV → planar YUV with the luma plane copied byte-for-byte and
/// the chroma planes resampled between two subsampling layouts. Caller
/// supplies the source and destination chroma division factors; the
/// helper routes them through the appropriate `yuv::chroma_*` primitive
/// (`444 ↔ 422`, `444 ↔ 420`, `422 ↔ 420`).
///
/// Dimensions are rejected when the source's chroma-width or height
/// does not divide cleanly into the destination's chroma grid (e.g.
/// odd width on a `4:2:2` source). The dispatch table only registers
/// six pairs — every combination on `(4:2:0, 4:2:2, 4:4:4)` — so the
/// match arms below are total.
fn chroma_resample(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_wsub: usize,
    src_hsub: usize,
    dst_wsub: usize,
    dst_hsub: usize,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid(
            "pixfmt: planar YUV source needs 3 planes (Y, U, V)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    // The dimension constraint is the LCM of the two layouts' chroma
    // grids — both src and dst need to express their U/V planes
    // tightly, so width must be a multiple of `max(src_wsub, dst_wsub)`
    // and height a multiple of `max(src_hsub, dst_hsub)`.
    let wsub_max = src_wsub.max(dst_wsub);
    let hsub_max = src_hsub.max(dst_hsub);
    if w % wsub_max != 0 || h % hsub_max != 0 {
        return Err(Error::invalid(
            "pixfmt: planar YUV chroma resample needs dimensions divisible by the wider subsampling",
        ));
    }
    let src_cw = w / src_wsub;
    let src_ch = h / src_hsub;
    let dst_cw = w / dst_wsub;

    // Luma plane: byte-for-byte copy (gather_tight already drops stride
    // padding).
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    // Chroma planes: gather then route through the appropriate
    // resampler. `u_src` and `v_src` are independent so the helper is
    // invoked twice with identical parameters.
    let u_src = gather_tight(&src.planes[1].data, src.planes[1].stride, src_cw, src_ch);
    let v_src = gather_tight(&src.planes[2].data, src.planes[2].stride, src_cw, src_ch);
    let (u_dst, v_dst) =
        resample_chroma_pair(&u_src, &v_src, w, h, src_wsub, src_hsub, dst_wsub, dst_hsub)?;

    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: dst_cw,
                data: u_dst,
            },
            VideoPlane {
                stride: dst_cw,
                data: v_dst,
            },
        ],
    ))
}

/// Resample a tightly-packed 8-bit chroma pair between two subsampling
/// layouts. Shared core of [`chroma_resample`] (alpha-less planar YUV)
/// and [`yuva_chroma_resample`] (`Yuva*` family, where luma and alpha
/// are copied and only the chroma pair goes through here).
#[allow(clippy::too_many_arguments)]
fn resample_chroma_pair(
    u_src: &[u8],
    v_src: &[u8],
    w: usize,
    h: usize,
    src_wsub: usize,
    src_hsub: usize,
    dst_wsub: usize,
    dst_hsub: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let dst_cw = w / dst_wsub;
    let dst_ch = h / dst_hsub;
    let mut u_dst = vec![0u8; dst_cw * dst_ch];
    let mut v_dst = vec![0u8; dst_cw * dst_ch];

    // The dispatch table registers ordered pairs over `(4:2:0, 4:2:2,
    // 4:4:4, 4:1:1)` — every combination not equal to the identity.
    // Anything else is a registry bug so we route through
    // `Error::unsupported`.
    match (src_wsub, src_hsub, dst_wsub, dst_hsub) {
        // 4:4:4 → 4:2:2  (horizontal pair-average)
        (1, 1, 2, 1) => {
            yuv::chroma_444_to_422(u_src, &mut u_dst, w, h);
            yuv::chroma_444_to_422(v_src, &mut v_dst, w, h);
        }
        // 4:2:2 → 4:4:4  (horizontal duplicate)
        (2, 1, 1, 1) => {
            yuv::chroma_422_to_444(u_src, &mut u_dst, w, h);
            yuv::chroma_422_to_444(v_src, &mut v_dst, w, h);
        }
        // 4:4:4 → 4:2:0  (2×2 box average)
        (1, 1, 2, 2) => {
            yuv::chroma_444_to_420(u_src, &mut u_dst, w, h);
            yuv::chroma_444_to_420(v_src, &mut v_dst, w, h);
        }
        // 4:2:0 → 4:4:4  (2×2 nearest)
        (2, 2, 1, 1) => {
            yuv::chroma_420_to_444(u_src, &mut u_dst, w, h);
            yuv::chroma_420_to_444(v_src, &mut v_dst, w, h);
        }
        // 4:2:2 → 4:2:0  (vertical pair-average; chroma width unchanged)
        (2, 1, 2, 2) => {
            yuv::chroma_422_to_420(u_src, &mut u_dst, w, h);
            yuv::chroma_422_to_420(v_src, &mut v_dst, w, h);
        }
        // 4:2:0 → 4:2:2  (vertical duplicate; chroma width unchanged)
        (2, 2, 2, 1) => {
            yuv::chroma_420_to_422(u_src, &mut u_dst, w, h);
            yuv::chroma_420_to_422(v_src, &mut v_dst, w, h);
        }
        // 4:4:4 → 4:1:1  (horizontal 4-sample box average)
        (1, 1, 4, 1) => {
            yuv::chroma_444_to_411(u_src, &mut u_dst, w, h);
            yuv::chroma_444_to_411(v_src, &mut v_dst, w, h);
        }
        // 4:1:1 → 4:4:4  (horizontal 4× duplicate)
        (4, 1, 1, 1) => {
            yuv::chroma_411_to_444(u_src, &mut u_dst, w, h);
            yuv::chroma_411_to_444(v_src, &mut v_dst, w, h);
        }
        // 4:2:2 → 4:1:1  (horizontal pair-average; vertical unchanged)
        (2, 1, 4, 1) => {
            yuv::chroma_422_to_411(u_src, &mut u_dst, w, h);
            yuv::chroma_422_to_411(v_src, &mut v_dst, w, h);
        }
        // 4:1:1 → 4:2:2  (horizontal pair-duplicate; vertical unchanged)
        (4, 1, 2, 1) => {
            yuv::chroma_411_to_422(u_src, &mut u_dst, w, h);
            yuv::chroma_411_to_422(v_src, &mut v_dst, w, h);
        }
        // 4:2:0 → 4:1:1  (horizontal pair-average + vertical duplicate)
        (2, 2, 4, 1) => {
            yuv::chroma_420_to_411(u_src, &mut u_dst, w, h);
            yuv::chroma_420_to_411(v_src, &mut v_dst, w, h);
        }
        // 4:1:1 → 4:2:0  (horizontal duplicate to 4:2:2 wsub + vertical
        // pair-average)
        (4, 1, 2, 2) => {
            yuv::chroma_411_to_420(u_src, &mut u_dst, w, h);
            yuv::chroma_411_to_420(v_src, &mut v_dst, w, h);
        }
        _ => {
            return Err(Error::unsupported(
                "pixfmt: unregistered planar YUV chroma resample pair",
            ))
        }
    }

    Ok((u_dst, v_dst))
}

/// `Yuva*` → `Yuva*` chroma resample: luma (plane 0) and the
/// full-resolution alpha plane (plane 3) are copied byte-for-byte; the
/// chroma pair is resampled via [`resample_chroma_pair`]. No colour
/// matrix — alpha and luma survive bit-exact.
fn yuva_chroma_resample(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_wsub: usize,
    src_hsub: usize,
    dst_wsub: usize,
    dst_hsub: usize,
) -> Result<VideoFrame> {
    if src.planes.len() < 4 {
        return Err(Error::invalid(
            "pixfmt: Yuva source needs 4 planes (Y, U, V, A)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let wsub_max = src_wsub.max(dst_wsub);
    let hsub_max = src_hsub.max(dst_hsub);
    if w % wsub_max != 0 || h % hsub_max != 0 {
        return Err(Error::invalid(
            "pixfmt: Yuva chroma resample needs dimensions divisible by the wider subsampling",
        ));
    }
    let src_cw = w / src_wsub;
    let src_ch = h / src_hsub;
    let dst_cw = w / dst_wsub;

    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let u_src = gather_tight(&src.planes[1].data, src.planes[1].stride, src_cw, src_ch);
    let v_src = gather_tight(&src.planes[2].data, src.planes[2].stride, src_cw, src_ch);
    let ap = gather_tight(&src.planes[3].data, src.planes[3].stride, w, h);
    let (u_dst, v_dst) =
        resample_chroma_pair(&u_src, &v_src, w, h, src_wsub, src_hsub, dst_wsub, dst_hsub)?;

    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: dst_cw,
                data: u_dst,
            },
            VideoPlane {
                stride: dst_cw,
                data: v_dst,
            },
            VideoPlane {
                stride: w,
                data: ap,
            },
        ],
    ))
}

/// 16-bit planar YUV → 16-bit planar YUV: luma copied word-for-word,
/// chroma resampled at full 16-bit precision through the
/// `yuv::chroma16le_*` primitives. Only the pairs over (4:2:0, 4:2:2,
/// 4:4:4) are registered.
fn chroma_resample16(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_wsub: usize,
    src_hsub: usize,
    dst_wsub: usize,
    dst_hsub: usize,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid(
            "pixfmt: planar YUV source needs 3 planes (Y, U, V)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let wsub_max = src_wsub.max(dst_wsub);
    let hsub_max = src_hsub.max(dst_hsub);
    if w % wsub_max != 0 || h % hsub_max != 0 {
        return Err(Error::invalid(
            "pixfmt: planar YUV chroma resample needs dimensions divisible by the wider subsampling",
        ));
    }
    let src_cw = w / src_wsub;
    let src_ch = h / src_hsub;
    let dst_cw = w / dst_wsub;

    // All planes are 16-bit LE: two bytes per sample.
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w * 2, h);
    let u_src = gather_tight(
        &src.planes[1].data,
        src.planes[1].stride,
        src_cw * 2,
        src_ch,
    );
    let v_src = gather_tight(
        &src.planes[2].data,
        src.planes[2].stride,
        src_cw * 2,
        src_ch,
    );
    let (u_dst, v_dst) =
        resample_chroma16_pair(&u_src, &v_src, w, h, src_wsub, src_hsub, dst_wsub, dst_hsub)?;

    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w * 2,
                data: yp,
            },
            VideoPlane {
                stride: dst_cw * 2,
                data: u_dst,
            },
            VideoPlane {
                stride: dst_cw * 2,
                data: v_dst,
            },
        ],
    ))
}

/// Resample a tightly-packed 16-bit LE chroma pair between two
/// subsampling layouts over (4:2:0, 4:2:2, 4:4:4). The `chroma16le_*`
/// primitives operate on whole LE words, so the same code path serves
/// every LSB-anchored significant width (10 / 12 / 16 bits). Shared by
/// [`chroma_resample16`] and the computed [`planar_family`] op.
#[allow(clippy::too_many_arguments)]
fn resample_chroma16_pair(
    u_src: &[u8],
    v_src: &[u8],
    w: usize,
    h: usize,
    src_wsub: usize,
    src_hsub: usize,
    dst_wsub: usize,
    dst_hsub: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let dst_cw = w / dst_wsub;
    let dst_ch = h / dst_hsub;
    let mut u_dst = vec![0u8; dst_cw * dst_ch * 2];
    let mut v_dst = vec![0u8; dst_cw * dst_ch * 2];

    match (src_wsub, src_hsub, dst_wsub, dst_hsub) {
        (1, 1, 2, 1) => {
            yuv::chroma16le_444_to_422(u_src, &mut u_dst, w, h);
            yuv::chroma16le_444_to_422(v_src, &mut v_dst, w, h);
        }
        (2, 1, 1, 1) => {
            yuv::chroma16le_422_to_444(u_src, &mut u_dst, w, h);
            yuv::chroma16le_422_to_444(v_src, &mut v_dst, w, h);
        }
        (1, 1, 2, 2) => {
            yuv::chroma16le_444_to_420(u_src, &mut u_dst, w, h);
            yuv::chroma16le_444_to_420(v_src, &mut v_dst, w, h);
        }
        (2, 2, 1, 1) => {
            yuv::chroma16le_420_to_444(u_src, &mut u_dst, w, h);
            yuv::chroma16le_420_to_444(v_src, &mut v_dst, w, h);
        }
        (2, 1, 2, 2) => {
            yuv::chroma16le_422_to_420(u_src, &mut u_dst, w, h);
            yuv::chroma16le_422_to_420(v_src, &mut v_dst, w, h);
        }
        (2, 2, 2, 1) => {
            yuv::chroma16le_420_to_422(u_src, &mut u_dst, w, h);
            yuv::chroma16le_420_to_422(v_src, &mut v_dst, w, h);
        }
        _ => {
            return Err(Error::unsupported(
                "pixfmt: unregistered 16-bit planar YUV chroma resample pair",
            ))
        }
    }

    Ok((u_dst, v_dst))
}

// -------------------------------------------------------------------------
// Computed planar-family ops (see `PlanarYuv` / `lookup_computed`).

/// Move one gathered (tight) plane of `count` samples between two
/// planar-family storage depths: 8-bit planes are a byte per sample,
/// deeper planes 16-bit LE words with `bits` significant low bits.
/// Equal depths degrade to a copy (masked for the deep widths); the
/// widen / narrow legs follow the crate-wide MSB-replicate / truncate
/// policy via the shared `yuv::depth_*` primitives.
fn plane_to_depth(src: &[u8], count: usize, src_bits: u32, dst_bits: u32) -> Vec<u8> {
    match (src_bits > 8, dst_bits > 8) {
        (false, false) => src[..count].to_vec(),
        (false, true) => {
            let mut out = vec![0u8; count * 2];
            yuv::depth_up_8_to_le16_plane(src, &mut out, count, dst_bits);
            out
        }
        (true, false) => {
            let mut out = vec![0u8; count];
            yuv::depth_down_le16_plane(src, &mut out, count, src_bits);
            out
        }
        (true, true) => {
            let mut out = vec![0u8; count * 2];
            yuv::depth_rescale_le16_plane(src, &mut out, count, src_bits, dst_bits);
            out
        }
    }
}

/// Resample a tightly-packed chroma pair at the storage depth implied
/// by `bits`: byte planes route through the 8-bit primitives, LE-word
/// planes through the 16-bit ones (identical rounding conventions).
#[allow(clippy::too_many_arguments)]
fn resample_pair_at_depth(
    u_src: &[u8],
    v_src: &[u8],
    w: usize,
    h: usize,
    src_wsub: usize,
    src_hsub: usize,
    dst_wsub: usize,
    dst_hsub: usize,
    bits: u32,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if bits > 8 {
        resample_chroma16_pair(u_src, v_src, w, h, src_wsub, src_hsub, dst_wsub, dst_hsub)
    } else {
        resample_chroma_pair(u_src, v_src, w, h, src_wsub, src_hsub, dst_wsub, dst_hsub)
    }
}

/// A plane of `count` opaque (full-scale) alpha samples at `bits`
/// depth: `0xFF` bytes for 8-bit storage, LE words of `(1 << bits) - 1`
/// for the deep widths.
fn opaque_plane(count: usize, bits: u32) -> Vec<u8> {
    if bits > 8 {
        let full = ((1u32 << bits) - 1) as u16;
        let [lo, hi] = full.to_le_bytes();
        let mut out = vec![0u8; count * 2];
        for word in out.chunks_exact_mut(2) {
            word[0] = lo;
            word[1] = hi;
        }
        out
    } else {
        vec![0xFF; count]
    }
}

/// A plane of `count` neutral chroma samples at `bits` depth: the exact
/// mid-code `1 << (bits - 1)` (128 / 512 / 2048 / 32768).
fn neutral_chroma_plane(count: usize, bits: u32) -> Vec<u8> {
    if bits > 8 {
        let mid = (1u32 << (bits - 1)) as u16;
        let [lo, hi] = mid.to_le_bytes();
        let mut out = vec![0u8; count * 2];
        for word in out.chunks_exact_mut(2) {
            word[0] = lo;
            word[1] = hi;
        }
        out
    } else {
        vec![128; count]
    }
}

/// Computed `PlanarFamily` op: any ordered pair inside the uniform
/// planar YUV(A) family. Luma (and a carried alpha plane) go through a
/// straight depth move; the chroma pair is resampled at the **deeper**
/// of the two depths — widen-then-resample when the destination is
/// deeper, resample-then-narrow otherwise — so no precision is thrown
/// away before an average is taken. Alpha is carried (depth-moved),
/// dropped, or synthesised opaque full-scale per the two descriptors.
fn planar_family(
    src: &VideoFrame,
    src_info: FrameInfo,
    s: PlanarYuv,
    d: PlanarYuv,
) -> Result<VideoFrame> {
    let need = if s.alpha { 4 } else { 3 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: planar YUV(A) source is missing planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let wsub_max = s.wsub.max(d.wsub);
    let hsub_max = s.hsub.max(d.hsub);
    if w % wsub_max != 0 || h % hsub_max != 0 {
        return Err(Error::invalid(
            "pixfmt: planar YUV(A) conversion needs dimensions divisible by the wider subsampling",
        ));
    }
    let scw = w / s.wsub;
    let sch = h / s.hsub;
    let dcw = w / d.wsub;
    let dch = h / d.hsub;
    let sb_src = s.sample_bytes();
    let sb_dst = d.sample_bytes();

    let y_src = gather_tight(&src.planes[0].data, src.planes[0].stride, w * sb_src, h);
    let u_src = gather_tight(&src.planes[1].data, src.planes[1].stride, scw * sb_src, sch);
    let v_src = gather_tight(&src.planes[2].data, src.planes[2].stride, scw * sb_src, sch);

    // Luma: straight depth move, never resampled.
    let yp = plane_to_depth(&y_src, w * h, s.bits, d.bits);

    // Chroma: resample at the deeper of the two depths.
    let (up, vp) = if (s.wsub, s.hsub) == (d.wsub, d.hsub) {
        (
            plane_to_depth(&u_src, scw * sch, s.bits, d.bits),
            plane_to_depth(&v_src, scw * sch, s.bits, d.bits),
        )
    } else if d.bits >= s.bits {
        let u_mid = plane_to_depth(&u_src, scw * sch, s.bits, d.bits);
        let v_mid = plane_to_depth(&v_src, scw * sch, s.bits, d.bits);
        resample_pair_at_depth(&u_mid, &v_mid, w, h, s.wsub, s.hsub, d.wsub, d.hsub, d.bits)?
    } else {
        let (u_mid, v_mid) =
            resample_pair_at_depth(&u_src, &v_src, w, h, s.wsub, s.hsub, d.wsub, d.hsub, s.bits)?;
        (
            plane_to_depth(&u_mid, dcw * dch, s.bits, d.bits),
            plane_to_depth(&v_mid, dcw * dch, s.bits, d.bits),
        )
    };

    let mut planes = vec![
        VideoPlane {
            stride: w * sb_dst,
            data: yp,
        },
        VideoPlane {
            stride: dcw * sb_dst,
            data: up,
        },
        VideoPlane {
            stride: dcw * sb_dst,
            data: vp,
        },
    ];
    if d.alpha {
        let ap = if s.alpha {
            let a_src = gather_tight(&src.planes[3].data, src.planes[3].stride, w * sb_src, h);
            plane_to_depth(&a_src, w * h, s.bits, d.bits)
        } else {
            opaque_plane(w * h, d.bits)
        };
        planes.push(VideoPlane {
            stride: w * sb_dst,
            data: ap,
        });
    }
    Ok(make_frame(src, planes))
}

/// Computed `PlanarFamilyToRgb` op: family member → `Rgb24` / `Rgba`.
/// Deep planes are truncated to 8 bits (crate depth policy) and decoded
/// through the proven 8-bit scalar/SIMD matrix for the source's chroma
/// grid; alpha is interleaved (reduced to 8 bits), synthesised opaque,
/// or dropped.
fn planar_family_to_rgb(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    s: PlanarYuv,
    alpha: bool,
) -> Result<VideoFrame> {
    let need = if s.alpha { 4 } else { 3 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: planar YUV(A) source is missing planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % s.wsub != 0 || h % s.hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUV → RGB requires dimensions divisible by chroma subsampling",
        ));
    }
    let cw = w / s.wsub;
    let ch = h / s.hsub;
    let sb = s.sample_bytes();
    let y_src = gather_tight(&src.planes[0].data, src.planes[0].stride, w * sb, h);
    let u_src = gather_tight(&src.planes[1].data, src.planes[1].stride, cw * sb, ch);
    let v_src = gather_tight(&src.planes[2].data, src.planes[2].stride, cw * sb, ch);
    let yp = plane_to_depth(&y_src, w * h, s.bits, 8);
    let up = plane_to_depth(&u_src, cw * ch, s.bits, 8);
    let vp = plane_to_depth(&v_src, cw * ch, s.bits, 8);

    let mut rgb_buf = vec![0u8; w * h * 3];
    match (s.wsub, s.hsub) {
        (1, 1) => yuv::yuv444_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 1) => yuv::yuv422_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 2) => yuv::yuv420_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        _ => return Err(Error::unsupported("pixfmt: unsupported YUV subsampling")),
    }

    if !alpha {
        return Ok(make_frame(
            src,
            vec![VideoPlane {
                stride: w * 3,
                data: rgb_buf,
            }],
        ));
    }
    let ap = if s.alpha {
        let a_src = gather_tight(&src.planes[3].data, src.planes[3].stride, w * sb, h);
        plane_to_depth(&a_src, w * h, s.bits, 8)
    } else {
        vec![255u8; w * h]
    };
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb_buf[i * 3];
        rgba[i * 4 + 1] = rgb_buf[i * 3 + 1];
        rgba[i * 4 + 2] = rgb_buf[i * 3 + 2];
        rgba[i * 4 + 3] = ap[i];
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: rgba,
        }],
    ))
}

/// Computed `RgbToPlanarFamily` op: `Rgb24` / `Rgba` → family member.
/// Encodes through the proven 8-bit path for the destination's chroma
/// grid, then widens every plane to the family depth (MSB replication —
/// identical bytes to the historical encode-then-`DepthUpYuv` staged
/// route). Alpha is split out of an `Rgba` source (widened) or
/// synthesised opaque full-scale.
fn rgb_to_planar_family(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    d: PlanarYuv,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % d.wsub != 0 || h % d.hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: RGB → YUV requires dimensions divisible by subsampling",
        ));
    }
    let cw = w / d.wsub;
    let ch = h / d.hsub;
    let sb = d.sample_bytes();

    let in_plane = &src.planes[0];
    let mut rgb24: Vec<u8> = Vec::with_capacity(w * h * 3);
    let mut alpha8: Vec<u8> = Vec::new();
    if alpha_in {
        alpha8 = vec![0xFF; w * h];
        for row in 0..h {
            let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
            for i in 0..w {
                rgb24.push(sr[i * 4]);
                rgb24.push(sr[i * 4 + 1]);
                rgb24.push(sr[i * 4 + 2]);
                alpha8[row * w + i] = sr[i * 4 + 3];
            }
        }
    } else {
        rgb24 = gather_tight(&in_plane.data, in_plane.stride, w * 3, h);
    }

    let mut yp8 = vec![0u8; w * h];
    let mut up8 = vec![0u8; cw * ch];
    let mut vp8 = vec![0u8; cw * ch];
    match (d.wsub, d.hsub) {
        (1, 1) => yuv::rgb24_to_yuv444(&rgb24, &mut yp8, &mut up8, &mut vp8, w, h, matrix),
        (2, 1) => yuv::rgb24_to_yuv422(&rgb24, &mut yp8, &mut up8, &mut vp8, w, h, matrix),
        (2, 2) => yuv::rgb24_to_yuv420(&rgb24, &mut yp8, &mut up8, &mut vp8, w, h, matrix),
        _ => return Err(Error::unsupported("pixfmt: unsupported YUV subsampling")),
    }

    let mut planes = vec![
        VideoPlane {
            stride: w * sb,
            data: plane_to_depth(&yp8, w * h, 8, d.bits),
        },
        VideoPlane {
            stride: cw * sb,
            data: plane_to_depth(&up8, cw * ch, 8, d.bits),
        },
        VideoPlane {
            stride: cw * sb,
            data: plane_to_depth(&vp8, cw * ch, 8, d.bits),
        },
    ];
    if d.alpha {
        let ap = if alpha_in {
            plane_to_depth(&alpha8, w * h, 8, d.bits)
        } else {
            opaque_plane(w * h, d.bits)
        };
        planes.push(VideoPlane {
            stride: w * sb,
            data: ap,
        });
    }
    Ok(make_frame(src, planes))
}

/// Computed `PlanarFamilyToGray` op: luma extraction from any family
/// member. Deep luma is truncated to 8 bits, then rescaled from the
/// family's limited range to full-range `Gray8`; chroma and alpha are
/// dropped. Only the luma plane is touched, so odd dimensions are fine
/// even on subsampled sources (mirrors the 8-bit `YuvLumaToGray` rows).
fn planar_family_to_gray(
    src: &VideoFrame,
    src_info: FrameInfo,
    s: PlanarYuv,
) -> Result<VideoFrame> {
    if src.planes.is_empty() {
        return Err(Error::invalid("pixfmt: YUV source needs a luma plane"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let sb = s.sample_bytes();
    let y_src = gather_tight(&src.planes[0].data, src.planes[0].stride, w * sb, h);
    let mut yp = plane_to_depth(&y_src, w * h, s.bits, 8);
    yuv::limited_to_full_luma(&mut yp);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: yp,
        }],
    ))
}

/// Computed `GrayToPlanarFamily` op: `Gray8` → any family member. The
/// gray plane becomes luma (full → limited range at 8 bits, then
/// widened); chroma is synthesised at the exact neutral mid-code
/// `1 << (bits - 1)`; alpha (when the destination carries it) is
/// synthesised opaque full-scale.
fn gray_to_planar_family(
    src: &VideoFrame,
    src_info: FrameInfo,
    d: PlanarYuv,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % d.wsub != 0 || h % d.hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: Gray8 → subsampled YUV requires dimensions divisible by the subsampling",
        ));
    }
    let cw = w / d.wsub;
    let ch = h / d.hsub;
    let sb = d.sample_bytes();
    let mut luma8 = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    yuv::full_to_limited_luma(&mut luma8);
    let mut planes = vec![
        VideoPlane {
            stride: w * sb,
            data: plane_to_depth(&luma8, w * h, 8, d.bits),
        },
        VideoPlane {
            stride: cw * sb,
            data: neutral_chroma_plane(cw * ch, d.bits),
        },
        VideoPlane {
            stride: cw * sb,
            data: neutral_chroma_plane(cw * ch, d.bits),
        },
    ];
    if d.alpha {
        planes.push(VideoPlane {
            stride: w * sb,
            data: opaque_plane(w * h, d.bits),
        });
    }
    Ok(make_frame(src, planes))
}

/// High-precision planar YUV (16-bit LE, `bits` significant) → 8-bit
/// planar YUV. Luma and both chroma planes are reduced by
/// [`yuv::depth_down_le16_plane`]; the chroma subsampling layout
/// (`wsub` / `hsub`) is preserved unchanged. Width / height must divide
/// cleanly into the chroma grid, else `Error::invalid`.
fn do_yuv_depth_down(
    src: &VideoFrame,
    src_info: FrameInfo,
    wsub: usize,
    hsub: usize,
    bits: u32,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid(
            "pixfmt: high-bit YUV source needs 3 planes (Y, U, V)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUV bit-depth conversion needs dimensions divisible by the subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    // Source planes are 16-bit LE: each sample is two bytes wide.
    let y_src = gather_tight(&src.planes[0].data, src.planes[0].stride, w * 2, h);
    let u_src = gather_tight(&src.planes[1].data, src.planes[1].stride, cw * 2, ch);
    let v_src = gather_tight(&src.planes[2].data, src.planes[2].stride, cw * 2, ch);
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    yuv::depth_down_le16_plane(&y_src, &mut yp, w * h, bits);
    yuv::depth_down_le16_plane(&u_src, &mut up, cw * ch, bits);
    yuv::depth_down_le16_plane(&v_src, &mut vp, cw * ch, bits);
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

/// 8-bit planar YUV → high-precision planar YUV (16-bit LE, `bits`
/// significant). The inverse of [`do_yuv_depth_down`]: each plane is
/// widened by [`yuv::depth_up_8_to_le16_plane`] (8-bit value in the high
/// bits with MSB replication into the low slack). Subsampling preserved.
fn do_yuv_depth_up(
    src: &VideoFrame,
    src_info: FrameInfo,
    wsub: usize,
    hsub: usize,
    bits: u32,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid(
            "pixfmt: planar YUV source needs 3 planes (Y, U, V)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUV bit-depth conversion needs dimensions divisible by the subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    let y_src = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let u_src = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let v_src = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);
    let mut yp = vec![0u8; w * h * 2];
    let mut up = vec![0u8; cw * ch * 2];
    let mut vp = vec![0u8; cw * ch * 2];
    yuv::depth_up_8_to_le16_plane(&y_src, &mut yp, w * h, bits);
    yuv::depth_up_8_to_le16_plane(&u_src, &mut up, cw * ch, bits);
    yuv::depth_up_8_to_le16_plane(&v_src, &mut vp, cw * ch, bits);
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w * 2,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: up,
            },
            VideoPlane {
                stride: cw * 2,
                data: vp,
            },
        ],
    ))
}

/// Cross-depth planar YUV (10 ↔ 12 bit, 16-bit LE storage on both
/// sides): every plane goes through [`yuv::depth_rescale_le16_plane`];
/// subsampling is preserved.
fn do_yuv_depth_rescale(
    src: &VideoFrame,
    src_info: FrameInfo,
    wsub: usize,
    hsub: usize,
    src_bits: u32,
    dst_bits: u32,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid(
            "pixfmt: high-bit YUV source needs 3 planes (Y, U, V)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUV bit-depth conversion needs dimensions divisible by the subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    let y_src = gather_tight(&src.planes[0].data, src.planes[0].stride, w * 2, h);
    let u_src = gather_tight(&src.planes[1].data, src.planes[1].stride, cw * 2, ch);
    let v_src = gather_tight(&src.planes[2].data, src.planes[2].stride, cw * 2, ch);
    let mut yp = vec![0u8; w * h * 2];
    let mut up = vec![0u8; cw * ch * 2];
    let mut vp = vec![0u8; cw * ch * 2];
    yuv::depth_rescale_le16_plane(&y_src, &mut yp, w * h, src_bits, dst_bits);
    yuv::depth_rescale_le16_plane(&u_src, &mut up, cw * ch, src_bits, dst_bits);
    yuv::depth_rescale_le16_plane(&v_src, &mut vp, cw * ch, src_bits, dst_bits);
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w * 2,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: up,
            },
            VideoPlane {
                stride: cw * 2,
                data: vp,
            },
        ],
    ))
}

// -------------------------------------------------------------------------
// Deep grayscale (Gray10Le / Gray12Le / Gray16Le) storage-width ladder.

/// `Gray10Le` / `Gray12Le` → `Gray8` (keep the top 8 significant bits).
fn do_gray_depth_down8(src: &VideoFrame, src_info: FrameInfo, bits: u32) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let tight = gather_tight(&in_plane.data, in_plane.stride, w * 2, h);
    let mut out = vec![0u8; w * h];
    yuv::depth_down_le16_plane(&tight, &mut out, w * h, bits);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}

/// `Gray8` → `Gray10Le` / `Gray12Le` (MSB-replicated widen; exact
/// inverse of [`do_gray_depth_down8`]).
fn do_gray_depth_up8(src: &VideoFrame, src_info: FrameInfo, bits: u32) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let tight = gather_tight(&in_plane.data, in_plane.stride, w, h);
    let mut out = vec![0u8; w * h * 2];
    yuv::depth_up_8_to_le16_plane(&tight, &mut out, w * h, bits);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: out,
        }],
    ))
}

/// Deep grayscale ↔ deep grayscale (10 ↔ 12 ↔ 16) storage rescale.
fn do_gray_depth_rescale(
    src: &VideoFrame,
    src_info: FrameInfo,
    src_bits: u32,
    dst_bits: u32,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let tight = gather_tight(&in_plane.data, in_plane.stride, w * 2, h);
    let mut out = vec![0u8; w * h * 2];
    yuv::depth_rescale_le16_plane(&tight, &mut out, w * h, src_bits, dst_bits);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: out,
        }],
    ))
}

fn nv_to_yuv420p(src: &VideoFrame, src_info: FrameInfo, is_nv12: bool) -> Result<VideoFrame> {
    if src.planes.len() < 2 {
        return Err(Error::invalid("pixfmt: NV source needs 2 planes"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let uv = gather_tight(&src.planes[1].data, src.planes[1].stride, cw * 2, ch);
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    if is_nv12 {
        yuv::nv12_uv_split(&uv, &mut up, &mut vp, cw, ch);
    } else {
        yuv::nv21_vu_split(&uv, &mut up, &mut vp, cw, ch);
    }
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

fn yuv420p_to_nv(src: &VideoFrame, src_info: FrameInfo, is_nv12: bool) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid("pixfmt: Yuv420P source needs 3 planes"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);
    let mut uv = vec![0u8; cw * ch * 2];
    if is_nv12 {
        yuv::nv12_uv_merge(&up, &vp, &mut uv, cw, ch);
    } else {
        yuv::nv21_vu_merge(&up, &vp, &mut uv, cw, ch);
    }
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: uv,
            },
        ],
    ))
}

/// NV12 / NV21 → packed RGB. Fused path: deinterleave the UV plane
/// into transient U / V planes, then run the proven planar 4:2:0 →
/// RGB decoder. Saves the caller a `Nv → Yuv420P → Rgb` two-step.
///
/// 4:2:0 subsampling pins width and height to multiples of 2; an odd
/// dimension is rejected with `Error::Invalid` (the format has no
/// representation for a half-pixel row or column).
fn nv_to_rgb(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    is_nv12: bool,
    alpha: bool,
) -> Result<VideoFrame> {
    if src.planes.len() < 2 {
        return Err(Error::invalid("pixfmt: NV source needs 2 planes"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 || h % 2 != 0 {
        return Err(Error::invalid(
            "pixfmt: NV12/NV21 requires even width and height",
        ));
    }
    let cw = w / 2;
    let ch = h / 2;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let uv = gather_tight(&src.planes[1].data, src.planes[1].stride, cw * 2, ch);
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    if is_nv12 {
        yuv::nv12_uv_split(&uv, &mut up, &mut vp, cw, ch);
    } else {
        yuv::nv21_vu_split(&uv, &mut up, &mut vp, cw, ch);
    }
    let mut rgb_buf = vec![0u8; w * h * 3];
    yuv::yuv420_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix);
    if !alpha {
        return Ok(make_frame(
            src,
            vec![VideoPlane {
                stride: w * 3,
                data: rgb_buf,
            }],
        ));
    }
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb_buf[i * 3];
        rgba[i * 4 + 1] = rgb_buf[i * 3 + 1];
        rgba[i * 4 + 2] = rgb_buf[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: rgba,
        }],
    ))
}

/// Packed RGB → NV12 / NV21. Reuses the planar 4:2:0 encoder, then
/// interleaves the resulting (U, V) planes back into the NV layout.
fn rgb_to_nv(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    is_nv12: bool,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 || h % 2 != 0 {
        return Err(Error::invalid(
            "pixfmt: NV12/NV21 requires even width and height",
        ));
    }
    let cw = w / 2;
    let ch = h / 2;
    let in_plane = &src.planes[0];
    let rgb24: Vec<u8> = if alpha_in {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            let row_bytes = w * 4;
            let sr = tight_row(&in_plane.data, in_plane.stride, row, row_bytes);
            for i in 0..w {
                out.push(sr[i * 4]);
                out.push(sr[i * 4 + 1]);
                out.push(sr[i * 4 + 2]);
            }
        }
        out
    } else {
        gather_tight(&in_plane.data, in_plane.stride, w * 3, h)
    };
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    yuv::rgb24_to_yuv420(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix);
    let mut uv = vec![0u8; cw * ch * 2];
    if is_nv12 {
        yuv::nv12_uv_merge(&up, &vp, &mut uv, cw, ch);
    } else {
        yuv::nv21_vu_merge(&up, &vp, &mut uv, cw, ch);
    }
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: uv,
            },
        ],
    ))
}

// -------------------------------------------------------------------------
// Packed 4:2:2 (YUYV / UYVY).
//
// Even-width frames are a hard requirement: the format pairs two luma
// samples per chroma pair, so an odd width has no representation. We
// return Error::Invalid rather than silently truncating.

fn packed422_to_yuv422p(
    src: &VideoFrame,
    src_info: FrameInfo,
    is_yuyv: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 {
        return Err(Error::invalid("pixfmt: packed 4:2:2 requires even width"));
    }
    let cw = w / 2;
    let in_plane = &src.planes[0];
    let packed = gather_tight(&in_plane.data, in_plane.stride, w * 2, h);
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * h];
    let mut vp = vec![0u8; cw * h];
    if is_yuyv {
        yuv::yuyv422_to_yuv422p(&packed, &mut yp, &mut up, &mut vp, w, h);
    } else {
        yuv::uyvy422_to_yuv422p(&packed, &mut yp, &mut up, &mut vp, w, h);
    }
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

fn yuv422p_to_packed422(
    src: &VideoFrame,
    src_info: FrameInfo,
    is_yuyv: bool,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid("pixfmt: Yuv422P source needs 3 planes"));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 {
        return Err(Error::invalid("pixfmt: packed 4:2:2 requires even width"));
    }
    let cw = w / 2;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, h);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, h);
    let mut packed = vec![0u8; w * h * 2];
    if is_yuyv {
        yuv::yuv422p_to_yuyv422(&yp, &up, &vp, &mut packed, w, h);
    } else {
        yuv::yuv422p_to_uyvy422(&yp, &up, &vp, &mut packed, w, h);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: packed,
        }],
    ))
}

fn packed422_swap(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 {
        return Err(Error::invalid("pixfmt: packed 4:2:2 requires even width"));
    }
    let in_plane = &src.planes[0];
    let mut packed = gather_tight(&in_plane.data, in_plane.stride, w * 2, h);
    yuv::yuyv_uyvy_swap(&mut packed);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: packed,
        }],
    ))
}

fn packed422_to_rgb(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    is_yuyv: bool,
    alpha: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 {
        return Err(Error::invalid("pixfmt: packed 4:2:2 requires even width"));
    }
    let cw = w / 2;
    let in_plane = &src.planes[0];
    let packed = gather_tight(&in_plane.data, in_plane.stride, w * 2, h);
    // Reuse the planar 4:2:2 → RGB path: deinterleave first, then go
    // through the proven scalar/SIMD planar decoder.
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * h];
    let mut vp = vec![0u8; cw * h];
    if is_yuyv {
        yuv::yuyv422_to_yuv422p(&packed, &mut yp, &mut up, &mut vp, w, h);
    } else {
        yuv::uyvy422_to_yuv422p(&packed, &mut yp, &mut up, &mut vp, w, h);
    }
    let mut rgb_buf = vec![0u8; w * h * 3];
    yuv::yuv422_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix);
    if !alpha {
        return Ok(make_frame(
            src,
            vec![VideoPlane {
                stride: w * 3,
                data: rgb_buf,
            }],
        ));
    }
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb_buf[i * 3];
        rgba[i * 4 + 1] = rgb_buf[i * 3 + 1];
        rgba[i * 4 + 2] = rgb_buf[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: rgba,
        }],
    ))
}

fn rgb_to_packed422(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    is_yuyv: bool,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % 2 != 0 {
        return Err(Error::invalid("pixfmt: packed 4:2:2 requires even width"));
    }
    let cw = w / 2;
    let in_plane = &src.planes[0];
    // Project the source to a tight RGB24 buffer first — matches the
    // existing planar 4:2:2 encode path.
    let rgb24: Vec<u8> = if alpha_in {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            let row_bytes = w * 4;
            let sr = tight_row(&in_plane.data, in_plane.stride, row, row_bytes);
            for i in 0..w {
                out.push(sr[i * 4]);
                out.push(sr[i * 4 + 1]);
                out.push(sr[i * 4 + 2]);
            }
        }
        out
    } else {
        gather_tight(&in_plane.data, in_plane.stride, w * 3, h)
    };
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * h];
    let mut vp = vec![0u8; cw * h];
    yuv::rgb24_to_yuv422(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix);
    let mut packed = vec![0u8; w * h * 2];
    if is_yuyv {
        yuv::yuv422p_to_yuyv422(&yp, &up, &vp, &mut packed, w, h);
    } else {
        yuv::yuv422p_to_uyvy422(&yp, &up, &vp, &mut packed, w, h);
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 2,
            data: packed,
        }],
    ))
}

// -------------------------------------------------------------------------
// Palette.

/// `Pal8` → packed RGB(A). The colour table is resolved in priority
/// order:
///
/// 1. the frame's own **palette side-channel** (a trailing `stride == 0`
///    plane attached via `VideoFrame::set_palette` — packed 3-byte RGB
///    entries, expanded here with opaque alpha), which is per-frame
///    ground truth from a decoder;
/// 2. the caller-supplied `ConvertOptions::palette` as before;
/// 3. neither present → `Error::Invalid`.
fn pal8_to_rgb(
    src: &VideoFrame,
    src_info: FrameInfo,
    opts: &ConvertOptions,
    alpha: bool,
) -> Result<VideoFrame> {
    // Frame-attached palette wins over the options bundle: it travels
    // with the frame it describes.
    let side_channel = src.palette().map(|raw| Palette {
        colors: raw
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2], 255])
            .collect(),
    });
    let palette =
        match (&side_channel, &opts.palette) {
            (Some(p), _) => p,
            (None, Some(p)) => p,
            (None, None) => return Err(Error::invalid(
                "pixfmt: Pal8 → RGB requires a frame-attached palette or ConvertOptions.palette",
            )),
        };
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    if alpha {
        let mut out = vec![0u8; w * h * 4];
        for row in 0..h {
            let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
            pal8::expand_row_to_rgba(sr, &mut out[row * w * 4..row * w * 4 + w * 4], palette, w);
        }
        Ok(make_frame(
            src,
            vec![VideoPlane {
                stride: w * 4,
                data: out,
            }],
        ))
    } else {
        let mut out = vec![0u8; w * h * 3];
        for row in 0..h {
            let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
            pal8::expand_row_to_rgb24(sr, &mut out[row * w * 3..row * w * 3 + w * 3], palette, w);
        }
        Ok(make_frame(
            src,
            vec![VideoPlane {
                stride: w * 3,
                data: out,
            }],
        ))
    }
}

/// Packed RGB(A) → `Pal8`. Quantises against `ConvertOptions::palette`
/// (required) and **attaches the colour table to the output frame** as
/// the palette side-channel (`VideoFrame::set_palette`, packed 3-byte
/// RGB entries), so the produced `Pal8` frame is self-describing: a
/// later `Pal8 → RGB` expansion needs no options bundle.
fn rgb_to_pal8(
    src: &VideoFrame,
    src_info: FrameInfo,
    opts: &ConvertOptions,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let palette = opts
        .palette
        .as_ref()
        .ok_or_else(|| Error::invalid("pixfmt: RGB → Pal8 requires ConvertOptions.palette"))?;
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h];
    if alpha_in {
        let tight = gather_tight(&in_plane.data, in_plane.stride, w * 4, h);
        pal8::quantise_rgba_to_pal8(&tight, &mut out, w, h, palette, opts.dither);
    } else {
        let tight = gather_tight(&in_plane.data, in_plane.stride, w * 3, h);
        pal8::quantise_rgb24_to_pal8(&tight, &mut out, w, h, palette, opts.dither);
    }
    // Attach the table that was actually used so the frame carries its
    // own colour meaning (side-channel format: 3-byte RGB per entry;
    // the Palette's alpha column is not representable there and is
    // dropped — consumers that need per-entry alpha keep the Palette).
    let side_channel: Vec<u8> = palette
        .colors
        .iter()
        .flat_map(|c| [c[0], c[1], c[2]])
        .collect();
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    )
    .with_palette(side_channel))
}

// -------------------------------------------------------------------------
// CMYK.

// -------------------------------------------------------------------------
// Yuva420P / Yuva422P / Yuva444P — planar YUV with an additional
// full-resolution alpha plane (plane 3 is `w × h` regardless of the
// chroma grid). The YUV planes (Y, U, V) are byte-identical to the
// alpha-less sibling, so every conversion path here borrows the
// existing planar encoder/decoder for the format's chroma grid and only
// differs in how the trailing alpha plane is created, dropped, or
// carried through.

/// Alpha-less planar YUV → the `Yuva*` sibling: copy Y / U / V verbatim
/// and append a full `w × h` plane of `0xFF` (opaque) alpha.
/// Tight-strided output.
fn do_yuv_to_yuva(
    src: &VideoFrame,
    src_info: FrameInfo,
    wsub: usize,
    hsub: usize,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid(
            "pixfmt: planar YUV source needs 3 planes (Y, U, V)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUV → YUVA requires dimensions divisible by the chroma subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);
    let ap = vec![0xFFu8; w * h];
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
            VideoPlane {
                stride: w,
                data: ap,
            },
        ],
    ))
}

/// `Yuva*` → the alpha-less planar sibling: copy the leading three
/// planes; drop alpha.
fn do_yuva_to_yuv(
    src: &VideoFrame,
    src_info: FrameInfo,
    wsub: usize,
    hsub: usize,
) -> Result<VideoFrame> {
    if src.planes.len() < 4 {
        return Err(Error::invalid(
            "pixfmt: Yuva source needs 4 planes (Y, U, V, A)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUVA → YUV requires dimensions divisible by the chroma subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);
    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

/// `Yuva*` → `Rgb24` / `Rgba`: decode the YUV part through the existing
/// planar scalar/SIMD path for the format's chroma grid, then either
/// drop the source's alpha plane (`alpha = false`, output `Rgb24`) or
/// interleave it into the destination's fourth byte (`alpha = true`,
/// output `Rgba`).
fn do_yuva_to_rgb(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    wsub: usize,
    hsub: usize,
    alpha: bool,
) -> Result<VideoFrame> {
    if src.planes.len() < 4 {
        return Err(Error::invalid(
            "pixfmt: Yuva source needs 4 planes (Y, U, V, A)",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: YUVA → RGB requires dimensions divisible by the chroma subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);

    let mut rgb_buf = vec![0u8; w * h * 3];
    match (wsub, hsub) {
        (1, 1) => yuv::yuv444_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 1) => yuv::yuv422_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 2) => yuv::yuv420_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        _ => return Err(Error::unsupported("pixfmt: unsupported YUVA subsampling")),
    }

    if !alpha {
        return Ok(make_frame(
            src,
            vec![VideoPlane {
                stride: w * 3,
                data: rgb_buf,
            }],
        ));
    }

    // Gather the alpha plane at luma resolution, then interleave into
    // the RGBA destination at the fourth byte of each pixel.
    let ap = gather_tight(&src.planes[3].data, src.planes[3].stride, w, h);
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb_buf[i * 3];
        rgba[i * 4 + 1] = rgb_buf[i * 3 + 1];
        rgba[i * 4 + 2] = rgb_buf[i * 3 + 2];
        rgba[i * 4 + 3] = ap[i];
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * 4,
            data: rgba,
        }],
    ))
}

/// `Rgb24` / `Rgba` → `Yuva*`: encode planar YUV through the existing
/// path for the destination's chroma grid, then either synthesise an
/// opaque alpha plane (`alpha_in = false`, source is `Rgb24`) or split
/// the source's alpha out into the trailing plane (`alpha_in = true`,
/// source is `Rgba`).
fn do_rgb_to_yuva(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    wsub: usize,
    hsub: usize,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: RGB → YUVA requires dimensions divisible by the chroma subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;

    let in_plane = &src.planes[0];
    // Tight RGB24 + alpha plane (full resolution; opaque if input is Rgb24).
    let mut rgb24: Vec<u8> = Vec::with_capacity(w * h * 3);
    let mut ap: Vec<u8> = vec![0xFFu8; w * h];
    if alpha_in {
        for row in 0..h {
            let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
            for i in 0..w {
                rgb24.push(sr[i * 4]);
                rgb24.push(sr[i * 4 + 1]);
                rgb24.push(sr[i * 4 + 2]);
                ap[row * w + i] = sr[i * 4 + 3];
            }
        }
    } else {
        rgb24 = gather_tight(&in_plane.data, in_plane.stride, w * 3, h);
        // ap stays opaque (all 0xFF).
    }

    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    match (wsub, hsub) {
        (1, 1) => yuv::rgb24_to_yuv444(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        (2, 1) => yuv::rgb24_to_yuv422(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        (2, 2) => yuv::rgb24_to_yuv420(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        _ => return Err(Error::unsupported("pixfmt: unsupported YUVA subsampling")),
    }

    Ok(make_frame(
        src,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
            VideoPlane {
                stride: w,
                data: ap,
            },
        ],
    ))
}

// -------------------------------------------------------------------------
// Planar GBR(A) ↔ packed deep RGB.
//
// GBR(A) carries RGB as separate planes in G, B, R(, A) order (the plane
// ordering documented on the oxideav-core `Gbrp*Le` / `Gbrap*Le`
// variants). Each sample is a 16-bit little-endian word with only the
// low `bits` (10 / 12 / 14) significant; the high bits are zero. The
// packed `Rgb48Le` / `Rgba64Le` targets store R, G, B(, A) as
// consecutive 16-bit little-endian words using the full 16-bit range.
//
// The conversion is therefore a pure plane reorder + a bit-significance
// rescale: shift each `bits`-significant sample left by `16 - bits` on
// the way to the 16-bit packed word, and right by `16 - bits` on the way
// back. This is bit-layout normalisation only — there is no colour
// matrix and no `ColorSpace` knob applies.

/// Read a little-endian 16-bit word at byte offset `off`.
#[inline]
fn rd16le(buf: &[u8], off: usize) -> u16 {
    (buf[off] as u16) | ((buf[off + 1] as u16) << 8)
}

/// Write `v` as a little-endian 16-bit word at byte offset `off`.
#[inline]
fn wr16le(buf: &mut [u8], off: usize, v: u16) {
    buf[off] = (v & 0xFF) as u8;
    buf[off + 1] = (v >> 8) as u8;
}

/// Planar GBR(A) → packed `Rgb48Le` / `Rgba64Le`.
fn do_gbr_to_packed_deep(
    src: &VideoFrame,
    src_info: FrameInfo,
    bits: u8,
    alpha_in: bool,
    alpha_out: bool,
) -> Result<VideoFrame> {
    let need = if alpha_in { 4 } else { 3 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: GBR(A) source needs G, B, R(, A) planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let shift = 16 - bits as u32;
    // GBR plane order: G=0, B=1, R=2, A=3. Each plane is `w` 16-bit words
    // per row (2 bytes each).
    let g = gather_tight(&src.planes[0].data, src.planes[0].stride, w * 2, h);
    let b = gather_tight(&src.planes[1].data, src.planes[1].stride, w * 2, h);
    let r = gather_tight(&src.planes[2].data, src.planes[2].stride, w * 2, h);
    // The alpha plane is only consulted when the packed target carries
    // it; a surplus source alpha is dropped.
    let a = if alpha_in && alpha_out {
        Some(gather_tight(
            &src.planes[3].data,
            src.planes[3].stride,
            w * 2,
            h,
        ))
    } else {
        None
    };
    let comps = if alpha_out { 4 } else { 3 };
    let mut out = vec![0u8; w * h * comps * 2];
    for i in 0..w * h {
        let rv = rd16le(&r, i * 2) << shift;
        let gv = rd16le(&g, i * 2) << shift;
        let bv = rd16le(&b, i * 2) << shift;
        let base = i * comps * 2;
        wr16le(&mut out, base, rv);
        wr16le(&mut out, base + 2, gv);
        wr16le(&mut out, base + 4, bv);
        if alpha_out {
            // Carry the source alpha (shifted like the colour words) or
            // synthesise the packed format's true opaque full-scale.
            let av = match &a {
                Some(a) => rd16le(a, i * 2) << shift,
                None => 0xFFFF,
            };
            wr16le(&mut out, base + 6, av);
        }
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * comps * 2,
            data: out,
        }],
    ))
}

/// Planar GBR(A) → 8-bit packed `Rgb24` / `Rgba`: reorder the G, B, R(, A)
/// planes into packed byte order while keeping the top 8 of each sample's
/// `bits` significant bits (truncation, consistent with the depth ladder).
fn do_gbr_to_packed8(
    src: &VideoFrame,
    src_info: FrameInfo,
    bits: u32,
    alpha: bool,
) -> Result<VideoFrame> {
    let need = if alpha { 4 } else { 3 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: GBR(A) source needs G, B, R(, A) planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let shift = bits - 8;
    let mask: u16 = ((1u32 << bits) - 1) as u16;
    let g = gather_tight(&src.planes[0].data, src.planes[0].stride, w * 2, h);
    let b = gather_tight(&src.planes[1].data, src.planes[1].stride, w * 2, h);
    let r = gather_tight(&src.planes[2].data, src.planes[2].stride, w * 2, h);
    let a = if alpha {
        Some(gather_tight(
            &src.planes[3].data,
            src.planes[3].stride,
            w * 2,
            h,
        ))
    } else {
        None
    };
    let comps = if alpha { 4 } else { 3 };
    let mut out = vec![0u8; w * h * comps];
    for i in 0..w * h {
        let base = i * comps;
        out[base] = ((rd16le(&r, i * 2) & mask) >> shift) as u8;
        out[base + 1] = ((rd16le(&g, i * 2) & mask) >> shift) as u8;
        out[base + 2] = ((rd16le(&b, i * 2) & mask) >> shift) as u8;
        if let Some(a) = &a {
            out[base + 3] = ((rd16le(a, i * 2) & mask) >> shift) as u8;
        }
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * comps,
            data: out,
        }],
    ))
}

/// 8-bit packed `Rgb24` / `Rgba` → planar GBR(A): split into planes and
/// widen each byte to `bits` significant bits with MSB replication (the
/// exact inverse of [`do_gbr_to_packed8`]).
fn do_packed8_to_gbr(
    src: &VideoFrame,
    src_info: FrameInfo,
    bits: u32,
    alpha: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let shift = bits - 8;
    let comps = if alpha { 4 } else { 3 };
    let in_plane = &src.planes[0];
    let packed = gather_tight(&in_plane.data, in_plane.stride, w * comps, h);
    let widen = |v: u8| -> u16 {
        let v = v as u32;
        (((v << shift) | (v >> (8 - shift))) & ((1u32 << bits) - 1)) as u16
    };
    let mut g = vec![0u8; w * h * 2];
    let mut b = vec![0u8; w * h * 2];
    let mut r = vec![0u8; w * h * 2];
    let mut a = if alpha {
        vec![0u8; w * h * 2]
    } else {
        Vec::new()
    };
    for i in 0..w * h {
        let base = i * comps;
        wr16le(&mut r, i * 2, widen(packed[base]));
        wr16le(&mut g, i * 2, widen(packed[base + 1]));
        wr16le(&mut b, i * 2, widen(packed[base + 2]));
        if alpha {
            wr16le(&mut a, i * 2, widen(packed[base + 3]));
        }
    }
    let mut planes = vec![
        VideoPlane {
            stride: w * 2,
            data: g,
        },
        VideoPlane {
            stride: w * 2,
            data: b,
        },
        VideoPlane {
            stride: w * 2,
            data: r,
        },
    ];
    if alpha {
        planes.push(VideoPlane {
            stride: w * 2,
            data: a,
        });
    }
    Ok(make_frame(src, planes))
}

/// Byte-tier planar GBR(A) (`Gbrp8` / `Gbrap8`) → packed `Rgb24` /
/// `Rgba`: a pure plane reorder — the G, B, R(, A) byte planes
/// interleave into packed R, G, B(, A) order with no depth math, so the
/// conversion is bit-exact and (for matched alpha flags) self-inverse
/// with [`do_packed8_to_gbr8`]. A missing alpha is synthesised opaque
/// 255; a surplus source alpha plane is dropped.
fn do_gbr8_to_packed8(
    src: &VideoFrame,
    src_info: FrameInfo,
    alpha_in: bool,
    alpha_out: bool,
) -> Result<VideoFrame> {
    let need = if alpha_in { 4 } else { 3 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: GBR(A) source needs G, B, R(, A) planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let g = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let b = gather_tight(&src.planes[1].data, src.planes[1].stride, w, h);
    let r = gather_tight(&src.planes[2].data, src.planes[2].stride, w, h);
    let a = if alpha_in && alpha_out {
        Some(gather_tight(
            &src.planes[3].data,
            src.planes[3].stride,
            w,
            h,
        ))
    } else {
        None
    };
    let comps = if alpha_out { 4 } else { 3 };
    let mut out = vec![0u8; w * h * comps];
    for i in 0..w * h {
        let base = i * comps;
        out[base] = r[i];
        out[base + 1] = g[i];
        out[base + 2] = b[i];
        if alpha_out {
            out[base + 3] = a.as_ref().map_or(255, |a| a[i]);
        }
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * comps,
            data: out,
        }],
    ))
}

/// Packed `Rgb24` / `Rgba` → byte-tier planar GBR(A): the inverse plane
/// split of [`do_gbr8_to_packed8`] — zero-math, bit-exact, same alpha
/// synthesis / drop convention.
fn do_packed8_to_gbr8(
    src: &VideoFrame,
    src_info: FrameInfo,
    alpha_in: bool,
    alpha_out: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let comps = if alpha_in { 4 } else { 3 };
    let in_plane = &src.planes[0];
    let packed = gather_tight(&in_plane.data, in_plane.stride, w * comps, h);
    let mut g = vec![0u8; w * h];
    let mut b = vec![0u8; w * h];
    let mut r = vec![0u8; w * h];
    let mut a = if alpha_out {
        opaque_plane(w * h, 8)
    } else {
        Vec::new()
    };
    for i in 0..w * h {
        let base = i * comps;
        r[i] = packed[base];
        g[i] = packed[base + 1];
        b[i] = packed[base + 2];
        if alpha_in && alpha_out {
            a[i] = packed[base + 3];
        }
    }
    let mut planes = vec![
        VideoPlane { stride: w, data: g },
        VideoPlane { stride: w, data: b },
        VideoPlane { stride: w, data: r },
    ];
    if alpha_out {
        planes.push(VideoPlane { stride: w, data: a });
    }
    Ok(make_frame(src, planes))
}

/// `Gbrp8` ↔ `Gbrap8` alpha append / drop: the G, B, R byte planes are
/// copied verbatim; `add` appends an opaque 255 full-resolution alpha
/// plane, `!add` drops plane 3.
fn do_gbr8_alpha(src: &VideoFrame, src_info: FrameInfo, add: bool) -> Result<VideoFrame> {
    let need = if add { 3 } else { 4 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: GBR(A) source needs G, B, R(, A) planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let mut planes: Vec<VideoPlane> = src.planes[..3]
        .iter()
        .map(|p| VideoPlane {
            stride: w,
            data: gather_tight(&p.data, p.stride, w, h),
        })
        .collect();
    if add {
        planes.push(VideoPlane {
            stride: w,
            data: opaque_plane(w * h, 8),
        });
    }
    Ok(make_frame(src, planes))
}

/// Byte-tier planar GBR(A) (`Gbrp8` / `Gbrap8`) → packed deep RGB
/// (`Rgb48Le` / `Rgba64Le`): plane reorder plus the exact ×257 widen
/// (the 8 → 16 MSB replication — zero maps to zero, 255 to 65535), the
/// same rule as `Rgb24 → Rgb48Le`. A carried alpha plane is widened
/// like the colour bytes; a missing one is synthesised opaque 65535
/// and a surplus one is dropped.
fn do_gbr8_to_packed_deep(
    src: &VideoFrame,
    src_info: FrameInfo,
    alpha_in: bool,
    alpha_out: bool,
) -> Result<VideoFrame> {
    let need = if alpha_in { 4 } else { 3 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: GBR(A) source needs G, B, R(, A) planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let g = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let b = gather_tight(&src.planes[1].data, src.planes[1].stride, w, h);
    let r = gather_tight(&src.planes[2].data, src.planes[2].stride, w, h);
    let a = if alpha_in && alpha_out {
        Some(gather_tight(
            &src.planes[3].data,
            src.planes[3].stride,
            w,
            h,
        ))
    } else {
        None
    };
    let comps = if alpha_out { 4 } else { 3 };
    let mut out = vec![0u8; w * h * comps * 2];
    for i in 0..w * h {
        let base = i * comps * 2;
        wr16le(&mut out, base, r[i] as u16 * 257);
        wr16le(&mut out, base + 2, g[i] as u16 * 257);
        wr16le(&mut out, base + 4, b[i] as u16 * 257);
        if alpha_out {
            let av = a.as_ref().map_or(0xFFFF, |a| a[i] as u16 * 257);
            wr16le(&mut out, base + 6, av);
        }
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * comps * 2,
            data: out,
        }],
    ))
}

/// Packed deep RGB (`Rgb48Le` / `Rgba64Le`) → byte-tier planar GBR(A):
/// plane split keeping the top byte of each 16-bit word (truncation —
/// the exact inverse of the ×257 widen, so 8-bit content round-trips
/// losslessly). Alpha is carried truncated, synthesised opaque 255, or
/// dropped per the flag pair.
fn do_packed_deep_to_gbr8(
    src: &VideoFrame,
    src_info: FrameInfo,
    alpha_in: bool,
    alpha_out: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let comps = if alpha_in { 4 } else { 3 };
    let in_plane = &src.planes[0];
    let packed = gather_tight(&in_plane.data, in_plane.stride, w * comps * 2, h);
    let mut g = vec![0u8; w * h];
    let mut b = vec![0u8; w * h];
    let mut r = vec![0u8; w * h];
    let mut a = if alpha_out {
        opaque_plane(w * h, 8)
    } else {
        Vec::new()
    };
    for i in 0..w * h {
        let base = i * comps * 2;
        r[i] = (rd16le(&packed, base) >> 8) as u8;
        g[i] = (rd16le(&packed, base + 2) >> 8) as u8;
        b[i] = (rd16le(&packed, base + 4) >> 8) as u8;
        if alpha_in && alpha_out {
            a[i] = (rd16le(&packed, base + 6) >> 8) as u8;
        }
    }
    let mut planes = vec![
        VideoPlane { stride: w, data: g },
        VideoPlane { stride: w, data: b },
        VideoPlane { stride: w, data: r },
    ];
    if alpha_out {
        planes.push(VideoPlane { stride: w, data: a });
    }
    Ok(make_frame(src, planes))
}

/// Planar GBR(A) → `Gray8`: narrow each colour plane to 8 bits (top
/// bits — the crate depth policy) and run the full-range luminance
/// projection under the Y' row of the selected primaries, exactly like
/// the packed `RgbToGray` rows. Alpha (when present) is dropped.
fn do_gbr_to_gray(
    src: &VideoFrame,
    src_info: FrameInfo,
    matrix: YuvMatrix,
    bits: u32,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let need = if alpha_in { 4 } else { 3 };
    if src.planes.len() < need {
        return Err(Error::invalid(
            "pixfmt: GBR(A) source needs G, B, R(, A) planes",
        ));
    }
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let sb = if bits > 8 { 2 } else { 1 };
    let g = gather_tight(&src.planes[0].data, src.planes[0].stride, w * sb, h);
    let b = gather_tight(&src.planes[1].data, src.planes[1].stride, w * sb, h);
    let r = gather_tight(&src.planes[2].data, src.planes[2].stride, w * sb, h);
    // Interleave the top 8 bits of each sample into an R, G, B byte
    // triple and reuse the proven packed projection kernel.
    let mut rgb24 = vec![0u8; w * h * 3];
    if bits > 8 {
        let mask: u16 = (((1u32 << bits) - 1) & 0xFFFF) as u16;
        let shift = bits - 8;
        for i in 0..w * h {
            rgb24[i * 3] = ((rd16le(&r, i * 2) & mask) >> shift) as u8;
            rgb24[i * 3 + 1] = ((rd16le(&g, i * 2) & mask) >> shift) as u8;
            rgb24[i * 3 + 2] = ((rd16le(&b, i * 2) & mask) >> shift) as u8;
        }
    } else {
        for i in 0..w * h {
            rgb24[i * 3] = r[i];
            rgb24[i * 3 + 1] = g[i];
            rgb24[i * 3 + 2] = b[i];
        }
    }
    let mut gray = vec![0u8; w * h];
    yuv::rgb24_to_gray8(&rgb24, &mut gray, w * h, matrix);
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: gray,
        }],
    ))
}

/// `Gray8` → planar GBR(A): broadcast the gray byte into the G, B and R
/// planes at the family depth (MSB-replicated widen — peak maps to
/// peak, and [`do_gbr_to_gray`] recovers the original exactly because
/// the projection of `r = g = b = v` is `v`). `alpha_out` appends an
/// opaque full-scale alpha plane.
fn do_gray_to_gbr(
    src: &VideoFrame,
    src_info: FrameInfo,
    bits: u32,
    alpha_out: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let gray = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let plane = if bits > 8 {
        let shift = bits - 8;
        let mask = (1u32 << bits) - 1;
        let mut out = vec![0u8; w * h * 2];
        for (i, &v) in gray.iter().enumerate() {
            let v = v as u32;
            wr16le(
                &mut out,
                i * 2,
                (((v << shift) | (v >> (8 - shift))) & mask) as u16,
            );
        }
        out
    } else {
        gray
    };
    let sb = if bits > 8 { 2 } else { 1 };
    let mut planes = vec![
        VideoPlane {
            stride: w * sb,
            data: plane.clone(),
        },
        VideoPlane {
            stride: w * sb,
            data: plane.clone(),
        },
        VideoPlane {
            stride: w * sb,
            data: plane,
        },
    ];
    if alpha_out {
        planes.push(VideoPlane {
            stride: w * sb,
            data: opaque_plane(w * h, bits),
        });
    }
    Ok(make_frame(src, planes))
}

/// Packed `Rgb48Le` / `Rgba64Le` → planar GBR(A).
fn do_packed_deep_to_gbr(
    src: &VideoFrame,
    src_info: FrameInfo,
    bits: u8,
    alpha_in: bool,
    alpha_out: bool,
) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let shift = 16 - bits as u32;
    let comps = if alpha_in { 4 } else { 3 };
    let in_plane = &src.planes[0];
    let packed = gather_tight(&in_plane.data, in_plane.stride, w * comps * 2, h);
    let mut g = vec![0u8; w * h * 2];
    let mut b = vec![0u8; w * h * 2];
    let mut r = vec![0u8; w * h * 2];
    let mut a = if alpha_out {
        vec![0u8; w * h * 2]
    } else {
        Vec::new()
    };
    // Synthesised opaque alpha at the planar depth (used when the
    // packed source has no alpha word to carry).
    let opaque = ((1u32 << bits) - 1) as u16;
    for i in 0..w * h {
        let base = i * comps * 2;
        let rv = rd16le(&packed, base) >> shift;
        let gv = rd16le(&packed, base + 2) >> shift;
        let bv = rd16le(&packed, base + 4) >> shift;
        wr16le(&mut r, i * 2, rv);
        wr16le(&mut g, i * 2, gv);
        wr16le(&mut b, i * 2, bv);
        if alpha_out {
            let av = if alpha_in {
                rd16le(&packed, base + 6) >> shift
            } else {
                opaque
            };
            wr16le(&mut a, i * 2, av);
        }
    }
    let mut planes = vec![
        VideoPlane {
            stride: w * 2,
            data: g,
        },
        VideoPlane {
            stride: w * 2,
            data: b,
        },
        VideoPlane {
            stride: w * 2,
            data: r,
        },
    ];
    if alpha_out {
        planes.push(VideoPlane {
            stride: w * 2,
            data: a,
        });
    }
    Ok(make_frame(src, planes))
}
