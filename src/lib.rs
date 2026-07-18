#![cfg_attr(feature = "nightly", feature(portable_simd))]
//! Pure-Rust pixel-format conversions for the oxideav framework.
//!
//! This crate extends [`oxideav_core::PixelFormat`] with the converters
//! that the rest of the codec/container ecosystem depends on — RGB/BGR
//! swizzles, YUV↔RGB (BT.601 and BT.709, limited and full range), chroma
//! subsampling changes (4:2:0 ↔ 4:2:2 ↔ 4:4:4), NV12/NV21 ↔ Yuv420P,
//! grayscale expansion, 16-bit/8-bit bit-depth changes, and palette
//! generation + Pal8 encode/decode with optional dithering.
//!
//! # Entry points
//!
//! - [`convert`] — the single conversion function; dispatches on
//!   `(src_info.format, dst_format)` (the caller provides the source
//!   format / dimensions via [`FrameInfo`]) and returns a freshly
//!   allocated [`VideoFrame`].
//! - [`generate_palette`] — build a [`Palette`] from one or more source
//!   frames, honouring the selected [`PaletteStrategy`].
//! - [`convert_in_place_if_same`] — trivial passthrough helper for the
//!   "no conversion needed" case so callers don't duplicate the check.
//!
//! # Colour science
//!
//! The YUV↔RGB paths are written as scalar integer pipelines against
//! BT.601 and BT.709 weights. The "limited" variants use the studio
//! range (Y in 16..=235, chroma in 16..=240); "full" variants use the
//! full 0..=255 range, matching JPEG / "J" YUV. Every converter clamps
//! to `[0, 255]` after reconstruction. See [`yuv`] for the exact matrix
//! coefficients.
//!
//! # Feature coverage
//!
//! Not every pair in the Cartesian product of [`PixelFormat`] variants
//! is supported; the first-tier matrix is:
//!
//! - RGB family (Rgb24/Bgr24/Rgba/Bgra/Argb/Abgr) all-to-all.
//! - Yuv420P/422P/444P ↔ Rgb24 / Rgba under BT.601 / BT.709 / BT.2020
//!   (limited range), plus the full-range YuvJ420P/422P/444P families
//!   directly ↔ Rgb24 / Rgba (the format pins the matrix range; the
//!   `ColorSpace` option picks the primaries).
//! - Gray8 ↔ every YUV family (luma extraction with range rescale /
//!   neutral-chroma synthesis; no colour matrix), and Rgb24 / Rgba →
//!   Gray8 full-range luminance projection.
//! - Yuv420P/422P/444P all-to-all direct (chroma resample only — luma
//!   copied byte-for-byte, no RGB hop), plus the same six pairs on the
//!   full-range `YuvJ*` family AND on the 16-bit `Yuv*P16Le` family
//!   (full 16-bit precision chroma resample, identical rounding).
//! - Yuv411P ↔ Yuv420P/422P/444P (chroma resample only — luma copied),
//!   plus Yuv411P ↔ Rgb24/Rgba under BT.601/709/2020. The 4:1:1 layout
//!   has chroma horizontally subsampled by 4 (NTSC DV-25 / JPEG
//!   `-sample 4x1`); RGB encode and decode stage through a 4:4:4
//!   chroma intermediate so no new colour math is introduced.
//! - YuvJ420P/422P/444P ↔ Yuv* equivalents — plane copy with range
//!   rescale.
//! - Nv12/Nv21 ↔ Yuv420P, plus direct ↔ Rgb24/Rgba via a fused path
//!   that runs the planar 4:2:0 encoder/decoder under the hood.
//! - Yuyv422/Uyvy422 packed 4:2:2 ↔ Yuv422P, plus direct ↔ Rgb24/Rgba
//!   under BT.601/709/2020 (limited or full range). Yuyv ↔ Uyvy is a
//!   zero-math byte swap.
//! - Gray8 ↔ Rgb24/Rgba broadcast.
//! - Ya8 ↔ Gray8/Rgb24/Rgba (luma broadcast; alpha is carried through to
//!   Rgba, dropped on the way to Rgb24, and synthesised as opaque 255 on
//!   the return paths from Gray8/Rgb24).
//! - Rgb48Le ↔ Rgb24, Rgba64Le ↔ Rgba (bit-shift).
//! - Planar GBR(A) at 10/12/14 bits ↔ both the deep packed formats
//!   (Rgb48Le / Rgba64Le) and the 8-bit packed ones (Rgb24 / Rgba).
//! - The full bit-depth ladder: planar YUV 8 ↔ 10 ↔ 12 ↔ 16 bit (same
//!   subsampling; MSB-replicated widen / truncating narrow, exact
//!   round-trips), and Gray8 ↔ Gray10Le ↔ Gray12Le ↔ Gray16Le. On the
//!   `Yuv*P16Le` trio every bit of the LE word is significant
//!   (full-scale 65535), so the 8 → 16 widen is the exact ×257 mapping.
//! - Gray16Le ↔ Gray8.
//! - MonoBlack/MonoWhite ↔ Gray8.
//! - Yuva420P/422P/444P (planar YUV + full-resolution alpha plane):
//!   promote/drop vs the alpha-less siblings, alpha-preserving chroma
//!   resample inside the family (luma + alpha bit-exact), Rgb24/Rgba
//!   interop (alpha carried / dropped / synthesised opaque), and Gray8
//!   luma extraction.
//! - Pal8 → Rgb24/Rgba uses the frame's attached palette side-channel
//!   ([`oxideav_core::VideoFrame::palette`]) when present, falling back
//!   to `opts.palette`; with neither, `Error::Invalid`.
//! - Rgb24/Rgba → Pal8 requires `opts.palette`; dithering per
//!   `opts.dither`. The output frame carries the table it was quantised
//!   against as its palette side-channel, so it is self-describing.
//! - Cmyk ↔ Rgb24/Rgba — uncalibrated device-CMYK approximation (see
//!   [`cmyk`] for the formula and the caveats around ICC profiles and
//!   Adobe-inverted JPEGs).
//!
//! Pairs without a direct entry are resolved automatically through a
//! **single-pivot staged conversion** (one intermediate format, chosen
//! in a fidelity-aware order: YUV pivots for YUV → YUV moves so no
//! colour matrix enters the path — with the 16-bit `Yuv*P16Le` tier
//! preferred when either endpoint is deeper than 8 bits, so chroma is
//! resampled at full precision — and alpha-capable / deep RGB pivots
//! preferred where the endpoints call for them). [`supports`] /
//! [`supports_direct`] report per-pair availability; anything neither
//! direct nor stageable returns `Error::Unsupported`.
//!
//! # Bit-depth precision policy
//!
//! Every depth change in the crate (8 ↔ 10 ↔ 12 ↔ 16 on planar YUV,
//! the deep grayscale ladder, and the GBR(A) widen/narrow steps) uses
//! one deterministic rule and **no dithering**:
//!
//! - **Widening** places the value in the top bits and replicates its
//!   MSBs into the freed low bits. Zero maps to zero, full-scale maps
//!   to full-scale, the mapping is strictly monotonic, and it tracks
//!   the ideal `v · (2^dst − 1) / (2^src − 1)` rescale within one
//!   output code (exactly, for 8 → 16, where the ratio is the integer
//!   257).
//! - **Narrowing** truncates the low bits (keeps the top ones) — the
//!   exact inverse of the replication fill, so widen → narrow and
//!   narrow-of-widen round-trips are lossless.
//!
//! Dithered quantisation is deliberately excluded from depth moves:
//! codec pipelines feeding this crate (lossless intermediates,
//! reference comparisons, bit-exact round-trip tests) need
//! deterministic, invertible mappings. `Dither` applies only to the
//! palette quantisation path (`Rgb24`/`Rgba` → `Pal8`).

pub mod alpha;
pub mod cmyk;
pub mod convert;
pub mod dither;
pub mod format_info;
pub mod gray;
pub mod pal8;
pub mod palette;
pub mod rgb;
mod simd_dispatch;
pub mod transfer;
pub mod yuv;
mod yuv_simd;

pub use alpha::{
    blit_alpha_mask, modulate_alpha, over_buffer, over_premul, over_straight, premultiply,
    unpremultiply,
};
pub use convert::{
    convert, convert_in_place_if_same, supports, supports_direct, ColorSpace, ConvertOptions,
    Dither, FrameInfo,
};
pub use format_info::{ChromaSubsampling, FormatInfo};
pub use palette::{generate_palette, Palette, PaletteGenOptions, PaletteStrategy};
