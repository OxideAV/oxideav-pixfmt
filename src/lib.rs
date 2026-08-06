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
//! - Ya16Le (packed 16-bit grey + alpha, full-scale LE16 words) ↔ Ya8
//!   (exact ×257 widen / high-byte narrow on both words), Gray16Le
//!   (luma word verbatim), Gray8, Rgba64Le (bit-exact broadcast +
//!   alpha carry out; rounded-mean luma derivation back, so
//!   grey-on-alpha content round-trips exactly) and Rgb24 / Rgba.
//! - Rgb48Le ↔ Rgb24, Rgba64Le ↔ Rgba (bit-shift), and
//!   Rgb48Le ↔ Rgba64Le (colour words verbatim; opaque 65535 alpha
//!   synthesised / dropped).
//! - The full ten-member planar GBR(A) ladder — `Gbrp8` / `Gbrap8`
//!   (byte planes)
//!   through the 10/12/14-bit members to the full-width `Gbrp16Le` /
//!   `Gbrap16Le` — ↔ **both** deep packed formats (Rgb48Le / Rgba64Le,
//!   alpha synthesised opaque full-scale or dropped when the shapes
//!   differ) and the 8-bit packed ones (Rgb24 / Rgba), plus Gray8
//!   interop: full-range luminance projection on the way out (same
//!   kernel as the packed RGB → Gray8 rows, alpha dropped) and an
//!   MSB-replicated `r = g = b` broadcast on the way in — gray content
//!   round-trips exactly.
//! - The full bit-depth ladder: planar YUV 8 ↔ 10 ↔ 12 ↔ 16 bit (same
//!   subsampling; MSB-replicated widen / truncating narrow, exact
//!   round-trips), and Gray8 ↔ Gray10Le ↔ Gray12Le ↔ Gray16Le. On the
//!   `Yuv*P16Le` trio every bit of the LE word is significant
//!   (full-scale 65535), so the 8 → 16 widen is the exact ×257 mapping.
//! - Gray16Le ↔ Gray8.
//! - MonoBlack/MonoWhite ↔ Gray8.
//! - The full planar YUV + alpha family — Yuva420P/422P/444P plus the
//!   deep 10/12/16-bit members at all three chroma sitings
//!   (Yuva420P/422P/444P × 10/12/16Le): promote/drop vs
//!   the alpha-less siblings, alpha-preserving chroma resample and
//!   depth moves inside the family (luma + alpha bit-exact at same
//!   depth, MSB-widened / truncated across depths), Rgb24/Rgba interop
//!   (alpha carried / dropped / synthesised opaque), and Gray8 luma
//!   extraction. Every ordered pair inside the uniform planar family
//!   ({Yuv,Yuva} × {420,422,444} × {8,10,12,16}) is a **direct**
//!   single-step conversion generated by a computed dispatch tier that
//!   fuses the depth move, the chroma resample (performed at the deeper
//!   of the two depths) and the alpha handling.
//! - Pal8 → Rgb24/Rgba uses the frame's attached palette side-channel
//!   ([`oxideav_core::VideoFrame::palette`]) when present, falling back
//!   to `opts.palette`; with neither, `Error::Invalid`.
//! - Rgb24/Rgba → Pal8 requires `opts.palette`; dithering per
//!   `opts.dither`. The output frame carries the table it was quantised
//!   against as its palette side-channel, so it is self-describing.
//! - Cmyk and CmykInverted ↔ Rgb24/Rgba — uncalibrated device-CMYK
//!   approximation in both ink conventions (see [`cmyk`] for the
//!   formulas and the ICC caveats), plus Cmyk ↔ CmykInverted as the
//!   exact per-byte complement.
//! - **Full-precision deep matrix**: the 16-bit planar tier
//!   ({Yuv,Yuva} × {420,422,444} `P16Le`) ↔ Rgb48Le / Rgba64Le runs
//!   the k-coefficient construction in Q30 over 16-bit samples — no
//!   8-bit narrowing anywhere on the path; chroma is resampled at
//!   16-bit precision and the Yuva alpha plane rides verbatim in the
//!   packed alpha word. The 10/12-bit family reaches these rows
//!   losslessly through the exact widen to the 16-bit tier.
//!
//! Pairs without a direct entry (table row or computed planar-family
//! op) are resolved automatically through a
//! **single-pivot staged conversion** (one intermediate format, chosen
//! in a fidelity-aware order: YUV pivots for YUV → YUV moves so no
//! colour matrix enters the path — with the 16-bit `Yuv(a)*P16Le` tier
//! preferred whenever a deep endpoint is YUV carriage, so chroma is
//! resampled and the matrix applied at full precision — and
//! alpha-capable / deep RGB pivots
//! preferred where the endpoints call for them). [`supports`] /
//! [`supports_direct`] report per-pair availability. The matrix is
//! fully closed: **every ordered pair** of the 61 `PixelFormat`
//! variants resolves — directly (976 pairs) or through one staged
//! pivot (all 3660).
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
//!
//! # Per-plane significant bits
//!
//! Source frames may carry the per-plane **significant-bits
//! side-channel** defined by [`oxideav_core::VideoFrame`] — one
//! LSB-anchored byte per image plane naming that plane's true depth,
//! e.g. `[12, 10, 10]` for a wavelet codec's 12-bit luma + 10-bit
//! chroma on a `Yuv444P12Le` / `Yuv444P16Le` surface. [`convert`]
//! honours the record (marked planes convert at their recorded depth),
//! rejects invalid records (`0` or above the surface's nominal depth)
//! with `Error::Invalid`, ignores records on `Pal8` (indices are not
//! magnitudes), and always emits nominal-depth output with no record
//! attached — see the policy section on [`convert`] for the details.

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
