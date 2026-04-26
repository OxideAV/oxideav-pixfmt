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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorSpace {
    #[default]
    Bt601Limited,
    Bt601Full,
    Bt709Limited,
    Bt709Full,
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
pub fn convert_in_place_if_same<'a>(
    src: &'a VideoFrame,
    src_info: FrameInfo,
    dst_format: PixelFormat,
) -> Option<&'a VideoFrame> {
    if src_info.format == dst_format {
        Some(src)
    } else {
        None
    }
}

/// Convert `src` to `dst_format`, producing a newly allocated frame.
pub fn convert(
    src: &VideoFrame,
    src_info: FrameInfo,
    dst_format: PixelFormat,
    opts: &ConvertOptions,
) -> Result<VideoFrame> {
    if src_info.format == dst_format {
        return Ok(src.clone());
    }
    let op = lookup(src_info.format, dst_format).ok_or_else(|| {
        Error::unsupported(format!(
            "pixfmt: conversion {:?} → {:?} not implemented",
            src_info.format, dst_format
        ))
    })?;
    op.apply(src, src_info, opts)
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

        // Gray ↔ RGB / Gray16 / Mono.
        (P::Gray8, P::Rgb24, Gray8ToPacked3),
        (P::Gray8, P::Rgba, Gray8ToPacked4),
        (P::Gray16Le, P::Gray8, Gray16ToGray8),
        (P::Gray8, P::Gray16Le, Gray8ToGray16),
        (P::MonoBlack, P::Gray8, MonoToGray { black_is_zero: true }),
        (P::MonoWhite, P::Gray8, MonoToGray { black_is_zero: false }),
        (P::Gray8, P::MonoBlack, GrayToMono { black_is_zero: true }),
        (P::Gray8, P::MonoWhite, GrayToMono { black_is_zero: false }),

        // YUV planar → packed RGB.
        (P::Yuv420P, P::Rgb24, YuvToRgb { wsub: 2, hsub: 2, alpha: false }),
        (P::Yuv422P, P::Rgb24, YuvToRgb { wsub: 2, hsub: 1, alpha: false }),
        (P::Yuv444P, P::Rgb24, YuvToRgb { wsub: 1, hsub: 1, alpha: false }),
        (P::Yuv420P, P::Rgba,  YuvToRgb { wsub: 2, hsub: 2, alpha: true }),
        (P::Yuv422P, P::Rgba,  YuvToRgb { wsub: 2, hsub: 1, alpha: true }),
        (P::Yuv444P, P::Rgba,  YuvToRgb { wsub: 1, hsub: 1, alpha: true }),

        // Packed RGB → YUV planar.
        (P::Rgb24, P::Yuv420P, RgbToYuv { wsub: 2, hsub: 2, alpha_in: false }),
        (P::Rgb24, P::Yuv422P, RgbToYuv { wsub: 2, hsub: 1, alpha_in: false }),
        (P::Rgb24, P::Yuv444P, RgbToYuv { wsub: 1, hsub: 1, alpha_in: false }),
        (P::Rgba,  P::Yuv420P, RgbToYuv { wsub: 2, hsub: 2, alpha_in: true }),
        (P::Rgba,  P::Yuv422P, RgbToYuv { wsub: 2, hsub: 1, alpha_in: true }),
        (P::Rgba,  P::Yuv444P, RgbToYuv { wsub: 1, hsub: 1, alpha_in: true }),

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

        // Palette.
        (P::Pal8, P::Rgb24, Pal8ToRgb { alpha: false }),
        (P::Pal8, P::Rgba,  Pal8ToRgb { alpha: true }),
        (P::Rgb24, P::Pal8, RgbToPal8 { alpha_in: false }),
        (P::Rgba,  P::Pal8, RgbToPal8 { alpha_in: true }),

        // CMYK ↔ RGB. Uncalibrated device-CMYK approximation; pure
        // bit-manipulation (no matrix / ColorSpace knob applies).
        (P::Cmyk,  P::Rgb24, CmykToRgb { alpha: false }),
        (P::Cmyk,  P::Rgba,  CmykToRgb { alpha: true }),
        (P::Rgb24, P::Cmyk,  RgbToCmyk { alpha_in: false }),
        (P::Rgba,  P::Cmyk,  RgbToCmyk { alpha_in: true }),
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
    Gray8ToPacked4,
    Gray16ToGray8,
    Gray8ToGray16,
    MonoToGray {
        black_is_zero: bool,
    },
    GrayToMono {
        black_is_zero: bool,
    },
    YuvToRgb {
        wsub: usize,
        hsub: usize,
        alpha: bool,
    },
    RgbToYuv {
        wsub: usize,
        hsub: usize,
        alpha_in: bool,
    },
    RescaleRange {
        wsub: usize,
        hsub: usize,
        to_full: bool,
    },
    NvToYuv420p {
        is_nv12: bool,
    },
    Yuv420pToNv {
        is_nv12: bool,
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
    },
    RgbToCmyk {
        /// When true, source is RGBA (alpha ignored). When false, Rgb24.
        alpha_in: bool,
    },
}

impl ConvertOp {
    fn apply(
        &self,
        src: &VideoFrame,
        src_info: FrameInfo,
        opts: &ConvertOptions,
    ) -> Result<VideoFrame> {
        // The YUV paths always want the limited-range matrix; YuvJ
        // input/output goes through RescaleRange, not this matrix.
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
            Self::Gray8ToPacked4 => gray_to_packed4(src, src_info),
            Self::Gray16ToGray8 => do_gray16_to_gray8(src, src_info),
            Self::Gray8ToGray16 => do_gray8_to_gray16(src, src_info),
            Self::MonoToGray { black_is_zero } => do_mono_to_gray(src, src_info, black_is_zero),
            Self::GrayToMono { black_is_zero } => do_gray_to_mono(src, src_info, black_is_zero),
            Self::YuvToRgb { wsub, hsub, alpha } => {
                do_yuv_to_rgb(src, src_info, matrix, wsub, hsub, alpha)
            }
            Self::RgbToYuv {
                wsub,
                hsub,
                alpha_in,
            } => do_rgb_to_yuv(src, src_info, matrix, wsub, hsub, alpha_in),
            Self::RescaleRange {
                wsub,
                hsub,
                to_full,
            } => rescale_range(src, src_info, wsub, hsub, to_full),
            Self::NvToYuv420p { is_nv12 } => nv_to_yuv420p(src, src_info, is_nv12),
            Self::Yuv420pToNv { is_nv12 } => yuv420p_to_nv(src, src_info, is_nv12),
            Self::Pal8ToRgb { alpha } => pal8_to_rgb(src, src_info, opts, alpha),
            Self::RgbToPal8 { alpha_in } => rgb_to_pal8(src, src_info, opts, alpha_in),
            Self::CmykToRgb { alpha } => do_cmyk_to_rgb(src, src_info, alpha),
            Self::RgbToCmyk { alpha_in } => do_rgb_to_cmyk(src, src_info, alpha_in),
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

fn gray_to_packed4(src: &VideoFrame, src_info: FrameInfo) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        gray::gray8_to_rgba(sr, &mut out[row * w * 4..row * w * 4 + w * 4], w);
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

// -------------------------------------------------------------------------
// Palette.

fn pal8_to_rgb(
    src: &VideoFrame,
    src_info: FrameInfo,
    opts: &ConvertOptions,
    alpha: bool,
) -> Result<VideoFrame> {
    let palette = opts
        .palette
        .as_ref()
        .ok_or_else(|| Error::invalid("pixfmt: Pal8 → RGB requires ConvertOptions.palette"))?;
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
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}

// -------------------------------------------------------------------------
// CMYK.

fn do_cmyk_to_rgb(src: &VideoFrame, src_info: FrameInfo, alpha: bool) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let bpp_out = if alpha { 4 } else { 3 };
    let mut out = vec![0u8; w * h * bpp_out];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        let dr = &mut out[row * w * bpp_out..row * w * bpp_out + w * bpp_out];
        if alpha {
            cmyk::cmyk_to_rgba(sr, dr, w);
        } else {
            cmyk::cmyk_to_rgb24(sr, dr, w);
        }
    }
    Ok(make_frame(
        src,
        vec![VideoPlane {
            stride: w * bpp_out,
            data: out,
        }],
    ))
}

fn do_rgb_to_cmyk(src: &VideoFrame, src_info: FrameInfo, alpha_in: bool) -> Result<VideoFrame> {
    let w = src_info.width as usize;
    let h = src_info.height as usize;
    let in_plane = &src.planes[0];
    let bpp_in = if alpha_in { 4 } else { 3 };
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * bpp_in);
        let dr = &mut out[row * w * 4..row * w * 4 + w * 4];
        if alpha_in {
            cmyk::rgba_to_cmyk(sr, dr, w);
        } else {
            cmyk::rgb24_to_cmyk(sr, dr, w);
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
