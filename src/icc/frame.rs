//! `VideoFrame` integration: read a frame's embedded ICC profile and
//! run an [`IccTransform`] over a packed frame.

use oxideav_core::{BlobKind, MetadataBlob, PixelFormat, VideoFrame, VideoPlane};

use super::{IccError, IccProfile, IccTransform};
use crate::FrameInfo;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sample {
    U8,
    U16,
    F32,
}

/// (colour channels, alpha?, sample type, inverted ink) of a packed
/// single-plane format this module can read and write.
fn layout(f: PixelFormat) -> Option<(usize, bool, Sample, bool)> {
    use PixelFormat as P;
    Some(match f {
        P::Gray8 => (1, false, Sample::U8, false),
        P::Ya8 => (1, true, Sample::U8, false),
        P::Gray16Le => (1, false, Sample::U16, false),
        P::Ya16Le => (1, true, Sample::U16, false),
        P::GrayF32Le => (1, false, Sample::F32, false),
        P::Rgb24 => (3, false, Sample::U8, false),
        P::Rgba => (3, true, Sample::U8, false),
        P::Rgb48Le => (3, false, Sample::U16, false),
        P::Rgba64Le => (3, true, Sample::U16, false),
        P::RgbF32Le => (3, false, Sample::F32, false),
        P::RgbaF32Le => (3, true, Sample::F32, false),
        P::Cmyk => (4, false, Sample::U8, false),
        P::CmykInverted => (4, false, Sample::U8, true),
        _ => return None,
    })
}

/// The pixel formats [`transform_frame`] accepts on either side.
pub fn frame_formats() -> &'static [PixelFormat] {
    use PixelFormat as P;
    &[
        P::Gray8,
        P::Ya8,
        P::Gray16Le,
        P::Ya16Le,
        P::GrayF32Le,
        P::Rgb24,
        P::Rgba,
        P::Rgb48Le,
        P::Rgba64Le,
        P::RgbF32Le,
        P::RgbaF32Le,
        P::Cmyk,
        P::CmykInverted,
    ]
}

/// The ICC profile a frame carries in its metadata-blob side channel
/// ([`BlobKind::ICC`]), parsed. `None` when the frame has none.
pub fn frame_profile(frame: &VideoFrame) -> Option<Result<IccProfile, IccError>> {
    frame
        .blob(&BlobKind::ICC)
        .map(|b| IccProfile::parse(&b.data))
}

fn read(data: &[u8], s: Sample, idx: usize) -> f64 {
    match s {
        Sample::U8 => data[idx] as f64 / 255.0,
        Sample::U16 => u16::from_le_bytes([data[2 * idx], data[2 * idx + 1]]) as f64 / 65535.0,
        Sample::F32 => f32::from_le_bytes([
            data[4 * idx],
            data[4 * idx + 1],
            data[4 * idx + 2],
            data[4 * idx + 3],
        ]) as f64,
    }
}

fn write(data: &mut [u8], s: Sample, idx: usize, v: f64) {
    match s {
        Sample::U8 => data[idx] = (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        Sample::U16 => {
            let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            data[2 * idx..2 * idx + 2].copy_from_slice(&q.to_le_bytes());
        }
        Sample::F32 => data[4 * idx..4 * idx + 4].copy_from_slice(&(v as f32).to_le_bytes()),
    }
}

fn width_of(s: Sample) -> usize {
    match s {
        Sample::U8 => 1,
        Sample::U16 => 2,
        Sample::F32 => 4,
    }
}

/// Run `t` over a packed frame and return a new frame in `dst_format`.
///
/// Both formats must be one of [`frame_formats`], with colour-channel
/// counts matching the transform (Gray 1, RGB 3, CMYK 4); convert other
/// layouts with [`crate::convert`] first. Alpha is carried when both
/// sides have it, synthesised opaque when only the destination does,
/// and dropped otherwise. Integer samples are normalised to `[0, 1]`
/// (`CmykInverted` is complemented first); float samples pass as-is.
///
/// The output keeps the source's `pts` and its metadata blobs, with the
/// ICC blob replaced by `dst_profile`'s bytes when one is given (pass
/// the profile the transform targets so the frame stays
/// self-describing).
pub fn transform_frame(
    t: &IccTransform,
    src: &VideoFrame,
    info: FrameInfo,
    dst_format: PixelFormat,
    dst_profile: Option<&IccProfile>,
) -> Result<VideoFrame, IccError> {
    let (ic, ia, is, iinv) =
        layout(info.format).ok_or(IccError::Unsupported("source pixel format"))?;
    let (oc, oa, os, oinv) =
        layout(dst_format).ok_or(IccError::Unsupported("destination pixel format"))?;
    if ic != t.input_channels() || oc != t.output_channels() {
        return Err(IccError::BufferSize);
    }
    let (w, h) = (info.width as usize, info.height as usize);
    let plane = src.image_planes().first().ok_or(IccError::BufferSize)?;
    let in_px = ic + usize::from(ia);
    let row_in = w * in_px * width_of(is);
    if h > 0 && (plane.stride < row_in || plane.data.len() < plane.stride * (h - 1) + row_in) {
        return Err(IccError::BufferSize);
    }
    let out_px = oc + usize::from(oa);
    let stride = w * out_px * width_of(os);
    let mut out = vec![0u8; stride * h];
    let mut dev_in = vec![0f32; w * h * ic];
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + row_in];
        for x in 0..w {
            for c in 0..ic {
                let v = read(row, is, x * in_px + c);
                dev_in[(y * w + x) * ic + c] = if iinv { 1.0 - v } else { v } as f32;
            }
        }
    }
    let mut dev_out = vec![0f32; w * h * oc];
    t.transform_f32(&dev_in, &mut dev_out)?;
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + row_in];
        let orow = &mut out[y * stride..(y + 1) * stride];
        for x in 0..w {
            for c in 0..oc {
                let v = dev_out[(y * w + x) * oc + c] as f64;
                write(orow, os, x * out_px + c, if oinv { 1.0 - v } else { v });
            }
            if oa {
                let a = if ia {
                    read(row, is, x * in_px + ic)
                } else {
                    1.0
                };
                write(orow, os, x * out_px + oc, a);
            }
        }
    }
    let mut blobs: Vec<MetadataBlob> = src
        .blobs()
        .into_iter()
        .filter(|b| b.kind != BlobKind::ICC)
        .collect();
    if let Some(p) = dst_profile {
        blobs.push(MetadataBlob::new(BlobKind::ICC, p.as_bytes().to_vec()));
    }
    let mut f = VideoFrame {
        pts: src.pts,
        planes: vec![VideoPlane { stride, data: out }],
    };
    if !blobs.is_empty() {
        f.set_blobs(blobs);
    }
    Ok(f)
}
