#![no_main]

//! Geometry fuzz harness for `oxideav_pixfmt::convert`.
//!
//! Unlike a byte-stream parser fuzzer, this target *constructs* a
//! well-formed source [`VideoFrame`] from the fuzzer's input — choosing a
//! source [`PixelFormat`], small / odd dimensions, and optional extra
//! stride padding — and then drives every supported `(src, dst)`
//! conversion through [`convert`]. The single invariant under test is
//! that a structurally-valid frame can never make a converter panic,
//! integer-overflow (debug builds abort on overflow), read out of
//! bounds, or OOM-abort. Returning `Err` for geometry a format cannot
//! represent (e.g. an odd width on a 4:2:0 layout) is the *correct*
//! behaviour and is accepted; only a crash is a finding.
//!
//! The motivating bug class: subsampled-chroma planar YUV → packed RGB
//! truncates `cw = w / wsub`, sizing the U/V planes one sample short of
//! what the decoder reads back for a trailing odd luma column/row — an
//! out-of-bounds chroma read. The harness biases toward 1×1, odd, and
//! near-power-of-two-plus-one dimensions to surface exactly that.
//!
//! ## Wire framing of the fuzz input
//!
//!   * byte 0 — source pixel-format selector (mod the format table len).
//!   * byte 1 — width selector → mapped to a small/odd value.
//!   * byte 2 — height selector → mapped to a small/odd value.
//!   * byte 3 — extra stride padding in bytes (clamped) added to every
//!     plane's tight row stride.
//!   * byte 4 — destination-format rotation: instead of every dst, start
//!     the dst sweep at this offset so successive inputs spread coverage.
//!   * remaining bytes (`fill`) — used to fill the plane data (cycled),
//!     so the per-pixel maths sees varied samples rather than a
//!     constant. The head of `fill` doubles as side-channel control:
//!     `fill[0] & 1` attaches a fuzz-length palette record to Pal8
//!     sources, and `fill[0] & 2` attaches a **hostile significant-bits
//!     record** (`fill[1] % 9` bytes taken verbatim from `fill[2..]` —
//!     including 0, values above the surface depth, and lengths that
//!     don't match the plane count). Per the documented `convert()`
//!     policy such records must be honoured, ignored, or rejected with
//!     `Err` — never panic.

use libfuzzer_sys::fuzz_target;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FormatInfo, FrameInfo, Palette};

/// Every source format the geometry harness knows how to build a
/// well-formed frame for. Keep this list in sync with the conversion
/// coverage table — any format that appears as a `src` in `convert`'s
/// dispatch should be buildable here so the fuzzer can reach it.
const FORMATS: &[PixelFormat] = &[
    PixelFormat::Rgb24,
    PixelFormat::Bgr24,
    PixelFormat::Rgba,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray8,
    PixelFormat::Gray16Le,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Ya8,
    PixelFormat::MonoBlack,
    PixelFormat::MonoWhite,
    PixelFormat::Pal8,
    PixelFormat::Cmyk,
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::Yuv411P,
    PixelFormat::Yuv420P10Le,
    PixelFormat::Yuv422P10Le,
    PixelFormat::Yuv444P10Le,
    PixelFormat::Yuv420P12Le,
    PixelFormat::Yuv422P12Le,
    PixelFormat::Yuv444P12Le,
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
    PixelFormat::Yuv420P16Le,
    PixelFormat::Yuv422P16Le,
    PixelFormat::Yuv444P16Le,
    PixelFormat::Yuva422P,
    PixelFormat::Yuva444P,
    PixelFormat::Yuva422P10Le,
    PixelFormat::Yuva422P12Le,
    PixelFormat::Yuva444P10Le,
    PixelFormat::Yuva444P12Le,
    PixelFormat::Yuva422P16Le,
    PixelFormat::Yuva444P16Le,
    PixelFormat::Gbrp8,
    PixelFormat::Gbrp16Le,
    PixelFormat::Gbrap16Le,
    PixelFormat::Yuva420P10Le,
    PixelFormat::Yuva420P12Le,
    PixelFormat::Yuva420P16Le,
    PixelFormat::Gbrap8,
    PixelFormat::Ya16Le,
    PixelFormat::CmykInverted,
];

/// Map a raw fuzzer byte to a small dimension, biased toward the values
/// that stress geometry maths: 1 (the degenerate corner), small odds,
/// and a couple of even sizes. Capped at 17 to keep per-iteration memory
/// tiny (worst case is ~17×17×8 B for the deep-RGBA layouts).
fn pick_dim(b: u8) -> u32 {
    const DIMS: [u32; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17];
    DIMS[(b as usize) % DIMS.len()]
}

/// Bytes per *packed pixel* for the single-plane packed formats, or per
/// *luma sample* for planar layouts (chroma sizing is derived from the
/// subsampling factors). Returns `None` for formats this harness builds
/// with a bespoke layout (mono, packed-422, NV, palette).
fn luma_bytes_per_sample(info: &FormatInfo) -> usize {
    if info.bit_depth > 8 {
        2
    } else {
        1
    }
}

/// Build a tight plane of `rows` rows, each `row_bytes` wide, with
/// `pad` extra padding bytes per row (stride = row_bytes + pad). Data is
/// filled by cycling `fill`.
fn build_plane(
    rows: usize,
    row_bytes: usize,
    pad: usize,
    fill: &[u8],
    seed: &mut usize,
) -> VideoPlane {
    let stride = row_bytes + pad;
    let mut data = vec![0u8; stride * rows];
    for byte in data.iter_mut() {
        *byte = if fill.is_empty() {
            (*seed & 0xFF) as u8
        } else {
            fill[*seed % fill.len()]
        };
        *seed = seed.wrapping_add(1);
    }
    VideoPlane { stride, data }
}

/// Construct a structurally-valid source frame for `fmt` at `w`×`h` with
/// `pad` extra stride bytes per plane, returning `None` for a format the
/// harness does not build (so the caller skips it).
fn build_frame(fmt: PixelFormat, w: u32, h: u32, pad: usize, fill: &[u8]) -> Option<VideoFrame> {
    let wu = w as usize;
    let hu = h as usize;
    let info = FormatInfo::of(fmt);
    let mut seed = 0usize;

    let planes: Vec<VideoPlane> = match fmt {
        // Single-plane packed RGB / gray / CMYK / deep-RGB. Bytes-per-pixel
        // is component_count × sample_bytes, but we look it up explicitly
        // to avoid guessing the component count from FormatInfo.
        PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
            vec![build_plane(hu, wu * 3, pad, fill, &mut seed)]
        }
        PixelFormat::Rgba
        | PixelFormat::Bgra
        | PixelFormat::Argb
        | PixelFormat::Abgr
        | PixelFormat::Cmyk
        | PixelFormat::CmykInverted => vec![build_plane(hu, wu * 4, pad, fill, &mut seed)],
        // Packed 16-bit grey + alpha — interleaved (Y, A) LE16 word
        // pairs, 4 bytes per pixel; any byte pattern is a legal sample.
        PixelFormat::Ya16Le => vec![build_plane(hu, wu * 4, pad, fill, &mut seed)],
        PixelFormat::Rgb48Le => vec![build_plane(hu, wu * 6, pad, fill, &mut seed)],
        PixelFormat::Rgba64Le => vec![build_plane(hu, wu * 8, pad, fill, &mut seed)],
        PixelFormat::Gray8 => vec![build_plane(hu, wu, pad, fill, &mut seed)],
        PixelFormat::Gray16Le | PixelFormat::Gray10Le | PixelFormat::Gray12Le => {
            vec![build_plane(hu, wu * 2, pad, fill, &mut seed)]
        }
        PixelFormat::Ya8 => vec![build_plane(hu, wu * 2, pad, fill, &mut seed)],
        // Mono — one bit per pixel, packed MSB-first into ceil(w/8) bytes
        // per row.
        PixelFormat::MonoBlack | PixelFormat::MonoWhite => {
            vec![build_plane(hu, wu.div_ceil(8), pad, fill, &mut seed)]
        }
        // Palette — one index byte per pixel.
        PixelFormat::Pal8 => vec![build_plane(hu, wu, pad, fill, &mut seed)],
        // Packed 4:2:2 — 2 bytes per pixel (Y/U or Y/V interleaved). Only
        // representable for even width; build the tight buffer regardless
        // and let `convert` reject odd width.
        PixelFormat::Yuyv422 | PixelFormat::Uyvy422 => {
            vec![build_plane(hu, wu * 2, pad, fill, &mut seed)]
        }
        // NV12 / NV21 — full-res Y plane + a half-res interleaved UV plane
        // (cw*2 bytes per chroma row, ch rows). Use truncating division so
        // an odd dimension produces the exact under-sized geometry the
        // converter must reject rather than read past.
        PixelFormat::Nv12 | PixelFormat::Nv21 => {
            let cw = wu / 2;
            let ch = hu / 2;
            vec![
                build_plane(hu, wu, pad, fill, &mut seed),
                build_plane(ch, cw * 2, pad, fill, &mut seed),
            ]
        }
        // Planar GBR(A) — 3 or 4 planes at 4:4:4, either 16-bit LE words
        // (2 bytes per sample) or, for Gbrp8, plain bytes.
        PixelFormat::Gbrp8
        | PixelFormat::Gbrap8
        | PixelFormat::Gbrp10Le
        | PixelFormat::Gbrp12Le
        | PixelFormat::Gbrp14Le
        | PixelFormat::Gbrp16Le
        | PixelFormat::Gbrap10Le
        | PixelFormat::Gbrap12Le
        | PixelFormat::Gbrap14Le
        | PixelFormat::Gbrap16Le => {
            let n = if info.has_alpha { 4 } else { 3 };
            let sb = luma_bytes_per_sample(&info);
            (0..n)
                .map(|_| build_plane(hu, wu * sb, pad, fill, &mut seed))
                .collect()
        }
        // Generic planar YUV (incl. Yuva420P). Y at full res; chroma at
        // (w/wsub)×(h/hsub) using truncating division — exactly the
        // geometry that exercises the odd-dimension chroma-index path.
        // A trailing alpha plane at luma resolution for Yuva420P.
        _ if info.is_planar => {
            let sb = luma_bytes_per_sample(&info);
            let cw = wu / info.chroma_w_sub as usize;
            let ch = hu / info.chroma_h_sub as usize;
            let mut planes = vec![
                build_plane(hu, wu * sb, pad, fill, &mut seed),
                build_plane(ch, cw * sb, pad, fill, &mut seed),
                build_plane(ch, cw * sb, pad, fill, &mut seed),
            ];
            if info.has_alpha {
                planes.push(build_plane(hu, wu * sb, pad, fill, &mut seed));
            }
            planes
        }
        // Any format not handled above — skip.
        _ => return None,
    };

    Some(VideoFrame { pts: None, planes })
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    let fmt = FORMATS[data[0] as usize % FORMATS.len()];
    let w = pick_dim(data[1]);
    let h = pick_dim(data[2]);
    // Extra stride padding, clamped small to keep allocations tiny.
    let pad = (data[3] % 8) as usize;
    let dst_start = data[4] as usize;
    let fill = &data[5..];

    let mut src = match build_frame(fmt, w, h, pad, fill) {
        Some(f) => f,
        None => return,
    };
    let info = FrameInfo::new(fmt, w, h);

    // Pal8 sources sometimes carry their colour table in-band as the
    // palette side-channel (a trailing stride-0 plane); attach one of
    // fuzz-controlled length (including deliberately short tables, whose
    // out-of-range indices must fall back to the missing-entry colour
    // rather than crash).
    if fmt == PixelFormat::Pal8 && !fill.is_empty() && fill[0] & 1 == 1 {
        let len = (fill.len().min(768) / 3) * 3;
        src.set_palette(fill[..len].to_vec());
    }

    // Optionally attach a hostile per-plane significant-bits record
    // (second side-channel kind, stride == usize::MAX sentinel): raw
    // fuzz bytes of fuzz-chosen length, deliberately including zero
    // bits, values above the surface's nominal depth, and lengths
    // shorter / longer than the image-plane count. The documented
    // convert() policy is honour-or-reject (Error::Invalid) — and on
    // Pal8 the record is ignored and composes with the palette record.
    // Any panic is a finding.
    if fill.len() >= 2 && fill[0] & 2 != 0 {
        let rec_len = (fill[1] % 9) as usize;
        let rec: Vec<u8> = fill[2..].iter().copied().take(rec_len).collect();
        src.set_significant_bits(rec);
    }

    // A palette is needed for Pal8 → RGB and RGB → Pal8; supply a full
    // 256-entry grayscale ramp so those arms are reached rather than
    // short-circuiting on a missing palette.
    let mut opts = ConvertOptions::default();
    let colors: Vec<[u8; 4]> = (0..=255u16)
        .map(|i| [i as u8, i as u8, i as u8, 255])
        .collect();
    opts.palette = Some(Palette { colors });

    // Drive every destination format. `convert` itself rejects same-fmt
    // and unsupported pairs cheaply; the contract is no-panic regardless.
    let n = FORMATS.len();
    for i in 0..n {
        let dst = FORMATS[(dst_start + i) % n];
        let _ = convert(&src, info, dst, &opts);
    }
});
