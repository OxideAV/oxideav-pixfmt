# oxideav-pixfmt

[![CI](https://github.com/OxideAV/oxideav-pixfmt/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-pixfmt/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-pixfmt.svg)](https://crates.io/crates/oxideav-pixfmt) [![docs.rs](https://docs.rs/oxideav-pixfmt/badge.svg)](https://docs.rs/oxideav-pixfmt) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust pixel-format conversions, palette quantisation, and dithering.

Two APIs in one crate:

* **Low-level, framework-agnostic** — functions in
  [`yuv`](https://docs.rs/oxideav-pixfmt/latest/oxideav_pixfmt/yuv/),
  [`rgb`](https://docs.rs/oxideav-pixfmt/latest/oxideav_pixfmt/rgb/),
  [`gray`](https://docs.rs/oxideav-pixfmt/latest/oxideav_pixfmt/gray/),
  and [`pal8`](https://docs.rs/oxideav-pixfmt/latest/oxideav_pixfmt/pal8/)
  operate directly on `&[u8]` / `&mut [u8]` buffers. No frame type or
  allocator is imposed — drop them into whatever image/video stack you
  already have.
* **High-level** — [`convert()`](https://docs.rs/oxideav-pixfmt/latest/oxideav_pixfmt/fn.convert.html)
  takes a `VideoFrame` from [`oxideav-core`](https://crates.io/crates/oxideav-core)
  and dispatches on `(src.format, dst_format)`. Convenient when you're
  already inside the oxideav framework.

Zero C dependencies. Zero FFI. Runtime-selected AVX2 (x86_64) and NEON
(aarch64) paths with a scalar fixed-point fallback on every other target.

## Install

```toml
[dependencies]
oxideav-pixfmt = "0.1"
```

Nightly users who want the `std::simd` path:

```toml
oxideav-pixfmt = { version = "0.1", features = ["nightly"] }
```

## What's supported

| category                | formats / operations                                                        |
| ----------------------- | --------------------------------------------------------------------------- |
| RGB / BGR family        | `Rgb24`, `Bgr24`, `Rgba`, `Bgra`, `Argb`, `Abgr` — all-to-all swizzles      |
| Deep RGB                | `Rgb48Le` ↔ `Rgb24`, `Rgba64Le` ↔ `Rgba`, `Rgb48Le` ↔ `Rgba64Le` (colour words verbatim, opaque 65535 synthesis / drop) |
| Deep matrix             | the 16-bit planar tier ({`Yuv`,`Yuva`} × {420,422,444} `P16Le`) ↔ `Rgb48Le` / `Rgba64Le` at **full 16-bit precision** (Q30 k-coefficient construction, chroma resampled at 16 bits, Yuva alpha word verbatim); the 10/12-bit family stages in losslessly via the exact widen |
| Planar GBR              | the full ten-member ladder — `Gbrp8` / `Gbrap8` (byte planes) through `Gbrp10/12/14Le` to the full-width `Gbrp16Le` / `Gbrap16Le` — ↔ **both** `Rgb48Le` and `Rgba64Le` (alpha synthesised opaque / dropped when the shapes differ) and ↔ `Rgb24` / `Rgba`; plus `Gray8` interop (full-range luminance projection out, MSB-replicated broadcast in) and `Gbrp8` ↔ `Gbrap8` alpha append / drop |
| YUV planar ↔ RGB        | `Yuv420P` / `Yuv422P` / `Yuv444P` ↔ `Rgb24` / `Rgba`, plus the full-range `YuvJ*` families direct ↔ RGB |
| Bit-depth ladder        | planar YUV 8 ↔ 10 ↔ 12 ↔ 16 bit (same layout, exact round-trips; `Yuv*P16Le` is full-scale 65535); `Gray8` ↔ `Gray10Le` ↔ `Gray12Le` ↔ `Gray16Le` |
| Chroma subsampling      | `4:4:4` ↔ `4:2:2` ↔ `4:2:0` (SIMD-accelerated up- and down-sample)          |
| Direct planar ↔ planar  | `Yuv420P` / `Yuv422P` / `Yuv444P` all-to-all + same on `YuvJ*` and on the 16-bit `Yuv*P16Le` trio (no RGB hop) |
| Semi-planar             | `NV12` / `NV21` ↔ `Yuv420P` / `Rgb24` / `Rgba`                              |
| Packed 4:2:2            | `Yuyv422` / `Uyvy422` ↔ `Yuv422P` / `Rgb24` / `Rgba` + Yuyv ↔ Uyvy swap     |
| Full ↔ limited range    | `YuvJ420P` / `YuvJ422P` / `YuvJ444P` ↔ `Yuv*`                               |
| Gray ↔ YUV              | `Gray8` ↔ every YUV family (luma extraction / neutral-chroma synthesis)     |
| Gray ↔ RGB              | `Gray8` → all six packed RGB orders; `Rgb24` / `Rgba` → `Gray8` (luminance) |
| Grayscale / mono        | `Gray8` / `Gray16Le`, `MonoBlack` / `MonoWhite` ↔ `Gray8`                   |
| Grey + alpha            | `Ya8` ↔ `Gray8` / `Rgb24` / `Rgba` (luma broadcast, alpha carried through); `Ya16Le` ↔ `Ya8` (exact ×257 / high-byte), `Gray16Le` (luma word verbatim), `Gray8`, `Rgba64Le` (bit-exact broadcast + alpha carry) and `Rgb24` / `Rgba` |
| CMYK                    | `Cmyk` and `CmykInverted` ↔ `Rgb24` / `Rgba` (uncalibrated device formula, lossless RGB round-trip); `Cmyk` ↔ `CmykInverted` exact per-byte complement |
| YUV + alpha             | the full 12-member Yuva family — `Yuva420P` / `Yuva422P` / `Yuva444P` plus the deep 10/12/16-bit members at **all three** chroma sitings — ↔ siblings / each other / `Rgb24` / `Rgba` / `Gray8`; every ordered pair inside the family is direct; alpha bit-exact at same depth, MSB-widened / truncated across depths |
| Significant bits        | source frames may carry the per-plane significant-bits side-channel (e.g. 12-bit luma + 10-bit chroma on a `P16Le` surface): `convert()` treats marked planes at their recorded depth, rejects invalid records with `Error::Invalid`, and never propagates a stale record |
| Palette                 | `Pal8` ↔ `Rgb24` / `Rgba`, nearest-colour quantisation with optional dither; frames carry their table in-band via the `VideoFrame` palette side-channel |
| Colour matrices         | BT.601 / BT.709 / BT.2020, limited (studio) / full (JPEG) range             |
| Dither strategies       | None, 8×8 ordered Bayer, Floyd–Steinberg                                    |
| Alpha / compositing     | Porter-Duff "over" (premul + straight), premul/unpremul, alpha-mask blit    |
| Format introspection    | `FormatInfo::of(fmt)` → planes / bit-depth / `ChromaSubsampling` typed view |
| Planar-family engine    | a computed dispatch tier makes *every* ordered pair inside the uniform planar YUV(A) family ({Yuv,Yuva} × {420,422,444} × {8,10,12,16}) a direct single-step conversion: depth move + chroma resample (at the deeper of the two depths) + alpha handling fused in one op |
| Staged fallback         | pairs without a direct entry route through ONE fidelity-chosen pivot (deep YUV moves pivot through the 16-bit tier, alpha-carrying deep moves through `Yuva*P16Le`) — **all 3660 ordered format pairs convert** (976 direct); `supports()` / `supports_direct()` report availability |

## Roadmap

The pixel-format universe used by general-purpose video tooling runs
to roughly two hundred entries; this crate currently covers the 61
`PixelFormat` variants oxideav-core defines, and the conversion
matrix is **fully closed**: every one of the 61 × 60 = 3660 ordered
pairs resolves — 976 directly, the rest through one staged pivot. The
formats below are *planned* — they're not implemented yet, but they
have real callers in the codecs/containers we want to support, so the
[`PixelFormat`](https://docs.rs/oxideav-core/latest/oxideav_core/enum.PixelFormat.html)
variants and `convert()` paths will land over time.

**Tier 1 — short-term targets:**

| family                   | additions                                                                  |
| ------------------------ | -------------------------------------------------------------------------- |
| 16-bit packed RGB        | `Rgb565Le/Be`, `Rgb555Le/Be`, `Rgb444Le/Be` (+ BGR mirrors)                |
| Padded 4-byte packed RGB | `0Rgb`, `Rgb0`, `0Bgr`, `Bgr0` (no-alpha 32-bit, alignment-friendly)       |
| GBR planar               | shipped in full — the ten-member ladder (`Gbrp8` / `Gbrap8` through `Gbrap16Le`) with alpha-crossing deep-packed hops and `Gray8` interop (see table above) |
| Legacy planar YUV        | `Yuv410P`, `Yuv440P` (+ `YuvJ*` mirrors) — DV, MJPEG, SD                   |
| 4:2:2 / 4:4:4 NV         | `Nv16`, `Nv24` — common on Android / embedded                              |
| Alpha-bearing YUV        | shipped in full — the 8-bit trio **and** the deep 10/12/16-bit members at all three chroma sitings, `Yuva420P*` included (see table above) |

**Tier 2 — mid-term:**

| family                | additions                                                                     |
| --------------------- | ----------------------------------------------------------------------------- |
| Big-endian mirrors    | `Rgb48Be`, `Rgba64Be`, `Gray16Be`, `Yuv420P10Be`, … of every `*Le` we ship    |
| Higher-precision YUV  | `Yuv420P9/14Le`, same for `422` / `444` (the 8 ↔ 10 ↔ 12 ↔ 16 ladder, 16-bit chroma resample, direct high-bit ↔ RGB / Gray8 interop AND the full-precision 16-bit deep matrix shipped — see table above) |
| 10/12/16-bit semi-pl. | `P010Le`, `P012Le`, `P016Le` — HEVC Main10, Dolby Vision                      |
| DCI / cinema          | `Xyz12Le`                                                                     |
| 8-bit low-bpp packed  | `Rgb8` (3-3-2), `Rgb4`, `Bgr4Byte`                                            |

**Out of scope (no plans):**

- Hardware-opaque surfaces (`cuda`, `vaapi`, `vdpau`, `qsv`,
  `videotoolbox`, `vulkan`, `drm_prime`, `mediacodec`, …) — these are
  zero-copy GPU descriptors, not something a CPU pixfmt layer would
  convert. The framework will surface them at the codec/IO boundary
  instead, leaving the GPU contents untouched.
- Bayer mosaic patterns (`bayer_*`) — a RAW-camera concern, outside
  this crate's video-pipeline scope.
- Niche packed YUV (`Ayuv64Le`, `Vuya`, `Vuyx`) — open to PRs if a
  consumer needs them.

## Low-level API — work on your own buffers

Every hot path is exposed as a function over `&[u8]` / `&mut [u8]`. The
buffer layout is always tightly packed (no stride padding) — strip and
re-apply stride yourself if your frames carry one.

### YUV420P → Rgb24 (BT.709 limited range)

```rust
use oxideav_pixfmt::yuv::{yuv420_to_rgb24, YuvMatrix};

// Your decoded YUV planes. Y is full resolution, U/V are each w/2 × h/2.
let (w, h) = (1920, 1080);
let y_plane: Vec<u8> = /* w * h bytes */ vec![0; w * h];
let u_plane: Vec<u8> = /* (w/2) * (h/2) bytes */ vec![128; (w / 2) * (h / 2)];
let v_plane: Vec<u8> = /* (w/2) * (h/2) bytes */ vec![128; (w / 2) * (h / 2)];

let mut rgb = vec![0u8; w * h * 3];
yuv420_to_rgb24(&y_plane, &u_plane, &v_plane, &mut rgb, w, h, YuvMatrix::BT709);
```

`YuvMatrix::BT601` / `YuvMatrix::BT709` default to limited (studio)
range. Call `.with_range(false)` to pick full-range (JPEG-style) coefficients,
or `YuvMatrix::from_color_space(ColorSpace::Bt601Full)` if you already
have a `ColorSpace` value.

### RGB swizzle (RGBA ↔ BGRA / ARGB / ABGR, RGB ↔ BGR)

```rust
use oxideav_pixfmt::rgb::{swizzle4, BGRA_POS, RGBA_POS};

let src: Vec<u8> = /* w * h * 4 bytes of RGBA */ vec![0; 1920 * 1080 * 4];
let mut dst = vec![0u8; src.len()];
swizzle4(&src, RGBA_POS, &mut dst, BGRA_POS, 1920 * 1080);
```

`RGBA_POS`, `BGRA_POS`, `ARGB_POS`, `ABGR_POS` describe where each
component sits in a 4-byte packed pixel; `swizzle4` emits the permuted
output in a single AVX2 `pshufb` per 8 pixels. For 3-byte packings the
mirrored function `swizzle3` takes `RGB_POS` / `BGR_POS`. Alpha-promote
and alpha-drop variants are `rgb3_to_rgba4` and `rgba4_to_rgb3`.

### NV12 → YUV420P (split the interleaved UV plane)

```rust
use oxideav_pixfmt::yuv::nv12_uv_split;

// NV12: one Y plane + one interleaved UV plane (cw * ch pixels, 2 bytes each).
let (cw, ch) = (960, 540);
let uv: Vec<u8> = vec![0; cw * ch * 2];

let mut u_plane = vec![0u8; cw * ch];
let mut v_plane = vec![0u8; cw * ch];
nv12_uv_split(&uv, &mut u_plane, &mut v_plane, cw, ch);
```

`nv21_vu_split`, `nv12_uv_merge`, and `nv21_vu_merge` cover the other
directions.

### Palette quantisation (animated-GIF / APNG style)

```rust
use oxideav_pixfmt::pal8::quantise_rgb24_to_pal8;
use oxideav_pixfmt::{generate_palette, Dither, PaletteGenOptions, PaletteStrategy};

// Build a palette from one or more reference frames. `generate_palette`
// takes &[&VideoFrame]; see `palette::Palette` if you want to construct
// one from a raw colour list instead.
let frames: Vec<&oxideav_core::VideoFrame> = collect_reference_frames();
let palette = generate_palette(
    &frames,
    &PaletteGenOptions {
        max_colors: 255,                     // u8 — 1..=255
        strategy: PaletteStrategy::MedianCut,
        transparency: None,
    },
).expect("palette generation");

// Quantise a tightly-packed RGB24 buffer against the palette.
let (w, h) = (320, 240);
let rgb24: Vec<u8> = vec![0; w * h * 3];
let mut indices = vec![0u8; w * h];
quantise_rgb24_to_pal8(&rgb24, &mut indices, w, h, &palette, Dither::FloydSteinberg);
```

Decode back with `pal8::expand_row_to_rgb24` or `expand_row_to_rgba`,
which take a row of palette indices plus the `Palette` and emit the
corresponding RGB scanline.

### Alpha-blending and compositing

Porter-Duff "over" primitives — the small bricks font renderers,
subtitle compositors and overlay pipelines build on top of:

```rust
use oxideav_pixfmt::{
    blit_alpha_mask, modulate_alpha, over_buffer, over_premul,
    over_straight, premultiply, unpremultiply,
};

// Per-pixel composite (premultiplied or straight):
let out = over_premul([128, 0, 0, 128], [0, 0, 255, 255]);   // semi-red over blue
let out = over_straight([255, 0, 0, 128], [0, 0, 255, 255]); // straight-alpha equivalent

// Premultiply / unpremultiply roundtrip (lossless at A=255, lossy at low A):
let p = premultiply([200, 100, 50, 128]);
let s = unpremultiply(p);

// Modulate the alpha channel by an opacity value:
let dim = modulate_alpha([200, 100, 50, 255], 128); // 50% opacity

// Blit a coloured glyph mask onto an RGBA framebuffer with edge-clipping:
let (w, h) = (320, 240);
let mut canvas = vec![0u8; w * h * 4];
let glyph: Vec<u8> = /* mw * mh u8 alpha */ vec![255; 8 * 8];
blit_alpha_mask(
    &mut canvas, w as u32, h as u32, w * 4,
    /* x = */ 16, /* y = */ 24,
    &glyph, 8, 8, 8,
    [255, 255, 255, 255], // colour the mask in white, fully opaque
);

// Bulk over-composite two same-size buffers:
let mut dst = vec![0u8; w * h * 4];
let src = vec![0u8; w * h * 4];
over_buffer(&mut dst, &src, w as u32, h as u32, w * 4, /* premultiplied = */ true);
```

All `u8 × u8` math goes through a bit-exact rounded `(a × b + 128) / 256`
shift trick — no division on the hot path, no third-party deps.

## High-level API — `VideoFrame` in, `VideoFrame` out

If you're already using `oxideav-core`, the one-line form handles every
conversion through the same dispatch table:

```rust
use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ColorSpace, ConvertOptions, FrameInfo};

let src = VideoFrame {
    pts: None,
    planes: vec![
        VideoPlane { stride: 1920, data: vec![16; 1920 * 1080] },
        VideoPlane { stride: 960,  data: vec![128; 960 * 540]   },
        VideoPlane { stride: 960,  data: vec![128; 960 * 540]   },
    ],
};
let src_info = FrameInfo::new(PixelFormat::Yuv420P, 1920, 1080);

let dst = convert(
    &src,
    src_info,
    PixelFormat::Rgb24,
    &ConvertOptions {
        color_space: ColorSpace::Bt709Limited,
        ..Default::default()
    },
).expect("convert");
```

`convert_in_place_if_same(src, src_info, dst_format)` is a zero-copy passthrough
you can call first to skip `convert()` when the source already matches.

When no direct `(src, dst)` table entry exists, `convert()` resolves
the pair through **one staged pivot**, chosen in a fidelity-aware
order: YUV pivots for YUV → YUV moves (the route never touches a
colour matrix — e.g. `Yuyv422 → Yuv420P` keeps luma byte-exact), RGB
pivots otherwise with alpha-capable before alpha-less (`Yuva420P →
Bgra` carries alpha bit-exact) and deep before 8-bit when either
endpoint is deeper than 8 bits (`Gbrp10Le → Gbrp12Le` round-trips
exactly through `Rgb48Le`; deep YUV → YUV moves pivot through the
16-bit `Yuv*P16Le` tier so `Yuv420P16Le → Yuv422P10Le` keeps the top
10 bits of every sample). Staged routes are exactly as correct as
their two legs but can round twice; `supports(src, dst)` and
`supports_direct(src, dst)` let callers tell the two cases apart.

## Colour science

The matrix is selected at runtime via `ConvertOptions::color_space` (for
`convert()`) or `YuvMatrix` (for the low-level entry points):

| variant              | primaries | range   | use case                       |
| -------------------- | --------- | ------- | ------------------------------ |
| `Bt601Limited`       | BT.601    | 16–235  | SD video, MPEG/H.264 default   |
| `Bt601Full`          | BT.601    | 0–255   | JPEG with YCbCr SOF            |
| `Bt709Limited`       | BT.709    | 16–235  | HD video                       |
| `Bt709Full`          | BT.709    | 0–255   | full-range HD, certain codecs  |
| `Bt2020Limited`      | BT.2020   | 16–235  | UHDTV, HDR carriage (NCL)      |
| `Bt2020Full`         | BT.2020   | 0–255   | full-range UHDTV               |

BT.2020 uses the non-constant-luminance Y'CbCr coefficients from
ITU-R BT.2020-2 Table 4 (`kr = 0.2627, kb = 0.0593`); the same matrix
is reused for the BT.2100-3 Table 6 HDR signal format.

The Q15 fixed-point matrices are verified three ways in the test
suite: against an independent f64 model built straight from the
k-coefficient construction (±1 code, encode and decode, all six
variants), against pinned spec-derived primary anchor codes (BT.601
limited red = (81, 90, 240), BT.709 limited red = (63, 102, 240), …),
and — on machines that have one — against the `ffmpeg` binary invoked
as a black-box CLI validator (±2 on matrixed paths, bit-exact on pure
repack paths).

The 16-bit planar tier additionally carries a **full-precision deep
matrix**: the same k-coefficient construction evaluated in Q30 over
i64 directly on 16-bit samples, used by the `Yuv(a)*P16Le` ↔
`Rgb48Le` / `Rgba64Le` rows. Limited-range offsets follow the n-bit
digital representation of the BT-series specs (offsets and spans
scale by 2^(n−8); at n = 16 luma is 4096 + [0, 56064] and chroma is
centred on the exact achromatic code 32768), so full-scale white
encodes to Y = 60160 exactly and `r = g = b` content pins chroma to
32768. The Q30 kernels are verified against an independent f64 model
to ±1 LSB at 16-bit scale across all six matrix variants.

Range rescaling between `YuvJ*` (full) and `Yuv*` (limited) planes is
exposed both through `convert()` and directly as
`yuv::{limited_to_full_luma, limited_to_full_chroma, full_to_limited_luma, full_to_limited_chroma}`
so callers can flip range without going through RGB.

### Transfer functions

The `transfer` module exposes opto-electronic and electro-optical
transfer functions on `f32`. They are orthogonal to the YUV/RGB matrix
above — pair them as the HDR / SDR pipeline requires:

| function                  | spec source                          |
| ------------------------- | ------------------------------------ |
| `bt709_oetf` (+ inverse)  | ITU-R BT.2020-2 Table 4 (10-bit α, β) |
| `bt2020_12_oetf`          | ITU-R BT.2020-2 Table 4 (12-bit α, β) |
| `bt1886_eotf` (+ inverse) | ITU-R BT.1886 Annex 1, γ = 2.40       |
| `pq_eotf` / `pq_inverse_eotf` | SMPTE ST 2084 / ITU-R BT.2100-3 Table 4 |
| `hlg_oetf` (+ inverse)    | ITU-R BT.2100-3 Table 5               |
| `hlg_apply_ootf`          | BT.2100-3 Table 5 HLG OOTF row        |
| `hlg_system_gamma(l_w)`   | BT.2100-3 Note 5f                     |

PQ peak luminance is exposed as `PQ_PEAK_CDM2 = 10_000.0` (signal
value 1.0 maps to that luminance per BT.2100-3 Table 4). The HLG
inverse OETF returns relative scene light; pass it through
`hlg_apply_ootf` with the user gain α and system gamma γ to obtain
display-light values.

## Performance

Every converter has a scalar Q15 fixed-point reference; SIMD paths are
validated against it to ±1 LSB in the test suite. Dispatch is lazy and
cached on first call per process.

**1920×1080, single Intel i9-14900K core, AVX2 path:**

| operation                  | scalar         | AVX2 (this crate) |
| -------------------------- | -------------- | ----------------- |
| `yuv420_to_rgb24`          | 3.14 ms        | 720 µs (8.0 GiB/s)|
| `yuv444_to_rgb24`          | 1.24 ms (720p) | 296 µs (8.7 GiB/s, 720p) |
| `rgb24_to_yuv420`          | 2.20 ms (720p) | 547 µs (5.7 GiB/s, 720p) |
| `rgb24_to_yuv422`          | —              | 5.9 GiB/s (720p)  |
| `swizzle4` (RGBA ↔ BGRA)   | 3.5 GiB/s      | 29 GiB/s          |
| `rgb3_to_rgba4`            | 4.2 GiB/s      | 33.6 GiB/s        |
| `chroma_420_to_444`        | 2.1 GiB/s      | 48.8 GiB/s        |
| `chroma_422_to_444`        | 2.1 GiB/s      | 42.8 GiB/s        |
| `gray8_to_rgba`            | 7.0 GiB/s      | 43.7 GiB/s        |
| `rgb48_to_rgb24`           | 4.5 GiB/s      | 14.8 GiB/s        |
| `nv12_uv_split`            | —              | 32.3 GiB/s        |

The YUV decode path processes 16 pixels per AVX2 iteration; encode runs
a fused 2-row luma + 2×2-chroma loop that does one `pshufb` deinterleave
per 8 pixels and pair-sums the chroma via `pmaddubsw`.

**Porter-Duff compositing (scalar, Apple M-series single core, indicative):**

| operation                           | scalar throughput |
| ----------------------------------- | ----------------- |
| `over_premul` (per-pixel, 1 Mpx)    | ~3.5 GiB/s        |
| `over_straight` (per-pixel, 1 Mpx)  | ~2.1 GiB/s        |
| `over_buffer` premul (1920×1080)    | ~6.3 GiB/s        |
| `over_buffer` straight (1920×1080)  | ~2.1 GiB/s        |
| `blit_alpha_mask` 16×16 glyph       | ~276 MiB/s        |
| `blit_alpha_mask` 64×64 glyph       | ~282 MiB/s        |
| `premultiply` (1 Mpx)               | ~5.6 GiB/s        |
| `unpremultiply` (1 Mpx, includes divide) | ~4.7 GiB/s   |
| `modulate_alpha` (1 Mpx)            | ~6.3 GiB/s        |

The compositing primitives are still scalar (no SIMD path yet); the
numbers above are the headroom a future hand-vectorised pass would
land against. `over_straight` is the floor because each pixel pays for
the divide-by-`out.a` rebuild.

### Dispatch summary

| target         | path                                                             |
| -------------- | ---------------------------------------------------------------- |
| x86_64 + AVX2  | AVX2 intrinsics (`pshufb`, `pmaddubsw`, `vpermq`, …)             |
| aarch64 + NEON | NEON decode (`vld3_u8`-style); encode falls back to scalar       |
| nightly +      | `std::simd` path via the `nightly` feature (portable 8-wide)     |
| everything     | scalar fixed-point — golden reference used by the SIMD tests     |

## Runtime controls

These env vars are consulted once per process (before the first call),
then cached:

* `OXIDEAV_PIXFMT_FORCE_SCALAR=1` — pin every path to scalar. Useful
  for benchmark baselines and correctness debugging.
* `OXIDEAV_PIXFMT_FORCE_PORTABLE_SIMD=1` — with
  `--features nightly`, pick `std::simd` over the hand-written
  intrinsics.

## Benchmarks

Criterion is an optional dependency behind the `bench` feature (kept off
the default graph so the test binaries don't link its `alloca` native
dep), so the benches need `--features bench`:

```sh
cargo bench --features bench                              # all suites
cargo bench --features bench --bench yuv_rgb             # just YUV encode/decode
cargo bench --features bench --bench pixel_ops          # RGB swizzle, NV12, chroma resample, gray, deep-RGB
cargo bench --features bench --bench alpha              # Porter-Duff over/blit/premultiply
cargo bench --features bench --bench depth_gray        # bit-depth ladder + RGB→Gray8 projection
OXIDEAV_PIXFMT_FORCE_SCALAR=1 cargo bench --features bench  # scalar baseline for comparison
```

Indicative `depth_gray` single-core numbers (Apple M-series, scalar):
8 → 10-bit plane widen 84 GiB/s, 10 → 8-bit narrow 48 GiB/s,
10 ↔ 12-bit rescale 11.7 / 34.4 GiB/s, `rgb24_to_gray8` 17.3 GiB/s.

Portable-SIMD numbers (nightly only):

```sh
cargo +nightly bench --features bench,nightly
OXIDEAV_PIXFMT_FORCE_PORTABLE_SIMD=1 cargo +nightly bench --features bench,nightly
```

## Fuzzing

A [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) harness lives
under [`fuzz/`](fuzz/) and runs daily in CI. The `convert_geometry`
target does not feed arbitrary bytes at a parser — there is none. It
instead *constructs* a structurally-valid source frame from the fuzzer's
input (a source pixel format — all 61 enum variants are buildable,
small / odd dimensions, extra stride padding, and optional hostile
side-channel records: fuzz-length palettes on `Pal8` and
significant-bits records with zero bits, above-nominal values, and
wrong lengths) and drives every
`(src, dst)` conversion, direct and staged alike, asserting
that none panics, integer-overflows, reads out of bounds, or aborts. A
converter may legitimately return `Err` for geometry it cannot represent
(e.g. an odd width on a 4:2:0 layout) or for an invalid record; only a
crash is a finding. This
target's first run caught an out-of-bounds chroma read on subsampled
YUV → RGB at odd dimensions.

```sh
cargo +nightly fuzz run convert_geometry
```

## License

MIT — see [LICENSE](LICENSE).
