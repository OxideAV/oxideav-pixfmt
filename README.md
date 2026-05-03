# oxideav-pixfmt

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
| Deep RGB                | `Rgb48Le` ↔ `Rgb24`, `Rgba64Le` ↔ `Rgba`                                    |
| YUV planar ↔ RGB        | `Yuv420P` / `Yuv422P` / `Yuv444P` ↔ `Rgb24` / `Rgba`                        |
| Chroma subsampling      | `4:4:4` ↔ `4:2:2` ↔ `4:2:0` (SIMD-accelerated up- and down-sample)          |
| Semi-planar             | `NV12` / `NV21` ↔ `Yuv420P`                                                 |
| Full ↔ limited range    | `YuvJ420P` / `YuvJ422P` / `YuvJ444P` ↔ `Yuv*`                               |
| Grayscale / mono        | `Gray8` / `Gray16Le`, `MonoBlack` / `MonoWhite` ↔ `Gray8`                   |
| Palette                 | `Pal8` ↔ `Rgb24` / `Rgba`, nearest-colour quantisation with optional dither |
| Colour matrices         | BT.601 / BT.709, limited (studio) / full (JPEG) range                       |
| Dither strategies       | None, 8×8 ordered Bayer, Floyd–Steinberg                                    |
| Alpha / compositing     | Porter-Duff "over" (premul + straight), premul/unpremul, alpha-mask blit    |

## Roadmap

ffmpeg's `-pix_fmts` lists ~200 entries; this crate currently covers
~28. The remaining gap is mostly long-tail or hardware-specific. The
formats below are *planned* — they're not implemented yet, but they
have real callers in the codecs/containers we want to support, so the
[`PixelFormat`](https://docs.rs/oxideav-core/latest/oxideav_core/enum.PixelFormat.html)
variants and `convert()` paths will land over time.

**Tier 1 — short-term targets:**

| family                   | additions                                                                  |
| ------------------------ | -------------------------------------------------------------------------- |
| 16-bit packed RGB        | `Rgb565Le/Be`, `Rgb555Le/Be`, `Rgb444Le/Be` (+ BGR mirrors)                |
| Padded 4-byte packed RGB | `0Rgb`, `Rgb0`, `0Bgr`, `Bgr0` (no-alpha 32-bit, alignment-friendly)       |
| GBR planar               | `Gbrp`, `Gbrp10/12/16Le` — JPEG-2000, ProRes 4444, lossless H.264 GBR mode |
| Legacy planar YUV        | `Yuv410P`, `Yuv440P` (+ `YuvJ*` mirrors) — DV, MJPEG, SD                   |
| 4:2:2 / 4:4:4 NV         | `Nv16`, `Nv24` — common on Android / embedded                              |
| Alpha-bearing YUV        | `YuvA422P`, `YuvA444P`, plus 10/12/16Le siblings of `Yuva420P`             |

**Tier 2 — mid-term:**

| family                | additions                                                                     |
| --------------------- | ----------------------------------------------------------------------------- |
| Big-endian mirrors    | `Rgb48Be`, `Rgba64Be`, `Gray16Be`, `Yuv420P10Be`, … of every `*Le` we ship    |
| Higher-precision YUV  | `Yuv420P9/14/16Le`, same for `422` / `444`                                    |
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

## Colour science

The matrix is selected at runtime via `ConvertOptions::color_space` (for
`convert()`) or `YuvMatrix` (for the low-level entry points):

| variant              | primaries | range   | use case                     |
| -------------------- | --------- | ------- | ---------------------------- |
| `Bt601Limited`       | BT.601    | 16–235  | SD video, MPEG/H.264 default |
| `Bt601Full`          | BT.601    | 0–255   | JPEG with YCbCr SOF          |
| `Bt709Limited`       | BT.709    | 16–235  | HD video                     |
| `Bt709Full`          | BT.709    | 0–255   | full-range HD, certain codecs |

Range rescaling between `YuvJ*` (full) and `Yuv*` (limited) planes is
exposed both through `convert()` and directly as
`yuv::{limited_to_full_luma, limited_to_full_chroma, full_to_limited_luma, full_to_limited_chroma}`
so callers can flip range without going through RGB.

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

```sh
cargo bench                                      # all suites
cargo bench --bench yuv_rgb                      # just YUV encode/decode
cargo bench --bench pixel_ops                    # RGB swizzle, NV12, chroma resample, gray, deep-RGB
OXIDEAV_PIXFMT_FORCE_SCALAR=1 cargo bench        # scalar baseline for comparison
```

Portable-SIMD numbers (nightly only):

```sh
cargo +nightly bench --features nightly
OXIDEAV_PIXFMT_FORCE_PORTABLE_SIMD=1 cargo +nightly bench --features nightly
```

## License

MIT — see [LICENSE](LICENSE).
