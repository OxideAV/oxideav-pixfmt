# oxideav-pixfmt

Shared pixel-format conversion layer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) pure-Rust media
framework:

* **`convert(src, dst_fmt, opts)`** — RGB all-to-all, YUV planar ↔ RGB
  under BT.601 / BT.709 × limited / full range, NV12/NV21 ↔ Yuv420P,
  Gray ↔ RGB, 16-bit ↔ 8-bit, Pal8 ↔ RGB.
* **`generate_palette(frames, opts)`** — MedianCut / Uniform across one
  or many frames; used by animated-GIF / APNG encoders.
* **Dithering**: None / 8×8 ordered Bayer / Floyd-Steinberg.
* **30+ PixelFormat variants** — RGB family, YUV planar at 8 / 10 / 12
  bit, NV12/NV21, YUV packed, Gray, Pal8, MonoBlack/White, alpha-bearing.

Zero C dependencies. Zero FFI.

## Colour coverage

Both **BT.601** and **BT.709**, in **limited (studio)** and **full
(JPEG)** range, are covered for every YUV ↔ RGB converter. The matrix is
selected at runtime via `ConvertOptions::color_space`. Range rescaling
between `YuvJ*` (full) and `Yuv*` (limited) planes is exposed as a
dedicated converter so callers can change range without going through
RGB.

## Performance

The YUV ↔ RGB inner loops dispatch to the fastest path available on the
host CPU:

* **Scalar fixed-point** — portable Q15 integer math; the golden
  fallback and the reference against which SIMD outputs are validated
  (±1 LSB tolerance).
* **AVX2** (`x86_64`) — 16 pixels / iteration on the decode side,
  8 pixels / iteration on the encode side.
* **NEON** (`aarch64`) — 8 pixels / iteration; decode only (encode
  falls through to the scalar path).
* **`std::simd`** — portable 8-wide decode, gated behind the `nightly`
  feature; requires a nightly toolchain.

Dispatch is picked lazily on first call. Set
`OXIDEAV_PIXFMT_FORCE_SCALAR=1` to disable SIMD (useful for
benchmarking and regression isolation). When built with
`--features nightly`, `OXIDEAV_PIXFMT_FORCE_PORTABLE_SIMD=1` picks the
`std::simd` path instead of the intrinsics.

### Benchmark numbers (1280×720, BT.709 limited, single Intel i9-14900K core)

| operation            | f32 scalar  | fixed-point scalar | AVX2        |
| -------------------- | ----------- | ------------------ | ----------- |
| `yuv420_to_rgb24`    | 7.22 ms     | 1.40 ms (5.2×)     | 320 µs (23×) |
| `yuv444_to_rgb24`    | 4.50 ms     | 1.24 ms (3.6×)     | 294 µs (15×) |
| `rgb24_to_yuv420`    | 7.82 ms     | 2.20 ms (3.6×)     | 769 µs (10×) |

1920×1080 `yuv420_to_rgb24` drops from 16.5 ms (f32) / 3.14 ms
(fixed-point) to **720 µs** on AVX2 — ≈8 GiB/s.

## Usage

```toml
[dependencies]
oxideav-pixfmt = "0.0"
```

Codecs declare `accepted_pixel_formats` on their `CodecCapabilities`;
the job-graph resolver auto-inserts a `PixConvert` stage when the
upstream format doesn't match.

## Benchmarks

```
cargo bench --bench yuv_rgb
```

The suite covers 720p/1080p YUV420 → RGB24, YUV444 → RGB24, and
RGB24 → YUV420. `OXIDEAV_PIXFMT_FORCE_SCALAR=1 cargo bench` produces
the scalar-path baseline for comparison.

To enable the portable-SIMD path (nightly only):

```
cargo +nightly bench --bench yuv_rgb --features nightly
# then OXIDEAV_PIXFMT_FORCE_PORTABLE_SIMD=1 to actually select it
```

## License

MIT — see [LICENSE](LICENSE).
