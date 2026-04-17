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

## Usage

```toml
[dependencies]
oxideav-pixfmt = "0.0"
```

Codecs declare `accepted_pixel_formats` on their `CodecCapabilities`;
the job-graph resolver auto-inserts a `PixConvert` stage when the
upstream format doesn't match.

## License

MIT — see [LICENSE](LICENSE).
