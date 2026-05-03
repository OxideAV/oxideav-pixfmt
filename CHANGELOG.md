# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `alpha` module — Porter-Duff "over" composite primitives for RGBA
  buffers. Per-pixel `over_premul` / `over_straight`, `premultiply` /
  `unpremultiply`, `modulate_alpha`, full-buffer `over_buffer`, and
  glyph-style `blit_alpha_mask` (single-channel u8 mask × RGBA colour
  → RGBA framebuffer, with destination clipping). All re-exported from
  the crate root for the upcoming `oxideav-scribe` font crate and any
  future subtitle / overlay compositor. No new third-party deps.

## [0.1.3](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.1.2...v0.1.3) - 2026-05-02

### Other

- stay on 0.1.x during heavy dev (semver_check=false)
- drop redundant 'a lifetime on convert_in_place_if_same
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

## [0.1.2](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.1.1...v0.1.2) - 2026-04-24

### Other

- bump criterion 0.5 → 0.8
- drop Cargo.lock — this crate is a library

## [0.1.1](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.1.0...v0.1.1) - 2026-04-19

### Other

- bump oxideav-core to 0.1.2
- add CMYK pixel format

## [0.0.5](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.0.4...v0.0.5) - 2026-04-18

### Other

- rustfmt + clippy needless_range_loop / implicit_saturating_sub
- *(readme)* dual-track usage for standalone + oxideav callers
- *(yuv)* AVX2 RGB→YUV encode with pshufb load and chroma pair-sum
- AVX2 chroma upsample, gray→rgba, rgb48→rgb24, NV split
- *(rgb)* AVX2 pshufb swizzle — 7× on swizzle, promote, demote
- cover RGB swizzle, NV12, gray, deep-RGB, chroma resample
- add nightly job exercising --features nightly portable_simd
