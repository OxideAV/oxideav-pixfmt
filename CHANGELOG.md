# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.5](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.0.4...v0.0.5) - 2026-04-18

### Other

- rustfmt + clippy needless_range_loop / implicit_saturating_sub
- *(readme)* dual-track usage for standalone + oxideav callers
- *(yuv)* AVX2 RGB→YUV encode with pshufb load and chroma pair-sum
- AVX2 chroma upsample, gray→rgba, rgb48→rgb24, NV split
- *(rgb)* AVX2 pshufb swizzle — 7× on swizzle, promote, demote
- cover RGB swizzle, NV12, gray, deep-RGB, chroma resample
- add nightly job exercising --features nightly portable_simd
