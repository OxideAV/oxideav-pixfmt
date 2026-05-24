# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Randomised property / round-trip test suite (`tests/property.rs`).
  A self-contained xorshift PRNG (no new dependency) sweeps the whole
  conversion table with pseudo-random pixel buffers across many
  dimensions and non-tight source strides. It pins the lossless families
  as byte-exact (RGB swizzles, 3↔4 alpha promote/demote, 8↔16-bit
  promote/demote, NV12/NV21 ↔ Yuv420P interleave, YuvJ↔Yuv rescale
  stability), bounds the tolerance families (YUV↔RGB 4:4:4 through the
  Q15 matrices at ≤3 LSB/channel — measured worst 2 over 300 000
  pixels × 3 matrices × 2 ranges; `premultiply↔unpremultiply` at the
  documented `ceil(255/a)`, ≤1 LSB for α ≥ 128), checks full-frame
  YUV↔RGB PSNR floors (4:4:4 > 36 dB, 4:2:2 > 32 dB, 4:2:0 > 29 dB)
  across all six colour spaces, and asserts panic-freedom (Ok-or-clean-
  Err, never panic) for every supported pair plus the odd-dimension
  RGB→4:2:0 divisibility guard. Hardening only — no behaviour change.
- `FormatInfo::of` arms for the 6 high-bit-depth planar GBR(A) variants
  (`Gbrp10Le`, `Gbrap10Le`, `Gbrp12Le`, `Gbrap12Le`, `Gbrp14Le`,
  `Gbrap14Le`) added in `oxideav-core` 0.1.18, plus the previously
  missing `Yuv422P12Le` / `Yuv444P12Le`. Without these, dev/test
  consumers tripped the wildcard arm's `debug_assert!` and got a
  conservative single-plane fallback descriptor. Each new variant now
  reports the correct plane count, bit depth, alpha flag and planar
  layout. Tests pin every value.
- `YuvMatrix::BT2020` plus `ColorSpace::Bt2020Limited` /
  `ColorSpace::Bt2020Full` variants. The non-constant-luminance
  Y'CbCr matrix from ITU-R BT.2020-2 Table 4 (`kr = 0.2627,
  kb = 0.0593`) is now selectable via the same `ConvertOptions`
  path as the BT.601 / BT.709 matrices. The same coefficients
  cover BT.2100-3 Table 6 HDR signalling. Roundtrip tests pin
  PSNR > 38 dB (limited) and > 42 dB (full) on a synthetic gradient,
  and confirm neutral-grey (`R=G=B=128`) projects onto
  `Cb=Cr=128 ± 1` for both range modes.
- `transfer` module — opto-electronic / electro-optical transfer
  functions on `f32`, sourced clean-room from
  `docs/video/signal-metadata/`:
  - `bt709_oetf` / `bt709_inverse_oetf` (10-bit constants per
    BT.2020-2 Table 4).
  - `bt2020_12_oetf` / `bt2020_12_inverse_oetf` (12-bit precision).
  - `bt1886_eotf` / `bt1886_inverse_eotf` + `bt1886_eotf_with_levels`
    for non-zero black / non-unity peak white (BT.1886 Annex 1,
    γ = 2.40).
  - `pq_eotf` / `pq_inverse_eotf` (SMPTE ST 2084 / BT.2100-3 Table 4,
    peak 10 000 cd/m², constants m1, m2, c1, c2, c3 quoted in source).
  - `hlg_oetf` / `hlg_inverse_oetf` + `hlg_apply_ootf` /
    `hlg_inverse_ootf` (BT.2100-3 Table 5, with `a = 0.17883277,
    b = 0.28466892, c = 0.55991073`).
  - `hlg_system_gamma(l_w)` per BT.2100-3 Note 5f.
  Tests pin (a) round-trip OETF/EOTF identity to ±5e-5 across the
  unit interval, (b) spec anchor points (PQ E'=1 → 10 000 cd/m²;
  HLG E=1/12 → E'=0.5; BT.1886 V=1 → L=1), and (c) monotonicity
  across each curve.

## [0.1.5](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.1.4...v0.1.5) - 2026-05-03

### Other

- handle PixelFormat::Yuv411P
- replace never-match regex with semver_check = false

## [0.1.4](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.1.3...v0.1.4) - 2026-05-03

### Other

- add Porter-Duff over + glyph-mask blit primitives
- bump v4 -> v6
- add miri job (org-wide policy, custom CI variant)

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
