# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Packed 4:2:2 conversions for `Yuyv422` (Y0 U0 Y1 V0) and `Uyvy422`
  (U0 Y0 V0 Y1). Both variants had `FormatInfo::of` entries since
  ~0.1.2 but no actual conversion path, so any caller hitting V4L2 /
  USB-webcam / DV / AVI feeds with `YUYV` or `UYVY` codec tags got
  `Error::Unsupported` from `convert()`. Now wired:
  - `Yuyv422` ↔ `Yuv422P` and `Uyvy422` ↔ `Yuv422P` — pure
    deinterleave / interleave shuffles with no colour math.
  - `Yuyv422` ↔ `Uyvy422` — two byte-swaps per 4-byte quad
    (`Y0 U` ↔ `U Y0`, `Y1 V` ↔ `V Y1`), exposed as the involutive
    `yuv::yuyv_uyvy_swap` helper.
  - `Yuyv422` / `Uyvy422` ↔ `Rgb24` / `Rgba` direct entries that
    bridge through the existing planar 4:2:2 colour math under any
    `ColorSpace` (BT.601/709/2020, limited/full). PSNR on a smooth
    gradient round-trip stays > 46 dB at BT.601 limited.
  - Odd widths are rejected with `Error::Invalid` (packed 4:2:2 has
    no representation for an unpaired luma sample) rather than
    silently truncating.
  - New `tests/packed_yuv422.rs` plus 3 unit tests in `yuv.rs` pin
    byte positions on hand-built 2×1 / 4×1 frames and check
    bit-exact planar↔packed round-trips on 16×8 / 32×16 random
    content. Total new test coverage: +13 tests.
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

### Fixed

- **`rgb48_to_rgb24` AVX2 path was writing 4 bytes past the end of `dst`
  on every call.** The SIMD inner loop stored a full 16-byte register
  every 2 pixels (advancing the destination cursor by 6) and only
  reserved a 2-pixel scalar tail (6 bytes), so the final iteration's
  10-byte spill ran 4 bytes past `dst[..pixels*3]`. On multi-row
  buffers the stomp landed in the next row (where it was promptly
  overwritten); on the **final** row the bytes went past the backing
  `Vec`'s allocation. When those happened to be glibc allocator
  metadata, malloc's `sysmalloc` assertion fired and aborted the
  process — observed on x86_64 Linux / Windows CI as
  `Fatal glibc error: malloc.c:2599 (sysmalloc): assertion failed`
  whenever a property-style sweep raised allocator churn enough to
  surface the corruption (commit `80e1d09` → reverted). Fix: enlarge
  the scalar-tail reserve from 2 to 6 pixels so the SIMD store's full
  16-byte footprint always fits inside the destination slice, and pin
  the contract with a regression unit test that runs `pixels = 1..=64`
  through tightly-sized `Vec` destinations plus an alloc-churn loop.
- **Pinned the over-store contract on the three sibling AVX2 swizzle
  paths that share the `rgb48_to_rgb24` bug shape.** `swizzle3`
  (15-byte advance, 16-byte store) and `rgba4_to_rgb3` (12-byte advance,
  16-byte store) emit a full register per iteration and rely on a
  reserved scalar tail (`pixels - 5` / `pixels - 4`) to keep the final
  store inside `dst`. Both were already correct, but had no regression
  test that would catch a shrunk tail reserve: the existing parity tests
  compare against an oversized scratch buffer, so a reintroduced over-run
  would stomp adjacent heap *without* failing a value assertion (exactly
  the silent-corruption mode that turned into a glibc `sysmalloc` abort
  for `rgb48_to_rgb24`). Added `src/rgb/swizzle_simd.rs` regression tests:
  tight-fit `dst` Vecs (backing allocation ends exactly at the output
  length) across `pixels = 1..=80`, plus a 2000-iteration alloc-churn
  loop that trips the allocator on any leaked over-write. `swizzle4` and
  `rgb3_to_rgba4` (store == advance, can't over-run) are covered too to
  lock in their read-side tail reserves. +5 unit tests; no API change.

### Changed

- `criterion` is now an optional dependency behind a new `bench` feature
  (the two `[[bench]]` targets carry `required-features = ["bench"]`).
  Criterion pulls the `alloca` native crate, whose glibc allocator hooks
  abort the integration-test process on Linux (`Fatal glibc error:
  malloc.c sysmalloc assertion failed`). Keeping criterion off the
  default graph stops it linking into the test binaries; run benches with
  `cargo bench --features bench`. No library-API change.

### Added

- Re-introduce `tests/property.rs`: randomised xorshift PRNG sweep over
  the conversion surface (RGB swizzles, 3↔4 alpha promote/demote,
  8↔16-bit promote/demote, NV12/NV21↔Yuv420P interleave, YuvJ↔Yuv
  range rescale, panic-freedom, YUV↔RGB tolerance/PSNR floors,
  premul/unpremul bound). Originally landed in `80e1d09`, reverted in
  `6ea6cc7` because it exposed the `rgb48_to_rgb24` AVX2 overflow
  above. With that bug fixed the suite passes on every CI target.

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
