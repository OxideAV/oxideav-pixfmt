# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(convert)* honour the per-plane significant-bits side-channel from
  oxideav-core 0.1.31: marked planes convert at their recorded depth
  (normalised to the surface's nominal depth by the MSB-replicating
  widen before dispatch — a `Yuv444P16Le` frame marked `[12, 10, 10]`
  converts as 12-bit luma + 10-bit chroma, not full-range 16); outputs
  are always nominal-depth and never carry a stale record; values of 0
  or above the surface's nominal depth reject with `Error::Invalid`;
  records on `Pal8` are ignored and compose with the palette
  side-channel
- *(convert)* computed planar-family dispatch tier: every ordered pair
  inside the uniform planar YUV(A) family ({Yuv,Yuva} × {420,422,444} ×
  {8,10,12,16}) is now a direct single-step conversion — depth move,
  chroma resample (performed at the deeper of the two depths) and alpha
  carry/drop/opaque-synthesis fused in one op — plus direct
  `Rgb24`/`Rgba`/`Gray8` interop for every family member. Closes the
  cross-depth + cross-subsampling gap (e.g. `Yuv420P10Le → Yuv422P12Le`)
  left by the previous round; ordered-pair coverage 1379/2070 →
  2480/2652 (714 direct), and every pair involving the six new deep Yuva
  formats resolves
- *(format_info)* descriptors for the six deep Yuva formats added in
  oxideav-core 0.1.31 (`Yuva422P10Le`/`12Le`/`16Le`,
  `Yuva444P10Le`/`12Le`/`16Le`): 4-plane planar, full-resolution alpha,
  16-bit LE words with 10/12/16 significant bits
- *(fuzz)* convert_geometry now builds the six deep Yuva formats as
  sources and attaches hostile significant-bits records (zero bits,
  above-nominal values, wrong lengths, composed with the Pal8 palette
  record) — the no-panic contract covers the whole side-channel policy

### Changed

- `Gray8` → deep planar YUV now synthesises chroma at the exact neutral
  mid-code `1 << (bits - 1)` (512 at 10 bits, 32768 at 16) instead of
  the widened 8-bit 128 the old staged route produced, and 8-bit ↔ deep
  cross-subsampling moves resample chroma at the deeper width before
  narrowing (previously quantised through an 8-bit pivot first)

## [0.1.6](https://github.com/OxideAV/oxideav-pixfmt/compare/v0.1.5...v0.1.6) - 2026-07-09

### Added

- *(gbr)* planar GBR(A) <-> 8-bit packed RGB(A) direct rows (coverage 883 -> 1135)
- *(convert)* single-pivot staged fallback (883/1640 pairs reachable) + RGB->Gray8 projection
- *(convert)* Gray8 <-> YUV family (19 pairs) + complete bit-depth ladder (16 pairs)
- *(yuvj)* direct full-range YuvJ420P/422P/444P <-> Rgb24/Rgba conversions
- *(gbr)* wire planar GBR(A) ↔ packed deep-RGB into convert() dispatch
- *(yuv411)* wire Yuv411P into convert() dispatch + RGB encode/decode
- *(convert)* direct planar YUV ↔ planar YUV chroma resample
- *(yuva)* wire Yuva420P ↔ Yuv420P / Rgb24 / Rgba into convert() dispatch
- *(nv)* direct NV12 / NV21 ↔ Rgb24 / Rgba in the convert() dispatch
- *(ya8)* wire Ya8 ↔ Gray8 / Rgb24 / Rgba into the convert() dispatch
- *(yuv)* packed 4:2:2 (YUYV / UYVY) ↔ Yuv422P / RGB conversions

### Fixed

- *(yuv)* reject odd-dimension subsampled YUV→RGB + add convert_geometry fuzz target
- *(rgb48_to_rgb24)* AVX2 path wrote 4 bytes past dst — glibc abort

### Other

- *(miri)* keep the CI miri job fast and process-spawn-free
- README + crate-doc refresh for the round-399 conversion subsystem work
- *(depth)* bench suite for depth/gray plane primitives + 2.7x narrowing rescale
- *(xcheck)* black-box cross-validation against the ffmpeg CLI validator
- *(matrix)* f64 reference-model verification + BT.601/709 primary anchor pins
- add CI / crates.io / docs.rs / MIT-license badges
- add high-bit YUV variants to convert_geometry fuzz FORMATS
- high-bit YUV planar ↔ 8-bit bit-depth conversion (10/12-bit)
- *(miri)* shrink palette + property corpora under cfg(miri) — unhang the CI miri job
- *(yuv)* pin BT.2020 NCL black/primary anchor vectors + extreme-point invertibility
- drop release-plz.toml — use release-plz defaults across the workspace
- typed ChromaSubsampling view + is_chroma_subsampled
- *(alpha)* property sweep + Criterion suite for Porter-Duff primitives
- *(swizzle)* pin AVX2 over-store contract on swizzle3 / rgba4_to_rgb3
- Revert "tests: merge prop_* suite into tests/conversions.rs to avoid Linux SIMD abort"
- merge prop_* suite into tests/conversions.rs to avoid Linux SIMD abort
- gate criterion behind a `bench` feature to fix glibc abort in test binaries
- randomised property / round-trip suite for the conversion surface
- skip the heavy SIMD parity sweeps under miri
- trim 1001-pt sweeps to 201-pt under miri
- add BT.2020/2100 matrix + PQ/HLG/BT.1886 transfer functions
- cover GBR(A) 10/12/14-bit + Yuv422/444P12Le

### Fixed

- Out-of-bounds chroma read on subsampled planar YUV → packed RGB at
  odd dimensions. `Yuv420P` / `Yuv422P` / `Yuv411P` (and the `YuvJ*`
  full-range equivalents) → `Rgb24` / `Rgba` truncated the chroma plane
  size as `cw = w / wsub` / `ch = h / hsub`, so the decoder read one
  sample past the U/V planes for a trailing odd luma column or row (e.g.
  a 3×4 4:2:0 frame, or any 1×1 input) — a panic in debug builds, a
  silent OOB in release. `convert()` now rejects such geometry up front
  with `Error::Invalid`, mirroring the existing RGB → YUV divisibility
  guard. The other subsampled paths (NV12/NV21, packed 4:2:2, planar
  chroma resample, `Yuva420P`) already had this guard.

### Added

- Planar GBR(A) ↔ 8-bit packed RGB(A) direct conversions (12 new
  pairs): `Gbrp10/12/14Le` ↔ `Rgb24` and `Gbrap10/12/14Le` ↔ `Rgba`,
  folding the plane reorder and the bit-depth step into one pass
  (narrow = keep top 8 bits, widen = MSB replication, so 8-bit content
  round-trips exactly). Combined with the staged fallback this bridges
  the GBR families to the whole 8-bit ecosystem — YUV, gray, palette,
  packed 4:2:2 — lifting total reachable coverage from 883 to 1135 of
  the 1640 ordered pairs. The staged pivot order now also prefers the
  deep packed pivots whenever either endpoint carries more than 8
  significant bits, so deep → deep routes (e.g. `Gbrp10Le → Gbrp12Le`)
  never quantise through an 8-bit intermediate.

- `depth_gray` Criterion bench suite (`cargo bench --features bench
  --bench depth_gray`) covering the per-plane bit-depth primitives and
  the Gray8 luminance projection at 1080p, and a loop restructuring in
  `yuv::depth_rescale_le16_plane` that hoists the widen/narrow branch
  out of the per-sample body: the narrowing direction (12-bit → 10-bit)
  improves ~2.7× (12.9 → 34.4 GiB/s single-core), widening ~8%
  (10.9 → 11.7 GiB/s).

- Black-box cross-validation suite against the `ffmpeg` binary used as
  an opaque CLI validator (skips silently when no binary is on `PATH`,
  so CI stays self-contained). Five agreements pinned: limited-range
  4:4:4 → RGB under BT.601 and BT.709 (±2), RGB → limited 4:4:4 (±2
  per plane), the new direct full-range 4:4:4 → RGB path (±2),
  4:2:0 → RGB on upsampler-neutral flat-chroma fixtures (±2), and a
  bit-exact YUYV → planar 4:2:2 deinterleave.

- Independent f64 reference-model verification of the Q15 fixed-point
  colour matrices. A test-only f64 implementation built straight from
  the k-coefficient construction (BT.601-7 / BT.709-6 / BT.2020-2
  Table 4 NCL) and the 8-bit quantisation rules (Y' = 16 + 219·E'_Y,
  C' = 128 + 224·E'_C limited; identity full) now cross-checks encode
  AND decode for all six `ColorSpace` variants over a dense sweep —
  every channel within ±1 code. Classic limited-range primary anchors
  are pinned as literals (BT.601 red (81, 90, 240) / green (145, 54,
  34) / blue (41, 240, 110); BT.709 red (63, 102, 240) / green (173,
  42, 26) / blue (32, 240, 118)), plus full-range rails, gray-identity
  and ±2 round-trip properties.

- Single-pivot staged conversion fallback in `convert()`. When no
  direct `(src, dst)` table entry exists, the dispatcher now routes
  through one intermediate format, trying pivots in a fidelity-aware
  order: YUV pivots first when both endpoints are YUV carriage (so the
  route stays free of any colour matrix — e.g. `Yuyv422 → Yuv420P`
  keeps luma byte-exact), RGB pivots first otherwise with alpha-capable
  before alpha-less (`Yuva420P → Bgra` carries the alpha plane
  bit-exact) and deep before 8-bit where it matters (`Gbrp10Le ↔
  Gbrp12Le` round-trips exactly through `Rgb48Le`). Reachable
  `convert()` coverage jumps from 205 to 883 of the 1640 ordered format
  pairs; a matrix test pins both floors. New `supports(src, dst)` /
  `supports_direct(src, dst)` predicates let callers distinguish staged
  routes (which can round twice) from single-table-entry ones.

- Direct `Rgb24` / `Rgba` → `Gray8` luminance projection using the Y'
  row of the selected primaries at full range (r = g = b inputs map to
  themselves exactly), plus `Gray8` → `Bgr24` / `Bgra` / `Argb` /
  `Abgr` broadcasts (a gray broadcast is byte-identical for RGB and BGR
  orders; the alpha-first orders place the opaque byte up front). New
  low-level primitive `yuv::rgb24_to_gray8`.

- Complete bit-depth ladder (16 new `convert()` pairs). Cross-depth
  planar YUV 10 ↔ 12 bit (`Yuv420P10Le` ↔ `Yuv420P12Le` and the
  4:2:2 / 4:4:4 siblings, both directions) — previously a 10 ↔ 12 move
  had to stage through the 8-bit sibling and lose the low bits of both
  depths; the direct rescale widens with MSB replication (peak → peak,
  10 → 12 → 10 bit-exact) and narrows by truncation. And deep-grayscale
  wiring: `Gray10Le` / `Gray12Le` — until now the only `PixelFormat`
  variants with zero conversion coverage — join the ladder with
  `Gray8` ↔ `Gray10Le` / `Gray12Le` (8-bit values round-trip exactly),
  `Gray10Le` ↔ `Gray12Le`, and `Gray10Le` / `Gray12Le` ↔ `Gray16Le`.
  New low-level primitive `yuv::depth_rescale_le16_plane` exposes the
  per-plane width rescale over `&[u8]`.

- Gray8 ↔ YUV-family conversions (19 new `convert()` pairs). YUV →
  `Gray8` extracts the full-resolution luma plane from any planar
  (`Yuv420P` / `Yuv422P` / `Yuv444P` / `Yuv411P`), full-range (`YuvJ*`),
  semi-planar (`Nv12` / `Nv21`) or alpha-carrying (`Yuva420P`) source —
  rescaling limited 16..=235 luma onto the full Gray8 range and copying
  `YuvJ*` luma verbatim; chroma (and alpha) is never read, so odd
  dimensions are accepted even on subsampled layouts. `Gray8` → YUV
  synthesises neutral (128) chroma around the (range-compressed) gray
  plane for all seven planar targets plus NV12/NV21. No colour matrix is
  involved in either direction; the `YuvJ*` round-trip is bit-exact and
  the limited-range round-trip is within ±1.

- Direct full-range `YuvJ420P` / `YuvJ422P` / `YuvJ444P` ↔ `Rgb24` /
  `Rgba` conversions (12 new `convert()` pairs). The `YuvJ*` families
  carry full-range samples by definition, so these paths pin the matrix
  to full range — `ConvertOptions::color_space` still selects the
  primaries (BT.601 / BT.709 / BT.2020) but its range half is
  overridden by the format. Previously a caller had to stage through
  the limited-range sibling (`YuvJ* → Yuv* → RGB`), paying an extra
  255→219/224 range squeeze and its quantisation error both ways; the
  direct path costs a single fixed-point matrix and keeps the
  full-range 4:4:4 round-trip tight to ±2.

- Bit-depth conversion between the high-precision planar YUV variants
  (`Yuv420P10Le` / `Yuv422P10Le` / `Yuv444P10Le` and the 12-bit
  siblings) and their 8-bit counterparts (`Yuv420P` / `Yuv422P` /
  `Yuv444P`), in both directions and for all three subsampling layouts
  (12 new `convert()` pairs). Up-conversion places the 8-bit value in the
  high bits of the `bits`-significant 16-bit little-endian word and
  replicates its MSBs into the low slack, so peak white reaches full
  scale; down-conversion drops those low bits, making the
  8 → high → 8 round-trip exact. Subsampling and colour are untouched —
  this is pure per-plane storage-width scaling. New low-level primitives
  `yuv::depth_up_8_to_le16_plane` / `yuv::depth_down_le16_plane` expose
  the per-plane operation over `&[u8]`. No `ColorSpace` knob applies.

- New `convert_geometry` cargo-fuzz target under `fuzz/`. Rather than
  feeding arbitrary bytes at a parser, it *constructs* well-formed source
  frames at fuzzer-chosen small / odd / extra-stride-padded dimensions
  and drives every registered `(src, dst)` conversion, asserting none
  panics, overflows, or reads out of bounds. Wired into the daily fuzz CI
  workflow. (This target's first run surfaced the odd-dimension OOB fixed
  above.)

- BT.2020 NCL anchor-vector tests: black and the three primaries are now
  pinned against 8-bit codes hand-derived from BT.2020-2 Table 4 (NCL
  column) + Table 5 quantization, in both limited and full range (e.g.
  limited red → (74, 97, 240) with C'R = 0.7373/1.4746 = 0.5 exactly;
  limited blue C'B likewise lands exactly on code 240). A third test
  asserts encode → decode invertibility within ±2 LSB at the gamut
  extremes (black / white / primaries), covering the full-range Cr/Cb
  255-cap clip on saturated red/blue. Complements the existing gradient
  PSNR roundtrips, neutral-grey and white anchors from the original
  BT.2020 matrix landing.

- Planar GBR(A) ↔ packed deep-RGB conversions wired into the `convert()`
  dispatch table. Twelve new pairs land: `Gbrp10Le` / `Gbrp12Le` /
  `Gbrp14Le` ↔ `Rgb48Le` (six pairs) and `Gbrap10Le` / `Gbrap12Le` /
  `Gbrap14Le` ↔ `Rgba64Le` (six pairs). Until now every src/dst touching
  a GBR(A) variant returned `Error::Unsupported`, even though all six
  variants had `FormatInfo::of` arms and OxideAV-core enum discriminants
  (35–40) since 0.1.18. GBR is RGB carried as separate planes in G, B,
  R(, A) order, each sample a 16-bit little-endian word with only the low
  `bits` (10 / 12 / 14) significant — the layout used by MagicYUV,
  JPEG 2000, ProRes 4444 and lossless H.264 GBR mode. The conversion is a
  pure plane reorder (G, B, R(, A) planes ↔ packed R, G, B(, A) words)
  plus a `16 - bits` left-shift toward the 16-bit packed container (and
  the reverse right-shift on the way back). No colour matrix enters — it
  is bit-layout normalisation only, so no `ColorSpace` knob applies and
  no new colour-science coefficients are introduced. A `Gbrap*` source
  carrying fewer than four planes rejects with `Error::Invalid` (alpha
  plane missing). Two new helpers `convert::{rd16le, wr16le}` keep the
  16-bit LE word access in one place. Four new tests in
  `tests/conversions.rs` pin: the G/B/R plane→packed reorder + shift
  against hand-computed expected packed words (`Gbrp10Le → Rgb48Le`), the
  alpha plane reaching the fourth packed component (`Gbrap12Le →
  Rgba64Le`), the full bit-exact round-trip across all three bit depths
  for both the no-alpha and alpha families (the widen-then-narrow shift
  is exactly invertible because the source samples already fit in `bits`
  significant bits), and short-plane-count rejection on a `Gbrap*` source.
- `Yuv411P` (4:1:1 planar — luma at full resolution, chroma horizontally
  subsampled by 4) wired into the `convert()` dispatch table. Until now
  every src/dst with `Yuv411P` returned `Error::Unsupported`, even
  though the variant had a `FormatInfo::of` entry and an OxideAV-core
  enum discriminant since 0.1.5. Ten new pairs land:
  - `Yuv411P ↔ Yuv420P / Yuv422P / Yuv444P` (six pairs, chroma resample
    only — luma copied byte-for-byte, no RGB hop). The widening
    directions (411 → 444 / 422) broadcast each chroma sample
    horizontally to the four (or two) luma columns it covers; the
    shrinking directions (444 / 422 → 411) horizontally box-average four
    (or two) source samples per destination sample. 411 ↔ 420 combines
    a vertical pair-average with the horizontal step.
  - `Yuv411P ↔ Rgb24 / Rgba` under any `ColorSpace` (BT.601 / 709 /
    2020, limited or full range). The RGB encode/decode stages through
    a transient 4:4:4 chroma intermediate before calling the proven
    scalar 4:4:4 ↔ RGB matrix, so no new colour math enters the crate.
    The `→ Rgba` direction synthesises an opaque alpha plane (0xFF
    everywhere); the `Rgba → Yuv411P` encode ignores the source's alpha
    column.
  Width must be a multiple of 4 (4:1:1 has no representation for a 1-,
  2-, or 3-luma trailing column) — odd-by-4 widths reject with
  `Error::Invalid` instead of silently truncating. Six new
  `chroma_411_*` primitives in `yuv::` (`chroma_444_to_411`,
  `chroma_411_to_444`, `chroma_422_to_411`, `chroma_411_to_422`,
  `chroma_420_to_411`, `chroma_411_to_420`) are public so low-level
  callers can drive the chroma step directly. Ten new tests in
  `tests/conversions.rs` pin the luma byte-for-byte invariant on every
  direction, the chroma broadcast (411 → 444 / 422) and box-average
  (444 / 422 / 420 → 411) shapes against hand-computed expected values,
  the 411 ↔ 444 and 411 ↔ 422 round-trip bit-exactness (the widening
  step broadcasts identical samples so the shrink step averages them
  back to the original byte), an `Yuv411P` ↔ `Rgb24` round-trip PSNR
  floor above 30 dB on a synthetic luma ramp, opaque-alpha synthesis on
  `→ Rgba`, and odd-by-4 width rejection. Closes the long-standing gap
  the `FormatInfo::chroma_subsampling()` enum's `C411` variant was
  added for in `[Unreleased]` above.
- `FormatInfo::chroma_subsampling()` typed view returning a new
  `ChromaSubsampling` enum (`None` / `C444` / `C422` / `C420` / `C411` /
  `Other`), plus a `FormatInfo::is_chroma_subsampled()` predicate. Lets
  callers branch on the named 4:n:m scheme — e.g. picking a 4:2:0,
  4:2:2, or 4:4:4 dispatch path — without open-coding the
  `(chroma_w_sub, chroma_h_sub) == (2, 2)` etc. tuple checks at every
  call site. The mapping is derived from the same factor pair already
  carried on `FormatInfo`, so the answer agrees by construction with
  the raw fields; the enum is `#[non_exhaustive]` so future schemes
  (4:4:0, 4:1:0, …) can land without a breaking change.
- Direct planar YUV ↔ planar YUV conversions wired into the `convert()`
  dispatch — twelve new pairs that previously returned
  `Error::Unsupported`. The full Cartesian product on
  `(4:2:0, 4:2:2, 4:4:4)` for the limited-range planar family
  (`Yuv420P ↔ Yuv422P`, `Yuv420P ↔ Yuv444P`, `Yuv422P ↔ Yuv444P` in
  both directions, six pairs) plus the same six pairs on the full-range
  `YuvJ*` family. The new `ChromaResample` op copies the luma plane
  byte-for-byte and routes the chroma planes through the existing
  `yuv::chroma_*` primitives — no colour math, no RGB hop. Callers
  switching chroma subsampling without changing colour space (e.g. an
  H.264 4:2:2 source feeding a 4:2:0 encoder, or a 4:4:4 ProRes frame
  staging into a 4:2:0 codec) save an `Rgb24` intermediate's worth of
  buffer churn and avoid the round-trip rounding error two YUV↔RGB
  passes would accumulate. Eleven new tests in `tests/conversions.rs`
  pin: luma byte-for-byte copy on every direction, chroma row-duplicate
  (4:2:0 → 4:2:2) and 2×2-nearest (4:2:0 → 4:4:4) widen invariants,
  chroma pair-average / box-average shrink invariants (4:4:4 → 4:2:2,
  4:4:4 → 4:2:0, 4:2:2 → 4:2:0), the `Yuv420P ↔ Yuv422P` and
  `Yuv420P ↔ Yuv444P` round trips' bit-exactness (the widening step
  duplicates samples, so the shrink step averages identical values
  back to the original byte), the `YuvJ*` family yielding chroma bytes
  identical to the `Yuv*` family on the same input (confirms the
  resampler is colour-space-agnostic), and odd-height rejection on a
  4:2:0 source.
- `Yuva420P` (planar 4:2:0 YUV + a full-resolution alpha plane) wired
  into the `convert()` dispatch. Six new pairs land:
  - `Yuv420P → Yuva420P` (append a `0xFF`-filled luma-resolution alpha
    plane to the existing Y/U/V triple; Y/U/V copied byte-for-byte).
  - `Yuva420P → Yuv420P` (drop the trailing alpha plane).
  - `Yuva420P → Rgba` / `Yuva420P → Rgb24` (decode the YUV part through
    the existing 4:2:0 → RGB scalar/SIMD path; for `Rgba` the source's
    alpha plane is carried into the 4th destination byte bit-exact, no
    chroma-style averaging).
  - `Rgba → Yuva420P` (split the source's alpha column into the trailing
    plane bit-exact; YUV from the proven 4:2:0 encoder).
  - `Rgb24 → Yuva420P` (synthesise an opaque luma-resolution alpha
    plane).
  Odd width or height is rejected with `Error::Invalid` (4:2:0 has no
  representation for a half pixel). Seven new tests in
  `tests/conversions.rs` pin the alpha-plane bit-exactness invariant
  (`Yuva420P → Rgba`, `Rgba → Yuva420P`, and the
  `Rgba → Yuva420P → Rgba` round-trip's alpha column), check that the
  Rgb24 and Rgba destinations agree on the R/G/B columns, verify
  opaque-alpha synthesis on the `Rgb24 → Yuva420P` direction, lock in
  the `Yuv420P ↔ Yuva420P` lossless round-trip, and assert odd-size
  rejection. Without these paths, every `Yuva420P` source / destination
  through `convert()` returned `Error::Unsupported`.
- Direct `Nv12` / `Nv21` ↔ `Rgb24` / `Rgba` conversions wired into the
  `convert()` dispatch table. Previously the only NV path was via
  `Yuv420P`; the eight new pairs (`Nv12 → Rgb24`, `Nv12 → Rgba`,
  `Nv21 → Rgb24`, `Nv21 → Rgba`, `Rgb24 → Nv12`, `Rgb24 → Nv21`,
  `Rgba → Nv12`, `Rgba → Nv21`) save callers an explicit
  `Yuv420P` staging step. The fused path runs the proven planar
  4:2:0 encoder / decoder under the hood, so output bytes are
  bit-exact to the previous two-step route — the dispatch saves one
  intermediate `VideoFrame` allocation. Odd width or height is
  rejected with `Error::Invalid` (4:2:0 has no representation for a
  half pixel). Six new tests in `tests/yuv_rgb.rs` cross-check the
  direct path against the staged route, verify opaque alpha synthesis
  on the `→ Rgba` direction, and pin the Rgb24 round-trip PSNR above
  the same 30 dB floor as the planar 4:2:0 path.
- `Ya8` (grey + alpha, 2 bytes/pixel) conversion paths now wired into
  the `convert()` dispatch table — previously every entry returned
  `Error::Unsupported`. Six new pairs land: `Ya8 ↔ Gray8`,
  `Ya8 → Rgb24`, `Ya8 ↔ Rgba`, and `Rgb24 → Ya8`. The Gray-side paths
  are byte-stride helpers (drop / synthesise alpha as 255); the RGB
  paths broadcast luma to R = G = B and derive Y on the reverse leg as
  the rounded mean `(R + G + B + 1) / 3` so a `Ya8 → Rgba → Ya8`
  round-trip is bit-exact for inputs that came out of `Ya8` in the
  first place (R = G = B by construction). Helpers live in
  `crate::gray::{ya8_to_gray8, ya8_to_alpha8, gray8_to_ya8,
  ya8_to_rgb24, ya8_to_rgba, rgba_to_ya8, rgb24_to_ya8}`. Four new
  exact-roundtrip tests in `tests/conversions.rs` pin the broadcast
  + alpha invariants.
- `tests/alpha_property.rs` — 15-test PRNG-driven property sweep for the
  Porter-Duff primitives (`over_premul`, `over_straight`, `over_buffer`,
  `blit_alpha_mask`, `modulate_alpha`, `premultiply` / `unpremultiply`).
  Sweeps cover: opaque-source = replacement, transparent-source = no-op,
  premultiplied invariant (`C ≤ A`) preservation, alpha-formula accuracy
  (`out.a = src.a + dst.a × (1 - src.a)` within ±1 LSB), monotonicity in
  `src.a`, `over_buffer` element-wise parity with the per-pixel helpers,
  and an `~10k`-placement `blit_alpha_mask` leak-freedom check that pads
  the destination with sentinel bytes and asserts none are overwritten.
  Backstops the alpha module's previously hand-picked anchor cases.
- `benches/alpha.rs` — Criterion bench covering the Porter-Duff hot
  path: per-pixel `over_premul` / `over_straight` (1 Mpx inner loop),
  `over_buffer` at 1920×1080 (premul + straight), `blit_alpha_mask`
  with 16×16 (ASCII-glyph) and 64×64 (CJK-glyph) mask sizes, and bulk
  `premultiply` / `unpremultiply` / `modulate_alpha` (1 Mpx). Gated
  behind the existing `bench` feature; documents the scalar baseline a
  future SIMD pass would land against.
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
- **CI miri job no longer hangs at its 6 h timeout.** The job (added
  2026-05-03) never went green: it hung first in `tests/palette.rs`
  (median-cut over a 256×256 random RGBA frame ≈ 65 k candidate colours,
  plus octree / Floyd–Steinberg roundtrips on 64–128 px frames) and would
  next have hung in `tests/property.rs` (loops of 100–500 random cases ×
  conversions, 50 000-pixel per-matrix sweeps, a 200 000-iteration
  premultiply sweep — the first property test alone needed ~7 min under
  the interpreter locally). Both files now shrink their corpus under
  `cfg(miri)` following the existing `tests/yuv_simd_parity.rs`
  convention — every test still runs and asserts under miri, only frame
  dimensions (palette: 256→24, 128→16, 64→8/16) and random case counts
  (property: 400/300/200/150→6-8, 50 k→400/matrix, 200 k→2 k, 500→10)
  shrink; statistical coverage stays the native run's job while miri
  exercises each code path's memory model. Native dimensions and counts
  are unchanged.

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
