//! Opto-electronic / electro-optical transfer functions.
//!
//! Transfers map between non-linear video signal values (the `[0, 1]`
//! domain encoded in a pixel buffer) and linear-light values (relative
//! scene-referred for an OETF, or display-referred luminance for an
//! EOTF). They are an orthogonal axis to the YUV↔RGB matrix selected by
//! [`crate::convert::ColorSpace`]; a complete HDR pipeline mixes a
//! BT.2020 / BT.2100 matrix with a PQ or HLG transfer.
//!
//! Equations and exact constants are reproduced from:
//!
//! - `docs/video/signal-metadata/R-REC-BT.2020-2-201510-I.pdf` Table 4
//!   (BT.709 / BT.2020 OETF; the same `α = 1.099, β = 0.018` curve has
//!   been the SDR transmission characteristic since BT.709).
//! - `docs/video/signal-metadata/R-REC-BT.1886-0-201103-I.pdf` Annex 1
//!   (BT.1886 reference EOTF, `L = a · max(V+b, 0)^γ`, `γ = 2.40`).
//! - `docs/video/signal-metadata/R-REC-BT.2100-3-202502-I.pdf` Tables 4–5
//!   (PQ EOTF / OETF, HLG OETF / EOTF / OOTF).
//!
//! All functions operate on `f32`. Their domains are explicitly
//! documented; callers are responsible for any preceding quantisation
//! step (10/12-bit narrow- or full-range integer formats per BT.2020
//! Table 5 / BT.2100 Table 9).
//!
//! The hot paths intentionally stay on f32 instead of fixed-point —
//! transfer-function callers tend to be downstream of float-RGB tone
//! mapping and are not in the per-pixel SDR YUV↔RGB hot loop.

/// Identifier for a transfer function.
///
/// `BT709Oetf` and `BT2020_10Oetf` share the same `α = 1.099, β = 0.018`
/// curve from BT.2020-2 Table 4 (the 10-bit values); they are separate
/// variants so that callers can document which signalling chain they
/// belong to. The 12-bit variant uses the more-precise `α = 1.0993,
/// β = 0.0181` constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferFunction {
    /// BT.709-style OETF (10-bit constants). Source: BT.2020-2 Table 4.
    Bt709Oetf,
    /// BT.2020 12-bit OETF (uses the more-precise α/β). Source: same.
    Bt2020_12Oetf,
    /// BT.1886 reference EOTF, γ = 2.40. Source: BT.1886 Annex 1.
    Bt1886Eotf,
    /// SMPTE ST 2084 / BT.2100 PQ EOTF. Source: BT.2100-3 Table 4.
    PqEotf,
    /// PQ inverse EOTF (display-light → non-linear signal). Source:
    /// BT.2100-3 Table 4, "Reference PQ OETF" row.
    PqInverseEotf,
    /// BT.2100 HLG OETF (scene-light → non-linear signal). Source:
    /// BT.2100-3 Table 5.
    HlgOetf,
    /// BT.2100 HLG inverse OETF (non-linear signal → scene-light).
    /// Source: BT.2100-3 Table 5 ("OETF⁻¹" used inside the EOTF).
    HlgInverseOetf,
}

// ------------------------------------------------------------------
// BT.709 / BT.2020 SDR OETF — BT.2020-2 Table 4 row "Non-linear
// transfer function": E' = 4.5·E for 0 ≤ E < β, otherwise
// E' = α·E^0.45 − (α − 1).
//
// The 10-bit constants are α = 1.099, β = 0.018; the 12-bit precision
// values are α = 1.0993, β = 0.0181 (same table, "for 12-bit systems"
// column).

const BT709_ALPHA: f32 = 1.099;
const BT709_BETA: f32 = 0.018;
const BT2020_12_ALPHA: f32 = 1.0993;
const BT2020_12_BETA: f32 = 0.0181;

/// BT.709 / BT.2020 OETF (10-bit precision constants).
///
/// Maps relative scene-light `E ∈ [0, 1]` to the non-linear signal
/// `E' ∈ [0, 1]`. Values below 0 are clamped to 0; values above 1 pass
/// through the non-linear branch (BT.2020-2 doesn't define behaviour
/// outside `[0, 1]` — callers wanting to clip should do so afterwards).
pub fn bt709_oetf(e: f32) -> f32 {
    bt2020_oetf_impl(e, BT709_ALPHA, BT709_BETA)
}

/// BT.2020 12-bit OETF (more-precise α/β constants).
pub fn bt2020_12_oetf(e: f32) -> f32 {
    bt2020_oetf_impl(e, BT2020_12_ALPHA, BT2020_12_BETA)
}

fn bt2020_oetf_impl(e: f32, alpha: f32, beta: f32) -> f32 {
    if e < 0.0 {
        // Per spec the curve is defined on [0, 1]; clamp negative
        // out-of-range inputs to 0 rather than letting them produce
        // imaginary numbers through the power function.
        0.0
    } else if e < beta {
        4.5 * e
    } else {
        alpha * e.powf(0.45) - (alpha - 1.0)
    }
}

/// Inverse of [`bt709_oetf`] (the EOTF the camera curve implies).
///
/// The crossover point in the inverse direction lies at
/// `E' = 4.5 · β`. We use the 10-bit constants by default.
pub fn bt709_inverse_oetf(e_prime: f32) -> f32 {
    bt2020_inverse_oetf_impl(e_prime, BT709_ALPHA, BT709_BETA)
}

pub fn bt2020_12_inverse_oetf(e_prime: f32) -> f32 {
    bt2020_inverse_oetf_impl(e_prime, BT2020_12_ALPHA, BT2020_12_BETA)
}

fn bt2020_inverse_oetf_impl(e_prime: f32, alpha: f32, beta: f32) -> f32 {
    if e_prime < 0.0 {
        0.0
    } else if e_prime < 4.5 * beta {
        e_prime / 4.5
    } else {
        ((e_prime + (alpha - 1.0)) / alpha).powf(1.0 / 0.45)
    }
}

// ------------------------------------------------------------------
// BT.1886 reference EOTF — BT.1886 Annex 1.
//
//   L = a · max(V + b, 0)^γ
//
// with γ = 2.40, and a/b determined to satisfy L(0) = L_B,
// L(1) = L_W. The infinite-contrast case L_B = 0 collapses to the
// pure power law L = V^γ (a = 1, b = 0), which is the form most
// frequently used in practice.

/// BT.1886 reference EOTF (default infinite-contrast case `L_B = 0`,
/// `L_W = 1`).
///
/// `v ∈ [0, 1]` (normalised video signal) → relative luminance `L`
/// (also `[0, 1]`).
pub fn bt1886_eotf(v: f32) -> f32 {
    if v <= 0.0 {
        0.0
    } else {
        v.powf(2.4)
    }
}

/// Inverse of [`bt1886_eotf`] (relative luminance → normalised signal).
pub fn bt1886_inverse_eotf(l: f32) -> f32 {
    if l <= 0.0 {
        0.0
    } else {
        l.powf(1.0 / 2.4)
    }
}

/// Generalised BT.1886 EOTF that honours non-zero black / non-unity
/// peak white. `l_w` and `l_b` are both expressed in the same arbitrary
/// luminance unit; the returned `L` is in the same unit.
pub fn bt1886_eotf_with_levels(v: f32, l_w: f32, l_b: f32) -> f32 {
    // a = (L_W^(1/γ) − L_B^(1/γ))^γ, b = L_B^(1/γ) / (L_W^(1/γ) − L_B^(1/γ))
    let inv_g = 1.0 / 2.4;
    let lw = l_w.max(0.0).powf(inv_g);
    let lb = l_b.max(0.0).powf(inv_g);
    let diff = lw - lb;
    if diff <= 0.0 {
        return 0.0;
    }
    let a = diff.powf(2.4);
    let b = lb / diff;
    let inner = (v + b).max(0.0);
    a * inner.powf(2.4)
}

// ------------------------------------------------------------------
// PQ — BT.2100-3 Table 4.
//
//   EOTF  (signal → display light):
//     F_D = 10000 · Y,
//     Y   = ( max(E'^(1/m2) − c1, 0) / (c2 − c3 · E'^(1/m2)) )^(1/m1)
//
//   inverse EOTF (display light → signal):
//     E' = ((c1 + c2 · Y^m1) / (1 + c3 · Y^m1))^m2
//
// where m1 = 2610/16384, m2 = 128 · 2523/4096, c1 = 3424/4096,
// c2 = 32 · 2413/4096, c3 = 32 · 2392/4096 and c1 = c3 − c2 + 1.
//
// E' is the non-linear PQ signal in [0, 1]; F_D is luminance in cd/m²
// with peak 10000 cd/m². Y is normalised in [0, 1].

const PQ_M1: f32 = 2610.0 / 16384.0;
const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;
const PQ_PEAK: f32 = 10000.0;

/// PQ EOTF normalised to `[0, 1]`. Maps a PQ-encoded signal `e' ∈ [0, 1]`
/// to relative luminance in `[0, 1]` (where 1.0 corresponds to the
/// 10 000 cd/m² peak). Multiply by [`PQ_PEAK`] to recover the absolute
/// cd/m² luminance.
pub fn pq_eotf_normalised(e_prime: f32) -> f32 {
    let e = e_prime.clamp(0.0, 1.0);
    let p = e.powf(1.0 / PQ_M2);
    let num = (p - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * p;
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1)
}

/// PQ EOTF returning absolute display luminance in cd/m² (peak 10 000).
pub fn pq_eotf(e_prime: f32) -> f32 {
    PQ_PEAK * pq_eotf_normalised(e_prime)
}

/// PQ inverse EOTF on the normalised range — luminance `y ∈ [0, 1]`
/// (where 1.0 is the 10 000 cd/m² peak) → non-linear signal `e' ∈ [0, 1]`.
pub fn pq_inverse_eotf_normalised(y: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    let ym = y.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * ym;
    let den = 1.0 + PQ_C3 * ym;
    (num / den).powf(PQ_M2)
}

/// PQ inverse EOTF accepting absolute cd/m² luminance.
pub fn pq_inverse_eotf(luminance_cdm2: f32) -> f32 {
    pq_inverse_eotf_normalised(luminance_cdm2 / PQ_PEAK)
}

// ------------------------------------------------------------------
// HLG — BT.2100-3 Table 5.
//
//   OETF  (scene light → signal):
//     E' = sqrt(3 E)                                  0 ≤ E ≤ 1/12
//        = a · ln(12 E − b) + c                       1/12 < E ≤ 1
//
//   with a = 0.17883277, b = 1 − 4·a = 0.28466892,
//        c = 0.5 − a · ln(4·a) = 0.55991073 (Note 5c).
//
//   OETF⁻¹ (signal → scene light):
//     x² / 3                                          0 ≤ x ≤ 1/2
//     (exp((x − c)/a) + b) / 12                       1/2 < x ≤ 1

// Spec constants from BT.2100-3 Table 5 / Note 5c. The b and c values
// are derived from `a` as `b = 1 − 4·a` and `c = 0.5 − a·ln(4·a)`;
// the literal forms are quoted in the standard for convenience.
// Truncated to f32 precision (≈7 significant figures).
const HLG_A: f32 = 0.178_832_77;
const HLG_B: f32 = 0.284_668_92;
const HLG_C: f32 = 0.559_910_7;

/// HLG OETF (scene light → non-linear signal). Both `e` and the result
/// live in `[0, 1]`; values below 0 are clamped, values above 1 are
/// extrapolated using the upper-branch logarithm.
pub fn hlg_oetf(e: f32) -> f32 {
    if e <= 0.0 {
        0.0
    } else if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        HLG_A * (12.0 * e - HLG_B).ln() + HLG_C
    }
}

/// HLG OETF⁻¹ (non-linear signal → relative scene light). This is the
/// inner kernel of the HLG EOTF; the full display-light path further
/// applies the OOTF (see [`hlg_apply_ootf`]).
pub fn hlg_inverse_oetf(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else if x <= 0.5 {
        (x * x) / 3.0
    } else {
        (((x - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// HLG luminance coefficients per BT.2100-3 Table 5 ("HLG Reference
/// OOTF" row): Y_S = 0.2627 R_S + 0.6780 G_S + 0.0593 B_S. These are
/// numerically identical to the BT.2020 NCL matrix coefficients in
/// [`crate::yuv::YuvMatrix::BT2020`].
pub const HLG_Y_R: f32 = 0.2627;
pub const HLG_Y_G: f32 = 0.6780;
pub const HLG_Y_B: f32 = 0.0593;

/// HLG OOTF — applies the system gamma to a single linear scene-light
/// component (`r_s`, `g_s` or `b_s`) given the normalised luminance
/// `y_s` of the pixel.
///
/// `alpha` is the user gain `L_W` in cd/m² (set to 1.0 for normalised
/// output); `gamma` is the system gamma (1.2 at the nominal 1000 cd/m²
/// peak, scaled per Note 5f for other peaks).
pub fn hlg_apply_ootf(component: f32, y_s: f32, alpha: f32, gamma: f32) -> f32 {
    if y_s <= 0.0 {
        0.0
    } else {
        alpha * y_s.powf(gamma - 1.0) * component
    }
}

/// HLG inverse OOTF — recover scene-light from display-light per
/// Note 5i.
pub fn hlg_inverse_ootf(display: f32, y_display: f32, alpha: f32, gamma: f32) -> f32 {
    if y_display <= 0.0 || alpha <= 0.0 {
        0.0
    } else {
        (y_display / alpha).powf((1.0 - gamma) / gamma) * (display / alpha)
    }
}

/// System gamma γ for a display with peak luminance `l_w` (cd/m²),
/// inside the "usual production monitoring range" of 400..2000 cd/m²
/// — γ = 1.2 + 0.42 · log10(L_W / 1000) per BT.2100-3 Note 5f.
pub fn hlg_system_gamma(l_w: f32) -> f32 {
    1.2 + 0.42 * (l_w / 1000.0).log10()
}

// ------------------------------------------------------------------
// PQ peak constant exposed for callers.

/// PQ peak luminance in cd/m². Per BT.2100-3 Table 4, the PQ EOTF maps
/// a non-linear signal of 1.0 to a display luminance of 10 000 cd/m².
pub const PQ_PEAK_CDM2: f32 = PQ_PEAK;
