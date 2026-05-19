//! Validation tests for the transfer-function module against the
//! anchor points documented in BT.2020-2, BT.1886 and BT.2100-3.
//!
//! The tests pin (a) round-trip OETF/EOTF identity across the unit
//! interval; (b) spec-stated anchor points (`E'=1 → 10 000 cd/m²` for
//! PQ, `E=1/12 → E'=0.5` for HLG, `V=1 → L=1` for BT.1886); and
//! (c) monotonicity across each curve. Numerical tolerances reflect
//! single-precision float round-off.

use oxideav_pixfmt::transfer::{
    bt1886_eotf, bt1886_eotf_with_levels, bt1886_inverse_eotf, bt2020_12_inverse_oetf,
    bt2020_12_oetf, bt709_inverse_oetf, bt709_oetf, hlg_inverse_oetf, hlg_oetf, pq_eotf,
    pq_eotf_normalised, pq_inverse_eotf, pq_inverse_eotf_normalised, PQ_PEAK_CDM2,
};

fn assert_close(actual: f32, expected: f32, eps: f32, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= eps,
        "{label}: got {actual}, expected {expected}, diff {diff} > eps {eps}"
    );
}

#[test]
fn bt709_oetf_roundtrips_unit_interval() {
    for i in 0..=100 {
        let e = i as f32 / 100.0;
        let e_prime = bt709_oetf(e);
        let e_back = bt709_inverse_oetf(e_prime);
        assert_close(e_back, e, 1e-5, "bt709 roundtrip");
    }
}

#[test]
fn bt709_oetf_anchor_points() {
    // E = 0 maps to E' = 0 (lower segment 4.5 * E).
    assert_close(bt709_oetf(0.0), 0.0, 1e-7, "bt709 E=0");
    // Just below the segment boundary the lower branch is in force:
    // 4.5 · E.
    let just_below = 0.017_f32;
    assert_close(
        bt709_oetf(just_below),
        4.5 * just_below,
        1e-6,
        "bt709 lower branch",
    );
    // E = 1 maps to E' = 1 because α·1^0.45 − (α−1) = α − α + 1 = 1.
    assert_close(bt709_oetf(1.0), 1.0, 1e-5, "bt709 E=1");
}

#[test]
fn bt2020_12_oetf_roundtrips_unit_interval() {
    for i in 0..=100 {
        let e = i as f32 / 100.0;
        let e_prime = bt2020_12_oetf(e);
        let e_back = bt2020_12_inverse_oetf(e_prime);
        assert_close(e_back, e, 1e-5, "bt2020-12 roundtrip");
    }
}

#[test]
fn bt2020_12_oetf_endpoints() {
    // E = 1 should give E' ≈ 1.0 (α·1^0.45 - (α-1) = 1).
    assert_close(bt2020_12_oetf(1.0), 1.0, 1e-5, "bt2020-12 E=1");
    assert_close(bt2020_12_oetf(0.0), 0.0, 1e-7, "bt2020-12 E=0");
}

#[test]
fn bt1886_eotf_endpoints() {
    // L(0) = 0, L(1) = 1 for the default infinite-contrast case.
    assert_close(bt1886_eotf(0.0), 0.0, 1e-7, "bt1886 V=0");
    assert_close(bt1886_eotf(1.0), 1.0, 1e-7, "bt1886 V=1");
    // Mid-gray V=0.5 → L = 0.5^2.4 ≈ 0.1895.
    assert_close(bt1886_eotf(0.5), 0.5f32.powf(2.4), 1e-7, "bt1886 V=0.5");
}

#[test]
fn bt1886_eotf_roundtrips() {
    for i in 0..=100 {
        let v = i as f32 / 100.0;
        let l = bt1886_eotf(v);
        let v_back = bt1886_inverse_eotf(l);
        assert_close(v_back, v, 1e-5, "bt1886 roundtrip");
    }
}

#[test]
fn bt1886_eotf_with_levels_honors_endpoints() {
    // With L_W = 100, L_B = 0.1, V=0 should give L_B and V=1 should
    // give L_W. The general formula has to honour the spec's
    // boundary conditions.
    let l_w = 100.0;
    let l_b = 0.1;
    assert_close(
        bt1886_eotf_with_levels(0.0, l_w, l_b),
        l_b,
        1e-4,
        "bt1886 levels V=0",
    );
    assert_close(
        bt1886_eotf_with_levels(1.0, l_w, l_b),
        l_w,
        2e-3,
        "bt1886 levels V=1",
    );
}

#[test]
fn pq_eotf_endpoints() {
    // E' = 0 → 0 cd/m².
    assert_close(pq_eotf(0.0), 0.0, 1e-3, "pq E'=0");
    // E' = 1 → 10 000 cd/m² (peak). Tolerance reflects the cascaded
    // f32 power operations.
    assert_close(pq_eotf(1.0), PQ_PEAK_CDM2, 1.0, "pq E'=1");
    // Normalised endpoint identity.
    assert_close(pq_eotf_normalised(1.0), 1.0, 1e-4, "pq norm E'=1");
}

#[test]
fn pq_roundtrips_unit_interval() {
    // 201-point grid — enough to catch a sign or branch flip without
    // making miri-strict-provenance unhappy about per-iteration cost.
    for i in 0..=200 {
        let e_prime = i as f32 / 200.0;
        let y = pq_eotf_normalised(e_prime);
        let e_back = pq_inverse_eotf_normalised(y);
        assert_close(e_back, e_prime, 5e-5, "pq roundtrip");
    }
}

#[test]
fn pq_absolute_roundtrip_at_reference_levels() {
    // The 203 cd/m² HDR reference white (BT.2100 Note 10a) is a
    // recurring HDR anchor — confirm a clean roundtrip there.
    let l = 203.0f32;
    let e_prime = pq_inverse_eotf(l);
    let l_back = pq_eotf(e_prime);
    assert_close(l_back, l, 0.05, "pq HDR-white 203 cd/m²");
}

#[test]
fn hlg_oetf_anchor_points() {
    // E = 0 → E' = 0.
    assert_close(hlg_oetf(0.0), 0.0, 1e-7, "hlg E=0");
    // E = 1/12 is the curve crossover; the lower branch yields
    // sqrt(3 / 12) = sqrt(0.25) = 0.5.
    assert_close(hlg_oetf(1.0 / 12.0), 0.5, 1e-6, "hlg E=1/12");
    // E = 1 should land on E' = 1 by construction (the constants a,
    // b, c are chosen so the upper branch passes through (1, 1)).
    assert_close(hlg_oetf(1.0), 1.0, 1e-6, "hlg E=1");
}

#[test]
fn hlg_roundtrips_unit_interval() {
    for i in 0..=200 {
        let e = i as f32 / 200.0;
        let e_prime = hlg_oetf(e);
        let e_back = hlg_inverse_oetf(e_prime);
        assert_close(e_back, e, 1e-5, "hlg roundtrip");
    }
}

#[test]
fn pq_is_monotonic() {
    let mut prev = -1.0f32;
    for i in 0..=200 {
        let v = pq_eotf_normalised(i as f32 / 200.0);
        assert!(
            v >= prev - 1e-6,
            "pq monotonicity broken at {i}: {v} < {prev}"
        );
        prev = v;
    }
}

#[test]
fn hlg_is_monotonic() {
    let mut prev = -1.0f32;
    for i in 0..=200 {
        let v = hlg_oetf(i as f32 / 200.0);
        assert!(
            v >= prev - 1e-6,
            "hlg monotonicity broken at {i}: {v} < {prev}"
        );
        prev = v;
    }
}
