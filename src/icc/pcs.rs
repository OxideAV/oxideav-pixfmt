//! Profile connection space: the D50 PCS white, PCSXYZ ↔ PCSLAB
//! (ICC.1:2022 §6.4 / Annex A) and the normalised `[0, 1]` encodings a
//! LUT sees on its PCS side (§6.3.4.2 Tables 11–13, §10.10 Tables
//! 42–43, Annex F.3 NOTE).

use super::lut::LutKind;

/// PCS white point, ICC.1:2022 §6.3.4.3 Table 14 (nCIEXYZ of D50).
pub const PCS_WHITE: [f64; 3] = [0.9642, 1.0, 0.8249];

/// PCSXYZ "1.0" as a fraction of the u1Fixed15 full scale (Table 11:
/// 1.0 = 8000h, the normalised table domain tops out at FFFFh).
const XYZ_NORM: f64 = 32768.0 / 65535.0;

/// Annex A f(t), with the §6.4 rule that the straight-line segment
/// below (6/29)³ is extended linearly below zero.
#[inline]
fn lab_f(t: f64) -> f64 {
    const E: f64 = 216.0 / 24389.0; // (6/29)^3
    if t > E {
        t.cbrt()
    } else {
        841.0 / 108.0 * t + 4.0 / 29.0
    }
}

#[inline]
fn lab_f_inv(f: f64) -> f64 {
    const D: f64 = 6.0 / 29.0;
    if f > D {
        f * f * f
    } else {
        (f - 4.0 / 29.0) * 108.0 / 841.0
    }
}

/// CIE XYZ → CIELAB relative to the white `wp` (ICC.1:2022 Annex A,
/// equations A.4–A.12).
pub fn xyz_to_lab_white(xyz: [f64; 3], wp: [f64; 3]) -> [f64; 3] {
    let fx = lab_f(xyz[0] / wp[0]);
    let fy = lab_f(xyz[1] / wp[1]);
    let fz = lab_f(xyz[2] / wp[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIELAB → CIE XYZ relative to the white `wp` (inverse of Annex A).
pub fn lab_to_xyz_white(lab: [f64; 3], wp: [f64; 3]) -> [f64; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    [
        lab_f_inv(fx) * wp[0],
        lab_f_inv(fy) * wp[1],
        lab_f_inv(fz) * wp[2],
    ]
}

/// PCSXYZ → PCSLAB (D50 PCS white).
pub fn xyz_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    xyz_to_lab_white(xyz, PCS_WHITE)
}

/// PCSLAB → PCSXYZ (D50 PCS white).
pub fn lab_to_xyz(lab: [f64; 3]) -> [f64; 3] {
    lab_to_xyz_white(lab, PCS_WHITE)
}

/// Which connection space a profile's PCS side uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PcsKind {
    Xyz,
    Lab,
}

/// Decode a LUT's normalised PCS-side output into PCS values.
pub(crate) fn decode_pcs(kind: PcsKind, lut: LutKind, n: [f64; 3]) -> [f64; 3] {
    match kind {
        PcsKind::Xyz => [n[0] / XYZ_NORM, n[1] / XYZ_NORM, n[2] / XYZ_NORM],
        PcsKind::Lab => {
            if lut == LutKind::Lut16 {
                // Legacy 16-bit PCSLAB (Tables 42/43): FF00h = 100 / 127.
                let s = 65535.0 / 65280.0;
                [
                    n[0] * s * 100.0,
                    n[1] * s * 255.0 - 128.0,
                    n[2] * s * 255.0 - 128.0,
                ]
            } else {
                // Tables 12/13: L 0..100, a/b −128..127 over the full scale.
                [n[0] * 100.0, n[1] * 255.0 - 128.0, n[2] * 255.0 - 128.0]
            }
        }
    }
}

/// Encode PCS values into a LUT's normalised PCS-side input.
pub(crate) fn encode_pcs(kind: PcsKind, lut: LutKind, v: [f64; 3]) -> [f64; 3] {
    match kind {
        PcsKind::Xyz => [v[0] * XYZ_NORM, v[1] * XYZ_NORM, v[2] * XYZ_NORM],
        PcsKind::Lab => {
            if lut == LutKind::Lut16 {
                let s = 65280.0 / 65535.0;
                [
                    v[0] / 100.0 * s,
                    (v[1] + 128.0) / 255.0 * s,
                    (v[2] + 128.0) / 255.0 * s,
                ]
            } else {
                [v[0] / 100.0, (v[1] + 128.0) / 255.0, (v[2] + 128.0) / 255.0]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_and_black_points() {
        let lab = xyz_to_lab(PCS_WHITE);
        assert!((lab[0] - 100.0).abs() < 1e-12 && lab[1].abs() < 1e-12 && lab[2].abs() < 1e-12);
        assert_eq!(xyz_to_lab([0.0; 3]), [0.0, 0.0, 0.0]);
        // Table 16 perceptual black: L* 3.1373 ↔ Y 0.003479 (the table's
        // own NOTE says the two differ by encoding precision).
        let l = xyz_to_lab([0.003357, 0.003479, 0.002869])[0];
        assert!((l - 3.1373).abs() < 0.01, "{l}");
    }

    #[test]
    fn lab_round_trip_including_negative_xyz() {
        for xyz in [
            [0.5, 0.4, 0.3],
            [0.001, 0.002, 0.0005],
            [-0.01, 0.003, 0.02],
            [0.96, 1.1, 0.7],
        ] {
            let back = lab_to_xyz(xyz_to_lab(xyz));
            for c in 0..3 {
                assert!((back[c] - xyz[c]).abs() < 1e-12, "{xyz:?} → {back:?}");
            }
        }
    }

    #[test]
    fn pcs_encodings() {
        // Table 14: 16-bit PCSXYZ white = 7B6Bh 8000h 6996h.
        let n = encode_pcs(PcsKind::Xyz, LutKind::AToB, PCS_WHITE);
        assert_eq!((n[0] * 65535.0).round() as u32, 0x7B6B);
        assert_eq!((n[1] * 65535.0).round() as u32, 0x8000);
        assert_eq!((n[2] * 65535.0).round() as u32, 0x6996);
        // Legacy Lab: L 100 = FF00h, a 0 = 8000h.
        let n = encode_pcs(PcsKind::Lab, LutKind::Lut16, [100.0, 0.0, 0.0]);
        assert_eq!((n[0] * 65535.0).round() as u32, 0xFF00);
        assert_eq!((n[1] * 65535.0).round() as u32, 0x8000);
        // v4 Lab: a 0 = 8080h.
        let n = encode_pcs(PcsKind::Lab, LutKind::AToB, [100.0, 0.0, 0.0]);
        assert_eq!((n[1] * 65535.0).round() as u32, 0x8080);
        for kind in [LutKind::Lut8, LutKind::Lut16, LutKind::AToB] {
            let v = [43.0, -20.5, 71.25];
            let back = decode_pcs(PcsKind::Lab, kind, encode_pcs(PcsKind::Lab, kind, v));
            for c in 0..3 {
                assert!((back[c] - v[c]).abs() < 1e-9);
            }
        }
    }
}
