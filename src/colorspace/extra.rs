//! Colour models defined by their original publications rather than by
//! a staged specification: CIE 1976 L\*u\*v\*, Jzazbz, Kodak PhotoYCC,
//! the Cineon printing-density log encoding and the cone-response
//! matrices behind LMS. Each function's documentation states the
//! defining formula, its constants and the publication they come from.

use crate::mat3::{Mat3, invert3, mul3v};

// --------------------------------------------------------------------
// CIE 1976 L*u*v* (CIE 15 "Colorimetry", §8.2.2).

#[inline]
fn lightness_f(t: f64) -> f64 {
    const E: f64 = 216.0 / 24389.0;
    if t > E {
        t.cbrt()
    } else {
        841.0 / 108.0 * t + 4.0 / 29.0
    }
}

#[inline]
fn lightness_f_inv(f: f64) -> f64 {
    if f > 6.0 / 29.0 {
        f * f * f
    } else {
        (f - 4.0 / 29.0) * 108.0 / 841.0
    }
}

/// The CIE 1976 UCS chromaticity `(u′, v′)` of an XYZ triple:
/// `u′ = 4X / (X + 15Y + 3Z)`, `v′ = 9Y / (X + 15Y + 3Z)`. `None` when
/// the denominator is zero (black has no chromaticity).
fn ucs_chromaticity(xyz: [f64; 3]) -> Option<[f64; 2]> {
    let d = xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2];
    if d == 0.0 {
        None
    } else {
        Some([4.0 * xyz[0] / d, 9.0 * xyz[1] / d])
    }
}

/// XYZ → CIE 1976 L\*u\*v\* relative to the reference white `white`
/// (CIE 15 §8.2.2): `L* = 116·f(Y / Yn) − 16` with the same `f` as
/// CIELAB, `u* = 13·L*·(u′ − u′n)`, `v* = 13·L*·(v′ − v′n)`, where
/// `u′ = 4X / (X + 15Y + 3Z)` and `v′ = 9Y / (X + 15Y + 3Z)`. Black
/// maps to `(0, 0, 0)`.
pub fn xyz_to_luv(xyz: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let l = 116.0 * lightness_f(xyz[1] / white[1]) - 16.0;
    let (Some(c), Some(n)) = (ucs_chromaticity(xyz), ucs_chromaticity(white)) else {
        return [l, 0.0, 0.0];
    };
    [l, 13.0 * l * (c[0] - n[0]), 13.0 * l * (c[1] - n[1])]
}

/// CIE 1976 L\*u\*v\* → XYZ relative to `white`: `u′ = u* / (13·L*) +
/// u′n`, `v′ = v* / (13·L*) + v′n`, `Y` from `L*`, then `X = Y·9u′ /
/// (4v′)` and `Z = Y·(12 − 3u′ − 20v′) / (4v′)`. `L* = 0` (or a
/// degenerate `v′ = 0`) maps to black.
pub fn luv_to_xyz(luv: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let l = luv[0];
    let Some(n) = ucs_chromaticity(white) else {
        return [0.0; 3];
    };
    if l == 0.0 {
        return [0.0; 3];
    }
    let u = luv[1] / (13.0 * l) + n[0];
    let v = luv[2] / (13.0 * l) + n[1];
    if v == 0.0 {
        return [0.0; 3];
    }
    let y = white[1] * lightness_f_inv((l + 16.0) / 116.0);
    [
        y * 9.0 * u / (4.0 * v),
        y,
        y * (12.0 - 3.0 * u - 20.0 * v) / (4.0 * v),
    ]
}

// --------------------------------------------------------------------
// Jzazbz (Safdar, Cui, Kim and Luo, "Perceptually uniform color space
// for image signals including high dynamic range and wide gamut",
// Optics Express 25(13), 2017). Input is absolute CIE XYZ (D65) in
// cd/m².

const JZ_B: f64 = 1.15;
const JZ_G: f64 = 0.66;
const JZ_D: f64 = -0.56;
const JZ_D0: f64 = 1.629_549_953_282_156_6e-11;
/// Perceptual-quantiser constants of the paper: `c1 = 3424 / 2^12`,
/// `c2 = 2413 / 2^7`, `c3 = 2392 / 2^7`, `n = 2610 / 2^14`,
/// `p = 1.7 · 2523 / 2^5`.
const JZ_C1: f64 = 3424.0 / 4096.0;
const JZ_C2: f64 = 2413.0 / 128.0;
const JZ_C3: f64 = 2392.0 / 128.0;
const JZ_N: f64 = 2610.0 / 16384.0;
const JZ_P: f64 = 1.7 * 2523.0 / 32.0;

const JZ_XYZ_TO_LMS: Mat3 = [
    [0.41478972, 0.579999, 0.0146480],
    [-0.2015100, 1.120649, 0.0531008],
    [-0.0166008, 0.264800, 0.6684799],
];
const JZ_LMS_TO_IAB: Mat3 = [
    [0.5, 0.5, 0.0],
    [3.524000, -4.066708, 0.542708],
    [0.199076, 1.096799, -1.295875],
];

/// `pq(0) = c1^p`, the value the non-linearity takes at zero signal.
fn jz_pq_zero() -> f64 {
    static ZERO: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *ZERO.get_or_init(|| JZ_C1.powf(JZ_P))
}

/// The paper's perceptual-quantiser non-linearity on a cone signal in
/// cd/m², mirrored through zero for negative (out-of-gamut) signals.
fn jz_pq(x: f64) -> f64 {
    let t = (x.abs() / 10000.0).powf(JZ_N);
    let v = ((JZ_C1 + JZ_C2 * t) / (1.0 + JZ_C3 * t)).powf(JZ_P);
    // The curve does not pass through zero (`pq(0) = c1^p`), so the
    // mirror is taken about that point to stay continuous.
    if x < 0.0 { 2.0 * jz_pq_zero() - v } else { v }
}

fn jz_pq_inv(v: f64) -> f64 {
    let zero = jz_pq_zero();
    let (neg, v) = if v < zero {
        (true, 2.0 * zero - v)
    } else {
        (false, v)
    };
    let t = v.powf(1.0 / JZ_P);
    let x = 10000.0
        * ((JZ_C1 - t) / (JZ_C3 * t - JZ_C2))
            .max(0.0)
            .powf(1.0 / JZ_N);
    if neg { -x } else { x }
}

/// Absolute CIE XYZ (D65, cd/m²) → Jzazbz (Safdar et al. 2017):
/// `X′ = b·X − (b − 1)·Z`, `Y′ = g·Y − (g − 1)·X` with `b = 1.15`,
/// `g = 0.66`; `LMS = M₁·(X′, Y′, Z)`; each cone signal through the
/// perceptual quantiser `((c1 + c2·(x / 10000)^n) / (1 + c3·(x /
/// 10000)^n))^p`; `(Iz, az, bz) = M₂·(L′, M′, S′)`; and `Jz = (1 +
/// d)·Iz / (1 + d·Iz) − d0` with `d = −0.56`, `d0 = 1.6295499532821566
/// × 10⁻¹¹`.
pub fn xyz_abs_to_jzazbz(xyz: [f64; 3]) -> [f64; 3] {
    let xp = JZ_B * xyz[0] - (JZ_B - 1.0) * xyz[2];
    let yp = JZ_G * xyz[1] - (JZ_G - 1.0) * xyz[0];
    let lms = mul3v(&JZ_XYZ_TO_LMS, [xp, yp, xyz[2]]);
    let iab = mul3v(&JZ_LMS_TO_IAB, lms.map(jz_pq));
    let jz = (1.0 + JZ_D) * iab[0] / (1.0 + JZ_D * iab[0]) - JZ_D0;
    [jz, iab[1], iab[2]]
}

/// Jzazbz → absolute CIE XYZ (D65, cd/m²); the exact inverse of
/// [`xyz_abs_to_jzazbz`].
pub fn jzazbz_to_xyz_abs(jab: [f64; 3]) -> [f64; 3] {
    let j = jab[0] + JZ_D0;
    let iz = j / (1.0 + JZ_D - JZ_D * j);
    static INVERSES: std::sync::OnceLock<(Mat3, Mat3)> = std::sync::OnceLock::new();
    let (inv_iab, inv_lms) = INVERSES.get_or_init(|| {
        (
            invert3(&JZ_LMS_TO_IAB).expect("Jzazbz opponent matrix is invertible"),
            invert3(&JZ_XYZ_TO_LMS).expect("Jzazbz cone matrix is invertible"),
        )
    });
    let lms = mul3v(inv_iab, [iz, jab[1], jab[2]]).map(jz_pq_inv);
    let [xp, yp, z] = mul3v(inv_lms, lms);
    let x = (xp + (JZ_B - 1.0) * z) / JZ_B;
    let y = (yp + (JZ_G - 1.0) * x) / JZ_G;
    [x, y, z]
}

// --------------------------------------------------------------------
// Kodak PhotoYCC (Photo CD system description, "PhotoYCC colour
// encoding").

/// The ITU-R BT.709 opto-electronic transfer `1.099·L^0.45 − 0.099`
/// (`4.5·L` below 0.018), mirrored through zero as PhotoYCC prescribes
/// for negative scene values.
fn photoycc_oetf(l: f64) -> f64 {
    let a = l.abs();
    let v = if a >= 0.018 {
        1.099 * a.powf(0.45) - 0.099
    } else {
        4.5 * a
    };
    v.copysign(l)
}

fn photoycc_oetf_inv(v: f64) -> f64 {
    let a = v.abs();
    let l = if a >= 4.5 * 0.018 {
        ((a + 0.099) / 1.099).powf(1.0 / 0.45)
    } else {
        a / 4.5
    };
    l.copysign(v)
}

/// Linear-light RGB (BT.709 primaries, D65) → Kodak PhotoYCC, each
/// component as its 8-bit code divided by 255.
///
/// The Photo CD definition: R′G′B′ from the BT.709 transfer function
/// (mirrored for negative values); `Y = 0.299·R′ + 0.587·G′ +
/// 0.114·B′`, `C1 = B′ − Y`, `C2 = R′ − Y`; and the 8-bit codes `Y₈ =
/// (255 / 1.402)·Y`, `C1₈ = 111.40·C1 + 156`, `C2₈ = 135.64·C2 + 137`.
/// A scene white of 1.0 therefore sits at `Y₈ ≈ 182`, leaving headroom
/// for highlights up to 1.402.
pub fn linear_rgb_to_photoycc(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb.map(photoycc_oetf);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    [
        y / 1.402,
        (111.40 * (b - y) + 156.0) / 255.0,
        (135.64 * (r - y) + 137.0) / 255.0,
    ]
}

/// Kodak PhotoYCC (codes / 255) → linear-light RGB; the exact inverse
/// of [`linear_rgb_to_photoycc`].
pub fn photoycc_to_linear_rgb(ycc: [f64; 3]) -> [f64; 3] {
    let y = ycc[0] * 1.402;
    let c1 = (ycc[1] * 255.0 - 156.0) / 111.40;
    let c2 = (ycc[2] * 255.0 - 137.0) / 135.64;
    let r = c2 + y;
    let b = c1 + y;
    let g = (y - 0.299 * r - 0.114 * b) / 0.587;
    [r, g, b].map(photoycc_oetf_inv)
}

// --------------------------------------------------------------------
// Cineon printing-density log encoding (Kodak, "Conversion of 10-bit
// Log Film Data To 8-bit Linear or Video Data").

/// Parameters of the Cineon 10-bit printing-density encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct CineonLog {
    /// 10-bit code of the reference white (a 90 % reflector): 685.
    pub reference_white: f64,
    /// 10-bit code of the reference black (a 1 % reflector): 95.
    pub reference_black: f64,
    /// Negative-film gamma: 0.6.
    pub film_gamma: f64,
    /// Printing density per code value: 0.002.
    pub density_per_code: f64,
}

impl Default for CineonLog {
    fn default() -> Self {
        Self::new()
    }
}

impl CineonLog {
    /// The standard parameters: white 685, black 95, gamma 0.6, 0.002
    /// density per code.
    pub const fn new() -> Self {
        Self {
            reference_white: 685.0,
            reference_black: 95.0,
            film_gamma: 0.6,
            density_per_code: 0.002,
        }
    }

    /// Set the reference white and black codes.
    pub const fn with_references(mut self, white: f64, black: f64) -> Self {
        self.reference_white = white;
        self.reference_black = black;
        self
    }

    /// Set the film gamma.
    pub const fn with_film_gamma(mut self, gamma: f64) -> Self {
        self.film_gamma = gamma;
        self
    }

    /// Code values per decade of exposure: `gamma / density_per_code`
    /// (300 with the standard parameters).
    fn codes_per_decade(&self) -> f64 {
        self.film_gamma / self.density_per_code
    }

    /// The exposure, relative to the reference white, that the
    /// reference black code stands for.
    fn black_offset(&self) -> f64 {
        10f64.powf((self.reference_black - self.reference_white) / self.codes_per_decade())
    }

    /// Linear light (0 at reference black, 1 at reference white) → the
    /// 10-bit code, as a real number: `code = white + (gamma /
    /// density)·log10(L·(1 − o) + o)` with `o = 10^((black − white)·
    /// density / gamma)`. The code is not clamped above (highlights
    /// beyond white run past 685); exposure at or below film base
    /// clamps to code 0.
    pub fn encode(&self, linear: f64) -> f64 {
        let o = self.black_offset();
        let floor = 10f64.powf(-self.reference_white / self.codes_per_decade());
        let e = (linear * (1.0 - o) + o).max(floor);
        self.reference_white + self.codes_per_decade() * e.log10()
    }

    /// 10-bit code → linear light; the inverse of [`Self::encode`].
    pub fn decode(&self, code: f64) -> f64 {
        let o = self.black_offset();
        let e = 10f64.powf((code - self.reference_white) / self.codes_per_decade());
        (e - o) / (1.0 - o)
    }
}

// --------------------------------------------------------------------
// Cone-response (LMS) matrices.

/// The matrix taking CIE XYZ to an LMS cone response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConeResponse {
    /// `M_CAT02`, the chromatic-adaptation matrix of CIECAM02 (CIE
    /// 159:2004): rows `(0.7328, 0.4296, −0.1624)`, `(−0.7036, 1.6975,
    /// 0.0061)`, `(0.0030, 0.0136, 0.9834)`.
    Cat02,
    /// The linearised Bradford matrix (ICC.1:2022 Annex E equation
    /// E.1): rows `(0.8951, 0.2664, −0.1614)`, `(−0.7502, 1.7135,
    /// 0.0367)`, `(0.0389, −0.0685, 1.0296)`.
    Bradford,
    /// Hunt–Pointer–Estevez cone fundamentals normalised to D65 (the
    /// `M_HPE` used by the RLAB and LLAB appearance models): rows
    /// `(0.4002, 0.7076, −0.0808)`, `(−0.2263, 1.1653, 0.0457)`,
    /// `(0, 0, 0.9182)`.
    HuntPointerEstevez,
}

impl ConeResponse {
    /// The XYZ → LMS matrix.
    pub const fn matrix(self) -> [[f64; 3]; 3] {
        match self {
            ConeResponse::Cat02 => [
                [0.7328, 0.4296, -0.1624],
                [-0.7036, 1.6975, 0.0061],
                [0.0030, 0.0136, 0.9834],
            ],
            ConeResponse::Bradford => super::cie::BRADFORD,
            ConeResponse::HuntPointerEstevez => [
                [0.4002, 0.7076, -0.0808],
                [-0.2263, 1.1653, 0.0457],
                [0.0, 0.0, 0.9182],
            ],
        }
    }
}
