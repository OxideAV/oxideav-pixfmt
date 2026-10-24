//! CIE colorimetry: chromaticities, RGB ↔ XYZ matrices from primaries,
//! linearised Bradford chromatic adaptation, CIELAB / LCh(ab), xyY and
//! the CIEDE2000 colour difference.

use crate::mat3::{Mat3, invert3, mul3, mul3v};

/// A CIE 1931 xy chromaticity.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Chromaticity {
    /// x.
    pub x: f64,
    /// y.
    pub y: f64,
}

impl Chromaticity {
    /// A chromaticity.
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// XYZ of this chromaticity at luminance `Y = 1` (Khronos Data
    /// Format 1.4 §14: `X = x / y`, `Z = (1 − x − y) / y`).
    pub fn to_xyz(self) -> [f64; 3] {
        [self.x / self.y, 1.0, (1.0 - self.x - self.y) / self.y]
    }
}

/// CIE D65 as the 4-figure chromaticity BT.709 / sRGB use
/// (x 0.3127, y 0.3290).
pub const D65: Chromaticity = Chromaticity::new(0.3127, 0.3290);
/// CIE D50 as the 4-figure chromaticity (x 0.3457, y 0.3585) used by
/// CSS Color 4 §19 for ProPhoto RGB and CIE Lab.
pub const D50: Chromaticity = Chromaticity::new(0.3457, 0.3585);

/// Red, green and blue primaries plus white of an RGB space.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RgbPrimaries {
    /// Red chromaticity.
    pub red: Chromaticity,
    /// Green chromaticity.
    pub green: Chromaticity,
    /// Blue chromaticity.
    pub blue: Chromaticity,
    /// White chromaticity.
    pub white: Chromaticity,
}

impl RgbPrimaries {
    /// Primaries from four chromaticities.
    pub const fn new(
        red: Chromaticity,
        green: Chromaticity,
        blue: Chromaticity,
        white: Chromaticity,
    ) -> Self {
        Self {
            red,
            green,
            blue,
            white,
        }
    }

    /// BT.709 / sRGB (BT.709-6 Part 2 item 1.3–1.4; Khronos §14.1).
    pub const BT709: Self = Self {
        red: Chromaticity::new(0.640, 0.330),
        green: Chromaticity::new(0.300, 0.600),
        blue: Chromaticity::new(0.150, 0.060),
        white: D65,
    };
    /// BT.2020 / BT.2100 (Khronos §14.4).
    pub const BT2020: Self = Self {
        red: Chromaticity::new(0.708, 0.292),
        green: Chromaticity::new(0.170, 0.797),
        blue: Chromaticity::new(0.131, 0.046),
        white: D65,
    };
    /// Display P3 (Khronos §14.9).
    pub const DISPLAY_P3: Self = Self {
        red: Chromaticity::new(0.680, 0.320),
        green: Chromaticity::new(0.265, 0.690),
        blue: Chromaticity::new(0.150, 0.060),
        white: D65,
    };
    /// Adobe RGB (1998) (Khronos §14.10).
    pub const ADOBE_RGB: Self = Self {
        red: Chromaticity::new(0.640, 0.330),
        green: Chromaticity::new(0.210, 0.710),
        blue: Chromaticity::new(0.150, 0.060),
        white: D65,
    };
    /// ProPhoto RGB / ROMM (CSS Color 4 §10.7), D50 white.
    pub const PROPHOTO: Self = Self {
        red: Chromaticity::new(0.734699, 0.265301),
        green: Chromaticity::new(0.159597, 0.840403),
        blue: Chromaticity::new(0.036598, 0.000105),
        white: D50,
    };

    /// The linear RGB → XYZ matrix (white at `Y = 1`), derived as in
    /// Khronos Data Format 1.4 §14: the primaries' XYZ at unit
    /// luminance, scaled so `R + G + B` reproduces the white.
    pub fn rgb_to_xyz(&self) -> [[f64; 3]; 3] {
        let r = self.red.to_xyz();
        let g = self.green.to_xyz();
        let b = self.blue.to_xyz();
        let p: Mat3 = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        let w = self.white.to_xyz();
        let s = mul3v(&invert3(&p).expect("primaries are not collinear"), w);
        let mut m = p;
        for row in &mut m {
            for c in 0..3 {
                row[c] *= s[c];
            }
        }
        m
    }
}

/// The linearised Bradford cone-response matrix `M_BFD`
/// (ICC.1:2022 Annex E, equation E.1).
pub const BRADFORD: [[f64; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Chromatic adaptation matrix from white `src` to white `dst` (both
/// XYZ) by the linearised Bradford transform (ICC.1:2022 Annex E,
/// equations E.2–E.4).
pub fn bradford_adaptation(src: [f64; 3], dst: [f64; 3]) -> [[f64; 3]; 3] {
    let s = mul3v(&BRADFORD, src);
    let d = mul3v(&BRADFORD, dst);
    let diag = [
        [d[0] / s[0], 0.0, 0.0],
        [0.0, d[1] / s[1], 0.0],
        [0.0, 0.0, d[2] / s[2]],
    ];
    let inv = invert3(&BRADFORD).expect("Bradford matrix is invertible");
    mul3(&inv, &mul3(&diag, &BRADFORD))
}

/// CIEDE2000 colour difference between two CIELAB colours, with the
/// parametric factors `kL = kC = kH = 1` — transcribed from the sample
/// code of CSS Color 4 §20.2 (validated there against Sharma's test
/// pairs).
pub fn delta_e2000(reference: [f64; 3], sample: [f64; 3]) -> f64 {
    use core::f64::consts::PI;
    let [l1, a1, b1] = reference;
    let [l2, a2, b2] = sample;
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let cbar = (c1 + c2) / 2.0;
    let c7 = cbar.powi(7);
    let gfactor = 25f64.powi(7);
    let g = 0.5 * (1.0 - (c7 / (c7 + gfactor)).sqrt());
    let adash1 = (1.0 + g) * a1;
    let adash2 = (1.0 + g) * a2;
    let cdash1 = (adash1 * adash1 + b1 * b1).sqrt();
    let cdash2 = (adash2 * adash2 + b2 * b2).sqrt();
    let r2d = 180.0 / PI;
    let d2r = PI / 180.0;
    let mut h1 = if adash1 == 0.0 && b1 == 0.0 {
        0.0
    } else {
        b1.atan2(adash1)
    };
    let mut h2 = if adash2 == 0.0 && b2 == 0.0 {
        0.0
    } else {
        b2.atan2(adash2)
    };
    if h1 < 0.0 {
        h1 += 2.0 * PI;
    }
    if h2 < 0.0 {
        h2 += 2.0 * PI;
    }
    h1 *= r2d;
    h2 *= r2d;
    let dl = l2 - l1;
    let dc = cdash2 - cdash1;
    let hdiff = h2 - h1;
    let hsum = h1 + h2;
    let habs = hdiff.abs();
    let dh = if cdash1 * cdash2 == 0.0 {
        0.0
    } else if habs <= 180.0 {
        hdiff
    } else if hdiff > 180.0 {
        hdiff - 360.0
    } else {
        hdiff + 360.0
    };
    let d_big_h = 2.0 * (cdash2 * cdash1).sqrt() * (dh * d2r / 2.0).sin();
    let ldash = (l1 + l2) / 2.0;
    let cdash = (cdash1 + cdash2) / 2.0;
    let cdash7 = cdash.powi(7);
    let hdash = if cdash1 * cdash2 == 0.0 {
        hsum
    } else if habs <= 180.0 {
        hsum / 2.0
    } else if hsum < 360.0 {
        (hsum + 360.0) / 2.0
    } else {
        (hsum - 360.0) / 2.0
    };
    let lsq = (ldash - 50.0) * (ldash - 50.0);
    let sl = 1.0 + (0.015 * lsq) / (20.0 + lsq).sqrt();
    let sc = 1.0 + 0.045 * cdash;
    let mut t = 1.0;
    t -= 0.17 * ((hdash - 30.0) * d2r).cos();
    t += 0.24 * ((2.0 * hdash) * d2r).cos();
    t += 0.32 * ((3.0 * hdash + 6.0) * d2r).cos();
    t -= 0.20 * ((4.0 * hdash - 63.0) * d2r).cos();
    let sh = 1.0 + 0.015 * cdash * t;
    let dtheta = 30.0 * (-((hdash - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (cdash7 / (cdash7 + gfactor)).sqrt();
    let rt = -(2.0 * dtheta * d2r).sin() * rc;
    let mut de = (dl / sl).powi(2);
    de += (dc / sc).powi(2);
    de += (d_big_h / sh).powi(2);
    de += rt * (dc / sc) * (d_big_h / sh);
    de.sqrt()
}
