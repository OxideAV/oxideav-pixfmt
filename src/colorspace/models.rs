//! Per-pixel colour-model conversions in natural units (`f64`).
//!
//! Sources, all under `docs/`:
//!
//! - sRGB transfer, HSL, HWB, CIELAB / LCh, Oklab / OkLCh: W3C CSS Color
//!   Module Level 4 (`text/opentype/spec/w3c-css-color-4.html`) §7, §8,
//!   §9, §10.2 and the §19 sample code;
//! - CIELAB ↔ XYZ for an arbitrary reference white: ICC.1:2022 Annex A
//!   (`image/icc/`);
//! - xyY and the RGB ↔ XYZ derivation from primaries: Khronos Data
//!   Format 1.4 §14 (`image/astc/khronos-dataformat-1.4.html`);
//! - CMY, CMYK (black generation / undercolour removal) and the device
//!   gray ↔ CMYK rules: ISO 32000-2 §10.4.2 (`document/pdf/`);
//! - Y′PbPr / Y′CbCr from `K_R`, `K_B`: ITU-T H.273 equations 44–46 and
//!   37 (full-range digital offset), with the BT.601 / BT.709 / BT.2020
//!   weights (`video/signal-metadata/`).

use super::cie::{RgbPrimaries, bradford_adaptation};
use crate::mat3::{Mat3, invert3, mul3v};

// --------------------------------------------------------------------
// sRGB transfer (CSS Color 4 §10.2 / §19 `lin_sRGB`, `gam_sRGB`),
// extended to negative values by reflection as the sample code does.

/// Gamma-encoded sRGB component → linear light.
pub fn srgb_to_linear(v: f64) -> f64 {
    let a = v.abs();
    if a <= 0.04045 {
        v / 12.92
    } else {
        v.signum() * ((a + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light → gamma-encoded sRGB component.
pub fn linear_to_srgb(v: f64) -> f64 {
    let a = v.abs();
    if a > 0.0031308 {
        v.signum() * (1.055 * a.powf(1.0 / 2.4) - 0.055)
    } else {
        12.92 * v
    }
}

fn map3(v: [f64; 3], f: fn(f64) -> f64) -> [f64; 3] {
    [f(v[0]), f(v[1]), f(v[2])]
}

// --------------------------------------------------------------------
// XYZ / xyY / Lab / LCh

/// XYZ → xyY (Khronos DF §14: `x = X / (X + Y + Z)`, `y = Y / (X + Y +
/// Z)`). Black (`X + Y + Z = 0`) maps to the chromaticity `white` with
/// `Y = 0`.
pub fn xyz_to_xyy(xyz: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let s = xyz[0] + xyz[1] + xyz[2];
    if s == 0.0 {
        let ws = white[0] + white[1] + white[2];
        return [white[0] / ws, white[1] / ws, 0.0];
    }
    [xyz[0] / s, xyz[1] / s, xyz[1]]
}

/// xyY → XYZ (`X = Y·x / y`, `Z = Y·(1 − x − y) / y`).
pub fn xyy_to_xyz(xyy: [f64; 3]) -> [f64; 3] {
    let [x, y, big_y] = xyy;
    if y == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [big_y * x / y, big_y, big_y * (1.0 - x - y) / y]
}

#[inline]
fn lab_f(t: f64) -> f64 {
    const E: f64 = 216.0 / 24389.0;
    if t > E {
        t.cbrt()
    } else {
        841.0 / 108.0 * t + 4.0 / 29.0
    }
}

#[inline]
fn lab_f_inv(f: f64) -> f64 {
    if f > 6.0 / 29.0 {
        f * f * f
    } else {
        (f - 4.0 / 29.0) * 108.0 / 841.0
    }
}

/// XYZ → CIELAB (`L*` 0..100) relative to the reference white `white`
/// (ICC.1:2022 Annex A equations A.4–A.12; the linear segment is
/// extended below zero as §6.4 prescribes).
pub fn xyz_to_lab(xyz: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let fx = lab_f(xyz[0] / white[0]);
    let fy = lab_f(xyz[1] / white[1]);
    let fz = lab_f(xyz[2] / white[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIELAB → XYZ relative to `white`.
pub fn lab_to_xyz(lab: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    [
        lab_f_inv(fx) * white[0],
        lab_f_inv(fy) * white[1],
        lab_f_inv(fz) * white[2],
    ]
}

/// Rectangular `(L, a, b)` → polar `(L, C, h)` with `h` in degrees
/// `[0, 360)` (CSS Color 4 §9.5). The hue of an achromatic colour is
/// "powerless"; it is returned as 0.
pub fn lab_to_lch(lab: [f64; 3]) -> [f64; 3] {
    let c = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    let mut h = lab[2].atan2(lab[1]).to_degrees();
    if h < 0.0 {
        h += 360.0;
    }
    if h >= 360.0 || c == 0.0 {
        h = if c == 0.0 { 0.0 } else { h - 360.0 };
    }
    [lab[0], c, h]
}

/// Polar `(L, C, h°)` → rectangular `(L, a, b)` (CSS Color 4 §9.6).
pub fn lch_to_lab(lch: [f64; 3]) -> [f64; 3] {
    let h = lch[2].to_radians();
    [lch[0], lch[1] * h.cos(), lch[1] * h.sin()]
}

// --------------------------------------------------------------------
// Oklab (CSS Color 4 §9.2 and §19 `XYZ_to_OKLab` / `OKLab_to_XYZ`).

#[allow(clippy::excessive_precision)] // transcribed verbatim
const XYZ_TO_LMS: Mat3 = [
    [0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
    [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
    [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
];
#[allow(clippy::excessive_precision)] // transcribed verbatim
const LMS_TO_OKLAB: Mat3 = [
    [0.2104542683093140, 0.7936177747023054, -0.0040720430116193],
    [1.9779985324311684, -2.4285922420485799, 0.4505937096174110],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
];
#[allow(clippy::excessive_precision)] // transcribed verbatim
const LMS_TO_XYZ: Mat3 = [
    [1.2268798758459243, -0.5578149944602171, 0.2813910456659647],
    [-0.0405757452148008, 1.1122868032803170, -0.0717110580655164],
    [-0.0763729366746601, -0.4214933324022432, 1.5869240198367816],
];
#[allow(clippy::excessive_precision)] // transcribed verbatim
const OKLAB_TO_LMS: Mat3 = [
    [1.0000000000000000, 0.3963377773761749, 0.2158037573099136],
    [1.0000000000000000, -0.1055613458156586, -0.0638541728258133],
    [1.0000000000000000, -0.0894841775298119, -1.2914855480194092],
];

/// XYZ (D65, `Y = 1` white) → Oklab (`L` 0..1).
pub fn xyz_d65_to_oklab(xyz: [f64; 3]) -> [f64; 3] {
    let lms = mul3v(&XYZ_TO_LMS, xyz);
    mul3v(&LMS_TO_OKLAB, [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()])
}

/// Oklab → XYZ (D65).
pub fn oklab_to_xyz_d65(lab: [f64; 3]) -> [f64; 3] {
    let l = mul3v(&OKLAB_TO_LMS, lab);
    mul3v(
        &LMS_TO_XYZ,
        [l[0] * l[0] * l[0], l[1] * l[1] * l[1], l[2] * l[2] * l[2]],
    )
}

// --------------------------------------------------------------------
// HSL / HWB (CSS Color 4 §7, §8). Hue in degrees, the rest in 0..1.

/// sRGB → HSL. An achromatic colour's powerless hue is returned as 0.
pub fn rgb_to_hsl(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let light = (min + max) / 2.0;
    let d = max - min;
    let (mut hue, mut sat) = (0.0, 0.0);
    if d != 0.0 {
        sat = if light == 0.0 || light == 1.0 {
            0.0
        } else {
            (max - light) / light.min(1.0 - light)
        };
        hue = hue_of(r, g, b, max, d);
    }
    if sat < 0.0 {
        hue += 180.0;
        sat = sat.abs();
    }
    if hue >= 360.0 {
        hue -= 360.0;
    }
    [hue, sat, light]
}

fn hue_of(r: f64, g: f64, b: f64, max: f64, d: f64) -> f64 {
    let h = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h * 60.0
}

/// HSL → sRGB.
pub fn hsl_to_rgb(hsl: [f64; 3]) -> [f64; 3] {
    let [hue, sat, light] = hsl;
    let f = |n: f64| {
        let k = (n + hue / 30.0).rem_euclid(12.0);
        let a = sat * light.min(1.0 - light);
        light - a * (-1f64).max((k - 3.0).min(9.0 - k).min(1.0))
    };
    [f(0.0), f(8.0), f(4.0)]
}

/// sRGB → HWB (hue°, whiteness, blackness).
pub fn rgb_to_hwb(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let mut hue = if d != 0.0 {
        hue_of(r, g, b, max, d)
    } else {
        0.0
    };
    if hue >= 360.0 {
        hue -= 360.0;
    }
    [hue, min, 1.0 - max]
}

/// HWB → sRGB.
pub fn hwb_to_rgb(hwb: [f64; 3]) -> [f64; 3] {
    let [hue, white, black] = hwb;
    if white + black >= 1.0 {
        let gray = white / (white + black);
        return [gray, gray, gray];
    }
    let rgb = hsl_to_rgb([hue, 1.0, 0.5]);
    rgb.map(|c| c * (1.0 - white - black) + white)
}

// --------------------------------------------------------------------
// Device CMY / CMYK / gray (ISO 32000-2 §10.4.2).

/// RGB → CMY (`c = 1 − red`, …; §10.4.2.4).
pub fn rgb_to_cmy(rgb: [f64; 3]) -> [f64; 3] {
    rgb.map(|v| 1.0 - v)
}

/// CMY → RGB.
pub fn cmy_to_rgb(cmy: [f64; 3]) -> [f64; 3] {
    cmy.map(|v| 1.0 - v)
}

/// RGB → CMYK with black generation `bg` and undercolour removal `ucr`
/// (§10.4.2.4): `k = min(c, m, y)`, `cyan = clamp(c − UCR(k))`, …,
/// `black = clamp(BG(k))`.
pub fn rgb_to_cmyk_with(
    rgb: [f64; 3],
    bg: &dyn Fn(f64) -> f64,
    ucr: &dyn Fn(f64) -> f64,
) -> [f64; 4] {
    let [c, m, y] = rgb_to_cmy(rgb);
    let k = c.min(m).min(y);
    let u = ucr(k);
    [
        (c - u).clamp(0.0, 1.0),
        (m - u).clamp(0.0, 1.0),
        (y - u).clamp(0.0, 1.0),
        bg(k).clamp(0.0, 1.0),
    ]
}

/// RGB → CMYK with full black generation and full undercolour removal
/// (`BG(k) = UCR(k) = k`), which [`cmyk_to_rgb`] inverts exactly.
pub fn rgb_to_cmyk(rgb: [f64; 3]) -> [f64; 4] {
    rgb_to_cmyk_with(rgb, &|k| k, &|k| k)
}

/// CMYK → RGB (§10.4.2.5: `red = 1 − min(1, cyan + black)`, …).
pub fn cmyk_to_rgb(cmyk: [f64; 4]) -> [f64; 3] {
    let k = cmyk[3];
    [
        1.0 - (cmyk[0] + k).min(1.0),
        1.0 - (cmyk[1] + k).min(1.0),
        1.0 - (cmyk[2] + k).min(1.0),
    ]
}

// --------------------------------------------------------------------
// Luma-based models.

/// Luma weights `(K_R, K_B)`; `K_G = 1 − K_R − K_B`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct LumaWeights {
    /// `K_R`.
    pub kr: f64,
    /// `K_B`.
    pub kb: f64,
}

impl LumaWeights {
    /// Weights from `K_R` and `K_B`.
    pub const fn new(kr: f64, kb: f64) -> Self {
        Self { kr, kb }
    }
    /// BT.601 (H.273 MatrixCoefficients 5 / 6): 0.299 / 0.114.
    pub const BT601: Self = Self::new(0.299, 0.114);
    /// BT.709 (MatrixCoefficients 1): 0.2126 / 0.0722.
    pub const BT709: Self = Self::new(0.2126, 0.0722);
    /// BT.2020 (MatrixCoefficients 9): 0.2627 / 0.0593.
    pub const BT2020: Self = Self::new(0.2627, 0.0593);
    /// ISO 32000-2 §10.4.2.2 device gray: 0.3 / 0.11 (`G` 0.59).
    pub const PDF_DEVICE_GRAY: Self = Self::new(0.3, 0.11);

    /// `K_R·R + K_G·G + K_B·B`.
    pub fn luma(self, rgb: [f64; 3]) -> f64 {
        self.kr * rgb[0] + (1.0 - self.kr - self.kb) * rgb[1] + self.kb * rgb[2]
    }
}

/// R′G′B′ → Y′PbPr (H.273 equations 44–46): `Pb = 0.5·(B′ − Y′) / (1 −
/// K_B)`, `Pr = 0.5·(R′ − Y′) / (1 − K_R)`, both in −0.5..0.5.
pub fn rgb_to_ypbpr(rgb: [f64; 3], w: LumaWeights) -> [f64; 3] {
    let y = w.luma(rgb);
    [
        y,
        0.5 * (rgb[2] - y) / (1.0 - w.kb),
        0.5 * (rgb[0] - y) / (1.0 - w.kr),
    ]
}

/// Y′PbPr → R′G′B′.
pub fn ypbpr_to_rgb(ypp: [f64; 3], w: LumaWeights) -> [f64; 3] {
    let [y, pb, pr] = ypp;
    let r = y + 2.0 * (1.0 - w.kr) * pr;
    let b = y + 2.0 * (1.0 - w.kb) * pb;
    let g = (y - w.kr * r - w.kb * b) / (1.0 - w.kr - w.kb);
    [r, g, b]
}

// --------------------------------------------------------------------
// Whole-pipeline model enum.

/// Reference white for CIELAB / LCh(ab).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LabWhite {
    /// D65 (sRGB's own white; XYZ used without adaptation).
    D65,
    /// D50 via the linearised Bradford adaptation (CSS Color 4 `lab()`,
    /// the ICC PCS).
    D50,
}

/// A three-component colour model reachable from (gamma-encoded) sRGB.
///
/// Values are in natural units: RGB, CMY, gray, HSL / HWB saturation,
/// lightness, whiteness and blackness in 0..1; hues in degrees; XYZ
/// with `Y = 1` for white; `L*` 0..100 with `a*`, `b*`, `C*` unscaled;
/// Oklab `L` 0..1; Y′PbPr chroma in −0.5..0.5 and Y′CbCr chroma offset
/// by +0.5 (the full-range digital form of H.273 equation 37,
/// normalised). Gray models return the luma in all three slots.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ColorModel {
    /// Gamma-encoded sRGB (identity).
    Srgb,
    /// Linear-light sRGB (CSS `srgb-linear`).
    LinearSrgb,
    /// CIE XYZ relative to D65.
    XyzD65,
    /// CIE XYZ adapted to D50 (Bradford).
    XyzD50,
    /// CIE xyY (D65 XYZ).
    XyY,
    /// CIELAB.
    Lab(LabWhite),
    /// CIE LCh(ab).
    Lch(LabWhite),
    /// Oklab.
    Oklab,
    /// OkLCh.
    Oklch,
    /// HSL.
    Hsl,
    /// HWB.
    Hwb,
    /// Device CMY.
    Cmy,
    /// Luma of the gamma-encoded values.
    Gray(LumaWeights),
    /// Luma of the linear-light values (stays linear).
    LinearGray(LumaWeights),
    /// Y′PbPr.
    YPbPr(LumaWeights),
    /// Full-range Y′CbCr (`Cb = Pb + 0.5`, `Cr = Pr + 0.5`).
    YCbCr(LumaWeights),
}

fn d65() -> [f64; 3] {
    super::cie::D65.to_xyz()
}

fn srgb_to_xyz_m() -> Mat3 {
    RgbPrimaries::BT709.rgb_to_xyz()
}

fn d65_to_d50() -> Mat3 {
    bradford_adaptation(d65(), super::cie::D50.to_xyz())
}

impl ColorModel {
    /// Convert gamma-encoded sRGB into this model.
    pub fn from_srgb(self, rgb: [f64; 3]) -> [f64; 3] {
        let xyz65 = || mul3v(&srgb_to_xyz_m(), map3(rgb, srgb_to_linear));
        let lab = |w: LabWhite| match w {
            LabWhite::D65 => xyz_to_lab(xyz65(), d65()),
            LabWhite::D50 => xyz_to_lab(mul3v(&d65_to_d50(), xyz65()), super::cie::D50.to_xyz()),
        };
        match self {
            ColorModel::Srgb => rgb,
            ColorModel::LinearSrgb => map3(rgb, srgb_to_linear),
            ColorModel::XyzD65 => xyz65(),
            ColorModel::XyzD50 => mul3v(&d65_to_d50(), xyz65()),
            ColorModel::XyY => xyz_to_xyy(xyz65(), d65()),
            ColorModel::Lab(w) => lab(w),
            ColorModel::Lch(w) => lab_to_lch(lab(w)),
            ColorModel::Oklab => xyz_d65_to_oklab(xyz65()),
            ColorModel::Oklch => lab_to_lch(xyz_d65_to_oklab(xyz65())),
            ColorModel::Hsl => rgb_to_hsl(rgb),
            ColorModel::Hwb => rgb_to_hwb(rgb),
            ColorModel::Cmy => rgb_to_cmy(rgb),
            ColorModel::Gray(w) => [w.luma(rgb); 3],
            ColorModel::LinearGray(w) => [w.luma(map3(rgb, srgb_to_linear)); 3],
            ColorModel::YPbPr(w) => rgb_to_ypbpr(rgb, w),
            ColorModel::YCbCr(w) => {
                let v = rgb_to_ypbpr(rgb, w);
                [v[0], v[1] + 0.5, v[2] + 0.5]
            }
        }
    }

    /// Convert from this model back to gamma-encoded sRGB (unclamped).
    pub fn to_srgb(self, v: [f64; 3]) -> [f64; 3] {
        let from_xyz65 = |xyz: [f64; 3]| {
            let m = invert3(&srgb_to_xyz_m()).expect("sRGB matrix is invertible");
            map3(mul3v(&m, xyz), linear_to_srgb)
        };
        let from_d50 = |xyz: [f64; 3]| {
            let m = invert3(&d65_to_d50()).expect("Bradford matrix is invertible");
            from_xyz65(mul3v(&m, xyz))
        };
        let from_lab = |lab: [f64; 3], w: LabWhite| match w {
            LabWhite::D65 => from_xyz65(lab_to_xyz(lab, d65())),
            LabWhite::D50 => from_d50(lab_to_xyz(lab, super::cie::D50.to_xyz())),
        };
        match self {
            ColorModel::Srgb => v,
            ColorModel::LinearSrgb => map3(v, linear_to_srgb),
            ColorModel::XyzD65 => from_xyz65(v),
            ColorModel::XyzD50 => from_d50(v),
            ColorModel::XyY => from_xyz65(xyy_to_xyz(v)),
            ColorModel::Lab(w) => from_lab(v, w),
            ColorModel::Lch(w) => from_lab(lch_to_lab(v), w),
            ColorModel::Oklab => from_xyz65(oklab_to_xyz_d65(v)),
            ColorModel::Oklch => from_xyz65(oklab_to_xyz_d65(lch_to_lab(v))),
            ColorModel::Hsl => hsl_to_rgb(v),
            ColorModel::Hwb => hwb_to_rgb(v),
            ColorModel::Cmy => cmy_to_rgb(v),
            ColorModel::Gray(_) => [v[0]; 3],
            ColorModel::LinearGray(_) => [linear_to_srgb(v[0]); 3],
            ColorModel::YPbPr(w) => ypbpr_to_rgb(v, w),
            ColorModel::YCbCr(w) => ypbpr_to_rgb([v[0], v[1] - 0.5, v[2] - 0.5], w),
        }
    }

    /// Convert between two models (through sRGB).
    pub fn convert(self, to: ColorModel, v: [f64; 3]) -> [f64; 3] {
        to.from_srgb(self.to_srgb(v))
    }
}
