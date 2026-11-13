//! Floating-point colour models and colorimetry.
//!
//! [`ColorModel`] converts gamma-encoded sRGB to and from:
//!
//! - linear sRGB, scRGB, gray and linear gray, device CMY;
//! - the hue models HSL, HWB, HSV / HSB, HSI, hexagonal HCL / HCLp;
//! - CIE XYZ (D65 / D50), xyY, CIELAB and LCh(ab), CIELUV and LCh(uv)
//!   (D65 or D50 white; also in hue-first "HCL" order), LMS through a
//!   named cone-response matrix;
//! - Y′PbPr and Y′CbCr (BT.601 / BT.709 / BT.2020 weights, full and
//!   narrow range), Y′IQ, Y′UV, Y′DbDr, Kodak PhotoYCC, Ohta I1I2I3;
//! - Oklab / OkLCh, Jzazbz / JzCzhz and the Cineon log encoding.
//!
//! Models defined on linear light linearise sRGB first. Device CMYK,
//! with black generation and undercolour removal, has its own
//! four-channel functions ([`rgb_to_cmyk_with`], [`cmyk_to_rgb`]).
//!
//! Everything is `f64` and unclamped, so a model → sRGB → model round
//! trip is exact to rounding and out-of-gamut values survive; quantise
//! (and clamp, if wanted) at the edges.

mod cie;
mod extra;
mod models;

pub use cie::{BRADFORD, Chromaticity, D50, D65, RgbPrimaries, bradford_adaptation, delta_e2000};
pub use extra::{
    CineonLog, ConeResponse, jzazbz_to_xyz_abs, linear_rgb_to_photoycc, luv_to_xyz,
    photoycc_to_linear_rgb, xyz_abs_to_jzazbz, xyz_to_luv,
};
pub use models::{
    ColorModel, LabWhite, LumaWeights, cmy_to_rgb, cmyk_to_rgb, hcl_to_rgb, hclp_to_rgb,
    hsi_to_rgb, hsl_to_rgb, hsv_to_rgb, hwb_to_rgb, lab_to_lch, lab_to_xyz, lch_to_lab,
    linear_to_srgb, oklab_to_xyz_d65, rgb_to_cmy, rgb_to_cmyk, rgb_to_cmyk_with, rgb_to_hcl,
    rgb_to_hsi, rgb_to_hsl, rgb_to_hsv, rgb_to_hwb, rgb_to_ypbpr, srgb_to_linear, xyy_to_xyz,
    xyz_d65_to_oklab, xyz_to_lab, xyz_to_xyy, ypbpr_to_rgb,
};
