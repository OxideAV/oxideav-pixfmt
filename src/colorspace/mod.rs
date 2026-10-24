//! Floating-point colour models and colorimetry.
//!
//! [`ColorModel`] converts gamma-encoded sRGB to and from linear sRGB,
//! CIE XYZ (D65 / D50), xyY, CIELAB and LCh(ab) (D65 or D50 white),
//! Oklab / OkLCh, HSL, HWB, device CMY, gray (gamma-encoded or linear
//! luma), Y′PbPr and full-range Y′CbCr — every model whose defining
//! formulas are staged in the workspace `docs/` tree (see
//! [`models`](self) for the clause list). Device CMYK, with black
//! generation and undercolour removal, has its own four-channel
//! functions ([`rgb_to_cmyk_with`], [`cmyk_to_rgb`]).
//!
//! Everything is `f64` and unclamped, so a model → sRGB → model round
//! trip is exact to rounding; quantise at the edges.

mod cie;
mod models;

pub use cie::{BRADFORD, Chromaticity, D50, D65, RgbPrimaries, bradford_adaptation, delta_e2000};
pub use models::{
    ColorModel, LabWhite, LumaWeights, cmy_to_rgb, cmyk_to_rgb, hsl_to_rgb, hwb_to_rgb, lab_to_lch,
    lab_to_xyz, lch_to_lab, linear_to_srgb, oklab_to_xyz_d65, rgb_to_cmy, rgb_to_cmyk,
    rgb_to_cmyk_with, rgb_to_hsl, rgb_to_hwb, rgb_to_ypbpr, srgb_to_linear, xyy_to_xyz,
    xyz_d65_to_oklab, xyz_to_lab, xyz_to_xyy, ypbpr_to_rgb,
};
