//! Floating-point colour-space conversions.

mod cie;

pub use cie::{BRADFORD, Chromaticity, D50, D65, RgbPrimaries, bradford_adaptation, delta_e2000};
