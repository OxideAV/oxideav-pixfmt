//! ICC colour management (feature `icc`).
//!
//! Parses version 2 and version 4 ICC profiles and builds transforms
//! between two of them, clean-room from **ICC.1:2022** (profile version
//! 4.4, `docs/image/icc/ICC.1-2022-05-profile-format-v4.4.pdf`; ISO
//! 15076-1 is the technically identical ISO text):
//!
//! - header (§7.2) and tag table (§7.3) — [`IccProfile::parse`];
//! - three-component matrix/TRC and monochrome grayTRC models
//!   (Annex F.3 / F.2), with the Annex F.1 curve-inversion rules;
//! - `curveType` and all five `parametricCurveType` functions
//!   (§10.6, §10.18 Table 68);
//! - LUT-based transforms: `lut8Type` (`mft1`), `lut16Type` (`mft2`,
//!   including the legacy 16-bit PCSLAB encoding of Tables 42/43),
//!   `lutAToBType` and `lutBToAType` (`mAB ` / `mBA `, any permitted
//!   element combination, 8- or 16-bit CLUTs with per-dimension grids);
//! - `multiProcessElementsType` (`mpet`, §10.16): the float `DToBx` /
//!   `BToDx` tags with segmented curve sets (`parf` formula and `samf`
//!   sampled segments), matrices with offsets, float CLUTs and the
//!   `bACS` / `eACS` placeholders, unclipped between elements, taken
//!   ahead of the integer tags per §8.10.2 a) and §8.10.3 a) (DToB3 /
//!   BToD3 as already ICC-absolute); a tag with an unknown element is
//!   skipped and the integer tags are used;
//! - chains of profiles ([`IccTransform::chain`]) with Abstract
//!   profiles (§8.8) and pass-through device profiles in the middle;
//! - the PCSXYZ ↔ PCSLAB conversion of §6.4 / Annex A and the PCS
//!   encodings of §6.3.4;
//! - all four rendering intents: the intent's AToBx / BToAx table with
//!   the §8.10.2 fallback order, and ICC-absolute colorimetric via the
//!   media-white-point scaling of §6.2.3 (equations 1–6);
//! - RGB, CMYK, Gray, Lab, XYZ and n-colour device spaces, on `f32`,
//!   8-bit and 16-bit interleaved buffers ([`IccTransform`]), and on
//!   packed `VideoFrame`s with alpha carry and the frame's own ICC blob
//!   ([`transform_frame`], [`frame_profile`]);
//! - multilinear or simplex ("tetrahedral") CLUT interpolation
//!   ([`ClutInterpolation`]), and any transform sampled into an
//!   in-memory device-link table for speed
//!   ([`IccTransform::precompute`]);
//! - a small profile writer and synthesized matrix/TRC profiles of the
//!   sRGB, Display P3, Adobe RGB (1998), BT.2020 and ProPhoto shapes
//!   ([`IccProfileBuilder`], [`IccProfile::srgb`] …).
//!
//! Not covered (see the crate README): black-point compensation (its
//! algorithm is ISO 18619, not part of ICC.1), NamedColor profiles,
//! DeviceLink profiles as chain members.
//!
//! Apart from the `VideoFrame` glue and the execution-budget hook
//! ([`TransformOptions::with_execution`]) the module depends on nothing
//! beyond `std`.

mod build;
mod curve;
mod frame;
mod lut;
mod mpe;
mod pcs;
mod profile;
mod transform;

pub use build::IccProfileBuilder;
pub use curve::Curve;
pub use frame::{frame_formats, frame_profile, transform_frame};
pub use lut::ClutInterpolation;
pub use pcs::{PCS_WHITE, lab_to_xyz, xyz_to_lab};
pub use profile::{DataColorSpace, IccHeader, IccProfile, ProfileClass, RenderingIntent, TagEntry};
pub use transform::{IccTransform, TransformOptions};

/// Errors from profile parsing and transform construction.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IccError {
    /// The data ends before a structure it declares.
    Truncated,
    /// Bytes 36–39 are not `acsp` (§7.2.9).
    BadSignature,
    /// A structure is internally inconsistent.
    Malformed(&'static str),
    /// A valid construct this implementation does not evaluate.
    Unsupported(&'static str),
    /// A matrix that must be inverted is singular.
    NotInvertible,
    /// A pixel buffer length does not match the transform's channels.
    BufferSize,
}

impl core::fmt::Display for IccError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IccError::Truncated => f.write_str("ICC profile truncated"),
            IccError::BadSignature => f.write_str("not an ICC profile (no 'acsp' signature)"),
            IccError::Malformed(w) => write!(f, "malformed ICC profile: {w}"),
            IccError::Unsupported(w) => write!(f, "unsupported ICC construct: {w}"),
            IccError::NotInvertible => f.write_str("ICC matrix is not invertible"),
            IccError::BufferSize => f.write_str("pixel buffer length does not match the transform"),
        }
    }
}

impl std::error::Error for IccError {}
