//! Writing profiles: a small builder for the ICC.1:2022 header (§7.2),
//! tag table (§7.3) and the tag types this module evaluates, plus
//! synthesized matrix/TRC profiles of the common RGB shapes.

use super::IccError;
use super::curve::Curve;
use super::pcs::PCS_WHITE;
use super::profile::{DataColorSpace, IccProfile, ProfileClass, RenderingIntent, sig};
use crate::colorspace::{RgbPrimaries, bradford_adaptation};
use crate::mat3::mul3;

/// Assembles an ICC profile from tags.
///
/// The header is written as version 4.4.0.0 unless
/// [`IccProfileBuilder::with_version`] says otherwise, with the D50 PCS
/// illuminant (§7.2.16), the `acsp` signature, a zero profile ID and a
/// zero creation date. Tags are laid out in insertion order, each on a
/// 4-byte boundary (§7.3.4); byte-identical tag data elements are stored
/// once and shared (§7.3.1 permits this).
#[derive(Clone, Debug)]
pub struct IccProfileBuilder {
    class: ProfileClass,
    color_space: DataColorSpace,
    pcs: DataColorSpace,
    version: (u8, u8),
    intent: RenderingIntent,
    tags: Vec<(u32, Vec<u8>)>,
}

fn put_s15f16(v: &mut Vec<u8>, x: f64) {
    let q = (x * 65536.0)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    v.extend_from_slice(&q.to_be_bytes());
}

impl IccProfileBuilder {
    /// Start a profile of the given class, data colour space and PCS.
    pub fn new(class: ProfileClass, color_space: DataColorSpace, pcs: DataColorSpace) -> Self {
        Self {
            class,
            color_space,
            pcs,
            version: (4, 0x40),
            intent: RenderingIntent::Perceptual,
            tags: Vec::new(),
        }
    }

    /// Header version: `major`, and `minor` as the BCD byte 9 (e.g.
    /// `0x40` for 4.4, `0x10` for 2.1).
    pub fn with_version(mut self, major: u8, minor_bcd: u8) -> Self {
        self.version = (major, minor_bcd);
        self
    }

    /// Header rendering intent (§7.2.15).
    pub fn with_rendering_intent(mut self, intent: RenderingIntent) -> Self {
        self.intent = intent;
        self
    }

    /// Add (or replace) a tag with its complete data element, type
    /// signature included.
    pub fn with_tag(mut self, signature: u32, data: Vec<u8>) -> Self {
        self.tags.retain(|(s, _)| *s != signature);
        self.tags.push((signature, data));
        self
    }

    /// Add an `XYZType` tag (§10.31).
    pub fn with_xyz(self, signature: u32, xyz: [f64; 3]) -> Self {
        self.with_tag(signature, Self::xyz_tag(xyz))
    }

    /// Add a curve tag (`curv` or `para`).
    pub fn with_curve(self, signature: u32, curve: &Curve) -> Self {
        self.with_tag(signature, Self::curve_tag(curve))
    }

    /// Add a `multiLocalizedUnicodeType` text tag with one en/US record.
    pub fn with_text(self, signature: u32, text: &str) -> Self {
        self.with_tag(signature, Self::mluc_tag(text))
    }

    /// Add an `s15Fixed16ArrayType` 3×3 matrix (row-major), e.g. `chad`.
    pub fn with_matrix(self, signature: u32, m: [[f64; 3]; 3]) -> Self {
        let mut v = b"sf32\0\0\0\0".to_vec();
        for x in m.iter().flatten() {
            put_s15f16(&mut v, *x);
        }
        self.with_tag(signature, v)
    }

    /// Encode an `XYZType` data element.
    pub fn xyz_tag(xyz: [f64; 3]) -> Vec<u8> {
        let mut v = b"XYZ \0\0\0\0".to_vec();
        for x in xyz {
            put_s15f16(&mut v, x);
        }
        v
    }

    /// Encode a curve data element: `curv` for identity / gamma /
    /// sampled curves (§10.6), `para` for parametric ones (§10.18).
    pub fn curve_tag(c: &Curve) -> Vec<u8> {
        match c {
            Curve::Identity => b"curv\0\0\0\0\0\0\0\0".to_vec(),
            Curve::Gamma(g) => {
                let mut v = b"curv\0\0\0\0\0\0\0\x01".to_vec();
                v.extend_from_slice(
                    &((g * 256.0).round().clamp(0.0, 65535.0) as u16).to_be_bytes(),
                );
                v
            }
            Curve::Sampled(t) => {
                let mut v = b"curv\0\0\0\0".to_vec();
                v.extend_from_slice(&(t.len() as u32).to_be_bytes());
                for x in t {
                    v.extend_from_slice(
                        &((x.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes(),
                    );
                }
                v
            }
            Curve::Parametric { function, params } => {
                let n = super::curve::parametric_param_count(*function).unwrap_or(1);
                let mut v = b"para\0\0\0\0".to_vec();
                v.extend_from_slice(&function.to_be_bytes());
                v.extend_from_slice(&[0, 0]);
                for p in params.iter().take(n) {
                    put_s15f16(&mut v, *p);
                }
                v
            }
        }
    }

    /// Encode a `multiLocalizedUnicodeType` element holding `text` as a
    /// single en/US record (§10.15).
    pub fn mluc_tag(text: &str) -> Vec<u8> {
        let units: Vec<u16> = text.encode_utf16().collect();
        let mut v = b"mluc\0\0\0\0".to_vec();
        v.extend_from_slice(&1u32.to_be_bytes());
        v.extend_from_slice(&12u32.to_be_bytes());
        v.extend_from_slice(b"enUS");
        v.extend_from_slice(&((units.len() * 2) as u32).to_be_bytes());
        v.extend_from_slice(&28u32.to_be_bytes());
        for u in units {
            v.extend_from_slice(&u.to_be_bytes());
        }
        v
    }

    /// Encode a `lutAToBType` (`a_to_b = true`) or `lutBToAType`
    /// element that holds only "B" curves (the permitted "B" element
    /// combination of §10.12.1 / §10.13.1), one per channel.
    pub fn b_curves_lut_tag(a_to_b: bool, curves: &[Curve]) -> Vec<u8> {
        let mut v = if a_to_b { b"mAB " } else { b"mBA " }.to_vec();
        v.extend_from_slice(&[0, 0, 0, 0]);
        let n = curves.len() as u8;
        v.extend_from_slice(&[n, n, 0, 0]);
        v.extend_from_slice(&32u32.to_be_bytes());
        v.extend_from_slice(&[0u8; 16]);
        for c in curves {
            v.extend_from_slice(&Self::curve_tag(c));
            while !v.len().is_multiple_of(4) {
                v.push(0);
            }
        }
        v
    }

    /// Serialise the profile.
    pub fn build(&self) -> Vec<u8> {
        let n = self.tags.len();
        let table_end = 128 + 4 + 12 * n;
        let mut body: Vec<u8> = Vec::new();
        let mut entries: Vec<(u32, u32, u32)> = Vec::with_capacity(n);
        let mut placed: Vec<(usize, u32)> = Vec::new(); // (tag index, offset)
        for (i, (s, data)) in self.tags.iter().enumerate() {
            let shared = placed
                .iter()
                .find(|(j, _)| self.tags[*j].1 == *data)
                .map(|(_, o)| *o);
            let off = match shared {
                Some(o) => o,
                None => {
                    let o = (table_end + body.len()) as u32;
                    body.extend_from_slice(data);
                    while !body.len().is_multiple_of(4) {
                        body.push(0);
                    }
                    placed.push((i, o));
                    o
                }
            };
            entries.push((*s, off, data.len() as u32));
        }
        let total = table_end + body.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&[0; 4]); // preferred CMM
        out.extend_from_slice(&[self.version.0, self.version.1, 0, 0]);
        out.extend_from_slice(&self.class.signature().to_be_bytes());
        out.extend_from_slice(&self.color_space.signature().to_be_bytes());
        out.extend_from_slice(&self.pcs.signature().to_be_bytes());
        out.extend_from_slice(&[0; 12]); // date
        out.extend_from_slice(b"acsp");
        out.extend_from_slice(&[0; 4]); // platform
        out.extend_from_slice(&[0; 4]); // flags
        out.extend_from_slice(&[0; 8]); // manufacturer, model
        out.extend_from_slice(&[0; 8]); // attributes
        out.extend_from_slice(&self.intent.value().to_be_bytes());
        for x in PCS_WHITE {
            put_s15f16(&mut out, x);
        }
        out.extend_from_slice(&[0; 4]); // creator
        out.extend_from_slice(&[0; 16]); // profile ID
        out.extend_from_slice(&[0; 28]); // reserved
        debug_assert_eq!(out.len(), 128);
        out.extend_from_slice(&(n as u32).to_be_bytes());
        for (s, o, l) in entries {
            out.extend_from_slice(&s.to_be_bytes());
            out.extend_from_slice(&o.to_be_bytes());
            out.extend_from_slice(&l.to_be_bytes());
        }
        out.extend_from_slice(&body);
        out
    }

    /// Serialise and parse back.
    pub fn build_profile(&self) -> Result<IccProfile, IccError> {
        IccProfile::parse(&self.build())
    }
}

/// A version-4 matrix/TRC display profile for an RGB space with the
/// given primaries and tone curve. The colorants are the space's
/// RGB → XYZ matrix adapted to the PCS white by the linearised Bradford
/// transform (Annex E), which is also stored as `chad`; `wtpt` is the
/// PCS white (§9.2.36 for displays).
fn matrix_trc_profile(desc: &str, primaries: &RgbPrimaries, trc: &Curve) -> IccProfile {
    let white = primaries.white.to_xyz();
    let chad = bradford_adaptation(white, PCS_WHITE);
    let m = mul3(&chad, &primaries.rgb_to_xyz());
    IccProfileBuilder::new(
        ProfileClass::Display,
        DataColorSpace::Rgb,
        DataColorSpace::Xyz,
    )
    .with_text(sig(b"desc"), desc)
    .with_text(sig(b"cprt"), "No copyright, use freely")
    .with_xyz(sig(b"wtpt"), PCS_WHITE)
    .with_matrix(sig(b"chad"), chad)
    .with_xyz(sig(b"rXYZ"), [m[0][0], m[1][0], m[2][0]])
    .with_xyz(sig(b"gXYZ"), [m[0][1], m[1][1], m[2][1]])
    .with_xyz(sig(b"bXYZ"), [m[0][2], m[1][2], m[2][2]])
    .with_curve(sig(b"rTRC"), trc)
    .with_curve(sig(b"gTRC"), trc)
    .with_curve(sig(b"bTRC"), trc)
    .build_profile()
    .expect("synthesized profile parses")
}

impl IccProfile {
    /// A synthesized sRGB-shaped display profile: BT.709 primaries, D65
    /// white, the sRGB tone curve as `para` type 3.
    pub fn srgb() -> Self {
        matrix_trc_profile("sRGB (synthesized)", &RgbPrimaries::BT709, &Curve::srgb())
    }

    /// A synthesized Display P3 profile (P3 primaries, D65, sRGB curve).
    pub fn display_p3() -> Self {
        matrix_trc_profile(
            "Display P3 (synthesized)",
            &RgbPrimaries::DISPLAY_P3,
            &Curve::srgb(),
        )
    }

    /// A synthesized Adobe RGB (1998)-shaped profile (gamma 563/256).
    pub fn adobe_rgb() -> Self {
        matrix_trc_profile(
            "Adobe RGB (1998) shape (synthesized)",
            &RgbPrimaries::ADOBE_RGB,
            &Curve::Gamma(563.0 / 256.0),
        )
    }

    /// A synthesized BT.2020 profile whose curve is the inverse of the
    /// BT.2020-2 Table 4 OETF with the 12-bit-system constants
    /// (`α = 1.0993`, `β = 0.0181`), as `para` type 3. (The 10-bit
    /// constants leave the two segments 0.3 % apart at `β`, which makes
    /// the stored curve non-monotonic.)
    pub fn rec2020() -> Self {
        let (a, b) = (1.0993, 0.0181);
        let curve = Curve::Parametric {
            function: 3,
            params: [
                1.0 / 0.45,
                1.0 / a,
                (a - 1.0) / a,
                1.0 / 4.5,
                4.5 * b,
                0.0,
                0.0,
            ],
        };
        matrix_trc_profile("BT.2020 (synthesized)", &RgbPrimaries::BT2020, &curve)
    }

    /// A synthesized ProPhoto RGB (ROMM) profile: CSS Color 4 §10.7
    /// primaries and curve (gamma 1.8, linear below 16/512).
    pub fn prophoto() -> Self {
        let curve = Curve::Parametric {
            function: 3,
            params: [1.8, 1.0, 0.0, 1.0 / 16.0, 16.0 / 512.0, 0.0, 0.0],
        };
        matrix_trc_profile(
            "ProPhoto RGB (synthesized)",
            &RgbPrimaries::PROPHOTO,
            &curve,
        )
    }

    /// A synthesized monochrome display profile with the given
    /// `grayTRCTag` curve (Annex F.2).
    pub fn gray(curve: &Curve) -> Self {
        IccProfileBuilder::new(
            ProfileClass::Display,
            DataColorSpace::Gray,
            DataColorSpace::Xyz,
        )
        .with_text(sig(b"desc"), "Gray (synthesized)")
        .with_text(sig(b"cprt"), "No copyright, use freely")
        .with_xyz(sig(b"wtpt"), PCS_WHITE)
        .with_curve(sig(b"kTRC"), curve)
        .build_profile()
        .expect("synthesized profile parses")
    }

    /// A ColorSpace-class CIELAB (D50) profile whose AToB0 / BToA0 are
    /// identities. Device values are PCSLAB in the Table 12/13
    /// normalisation: `L* / 100`, `(a* + 128) / 255`, `(b* + 128) / 255`.
    pub fn lab_d50() -> Self {
        let id = [Curve::Identity, Curve::Identity, Curve::Identity];
        IccProfileBuilder::new(
            ProfileClass::ColorSpace,
            DataColorSpace::Lab,
            DataColorSpace::Lab,
        )
        .with_text(sig(b"desc"), "CIELAB D50 (synthesized)")
        .with_text(sig(b"cprt"), "No copyright, use freely")
        .with_xyz(sig(b"wtpt"), PCS_WHITE)
        .with_tag(sig(b"A2B0"), IccProfileBuilder::b_curves_lut_tag(true, &id))
        .with_tag(
            sig(b"B2A0"),
            IccProfileBuilder::b_curves_lut_tag(false, &id),
        )
        .build_profile()
        .expect("synthesized profile parses")
    }
}
