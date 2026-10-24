//! Profile-to-profile colour transforms (ICC.1:2022 §6.2, §8.10, Annex F).

use super::IccError;
use super::curve::Curve;
use super::lut::Lut;
use super::pcs::{self, PCS_WHITE, PcsKind};
use super::profile::{DataColorSpace, IccProfile, ProfileClass, RenderingIntent};
use crate::mat3::{Mat3, invert3, mul3v};

/// Options for building an [`IccTransform`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct TransformOptions {
    /// Rendering intent (Table 23). Default: perceptual.
    pub intent: RenderingIntent,
}

impl TransformOptions {
    /// Default options (perceptual intent).
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the rendering intent.
    pub fn with_intent(mut self, intent: RenderingIntent) -> Self {
        self.intent = intent;
        self
    }
}

/// Device → PCS half of a transform.
#[derive(Clone, Debug)]
enum ToPcs {
    /// AToBx table; output decoded per the profile's PCS.
    Lut { lut: Lut, pcs: PcsKind },
    /// Annex F.3: `XYZ = M · TRC(device)`.
    Matrix { m: Mat3, curves: [Curve; 3] },
    /// Annex F.2: `connection = grayTRC(device)` scaled by the PCS white.
    Gray { curve: Curve, pcs: PcsKind },
}

/// PCS → device half of a transform.
#[derive(Clone, Debug)]
enum FromPcs {
    Lut {
        lut: Lut,
        pcs: PcsKind,
    },
    /// Annex F.3 inverse (equations F.7–F.16).
    Matrix {
        inv: Mat3,
        curves: [Curve; 3],
    },
    /// Annex F.2 inverse (equation F.2).
    Gray {
        curve: Curve,
        pcs: PcsKind,
    },
}

/// A colour transform from one profile's device space to another's,
/// through the PCS.
///
/// Built with [`IccTransform::new`]; the tag chosen on each side follows
/// the §8.10.2 precedence (the intent's AToBx / BToAx table, then the
/// `x = 0` table, then matrix/TRC or grayTRC). `DToBx` / `BToDx`
/// (multiProcessElementsType) tables are not evaluated, which §8.10.2 a)
/// allows ("except where this tag is not … supported by the CMM").
/// The ICC-absolute intent runs the media-relative tables and rescales
/// the PCS by the ratio of the two `mediaWhitePointTag`s (§6.2.3,
/// equations 1–6).
#[derive(Clone, Debug)]
pub struct IccTransform {
    src: ToPcs,
    dst: FromPcs,
    /// ICC-absolute scaling `src_wtpt / dst_wtpt`, applied in PCSXYZ.
    abs_scale: Option<[f64; 3]>,
    in_ch: usize,
    out_ch: usize,
    in_space: DataColorSpace,
    out_space: DataColorSpace,
}

fn pcs_kind(p: &IccProfile) -> Result<PcsKind, IccError> {
    match p.header.pcs {
        DataColorSpace::Xyz => Ok(PcsKind::Xyz),
        DataColorSpace::Lab => Ok(PcsKind::Lab),
        _ => Err(IccError::Unsupported("PCS other than XYZ / Lab")),
    }
}

fn check_class(p: &IccProfile) -> Result<(), IccError> {
    match p.header.class {
        ProfileClass::DeviceLink => Err(IccError::Unsupported("DeviceLink profile as an endpoint")),
        ProfileClass::NamedColor => Err(IccError::Unsupported("NamedColor profile")),
        _ => Ok(()),
    }
}

fn device_channels(p: &IccProfile) -> Result<usize, IccError> {
    p.header
        .color_space
        .channels()
        .ok_or(IccError::Unsupported("unknown data colour space"))
}

fn pick_lut(tables: &[Option<Lut>; 3], intent: RenderingIntent) -> Option<&Lut> {
    tables[intent.table_index()].as_ref().or(tables[0].as_ref())
}

impl ToPcs {
    fn build(p: &IccProfile, intent: RenderingIntent, ch: usize) -> Result<Self, IccError> {
        let pcs = pcs_kind(p)?;
        if let Some(lut) = pick_lut(&p.a2b, intent) {
            if lut.in_ch != ch || lut.out_ch != 3 {
                return Err(IccError::Malformed("AToB channel counts"));
            }
            return Ok(ToPcs::Lut {
                lut: lut.clone(),
                pcs,
            });
        }
        if ch == 3
            && let (Some(m), Some(trc)) = (p.colorants, p.trc.as_ref())
        {
            return Ok(ToPcs::Matrix {
                m,
                curves: trc.clone(),
            });
        }
        if ch == 1
            && let Some(c) = p.gray_trc.as_ref()
        {
            return Ok(ToPcs::Gray {
                curve: c.clone(),
                pcs,
            });
        }
        Err(IccError::Unsupported(
            "no device-to-PCS transform in profile",
        ))
    }

    /// Returns the PCS value and whether it is Lab (`true`) or XYZ.
    fn eval(&self, d: &[f64]) -> ([f64; 3], PcsKind) {
        match self {
            ToPcs::Lut { lut, pcs } => {
                let mut n = [0.0; 3];
                lut.eval(d, &mut n);
                (pcs::decode_pcs(*pcs, lut.kind, n), *pcs)
            }
            ToPcs::Matrix { m, curves } => {
                let lin = [
                    curves[0].eval(d[0]),
                    curves[1].eval(d[1]),
                    curves[2].eval(d[2]),
                ];
                (mul3v(m, lin), PcsKind::Xyz)
            }
            ToPcs::Gray { curve, pcs } => {
                let v = curve.eval(d[0]);
                match pcs {
                    PcsKind::Xyz => (
                        [v * PCS_WHITE[0], v * PCS_WHITE[1], v * PCS_WHITE[2]],
                        PcsKind::Xyz,
                    ),
                    PcsKind::Lab => ([v * 100.0, 0.0, 0.0], PcsKind::Lab),
                }
            }
        }
    }
}

impl FromPcs {
    fn build(p: &IccProfile, intent: RenderingIntent, ch: usize) -> Result<Self, IccError> {
        let pcs = pcs_kind(p)?;
        if let Some(lut) = pick_lut(&p.b2a, intent) {
            if lut.in_ch != 3 || lut.out_ch != ch {
                return Err(IccError::Malformed("BToA channel counts"));
            }
            return Ok(FromPcs::Lut {
                lut: lut.clone(),
                pcs,
            });
        }
        if ch == 3
            && let (Some(m), Some(trc)) = (p.colorants, p.trc.as_ref())
        {
            let inv = invert3(&m).ok_or(IccError::NotInvertible)?;
            return Ok(FromPcs::Matrix {
                inv,
                curves: trc.clone(),
            });
        }
        if ch == 1
            && let Some(c) = p.gray_trc.as_ref()
        {
            return Ok(FromPcs::Gray {
                curve: c.clone(),
                pcs,
            });
        }
        Err(IccError::Unsupported(
            "no PCS-to-device transform in profile",
        ))
    }

    fn wants(&self) -> PcsKind {
        match self {
            FromPcs::Lut { pcs, .. } | FromPcs::Gray { pcs, .. } => *pcs,
            FromPcs::Matrix { .. } => PcsKind::Xyz,
        }
    }

    fn eval(&self, v: [f64; 3], out: &mut [f64]) {
        match self {
            FromPcs::Lut { lut, pcs } => {
                let mut n = pcs::encode_pcs(*pcs, lut.kind, v);
                for c in &mut n {
                    *c = c.clamp(0.0, 1.0);
                }
                lut.eval(&n, out);
            }
            FromPcs::Matrix { inv, curves } => {
                let lin = mul3v(inv, v);
                for c in 0..3 {
                    // F.8–F.16: clamp the linear value, then invert.
                    out[c] = curves[c].eval_inverse(lin[c].clamp(0.0, 1.0));
                }
            }
            FromPcs::Gray { curve, pcs } => {
                let conn = match pcs {
                    PcsKind::Xyz => v[1] / PCS_WHITE[1],
                    PcsKind::Lab => v[0] / 100.0,
                };
                out[0] = curve.eval_inverse(conn.clamp(0.0, 1.0));
            }
        }
    }
}

/// The media white used for ICC-absolute scaling: the
/// `mediaWhitePointTag`, except for Display-class profiles, whose media
/// white is the PCS illuminant by definition (§9.2.36 "For displays,
/// the values specified shall be those of the PCS illuminant") — version
/// 2 display profiles commonly stored the unadapted display white there
/// instead, which would otherwise turn display white into a tinted
/// colour.
fn absolute_white(p: &IccProfile) -> [f64; 3] {
    if p.header.class == ProfileClass::Display {
        return PCS_WHITE;
    }
    p.white_point.unwrap_or(PCS_WHITE)
}

impl IccTransform {
    /// Build a transform from `src`'s device space to `dst`'s device
    /// space with the given intent.
    pub fn new(
        src: &IccProfile,
        dst: &IccProfile,
        intent: RenderingIntent,
    ) -> Result<Self, IccError> {
        Self::with_options(src, dst, TransformOptions::new().with_intent(intent))
    }

    /// Build a transform with explicit [`TransformOptions`].
    pub fn with_options(
        src: &IccProfile,
        dst: &IccProfile,
        opts: TransformOptions,
    ) -> Result<Self, IccError> {
        check_class(src)?;
        check_class(dst)?;
        let in_ch = device_channels(src)?;
        let out_ch = device_channels(dst)?;
        let s = ToPcs::build(src, opts.intent, in_ch)?;
        let d = FromPcs::build(dst, opts.intent, out_ch)?;
        let abs_scale = if opts.intent == RenderingIntent::AbsoluteColorimetric {
            let sw = absolute_white(src);
            let dw = absolute_white(dst);
            if dw.iter().any(|&v| v <= 0.0) {
                return Err(IccError::Malformed("media white point"));
            }
            Some([sw[0] / dw[0], sw[1] / dw[1], sw[2] / dw[2]])
        } else {
            None
        };
        Ok(Self {
            src: s,
            dst: d,
            abs_scale,
            in_ch,
            out_ch,
            in_space: src.header.color_space,
            out_space: dst.header.color_space,
        })
    }

    /// Device channels consumed per pixel.
    pub fn input_channels(&self) -> usize {
        self.in_ch
    }

    /// Device channels produced per pixel.
    pub fn output_channels(&self) -> usize {
        self.out_ch
    }

    /// Source data colour space.
    pub fn input_color_space(&self) -> DataColorSpace {
        self.in_space
    }

    /// Destination data colour space.
    pub fn output_color_space(&self) -> DataColorSpace {
        self.out_space
    }

    /// Transform one pixel of normalised device values (`[0, 1]`, e.g.
    /// `code / 65535`). `input` must hold at least
    /// [`Self::input_channels`] values and `output` at least
    /// [`Self::output_channels`].
    pub fn transform_pixel(&self, input: &[f64], output: &mut [f64]) {
        let (v, kind) = self.src.eval(input);
        let want = self.dst.wants();
        let v = match self.abs_scale {
            Some(s) => {
                let xyz = if kind == PcsKind::Lab {
                    pcs::lab_to_xyz(v)
                } else {
                    v
                };
                let xyz = [xyz[0] * s[0], xyz[1] * s[1], xyz[2] * s[2]];
                if want == PcsKind::Lab {
                    pcs::xyz_to_lab(xyz)
                } else {
                    xyz
                }
            }
            None => match (kind, want) {
                (PcsKind::Xyz, PcsKind::Lab) => pcs::xyz_to_lab(v),
                (PcsKind::Lab, PcsKind::Xyz) => pcs::lab_to_xyz(v),
                _ => v,
            },
        };
        self.dst.eval(v, output);
    }

    fn check_len(&self, src: usize, dst: usize) -> Result<usize, IccError> {
        if !src.is_multiple_of(self.in_ch) {
            return Err(IccError::BufferSize);
        }
        let px = src / self.in_ch;
        if dst < px * self.out_ch {
            return Err(IccError::BufferSize);
        }
        Ok(px)
    }

    /// Transform interleaved `f32` pixels (normalised device values).
    pub fn transform_f32(&self, src: &[f32], dst: &mut [f32]) -> Result<(), IccError> {
        let px = self.check_len(src.len(), dst.len())?;
        let mut i = [0f64; 15];
        let mut o = [0f64; 15];
        for p in 0..px {
            for c in 0..self.in_ch {
                i[c] = src[p * self.in_ch + c] as f64;
            }
            self.transform_pixel(&i, &mut o);
            for c in 0..self.out_ch {
                dst[p * self.out_ch + c] = o[c] as f32;
            }
        }
        Ok(())
    }

    /// Transform interleaved 16-bit pixels (`0..=65535` full scale).
    pub fn transform_u16(&self, src: &[u16], dst: &mut [u16]) -> Result<(), IccError> {
        let px = self.check_len(src.len(), dst.len())?;
        let mut i = [0f64; 15];
        let mut o = [0f64; 15];
        for p in 0..px {
            for c in 0..self.in_ch {
                i[c] = src[p * self.in_ch + c] as f64 / 65535.0;
            }
            self.transform_pixel(&i, &mut o);
            for c in 0..self.out_ch {
                dst[p * self.out_ch + c] = (o[c].clamp(0.0, 1.0) * 65535.0).round() as u16;
            }
        }
        Ok(())
    }

    /// Transform interleaved 8-bit pixels (`0..=255` full scale).
    pub fn transform_u8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), IccError> {
        let px = self.check_len(src.len(), dst.len())?;
        let mut i = [0f64; 15];
        let mut o = [0f64; 15];
        for p in 0..px {
            for c in 0..self.in_ch {
                i[c] = src[p * self.in_ch + c] as f64 / 255.0;
            }
            self.transform_pixel(&i, &mut o);
            for c in 0..self.out_ch {
                dst[p * self.out_ch + c] = (o[c].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        Ok(())
    }
}
