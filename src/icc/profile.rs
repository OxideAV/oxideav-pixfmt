//! ICC profile parsing: the 128-byte header (ICC.1:2022 §7.2), the tag
//! table (§7.3) and the colour-transform tag types of Clause 10.

use super::IccError;
use super::curve::{Curve, parametric_param_count};
use super::lut::{Clut, Lut, LutKind, Stage};

/// Profile/device class (ICC.1:2022 §7.2.5, Table 18).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ProfileClass {
    /// `scnr` — input device.
    Input,
    /// `mntr` — display device.
    Display,
    /// `prtr` — output device.
    Output,
    /// `link` — DeviceLink.
    DeviceLink,
    /// `spac` — ColorSpace.
    ColorSpace,
    /// `abst` — Abstract.
    Abstract,
    /// `nmcl` — NamedColor.
    NamedColor,
    /// Any other signature.
    Other(u32),
}

impl ProfileClass {
    pub(crate) fn from_sig(s: u32) -> Self {
        match &s.to_be_bytes() {
            b"scnr" => Self::Input,
            b"mntr" => Self::Display,
            b"prtr" => Self::Output,
            b"link" => Self::DeviceLink,
            b"spac" => Self::ColorSpace,
            b"abst" => Self::Abstract,
            b"nmcl" => Self::NamedColor,
            _ => Self::Other(s),
        }
    }

    /// The four-byte signature.
    pub fn signature(self) -> u32 {
        u32::from_be_bytes(*match self {
            Self::Input => b"scnr",
            Self::Display => b"mntr",
            Self::Output => b"prtr",
            Self::DeviceLink => b"link",
            Self::ColorSpace => b"spac",
            Self::Abstract => b"abst",
            Self::NamedColor => b"nmcl",
            Self::Other(s) => return s,
        })
    }
}

/// Data colour space / PCS signature (ICC.1:2022 §7.2.6, Table 19).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DataColorSpace {
    /// `XYZ ` — nCIEXYZ or PCSXYZ.
    Xyz,
    /// `Lab ` — CIELAB or PCSLAB.
    Lab,
    /// `Luv `.
    Luv,
    /// `YCbr`.
    YCbCr,
    /// `Yxy `.
    Yxy,
    /// `RGB `.
    Rgb,
    /// `GRAY`.
    Gray,
    /// `HSV `.
    Hsv,
    /// `HLS `.
    Hls,
    /// `CMYK`.
    Cmyk,
    /// `CMY `.
    Cmy,
    /// `2CLR` … `FCLR` — the generic n-colour spaces (n = 2..=15).
    Color(u8),
    /// Any other signature.
    Other(u32),
}

impl DataColorSpace {
    pub(crate) fn from_sig(s: u32) -> Self {
        let b = s.to_be_bytes();
        match &b {
            b"XYZ " => Self::Xyz,
            b"Lab " => Self::Lab,
            b"Luv " => Self::Luv,
            b"YCbr" => Self::YCbCr,
            b"Yxy " => Self::Yxy,
            b"RGB " => Self::Rgb,
            b"GRAY" => Self::Gray,
            b"HSV " => Self::Hsv,
            b"HLS " => Self::Hls,
            b"CMYK" => Self::Cmyk,
            b"CMY " => Self::Cmy,
            [d, b'C', b'L', b'R'] => match d {
                b'2'..=b'9' => Self::Color(d - b'0'),
                b'A'..=b'F' => Self::Color(d - b'A' + 10),
                _ => Self::Other(s),
            },
            _ => Self::Other(s),
        }
    }

    /// The four-byte signature.
    pub fn signature(self) -> u32 {
        let b: [u8; 4] = match self {
            Self::Xyz => *b"XYZ ",
            Self::Lab => *b"Lab ",
            Self::Luv => *b"Luv ",
            Self::YCbCr => *b"YCbr",
            Self::Yxy => *b"Yxy ",
            Self::Rgb => *b"RGB ",
            Self::Gray => *b"GRAY",
            Self::Hsv => *b"HSV ",
            Self::Hls => *b"HLS ",
            Self::Cmyk => *b"CMYK",
            Self::Cmy => *b"CMY ",
            Self::Color(n) => {
                let d = if n < 10 { b'0' + n } else { b'A' + n - 10 };
                [d, b'C', b'L', b'R']
            }
            Self::Other(s) => return s,
        };
        u32::from_be_bytes(b)
    }

    /// Number of channels (Table 41's channel assignment); `None` for an
    /// unknown signature.
    pub fn channels(self) -> Option<usize> {
        Some(match self {
            Self::Gray => 1,
            Self::Cmyk => 4,
            Self::Color(n) => n as usize,
            Self::Other(_) => return None,
            _ => 3,
        })
    }
}

/// Rendering intent (ICC.1:2022 §7.2.15, Table 23).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum RenderingIntent {
    /// 0 — perceptual.
    #[default]
    Perceptual,
    /// 1 — media-relative colorimetric.
    RelativeColorimetric,
    /// 2 — saturation.
    Saturation,
    /// 3 — ICC-absolute colorimetric.
    AbsoluteColorimetric,
}

impl RenderingIntent {
    /// Table 23 value.
    pub fn value(self) -> u32 {
        match self {
            Self::Perceptual => 0,
            Self::RelativeColorimetric => 1,
            Self::Saturation => 2,
            Self::AbsoluteColorimetric => 3,
        }
    }

    /// From a Table 23 value.
    pub fn from_value(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Perceptual,
            1 => Self::RelativeColorimetric,
            2 => Self::Saturation,
            3 => Self::AbsoluteColorimetric,
            _ => return None,
        })
    }

    /// Index of the AToBx / BToAx tag carrying this intent (Table 25:
    /// ICC-absolute uses the media-relative tables plus white-point
    /// scaling, §6.2.3).
    pub(crate) fn table_index(self) -> usize {
        match self {
            Self::Perceptual => 0,
            Self::RelativeColorimetric | Self::AbsoluteColorimetric => 1,
            Self::Saturation => 2,
        }
    }
}

/// The 128-byte profile header (ICC.1:2022 §7.2).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct IccHeader {
    /// Bytes 0–3: profile size.
    pub size: u32,
    /// Bytes 4–7: preferred CMM signature (0 = none).
    pub cmm: u32,
    /// Bytes 8–9: version as (major, minor, bug-fix).
    pub version: (u8, u8, u8),
    /// Bytes 12–15.
    pub class: ProfileClass,
    /// Bytes 16–19: data colour space (A side).
    pub color_space: DataColorSpace,
    /// Bytes 20–23: PCS (B side).
    pub pcs: DataColorSpace,
    /// Bytes 24–35: creation date (year, month, day, hour, minute, second).
    pub date: [u16; 6],
    /// Bytes 40–43: primary platform signature.
    pub platform: u32,
    /// Bytes 44–47: profile flags.
    pub flags: u32,
    /// Bytes 48–51: device manufacturer.
    pub manufacturer: u32,
    /// Bytes 52–55: device model.
    pub model: u32,
    /// Bytes 56–63: device attributes.
    pub attributes: u64,
    /// Bytes 64–67: rendering intent (low 16 bits; Table 23).
    pub rendering_intent: u32,
    /// Bytes 68–79: PCS illuminant (nCIEXYZ).
    pub illuminant: [f64; 3],
    /// Bytes 80–83: profile creator.
    pub creator: u32,
    /// Bytes 84–99: profile ID (MD5 digest, or zero).
    pub id: [u8; 16],
}

/// One entry of the tag table (§7.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TagEntry {
    /// Tag signature.
    pub signature: u32,
    /// Offset from the start of the profile.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
}

/// A parsed ICC profile (v2 or v4).
///
/// [`IccProfile::parse`] reads the header and the tag table and decodes
/// every tag the colour transforms use: `wtpt`, `chad`, `rXYZ` / `gXYZ`
/// / `bXYZ`, `rTRC` / `gTRC` / `bTRC`, `kTRC`, `A2B0..2` and `B2A0..2`
/// (`mft1`, `mft2`, `mAB `, `mBA `). Any other tag stays available as
/// raw bytes through [`IccProfile::tag_data`].
#[derive(Clone, Debug)]
pub struct IccProfile {
    pub(crate) header: IccHeader,
    pub(crate) tags: Vec<TagEntry>,
    pub(crate) raw: Vec<u8>,
    pub(crate) white_point: Option<[f64; 3]>,
    pub(crate) chad: Option<[[f64; 3]; 3]>,
    pub(crate) colorants: Option<[[f64; 3]; 3]>,
    pub(crate) trc: Option<[Curve; 3]>,
    pub(crate) gray_trc: Option<Curve>,
    pub(crate) a2b: [Option<Lut>; 3],
    pub(crate) b2a: [Option<Lut>; 3],
}

// --------------------------------------------------------------------
// Byte readers

fn be16(b: &[u8], o: usize) -> Result<u16, IccError> {
    b.get(o..o + 2)
        .map(|s| u16::from_be_bytes([s[0], s[1]]))
        .ok_or(IccError::Truncated)
}

fn be32(b: &[u8], o: usize) -> Result<u32, IccError> {
    b.get(o..o + 4)
        .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or(IccError::Truncated)
}

/// s15Fixed16Number (§4.6).
fn s15f16(b: &[u8], o: usize) -> Result<f64, IccError> {
    Ok(be32(b, o)? as i32 as f64 / 65536.0)
}

fn xyz_number(b: &[u8], o: usize) -> Result<[f64; 3], IccError> {
    Ok([s15f16(b, o)?, s15f16(b, o + 4)?, s15f16(b, o + 8)?])
}

pub(crate) const fn sig(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

/// Upper bound on any single CLUT's sample count, so a hostile header
/// cannot request an enormous allocation (the data must also be present
/// in the tag, which bounds it again).
const MAX_CLUT_SAMPLES: usize = 1 << 26;

impl IccProfile {
    /// Parse a profile from its bytes (an `.icc` / `.icm` file or an
    /// embedded payload).
    pub fn parse(bytes: &[u8]) -> Result<Self, IccError> {
        if bytes.len() < 132 {
            return Err(IccError::Truncated);
        }
        if &bytes[36..40] != b"acsp" {
            return Err(IccError::BadSignature);
        }
        let size = be32(bytes, 0)?;
        // Embedded payloads are sometimes padded; a declared size beyond
        // the buffer is truncation.
        if (size as usize) > bytes.len() {
            return Err(IccError::Truncated);
        }
        let data = if size >= 132 {
            &bytes[..size as usize]
        } else {
            bytes
        };
        let mut date = [0u16; 6];
        for (i, d) in date.iter_mut().enumerate() {
            *d = be16(data, 24 + 2 * i)?;
        }
        let mut id = [0u8; 16];
        id.copy_from_slice(&data[84..100]);
        let header = IccHeader {
            size,
            cmm: be32(data, 4)?,
            version: (data[8], data[9] >> 4, data[9] & 0x0F),
            class: ProfileClass::from_sig(be32(data, 12)?),
            color_space: DataColorSpace::from_sig(be32(data, 16)?),
            pcs: DataColorSpace::from_sig(be32(data, 20)?),
            date,
            platform: be32(data, 40)?,
            flags: be32(data, 44)?,
            manufacturer: be32(data, 48)?,
            model: be32(data, 52)?,
            attributes: ((be32(data, 56)? as u64) << 32) | be32(data, 60)? as u64,
            rendering_intent: be32(data, 64)? & 0xFFFF,
            illuminant: xyz_number(data, 68)?,
            creator: be32(data, 80)?,
            id,
        };
        let count = be32(data, 128)? as usize;
        if count > (data.len() - 132) / 12 {
            return Err(IccError::Truncated);
        }
        let mut tags = Vec::with_capacity(count);
        for i in 0..count {
            let o = 132 + 12 * i;
            let e = TagEntry {
                signature: be32(data, o)?,
                offset: be32(data, o + 4)?,
                size: be32(data, o + 8)?,
            };
            let end = (e.offset as u64) + (e.size as u64);
            if end > data.len() as u64 || e.size < 8 {
                return Err(IccError::Malformed("tag outside the profile"));
            }
            tags.push(e);
        }
        let mut p = IccProfile {
            header,
            tags,
            raw: data.to_vec(),
            white_point: None,
            chad: None,
            colorants: None,
            trc: None,
            gray_trc: None,
            a2b: [None, None, None],
            b2a: [None, None, None],
        };
        p.decode_tags()?;
        Ok(p)
    }

    fn decode_tags(&mut self) -> Result<(), IccError> {
        if let Some(t) = self.tag_data(sig(b"wtpt")) {
            self.white_point = Some(parse_xyz_tag(t)?);
        }
        if let Some(t) = self.tag_data(sig(b"chad")) {
            self.chad = Some(parse_sf32_matrix(t)?);
        }
        let cols = [sig(b"rXYZ"), sig(b"gXYZ"), sig(b"bXYZ")];
        if cols.iter().all(|s| self.tag_data(*s).is_some()) {
            let mut m = [[0.0; 3]; 3];
            for (c, s) in cols.iter().enumerate() {
                let v = parse_xyz_tag(self.tag_data(*s).unwrap())?;
                for r in 0..3 {
                    m[r][c] = v[r];
                }
            }
            self.colorants = Some(m);
        }
        let trcs = [sig(b"rTRC"), sig(b"gTRC"), sig(b"bTRC")];
        if trcs.iter().all(|s| self.tag_data(*s).is_some()) {
            let r = parse_curve(self.tag_data(trcs[0]).unwrap())?.0;
            let g = parse_curve(self.tag_data(trcs[1]).unwrap())?.0;
            let b = parse_curve(self.tag_data(trcs[2]).unwrap())?.0;
            self.trc = Some([r, g, b]);
        }
        if let Some(t) = self.tag_data(sig(b"kTRC")) {
            self.gray_trc = Some(parse_curve(t)?.0);
        }
        let a2b = [sig(b"A2B0"), sig(b"A2B1"), sig(b"A2B2")];
        let b2a = [sig(b"B2A0"), sig(b"B2A1"), sig(b"B2A2")];
        for i in 0..3 {
            if let Some(t) = self.tag_data(a2b[i]) {
                self.a2b[i] = Some(parse_lut(
                    t,
                    self.header.color_space == DataColorSpace::Xyz,
                )?);
            }
            if let Some(t) = self.tag_data(b2a[i]) {
                self.b2a[i] = Some(parse_lut(t, self.header.pcs == DataColorSpace::Xyz)?);
            }
        }
        Ok(())
    }

    /// The decoded header.
    pub fn header(&self) -> &IccHeader {
        &self.header
    }

    /// The tag table, in file order.
    pub fn tags(&self) -> &[TagEntry] {
        &self.tags
    }

    /// The raw bytes of the profile (as parsed, trimmed to the declared
    /// size).
    pub fn as_bytes(&self) -> &[u8] {
        &self.raw
    }

    /// The raw tag data element for `signature` (type signature
    /// included), if present.
    pub fn tag_data(&self, signature: u32) -> Option<&[u8]> {
        let e = self.tags.iter().find(|e| e.signature == signature)?;
        self.raw
            .get(e.offset as usize..(e.offset as usize + e.size as usize))
    }

    /// `mediaWhitePointTag` (§9.2.36), nCIEXYZ.
    pub fn media_white_point(&self) -> Option<[f64; 3]> {
        self.white_point
    }

    /// `chromaticAdaptationTag` (§9.2.15), row-major.
    pub fn chromatic_adaptation(&self) -> Option<[[f64; 3]; 3]> {
        self.chad
    }

    /// The matrix/TRC colorant matrix: columns are `rXYZ`, `gXYZ`,
    /// `bXYZ` (Annex F.3 equation F.6).
    pub fn colorant_matrix(&self) -> Option<[[f64; 3]; 3]> {
        self.colorants
    }

    /// The three matrix/TRC tone curves (`rTRC`, `gTRC`, `bTRC`).
    pub fn rgb_trc(&self) -> Option<&[Curve; 3]> {
        self.trc.as_ref()
    }

    /// The `grayTRCTag` curve (`kTRC`).
    pub fn gray_trc(&self) -> Option<&Curve> {
        self.gray_trc.as_ref()
    }

    /// True when the profile carries a usable matrix/TRC model.
    pub fn is_matrix_shaper(&self) -> bool {
        self.colorants.is_some() && self.trc.is_some()
    }

    /// True when the profile has an `AToBx` table for `intent` (or the
    /// `AToB0` fallback of §8.10.2 c).
    pub fn has_lut_to_pcs(&self, intent: RenderingIntent) -> bool {
        self.a2b[intent.table_index()].is_some() || self.a2b[0].is_some()
    }

    /// True when the profile has a `BToAx` table for `intent` (or the
    /// `BToA0` fallback).
    pub fn has_lut_from_pcs(&self, intent: RenderingIntent) -> bool {
        self.b2a[intent.table_index()].is_some() || self.b2a[0].is_some()
    }

    /// The profile description (`desc`), when it is stored as a
    /// `multiLocalizedUnicodeType` (§10.15, first record) or a
    /// `textType` (§10.24). Version 2's `textDescriptionType` is not
    /// defined by ICC.1:2022, so such a tag yields `None` (the raw bytes
    /// stay available through [`Self::tag_data`]).
    pub fn description(&self) -> Option<String> {
        self.text_tag(sig(b"desc"))
    }

    /// The copyright text (`cprt`), decoded like [`Self::description`].
    pub fn copyright(&self) -> Option<String> {
        self.text_tag(sig(b"cprt"))
    }

    fn text_tag(&self, s: u32) -> Option<String> {
        let t = self.tag_data(s)?;
        match &t[0..4] {
            b"mluc" => {
                let n = be32(t, 8).ok()?;
                let rec = be32(t, 12).ok()? as usize;
                if n == 0 || rec < 12 {
                    return None;
                }
                let len = be32(t, 20).ok()? as usize;
                let off = be32(t, 24).ok()? as usize;
                let s = t.get(off..off.checked_add(len)?)?;
                let units: Vec<u16> = s
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_be_bytes(*c))
                    .collect();
                Some(String::from_utf16_lossy(&units))
            }
            b"text" => {
                let s = &t[8..];
                let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
                Some(String::from_utf8_lossy(&s[..end]).into_owned())
            }
            _ => None,
        }
    }
}

// --------------------------------------------------------------------
// Tag type decoders

fn check_type(t: &[u8], want: &[u8; 4]) -> Result<(), IccError> {
    if t.len() < 8 {
        return Err(IccError::Truncated);
    }
    if &t[0..4] != want {
        return Err(IccError::Malformed("unexpected tag type"));
    }
    Ok(())
}

/// `XYZType` (§10.31): first XYZNumber.
fn parse_xyz_tag(t: &[u8]) -> Result<[f64; 3], IccError> {
    check_type(t, b"XYZ ")?;
    xyz_number(t, 8)
}

/// `s15Fixed16ArrayType` (§10.22) holding a row-major 3×3 matrix.
fn parse_sf32_matrix(t: &[u8]) -> Result<[[f64; 3]; 3], IccError> {
    check_type(t, b"sf32")?;
    let mut m = [[0.0; 3]; 3];
    for (k, v) in m.iter_mut().flatten().enumerate() {
        *v = s15f16(t, 8 + 4 * k)?;
    }
    Ok(m)
}

/// `curveType` (§10.6) or `parametricCurveType` (§10.18). Returns the
/// curve and its byte length (for packed curve sequences).
pub(crate) fn parse_curve(t: &[u8]) -> Result<(Curve, usize), IccError> {
    if t.len() < 12 {
        return Err(IccError::Truncated);
    }
    match &t[0..4] {
        b"curv" => {
            let n = be32(t, 8)? as usize;
            let len = 12usize
                .checked_add(n.checked_mul(2).ok_or(IccError::Truncated)?)
                .ok_or(IccError::Truncated)?;
            if t.len() < len {
                return Err(IccError::Truncated);
            }
            let c = match n {
                0 => Curve::Identity,
                1 => Curve::Gamma(be16(t, 12)? as f64 / 256.0),
                _ => Curve::Sampled(
                    t[12..len]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| u16::from_be_bytes(*c) as f64 / 65535.0)
                        .collect(),
                ),
            };
            Ok((c, len))
        }
        b"para" => {
            let f = be16(t, 8)?;
            let n =
                parametric_param_count(f).ok_or(IccError::Unsupported("parametric curve type"))?;
            let mut p = [0.0; 7];
            for (k, v) in p.iter_mut().take(n).enumerate() {
                *v = s15f16(t, 12 + 4 * k)?;
            }
            Ok((
                Curve::Parametric {
                    function: f,
                    params: p,
                },
                12 + 4 * n,
            ))
        }
        _ => Err(IccError::Unsupported("curve tag type")),
    }
}

/// Parse `n` curves packed back to back from `t[off..]`, each starting
/// on a 4-byte boundary relative to the tag start (§10.12).
fn parse_curve_seq(t: &[u8], mut off: usize, n: usize) -> Result<Vec<Curve>, IccError> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let (c, len) = parse_curve(t.get(off..).ok_or(IccError::Truncated)?)?;
        v.push(c);
        off += len;
        off = (off + 3) & !3;
    }
    Ok(v)
}

fn clut_samples(grid: &[usize], out_ch: usize) -> Result<usize, IccError> {
    let mut n = out_ch;
    for &g in grid {
        n = n
            .checked_mul(g)
            .ok_or(IccError::Malformed("CLUT too large"))?;
        if n > MAX_CLUT_SAMPLES {
            return Err(IccError::Malformed("CLUT too large"));
        }
    }
    Ok(n)
}

/// lut8 / lut16 / lutAToB / lutBToA. `pcs_xyz_input` says whether the
/// tag's input side is PCSXYZ (only then is the lut8 / lut16 matrix
/// applied, §10.10 / §10.11).
pub(crate) fn parse_lut(t: &[u8], pcs_xyz_input: bool) -> Result<Lut, IccError> {
    if t.len() < 12 {
        return Err(IccError::Truncated);
    }
    match &t[0..4] {
        b"mft1" | b"mft2" => parse_mft(t, pcs_xyz_input),
        b"mAB " => parse_mab(t, true),
        b"mBA " => parse_mab(t, false),
        _ => Err(IccError::Unsupported("LUT tag type")),
    }
}

fn parse_mft(t: &[u8], pcs_xyz_input: bool) -> Result<Lut, IccError> {
    let sixteen = &t[0..4] == b"mft2";
    let i = *t.get(8).ok_or(IccError::Truncated)? as usize;
    let o = *t.get(9).ok_or(IccError::Truncated)? as usize;
    let g = *t.get(10).ok_or(IccError::Truncated)? as usize;
    if i == 0 || o == 0 || i > 15 || o > 15 {
        return Err(IccError::Malformed("LUT channel count"));
    }
    let mut m = [[0.0; 3]; 3];
    for (k, v) in m.iter_mut().flatten().enumerate() {
        *v = s15f16(t, 12 + 4 * k)?;
    }
    let (n_in, n_out, mut off, w) = if sixteen {
        (
            be16(t, 48)? as usize,
            be16(t, 50)? as usize,
            52usize,
            2usize,
        )
    } else {
        (256, 256, 48, 1)
    };
    if n_in < 2 || n_out < 2 {
        return Err(IccError::Malformed("LUT table length"));
    }
    let read = |off: usize, n: usize| -> Result<Vec<f64>, IccError> {
        let bytes = t.get(off..off + n * w).ok_or(IccError::Truncated)?;
        Ok(if w == 2 {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_be_bytes(*c) as f64 / 65535.0)
                .collect()
        } else {
            bytes.iter().map(|&b| b as f64 / 255.0).collect()
        })
    };
    let mut input = Vec::with_capacity(i);
    for _ in 0..i {
        input.push(Curve::Sampled(read(off, n_in)?));
        off += n_in * w;
    }
    let grid = vec![g; i];
    let samples = clut_samples(&grid, o)?;
    let bytes = t.get(off..off + samples * w).ok_or(IccError::Truncated)?;
    let data: Vec<u16> = if w == 2 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_be_bytes(*c))
            .collect()
    } else {
        bytes.iter().map(|&b| b as u16 * 257).collect()
    };
    off += samples * w;
    let mut output = Vec::with_capacity(o);
    for _ in 0..o {
        output.push(Curve::Sampled(read(off, n_out)?));
        off += n_out * w;
    }
    let mut stages = Vec::new();
    let identity = m == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if pcs_xyz_input && i == 3 && !identity {
        stages.push(Stage::Matrix3(m));
    }
    stages.push(Stage::Curves(input));
    if g < 1 {
        return Err(IccError::Malformed("CLUT grid"));
    }
    stages.push(Stage::Clut(Clut {
        grid,
        out_ch: o,
        data,
    }));
    stages.push(Stage::Curves(output));
    Ok(Lut {
        kind: if sixteen {
            LutKind::Lut16
        } else {
            LutKind::Lut8
        },
        in_ch: i,
        out_ch: o,
        stages,
    })
}

fn parse_mab(t: &[u8], a_to_b: bool) -> Result<Lut, IccError> {
    let i = *t.get(8).ok_or(IccError::Truncated)? as usize;
    let o = *t.get(9).ok_or(IccError::Truncated)? as usize;
    if i == 0 || o == 0 || i > 15 || o > 15 {
        return Err(IccError::Malformed("LUT channel count"));
    }
    let off_b = be32(t, 12)? as usize;
    let off_mat = be32(t, 16)? as usize;
    let off_m = be32(t, 20)? as usize;
    let off_clut = be32(t, 24)? as usize;
    let off_a = be32(t, 28)? as usize;
    // Channel counts at each element: AToB runs A(i) → CLUT(i→o) →
    // M(o) → matrix(3→3) → B(o); BToA runs B(i) → matrix → M(i) →
    // CLUT(i→o) → A(o).
    let (b_n, m_n, a_n) = if a_to_b { (o, o, i) } else { (i, i, o) };
    let curves = |off: usize, n: usize| -> Result<Option<Stage>, IccError> {
        if off == 0 {
            Ok(None)
        } else {
            Ok(Some(Stage::Curves(parse_curve_seq(t, off, n)?)))
        }
    };
    let matrix = if off_mat != 0 {
        let mut m = [[0.0; 3]; 3];
        let mut c = [0.0; 3];
        for (k, v) in m.iter_mut().flatten().enumerate() {
            *v = s15f16(t, off_mat + 4 * k)?;
        }
        for (k, v) in c.iter_mut().enumerate() {
            *v = s15f16(t, off_mat + 36 + 4 * k)?;
        }
        Some(Stage::Matrix34(m, c))
    } else {
        None
    };
    let clut = if off_clut != 0 {
        let hdr = t.get(off_clut..off_clut + 20).ok_or(IccError::Truncated)?;
        let grid: Vec<usize> = hdr[..i].iter().map(|&g| g as usize).collect();
        if grid.contains(&0) {
            return Err(IccError::Malformed("CLUT grid"));
        }
        let prec = hdr[16] as usize;
        if prec != 1 && prec != 2 {
            return Err(IccError::Malformed("CLUT precision"));
        }
        let samples = clut_samples(&grid, o)?;
        let start = off_clut + 20;
        let bytes = t
            .get(start..start + samples * prec)
            .ok_or(IccError::Truncated)?;
        let data: Vec<u16> = if prec == 2 {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_be_bytes(*c))
                .collect()
        } else {
            bytes.iter().map(|&b| b as u16 * 257).collect()
        };
        Some(Stage::Clut(Clut {
            grid,
            out_ch: o,
            data,
        }))
    } else {
        None
    };
    if matrix.is_some() && (if a_to_b { o } else { i }) != 3 {
        return Err(IccError::Malformed("matrix needs three channels"));
    }
    if clut.is_none() && i != o {
        return Err(IccError::Malformed(
            "CLUT required when channel counts differ",
        ));
    }
    let mut stages = Vec::new();
    if a_to_b {
        if let Some(s) = curves(off_a, a_n)? {
            stages.push(s);
        }
        if let Some(s) = clut {
            stages.push(s);
        }
        if let Some(s) = curves(off_m, m_n)? {
            stages.push(s);
        }
        if let Some(s) = matrix {
            stages.push(s);
        }
        if let Some(s) = curves(off_b, b_n)? {
            stages.push(s);
        }
    } else {
        if let Some(s) = curves(off_b, b_n)? {
            stages.push(s);
        }
        if let Some(s) = matrix {
            stages.push(s);
        }
        if let Some(s) = curves(off_m, m_n)? {
            stages.push(s);
        }
        if let Some(s) = clut {
            stages.push(s);
        }
        if let Some(s) = curves(off_a, a_n)? {
            stages.push(s);
        }
    }
    if stages.is_empty() {
        return Err(IccError::Malformed("LUT without processing elements"));
    }
    Ok(Lut {
        kind: if a_to_b { LutKind::AToB } else { LutKind::BToA },
        in_ch: i,
        out_ch: o,
        stages,
    })
}
