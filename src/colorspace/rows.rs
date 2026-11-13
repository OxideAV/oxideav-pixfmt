//! Buffer conversions between colour models: [`ModelConverter`].
//!
//! Every [`ColorModel`] is `post(M · base(sRGB) + o)`: a base domain
//! (gamma-encoded or linear-light sRGB), an affine map and an optional
//! non-linear step (Lab, hue, …). A converter is planned once — the
//! matrices are derived and inverted at construction, and when both
//! models share a base domain and nothing is clamped the two affine
//! maps fuse into one with no transfer function in between — and then
//! applied to interleaved rows.

use oxideav_core::{Error, ExecutionContext, Result};

use super::extra::{
    CineonLog, ConeResponse, jzazbz_to_xyz_abs, linear_rgb_to_photoycc, luv_to_xyz,
    photoycc_to_linear_rgb, xyz_abs_to_jzazbz, xyz_to_luv,
};
use super::models::{
    ColorModel, LMS_TO_OKLAB, LMS_TO_XYZ, LabWhite, LumaWeights, OHTA_M, OKLAB_TO_LMS, XYZ_TO_LMS,
    d65, d65_to_d50, hcl_to_rgb, hclp_to_rgb, hsi_to_rgb, hsl_to_rgb, hsv_to_rgb, hwb_to_rgb,
    lab_to_lch, lab_to_xyz, lch_to_lab, linear_to_srgb, rgb_to_hcl, rgb_to_hsi, rgb_to_hsl,
    rgb_to_hsv, rgb_to_hwb, srgb_to_linear, srgb_to_xyz_m, xyy_to_xyz, xyz_to_lab, xyz_to_xyy,
    ydbdr_matrix, yiq_matrix, yuv_matrix,
};
use crate::mat3::{Mat3, invert3, mul3, mul3v};

/// What a conversion does with colours outside the sRGB gamut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GamutPolicy {
    /// Keep them: the intermediate sRGB value is not clamped, so float
    /// outputs carry out-of-range values and a round trip is lossless.
    /// Integer outputs still saturate at their code range. The default.
    #[default]
    Preserve,
    /// Clamp the intermediate sRGB value to 0..1 per component before
    /// it enters the destination model.
    Clamp,
}

/// The domain a model's affine map applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    Gamma,
    Linear,
}

/// The non-linear step after the affine map.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Post {
    None,
    Xyy,
    Lab([f64; 3]),
    LchAb([f64; 3]),
    HclAb([f64; 3]),
    Luv([f64; 3]),
    LchUv([f64; 3]),
    HclUv([f64; 3]),
    Oklab,
    Oklch,
    Jz,
    JzCh,
    Hsl,
    Hwb,
    Hsv,
    Hsi,
    Hcl(LumaWeights),
    Hclp(LumaWeights),
    Gray(LumaWeights),
    PhotoYcc,
    Log(CineonLog),
}

#[derive(Clone, Copy, Debug)]
struct Side {
    base: Base,
    m: Mat3,
    off: [f64; 3],
    post: Post,
}

const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn scale_rows(m: &Mat3, s: [f64; 3]) -> Mat3 {
    let mut out = *m;
    for (row, k) in out.iter_mut().zip(s) {
        for v in row.iter_mut() {
            *v *= k;
        }
    }
    out
}

fn ypbpr_matrix(w: LumaWeights) -> Mat3 {
    let y = [w.kr, 1.0 - w.kr - w.kb, w.kb];
    let (sb, sr) = (0.5 / (1.0 - w.kb), 0.5 / (1.0 - w.kr));
    [
        y,
        [-sb * y[0], -sb * y[1], sb * (1.0 - y[2])],
        [sr * (1.0 - y[0]), -sr * y[1], -sr * y[2]],
    ]
}

impl Side {
    fn of(model: ColorModel) -> Self {
        let s2x = srgb_to_xyz_m();
        let lin = |m: Mat3, post: Post| Side {
            base: Base::Linear,
            m,
            off: [0.0; 3],
            post,
        };
        let gam = |m: Mat3, off: [f64; 3], post: Post| Side {
            base: Base::Gamma,
            m,
            off,
            post,
        };
        let xyz_for = |w: LabWhite| match w {
            LabWhite::D65 => (s2x, d65()),
            LabWhite::D50 => (mul3(&d65_to_d50(), &s2x), super::cie::D50.to_xyz()),
        };
        match model {
            ColorModel::Srgb => gam(IDENTITY, [0.0; 3], Post::None),
            ColorModel::LinearSrgb | ColorModel::ScRgb => lin(IDENTITY, Post::None),
            ColorModel::XyzD65 => lin(s2x, Post::None),
            ColorModel::XyzD50 => lin(mul3(&d65_to_d50(), &s2x), Post::None),
            ColorModel::XyY => lin(s2x, Post::Xyy),
            ColorModel::Lab(w) => lin(xyz_for(w).0, Post::Lab(xyz_for(w).1)),
            ColorModel::Lch(w) => lin(xyz_for(w).0, Post::LchAb(xyz_for(w).1)),
            ColorModel::HclAb(w) => lin(xyz_for(w).0, Post::HclAb(xyz_for(w).1)),
            ColorModel::Luv(w) => lin(xyz_for(w).0, Post::Luv(xyz_for(w).1)),
            ColorModel::Lchuv(w) => lin(xyz_for(w).0, Post::LchUv(xyz_for(w).1)),
            ColorModel::HclUv(w) => lin(xyz_for(w).0, Post::HclUv(xyz_for(w).1)),
            ColorModel::Oklab => lin(mul3(&XYZ_TO_LMS, &s2x), Post::Oklab),
            ColorModel::Oklch => lin(mul3(&XYZ_TO_LMS, &s2x), Post::Oklch),
            ColorModel::Jzazbz(n) => lin(scale_rows(&s2x, [n; 3]), Post::Jz),
            ColorModel::JzCzhz(n) => lin(scale_rows(&s2x, [n; 3]), Post::JzCh),
            ColorModel::Hsl => gam(IDENTITY, [0.0; 3], Post::Hsl),
            ColorModel::Hwb => gam(IDENTITY, [0.0; 3], Post::Hwb),
            ColorModel::Hsv => gam(IDENTITY, [0.0; 3], Post::Hsv),
            ColorModel::Hsi => gam(IDENTITY, [0.0; 3], Post::Hsi),
            ColorModel::Hcl(w) => gam(IDENTITY, [0.0; 3], Post::Hcl(w)),
            ColorModel::Hclp(w) => gam(IDENTITY, [0.0; 3], Post::Hclp(w)),
            ColorModel::Cmy => gam(scale_rows(&IDENTITY, [-1.0; 3]), [1.0; 3], Post::None),
            ColorModel::Gray(w) => gam(IDENTITY, [0.0; 3], Post::Gray(w)),
            ColorModel::LinearGray(w) => lin(IDENTITY, Post::Gray(w)),
            ColorModel::YPbPr(w) => gam(ypbpr_matrix(w), [0.0; 3], Post::None),
            ColorModel::YCbCr(w) => gam(ypbpr_matrix(w), [0.0, 0.5, 0.5], Post::None),
            ColorModel::YCbCrLimited(w) => gam(
                scale_rows(
                    &ypbpr_matrix(w),
                    [219.0 / 255.0, 224.0 / 255.0, 224.0 / 255.0],
                ),
                [16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0],
                Post::None,
            ),
            ColorModel::Yiq => gam(yiq_matrix(), [0.0; 3], Post::None),
            ColorModel::Yuv => gam(yuv_matrix(), [0.0; 3], Post::None),
            ColorModel::YDbDr => gam(ydbdr_matrix(), [0.0; 3], Post::None),
            ColorModel::Ohta => gam(OHTA_M, [0.0; 3], Post::None),
            ColorModel::Lms => lin(mul3(&ConeResponse::Cat02.matrix(), &s2x), Post::None),
            ColorModel::LmsOf(c) => lin(mul3(&c.matrix(), &s2x), Post::None),
            ColorModel::PhotoYcc => lin(IDENTITY, Post::PhotoYcc),
            ColorModel::Log(p) => lin(IDENTITY, Post::Log(p)),
        }
    }
}

impl Post {
    /// Affine-domain value → model value.
    #[inline]
    fn forward(&self, u: [f64; 3]) -> [f64; 3] {
        let hue_first = |v: [f64; 3]| [v[2], v[1], v[0]];
        match *self {
            Post::None => u,
            Post::Xyy => xyz_to_xyy(u, d65()),
            Post::Lab(w) => xyz_to_lab(u, w),
            Post::LchAb(w) => lab_to_lch(xyz_to_lab(u, w)),
            Post::HclAb(w) => hue_first(lab_to_lch(xyz_to_lab(u, w))),
            Post::Luv(w) => xyz_to_luv(u, w),
            Post::LchUv(w) => lab_to_lch(xyz_to_luv(u, w)),
            Post::HclUv(w) => hue_first(lab_to_lch(xyz_to_luv(u, w))),
            Post::Oklab => mul3v(&LMS_TO_OKLAB, u.map(f64::cbrt)),
            Post::Oklch => lab_to_lch(mul3v(&LMS_TO_OKLAB, u.map(f64::cbrt))),
            Post::Jz => xyz_abs_to_jzazbz(u),
            Post::JzCh => lab_to_lch(xyz_abs_to_jzazbz(u)),
            Post::Hsl => rgb_to_hsl(u),
            Post::Hwb => rgb_to_hwb(u),
            Post::Hsv => rgb_to_hsv(u),
            Post::Hsi => rgb_to_hsi(u),
            Post::Hcl(w) | Post::Hclp(w) => rgb_to_hcl(u, w),
            Post::Gray(w) => [w.luma(u); 3],
            Post::PhotoYcc => linear_rgb_to_photoycc(u),
            Post::Log(p) => u.map(|c| p.encode(c) / 1023.0),
        }
    }

    /// Model value → affine-domain value.
    #[inline]
    fn inverse(&self, v: [f64; 3]) -> [f64; 3] {
        let lch = |v: [f64; 3]| lch_to_lab([v[2], v[1], v[0]]);
        let cube = |l: [f64; 3]| l.map(|c| c * c * c);
        match *self {
            Post::None => v,
            Post::Xyy => xyy_to_xyz(v),
            Post::Lab(w) => lab_to_xyz(v, w),
            Post::LchAb(w) => lab_to_xyz(lch_to_lab(v), w),
            Post::HclAb(w) => lab_to_xyz(lch(v), w),
            Post::Luv(w) => luv_to_xyz(v, w),
            Post::LchUv(w) => luv_to_xyz(lch_to_lab(v), w),
            Post::HclUv(w) => luv_to_xyz(lch(v), w),
            Post::Oklab => cube(mul3v(&OKLAB_TO_LMS, v)),
            Post::Oklch => cube(mul3v(&OKLAB_TO_LMS, lch_to_lab(v))),
            Post::Jz => jzazbz_to_xyz_abs(v),
            Post::JzCh => jzazbz_to_xyz_abs(lch_to_lab(v)),
            Post::Hsl => hsl_to_rgb(v),
            Post::Hwb => hwb_to_rgb(v),
            Post::Hsv => hsv_to_rgb(v),
            Post::Hsi => hsi_to_rgb(v),
            Post::Hcl(w) => hcl_to_rgb(v, w),
            Post::Hclp(w) => hclp_to_rgb(v, w),
            Post::Gray(_) => [v[0]; 3],
            Post::PhotoYcc => photoycc_to_linear_rgb(v),
            Post::Log(p) => v.map(|c| p.decode(c * 1023.0)),
        }
    }
}

/// The inverse of the affine map `u = M·x` of an Oklab-family side is
/// known in closed form from the publication; every other matrix is
/// inverted numerically.
fn side_inverse(side: &Side) -> Mat3 {
    if matches!(side.post, Post::Oklab | Post::Oklch) {
        let x2s = invert3(&srgb_to_xyz_m()).expect("sRGB matrix is invertible");
        return mul3(&x2s, &LMS_TO_XYZ);
    }
    invert3(&side.m).expect("colour-model matrix is invertible")
}

/// The sRGB transfer step between the two models' base domains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transfer {
    None,
    Linearise,
    Encode,
}

impl ColorModel {
    /// The natural-unit interval of each component that the integer
    /// paths of [`ModelConverter`] map onto the full code range
    /// (`lo` ↔ code 0, `hi` ↔ the maximum code).
    ///
    /// Unit components (RGB, CMY, gray, saturation, lightness, …) use
    /// 0..1 and hues 0..360°. The others: XYZ and LMS 0..2; `L*` 0..100
    /// with `a*`, `b*` −128..127 and `C*` 0..255; `u*` −134..220 and
    /// `v*` −140..122; Oklab and Jzazbz `a`, `b` −0.5..0.5 with chroma
    /// 0..0.5; Y′PbPr and Ohta chroma −0.5..0.5; Y′UV `U` ±0.436, `V`
    /// ±0.615; Y′IQ `I` ±0.5957, `Q` ±0.5226; Y′DbDr ±1.333; scRGB the
    /// IEC 61966-2-2 16-bit rule `code = 8192·v + 4096`, i.e.
    /// −0.5..(65535 − 4096) / 8192.
    pub fn component_ranges(self) -> [(f64, f64); 3] {
        const U: (f64, f64) = (0.0, 1.0);
        const H: (f64, f64) = (0.0, 360.0);
        const S: (f64, f64) = (-0.5, 0.5);
        const AB: (f64, f64) = (-128.0, 127.0);
        match self {
            ColorModel::XyzD65 | ColorModel::XyzD50 | ColorModel::Lms | ColorModel::LmsOf(_) => {
                [(0.0, 2.0); 3]
            }
            ColorModel::ScRgb => [(-0.5, (65535.0 - 4096.0) / 8192.0); 3],
            ColorModel::Lab(_) => [(0.0, 100.0), AB, AB],
            ColorModel::Lch(_) | ColorModel::Lchuv(_) => [(0.0, 100.0), (0.0, 255.0), H],
            ColorModel::HclAb(_) | ColorModel::HclUv(_) => [H, (0.0, 255.0), (0.0, 100.0)],
            ColorModel::Luv(_) => [(0.0, 100.0), (-134.0, 220.0), (-140.0, 122.0)],
            ColorModel::Oklab | ColorModel::Jzazbz(_) => [U, S, S],
            ColorModel::Oklch | ColorModel::JzCzhz(_) => [U, (0.0, 0.5), H],
            ColorModel::Hsl
            | ColorModel::Hwb
            | ColorModel::Hsv
            | ColorModel::Hsi
            | ColorModel::Hcl(_)
            | ColorModel::Hclp(_) => [H, U, U],
            ColorModel::YPbPr(_) | ColorModel::Ohta => [U, S, S],
            ColorModel::Yuv => [U, (-0.436, 0.436), (-0.615, 0.615)],
            ColorModel::Yiq => [U, (-0.5957, 0.5957), (-0.5226, 0.5226)],
            ColorModel::YDbDr => [U, (-1.333, 1.333), (-1.333, 1.333)],
            ColorModel::Srgb
            | ColorModel::LinearSrgb
            | ColorModel::XyY
            | ColorModel::Cmy
            | ColorModel::Gray(_)
            | ColorModel::LinearGray(_)
            | ColorModel::YCbCr(_)
            | ColorModel::YCbCrLimited(_)
            | ColorModel::PhotoYcc
            | ColorModel::Log(_) => [U; 3],
        }
    }
}

// --------------------------------------------------------------------
// Tabulated sRGB encode for the integer paths.

/// Binades `[2^-(ENC_BINADES), 1)` are each cut into `2^ENC_BITS`
/// equal pieces; the curve is the chord across each piece.
const ENC_BINADES: usize = 9;
const ENC_BITS: usize = 7;

/// Piecewise-linear `linear → sRGB` over `[2^-9, 1)`.
///
/// Within a piece of relative width `2^-7` the chord of `f(x) =
/// 1.055·x^(1/2.4) − 0.055` deviates by at most `h²/8 · max|f″|` =
/// `2^-14 / 8 · 0.2564 · x^(1/2.4)` < 2 × 10⁻⁶ — 0.13 of a 16-bit code.
/// Values below `2^-9` (which includes the whole linear toe, ≤
/// 0.0031308) and outside 0..1 go to the exact formula.
struct EncodeTable {
    /// `(intercept, slope)` per piece: `y = intercept + slope·x`.
    pieces: Vec<(f32, f32)>,
}

impl EncodeTable {
    fn get() -> &'static Self {
        static TABLE: std::sync::OnceLock<EncodeTable> = std::sync::OnceLock::new();
        TABLE.get_or_init(|| {
            let per = 1usize << ENC_BITS;
            let mut pieces = Vec::with_capacity(ENC_BINADES * per);
            for b in 0..ENC_BINADES {
                let lo = 2f64.powi(b as i32 - ENC_BINADES as i32);
                for k in 0..per {
                    let x0 = lo * (1.0 + k as f64 / per as f64);
                    let x1 = lo * (1.0 + (k + 1) as f64 / per as f64);
                    let (y0, y1) = (linear_to_srgb(x0), linear_to_srgb(x1));
                    let slope = (y1 - y0) / (x1 - x0);
                    pieces.push(((y0 - slope * x0) as f32, slope as f32));
                }
            }
            EncodeTable { pieces }
        })
    }

    #[inline]
    fn encode(&self, x: f64) -> f64 {
        let xf = x as f32;
        let lo = 1.0 / (1u32 << ENC_BINADES) as f32;
        if !(lo..1.0).contains(&xf) {
            return linear_to_srgb(x);
        }
        // Exponent and the top mantissa bits select the piece.
        let first = (127 - ENC_BINADES as u32) << ENC_BITS;
        let idx = (xf.to_bits() >> (23 - ENC_BITS)) - first;
        let (a, b) = self.pieces[idx as usize];
        (a + b * xf) as f64
    }
}

// --------------------------------------------------------------------

/// Integer sample types the converter reads and writes.
trait Code: Copy + Send + Sync {
    const MAX: f64;
    fn to_index(self) -> usize;
    fn from_code(v: f64) -> Self;
    /// `srgb_to_linear` of every code, built once per process.
    fn linear_table() -> &'static [f64];
}

fn build_linear_table(max: usize) -> Vec<f64> {
    // The same arithmetic as the per-sample path (`code · (1 / max)`),
    // so both give identical results.
    let scale = 1.0 / max as f64;
    (0..=max)
        .map(|k| srgb_to_linear(k as f64 * scale))
        .collect()
}

impl Code for u8 {
    const MAX: f64 = 255.0;
    #[inline]
    fn to_index(self) -> usize {
        self as usize
    }
    #[inline]
    fn from_code(v: f64) -> Self {
        // `as` saturates and maps NaN to 0.
        (v + 0.5) as u8
    }
    fn linear_table() -> &'static [f64] {
        static T: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
        T.get_or_init(|| build_linear_table(255))
    }
}

impl Code for u16 {
    const MAX: f64 = 65535.0;
    #[inline]
    fn to_index(self) -> usize {
        self as usize
    }
    #[inline]
    fn from_code(v: f64) -> Self {
        (v + 0.5) as u16
    }
    fn linear_table() -> &'static [f64] {
        static T: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
        T.get_or_init(|| build_linear_table(65535))
    }
}

/// A planned conversion between two colour models over interleaved
/// buffers.
///
/// ```
/// use oxideav_pixfmt::colorspace::{ColorModel, LabWhite, ModelConverter};
///
/// let to_lab = ModelConverter::new(ColorModel::Srgb, ColorModel::Lab(LabWhite::D50));
/// let rgb = [1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0];
/// let mut lab = [0f32; 6];
/// to_lab.convert_f32(&rgb, &mut lab).unwrap();
/// assert!((lab[0] - 100.0).abs() < 1e-3 && lab[3].abs() < 1e-3);
/// ```
///
/// Float buffers hold natural units (see [`ColorModel`]) and are exact:
/// the sRGB transfer function is evaluated per sample. Integer buffers
/// map each component's [`ColorModel::component_ranges`] interval onto
/// the full code range, round to nearest and saturate; on those paths a
/// plain sRGB source is linearised through an exact per-code table, and
/// a plain sRGB destination is encoded through a piecewise-linear table
/// whose error is below 2 × 10⁻⁶ (0.13 of a 16-bit code).
///
/// Serial by default; [`Self::with_threads`] / [`Self::with_execution`]
/// split the buffer over scoped threads with identical output.
#[derive(Clone, Debug)]
pub struct ModelConverter {
    from: ColorModel,
    to: ColorModel,
    gamut: GamutPolicy,
    threads: usize,
    channels: usize,
    plan: Plan,
}

#[derive(Clone, Copy, Debug)]
struct Plan {
    /// Model value → (post-inverse) → `front_m·u + front_off`.
    front_post: Post,
    front_m: Mat3,
    front_off: [f64; 3],
    clamp: bool,
    transfer: Transfer,
    /// `back_m·x + back_off` → (post-forward) → model value.
    back_m: Mat3,
    back_off: [f64; 3],
    back_post: Post,
    /// The source is plain gamma-encoded sRGB and gets linearised.
    plain_srgb_in: bool,
    /// The destination is plain gamma-encoded sRGB after an encode.
    plain_srgb_out: bool,
    /// Front and back are both the identity: samples are copied.
    identity: bool,
}

fn is_identity(m: &Mat3, off: &[f64; 3]) -> bool {
    *m == IDENTITY && *off == [0.0; 3]
}

impl Plan {
    fn new(from: ColorModel, to: ColorModel, gamut: GamutPolicy) -> Self {
        let a = Side::of(from);
        let b = Side::of(to);
        let inv_a = side_inverse(&a);
        // x = inv_a·(u − off_a)
        let neg = mul3v(&inv_a, a.off).map(|v| -v);
        let clamp = gamut == GamutPolicy::Clamp;
        let transfer = match (a.base, b.base) {
            (Base::Gamma, Base::Linear) => Transfer::Linearise,
            (Base::Linear, Base::Gamma) => Transfer::Encode,
            _ => Transfer::None,
        };
        let plain =
            |s: &Side| s.base == Base::Gamma && s.post == Post::None && is_identity(&s.m, &s.off);
        let mut plan = Plan {
            front_post: a.post,
            front_m: inv_a,
            front_off: neg,
            clamp,
            transfer,
            back_m: b.m,
            back_off: b.off,
            back_post: b.post,
            plain_srgb_in: plain(&a) && transfer == Transfer::Linearise,
            plain_srgb_out: plain(&b) && transfer == Transfer::Encode,
            identity: false,
        };
        if !clamp && transfer == Transfer::None {
            // Fuse the two affine maps; the back becomes the identity.
            if from == to {
                plan.front_m = IDENTITY;
                plan.front_off = [0.0; 3];
            } else {
                plan.front_m = mul3(&b.m, &inv_a);
                let o = mul3v(&b.m, neg);
                plan.front_off = [o[0] + b.off[0], o[1] + b.off[1], o[2] + b.off[2]];
            }
            plan.back_m = IDENTITY;
            plan.back_off = [0.0; 3];
            plan.identity = from == to;
        }
        plan
    }

    #[inline]
    fn affine(m: &Mat3, off: &[f64; 3], v: [f64; 3]) -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + off[0],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + off[1],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + off[2],
        ]
    }

    /// Model value → base-domain value of the source, clamped if asked.
    #[inline]
    fn front(&self, v: [f64; 3]) -> [f64; 3] {
        let x = Self::affine(&self.front_m, &self.front_off, self.front_post.inverse(v));
        if self.clamp {
            x.map(|c| c.clamp(0.0, 1.0))
        } else {
            x
        }
    }

    #[inline]
    fn transfer(&self, x: [f64; 3]) -> [f64; 3] {
        match self.transfer {
            Transfer::None => x,
            Transfer::Linearise => x.map(srgb_to_linear),
            Transfer::Encode => x.map(linear_to_srgb),
        }
    }

    /// Base-domain value of the destination → model value.
    #[inline]
    fn back(&self, x: [f64; 3]) -> [f64; 3] {
        self.back_post
            .forward(Self::affine(&self.back_m, &self.back_off, x))
    }

    #[inline]
    fn pixel(&self, v: [f64; 3]) -> [f64; 3] {
        self.back(self.transfer(self.front(v)))
    }
}

/// Pixels per worker below which a buffer is not split further.
const MIN_PIXELS_PER_THREAD: usize = 4096;

impl ModelConverter {
    /// Plan the conversion `from → to` (serial, out-of-gamut values
    /// preserved, three interleaved channels).
    pub fn new(from: ColorModel, to: ColorModel) -> Self {
        Self {
            from,
            to,
            gamut: GamutPolicy::Preserve,
            threads: 1,
            channels: 3,
            plan: Plan::new(from, to, GamutPolicy::Preserve),
        }
    }

    /// Choose the out-of-gamut policy.
    pub fn with_gamut(mut self, gamut: GamutPolicy) -> Self {
        self.gamut = gamut;
        self.plan = Plan::new(self.from, self.to, gamut);
        self
    }

    /// Run the buffer conversions on up to `threads` scoped threads
    /// (`0` and `1` both mean serial). The output does not depend on
    /// it.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Take the thread budget from an execution context.
    pub fn with_execution(self, execution: &ExecutionContext) -> Self {
        self.with_threads(execution.threads)
    }

    /// Samples per pixel in the buffers (at least 3; default 3). The
    /// first three are the colour; any others (alpha, padding) are
    /// copied through unchanged. Values below 3 are raised to 3.
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels.max(3);
        self
    }

    /// The source model.
    pub fn from_model(&self) -> ColorModel {
        self.from
    }

    /// The destination model.
    pub fn to_model(&self) -> ColorModel {
        self.to
    }

    /// Convert one colour in natural units.
    pub fn convert_pixel(&self, v: [f64; 3]) -> [f64; 3] {
        self.plan.pixel(v)
    }

    fn check(&self, src: usize, dst: usize) -> Result<usize> {
        if !src.is_multiple_of(self.channels) || dst != src {
            return Err(Error::invalid(format!(
                "colour-model buffers must hold whole {}-channel pixels and match in length \
                 (source {src}, destination {dst})",
                self.channels
            )));
        }
        Ok(src / self.channels)
    }

    /// Split `px` pixels over the thread budget and run `work` on each
    /// part.
    fn drive<S, D>(&self, src: &[S], dst: &mut [D], px: usize, work: impl Fn(&[S], &mut [D]) + Sync)
    where
        S: Sync,
        D: Send,
    {
        let threads = ExecutionContext::with_threads(self.threads)
            .effective_workers(px / MIN_PIXELS_PER_THREAD);
        if threads == 1 {
            work(src, dst);
            return;
        }
        let per = px.div_ceil(threads) * self.channels;
        std::thread::scope(|sc| {
            for (s, d) in src.chunks(per).zip(dst.chunks_mut(per)) {
                let work = &work;
                sc.spawn(move || work(s, d));
            }
        });
    }

    /// Convert interleaved `f64` pixels in natural units.
    pub fn convert_f64(&self, src: &[f64], dst: &mut [f64]) -> Result<()> {
        let px = self.check(src.len(), dst.len())?;
        self.float_rows(src, dst, px, |v| v, |v| v);
        Ok(())
    }

    /// Convert interleaved `f32` pixels in natural units (computed in
    /// `f64`, stored rounded to `f32`).
    pub fn convert_f32(&self, src: &[f32], dst: &mut [f32]) -> Result<()> {
        let px = self.check(src.len(), dst.len())?;
        self.float_rows(src, dst, px, |v| v as f64, |v| v as f32);
        Ok(())
    }

    fn float_rows<T: Copy + Send + Sync>(
        &self,
        src: &[T],
        dst: &mut [T],
        px: usize,
        load: impl Fn(T) -> f64 + Sync,
        store: impl Fn(f64) -> T + Sync,
    ) {
        let ch = self.channels;
        let plan = self.plan;
        if plan.identity {
            dst.copy_from_slice(src);
            return;
        }
        let affine_only = plan.front_post == Post::None
            && plan.back_post == Post::None
            && plan.transfer == Transfer::None
            && !plan.clamp;
        self.drive(src, dst, px, |s, d| {
            if affine_only {
                // One fused 3 × 3 + offset: a branch-free loop.
                for (s, d) in s.chunks_exact(ch).zip(d.chunks_exact_mut(ch)) {
                    let v = [load(s[0]), load(s[1]), load(s[2])];
                    let o = Plan::affine(&plan.front_m, &plan.front_off, v);
                    d[0] = store(o[0]);
                    d[1] = store(o[1]);
                    d[2] = store(o[2]);
                    d[3..].copy_from_slice(&s[3..]);
                }
                return;
            }
            for (s, d) in s.chunks_exact(ch).zip(d.chunks_exact_mut(ch)) {
                let o = plan.pixel([load(s[0]), load(s[1]), load(s[2])]);
                d[0] = store(o[0]);
                d[1] = store(o[1]);
                d[2] = store(o[2]);
                d[3..].copy_from_slice(&s[3..]);
            }
        });
    }

    /// Convert interleaved 8-bit pixels; each component's
    /// [`ColorModel::component_ranges`] interval spans codes 0..=255.
    pub fn convert_u8(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        let px = self.check(src.len(), dst.len())?;
        self.code_rows(src, dst, px);
        Ok(())
    }

    /// Convert interleaved 16-bit pixels; each component's
    /// [`ColorModel::component_ranges`] interval spans codes
    /// 0..=65535.
    pub fn convert_u16(&self, src: &[u16], dst: &mut [u16]) -> Result<()> {
        let px = self.check(src.len(), dst.len())?;
        self.code_rows(src, dst, px);
        Ok(())
    }

    fn code_rows<T: Code>(&self, src: &[T], dst: &mut [T], px: usize) {
        let ch = self.channels;
        let plan = self.plan;
        if plan.identity {
            dst.copy_from_slice(src);
            return;
        }
        let ri = self.from.component_ranges();
        let ro = self.to.component_ranges();
        // code → natural: lo + code·(hi − lo) / MAX; natural → code:
        // (v − lo)·MAX / (hi − lo).
        let in_scale = ri.map(|(lo, hi)| (hi - lo) / T::MAX);
        let in_lo = ri.map(|(lo, _)| lo);
        let out_scale = ro.map(|(lo, hi)| T::MAX / (hi - lo));
        let out_lo = ro.map(|(lo, _)| lo);
        let store = |d: &mut [T], o: [f64; 3]| {
            d[0] = T::from_code((o[0] - out_lo[0]) * out_scale[0]);
            d[1] = T::from_code((o[1] - out_lo[1]) * out_scale[1]);
            d[2] = T::from_code((o[2] - out_lo[2]) * out_scale[2]);
        };
        let load = |s: &[T]| {
            [
                in_lo[0] + s[0].to_index() as f64 * in_scale[0],
                in_lo[1] + s[1].to_index() as f64 * in_scale[1],
                in_lo[2] + s[2].to_index() as f64 * in_scale[2],
            ]
        };

        // Plain sRGB in, linearised: one exact table lookup per sample
        // (the 65536-entry table is only worth building for a buffer
        // of some size).
        let linear_of_code: Option<&'static [f64]> =
            (plan.plain_srgb_in && px * 4 > T::MAX as usize).then(T::linear_table);
        let encode = EncodeTable::get();
        let affine_only = plan.front_post == Post::None
            && plan.back_post == Post::None
            && plan.transfer == Transfer::None
            && !plan.clamp;

        self.drive(src, dst, px, |s, d| {
            let rows = s.chunks_exact(ch).zip(d.chunks_exact_mut(ch));
            if affine_only {
                for (s, d) in rows {
                    store(d, Plan::affine(&plan.front_m, &plan.front_off, load(s)));
                    d[3..].copy_from_slice(&s[3..]);
                }
            } else if let Some(lut) = linear_of_code {
                for (s, d) in rows {
                    let x = [
                        lut[s[0].to_index()],
                        lut[s[1].to_index()],
                        lut[s[2].to_index()],
                    ];
                    store(d, plan.back(x));
                    d[3..].copy_from_slice(&s[3..]);
                }
            } else if plan.plain_srgb_out {
                for (s, d) in rows {
                    let x = plan.front(load(s));
                    store(d, x.map(|c| encode.encode(c)));
                    d[3..].copy_from_slice(&s[3..]);
                }
            } else {
                for (s, d) in rows {
                    store(d, plan.pixel(load(s)));
                    d[3..].copy_from_slice(&s[3..]);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_table_error_is_below_the_documented_bound() {
        let t = EncodeTable::get();
        let mut worst = 0f64;
        let n = if cfg!(miri) { 2_000 } else { 2_000_000 };
        for k in 0..=n {
            let x = k as f64 / n as f64;
            // The table sees the value rounded to f32.
            let exact = linear_to_srgb(x as f32 as f64);
            worst = worst.max((t.encode(x) - exact).abs());
        }
        assert!(worst < 2e-6, "{worst:e}");
        // Outside 0..1 the exact formula is used.
        assert_eq!(t.encode(-0.25), linear_to_srgb(-0.25));
        assert_eq!(t.encode(1.5), linear_to_srgb(1.5));
        assert_eq!(t.encode(1.0), linear_to_srgb(1.0));
    }
}
