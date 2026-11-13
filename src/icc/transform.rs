//! Profile-to-profile colour transforms (ICC.1:2022 §6.2, §8.10, Annex F).

use super::IccError;
use super::curve::Curve;
use super::lut::{ClutInterpolation, Lut, simplex};
use super::mpe::Mpe;
use super::pcs::{self, PCS_WHITE, PcsKind};
use super::profile::{DataColorSpace, IccProfile, ProfileClass, RenderingIntent};
use crate::mat3::{Mat3, invert3, mul3v};

/// Options for building an [`IccTransform`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct TransformOptions {
    /// Rendering intent (Table 23). Default: perceptual.
    pub intent: RenderingIntent,
    /// Worker threads for the buffer transforms (`0` and `1` both mean
    /// serial, the default). The output does not depend on it.
    pub threads: usize,
    /// How the profiles' colour lookup tables are interpolated.
    /// Default: multilinear.
    pub interpolation: ClutInterpolation,
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

    /// Run the buffer transforms on up to `threads` scoped threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Take the thread budget from an execution context.
    pub fn with_execution(self, execution: &oxideav_core::ExecutionContext) -> Self {
        self.with_threads(execution.threads)
    }

    /// Select how the profiles' integer colour lookup tables (`mft1`,
    /// `mft2`, `mAB `, `mBA `) are interpolated.
    pub fn with_interpolation(mut self, interpolation: ClutInterpolation) -> Self {
        self.interpolation = interpolation;
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
    /// DToBx (`mpet`, §10.16): actual PCS values out (§6.3.4.2). With
    /// `absolute`, the tag is DToB3 and already ICC-absolute.
    Mpe {
        mpe: Mpe,
        pcs: PcsKind,
        absolute: bool,
    },
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
    /// BToDx (`mpet`): actual PCS values in, unclipped device values
    /// out. With `absolute`, the tag is BToD3.
    Mpe {
        mpe: Mpe,
        pcs: PcsKind,
        absolute: bool,
    },
}

/// A colour transform from one profile's device space to another's,
/// through the PCS.
///
/// Built with [`IccTransform::new`]; the tag chosen on each side follows
/// the §8.10.2 precedence (the intent's AToBx / BToAx table, then the
/// `x = 0` table, then matrix/TRC or grayTRC), with the intent's own
/// float `DToBx` / `BToDx` (multiProcessElementsType) tag ahead of all
/// of them when it is present and evaluable. The ICC-absolute intent
/// uses DToB3 / BToD3 directly when present; otherwise it runs the
/// media-relative tables and rescales the PCS by the media white over
/// the PCS white (§6.2.3, equations 1–6).
#[derive(Clone, Debug)]
pub struct IccTransform {
    src: ToPcs,
    dst: FromPcs,
    /// A DeviceLink's DToB0 / AToB0, evaluated directly (`src` / `dst`
    /// unused).
    link: Option<Link>,
    /// Abstract / device profiles between `src` and `dst` (chains).
    mid: Vec<Mid>,
    /// ICC-absolute scaling `src_wtpt / dst_wtpt`, applied in PCSXYZ.
    abs_scale: Option<[f64; 3]>,
    in_ch: usize,
    out_ch: usize,
    in_space: DataColorSpace,
    out_space: DataColorSpace,
    threads: usize,
}

/// A profile between the two endpoints of a chain.
#[derive(Clone, Debug)]
enum Mid {
    /// An Abstract profile (§8.8): a PCS → PCS table, the input in the
    /// header's colour space, the output in its PCS.
    Abstract {
        table: Link,
        input: PcsKind,
        output: PcsKind,
    },
    /// A device profile in the middle (e.g. a proofing target): PCS →
    /// device → PCS, with the ICC-absolute rescaling into it.
    Device {
        from: Box<FromPcs>,
        to: Box<ToPcs>,
        scale: Option<[f64; 3]>,
    },
}

/// Bring a PCS value of encoding `kind` to `want`, rescaling it in
/// PCSXYZ by `scale` (ICC-absolute media-white ratios) on the way.
fn connect(v: [f64; 3], kind: PcsKind, want: PcsKind, scale: Option<[f64; 3]>) -> [f64; 3] {
    match scale {
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
    }
}

/// The single table of a DeviceLink.
#[derive(Clone, Debug)]
enum Link {
    Lut(Lut),
    Mpe(Mpe),
    /// A whole transform sampled on a regular grid
    /// ([`IccTransform::precompute`]).
    Baked(Baked),
}

/// A transform tabulated on a regular grid over its input device space,
/// `out_ch` `f32` values per node, the first input channel varying
/// least rapidly; evaluated by simplex interpolation.
#[derive(Clone, Debug)]
struct Baked {
    grid: usize,
    in_ch: usize,
    out_ch: usize,
    data: Vec<f32>,
}

/// Upper bound on the `f32` values a precomputed table may hold (64 MiB).
const MAX_BAKED_VALUES: usize = 1 << 24;

impl Baked {
    /// Interpolate at normalised device values (clamped to `[0, 1]`).
    fn eval(&self, input: &[f64], out: &mut [f64]) {
        let mut x = [0f64; 4];
        for (d, v) in x.iter_mut().zip(input) {
            *d = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        }
        let grid = [self.grid; 4];
        simplex(
            &grid[..self.in_ch],
            self.out_ch,
            &self.data,
            &x[..self.in_ch],
            out,
        );
    }

    /// Interpolate from per-channel cell positions: `cell[c]` is the
    /// flat offset of the channel's lower grid point, `frac[c]` the
    /// fraction above it, `step[c]` the flat distance to its upper
    /// neighbour (0 at the last grid point).
    #[inline]
    fn eval_cell(&self, origin: usize, mut frac: [f32; 4], mut step: [usize; 4], out: &mut [f32]) {
        let n = self.in_ch;
        // Sort the (at most four) channels by descending fraction.
        for i in 1..n {
            let mut j = i;
            while j > 0 && frac[j - 1] < frac[j] {
                frac.swap(j - 1, j);
                step.swap(j - 1, j);
                j -= 1;
            }
        }
        let o = self.out_ch;
        let mut off = origin;
        let mut w = 1.0 - frac[0];
        for (a, &v) in out[..o].iter_mut().zip(&self.data[off..off + o]) {
            *a = w * v;
        }
        for k in 0..n {
            off += step[k];
            w = frac[k] - if k + 1 < n { frac[k + 1] } else { 0.0 };
            for (a, &v) in out[..o].iter_mut().zip(&self.data[off..off + o]) {
                *a += w * v;
            }
        }
    }

    /// The flat stride of input channel `c`.
    fn stride(&self, c: usize) -> usize {
        self.out_ch * self.grid.pow((self.in_ch - 1 - c) as u32)
    }

    /// Cell position of a normalised value on channel `c`:
    /// `(offset, fraction, step)`.
    #[inline]
    fn position(&self, c: usize, v: f32) -> (usize, f32, usize) {
        let g = self.grid;
        let pos = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) } * (g - 1) as f32;
        let i = (pos as usize).min(g - 1);
        let stride = self.stride(c);
        // At the last grid point there is no upper neighbour: the
        // fraction is zero and the step must not leave the table.
        let step = if i + 1 < g { stride } else { 0 };
        (i * stride, pos - i as f32, step)
    }

    /// Run a buffer: `pos(sample, channel)` gives the cell position.
    fn run<S, D>(
        &self,
        src: &[S],
        dst: &mut [D],
        threads: usize,
        pos: impl Fn(S, usize) -> (usize, f32, usize) + Sync,
        quant: impl Fn(f32) -> D + Sync,
    ) where
        S: Copy + Sync,
        D: Send,
    {
        let (ic, oc) = (self.in_ch, self.out_ch);
        let px = src.len() / ic;
        let work = |src: &[S], dst: &mut [D]| {
            let mut o = [0f32; 16];
            for (s, d) in src.chunks_exact(ic).zip(dst.chunks_exact_mut(oc)) {
                let mut origin = 0;
                let mut frac = [0f32; 4];
                let mut step = [0usize; 4];
                for c in 0..ic {
                    let (at, f, st) = pos(s[c], c);
                    origin += at;
                    frac[c] = f;
                    step[c] = st;
                }
                self.eval_cell(origin, frac, step, &mut o);
                for c in 0..oc {
                    d[c] = quant(o[c]);
                }
            }
        };
        let threads = threads.clamp(1, (px / 4096).max(1));
        if threads == 1 {
            work(src, &mut dst[..px * oc]);
            return;
        }
        let per = px.div_ceil(threads);
        std::thread::scope(|sc| {
            for (s, d) in src
                .chunks(per * ic)
                .zip(dst[..px * oc].chunks_mut(per * oc))
            {
                let work = &work;
                sc.spawn(move || work(s, d));
            }
        });
    }

    /// Per-code cell positions for an integer input of `max + 1` codes.
    fn code_positions(&self, max: usize) -> Vec<Vec<(usize, f32, usize)>> {
        (0..self.in_ch)
            .map(|c| {
                (0..=max)
                    .map(|k| {
                        // Exact rational position `k·(g − 1) / max`.
                        let num = k * (self.grid - 1);
                        let i = (num / max).min(self.grid - 1);
                        let stride = self.stride(c);
                        let step = if i + 1 < self.grid { stride } else { 0 };
                        (i * stride, (num - i * max) as f32 / max as f32, step)
                    })
                    .collect()
            })
            .collect()
    }
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
    fn set_interpolation(&mut self, scheme: ClutInterpolation) {
        if let ToPcs::Lut { lut, .. } = self {
            lut.set_interpolation(scheme);
        }
    }

    fn build(p: &IccProfile, intent: RenderingIntent, ch: usize) -> Result<Self, IccError> {
        let pcs = pcs_kind(p)?;
        if let Some(mpe) = &p.d2b[intent.float_table_index()]
            && mpe.in_ch == ch
            && mpe.out_ch == 3
        {
            return Ok(ToPcs::Mpe {
                mpe: mpe.clone(),
                pcs,
                absolute: intent == RenderingIntent::AbsoluteColorimetric,
            });
        }
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

    /// The per-channel curves applied first to the device values.
    fn first_curves(&self) -> Option<&[Curve]> {
        match self {
            ToPcs::Lut { lut, .. } => lut.first_curves(),
            ToPcs::Matrix { curves, .. } => Some(curves),
            ToPcs::Gray { curve, .. } => Some(core::slice::from_ref(curve)),
            ToPcs::Mpe { .. } => None,
        }
    }

    /// Returns the PCS value and its encoding. With `pre`, `d` already
    /// holds the output of [`Self::first_curves`].
    fn eval(&self, d: &[f64], pre: bool) -> ([f64; 3], PcsKind) {
        match self {
            ToPcs::Lut { lut, pcs } => {
                let mut n = [0.0; 3];
                lut.eval_from(d, &mut n, usize::from(pre));
                (pcs::decode_pcs(*pcs, lut.kind, n), *pcs)
            }
            ToPcs::Matrix { m, curves } => {
                let lin = if pre {
                    [d[0], d[1], d[2]]
                } else {
                    [
                        curves[0].eval(d[0]),
                        curves[1].eval(d[1]),
                        curves[2].eval(d[2]),
                    ]
                };
                (mul3v(m, lin), PcsKind::Xyz)
            }
            ToPcs::Gray { curve, pcs } => {
                let v = if pre { d[0] } else { curve.eval(d[0]) };
                match pcs {
                    PcsKind::Xyz => (
                        [v * PCS_WHITE[0], v * PCS_WHITE[1], v * PCS_WHITE[2]],
                        PcsKind::Xyz,
                    ),
                    PcsKind::Lab => ([v * 100.0, 0.0, 0.0], PcsKind::Lab),
                }
            }
            ToPcs::Mpe { mpe, pcs, .. } => {
                let mut n = [0.0; 3];
                mpe.eval(d, &mut n);
                (n, *pcs)
            }
        }
    }
}

impl FromPcs {
    fn set_interpolation(&mut self, scheme: ClutInterpolation) {
        if let FromPcs::Lut { lut, .. } = self {
            lut.set_interpolation(scheme);
        }
    }

    fn build(p: &IccProfile, intent: RenderingIntent, ch: usize) -> Result<Self, IccError> {
        let pcs = pcs_kind(p)?;
        if let Some(mpe) = &p.b2d[intent.float_table_index()]
            && mpe.in_ch == 3
            && mpe.out_ch == ch
        {
            return Ok(FromPcs::Mpe {
                mpe: mpe.clone(),
                pcs,
                absolute: intent == RenderingIntent::AbsoluteColorimetric,
            });
        }
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
            FromPcs::Lut { pcs, .. } | FromPcs::Gray { pcs, .. } | FromPcs::Mpe { pcs, .. } => *pcs,
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
            FromPcs::Mpe { mpe, .. } => mpe.eval(&v, out),
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
        Self::chain(&[src, dst], opts)
    }

    /// Build a transform through a chain of profiles: the first one's
    /// device space to the last one's, through every profile in
    /// between. A middle profile is either an Abstract profile (§8.8;
    /// its DToB0 or AToB0 runs PCS → PCS, §8.10.3) or a device profile,
    /// which the colours pass through (PCS → device → PCS, e.g. to see
    /// a print simulation on a display). Every table follows the
    /// options' intent with the §8.10.2 precedence; under the
    /// ICC-absolute intent each step into a device profile rescales by
    /// the media whites of the device profiles on either side (§6.2.3),
    /// and an Abstract profile works on the values as they arrive.
    ///
    /// Endpoints must be device, ColorSpace or Lab-style profiles (not
    /// Abstract, DeviceLink or NamedColor); a chain of two profiles is
    /// [`IccTransform::with_options`].
    pub fn chain(profiles: &[&IccProfile], opts: TransformOptions) -> Result<Self, IccError> {
        let [first, middle @ .., last] = profiles else {
            return Err(IccError::Unsupported("a chain needs at least two profiles"));
        };
        for p in [first, last] {
            check_class(p)?;
            if p.header.class == ProfileClass::Abstract {
                return Err(IccError::Unsupported("Abstract profile as an endpoint"));
            }
        }
        let intent = opts.intent;
        let absolute = intent == RenderingIntent::AbsoluteColorimetric;
        // The media white a PCS value is relative to, and the scaling
        // into a consumer (a DToB3 / BToD3 side is absolute already).
        let produced = |to: &ToPcs, p: &IccProfile| match to {
            ToPcs::Mpe { absolute: true, .. } => PCS_WHITE,
            _ => absolute_white(p),
        };
        let scale_into = |from: &FromPcs, p: &IccProfile, cur: [f64; 3]| {
            if !absolute {
                return Ok(None);
            }
            let w = match from {
                FromPcs::Mpe { absolute: true, .. } => PCS_WHITE,
                _ => absolute_white(p),
            };
            if w.iter().any(|&v| v <= 0.0) {
                return Err(IccError::Malformed("media white point"));
            }
            let k = [cur[0] / w[0], cur[1] / w[1], cur[2] / w[2]];
            Ok((k != [1.0; 3]).then_some(k))
        };
        let in_ch = device_channels(first)?;
        let out_ch = device_channels(last)?;
        let scheme = opts.interpolation;
        let mut s = ToPcs::build(first, intent, in_ch)?;
        s.set_interpolation(scheme);
        let mut cur = produced(&s, first);
        let mut mid = Vec::with_capacity(middle.len());
        for p in middle {
            match p.header.class {
                ProfileClass::Abstract => {
                    let input = match p.header.color_space {
                        DataColorSpace::Xyz => PcsKind::Xyz,
                        DataColorSpace::Lab => PcsKind::Lab,
                        _ => return Err(IccError::Malformed("Abstract profile input space")),
                    };
                    let output = pcs_kind(p)?;
                    let table = match &p.d2b[0] {
                        Some(m) if m.in_ch == 3 && m.out_ch == 3 => Link::Mpe(m.clone()),
                        _ => {
                            let l = p.a2b[0]
                                .clone()
                                .ok_or(IccError::Unsupported("Abstract profile without AToB0"))?;
                            if l.in_ch != 3 || l.out_ch != 3 {
                                return Err(IccError::Malformed("AToB0 channel counts"));
                            }
                            let mut l = l;
                            l.set_interpolation(scheme);
                            Link::Lut(l)
                        }
                    };
                    mid.push(Mid::Abstract {
                        table,
                        input,
                        output,
                    });
                }
                ProfileClass::DeviceLink => {
                    return Err(IccError::Unsupported("DeviceLink profile inside a chain"));
                }
                _ => {
                    check_class(p)?;
                    let ch = device_channels(p)?;
                    let mut from = FromPcs::build(p, intent, ch)?;
                    let mut to = ToPcs::build(p, intent, ch)?;
                    from.set_interpolation(scheme);
                    to.set_interpolation(scheme);
                    let scale = scale_into(&from, p, cur)?;
                    cur = produced(&to, p);
                    mid.push(Mid::Device {
                        from: Box::new(from),
                        to: Box::new(to),
                        scale,
                    });
                }
            }
        }
        let mut d = FromPcs::build(last, intent, out_ch)?;
        d.set_interpolation(scheme);
        let abs_scale = scale_into(&d, last, cur)?;
        Ok(Self {
            src: s,
            dst: d,
            link: None,
            mid,
            abs_scale,
            in_ch,
            out_ch,
            in_space: first.header.color_space,
            out_space: last.header.color_space,
            threads: opts.threads.max(1),
        })
    }

    /// The single transform of a DeviceLink profile (§8.6): its DToB0
    /// (when present and evaluable, §8.10.3 a) or AToB0 table maps the
    /// header's data colour space straight to the colour space named in
    /// its PCS field.
    pub fn device_link(link: &IccProfile) -> Result<Self, IccError> {
        Self::device_link_with(link, TransformOptions::new())
    }

    /// [`Self::device_link`] with explicit options (threads and CLUT
    /// interpolation; a DeviceLink has a single table, so the intent
    /// is not used).
    pub fn device_link_with(link: &IccProfile, opts: TransformOptions) -> Result<Self, IccError> {
        if link.header.class != ProfileClass::DeviceLink {
            return Err(IccError::Unsupported("not a DeviceLink profile"));
        }
        let in_ch = device_channels(link)?;
        let out_ch = link
            .header
            .pcs
            .channels()
            .ok_or(IccError::Unsupported("unknown output colour space"))?;
        let lut = match &link.d2b[0] {
            Some(m) if m.in_ch == in_ch && m.out_ch == out_ch => Link::Mpe(m.clone()),
            _ => {
                let lut = link.a2b[0]
                    .clone()
                    .ok_or(IccError::Unsupported("DeviceLink without AToB0"))?;
                if lut.in_ch != in_ch || lut.out_ch != out_ch {
                    return Err(IccError::Malformed("AToB0 channel counts"));
                }
                let mut lut = lut;
                lut.set_interpolation(opts.interpolation);
                Link::Lut(lut)
            }
        };
        // Placeholders: never evaluated while `link` is set.
        let id = Curve::Identity;
        Ok(Self {
            src: ToPcs::Gray {
                curve: id.clone(),
                pcs: PcsKind::Xyz,
            },
            dst: FromPcs::Gray {
                curve: id,
                pcs: PcsKind::Xyz,
            },
            link: Some(lut),
            mid: Vec::new(),
            abs_scale: None,
            in_ch,
            out_ch,
            in_space: link.header.color_space,
            out_space: link.header.pcs,
            threads: opts.threads.max(1),
        })
    }

    /// Tabulate this transform on a regular grid of `grid_points` per
    /// input channel and return a transform that interpolates the table
    /// instead of running the profiles — a device link built in memory.
    ///
    /// The table costs `grid_points^channels` evaluations of this
    /// transform once and makes every pixel a handful of table reads
    /// (simplex interpolation, [`ClutInterpolation::Tetrahedral`]),
    /// whatever the profiles contain. It is an approximation between
    /// grid points: 33 points for RGB and 17 for CMYK keep smooth
    /// transforms within a fraction of a ΔE (see the crate README for
    /// measured figures); grid points themselves are exact to `f32`.
    /// Inputs are clamped to `[0, 1]`.
    ///
    /// Supports 1 to 4 input channels and `2..=256` grid points, with
    /// at most 2²⁴ table values in total; anything else is
    /// [`IccError::Unsupported`]. The thread budget of this transform
    /// is used for the tabulation and kept by the result.
    pub fn precompute(&self, grid_points: usize) -> Result<Self, IccError> {
        let (ic, oc) = (self.in_ch, self.out_ch);
        if !(1..=4).contains(&ic) {
            return Err(IccError::Unsupported(
                "precomputed table for more than four input channels",
            ));
        }
        if !(2..=256).contains(&grid_points) {
            return Err(IccError::Unsupported("precomputed table grid size"));
        }
        let nodes = grid_points.pow(ic as u32);
        if nodes.saturating_mul(oc) > MAX_BAKED_VALUES {
            return Err(IccError::Unsupported("precomputed table too large"));
        }
        let mut data = vec![0f32; nodes * oc];
        let fill = |first: usize, part: &mut [f32]| {
            let mut i = [0f64; 15];
            let mut o = [0f64; 15];
            for (k, cell) in part.chunks_exact_mut(oc).enumerate() {
                let mut node = first + k;
                for c in (0..ic).rev() {
                    i[c] = (node % grid_points) as f64 / (grid_points - 1) as f64;
                    node /= grid_points;
                }
                self.pixel(&i, &mut o, false);
                for (d, v) in cell.iter_mut().zip(&o) {
                    *d = *v as f32;
                }
            }
        };
        let threads = self.threads.clamp(1, (nodes / 1024).max(1));
        if threads == 1 {
            fill(0, &mut data);
        } else {
            let per = nodes.div_ceil(threads);
            std::thread::scope(|sc| {
                for (n, part) in data.chunks_mut(per * oc).enumerate() {
                    let fill = &fill;
                    sc.spawn(move || fill(n * per, part));
                }
            });
        }
        let mut out = self.clone();
        out.mid.clear();
        out.link = Some(Link::Baked(Baked {
            grid: grid_points,
            in_ch: ic,
            out_ch: oc,
            data,
        }));
        Ok(out)
    }

    /// Whether this transform interpolates a precomputed table
    /// ([`Self::precompute`]).
    pub fn is_precomputed(&self) -> bool {
        matches!(self.link, Some(Link::Baked(_)))
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
        self.pixel(input, output, false);
    }

    fn pixel(&self, input: &[f64], output: &mut [f64], pre: bool) {
        match &self.link {
            Some(Link::Lut(l)) => {
                l.eval_from(input, output, usize::from(pre));
                return;
            }
            Some(Link::Mpe(m)) => {
                m.eval(input, output);
                return;
            }
            Some(Link::Baked(b)) => {
                b.eval(input, output);
                return;
            }
            None => {}
        }
        let (mut v, mut kind) = self.src.eval(input, pre);
        for m in &self.mid {
            match m {
                Mid::Abstract {
                    table,
                    input,
                    output,
                } => {
                    let x = connect(v, kind, *input, None);
                    let mut y = [0.0; 3];
                    match table {
                        Link::Lut(l) => {
                            let n = pcs::encode_pcs(*input, l.kind, x).map(|c| c.clamp(0.0, 1.0));
                            l.eval(&n, &mut y);
                            y = pcs::decode_pcs(*output, l.kind, y);
                        }
                        Link::Mpe(mp) => mp.eval(&x, &mut y),
                        Link::Baked(bk) => bk.eval(&x, &mut y),
                    }
                    (v, kind) = (y, *output);
                }
                Mid::Device { from, to, scale } => {
                    let mut dev = [0f64; 15];
                    from.eval(connect(v, kind, from.wants(), *scale), &mut dev);
                    (v, kind) = to.eval(&dev, false);
                }
            }
        }
        let v = connect(v, kind, self.dst.wants(), self.abs_scale);
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

    /// Shared batch driver: optional pre-tabulated first-stage curves
    /// (`tables[c][code]`), split over `self.threads` scoped threads.
    #[allow(clippy::too_many_arguments)]
    fn run<S, D>(
        &self,
        src: &[S],
        dst: &mut [D],
        px: usize,
        norm: impl Fn(S) -> f64 + Sync,
        code: impl Fn(S) -> usize + Sync,
        tables: Option<&[Vec<f64>]>,
        quant: impl Fn(f64) -> D + Sync,
    ) where
        S: Copy + Sync,
        D: Send,
    {
        let (ic, oc) = (self.in_ch, self.out_ch);
        let work = |src: &[S], dst: &mut [D]| {
            let mut i = [0f64; 15];
            let mut o = [0f64; 15];
            for (s, d) in src.chunks_exact(ic).zip(dst.chunks_exact_mut(oc)) {
                match tables {
                    Some(t) => {
                        for c in 0..ic {
                            i[c] = t[c][code(s[c])];
                        }
                        self.pixel(&i, &mut o, true);
                    }
                    None => {
                        for c in 0..ic {
                            i[c] = norm(s[c]);
                        }
                        self.pixel(&i, &mut o, false);
                    }
                }
                for c in 0..oc {
                    d[c] = quant(o[c]);
                }
            }
        };
        let threads = self.threads.clamp(1, px.max(1));
        if threads == 1 {
            work(&src[..px * ic], &mut dst[..px * oc]);
            return;
        }
        let per = px.div_ceil(threads);
        std::thread::scope(|sc| {
            for (s, d) in src[..px * ic]
                .chunks(per * ic)
                .zip(dst[..px * oc].chunks_mut(per * oc))
            {
                let work = &work;
                sc.spawn(move || work(s, d));
            }
        });
    }

    /// Tabulate the first-stage curves for every input code `0..=max`.
    fn tables(&self, max: usize) -> Option<Vec<Vec<f64>>> {
        let curves = match &self.link {
            Some(Link::Lut(l)) => l.first_curves()?,
            Some(Link::Mpe(_) | Link::Baked(_)) => return None,
            None => self.src.first_curves()?,
        };
        Some(
            curves
                .iter()
                .map(|c| (0..=max).map(|k| c.eval(k as f64 / max as f64)).collect())
                .collect(),
        )
    }

    /// Transform interleaved `f32` pixels (normalised device values).
    pub fn transform_f32(&self, src: &[f32], dst: &mut [f32]) -> Result<(), IccError> {
        let px = self.check_len(src.len(), dst.len())?;
        if let Some(Link::Baked(b)) = &self.link {
            let src = &src[..px * self.in_ch];
            b.run(src, dst, self.threads, |v, c| b.position(c, v), |v| v);
            return Ok(());
        }
        self.run(src, dst, px, |v| v as f64, |_| 0, None, |v| v as f32);
        Ok(())
    }

    /// Transform interleaved 16-bit pixels (`0..=65535` full scale).
    pub fn transform_u16(&self, src: &[u16], dst: &mut [u16]) -> Result<(), IccError> {
        let px = self.check_len(src.len(), dst.len())?;
        if let Some(Link::Baked(b)) = &self.link {
            let src = &src[..px * self.in_ch];
            b.run(
                src,
                dst,
                self.threads,
                |v, c| b.position(c, v as f32 / 65535.0),
                |v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
            );
            return Ok(());
        }
        // Tabulating 65536 codes per channel pays off on larger images.
        let tables = if px >= 1 << 15 {
            self.tables(65535)
        } else {
            None
        };
        self.run(
            src,
            dst,
            px,
            |v| v as f64 / 65535.0,
            |v| v as usize,
            tables.as_deref(),
            |v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16,
        );
        Ok(())
    }

    /// Transform interleaved 8-bit pixels (`0..=255` full scale).
    pub fn transform_u8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), IccError> {
        let px = self.check_len(src.len(), dst.len())?;
        if let Some(Link::Baked(b)) = &self.link {
            let src = &src[..px * self.in_ch];
            let at = b.code_positions(255);
            b.run(
                src,
                dst,
                self.threads,
                |v, c| at[c][v as usize],
                |v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            );
            return Ok(());
        }
        let tables = if px >= 256 { self.tables(255) } else { None };
        self.run(
            src,
            dst,
            px,
            |v| v as f64 / 255.0,
            |v| v as usize,
            tables.as_deref(),
            |v| (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        );
        Ok(())
    }
}
