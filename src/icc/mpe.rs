//! `multiProcessElementsType` (`mpet`, ICC.1:2022 §10.16): the
//! float32 transforms of the `DToBx` / `BToDx` tags (§9.2.9–9.2.12,
//! §9.2.25–9.2.28). A tag is a list of processing elements run in
//! order — segmented curve sets (`cvst`, §10.16.2.2), matrices with
//! offsets (`matf`, §10.16.2.3), float CLUTs (`clut`, §10.16.2.4) and
//! the pass-through `bACS` / `eACS` placeholders (§10.16.2.5). Values
//! are never clipped between elements; only a CLUT clamps its own
//! input to `[0, 1]`.

use super::IccError;
use super::lut::multilinear;

/// The most channels an element (or the tag) may carry here; the
/// processing buffers are fixed-size. A tag declaring more is treated
/// as unsupported, which sends the transform to the §8.10 fallback.
pub(crate) const MAX_MPE_CHANNELS: usize = 16;

/// Cap on CLUT grid samples (same bound as the integer LUT tags).
const MAX_CLUT_SAMPLES: usize = 1 << 26;

/// Cap on elements per tag and on segments per curve.
const MAX_ELEMENTS: usize = 1024;

/// One segment of a `curf` curve (Tables 59–61).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Segment {
    /// `parf` with its function type (Table 60) and parameters in the
    /// table's order (`γ, a, b, c[, d]` or `a, b, c, d, e`).
    Formula { function: u16, p: [f64; 5] },
    /// `samf`: the value at the segment's lower break-point (the
    /// preceding segment's output there) followed by the stored samples,
    /// equally spaced up to the upper break-point.
    Sampled { lo: f64, hi: f64, values: Vec<f64> },
}

impl Segment {
    fn eval(&self, x: f64) -> f64 {
        match self {
            Segment::Formula { function, p } => match function {
                // Y = (aX + b)^γ + c
                0 => (p[1] * x + p[2]).powf(p[0]) + p[3],
                // Y = a·log10(b·X^γ + c) + d
                1 => p[1] * (p[2] * x.powf(p[0]) + p[3]).log10() + p[4],
                // Y = a·b^(cX + d) + e
                _ => p[0] * p[1].powf(p[2] * x + p[3]) + p[4],
            },
            Segment::Sampled { lo, hi, values } => {
                // values[0] sits at `lo`, values[j] at lo + j·(hi − lo)/n.
                let n = values.len() - 1;
                let span = hi - lo;
                let pos = if span > 0.0 {
                    ((x - lo) / span * n as f64).clamp(0.0, n as f64)
                } else {
                    n as f64
                };
                let i = (pos.floor() as usize).min(n.saturating_sub(1));
                if n == 0 {
                    return values[0];
                }
                let f = pos - i as f64;
                values[i] + f * (values[i + 1] - values[i])
            }
        }
    }
}

/// A `curf` segmented curve (Table 58).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SegmentedCurve {
    /// `n − 1` non-decreasing break-points.
    pub breaks: Vec<f64>,
    /// `n` segments; segment `k` covers `(breaks[k − 1], breaks[k]]`.
    pub segments: Vec<Segment>,
}

impl SegmentedCurve {
    pub fn eval(&self, x: f64) -> f64 {
        // "The output value for an input shall be specified by the first
        // segment in the segment list that contains that input."
        let k = self
            .breaks
            .iter()
            .position(|&b| x <= b)
            .unwrap_or(self.breaks.len());
        self.segments[k].eval(x)
    }
}

/// One processing element.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Element {
    /// One segmented curve per channel.
    Curves(Vec<SegmentedCurve>),
    /// `q × p` matrix (row per output) plus `q` offsets (equation 21).
    Matrix {
        p: usize,
        q: usize,
        m: Vec<f64>,
        offset: Vec<f64>,
    },
    /// Float CLUT; input clamped to `[0, 1]`.
    Clut {
        grid: Vec<usize>,
        q: usize,
        data: Vec<f32>,
    },
    /// `bACS` / `eACS`: no change.
    Pass,
}

/// A parsed `mpet` tag.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Mpe {
    pub in_ch: usize,
    pub out_ch: usize,
    pub elements: Vec<Element>,
}

impl Mpe {
    /// Run the elements in order. `input` holds `in_ch` values and
    /// `output` receives `out_ch` values, unclipped.
    pub fn eval(&self, input: &[f64], output: &mut [f64]) {
        let mut a = [0f64; MAX_MPE_CHANNELS];
        let mut b = [0f64; MAX_MPE_CHANNELS];
        a[..self.in_ch].copy_from_slice(&input[..self.in_ch]);
        for el in &self.elements {
            match el {
                Element::Curves(cs) => {
                    for (v, c) in a.iter_mut().zip(cs) {
                        *v = c.eval(*v);
                    }
                }
                Element::Matrix { p, q, m, offset } => {
                    for r in 0..*q {
                        let row = &m[r * p..(r + 1) * p];
                        b[r] =
                            row.iter().zip(&a[..*p]).map(|(e, x)| e * x).sum::<f64>() + offset[r];
                    }
                    a[..*q].copy_from_slice(&b[..*q]);
                }
                Element::Clut { grid, q, data } => {
                    let p = grid.len();
                    let mut x = [0f64; MAX_MPE_CHANNELS];
                    for (d, s) in x.iter_mut().zip(&a[..p]) {
                        *d = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
                    }
                    multilinear(grid, *q, data, &x[..p], &mut b[..*q]);
                    a[..*q].copy_from_slice(&b[..*q]);
                }
                Element::Pass => {}
            }
        }
        output[..self.out_ch].copy_from_slice(&a[..self.out_ch]);
    }
}

// --------------------------------------------------------------------
// Parsing

fn be16(b: &[u8], o: usize) -> Result<usize, IccError> {
    b.get(o..o + 2)
        .map(|s| u16::from_be_bytes([s[0], s[1]]) as usize)
        .ok_or(IccError::Truncated)
}

fn be32(b: &[u8], o: usize) -> Result<usize, IccError> {
    b.get(o..o + 4)
        .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or(IccError::Truncated)
}

fn f32be(b: &[u8], o: usize) -> Result<f64, IccError> {
    b.get(o..o + 4)
        .map(|s| f32::from_be_bytes([s[0], s[1], s[2], s[3]]) as f64)
        .ok_or(IccError::Truncated)
}

/// `(offset, size)` slice of `t`, checked.
fn sub(t: &[u8], off: usize, size: usize) -> Result<&[u8], IccError> {
    off.checked_add(size)
        .and_then(|end| t.get(off..end))
        .ok_or(IccError::Truncated)
}

fn check_channels(n: usize) -> Result<usize, IccError> {
    if n == 0 {
        return Err(IccError::Malformed("mpet element with no channels"));
    }
    if n > MAX_MPE_CHANNELS {
        return Err(IccError::Unsupported(
            "mpet element with more than 16 channels",
        ));
    }
    Ok(n)
}

/// Parse an `mpet` tag (Table 55). Unknown element types are
/// `Unsupported` (the caller falls back per §8.10).
pub(crate) fn parse_mpet(t: &[u8]) -> Result<Mpe, IccError> {
    if t.get(0..4) != Some(b"mpet") {
        return Err(IccError::Malformed("not an mpet tag"));
    }
    let in_ch = check_channels(be16(t, 8)?)?;
    let out_ch = check_channels(be16(t, 10)?)?;
    let n = be32(t, 12)?;
    if n == 0 {
        return Err(IccError::Malformed("mpet without elements"));
    }
    if n > MAX_ELEMENTS || n > t.len() / 8 {
        return Err(IccError::Truncated);
    }
    let mut elements = Vec::with_capacity(n);
    let mut ch = in_ch;
    // Elements and curves may share data (§10.16.1, §10.16.2.2), so a
    // small tag could name the same large CLUT many times; bound the
    // values decoded in total by the tag's own size.
    let mut budget = t.len() / 4 + 4096;
    for i in 0..n {
        let off = be32(t, 16 + 8 * i)?;
        let size = be32(t, 20 + 8 * i)?;
        let e = sub(t, off, size)?;
        let (el, p, q) = parse_element(e, &mut budget)?;
        if p != ch {
            return Err(IccError::Malformed("mpet element channel chain"));
        }
        ch = q;
        elements.push(el);
    }
    if ch != out_ch {
        return Err(IccError::Malformed("mpet output channel count"));
    }
    Ok(Mpe {
        in_ch,
        out_ch,
        elements,
    })
}

/// One element (Table 56) → `(element, p, q)`.
fn parse_element(e: &[u8], budget: &mut usize) -> Result<(Element, usize, usize), IccError> {
    let sig = e.get(0..4).ok_or(IccError::Truncated)?;
    let p = check_channels(be16(e, 8)?)?;
    let q = check_channels(be16(e, 10)?)?;
    let el = match sig {
        b"cvst" => {
            if p != q {
                return Err(IccError::Malformed("curve set with p != q"));
            }
            let mut curves = Vec::with_capacity(p);
            for c in 0..p {
                let off = be32(e, 12 + 8 * c)?;
                let size = be32(e, 16 + 8 * c)?;
                curves.push(parse_curf(sub(e, off, size)?, budget)?);
            }
            Element::Curves(curves)
        }
        b"matf" => {
            let count = q * (p + 1);
            spend(budget, count)?;
            let mut v = Vec::with_capacity(count);
            for k in 0..count {
                v.push(f32be(e, 12 + 4 * k)?);
            }
            let offset = v.split_off(p * q);
            Element::Matrix { p, q, m: v, offset }
        }
        b"clut" => {
            let mut grid = Vec::with_capacity(p);
            let mut total = q;
            for d in 0..p {
                let g = *e.get(12 + d).ok_or(IccError::Truncated)? as usize;
                if g < 2 {
                    return Err(IccError::Malformed(
                        "mpet CLUT with fewer than 2 grid points",
                    ));
                }
                grid.push(g);
                total = total
                    .checked_mul(g)
                    .filter(|&t| t <= MAX_CLUT_SAMPLES)
                    .ok_or(IccError::Malformed("mpet CLUT too large"))?;
            }
            spend(budget, total)?;
            let bytes = sub(e, 28, total * 4)?;
            let data = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_be_bytes(*c))
                .collect();
            Element::Clut { grid, q, data }
        }
        b"bACS" | b"eACS" => {
            if p != q {
                return Err(IccError::Malformed("bACS / eACS with p != q"));
            }
            Element::Pass
        }
        _ => return Err(IccError::Unsupported("unknown mpet element")),
    };
    Ok((el, p, q))
}

fn spend(budget: &mut usize, n: usize) -> Result<(), IccError> {
    *budget = budget
        .checked_sub(n)
        .ok_or(IccError::Malformed("mpet data larger than its tag"))?;
    Ok(())
}

/// A `curf` curve (Table 58) with its segments (Tables 59, 61).
fn parse_curf(c: &[u8], budget: &mut usize) -> Result<SegmentedCurve, IccError> {
    if c.get(0..4) != Some(b"curf") {
        return Err(IccError::Malformed("curve set entry is not curf"));
    }
    let n = be16(c, 8)?;
    if n == 0 {
        return Err(IccError::Malformed("curf without segments"));
    }
    if n > MAX_ELEMENTS {
        return Err(IccError::Unsupported("curf with too many segments"));
    }
    spend(budget, 6 * n)?;
    let mut breaks = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        let b = f32be(c, 12 + 4 * k)?;
        if breaks.last().is_some_and(|&l| b < l) || b.is_nan() {
            return Err(IccError::Malformed("decreasing curf break-points"));
        }
        breaks.push(b);
    }
    let mut off = 12 + 4 * (n - 1);
    let mut segments: Vec<Segment> = Vec::with_capacity(n);
    for k in 0..n {
        let sig = c.get(off..off + 4).ok_or(IccError::Truncated)?;
        match sig {
            b"parf" => {
                let function = be16(c, off + 8)? as u16;
                let count = match function {
                    0 => 4,
                    1 | 2 => 5,
                    _ => return Err(IccError::Unsupported("parf function type")),
                };
                let mut p = [0.0; 5];
                for (j, v) in p.iter_mut().take(count).enumerate() {
                    *v = f32be(c, off + 12 + 4 * j)?;
                }
                segments.push(Segment::Formula { function, p });
                off += 12 + 4 * count;
            }
            b"samf" => {
                // §10.16.2.2: the first and last segments are formulas.
                if k == 0 || k == n - 1 {
                    return Err(IccError::Malformed("sampled first / last curf segment"));
                }
                let count = be32(c, off + 8)?;
                if count == 0 {
                    return Err(IccError::Malformed("samf without entries"));
                }
                if count > (c.len().saturating_sub(off + 12)) / 4 {
                    return Err(IccError::Truncated);
                }
                spend(budget, count)?;
                let (lo, hi) = (breaks[k - 1], breaks[k]);
                let mut values = Vec::with_capacity(count + 1);
                // The value at the lower break-point comes from the
                // preceding segment.
                values.push(segments[k - 1].eval(lo));
                for j in 0..count {
                    values.push(f32be(c, off + 12 + 4 * j)?);
                }
                segments.push(Segment::Sampled { lo, hi, values });
                off += 12 + 4 * count;
            }
            _ => return Err(IccError::Malformed("unknown curf segment")),
        }
        // Segments are 4-byte aligned already (every field is 4 bytes).
    }
    Ok(SegmentedCurve { breaks, segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_segment_interpolates_from_previous_segment() {
        // Segment 1: y = x (formula, γ = 1, a = 1); break at 0.
        // Segment 2: samples 0.5, 1.0 over (0, 1]; value at 0 is 0.
        // Segment 3: y = 1 above 1 (γ = 1, a = 0, b = 0, c = 1).
        let c = SegmentedCurve {
            breaks: vec![0.0, 1.0],
            segments: vec![
                Segment::Formula {
                    function: 0,
                    p: [1.0, 1.0, 0.0, 0.0, 0.0],
                },
                Segment::Sampled {
                    lo: 0.0,
                    hi: 1.0,
                    values: vec![0.0, 0.5, 1.0],
                },
                Segment::Formula {
                    function: 0,
                    p: [1.0, 0.0, 0.0, 1.0, 0.0],
                },
            ],
        };
        assert_eq!(c.eval(-0.5), -0.5);
        assert_eq!(c.eval(0.0), 0.0);
        assert!((c.eval(0.25) - 0.25).abs() < 1e-12);
        assert!((c.eval(0.75) - 0.75).abs() < 1e-12);
        assert_eq!(c.eval(1.0), 1.0);
        assert_eq!(c.eval(3.0), 1.0);
    }

    #[test]
    fn formula_types() {
        let f = |function, p| Segment::Formula { function, p }.eval(2.0);
        // (aX + b)^γ + c = (1·2 + 1)^2 + 1
        assert_eq!(f(0, [2.0, 1.0, 1.0, 1.0, 0.0]), 10.0);
        // a·log10(b·X^γ + c) + d = 2·log10(5·2^1 + 0) + 1
        assert!((f(1, [1.0, 2.0, 5.0, 0.0, 1.0]) - 3.0).abs() < 1e-12);
        // a·b^(cX + d) + e = 1·10^(1·2 + 0) + 0
        assert!((f(2, [1.0, 10.0, 1.0, 0.0, 0.0]) - 100.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    fn be(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_be_bytes());
    }

    #[test]
    fn shared_clut_data_is_bounded_by_the_tag_size() {
        // One 3→3 CLUT element (16³ grid) referenced 64 times.
        let mut clut = b"clut\0\0\0\0\0\x03\0\x03".to_vec();
        let mut g = [0u8; 16];
        g[..3].fill(16);
        clut.extend_from_slice(&g);
        clut.resize(28 + 16 * 16 * 16 * 3 * 4, 0);
        let n = 64u32;
        let mut t = b"mpet\0\0\0\0\0\x03\0\x03".to_vec();
        be(&mut t, n);
        let off = 16 + 8 * n;
        for _ in 0..n {
            be(&mut t, off);
            be(&mut t, clut.len() as u32);
        }
        t.extend_from_slice(&clut);
        assert!(matches!(parse_mpet(&t), Err(IccError::Malformed(_))));
        // A single reference is fine.
        let mut one = b"mpet\0\0\0\0\0\x03\0\x03".to_vec();
        be(&mut one, 1);
        be(&mut one, 24);
        be(&mut one, clut.len() as u32);
        one.extend_from_slice(&clut);
        assert!(parse_mpet(&one).is_ok());
    }
}
