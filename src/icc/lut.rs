//! LUT-based transforms: `lut8Type` / `lut16Type` (ICC.1:2022 §10.11 /
//! §10.10) and `lutAToBType` / `lutBToAType` (§10.12 / §10.13), all
//! reduced to one ordered list of processing stages over values
//! normalised to `[0, 1]`.

use super::curve::{Curve, clamp01};

/// A multi-dimensional colour lookup table. Grid samples are stored as
/// 16-bit codes (8-bit tables are widened by ×257, which keeps
/// `code / 255 == wide / 65535` exact).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Clut {
    pub grid: Vec<usize>,
    pub out_ch: usize,
    pub data: Vec<u16>,
}

impl Clut {
    /// Multilinear interpolation (the "linear interpolation" every LUT
    /// clause of ICC.1:2022 prescribes) over all input dimensions. The
    /// first input channel varies least rapidly (§10.10, §10.12.3).
    pub fn eval(&self, input: &[f64], out: &mut [f64]) {
        let n = self.grid.len();
        let o = self.out_ch;
        // Per-dimension cell origin and fraction; dimensions whose
        // fraction is zero (or whose grid has one point) contribute only
        // their lower corner, so they drop out of the corner walk.
        let mut frac = [0f64; 16];
        let mut step = [0usize; 16];
        let mut live = 0usize;
        let mut origin = 0usize;
        let mut stride = o;
        for d in (0..n).rev() {
            let g = self.grid[d];
            if g >= 2 {
                let x = clamp01(input[d]);
                let pos = x * (g - 1) as f64;
                let i = (pos.floor() as usize).min(g - 2);
                let f = pos - i as f64;
                origin += i * stride;
                if f > 0.0 {
                    frac[live] = f;
                    step[live] = stride;
                    live += 1;
                }
            }
            stride *= g;
        }
        let mut acc = [0f64; 16];
        for mask in 0..(1usize << live) {
            let mut w = 1.0;
            let mut off = origin;
            for k in 0..live {
                if mask & (1 << k) != 0 {
                    w *= frac[k];
                    off += step[k];
                } else {
                    w *= 1.0 - frac[k];
                }
            }
            let cell = &self.data[off..off + o];
            for (a, &v) in acc.iter_mut().zip(cell) {
                *a += w * v as f64;
            }
        }
        for (v, a) in out.iter_mut().zip(&acc[..o]) {
            *v = a / 65535.0;
        }
    }
}

/// One processing element.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stage {
    /// One curve per channel.
    Curves(Vec<Curve>),
    /// 3×3 matrix (lut8 / lut16, e1..e9 row-major), result clipped to
    /// `[0, 1]`.
    Matrix3([[f64; 3]; 3]),
    /// 3×3 matrix plus offsets (lutAToB / lutBToA, e1..e12), result
    /// clipped to `[0, 1]` (§10.12.5, §10.13.3).
    Matrix34([[f64; 3]; 3], [f64; 3]),
    /// Multi-dimensional table.
    Clut(Clut),
}

/// Which tag type a LUT came from — it fixes the PCSLAB encoding on the
/// PCS side (§10.8 legacy 16-bit for `mft2`, Tables 12/13 otherwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LutKind {
    Lut8,
    Lut16,
    AToB,
    BToA,
}

/// A parsed LUT tag: channel counts and the stage list in processing
/// order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Lut {
    pub kind: LutKind,
    pub in_ch: usize,
    pub out_ch: usize,
    pub stages: Vec<Stage>,
}

impl Lut {
    /// Run the pipeline. `input` holds `in_ch` normalised values;
    /// `output` receives `out_ch` normalised values.
    pub fn eval(&self, input: &[f64], output: &mut [f64]) {
        self.eval_from(input, output, 0);
    }

    /// The per-channel curves of the first stage, when the pipeline
    /// starts with one (callers may pre-tabulate them).
    pub fn first_curves(&self) -> Option<&[Curve]> {
        match self.stages.first() {
            Some(Stage::Curves(c)) => Some(c),
            _ => None,
        }
    }

    /// Run the pipeline from stage `start` (`input` already holds the
    /// output of the stages before it).
    pub fn eval_from(&self, input: &[f64], output: &mut [f64], start: usize) {
        let mut a = [0f64; 16];
        let mut b = [0f64; 16];
        let mut ch = self.in_ch;
        for (i, v) in input.iter().take(ch).enumerate() {
            a[i] = *v;
        }
        for st in &self.stages[start..] {
            match st {
                Stage::Curves(cs) => {
                    for (i, c) in cs.iter().enumerate().take(ch) {
                        a[i] = c.eval(a[i]);
                    }
                }
                Stage::Matrix3(m) => {
                    let x = [a[0], a[1], a[2]];
                    for r in 0..3 {
                        a[r] = clamp01(m[r][0] * x[0] + m[r][1] * x[1] + m[r][2] * x[2]);
                    }
                    ch = 3;
                }
                Stage::Matrix34(m, off) => {
                    let x = [a[0], a[1], a[2]];
                    for r in 0..3 {
                        a[r] = clamp01(m[r][0] * x[0] + m[r][1] * x[1] + m[r][2] * x[2] + off[r]);
                    }
                    ch = 3;
                }
                Stage::Clut(t) => {
                    t.eval(&a[..ch], &mut b[..t.out_ch]);
                    ch = t.out_ch;
                    a[..ch].copy_from_slice(&b[..ch]);
                }
            }
        }
        for (i, v) in output.iter_mut().take(self.out_ch).enumerate() {
            *v = clamp01(a[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clut_multilinear_reproduces_linear_function() {
        // 2-input, 1-output table of f(x, y) = 0.25x + 0.5y on a 3×5 grid
        // (first input varies least rapidly). Multilinear interpolation
        // is exact for a function linear in each input.
        let grid = vec![3usize, 5];
        let mut data = Vec::new();
        for i in 0..3 {
            for j in 0..5 {
                let x = i as f64 / 2.0;
                let y = j as f64 / 4.0;
                data.push(((0.25 * x + 0.5 * y) * 65535.0).round() as u16);
            }
        }
        let t = Clut {
            grid,
            out_ch: 1,
            data,
        };
        let mut out = [0.0];
        for (x, y) in [(0.1, 0.9), (0.5, 0.5), (1.0, 1.0), (0.0, 0.33)] {
            t.eval(&[x, y], &mut out);
            assert!((out[0] - (0.25 * x + 0.5 * y)).abs() < 2e-5, "{x} {y}");
        }
    }
}
