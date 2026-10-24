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
        // Per-dimension base index, fraction and stride.
        let mut base = [0usize; 16];
        let mut frac = [0f64; 16];
        let mut stride = [0usize; 16];
        let mut s = o;
        for d in (0..n).rev() {
            stride[d] = s;
            s *= self.grid[d];
        }
        let mut origin = 0usize;
        for d in 0..n {
            let g = self.grid[d];
            let x = clamp01(input[d]);
            if g < 2 {
                base[d] = 0;
                frac[d] = 0.0;
                continue;
            }
            let pos = x * (g - 1) as f64;
            let i = (pos.floor() as usize).min(g - 2);
            base[d] = i;
            frac[d] = pos - i as f64;
            origin += i * stride[d];
        }
        for v in out.iter_mut().take(o) {
            *v = 0.0;
        }
        // Visit the 2^n corners of the enclosing cell. Dimensions whose
        // fraction is exactly zero contribute only their lower corner.
        let corners = 1usize << n;
        for mask in 0..corners {
            let mut w = 1.0;
            let mut off = origin;
            let mut skip = false;
            for d in 0..n {
                if mask & (1 << d) != 0 {
                    if frac[d] == 0.0 || self.grid[d] < 2 {
                        skip = true;
                        break;
                    }
                    w *= frac[d];
                    off += stride[d];
                } else {
                    w *= 1.0 - frac[d];
                }
            }
            if skip || w == 0.0 {
                continue;
            }
            for (c, v) in out.iter_mut().take(o).enumerate() {
                *v += w * self.data[off + c] as f64;
            }
        }
        for v in out.iter_mut().take(o) {
            *v /= 65535.0;
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
        let mut a = [0f64; 16];
        let mut b = [0f64; 16];
        let mut ch = self.in_ch;
        for (i, v) in input.iter().take(ch).enumerate() {
            a[i] = *v;
        }
        for st in &self.stages {
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
