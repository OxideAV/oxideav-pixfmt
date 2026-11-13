//! LUT-based transforms: `lut8Type` / `lut16Type` (ICC.1:2022 §10.11 /
//! §10.10) and `lutAToBType` / `lutBToAType` (§10.12 / §10.13), all
//! reduced to one ordered list of processing stages over values
//! normalised to `[0, 1]`.

use super::curve::{Curve, clamp01};

/// How values between the grid points of a colour lookup table are
/// obtained. ICC.1:2022 asks for "linear interpolation" between table
/// entries (§10.10, §10.12.3) without fixing how it extends to several
/// dimensions; both schemes below reproduce the grid points exactly and
/// are exact for a table that is linear in every input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[non_exhaustive]
pub enum ClutInterpolation {
    /// Multilinear: the weighted mean of all `2^n` corners of the
    /// enclosing grid cell (trilinear for three inputs). The default.
    #[default]
    Multilinear,
    /// Simplex ("tetrahedral" for three inputs): the cell is cut into
    /// `n!` simplices that all share its main diagonal, and the value
    /// is the barycentric mean of the `n + 1` corners of the simplex
    /// holding the point. It keeps the neutral axis of a table free of
    /// the cross-talk multilinear weights introduce, and reads `n + 1`
    /// grid points instead of `2^n`.
    Tetrahedral,
}

/// A multi-dimensional colour lookup table. Grid samples are stored as
/// 16-bit codes (8-bit tables are widened by ×257, which keeps
/// `code / 255 == wide / 65535` exact).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Clut {
    pub grid: Vec<usize>,
    pub out_ch: usize,
    pub data: Vec<u16>,
    pub interpolation: ClutInterpolation,
}

impl Clut {
    /// Interpolate over all input dimensions. The first input channel
    /// varies least rapidly (§10.10, §10.12.3).
    pub fn eval(&self, input: &[f64], out: &mut [f64]) {
        let mut x = [0f64; 16];
        for (d, v) in x.iter_mut().zip(input) {
            *d = clamp01(*v);
        }
        interpolate(
            self.interpolation,
            &self.grid,
            self.out_ch,
            &self.data,
            &x[..self.grid.len()],
            out,
        );
        for v in out.iter_mut().take(self.out_ch) {
            *v /= 65535.0;
        }
    }
}

/// The grid cell holding a point: the flat offset of its lowest corner
/// and, for every dimension the point is strictly inside of, the
/// fraction along it and the flat step to the next grid point.
struct Cell {
    origin: usize,
    frac: [f64; 16],
    step: [usize; 16],
    live: usize,
}

/// Locate `input` (already in `[0, 1]`) in a grid whose first dimension
/// varies least rapidly. Dimensions whose fraction is zero (or whose
/// grid has one point) contribute only their lower corner, so they drop
/// out of the corner walk.
fn locate(grid: &[usize], out_ch: usize, input: &[f64]) -> Cell {
    let mut cell = Cell {
        origin: 0,
        frac: [0.0; 16],
        step: [0; 16],
        live: 0,
    };
    let mut stride = out_ch;
    for d in (0..grid.len()).rev() {
        let g = grid[d];
        if g >= 2 {
            let pos = input[d] * (g - 1) as f64;
            let i = (pos.floor() as usize).min(g - 2);
            let f = pos - i as f64;
            cell.origin += i * stride;
            if f > 0.0 {
                cell.frac[cell.live] = f;
                cell.step[cell.live] = stride;
                cell.live += 1;
            }
        }
        stride *= g;
    }
    cell
}

/// Interpolate into a grid table with the chosen scheme.
pub(crate) fn interpolate<T: Copy + Into<f64>>(
    scheme: ClutInterpolation,
    grid: &[usize],
    out_ch: usize,
    data: &[T],
    input: &[f64],
    out: &mut [f64],
) {
    match scheme {
        ClutInterpolation::Multilinear => multilinear(grid, out_ch, data, input, out),
        ClutInterpolation::Tetrahedral => simplex(grid, out_ch, data, input, out),
    }
}

/// Multilinear interpolation into a grid table (first input dimension
/// varies least rapidly; `out_ch` values per grid point). `input` must
/// already lie in `[0, 1]`. Shared by the integer LUT tags and the
/// float `mpet` CLUT element (§10.16.2.4).
pub(crate) fn multilinear<T: Copy + Into<f64>>(
    grid: &[usize],
    out_ch: usize,
    data: &[T],
    input: &[f64],
    out: &mut [f64],
) {
    let o = out_ch;
    let Cell {
        origin,
        frac,
        step,
        live,
    } = locate(grid, o, input);
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
        let cell = &data[off..off + o];
        for (a, &v) in acc.iter_mut().zip(cell) {
            *a += w * v.into();
        }
    }
    out[..o].copy_from_slice(&acc[..o]);
}

/// Simplex interpolation into a grid table (same layout and input
/// contract as [`multilinear`]).
///
/// With the fractions sorted `f₁ ≥ f₂ ≥ … ≥ fₙ`, the point lies in the
/// simplex whose corners are reached from the cell's lowest corner `V₀`
/// by stepping along the dimensions in that order (`V₁`, …, `Vₙ`, the
/// last being the opposite corner), and its value is
/// `(1 − f₁)·V₀ + (f₁ − f₂)·V₁ + … + fₙ·Vₙ`.
pub(crate) fn simplex<T: Copy + Into<f64>>(
    grid: &[usize],
    out_ch: usize,
    data: &[T],
    input: &[f64],
    out: &mut [f64],
) {
    let o = out_ch;
    let Cell {
        origin,
        mut frac,
        mut step,
        live,
    } = locate(grid, o, input);
    // Insertion sort by descending fraction (at most 16 entries).
    for i in 1..live {
        let mut j = i;
        while j > 0 && frac[j - 1] < frac[j] {
            frac.swap(j - 1, j);
            step.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut acc = [0f64; 16];
    let mut off = origin;
    let mut weight = 1.0 - if live > 0 { frac[0] } else { 0.0 };
    for k in 0..=live {
        let cell = &data[off..off + o];
        for (a, &v) in acc.iter_mut().zip(cell) {
            *a += weight * v.into();
        }
        if k < live {
            off += step[k];
            weight = frac[k] - if k + 1 < live { frac[k + 1] } else { 0.0 };
        }
    }
    out[..o].copy_from_slice(&acc[..o]);
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
    /// Select the interpolation scheme of every CLUT stage.
    pub fn set_interpolation(&mut self, scheme: ClutInterpolation) {
        for st in &mut self.stages {
            if let Stage::Clut(c) = st {
                c.interpolation = scheme;
            }
        }
    }

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
        let mut t = Clut {
            grid,
            out_ch: 1,
            data,
            interpolation: ClutInterpolation::Multilinear,
        };
        let mut out = [0.0];
        for scheme in [
            ClutInterpolation::Multilinear,
            ClutInterpolation::Tetrahedral,
        ] {
            t.interpolation = scheme;
            for (x, y) in [(0.1, 0.9), (0.5, 0.5), (1.0, 1.0), (0.0, 0.33)] {
                t.eval(&[x, y], &mut out);
                assert!((out[0] - (0.25 * x + 0.5 * y)).abs() < 2e-5, "{x} {y}");
            }
        }
    }
}

#[cfg(test)]
mod simplex_tests {
    use super::*;

    /// A 3-input, 2-output table of a function that is not linear.
    fn table(g: usize) -> Clut {
        let mut data = Vec::new();
        for i in 0..g {
            for j in 0..g {
                for k in 0..g {
                    let (x, y, z) = (
                        i as f64 / (g - 1) as f64,
                        j as f64 / (g - 1) as f64,
                        k as f64 / (g - 1) as f64,
                    );
                    data.push(((x * y * 0.5 + z * z * 0.5) * 65535.0).round() as u16);
                    data.push((((x + y + z) / 3.0).sqrt() * 65535.0).round() as u16);
                }
            }
        }
        Clut {
            grid: vec![g; 3],
            out_ch: 2,
            data,
            interpolation: ClutInterpolation::Tetrahedral,
        }
    }

    #[test]
    fn both_schemes_reproduce_the_grid_points() {
        let g = 5;
        let mut t = table(g);
        for scheme in [
            ClutInterpolation::Multilinear,
            ClutInterpolation::Tetrahedral,
        ] {
            t.interpolation = scheme;
            for i in 0..g {
                for j in 0..g {
                    for k in 0..g {
                        let p = [i, j, k].map(|v| v as f64 / (g - 1) as f64);
                        let mut out = [0.0; 2];
                        t.eval(&p, &mut out);
                        let at = ((i * g + j) * g + k) * 2;
                        for (got, &code) in out.iter().zip(&t.data[at..at + 2]) {
                            let want = code as f64 / 65535.0;
                            assert!((got - want).abs() < 1e-12, "{scheme:?} {p:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simplex_is_linear_along_the_cell_diagonal_and_bounded_by_corners() {
        let g = 3;
        let t = table(g);
        // On the diagonal of a cell every fraction is equal, so only
        // the two end corners carry weight.
        let lo = [0.0, 0.5, 0.5];
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        t.eval(&lo, &mut a);
        t.eval(&[0.5, 1.0, 1.0], &mut b);
        for f in [0.25, 0.5, 0.8] {
            let p = [lo[0] + 0.5 * f, lo[1] + 0.5 * f, lo[2] + 0.5 * f];
            let mut out = [0.0; 2];
            t.eval(&p, &mut out);
            for c in 0..2 {
                let want = a[c] + f * (b[c] - a[c]);
                assert!((out[c] - want).abs() < 1e-12);
            }
        }
        // Anywhere, the result is a convex mix of table entries, and
        // the two schemes stay close on a smooth table.
        let mut m = t.clone();
        m.interpolation = ClutInterpolation::Multilinear;
        let mut s: u32 = 7;
        for _ in 0..500 {
            let mut p = [0.0; 3];
            for v in &mut p {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (s >> 8) as f64 / (1u32 << 24) as f64;
            }
            let (mut x, mut y) = ([0.0; 2], [0.0; 2]);
            t.eval(&p, &mut x);
            m.eval(&p, &mut y);
            assert!(x.iter().all(|v| (0.0..=1.0).contains(v)));
            // The first output is a smooth polynomial: on a 3-point
            // grid the two schemes differ by less than its curvature
            // term (h² / 8 · 1 per axis pair).
            assert!((x[0] - y[0]).abs() < 0.04, "{p:?}: {x:?} vs {y:?}");
        }
    }
}
