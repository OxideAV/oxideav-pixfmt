//! One-dimensional tone curves: `curveType` (ICC.1:2022 §10.6) and
//! `parametricCurveType` (§10.18, Table 68), plus their inverses per
//! Annex F.1.

/// A one-dimensional tone curve with domain and range `[0, 1]`.
///
/// Built from a `curv` tag (identity, a single gamma value, or a sampled
/// table — ICC.1:2022 §10.6) or a `para` tag (one of the five Table 68
/// function types — §10.18). Inputs outside `[0, 1]` are clamped and
/// every result is clipped to `[0, 1]`, as both clauses require.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Curve {
    /// `curv` with a zero entry count: `y = x`.
    Identity,
    /// `curv` with a single entry: `y = x^gamma` (the u8Fixed8Number is
    /// the exponent, "not an inverse").
    Gamma(f64),
    /// `curv` with two or more entries, already normalised to `[0, 1]`
    /// (entry / 65535). Entry `k` sits at input `k / (n − 1)`; values in
    /// between are linearly interpolated.
    Sampled(Vec<f64>),
    /// `para`: function type 0..=4 of Table 68 with its parameters in
    /// the table's left-to-right order (`g a b c d e f`; unused trailing
    /// entries are zero).
    Parametric {
        /// Table 68 function type (0..=4).
        function: u16,
        /// Parameters `g, a, b, c, d, e, f`.
        params: [f64; 7],
    },
}

/// Number of `s15Fixed16Number` parameters for each Table 68 type.
pub(crate) fn parametric_param_count(function: u16) -> Option<usize> {
    match function {
        0 => Some(1),
        1 => Some(3),
        2 => Some(4),
        3 => Some(5),
        4 => Some(7),
        _ => None,
    }
}

impl Curve {
    /// The sRGB-shaped type-3 parametric curve (Table 68 cites IEC
    /// 61966-2-1 for it): `g = 2.4`, `a = 1 / 1.055`, `b = 0.055 / 1.055`,
    /// `c = 1 / 12.92`, `d = 0.04045` — the decode constants of
    /// `docs/text/opentype/srgb-transfer-function.md`.
    pub fn srgb() -> Self {
        Curve::Parametric {
            function: 3,
            params: [
                2.4,
                1.0 / 1.055,
                0.055 / 1.055,
                1.0 / 12.92,
                0.04045,
                0.0,
                0.0,
            ],
        }
    }

    /// Construct a parametric curve, validating the function type.
    /// `params` lists `g a b c d e f` (missing trailing values are 0).
    pub fn parametric(function: u16, params: &[f64]) -> Option<Self> {
        let n = parametric_param_count(function)?;
        if params.len() < n {
            return None;
        }
        let mut p = [0.0; 7];
        p[..n].copy_from_slice(&params[..n]);
        Some(Curve::Parametric {
            function,
            params: p,
        })
    }

    /// True when the curve is exactly `y = x`.
    pub fn is_identity(&self) -> bool {
        match self {
            Curve::Identity => true,
            Curve::Gamma(g) => *g == 1.0,
            Curve::Parametric {
                function: 0,
                params,
            } => params[0] == 1.0,
            _ => false,
        }
    }

    /// Evaluate the curve. `x` is clamped to `[0, 1]`; the result is
    /// clipped to `[0, 1]`.
    pub fn eval(&self, x: f64) -> f64 {
        let x = clamp01(x);
        let y = match self {
            Curve::Identity => x,
            Curve::Gamma(g) => x.powf(*g),
            Curve::Sampled(t) => sample_table(t, x),
            Curve::Parametric { function, params } => eval_parametric(*function, params, x),
        };
        clamp01(y)
    }

    /// Evaluate the inverse curve per ICC.1:2022 Annex F.1: `y` outside
    /// the curve's range maps to the closest reachable value; a flat run
    /// that does not end at the domain's end inverts to its highest `x`,
    /// one that does to its lowest `x`. A constant curve is not
    /// invertible; it returns 0.
    pub fn eval_inverse(&self, y: f64) -> f64 {
        let y = clamp01(y);
        let x = match self {
            Curve::Identity => y,
            Curve::Gamma(g) => {
                if *g > 0.0 {
                    y.powf(1.0 / *g)
                } else {
                    return invert_by_sampling(self, y);
                }
            }
            Curve::Sampled(t) => invert_table(t, y),
            Curve::Parametric { function, params } => match invert_parametric(*function, params, y)
            {
                Some(x) => x,
                None => return invert_by_sampling(self, y),
            },
        };
        clamp01(x)
    }
}

#[inline]
pub(crate) fn clamp01(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

/// Linear interpolation into a uniformly spaced table over `[0, 1]`.
pub(crate) fn sample_table(t: &[f64], x: f64) -> f64 {
    match t.len() {
        0 => x,
        1 => t[0],
        n => {
            let pos = x * (n - 1) as f64;
            let i = (pos.floor() as usize).min(n - 2);
            let f = pos - i as f64;
            t[i] + (t[i + 1] - t[i]) * f
        }
    }
}

/// `(a·x + b)^g` with a non-negative base (a negative base is outside
/// every Table 68 branch's intended domain; NOTE 1 calls the result
/// undefined, we pin it to 0).
#[inline]
fn pow_base(a: f64, b: f64, g: f64, x: f64) -> f64 {
    let base = a * x + b;
    if base <= 0.0 { 0.0 } else { base.powf(g) }
}

fn eval_parametric(function: u16, p: &[f64; 7], x: f64) -> f64 {
    let [g, a, b, c, d, e, f] = *p;
    match function {
        0 => x.powf(g),
        1 => {
            if a != 0.0 && x >= -b / a {
                pow_base(a, b, g, x)
            } else {
                0.0
            }
        }
        2 => {
            if a != 0.0 && x >= -b / a {
                pow_base(a, b, g, x) + c
            } else {
                c
            }
        }
        3 => {
            if x >= d {
                pow_base(a, b, g, x)
            } else {
                c * x
            }
        }
        4 => {
            if x >= d {
                pow_base(a, b, g, x) + e
            } else {
                c * x + f
            }
        }
        _ => x,
    }
}

/// Closed-form inverse for the increasing shapes of Table 68 (`g > 0`,
/// `a > 0`, and `c > 0` where a linear toe is used). `None` asks the
/// caller to fall back to numeric inversion.
fn invert_parametric(function: u16, p: &[f64; 7], y: f64) -> Option<f64> {
    let [g, a, b, c, d, e, f] = *p;
    if g <= 0.0 || (function != 0 && a <= 0.0) {
        return None;
    }
    let up = |v: f64| -> f64 {
        // inverse of (a·x + b)^g = v
        (v.max(0.0).powf(1.0 / g) - b) / a
    };
    Some(match function {
        0 => y.powf(1.0 / g),
        1 => {
            if y <= 0.0 {
                // flat run [0, -b/a] → 0: highest x of the run.
                (-b / a).max(0.0)
            } else {
                up(y)
            }
        }
        2 => {
            if y <= c {
                (-b / a).max(0.0)
            } else {
                up(y - c)
            }
        }
        // Types 3 and 4: when the linear toe and the power segment
        // overlap slightly at `d` (s15Fixed16 rounding of constants that
        // were meant to meet), the toe solution wins for every `y` the
        // toe reaches, so device values below `d` round-trip exactly.
        3 => {
            if c > 0.0 && y < c * d {
                y / c
            } else if y >= pow_base(a, b, g, d) {
                up(y).max(d)
            } else if c > 0.0 {
                d
            } else {
                return None;
            }
        }
        4 => {
            if c > 0.0 && y >= f && y < c * d + f {
                (y - f) / c
            } else if y >= pow_base(a, b, g, d) + e {
                up(y - e).max(d)
            } else if c > 0.0 {
                if y < f { 0.0 } else { d }
            } else {
                return None;
            }
        }
        _ => return None,
    })
}

/// Inverse of a sampled table (Annex F.1), monotonic in either
/// direction. Small non-monotonic wiggles are flattened by a running
/// extremum so the search stays well defined.
fn invert_table(t: &[f64], y: f64) -> f64 {
    let n = t.len();
    if n < 2 {
        return y;
    }
    let increasing = t[n - 1] >= t[0];
    if t[n - 1] == t[0] {
        return 0.0; // constant: not invertible
    }
    // Monotone envelope, oriented so it increases with the index.
    let env: Vec<f64> = if increasing {
        let mut m = f64::NEG_INFINITY;
        t.iter()
            .map(|&v| {
                m = m.max(v);
                m
            })
            .collect()
    } else {
        // Decreasing curve: walk from the end so the envelope rises as
        // the index falls, then index it backwards.
        let mut m = f64::NEG_INFINITY;
        let mut rev: Vec<f64> = t
            .iter()
            .rev()
            .map(|&v| {
                m = m.max(v);
                m
            })
            .collect();
        rev.reverse();
        rev
    };
    let step = 1.0 / (n - 1) as f64;
    if increasing {
        let lo = env[0];
        let hi = env[n - 1];
        let y = y.clamp(lo, hi);
        if y >= hi {
            // Lowest x of the final flat run (it ends at x_n).
            let mut k = n - 1;
            while k > 0 && env[k - 1] >= hi {
                k -= 1;
            }
            return k as f64 * step;
        }
        // Largest i with env[i] <= y (so env[i + 1] > y).
        let i = env.partition_point(|&v| v <= y) - 1;
        let (y0, y1) = (env[i], env[i + 1]);
        (i as f64 + (y - y0) / (y1 - y0)) * step
    } else {
        // env is non-increasing in the index. Mirror: index j = n-1-i.
        let hi = env[0];
        let lo = env[n - 1];
        let y = y.clamp(lo, hi);
        if y <= lo {
            // The final run (ending at x_n) holds the minimum: lowest x.
            let mut k = n - 1;
            while k > 0 && env[k - 1] <= lo {
                k -= 1;
            }
            return k as f64 * step;
        }
        // Largest i with env[i] >= y (env[i + 1] < y).
        let i = env.partition_point(|&v| v >= y) - 1;
        let (y0, y1) = (env[i], env[i + 1]);
        (i as f64 + (y0 - y) / (y0 - y1)) * step
    }
}

/// Numeric fallback: sample the forward curve on a fine grid and invert
/// the resulting table.
fn invert_by_sampling(c: &Curve, y: f64) -> f64 {
    const N: usize = 4096;
    let t: Vec<f64> = (0..N).map(|k| c.eval(k as f64 / (N - 1) as f64)).collect();
    clamp01(invert_table(&t, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_round_trip() {
        let c = Curve::srgb();
        for k in 0..=1000 {
            let x = k as f64 / 1000.0;
            let y = c.eval(x);
            assert!((c.eval_inverse(y) - x).abs() < 1e-9, "x={x}");
        }
        // Table 68 type 3 against the staged decode formula.
        let x: f64 = 0.5;
        let expect = ((x + 0.055) / 1.055).powf(2.4);
        assert!((c.eval(x) - expect).abs() < 1e-12);
        assert!((c.eval(0.02) - 0.02 / 12.92).abs() < 1e-12);
    }

    #[test]
    fn table_inverse_flat_runs() {
        // Flat start [0, 0.25] → 0, flat end [0.75, 1] → 1.
        let t = vec![0.0, 0.0, 0.5, 1.0, 1.0];
        let c = Curve::Sampled(t);
        assert_eq!(c.eval_inverse(0.0), 0.25); // highest x of a non-final run
        assert_eq!(c.eval_inverse(1.0), 0.75); // lowest x of the final run
        assert!((c.eval_inverse(0.25) - 0.375).abs() < 1e-12);
        // Decreasing curve.
        let d = Curve::Sampled(vec![1.0, 0.5, 0.0]);
        assert!((d.eval_inverse(0.25) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn parametric_inverses() {
        let cases = [
            Curve::parametric(0, &[2.2]).unwrap(),
            Curve::parametric(1, &[2.2, 1.1, -0.1]).unwrap(),
            Curve::parametric(2, &[2.2, 1.0, 0.0, 0.05]).unwrap(),
            Curve::parametric(4, &[2.4, 0.9, 0.1, 0.07, 0.05, 0.01, 0.002]).unwrap(),
        ];
        for c in &cases {
            for k in 1..100 {
                let x = k as f64 / 100.0;
                let y = c.eval(x);
                let back = c.eval(c.eval_inverse(y));
                assert!((back - y).abs() < 1e-9, "{c:?} x={x}");
            }
        }
    }
}
