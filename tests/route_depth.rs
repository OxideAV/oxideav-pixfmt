//! Route-depth audit: the planner must never silently lose precision,
//! alpha or colour on a staged conversion.
//!
//! For every ordered `(src, dst)` pair the crate supports, the route
//! `convert()` takes (direct, or `src → pivot → dst` as reported by
//! [`staged_pivot`]) is checked against four invariants:
//!
//! 1. **Depth.** When both endpoints carry more than 8 bits per channel
//!    (10 / 12 / 14 / 16-bit integer or binary32 float), every
//!    intermediate carries at least `min(depth(src), depth(dst))` bits.
//! 2. **Alpha.** When both endpoints carry alpha, no intermediate drops
//!    it.
//! 3. **Palette.** No route pivots through `Pal8`.
//! 4. **Float.** A float endpoint never pivots through a ≤ 16-bit
//!    integer format unless the other endpoint is itself integer.
//!
//! The audit is a permanent test. While the known violations are being
//! closed the test pins the exact list (`known::KNOWN`, which only ever
//! shrinks); it tightens to empty once the list is.

#[path = "route_depth/known.rs"]
mod known;

use oxideav_core::PixelFormat;
use oxideav_pixfmt::{staged_pivot, supports, supports_direct, FormatInfo};

/// Every PixelFormat variant (kept in discriminant order).
const ALL_FORMATS: &[PixelFormat] = &[
    PixelFormat::Yuv420P,
    PixelFormat::Yuv422P,
    PixelFormat::Yuv444P,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Gray8,
    PixelFormat::Pal8,
    PixelFormat::Bgr24,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gray16Le,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Yuv420P10Le,
    PixelFormat::Yuv422P10Le,
    PixelFormat::Yuv444P10Le,
    PixelFormat::Yuv420P12Le,
    PixelFormat::Yuv422P12Le,
    PixelFormat::Yuv444P12Le,
    PixelFormat::YuvJ420P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ444P,
    PixelFormat::Nv12,
    PixelFormat::Nv21,
    PixelFormat::Ya8,
    PixelFormat::Yuva420P,
    PixelFormat::MonoBlack,
    PixelFormat::MonoWhite,
    PixelFormat::Yuyv422,
    PixelFormat::Uyvy422,
    PixelFormat::Cmyk,
    PixelFormat::Yuv411P,
    PixelFormat::Gbrp10Le,
    PixelFormat::Gbrap10Le,
    PixelFormat::Gbrp12Le,
    PixelFormat::Gbrap12Le,
    PixelFormat::Gbrp14Le,
    PixelFormat::Gbrap14Le,
    PixelFormat::Yuv420P16Le,
    PixelFormat::Yuv422P16Le,
    PixelFormat::Yuv444P16Le,
    PixelFormat::Yuva422P,
    PixelFormat::Yuva444P,
    PixelFormat::Yuva422P10Le,
    PixelFormat::Yuva422P12Le,
    PixelFormat::Yuva444P10Le,
    PixelFormat::Yuva444P12Le,
    PixelFormat::Yuva422P16Le,
    PixelFormat::Yuva444P16Le,
    PixelFormat::Gbrp8,
    PixelFormat::Gbrp16Le,
    PixelFormat::Gbrap16Le,
    PixelFormat::Yuva420P10Le,
    PixelFormat::Yuva420P12Le,
    PixelFormat::Yuva420P16Le,
    PixelFormat::Gbrap8,
    PixelFormat::Ya16Le,
    PixelFormat::CmykInverted,
    PixelFormat::Yuv440P,
    PixelFormat::Yuv440P10Le,
    PixelFormat::Yuv440P12Le,
    PixelFormat::Yuv440P16Le,
    PixelFormat::GrayF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::RgbaF32Le,
    PixelFormat::GbrpF32Le,
    PixelFormat::GbrapF32Le,
];

/// Bits per channel a format carries: `FormatInfo::bit_depth` (32 for
/// the binary32 float family, 1 for the mono bitmaps).
fn depth(f: PixelFormat) -> u32 {
    FormatInfo::of(f).bit_depth as u32
}

fn has_alpha(f: PixelFormat) -> bool {
    FormatInfo::of(f).has_alpha
}

fn is_float(f: PixelFormat) -> bool {
    f.is_float()
}

/// One violation of the route invariants: the pair, its pivot and the
/// rule it breaks.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Violation {
    src: PixelFormat,
    pivot: PixelFormat,
    dst: PixelFormat,
    rule: &'static str,
}

/// Every `(src, pivot, dst)` triple whose staged route breaks one of
/// the four invariants, in format-table order.
fn violations() -> Vec<Violation> {
    let mut out = Vec::new();
    for &src in ALL_FORMATS {
        for &dst in ALL_FORMATS {
            if src == dst || supports_direct(src, dst) {
                continue;
            }
            assert!(supports(src, dst), "{src:?} → {dst:?} is not supported");
            let pivot = staged_pivot(src, dst)
                .unwrap_or_else(|| panic!("{src:?} → {dst:?}: supported but no route reported"));
            let mut push = |rule: &'static str| {
                out.push(Violation {
                    src,
                    pivot,
                    dst,
                    rule,
                })
            };
            if pivot == PixelFormat::Pal8 {
                push("palette pivot");
            }
            let (ds, dd, dp) = (depth(src), depth(dst), depth(pivot));
            if ds > 8 && dd > 8 && dp < ds.min(dd) {
                push("depth");
            }
            if has_alpha(src) && has_alpha(dst) && !has_alpha(pivot) {
                push("alpha");
            }
            if (is_float(src) || is_float(dst))
                && !is_float(pivot)
                && dp <= 16
                && is_float(src)
                && is_float(dst)
            {
                push("float through integer");
            }
        }
    }
    out
}

/// The route-depth invariants hold on every supported pair not in the
/// pinned known list — and the known list never grows.
#[test]
fn every_staged_route_preserves_depth_alpha_and_colour() {
    let found = violations();
    let mut report = String::new();
    let mut unexpected = 0usize;
    for v in &found {
        let pinned = known::KNOWN
            .iter()
            .any(|&(s, p, d)| s == v.src && p == v.pivot && d == v.dst);
        if !pinned {
            unexpected += 1;
        }
        report.push_str(&format!(
            "  {:?} → {:?} → {:?} [{}]{}\n",
            v.src,
            v.pivot,
            v.dst,
            v.rule,
            if pinned { "" } else { " NEW" }
        ));
    }
    eprintln!(
        "route-depth audit: {} violations ({} pinned, {} new)",
        found.len(),
        found.len() - unexpected,
        unexpected
    );
    assert_eq!(
        unexpected,
        0,
        "{} route-depth violations ({unexpected} not in the known list):\n{report}",
        found.len()
    );
    // Closed pairs must leave the known list (it only shrinks).
    let stale: Vec<_> = known::KNOWN
        .iter()
        .filter(|&&(s, p, d)| {
            !found
                .iter()
                .any(|v| v.src == s && v.pivot == p && v.dst == d)
        })
        .collect();
    assert!(
        stale.is_empty(),
        "known-violation entries no longer observed (remove them): {stale:?}"
    );
}

/// The full matrix stays closed and `staged_pivot` agrees with
/// `supports_direct` on every pair.
#[test]
fn staged_pivot_agrees_with_supports_direct() {
    let mut direct = 0usize;
    let mut staged = 0usize;
    for &src in ALL_FORMATS {
        for &dst in ALL_FORMATS {
            if src == dst {
                assert_eq!(staged_pivot(src, dst), None);
                continue;
            }
            assert!(supports(src, dst), "{src:?} → {dst:?} unsupported");
            match staged_pivot(src, dst) {
                None => {
                    assert!(supports_direct(src, dst), "{src:?} → {dst:?}");
                    direct += 1;
                }
                Some(p) => {
                    assert!(!supports_direct(src, dst), "{src:?} → {dst:?}");
                    assert!(supports_direct(src, p) && supports_direct(p, dst));
                    staged += 1;
                }
            }
        }
    }
    assert_eq!(direct + staged, ALL_FORMATS.len() * (ALL_FORMATS.len() - 1));
    eprintln!("route census: {direct} direct, {staged} staged");
}
