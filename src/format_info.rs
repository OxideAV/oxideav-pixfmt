//! Per-format metadata: subsampling, planes, and bit depth.
//!
//! Callers that need to allocate or walk plane strides for a given
//! [`PixelFormat`] can look the format up here instead of open-coding
//! the decision tree.

use oxideav_core::PixelFormat;

/// Compact description of a pixel format's layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatInfo {
    /// Component bit depth (before packing). 8 for Rgb24, 16 for Gray16Le,
    /// 10 for Yuv420P10Le, …
    pub bit_depth: u8,
    /// Number of distinct planes — matches [`PixelFormat::plane_count`].
    pub planes: u8,
    /// Chroma horizontal subsampling factor (1 = no subsample). 2 for
    /// 4:2:x, 1 for 4:4:4 / non-YUV.
    pub chroma_w_sub: u8,
    /// Chroma vertical subsampling factor (1 = no subsample). 2 for
    /// 4:2:0, 1 for 4:2:2 / 4:4:4.
    pub chroma_h_sub: u8,
    /// True for any planar YUV-style layout.
    pub is_planar: bool,
    /// True when alpha is carried as part of the format (explicit or
    /// through a separate plane).
    pub has_alpha: bool,
    /// True for `Pal8`.
    pub is_palette: bool,
}

impl FormatInfo {
    /// Look up static metadata for `fmt`.
    pub const fn of(fmt: PixelFormat) -> Self {
        use PixelFormat as P;
        match fmt {
            // 8-bit YUV planar
            P::Yuv420P | P::YuvJ420P => Self::yuv(8, 2, 2),
            P::Yuv422P | P::YuvJ422P => Self::yuv(8, 2, 1),
            P::Yuv444P | P::YuvJ444P => Self::yuv(8, 1, 1),
            // 4:1:1 — luma at full res, chroma horizontally subsampled by 4.
            P::Yuv411P => Self::yuv(8, 4, 1),
            P::Yuv420P10Le => Self::yuv(10, 2, 2),
            P::Yuv422P10Le => Self::yuv(10, 2, 1),
            P::Yuv444P10Le => Self::yuv(10, 1, 1),
            P::Yuv420P12Le => Self::yuv(12, 2, 2),
            P::Yuv422P12Le => Self::yuv(12, 2, 1),
            P::Yuv444P12Le => Self::yuv(12, 1, 1),
            // Planar GBR / GBRA — 3-or-4 planes at 4:4:4 sampling, alpha
            // only on the `Gbrap*` variants. Each sample is stored as a
            // 16-bit LE word with the high bits zero; the bit-depth field
            // carries the *significant* bit count, matching how the
            // 10/12-bit YUV variants report theirs.
            P::Gbrp10Le => Self::gbr(10, false),
            P::Gbrap10Le => Self::gbr(10, true),
            P::Gbrp12Le => Self::gbr(12, false),
            P::Gbrap12Le => Self::gbr(12, true),
            P::Gbrp14Le => Self::gbr(14, false),
            P::Gbrap14Le => Self::gbr(14, true),
            P::Yuva420P => Self {
                bit_depth: 8,
                planes: 4,
                chroma_w_sub: 2,
                chroma_h_sub: 2,
                is_planar: true,
                has_alpha: true,
                is_palette: false,
            },
            P::Nv12 | P::Nv21 => Self {
                bit_depth: 8,
                planes: 2,
                chroma_w_sub: 2,
                chroma_h_sub: 2,
                is_planar: true,
                has_alpha: false,
                is_palette: false,
            },
            // Packed 4:2:2
            P::Yuyv422 | P::Uyvy422 => Self::packed(8, false),
            // RGB family
            P::Rgb24 | P::Bgr24 => Self::packed(8, false),
            P::Rgba | P::Bgra | P::Argb | P::Abgr => Self::packed(8, true),
            // CMYK — 4 components packed 8-bit, no alpha.
            P::Cmyk => Self::packed(8, false),
            P::Rgb48Le => Self::packed(16, false),
            P::Rgba64Le => Self::packed(16, true),
            // Gray
            P::Gray8 => Self::packed(8, false),
            P::Gray16Le => Self::packed(16, false),
            P::Gray10Le => Self::packed(10, false),
            P::Gray12Le => Self::packed(12, false),
            P::Ya8 => Self::packed(8, true),
            P::MonoBlack | P::MonoWhite => Self {
                bit_depth: 1,
                planes: 1,
                chroma_w_sub: 1,
                chroma_h_sub: 1,
                is_planar: false,
                has_alpha: false,
                is_palette: false,
            },
            // Palette
            P::Pal8 => Self {
                bit_depth: 8,
                planes: 1,
                chroma_w_sub: 1,
                chroma_h_sub: 1,
                is_planar: false,
                has_alpha: false,
                is_palette: true,
            },
            // The enum is `#[non_exhaustive]`; a future variant that
            // lands in oxideav-core without a matching arm here falls
            // back to a conservative "single packed 8-bit plane"
            // descriptor. Flagged in dev builds so we catch the gap.
            _ => {
                debug_assert!(false, "FormatInfo::of: unhandled PixelFormat variant");
                Self::packed(8, false)
            }
        }
    }

    const fn yuv(bits: u8, wsub: u8, hsub: u8) -> Self {
        Self {
            bit_depth: bits,
            planes: 3,
            chroma_w_sub: wsub,
            chroma_h_sub: hsub,
            is_planar: true,
            has_alpha: false,
            is_palette: false,
        }
    }

    const fn packed(bits: u8, alpha: bool) -> Self {
        Self {
            bit_depth: bits,
            planes: 1,
            chroma_w_sub: 1,
            chroma_h_sub: 1,
            is_planar: false,
            has_alpha: alpha,
            is_palette: false,
        }
    }

    /// Planar GBR / GBRA descriptor: 3 planes (G, B, R) — or 4 with an
    /// extra alpha plane — all at 4:4:4 sampling, no chroma subsample.
    const fn gbr(bits: u8, alpha: bool) -> Self {
        Self {
            bit_depth: bits,
            planes: if alpha { 4 } else { 3 },
            chroma_w_sub: 1,
            chroma_h_sub: 1,
            is_planar: true,
            has_alpha: alpha,
            is_palette: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::PixelFormat as P;

    #[test]
    fn gbr_planar_variants() {
        // 10-bit planar GBR — 3 planes, no alpha, no chroma subsample.
        let info = FormatInfo::of(P::Gbrp10Le);
        assert_eq!(info.bit_depth, 10);
        assert_eq!(info.planes, 3);
        assert_eq!(info.chroma_w_sub, 1);
        assert_eq!(info.chroma_h_sub, 1);
        assert!(info.is_planar);
        assert!(!info.has_alpha);
        assert!(!info.is_palette);

        // 10-bit planar GBR + alpha — 4 planes, alpha set.
        let info = FormatInfo::of(P::Gbrap10Le);
        assert_eq!(info.bit_depth, 10);
        assert_eq!(info.planes, 4);
        assert!(info.is_planar);
        assert!(info.has_alpha);

        // 12-bit no-alpha and alpha pair.
        let info = FormatInfo::of(P::Gbrp12Le);
        assert_eq!(info.bit_depth, 12);
        assert_eq!(info.planes, 3);
        assert!(!info.has_alpha);
        let info = FormatInfo::of(P::Gbrap12Le);
        assert_eq!(info.bit_depth, 12);
        assert_eq!(info.planes, 4);
        assert!(info.has_alpha);

        // 14-bit no-alpha and alpha pair.
        let info = FormatInfo::of(P::Gbrp14Le);
        assert_eq!(info.bit_depth, 14);
        assert_eq!(info.planes, 3);
        assert!(!info.has_alpha);
        let info = FormatInfo::of(P::Gbrap14Le);
        assert_eq!(info.bit_depth, 14);
        assert_eq!(info.planes, 4);
        assert!(info.has_alpha);
    }

    #[test]
    fn gbr_plane_count_matches_pixel_format() {
        // FormatInfo::planes must agree with the canonical
        // PixelFormat::plane_count for every GBR variant.
        for fmt in [
            P::Gbrp10Le,
            P::Gbrap10Le,
            P::Gbrp12Le,
            P::Gbrap12Le,
            P::Gbrp14Le,
            P::Gbrap14Le,
        ] {
            let info = FormatInfo::of(fmt);
            assert_eq!(
                info.planes as usize,
                fmt.plane_count(),
                "plane count mismatch for {fmt:?}",
            );
            assert_eq!(
                info.has_alpha,
                fmt.has_alpha(),
                "alpha flag mismatch for {fmt:?}",
            );
            assert_eq!(
                info.is_planar,
                fmt.is_planar(),
                "planar flag mismatch for {fmt:?}",
            );
        }
    }

    #[test]
    fn yuv_12bit_422_and_444_variants() {
        // 12-bit 4:2:2 — 3 planes, horizontal chroma subsample by 2.
        let info = FormatInfo::of(P::Yuv422P12Le);
        assert_eq!(info.bit_depth, 12);
        assert_eq!(info.planes, 3);
        assert_eq!(info.chroma_w_sub, 2);
        assert_eq!(info.chroma_h_sub, 1);
        assert!(info.is_planar);
        assert!(!info.has_alpha);
        assert!(!info.is_palette);

        // 12-bit 4:4:4 — 3 planes, no chroma subsample.
        let info = FormatInfo::of(P::Yuv444P12Le);
        assert_eq!(info.bit_depth, 12);
        assert_eq!(info.planes, 3);
        assert_eq!(info.chroma_w_sub, 1);
        assert_eq!(info.chroma_h_sub, 1);
        assert!(info.is_planar);
        assert!(!info.has_alpha);
    }

    #[test]
    fn yuv_12bit_mirrors_10bit_layout() {
        // 12-bit YUV variants must mirror the 10-bit equivalents in
        // every dimension except bit_depth — they only differ in
        // sample width.
        let pairs = [
            (P::Yuv420P10Le, P::Yuv420P12Le),
            (P::Yuv422P10Le, P::Yuv422P12Le),
            (P::Yuv444P10Le, P::Yuv444P12Le),
        ];
        for (a, b) in pairs {
            let ia = FormatInfo::of(a);
            let ib = FormatInfo::of(b);
            assert_eq!(ia.planes, ib.planes);
            assert_eq!(ia.chroma_w_sub, ib.chroma_w_sub);
            assert_eq!(ia.chroma_h_sub, ib.chroma_h_sub);
            assert_eq!(ia.is_planar, ib.is_planar);
            assert_eq!(ia.has_alpha, ib.has_alpha);
            assert_eq!(ia.is_palette, ib.is_palette);
            assert_eq!(ia.bit_depth, 10);
            assert_eq!(ib.bit_depth, 12);
        }
    }
}
