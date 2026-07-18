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
            // 16-bit planar YUV — same three-plane LE-word layout as the
            // 10/12-bit variants, but ALL 16 bits of every word are
            // significant (full-scale is 65535, per the oxideav-core
            // variant docs). `bit_depth` therefore reports 16.
            P::Yuv420P16Le => Self::yuv(16, 2, 2),
            P::Yuv422P16Le => Self::yuv(16, 2, 1),
            P::Yuv444P16Le => Self::yuv(16, 1, 1),
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
            // 8-bit planar YUV + a full-resolution alpha plane (plane 3
            // is always `w × h` regardless of the chroma subsampling).
            P::Yuva420P => Self::yuva(2, 2),
            P::Yuva422P => Self::yuva(2, 1),
            P::Yuva444P => Self::yuva(1, 1),
            P::Nv12 | P::Nv21 => Self {
                bit_depth: 8,
                planes: 2,
                chroma_w_sub: 2,
                chroma_h_sub: 2,
                is_planar: true,
                has_alpha: false,
                is_palette: false,
            },
            // Packed 4:2:2 — single byte-stream plane, but chroma is
            // horizontally subsampled by 2 (one U/V pair shared
            // between two luma samples), matching the field doc that
            // says "2 for 4:2:x".
            P::Yuyv422 | P::Uyvy422 => Self {
                bit_depth: 8,
                planes: 1,
                chroma_w_sub: 2,
                chroma_h_sub: 1,
                is_planar: false,
                has_alpha: false,
                is_palette: false,
            },
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

    /// 8-bit planar YUV + alpha descriptor: 4 planes (Y, U, V, A) where
    /// the alpha plane is always full resolution and only U/V follow the
    /// chroma subsampling factors.
    const fn yuva(wsub: u8, hsub: u8) -> Self {
        Self {
            bit_depth: 8,
            planes: 4,
            chroma_w_sub: wsub,
            chroma_h_sub: hsub,
            is_planar: true,
            has_alpha: true,
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

    /// Typed view of the chroma subsampling for a YUV-style format.
    ///
    /// Translates the raw `chroma_w_sub` / `chroma_h_sub` factor pair
    /// into the named 4:n:m convention the rest of the framework
    /// addresses by name — useful for callers that branch on the
    /// scheme (e.g. picking a packed-422 or NV12 dispatch path) without
    /// open-coding the `(w, h) == (2, 2)` etc. checks at every site.
    ///
    /// Returns:
    ///
    /// - [`ChromaSubsampling::None`] for formats with no chroma plane
    ///   (RGB family, grayscale, mono, palette, packed CMYK, packed
    ///   GBR(A) — anything where `chroma_w_sub == chroma_h_sub == 1`
    ///   and `is_planar == false`, or the planar GBR(A) variants where
    ///   the three colour planes are co-sited at 4:4:4 but the format
    ///   is RGB rather than YUV).
    /// - [`ChromaSubsampling::C444`] for 1×1 chroma siting on a YUV
    ///   carrier (`Yuv444P*`, `YuvJ444P`, `Yuva444P`).
    /// - [`ChromaSubsampling::C422`] for 2×1 chroma siting
    ///   (`Yuv422P*`, `YuvJ422P`, `Yuva422P`, `Yuyv422`, `Uyvy422`).
    /// - [`ChromaSubsampling::C420`] for 2×2 chroma siting
    ///   (`Yuv420P*`, `YuvJ420P`, `Nv12`, `Nv21`, `Yuva420P`).
    /// - [`ChromaSubsampling::C411`] for 4×1 chroma siting
    ///   (`Yuv411P` — native NTSC DV-25 layout).
    /// - [`ChromaSubsampling::Other`] for any future YUV variant whose
    ///   `(w, h)` pair this method doesn't yet name. Lets callers
    ///   detect "subsampled but not one of the four common schemes"
    ///   without losing the raw factors carried on the `FormatInfo`.
    pub const fn chroma_subsampling(&self) -> ChromaSubsampling {
        // Palette indices and any RGB/grayscale carrier with no chroma
        // subsampling map to None.
        if self.is_palette {
            return ChromaSubsampling::None;
        }
        // A chroma-less carrier shows up as (1, 1) on a non-planar
        // format: RGB / grayscale / mono / packed CMYK. Planar layouts
        // with (1, 1) are either YUV 4:4:4 or planar GBR(A) — both
        // collapse to C444 below (callers that need to distinguish
        // YUV-vs-GBR look at PixelFormat directly).
        if self.chroma_w_sub == 1 && self.chroma_h_sub == 1 && !self.is_planar {
            return ChromaSubsampling::None;
        }
        if self.chroma_w_sub == 1 && self.chroma_h_sub == 1 {
            // YUV 4:4:4 has 3 planes and no alpha (or 4 with alpha on
            // a future variant); planar GBR also has 3-or-4. The
            // difference is whether this is a YUV carrier. We can't
            // tell that from `FormatInfo` alone, so we map both 4:4:4
            // planar layouts to `C444` here — callers that need to
            // distinguish "luma+chroma 4:4:4" from "GBR 4:4:4" should
            // branch on the `PixelFormat` directly. This matches how
            // the rest of the conversion surface treats them: the
            // chroma-subsample factor pair is the conservative answer.
            return ChromaSubsampling::C444;
        }
        match (self.chroma_w_sub, self.chroma_h_sub) {
            (2, 2) => ChromaSubsampling::C420,
            (2, 1) => ChromaSubsampling::C422,
            (4, 1) => ChromaSubsampling::C411,
            _ => ChromaSubsampling::Other,
        }
    }

    /// True when the format carries chroma planes at less than full
    /// luma resolution (`C422`, `C420`, `C411`, or `Other`). Sugar
    /// over [`Self::chroma_subsampling`] for the common
    /// "is this losing chroma detail relative to 4:4:4?" predicate.
    pub const fn is_chroma_subsampled(&self) -> bool {
        matches!(
            self.chroma_subsampling(),
            ChromaSubsampling::C422
                | ChromaSubsampling::C420
                | ChromaSubsampling::C411
                | ChromaSubsampling::Other,
        )
    }
}

/// Named view of the 4:n:m chroma siting scheme on a YUV-style format.
///
/// Returned by [`FormatInfo::chroma_subsampling`]. The variants line up
/// with how the rest of the framework addresses chroma layouts by name
/// — see [`oxideav_core::PixelFormat`] for the per-variant docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// No chroma plane at all — RGB / grayscale / mono / palette /
    /// packed-CMYK carriers.
    None,
    /// 4:4:4 chroma siting (`chroma_w_sub == chroma_h_sub == 1`) on a
    /// planar carrier.
    C444,
    /// 4:2:2 chroma siting (horizontal subsample by 2, vertical full).
    C422,
    /// 4:2:0 chroma siting (horizontal AND vertical subsample by 2).
    C420,
    /// 4:1:1 chroma siting (horizontal subsample by 4, vertical full).
    /// Used by NTSC DV-25 and as a legal JPEG sampling layout.
    C411,
    /// Any other `(chroma_w_sub, chroma_h_sub)` pair on a YUV carrier
    /// that this enum doesn't yet name. Callers that need the exact
    /// factors should read [`FormatInfo::chroma_w_sub`] and
    /// [`FormatInfo::chroma_h_sub`] directly.
    Other,
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
    fn chroma_subsampling_named_view() {
        use ChromaSubsampling as C;

        // 4:2:0 — both 8-bit and full-range, planar and semi-planar
        // carriers map to C420.
        for fmt in [P::Yuv420P, P::YuvJ420P, P::Nv12, P::Nv21, P::Yuva420P] {
            assert_eq!(
                FormatInfo::of(fmt).chroma_subsampling(),
                C::C420,
                "expected C420 for {fmt:?}",
            );
        }
        // 10/12/16-bit 4:2:0 mirror the 8-bit answer — bit depth is
        // orthogonal to chroma siting.
        for fmt in [P::Yuv420P10Le, P::Yuv420P12Le, P::Yuv420P16Le] {
            assert_eq!(FormatInfo::of(fmt).chroma_subsampling(), C::C420);
        }

        // 4:2:2 — planar + packed carriers, all bit depths, plus the
        // alpha-bearing planar layout.
        for fmt in [
            P::Yuv422P,
            P::YuvJ422P,
            P::Yuyv422,
            P::Uyvy422,
            P::Yuv422P10Le,
            P::Yuv422P12Le,
            P::Yuv422P16Le,
            P::Yuva422P,
        ] {
            assert_eq!(
                FormatInfo::of(fmt).chroma_subsampling(),
                C::C422,
                "expected C422 for {fmt:?}",
            );
        }

        // 4:4:4 — all bit depths. Planar GBR(A) lands on C444 too
        // because its sample siting is 4:4:4 — callers that need to
        // distinguish YUV-vs-GBR look at PixelFormat directly.
        for fmt in [
            P::Yuv444P,
            P::YuvJ444P,
            P::Yuv444P10Le,
            P::Yuv444P12Le,
            P::Yuv444P16Le,
            P::Yuva444P,
            P::Gbrp10Le,
            P::Gbrap10Le,
            P::Gbrp12Le,
            P::Gbrap12Le,
            P::Gbrp14Le,
            P::Gbrap14Le,
        ] {
            assert_eq!(
                FormatInfo::of(fmt).chroma_subsampling(),
                C::C444,
                "expected C444 for {fmt:?}",
            );
        }

        // 4:1:1 — NTSC DV-25 layout.
        assert_eq!(FormatInfo::of(P::Yuv411P).chroma_subsampling(), C::C411,);

        // Chroma-less carriers — RGB family, grayscale, mono, palette,
        // packed CMYK. All map to None and `is_chroma_subsampled`
        // returns false.
        for fmt in [
            P::Rgb24,
            P::Bgr24,
            P::Rgba,
            P::Bgra,
            P::Argb,
            P::Abgr,
            P::Rgb48Le,
            P::Rgba64Le,
            P::Gray8,
            P::Gray16Le,
            P::Gray10Le,
            P::Gray12Le,
            P::Ya8,
            P::MonoBlack,
            P::MonoWhite,
            P::Pal8,
            P::Cmyk,
        ] {
            let info = FormatInfo::of(fmt);
            assert_eq!(
                info.chroma_subsampling(),
                C::None,
                "expected None for {fmt:?}",
            );
            assert!(
                !info.is_chroma_subsampled(),
                "{fmt:?} unexpectedly flagged as chroma-subsampled",
            );
        }

        // `is_chroma_subsampled` discriminates C422 / C420 / C411
        // (true) from C444 / None (false).
        assert!(FormatInfo::of(P::Yuv420P).is_chroma_subsampled());
        assert!(FormatInfo::of(P::Yuv422P).is_chroma_subsampled());
        assert!(FormatInfo::of(P::Yuv411P).is_chroma_subsampled());
        assert!(!FormatInfo::of(P::Yuv444P).is_chroma_subsampled());
        assert!(!FormatInfo::of(P::Gbrp10Le).is_chroma_subsampled());
    }

    #[test]
    fn chroma_subsampling_matches_raw_factors() {
        // Every variant the static `FormatInfo::of` table covers must
        // line up: chroma_subsampling()'s decision agrees with the
        // raw (chroma_w_sub, chroma_h_sub) pair.
        use ChromaSubsampling as C;
        let cases: &[(P, C)] = &[
            (P::Yuv420P, C::C420),
            (P::Yuv422P, C::C422),
            (P::Yuv444P, C::C444),
            (P::Yuv411P, C::C411),
            (P::Nv12, C::C420),
            (P::Yuva420P, C::C420),
            (P::Rgb24, C::None),
            (P::Gray8, C::None),
            (P::Gbrp10Le, C::C444),
        ];
        for &(fmt, expected) in cases {
            let info = FormatInfo::of(fmt);
            assert_eq!(info.chroma_subsampling(), expected);
            // Sanity: factors and the named view agree.
            match expected {
                C::C420 => {
                    assert_eq!(info.chroma_w_sub, 2);
                    assert_eq!(info.chroma_h_sub, 2);
                }
                C::C422 => {
                    assert_eq!(info.chroma_w_sub, 2);
                    assert_eq!(info.chroma_h_sub, 1);
                }
                C::C411 => {
                    assert_eq!(info.chroma_w_sub, 4);
                    assert_eq!(info.chroma_h_sub, 1);
                }
                C::C444 => {
                    assert_eq!(info.chroma_w_sub, 1);
                    assert_eq!(info.chroma_h_sub, 1);
                }
                C::None => {
                    // For chroma-less carriers the factors are also
                    // trivially 1×1.
                    assert_eq!(info.chroma_w_sub, 1);
                    assert_eq!(info.chroma_h_sub, 1);
                }
                C::Other => {}
            }
        }
    }

    #[test]
    fn yuv_16bit_mirrors_10bit_layout_at_full_width() {
        // The 16-bit trio must mirror the 10-bit layouts in every
        // dimension except bit_depth, which reports the full 16 (all 16
        // bits of every LE word are significant — full-scale 65535).
        let pairs = [
            (P::Yuv420P10Le, P::Yuv420P16Le),
            (P::Yuv422P10Le, P::Yuv422P16Le),
            (P::Yuv444P10Le, P::Yuv444P16Le),
        ];
        for (a, b) in pairs {
            let ia = FormatInfo::of(a);
            let ib = FormatInfo::of(b);
            assert_eq!(ia.planes, ib.planes);
            assert_eq!(ia.chroma_w_sub, ib.chroma_w_sub);
            assert_eq!(ia.chroma_h_sub, ib.chroma_h_sub);
            assert_eq!(ia.is_planar, ib.is_planar);
            assert_eq!(ia.has_alpha, ib.has_alpha);
            assert_eq!(ib.bit_depth, 16);
        }
        // Core agreement: plane count / alpha / planar flags.
        for fmt in [P::Yuv420P16Le, P::Yuv422P16Le, P::Yuv444P16Le] {
            let info = FormatInfo::of(fmt);
            assert_eq!(info.planes as usize, fmt.plane_count());
            assert_eq!(info.has_alpha, fmt.has_alpha());
            assert_eq!(info.is_planar, fmt.is_planar());
        }
    }

    #[test]
    fn yuva_family_shares_layout_with_alpha_less_siblings() {
        // Each Yuva* variant mirrors its Yuv* sibling's chroma grid and
        // adds exactly one (full-resolution, 8-bit) alpha plane.
        let pairs = [
            (P::Yuv420P, P::Yuva420P),
            (P::Yuv422P, P::Yuva422P),
            (P::Yuv444P, P::Yuva444P),
        ];
        for (yuv, yuva) in pairs {
            let iy = FormatInfo::of(yuv);
            let ia = FormatInfo::of(yuva);
            assert_eq!(ia.bit_depth, 8);
            assert_eq!(ia.planes, iy.planes + 1, "{yuva:?}");
            assert_eq!(ia.chroma_w_sub, iy.chroma_w_sub);
            assert_eq!(ia.chroma_h_sub, iy.chroma_h_sub);
            assert!(ia.is_planar);
            assert!(ia.has_alpha);
            assert!(!ia.is_palette);
            // Core agreement.
            assert_eq!(ia.planes as usize, yuva.plane_count());
            assert_eq!(ia.has_alpha, yuva.has_alpha());
            assert_eq!(ia.is_planar, yuva.is_planar());
        }
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
