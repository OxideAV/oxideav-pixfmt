//! Deep Yuva family suite — the 10/12/16-bit planar YUV +
//! full-resolution-alpha formats added in oxideav-core 0.1.31
//! (`Yuva422P10Le`/`12Le`/`16Le`, `Yuva444P10Le`/`12Le`/`16Le`) and
//! 0.1.33 (`Yuva420P10Le`/`12Le`/`16Le` — the remaining 4:2:0 siting),
//! wired through the computed planar-family dispatch tier.
//!
//! Invariants mirror the established 8-bit Yuva suites plus the deep
//! depth-ladder rules: the YUV planes behave exactly like the alpha-less
//! deep sibling's, the alpha plane is a full-resolution sample carrier
//! at the same depth (bit-exact across same-depth moves, widened /
//! truncated across depth moves with the crate-wide MSB-replicate /
//! truncate policy), and every ordered pair involving the nine deep
//! formats resolves through `convert()`.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, supports, supports_direct, ConvertOptions, FrameInfo};

/// MSB-replicating widen (the crate depth policy) for reference values.
fn widen(v: u16, from: u32, to: u32) -> u16 {
    let d = to - from;
    (((v as u32) << d) | ((v as u32) >> (from - d))) as u16
}

fn rd16(buf: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]])
}

fn le16_plane(vals: impl Iterator<Item = u16>) -> Vec<u8> {
    vals.flat_map(|v| v.to_le_bytes()).collect()
}

/// Deterministic deep planar YUVA frame: Y/A at full resolution, U/V on
/// the (wsub, hsub) chroma grid, every sample a LE16 word masked to
/// `bits` significant bits.
fn synth_deep_yuva(bits: u32, w: usize, h: usize, wsub: usize, hsub: usize) -> VideoFrame {
    let mask = ((1u32 << bits) - 1) as u16;
    let (cw, ch) = (w / wsub, h / hsub);
    let yp = le16_plane((0..w * h).map(|i| (i as u16).wrapping_mul(2311).wrapping_add(17) & mask));
    let ap = le16_plane((0..w * h).map(|i| (i as u16).wrapping_mul(4093).wrapping_add(5) & mask));
    let up = le16_plane((0..cw * ch).map(|i| (i as u16).wrapping_mul(929).wrapping_add(3) & mask));
    let vp =
        le16_plane((0..cw * ch).map(|i| (i as u16).wrapping_mul(1597).wrapping_add(41) & mask));
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: up,
            },
            VideoPlane {
                stride: cw * 2,
                data: vp,
            },
            VideoPlane {
                stride: w * 2,
                data: ap,
            },
        ],
    }
}

/// (format, bits, wsub, hsub) rows for the nine deep Yuva formats.
const DEEP: &[(PixelFormat, u32, usize, usize)] = &[
    (PixelFormat::Yuva420P10Le, 10, 2, 2),
    (PixelFormat::Yuva420P12Le, 12, 2, 2),
    (PixelFormat::Yuva420P16Le, 16, 2, 2),
    (PixelFormat::Yuva422P10Le, 10, 2, 1),
    (PixelFormat::Yuva422P12Le, 12, 2, 1),
    (PixelFormat::Yuva422P16Le, 16, 2, 1),
    (PixelFormat::Yuva444P10Le, 10, 1, 1),
    (PixelFormat::Yuva444P12Le, 12, 1, 1),
    (PixelFormat::Yuva444P16Le, 16, 1, 1),
];

/// The alpha-less deep sibling with the same grid and depth.
fn alpha_less_sibling(fmt: PixelFormat) -> PixelFormat {
    use PixelFormat as P;
    match fmt {
        P::Yuva420P10Le => P::Yuv420P10Le,
        P::Yuva420P12Le => P::Yuv420P12Le,
        P::Yuva420P16Le => P::Yuv420P16Le,
        P::Yuva422P10Le => P::Yuv422P10Le,
        P::Yuva422P12Le => P::Yuv422P12Le,
        P::Yuva422P16Le => P::Yuv422P16Le,
        P::Yuva444P10Le => P::Yuv444P10Le,
        P::Yuva444P12Le => P::Yuv444P12Le,
        P::Yuva444P16Le => P::Yuv444P16Le,
        _ => panic!("not a deep yuva format"),
    }
}

/// The 8-bit Yuva sibling with the same grid.
fn eight_bit_sibling(fmt: PixelFormat) -> PixelFormat {
    use PixelFormat as P;
    match fmt {
        P::Yuva420P10Le | P::Yuva420P12Le | P::Yuva420P16Le => P::Yuva420P,
        P::Yuva422P10Le | P::Yuva422P12Le | P::Yuva422P16Le => P::Yuva422P,
        P::Yuva444P10Le | P::Yuva444P12Le | P::Yuva444P16Le => P::Yuva444P,
        _ => panic!("not a deep yuva format"),
    }
}

/// Every ordered pair inside the full 12-member Yuva family (8-bit trio
/// + nine deep members) is a *direct* single-step conversion.
#[test]
fn yuva_family_all_pairs_direct() {
    use PixelFormat as P;
    const FAMILY: &[PixelFormat] = &[
        P::Yuva420P,
        P::Yuva422P,
        P::Yuva444P,
        P::Yuva420P10Le,
        P::Yuva420P12Le,
        P::Yuva420P16Le,
        P::Yuva422P10Le,
        P::Yuva422P12Le,
        P::Yuva422P16Le,
        P::Yuva444P10Le,
        P::Yuva444P12Le,
        P::Yuva444P16Le,
    ];
    for &a in FAMILY {
        for &b in FAMILY {
            assert!(supports_direct(a, b), "{a:?} → {b:?} must be direct");
        }
    }
}

/// 8-bit Yuva content round-trips losslessly through every deep member
/// (widen = MSB replication, narrow = truncation — exact inverses), on
/// all four planes including alpha.
#[test]
fn eight_bit_roundtrips_through_every_depth() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(deep_fmt, _, wsub, hsub) in DEEP {
        let eight = eight_bit_sibling(deep_fmt);
        // Build an 8-bit source from a masked deep frame's low bytes.
        let (cw, ch) = (w / wsub, h / hsub);
        let mk = |mul: usize, add: usize, n: usize| -> Vec<u8> {
            (0..n).map(|i| ((i * mul + add) & 0xFF) as u8).collect()
        };
        let src = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w,
                    data: mk(13, 7, w * h),
                },
                VideoPlane {
                    stride: cw,
                    data: mk(5, 3, cw * ch),
                },
                VideoPlane {
                    stride: cw,
                    data: mk(29, 17, cw * ch),
                },
                VideoPlane {
                    stride: w,
                    data: mk(7, 19, w * h),
                },
            ],
        };
        let info = FrameInfo::new(eight, w as u32, h as u32);
        let up = convert(&src, info, deep_fmt, &opts).expect("8 → deep");
        assert_eq!(up.planes.len(), 4, "{deep_fmt:?}");
        let back = convert(
            &up,
            FrameInfo::new(deep_fmt, w as u32, h as u32),
            eight,
            &opts,
        )
        .expect("deep → 8");
        for p in 0..4 {
            assert_eq!(
                back.planes[p].data, src.planes[p].data,
                "{deep_fmt:?} plane {p} round-trip"
            );
        }
    }
}

/// 10/12-bit content round-trips losslessly through the 16-bit member
/// of the same grid, alpha included, and peak maps to peak on the way
/// up.
#[test]
fn deep_roundtrips_through_16bit() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    let cases = [
        (
            PixelFormat::Yuva422P10Le,
            PixelFormat::Yuva422P16Le,
            10,
            2,
            1,
        ),
        (
            PixelFormat::Yuva422P12Le,
            PixelFormat::Yuva422P16Le,
            12,
            2,
            1,
        ),
        (
            PixelFormat::Yuva444P10Le,
            PixelFormat::Yuva444P16Le,
            10,
            1,
            1,
        ),
        (
            PixelFormat::Yuva444P12Le,
            PixelFormat::Yuva444P16Le,
            12,
            1,
            1,
        ),
        (
            PixelFormat::Yuva420P10Le,
            PixelFormat::Yuva420P16Le,
            10,
            2,
            2,
        ),
        (
            PixelFormat::Yuva420P12Le,
            PixelFormat::Yuva420P16Le,
            12,
            2,
            2,
        ),
    ];
    for (lo_fmt, hi_fmt, bits, wsub, hsub) in cases {
        let mut src = synth_deep_yuva(bits, w, h, wsub, hsub);
        // Force the rails onto the first two luma samples.
        src.planes[0].data[0..2].copy_from_slice(&0u16.to_le_bytes());
        let full = ((1u32 << bits) - 1) as u16;
        src.planes[0].data[2..4].copy_from_slice(&full.to_le_bytes());
        let info = FrameInfo::new(lo_fmt, w as u32, h as u32);
        let hi = convert(&src, info, hi_fmt, &opts).expect("lo → 16");
        assert_eq!(rd16(&hi.planes[0].data, 0), 0, "{lo_fmt:?} zero rail");
        assert_eq!(rd16(&hi.planes[0].data, 1), 65535, "{lo_fmt:?} peak rail");
        let back = convert(
            &hi,
            FrameInfo::new(hi_fmt, w as u32, h as u32),
            lo_fmt,
            &opts,
        )
        .expect("16 → lo");
        for p in 0..4 {
            assert_eq!(
                back.planes[p].data, src.planes[p].data,
                "{lo_fmt:?} plane {p} round-trip"
            );
        }
    }
}

/// Same-depth chroma resample inside the deep family: luma and alpha
/// are copied word-exact, and the resampled chroma equals what the
/// alpha-less deep sibling produces on the same planes (shared 16-bit
/// primitives — no divergent math).
#[test]
fn same_depth_resample_luma_alpha_exact_chroma_matches_sibling() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    let moves = [
        (
            PixelFormat::Yuva422P16Le,
            PixelFormat::Yuva444P16Le,
            16u32,
            2usize,
            1usize,
        ),
        (
            PixelFormat::Yuva444P16Le,
            PixelFormat::Yuva422P16Le,
            16,
            1,
            1,
        ),
        (
            PixelFormat::Yuva422P10Le,
            PixelFormat::Yuva444P10Le,
            10,
            2,
            1,
        ),
        (
            PixelFormat::Yuva444P12Le,
            PixelFormat::Yuva422P12Le,
            12,
            1,
            1,
        ),
        (
            PixelFormat::Yuva420P10Le,
            PixelFormat::Yuva444P10Le,
            10,
            2,
            2,
        ),
        (
            PixelFormat::Yuva422P16Le,
            PixelFormat::Yuva420P16Le,
            16,
            2,
            1,
        ),
    ];
    for (src_fmt, dst_fmt, bits, wsub, hsub) in moves {
        let yuva_src = synth_deep_yuva(bits, w, h, wsub, hsub);
        let yuv_src = VideoFrame {
            pts: None,
            planes: yuva_src.planes[..3].to_vec(),
        };
        let got = convert(
            &yuva_src,
            FrameInfo::new(src_fmt, w as u32, h as u32),
            dst_fmt,
            &opts,
        )
        .unwrap_or_else(|e| panic!("{src_fmt:?} → {dst_fmt:?}: {e:?}"));
        assert_eq!(got.planes.len(), 4);
        assert_eq!(got.planes[0].data, yuva_src.planes[0].data, "luma copy");
        assert_eq!(got.planes[3].data, yuva_src.planes[3].data, "alpha copy");
        let sibling = convert(
            &yuv_src,
            FrameInfo::new(alpha_less_sibling(src_fmt), w as u32, h as u32),
            alpha_less_sibling(dst_fmt),
            &opts,
        )
        .expect("sibling resample");
        assert_eq!(got.planes[1].data, sibling.planes[1].data, "U plane");
        assert_eq!(got.planes[2].data, sibling.planes[2].data, "V plane");
    }
}

/// Cross-depth + cross-subsampling inside the family: chroma is
/// resampled at the deeper of the two depths. `Yuva422P10Le →
/// Yuva444P12Le` widens first (10 → 12) and then duplicates
/// horizontally, so every output chroma pair carries the exact widened
/// word; luma and alpha are the straight widen.
#[test]
fn cross_depth_cross_subsampling_reference_model() {
    let opts = ConvertOptions::default();
    let (w, h) = (8usize, 4usize);
    let src = synth_deep_yuva(10, w, h, 2, 1);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva422P10Le, w as u32, h as u32),
        PixelFormat::Yuva444P12Le,
        &opts,
    )
    .expect("422P10 → 444P12");
    let cw = w / 2;
    for i in 0..w * h {
        assert_eq!(
            rd16(&out.planes[0].data, i),
            widen(rd16(&src.planes[0].data, i), 10, 12),
            "luma {i}"
        );
        assert_eq!(
            rd16(&out.planes[3].data, i),
            widen(rd16(&src.planes[3].data, i), 10, 12),
            "alpha {i}"
        );
    }
    for row in 0..h {
        for cc in 0..cw {
            let want_u = widen(rd16(&src.planes[1].data, row * cw + cc), 10, 12);
            let want_v = widen(rd16(&src.planes[2].data, row * cw + cc), 10, 12);
            for dx in 0..2 {
                assert_eq!(
                    rd16(&out.planes[1].data, row * w + cc * 2 + dx),
                    want_u,
                    "U ({row},{cc},{dx})"
                );
                assert_eq!(
                    rd16(&out.planes[2].data, row * w + cc * 2 + dx),
                    want_v,
                    "V ({row},{cc},{dx})"
                );
            }
        }
    }
}

/// 4:2:0 cross-depth + cross-subsampling reference model:
/// `Yuva420P10Le → Yuva444P12Le` widens first (10 → 12, chroma is
/// resampled at the deeper depth) and then replicates each chroma
/// sample over its 2×2 luma block, so every output chroma word in the
/// block equals the widened source word; luma and alpha are the
/// straight widen, untouched by the resampler.
#[test]
fn deep_420_cross_depth_upsample_reference_model() {
    let opts = ConvertOptions::default();
    let (w, h) = (8usize, 4usize);
    let src = synth_deep_yuva(10, w, h, 2, 2);
    let out = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva420P10Le, w as u32, h as u32),
        PixelFormat::Yuva444P12Le,
        &opts,
    )
    .expect("420P10 → 444P12");
    let (cw, ch) = (w / 2, h / 2);
    for i in 0..w * h {
        assert_eq!(
            rd16(&out.planes[0].data, i),
            widen(rd16(&src.planes[0].data, i), 10, 12),
            "luma {i}"
        );
        assert_eq!(
            rd16(&out.planes[3].data, i),
            widen(rd16(&src.planes[3].data, i), 10, 12),
            "alpha {i}"
        );
    }
    for cr in 0..ch {
        for cc in 0..cw {
            let want_u = widen(rd16(&src.planes[1].data, cr * cw + cc), 10, 12);
            let want_v = widen(rd16(&src.planes[2].data, cr * cw + cc), 10, 12);
            for dy in 0..2 {
                for dx in 0..2 {
                    let at = (cr * 2 + dy) * w + cc * 2 + dx;
                    assert_eq!(
                        rd16(&out.planes[1].data, at),
                        want_u,
                        "U ({cr},{cc},{dy},{dx})"
                    );
                    assert_eq!(
                        rd16(&out.planes[2].data, at),
                        want_v,
                        "V ({cr},{cc},{dy},{dx})"
                    );
                }
            }
        }
    }
}

/// Alpha drop to the alpha-less deep sibling copies the three colour
/// planes byte-identically; alpha synthesis from it produces full-scale
/// words at the format depth.
#[test]
fn alpha_drop_and_synthesis() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(deep_fmt, bits, wsub, hsub) in DEEP {
        let sibling = alpha_less_sibling(deep_fmt);
        let yuva_src = synth_deep_yuva(bits, w, h, wsub, hsub);
        let dropped = convert(
            &yuva_src,
            FrameInfo::new(deep_fmt, w as u32, h as u32),
            sibling,
            &opts,
        )
        .expect("yuva → yuv");
        assert_eq!(dropped.planes.len(), 3, "{deep_fmt:?}");
        for p in 0..3 {
            assert_eq!(
                dropped.planes[p].data, yuva_src.planes[p].data,
                "{deep_fmt:?} plane {p}"
            );
        }
        let yuv_src = VideoFrame {
            pts: None,
            planes: yuva_src.planes[..3].to_vec(),
        };
        let promoted = convert(
            &yuv_src,
            FrameInfo::new(sibling, w as u32, h as u32),
            deep_fmt,
            &opts,
        )
        .expect("yuv → yuva");
        assert_eq!(promoted.planes.len(), 4);
        for p in 0..3 {
            assert_eq!(promoted.planes[p].data, yuv_src.planes[p].data);
        }
        let full = ((1u32 << bits) - 1) as u16;
        for i in 0..w * h {
            assert_eq!(
                rd16(&promoted.planes[3].data, i),
                full,
                "{deep_fmt:?} opaque alpha {i}"
            );
        }
    }
}

/// Rgba interop: the colour math matches the alpha-less deep sibling's
/// route to Rgb24 channel-for-channel, and alpha rides across as the
/// top 8 of the deep word (truncation, per the crate depth policy).
#[test]
fn rgba_interop_matches_sibling_and_carries_alpha() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(deep_fmt, bits, wsub, hsub) in DEEP {
        let yuva_src = synth_deep_yuva(bits, w, h, wsub, hsub);
        let yuv_src = VideoFrame {
            pts: None,
            planes: yuva_src.planes[..3].to_vec(),
        };
        let rgba = convert(
            &yuva_src,
            FrameInfo::new(deep_fmt, w as u32, h as u32),
            PixelFormat::Rgba,
            &opts,
        )
        .expect("deep yuva → rgba");
        let rgb = convert(
            &yuv_src,
            FrameInfo::new(alpha_less_sibling(deep_fmt), w as u32, h as u32),
            PixelFormat::Rgb24,
            &opts,
        )
        .expect("deep yuv → rgb24");
        for p in 0..w * h {
            for c in 0..3 {
                assert_eq!(
                    rgba.planes[0].data[p * 4 + c],
                    rgb.planes[0].data[p * 3 + c],
                    "{deep_fmt:?} channel {c} at {p}"
                );
            }
            let want_a = (rd16(&yuva_src.planes[3].data, p) >> (bits - 8)) as u8;
            assert_eq!(
                rgba.planes[0].data[p * 4 + 3],
                want_a,
                "{deep_fmt:?} alpha at {p}"
            );
        }
    }
}

/// Rgba → deep Yuva splits the alpha out and widens it (8-bit alpha
/// round-trips exactly through any depth), and the whole
/// Rgba → Yuva444P16Le → Rgba trip keeps alpha bit-exact with colour
/// channels within the usual double-rounded matrix tolerance.
#[test]
fn from_rgba_alpha_widen_and_roundtrip() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 16usize);
    let mut data = Vec::with_capacity(w * h * 4);
    for j in 0..h {
        for i in 0..w {
            data.push(((i * 13 + j * 7) & 0xFF) as u8);
            data.push(((i * 31 + j * 3 + 9) & 0xFF) as u8);
            data.push(((i * 5 + j * 41 + 77) & 0xFF) as u8);
            data.push(((i * 17 + j * 29 + 100) & 0xFF) as u8);
        }
    }
    let rgba = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data,
        }],
    };
    let rgba_info = FrameInfo::new(PixelFormat::Rgba, w as u32, h as u32);
    for &(deep_fmt, bits, _, _) in DEEP {
        let yuva = convert(&rgba, rgba_info, deep_fmt, &opts).expect("rgba → deep yuva");
        assert_eq!(yuva.planes.len(), 4);
        for p in 0..w * h {
            let a8 = rgba.planes[0].data[p * 4 + 3];
            let want = if bits == 16 {
                a8 as u16 * 257
            } else {
                widen(a8 as u16, 8, bits)
            };
            assert_eq!(
                rd16(&yuva.planes[3].data, p),
                want,
                "{deep_fmt:?} widened alpha at {p}"
            );
        }
    }
    // Full round-trip through the 16-bit 4:4:4 member.
    let yuva = convert(&rgba, rgba_info, PixelFormat::Yuva444P16Le, &opts).expect("to 444P16");
    let back = convert(
        &yuva,
        FrameInfo::new(PixelFormat::Yuva444P16Le, w as u32, h as u32),
        PixelFormat::Rgba,
        &opts,
    )
    .expect("back to rgba");
    for p in 0..w * h {
        assert_eq!(
            back.planes[0].data[p * 4 + 3],
            rgba.planes[0].data[p * 4 + 3],
            "alpha at {p}"
        );
        for c in 0..3 {
            let a = rgba.planes[0].data[p * 4 + c] as i32;
            let b = back.planes[0].data[p * 4 + c] as i32;
            assert!(
                (a - b).abs() <= 2,
                "channel {c} at {p}: {a} vs {b} (limited-range matrix rounds twice)"
            );
        }
    }
}

/// Rgba64Le interop resolves (staged through the Rgba pivot: the colour
/// math is the crate's 8-bit matrix, so the deep packed side carries
/// the ×257-widened 8-bit result) and preserves the alpha top byte.
#[test]
fn rgba64_interop_via_rgba_pivot() {
    let opts = ConvertOptions::default();
    let (w, h) = (8usize, 8usize);
    for &(deep_fmt, bits, wsub, hsub) in DEEP {
        assert!(supports(deep_fmt, PixelFormat::Rgba64Le));
        assert!(supports(PixelFormat::Rgba64Le, deep_fmt));
        let src = synth_deep_yuva(bits, w, h, wsub, hsub);
        let out = convert(
            &src,
            FrameInfo::new(deep_fmt, w as u32, h as u32),
            PixelFormat::Rgba64Le,
            &opts,
        )
        .expect("deep yuva → rgba64");
        assert_eq!(out.planes[0].data.len(), w * h * 8);
        for p in 0..w * h {
            let a8 = rd16(&src.planes[3].data, p) >> (bits - 8);
            let got =
                u16::from_le_bytes([out.planes[0].data[p * 8 + 6], out.planes[0].data[p * 8 + 7]]);
            assert_eq!(got, a8 * 257, "{deep_fmt:?} alpha word at {p}");
        }
    }
}

/// Gray8 extraction and synthesis: deep Yuva → Gray8 equals the
/// alpha-less sibling's luma extraction; Gray8 → deep member sets the
/// exact neutral chroma mid-code and opaque full-scale alpha.
#[test]
fn gray8_extraction_and_synthesis() {
    let opts = ConvertOptions::default();
    let (w, h) = (16usize, 8usize);
    for &(deep_fmt, bits, wsub, hsub) in DEEP {
        let yuva_src = synth_deep_yuva(bits, w, h, wsub, hsub);
        let yuv_src = VideoFrame {
            pts: None,
            planes: yuva_src.planes[..3].to_vec(),
        };
        let a = convert(
            &yuva_src,
            FrameInfo::new(deep_fmt, w as u32, h as u32),
            PixelFormat::Gray8,
            &opts,
        )
        .expect("deep yuva → gray");
        let b = convert(
            &yuv_src,
            FrameInfo::new(alpha_less_sibling(deep_fmt), w as u32, h as u32),
            PixelFormat::Gray8,
            &opts,
        )
        .expect("deep yuv → gray");
        assert_eq!(a.planes[0].data, b.planes[0].data, "{deep_fmt:?}");
        assert_eq!(a.planes.len(), 1);

        // Synthesis: neutral chroma is the exact mid-code, alpha opaque.
        let gray = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w,
                data: (0..w * h).map(|i| (i & 0xFF) as u8).collect(),
            }],
        };
        let out = convert(
            &gray,
            FrameInfo::new(PixelFormat::Gray8, w as u32, h as u32),
            deep_fmt,
            &opts,
        )
        .expect("gray → deep yuva");
        assert_eq!(out.planes.len(), 4);
        let mid = (1u32 << (bits - 1)) as u16;
        let full = ((1u32 << bits) - 1) as u16;
        for i in 0..(w / wsub) * (h / hsub) {
            assert_eq!(rd16(&out.planes[1].data, i), mid, "{deep_fmt:?} U mid");
            assert_eq!(rd16(&out.planes[2].data, i), mid, "{deep_fmt:?} V mid");
        }
        for i in 0..w * h {
            assert_eq!(rd16(&out.planes[3].data, i), full, "{deep_fmt:?} opaque");
        }
    }
}

/// Full-matrix reachability: every ordered pair between each of the six
/// new formats and *every* other `PixelFormat` variant resolves.
#[test]
fn every_ordered_pair_involving_deep_yuva_resolves() {
    use PixelFormat as P;
    const ALL: &[PixelFormat] = &[
        P::Yuv420P,
        P::Yuv422P,
        P::Yuv444P,
        P::Rgb24,
        P::Rgba,
        P::Gray8,
        P::Pal8,
        P::Bgr24,
        P::Bgra,
        P::Argb,
        P::Abgr,
        P::Rgb48Le,
        P::Rgba64Le,
        P::Gray16Le,
        P::Gray10Le,
        P::Gray12Le,
        P::Yuv420P10Le,
        P::Yuv422P10Le,
        P::Yuv444P10Le,
        P::Yuv420P12Le,
        P::Yuv422P12Le,
        P::Yuv444P12Le,
        P::YuvJ420P,
        P::YuvJ422P,
        P::YuvJ444P,
        P::Nv12,
        P::Nv21,
        P::Ya8,
        P::Yuva420P,
        P::MonoBlack,
        P::MonoWhite,
        P::Yuyv422,
        P::Uyvy422,
        P::Cmyk,
        P::Yuv411P,
        P::Gbrp10Le,
        P::Gbrap10Le,
        P::Gbrp12Le,
        P::Gbrap12Le,
        P::Gbrp14Le,
        P::Gbrap14Le,
        P::Yuv420P16Le,
        P::Yuv422P16Le,
        P::Yuv444P16Le,
        P::Yuva422P,
        P::Yuva444P,
        P::Yuva422P10Le,
        P::Yuva422P12Le,
        P::Yuva444P10Le,
        P::Yuva444P12Le,
        P::Yuva422P16Le,
        P::Yuva444P16Le,
        P::Yuva420P10Le,
        P::Yuva420P12Le,
        P::Yuva420P16Le,
    ];
    for &(deep_fmt, _, _, _) in DEEP {
        for &other in ALL {
            assert!(supports(deep_fmt, other), "{deep_fmt:?} → {other:?}");
            assert!(supports(other, deep_fmt), "{other:?} → {deep_fmt:?}");
        }
    }
}
