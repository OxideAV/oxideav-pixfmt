//! The scene-referred float family (`GrayF32Le` / `RgbF32Le` /
//! `RgbaF32Le` / `GbrpF32Le` / `GbrapF32Le`, oxideav-core 0.1.35):
//! value semantics of every hop, exactness where the crate promises it,
//! and the fidelity of the staged routes.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, float, supports, supports_direct, ColorSpace, ConvertOptions, FormatInfo, FrameInfo,
};

const FLOATS: [PixelFormat; 5] = [
    PixelFormat::GrayF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::RgbaF32Le,
    PixelFormat::GbrpF32Le,
    PixelFormat::GbrapF32Le,
];

const INTS: [PixelFormat; 18] = [
    PixelFormat::Gray8,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Gray16Le,
    PixelFormat::Rgb24,
    PixelFormat::Rgba,
    PixelFormat::Rgb48Le,
    PixelFormat::Rgba64Le,
    PixelFormat::Gbrp8,
    PixelFormat::Gbrap8,
    PixelFormat::Gbrp10Le,
    PixelFormat::Gbrap10Le,
    PixelFormat::Gbrp12Le,
    PixelFormat::Gbrap12Le,
    PixelFormat::Gbrp14Le,
    PixelFormat::Gbrap14Le,
    PixelFormat::Gbrp16Le,
    PixelFormat::Gbrap16Le,
];

fn opts() -> ConvertOptions {
    ConvertOptions::default()
}

fn conv(frame: &VideoFrame, src: PixelFormat, dst: PixelFormat, w: usize, h: usize) -> VideoFrame {
    convert(frame, FrameInfo::new(src, w as u32, h as u32), dst, &opts())
        .unwrap_or_else(|e| panic!("{src:?} → {dst:?}: {e:?}"))
}

/// Build an `RgbaF32Le` frame from a per-pixel generator (`pad` extra
/// stride bytes).
fn rgba_f32(w: usize, h: usize, pad: usize, f: impl Fn(usize, usize) -> [f32; 4]) -> VideoFrame {
    let stride = w * 16 + pad;
    let mut data = vec![0u8; stride * h];
    for y in 0..h {
        for x in 0..w {
            let p = f(x, y);
            for (c, v) in p.iter().enumerate() {
                let off = y * stride + x * 16 + c * 4;
                data[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

fn f32s(plane: &[u8]) -> Vec<f32> {
    plane
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A smooth in-range generator with distinct channels.
fn smooth(x: usize, y: usize) -> [f32; 4] {
    [
        (x as f32 * 0.07 + y as f32 * 0.011) % 1.0,
        (x as f32 * 0.013 + y as f32 * 0.09) % 1.0,
        (x as f32 * 0.031 + 0.2) % 1.0,
        ((x + y) as f32 * 0.05) % 1.0,
    ]
}

#[test]
fn format_info_and_coverage() {
    for &f in &FLOATS {
        let info = FormatInfo::of(f);
        assert_eq!(info.bit_depth, 32, "{f:?}");
        assert!(f.is_float());
        assert_eq!(info.planes as usize, f.plane_count());
        assert_eq!(info.has_alpha, f.has_alpha());
        assert_eq!(info.is_planar, f.is_planar());
        for &g in &FLOATS {
            assert!(supports_direct(f, g), "{f:?} → {g:?}");
        }
        for &i in &INTS {
            assert!(supports_direct(f, i), "{f:?} → {i:?}");
            assert!(supports_direct(i, f), "{i:?} → {f:?}");
        }
        for deep in [
            PixelFormat::Yuv444P16Le,
            PixelFormat::Yuv420P16Le,
            PixelFormat::Yuv440P16Le,
            PixelFormat::Yuva422P16Le,
        ] {
            assert!(supports_direct(f, deep), "{f:?} → {deep:?}");
            assert!(supports_direct(deep, f), "{deep:?} → {f:?}");
        }
        // Everything else is closed through one pivot.
        for other in [
            PixelFormat::Yuv420P,
            PixelFormat::Yuv444P10Le,
            PixelFormat::YuvJ420P,
            PixelFormat::Nv12,
            PixelFormat::Yuyv422,
            PixelFormat::Pal8,
            PixelFormat::MonoWhite,
            PixelFormat::Cmyk,
            PixelFormat::Ya16Le,
            PixelFormat::Bgra,
        ] {
            assert!(supports(f, other), "{f:?} → {other:?}");
            assert!(supports(other, f), "{other:?} → {f:?}");
        }
    }
}

/// Every integer code at every depth survives integer → float →
/// integer exactly, on every shape.
#[test]
fn int_float_int_round_trip_is_exact() {
    let (w, h) = (16, 4);
    for &i in &INTS {
        let info = FormatInfo::of(i);
        let bits = info.bit_depth as u32;
        let sb = if bits > 8 { 2 } else { 1 };
        let comps = if info.is_planar {
            1
        } else if info.has_alpha {
            4
        } else if i == PixelFormat::Rgb24 || i == PixelFormat::Rgb48Le {
            3
        } else {
            1
        };
        let mut seed = 7u32 ^ (i as u32);
        let mk = || {
            let mut data = vec![0u8; w * h * comps * sb];
            for k in 0..w * h * comps {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let code = (seed >> 8) & ((1u32 << bits) - 1);
                if sb == 2 {
                    data[k * 2..k * 2 + 2].copy_from_slice(&(code as u16).to_le_bytes());
                } else {
                    data[k] = code as u8;
                }
            }
            VideoPlane {
                stride: w * comps * sb,
                data,
            }
        };
        let mut planes = Vec::new();
        let mut mk = mk;
        for _ in 0..info.planes {
            planes.push(mk());
        }
        // Pin the extremes too.
        planes[0].data[..sb].fill(0);
        let full = ((1u32 << bits) - 1) as u16;
        if sb == 2 {
            planes[0].data[2..4].copy_from_slice(&full.to_le_bytes());
        } else {
            planes[0].data[1] = full as u8;
        }
        let src = VideoFrame { pts: None, planes };
        // The float shape must be able to carry the integer shape.
        let carriers: &[PixelFormat] = if info.is_planar || info.has_alpha {
            &[PixelFormat::RgbaF32Le, PixelFormat::GbrapF32Le]
        } else if comps == 1 {
            &FLOATS
        } else {
            &[
                PixelFormat::RgbF32Le,
                PixelFormat::GbrpF32Le,
                PixelFormat::RgbaF32Le,
            ]
        };
        for &f in carriers {
            let up = conv(&src, i, f, w, h);
            let back = conv(&up, f, i, w, h);
            for p in 0..info.planes as usize {
                assert_eq!(
                    back.planes[p].data, src.planes[p].data,
                    "{i:?} via {f:?} plane {p}"
                );
            }
        }
    }
}

/// Normalisation anchors: 0 → 0.0, full-scale → exactly 1.0, mid-code
/// → code / full, at every depth.
#[test]
fn normalisation_anchors() {
    let w = 3;
    for (fmt, bits) in [
        (PixelFormat::Gray8, 8u32),
        (PixelFormat::Gray10Le, 10),
        (PixelFormat::Gray12Le, 12),
        (PixelFormat::Gray16Le, 16),
    ] {
        let full = (1u32 << bits) - 1;
        let codes = [0u32, full, full / 3];
        let sb = if bits > 8 { 2 } else { 1 };
        let mut data = vec![0u8; w * sb];
        for (k, &c) in codes.iter().enumerate() {
            if sb == 2 {
                data[k * 2..k * 2 + 2].copy_from_slice(&(c as u16).to_le_bytes());
            } else {
                data[k] = c as u8;
            }
        }
        let src = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w * sb,
                data,
            }],
        };
        let f = conv(&src, fmt, PixelFormat::GrayF32Le, w, 1);
        let got = f32s(&f.planes[0].data);
        assert_eq!(got[0], 0.0, "{fmt:?}");
        assert_eq!(got[1], 1.0, "{fmt:?}");
        assert_eq!(got[2], (full / 3) as f32 / full as f32, "{fmt:?}");
    }
}

/// Float → integer saturates: > 1 clamps to full-scale, < 0 to zero,
/// NaN to zero, ±∞ to the respective end — never a panic.
#[test]
fn quantisation_saturates() {
    let vals = [
        1.5f32,
        -0.5,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.5,
    ];
    let w = vals.len();
    let src = rgba_f32(w, 1, 0, |x, _| [vals[x], vals[x], vals[x], vals[x]]);
    for (fmt, full) in [
        (PixelFormat::Rgba, 255u32),
        (PixelFormat::Rgba64Le, 65535),
        (PixelFormat::Gbrap8, 255),
        (PixelFormat::Gbrap12Le, 4095),
        (PixelFormat::Gray16Le, 65535),
    ] {
        let out = conv(&src, PixelFormat::RgbaF32Le, fmt, w, 1);
        let info = FormatInfo::of(fmt);
        let sb = if info.bit_depth > 8 { 2 } else { 1 };
        let comps = if info.is_planar {
            1
        } else if info.has_alpha {
            4
        } else {
            1
        };
        let code = |px: usize| -> u32 {
            let d = &out.planes[0].data;
            let k = px * comps;
            if sb == 2 {
                u16::from_le_bytes([d[k * 2], d[k * 2 + 1]]) as u32
            } else {
                d[k] as u32
            }
        };
        assert_eq!(code(0), full, "{fmt:?} >1");
        assert_eq!(code(1), 0, "{fmt:?} <0");
        assert_eq!(code(2), 0, "{fmt:?} NaN");
        assert_eq!(code(3), full, "{fmt:?} +inf");
        assert_eq!(code(4), 0, "{fmt:?} -inf");
        assert_eq!(
            code(5),
            float::f32_to_unorm(0.5, info.bit_depth as u32),
            "{fmt:?} 0.5"
        );
    }
}

/// Float → float moves never clamp: out-of-range light (speculars above
/// 1.0, negative excursions) and NaN payloads survive packed ↔ planar
/// and alpha add / drop bit-for-bit.
#[test]
fn float_to_float_preserves_out_of_range_light() {
    let (w, h) = (5, 3);
    let src = rgba_f32(w, h, 4, |x, y| {
        [
            4.5 + x as f32,
            -0.75 * y as f32,
            if x == 2 { f32::NAN } else { 0.25 },
            0.5,
        ]
    });
    // Packed → planar → packed.
    let planar = conv(&src, PixelFormat::RgbaF32Le, PixelFormat::GbrapF32Le, w, h);
    assert_eq!(planar.planes.len(), 4);
    let back = conv(
        &planar,
        PixelFormat::GbrapF32Le,
        PixelFormat::RgbaF32Le,
        w,
        h,
    );
    let tight: Vec<u8> = (0..h)
        .flat_map(|y| src.planes[0].data[y * (w * 16 + 4)..y * (w * 16 + 4) + w * 16].to_vec())
        .collect();
    assert_eq!(back.planes[0].data, tight);
    // Plane order is G, B, R, A.
    let g = f32s(&planar.planes[0].data);
    let r = f32s(&planar.planes[2].data);
    assert_eq!(g[1], -0.0 * 0.75);
    assert_eq!(r[1], 5.5);
    // Alpha drop then re-add synthesises 1.0; colour untouched.
    let rgb = conv(&src, PixelFormat::RgbaF32Le, PixelFormat::RgbF32Le, w, h);
    let rgba = conv(&rgb, PixelFormat::RgbF32Le, PixelFormat::RgbaF32Le, w, h);
    let a = f32s(&rgba.planes[0].data);
    let t = f32s(&tight);
    for i in 0..w * h {
        assert_eq!(a[i * 4].to_bits(), t[i * 4].to_bits());
        assert_eq!(a[i * 4 + 1].to_bits(), t[i * 4 + 1].to_bits());
        assert_eq!(a[i * 4 + 2].to_bits(), t[i * 4 + 2].to_bits());
        assert_eq!(a[i * 4 + 3], 1.0);
    }
}

/// Gray broadcast and linear luminance projection: gray → RGB gives
/// r = g = b = v (alpha 1.0); RGB → gray is the Kr / Kg / Kb row of the
/// selected primaries in linear light, so a neutral pixel projects to
/// itself and a 2× brighter pixel to 2× the luminance.
#[test]
fn gray_broadcast_and_linear_luminance() {
    let (w, h) = (4, 2);
    let gray = VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data: [0.0f32, 0.5, 1.0, 3.0, 0.1, 0.2, 0.3, 0.4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        }],
    };
    let rgba = conv(&gray, PixelFormat::GrayF32Le, PixelFormat::RgbaF32Le, w, h);
    let v = f32s(&rgba.planes[0].data);
    assert_eq!(&v[12..16], &[3.0, 3.0, 3.0, 1.0]);
    let back = conv(&rgba, PixelFormat::RgbaF32Le, PixelFormat::GrayF32Le, w, h);
    let g = f32s(&back.planes[0].data);
    for (a, b) in g.iter().zip(f32s(&gray.planes[0].data)) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
    // Coloured pixel: exact Kr/Kg/Kb row under BT.709 (the default
    // primaries) and BT.601 when asked.
    let src = rgba_f32(1, 1, 0, |_, _| [2.0, 0.5, 0.25, 1.0]);
    for (cs, kr, kb) in [
        (ColorSpace::Bt709Limited, 0.2126f32, 0.0722f32),
        (ColorSpace::Bt601Full, 0.299, 0.114),
    ] {
        let out = convert(
            &src,
            FrameInfo::new(PixelFormat::RgbaF32Le, 1, 1),
            PixelFormat::GrayF32Le,
            &ConvertOptions {
                color_space: cs,
                ..Default::default()
            },
        )
        .unwrap();
        let y = f32s(&out.planes[0].data)[0];
        let want = kr * 2.0 + (1.0 - kr - kb) * 0.5 + kb * 0.25;
        assert!((y - want).abs() < 1e-6, "{cs:?}: {y} vs {want}");
    }
}

/// Float ↔ the 16-bit planar tier runs through the packed 16-bit RGB(A)
/// intermediate and the Q30 deep matrix; pinned bit-for-bit against
/// the explicit two-step route, at every siting and with alpha.
#[test]
fn deep_yuv_matches_explicit_rgb48_route() {
    let (w, h) = (8, 4);
    let src = rgba_f32(w, h, 0, smooth);
    for (yuv, packed, alpha) in [
        (PixelFormat::Yuv444P16Le, PixelFormat::Rgb48Le, false),
        (PixelFormat::Yuv420P16Le, PixelFormat::Rgb48Le, false),
        (PixelFormat::Yuv440P16Le, PixelFormat::Rgb48Le, false),
        (PixelFormat::Yuva422P16Le, PixelFormat::Rgba64Le, true),
        (PixelFormat::Yuva420P16Le, PixelFormat::Rgba64Le, true),
    ] {
        for f in [PixelFormat::RgbaF32Le, PixelFormat::GbrpF32Le] {
            let fsrc = conv(&src, PixelFormat::RgbaF32Le, f, w, h);
            let direct = conv(&fsrc, f, yuv, w, h);
            let via = conv(&conv(&fsrc, f, packed, w, h), packed, yuv, w, h);
            assert_eq!(direct.planes.len(), via.planes.len());
            for p in 0..direct.planes.len() {
                assert_eq!(
                    direct.planes[p].data, via.planes[p].data,
                    "{f:?} → {yuv:?} plane {p}"
                );
            }
            let back = conv(&direct, yuv, f, w, h);
            let via = conv(&conv(&direct, yuv, packed, w, h), packed, f, w, h);
            for p in 0..back.planes.len() {
                assert_eq!(
                    back.planes[p].data, via.planes[p].data,
                    "{yuv:?} → {f:?} plane {p}"
                );
            }
            if alpha && f == PixelFormat::RgbaF32Le {
                // Alpha rides the 16-bit plane: exact for 16-bit codes.
                let a: Vec<f32> = f32s(&back.planes[0].data).chunks(4).map(|p| p[3]).collect();
                let src_a: Vec<f32> = f32s(&src.planes[0].data).chunks(4).map(|p| p[3]).collect();
                for (x, y) in a.iter().zip(src_a) {
                    assert!((x - y).abs() <= 0.5 / 65535.0, "{x} vs {y}");
                }
            }
        }
    }
}

/// Staged routes from the 8 / 10 / 12-bit YUV families reach the float
/// family through the 16-bit tier (exact widen + deep matrix), never
/// through an 8-bit RGB hop.
#[test]
fn staged_routes_keep_full_precision() {
    let (w, h) = (8, 4);
    let src = rgba_f32(w, h, 0, smooth);
    // Float → Yuv420P10Le → float: compare with the explicit
    // 16-bit-tier route.
    let y10 = conv(&src, PixelFormat::RgbaF32Le, PixelFormat::Yuv420P10Le, w, h);
    let via = conv(
        &conv(
            &src,
            PixelFormat::RgbaF32Le,
            PixelFormat::Yuva420P16Le,
            w,
            h,
        ),
        PixelFormat::Yuva420P16Le,
        PixelFormat::Yuv420P10Le,
        w,
        h,
    );
    for p in 0..3 {
        assert_eq!(y10.planes[p].data, via.planes[p].data, "plane {p}");
    }
    let back = conv(&y10, PixelFormat::Yuv420P10Le, PixelFormat::RgbF32Le, w, h);
    let via = conv(
        &conv(
            &y10,
            PixelFormat::Yuv420P10Le,
            PixelFormat::Yuv420P16Le,
            w,
            h,
        ),
        PixelFormat::Yuv420P16Le,
        PixelFormat::RgbF32Le,
        w,
        h,
    );
    assert_eq!(back.planes[0].data, via.planes[0].data);
    // A 10-bit code that is not an 8-bit multiple must be visible in the
    // float output: 10-bit luma ramp → GrayF32 keeps 1024 levels.
    let mut data = vec![0u8; 1024 * 2];
    for c in 0..1024u16 {
        data[c as usize * 2..c as usize * 2 + 2].copy_from_slice(&c.to_le_bytes());
    }
    let ramp = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane { stride: 2048, data },
            VideoPlane {
                stride: 2048,
                data: vec![0; 2048],
            },
            VideoPlane {
                stride: 2048,
                data: vec![0; 2048],
            },
        ],
    };
    // Neutral chroma at 10 bits is 512.
    let mut ramp = ramp;
    for p in 1..3 {
        for c in ramp.planes[p].data.chunks_mut(2) {
            c.copy_from_slice(&512u16.to_le_bytes());
        }
    }
    let g = conv(
        &ramp,
        PixelFormat::Yuv444P10Le,
        PixelFormat::GrayF32Le,
        1024,
        1,
    );
    let v = f32s(&g.planes[0].data);
    let distinct = v.windows(2).filter(|p| p[0] != p[1]).count();
    assert!(
        distinct > 700,
        "only {distinct} distinct steps — an 8-bit hop crept in"
    );
}

/// Stride padding on every plane is transparent, and odd dimensions
/// are fine (no chroma grid anywhere in the family).
#[test]
fn padding_and_odd_dimensions() {
    let (w, h) = (5, 3);
    let tight = rgba_f32(w, h, 0, smooth);
    let padded = rgba_f32(w, h, 7, smooth);
    for dst in [
        PixelFormat::GbrapF32Le,
        PixelFormat::GrayF32Le,
        PixelFormat::Rgba64Le,
        PixelFormat::Gbrp10Le,
        PixelFormat::Yuv444P16Le,
    ] {
        let a = conv(&tight, PixelFormat::RgbaF32Le, dst, w, h);
        let b = conv(&padded, PixelFormat::RgbaF32Le, dst, w, h);
        for p in 0..a.planes.len() {
            assert_eq!(a.planes[p].data, b.planes[p].data, "{dst:?} plane {p}");
        }
    }
    // Planar float with padded strides.
    let planar = conv(
        &padded,
        PixelFormat::RgbaF32Le,
        PixelFormat::GbrapF32Le,
        w,
        h,
    );
    let mut padded_planar = planar.clone();
    for p in padded_planar.planes.iter_mut() {
        let mut data = Vec::new();
        for y in 0..h {
            data.extend_from_slice(&p.data[y * w * 4..(y + 1) * w * 4]);
            data.extend_from_slice(&[0xAA; 6]);
        }
        p.stride = w * 4 + 6;
        p.data = data;
    }
    let a = conv(&planar, PixelFormat::GbrapF32Le, PixelFormat::Rgba, w, h);
    let b = conv(
        &padded_planar,
        PixelFormat::GbrapF32Le,
        PixelFormat::Rgba,
        w,
        h,
    );
    assert_eq!(a.planes[0].data, b.planes[0].data);
    // Missing planes are an error, not a panic.
    let short = VideoFrame {
        pts: None,
        planes: planar.planes[..2].to_vec(),
    };
    assert!(convert(
        &short,
        FrameInfo::new(PixelFormat::GbrapF32Le, w as u32, h as u32),
        PixelFormat::Rgba,
        &opts()
    )
    .is_err());
}

/// The low-level plane helpers agree with the frame path.
#[test]
fn low_level_plane_helpers() {
    let codes: Vec<u8> = (0..=255).collect();
    let mut f = vec![0u8; 256 * 4];
    float::plane_u8_to_f32le(&codes, &mut f, 256);
    let mut back = vec![0u8; 256];
    float::plane_f32le_to_u8(&f, &mut back, 256);
    assert_eq!(back, codes);
    let words: Vec<u8> = (0..4096u16).flat_map(|c| c.to_le_bytes()).collect();
    let mut f = vec![0u8; 4096 * 4];
    float::plane_le16_to_f32le(&words, &mut f, 4096, 12);
    assert_eq!(float::read_f32le(&f, 4095), 1.0);
    let mut back = vec![0u8; 4096 * 2];
    float::plane_f32le_to_le16(&f, &mut back, 4096, 12);
    assert_eq!(back, words);
}
