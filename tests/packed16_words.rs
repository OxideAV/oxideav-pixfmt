//! The packed 16-bit RGB words of oxideav-core 0.1.41 (`Rgb555Le` /
//! `Rgb555Be` / `Bgr555Le` / `Bgr555Be` / `Rgb565Le` / `Rgb565Be` /
//! `Bgr565Le` / `Bgr565Be`): bit-layout pins against the core variant
//! table, the replication widen / truncating quantiser round-trips,
//! relayout parity with staging through 8 bits, the four-byte orders,
//! `Gray8` / `Pal8` hops, the pinned ordered and error-diffusion
//! quantisers, odd widths on odd strides, and thread parity.

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::packed16::{self, Packed16Layout};
use oxideav_pixfmt::{
    ConvertContext, ConvertOptions, Dither, FormatInfo, FrameInfo, Palette, convert, convert_with,
    staged_pivot, supports_direct,
};

const WORDS: [PixelFormat; 8] = [
    PixelFormat::Rgb555Le,
    PixelFormat::Rgb555Be,
    PixelFormat::Bgr555Le,
    PixelFormat::Bgr555Be,
    PixelFormat::Rgb565Le,
    PixelFormat::Rgb565Be,
    PixelFormat::Bgr565Le,
    PixelFormat::Bgr565Be,
];
const PACKED8: [PixelFormat; 6] = [
    PixelFormat::Rgb24,
    PixelFormat::Bgr24,
    PixelFormat::Rgba,
    PixelFormat::Bgra,
    PixelFormat::Argb,
    PixelFormat::Abgr,
];
const PADDED: [PixelFormat; 4] = [
    PixelFormat::Rgbx,
    PixelFormat::Bgrx,
    PixelFormat::Xrgb,
    PixelFormat::Xbgr,
];

/// The 8×8 ordered-dither matrix the crate documents (values 0..=63),
/// restated here so the pins do not read the crate's own table.
const BAYER: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

fn tight(data: Vec<u8>, stride: usize) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

fn conv(src: &VideoFrame, sf: PixelFormat, df: PixelFormat, w: u32, h: u32) -> VideoFrame {
    convert(
        src,
        FrameInfo::new(sf, w, h),
        df,
        &ConvertOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{sf:?} → {df:?}: {e:?}"))
}

fn conv_opts(
    src: &VideoFrame,
    sf: PixelFormat,
    df: PixelFormat,
    w: u32,
    h: u32,
    opts: &ConvertOptions,
) -> VideoFrame {
    convert(src, FrameInfo::new(sf, w, h), df, opts)
        .unwrap_or_else(|e| panic!("{sf:?} → {df:?}: {e:?}"))
}

fn rng(seed: &mut u32) -> u8 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 24) as u8
}

fn random_bytes(n: usize, seed: &mut u32) -> Vec<u8> {
    (0..n).map(|_| rng(seed)).collect()
}

/// Every word 0..=65535 as a 256 × 256 picture (or a 16 × 16 subset
/// under miri), in the layout's byte order.
fn all_words(l: Packed16Layout) -> (Vec<u8>, u32, u32, Vec<u16>) {
    let (w, h) = if cfg!(miri) { (16, 16) } else { (256, 256) };
    let words: Vec<u16> = (0..w * h)
        .map(|i| {
            if cfg!(miri) {
                (i * 257) as u16
            } else {
                i as u16
            }
        })
        .collect();
    let mut data = vec![0u8; words.len() * 2];
    for (px, &word) in data.as_chunks_mut::<2>().0.iter_mut().zip(&words) {
        l.write_word(px, word);
    }
    (data, w as u32, h as u32, words)
}

fn layout(f: PixelFormat) -> Packed16Layout {
    Packed16Layout::of(f).unwrap_or_else(|| panic!("{f:?} is not a 16-bit word layout"))
}

/// The 5-5-5 top bit is padding: a word read back after any round
/// trip has it cleared.
fn canonical(l: Packed16Layout, word: u16) -> u16 {
    if l.g_bits == 5 { word & 0x7FFF } else { word }
}

#[test]
fn the_brief_pairs_are_direct() {
    for &wf in &WORDS {
        for &other in PACKED8
            .iter()
            .chain(&PADDED)
            .chain(&WORDS)
            .chain(&[PixelFormat::Gray8, PixelFormat::Pal8])
        {
            if other == wf {
                continue;
            }
            assert!(supports_direct(wf, other), "{wf:?} → {other:?} not direct");
            assert!(supports_direct(other, wf), "{other:?} → {wf:?} not direct");
        }
        // YUV carriage stages through an 8-bit RGB pivot, no deeper.
        let p = staged_pivot(wf, PixelFormat::Yuv420P).expect("staged");
        assert!(
            matches!(p, PixelFormat::Rgba | PixelFormat::Rgb24),
            "{wf:?} → Yuv420P pivots through {p:?}"
        );
        let p = staged_pivot(PixelFormat::Yuv420P, wf).expect("staged");
        assert!(matches!(p, PixelFormat::Rgba | PixelFormat::Rgb24));
    }
}

#[test]
fn format_info_and_core_geometry_agree() {
    for &f in &WORDS {
        let info = FormatInfo::of(f);
        assert_eq!(info.planes, 1, "{f:?}");
        assert!(
            !info.has_alpha && !info.is_planar && !info.is_palette,
            "{f:?}"
        );
        let expect_bits = if layout(f).g_bits == 6 { 6 } else { 5 };
        assert_eq!(info.bit_depth, expect_bits, "{f:?}");
        assert_eq!(f.plane_row_bytes(0, 7), Some(14), "{f:?}");
        assert!(f.is_packed_rgb() && f.is_rgb() && !f.has_alpha(), "{f:?}");
    }
}

#[test]
fn word_to_rgb24_pins_against_the_core_bit_table() {
    // (format, word bytes in memory, expected R, G, B)
    let cases: &[(PixelFormat, [u8; 2], [u8; 3])] = &[
        // 0x1234 = 0001 0010 0011 0100.
        // Rgb565: R = 2 → 16, G = 17 → 69, B = 20 → 165.
        (PixelFormat::Rgb565Le, [0x34, 0x12], [16, 69, 165]),
        (PixelFormat::Rgb565Be, [0x12, 0x34], [16, 69, 165]),
        // Bgr565: B = 2, G = 17, R = 20.
        (PixelFormat::Bgr565Le, [0x34, 0x12], [165, 69, 16]),
        (PixelFormat::Bgr565Be, [0x12, 0x34], [165, 69, 16]),
        // Rgb555: R = 4 → 33, G = 17 → 140, B = 20 → 165.
        (PixelFormat::Rgb555Le, [0x34, 0x12], [33, 140, 165]),
        (PixelFormat::Rgb555Be, [0x12, 0x34], [33, 140, 165]),
        // The 5-5-5 top bit is ignored: 0x9234 reads like 0x1234.
        (PixelFormat::Rgb555Le, [0x34, 0x92], [33, 140, 165]),
        (PixelFormat::Rgb555Be, [0x92, 0x34], [33, 140, 165]),
        (PixelFormat::Bgr555Le, [0x34, 0x12], [165, 140, 33]),
        (PixelFormat::Bgr555Be, [0x12, 0x34], [165, 140, 33]),
        // Pure components and full scale.
        (PixelFormat::Rgb565Le, [0x00, 0xF8], [255, 0, 0]),
        (PixelFormat::Rgb565Le, [0xE0, 0x07], [0, 255, 0]),
        (PixelFormat::Rgb565Le, [0x1F, 0x00], [0, 0, 255]),
        (PixelFormat::Rgb555Le, [0x00, 0x7C], [255, 0, 0]),
        (PixelFormat::Rgb555Le, [0xE0, 0x03], [0, 255, 0]),
        (PixelFormat::Rgb555Le, [0x1F, 0x00], [0, 0, 255]),
        (PixelFormat::Rgb555Le, [0xFF, 0xFF], [255, 255, 255]),
        (PixelFormat::Rgb565Be, [0xFF, 0xFF], [255, 255, 255]),
        (PixelFormat::Bgr555Be, [0x00, 0x00], [0, 0, 0]),
    ];
    for &(f, bytes, rgb) in cases {
        let src = tight(bytes.to_vec(), 2);
        let out = conv(&src, f, PixelFormat::Rgb24, 1, 1);
        assert_eq!(out.planes[0].data, rgb, "{f:?} {bytes:02X?}");
        let bgr = conv(&src, f, PixelFormat::Bgr24, 1, 1);
        assert_eq!(
            bgr.planes[0].data,
            [rgb[2], rgb[1], rgb[0]],
            "{f:?} → Bgr24"
        );
    }
}

#[test]
fn rgb24_to_word_pins() {
    let cases: &[(PixelFormat, [u8; 3], [u8; 2])] = &[
        // [200, 100, 50] → r 25, g6 25 / g5 12, b 6.
        (PixelFormat::Rgb565Le, [200, 100, 50], [0x26, 0xCB]),
        (PixelFormat::Rgb565Be, [200, 100, 50], [0xCB, 0x26]),
        (PixelFormat::Rgb555Le, [200, 100, 50], [0x86, 0x65]),
        (PixelFormat::Rgb555Be, [200, 100, 50], [0x65, 0x86]),
        // Bgr565: (6 << 11) | (25 << 5) | 25 = 0x3339.
        (PixelFormat::Bgr565Le, [200, 100, 50], [0x39, 0x33]),
        (PixelFormat::Bgr565Be, [200, 100, 50], [0x33, 0x39]),
        // Bgr555: (6 << 10) | (12 << 5) | 25 = 0x1999.
        (PixelFormat::Bgr555Le, [200, 100, 50], [0x99, 0x19]),
        (PixelFormat::Bgr555Be, [200, 100, 50], [0x19, 0x99]),
        // Full scale: 5-5-5 leaves bit 15 clear, 5-6-5 fills the word.
        (PixelFormat::Rgb555Le, [255, 255, 255], [0xFF, 0x7F]),
        (PixelFormat::Rgb555Be, [255, 255, 255], [0x7F, 0xFF]),
        (PixelFormat::Bgr555Le, [255, 255, 255], [0xFF, 0x7F]),
        (PixelFormat::Rgb565Le, [255, 255, 255], [0xFF, 0xFF]),
        (PixelFormat::Rgb565Le, [16, 69, 165], [0x34, 0x12]),
        (PixelFormat::Rgb555Le, [0, 0, 0], [0, 0]),
        // Truncation: 7 → 0, 8 → 1 on 5 bits; 3 → 0, 4 → 1 on 6 bits.
        (PixelFormat::Rgb565Le, [7, 3, 7], [0x00, 0x00]),
        (PixelFormat::Rgb565Le, [8, 4, 8], [0x21, 0x08]),
    ];
    for &(f, rgb, bytes) in cases {
        let out = conv(&tight(rgb.to_vec(), 3), PixelFormat::Rgb24, f, 1, 1);
        assert_eq!(out.planes[0].data, bytes, "{f:?} {rgb:?}");
        let bgr = tight(vec![rgb[2], rgb[1], rgb[0]], 3);
        let out = conv(&bgr, PixelFormat::Bgr24, f, 1, 1);
        assert_eq!(out.planes[0].data, bytes, "Bgr24 → {f:?} {rgb:?}");
    }
}

#[test]
fn every_word_round_trips_through_rgb24() {
    for &f in &WORDS {
        let l = layout(f);
        let (data, w, h, words) = all_words(l);
        let src = tight(data, w as usize * 2);
        let rgb = conv(&src, f, PixelFormat::Rgb24, w, h);
        let back = conv(&rgb, PixelFormat::Rgb24, f, w, h);
        for (i, (px, &word)) in back.planes[0]
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&words)
            .enumerate()
        {
            assert_eq!(l.read_word(px), canonical(l, word), "{f:?} word {i}");
        }
        // The expansion is the replication rule, component by component.
        for (px, &word) in rgb.planes[0].data.as_chunks::<3>().0.iter().zip(&words) {
            let (r, g, b) = l.split(word);
            let g8 = if l.g_bits == 6 {
                packed16::expand6(g)
            } else {
                packed16::expand5(g)
            };
            assert_eq!(
                *px,
                [packed16::expand5(r), g8, packed16::expand5(b)],
                "{f:?}"
            );
        }
    }
}

#[test]
fn quantise_then_widen_is_a_fixed_point() {
    let (w, h) = if cfg!(miri) { (5u32, 3u32) } else { (37, 23) };
    let mut seed = 0x5a5a_1234u32;
    let rgb = tight(
        random_bytes(w as usize * h as usize * 3, &mut seed),
        w as usize * 3,
    );
    for &f in &WORDS {
        let word1 = conv(&rgb, PixelFormat::Rgb24, f, w, h);
        let rgb2 = conv(&word1, f, PixelFormat::Rgb24, w, h);
        let word2 = conv(&rgb2, PixelFormat::Rgb24, f, w, h);
        assert_eq!(word1.planes[0].data, word2.planes[0].data, "{f:?}");
        let rgb3 = conv(&word2, f, PixelFormat::Rgb24, w, h);
        assert_eq!(rgb2.planes[0].data, rgb3.planes[0].data, "{f:?}");
        // Every quantised component is the truncation of the source.
        let l = layout(f);
        for (px, s) in word1.planes[0]
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(rgb.planes[0].data.as_chunks::<3>().0)
        {
            assert_eq!(l.read_word(px), l.pack(s[0], s[1], s[2]), "{f:?}");
        }
    }
}

#[test]
fn relayout_direct_equals_staging_through_rgb24() {
    for &sf in &WORDS {
        let sl = layout(sf);
        let (data, w, h, _) = all_words(sl);
        let src = tight(data, w as usize * 2);
        let rgb = conv(&src, sf, PixelFormat::Rgb24, w, h);
        for &df in &WORDS {
            if df == sf {
                continue;
            }
            let direct = conv(&src, sf, df, w, h);
            let staged = conv(&rgb, PixelFormat::Rgb24, df, w, h);
            assert_eq!(
                direct.planes[0].data, staged.planes[0].data,
                "{sf:?} → {df:?}"
            );
            assert_eq!(direct.planes[0].stride, w as usize * 2);
        }
        // Same bit layout, other byte order: a pure byte swap, and the
        // 5-5-5 top bit comes out clear.
        let swapped = conv(&src, sf, swap_order(sf), w, h);
        for (d, s) in swapped.planes[0]
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(src.planes[0].data.as_chunks::<2>().0)
        {
            let expect = canonical(sl, sl.read_word(s));
            assert_eq!(layout(swap_order(sf)).read_word(d), expect, "{sf:?}");
        }
    }
}

fn swap_order(f: PixelFormat) -> PixelFormat {
    match f {
        PixelFormat::Rgb555Le => PixelFormat::Rgb555Be,
        PixelFormat::Rgb555Be => PixelFormat::Rgb555Le,
        PixelFormat::Bgr555Le => PixelFormat::Bgr555Be,
        PixelFormat::Bgr555Be => PixelFormat::Bgr555Le,
        PixelFormat::Rgb565Le => PixelFormat::Rgb565Be,
        PixelFormat::Rgb565Be => PixelFormat::Rgb565Le,
        PixelFormat::Bgr565Le => PixelFormat::Bgr565Be,
        PixelFormat::Bgr565Be => PixelFormat::Bgr565Le,
        other => other,
    }
}

/// Byte positions of R, G, B and the fourth lane for a 4-byte order.
fn pos4(f: PixelFormat) -> [usize; 4] {
    match f {
        PixelFormat::Rgba | PixelFormat::Rgbx => [0, 1, 2, 3],
        PixelFormat::Bgra | PixelFormat::Bgrx => [2, 1, 0, 3],
        PixelFormat::Argb | PixelFormat::Xrgb => [1, 2, 3, 0],
        PixelFormat::Abgr | PixelFormat::Xbgr => [3, 2, 1, 0],
        other => panic!("{other:?} is not a 4-byte order"),
    }
}

#[test]
fn four_byte_orders_carry_colour_and_a_constant_fourth_lane() {
    let (w, h) = if cfg!(miri) { (5u32, 2u32) } else { (33, 9) };
    let mut seed = 0x0004_1234u32;
    for &wf in &WORDS {
        let src = tight(
            random_bytes(w as usize * h as usize * 2, &mut seed),
            w as usize * 2,
        );
        let rgb = conv(&src, wf, PixelFormat::Rgb24, w, h);
        for &f4 in PACKED8[2..].iter().chain(&PADDED) {
            let out = conv(&src, wf, f4, w, h);
            let [r, g, b, a] = pos4(f4);
            let fill = if f4.has_alpha() { 255 } else { 0 };
            for (px, s) in out.planes[0]
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .zip(rgb.planes[0].data.as_chunks::<3>().0)
            {
                assert_eq!(
                    [px[r], px[g], px[b], px[a]],
                    [s[0], s[1], s[2], fill],
                    "{wf:?} → {f4:?}"
                );
            }
            // The way back ignores the fourth lane entirely.
            let mut junk = out.clone();
            for px in junk.planes[0].data.as_chunks_mut::<4>().0 {
                px[a] = rng(&mut seed);
            }
            let back = conv(&junk, f4, wf, w, h);
            let via_rgb = conv(&rgb, PixelFormat::Rgb24, wf, w, h);
            assert_eq!(
                back.planes[0].data, via_rgb.planes[0].data,
                "{f4:?} → {wf:?}"
            );
        }
    }
}

#[test]
fn gray8_and_pal8_hops() {
    let (w, h) = if cfg!(miri) { (4u32, 2u32) } else { (16, 5) };
    let mut seed = 0x6a6a_0001u32;
    let gray = tight(random_bytes(w as usize * h as usize, &mut seed), w as usize);
    let colors: Vec<[u8; 4]> = (0..=255u16)
        .map(|_| {
            [
                rng(&mut seed),
                rng(&mut seed),
                rng(&mut seed),
                rng(&mut seed),
            ]
        })
        .collect();
    let opts = ConvertOptions {
        palette: Some(Palette {
            colors: colors.clone(),
        }),
        ..Default::default()
    };
    let pal = tight(random_bytes(w as usize * h as usize, &mut seed), w as usize);
    for &wf in &WORDS {
        let l = layout(wf);
        // Gray8 → word: pack(g, g, g).
        let out = conv(&gray, PixelFormat::Gray8, wf, w, h);
        for (px, &g) in out.planes[0]
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&gray.planes[0].data)
        {
            assert_eq!(l.read_word(px), l.pack(g, g, g), "Gray8 → {wf:?}");
        }
        // Word → Gray8: the Rgb24 route's luminance projection.
        let via = conv(
            &conv(&out, wf, PixelFormat::Rgb24, w, h),
            PixelFormat::Rgb24,
            PixelFormat::Gray8,
            w,
            h,
        );
        let direct = conv(&out, wf, PixelFormat::Gray8, w, h);
        assert_eq!(direct.planes[0].data, via.planes[0].data, "{wf:?} → Gray8");
        // Pal8 → word: each index's palette colour packed (alpha dropped).
        let out = conv_opts(&pal, PixelFormat::Pal8, wf, w, h, &opts);
        for (px, &i) in out.planes[0]
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&pal.planes[0].data)
        {
            let c = colors[i as usize];
            assert_eq!(l.read_word(px), l.pack(c[0], c[1], c[2]), "Pal8 → {wf:?}");
        }
        // Word → Pal8: the Rgb24 route's nearest-colour search, and the
        // output frame carries the table.
        let direct = conv_opts(&out, wf, PixelFormat::Pal8, w, h, &opts);
        let rgb = conv(&out, wf, PixelFormat::Rgb24, w, h);
        let via = conv_opts(&rgb, PixelFormat::Rgb24, PixelFormat::Pal8, w, h, &opts);
        assert_eq!(direct.planes[0].data, via.planes[0].data, "{wf:?} → Pal8");
        assert!(
            direct.palette().is_some(),
            "{wf:?} → Pal8 carries its table"
        );
    }
}

#[test]
fn ordered_dither_to_565_is_pinned() {
    // A flat [100, 100, 100] 8 × 8 picture. Red / blue (5 bits): 100 =
    // 12 · 8 + 4, so the threshold t = m >> 3 (0..=7) lifts the code to
    // 13 exactly when t ≥ 4, i.e. m ≥ 32. Green (6 bits): 100 = 25 · 4
    // + 0, and t = m >> 4 ≤ 3 never reaches the next code.
    let rgb = tight(vec![100u8; 8 * 8 * 3], 24);
    let opts = ConvertOptions {
        dither: Dither::Bayer8x8,
        ..Default::default()
    };
    let out = conv_opts(&rgb, PixelFormat::Rgb24, PixelFormat::Rgb565Le, 8, 8, &opts);
    for y in 0..8 {
        for x in 0..8 {
            let q = if BAYER[y & 7][x & 7] >= 32 { 13u16 } else { 12 };
            let expect = (q << 11) | (25 << 5) | q;
            let px = &out.planes[0].data[(y * 8 + x) * 2..][..2];
            assert_eq!(u16::from_le_bytes([px[0], px[1]]), expect, "({x}, {y})");
        }
    }
    // The dither is a pure function of (value, x, y): a wider picture
    // repeats the 8 × 8 tile, and the same rule holds on every layout
    // (the 5-5-5 green behaves like red / blue).
    let rgb = tight(vec![100u8; 19 * 11 * 3], 19 * 3);
    let out = conv_opts(
        &rgb,
        PixelFormat::Rgb24,
        PixelFormat::Bgr555Be,
        19,
        11,
        &opts,
    );
    for y in 0..11 {
        for x in 0..19 {
            let q = if BAYER[y & 7][x & 7] >= 32 { 13u16 } else { 12 };
            let expect = (q << 10) | (q << 5) | q;
            let px = &out.planes[0].data[(y * 19 + x) * 2..][..2];
            assert_eq!(u16::from_be_bytes([px[0], px[1]]), expect, "({x}, {y})");
        }
    }
    // Saturation: 255 + t clamps to 255, never wraps.
    let white = tight(vec![255u8; 8 * 8 * 3], 24);
    let out = conv_opts(
        &white,
        PixelFormat::Rgb24,
        PixelFormat::Rgb565Le,
        8,
        8,
        &opts,
    );
    assert!(out.planes[0].data.iter().all(|&b| b == 0xFF));
    // The way out is a widen: the dither setting changes nothing.
    let plain = conv(&out, PixelFormat::Rgb565Le, PixelFormat::Rgb24, 8, 8);
    let dithered = conv_opts(&out, PixelFormat::Rgb565Le, PixelFormat::Rgb24, 8, 8, &opts);
    assert_eq!(plain.planes[0].data, dithered.planes[0].data);
}

#[test]
fn ordered_dither_on_a_565_to_555_relayout_touches_green_only() {
    // 565 words with R = 12, G = 25, B = 13: green widens to 101 and
    // the 5-bit threshold t = m >> 3 lifts it to 13 when 101 + t ≥ 104,
    // i.e. m ≥ 24; red and blue are copied as they are.
    let word = (12u16 << 11) | (25 << 5) | 13;
    let mut data = vec![0u8; 8 * 8 * 2];
    for px in data.as_chunks_mut::<2>().0 {
        px.copy_from_slice(&word.to_le_bytes());
    }
    let src = tight(data, 16);
    let opts = ConvertOptions {
        dither: Dither::Bayer8x8,
        ..Default::default()
    };
    let out = conv_opts(
        &src,
        PixelFormat::Rgb565Le,
        PixelFormat::Rgb555Le,
        8,
        8,
        &opts,
    );
    for y in 0..8 {
        for x in 0..8 {
            let g = if BAYER[y & 7][x & 7] >= 24 { 13u16 } else { 12 };
            let expect = (12 << 10) | (g << 5) | 13;
            let px = &out.planes[0].data[(y * 8 + x) * 2..][..2];
            assert_eq!(u16::from_le_bytes([px[0], px[1]]), expect, "({x}, {y})");
        }
    }
    // 555 → 565 only widens, so the dither setting changes nothing.
    let plain = conv(&out, PixelFormat::Rgb555Le, PixelFormat::Bgr565Be, 8, 8);
    let dithered = conv_opts(
        &out,
        PixelFormat::Rgb555Le,
        PixelFormat::Bgr565Be,
        8,
        8,
        &opts,
    );
    assert_eq!(plain.planes[0].data, dithered.planes[0].data);
}

/// The documented integer error-diffusion rule, restated independently.
fn reference_floyd_steinberg(
    src: &[u8],
    bpp: usize,
    pos: [usize; 3],
    w: usize,
    h: usize,
    bits: [u8; 3],
) -> Vec<[u8; 3]> {
    // `bits == 8` names a component that is copied, not quantised.
    let expand = |q: u8, bits: u8| -> i32 {
        match bits {
            8 => q as i32,
            6 => ((q << 2) | (q >> 4)) as i32,
            _ => ((q << 3) | (q >> 2)) as i32,
        }
    };
    let mut cur = vec![[0i32; 3]; w];
    let mut next = vec![[0i32; 3]; w];
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut q = [0u8; 3];
            for c in 0..3 {
                let v = (src[(y * w + x) * bpp + pos[c]] as i32 + cur[x][c]).clamp(0, 255);
                q[c] = (v as u8) >> (8 - bits[c]);
                let e = v - expand(q[c], bits[c]);
                let e7 = e * 7 / 16;
                let e3 = e * 3 / 16;
                let e5 = e * 5 / 16;
                let e1 = e - e7 - e3 - e5;
                if x + 1 < w {
                    cur[x + 1][c] += e7;
                    next[x + 1][c] += e1;
                }
                if x > 0 {
                    next[x - 1][c] += e3;
                }
                next[x][c] += e5;
            }
            out.push(q);
        }
        std::mem::swap(&mut cur, &mut next);
        next.iter_mut().for_each(|c| *c = [0; 3]);
    }
    out
}

#[test]
fn floyd_steinberg_to_565_is_pinned() {
    let opts = ConvertOptions {
        dither: Dither::FloydSteinberg,
        ..Default::default()
    };
    // Hand-walked row of four [128, 128, 128] pixels (see the module
    // doc of `packed16` for the rule): red / blue alternate 16, 15,
    // 16, 15 as the −4 / +4 residuals ripple right; green stays 32.
    let row = tight(vec![128u8; 4 * 3], 12);
    let out = conv_opts(&row, PixelFormat::Rgb24, PixelFormat::Rgb565Le, 4, 1, &opts);
    let words: Vec<u16> = out.planes[0]
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|px| u16::from_le_bytes([px[0], px[1]]))
        .collect();
    assert_eq!(words, [0x8410, 0x7C0F, 0x8410, 0x7C0F]);
    // A random picture against the independent reference, on every
    // layout and from both the 3- and 4-byte orders.
    let (w, h) = if cfg!(miri) {
        (6usize, 3usize)
    } else {
        (23, 11)
    };
    let mut seed = 0xF5F5_0001u32;
    let rgb = random_bytes(w * h * 3, &mut seed);
    let bgra = random_bytes(w * h * 4, &mut seed);
    for &wf in &WORDS {
        let l = layout(wf);
        let bits = [5, l.g_bits, 5];
        let out = conv_opts(
            &tight(rgb.clone(), w * 3),
            PixelFormat::Rgb24,
            wf,
            w as u32,
            h as u32,
            &opts,
        );
        let expect = reference_floyd_steinberg(&rgb, 3, [0, 1, 2], w, h, bits);
        for (i, (px, q)) in out.planes[0]
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&expect)
            .enumerate()
        {
            assert_eq!(
                l.split(l.read_word(px)),
                (q[0], q[1], q[2]),
                "{wf:?} px {i}"
            );
        }
        let out = conv_opts(
            &tight(bgra.clone(), w * 4),
            PixelFormat::Bgra,
            wf,
            w as u32,
            h as u32,
            &opts,
        );
        let expect = reference_floyd_steinberg(&bgra, 4, [2, 1, 0], w, h, bits);
        for (i, (px, q)) in out.planes[0]
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&expect)
            .enumerate()
        {
            assert_eq!(
                l.split(l.read_word(px)),
                (q[0], q[1], q[2]),
                "Bgra → {wf:?} px {i}"
            );
        }
    }
    // Error diffusion on a 565 → 555 relayout: green only, through the
    // widened 6-bit value; red and blue copied.
    let src565: Vec<u8> = conv(
        &tight(rgb.clone(), w * 3),
        PixelFormat::Rgb24,
        PixelFormat::Rgb565Le,
        w as u32,
        h as u32,
    )
    .planes[0]
        .data
        .clone();
    let out = conv_opts(
        &tight(src565.clone(), w * 2),
        PixelFormat::Rgb565Le,
        PixelFormat::Rgb555Le,
        w as u32,
        h as u32,
        &opts,
    );
    let widened: Vec<u8> = src565
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|px| {
            let (r, g, b) = packed16::RGB565_LE.split(u16::from_le_bytes([px[0], px[1]]));
            [r, packed16::expand6(g), b]
        })
        .collect();
    let expect = reference_floyd_steinberg(&widened, 3, [0, 1, 2], w, h, [8, 5, 8]);
    for (i, (px, q)) in out.planes[0]
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&expect)
        .enumerate()
    {
        assert_eq!(
            packed16::RGB555_LE.split(u16::from_le_bytes([px[0], px[1]])),
            (q[0], q[1], q[2]),
            "px {i}"
        );
    }
}

#[test]
fn odd_widths_on_odd_strides() {
    let sizes: &[(u32, u32)] = if cfg!(miri) {
        &[(1, 1), (3, 2)]
    } else {
        &[(1, 1), (1, 3), (3, 1), (3, 2), (5, 3), (7, 5), (9, 1)]
    };
    let pads: &[usize] = if cfg!(miri) { &[1] } else { &[0, 1, 3, 5] };
    let mut seed = 0x0dd5_7a7eu32;
    let targets: Vec<PixelFormat> = PACKED8
        .iter()
        .chain(&PADDED)
        .chain(&WORDS)
        .chain(&[PixelFormat::Gray8])
        .copied()
        .collect();
    for &(w, h) in sizes {
        for &pad in pads {
            for &wf in &WORDS {
                // Padded-stride source: the words sit at odd offsets.
                let stride = w as usize * 2 + pad;
                let data = random_bytes(stride * h as usize, &mut seed);
                let padded = tight(data.clone(), stride);
                let mut tight_data = Vec::new();
                for row in 0..h as usize {
                    tight_data
                        .extend_from_slice(&data[row * stride..row * stride + w as usize * 2]);
                }
                let tight_src = tight(tight_data, w as usize * 2);
                for &df in &targets {
                    if df == wf {
                        continue;
                    }
                    let a = conv(&padded, wf, df, w, h);
                    let b = conv(&tight_src, wf, df, w, h);
                    assert_eq!(
                        a.planes[0].data, b.planes[0].data,
                        "{wf:?} → {df:?} {w}×{h} pad {pad}"
                    );
                    assert_eq!(
                        a.planes[0].data.len(),
                        df.plane_row_bytes(0, w).unwrap() * h as usize
                    );
                }
                // And into the words from a padded-stride 8-bit source.
                let stride = w as usize * 3 + pad;
                let data = random_bytes(stride * h as usize, &mut seed);
                let padded = tight(data.clone(), stride);
                let mut tight_data = Vec::new();
                for row in 0..h as usize {
                    tight_data
                        .extend_from_slice(&data[row * stride..row * stride + w as usize * 3]);
                }
                let tight_src = tight(tight_data, w as usize * 3);
                for dither in [Dither::None, Dither::Bayer8x8, Dither::FloydSteinberg] {
                    let opts = ConvertOptions {
                        dither,
                        ..Default::default()
                    };
                    let a = conv_opts(&padded, PixelFormat::Bgr24, wf, w, h, &opts);
                    let b = conv_opts(&tight_src, PixelFormat::Bgr24, wf, w, h, &opts);
                    assert_eq!(
                        a.planes[0].data, b.planes[0].data,
                        "Bgr24 → {wf:?} {w}×{h} pad {pad} {dither:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn short_planes_are_errors_not_panics() {
    let short = tight(vec![0u8; 5], 8); // 4 × 2 words need 16 bytes
    for &df in PACKED8
        .iter()
        .chain(&WORDS)
        .chain(&[PixelFormat::Gray8, PixelFormat::Yuv420P])
    {
        if df == PixelFormat::Rgb565Le {
            continue;
        }
        let r = convert(
            &short,
            FrameInfo::new(PixelFormat::Rgb565Le, 4, 2),
            df,
            &ConvertOptions::default(),
        );
        assert!(r.is_err(), "Rgb565Le → {df:?} on a short plane");
    }
    let short = tight(vec![0u8; 5], 12);
    for &wf in &WORDS {
        for dither in [Dither::None, Dither::FloydSteinberg] {
            let opts = ConvertOptions {
                dither,
                ..Default::default()
            };
            let r = convert(&short, FrameInfo::new(PixelFormat::Rgb24, 4, 2), wf, &opts);
            assert!(r.is_err(), "Rgb24 → {wf:?} on a short plane");
        }
    }
    // Zero-sized pictures are fine.
    let empty = tight(Vec::new(), 0);
    let out = conv(&empty, PixelFormat::Rgb565Le, PixelFormat::Rgba, 0, 0);
    assert!(out.planes[0].data.is_empty());
}

#[test]
fn thread_budget_does_not_change_the_bytes() {
    let (w, h) = if cfg!(miri) { (7u32, 5u32) } else { (64, 37) };
    let mut seed = 0x7777_0001u32;
    let rgb = tight(
        random_bytes(w as usize * h as usize * 3, &mut seed),
        w as usize * 3,
    );
    let words = tight(
        random_bytes(w as usize * h as usize * 2, &mut seed),
        w as usize * 2,
    );
    let opts = ConvertOptions::default();
    let serial = ConvertContext::new();
    let banded = ConvertContext::new().with_execution(ExecutionContext::with_threads(4));
    let pairs: &[(&VideoFrame, PixelFormat, PixelFormat)] = &[
        (&rgb, PixelFormat::Rgb24, PixelFormat::Rgb565Le),
        (&rgb, PixelFormat::Rgb24, PixelFormat::Bgr555Be),
        (&words, PixelFormat::Rgb565Be, PixelFormat::Rgb24),
        (&words, PixelFormat::Bgr555Le, PixelFormat::Rgba),
        (&words, PixelFormat::Rgb565Le, PixelFormat::Xbgr),
        (&words, PixelFormat::Rgb565Le, PixelFormat::Rgb555Be),
        (&words, PixelFormat::Rgb555Le, PixelFormat::Yuv420P),
        (&words, PixelFormat::Bgr565Be, PixelFormat::Gray8),
    ];
    for &(src, sf, df) in pairs {
        let info = FrameInfo::new(sf, w, h);
        let a = convert_with(src, info, df, &opts, &serial).expect("serial");
        let b = convert_with(src, info, df, &opts, &banded).expect("banded");
        assert_eq!(a.planes.len(), b.planes.len(), "{sf:?} → {df:?}");
        for (pa, pb) in a.planes.iter().zip(&b.planes) {
            assert_eq!(pa.data, pb.data, "{sf:?} → {df:?}");
        }
    }
}

#[test]
fn staged_yuv_route_equals_the_manual_two_step() {
    let (w, h) = if cfg!(miri) { (4u32, 2u32) } else { (16, 10) };
    let mut seed = 0x9999_0001u32;
    let words = tight(
        random_bytes(w as usize * h as usize * 2, &mut seed),
        w as usize * 2,
    );
    for &wf in &WORDS {
        let pivot = staged_pivot(wf, PixelFormat::Yuv420P).expect("staged");
        let staged = conv(&words, wf, PixelFormat::Yuv420P, w, h);
        let mid = conv(&words, wf, pivot, w, h);
        let manual = conv(&mid, pivot, PixelFormat::Yuv420P, w, h);
        for (a, b) in staged.planes.iter().zip(&manual.planes) {
            assert_eq!(a.data, b.data, "{wf:?} → {pivot:?} → Yuv420P");
        }
        // And back: Yuv420P → word through the same kind of pivot.
        let pivot = staged_pivot(PixelFormat::Yuv420P, wf).expect("staged");
        let back = conv(&staged, PixelFormat::Yuv420P, wf, w, h);
        let mid = conv(&staged, PixelFormat::Yuv420P, pivot, w, h);
        let manual = conv(&mid, pivot, wf, w, h);
        assert_eq!(
            back.planes[0].data, manual.planes[0].data,
            "Yuv420P → {pivot:?} → {wf:?}"
        );
    }
}
