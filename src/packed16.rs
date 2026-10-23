//! Packed 16-bit RGB words — the `Rgb555Le` / `Rgb555Be` / `Bgr555Le` /
//! `Bgr555Be` / `Rgb565Le` / `Rgb565Be` / `Bgr565Le` / `Bgr565Be`
//! layouts of oxideav-core 0.1.41 — and the exact rules this crate uses
//! to move between their 5 / 6-bit components and 8-bit samples.
//!
//! # Layout
//!
//! One 16-bit word per pixel. The format name lists the components from
//! the word's most significant bits downward and the suffix is the byte
//! order of the word in memory (`Le`: low byte first, `Be`: high byte
//! first). A 5-5-5 word leaves bit 15 unused: it is **not** alpha —
//! readers ignore it and every writer in this crate emits 0 (the core
//! variant docs' ruling). 5-6-5 uses all sixteen bits.
//!
//! | layout   | bit 15 | 14..10 | 15..11 | 10..5 | 9..5 | 4..0 |
//! |----------|--------|--------|--------|-------|------|------|
//! | `Rgb555` | unused | R      |        |       | G    | B    |
//! | `Bgr555` | unused | B      |        |       | G    | R    |
//! | `Rgb565` |        |        | R      | G     |      | B    |
//! | `Bgr565` |        |        | B      | G     |      | R    |
//!
//! # Widening and quantisation
//!
//! The core variant docs leave the 5 / 6 → 8-bit widening to the
//! consumer ("bit replication or scaling"). This crate uses the same
//! rule as every other depth move it performs (see the crate-level
//! *Bit-depth precision policy*): the value is placed in the top bits
//! and its own most significant bits are replicated into the freed low
//! bits —
//!
//! - 5 → 8: `(v << 3) | (v >> 2)` — 0 → 0, 31 → 255, strictly monotonic,
//!   within one code of the ideal `v · 255 / 31`;
//! - 6 → 8: `(v << 2) | (v >> 4)` — 0 → 0, 63 → 255, likewise.
//!
//! The way *in* (8 → 5 / 6) truncates the low bits (`v >> 3`, `v >> 2`)
//! — the exact inverse of the replication fill, so a packed word
//! widened to 8 bits and packed again is bit-identical, and the 5 ↔ 6
//! bit green move of a 5-5-5 ↔ 5-6-5 relayout (`(g << 1) | (g >> 4)`
//! up, `g >> 1` down) is byte-identical to staging through 8 bits. This
//! is a crate-local rule: the core docs state the layout only, and the
//! truncating default is the one every other narrow in this crate uses.
//!
//! # Dither
//!
//! Quantising 8-bit colour to the 16-bit words honours
//! [`ConvertOptions::dither`](crate::ConvertOptions) (the default
//! `Dither::None` is the plain truncation above):
//!
//! - **`Bayer8x8`** — ordered dither: with `step = 256 >> bits` (8 for a
//!   5-bit component, 4 for 6-bit) and `m = BAYER8[y & 7][x & 7]` (0..=63),
//!   the threshold `t = (m · step) >> 6` (0..step) is added before the
//!   truncation: `q = min(v + t, 255) >> (8 − bits)`. Every value is an
//!   integer; the output is a pure function of `(v, x, y)`.
//! - **`FloydSteinberg`** — error diffusion in integer arithmetic: the
//!   residual `e = v' − widen(q)` of the clamped, error-corrected sample
//!   `v'` is split `e·7/16` right, `e·3/16` below-left, `e·5/16` below
//!   and the remainder (so the error is conserved exactly) below-right,
//!   with integer division truncating toward zero and shares that would
//!   leave the picture dropped; rows are walked left to right, top to
//!   bottom. The sweep is sequential and deterministic.
//!
//! `Dither` never touches the way *out* (16 → 8 is a widen, nothing to
//! dither) or a relayout that keeps every component's width.

use crate::convert::Dither;
use crate::dither::BAYER8;
use crate::rgb::{Rgb3, Rgba4};
use oxideav_core::PixelFormat;

/// Bit layout of one packed 16-bit RGB word.
///
/// Instances come from the eight constants below or
/// [`Packed16Layout::of`]; the struct is `#[non_exhaustive]` so new
/// descriptors can be added without a semver break.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Packed16Layout {
    /// Bit position of the 5-bit red component's lowest bit.
    pub r_shift: u8,
    /// Bit position of the green component's lowest bit.
    pub g_shift: u8,
    /// Bit position of the 5-bit blue component's lowest bit.
    pub b_shift: u8,
    /// Width of the green component: 5 (5-5-5) or 6 (5-6-5). Red and
    /// blue are always 5 bits.
    pub g_bits: u8,
    /// `true` when the word is stored most significant byte first.
    pub big_endian: bool,
}

const fn layout(r: u8, g: u8, b: u8, g_bits: u8, big_endian: bool) -> Packed16Layout {
    Packed16Layout {
        r_shift: r,
        g_shift: g,
        b_shift: b,
        g_bits,
        big_endian,
    }
}

/// `Rgb555Le`: R in bits 14..10, G in 9..5, B in 4..0; low byte first.
pub const RGB555_LE: Packed16Layout = layout(10, 5, 0, 5, false);
/// `Rgb555Be`: R in bits 14..10, G in 9..5, B in 4..0; high byte first.
pub const RGB555_BE: Packed16Layout = layout(10, 5, 0, 5, true);
/// `Bgr555Le`: B in bits 14..10, G in 9..5, R in 4..0; low byte first.
pub const BGR555_LE: Packed16Layout = layout(0, 5, 10, 5, false);
/// `Bgr555Be`: B in bits 14..10, G in 9..5, R in 4..0; high byte first.
pub const BGR555_BE: Packed16Layout = layout(0, 5, 10, 5, true);
/// `Rgb565Le`: R in bits 15..11, G in 10..5, B in 4..0; low byte first.
pub const RGB565_LE: Packed16Layout = layout(11, 5, 0, 6, false);
/// `Rgb565Be`: R in bits 15..11, G in 10..5, B in 4..0; high byte first.
pub const RGB565_BE: Packed16Layout = layout(11, 5, 0, 6, true);
/// `Bgr565Le`: B in bits 15..11, G in 10..5, R in 4..0; low byte first.
pub const BGR565_LE: Packed16Layout = layout(0, 5, 11, 6, false);
/// `Bgr565Be`: B in bits 15..11, G in 10..5, R in 4..0; high byte first.
pub const BGR565_BE: Packed16Layout = layout(0, 5, 11, 6, true);

impl Packed16Layout {
    /// The layout of a packed 16-bit RGB `PixelFormat`, or `None` for
    /// any other format.
    pub const fn of(fmt: PixelFormat) -> Option<Self> {
        use PixelFormat as P;
        Some(match fmt {
            P::Rgb555Le => RGB555_LE,
            P::Rgb555Be => RGB555_BE,
            P::Bgr555Le => BGR555_LE,
            P::Bgr555Be => BGR555_BE,
            P::Rgb565Le => RGB565_LE,
            P::Rgb565Be => RGB565_BE,
            P::Bgr565Le => BGR565_LE,
            P::Bgr565Be => BGR565_BE,
            _ => return None,
        })
    }

    /// Mask of the green component at bit 0 (`0x1F` or `0x3F`).
    #[inline]
    const fn g_mask(&self) -> u16 {
        if self.g_bits == 6 { 0x3F } else { 0x1F }
    }

    /// Read the word at the start of `px` (two bytes) in this layout's
    /// byte order.
    #[inline]
    pub fn read_word(&self, px: &[u8]) -> u16 {
        if self.big_endian {
            u16::from_be_bytes([px[0], px[1]])
        } else {
            u16::from_le_bytes([px[0], px[1]])
        }
    }

    /// Write `word` to the start of `px` (two bytes) in this layout's
    /// byte order.
    #[inline]
    pub fn write_word(&self, px: &mut [u8], word: u16) {
        let b = if self.big_endian {
            word.to_be_bytes()
        } else {
            word.to_le_bytes()
        };
        px[0] = b[0];
        px[1] = b[1];
    }

    /// Split a word into its raw `(r, g, b)` components (5, 5-or-6, 5
    /// bits; the unused top bit of a 5-5-5 word is ignored).
    #[inline]
    pub const fn split(&self, word: u16) -> (u8, u8, u8) {
        (
            ((word >> self.r_shift) & 0x1F) as u8,
            ((word >> self.g_shift) & self.g_mask()) as u8,
            ((word >> self.b_shift) & 0x1F) as u8,
        )
    }

    /// Assemble raw `(r, g, b)` components (already 5 / 6-bit) into a
    /// word. Out-of-range inputs are masked; the unused top bit of a
    /// 5-5-5 word is 0.
    #[inline]
    pub const fn join(&self, r: u8, g: u8, b: u8) -> u16 {
        (((r & 0x1F) as u16) << self.r_shift)
            | (((g as u16) & self.g_mask()) << self.g_shift)
            | (((b & 0x1F) as u16) << self.b_shift)
    }

    /// Decode a word to 8-bit `(r, g, b)` by the replication rule.
    #[inline]
    pub const fn unpack(&self, word: u16) -> (u8, u8, u8) {
        let (r, g, b) = self.split(word);
        let g8 = if self.g_bits == 6 {
            expand6(g)
        } else {
            expand5(g)
        };
        (expand5(r), g8, expand5(b))
    }

    /// Encode 8-bit `(r, g, b)` to a word by truncation.
    #[inline]
    pub const fn pack(&self, r: u8, g: u8, b: u8) -> u16 {
        let gq = if self.g_bits == 6 {
            quantise6(g)
        } else {
            quantise5(g)
        };
        self.join(quantise5(r), gq, quantise5(b))
    }

    /// Bit width of each component in dither-quantiser order `[r, g, b]`.
    #[inline]
    const fn widths(&self) -> [u8; 3] {
        [5, self.g_bits, 5]
    }
}

/// 5 → 8 bits by replication: `(v << 3) | (v >> 2)`.
#[inline]
pub const fn expand5(v: u8) -> u8 {
    let v = v & 0x1F;
    (v << 3) | (v >> 2)
}

/// 6 → 8 bits by replication: `(v << 2) | (v >> 4)`.
#[inline]
pub const fn expand6(v: u8) -> u8 {
    let v = v & 0x3F;
    (v << 2) | (v >> 4)
}

/// 8 → 5 bits by truncation: `v >> 3`.
#[inline]
pub const fn quantise5(v: u8) -> u8 {
    v >> 3
}

/// 8 → 6 bits by truncation: `v >> 2`.
#[inline]
pub const fn quantise6(v: u8) -> u8 {
    v >> 2
}

/// Widen a `bits`-wide (5 or 6) component to 8 bits by replication.
#[inline]
const fn expand(v: u8, bits: u8) -> u8 {
    if bits == 6 { expand6(v) } else { expand5(v) }
}

// -------------------------------------------------------------------------
// Row kernels (tight buffers, `pixels` pixels).
//
// Every kernel is monomorphised on what would otherwise be a per-pixel
// branch or a scattered store: the green width (`G6`), the byte order
// of the word (`BE`) and the byte order of the 8-bit side (`ORDER`:
// `Rgb24` / `Bgr24`, or `Rgba` / `Bgra` / `Argb` / `Abgr` positions —
// the padded layouts share the alpha twins' positions). With the byte
// indices constant the compiler emits the interleaved loads / stores
// as vector groups; an `Rgb3` / `Rgba4` outside the six canonical
// orders falls back to a generic scatter.

macro_rules! specialise {
    ($l:expr_2021, $f:ident, $o:literal, ($($arg:expr_2021),*)) => {
        match ($l.g_bits == 6, $l.big_endian) {
            (false, false) => $f::<false, false, $o>($($arg),*),
            (false, true) => $f::<false, true, $o>($($arg),*),
            (true, false) => $f::<true, false, $o>($($arg),*),
            (true, true) => $f::<true, true, $o>($($arg),*),
        }
    };
}

/// Decode one word to 8-bit `(r, g, b)`.
#[inline(always)]
fn decode_px<const G6: bool, const BE: bool>(px: &[u8], rs: u32, gs: u32, bs: u32) -> (u8, u8, u8) {
    let w = if BE {
        u16::from_be_bytes([px[0], px[1]])
    } else {
        u16::from_le_bytes([px[0], px[1]])
    };
    let r = expand5(((w >> rs) & 0x1F) as u8);
    let b = expand5(((w >> bs) & 0x1F) as u8);
    let g = if G6 {
        expand6(((w >> gs) & 0x3F) as u8)
    } else {
        expand5(((w >> gs) & 0x1F) as u8)
    };
    (r, g, b)
}

/// Encode 8-bit `(r, g, b)` to one word (truncating quantiser).
#[inline(always)]
fn encode_px<const G6: bool, const BE: bool>(
    r: u8,
    g: u8,
    b: u8,
    rs: u32,
    gs: u32,
    bs: u32,
) -> [u8; 2] {
    let gq = if G6 { quantise6(g) } else { quantise5(g) } as u16;
    let w = ((quantise5(r) as u16) << rs) | (gq << gs) | ((quantise5(b) as u16) << bs);
    if BE { w.to_be_bytes() } else { w.to_le_bytes() }
}

#[inline(always)]
fn shifts(l: Packed16Layout) -> (u32, u32, u32) {
    (l.r_shift as u32, l.g_shift as u32, l.b_shift as u32)
}

/// Canonical 3-byte orders: 0 = R, G, B; 1 = B, G, R; `None` otherwise.
#[inline(always)]
fn order3(pos: Rgb3) -> Option<usize> {
    if pos == crate::rgb::RGB_POS {
        Some(0)
    } else if pos == crate::rgb::BGR_POS {
        Some(1)
    } else {
        None
    }
}

/// Canonical 4-byte orders: 0 = R, G, B, X; 1 = B, G, R, X; 2 = X, R,
/// G, B; 3 = X, B, G, R; `None` otherwise.
#[inline(always)]
fn order4(pos: Rgba4) -> Option<usize> {
    use crate::rgb::{ABGR_POS, ARGB_POS, BGRA_POS, RGBA_POS};
    if pos == RGBA_POS {
        Some(0)
    } else if pos == BGRA_POS {
        Some(1)
    } else if pos == ARGB_POS {
        Some(2)
    } else if pos == ABGR_POS {
        Some(3)
    } else {
        None
    }
}

/// Write `(r, g, b)` (and `x`) into a 3- or 4-byte group in `ORDER`
/// (see [`order3`] / [`order4`]); constant indices per monomorph.
#[inline(always)]
fn put3<const ORDER: usize>(d: &mut [u8], r: u8, g: u8, b: u8) {
    if ORDER == 1 {
        d[0] = b;
        d[1] = g;
        d[2] = r;
    } else {
        d[0] = r;
        d[1] = g;
        d[2] = b;
    }
}

#[inline(always)]
fn put4<const ORDER: usize>(d: &mut [u8], r: u8, g: u8, b: u8, x: u8) {
    match ORDER {
        1 => {
            d[0] = b;
            d[1] = g;
            d[2] = r;
            d[3] = x;
        }
        2 => {
            d[0] = x;
            d[1] = r;
            d[2] = g;
            d[3] = b;
        }
        3 => {
            d[0] = x;
            d[1] = b;
            d[2] = g;
            d[3] = r;
        }
        _ => {
            d[0] = r;
            d[1] = g;
            d[2] = b;
            d[3] = x;
        }
    }
}

#[inline(always)]
fn get3<const ORDER: usize>(s: &[u8]) -> (u8, u8, u8) {
    if ORDER == 1 {
        (s[2], s[1], s[0])
    } else {
        (s[0], s[1], s[2])
    }
}

#[inline(always)]
fn get4<const ORDER: usize>(s: &[u8]) -> (u8, u8, u8) {
    match ORDER {
        1 => (s[2], s[1], s[0]),
        2 => (s[1], s[2], s[3]),
        3 => (s[3], s[2], s[1]),
        _ => (s[0], s[1], s[2]),
    }
}

/// Packed 16-bit words → a 3-byte packed order (`Rgb24` / `Bgr24`
/// positions in `pos`).
pub fn to_packed3(src: &[u8], l: Packed16Layout, dst: &mut [u8], pos: Rgb3, pixels: usize) {
    match order3(pos) {
        Some(0) => specialise!(l, to_packed3_impl, 0, (src, l, dst, pixels)),
        Some(_) => specialise!(l, to_packed3_impl, 1, (src, l, dst, pixels)),
        None => {
            let (rs, gs, bs) = shifts(l);
            let (pr, pg, pb) = (pos.r.min(2), pos.g.min(2), pos.b.min(2));
            for (s, d) in src[..pixels * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..pixels * 3].as_chunks_mut::<3>().0)
            {
                let (r, g, b) = decode_generic(s, l, rs, gs, bs);
                d[pr] = r;
                d[pg] = g;
                d[pb] = b;
            }
        }
    }
}

/// Un-specialised decode for the generic fallbacks.
#[inline(always)]
fn decode_generic(px: &[u8], l: Packed16Layout, rs: u32, gs: u32, bs: u32) -> (u8, u8, u8) {
    match (l.g_bits == 6, l.big_endian) {
        (false, false) => decode_px::<false, false>(px, rs, gs, bs),
        (false, true) => decode_px::<false, true>(px, rs, gs, bs),
        (true, false) => decode_px::<true, false>(px, rs, gs, bs),
        (true, true) => decode_px::<true, true>(px, rs, gs, bs),
    }
}

#[inline(always)]
fn to_packed3_impl<const G6: bool, const BE: bool, const ORDER: usize>(
    src: &[u8],
    l: Packed16Layout,
    dst: &mut [u8],
    pixels: usize,
) {
    let (rs, gs, bs) = shifts(l);
    let src = &src[..pixels * 2];
    let dst = &mut dst[..pixels * 3];
    for (s, d) in src
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<3>().0)
    {
        let (r, g, b) = decode_px::<G6, BE>(s, rs, gs, bs);
        put3::<ORDER>(d, r, g, b);
    }
}

/// Packed 16-bit words → a 4-byte packed order: colour at `pos`, and
/// `fill` in the fourth lane (`255` for an alpha layout, `0` for a
/// padded one).
pub fn to_packed4(
    src: &[u8],
    l: Packed16Layout,
    dst: &mut [u8],
    pos: Rgba4,
    fill: u8,
    pixels: usize,
) {
    match order4(pos) {
        Some(0) => specialise!(l, to_packed4_impl, 0, (src, l, dst, fill, pixels)),
        Some(1) => specialise!(l, to_packed4_impl, 1, (src, l, dst, fill, pixels)),
        Some(2) => specialise!(l, to_packed4_impl, 2, (src, l, dst, fill, pixels)),
        Some(_) => specialise!(l, to_packed4_impl, 3, (src, l, dst, fill, pixels)),
        None => {
            let (rs, gs, bs) = shifts(l);
            let (pr, pg, pb, pa) = (pos.r & 3, pos.g & 3, pos.b & 3, pos.a & 3);
            for (s, d) in src[..pixels * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..pixels * 4].as_chunks_mut::<4>().0)
            {
                let (r, g, b) = decode_generic(s, l, rs, gs, bs);
                d[pa] = fill;
                d[pr] = r;
                d[pg] = g;
                d[pb] = b;
            }
        }
    }
}

#[inline(always)]
fn to_packed4_impl<const G6: bool, const BE: bool, const ORDER: usize>(
    src: &[u8],
    l: Packed16Layout,
    dst: &mut [u8],
    fill: u8,
    pixels: usize,
) {
    let (rs, gs, bs) = shifts(l);
    let src = &src[..pixels * 2];
    let dst = &mut dst[..pixels * 4];
    for (s, d) in src
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0)
    {
        let (r, g, b) = decode_px::<G6, BE>(s, rs, gs, bs);
        put4::<ORDER>(d, r, g, b, fill);
    }
}

/// A 3-byte packed order → packed 16-bit words (truncating quantiser).
pub fn from_packed3(src: &[u8], pos: Rgb3, dst: &mut [u8], l: Packed16Layout, pixels: usize) {
    match order3(pos) {
        Some(0) => specialise!(l, from_packed3_impl, 0, (src, dst, l, pixels)),
        Some(_) => specialise!(l, from_packed3_impl, 1, (src, dst, l, pixels)),
        None => {
            let (pr, pg, pb) = (pos.r.min(2), pos.g.min(2), pos.b.min(2));
            for (s, d) in src[..pixels * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .zip(dst[..pixels * 2].as_chunks_mut::<2>().0)
            {
                l.write_word(d, l.pack(s[pr], s[pg], s[pb]));
            }
        }
    }
}

#[inline(always)]
fn from_packed3_impl<const G6: bool, const BE: bool, const ORDER: usize>(
    src: &[u8],
    dst: &mut [u8],
    l: Packed16Layout,
    pixels: usize,
) {
    let (rs, gs, bs) = shifts(l);
    let src = &src[..pixels * 3];
    let dst = &mut dst[..pixels * 2];
    for (s, d) in src
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<2>().0)
    {
        let (r, g, b) = get3::<ORDER>(s);
        d.copy_from_slice(&encode_px::<G6, BE>(r, g, b, rs, gs, bs));
    }
}

/// A 4-byte packed order → packed 16-bit words (truncating quantiser;
/// the fourth lane — alpha or padding — is ignored).
pub fn from_packed4(src: &[u8], pos: Rgba4, dst: &mut [u8], l: Packed16Layout, pixels: usize) {
    match order4(pos) {
        Some(0) => specialise!(l, from_packed4_impl, 0, (src, dst, l, pixels)),
        Some(1) => specialise!(l, from_packed4_impl, 1, (src, dst, l, pixels)),
        Some(2) => specialise!(l, from_packed4_impl, 2, (src, dst, l, pixels)),
        Some(_) => specialise!(l, from_packed4_impl, 3, (src, dst, l, pixels)),
        None => {
            let (pr, pg, pb) = (pos.r & 3, pos.g & 3, pos.b & 3);
            for (s, d) in src[..pixels * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst[..pixels * 2].as_chunks_mut::<2>().0)
            {
                l.write_word(d, l.pack(s[pr], s[pg], s[pb]));
            }
        }
    }
}

#[inline(always)]
fn from_packed4_impl<const G6: bool, const BE: bool, const ORDER: usize>(
    src: &[u8],
    dst: &mut [u8],
    l: Packed16Layout,
    pixels: usize,
) {
    let (rs, gs, bs) = shifts(l);
    let src = &src[..pixels * 4];
    let dst = &mut dst[..pixels * 2];
    for (s, d) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<2>().0)
    {
        let (r, g, b) = get4::<ORDER>(s);
        d.copy_from_slice(&encode_px::<G6, BE>(r, g, b, rs, gs, bs));
    }
}

/// Packed 16-bit words → another packed 16-bit layout. Same-width
/// components are copied exactly; green widens 5 → 6 by replication
/// (`(g << 1) | (g >> 4)`) and narrows 6 → 5 by truncation (`g >> 1`)
/// — byte-identical to staging through 8 bits.
pub fn relayout(src: &[u8], sl: Packed16Layout, dst: &mut [u8], dl: Packed16Layout, pixels: usize) {
    let src = &src[..pixels * 2];
    let dst = &mut dst[..pixels * 2];
    for (s, d) in src
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<2>().0)
    {
        let (r, g, b) = sl.split(sl.read_word(s));
        let g = move_green(g, sl.g_bits, dl.g_bits);
        dl.write_word(d, dl.join(r, g, b));
    }
}

/// Green between the 5-bit and 6-bit widths.
#[inline]
const fn move_green(g: u8, from: u8, to: u8) -> u8 {
    match (from, to) {
        (5, 6) => (g << 1) | (g >> 4),
        (6, 5) => g >> 1,
        _ => g,
    }
}

// -------------------------------------------------------------------------
// Dithered quantisation (whole picture — the dither needs `(x, y)`).

/// Integer Floyd–Steinberg error rows: the residual assigned to each
/// column of the current row and of the row beneath, per component.
struct Diffusion {
    w: usize,
    cur: Vec<[i32; 3]>,
    next: Vec<[i32; 3]>,
}

impl Diffusion {
    fn new(w: usize) -> Self {
        Self {
            w,
            cur: vec![[0; 3]; w],
            next: vec![[0; 3]; w],
        }
    }

    fn advance_row(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.next);
        for c in self.next.iter_mut() {
            *c = [0; 3];
        }
    }

    /// Distribute component `c`'s residual `e` at column `x`: 7/16
    /// right, 3/16 below-left, 5/16 below, the exact remainder
    /// below-right.
    #[inline]
    fn diffuse(&mut self, x: usize, c: usize, e: i32) {
        let e7 = e * 7 / 16;
        let e3 = e * 3 / 16;
        let e5 = e * 5 / 16;
        let e1 = e - e7 - e3 - e5;
        if x + 1 < self.w {
            self.cur[x + 1][c] += e7;
            self.next[x + 1][c] += e1;
        }
        if x > 0 {
            self.next[x - 1][c] += e3;
        }
        self.next[x][c] += e5;
    }
}

/// The ordered-dither threshold for an 8-bit sample quantised to
/// `bits` (5 or 6) at `(x, y)`: `(BAYER8[y & 7][x & 7] · (256 >> bits)) >> 6`.
#[inline]
pub const fn bayer_threshold(x: usize, y: usize, bits: u8) -> u8 {
    let m = BAYER8[y & 7][x & 7] as u16;
    ((m * (256 >> bits)) >> 6) as u8
}

/// Quantise one 8-bit sample to `bits` under the ordered dither.
#[inline]
pub const fn quantise_ordered(v: u8, x: usize, y: usize, bits: u8) -> u8 {
    let t = bayer_threshold(x, y, bits) as u16;
    let s = v as u16 + t;
    let s = if s > 255 { 255 } else { s as u8 };
    s >> (8 - bits)
}

/// Quantise a tight packed 8-bit picture (`src_bpp` bytes per pixel,
/// colour at byte offsets `r` / `g` / `b`) to packed 16-bit words under
/// `dither`. `dst` must hold `w × h` words.
#[allow(clippy::too_many_arguments)]
pub fn quantise_rows(
    src: &[u8],
    src_bpp: usize,
    r: usize,
    g: usize,
    b: usize,
    w: usize,
    h: usize,
    l: Packed16Layout,
    dst: &mut [u8],
    dither: Dither,
) {
    quantise_rows_at(src, src_bpp, r, g, b, w, h, 0, l, dst, dither);
}

/// [`quantise_rows`] for a band of `h` rows whose first row is row
/// `y0` of the picture: the ordered dither reads the threshold at the
/// picture's `(x, y)`, so bands tile exactly like the whole picture.
/// Error diffusion carries no state across bands and must be called
/// with `y0 == 0` on the whole picture.
#[allow(clippy::too_many_arguments)]
pub fn quantise_rows_at(
    src: &[u8],
    src_bpp: usize,
    r: usize,
    g: usize,
    b: usize,
    w: usize,
    h: usize,
    y0: usize,
    l: Packed16Layout,
    dst: &mut [u8],
    dither: Dither,
) {
    let src = &src[..w * h * src_bpp];
    let dst = &mut dst[..w * h * 2];
    match dither {
        Dither::None => {
            for (s, d) in src.chunks_exact(src_bpp).zip(dst.as_chunks_mut::<2>().0) {
                l.write_word(d, l.pack(s[r], s[g], s[b]));
            }
        }
        Dither::Bayer8x8 => {
            let widths = l.widths();
            for (y, (srow, drow)) in src
                .chunks_exact(w * src_bpp)
                .zip(dst.chunks_exact_mut(w * 2))
                .enumerate()
            {
                // The row's eight thresholds per component, hoisted.
                let y = y0 + y;
                let tr: [u8; 8] = std::array::from_fn(|x| bayer_threshold(x, y, widths[0]));
                let tg: [u8; 8] = std::array::from_fn(|x| bayer_threshold(x, y, widths[1]));
                let tb: [u8; 8] = std::array::from_fn(|x| bayer_threshold(x, y, widths[2]));
                let (sr, sg, sb) = (8 - widths[0], 8 - widths[1], 8 - widths[2]);
                for (x, (s, d)) in srow
                    .chunks_exact(src_bpp)
                    .zip(drow.as_chunks_mut::<2>().0)
                    .enumerate()
                {
                    let rq = s[r].saturating_add(tr[x & 7]) >> sr;
                    let gq = s[g].saturating_add(tg[x & 7]) >> sg;
                    let bq = s[b].saturating_add(tb[x & 7]) >> sb;
                    l.write_word(d, l.join(rq, gq, bq));
                }
            }
        }
        Dither::FloydSteinberg => {
            let widths = l.widths();
            let mut err = Diffusion::new(w);
            for (srow, drow) in src
                .chunks_exact(w * src_bpp)
                .zip(dst.chunks_exact_mut(w * 2))
            {
                for (x, (s, d)) in srow
                    .chunks_exact(src_bpp)
                    .zip(drow.as_chunks_mut::<2>().0)
                    .enumerate()
                {
                    let mut q = [0u8; 3];
                    for (c, &off) in [r, g, b].iter().enumerate() {
                        let v = (s[off] as i32 + err.cur[x][c]).clamp(0, 255);
                        let bits = widths[c];
                        q[c] = (v as u8) >> (8 - bits);
                        err.diffuse(x, c, v - expand(q[c], bits) as i32);
                    }
                    l.write_word(d, l.join(q[0], q[1], q[2]));
                }
                err.advance_row();
            }
        }
    }
}

/// Relayout a tight `w × h` picture of packed 16-bit words under
/// `dither`. Only a 6 → 5 bit green narrow quantises (the 6-bit value
/// is widened to 8 bits and run through the dithered 5-bit quantiser;
/// red and blue are copied exactly); every other pair of layouts is
/// the exact [`relayout`] whatever the dither setting.
pub fn relayout_rows(
    src: &[u8],
    sl: Packed16Layout,
    dst: &mut [u8],
    dl: Packed16Layout,
    w: usize,
    h: usize,
    dither: Dither,
) {
    let n = w * h;
    if dither == Dither::None || sl.g_bits <= dl.g_bits {
        relayout(src, sl, dst, dl, n);
        return;
    }
    let src = &src[..n * 2];
    let dst = &mut dst[..n * 2];
    match dither {
        Dither::Bayer8x8 => {
            for (y, (srow, drow)) in src
                .chunks_exact(w * 2)
                .zip(dst.chunks_exact_mut(w * 2))
                .enumerate()
            {
                for (x, (s, d)) in srow
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(drow.as_chunks_mut::<2>().0)
                    .enumerate()
                {
                    let (r, g, b) = sl.split(sl.read_word(s));
                    let gq = quantise_ordered(expand6(g), x, y, 5);
                    dl.write_word(d, dl.join(r, gq, b));
                }
            }
        }
        _ => {
            let mut err = Diffusion::new(w);
            for (srow, drow) in src.chunks_exact(w * 2).zip(dst.chunks_exact_mut(w * 2)) {
                for (x, (s, d)) in srow
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(drow.as_chunks_mut::<2>().0)
                    .enumerate()
                {
                    let (r, g, b) = sl.split(sl.read_word(s));
                    let v = (expand6(g) as i32 + err.cur[x][1]).clamp(0, 255);
                    let gq = (v as u8) >> 3;
                    err.diffuse(x, 1, v - expand5(gq) as i32);
                    dl.write_word(d, dl.join(r, gq, b));
                }
                err.advance_row();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replication_endpoints_and_round_trip() {
        assert_eq!(expand5(0), 0);
        assert_eq!(expand5(31), 255);
        assert_eq!(expand6(0), 0);
        assert_eq!(expand6(63), 255);
        for v in 0..32u8 {
            assert_eq!(quantise5(expand5(v)), v);
            // Within one code of the ideal rescale.
            let ideal = (v as u32 * 255 + 15) / 31;
            assert!((expand5(v) as i32 - ideal as i32).abs() <= 1, "{v}");
        }
        for v in 0..64u8 {
            assert_eq!(quantise6(expand6(v)), v);
            let ideal = (v as u32 * 255 + 31) / 63;
            assert!((expand6(v) as i32 - ideal as i32).abs() <= 1, "{v}");
        }
        // Monotonic.
        for v in 1..32u8 {
            assert!(expand5(v) > expand5(v - 1));
        }
        for v in 1..64u8 {
            assert!(expand6(v) > expand6(v - 1));
        }
    }

    #[test]
    fn layouts_match_the_core_bit_table() {
        // Rgb555: R 14..10, G 9..5, B 4..0, bit 15 clear.
        assert_eq!(RGB555_LE.join(31, 0, 0), 0x7C00);
        assert_eq!(RGB555_LE.join(0, 31, 0), 0x03E0);
        assert_eq!(RGB555_LE.join(0, 0, 31), 0x001F);
        assert_eq!(BGR555_LE.join(31, 0, 0), 0x001F);
        assert_eq!(BGR555_LE.join(0, 0, 31), 0x7C00);
        // Rgb565: R 15..11, G 10..5, B 4..0.
        assert_eq!(RGB565_LE.join(31, 0, 0), 0xF800);
        assert_eq!(RGB565_LE.join(0, 63, 0), 0x07E0);
        assert_eq!(RGB565_LE.join(0, 0, 31), 0x001F);
        assert_eq!(BGR565_LE.join(31, 0, 0), 0x001F);
        assert_eq!(BGR565_LE.join(0, 0, 31), 0xF800);
        // The 5-5-5 top bit is ignored on read.
        assert_eq!(RGB555_LE.split(0xFFFF), (31, 31, 31));
        assert_eq!(RGB555_LE.pack(255, 255, 255), 0x7FFF);
        // Byte order.
        let mut px = [0u8; 2];
        RGB565_LE.write_word(&mut px, 0xF81F);
        assert_eq!(px, [0x1F, 0xF8]);
        RGB565_BE.write_word(&mut px, 0xF81F);
        assert_eq!(px, [0xF8, 0x1F]);
        assert_eq!(RGB565_BE.read_word(&[0xF8, 0x1F]), 0xF81F);
    }

    #[test]
    fn green_move_matches_staging_through_8_bits() {
        for g in 0..32u8 {
            assert_eq!(move_green(g, 5, 6), quantise6(expand5(g)));
        }
        for g in 0..64u8 {
            assert_eq!(move_green(g, 6, 5), quantise5(expand6(g)));
        }
    }

    #[test]
    fn bayer_threshold_range() {
        for y in 0..8 {
            for x in 0..8 {
                assert!(bayer_threshold(x, y, 5) < 8);
                assert!(bayer_threshold(x, y, 6) < 4);
            }
        }
        // m = 0 at (0, 0): no offset; m = 63 at (0, 7): the top step.
        assert_eq!(bayer_threshold(0, 0, 5), 0);
        assert_eq!(bayer_threshold(0, 7, 5), 7);
        assert_eq!(bayer_threshold(0, 7, 6), 3);
        // Saturation: 255 + 7 clamps to 255 → 31.
        assert_eq!(quantise_ordered(255, 0, 7, 5), 31);
        assert_eq!(quantise_ordered(0, 0, 7, 5), 0);
    }
}
