#![no_main]

//! Fuzz harness for the ICC transform builder and its buffer paths
//! (`oxideav_pixfmt::icc`, feature `icc`).
//!
//! The first bytes choose the options (rendering intent, CLUT
//! interpolation scheme, thread budget, precomputed-table grid size,
//! which built-in profile sits on the other side); the rest is parsed
//! as a profile, with a valid header spliced in front when the input
//! does not carry one so the tag decoders are reached. Whatever parses
//! is used as source, destination, both ends of a round trip, the
//! middle of a three-profile chain and as a DeviceLink; every transform
//! that builds runs the per-pixel, 8-bit, 16-bit and `f32` paths (with
//! out-of-range and non-finite float samples), and a small precomputed
//! table is built from it and run too. Invariant: no panic, no
//! overflow, no out-of-bounds read, no runaway allocation — `Err` is
//! always an acceptable answer.

use libfuzzer_sys::fuzz_target;
use oxideav_pixfmt::icc::{
    ClutInterpolation, IccProfile, IccTransform, RenderingIntent, TransformOptions,
};

fn exercise(t: &IccTransform, seed: u8) {
    let (ic, oc) = (t.input_channels(), t.output_channels());
    let px = 5;
    let s16: Vec<u16> = (0..ic * px)
        .map(|i| (i as u16).wrapping_mul(9973).wrapping_add(seed as u16 * 257))
        .collect();
    let s8: Vec<u8> = s16.iter().map(|&v| (v >> 8) as u8).collect();
    let mut sf: Vec<f32> = s16.iter().map(|&v| v as f32 / 65535.0).collect();
    if let Some(v) = sf.get_mut(0) {
        *v = f32::NAN;
    }
    if let Some(v) = sf.get_mut(1) {
        *v = -3.5;
    }
    if let Some(v) = sf.get_mut(2) {
        *v = f32::INFINITY;
    }
    let mut d16 = vec![0u16; oc * px];
    let mut d8 = vec![0u8; oc * px];
    let mut df = vec![0f32; oc * px];
    let _ = t.transform_u16(&s16, &mut d16);
    let _ = t.transform_u8(&s8, &mut d8);
    let _ = t.transform_f32(&sf, &mut df);
    // Short destination and ragged source must be errors, not panics.
    let _ = t.transform_u8(&s8, &mut d8[..oc * px - 1]);
    let _ = t.transform_u16(&s16[..s16.len() - 1], &mut d16);
}

fuzz_target!(|data: &[u8]| {
    let [a, b, rest @ ..] = data else { return };
    let intent = match a & 3 {
        0 => RenderingIntent::Perceptual,
        1 => RenderingIntent::RelativeColorimetric,
        2 => RenderingIntent::Saturation,
        _ => RenderingIntent::AbsoluteColorimetric,
    };
    let interpolation = if a & 4 != 0 {
        ClutInterpolation::Tetrahedral
    } else {
        ClutInterpolation::Multilinear
    };
    let opts = TransformOptions::new()
        .with_intent(intent)
        .with_interpolation(interpolation)
        .with_threads(1 + ((a >> 3) & 3) as usize);
    let grid = 2 + ((a >> 5) & 3) as usize;
    let other = match b & 3 {
        0 => IccProfile::srgb(),
        1 => IccProfile::lab_d50(),
        2 => IccProfile::display_p3(),
        _ => IccProfile::rec2020(),
    };

    // Reach the tag decoders: when the bytes are not a profile on
    // their own, give them a header that says they are one.
    let parsed = IccProfile::parse(rest).ok().or_else(|| {
        let mut bytes = IccProfile::srgb().as_bytes()[..128].to_vec();
        bytes.extend_from_slice(rest);
        let n = (bytes.len() as u32).to_be_bytes();
        bytes[..4].copy_from_slice(&n);
        // A few header fields the tag decoders branch on.
        if let [c, k, p, ..] = rest {
            let classes = [b"scnr", b"mntr", b"prtr", b"link", b"spac", b"abst"];
            bytes[12..16].copy_from_slice(classes[*c as usize % classes.len()]);
            let spaces = [b"RGB ", b"CMYK", b"GRAY", b"Lab ", b"XYZ ", b"3CLR", b"FCLR"];
            bytes[16..20].copy_from_slice(spaces[*k as usize % spaces.len()]);
            bytes[20..24].copy_from_slice(if p & 1 == 0 { b"XYZ " } else { b"Lab " });
        }
        IccProfile::parse(&bytes).ok()
    });
    let Some(p) = parsed else { return };
    let _ = p.description();
    let _ = p.copyright();

    let mut built = Vec::new();
    built.extend(IccTransform::with_options(&p, &other, opts).ok());
    built.extend(IccTransform::with_options(&other, &p, opts).ok());
    built.extend(IccTransform::with_options(&p, &p, opts).ok());
    built.extend(IccTransform::chain(&[&other, &p, &other], opts).ok());
    built.extend(IccTransform::device_link_with(&p, opts).ok());
    for t in &built {
        exercise(t, *b);
        if let Ok(baked) = t.precompute(grid) {
            exercise(&baked, *b);
        }
    }
});
