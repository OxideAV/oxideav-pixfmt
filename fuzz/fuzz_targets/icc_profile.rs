#![no_main]

//! Byte-stream fuzz harness for the ICC profile parser
//! (`oxideav_pixfmt::icc`, feature `icc`).
//!
//! The input is split in two at a fuzzer-chosen point; each half is
//! parsed as a profile (most inputs bounce off the header checks, so a
//! valid header is spliced in front of short inputs to reach the tag
//! decoders). When both parse, a transform is built for every intent
//! and a handful of pixels are pushed through it. Invariant: no panic,
//! no overflow, no out-of-bounds read, no runaway allocation — `Err` is
//! always an acceptable answer.

use libfuzzer_sys::fuzz_target;
use oxideav_pixfmt::icc::{IccProfile, IccTransform, RenderingIntent};

fn parse(bytes: &[u8]) -> Option<IccProfile> {
    IccProfile::parse(bytes).ok()
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let cut = (data[0] as usize * data.len()) / 256;
    let (a, b) = data[1..].split_at(cut.min(data.len() - 1));
    let pa = parse(a);
    let pb = parse(b);
    let srgb = IccProfile::srgb();
    let pairs = [
        (pa.as_ref(), pb.as_ref()),
        (pa.as_ref(), Some(&srgb)),
        (Some(&srgb), pa.as_ref()),
    ];
    for (s, d) in pairs {
        let (Some(s), Some(d)) = (s, d) else { continue };
        for intent in [
            RenderingIntent::Perceptual,
            RenderingIntent::RelativeColorimetric,
            RenderingIntent::Saturation,
            RenderingIntent::AbsoluteColorimetric,
        ] {
            let Ok(t) = IccTransform::new(s, d, intent) else { continue };
            let n = t.input_channels();
            let src: Vec<u16> = (0..n * 4).map(|i| (i as u16).wrapping_mul(9973)).collect();
            let mut dst = vec![0u16; 4 * t.output_channels()];
            let _ = t.transform_u16(&src, &mut dst);
        }
    }
});
