//! Over-range samples on the round-476 deep routes: a 10 / 12 / 14-bit
//! word whose high (padding) bits are set is legal input — hostile
//! frames, containers that leave the padding bits dirty — and every
//! route masks it to the declared depth, never indexes with it. Each
//! case converts an all-`0xFFFF` picture and checks it equals the same
//! picture masked to the nominal depth, both directions, serial and
//! threaded, at a size below and above the composed-table threshold.
//! (The fuzz run of 2026-10-16 caught `Yuv*P10Le → Gray16Le` indexing
//! a 1024-entry table with 1284.)

use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, convert_with, ConvertContext, ConvertOptions, FormatInfo, FrameInfo,
};

fn frame_filled(fmt: PixelFormat, w: u32, h: u32, word: u16) -> VideoFrame {
    let planes = (0..fmt.plane_count())
        .map(|p| {
            let (pw, ph) = fmt.plane_dimensions(p, w, h).unwrap();
            let row = fmt.plane_row_bytes(p, w).unwrap();
            let _ = pw;
            let data: Vec<u8> = (0..row * ph as usize)
                .map(|i| {
                    if i % 2 == 0 {
                        (word & 0xFF) as u8
                    } else {
                        (word >> 8) as u8
                    }
                })
                .collect();
            VideoPlane { stride: row, data }
        })
        .collect();
    VideoFrame { pts: None, planes }
}

/// Every (deep source, destination) pair the round added a route for,
/// where the source can carry over-range words.
fn cases() -> Vec<(PixelFormat, PixelFormat)> {
    use PixelFormat as P;
    let deep_src = [
        P::Gray10Le,
        P::Gray12Le,
        P::Yuv420P10Le,
        P::Yuv444P12Le,
        P::Yuva422P10Le,
        P::Gbrp10Le,
        P::Gbrap12Le,
        P::Gbrp14Le,
    ];
    let dsts = [
        P::Gray10Le,
        P::Gray12Le,
        P::Gray16Le,
        P::Rgb48Le,
        P::Rgba64Le,
        P::Yuv420P10Le,
        P::Yuv444P16Le,
        P::Yuva444P12Le,
        P::Yuv420P,
        P::Gbrp10Le,
        P::Gbrp16Le,
        P::Gbrap8,
        P::Ya16Le,
        P::GrayF32Le,
    ];
    let mut v = Vec::new();
    for s in deep_src {
        for d in dsts {
            if s != d {
                v.push((s, d));
            }
        }
    }
    v
}

#[test]
fn over_range_words_are_masked_never_indexed() {
    let opts = ConvertOptions::default();
    let mt = ConvertContext::new().with_execution(ExecutionContext::with_threads(4));
    let sizes: &[(u32, u32)] = if cfg!(miri) {
        &[(2, 2)]
    } else {
        &[(2, 2), (70, 34), (256, 17)]
    };
    for (sf, df) in cases() {
        let bits = FormatInfo::of(sf).bit_depth as u32;
        let mask = ((1u32 << bits) - 1) as u16;
        for &(w, h) in sizes {
            let dirty = frame_filled(sf, w, h, 0xFFFF);
            let clean = frame_filled(sf, w, h, mask);
            let info = FrameInfo::new(sf, w, h);
            let a = convert(&dirty, info, df, &opts)
                .unwrap_or_else(|e| panic!("{sf:?} → {df:?} {w}×{h}: {e}"));
            let b = convert(&clean, info, df, &opts).unwrap();
            assert_eq!(a.planes.len(), b.planes.len(), "{sf:?} → {df:?}");
            for (p, (x, y)) in a.planes.iter().zip(b.planes.iter()).enumerate() {
                assert_eq!(
                    x.data, y.data,
                    "{sf:?} → {df:?} {w}×{h} plane {p}: dirty != masked"
                );
            }
            let c = convert_with(&dirty, info, df, &opts, &mt).unwrap();
            for (p, (x, y)) in a.planes.iter().zip(c.planes.iter()).enumerate() {
                assert_eq!(
                    x.data, y.data,
                    "{sf:?} → {df:?} {w}×{h} plane {p}: threaded"
                );
            }
        }
    }
}
