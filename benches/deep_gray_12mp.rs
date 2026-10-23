//! Criterion benchmarks for the round-476 depth-preserving routes at a
//! 12-megapixel still (4032×3024): the deep grey ↔ packed-16 / 16-bit
//! YUV pairs the image gateway's 16-bit views take, through the
//! high-level [`convert`] call (plane access, kernel, output
//! allocation).
//!
//! Each new direct row is measured next to the route the planner took
//! before the round — two `convert` calls staged through `Gray8` — so
//! the before / after numbers come from the same binary. Every case
//! runs serial (`convert`) and with the host's thread budget
//! (`convert_with` + `ExecutionContext::auto()`, suffix `_mt`).
//!
//! Run with `cargo bench --features bench --bench deep_gray_12mp`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{ConvertContext, ConvertOptions, FrameInfo, convert, convert_with};

const W: u32 = 4032;
const H: u32 = 3024;

fn synth_gray16(w: u32, h: u32) -> VideoFrame {
    let (w, h) = (w as usize, h as usize);
    let mut data = Vec::with_capacity(w * h * 2);
    for y in 0..h {
        for x in 0..w {
            let v = (((x * 65535) / w) ^ ((y * 65535) / h)) as u16;
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 2,
            data,
        }],
    }
}

fn synth_rgb48(w: u32, h: u32) -> VideoFrame {
    let (w, h) = (w as usize, h as usize);
    let mut data = Vec::with_capacity(w * h * 6);
    for y in 0..h {
        for x in 0..w {
            for v in [
                ((x * 65535) / w) as u16,
                ((y * 65535) / h) as u16,
                (((x + y) * 65535) / (w + h)) as u16,
            ] {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 6,
            data,
        }],
    }
}

fn staged_from(src: &VideoFrame, sf: PixelFormat, df: PixelFormat) -> VideoFrame {
    convert(
        src,
        FrameInfo::new(sf, W, H),
        df,
        &ConvertOptions::default(),
    )
    .expect("stage")
}

fn bench_deep_gray_12mp(c: &mut Criterion) {
    use PixelFormat as P;
    let opts = ConvertOptions::default();
    let mt = ConvertContext::new().with_execution(ExecutionContext::auto());
    let gray16 = synth_gray16(W, H);
    let rgb48 = synth_rgb48(W, H);
    let yuv444p16 = staged_from(&rgb48, P::Rgb48Le, P::Yuv444P16Le);
    let yuv420p16 = staged_from(&rgb48, P::Rgb48Le, P::Yuv420P16Le);
    let gbrp10 = staged_from(&rgb48, P::Rgb48Le, P::Gbrp10Le);
    let gbrp16 = staged_from(&rgb48, P::Rgb48Le, P::Gbrp16Le);

    // (source frame, source format, destination, legacy pivot)
    let cases: Vec<(&VideoFrame, P, P, Option<P>)> = vec![
        (&gray16, P::Gray16Le, P::Rgb48Le, Some(P::Gray8)),
        (&rgb48, P::Rgb48Le, P::Gray16Le, Some(P::Gray8)),
        (&yuv444p16, P::Yuv444P16Le, P::Gray16Le, Some(P::Gray8)),
        (&yuv420p16, P::Yuv420P16Le, P::Gray16Le, Some(P::Gray8)),
        (&gray16, P::Gray16Le, P::Yuv444P16Le, Some(P::Gray8)),
        (&gray16, P::Gray16Le, P::Yuv420P10Le, Some(P::Gray8)),
        (&rgb48, P::Rgb48Le, P::Yuv420P16Le, None),
        (&rgb48, P::Rgb48Le, P::Yuv420P10Le, Some(P::Yuv444P16Le)),
        (&gbrp16, P::Gbrp16Le, P::Gray16Le, Some(P::Rgb48Le)),
        (&gray16, P::Gray16Le, P::Gbrp16Le, Some(P::Rgb48Le)),
        (&gbrp10, P::Gbrp10Le, P::Yuv420P10Le, Some(P::Rgba64Le)),
    ];
    let mut group = c.benchmark_group("deep_gray_12mp");
    group.sample_size(10);
    group.throughput(Throughput::Elements(W as u64 * H as u64));
    for (src, sf, df, legacy) in cases {
        let info = FrameInfo::new(sf, W, H);
        if let Some(pivot) = legacy {
            let pinfo = FrameInfo::new(pivot, W, H);
            group.bench_function(format!("{sf:?}_to_{df:?}_via_{pivot:?}_before"), |b| {
                b.iter(|| {
                    let mid = convert(src, info, pivot, &opts).expect("leg 1");
                    convert(&mid, pinfo, df, &opts).expect("leg 2")
                });
            });
        }
        group.bench_function(format!("{sf:?}_to_{df:?}"), |b| {
            b.iter(|| convert(src, info, df, &opts).expect("convert"));
        });
        group.bench_function(format!("{sf:?}_to_{df:?}_mt"), |b| {
            b.iter(|| convert_with(src, info, df, &opts, &mt).expect("convert"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_deep_gray_12mp);
criterion_main!(benches);
