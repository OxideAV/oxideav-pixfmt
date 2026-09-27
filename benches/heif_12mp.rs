//! Criterion benchmarks for the production-HEIF conversion set at a
//! 12-megapixel still (4032×3024, the common phone-camera HEIC size).
//!
//! Every case measures the full high-level [`convert`] call — plane
//! gathering, the colour kernel and the output allocation — because that
//! is what an image pipeline pays per decoded picture. Source frames are
//! synthesised once (smooth gradients staged through `convert` itself)
//! outside the measured region.
//!
//! Run with `cargo bench --features bench --bench heif_12mp`.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

const W: u32 = 4032;
const H: u32 = 3024;

fn synth_rgba(w: u32, h: u32) -> VideoFrame {
    let (w, h) = (w as usize, h as usize);
    let mut data = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            data.push(((x * 255) / w) as u8);
            data.push(((y * 255) / h) as u8);
            data.push((((x + y) * 255) / (w + h)) as u8);
            data.push((((x ^ y) * 7) & 0xff) as u8);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data,
        }],
    }
}

fn staged(fmt: PixelFormat) -> VideoFrame {
    let rgba = synth_rgba(W, H);
    let info = FrameInfo::new(PixelFormat::Rgba, W, H);
    convert(&rgba, info, fmt, &ConvertOptions::default()).expect("stage source frame")
}

fn bench_heif_12mp(c: &mut Criterion) {
    let cases: &[(PixelFormat, PixelFormat)] = &[
        (PixelFormat::Yuv420P, PixelFormat::Rgb24),
        (PixelFormat::YuvJ420P, PixelFormat::Rgb24),
        (PixelFormat::Yuv420P, PixelFormat::Rgba),
        (PixelFormat::Yuv420P10Le, PixelFormat::Rgb24),
        (PixelFormat::Yuv420P10Le, PixelFormat::Rgb48Le),
        (PixelFormat::Yuva420P, PixelFormat::Rgba),
        (PixelFormat::Yuv444P, PixelFormat::Rgb24),
        (PixelFormat::Rgb24, PixelFormat::Yuv420P),
        (PixelFormat::Gray8, PixelFormat::Rgb24),
        (PixelFormat::Yuv420P, PixelFormat::Gray8),
        (PixelFormat::Rgb24, PixelFormat::Gray8),
    ];
    let opts = ConvertOptions::default();
    let mut group = c.benchmark_group("heif_12mp");
    group.sample_size(10);
    group.throughput(Throughput::Elements(W as u64 * H as u64));
    for &(sf, df) in cases {
        let src = staged(sf);
        let info = FrameInfo::new(sf, W, H);
        group.bench_function(format!("{sf:?}_to_{df:?}"), |b| {
            b.iter(|| convert(&src, info, df, &opts).expect("convert"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_heif_12mp);
criterion_main!(benches);
