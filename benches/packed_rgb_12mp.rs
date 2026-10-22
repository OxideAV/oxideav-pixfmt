//! Criterion benchmarks for the core 0.1.41 packed RGB tier at a
//! 12-megapixel still (4032×3024): the 16-bit `Rgb565Le` / `Rgb555Be`
//! words and the padded `Bgrx` layout against the 8-bit packed family
//! and `Yuv420P`, through the high-level [`convert`] call (plane
//! access, kernel, output allocation).
//!
//! Every case runs serial (`convert`) and with the host's thread budget
//! (`convert_with` + `ExecutionContext::auto()`, suffix `_mt`); the
//! dithered quantisers (`Bayer8x8`, `FloydSteinberg`) are single-pass
//! by construction and run serial only.
//!
//! Run with `cargo bench --features bench --bench packed_rgb_12mp`.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_core::{ExecutionContext, PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, convert_with, ConvertContext, ConvertOptions, Dither, FrameInfo};

const W: u32 = 4032;
const H: u32 = 3024;

fn synth_rgb24(w: u32, h: u32) -> VideoFrame {
    let (w, h) = (w as usize, h as usize);
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            data.push(((x * 255) / w) as u8);
            data.push(((y * 255) / h) as u8);
            data.push((((x + y) * 255) / (w + h)) as u8);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 3,
            data,
        }],
    }
}

fn via(src: &VideoFrame, sf: PixelFormat, df: PixelFormat) -> VideoFrame {
    convert(
        src,
        FrameInfo::new(sf, W, H),
        df,
        &ConvertOptions::default(),
    )
    .expect("convert")
}

fn bench_packed_rgb_12mp(c: &mut Criterion) {
    let rgb24 = synth_rgb24(W, H);
    let rgb565 = via(&rgb24, PixelFormat::Rgb24, PixelFormat::Rgb565Le);
    let bgrx = via(&rgb24, PixelFormat::Rgb24, PixelFormat::Bgrx);
    let rgba = via(&rgb24, PixelFormat::Rgb24, PixelFormat::Rgba);
    let yuv420 = via(&rgb24, PixelFormat::Rgb24, PixelFormat::Yuv420P);
    let opts = ConvertOptions::default();
    let mt = ConvertContext::new().with_execution(ExecutionContext::auto());

    let cases: Vec<(&VideoFrame, PixelFormat, PixelFormat)> = vec![
        (&rgb565, PixelFormat::Rgb565Le, PixelFormat::Rgb24),
        (&rgb565, PixelFormat::Rgb565Le, PixelFormat::Rgba),
        (&rgb565, PixelFormat::Rgb565Le, PixelFormat::Bgra),
        (&rgb565, PixelFormat::Rgb565Le, PixelFormat::Rgb555Be),
        (&rgb24, PixelFormat::Rgb24, PixelFormat::Rgb565Le),
        (&rgba, PixelFormat::Rgba, PixelFormat::Rgb565Le),
        (&rgb565, PixelFormat::Rgb565Le, PixelFormat::Yuv420P),
        (&yuv420, PixelFormat::Yuv420P, PixelFormat::Rgb565Le),
        (&bgrx, PixelFormat::Bgrx, PixelFormat::Rgba),
        (&bgrx, PixelFormat::Bgrx, PixelFormat::Rgb24),
        (&rgba, PixelFormat::Rgba, PixelFormat::Bgrx),
        (&rgb24, PixelFormat::Rgb24, PixelFormat::Bgrx),
        (&bgrx, PixelFormat::Bgrx, PixelFormat::Yuv420P),
        (&yuv420, PixelFormat::Yuv420P, PixelFormat::Bgrx),
    ];

    let mut group = c.benchmark_group("packed_rgb_12mp");
    group.sample_size(10);
    for (src, sf, df) in cases {
        let info = FrameInfo::new(sf, W, H);
        let bytes =
            (sf.frame_size_bytes(W, H).unwrap() + df.frame_size_bytes(W, H).unwrap()) as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(format!("{sf:?}_to_{df:?}"), |b| {
            b.iter(|| convert(src, info, df, &opts).expect("convert"))
        });
        group.bench_function(format!("{sf:?}_to_{df:?}_mt"), |b| {
            b.iter(|| convert_with(src, info, df, &opts, &mt).expect("convert"))
        });
    }
    for dither in [Dither::Bayer8x8, Dither::FloydSteinberg] {
        let opts = ConvertOptions {
            dither,
            ..Default::default()
        };
        let info = FrameInfo::new(PixelFormat::Rgb24, W, H);
        group.throughput(Throughput::Bytes((W as u64) * (H as u64) * 5));
        group.bench_function(format!("Rgb24_to_Rgb565Le_{dither:?}"), |b| {
            b.iter(|| convert(&rgb24, info, PixelFormat::Rgb565Le, &opts).expect("convert"))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_packed_rgb_12mp);
criterion_main!(benches);
