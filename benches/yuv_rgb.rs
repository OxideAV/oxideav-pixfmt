//! Criterion benchmarks for the YUV ↔ RGB inner loops.
//!
//! Each bench synthesises a plausible-looking frame (smooth gradients) and
//! measures the tight conversion call. No allocation inside the measured
//! region — the destination buffer is reused across iterations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_pixfmt::yuv::{self, YuvMatrix};

fn synth_rgb24(w: usize, h: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = (((x * 255) / w.max(1)) & 0xff) as u8;
            let g = (((y * 255) / h.max(1)) & 0xff) as u8;
            let b = ((((x + y) * 255) / (w + h).max(1)) & 0xff) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

fn synth_yuv420(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let ch = h / 2;
    let src = synth_rgb24(w, h);
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    yuv::rgb24_to_yuv420(&src, &mut yp, &mut up, &mut vp, w, h, YuvMatrix::BT709);
    (yp, up, vp)
}

fn synth_yuv444(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let src = synth_rgb24(w, h);
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; w * h];
    let mut vp = vec![0u8; w * h];
    yuv::rgb24_to_yuv444(&src, &mut yp, &mut up, &mut vp, w, h, YuvMatrix::BT709);
    (yp, up, vp)
}

fn bench_yuv420_to_rgb24(c: &mut Criterion) {
    let mut group = c.benchmark_group("yuv420_to_rgb24");
    for (w, h, mtx, label) in [
        (
            1280usize,
            720usize,
            YuvMatrix::BT601,
            "1280x720_bt601_limited",
        ),
        (1280, 720, YuvMatrix::BT709, "1280x720_bt709_limited"),
        (1920, 1080, YuvMatrix::BT709, "1920x1080_bt709_limited"),
    ] {
        let (yp, up, vp) = synth_yuv420(w, h);
        let mut dst = vec![0u8; w * h * 3];
        group.throughput(Throughput::Bytes((w * h * 3) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &(w, h), |b, &(w, h)| {
            b.iter(|| {
                yuv::yuv420_to_rgb24(&yp, &up, &vp, &mut dst, w, h, mtx);
            });
        });
    }
    group.finish();
}

fn bench_yuv444_to_rgb24(c: &mut Criterion) {
    let mut group = c.benchmark_group("yuv444_to_rgb24");
    let (w, h) = (1280, 720);
    let (yp, up, vp) = synth_yuv444(w, h);
    let mut dst = vec![0u8; w * h * 3];
    group.throughput(Throughput::Bytes((w * h * 3) as u64));
    group.bench_function("1280x720_bt709_limited", |b| {
        b.iter(|| {
            yuv::yuv444_to_rgb24(&yp, &up, &vp, &mut dst, w, h, YuvMatrix::BT709);
        });
    });
    group.finish();
}

fn bench_rgb24_to_yuv420(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb24_to_yuv420");
    let (w, h) = (1280, 720);
    let src = synth_rgb24(w, h);
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; (w / 2) * (h / 2)];
    let mut vp = vec![0u8; (w / 2) * (h / 2)];
    group.throughput(Throughput::Bytes((w * h * 3) as u64));
    group.bench_function("1280x720_bt709_limited", |b| {
        b.iter(|| {
            yuv::rgb24_to_yuv420(&src, &mut yp, &mut up, &mut vp, w, h, YuvMatrix::BT709);
        });
    });
    group.finish();
}

fn bench_rgb24_to_yuv422(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb24_to_yuv422");
    let (w, h) = (1280, 720);
    let src = synth_rgb24(w, h);
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; (w / 2) * h];
    let mut vp = vec![0u8; (w / 2) * h];
    group.throughput(Throughput::Bytes((w * h * 3) as u64));
    group.bench_function("1280x720_bt709_limited", |b| {
        b.iter(|| {
            yuv::rgb24_to_yuv422(&src, &mut yp, &mut up, &mut vp, w, h, YuvMatrix::BT709);
        });
    });
    group.finish();
}

fn bench_rgb24_to_yuv444(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb24_to_yuv444");
    let (w, h) = (1280, 720);
    let src = synth_rgb24(w, h);
    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; w * h];
    let mut vp = vec![0u8; w * h];
    group.throughput(Throughput::Bytes((w * h * 3) as u64));
    group.bench_function("1280x720_bt709_limited", |b| {
        b.iter(|| {
            yuv::rgb24_to_yuv444(&src, &mut yp, &mut up, &mut vp, w, h, YuvMatrix::BT709);
        });
    });
    group.finish();
}

fn bench_yuv422_to_rgb24(c: &mut Criterion) {
    let mut group = c.benchmark_group("yuv422_to_rgb24");
    let (w, h) = (1280, 720);
    let (yp, up, vp) = {
        let src = synth_rgb24(w, h);
        let cw = w / 2;
        let mut yp = vec![0u8; w * h];
        let mut up = vec![0u8; cw * h];
        let mut vp = vec![0u8; cw * h];
        yuv::rgb24_to_yuv422(&src, &mut yp, &mut up, &mut vp, w, h, YuvMatrix::BT709);
        (yp, up, vp)
    };
    let mut dst = vec![0u8; w * h * 3];
    group.throughput(Throughput::Bytes((w * h * 3) as u64));
    group.bench_function("1280x720_bt709_limited", |b| {
        b.iter(|| {
            yuv::yuv422_to_rgb24(&yp, &up, &vp, &mut dst, w, h, YuvMatrix::BT709);
        });
    });
    group.finish();
}

criterion_group!(
    name = yuv_benches;
    config = Criterion::default().sample_size(30);
    targets =
        bench_yuv420_to_rgb24,
        bench_yuv422_to_rgb24,
        bench_yuv444_to_rgb24,
        bench_rgb24_to_yuv420,
        bench_rgb24_to_yuv422,
        bench_rgb24_to_yuv444
);
criterion_main!(yuv_benches);
