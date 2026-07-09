//! Benchmarks for the per-plane bit-depth primitives and the Gray8
//! luminance projection — the loops behind the depth-ladder and
//! RGB → Gray8 `convert()` paths.
//!
//! Run with: `cargo bench --features bench --bench depth_gray`

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use oxideav_pixfmt::yuv::{
    depth_down_le16_plane, depth_rescale_le16_plane, depth_up_8_to_le16_plane, rgb24_to_gray8,
    YuvMatrix,
};

const W: usize = 1920;
const H: usize = 1080;

fn bench_depth(c: &mut Criterion) {
    let count = W * H;
    let src8: Vec<u8> = (0..count).map(|i| (i % 256) as u8).collect();
    let mut src16 = vec![0u8; count * 2];
    depth_up_8_to_le16_plane(&src8, &mut src16, count, 10);

    let mut g = c.benchmark_group("depth_plane_1080p");
    g.throughput(Throughput::Bytes((count * 2) as u64));

    g.bench_function("up_8_to_10le", |b| {
        let mut dst = vec![0u8; count * 2];
        b.iter(|| depth_up_8_to_le16_plane(&src8, &mut dst, count, 10));
    });
    g.bench_function("down_10le_to_8", |b| {
        let mut dst = vec![0u8; count];
        b.iter(|| depth_down_le16_plane(&src16, &mut dst, count, 10));
    });
    g.bench_function("rescale_10le_to_12le", |b| {
        let mut dst = vec![0u8; count * 2];
        b.iter(|| depth_rescale_le16_plane(&src16, &mut dst, count, 10, 12));
    });
    g.bench_function("rescale_12le_to_10le", |b| {
        let mut dst = vec![0u8; count * 2];
        b.iter(|| depth_rescale_le16_plane(&src16, &mut dst, count, 12, 10));
    });
    g.finish();
}

fn bench_gray(c: &mut Criterion) {
    let pixels = W * H;
    let rgb: Vec<u8> = (0..pixels * 3).map(|i| ((i * 7) % 256) as u8).collect();

    let mut g = c.benchmark_group("rgb_to_gray_1080p");
    g.throughput(Throughput::Bytes((pixels * 3) as u64));
    g.bench_function("rgb24_to_gray8_bt601_full", |b| {
        let mut dst = vec![0u8; pixels];
        let m = YuvMatrix::BT601.with_range(false);
        b.iter(|| rgb24_to_gray8(&rgb, &mut dst, pixels, m));
    });
    g.finish();
}

criterion_group!(benches, bench_depth, bench_gray);
criterion_main!(benches);
