//! Criterion benchmarks for the memory-bound pixel-format paths that
//! the YUV bench doesn't touch: RGB swizzles, Gray/Mono, NV12↔YUV420,
//! deep-RGB bit-depth changes, and the planar chroma resamplers.
//!
//! All benches report throughput based on the output-buffer size so the
//! numbers compare directly against the YUV benches. Frames are
//! synthesised once outside the measured region.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_pixfmt::{
    gray,
    rgb::{self, Rgb3, Rgba4, ABGR_POS, ARGB_POS, BGRA_POS, BGR_POS, RGBA_POS, RGB_POS},
    yuv,
};

fn synth(w: usize, h: usize, stride: usize) -> Vec<u8> {
    // Cheap deterministic filler; content doesn't matter for format shuffles.
    let n = w * h * stride;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(((i * 2654435761) >> 24) as u8);
    }
    v
}

// -------------------------------------------------------------------------
// RGB swizzles — rgb↔bgr, rgba↔bgra↔argb↔abgr.

fn run_swizzle3(src: &[u8], dst: &mut [u8], sp: Rgb3, dp: Rgb3, w: usize, h: usize) {
    for row in 0..h {
        rgb::swizzle3(
            &src[row * w * 3..row * w * 3 + w * 3],
            sp,
            &mut dst[row * w * 3..row * w * 3 + w * 3],
            dp,
            w,
        );
    }
}

fn run_swizzle4(src: &[u8], dst: &mut [u8], sp: Rgba4, dp: Rgba4, w: usize, h: usize) {
    for row in 0..h {
        rgb::swizzle4(
            &src[row * w * 4..row * w * 4 + w * 4],
            sp,
            &mut dst[row * w * 4..row * w * 4 + w * 4],
            dp,
            w,
        );
    }
}

fn bench_rgb_swizzles(c: &mut Criterion) {
    let (w, h) = (1920, 1080);
    let src3 = synth(w, h, 3);
    let src4 = synth(w, h, 4);
    let mut dst3 = vec![0u8; w * h * 3];
    let mut dst4 = vec![0u8; w * h * 4];

    let mut g3 = c.benchmark_group("rgb24_swizzle");
    g3.throughput(Throughput::Bytes((w * h * 3) as u64));
    g3.bench_function("rgb_to_bgr_1920x1080", |b| {
        b.iter(|| run_swizzle3(&src3, &mut dst3, RGB_POS, BGR_POS, w, h));
    });
    g3.finish();

    let mut g4 = c.benchmark_group("rgba_swizzle");
    g4.throughput(Throughput::Bytes((w * h * 4) as u64));
    g4.bench_function("rgba_to_bgra_1920x1080", |b| {
        b.iter(|| run_swizzle4(&src4, &mut dst4, RGBA_POS, BGRA_POS, w, h));
    });
    g4.bench_function("rgba_to_argb_1920x1080", |b| {
        b.iter(|| run_swizzle4(&src4, &mut dst4, RGBA_POS, ARGB_POS, w, h));
    });
    g4.bench_function("rgba_to_abgr_1920x1080", |b| {
        b.iter(|| run_swizzle4(&src4, &mut dst4, RGBA_POS, ABGR_POS, w, h));
    });
    g4.finish();
}

// -------------------------------------------------------------------------
// RGB 3 ↔ 4 promote/demote.

fn bench_rgb_promote_demote(c: &mut Criterion) {
    let (w, h) = (1920, 1080);
    let src3 = synth(w, h, 3);
    let src4 = synth(w, h, 4);
    let mut dst3 = vec![0u8; w * h * 3];
    let mut dst4 = vec![0u8; w * h * 4];

    let mut gp = c.benchmark_group("rgb24_to_rgba");
    gp.throughput(Throughput::Bytes((w * h * 4) as u64));
    gp.bench_function("1920x1080", |b| {
        b.iter(|| {
            for row in 0..h {
                rgb::rgb3_to_rgba4(
                    &src3[row * w * 3..row * w * 3 + w * 3],
                    RGB_POS,
                    &mut dst4[row * w * 4..row * w * 4 + w * 4],
                    RGBA_POS,
                    w,
                );
            }
        });
    });
    gp.finish();

    let mut gd = c.benchmark_group("rgba_to_rgb24");
    gd.throughput(Throughput::Bytes((w * h * 3) as u64));
    gd.bench_function("1920x1080", |b| {
        b.iter(|| {
            for row in 0..h {
                rgb::rgba4_to_rgb3(
                    &src4[row * w * 4..row * w * 4 + w * 4],
                    RGBA_POS,
                    &mut dst3[row * w * 3..row * w * 3 + w * 3],
                    RGB_POS,
                    w,
                );
            }
        });
    });
    gd.finish();
}

// -------------------------------------------------------------------------
// Gray8 broadcast + Gray16 truncate + RGB48 high-byte extract.

fn bench_gray(c: &mut Criterion) {
    let (w, h) = (1920, 1080);
    let src8 = synth(w, h, 1);
    let src16 = synth(w, h, 2);
    let mut dst_rgb = vec![0u8; w * h * 3];
    let mut dst_rgba = vec![0u8; w * h * 4];
    let mut dst_g8 = vec![0u8; w * h];

    let mut g = c.benchmark_group("gray");
    g.throughput(Throughput::Bytes((w * h * 3) as u64));
    g.bench_function("gray8_to_rgb24_1920x1080", |b| {
        b.iter(|| gray::gray8_to_rgb24(&src8, &mut dst_rgb, w * h));
    });
    g.throughput(Throughput::Bytes((w * h * 4) as u64));
    g.bench_function("gray8_to_rgba_1920x1080", |b| {
        b.iter(|| gray::gray8_to_rgba(&src8, &mut dst_rgba, w * h));
    });
    g.throughput(Throughput::Bytes((w * h) as u64));
    g.bench_function("gray16le_to_gray8_1920x1080", |b| {
        b.iter(|| gray::gray16le_to_gray8(&src16, &mut dst_g8, w * h));
    });
    g.finish();
}

fn bench_deep_rgb(c: &mut Criterion) {
    let (w, h) = (1920, 1080);
    let src48 = synth(w, h, 6);
    let src24 = synth(w, h, 3);
    let mut dst24 = vec![0u8; w * h * 3];
    let mut dst48 = vec![0u8; w * h * 6];

    let mut g = c.benchmark_group("rgb_bitdepth");
    g.throughput(Throughput::Bytes((w * h * 3) as u64));
    g.bench_function("rgb48_to_rgb24_1920x1080", |b| {
        b.iter(|| rgb::rgb48_to_rgb24(&src48, &mut dst24, w * h));
    });
    g.throughput(Throughput::Bytes((w * h * 6) as u64));
    g.bench_function("rgb24_to_rgb48_1920x1080", |b| {
        b.iter(|| rgb::rgb24_to_rgb48(&src24, &mut dst48, w * h));
    });
    g.finish();
}

// -------------------------------------------------------------------------
// NV12 / NV21 split + merge at chroma resolution.

fn bench_nv12(c: &mut Criterion) {
    let (w, h) = (1920, 1080);
    let cw = w / 2;
    let ch = h / 2;
    let uv = synth(cw, ch, 2);
    let up = vec![0u8; cw * ch];
    let vp = vec![0u8; cw * ch];
    let mut up2 = up.clone();
    let mut vp2 = vp.clone();
    let mut uv_out = vec![0u8; cw * ch * 2];

    let mut g = c.benchmark_group("nv_chroma");
    g.throughput(Throughput::Bytes((cw * ch * 2) as u64));
    g.bench_function("nv12_split_1920x1080", |b| {
        b.iter(|| yuv::nv12_uv_split(&uv, &mut up2, &mut vp2, cw, ch));
    });
    g.bench_function("nv21_split_1920x1080", |b| {
        b.iter(|| yuv::nv21_vu_split(&uv, &mut up2, &mut vp2, cw, ch));
    });
    g.bench_function("nv12_merge_1920x1080", |b| {
        b.iter(|| yuv::nv12_uv_merge(&up, &vp, &mut uv_out, cw, ch));
    });
    g.finish();
}

// -------------------------------------------------------------------------
// Planar chroma resample 444↔422↔420.

fn bench_chroma_resample(c: &mut Criterion) {
    let (w, h) = (1920, 1080);
    let src444 = synth(w, h, 1);
    let src422 = synth(w / 2, h, 1);
    let src420 = synth(w / 2, h / 2, 1);
    let mut dst_422 = vec![0u8; (w / 2) * h];
    let mut dst_420 = vec![0u8; (w / 2) * (h / 2)];
    let mut dst_444 = vec![0u8; w * h];

    let mut g = c.benchmark_group("chroma_resample");
    g.throughput(Throughput::Bytes(((w / 2) * h) as u64));
    g.bench_function("444_to_422_1920x1080", |b| {
        b.iter(|| yuv::chroma_444_to_422(&src444, &mut dst_422, w, h));
    });
    g.throughput(Throughput::Bytes(((w / 2) * (h / 2)) as u64));
    g.bench_function("444_to_420_1920x1080", |b| {
        b.iter(|| yuv::chroma_444_to_420(&src444, &mut dst_420, w, h));
    });
    g.throughput(Throughput::Bytes(((w / 2) * (h / 2)) as u64));
    g.bench_function("422_to_420_1920x1080", |b| {
        b.iter(|| yuv::chroma_422_to_420(&src422, &mut dst_420, w, h));
    });
    g.throughput(Throughput::Bytes((w * h) as u64));
    g.bench_function("422_to_444_1920x1080", |b| {
        b.iter(|| yuv::chroma_422_to_444(&src422, &mut dst_444, w, h));
    });
    g.throughput(Throughput::Bytes(((w / 2) * h) as u64));
    g.bench_function("420_to_422_1920x1080", |b| {
        b.iter(|| yuv::chroma_420_to_422(&src420, &mut dst_422, w, h));
    });
    g.throughput(Throughput::Bytes((w * h) as u64));
    g.bench_function("420_to_444_1920x1080", |b| {
        b.iter(|| yuv::chroma_420_to_444(&src420, &mut dst_444, w, h));
    });
    g.finish();
}

criterion_group!(
    name = pixel_ops;
    config = Criterion::default().sample_size(30);
    targets =
        bench_rgb_swizzles,
        bench_rgb_promote_demote,
        bench_gray,
        bench_deep_rgb,
        bench_nv12,
        bench_chroma_resample
);
criterion_main!(pixel_ops);
