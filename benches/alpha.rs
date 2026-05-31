//! Criterion benchmarks for the Porter-Duff alpha primitives.
//!
//! `alpha.rs` powers the font / subtitle / overlay compositing path —
//! every glyph blit and every full-frame overlay routes through
//! `blit_alpha_mask` and `over_buffer`. Until now those hot paths had
//! tests but no throughput coverage; this suite gives us the baseline
//! so future SIMD work has something to land against.
//!
//! Sizes:
//! * **Per-pixel**: `over_premul` / `over_straight` looped over a
//!   1 Mpx buffer — characterises the scalar inner loop.
//! * **Bulk**: `over_buffer` premul + straight at 1920×1080 — the
//!   actual full-frame composite latency.
//! * **Mask blit**: a 64×64 glyph (typical CJK ideograph) blitted onto
//!   1920×1080, plus a tighter 16×16 ASCII-glyph case so glyph-cache
//!   regressions show up as the inner-loop overhead.
//! * **Premultiply roundtrip**: bulk forward + inverse over 1 Mpx —
//!   used by every straight-alpha decoder that hands frames to a
//!   premultiplied compositor.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_pixfmt::{
    blit_alpha_mask, modulate_alpha, over_buffer, over_premul, over_straight, premultiply,
    unpremultiply,
};

fn synth(n: usize, seed: u64) -> Vec<u8> {
    // Deterministic pseudo-random filler. The Porter-Duff hot path is
    // data-independent in branch profile (no early-out on transparent
    // pixels, except in `over_straight`'s endpoint short-circuit), so
    // any well-mixed byte stream is representative.
    let mut v = Vec::with_capacity(n);
    let mut s = seed.wrapping_add(0x9E37_79B9);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push((s >> 24) as u8);
    }
    v
}

/// Build a premultiplied-alpha buffer: pick a random colour direction and
/// random alpha, then scale colour by alpha so `C ≤ A` holds per pixel.
fn synth_premul(n_pixels: usize, seed: u64) -> Vec<u8> {
    let raw = synth(n_pixels * 4, seed);
    let mut out = Vec::with_capacity(n_pixels * 4);
    for px in raw.chunks_exact(4) {
        let a = px[3] as u32;
        for c in &px[..3] {
            out.push(((*c as u32 * a) / 255) as u8);
        }
        out.push(px[3]);
    }
    out
}

fn bench_per_pixel(c: &mut Criterion) {
    // 1 Mpx — chosen so the working set fits in L2 (4 MiB per buffer) and
    // we measure the inner loop, not the memory subsystem.
    let n = 1024 * 1024;
    let src = synth_premul(n, 0x1);
    let dst = synth_premul(n, 0x2);
    let mut g = c.benchmark_group("over_per_pixel");
    g.throughput(Throughput::Bytes((n * 4) as u64));
    g.bench_function("over_premul_1Mpx", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for i in 0..n {
                let s = [src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]];
                let d = [dst[i * 4], dst[i * 4 + 1], dst[i * 4 + 2], dst[i * 4 + 3]];
                let o = over_premul(s, d);
                acc = acc.wrapping_add(o[0] as u32);
            }
            black_box(acc);
        });
    });
    let src_s = synth(n * 4, 0x3);
    let dst_s = synth(n * 4, 0x4);
    g.bench_function("over_straight_1Mpx", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for i in 0..n {
                let s = [
                    src_s[i * 4],
                    src_s[i * 4 + 1],
                    src_s[i * 4 + 2],
                    src_s[i * 4 + 3],
                ];
                let d = [
                    dst_s[i * 4],
                    dst_s[i * 4 + 1],
                    dst_s[i * 4 + 2],
                    dst_s[i * 4 + 3],
                ];
                let o = over_straight(s, d);
                acc = acc.wrapping_add(o[0] as u32);
            }
            black_box(acc);
        });
    });
    g.finish();
}

fn bench_over_buffer_1080p(c: &mut Criterion) {
    let w = 1920u32;
    let h = 1080u32;
    let stride = (w * 4) as usize;
    let src_pm = synth_premul((w * h) as usize, 0xa);
    let dst_pm = synth_premul((w * h) as usize, 0xb);
    let src_st = synth(stride * h as usize, 0xc);
    let dst_st = synth(stride * h as usize, 0xd);

    let mut g = c.benchmark_group("over_buffer_1920x1080");
    g.throughput(Throughput::Bytes((stride * h as usize) as u64));
    g.bench_function("premul", |b| {
        b.iter_batched_ref(
            || dst_pm.clone(),
            |dst| over_buffer(dst, &src_pm, w, h, stride, true),
            criterion::BatchSize::LargeInput,
        );
    });
    g.bench_function("straight", |b| {
        b.iter_batched_ref(
            || dst_st.clone(),
            |dst| over_buffer(dst, &src_st, w, h, stride, false),
            criterion::BatchSize::LargeInput,
        );
    });
    g.finish();
}

fn bench_blit_alpha_mask(c: &mut Criterion) {
    let dst_w = 1920u32;
    let dst_h = 1080u32;
    let dst_stride = (dst_w * 4) as usize;
    // Pre-build the destination once outside the measured region. Each
    // iteration restarts from a clone so consecutive blits don't dirty
    // the next iteration's input (would bias the second run's branch
    // history toward the result of the first).
    let dst_template = synth(dst_stride * dst_h as usize, 0x10);

    // 16×16 glyph: typical Latin ASCII at 16 px line-height. The mask
    // is anti-aliased grayscale; the bench blits 1024 of them in a row
    // (covering 1 Mpx total) so the per-glyph fixed cost shows up.
    let small_mask = synth(16 * 16, 0x11);
    // 64×64 glyph: typical CJK ideograph at 64 px line-height.
    let big_mask = synth(64 * 64, 0x12);

    let mut g = c.benchmark_group("blit_alpha_mask");
    // Report mask-pixel throughput so the two sizes are directly comparable.
    g.throughput(Throughput::Bytes(16 * 16 * 1024));
    g.bench_function("16x16_glyph_1024_times", |b| {
        b.iter_batched_ref(
            || dst_template.clone(),
            |dst| {
                for k in 0..1024i32 {
                    let x = (k % 120) * 16;
                    let y = (k / 120) * 16;
                    blit_alpha_mask(
                        dst,
                        dst_w,
                        dst_h,
                        dst_stride,
                        x,
                        y,
                        &small_mask,
                        16,
                        16,
                        16,
                        [200, 100, 50, 255],
                    );
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });
    g.throughput(Throughput::Bytes(64 * 64 * 256));
    g.bench_function("64x64_glyph_256_times", |b| {
        b.iter_batched_ref(
            || dst_template.clone(),
            |dst| {
                for k in 0..256i32 {
                    let x = (k % 30) * 64;
                    let y = (k / 30) * 64;
                    blit_alpha_mask(
                        dst,
                        dst_w,
                        dst_h,
                        dst_stride,
                        x,
                        y,
                        &big_mask,
                        64,
                        64,
                        64,
                        [60, 180, 220, 255],
                    );
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });
    g.finish();
}

fn bench_premultiply_roundtrip(c: &mut Criterion) {
    let n = 1024 * 1024;
    let raw = synth(n * 4, 0x20);
    let mut g = c.benchmark_group("premultiply_roundtrip");
    g.throughput(Throughput::Bytes((n * 4) as u64));
    g.bench_function("premultiply_1Mpx", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for i in 0..n {
                let p = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                let o = premultiply(p);
                acc = acc.wrapping_add(o[0] as u32);
            }
            black_box(acc);
        });
    });
    g.bench_function("unpremultiply_1Mpx", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for i in 0..n {
                let p = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                let o = unpremultiply(p);
                acc = acc.wrapping_add(o[0] as u32);
            }
            black_box(acc);
        });
    });
    g.bench_function("modulate_alpha_1Mpx", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for i in 0..n {
                let p = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                let o = modulate_alpha(p, 128);
                acc = acc.wrapping_add(o[3] as u32);
            }
            black_box(acc);
        });
    });
    g.finish();
}

criterion_group!(
    name = alpha;
    config = Criterion::default().sample_size(30);
    targets =
        bench_per_pixel,
        bench_over_buffer_1080p,
        bench_blit_alpha_mask,
        bench_premultiply_roundtrip
);
criterion_main!(alpha);
