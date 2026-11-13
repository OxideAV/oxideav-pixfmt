//! Criterion benchmarks for `icc::IccTransform` at 1 MP and 12 MP:
//! matrix/TRC → matrix/TRC (built-in sRGB → Display P3) and, when the
//! macOS system profile is present, the 4-D CLUT of Generic CMYK in
//! both directions — each as the exact transform (`exact_*`), with
//! tetrahedral CLUT interpolation (`tetra_*`) and through a
//! precomputed table (`table_*`: 33 points per RGB channel, 17 per
//! CMYK channel).
//!
//! Suffix `_mt` runs with the host's thread budget.
//!
//! Run with `cargo bench --features bench,icc --bench icc_transform`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxideav_core::ExecutionContext;
use oxideav_pixfmt::icc::{
    ClutInterpolation, IccProfile, IccTransform, RenderingIntent, TransformOptions,
};

fn synth_u16(n: usize) -> Vec<u16> {
    let mut s: u32 = 0x9E37_79B9;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u16
        })
        .collect()
}

fn bench_pair(
    c: &mut Criterion,
    name: &str,
    src: &IccProfile,
    dst: &IccProfile,
    grid: usize,
    sizes: &[(&str, usize)],
) {
    let base = TransformOptions::new().with_intent(RenderingIntent::RelativeColorimetric);
    let auto = ExecutionContext::auto();
    let exact = IccTransform::with_options(src, dst, base).unwrap();
    let exact_mt = IccTransform::with_options(src, dst, base.with_execution(&auto)).unwrap();
    let tetra = IccTransform::with_options(
        src,
        dst,
        base.with_interpolation(ClutInterpolation::Tetrahedral),
    )
    .unwrap();
    let table = exact.precompute(grid).unwrap();
    let table_mt = exact_mt.precompute(grid).unwrap();
    let (ic, oc) = (exact.input_channels(), exact.output_channels());
    c.bench_function(format!("icc_{name}/build_table_{grid}"), |b| {
        b.iter(|| exact.precompute(grid).unwrap())
    });
    for &(label, px) in sizes {
        let s16 = synth_u16(px * ic);
        let s8: Vec<u8> = s16.iter().map(|&v| (v >> 8) as u8).collect();
        let sf: Vec<f32> = s16.iter().map(|&v| v as f32 / 65535.0).collect();
        let mut d16 = vec![0u16; px * oc];
        let mut d8 = vec![0u8; px * oc];
        let mut df = vec![0f32; px * oc];
        let mut g = c.benchmark_group(format!("icc_{name}_{label}"));
        g.throughput(Throughput::Elements(px as u64));
        g.sample_size(10);
        let variants: [(&str, &IccTransform); 5] = [
            ("exact", &exact),
            ("exact_mt", &exact_mt),
            ("tetra", &tetra),
            ("table", &table),
            ("table_mt", &table_mt),
        ];
        for (v, t) in variants {
            g.bench_function(format!("{v}_u8"), |b| {
                b.iter(|| t.transform_u8(&s8, &mut d8).unwrap())
            });
            g.bench_function(format!("{v}_u16"), |b| {
                b.iter(|| t.transform_u16(&s16, &mut d16).unwrap())
            });
            g.bench_function(format!("{v}_f32"), |b| {
                b.iter(|| t.transform_f32(&sf, &mut df).unwrap())
            });
        }
        g.finish();
    }
}

fn bench_icc(c: &mut Criterion) {
    let sizes = [("1mp", 1_000_000usize), ("12mp", 4032 * 3024)];
    let srgb = IccProfile::srgb();
    let p3 = IccProfile::display_p3();
    bench_pair(c, "srgb_to_p3", &srgb, &p3, 33, &sizes);
    // A CLUT-based printer profile, read as an opaque input when the
    // host has one.
    let path = "/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc";
    if let Some(cmyk) = std::fs::read(path)
        .ok()
        .and_then(|b| IccProfile::parse(&b).ok())
    {
        bench_pair(c, "srgb_to_cmyk", &srgb, &cmyk, 33, &sizes[..1]);
        bench_pair(c, "cmyk_to_srgb", &cmyk, &srgb, 17, &sizes[..1]);
    } else {
        eprintln!("no system CMYK profile; CLUT cases skipped");
    }
}

criterion_group!(benches, bench_icc);
criterion_main!(benches);
