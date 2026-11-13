//! Criterion benchmarks for `colorspace::ModelConverter` at 1 MP and
//! 12 MP, on 8-bit, 16-bit and `f32` buffers, next to the per-colour
//! `ColorModel::convert` loop it replaces (`percolour_*`: one `f64`
//! call per pixel, the matrices re-derived each time).
//!
//! Suffix `_mt` runs with the host's thread budget.
//!
//! Run with `cargo bench --features bench --bench colorspace_models`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxideav_core::ExecutionContext;
use oxideav_pixfmt::colorspace::{ColorModel, LabWhite, LumaWeights, ModelConverter};

fn synth_u16(px: usize) -> Vec<u16> {
    let mut s: u32 = 0x9E37_79B9;
    (0..px * 3)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u16
        })
        .collect()
}

fn bench_models(c: &mut Criterion) {
    let pairs: [(&str, ColorModel, ColorModel); 7] = [
        ("srgb_to_linear", ColorModel::Srgb, ColorModel::LinearSrgb),
        ("linear_to_srgb", ColorModel::LinearSrgb, ColorModel::Srgb),
        ("srgb_to_xyz", ColorModel::Srgb, ColorModel::XyzD65),
        (
            "srgb_to_lab",
            ColorModel::Srgb,
            ColorModel::Lab(LabWhite::D50),
        ),
        (
            "lab_to_srgb",
            ColorModel::Lab(LabWhite::D50),
            ColorModel::Srgb,
        ),
        ("srgb_to_oklch", ColorModel::Srgb, ColorModel::Oklch),
        (
            "srgb_to_ycbcr709",
            ColorModel::Srgb,
            ColorModel::YCbCr(LumaWeights::BT709),
        ),
    ];
    for (label, px) in [("1mp", 1_000_000usize), ("12mp", 4032 * 3024)] {
        // In-gamut sRGB colours; each case's input is their image in
        // the case's source model.
        let rgb16 = synth_u16(px);
        let rgb8: Vec<u8> = rgb16.iter().map(|&v| (v >> 8) as u8).collect();
        let rgbf: Vec<f32> = rgb16.iter().map(|&v| v as f32 / 65535.0).collect();
        let mut src16 = vec![0u16; rgb16.len()];
        let mut src8 = vec![0u8; rgb8.len()];
        let mut srcf = vec![0f32; rgbf.len()];
        let mut dst16 = vec![0u16; rgb16.len()];
        let mut dst8 = vec![0u8; rgb8.len()];
        let mut dstf = vec![0f32; rgbf.len()];
        let mut g = c.benchmark_group(format!("colorspace_{label}"));
        g.throughput(Throughput::Elements(px as u64));
        g.sample_size(10);
        for (name, a, b) in pairs {
            let prep =
                ModelConverter::new(ColorModel::Srgb, a).with_execution(&ExecutionContext::auto());
            prep.convert_u16(&rgb16, &mut src16).unwrap();
            prep.convert_u8(&rgb8, &mut src8).unwrap();
            prep.convert_f32(&rgbf, &mut srcf).unwrap();
            let conv = ModelConverter::new(a, b);
            let mt = conv.clone().with_execution(&ExecutionContext::auto());
            g.bench_function(format!("{name}_u8"), |bn| {
                bn.iter(|| conv.convert_u8(&src8, &mut dst8).unwrap())
            });
            g.bench_function(format!("{name}_u16"), |bn| {
                bn.iter(|| conv.convert_u16(&src16, &mut dst16).unwrap())
            });
            g.bench_function(format!("{name}_f32"), |bn| {
                bn.iter(|| conv.convert_f32(&srcf, &mut dstf).unwrap())
            });
            g.bench_function(format!("{name}_f32_mt"), |bn| {
                bn.iter(|| mt.convert_f32(&srcf, &mut dstf).unwrap())
            });
            g.bench_function(format!("{name}_u8_mt"), |bn| {
                bn.iter(|| mt.convert_u8(&src8, &mut dst8).unwrap())
            });
            if label == "1mp" {
                g.bench_function(format!("percolour_{name}_f32"), |bn| {
                    bn.iter(|| {
                        for (s, d) in srcf
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .zip(dstf.as_chunks_mut::<3>().0)
                        {
                            let v = a.convert(b, [s[0] as f64, s[1] as f64, s[2] as f64]);
                            d[0] = v[0] as f32;
                            d[1] = v[1] as f32;
                            d[2] = v[2] as f32;
                        }
                    })
                });
            }
        }
        g.finish();
    }
}

criterion_group!(benches, bench_models);
criterion_main!(benches);
