//! YUV ↔ RGB roundtrip tests. 4:4:4 is near-lossless (> 38 dB); 4:2:0
//! loses detail on chroma transitions (> 30 dB is the expected floor).

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ColorSpace, ConvertOptions, FrameInfo};

fn synth_rgb24(w: u32, h: u32) -> (VideoFrame, FrameInfo) {
    // Smooth gradients in each channel — the usual PSNR benchmark. High-
    // frequency noise patterns are out of scope for a subsample-loss
    // assertion.
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / (w - 1).max(1)) as u8;
            let g = ((y * 255) / (h - 1).max(1)) as u8;
            let b = (((x + y) * 255) / ((w + h) - 2).max(1)) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    (
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        },
        FrameInfo::new(PixelFormat::Rgb24, w, h),
    )
}

fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut sq = 0.0f64;
    for i in 0..n {
        let d = a[i] as f64 - b[i] as f64;
        sq += d * d;
    }
    if sq == 0.0 {
        return f64::INFINITY;
    }
    let mse = sq / n as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

#[test]
fn rgb_to_yuv444_and_back_is_near_lossless() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(64, 48);
    let yuv = convert(&src, src_info, PixelFormat::Yuv444P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv444P, src_info.width, src_info.height);
    let back = convert(&yuv, yuv_info, PixelFormat::Rgb24, &opts).unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("yuv444 psnr = {psnr:.2}");
    assert!(psnr > 38.0, "yuv444 psnr too low: {psnr}");
}

#[test]
fn rgb_to_yuv420_and_back_exceeds_30_db() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(64, 48);
    let yuv = convert(&src, src_info, PixelFormat::Yuv420P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv420P, src_info.width, src_info.height);
    let back = convert(&yuv, yuv_info, PixelFormat::Rgb24, &opts).unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("yuv420 psnr = {psnr:.2}");
    assert!(psnr > 30.0, "yuv420 psnr too low: {psnr}");
}

#[test]
fn rgb_to_yuv422_intermediate() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(64, 48);
    let yuv = convert(&src, src_info, PixelFormat::Yuv422P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv422P, src_info.width, src_info.height);
    let back = convert(&yuv, yuv_info, PixelFormat::Rgb24, &opts).unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("yuv422 psnr = {psnr:.2}");
    assert!(psnr > 33.0, "yuv422 psnr too low: {psnr}");
}

#[test]
fn nv12_roundtrips_yuv420p() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    let yuv = convert(&src, src_info, PixelFormat::Yuv420P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv420P, src_info.width, src_info.height);
    let nv12 = convert(&yuv, yuv_info, PixelFormat::Nv12, &opts).unwrap();
    let nv12_info = FrameInfo::new(PixelFormat::Nv12, src_info.width, src_info.height);
    let back = convert(&nv12, nv12_info, PixelFormat::Yuv420P, &opts).unwrap();
    assert_eq!(yuv.planes[0].data, back.planes[0].data, "Y plane");
    assert_eq!(yuv.planes[1].data, back.planes[1].data, "U plane");
    assert_eq!(yuv.planes[2].data, back.planes[2].data, "V plane");
}

#[test]
fn nv21_roundtrips_yuv420p() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    let yuv = convert(&src, src_info, PixelFormat::Yuv420P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv420P, src_info.width, src_info.height);
    let nv21 = convert(&yuv, yuv_info, PixelFormat::Nv21, &opts).unwrap();
    let nv21_info = FrameInfo::new(PixelFormat::Nv21, src_info.width, src_info.height);
    let back = convert(&nv21, nv21_info, PixelFormat::Yuv420P, &opts).unwrap();
    assert_eq!(yuv.planes[1].data, back.planes[1].data, "U plane");
    assert_eq!(yuv.planes[2].data, back.planes[2].data, "V plane");
}

// Direct NV12 / NV21 ↔ Rgb24 / Rgba. The fused path must produce the
// same bytes as the two-step `Nv → Yuv420P → Rgb` route — the
// dispatch just spares the caller one frame allocation, so the output
// is identical by construction.

#[test]
fn nv12_to_rgb24_matches_staged_nv12_yuv420p_rgb24() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    // Build an NV12 frame from the RGB source via Yuv420P.
    let yuv = convert(&src, src_info, PixelFormat::Yuv420P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv420P, src_info.width, src_info.height);
    let nv12 = convert(&yuv, yuv_info, PixelFormat::Nv12, &opts).unwrap();
    let nv12_info = FrameInfo::new(PixelFormat::Nv12, src_info.width, src_info.height);
    // Compare the fused path against the explicit two-step route.
    let direct = convert(&nv12, nv12_info, PixelFormat::Rgb24, &opts).unwrap();
    let staged_yuv = convert(&nv12, nv12_info, PixelFormat::Yuv420P, &opts).unwrap();
    let staged_yuv_info = yuv_info;
    let staged = convert(&staged_yuv, staged_yuv_info, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(direct.planes[0].data, staged.planes[0].data);
}

#[test]
fn nv21_to_rgba_matches_staged_nv21_yuv420p_rgba() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    let yuv = convert(&src, src_info, PixelFormat::Yuv420P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv420P, src_info.width, src_info.height);
    let nv21 = convert(&yuv, yuv_info, PixelFormat::Nv21, &opts).unwrap();
    let nv21_info = FrameInfo::new(PixelFormat::Nv21, src_info.width, src_info.height);
    let direct = convert(&nv21, nv21_info, PixelFormat::Rgba, &opts).unwrap();
    let staged_yuv = convert(&nv21, nv21_info, PixelFormat::Yuv420P, &opts).unwrap();
    let staged = convert(&staged_yuv, yuv_info, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(direct.planes[0].data, staged.planes[0].data);
    // And the alpha channel is opaque white throughout.
    for i in 0..(src_info.width * src_info.height) as usize {
        assert_eq!(direct.planes[0].data[i * 4 + 3], 255);
    }
}

#[test]
fn rgb24_to_nv12_matches_staged_rgb24_yuv420p_nv12() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    let direct = convert(&src, src_info, PixelFormat::Nv12, &opts).unwrap();
    let staged_yuv = convert(&src, src_info, PixelFormat::Yuv420P, &opts).unwrap();
    let staged_yuv_info = FrameInfo::new(PixelFormat::Yuv420P, src_info.width, src_info.height);
    let staged = convert(&staged_yuv, staged_yuv_info, PixelFormat::Nv12, &opts).unwrap();
    assert_eq!(direct.planes[0].data, staged.planes[0].data, "Y plane");
    assert_eq!(direct.planes[1].data, staged.planes[1].data, "UV plane");
}

#[test]
fn rgba_to_nv21_matches_staged_rgba_yuv420p_nv21() {
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(32, 16);
    // Promote to Rgba first so we can exercise the `alpha_in` path.
    let rgba = convert(&src, src_info, PixelFormat::Rgba, &opts).unwrap();
    let rgba_info = FrameInfo::new(PixelFormat::Rgba, src_info.width, src_info.height);
    let direct = convert(&rgba, rgba_info, PixelFormat::Nv21, &opts).unwrap();
    let staged_yuv = convert(&rgba, rgba_info, PixelFormat::Yuv420P, &opts).unwrap();
    let staged_yuv_info = FrameInfo::new(PixelFormat::Yuv420P, src_info.width, src_info.height);
    let staged = convert(&staged_yuv, staged_yuv_info, PixelFormat::Nv21, &opts).unwrap();
    assert_eq!(direct.planes[0].data, staged.planes[0].data, "Y plane");
    assert_eq!(direct.planes[1].data, staged.planes[1].data, "VU plane");
}

#[test]
fn rgb_nv12_roundtrip_meets_yuv420_psnr_floor() {
    // Round-trips Rgb24 → Nv12 → Rgb24 and asserts the gradient PSNR
    // stays in the same envelope as the planar 4:2:0 path. The fused
    // route shares the planar encoder/decoder so this is a coverage
    // backstop, not an independent quality measurement.
    let opts = ConvertOptions::default();
    let (src, src_info) = synth_rgb24(64, 48);
    let nv12 = convert(&src, src_info, PixelFormat::Nv12, &opts).unwrap();
    let nv12_info = FrameInfo::new(PixelFormat::Nv12, src_info.width, src_info.height);
    let back = convert(&nv12, nv12_info, PixelFormat::Rgb24, &opts).unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("nv12 rgb24 round-trip psnr = {psnr:.2}");
    assert!(psnr > 30.0, "nv12 round-trip psnr too low: {psnr}");
}

#[test]
fn nv12_to_rgb24_rejects_odd_dimensions() {
    let opts = ConvertOptions::default();
    // Build a tiny NV12 frame with odd width — the dispatch must
    // refuse rather than truncate.
    let bad = VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: 3,
                data: vec![16u8; 3 * 4],
            },
            VideoPlane {
                stride: 2,
                data: vec![128u8; 2 * 2],
            },
        ],
    };
    let info = FrameInfo::new(PixelFormat::Nv12, 3, 4);
    assert!(convert(&bad, info, PixelFormat::Rgb24, &opts).is_err());
}

// BT.2020 NCL matrix — kr = 0.2627, kb = 0.0593 from BT.2020-2 Table 4
// (same coefficients re-used by BT.2100-3 Table 6).

#[test]
fn bt2020_limited_yuv444_roundtrip_is_near_lossless() {
    let opts = ConvertOptions {
        color_space: ColorSpace::Bt2020Limited,
        ..Default::default()
    };
    let (src, src_info) = synth_rgb24(64, 48);
    let yuv = convert(&src, src_info, PixelFormat::Yuv444P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv444P, src_info.width, src_info.height);
    let back = convert(&yuv, yuv_info, PixelFormat::Rgb24, &opts).unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("bt2020 yuv444 psnr = {psnr:.2}");
    assert!(psnr > 38.0, "bt2020 yuv444 psnr too low: {psnr}");
}

#[test]
fn bt2020_full_yuv444_roundtrip_is_lossless_for_neutrals() {
    let opts = ConvertOptions {
        color_space: ColorSpace::Bt2020Full,
        ..Default::default()
    };
    let (src, src_info) = synth_rgb24(32, 32);
    let yuv = convert(&src, src_info, PixelFormat::Yuv444P, &opts).unwrap();
    let yuv_info = FrameInfo::new(PixelFormat::Yuv444P, src_info.width, src_info.height);
    let back = convert(&yuv, yuv_info, PixelFormat::Rgb24, &opts).unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("bt2020 full yuv444 psnr = {psnr:.2}");
    // Full-range avoids the 16–235 narrow-range quantisation step and
    // is normally several dB cleaner than the limited variant.
    assert!(psnr > 42.0, "bt2020 full yuv444 psnr too low: {psnr}");
}

#[test]
fn bt2020_neutral_grey_round_trips_to_chroma_128() {
    // Neutral grey (R = G = B = 128) must produce Cb = Cr = 128 in
    // limited and full range alike — the YCbCr coefficient design
    // intrinsically forces equal-RGB pixels onto the chroma origin.
    use oxideav_pixfmt::yuv::{rgb_to_yuv, YuvMatrix};
    let mat = YuvMatrix::BT2020.with_range(true);
    let (_, cb, cr) = rgb_to_yuv(128, 128, 128, mat);
    assert!(
        cb.abs_diff(128) <= 1 && cr.abs_diff(128) <= 1,
        "expected cb=cr=128, got cb={cb} cr={cr}"
    );

    let mat_full = YuvMatrix::BT2020.with_range(false);
    let (_, cbf, crf) = rgb_to_yuv(128, 128, 128, mat_full);
    assert!(
        cbf.abs_diff(128) <= 1 && crf.abs_diff(128) <= 1,
        "expected full cb=cr=128, got cb={cbf} cr={crf}"
    );
}

#[test]
fn bt2020_pure_white_lands_in_luma() {
    // R=G=B=255 should produce the limited-range white code (Y' ≈ 235)
    // with neutral chroma. Verifies that the luma coefficient sum
    // 0.2627 + 0.6780 + 0.0593 = 1.0 is wired correctly.
    use oxideav_pixfmt::yuv::{rgb_to_yuv, YuvMatrix};
    let (y, cb, cr) = rgb_to_yuv(255, 255, 255, YuvMatrix::BT2020.with_range(true));
    assert!((233..=237).contains(&y), "expected y near 235, got {y}");
    assert!(cb.abs_diff(128) <= 1, "expected cb ≈ 128, got {cb}");
    assert!(cr.abs_diff(128) <= 1, "expected cr ≈ 128, got {cr}");
}
