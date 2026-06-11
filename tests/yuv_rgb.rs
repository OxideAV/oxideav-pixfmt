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

// Anchor vectors derived by hand from BT.2020-2 Table 4 (NCL column) +
// Table 5 quantization at n = 8 (so the 2^(n-8) factor is 1):
//
//   Y'  = 0.2627 R' + 0.6780 G' + 0.0593 B'
//   C'B = (B' - Y') / 1.8814         (1.8814 = 2·(1 - 0.0593))
//   C'R = (R' - Y') / 1.4746         (1.4746 = 2·(1 - 0.2627))
//   DY' = INT[219·Y' + 16]    DC' = INT[224·C' + 128]   (limited range)
//
// Full-range rows replace the 219/+16 and 224 scalings with 255 (the
// crate-wide full-range convention; chroma keeps its +128 offset).
// The exact pre-rounding values are noted per row; the assertions allow
// ±1 LSB for the fixed-point rounding of the implementation.

#[test]
fn bt2020_limited_black_and_primary_anchors() {
    use oxideav_pixfmt::yuv::{rgb_to_yuv, YuvMatrix};
    let m = YuvMatrix::BT2020.with_range(true);
    // (rgb, expected (y, cb, cr)) — exact values in comments.
    let vectors = [
        // Black: Y' = 0 → 16; chroma at the 128 origin.
        ((0u8, 0u8, 0u8), (16u8, 128u8, 128u8)),
        // Red: Y' = 0.2627 → 219·0.2627+16 = 73.53;
        //      C'B = -0.2627/1.8814 → 224·(-0.139630)+128 = 96.72;
        //      C'R = 0.7373/1.4746 = 0.5 exactly → 240.
        ((255, 0, 0), (74, 97, 240)),
        // Green: Y' = 0.6780 → 164.48; C'B = -0.6780/1.8814 → 47.28;
        //        C'R = -0.6780/1.4746 → 25.01.
        ((0, 255, 0), (164, 47, 25)),
        // Blue: Y' = 0.0593 → 28.99; C'B = 0.9407/1.8814 = 0.5 exactly
        //       → 240; C'R = -0.0593/1.4746 → 118.99.
        ((0, 0, 255), (29, 240, 119)),
    ];
    for ((r, g, b), (ey, ecb, ecr)) in vectors {
        let (y, cb, cr) = rgb_to_yuv(r, g, b, m);
        assert!(
            y.abs_diff(ey) <= 1 && cb.abs_diff(ecb) <= 1 && cr.abs_diff(ecr) <= 1,
            "rgb({r},{g},{b}): expected ({ey},{ecb},{ecr})±1, got ({y},{cb},{cr})"
        );
    }
}

#[test]
fn bt2020_full_black_and_primary_anchors() {
    use oxideav_pixfmt::yuv::{rgb_to_yuv, YuvMatrix};
    let m = YuvMatrix::BT2020.with_range(false);
    let vectors = [
        // Black: full-range zero code, chroma origin.
        ((0u8, 0u8, 0u8), (0u8, 128u8, 128u8)),
        // Red: Y = 255·0.2627 = 66.99; Cb = 255·(-0.139630)+128 = 92.39;
        //      Cr = 255·0.5+128 = 255.5 → saturates at the 255 code cap.
        ((255, 0, 0), (67, 92, 255)),
        // Green: Y = 172.89; Cb = 255·(-0.360370)+128 = 36.11;
        //        Cr = 255·(-0.459786)+128 = 10.76.
        ((0, 255, 0), (173, 36, 11)),
        // Blue: Y = 15.12; Cb = 255.5 → 255 cap;
        //       Cr = 255·(-0.040214)+128 = 117.75.
        ((0, 0, 255), (15, 255, 118)),
    ];
    for ((r, g, b), (ey, ecb, ecr)) in vectors {
        let (y, cb, cr) = rgb_to_yuv(r, g, b, m);
        assert!(
            y.abs_diff(ey) <= 1 && cb.abs_diff(ecb) <= 1 && cr.abs_diff(ecr) <= 1,
            "rgb({r},{g},{b}): expected ({ey},{ecb},{ecr})±1, got ({y},{cb},{cr})"
        );
    }
}

#[test]
fn bt2020_anchor_encode_decode_invertibility() {
    // The decode matrix must invert the encode matrix at the gamut
    // extremes, not just on smooth gradients: black, white and the
    // three primaries must survive an encode → decode trip within the
    // ±2 LSB budget that 8-bit chroma quantisation permits (full-range
    // red/blue clip half an LSB of Cr/Cb at the 255 code cap, which is
    // unrecoverable by design and stays inside the same budget).
    use oxideav_pixfmt::yuv::{rgb_to_yuv, yuv_to_rgb, YuvMatrix};
    for limited in [true, false] {
        let m = YuvMatrix::BT2020.with_range(limited);
        for (r, g, b) in [
            (0u8, 0u8, 0u8),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ] {
            let (y, cb, cr) = rgb_to_yuv(r, g, b, m);
            let (r2, g2, b2) = yuv_to_rgb(y, cb, cr, m);
            assert!(
                r2.abs_diff(r) <= 2 && g2.abs_diff(g) <= 2 && b2.abs_diff(b) <= 2,
                "limited={limited} rgb({r},{g},{b}) → yuv({y},{cb},{cr}) → \
                 rgb({r2},{g2},{b2}) drifted more than ±2"
            );
        }
    }
}
