//! YUV ↔ RGB roundtrip tests. 4:4:4 is near-lossless (> 38 dB); 4:2:0
//! loses detail on chroma transitions (> 30 dB is the expected floor).

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

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
