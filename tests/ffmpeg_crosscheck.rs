//! Black-box cross-validation of `convert()` against the `ffmpeg`
//! binary used strictly as an opaque CLI validator: raw frames go in,
//! raw frames come out, and the two implementations must agree within a
//! small rounding tolerance. No third-party implementation detail is
//! consulted — only the documented command-line interface.
//!
//! The whole file degrades to a no-op skip when no `ffmpeg` binary is on
//! `PATH` (e.g. bare CI runners), so it adds coverage on developer
//! machines without making CI depend on external tooling.
//!
//! Fixture design notes:
//! * 4:4:4 layouts avoid chroma resampling entirely, so any difference
//!   is pure matrix/rounding — the tolerance is ±2 (each side rounds
//!   once, independently).
//! * The 4:2:0 decode check uses *flat* chroma planes: every chroma
//!   upsampler (nearest, bilinear, …) reproduces a constant plane
//!   exactly, so the comparison stays about the matrix, not about
//!   interpolation policy.
//! * Sample codes stay inside the legal limited range (or full range
//!   where stated) so clamp behaviour at illegal codes — which the two
//!   implementations are free to differ on — never enters the picture.

// Spawning an external process is a foreign operation the miri
// interpreter cannot perform; the whole file is host-only.
#![cfg(not(miri))]

use std::process::Command;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ColorSpace, ConvertOptions, FrameInfo};

const W: usize = 64;
const H: usize = 32;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run ffmpeg as a black box: feed `input` as rawvideo of `in_pix_fmt`,
/// receive rawvideo of `out_pix_fmt`, with explicit matrix / range
/// selection on the scale filter so nothing depends on defaults.
fn ffmpeg_convert(input: &[u8], in_pix_fmt: &str, out_pix_fmt: &str, vf: &str) -> Vec<u8> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir();
    let tag = format!(
        "oxideav-pixfmt-xcheck-{}-{}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed),
        in_pix_fmt,
        out_pix_fmt
    );
    let in_path = dir.join(format!("{tag}.in"));
    let out_path = dir.join(format!("{tag}.out"));
    std::fs::write(&in_path, input).expect("write fixture");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-video_size",
            &format!("{W}x{H}"),
            "-pixel_format",
            in_pix_fmt,
            "-i",
        ])
        .arg(&in_path)
        .args(["-vf", vf, "-f", "rawvideo", "-pix_fmt", out_pix_fmt])
        .arg(&out_path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    let out = std::fs::read(&out_path).expect("read ffmpeg output");
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    out
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len(), "buffer sizes differ");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap_or(0)
}

/// Deterministic in-range sample generator.
fn ramp(n: usize, lo: u8, hi: u8, mul: usize, add: usize) -> Vec<u8> {
    let span = (hi - lo) as usize + 1;
    (0..n)
        .map(|i| lo + ((i * mul + add) % span) as u8)
        .collect()
}

fn planar_frame(planes: Vec<(usize, Vec<u8>)>) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: planes
            .into_iter()
            .map(|(stride, data)| VideoPlane { stride, data })
            .collect(),
    }
}

/// Limited-range 4:4:4 → RGB24 under BT.601 and BT.709: ours vs the
/// black-box validator within ±2.
#[test]
fn yuv444_to_rgb24_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let y = ramp(W * H, 16, 235, 7, 3);
    let u = ramp(W * H, 16, 240, 11, 40);
    let v = ramp(W * H, 16, 240, 13, 80);
    let mut raw = Vec::new();
    raw.extend_from_slice(&y);
    raw.extend_from_slice(&u);
    raw.extend_from_slice(&v);

    for (cs, mat) in [
        (ColorSpace::Bt601Limited, "bt601"),
        (ColorSpace::Bt709Limited, "bt709"),
    ] {
        let theirs = ffmpeg_convert(
            &raw,
            "yuv444p",
            "rgb24",
            &format!("scale=in_color_matrix={mat}:in_range=tv:flags=accurate_rnd"),
        );
        let src = planar_frame(vec![(W, y.clone()), (W, u.clone()), (W, v.clone())]);
        let ours = convert(
            &src,
            FrameInfo::new(PixelFormat::Yuv444P, W as u32, H as u32),
            PixelFormat::Rgb24,
            &ConvertOptions {
                color_space: cs,
                ..Default::default()
            },
        )
        .expect("convert");
        let diff = max_abs_diff(&ours.planes[0].data, &theirs);
        assert!(diff <= 2, "{mat}: max diff {diff}");
    }
}

/// RGB24 → limited-range 4:4:4 under BT.601: every plane within ±2.
#[test]
fn rgb24_to_yuv444_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let rgb = ramp(W * H * 3, 0, 255, 5, 11);
    let theirs = ffmpeg_convert(
        &rgb,
        "rgb24",
        "yuv444p",
        "scale=out_color_matrix=bt601:out_range=tv:flags=accurate_rnd",
    );
    let src = planar_frame(vec![(W * 3, rgb.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Rgb24, W as u32, H as u32),
        PixelFormat::Yuv444P,
        &ConvertOptions::default(),
    )
    .expect("convert");
    for (p, name) in [(0usize, "Y"), (1, "U"), (2, "V")] {
        let their_plane = &theirs[p * W * H..(p + 1) * W * H];
        let diff = max_abs_diff(&ours.planes[p].data, their_plane);
        assert!(diff <= 2, "{name} plane: max diff {diff}");
    }
}

/// Full-range 4:4:4 → RGB24 (the new direct YuvJ path) against the
/// validator run with in_range=pc.
#[test]
fn yuvj444_to_rgb24_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let y = ramp(W * H, 0, 255, 7, 0);
    let u = ramp(W * H, 0, 255, 11, 60);
    let v = ramp(W * H, 0, 255, 13, 120);
    let mut raw = Vec::new();
    raw.extend_from_slice(&y);
    raw.extend_from_slice(&u);
    raw.extend_from_slice(&v);
    let theirs = ffmpeg_convert(
        &raw,
        "yuv444p",
        "rgb24",
        "scale=in_color_matrix=bt601:in_range=pc:flags=accurate_rnd",
    );
    let src = planar_frame(vec![(W, y), (W, u), (W, v)]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::YuvJ444P, W as u32, H as u32),
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let diff = max_abs_diff(&ours.planes[0].data, &theirs);
    assert!(diff <= 2, "max diff {diff}");
}

/// 4:2:0 → RGB24 with flat chroma (upsampler-neutral fixture): the
/// matrix agreement carries over to the subsampled decode path.
#[test]
fn yuv420_flat_chroma_to_rgb24_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let cw = W / 2;
    let ch = H / 2;
    let y = ramp(W * H, 16, 235, 3, 9);
    for (uc, vc) in [(128u8, 128u8), (90, 170), (200, 40)] {
        let u = vec![uc; cw * ch];
        let v = vec![vc; cw * ch];
        let mut raw = Vec::new();
        raw.extend_from_slice(&y);
        raw.extend_from_slice(&u);
        raw.extend_from_slice(&v);
        // `full_chroma_int` asks the validator for full-precision chroma
        // on subsampled input — without it, its 4:2:0 fast path rounds
        // chroma more coarsely than the 4:4:4 route and the comparison
        // would measure that policy, not our matrix.
        let theirs = ffmpeg_convert(
            &raw,
            "yuv420p",
            "rgb24",
            "scale=in_color_matrix=bt601:in_range=tv:flags=accurate_rnd+full_chroma_int+full_chroma_inp",
        );
        let src = planar_frame(vec![(W, y.clone()), (cw, u), (cw, v)]);
        let ours = convert(
            &src,
            FrameInfo::new(PixelFormat::Yuv420P, W as u32, H as u32),
            PixelFormat::Rgb24,
            &ConvertOptions::default(),
        )
        .expect("convert");
        let diff = max_abs_diff(&ours.planes[0].data, &theirs);
        assert!(diff <= 2, "chroma ({uc},{vc}): max diff {diff}");
    }
}

/// Packed 4:2:2 deinterleave has no colour math at all — the validator
/// and our converter must agree bit-exactly on YUYV → planar 4:2:2.
#[test]
fn yuyv_to_planar_bit_exact_vs_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let packed = ramp(W * H * 2, 0, 255, 9, 1);
    // Format-only change: no scale filter needed, `format` pins the
    // target layout.
    let theirs = ffmpeg_convert(&packed, "yuyv422", "yuv422p", "format=yuv422p");
    let src = planar_frame(vec![(W * 2, packed.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuyv422, W as u32, H as u32),
        PixelFormat::Yuv422P,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let cw = W / 2;
    assert_eq!(ours.planes[0].data, theirs[..W * H], "Y plane");
    assert_eq!(
        ours.planes[1].data,
        theirs[W * H..W * H + cw * H],
        "U plane"
    );
    assert_eq!(ours.planes[2].data, theirs[W * H + cw * H..], "V plane");
}
