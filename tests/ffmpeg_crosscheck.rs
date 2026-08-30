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

/// Limited-range 4:4:0 (full-width, half-height chroma) → RGB24 with
/// flat chroma planes — the same design as the 4:2:0 check: a constant
/// chroma plane is reproduced exactly by any vertical upsampler, so the
/// comparison isolates the matrix and the 4:4:0 plane geometry (a
/// wrong chroma stride or row pairing would scramble the colours).
#[test]
fn yuv440_flat_chroma_to_rgb24_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let ch = H / 2;
    let y = ramp(W * H, 16, 235, 3, 9);
    for (uc, vc) in [(128u8, 128u8), (90, 170), (200, 40)] {
        let u = vec![uc; W * ch];
        let v = vec![vc; W * ch];
        let mut raw = Vec::new();
        raw.extend_from_slice(&y);
        raw.extend_from_slice(&u);
        raw.extend_from_slice(&v);
        let theirs = ffmpeg_convert(
            &raw,
            "yuv440p",
            "rgb24",
            "scale=in_color_matrix=bt601:in_range=tv:flags=accurate_rnd+full_chroma_int+full_chroma_inp",
        );
        let src = planar_frame(vec![(W, y.clone()), (W, u), (W, v)]);
        let ours = convert(
            &src,
            FrameInfo::new(PixelFormat::Yuv440P, W as u32, H as u32),
            PixelFormat::Rgb24,
            &ConvertOptions::default(),
        )
        .expect("convert");
        let diff = max_abs_diff(&ours.planes[0].data, &theirs);
        assert!(diff <= 2, "chroma ({uc},{vc}): max diff {diff}");
    }
}

/// 4:4:4 → 4:4:0 with flat chroma planes: any vertical downsampling
/// filter reproduces a constant plane exactly, so ours and the
/// validator's output must agree bit-for-bit on every plane — which
/// pins the 4:4:0 output geometry (full-width, half-height chroma,
/// luma copied verbatim) independently of resampling policy.
#[test]
fn yuv444_to_yuv440_flat_chroma_bit_exact_vs_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let y = ramp(W * H, 16, 235, 7, 3);
    for (uc, vc) in [(128u8, 128u8), (90, 170)] {
        let u = vec![uc; W * H];
        let v = vec![vc; W * H];
        let mut raw = Vec::new();
        raw.extend_from_slice(&y);
        raw.extend_from_slice(&u);
        raw.extend_from_slice(&v);
        let theirs = ffmpeg_convert(&raw, "yuv444p", "yuv440p", "null");
        assert_eq!(theirs.len(), W * H + 2 * W * (H / 2));
        let src = planar_frame(vec![(W, y.clone()), (W, u), (W, v)]);
        let ours = convert(
            &src,
            FrameInfo::new(PixelFormat::Yuv444P, W as u32, H as u32),
            PixelFormat::Yuv440P,
            &ConvertOptions::default(),
        )
        .expect("convert");
        assert_eq!(ours.planes[0].data, theirs[..W * H]);
        assert_eq!(ours.planes[1].data, theirs[W * H..W * H + W * (H / 2)]);
        assert_eq!(ours.planes[2].data, theirs[W * H + W * (H / 2)..]);
    }
}

/// Integer → float normalisation against the validator: a 16-bit gray
/// ramp becomes `grayf32le` with every sample equal to `code / 65535`,
/// and both implementations agree to within a hundredth of a 16-bit
/// code (bit-exact in practice). This pins the "no transfer function
/// on the hop" rule as a shared convention rather than a private
/// choice — a gamma curve would differ by orders of magnitude more.
/// (The validator's planar-GBR float route is a few 16-bit codes off a
/// plain normalisation, so only the gray hop is compared exactly.)
#[test]
fn gray16_to_grayf32_normalisation_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    if !validator_supports_pix_fmt("grayf32le") {
        eprintln!("skipping: validator lacks grayf32le");
        return;
    }
    let as_f32 = |buf: &[u8]| -> Vec<f32> {
        buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let codes: Vec<u16> = (0..W * H).map(|i| ((i * 977) % 65536) as u16).collect();
    let raw: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
    let theirs = as_f32(&ffmpeg_convert(&raw, "gray16le", "grayf32le", "null"));
    let src = planar_frame(vec![(W * 2, raw.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Gray16Le, W as u32, H as u32),
        PixelFormat::GrayF32Le,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let ours = as_f32(&ours.planes[0].data);
    assert_eq!(ours.len(), theirs.len());
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert!(
            (a - b).abs() <= 1.0 / 65535.0 * 0.01,
            "gray {i}: {a} vs {b}"
        );
    }
}

/// Limited-range YUVA 4:4:4 → RGBA: the colour channels agree with the
/// validator within ±2 (pure matrix, no chroma resampling at 4:4:4) and
/// the full-resolution alpha plane is carried bit-exactly by both
/// implementations.
#[test]
fn yuva444_to_rgba_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let y = ramp(W * H, 16, 235, 7, 3);
    let u = ramp(W * H, 16, 240, 11, 40);
    let v = ramp(W * H, 16, 240, 13, 80);
    let a = ramp(W * H, 0, 255, 5, 17);
    let mut raw = Vec::new();
    raw.extend_from_slice(&y);
    raw.extend_from_slice(&u);
    raw.extend_from_slice(&v);
    raw.extend_from_slice(&a);
    let theirs = ffmpeg_convert(
        &raw,
        "yuva444p",
        "rgba",
        "scale=in_color_matrix=bt601:in_range=tv:flags=accurate_rnd",
    );
    let src = planar_frame(vec![
        (W, y.clone()),
        (W, u.clone()),
        (W, v.clone()),
        (W, a.clone()),
    ]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva444P, W as u32, H as u32),
        PixelFormat::Rgba,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let od = &ours.planes[0].data;
    for p in 0..W * H {
        for c in 0..3 {
            let diff = (od[p * 4 + c] as i32 - theirs[p * 4 + c] as i32).abs();
            assert!(diff <= 2, "pixel {p} channel {c}: diff {diff}");
        }
        assert_eq!(od[p * 4 + 3], theirs[p * 4 + 3], "alpha at pixel {p}");
        assert_eq!(od[p * 4 + 3], a[p], "alpha must equal the source plane");
    }
}

/// RGBA → limited-range YUVA 4:4:4: Y/U/V within ±2 of the validator,
/// alpha plane split out bit-exactly by both sides.
#[test]
fn rgba_to_yuva444_matches_validator() {
    if !ffmpeg_available() {
        eprintln!("skipping: no ffmpeg binary on PATH");
        return;
    }
    let rgba = ramp(W * H * 4, 0, 255, 5, 11);
    let theirs = ffmpeg_convert(
        &rgba,
        "rgba",
        "yuva444p",
        "scale=out_color_matrix=bt601:out_range=tv:flags=accurate_rnd",
    );
    let src = planar_frame(vec![(W * 4, rgba.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Rgba, W as u32, H as u32),
        PixelFormat::Yuva444P,
        &ConvertOptions::default(),
    )
    .expect("convert");
    for (p, name) in [(0usize, "Y"), (1, "U"), (2, "V")] {
        let their_plane = &theirs[p * W * H..(p + 1) * W * H];
        let diff = max_abs_diff(&ours.planes[p].data, their_plane);
        assert!(diff <= 2, "{name} plane: max diff {diff}");
    }
    // Alpha plane: ours is a verbatim copy of the source's 4th bytes —
    // exact by construction. The validator routes alpha through a
    // deeper intermediate with its own rounding (observed +1 on codes
    // ≥ 128), so the cross-comparison gets a ±1 tolerance while the
    // source comparison stays bit-exact.
    let their_alpha = &theirs[3 * W * H..4 * W * H];
    let diff = max_abs_diff(&ours.planes[3].data, their_alpha);
    assert!(diff <= 1, "alpha plane vs validator: max diff {diff}");
    for p in 0..W * H {
        assert_eq!(ours.planes[3].data[p], rgba[p * 4 + 3]);
    }
}

/// True when the validator binary lists `name` among its raw pixel
/// formats (still a black-box probe: only the documented `-pix_fmts`
/// listing is consulted).
fn validator_supports_pix_fmt(name: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-pix_fmts"])
        .output()
        .map(|o| {
            o.status.success()
                && String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .any(|l| l.split_whitespace().nth(1) == Some(name))
        })
        .unwrap_or(false)
}

/// Widen an 8-bit code to a 10-bit LE word with MSB replication — the
/// closest 10-bit code to the ideal rescale, so both implementations'
/// narrowing policies (truncation here, rounding there) land back on
/// the same 8-bit value and the comparison stays about the deep-plane
/// plumbing and the matrix, not about narrowing policy.
fn widen10(v: u8) -> u16 {
    ((v as u16) << 2) | ((v as u16) >> 6)
}

fn le16_plane_from8(codes: &[u8]) -> Vec<u8> {
    codes
        .iter()
        .flat_map(|&v| widen10(v).to_le_bytes())
        .collect()
}

/// Deep YUVA 4:4:4 (10-bit) → RGBA: colour channels within ±2 of the
/// validator, alpha within ±1 (its alpha narrowing rounds where ours
/// truncates — the fixture's widened codes make both land on the same
/// 8-bit value, verified exactly against the source model).
#[test]
fn yuva444p10_to_rgba_matches_validator() {
    if !ffmpeg_available() || !validator_supports_pix_fmt("yuva444p10le") {
        eprintln!("skipping: no ffmpeg binary / no yuva444p10le support");
        return;
    }
    let y8 = ramp(W * H, 16, 235, 7, 3);
    let u8v = ramp(W * H, 16, 240, 11, 40);
    let v8 = ramp(W * H, 16, 240, 13, 80);
    let a8 = ramp(W * H, 0, 255, 5, 17);
    let (y, u, v, a) = (
        le16_plane_from8(&y8),
        le16_plane_from8(&u8v),
        le16_plane_from8(&v8),
        le16_plane_from8(&a8),
    );
    let mut raw = Vec::new();
    raw.extend_from_slice(&y);
    raw.extend_from_slice(&u);
    raw.extend_from_slice(&v);
    raw.extend_from_slice(&a);
    let theirs = ffmpeg_convert(
        &raw,
        "yuva444p10le",
        "rgba",
        "scale=in_color_matrix=bt601:in_range=tv:flags=accurate_rnd",
    );
    let src = planar_frame(vec![(W * 2, y), (W * 2, u), (W * 2, v), (W * 2, a)]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva444P10Le, W as u32, H as u32),
        PixelFormat::Rgba,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let od = &ours.planes[0].data;
    for p in 0..W * H {
        for c in 0..3 {
            let diff = (od[p * 4 + c] as i32 - theirs[p * 4 + c] as i32).abs();
            assert!(diff <= 2, "pixel {p} channel {c}: diff {diff}");
        }
        let adiff = (od[p * 4 + 3] as i32 - theirs[p * 4 + 3] as i32).abs();
        assert!(adiff <= 1, "alpha at pixel {p}: diff {adiff}");
        assert_eq!(od[p * 4 + 3], a8[p], "alpha must round-trip the 8-bit code");
    }
}

/// Alpha drop on the deep family is pure plumbing (no colour math): our
/// `Yuva444P10Le → Yuv444P10Le` must agree bit-exactly with the
/// validator's format-only conversion, word for LE word.
#[test]
fn deep_yuva_alpha_drop_bit_exact_vs_validator() {
    if !ffmpeg_available()
        || !validator_supports_pix_fmt("yuva444p10le")
        || !validator_supports_pix_fmt("yuv444p10le")
    {
        eprintln!("skipping: no ffmpeg binary / no deep yuva support");
        return;
    }
    let y = le16_plane_from8(&ramp(W * H, 16, 235, 7, 3));
    let u = le16_plane_from8(&ramp(W * H, 16, 240, 11, 40));
    let v = le16_plane_from8(&ramp(W * H, 16, 240, 13, 80));
    let a = le16_plane_from8(&ramp(W * H, 0, 255, 5, 17));
    let mut raw = Vec::new();
    raw.extend_from_slice(&y);
    raw.extend_from_slice(&u);
    raw.extend_from_slice(&v);
    raw.extend_from_slice(&a);
    let theirs = ffmpeg_convert(&raw, "yuva444p10le", "yuv444p10le", "format=yuv444p10le");
    let src = planar_frame(vec![
        (W * 2, y.clone()),
        (W * 2, u.clone()),
        (W * 2, v.clone()),
        (W * 2, a),
    ]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva444P10Le, W as u32, H as u32),
        PixelFormat::Yuv444P10Le,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let n = W * H * 2;
    assert_eq!(ours.planes.len(), 3);
    assert_eq!(ours.planes[0].data, theirs[..n], "Y plane");
    assert_eq!(ours.planes[1].data, theirs[n..2 * n], "U plane");
    assert_eq!(ours.planes[2].data, theirs[2 * n..], "V plane");
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

/// `Gbrp8` ↔ `Rgb24` is a zero-math plane reorder on both sides: the
/// validator's format-only conversion must agree bit-exactly in both
/// directions (G, B, R plane order per the format's definition).
#[test]
fn gbrp8_rgb24_bit_exact_vs_validator() {
    if !ffmpeg_available() || !validator_supports_pix_fmt("gbrp") {
        eprintln!("skipping: no ffmpeg binary / no gbrp support");
        return;
    }
    // Planar G, B, R input.
    let g = ramp(W * H, 0, 255, 7, 0);
    let b = ramp(W * H, 0, 255, 11, 3);
    let r = ramp(W * H, 0, 255, 5, 1);
    let mut raw = Vec::new();
    raw.extend_from_slice(&g);
    raw.extend_from_slice(&b);
    raw.extend_from_slice(&r);
    let theirs = ffmpeg_convert(&raw, "gbrp", "rgb24", "format=rgb24");
    let src = planar_frame(vec![(W, g.clone()), (W, b.clone()), (W, r.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrp8, W as u32, H as u32),
        PixelFormat::Rgb24,
        &ConvertOptions::default(),
    )
    .expect("convert");
    assert_eq!(ours.planes[0].data, theirs, "gbrp → rgb24");

    // And the packed → planar direction.
    let rgb = ramp(W * H * 3, 0, 255, 9, 2);
    let theirs = ffmpeg_convert(&rgb, "rgb24", "gbrp", "format=gbrp");
    let src = planar_frame(vec![(W * 3, rgb.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Rgb24, W as u32, H as u32),
        PixelFormat::Gbrp8,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let n = W * H;
    assert_eq!(ours.planes[0].data, theirs[..n], "G plane");
    assert_eq!(ours.planes[1].data, theirs[n..2 * n], "B plane");
    assert_eq!(ours.planes[2].data, theirs[2 * n..], "R plane");
}

/// `Gbrp16Le` → `Rgb48Le` degenerates to a pure plane reorder (16
/// significant bits, shift by zero) — bit-exact against the validator's
/// format-only conversion, word for LE word.
#[test]
fn gbrp16_to_rgb48_bit_exact_vs_validator() {
    if !ffmpeg_available()
        || !validator_supports_pix_fmt("gbrp16le")
        || !validator_supports_pix_fmt("rgb48le")
    {
        eprintln!("skipping: no ffmpeg binary / no gbrp16le support");
        return;
    }
    // Full-width 16-bit words: any byte pattern is a legal sample.
    let g: Vec<u8> = ramp(W * H * 2, 0, 255, 7, 0);
    let b: Vec<u8> = ramp(W * H * 2, 0, 255, 11, 3);
    let r: Vec<u8> = ramp(W * H * 2, 0, 255, 5, 1);
    let mut raw = Vec::new();
    raw.extend_from_slice(&g);
    raw.extend_from_slice(&b);
    raw.extend_from_slice(&r);
    let theirs = ffmpeg_convert(&raw, "gbrp16le", "rgb48le", "format=rgb48le");
    let src = planar_frame(vec![(W * 2, g), (W * 2, b), (W * 2, r)]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrp16Le, W as u32, H as u32),
        PixelFormat::Rgb48Le,
        &ConvertOptions::default(),
    )
    .expect("convert");
    assert_eq!(ours.planes[0].data, theirs);
}

/// Alpha drop at the deep 4:2:0 siting is pure plumbing: our
/// `Yuva420P10Le → Yuv420P10Le` must agree bit-exactly with the
/// validator's format-only conversion on all three surviving planes.
#[test]
fn deep_420_yuva_alpha_drop_bit_exact_vs_validator() {
    if !ffmpeg_available()
        || !validator_supports_pix_fmt("yuva420p10le")
        || !validator_supports_pix_fmt("yuv420p10le")
    {
        eprintln!("skipping: no ffmpeg binary / no deep 4:2:0 yuva support");
        return;
    }
    let cw = W / 2;
    let ch = H / 2;
    let y = le16_plane_from8(&ramp(W * H, 16, 235, 7, 3));
    let u = le16_plane_from8(&ramp(cw * ch, 16, 240, 11, 40));
    let v = le16_plane_from8(&ramp(cw * ch, 16, 240, 13, 80));
    let a = le16_plane_from8(&ramp(W * H, 0, 255, 5, 17));
    let mut raw = Vec::new();
    raw.extend_from_slice(&y);
    raw.extend_from_slice(&u);
    raw.extend_from_slice(&v);
    raw.extend_from_slice(&a);
    let theirs = ffmpeg_convert(&raw, "yuva420p10le", "yuv420p10le", "format=yuv420p10le");
    let src = planar_frame(vec![
        (W * 2, y.clone()),
        (cw * 2, u.clone()),
        (cw * 2, v.clone()),
        (W * 2, a),
    ]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Yuva420P10Le, W as u32, H as u32),
        PixelFormat::Yuv420P10Le,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let ny = W * H * 2;
    let nc = cw * ch * 2;
    assert_eq!(ours.planes.len(), 3);
    assert_eq!(ours.planes[0].data, theirs[..ny], "Y plane");
    assert_eq!(ours.planes[1].data, theirs[ny..ny + nc], "U plane");
    assert_eq!(ours.planes[2].data, theirs[ny + nc..ny + 2 * nc], "V plane");
}

/// `Gbrap16Le` → `Rgba64Le` is the four-plane pure reorder — bit-exact
/// against the validator, alpha word included.
#[test]
fn gbrap16_to_rgba64_bit_exact_vs_validator() {
    if !ffmpeg_available()
        || !validator_supports_pix_fmt("gbrap16le")
        || !validator_supports_pix_fmt("rgba64le")
    {
        eprintln!("skipping: no ffmpeg binary / no gbrap16le support");
        return;
    }
    let g: Vec<u8> = ramp(W * H * 2, 0, 255, 7, 0);
    let b: Vec<u8> = ramp(W * H * 2, 0, 255, 11, 3);
    let r: Vec<u8> = ramp(W * H * 2, 0, 255, 5, 1);
    let a: Vec<u8> = ramp(W * H * 2, 0, 255, 13, 9);
    let mut raw = Vec::new();
    raw.extend_from_slice(&g);
    raw.extend_from_slice(&b);
    raw.extend_from_slice(&r);
    raw.extend_from_slice(&a);
    let theirs = ffmpeg_convert(&raw, "gbrap16le", "rgba64le", "format=rgba64le");
    let src = planar_frame(vec![(W * 2, g), (W * 2, b), (W * 2, r), (W * 2, a)]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrap16Le, W as u32, H as u32),
        PixelFormat::Rgba64Le,
        &ConvertOptions::default(),
    )
    .expect("convert");
    assert_eq!(ours.planes[0].data, theirs);
}

/// `Gbrap8` ↔ `Rgba` is the byte-tier four-plane pure reorder —
/// bit-exact against the validator in both directions, alpha included.
#[test]
fn gbrap8_rgba_bit_exact_vs_validator() {
    if !ffmpeg_available() || !validator_supports_pix_fmt("gbrap") {
        eprintln!("skipping: no ffmpeg binary / no gbrap support");
        return;
    }
    let g = ramp(W * H, 0, 255, 7, 0);
    let b = ramp(W * H, 0, 255, 11, 3);
    let r = ramp(W * H, 0, 255, 5, 1);
    let a = ramp(W * H, 0, 255, 13, 5);
    let mut raw = Vec::new();
    raw.extend_from_slice(&g);
    raw.extend_from_slice(&b);
    raw.extend_from_slice(&r);
    raw.extend_from_slice(&a);
    let theirs = ffmpeg_convert(&raw, "gbrap", "rgba", "format=rgba");
    let src = planar_frame(vec![
        (W, g.clone()),
        (W, b.clone()),
        (W, r.clone()),
        (W, a.clone()),
    ]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Gbrap8, W as u32, H as u32),
        PixelFormat::Rgba,
        &ConvertOptions::default(),
    )
    .expect("convert");
    assert_eq!(ours.planes[0].data, theirs, "gbrap → rgba");

    // Packed → planar direction.
    let rgba = ramp(W * H * 4, 0, 255, 9, 2);
    let theirs = ffmpeg_convert(&rgba, "rgba", "gbrap", "format=gbrap");
    let src = planar_frame(vec![(W * 4, rgba.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Rgba, W as u32, H as u32),
        PixelFormat::Gbrap8,
        &ConvertOptions::default(),
    )
    .expect("convert");
    let n = W * H;
    assert_eq!(ours.planes[0].data, theirs[..n], "G plane");
    assert_eq!(ours.planes[1].data, theirs[n..2 * n], "B plane");
    assert_eq!(ours.planes[2].data, theirs[2 * n..3 * n], "R plane");
    assert_eq!(ours.planes[3].data, theirs[3 * n..], "A plane");
}

/// `Ya16Le` → `Gray16Le` is pure plumbing (luma word verbatim, alpha
/// dropped) — bit-exact against the validator.
#[test]
fn ya16le_to_gray16le_bit_exact_vs_validator() {
    if !ffmpeg_available()
        || !validator_supports_pix_fmt("ya16le")
        || !validator_supports_pix_fmt("gray16le")
    {
        eprintln!("skipping: no ffmpeg binary / no ya16le support");
        return;
    }
    // Interleaved LE16 (Y, A) word pairs — any byte pattern is legal.
    let raw = ramp(W * H * 4, 0, 255, 7, 3);
    let theirs = ffmpeg_convert(&raw, "ya16le", "gray16le", "format=gray16le");
    let src = planar_frame(vec![(W * 4, raw.clone())]);
    let ours = convert(
        &src,
        FrameInfo::new(PixelFormat::Ya16Le, W as u32, H as u32),
        PixelFormat::Gray16Le,
        &ConvertOptions::default(),
    )
    .expect("convert");
    assert_eq!(ours.planes[0].data, theirs, "ya16le → gray16le");
}
