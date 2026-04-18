//! Runtime SIMD dispatch for the YUV ↔ RGB inner loops.
//!
//! The public YUV converters in [`crate::yuv`] thunk here; this module
//! picks the best implementation at first call (cached) and falls back
//! to the scalar fixed-point path on CPUs that lack vector support or
//! when the frame is too narrow to vectorise.

use crate::yuv::{self, YuvMatrix};
use core::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Path {
    Unknown = 0,
    Scalar = 1,
    Avx2 = 2,
    Neon = 3,
    Portable = 4,
}

static DISPATCH: AtomicU8 = AtomicU8::new(Path::Unknown as u8);

fn select_path() -> Path {
    if std::env::var_os("OXIDEAV_PIXFMT_FORCE_SCALAR").is_some() {
        return Path::Scalar;
    }
    #[cfg(feature = "nightly")]
    {
        if std::env::var_os("OXIDEAV_PIXFMT_FORCE_PORTABLE_SIMD").is_some() {
            return Path::Portable;
        }
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return Path::Avx2;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Path::Neon;
        }
    }
    Path::Scalar
}

fn path() -> Path {
    let v = DISPATCH.load(Ordering::Relaxed);
    if v != Path::Unknown as u8 {
        return match v {
            x if x == Path::Avx2 as u8 => Path::Avx2,
            x if x == Path::Neon as u8 => Path::Neon,
            x if x == Path::Portable as u8 => Path::Portable,
            _ => Path::Scalar,
        };
    }
    let p = select_path();
    DISPATCH.store(p as u8, Ordering::Relaxed);
    p
}

// ---------------------------------------------------------------------
// Public entrypoints.

pub(crate) fn yuv420_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2::yuv420_to_rgb24(yp, up, vp, dst, w, h, matrix) },
        #[cfg(target_arch = "aarch64")]
        Path::Neon => unsafe { neon::yuv420_to_rgb24(yp, up, vp, dst, w, h, matrix) },
        #[cfg(feature = "nightly")]
        Path::Portable => portable::yuv420_to_rgb24(yp, up, vp, dst, w, h, matrix),
        _ => yuv::yuv420_to_rgb24_scalar(yp, up, vp, dst, w, h, matrix),
    }
}

pub(crate) fn yuv422_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2::yuv422_to_rgb24(yp, up, vp, dst, w, h, matrix) },
        #[cfg(target_arch = "aarch64")]
        Path::Neon => unsafe { neon::yuv422_to_rgb24(yp, up, vp, dst, w, h, matrix) },
        _ => yuv::yuv422_to_rgb24_scalar(yp, up, vp, dst, w, h, matrix),
    }
}

pub(crate) fn yuv444_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2::yuv444_to_rgb24(yp, up, vp, dst, w, h, matrix) },
        #[cfg(target_arch = "aarch64")]
        Path::Neon => unsafe { neon::yuv444_to_rgb24(yp, up, vp, dst, w, h, matrix) },
        _ => yuv::yuv444_to_rgb24_scalar(yp, up, vp, dst, w, h, matrix),
    }
}

pub(crate) fn rgb24_to_yuv420(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2::rgb24_to_yuv420(src, yp, up, vp, w, h, matrix) },
        _ => yuv::rgb24_to_yuv420_scalar(src, yp, up, vp, w, h, matrix),
    }
}

pub(crate) fn rgb24_to_yuv422(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2::rgb24_to_yuv422(src, yp, up, vp, w, h, matrix) },
        _ => yuv::rgb24_to_yuv422_scalar(src, yp, up, vp, w, h, matrix),
    }
}

pub(crate) fn rgb24_to_yuv444(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    match path() {
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { avx2::rgb24_to_yuv444(src, yp, up, vp, w, h, matrix) },
        _ => yuv::rgb24_to_yuv444_scalar(src, yp, up, vp, w, h, matrix),
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(feature = "nightly")]
pub(crate) mod portable;
