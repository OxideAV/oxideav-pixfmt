//! Tiny shared runtime dispatch: "can we use AVX2?".
//!
//! A single cached atomic answers whether the host supports AVX2 and
//! whether the `OXIDEAV_PIXFMT_FORCE_SCALAR` escape hatch is active.
//! The various SIMD sites (RGB swizzles, YUV encode/decode, NV12
//! deinterleave, chroma upsample, …) share this so there's one
//! source-of-truth for feature detection and env-var override.

use core::sync::atomic::{AtomicU8, Ordering};

const UNKNOWN: u8 = 0;
const NO: u8 = 1;
const YES: u8 = 2;

static AVX2: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Returns `true` when the host supports AVX2 and the scalar override
/// env var is *not* set.
#[inline]
pub(crate) fn has_avx2() -> bool {
    let v = AVX2.load(Ordering::Relaxed);
    if v != UNKNOWN {
        return v == YES;
    }
    let ok = detect();
    AVX2.store(if ok { YES } else { NO }, Ordering::Relaxed);
    ok
}

fn detect() -> bool {
    if std::env::var_os("OXIDEAV_PIXFMT_FORCE_SCALAR").is_some() {
        return false;
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        return std::is_x86_feature_detected!("avx2");
    }
    #[allow(unreachable_code)]
    false
}
