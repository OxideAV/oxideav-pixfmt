//! Palette types and quantisation strategies.
//!
//! A [`Palette`] is a flat list of RGBA colours (up to 256 entries).
//! [`generate_palette`] is the public entry point for building one from
//! a batch of source [`VideoFrame`](oxideav_core::VideoFrame)s. Three
//! strategies are supported:
//!
//! - [`PaletteStrategy::Uniform`] — fixed 3-3-2 cube, 256 entries.
//! - [`PaletteStrategy::MedianCut`] — Heckbert's 1982 median-cut
//!   scheme. Starts with one box containing every sampled colour and
//!   repeatedly splits the box with the largest range on the widest
//!   axis until `max_colors` boxes are left.
//! - [`PaletteStrategy::Octree`] — Gervautz & Purgathofer (1988) octree
//!   quantisation. Inserts every pixel into an 8-ary tree keyed by the
//!   RGB bits MSB-first; once leaf count exceeds `max_colors`, the
//!   deepest internal node whose children are all leaves is merged
//!   into a single leaf, bounding tree size throughout the walk.

use oxideav_core::{Error, PixelFormat, Result, VideoFrame};

use crate::convert::FrameInfo;

/// An indexed-colour palette.
#[derive(Clone, Debug, Default)]
pub struct Palette {
    /// RGBA colour entries. Typically ≤ 256.
    pub colors: Vec<[u8; 4]>,
}

/// Palette-generation strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaletteStrategy {
    MedianCut,
    /// Reserved for v2 — returns `Unsupported` in v1.
    Octree,
    /// 3-3-2 uniform cube (or nearest fit to `max_colors`).
    Uniform,
}

/// Options governing [`generate_palette`].
#[derive(Clone, Debug)]
pub struct PaletteGenOptions {
    pub strategy: PaletteStrategy,
    /// Maximum palette entries. A value of 0 is treated as 1.
    pub max_colors: u8,
    /// If set, the resulting palette will include an entry with the
    /// given index reserved for transparency (alpha = 0).
    pub transparency: Option<u8>,
}

impl Default for PaletteGenOptions {
    fn default() -> Self {
        Self {
            strategy: PaletteStrategy::MedianCut,
            max_colors: 255,
            transparency: None,
        }
    }
}

/// Build a palette from a batch of frames. Every frame must be
/// `Rgb24` or `Rgba` — use [`crate::convert`] to stage through one
/// of those first. Each frame is paired with its [`FrameInfo`]
/// because stream-level properties (format, width, height) are no
/// longer carried on [`VideoFrame`].
pub fn generate_palette(
    frames: &[(&VideoFrame, FrameInfo)],
    opts: &PaletteGenOptions,
) -> Result<Palette> {
    if frames.is_empty() {
        return Err(Error::invalid("generate_palette: no frames"));
    }
    let pixels = collect_pixels(frames)?;
    let max = opts.max_colors.max(1) as usize;

    let mut colors = match opts.strategy {
        PaletteStrategy::MedianCut => median_cut(&pixels, max),
        PaletteStrategy::Uniform => uniform_palette(max),
        PaletteStrategy::Octree => octree_quantise(&pixels, max),
    };

    if let Some(idx) = opts.transparency {
        let i = idx as usize;
        if i < colors.len() {
            colors[i] = [0, 0, 0, 0];
        } else {
            colors.push([0, 0, 0, 0]);
        }
    }

    Ok(Palette { colors })
}

/// Gather tightly packed (R, G, B, A) pixels from each frame, dropping
/// stride padding.
fn collect_pixels(frames: &[(&VideoFrame, FrameInfo)]) -> Result<Vec<[u8; 4]>> {
    let mut out = Vec::new();
    for (frame, info) in frames {
        let w = info.width as usize;
        let h = info.height as usize;
        match info.format {
            PixelFormat::Rgb24 => {
                let plane = &frame.planes[0];
                for row in 0..h {
                    let off = row * plane.stride;
                    for col in 0..w {
                        out.push([
                            plane.data[off + col * 3],
                            plane.data[off + col * 3 + 1],
                            plane.data[off + col * 3 + 2],
                            255,
                        ]);
                    }
                }
            }
            PixelFormat::Rgba => {
                let plane = &frame.planes[0];
                for row in 0..h {
                    let off = row * plane.stride;
                    for col in 0..w {
                        out.push([
                            plane.data[off + col * 4],
                            plane.data[off + col * 4 + 1],
                            plane.data[off + col * 4 + 2],
                            plane.data[off + col * 4 + 3],
                        ]);
                    }
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "generate_palette: frames must be Rgb24 or Rgba, got {other:?}"
                )));
            }
        }
    }
    Ok(out)
}

/// Heckbert median-cut quantisation on an opaque RGB triple set (alpha
/// is preserved from the first sampled pixel in each leaf box).
fn median_cut(pixels: &[[u8; 4]], max_colors: usize) -> Vec<[u8; 4]> {
    if pixels.is_empty() {
        return Vec::new();
    }

    // Start with one box holding every sampled colour.
    let mut boxes: Vec<Box3> = vec![Box3::from(pixels)];
    while boxes.len() < max_colors {
        // Pick the box with the largest single-axis range.
        let idx = match boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.colors.len() > 1)
            .max_by_key(|(_, b)| b.max_range())
        {
            Some((i, _)) => i,
            None => break, // every remaining box is already a single colour
        };
        let taken = boxes.swap_remove(idx);
        let (a, b) = taken.split();
        boxes.push(a);
        boxes.push(b);
    }

    // Output each box as the average of its colours.
    boxes.iter().map(|b| b.average()).collect()
}

struct Box3 {
    colors: Vec<[u8; 4]>,
}

impl Box3 {
    fn from(p: &[[u8; 4]]) -> Self {
        Self { colors: p.to_vec() }
    }

    fn max_range(&self) -> i32 {
        let (rmin, rmax) = self.range(0);
        let (gmin, gmax) = self.range(1);
        let (bmin, bmax) = self.range(2);
        (rmax - rmin).max(gmax - gmin).max(bmax - bmin)
    }

    fn range(&self, c: usize) -> (i32, i32) {
        let mut lo = 255i32;
        let mut hi = 0i32;
        for p in &self.colors {
            let v = p[c] as i32;
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        (lo, hi)
    }

    fn widest_axis(&self) -> usize {
        let mut best = 0usize;
        let mut best_range = -1i32;
        for c in 0..3 {
            let (lo, hi) = self.range(c);
            if hi - lo > best_range {
                best_range = hi - lo;
                best = c;
            }
        }
        best
    }

    fn split(self) -> (Self, Self) {
        let axis = self.widest_axis();
        let mut colors = self.colors;
        colors.sort_unstable_by_key(|p| p[axis]);
        let mid = colors.len() / 2;
        let b = colors.split_off(mid);
        (Self { colors }, Self { colors: b })
    }

    fn average(&self) -> [u8; 4] {
        if self.colors.is_empty() {
            return [0, 0, 0, 255];
        }
        let mut sum = [0u64; 4];
        for p in &self.colors {
            for c in 0..4 {
                sum[c] += p[c] as u64;
            }
        }
        let n = self.colors.len() as u64;
        [
            (sum[0] / n) as u8,
            (sum[1] / n) as u8,
            (sum[2] / n) as u8,
            (sum[3] / n) as u8,
        ]
    }
}

// -------------------------------------------------------------------------
// Octree quantisation (Gervautz & Purgathofer, 1988).
//
// The tree is an arena-backed 8-ary structure keyed by the RGB bits
// MSB-first. Inserting a pixel walks the tree, creating nodes as
// needed, until a leaf at depth 8 accumulates the sample. When leaf
// count exceeds `max_colors`, `reduce_step` picks the deepest internal
// node whose children are all leaves and merges its children's sums
// into itself, converting it into a leaf. That keeps the tree bounded
// throughout the walk. Output colours are the averaged sums at each
// surviving leaf.

const OCT_MAX_DEPTH: u8 = 8;

#[derive(Clone)]
struct OctNode {
    children: [i32; 8],
    r_sum: u64,
    g_sum: u64,
    b_sum: u64,
    pix_count: u64,
    is_leaf: bool,
    depth: u8,
}

impl OctNode {
    fn new(depth: u8) -> Self {
        Self {
            children: [-1; 8],
            r_sum: 0,
            g_sum: 0,
            b_sum: 0,
            pix_count: 0,
            is_leaf: depth == OCT_MAX_DEPTH,
            depth,
        }
    }
}

struct Octree {
    nodes: Vec<OctNode>,
    leaf_count: usize,
}

impl Octree {
    fn new() -> Self {
        Self {
            nodes: vec![OctNode::new(0)],
            leaf_count: 0,
        }
    }

    fn child_index(r: u8, g: u8, b: u8, depth: u8) -> usize {
        // depth in 0..OCT_MAX_DEPTH; bit 7 at depth 0, bit 0 at depth 7.
        let bit = 7 - depth;
        let rb = ((r >> bit) & 1) as usize;
        let gb = ((g >> bit) & 1) as usize;
        let bb = ((b >> bit) & 1) as usize;
        (rb << 2) | (gb << 1) | bb
    }

    fn insert(&mut self, r: u8, g: u8, b: u8) {
        let mut idx = 0usize;
        loop {
            if self.nodes[idx].is_leaf {
                self.nodes[idx].r_sum += r as u64;
                self.nodes[idx].g_sum += g as u64;
                self.nodes[idx].b_sum += b as u64;
                self.nodes[idx].pix_count += 1;
                return;
            }
            let depth = self.nodes[idx].depth;
            let ci = Self::child_index(r, g, b, depth);
            let child = self.nodes[idx].children[ci];
            if child < 0 {
                let new_depth = depth + 1;
                let new_idx = self.nodes.len() as i32;
                self.nodes.push(OctNode::new(new_depth));
                self.nodes[idx].children[ci] = new_idx;
                if new_depth == OCT_MAX_DEPTH {
                    self.leaf_count += 1;
                }
                idx = new_idx as usize;
            } else {
                idx = child as usize;
            }
        }
    }

    /// Pick the deepest internal node whose children are all leaves, and
    /// merge those children's sums into it, turning it into a leaf.
    /// Returns `false` when no such node exists (tree already minimal).
    fn reduce_step(&mut self) -> bool {
        let mut best: Option<(usize, u8)> = None;
        for (i, n) in self.nodes.iter().enumerate() {
            if n.is_leaf {
                continue;
            }
            let mut has_any = false;
            let mut all_leaves = true;
            for &c in &n.children {
                if c >= 0 {
                    has_any = true;
                    if !self.nodes[c as usize].is_leaf {
                        all_leaves = false;
                        break;
                    }
                }
            }
            if !has_any || !all_leaves {
                continue;
            }
            match best {
                None => best = Some((i, n.depth)),
                Some((_, bd)) if n.depth > bd => best = Some((i, n.depth)),
                _ => {}
            }
        }
        let idx = match best {
            Some((i, _)) => i,
            None => return false,
        };

        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;
        let mut pix = 0u64;
        let mut merged = 0usize;
        let children = self.nodes[idx].children;
        for &c in &children {
            if c >= 0 {
                let ch = &self.nodes[c as usize];
                r_sum += ch.r_sum;
                g_sum += ch.g_sum;
                b_sum += ch.b_sum;
                pix += ch.pix_count;
                merged += 1;
            }
        }
        self.nodes[idx].r_sum = r_sum;
        self.nodes[idx].g_sum = g_sum;
        self.nodes[idx].b_sum = b_sum;
        self.nodes[idx].pix_count = pix;
        self.nodes[idx].is_leaf = true;
        self.nodes[idx].children = [-1; 8];
        // +1 for the newly promoted leaf, -merged for the absorbed ones.
        // A single-child merge leaves the count unchanged but still
        // shrinks the tree, so the outer while loop eventually finds a
        // multi-child reducible.
        self.leaf_count = self.leaf_count + 1 - merged;
        true
    }

    fn colors(&self) -> Vec<[u8; 4]> {
        // DFS from root so orphaned subtrees (children whose parent was
        // reduced into a leaf, but which still live in the arena) don't
        // get emitted as colours.
        let mut out = Vec::with_capacity(self.leaf_count);
        let mut stack = vec![0usize];
        while let Some(idx) = stack.pop() {
            let n = &self.nodes[idx];
            if n.is_leaf {
                if n.pix_count > 0 {
                    let c = n.pix_count;
                    out.push([
                        (n.r_sum / c) as u8,
                        (n.g_sum / c) as u8,
                        (n.b_sum / c) as u8,
                        255,
                    ]);
                }
                continue;
            }
            for &c in &n.children {
                if c >= 0 {
                    stack.push(c as usize);
                }
            }
        }
        out
    }
}

fn octree_quantise(pixels: &[[u8; 4]], max_colors: usize) -> Vec<[u8; 4]> {
    if pixels.is_empty() {
        return Vec::new();
    }
    let max = max_colors.clamp(1, 256);
    let mut tree = Octree::new();
    for &[r, g, b, _a] in pixels {
        tree.insert(r, g, b);
        while tree.leaf_count > max {
            if !tree.reduce_step() {
                break;
            }
        }
    }
    tree.colors()
}

/// Uniform 3-3-2 RGB cube (or truncated to `max` entries).
fn uniform_palette(max: usize) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(256);
    for r in 0..8u8 {
        for g in 0..8u8 {
            for b in 0..4u8 {
                // Spread the 3 or 2 bits evenly over 0..=255.
                let rr = (r as u32 * 255 / 7) as u8;
                let gg = (g as u32 * 255 / 7) as u8;
                let bb = (b as u32 * 255 / 3) as u8;
                out.push([rr, gg, bb, 255]);
                if out.len() >= max {
                    return out;
                }
            }
        }
    }
    out
}
