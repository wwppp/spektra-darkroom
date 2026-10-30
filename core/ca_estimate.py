"""ca_estimate.py — automatic lateral chromatic-aberration estimation.

Why this exists
---------------
Lateral CA is corrected by a per-channel radial rescale (``ui/canvas_viewport.py`` and
``core/lens_warp.py``). Until now the coefficients could only come from an Adobe LCP
profile — and Sony's LCP files carry the distortion and vignette models but *no*
``ChromaticRedGreenModel`` / ``ChromaticBlueGreenModel`` at all. On every real test frame
(DSC02654 / _DSC3921 / DSC02662 / DSC02642, Sony + Tamron) the parsed coefficients are
exactly (0, 0, 0), so with a profile selected the CA stage was still a no-op.

This module measures the channel misregistration from the image itself and emits the same
three-term polynomial the shader consumes.

How it measures
---------------
For one channel pair (R against G, and B against G) it looks for the scale ``K`` that makes
the two channels agree after warping:

    K(r) = 1 + c0 * r^2 + c1 * r^4        (r = corner-normalised radius, same space as
                                           ``lens_remap_maps``)

The cost is *symmetric* — R is sampled at ``p * K^(1/2)`` and G at ``p * K^(-1/2)``. A
one-sided cost (warp only R onto G) has a systematic bias: shrinking R always smooths it,
and smoothing lowers the residual a little, so the minimum drifts toward a blur. Warping
both ways by half cancels that to first order.

Only edge pixels are used, selected by G's gradient, then their weights are equalised
across radius bins so a crowded corner cannot outvote the mid-field in the polynomial fit.
Highlight-clipped pixels are excluded (a clipped channel has no fringe left to measure).

The search is a coarse 1-D scan followed by two parabolic refinements. The r^4 term is
fitted only if it buys a real cost reduction, because c0 and c1 are strongly collinear on
a single frame and an unearned c1 just adds noise at the corners.
"""

from __future__ import annotations

import numpy as np

try:  # cv2 is a hard dependency of the pipeline; keep the import defensive anyway.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# Hard bounds, in the shader's corner-normalised space. The constant magnification shifts a
# channel by m*corner at the corner: 0.004 is ~15 px on a 24 MP frame, i.e. a broken lens.
CA_SCALE_MAX = 0.0040
# Bound for the field-dependent polynomial (r^2 basis) on top of it.
CA_C0_MAX = 0.0050
CA_C1_MAX = 0.0020
# An extra polynomial term must cut the cost by at least this fraction to be kept.
CA_TERM_MIN_GAIN = 0.92
# Below this the correction would be invisible; report "no CA" instead of a tiny number.
CA_MIN_CORNER_PX = 0.15
# The estimate must actually remove this fraction of the measured fringe energy.
CA_MIN_GAIN = 0.10

_EDGE_PERCENTILE = 97.0
_RADIUS_BINS = 8


def _to_float_gray(rgb: np.ndarray, channel: int = 1) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, channel], dtype=np.float32)


def _clip_mask(rgb: np.ndarray, radius: int = 3) -> np.ndarray:
    """Pixels (and their neighbourhood) where any channel is at the clipping ceiling.

    A clipped channel has no fringe left to measure — its value is simply the ceiling on both
    sides of the edge — so including those pixels drags the fit toward "no correction". On
    the worst test frame 31% of the candidate edge pixels sit against a blown highlight, and
    excluding them doubled the measured blue shift.
    """
    mx = rgb.max(axis=2)
    hit = mx >= np.float32(0.97)
    if not hit.any():
        return hit
    if cv2 is None:  # pragma: no cover
        return hit
    k = 2 * int(radius) + 1
    return cv2.dilate(hit.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def _select_edge_pixels(g: np.ndarray, aspect: float, clip: np.ndarray | None = None):
    """Return (xs, ys, weights, gradient magnitude) for pixels worth measuring."""
    h, w = g.shape[:2]
    if cv2 is not None:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        del gx, gy
    else:  # pragma: no cover
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
        gmag = np.sqrt(gx * gx + gy * gy)

    # Percentile on a subsample: np.percentile on 24 MP sorts 24 M floats for nothing.
    stride = max(1, int(np.sqrt(gmag.size / 400_000.0)))
    thr = float(np.percentile(gmag[::stride, ::stride], _EDGE_PERCENTILE))
    thr = max(thr, 1e-4)

    mask = gmag > thr
    margin = 8
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return xs, ys, np.zeros(0, np.float32), gmag

    if clip is not None:
        keep = ~clip[ys, xs]
        ys, xs = ys[keep], xs[keep]
        if xs.size == 0:
            return xs, ys, np.zeros(0, np.float32), gmag

    wts = gmag[ys, xs].astype(np.float32)
    # Weight by the squared gradient: strong, well-defined edges carry the most reliable
    # sub-pixel information; a faint edge mostly contributes noise and demosaic artefacts.
    wts = wts * wts
    return xs, ys, wts, gmag


def _equalise_radius_bins(r2: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """Give every radius band the same total weight.

    Without this a frame whose subject sits in one corner fits that corner and calls it a
    day: the polynomial is then extrapolated, not fitted, over the rest of the field.
    """
    bins = np.clip((r2 * _RADIUS_BINS).astype(np.int32), 0, _RADIUS_BINS - 1)
    sums = np.bincount(bins, weights=wts, minlength=_RADIUS_BINS).astype(np.float64)
    live = sums[sums > 0]
    floor = 0.25 * (float(live.mean()) if live.size else 1.0)
    scale = (float(wts.sum()) / _RADIUS_BINS) / np.maximum(sums + floor, 1e-12)
    out = (wts * scale[bins]).astype(np.float32)
    return out


class _PairCost:
    """Bilinear sampler + symmetric fringe cost for one channel against green."""

    def __init__(self, chan: np.ndarray, green: np.ndarray, xs, ys, r2, wts):
        h, w = chan.shape[:2]
        self.w, self.h = w, h
        self.a = np.ascontiguousarray(chan).reshape(-1)
        self.b = np.ascontiguousarray(green).reshape(-1)
        self.wts = wts.astype(np.float32)
        self.wsum = float(self.wts.sum()) or 1.0

        dx = (xs.astype(np.float32) + 0.5) / float(w) - 0.5
        dy = (ys.astype(np.float32) + 0.5) / float(h) - 0.5
        self.xb = (dx + 0.5) * w - 0.5
        self.yb = (dy + 0.5) * h - 0.5
        # x = xb + dx*w*(K-1); half of that on each side of the symmetric cost.
        self.hx = dx * float(w) * 0.5
        self.hy = dy * float(h) * 0.5
        self.r2 = r2.astype(np.float32)
        self.r4 = self.r2 * self.r2
        self._ix = None

    def _sample(self, flat, xs, ys):
        h, w = self.h, self.w
        np.clip(xs, 0.0, w - 1.000001, out=xs)
        np.clip(ys, 0.0, h - 1.000001, out=ys)
        x0 = xs.astype(np.int32)
        y0 = ys.astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = xs - x0
        fy = ys - y0
        r0 = y0 * w
        r1 = y1 * w
        s0 = flat[r0 + x0]
        s1 = flat[r0 + x1]
        top = s0 + (s1 - s0) * fx
        s2 = flat[r1 + x0]
        s3 = flat[r1 + x1]
        bot = s2 + (s3 - s2) * fx
        top += (bot - top) * fy
        return top

    def cost(self, m: float, c0: float, c1: float = 0.0) -> float:
        """``t`` is K-1: a constant magnification plus the field-dependent terms."""
        t = self.r2 * np.float32(c0)
        if c1:
            t += self.r4 * np.float32(c1)
        if m:
            t += np.float32(m)
        # hx/hy already carry the 1/2 of the symmetric split, so the relative displacement
        # between the two taps is exactly the full (K-1).
        xa = self.xb + self.hx * t
        ya = self.yb + self.hy * t
        xb = self.xb - self.hx * t
        yb = self.yb - self.hy * t
        va = self._sample(self.a, xa, ya)
        vb = self._sample(self.b, xb, yb)
        d = va - vb
        d *= d
        d *= self.wts
        return float(d.sum()) / self.wsum


def _refine(cost_fn, best: float, step: float, lo: float, hi: float) -> float:
    """Two parabolic refinements around the coarse minimum."""
    c = best
    for _ in range(2):
        e0 = cost_fn(c)
        e_lo = cost_fn(max(lo, c - step))
        e_hi = cost_fn(min(hi, c + step))
        denom = e_lo - 2.0 * e0 + e_hi
        if abs(denom) < 1e-15:
            break
        delta = 0.5 * (e_lo - e_hi) / denom
        delta = float(np.clip(delta, -1.0, 1.0)) * step
        if abs(delta) < 1e-9:
            break
        c = float(np.clip(c + delta, lo, hi))
        step *= 0.5
    return c


def _scan(cost_fn, lo: float, hi: float, n: int):
    grid = np.linspace(lo, hi, n)
    vals = [cost_fn(float(c)) for c in grid]
    i = int(np.argmin(vals))
    return float(grid[i]), float(vals[i]), float(grid[1] - grid[0])


def _newton2(fn, m, c0, lo_m, hi_m, lo_c, hi_c, iters=3):
    """Two-parameter Newton refinement with a numerical Hessian.

    The cost surface is a long diagonal valley: m and c0 trade off almost exactly, so a
    coordinate-wise search stalls wherever the first pass lands (it walked to m-only on the
    synthetic frame and left 55% of the shift uncorrected). The Newton step follows the
    valley instead of zig-zagging across it.
    """
    best = fn(m, c0)
    span_m = hi_m - lo_m
    span_c = hi_c - lo_c
    for _ in range(iters):
        hm = max(2e-5, abs(m) * 0.15)
        hc = max(2e-5, abs(c0) * 0.15)
        e = best
        e_mp, e_mm = fn(m + hm, c0), fn(m - hm, c0)
        e_cp, e_cm = fn(m, c0 + hc), fn(m, c0 - hc)
        g_m = (e_mp - e_mm) / (2 * hm)
        g_c = (e_cp - e_cm) / (2 * hc)
        h_mm = (e_mp - 2 * e + e_mm) / (hm * hm)
        h_cc = (e_cp - 2 * e + e_cm) / (hc * hc)
        h_mc = (fn(m + hm, c0 + hc) - fn(m + hm, c0 - hc)
                - fn(m - hm, c0 + hc) + fn(m - hm, c0 - hc)) / (4 * hm * hc)
        det = h_mm * h_cc - h_mc * h_mc
        if det <= 0 or not np.isfinite(det):
            break
        dm = -(h_cc * g_m - h_mc * g_c) / det
        dc = -(-h_mc * g_m + h_mm * g_c) / det
        # Never jump more than a fifth of the search box in one step.
        dm = float(np.clip(dm, -0.2 * span_m, 0.2 * span_m))
        dc = float(np.clip(dc, -0.2 * span_c, 0.2 * span_c))
        nm = float(np.clip(m + dm, lo_m, hi_m))
        nc = float(np.clip(c0 + dc, lo_c, hi_c))
        if nm == m and nc == c0:
            break
        cand = fn(nm, nc)
        if not (cand < best * 0.999999):
            break
        m, c0, best = nm, nc, cand
    return m, c0, best


def _estimate_channel(pair: _PairCost, max_m: float, max_c0: float, use_c1: bool,
                      coarse: _PairCost | None = None):
    """Fit K-1 = m + c0*r^2 + c1*r^4 with a joint (m, c0) search.

    ``m`` (a constant per-channel magnification) is the leading term of lateral CA: the
    channel displacement is m*rho, i.e. linear in radius, which is what a magnification
    difference between colour channels actually produces. The Adobe polynomial alone can only
    express an r^3-shaped displacement — zero at the centre — so fitting real frames with it
    put nearly all of the correction in the corners and left the mid-field untouched.

    ``coarse`` is the same pair built on a subsample; the grid search runs there and only the
    Newton refinement pays for the full pixel set.
    """
    base = pair.cost(0.0, 0.0)
    probe = coarse if coarse is not None else pair

    grid_m = np.linspace(-max_m, max_m, 9)
    grid_c = np.linspace(-max_c0, max_c0, 13)
    best = (float("inf"), 0.0, 0.0)
    for gm in grid_m:
        for gc in grid_c:
            v = probe.cost(float(gm), float(gc))
            if v < best[0]:
                best = (v, float(gm), float(gc))

    m, c0, e0 = _newton2(lambda a, b: pair.cost(a, b), best[1], best[2],
                         -max_m, max_m, -max_c0, max_c0)
    c1 = 0.0

    if use_c1:
        span = min(CA_C1_MAX, max(4e-5, max(abs(c0), abs(m)) * 0.5))
        c1_try, e1, step1 = _scan(lambda v: pair.cost(m, c0, v), -span, span, 13)
        c1_try = _refine(lambda v: pair.cost(m, c0, v), c1_try, step1, -span, span)
        e1 = pair.cost(m, c0, c1_try)
        # The r^4 term has to earn its place: it trades off against c0, and a term fitted to
        # noise just reshapes the corners.
        if e1 < e0 * CA_TERM_MIN_GAIN:
            c1, e0 = c1_try, e1
            m, c0, e0 = _newton2(lambda a, b: pair.cost(a, b, c1), m, c0,
                                 -max_m, max_m, -max_c0, max_c0)

    gain = 1.0 - (e0 / base) if base > 0 else 0.0
    return m, c0, c1, gain


def _corner_radius_px(w: int, h: int) -> float:
    return 0.5 * float(np.hypot(w, h))


def estimate_lateral_ca(rgb: np.ndarray, aspect: float | None = None,
                        max_samples: int = 350_000, use_c1: bool = True,
                        min_gain: float = CA_MIN_GAIN,
                        min_corner_px: float = CA_MIN_CORNER_PX) -> dict:
    """Estimate lateral CA and return coefficients ready for ``compute_effective_lens_params``.

    ``rgb`` is a scene-linear float image (H, W, 3) in R, G, B order — the same array the
    lens stage receives. Returns a dict with ``ca_r_coeffs`` / ``ca_b_coeffs`` (three
    floats each), the corner shift in pixels per channel, a 0..1 confidence and whether the
    estimate was strong enough to be used at all. ``raw_r`` / ``raw_b`` carry the ungated
    c0 so callers can show what was measured even when the gate rejected it.
    """
    result = {
        "ca_r_coeffs": (0.0, 0.0, 0.0),
        "ca_b_coeffs": (0.0, 0.0, 0.0),
        "ca_r_scale": 0.0,
        "ca_b_scale": 0.0,
        "r_shift_px": 0.0,
        "b_shift_px": 0.0,
        "raw_r": 0.0,
        "raw_b": 0.0,
        "confidence": 0.0,
        "applied": False,
    }
    if rgb is None or not hasattr(rgb, "shape") or rgb.ndim != 3 or rgb.shape[2] < 3:
        return result
    h, w = rgb.shape[:2]
    if h < 64 or w < 64:
        return result

    aspect = float(aspect) if aspect else float(w) / float(h)
    green = _to_float_gray(rgb, 1)
    xs, ys, wts, _gmag = _select_edge_pixels(green, aspect, _clip_mask(rgb))
    if xs.size < 2000:
        return result

    # Deterministic thinning: a fixed stride keeps the result reproducible frame to frame.
    if xs.size > max_samples:
        step = int(np.ceil(xs.size / float(max_samples)))
        sl = slice(None, None, step)
        xs, ys, wts = xs[sl], ys[sl], wts[sl]

    dx = (xs.astype(np.float32) + 0.5) / float(w) - 0.5
    dy = (ys.astype(np.float32) + 0.5) / float(h) - 0.5
    max_r2 = 0.25 * (aspect * aspect + 1.0)
    ax = dx * np.float32(aspect)
    r2 = (ax * ax + dy * dy) / np.float32(max_r2)
    del ax, dx, dy

    wts = _equalise_radius_bins(r2, wts)

    corner = _corner_radius_px(w, h)
    max_m = min(CA_SCALE_MAX, max(2e-4, 8.0 / max(corner, 1.0)))
    max_c0 = min(CA_C0_MAX, max(2e-4, 8.0 / max(corner, 1.0)))

    # Coarse search runs on a subsample: the 117-point grid is the expensive part and the
    # minimum does not need 350k pixels to locate.
    stride = max(1, int(np.ceil(xs.size / 30_000.0)))
    cs = slice(None, None, stride)
    coarse_r2 = r2[cs]
    coarse_wts = _equalise_radius_bins(coarse_r2, wts[cs])

    out = {}
    for key, ch in (("r", 0), ("b", 2)):
        chan = np.ascontiguousarray(rgb[:, :, ch], dtype=np.float32)
        pair = _PairCost(chan, green, xs, ys, r2, wts)
        coarse = _PairCost(chan, green, xs[cs], ys[cs], coarse_r2, coarse_wts)
        out[key] = _estimate_channel(pair, max_m, max_c0, use_c1, coarse)
        del chan, pair, coarse

    # Corner displacement = (m + c0 + c1) * corner, since r^2 = 1 there.
    def _corner_px(v):
        return (v[0] + v[1] + v[2]) * corner

    # A shift under a fifth of a pixel is not a defect; don't rewrite the frame for it.
    def _accept(v):
        return abs(_corner_px(v)) >= min_corner_px and v[3] >= min_gain

    result["raw_r"] = float(_corner_px(out["r"]))
    result["raw_b"] = float(_corner_px(out["b"]))
    r_ok, b_ok = _accept(out["r"]), _accept(out["b"])
    if not (r_ok or b_ok):
        result["confidence"] = 0.0
        return result

    if r_ok:
        result["ca_r_scale"] = float(out["r"][0])
        result["ca_r_coeffs"] = (float(out["r"][1]), float(out["r"][2]), 0.0)
        result["r_shift_px"] = float(_corner_px(out["r"]))
    if b_ok:
        result["ca_b_scale"] = float(out["b"][0])
        result["ca_b_coeffs"] = (float(out["b"][1]), float(out["b"][2]), 0.0)
        result["b_shift_px"] = float(_corner_px(out["b"]))
    gains = [v[3] for v, ok in ((out["r"], r_ok), (out["b"], b_ok)) if ok]
    result["confidence"] = float(np.clip(min(gains), 0.0, 1.0))
    result["applied"] = True
    return result


def describe(result: dict) -> str:
    """Human-readable one-liner for logs and the status label."""
    if not result or not result.get("applied"):
        return "自动色差: 未检出"
    return (f"自动色差: R {result['r_shift_px']:+.2f}px / "
            f"B {result['b_shift_px']:+.2f}px (置信度 {result['confidence']*100:.0f}%)")
