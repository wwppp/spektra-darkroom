"""lens_warp.py - CPU mirror of the viewport's lens-correction block.

The GPU draws lens correction inside the fragment shader (``ui/canvas_viewport.py``,
"Apply Lens Distortion Warp" / "Lateral CA" / "Lens Vignette"). Everything else — the
CPU export pipeline and the CPU thumbnail renderer — had no lens stage at all, so a
frame exported with hardware acceleration off silently lost its distortion, vignetting
and CA correction while the preview still showed them.

This module is the single CPU implementation of that stage. It is a line-for-line
mirror of the shader maths, deliberately: the two must agree to sub-pixel, and the
only way to keep them agreeing is to keep the constants in one shape. If you change
``r_dist``, ``max_r2``, ``scale`` or the vignette exponent in the shader, change it
here in the same commit.

Order of operations matches the shader exactly:
    distortion (resample) -> lateral CA (per-channel resample) -> vignette (linear gain)

Performance notes (24 MP, 6024x4024):
  * Every radius term is separable — ``r^2 = A(x) + B(y)`` — so the grid is built from
    two 1-D arrays broadcast once instead of three full meshgrids.
  * ``cv2.pow`` is used for the vignette exponent (46 ms) instead of ``np.power``
    (164 ms).
  * The gain is applied per channel in place; broadcasting a (H,W,1) gain over an
    (H,W,3) frame measured 6x slower than three in-place channel multiplies.
"""

from __future__ import annotations

import numpy as np

try:  # cv2 is a hard dependency of the pipeline, but keep the import defensive.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


SQRT_2 = 1.4142135623730951


def _lens_values(eff):
    """Read the effective lens dict defensively; missing keys mean 'do nothing'."""
    eff = eff or {}
    try:
        k1 = float(eff.get("k1", 0.0) or 0.0)
        k2 = float(eff.get("k2", 0.0) or 0.0)
        k3 = float(eff.get("k3", 0.0) or 0.0)
    except (TypeError, ValueError):
        k1 = k2 = k3 = 0.0
    try:
        v_amt = float(eff.get("vignette_amount", 0.0) or 0.0)
        v_mid = float(eff.get("vignette_midpoint", 50.0) or 50.0)
    except (TypeError, ValueError):
        v_amt, v_mid = 0.0, 50.0
    try:
        ca_r = float(eff.get("ca_kr", 0.0) or 0.0)
        ca_b = float(eff.get("ca_kb", 0.0) or 0.0)
    except (TypeError, ValueError):
        ca_r = ca_b = 0.0
    return k1, k2, k3, v_amt, v_mid, ca_r, ca_b


def has_distortion(eff) -> bool:
    k1, k2, k3, _v, _m, _ca_r, _ca_b = _lens_values(eff)
    return bool((eff or {}).get("enabled", False)) and (
        abs(k1) > 1e-5 or abs(k2) > 1e-5 or abs(k3) > 1e-5
    )


def _ca_coeffs(eff, channel: str):
    """Three-term lateral-CA polynomial for one channel, with the legacy single-term key
    as a fallback for sidecars written before the LCP CA models were parsed."""
    if not isinstance(eff, dict) or channel not in ("r", "b"):
        return None
    raw = eff.get("ca_r_coeffs" if channel == "r" else "ca_b_coeffs")
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        try:
            return float(raw[0]), float(raw[1]), float(raw[2])
        except (TypeError, ValueError):
            pass
    legacy = eff.get("ca_kr" if channel == "r" else "ca_kb", 0.0) or 0.0
    try:
        return float(legacy), 0.0, 0.0
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0


def _ca_scale(eff, channel: str) -> float:
    """Constant per-channel magnification for lateral CA.

    A real lens focuses red and blue at slightly different magnifications, so the channel
    displacement grows linearly with radius — the leading, and usually dominant, term. The
    Adobe polynomial only covers the residual field-dependent part (its first basis is r^2,
    i.e. displacement ~ r^3, which vanishes at the centre), which is why a profile-style fit
    alone piles the correction into the corners.
    """
    if not isinstance(eff, dict) or channel not in ("r", "b"):
        return 0.0
    try:
        return float(eff.get("ca_r_scale" if channel == "r" else "ca_b_scale", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ca_active(coeffs) -> bool:
    return bool(coeffs) and any(abs(c) > 1e-9 for c in coeffs)


def has_ca(eff) -> bool:
    if not isinstance(eff, dict) or not eff.get("remove_ca", False):
        return False
    return (_ca_active(_ca_coeffs(eff, "r")) or _ca_active(_ca_coeffs(eff, "b"))
            or abs(_ca_scale(eff, "r")) > 1e-9 or abs(_ca_scale(eff, "b")) > 1e-9)


def has_defringe(eff) -> bool:
    """True when the shader's defringe block would run. It is gated on ``remove_ca`` alone,
    not on ``enabled`` — the same rule the viewport uses for ``u_lens_ca_enabled``."""
    return bool(isinstance(eff, dict) and eff.get("remove_ca", False))


def has_geometry(eff) -> bool:
    """True when distortion or CA would move pixels."""
    return has_distortion(eff) or has_ca(eff)


def has_vignette(eff) -> bool:
    _k1, _k2, _k3, v_amt, _m, _ca_r, _ca_b = _lens_values(eff)
    return bool((eff or {}).get("enabled", False)) and abs(v_amt) > 1e-5


def fill_scale(k1: float, k2: float, k3: float, aspect: float,
               ca_expand: float = 0.0, samples: int = 257) -> float:
    """Largest uniform scale that keeps the *whole* border inside the source frame.

    The renderers used to normalise by the corner term alone, ``1/(1 + k1 + k2 + k3)``,
    which only satisfies the border constraint when the corner happens to be the worst
    offender. Barrel correction pushes the mid-edges out further than the corners, so the
    sampling ran off the frame there and the clamp/replicate filled the gap with stretched
    pixels -- the "拉丝" band along the edges.

    Concretely: the output sample is ``0.5 + delta * r_dist * scale``, so the frame is
    covered iff ``|delta_i * r_dist * scale| <= 0.5`` for every border point i. The scale
    is therefore the minimum over the border, and it is *smaller* than the corner-only
    value whenever the corner is not the extremum -- i.e. this can only ever zoom in a
    touch more, never leave a gap.

    ``ca_expand`` is the largest positive CA radial scale; it multiplies the sampled
    coordinate after the distortion term, so it has to be divided out here as well.
    """
    if abs(k1) < 1e-12 and abs(k2) < 1e-12 and abs(k3) < 1e-12:
        return 1.0
    aspect = float(aspect) if aspect and aspect > 0 else 1.5
    max_r2 = 0.25 * (aspect * aspect + 1.0)
    t = np.linspace(-0.5, 0.5, samples, dtype=np.float64)
    edge = np.full_like(t, 0.5)
    edge_n = -edge
    dx = np.concatenate([edge, edge_n, t, t])
    dy = np.concatenate([t, t, edge, edge_n])
    r2 = (dx * dx * aspect * aspect + dy * dy) / max_r2
    rd = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    sx = np.abs(dx * rd)
    sy = np.abs(dy * rd)
    lim = np.full_like(sx, np.inf)
    np.divide(0.5, sx, out=lim, where=sx > 1e-6)
    lim2 = np.full_like(sy, np.inf)
    np.divide(0.5, sy, out=lim2, where=sy > 1e-6)
    s = float(np.minimum(lim, lim2).min())
    if not np.isfinite(s):
        return 1.0
    s /= max(1.0, 1.0 + max(0.0, float(ca_expand)))
    return float(np.clip(s, 0.05, 4.0))


def _axis_uv(w: int, h: int):
    """1-D pixel-centre UV offsets from the frame centre (fully separable radius terms)."""
    dx = (np.arange(w, dtype=np.float32) + 0.5) / float(w) - 0.5
    dy = (np.arange(h, dtype=np.float32) + 0.5) / float(h) - 0.5
    return dx, dy


def lens_remap_maps(w: int, h: int, eff, channel: str = "g"):
    """Build cv2.remap maps reproducing the shader's distortion (+ optional CA) shift.

    ``channel`` selects the lateral-CA coefficient applied on top of the distortion
    shift: red uses ``ca_kr``, blue ``ca_kb``, green none.
    """
    k1, k2, k3, _v, _m, ca_r, ca_b = _lens_values(eff)
    dx, dy = _axis_uv(w, h)
    aspect = float(w) / float(h)
    max_r2 = 0.25 * (aspect * aspect + 1.0)

    # r^2 = A(x) + B(y): keep both as 1-D and broadcast exactly once.
    ax = np.multiply(dx, aspect)
    np.square(ax, out=ax)
    ax /= max_r2
    by = np.square(dy)
    by /= max_r2

    if has_distortion(eff):
        r2 = ax[None, :] + by[:, None]
        r2sq = np.multiply(r2, r2)
        # Horner-free but in place: rd = 1 + k1*r2 + k2*r2^2 + k3*r2^3
        rd = np.multiply(r2, np.float32(k1))
        rd += np.float32(1.0)
        if abs(k2) > 1e-9:
            rd += r2sq * np.float32(k2)
        if abs(k3) > 1e-9:
            rd += (r2sq * r2sq) * np.float32(k3)
        # Same guard as the shader. Both directions are rescaled: skipping it for
        # pincushion correction pushed the corners past the frame and the border clamp
        # smeared them into streaks. ``dist_scale`` overrides it for profiles that carry
        # their own normalisation (Sony's tables are normalised at the centre, not the
        # corner, so re-normalising here would shrink the frame by a further 15%).
        corner_k = k1 + k2 + k3
        try:
            ds = float(eff.get("dist_scale", 0.0) or 0.0)
        except (TypeError, ValueError):
            ds = 0.0
        if abs(ds) > 1e-9:
            scale = ds
        else:
            # Border-exact scale from compute_effective_lens_params; the corner-only
            # fallback remains for callers that build an eff dict by hand.
            fs = 0.0
            try:
                fs = float(eff.get("fill_scale", 0.0) or 0.0)
            except (TypeError, ValueError):
                fs = 0.0
            scale = fs if fs > 1e-6 else 1.0 / max(0.05, 1.0 + corner_k)
        su = np.multiply(rd, dx[None, :])
        sv = np.multiply(rd, dy[:, None])
        su *= np.float32(scale)
        sv *= np.float32(scale)
        su += np.float32(0.5)
        sv += np.float32(0.5)
        del r2, r2sq, rd
    else:
        su = np.empty((h, w), np.float32)
        sv = np.empty((h, w), np.float32)
        su[:] = (dx + np.float32(0.5))[None, :]
        sv[:] = (dy + np.float32(0.5))[:, None]

    # Lateral CA rides on the already-distorted coordinate, exactly like the shader.
    coeffs = _ca_coeffs(eff, channel)
    ca_const = _ca_scale(eff, channel)
    if (_ca_active(coeffs) or abs(ca_const) > 1e-9) and bool((eff or {}).get("remove_ca", False)):
        c0, c1, c2 = coeffs
        su -= np.float32(0.5)
        sv -= np.float32(0.5)
        # Same corner-normalised radius the distortion term uses.
        ax_c = np.multiply(su, np.float32(aspect))
        np.square(ax_c, out=ax_c)
        r2_ca = ax_c
        r2_ca += np.square(sv)
        r2_ca /= np.float32(max_r2)
        r4_ca = np.multiply(r2_ca, r2_ca)
        r6_ca = np.multiply(r4_ca, r2_ca)
        scale_ca = np.multiply(r2_ca, np.float32(c0))
        scale_ca += np.multiply(r4_ca, np.float32(c1))
        scale_ca += np.multiply(r6_ca, np.float32(c2))
        scale_ca += np.float32(1.0 + ca_const)
        # Fade the shift out over the last few pixels of the frame.  At the border the R and B
        # samples clamp to opposite sides of the edge texel, so the outermost column came out
        # as a saturated rim of its own -- a fringe that belongs to the resampler, not the lens.
        edge_x = (np.float32(0.5) - np.abs(su)) * np.float32(w)
        edge_y = (np.float32(0.5) - np.abs(sv)) * np.float32(h)
        fade = np.clip(np.minimum(edge_x, edge_y) / np.float32(3.0), 0.0, 1.0)
        scale_ca = np.float32(1.0) + (scale_ca - np.float32(1.0)) * fade
        su *= scale_ca
        sv *= scale_ca
        su += np.float32(0.5)
        sv += np.float32(0.5)

    su *= np.float32(w)
    sv *= np.float32(h)
    su -= np.float32(0.5)
    sv -= np.float32(0.5)
    return su, sv


def lens_vignette_gain(w: int, h: int, eff) -> np.ndarray | None:
    """Radial gain map in linear light, matching the shader's vignette block."""
    _k1, _k2, _k3, v_amt, v_mid, _ca_r, _ca_b = _lens_values(eff)
    if not bool((eff or {}).get("enabled", False)) or abs(v_amt) <= 1e-5:
        return None
    dx, dy = _axis_uv(w, h)
    # r_norm = |2*delta| / sqrt(2)  ->  r_norm^2 = 2*(dx^2 + dy^2)
    ax = np.multiply(dx, dx) * np.float32(2.0)
    by = np.multiply(dy, dy) * np.float32(2.0)
    r_sq = ax[None, :] + by[:, None]
    np.minimum(r_sq, np.float32(1.0), out=r_sq)
    np.maximum(r_sq, np.float32(0.0), out=r_sq)
    v_exp = 1.2 + (4.0 - 1.2) * (1.0 - min(max(v_mid / 100.0, 0.1), 0.9))
    # gain = 1 + amount * r_norm^v_exp, and r_norm = sqrt(r_sq) -> exponent halves.
    half_exp = float(v_exp) * 0.5
    if cv2 is not None:
        gain = cv2.pow(r_sq, half_exp)
    else:  # pragma: no cover
        gain = np.power(r_sq, np.float32(half_exp))
    gain *= np.float32(v_amt)
    gain += np.float32(1.0)
    np.maximum(gain, np.float32(0.0), out=gain)
    return gain


def apply_lens_geometry(rgb: np.ndarray, eff) -> np.ndarray:
    """Apply distortion + CA + vignette to a scene-linear float32 RGB frame.

    A no-op (returns the input object) when lens correction is off, which keeps the
    common path free of any allocation.
    """
    if rgb is None or not isinstance(eff, dict):
        return rgb
    if not has_geometry(eff) and not has_vignette(eff) and not has_defringe(eff):
        return rgb
    if cv2 is None:
        return rgb

    out = rgb
    h, w = out.shape[:2]
    interpolate = cv2.INTER_LINEAR
    border = cv2.BORDER_REPLICATE  # mirrors the shader's clamp(sample_uv, half_texel, ...)

    if has_geometry(eff):
        mx, my = lens_remap_maps(w, h, eff, "g")
        out = cv2.remap(out, mx, my, interpolate, borderMode=border)
        del mx, my
        if has_ca(eff):
            # Split into channels only when the correction actually moves R and B.
            r_map_x, r_map_y = lens_remap_maps(w, h, eff, "r")
            b_map_x, b_map_y = lens_remap_maps(w, h, eff, "b")
            ch_r = cv2.remap(out[:, :, 0], r_map_x, r_map_y, interpolate, borderMode=border)
            ch_b = cv2.remap(out[:, :, 2], b_map_x, b_map_y, interpolate, borderMode=border)
            del r_map_x, r_map_y, b_map_x, b_map_y
            out = out.copy()
            out[:, :, 0] = ch_r
            out[:, :, 2] = ch_b

    # Shader order: distortion -> lateral CA -> defringe -> vignette.
    if has_defringe(eff):
        out = lens_defringe(out, ref=rgb)

    gain = lens_vignette_gain(w, h, eff)
    if gain is not None:
        if out is rgb:
            out = out.copy()
        # Per-channel in place: ~130 ms at 24 MP vs ~760 ms for a (H,W,1) broadcast.
        for c in range(3):
            out[:, :, c] *= gain

    return out


# Must stay in sync with the constants in ui/canvas_viewport.py.
_DF_STRENGTH = 1.5
_DF_HUE_MIN = 0.55
_DF_EXCESS_FLOOR = 0.50
_DF_RIDGE_TOL = 0.98
_DF_CONTRAST_MIN = 0.08
_DF_REL_FLOOR = 0.08
_DF_RADIUS = 2
# Both sides of the fringe, along the luminance normal, must be at most 1/1.5 of the
# pixel's own chroma. See lens_defringe() for why a ring test cannot do this.
_DF_SIDE_RATIO = 2.0
_DF_SIDE_TAPS = (2.0, 3.0)
_DF_LUM = (np.float32(0.299), np.float32(0.587), np.float32(0.114))
# Cross-shaped 5x5: the four neighbours at radius 2 plus the centre (dilate/erode keep the
# centre), i.e. exactly the ring the shader samples.
_DF_RING = np.array([[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]], np.uint8)


def _sample_bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear tap at float coordinates, edge-replicated — the shader's textureGrad path."""
    h, w = a.shape[:2]
    x = np.clip(x, 0.0, w - 1.000001)
    y = np.clip(y, 0.0, h - 1.000001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    s0 = a[y0, x0] * (1.0 - fx) + a[y0, x1] * fx
    s1 = a[y1, x0] * (1.0 - fx) + a[y1, x1] * fx
    return s0 * (1.0 - fy) + s1 * fy


def _tap_int(a: np.ndarray, ddx: int, ddy: int) -> np.ndarray:
    """Integer-offset tap with edge replication — matches one axis-aligned textureGrad tap."""
    h, w = a.shape[:2]
    ys = np.clip(np.arange(h) + ddy, 0, h - 1)
    xs = np.clip(np.arange(w) + ddx, 0, w - 1)
    return a[ys[:, None], xs[None, :]]


def _normal_profile_weight(ch: np.ndarray, lum: np.ndarray, sel: np.ndarray,
                           w: int) -> np.ndarray:
    """Per-pixel weight in 0..1 for "this chroma ridge is a band, not a step".

    ``sel`` holds flat indices of the survivors of the local tests. The normal comes from the
    luminance gradient (a fringe always lies across a luminance step, whatever its hue), and
    the chroma is tapped at ±2 and ±3 px along it. Everything here is expressed as the four
    texture taps the shader takes, so the two implementations stay in step.

    The gate is a smooth ramp rather than a hard threshold: a fringe sitting exactly at the
    cut-off would otherwise flip between passing and failing on a rounding difference between
    the GPU and CPU tap positions, which showed up as a 30 dB CPU/GPU agreement on the 2 px
    synthetic band even though both sides removed the same fringe.
    """
    ys, xs = np.divmod(sel, w)
    fx = xs.astype(np.float32)
    fy = ys.astype(np.float32)

    # Same four taps the shader uses for its ring: (x+2, x-2, y+2, y-2).
    dx = (_tap_int(lum, 2, 0).ravel()[sel] - _tap_int(lum, -2, 0).ravel()[sel])
    dy = (_tap_int(lum, 0, 2).ravel()[sel] - _tap_int(lum, 0, -2).ravel()[sel])
    nrm = np.hypot(dx, dy)
    # A flat neighbourhood has no normal; the contrast gate has already rejected those, so let
    # them through rather than inventing a direction from rounding noise.
    live = nrm > np.float32(1e-6)
    inv = np.zeros_like(nrm)
    np.divide(1.0, nrm, out=inv, where=live)
    ux = dx * inv
    uy = dy * inv

    side = np.zeros(sel.size, np.float32)
    for t in _DF_SIDE_TAPS:
        for sgn in (-1.0, 1.0):
            off = np.float32(t * sgn)
            np.maximum(side, _sample_bilinear(ch, fx + ux * off, fy + uy * off), out=side)
    ramp = (ch.ravel()[sel] / np.maximum(side, np.float32(1e-6)) - np.float32(1.0)) \
        / np.float32(_DF_SIDE_RATIO - 1.0)
    np.clip(ramp, 0.0, 1.0, out=ramp)
    ramp = ramp * ramp * (np.float32(3.0) - np.float32(2.0) * ramp)
    # No normal means no way to ask the question — and setting ux=uy=0 would sample the pixel
    # itself and force the weight to zero. A fringe narrower than the ±2 px probe is exactly
    # this case (an isolated 1 px band is symmetric about its own centre), so it must pass.
    ramp[~live] = np.float32(1.0)

    return ramp


def lens_defringe(rgb: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Mirror of the shader's edge-defringe block (``u_lens_ca_enabled``).

    A colour fringe is additive colour laid over a surface, so the signature is a CHROMA RIDGE:
    the pixel is more chromatic than the ring around it, sits on a CA hue, and is at a local
    luminance edge. Chroma *valleys* (anti-aliasing between two surfaces, petal creases) are
    deliberately left alone — on the branch test frame 77% of what looks like a purple rim is a
    valley, i.e. neutral AA, not added colour.

    The ring test alone is not enough, though. A 2 px cross ring only compares the pixel with
    the most chromatic neighbour it can find, and a *step* passes that: at the rim of a small
    saturated object (the blossom frame's flowers are ~5 px across) one side of the ring is the
    object's own interior, so "as chromatic as the ring" holds and the entire contour gets
    neutralised. Measured on DSC02662 that cost 11.1% of the flowers' chroma while touching
    2.96% of the frame.

    What actually separates a fringe from an object's own edge is the shape of the profile
    ACROSS the luminance edge: a fringe is a band with low chroma on BOTH sides, an object edge
    is a step that stays chromatic on one side. So the pixel must also be at least
    ``_DF_SIDE_RATIO`` times more chromatic than the brightest chroma found along the luminance
    normal at ±2 and ±4 px. On DSC02729 this keeps the fringe removal bit-identical
    (overexposed-edge magenta 0.046 -> 0.028 either way) while the blossom chroma loss drops
    from 11.1% to 2.3%.

    The repair is a neutralisation at the pixel's own luminance: nothing is borrowed from a
    neighbour, so a real subject can never be dragged toward a colour it does not have.
    """
    if rgb is None or cv2 is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        return rgb

    src_dtype = rgb.dtype
    img = np.asarray(rgb, dtype=np.float32)
    # ``ref`` is the frame the *neighbours* are taken from. The shader samples the uncorrected
    # texture for its ring taps while evaluating the corrected pixel, so the CPU has to do the
    # same or the two drift apart once lateral CA actually moves something.
    ref_img = img if ref is None else np.asarray(ref, dtype=np.float32)

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    lum = r * _DF_LUM[0] + g * _DF_LUM[1] + b * _DF_LUM[2]
    mx = np.maximum(np.maximum(r, g), b)
    np.maximum(mx, np.float32(1e-4), out=mx)
    ch = mx - np.minimum(np.minimum(r, g), b)

    if ref_img is img:
        ref_lum, ref_ch = lum, ch
    else:
        rr = ref_img[:, :, 0]
        rg = ref_img[:, :, 1]
        rb = ref_img[:, :, 2]
        ref_lum = rr * _DF_LUM[0] + rg * _DF_LUM[1] + rb * _DF_LUM[2]
        ref_ch = np.maximum(np.maximum(rr, rg), rb) - np.minimum(np.minimum(rr, rg), rb)

    # The 2 px ring as a cross: identical to the four texture taps the shader takes, but a
    # morphological erode/dilate costs ~60 ms for 24 MP against ~1.5 s of array arithmetic.
    c_hi = cv2.dilate(ch, _DF_RING)
    c_lo = cv2.erode(ch, _DF_RING)
    l_hi = cv2.dilate(lum, _DF_RING)
    l_lo = cv2.erode(lum, _DF_RING)

    excess = (ch - c_lo) / np.maximum(ch, np.float32(1e-6))
    contrast = (l_hi - l_lo) / np.maximum(lum, np.float32(1e-3))
    gate = (ch > np.float32(_DF_REL_FLOOR) * mx)
    gate &= excess > np.float32(_DF_EXCESS_FLOOR)
    gate &= ch >= c_hi * np.float32(_DF_RIDGE_TOL)
    gate &= contrast > np.float32(_DF_CONTRAST_MIN)
    del c_hi, c_lo, l_hi, l_lo, contrast

    sel = np.flatnonzero(gate)
    del gate
    if sel.size == 0:
        return rgb

    # Hue and weight only for the survivors: the four CA-hue terms on all 24 M pixels cost more
    # than everything else in this function put together.
    flat = img.reshape(-1, 3)
    sub = flat[sel]
    sr, sg, sb = sub[:, 0], sub[:, 1], sub[:, 2]
    hue = np.maximum(np.float32(0.0), np.minimum(sr, sb) - sg)
    hue += np.maximum(np.float32(0.0), sb - np.maximum(sr, sg))
    hue += np.maximum(np.float32(0.0), sg - np.maximum(sr, sb))
    hue += np.maximum(np.float32(0.0), sr - np.maximum(sg, sb))
    hue /= np.maximum(ch.ravel()[sel], np.float32(1e-6))

    keep = hue > np.float32(_DF_HUE_MIN)
    if not keep.any():
        return rgb
    sel = sel[keep]
    sub = sub[keep]
    hue = hue[keep]

    soft = _normal_profile_weight(ref_ch, ref_lum, sel, w)
    nz = soft > 0.0
    if not nz.all():
        sel = sel[nz]
        sub = sub[nz]
        hue = hue[nz]
        soft = soft[nz]
        if sel.size == 0:
            return rgb

    wgt = np.minimum(excess.ravel()[sel] * np.float32(_DF_STRENGTH), np.float32(1.0))
    wgt *= np.minimum(hue / np.float32(_DF_HUE_MIN), np.float32(1.0))
    wgt *= soft
    del excess, hue, ch, mx, soft

    lum_px = sub[:, 0] * _DF_LUM[0] + sub[:, 1] * _DF_LUM[1] + sub[:, 2] * _DF_LUM[2]
    out = img.copy()
    out.reshape(-1, 3)[sel] = sub + (lum_px[:, None] - sub) * wgt[:, None]
    return out if src_dtype == np.float32 else out.astype(src_dtype, copy=False)


def lens_warp_status(eff) -> str:
    """One-line description used by export logging / diagnostics."""
    if not isinstance(eff, dict):
        return "lens: off"
    k1, k2, k3, v, _m, ca_r, ca_b = _lens_values(eff)
    return (f"lens: enabled={bool(eff.get('enabled'))} k=({k1:.5f},{k2:.5f},{k3:.5f}) "
            f"vignette={v:.3f} ca=({ca_r:.5f},{ca_b:.5f}) "
            f"ca_scale=({_ca_scale(eff, 'r'):+.5f},{_ca_scale(eff, 'b'):+.5f}) "
            f"remove_ca={bool(eff.get('remove_ca'))}")
