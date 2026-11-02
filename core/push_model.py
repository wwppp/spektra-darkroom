"""Push/pull development: the sidecar table and how it changes a film profile.

Why this is not a second profile
--------------------------------
The profile format stores ``density_curves`` **baseline-zeroed** -- ``develop()`` subtracts
``nanmin`` before interpolating -- so base fog, the one thing push development changes most
reliably, cannot be expressed in the curves at all.  A per-push profile would also put "Portra 800
Push 1" in the film list as a separate stock.  So a stock keeps one profile plus one table of
development conditions (``resources/profiles/push/{stock}.json``), and the condition is applied by
cloning the profile at bake time.

What a condition does to the profile
------------------------------------
1. **Characteristic curve** -- replaced outright.  The table carries the measured (or estimated)
   curve for that condition; there is nothing to blend.
2. **Sub-layer curves** -- the per-sub-layer split is kept but rescaled by the same per-exposure
   ratio as the total curve, so the three octaves of grain keep the structure P2 gave them.
3. **Fog** -- added as a *neutral* spectral density: ``dMin * sum_k channel_density[l, k]``.
   That is the spectral shape of one unit of neutral density in all three layers at once, which
   is what extra developed density in an unexposed area looks like.  It has to go in here rather
   than in the curves because ``develop()`` normalises them.

The proxy that is *not* used: ``density_curve_gamma``.  Measured earlier (docs/push-grain-plan.md
section 6): gamma = 2 makes the curve *thinner* (D 0.167 -> 0.001), it is not a push control.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

_TABLE_CACHE: dict[str, tuple[float, dict | None]] = {}
_LOCK = threading.Lock()

# How a push changes the *look* beyond the measured curve.  Both are per stop, both saturate, and
# both are 0 at standard development -- a stock at 0 stops must render exactly as it did before
# this existed.
PUSH_TOE_PER_STOP = 0.068      # display-domain toe: black lift and shadow compression
GRAIN_SCALE_PER_STOP = 0.60    # overall grain amplitude
GRAIN_SHADOW_PER_STOP = 1.60   # extra amplitude at zero density, fading to 0 at the anchor
GRAIN_FOG_PER_STOP = 2.60      # additive shadow grain (the fog's own grain), in DENS_GAIN units
FOG_GRAIN_RELATIVE = 0.15      # fluctuation of the lifted black, as a fraction of the lift itself
FOG_GRAIN_AT_FULL_SLIDER = 2.5 # the 颗粒 slider multiplies it, up to this, over its 0.40 default


def toe_amount_for(stops):
    """Display-domain toe for a push: lift the black point and flatten the shadows.

    Two things a pushed frame shows that the print stage will not carry.  The fog cannot ride in
    ``base_density`` -- it is a spatially uniform, spectrally fixed attenuation there, and the
    print's per-channel exposure correction is its exact inverse (measured: bit-identical output
    with and without, Tri-X +2, grey patches 31.0/59.0/107.0/154.0/189.0 both ways).  Nor can the
    toe ride in the characteristic curve: flattening it there measured as a no-op on the print
    (p50 61 -> 62, shadow detail 1.89 -> 1.87) because the print rebalances the whole curve.

    So both are applied where they survive, after the LUT: ``v + a(1-v)^3`` leaves white at 1.0,
    lifts the black point to ``a``, and drops the deepest shadow slope to ``1 - 3a`` -- the
    muddy-shoulder look, with the highlights untouched.
    """
    return float(min(0.20, PUSH_TOE_PER_STOP * max(0, int(stops))))


def grain_shape_for(stops):
    """(overall amplitude, extra shadow amplitude, shadow floor) for a push.

    1.0 / 0.0 / 0.0 at standard development.  The third number is the fog's own grain, in units
    of ``DENS_GAIN``: it is added (not multiplied) at low density, because the stock's law is a
    function of density and reports no fluctuation where nothing was developed -- which is the
    one place a pushed frame is actually the grainiest.
    """
    stops = max(0, int(stops))
    return (1.0 + GRAIN_SCALE_PER_STOP * stops,
            min(3.0, GRAIN_SHADOW_PER_STOP * stops),
            min(4.2, GRAIN_FOG_PER_STOP * stops))


def fog_grain_for(stops):
    """Fluctuation of the lifted black point, relative to the lift itself.  0 at standard.

    The toe lifts the crushed end of the frame to a constant ``a``.  Measured on a frame that
    renders below the paper's black point, that end is *exactly* one value: every pixel the same,
    sigma 0.00, 100% zoom included -- because the grain stage multiplies the film's transmittance
    before the LUT, and a transmittance of ~0 stays ~0 through the print whatever you multiply it
    by, while the lift is painted on afterwards as a constant.

    Developed fog is not a constant, though: it is silver, and it is granular.  This factor is
    that granularity, expressed against the lift so it scales with the push and vanishes with it.
    """
    if not stops:
        return 0.0
    return FOG_GRAIN_RELATIVE


# A negative this empty carries nothing but fog; from BLANK_IMAGE_DENSITY up it has image of its
# own and the ordinary density law already gives it grain.  Between the two the weight fades.
BLANK_FOG_DENSITY = 0.06
BLANK_IMAGE_DENSITY = 0.30


def blank_weight(density):
    """How empty the negative is at each pixel: 1 at fog level, 0 once the image starts.

    The display-domain grain cannot be keyed on print brightness.  Measured on a real frame
    (Gold 200 / Endura, -2 EV, +2): one patch went sigma 0.95 -> 0.14 as the paper exposure rose
    from x2.6 to x4.0, because that patch sat on a flat part of the paper curve -- and a flat
    curve turns any density modulation into no brightness change at all.  Stepping back to the
    film shows what actually separates the two cases: an empty negative (D at fog level) has no
    dye clouds, only silver fog, and silver fog is grainy no matter where the print places it.
    So the weight is the negative's emptiness, not the print's darkness.
    """
    d = np.asarray(density, dtype=np.float32)
    if d.ndim == 3:
        d = d.mean(axis=2)
    t = np.clip((BLANK_IMAGE_DENSITY - d) / (BLANK_IMAGE_DENSITY - BLANK_FOG_DENSITY), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def push_dir(resources_dir):
    return os.path.join(resources_dir, "profiles", "push")


def load_table(resources_dir, stock_id):
    """The sidecar for ``stock_id``, or ``None``.  Cached on (path, mtime)."""
    if not stock_id:
        return None
    path = os.path.join(push_dir(resources_dir), f"{stock_id}.json")
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    with _LOCK:
        cached = _TABLE_CACHE.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    try:
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    except (OSError, ValueError):
        return None
    table["log_exposure"] = np.asarray(table["log_exposure"], dtype=np.float64)
    for entry in table.get("conditions", {}).values():
        entry["curves"] = np.asarray(entry["curves"], dtype=np.float64)
    with _LOCK:
        _TABLE_CACHE[path] = (stamp, table)
    return table


def available_stops(resources_dir, stock_id):
    """Sorted push stop counts this stock has a table for (empty when it has none)."""
    table = load_table(resources_dir, stock_id)
    if not table:
        return []
    stops = []
    for key in table.get("conditions", {}):
        try:
            stops.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(s for s in stops if s > 0)


def max_stops(resources_dir, stock_id):
    stops = available_stops(resources_dir, stock_id)
    return max(stops) if stops else 0


def condition(table, stops):
    """The entry for ``stops``, or the nearest lower one (a +3 request on a +1/+2 table)."""
    if not table:
        return None, 0
    available = sorted(int(k) for k in table.get("conditions", {}))
    if not available:
        return None, 0
    chosen = max((s for s in available if s <= stops), default=None)
    if chosen is None:
        chosen = min(available)
    return table["conditions"][str(chosen)], chosen


def curve_ratio(base_curves, pushed_curves):
    """Per-record multiplier that carries the base curve onto the pushed one.

    ``develop()`` subtracts ``nanmin`` before interpolating, so only shape matters here; the ratio
    is what the sub-layer curves need to follow the total curve.  Where the base curve is flat
    (its floor) the ratio is held at its neighbour instead of exploding.
    """
    base = np.asarray(base_curves, dtype=np.float64)
    pushed = np.asarray(pushed_curves, dtype=np.float64)
    floor = base - np.nanmin(base, axis=0, keepdims=True)
    top = pushed - np.nanmin(pushed, axis=0, keepdims=True)
    ratio = np.ones_like(floor)
    usable = floor > 1e-3
    ratio[usable] = top[usable] / floor[usable]
    return np.clip(ratio, 0.2, 5.0)


def apply_to_profile(profile, table, stops):
    """Clone ``profile`` with the pushed curve and its sub-layer split.  ``profile`` is untouched.

    The table's ``dMin`` no longer goes into ``base_density``: measured, the print stage's
    exposure correction cancels a uniform neutral fog exactly (bit-identical output), so the fog
    is applied after the LUT by :func:`black_lift_for` instead.  ``dMin`` still moves the grain
    preset's ``density_min``, which is where it does have an effect.
    """
    entry, applied = condition(table, stops)
    if entry is None or profile is None:
        return profile, 0
    data = profile.data
    base_curves = np.asarray(data.density_curves, dtype=np.float64)
    pushed = np.asarray(entry["curves"], dtype=np.float64)
    if pushed.shape != base_curves.shape:
        return profile, 0

    ratio = curve_ratio(base_curves, pushed)
    layers = np.asarray(data.density_curves_layers, dtype=np.float64)
    if layers.ndim == 3 and layers.shape[0] == ratio.shape[0]:
        layers = layers * ratio[:, :, None]

    pushed_profile = profile.clone()
    pushed_profile.update_data(
        density_curves=pushed,
        density_curves_layers=layers,
    )
    return pushed_profile, applied
