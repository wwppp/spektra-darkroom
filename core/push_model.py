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
    """Clone ``profile`` with the pushed curve, sub-layer split and fog.  ``profile`` is untouched."""
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

    d_min = float(entry.get("dMin") or 0.0)
    base_density = np.asarray(data.base_density, dtype=np.float64)
    fog = np.zeros_like(base_density)
    if d_min and base_density.size:
        channel_density = np.asarray(data.channel_density, dtype=np.float64)
        fog = d_min * channel_density.sum(axis=1)      # (wavelengths,) -- one neutral density unit

    pushed_profile = profile.clone()
    pushed_profile.update_data(
        density_curves=pushed,
        density_curves_layers=layers,
        base_density=base_density + fog,
    )
    return pushed_profile, applied
