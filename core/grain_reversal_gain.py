"""Per-stock granularity-against-density curves measured off the engine, for reversals.

The analytic density laws in ``grain_bake`` are what fotufilm's *clump* grain model evaluates.
The reference we compare against is ``grainModel: film``, which does not use them at all: it
lays the emulsion's own crystal population and lets the *simulation* produce the density
dependence (see ``docs/fotufilm-port-plan-2026-11-06.md`` §40).  For a negative the analytic
``dyeCloud`` profile is close enough to the simulation (1.6-1.8x, accepted in §36).  For a
reversal it is not: ``q / (1 + q)`` is monotone in density, so it puts the grain where a slide
is *dark* and collapses to almost nothing in the clear highlights, while the simulation peaks
in the midtones and keeps a quarter of its amplitude at the clear end.  Measured on a
synthetic full-frame ramp, the midtones came out 8-25x quiet and the highlights 11-100x.

Rather than port the whole crystal model to get a one-dimensional function, the curve is
measured: render the same ramp through the engine at grain 0 and 1, take the pure-grain
sigma per tone, and divide by the sigma our own chain produces with unit modulation.  What
lands here is that target modulation, per record, on a normalised density axis.

    scratch/_rev_measure.py     regenerates this file (9 reversal stocks, ~8 minutes)

The values replace the analytic law outright for a stock that has one -- they are absolute
modulations, not corrections to the law.

Known residual: our variance goes to zero with net density (``var ~ d * sat``) while the
simulation keeps a floor, so the last few percent of the axis (the brightest highlights) stay
quieter than the engine.  No multiplicative curve can close that; it needs a variance floor in
the model itself.  See §40.4.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["load", "stamp", "FILE"]

FILE = "reversal_gain.json"
_cache = {}


def _path():
    from .path_utils import get_resource_dir
    return os.path.join(get_resource_dir(), "fotufilm_chain", FILE)


def stamp(stock=None):
    """A cheap fingerprint for cache keys: the file's mtime and size, or None."""
    try:
        st = os.stat(_path())
    except OSError:
        return None
    return (os.path.basename(_path()), st.st_mtime_ns, st.st_size)


def load(stock):
    """The measured curve for ``stock``, or ``None``.

    Returns ``{"span": (3,), "grid": (N,), "mod": (3, N)}`` -- ``grid`` is a normalised
    density axis, so a lookup is ``np.interp(density / span[channel], grid, mod[channel])``.
    """
    path = _path()
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    table = _cache.get("table")
    if _cache.get("key") != key:
        try:
            with open(path, encoding="utf-8") as fp:
                raw = json.load(fp)
        except (OSError, ValueError):
            return None
        table = {name: {
            "span": np.asarray(entry["span"], dtype=np.float64),
            "grid": np.linspace(0.0, 1.0, int(entry.get("grid", len(entry["mod"][0])))),
            "mod": np.asarray(entry["mod"], dtype=np.float64),
        } for name, entry in raw.items()}
        _cache["table"] = table
        _cache["key"] = key
    return table.get(stock)


def lookup(curve, channel, density):
    """Target modulation at ``density`` (net, per-record units) for one record."""
    span = max(float(curve["span"][channel]), 1e-6)
    d = np.asarray(density, dtype=np.float64) / span
    return np.interp(np.clip(d, 0.0, 1.0), curve["grid"], curve["mod"][channel])
