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

__all__ = ["load", "load_negative", "stamp", "FILE", "FILE_NEG"]

FILE = "reversal_gain.json"
# Native stocks that the engine also ships (instax, provia 100F, E100, Kodachrome 64) have no
# point table -- their profiles are spektrafilm's own, so converting them would move the colour
# (§44).  They still need the *shape*, which is a property of reversal development rather than
# of a stock: the nine measured stocks pool onto one curve.  So they get that shape with a
# per-stock level, kept apart so `_rev_measure.py` can rewrite the measured file freely.
FILE_NATIVE = "reversal_gain_native.json"
FILE_NEG = "negative_gain.json"
_cache = {}
_neg_cache = {}


def _path(name=FILE):
    from .path_utils import get_resource_dir
    return os.path.join(get_resource_dir(), "fotufilm_chain", name)


def stamp(stock=None):
    """A cheap fingerprint for cache keys: the mtime and size of every measured file.

    All the files go in, so touching any of them invalidates the baked bundles -- a correction
    that silently kept serving the old layer table would be indistinguishable from a no-op.
    """
    parts = []
    for name in (FILE, FILE_NATIVE, FILE_NEG):
        try:
            st = os.stat(_path(name))
        except OSError:
            continue
        parts.append((name, st.st_mtime_ns, st.st_size))
    return tuple(parts) or None


def _curves(name):
    """Every entry in one of the curve files, parsed; ``{}`` if the file is absent."""
    path = _path(name)
    try:
        st = os.stat(path)
    except OSError:
        return {}
    key = (path, st.st_mtime_ns, st.st_size)
    cached = _cache.get(name)
    if cached and cached[0] == key:
        return cached[1]
    try:
        with open(path, encoding="utf-8") as fp:
            raw = json.load(fp)
    except (OSError, ValueError):
        return {}
    table = {stock: {
        "span": np.asarray(entry["span"], dtype=np.float64),
        "grid": np.linspace(0.0, 1.0, int(entry.get("grid", len(entry["mod"][0])))),
        "mod": np.asarray(entry["mod"], dtype=np.float64),
    } for stock, entry in raw.items()}
    _cache[name] = (key, table)
    return table


def load(stock):
    """The measured curve for ``stock``, or ``None``.

    Returns ``{"span": (3,), "grid": (N,), "mod": (3, N)}`` -- ``grid`` is a normalised
    density axis, so a lookup is ``np.interp(density / span[channel], grid, mod[channel])``.

    Measured stocks win; the native fallback only covers stocks no measurement can reach
    (no point table, so not in ``_rev_measure.py``'s list) yet which still need the shape.
    """
    curve = _curves(FILE).get(stock)
    if curve is not None:
        return curve
    return _curves(FILE_NATIVE).get(stock)


def lookup(curve, channel, density):
    """Target modulation at ``density`` (net, per-record units) for one record."""
    span = max(float(curve["span"][channel]), 1e-6)
    d = np.asarray(density, dtype=np.float64) / span
    return np.interp(np.clip(d, 0.0, 1.0), curve["grid"], curve["mod"][channel])


def load_negative(stock):
    """The measured amplitude correction for a colour negative: ``(3,)`` or ``None``.

    A *level*, not a curve.  For a reversal the analytic law has the wrong shape and gets
    replaced by a measured one; for a negative the ``dyeCloud`` profile already carries the
    right shape (its residual is the 1.6-1.8x level accepted in §36, plus a per-record
    imbalance).  So this is one number per record, applied on top of the law.

    Measured on a real photograph, whole frame, against the engine in film mode.  The
    synthetic ramp that works for reversals is **not** valid here: a negative goes through
    the optical print and a colour scene puts each record at a different density for the same
    display brightness, so the engine's own sigma at matched brightness disagrees by ~2x
    between ramp and photograph.  Whole-frame ratios are the only numbers we trust, which is
    why this is a scale and not a curve (see docs §43).
    """
    path = _path(FILE_NEG)
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    if _neg_cache.get("key") != key:
        try:
            with open(path, encoding="utf-8") as fp:
                raw = json.load(fp)
        except (OSError, ValueError):
            return None
        _neg_cache["table"] = {name: np.asarray(entry["gain"], dtype=np.float64)
                               for name, entry in raw.items()}
        _neg_cache["key"] = key
    gain = _neg_cache["table"].get(stock)
    if gain is None or gain.shape != (3,) or not np.isfinite(gain).all():
        return None
    return gain

