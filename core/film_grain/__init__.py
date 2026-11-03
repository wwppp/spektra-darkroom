"""Film grain, ported from fotufilm-engine's film-tile model.

Status: **P3** (2026-11-02).  ``population.py`` fits the crystal population off each stock's
curve, ``tile_build.py`` lays the film, ``tiles_cache.py`` keeps it, ``sampler.py`` reads it for a
frame (the export), and ``ui/gl_film_grain.py`` reads the same tables for the canvas.  The two
agree pixel for pixel -- ``scratch/_p3_gpu_vs_cpu.py`` -- so a photographer judges the grain the
export will write, not a look-alike.  See ``docs/grain-port-plan.md``.

Why a port and not a patch
--------------------------
The removed implementation summed a handful of smooth noise octaves.  Its ceiling was "a few
blobs added together": the eye reads the blob size, never a grain population, and every knob
(amplitude, chroma, particle size) re-coupled to the others.  fotufilm does something else
entirely: each stock carries a **pre-rendered piece of film** -- crystals hashed onto millimetre
coordinates, developed or not by the local density, each developed crystal spreading into a dye
cloud -- and a pixel is just light that passed through one small patch of it.

Consequences that matter here:

* grain strength follows the *density*, so a push gets coarser grain with no per-stop patch;
* changing the output resolution re-samples the same film (0.25 µm averaged back to 1 µm
  correlates > 0.9), which is what makes preview and export agree;
* black-and-white is silver (no chroma) and colour is dye clouds; the two are not one mix.

Layout
------
``curve.py``         the H&D curve a population is read off
``population.py``    crystal population per stock, from the characteristic curve
``tile_build.py``    256x256 texel tile (1 texel = 1 um) x 17 density layers
``tiles_cache.py``   disk cache, keyed by (stock, record, seed, version)
``sampler.py``       CPU sampling: 128 um blocks, two layings, layer blending, aperture mean
``profiles.py``      which grain a stock id carries, from ``resources/profiles/grain_params.json``

The GPU sampler lives in ``ui/gl_film_grain.py``: same tables, same placement, same constants.

The tile data and the constants live in exactly one place so the GPU sampler (``ui/``) and the
CPU sampler cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "FILM_GRAIN_ACTIVE",
    "GrainParams",
    "params_for",
    "apply_block",
    "layer_value",
    "grain_source",
    "DEFAULT_DENSITY_MAX",
]

# The sampler renders, so the stage is live: a stock whose grain is in ``grain_params.json`` gets
# its film's grain through the export.  Stocks that are not in the table stay clean -- see
# ``profiles.stock_for``.
FILM_GRAIN_ACTIVE = True

# Reversal films' develop stage returns ``dmax - D``; this is the fallback ceiling when a profile
# declares no density range.  Same number the removed model used (DENS_DMAX).
DEFAULT_DENSITY_MAX = 2.24


@dataclass(frozen=True)
class GrainParams:
    """The four controls fotufilm exposes, in its own units.

    ``amount``   Grain slider (0..1); the amount of developed crystal the tile is read at.
    ``size``     Film Grain Size multiplier on the physical crystal radius.
    ``colour``   Colour Grain (0 = monochrome silver, 1 = independent dye clouds per record).
    ``scan_softness``  Reading aperture, in pixels; the mean is taken over it before -log10.
    ``layers``   Each record's share of the grain (red, green, blue).

    ``grain_cloud_blur`` from the old model is deliberately absent: fotufilm has no such control,
    so keeping it would have meant inventing a knob the reference cannot answer for.
    """

    amount: float = 0.4
    size: float = 1.0
    colour: float = 1.0
    scan_softness: float = 1.0
    layers: tuple[float, float, float] = (1.0, 1.0, 1.0)


def params_for(params: Mapping[str, Any] | None) -> GrainParams:
    """Read the grain controls out of a parameter dict, with fotufilm's defaults."""
    p = params or {}

    def number(key: str, default: float) -> float:
        value = p.get(key)
        return float(default if value is None else value)

    return GrainParams(
        amount=number("grain", 0.4),
        # No ``or 1.0`` here: the size slider starts at 0, and falling back to 1.0 would make the
        # bottom of its travel mean "standard" instead of "smallest".  ``look_geometry`` clamps.
        size=number("grain_size", 1.0),
        colour=number("grain_colour", 1.0),
        scan_softness=number("scan_softness", 1.0) or 1.0,
        layers=(number("grain_layer_r", 1.0), number("grain_layer_g", 1.0),
                number("grain_layer_b", 1.0)),
    )


def layer_value(spec, layer: int, default: float) -> float:
    """One record's value out of a per-record spec (scalar, list, or list of tuples)."""
    if spec is None:
        return float(default)
    if isinstance(spec, (int, float, np.floating, np.integer)):
        return float(spec)
    if len(spec) > layer:
        value = spec[layer]
    elif len(spec) > 0:
        value = spec[0]
    else:
        return float(default)
    return float(value[0]) if isinstance(value, (tuple, list)) else float(value)


# -- the sampler's caches ---------------------------------------------------------------------
_STORE = None            # tiles_cache.TileStore
_SAMPLERS: dict = {}     # (stock, pitch key, size, softness) -> sampler.TileSampler
_LAST: dict = {}         # stock id -> what the last call did, for the UI / the report


def grain_source(stock_id: str | None) -> str | None:
    """Where a stock's grain parameters came from, or ``None`` when it has none."""
    from . import profiles
    return profiles.source_for(stock_id)


def _store():
    global _STORE
    if _STORE is None:
        from .tiles_cache import TileStore
        _STORE = TileStore()
    return _STORE


def _sampler(stock, tiles, px_per_mm: float, params: GrainParams, seed: int):
    # ``tiles.signature`` separates developments of the same stock: the id alone would hand a
    # pushed frame the standard development's sampler, which holds the wrong film's statistics.
    key = (stock.stock_id, getattr(tiles, "signature", ""), round(px_per_mm, 3),
           round(params.size, 3), round(params.scan_softness, 3), seed)
    cached = _SAMPLERS.get(key)
    if cached is None:
        from .sampler import TileSampler
        cached = TileSampler(tiles, px_per_mm, seed=seed, size=params.size,
                             softness=params.scan_softness)
        if len(_SAMPLERS) > 4:
            _SAMPLERS.pop(next(iter(_SAMPLERS)))
        _SAMPLERS[key] = cached
    return cached


def apply_block(linear: np.ndarray, params: GrainParams, *, stock: str | None = None,
                net_density: np.ndarray | None = None, px_per_mm: float = 0.0,
                row_offset: int = 0, seed: int = 1, push_stops: int = 0) -> np.ndarray:
    """Multiply scene-linear transmittance by the stock's grain.

    ``linear``      ``(height, width, 3)`` negative transmittance for one chunk of the frame.
    ``net_density`` ``(height, width, 3)`` the negative's developed density above base per record,
                    in the stock's own units -- what picks the tile level.  ``None`` falls back to
                    ``-log10(linear)`` less the stock's base, which is close but not the same.
    ``px_per_mm``   the frame's sampling: ``long edge in pixels / the frame's long edge in mm``.
    ``row_offset``  the chunk's first row in the frame, so the grain does not restart at every
                    chunk.

    The grain is applied where the reference applies it -- on the negative, before the enlarger --
    so the print's spread applies to it as it does to the film's own.
    """
    if not FILM_GRAIN_ACTIVE or params.amount <= 0 or linear.size == 0:
        return linear
    from . import profiles
    grain_stock = profiles.stock_for(stock, push_stops)
    if grain_stock is None:
        _LAST[str(stock)] = {"active": False, "reason": "no baked grain parameters"}
        return linear

    tiles = _store().get(grain_stock)
    pitch = px_per_mm if px_per_mm and px_per_mm > 0 else 1.0 / 0.035
    sampler = _sampler(grain_stock, tiles, pitch, params, seed)

    density = net_density
    if density is None:
        density = -np.log10(np.maximum(linear, 1e-6)) - np.asarray(tiles.d_min, dtype=np.float32)
    fluctuation = sampler.apply(density, row_offset, params.amount, params.colour,
                                params.layers, monochrome=grain_stock.is_monochrome,
                                luma_correlation=grain_stock.luma_correlation)
    _LAST[str(stock)] = {
        "active": True,
        "pitch": float(sampler.pitch),
        "footprint": float(sampler.footprint),
        "source": profiles.source_for(stock),
        "sigma": float(fluctuation[..., 1].std()) if fluctuation.size else 0.0,
    }
    return (linear * np.power(10.0, -fluctuation.astype(np.float32))).astype(linear.dtype)


def last_report() -> dict:
    """What the last ``apply_block`` call did, per stock: pitch, source, measured σ."""
    return dict(_LAST)
