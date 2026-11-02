"""Film grain, ported from fotufilm-engine's film-tile model.

Status: **P1 done** (2026-11-02).  The old octave model (``core/grain_model.py`` + shader stage
3.5) is gone; ``population.py`` now fits the crystal population off each stock's curve and
``tile_build.py`` lays a record's film tile, both checked against the reference in
``tests/test_film_tile_build.py``.  ``apply_block`` is still a pass-through: the CPU sampler that
reads the tiles is P2.  See ``docs/grain-port-plan.md``.

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

Layout (filled in by P1/P2)
---------------------------
``population.py``   crystal population per stock, from the characteristic curves
``tile_build.py``   256x256 texel tile (1 texel = 1 um) x 17 density layers
``tiles_cache.py``  disk cache, keyed by (stock, record, seed, version)
``sampler.py``      CPU sampling: 128 um blocks, two lays, layer blending, aperture mean

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
    "DEFAULT_DENSITY_MAX",
]

# Flip to True when the tile sampler can render.  Until then every entry point is inert, which
# keeps the stage wired end to end (params -> export -> preview) with nothing to switch on.
FILM_GRAIN_ACTIVE = False

# Reversal films' develop stage returns ``dmax - D``; this is the fallback ceiling when a profile
# declares no density range.  Same number the removed model used (DENS_DMAX).
DEFAULT_DENSITY_MAX = 2.24


@dataclass(frozen=True)
class GrainParams:
    """The four controls fotufilm exposes, in its own units.

    ``amount``   Grain slider (0..1); the amount of developed crystal the tile is read at.
    ``size``     Film Grain Size multiplier on the physical crystal radius.
    ``colour``   Colour Grain (0 = monochrome silver, 1 = independent dye clouds per record).
    ``scan_softness``  Reading aperture, in um; the mean is taken over it before -log10.

    ``grain_cloud_blur`` from the old model is deliberately absent: fotufilm has no such control,
    so keeping it would have meant inventing a knob the reference cannot answer for.
    """

    amount: float = 0.4
    size: float = 1.0
    colour: float = 1.0
    scan_softness: float = 1.0


def params_for(params: Mapping[str, Any] | None) -> GrainParams:
    """Read the grain controls out of a parameter dict, with fotufilm's defaults."""
    p = params or {}
    return GrainParams(
        amount=float(p.get("grain", 0.4) or 0.0),
        size=float(p.get("grain_size", 1.0) or 1.0),
        colour=float(p.get("grain_colour", 1.0) or 0.0),
        scan_softness=float(p.get("scan_softness", 1.0) or 1.0),
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


def apply_block(linear: np.ndarray, params: GrainParams, **kwargs) -> np.ndarray:
    """Apply grain to a block of scene-linear transmittance.

    P0: returns ``linear`` unchanged.  P2 replaces the body with the tile sampler; the caller
    already passes everything it needs (the negative's density cube, the stock, the frame size),
    so only this function changes.
    """
    if not FILM_GRAIN_ACTIVE:
        return linear
    raise NotImplementedError(
        "film grain sampler not ported yet -- see docs/grain-port-plan.md P2")
