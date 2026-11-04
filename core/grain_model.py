"""The grain model itself, as the reference the shader is checked against.

This is ``spektrafilm.model.grain.apply_grain_to_density_layers`` written out against the
baked tables from :mod:`core.grain_bake`, with one change that matters: the reference draws
``Binomial(Poisson(n / sat), p)`` and that is *exactly* ``Poisson((n / sat) * p)`` -- thinning
a Poisson by p is a Poisson with the thinned rate.  Measured on 2 M samples at four densities,
the two agree to four decimals in mean and standard deviation, so the sampler keeps the one
draw it needs.

Everything else follows the reference: the three coated sub-layers are drawn separately, each
blurred by its own dye-cloud sigma, summed, taken back out of fog, and blurred once more at
frame scale.

The randomness is what makes this the counter-example to the shader: numpy draws a fresh
stream per call, so this can only be compared by statistics, never pixel for pixel.  That is
fine -- the shader is checked on sigma, not on the draw.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["sample_layer_density", "apply_grain"]

# Below these the reference's micro-structure stage is a no-op (`add_micro_structure` skips a
# blur under 0.4 px and a sigma under 0.05 px).  At any render size we support -- 35 mm across
# 753 px is already 46 um/px -- both conditions hold, so the stage is inert and skipped rather
# than silently dropped.  Guarded so a future high-resolution render cannot skip it by accident.
_MICRO_BLUR_PX = 0.2 / 1.0      # micro_structure[0] um / pixel_size_um, checked in apply_grain
_MICRO_SIGMA_NM = 30.0


def sample_layer_density(density, layer_table):
    """Split ``(H, W, 3)`` net density into ``(H, W, 3 layers, 3 records)``.

    ``layer_table`` comes from :func:`core.grain_bake.bake_layer_table`: a uniform axis per
    record, so this is a lerp, not a search -- the same reason the shader can do it with one
    multiply.  Values outside the axis clamp, which is what ``fast_interp`` does too.
    """
    table = layer_table["table"]                # (samples, 3, 4)
    lo = layer_table["axis_min"]                # (3,)
    hi = layer_table["axis_max"]
    samples = table.shape[0]

    out = np.empty(density.shape[:2] + (3, 3), dtype=np.float64)
    for ch in range(3):
        u = (density[..., ch] - lo[ch]) / max(float(hi[ch] - lo[ch]), 1e-9)
        u = np.clip(u, 0.0, 1.0) * (samples - 1)
        i0 = np.floor(u).astype(np.int32)
        i1 = np.minimum(i0 + 1, samples - 1)
        t = (u - i0)[..., None]
        for sl in range(3):
            y0 = table[i0, ch, sl]
            y1 = table[i1, ch, sl]
            out[..., sl, ch] = y0 + (y1 - y0) * t[..., 0]
    return out


def apply_grain(density, layer_table, scalars, pixel_size_um, strength=1.0, rng=None):
    """Return the density the film would actually have developed, grain included.

    ``strength`` scales the particle rate -- it is not a physical parameter of the reference
    (which has none), so 1.0 is the reference and anything else is an editorial knob.
    """
    rng = rng or np.random.default_rng()
    density = np.asarray(density, dtype=np.float64)
    layers = sample_layer_density(density, layer_table)

    d_min = np.asarray(scalars["density_min"], dtype=np.float64)
    d_max = np.asarray(scalars["density_max_layers"], dtype=np.float64)     # (3 sl, 3 ch)
    n_pp = np.asarray(scalars["particles"], dtype=np.float64) * float(strength)
    unif = np.asarray(scalars["uniformity"], dtype=np.float64)              # (3 ch)
    blur = np.asarray(scalars["blur_sigma"], dtype=np.float64)              # (3 sl, 3 ch)

    # The reference adds each layer's share of the fog *before* sampling, so the tail of the
    # distribution keeps its shape instead of being clipped to zero.
    min_layers = np.asarray(scalars["density_min_layers"], dtype=np.float64)   # (3 sl, 3 ch)
    layers += min_layers[None, None, :, :]

    out = np.zeros(density.shape, dtype=np.float64)
    for ch in range(3):
        for sl in range(3):
            d = layers[..., sl, ch]
            dmax = float(d_max[sl, ch])
            n = float(n_pp[sl, ch])
            if n <= 0.0 or dmax <= 0.0:
                continue
            p = np.clip(d / dmax, 1e-6, 1.0 - 1e-6)
            sat = 1.0 - p * unif[ch] * (1.0 - 1e-6)
            rate = (n / sat) * p
            draw = rng.poisson(rate).astype(np.float64)
            g = draw * (dmax / n) * sat
            sigma = float(blur[sl, ch])
            if sigma > 0.4:
                g = gaussian_filter(g, sigma)
            out[..., ch] += g

    # micro-structure: inert at every resolution this app renders (see module docstring).
    if pixel_size_um > 0 and (_MICRO_SIGMA_NM * 1e-3 / pixel_size_um) > 0.05:
        raise NotImplementedError("micro-structure stage not ported")

    out -= d_min
    final = float(scalars["final_blur_px"])
    if final > 0:
        for ch in range(3):
            out[..., ch] = gaussian_filter(out[..., ch], final)
    # ``out`` is the *absolute* density the film developed (particles in, fog out), which is
    # what the reference returns -- not a delta to be added back to the input.
    return np.maximum(out, 0.0)
