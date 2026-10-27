"""Single source of truth for film-grain parameters and synthesis (plan item P1).

Why this module exists
----------------------
Grain used to be implemented twice: a GLSL block in ``ui/canvas_viewport.py`` for the live
preview and a numpy block in ``SpektraEngine.export_image`` for CPU export.  Both had their
own copy of the octave weights, the layer-correlation mix and the density response, and the
two copies had drifted: the preview's noise streams carry a standard deviation of about 0.20
while the export's were renormalised to unit variance and then multiplied by 0.5, giving about
0.51.  The same slider position therefore produced grain about 2.5x stronger in the exported
file than on screen.

Everything that decides *how much* and *where* grain appears now lives here.  The GLSL block
consumes these numbers as uniforms; the CPU renderer calls :func:`apply_to_linear` directly.
Only the noise *synthesis* differs between the two (value noise on the GPU, Gaussian blur on
the CPU) -- and that difference is deliberately confined to spectral character, not amplitude,
because both render every octave at ``VALUE_NOISE_STD`` and share the mixing formula above.

Conventions
-----------
* Stock presets carry ``size_mm`` (crystal size), ``strength`` and ``layer_weights``.
* ``scale_px`` is the grain cell size **in pixels of the image being rendered**, derived from
  the film-plane pixel pitch.  It shrinks as the output resolution drops, which keeps the
  physical grain size constant relative to the frame.
* All amplitude maths is done on ``layer_grain``, the per-layer noise whose standard deviation
  is :func:`expected_layer_std`.  Every subsequent step (strength, density response,
  transmittance) is shared verbatim.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Octave structure
# ---------------------------------------------------------------------------
# Spatial frequencies of the three dye-cloud octaves, in grain cells.  The preview evaluates
# value noise at ``film_coord / scale * FREQ``; the CPU renderer blurs white noise to the
# matching feature size.  0.75 ~ 7 um micro-crystals, 0.45 ~ 11 um dye clouds, 0.22 ~ 21 um mottle.
OCTAVE_FREQ = (0.75, 0.45, 0.22)
# Per-octave contrast gains applied in the shader before weighting.
OCTAVE_GAIN = (2.0, 1.8, 1.6)

# Standard deviation of GLSL ``smooth_cloud_noise``: smoothstep-bilinear interpolation of a
# uniform [0,1) hash lattice.  Measured empirically over 10.5 M samples (see
# docs/grain-calibration.md); the theoretical uniform lattice value is 1/sqrt(12) = 0.2887,
# interpolation smooths it down to this constant.
VALUE_NOISE_STD = 0.144287

# ---------------------------------------------------------------------------
# Density response (negative granularity vs optical density)
# ---------------------------------------------------------------------------
# Fitted to the spektrafilm particle-model reference (plan item P0); see
# docs/grain-calibration.md and tools/grain_params.json.  The curve is a bell in optical
# density: it rises through the toe, peaks in the mid-densities and rolls off in the shoulder,
# then fades out at the film's Dmax.
#   D = -log10(clamp(lin, 1e-4, 1))         negative optical density
#   z = D - DENS_PEAK
#   w = DENS_W_SHOULDER (z > 0) else DENS_W_TOE
#   bell = DENS_FLOOR + (1 - DENS_FLOOR) * exp(-0.5 * (z / w)^2)
#   t    = clamp((D - (DENS_DMAX - DENS_W_SAT)) / DENS_W_SAT, 0, 1)
#   response = DENS_GAIN * bell * (1 - smoothstep(t))
# ``floor = 0`` is physical: at zero density no grains are developed, so granularity is zero.
DENS_GAIN = 0.72                 # peak amplitude; keeps P1's amplitude reference unchanged
DENS_FLOOR = 0.0
DENS_PEAK = 0.8382
DENS_W_TOE = 0.6149
DENS_W_SHOULDER = 1.4884
DENS_MIN_TRANSMITTANCE = 1e-4    # clamp before the log
# Film saturation.  The profiles declare density_max_curves = 2.2 and density_min ~ 0.04, so a
# stock cannot exceed D ~ 2.24.  The particle-model reference collapses there (once every crystal
# is developed the granularity vanishes) instead of decaying smoothly, and every sample above Dmax
# returns the *same* saturated field.  Fading the response out below Dmax keeps grain out of fully
# clipped highlights instead of extrapolating a Gaussian into a region that is really a step.
DENS_DMAX = 2.24
DENS_W_SAT = 0.20

# Strength pipeline: effective strength -> density modulation -> transmittance.
STRENGTH_GAIN = 0.35
SLIDER_REFERENCE = 0.40          # slider value at which strength == stock strength
STOCK_STRENGTH_REFERENCE = 0.010
LOG10_E = 2.302585092994046
CHROMA_MOTTLE_GAIN = 1.15        # coarse mottle emphasis on the per-layer stream

# Physical amplitude law: per-pixel density noise is inversely proportional to the pixel pitch
# (fewer crystals per pixel as sampling gets finer), so sigma is proportional to the long edge.
# The particle-model reference measures exactly this (5857 px -> 0.0297, 3000 px -> 0.0154,
# ratio 1.93 vs long-edge ratio 1.95); see docs/grain-calibration.md section 4.3.
REFERENCE_LONG_EDGE = 5857

# Inter-layer correlation target from the same reference (colour_grain = 0.5).
# P0 found it essentially flat across density (0.781 - 0.797), so no density term is invented.
TARGET_CHANNEL_CORRELATION = 0.79

# Minimum Gaussian sigma (px) for the finest octave.  Below ~0.35 px a blur is numerically
# indistinguishable from no blur at all, which is what used to turn small exports into white
# noise; the floor keeps the octave structure intact at any output size.
MIN_OCTAVE_SIGMA_PX = 0.35
# Gaussian sigma / feature size.  A value-noise cell of ``scale/freq`` pixels is approximated
# by a Gaussian of this fraction of the cell size.  Calibrated so the *radial autocorrelation\r\n# half-width* matches the particle-model reference (1.81 px at 100% output).  Do not tune this\r\n# against the PSD slope alone: that single number cannot tell `white noise plus a steep knee`\r\n# apart from `a smooth sum of wide Gaussians`, and tuning to it made the grain visibly coarser\r\n# than the reference (see docs/grain-calibration.md section 7).
SIGMA_PER_CELL = 0.40


def octave_weights(cloud_blur):
    """Per-octave mixing weights from the "染料云团柔化" slider.

    Low blur: sharp silver crystal edges bite through (fine octave dominates).
    High blur: organic dye cloud mottle dominates.
    """
    blur_t = min(1.2, max(0.1, float(cloud_blur) * 0.45))
    w_fine = max(0.15, 0.45 - 0.22 * blur_t)
    w_cloud = 0.50 + 0.20 * blur_t
    w_mottle = 0.28 + 0.12 * blur_t
    return w_fine, w_cloud, w_mottle


def octave_sigmas(scale_px):
    """Gaussian sigmas (px) approximating the three octaves, finest first."""
    sig = tuple(max(MIN_OCTAVE_SIGMA_PX, float(scale_px) / f * SIGMA_PER_CELL) for f in OCTAVE_FREQ)
    return sig


def _stream_stds(cloud_blur):
    """(raw_std, shared_std) -- the two independent noise streams' magnitudes."""
    w_fine, w_cloud, w_mottle = octave_weights(cloud_blur)
    g_fine, g_cloud, g_mottle = OCTAVE_GAIN
    raw_std = np.sqrt((w_cloud * g_cloud * VALUE_NOISE_STD) ** 2
                      + (w_mottle * g_mottle * VALUE_NOISE_STD) ** 2) * CHROMA_MOTTLE_GAIN
    shared_std = np.sqrt((w_fine * g_fine * VALUE_NOISE_STD) ** 2
                         + (w_cloud * g_cloud * VALUE_NOISE_STD) ** 2
                         + (w_mottle * g_mottle * VALUE_NOISE_STD) ** 2)
    return raw_std, shared_std


def expected_layer_std(cloud_blur, rho, layer_weights):
    """Per-channel std of ``layer_grain`` for a given stock, blur and correlation.

    Both renderers guarantee that every octave stream carries ``VALUE_NOISE_STD``, so this is
    the amplitude the *rendered* grain has, and it is what the preview has always produced.
    The CPU path must therefore match this value rather than renormalise it away: the point of
    P1 is to pull the export onto the preview's amplitude, not to move both.

    Models the two streams as independent:
      * ``raw_noise``    -- per-layer stream, built from the cloud + mottle octaves only
                            (deliberately excludes the fine octave, so chroma never speckles
                            at the pixel level);
      * ``shared_noise`` -- luminance stream across coupled layers, all three octaves.
    """
    raw_std, shared_std = _stream_stds(cloud_blur)
    lw = np.asarray(layer_weights, dtype=np.float64)
    rho = float(np.clip(rho, 0.0, 1.0))
    return np.sqrt((np.sqrt(1.0 - rho) * raw_std * lw) ** 2
                   + (np.sqrt(rho) * shared_std * float(lw.mean())) ** 2)


def predicted_channel_correlation(cloud_blur, rho, layer_weights):
    """Mean pairwise correlation between the three channels' grain for a given mix.

    The per-layer stream is independent per channel; only the shared stream correlates them.
    The mean is taken over all three channel pairs because a layer-weight spread makes the
    pairs unequal (R-G correlates more than R-B), and that mean is what the reference measures.
    """
    raw_std, shared_std = _stream_stds(cloud_blur)
    lw = np.asarray(layer_weights, dtype=np.float64)
    rho = float(np.clip(rho, 0.0, 1.0))
    lw_mean = float(lw.mean())
    variance = expected_layer_std(cloud_blur, rho, lw) ** 2
    cov = rho * (lw_mean * shared_std) ** 2
    vals = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        denom = float(np.sqrt(variance[i] * variance[j]))
        vals.append(cov / denom if denom > 1e-12 else 0.0)
    return float(np.mean(vals))


def rho_for_target_correlation(target, cloud_blur, layer_weights, grain_colour=1.0):
    """Solve for the mix ratio that yields ``target`` inter-channel correlation.

    Channel correlation is what we can measure and what P0 calibrated, so the implementation
    inverts the relationship instead of guessing ``rho`` directly: the observable then stays on
    target regardless of a stock's layer-weight spread.  ``predicted_channel_correlation`` is a
    monotone function of rho, so a bisection is enough.

    ``grain_colour`` (0 = monochrome, 1 = colour film) scales the correlation range: at 0 the
    layers are fully correlated by construction (pure luminance grain).
    """
    lw = np.asarray(layer_weights, dtype=np.float64)
    target = float(np.clip(target, 0.0, 0.999))
    colour = float(np.clip(grain_colour, 0.0, 1.0))
    goal = 1.0 + (target - 1.0) * colour
    if goal >= 0.999:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if predicted_channel_correlation(cloud_blur, mid, lw) < goal:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Per-stock density law (granularity against density)
# ---------------------------------------------------------------------------
# The P0 bell above is one curve fitted to one reference.  A real emulsion's granularity against
# density is a property of the emulsion, and fotufilm-engine states it per stock; the profiles
# carry those declarations, which is what this section consumes.  Source of truth:
#   fotufilm-engine/Sources/FotufilmCore/FilmStock.swift  (GrainDensityLaw, grainDensityModulation)
#   fotufilm-engine/Sources/FotufilmHalide/Stages/Grain.h  (the variance expressions)
#
# Every law returns a *variance* in arbitrary units; the modulation is the square root of its
# ratio to the stock's own reading density, so each law is normalised where its published
# granularity figure was read.  That normalisation is what lets `DENS_GAIN` stay the single
# amplitude constant: whatever the law, the response at the anchor density is exactly DENS_GAIN.
DENSITY_LAW_DYE_CLOUD = 0        # chromogenic negative: several sub-layers of speed/crystal size
DENSITY_LAW_SILVER = 1           # silver image: covered area, Selwyn at low D then steeper
DENSITY_LAW_SELWYN = 2           # explicit legacy law, sigma ∝ sqrt(D)
DENSITY_LAW_REVERSAL = 3         # dye reversal: rises toward the shoulder, saturating
DEFAULT_DENSITY_LAW = DENSITY_LAW_SELWYN

DEFAULT_FOG_DENSITY = 0.03       # FilmStock.defaultGrainFogDensity
DEFAULT_REVERSAL_PROFILE = (1.1, 3.0)   # FilmStock.defaultGrainReversalProfile
DEFAULT_DENSITY_MIN = 0.0
DEFAULT_DENSITY_MAX = DENS_DMAX  # the P0 calibration brief; profiles override it per stock

# Family default for a chromogenic negative that declares no profile of its own
# (FilmStockDefaults.grainDensityRecords): [amplitude, toe, decay, hump, humpDensity, humpWidth]
DEFAULT_DENSITY_RECORDS = (
    (20.0, 0.3373, 0.2629, 0.8187, 1.1811, 0.3212),
    (20.0, 0.3583, 0.3183, 1.6514, 1.4856, 0.3631),
    (20.0, 0.1069, 0.5457, 4.2644, 1.3800, 0.3446),
)

# The response is baked into a lookup table shared by both renderers: a 512-entry RGB ramp over
# the whole reachable density range.  ``D = -log10(clamp(lin, 1e-4, 1))`` cannot exceed 4, so the
# table covers the domain exactly and the shader's sample coordinate never needs clamping.
DENSITY_LUT_SIZE = 512
DENSITY_LUT_MAX = 4.0


def _flat(values):
    """Flatten ``(1.1, 3.0)`` / ``((1.1,), (3.0,))`` style specs into a flat list of floats."""
    out = []
    for value in values:
        if isinstance(value, (tuple, list)):
            out.extend(float(x) for x in value)
        else:
            out.append(float(value))
    return out


def _dye_cloud_variance(row, density):
    """Chromogenic negative: ``(1-e^(-D/toe)) * (1 + A e^(-D/decay) + H e^(-((D-hD)/hW)^2/2))``.

    The first factor is how much of the emulsion has developed; the bracket is the variance each
    unit of density carries -- high while the fast coarse sub-layer develops, decaying onto the
    fine one's floor of 1, with a second rise where the slow sub-layer's own coarse population
    comes in.  With the published coefficients this peaks about 1.5x the read density's figure a
    little above D-min, which is what every Kodak Vision3 sheet shows.
    """
    row = [float(x) for x in row]
    amplitude, toe, decay, hump, hump_density, hump_width = row
    toe = max(toe, 1e-4)
    decay = max(decay, 1e-4)
    hump_width = max(hump_width, 1e-4)
    offset = (density - hump_density) / hump_width
    return ((1.0 - np.exp(-density / toe))
            * (1.0 + amplitude * np.exp(-density / decay) + hump * np.exp(-0.5 * offset * offset)))


def _silver_variance(density):
    """Silver image: ``D * 10^(0.21004 D + 0.06114 D^2)`` -- Selwyn at low D, then steeper."""
    return density * np.power(10.0, 0.21004 * density + 0.06114 * density * density)


def _reversal_variance(profile, density):
    """Dye reversal: ``D^(2p) / (1 + (D/Ds)^(2p))`` -- rises, then saturates at the shoulder."""
    flat = _flat(profile) if isinstance(profile, (tuple, list)) else [float(profile)]
    exponent = max(float(flat[0]), 1e-4)
    shoulder = max(float(flat[1]) if len(flat) > 1 else 3.0, 1e-4)
    power = np.power(density / shoulder, 2.0 * exponent)
    return power / (1.0 + power)


def granularity_anchor_density(read_density, d_min, d_max):
    """The density the stock's published granularity figure was read at.

    Mirrors ``FilmStock.granularityAnchorDensity``: diffuse density 1.0 either above minimum
    density ("net", every negative sheet) or gross ("gross", Kodak's reversal sheets), clamped
    into the stock's own density range so a curve too short to reach the read density cannot
    anchor outside its scale.
    """
    d_min = float(d_min)
    d_max = float(d_max)
    span = d_max - d_min
    if span <= 0.0:
        return 0.5
    net = (1.0 - d_min) if read_density == "gross" else 1.0
    return float(min(max(net, 0.05 * span), 0.9 * span))


def correlation_for_luma(luma_correlation):
    """Channel correlation implied by a stock's ``grainLumaCorrelation``.

    The parameter is the share of the grain that is common luminance: 0 leaves the three records
    independent, 1 makes the grain pure luminance.  fotufilm pins it to 1 for every monochrome
    stock, which is the invariant that fixes the curve's direction -- the two endpoints are
    ``corr(0) = 0`` and ``corr(1) = 1``.  ``corr = 2c - c^2`` is the mixture model's correlation
    and lands on 0.75 at c = 0.5, which is what the P0 reference measured (0.79) with
    ``colour_grain = 0.5``; that reading is why the curve is normalised there rather than at 1.

    It is deliberately *not* ``1 - c^2``: that mirrors the curve, so a monochrome stock (c = 1)
    would land on 0 and get independent RGB speckle.
    """
    c = float(np.clip(luma_correlation, 0.0, 1.0))
    return c * (2.0 - c)


def _preset_value(preset, key, default):
    """Read a preset field, treating an explicit ``None`` as "not declared" -> default."""
    if not preset:
        return default
    value = preset.get(key)
    return default if value is None else value


def _freeze(value):
    """Recursively turn preset values into hashable structures, for the LUT cache key."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def _preset_signature(preset):
    """Hashable summary of every field the density law reads, for the LUT cache."""
    preset = preset or {}
    law = preset.get("density_law")
    law = DEFAULT_DENSITY_LAW if law is None else int(law)
    profile = _freeze(preset.get("density_profile"))
    if profile is None:
        profile = DEFAULT_DENSITY_RECORDS
    # three rows of six coefficients, as GrainDensityProfile.pad() guarantees
    profile = tuple(tuple(row) if len(row) == 6 else tuple(row) + (0.0, 1.0, 0.3)
                    for row in profile)
    return (
        law,
        profile,
        _freeze(preset.get("density_min")),
        _freeze(preset.get("density_max")),
        str(_preset_value(preset, "granularity_read_density", "net")),
        float(_preset_value(preset, "fog_density", DEFAULT_FOG_DENSITY)),
        _freeze(_preset_value(preset, "reversal_profile", DEFAULT_REVERSAL_PROFILE)),
    )


_LUT_CACHE = {}


def density_law_summary(preset):
    """Human-readable description of the law a preset selects (for logs and reports)."""
    law = int(_preset_value(preset, "density_law", DEFAULT_DENSITY_LAW))
    names = {
        DENSITY_LAW_DYE_CLOUD: "dyeCloud",
        DENSITY_LAW_SILVER: "silver",
        DENSITY_LAW_SELWYN: "selwyn",
        DENSITY_LAW_REVERSAL: "dyeCloudReversal",
    }
    return names.get(law, f"unknown({law})")


def _layer_value(spec, layer, default):
    """Pick one record's value out of a per-record spec (scalar, list, or list of tuples)."""
    if spec is None:
        return float(default)
    if isinstance(spec, (int, float)):
        return float(spec)
    if len(spec) > layer:
        value = spec[layer]
    elif len(spec) > 0:
        value = spec[0]
    else:
        return float(default)
    return float(value[0]) if isinstance(value, tuple) else float(value)


def bake_density_lut(preset):
    """Bake a preset's density law into a ``(DENSITY_LUT_SIZE, 3)`` response table.

    The table holds ``DENS_GAIN * sqrt(V(here)/V(anchor)) * saturation`` per record, evaluated on
    a uniform density ramp over ``[0, DENSITY_LUT_MAX]``.  Both renderers read this one table:
    the shader samples it as a texture (linear filtering between entries) and the CPU renderer
    interpolates it the same way, so a stock's law cannot be implemented twice or drift.

    Normalisation is the point.  Each law is divided by its own value at the stock's reading
    density, so ``response(anchor) == DENS_GAIN`` exactly for every stock and every law; only the
    *shape* against density changes.  That keeps ``DENS_GAIN`` the single amplitude calibration
    and means adopting a law cannot silently change how strong grain is at the mid-density.

    Returns ``None`` when there is nothing to bake (no preset, or a preset that declares no
    law at all), which selects the P0 bell.
    """
    if not preset or preset.get("density_law") is None:
        return None
    key = _preset_signature(preset)
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached

    law, profile, d_min, d_max, read, fog, reversal = key
    if profile is None:
        profile = DEFAULT_DENSITY_RECORDS
    grid = np.linspace(0.0, DENSITY_LUT_MAX, DENSITY_LUT_SIZE, dtype=np.float64)
    lut = np.zeros((DENSITY_LUT_SIZE, 3), dtype=np.float32)
    for layer in range(3):
        row = profile[min(layer, len(profile) - 1)]
        lo = _layer_value(d_min, layer, DEFAULT_DENSITY_MIN)
        hi = _layer_value(d_max, layer, DEFAULT_DENSITY_MAX)
        anchor = granularity_anchor_density(read, lo, hi) + fog
        here = np.maximum(grid, 0.0) + fog

        if law == DENSITY_LAW_DYE_CLOUD:
            ratio = _dye_cloud_variance(row, here) / max(
                float(_dye_cloud_variance(row, np.array([anchor]))[0]), 1e-6)
        elif law == DENSITY_LAW_SILVER:
            ratio = _silver_variance(here) / max(float(_silver_variance(np.array([anchor]))[0]), 1e-6)
        elif law == DENSITY_LAW_REVERSAL:
            ratio = _reversal_variance(reversal, here) / max(
                float(_reversal_variance(reversal, np.array([anchor]))[0]), 1e-20)
        else:
            # DENSITY_LAW_SELWYN (legacy sigma proportional to sqrt(D)), and the safe landing
            # spot for an unknown law: the simplest of the four, never wildly out of range.
            ratio = here / max(anchor, 1e-6)
        modulation = np.sqrt(np.maximum(ratio, 0.0))

        # Fade out just below the stock's own Dmax: past it the emulsion is fully developed and
        # carries no fluctuation (the particle model returns the same saturated field there).
        sat = np.clip((grid - (hi - DENS_W_SAT)) / DENS_W_SAT, 0.0, 1.0)
        roll = 1.0 - (3.0 * sat * sat - 2.0 * sat * sat * sat)
        lut[:, layer] = (DENS_GAIN * modulation * roll).astype(np.float32)

    _LUT_CACHE[key] = lut
    return lut


def _sample_density_lut(lut, density):
    """Linear interpolation into a baked table -- the CPU's copy of the shader's texture fetch."""
    table = np.asarray(lut, dtype=np.float32)
    span = DENSITY_LUT_SIZE - 1
    pos = np.clip(density, 0.0, DENSITY_LUT_MAX).astype(np.float32) * (span / DENSITY_LUT_MAX)
    lo = pos.astype(np.int32)
    np.clip(lo, 0, span - 1, out=lo)
    frac = (pos - lo).astype(np.float32)
    out = np.empty(density.shape, dtype=np.float32)
    for channel in range(3):
        column = table[:, channel]
        base = column[lo[..., channel]]
        next_ = column[lo[..., channel] + 1]
        out[..., channel] = base + (next_ - base) * frac[..., channel]
    return out


def density_response(linear_rgb, lut=None):
    """Granularity as a function of negative optical density (per channel).

    The renderer's ``linear_rgb`` is scene-linear exposure; it is read as the negative's
    transmittance, so ``D = -log10(lin)``.  Shape and constants are fitted to the particle-model
    reference (P0); mirrors the GLSL expression exactly, which takes the constants as uniforms.

    ``lut`` selects the per-stock density law baked by :func:`bake_density_lut`.  When it is
    ``None`` the P0 bell is used -- that is the path the two renderers fall back to, and the one
    the global calibration targets still describe.  Both paths return the *same* quantity: the
    density modulation in units where ``DENS_GAIN`` is the amplitude at the reference density,
    so swapping one for the other cannot change the overall grain level.
    """
    lin = np.clip(np.asarray(linear_rgb, dtype=np.float32), DENS_MIN_TRANSMITTANCE, 1.0)
    density = -np.log10(lin)
    z = density - DENS_PEAK
    w = np.where(z > 0.0, DENS_W_SHOULDER, DENS_W_TOE).astype(np.float32)
    bell = DENS_FLOOR + (1.0 - DENS_FLOOR) * np.exp(-0.5 * (z / w) ** 2)
    # fade to zero as the film saturates at its Dmax
    t = np.clip((density - (DENS_DMAX - DENS_W_SAT)) / DENS_W_SAT, 0.0, 1.0).astype(np.float32)
    roll = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    if lut is None:
        return (DENS_GAIN * bell * roll).astype(np.float32)
    return _sample_density_lut(lut, density)


def effective_strength(grain_slider, stock_strength):
    """User slider -> strength relative to the stock's own strength."""
    base = float(stock_strength) if float(stock_strength) > 0.0001 else STOCK_STRENGTH_REFERENCE
    return (float(grain_slider) / SLIDER_REFERENCE) * (base / STOCK_STRENGTH_REFERENCE)


# ---------------------------------------------------------------------------
# Derived specification
# ---------------------------------------------------------------------------
class GrainSpec:
    """Everything the two renderers need to agree on, computed in one place."""

    __slots__ = ("scale_px", "sigmas", "w_fine", "w_cloud", "w_mottle", "rho", "lw",
                 "lw_mean", "strength", "amplitude", "active", "density_lut",
                 "luma_correlation")

    def __init__(self, scale_px, sigmas, weights, rho, layer_weights, strength, amplitude, active,
                 density_lut=None, luma_correlation=None):
        self.scale_px = scale_px
        self.sigmas = sigmas
        self.w_fine, self.w_cloud, self.w_mottle = weights
        self.rho = rho
        self.lw = np.asarray(layer_weights, dtype=np.float32)
        self.lw_mean = float(self.lw.mean())
        self.strength = strength
        self.amplitude = amplitude
        self.active = active
        # Per-stock density law as a (DENSITY_LUT_SIZE, 3) response table, or None for the P0
        # bell.  Both renderers read this one table (texture on the GPU, interpolation on the CPU).
        self.density_lut = density_lut
        self.luma_correlation = luma_correlation

    def __repr__(self):
        law = "bell" if self.density_lut is None else "lut"
        return (f"GrainSpec(scale_px={self.scale_px:.4f}, sigmas={tuple(round(s, 3) for s in self.sigmas)}, "
                f"w=({self.w_fine:.3f},{self.w_cloud:.3f},{self.w_mottle:.3f}), rho={self.rho:.3f}, "
                f"strength={self.strength:.4f}, amplitude={self.amplitude:.4f}, density={law})")


def amplitude_for_long_edge(long_edge_px):
    """Physical amplitude law: per-pixel density noise scales with the sampling pitch.

    Fewer crystals fall in a pixel as sampling gets finer, so the per-pixel density noise of a
    given emulsion is inversely proportional to the pixel pitch, i.e. proportional to the long
    edge.  The particle-model reference measures exactly this (see
    docs/grain-calibration.md section 4.3).  At the calibration size the factor is 1.0, so the
    preview's amplitude is unchanged; only other output sizes move.
    """
    return max(1e-6, float(long_edge_px)) / float(REFERENCE_LONG_EDGE)


def derive(grain_slider, stock_strength, layer_weights, cloud_blur, rho,
           fmt_mm, long_edge_px, grain_size=1.0, stock_size_mm=0.005,
           grain_colour=1.0, target_correlation=None, density_preset=None):
    """Build a :class:`GrainSpec` for one render.

    ``long_edge_px`` must be the long edge **of the image being rendered**, so the physical
    grain size stays tied to the film plane rather than to the output resolution, and so the
    amplitude law in :func:`amplitude_for_long_edge` applies.

    ``rho`` is accepted for backwards compatibility but the value actually used is solved from
    ``target_correlation``, because inter-channel correlation is the quantity P0 calibrated.
    That target is :data:`TARGET_CHANNEL_CORRELATION` unless the stock declares a
    ``luma_correlation``, in which case :func:`correlation_for_luma` converts it.

    ``density_preset`` is the stock's ``crystal_grain`` dict (as returned by
    ``get_film_grain_preset``).  It selects the stock's density law, which is baked into the
    lookup table carried by the spec; without it the P0 bell is used.
    """
    from .darkroom_math import grain_scale_px

    scale_px = grain_scale_px(fmt_mm, long_edge_px, grain_size, stock_size_mm)
    weights = octave_weights(cloud_blur)
    luma = None
    if density_preset:
        declared = density_preset.get("luma_correlation")
        if declared is not None:
            luma = correlation_for_luma(declared)
    if target_correlation is not None:
        target = target_correlation
    elif luma is not None:
        target = luma
    else:
        target = TARGET_CHANNEL_CORRELATION
    solved_rho = rho_for_target_correlation(target, cloud_blur, layer_weights, grain_colour)
    spec = GrainSpec(
        scale_px=scale_px,
        sigmas=octave_sigmas(scale_px),
        weights=weights,
        rho=solved_rho,
        layer_weights=layer_weights,
        strength=effective_strength(grain_slider, stock_strength),
        amplitude=amplitude_for_long_edge(long_edge_px),
        active=float(grain_slider) > 0.01,
        density_lut=bake_density_lut(density_preset),
        luma_correlation=luma,
    )
    return spec


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CPU synthesis
# ---------------------------------------------------------------------------
_GAUSS_GAIN_CACHE = {}
_NUMBA_RNG = None
_NUMBA_RNG_TRIED = False


def _gauss_ksize(sigma):
    """Kernel size matching what cv2.GaussianBlur would pick for this sigma."""
    r = max(1, int(round(3.0 * float(sigma))))
    return 2 * r + 1, r


def gauss_white_gain(sigma):
    """Factor that restores unit variance after blurring unit-variance white noise.

    Blurring white noise shrinks its standard deviation by ``sqrt(sum(k^2))`` for the
    normalised separable kernel ``k``.  That factor is known analytically, so measuring
    ``arr.std()`` after every blur is unnecessary -- and on a 4K field that measurement alone
    cost ~72 ms per octave.
    """
    key = round(float(sigma), 6)
    cached = _GAUSS_GAIN_CACHE.get(key)
    if cached is not None:
        return cached
    _ksize, r = _gauss_ksize(sigma)
    xs = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (xs / float(sigma)) ** 2)
    g /= g.sum()
    gain = 1.0 / float((g ** 2).sum())
    _GAUSS_GAIN_CACHE[key] = gain
    return gain


def _randn_field(seed, height, width, channels):
    """Unit-variance Gaussian noise field.

    numpy's generator is the most expensive single step of the whole synthesis (~232 ms for a
    4K RGB field).  This numba kernel builds each sample from twelve uniforms, which has
    exactly unit variance by construction, needs no transcendental, and runs ~5x faster.  It
    also lets us drop the per-octave ``std()`` reduction (see :func:`gauss_white_gain`).
    """
    global _NUMBA_RNG, _NUMBA_RNG_TRIED
    if not _NUMBA_RNG_TRIED:
        _NUMBA_RNG_TRIED = True
        try:
            from numba import njit, prange

            @njit(cache=True, parallel=True, fastmath=True)
            def _kernel(seed, h, w, ch):
                out = np.empty((h, w, ch), dtype=np.float32)
                for y in prange(h):
                    for x in range(w):
                        # per-pixel state, so parallel workers never share a stream
                        state = (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
                                 ^ (np.uint64(y) * np.uint64(0xC2B2AE3D27D4EB4F))
                                 ^ (np.uint64(x) * np.uint64(0x165667B19E3779F9)))
                        for c in range(ch):
                            s = np.float32(0.0)
                            for _ in range(12):
                                state = (state * np.uint64(6364136223846793005)
                                         + np.uint64(1442695040888963407))
                                s += np.float32((state >> np.uint64(40))
                                                & np.uint64(0xFFFFFF)) * np.float32(1.0 / 16777216.0)
                            out[y, x, c] = s - np.float32(6.0)
                return out

            _NUMBA_RNG = _kernel
        except Exception:
            _NUMBA_RNG = None
    if _NUMBA_RNG is None:
        # fallback: numpy generator rescaled to unit variance
        n = np.random.default_rng(seed).standard_normal((height, width, channels)).astype(np.float32)
        sd = float(n.std())
        return (n / sd).astype(np.float32) if sd > 1e-9 else n
    return _NUMBA_RNG(int(seed) & 0x7FFFFFFF, height, width, channels)


def _gaussian_blur(x, sigma):
    import cv2
    ksize, _r = _gauss_ksize(sigma)
    return cv2.GaussianBlur(x, (ksize, ksize), float(sigma), borderType=cv2.BORDER_REFLECT)


def synthesize_layer_grain(spec, height, width, rng):
    """Per-channel ``layer_grain`` with the octave structure of the preview.

    Returns ``(h, w, 3)`` float32 with std matching the GLSL ``layer_grain`` by construction
    of :func:`expected_layer_std`.

    The CPU approximates the shader's value-noise octaves with Gaussian-blurred white noise.
    Every octave is rescaled to ``VALUE_NOISE_STD`` -- the constant the shader's value noise
    carries -- using the analytic blur factor, so the two renderers stay on the same amplitude
    model while their spectral character may differ.
    """
    s_fine, s_cloud, s_mottle = spec.sigmas
    g_fine, g_cloud, g_mottle = OCTAVE_GAIN
    # one seed per octave and stream, so the octaves and the two streams are independent
    base_seed = int(rng.integers(0, 2 ** 31 - 1))
    seeds = [base_seed + i * 7919 for i in range(5)]

    def octave(seed, sigma, channels):
        n = _randn_field(seed, height, width, channels)
        n = _gaussian_blur(n, sigma)
        n = np.asarray(n, dtype=np.float32).reshape(height, width, channels)
        return (n * np.float32(gauss_white_gain(sigma) * VALUE_NOISE_STD)).astype(np.float32)

    # Per-layer stream: cloud + mottle octaves only (no pixel-level chroma speckle).
    rgb_cloud = octave(seeds[0], s_cloud, 3)
    rgb_mottle = octave(seeds[1], s_mottle, 3)
    raw_noise = (rgb_cloud * (spec.w_cloud * g_cloud)
                 + rgb_mottle * (spec.w_mottle * g_mottle)) * CHROMA_MOTTLE_GAIN

    # Shared luminance stream across coupled emulsion layers: all three octaves.
    shared_noise = (octave(seeds[2], s_fine, 1) * (spec.w_fine * g_fine)
                    + octave(seeds[3], s_cloud, 1) * (spec.w_cloud * g_cloud)
                    + octave(seeds[4], s_mottle, 1) * (spec.w_mottle * g_mottle))

    layer = (np.sqrt(1.0 - spec.rho) * raw_noise * spec.lw[None, None, :]
             + np.sqrt(spec.rho) * shared_noise * spec.lw_mean)
    return (layer * spec.amplitude).astype(np.float32)

def apply_to_linear(linear_rgb, spec, rng, chunk_rows=512):
    """Apply pre-LUT transmittance grain to a scene-linear RGB image, in place-safe fashion.

    ``T_out = T_in * 10^(-dD)`` with ``dD = layer_grain * (strength * STRENGTH_GAIN) * dens_mod``.
    Chunked so a 64 MP frame does not need a second full-size noise field at once.
    """
    if not spec.active:
        return linear_rgb
    h, w = linear_rgb.shape[:2]
    out = np.empty_like(linear_rgb)
    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        block = linear_rgb[y0:y1]
        layer = synthesize_layer_grain(spec, y1 - y0, w, rng)
        response = density_response(block, spec.density_lut)
        dD = layer * (spec.strength * STRENGTH_GAIN) * response
        out[y0:y1] = np.maximum(0.0, block * np.exp(-dD * LOG10_E))
    return out
