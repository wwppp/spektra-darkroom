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
import threading
from scipy import ndimage

from .shaper import shaper_forward, shaper_inverse

# ---------------------------------------------------------------------------
# Octave structure
# ---------------------------------------------------------------------------
# Spatial frequencies of the three dye-cloud octaves, in grain cells.  The preview evaluates
# value noise at ``film_coord / scale * FREQ``; the CPU renderer blurs white noise to the
# matching feature size.  0.75 ~ 7 um micro-crystals, 0.45 ~ 11 um dye clouds, 0.22 ~ 21 um mottle.
OCTAVE_FREQ = (0.75, 0.45, 0.22)
# Per-octave contrast gains applied in the shader before weighting.
OCTAVE_GAIN = (2.0, 1.8, 1.6)

# Standard deviation of one unit of GLSL ``smooth_cloud_noise`` -- i.e. the value the shader
# multiplies by ``(noise - 0.5) * 2`` before applying the octave gain.
#
# 0.2144 is analytic, not a fit: value noise is ``sum w_i u_i`` with ``u_i ~ U[0,1]`` and
# smoothstep weights, so ``std = (1/sqrt(12)) * sqrt(E[sum w_i^2])``.  For smoothstep,
# ``E[(1-s)^2 + s^2] = 1 - 2E[s] + 2E[s^2] = 0.7428`` with ``s = 3f^2-2f^3``, hence
# ``E[sum w^2] = 0.7428^2 = 0.5518`` and ``std = 0.2887 * 0.7428 = 0.2144``.
# Measured in numpy over 512..1024 px fields: 0.2141 .. 0.2161, matching.
VALUE_NOISE_REALISED_STD = 0.2144

# Ratio the *preview* must scale its noise by so its realised std matches the CPU's.
#
# VALUE_NOISE_STD below is 0.144287, which is exactly half of 1/sqrt(12).  The CPU multiplies
# its Gaussian-blurred white noise (normalised to unit variance) by that number, so its octave
# std is 0.144287 * gain.  The shader's value noise really carries 0.2144 per unit, so at the
# same nominal sigma the preview ran 1.46x louder than the export -- at *every* grain size and
# every resolution, which is what the amplitude sweep (scratch/grain_amp_matrix.py) measured.
#
# Applied in the shader rather than by raising the CPU constant: the export carries the P0
# calibration and must not drift.  Calibrated against the measured 1.4606 ratio.
PREVIEW_NOISE_FIX = 0.685

# Legacy name, kept because the shader-side octave scaling and the calibration docs refer to it.
# It is the *per-unit* std the CPU stream is normalised to, not the value noise's real std.
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

# Side length of the negative-density cube (C/M/Y per node) baked by the engine on the same
# shaper lattice as the 3D LUT.  float32 -> 33^3 * 3 * 4 = 421 KB per stock, trivial next to the
# LUT itself (33^3 * 3 * 4 = 421 KB as well); the lattice is coarser than the response table
# because density is a smooth, monotonic function of exposure.
FILM_DENSITY_LUT_SIZE = 33

# ---------------------------------------------------------------------------
# Multi-layer structure (P2)
# ---------------------------------------------------------------------------
# A chromogenic emulsion coats three sub-layers of decreasing speed and crystal size.  The fast
# coarse one develops first and saturates early, so its granularity lives in the toe; the slow
# fine one is still developing in the shoulder.  The three spatial octaves are exactly those
# three sub-layers, so they must not share one density response: the mottle octave has to follow
# the coarse layer and the fine octave the slow one.
#
# What is *not* done here matters as much.  The per-stock granularity-vs-density law (the only
# per-stock *measured* data we have) is not re-fitted or split by guesswork: the per-octave
# responses are normalised against the noise each octave really carries, so their signal-weighted
# RMS equals the declared law at every density.  The measured curve therefore survives untouched;
# only the split across spatial scales moves -- shadows turn coarse and mottled, highlights turn
# fine, and the slider still controls the total.
#
# Octave order is (fine, cloud, mottle) throughout this module; sub-layer order is the profile's
# (coarse -> fine), so the mapping is reversed.
OCTAVE_TO_LAYER = (2, 1, 0)
# Where each sub-layer sits on the density axis, as a fraction of the stock's total Dmax, when
# the profile's layer curves cannot separate them (they are pure scalings of the total: 38 of the
# 61 stocks that carry layer curves).  This is a *structural* constant -- three sub-layers of
# decreasing speed -- not a per-stock calibration; the data overrides it wherever it can.
# Fast/coarse: a third of the way up the curve, saturating early.  Slow/fine: near the shoulder.
LAYER_CENTER_FALLBACK = (0.72, 0.40, 0.15)
LAYER_CENTER_MIN = 0.03
LAYER_CENTER_MAX = 0.94
# Response half-width, as a fraction of the layer's own 20..80 % development span, with a floor so
# a nearly step-like layer curve does not produce a delta.
LAYER_WIDTH_PER_SPAN = 0.45
LAYER_WIDTH_PER_CENTER = 0.45      # used with LAYER_CENTER_FALLBACK
LAYER_WIDTH_MIN = 0.13
# A sub-layer keeps developing below its half-development density, so its activity falls off much
# more slowly on the toe side than above the peak (where it is saturating).  Without this
# asymmetry a narrow coarse peak leaves nothing to carry the deep shadows and the model would have
# to amplify it back up; with it the coarse octave simply owns the toe, which is the physics.
LAYER_LEFT_WIDTH_RATIO = 2.2
# Two sub-layers closer than this on the density axis carry no distinguishable structure; the
# fallback split is used instead of pretending three identical curves are three layers.
LAYER_SEPARATION_MIN = 0.05
# The redistribution gain needs no clamp: it is |signal| / |signal * shape|, and every shape is at
# most 1, so the per-octave response can never exceed |signal| / signal_o times the law -- about
# 2.2x for the widest mix this project uses.  Above the last sub-layer's half-development density
# the split is frozen (see bake_layer_lut) so that bound actually holds instead of decaying.


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


def _lattice_index(density):
    """Density -> (lower entry index, fraction) for a linear fetch of a baked table.

    Shared by both tables so the CPU can never interpolate differently from the GPU's
    ``GL_LINEAR`` fetch along the density axis.
    """
    span = DENSITY_LUT_SIZE - 1
    pos = np.clip(density, 0.0, DENSITY_LUT_MAX).astype(np.float32) * (span / DENSITY_LUT_MAX)
    lo = pos.astype(np.int32)
    np.clip(lo, 0, span - 1, out=lo)
    return lo, (pos - lo).astype(np.float32)


def _sample_density_lut(lut, density):
    """Linear interpolation into a baked table -- the CPU's copy of the shader's texture fetch."""
    table = np.asarray(lut, dtype=np.float32)
    lo, frac = _lattice_index(density)
    out = np.empty(density.shape, dtype=np.float32)
    for channel in range(3):
        column = table[:, channel]
        base = column[lo[..., channel]]
        next_ = column[lo[..., channel] + 1]
        out[..., channel] = base + (next_ - base) * frac[..., channel]
    return out


_BELL_LUT = None


def bake_bell_lut():
    """The P0 bell as a table, in the same units as ``bake_density_lut``.

    Only used to build the layered table for callers with no declared law (no film selected,
    thumbnails, offline probes): the layered path needs a total law to redistribute, and it must
    be the same bell both renderers already fall back to.
    """
    global _BELL_LUT
    if _BELL_LUT is not None:
        return _BELL_LUT
    grid = np.linspace(0.0, DENSITY_LUT_MAX, DENSITY_LUT_SIZE, dtype=np.float64)
    z = grid - DENS_PEAK
    w = np.where(z > 0.0, DENS_W_SHOULDER, DENS_W_TOE)
    bell = DENS_FLOOR + (1.0 - DENS_FLOOR) * np.exp(-0.5 * (z / w) ** 2)
    t = np.clip((grid - (DENS_DMAX - DENS_W_SAT)) / DENS_W_SAT, 0.0, 1.0)
    roll = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    column = (DENS_GAIN * bell * roll).astype(np.float32)
    _BELL_LUT = np.repeat(column[:, None], 3, axis=1)
    return _BELL_LUT


def _layer_geometry(preset):
    """Per record, per octave ``(center, width)`` in units of the stock's total density.

    Read from the profile's own sub-layer curves (``layer_structure``, built by
    ``app_core.get_film_grain_preset``): for each sub-layer, where it is half developed and how
    wide its 20..80 % development span is, both divided by the stock's total net Dmax.  A layer
    that halves early is the fast coarse one, which is exactly where the mottle octave belongs.

    Degenerate profiles -- layer curves that are pure scalings of the total, so every layer halves
    at the same density -- fall back to :data:`LAYER_CENTER_FALLBACK`.  That is a structural
    statement (three sub-layers of decreasing speed), not a per-stock measurement, and the
    fallback is applied per record so a stock that separates one record keeps the measured
    geometry for it.
    """
    preset = preset or {}
    structure = preset.get("layer_structure")
    geometry = np.zeros((3, 3, 2), dtype=np.float64)
    for record in range(3):
        rows = structure[record] if structure and len(structure) > record else None
        centers, widths = [], []
        for octave in range(3):
            layer = OCTAVE_TO_LAYER[octave]
            row = rows[layer] if rows and len(rows) > layer else None
            if row and float(row[0]) > 1e-4 and float(row[2]) > 1e-4:
                centers.append(float(row[1]))
                widths.append(max(LAYER_WIDTH_MIN, LAYER_WIDTH_PER_SPAN * float(row[2])))
            else:
                centers = None
                break
        if not centers or (max(centers) - min(centers)) < LAYER_SEPARATION_MIN:
            centers = [float(c) for c in LAYER_CENTER_FALLBACK]
            widths = [max(LAYER_WIDTH_MIN, LAYER_WIDTH_PER_CENTER * c) for c in centers]
        else:
            # Which sub-layer is coarsest is structural (crystal size), not a matter of array
            # order: the fast coarse layer is always the one that half-develops first.  A profile
            # whose three half-development densities come out non-monotonic (10 of 61 do, by up to
            # 0.1 D) is therefore assigned by rank -- the three measured positions are kept, only
            # their octave assignment is corrected.  Nothing is invented.
            rank = np.argsort(centers)
            centers = [centers[i] for i in rank][::-1]      # octave order: fine > cloud > mottle
            widths = [widths[i] for i in rank][::-1]
        geometry[record, :, 0] = np.clip(centers, LAYER_CENTER_MIN, LAYER_CENTER_MAX)
        geometry[record, :, 1] = widths
    return geometry


def _layer_share_weights(preset):
    """Per-octave amplitude from ``crystal_grain.population.sublayerShares``.

    The shares say how much of the coating each sub-layer carries, so a layer with a small share
    contributes a smaller slice of the granularity.  Every stock currently declares the same
    ``[0.25, 0.25, 0.25, 0.25]``, which makes this a no-op today -- it is wired so a per-stock
    population (the field exists in the profiles and nothing read it before P2) is honoured the
    moment someone fills it in.  ``radiusSpan`` fixes which sub-layer is coarsest, which is what
    :data:`OCTAVE_TO_LAYER` encodes; it is not a per-stock measurement, so it does not enter the
    amplitude.
    """
    population = (preset or {}).get("population") or {}
    shares = population.get("sublayerShares")
    if not shares or len(shares) < 3:
        return np.ones(3, dtype=np.float64)
    values = np.asarray([float(x) for x in shares[:3]], dtype=np.float64)
    peak = float(values.max())
    if peak <= 1e-9:
        return np.ones(3, dtype=np.float64)
    values = values / peak
    return np.clip(np.asarray([values[OCTAVE_TO_LAYER[o]] for o in range(3)]), 0.05, 1.0)


def _octave_signal(octave_weights, rho, layer_weights):
    """How much noise each octave actually carries, per record: ``(3 records, 3 octaves)``.

    The split must never hand a density band to an octave that has no signal there.  The fine
    octave feeds the shared luminance stream only, so on a stock whose grain is deliberately
    independent per channel (``luma_correlation = 0`` -> ``rho = 0``, e.g. Harman Phoenix) it
    carries *nothing*; redistributing the shoulder onto it emptied the highlights.  This is the
    per-octave std of the synthesised streams, so ``shape * signal`` is what actually lands in
    the output.
    """
    w = np.asarray(octave_weights, dtype=np.float64)
    g = np.asarray(OCTAVE_GAIN, dtype=np.float64)
    rho = float(np.clip(rho, 0.0, 1.0))
    lw = np.asarray(layer_weights, dtype=np.float64)
    lw_mean = float(lw.mean())
    shared = np.sqrt(rho) * lw_mean
    chroma = np.sqrt(1.0 - rho) * CHROMA_MOTTLE_GAIN
    signal = np.empty((3, 3), dtype=np.float64)
    for record in range(3):
        chroma_var = (chroma * lw[record]) ** 2 + shared ** 2
        signal[record, 0] = w[0] * g[0] * shared                 # fine: shared stream only
        signal[record, 1] = w[1] * g[1] * np.sqrt(chroma_var)
        signal[record, 2] = w[2] * g[2] * np.sqrt(chroma_var)
    return signal


_LAYER_LUT_CACHE = {}


def bake_layer_lut(preset, octave_weights=None, rho=None, layer_weights=None):
    """Bake the per-octave response table: ``(3 records, DENSITY_LUT_SIZE, 3 octaves)``.

    ``layer_lut[record, i, octave] = law_record(D_i) * redistribution``.  The redistribution moves
    the declared law's granularity between the three coated sub-layers; it never changes the
    total, because:

        ``sqrt(sum_o (signal_o * response_o)^2) == sqrt(sum_o signal_o^2) * law``

    at every density, where ``signal_o`` is the per-octave std the synthesiser really produces
    (:func:`_octave_signal`).  Preserving the *signal-weighted* total rather than the raw
    weights is what makes this hold for a stock whose grain is per-channel independent
    (``rho = 0``): its fine octave carries no signal, so the shoulder cannot be dumped on it.
    ``DENS_GAIN`` therefore stays the single amplitude knob and the measured granularity-vs-density
    law -- the only per-stock measured data we have -- survives untouched.

    ``octave_weights`` (the slider mix), ``rho`` and ``layer_weights`` all enter that
    normalisation, so the table is baked per blur setting and per stock; it is 18 KB.
    """
    weights = tuple(float(x) for x in ((0.45, 0.50, 0.28) if octave_weights is None
                                       else octave_weights))
    rho = 0.79 if rho is None else float(rho)
    layer_weights = tuple(float(x) for x in (layer_weights if layer_weights is not None
                                             else (0.70, 1.0, 1.35)))
    structure = (preset or {}).get("layer_structure")
    key = (_preset_signature(preset) if preset else None,
           _freeze(structure) if structure else None,
           _freeze((preset or {}).get("population")),
           weights, round(rho, 6), layer_weights)
    cached = _LAYER_LUT_CACHE.get(key)
    if cached is not None:
        return cached

    base = bake_density_lut(preset) if preset else None
    if base is None:
        base = bake_bell_lut()
    geometry = _layer_geometry(preset)
    shares = _layer_share_weights(preset)
    signal = _octave_signal(weights, rho, layer_weights)
    grid = np.linspace(0.0, DENSITY_LUT_MAX, DENSITY_LUT_SIZE, dtype=np.float64)
    d_max = (preset or {}).get("density_max")

    out = np.zeros((3, DENSITY_LUT_SIZE, 3), dtype=np.float32)
    for record in range(3):
        ceiling = max(_layer_value(d_max, record, DEFAULT_DENSITY_MAX), 0.05)
        x = np.clip(grid / ceiling, 0.0, 1.5)
        # Freeze the split above the last sub-layer's peak: past it every layer is saturating, so
        # where the granularity lives stops moving, while the *total* keeps following the law's
        # Dmax fade.  Without the freeze the basis collapses to zero there and the normalisation
        # would have to amplify a vanishing number.
        x_shape = np.minimum(x, float(geometry[record, :, 0].max()))
        shape = np.empty((3, grid.size), dtype=np.float64)
        for octave in range(3):
            center, width = geometry[record, octave]
            side = np.where(x_shape < center, width * LAYER_LEFT_WIDTH_RATIO, width)
            shape[octave] = shares[octave] * np.exp(-0.5 * ((x_shape - center) / side) ** 2)

        sig = signal[record]
        total = float(np.sqrt((sig ** 2).sum()))
        carried = np.sqrt(((sig[:, None] * shape) ** 2).sum(axis=0))
        # r_o = law * |signal| * shape_o / |signal * shape|: the signal-weighted share, which is
        # bounded by |signal| / max(signal) * law and needs no clamp.
        gain = np.where(carried > 1e-9, total / np.maximum(carried, 1e-9), 0.0)
        for octave in range(3):
            out[record, :, octave] = (base[:, record] * shape[octave] * gain).astype(np.float32)

    _LAYER_LUT_CACHE[key] = out
    return out


def _sample_layer_lut(lut, density, octave):
    """One octave's response (per record) at the given density -- the shader's fetch, in numpy.

    Only used by the slow path (no numba) and by the checks; :func:`layer_response_dd` fuses the
    same arithmetic into one compiled pass, because nine gathers per pixel cost ~3 s on a 24 MP
    frame while the fused kernel is bandwidth-bound.
    """
    table = np.asarray(lut, dtype=np.float32)[:, :, int(octave)]
    lo, frac = _lattice_index(density)
    out = np.empty(np.shape(density), dtype=np.float32)
    for record in range(3):
        column = table[record]
        base = column[lo[..., record]]
        next_ = column[lo[..., record] + 1]
        out[..., record] = base + (next_ - base) * frac[..., record]
    return out


_RESPONSE_KERNEL = None
_RESPONSE_KERNEL_TRIED = False
_RESPONSE_KERNEL_LOCK = threading.Lock()


def _response_kernel():
    """Compiled (density, 3 tables, 3 octave streams) -> dD, or None without numba."""
    global _RESPONSE_KERNEL, _RESPONSE_KERNEL_TRIED
    if _RESPONSE_KERNEL_TRIED:
        return _RESPONSE_KERNEL
    with _RESPONSE_KERNEL_LOCK:
        if _RESPONSE_KERNEL_TRIED:
            return _RESPONSE_KERNEL
        try:
            from numba import njit, prange

            span = DENSITY_LUT_SIZE - 1
            scale = span / DENSITY_LUT_MAX

            @njit(cache=True, parallel=True, fastmath=True)
            def kernel(density, table, fine, cloud, mottle, out):
                h = density.shape[0]
                w = density.shape[1]
                for y in prange(h):
                    for x in range(w):
                        for c in range(3):
                            pos = density[y, x, c] * scale
                            if pos < 0.0:
                                pos = 0.0
                            elif pos > span:
                                pos = span
                            i = int(pos)
                            if i > span - 1:
                                i = span - 1
                            f = pos - i
                            total = 0.0
                            acc = fine[y, x, c]
                            b0 = table[c, i, 0]
                            b1 = table[c, i + 1, 0]
                            total += acc * (b0 + (b1 - b0) * f)
                            acc = cloud[y, x, c]
                            b0 = table[c, i, 1]
                            b1 = table[c, i + 1, 1]
                            total += acc * (b0 + (b1 - b0) * f)
                            acc = mottle[y, x, c]
                            b0 = table[c, i, 2]
                            b1 = table[c, i + 1, 2]
                            total += acc * (b0 + (b1 - b0) * f)
                            out[y, x, c] = total

            _RESPONSE_KERNEL = kernel
        except Exception:
            _RESPONSE_KERNEL = None
        _RESPONSE_KERNEL_TRIED = True
    return _RESPONSE_KERNEL


def layer_response_dd(layer_lut, density, fine, cloud, mottle):
    """``sum_o stream_o * response_o(density)`` in one pass (the layered dD, before gain)."""
    table = np.ascontiguousarray(layer_lut, dtype=np.float32)
    kernel = _response_kernel()
    if kernel is not None:
        out = np.empty(density.shape, dtype=np.float32)
        kernel(np.ascontiguousarray(density, dtype=np.float32), table,
               np.ascontiguousarray(fine, dtype=np.float32),
               np.ascontiguousarray(cloud, dtype=np.float32),
               np.ascontiguousarray(mottle, dtype=np.float32), out)
        return out
    dD = fine * _sample_layer_lut(table, density, 0)
    dD += cloud * _sample_layer_lut(table, density, 1)
    dD += mottle * _sample_layer_lut(table, density, 2)
    return dD


def density_from_linear(linear_rgb, grid):
    """Look up the negative's net density for a scene-linear RGB pixel.

    ``grid`` is the density cube baked by ``app_core`` on the same shaper lattice as the 3D
    LUT (``FILM_DENSITY_LUT_SIZE`` cubed, C/M/Y per node).  Both renderers read it: the shader
    as a 3D texture, this function as trilinear interpolation over the same lattice, so the
    two cannot disagree about *what density a pixel has*.

    Axis order matters.  The cube is indexed ``[b][g][r]`` -- it comes from ``develop()``'
    ``(N, N*N, 3)`` image whose last axis is RGB, reshaped cube-last -- which is also how
    ``glTexImage3D`` lays the data out (last axis = fastest = x).  A pixel's ``(r, g, b)``
    therefore maps to coordinates ``(r, g, b)`` in x/y/z order, and getting this backwards is
    invisible on a neutral ramp while scrambling every colour.

    Returns ``None`` when no cube was supplied, and the caller falls back to the historical
    ``D = -log10(clamp(lin))`` reading.
    """
    if grid is None:
        return None
    lin = np.asarray(linear_rgb, dtype=np.float32)
    n = int(grid.shape[0])
    if n < 2:
        return None
    # linear -> shaper coordinate, then into lattice index space (same mapping as the shader).
    pos = shaper_forward(lin) * (n - 1)
    coords = np.stack([pos[..., 2], pos[..., 1], pos[..., 0]], axis=0).astype(np.float32)
    out = np.empty(lin.shape, dtype=np.float32)
    for channel in range(3):
        # order=1 is trilinear, mode="nearest" is the GPU's CLAMP_TO_EDGE.  map_coordinates is
        # ~3x faster than eight fancy-index gathers, which matters: this runs per chunk per
        # export, and a 12 MP frame has ~9 M samples per channel.
        out[..., channel] = ndimage.map_coordinates(
            np.ascontiguousarray(grid[..., channel]), coords,
            order=1, mode="nearest", prefilter=False)
    return out


def density_for_linear(linear_rgb, film_density=None):
    """Negative net density for a scene-linear block: the cube when there is one, else -log10.

    The legacy reading treats the scene-linear value as the negative's transmittance; it is only
    for callers with no LUT context (thumbnails, offline probes) and is what zeroed the grain
    below -3 EV before P1.  Both the layered and the flat path read density through here so they
    cannot disagree about *what density a pixel has*.
    """
    density = density_from_linear(linear_rgb, film_density)
    if density is not None:
        return density
    lin = np.clip(np.asarray(linear_rgb, dtype=np.float32), DENS_MIN_TRANSMITTANCE, 1.0)
    return (-np.log10(lin)).astype(np.float32)


def density_response(linear_rgb, lut=None, film_density=None):
    """Granularity as a function of negative optical density (per channel).

    ``film_density`` (baked density cube) is the physically right input: the negative's own net
    density at that pixel, taken from the same film pipeline that produced the LUT.  When it is
    absent the legacy reading ``D = -log10(lin)`` is used, which treats scene-linear exposure as
    the negative's transmittance -- that is what zeroed the grain below -3 EV (see
    docs/push-grain-plan.md section 2.1) and it survives only as a fallback for callers with no
    LUT context.

    ``lut`` selects the per-stock density law baked by :func:`bake_density_lut`.  When it is
    ``None`` the P0 bell is used -- that is the path the two renderers fall back to, and the one
    the global calibration targets still describe.  Both paths return the *same* quantity: the
    density modulation in units where ``DENS_GAIN`` is the amplitude at the reference density,
    so swapping one for the other cannot change the overall grain level.

    This is the *total* response (all octaves together).  The layered path does not call it: it
    fetches per-octave responses from :func:`bake_layer_lut`, which sums to this same curve.
    """
    density = density_for_linear(linear_rgb, film_density)
    if lut is not None:
        return _sample_density_lut(lut, density)
    z = density - DENS_PEAK
    w = np.where(z > 0.0, DENS_W_SHOULDER, DENS_W_TOE).astype(np.float32)
    bell = DENS_FLOOR + (1.0 - DENS_FLOOR) * np.exp(-0.5 * (z / w) ** 2)
    # fade to zero as the film saturates at its Dmax
    t = np.clip((density - (DENS_DMAX - DENS_W_SAT)) / DENS_W_SAT, 0.0, 1.0).astype(np.float32)
    roll = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    return (DENS_GAIN * bell * roll).astype(np.float32)


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
                 "lw_mean", "strength", "amplitude", "active", "density_lut", "layer_lut",
                 "luma_correlation", "film_density")

    def __init__(self, scale_px, sigmas, weights, rho, layer_weights, strength, amplitude, active,
                 density_lut=None, luma_correlation=None, film_density=None, layer_lut=None):
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
        # Per-octave response table (3 records, LUT size, 3 octaves), or None to keep the single
        # total response.  Built by bake_layer_lut() from the profile's sub-layer geometry (P2).
        self.layer_lut = layer_lut
        # Negative net density on the LUT's shaper lattice (N, N, N, 3), or None to keep the
        # historical -log10(lin) reading.  This is the input the density law is a function of.
        self.film_density = film_density

    def __repr__(self):
        law = "bell" if self.density_lut is None else "lut"
        dens = "linear" if self.film_density is None else "film"
        layers = "layered" if self.layer_lut is not None else "flat"
        return (f"GrainSpec(scale_px={self.scale_px:.4f}, sigmas={tuple(round(s, 3) for s in self.sigmas)}, "
                f"w=({self.w_fine:.3f},{self.w_cloud:.3f},{self.w_mottle:.3f}), rho={self.rho:.3f}, "
                f"strength={self.strength:.4f}, amplitude={self.amplitude:.4f}, density={law}, "
                f"domain={dens}, response={layers})")


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
           grain_colour=1.0, target_correlation=None, density_preset=None,
           film_density=None):
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

    ``film_density`` is the negative-density cube for the film+paper+DIR combination being
    rendered.  With it the law is evaluated at the pixel's real density; without it the legacy
    ``-log10(lin)`` reading is used.
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
        film_density=film_density,
        # The octave weights are part of the layered table's normalisation, so it is baked per
        # blur setting; it is 6 KB and cached by the slider tuple.
        layer_lut=bake_layer_lut(density_preset, weights, rho=solved_rho,
                                 layer_weights=layer_weights),
    )
    return spec


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CPU synthesis
# ---------------------------------------------------------------------------
_GAUSS_GAIN_CACHE = {}
_NUMBA_RNG = None
_NUMBA_RNG_TRIED = False
# Compiling the kernel takes a second or two. Export workers hit this
# concurrently on the very first frame, and without the lock the threads that
# arrive mid-compile see `_NUMBA_RNG is None` and silently fall back to numpy
# noise -- so different chunks of the same frame got different grain fields.
# The second export was then stable, which made it look like a one-off.
_NUMBA_RNG_LOCK = threading.Lock()


def _gauss_ksize(sigma):
    """Kernel size matching what cv2.GaussianBlur would pick for this sigma."""
    r = max(1, int(round(3.0 * float(sigma))))
    return 2 * r + 1, r


def gauss_white_gain(sigma):
    """Factor that restores the *nominal* std after blurring unit-variance white noise.

    Blurring white noise shrinks its standard deviation by ``sqrt(sum(k^2))`` for the normalised
    separable kernel ``k``.  That factor is known analytically -- and using it is wrong for small
    sigma: ``cv2.GaussianBlur``'s truncated kernel plus float32 rounding leaves a real deviation
    from the analytic value once sigma drops below about one pixel, and the finest octave sits
    exactly there on every output below ~2300 px (``MIN_OCTAVE_SIGMA_PX`` is 0.35).  Measured on a
    128x128 field, the analytic correction was off by up to 29 %, which is what made the CPU's
    fine octave louder than the shader's (the plan's A2).

    So the factor is measured, on a fixed deterministic field, once per sigma and cached.  The
    measurement costs one blur of a small field; it used to cost a ``std()`` over the whole 4K
    render, which is why this looked expensive the first time round.
    """
    key = round(float(sigma), 6)
    cached = _GAUSS_GAIN_CACHE.get(key)
    if cached is not None:
        return cached
    field = np.random.default_rng(20260130).standard_normal((192, 192)).astype(np.float32)
    blurred = _gaussian_blur(field, float(sigma))
    measured = float(np.asarray(blurred, dtype=np.float64).std())
    gain = 1.0 / measured if measured > 1e-9 else 1.0
    _GAUSS_GAIN_CACHE[key] = gain
    return gain


def _ensure_numba_rng():
    """Compile the numba kernel once, blocking other workers until it exists.

    Returns the kernel, or None when numba is unavailable.  The double-checked
    lock matters: export runs this from a thread pool on the first frame, and
    the old code published `_NUMBA_RNG_TRIED = True` *before* compiling, so
    every worker that arrived during the one-to-two second compile saw
    `_NUMBA_RNG is None` and fell back to numpy noise.  Chunks of the same
    frame then carried two different grain fields, which is why the first
    export of a session did not match the next one.
    """
    global _NUMBA_RNG, _NUMBA_RNG_TRIED
    if _NUMBA_RNG_TRIED:
        return _NUMBA_RNG
    with _NUMBA_RNG_LOCK:
        if _NUMBA_RNG_TRIED:
            return _NUMBA_RNG
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
        # Only publish "decided" once the decision is final.
        _NUMBA_RNG_TRIED = True
    return _NUMBA_RNG


def _randn_field(seed, height, width, channels):
    """Unit-variance Gaussian noise field.

    numpy's generator is the most expensive single step of the whole synthesis (~232 ms for a
    4K RGB field).  This numba kernel builds each sample from twelve uniforms, which has
    exactly unit variance by construction, needs no transcendental, and runs ~5x faster.  It
    also lets us drop the per-octave ``std()`` reduction (see :func:`gauss_white_gain`).
    """
    kernel = _ensure_numba_rng()
    if kernel is None:
        # fallback: numpy generator rescaled to unit variance
        n = np.random.default_rng(seed).standard_normal((height, width, channels)).astype(np.float32)
        sd = float(n.std())
        return (n / sd).astype(np.float32) if sd > 1e-9 else n
    return kernel(int(seed) & 0x7FFFFFFF, height, width, channels)


def _gaussian_blur(x, sigma):
    import cv2
    ksize, _r = _gauss_ksize(sigma)
    return cv2.GaussianBlur(x, (ksize, ksize), float(sigma), borderType=cv2.BORDER_REFLECT)


def synthesize_octave_grain(spec, height, width, rng):
    """The three octaves of ``layer_grain``, separately: ``(fine, cloud, mottle)``.

    Same seeds, same order and same mixing as the flat mixture, so summing the triple reproduces
    :func:`synthesize_layer_grain` bit for bit when all three carry the same response.  They are
    returned separately because P2 gives each octave its own density response: a saturated coarse
    sub-layer has no granularity left in the shoulder, so the mottle octave must fade out there
    while the fine octave takes over, and that is only expressible octave by octave.

    Each octave's per-layer (chroma) part excludes the fine micro-crystal octave, exactly as
    before -- the fine octave only feeds the shared luminance stream, so chroma never speckles at
    the pixel level.
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

    # Shared luminance stream across coupled emulsion layers: all three octaves.
    shared_fine = octave(seeds[2], s_fine, 1)
    shared_cloud = octave(seeds[3], s_cloud, 1)
    shared_mottle = octave(seeds[4], s_mottle, 1)

    rho = spec.rho
    lw = spec.lw[None, None, :]
    lw_mean = spec.lw_mean
    shared_gain = np.float32(np.sqrt(rho) * lw_mean)
    chroma_gain = np.float32(np.sqrt(1.0 - rho))

    def blend(raw, shared):
        return (chroma_gain * raw * lw + shared * shared_gain).astype(np.float32)

    fine = blend(np.float32(0.0), shared_fine * np.float32(spec.w_fine * g_fine))
    cloud = blend(rgb_cloud * np.float32(spec.w_cloud * g_cloud * CHROMA_MOTTLE_GAIN),
                  shared_cloud * np.float32(spec.w_cloud * g_cloud))
    mottle = blend(rgb_mottle * np.float32(spec.w_mottle * g_mottle * CHROMA_MOTTLE_GAIN),
                   shared_mottle * np.float32(spec.w_mottle * g_mottle))
    amplitude = np.float32(spec.amplitude)
    return (fine * amplitude, cloud * amplitude, mottle * amplitude)


def synthesize_layer_grain(spec, height, width, rng):
    """Per-channel ``layer_grain`` with the octave structure of the preview.

    Returns ``(h, w, 3)`` float32 with std matching the GLSL ``layer_grain`` by construction
    of :func:`expected_layer_std`.

    The CPU approximates the shader's value-noise octaves with Gaussian-blurred white noise.
    Every octave is rescaled to ``VALUE_NOISE_STD`` -- the constant the shader's value noise
    carries -- using the analytic blur factor, so the two renderers stay on the same amplitude
    model while their spectral character may differ.

    This is the *sum* of :func:`synthesize_octave_grain`'s three streams; callers that apply a
    per-octave response use that function directly.
    """
    fine, cloud, mottle = synthesize_octave_grain(spec, height, width, rng)
    return (fine + cloud + mottle).astype(np.float32)


def apply_block(linear_rgb, spec, rng):
    """Apply pre-LUT transmittance grain to one block, in place-safe fashion.

    ``T_out = T_in * 10^(-dD)`` with ``dD = sum_o layer_grain_o * response_o * (strength * GAIN)``.

    With a layered response table each octave is scaled by its own curve at the pixel's density
    before the sum, so the coarse sub-layer's share collapses as it saturates and the fine one
    takes over in the shoulder.  Without one the three octaves are summed first and scaled by the
    single total response, which is the P1 behaviour and still the path used by callers that pass
    no stock.
    """
    h, w = linear_rgb.shape[:2]
    gain = np.float32(spec.strength * STRENGTH_GAIN)

    if spec.layer_lut is None:
        layer = synthesize_layer_grain(spec, h, w, rng)
        response = density_response(linear_rgb, spec.density_lut, spec.film_density)
        dD = layer * gain * response
    else:
        octaves = synthesize_octave_grain(spec, h, w, rng)
        density = density_for_linear(linear_rgb, spec.film_density)
        dD = layer_response_dd(spec.layer_lut, density, *octaves)
        dD *= gain
    return np.maximum(0.0, linear_rgb * np.exp(-dD * LOG10_E)).astype(np.float32)


def apply_to_linear(linear_rgb, spec, rng, chunk_rows=512):
    """Chunked :func:`apply_block` -- a 64 MP frame does not need every noise field at once."""
    if not spec.active:
        return linear_rgb
    h = linear_rgb.shape[0]
    out = np.empty_like(linear_rgb)
    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        out[y0:y1] = apply_block(linear_rgb[y0:y1], spec, rng)
    return out
