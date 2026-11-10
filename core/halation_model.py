"""Physical halation model: the single source of truth for scatter + halation.

Both renderers read their parameters from here:

  * CPU export   -> ``apply_halation_np()``      (numpy / cv2, full resolution)
  * GPU preview  -> ``ui/gl_halation.py``        (separable Gaussians on an FBO chain)

The point of this module is that neither of them owns a constant. Every number is
either read from the film profile, derived from profile geometry, or an explicit
engine calibration -- and it says which, in ``HalationParams.sources``.

Physics (see docs/halation-engine-port-plan.md):

    boost:    y = x + max_raw * k * (exp(a*dx) - a*dx - 1),  dx = (x - x0)/max_raw
    scatter:  E' = (1-s)*E + s*[(1-w)*G(sigma_c)*E + w*G(sigma_t)*E]
    halation: out_c = E'_c + sum_j M[c][j] * (S_j - E'_j)
              S_j = sum_k w[j][k] * G(sigma_k) * E'_j

The three ``sigma_k`` are **shared across records** and come from
``core/halation_kernel.py`` -- a fit of the support's real edge-spread, not a geometric
rule of thumb. Only the weights ``w[j][k]`` differ per record, and that asymmetry is the
halo's colour (CineStill's red halo; a neutral halo for a monochrome stock). ``M`` routes
each receiver's share across the *source* records through the sheet's spectral return
matrix, so the halo is not obliged to be the colour of the light that made it.

``M``'s diagonal is ``mix = s/(1+s)`` with ``s = returned * amount``, which is why
``(E + s*S)/(1+s)`` and ``E + mix*(S - E)`` are the same thing -- the engine writes the
second, this module writes the second too now that the off-diagonal terms exist.

The exponential scatter tail is approximated by a Gaussian on purpose: a recursive
exponential filter cannot be expressed as a separable convolution pass, so it would
force the GPU to sample it with taps while the CPU filters it exactly -- which is the
exact mismatch this module exists to remove. Two Gaussians (core + tail) are what the
current CPU path already does, and both sides can render them identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .halation_kernel import HalationKernel, kernel_for

# --- The amount track -------------------------------------------------------------
# The engine draws `halationAmount` as Stops, not as a multiple: `HalationAmount.travel`
# is `EditorControlUnit.offStops...6`, and `scale(fromStops:)` stores `exp2(stops)`. So
# the slider is logarithmic about the stock's authored look, and 0 stops returns that
# look bit for bit. The bottom stop means off (`scale` is 0 there) -- the engine keeps a
# sliver of halo below it rather than claiming a precision the track does not have.
#
# Why it has to be stops and not a 0..3x multiple: the composite is
# `mix = s/(1+s)` with `s = returned * scale`, so the share saturates. At CineStill's
# authored 0.03 the share is 2.9% at 0 stops, 65.8% at +6 stops. A linear 1..3 track can
# only reach 9% of that range, which is why maxed out it looked like a thin rim.
HALATION_OFF_STOPS = -6.0
HALATION_MAX_STOPS = 6.0


def scale_from_stops(stops: float) -> float:
    """The linear multiple the renderers take, from the drawn stops (off below -6)."""
    value = float(stops)
    if value <= HALATION_OFF_STOPS:
        return 0.0
    return 2.0 ** min(value, HALATION_MAX_STOPS)


def stops_from_scale(scale: float) -> float:
    """Inverse of :func:`scale_from_stops`, clamped to the drawn track."""
    value = float(scale)
    if value <= 0.0:
        return HALATION_OFF_STOPS
    return max(HALATION_OFF_STOPS, min(math.log2(value), HALATION_MAX_STOPS))


# --- Engine calibration constants ------------------------------------------------
# These are NOT profile data. Each one is marked "engine" in HalationParams.sources
# so the UI can say "calibrated" instead of pretending it was measured.

# In-emulsion scatter core sigma per channel (um). Matches
# spektrafilm/runtime/params_schema.py:116-118, which is where the whole pipeline
# gets them from -- the profiles do not carry usable values (diffuseSigmaMM is
# present in 2/53 and zero in both).
SCATTER_CORE_UM = (2.2, 2.0, 1.6)
SCATTER_TAIL_UM = (9.3, 9.7, 9.1)
SCATTER_TAIL_WEIGHT = (0.78, 0.65, 0.67)

# 35 mm base thickness is an industry standard, not a measurement of any one stock.
# Used only when the profile omits it. VISION3 / ETERNA / the other cine products sit
# on ESTAR, which is thinner and has a much higher index, so they declare their own.
FALLBACK_THICKNESS_MM = 0.13
FALLBACK_REFRACTIVE_INDEX = 1.48

#: Acetate safety base, the support under almost every roll film and cine negative.
ACETATE_INDICES = (1.474, 1.480, 1.489)
#: Kodak ESTAR (polyester): sheet film and the thin high-strength cine bases.
ESTAR_INDICES = (1.635, 1.643, 1.657)

# Highlight boost curve constants (spektrafilm params_schema defaults).
DEFAULT_BOOST_RANGE = 0.3
DEFAULT_PROTECT_EV = 4.0
BOOST_MIDGRAY = 0.184

# Halation spatial scale is clamped so a 0 slider cannot produce a 0-width kernel.
MIN_SPATIAL_SCALE = 0.2

#: The halo radius is a physical quantity now -- it comes out of each stock's own
#: support geometry and profile fit, and the engine's design is explicitly "changing
#: the creative amount does not change the radius" (``halationLookScale`` scales the
#: returned fraction, not the kernel). The old radius knob existed only to compensate
#: for the single 168.5/238.3/292.2 um ladder every stock used to share, so it is off.
#: Sessions that still carry a non-1.0 value are clamped rather than honoured: the
#: value was fitted against a wrong radius and would now distort a right one.
SPATIAL_SCALE_ENABLED = False

SOURCE_PROFILE = "profile"
SOURCE_DERIVED = "derived"
SOURCE_ENGINE = "engine"


def _as_rgb(value: Any, default: Sequence[float]) -> tuple[float, float, float]:
    """Coerce a profile field to a 3-tuple of floats, falling back to ``default``."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            pass
    return (float(default[0]), float(default[1]), float(default[2]))


@dataclass(frozen=True)
class HalationParams:
    """Resolved halation parameters plus where each number came from."""

    strength: tuple[float, float, float]
    #: The fitted kernel: three shared sigmas, per-record weights, and the routing matrix.
    kernel: HalationKernel
    scatter_core_um: tuple[float, float, float]
    scatter_tail_um: tuple[float, float, float]
    scatter_tail_weight: tuple[float, float, float]
    boost_range: float
    protect_ev: float
    use: str = "still"
    antihalation: str = "strong"
    film_stock: str = ""
    sources: dict[str, str] = field(default_factory=dict)
    # Raw profile blocks, kept for diagnostics; never used for math.
    raw: dict[str, Any] = field(default_factory=dict)

    # -- views of the kernel, for callers that think in the old vocabulary -------
    @property
    def sigma_h_um(self) -> tuple[float, float, float]:
        """The three shared Gaussian sigmas, ascending."""
        return self.kernel.sigmas_um

    @property
    def weights(self) -> tuple[tuple[float, float, float], ...]:
        """``[record][scale]``; each row sums to one."""
        return self.kernel.weights

    @property
    def bounce_decay(self) -> tuple[tuple[float, float, float], ...]:
        """``[scale][record]`` -- the layout the blur accumulators want."""
        w = self.kernel.weights
        return tuple(tuple(w[c][k] for c in range(3)) for k in range(3))

    @property
    def n_bounces(self) -> int:
        """Number of blur scales (always three: the engine fits three Gaussians)."""
        return 3

    def source_table(self) -> str:
        """Human-readable provenance table, one line per parameter."""
        k = self.kernel
        lines = [f"halation parameters for {self.film_stock or '<none>'} "
                 f"(use={self.use}, antihalation={self.antihalation}, "
                 f"kernel={k.source}, fit residual {k.residual:.4f})"]
        rows = [
            ("strength", tuple(round(float(v), 6) for v in self.strength),
             self.sources.get("strength", "?")),
            ("mix s/(1+s)", tuple(round(float(v), 6) for v in k.mix),
             self.sources.get("mix", "?")),
            ("sigma um", tuple(round(float(v), 1) for v in k.sigmas_um),
             self.sources.get("sigma_h_um", "?")),
            ("halo radius um", (round(k.critical_radius_um, 1),), SOURCE_DERIVED),
        ]
        for key, val, src in rows:
            lines.append(f"  {key:20s} {str(val):44s} [{src}]")
        for c in range(3):
            row = ",".join(f"{float(v):.4f}" for v in k.weights[c])
            lines.append(f"  {'weights ' + 'RGB'[c]:20s} {row:44s} "
                         f"[{self.sources.get('weights', '?')}]")
        for c in range(3):
            row = ",".join(f"{float(v):.5f}" for v in k.matrix[c])
            lines.append(f"  {'matrix ' + 'RGB'[c]:20s} {row:44s} "
                         f"[{self.sources.get('matrix', '?')}]")
        for key in ("scatter_core_um", "scatter_tail_um", "scatter_tail_weight",
                    "boost_range", "protect_ev"):
            val = getattr(self, key)
            src = self.sources.get(key, "?")
            if isinstance(val, (int, float)):
                shown = f"{val:.4f}"
            else:
                shown = tuple(round(float(v), 5) for v in val)
            lines.append(f"  {key:20s} {str(shown):44s} [{src}]")
        return "\n".join(lines)


def _sigma_from_geometry(thickness_mm: float, n_index: float) -> float:
    """Back-reflection halo radius for a given base thickness and refractive index.

    Light refracts into the base at asin(1/n), travels the round trip through the
    base, and returns: the lateral radius is 2*d*tan(asin(1/n)). The halo is a disc,
    so the Gaussian that matches it takes sigma = radius/sqrt(2).

    Kept for diagnostics only -- the sigma the renderers use is fitted from the
    support's full edge-spread by ``core/halation_kernel.py``.
    """
    d_um = thickness_mm * 1000.0
    n = max(float(n_index), 1.01)
    radius = 2.0 * d_um * math.tan(math.asin(1.0 / n))
    return radius / math.sqrt(2.0)


def resolve_halation_params(film_stock: str,
                            profile_getter: Callable[[str], Any] | None = None) -> HalationParams:
    """Resolve every halation parameter for ``film_stock``, three layers deep.

    ``profile_getter`` returns either a processed profile object (the form
    ``SpektraEngine._load_profile`` produces: ``.info``, ``.data.physical_extensions``)
    or a plain dict with the same shape. Passing None yields the engine-calibrated
    set, which is what the "no stock" bypass path uses.
    """
    sources: dict[str, str] = {}
    raw: dict[str, Any] = {}

    if not film_stock or film_stock == "none" or profile_getter is None:
        # Bypass: no stock, so no profile and no sheet matrix. The kernel still comes
        # from the physical model -- a generic acetate support at the engine's own
        # antihalation-free returned fractions -- rather than from a bare sigma.
        strength = (0.015, 0.005, 0.0)
        sources.update({k: SOURCE_ENGINE for k in (
            "strength", "mix", "sigma_h_um", "weights", "matrix",
            "scatter_core_um", "scatter_tail_um", "scatter_tail_weight",
            "boost_range", "protect_ev")})
        kernel = kernel_for(strength, FALLBACK_THICKNESS_MM, ACETATE_INDICES)
        return HalationParams(
            strength=strength,
            kernel=kernel,
            scatter_core_um=SCATTER_CORE_UM,
            scatter_tail_um=SCATTER_TAIL_UM,
            scatter_tail_weight=SCATTER_TAIL_WEIGHT,
            boost_range=DEFAULT_BOOST_RANGE,
            protect_ev=DEFAULT_PROTECT_EV,
            antihalation="none",
            sources=sources,
        )

    prof = profile_getter(film_stock)

    # --- info tags (used only for the engine fallback tiers) ---
    info = getattr(prof, "info", None)
    if info is not None:
        use = str(getattr(info, "use", "still") or "still")
        antihalation = str(getattr(info, "antihalation", "strong") or "strong")
    elif isinstance(prof, Mapping):
        info_d = prof.get("info") or {}
        use = str(info_d.get("use", "still") or "still")
        antihalation = str(info_d.get("antihalation", "strong") or "strong")
    else:
        use, antihalation = "still", "strong"

    # --- physical_extensions.halation block ---
    block: dict[str, Any] = {}
    data = getattr(prof, "data", None)
    phys = getattr(data, "physical_extensions", None) if data is not None else None
    if phys is None and isinstance(prof, Mapping):
        phys = (prof.get("data") or {}).get("physical_extensions")
    if isinstance(phys, Mapping):
        block = dict(phys.get("halation") or phys.get("film_base") or {})
    raw["halation_block"] = block
    sub = block.get("profile") if isinstance(block.get("profile"), Mapping) else {}
    raw["profile_block"] = sub

    # --- strength: 53/53 have it; engine tier only if the block is missing ---
    if isinstance(block.get("strength"), (list, tuple)) and len(block["strength"]) == 3:
        strength = _as_rgb(block["strength"], (0.015, 0.005, 0.0))
        sources["strength"] = SOURCE_PROFILE
    else:
        strength = _engine_strength_for(antihalation)
        sources["strength"] = SOURCE_ENGINE

    # --- look scale and the return matrix ------------------------------------------------
    # The reference engine applies `halationLookScale` as a plain multiplier on the return
    # triple, deliberately kept outside the calibrated numbers ("a scale of 1/lookScale
    # recovers it exactly").  Our profiles do not carry it yet, so this defaults to 1 and
    # changes nothing -- it is wired so a stock that does declare it is honoured instead of
    # silently ignored.
    look_scale = 1.0
    for holder in (block, phys if isinstance(phys, Mapping) else None):
        if not isinstance(holder, Mapping):
            continue
        # The converter writes `look_scale`; `halationLookScale` is the raw fotufilm name.
        # This used to read only the camel-case key, so the value it was given never landed.
        raw_scale = holder.get("look_scale")
        if raw_scale is None:
            raw_scale = holder.get("halationLookScale")
        if raw_scale is not None:
            try:
                look_scale = max(float(raw_scale), 0.0)
            except (TypeError, ValueError):
                look_scale = 1.0
            break
    if look_scale != 1.0:
        strength = tuple(v * look_scale for v in strength)
        sources["strength"] = f"{sources['strength']}*lookScale{look_scale:g}"

    # --- return matrix: distribution only -----------------------------------------------
    # `halationReturnMatrix` spreads each receiver's share across the *source* records;
    # amplitude stays the strength's. Its one qualitative case is a monochrome stock,
    # where every row is uniform -- the returned light goes neutral whatever came in.
    # Until now we could not reproduce a general 3x3 (there was no cross-channel pass) and
    # faked the monochrome case by averaging the strength. Both renderers now composite
    # through the matrix, so it is applied for real and the averaging hack is gone.
    return_matrix = None
    for holder in (block, phys if isinstance(phys, Mapping) else None):
        candidate = holder.get("return_matrix") if isinstance(holder, Mapping) else None
        if (isinstance(candidate, (list, tuple)) and len(candidate) == 3
                and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in candidate)):
            return_matrix = candidate
            break
    sources["matrix"] = SOURCE_PROFILE if return_matrix is not None else SOURCE_ENGINE

    if return_matrix is None:
        # Only the monochrome case needs help: white light developed in silver returns
        # grey whatever colour went in, and a diagonal matrix cannot say that.
        channel_model = ""
        if info is not None:
            channel_model = str(getattr(info, "channel_model", "") or "")
        if not channel_model and isinstance(prof, Mapping):
            channel_model = str((prof.get("info") or {}).get("channel_model", "") or "")
        if channel_model == "bw":
            mean_strength = sum(strength) / 3.0
            strength = (mean_strength, mean_strength, mean_strength)
            sources["strength"] = f"{sources['strength']} (neutralised: monochrome, no sheet matrix)"

    # --- geometry: thickness and per-record refractive index ---------------------------
    thickness_raw = block.get("base_thickness_mm", block.get("thickness_mm"))
    try:
        thickness_mm = float(thickness_raw) if thickness_raw is not None else None
    except (TypeError, ValueError):
        thickness_mm = None
    if thickness_mm is None or thickness_mm <= 0:
        thickness_mm = FALLBACK_THICKNESS_MM
        geom_src = SOURCE_DERIVED + "(standard)"
    else:
        geom_src = SOURCE_PROFILE

    record_indices = block.get("record_refractive_indices")
    if (isinstance(record_indices, (list, tuple)) and len(record_indices) == 3
            and all(isinstance(v, (int, float)) and v > 1.0 for v in record_indices)):
        indices = [float(v) for v in record_indices]
    else:
        n_index = block.get("refractive_index")
        try:
            n_base = float(n_index) if n_index is not None else None
        except (TypeError, ValueError):
            n_base = None
        if n_base is not None and n_base > 1.0:
            indices = [n_base] * 3
        else:
            # Acetate is the support under almost every roll film and cine negative.
            indices = list(ACETATE_INDICES)
            geom_src = SOURCE_DERIVED + "(standard)"

    # --- haze: substrate scatter, convolved on top of the fit (0 mm is bit-identical) ---
    try:
        haze_mm = max(float(block.get("hazeMM") or 0.0), 0.0)
    except (TypeError, ValueError):
        haze_mm = 0.0

    kernel = kernel_for(strength, thickness_mm, indices, sub or None,
                        return_matrix, haze_mm)
    sources["sigma_h_um"] = f"{kernel.source} kernel / {geom_src} geometry"
    sources["weights"] = kernel.source
    sources["mix"] = sources["strength"]

    # --- scatter: no usable profile data (0/54), always engine ---
    core = sub.get("diffuseSigmaMM")
    if isinstance(core, (list, tuple)) and len(core) == 3 and any(float(v) > 0 for v in core):
        scatter_core = tuple(float(v) * 1000.0 for v in core)  # mm -> um
        sources["scatter_core_um"] = SOURCE_PROFILE
    else:
        scatter_core = SCATTER_CORE_UM
        sources["scatter_core_um"] = SOURCE_ENGINE
    sources["scatter_tail_um"] = SOURCE_ENGINE
    sources["scatter_tail_weight"] = SOURCE_ENGINE
    sources["boost_range"] = SOURCE_ENGINE
    sources["protect_ev"] = SOURCE_ENGINE

    return HalationParams(
        strength=strength,
        kernel=kernel,
        scatter_core_um=scatter_core,
        scatter_tail_um=SCATTER_TAIL_UM,
        scatter_tail_weight=SCATTER_TAIL_WEIGHT,
        boost_range=DEFAULT_BOOST_RANGE,
        protect_ev=DEFAULT_PROTECT_EV,
        use=use,
        antihalation=antihalation,
        film_stock=film_stock,
        sources=sources,
        raw=raw,
    )


def _engine_strength_for(antihalation: str) -> tuple[float, float, float]:
    """Engine-calibrated strength by antihalation class (only used when the profile is silent)."""
    tag = (antihalation or "strong").lower()
    if tag.startswith("no"):
        return (0.030, 0.001, 0.0)
    if tag.startswith("weak"):
        return (0.015, 0.005, 0.0)
    return (0.0055, 0.0001, 0.0)


def _decay_from_retention(retention: Any) -> tuple[tuple[float, float, float], ...]:
    """RETIRED: kept only so old sessions/diagnostics that import it still resolve.

    Bounce weights now come out of ``core/halation_kernel.py`` as a fitted
    ``[record][scale]`` table; ``bounceRetention`` feeds that fit instead of becoming
    a geometric series here. Reading that triple as three consecutive bounce weights
    used to invert the shape -- see the git history of this module.
    """
    ratios: list[float] = [0.5, 0.5, 0.5]
    if isinstance(retention, (list, tuple)) and len(retention) >= 3:
        try:
            ratios = [min(max(float(v), 0.0), 1.0) for v in retention[:3]]
        except (TypeError, ValueError):
            ratios = [0.5, 0.5, 0.5]
    totals = [sum(r ** k for k in range(3)) for r in ratios]
    rows = []
    for k in range(3):
        rows.append(tuple(
            (ratios[c] ** k) / totals[c] if totals[c] > 1e-12 else 1.0 / 3
            for c in range(3)))
    return tuple(rows)


def halation_overlap_px(params: HalationParams, pixel_size_um: float,
                        spatial_scale: float = 1.0, n_sigma: float = 3.0) -> int:
    """Kernel support in pixels, for tiled export overlap.

    The widest kernel in the chain is the outermost of the three fitted scales
    (``kernel.sigmas_um[2]``, which already spans the critical-angle shoulder); the
    scatter tail is narrower for every stock we have. Tiles must overlap by this much
    or the halo breaks at the seam.
    """
    scale = max(MIN_SPATIAL_SCALE, float(spatial_scale)) if SPATIAL_SCALE_ENABLED else 1.0
    sigma_um = max(params.kernel.sigmas_um) * scale
    sigma_px = sigma_um / max(1e-6, pixel_size_um)
    return int(math.ceil(n_sigma * sigma_px)) + 1


def _boost_highlights(x: np.ndarray, boost_ev: float, max_raw: float,
                      boost_range: float = DEFAULT_BOOST_RANGE,
                      protect_ev: float = DEFAULT_PROTECT_EV) -> np.ndarray:
    """spektrafilm's EV-domain highlight reconstruction.

    Normalised by the image-wide max, which makes it a *global* transform: the same
    input value maps to different outputs depending on the rest of the frame. That is
    why it can never be baked into a static 3D LUT (see params_builder.py:99-120) and
    why it has to run before the LUT on both paths.

    Everything at or below ``x0`` is untouched -- that gate is the whole point of the
    curve, and it is not optional: ``exp(a*dx) - a*dx - 1`` is non-negative for *any*
    dx and grows like ``-a*dx`` as dx goes negative, so applying it below x0 adds a
    term that increases the further a pixel sits below the protection point. Dropping
    the gate (as an earlier port of this function did) turns a 2 EV boost into a
    full-frame white-out: measured mean 254.9/255 over the entire export.
    """
    if boost_ev <= 1e-4:
        return x
    m = float(max_raw)
    if not math.isfinite(m) or m <= 0:
        return x
    # Reference clamps the protection point to the frame max and gives up when they meet;
    # without that, a normal photo (max ~1.0) would be shaped against a point it never reaches.
    x0 = min(max(BOOST_MIDGRAY * (2.0 ** protect_ev), 0.0), m)
    if x0 >= m:
        return x
    a = 28.0 ** (1.0 - boost_range)
    t0 = a * (1.0 - x0 / m)
    denom = math.exp(t0) - t0 - 1.0
    if denom <= 1e-9:
        return x
    k = (2.0 ** boost_ev - 1.0) / denom
    dx = (x - x0) / m
    shaped = np.exp(np.clip(a * dx, -60.0, 60.0)) - a * dx - 1.0
    # At x == m the bracket equals denom, so the brightest pixel lands on exactly
    # m * 2**boost_ev -- that is what the "EV" in the slider name refers to.
    return np.where(x > x0, x + m * k * shaped, x)


def _gauss(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Single-channel Gaussian blur with automatic kernel sizing."""
    import cv2
    if sigma_px <= 0.01:
        return img
    ksize = int(sigma_px * 3.0) * 2 + 1
    ksize = max(3, min(ksize, 601))
    return cv2.GaussianBlur(img, (ksize, ksize), sigma_px)


# A halo sigma of 170 um on a 24 MP 35 mm frame is ~28 px. Blurring that at full
# resolution costs more than everything else in the export combined, and the result
# is low-frequency by construction, so it survives decimation untouched. Measured
# on a 6 MP structured test image, blurring at 1/ds then resampling back agrees
# with the full-resolution blur to:
#     ds=2 -> 78 dB (max 0.6/255)   ds=4 -> 71 dB (max 1.2/255)   ds=8 -> 65 dB
# The kernel is only ever evaluated at TARGET_SIGMA pixels, so larger sigmas buy
# speed instead of cost.
_DECIMATE_MIN_SIGMA_PX = 6.0
_DECIMATE_TARGET_SIGMA_PX = 4.0


def _pick_decimation(sigma_px: float) -> int:
    """Power-of-two decimation that keeps the working sigma near the target."""
    if sigma_px < _DECIMATE_MIN_SIGMA_PX:
        return 1
    ds = int(2 ** math.floor(math.log2(max(1.0, sigma_px / _DECIMATE_TARGET_SIGMA_PX))))
    return max(1, min(ds, 8))


def _gauss_decimated(img: np.ndarray, sigma_px: float, ds: int) -> np.ndarray:
    """Gaussian with sigma_px, evaluated on a 1/ds grid and resampled back."""
    if ds <= 1 or sigma_px < _DECIMATE_MIN_SIGMA_PX:
        return _gauss(img, sigma_px)
    import cv2
    h, w = img.shape[:2]
    sh, sw = max(2, h // ds), max(2, w // ds)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = _gauss(small, sigma_px / ds)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _multi_bounce_blur(work: np.ndarray, sigma_px: Sequence[Sequence[float]],
                       decay: Sequence[Sequence[float]], ds: int) -> np.ndarray:
    """Weighted sum of ``N`` Gaussians per channel, on one shared decimated grid.

    ``decay`` carries a (R, G, B) weight per bounce. Decimating once per channel instead of
    once per (channel, bounce) is what makes this affordable: at 24 MP a single
    full-resolution resample costs ~120 ms, and the naive nesting was paying for 18 of them
    per frame. Here it is 3 down + 3 up, with all bounce layers accumulating inside the
    small grid where blur is cheap.
    """
    import cv2
    h, w = work.shape[:2]
    out = np.empty_like(work)
    if ds > 1:
        sh, sw = max(2, h // ds), max(2, w // ds)
    for c in range(3):
        sigmas = [row[c] for row in sigma_px]
        weights = [float(row[c]) for row in decay]
        if ds > 1:
            small = cv2.resize(work[..., c], (sw, sh), interpolation=cv2.INTER_AREA)
            acc = np.zeros_like(small)
            for sigma, w_k in zip(sigmas, weights):
                if w_k <= 1e-6 or sigma <= 0.01:
                    continue
                acc += _gauss(small, sigma / ds) * w_k
            out[..., c] = cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            acc = np.zeros_like(work[..., c])
            for sigma, w_k in zip(sigmas, weights):
                if w_k <= 1e-6 or sigma <= 0.01:
                    continue
                acc += _gauss(work[..., c], sigma) * w_k
            out[..., c] = acc
    return out


def apply_halation_np(linear: np.ndarray,
                      params: HalationParams,
                      pixel_size_um: float,
                      amount: float = 1.0,
                      scatter_amount: float = 0.0,
                      spatial_scale: float = 1.0,
                      boost_ev: float = 0.0,
                      max_raw: float | None = None,
                      legacy: bool = False,
                      decimate: bool = True) -> np.ndarray:
    """Apply boost -> scatter -> halation to a scene-linear HxWx3 float32 image.

    ``legacy=True`` reproduces the pre-rework CPU implementation bit-for-bit
    (single sigma for all three channels, hard-coded 2.0/9.5 um, 0.5^k decay,
    brightness-keyed boost). It exists only so the swap-in can be proven
    non-destructive before the parameter behaviour deliberately changes.
    """
    if linear is None or linear.ndim != 3 or linear.shape[2] < 3:
        return linear

    amount = float(amount)
    scatter_amount = float(scatter_amount)
    if not SPATIAL_SCALE_ENABLED:
        spatial_scale = 1.0
    scale = max(MIN_SPATIAL_SCALE, float(spatial_scale)) if not legacy else max(0.2, float(spatial_scale))
    if amount <= 0.001 and scatter_amount <= 0.001:
        return linear

    work = np.ascontiguousarray(linear[..., :3], dtype=np.float32)
    px = max(1e-6, float(pixel_size_um))

    if legacy:
        return _apply_legacy(work, params, px, amount, scatter_amount, scale, boost_ev)

    # --- boost: global, before everything else ---
    if boost_ev > 1e-4 and amount > 0.001:
        m = float(max_raw) if max_raw is not None else float(work.max())
        work = _boost_highlights(work, boost_ev, m, params.boost_range, params.protect_ev)

    # --- scatter: energy-preserving core + tail, per channel ---
    if scatter_amount > 0.001:
        eff_s = min(0.5, scatter_amount * 0.06)
        out = np.empty_like(work)
        for c in range(3):
            sigma_c = max(0.5, params.scatter_core_um[c] * scale / px)
            sigma_t = max(1.0, params.scatter_tail_um[c] * scale / px)
            w_t = float(params.scatter_tail_weight[c])
            core = _gauss(work[..., c], sigma_c)
            tail = _gauss(work[..., c], sigma_t)
            blended = core * (1.0 - w_t) + tail * w_t
            out[..., c] = (1.0 - eff_s) * work[..., c] + eff_s * blended
        work = out

    # --- halation: three shared blur scales, per-record weights, routed by the sheet ---
    # The engine's composite is ``out_c = E_c + sum_j M[c][j] * (S_j - E_j)`` where S_j is
    # record j's own blur. When M is diagonal this is algebraically ``(E + s*S)/(1+s)`` --
    # the form this module used to use -- so the only new thing is the off-diagonal part,
    # which lets a red halo land partly in the green and blue records.
    if amount > 0.001:
        matrix = np.asarray(params.kernel.matrix_for(amount), dtype=np.float32)
        if np.any(matrix):
            # Sigmas are shared across records; only the weights differ per record.
            sigmas_px = [[float(s) * scale / px] * 3 for s in params.kernel.sigmas_um]
            weights = [[params.kernel.weights[c][k] for c in range(3)] for k in range(3)]
            ds = _pick_decimation(max(row[0] for row in sigmas_px)) if decimate else 1
            scattered = _multi_bounce_blur(work, sigmas_px, weights, ds)
            delta = scattered - work
            work = work + delta @ matrix.T

    return work.astype(linear.dtype, copy=False)


def _apply_legacy(work: np.ndarray, params: HalationParams, px: float, amount: float,
                  scatter_amount: float, scale: float, boost_ev: float) -> np.ndarray:
    """The pre-rework CPU path: single sigma, 0.5^k decay, brightness-keyed boost.

    Kept so a shape change can be separated from a parameter change when comparing
    renders. It no longer reproduces the old numbers bit-for-bit -- the sigma it reads
    is now the kernel's first fitted scale, not the 168 um geometric ladder.
    """
    import cv2
    eff_strength = np.array(params.strength, dtype=np.float32) * amount

    if scatter_amount > 0.001:
        sigma_c_px = max(0.5, (2.0 * scale) / px)
        sigma_t_px = max(1.0, (9.5 * scale) / px)
        kc = int(sigma_c_px * 3.0) * 2 + 1
        kt = int(sigma_t_px * 3.0) * 2 + 1
        core = cv2.GaussianBlur(work, (min(kc, 101), min(kc, 101)), sigma_c_px)
        tail = cv2.GaussianBlur(work, (min(kt, 151), min(kt, 151)), sigma_t_px)
        w_tail = np.array([0.78, 0.65, 0.67], dtype=np.float32)
        scattered = core * (1.0 - w_tail) + tail * w_tail
        eff_s = min(0.5, scatter_amount * 0.06)
        work = (1.0 - eff_s) * work + eff_s * scattered

    if amount <= 0.001 or not np.any(eff_strength > 0):
        return work

    # The old preset carried a single scalar sigma, taken from the first radius, and the
    # legacy decay is the 0.5^k series -- identical on every record.
    sigma_h_px = (float(params.sigma_h_um[0]) * scale) / px
    decay = np.array([[0.5 ** (k - 1)] * 3 for k in range(1, 4)], dtype=np.float32)
    decay /= decay.sum(axis=0, keepdims=True)

    hal_input = work
    if boost_ev > 0.001:
        luma_in = 0.2126 * work[..., 0] + 0.7152 * work[..., 1] + 0.0722 * work[..., 2]
        boost_scale = 1.0 + np.maximum(0.0, luma_in - 0.7) * (boost_ev * 0.6)
        hal_input = work * boost_scale[..., None]

    hal_blur = np.zeros_like(work)
    for k, wk in zip(range(1, 4), decay):
        sigma_k = sigma_h_px * math.sqrt(k)
        ksize = int(sigma_k * 3.0) * 2 + 1
        ksize = min(ksize, 301)
        hal_blur += cv2.GaussianBlur(hal_input, (ksize, ksize), sigma_k) * wk.reshape(1, 1, 3)

    return (work + eff_strength * hal_blur) / (1.0 + eff_strength)
