"""Physical halation model: the single source of truth for scatter + halation.

Both renderers read their parameters from here:

  * CPU export   -> ``apply_halation_np()``      (numpy / cv2, full resolution)
  * GPU preview  -> ``ui/gl_halation.py``        (separable Gaussians on an FBO chain)

The point of this module is that neither of them owns a constant. Every number is
either read from the film profile, derived from profile geometry, or an explicit
engine calibration -- and it says which, in ``HalationParams.sources``.

Physics (see docs/halation-rework-plan.md §4.1):

    boost:    y = x + max_raw * k * (exp(a*dx) - a*dx - 1),  dx = (x - x0)/max_raw
    scatter:  E' = (1-s)*E + s*[(1-w)*G(sigma_c)*E + w*G(sigma_t)*E]
    halation: E'' = (E' + a_tot * sum_k decay_k * G(sigma_h*sqrt(k)) * E') / (1 + a_tot)

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
# Used only when the profile omits it (20/53 stocks, including all the main ones).
FALLBACK_THICKNESS_MM = 0.13
FALLBACK_REFRACTIVE_INDEX = 1.48
FALLBACK_SIGMA_H_UM = 65.0

# Bounce decay when the profile has no bounceRetention (51/53).
DEFAULT_N_BOUNCES = 3

# Highlight boost curve constants (spektrafilm params_schema defaults).
DEFAULT_BOOST_RANGE = 0.3
DEFAULT_PROTECT_EV = 4.0
BOOST_MIDGRAY = 0.184

# Halation spatial scale is clamped so a 0 slider cannot produce a 0-width kernel.
MIN_SPATIAL_SCALE = 0.2

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
    sigma_h_um: tuple[float, float, float]
    scatter_core_um: tuple[float, float, float]
    scatter_tail_um: tuple[float, float, float]
    scatter_tail_weight: tuple[float, float, float]
    bounce_decay: tuple[float, ...]
    n_bounces: int
    boost_range: float
    protect_ev: float
    use: str = "still"
    antihalation: str = "strong"
    film_stock: str = ""
    sources: dict[str, str] = field(default_factory=dict)
    # Raw profile blocks, kept for diagnostics; never used for math.
    raw: dict[str, Any] = field(default_factory=dict)

    def source_table(self) -> str:
        """Human-readable provenance table, one line per parameter."""
        lines = [f"halation parameters for {self.film_stock or '<none>'} "
                 f"(use={self.use}, antihalation={self.antihalation})"]
        for key in ("strength", "sigma_h_um", "scatter_core_um", "scatter_tail_um",
                    "scatter_tail_weight", "bounce_decay", "boost_range", "protect_ev"):
            val = getattr(self, key)
            src = self.sources.get(key, "?")
            shown = val if isinstance(val, (int, float)) else tuple(round(float(v), 5) for v in val)
            lines.append(f"  {key:20s} {str(shown):44s} [{src}]")
        return "\n".join(lines)


def _sigma_from_geometry(thickness_mm: float, n_index: float) -> float:
    """Back-reflection halo radius for a given base thickness and refractive index.

    Light refracts into the base at asin(1/n), travels the round trip through the
    base, and returns: the lateral radius is 2*d*tan(asin(1/n)). The halo is a disc,
    so the Gaussian that matches it takes sigma = radius/sqrt(2).
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
        sources.update({k: SOURCE_ENGINE for k in (
            "strength", "sigma_h_um", "scatter_core_um", "scatter_tail_um",
            "scatter_tail_weight", "bounce_decay", "boost_range", "protect_ev")})
        return HalationParams(
            strength=(0.015, 0.005, 0.0),
            sigma_h_um=(FALLBACK_SIGMA_H_UM,) * 3,
            scatter_core_um=SCATTER_CORE_UM,
            scatter_tail_um=SCATTER_TAIL_UM,
            scatter_tail_weight=SCATTER_TAIL_WEIGHT,
            bounce_decay=_decay_from_retention(None),
            n_bounces=DEFAULT_N_BOUNCES,
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
        if isinstance(holder, Mapping) and holder.get("halationLookScale") is not None:
            try:
                look_scale = max(float(holder["halationLookScale"]), 0.0)
            except (TypeError, ValueError):
                look_scale = 1.0
            break
    if look_scale != 1.0:
        strength = tuple(v * look_scale for v in strength)
        sources["strength"] = f"{sources['strength']}*lookScale{look_scale:g}"

    # `halationReturnMatrix` is *distribution only*: the engine normalises each row and spreads
    # that receiver's share across the source records, amplitude stays the strength's.  Its one
    # qualitative case is a monochrome stock, where every row is uniform -- the returned light
    # goes neutral whatever colour came in.  We cannot reproduce a general 3x3 without a
    # cross-channel pass, but the monochrome case is exact and free: a uniform row is what a
    # neutral triple already is.  Measured over the 39 stocks that ship a matrix: colour stocks
    # move a median 5.9% of the halo across records (max 100% on one near-empty blue row of
    # Aerocolor), monochrome stocks move 66.7% today and 0% after this line.
    channel_model = ""
    if info is not None:
        channel_model = str(getattr(info, "channel_model", "") or "")
    if not channel_model and isinstance(prof, Mapping):
        channel_model = str((prof.get("info") or {}).get("channel_model", "") or "")
    if channel_model == "bw":
        mean_strength = sum(strength) / 3.0
        strength = (mean_strength, mean_strength, mean_strength)
        sources["strength"] = f"{sources['strength']} (neutralised: monochrome return matrix)"

    # --- sigma_h: derived from geometry (thickness/refractive index) ---
    # Three tiers, and the label says which one fired:
    #   derived          -> both numbers came from the profile
    #   derived(standard)-> geometry taken from a declared industry constant
    #   engine           -> geometry unusable, fell back to the 65 um calibration
    thickness_mm = block.get("base_thickness_mm", block.get("thickness_mm"))
    n_index = block.get("refractive_index")
    if thickness_mm is None or n_index is None:
        thickness_mm = FALLBACK_THICKNESS_MM if thickness_mm is None else thickness_mm
        n_index = FALLBACK_REFRACTIVE_INDEX if n_index is None else n_index
        sigma_src = SOURCE_DERIVED + "(standard)"
    else:
        sigma_src = SOURCE_DERIVED
    try:
        sigma_scalar = _sigma_from_geometry(float(thickness_mm), float(n_index))
        # Per-channel spread follows the same 0.4 % refractive-index band the
        # existing preset uses; retained so the R/B asymmetry does not vanish.
        n_base = float(n_index)
        sigmas = [_sigma_from_geometry(float(thickness_mm), n_base * k)
                  for k in (0.996, 1.0, 1.006)]
        sigma_h_um = (float(sigmas[0]), float(sigmas[1]), float(sigmas[2]))
        if not all(math.isfinite(v) and v > 0 for v in sigma_h_um):
            raise ValueError("non-positive sigma")
    except (TypeError, ValueError):
        sigma_h_um = (FALLBACK_SIGMA_H_UM,) * 3
        sigma_src = SOURCE_ENGINE
    sources["sigma_h_um"] = sigma_src

    # --- scatter: no usable profile data (0/53), always engine ---
    core = sub.get("diffuseSigmaMM")
    if isinstance(core, (list, tuple)) and len(core) == 3 and any(float(v) > 0 for v in core):
        scatter_core = tuple(float(v) * 1000.0 for v in core)  # mm -> um
        sources["scatter_core_um"] = SOURCE_PROFILE
    else:
        scatter_core = SCATTER_CORE_UM
        sources["scatter_core_um"] = SOURCE_ENGINE
    sources["scatter_tail_um"] = SOURCE_ENGINE
    sources["scatter_tail_weight"] = SOURCE_ENGINE

    # --- bounce decay: 2/53 carry it (Cinestill), rest use the 0.5^k law ---
    retention = sub.get("bounceRetention")
    if isinstance(retention, (list, tuple)) and len(retention) >= 1 and any(float(v) > 0 for v in retention):
        bounce_decay = _decay_from_retention(retention)
        sources["bounce_decay"] = SOURCE_PROFILE
    else:
        bounce_decay = _decay_from_retention(None)
        sources["bounce_decay"] = SOURCE_ENGINE
    sources["boost_range"] = SOURCE_ENGINE
    sources["protect_ev"] = SOURCE_ENGINE

    return HalationParams(
        strength=strength,
        sigma_h_um=sigma_h_um,
        scatter_core_um=scatter_core,
        scatter_tail_um=SCATTER_TAIL_UM,
        scatter_tail_weight=SCATTER_TAIL_WEIGHT,
        bounce_decay=bounce_decay,
        n_bounces=DEFAULT_N_BOUNCES,
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


def _decay_from_retention(retention: Any) -> tuple[float, ...]:
    """Normalise bounce retention into decay weights.

    ``bounceRetention`` is the fraction of light retained per round trip, given as one
    entry per bounce (Cinestill: ``[0.283, 0.1375, 0.0651]``, whose R:G:B asymmetry is
    the physical origin of the red halo). The entries are already the per-bounce
    weights, so they only need normalising -- not collapsing into a geometric series,
    which would discard exactly the information the profile took the trouble to record.
    """
    if retention is not None:
        try:
            vals = [float(v) for v in retention]
        except (TypeError, ValueError):
            vals = []
        if len(vals) >= DEFAULT_N_BOUNCES and all(v > 0 for v in vals[:DEFAULT_N_BOUNCES]):
            weights = vals[:DEFAULT_N_BOUNCES]
            total = sum(weights)
            return tuple(float(w / total) for w in weights)
    weights = [0.5 ** k for k in range(DEFAULT_N_BOUNCES)]
    total = sum(weights)
    return tuple(float(w / total) for w in weights)


def halation_overlap_px(params: HalationParams, pixel_size_um: float,
                        spatial_scale: float = 1.0, n_sigma: float = 3.0) -> int:
    """Kernel support in pixels, for tiled export overlap.

    The widest kernel in the chain is the outermost halation bounce
    (sigma_h * sqrt(n_bounces)) -- the scatter tail is narrower for every stock we
    have. Tiles must overlap by this much or the halo breaks at the seam.
    """
    scale = max(MIN_SPATIAL_SCALE, float(spatial_scale))
    sigma_um = max(params.sigma_h_um) * math.sqrt(max(1, params.n_bounces)) * scale
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
                       decay: Sequence[float], ds: int) -> np.ndarray:
    """Weighted sum of ``N`` Gaussians per channel, on one shared decimated grid.

    Decimating once per channel instead of once per (channel, bounce) is what makes
    this affordable: at 24 MP a single full-resolution resample costs ~120 ms, and the
    naive nesting was paying for 18 of them per frame. Here it is 3 down + 3 up, with
    all bounce layers accumulating inside the small grid where blur is cheap.
    """
    import cv2
    h, w = work.shape[:2]
    out = np.empty_like(work)
    if ds > 1:
        sh, sw = max(2, h // ds), max(2, w // ds)
    for c in range(3):
        sigmas = [row[c] for row in sigma_px]
        if ds > 1:
            small = cv2.resize(work[..., c], (sw, sh), interpolation=cv2.INTER_AREA)
            acc = np.zeros_like(small)
            for sigma, w_k in zip(sigmas, decay):
                if w_k <= 1e-6 or sigma <= 0.01:
                    continue
                acc += _gauss(small, sigma / ds) * w_k
            out[..., c] = cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            acc = np.zeros_like(work[..., c])
            for sigma, w_k in zip(sigmas, decay):
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

    eff = np.array(params.strength, dtype=np.float32) * amount

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

    # --- halation: multi-bounce additive sum, DC-normalised ---
    if amount > 0.001 and np.any(eff > 0):
        # Each bounce is a wide, low-frequency kernel, so the whole sum is evaluated on
        # a decimated grid (see _gauss_decimated). The decimation is chosen once from
        # the widest bounce so all three layers share the same working resolution --
        # mixing grids between layers would make the sum's support jump.
        sigmas_px = [[params.sigma_h_um[c] * math.sqrt(k) * scale / px
                      for c in range(3)]
                     for k in range(1, params.n_bounces + 1)]
        ds = _pick_decimation(max(max(row) for row in sigmas_px)) if decimate else 1
        hal_blur = _multi_bounce_blur(work, sigmas_px, params.bounce_decay, ds)
        work = (work + eff * hal_blur) / (1.0 + eff)

    return work.astype(linear.dtype, copy=False)


def _apply_legacy(work: np.ndarray, params: HalationParams, px: float, amount: float,
                  scatter_amount: float, scale: float, boost_ev: float) -> np.ndarray:
    """The pre-rework CPU path, reproduced exactly for regression pinning."""
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

    # The old preset carried a single scalar sigma, taken from the first radius.
    sigma_h_px = (float(params.sigma_h_um[0]) * scale) / px
    decay = np.array([0.5 ** (k - 1) for k in range(1, 4)], dtype=np.float32)
    decay /= decay.sum()

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
        hal_blur += cv2.GaussianBlur(hal_input, (ksize, ksize), sigma_k) * wk

    return (work + eff_strength * hal_blur) / (1.0 + eff_strength)
