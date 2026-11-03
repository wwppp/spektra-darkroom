"""One record's film, laid once at 1 µm texels over a 256 µm seamless square.

Ported from ``Sources/FotufilmCore/FilmGrain.swift`` (``renderTile``) and
``FilmGrainTiles.swift`` (``builtLights`` / ``tileSigma48`` / ``blockSharing``).

The idea, in one line: **crystals are hashed onto film millimetres, and the exposure decides
which of them develop.**
Each cell of the film draws a Poisson number of crystals; every crystal draws its position, its
development draw and its dye mark from the hash of its cell, so the same film sits under any
exposure.  A developed crystal leaves its demand at its place, the demand spreads as the dye cloud
Jarvis measured (``exp(-r/k)``, laid as four Gaussian terms), the sublayer's clouds add before
they saturate against its coupler pool, and a pixel is the light that passed.

Two things this file gets right that a "blobs of noise" implementation cannot:

* **the same crystals at every density.**  A level is the same film with more of it developed, so
  neighbouring levels blend coherently and the tile reads the sheet at the read density;
* **light averages, then ``-log10``.**  Density is not averaged; transmittance is, over the
  supersampled grid, exactly as a microdensitometer reads.

Levels are gross densities from ``d_min`` to ``d_max``.  P1 builds one record at a time; the
multi-record bank (record R/G/B, seam cross-fade, two layings) is P4/P5.

Where this differs from the reference, and why
---------------------------------------------
* **The blur wraps.**  The reference pads the tile by a 64 µm margin and lets the blur fall off
  the edge; we convolve on the tile itself, periodically, so a cloud crossing the edge reappears
  on the far side.  The field's statistics are the same and the tile is exactly seamless, which
  the block laying in P4 wants.
* **A grain narrower than a sample lays its area, blurred by a matched Gaussian.**  The reference
  rasterises each square grain's exact overlap with the samples it touches, which at the top of a
  Tri-X curve is 25M squares -- seconds per level in numpy.  Laying the area and blurring it by a
  Gaussian of the square's own second moment reads the same through a 48 µm aperture, at a
  fraction of the cost.  Overlapping squares are the one case it treats approximately.
* **The blur is one Fourier product per sublayer** instead of four stacked terms on two grids.
  Identical arithmetic in the continuum: a sum of Gaussians is a sum of their transforms.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .population import (
    APERTURE_RADIUS_MM,
    MARK_DISPERSION,
    CrystalPopulation,
)

# Crystals drawn at once when averaging the gamma mark.  Small enough that one slice is a couple
# of megabytes, large enough that the per-slice overhead disappears; the value has no effect on
# the result, only on the peak allocation (see ``_developed``).
_MARK_BLOCK = 8 * 1024 * 1024
_SPLAT_BLOCK = 8 * 1024 * 1024         # crystals per scatter-add chunk (peak-memory guard)

__all__ = [
    "TILE_SIDE",
    "TILE_TEXEL_MM",
    "TILE_LEVELS",
    "TABLE_SAMPLES",
    "DYE_CLOUD_DECAY_MM",
    "Sublayer",
    "FilmRecord",
    "sublayers_for",
    "build_level",
    "build_levels",
    "sigma48",
    "anchor_to_sheet",
]

TILE_SIDE = 256                       # texels a side
TILE_TEXEL_MM = 0.001                 # 1 µm a texel: 256 µm of film
TILE_LEVELS = 17                      # gross densities from D-min to D-max
TILE_BLOCK_TEXELS = 128               # frames lay the tile in 128 µm blocks
TILE_SEAM_TEXELS = 16                 # cross-faded over 16 µm either side of an edge
TABLE_SAMPLES = 129                   # developed-fraction samples over the density range
FINEST_SAMPLE_MM = 0.00035            # finest sub-pixel sample spacing
MAX_SUPERSAMPLE = 8

DYE_CLOUD_DECAY_MM = 0.00145          # Jarvis, J. Photogr. Sci. 40:105 / 43:136
#: Jarvis's ``exp(-r/k)`` as four Gaussians ``weight · exp(-r²/2σ²)``; Σ weight σ² = 1, so each
#: cloud carries exactly its dye.  Fitted to the transfer function to 9 % above 10⁻³.
DYE_CLOUD_TERMS: tuple[tuple[float, float], ...] = (
    (0.1162, 0.1096), (0.3089, 0.2167), (0.7469, 0.4030), (1.7320, 0.2510),
)
SILVER_GRAIN_EDGE = 12.0              # peak demand over local density (a stance)
SILVER_GRAIN_DENSITY = 2.0            # transmits one percent (a stance)
FASTEST_CRYSTAL_MM = 0.0012           # fastest class's crystal width (a stance)
ANCHOR_PASSES = 8
ANCHOR_TOLERANCE = 0.02               # stop within 2 % in log ratio
_SQRT2PI = np.sqrt(2.0 * np.pi)

#: Granularity-against-density shape per record, fitted to Kodak's published diffuse RMS
#: granularity against density for Vision3 (5203/5207/5213/5219, 48 µm aperture) — the pack
#: default ``FilmStockDefaults.grainDensityRecords``, used wherever a stock states none:
#:     s²(D) = (1 - e^(-D/toe)) · (1 + amp·e^(-D/decay)
#:                                   + hump·e^(-((D - hump_d)/hump_w)²/2))
#: The first factor is how much of the emulsion developed at all; the bracket is the variance
#: one unit of density carries — loud while the fast coarse sub-layer develops, decaying onto
#: the fine slow one, then rising again near net 1.1–1.5, most in the blue record, where that
#: slow sub-layer's own coarse population comes in.  Normalised at the record's read density,
#: so the anchor contract (σ48@read == read_sigma) is untouched.
DENSITY_PROFILE_DEFAULT: tuple[tuple[float, float, float, float, float, float], ...] = (
    (20.0, 0.3373, 0.2629, 0.8187, 1.1811, 0.3212),
    (20.0, 0.3583, 0.3183, 1.6514, 1.4856, 0.3631),
    (20.0, 0.1069, 0.5457, 4.2644, 1.38, 0.3446),
)


def dye_cloud_variance(density: float, row: tuple[float, ...]) -> float:
    """``s²(D)`` of the fitted chromogenic shape, in arbitrary units."""
    amp, toe, decay, hump, hump_d, hump_w = row
    offset = (density - hump_d) / max(hump_w, 1e-4)
    return float((1.0 - np.exp(-density / max(toe, 1e-4)))
                 * (1.0 + amp * np.exp(-density / max(decay, 1e-4))
                    + hump * np.exp(-0.5 * offset * offset)))


def density_modulation(net: float, fog: float, anchor_net: float,
                       row: tuple[float, ...]) -> float:
    """σ(D)/σ(anchor) of the fitted shape, fog counted at both densities."""
    here = max(net, 0.0) + fog
    anchor = max(anchor_net + fog, 1e-4)
    ratio = dye_cloud_variance(here, row) / max(dye_cloud_variance(anchor, row), 1e-6)
    return float(np.sqrt(max(ratio, 0.0)))


def supersample_for(pitch_mm: float) -> int:
    return int(min(max(np.ceil(pitch_mm / FINEST_SAMPLE_MM), 1), MAX_SUPERSAMPLE))


# ---------------------------------------------------------------------------------------------
# How a crystal's demand is laid
# ---------------------------------------------------------------------------------------------
def _gamma_quantile(p: np.ndarray, shape: int) -> np.ndarray:
    """Quantile of a gamma distribution with integer shape and unit scale (Erlang CDF)."""
    def cdf(x: np.ndarray) -> np.ndarray:
        term = np.ones_like(x)
        total = np.ones_like(x)
        for n in range(1, shape):
            term = term * x / n
            total = total + term
        return 1.0 - np.exp(-x) * total

    lo = np.zeros_like(p)
    hi = np.full_like(p, shape * 20.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _profile_reach(peak: float, silver: bool) -> float:
    widest = 1.0 if silver else DYE_CLOUD_TERMS[-1][0]
    return widest * np.sqrt(2.0 * np.log(max(peak * 8.0, 1.0) / 1e-7))


def _profile_value(rho: np.ndarray, silver: bool) -> np.ndarray:
    if silver:
        return np.exp(-rho * rho / 2.0)
    total = np.zeros_like(rho)
    for sigma, weight in DYE_CLOUD_TERMS:
        total += weight * np.exp(-rho * rho / (2.0 * sigma * sigma))
    return total


def _occupied_area(peak: float, silver: bool) -> float:
    """``∫ (1 - exp(-peak · profile(ρ))) d²ρ``: the share of capacity a crystal fills."""
    p = max(peak, 0.0)
    if silver:
        if p < 6.0:
            term, total = p, 0.0
            for k in range(1, 61):
                total += term / k
                term *= -p / (k + 1)
                if abs(term) < 1e-14:
                    break
            return 2.0 * np.pi * total
        e1 = np.exp(-p) / p * (1.0 - 1.0 / p + 2.0 / (p * p) - 6.0 / (p ** 3))
        return 2.0 * np.pi * (e1 + np.log(p) + 0.5772156649)
    steps = 800
    reach = _profile_reach(p, silver)
    dr = reach / steps
    rho = (np.arange(steps) + 0.5) * dr
    return float(2.0 * np.pi * np.sum((1.0 - np.exp(-p * _profile_value(rho, silver)))
                                      * rho * dr))


def unit_cloud_dye(capacity: float, edge: float, silver: bool) -> float:
    """Dye one crystal of unit demand length forms, as a densitometer reads it."""
    shape = int(round(1.0 / (MARK_DISPERSION * MARK_DISPERSION)))
    marks = _gamma_quantile((np.arange(24) + 0.5) / 24.0, shape) / shape
    steps = 800
    top = _profile_reach(edge * float(marks.max()), silver)
    dr = top / steps
    rho = (np.arange(steps) + 0.5) * dr
    values = _profile_value(rho, silver)
    total = 0.0
    for mark in marks:
        peak = edge * float(mark)
        dye = capacity * (1.0 - np.exp(-peak * values))
        total += 0.4342944819 * 2.0 * np.pi * float(np.sum((1.0 - np.exp(-np.log(10.0) * dye))
                                                           * rho * dr))
    return total / marks.size


def unit_void_integral(edge: float, silver: bool) -> float:
    """``E_mark ∫ (1 - exp(-edge · mark · profile(ρ))) d²ρ`` for a unit demand length."""
    shape = int(round(1.0 / (MARK_DISPERSION * MARK_DISPERSION)))
    marks = _gamma_quantile((np.arange(24) + 0.5) / 24.0, shape) / shape
    return float(np.mean([_occupied_area(edge * float(m), silver) for m in marks]))


@dataclass(frozen=True)
class Sublayer:
    """One coated sublayer, as the tile lays it."""

    silver: bool
    coated_per_mm2: float          # crystals coated per mm²
    sigma_mm: float                # demand length: a dye cloud's decay, a silver grain's sigma
    peak_demand: float             # peak demand of a mark-1 crystal, density
    capacity: float                # most density a point can form
    dye_per_cloud_mm2: float       # dye one mark-1 crystal forms, density × mm²
    cell_mm: float                 # side of the cell its crystals are hashed in
    forming: np.ndarray            # developed fraction against gross density, TABLE_SAMPLES
    void_integral_mm2: float
    smallest_sigma_mm: float = DYE_CLOUD_DECAY_MM   # narrowest length the demand may take

    @property
    def edge(self) -> float:
        return self.peak_demand / max(self.capacity, 1e-6)

    def fraction_at(self, gross: float, d_min: float, d_max: float) -> float:
        t = min(max((gross - d_min) / max(d_max - d_min, 1e-6), 0.0), 1.0) * (TABLE_SAMPLES - 1)
        i = min(int(t), TABLE_SAMPLES - 2)
        f = t - i
        return float(self.forming[i] * (1.0 - f) + self.forming[i + 1] * f)

    def with_dye(self, factor: float) -> "Sublayer":
        """Each crystal forms ``factor`` times the dye, ``1/factor`` as many are coated."""
        if factor <= 0 or factor == 1:
            return self
        return _shape(replace(self, coated_per_mm2=self.coated_per_mm2 / factor,
                              dye_per_cloud_mm2=self.dye_per_cloud_mm2 * factor))


def _shape(layer: Sublayer) -> Sublayer:
    """Solve the demand a crystal's dye needs.

    A dye cloud keeps its measured decay length and takes whatever peak its dye needs.  A silver
    grain is as wide as its edge makes it, or held at its crystal's width with its peak lowered
    until it forms that density-area.
    """
    capacity = layer.capacity
    smallest = layer.smallest_sigma_mm
    if not layer.silver:
        peak = _solve_peak(layer.dye_per_cloud_mm2, capacity, smallest, silver=False)
        return replace(layer, sigma_mm=smallest, peak_demand=peak * capacity,
                       void_integral_mm2=(smallest * smallest
                                          * unit_void_integral(peak, silver=False)))
    edge = SILVER_GRAIN_EDGE
    free = np.sqrt(max(layer.dye_per_cloud_mm2, 0.0)
                   / max(unit_cloud_dye(capacity, edge, True), 1e-12))
    if free >= smallest:
        sigma, peak = free, edge
    else:
        sigma = smallest
        peak = _solve_peak(layer.dye_per_cloud_mm2, capacity, sigma, silver=True)
    return replace(layer, sigma_mm=sigma, peak_demand=peak * capacity,
                   void_integral_mm2=sigma * sigma * unit_void_integral(peak, True))


def _solve_peak(dye_per_cloud: float, capacity: float, sigma: float, silver: bool) -> float:
    """Peak demand over capacity that forms ``dye_per_cloud`` at ``sigma`` (Illinois secant)."""
    wanted = np.log(max(dye_per_cloud, 1e-30) / (sigma * sigma))

    def miss(x: float) -> float:
        return np.log(max(unit_cloud_dye(capacity, float(np.exp(x)), silver), 1e-30)) - wanted

    low, high = np.log(1e-6), np.log(1e4)
    f_low, f_high = miss(low), miss(high)
    best = low if f_low >= 0 else high
    if f_low < 0 < f_high:
        side = 0
        for _ in range(40):
            best = (low * f_high - high * f_low) / (f_high - f_low)
            f_mid = miss(best)
            if abs(f_mid) < 1e-6 or high - low < 1e-6:
                break
            if f_mid < 0:
                low, f_low = best, f_mid
                if side == -1:
                    f_high /= 2
                side = -1
            else:
                high, f_high = best, f_mid
                if side == 1:
                    f_low /= 2
                side = 1
    return float(np.exp(best))


@dataclass(frozen=True)
class FilmRecord:
    """One record as the tiles lay it: its sublayers and the density range they cover."""

    d_min: float
    d_max: float
    sublayers: tuple[Sublayer, ...]

    def grosses(self, levels: int = TILE_LEVELS) -> list[float]:
        span = max(self.d_max - self.d_min, 1e-3)
        return [self.d_min + span * k / (levels - 1) for k in range(levels)]


def sublayers_for(model: CrystalPopulation, levels: int = TABLE_SAMPLES,
                  d_min: float | None = None, d_max: float | None = None) -> FilmRecord:
    """Turn a fitted population into the sublayers a tile lays.

    The developed-fraction table is indexed by *net* density, level 0 at D-min and the last level
    at D-max, because that is what the sampler hands it: a pixel's own developed density above
    base.  The exposure it implies has to come from ``log_exposure_for``, which turns a slide
    around -- a reversal's dense end is its *least* exposed, so reading the curve at the density
    directly gave a slide the same table as a negative and laid its dye on the wrong end of the
    ladder (measured on Provia 100F: 0.98 formed at net 0, 0.00 at net D-max).
    """
    curve = model.developed_curve
    lo = curve.d_min if d_min is None else d_min
    hi = max(curve.d_max if d_max is None else d_max, lo + 1e-3)
    nets = (hi - lo) * np.arange(levels) / (levels - 1)
    fractions = np.array([model.bin_fractions(model.log_exposure_for(float(net))) for net in nets])

    silver = model.is_silver
    radii = [b.cloud_radius_mm for b in model.bins]
    fastest = next((r for r in radii if r > 0), TILE_TEXEL_MM)
    layers: list[Sublayer] = []
    for index, bin_ in enumerate(model.bins):
        if bin_.crystals_per_mm2 <= 0 or bin_.dye_per_cloud <= 0:
            continue
        edge = SILVER_GRAIN_EDGE if silver else 1.0
        if silver:
            smallest = (FASTEST_CRYSTAL_MM * bin_.cloud_radius_mm / fastest
                        / (2.0 * np.sqrt(2.0 * np.log(edge / np.log(2.0)))))
        else:
            smallest = DYE_CLOUD_DECAY_MM
        cell = float(min(max(np.sqrt(8.0 / bin_.crystals_per_mm2), 0.00025), 0.004))
        layer = Sublayer(
            silver=silver,
            coated_per_mm2=bin_.crystals_per_mm2,
            sigma_mm=smallest,
            peak_demand=0.0,
            capacity=SILVER_GRAIN_DENSITY if silver else max(bin_.pool, 0.05),
            dye_per_cloud_mm2=bin_.dye_per_cloud,
            cell_mm=cell,
            forming=fractions[:, index].astype(np.float64),
            void_integral_mm2=0.0,
            smallest_sigma_mm=smallest,
        )
        layers.append(_shape(layer))
    return FilmRecord(d_min=lo, d_max=hi, sublayers=tuple(layers))


# ---------------------------------------------------------------------------------------------
# Laying one level
# ---------------------------------------------------------------------------------------------
def _developed(layer: Sublayer, gross: float, record: FilmRecord, seed: int, period_mm: float,
               pixel_mm: float, sublayer_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The crystals of one sublayer that develop at ``gross``: position and mark.

    Every crystal draws the same numbers whether or not it develops, so the coating is one film
    under any exposure.  Positions are in samples (sample ``i`` is centred at ``i + 0.5``).
    """
    cells = max(int(round(period_mm / layer.cell_mm)), 1)
    cell = period_mm / cells
    rng = np.random.default_rng([seed, sublayer_index])
    counts = rng.poisson(layer.coated_per_mm2 * cell * cell, size=(cells, cells))
    total = int(counts.sum())
    empty = (np.empty(0), np.empty(0), np.empty(0))
    if total == 0:
        return empty

    # A coarse-grained stock can coat tens of millions of crystals in one period, and the cheap
    # thing to write -- five full-length float64 arrays plus a (shape, total) mark block -- peaks
    # above 2 GB.  On a 16 GB machine with the app running that is an allocation failure, not a
    # slow path (measured: `Unable to allocate 868 MiB for (4, 28437706)`).  So: int32 for the
    # cell indices, and the positions are only computed for the crystals that actually develop.
    flat = np.repeat(np.arange(counts.size, dtype=np.int32), counts.ravel())
    offset_x = rng.random(total, dtype=np.float64)
    offset_y = rng.random(total, dtype=np.float64)
    develop = rng.random(total, dtype=np.float64)
    keep = develop < layer.fraction_at(gross, record.d_min, record.d_max)
    if not np.any(keep):
        return empty
    del develop

    cy, cx = np.divmod(flat[keep], cells)
    del flat
    px = ((cx + offset_x[keep]) * cell) / pixel_mm - 0.5
    py = ((cy + offset_y[keep]) * cell) / pixel_mm - 0.5

    # ``mark`` is the mean of `shape` gamma draws per crystal.  Drawn as one (shape, total) block
    # it is the largest single allocation here; drawn row by row in the same order it is the same
    # stream -- numpy fills C-order -- so the numbers are bit-identical and the peak is one row.
    shape = int(round(1.0 / (MARK_DISPERSION * MARK_DISPERSION)))
    keep_at = np.flatnonzero(keep)
    kept = keep_at.size
    mark = np.zeros(kept, dtype=np.float64)
    step = max(1, _MARK_BLOCK // 8)              # crystals drawn per slice of a row
    for _row in range(shape):
        for start in range(0, total, step):
            stop = min(start + step, total)
            block = rng.random(stop - start)
            np.maximum(block, 1e-7, out=block)
            np.log(block, out=block)
            # Only the kept slice of this row is needed; the rest is still drawn to stay in stream.
            lo = np.searchsorted(keep_at, start, side="left")
            hi = np.searchsorted(keep_at, stop, side="left")
            if hi > lo:
                mark[lo:hi] += block[keep_at[lo:hi] - start]
    mark /= -shape                             # -mean(log u): the gamma mark, mean 1
    return px, py, mark


def _splat(index: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(index, weights=weights, minlength=size).astype(np.float64)


def _lay(layer: Sublayer, gross: float, record: FilmRecord, seed: int, period_mm: float,
         width: int, height: int, pixel_mm: float, sublayer_index: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """One sublayer's demand and its uncovered share, on the supersampled grid.

    A cloud two samples wide is point-sampled and blurred.  One narrower than that would fall
    between samples -- a sub-micron silver grain at a 1 µm texel -- so it lays the *area* it fills
    instead, and the samples it covers are at capacity: that is what stops a grain smaller than a
    pixel from reading as an impossibly dense point.  Between the two it lays both, weighted, so a
    cloud growing through the sample size changes the film smoothly.
    """
    size = width * height
    demand = np.zeros(size)
    log_uncovered = np.zeros(size)
    px, py, mark = _developed(layer, gross, record, seed, period_mm, pixel_mm, sublayer_index)
    if px.size == 0:
        return demand.reshape(height, width), log_uncovered.reshape(height, width)

    resolved = float(min(max((layer.sigma_mm / pixel_mm - 0.5) / 1.5, 0.0), 1.0))
    amounts = layer.peak_demand * mark / max(layer.capacity, 1e-6)   # demand over capacity

    if resolved > 0:
        ix = np.floor(px)
        iy = np.floor(py)
        ax = px - ix
        ay = py - iy
        # int32: the tile is 256², so every index fits — and a 38M-crystal population's
        # per-corner index arrays stay half the size, which is the difference between
        # fitting in what an 8 GB box has free and a MemoryError (superia_200).
        ix0 = np.mod(ix.astype(np.int32), width)
        iy0 = np.mod(iy.astype(np.int32), height)
        ix1 = np.mod(ix0 + 1, width)
        iy1 = np.mod(iy0 + 1, height)
        vals = amounts * resolved
        # Chunked: a 38M-crystal level's corner arrays are ~4 GB together in float64; in
        # 8M-crystal blocks the peak stays under ~700 MB and the result is bit-identical
        # (the splat is a pure per-crystal scatter-add).
        for lo in range(0, vals.size, _SPLAT_BLOCK):
            hi = min(lo + _SPLAT_BLOCK, vals.size)
            for xs, wx in ((ix0[lo:hi], 1.0 - ax[lo:hi]), (ix1[lo:hi], ax[lo:hi])):
                for ys, wy in ((iy0[lo:hi], 1.0 - ay[lo:hi]), (iy1[lo:hi], ay[lo:hi])):
                    demand += _splat(ys * width + xs, vals[lo:hi] * wx * wy, size)

    covering = 1.0 - resolved
    if covering > 0:
        # The square of film one crystal fills, in samples, centred on the crystal.  Rather than
        # rasterising every square's exact overlap with every sample it touches -- 25M crystals at
        # the top of a Tri-X curve -- lay the covered *area* and blur it by a Gaussian with the
        # square's own second moment.  At the 3x supersample and through a 48 µm aperture the two
        # are indistinguishable, and the tile stays a fraction of a second per level.
        grid = np.linspace(float(mark.min()), float(mark.max()), 48)
        occupied = np.array([_occupied_area(layer.edge * float(g), layer.silver) for g in grid])
        side = layer.sigma_mm * np.sqrt(np.interp(mark, grid, occupied)) / pixel_mm
        area = covering * side * side
        index = (np.mod(np.floor(py).astype(np.int64), height) * width
                 + np.mod(np.floor(px).astype(np.int64), width))
        covered = _splat(index, area, size)
        if covered.any():
            weight = float(np.sum(area))
            sigma = float(np.sqrt(np.sum(area * side * side) / max(weight, 1e-30)) / np.sqrt(12.0))
            covered = _blur(covered.reshape(height, width), sigma)
            # What each sample keeps clear.  One crystal covering a sample whole leaves nothing, so
            # the share is `1 - covered` rather than `exp(-covered)`; overlapping squares are the
            # second-order case the reference's product-of-shares treats exactly and this does not.
            log_uncovered = np.log(np.maximum(1.0 - covered, 1e-6))

    return demand.reshape(height, width), log_uncovered.reshape(height, width)


def build_level(record: FilmRecord, gross: float, seed: int = 1, side: int = TILE_SIDE,
                supersample: int | None = None) -> np.ndarray:
    """Light through the tile at one gross density, ``side²`` texels.

    The supersampled grid is averaged in transmittance, then ``-log10``: a pixel counts the light
    its patch of film passes, never a mean of densities.
    """
    pitch = TILE_TEXEL_MM
    s = supersample or supersample_for(pitch)
    h = pitch / s
    sw = sh = side * s
    period = side * pitch

    log_transmit = np.zeros((sh, sw))
    for index, layer in enumerate(record.sublayers):
        demand, log_uncovered = _lay(layer, gross, record, seed, period, sw, sh, h, index)
        uncovered = np.exp(np.minimum(log_uncovered, 0.0))
        resolved_dye = layer.capacity * (1.0 - np.exp(-_spread(demand, layer, h)))
        full = np.exp(-np.log(10.0) * layer.capacity)
        transmit = (uncovered * np.exp(-np.log(10.0) * resolved_dye)
                    + (1.0 - uncovered) * full)
        log_transmit += np.log(np.maximum(transmit, 1e-12))

    transmit = np.exp(log_transmit).reshape(side, s, side, s).mean(axis=(1, 3))
    return transmit


def _blur(field: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Periodic Gaussian blur of one field, in Fourier."""
    if sigma_samples <= 0:
        return field
    height, width = field.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    transfer = np.exp(-2.0 * np.pi ** 2 * sigma_samples ** 2 * (fx * fx + fy * fy))
    return np.fft.irfft2(np.fft.rfft2(field) * transfer, s=(height, width))


def _spread(deposits: np.ndarray, layer: Sublayer, sample_mm: float) -> np.ndarray:
    """Blur the deposits by the cloud's terms -- one separable Gaussian sum, in Fourier."""
    height, width = deposits.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    terms = (((layer.sigma_mm, 1.0),) if layer.silver
             else tuple((term * layer.sigma_mm, weight) for term, weight in DYE_CLOUD_TERMS))
    transfer = np.zeros((height, width // 2 + 1), dtype=np.float64)
    for sigma, weight in terms:
        sigma_samples = sigma / sample_mm
        # The kernel is `weight · exp(-r²/2σ²)` in samples, so its discrete transform carries the
        # 2πσ² that turns a crystal's peak demand into the mass a blur spreads over the film.
        transfer += (weight * 2.0 * np.pi * sigma_samples ** 2
                     * np.exp(-2.0 * np.pi ** 2 * sigma_samples ** 2 * (fx * fx + fy * fy)))
    return np.fft.irfft2(np.fft.rfft2(deposits) * transfer, s=(height, width))


# ---------------------------------------------------------------------------------------------
# Reading the tile the way a microdensitometer does
# ---------------------------------------------------------------------------------------------
def _block_sharing(max_lag: int = TILE_BLOCK_TEXELS) -> np.ndarray:
    """Covariance two points a lag apart keep through the blocks' cross-faded seams."""
    block, seam, samples = TILE_BLOCK_TEXELS, TILE_SEAM_TEXELS, 16

    def weight(c: np.ndarray, b: np.ndarray) -> np.ndarray:
        start = b * block
        return (np.clip((c - start + seam) / (2 * seam), 0.0, 1.0)
                * np.clip((start + block + seam - c) / (2 * seam), 0.0, 1.0))

    table = np.empty(max_lag + 1)
    positions = (np.arange(block * samples) + 0.5) / samples
    for lag in range(max_lag + 1):
        c = positions
        d = c + lag
        last = 1 + (lag + block - 1) // block
        shared = np.zeros_like(c)
        norm_c = np.zeros_like(c)
        norm_d = np.zeros_like(c)
        for b in range(-1, last + 1):
            wc = weight(c, b)
            wd = weight(d, b)
            shared += wc * wd
            norm_c += wc * wc
            norm_d += wd * wd
        table[lag] = float(np.mean(shared / np.sqrt(norm_c * norm_d)))
    return table


_SHARING = None


def sigma48(light: np.ndarray, reach: int = 16) -> float:
    """RMS granularity through the 48 µm aperture, read as a microdensitometer reads it.

    The aperture keeps the light field's noise power at zero frequency over its area -- the
    autocovariance summed over every lag it reaches -- and the density's is that over the light's
    mean, by ``ln 10``.  Frames lay the tile in blocks at independent offsets, cross-faded at
    their seams, so two points a lag apart keep only ``sharing(dx) · sharing(dy)`` of their
    covariance.
    """
    global _SHARING
    if _SHARING is None:
        _SHARING = _block_sharing()
    mean = float(light.mean())
    d = light - mean
    n = d.shape[0]
    spectrum = np.fft.rfft2(d)
    covariance = np.fft.irfft2(spectrum * np.conj(spectrum), s=d.shape) / (n * n)
    # covariance[dy, dx] for non-negative dy/dx; the field is real, so signs do not matter.
    power = 0.0
    for dy in range(reach + 1):
        share_y = _SHARING[dy]
        for dx in range(reach + 1):
            share = share_y * _SHARING[dx] * (1 if dx == 0 or dy == 0 else 1)
            weight = share * (1 if (dx == 0 and dy == 0) else 2 if (dx == 0 or dy == 0) else 4)
            power += float(covariance[dy, dx]) * weight
    radius = APERTURE_RADIUS_MM / TILE_TEXEL_MM
    light_variance = max(power, 0.0) / (np.pi * radius * radius)
    return float(np.sqrt(light_variance) / (mean * np.log(10.0)))


def anchor_to_sheet(record: FilmRecord, gross: float, target_sigma: float, seed: int = 1,
                    side: int = TILE_SIDE, passes: int = ANCHOR_PASSES,
                    tolerance: float = ANCHOR_TOLERANCE, verbose: bool = False) -> FilmRecord:
    """Scale each crystal's dye until the tile reads the sheet's 48 µm granularity.

    The population's dye per crystal was set so that an *additive* density field reads the sheet.
    Clouds that saturate and light that averages read differently, so the anchor is taken again on
    the model's own render: a flat patch at the read density, read through the 48 µm aperture.
    How far the reading moves with the dye depends on how full the clouds are, so each step takes
    the slope the last one measured.
    """
    if target_sigma <= 0 or not record.sublayers:
        return record
    solved = 1.0
    slope = 0.5
    last: tuple[float, float] | None = None
    current = record
    best = record                                   # the closest reading so far
    best_miss = np.inf
    for _ in range(passes):
        measured = sigma48(build_level(current, gross, seed=seed, side=side))
        if measured <= 0:
            break
        miss = float(np.log(target_sigma / measured))
        if verbose:
            print("  anchor: measured %.5f, target %.5f, factor %.4f"
                  % (measured, target_sigma, solved))
        # Where the clouds have merged, sigma is not monotone in the dye: a step can overshoot and
        # the next one overshoot back, and the loop leaves the last record -- which may be much
        # further from the sheet than one it passed through.  Keep the closest instead.
        if abs(miss) < best_miss:
            best, best_miss = current, abs(miss)
        if abs(miss) < tolerance:
            return current
        if last is not None and abs(np.log(solved / last[0])) > 1e-3:
            slope = float(min(max(np.log(measured / last[1]) / np.log(solved / last[0]), 0.25), 1.5))
        last = (solved, measured)
        step = float(np.exp(miss / slope))
        current = replace(current, sublayers=tuple(layer.with_dye(step)
                                                   for layer in current.sublayers))
        solved *= step
    return best


def build_levels(record: FilmRecord, grosses: Sequence[float] | None = None, seed: int = 1,
                 side: int = TILE_SIDE) -> list[np.ndarray]:
    levels = record.grosses() if grosses is None else list(grosses)
    return [build_level(record, gross, seed=seed, side=side) for gross in levels]
