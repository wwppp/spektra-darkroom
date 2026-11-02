"""The crystal population a record coats, read off its characteristic curve.

Ported from ``Sources/FotufilmCore/CrystalGrainModel.swift``.  The port is deliberately literal:
the numbers this module produces (cloud radii, crystals per mm², dye per cloud, the 48 µm
granularity) are the ones the reference tiles are built from, so a "close enough" rewrite would
have made the P2 comparison meaningless.

What the model says
-------------------
**Exposure** picks which crystals develop.  A record is a population of crystals, each with a
speed -- the exposure at which it absorbs one photon on average -- and a crystal develops once a
Poisson count of absorbed photons reaches three.  The count is what the curve is: a ladder of 36
speed classes whose threshold-hit responses, weighted non-negatively, add up to the record's
density above base.  Speed goes with volume, so the same ladder places each class on a size
ladder ``r ∝ 10^(-E0/3)``, clipped to the stock's authored span.

**Development** turns latent crystals into what is seen.  A developed crystal forms dye in
proportion to its projected area, ``d ∝ r²``, with a two-point ±50 % mass dispersion.  The
classes are coated as four sublayers from the fastest down, using the stock's target shares,
each with its own finite coupler pool: with ``demand`` the dye its crystals ask for, the dye that
forms is ``C (1 - exp(-demand / C))``, sized so a fully developed sublayer's next crystal forms
a tenth of what an unstarved one would.  That starvation is what makes granularity fall past its
peak; silver has no pool.

**The sheet anchors the scale.**  The published 48 µm RMS granularity at its read density fixes
how many crystals the record coats per mm² and how much dye each forms; ``grainSizeMM`` fixes the
cloud.  Nothing in this file is fitted to an image.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
from scipy.special import i0e, i1e

from .curve import Curve

__all__ = [
    "CrystalGrainPopulation",
    "CrystalPopulation",
    "GrainStock",
    "BIN_COUNT",
    "APERTURE_RADIUS_MM",
    "granularity_aperture_response",
]

BIN_COUNT = 4
APERTURE_RADIUS_MM = 0.024                     # the standard 48 µm microdensitometer aperture
FIT_LOG_EXPOSURES = np.arange(-4.0, 4.0 + 1e-9, 0.05)
CLASS_LOG_EXPOSURES = np.arange(-3.0, 4.0 + 1e-9, 0.2)
HIT_THRESHOLD = 3
SIZE_EXPONENT = 1.0 / 3.0
POOL_GAIN = 0.1                                # an unstarved crystal, once its pool is gone
DYE_EXPONENT = 2.0
MARK_DISPERSION = 0.5
MARK_SECOND_MOMENT = 1.0 + MARK_DISPERSION * MARK_DISPERSION
DEFAULT_GRAIN_FOG_DENSITY = 0.03


def granularity_aperture_response(clump_sigma_mm: float) -> float:
    """Fraction of a Gaussian clump field's granularity the 48 µm aperture keeps.

    Exact for a Gaussian clump and a circular aperture: ``sqrt(1 - e^-z (I0(z) + I1(z)))``
    with ``z = R² / (2σ²)``.  Approaches 1 as the clump shrinks.
    """
    if clump_sigma_mm <= 0:
        return 1.0
    z = APERTURE_RADIUS_MM * APERTURE_RADIUS_MM / (2.0 * clump_sigma_mm * clump_sigma_mm)
    lost = float(i0e(z) + i1e(z))
    return float(np.sqrt(max(1.0 - lost, 1e-4)))


@dataclass(frozen=True)
class CrystalGrainPopulation:
    """Stock-authoring parameters for the crystal population.

    ``radius_span``  maximum radius ratio on the class ladder (1 = equal radii).
    ``sublayer_shares``  target fractions of fitted density weight, fastest to slowest.
    ``coating_density_scale``  effective crystal count relative to the RMS-anchored population;
    each crystal then forms inversely as much dye, so mean demand and pool capacity hold.
    """

    radius_span: float = 6.0
    sublayer_shares: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    coating_density_scale: float = 1.0

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "CrystalGrainPopulation":
        if not data:
            return cls()
        return cls(radius_span=float(data["radiusSpan"]),
                   sublayer_shares=tuple(float(s) for s in data["sublayerShares"]),
                   coating_density_scale=float(data["coatingDensityScale"]))


@dataclass(frozen=True)
class GrainStock:
    """Everything the population needs from one stock, and nothing else.

    Built from a fotufilm pack by ``from_fotufilm_json`` (dev tooling and the sync script) and,
    at runtime, from ``resources/profiles/grain_params.json``.  Only the fields below matter to
    the population; the rest of a stock is the rest of the pipeline's business.
    """

    stock_id: str
    curves: tuple[Curve, ...]
    grain_size_mm: float
    grain_strength: float
    grain_layer_weights: tuple[float, ...]
    grain_layer_size_ratio: tuple[float, ...] = (1.0, 1.0, 1.0)
    population: CrystalGrainPopulation = field(default_factory=CrystalGrainPopulation)
    is_monochrome: bool = False
    is_reversal: bool = False
    is_silver: bool = False
    grain_fog_density: float = DEFAULT_GRAIN_FOG_DENSITY
    granularity_read_density: str = "net"      # "net" (1 above base) or "gross"
    luma_correlation: float = 0.0

    # -- derived ----------------------------------------------------------------------------
    def layer_weight(self, layer: int) -> float:
        weights = self.grain_layer_weights
        return float(weights[layer]) if layer < len(weights) else 1.0

    def layer_size_ratio(self, layer: int) -> float:
        ratios = self.grain_layer_size_ratio
        return float(ratios[layer]) if layer < len(ratios) else 1.0

    def granularity_anchor_density(self, layer: int) -> float:
        """Density above base the published granularity was read at."""
        curve = self.curves[layer]
        net = 1.0 - curve.d_min if self.granularity_read_density == "gross" else 1.0
        density_range = curve.d_max - curve.d_min
        if density_range <= 0:
            return 0.5
        return float(min(max(net, 0.05 * density_range), 0.9 * density_range))

    # -- loading ----------------------------------------------------------------------------
    @classmethod
    def from_fotufilm_json(cls, data: Mapping[str, Any]) -> "GrainStock":
        law = data.get("grainDensityLaw")
        monochrome = bool(data.get("isMonochrome", False))
        reversal = bool(data.get("isReversal", False))
        silver = (law == "silver") if law else monochrome
        ratios = data.get("grainLayerSizeRatio") or [1.0, 1.0, 1.0]
        return cls(
            stock_id=str(data.get("id") or data.get("name") or ""),
            curves=tuple(Curve.from_json(c) for c in data["curves"]),
            grain_size_mm=float(data["grainSizeMM"]),
            grain_strength=float(data["grainStrength"]),
            grain_layer_weights=tuple(float(w) for w in data["grainLayerWeights"]),
            grain_layer_size_ratio=tuple(float(r) for r in ratios),
            population=CrystalGrainPopulation.from_json(data.get("crystalGrainPopulation")),
            is_monochrome=monochrome,
            is_reversal=reversal,
            is_silver=silver,
            grain_fog_density=float(data.get("grainFogDensity", DEFAULT_GRAIN_FOG_DENSITY)),
            granularity_read_density=("gross" if data.get("granularityReadDensity") == "gross"
                                      else "net"),
            luma_correlation=float(data.get("grainLumaCorrelation") or 0.0),
        )


@dataclass(frozen=True)
class Bin:
    """One coated sublayer: the crystals of one size bin as this process develops them."""

    cloud_radius_mm: float
    dye_per_cloud: float
    crystals_per_mm2: float
    pool: float
    demand_at_full: float
    gain: float
    classes: tuple[int, ...]
    shares: tuple[float, ...]


def _saturated(forming: np.ndarray | float, gain: float) -> np.ndarray:
    """Dye one unit of weight forms when ``forming`` of its crystals form it through a pool."""
    if gain >= 1:
        return np.asarray(forming, dtype=np.float64)
    forming = np.asarray(forming, dtype=np.float64)
    return (1.0 - np.power(gain, forming)) / (1.0 - gain)


def _dye(demand: float, pool: float) -> float:
    return pool * (1.0 - np.exp(-demand / pool)) if pool > 0 else demand


def _hit_fraction(log_exposure: np.ndarray, class_log_exposure: np.ndarray) -> np.ndarray:
    """Fraction of a class developable: a Poisson count of ≥3 photons."""
    mean = np.power(10.0, log_exposure[:, None] - class_log_exposure[None, :])
    below = 1.0 + mean + 0.5 * mean * mean
    return np.clip(1.0 - np.exp(-mean) * below, 0.0, 1.0)


def _nnls(basis: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Non-negative least squares by projected coordinate descent on the normal equations.

    Deterministic, like the reference's, so every host packs the same population.
    """
    gram = basis.T @ basis
    moment = basis.T @ targets
    diagonal = np.diag(gram).copy()
    weights = np.zeros(basis.shape[1])
    for _ in range(5000):
        moved = 0.0
        for i in range(weights.size):
            if diagonal[i] <= 0:
                continue
            residual = moment[i] - (gram[i] @ weights) + gram[i, i] * weights[i]
            updated = max(residual / diagonal[i], 0.0)
            moved = max(moved, abs(updated - weights[i]))
            weights[i] = updated
        if moved < 1e-9:
            break
    return weights


def _fog_fractions(weights: np.ndarray, log_radius: np.ndarray, density: float) -> np.ndarray:
    """Fog forms at surface specks, so each class fogs with its crystals' surface, ``r²``."""
    largest = log_radius[weights > 0].max() if np.any(weights > 0) else 0.0
    surface = np.power(10.0, 2.0 * (log_radius - largest))

    def formed(scale: float) -> float:
        return float(np.sum(weights * np.minimum(scale * surface, 0.5)))

    if density <= 0 or formed(np.inf) <= 0:
        return np.zeros_like(weights)
    low, high = 0.0, 1.0
    while formed(high) < density and high < 1e9:
        high *= 2.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if formed(mid) < density:
            low = mid
        else:
            high = mid
    return np.minimum(high * surface, 0.5)


class CrystalPopulation:
    """One record of one stock, as developed: the four sublayers and what they form.

    ``model.sigma(net_density)`` is the 48 µm RMS granularity the sheet publishes, reproduced
    from the curve and the coating alone -- that is the check that the port is right.
    """

    bin_count = BIN_COUNT
    class_log_exposures = CLASS_LOG_EXPOSURES

    def __init__(self, stock: GrainStock, layer: int, reference: GrainStock | None = None,
                 grain_scale: float = 1.0) -> None:
        reference_stock = reference or stock
        population = reference_stock.population
        reference_curve = reference_stock.curves[layer]
        developed_curve = stock.curves[layer]
        reversal = stock.is_reversal
        silver = stock.is_silver
        gain = 1.0 if silver else POOL_GAIN
        kappa = 1.0 if gain >= 1 else -np.log(gain) / (1.0 - gain)

        self.stock = stock
        self.layer = layer
        self.developed_curve = developed_curve
        self.d_min = developed_curve.d_min
        self.range = max(developed_curve.d_max - developed_curve.d_min, 1e-3)
        self.is_reversal = reversal
        self.is_silver = silver

        classes = CLASS_LOG_EXPOSURES
        targets = np.asarray([reference_curve.density(x) - reference_curve.d_min
                              for x in FIT_LOG_EXPOSURES], dtype=np.float64)

        # Exposure: the population that forms the reference curve.  One unit of a class's weight
        # forms `saturated(hit fraction)` through its own pool at the reference process, so the
        # fit stays linear in the weights.
        hit = _hit_fraction(FIT_LOG_EXPOSURES, classes)
        forming = 1.0 - hit if reversal else hit
        dye = _saturated(forming, gain)
        basis = 1.0 - dye if reversal else dye
        weights = _nnls(basis, targets)
        worst = float(np.max(np.abs(basis @ weights - targets))) if targets.size else 0.0

        # Size from speed, over the span the record holds, centred on the weight's log-mean.
        log_radius = -SIZE_EXPONENT * classes
        active = np.nonzero(weights > 1e-6 * weights.max())[0] if weights.max() > 0 else np.array([], int)
        if active.size:
            mean = float(np.sum(weights[active] * log_radius[active]) / np.sum(weights[active]))
            half = np.log10(population.radius_span) / 2.0
            log_radius = np.clip(log_radius, mean - half, mean + half)

        # Development: sublayers along speed, fastest first, at the authored cumulative
        # density-weight boundaries.
        members: list[list[int]] = [[] for _ in range(BIN_COUNT)]
        if active.size:
            cumulative_share = 0.0
            cuts: list[float] = []
            for share in population.sublayer_shares[:-1]:
                cumulative_share += share
                cuts.append(cumulative_share)
            active_weight = float(np.sum(weights[active]))
            cumulative = 0.0
            for k in active:                          # ascending class index = descending speed
                cumulative += weights[k]
                fraction = cumulative / active_weight - 1e-6
                bin_index = next((i for i, cut in enumerate(cuts) if fraction < cut), BIN_COUNT - 1)
                members[min(bin_index, BIN_COUNT - 1)].append(int(k))

        bins: list[Bin] = []
        for ks in members:
            if not ks:
                continue
            weight = float(np.sum(weights[ks]))
            log_mean = float(np.sum(weights[ks] * log_radius[ks]) / weight)
            bins.append(Bin(cloud_radius_mm=float(10.0 ** log_mean), dye_per_cloud=0.0,
                            crystals_per_mm2=0.0, pool=0.0, demand_at_full=weight * kappa,
                            gain=1.0, classes=tuple(ks),
                            shares=tuple(float(weights[k] / weight) for k in ks)))

        # The sublayers are what render, each drawing on one pool at its own mean developed
        # fraction, and that is not quite the sum of its classes each drawing on their own.  So
        # refit the sublayer weights to the curve with their class shares held.
        if bins:
            fractions = _hit_fraction(FIT_LOG_EXPOSURES, classes)
            bin_basis = np.empty((FIT_LOG_EXPOSURES.size, len(bins)))
            for row, x in enumerate(FIT_LOG_EXPOSURES):
                for col, bin_ in enumerate(bins):
                    developed = float(sum(fractions[row, k] * s
                                          for k, s in zip(bin_.classes, bin_.shares)))
                    formed = 1.0 - developed if reversal else developed
                    value = _saturated(formed, gain)
                    bin_basis[row, col] = 1.0 - value if reversal else value
            refitted = _nnls(bin_basis, targets)
            for i, bin_ in enumerate(bins):
                before = bin_.demand_at_full / kappa
                scale = refitted[i] / before if before > 0 else 0.0
                bins[i] = replace(bin_, demand_at_full=float(refitted[i] * kappa))
                for k in bin_.classes:
                    weights[k] *= scale
            worst = float(np.max(np.abs(bin_basis @ refitted - targets))) if targets.size else 0.0

        # Empty bins carry nothing: a zero count table and a zero cloud, which the kernel adds
        # as nothing.
        while len(bins) < BIN_COUNT:
            bins.append(Bin(cloud_radius_mm=0.0, dye_per_cloud=0.0, crystals_per_mm2=0.0,
                            pool=0.0, demand_at_full=0.0, gain=1.0, classes=(), shares=()))

        # The fastest sublayer's cloud is the scan-resolved clump radius; the rest follow the
        # size ladder, and what each forms follows its area.
        anchor = (reference_stock.grain_size_mm * reference_stock.layer_size_ratio(layer)
                  * grain_scale)
        fastest = bins[0].cloud_radius_mm
        radii = [b.cloud_radius_mm for b in bins]
        for i, bin_ in enumerate(bins):
            if bin_.cloud_radius_mm <= 0:
                continue
            relative = radii[i] / fastest
            pool = 0.0 if silver else bin_.demand_at_full / -np.log(gain)
            bins[i] = replace(bin_, cloud_radius_mm=anchor * relative,
                              dye_per_cloud=float(relative ** DYE_EXPONENT), pool=pool,
                              crystals_per_mm2=bin_.demand_at_full / float(relative ** DYE_EXPONENT))

        self.weights = weights
        self.log_radius = log_radius
        self.bins = tuple(bins)
        self.fog_fractions = _fog_fractions(weights, log_radius, stock.grain_fog_density)
        self.developability = 1.0
        self.fit_error = worst

        # The sheet's figure fixes the dye per crystal of the reference size at the reference
        # process, and with it the crystal count every weight implies.
        self.read_net_density = reference_stock.granularity_anchor_density(layer)
        self.read_sigma = (reference_stock.grain_strength * reference_stock.layer_weight(layer)
                           * grain_scale)
        reference_model = self._with_curve(reference_curve)
        unit_variance = reference_model.aperture_variance(
            reference_model.log_exposure_for(reference_model.read_net_density))
        per_cloud = (self.read_sigma ** 2 / unit_variance) if unit_variance > 0 else 0.0
        self.bins = tuple(
            replace(b, dye_per_cloud=b.dye_per_cloud * per_cloud,
                    crystals_per_mm2=(b.demand_at_full / (b.dye_per_cloud * per_cloud)
                                      if b.dye_per_cloud * per_cloud > 0 else 0.0))
            if b.cloud_radius_mm > 0 else b for b in self.bins)

        if not self._same_curve(reference_curve, developed_curve):
            # A development condition away from the reference: the same coating, developed
            # further or less far.  P2/P5 fills this in; until then it must not be silent.
            raise NotImplementedError(
                "pushed/pulled development is not ported yet (P2); curves differ from reference")

        scale = population.coating_density_scale
        if scale != 1.0:
            self.bins = tuple(replace(b, crystals_per_mm2=b.crystals_per_mm2 * scale,
                                      dye_per_cloud=b.dye_per_cloud / scale)
                              for b in self.bins)
            self.read_sigma /= np.sqrt(scale)

    # -- shape ------------------------------------------------------------------------------
    def _with_curve(self, curve: Curve) -> "CrystalPopulation":
        clone = object.__new__(CrystalPopulation)
        clone.__dict__.update(self.__dict__)
        clone.developed_curve = curve
        clone.d_min = curve.d_min
        clone.range = max(curve.d_max - curve.d_min, 1e-3)
        return clone

    @staticmethod
    def _same_curve(a: Curve, b: Curve) -> bool:
        return bool(np.all(np.abs(np.asarray([a.density(x) for x in FIT_LOG_EXPOSURES])
                                  - np.asarray([b.density(x) for x in FIT_LOG_EXPOSURES])) < 1e-4))

    # -- what the record states -------------------------------------------------------------
    def forming_fractions(self, log_exposure: float) -> np.ndarray:
        """Dye-forming fraction of every class, fog included; a reversal inverts."""
        latent = self._latent_fraction(log_exposure)
        developed = 1.0 - (1.0 - latent) * (1.0 - self.fog_fractions)
        return 1.0 - developed if self.is_reversal else developed

    def _latent_fraction(self, log_exposure: float) -> np.ndarray:
        threshold = _hit_fraction(np.array([log_exposure]), CLASS_LOG_EXPOSURES)[0]
        developability = self.developability
        if developability <= 1.0:
            return threshold * max(developability, 0.0)
        mean = np.power(10.0, log_exposure - CLASS_LOG_EXPOSURES)
        short = np.exp(-mean) * (mean * mean / 2.0)       # exactly threshold-1 = 2 hits
        return np.minimum(threshold + (developability - 1.0) * short, 1.0)

    def bin_fractions(self, log_exposure: float) -> np.ndarray:
        fractions = self.forming_fractions(log_exposure)
        return np.array([sum(fractions[k] * s for k, s in zip(b.classes, b.shares))
                         for b in self.bins], dtype=np.float64)

    def mean_density(self, log_exposure: float) -> float:
        fractions = self.bin_fractions(log_exposure)
        total = sum(_dye(b.demand_at_full * float(f), b.pool)
                    for b, f in zip(self.bins, fractions))
        return float(total)

    def aperture_variance(self, log_exposure: float) -> float:
        """Density fluctuation variance through the 48 µm aperture, in density²."""
        fractions = self.bin_fractions(log_exposure)
        aperture_area = np.pi * APERTURE_RADIUS_MM * APERTURE_RADIUS_MM
        total = 0.0
        for b, fraction in zip(self.bins, fractions):
            if b.cloud_radius_mm <= 0 or b.dye_per_cloud <= 0:
                continue
            demand = b.demand_at_full * float(fraction)
            pool_gain = np.exp(-demand / b.pool) if b.pool > 0 else 1.0
            developed = b.crystals_per_mm2 * float(fraction)
            asked = developed * b.dye_per_cloud * b.dye_per_cloud * MARK_SECOND_MOMENT
            response = granularity_aperture_response(b.cloud_radius_mm / 2.0)
            total += pool_gain * pool_gain * asked * response * response / aperture_area
        return float(total)

    def log_exposure_for(self, net_density: float) -> float:
        target = (self.d_min + self.range - net_density if self.is_reversal
                  else self.d_min + net_density)
        clamped = min(max(target, self.d_min), self.d_min + self.range)
        return self.developed_curve.log_exposure(clamped)

    def sigma(self, net_density: float) -> float:
        return float(np.sqrt(self.aperture_variance(self.log_exposure_for(net_density))))

    def report(self) -> str:
        lines = ["fit %.3f D, fog %.4f…%.4f, developability %.2f, read net %.2f D → σ48 %.4f"
                 % (self.fit_error,
                    min((s for w, s in zip(self.weights, self.fog_fractions) if w > 0), default=0),
                    float(self.fog_fractions.max()), self.developability,
                    self.read_net_density, self.read_sigma)]
        for index, b in enumerate(self.bins):
            if b.cloud_radius_mm <= 0:
                continue
            speeds = [CLASS_LOG_EXPOSURES[k] for k in b.classes]
            lines.append("sublayer %d: cloud radius %.2f µm, dye per cloud %.4f D·µm², "
                         "%.2f crystals/µm², pool %.2f D, demand at full %.2f D, gain %.2f, "
                         "speed %+.1f…%+.1f"
                         % (index, b.cloud_radius_mm * 1000, b.dye_per_cloud * 1e6,
                            b.crystals_per_mm2 * 1e-6, b.pool, b.demand_at_full, b.gain,
                            min(speeds), max(speeds)))
        return "\n".join(lines)


def load_fotufilm_stock(path: str) -> GrainStock:
    """Dev-only: read a fotufilm pack straight into the fields the population needs."""
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return GrainStock.from_fotufilm_json(json.load(handle))
