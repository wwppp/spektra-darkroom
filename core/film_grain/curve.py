"""Hurter--Driffield characteristic curve, in the shape fotufilm-engine packs it.

The crystal population is *read off the curve*: the speed-class weights are a non-negative
least-squares fit whose basis is what one unit of a class's weight forms, and the targets are
the record's density above base on the fit grid.  So the curve has to be evaluated and inverted
exactly as the reference does it, or the fitted population drifts.

Two forms, both ported from ``Sources/FotufilmCore/CharacteristicCurve.swift`` and
``SampledCharacteristicCurve.swift``:

* **sampled** -- a table of (log exposure, density) pairs joined by monotone cubic Hermite
  segments (Fritsch--Carlson tangents).  Every stock in this pack is sampled; the analytic form
  is kept because a profile may carry one and because the tests use it.
* **analytic** -- softplus toe and shoulder, plus an optional second speed group.

Densities are gross (base included).  ``density(x) - d_min`` is what the population fit sees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["Curve", "CurveComponent", "SampledCurve"]


def _softplus(x: np.ndarray | float) -> np.ndarray:
    return np.logaddexp(0.0, x)


@dataclass(frozen=True)
class SampledCurve:
    """Monotone-cubic sampled curve with strictly increasing exposures."""

    log_exposure: np.ndarray
    density: np.ndarray
    slopes: np.ndarray

    @classmethod
    def from_points(cls, log_exposure: Sequence[float],
                    density: Sequence[float]) -> "SampledCurve":
        x = np.asarray(log_exposure, dtype=np.float64)
        y = np.asarray(density, dtype=np.float64)
        if x.ndim != 1 or x.size < 2 or x.size != y.size:
            raise ValueError("sampled curve needs >=2 finite (exposure, density) pairs")
        if not np.all(np.diff(x) > 0):
            raise ValueError("sampled curve exposures must strictly increase")
        # Fritsch--Carlson tangents: the harmonic mean of the neighbouring secants, zero at a
        # local extremum.  Same arithmetic as the reference, in float64.
        slopes = np.zeros_like(y)
        if y.size > 2:
            h0 = x[1:-1] - x[:-2]
            h1 = x[2:] - x[1:-1]
            d0 = (y[1:-1] - y[:-2]) / h0
            d1 = (y[2:] - y[1:-1]) / h1
            same = d0 * d1 > 0
            w0 = 2 * h1 + h0
            w1 = h1 + 2 * h0
            with np.errstate(divide="ignore", invalid="ignore"):
                tangent = (w0 + w1) / (w0 / d0 + w1 / d1)
            slopes[1:-1] = np.where(same, tangent, 0.0)
        if not np.all(np.isfinite(slopes)):
            raise ValueError("sampled curve tangents must be finite")
        return cls(x, y, slopes)

    def value(self, x: np.ndarray | float) -> np.ndarray:
        xs = np.asarray(x, dtype=np.float64)
        out = np.empty_like(xs)
        flat = xs.reshape(-1)
        result = out.reshape(-1)
        x0 = self.log_exposure
        y0 = self.density
        m = self.slopes
        low = np.clip(np.searchsorted(x0, flat, side="right") - 1, 0, x0.size - 2)
        xa, xb = x0[low], x0[low + 1]
        ya, yb = y0[low], y0[low + 1]
        ma, mb = m[low], m[low + 1]
        h = xb - xa
        t = np.clip((flat - xa) / h, 0.0, 1.0)
        delta = yb - ya
        a = h * ma
        b = h * mb
        hermite = ya + t * (a + t * (3 * delta - 2 * a - b + t * (-2 * delta + a + b)))
        result[...] = np.where(flat <= x0[0], y0[0],
                               np.where(flat >= x0[-1], y0[-1], hermite))
        return out


@dataclass(frozen=True)
class CurveComponent:
    """A second coated speed group, additive on top of the main population."""

    gamma: float
    toe: float
    toe_width: float
    shoulder: float
    shoulder_width: float

    def density(self, x: np.ndarray | float) -> np.ndarray:
        xs = np.asarray(x, dtype=np.float64)
        t = self.toe_width * _softplus((xs - self.toe) / self.toe_width)
        s = self.shoulder_width * _softplus((xs - self.shoulder) / self.shoulder_width)
        return self.gamma * np.clip(t - s, 0.0, self.shoulder - self.toe)

    @property
    def density_range(self) -> float:
        return self.gamma * (self.shoulder - self.toe)


@dataclass(frozen=True)
class Curve:
    """One record's density against log exposure (gross, base included)."""

    d_min: float
    gamma: float = 0.0
    toe: float = 0.0
    toe_width: float = 1.0
    shoulder: float = 1.0
    shoulder_width: float = 1.0
    secondary: CurveComponent | None = None
    sampled: SampledCurve | None = None

    # -- shape ------------------------------------------------------------------------------
    @property
    def d_max(self) -> float:
        if self.sampled is not None:
            return float(self.sampled.density.max())
        return (self.d_min + self.gamma * (self.shoulder - self.toe)
                + (self.secondary.density_range if self.secondary else 0.0))

    @property
    def log_exposure_span(self) -> tuple[float, float]:
        if self.sampled is not None:
            return float(self.sampled.log_exposure[0]), float(self.sampled.log_exposure[-1])
        return self.toe - 6.0, self.shoulder + 6.0

    def density(self, log_exposure: np.ndarray | float) -> np.ndarray:
        xs = np.asarray(log_exposure, dtype=np.float64)
        if self.sampled is not None:
            return self.sampled.value(xs)
        t = self.toe_width * _softplus((xs - self.toe) / self.toe_width)
        s = self.shoulder_width * _softplus((xs - self.shoulder) / self.shoulder_width)
        out = (self.d_min
               + self.gamma * np.clip(t - s, 0.0, self.shoulder - self.toe))
        if self.secondary is not None:
            out = out + self.secondary.density(xs)
        return out

    # -- inversion --------------------------------------------------------------------------
    def log_exposure(self, target: float) -> float:
        """An exposure that forms ``target`` gross density.

        Sampled curves keep local extrema, so more than one exposure can form a density; the
        reference takes the first rising crossing, and clamps to the peak when the requested
        density is unattainable.  Bisection for 60 steps, exactly as the reference does it.
        """
        lo, hi = self.log_exposure_span
        sampled = self.sampled
        if sampled is not None and target > float(sampled.density[-1]):
            crossing = np.nonzero(sampled.density >= target)[0]
            if crossing.size == 0:
                peak = int(np.argmax(sampled.density))
                return float(sampled.log_exposure[peak])
            upper = int(crossing[0])
            if upper == 0:
                return float(sampled.log_exposure[0])
            lo = float(sampled.log_exposure[upper - 1])
            hi = float(sampled.log_exposure[upper])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(self.density(mid)) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # -- loading ----------------------------------------------------------------------------
    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Curve":
        secondary = data.get("secondary")
        sampled = data.get("sampled")
        return cls(
            d_min=float(data["dMin"]),
            gamma=float(data.get("gamma", 0.0)),
            toe=float(data.get("toe", 0.0)),
            toe_width=float(data.get("toeWidth", 1.0)),
            shoulder=float(data.get("shoulder", 1.0)),
            shoulder_width=float(data.get("shoulderWidth", 1.0)),
            secondary=(CurveComponent(
                gamma=float(secondary["gamma"]), toe=float(secondary["toe"]),
                toe_width=float(secondary["toeWidth"]), shoulder=float(secondary["shoulder"]),
                shoulder_width=float(secondary["shoulderWidth"])) if secondary else None),
            sampled=(SampledCurve.from_points(sampled["logExposure"], sampled["density"])
                     if sampled else None),
        )
