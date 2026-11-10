"""fotufilm's halation kernel, ported line for line.

This module answers one question and nothing else: *given a stock's support
geometry and its returned-light profile, where does the halo land and how much
of it does each colour record get?* It is pure numpy, has no Qt, no GL, and no
idea what a film stock is called -- it takes the numbers and returns numbers.

The reference is ``fotufilm-engine/Sources/FotufilmCore/Halation.swift``. Where
this file and that one disagree, that one is right; the port is deliberately
literal (same quadrature, same 28-step bisection, same 14-candidate grid, same
simplex fit) so that a disagreement is a bug and not a "taste".

Two paths, both from the engine:

* **PhysicalProfile** (stock ships a ``profile`` block). Every bounce and every
  angle contributes a ray with its own radius ``2·m·d·tan(theta)`` and its own
  Fresnel/absorption weight. That edge-spread is then fitted with three shared
  centred Gaussians.
* **Profile** (no ``profile`` block -- 15 of our 54 stocks). The engine falls
  back to inverting ``returnedFraction`` to a round-trip transmittance, tabulating
  the same rays, and placing the three Gaussians at ``anchor/1.7, anchor,
  anchor*1.7`` where ``anchor`` is the sigma whose halo falls to half at the
  radius where the real one does.

Why three shared sigmas and not one per record: the renderer has three blur
passes, and a positive mixture of centred Gaussians is monotone with no ring
boundaries. Only the *weights* differ per record, and that asymmetry is exactly
what gives the halo its colour (CineStill's red halo, the neutral halo of a
monochrome stock).

The output ``matrix`` is the piece we have never used before: the engine routes
each receiver's share across the *source* records through the sheet's spectral
return matrix, so a halo is not obliged to be the same colour as the light that
caused it.

Everything here is cached by value -- building a kernel is ~0.1 s, which is fine
once per stock but not once per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# --- engine constants (Halation.swift) -------------------------------------------
SCALE_COUNT = 3
SCALE_RATIO = 1.7
BOUNCE_COUNT = 5
ANGLE_SAMPLES = 256

#: Sigma of the Gaussian whose halo is half value at the same radius as the real
#: one: ``1 / 0.6744897`` where 0.6744897 is the 75th percentile of a unit normal.
_HALF_WIDTH_TO_SIGMA = 1.0 / 0.6744897

SOURCE_PROFILE = "profile"
SOURCE_DERIVED = "derived"


# --------------------------------------------------------------------- optics
def _polarized_reflectance(cos_theta: np.ndarray, n: float):
    """Fresnel s/p reflectance of the base/air interface, seen from inside the base."""
    cos_in = np.clip(cos_theta, 0.0, 1.0)
    sin_in = np.sqrt(np.maximum(1.0 - cos_in * cos_in, 0.0))
    sin_out = n * sin_in
    blocked = sin_out >= 1.0
    cos_out = np.sqrt(np.clip(1.0 - sin_out * sin_out, 0.0, None))
    denom_s = n * cos_in + cos_out
    denom_p = n * cos_out + cos_in
    s = np.where(blocked, 1.0, (n * cos_in - cos_out) / np.where(denom_s == 0, 1.0, denom_s))
    p = np.where(blocked, 1.0, (n * cos_out - cos_in) / np.where(denom_p == 0, 1.0, denom_p))
    return s * s, p * p


@dataclass(frozen=True)
class _Optics:
    """Quadrature abscissae and per-abscissa Fresnel + Lambertian weight."""

    index: float
    cosines: np.ndarray
    sines: np.ndarray
    reflectances_s: np.ndarray
    reflectances_p: np.ndarray
    lambertian: np.ndarray
    step: float


_OPTICS_CACHE: dict[float, _Optics] = {}


def optics(index: float) -> _Optics:
    """Memoised per refractive index, as the engine does."""
    n = max(float(index), 1.001)
    found = _OPTICS_CACHE.get(n)
    if found is not None:
        return found
    step = math.pi / 2.0 / ANGLE_SAMPLES
    theta = (np.arange(ANGLE_SAMPLES) + 0.5) * step
    cosines = np.cos(theta)
    sines = np.sin(theta)
    rs, rp = _polarized_reflectance(cosines, n)
    built = _Optics(index=n, cosines=cosines, sines=sines,
                    reflectances_s=rs, reflectances_p=rp,
                    lambertian=2.0 * cosines * sines, step=step)
    _OPTICS_CACHE[n] = built
    return built


def _survival(t: float, cos_theta: np.ndarray) -> np.ndarray:
    """``T^(1/cos)``: what survives one slant round trip."""
    exponent = math.log(max(float(t), 1e-30)) / np.maximum(cos_theta, 1e-6)
    return np.where(exponent < -80.0, 0.0, np.exp(np.maximum(exponent, -700.0)))


def returned_fraction(t: float, opt: _Optics) -> float:
    """Fraction of a layer's exposure that comes back off the base."""
    per_trip = _survival(t, opt.cosines)
    total = np.zeros_like(per_trip)
    survival = per_trip.copy()
    for m in range(1, BOUNCE_COUNT + 1):
        rear = 0.5 * (opt.reflectances_s ** (2 * m - 1) + opt.reflectances_p ** (2 * m - 1))
        total += rear * survival
        survival = survival * per_trip
    return float((total * opt.lambertian).sum() * opt.step)


def transmittance(fraction: float, opt: _Optics) -> float:
    """Inverse of :func:`returned_fraction`, by the engine's 28-step bisection."""
    if fraction <= 0:
        return 0.0
    low, high = 1e-6, 0.999
    if returned_fraction(high, opt) <= fraction:
        return high
    for _ in range(28):
        mid = math.sqrt(low * high)
        if returned_fraction(mid, opt) < fraction:
            low = mid
        else:
            high = mid
    return math.sqrt(low * high)


def _gaussian_edge_spread(distance, sigma: float):
    from math import erfc
    return 0.5 * np.vectorize(erfc, otypes=[float])(np.asarray(distance) / (sigma * math.sqrt(2.0)))


def gaussian_edge_spread(distance: float, sigma: float) -> float:
    from math import erfc
    return float(0.5 * erfc(float(distance) / (float(sigma) * math.sqrt(2.0))))


# ------------------------------------------------------------------- profiles
class _RayProfile:
    """One layer's scattered light tabulated as (radius, weight) rays."""

    __slots__ = ("radii", "weights", "diffuse_share", "diffuse_sigma_mm", "base_radius")

    def __init__(self, radii: np.ndarray, weights: np.ndarray, diffuse_share: float,
                 diffuse_sigma_mm: float, base_radius: float):
        self.radii = radii
        self.weights = weights
        self.diffuse_share = diffuse_share
        self.diffuse_sigma_mm = diffuse_sigma_mm
        self.base_radius = base_radius


def _legacy_profile(returned: float, thickness_mm: float, n_index: float,
                    base_radius: float) -> _RayProfile:
    """The no-profile path: invert ``returnedFraction``, then tabulate the rays."""
    opt = optics(n_index)
    t = transmittance(returned, opt)
    per_trip = _survival(t, opt.cosines)
    radii, weights = [], []
    survival = per_trip.copy()
    total = 0.0
    for m in range(1, BOUNCE_COUNT + 1):
        rear = 0.5 * (opt.reflectances_s ** (2 * m - 1) + opt.reflectances_p ** (2 * m - 1))
        w = rear * survival * opt.lambertian
        bounce_total = float(w.sum())
        radii.append(2.0 * m * thickness_mm * opt.sines / np.maximum(opt.cosines, 1e-6))
        weights.append(w)
        total += bounce_total
        survival = survival * per_trip
        if bounce_total < total * 1e-3 and m > 1:
            break
    r = np.concatenate(radii) if radii else np.zeros(0)
    w = np.concatenate(weights) if weights else np.zeros(0)
    if total > 0:
        w = w / total
    return _RayProfile(r, w, 0.0, 0.0, base_radius)


def _physical_profile(thickness_mm: float, record_depth_mm: float, n_index: float,
                      optical_depth: float, angular_exponent: float,
                      diffuse_share: float, diffuse_sigma_mm: float,
                      bounce_retention: float, base_radius: float) -> _RayProfile:
    """The profile path: energy normalised independently of spatial shape."""
    opt = optics(n_index)
    layer_thickness = thickness_mm + max(record_depth_mm, 0.0)
    radii, weights = [], []
    total = 0.0
    for bounce in range(1, BOUNCE_COUNT + 1):
        m = float(bounce)
        angular = ((angular_exponent + 1.0)
                   * np.power(opt.cosines, angular_exponent) * opt.sines)
        path_exponent = -m * optical_depth / np.maximum(opt.cosines, 1e-6)
        path = np.where(path_exponent < -80.0, 0.0, np.exp(np.clip(path_exponent, -700.0, 0.0)))
        rear = 0.5 * (opt.reflectances_s ** m + opt.reflectances_p ** m)
        repeats = 1.0 if bounce == 1 else bounce_retention ** (bounce - 1)
        w = angular * path * rear * repeats
        bounce_total = float(w.sum())
        radii.append(2.0 * m * layer_thickness * opt.sines / np.maximum(opt.cosines, 1e-6))
        weights.append(w)
        total += bounce_total
        if bounce > 1 and bounce_total < total * 1e-3:
            break
    r = np.concatenate(radii) if radii else np.zeros(0)
    w = np.concatenate(weights) if weights else np.zeros(0)
    if total > 0:
        w = w / total
    return _RayProfile(r, w, diffuse_share, diffuse_sigma_mm, base_radius)


def edge_spread(profile: _RayProfile, distances: np.ndarray) -> np.ndarray:
    """Halo brightness outside a straight edge, as a fraction of the returned light."""
    d = np.asarray(distances, dtype=float)[:, None]
    r = profile.radii[None, :]
    w = profile.weights[None, :]
    beyond = r > d
    ratio = np.clip(np.divide(d, np.where(r > 0, r, 1.0)), 0.0, 1.0)
    reflex = (np.where(beyond, w * np.arccos(ratio), 0.0)).sum(axis=1) / math.pi
    if profile.diffuse_share > 0:
        diffuse = np.asarray([gaussian_edge_spread(float(v), profile.diffuse_sigma_mm)
                              for v in np.asarray(distances)])
        return profile.diffuse_share * diffuse + (1.0 - profile.diffuse_share) * reflex
    return reflex


def anchor_sigma_mm(profile: _RayProfile) -> float:
    """Sigma of the Gaussian whose half-value radius matches the real one."""
    if not np.any(profile.weights > 0):
        return 0.0
    low = 0.1 * profile.base_radius
    high = 30.0 * profile.base_radius
    for _ in range(14):
        mid = math.sqrt(low * high)
        if float(edge_spread(profile, np.array([mid]))[0]) > 0.25:
            low = mid
        else:
            high = mid
    return math.sqrt(low * high) * _HALF_WIDTH_TO_SIGMA


# ---------------------------------------------------------------- simplex fit
def _solve_symmetric(a: np.ndarray, b: np.ndarray):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    for pivot in range(n):
        row = int(np.argmax(np.abs(a[pivot:, pivot]))) + pivot
        if abs(a[row, pivot]) <= 1e-20:
            return None
        if row != pivot:
            a[[pivot, row]] = a[[row, pivot]]
            b[[pivot, row]] = b[[row, pivot]]
        for below in range(pivot + 1, n):
            factor = a[below, pivot] / a[pivot, pivot]
            if factor == 0:
                continue
            a[below, pivot:] -= factor * a[pivot, pivot:]
            b[below] -= factor * b[pivot]
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        x[row] = (b[row] - a[row, row + 1:] @ x[row + 1:]) / a[row, row]
    return x


def _solve_on_face(basis, target, active, count):
    weights = np.zeros(count)
    last = active[-1]
    free = active[:-1]
    if not free:
        weights[last] = 1.0
        return weights
    size = len(free)
    normal = np.zeros((size, size))
    rhs = np.zeros(size)
    for r, j in enumerate(free):
        dj = basis[j] - basis[last]
        for c, k in enumerate(free):
            normal[r, c] = float(dj @ (basis[k] - basis[last]))
        rhs[r] = float(dj @ (target - basis[last]))
    sol = _solve_symmetric(normal, rhs)
    if sol is None:
        return None
    if (sol < -1e-6).any():
        return None
    used = float(np.clip(sol, 0, None).sum())
    if used > 1 + 1e-6:
        return None
    for r, j in enumerate(free):
        weights[j] = max(sol[r], 0.0)
    weights[last] = 1.0 - used
    return weights


def simplex_fit(basis: Sequence[np.ndarray], target: np.ndarray) -> np.ndarray:
    """Least squares over ``w >= 0, sum(w) == 1`` by exhausting every face."""
    count = len(basis)
    best, best_res = np.zeros(count), math.inf
    for mask in range(1, 1 << count):
        active = [i for i in range(count) if mask & (1 << i)]
        cand = _solve_on_face(basis, target, active, count)
        if cand is None:
            continue
        fitted = np.stack(basis) * cand[:, None]
        res = float(((fitted.sum(0) - target) ** 2).sum())
        if res < best_res:
            best, best_res = cand, res
    return best


# -------------------------------------------------------------------- fitting
def _fit_distances(sigmas: Sequence[float]) -> np.ndarray:
    """Log-spaced sample distances for the legacy path, plus the boundary itself."""
    count = 40
    low = 0.02 * sigmas[0]
    high = 8.0 * sigmas[2]
    step = (math.log(high) - math.log(low)) / (count - 1)
    return np.concatenate([[0.0], [math.exp(math.log(low) + i * step) for i in range(count)]])


def _physical_fit_distances(radius: float) -> np.ndarray:
    """Linear samples resolve the critical-angle shoulder; log samples the tail."""
    linear = np.arange(161) / 160.0 * 4.0 * radius
    t = np.arange(1, 49) / 48.0
    tail = 4.0 * radius * np.exp(t * math.log(4.0))
    return np.concatenate([linear, tail])


def _continuous_gaussian_fit(profiles: Sequence[_RayProfile], radius: float):
    """Pick three shared sigmas and per-record non-negative weights."""
    distances = _physical_fit_distances(radius)
    targets = [edge_spread(p, distances) for p in profiles]
    low, high = math.log(0.02), math.log(4.0)
    candidates = [radius * math.exp(low + i / 13.0 * (high - low)) for i in range(14)]
    candidates += [p.diffuse_sigma_mm for p in profiles if p.diffuse_share > 0]
    candidates.sort()
    unique: list[float] = []
    for c in candidates:
        if not unique or abs(unique[-1] - c) > 1e-6 * radius:
            unique.append(c)
    candidates = unique
    basis_all = [np.asarray([gaussian_edge_spread(float(d), s) for d in distances])
                 for s in candidates]
    n = len(candidates)
    best_res, best_sig = math.inf, candidates[:SCALE_COUNT]
    best_w = [np.zeros(SCALE_COUNT) for _ in profiles]
    for a in range(n - 2):
        for b in range(a + 1, n - 1):
            for c in range(b + 1, n):
                basis = [basis_all[a], basis_all[b], basis_all[c]]
                rows = [simplex_fit(basis, t) for t in targets]
                res = 0.0
                for t, row in zip(targets, rows):
                    fitted = sum(w * bf for w, bf in zip(row, basis))
                    res += float(((fitted - t) ** 2).sum())
                if res < best_res:
                    best_res, best_sig, best_w = res, [candidates[a], candidates[b], candidates[c]], rows
    return best_sig, np.array(best_w), best_res


# --------------------------------------------------------------------- kernel
@dataclass(frozen=True)
class HalationKernel:
    """Everything the two renderers need, in micrometres on the film."""

    sigmas_um: tuple[float, float, float]
    #: One row per record, over the three shared scales; each row sums to 1.
    weights: tuple[tuple[float, float, float], ...]
    #: ``s / (1 + s)`` per record, at ``returned`` (i.e. before any slider).
    mix: tuple[float, float, float]
    #: Receiver rows by source columns, each row summing to that receiver's mix.
    matrix: tuple[tuple[float, float, float], ...]
    #: The returned fractions the kernel was built from (strength x lookScale).
    returned: tuple[float, float, float]
    #: Raw sheet matrix, kept so a changed slider can re-route without a rebuild.
    return_matrix: tuple[tuple[float, float, float], ...] | None = None
    source: str = SOURCE_PROFILE
    residual: float = 0.0
    critical_radius_um: float = 0.0

    # -- derived ------------------------------------------------------------
    def matrix_for(self, amount: float = 1.0) -> tuple[tuple[float, float, float], ...]:
        """Route the mix for a user amount, exactly as ``halationScale`` does.

        The engine multiplies the *returned fraction* and only then takes
        ``mix = s/(1+s)``, so the share **saturates**: 64x a 3% return is a 66% share,
        not 192%. The diagonal of the returned matrix is that share per record and the
        off-diagonals split it across source records.
        """
        a = max(float(amount), 0.0)
        if abs(a - 1.0) <= 1e-9:
            return self.matrix
        return _mix_matrix(mix_of(tuple(v * a for v in self.returned)), self.return_matrix)

    def sigma_px(self, pixel_size_um: float) -> tuple[float, float, float]:
        px = max(float(pixel_size_um), 1e-6)
        return tuple(s / px for s in self.sigmas_um)


def _mix_matrix(mix: Sequence[float], return_matrix) -> tuple[tuple[float, float, float], ...]:
    """Split each receiver's share across the source records by the sheet matrix.

    A missing or degenerate row keeps the whole share on the diagonal, which is
    the legacy scalar mix exactly.
    """
    rows = []
    for receiver in range(3):
        row = None
        if (isinstance(return_matrix, (list, tuple)) and len(return_matrix) == 3
                and isinstance(return_matrix[receiver], (list, tuple))
                and len(return_matrix[receiver]) == 3):
            try:
                row = [max(float(v), 0.0) for v in return_matrix[receiver]]
            except (TypeError, ValueError):
                row = None
        if row is None or sum(row) <= 0:
            rows.append(tuple(float(mix[receiver]) if c == receiver else 0.0 for c in range(3)))
        else:
            total = sum(row)
            rows.append(tuple(float(mix[receiver]) * v / total for v in row))
    return tuple(rows)


def mix_of(fraction: Sequence[float]) -> tuple[float, float, float]:
    """``s / (1 + s)``: the share of a layer's exposure that arrives as returned light."""
    return tuple(max(float(v), 0.0) / (1.0 + max(float(v), 0.0)) for v in fraction)


def _hazed(sigmas: Sequence[float], haze_mm: float) -> tuple[float, float, float]:
    """Support haze convolves the fit, so the sigmas add in quadrature."""
    if not haze_mm or haze_mm <= 0:
        return tuple(float(s) for s in sigmas)
    return tuple(math.sqrt(s * s + haze_mm * haze_mm) for s in sigmas)


def build_kernel(returned: Sequence[float],
                 thickness_mm: float,
                 refractive_indices: Sequence[float] | None = None,
                 profile: Mapping[str, Any] | None = None,
                 return_matrix=None,
                 haze_mm: float = 0.0) -> HalationKernel:
    """Build the kernel the way ``Halation.kernel(returned:base:profile:...)`` does.

    ``returned`` is each record's reflected fraction (the profile's ``strength``,
    already multiplied by ``look_scale`` and by any render option).
    """
    ret = tuple(max(float(v), 0.0) for v in returned)
    if len(ret) != 3:
        raise ValueError("returned must have three records")

    if refractive_indices and len(refractive_indices) == 3:
        indices = [max(float(v), 1.001) for v in refractive_indices]
    else:
        indices = [max(float(np.mean(refractive_indices or [1.48])), 1.001)] * 3

    thickness = max(float(thickness_mm), 0.0)
    critical = 0.0
    for index in indices:
        critical = max(critical, 2.0 * thickness * math.tan(math.asin(1.0 / index)))
    critical = max(critical, 1e-9)  # mm

    profile = profile if isinstance(profile, Mapping) else None
    if profile is None:
        profiles = [_legacy_profile(ret[layer], thickness, indices[layer], critical)
                    for layer in range(3)]
        anchor = max(max(anchor_sigma_mm(p) for p in profiles), 1e-9)
        sigmas = (anchor / SCALE_RATIO, anchor, anchor * SCALE_RATIO)
        distances = _fit_distances(sigmas)
        basis = [np.asarray([gaussian_edge_spread(float(d), s) for d in distances])
                 for s in sigmas]
        weights = np.array([simplex_fit(basis, edge_spread(p, distances)) for p in profiles])
        residual = 0.0
        source = SOURCE_DERIVED
    else:
        depths = profile.get("recordDepthMM")
        depths = depths if isinstance(depths, (list, tuple)) and len(depths) == 3 else [0.0] * 3
        optical = _triple(profile.get("roundTripOpticalDepth"))
        angular = _triple(profile.get("angularExponent"))
        diffuse_share = _triple(profile.get("diffuseShare"))
        diffuse_sigma = _triple(profile.get("diffuseSigmaMM"))
        retention = _triple(profile.get("bounceRetention"))
        profiles, radii = [], []
        for layer in range(3):
            depth = max(float(depths[layer]), 0.0)
            radii.append(2.0 * (thickness + depth) * math.tan(math.asin(1.0 / indices[layer])))
            profiles.append(_physical_profile(
                thickness, depth, indices[layer],
                max(optical[layer], 0.0),
                max(angular[layer], 0.0),
                min(max(diffuse_share[layer], 0.0), 1.0),
                max(diffuse_sigma[layer], 1e-6),
                min(max(retention[layer], 0.0), 1.0),
                critical))
        sigmas, weights, residual = _continuous_gaussian_fit(profiles, max(max(radii), 1e-9))
        source = SOURCE_PROFILE

    mixes = mix_of(ret)
    matrix = _mix_matrix(mixes, return_matrix)
    return HalationKernel(
        sigmas_um=tuple(float(s) * 1000.0 for s in _hazed(sigmas, haze_mm)),
        weights=tuple(tuple(float(w) for w in row) for row in np.asarray(weights)),
        mix=mixes,
        matrix=matrix,
        returned=ret,
        return_matrix=(tuple(tuple(float(v) for v in row) for row in return_matrix)
                       if isinstance(return_matrix, (list, tuple)) and len(return_matrix) == 3
                       else None),
        source=source,
        residual=float(residual),
        critical_radius_um=critical * 1000.0,
    )


def _triple(value) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            pass
    return [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------- cache
_KERNEL_CACHE: dict[tuple, HalationKernel] = {}
_KERNEL_CACHE_LIMIT = 96


def _cache_key(returned, thickness_mm, indices, profile, return_matrix, haze_mm) -> tuple:
    def r(v):
        try:
            return round(float(v), 9)
        except (TypeError, ValueError):
            return 0.0

    prof_key = None
    if isinstance(profile, Mapping):
        prof_key = tuple(
            r(v)
            for name in ("roundTripOpticalDepth", "angularExponent", "diffuseShare",
                         "diffuseSigmaMM", "bounceRetention", "recordDepthMM")
            for v in (profile.get(name) or [0.0, 0.0, 0.0])
            if isinstance(v, (int, float))
        )
    rm_key = None
    if isinstance(return_matrix, (list, tuple)) and len(return_matrix) == 3:
        rm_key = tuple(r(v) for row in return_matrix if isinstance(row, (list, tuple)) for v in row)
    return (tuple(r(v) for v in returned), r(thickness_mm),
            tuple(r(v) for v in (indices or [])), prof_key, rm_key, r(haze_mm))


def kernel_for(returned, thickness_mm, refractive_indices=None, profile=None,
               return_matrix=None, haze_mm: float = 0.0) -> HalationKernel:
    """Cached :func:`build_kernel`. ~0.1 s on a miss, microseconds on a hit."""
    key = _cache_key(returned, thickness_mm, refractive_indices, profile, return_matrix, haze_mm)
    found = _KERNEL_CACHE.get(key)
    if found is not None:
        return found
    built = build_kernel(returned, thickness_mm, refractive_indices, profile,
                         return_matrix, haze_mm)
    if len(_KERNEL_CACHE) >= _KERNEL_CACHE_LIMIT:
        _KERNEL_CACHE.clear()
    _KERNEL_CACHE[key] = built
    return built


def clear_cache() -> None:
    _KERNEL_CACHE.clear()
