"""Diffusion-filter kernel as a Gaussian mixture, shared by CPU and GPU.

Why this module exists: the physical PSF in ``spektrafilm.model.diffusion`` is an
exponential-radial kernel, ``exp(-r/lambda) / (2*pi*lambda^2)``, convolved via FFT.
That kernel is not separable, so the viewport cannot evaluate it as a chain of cheap
passes -- which is why the shader has always carried a different, hand-tuned bloom and
why diffusion-on frames disagreed with the export by 20-37 dB (measured 2026-10-30,
worse than the halation gap that motivated the halation rework).

A sum of exponential sub-components (the per-family shapes in
``_DIFFUSION_FILTER_SHAPES``) is however extremely well approximated by a *Gaussian
mixture*: with 5 free-width components the fitted kernel matches the analytic curve to
L1 <= 0.8% / max <= 2% across all four families. A Gaussian mixture is separable per
component, so both renderers can evaluate the *same* kernel: the CPU with decimated
``cv2.GaussianBlur`` passes, the GPU with its separable-blur chain. One calibration
table, one look, no second implementation to drift.

Fitting is done once per (family, group, channel) in *dimensionless* form. A group's
curve depends only on the ratios of its sub-component lambdas and their weights -- the
physical kernel scales exactly, ``K_{lam_c}(r) = F(r/lam_c) / (2*pi*lam_c^2)`` -- so the
fit lives in rho = r / lambda_center space and the sigmas scale linearly with lambda at
runtime. Results are cached at module level.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Components per fitted group. 5 keeps every family at L1 <= 0.8%; the mix applies
#: through a separable blur, so the cost is linear in this number.
FIT_COMPONENTS = 5

#: Bloom lambdas reach 400 px at a 6 um/px export; evaluating that at full resolution
#: is pointless for a veil. Halo is milder (8-31 px). These are *minimum* divisors:
#: :func:`effective_divisor` raises them in powers of two so the working-grid sigma
#: stays within the blur kernels' 24-tap reach (3*sigma <= 24). Both renderers must
#: apply the same rule or their decimation grids diverge.
HALO_DIVISOR = 2
BLOOM_DIVISOR = 8
MAX_WORK_SIGMA_PX = 7.0     # 3*7 = 21 <= the blur shaders' 24-tap reach
MAX_DIVISOR = 256
#: Components below this sigma run at full resolution on BOTH renderers (the CPU's
#: halation_model._gauss_decimated ignores decimation under the same threshold, and a
#: sub-threshold sigma on a decimated grid would smear the component over the cell).
COMPONENT_FULLRES_SIGMA_PX = 6.0


def effective_divisor(divisor: int, sigmas_px) -> int:
    """Smallest power-of-two divisor >= ``divisor`` keeping max(sigma)/div <= 8 px."""
    div = max(1, int(divisor))
    max_s = max(sigmas_px) if len(sigmas_px) else 0.0
    while max_s / div > MAX_WORK_SIGMA_PX and div < MAX_DIVISOR:
        div *= 2
    return div

_FIT_R_MAX = 12.0        # multiples of lambda_center; the CPU truncates the PSF at 8,
                         # the fit sees a little more tail so the truncation is softer
_FIT_N_R = 2400


class GroupMix:
    """One kernel group as a Gaussian mixture in *pixel* units.

    ``amps`` is a tuple over sigmas; for the halo it is a tuple of per-sigma
    ``(r, g, b)`` amplitude triples. The kernel is
    ``sum_i amps[i] * G2d(sigmas_px[i])`` with the amplitudes summing to 1
    (unit mass), matching ``_build_group_kernel``'s unit grid-sum.
    ``divisor`` is the resolution reduction the renderers apply to this group.
    """

    __slots__ = ("sigmas_px", "amps", "divisor")

    def __init__(self, sigmas_px, amps, divisor):
        self.sigmas_px = tuple(float(v) for v in sigmas_px)
        self.amps = amps
        self.divisor = int(divisor)


class DiffusionMix:
    """Everything both renderers need for one diffusion evaluation.

    The physical model mixes three groups as
    ``out = (1 - p_s) * E + p_s * [ a * K_ch(E) + b * Bloom(E) ]`` with
    ``a = (w_c + w_h) / tot``, ``b = w_b / tot`` and
    ``K_ch = (w_c * core + w_h * halo_ch) / (w_c + w_h)``. The weights that turn
    ``p_s * a`` into per-group contributions are exposed pre-multiplied as
    ``w_core`` / ``w_halo`` / ``w_bloom`` so a renderer writes exactly
    ``out = keep*E + w_core*Core(E) + w_halo_c*Halo_c(E) + w_bloom*Bloom(E)``.
    """

    __slots__ = ("family", "core", "halo", "bloom", "p_s",
                 "w_core", "w_halo", "w_bloom", "strength")

    def __init__(self, family, core, halo, bloom, p_s, w_core, w_halo, w_bloom, strength):
        self.family = family
        self.core = core
        self.halo = halo
        self.bloom = bloom
        self.p_s = float(p_s)
        self.w_core = float(w_core)
        self.w_halo = float(w_halo)
        self.w_bloom = float(w_bloom)
        self.strength = float(strength)


def _fit_curve(sigmas0, target, rho):
    """Least-squares fit of ``sum a_i * exp(-rho^2 / 2 s_i^2)`` to ``target``.

    Free widths, multi-start, amplitudes constrained non-negative through the bounds.
    Returns (sigmas, amps) sorted by sigma.
    """
    from scipy.optimize import least_squares

    M = len(sigmas0)
    # Relative error, floored so the far tail (target ~ 0) still counts a little.
    wr = 1.0 / (target + 0.02 * target.max())

    def resid(p):
        s, a = p[:M], p[M:]
        out = np.zeros_like(rho)
        for si, ai in zip(s, a):
            out += ai * np.exp(-rho ** 2 / (2 * si ** 2))
        return (out - target) * wr

    best, best_cost = None, np.inf
    for seed in range(6):
        rng = np.random.default_rng(seed)
        s0 = sigmas0 * rng.uniform(0.6, 1.6, M)
        a0 = np.full(M, target.max())
        p0 = np.concatenate([s0, a0])
        res = least_squares(resid, p0,
                            bounds=(np.concatenate([np.full(M, 1e-4), np.zeros(M)]),
                                    np.concatenate([np.full(M, rho.max() * 2.0), np.full(M, 1e8)])),
                            max_nfev=400)
        if res.cost < best_cost:
            best, best_cost = res.x, res.cost
    s, a = best[:M], best[M:]
    order = np.argsort(s)
    return s[order], a[order]


@lru_cache(maxsize=64)
def _fit_shape(kind: str, family: str, ratios: tuple, weights: tuple):
    """Fit one group's shape in lambda_center units. Cached.

    The dimensionless shape of the *unit-mass* kernel group is
    ``F(rho) = sum_k w_k / x_k^2 * exp(-rho / x_k)``: each sub-component brings its own
    ``1/lambda_k^2`` normalisation, which in scaled units becomes ``1/x_k^2``. The fit
    preserves mass automatically (each Gaussian component integrates to ``2*pi*s_i^2``
    times its amplitude), and the amplitudes are renormalised to sum to 1 anyway.
    """
    M = FIT_COMPONENTS
    rho = np.geomspace(1e-3, _FIT_R_MAX, _FIT_N_R)
    target = np.zeros_like(rho)
    for xk, wk in zip(ratios, weights):
        target += wk / (xk * xk) * np.exp(-rho / xk)
    target /= target.max()
    sig0 = np.geomspace(0.05, _FIT_R_MAX * 0.5, M)
    s, a = _fit_curve(sig0, target, rho)
    w = a * s * s
    w = w / w.sum()
    return tuple(float(v) for v in s), tuple(float(v) for v in w)


def _sigma_px(lambda_um: float, spatial_scale: float, pixel_size_um: float) -> float:
    return max(lambda_um * max(float(spatial_scale), 1e-6) / max(pixel_size_um, 1e-6), 1e-3)


@lru_cache(maxsize=16)
def _family_cfg(family: str, warmth: float) -> dict:
    from spektrafilm.model.diffusion import (
        _DIFFUSION_FILTER_SHAPES, _expand_group, _halo_channel_weights)

    cfg = _DIFFUSION_FILTER_SHAPES[family]
    core_l, core_w = _expand_group(cfg["core"], kind="core")
    halo_l, halo_w = _expand_group(cfg["halo"], kind="halo")
    bloom_l, bloom_w = _expand_group(cfg["bloom"], kind="bloom")
    # The effective warmth is the family's base PLUS the user's dial -- the base is
    # part of the family's look (black_pro_mist ships at +0.65). Forgetting it
    # flattens the per-channel halo weights and costs ~10 dB on warm families.
    effective = float(np.clip(cfg.get("halo_warmth_base", 0.0) + warmth, -1.5, 1.5))
    halo_per_ch = _halo_channel_weights(halo_w, effective)
    return {"core": (core_l, core_w), "halo": halo_l, "bloom": (bloom_l, bloom_w),
            "halo_per_ch": halo_per_ch,
            "w_c": cfg["w_c"], "w_h": cfg["w_h"], "w_b": cfg["w_b"]}


def resolve_family_name(family: str) -> str | None:
    """Map the loose family strings used by sessions onto the four canonical names."""
    f = str(family).lower().strip()
    if f in ("", "none"):
        return None
    if f in ("glimmerglass", "black_pro_mist", "pro_mist", "cinebloom"):
        return f
    if "glimmer" in f:
        return "glimmerglass"
    if "cine" in f:
        return "cinebloom"
    if "pro_mist" in f or "promist" in f:
        return "black_pro_mist" if "black" in f else "pro_mist"
    return None


def resolve_diffusion_mix(family: str, strength: float, spatial_scale: float,
                          pixel_size_um: float, halo_warmth: float = 0.0) -> DiffusionMix | None:
    """Build the shared Gaussian-mixture plan for one diffusion evaluation.

    ``pixel_size_um`` is the pitch of the image being filtered (export: sensor pitch at
    full resolution). ``strength`` follows the UI's 0..2+ scale. Returns None when the
    filter is a no-op.
    """
    from spektrafilm.model.diffusion import _strength_to_scatter
    from scipy.optimize import nnls

    fam_name = resolve_family_name(family)
    if fam_name is None:
        return None
    strength = float(strength)
    p_s = _strength_to_scatter(strength, fam_name) if strength > 0.001 else 0.0
    if p_s <= 0.0:
        return None
    warmth = float(np.clip(float(halo_warmth), -1.5, 1.5))
    cfg = _family_cfg(fam_name, round(warmth, 4))
    M = FIT_COMPONENTS
    rho = np.geomspace(1e-3, _FIT_R_MAX, _FIT_N_R)

    def px(lam_um):
        return _sigma_px(lam_um, spatial_scale, pixel_size_um)

    # --- core: grey kernel, one fit, full resolution ---
    core_l, core_w = cfg["core"]
    lam_c = float(np.mean(core_l))
    s_c, a_c = _fit_shape("core", fam_name,
                          tuple(float(v / lam_c) for v in core_l),
                          tuple(float(v) for v in core_w))
    core = GroupMix(sigmas_px=[px(lam_c * s) for s in s_c], amps=a_c, divisor=1)

    # --- halo: three channels share the sigma set; amplitudes are per channel ---
    halo_l = cfg["halo"]
    halo_per_ch = cfg["halo_per_ch"]
    lam_c = float(np.mean(halo_l))
    ratios = tuple(float(v / lam_c) for v in halo_l)
    mean_w = tuple(float(v) for v in np.mean(halo_per_ch, axis=0))
    s_h, _ = _fit_shape("halo", fam_name, ratios, mean_w)
    # Design matrix on the fixed sigma set, then one NNLS per channel. Everything here
    # lives in the dimensionless rho = r / lambda_center space: the sigmas of the fit
    # ARE the ratios, so no lambda factor enters the Gaussian argument (mixing one in
    # flattens every column to ~1 and the NNLS collapses onto the narrowest one).
    A = np.stack([np.exp(-rho ** 2 / (2 * s ** 2)) for s in s_h], axis=1)
    amps_ch = []
    for c in range(3):
        t = np.zeros_like(rho)
        for xk, wk in zip(ratios, halo_per_ch[c]):
            t += wk / (xk * xk) * np.exp(-rho / xk)
        coef, _ = nnls(A, t)
        w = coef * np.asarray(s_h) ** 2
        w = w / max(w.sum(), 1e-12)
        amps_ch.append(w)
    halo = GroupMix(sigmas_px=[px(lam_c * s) for s in s_h],
                    amps=tuple(tuple(float(amps_ch[c][i]) for c in range(3)) for i in range(M)),
                    divisor=effective_divisor(HALO_DIVISOR,
                                              [px(lam_c * sv) for sv in s_h]))

    # --- bloom: long-reach veil, evaluated on a decimated grid ---
    bloom_l, bloom_w = cfg["bloom"]
    lam_c = float(np.mean(bloom_l))
    s_b, a_b = _fit_shape("bloom", fam_name,
                          tuple(float(v / lam_c) for v in bloom_l),
                          tuple(float(v) for v in bloom_w))
    bloom = GroupMix(sigmas_px=[px(lam_c * s) for s in s_b], amps=a_b,
                     divisor=effective_divisor(BLOOM_DIVISOR,
                                               [px(lam_c * sv) for sv in s_b]))

    w_c, w_h, w_b = cfg["w_c"], cfg["w_h"], cfg["w_b"]
    tot = w_c + w_h + w_b
    a_norm = (w_c + w_h) / tot
    return DiffusionMix(family=fam_name, core=core, halo=halo, bloom=bloom,
                        p_s=p_s,
                        w_core=p_s * a_norm * w_c / (w_c + w_h),
                        w_halo=p_s * a_norm * w_h / (w_c + w_h),
                        w_bloom=p_s * w_b / tot,
                        strength=strength)


def apply_diffusion_mix_np(image: np.ndarray, mix: DiffusionMix) -> np.ndarray:
    """CPU evaluation of the mixture plan -- the export-side kernel.

    Mirrors the structure of ``apply_diffusion_filter_um``: core at full resolution,
    halo mildly decimated, bloom strongly decimated, energy-conserving mix at the end.
    Large sigmas go through the decimated Gaussian from ``halation_model`` (measured at
    120 dB against the full-resolution blur there).

    Sigmas below ``DISCRETE_SIGMA_PX`` use an explicit grid-sum-normalised kernel
    instead of ``cv2.GaussianBlur``: the physical reference kernel is normalised by its
    *discrete* grid sum (``_build_group_kernel``), and for a sub-pixel lambda the two
    normalisations diverge badly -- a continuous Gaussian tends to a delta of infinite
    height, the grid-sum kernel to a delta of height *amp* (mass spread over the
    neighbours it still touches). Measured on a delta frame at 44.9 um/px, ignoring
    this made the halo centre read +23..40% against the reference; with it the whole
    radial profile re-aligns.
    """
    import cv2
    from .halation_model import _gauss_decimated

    DISCRETE_SIGMA_PX = 0.8

    img32 = image.astype(np.float32, copy=False)
    H, W = img32.shape[:2]

    small_kernel_cache: dict[float, tuple] = {}

    def blur(img, sigma, divisor):
        if sigma < 0.05:
            return img
        # A decimated pass must not see a sub-pixel sigma: the box average would smear
        # a delta-like component over the whole decimation cell. Small sigmas always
        # run at full resolution with the grid-sum kernel.
        if sigma < DISCRETE_SIGMA_PX * max(1, divisor):
            k1 = small_kernel_cache.get(sigma)
            if k1 is None:
                R = max(1, int(np.ceil(3.0 * sigma)))
                ax = np.arange(-R, R + 1, dtype=np.float32)
                k1 = np.exp(-ax ** 2 / (2 * sigma ** 2))
                k1 /= k1.sum()
                small_kernel_cache[sigma] = (k1,)
            k = small_kernel_cache[sigma][0]
            return cv2.sepFilter2D(img, -1, k, k)
        return _gauss_decimated(img, sigma, divisor)

    out = img32 * (1.0 - mix.p_s)

    # core: grey kernel (same for every channel)
    if mix.w_core > 1e-9:
        acc = np.zeros_like(img32)
        for s, a in zip(mix.core.sigmas_px, mix.core.amps):
            if a <= 1e-6:
                continue
            acc += blur(img32, s, 1) * a
        out += acc * mix.w_core

    # halo: per-channel amplitudes over a shared sigma set
    if mix.w_halo > 1e-9:
        acc = np.zeros_like(img32)
        for i, s in enumerate(mix.halo.sigmas_px):
            if s < 0.05:
                continue
            b = blur(img32, s, mix.halo.divisor)
            for c in range(3):
                acc[..., c] += b[..., c] * mix.halo.amps[i][c]
        out += acc * mix.w_halo

    # bloom: strongly decimated grid
    if mix.w_bloom > 1e-9:
        dn = mix.bloom.divisor
        hs, ws = max(2, H // dn), max(2, W // dn)
        small = cv2.resize(img32, (ws, hs), interpolation=cv2.INTER_AREA)
        acc = np.zeros_like(small)
        for s, a in zip(mix.bloom.sigmas_px, mix.bloom.amps):
            if a <= 1e-6:
                continue
            acc += blur(small, max(s / dn, 0.05), 1) * a
        out += cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR) * mix.w_bloom

    return out.astype(image.dtype, copy=False)
