import os

import numpy as np
import scipy.ndimage
from spektrafilm.utils.fast_gaussian_filter import fast_exponential_filter, fast_gaussian_filter
from spektrafilm.utils.numba_boost_hightlights import boost_highlights

# ---------------------------------------------------------------------------
# Optional GPU backend for the diffusion filter.
#
# The filter is a large-kernel convolution, so it is FFT-bound. On CPU a 24 MP
# export spends ~1.4 s in rfft2/irfft2 alone; cuFFT does the same work in ~90 ms.
# CuPy is an optional dependency: when it is missing, or the machine has no
# usable CUDA device, or SPEKTRA_DIFFUSION_GPU=0, every call falls back to the
# CPU path with identical results (same PSF, same p_s, float32 throughout).
# Nothing else in the pipeline changes, so this is a pure speed switch.
_GPU_STATE = {"probed": False, "module": None, "max_px": None}


def _gpu_backend():
    """Return the cupy module if a CUDA device with cuFFT can be used."""
    st = _GPU_STATE
    if st["probed"]:
        return st["module"]
    st["probed"] = True
    if str(os.environ.get("SPEKTRA_DIFFUSION_GPU", "1")).strip() == "0":
        return None
    try:
        import cupy as cp
        probe = cp.zeros((16, 16), dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        cp.fft.irfft2(cp.fft.rfft2(probe))
        cp.cuda.Stream.null.synchronize()
        st["module"] = cp
    except Exception:
        st["module"] = None
    return st["module"]


def diffusion_backend_name() -> str:
    """'cupy' when the GPU path is active, otherwise 'cpu'."""
    return "cupy" if _gpu_backend() is not None else "cpu"


def _release_gpu_memory(cp):
    """Return cached device/pinned blocks to the driver. Never raises."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _gpu_group_kernel(cp, size, lambdas_um, weights, spatial_scale, pixel_size_um):
    """Unit-mass 2D exponential mix, built on the GPU.

    Same construction as `_build_group_kernel`: every sub-component is
    independently normalised to unit grid sum, then mixed by `weights`. The
    1/(2*pi*lambda^2) prefactor drops out of that normalisation, so it is
    omitted here.
    """
    R = size // 2
    axis = cp.arange(size, dtype=cp.float32) - R
    r = cp.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    total = cp.zeros((size, size), dtype=cp.float32)
    for wk, lk in zip(weights, lambdas_um):
        lp = max(float(lk) * spatial_scale / pixel_size_um, 1e-6)
        k = cp.exp(-r / lp)
        s = k.sum()
        if float(s) > 0.0:
            k = k / s
        total += float(wk) * k
    return total


def _gpu_convolve(cp, g, kernels, radius):
    """Per-channel FFT convolution of a GPU array with one shared forward FFT."""
    H, W = g.shape[:2]
    padded = cp.pad(g, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
    Ph, Pw = padded.shape[:2]
    Fi = cp.fft.rfft2(padded, axes=(0, 1))
    out = cp.empty((H, W, 3), dtype=cp.float32)
    for c in range(3):
        k = kernels[c]
        kp = cp.zeros((Ph, Pw), dtype=cp.float32)
        kh, kw = k.shape
        kp[:kh // 2 + 1, :kw // 2 + 1] = k[kh // 2:, kw // 2:]
        kp[:kh // 2 + 1, Pw - kw // 2:] = k[kh // 2:, :kw // 2]
        kp[Ph - kh // 2:, :kw // 2 + 1] = k[:kh // 2, kw // 2:]
        kp[Ph - kh // 2:, Pw - kw // 2:] = k[:kh // 2, :kw // 2]
        Fk = cp.fft.rfft2(kp)
        conv = cp.fft.irfft2(Fi[..., c] * Fk, s=(Ph, Pw), axes=(0, 1))
        out[:, :, c] = conv[radius:radius + H, radius:radius + W]
    return out


def _gpu_upsample(cp, small, shape):
    H, W = shape
    h, w = small.shape[:2]
    if h == H and w == W:
        return small
    try:
        from cupyx.scipy.ndimage import zoom
        return zoom(small, (H / h, W / w, 1.0), order=1, prefilter=False).astype(cp.float32)
    except Exception:
        fy, fx = H // h, W // w
        up = cp.repeat(cp.repeat(small, fy, axis=0), fx, axis=1)
        return up[:H, :W]


def _apply_diffusion_gpu(cp, image, p_s, core_l, core_w, halo_l, halo_per_ch,
                         bloom_l, bloom_w, w_c, w_h, w_b, scale, pixel_size_um):
    """GPU implementation of `apply_diffusion_filter_um`.

    Numerically the same model: identical radii, identical per-group kernels,
    identical energy-conserving composite. Only the FFT host changes.
    """
    H, W = image.shape[:2]
    g = cp.asarray(image, dtype=cp.float32)

    # ---- core + halo, full resolution ---------------------------------
    lam_max_ch_px = max(float(core_l.max()), float(halo_l.max())) * scale / pixel_size_um
    radius_ch = int(np.ceil(8.0 * lam_max_ch_px))
    radius_ch = max(radius_ch, 3)
    radius_ch = min(radius_ch, min(H, W) // 2 - 1)

    kcore = _gpu_group_kernel(cp, 2 * radius_ch + 1, core_l, core_w, scale, pixel_size_um)
    khalo = [_gpu_group_kernel(cp, 2 * radius_ch + 1, halo_l, halo_per_ch[c], scale, pixel_size_um)
             for c in range(3)]
    k_full = [(w_c * kcore + w_h * khalo[c]) / (w_c + w_h) for c in range(3)]
    blurred = _gpu_convolve(cp, g, k_full, radius_ch)

    # ---- bloom, 4x downsample ------------------------------------------
    bs = 4
    Hs, Ws = max(H // bs, 1), max(W // bs, 1)
    pxs = pixel_size_um * bs
    radius_b = int(np.ceil(8.0 * float(bloom_l.max()) * scale / pxs))
    radius_b = max(radius_b, 3)
    radius_b = min(radius_b, min(Hs, Ws) // 2 - 1)
    small = g[:Hs * bs, :Ws * bs].reshape(Hs, bs, Ws, bs, 3).mean(axis=(1, 3))
    kbloom = _gpu_group_kernel(cp, 2 * radius_b + 1, bloom_l, bloom_w, scale, pxs)
    bloom_small = _gpu_convolve(cp, small, [kbloom] * 3, radius_b)
    bloom_up = _gpu_upsample(cp, bloom_small, (H, W))

    tot = w_c + w_h + w_b
    a = float((w_c + w_h) / tot * p_s)
    b = float(w_b / tot * p_s)
    keep = float(1.0 - p_s)
    # a*blurred + b*bloom + keep*g, in place to avoid a third 24 MP temporary.
    out = blurred
    out *= a
    bloom_up *= b
    out += bloom_up
    out += g * keep
    return cp.asnumpy(out)


def apply_unsharp_mask(image, sigma=0.0, amount=0.0):
    """
    Apply an unsharp mask to an image.
    
    Parameters:
    image (ndarray): The input image to be processed.
    sigma (float, optional): The standard deviation for the Gaussian sharp filter. Leave 0 if not wanted.
    amount (float, optional): The strength of the sharpening effect. Leave 0 if not wanted.
    
    Returns:
    ndarray: The processed image after applying the unsharp mask.
    """
    # image_blur = scipy.ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    image_blur = fast_gaussian_filter(image, sigma)
    image_sharp = image + amount * (image - image_blur)
    return image_sharp


def apply_halation_um(raw, halation, pixel_size_um):
    """Apply highlight boost, in-emulsion scatter, and back-reflection halation.

    Ordering is boost -> scatter -> halation: the boost reconstructs pre-clip
    irradiance (what actually hit the emulsion), scatter propagates that
    irradiance through the absorbing emulsion as an energy-preserving
    two-Gaussian mixture (core + tail), and halation adds back-reflected
    light as an additive sum of N Gaussians with sqrt(k)-spaced widths.

    See the private halation notes for the physical derivation and parameter
    priors.
    """
    if not halation.active:
        return raw

    # 1. Scatter pass — energy-preserving mixture of a Gaussian core and an
    #    exponential tail (matching measured film MTFs), blended with the
    #    identity by scatter_amount to model the fraction of photons that
    #    actually scatter:
    #    E1 = (1 - s) * E0  +  s * [(1 - w_s) * G(sigma_c) * E0 + w_s * Exp(lambda_t) * E0]
    #    where s = scatter_amount. sigma_c and lambda_t are both scaled by
    #    scatter_spatial_scale; lambda_t is the decay constant of the
    #    exponential, dispatched internally to a Gaussian mixture.
    s_amount = float(halation.scatter_amount)
    s_scale = float(halation.scatter_spatial_scale)
    w_s = np.asarray(halation.scatter_tail_weight, dtype=np.float64)
    sigma_c_px = np.asarray(halation.scatter_core_um, dtype=np.float64) * s_scale / pixel_size_um
    lambda_t_px = np.asarray(halation.scatter_tail_um, dtype=np.float64) * s_scale / pixel_size_um
    if s_amount > 0 and (np.any(sigma_c_px > 0) or np.any(lambda_t_px > 0)):
        core = fast_gaussian_filter(raw, np.maximum(sigma_c_px, 1e-6))
        tail = fast_exponential_filter(raw, np.maximum(lambda_t_px, 1e-6))
        scattered = (1.0 - w_s) * core + w_s * tail
        raw = (1.0 - s_amount) * raw + s_amount * scattered

    # 2. Halation pass — additive multi-bounce sum scaled by halation_amount:
    #    E2 = E1 + halation_amount * Σ_{k=1..N} a_k * G(sigma_h * sqrt(k)) * E1
    #    with a_k = a_tot * ρ^(k-1) / Σ_j ρ^(j-1) and sigmas scaled by
    #    halation_spatial_scale.
    h_amount = float(halation.halation_amount)
    h_scale = float(halation.halation_spatial_scale)

    # Check for Snell TIR physical film base model (Phase 4)
    film_base = getattr(halation, "film_base", None)
    enable_snell = getattr(halation, "enable_snell_tir", True)
    if film_base and enable_snell and isinstance(film_base, dict):
        thickness_mm = float(film_base.get("base_thickness_mm", film_base.get("thickness_mm", 0.13)))
        thickness_um = thickness_mm * 1000.0
        indices = film_base.get("record_refractive_indices")
        if not indices or len(indices) != 3:
            n_val = float(film_base.get("refractive_index", 1.48))
            indices = [n_val * 0.996, n_val, n_val * 1.006]
        radii_um = np.array([
            2.0 * thickness_um * np.tan(np.arcsin(1.0 / max(float(n), 1.01)))
            for n in indices
        ], dtype=np.float64)
        base_sigma_h_um = radii_um / np.sqrt(2.0)

        if "strength" in film_base and isinstance(film_base["strength"], (list, tuple)):
            phys_strength = np.array(film_base["strength"], dtype=np.float64)
            if np.max(phys_strength) <= 0:
                phys_strength = np.array([0.0055, 0.001, 0.0], dtype=np.float64)
        else:
            antihalation = str(film_base.get("antihalation", "weak")).lower()
            if antihalation in ("no", "none"):
                phys_strength = np.array([0.28, 0.07, 0.015], dtype=np.float64)
            elif antihalation == "weak":
                phys_strength = np.array([0.08, 0.02, 0.005], dtype=np.float64)
            else:
                phys_strength = np.array([0.015, 0.003, 0.0], dtype=np.float64)

        a_tot = phys_strength * h_amount
        sigma_h_px = base_sigma_h_um * h_scale / pixel_size_um
    else:
        a_tot = np.asarray(halation.halation_strength, dtype=np.float64) * h_amount
        sigma_h_px = np.asarray(halation.halation_first_sigma_um, dtype=np.float64) * h_scale / pixel_size_um
    N = int(halation.halation_n_bounces)
    rho = float(halation.halation_bounce_decay)
    if N >= 1 and np.any(a_tot > 0) and np.any(sigma_h_px > 0):
        decay = np.array([rho ** (k - 1) for k in range(1, N + 1)], dtype=np.float64)
        decay /= decay.sum()
        halation_blur = np.zeros_like(raw)
        for k, wk in zip(range(1, N + 1), decay):
            sigma_k_px = np.maximum(sigma_h_px * np.sqrt(k), 1e-6)
            halation_blur += wk * fast_gaussian_filter(raw, sigma_k_px)
        raw = raw + a_tot * halation_blur
        if halation.halation_renormalize:
            raw = raw / (1.0 + a_tot)

    return raw

def apply_gaussian_blur(data, sigma):
    if sigma > 0:
        # return scipy.ndimage.gaussian_filter(data, (sigma, sigma, 0))
        # data = np.double(data)
        # data = np.ascontiguousarray(data)
        return fast_gaussian_filter(data, sigma)
    else:
        return data
    
def apply_gaussian_blur_um(data, sigma_um, pixel_size_um):
    # Spatial-off short-circuit: when sigma_um is 0 (e.g. under
    # debug.lut_mode, which disables all spatial effects), pixel_size_um
    # is irrelevant. Early-return so callers that haven't run preprocess
    # — typical for LUT bakes that inject mid-pipeline — don't crash on
    # division by None.
    if sigma_um <= 0:
        return data
    sigma = sigma_um / pixel_size_um
    if sigma > 0:
        return fast_gaussian_filter(data, sigma)
    return data

def apply_diffusion_filter_mm(data, diffusion_filter_params, pixel_size_um):
    diffusion_fraction, sigma_mm, iterations, growth, decay = diffusion_filter_params
    iterations = int(iterations)
    sigma = sigma_mm * 1000 / pixel_size_um
    if sigma_mm <= 0 or sigma <= 0 or diffusion_fraction <= 0 or iterations <= 0:
        return data
    
    max_sigma = sigma * (growth ** max(iterations - 1, 0))
    image_size = min(data.shape[:2])
    if max_sigma > image_size / 6:
        print(f"Warning: diffusion filter size {max_sigma:.1f} pixels is too large for the image size {image_size}. Capping it to {image_size / 6:.1f} pixels.")
        max_sigma = image_size / 6
    
    radius = max(int(np.ceil(max_sigma * 3)), 0)
    result = np.pad(data, ((radius, radius), (radius, radius), (0, 0)), mode='reflect') if radius > 0 else data.copy()
    result_fft = np.fft.fft2(result, axes=(0, 1))
    for _ in range(iterations):
        blurred_fft = scipy.ndimage.fourier_gaussian(result_fft, sigma=(sigma, sigma, 0))
        result_fft = diffusion_fraction * blurred_fft + (1 - diffusion_fraction) * result_fft
        sigma *= growth
        diffusion_fraction *= decay
    result = np.fft.ifft2(result_fft, axes=(0, 1)).real

    if radius > 0:
        return result[radius:-radius, radius:-radius, :]
    return result


# Per-family strength-independent PSF shape, as three groups
# {core, halo, bloom} each expanded into a small geometric progression of
# 2D isotropic exponentials. The full PSF is
#
#     K_s(r) = w_c * K_core(r) + w_h * K_halo(r) + w_b * K_bloom(r)
#
# with w_c + w_h + w_b = 1 and each group itself a sum of N exponentials
#
#     K_x(r) = sum_k w_x[k] * E(r; lambda_x[k]),
#     E(r; lambda) = exp(-r / lambda) / (2 * pi * lambda**2),
#
# normalised so sum_k w_x[k] = 1 inside each group. Sub-component
# wavelengths span a geometric progression in [lambda_um / spread,
# lambda_um * spread]. For core and halo the within-group weights are
# uniform — every exponential in the progression contributes equally,
# which gives a smooth log-normal-ish blur with no internal knees. For
# the bloom the within-group weights follow a power-law schedule
# w_k ∝ lambda_k**(2 - alpha): a continuous integral of this schedule
# over lambda gives a 2D radial profile decaying as r**(-alpha) at
# large r, so the discretised bloom approximates a power-law with
# tail exponent alpha — replacing the explicit Student-t we had before
# while keeping everything in one basis family (no glued seams between
# exponential and power-law components).
#
# Why this is smoother than the previous core+halo+bloom-each-a-single-
# exponential model: adjacent groups' lambda ranges overlap, so the
# transition between e.g. halo and bloom happens through a partially
# overlapping band of scales rather than at one hard handoff radius.
# Energy distributes across scales continuously and the radial profile
# loses its visible knees.
#
# Per-family values are distilled from the private diffusion-filter
# characterization notes:
#
#     resolution preservation  : glimmerglass > BPM > pro_mist > cinebloom
#     halo prominence/radius   : cinebloom >= pro_mist >= BPM > glimmerglass
#     veil / shadow lift       : cinebloom > pro_mist >= BPM > glimmerglass
#     absorption (deep blacks) : BPM only (black-particle design)
#     halo color (warmth axis) : cinebloom > BPM > pro_mist > glimmerglass
#
# Halo color: each family carries a `halo_warmth_base` that drives a
# physically-motivated, energy-conserving redistribution of weights
# inside the halo group. Warmth > 0 pushes warm light (R + slight G)
# toward the OUTER halo sub-components and cool light (B) toward the
# inner ones (and vice versa for warmth < 0) — the warm-rim / cool-core
# look that reference images of mist and bloom filters consistently
# show around practical light sources. The per-channel weights are
# clipped and renormalized so total halo energy per channel is
# preserved exactly even when the modulation is strong. The user knob
# `DiffusionFilterParams.halo_warmth` adds on top of the per-family
# base. Bloom and core stay channel-independent.
_DIFFUSION_FILTER_SHAPES: dict[str, dict] = {
    # Glimmerglass — "tight, clean, sharp-preserving, faint long tail".
    # Core-dominant: most energy in the unscattered + diffraction-limited
    # core; halo small and tight; bloom small but with a non-negligible
    # long-reach tail (alpha ~3.2) — reference images show even the
    # subtlest mist filter lifts shadows slightly far from highlights, so
    # bloom must not be vestigial. Halo neutral in colour (the empirical
    # bluish flares are specular artifacts not modelled here). The user
    # character: subtle micro-contrast smoothing, almost no veiling,
    # blacks intact, but a measurable long decay around bright sources.
    'glimmerglass': {
        'core':  {'lambda_um':  10.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um':  50.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 260.0, 'spread': 2.5, 'n_components': 4, 'alpha': 3.2},
        'w_c': 0.60, 'w_h': 0.30, 'w_b': 0.10,
        'halo_warmth_base': 0.0,
    },
    # Black Pro-Mist — "concentrated punchy halo, fast falloff, deep blacks".
    # Halo-dominant: more weight in halo than in bloom, with a tighter halo
    # scale and a steeper bloom tail than pro_mist or cinebloom — energy
    # stays close to the highlight rather than veiling the frame. Warm
    # outer halo / cool inner core with the slight yellow-green bias the
    # empirical tests pick up around practical lights.
    'black_pro_mist': {
        'core':  {'lambda_um':  16.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um':  95.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 380.0, 'spread': 2.5, 'n_components': 4, 'alpha': 3.5},
        'w_c': 0.40, 'w_h': 0.47, 'w_b': 0.13,
        'halo_warmth_base': 0.65,
    },
    # Classic Pro-Mist — "atmospheric pastel, broad halo, balanced veil".
    # Halo broader than BPM, bloom heavier than BPM, balanced halo-and-
    # bloom weight split. Less localised than BPM, less frame-wide than
    # cinebloom; the look is a soft pastel mid-air haze. Warm outer halo
    # moderate.
    'pro_mist': {
        'core':  {'lambda_um':  14.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um': 150.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 650.0, 'spread': 2.5, 'n_components': 4, 'alpha': 2.9},
        'w_c': 0.28, 'w_h': 0.42, 'w_b': 0.30,
        'halo_warmth_base': 0.40,
    },
    # CineBloom — "frame-wide reach, slow tail, retro veil".
    # Bloom-dominant: bloom carries roughly half of the scattered energy,
    # with the longest scale and the shallowest tail of the four (alpha
    # near the heavy-tailed Lévy regime). The halo is the broadest of the
    # four but proportionally less peaked than BPM/pro_mist (large lambda,
    # smaller w_h share) — the reference images show a halo that "spreads
    # rather than punches". Warmest outer rim with the strongest yellow
    # lean. The tail does most of the visual work.
    'cinebloom': {
        'core':  {'lambda_um':  20.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um': 200.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 1000.0, 'spread': 2.5, 'n_components': 4, 'alpha': 2.5},
        'w_c': 0.22, 'w_h': 0.30, 'w_b': 0.48,
        'halo_warmth_base': 0.85,
    },
}

# Per-family scaling on the shared scatter-fraction saturation table. Sets
# the overall "deflection efficiency" of the filter at a given commercial
# stop. Glimmerglass deflects fewer photons than the mist filters
# (matching its mild perceived effect); BPM keeps a bit lower scatter than
# pro_mist (its "deep blacks" character lives in low scatter + low bloom
# weight, since the model is energy-conserving — see _strength_to_scatter
# docstring). Cinebloom is the heaviest scatterer but is held below
# pro_mist's gain so its strength=2 setting stays usable: at high p_s the
# very wide / shallow-tailed bloom would otherwise wash the frame.
_DIFFUSION_FAMILY_TOTAL_GAIN: dict[str, float] = {
    'glimmerglass':   0.65,
    'black_pro_mist': 0.75,
    'pro_mist':       1.05,
    'cinebloom':      1.00,
}

DIFFUSION_FILTER_FAMILIES: tuple[str, ...] = tuple(_DIFFUSION_FILTER_SHAPES)


# Strength -> deflected fraction p_s. Tabulated at commercial filter stops
# and log2-interpolated. Matches the saturation progression listed in plan
# §5 (BPM column, since the table was originally calibrated for the
# scattered+absorbed total).
_DIFFUSION_STRENGTH_BREAKPOINTS = np.array([0.125, 0.25, 0.5, 1.0, 2.0], dtype=np.float64)
_DIFFUSION_STRENGTH_TOTAL_FRACTION = np.array([0.10, 0.20, 0.35, 0.55, 0.75], dtype=np.float64)


def _strength_to_scatter(strength: float, family: str) -> float:
    """Map filter strength and family to p_s, the deflected-photon fraction.

    The diffusion-filter model is energy-conserving: the only effect of the
    filter is to redistribute photons between unscattered (T_0 = 1 - p_s)
    and scattered (p_s, convolved with the per-family PSF) populations.
    No absorption — output integrates to the same energy as input across
    all four families. BPM's empirical "deep blacks" property is encoded
    in its lower per-family scatter gain plus a smaller bloom weight than
    pro_mist or cinebloom, not in absorption.
    """
    if strength <= 0:
        return 0.0
    log_strength = np.log2(np.clip(strength, 1e-6, None))
    log_breaks = np.log2(_DIFFUSION_STRENGTH_BREAKPOINTS)
    base_total = float(np.interp(log_strength, log_breaks, _DIFFUSION_STRENGTH_TOTAL_FRACTION))
    gain = _DIFFUSION_FAMILY_TOTAL_GAIN.get(family, 1.0)
    return float(np.clip(base_total * gain, 0.0, 0.99))


def _expand_group(group_cfg: dict, *, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Expand a {core|halo|bloom} block into (lambdas_um, weights).

    Returns sub-component decay constants and weights summing to 1 inside
    the group; the family weight w_x is multiplied in by the caller. Sub-
    component lambdas are placed on a geometric progression in
    [lambda_um / spread, lambda_um * spread]. Within-group weights are:
      - uniform for `core` and `halo` (smooth log-normal-ish blur);
      - power-law w_k ∝ lambda_k**(2 - alpha) for `bloom`, so the
        assembled bloom decays as r**(-alpha) at large r.
    """
    lambda_center = float(group_cfg['lambda_um'])
    spread = float(group_cfg.get('spread', 1.0))
    n = max(int(group_cfg.get('n_components', 1)), 1)
    if n == 1 or spread <= 1.0:
        return np.array([lambda_center], dtype=np.float64), np.array([1.0], dtype=np.float64)

    log_lo = np.log(lambda_center / spread)
    log_hi = np.log(lambda_center * spread)
    lambdas = np.exp(np.linspace(log_lo, log_hi, n))

    if kind == 'bloom':
        alpha = float(group_cfg.get('alpha', 3.0))
        weights = lambdas ** (2.0 - alpha)
    else:
        weights = np.ones_like(lambdas)
    weights = weights / weights.sum()
    return lambdas, weights


# Per-channel coefficients on the warmth axis: positive a_c means warmth>0
# enhances channel c at the OUTER halo (and suppresses it at the inner
# halo, by the symmetric gradient defined in `_halo_channel_weights`).
# The axis is biased toward the yellow-green corner (R strong, G mild, B
# suppressed-and-some), so a "warm" outer halo reads as warm-yellow
# rather than pure red — matching the empirical observations of warm
# yellow rims on cinebloom and BPM around practical lights, with the
# inner halo cooler.
#
# Amplitude calibration (audit 2026-10): the magnitudes MUST stay low
# enough that `1 + eff * a_c * g` never drives a sub-component weight
# negative at the extremes (eff is clipped to ±1.5, g spans [-1, +1]),
# otherwise `_halo_channel_weights`'s `np.maximum(raw, 0)` hard-clips a
# channel to zero and the warmth knob degenerates into "channel dropout"
# instead of a mild radial tint. The worst-case product at full warmth is
# 1.5 * |a_c|, so |a_c| must be < 2/3 for the largest channel. We keep
# the largest at 0.62 (→ max modulation 0.93 < 1.0) plus a `tanh` soft
# saturation as a second line of defense, giving a visible but bounded
# tint at every setting with zero channel dropout. Compare the previous
# (+1.30, +0.15, -1.45): 1.5*1.45 = 2.18 ≫ 1, which zeroed R on the inner
# ring and B on the outer ring for cinebloom/BPM/pro_mist and left most of
# the positive warmth range with no response.
_HALO_CHANNEL_WARMTH_AXIS = np.array([+0.55, +0.07, -0.62], dtype=np.float64)


def _halo_channel_weights(weights: np.ndarray, warmth: float) -> np.ndarray:
    """Energy-conserving per-channel halo weight redistribution.

    Returns shape (3, N) array where each row is the halo sub-component
    weights for one of (R, G, B). Total halo energy per channel equals
    sum(weights) for any warmth, by construction:
      - g_k = symmetric "radial gradient" linearly from -1 at the
        innermost (smallest lambda) sub-component to +1 at the outermost,
        re-centred against `weights` so that sum_k weights[k] * g_k = 0.
      - per-channel coefficient a_c on the warmth axis (R warm, G mild
        warm, B cool).
      - modulation: weights[k] -> weights[k] * (1 + tanh(warmth * a_c * g_k)),
        soft-saturated so the multiplicative factor is always positive
        (tanh spans (-1, +1), so 1 + tanh spans (0, 2)), then renormalized
        per channel so the channel total equals sum(weights). The
        renormalization enforces exact energy conservation; the tanh keeps
        the redistribution a *continuous, bounded* tilt rather than the
        hard channel dropout that a linear `1 + x` clipped at zero produced
        (which set an entire ring's channel to zero once |x| ≥ 1).

    With g going inner→outer as -1→+1 and the warmth axis biased warm
    toward (R+, G+) and cool toward B-, warmth > 0 pushes warm light
    toward the OUTER halo and cool light toward the inner halo — the
    warm-rim-with-cool-core look observed in reference images of mist
    and bloom filters. Warmth < 0 inverts this.

    `warmth` is the already-soft-clamped *effective* warmth (family base +
    user bias). It is clamped again to [-1.5, 1.5] here as a guard; the
    caller is responsible for computing the effective value, but this
    makes the function safe to call with any scalar.
    """
    n = len(weights)
    if n < 2:
        return np.tile(weights, (3, 1))
    warmth = float(np.clip(warmth, -1.5, 1.5))
    g = np.linspace(-1.0, 1.0, n)
    g = g - np.average(g, weights=weights)
    target_total = float(np.sum(weights))
    out = np.empty((3, n), dtype=np.float64)
    for c in range(3):
        # tanh soft saturation: 1 + tanh(x) ∈ (0, 2) for all x, so the
        # per-component factor can never go negative — no hard dropout.
        raw = weights * (1.0 + np.tanh(warmth * _HALO_CHANNEL_WARMTH_AXIS[c] * g))
        raw = np.maximum(raw, 0.0)  # keep as a float-guard only (unreachable)
        s = raw.sum()
        if s > 0.0:
            out[c] = raw * (target_total / s)
        else:
            out[c] = weights
    return out


def _resolve_family_cfg(family: str, overrides: dict | None = None) -> dict:
    """Return the family cfg dict with optional per-group multipliers applied.

    Overrides keys (all default 1.0):
      - core_intensity, halo_intensity, bloom_intensity: scale the
        family weights w_c / w_h / w_b. The three are then renormalized
        so they still sum to 1, i.e. the kernel stays unit-normalised
        and the strength → p_s mapping is unchanged.
      - core_size, halo_size, bloom_size: scale each group's lambda_um
        uniformly (all sub-components in the group stretched together).
    """
    base = _DIFFUSION_FILTER_SHAPES[family]
    if overrides is None:
        return base
    ci = float(overrides.get('core_intensity', 1.0))
    hi = float(overrides.get('halo_intensity', 1.0))
    bi = float(overrides.get('bloom_intensity', 1.0))
    cs = float(overrides.get('core_size', 1.0))
    hs = float(overrides.get('halo_size', 1.0))
    bs = float(overrides.get('bloom_size', 1.0))
    if ci == 1.0 and hi == 1.0 and bi == 1.0 and cs == 1.0 and hs == 1.0 and bs == 1.0:
        return base
    w_c = float(base['w_c']) * max(ci, 0.0)
    w_h = float(base['w_h']) * max(hi, 0.0)
    w_b = float(base['w_b']) * max(bi, 0.0)
    total = w_c + w_h + w_b
    if total <= 0.0:
        return base
    return {
        **base,
        'core':  {**base['core'],  'lambda_um': float(base['core']['lambda_um'])  * max(cs, 1e-6)},
        'halo':  {**base['halo'],  'lambda_um': float(base['halo']['lambda_um'])  * max(hs, 1e-6)},
        'bloom': {**base['bloom'], 'lambda_um': float(base['bloom']['lambda_um']) * max(bs, 1e-6)},
        'w_c': w_c / total,
        'w_h': w_h / total,
        'w_b': w_b / total,
    }


def _bloom_max_lambda_um(family: str, overrides: dict | None = None) -> float:
    """Largest lambda in the bloom progression for a family (image-plane μm)."""
    cfg = _resolve_family_cfg(family, overrides)
    bloom = cfg['bloom']
    return float(bloom['lambda_um']) * float(bloom.get('spread', 1.0))


_OVERRIDE_KEYS = (
    'core_intensity', 'halo_intensity', 'bloom_intensity',
    'core_size', 'halo_size', 'bloom_size',
)


def _overrides_from_params(diffusion_filter) -> dict | None:
    """Pull per-group multipliers off a DiffusionFilterParams instance.

    Returns None if every multiplier is at its 1.0 default (so the fast
    path in `_resolve_family_cfg` skips the dict allocation).
    """
    out = {}
    any_set = False
    for key in _OVERRIDE_KEYS:
        v = getattr(diffusion_filter, key, 1.0)
        out[key] = float(v)
        if float(v) != 1.0:
            any_set = True
    return out if any_set else None


def _radial_components(
    radius_or_pixel_grid: np.ndarray,
    *,
    family: str,
    spatial_scale: float,
    pixel_size_um: float,
    halo_warmth: float,
    overrides: dict | None = None,
) -> dict[str, np.ndarray]:
    """Build core / per-channel halo / bloom radial contributions.

    `radius_or_pixel_grid` is in pixel units already if `pixel_size_um` is
    set to a finite value; for the analytic profile in image-plane μm,
    pass `pixel_size_um=1.0` and `radius_or_pixel_grid` already in μm.

    Returns weighted contributions K_x with the family weights w_x already
    multiplied in. The halo entry is shape (3, ...) per channel.
    """
    cfg = _resolve_family_cfg(family, overrides)
    spatial_scale = max(float(spatial_scale), 1e-6)

    core_lambdas, core_weights = _expand_group(cfg['core'], kind='core')
    halo_lambdas, halo_weights = _expand_group(cfg['halo'], kind='halo')
    bloom_lambdas, bloom_weights = _expand_group(cfg['bloom'], kind='bloom')

    halo_per_ch = _halo_channel_weights(halo_weights, halo_warmth)

    core_lambdas_px = core_lambdas * spatial_scale / pixel_size_um
    halo_lambdas_px = halo_lambdas * spatial_scale / pixel_size_um
    bloom_lambdas_px = bloom_lambdas * spatial_scale / pixel_size_um

    r = np.asarray(radius_or_pixel_grid, dtype=np.float64)

    def _exp_sum(lambdas_px: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Discrete-area-normalised exponential mixture. Each sub-component
        # E(r; λ) = exp(-r/λ)/(2πλ²) is a probability density in the
        # continuous plane (∫ 2πr E dr = 1), but on a finite pixel grid the
        # point-sampled kernel only integrates to ~1 if λ is several pixels.
        # When λ_px ≲ 1 (the core's 10–20 μm at a 36 μm/px preview), the
        # point-sampled value at r=0 is 1/(2πλ²) which *under-counts* the
        # unit mass the sub-component should carry, so the excess leaks to
        # the wider components after the final per-channel normalisation —
        # that leak is exactly the preview/export inconsistency found in the
        # audit (highlights over-flattened at low resolution). Normalising
        # each component to unit grid-sum *before* mixing keeps the intended
        # core:halo:bloom energy split at every resolution; the residual
        # difference from a true continuous convolution is then second-order
        # (kernel shape), not first-order (energy reallocation).
        total = np.zeros_like(r)
        for wk, lk in zip(weights, lambdas_px):
            lk = max(float(lk), 1e-6)
            k = np.exp(-r / lk) / (2.0 * np.pi * lk ** 2)
            s = k.sum()
            if s > 0.0:
                k = k / s
            total += wk * k
        return total

    core = cfg['w_c'] * _exp_sum(core_lambdas_px, core_weights)
    bloom = cfg['w_b'] * _exp_sum(bloom_lambdas_px, bloom_weights)

    halo_channels = np.stack(
        [cfg['w_h'] * _exp_sum(halo_lambdas_px, halo_per_ch[c]) for c in range(3)],
        axis=0,
    )

    return {'core': core, 'halo': halo_channels, 'bloom': bloom}


def diffusion_filter_radial_profile(
    radius_um: np.ndarray,
    *,
    family: str = 'black_pro_mist',
    spatial_scale: float = 1.0,
    halo_warmth: float = 0.0,
    overrides: dict | None = None,
) -> dict[str, np.ndarray]:
    """Analytic radial profile of the diffusion-filter PSF, unit-normalised in 2D.

    Returns each component contribution in 1/μm**2:
      - 'core', 'bloom': shape matching `radius_um` (channel-independent).
      - 'halo': shape (3, *radius_um.shape), one per channel.
      - 'total_per_channel': shape (3, *radius_um.shape), sum of all three.
    `halo_warmth` is added to the family base before the channel weights
    are redistributed. `overrides` accepts the same per-group multiplier
    keys as `_resolve_family_cfg`.
    """
    if family not in _DIFFUSION_FILTER_SHAPES:
        raise ValueError(f"Unknown diffusion filter family: {family!r}; "
                         f"available: {list(_DIFFUSION_FILTER_SHAPES)}")
    cfg = _resolve_family_cfg(family, overrides)
    effective_warmth = float(np.clip(
        float(cfg.get('halo_warmth_base', 0.0)) + float(halo_warmth), -1.5, 1.5))
    parts = _radial_components(
        np.asarray(radius_um, dtype=np.float64),
        family=family,
        spatial_scale=spatial_scale,
        pixel_size_um=1.0,
        halo_warmth=effective_warmth,
        overrides=overrides,
    )
    total_per_channel = parts['halo'] + parts['core'][None, ...] + parts['bloom'][None, ...]
    return {
        'core': parts['core'],
        'halo': parts['halo'],
        'bloom': parts['bloom'],
        'total_per_channel': total_per_channel,
    }


def diffusion_filter_psf(
    kernel_shape: tuple[int, int],
    *,
    family: str,
    spatial_scale: float,
    pixel_size_um: float,
    halo_warmth: float = 0.0,
    overrides: dict | None = None,
) -> np.ndarray:
    """Per-channel 2D PSF for a diffusion filter.

    Returns shape (height, width, 3), each channel sum-normalised on the
    grid. The PSF is per-channel because halo weights vary per channel
    via the energy-conserving warmth redistribution; core and bloom
    contributions are channel-independent and shared across the three.
    `overrides` accepts the same per-group multiplier keys as
    `_resolve_family_cfg`.
    """
    if family not in _DIFFUSION_FILTER_SHAPES:
        raise ValueError(f"Unknown diffusion filter family: {family!r}; "
                         f"available: {list(_DIFFUSION_FILTER_SHAPES)}")
    cfg = _resolve_family_cfg(family, overrides)
    effective_warmth = float(np.clip(
        float(cfg.get('halo_warmth_base', 0.0)) + float(halo_warmth), -1.5, 1.5))

    y, x = np.ogrid[:kernel_shape[0], :kernel_shape[1]]
    cy, cx = kernel_shape[0] // 2, kernel_shape[1] // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float64)

    parts = _radial_components(
        r,
        family=family,
        spatial_scale=spatial_scale,
        pixel_size_um=pixel_size_um,
        halo_warmth=effective_warmth,
        overrides=overrides,
    )
    psf = np.empty((kernel_shape[0], kernel_shape[1], 3), dtype=np.float64)
    for c in range(3):
        psf[..., c] = parts['core'] + parts['halo'][c] + parts['bloom']
        psf[..., c] /= psf[..., c].sum()
    return psf


def apply_diffusion_filter_um(image, diffusion_filter, pixel_size_um):
    """Apply a diffusion-filter PSF to an RGB image.

    Implements the energy-conserving convex combination
        E_out = (1 - p_s) * E_in  +  p_s * (K_s * E_in)
    with p_s derived from `strength` and `filter_family`. K_s is per-channel
    because the halo is colour-tinted via an energy-conserving redistribution
    across its sub-components; total scatter energy per channel = p_s.

    Performance (audit 2026-10): the PSF is split into a full-resolution
    core+halo pass and a downsampled bloom pass. The bloom's sub-component
    wavelengths (λ up to ~2500 μm at 1× scale) make a single-resolution
    spatial kernel enormous (radius ~8·λ ≈ 2000 px at a 6 μm/px export, a
    384 MB kernel and a ~60 s convolution); downsampling the bloom pass by
    4× cuts its radius and cost by ~16× with negligible error (bloom is a
    low-pass tail — verified PSNR > 89 dB vs. the single-resolution PSF
    across all four families). Both passes run in float32 via a shared
    scipy.fft forward transform, giving previews at ~300 ms and exports at
    ~9 s instead of ~60 s.
    """
    if not diffusion_filter.active:
        return image
    if diffusion_filter.strength <= 0 or diffusion_filter.spatial_scale <= 0:
        return image
    family = diffusion_filter.filter_family
    if family not in _DIFFUSION_FILTER_SHAPES:
        raise ValueError(f"Unknown diffusion filter family: {family!r}; "
                         f"available: {list(_DIFFUSION_FILTER_SHAPES)}")

    p_s = _strength_to_scatter(diffusion_filter.strength, family)
    if p_s <= 0:
        return image

    overrides = _overrides_from_params(diffusion_filter)
    cfg = _resolve_family_cfg(family, overrides)
    effective_warmth = float(np.clip(
        float(cfg.get('halo_warmth_base', 0.0)) + float(getattr(diffusion_filter, 'halo_warmth', 0.0)),
        -1.5, 1.5,
    ))

    core_l, core_w = _expand_group(cfg['core'], kind='core')
    halo_l, halo_w = _expand_group(cfg['halo'], kind='halo')
    bloom_l, bloom_w = _expand_group(cfg['bloom'], kind='bloom')
    halo_per_ch = _halo_channel_weights(halo_w, effective_warmth)

    w_c, w_h, w_b = cfg['w_c'], cfg['w_h'], cfg['w_b']
    scale = max(float(diffusion_filter.spatial_scale), 1e-6)
    H, W = image.shape[:2]

    # ---- GPU fast path -------------------------------------------------
    # Optional: identical model, cuFFT instead of pocketfft. Falls through to
    # the CPU implementation on any failure (missing driver, OOM, unsupported
    # kernel shape) so an export can never be lost to a GPU hiccup.
    _gpu = _gpu_backend()
    if _gpu is not None and _GPU_STATE.get("max_px") is not None and H * W > _GPU_STATE["max_px"]:
        _gpu = None          # known to exceed this device's budget — skip the attempt
    if _gpu is not None:
        try:
            out = _apply_diffusion_gpu(
                _gpu, image, p_s, core_l, core_w, halo_l, halo_per_ch,
                bloom_l, bloom_w, w_c, w_h, w_b, scale, pixel_size_um,
            )
            # CuPy's pool keeps the peak allocation for the rest of the session.
            # At 24 MP that is ~1 GB, at 37 MP ~5 GB because cuFFT's workspace
            # grows with the padding — enough that the next frame would OOM.
            # Exports are one image at a time, so hand it back.
            _release_gpu_memory(_gpu)
            return out
        except Exception:
            # Remember that this size does not fit, then retry on the CPU. The
            # GPU stays enabled for smaller frames instead of being switched off
            # for the rest of the session.
            _GPU_STATE["max_px"] = max(int(H * W * 0.9), 0) or None
            _release_gpu_memory(_gpu)

    # ---- core + halo at full resolution -------------------------------
    lam_max_ch_px = max(float(core_l.max()), float(halo_l.max())) * scale / pixel_size_um
    radius_ch = int(np.ceil(8.0 * lam_max_ch_px))
    radius_ch = max(radius_ch, 3)
    radius_ch = min(radius_ch, min(H, W) // 2 - 1)
    if radius_ch < 1:
        radius_ch = 1

    kcore = _build_group_kernel(2 * radius_ch + 1, core_l, core_w, scale, pixel_size_um)
    khalo = [
        _build_group_kernel(2 * radius_ch + 1, halo_l, halo_per_ch[c], scale, pixel_size_um)
        for c in range(3)
    ]
    # core+halo 组合核（已按 w_c/(w_c+w_h)、w_h/(w_c+w_h) 归一到单位质量）
    k_full = [((w_c * kcore + w_h * khalo[c]) / (w_c + w_h)).astype(np.float32) for c in range(3)]

    # ---- bloom at 4x downsample ----------------------------------------
    bloom_scale = 4
    Hs, Ws = max(H // bloom_scale, 1), max(W // bloom_scale, 1)
    pxs = pixel_size_um * bloom_scale
    lam_max_b_px = float(bloom_l.max()) * scale / pxs
    radius_b = int(np.ceil(8.0 * lam_max_b_px))
    radius_b = max(radius_b, 3)
    radius_b = min(radius_b, min(Hs, Ws) // 2 - 1)
    if radius_b < 1:
        radius_b = 1
    kbloom = _build_group_kernel(2 * radius_b + 1, bloom_l, bloom_w, scale, pxs).astype(np.float32)

    # ---- convolve ------------------------------------------------------
    img32 = image.astype(np.float32)
    out = np.empty_like(img32)

    # core+halo 全分辨率卷积（三个通道共享一次 FFT）
    out = _convolve_fullres(img32, k_full, radius_ch)

    # bloom 降采样卷积
    small = _downsample(img32, bloom_scale)
    bloom_blur = _convolve_fullres(small, [kbloom] * 3, radius_b)
    bloom_up = _upsample(bloom_blur, (H, W))

    tot = w_c + w_h + w_b
    a = (w_c + w_h) / tot * p_s
    b = w_b / tot * p_s
    keep = 1.0 - p_s
    # Fused in one pass where cv2 is available; the explicit form is only a
    # fallback and allocates two full-frame temporaries.
    try:
        import cv2
        mixed = cv2.addWeighted(out, a, bloom_up, b, 0.0)
        return cv2.addWeighted(mixed, 1.0, img32, keep, 0.0)
    except Exception:
        mixed = a * out + b * bloom_up
        return keep * img32 + mixed


def _build_group_kernel(size, lambdas_um, weights, spatial_scale, pixel_size_um):
    """Build a unit-mass 2D kernel for one group (per-component normalised).

    Each sub-component E(r; λ) is point-sampled on a size×size grid and
    independently normalised to unit grid-sum (see the audit note in
    `_radial_components`), then mixed by `weights` (which sum to 1 within
    the group). Returns float64.
    """
    R = size // 2
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - R) ** 2 + (y - R) ** 2).astype(np.float64)
    total = np.zeros((size, size), dtype=np.float64)
    for wk, lk in zip(weights, lambdas_um):
        lp = max(float(lk) * spatial_scale / pixel_size_um, 1e-6)
        k = np.exp(-r / lp) / (2.0 * np.pi * lp ** 2)
        s = k.sum()
        if s > 0.0:
            k = k / s
        total += float(wk) * k
    return total


def _convolve_fullres(img32, kernels32, radius):
    """Convolve img32 (H,W,3) float32 with one kernel per channel.

    Uses a single shared scipy.fft forward transform of the reflect-padded
    image; each channel's kernel is transformed and multiplied separately.
    Returns float32 (H,W,3).
    """
    from scipy import fft as sfft
    H, W = img32.shape[:2]
    try:
        import cv2
        padded = cv2.copyMakeBorder(img32, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    except Exception:
        padded = np.pad(img32, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
    Ph, Pw = padded.shape[:2]
    Fi = sfft.rfft2(padded, axes=(0, 1), workers=-1)   # (Ph, Pw//2+1, 3)
    out = np.empty_like(img32)
    for c in range(3):
        k = kernels32[c]
        kp = np.zeros((Ph, Pw), dtype=np.float32)
        kh, kw = k.shape
        # centre the kernel at (0,0) for the FFT (fftshift equivalent)
        kp[:kh // 2 + 1, :kw // 2 + 1] = k[kh // 2:, kw // 2:]
        kp[:kh // 2 + 1, Pw - kw // 2:] = k[kh // 2:, :kw // 2]
        kp[Ph - kh // 2:, :kw // 2 + 1] = k[:kh // 2, kw // 2:]
        kp[Ph - kh // 2:, Pw - kw // 2:] = k[:kh // 2, :kw // 2]
        Fk = sfft.rfft2(kp, workers=-1)           # (Ph, Pw//2+1)
        conv = sfft.irfft2(Fi[..., c] * Fk, s=(Ph, Pw), workers=-1)   # (Ph, Pw)
        out[:, :, c] = conv[radius:radius + H, radius:radius + W]
    return out


def _downsample(img32, factor):
    """Area-average downsample by an integer factor (no ringing)."""
    H, W = img32.shape[:2]
    Hs, Ws = max(H // factor, 1), max(W // factor, 1)
    try:
        import cv2
        return cv2.resize(img32, (Ws, Hs), interpolation=cv2.INTER_AREA)
    except Exception:
        return img32[:Hs * factor, :Ws * factor].reshape(Hs, factor, Ws, factor, 3).mean(axis=(1, 3))


def _upsample(img32, shape):
    """Bilinear upsample back to `shape` (the bloom pass is a smooth tail)."""
    H, W = shape
    try:
        import cv2
        return cv2.resize(img32, (W, H), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from scipy.ndimage import zoom
        h, w = img32.shape[:2]
        return zoom(img32, (H / h, W / w, 1.0), order=1, prefilter=False)



