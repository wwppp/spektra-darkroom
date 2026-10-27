"""Shared CPU implementations of darkroom adjustments.

These functions mirror the GLSL shader in ui/canvas_viewport.py exactly, so that
GPU preview, CPU export and thumbnails stay consistent. Any formula change here
must be reflected in the shader (and vice versa).
"""

import math
import numpy as np

# ProPhoto (ROMM) RGB luminance weights - the scene-linear working space
PROPHOTO_LUMA = np.array([0.2880, 0.7119, 0.0001], dtype=np.float32)

# White balance: per-stop strength of red/blue gain vs. log2(T / T_base)
WB_STRENGTH = 0.40

# Paper response model used for enlarger-side adjustments: print = 1 / (1 + E^PAPER_GAMMA)
PAPER_GAMMA = 2.5
PAPER_CLIP = 1e-4
# Enlarger CMY slider unit expressed as log10 exposure change per slider unit
CMY_LOG_UNIT = 0.004


def wb_gains(temp, base_temp, tint, base_tint):
    """Exponential, luminance-preserving white-balance gains (R, G, B)."""
    b_temp = base_temp if base_temp > 1000.0 else 5500.0
    b_tint = base_tint if base_tint > 0.1 else 1.0
    s = math.log2(max(1000.0, float(temp)) / b_temp)
    g = np.array([2.0 ** (WB_STRENGTH * s), float(tint) / b_tint, 2.0 ** (-WB_STRENGTH * s)], dtype=np.float32)
    y = float(np.dot(g, PROPHOTO_LUMA))
    return g / max(1e-4, y)


def paper_domain_adjust(img, cmy=(0.0, 0.0, 0.0), print_exp=1.0, preflash=0.0):
    """Enlarger CMY filtration, print exposure and pre-flash applied in paper-exposure domain.

    Inverts the output through the paper model, scales/offsets the paper exposure
    per channel, and maps back through the paper curve.
    """
    cmy = np.asarray(cmy, dtype=np.float32)
    if np.all(np.abs(cmy) < 0.01) and abs(print_exp - 1.0) < 0.005 and preflash <= 0.0005:
        return img
    pv = np.clip(img, PAPER_CLIP, 1.0 - PAPER_CLIP)
    e = (1.0 / pv - 1.0) ** (1.0 / PAPER_GAMMA)
    gain = (10.0 ** (-CMY_LOG_UNIT * cmy)) / max(0.05, float(print_exp))
    e = e * gain.astype(np.float32) + max(0.0, float(preflash))
    return (1.0 / (1.0 + e ** PAPER_GAMMA)).astype(np.float32)


def neutral_baked():
    return {
        "dir_amount": 1.0,
        "dir_interlayer": 1.0,
        "dir_samelayer": 1.0,
        "morph_gamma": 1.0,
        "developer_exhaustion": 0.0,
    }


def chemistry_delta(img, params, baked=None, apply_dir=True, apply_morph=True):
    """Approximate DIR / paper-morph adjustments for the part NOT already baked into the LUT.

    `baked` holds the values the current LUT was generated with; only the delta
    between the requested value and the baked value is approximated here.
    """
    b = baked or neutral_baked()
    cur_amt = float(params.get("dir_amount", 1.0))
    bak_amt = float(b.get("dir_amount", 1.0))
    eff_inter = cur_amt * float(params.get("dir_interlayer", 1.0))
    bak_inter = bak_amt * float(b.get("dir_interlayer", 1.0))
    eff_same = cur_amt * float(params.get("dir_samelayer", 1.0))
    bak_same = bak_amt * float(b.get("dir_samelayer", 1.0))
    d_inter = eff_inter - bak_inter
    d_same = eff_same - bak_same
    out = img
    if apply_dir and (abs(d_inter) > 0.01 or abs(d_same) > 0.01):
        # Asymmetric cross-layer coupler inhibition in paper log-exposure domain
        # Mirrors spektrafilm's M_inter (R/G layers strongly inhibit G/B layers) + M_self
        pv = np.clip(out, PAPER_CLIP, 1.0 - PAPER_CLIP)
        le = np.log10(1.0 / pv - 1.0) / PAPER_GAMMA
        # Linear regression of spektrafilm negative-film DIR coupler response in log10(E_paper):
        a_inter = np.array([
            [-0.0766, -0.2029, -0.2196],
            [-0.0977, -0.0508, -0.0269],
            [-0.1714, -0.2069, -0.1958],
        ], dtype=np.float32)
        b_inter = np.array([0.0370, 0.0558, 0.0753], dtype=np.float32)
        a_same = np.array([
            [-0.0195, -0.0597, -0.0596],
            [-0.0039,  0.0227, -0.0057],
            [-0.0537, -0.0568, -0.0212],
        ], dtype=np.float32)
        b_same = np.array([0.0105, 0.0167, 0.0197], dtype=np.float32)
        dle = d_inter * (le @ a_inter + b_inter) + d_same * (le @ a_same + b_same)
        e_new = 10.0 ** ((le + dle) * PAPER_GAMMA)
        out = (1.0 / (1.0 + e_new)).astype(np.float32)
    if apply_morph:
        rg = float(params.get("morph_gamma", 1.0)) / max(0.05, float(b.get("morph_gamma", 1.0)))
        if abs(rg - 1.0) > 0.01:
            # S-curve contrast around paper mid-grey pivot (0.42), matching spektrafilm s023 coupled-gamma morph
            pivot = 0.42
            c = np.clip(out, 1e-4, 1.0 - 1e-4)
            ratio = (c / pivot) / ((1.0 - c) / (1.0 - pivot))
            odds = ratio ** rg
            out = (odds * pivot) / (odds * pivot + (1.0 - pivot))
        d_exh = float(params.get("developer_exhaustion", 0.0)) - float(b.get("developer_exhaustion", 0.0))
        if abs(d_exh) > 0.005:
            # Mirrors spektrafilm s023 Gumbel-max CDF morph: D(0) and D_max fixed,
            # asymmetrically softens print high-density shoulder and lifts upper-midtones
            c = np.clip(out, 0.0, 1.0)
            out = np.clip(out + d_exh * c * (1.0 - c) * (c - 0.42) * 0.35, 0.0, 1.0)
    return out.astype(np.float32)


def grain_colour_rho(stock_strength, grain_colour=1.0):
    """Inter-layer grain correlation. Scanned colour negatives read mostly as luminance grain
    with subtle low-frequency colour mottle; coarser (pushed / high-speed) stocks are slightly
    more correlated so grain reads as clumps rather than chroma speckle."""
    rho_colour = min(0.95, max(0.88, 0.88 + (float(stock_strength) - 0.011) * 10.0))
    c = min(1.0, max(0.0, float(grain_colour)))
    return 1.0 + (rho_colour - 1.0) * c


def grain_scale_px(fmt_mm, master_long_px, grain_size=1.0, stock_size_mm=0.005):
    """Grain feature scale in master pixels, from physical film-plane pixel pitch.
    Calibrated so 35mm @ 6024 px (~5.98 um/px) gives 1.0 for a 5 um stock."""
    um_per_px = (max(5.0, float(fmt_mm)) * 1000.0) / max(1.0, float(master_long_px))
    return max(0.20, (5.976 / um_per_px) * max(0.2, float(grain_size)) * (float(stock_size_mm) / 0.005))


def apply_black_point(img, black_lift=0.0):
    """Bidirectional black-point calibration preserving the white point (1.0).
    - black_lift > 0: lifts Dmax black point toward grey (vintage fade / airy shadows) without clipping highlights.
    - black_lift < 0: subtracts base fog and deepens shadows toward pure black (dehaze / solid blacks).
    """
    b = float(black_lift)
    if abs(b) <= 1e-4:
        return img
    if b > 0.0:
        return b + img * (1.0 - b)
    c = min(0.95, -b)
    return np.maximum(0.0, (img - c) / (1.0 - c)).astype(np.float32)
