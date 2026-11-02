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

# --- Highlights / shadows weight calibration -------------------------------------------
# The weight decides *where* a slider bites, so it is a calibration, not a taste knob.
#
# Highlights key off max(R,G,B) because over-exposure is a per-channel event: a saturated
# neon, tail light or sunset highlight reaches the clip point while its luminance is still
# low, and keying off luminance left pure blue at weight 0.00 (the slider did nothing).
# The range is calibrated against what a Clip decode can actually reach - it stops at 1.0,
# so an "full weight 4 stops over 18% grey" curve (q = 2.88) never exceeded 67% authority.
# Full weight now lands exactly on the clip point, 2.4709 stops over grey.
#
# The foot sits on 18% grey rather than well above it. A narrow ramp is worse than a wide one:
# with the foot at 2.0 stops the response went from 0 to full inside 0.97 stops, so a sky that
# straddled the threshold came out in two flat plateaus (42 levels of shift on the bright side,
# 3.6 on the dark side, nothing in between) - which reads exactly like a hard cutoff. Over
# 2.47 stops the same sky gets a continuous ramp, and because a smoothstep has zero slope at
# its foot, 18% grey and below are still untouched.
#
# Shadows were calibrated the mirror image of that mistake: full weight at -4 stops meant the
# steep part of the ramp sat at sRGB 30, while what an eye calls "shadow" is sRGB 60-120. So
# only the blackest corner lifted and everything else stayed put - the slider looked like it
# "only brightens part of the frame". Full weight now sits at -2.2 stops (sRGB ~45).
HS_HL_LO = 0.0          # log2(q / 0.18): q = 0.18 - the smoothstep's zero-slope foot
HS_HL_HI = 2.4709       # log2(1.0 / 0.18): q = 1.0, the decode ceiling
# Shadows use a BELL, not a ramp. A ramp reaching full weight below some threshold lifts the
# blacks along with the shadows: with the ramp at -2.2 stops the 2nd percentile went from 5.9 to
# 30 and the midtones moved further than the shadows did (+51 levels in 40-70 against +27 in
# 0-20), which reads as grey fog rather than "the shadows came up". A multiplicative lift always
# moves bright pixels further in absolute terms, so the weight has to fall back to zero again at
# the very bottom. t is 0 at 18% grey and 1 at -6 stops; t*(1-t)*4 peaks at -3 stops (y=0.0225,
# sRGB ~50) and is 0 at 18% grey and at -6 stops, so pure black stays black.
HS_SH_SPAN = 6.0        # stops below 18% grey mapped onto t in [0, 1]
# How many stops a slider at +/-100 is worth. Highlights use 1.0 stop and shadows 2.0: a
# highlight reduction is read against what a converter like ACR does at its own -100, which is
# roughly one stop concentrated above the midtones; two stops read as "the whole picture just
# got darker". The shadows side keeps its full 2 stops because there the complaint was the
# opposite - the slider could not lift anything.
HS_HL_STOPS = 0.01
HS_SH_STOPS = 0.015

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


def scene_gain(exposure_ev, temp, base_temp, tint, base_tint):
    """Exposure and white balance folded into one per-channel gain.

    The shader receives this as ``u_scene_gain``: ``compute_scene_radiance`` is evaluated once
    per tap inside the diffusion loop (32x per pixel), and the WB solve - one log2, three
    exp2, a divide and a dot - does not depend on the tap. Lifting it out of the loop removes
    four transcendental calls per tap with no change in the result.
    """
    g = wb_gains(temp, base_temp, tint, base_tint)
    k = 2.0 ** float(exposure_ev)
    return (float(g[0]) * k, float(g[1]) * k, float(g[2]) * k)


def hs_weights(col):
    """Highlights (w_hl) and shadows (w_sh) weights for scene-linear ProPhoto ``col``.

    Highlights are a 0..1 smoothstep ramp; shadows are a 0..1 bell (see HS_SH_SPAN). Highlights
    read the brightest CHANNEL, shadows read ProPhoto luminance, because "is it dark" is a
    luminance question - a saturated dark blue is bright in its own channel but reads near-black.

    Returns ``(w_hl, w_sh)`` as arrays broadcastable against ``col``.
    """
    col = np.asarray(col, dtype=np.float32)
    hi = col.max(axis=-1)
    lum = np.einsum('...c,c->...', col, PROPHOTO_LUMA)

    # Below the thresholds the weight is exactly zero, and log2 of a near-zero argument is
    # the only thing that could produce a NaN here, so clamp the arguments first.
    span = HS_HL_HI - HS_HL_LO
    t_hl = np.clip((np.log2(np.maximum(hi, np.float32(1e-9)) / np.float32(0.18)) - HS_HL_LO)
                   / np.float32(span), 0.0, 1.0)
    w_hl = t_hl * t_hl * (np.float32(3.0) - np.float32(2.0) * t_hl)

    t_sh = np.clip(-np.log2(np.maximum(lum, np.float32(1e-9)) / np.float32(0.18))
                   / np.float32(HS_SH_SPAN), 0.0, 1.0)
    # Bell, not a ramp: zero at both ends. The noise floor needs no separate guard - log2
    # saturating to t = 1 lands on a zero weight by construction.
    w_sh = t_sh * (np.float32(1.0) - t_sh) * np.float32(4.0)
    return w_hl, w_sh


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
