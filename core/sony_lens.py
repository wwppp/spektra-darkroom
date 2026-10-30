"""sony_lens.py — read the lens-correction tables Sony writes into its own ARW files.

Why this exists
---------------
Sony never published the format, but every ARW carries three spline tables in a SubIFD of
IFD0 (reachable through tag 0x014a, or 0x7200 for the SR2 copy):

    0x7031 VignettingCorrection        / 0x7032 VignettingCorrParams
    0x7034 ChromaticAberrationCorrection / 0x7035 ChromaticAberrationCorrParams
    0x7036 DistortionCorrection        / 0x7037 DistortionCorrParams

The neighbouring Adobe LCP route is a dead end for these bodies: Sony's LCP files carry the
distortion and vignette models but no ``ChromaticRedGreenModel`` at all, and the 312 Sony
LCPs on this machine contain no DSC-RX100 entry — ``match_lens_from_exif`` returns None.

Encoding (cross-checked against darktable, RawTherapee and two independent write-ups)
-------------------------------------------------------------------------------------
Each table is a count-prefixed array of signed 16-bit values; the first value is the number
of live knots and the rest is padding. Knot ``i`` sits at ``i / (nc - 1)`` of the corner
radius, so knot 0 is the optical centre.

    distortion  g(r) = 1 + v * 2^-14, sampled on the knots, linearly interpolated
    lateral CA  g(r) = 1 + v * 2^-21, same knots, red half then blue half
    vignette    gain(r) = 1 / 2^(0.5 - 2^(v * 2^-13 - 1))

Two details that are easy to get wrong and cost real accuracy:

  * The distortion spline is **normalised by its maximum** before use. On the RX100 the first
    knot is +958, i.e. 5.85% — reading it as an absolute offset would mean the optical centre
    is already distorted. stannum's reverse-engineering puts a ``/ s`` with ``s = max g`` in
    the mapping; measured against the camera's own JPEG, normalising takes the geometry
    residual from 8.2 px to 1.6 px at a 1368 px width.
  * The result must **not** be re-normalised so the corner lands back on 1.0. This module
    therefore emits ``dist_scale = 1.0`` and lets the renderer keep the corrected frame's
    natural (slightly narrower) field of view, which is what the camera's own JPEG shows.
    Re-applying the renderer's corner normalisation pushed the residual back to 12.6 px.

Calibration (DSC02729, RX100M2 @ 10.4 mm, 2048x1368 decode vs the 1616x1080 embedded JPEG,
mean residual of a block phase-correlation field after fitting one global affine):

    no correction        8.24 px
    alpha 0.8 / 1.0 / 1.2   2.48 / 1.58 / 4.70 px

so the table's units are right as written and no fudge factor is used.
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}

IFD0_SUBIFDS = 0x014A
SR2_SUBIFD_OFFSET = 0x7200
TAG_VIG_PARAMS = 0x7032
TAG_CA_PARAMS = 0x7035
TAG_DIST_PARAMS = 0x7037
TAG_VIG_SETTING = 0x7031
TAG_CA_SETTING = 0x7034
TAG_DIST_SETTING = 0x7036

DIST_EXP = -14.0
CA_EXP = -21.0
VIG_EXP = -13.0

# Min/max accepted knot counts. Sony writes 16 knots on full frame and 11 in APS-C crop mode
# for distortion and vignette, 32 / 22 for CA (red half then blue half).
_MIN_KNOTS, _MAX_KNOTS = 2, 16
_MIN_CA_KNOTS, _MAX_CA_KNOTS = 4, 32


def _read_ifd(data: bytes, off: int, endian: str) -> dict:
    if off <= 0 or off + 2 > len(data):
        return {}
    n = struct.unpack(endian + "H", data[off:off + 2])[0]
    out = {}
    for i in range(n):
        e = off + 2 + i * 12
        if e + 12 > len(data):
            break
        tag, typ, cnt = struct.unpack(endian + "HHI", data[e:e + 8])
        size = _SIZES.get(typ, 1) * cnt
        voff = e + 8 if size <= 4 else struct.unpack(endian + "I", data[e + 8:e + 12])[0]
        out[tag] = (typ, cnt, voff, e)
    return out


def _int16s(data: bytes, entry, endian: str) -> np.ndarray:
    typ, cnt, voff, _ = entry
    n = _SIZES.get(typ, 1) * cnt
    if voff + n > len(data):
        return np.zeros(0, np.int32)
    raw = data[voff:voff + n]
    if typ == 3:      # SHORT
        return np.frombuffer(raw, dtype=endian + "u2", count=cnt).astype(np.int32)
    if typ == 8:      # SSHORT
        return np.frombuffer(raw, dtype=endian + "i2", count=cnt).astype(np.int32)
    return np.zeros(0, np.int32)


def _payload(vals: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Drop the count prefix and the padding; reject counts outside [lo, hi]."""
    if vals.size == 0:
        return np.zeros(0, np.float64)
    n = int(vals[0])
    if n < lo or n > hi or n + 1 > vals.size:
        return np.zeros(0, np.float64)
    return vals[1:n + 1].astype(np.float64)


def read_tables(path: str) -> Optional[dict]:
    """Return the three raw knot arrays, or None when the file carries no Sony tables."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        return None
    ifd0 = _read_ifd(data, struct.unpack(endian + "I", data[4:8])[0], endian)
    if not ifd0:
        return None

    sub = {}
    if IFD0_SUBIFDS in ifd0:
        e = ifd0[IFD0_SUBIFDS][3]
        sub = _read_ifd(data, struct.unpack(endian + "I", data[e + 8:e + 12])[0], endian)
    if TAG_DIST_PARAMS not in sub and SR2_SUBIFD_OFFSET in ifd0:
        e = ifd0[SR2_SUBIFD_OFFSET][3]
        sub = _read_ifd(data, struct.unpack(endian + "I", data[e + 8:e + 12])[0], endian)
    if TAG_DIST_PARAMS not in sub:
        return None

    def grab(tag, lo, hi):
        return _payload(_int16s(data, sub[tag], endian), lo, hi) if tag in sub \
            else np.zeros(0, np.float64)

    def flag(tag):
        if tag not in sub:
            return None
        v = _int16s(data, sub[tag], endian)
        return int(v[0]) if v.size else None

    ca = grab(TAG_CA_PARAMS, _MIN_CA_KNOTS, _MAX_CA_KNOTS)
    half = ca.size // 2
    return {
        "distortion": grab(TAG_DIST_PARAMS, _MIN_KNOTS, _MAX_KNOTS),
        "ca_r": ca[:half],
        "ca_b": ca[half:],
        "vignette": grab(TAG_VIG_PARAMS, _MIN_KNOTS, _MAX_KNOTS),
        # Only records what the camera did to its own JPEG; the correction is built regardless.
        "flags": {"distortion": flag(TAG_DIST_SETTING), "ca": flag(TAG_CA_SETTING),
                  "vignette": flag(TAG_VIG_SETTING)},
    }


def _knots(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / max(n - 1, 1)


def _curve(raw: np.ndarray, exp: float, normalise: bool = False) -> tuple:
    k = _knots(raw.size)
    g = 1.0 + raw * 2.0 ** exp
    if normalise:
        g = g / float(g.max())
    return k, g


def _fit_odd(k: np.ndarray, target: np.ndarray) -> tuple:
    """Fit m + c0 r^2 + c1 r^4 + c2 r^6 to a radial curve (m fixed at target(0) = 0 here)."""
    a = np.stack([np.ones_like(k), k ** 2, k ** 4, k ** 6], axis=1)
    coef, *_ = np.linalg.lstsq(a, target, rcond=None)
    return tuple(float(c) for c in coef)


def builtin_lens_params(path: str) -> Optional[dict]:
    """Turn an ARW's embedded tables into the renderer's effective-lens keys.

    Returns None when the file has no usable tables (non-Sony, or a body that writes none).
    Unused models come back as zeros, which the renderer treats as "do nothing".
    """
    tab = read_tables(path)
    if not tab:
        return None

    out = {
        "k1": 0.0, "k2": 0.0, "k3": 0.0,
        "ca_r_scale": 0.0, "ca_b_scale": 0.0,
        "ca_r_coeffs": (0.0, 0.0, 0.0), "ca_b_coeffs": (0.0, 0.0, 0.0),
        "vignette_amount": 0.0, "vignette_midpoint": 50.0,
        "dist_scale": 1.0,
        "available": {"distortion": False, "ca": False, "vignette": False},
        "flags": tab["flags"],
    }

    d = tab["distortion"]
    if d.size >= 2:
        k, g = _curve(d, DIST_EXP, normalise=True)
        _m, c0, c1, c2 = _fit_odd(k, g - 1.0)
        out["k1"], out["k2"], out["k3"] = c0, c1, c2
        out["available"]["distortion"] = True

    for key, raw in (("r", tab["ca_r"]), ("b", tab["ca_b"])):
        if raw.size < 2:
            continue
        k, g = _curve(raw, CA_EXP)
        m, c0, c1, c2 = _fit_odd(k, g - 1.0)
        # m is the constant magnification; the polynomial terms are the field-dependent rest.
        out[f"ca_{key}_scale"] = float(m)
        out[f"ca_{key}_coeffs"] = (float(c0), float(c1), float(c2))
        out["available"]["ca"] = True

    v = tab["vignette"]
    if v.size >= 2:
        # Not a linear spline like the other two: the stored value is an exponent of an
        # exponent, gain = 1 / 2^(0.5 - 2^(v * 2^-13 - 1)), and the result is the gain the
        # camera would APPLY to undo the falloff (1.30 at the corner on this frame).
        k = _knots(v.size)
        gain = 1.0 / np.power(2.0, 0.5 - np.power(2.0, v * 2.0 ** VIG_EXP - 1.0))
        target = gain - 1.0
        # The renderer's model is amount * r^v_exp with v_exp derived from the midpoint, which
        # is clamped to [0.1, 0.9] -> v_exp in [1.48, 3.72]. Pick the exponent that fits the
        # Sony curve best, then the amount by least squares at that exponent.
        best = None
        for exp in np.linspace(1.48, 3.72, 57):
            base = k ** exp
            denom = float(np.sum(base * base))
            if denom <= 1e-12:
                continue
            amt = float(np.sum(base * target) / denom)
            err = float(np.sum((amt * base - target) ** 2))
            if best is None or err < best[0]:
                best = (err, amt, float(exp))
        if best is not None:
            out["vignette_amount"] = best[1]
            # Invert v_exp = 1.2 + 2.8 * (1 - midpoint/100).
            mid = 100.0 * (1.0 - (best[2] - 1.2) / 2.8)
            out["vignette_midpoint"] = float(np.clip(mid, 10.0, 90.0))
            out["available"]["vignette"] = True

    return out


def describe(path: str) -> str:
    p = builtin_lens_params(path)
    if not p:
        return "内置校正: 无"
    avail = "/".join(n for n, ok in p["available"].items() if ok) or "无"
    return (f"内置校正: 畸变 k=({p['k1']:+.4f},{p['k2']:+.4f},{p['k3']:+.4f}) "
            f"暗角 {p['vignette_amount']:+.3f} CA=({p['ca_r_scale']:+.5f},{p['ca_b_scale']:+.5f}) "
            f"[{avail}]")
