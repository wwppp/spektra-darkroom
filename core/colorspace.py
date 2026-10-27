"""Colour-space conversion for export: one resolver, one implementation.

Why this module exists
----------------------
`convert_image_colorspace` and `get_icc_profile_bytes` used to sit side by side in
``core/app_core.py`` with two *independent* sets of string-matching rules.  Whenever the
two disagreed, the writer emitted pixels in one space carrying an ICC profile for
another -- a silent colour cast on the customer's file that is very hard to trace back
to the app.  This module collapses both decisions into a single ``resolve()`` so the
pixels and the embedded profile can no longer diverge.

Accuracy target: the previous implementation (colour-science, float64) is the reference.
Everything here must reproduce it within 1 LSB at 8-bit; per-pixel differences are
asserted by ``tools/export_colorspace_check.py``.

Performance note: colour-science costs 30-38 s for a 64 MP non-sRGB export (it is a
generic, per-channel Python/scipy pipeline).  The numba kernel below does the same work
in well under a second, which is what makes the feature usable at full resolution.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Canonical colour spaces
# ---------------------------------------------------------------------------
# ``key`` is the canonical name used everywhere (pixels, ICC lookup, EXIF, JXL CICP).
# ``aliases`` lists the substrings that used to be matched ad hoc; order matters,
# first hit wins, and every alias previously accepted must keep resolving the same way.
_SPACES = (
    # key,                  match substrings (lower-case),                 icc file
    ("Display P3",          ("p3",),                                       "DisplayP3.icc"),
    ("Adobe RGB (1998)",    ("adobe",),                                    "AdobeRGB1998.icc"),
    ("Rec. 2020",           ("2020", "rec2020", "bt2020"),                 "Rec2020.icc"),
    ("ProPhoto RGB",        ("prophoto",),                                 "ProPhotoRGB.icc"),
)
_DEFAULT = ("sRGB", "sRGB.icc")

CANONICAL_NAMES = (("sRGB",) + tuple(s[0] for s in _SPACES))


def resolve(target_colorspace):
    """Resolve any user/UI string to ``(canonical_name, icc_filename)``.

    Unknown or empty names resolve to sRGB, which is also what the old ICC lookup did,
    so unlabelled pixels can never be paired with a wide-gamut profile.
    """
    text = "" if target_colorspace is None else str(target_colorspace).strip().lower()
    if text:
        for key, needles, icc in _SPACES:
            if any(n in text for n in needles):
                return key, icc
    return _DEFAULT


def icc_filename(target_colorspace):
    return resolve(target_colorspace)[1]


# ---------------------------------------------------------------------------
# Linear matrices: linear sRGB (D65) -> linear target
# ---------------------------------------------------------------------------
# Extracted from colour-science 0.4.7 via matrix_RGB_to_RGB("sRGB", <space>), which
# applies the CAT02 D65->D50 chromatic adaptation where the target white point requires
# it (ProPhoto).  Each row sums to 1.0, i.e. sRGB white maps to neutral white.
MATRICES = {
    "Display P3": np.array([
        [0.8225336512, 0.1775458067, 0.0000536948],
        [0.0331170846, 0.9668481430, -0.0000031500],
        [0.0170564417, 0.0724006114, 0.9104892659],
    ], dtype=np.float32),
    "Adobe RGB (1998)": np.array([
        [0.7151773010, 0.2848856160, 0.0000474080],
        [-0.0000812460, 1.0000474720, -0.0000000060],
        [-0.0000278990, 0.0411633360, 0.9587994130],
    ], dtype=np.float32),
    "Rec. 2020": np.array([
        [0.6274413721, 0.3292974595, 0.0433514584],
        [0.0690276171, 0.9195806669, 0.0113614226],
        [0.0163642351, 0.0880171625, 0.8955649727],
    ], dtype=np.float32),
    "ProPhoto RGB": np.array([
        [0.5288241004, 0.3340609866, 0.1373616909],
        [0.0975294148, 0.8790074094, 0.0233981175],
        [0.0163599018, 0.1066124933, 0.8772485185],
    ], dtype=np.float32),
}

# Transfer-function branch identifiers shared with the numba kernel below.
ENC_SRGB = 0        # sRGB / Display P3 piecewise
ENC_ADOBE = 1       # Adobe RGB (1998): pure power, gamma 2.19921875 (= 563/256)
ENC_BT2020 = 2      # ITU-R BT.2020 OETF, rounded constants used by colour-science
ENC_ROMM = 3        # ProPhoto RGB: x*16 toe below 1/512, else x^(1/1.8)

ENCODERS = {
    "sRGB": ENC_SRGB,
    "Display P3": ENC_SRGB,
    "Adobe RGB (1998)": ENC_ADOBE,
    "Rec. 2020": ENC_BT2020,
    "ProPhoto RGB": ENC_ROMM,
}

_ADOBE_EXP = 0.4547069271758437        # 256/563
_BT2020_ALPHA = 1.099                  # colour-science rounds these two
_BT2020_BETA = 0.018
_ROMM_TOE = 1.0 / 512.0


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _encode_np(x, kind):
    """Reference vectorised encoder (used when numba is unavailable)."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    if kind == ENC_SRGB:
        return np.where(x <= 0.0031308, x * 12.92,
                        1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)
    if kind == ENC_ADOBE:
        return np.power(x, _ADOBE_EXP).astype(np.float32)
    if kind == ENC_BT2020:
        return np.where(x < _BT2020_BETA, x * 4.5,
                        _BT2020_ALPHA * np.power(x, 0.45) - (_BT2020_ALPHA - 1.0)).astype(np.float32)
    return np.where(x < _ROMM_TOE, x * 16.0, np.power(x, 1.0 / 1.8)).astype(np.float32)


def _convert_numpy(img, matrix, kind):
    lin = img.astype(np.float32, copy=True)
    np.clip(lin, 0.0, 1.0, out=lin)
    low = lin <= 0.04045
    # sRGB decode: linear toe below 0.04045, power above
    lin = np.where(low, lin / 12.92,
                   np.power((lin + 0.055) / 1.055, 2.4)).astype(np.float32)
    h, w = lin.shape[:2]
    flat = lin.reshape(-1, 3)
    out = np.empty_like(flat)
    m = matrix
    out[:, 0] = m[0, 0] * flat[:, 0] + m[0, 1] * flat[:, 1] + m[0, 2] * flat[:, 2]
    out[:, 1] = m[1, 0] * flat[:, 0] + m[1, 1] * flat[:, 1] + m[1, 2] * flat[:, 2]
    out[:, 2] = m[2, 0] * flat[:, 0] + m[2, 1] * flat[:, 1] + m[2, 2] * flat[:, 2]
    return _encode_np(out.reshape(h, w, 3), kind)


_NUMBA_KERNEL = None
_NUMBA_KERNEL_LUT = None
_NUMBA_TRIED = False

_DECODE_LUT = None
_ENCODE_LUTS = {}


def _decode_lut():
    """sRGB decode table.

    Every pixel does three random lookups here.  A 16-bit table (256 KB) keeps the
    quantisation error of the encoded input down to 1/65535 -- which matters for the
    steep transfer functions (Adobe RGB, ProPhoto) whose encode stage amplifies shadow
    error -- and still costs only ~25 ms more at 64 MP than a 4096-node table, because
    the kernel is bandwidth-bound rather than cache-bound.
    """
    global _DECODE_LUT
    if _DECODE_LUT is None:
        x = np.linspace(0.0, 1.0, 65536, dtype=np.float64)
        _DECODE_LUT = np.where(x <= 0.04045, x / 12.92,
                               np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)
    return _DECODE_LUT


def _encode_lut(kind):
    """Transfer-function table (65537 nodes) used with linear interpolation."""
    if kind not in _ENCODE_LUTS:
        x = np.linspace(0.0, 1.0, 65537, dtype=np.float64)
        if kind == ENC_SRGB:
            y = np.where(x <= 0.0031308, x * 12.92,
                         1.055 * np.power(x, 1.0 / 2.4) - 0.055)
        elif kind == ENC_ADOBE:
            y = np.power(x, _ADOBE_EXP)
        elif kind == ENC_BT2020:
            y = np.where(x < _BT2020_BETA, x * 4.5,
                         _BT2020_ALPHA * np.power(x, 0.45) - (_BT2020_ALPHA - 1.0))
        else:
            y = np.where(x < _ROMM_TOE, x * 16.0, np.power(x, 1.0 / 1.8))
        _ENCODE_LUTS[kind] = np.clip(y, 0.0, 1.0).astype(np.float32)
    return _ENCODE_LUTS[kind]


def _numba_kernel():
    """Compile (once) and return ``(exact_kernel, lut_kernel)``, or ``(None, None)``.

    ``exact_kernel`` evaluates the transfer functions with real pow() calls and is used
    for 16- and 32-bit output.  ``lut_kernel`` replaces both transfer functions with
    table lookups; it is used for 8-bit output, where the tables are accurate to well
    under one 8-bit step, and is ~4.5x faster.
    """
    global _NUMBA_KERNEL, _NUMBA_KERNEL_LUT, _NUMBA_TRIED
    if _NUMBA_TRIED:
        return _NUMBA_KERNEL, _NUMBA_KERNEL_LUT
    _NUMBA_TRIED = True
    try:
        from numba import njit, prange

        @njit(cache=True, inline="always")
        def _enc(v, kind):
            if v <= 0.0:
                return np.float32(0.0)
            if v >= 1.0:
                return np.float32(1.0)
            if kind == 0:
                if v <= 0.0031308:
                    return np.float32(v * 12.92)
                return np.float32(1.055 * v ** (1.0 / 2.4) - 0.055)
            if kind == 1:
                return np.float32(v ** 0.4547069271758437)
            if kind == 2:
                if v < 0.018:
                    return np.float32(v * 4.5)
                return np.float32(1.099 * v ** 0.45 - 0.099)
            if v < 0.001953125:
                return np.float32(v * 16.0)
            return np.float32(v ** (1.0 / 1.8))

        @njit(cache=True, inline="always")
        def _clamp01(v):
            if v < 0.0:
                return np.float32(0.0)
            if v > 1.0:
                return np.float32(1.0)
            return v

        @njit(cache=True, parallel=True)
        def _kernel(img, m00, m01, m02, m10, m11, m12, m20, m21, m22, kind):
            h, w = img.shape[0], img.shape[1]
            out = np.empty((h, w, 3), dtype=np.float32)
            for y in prange(h):
                for x in range(w):
                    r = _clamp01(img[y, x, 0])
                    g = _clamp01(img[y, x, 1])
                    b = _clamp01(img[y, x, 2])
                    lr = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
                    lg = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
                    lb = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
                    out[y, x, 0] = _enc(m00 * lr + m01 * lg + m02 * lb, kind)
                    out[y, x, 1] = _enc(m10 * lr + m11 * lg + m12 * lb, kind)
                    out[y, x, 2] = _enc(m20 * lr + m21 * lg + m22 * lb, kind)
            return out

        @njit(cache=True, parallel=True)
        def _kernel_lut(img, dec, enc, m00, m01, m02, m10, m11, m12, m20, m21, m22):
            h, w = img.shape[0], img.shape[1]
            nd = dec.shape[0] - 1
            ne = enc.shape[0] - 1
            out = np.empty((h, w, 3), dtype=np.float32)
            for y in prange(h):
                for x in range(w):
                    ir = int(_clamp01(img[y, x, 0]) * nd + 0.5)
                    ig = int(_clamp01(img[y, x, 1]) * nd + 0.5)
                    ib = int(_clamp01(img[y, x, 2]) * nd + 0.5)
                    lr = dec[ir]
                    lg = dec[ig]
                    lb = dec[ib]
                    tr = m00 * lr + m01 * lg + m02 * lb
                    tg = m10 * lr + m11 * lg + m12 * lb
                    tb = m20 * lr + m21 * lg + m22 * lb
                    # linear interpolation between encode-table nodes
                    if tr <= 0.0:
                        out[y, x, 0] = 0.0
                    elif tr >= 1.0:
                        out[y, x, 0] = 1.0
                    else:
                        f = tr * ne
                        i0 = int(f)
                        t = f - i0
                        out[y, x, 0] = enc[i0] * (1.0 - t) + enc[i0 + 1] * t
                    if tg <= 0.0:
                        out[y, x, 1] = 0.0
                    elif tg >= 1.0:
                        out[y, x, 1] = 1.0
                    else:
                        f = tg * ne
                        i0 = int(f)
                        t = f - i0
                        out[y, x, 1] = enc[i0] * (1.0 - t) + enc[i0 + 1] * t
                    if tb <= 0.0:
                        out[y, x, 2] = 0.0
                    elif tb >= 1.0:
                        out[y, x, 2] = 1.0
                    else:
                        f = tb * ne
                        i0 = int(f)
                        t = f - i0
                        out[y, x, 2] = enc[i0] * (1.0 - t) + enc[i0 + 1] * t
            return out

        _NUMBA_KERNEL = _kernel
        _NUMBA_KERNEL_LUT = _kernel_lut
    except Exception:  # numba missing or unsupported numpy -- numpy path still works
        _NUMBA_KERNEL = None
        _NUMBA_KERNEL_LUT = None
    return _NUMBA_KERNEL, _NUMBA_KERNEL_LUT


def warm_up():
    """Trigger JIT compilation ahead of the first export. Safe to call from a thread."""
    exact, lut = _numba_kernel()
    if exact is None:
        return False
    try:
        eye = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        exact(np.zeros((4, 4, 3), dtype=np.float32), *eye, ENC_SRGB)
        if lut is not None:
            lut(np.zeros((4, 4, 3), dtype=np.float32), _decode_lut(), _encode_lut(ENC_SRGB), *eye)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def convert(img_rgb_float, target_colorspace="sRGB", raise_on_error=False, bit_depth=8):
    """Convert sRGB non-linear image data to ``target_colorspace``.

    The input is *display sRGB* (non-linear, 0..1), which is what both renderers emit.

    ``bit_depth`` selects the accuracy tier, not the output dtype (the result is always
    float32): 8-bit output tolerates the table-driven fast path, 16-bit and float output
    need the exact power-law evaluation.

    Returns ``(out, effective_colorspace)``.  ``effective_colorspace`` is the space the
    returned pixels are actually in -- callers MUST embed the ICC profile for that name,
    never for the requested one, so a failed conversion degrades to "sRGB pixels +
    sRGB ICC" instead of "sRGB pixels + wide-gamut ICC".
    """
    canonical, _icc = resolve(target_colorspace)
    img = np.asarray(img_rgb_float)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB array, got %r" % (img.shape,))

    if canonical == "sRGB":
        return np.clip(img, 0.0, 1.0), canonical

    matrix = MATRICES[canonical]
    kind = ENCODERS[canonical]
    exact, lut = _numba_kernel()
    try:
        if exact is not None:
            m = matrix
            coefs = (float(m[0, 0]), float(m[0, 1]), float(m[0, 2]),
                     float(m[1, 0]), float(m[1, 1]), float(m[1, 2]),
                     float(m[2, 0]), float(m[2, 1]), float(m[2, 2]))
            if int(bit_depth) <= 8 and lut is not None:
                out = lut(img, _decode_lut(), _encode_lut(kind), *coefs)
            else:
                out = exact(img, *coefs, int(kind))
        else:
            out = _convert_numpy(img, matrix, kind)
        # Every kernel above clamps per channel, so no extra 3-byte-per-pixel clip pass
        # here: on a 64 MP export that redundant copy alone cost ~0.3 s.
        return out, canonical
    except Exception as exc:
        if raise_on_error:
            raise
        print(f"[SpektraEngine] Colorspace conversion failed ({canonical}): {exc}; "
              f"falling back to sRGB pixels")
        return np.clip(img, 0.0, 1.0), "sRGB"
