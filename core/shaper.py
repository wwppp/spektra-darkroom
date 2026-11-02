"""The log shaper that defines the 3D LUT lattice.

Both the LUT bake and the grain density cube are sampled on this lattice, and the shader
re-implements it as GLSL ``linear_to_shaper``.  It used to live in ``core/app_core.py``;
it moved here so the grain modules can use it without importing the engine (which
imports them, so that would be a cycle).

``core.app_core`` re-exports these names, so ``from core.app_core import shaper_forward``
keeps working.
"""
from __future__ import annotations

import numpy as np

# Log Shaper constants: maps scene-linear HDR [0.0, 8.0] into [0.0, 1.0] with dense shadow resolution
SHAPER_C = 60.0
SHAPER_MAX = 8.0
SHAPER_S = float(np.log(1.0 + SHAPER_C * SHAPER_MAX))  # ~6.175867147775916
SHAPER_INV_S = float(1.0 / SHAPER_S)                    # ~0.16192058097945077


def shaper_forward(lin):
    """Map scene-linear HDR [0.0, 8.0+] into [0.0, 1.0] with logarithmic shadow allocation."""
    c = np.maximum(0.0, lin)
    return (np.log(1.0 + SHAPER_C * c) * SHAPER_INV_S).astype(np.float32)


def shaper_inverse(t):
    """Map shaper lattice coordinates [0.0, 1.0] into scene-linear HDR [0.0, 8.0]."""
    return ((np.exp(t * SHAPER_S) - 1.0) / SHAPER_C).astype(np.float32)
