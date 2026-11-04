"""The GPU grain stage: two cubes and, between them, the film's particles.

Why two cubes instead of one: the grain law is a function of *negative density*, and a single
RGB->RGB table never exposes it.  Splitting the chain at the density lets the shader do

    linear RGB --[A]--> (C, M, Y) density --+grain--> --[B]--> output RGB

which is the order the reference runs too.  Both cubes are baked by the engine from the same
pipeline, keyed together, so they cannot describe two different prints.

The GLSL below is spliced into the canvas fragment shader at ``// {{GRAIN}}``; it relies on
``linear_to_shaper`` being defined above it and declares its own uniforms.
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL

GRAIN_GLSL = r"""
// ---------------------------------------------------------------------------------------------
// Negative density and the print, as two tables with room for grain in between.
// ---------------------------------------------------------------------------------------------
uniform int       u_grain_cubes;      // 1 = the two-cube path is uploaded and current
uniform sampler3D u_density_lut;      // shaper(linear RGB) -> (C, M, Y) net density
uniform sampler3D u_print_lut;        // normalised (C, M, Y) -> output RGB
uniform vec3      u_density_min;      // the density range the print cube covers
uniform vec3      u_density_max;

vec3 density_from_linear(vec3 linear_rgb) {
    float dim = float(textureSize(u_density_lut, 0).x);
    vec3 coord = linear_to_shaper(linear_rgb) * ((dim - 1.0) / dim) + (0.5 / dim);
    return texture(u_density_lut, coord).rgb;              // C, M, Y
}

vec3 develop_from_density(vec3 density) {
    float dim = float(textureSize(u_print_lut, 0).x);
    vec3 u = (density - u_density_min) / max(u_density_max - u_density_min, vec3(1e-6));
    vec3 coord = clamp(u, 0.0, 1.0) * ((dim - 1.0) / dim) + (0.5 / dim);
    return texture(u_print_lut, coord).rgb;
}
"""


class GLCubeGrain:
    """Owns the two 3-D tables the stage reads. One instance per canvas context."""

    def __init__(self):
        self._density_tex = 0
        self._density_size = 0
        self._print_tex = 0
        self._print_size = 0
        self.density_min = np.zeros(3, dtype=np.float32)
        self.density_max = np.ones(3, dtype=np.float32)
        self.ready = False
        self._signature = None

    # -- upload ------------------------------------------------------------------------------
    def set_bundle(self, bundle):
        """Stage a bundle for the next paint; ``None`` turns the path off."""
        if bundle is None:
            self._pending = None
            self.ready = False
            return
        signature = (id(bundle.get("density")), id(bundle.get("cube")))
        if signature == self._signature:
            return
        self._signature = signature
        self._pending = bundle

    def upload_pending(self, unit_a=5, unit_b=6):
        """Push a staged bundle to the GPU. Call with the canvas context current."""
        pending = getattr(self, "_pending", None)
        if pending is None:
            return
        self._density_tex, self._density_size = self._upload_cube(
            self._density_tex, self._density_size, pending["density"], unit_a)
        self._print_tex, self._print_size = self._upload_cube(
            self._print_tex, self._print_size, pending["cube"], unit_b)
        self.density_min = np.asarray(pending["density_min"], dtype=np.float32)
        self.density_max = np.asarray(pending["density_max"], dtype=np.float32)
        self.ready = True
        self._pending = None

    @staticmethod
    def _upload_cube(tex, size, array, unit):
        data = np.ascontiguousarray(array, dtype=np.float32)
        n = data.shape[0]
        GL.glActiveTexture(0x84C0 + unit)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if tex and size == n:
            GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
            GL.glTexSubImage3D(GL.GL_TEXTURE_3D, 0, 0, 0, 0, n, n, n,
                               GL.GL_RGB, GL.GL_FLOAT, data)
        else:
            if tex:
                GL.glDeleteTextures([tex])
            tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGB16F, n, n, n, 0,
                            GL.GL_RGB, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        return tex, n

    # -- bind --------------------------------------------------------------------------------
    def bind(self, u_1i, unit_a=5, unit_b=6):
        if not self.ready:
            u_1i("u_grain_cubes", 0)
            return False
        for unit, tex, name in ((unit_a, self._density_tex, "u_density_lut"),
                                (unit_b, self._print_tex, "u_print_lut")):
            GL.glActiveTexture(0x84C0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
            u_1i(name, unit)
        u_1i("u_grain_cubes", 1)
        return True

    def release(self):
        for attr in ("_density_tex", "_print_tex"):
            tex = getattr(self, attr, 0)
            if tex:
                try:
                    GL.glDeleteTextures([tex])
                except Exception:
                    pass
                setattr(self, attr, 0)
        self.ready = False
        self._signature = None
