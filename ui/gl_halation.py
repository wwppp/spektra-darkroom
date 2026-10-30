"""Host-agnostic GPU halation + in-emulsion scatter: decimated separable Gaussians.

Wired into both render paths as of 2026-10-30: ``CanvasViewport.render_offscreen``
(export / thumbnails / histogram) and ``CanvasViewport.paintGL`` (live preview). Measured
CPU/GPU agreement on a 24 MP frame went from 45.20 dB to 60.14 dB (halation alone: 64.01,
scatter alone: 60.99). See docs/halation-rework-plan.md P2 for the numbers and the two
numerical defects this replaced.

Why it exists: the shader used to sample the halo with 16 points on a Vogel spiral, which
cannot converge to a Gaussian with sigma = 28 px -- it was already reading 1/30-resolution
mip levels, so a "tap" is a ~1 px box and eight boxes ringed at 2.3 px overlap into a shape
that is neither the kernel nor a disc. Measured: replacing the spiral with three radial
bands made agreement *worse* (45.20 -> 39.89 dB). Point sampling is a dead end at this
scale; the kernel has to be applied as a kernel.

Pipeline (all RGBA16F, all inside the caller's frame):

    sharp radiance (full res)
      -> scatter H/V         (separable, per-channel mixture of the profile's core+tail
                              Gaussians, then mix(E, blur, 0.06))
      -> decimate 1/2 x n    (exact 2x2 box average at every step)
      -> halation H/V x3     (one pass pair per bounce, sigma lands in 2-4 px)
      -> composite           (bilinear upsample, decay weights, (E + a*blur)/(1 + a))

The decimation is what makes it accurate, not a compromise: on the CPU the same trick
scores 120 dB against the full-resolution blur. At 1/16 the widest bounce has sigma ~3 px,
which is exactly the regime where a separable filter is both correct and cheap.

Scope: this class owns GL objects and nothing else. It does not read widget state, does not
know about zoom, and takes its physics as arguments, so the preview and the offscreen
export call it with the same numbers. Everything it touches in the GL state machine is
saved on entry and restored on every exit path.
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram

_VERT = """
#version 330 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

# Scene-radiance shaping shared by the decimate pass and the composite pass. Both need
# it: the blur must be computed on the *shaped* radiance, and the composite has to
# reintroduce the same E on the full-resolution side, or the DC normalisation would mix
# a shaped numerator with an unshaped denominator.
# Constants mirror core/darkroom_math.py and the main fragment shader.
_HS_GLSL = """
uniform float u_highlights;
uniform float u_shadows;
const float HS_HL_LO = 0.0;
const float HS_HL_SPAN = 2.4709;
const float HS_SH_SPAN = 6.0;
const float HS_HL_STOPS = 0.01;
const float HS_SH_STOPS = 0.015;
vec3 shape_radiance(vec3 col) {
    if (abs(u_highlights) < 0.001 && abs(u_shadows) < 0.001) return col;
    float hi = max(col.r, max(col.g, col.b));
    float lum = dot(col, vec3(0.2880, 0.7119, 0.0001));
    float t_hl = clamp((log2(max(hi, 1e-9) / 0.18) - HS_HL_LO) / HS_HL_SPAN, 0.0, 1.0);
    float w_hl = t_hl * t_hl * (3.0 - 2.0 * t_hl);
    float t_sh = clamp(-log2(max(lum, 1e-9) / 0.18) / HS_SH_SPAN, 0.0, 1.0);
    float w_sh = t_sh * (1.0 - t_sh) * 4.0;
    return col * exp2((u_shadows * HS_SH_STOPS) * w_sh + (u_highlights * HS_HL_STOPS) * w_hl);
}
"""

# Halve the resolution with a true 2x2 box average (one bilinear sample at the centre of
# each 2x2 texel block), optionally converting to scene radiance on the first step.
# A single bilinear sample *is* an exact box average for a factor-2 reduction, so chaining
# this pass log2(k) times equals cv2.INTER_AREA by k. Rendering straight from full res into
# a 1/k target does NOT: one bilinear tap can only average 4 texels, so a factor-4 reduction
# aliases and a point source lands as a single delta. That alone accounted for a 15-20%
# error in the inner halo.
_DOWN_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
uniform vec2 u_texel;     // 1/src_width, 1/src_height
uniform vec3 u_gain;
uniform int u_shape;      // 1 on the first step only
""" + _HS_GLSL + """
void main() {
    vec2 o = u_texel * 0.5;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec2 dir = vec2(float((i & 1) * 2 - 1), float((i >> 1) * 2 - 1)) * 0.5;
        vec3 c = texture(u_src, v_uv + dir * u_texel).rgb;
        if (u_shape == 1) c = shape_radiance(c * u_gain);
        acc += c;
    }
    fragColor = vec4(acc * 0.25, 1.0);
}
"""

# One direction of a separable Gaussian. The radius follows 3*sigma rather than a fixed
# tap count: at the working resolution sigma can reach ~6 px, and a ±4 texel window would
# cut the tail off a sigma-5 kernel (~40% of its energy), which reads as a halo that is
# too narrow and too bright near the source.
_BLUR_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
uniform vec2 u_step;     // one texel along the blur axis, in uv units
uniform float u_sigma;   // in texels of the working grid
const int MAX_TAP = 24;
void main() {
    float s = max(0.4, u_sigma);
    float r = min(3.0 * s, float(MAX_TAP));
    vec3 acc = texture(u_src, v_uv).rgb;
    float wsum = 1.0;
    for (int i = 1; i <= MAX_TAP; i++) {
        float x = float(i);
        if (x > r) break;
        float w = exp(-0.5 * x * x / (s * s));
        acc += (texture(u_src, v_uv + u_step * x).rgb
              + texture(u_src, v_uv - u_step * x).rgb) * w;
        wsum += 2.0 * w;
    }
    fragColor = vec4(acc / wsum, 1.0);
}
"""

# Upsample the three bounce layers, sum them with the profile's decay, and apply
# the same DC normalisation the CPU uses: (E + a*blur) / (1 + a).
# In-emulsion forward scatter: one axis of a *mixture* of two Gaussians (the profile's
# core and tail sigmas), per channel. The CPU does core*(1-w) + tail*w on each channel;
# folding both kernels into the tap weights keeps that exact while paying for one pass.
# Doing it as a single Gaussian with matched variance would also be close, but the
# mixture is free here and leaves nothing to argue about.
_SCAT_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
uniform vec2 u_step;
uniform vec3 u_sig_core;   // texels
uniform vec3 u_sig_tail;
uniform vec3 u_w_tail;
uniform vec3 u_gain;
uniform int u_shape;
const int MAX_TAP = 12;
""" + _HS_GLSL + """
void main() {
    vec3 sc = max(u_sig_core, vec3(0.1));
    vec3 st = max(u_sig_tail, vec3(0.1));
    float r = min(3.0 * max(max(sc.r, sc.g), max(sc.b, max(st.r, max(st.g, st.b)))), float(MAX_TAP));
    vec3 acc = vec3(0.0);
    vec3 wsum = vec3(0.0);
    for (int i = 0; i <= MAX_TAP; i++) {
        float x = float(i);
        if (x > r) break;
        vec3 w = (vec3(1.0) - u_w_tail) * exp(-0.5 * x * x / (sc * sc))
               + u_w_tail * exp(-0.5 * x * x / (st * st));
        vec3 c1 = texture(u_src, v_uv + u_step * x).rgb;
        vec3 c2 = texture(u_src, v_uv - u_step * x).rgb;
        if (u_shape == 1) { c1 = shape_radiance(c1 * u_gain); c2 = shape_radiance(c2 * u_gain); }
        acc += (c1 + c2) * w;
        wsum += 2.0 * w;
    }
    vec3 c0 = texture(u_src, v_uv).rgb;
    if (u_shape == 1) c0 = shape_radiance(c0 * u_gain);
    fragColor = vec4((acc + c0) / (wsum + vec3(1.0)), 1.0);
}
"""

# Mix the scattered image back over the unblurred one: the CPU's energy-preserving
# `(1-eff)*x + eff*scattered` with eff = clamp(amount*0.06, 0, 0.5).
_MIX_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_sharp;    // untouched negative
uniform sampler2D u_blur;     // scattered, already in shaped radiance
uniform float u_eff;
uniform vec3 u_gain;
""" + _HS_GLSL + """
void main() {
    // The sharp side has to be shaped here or the mix would blend two different radiometric
    // domains: the blurred side went through shape() in the first scatter pass.
    vec3 sharp = shape_radiance(texture(u_sharp, v_uv).rgb * u_gain);
    fragColor = vec4(mix(sharp, texture(u_blur, v_uv).rgb, u_eff), 1.0);
}
"""

_COMP_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_raw;      // full-res E: the negative, or the scattered radiance
uniform sampler2D u_b0;       // bounce 1 (working grid)
uniform sampler2D u_b1;       // bounce 2
uniform sampler2D u_b2;       // bounce 3
uniform vec3 u_gain;
uniform int u_e_shaped;       // 1 when u_raw already holds shaped radiance
uniform vec3 u_decay;         // normalised bounce weights
uniform vec3 u_a_tot;         // strength * physical multiplier, per channel
uniform float u_amount;
""" + _HS_GLSL + """
void main() {
    vec3 e = texture(u_raw, v_uv).rgb;
    if (u_e_shaped == 0) e = shape_radiance(e * u_gain);
    vec3 blur = texture(u_b0, v_uv).rgb * u_decay.x
              + texture(u_b1, v_uv).rgb * u_decay.y
              + texture(u_b2, v_uv).rgb * u_decay.z;
    vec3 a = u_a_tot * u_amount;
    fragColor = vec4((e + a * blur) / (vec3(1.0) + a), 1.0);
}
"""


class GLHalationChain:
    """Decimated separable-Gaussian halation, rendered into an RGBA16F texture."""

    #: Decimation is chosen so the widest bounce stays in this band. Below 2 px a
    #: separable kernel is no better than a tap; above 4 px the working grid is
    #: doing more work than it needs to.
    TARGET_SIGMA_MIN = 2.0
    TARGET_SIGMA_MAX = 4.0
    MAX_DECIMATION = 16

    def __init__(self):
        self._down = None
        self._blur = None
        self._comp = None
        self._scat = None
        self._mix = None
        self._vbo = None
        self._vao = None
        self._targets = {}
        self._size = (0, 0)
        self._ready = False
        self._failed = False
        self.last_error = ""

    # ------------------------------------------------------------------ setup
    def _compile(self, frag_src: str, label: str):
        prog = QOpenGLShaderProgram()
        if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, _VERT):
            self.last_error = f"{label} vertex: {prog.log()}"
            return None
        if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, frag_src):
            self.last_error = f"{label} fragment: {prog.log()}"
            return None
        if not prog.link():
            self.last_error = f"{label} link: {prog.log()}"
            return None
        return prog

    def ensure(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        self._down = self._compile(_DOWN_FRAG, "halation.down")
        self._blur = self._compile(_BLUR_FRAG, "halation.blur")
        self._comp = self._compile(_COMP_FRAG, "halation.comp")
        self._scat = self._compile(_SCAT_FRAG, "halation.scatter")
        self._mix = self._compile(_MIX_FRAG, "halation.mix")
        if not (self._down and self._blur and self._comp and self._scat and self._mix):
            print(f"[GLHalation] shader setup failed: {self.last_error}")
            self._failed = True
            return False

        verts = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float32)
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self._ready = True
        return True

    def _target(self, key: str, w: int, h: int):
        """Lazily (re)allocate an RGBA16F colour target at w x h."""
        entry = self._targets.get(key)
        if entry is not None and entry[2] == w and entry[3] == h:
            return entry[0], entry[1]
        if entry is not None:
            GL.glDeleteFramebuffers(1, [entry[1]])
            GL.glDeleteTextures([entry[0]])

        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, tex, 0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            print(f"[GLHalation] incomplete FBO for {key} ({w}x{h}): 0x{status:x}")
            GL.glDeleteFramebuffers(1, [fbo])
            GL.glDeleteTextures([tex])
            return None, None

        self._targets[key] = (tex, fbo, w, h)
        return tex, fbo

    # ------------------------------------------------------------------ draw
    # PySide6's setUniformValue has no (name, float) overload and wants bytes names;
    # going through PyOpenGL keeps the call sites readable and type-correct.
    @staticmethod
    def _u1i(prog, name, v):
        loc = prog.uniformLocation(name)
        if loc != -1:
            GL.glUniform1i(loc, int(v))

    @staticmethod
    def _u1f(prog, name, v):
        loc = prog.uniformLocation(name)
        if loc != -1:
            GL.glUniform1f(loc, float(v))

    @staticmethod
    def _u2f(prog, name, x, y):
        loc = prog.uniformLocation(name)
        if loc != -1:
            GL.glUniform2f(loc, float(x), float(y))

    @staticmethod
    def _u3f(prog, name, x, y, z):
        loc = prog.uniformLocation(name)
        if loc != -1:
            GL.glUniform3f(loc, float(x), float(y), float(z))

    def _draw(self, prog, fbo, w: int, h: int):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, w, h)
        prog.bind()
        # Own VAO: enabling/disabling attribute 0 on the *caller's* VAO would leave the main
        # pass with a different vertex layout than it set up, which is one of the ways this
        # chain used to break the frame it was embedded in.
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        prog.release()

    @staticmethod
    def decimation_for(max_sigma_px: float) -> int:
        """Power-of-two factor that puts the widest bounce in TARGET_SIGMA band.

        The band is what keeps the blur cheap: with sigma on the working grid near 3 px the
        kernel needs ~9 taps per axis, and the pass runs at 1/k^2 of the pixels. It is
        deliberately not the same k the CPU uses (core.halation_model._pick_decimation aims
        at 4 px), because the two only have to agree on the *continuous* kernel -- the
        decimation is exact area averaging on both sides, and the residual difference is
        smaller than the 8-bit quantisation of the result.
        """
        if max_sigma_px <= GLHalationChain.TARGET_SIGMA_MAX:
            return 1
        k = int(2 ** round(np.log2(max_sigma_px / 3.5)))
        return max(1, min(GLHalationChain.MAX_DECIMATION, k))

    # ------------------------------------------------------------------ run
    def run(self, raw_tex: int, width: int, height: int, scene_gain,
            sigma_px, decay, a_tot, amount: float, hs=(0.0, 0.0), scatter=None):
        """Render halation over ``raw_tex`` and return the result texture id.

        raw_tex   -- full-resolution scene-linear negative (RGBA)
        scene_gain-- vec3 folded exposure + white balance, applied per sample
        sigma_px  -- per-channel bounce-1 halo sigma in full-resolution pixels
        decay     -- normalised bounce weights (3 values)
        a_tot     -- per-channel strength, before the user's amount multiplier
        amount    -- user's physical multiplier
        scatter   -- optional dict {"core": (r,g,b) px, "tail": (..), "w_tail": (..),
                     "amount": float}; when present the in-emulsion scatter is evaluated
                     here too, in the same order the CPU uses (scatter, then halation)

        Returns None if the chain could not run (caller should fall back).
        """
        if not self.ensure():
            return None
        self.last_did_scatter = False
        want_hal = amount > 0.001 and max(sigma_px) > 0.01
        want_scat = bool(scatter) and float(scatter.get("amount", 0.0)) > 0.001
        if not want_hal and not want_scat:
            return None
        self.last_did_halation = bool(want_hal)

        # Everything this chain touches is captured up front and put back afterwards. It is
        # embedded in the middle of somebody else's frame: leaving the FBO bound, a foreign
        # VAO current, blend enabled or a stale texture on unit 0 corrupts the main pass in
        # ways that only show up later (that is what bit the first wiring attempt).
        prev_fbo = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
        prev_vp = [int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT)]
        prev_prog = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        prev_vao = int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING))
        prev_active = int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE))
        prev_units = {}
        for unit in range(4):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            prev_units[unit] = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        prev_blend = bool(GL.glIsEnabled(GL.GL_BLEND))
        prev_depth = bool(GL.glIsEnabled(GL.GL_DEPTH_TEST))
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)

        def _restore():
            for unit in range(4):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, prev_units[unit])
            GL.glActiveTexture(prev_active)
            GL.glBindVertexArray(prev_vao)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fbo)
            if len(prev_vp) == 4:
                GL.glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3])
            if prev_prog:
                GL.glUseProgram(prev_prog)
            if prev_blend:
                GL.glEnable(GL.GL_BLEND)
            if prev_depth:
                GL.glEnable(GL.GL_DEPTH_TEST)

        def _chk(tag):
            e = GL.glGetError()
            if e:
                print(f"[GLHalation] GL error at {tag}: {e}")

        sw = max(1, width)
        sh = max(1, height)
        max_sigma = float(max(sigma_px)) * float(np.sqrt(max(1, len(decay))))
        k = self.decimation_for(max_sigma)
        if k <= 1:
            return None  # sigma already tiny; the tap path is equivalent and cheaper

        _chk("start")
        t_h, f_h = None, None
        g3 = [float(v) for v in scene_gain[:3]]

        # --- scatter: sharp -> H -> V -> mix, all at full resolution ---
        # Scatter sigmas are 0.5-2 px, so decimating them would be butchery; the cost is two
        # full-res passes with <=12 taps, which is still cheaper than the halation chain.
        e_tex, e_shaped = int(raw_tex), 0
        if want_scat:
            sc_core = [float(v) for v in scatter.get("core", (0.5, 0.5, 0.5))]
            sc_tail = [float(v) for v in scatter.get("tail", (1.0, 1.0, 1.0))]
            sc_w = [float(v) for v in scatter.get("w_tail", (0.0, 0.0, 0.0))]
            t_sh, f_sh = self._target("sh", sw, sh)
            t_s, f_s = self._target("s", sw, sh)
            t_m, f_m = self._target("sm", sw, sh)
            if t_sh is None or t_s is None or t_m is None:
                _restore()
                return None
            for axis, (src, dst, f_dst, do_shape) in enumerate((
                    (int(raw_tex), t_sh, f_sh, 1),
                    (t_sh, t_s, f_s, 0))):
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, src)
                self._scat.bind()
                self._u1i(self._scat, "u_src", 0)
                self._u2f(self._scat, "u_step", 1.0 / sw if axis == 0 else 0.0,
                          0.0 if axis == 0 else 1.0 / sh)
                self._u3f(self._scat, "u_sig_core", *sc_core)
                self._u3f(self._scat, "u_sig_tail", *sc_tail)
                self._u3f(self._scat, "u_w_tail", *sc_w)
                self._u3f(self._scat, "u_gain", *g3)
                self._u1f(self._scat, "u_highlights", hs[0])
                self._u1f(self._scat, "u_shadows", hs[1])
                self._u1i(self._scat, "u_shape", do_shape)
                self._draw(self._scat, f_dst, sw, sh)
                self._scat.release()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, raw_tex)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, t_s)
            self._mix.bind()
            self._u1i(self._mix, "u_sharp", 0)
            self._u1i(self._mix, "u_blur", 1)
            self._u1f(self._mix, "u_eff", float(scatter.get("eff", 0.06)))
            self._u3f(self._mix, "u_gain", *g3)
            self._u1f(self._mix, "u_highlights", hs[0])
            self._u1f(self._mix, "u_shadows", hs[1])
            self._draw(self._mix, f_m, sw, sh)
            self._mix.release()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            e_tex, e_shaped = t_m, 1
            self.last_did_scatter = True
            _chk("scatter")


        # --- pass 1..n: reduce by exactly 2 each time (true box average) ---
        # The first step folds exposure/WB and the highlight-shadow shaping into the
        # reduction, so the taps are shaped *before* they are averaged. That is the order
        # the CPU uses (it shapes the full-res image, then decimates). Later steps must not
        # shape again: shape() is non-linear, so re-applying it after averaging would
        # brighten the low-contrast halo more than the CPU does.
        src_tex = e_tex
        cur_w, cur_h = sw, sh
        first = (e_shaped == 0)
        level = 0
        while level < 4 and (k >> level) > 1 and cur_w // 2 >= 2 and cur_h // 2 >= 2:
            nw, nh = max(2, cur_w // 2), max(2, cur_h // 2)
            t_dst, f_dst = self._target(f"d{level}", nw, nh)
            if t_dst is None:
                _restore()
                return None
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, src_tex)
            self._down.bind()
            self._u1i(self._down, "u_src", 0)
            self._u2f(self._down, "u_texel", 1.0 / cur_w, 1.0 / cur_h)
            self._u3f(self._down, "u_gain", *g3)
            self._u1f(self._down, "u_highlights", hs[0])
            self._u1f(self._down, "u_shadows", hs[1])
            self._u1i(self._down, "u_shape", 1 if first else 0)
            self._draw(self._down, f_dst, nw, nh)
            self._down.release()
            src_tex, cur_w, cur_h = t_dst, nw, nh
            first = False
            level += 1
        _chk("decimate")
        t_l0, ww, wh = src_tex, cur_w, cur_h

        t_h, f_h = self._target("h", ww, wh)
        if t_h is None:
            _restore()
            return None
        b_tex, b_fbo = [], []
        for i in range(3):
            tx, fb = self._target(f"b{i}", ww, wh)
            if tx is None:
                _restore()
                return None
            b_tex.append(tx)
            b_fbo.append(fb)
        _chk("targets b")
        t_out, f_out = self._target("out", sw, sh)
        _chk("target out")
        if t_out is None:
            _restore()
            return None
        _chk("down pass")

        # --- passes 2-7: one H/V pair per bounce ---
        # The divisor is the reduction actually achieved, not the requested k: the loop
        # above stops early on small targets and the sigma has to follow the grid it is
        # applied on, or the halo radius silently shrinks with the canvas.
        red = float(sw) / float(ww)
        for i in range(3):
            sigma = float(sigma_px[i]) * float(np.sqrt(i + 1)) / red
            # horizontal: l0 -> h
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, t_l0)
            self._blur.bind()
            self._u1i(self._blur, "u_src", 0)
            self._u2f(self._blur, "u_step", 1.0 / ww, 0.0)
            self._u1f(self._blur, "u_sigma", sigma)
            self._draw(self._blur, f_h, ww, wh)
            self._blur.release()
            # vertical: h -> b_i
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, t_h)
            self._blur.bind()
            self._u1i(self._blur, "u_src", 0)
            self._u2f(self._blur, "u_step", 0.0, 1.0 / wh)
            self._u1f(self._blur, "u_sigma", sigma)
            self._draw(self._blur, b_fbo[i], ww, wh)
            self._blur.release()

        # --- pass 8: sum the bounces, upsample, DC normalise ---
        for unit, tex in ((0, e_tex), (1, b_tex[0]), (2, b_tex[1]), (3, b_tex[2])):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        self._comp.bind()
        self._u1i(self._comp, "u_raw", 0)
        self._u1i(self._comp, "u_b0", 1)
        self._u1i(self._comp, "u_b1", 2)
        self._u1i(self._comp, "u_b2", 3)
        g3c = [float(v) for v in scene_gain[:3]]
        self._u3f(self._comp, "u_gain", *g3c)
        self._u1f(self._comp, "u_highlights", hs[0])
        self._u1f(self._comp, "u_shadows", hs[1])
        d = [float(v) for v in (list(decay) + [0.0, 0.0, 0.0])[:3]]
        self._u3f(self._comp, "u_decay", *d)
        a = [float(v) for v in (list(a_tot) + [0.0, 0.0, 0.0])[:3]]
        self._u3f(self._comp, "u_a_tot", *a)
        self._u1f(self._comp, "u_amount", amount)
        self._u1i(self._comp, "u_e_shaped", int(e_shaped))
        self._draw(self._comp, f_out, sw, sh)
        self._comp.release()
        _chk("comp pass")

        _restore()
        return t_out

    def release(self):
        for entry in self._targets.values():
            try:
                GL.glDeleteFramebuffers(1, [entry[1]])
                GL.glDeleteTextures([entry[0]])
            except Exception:
                pass
        self._targets.clear()
        for prog in (self._down, self._blur, self._comp, self._scat, self._mix):
            if prog is not None:
                try:
                    prog.delete()
                except Exception:
                    pass
        self._down = self._blur = self._comp = self._scat = self._mix = None
        if self._vbo:
            try:
                GL.glDeleteBuffers(1, [self._vbo])
            except Exception:
                pass
            self._vbo = None
        if self._vao:
            try:
                GL.glDeleteVertexArrays(1, [self._vao])
            except Exception:
                pass
            self._vao = None
        self._ready = False
