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

import math
import os

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

# Spektrafilm's EV-domain highlight reconstruction, mirroring
# core/halation_model._boost_highlights. Everything at or below the protection
# point x0 is untouched; above it the curve is built so the frame max lands on
# exactly m * 2**boost_ev. m is the image-wide max of the *shaped* radiance --
# the CPU takes it after exposure/WB and highlight-shadow shaping, before the
# scatter pass -- so it has to be measured on the same domain here.
_BOOST_GLSL = """
uniform float u_boost_ev;
uniform float u_boost_max;
uniform float u_boost_range;
uniform float u_protect_ev;
vec3 boost_radiance(vec3 c) {
    if (u_boost_ev <= 1e-4) return c;
    float m = u_boost_max;
    if (!(m > 0.0)) return c;
    float x0 = min(max(0.184 * exp2(u_protect_ev), 0.0), m);
    if (x0 >= m) return c;
    float a = pow(28.0, 1.0 - u_boost_range);
    float t0 = a * (1.0 - x0 / m);
    float den = exp(t0) - t0 - 1.0;
    if (den <= 1e-9) return c;
    float kk = (exp2(u_boost_ev) - 1.0) / den;
    vec3 dx = (c - vec3(x0)) / m;
    vec3 t = clamp(a * dx, vec3(-60.0), vec3(60.0));
    vec3 shaped = exp(t) - a * dx - 1.0;
    return mix(c, c + m * kk * shaped, step(vec3(x0), c));
}
"""

# Convert to scene radiance, then apply the boost. This pass only exists when the
# boost is enabled: without it the scatter pass can shape the raw negative in place
# and no full-res intermediate is needed. Running it unconditionally would cost a
# full-res read+write on every frame for a control that sits at 0 by default.
_PREP_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
uniform vec3 u_gain;
""" + _HS_GLSL + _BOOST_GLSL + """
void main() {
    vec3 c = shape_radiance(texture(u_src, v_uv).rgb * u_gain);
    fragColor = vec4(boost_radiance(c), 1.0);
}
"""

# Boost alone, on an already-shaped frame: the diffusion stage sits between the
# radiometric prep and the boost, so when both run the prep cannot fold the boost in.
_BOOSTONLY_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
""" + _BOOST_GLSL + """
void main() {
    fragColor = vec4(boost_radiance(texture(u_src, v_uv).rgb), 1.0);
}
"""

# Halve the resolution taking the 2x2 maximum instead of the average. Chained from
# the full frame down to an 8x8 tile this measures the image-wide max of the shaped
# radiance at ~1.33x one fullscreen pass, which the boost needs for its global
# normalisation. Averaging (the decimate pass) would destroy exactly the values the
# boost cares about: a point source diluted over a 16x16 block reads 1/256 of its
# energy and the reconstruction would key off the wrong frame max.
_REDUCE_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform vec3 u_gain;
uniform int u_shape;
""" + _HS_GLSL + """
void main() {
    // Max must NOT use the bilinear-boundary pattern the box average used to: a max
    // over boundary straddles dilutes a point source by half every level (24 -> 5.5
    // measured), and the boost then keys off the wrong frame max. Sample the four
    // texel *centres* covered by this target pixel via gl_FragCoord. Shaping runs
    // per texel BEFORE the max: shape is not monotonic (the shadow weight is a bell),
    // so max(shape(x)) != shape(max(x)) and the CPU takes the max over shaped pixels.
    vec2 bl = (floor(gl_FragCoord.xy) * 2.0 + 0.5) * u_texel;
    vec3 m = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec2 o = vec2(float(i & 1), float(i >> 1)) * u_texel;
        vec3 c = texture(u_src, bl + o).rgb;
        if (u_shape == 1) c = shape_radiance(c * u_gain);
        m = max(m, c);
    }
    fragColor = vec4(m, 1.0);
}
"""

# Halve the resolution with a true 2x2 box average. One bilinear sample taken at the
# centre of a 2x2 texel block -- which is exactly v_uv of the halved target -- weighs
# the four texels 0.25 each, i.e. cv2.INTER_AREA by 2. The earlier implementation
# sampled v_uv +/- 0.5 texel, which lands on texel *boundaries*: a 4-tap sliding
# window sharing texels with its neighbours. Still a low-pass (which is why the
# chain passed its accuracy gates), but not the kernel the comment claimed, and the
# mismatch showed up the moment a max-reduce was built on the same pattern.
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
    // One bilinear sample at v_uv == the centre of this pixel's 2x2 source block.
    // (An explicit 4-tap version was measured slightly WORSE end-to-end: 46.8 -> 45.4
    // dB on glimmerglass. The halation decimation is validated at 120 dB against
    // cv2.INTER_AREA with this form; keep it.)
    vec3 c = texture(u_src, v_uv).rgb;
    if (u_shape == 1) c = shape_radiance(c * u_gain);
    fragColor = vec4(c, 1.0);
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
uniform int u_shaped;         // 1 when u_sharp is still an unshaped negative
""" + _HS_GLSL + """
void main() {
    // The sharp side has to be shaped here or the mix would blend two different radiometric
    // domains: the blurred side went through shape() in the first scatter pass. But when
    // the boost prep pass already shaped the frame, shaping again would apply the gain
    // twice -- hence the flag, same contract as the composite's u_e_shaped.
    vec3 sharp = texture(u_sharp, v_uv).rgb;
    if (u_shaped == 1) sharp = shape_radiance(sharp * u_gain);
    fragColor = vec4(mix(sharp, texture(u_blur, v_uv).rgb, u_eff), 1.0);
}
"""

# --- diffusion stages (core/diffusion_model.py plan) -------------------------
# One axis of a separable Gaussian with an optional weighted accumulation: the
# vertical pass reads the running accumulator and writes back acc + amp*blur, so
# each sigma costs exactly two passes and no per-sigma temporaries survive.
# The tap weights are normalised over the taps actually summed (acc/wsum), which
# for small sigmas degenerates to the grid-sum kernel the CPU's sepFilter2D uses --
# the two renderers agree on sub-pixel sigmas for the same reason.
_BLURACC_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
uniform sampler2D u_acc;     // running accumulator (unused when u_has_acc == 0)
uniform vec2 u_step;
uniform float u_sigma;
uniform vec3 u_amp;          // per-channel weight of this component's result
uniform int u_has_acc;
const int MAX_TAP = 24;
void main() {
    float s = max(0.05, u_sigma);
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
    vec3 res = acc / wsum;
    vec3 outc = res * u_amp;
    if (u_has_acc == 1) outc += texture(u_acc, v_uv).rgb;
    fragColor = vec4(outc, 1.0);
}
"""

# acc = amp * src, used to seed the accumulator with (1 - p_s) * E.
_SCALE_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_src;
uniform vec3 u_amp;
void main() {
    fragColor = vec4(texture(u_src, v_uv).rgb * u_amp, 1.0);
}
"""

# acc_full += bilinear(src_low): folding the upsample of a decimated group's
# accumulator into a single pass. One bilinear tap from a 1/n texture is exactly the
# interpolation cv2.resize(..., INTER_LINEAR) performs on the way back up.
_UPADD_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_acc;
uniform sampler2D u_src;
uniform vec3 u_amp;
void main() {
    fragColor = vec4(texture(u_acc, v_uv).rgb + texture(u_src, v_uv).rgb * u_amp, 1.0);
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


#: Components under this sigma run at full resolution -- mirrors
#: core.diffusion_model.COMPONENT_FULLRES_SIGMA_PX / halation_model's decimate floor.
#: Keep the two in sync; the CPU routes each component with its copy of the rule.
_COMPONENT_FULLRES_SIGMA = 6.0


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
        self._prep = None
        self._reduce = None
        self._bluracc = None
        self._scale = None
        self._upadd = None
        self._boostonly = None
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
        self._prep = self._compile(_PREP_FRAG, "halation.prep")
        self._reduce = self._compile(_REDUCE_FRAG, "halation.reduce")
        self._bluracc = self._compile(_BLURACC_FRAG, "halation.bluracc")
        self._scale = self._compile(_SCALE_FRAG, "halation.scale")
        self._upadd = self._compile(_UPADD_FRAG, "halation.upadd")
        self._boostonly = self._compile(_BOOSTONLY_FRAG, "halation.boostonly")
        if not (self._down and self._blur and self._comp and self._scat and self._mix
                and self._prep and self._reduce and self._bluracc and self._scale
                and self._upadd and self._boostonly):
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

    def _target(self, key: str, w: int, h: int, f32: bool = False):
        """Lazily (re)allocate an RGBA16F (or RGBA32F) colour target at w x h.

        Kept for callers that need it, but measured NOT to help the boost path:
        promoting the whole chain (or just the low-res blur targets) to RGBA32F
        changed nothing (65.84/54.45/41.38 dB identical to 16F), so the residual
        against the CPU is quadrature geometry, not storage precision, and the
        24 MP chain cost 75% more at 32F. Half floats stay.
        """
        fmt = GL.GL_RGBA32F if f32 else GL.GL_RGBA16F
        entry = self._targets.get(key)
        if entry is not None and entry[2] == w and entry[3] == h and entry[4] == fmt:
            return entry[0], entry[1]
        if entry is not None:
            GL.glDeleteFramebuffers(1, [entry[1]])
            GL.glDeleteTextures([entry[0]])

        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, fmt, w, h, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        # Legacy B (docs/push-grain-plan.md 14.2): the CPU side blurs with
        # cv2.BORDER_REFLECT_101 while GL clamps to the edge, so the border ring differs.
        # On the 47 px working grid that ring is ~8 % of the area -- the right order to
        # explain the remaining 1.5 dB on cinebloom 1.0.  HAL_BORDER=mirror switches the
        # intermediate targets to mirroring so the hypothesis can be tested without a patch.
        _wrap = (GL.GL_MIRRORED_REPEAT if os.environ.get("HAL_BORDER") == "mirror"
                 else GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, _wrap)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, _wrap)
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

        self._targets[key] = (tex, fbo, w, h, fmt)
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

    # ------------------------------------------------------------------ boost
    def _reduce_max(self, tex: int, w: int, h: int, g3, hs) -> float:
        """Image-wide max of the shaped radiance, via chained 2x2 max reductions.

        The first level folds gain + highlight-shadow shaping, matching where the CPU
        takes ``linear_full.max()`` (after shaping, before the boost). Max is associative,
        so the intermediate resolutions lose nothing; the 8x8 tail is read back in one
        glReadPixels of 256 floats. The sync stall is only paid when the boost is on.
        """
        cur_tex, cw, ch = int(tex), w, h
        level = 0
        while cw > 8 or ch > 8:
            nw, nh = max(8, cw // 2), max(8, ch // 2)
            t_dst, f_dst = self._target(f"r{level}", nw, nh)
            if t_dst is None:
                return -1.0
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, cur_tex)
            self._reduce.bind()
            self._u1i(self._reduce, "u_src", 0)
            self._u2f(self._reduce, "u_texel", 1.0 / cw, 1.0 / ch)
            self._u3f(self._reduce, "u_gain", *g3)
            self._u1f(self._reduce, "u_highlights", hs[0])
            self._u1f(self._reduce, "u_shadows", hs[1])
            self._u1i(self._reduce, "u_shape", 1 if level == 0 else 0)
            self._draw(self._reduce, f_dst, nw, nh)
            self._reduce.release()
            cur_tex, cw, ch = t_dst, nw, nh
            level += 1
        fbo = self._targets[f"r{level - 1}"][1]
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        buf = GL.glReadPixels(0, 0, cw, ch, GL.GL_RGBA, GL.GL_FLOAT)
        arr = np.frombuffer(buf, dtype=np.float32).reshape(ch, cw, 4)
        return float(arr[..., :3].max())

    # ------------------------------------------------------------------ diffusion
    def _run_diffusion(self, e_tex: int, sw: int, sh: int, mix) -> int | None:
        """Evaluate the Gaussian-mixture diffusion plan (core/diffusion_model.py).

        out = (1-p_s)*E + w_core*Core(E) + w_halo_c*Halo_c(E) + w_bloom*Bloom(E),
        each group a sum of per-sigma separable blurs accumulated by the vertical
        bluracc pass. Halo runs on a 1/2 grid and bloom on a 1/8 grid, mirroring the
        CPU's divisors; a component whose sigma would drop below the discrete-kernel
        band on its group's grid runs at full resolution instead, exactly like the
        CPU's blur() threshold. Returns the accumulator texture (an owned target, so
        it stays valid until the next run reuses it).
        """
        keep = 1.0 - float(mix.p_s)

        t_acc, f_acc = self._target("dacc", sw, sh)
        t_hf, f_hf = self._target("dfh", sw, sh)
        if t_acc is None or t_hf is None:
            return None
        # seed: acc = (1 - p_s) * E
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, e_tex)
        self._scale.bind()
        self._u1i(self._scale, "u_src", 0)
        self._u3f(self._scale, "u_amp", keep, keep, keep)
        self._draw(self._scale, f_acc, sw, sh)
        self._scale.release()

        # pair helper: H from src into hf, V from hf into acc with the weight
        def hv_add(src_tex, acc_tex, acc_fbo, w, h, sigma, amp_vec, hf_tex, hf_fbo):
            # horizontal: src -> hf (plain blur, amp 1)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, src_tex)
            self._bluracc.bind()
            self._u1i(self._bluracc, "u_src", 0)
            self._u1i(self._bluracc, "u_has_acc", 0)
            self._u2f(self._bluracc, "u_step", 1.0 / w, 0.0)
            self._u1f(self._bluracc, "u_sigma", sigma)
            self._u3f(self._bluracc, "u_amp", 1.0, 1.0, 1.0)
            self._draw(self._bluracc, hf_fbo, w, h)
            # vertical: hf -> acc + amp*blur (re-bind: _draw released the program)
            self._bluracc.bind()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, hf_tex)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, acc_tex)
            self._u1i(self._bluracc, "u_src", 0)
            self._u1i(self._bluracc, "u_acc", 1)
            self._u1i(self._bluracc, "u_has_acc", 1)
            self._u2f(self._bluracc, "u_step", 0.0, 1.0 / h)
            self._u1f(self._bluracc, "u_sigma", sigma)
            self._u3f(self._bluracc, "u_amp", *amp_vec)
            self._draw(self._bluracc, acc_fbo, w, h)
            self._bluracc.release()
            GL.glActiveTexture(GL.GL_TEXTURE0)

        def decimate(src_tex, src_w, src_h, key):
            """One 2x box-average step (the halation decimate pass, u_shape=0)."""
            nw, nh = max(2, src_w // 2), max(2, src_h // 2)
            t_d, f_d = self._target(key, nw, nh)
            if t_d is None:
                return None, 0, 0
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, src_tex)
            self._down.bind()
            self._u1i(self._down, "u_src", 0)
            self._u2f(self._down, "u_texel", 1.0 / src_w, 1.0 / src_h)
            self._u3f(self._down, "u_gain", 1.0, 1.0, 1.0)
            self._u1f(self._down, "u_highlights", 0.0)
            self._u1f(self._down, "u_shadows", 0.0)
            self._u1i(self._down, "u_shape", 0)
            self._draw(self._down, f_d, nw, nh)
            self._down.release()
            return t_d, nw, nh

        def upadd(low_tex, acc_tex, acc_fbo, amp):
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, acc_tex)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, low_tex)
            self._upadd.bind()
            self._u1i(self._upadd, "u_acc", 0)
            self._u1i(self._upadd, "u_src", 1)
            self._u3f(self._upadd, "u_amp", *amp)
            self._draw(self._upadd, acc_fbo, sw, sh)
            self._upadd.release()
            GL.glActiveTexture(GL.GL_TEXTURE0)

        # --- core: full resolution, grey weights ---
        for s, a in zip(mix.core.sigmas_px, mix.core.amps):
            if a <= 1e-6 or s < 0.05:
                continue
            hv_add(e_tex, t_acc, f_acc, sw, sh, s,
                   (mix.w_core * a,) * 3, t_hf, f_hf)

        # --- halo: decimated grid unless a component would go sub-pixel ---
        if mix.w_halo > 1e-9:
            div = max(1, int(mix.halo.divisor))
            # `divisor` is a FACTOR (CPU: cv2.resize to w//ds), not a level count: a
            # divisor of 8 means three halves. The loop used to run `divisor` times,
            # i.e. 2**divisor -- a 256x reduction for halation and 2**128 for bloom
            # (capped only by the >=2 px guard, so the working grid collapsed to a
            # handful of pixels). Measured 2026-10-30 on the real chain: treating it
            # as a factor is worth +22 dB at glimmerglass 1.0 / +20 dB at black pro
            # mist 1.0.
            nlev = max(0, int(round(math.log2(div))))
            d_tex, dw, dh = e_tex, sw, sh
            level = 0
            while level < nlev and dw // 2 >= 2 and dh // 2 >= 2:
                d_tex, dw, dh = decimate(d_tex, dw, dh, f"dd{level}")
                if not d_tex:
                    return None
                level += 1
            t_a2, f_a2 = self._target("dacc2", dw, dh)
            t_h2, f_h2 = self._target("dfh2", dw, dh)
            if t_a2 is None or t_h2 is None:
                return None
            # zero the low-res accumulator with a scale pass (amp 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, d_tex)
            self._scale.bind()
            self._u1i(self._scale, "u_src", 0)
            self._u3f(self._scale, "u_amp", 0.0, 0.0, 0.0)
            self._draw(self._scale, f_a2, dw, dh)
            self._scale.release()
            # split: components fine on the decimated grid vs full-resolution ones.
            # The threshold mirrors the CPU's _gauss_decimated (sigma < 6 px never
            # touches a decimated grid) -- the two renderers must route each component
            # identically or the grids diverge.
            red = sw / dw
            fine = [(s, a) for s, a in zip(mix.halo.sigmas_px, mix.halo.amps)
                    if s >= _COMPONENT_FULLRES_SIGMA and max(a) > 1e-6]
            coarse = [(s, a) for s, a in zip(mix.halo.sigmas_px, mix.halo.amps)
                      if s < _COMPONENT_FULLRES_SIGMA and max(a) > 1e-6]
            for s, a in fine:
                hv_add(d_tex, t_a2, f_a2, dw, dh, s / (sw / dw),
                       tuple(mix.w_halo * a[c] for c in range(3)), t_h2, f_h2)
            for s, a in coarse:
                hv_add(e_tex, t_acc, f_acc, sw, sh, s,
                       tuple(mix.w_halo * a[c] for c in range(3)), t_hf, f_hf)
            if fine:
                upadd(t_a2, t_acc, f_acc, (1.0, 1.0, 1.0))

        # --- bloom: 1/8 grid, all components decimated (the CPU does the same) ---
        if mix.w_bloom > 1e-9:
            div = max(1, int(mix.bloom.divisor))
            nlev = max(0, int(round(math.log2(div))))   # factor, not level count
            d_tex, dw, dh = e_tex, sw, sh
            level = 0
            while level < nlev and dw // 2 >= 2 and dh // 2 >= 2:
                d_tex, dw, dh = decimate(d_tex, dw, dh, f"bd{level}")
                if not d_tex:
                    return None
                level += 1
            t_a8, f_a8 = self._target("dacc8", dw, dh)
            t_h8, f_h8 = self._target("dfh8", dw, dh)
            if t_a8 is None or t_h8 is None:
                return None
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, d_tex)
            self._scale.bind()
            self._u1i(self._scale, "u_src", 0)
            self._u3f(self._scale, "u_amp", 0.0, 0.0, 0.0)
            self._draw(self._scale, f_a8, dw, dh)
            self._scale.release()
            red = sw / dw
            for s, a in zip(mix.bloom.sigmas_px, mix.bloom.amps):
                if a <= 1e-6:
                    continue
                if s < _COMPONENT_FULLRES_SIGMA:
                    hv_add(e_tex, t_acc, f_acc, sw, sh, s, (mix.w_bloom * a,) * 3, t_hf, f_hf)
                else:
                    hv_add(d_tex, t_a8, f_a8, dw, dh, s / red,
                           (mix.w_bloom * a,) * 3, t_h8, f_h8)
            upadd(t_a8, t_acc, f_acc, (1.0, 1.0, 1.0))

        return t_acc

    @staticmethod
    def decimation_for(max_sigma_px: float) -> int:
        """Power-of-two factor that puts the widest bounce in TARGET_SIGMA band.

        The band is what keeps the blur cheap: with sigma on the working grid near 3 px the
        kernel needs ~9 taps per axis, and the pass runs at 1/k^2 of the pixels. It is
        deliberately not the same k the CPU uses (core.halation_model._pick_decimation aims
        at 4 px). Measured 2026-10-30: aligning the two (GPU target 4 px) changed the
        end-to-end PSNR by exactly 0.00 dB while costing 106 -> 156 ms at 24 MP, so the
        residual between the renderers is kernel quadrature, not grid mismatch -- the
        grids stay free to optimise for cost.
        """
        if max_sigma_px <= GLHalationChain.TARGET_SIGMA_MAX:
            return 1
        k = int(2 ** round(np.log2(max_sigma_px / 3.5)))
        return max(1, min(GLHalationChain.MAX_DECIMATION, k))

    # ------------------------------------------------------------------ run
    def run(self, raw_tex: int, width: int, height: int, scene_gain,
            sigma_px, decay, a_tot, amount: float, hs=(0.0, 0.0), scatter=None,
            boost=None, diffusion=None):
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
        boost     -- optional dict {"ev": float, "range": float, "protect": float,
                     "max": float|None}; "max" may carry a pre-computed image-wide max
                     of the shaped radiance (required for tiled rendering, where a tile
                     cannot see the whole frame). When absent the chain measures it
                     itself with a chained 2x2-max reduce.

        Returns None if the chain could not run (caller should fall back).
        """
        if not self.ensure():
            return None
        self.last_did_scatter = False
        self.last_did_boost = False
        self.last_did_diffusion = False
        want_hal = amount > 0.001 and max(sigma_px) > 0.01
        want_scat = bool(scatter) and float(scatter.get("amount", 0.0)) > 0.001
        want_diff = diffusion is not None
        boost_ev = float(boost.get("ev", 0.0)) if boost else 0.0
        # The CPU gates the boost on amount only (halation_model.apply_halation_np:
        # ``boost_ev > 1e-4 and amount > 0.001``); the scatter/halation switches are
        # independent of it.
        want_boost = boost_ev > 1e-4 and amount > 0.001
        if not want_hal and not want_scat and not want_diff:
            return None
        self.last_did_halation = bool(want_hal)
        self.last_did_boost = bool(want_boost)
        self.last_did_diffusion = bool(want_diff)

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
        if k <= 1 and not want_diff:
            return None  # sigma already tiny; the tap path is equivalent and cheaper

        _chk("start")
        t_h, f_h = None, None
        g3 = [float(v) for v in scene_gain[:3]]

        # --- prep -> diffusion -> boost, in exactly the CPU's order ---
        # CPU: exposure/WB -> HS -> (diffusion) -> boost -> scatter -> halation, and
        # the boost's normaliser is the max of the frame *after* diffusion. So: the
        # prep pass folds gain+shape (needed whenever diffusion or boost runs, because
        # both must operate on shaped radiance), the diffusion stage evaluates the
        # Gaussian-mixture plan from core/diffusion_model.py, and only then is the
        # frame max measured and the boost applied. Everything downstream (scatter,
        # decimate) consumes the result with u_shape=0. A caller that renders in tiles
        # must pass boost["max"] -- a tile cannot see the whole frame.
        e_tex, e_shaped = int(raw_tex), 0
        if want_diff or want_boost:
            t_p, f_p = self._target("prep", sw, sh)
            if t_p is None:
                _restore()
                return None
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, raw_tex)
            self._prep.bind()
            self._u1i(self._prep, "u_src", 0)
            self._u3f(self._prep, "u_gain", *g3)
            self._u1f(self._prep, "u_highlights", hs[0])
            self._u1f(self._prep, "u_shadows", hs[1])
            # boost constants are set but inert while u_boost_ev == 0
            self._u1f(self._prep, "u_boost_ev", 0.0)
            self._u1f(self._prep, "u_boost_max", 1.0)
            self._u1f(self._prep, "u_boost_range", float(boost.get("range", 0.3)) if boost else 0.3)
            self._u1f(self._prep, "u_protect_ev", float(boost.get("protect", 4.0)) if boost else 4.0)
            self._draw(self._prep, f_p, sw, sh)
            self._prep.release()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            e_tex, e_shaped = t_p, 1
            _chk("prep")

        if want_diff:
            e_tex = self._run_diffusion(e_tex, sw, sh, diffusion)
            if e_tex is None:
                import os as _o
                if _o.environ.get("HAL_DEBUG"):
                    print("[RUN] diffusion stage returned None")
                _restore()
                return None
            _chk("diffusion")

        if want_boost:
            b_max = boost.get("max")
            if b_max is None:
                b_max = self._reduce_max(e_tex, sw, sh, (1.0, 1.0, 1.0), (0.0, 0.0))
            if not (b_max > 0.0):
                _restore()
                return None
            t_b, f_b = self._target("boost", sw, sh)
            if t_b is None:
                _restore()
                return None
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, e_tex)
            self._boostonly.bind()
            self._u1i(self._boostonly, "u_src", 0)
            self._u1f(self._boostonly, "u_boost_ev", boost_ev)
            self._u1f(self._boostonly, "u_boost_max", float(b_max))
            self._u1f(self._boostonly, "u_boost_range", float(boost.get("range", 0.3)))
            self._u1f(self._boostonly, "u_protect_ev", float(boost.get("protect", 4.0)))
            self._draw(self._boostonly, f_b, sw, sh)
            self._boostonly.release()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            e_tex, e_shaped = t_b, 1
            _chk("boost")

        # --- scatter: sharp -> H -> V -> mix, all at full resolution ---
        # Scatter sigmas are 0.5-2 px, so decimating them would be butchery; the cost is two
        # full-res passes with <=12 taps, which is still cheaper than the halation chain.
        # The shaping flag follows e_shaped: when the boost ran, its prep pass already
        # converted the frame to shaped radiance and shaping again would be non-linear
        # re-application (the same reason the later decimate steps do not shape).
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
            sc_shape = 0 if e_shaped else 1
            for axis, (src, dst, f_dst, do_shape) in enumerate((
                    (e_tex, t_sh, f_sh, sc_shape),
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
            GL.glBindTexture(GL.GL_TEXTURE_2D, e_tex)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, t_s)
            self._mix.bind()
            self._u1i(self._mix, "u_sharp", 0)
            self._u1i(self._mix, "u_blur", 1)
            self._u1f(self._mix, "u_eff", float(scatter.get("eff", 0.06)))
            self._u3f(self._mix, "u_gain", *g3)
            self._u1f(self._mix, "u_highlights", hs[0])
            self._u1f(self._mix, "u_shadows", hs[1])
            self._u1i(self._mix, "u_shaped", 0 if e_shaped else 1)
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
        for prog in (self._down, self._blur, self._comp, self._scat, self._mix,
                     self._prep, self._reduce, self._bluracc, self._scale, self._upadd,
                     self._boostonly):
            if prog is not None:
                try:
                    prog.delete()
                except Exception:
                    pass
        self._down = self._blur = self._comp = self._scat = self._mix = None
        self._prep = self._reduce = None
        self._bluracc = self._scale = self._upadd = self._boostonly = None
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
