"""The GPU grain stage: two cubes and, between them, the film's particles.

Why two cubes instead of one: the grain law is a function of *negative density*, and a single
RGB->RGB table never exposes it.  Splitting the chain at the density lets the shader do

    linear RGB --[A]--> (C, M, Y) density --+grain--> --[B]--> output RGB

which is the order the reference runs too.  Both cubes are baked by the engine from the same
pipeline, keyed together, so they cannot describe two different prints.

The grain itself is the reference's particle model, evaluated per pixel:

    p     = D / Dmax                      probability a grain develops
    sat   = 1 - p * uniformity            the layer saturates, so the rate rises
    X     ~ Poisson((n / sat) * p)        the binomial(Poisson) of the reference, thinned
    grain = X * (Dmax / n) * sat

Randomness is a counter-based hash keyed on the *absolute image pixel*, so a render is
reproducible and a chunked export has no seams: the same pixel always draws the same particle
count.  (The frame-wide blur the reference applies afterwards is spatial and is handled
separately -- see ``u_grain_taps``.)
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL

# The document/export parameter ``grain`` is a fotufilm-era slider value (0.4 by default),
# not a physical quantity.  The gold200 calibration was made at reference strength 1.0, so
# 0.4 has to land on 1.0: the slider becomes a multiplier around the calibrated reference
# instead of quietly halving the grain on every fresh document.  0 is a real "off".
# M5 will rebuild the panel; M4 only makes the value live on both the preview and the export.
GRAIN_REFERENCE = 0.4

GRAIN_GLSL = r"""
// ---------------------------------------------------------------------------------------------
// Negative density and the print, as two tables with room for grain in between.
// ---------------------------------------------------------------------------------------------
uniform int       u_grain_cubes;      // 1 = the two-cube path is uploaded and current
uniform sampler3D u_density_lut;      // shaper(linear RGB) -> (C, M, Y) net density
uniform sampler3D u_print_lut;        // normalised (C, M, Y) -> output RGB
uniform vec3      u_density_min;      // the density range the print cube covers
uniform vec3      u_density_max;

// The sub-layer split, resampled onto a uniform density axis: texel (density, record) holds the
// three coated layers' densities, so the shader does one multiply and one fetch.
uniform sampler2D u_grain_table;
uniform vec3      u_grain_axis_min;
uniform vec3      u_grain_axis_max;

// Per (sub-layer, record): the layer ceiling with its fog share, and the particle rate per um^2.
uniform vec3      u_grain_dmax_layer[3];
uniform vec3      u_grain_min_layer[3];
uniform vec3      u_grain_ppu[3];
uniform vec3      u_grain_uniformity;
uniform vec3      u_grain_dmin;

uniform float     u_grain_px_um;      // film pitch of *this* render, in micrometres
uniform float     u_grain_strength;   // editorial multiplier on the particle rate (1 = reference)
uniform float     u_grain_size;       // editorial multiplier on the particle area (1 = reference)
uniform float     u_grain_colour;     // 1 = independent dye clouds, 0 = shared luminance grain
uniform int       u_grain_seed;
uniform int       u_grain_taps;       // 1 = off for the spatial blur, 9 = 3x3 gaussian
uniform int       u_grain_probe;      // diagnostic: paint the density delta instead of the print

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

// --- randomness ------------------------------------------------------------------------------
// Counter-based: the state is seeded from the pixel, so the draw is a pure function of position
// and never depends on how the frame was split into tiles or fragments.
uvec3 pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

uint rng_next(inout uvec3 state) {
    state = pcg3d(state);
    return state.x;
}

float rng_unit(inout uvec3 state) {
    return float(rng_next(state)) * (1.0 / 4294967296.0);
}

float log_gamma(float x) {
    const float c[6] = float[6](76.18009172947146, -86.50532032941677, 24.01409824083091,
                                -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5);
    float y = x;
    float tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);
    float ser = 1.000000000190015;
    for (int j = 0; j < 6; j++) {
        y += 1.0;
        ser += c[j] / y;
    }
    return -tmp + log(2.5066282746310005 * ser / x);
}

// PTRS: the sampler numpy uses, so the distribution is the reference's, not an approximation.
float poisson(inout uvec3 state, float mu) {
    if (mu < 10.0) {
        float enlam = exp(-mu);
        float X = 0.0;
        float prod = 1.0;
        for (int i = 0; i < 64; i++) {
            prod *= rng_unit(state);
            if (prod > enlam) X += 1.0; else return X;
        }
        return X;
    }
    float slam = sqrt(mu);
    float loglam = log(mu);
    float b = 0.931 + 2.53 * slam;
    float a = -0.059 + 0.02483 * b;
    float invalpha = 1.1239 + 1.1328 / (b - 3.4);
    float vr = 0.9277 - 3.6224 / (b - 2.0);
    for (int i = 0; i < 64; i++) {
        float U = rng_unit(state) - 0.5;
        float V = rng_unit(state);
        float us = 0.5 - abs(U);
        float k = floor((2.0 * a / us + b) * U + mu + 0.43);
        if (us >= 0.07 && V <= vr) return k;
        if (k < 0.0 || (us < 0.013 && V > us)) continue;
        if (log(V) + log(invalpha) - log(a / (us * us) + b)
                <= -mu + k * loglam - log_gamma(k + 1.0)) return k;
    }
    return floor(mu);
}

// Standard normal from two uniforms (Box-Muller).  The tail is the reason the model uses a
// gaussian rather than a fixed multiple of the sd: grain has to be able to spike.
float gauss_unit(inout uvec3 state) {
    float u1 = max(rng_unit(state), 1e-7);
    float u2 = rng_unit(state);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
// One draw per record, not one per sub-layer.
//
// The reference draws each of the three coated sub-layers from its own Poisson(rate), then
// sums them with per-layer weights.  At any pitch this app renders, every rate is in the
// hundreds (750 particles per pixel across the three layers at 5.8 um, thousands at preview
// pitches), and a sum of Poissons with those means is normal to far better than the 8 bits the
// frame is written in.  So the three layers are folded into the two moments that survive:
//
//     mean = sum_l amp_l * mu_l          var = sum_l amp_l^2 * mu_l
//
// with amp_l = dmax_l/npp_l * sat_l and mu_l = (npp_l/sat_l) * p_l.  `mean` is exactly the
// layer sum the reference would average to, and `var` is its variance -- so sigma, which is
// what the grain *is*, is unchanged.  What it buys is 9 Poisson draws per pixel becoming 3
// gaussians, which is what makes the 5x5 dye-cloud convolution below affordable.
//
// Returns the density the film actually developed -- an *absolute* density, not a delta: the
// reference's `apply_grain_to_density_layers` hands back the summed particle density, and the
// caller replaces.  Getting this backwards brightens the whole frame.
vec3 grain_apply(vec3 density, ivec2 px) {
    if (u_grain_strength <= 0.0) return density;
    float area = u_grain_px_um * u_grain_px_um;
    vec3 total = vec3(0.0);
    vec3 mean_total = vec3(0.0);          // noiseless layer sum, i.e. the reference's base
    for (int ch = 0; ch < 3; ch++) {
        float lo = u_grain_axis_min[ch];
        float hi = u_grain_axis_max[ch];
        float u = clamp((density[ch] - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
        vec4 layer = texture(u_grain_table, vec2(u, (float(ch) + 0.5) / 3.0));
        float mean = 0.0;
        float var = 0.0;
        for (int sl = 0; sl < 3; sl++) {
            float dmax = u_grain_dmax_layer[sl][ch];
            float npp = u_grain_ppu[sl][ch] * area * u_grain_strength / u_grain_size;
            if (npp <= 0.0 || dmax <= 0.0) continue;
            float d = layer[sl] + u_grain_min_layer[sl][ch];
            float p = clamp(d / dmax, 1e-6, 1.0 - 1e-6);
            float sat = 1.0 - p * u_grain_uniformity[ch] * (1.0 - 1e-6);
            float mu = (npp / sat) * p;
            float amp = (dmax / npp) * sat;
            mean += amp * mu;
            var += amp * amp * mu;
        }
        uvec3 state = uvec3(uint(px.x), uint(px.y), uint(u_grain_seed + ch));
        total[ch] = mean + sqrt(max(var, 0.0)) * gauss_unit(state);
        mean_total[ch] = mean;
    }
    // The reference's baseline is the *layer sum*, not the density it was handed:
    // ``apply_grain_to_density_layers`` sets ``density_cmy_base = sum(layers)``, perturbs the
    // particles, and then puts the mixed fluctuation back on that base.  Whenever the layer
    // split does not reproduce the density exactly -- which for a reversal stock it does not,
    // by a lot at high density -- the difference is a real tonal modulation of the frame and
    // it belongs in the output.  Using ``density`` here instead discards it, and on a positive
    // that is most of what the eye reads as grain.
    vec3 base = mean_total - u_grain_dmin;
    if (u_grain_probe == 6) return total;
    // The reference does not ship three independent dye-cloud fields: it mixes them back
    // towards their common mean (fotufilm's ``Look.mix``, ``mix_chroma_grain`` upstream).
    // ``own`` keeps each record's own draw, ``shared`` adds the mean of the three, and the
    // weights are chosen so the per-record variance is unchanged while the *luminance*
    // variance rises by 1/sqrt(0.557) = 1.34x -- the grain stops reading as coloured noise
    // and starts reading as silver.  Skipping this was a large part of "the grain is weak":
    // two thirds of the energy was landing in chroma, where the eye barely sees it.
    vec3 net = total - u_grain_dmin;
    vec3 fluct = net - base;
    float own = clamp(u_grain_colour, 0.0, 1.0);
    float shared = sqrt(max(0.0, 3.0 - 2.0 * own * own)) - own;
    float mean_fluct = (fluct.r + fluct.g + fluct.b) / 3.0;
    return base + own * fluct + shared * mean_fluct;
}

// The reference ends with two convolutions on the assembled density: the four Jarvis dye-cloud
// gaussians, then a frame blur (0.65 px upstream -- 0.45 here, see `grain_bake.FRAME_BLUR_PX`).
// Both are isotropic, so they are one separable kernel,
// and `core.grain_bake.grain_kernel_1d` reads that kernel's centre row back off the reference's
// own cascade at this pitch -- the weights here are the reference's numbers, not a re-derivation
// (a re-derived gaussian came out 13 % too wide and lost 13 % of the sigma).
//
// A pointwise renderer cannot express a spatial filter for free, so the taps are evaluated:
// every tap is the same pure function of its own pixel, which is what a two-pass blur does.
uniform float u_grain_kernel[5];

vec3 grain_field(vec3 density, vec2 image_px) {
    ivec2 px = ivec2(floor(image_px + 0.5));
    if (u_grain_taps < 2) return grain_apply(density, px);
    vec3 acc = vec3(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            float w = u_grain_kernel[dx + 2] * u_grain_kernel[dy + 2];
            acc += grain_apply(density, px + ivec2(dx, dy)) * w;
        }
    }
    return acc;
}
"""


class GLCubeGrain:
    """Owns the tables the stage reads. One instance per canvas context."""

    def __init__(self):
        self._density_tex = 0
        self._density_size = 0
        self._print_tex = 0
        self._print_size = 0
        self._table_tex = 0
        self._table_signature = None
        self.density_min = np.zeros(3, dtype=np.float32)
        self.density_max = np.ones(3, dtype=np.float32)
        self.grain = {}
        self.unsharp = (0.0, 0.0)
        self.layer_axis_min = np.zeros(3, dtype=np.float32)
        self.layer_axis_max = np.ones(3, dtype=np.float32)
        self._kernel = np.full(5, 0.2, dtype=np.float32)
        self._kernel_key = None
        self.ready = False
        self._signature = None
        self._pending = None
        # Which bake the uploaded cubes came from (`get_grain_bundle` records its LUT key).
        # `bind` refuses to draw when the caller asks for a different one.
        self.key = None

    # -- upload ------------------------------------------------------------------------------
    def set_bundle(self, bundle):
        """Stage a bundle for the next paint; ``None`` turns the path off."""
        if bundle is None:
            self._pending = None
            self._signature = None
            self.key = None
            self.ready = False
            return
        signature = (id(bundle.get("density")), id(bundle.get("cube")))
        self.key = bundle.get("key")
        if signature == self._signature:
            return
        self._signature = signature
        self._pending = bundle

    def upload_pending(self, unit_a=5, unit_b=6, unit_c=7):
        """Push a staged bundle to the GPU. Call with the canvas context current."""
        pending = self._pending
        if pending is None:
            return
        self._density_tex, self._density_size = self._upload_cube(
            self._density_tex, self._density_size, pending["density"], unit_a)
        self._print_tex, self._print_size = self._upload_cube(
            self._print_tex, self._print_size, pending["cube"], unit_b)

        layer_table = pending["layer_table"]
        self._upload_table(layer_table, unit_c)

        self.density_min = np.asarray(pending["density_min"], dtype=np.float32)
        self.density_max = np.asarray(pending["density_max"], dtype=np.float32)
        self.layer_axis_min = np.asarray(layer_table["axis_min"], dtype=np.float32)
        self.layer_axis_max = np.asarray(layer_table["axis_max"], dtype=np.float32)
        self.grain = pending["grain"]
        self.unsharp = tuple(pending.get("unsharp") or (0.0, 0.0))
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

    def _upload_table(self, layer_table, unit):
        """256 samples x 3 records, RGBA = the three sub-layer densities of that record."""
        table = np.asarray(layer_table["table"], dtype=np.float32)   # (samples, 3, 4)
        data = np.ascontiguousarray(np.transpose(table, (1, 0, 2)))  # (3, samples, 4)
        samples = data.shape[1]
        signature = (samples, id(layer_table["table"]))
        GL.glActiveTexture(0x84C0 + unit)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if self._table_tex and self._table_signature == signature:
            return
        if self._table_tex:
            GL.glDeleteTextures([self._table_tex])
        self._table_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._table_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, samples, 3, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._table_signature = signature

    # -- bind --------------------------------------------------------------------------------
    def bind(self, u_1i, u_1f, u_3f, u_3fv, pixel_size_um, params,
             unit_a=5, unit_b=6, unit_c=7, u_1fv=None, expect_key=None):
        if not self.ready:
            u_1i("u_grain_cubes", 0)
            return False
        # The cubes describe one specific bake. Drawing a frame whose parameters belong to a
        # different one is not a small error: the density axis and the print table stop
        # agreeing, and a negative comes out as an orange positive (measured 2026-11-05).
        # The caller says which bake this frame is; if it is not the one we hold, hand the
        # frame to the single 3D LUT instead -- less grain, but the right picture.
        if expect_key is not None and self.key is not None and expect_key != self.key:
            u_1i("u_grain_cubes", 0)
            return False
        for unit, tex, name in ((unit_a, self._density_tex, "u_density_lut"),
                                (unit_b, self._print_tex, "u_print_lut"),
                                (unit_c, self._table_tex, "u_grain_table")):
            GL.glActiveTexture(0x84C0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_3D if name != "u_grain_table" else GL.GL_TEXTURE_2D, tex)
            u_1i(name, unit)

        u_1i("u_grain_cubes", 1)
        u_3f("u_density_min", *[float(v) for v in self.density_min])
        u_3f("u_density_max", *[float(v) for v in self.density_max])
        u_3f("u_grain_axis_min", *[float(v) for v in self.layer_axis_min])
        u_3f("u_grain_axis_max", *[float(v) for v in self.layer_axis_max])

        g = self.grain or {}
        if not g:
            return True
        # (sl, ch) arrays reach the shader as one vec3 per sub-layer, indexed by record.
        for name, key in (("u_grain_dmax_layer", "density_max_layers"),
                          ("u_grain_min_layer", "density_min_layers"),
                          ("u_grain_ppu", "particles_per_um2")):
            arr = np.asarray(g.get(key, np.zeros((3, 3))), dtype=np.float32)
            u_3fv(name, arr)
        u_3f("u_grain_uniformity", *[float(v) for v in np.asarray(g.get("uniformity", np.ones(3)))])
        u_3f("u_grain_dmin", *[float(v) for v in np.asarray(g.get("density_min", np.zeros(3)))])

        u_1f("u_grain_px_um", float(pixel_size_um))
        # M4 parameter mapping.  `grain_amount` is a diagnostic key used by scratch probes
        # (including the grain-free half of the residual acceptance pair); it wins when present.
        # Otherwise the real document/export key `grain` drives the particle rate -- and
        # `eff_params` reaches here on the offscreen path, so an export honours its own grain
        # setting rather than silently rendering at full strength.
        amount = params.get("grain_amount", None)
        strength = (float(amount) if amount is not None
                    else float(params.get("grain", GRAIN_REFERENCE)) / GRAIN_REFERENCE)
        u_1f("u_grain_strength", max(0.0, strength))
        u_1f("u_grain_size", max(0.05, float(params.get("grain_size", 1.0))))
        # The colour slider is M5's job: for now the calibrated per-bundle value stands, so a
        # future per-stock colour mapping is not overridden by a stale fotufilm-era param.
        u_1f("u_grain_colour", float(g.get("colour_grain", 1.0)))
        u_1i("u_grain_seed", int(params.get("grain_seed", 0)))
        u_1i("u_grain_taps", int(params.get("grain_taps", 25)))
        u_1i("u_grain_probe", int(params.get("grain_probe", 0)))

        # The dye-cloud + frame-blur kernel belongs to the pitch, so it is sized here rather
        # than baked into the bundle.  Five nodes: the reference's cascade has fallen below
        # 1 % of its centre by |dx| = 2 at every pitch this app renders.
        # 0 = the reference cascade as-is, 1 = frame blur only (no dye cloud),
        # 2 = no convolution at all.  The last two exist so the cost of each stage can be
        # measured instead of guessed at.
        mode = int(params.get("grain_kernel_mode", 0))
        final_blur = float(g.get("final_blur_px", 0.0) or 0.0)
        # The decay length is calibration, not a per-stock value (upstream hard-codes 1.45),
        # so it rides with the bundle rather than being repeated here.
        decay_um = 0.0 if mode == 1 else float(g.get("dye_cloud_decay_um", 1.45))
        unsharp = self.unsharp if mode == 0 else (0.0, 0.0)
        if mode == 2:
            final_blur = 0.0
            decay_um = 0.0
        key = (round(float(pixel_size_um), 5), round(final_blur, 4), decay_um,
               unsharp, bool(u_1fv))
        if self._kernel_key != key:
            import core.grain_bake as grain_bake
            self._kernel = grain_bake.grain_kernel_1d(
                float(pixel_size_um), final_blur_px=final_blur, decay_um=decay_um,
                unsharp=unsharp)
            self._kernel_key = key
        if u_1fv is not None:
            u_1fv("u_grain_kernel", self._kernel)
        return True

    def release(self):
        for attr in ("_density_tex", "_print_tex", "_table_tex"):
            tex = getattr(self, attr, 0)
            if tex:
                try:
                    GL.glDeleteTextures([tex])
                except Exception:
                    pass
                setattr(self, attr, 0)
        self.ready = False
        self._signature = None
        self._table_signature = None
