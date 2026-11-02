import os
import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QVariantAnimation, QEasingCurve, QRectF, QTimer
from PySide6.QtGui import QCursor, QColor, QOpenGLContext, QAction, QActionGroup, QKeySequence, QPainter, QPen, QBrush, QFont, QPixmap, QIcon
from PySide6.QtWidgets import QMenu, QLabel, QWidget, QPushButton, QVBoxLayout
from ui.window_utils import get_darkroom_menu_style, get_icon
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLShaderProgram, QOpenGLShader, QOpenGLBuffer, QOpenGLFramebufferObject
)
from core.path_utils import get_resource_dir
from core import config_manager
from core.darkroom_math import scene_gain
from OpenGL import GL


_DIFFUSION_FAMILIES = ("glimmerglass", "black_pro_mist", "pro_mist", "cinebloom")

# Truncation of the fitted exponential disc, in mean free paths. 8 keeps 99.95%
# of a true 2D exponential; larger values compensate for collapsing each group's
# multi-exponential PSF onto a single one.
_DIFF_PSF_CUTOFF = 8.0


def _ca_coeffs(lens_cfg, key, legacy_key):
    """Three-term lateral-CA coefficients from a lens config, with the legacy single-term
    ``ca_kr``/``ca_kb`` as a fallback for sidecars written before the LCP models were read."""
    cfg = lens_cfg if isinstance(lens_cfg, dict) else {}
    raw = cfg.get(key)
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        try:
            return float(raw[0]), float(raw[1]), float(raw[2])
        except (TypeError, ValueError):
            pass
    try:
        return float(cfg.get(legacy_key, 0.0) or 0.0), 0.0, 0.0
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0


def diffusion_shader_profile(family, strength):
    """Uniform payload for the viewport's diffusion filter.

    Mirrors the export's PSF split so the preview and the export blur the same
    physical distance. The export (`apply_diffusion_filter_um`) builds each
    family from core/halo/bloom groups of weighted 2D exponentials; the shader
    cannot afford that many taps, so each group is collapsed to a single
    exponential whose mean free path matches the group's second moment
    (lambda_eff = sqrt(sum w_i * lambda_i^2)) and whose truncation radius is
    8 * lambda_eff, exactly as the export truncates. Weights are the CPU's own
    group weights. Radii are in micrometres, so they are resolution-independent.
    """
    zero = {"ps": 0.0, "core_w": 1.0, "halo_w": 0.0, "bloom_w": 0.0,
            "halo_um": 0.0, "bloom_um": 0.0, "warmth_base": 0.0,
            "cutoff": float(_DIFF_PSF_CUTOFF)}
    fam = str(family or "none").lower()
    try:
        s = float(strength)
    except (TypeError, ValueError):
        return zero
    if s <= 0.0005 or fam in ("none", ""):
        return zero
    if fam not in _DIFFUSION_FAMILIES:
        fam = "black_pro_mist"

    from spektrafilm.model.diffusion import (
        _expand_group, _resolve_family_cfg, _strength_to_scatter,
    )
    cfg = _resolve_family_cfg(fam)
    total = cfg["w_c"] + cfg["w_h"] + cfg["w_b"]
    out = {
        "ps": float(_strength_to_scatter(s, fam)),
        "core_w": cfg["w_c"] / total,
        "halo_w": cfg["w_h"] / total,
        "bloom_w": cfg["w_b"] / total,
        "warmth_base": float(cfg.get("halo_warmth_base", 0.0)),
        "cutoff": float(_DIFF_PSF_CUTOFF),
    }
    for kind in ("halo", "bloom"):
        lam, wt = _expand_group(cfg[kind], kind=kind)
        # Match the export's truncation: it sizes the kernel from the group's
        # LONGEST component (8 * lambda_max), not from an effective wavelength.
        # Using sqrt(sum w*lambda^2) here made the shader's disc 1.5-2.4x
        # narrower than the exported one, which read as the export being both
        # softer and brighter than the preview.
        out[f"{kind}_um"] = _DIFF_PSF_CUTOFF * float(lam.max())
    return out


VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_texcoord;

out vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D u_image;
uniform sampler3D u_lut;
uniform int u_has_image;
uniform int u_has_lut;
uniform int u_is_bypass;
uniform vec3 u_bg_color;

// Camera metering, kept separate from u_scene_gain so the untouched (Before) view can show
// the same exposure calibration the developed view uses without inheriting the user's EV
// slider. Everything else (exposure, temperature, tint) lives in u_scene_gain.
uniform float u_base_ev;
uniform float u_highlights;
uniform float u_shadows;
uniform vec3 u_cmy;
uniform float u_print_exposure;
uniform float u_preflash;
uniform float u_black_lift;

// Halation physics (Official Spektrafilm Model)
uniform float u_halation;           // halation_amount (1.0 = physical default)
uniform float u_hal_spatial_scale;  // halation_spatial_scale (1.0 = physical default)
uniform float u_scatter_amount;     // scatter_amount (1.0 = physical default)
uniform vec3  u_hal_strength_rgb;   // profile-driven RGB preset
uniform float u_hal_sigma_um;       // profile-driven first bounce sigma (um)
// 1.0 when the caller already ran the separable-Gaussian halation chain
// (ui/gl_halation.py) and handed us its output. The halo is then already applied, so the
// tap-based path below must stay off and the gain must not be re-multiplied.
uniform float u_scene_prepared;


// Diffusion filter
// u_diffusion_ps is the deflected-photon fraction p_s, computed on the CPU by
// spektrafilm.model.diffusion._strength_to_scatter() so the preview and the
// export share one calibration table. The shader only consumes it; it must not
// infer intensity from u_diffusion_strength (that was an unbounded additive
// gain and blew the frame out at 2 stops).
uniform int u_diffusion_type;
uniform float u_diffusion_strength;
uniform float u_diffusion_warmth;
uniform float u_diffusion_ps;
// The three PSF groups, mirrored from spektrafilm.model.diffusion so the
// viewport reproduces the export's core/halo/bloom split exactly. Each *_w is
// the group's share of the total PSF (they sum to 1) and each *_um is the
// truncation diameter in micrometres, 8 x the group's energy-equivalent
// exponential mean free path. Radii are therefore physical and independent of
// preview resolution; only p_s changes with the filter stop.
uniform float u_diff_core_w;
uniform float u_diff_halo_w;
uniform float u_diff_bloom_w;
uniform float u_diff_halo_um;
uniform float u_diff_bloom_um;
uniform float u_diff_warmth_base;
uniform float u_diff_cutoff;

// DIR Couplers: fully baked into the 3D LUT only. There is deliberately no shader-side
// DIR approximation - it is a strongly non-linear cross-layer response and a linear
// extrapolation over the slider range is off by up to ~90/255 (it made the drag preview
// disagree with the committed bake).

// Chemistry Morph
uniform float u_morph_gamma;
uniform float u_developer_exhaustion;

// Values already baked into the current 3D LUT (only the delta is approximated post-LUT)
uniform float u_lut_morph_gamma;
uniform float u_lut_dev_exh;
uniform float u_film_width_um;

// Lens Optical & Geometric Correction
uniform int u_lens_enabled;
uniform vec3 u_lens_distort;
// 0 = derive the guard scale from the corner gain (the classic behaviour); anything else
// overrides it. Sony's embedded tables are normalised at the centre, so re-normalising at the
// corner would shrink the frame a further 15% and move it away from the camera's own JPEG.
uniform float u_lens_dist_scale;
// Precomputed border-exact guard (core.lens_warp.fill_scale): the largest scale that keeps
// every one of the four edges inside the source. Unlike the corner-only fallback it is
// derived from the whole border, so nothing samples past the frame and the clamp never has
// to invent pixels -- that stretch is what users saw as a smeared band along the edges.
uniform float u_lens_fill_scale;
uniform vec2 u_lens_vignette;
uniform int u_lens_ca_enabled;
// Lateral CA has two parts. The leading term is a CONSTANT magnification difference between
// channels (a lens focuses red/blue at slightly different scales), whose displacement grows
// linearly with radius. Adobe's LCP polynomial (the vec3 below) only describes the residual
// field-dependent part, because its first basis function is r^2 -> displacement ~ r^3, which
// is exactly zero at the centre. Fitting a real frame with the polynomial alone therefore
// puts nearly all of the correction in the corners and almost nothing in the mid-field; the
// scalar below carries the constant scale and the vec3 the profile residual.
uniform float u_lens_ca_scale_r;
uniform float u_lens_ca_scale_b;
uniform vec3 u_lens_ca_r;
uniform vec3 u_lens_ca_b;

// Interactive Crop & Straighten
uniform int u_crop_enabled;
uniform vec4 u_crop_rect;
uniform float u_crop_angle;
uniform int u_is_crop_editing;
uniform int u_crop_rot90;
uniform vec2 u_crop_flip;

uniform float u_split_x;
uniform int u_view_mode;
uniform float u_screen_aspect;
uniform float u_image_aspect;
uniform float u_native_aspect;
uniform vec2 u_pan;
uniform float u_zoom;
uniform vec2 u_viewport_size;

// Digital (no-film) display transform - must stay in step with core.app_core:
// DIGITAL_TOE_LINEAR / DIGITAL_KNEE_LINEAR / DIGITAL_WHITE_LINEAR / DIGITAL_WHITE_COMP /
// DIGITAL_TAIL_SCALE. Identity below 18% grey, C1 Hermite shoulder to paper white,
// exponential headroom tail. Changing one side without the other moves the bypass preview
// away from what export_image() writes.
const float DIG_TOE = 0.002;
const float DIG_KNEE = 0.18;
const float DIG_WHITE = 1.0;
const float DIG_WHITE_COMP = 0.97;
const float DIG_TAIL = 0.6;

// How strongly a detected fringe is pulled to a neutral of its own luminance.
const float DEFRINGE_STRENGTH = 1.5;
// Minimum fraction of the pixel's own chroma that must sit on a CA hue (magenta/cyan/green/red).
const float DEFRINGE_HUE_MIN = 0.55;
// The pixel must be at least this much more chromatic than the least chromatic pixel of its ring.
const float DEFRINGE_EXCESS_FLOOR = 0.50;
// ... and at least this chromatic relative to the most chromatic one (local ridge, not valley).
const float DEFRINGE_RIDGE_TOL = 0.98;
// Local luminance contrast required, relative to the pixel's own luminance.
const float DEFRINGE_CONTRAST_MIN = 0.08;
// Both sides of the fringe along the luminance normal must stay below 1/1.5 of the pixel's
// own chroma. A ring test cannot make this distinction: at the rim of a small saturated
// object one side of the ring IS the object's interior, so the "ridge" test passes and the
// whole contour gets neutralised. Measured on the blossom frame that cost 11.1% of the
// flowers' chroma while touching 2.96% of the frame; with this test the fringe removal on
// DSC02729 is unchanged and the blossom loss drops to 2.3%.
const float DEFRINGE_SIDE_RATIO = 2.00;

float chroma_of(vec3 c) {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

vec3 digital_display_comp(vec3 x) {
    vec3 xn = max(x - vec3(DIG_TOE), vec3(0.0)) / (1.0 - DIG_TOE);
    vec3 t = clamp((xn - vec3(DIG_KNEE)) / (DIG_WHITE - DIG_KNEE), 0.0, 1.0);
    vec3 shoulder = vec3(DIG_KNEE) + (DIG_WHITE_COMP - DIG_KNEE) * (t + t * t - t * t * t);
    vec3 tail = vec3(DIG_WHITE_COMP)
              + (1.0 - DIG_WHITE_COMP) * (vec3(1.0) - exp(-max(xn - vec3(DIG_WHITE), vec3(0.0)) / DIG_TAIL));
    // mix(a, b, t) returns b when t == 1, so the branch that must apply BELOW the knee (xn
    // itself) has to be the second argument. Swapping these put shoulder(0) = 0.18 on every
    // sub-knee pixel, which displayed as a flat 46% grey: the Before panel and the no-film
    // bypass both washed out while the export (numpy side, correct) stayed clean.
    return mix(mix(shoulder, tail, step(DIG_WHITE, xn)), xn, step(xn, vec3(DIG_KNEE)));
}

// Accurate Linear ProPhoto RGB to Display sRGB color space conversion
vec3 prophoto_to_srgb(vec3 c) {
    vec3 lin;
    lin.r =  2.03649172 * c.r - 0.73759065 * c.g - 0.29925987 * c.b;
    lin.g = -0.22571798 * c.r + 1.22317653 * c.g + 0.00272522 * c.b;
    lin.b = -0.01054513 * c.r - 0.13487985 * c.g + 1.14521015 * c.b;
    lin = digital_display_comp(max(vec3(0.0), lin));
    vec3 higher = 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055;
    vec3 lower = lin * 12.92;
    vec3 cutoff = step(vec3(0.0031308), lin);
    return clamp(mix(lower, higher, cutoff), 0.0, 1.0);
}

// Log Shaper: maps scene-linear ProPhoto [0.0, 8.0] into quasi-log space [0.0, 1.0]
const float SHAPER_C = 60.0;
const float SHAPER_INV_LOG = 0.16192058; // 1.0 / log(1.0 + 60.0 * 8.0)

vec3 linear_to_shaper(vec3 lin) {
    vec3 c = max(vec3(0.0), lin);
    return clamp(log(vec3(1.0) + c * SHAPER_C) * SHAPER_INV_LOG, 0.0, 1.0);
}

// Filmic soft highlight roll-off (smooth C1 shoulder compression above threshold)
vec3 filmic_shoulder(vec3 c, float threshold) {
    vec3 over = max(vec3(0.0), c - threshold);
    float span = 1.0 - threshold;
    return min(c, vec3(threshold)) + span * (over / (span + over));
}

// Mean nearest-neighbour spacing of the 32-tap Vogel set, in units of the disc
// radius. The mipmap LOD must blur at least this much or adjacent taps stop
// touching and the disc falls apart into a visible dot lattice.
const float DIFF_TAP_SPACING = 0.298;

// 16-tap isotropic Vogel spiral disc offsets
const vec2 VOGEL_TAPS[16] = vec2[](
    vec2( 0.17066,  0.03847),
    vec2(-0.21957,  0.22227),
    vec2( 0.07168, -0.38914),
    vec2( 0.21175,  0.42065),
    vec2(-0.48512, -0.21272),
    vec2( 0.54011, -0.19827),
    vec2(-0.30154,  0.56306),
    vec2(-0.13454, -0.67205),
    vec2( 0.57393,  0.43961),
    vec2(-0.76019,  0.03043),
    vec2( 0.55180, -0.56943),
    vec2(-0.06820,  0.84478),
    vec2(-0.52458, -0.71615),
    vec2( 0.88410,  0.25209),
    vec2(-0.80665,  0.46323),
    vec2( 0.32353, -0.93043)
);

// 8-point uniform ring, used by the halation/scatter convolutions. The Vogel
// spirals above are scaled to fill a *disc* (radius 0 -> 1), which means every tap
// sits at a different radius -- fine for a low-pass bloom, wrong for a Gaussian,
// whose kernel is stratified by radius. Sampling a Gaussian with a disc-filling
// spiral puts almost no taps in the core and over-weights the outer edge, which is
// what made the halo read as a hollow star instead of a disc.
const vec2 RING8[8] = vec2[](
    vec2( 1.00000,  0.00000),
    vec2( 0.70711,  0.70711),
    vec2( 0.00000,  1.00000),
    vec2(-0.70711,  0.70711),
    vec2(-1.00000,  0.00000),
    vec2(-0.70711, -0.70711),
    vec2( 0.00000, -1.00000),
    vec2( 0.70711, -0.70711)
);
// Same ring, rotated 22.5 degrees. Interleaving the two rings doubles the angular
// resolution without doubling the tap count per ring, so the 16-gon approximation
// of the circle has its corners offset by half a facet.
const vec2 RING8R[8] = vec2[](
    vec2( 0.92388,  0.38268),
    vec2( 0.38268,  0.92388),
    vec2(-0.38268,  0.92388),
    vec2(-0.92388,  0.38268),
    vec2(-0.92388, -0.38268),
    vec2(-0.38268, -0.92388),
    vec2( 0.38268, -0.92388),
    vec2( 0.92388, -0.38268)
);


// ~45 px radius, so the mean tap separation is ~18 px while the mipmap LOD
// supplied only ~7 px of pre-filtering. The 11 px of empty space between taps
// showed up as a visibly regular dot/grid pattern over flat bright areas
// (sky). 32 taps halve the separation; LOD is now derived from the actual
// spacing (see DIFF_LOD_K below) so adjacent taps always overlap.
const vec2 VOGEL_TAPS32[32] = vec2[](
    vec2(+0.12500, +0.00000), vec2(-0.15965, +0.14625),
    vec2(+0.02444, -0.27844), vec2(+0.20122, +0.26246),
    vec2(-0.36927, -0.06532), vec2(+0.34980, -0.22252),
    vec2(-0.11700, +0.43524), vec2(-0.22314, -0.42963),
    vec2(+0.48412, +0.17680), vec2(-0.50364, +0.20790),
    vec2(+0.24279, -0.51882), vec2(+0.17941, +0.57200),
    vec2(-0.54076, -0.31338), vec2(+0.63437, -0.13946),
    vec2(-0.38715, +0.55068), vec2(-0.08944, -0.69020),
    vec2(+0.54907, +0.46276), vec2(-0.73888, +0.03055),
    vec2(+0.53896, -0.53633), vec2(-0.03606, +0.77979),
    vec2(-0.51282, -0.61453), vec2(+0.81236, +0.10930),
    vec2(-0.68831, +0.47891), vec2(+0.18809, -0.83606),
    vec2(+0.43503, +0.75919), vec2(-0.85045, -0.27132),
    vec2(+0.82610, -0.38168), vec2(-0.35789, +0.85516),
    vec2(-0.31941, -0.88803), vec2(+0.84991, +0.44669),
    vec2(-0.94403, +0.24884), vec2(+0.53660, -0.83453)
);

// Exposure and luminance-preserving white balance, folded into one gain vector on the host
// (core.darkroom_math.scene_gain). This function runs once per tap inside the diffusion
// loop - 32 coalesced evaluations per pixel - and the WB solve (one log2, three exp2, a
// divide and a dot) is tap-independent, so it is lifted out and uploaded as u_scene_gain.
uniform vec3 u_scene_gain;

// Highlights / shadows weight calibration - must stay in sync with core/darkroom_math.py.
// Highlights key off max(R,G,B): over-exposure is a per-channel event, so a saturated neon or
// tail light must be reachable, and keying off luminance left pure blue at weight 0.0.
// The top of the range sits on the Clip decode ceiling (q = 1.0), not 4 stops above grey,
// which is why the slider could never exceed 67% authority before.
// A wide ramp, not a narrow one: foot on 18% grey and full weight on the clip point. A 0.97-
// stop ramp turned one continuous sky into two flat plateaus (42 levels of shift on the bright
// side, 3.6 on the dark side) which reads as a hard cutoff. Zero slope at the foot keeps grey
// and below untouched.
const float HS_HL_LO = 0.0;      // log2(q / 0.18): q = 0.18
const float HS_HL_SPAN = 2.4709; // full weight at log2(1.0 / 0.18)
// Shadows are a BELL, not a ramp: a ramp lifts the blacks along with the shadows and the frame
// turns to grey fog (measured: 2nd percentile 5.9 -> 30, midtones +51 levels against +27 in the
// shadows). t*(1-t)*4 is zero at 18% grey and at -6 stops, peaking at -3 stops (sRGB ~50).
const float HS_SH_SPAN = 6.0;
// Stops per slider unit at full weight. Highlights are deliberately softer than shadows: see
// the note in core/darkroom_math.py.
const float HS_HL_STOPS = 0.01;
const float HS_SH_STOPS = 0.015;

// Computes scene-linear radiance accounting for exposure, white balance tint, and highlights/shadows shaping
vec3 compute_scene_radiance(vec3 raw_c) {
    // Pre-rendered path (ui/gl_halation.py): the texture already holds scene radiance with
    // exposure, WB and HS shaping baked in, so applying the gain again would square it.
    if (u_scene_prepared > 0.5) return raw_c;
    vec3 col = raw_c * u_scene_gain;

    if (abs(u_highlights) > 0.001 || abs(u_shadows) > 0.001) {
        float hi = max(col.r, max(col.g, col.b));
        // ProPhoto luminance: the same row the white-balance solve normalises with. Using the
        // Rec.709 row here put the blue coefficient off by 840x against the WB gains.
        float lum = dot(col, vec3(0.2880, 0.7119, 0.0001));

        float t_hl = clamp((log2(max(hi, 1e-9) / 0.18) - HS_HL_LO) / HS_HL_SPAN, 0.0, 1.0);
        float w_hl = t_hl * t_hl * (3.0 - 2.0 * t_hl);
        // Bell: zero at 18% grey and at -5 stops, so pure black is neither lifted nor is the
        // noise floor amplified (log2 saturates to t = 1, a zero weight).
        float t_sh = clamp(-log2(max(lum, 1e-9) / 0.18) / HS_SH_SPAN, 0.0, 1.0);
        float w_sh = t_sh * (1.0 - t_sh) * 4.0;

        float delta_z = (u_shadows * HS_SH_STOPS) * w_sh + (u_highlights * HS_HL_STOPS) * w_hl;
        // The old form re-aimed luminance at 0.18 * exp2(z + delta_z) and divided by the
        // luminance, which is algebraically exp2(delta_z) - one exp2 and one mul instead of
        // two exp2 plus a divide.
        col *= exp2(delta_z);
    }
    return col;
}

// One PSF group of the diffusion filter: an exponential disc truncated at 8
// mean-free-paths, sampled on the 32-tap Vogel set with a mipmap LOD matched to
// the tap spacing. This is the shader-side (K_group * E) term. The radius is
// physical (converted from micrometres by the caller), so the same scene blurs
// identically at any preview resolution.
vec3 diffusion_disc(vec2 sample_uv, vec2 radius_uv) {
    vec2 tex_sz = max(vec2(100.0), vec2(textureSize(u_image, 0)));
    float spacing_px = DIFF_TAP_SPACING * max(radius_uv.x * tex_sz.x, radius_uv.y * tex_sz.y);
    float lod_base = clamp(log2(max(1.0, spacing_px)), 0.0, 9.0);
    vec3 acc = vec3(0.0);
    float w_sum = 0.0;
    for (int i = 0; i < 32; i++) {
        vec2 off = VOGEL_TAPS32[i];
        float r_norm = length(off);
        vec2 tap_uv = clamp(sample_uv + off * radius_uv, 0.0, 1.0);
        float tap_lod = clamp(lod_base + r_norm * 0.35, 0.0, 9.0);
        vec3 tap_rgb = compute_scene_radiance(textureLod(u_image, tap_uv, tap_lod).rgb);
        // exp(-r/lambda), with the disc edge at r = 8 lambda
        float w = exp(-u_diff_cutoff * r_norm);
        acc += tap_rgb * w;
        w_sum += w;
    }
    return acc / max(0.0001, w_sum);
}

// {{FILM_GRAIN}}

void main() {
    if (u_has_image == 0) {
        fragColor = vec4(u_bg_color, 1.0);
        return;
    }
    
    float eff_screen_aspect = u_screen_aspect;
    vec2 cur_coord = v_texcoord;
    bool is_left = (v_texcoord.x < 0.5);

    if (u_view_mode == 2) {
        // Mode 2: Global side-by-side comparison (Both complete original and edited images)
        float vp_w = (u_viewport_size.x > 1.0) ? u_viewport_size.x : (1.0 / max(0.00001, fwidth(v_texcoord.x)));
        float dist_px = abs(v_texcoord.x - 0.5) * vp_w;
        // Delicate hairline divider (solid brand amber #f59e0b, 1px~1.5px, no black edges)
        if (dist_px <= 0.85) {
            fragColor = vec4(0.961, 0.620, 0.043, 1.0); // #f59e0b
            return;
        }

        eff_screen_aspect = u_screen_aspect * 0.5;
        if (is_left) {
            cur_coord.x = v_texcoord.x * 2.0;
        } else {
            cur_coord.x = (v_texcoord.x - 0.5) * 2.0;
        }
    }

    vec2 centered = (cur_coord - 0.5) * 2.0;
    
    vec2 quad_scale;
    if (eff_screen_aspect > u_image_aspect) {
        quad_scale = vec2(u_image_aspect / eff_screen_aspect, 1.0);
    } else {
        quad_scale = vec2(1.0, eff_screen_aspect / u_image_aspect);
    }
    
    vec2 img_uv = (centered - u_pan) / (quad_scale * u_zoom);
    img_uv = img_uv * 0.5 + 0.5;

    // Uniform derivatives evaluated across 2x2 fragment quads BEFORE any conditional returns
    vec2 d_img_dx = dFdx(img_uv);
    vec2 d_img_dy = dFdy(img_uv);

    // Discard any fragment outside the active viewport photo quad (only in normal non-crop-editing view)
    if (u_is_crop_editing != 1 && (img_uv.x < 0.0 || img_uv.x > 1.0 || img_uv.y < 0.0 || img_uv.y > 1.0)) {
        fragColor = vec4(u_bg_color, 1.0);
        return;
    }

    // Y-flip and Crop & Straighten transform
    vec2 sample_uv;
    vec2 d_sample_dx;
    vec2 d_sample_dy;

    if (u_crop_enabled == 1) {
        vec2 img_center = vec2(0.5, 0.5);
        if (u_is_crop_editing == 1) {
            // Live interactive preview in crop tool: Rotate full image around image center (0.5, 0.5)
            vec2 raw_uv = vec2(img_uv.x, 1.0 - img_uv.y);
            d_sample_dx = vec2(d_img_dx.x, -d_img_dx.y);
            d_sample_dy = vec2(d_img_dy.x, -d_img_dy.y);

            // Apply flip
            if (u_crop_flip.x > 0.5) {
                raw_uv.x = 1.0 - raw_uv.x;
                d_sample_dx.x = -d_sample_dx.x;
                d_sample_dy.x = -d_sample_dy.x;
            }
            if (u_crop_flip.y > 0.5) {
                raw_uv.y = 1.0 - raw_uv.y;
                d_sample_dx.y = -d_sample_dx.y;
                d_sample_dy.y = -d_sample_dy.y;
            }

            // Apply 90-degree steps
            vec2 rot_uv = raw_uv;
            if (u_crop_rot90 == 90) {
                rot_uv = vec2(raw_uv.y, 1.0 - raw_uv.x);
                d_sample_dx = vec2(d_sample_dx.y, -d_sample_dx.x);
                d_sample_dy = vec2(d_sample_dy.y, -d_sample_dy.x);
            } else if (u_crop_rot90 == 180) {
                rot_uv = vec2(1.0 - raw_uv.x, 1.0 - raw_uv.y);
                d_sample_dx = vec2(-d_sample_dx.x, -d_sample_dx.y);
                d_sample_dy = vec2(-d_sample_dy.x, -d_sample_dy.y);
            } else if (u_crop_rot90 == 270) {
                rot_uv = vec2(1.0 - raw_uv.y, raw_uv.x);
                d_sample_dx = vec2(-d_sample_dx.y, d_sample_dx.x);
                d_sample_dy = vec2(-d_sample_dy.y, d_sample_dy.x);
            }

            // Fine straighten angle
            if (abs(u_crop_angle) > 0.0001) {
                float eff_aspect = (u_crop_rot90 == 90 || u_crop_rot90 == 270) ? (1.0 / u_native_aspect) : u_native_aspect;
                vec2 p = (rot_uv - img_center) * vec2(eff_aspect, 1.0);
                float ca = cos(u_crop_angle);
                float sa = sin(-u_crop_angle);
                vec2 p_rot = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
                sample_uv = img_center + vec2(p_rot.x / eff_aspect, p_rot.y);

                vec2 px = d_sample_dx * vec2(eff_aspect, 1.0);
                vec2 py = d_sample_dy * vec2(eff_aspect, 1.0);
                d_sample_dx = vec2((px.x * ca - px.y * sa) / eff_aspect, px.x * sa + px.y * ca);
                d_sample_dy = vec2((py.x * ca - py.y * sa) / eff_aspect, py.x * sa + py.y * ca);
            } else {
                sample_uv = rot_uv;
            }
        } else {
            // Normal view: Display the cropped area upright, filling the quad seamlessly
            vec2 crop_center = u_crop_rect.xy + u_crop_rect.zw * 0.5;
            vec2 delta_p = vec2(img_uv.x - 0.5, (1.0 - img_uv.y) - 0.5) * u_crop_rect.zw;
            vec2 unrotated_p = crop_center + delta_p;
            d_sample_dx = vec2(d_img_dx.x, -d_img_dx.y) * u_crop_rect.zw;
            d_sample_dy = vec2(d_img_dy.x, -d_img_dy.y) * u_crop_rect.zw;

            // Apply flip
            if (u_crop_flip.x > 0.5) {
                unrotated_p.x = 1.0 - unrotated_p.x;
                d_sample_dx.x = -d_sample_dx.x;
                d_sample_dy.x = -d_sample_dy.x;
            }
            if (u_crop_flip.y > 0.5) {
                unrotated_p.y = 1.0 - unrotated_p.y;
                d_sample_dx.y = -d_sample_dx.y;
                d_sample_dy.y = -d_sample_dy.y;
            }

            // Apply 90-degree steps
            vec2 rot_p = unrotated_p;
            if (u_crop_rot90 == 90) {
                rot_p = vec2(unrotated_p.y, 1.0 - unrotated_p.x);
                d_sample_dx = vec2(d_sample_dx.y, -d_sample_dx.x);
                d_sample_dy = vec2(d_sample_dy.y, -d_sample_dy.x);
            } else if (u_crop_rot90 == 180) {
                rot_p = vec2(1.0 - unrotated_p.x, 1.0 - unrotated_p.y);
                d_sample_dx = vec2(-d_sample_dx.x, -d_sample_dx.y);
                d_sample_dy = vec2(-d_sample_dy.x, -d_sample_dy.y);
            } else if (u_crop_rot90 == 270) {
                rot_p = vec2(1.0 - unrotated_p.y, unrotated_p.x);
                d_sample_dx = vec2(-d_sample_dx.y, d_sample_dx.x);
                d_sample_dy = vec2(-d_sample_dy.y, d_sample_dy.x);
            }

            if (abs(u_crop_angle) > 0.0001) {
                float eff_aspect = (u_crop_rot90 == 90 || u_crop_rot90 == 270) ? (1.0 / u_native_aspect) : u_native_aspect;
                vec2 p = (rot_p - img_center) * vec2(eff_aspect, 1.0);
                float ca = cos(u_crop_angle);
                float sa = sin(-u_crop_angle);
                vec2 p_rot = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
                sample_uv = img_center + vec2(p_rot.x / eff_aspect, p_rot.y);

                vec2 px = d_sample_dx * vec2(eff_aspect, 1.0);
                vec2 py = d_sample_dy * vec2(eff_aspect, 1.0);
                d_sample_dx = vec2((px.x * ca - px.y * sa) / eff_aspect, px.x * sa + px.y * ca);
                d_sample_dy = vec2((py.x * ca - py.y * sa) / eff_aspect, py.x * sa + py.y * ca);
            } else {
                sample_uv = rot_p;
            }
        }
    } else {
        sample_uv = vec2(img_uv.x, 1.0 - img_uv.y);
        d_sample_dx = vec2(d_img_dx.x, -d_img_dx.y);
        d_sample_dy = vec2(d_img_dy.x, -d_img_dy.y);
    }

    // Lens vignetting is a property of the sensor position, not of the warped sample, so it
    // is evaluated on the pre-warp coordinate (matches core/lens_warp.py).
    vec2 pre_dist_uv = sample_uv;
    vec2 pre_dist_dx = d_sample_dx;
    vec2 pre_dist_dy = d_sample_dy;

    // Apply Lens Distortion Warp if enabled
    if (u_lens_enabled == 1 && (abs(u_lens_distort.x) > 0.00001 || abs(u_lens_distort.y) > 0.00001 || abs(u_lens_distort.z) > 0.00001)) {
        vec2 tex_dim = max(vec2(1.0), vec2(textureSize(u_image, 0)));
        float aspect = tex_dim.x / tex_dim.y;
        vec2 delta = sample_uv - 0.5;
        // Radial distance normalized by semi-diagonal so r=1.0 at image corners (Adobe ACM standard)
        float max_r2 = 0.25 * (aspect * aspect + 1.0);
        float r2 = (delta.x * aspect * delta.x * aspect + delta.y * delta.y) / max_r2;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float r_dist = 1.0 + u_lens_distort.x * r2 + u_lens_distort.y * r4 + u_lens_distort.z * r6;

        // Auto-scale factor to prevent edge clamping / black borders. Applied for both
        // directions: only barrel correction used to be rescaled, so pincushion correction
        // pushed the corners past the frame and the clamp turned them into smeared streaks.
        float corner_k = u_lens_distort.x + u_lens_distort.y + u_lens_distort.z;
        float scale = abs(u_lens_dist_scale) > 1e-9
                    ? u_lens_dist_scale
                    : (u_lens_fill_scale > 1e-6
                        ? u_lens_fill_scale
                        : 1.0 / max(0.05, 1.0 + corner_k));
        sample_uv = 0.5 + (delta * r_dist) * scale;

        // Local Jacobian of the warp. The sampling density changes with radius, so the mip
        // level has to follow it; leaving the gradients untouched made a strong correction
        // alias at the centre and over-blur at the edges.
        float dK = u_lens_distort.x + 2.0 * u_lens_distort.y * r2 + 3.0 * u_lens_distort.z * r4;
        vec2 grad_r2 = vec2(2.0 * aspect * aspect * delta.x, 2.0 * delta.y) / max_r2;
        float j00 = scale * (r_dist + delta.x * grad_r2.x * dK);
        float j01 = scale * (delta.x * grad_r2.y * dK);
        float j10 = scale * (delta.y * grad_r2.x * dK);
        float j11 = scale * (r_dist + delta.y * grad_r2.y * dK);
        mat2 warp_jacobian = mat2(j00, j10, j01, j11);
        d_sample_dx = warp_jacobian * d_sample_dx;
        d_sample_dy = warp_jacobian * d_sample_dy;
    }
    
    // Clamping & Letterbox background
    vec2 tex_dim = max(vec2(1.0), vec2(textureSize(u_image, 0)));
    vec2 half_texel = 0.5 / tex_dim;
    if (u_crop_enabled == 1 && u_is_crop_editing == 1) {
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            fragColor = vec4(u_bg_color, 1.0);
            return;
        }
    } else {
        sample_uv = clamp(sample_uv, half_texel, vec2(1.0) - half_texel);
    }
    
    vec3 orig_rgb = textureGrad(u_image, sample_uv, d_sample_dx, d_sample_dy).rgb;

    // 1. Lateral CA Radial Shift (only when calibrated non-zero params provided)
    if (u_lens_ca_enabled == 1 &&
        (abs(u_lens_ca_scale_r) + abs(u_lens_ca_scale_b) +
         abs(u_lens_ca_r.x) + abs(u_lens_ca_r.y) + abs(u_lens_ca_r.z) +
         abs(u_lens_ca_b.x) + abs(u_lens_ca_b.y) + abs(u_lens_ca_b.z) > 0.00001)) {
        // Radius in the same corner-normalised space the distortion term uses, so the
        // profile's r = R/f coefficients transfer without a second normalisation.
        float asp_ca = tex_dim.x / tex_dim.y;
        float max_r2_ca = 0.25 * (asp_ca * asp_ca + 1.0);
        vec2 lp_ca = (sample_uv - 0.5);
        float r2_ca = (lp_ca.x * asp_ca * lp_ca.x * asp_ca + lp_ca.y * lp_ca.y) / max_r2_ca;
        float r4_ca = r2_ca * r2_ca;
        float r6_ca = r4_ca * r2_ca;
        float scale_r = 1.0 + u_lens_ca_scale_r
                        + u_lens_ca_r.x * r2_ca + u_lens_ca_r.y * r4_ca + u_lens_ca_r.z * r6_ca;
        float scale_b = 1.0 + u_lens_ca_scale_b
                        + u_lens_ca_b.x * r2_ca + u_lens_ca_b.y * r4_ca + u_lens_ca_b.z * r6_ca;
        vec2 uv_r = clamp(0.5 + lp_ca * scale_r, half_texel, vec2(1.0) - half_texel);
        vec2 uv_b = clamp(0.5 + lp_ca * scale_b, half_texel, vec2(1.0) - half_texel);
        orig_rgb.r = textureGrad(u_image, uv_r, d_sample_dx, d_sample_dy).r;
        orig_rgb.b = textureGrad(u_image, uv_b, d_sample_dx, d_sample_dy).b;
    }

    // 2. Edge defringing. A colour fringe is *additive* colour laid on top of a surface, so the
    //    only stable signature is a CHROMA RIDGE: the pixel is more chromatic than the ring
    //    around it. Everything a previous version of this block did wrong follows from missing
    //    that test —
    //      * a 3x3/4px/8px mean is filled by the fringe itself on a 1-2 px band, so the excess
    //        never opened up (measured 18% removal at every band width),
    //      * a wide reference blends across the edge, which made the "repair" pull legit
    //        colour toward the average: petals lost their own edges (measured -7.7% chroma on
    //        the blossom test frame, 15% of pixels touched),
    //      * and it fired on chroma VALLEYS too. Measured on the branch test frame, 77% of the
    //        strong "purple rim" pixels are valleys — they are neutral anti-aliasing between a
    //        bright sky and a dark branch, not colour that was added.
    //    The repair is a neutralisation at the pixel's own luminance, which cannot import a tint
    //    from anywhere else.
    if (u_lens_ca_enabled == 1) {
        float r = orig_rgb.r;
        float g = orig_rgb.g;
        float b = orig_rgb.b;
        float l0 = dot(orig_rgb, vec3(0.299, 0.587, 0.114));
        float mx0 = max(1e-4, max(r, max(g, b)));
        float ch0 = mx0 - min(r, min(g, b));

        float magenta = max(0.0, min(r, b) - g);          // R and B above G
        float cyan = max(0.0, b - max(r, g));             // B above both
        float green = max(0.0, g - max(r, b));            // G above both
        float red = max(0.0, r - max(g, b));              // R above both
        float hue = clamp((magenta + cyan + green + red) / max(ch0, 1e-6), 0.0, 1.0);

        if (ch0 / mx0 > 0.08 && hue > DEFRINGE_HUE_MIN) {
            vec2 ts = 2.0 * (1.0 / tex_dim);
            vec3 n1 = textureGrad(u_image, sample_uv + vec2(ts.x, 0.0), d_sample_dx, d_sample_dy).rgb;
            vec3 n2 = textureGrad(u_image, sample_uv - vec2(ts.x, 0.0), d_sample_dx, d_sample_dy).rgb;
            vec3 n3 = textureGrad(u_image, sample_uv + vec2(0.0, ts.y), d_sample_dx, d_sample_dy).rgb;
            vec3 n4 = textureGrad(u_image, sample_uv - vec2(0.0, ts.y), d_sample_dx, d_sample_dy).rgb;

            float c1 = max(n1.r, max(n1.g, n1.b)) - min(n1.r, min(n1.g, n1.b));
            float c2 = max(n2.r, max(n2.g, n2.b)) - min(n2.r, min(n2.g, n2.b));
            float c3 = max(n3.r, max(n3.g, n3.b)) - min(n3.r, min(n3.g, n3.b));
            float c4 = max(n4.r, max(n4.g, n4.b)) - min(n4.r, min(n4.g, n4.b));
            float c_lo = min(min(c1, c2), min(c3, c4));
            float c_hi = max(max(c1, c2), max(c3, c4));

            float l1 = dot(n1, vec3(0.299, 0.587, 0.114));
            float l2 = dot(n2, vec3(0.299, 0.587, 0.114));
            float l3 = dot(n3, vec3(0.299, 0.587, 0.114));
            float l4 = dot(n4, vec3(0.299, 0.587, 0.114));
            float l_lo = min(min(l1, l2), min(l3, l4));
            float l_hi = max(max(l1, l2), max(l3, l4));

            float excess = clamp((ch0 - c_lo) / max(ch0, 1e-6), 0.0, 1.0);
            float contrast = (l_hi - l_lo) / max(l0, 1e-3);

            // Ridge: this pixel must be at least as chromatic as its ring. A petal crease and an
            // anti-aliased branch edge are both valleys and stay untouched.
            //
            // The ring test alone accepts a STEP as well: at the rim of a small saturated
            // object (the blossom frame's flowers are ~5 px across) one side of the ring is the
            // object's own interior, so "as chromatic as the ring" holds and the whole contour
            // is neutralised. A fringe is a band with low chroma on BOTH sides, so the last test
            // taps the chroma at ±2 and ±3 px along the luminance normal and requires both
            // sides to stay below ch0 / DEFRINGE_SIDE_RATIO.
            if (excess > DEFRINGE_EXCESS_FLOOR && ch0 >= c_hi * DEFRINGE_RIDGE_TOL
                    && contrast > DEFRINGE_CONTRAST_MIN) {
                vec2 grad_l = vec2(l1 - l2, l3 - l4);
                float gm = length(grad_l);
                // No normal means no way to ask the question — and leaving side at 0 would
                // sample nothing meaningful. A fringe narrower than the ±2 px probe is exactly
                // this case (an isolated 1 px band is symmetric about its own centre), so it
                // passes at full strength.
                float ramp = 1.0;
                if (gm > 1e-6) {
                    vec2 ng = grad_l / gm;
                    vec2 p1 = 1.0 / tex_dim;
                    vec2 o2 = ng * (2.0 * p1);
                    vec2 o3 = ng * (3.0 * p1);
                    float side = 0.0;
                    side = max(side, chroma_of(textureLod(u_image, sample_uv + o2, 0.0).rgb));
                    side = max(side, chroma_of(textureLod(u_image, sample_uv - o2, 0.0).rgb));
                    side = max(side, chroma_of(textureLod(u_image, sample_uv + o3, 0.0).rgb));
                    side = max(side, chroma_of(textureLod(u_image, sample_uv - o3, 0.0).rgb));
                    // Smooth ramp, not a hard cut: a fringe sitting exactly at the threshold
                    // would otherwise flip between pass and fail on a rounding difference
                    // between the GPU and CPU tap positions.
                    ramp = clamp((ch0 / max(side, 1e-6) - 1.0) / (DEFRINGE_SIDE_RATIO - 1.0),
                                 0.0, 1.0);
                    ramp = ramp * ramp * (3.0 - 2.0 * ramp);
                }
                float w = clamp(excess * DEFRINGE_STRENGTH, 0.0, 1.0)
                        * clamp(hue / DEFRINGE_HUE_MIN, 0.0, 1.0) * ramp;
                orig_rgb = mix(orig_rgb, vec3(l0), w);
            }
        }
    }

    // Apply Lens Vignette compensation/darkening if enabled
    if (u_lens_enabled == 1 && abs(u_lens_vignette.x) > 0.001) {
        float r_norm = length((pre_dist_uv - 0.5) * 2.0) / 1.41421356;
        float v_exp = mix(1.2, 4.0, 1.0 - clamp(u_lens_vignette.y / 100.0, 0.1, 0.9));
        float vig_gain = max(0.0, 1.0 + u_lens_vignette.x * pow(clamp(r_norm, 0.0, 1.0), v_exp));
        orig_rgb *= vig_gain;
    }

    // Accurate untouched base display color (Before / 原片)
    // raw_preview deliberately stores pure scene radiance with base_ev NOT pre-multiplied (that
    // keeps re-imports idempotent), so the untouched view has to apply it here. Without it the
    // Before side sits up to ~1 stop hot and flat next to the developed side: the sky clips to
    // white and the shadows lift, which reads as a grey milky wash rather than a neutral baseline.
    vec3 before_lin = orig_rgb * exp2(u_base_ev);
    vec3 before_color = (u_is_bypass == 2) ? clamp(orig_rgb, 0.0, 1.0) : clamp(prophoto_to_srgb(before_lin), 0.0, 1.0);

    // -------------------------------------------------------------
    // Compare Modes (Must execute BEFORE bypass early return!)
    // -------------------------------------------------------------
    if (u_view_mode == 1) {
        // Mode 1: Split screen drag compare (Before / After)
        float vp_w = (u_viewport_size.x > 1.0) ? u_viewport_size.x : (1.0 / max(0.00001, fwidth(v_texcoord.x)));
        float dist_px = abs(v_texcoord.x - u_split_x) * vp_w;

        // Delicate hairline split divider (solid brand amber #f59e0b, 1px~1.5px, no black edges)
        if (dist_px <= 0.85) {
            fragColor = vec4(0.961, 0.620, 0.043, 1.0); // #f59e0b
            return;
        }

        // Left region: Strictly display untouched original image (Before / 原片)
        if (v_texcoord.x < u_split_x) {
            fragColor = vec4(before_color, 1.0);
            return;
        }
        // Right region: Continue forward into developed pipeline (After / 已修改)
    } else if (u_view_mode == 2 && is_left) {
        // Mode 2 Left viewport: Full complete original (Before / 原片)
        fragColor = vec4(before_color, 1.0);
        return;
    } else if (u_view_mode == 3) {
        // Mode 3: Temporary Full Original Before Compare (Long Press)
        fragColor = vec4(before_color, 1.0);
        return;
    }

    // -------------------------------------------------------------
    // After Pipeline (Applied to single view mode or split right region)
    // -------------------------------------------------------------
    // 0. Bypass Mode:
    // 2: Direct sRGB passthrough (0ms placeholder / embedded preview from filmstrip or JPEG)
    if (u_is_bypass == 2) {
        fragColor = vec4(clamp(orig_rgb, 0.0, 1.0), 1.0);
        return;
    }

    // 1: Pure Bypass Mode: When no film is selected, pipeline continues to apply exposure & optics
    // and displays through filmic sRGB curve.
    if (u_is_bypass == 1) {
        // Handled through normal radiance & filmic transform pipeline
    }
    
    // 1 & 2. Exposure, White Balance, and Highlights & Shadows Tone Shaping
    vec3 linear_rgb = compute_scene_radiance(orig_rgb);
    
    // 3. Optical Diffusion Filter (Physical Convolution Bloom with Vogel Spiral & Trilinear LOD)
    int eff_diff_type = u_diffusion_type;
    if (eff_diff_type == 0 && u_diffusion_strength > 0.001) {
        eff_diff_type = 1; // Default to Black Pro-Mist if strength > 0
    }
    if (eff_diff_type > 0 && u_diffusion_ps > 0.0005 && u_scene_prepared < 0.5) {
        // u_scene_prepared: the GLHalationChain already evaluated the diffusion plan
        // (core/diffusion_model.py) upstream; running this tap-based bloom on top of
        // its output would apply the filter twice.
        // The PSF is the CPU model's own three-group split — core (stays at the
        // pixel), halo and bloom — each an exponential disc truncated at 8 mean
        // free paths. The radii arrive in micrometres from the same table the
        // export uses, so a given filter blurs the same physical distance at any
        // preview resolution. The stop changes only p_s (density), never radius;
        // scaling radius with strength is what made 2 stops read as a different
        // filter instead of a denser one.
        vec2 tex_sz = max(vec2(100.0), vec2(textureSize(u_image, 0)));
        float max_dim = max(tex_sz.x, tex_sz.y);
        float um_per_px = max(1000.0, u_film_width_um) / max_dim;
        vec2 halo_r  = vec2(u_diff_halo_um  / um_per_px / tex_sz.x,
                            u_diff_halo_um  / um_per_px / tex_sz.y);
        vec2 bloom_r = vec2(u_diff_bloom_um / um_per_px / tex_sz.x,
                            u_diff_bloom_um / um_per_px / tex_sz.y);

        vec3 halo_avg  = diffusion_disc(sample_uv, halo_r);
        vec3 bloom_avg = diffusion_disc(sample_uv, bloom_r);

        // Warmth is a hue shift in the scattered light, normalised to unit
        // luminance so it cannot change scene brightness.
        float eff_warmth = clamp(u_diff_warmth_base + u_diffusion_warmth, -1.5, 1.5);
        vec3 diff_tint = vec3(
            1.0 + eff_warmth * 0.35,
            1.0 + eff_warmth * 0.08,
            1.0 - eff_warmth * 0.35
        );
        diff_tint /= max(0.2, dot(diff_tint, vec3(0.2126, 0.7152, 0.0722)));

        // K * E: the three groups combined with the CPU's own weights (sum = 1).
        // Core is a delta, so it keeps the pixel's own radiance.
        vec3 k_blur = u_diff_core_w * linear_rgb
                    + u_diff_halo_w * halo_avg  * diff_tint
                    + u_diff_bloom_w * bloom_avg * diff_tint;

        // Energy-conserving composite, literally the CPU path's formula:
        //     E_out = (1 - p_s) * E_in + p_s * (K * E_in)
        //           = E_in + p_s * (K*E - E_in)
        // In a flat field K*E == E_in, so nothing happens; next to a highlight the
        // pixel gives up p_s of its own radiance and receives p_s of the haze that
        // landed on it. Energy is preserved, so the frame's mean brightness does
        // not drift with the stop. (The old `E += bloom * strength * intensity`
        // injected every scattered photon twice and blew a quarter of the frame
        // to white at 2 stops.)
        float p_s = clamp(u_diffusion_ps, 0.0, 0.95);
        linear_rgb += p_s * (k_blur - linear_rgb);
    }

    // 3.5. Official Spektrafilm Physical Halation & In-Emulsion Scatter
    if (u_halation > 0.001 || u_scatter_amount > 0.001) {
        vec2 tex_sz_hal = max(vec2(100.0), vec2(textureSize(u_image, 0)));
        float aspect_hal = tex_sz_hal.x / tex_sz_hal.y;
        float max_dim = max(tex_sz_hal.x, tex_sz_hal.y);

        // Physical sigma scaled by 35mm film baseline (35000 um)
        float base_hal_rad = max(0.0005, (u_hal_sigma_um / max(5000.0, u_film_width_um)) * max(0.2, u_hal_spatial_scale));
        vec2 hal_spread = vec2(base_hal_rad * 2.5, base_hal_rad * 2.5 * aspect_hal);

        // 1. In-Emulsion Forward Scatter: Energy-preserving micro-diffusion
        if (u_scatter_amount > 0.001) {
            vec2 scat_spread = hal_spread * 0.15;
            vec3 scat_accum = vec3(0.0);
            float scat_w = 0.0;
            for (int i = 0; i < 8; i++) {
                vec2 offset = VOGEL_TAPS[i];
                vec2 tap_uv = clamp(sample_uv + offset * scat_spread, 0.0, 1.0);
                vec3 tap_rgb = compute_scene_radiance(textureLod(u_image, tap_uv, 0.5).rgb);
                float w = exp(-length(offset) * 2.0);
                scat_accum += tap_rgb * w;
                scat_w += w;
            }
            scat_accum /= max(0.0001, scat_w);
            float eff_s = clamp(u_scatter_amount * 0.06, 0.0, 0.5);
            linear_rgb = mix(linear_rgb, scat_accum, eff_s);
        }

        // 2. Base Back-Reflection Halation: Multi-bounce additive sum with DC energy normalization
        //    Skipped when the caller pre-rendered the halo: ui/gl_halation.py evaluates the
        //    same DC-normalised sum with a real separable Gaussian, and running the tap
        //    version on top would apply it twice.
        if (u_halation > 0.001 && u_scene_prepared < 0.5) {
            vec3 a_tot = u_hal_strength_rgb * u_halation;
            vec3 hal_accum = vec3(0.0);
            float hal_total_w = 0.0;

            for (int b = 1; b <= 3; b++) {
                float bounce_w = pow(0.5, float(b - 1));
                float bounce_rad_mult = sqrt(float(b));
                // 16-tap Vogel mean nearest-neighbour ≈ 0.42 of the unit radius,
                // and hal_spread is 2.5× the base radius, so the tap separation
                // is ~1.05 × base_hal_rad × max_dim. The LOD must match that or
                // the halo breaks into the same dot pattern as the bloom did.
                float lod_b = clamp(log2(base_hal_rad * bounce_rad_mult * max_dim * 1.05), 0.0, 8.0);

                for (int i = 0; i < 16; i++) {
                    vec2 offset = VOGEL_TAPS[i] * bounce_rad_mult;
                    float r_norm = length(offset);
                    vec2 tap_uv = clamp(sample_uv + offset * hal_spread, 0.0, 1.0);
                    vec3 tap_rgb = compute_scene_radiance(textureLod(u_image, tap_uv, lod_b).rgb);
                    // No highlight boost here: the real EV reconstruction is an image-global
                    // transform (normalised by the frame max) and the GLHalationChain owns
                    // it. A tap-level stand-in would need a max the shader cannot have;
                    // on the fallback path the boost simply does not apply.

                    float w = exp(-r_norm * 1.5) * bounce_w;
                    hal_accum += tap_rgb * w;
                    hal_total_w += w;
                }
            }
            vec3 hal_blur = hal_accum / max(0.0001, hal_total_w);

            // Official Spektrafilm physical halation DC normalization:
            linear_rgb = (linear_rgb + a_tot * hal_blur) / (vec3(1.0) + a_tot);
        }
    }

    // 3.6. Film grain: the pixel is the mean transmittance of one small patch of the stock's
    // pre-rendered film tile, so it has to run on the negative's side of the LUT. The stage is
    // ui/gl_film_grain.py's port of fotufilm's tile sampler -- the same film, the same placement,
    // the same read the CPU export makes. It reads the pre-distortion coordinate: grain belongs
    // to the film, not to the lens the film was shot through.
    if (u_grain_on == 1) {
        vec3 grain_dD = grain_field(linear_rgb, pre_dist_uv, pre_dist_dx, pre_dist_dy);
        // A diagnostic read of the stage: the density field itself, so the GPU sampler can be
        // compared against the CPU one pixel for pixel. Unreachable from the UI.
        if (u_grain_probe == 1) {
            fragColor = vec4(grain_dD * 0.5 + 0.5, 1.0);
            return;
        }
        linear_rgb *= exp(-grain_dD * 2.302585092994046);   // 10^-dD
    } else if (u_grain_probe == 1) {
        fragColor = vec4(vec3(0.5), 1.0);
        return;
    }

    // 4. Physical 3D LUT sampling with Log Shaper (Film emulsion + Paper D-max)
    vec3 film_rgb;
    if (u_has_lut == 1) {
        vec3 shaper_in = linear_to_shaper(linear_rgb);
        float lut_dim = float(textureSize(u_lut, 0).x);
        vec3 lut_coord = shaper_in * ((lut_dim - 1.0) / lut_dim) + (0.5 / lut_dim);
        film_rgb = texture(u_lut, lut_coord).rgb;
    } else {
        film_rgb = prophoto_to_srgb(linear_rgb);
    }

    // 5. DIR Coupler inhibition is already fully baked into the LUT above (film development
    //    stage). No post-LUT approximation is applied: dragging the DIR sliders rebakes the
    //    LUT at full resolution through the same code path, so preview == commit.

    // 6. Enlarger CMY filtration, print exposure & pre-flash in paper-exposure domain
    //    print = 1 / (1 + E^2.5)  ->  E' = E * 10^(-0.004 * cmy) / print_exposure + preflash
    //    (mirrors core/darkroom_math.paper_domain_adjust)
    if (any(greaterThan(abs(u_cmy), vec3(0.01))) || abs(u_print_exposure - 1.0) > 0.005 || u_preflash > 0.0005) {
        vec3 pv = clamp(film_rgb, 1e-4, 1.0 - 1e-4);
        vec3 e_paper = pow(vec3(1.0) / pv - vec3(1.0), vec3(0.4));
        e_paper = e_paper * pow(vec3(10.0), -0.004 * u_cmy) / max(0.05, u_print_exposure) + vec3(max(0.0, u_preflash));
        film_rgb = vec3(1.0) / (vec3(1.0) + pow(e_paper, vec3(2.5)));
    }

    // 7. Paper Chemistry Morph & Developer Exhaustion - delta vs. LUT-baked values only
    float morph_r = u_morph_gamma / max(0.05, u_lut_morph_gamma);
    if (abs(morph_r - 1.0) > 0.01) {
        vec3 c_m = clamp(film_rgb, 1e-4, 1.0 - 1e-4);
        vec3 odds = pow((c_m / 0.42) / ((vec3(1.0) - c_m) / 0.58), vec3(morph_r));
        film_rgb = (odds * 0.42) / (odds * 0.42 + vec3(0.58));
    }
    float exh_d = u_developer_exhaustion - u_lut_dev_exh;
    if (abs(exh_d) > 0.005) {
        vec3 c_exh = clamp(film_rgb, 0.0, 1.0);
        film_rgb = clamp(film_rgb + exh_d * c_exh * (vec3(1.0) - c_exh) * (c_exh - vec3(0.42)) * 0.35, 0.0, 1.0);
    }

    // 8. Bidirectional Black Point calibration (white-point preserving: >0 lifts shadows/fades, <0 subtracts base fog/deepens blacks)
    if (u_black_lift > 0.0001) {
        film_rgb = vec3(u_black_lift) + film_rgb * (1.0 - u_black_lift);
    } else if (u_black_lift < -0.0001) {
        float c_blk = min(0.95, -u_black_lift);
        film_rgb = max(vec3(0.0), (film_rgb - vec3(c_blk)) / (1.0 - c_blk));
    }
    fragColor = vec4(clamp(film_rgb, 0.0, 1.0), 1.0);
}
"""

# The grain stage is one self-contained block of GLSL: it lives next to the sampler it mirrors,
# and it is spliced in here so the two can never drift without the file moving.
from .gl_film_grain import GRAIN_GLSL as _GRAIN_GLSL          # noqa: E402

FRAGMENT_SHADER_SRC = FRAGMENT_SHADER_SRC.replace("// {{FILM_GRAIN}}", _GRAIN_GLSL)


CANVAS_BG_COLORS = {
    "white": (1.0, 1.0, 1.0),
    "light_gray": (0.67, 0.67, 0.67),
    "dark_gray": (0.33, 0.33, 0.33),
    "black": (0.0, 0.0, 0.0),
}

CANVAS_BG_NAMES = [
    ("白色", "white"),
    ("浅灰", "light_gray"),
    ("深灰", "dark_gray"),
    ("黑色", "black"),
]


def get_bg_color_circle_icon(key: str, is_dark: bool = True) -> QIcon:
    """Generate high-DPI solid circular swatch icon for canvas background color options."""
    fill_map = {
        "white": QColor(255, 255, 255),
        "light_gray": QColor(171, 171, 171),
        "dark_gray": QColor(84, 84, 84),
        "black": QColor(0, 0, 0),
    }
    if is_dark:
        border_map = {
            "white": QColor("#cbd5e1"),
            "light_gray": QColor("#64748b"),
            "dark_gray": QColor("#475569"),
            "black": QColor("#64748b"),
        }
    else:
        border_map = {
            "white": QColor("#94a3b8"),
            "light_gray": QColor("#94a3b8"),
            "dark_gray": QColor("#475569"),
            "black": QColor("#334155"),
        }

    size = 14
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    r = 4.5
    cx, cy = size / 2.0, size / 2.0
    p.setBrush(QBrush(fill_map.get(key, QColor(128, 128, 128))))
    p.setPen(QPen(border_map.get(key, QColor(100, 100, 100)), 1.0))
    p.drawEllipse(QRectF(cx - r, cy - r, r * 2.0, r * 2.0))
    p.end()
    return QIcon(pm)


class DarkroomGLCanvas(QOpenGLWidget):
    """Hardware-accelerated OpenGL viewport:
    - 60+ FPS real-time GLSL rendering of 3D LUT, Exposure, Enlarger CMY, Halation, Grain
    - Smooth pan & zoom
    - Draggable Before/After split screen
    """
    zoomChanged = Signal(float)
    requestUndo = Signal()
    requestRedo = Signal()
    requestReset = Signal()
    requestResetToOpen = Signal()
    requestClearCrop = Signal()
    # A stock's film finished laying on the build thread; the next paint can read it.
    grainReady = Signal()
    # Emitted when a render had to leave the GPU (currently only the >16384 px texture limit).
    # The console line alone was invisible: the user saw "export finished" with no hint that it
    # went through the CPU, i.e. that it took minutes instead of seconds.
    gpuFallback = Signal(str)
    requestExport = Signal()
    requestQuickExport = Signal()
    requestToggleSplit = Signal()
    requestCompareStart = Signal()
    requestCompareEnd = Signal()
    importRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.engine = getattr(parent, "engine", None)

        # Canvas Background Color Configuration
        self._bg_choice = config_manager.get_preference("canvas_background", "dark_gray")
        if self._bg_choice not in CANVAS_BG_COLORS:
            self._bg_choice = "dark_gray"
        self._bg_color = CANVAS_BG_COLORS[self._bg_choice]

        # Texture state
        self._image_tex_id = None
        self._lut_tex_id = None
        # Separable-Gaussian halation chain (ui/gl_halation.py). Created lazily on the first
        # render that needs it, so no GL objects exist before a context is current.
        self._halation_chain = None
        self._image_width = 0
        self._image_height = 0
        self._master_width = 0
        self._master_height = 0
        self._has_image = False
        self._has_lut = False
        self._is_placeholder = False
        self._has_library_photos = False
        self._pending_image = None
        self._pending_lut = None
        # Negative-density cube travelling with _pending_lut: the grain sampler indexes
        # its density layers by it, so it keeps riding along with the LUT bake.
        self._film_density = None
        self._pending_density = None
        # The GPU sampler (ui/gl_film_grain.py): one stock's film resident at a time.
        self._grain = None
        self.grainReady.connect(self.update)
        self._gl_initialized = False

        # Long-press on canvas to preview original (Before)
        self._canvas_long_press_timer = QTimer(self)
        self._canvas_long_press_timer.setSingleShot(True)
        self._canvas_long_press_timer.setInterval(220)
        self._canvas_long_press_timer.timeout.connect(self._on_canvas_long_press_timeout)
        self._is_canvas_long_pressing = False
        self._press_start_pos = None

        # Parameters
        self.params = {
            "film_stock": "none",
            "exposure_ev": 0.0,
            "highlights": 0.0,
            "shadows": 0.0,
            "color_temp": 5500.0,
            "tint": 1.0,
            "film_format_mm": 35.0,
            "diffusion_family": "none",
            "diffusion_strength": 0.0,
            "diffusion_warmth": 0.0,
            "dir_amount": 1.0,
            "dir_interlayer": 1.0,
            "dir_samelayer": 1.0,
            "enlarger_illuminant": "TH-KG3",
            "enlarger_cyan": 0.0,
            "enlarger_magenta": 0.0,
            "enlarger_yellow": 0.0,
            "print_exposure": 1.0,
            "pre_flash": 0.0,
            "black_lift": 0.0,
            "morph_gamma": 1.0,
            "developer_exhaustion": 0.0,
            "halation": 1.0,
            "halation_amount": 1.0,
            "halation_spatial_scale": 1.0,
            "scatter_amount": 1.0,
            "boost_ev": 0.0,
            "halation_boost": 0.0,
            "_halation_preset": None,
            "grain": 0.4,
            "grain_size": 1.0,
            "grain_colour": 1.0,
            "split_x": 0.5,
            "view_mode": 0   # 0: single, 1: split
        }

        # Viewport Pan & Zoom
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._is_panning = False
        self._is_split_dragging = False
        self._last_mouse_pos = QPointF(0, 0)

        # Floating HUD Badges for Before / After Compare (Native QLabels immune to OpenGL texture corruption)
        self.badge_before = QLabel("原片", self)
        self.badge_before.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_before.setStyleSheet("""
            QLabel {
                background-color: rgba(18, 19, 24, 215);
                color: #94a3b8;
                border: 1px solid rgba(255, 255, 255, 0.20);
                border-radius: 4px;
                font-family: "Microsoft YaHei UI", -apple-system, sans-serif;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 6px;
            }
        """)
        self.badge_before.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.badge_before.hide()

        # Interactive Crop Overlay
        from ui.crop_overlay import CropOverlayWidget
        self.crop_overlay = CropOverlayWidget(self, self)
        self.is_crop_editing = False

        self.badge_after = QLabel("已修改", self)
        self.badge_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_after.setStyleSheet("""
            QLabel {
                background-color: rgba(28, 20, 10, 225);
                color: #f59e0b;
                border: 1px solid rgba(245, 158, 11, 0.45);
                border-radius: 4px;
                font-family: "Microsoft YaHei UI", -apple-system, sans-serif;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 6px;
            }
        """)
        self.badge_after.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.badge_after.hide()

        # Empty State Overlay Widget (shown when no photo is loaded or filmstrip is cleared)
        self.empty_overlay = QWidget(self)
        self.empty_overlay.setObjectName("canvasEmptyOverlay")
        self.empty_overlay.setStyleSheet("""
            QWidget#canvasEmptyOverlay {
                background: rgba(14, 15, 18, 0.90);
                border: 1.5px dashed #303340;
                border-radius: 12px;
            }
            QWidget#canvasEmptyOverlay:hover {
                border-color: #f59e0b;
                background: rgba(20, 21, 26, 0.94);
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)
        empty_layout = QVBoxLayout(self.empty_overlay)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.setSpacing(12)

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            lbl_empty_icon = QLabel(self.empty_overlay)
            lbl_empty_icon.setFixedSize(48, 48)
            pix = QPixmap(icon_path).scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_empty_icon.setPixmap(pix)
            lbl_empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(lbl_empty_icon, 0, Qt.AlignmentFlag.AlignCenter)

        lbl_empty_title = QLabel("暂无打开的底片", self.empty_overlay)
        lbl_empty_title.setStyleSheet("color: #e2e8f0; font-size: 15px; font-weight: bold;")
        lbl_empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl_empty_title)

        lbl_empty_hint = QLabel("拖入照片 / RAW 负片，或点击下方按钮导入", self.empty_overlay)
        lbl_empty_hint.setStyleSheet("color: #71717a; font-size: 12px;")
        lbl_empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl_empty_hint)

        btn_empty_import = QPushButton("＋ 导入底片...", self.empty_overlay)
        btn_empty_import.setFixedSize(140, 32)
        btn_empty_import.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_empty_import.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #111113;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #d97706;
            }
            QPushButton:pressed {
                background: #b45309;
            }
        """)
        btn_empty_import.clicked.connect(self.importRequested.emit)
        empty_layout.addWidget(btn_empty_import, 0, Qt.AlignmentFlag.AlignCenter)
        self.empty_overlay.hide()

        # Item 17: Smooth Animation for zoom and centering
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setDuration(180)
        self._zoom_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom_anim.valueChanged.connect(self._on_anim_step)
        self._start_zoom = 1.0
        self._target_zoom = 1.0
        self._start_pan_x = 0.0
        self._target_pan_x = 0.0
        self._start_pan_y = 0.0
        self._target_pan_y = 0.0

        # GL objects
        self.shader_program = None
        self.vbo = None

    def _release_gl_resources(self):
        """Drop GL objects from a previous initialisation.

        `render_offscreen()` may call `initializeGL()` before the widget is ever shown, and
        Qt calls it a second time on the first real paint. Without this the first program
        and VBO were orphaned on the GPU for the lifetime of the process.
        """
        prog = getattr(self, "shader_program", None)
        if prog is not None:
            try:
                prog.removeAllShaders()
            except Exception:
                pass
            try:
                prog.deleteLater()
            except Exception:
                pass
            self.shader_program = None

        vbo = getattr(self, "vbo", None)
        if vbo is not None:
            try:
                if vbo.isCreated():
                    vbo.destroy()
            except Exception:
                pass
            self.vbo = None

        grain = getattr(self, "_grain", None)
        if grain is not None:
            grain.release()
            self._grain = None

    def initializeGL(self):
        self._release_gl_resources()
        self.shader_program = QOpenGLShaderProgram(self)
        vert_ok = self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER_SRC)
        frag_ok = self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER_SRC)
        # link() reports success even when a stage failed to compile -- Qt only links the
        # stages it managed to build. The failure then surfaces as "every uniform
        # location is -1" and a completely black canvas, with log() empty by the time
        # anyone looks. Read the log *here*, immediately, before link() clears it.
        if not vert_ok or not frag_ok:
            _c_log = self.shader_program.log()
            print("[DarkroomGLCanvas] SHADER COMPILE FAILED "
                  f"(vertex={vert_ok} fragment={frag_ok}):\n{_c_log}")
        self.shader_program.link()
        if not self.shader_program.isLinked():
            print("[DarkroomGLCanvas] SHADER LINK FAILED:\n" + self.shader_program.log())

        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype=np.float32)

        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(vertices.tobytes(), vertices.nbytes)
        self.vbo.release()

        self._gl_initialized = True

        # Process any pending image / LUT uploads that arrived before widget was mapped
        if self._pending_image is not None:
            self._do_upload_image(self._pending_image)
            self._pending_image = None
            self.fit_to_view()
            self._update_badges()
            self.update()
        if self._pending_lut is not None:
            self._do_upload_lut(self._pending_lut)
            self._pending_lut = None
            self.update()

    def set_canvas_background(self, choice: str):
        """Change canvas background color mode ('white', 'light_gray', 'dark_gray', 'black')."""
        if choice in CANVAS_BG_COLORS:
            self._bg_choice = choice
            self._bg_color = CANVAS_BG_COLORS[choice]
            config_manager.set_preference("canvas_background", choice)
            self.update()

    def get_canvas_background(self) -> str:
        """Returns the current canvas background key ('white', 'light_gray', 'dark_gray', 'black')."""
        return self._bg_choice

    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        if not self.shader_program or not self.vbo:
            return

        if self._pending_image is not None:
            self._do_upload_image(self._pending_image)
            self._pending_image = None
        if self._pending_lut is not None:
            self._do_upload_lut(self._pending_lut)
            self._pending_lut = None

        funcs = QOpenGLContext.currentContext().functions()
        GL.glClearColor(self._bg_color[0], self._bg_color[1], self._bg_color[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        self.shader_program.bind()
        self.vbo.bind()
        self.shader_program.enableAttributeArray(0)
        self.shader_program.setAttributeBuffer(0, GL.GL_FLOAT, 0, 2, 4 * 4)
        self.shader_program.enableAttributeArray(1)
        self.shader_program.setAttributeBuffer(1, GL.GL_FLOAT, 2 * 4, 2, 4 * 4)

        # Helper setters using QOpenGLFunctions to ensure compatibility across PySide6 versions
        def u_1f(name, val):
            loc = self.shader_program.uniformLocation(name)
            if loc >= 0:
                funcs.glUniform1f(loc, float(val))

        def u_1i(name, val):
            loc = self.shader_program.uniformLocation(name)
            if loc >= 0:
                funcs.glUniform1i(loc, int(val))

        def u_2f(name, x, y):
            loc = self.shader_program.uniformLocation(name)
            if loc >= 0:
                funcs.glUniform2f(loc, float(x), float(y))

        def u_3f(name, x, y, z):
            loc = self.shader_program.uniformLocation(name)
            if loc >= 0:
                funcs.glUniform3f(loc, float(x), float(y), float(z))

        def u_4f(name, x, y, z, w):
            loc = self.shader_program.uniformLocation(name)
            if loc >= 0:
                funcs.glUniform4f(loc, float(x), float(y), float(z), float(w))

        u_3f("u_bg_color", self._bg_color[0], self._bg_color[1], self._bg_color[2])

        screen_aspect = float(max(1, self.width())) / float(max(1, self.height()))
        m_w = self._master_width if self._master_width > 0 else self._image_width
        m_h = self._master_height if self._master_height > 0 else self._image_height
        native_aspect = float(m_w) / float(max(1, m_h))

        # Adjust quad aspect ratio if crop is active in normal view
        crop_cfg = self.params.get("crop", {})
        crop_enabled = 1 if (isinstance(crop_cfg, dict) and crop_cfg.get("enabled", False)) else 0
        rot_90 = int(crop_cfg.get("rot_90", 0)) % 360 if isinstance(crop_cfg, dict) else 0
        cw = float(crop_cfg.get("w", 1.0)) if isinstance(crop_cfg, dict) else 1.0
        ch = float(crop_cfg.get("h", 1.0)) if isinstance(crop_cfg, dict) else 1.0
        is_editing = getattr(self, "is_crop_editing", False)
        eff_native_aspect = (1.0 / native_aspect) if (rot_90 in (90, 270)) else native_aspect
        if crop_enabled == 1 and not is_editing and cw > 0.01 and ch > 0.01:
            quad_aspect = eff_native_aspect * (cw / ch)
        else:
            quad_aspect = eff_native_aspect

        u_1f("u_screen_aspect", screen_aspect)
        u_1f("u_image_aspect", quad_aspect)
        u_1f("u_native_aspect", native_aspect)
        u_2f("u_pan", self.pan_x, self.pan_y)
        u_1f("u_zoom", self.zoom)

        u_1i("u_has_image", 1 if self._has_image else 0)
        u_1i("u_has_lut", 1 if self._has_lut else 0)
        if getattr(self, "_is_placeholder", False):
            is_bypass = 2
        else:
            is_bypass = 0
        u_1i("u_is_bypass", is_bypass)

        eff_exposure = float(self.params.get("exposure_ev", 0.0)) + float(self.params.get("base_ev", 0.0))
        u_1f("u_base_ev", float(self.params.get("base_ev", 0.0)))
        u_1f("u_highlights", float(self.params.get("highlights", 0.0)))
        u_1f("u_shadows", float(self.params.get("shadows", 0.0)))
        u_3f("u_scene_gain", *scene_gain(
            eff_exposure,
            self.params.get("color_temp", 5500.0),
            self.params.get("base_temp", 5500.0),
            self.params.get("tint", 1.0),
            self.params.get("base_tint", 1.0)))
        film_id = self.params.get("film_stock", getattr(self, "current_film_stock", "none"))
        is_pos = self.engine.is_positive_film(film_id) if hasattr(self.engine, 'is_positive_film') else False
        # 数码原片旁通 (no film media): exposure / WB / highlights-shadows / optics stay live,
        # every film-medium stage (halation, scatter, grain, paper chemistry) must be inert.
        is_film_none = str(film_id).lower() in ("none", "")
        paper_id = self.params.get("paper_profile", getattr(self, "current_paper_stock", "none"))
        pmode = self.params.get("print_mode", getattr(self, "current_print_mode", "optical"))
        is_scan = (paper_id == "none" or pmode == "scan")
        if is_pos or is_scan:
            u_3f("u_cmy", 0.0, 0.0, 0.0)
            u_1f("u_print_exposure", 1.0)
            u_1f("u_preflash", 0.0)
            u_1f("u_black_lift", float(self.params.get("black_lift", 0.0)))
        else:
            u_3f("u_cmy", float(self.params.get("enlarger_cyan", 0.0)), float(self.params.get("enlarger_magenta", 0.0)), float(self.params.get("enlarger_yellow", 0.0)))
            u_1f("u_print_exposure", float(self.params.get("print_exposure", 1.0)))
            u_1f("u_preflash", float(self.params.get("pre_flash", 0.0)))
            u_1f("u_black_lift", float(self.params.get("black_lift", 0.0)))

        # Lens correction uniforms
        lens_cfg = self.params.get("lens", {})
        if isinstance(lens_cfg, dict):
            lens_en = 1 if lens_cfg.get("enabled", False) else 0
            u_1i("u_lens_enabled", lens_en)
            u_3f("u_lens_distort", lens_cfg.get("k1", 0.0), lens_cfg.get("k2", 0.0), lens_cfg.get("k3", 0.0))
            u_1f("u_lens_dist_scale", lens_cfg.get("dist_scale", 0.0) or 0.0)
            u_1f("u_lens_fill_scale", lens_cfg.get("fill_scale", 0.0) or 0.0)
            u_2f("u_lens_vignette", lens_cfg.get("vignette_amount", 0.0), lens_cfg.get("vignette_midpoint", 50.0))
            ca_en = 1 if lens_cfg.get("remove_ca", False) else 0
            u_1i("u_lens_ca_enabled", ca_en)
            cr0, cr1, cr2 = _ca_coeffs(lens_cfg, "ca_r_coeffs", "ca_kr")
            cb0, cb1, cb2 = _ca_coeffs(lens_cfg, "ca_b_coeffs", "ca_kb")
            u_3f("u_lens_ca_r", cr0, cr1, cr2)
            u_3f("u_lens_ca_b", cb0, cb1, cb2)
            u_1f("u_lens_ca_scale_r", lens_cfg.get("ca_r_scale", 0.0) or 0.0)
            u_1f("u_lens_ca_scale_b", lens_cfg.get("ca_b_scale", 0.0) or 0.0)
        else:
            u_1i("u_lens_enabled", 0)
            u_3f("u_lens_distort", 0.0, 0.0, 0.0)
            u_1f("u_lens_dist_scale", 0.0)
            u_1f("u_lens_fill_scale", 0.0)
            u_2f("u_lens_vignette", 0.0, 50.0)
            u_1i("u_lens_ca_enabled", 0)
            u_3f("u_lens_ca_r", 0.0, 0.0, 0.0)
            u_3f("u_lens_ca_b", 0.0, 0.0, 0.0)
            u_1f("u_lens_ca_scale_r", 0.0)
            u_1f("u_lens_ca_scale_b", 0.0)

        # Crop uniforms
        is_editing = getattr(self, "is_crop_editing", False)
        if isinstance(crop_cfg, dict):
            c_angle_rad = math.radians(float(crop_cfg.get("angle", 0.0)))
            c_rot90 = int(crop_cfg.get("rot_90", 0)) % 360
            flip_h = 1.0 if crop_cfg.get("flip_h", False) else 0.0
            flip_v = 1.0 if crop_cfg.get("flip_v", False) else 0.0
            u_1i("u_crop_enabled", 1 if (crop_enabled == 1 or is_editing) else 0)
            u_4f("u_crop_rect", crop_cfg.get("x", 0.0), crop_cfg.get("y", 0.0), cw, ch)
            u_1f("u_crop_angle", c_angle_rad)
            u_1i("u_crop_rot90", c_rot90)
            u_2f("u_crop_flip", flip_h, flip_v)
        else:
            u_1i("u_crop_enabled", 1 if is_editing else 0)
            u_4f("u_crop_rect", 0.0, 0.0, 1.0, 1.0)
            u_1f("u_crop_angle", 0.0)
            u_1i("u_crop_rot90", 0)
            u_2f("u_crop_flip", 0.0, 0.0)
        u_1i("u_is_crop_editing", 1 if is_editing else 0)

        # Halation physics (Official Spektrafilm Model)
        hal_preset = self.params.get("_halation_preset")
        if not hal_preset:
            eng = getattr(self, "engine", None)
            if not eng and hasattr(self.parent(), "engine"):
                eng = self.parent().engine
            if eng:
                stock = self.params.get("film_stock", "kodak_portra_400")
                hal_preset = eng.get_film_halation_preset(stock)
        if not hal_preset:
            hal_preset = {"sigma_h": 65.0, "strength": [0.08, 0.02, 0.0]}
        strength = hal_preset.get("strength", [0.08, 0.02, 0.0])
        sigma_um = float(hal_preset.get("sigma_h", 65.0))

        hal_amt = 0.0 if is_film_none else float(self.params.get("halation_amount", self.params.get("halation", 1.0)))
        hal_scale = float(self.params.get("halation_spatial_scale", 1.0))
        scatter_amt = 0.0 if is_film_none else float(self.params.get("scatter_amount", 1.0))
        boost_ev = 0.0 if is_film_none else float(self.params.get("boost_ev", self.params.get("halation_boost", 0.0)))

        u_1f("u_halation", hal_amt)
        u_1f("u_hal_spatial_scale", hal_scale)
        u_1f("u_scatter_amount", scatter_amt)
        u_3f("u_hal_strength_rgb", float(strength[0]), float(strength[1]), float(strength[2]))
        u_1f("u_hal_sigma_um", sigma_um)
        # The live viewport always feeds the raw texture, so the shader must apply gain and
        # its own halation. The offscreen path sets this to 1 when ui/gl_halation.py has
        # pre-rendered the halo; uniforms persist per program, so it has to be reset here.
        u_1f("u_scene_prepared", 0.0)

        # Grain: removed with the old octave model (see docs/grain-port-plan.md P0).
        # The film-tile sampler lands in P3; until then the stage is a pass-through.
        u_1f("u_film_width_um", max(5.0, float(self.params.get("film_format_mm", 35.0))) * 1000.0)

        # Diffusion filter
        diff_family = str(self.params.get("diffusion_family", "none")).lower()
        family_map = {
            "none": 0,
            "black_pro_mist": 1,
            "glimmerglass": 2,
            "pro_mist": 3,
            "cinebloom": 4
        }
        u_1i("u_diffusion_type", family_map.get(diff_family, 0))
        u_1f("u_diffusion_strength", float(self.params.get("diffusion_strength", 0.0)))
        u_1f("u_diffusion_warmth", float(self.params.get("diffusion_warmth", 0.0)))
        # Density and PSF geometry both come from the export's own table, so the
        # shader never derives its own intensity or radius.
        _dp = diffusion_shader_profile(diff_family, self.params.get("diffusion_strength", 0.0))
        u_1f("u_diffusion_ps", _dp["ps"])
        u_1f("u_diff_core_w", _dp["core_w"])
        u_1f("u_diff_halo_w", _dp["halo_w"])
        u_1f("u_diff_bloom_w", _dp["bloom_w"])
        u_1f("u_diff_halo_um", _dp["halo_um"])
        u_1f("u_diff_bloom_um", _dp["bloom_um"])
        u_1f("u_diff_warmth_base", _dp["warmth_base"])
        u_1f("u_diff_cutoff", _dp["cutoff"])

        # Chemistry Morph / Developer Exhaustion deltas (DIR is fully baked in the LUT)
        _bk = getattr(self, "_lut_baked", None) or self._lookup_lut_baked(None)
        u_1f("u_lut_morph_gamma", float(_bk.get("morph_gamma", 1.0)))
        u_1f("u_lut_dev_exh", float(_bk.get("developer_exhaustion", 0.0)))

        # Chemistry Morph (neutralized for positive slide film and film-less bypass)
        if is_pos or is_film_none:
            u_1f("u_morph_gamma", 1.0)
            u_1f("u_developer_exhaustion", 0.0)
        else:
            u_1f("u_morph_gamma", float(self.params.get("morph_gamma", 1.0)))
            u_1f("u_developer_exhaustion", float(self.params.get("developer_exhaustion", 0.0)))

        u_1f("u_split_x", self.params.get("split_x", 0.5))
        u_1i("u_view_mode", self.params.get("view_mode", 0))
        u_2f("u_viewport_size", float(max(1, self.width())), float(max(1, self.height())))

        # --- separable-Gaussian halo / scatter, the same chain the export uses -------
        # Without this the viewport keeps the 16-tap approximation while the export uses a
        # real Gaussian, so the two paths disagree by 20% of the pixels -- the very thing
        # "preview and export differ by construction" is meant to rule out.
        prepared_tex = None
        if (self._image_tex_id and (hal_amt > 0.001 or scatter_amt > 0.001)
                and not is_film_none and diff_family == "none" and boost_ev <= 0.001):
            try:
                _eng = getattr(self, "engine", None)
                if _eng is None and hasattr(self.parent(), "engine"):
                    _eng = self.parent().engine
                if _eng is not None and hasattr(_eng, "get_halation_params"):
                    hp_live = _eng.get_halation_params(self.params.get("film_stock", "none"))
                    tex_w = max(1, int(getattr(self, "_image_width", 0) or 1))
                    tex_h = max(1, int(getattr(self, "_image_height", 0) or 1))
                    fmt_mm = float(self.params.get("film_format_mm", 35.0))
                    um_px = (max(5.0, fmt_mm) * 1000.0) / float(max(tex_w, tex_h))
                    sc_live = max(0.2, float(hal_scale))
                    sig_live = [float(v) * sc_live / um_px for v in hp_live.sigma_h_um]
                    scat_live = None
                    if scatter_amt > 0.001:
                        scat_live = {
                            "core": [max(0.5, float(hp_live.scatter_core_um[c]) * sc_live / um_px)
                                     for c in range(3)],
                            "tail": [max(1.0, float(hp_live.scatter_tail_um[c]) * sc_live / um_px)
                                     for c in range(3)],
                            "w_tail": [float(v) for v in hp_live.scatter_tail_weight],
                            "eff": min(0.5, float(scatter_amt) * 0.06),
                            "amount": float(scatter_amt),
                        }
                    if max(sig_live) * (3.0 ** 0.5) > 6.0 or scat_live:
                        if self._halation_chain is None:
                            from .gl_halation import GLHalationChain
                            self._halation_chain = GLHalationChain()
                        _gain_live = scene_gain(
                            float(self.params.get("exposure_ev", 0.0)) + float(self.params.get("base_ev", 0.0)),
                            self.params.get("color_temp", 5500.0),
                            self.params.get("base_temp", 5500.0),
                            self.params.get("tint", 1.0),
                            self.params.get("base_tint", 1.0))
                        prepared_tex = self._halation_chain.run(
                            int(self._image_tex_id), tex_w, tex_h, _gain_live,
                            sig_live, hp_live.bounce_decay, hp_live.strength,
                            float(hal_amt) if max(sig_live) * (3.0 ** 0.5) > 6.0 else 0.0,
                            hs=(float(self.params.get("highlights", 0.0)),
                                float(self.params.get("shadows", 0.0))),
                            scatter=scat_live)
            except Exception as _ex_live:
                print(f"[DarkroomGLCanvas] live halation chain skipped: {_ex_live}")
                prepared_tex = None

        if prepared_tex:
            if getattr(self._halation_chain, "last_did_halation", True):
                u_1f("u_halation", 0.0)
            if getattr(self._halation_chain, "last_did_scatter", False):
                u_1f("u_scatter_amount", 0.0)
            u_1f("u_scene_prepared", 1.0)

        # Bind 2D Image Texture (Unit 0)
        if self._image_tex_id:
            funcs.glActiveTexture(0x84C0)  # GL_TEXTURE0
            GL.glBindTexture(GL.GL_TEXTURE_2D, prepared_tex if prepared_tex else self._image_tex_id)
            u_1i("u_image", 0)

        # Bind 3D LUT Texture (Unit 1)
        if self._lut_tex_id:
            funcs.glActiveTexture(0x84C1)  # GL_TEXTURE1
            GL.glBindTexture(GL.GL_TEXTURE_3D, self._lut_tex_id)
            u_1i("u_lut", 1)

        # Film grain (units 3 and 4): the stock's laid film and the density cube it is indexed by.
        self._bind_grain(u_1f, u_1i, u_2f, u_3f, (self._image_w, self._image_h))

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        self.shader_program.disableAttributeArray(0)
        self.shader_program.disableAttributeArray(1)
        self.vbo.release()
        self.shader_program.release()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'crop_overlay') and self.crop_overlay.isVisible():
            self.crop_overlay.setGeometry(self.rect())
        if self.params.get("view_mode") == 1:
            self._clamp_split_to_image()
        self._update_badges()

    def _update_badges(self):
        has_lib_photos = getattr(self, "_has_library_photos", False)
        if not has_lib_photos:
            p = self.parent() if hasattr(self, "parent") else None
            while p:
                if hasattr(p, "photos") and len(getattr(p, "photos", [])) > 0:
                    has_lib_photos = True
                    break
                p = p.parent() if hasattr(p, "parent") else None

        if not self._has_image and not has_lib_photos and self.width() > 0:
            if hasattr(self, 'empty_overlay'):
                ow, oh = 380, 210
                ox = max(10, (self.width() - ow) // 2)
                oy = max(10, (self.height() - oh) // 2)
                self.empty_overlay.setGeometry(ox, oy, ow, oh)
                self.empty_overlay.show()
                self.empty_overlay.raise_()
        else:
            if hasattr(self, 'empty_overlay'):
                self.empty_overlay.hide()

        if self.params.get("view_mode") == 1 and self._has_image and self.width() > 0:
            split_x = float(self.params.get("split_x", 0.5)) * float(self.width())
            left, right, _, _ = self.get_image_bounds_norm()
            img_vis_left_px = max(0.0, left) * float(self.width())
            img_vis_right_px = min(1.0, right) * float(self.width())

            b1_w, b1_h = 56, 26
            b2_w, b2_h = 68, 26
            b1_x = int(max(img_vis_left_px + 12.0, min(self.width() - 140.0, split_x - b1_w - 12.0)))
            b2_x = int(max(80.0, min(img_vis_right_px - b2_w - 12.0, split_x + 12.0)))
            self.badge_before.setText("原片")
            self.badge_before.setGeometry(b1_x, 16, b1_w, b1_h)
            self.badge_after.setGeometry(b2_x, 16, b2_w, b2_h)
            self.badge_before.show()
            self.badge_after.show()
            self.badge_before.raise_()
            self.badge_after.raise_()
        elif self.params.get("view_mode") == 2 and self._has_image and self.width() > 0:
            b1_w, b1_h = 56, 26
            b2_w, b2_h = 68, 26
            half_w = float(self.width()) / 2.0
            b1_x = int(half_w / 2.0 - b1_w / 2.0)
            b2_x = int(half_w + half_w / 2.0 - b2_w / 2.0)
            self.badge_before.setText("原片")
            self.badge_before.setGeometry(b1_x, 16, b1_w, b1_h)
            self.badge_after.setGeometry(b2_x, 16, b2_w, b2_h)
            self.badge_before.show()
            self.badge_after.show()
            self.badge_before.raise_()
            self.badge_after.raise_()
        elif self.params.get("view_mode") == 3 and self._has_image and self.width() > 0:
            b1_w, b1_h = 56, 26
            b1_x = int(self.width() / 2.0 - b1_w / 2.0)
            self.badge_before.setText("原片")
            self.badge_before.setGeometry(b1_x, 16, b1_w, b1_h)
            self.badge_before.show()
            self.badge_before.raise_()
            self.badge_after.hide()
        else:
            self.badge_before.setText("原片")
            self.badge_before.hide()
            self.badge_after.hide()

    def set_image(self, float_img_rgb, master_width=None, master_height=None, reset_zoom=True, photo_id=None, is_placeholder=False):
        """Upload source image array (float32 [H, W, 3]) to 2D texture, or clear if None."""
        self._is_placeholder = is_placeholder
        if float_img_rgb is None:
            self._image_width = 0
            self._image_height = 0
            self._master_width = 0
            self._master_height = 0
            self._pending_image = None
            self._has_image = False
            self._is_placeholder = False
            self._loaded_photo_id = None
            if getattr(self, "_gl_initialized", False) and self.isValid():
                self.makeCurrent()
                try:
                    if self._image_tex_id:
                        GL.glDeleteTextures([self._image_tex_id])
                        self._image_tex_id = None
                finally:
                    self.doneCurrent()
            self._update_badges()
            self.update()
            return

        self._loaded_photo_id = photo_id
        try:
            h, w = float_img_rgb.shape[:2]
            self._image_width = w
            self._image_height = h
            self._master_width = int(master_width) if master_width else w
            self._master_height = int(master_height) if master_height else h
            # Enforce orientation consistency between uploaded texture array and master dimensions
            if (h > w and self._master_width > self._master_height) or (w > h and self._master_height > self._master_width):
                self._master_width, self._master_height = self._master_height, self._master_width
        except Exception:
            pass
        if not getattr(self, "_gl_initialized", False) or not self.isValid():
            self._pending_image = float_img_rgb
            self._has_image = True
            if hasattr(self, 'empty_overlay'):
                self.empty_overlay.hide()
            if reset_zoom:
                self.fit_to_view(animate=False)
                if self.params.get("view_mode") == 1:
                    self.reset_split_to_center()
            else:
                if self.params.get("view_mode") == 1:
                    self._clamp_split_to_image()
            self._update_badges()
            return
        self.makeCurrent()
        try:
            self._do_upload_image(float_img_rgb)
        finally:
            self.doneCurrent()
        if reset_zoom:
            self.fit_to_view()
            if self.params.get("view_mode") == 1:
                self.reset_split_to_center()
        else:
            if self.params.get("view_mode") == 1:
                self._clamp_split_to_image()
        self._update_badges()
        self.update()

    def set_placeholder_image(self, rgb_array, master_width=None, master_height=None, photo_id=None):
        """Upload low-res or embedded sRGB placeholder image (uint8 or float32 [H, W, 3])
        with direct sRGB bypass (u_is_bypass = 2) for instant 0ms visual continuity.
        """
        if rgb_array is None:
            return
        if rgb_array.dtype == np.uint8:
            float_data = (rgb_array.astype(np.float32) / 255.0)
        else:
            float_data = np.asarray(rgb_array, dtype=np.float32)
            if float_data.max() > 1.05:
                float_data = float_data / 255.0
        self.set_image(float_data, master_width=master_width, master_height=master_height, reset_zoom=True, photo_id=photo_id, is_placeholder=True)

    def set_has_library_photos(self, has_photos: bool):
        """Notifies canvas whether the session photo library has photos.
        Prevents empty overlay banner from appearing during photo switches.
        """
        self._has_library_photos = bool(has_photos)
        self._update_badges()

    def _do_upload_image(self, float_img_rgb):
        h, w = float_img_rgb.shape[:2]
        self._image_width = w
        self._image_height = h

        if self._image_tex_id:
            GL.glDeleteTextures([self._image_tex_id])
            self._image_tex_id = None

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        self._image_tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._image_tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        data = np.ascontiguousarray(float_img_rgb, dtype=np.float32)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, w, h, 0, GL.GL_RGB, GL.GL_FLOAT, data)
        try:
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        except Exception:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._has_image = True

    def _lookup_lut_baked(self, lut_array_3d):
        """Chemistry values (DIR / paper morph) already baked into a LUT produced by the engine."""
        eng = getattr(self, "engine", None)
        if eng is not None and hasattr(eng, "get_lut_baked_params") and lut_array_3d is not None:
            try:
                return eng.get_lut_baked_params(lut_array_3d)
            except Exception:
                pass
        return {"dir_amount": 1.0, "dir_interlayer": 1.0, "dir_samelayer": 1.0, "morph_gamma": 1.0, "developer_exhaustion": 0.0}

    def set_3d_lut(self, lut_array_3d, film_density=None):
        """Upload 3D LUT array (float32 [N, N, N, 3]) to 3D texture.

        ``film_density`` is the negative-density cube baked alongside that LUT.  It has to
        travel with the LUT, not be looked up later: the grain law is indexed by density, and a
        cube from a different (film, paper, DIR) bake would silently misplace the whole curve.
        """
        self._lut_baked = self._lookup_lut_baked(lut_array_3d)
        self._pending_lut = lut_array_3d
        self._film_density = film_density
        self._pending_density = film_density
        self.update()

    set_lut = set_3d_lut

    def _do_upload_lut(self, lut_array_3d):
        size = lut_array_3d.shape[0]
        data = np.ascontiguousarray(lut_array_3d, dtype=np.float32)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if self._lut_tex_id and getattr(self, "_lut_tex_size", 0) == size:
            GL.glBindTexture(GL.GL_TEXTURE_3D, self._lut_tex_id)
            GL.glTexSubImage3D(GL.GL_TEXTURE_3D, 0, 0, 0, 0, size, size, size, GL.GL_RGB, GL.GL_FLOAT, data)
            GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
            self._has_lut = True
            return

        if self._lut_tex_id:
            GL.glDeleteTextures([self._lut_tex_id])
            self._lut_tex_id = None

        self._lut_tex_id = GL.glGenTextures(1)
        self._lut_tex_size = size
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._lut_tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGB16F, size, size, size, 0, GL.GL_RGB, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        self._has_lut = True

        # The grain stage's density cube is uploaded here too: it is only meaningful against the
        # LUT that was baked with it, and this is the one place both are guaranteed in hand.
        if self._pending_density is not None:
            try:
                if self._grain is None:
                    from .gl_film_grain import GLFilmGrain
                    self._grain = GLFilmGrain()
                self._grain.set_density(self._pending_density)
            except Exception as exc:                 # noqa: BLE001 - never break a paint
                print(f"[SpektraGrain] density upload skipped: {exc}")
            self._pending_density = None

    # --- film grain ------------------------------------------------------------------------
    def _bind_grain(self, u_1f, u_1i, u_2f, u_3f, size) -> None:
        """Make the current stock's film resident and hand the stage everything it needs.

        A stock with no baked grain parameters (or no LUT, or a placeholder) switches the stage
        off instead of drawing something plausible: "no grain" is a truthful answer, a look-alike
        is not.
        """
        if self._grain is None:
            from .gl_film_grain import GLFilmGrain
            self._grain = GLFilmGrain()
            # The film is laid off the paint thread; the signal is queued, so the repaint lands
            # on the GUI thread no matter which one finished the work.
            self._grain.set_ready_callback(self.grainReady.emit)
        grain = self._grain
        if isinstance(self._film_density, np.ndarray):
            grain.set_density(self._film_density)
        stock = self.params.get("film_stock")
        if grain is None or not stock or stock == "none" \
                or getattr(self, "_is_placeholder", False):
            u_1i("u_grain_on", 0)
            return
        if not grain.ensure_stock(stock):
            u_1i("u_grain_on", 0)
            return

        px_per_mm = float(max(size[0], size[1], 1)) / max(
            5.0, float(self.params.get("film_format_mm", 35.0)))
        for name, value in grain.scalars(self.params, px_per_mm, size).items():
            if isinstance(value, tuple):
                (u_2f if len(value) == 2 else u_3f)(name, *value)
            elif isinstance(value, int):
                u_1i(name, value)
            else:
                u_1f(name, value)

        for name, data in grain.arrays().items():
            loc = self.shader_program.uniformLocation(name)
            if loc >= 0:
                GL.glUniform1fv(loc, int(data.size), data)

        if grain.bind(3) is not None:
            u_1i("u_grain_sat", 3)
        if grain.density_unit(4) is not None:
            u_1i("u_grain_density", 4)

    @property
    def _image_w(self):
        mw = getattr(self, "_master_width", 0)
        return mw if mw > 1 else max(1, getattr(self, "_image_width", 1))

    @property
    def _image_h(self):
        mh = getattr(self, "_master_height", 0)
        return mh if mh > 1 else max(1, getattr(self, "_image_height", 1))

    @property
    def master_width(self):
        return self._image_w

    @property
    def master_height(self):
        return self._image_h

    def update_params(self, **kwargs):
        """Update shader uniforms instantly and redraw."""
        self.params.update(kwargs)
        self.update()


    # -------------------------------------------------------------
    # Crop Mode Methods
    # -------------------------------------------------------------
    def _get_crop_fit_zoom(self, base_margin=0.90):
        """Calculate zoom so rotated photo bounding box fits comfortably inside the viewport."""
        crop_cfg = self.params.get("crop", {})
        if not isinstance(crop_cfg, dict):
            return base_margin
        angle = abs(float(crop_cfg.get("angle", 0.0)))
        rot_90 = int(crop_cfg.get("rot_90", 0)) % 360
        if angle <= 0.05:
            return base_margin
        rad = math.radians(angle)
        m_w = self._master_width if getattr(self, "_master_width", 0) > 0 else self._image_width
        m_h = self._master_height if getattr(self, "_master_height", 0) > 0 else self._image_height
        base_aspect = float(m_w) / float(max(1, m_h)) if m_h > 0 else 1.0
        eff_aspect = (1.0 / base_aspect) if (rot_90 in (90, 270)) else base_aspect
        ca = math.cos(rad)
        sa = math.sin(rad)
        s_x = ca + sa / eff_aspect
        s_y = ca + sa * eff_aspect
        scale_factor = max(1.0, max(s_x, s_y))
        return base_margin / scale_factor

    def start_crop_mode(self, crop_dict=None):
        if not self._has_image:
            return
        self.is_crop_editing = True
        self._pre_crop_zoom = getattr(self, "zoom", 1.0)
        self._pre_crop_pan = (getattr(self, "pan_x", 0.0), getattr(self, "pan_y", 0.0))
        cur_crop = dict(crop_dict or self.params.get("crop", {}))
        self.params["crop"] = cur_crop
        # Inset image so rotated bounding box and crop handles are not clipped by viewport edges
        self.zoom = self._get_crop_fit_zoom()
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoomChanged.emit(self.get_actual_zoom_ratio())
        self.update()
        self.crop_overlay.start_crop_session(cur_crop)

    def exit_crop_mode(self):
        self.is_crop_editing = False
        self.crop_overlay.hide()
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoomChanged.emit(self.get_actual_zoom_ratio())
        self.update()

    def get_image_bounds_norm(self):
        """Calculate image normalized bounding box in viewport coordinates [0.0, 1.0].
        Returns: (left, right, top, bottom) where:
          - left, right: horizontal screen coordinates (0.0=left, 1.0=right)
          - top, bottom: vertical Qt screen coordinates (0.0=top, 1.0=bottom)
        """
        if not self._has_image or self.width() <= 0 or self.height() <= 0:
            return 0.0, 1.0, 0.0, 1.0

        screen_aspect = float(self.width()) / float(max(1, self.height()))
        m_w = self._master_width if getattr(self, "_master_width", 0) > 0 else self._image_width
        m_h = self._master_height if getattr(self, "_master_height", 0) > 0 else self._image_height
        img_aspect = float(m_w) / float(max(1, m_h)) if m_h > 0 else 1.0

        # In normal view with crop applied, aspect ratio is governed by the cropped region
        crop_cfg = self.params.get("crop", {})
        crop_enabled = 1 if (isinstance(crop_cfg, dict) and crop_cfg.get("enabled", False)) else 0
        rot_90 = int(crop_cfg.get("rot_90", 0)) % 360 if isinstance(crop_cfg, dict) else 0
        cw = float(crop_cfg.get("w", 1.0)) if isinstance(crop_cfg, dict) else 1.0
        ch = float(crop_cfg.get("h", 1.0)) if isinstance(crop_cfg, dict) else 1.0
        is_editing = getattr(self, "is_crop_editing", False)
        eff_native_aspect = (1.0 / img_aspect) if (rot_90 in (90, 270)) else img_aspect
        if crop_enabled == 1 and not is_editing and cw > 0.01 and ch > 0.01:
            img_aspect = eff_native_aspect * (cw / ch)
        else:
            img_aspect = eff_native_aspect

        eff_screen_aspect = screen_aspect * (0.5 if self.params.get("view_mode") == 2 else 1.0)
        if eff_screen_aspect > img_aspect:
            quad_scale_x = img_aspect / eff_screen_aspect
            quad_scale_y = 1.0
        else:
            quad_scale_x = 1.0
            quad_scale_y = eff_screen_aspect / img_aspect

        half_w = 0.5 * quad_scale_x * self.zoom
        center_x = 0.5 + 0.5 * self.pan_x
        img_left = center_x - half_w
        img_right = center_x + half_w

        half_h = 0.5 * quad_scale_y * self.zoom
        center_y = 0.5 - 0.5 * self.pan_y  # Qt top-down Y coordinate
        img_top = center_y - half_h
        img_bottom = center_y + half_h

        return img_left, img_right, img_top, img_bottom

    def get_image_visible_bounds_norm(self):
        """Calculate visible intersection of image with viewport boundaries [0.0, 1.0].
        Returns: (vis_left, vis_right, vis_top, vis_bottom)
        """
        left, right, top, bottom = self.get_image_bounds_norm()
        vis_left = max(0.0, min(1.0, left))
        vis_right = max(0.0, min(1.0, right))
        vis_top = max(0.0, min(1.0, top))
        vis_bottom = max(0.0, min(1.0, bottom))
        return vis_left, vis_right, vis_top, vis_bottom

    def reset_split_to_center(self):
        """Reset split compare divider line to the geometric center of visible image area."""
        vis_left, vis_right, _, _ = self.get_image_visible_bounds_norm()
        if vis_right > vis_left:
            self.params["split_x"] = (vis_left + vis_right) * 0.5
        else:
            self.params["split_x"] = 0.5
        self._update_badges()
        self.update()

    def _clamp_split_to_image(self):
        """Clamp split_x strictly to the visible image bounds to avoid cutting empty letterbox bars."""
        if not self._has_image or self.width() <= 0:
            return
        left, right, _, _ = self.get_image_bounds_norm()
        margin = max(0.01, 8.0 / float(max(1, self.width())))
        clamp_min = max(0.0, left) + margin
        clamp_max = min(1.0, right) - margin
        if clamp_min < clamp_max:
            cur_split = float(self.params.get("split_x", 0.5))
            self.params["split_x"] = max(clamp_min, min(clamp_max, cur_split))

    def _is_hovering_split_line(self, pos, tolerance=0.02) -> bool:
        """Check whether mouse position is over the split line within the visible image bounds."""
        if self.params.get("view_mode") != 1 or not self._has_image or self.width() <= 0 or self.height() <= 0:
            return False
        norm_x = pos.x() / float(max(1, self.width()))
        norm_y = pos.y() / float(max(1, self.height()))
        split_x = float(self.params.get("split_x", 0.5))

        if abs(norm_x - split_x) > tolerance:
            return False

        # Vertical & horizontal bounds check: ensure mouse is inside image area (with small buffer)
        left, right, top, bottom = self.get_image_bounds_norm()
        pad_y = max(0.02, 12.0 / float(max(1, self.height())))
        pad_x = max(0.02, 12.0 / float(max(1, self.width())))
        vis_top = max(0.0, top) - pad_y
        vis_bottom = min(1.0, bottom) + pad_y
        vis_left = max(0.0, left) - pad_x
        vis_right = min(1.0, right) + pad_x

        return (vis_left <= norm_x <= vis_right) and (vis_top <= norm_y <= vis_bottom)

    def set_view_mode(self, mode):
        # 0: single, 1: split
        old_mode = self.params.get("view_mode", 0)
        self.params["view_mode"] = mode
        if mode == 1:
            if old_mode != 1:
                if old_mode == 3:
                    self._clamp_split_to_image()
                else:
                    self.reset_split_to_center()
            else:
                self._clamp_split_to_image()
        self._update_badges()
        self.update()

    def get_actual_zoom_ratio(self) -> float:
        """Calculate zoom ratio relative to full photo native resolution (1.0 = 100% 1:1 pixel scale)."""
        ref_w = float(getattr(self, "_master_width", 0) or self._image_width)
        ref_h = float(getattr(self, "_master_height", 0) or self._image_height)
        if ref_w <= 10 or ref_h <= 10 or self.width() <= 0 or self.height() <= 0:
            return 1.0
        aspect = ref_w / float(max(1.0, ref_h))
        screen_aspect = float(self.width()) / float(max(1.0, self.height()))
        if screen_aspect > aspect:
            base_w = self.height() * aspect
        else:
            base_w = self.width()
        zoom_100 = ref_w / float(max(1.0, base_w))
        if zoom_100 <= 0:
            return 1.0
        return self.zoom / zoom_100

    def _on_anim_step(self, val):
        t = float(val)
        self.zoom = self._start_zoom + (self._target_zoom - self._start_zoom) * t
        self.pan_x = self._start_pan_x + (self._target_pan_x - self._start_pan_x) * t
        self.pan_y = self._start_pan_y + (self._target_pan_y - self._start_pan_y) * t
        if self.params.get("view_mode") == 1:
            self._clamp_split_to_image()
            self._update_badges()
        self.zoomChanged.emit(self.get_actual_zoom_ratio())
        self.update()

    def animate_to(self, target_zoom, target_pan_x=0.0, target_pan_y=0.0):
        self._zoom_anim.stop()
        self._start_zoom = self.zoom
        self._target_zoom = float(target_zoom)
        self._start_pan_x = self.pan_x
        self._target_pan_x = float(target_pan_x)
        self._start_pan_y = self.pan_y
        self._target_pan_y = float(target_pan_y)
        self._zoom_anim.setStartValue(0.0)
        self._zoom_anim.setEndValue(1.0)
        self._zoom_anim.start()

    def fit_to_view(self, animate=True):
        target_z = self._get_crop_fit_zoom() if getattr(self, "is_crop_editing", False) else 1.0
        if animate:
            self.animate_to(target_z, 0.0, 0.0)
        else:
            self.zoom = target_z
            self.pan_x = 0.0
            self.pan_y = 0.0
            if self.params.get("view_mode") == 1:
                self.reset_split_to_center()
            self.zoomChanged.emit(self.get_actual_zoom_ratio())
            self.update()

    def set_zoom_100(self, animate=True, mouse_pos=None):
        # 1:1 camera sensor native pixel to screen display pixel (100% scale)
        ref_w = float(getattr(self, "_master_width", 0) or self._image_width)
        ref_h = float(getattr(self, "_master_height", 0) or self._image_height)
        if ref_w > 0 and self.width() > 0:
            aspect = float(ref_w) / float(max(1.0, ref_h))
            screen_aspect = float(self.width()) / float(max(1.0, self.height()))
            if screen_aspect > aspect:
                base_w = self.height() * aspect
            else:
                base_w = self.width()
            target_zoom = float(ref_w) / float(max(1.0, base_w))
        else:
            target_zoom = 1.0


        if mouse_pos is not None:
            mx = mouse_pos.x() if hasattr(mouse_pos, 'x') else float(mouse_pos[0])
            my = mouse_pos.y() if hasattr(mouse_pos, 'y') else float(mouse_pos[1])
            mouse_norm_x = (mx / float(max(1, self.width())) - 0.5) * 2.0
            mouse_norm_y = ((self.height() - my) / float(max(1, self.height())) - 0.5) * 2.0
            old_zoom = max(0.001, self.zoom)
            target_pan_x = mouse_norm_x - (mouse_norm_x - self.pan_x) * (target_zoom / old_zoom)
            target_pan_y = mouse_norm_y - (mouse_norm_y - self.pan_y) * (target_zoom / old_zoom)
        else:
            target_pan_x = 0.0
            target_pan_y = 0.0

        if animate:
            self.animate_to(target_zoom, target_pan_x, target_pan_y)
        else:
            self.zoom = target_zoom
            self.pan_x = target_pan_x
            self.pan_y = target_pan_y
            if self.params.get("view_mode") == 1:
                self._clamp_split_to_image()
                self._update_badges()
            self.zoomChanged.emit(self.get_actual_zoom_ratio())
            self.update()

    def zoom_in(self, animate=True):
        ref_w = float(getattr(self, "_master_width", 0) or self._image_width)
        ref_h = float(getattr(self, "_master_height", 0) or self._image_height)
        aspect = ref_w / float(max(1.0, ref_h)) if ref_w > 0 else 1.0
        screen_aspect = float(self.width()) / float(max(1.0, self.height())) if self.width() > 0 else 1.0
        base_w = self.height() * aspect if screen_aspect > aspect else self.width()
        zoom_100 = (ref_w / float(max(1.0, base_w))) if (ref_w > 0 and base_w > 0) else 1.0
        max_zoom = max(24.0, zoom_100 * 4.0)
        new_zoom = min(max_zoom, self.zoom * 1.25)
        if animate:
            self.animate_to(new_zoom, self.pan_x, self.pan_y)
        else:
            self.zoom = new_zoom
            if self.params.get("view_mode") == 1:
                self._clamp_split_to_image()
                self._update_badges()
            self.zoomChanged.emit(self.get_actual_zoom_ratio())
            self.update()

    def zoom_out(self, animate=True):
        new_zoom = max(0.2, self.zoom * 0.8)
        if animate:
            self.animate_to(new_zoom, self.pan_x, self.pan_y)
        else:
            self.zoom = new_zoom
            if self.params.get("view_mode") == 1:
                self._clamp_split_to_image()
                self._update_badges()
            self.zoomChanged.emit(self.get_actual_zoom_ratio())
            self.update()

    def contextMenuEvent(self, event):
        if hasattr(self, '_canvas_long_press_timer') and self._canvas_long_press_timer.isActive():
            self._canvas_long_press_timer.stop()
        if getattr(self, '_is_canvas_long_pressing', False):
            self._is_canvas_long_pressing = False
            self.requestCompareEnd.emit()
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        act_in = menu.addAction("放大 (Zoom In)")
        act_in.setIcon(get_icon("zoom_in.png"))
        act_in.setShortcut(QKeySequence("Ctrl++"))
        act_in.triggered.connect(self.zoom_in)

        act_out = menu.addAction("缩小 (Zoom Out)")
        act_out.setIcon(get_icon("zoom_out.png"))
        act_out.setShortcut(QKeySequence("Ctrl+-"))
        act_out.triggered.connect(self.zoom_out)

        act_fit = menu.addAction("适应图像 (Fit Image)")
        act_fit.setIcon(get_icon("fit_window.png"))
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self.fit_to_view)

        act_100 = menu.addAction("1:1 实际像素 (100%)")
        act_100.setIcon(get_icon("zoom_100.png"))
        act_100.setShortcut(QKeySequence("Ctrl+1"))
        act_100.triggered.connect(self.set_zoom_100)

        win = self.window()
        if hasattr(win, "_update_menu_actions_state"):
            try:
                win._update_menu_actions_state()
            except Exception:
                pass

        can_undo = win.act_undo.isEnabled() if hasattr(win, 'act_undo') else getattr(self, "can_undo", False)
        can_redo = win.act_redo.isEnabled() if hasattr(win, 'act_redo') else getattr(self, "can_redo", False)
        can_reset = win.act_reset_all.isEnabled() if hasattr(win, 'act_reset_all') else getattr(self, "can_reset", False)
        can_compare = win.act_compare.isEnabled() if hasattr(win, 'act_compare') else getattr(self, "can_compare", False)

        act_split = menu.addAction("原片前后对比")
        act_split.setIcon(get_icon("compare.png"))
        act_split.setShortcut(QKeySequence("\\"))
        act_split.setEnabled(can_compare)
        act_split.triggered.connect(self.requestToggleSplit.emit)

        menu.addSeparator()

        act_undo = menu.addAction("撤销 (Undo)")
        act_undo.setIcon(get_icon("undo.png"))
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.setEnabled(can_undo)
        act_undo.triggered.connect(self.requestUndo.emit)

        act_redo = menu.addAction("重做 (Redo)")
        act_redo.setIcon(get_icon("redo.png"))
        act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        act_redo.setEnabled(can_redo)
        act_redo.triggered.connect(self.requestRedo.emit)

        act_reset = menu.addAction("重置所有参数 (Reset All)")
        act_reset.setIcon(get_icon("reset.png"))
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.setEnabled(can_reset)
        act_reset.triggered.connect(self.requestReset.emit)

        crop_cfg = self.params.get("crop", {})
        has_active_crop = False
        if isinstance(crop_cfg, dict):
            if crop_cfg.get("enabled", False):
                has_active_crop = True
            else:
                cw = float(crop_cfg.get("w", 1.0))
                ch = float(crop_cfg.get("h", 1.0))
                cx = float(crop_cfg.get("x", 0.0))
                cy = float(crop_cfg.get("y", 0.0))
                rot = int(crop_cfg.get("rot_90", 0)) % 360
                ang = float(crop_cfg.get("angle", 0.0))
                if abs(cw - 1.0) > 0.001 or abs(ch - 1.0) > 0.001 or abs(cx) > 0.001 or abs(cy) > 0.001 or rot != 0 or abs(ang) > 0.01 or crop_cfg.get("flip_h") or crop_cfg.get("flip_v"):
                    has_active_crop = True
        act_clear_crop = menu.addAction("清除裁切")
        act_clear_crop.setEnabled(has_active_crop)
        act_clear_crop.triggered.connect(self.requestClearCrop.emit)

        menu.addSeparator()

        menu_bg = menu.addMenu("背景颜色")
        menu_bg.setStyleSheet(get_darkroom_menu_style())
        bg_group = QActionGroup(self)
        bg_group.setExclusive(True)
        try:
            from ui.theme_manager import theme_manager
            is_dark_theme = theme_manager.is_dark()
        except Exception:
            is_dark_theme = True
        for label, key in CANVAS_BG_NAMES:
            icon = get_bg_color_circle_icon(key, is_dark=is_dark_theme)
            act = menu_bg.addAction(icon, label)
            act.setCheckable(True)
            if key == self._bg_choice:
                act.setChecked(True)
            act.triggered.connect(lambda chk=False, k=key: self.set_canvas_background(k))
            bg_group.addAction(act)

        menu.addSeparator()

        act_quick = menu.addAction("快速扫描 (上次参数)")
        act_quick.setIcon(get_icon("quick_export.png"))
        act_quick.setShortcut(QKeySequence("Ctrl+Shift+E"))
        act_quick.triggered.connect(self.requestQuickExport.emit)

        act_export = menu.addAction("扫描照片 (Scan)...")
        act_export.setIcon(get_icon("export.png"))
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.requestExport.emit)

        menu.exec(event.globalPos())

    def _clamp_pan(self):
        """Clamp pan_x and pan_y to prevent the image from being thrown out of the viewport."""
        limit = max(1.5, self.zoom * 2.0)
        self.pan_x = max(-limit, min(limit, self.pan_x))
        self.pan_y = max(-limit, min(limit, self.pan_y))

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return

        ref_w = float(getattr(self, "_master_width", 0) or self._image_width)
        ref_h = float(getattr(self, "_master_height", 0) or self._image_height)
        aspect = ref_w / float(max(1.0, ref_h)) if ref_w > 0 else 1.0
        screen_aspect = float(self.width()) / float(max(1.0, self.height())) if self.width() > 0 else 1.0
        base_w = self.height() * aspect if screen_aspect > aspect else self.width()
        zoom_100 = (ref_w / float(max(1.0, base_w))) if (ref_w > 0 and base_w > 0) else 1.0
        max_zoom = max(24.0, zoom_100 * 4.0)

        factor = 1.15 if delta > 0 else 0.87
        old_zoom = self.zoom
        new_zoom = max(0.2, min(max_zoom, self.zoom * factor))
        self.zoom = new_zoom

        # Mouse relative zoom
        mouse_norm_x = (event.position().x() / float(max(1, self.width())) - 0.5) * 2.0
        mouse_norm_y = ((self.height() - event.position().y()) / float(max(1, self.height())) - 0.5) * 2.0

        self.pan_x = mouse_norm_x - (mouse_norm_x - self.pan_x) * (new_zoom / old_zoom)
        self.pan_y = mouse_norm_y - (mouse_norm_y - self.pan_y) * (new_zoom / old_zoom)
        self._clamp_pan()

        if self.params.get("view_mode") == 1:
            self._clamp_split_to_image()
            self._update_badges()

        self.zoomChanged.emit(self.get_actual_zoom_ratio())
        self.update()
        event.accept()

    def _on_canvas_long_press_timeout(self):
        if not self._has_image or getattr(self, '_is_crop_editing', False):
            return
        self._is_canvas_long_pressing = True
        self._is_panning = False
        self.requestCompareStart.emit()

    def mousePressEvent(self, event):
        btn = event.button()
        if btn == Qt.MouseButton.LeftButton:
            # Check if clicking on split divider within visible image bounds
            if self._is_hovering_split_line(event.position()):
                self._is_split_dragging = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self._is_panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                if self._has_image and not getattr(self, '_is_crop_editing', False):
                    self._press_start_pos = event.position()
                    self._is_canvas_long_pressing = False
                    self._canvas_long_press_timer.start(220)
            self._last_mouse_pos = event.position()
            event.accept()
        elif btn == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._last_mouse_pos = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, '_is_canvas_long_pressing', False):
            event.accept()
            return

        if hasattr(self, '_canvas_long_press_timer') and self._canvas_long_press_timer.isActive() and self._press_start_pos is not None:
            delta = event.position() - self._press_start_pos
            if (abs(delta.x()) + abs(delta.y())) > 5.0:
                self._canvas_long_press_timer.stop()

        norm_x = event.position().x() / float(max(1, self.width()))

        if self._is_split_dragging:
            left, right, _, _ = self.get_image_bounds_norm()
            margin = max(0.01, 8.0 / float(max(1, self.width())))
            clamp_min = max(0.0, left) + margin
            clamp_max = min(1.0, right) - margin
            if clamp_min < clamp_max:
                self.params["split_x"] = max(clamp_min, min(clamp_max, norm_x))
            else:
                self.params["split_x"] = max(0.01, min(0.99, norm_x))
            self._update_badges()
            self.update()
            event.accept()
        elif self._is_panning:
            delta_px = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()

            # Convert pixel delta to normalized coordinate delta
            dx = (delta_px.x() / float(max(1, self.width()))) * 2.0
            dy = -(delta_px.y() / float(max(1, self.height()))) * 2.0

            self.pan_x += dx
            self.pan_y += dy
            self._clamp_pan()
            if self.params.get("view_mode") == 1:
                self._clamp_split_to_image()
                self._update_badges()
            self.update()
            event.accept()
        else:
            # Hover cursor check: split cursor only within visible image area
            if self._is_hovering_split_line(event.position()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        btn = event.button()
        if btn == Qt.MouseButton.LeftButton:
            if hasattr(self, '_canvas_long_press_timer') and self._canvas_long_press_timer.isActive():
                self._canvas_long_press_timer.stop()
            if getattr(self, '_is_canvas_long_pressing', False):
                self._is_canvas_long_pressing = False
                self.requestCompareEnd.emit()

        if btn in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._is_panning = False
            self._is_split_dragging = False
            if self._is_hovering_split_line(event.position()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if hasattr(self, '_canvas_long_press_timer') and self._canvas_long_press_timer.isActive():
            self._canvas_long_press_timer.stop()
        if getattr(self, '_is_canvas_long_pressing', False):
            self._is_canvas_long_pressing = False
            self.requestCompareEnd.emit()
        super().leaveEvent(event)

    def focusOutEvent(self, event):
        if hasattr(self, '_canvas_long_press_timer') and self._canvas_long_press_timer.isActive():
            self._canvas_long_press_timer.stop()
        if getattr(self, '_is_canvas_long_pressing', False):
            self._is_canvas_long_pressing = False
            self.requestCompareEnd.emit()
        super().focusOutEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Double-click on split line instantly resets divider to image center
            if self.params.get("view_mode") == 1 and self._is_hovering_split_line(event.position(), tolerance=0.035):
                self.reset_split_to_center()
                event.accept()
                return

            if abs(self.zoom - 1.0) < 0.15:
                self.set_zoom_100(animate=True, mouse_pos=event.position())
            else:
                self.fit_to_view(animate=True)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def render_offscreen(self, target_w, target_h, float_img_rgb=None, params_override=None, bit_depth=8, lut_override=None, is_16bit=None, density_override=None):
        """Render image at target resolution on GPU using an offscreen FBO.
        Returns uint8 or uint16 RGB numpy array (target_h, target_w, 3) or None on error.
        """
        # Oversize guard: GL textures cap at GL_MAX_TEXTURE_SIZE (typically 16384) and
        # there is no tiled renderer yet. Failing loudly beats a silently clipped or
        # black export; the tile renderer (with the 3-sigma overlap from
        # halation_model.halation_overlap_px) is the planned fix.
        if max(int(target_w), int(target_h)) > 16384:
            message = (f"{target_w}x{target_h} 超出 GPU 16384 px 纹理上限，"
                       f"已改用 CPU 导出（会明显变慢）")
            print(f"[DarkroomGLCanvas] render_offscreen: {message}")
            self.gpuFallback.emit(message)
            return None
        if is_16bit is not None:
            bit_depth = 16 if is_16bit else 8
        if not getattr(self, "_gl_initialized", False) or not self.isValid():
            try:
                self.grabFramebuffer()
            except Exception:
                pass
        if not getattr(self, "_gl_initialized", False):
            try:
                self.makeCurrent()
                self.initializeGL()
            except Exception:
                pass
            finally:
                # Release the context grabbed just for this bootstrap; Qt will make its own
                # current again when it first paints. Leaving it current kept a stale
                # context bound to the calling thread.
                try:
                    self.doneCurrent()
                except Exception:
                    pass
        if not getattr(self, "_gl_initialized", False) or not self.isValid():
            return None

        self.makeCurrent()
        orig_tex_id = self._image_tex_id
        temp_tex_id = None
        temp_lut_tex_id = None
        fbo = None
        _saved_film_density = self._film_density
        try:
            funcs = QOpenGLContext.currentContext().functions()

            # If a high-res image is provided, upload to a temporary texture
            if float_img_rgb is not None:
                src_h, src_w = float_img_rgb.shape[:2]
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                temp_tex_id = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, temp_tex_id)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                data = np.ascontiguousarray(float_img_rgb, dtype=np.float32)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, src_w, src_h, 0, GL.GL_RGB, GL.GL_FLOAT, data)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                active_tex_id = temp_tex_id
            else:
                active_tex_id = orig_tex_id
                src_w = float(getattr(self, "_master_width", 0) or getattr(self, "_image_width", 1) or 1)
                src_h = float(getattr(self, "_master_height", 0) or getattr(self, "_image_height", 1) or 1)

            if not active_tex_id:
                return None

            if getattr(self, "_pending_lut", None) is not None:
                self._do_upload_lut(self._pending_lut)
                self._pending_lut = None

            if lut_override is not None:
                size = lut_override.shape[0]
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                temp_lut_tex_id = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_3D, temp_lut_tex_id)
                GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
                data_lut = np.ascontiguousarray(lut_override, dtype=np.float32)
                GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGB16F, size, size, size, 0, GL.GL_RGB, GL.GL_FLOAT, data_lut)
                GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
                active_lut_tex_id = temp_lut_tex_id
            else:
                active_lut_tex_id = self._lut_tex_id

            # The density cube has to follow the LUT it was baked with. Swapping the LUT without
            # swapping the cube would index one bake's grain law with another bake's densities.
            if lut_override is not None:
                self._film_density = density_override

            # Merge params
            eff_params = dict(self.params)
            if params_override:
                eff_params.update(params_override)

            # Create FBO (32-bit float for bit_depth==32, 16-bit float for bit_depth==16)
            is_32bit = (int(bit_depth) == 32)
            is_16bit = (int(bit_depth) == 16)
            if is_32bit:
                from PySide6.QtOpenGL import QOpenGLFramebufferObjectFormat
                fbo_fmt = QOpenGLFramebufferObjectFormat()
                fbo_fmt.setInternalTextureFormat(GL.GL_RGBA32F)
                fbo = QOpenGLFramebufferObject(target_w, target_h, fbo_fmt)
            elif is_16bit:
                from PySide6.QtOpenGL import QOpenGLFramebufferObjectFormat
                fbo_fmt = QOpenGLFramebufferObjectFormat()
                fbo_fmt.setInternalTextureFormat(GL.GL_RGBA16F)
                fbo = QOpenGLFramebufferObject(target_w, target_h, fbo_fmt)
            else:
                fbo = QOpenGLFramebufferObject(target_w, target_h, QOpenGLFramebufferObject.Attachment.NoAttachment)

            if not fbo.isValid():
                return None

            fbo.bind()
            GL.glViewport(0, 0, target_w, target_h)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.shader_program.bind()
            self.vbo.bind()
            self.shader_program.enableAttributeArray(0)
            self.shader_program.setAttributeBuffer(0, GL.GL_FLOAT, 0, 2, 4 * 4)
            self.shader_program.enableAttributeArray(1)
            self.shader_program.setAttributeBuffer(1, GL.GL_FLOAT, 2 * 4, 2, 4 * 4)

            def u_1f(name, val):
                loc = self.shader_program.uniformLocation(name)
                if loc >= 0: funcs.glUniform1f(loc, float(val))

            def u_1i(name, val):
                loc = self.shader_program.uniformLocation(name)
                if loc >= 0: funcs.glUniform1i(loc, int(val))

            def u_2f(name, x, y):
                loc = self.shader_program.uniformLocation(name)
                if loc >= 0: funcs.glUniform2f(loc, float(x), float(y))

            def u_3f(name, x, y, z):
                loc = self.shader_program.uniformLocation(name)
                if loc >= 0: funcs.glUniform3f(loc, float(x), float(y), float(z))

            # Full-bleed 1:1 mapping: offscreen target is the image, zero letterboxing / zero black bars!
            u_1f("u_screen_aspect", 1.0)
            u_1f("u_image_aspect", 1.0)
            u_1f("u_native_aspect", float(src_w) / float(max(1, src_h)))
            u_2f("u_pan", 0.0, 0.0)
            u_1f("u_zoom", 1.0)

            u_1i("u_has_image", 1)
            u_1i("u_has_lut", 1 if (active_lut_tex_id and (self._has_lut or lut_override is not None)) else 0)
            if getattr(self, "_is_placeholder", False):
                is_bypass = 2
            else:
                is_bypass = 0
            u_1i("u_is_bypass", is_bypass)

            eff_exp = float(eff_params.get("exposure_ev", 0.0)) + float(eff_params.get("base_ev", 0.0))
            u_1f("u_base_ev", float(eff_params.get("base_ev", 0.0)))
            u_1f("u_highlights", float(eff_params.get("highlights", 0.0)))
            u_1f("u_shadows", float(eff_params.get("shadows", 0.0)))
            u_3f("u_scene_gain", *scene_gain(
                eff_exp,
                eff_params.get("color_temp", 5500.0),
                eff_params.get("base_temp", 5500.0),
                eff_params.get("tint", 1.0),
                eff_params.get("base_tint", 1.0)))
            film_id_off = eff_params.get("film_stock", getattr(self, "current_film_stock", "none"))
            is_pos_off = self.engine.is_positive_film(film_id_off) if hasattr(self.engine, 'is_positive_film') else False
            # Mirror paintGL: film-less bypass must drop every film-medium stage offscreen too,
            # otherwise thumbnails / histogram / GPU export would disagree with the viewport.
            is_film_none_off = str(film_id_off).lower() in ("none", "")
            paper_id_off = eff_params.get("paper_profile", getattr(self, "current_paper_stock", "none"))
            pmode_off = eff_params.get("print_mode", getattr(self, "current_print_mode", "optical"))
            is_scan_off = (paper_id_off == "none" or pmode_off == "scan")
            if is_pos_off or is_scan_off:
                u_3f("u_cmy", 0.0, 0.0, 0.0)
                u_1f("u_print_exposure", 1.0)
                u_1f("u_preflash", 0.0)
                u_1f("u_black_lift", float(eff_params.get("black_lift", 0.0)))
            else:
                u_3f("u_cmy", float(eff_params.get("enlarger_cyan", 0.0)), float(eff_params.get("enlarger_magenta", 0.0)), float(eff_params.get("enlarger_yellow", 0.0)))
                u_1f("u_print_exposure", float(eff_params.get("print_exposure", 1.0)))
                u_1f("u_preflash", float(eff_params.get("pre_flash", 0.0)))
                u_1f("u_black_lift", float(eff_params.get("black_lift", 0.0)))

            def u_4f(name, x, y, z, w):
                loc = self.shader_program.uniformLocation(name)
                if loc >= 0: funcs.glUniform4f(loc, float(x), float(y), float(z), float(w))

            # Lens correction in offscreen export
            off_lens = eff_params.get("lens", {})
            if isinstance(off_lens, dict) and off_lens.get("enabled", False):
                u_1i("u_lens_enabled", 1)
                u_3f("u_lens_distort", off_lens.get("k1", 0.0), off_lens.get("k2", 0.0), off_lens.get("k3", 0.0))
                u_1f("u_lens_dist_scale", off_lens.get("dist_scale", 0.0) or 0.0)
                u_1f("u_lens_fill_scale", off_lens.get("fill_scale", 0.0) or 0.0)
                u_2f("u_lens_vignette", off_lens.get("vignette_amount", 0.0), off_lens.get("vignette_midpoint", 50.0))
            else:
                u_1i("u_lens_enabled", 0)
                u_3f("u_lens_distort", 0.0, 0.0, 0.0)
                u_1f("u_lens_dist_scale", 0.0)
                u_1f("u_lens_fill_scale", 0.0)
                u_2f("u_lens_vignette", 0.0, 50.0)
            off_ca = 1 if (isinstance(off_lens, dict) and off_lens.get("remove_ca", False)) else 0
            u_1i("u_lens_ca_enabled", off_ca)
            off_ca_r = _ca_coeffs(off_lens, "ca_r_coeffs", "ca_kr") if isinstance(off_lens, dict) else (0.0, 0.0, 0.0)
            off_ca_b = _ca_coeffs(off_lens, "ca_b_coeffs", "ca_kb") if isinstance(off_lens, dict) else (0.0, 0.0, 0.0)
            u_3f("u_lens_ca_r", *off_ca_r)
            u_3f("u_lens_ca_b", *off_ca_b)
            u_1f("u_lens_ca_scale_r",
                 (off_lens.get("ca_r_scale", 0.0) or 0.0) if isinstance(off_lens, dict) else 0.0)
            u_1f("u_lens_ca_scale_b",
                 (off_lens.get("ca_b_scale", 0.0) or 0.0) if isinstance(off_lens, dict) else 0.0)

            # Crop in offscreen export
            off_crop = eff_params.get("crop", {})
            if isinstance(off_crop, dict) and off_crop.get("enabled", False):
                u_1i("u_crop_enabled", 1)
                u_4f("u_crop_rect", off_crop.get("x", 0.0), off_crop.get("y", 0.0), off_crop.get("w", 1.0), off_crop.get("h", 1.0))
                u_1f("u_crop_angle", math.radians(float(off_crop.get("angle", 0.0))))
                u_1i("u_crop_rot90", int(off_crop.get("rot_90", 0)) % 360)
                u_2f("u_crop_flip", 1.0 if off_crop.get("flip_h", False) else 0.0, 1.0 if off_crop.get("flip_v", False) else 0.0)
            else:
                u_1i("u_crop_enabled", 0)
                u_4f("u_crop_rect", 0.0, 0.0, 1.0, 1.0)
                u_1f("u_crop_angle", 0.0)
                u_1i("u_crop_rot90", 0)
                u_2f("u_crop_flip", 0.0, 0.0)
            u_1i("u_is_crop_editing", 0)

            # Halation physics (Official Spektrafilm Model)
            hal_preset = eff_params.get("_halation_preset")
            if not hal_preset:
                eng = getattr(self, "engine", None)
                if not eng and hasattr(self.parent(), "engine"):
                    eng = self.parent().engine
                if eng:
                    stock = eff_params.get("film_stock", "kodak_portra_400")
                    hal_preset = eng.get_film_halation_preset(stock)
            if not hal_preset:
                hal_preset = {"sigma_h": 65.0, "strength": [0.08, 0.02, 0.0]}
            strength = hal_preset.get("strength", [0.08, 0.02, 0.0])
            sigma_um = float(hal_preset.get("sigma_h", 65.0))

            hal_amt = 0.0 if is_film_none_off else float(eff_params.get("halation_amount", eff_params.get("halation", 1.0)))
            hal_scale = float(eff_params.get("halation_spatial_scale", 1.0))
            scatter_amt = 0.0 if is_film_none_off else float(eff_params.get("scatter_amount", 1.0))
            boost_ev = 0.0 if is_film_none_off else float(eff_params.get("boost_ev", eff_params.get("halation_boost", 0.0)))

            u_1f("u_halation", hal_amt)
            u_1f("u_hal_spatial_scale", hal_scale)
            u_1f("u_scatter_amount", scatter_amt)
            u_3f("u_hal_strength_rgb", float(strength[0]), float(strength[1]), float(strength[2]))
            u_1f("u_hal_sigma_um", sigma_um)

            # Grain: removed with the old octave model (see docs/grain-port-plan.md P0).
            u_1f("u_film_width_um", max(5.0, float(eff_params.get("film_format_mm", 35.0))) * 1000.0)

            diff_family = str(eff_params.get("diffusion_family", "none")).lower()
            family_map = {"none": 0, "black_pro_mist": 1, "glimmerglass": 2, "pro_mist": 3, "cinebloom": 4}
            u_1i("u_diffusion_type", family_map.get(diff_family, 0))
            u_1f("u_diffusion_strength", float(eff_params.get("diffusion_strength", 0.0)))
            u_1f("u_diffusion_warmth", float(eff_params.get("diffusion_warmth", 0.0)))
            _dp = diffusion_shader_profile(diff_family, eff_params.get("diffusion_strength", 0.0))
            u_1f("u_diffusion_ps", _dp["ps"])
            u_1f("u_diff_core_w", _dp["core_w"])
            u_1f("u_diff_halo_w", _dp["halo_w"])
            u_1f("u_diff_bloom_w", _dp["bloom_w"])
            u_1f("u_diff_halo_um", _dp["halo_um"])
            u_1f("u_diff_bloom_um", _dp["bloom_um"])
            u_1f("u_diff_warmth_base", _dp["warmth_base"])
            u_1f("u_diff_cutoff", _dp["cutoff"])

            _bk = self._lookup_lut_baked(lut_override) if lut_override is not None else (getattr(self, "_lut_baked", None) or self._lookup_lut_baked(None))
            u_1f("u_lut_morph_gamma", float(_bk.get("morph_gamma", 1.0)))
            u_1f("u_lut_dev_exh", float(_bk.get("developer_exhaustion", 0.0)))
            if is_pos_off or is_film_none_off:
                u_1f("u_morph_gamma", 1.0)
                u_1f("u_developer_exhaustion", 0.0)
            else:
                u_1f("u_morph_gamma", float(eff_params.get("morph_gamma", 1.0)))
                u_1f("u_developer_exhaustion", float(eff_params.get("developer_exhaustion", 0.0)))

            view_mode = eff_params.get("view_mode", 0) if (params_override and "view_mode" in params_override) else 0
            split_x = float(eff_params.get("split_x", 0.5))
            u_1f("u_split_x", split_x)
            u_1i("u_view_mode", view_mode)
            u_2f("u_viewport_size", float(target_w), float(target_h))

            # --- separable-Gaussian halation (ui/gl_halation.py, plan P2) -------------
            # Runs before the main draw and hands its result to the shader as the scene
            # texture. The halo is then a real Gaussian evaluated on the output grid,
            # instead of 16 taps reading whatever mip level happened to be handy -- which
            # is what made the halo shape depend on the preview size.
            prepared_tex = None
            diff_strength = float(eff_params.get("diffusion_strength", 0.0))
            _hal_elig = ((hal_amt > 0.001 or scatter_amt > 0.001
                          or (diff_family != "none" and diff_strength > 0.001))
                         and not is_film_none_off)
            if _hal_elig:
                try:
                    eng_off = getattr(self, "engine", None)
                    if eng_off is None and hasattr(self.parent(), "engine"):
                        eng_off = self.parent().engine
                    if eng_off is not None and hasattr(eng_off, "get_halation_params"):
                        hp = eng_off.get_halation_params(eff_params.get("film_stock", "none"))
                        fmt_mm = float(eff_params.get("film_format_mm", 35.0))
                        um_per_px = (max(5.0, fmt_mm) * 1000.0) / float(max(target_w, target_h))
                        sc = max(0.2, float(hal_scale))
                        sig_px = [float(s_) * sc / um_per_px for s_ in hp.sigma_h_um]
                        # The halation blur is only worth decimating past ~6 px; below that
                        # the tap path is equivalent. Scatter, on the other hand, is always
                        # worth doing here: the shader's version is an 8-tap ring at 4x the
                        # right radius, and it is what the CPU/GPU gap was made of.
                        want_hal = max(sig_px) * (3.0 ** 0.5) > 6.0
                        scat_arg = None
                        if scatter_amt > 0.001:
                            scat_arg = {
                                "core": [max(0.5, float(hp.scatter_core_um[c]) * sc / um_per_px)
                                         for c in range(3)],
                                "tail": [max(1.0, float(hp.scatter_tail_um[c]) * sc / um_per_px)
                                         for c in range(3)],
                                "w_tail": [float(v) for v in hp.scatter_tail_weight],
                                "eff": min(0.5, float(scatter_amt) * 0.06),
                                "amount": float(scatter_amt),
                            }
                        diff_arg = None
                        if diff_family != "none" and diff_strength > 0.001:
                            from core.diffusion_model import resolve_diffusion_mix
                            # The CPU diffusion (apply_diffusion_np) sizes its pixels as
                            # 36000/max(W,H) -- a 36 mm frame, NOT the halation chain's
                            # fmt_mm*1000. The 2.9% sigma mismatch on a 500 px bloom
                            # component shears the veil profile and costs ~15 dB.
                            diff_px_um = 36000.0 / float(max(target_w, target_h))
                            # spatial_scale: the CPU diffusion path (apply_diffusion_np)
                            # runs at 1.0 -- the halation spatial-scale dial must NOT
                            # leak into the diffusion sigmas or the two ends disagree
                            # the moment the user touches that slider.
                            diff_arg = resolve_diffusion_mix(
                                diff_family, float(diff_strength), 1.0,
                                diff_px_um,
                                float(eff_params.get("diffusion_warmth", 0.0)))
                        if want_hal or scat_arg or diff_arg:
                            if self._halation_chain is None:
                                from .gl_halation import GLHalationChain
                                self._halation_chain = GLHalationChain()
                            _g = scene_gain(
                                eff_exp,
                                eff_params.get("color_temp", 5500.0),
                                eff_params.get("base_temp", 5500.0),
                                eff_params.get("tint", 1.0),
                                eff_params.get("base_tint", 1.0))
                            prepared_tex = self._halation_chain.run(
                                int(active_tex_id), int(target_w), int(target_h), _g,
                                sig_px, hp.bounce_decay, hp.strength,
                                float(hal_amt) if want_hal else 0.0,
                                hs=(float(eff_params.get("highlights", 0.0)),
                                    float(eff_params.get("shadows", 0.0))),
                                scatter=scat_arg,
                                boost={"ev": float(boost_ev),
                                       "range": float(hp.boost_range),
                                       "protect": float(hp.protect_ev)},
                                diffusion=diff_arg)
                except Exception as _ex_hal_chain:
                    print(f"[DarkroomGLCanvas] halation chain skipped: {_ex_hal_chain}")
                    prepared_tex = None

            if prepared_tex:
                # Whatever the chain evaluated must not be evaluated again by the shader.
                if getattr(self._halation_chain, "last_did_halation", True):
                    u_1f("u_halation", 0.0)
                if getattr(self._halation_chain, "last_did_scatter", False):
                    u_1f("u_scatter_amount", 0.0)
                if getattr(self._halation_chain, "last_did_diffusion", False):
                    u_1f("u_diffusion_ps", 0.0)
                u_1f("u_scene_prepared", 1.0)
            else:
                u_1f("u_scene_prepared", 0.0)

            funcs.glActiveTexture(0x84C0)  # GL_TEXTURE0
            GL.glBindTexture(GL.GL_TEXTURE_2D, prepared_tex if prepared_tex else active_tex_id)
            u_1i("u_image", 0)

            if active_lut_tex_id:
                funcs.glActiveTexture(0x84C1)  # GL_TEXTURE1
                GL.glBindTexture(GL.GL_TEXTURE_3D, active_lut_tex_id)
                u_1i("u_lut", 1)

            # Film grain (units 3 and 4): the offscreen read is the same frame the canvas shows,
            # so it must carry the same film -- and at the offscreen resolution's own pitch.
            self._bind_grain(u_1f, u_1i, u_2f, u_3f, (int(src_w), int(src_h)))

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

            self.shader_program.disableAttributeArray(0)
            self.shader_program.disableAttributeArray(1)
            self.vbo.release()
            self.shader_program.release()

            # Read pixels from FBO (target_w, target_h)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            if is_32bit:
                raw_bytes = GL.glReadPixels(0, 0, target_w, target_h, GL.GL_RGBA, GL.GL_FLOAT)
                fbo.release()
                rgba = np.frombuffer(raw_bytes, dtype=np.float32).reshape((target_h, target_w, 4))
                try:
                    import cv2
                    img_arr = cv2.flip(cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB), 0)
                except Exception:
                    img_arr = np.ascontiguousarray(np.flipud(rgba[..., :3]))
            elif is_16bit:
                raw_bytes = GL.glReadPixels(0, 0, target_w, target_h, GL.GL_RGBA, GL.GL_UNSIGNED_SHORT)
                fbo.release()
                rgba = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((target_h, target_w, 4))
                try:
                    import cv2
                    img_arr = cv2.flip(cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB), 0)
                except Exception:
                    img_arr = np.ascontiguousarray(np.flipud(rgba[..., :3]))
            else:
                raw_bytes = GL.glReadPixels(0, 0, target_w, target_h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
                fbo.release()
                rgba = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((target_h, target_w, 4))
                try:
                    import cv2
                    img_arr = cv2.flip(cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB), 0)
                except Exception:
                    img_arr = np.ascontiguousarray(np.flipud(rgba[..., :3]))

            return img_arr

        except Exception as e:
            print(f"[DarkroomGLCanvas] render_offscreen error: {e}")
            return None
        finally:
            self._film_density = _saved_film_density
            if temp_tex_id:
                try:
                    GL.glDeleteTextures([temp_tex_id])
                except Exception:
                    pass
            if temp_lut_tex_id:
                try:
                    GL.glDeleteTextures([temp_lut_tex_id])
                except Exception:
                    pass
            if fbo:
                del fbo
            # Restore viewport
            GL.glViewport(0, 0, max(1, self.width()), max(1, self.height()))
            self.doneCurrent()


CanvasViewport = DarkroomGLCanvas
