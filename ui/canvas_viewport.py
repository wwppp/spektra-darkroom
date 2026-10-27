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
from core.darkroom_math import grain_scale_px, grain_colour_rho
from core import grain_model as grain_model_ref
from core.grain_model import octave_weights as grain_octave_weights
from OpenGL import GL


def grain_derive_spec(params, stock_strength, stock_layer_weights, stock_size_mm,
                      fmt_mm, long_edge_px, grain_preset=None):
    """Build the shared GrainSpec for a shader draw.

    Everything the shader needs about grain amplitude, octave mixing, layer correlation and
    the density curve comes from core/grain_model.py; this helper is only the adapter between
    the parameter dict and the model (P1/P2).

    ``grain_preset`` is the stock's crystal_grain dict: it selects the stock's own density law
    and its declared luma correlation.  Without it the P0 bell and the global correlation target
    are used, which is the behaviour every stock had before the law was wired up.
    """
    return grain_model_ref.derive(
        grain_slider=params.get("grain", 0.4),
        stock_strength=stock_strength,
        layer_weights=stock_layer_weights,
        cloud_blur=params.get("grain_cloud_blur", 1.0),
        rho=grain_colour_rho(stock_strength, float(params.get("grain_colour", 1.0))),
        fmt_mm=fmt_mm,
        long_edge_px=long_edge_px,
        grain_size=params.get("grain_size", 1.0),
        stock_size_mm=stock_size_mm,
        grain_colour=float(params.get("grain_colour", 1.0)),
        density_preset=grain_preset,
    )


def _bind_density_uniforms(u_1f):
    """Density-response constants, from the single definition in core/grain_model.py."""
    u_1f("u_grain_dens_gain", grain_model_ref.DENS_GAIN)
    u_1f("u_grain_dens_floor", grain_model_ref.DENS_FLOOR)
    u_1f("u_grain_dens_peak", grain_model_ref.DENS_PEAK)
    u_1f("u_grain_dens_wtoe", grain_model_ref.DENS_W_TOE)
    u_1f("u_grain_dens_wshoulder", grain_model_ref.DENS_W_SHOULDER)
    u_1f("u_grain_dens_dmax", grain_model_ref.DENS_DMAX)
    u_1f("u_grain_dens_wsat", grain_model_ref.DENS_W_SAT)
    u_1f("u_grain_density_scale", 1.0 / grain_model_ref.DENSITY_LUT_MAX)


# The density-law ramp lives on its own texture unit, clear of the image (0) and the 3D LUT (1).
GRAIN_DENSITY_TEX_UNIT = 2


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

uniform float u_exposure;
uniform float u_temp;
uniform float u_tint;
uniform float u_base_temp;
uniform float u_base_tint;
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
uniform float u_halation_boost;     // boost_ev (0.0 = physical default)
uniform vec3  u_hal_strength_rgb;   // profile-driven RGB preset
uniform float u_hal_sigma_um;       // profile-driven first bounce sigma (um)

// Grain physics & format
uniform float u_grain;
uniform float u_grain_scale;
uniform float u_grain_size;
uniform float u_grain_blur;
uniform float u_grain_colour;
// Derived once on the CPU by core/grain_model.py and shared with the CPU export renderer:
// per-octave mixing weights.  Keeping these out of the shader is what stops preview and
// export drifting apart (P1).
uniform float u_grain_w_fine;
uniform float u_grain_w_cloud;
uniform float u_grain_w_mottle;
// Density response (bell in negative optical density) and the physical amplitude law.
// Values come from core/grain_model.py so the CPU export uses exactly the same numbers (P2).
uniform float u_grain_amplitude;
uniform float u_grain_dens_gain;
uniform float u_grain_dens_floor;
uniform float u_grain_dens_peak;
uniform float u_grain_dens_wtoe;
uniform float u_grain_dens_wshoulder;
uniform float u_grain_dens_dmax;
uniform float u_grain_dens_wsat;
// Per-stock density law, baked by core/grain_model.bake_density_lut() into a 512x1 RGB ramp over
// D = 0..4 and sampled here.  Keeping the law in a texture rather than in GLSL means a stock's
// law is implemented exactly once (in Python) and both renderers read the same numbers.
// u_grain_density_enable == 0 falls back to the P0 bell above, so a missing texture can never
// blank the grain.
uniform sampler2D u_grain_density_lut;
uniform float u_grain_density_scale;
uniform float u_grain_density_enable;
uniform float u_stock_grain_size;
uniform float u_stock_grain_strength;
uniform vec3  u_stock_layer_weights;
uniform vec2 u_master_resolution;

// Diffusion filter
uniform int u_diffusion_type;
uniform float u_diffusion_strength;
uniform float u_diffusion_warmth;

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
uniform float u_grain_rho;
uniform float u_film_width_um;

// Lens Optical & Geometric Correction
uniform int u_lens_enabled;
uniform vec3 u_lens_distort;
uniform vec2 u_lens_vignette;
uniform int u_lens_ca_enabled;
uniform vec2 u_lens_ca_params;

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

// 32-bit Integer PCG (Permuted Congruential Generator) Hash
// Cryptographic quality pseudo-randomness with ZERO periodic banding or grid artifacts
uint pcg_hash(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v.x ^ v.y;
}

float pcg_float(uvec2 p) {
    return float(pcg_hash(p)) * (1.0 / 4294967296.0);
}

// Fast 3-channel decorrelated pseudo-random vector from 32-bit PCG hash
vec3 pcg_vec3(uvec2 p) {
    uint h = pcg_hash(p);
    float r = float(h & 0x3FFu) * (1.0 / 1023.0);
    float g = float((h >> 10u) & 0x3FFu) * (1.0 / 1023.0);
    float b = float((h >> 20u) & 0x3FFu) * (1.0 / 1023.0);
    return vec3(r, g, b);
}

// Smooth continuous organic dye cloud noise with Hermite interpolation
float smooth_cloud_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 s = f * f * (3.0 - 2.0 * f);
    
    uvec2 u = uvec2(ivec2(i));
    float n00 = pcg_float(u);
    float n10 = pcg_float(u + uvec2(1u, 0u));
    float n01 = pcg_float(u + uvec2(0u, 1u));
    float n11 = pcg_float(u + uvec2(1u, 1u));
    
    float nx0 = mix(n00, n10, s.x);
    float nx1 = mix(n01, n11, s.x);
    return mix(nx0, nx1, s.y);
}

// Multi-channel organic dye cloud noise vector with Hermite interpolation
vec3 smooth_cloud_noise_rgb(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 s = f * f * (3.0 - 2.0 * f);
    
    uvec2 u = uvec2(ivec2(i));
    vec3 n00 = pcg_vec3(u);
    vec3 n10 = pcg_vec3(u + uvec2(1u, 0u));
    vec3 n01 = pcg_vec3(u + uvec2(0u, 1u));
    vec3 n11 = pcg_vec3(u + uvec2(1u, 1u));
    
    vec3 nx0 = mix(n00, n10, s.x);
    vec3 nx1 = mix(n01, n11, s.x);
    return mix(nx0, nx1, s.y);
}

// Accurate Linear ProPhoto RGB to Display sRGB color space conversion
vec3 prophoto_to_srgb(vec3 c) {
    vec3 lin;
    lin.r =  2.03649172 * c.r - 0.73759065 * c.g - 0.29925987 * c.b;
    lin.g = -0.22571798 * c.r + 1.22317653 * c.g + 0.00272522 * c.b;
    lin.b = -0.01054513 * c.r - 0.13487985 * c.g + 1.14521015 * c.b;
    lin = max(vec3(0.0), lin);
    // Smooth highlight compression knee before display transfer
    vec3 over = max(vec3(0.0), lin - vec3(0.8));
    lin = min(lin, vec3(0.8)) + 0.2 * (over / (0.2 + over * 0.5));
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

// Hermite soft-knee thresholding: smoothly blends from zero to highlight glow
float soft_knee_hl(float luma, float thr, float knee) {
    float lo = max(0.001, thr - knee);
    float hi = thr + knee;
    float t = clamp((luma - lo) / max(0.0001, (hi - lo)), 0.0, 1.0);
    float smooth_t = t * t * (3.0 - 2.0 * t);
    return smooth_t * max(0.0, luma - lo);
}

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

// Computes scene-linear radiance accounting for exposure, white balance tint, and highlights/shadows shaping
vec3 compute_scene_radiance(vec3 raw_c) {
    vec3 col = raw_c * exp2(u_exposure);
    float b_temp = (u_base_temp > 1000.0) ? u_base_temp : 5500.0;
    float b_tint = (u_base_tint > 0.1) ? u_base_tint : 1.0;
    // Exponential, luminance-preserving white balance (mirrors core/darkroom_math.wb_gains)
    float wb_s = log2(max(1000.0, u_temp) / b_temp);
    vec3 wb_g = vec3(exp2(0.40 * wb_s), u_tint / b_tint, exp2(-0.40 * wb_s));
    wb_g /= max(1e-4, dot(wb_g, vec3(0.2880, 0.7119, 0.0001)));
    col *= wb_g;

    if (abs(u_highlights) > 0.001 || abs(u_shadows) > 0.001) {
        float y_lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
        if (y_lum > 0.00001) {
            float z = log2(y_lum / 0.18);
            float w_sh = clamp(-z * 0.25, 0.0, 1.0);
            w_sh = w_sh * w_sh * (3.0 - 2.0 * w_sh);
            float w_hl = clamp(z * 0.25, 0.0, 1.0);
            w_hl = w_hl * w_hl * (3.0 - 2.0 * w_hl);
            float delta_z = (u_shadows * 0.02) * w_sh + (u_highlights * 0.02) * w_hl;
            float y_target = 0.18 * exp2(z + delta_z);
            col *= (y_target / y_lum);
        }
    }
    return col;
}

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
        
        // Auto-scale factor to prevent edge clamping / black borders
        float corner_k = u_lens_distort.x + u_lens_distort.y + u_lens_distort.z;
        float scale = (corner_k > 0.0) ? (1.0 / (1.0 + corner_k)) : 1.0;
        sample_uv = 0.5 + (delta * r_dist) * scale;
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
    if (u_lens_ca_enabled == 1 && (abs(u_lens_ca_params.x) > 0.00001 || abs(u_lens_ca_params.y) > 0.00001)) {
        vec2 lp_ca = (sample_uv - 0.5);
        float r2_ca = dot(lp_ca, lp_ca);
        vec2 uv_r = clamp(sample_uv + lp_ca * (u_lens_ca_params.x * r2_ca), half_texel, vec2(1.0) - half_texel);
        vec2 uv_b = clamp(sample_uv + lp_ca * (u_lens_ca_params.y * r2_ca), half_texel, vec2(1.0) - half_texel);
        orig_rgb.r = textureGrad(u_image, uv_r, d_sample_dx, d_sample_dy).r;
        orig_rgb.b = textureGrad(u_image, uv_b, d_sample_dx, d_sample_dy).b;
    }

    // 2. High-Contrast Edge Defringing (Removes purple, green, cyan & red color fringing along contrast boundaries)
    if (u_lens_ca_enabled == 1) {
        vec2 texel_step = 1.0 / tex_dim;
        vec3 col_px = textureGrad(u_image, sample_uv + vec2(texel_step.x, 0.0), d_sample_dx, d_sample_dy).rgb;
        vec3 col_py = textureGrad(u_image, sample_uv + vec2(0.0, texel_step.y), d_sample_dx, d_sample_dy).rgb;

        float luma_c = 0.299 * orig_rgb.r + 0.587 * orig_rgb.g + 0.114 * orig_rgb.b;
        float luma_x = 0.299 * col_px.r + 0.587 * col_px.g + 0.114 * col_px.b;
        float luma_y = 0.299 * col_py.r + 0.587 * col_py.g + 0.114 * col_py.b;

        float edge_val = max(abs(luma_c - luma_x), abs(luma_c - luma_y));
        float edge_mask = clamp(edge_val * 15.0, 0.0, 1.0);

        if (edge_mask > 0.01) {
            float r = orig_rgb.r;
            float g = orig_rgb.g;
            float b = orig_rgb.b;

            float purple = max(0.0, min(r, b) - g);
            float green = max(0.0, g - max(r, b));
            float cyan = max(0.0, b - max(r, g));
            float red = max(0.0, r - max(g, b));

            orig_rgb.r = max(0.0, r - (purple * 0.90 + red * 0.90) * edge_mask);
            orig_rgb.g = max(0.0, g - (green * 0.90) * edge_mask);
            orig_rgb.b = max(0.0, b - (purple * 0.90 + cyan * 0.90) * edge_mask);
        }
    }

    // Apply Lens Vignette compensation/darkening if enabled
    if (u_lens_enabled == 1 && abs(u_lens_vignette.x) > 0.001) {
        float r_norm = length((sample_uv - 0.5) * 2.0) / 1.41421356;
        float v_exp = mix(1.2, 4.0, 1.0 - clamp(u_lens_vignette.y / 100.0, 0.1, 0.9));
        float vig_gain = max(0.0, 1.0 + u_lens_vignette.x * pow(clamp(r_norm, 0.0, 1.0), v_exp));
        orig_rgb *= vig_gain;
    }

    // Accurate untouched base display color (Before / 原片)
    vec3 before_color = (u_is_bypass == 2) ? clamp(orig_rgb, 0.0, 1.0) : clamp(prophoto_to_srgb(orig_rgb), 0.0, 1.0);

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

    // 1: Pure Bypass Mode: When no film is selected (film_stock == "none")
    if (u_is_bypass == 1) {
        fragColor = vec4(before_color, 1.0);
        return;
    }
    
    // 1 & 2. Exposure, White Balance, and Highlights & Shadows Tone Shaping
    vec3 linear_rgb = compute_scene_radiance(orig_rgb);
    
    // 3. Optical Diffusion Filter (Physical Convolution Bloom with Vogel Spiral & Trilinear LOD)
    int eff_diff_type = u_diffusion_type;
    if (eff_diff_type == 0 && u_diffusion_strength > 0.001) {
        eff_diff_type = 1; // Default to Black Pro-Mist if strength > 0
    }
    if (eff_diff_type > 0 && u_diffusion_strength > 0.001) {
        // Physical radius scaled by strength
        float base_rad = 0.012 * max(0.20, u_diffusion_strength);
        float thr = 0.35;
        float knee = 0.18;
        float halo_w = 0.65;
        float intensity = 1.05;

        // Align per-family optical profile to spektrafilm physical specs
        if (eff_diff_type == 1) {
            // Black Pro-Mist: Punchy core + halo, fast falloff, deep clean blacks intact
            thr = 0.36;
            knee = 0.16;
            base_rad *= 1.0;
            intensity = 1.05;
            halo_w = 0.65;
        } else if (eff_diff_type == 2) {
            // Glimmerglass: Ultra-clean micro-bloom, crisp highlights, minimal veiling
            thr = 0.42;
            knee = 0.14;
            base_rad *= 0.75;
            intensity = 0.85;
            halo_w = 0.0;
        } else if (eff_diff_type == 3) {
            // Pro-Mist: Classic atmospheric pastel haze, balanced halo & veil
            thr = 0.24;
            knee = 0.20;
            base_rad *= 1.45;
            intensity = 1.25;
            halo_w = 0.40;
        } else if (eff_diff_type == 4) {
            // CineBloom: Wide cinematic atmospheric flare, heavy tail
            thr = 0.20;
            knee = 0.22;
            base_rad *= 1.85;
            intensity = 1.40;
            halo_w = 0.85;
        }

        vec2 tex_sz = max(vec2(100.0), vec2(textureSize(u_image, 0)));
        // Correct aspect ratio so the bloom spread is physically isotropic (circular)
        float aspect = tex_sz.x / tex_sz.y;
        vec2 spread = vec2(base_rad, base_rad * aspect);

        vec3 bloom_accum = vec3(0.0);
        float total_weight = 0.0;

        // Trilinear Mipmap LOD level: higher LOD for outer taps guarantees continuous pre-filtering
        float lod_base = clamp(log2(base_rad * max(tex_sz.x, tex_sz.y) * 0.15), 0.0, 4.0);

        for (int i = 0; i < 16; i++) {
            vec2 offset = VOGEL_TAPS[i];
            float r_norm = length(offset);
            vec2 tap_uv = clamp(sample_uv + offset * spread, 0.0, 1.0);
            
            // Sample with hardware trilinear mipmap pre-filtering
            float tap_lod = clamp(lod_base + r_norm * 1.8, 0.0, 4.5);
            vec3 tap_rgb = compute_scene_radiance(textureLod(u_image, tap_uv, tap_lod).rgb);
            
            float tap_luma = dot(tap_rgb, vec3(0.2126, 0.7152, 0.0722));
            float hl = soft_knee_hl(tap_luma, thr, knee);

            // Isotropic exponential falloff weighting
            float w = exp(-r_norm * 1.8);
            bloom_accum += tap_rgb * (hl * w);
            total_weight += w;
        }
        bloom_accum /= max(0.0001, total_weight);

        // Physical halo warmth tint (outer rim warm yellow-orange, core cool)
        float eff_warmth = clamp(halo_w + u_diffusion_warmth, -1.5, 1.5);
        vec3 diff_tint = vec3(
            1.0 + eff_warmth * 0.35,
            1.0 + eff_warmth * 0.08,
            1.0 - eff_warmth * 0.35
        );

        linear_rgb += bloom_accum * (u_diffusion_strength * intensity) * diff_tint;

        // Classic Pro-Mist & CineBloom lift deep shadows slightly (veiling glare)
        if (eff_diff_type == 3) {
            linear_rgb = mix(linear_rgb, linear_rgb + vec3(0.015 * u_diffusion_strength), 0.20);
        } else if (eff_diff_type == 4) {
            linear_rgb = mix(linear_rgb, linear_rgb + vec3(0.022 * u_diffusion_strength), 0.25);
        }
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
        if (u_halation > 0.001) {
            vec3 a_tot = u_hal_strength_rgb * u_halation;
            vec3 hal_accum = vec3(0.0);
            float hal_total_w = 0.0;

            for (int b = 1; b <= 3; b++) {
                float bounce_w = pow(0.5, float(b - 1));
                float bounce_rad_mult = sqrt(float(b));
                float lod_b = clamp(log2(base_hal_rad * bounce_rad_mult * max_dim * 0.4), 0.0, 4.5);

                for (int i = 0; i < 16; i++) {
                    vec2 offset = VOGEL_TAPS[i] * bounce_rad_mult;
                    float r_norm = length(offset);
                    vec2 tap_uv = clamp(sample_uv + offset * hal_spread, 0.0, 1.0);
                    vec3 tap_rgb = compute_scene_radiance(textureLod(u_image, tap_uv, lod_b).rgb);
                    
                    if (u_halation_boost > 0.001) {
                        float tap_luma = dot(tap_rgb, vec3(0.2126, 0.7152, 0.0722));
                        float boost_f = 1.0 + max(0.0, tap_luma - 0.7) * (u_halation_boost * 0.6);
                        tap_rgb *= boost_f;
                    }

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

    // 3.5. Silver Halide Organic Film Grain & Multi-Octave Dye Cloud Mottle (Pre-LUT Transmittance Space)
    if (u_grain > 0.01) {
        // Golden-ratio irrational rotation matrix (breaks all raster / coordinate grid alignment)
        const mat2 rot = mat2(0.9114, -0.4115, 0.4115, 0.9114);
        float base_stock_size = (u_stock_grain_size > 0.0001) ? u_stock_grain_size : 0.005;
        float scale = max(0.20, u_grain_scale * max(0.2, u_grain_size) * (base_stock_size / 0.005));

        // Anchor grain coordinate to image/film master pixel space (WYSIWYG preview & export)
        vec2 master_res = (u_master_resolution.x > 1.0) ? u_master_resolution : vec2(textureSize(u_image, 0));
        vec2 film_coord = sample_uv * master_res;
        vec2 gcoord = rot * (film_coord / scale);

        // Anti-aliasing / Central Limit Theorem screen downsampling integration
        // Averages micro-crystals across screen pixel footprint:
        // - At 100%/91% zoom (footprint ~ 1.0): 100% full tactile micro-crystal bite & dye clouds.
        // - At Fit view (15% zoom, footprint ~ 6.7): micro-crystals average down by CLT ~ 1/sqrt(N),
        //   producing smooth, velvety, delicate 35mm negative film texture instead of coarse static!
        vec2 g_dx = rot * (d_sample_dx * master_res / scale);
        vec2 g_dy = rot * (d_sample_dy * master_res / scale);
        float footprint = max(length(g_dx), length(g_dy));
        // Box-filter averaging of spatially correlated grain over a footprint of f grain cells
        // reduces its std ~ 1/f (2D: f^2 cells, sqrt averaging) - matches downscaled export.
        float grain_att = 1.0 / max(1.0, footprint);

        // Multi-scale organic dye cloud clusters & micro-crystals:
        // 1. High-frequency micro-crystals (~1.2 master px, ~7 um) with continuous subpixel continuity
        vec3 fine_rgb = (smooth_cloud_noise_rgb(gcoord * 0.75) - vec3(0.5)) * 2.0;
        // 2. Mid-frequency dye clouds (~1.8 master px, ~11 um)
        vec3 cloud_rgb = (smooth_cloud_noise_rgb(gcoord * 0.45 + vec2(43.1, 89.4)) - vec3(0.5)) * 1.8;
        // 3. Low-frequency organic dye mottle (~3.5 master px, ~21 um)
        vec3 mottle_rgb = (smooth_cloud_noise_rgb(gcoord * 0.22 + vec2(17.3, 31.7)) - vec3(0.5)) * 1.6;

        // Additive physical mixture governed by "染料云团柔化 (u_grain_blur)".
        // Weights come from core/grain_model.octave_weights() so the CPU export renderer uses
        // exactly the same numbers (P1: single source of truth).
        float w_fine = u_grain_w_fine;
        float w_cloud = u_grain_w_cloud;
        float w_mottle = u_grain_w_mottle;
        // Per-layer (chroma) component: dye-cloud mottle is spatially coarser than the luminance
        // grain, so the independent part excludes the fine micro-crystal octave (no RGB pixel speckle).
        vec3 raw_noise = (cloud_rgb * w_cloud + mottle_rgb * w_mottle) * 1.15;

        // Shared luminance grain stream across coupled emulsion layers
        float shared_fine = (smooth_cloud_noise(gcoord * 0.75 + vec2(53.2, 19.8)) - 0.5) * 2.0;
        float shared_cloud = (smooth_cloud_noise(gcoord * 0.45 + vec2(11.7, 73.2)) - 0.5) * 1.8;
        float shared_mottle = (smooth_cloud_noise(gcoord * 0.22 + vec2(91.2, 53.4)) - 0.5) * 1.6;
        float shared_noise = shared_fine * w_fine + shared_cloud * w_cloud + shared_mottle * w_mottle;

        // Coupled C-41 emulsion layer correlation (organic dye clouds vs digital chroma noise).
        // u_grain_rho is not a literal: core/grain_model.rho_for_target_correlation() solves it
        // so the *measured* mean inter-channel correlation lands on the calibrated target
        // (0.79 at colour_grain = 1.0, from the P0 reference), whatever the stock's layer
        // weights are. colour_grain = 0 (B&W) collapses rho to 1.0 (pure monochrome grain).
        float rho = clamp(u_grain_rho, 0.0, 1.0);
        // Apply film stock layer weights (e.g. [0.70, 1.0, 1.35] or [0.90, 1.0, 2.71]).
        // The shared (luminance) stream uses the mean weight so it stays neutral; the per-layer
        // spread only shapes the independent chroma mottle (avoids blue/yellow pixel speckle).
        float lw_mean = dot(u_stock_layer_weights, vec3(1.0 / 3.0));
        // u_grain_amplitude is the physical amplitude law from core/grain_model.py: per-pixel
        // density noise is proportional to the long edge (fewer crystals per pixel as sampling
        // gets finer). It is 1.0 at the calibration size, so the preview is unchanged there.
        vec3 layer_grain = (sqrt(1.0 - rho) * raw_noise * u_stock_layer_weights
                          + sqrt(rho) * vec3(shared_noise * lw_mean)) * u_grain_amplitude;

        // Stock strength & user slider modulation
        float base_stock_strength = (u_stock_grain_strength > 0.0001) ? u_stock_grain_strength : 0.010;
        float eff_strength = (u_grain / 0.40) * (base_stock_strength / 0.010);

        // Granularity vs negative optical density. linear_rgb is read as the negative's
        // transmittance, so D = -log10(lin).  Two paths, and the choice is made on the CPU:
        //   u_grain_density_enable == 1: the stock's own law, baked by
        //     core/grain_model.bake_density_lut() and fetched from a 512x1 ramp.  The table is
        //     normalised so that its value at the stock's reading density is exactly
        //     u_grain_dens_gain, which is why adopting a law cannot change overall amplitude.
        //   u_grain_density_enable == 0: the P0 bell, i.e. the global curve fitted to the
        //     particle-model reference.  Used when no table could be uploaded.
        vec3 lin_c = clamp(linear_rgb, vec3(1e-4), vec3(1.0));
        vec3 dens_D = -log(lin_c) * (1.0 / 2.302585092994046);
        vec3 dens_mod;
        if (u_grain_density_enable > 0.5) {
            // One fetch per record; the ramp is linear-filtered, and the CPU interpolates the
            // same table the same way, so preview and export agree on the law.
            dens_mod = vec3(
                textureLod(u_grain_density_lut, vec2(dens_D.r * u_grain_density_scale, 0.5), 0.0).r,
                textureLod(u_grain_density_lut, vec2(dens_D.g * u_grain_density_scale, 0.5), 0.0).g,
                textureLod(u_grain_density_lut, vec2(dens_D.b * u_grain_density_scale, 0.5), 0.0).b);
        } else {
            vec3 dens_z = dens_D - vec3(u_grain_dens_peak);
            vec3 dens_w = mix(vec3(u_grain_dens_wtoe), vec3(u_grain_dens_wshoulder),
                              step(vec3(0.0), dens_z));
            vec3 dens_bell = vec3(u_grain_dens_floor)
                           + (vec3(1.0) - vec3(u_grain_dens_floor))
                             * exp(-0.5 * (dens_z / dens_w) * (dens_z / dens_w));
            // Film saturation: fade to zero just below the stock's Dmax. Above it the particle-model
            // reference returns a constant saturated field (every crystal developed), so extending
            // the Gaussian there would put grain into fully clipped highlights.
            vec3 dens_t = clamp((dens_D - vec3(u_grain_dens_dmax - u_grain_dens_wsat))
                                / vec3(u_grain_dens_wsat), 0.0, 1.0);
            vec3 dens_roll = vec3(1.0) - (vec3(3.0) * dens_t * dens_t - vec3(2.0) * dens_t * dens_t * dens_t);
            dens_mod = vec3(u_grain_dens_gain) * dens_bell * dens_roll;
        }

        // Transmittance modulation: T = T_in * 10^(-dD) = T_in * exp(-dD * 2.3025851)
        vec3 dD = layer_grain * (eff_strength * 0.35 * grain_att) * dens_mod;
        linear_rgb = max(vec3(0.0), linear_rgb * exp(-dD * 2.3025851));
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
    """Hardware-accelerated OpenGL / Vulkan-compatible viewport:
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
        # 512x1 RGB ramp holding the current stock's density law (core/grain_model.py), plus the
        # identity of the table it holds so switching stock re-uploads instead of re-baking.
        self._grain_density_tex_id = None
        self._grain_density_lut_id = None
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
            "grain_cloud_blur": 1.0,
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

    def initializeGL(self):
        self.shader_program = QOpenGLShaderProgram(self)
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER_SRC)
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER_SRC)
        self.shader_program.link()

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

    def _upload_grain_density_lut(self, lut):
        """Upload (or refresh) the 512x1 density-law ramp and return its texture id.

        The table is baked once per stock by core/grain_model.bake_density_lut() and cached there,
        so the same array object comes back every frame; ``id()`` is enough to notice a stock
        change without re-uploading 6 KB of constants on every paint.
        """
        if lut is None:
            return None
        key = (id(lut), lut.shape)
        tex = getattr(self, "_grain_density_tex_id", None)
        if tex and getattr(self, "_grain_density_lut_id", None) == key:
            return tex
        data = np.ascontiguousarray(lut, dtype=np.float32).reshape(1, int(lut.shape[0]), 3)
        height, width = data.shape[0], data.shape[1]
        if not tex:
            tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, width, height, 0,
                            GL.GL_RGB, GL.GL_FLOAT, data)
            self._grain_density_tex_id = tex
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height,
                               GL.GL_RGB, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._grain_density_lut_id = key
        return tex

    def _bind_grain_density_lut(self, funcs, u_1f, u_1i, spec):
        """Bind the stock's density law, or switch the shader back to the P0 bell."""
        tex = None
        lut = getattr(spec, "density_lut", None)
        if lut is not None:
            try:
                tex = self._upload_grain_density_lut(lut)
            except Exception as exc:                # noqa: BLE001 - never break a paint
                print(f"[SpektraGrain] 密度律纹理上传失败，回退 P0 钟形: {exc}")
                tex = None
        if not tex:
            # No table: the shader keeps the global P0 bell.  Both branches return the same
            # quantity in the same units, so the grain stays continuous either way.
            u_1f("u_grain_density_enable", 0.0)
            return
        funcs.glActiveTexture(0x84C0 + GRAIN_DENSITY_TEX_UNIT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        u_1i("u_grain_density_lut", GRAIN_DENSITY_TEX_UNIT)
        u_1f("u_grain_density_enable", 1.0)
        funcs.glActiveTexture(0x84C0)   # callers bind the image on unit 0 right after this

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
            is_bypass = 1 if str(self.params.get("film_stock", "none")).lower() in ("none", "") else 0
        u_1i("u_is_bypass", is_bypass)

        eff_exposure = float(self.params.get("exposure_ev", 0.0)) + float(self.params.get("base_ev", 0.0))
        u_1f("u_exposure", eff_exposure)
        u_1f("u_temp", self.params.get("color_temp", 5500.0))
        u_1f("u_tint", self.params.get("tint", 1.0))
        u_1f("u_base_temp", self.params.get("base_temp", 5500.0))
        u_1f("u_base_tint", self.params.get("base_tint", 1.0))
        u_1f("u_highlights", float(self.params.get("highlights", 0.0)))
        u_1f("u_shadows", float(self.params.get("shadows", 0.0)))
        film_id = self.params.get("film_stock", getattr(self, "current_film_stock", "none"))
        is_pos = self.engine.is_positive_film(film_id) if hasattr(self.engine, 'is_positive_film') else False
        if is_pos:
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
            u_2f("u_lens_vignette", lens_cfg.get("vignette_amount", 0.0), lens_cfg.get("vignette_midpoint", 50.0))
            ca_en = 1 if lens_cfg.get("remove_ca", False) else 0
            u_1i("u_lens_ca_enabled", ca_en)
            ca_r = float(lens_cfg.get("ca_kr", 0.0))
            ca_b = float(lens_cfg.get("ca_kb", 0.0))
            u_2f("u_lens_ca_params", ca_r, ca_b)
        else:
            u_1i("u_lens_enabled", 0)
            u_3f("u_lens_distort", 0.0, 0.0, 0.0)
            u_2f("u_lens_vignette", 0.0, 50.0)
            u_1i("u_lens_ca_enabled", 0)
            u_2f("u_lens_ca_params", 0.0, 0.0)

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

        hal_amt = float(self.params.get("halation_amount", self.params.get("halation", 1.0)))
        hal_scale = float(self.params.get("halation_spatial_scale", 1.0))
        scatter_amt = float(self.params.get("scatter_amount", 1.0))
        boost_ev = float(self.params.get("boost_ev", self.params.get("halation_boost", 0.0)))

        u_1f("u_halation", hal_amt)
        u_1f("u_hal_spatial_scale", hal_scale)
        u_1f("u_scatter_amount", scatter_amt)
        u_1f("u_halation_boost", boost_ev)
        u_3f("u_hal_strength_rgb", float(strength[0]), float(strength[1]), float(strength[2]))
        u_1f("u_hal_sigma_um", sigma_um)

        # Film Stock Crystal Grain physics (Per-stock crystal size, strength, layer weights)
        grain_preset = self.params.get("_grain_preset")
        if not grain_preset:
            eng = getattr(self, "engine", None)
            if not eng and hasattr(self.parent(), "engine"):
                eng = self.parent().engine
            if eng and hasattr(eng, "get_film_grain_preset"):
                stock = self.params.get("film_stock", "kodak_portra_400")
                grain_preset = eng.get_film_grain_preset(stock)
        if not grain_preset:
            grain_preset = {"size_mm": 0.005, "strength": 0.010, "layer_weights": [0.70, 1.0, 1.35]}

        st_size = float(grain_preset.get("size_mm", 0.005))
        st_strength = float(grain_preset.get("strength", 0.010))
        st_lw = grain_preset.get("layer_weights", [0.70, 1.0, 1.35])

        u_1f("u_stock_grain_size", st_size)
        u_1f("u_stock_grain_strength", st_strength)
        u_3f("u_stock_layer_weights", float(st_lw[0]), float(st_lw[1]), float(st_lw[2]))

        # Grain physics & format
        u_1f("u_grain", self.params.get("grain", 0.4))
        fmt_mm = float(self.params.get("film_format_mm", 35.0))
        _mw = float(getattr(self, "_master_width", 0) or self._image_width or 6024)
        _mh = float(getattr(self, "_master_height", 0) or self._image_height or 4024)
        u_1f("u_grain_scale", grain_scale_px(fmt_mm, max(_mw, _mh)))
        u_1f("u_film_width_um", max(5.0, fmt_mm) * 1000.0)
        u_1f("u_grain_size", float(self.params.get("grain_size", 1.0)))
        u_1f("u_grain_blur", self.params.get("grain_cloud_blur", 1.0))
        u_1f("u_grain_colour", float(self.params.get("grain_colour", 1.0)))
        _spec = grain_derive_spec(self.params, st_strength, st_lw, st_size, fmt_mm,
                                 max(_mw, _mh), grain_preset)
        u_1f("u_grain_rho", _spec.rho)
        u_1f("u_grain_w_fine", _spec.w_fine)
        u_1f("u_grain_w_cloud", _spec.w_cloud)
        u_1f("u_grain_w_mottle", _spec.w_mottle)
        u_1f("u_grain_amplitude", _spec.amplitude)
        _bind_density_uniforms(u_1f)
        self._bind_grain_density_lut(funcs, u_1f, u_1i, _spec)
        master_w = float(getattr(self, "_master_width", 0) or self._image_width)
        master_h = float(getattr(self, "_master_height", 0) or self._image_height)
        u_2f("u_master_resolution", master_w, master_h)

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

        # Chemistry Morph / Developer Exhaustion deltas (DIR is fully baked in the LUT)
        _bk = getattr(self, "_lut_baked", None) or self._lookup_lut_baked(None)
        u_1f("u_lut_morph_gamma", float(_bk.get("morph_gamma", 1.0)))
        u_1f("u_lut_dev_exh", float(_bk.get("developer_exhaustion", 0.0)))

        # Chemistry Morph (neutralized for positive slide film)
        if is_pos:
            u_1f("u_morph_gamma", 1.0)
            u_1f("u_developer_exhaustion", 0.0)
        else:
            u_1f("u_morph_gamma", float(self.params.get("morph_gamma", 1.0)))
            u_1f("u_developer_exhaustion", float(self.params.get("developer_exhaustion", 0.0)))

        u_1f("u_split_x", self.params.get("split_x", 0.5))
        u_1i("u_view_mode", self.params.get("view_mode", 0))
        u_2f("u_viewport_size", float(max(1, self.width())), float(max(1, self.height())))

        # Bind 2D Image Texture (Unit 0)
        if self._image_tex_id:
            funcs.glActiveTexture(0x84C0)  # GL_TEXTURE0
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._image_tex_id)
            u_1i("u_image", 0)

        # Bind 3D LUT Texture (Unit 1)
        if self._lut_tex_id:
            funcs.glActiveTexture(0x84C1)  # GL_TEXTURE1
            GL.glBindTexture(GL.GL_TEXTURE_3D, self._lut_tex_id)
            u_1i("u_lut", 1)

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

    def set_3d_lut(self, lut_array_3d):
        """Upload 3D LUT array (float32 [N, N, N, 3]) to 3D texture."""
        self._lut_baked = self._lookup_lut_baked(lut_array_3d)
        self._pending_lut = lut_array_3d
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

    def render_offscreen(self, target_w, target_h, float_img_rgb=None, params_override=None, bit_depth=8, lut_override=None, is_16bit=None):
        """Render image at target resolution on GPU using an offscreen FBO.
        Returns uint8 or uint16 RGB numpy array (target_h, target_w, 3) or None on error.
        """
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
        if not getattr(self, "_gl_initialized", False) or not self.isValid():
            return None

        self.makeCurrent()
        orig_tex_id = self._image_tex_id
        temp_tex_id = None
        temp_lut_tex_id = None
        fbo = None
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
                is_bypass = 1 if str(eff_params.get("film_stock", "none")).lower() in ("none", "") else 0
            u_1i("u_is_bypass", is_bypass)

            eff_exp = float(eff_params.get("exposure_ev", 0.0)) + float(eff_params.get("base_ev", 0.0))
            u_1f("u_exposure", eff_exp)
            u_1f("u_temp", eff_params.get("color_temp", 5500.0))
            u_1f("u_tint", eff_params.get("tint", 1.0))
            u_1f("u_base_temp", eff_params.get("base_temp", 5500.0))
            u_1f("u_base_tint", eff_params.get("base_tint", 1.0))
            u_1f("u_highlights", float(eff_params.get("highlights", 0.0)))
            u_1f("u_shadows", float(eff_params.get("shadows", 0.0)))
            film_id_off = eff_params.get("film_stock", getattr(self, "current_film_stock", "none"))
            is_pos_off = self.engine.is_positive_film(film_id_off) if hasattr(self.engine, 'is_positive_film') else False
            if is_pos_off:
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
                u_2f("u_lens_vignette", off_lens.get("vignette_amount", 0.0), off_lens.get("vignette_midpoint", 50.0))
            else:
                u_1i("u_lens_enabled", 0)
                u_3f("u_lens_distort", 0.0, 0.0, 0.0)
                u_2f("u_lens_vignette", 0.0, 50.0)
            off_ca = 1 if (isinstance(off_lens, dict) and off_lens.get("remove_ca", False)) else 0
            u_1i("u_lens_ca_enabled", off_ca)
            off_ca_r = float(off_lens.get("ca_kr", 0.0)) if isinstance(off_lens, dict) else 0.0
            off_ca_b = float(off_lens.get("ca_kb", 0.0)) if isinstance(off_lens, dict) else 0.0
            u_2f("u_lens_ca_params", off_ca_r, off_ca_b)

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

            hal_amt = float(eff_params.get("halation_amount", eff_params.get("halation", 1.0)))
            hal_scale = float(eff_params.get("halation_spatial_scale", 1.0))
            scatter_amt = float(eff_params.get("scatter_amount", 1.0))
            boost_ev = float(eff_params.get("boost_ev", eff_params.get("halation_boost", 0.0)))

            u_1f("u_halation", hal_amt)
            u_1f("u_hal_spatial_scale", hal_scale)
            u_1f("u_scatter_amount", scatter_amt)
            u_1f("u_halation_boost", boost_ev)
            u_3f("u_hal_strength_rgb", float(strength[0]), float(strength[1]), float(strength[2]))
            u_1f("u_hal_sigma_um", sigma_um)

            # Film Stock Crystal Grain physics (Per-stock crystal size, strength, layer weights)
            grain_preset = eff_params.get("_grain_preset")
            if not grain_preset:
                eng = getattr(self, "engine", None)
                if not eng and hasattr(self.parent(), "engine"):
                    eng = self.parent().engine
                if eng and hasattr(eng, "get_film_grain_preset"):
                    stock = eff_params.get("film_stock", "kodak_portra_400")
                    grain_preset = eng.get_film_grain_preset(stock)
            if not grain_preset:
                grain_preset = {"size_mm": 0.005, "strength": 0.010, "layer_weights": [0.70, 1.0, 1.35]}

            st_size = float(grain_preset.get("size_mm", 0.005))
            st_strength = float(grain_preset.get("strength", 0.010))
            st_lw = grain_preset.get("layer_weights", [0.70, 1.0, 1.35])

            u_1f("u_stock_grain_size", st_size)
            u_1f("u_stock_grain_strength", st_strength)
            u_3f("u_stock_layer_weights", float(st_lw[0]), float(st_lw[1]), float(st_lw[2]))

            is_thumbnail = (max(target_w, target_h) <= 512) or bool(eff_params.get("is_thumbnail", False)) or (params_override and params_override.get("grain") == 0.0)
            if is_thumbnail:
                u_1f("u_grain", 0.0)
            else:
                u_1f("u_grain", eff_params.get("grain", 0.4))
            fmt_mm = float(eff_params.get("film_format_mm", 35.0))
            u_1f("u_grain_scale", grain_scale_px(fmt_mm, max(float(target_w), float(target_h))))
            u_1f("u_film_width_um", max(5.0, fmt_mm) * 1000.0)
            u_1f("u_grain_size", float(eff_params.get("grain_size", 1.0)))
            u_1f("u_grain_blur", eff_params.get("grain_cloud_blur", 1.0))
            u_1f("u_grain_colour", float(eff_params.get("grain_colour", 1.0)))
            # The offscreen pass renders at the export resolution, so the spec is derived with
            # the target long edge -- that is what applies the physical amplitude law.
            _spec = grain_derive_spec(eff_params, st_strength, st_lw, st_size, fmt_mm,
                                     max(float(target_w), float(target_h)), grain_preset)
            u_1f("u_grain_rho", _spec.rho)
            u_1f("u_grain_w_fine", _spec.w_fine)
            u_1f("u_grain_w_cloud", _spec.w_cloud)
            u_1f("u_grain_w_mottle", _spec.w_mottle)
            u_1f("u_grain_amplitude", _spec.amplitude)
            _bind_density_uniforms(u_1f)
            self._bind_grain_density_lut(funcs, u_1f, u_1i, _spec)
            u_2f("u_master_resolution", float(target_w), float(target_h))

            diff_family = str(eff_params.get("diffusion_family", "none")).lower()
            family_map = {"none": 0, "black_pro_mist": 1, "glimmerglass": 2, "pro_mist": 3, "cinebloom": 4}
            u_1i("u_diffusion_type", family_map.get(diff_family, 0))
            u_1f("u_diffusion_strength", float(eff_params.get("diffusion_strength", 0.0)))
            u_1f("u_diffusion_warmth", float(eff_params.get("diffusion_warmth", 0.0)))

            _bk = self._lookup_lut_baked(lut_override) if lut_override is not None else (getattr(self, "_lut_baked", None) or self._lookup_lut_baked(None))
            u_1f("u_lut_morph_gamma", float(_bk.get("morph_gamma", 1.0)))
            u_1f("u_lut_dev_exh", float(_bk.get("developer_exhaustion", 0.0)))
            if is_pos_off:
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

            funcs.glActiveTexture(0x84C0)  # GL_TEXTURE0
            GL.glBindTexture(GL.GL_TEXTURE_2D, active_tex_id)
            u_1i("u_image", 0)

            if active_lut_tex_id:
                funcs.glActiveTexture(0x84C1)  # GL_TEXTURE1
                GL.glBindTexture(GL.GL_TEXTURE_3D, active_lut_tex_id)
                u_1i("u_lut", 1)

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
