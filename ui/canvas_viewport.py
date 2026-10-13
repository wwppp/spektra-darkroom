import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QVariantAnimation, QEasingCurve, QRectF
from PySide6.QtGui import QCursor, QColor, QOpenGLContext, QAction, QKeySequence, QPainter, QPen, QBrush, QFont, QPixmap, QIcon
from PySide6.QtWidgets import QMenu, QLabel, QWidget, QPushButton, QVBoxLayout
from ui.window_utils import get_darkroom_menu_style, get_icon
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLShaderProgram, QOpenGLShader, QOpenGLBuffer, QOpenGLFramebufferObject
)
from path_utils import get_resource_dir
from OpenGL import GL


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
uniform vec2 u_master_resolution;

// Diffusion filter
uniform int u_diffusion_type;
uniform float u_diffusion_strength;
uniform float u_diffusion_warmth;

// DIR Couplers
uniform float u_dir_interlayer;
uniform float u_dir_samelayer;

// Chemistry Morph
uniform float u_morph_gamma;
uniform float u_developer_exhaustion;

uniform float u_split_x;
uniform int u_view_mode;
uniform float u_screen_aspect;
uniform float u_image_aspect;
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
    float temp_factor = (u_temp - b_temp) / 3000.0;
    col.r *= (1.0 + temp_factor * 0.22);
    col.b *= (1.0 - temp_factor * 0.22);
    col.g *= (u_tint / b_tint);

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
        fragColor = vec4(0.08, 0.08, 0.08, 1.0);
        return;
    }
    
    float eff_screen_aspect = u_screen_aspect;
    vec2 cur_coord = v_texcoord;
    bool is_left = (v_texcoord.x < 0.5);

    if (u_view_mode == 2) {
        // Mode 2: Global side-by-side comparison (Both complete original and edited images)
        float vp_w = (u_viewport_size.x > 1.0) ? u_viewport_size.x : (1.0 / max(0.00001, fwidth(v_texcoord.x)));
        float dist_px = abs(v_texcoord.x - 0.5) * vp_w;
        if (dist_px <= 2.2) {
            vec3 amber_gold = vec3(0.96, 0.62, 0.05);
            vec3 core_hl   = vec3(1.00, 0.88, 0.45);
            vec3 dark_edge  = vec3(0.05, 0.05, 0.06);
            vec3 line_col = (dist_px <= 0.8) ? mix(core_hl, amber_gold, dist_px / 0.8) : mix(amber_gold, dark_edge, clamp((dist_px - 0.8) / 1.4, 0.0, 1.0));
            fragColor = vec4(line_col, 1.0);
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
    
    // Y-flip for OpenGL texture coordinates (row 0 at top)
    vec2 sample_uv = vec2(img_uv.x, 1.0 - img_uv.y);
    
    // Letterbox dark background
    if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
        fragColor = vec4(0.07, 0.07, 0.07, 1.0);
        return;
    }
    
    vec3 orig_rgb = texture(u_image, sample_uv).rgb;

    // Accurate untouched base display color (Before / 原片)
    vec3 before_color = (u_is_bypass == 2) ? clamp(orig_rgb, 0.0, 1.0) : clamp(prophoto_to_srgb(orig_rgb), 0.0, 1.0);

    // -------------------------------------------------------------
    // Compare Modes (Must execute BEFORE bypass early return!)
    // -------------------------------------------------------------
    if (u_view_mode == 1) {
        // Mode 1: Split screen drag compare (Before / After)
        float vp_w = (u_viewport_size.x > 1.0) ? u_viewport_size.x : (1.0 / max(0.00001, fwidth(v_texcoord.x)));
        float dist_px = abs(v_texcoord.x - u_split_x) * vp_w;

        // High-visibility dual-layer hairline divider (solid amber gold core + drop shadow rim)
        // 100% visible against bright snow, highlights, and deep shadows!
        if (dist_px <= 2.2) {
            vec3 amber_gold = vec3(0.96, 0.62, 0.05); // #f59e0b
            vec3 core_hl   = vec3(1.00, 0.88, 0.45);
            vec3 dark_edge  = vec3(0.05, 0.05, 0.06);

            vec3 line_col;
            if (dist_px <= 0.8) {
                line_col = mix(core_hl, amber_gold, dist_px / 0.8);
            } else {
                float edge_t = clamp((dist_px - 0.8) / 1.4, 0.0, 1.0);
                line_col = mix(amber_gold, dark_edge, edge_t);
            }
            fragColor = vec4(line_col, 1.0);
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
        float base_hal_rad = max(0.0005, (u_hal_sigma_um / 35000.0) * max(0.2, u_hal_spatial_scale));
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
    
    // 5. DIR Coupler inhibition response (Color separation & samelayer contrast)
    if (abs(u_dir_interlayer - 1.0) > 0.01 || abs(u_dir_samelayer - 1.0) > 0.01) {
        float luma_f = dot(film_rgb, vec3(0.299, 0.587, 0.114));
        vec3 chroma = film_rgb - vec3(luma_f);
        film_rgb = vec3(luma_f) + chroma * u_dir_interlayer;
        film_rgb = pow(clamp(film_rgb, 0.0001, 1.0), vec3(1.0 / max(0.2, u_dir_samelayer)));
    }

    // 6. Enlarger Color Timing (CMY filters & print exposure)
    vec3 cmy_factor = vec3(1.0) - u_cmy * 0.015;
    film_rgb = pow(clamp(film_rgb, 0.0001, 1.0), cmy_factor) * u_print_exposure;
    film_rgb += vec3(u_preflash);
    
    // 7. Paper Chemistry Morph & Developer Exhaustion
    if (abs(u_morph_gamma - 1.0) > 0.01) {
        film_rgb = pow(clamp(film_rgb, 0.0001, 1.0), vec3(u_morph_gamma));
    }
    if (u_developer_exhaustion > 0.005) {
        film_rgb = mix(film_rgb, film_rgb * 0.88 + vec3(0.03), u_developer_exhaustion * 0.6);
    }

    // 8. Multi-bounce Halation & Highlight Boost (Processed in scene-linear irradiance space before LUT)
    
    // 9. Silver Halide Organic Film Grain & Dye Cloud Blur (Film-space anchored)
    if (u_grain > 0.01) {
        float density = 1.0 - clamp(dot(film_rgb, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
        // Optical density bell curve (most pronounced in midtones, decays smoothly in specular highlights and deep blacks)
        float grain_mask = sqrt(max(0.0, density * (1.0 - density))) * 2.0;

        // Golden-ratio irrational rotation matrix (breaks all raster / coordinate grid alignment)
        const mat2 rot = mat2(0.9114, -0.4115, 0.4115, 0.9114);
        float scale = max(0.25, u_grain_scale * max(0.2, u_grain_size));

        // Anchor grain coordinate to image/film master pixel space (WYSIWYG preview & export)
        vec2 master_res = (u_master_resolution.x > 1.0) ? u_master_resolution : vec2(textureSize(u_image, 0));
        vec2 film_coord = sample_uv * master_res;
        vec2 gcoord = rot * (film_coord / scale);

        // Anti-aliasing / Central Limit Theorem screen downsampling integration
        // When zoomed out (e.g. Fit to View), 1 screen pixel averages an area of film pixels,
        // which physically attenuates high-frequency micro-crystals, perfectly matching viewer downsampling!
        vec2 d_gcoord = fwidth(gcoord);
        float footprint = max(d_gcoord.x, d_gcoord.y);
        float crystal_att = 1.0 / sqrt(max(1.0, footprint));
        float cloud_att = 1.0 / sqrt(max(1.0, footprint * 0.42));

        // 1. High-frequency organic silver halide crystals (Gaussian distribution via Central Limit Theorem)
        uvec2 u_crystal = uvec2(ivec2(floor(gcoord)));
        float r1 = pcg_float(u_crystal);
        float r2 = pcg_float(u_crystal + uvec2(1597334677u, 3812015801u));
        float crystal_noise = ((r1 + r2) * 0.5 - 0.5) * crystal_att;

        // 2. Multi-scale organic dye cloud clusters (smooth continuous interpolation)
        float cloud_noise = (smooth_cloud_noise(gcoord * 0.42) - 0.5) * cloud_att;

        // Blend between fine micro-crystals and soft organic dye clouds
        float blur_w = clamp(u_grain_blur * 0.45, 0.0, 0.70);
        float combined_noise = mix(crystal_noise, cloud_noise, blur_w);

        float noise = combined_noise * u_grain * grain_mask * 0.28;
        film_rgb += vec3(noise);
    }
    
    fragColor = vec4(clamp(film_rgb, 0.0, 1.0), 1.0);
}
"""


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
    requestExport = Signal()
    requestQuickExport = Signal()
    requestToggleSplit = Signal()
    importRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.engine = getattr(parent, "engine", None)

        # Texture state
        self._image_tex_id = None
        self._lut_tex_id = None
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
        self.badge_before = QLabel("原图", self)
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

    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        if not self.shader_program or not self.vbo:
            return

        funcs = QOpenGLContext.currentContext().functions()
        GL.glClearColor(0.07, 0.07, 0.07, 1.0)
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

        screen_aspect = float(max(1, self.width())) / float(max(1, self.height()))
        m_w = self._master_width if self._master_width > 0 else self._image_width
        m_h = self._master_height if self._master_height > 0 else self._image_height
        img_aspect = float(m_w) / float(max(1, m_h))

        u_1f("u_screen_aspect", screen_aspect)
        u_1f("u_image_aspect", img_aspect)
        u_2f("u_pan", self.pan_x, self.pan_y)
        u_1f("u_zoom", self.zoom)

        u_1i("u_has_image", 1 if self._has_image else 0)
        u_1i("u_has_lut", 1 if self._has_lut else 0)
        if getattr(self, "_is_placeholder", False):
            is_bypass = 2
        else:
            is_bypass = 1 if str(self.params.get("film_stock", "none")).lower() in ("none", "") else 0
        u_1i("u_is_bypass", is_bypass)

        u_1f("u_exposure", self.params.get("exposure_ev", 0.0))
        u_1f("u_temp", self.params.get("color_temp", 5500.0))
        u_1f("u_tint", self.params.get("tint", 1.0))
        u_1f("u_base_temp", self.params.get("base_temp", 5500.0))
        u_1f("u_base_tint", self.params.get("base_tint", 1.0))
        u_1f("u_highlights", float(self.params.get("highlights", 0.0)))
        u_1f("u_shadows", float(self.params.get("shadows", 0.0)))
        u_3f("u_cmy", self.params.get("enlarger_cyan", 0.0), self.params.get("enlarger_magenta", 0.0), self.params.get("enlarger_yellow", 0.0))
        u_1f("u_print_exposure", self.params.get("print_exposure", 1.0))
        u_1f("u_preflash", self.params.get("pre_flash", 0.0))

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

        # Grain physics & format
        u_1f("u_grain", self.params.get("grain", 0.4))
        fmt_mm = float(self.params.get("film_format_mm", 35.0))
        grain_scale = 35.0 / max(5.0, fmt_mm)
        u_1f("u_grain_scale", grain_scale)
        u_1f("u_grain_size", float(self.params.get("grain_size", 1.0)))
        u_1f("u_grain_blur", self.params.get("grain_cloud_blur", 1.0))
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

        # DIR Couplers
        u_1f("u_dir_interlayer", self.params.get("dir_interlayer", 1.0))
        u_1f("u_dir_samelayer", self.params.get("dir_samelayer", 1.0))

        # Chemistry Morph
        u_1f("u_morph_gamma", self.params.get("morph_gamma", 1.0))
        u_1f("u_developer_exhaustion", self.params.get("developer_exhaustion", 0.0))

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

    def paintEvent(self, event):
        super().paintEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
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
            b1_w, b1_h = 96, 26
            b1_x = int(self.width() / 2.0 - b1_w / 2.0)
            self.badge_before.setText("原片 (Before)")
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

    def set_3d_lut(self, lut_array_3d):
        """Upload 3D LUT array (float32 [N, N, N, 3]) to 3D texture."""
        if not getattr(self, "_gl_initialized", False) or not self.isValid():
            self._pending_lut = lut_array_3d
            return
        self.makeCurrent()
        try:
            self._do_upload_lut(lut_array_3d)
        finally:
            self.doneCurrent()
        self.update()

    set_lut = set_3d_lut

    def _do_upload_lut(self, lut_array_3d):
        size = lut_array_3d.shape[0]

        if self._lut_tex_id:
            GL.glDeleteTextures([self._lut_tex_id])
            self._lut_tex_id = None

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        self._lut_tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._lut_tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

        data = np.ascontiguousarray(lut_array_3d, dtype=np.float32)
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

    def update_params(self, **kwargs):
        """Update shader uniforms instantly and redraw."""
        self.params.update(kwargs)
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
        if animate:
            self.animate_to(1.0, 0.0, 0.0)
        else:
            self.zoom = 1.0
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

        if target_zoom < 1.25:
            target_zoom = max(2.0, target_zoom)

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

        act_split = menu.addAction("原片前后对比")
        act_split.setIcon(get_icon("compare.png"))
        act_split.setShortcut(QKeySequence("\\"))
        act_split.triggered.connect(self.requestToggleSplit.emit)

        menu.addSeparator()

        act_undo = menu.addAction("撤销 (Undo)")
        act_undo.setIcon(get_icon("undo.png"))
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self.requestUndo.emit)

        act_redo = menu.addAction("重做 (Redo)")
        act_redo.setIcon(get_icon("redo.png"))
        act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        act_redo.triggered.connect(self.requestRedo.emit)

        act_reset = menu.addAction("重置所有参数 (Reset All)")
        act_reset.setIcon(get_icon("reset.png"))
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.triggered.connect(self.requestReset.emit)

        menu.addSeparator()

        act_quick = menu.addAction("快速冲印 (上次参数)")
        act_quick.setIcon(get_icon("quick_export.png"))
        act_quick.setShortcut(QKeySequence("Ctrl+Shift+E"))
        act_quick.triggered.connect(self.requestQuickExport.emit)

        act_export = menu.addAction("冲印照片 (Print)...")
        act_export.setIcon(get_icon("export.png"))
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.requestExport.emit)

        menu.exec(event.globalPos())

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

        if self.params.get("view_mode") == 1:
            self._clamp_split_to_image()
            self._update_badges()

        self.zoomChanged.emit(self.get_actual_zoom_ratio())
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on split divider within visible image bounds
            if self._is_hovering_split_line(event.position()):
                self._is_split_dragging = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self._is_panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._last_mouse_pos = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
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
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self._is_split_dragging = False
            if self._is_hovering_split_line(event.position()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

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
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                data = np.ascontiguousarray(float_img_rgb, dtype=np.float32)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, src_w, src_h, 0, GL.GL_RGB, GL.GL_FLOAT, data)
                try:
                    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
                except Exception:
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                active_tex_id = temp_tex_id
            else:
                active_tex_id = orig_tex_id

            if not active_tex_id:
                return None

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

            # Create FBO (16-bit float buffer for high dynamic range precision when bit_depth >= 16)
            is_16bit = (int(bit_depth) >= 16)
            if is_16bit:
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
            u_2f("u_pan", 0.0, 0.0)
            u_1f("u_zoom", 1.0)

            u_1i("u_has_image", 1)
            u_1i("u_has_lut", 1 if (active_lut_tex_id and (self._has_lut or lut_override is not None)) else 0)
            if getattr(self, "_is_placeholder", False):
                is_bypass = 2
            else:
                is_bypass = 1 if str(eff_params.get("film_stock", "none")).lower() in ("none", "") else 0
            u_1i("u_is_bypass", is_bypass)

            u_1f("u_exposure", eff_params.get("exposure_ev", 0.0))
            u_1f("u_temp", eff_params.get("color_temp", 5500.0))
            u_1f("u_tint", eff_params.get("tint", 1.0))
            u_1f("u_base_temp", eff_params.get("base_temp", 5500.0))
            u_1f("u_base_tint", eff_params.get("base_tint", 1.0))
            u_1f("u_highlights", float(eff_params.get("highlights", 0.0)))
            u_1f("u_shadows", float(eff_params.get("shadows", 0.0)))
            u_3f("u_cmy", eff_params.get("enlarger_cyan", 0.0), eff_params.get("enlarger_magenta", 0.0), eff_params.get("enlarger_yellow", 0.0))
            u_1f("u_print_exposure", eff_params.get("print_exposure", 1.0))
            u_1f("u_preflash", eff_params.get("pre_flash", 0.0))

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

            u_1f("u_grain", eff_params.get("grain", 0.4))
            fmt_mm = float(eff_params.get("film_format_mm", 35.0))
            grain_scale = 35.0 / max(5.0, fmt_mm)
            u_1f("u_grain_scale", grain_scale)
            u_1f("u_grain_size", float(eff_params.get("grain_size", 1.0)))
            u_1f("u_grain_blur", eff_params.get("grain_cloud_blur", 1.0))
            u_2f("u_master_resolution", float(target_w), float(target_h))

            diff_family = str(eff_params.get("diffusion_family", "none")).lower()
            family_map = {"none": 0, "black_pro_mist": 1, "glimmerglass": 2, "pro_mist": 3, "cinebloom": 4}
            u_1i("u_diffusion_type", family_map.get(diff_family, 0))
            u_1f("u_diffusion_strength", float(eff_params.get("diffusion_strength", 0.0)))
            u_1f("u_diffusion_warmth", float(eff_params.get("diffusion_warmth", 0.0)))

            u_1f("u_dir_interlayer", eff_params.get("dir_interlayer", 1.0))
            u_1f("u_dir_samelayer", eff_params.get("dir_samelayer", 1.0))
            u_1f("u_morph_gamma", eff_params.get("morph_gamma", 1.0))
            u_1f("u_developer_exhaustion", eff_params.get("developer_exhaustion", 0.0))

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
            if is_16bit:
                raw_bytes = GL.glReadPixels(0, 0, target_w, target_h, GL.GL_RGBA, GL.GL_UNSIGNED_SHORT)
                fbo.release()
                img_arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((target_h, target_w, 4))[..., :3]
            else:
                raw_bytes = GL.glReadPixels(0, 0, target_w, target_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
                fbo.release()
                img_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((target_h, target_w, 3))

            img_arr = np.ascontiguousarray(np.flipud(img_arr))
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
