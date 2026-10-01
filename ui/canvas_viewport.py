import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QVariantAnimation, QEasingCurve, QRectF
from PySide6.QtGui import QCursor, QColor, QOpenGLContext, QAction, QKeySequence, QPainter, QPen, QBrush, QFont, QPixmap, QIcon
from PySide6.QtWidgets import QMenu, QLabel, QWidget, QPushButton, QVBoxLayout
from ui.window_utils import get_darkroom_menu_style
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

uniform float u_exposure;
uniform float u_temp;
uniform float u_tint;
uniform float u_base_temp;
uniform float u_base_tint;
uniform vec3 u_cmy;
uniform float u_print_exposure;
uniform float u_preflash;

// Halation physics
uniform float u_halation;
uniform int u_halation_bounces;
uniform float u_halation_decay;
uniform float u_halation_boost;

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
    vec3 higher = 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055;
    vec3 lower = lin * 12.92;
    vec3 cutoff = step(vec3(0.0031308), lin);
    return clamp(mix(lower, higher, cutoff), 0.0, 1.0);
}

// Filmic soft highlight roll-off (smooth C1 shoulder compression above threshold)
vec3 filmic_shoulder(vec3 c, float threshold) {
    vec3 over = max(vec3(0.0), c - threshold);
    float span = 1.0 - threshold;
    return min(c, vec3(threshold)) + span * (over / (span + over));
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
        float dx = fwidth(v_texcoord.x);
        float dist_px = abs(v_texcoord.x - 0.5) / max(0.00001, dx);
        if (dist_px < 1.0) {
            float v_fade = smoothstep(0.0, 0.08, v_texcoord.y) * smoothstep(1.0, 0.92, v_texcoord.y);
            vec3 line_color = mix(vec3(0.20, 0.22, 0.28), vec3(0.96, 0.62, 0.05), v_fade);
            fragColor = vec4(line_color, 1.0);
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
    
    // Split screen divider hairline (Mode 1: Before / After drag compare)
    if (u_view_mode == 1) {
        float dx = fwidth(v_texcoord.x);
        float dist_px = abs(v_texcoord.x - u_split_x) / max(0.00001, dx);
        if (dist_px < 1.0) {
            float v_fade = smoothstep(0.0, 0.08, v_texcoord.y) * smoothstep(1.0, 0.92, v_texcoord.y);
            vec3 line_color = mix(vec3(0.20, 0.22, 0.28), vec3(0.96, 0.62, 0.05), v_fade);
            fragColor = vec4(line_color, 1.0);
            return;
        }
        if (v_texcoord.x < u_split_x) {
            fragColor = vec4(prophoto_to_srgb(orig_rgb), 1.0);
            return;
        }
    } else if (u_view_mode == 2 && is_left) {
        // Mode 2 Left viewport: Full complete original (Before)
        fragColor = vec4(prophoto_to_srgb(orig_rgb), 1.0);
        return;
    } else if (u_view_mode == 3) {
        // Mode 3: Temporary Full Original Before Compare (Long Press on Compare button)
        fragColor = vec4(prophoto_to_srgb(orig_rgb), 1.0);
        return;
    }
    
    // 1. Exposure compensation
    vec3 linear_rgb = orig_rgb * exp2(u_exposure);
    
    // 2. White balance / Temperature tint calibrated against camera as-shot baseline
    float b_temp = (u_base_temp > 1000.0) ? u_base_temp : 5500.0;
    float b_tint = (u_base_tint > 0.1) ? u_base_tint : 1.0;
    float temp_factor = (u_temp - b_temp) / 3000.0;
    linear_rgb.r *= (1.0 + temp_factor * 0.22);
    linear_rgb.b *= (1.0 - temp_factor * 0.22);
    linear_rgb.g *= (u_tint / b_tint);
    
    // 3. Optical Diffusion Filter (Physical Convolution Bloom)
    int eff_diff_type = u_diffusion_type;
    if (eff_diff_type == 0 && u_diffusion_strength > 0.001) {
        eff_diff_type = 1; // Default to Black Pro-Mist if strength > 0
    }
    if (eff_diff_type > 0 && u_diffusion_strength > 0.001) {
        float base_rad = 0.010 * max(0.25, u_diffusion_strength);
        if (eff_diff_type == 4) {
            base_rad *= 1.8; // CineBloom: wide cinematic atmospheric flare
        } else if (eff_diff_type == 2) {
            base_rad *= 0.85; // Glimmerglass: crisp subtle micro-bloom
        } else if (eff_diff_type == 3) {
            base_rad *= 1.4; // Pro-Mist: classic milky luminous mist
        }
        
        vec2 tex_sz = max(vec2(100.0), vec2(textureSize(u_image, 0)));
        vec2 spread = max(vec2(1.0) / tex_sz * 2.0, vec2(base_rad));
        
        vec3 bloom_accum = vec3(0.0);
        float total_weight = 0.0;
        
        // 8-tap concentric kernel
        vec2 offsets[8] = vec2[](
            vec2( 1.0,  0.0), vec2(-1.0,  0.0),
            vec2( 0.0,  1.0), vec2( 0.0, -1.0),
            vec2( 0.707,  0.707) * 1.5, vec2(-0.707, -0.707) * 1.5,
            vec2(-0.707,  0.707) * 2.2, vec2( 0.707, -0.707) * 2.2
        );
        
        // Black Pro-Mist has higher threshold to keep deep blacks clean
        float thr = (eff_diff_type == 1) ? 0.35 : 0.25;
        
        for (int i = 0; i < 8; i++) {
            vec2 tap_uv = clamp(sample_uv + offsets[i] * spread, 0.0, 1.0);
            vec3 tap_rgb = texture(u_image, tap_uv).rgb * exp2(u_exposure);
            float tap_luma = dot(tap_rgb, vec3(0.2126, 0.7152, 0.0722));
            float hl = max(0.0, tap_luma - thr);
            float w = 1.0 / (1.0 + float(i) * 0.35);
            bloom_accum += tap_rgb * (hl * w);
            total_weight += w;
        }
        bloom_accum /= max(0.001, total_weight);
        
        vec3 diff_tint = vec3(
            1.0 + u_diffusion_warmth * 0.35,
            1.0 + u_diffusion_warmth * 0.08,
            1.0 - u_diffusion_warmth * 0.35
        );
        
        float intensity = (eff_diff_type == 3) ? 1.25 : ((eff_diff_type == 4) ? 1.45 : 1.10);
        linear_rgb += bloom_accum * (u_diffusion_strength * intensity) * diff_tint;
        
        // Classic Pro-Mist lifts shadows slightly
        if (eff_diff_type == 3) {
            linear_rgb = mix(linear_rgb, linear_rgb + vec3(0.015 * u_diffusion_strength), 0.25);
        }
    }

    // 4. Physical 3D LUT sampling (Film emulsion + Paper D-max)
    vec3 film_rgb;
    if (u_has_lut == 1) {
        vec3 rolled_in = filmic_shoulder(max(vec3(0.0), linear_rgb), 0.75);
        vec3 clamped_in = clamp(rolled_in, 0.0, 1.0);
        film_rgb = texture(u_lut, clamped_in).rgb;
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

    // 8. Multi-bounce Halation & Highlight Boost
    if (u_halation > 0.01) {
        float luma_orig = dot(orig_rgb, vec3(0.299, 0.587, 0.114));
        float highlight = max(0.0, luma_orig - 0.70 + u_halation_boost * 0.10) / 0.30;
        float hal_accum = 0.0;
        float cur_decay = 1.0;
        for (int b = 0; b < 4; b++) {
            if (b < u_halation_bounces) {
                hal_accum += pow(highlight, 1.6 + float(b) * 0.4) * cur_decay;
                cur_decay *= u_halation_decay;
            }
        }
        vec3 halation_tint = vec3(1.0, 0.14, 0.04) * (hal_accum * u_halation * 0.35);
        film_rgb += halation_tint;
    }
    
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

        # Texture state
        self._image_tex_id = None
        self._lut_tex_id = None
        self._image_width = 1
        self._image_height = 1
        self._master_width = 1
        self._master_height = 1
        self._has_image = False
        self._has_lut = False
        self._pending_image = None
        self._pending_lut = None
        self._gl_initialized = False

        # Parameters
        self.params = {
            "exposure_ev": 0.0,
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
            "halation": 0.5,
            "halation_bounces": 2,
            "halation_decay": 0.5,
            "halation_boost": 0.0,
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
        if self._pending_lut is not None:
            self._do_upload_lut(self._pending_lut)
            self._pending_lut = None

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
        img_aspect = float(self._image_width) / float(max(1, self._image_height))

        u_1f("u_screen_aspect", screen_aspect)
        u_1f("u_image_aspect", img_aspect)
        u_2f("u_pan", self.pan_x, self.pan_y)
        u_1f("u_zoom", self.zoom)

        u_1i("u_has_image", 1 if self._has_image else 0)
        u_1i("u_has_lut", 1 if self._has_lut else 0)

        u_1f("u_exposure", self.params.get("exposure_ev", 0.0))
        u_1f("u_temp", self.params.get("color_temp", 5500.0))
        u_1f("u_tint", self.params.get("tint", 1.0))
        u_1f("u_base_temp", self.params.get("base_temp", 5500.0))
        u_1f("u_base_tint", self.params.get("base_tint", 1.0))
        u_3f("u_cmy", self.params.get("enlarger_cyan", 0.0), self.params.get("enlarger_magenta", 0.0), self.params.get("enlarger_yellow", 0.0))
        u_1f("u_print_exposure", self.params.get("print_exposure", 1.0))
        u_1f("u_preflash", self.params.get("pre_flash", 0.0))

        # Halation physics
        u_1f("u_halation", self.params.get("halation", 0.5))
        u_1i("u_halation_bounces", int(self.params.get("halation_bounces", 2)))
        u_1f("u_halation_decay", self.params.get("halation_decay", 0.5))
        u_1f("u_halation_boost", self.params.get("halation_boost", 0.0))

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
        self._update_badges()

    def _update_badges(self):
        if not self._has_image and self.width() > 0:
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
            b1_w, b1_h = 56, 26
            b2_w, b2_h = 68, 26
            b1_x = int(max(16.0, min(self.width() - 140.0, split_x - b1_w - 12.0)))
            b2_x = int(max(80.0, min(self.width() - b2_w - 16.0, split_x + 12.0)))
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

    def set_image(self, float_img_rgb, master_width=None, master_height=None):
        """Upload source image array (float32 [H, W, 3]) to 2D texture, or clear if None."""
        if float_img_rgb is None:
            self._image_width = 0
            self._image_height = 0
            self._master_width = 0
            self._master_height = 0
            self._pending_image = None
            self._has_image = False
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

        try:
            h, w = float_img_rgb.shape[:2]
            self._image_width = w
            self._image_height = h
            self._master_width = int(master_width) if master_width else w
            self._master_height = int(master_height) if master_height else h
        except Exception:
            pass
        if not getattr(self, "_gl_initialized", False) or not self.isValid():
            self._pending_image = float_img_rgb
            return
        self.makeCurrent()
        try:
            self._do_upload_image(float_img_rgb)
        finally:
            self.doneCurrent()
        self.fit_to_view()
        self._update_badges()
        self.update()

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
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        data = np.ascontiguousarray(float_img_rgb, dtype=np.float32)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, w, h, 0, GL.GL_RGB, GL.GL_FLOAT, data)
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

    def update_params(self, **kwargs):
        """Update shader uniforms instantly and redraw."""
        self.params.update(kwargs)
        self.update()

    def set_view_mode(self, mode):
        # 0: single, 1: split
        self.params["view_mode"] = mode
        self._update_badges()
        self.update()

    def get_actual_zoom_ratio(self) -> float:
        """Calculate zoom ratio relative to full photo native resolution (1.0 = 100% 1:1 pixel scale)."""
        if self._image_width <= 10 or self._image_height <= 10 or self.width() <= 0 or self.height() <= 0:
            return 1.0
        aspect = float(self._image_width) / float(max(1, self._image_height))
        screen_aspect = float(self.width()) / float(max(1, self.height()))
        if screen_aspect > aspect:
            base_w = self.height() * aspect
        else:
            base_w = self.width()
        zoom_100 = float(self._image_width) / float(max(1, base_w))
        if zoom_100 <= 0:
            return 1.0
        return self.zoom / zoom_100

    def _on_anim_step(self, val):
        t = float(val)
        self.zoom = self._start_zoom + (self._target_zoom - self._start_zoom) * t
        self.pan_x = self._start_pan_x + (self._target_pan_x - self._start_pan_x) * t
        self.pan_y = self._start_pan_y + (self._target_pan_y - self._start_pan_y) * t
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
            self.zoomChanged.emit(self.get_actual_zoom_ratio())
            self.update()

    def set_zoom_100(self, animate=True):
        # 1:1 image pixel to screen pixel
        if self._image_width > 0 and self.width() > 0:
            aspect = float(self._image_width) / float(max(1, self._image_height))
            screen_aspect = float(self.width()) / float(max(1, self.height()))
            if screen_aspect > aspect:
                base_w = self.height() * aspect
            else:
                base_w = self.width()
            target_zoom = float(self._image_width) / float(max(1, base_w))
        else:
            target_zoom = 1.0
        if animate:
            self.animate_to(target_zoom, 0.0, 0.0)
        else:
            self.zoom = target_zoom
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.zoomChanged.emit(self.get_actual_zoom_ratio())
            self.update()

    def zoom_in(self):
        new_zoom = min(8.0, self.zoom * 1.25)
        self.animate_to(new_zoom, self.pan_x, self.pan_y)

    def zoom_out(self):
        new_zoom = max(0.2, self.zoom * 0.8)
        self.animate_to(new_zoom, self.pan_x, self.pan_y)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        act_in = menu.addAction("放大 (Zoom In)")
        act_in.setShortcut(QKeySequence("Ctrl++"))
        act_in.triggered.connect(self.zoom_in)

        act_out = menu.addAction("缩小 (Zoom Out)")
        act_out.setShortcut(QKeySequence("Ctrl+-"))
        act_out.triggered.connect(self.zoom_out)

        act_fit = menu.addAction("适应图像 (Fit Image)")
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self.fit_to_view)

        act_100 = menu.addAction("1:1 实际像素 (100%)")
        act_100.setShortcut(QKeySequence("Ctrl+1"))
        act_100.triggered.connect(self.set_zoom_100)

        act_split = menu.addAction("原片前后对比")
        act_split.setShortcut(QKeySequence("\\"))
        act_split.triggered.connect(self.requestToggleSplit.emit)

        menu.addSeparator()

        act_undo = menu.addAction("撤销 (Undo)")
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self.requestUndo.emit)

        act_redo = menu.addAction("重做 (Redo)")
        act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        act_redo.triggered.connect(self.requestRedo.emit)

        act_reset = menu.addAction("重置所有参数 (Reset All)")
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.triggered.connect(self.requestReset.emit)

        menu.addSeparator()

        act_quick = menu.addAction("快速导出 (上次参数)")
        act_quick.setShortcut(QKeySequence("Ctrl+Shift+E"))
        act_quick.triggered.connect(self.requestQuickExport.emit)

        act_export = menu.addAction("导出此图像 (Export)...")
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.requestExport.emit)

        menu.exec(event.globalPos())

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = 1.15 if delta > 0 else 0.87
        old_zoom = self.zoom
        new_zoom = max(0.2, min(8.0, self.zoom * factor))
        self.zoom = new_zoom

        # Mouse relative zoom
        mouse_norm_x = (event.position().x() / float(max(1, self.width())) - 0.5) * 2.0
        mouse_norm_y = ((self.height() - event.position().y()) / float(max(1, self.height())) - 0.5) * 2.0

        self.pan_x = mouse_norm_x - (mouse_norm_x - self.pan_x) * (new_zoom / old_zoom)
        self.pan_y = mouse_norm_y - (mouse_norm_y - self.pan_y) * (new_zoom / old_zoom)

        self.zoomChanged.emit(self.get_actual_zoom_ratio())
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on split divider
            norm_x = event.position().x() / float(max(1, self.width()))
            if self.params["view_mode"] == 1 and abs(norm_x - self.params["split_x"]) < 0.02:
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
            self.update()
            event.accept()
        else:
            # Hover cursor check
            if self.params["view_mode"] == 1 and abs(norm_x - self.params["split_x"]) < 0.02:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self._is_split_dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if abs(self.zoom - 1.0) < 0.15:
                self.set_zoom_100()
            else:
                self.fit_to_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def render_offscreen(self, target_w, target_h, float_img_rgb=None, params_override=None, bit_depth=8, lut_override=None):
        """Render image at target resolution on GPU using an offscreen FBO.
        Returns uint8 or uint16 RGB numpy array (target_h, target_w, 3) or None on error.
        """
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
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                data = np.ascontiguousarray(float_img_rgb, dtype=np.float32)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, src_w, src_h, 0, GL.GL_RGB, GL.GL_FLOAT, data)
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

            u_1f("u_exposure", eff_params.get("exposure_ev", 0.0))
            u_1f("u_temp", eff_params.get("color_temp", 5500.0))
            u_1f("u_tint", eff_params.get("tint", 1.0))
            u_1f("u_base_temp", eff_params.get("base_temp", 5500.0))
            u_1f("u_base_tint", eff_params.get("base_tint", 1.0))
            u_3f("u_cmy", eff_params.get("enlarger_cyan", 0.0), eff_params.get("enlarger_magenta", 0.0), eff_params.get("enlarger_yellow", 0.0))
            u_1f("u_print_exposure", eff_params.get("print_exposure", 1.0))
            u_1f("u_preflash", eff_params.get("pre_flash", 0.0))

            u_1f("u_halation", eff_params.get("halation", 0.5))
            u_1i("u_halation_bounces", int(eff_params.get("halation_bounces", 2)))
            u_1f("u_halation_decay", eff_params.get("halation_decay", 0.5))
            u_1f("u_halation_boost", eff_params.get("halation_boost", 0.0))

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

            u_1f("u_split_x", 0.5)
            u_1i("u_view_mode", 0)  # Always render final developed image

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
