"""fotufilm 点式链的 GPU 实现。

取代 `resources/fotufilm_chain/<卷>__<纸>__65.npz` 那张"整条链烘成一张表"的做法：
65³ 在胶片 toe 底附近近似失效（测量：彩色卷 16–70/255，见 docs §13），而且 DIR 被烘死、
一动滑块就只能换引擎。这里把链拆回上游的原生结构，逐像素算：

    scene-linear Rec.2020
      → 色度归一化 + exposure 33³（**四面体**，实测用线性会差 27/255）
      → log10 → 胶片曲线 → DIR（Hill 释放 → K 3×3 → warp 查表）
      → 胶片曲线 → (complement) → 净密度
      → filmOut 33³（硬件三线性）→ 相纸曲线 → paperOut 33³
      → P3 → sRGB

表的数据来自 `resources/fotufilm_chain/point/`（见 scratch/fotufilm_port/export_gpu_assets.py）。

为什么 exposure 必须四面体而另外两张可以用硬件线性：上游引擎自己就是这么分的
（`SpectralLUT.sample` 是四面体，film/paper 走 Halide `texture3d` 的硬件线性）。
把 exposure 也换成线性，实测引入 27/255。
"""

from __future__ import annotations

import os

import numpy as np

try:                                            # 无 GL 上下文时（纯导入/测试）不炸
    from OpenGL import GL
except Exception:                               # noqa: BLE001
    GL = None

# --- 常量（与 chain_min / bake 脚本同值，ui/ 不能依赖 scratch/ 所以复制）-----------------
# Rec.2020 亮度权重，chain_min.LUMA_2020
LUMA_2020 = (0.2627002, 0.6779981, 0.0593017)
# Rec.2020 → 染料域，chain_min.M_2020_TO_DOMAIN
M_2020_TO_DOMAIN = (
    (0.670231843, 0.152168745, 0.177599412),
    (0.044501114, 0.854482372, 0.101016514),
    (0.0,          0.025777047, 0.974222953),
)
# scene-linear ProPhoto → scene-linear Rec.2020（画布给的是 ProPhoto，链期望 Rec.2020）
M_PP_TO_2020 = (
    (1.200676622, -0.057464108, -0.143130619),
    (-0.069923930, 1.080593273, -0.010682235),
    (0.005535414, -0.040767654, 1.035017910),
)
# 线性 P3 → 线性 sRGB（chain_min 的 inv(M_SRGB2XYZ) @ M_P3_2XYZ）
M_P3_TO_SRGB = (
    ( 1.224745485, -0.224904439, -0.0000000365),
    (-0.042058082,  1.042080996,  0.0000000030),
    (-0.019642260, -0.078654881,  1.098537162),
)

CURVE_SAMPLES = 2048
WARP_SAMPLES = 128
WARP_LOW, WARP_HIGH = -4.0, 4.0
TABLE_DIM = 33

FFCHAIN_GLSL = r"""
// ---------------------------------------------------------------- fotufilm 点式链
uniform int       u_ff_on;
uniform sampler3D u_ff_exposure;      // 33³ RGBA：色度归一化后的曝光
uniform sampler3D u_ff_film;          // 33³ RGBA：净密度 → 相纸相对曝光
uniform sampler3D u_ff_paper;         // 33³ RGBA：相纸激活 → 显示（线性 P3）
uniform sampler2D u_ff_curve_film;    // 2048×3：logE → 密度
uniform sampler2D u_ff_curve_paper;   // 2048×3
uniform sampler3D u_ff_warp;          // 128×3×17：DIR 中性锚定补偿，按总量分档
uniform vec3      u_ff_luma;
uniform mat3      u_ff_pp_to_2020;
uniform mat3      u_ff_domain;
uniform mat3      u_ff_p3_to_srgb;
uniform vec3      u_ff_dmin;
uniform vec3      u_ff_rng;
uniform vec3      u_ff_pdmin;
uniform vec3      u_ff_prng;
uniform vec3      u_ff_mid;
uniform vec3      u_ff_mask;
uniform vec3      u_ff_release;
uniform mat3      u_ff_K;
uniform float     u_ff_scale;
uniform vec2      u_ff_range;         // 曲线网格 min,max
uniform int       u_ff_complement;
uniform int       u_ff_warp_nl;
uniform float     u_ff_warp_last;     // 档位表的最后一档总量（2.0）
uniform float     u_ff_dir;           // DIR 总量（1.0 = 上游默认）
// 供体层（donor）：只释放抑制剂，本身不进画面。曝光在 exposure 表的第 4 通道。
uniform int       u_ff_has_donor;
uniform vec4      u_ff_donor_a;       // dMin, gamma, toe, toeWidth
uniform vec2      u_ff_donor_b;       // shoulder, shoulderWidth
uniform vec3      u_ff_donor_release; // 该层释放物对 R/G/B 三个染料层的权重
uniform float     u_ff_donor_gamma;   // 释放指数的单值（DonorCaptureLayer.releaseGamma: Float）
uniform int       u_ff_debug;         // 逐段导出中间量，排查用；0 = 正常

// 四面体插值。必须用 texelFetch：texture() 会把每个角也插一遍，四面体就白做了。
vec4 ff_tetra3(sampler3D t, vec3 p, float dim) {
    vec3 q = clamp(p, 0.0, 1.0) * (dim - 1.0);
    ivec3 i0 = ivec3(min(floor(q), vec3(dim - 2.0)));
    vec3 f = q - vec3(i0);
    vec4 c000 = texelFetch(t, i0, 0);
    vec4 c111 = texelFetch(t, i0 + ivec3(1), 0);
    if (f.x >= f.y) {
        if (f.y >= f.z) {
            vec4 c100 = texelFetch(t, i0 + ivec3(1,0,0), 0);
            vec4 c110 = texelFetch(t, i0 + ivec3(1,1,0), 0);
            return c000 + f.x*(c100-c000) + f.y*(c110-c100) + f.z*(c111-c110);
        }
        if (f.x >= f.z) {
            vec4 c100 = texelFetch(t, i0 + ivec3(1,0,0), 0);
            vec4 c101 = texelFetch(t, i0 + ivec3(1,0,1), 0);
            return c000 + f.x*(c100-c000) + f.z*(c101-c100) + f.y*(c111-c101);
        }
        vec4 c001 = texelFetch(t, i0 + ivec3(0,0,1), 0);
        vec4 c101 = texelFetch(t, i0 + ivec3(1,0,1), 0);
        return c000 + f.z*(c001-c000) + f.x*(c101-c001) + f.y*(c111-c101);
    }
    if (f.x >= f.z) {
        vec4 c010 = texelFetch(t, i0 + ivec3(0,1,0), 0);
        vec4 c110 = texelFetch(t, i0 + ivec3(1,1,0), 0);
        return c000 + f.y*(c010-c000) + f.x*(c110-c010) + f.z*(c111-c110);
    }
    if (f.y >= f.z) {
        vec4 c010 = texelFetch(t, i0 + ivec3(0,1,0), 0);
        vec4 c011 = texelFetch(t, i0 + ivec3(0,1,1), 0);
        return c000 + f.y*(c010-c000) + f.z*(c011-c010) + f.x*(c111-c011);
    }
    vec4 c001 = texelFetch(t, i0 + ivec3(0,0,1), 0);
    vec4 c011 = texelFetch(t, i0 + ivec3(0,1,1), 0);
    return c000 + f.z*(c001-c000) + f.y*(c011-c001) + f.x*(c111-c011);
}

// 手写三线性（texelFetch）：与 chain_min.trilinear 同一套算术。
// 原本用 texture() 走硬件过滤，结果与 Python 对不上，改成显式插值后可控。
vec4 ff_linear3(sampler3D t, vec3 p, float dim) {
    vec3 q = clamp(p, 0.0, 1.0) * (dim - 1.0);
    ivec3 i0 = ivec3(min(floor(q), vec3(dim - 2.0)));
    vec3 f = q - vec3(i0);
    vec4 c00 = mix(texelFetch(t, i0 + ivec3(0,0,0), 0),
                   texelFetch(t, i0 + ivec3(1,0,0), 0), f.x);
    vec4 c10 = mix(texelFetch(t, i0 + ivec3(0,1,0), 0),
                   texelFetch(t, i0 + ivec3(1,1,0), 0), f.x);
    vec4 c01 = mix(texelFetch(t, i0 + ivec3(0,0,1), 0),
                   texelFetch(t, i0 + ivec3(1,0,1), 0), f.x);
    vec4 c11 = mix(texelFetch(t, i0 + ivec3(0,1,1), 0),
                   texelFetch(t, i0 + ivec3(1,1,1), 0), f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

// texelFetch + 手动线性插值：与 Python 侧同一套算术，不经过归一化坐标和过滤状态
float ff_curve(sampler2D t, int ch, float x) {
    int n = textureSize(t, 0).x;
    float q = clamp((x - u_ff_range.x) * float(n - 1) / (u_ff_range.y - u_ff_range.x),
                    0.0, float(n - 1));
    int i0 = min(int(q), n - 2);
    float f = q - float(i0);
    float lo = texelFetch(t, ivec2(i0, ch), 0).r;
    float hi = texelFetch(t, ivec2(i0 + 1, ch), 0).r;
    return lo + f * (hi - lo);
}

// Curves.h::curve_density —— 供体层用的是**解析**曲线，不是查表（它只有 6 个参数，
// 没有 filmCurve 那样的 2048 点表；两者在负片的宽肩下等价）。
float ff_softplus(float z) {
    // 数值稳定写法：max(z,0) + log(1 + e^{-|z|})，避免 exp 溢出
    return max(z, 0.0) + log(1.0 + exp(-abs(z)));
}

float ff_analytic_density(vec4 a, vec2 b, float x) {
    float dmin = a.x, gamma = a.y, toe = a.z;
    float toe_w = max(a.w, 1e-6);
    float sh = b.x, sh_w = max(b.y, 1e-6);
    float t = toe_w * ff_softplus((x - toe) / toe_w);
    float s = sh_w * ff_softplus((x - sh) / sh_w);
    return dmin + gamma * clamp(t - s, 0.0, sh - toe);
}

// Couplers.h::coupler_warp —— 中性锚定表的反查，线性 / 单调三次两条分支。
// 表按 DIR 总量分档（17 档，0..2 步长 0.125）：warp 与 K 是同一组联立解，
// 总量变了 u 就变，单张表在新 u 上不再满足 u+warp(u)=logE，中性灰会整幅偏色。
float ff_warp_band(int ch, int band, float u) {
    float n = float(textureSize(u_ff_warp, 0).x);
    float q = clamp((u - (-4.0)) * (n - 1.0) / 8.0, 0.0, n - 1.0);
    int i0 = min(int(q), int(n) - 2);
    float f = q - float(i0);
    float lo = texelFetch(u_ff_warp, ivec3(i0, ch, band), 0).r;
    float hi = texelFetch(u_ff_warp, ivec3(i0 + 1, ch, band), 0).r;
    float linear = lo + f * (hi - lo);
    if (u_ff_warp_nl == 0) return linear;
    float prev = texelFetch(u_ff_warp, ivec3(max(i0 - 1, 0), ch, band), 0).r;
    float foll = texelFetch(u_ff_warp, ivec3(min(i0 + 2, int(n) - 1), ch, band), 0).r;
    float delta = max(hi - lo, 0.0);
    float low_slope = clamp(0.5 * (hi - prev), 0.0, 3.0 * delta);
    float high_slope = clamp(0.5 * (foll - lo), 0.0, 3.0 * delta);
    float f2 = f * f, f3 = f2 * f;
    return (2.0*f3 - 3.0*f2 + 1.0) * lo + (f3 - 2.0*f2 + f) * low_slope
         + (-2.0*f3 + 3.0*f2) * hi + (f3 - f2) * high_slope;
}

float ff_warp(int ch, float u) {
    // 档位索引 = 总量 / 0.125；在相邻两档间线性插值（档间残差实测 0.4/255）。
    // 上界从纹理深度取，别用 u_ff_warp_last 换算（会算成 1，最后一档就取不到）。
    int depth = textureSize(u_ff_warp, 0).z;
    float bf = clamp(u_ff_dir, 0.0, u_ff_warp_last) * 8.0;
    int b0 = min(int(bf), depth - 2);
    float fb = bf - float(b0);
    if (fb <= 0.0) return ff_warp_band(ch, b0, u);
    return mix(ff_warp_band(ch, b0, u), ff_warp_band(ch, b0 + 1, u), fb);
}

// 上游的结构是 develop → 净密度 → 颗粒 → print，这里照做。过去整条链是个黑盒，
// 颗粒只能从 33³ 表反推密度，实测亮部误差 0.221（≈0.73 档），而亮部正是颗粒最显眼
// 的地方。切开之后 net 是逐像素算出来的浮点数，没有反推这一步。
//
// 返回 vec4 是为了让诊断码逐位保持不变：alpha 为 1 表示"这是 u_ff_debug 命中的诊断色，
// 不是密度，调用方别往下算"。scratch 里那些逐段对比脚本因此不用改。
vec4 ffchain_develop(vec3 prophoto) {
    if (u_ff_debug == 70) return vec4(clamp(prophoto, 0.0, 1.0), 1.0);
    vec3 rec = u_ff_pp_to_2020 * prophoto;
    if (u_ff_debug == 71) return vec4(clamp(rec, 0.0, 1.0), 1.0);
    // 1. 色度归一化后采样曝光表（四面体）
    float yv = dot(rec, u_ff_luma);
    vec3 d = u_ff_domain * rec;
    // 显式分量，不用 d[c] 动态索引：部分驱动对 vec3 的动态索引返回未定义值
    vec3 tr = vec3(1.0);
    if (d.x < 0.0) tr.x = yv / max(yv - d.x, 1e-30);
    if (d.y < 0.0) tr.y = yv / max(yv - d.y, 1e-30);
    if (d.z < 0.0) tr.z = yv / max(yv - d.z, 1e-30);
    float s = min(tr.x, min(tr.y, tr.z));
    bool outside = any(lessThan(d, vec3(0.0)));
    vec3 pv = yv <= 0.0 ? vec3(0.0) : (outside ? yv + s * (d - yv) : d);
    vec3 w = max(pv, vec3(0.0));
    float scale = max(max(w.x, max(w.y, w.z)), 1e-6);
    vec3 E = ff_tetra3(u_ff_exposure, w / scale, 33.0).rgb * (scale / 0.18);
    vec3 logE = log(max(E, vec3(1e-12))) * 0.4342944819;   // 1/ln10
    if (u_ff_debug == 1) return vec4(w / scale, 1.0);
    if (u_ff_debug == 2) return vec4(clamp(logE * 0.2 + 0.5, 0.0, 1.0), 1.0);

    // 2. DIR：释放 → 跨层抑制 → 中性锚定补偿
    vec3 formed0 = vec3(ff_curve(u_ff_curve_film, 0, logE.x),
                        ff_curve(u_ff_curve_film, 1, logE.y),
                        ff_curve(u_ff_curve_film, 2, logE.z));
    vec3 a = clamp((formed0 - u_ff_dmin) / u_ff_rng, 0.0, 1.0);
    vec3 ga = pow(max(a, vec3(1e-8)), u_ff_release);
    vec3 gb = pow(max(1.0 - a, vec3(1e-8)), u_ff_release);
    vec3 rel = ga / max(ga + gb, vec3(1e-8));
    vec3 inhib = (u_ff_K * rel) * (u_ff_scale * u_ff_dir);

    // 2.5 供体层（donor）：只释放抑制剂，本身不进画面。
    // 它的曝光在 exposure 表的第 4 通道；色度点与主通道完全相同，只是取另一条记录。
    // 中性轴上供体与三层曝光相同（共享同一套中性归一化），所以直接复用 w / scale。
    if (u_ff_has_donor == 1) {
        float e_donor = ff_tetra3(u_ff_exposure, w / scale, 33.0).a * (scale / 0.18);
        // 下限用 1e-6，与 chain_min 的 donor 分支一致（主通道用的才是 1e-12）
        float donor_log = log(max(e_donor, 1e-6)) * 0.4342944819;
        float dens = ff_analytic_density(u_ff_donor_a, u_ff_donor_b, donor_log);
        float drange = max(u_ff_donor_a.y * (u_ff_donor_b.x - u_ff_donor_a.z), 1e-6);
        float dact = clamp((dens - u_ff_donor_a.x) / drange, 0.0, 1.0);
        float drel;
        if (u_ff_donor_gamma == 1.0) {
            drel = dact;                    // gamma 1 是精确线性兼容模式
        } else {
            float ra = pow(max(dact, 1e-8), u_ff_donor_gamma);
            float rb = pow(max(1.0 - dact, 1e-8), u_ff_donor_gamma);
            drel = ra / max(ra + rb, 1e-8);
        }
        inhib += u_ff_donor_release * drel * (u_ff_scale * u_ff_dir);
    }

    vec3 u = logE - inhib;
    vec3 eff = vec3(u.x + ff_warp(0, u.x),
                    u.y + ff_warp(1, u.y),
                    u.z + ff_warp(2, u.z));

    if (u_ff_debug == 3) return vec4(clamp(eff * 0.2 + 0.5, 0.0, 1.0), 1.0);
    if (u_ff_debug == 8) return vec4(clamp(inhib * 0.2 + 0.5, 0.0, 1.0), 1.0);
    if (u_ff_debug == 9) return vec4(clamp(formed0 / 3.0, 0.0, 1.0), 1.0);
    if (u_ff_debug == 10) {
        ivec2 s2 = textureSize(u_ff_curve_film, 0);
        return vec4(float(s2.x) / 2048.0, float(s2.y) / 3.0, 0.0, 1.0);
    }
    if (u_ff_debug == 11) return vec4(ff_curve(u_ff_curve_film, 0, 0.0) / 3.0,
                                      ff_curve(u_ff_curve_film, 1, 0.0) / 3.0,
                                      ff_curve(u_ff_curve_film, 2, 0.0) / 3.0, 1.0);
    if (u_ff_debug == 60) return vec4((u_ff_pp_to_2020 * vec3(0.2, 0.5, 0.8)) * 0.5, 1.0);
    if (u_ff_debug == 61) return vec4((u_ff_domain * vec3(0.2, 0.5, 0.8)) * 0.5, 1.0);
    if (u_ff_debug == 50) return vec4(texelFetch(u_ff_exposure, ivec3(32,32,32), 0).rgb, 1.0);
    if (u_ff_debug == 51) return vec4(texelFetch(u_ff_exposure, ivec3(0,0,0), 0).rgb, 1.0);
    if (u_ff_debug == 52) return vec4(texelFetch(u_ff_exposure, ivec3(16,16,16), 0).rgb, 1.0);
    if (u_ff_debug == 53) return vec4(texelFetch(u_ff_exposure, ivec3(32,0,0), 0).rgb, 1.0);
    if (u_ff_debug == 40) return vec4((u_ff_domain * vec3(1.0)) / 3.0, 1.0);
    if (u_ff_debug == 41) return vec4((u_ff_pp_to_2020 * vec3(1.0)) / 3.0, 1.0);
    if (u_ff_debug == 42) return vec4((u_ff_K * vec3(1.0)) / 3.0, 1.0);
    if (u_ff_debug == 12) return vec4((u_ff_range.x + 8.0) / 16.0, u_ff_range.y / 16.0, 0.0, 1.0);
    if (u_ff_debug == 13) return vec4(u_ff_dmin / 3.0, 1.0);
    if (u_ff_debug == 14) return vec4(vec3(ff_curve(u_ff_curve_film, 0, 0.0),
                                           ff_curve(u_ff_curve_film, 1, 0.0),
                                           ff_curve(u_ff_curve_film, 2, 0.0)) / 3.0 * 0.0
                                      + vec3(float(textureSize(u_ff_curve_film, 0).x) / 2048.0,
                                             (0.0 + 0.5) / 3.0, 0.0), 1.0);
    // 3. 显影 → 净密度
    vec3 formed = vec3(ff_curve(u_ff_curve_film, 0, eff.x),
                       ff_curve(u_ff_curve_film, 1, eff.y),
                       ff_curve(u_ff_curve_film, 2, eff.z));
    vec3 developed = u_ff_complement == 1
        ? (u_ff_dmin + u_ff_rng - (formed - u_ff_dmin)) : formed;
    vec3 net = developed - u_ff_dmin;

    if (u_ff_debug == 4) return vec4(clamp(net / max(u_ff_rng, vec3(1e-6)), 0.0, 1.0), 1.0);
    return vec4(net, 0.0);          // alpha 0：这是真密度，调用方可以往里插颗粒
}

// 4. filmOut → 相纸曲线 → paperOut，出口是显示值。
// 输入是**净密度**（不是 logE，不是显示值），所以颗粒只要加在调用点之间就够了。
vec3 ffchain_print(vec3 net) {
    vec3 rel2 = ff_linear3(u_ff_film, net / u_ff_rng, 33.0).rgb;
    vec3 pact;
    pact.x = (ff_curve(u_ff_curve_paper, 0, u_ff_mid.x + u_ff_mask.x * rel2.x) - u_ff_pdmin.x)
             / u_ff_prng.x;
    pact.y = (ff_curve(u_ff_curve_paper, 1, u_ff_mid.y + u_ff_mask.y * rel2.y) - u_ff_pdmin.y)
             / u_ff_prng.y;
    pact.z = (ff_curve(u_ff_curve_paper, 2, u_ff_mid.z + u_ff_mask.z * rel2.z) - u_ff_pdmin.z)
             / u_ff_prng.z;
    if (u_ff_debug == 5) return clamp(rel2, 0.0, 1.0);
    if (u_ff_debug == 6) return clamp(pact, 0.0, 1.0);
    vec3 lin_p3 = ff_linear3(u_ff_paper, pact, 33.0).rgb;
    if (u_ff_debug == 7) return clamp(lin_p3, 0.0, 1.0);

    // 5. P3 → sRGB 线性 → 编码
    vec3 lin_srgb = u_ff_p3_to_srgb * max(lin_p3, vec3(0.0));
    vec3 v = clamp(lin_srgb, 0.0, 1.0);
    return mix(v * 12.92, 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), v));
}

// 整条链一次算完的兼容入口。诊断脚本和"不要颗粒"的调用点走这里。
vec3 ffchain_run(vec3 prophoto) {
    vec4 d = ffchain_develop(prophoto);
    if (d.a != 0.0) return d.rgb;        // debug 命中：诊断色直出，不进 print
    return ffchain_print(d.rgb);
}
"""


def _reset_unpack():
    """清掉外部代码可能留下的 UNPACK 状态。

    canvas 上传 LUT / grain 立方体时设过 ROW_LENGTH / IMAGE_HEIGHT 之类，
    留着会让我这边的数据被按错误的行跨度读取 —— 表现是纹理尺寸对、内容全错
    （实测 curve(0) 该是 0.89 却读到 0.19）。
    """
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    for cap in (0x0CF2, 0x0CF3, 0x0CF4, 0x0CF5, 0x0CF6):  # ROW_LENGTH..SKIP_IMAGES
        try:
            GL.glPixelStorei(cap, 0)
        except Exception:                           # noqa: BLE001 - 旧上下文可能不认
            pass


class GLFFChain:
    """Owns the tables this stage reads. One instance per canvas context."""

    def __init__(self):
        self._tex = {}
        self._loc = {}
        self.scalars = {}
        self.ready = False
        self.key = None
        self.debug = 0
        self._pending = None
        self._signature = None

    # -- staging ---------------------------------------------------------------------------
    def set_bundle(self, bundle):
        """Stage a bundle for the next paint; ``None`` turns the path off.

        签名用内容（key + 各表形状），不用 id()——对象地址会被 Python 回收复用，
        换卷后可能"看起来没变"而沿用上一支卷的表（grain 那边踩过这个坑）。
        """
        if bundle is None:
            self._pending = None
            self._signature = None
            self.key = None
            self.ready = False
            return
        signature = (bundle.get("key"),
                     np.shape(bundle.get("exposure")),
                     np.shape(bundle.get("filmCurve")))
        if signature == self._signature:
            return
        self._signature = signature
        self.key = bundle.get("key")
        self._pending = bundle

    # -- upload ----------------------------------------------------------------------------
    def upload_pending(self, unit_base=8):
        """Push a staged bundle to the GPU. Call with the canvas context current."""
        p = self._pending
        if p is None or GL is None:
            return
        self._tex["exposure"] = self._upload3d(self._tex.get("exposure", 0),
                                               p["exposure"], unit_base + 0)
        self._tex["film"] = self._upload3d(self._tex.get("film", 0),
                                           p["filmOut"], unit_base + 1)
        self._tex["paper"] = self._upload3d(self._tex.get("paper", 0),
                                            p["paperOut"], unit_base + 2)
        self._tex["curveFilm"] = self._upload2d(self._tex.get("curveFilm", 0),
                                                self._rows(p["filmCurve"]), unit_base + 3)
        self._tex["curvePaper"] = self._upload2d(self._tex.get("curvePaper", 0),
                                                 self._rows(p["paperCurve"]), unit_base + 4)
        # warp：多档中性锚定表 (17,3,128) → 3D 纹理 [band][ch][u]，
        # 这样 texelFetch(i, ch, band) 正好取到 bands[band][ch][i]。
        # 兼容只带单张 warp 的旧包（退化成 1 档）。
        if "warpBands" in p:
            bands = np.asarray(p["warpBands"], np.float32)
            scales = np.asarray(p["warpScales"], np.float32)
        else:
            bands = np.asarray(p["warp"], np.float32)[None]
            scales = np.array([float(np.ravel(p["couplerScale"])[0])], np.float32)
        wdata = np.zeros(bands.shape + (4,), np.float32)
        wdata[:, :, :, 0] = bands
        self._tex["warp"] = self._upload3d(self._tex.get("warp", 0), wdata, unit_base + 5)
        self._warp_last = float(scales[-1])

        self.scalars = {
            "dmin": np.asarray(p["dmin"], np.float32),
            "rng": np.asarray(p["rng"], np.float32),
            "pdmin": np.asarray(p["paperDmin"], np.float32),
            "prng": np.asarray(p["paperRng"], np.float32),
            "mid": np.asarray(p["midpoints"], np.float32),
            "mask": np.asarray(p["masking"], np.float32),
            "release": np.asarray(p["releaseGamma"], np.float32),
            "K": np.asarray(p["K"], np.float32),
            # ascontiguousarray 会把 0 维标量变成 (1,)，统一 ravel 取 [0]
            "scale": float(np.ravel(p["couplerScale"])[0]),
            "range": np.array([float(np.ravel(p["curveMin"])[0]),
                               float(np.ravel(p["curveMax"])[0])], np.float32),
            "complement": int(np.ravel(p["complement"])[0]),
            "warpNl": int(np.any(np.asarray(p["releaseGamma"]) != 1.0)
                          or ("donorGamma" in p and np.any(np.asarray(p["donorGamma"]) != 1.0))),
        }
        if "donorCurve" in p:
            dc = np.asarray(p["donorCurve"], np.float32).ravel()   # dMin,gamma,toe,toeW,sh,shW
            dg = np.asarray(p["donorGamma"], np.float32).ravel()
            self.scalars.update({
                "hasDonor": 1,
                "donorA": np.array([dc[0], dc[1], dc[2], dc[3]], np.float32),
                "donorB": np.array([dc[4], dc[5]], np.float32),
                "donorRel": np.asarray(p["donorRelease"], np.float32).ravel(),
                # 单值：配置 8791 只有一个槽，后两个是别的字段
                "donorG": float(dg[0]),
            })
        else:
            self.scalars.update({
                "hasDonor": 0,
                "donorA": np.zeros(4, np.float32),
                "donorB": np.ones(2, np.float32),
                "donorRel": np.zeros(3, np.float32),
                "donorG": 1.0,
            })
        self.ready = True
        self._pending = None

    @staticmethod
    def _rows_wide(arr):
        """(3, N) → (3, N, 4)：本来就是一通道一行。warp 表是这种布局。"""
        a = np.asarray(arr, np.float32)
        out = np.zeros((3, a.shape[1], 4), np.float32)
        for c in range(3):
            out[c, :, 0] = a[c, :]
        return out

    @staticmethod
    def _rows(arr):
        """(N,3) → (3,N,4)：纹理布局是"每通道一行"，值写在 R。"""
        a = np.asarray(arr, np.float32)
        out = np.zeros((3, a.shape[0], 4), np.float32)
        for c in range(3):
            out[c, :, 0] = a[:, c]
        return out

    @staticmethod
    def _upload3d(tex, array, unit):
        # numpy 是 [z][y][x][c]、最内维最快，正好对上 glTexImage3D 的 width；
        # texelFetch(x,y,z) 读到的就是 arr[z][y][x]，与 chain_min.sample_lut 同序。
        # 允许非立方体（warp 的档位纹理是 128×3×17）。
        data = np.ascontiguousarray(array, dtype=np.float32)
        d, h, w = data.shape[0], data.shape[1], data.shape[2]
        GL.glActiveTexture(0x84C0 + unit)
        _reset_unpack()
        if tex:
            GL.glDeleteTextures([tex])
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        for wname in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R):
            GL.glTexParameteri(GL.GL_TEXTURE_3D, wname, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA16F, w, h, d, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, data)
        _e = GL.glGetError()
        if _e:
            print(f"[FFChain] 3D 上传失败 unit={unit} tex={tex} {w}x{h}x{d} err={_e}")
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        GL.glActiveTexture(0x84C0)
        return tex

    @staticmethod
    def _upload2d(tex, array, unit):
        data = np.ascontiguousarray(array, dtype=np.float32)
        h, w = data.shape[0], data.shape[1]
        GL.glActiveTexture(0x84C0 + unit)
        _reset_unpack()
        if tex:
            GL.glDeleteTextures([tex])
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, data)
        _e = GL.glGetError()
        if _e:
            print(f"[FFChain] 2D 上传失败 unit={unit} tex={tex} {w}x{h} err={_e}")
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(0x84C0)
        return tex

    # -- bind ------------------------------------------------------------------------------
    def _u(self, program, name):
        """`program` 是 QOpenGLShaderProgram（canvas 里那个），也可以直接给 GL id。"""
        loc = self._loc.get(name)
        if loc is not None:
            return loc
        try:
            loc = program.uniformLocation(name)
        except Exception:                           # noqa: BLE001 - 传的是裸 GL id
            loc = GL.glGetUniformLocation(program, name)
        if loc is None or (isinstance(loc, int) and loc < 0):
            loc = -1
        self._loc[name] = loc
        return loc

    def _mat3(self, program, name, m):
        loc = self._u(program, name)
        if loc == -1:
            return
        # GL_FALSE = 数据按列主序解释，numpy 的行主序矩阵要转置后再给
        GL.glUniformMatrix3fv(loc, 1, GL.GL_FALSE,
                              np.ascontiguousarray(np.asarray(m, np.float32).T))

    def _vec3(self, program, name, v):
        loc = self._u(program, name)
        if loc == -1:
            return
        GL.glUniform3fv(loc, 1, np.asarray(v, np.float32).ravel())

    def _float(self, program, name, v):
        loc = self._u(program, name)
        if loc != -1:
            GL.glUniform1f(loc, float(v))

    def _int(self, program, name, v):
        loc = self._u(program, name)
        if loc != -1:
            GL.glUniform1i(loc, int(v))

    def bind(self, program, unit_base=8, dir_total=1.0):
        """Bind everything this stage needs. Returns True when the shader should branch in."""
        if not self.ready or GL is None:
            self._int(program, "u_ff_on", 0)
            return False
        # 总量必须和档位表同域：超出最后一档时 warp 会 clamp 而抑制量继续放大，
        # 中性锚定又会断，所以在这里一起 clamp。
        dir_total = max(0.0, min(float(dir_total), getattr(self, "_warp_last", 2.0)))
        for i, name in enumerate(("exposure", "film", "paper",
                                  "curveFilm", "curvePaper", "warp")):
            self._id_bind(program, name, unit_base + i)
        s = self.scalars
        self._vec3(program, "u_ff_luma", LUMA_2020)
        self._mat3(program, "u_ff_pp_to_2020", M_PP_TO_2020)
        self._mat3(program, "u_ff_domain", M_2020_TO_DOMAIN)
        self._mat3(program, "u_ff_p3_to_srgb", M_P3_TO_SRGB)
        self._vec3(program, "u_ff_dmin", s["dmin"])
        self._vec3(program, "u_ff_rng", s["rng"])
        self._vec3(program, "u_ff_pdmin", s["pdmin"])
        self._vec3(program, "u_ff_prng", s["prng"])
        self._vec3(program, "u_ff_mid", s["mid"])
        self._vec3(program, "u_ff_mask", s["mask"])
        self._vec3(program, "u_ff_release", s["release"])
        self._mat3(program, "u_ff_K", s["K"])
        self._float(program, "u_ff_scale", s["scale"])
        self._int(program, "u_ff_complement", s["complement"])
        self._int(program, "u_ff_warp_nl", s["warpNl"])
        self._float(program, "u_ff_warp_last", getattr(self, "_warp_last", 2.0))
        self._float(program, "u_ff_dir", dir_total)
        self._int(program, "u_ff_has_donor", s["hasDonor"])
        if s["hasDonor"]:
            loc = self._u(program, "u_ff_donor_a")
            if loc != -1:
                GL.glUniform4fv(loc, 1, np.asarray(s["donorA"], np.float32).ravel())
            loc = self._u(program, "u_ff_donor_b")
            if loc != -1:
                GL.glUniform2fv(loc, 1, np.asarray(s["donorB"], np.float32).ravel())
            self._vec3(program, "u_ff_donor_release", s["donorRel"])
            self._float(program, "u_ff_donor_gamma", s["donorG"])
        self._int(program, "u_ff_debug", self.debug)
        loc = self._u(program, "u_ff_range")
        if loc != -1:
            GL.glUniform2fv(loc, 1, np.asarray(s["range"], np.float32).ravel())
        self._int(program, "u_ff_on", 1)
        GL.glActiveTexture(0x84C0)
        return True

    _TARGETS = {
        "exposure": 0x806F,   # GL_TEXTURE_3D
        "film": 0x806F,
        "paper": 0x806F,
        "curveFilm": 0x0DE1,  # GL_TEXTURE_2D
        "curvePaper": 0x0DE1,
        "warp": 0x806F,       # GL_TEXTURE_3D（按 DIR 总量分档）
    }

    # 纹理键名 → GLSL uniform 名。不能直接 "u_ff_" + name：curveFilm 对的是
    # u_ff_curve_film，拼错了 sampler 就永远没人设单元号、默认读 unit 0 的原图。
    _UNIFORMS = {
        "exposure": "u_ff_exposure",
        "film": "u_ff_film",
        "paper": "u_ff_paper",
        "curveFilm": "u_ff_curve_film",
        "curvePaper": "u_ff_curve_paper",
        "warp": "u_ff_warp",
    }

    def _id_bind(self, program, name, unit):
        """设 sampler 单元号**并重新绑定纹理**。

        上传末尾把纹理解绑了，只设单元号的话 sampler3D 会指向一个空 unit，
        glDrawArrays 直接报 GL_INVALID_OPERATION（1282）。
        """
        tex = self._tex.get(name)
        if not tex:
            return
        target = self._TARGETS[name]
        GL.glActiveTexture(0x84C0 + unit)
        GL.glBindTexture(target, tex)
        loc = self._u(program, self._UNIFORMS[name])
        if loc != -1:
            GL.glUniform1i(loc, int(unit))

    def release(self):
        if GL is None:
            return
        for tex in self._tex.values():
            if tex:
                GL.glDeleteTextures([tex])
        self._tex.clear()
        self._loc.clear()
        self.ready = False
