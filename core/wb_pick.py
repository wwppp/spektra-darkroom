"""白平衡吸管的核心数学：屏幕坐标 → 纹理像素 → 反解色温/色调。

三条口径必须与画布严格一致，任何一条偏了吸管就会读错位置或算错白平衡：

1. **色彩域**：画布纹理 `_float_img_rgb` 是**场景线性 ProPhoto、未做白平衡**的原始预览，
   白平衡是 shader 里 `u_scene_gain` 乘上去的。所以取样只能从纹理取，不能读屏幕像素
   （屏幕上那一帧已经过 LUT、相纸、颗粒和显示域）。

2. **坐标**：`get_image_bounds_norm()` 给的是**Qt 顶左原点**的照片矩形。quad 上边缘
   对应纹理第 0 行（图像顶），下边缘对应最后一行。

3. **反解**：与 `core.darkroom_math.wb_gains` 互逆：

       g = [2^(W·s), tint/base_tint, 2^(-W·s)]，s = log2(temp / base_temp)
       s    = log2(c.B / c.R) / (2W)
       temp = base_temp · 2^s
       tint = base_tint · (c.R / c.G) · 2^(W·s)

   白平衡增益最后会按 ProPhoto 亮度归一化，那只影响整体亮度、不影响通道比值，
   所以反解不需要它。
"""

import math

import numpy as np

from core.darkroom_math import WB_STRENGTH

TEMP_MIN = 2500.0
TEMP_MAX = 9500.0
TINT_MIN = 0.70
TINT_MAX = 1.30
TEMP_STEP = 25.0
TINT_STEP = 0.01

# 取样时剔除"已经过曝/死黑"的像素：那些点的通道比值不可信（clip 后 R/B 比会饱和）。
CLIP_HIGH = 0.985
CLIP_LOW = 2.0e-4


def solve_wb_from_linear(rgb, base_temp=5500.0, base_tint=1.0):
    """由一段场景线性 ProPhoto 颜色反解 (色温, 色调)。

    返回 dict：
        temp / tint       —— 已按滑块步进取整并夹到行程内的落点
        raw_temp / raw_tint —— 夹之前的值（用于判断触界）
        clipped_temp / clipped_tint —— 是否撞到滑块边界
        valid / reason    —— 通道退化（黑点、单一通道为 0）时给 False
    """
    out = {"temp": None, "tint": None, "raw_temp": None, "raw_tint": None,
           "clipped_temp": False, "clipped_tint": False, "valid": False, "reason": ""}
    if rgb is None:
        out["reason"] = "no_sample"
        return out
    c = np.asarray(rgb, dtype=np.float64).reshape(-1)[:3]
    if c.size < 3 or not np.all(np.isfinite(c)):
        out["reason"] = "bad_sample"
        return out
    r, g, b = float(c[0]), float(c[1]), float(c[2])
    if r <= 1e-7 or b <= 1e-7 or g <= 1e-7:
        out["reason"] = "degenerate"
        return out

    s = math.log2(b / r) / (2.0 * WB_STRENGTH)
    raw_temp = float(base_temp) * (2.0 ** s)
    raw_tint = float(base_tint) * (r / g) * (2.0 ** (WB_STRENGTH * s))

    out["raw_temp"] = raw_temp
    out["raw_tint"] = raw_tint

    temp = min(TEMP_MAX, max(TEMP_MIN, raw_temp))
    tint = min(TINT_MAX, max(TINT_MIN, raw_tint))
    out["clipped_temp"] = abs(temp - raw_temp) > 1e-6
    out["clipped_tint"] = abs(tint - raw_tint) > 1e-9

    out["temp"] = round(temp / TEMP_STEP) * TEMP_STEP
    out["tint"] = round(tint / TINT_STEP) * TINT_STEP
    out["valid"] = True
    return out


def reduce_patch(patch, reject_clip=True):
    """把 HxWx3 的线性取样块压成一个颜色；过曝/死黑像素先剔掉，全被剔掉就退回全量均值。"""
    if patch is None:
        return None
    arr = np.asarray(patch, dtype=np.float32)
    if arr.size == 0:
        return None
    flat = arr.reshape(-1, 3)
    if reject_clip and flat.shape[0] > 1:
        peak = flat.max(axis=1)
        keep = (peak < CLIP_HIGH) & (peak > CLIP_LOW)
        if keep.any():
            flat = flat[keep]
    return flat.mean(axis=0)


def quad_uv_from_widget(pos_x, pos_y, widget_w, widget_h, bounds):
    """widget 像素坐标 → 照片 quad 内的归一化坐标（0..1，Qt 顶左原点）。

    `bounds` 是 `get_image_bounds_norm()` 的 (left, right, top, bottom)，归一化到控件尺寸。
    quad 外的点返回 None（letterbox 黑边）。
    """
    l, r, tp, bt = bounds
    w = max(1, int(widget_w))
    h = max(1, int(widget_h))
    if r - l <= 1e-6 or bt - tp <= 1e-6:
        return None
    u = (float(pos_x) / w - l) / (r - l)
    v = (float(pos_y) / h - tp) / (bt - tp)
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return None
    return u, v


def texture_uv_from_quad_uv(u, v, crop=None, native_aspect=1.5):
    """quad 内坐标 → 纹理坐标。逐行照抄 shader 的"普通视图（非裁切编辑）"分支。

    quad 上边缘 = 图像顶（纹理第 0 行），所以这里先按 shader 的约定取
    img_uv = (u, 1 - v)，再走 flip / rot_90 / 摆正角，最后得到 sample_uv。
    调用方负责把 sample_uv 换回数组下标（见 `texture_pixel_index`）。
    """
    crop = crop if isinstance(crop, dict) else {}
    enabled = bool(crop.get("enabled", False))
    img_x = float(u)
    img_y = 1.0 - float(v)

    if not enabled:
        return img_x, 1.0 - img_y

    cx = float(crop.get("x", 0.0))
    cy = float(crop.get("y", 0.0))
    cw = max(1e-6, float(crop.get("w", 1.0)))
    ch = max(1e-6, float(crop.get("h", 1.0)))
    rot = int(crop.get("rot_90", 0)) % 360
    flip_h = bool(crop.get("flip_h", False))
    flip_v = bool(crop.get("flip_v", False))
    angle = float(crop.get("angle", 0.0) or 0.0)

    # 裁切区中心 + 偏移（shader 里对 img_uv.y 先做了 1- 翻转）
    center_x = cx + cw * 0.5
    center_y = cy + ch * 0.5
    px = center_x + (img_x - 0.5) * cw
    py = center_y + ((1.0 - img_y) - 0.5) * ch

    if flip_h:
        px = 1.0 - px
    if flip_v:
        py = 1.0 - py

    if rot == 90:
        px, py = py, 1.0 - px
    elif rot == 180:
        px, py = 1.0 - px, 1.0 - py
    elif rot == 270:
        px, py = 1.0 - py, px

    if abs(angle) > 1e-4:
        eff_aspect = (1.0 / native_aspect) if rot in (90, 270) else native_aspect
        dx = (px - 0.5) * eff_aspect
        dy = (py - 0.5)
        ca = math.cos(angle)
        sa = math.sin(-angle)
        px = 0.5 + (dx * ca - dy * sa) / eff_aspect
        py = 0.5 + (dx * sa + dy * ca)

    # 回成 shader 的 sample_uv 口径（y 已翻转过一次）
    return px, py


def texture_pixel_index(sample_uv, width, height):
    """sample_uv → 数组下标 (row, col)。纹理第 0 行 = 数组第 0 行。"""
    w = max(1, int(width))
    h = max(1, int(height))
    x = min(1.0, max(0.0, float(sample_uv[0])))
    y = min(1.0, max(0.0, float(sample_uv[1])))
    col = int(round(x * (w - 1)))
    row = int(round(y * (h - 1)))
    return row, col
