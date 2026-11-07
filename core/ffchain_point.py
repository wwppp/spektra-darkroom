# -*- coding: utf-8 -*-
"""点式链的 numpy 实现：**只在格点上求值**，给 LUT 烘焙用。

为什么需要它：画布走的是 `ui/gl_ffchain.py` 的 GLSL，逐像素；但缩略图、胶片条卡片、
CPU 导出这三类消费者要的是一张 `size³` 的显示 LUT（`get_3d_lut`）。以前那张表是离线烘的
`<卷>__<纸>__65.npz`（29 MB），而它已经不需要精确——只要在 `size³` 个格点上和 GLSL
同值即可，17³ = 4913 点、33³ = 35937 点，向量化一遍是毫秒级。

逐行对应 GLSL 的顺序，不要"顺手优化"：两条实现必须能互相对拍（判据 ≤1e-3，
见 docs/fotufilm-port-plan-2026-11-06.md §13 / §22.2）。

坐标系约定与 `GLFFChain._upload3d` 一致：numpy 是 [z][y][x][c]，
`texelFetch(x,y,z)` 取到的就是 `table[z][y][x]`。
"""

from __future__ import annotations

import numpy as np

# --- 常量（与 ui/gl_ffchain.py / chain_min 同值，改一处要三处一起改）---------------------------
LUMA_2020 = np.array([0.2627002, 0.6779981, 0.0593017], dtype=np.float64)

M_2020_TO_DOMAIN = np.array([
    [0.670231843, 0.152168745, 0.177599412],
    [0.044501114, 0.854482372, 0.101016514],
    [0.0,         0.025777047, 0.974222953],
], dtype=np.float64)

M_PP_TO_2020 = np.array([
    [1.200676622, -0.057464108, -0.143130619],
    [-0.069923930, 1.080593273, -0.010682235],
    [0.005535414, -0.040767654, 1.035017910],
], dtype=np.float64)

M_P3_TO_SRGB = np.array([
    [ 1.224745485, -0.224904439, -0.0000000365],
    [-0.042058082,  1.042080996,  0.0000000030],
    [-0.019642260, -0.078654881,  1.098537162],
], dtype=np.float64)

TABLE_DIM = 33
WARP_LOW, WARP_HIGH = -4.0, 4.0


def srgb_encode(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


# --- 采样 -------------------------------------------------------------------------------------
def _corners(table, i0):
    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    return {
        (0, 0, 0): table[z0, y0, x0],
        (1, 0, 0): table[z0, y0, x0 + 1],
        (0, 1, 0): table[z0, y0 + 1, x0],
        (1, 1, 0): table[z0, y0 + 1, x0 + 1],
        (0, 0, 1): table[z0 + 1, y0, x0],
        (1, 0, 1): table[z0 + 1, y0, x0 + 1],
        (0, 1, 1): table[z0 + 1, y0 + 1, x0],
        (1, 1, 1): table[z0 + 1, y0 + 1, x0 + 1],
    }


def _grid_index(p, dim):
    q = np.clip(p, 0.0, 1.0) * (dim - 1)
    i0 = np.minimum(np.floor(q).astype(np.int32), dim - 2)
    return i0, q - i0


def tetra3(table, p):
    """SpectralLUT.sample 的六分支四面体插值。table (D,D,D,C)，p (N,3) ∈ [0,1]。"""
    dim = table.shape[0]
    i0, f = _grid_index(p, dim)
    c = _corners(table, i0)
    c000, c111 = c[(0, 0, 0)], c[(1, 1, 1)]
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]

    xy = (fx >= fy)[:, 0]
    xz = (fx >= fz)[:, 0]
    yz = (fy >= fz)[:, 0]

    out = np.empty_like(c000)
    # 分支顺序与 GLSL 一致，用 where 链写成"后写覆盖"，结果等价于 if/else 树
    out = np.where(
        xy[:, None] & yz[:, None],
        c000 + fx * (c[(1, 0, 0)] - c000) + fy * (c[(1, 1, 0)] - c[(1, 0, 0)])
        + fz * (c111 - c[(1, 1, 0)]),
        out)
    out = np.where(
        xy[:, None] & ~yz[:, None] & xz[:, None],
        c000 + fx * (c[(1, 0, 0)] - c000) + fz * (c[(1, 0, 1)] - c[(1, 0, 0)])
        + fy * (c111 - c[(1, 0, 1)]),
        out)
    out = np.where(
        xy[:, None] & ~yz[:, None] & ~xz[:, None],
        c000 + fz * (c[(0, 0, 1)] - c000) + fx * (c[(1, 0, 1)] - c[(0, 0, 1)])
        + fy * (c111 - c[(1, 0, 1)]),
        out)
    out = np.where(
        ~xy[:, None] & xz[:, None],
        c000 + fy * (c[(0, 1, 0)] - c000) + fx * (c[(1, 1, 0)] - c[(0, 1, 0)])
        + fz * (c111 - c[(1, 1, 0)]),
        out)
    out = np.where(
        ~xy[:, None] & ~xz[:, None] & yz[:, None],
        c000 + fy * (c[(0, 1, 0)] - c000) + fz * (c[(0, 1, 1)] - c[(0, 1, 0)])
        + fx * (c111 - c[(0, 1, 1)]),
        out)
    out = np.where(
        ~xy[:, None] & ~xz[:, None] & ~yz[:, None],
        c000 + fz * (c[(0, 0, 1)] - c000) + fy * (c[(0, 1, 1)] - c[(0, 0, 1)])
        + fx * (c111 - c[(0, 1, 1)]),
        out)
    return out


def linear3(table, p):
    """手写三线性（与 ff_linear3 同算术）。"""
    dim = table.shape[0]
    i0, f = _grid_index(p, dim)
    c = _corners(table, i0)
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    c00 = c[(0, 0, 0)] + fx * (c[(1, 0, 0)] - c[(0, 0, 0)])
    c10 = c[(0, 1, 0)] + fx * (c[(1, 1, 0)] - c[(0, 1, 0)])
    c01 = c[(0, 0, 1)] + fx * (c[(1, 0, 1)] - c[(0, 0, 1)])
    c11 = c[(0, 1, 1)] + fx * (c[(1, 1, 1)] - c[(0, 1, 1)])
    lo = c00 + fy * (c10 - c00)
    hi = c01 + fy * (c11 - c01)
    return lo + fz * (hi - lo)


def _curve1(table, ch, x, lo, hi):
    """2048 点曲线表的线性插值（ff_curve 的逐通道版）。"""
    n = table.shape[0]
    q = np.clip((x - lo) * (n - 1) / (hi - lo), 0.0, n - 1)
    i0 = np.minimum(np.floor(q).astype(np.int32), n - 2)
    f = q - i0
    v0 = table[i0, ch]
    v1 = table[i0 + 1, ch]
    return v0 + f * (v1 - v0)


def _warp_band(bands, ch, band, u, nl):
    n = bands.shape[2]
    q = np.clip((u - WARP_LOW) * (n - 1) / (WARP_HIGH - WARP_LOW), 0.0, n - 1)
    i0 = np.minimum(np.floor(q).astype(np.int32), n - 2)
    f = q - i0
    row = bands[band, ch]
    lo = row[i0]
    hi = row[i0 + 1]
    if not nl:
        return lo + f * (hi - lo)
    prev = row[np.maximum(i0 - 1, 0)]
    foll = row[np.minimum(i0 + 2, n - 1)]
    delta = np.maximum(hi - lo, 0.0)
    low_slope = np.clip(0.5 * (hi - prev), 0.0, 3.0 * delta)
    high_slope = np.clip(0.5 * (foll - lo), 0.0, 3.0 * delta)
    f2 = f * f
    f3 = f2 * f
    return ((2.0 * f3 - 3.0 * f2 + 1.0) * lo + (f3 - 2.0 * f2 + f) * low_slope
            + (-2.0 * f3 + 3.0 * f2) * hi + (f3 - f2) * high_slope)


def _warp(bands, ch, u, dir_total, nl):
    depth = bands.shape[0]
    last = float(bands.shape[0] - 1) / 8.0
    bf = np.clip(dir_total, 0.0, last) * 8.0
    b0 = min(int(bf), depth - 2)
    fb = bf - b0
    w0 = _warp_band(bands, ch, b0, u, nl)
    if fb <= 0.0:
        return w0
    w1 = _warp_band(bands, ch, b0 + 1, u, nl)
    return w0 + fb * (w1 - w0)


def _softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _analytic_density(a, b, x):
    dmin, gamma, toe, toe_w = a
    sh, sh_w = b
    toe_w = max(toe_w, 1e-6)
    sh_w = max(sh_w, 1e-6)
    t = toe_w * _softplus((x - toe) / toe_w)
    s = sh_w * _softplus((x - sh) / sh_w)
    return dmin + gamma * np.clip(t - s, 0.0, sh - toe)


def _scalar(bundle, key, default=0.0):
    v = bundle.get(key)
    if v is None:
        return default
    return float(np.ravel(v)[0])


# --- 链 ---------------------------------------------------------------------------------------
def develop(bundle, rec, dir_total=1.0):
    """scene-linear Rec.2020 (N,3) → 净密度 (N,3)。对应 GLSL ffchain_develop 的非 debug 路径。"""
    dmin = np.asarray(bundle["dmin"], np.float64)
    rng = np.asarray(bundle["rng"], np.float64)
    release = np.asarray(bundle["releaseGamma"], np.float64)
    K = np.asarray(bundle["K"], np.float64)
    scale = _scalar(bundle, "couplerScale", 1.0)
    curve_min = _scalar(bundle, "curveMin", -8.0)
    curve_max = _scalar(bundle, "curveMax", 8.0)
    complement = int(_scalar(bundle, "complement", 0.0))
    film_curve = np.asarray(bundle["filmCurve"], np.float64)
    exposure = np.asarray(bundle["exposure"], np.float64)

    # 1. 色度归一化后采样曝光表（四面体）
    yv = rec @ LUMA_2020
    d = rec @ M_2020_TO_DOMAIN.T
    tr = np.ones_like(d)
    neg = d < 0.0
    tr = np.where(neg, yv[:, None] / np.maximum(yv[:, None] - d, 1e-30), tr)
    s = tr.min(axis=1)
    outside = neg.any(axis=1)
    pv = np.where(yv[:, None] <= 0.0, 0.0,
                  np.where(outside[:, None], yv[:, None] + s[:, None] * (d - yv[:, None]), d))
    w = np.maximum(pv, 0.0)
    dens_scale = np.maximum(w.max(axis=1), 1e-6)
    E = tetra3(exposure, w / dens_scale[:, None])[:, :3] * (dens_scale / 0.18)[:, None]
    logE = np.log(np.maximum(E, 1e-12)) * 0.4342944819

    # 2. DIR：释放 → 跨层抑制 → 中性锚定补偿
    formed0 = np.stack([_curve1(film_curve, c, logE[:, c], curve_min, curve_max)
                        for c in range(3)], axis=1)
    a = np.clip((formed0 - dmin) / rng, 0.0, 1.0)
    ga = np.power(np.maximum(a, 1e-8), release)
    gb = np.power(np.maximum(1.0 - a, 1e-8), release)
    rel = ga / np.maximum(ga + gb, 1e-8)
    inhib = (rel @ K.T) * (scale * dir_total)

    # 2.5 供体层：只释放抑制剂，本身不进画面
    if "donorCurve" in bundle:
        dc = np.asarray(bundle["donorCurve"], np.float64).ravel()
        dg = float(np.ravel(bundle["donorGamma"])[0])
        dr = np.asarray(bundle["donorRelease"], np.float64).ravel()
        e_donor = tetra3(exposure, w / dens_scale[:, None])[:, 3] * (dens_scale / 0.18)
        donor_log = np.log(np.maximum(e_donor, 1e-6)) * 0.4342944819
        dens = _analytic_density(dc[:4], dc[4:6], donor_log)
        drange = max(dc[1] * (dc[4] - dc[2]), 1e-6)
        dact = np.clip((dens - dc[0]) / drange, 0.0, 1.0)
        if dg == 1.0:
            drel = dact
        else:
            ra = np.power(np.maximum(dact, 1e-8), dg)
            rb = np.power(np.maximum(1.0 - dact, 1e-8), dg)
            drel = ra / np.maximum(ra + rb, 1e-8)
        inhib = inhib + dr[None, :] * drel[:, None] * (scale * dir_total)

    u = logE - inhib
    if "warpBands" in bundle:
        bands = np.asarray(bundle["warpBands"], np.float64)
    else:
        bands = np.asarray(bundle["warp"], np.float64)[None]
    nl = (np.any(release != 1.0)
          or ("donorGamma" in bundle and np.any(np.asarray(bundle["donorGamma"]) != 1.0)))
    eff = np.stack([u[:, c] + _warp(bands, c, u[:, c], dir_total, bool(nl))
                    for c in range(3)], axis=1)

    formed = np.stack([_curve1(film_curve, c, eff[:, c], curve_min, curve_max)
                       for c in range(3)], axis=1)
    developed = (dmin + rng - (formed - dmin)) if complement == 1 else formed
    return developed - dmin


def print_net(bundle, net):
    """净密度 (N,3) → 显示 sRGB 编码值 (N,3)。对应 GLSL ffchain_print。"""
    rng = np.asarray(bundle["rng"], np.float64)
    pdmin = np.asarray(bundle["paperDmin"], np.float64)
    prng = np.asarray(bundle["paperRng"], np.float64)
    mask = np.asarray(bundle["masking"], np.float64)
    mid = np.asarray(bundle["midpoints"], np.float64)
    curve_min = _scalar(bundle, "curveMin", -8.0)
    curve_max = _scalar(bundle, "curveMax", 8.0)
    film_out = np.asarray(bundle["filmOut"], np.float64)
    paper_out = np.asarray(bundle["paperOut"], np.float64)
    paper_curve = np.asarray(bundle["paperCurve"], np.float64)

    rel2 = linear3(film_out, net / rng)[:, :3]
    pact = np.stack([
        (_curve1(paper_curve, c, mid[c] + mask[c] * rel2[:, c], curve_min, curve_max) - pdmin[c])
        / prng[c] for c in range(3)], axis=1)
    lin_p3 = linear3(paper_out, pact)[:, :3]
    lin_srgb = np.maximum(lin_p3, 0.0) @ M_P3_TO_SRGB.T
    return srgb_encode(lin_srgb)


def render(bundle, lattice_prophoto, dir_total=1.0):
    """scene-linear ProPhoto (N,3) → 显示 sRGB (N,3)。整条链的对外入口。"""
    rec = np.asarray(lattice_prophoto, np.float64) @ M_PP_TO_2020.T
    return print_net(bundle, develop(bundle, rec, dir_total))


def bake_lut(bundle, size, dir_total=1.0):
    """按 app 的 shaper 格点烘一张 size³ 显示 LUT：输入 shaper t∈[0,1] ↔ scene-linear
    ProPhoto [0,8]，输出显示 sRGB。布局 `[b][g][r][c]`（与 _build_3d_lut 一致）。"""
    from .shaper import shaper_inverse

    t = np.linspace(0.0, 1.0, size, dtype=np.float64)
    x = shaper_inverse(t)
    b, g, r = np.meshgrid(x, x, x, indexing="ij")
    lattice = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    out = render(bundle, lattice, dir_total)
    return np.clip(out, 0.0, 1.0).reshape(size, size, size, 3).astype(np.float32)
