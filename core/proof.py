"""校样（soft proof）：让视口先显示「导出成 X 会长什么样」。

位置
----
校样只存在于**显示端的最后一步**，不改渲染域、不进 LUT key：

    线性 P3 ──M_p3→target──> 目标线性 ──clip──> ──M_target→P3 · M_display──> 显示器线性

之所以在 CPU 侧把回程矩阵和显示器矩阵乘在一起，是为了让 shader 尾段只多**一个**
mat3：点式链的 shader 已经贴着 Intel 驱动的上限（P0 就是在这里被动态下标和三元里
的 mat3 乘法搞崩过两次），多一个 mat3 就多一份非法 ISA 的风险。

相对比色 = clip
---------------
PS 的默认意图「相对比色 + 黑场补偿」在 RGB→RGB 上等价于「在目标域里夹掉越界值」，
所以这里没有色域映射算法，只有 clamp。这也是「校样 = 导出」闭环成立的前提：
导出（`core/colorspace.py`）同样只是 clip。

可感知意图需要 Oklab 色度衰减，与导出的 clip 不一致，会让「预览一个样、文件另一个
样」，因此不进 P1。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from . import colorspace as _cs
from .display_profile import M_P3_TO_XYZ  # noqa: F401  （留作文档锚点，未直接使用）

# 线性 P3 → 线性 sRGB。与 `ui/gl_ffchain.py::M_P3_TO_SRGB` 必须逐位一致：
# 视口用这个数，导出用 `colorspace.MATRICES`，两边不同源就会出现「校样与导出差 1 级」。
M_P3_TO_SRGB = np.array(
    (
        (1.224745485, -0.224904439, -0.0000000365),
        (-0.042058082, 1.042080996, 0.0000000030),
        (-0.019642260, -0.078654881, 1.098537162),
    ),
    dtype=np.float64,
)

# 校样目标。顺序即 UI 顺序；sRGB 在最前（最常见的「给别人看」目标）。
TARGETS = ("sRGB", "Display P3", "Adobe RGB (1998)", "Rec. 2020", "ProPhoto RGB")

# 越界判定的容差。矩阵往返本身有 ~2e-4 的舍入（尾段常量 `M_P3_TO_SRGB` 当年被手工
# 抹过小项），恰好落在 0/1 上的像素会被判成越界 —— 视觉上是一片没有意义的品红麻点。
# 1/512 ≈ 0.002 相当于 8-bit 的半个量化级，这个容差既不误报也漏不掉真正的越界。
GAMUT_EPS = 1.0 / 512.0


@dataclass(frozen=True)
class ProofSettings:
    """校样状态。``on=False`` 时其余字段仍然保留，方便下次开启时恢复。"""

    on: bool = False
    target: str = "sRGB"
    warn: bool = False

    def canonical_target(self) -> str:
        return _cs.resolve(self.target)[0]

    @property
    def label(self) -> str:
        return f"校样 · {self.canonical_target()}"


def resolve_target(name: str) -> str:
    """UI 传进来的目标名归一（别名 / 空值都落到 sRGB），回写设置前先过这一步。"""
    return _cs.resolve(name)[0]


def canonical_target(name: str) -> str:
    """任意用户串 → 规范目标名（不认识的回落到 sRGB）。"""
    return _cs.resolve(name)[0]


def matrix_p3_to_target(target: str) -> np.ndarray:
    """线性 P3 → 线性目标（float64）。

    注意 sRGB 目标**不是单位阵** —— 线性 P3 和线性 sRGB 是两个不同的域，
    ``M_P3_TO_SRGB`` 要老实乘上去。恒等只发生在「校样总变换」上（见
    ``compose_display`` 与 ``scratch/_proof_p1_math.py`` §2）。
    """
    name = _cs.resolve(target)[0]
    m_srgb_to_target = (np.eye(3, dtype=np.float64) if name == "sRGB"
                        else np.asarray(_cs.MATRICES[name], dtype=np.float64))
    return m_srgb_to_target @ M_P3_TO_SRGB


def matrix_target_to_p3(target: str) -> np.ndarray:
    return np.linalg.inv(matrix_p3_to_target(target))


def proof_matrix(settings: ProofSettings) -> Optional[np.ndarray]:
    """GPU 上传用的「线性 P3 → 目标线性」；``on=False`` 返回 None（不接管）。"""
    if not settings.on:
        return None
    return matrix_p3_to_target(settings.canonical_target())


def compose_display(display_matrix: np.ndarray, settings: ProofSettings) -> np.ndarray:
    """把「目标线性 → 线性 P3」折进显示器矩阵，得到尾段要上传的那一个 mat3。

    校样关闭时原样返回，调用方不需要分支。
    """
    if not settings.on:
        return np.asarray(display_matrix, dtype=np.float64)
    back = matrix_target_to_p3(settings.canonical_target())
    return np.asarray(display_matrix, dtype=np.float64) @ back


# --- 纯 numpy 参考实现（验收脚本与 CPU 侧对拍用；产品热路径不调） -----------------------


def to_target(lin_p3: np.ndarray, settings: ProofSettings) -> np.ndarray:
    """线性 P3 → 目标线性（未 clip）。"""
    return np.asarray(lin_p3, np.float64) @ matrix_p3_to_target(settings.canonical_target()).T


def roundtrip(lin_p3: np.ndarray, settings: ProofSettings) -> np.ndarray:
    """线性 P3 → 目标 →(clip)→ 回线性 P3。这就是 shader 里发生的事。"""
    if not settings.on:
        return np.asarray(lin_p3, np.float64)
    m = matrix_p3_to_target(settings.canonical_target())
    t = np.clip(np.asarray(lin_p3, np.float64) @ m.T, 0.0, 1.0)
    return t @ np.linalg.inv(m).T


def out_of_gamut(lin_p3: np.ndarray, target: str) -> np.ndarray:
    """越界掩膜：目标域里任一通道落在 [0,1] 之外（含容差）。"""
    t = np.asarray(lin_p3, np.float64) @ matrix_p3_to_target(target).T
    return ((t < -GAMUT_EPS).any(axis=-1)) | ((t > 1.0 + GAMUT_EPS).any(axis=-1))


def gamut_coverage(lin_p3: np.ndarray, target: str) -> float:
    """越界像素占比，给状态行和验收脚本用。"""
    a = np.asarray(lin_p3)
    if a.size == 0:
        return 0.0
    return float(out_of_gamut(a.reshape(-1, 3), target).mean())


def describe_matrix(target: str) -> str:
    """一行诊断文本（脚本用）。"""
    m = matrix_p3_to_target(target)
    return " / ".join("%.6f" % v for v in m[0])
