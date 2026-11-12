"""显示端色彩管理：读这台显示器的 ICC，算出「线性 P3 → 显示器」的矩阵与编码曲线。

P0 的全部数值都出自这里。渲染域完全不碰 —— 链尾仍是线性 P3，只有最后上屏那一步
换成 ``to_display()``。

为什么出口是 P3 而不是 sRGB
--------------------------
``ffchain_print`` 里 ``paperOut`` 表的通道本来就是线性 P3（染料光谱决定的，不是选择）。
sRGB 只是历史上一跳矩阵压出来的。所以显示补偿的起点天然是 P3。

数值约定
--------
* ICC 的 rXYZ/gXYZ/bXYZ 是 **D50 PCS** 相对量，必须过 Bradford 转 D65 才能和 P3(D65) 相乘。
  漏掉这一步的典型症状是整体偏黄绿 2–4 ΔE，且在 sRGB profile 上看不出来（sRGB 也有同样偏移）。
* ``matrix`` = ``inv(M_display_to_XYZ_D65) @ M_P3_to_XYZ_D65``。
  当 profile 恰是 sRGB 时它应当等于 ``gl_ffchain.M_P3_TO_SRGB``（差 <1e-6）—— 这是最强回归判据。
* ``trc`` 是 **编码** 方向（线性 → 设备值），和 ICC 标签语义一致。

降级策略
--------
读不到 profile / 是 LUT 型 / 解析失败 → ``identity()``：矩阵为单位阵、TRC 为 sRGB。
此时 shader 尾段退化成今天的行为，出图逐字节不变。宁可不管也别管错。
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --- 常数 -----------------------------------------------------------------------------

# 线性 Display P3 → XYZ(D65)。与 colour-science 的 RGB_COLOURSPACE 同值。
M_P3_TO_XYZ = np.array(
    [
        (0.4865709486482162, 0.26566769316909306, 0.1982172852343625),
        (0.2289745640697488, 0.6917385218365064, 0.0792869140937450),
        (0.0000000000000000, 0.04511338185890264, 1.0439443689009760),
    ],
    dtype=np.float64,
)

# Bradford D50 ↔ D65（ICC 规范附录推荐的色适应变换）
_BRADFORD_D50_TO_D65 = np.array(
    [
        (0.955473421488075, -0.02309845494876471, 0.06325924320057072),
        (-0.0283697093338637, 1.0099953980813041, 0.021041441191917323),
        (0.012314014864481998, -0.020507649298898964, 1.330365926242124),
    ],
    dtype=np.float64,
)

ICC_HEADER_SIZE = 128

# TRC 类型标记。给 shader 的分支用：0=sRGB 分段，1=纯 gamma，2=查表
TRC_SRGB = 0
TRC_GAMMA = 1
TRC_TABLE = 2


@dataclass(frozen=True)
class DisplayProfile:
    """一台显示器的显示变换。"""

    path: str
    # 线性 P3 → 显示器线性 RGB
    matrix: np.ndarray
    # (kind, param)：kind=TRC_SRGB 时 param 忽略；TRC_GAMMA 时 param 是 gamma；
    # TRC_TABLE 时 param 是 1024 点查表（线性 → 设备值，0..1）
    trc: Tuple[int, object]
    white_xyz: Tuple[float, float, float]
    # True = 什么都不做（矩阵单位阵 + sRGB 编码），出图与未接入前逐字节一致
    identity: bool = False
    note: str = ""

    @property
    def label(self) -> str:
        if self.identity:
            return "sRGB（直通）" if not self.note else f"sRGB（{self.note}）"
        name = os.path.basename(self.path) if self.path else "未知"
        return f"{name} · {_trc_label(self.trc)}"


def _trc_label(trc: Tuple[int, object]) -> str:
    kind, param = trc
    if kind == TRC_SRGB:
        return "sRGB 曲线"
    if kind == TRC_GAMMA:
        return f"gamma {float(param):.3f}"
    return "查表曲线"


def identity(note: str = "") -> DisplayProfile:
    """跑今天的老路：P3 → 线性 sRGB → sRGB 编码。

    ``matrix`` 仍按「P3 → 显示器线性」的语义填，此时就是 M_P3_TO_SRGB；``identity``
    标志让 GPU 端直接跳过上传、复用原常量，所以这条路径上的像素一个 bit 都不变。
    """
    return DisplayProfile(
        path="",
        matrix=_fallback_srgb_matrix(),
        trc=(TRC_SRGB, None),
        white_xyz=(0.95047, 1.0, 1.08883),
        identity=True,
        note=note,
    )


def _fallback_srgb_matrix() -> np.ndarray:
    """P3 → 线性 sRGB，与 ui/gl_ffchain.py 的 M_P3_TO_SRGB 同值。"""
    return np.array(
        (
            (1.224745485, -0.224904439, -0.0000000365),
            (-0.042058082, 1.042080996, 0.0000000030),
            (-0.019642260, -0.078654881, 1.098537162),
        ),
        dtype=np.float64,
    )


# --- ICC 解析 -------------------------------------------------------------------------


def _s15f16(raw: bytes) -> float:
    return struct.unpack(">i", raw)[0] / 65536.0


def _read_tag_table(blob: bytes) -> dict:
    if len(blob) < ICC_HEADER_SIZE or blob[36:40] != b"acsp":
        raise ValueError("not an ICC profile")
    count = struct.unpack(">I", blob[128:132])[0]
    if count <= 0 or count > 4096:
        raise ValueError(f"implausible tag count {count}")
    tags = {}
    for i in range(count):
        base = 132 + i * 12
        if base + 12 > len(blob):
            break
        sig = blob[base : base + 4]
        off, size = struct.unpack(">II", blob[base + 4 : base + 12])
        if off + size <= len(blob) and size > 0:
            tags[sig] = blob[off : off + size]
    return tags


def _parse_xyz(tag: bytes) -> np.ndarray:
    if tag[:4] not in (b"XYZ ", b"XYZ\0"):
        raise ValueError("not XYZType")
    return np.array([_s15f16(tag[8 + 4 * i : 12 + 4 * i]) for i in range(3)], dtype=np.float64)


def _parse_trc(tag: bytes) -> Tuple[int, object]:
    """返回 (kind, param)，方向是「线性 → 设备」。

    ICC 里 TRC 标签的语义是 **设备值 → 线性**（profile 是设备→PCS 方向），上屏要的是
    反过来。这个方向搞错的表现不是"稍微偏"，而是整幅发灰/发黑：线性值被当成设备值再
    套一次解码曲线。所以表格型必须数值反演，不能直接拿来用。
    """
    sig = tag[:4]
    if sig == b"curv":
        count = struct.unpack(">I", tag[8:12])[0]
        if count == 0:
            return (TRC_GAMMA, 1.0)
        if count == 1:
            # u8Fixed8：lin = dev^g  ⇒  dev = lin^(1/g)
            g = struct.unpack(">H", tag[12:14])[0] / 256.0
            return (TRC_GAMMA, g if g > 0 else 1.0)
        dev_to_lin = np.frombuffer(tag[12 : 12 + 2 * count], dtype=">u2").astype(np.float64) / 65535.0
        return (TRC_TABLE, _invert_curve(dev_to_lin))
    if sig == b"para":
        ftype = struct.unpack(">H", tag[8:10])[0]
        n_params = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(ftype)
        if n_params is None:
            raise ValueError(f"unsupported parametric curve type {ftype}")
        p = [_s15f16(tag[12 + 4 * i : 16 + 4 * i]) for i in range(n_params)]
        if ftype == 0 and abs(p[0] - 1.0) < 1e-6:
            return (TRC_GAMMA, 1.0)
        if ftype == 0:
            return (TRC_GAMMA, p[0])
        if ftype == 3 and _is_srgb_params(p):
            return (TRC_SRGB, None)
        dev = _invert_parametric(ftype, p)
        return (TRC_TABLE, dev)
    raise ValueError(f"unsupported TRC type {sig!r}")


def _is_srgb_params(p: Sequence[float]) -> bool:
    g, a, b, c, d = p
    return (
        abs(g - 2.4) < 0.05
        and abs(a - (1.0 / 1.055)) < 0.005
        and abs(b - 0.055 / 1.055) < 0.005
        and abs(c - (1.0 / 12.92)) < 0.002
        and abs(d - 0.04045) < 0.002
    )


def _invert_curve(dev_to_lin: np.ndarray, n: int = 1024) -> np.ndarray:
    """设备→线性 的采样表 → 线性→设备 的 1024 点表。

    先单调化：ICC 表在 16-bit 量化下会有相邻相等的点，不平滑掉会让 `np.interp`
    在平台上取到平台起点，暗部出现台阶。
    """
    if dev_to_lin.size < 2:
        return np.linspace(0.0, 1.0, n)
    lin = np.maximum.accumulate(np.clip(dev_to_lin, 0.0, 1.0))
    dev = np.linspace(0.0, 1.0, dev_to_lin.size)
    return np.interp(np.linspace(0.0, 1.0, n), lin, dev)


def _invert_parametric(ftype: int, p: Sequence[float], n: int = 1024) -> np.ndarray:
    """ICC parametricCurveType 的解析反演，出口是 线性→设备。

    定义（ICC.1:2010 §10.18，X=设备值，Y=线性值）::

        type 0: Y = X^g
        type 1: Y = (aX+b)^g          X >= -b/a
        type 2: Y = (aX+b)^g + c
        type 3: Y = (aX+b)^g  X>=d ;  cX  X<d
        type 4: Y = (aX+b)^g+e X>=d ; cX+f  X<d
    """
    g = p[0]
    inv_g = 1.0 / g if g else 1.0
    y = np.linspace(0.0, 1.0, n)
    if ftype == 0:
        return np.power(y, inv_g)
    if ftype == 1:
        a, b = p[1], p[2]
        return np.clip((np.power(y, inv_g) - b) / a if a else y, 0.0, 1.0)
    if ftype == 2:
        a, b, c = p[1], p[2], p[3]
        return np.clip((np.power(np.maximum(y - c, 0.0), inv_g) - b) / a if a else y, 0.0, 1.0)
    if ftype == 3:
        a, b, c, d = p[1], p[2], p[3], p[4]
        knee = (a * d + b) ** g if a * d + b > 0 else 0.0
        upper = np.power(np.maximum(y, 0.0), inv_g) - b
        upper = upper / a if a else y
        lower = y / c if c else y
        return np.clip(np.where(y >= knee, upper, lower), 0.0, 1.0)
    if ftype == 4:
        a, b, c, d, e, f = p[1], p[2], p[3], p[4], p[5], p[6]
        knee = (a * d + b) ** g + e if a * d + b > 0 else 0.0
        upper = (np.power(np.maximum(y - e, 0.0), inv_g) - b)
        upper = upper / a if a else y
        lower = (y - f) / c if c else y
        return np.clip(np.where(y >= knee, upper, lower), 0.0, 1.0)
    raise ValueError(f"unsupported para type {ftype}")


def compute_matrix(r_xyz: np.ndarray, g_xyz: np.ndarray, b_xyz: np.ndarray) -> np.ndarray:
    """ICC 的三个 XYZ 列向量 → 「线性 P3 → 显示器线性」矩阵。"""
    m_rgb2xyz_d50 = np.stack([r_xyz, g_xyz, b_xyz], axis=1)
    try:
        if abs(np.linalg.det(m_rgb2xyz_d50)) < 1e-9:
            raise np.linalg.LinAlgError("singular primaries")
        m_rgb2xyz_d65 = _BRADFORD_D50_TO_D65 @ m_rgb2xyz_d50
        return np.linalg.inv(m_rgb2xyz_d65) @ M_P3_TO_XYZ
    except np.linalg.LinAlgError:
        return _fallback_srgb_matrix()


def parse_icc(blob: bytes, path: str = "") -> DisplayProfile:
    """matrix/TRC 型 ICC → DisplayProfile。其他类型抛 ValueError（调用方降级）。"""
    tags = _read_tag_table(blob)
    missing = [t for t in (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC") if t not in tags]
    if missing:
        raise ValueError(f"missing tags {[m.decode() for m in missing]} (LUT-type profile?)")

    r_xyz = _parse_xyz(tags[b"rXYZ"])
    g_xyz = _parse_xyz(tags[b"gXYZ"])
    b_xyz = _parse_xyz(tags[b"bXYZ"])

    trc_r = _parse_trc(tags[b"rTRC"])
    # 灰阶/独立通道的显示器：三通道 TRC 应一致；不一致就用 R 通道（少见，记日志）
    for name in (b"gTRC", b"bTRC"):
        if name in tags:
            try:
                if not _trc_equal(trc_r, _parse_trc(tags[name])):
                    logger.info("display profile %s: per-channel TRC differ, using red", path)
            except Exception:  # noqa: BLE001
                pass

    wtpt = (0.9642, 1.0, 0.8249)
    if b"wtpt" in tags:
        try:
            w = _parse_xyz(tags[b"wtpt"])
            wtpt = (float(w[0]), float(w[1]), float(w[2]))
        except Exception:  # noqa: BLE001
            pass

    matrix = compute_matrix(r_xyz, g_xyz, b_xyz)

    # 认定「等价 sRGB」：矩阵贴近 M_P3_TO_SRGB 且曲线贴近 sRGB —— 走恒等路径，
    # 避免为了一个 1e-4 的差别去冒险改像素。判曲线用数值比较而不是看标签类型：
    # 系统自带的 sRGB Color Space Profile.icm 就是 1024 点 curv 表，不是 para。
    if _trc_is_srgb(trc_r) and np.abs(matrix - _fallback_srgb_matrix()).max() < 1e-3:
        return DisplayProfile(
            path=path,
            matrix=_fallback_srgb_matrix(),
            trc=(TRC_SRGB, None),
            white_xyz=wtpt,
            identity=True,
            note="系统 sRGB profile",
        )

    return DisplayProfile(
        path=path,
        matrix=matrix,
        trc=trc_r,
        white_xyz=wtpt,
        identity=False,
    )


def _trc_is_srgb(trc: Tuple[int, object], tol: float = 1.5 / 255.0) -> bool:
    """曲线是否等价于 sRGB 分段（按 8-bit 判，因为上屏就是 8-bit）。"""
    if trc[0] == TRC_SRGB:
        return True
    lin = np.linspace(0.0, 1.0, 256)
    ref = np.where(
        lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055
    )
    return float(np.abs(_trc_encode(trc, lin) - ref).max()) < tol


def _trc_encode(trc: Tuple[int, object], lin: np.ndarray) -> np.ndarray:
    """线性 → 设备值（0..1）。"""
    kind, param = trc
    v = np.clip(lin, 0.0, 1.0)
    if kind == TRC_SRGB:
        return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    if kind == TRC_GAMMA:
        g = float(param)
        return np.power(v, 1.0 / g) if g else v
    table = np.asarray(param, dtype=np.float64)
    if table.size < 2:
        return v
    idx = v * (table.size - 1)
    return np.interp(idx, np.arange(table.size), table)


def _trc_equal(a: Tuple[int, object], b: Tuple[int, object], tol: float = 1e-3) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == TRC_SRGB:
        return True
    if a[0] == TRC_GAMMA:
        return abs(float(a[1]) - float(b[1])) < 1e-3
    ta, tb = np.asarray(a[1]), np.asarray(b[1])
    return ta.shape == tb.shape and float(np.abs(ta - tb).max()) < tol


# --- 取当前显示器的 profile ------------------------------------------------------------


_GDI32 = None


def _gdi32():
    """缓存 gdi32 句柄（理由同 `_user32`：反复 WinDLL 会在退出时崩）。"""
    global _GDI32
    if _GDI32 is not None:
        return _GDI32
    g = ctypes.WinDLL("gdi32", use_last_error=True)
    g.GetICMProfileW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_wchar_p,
    ]
    g.GetICMProfileW.restype = ctypes.c_int
    g.CreateDCW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_void_p,
    ]
    g.CreateDCW.restype = ctypes.c_void_p
    g.DeleteDC.argtypes = [ctypes.c_void_p]
    g.DeleteDC.restype = ctypes.c_int
    _GDI32 = g
    return g


def _windows_profile_path(device_name: Optional[str] = None) -> Optional[str]:
    """用 GetICMProfileW 取 profile。device_name 为空时取主显示器（GetDC(NULL)）。

    3 秒轮询会反复调这里，所以句柄全部走缓存版 `_gdi32()` / `_user32()`。
    """
    if sys.platform != "win32":
        return None
    try:
        gdi32 = _gdi32()
        if device_name:
            hdc = gdi32.CreateDCW(device_name, None, None, None)
            own = True
        else:
            user32 = _user32()
            hdc = user32.GetDC(None)
            own = False
        if not hdc:
            return None
        try:
            size = ctypes.c_uint32(1024)
            buf = ctypes.create_unicode_buffer(1024)
            if gdi32.GetICMProfileW(hdc, ctypes.byref(size), buf):
                return buf.value or None
            return None
        finally:
            if own:
                gdi32.DeleteDC(hdc)
            else:
                user32.ReleaseDC(None, hdc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetICMProfile failed: %s", exc)
        return None


def _system_srgb_path() -> Optional[str]:
    root = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    for name in ("sRGB Color Space Profile.icm", "sRGB Color Space Profile.icc"):
        p = os.path.join(root, "System32", "spool", "drivers", "color", name)
        if os.path.exists(p):
            return p
    return None


def load(path: Optional[str]) -> DisplayProfile:
    if not path:
        return identity("no profile")
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError as exc:
        return identity(f"read failed: {exc.__class__.__name__}")
    try:
        return parse_icc(blob, path=path)
    except Exception as exc:  # noqa: BLE001 - 任何解析问题都降级，绝不改像素
        logger.info("display profile %s unusable: %s", path, exc)
        return identity(f"{exc.__class__.__name__}")


def current(device_name: Optional[str] = None) -> DisplayProfile:
    """读当前显示器。任何失败都返回 identity，调用方无需处理异常。"""
    if sys.platform != "win32":
        return identity("non-Windows")
    path = _windows_profile_path(device_name)
    if not path:
        path = _system_srgb_path()
        if path:
            return load(path)
        return identity("no profile file")
    return load(path)


# --- Windows ACM（自动色彩管理）状态 ---------------------------------------------------
#
# 这是 P0 唯一的硬阻塞：ACM 开着时系统已经把「未标记窗口」当 sRGB 转到显示器 profile，
# 我们再自己转一次就是双重转换，颜色反而错。所以 ACM 开 → 默认不自管。


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _PATH_SOURCE(ctypes.Structure):
    # 20 字节：LUID(8) + id(4) + modeInfoIdx(4) + statusFlags(4)。
    # 这里的最深坑：modeInfoIdx 内部的 16+16 是它**自己的联合体**（cloneGroupId /
    # sourceModeInfoIdx），不是 statusFlags —— 我把两者合成一个 32 位位域后 SOURCE 变成
    # 16 字节，targetInfo 整体前移 4 字节，QueryDisplayConfig 仍返回 0（长度对得上），
    # 但 DisplayConfigGetDeviceInfo 拿到的 adapterId/id 是错位的 → 恒回
    # ERROR_INVALID_PARAMETER(87) → ACM 永远探测不到，且看起来像"系统不支持"。
    _fields_ = [
        ("adapterId", _LUID),
        ("id", ctypes.c_uint32),
        ("modeInfoIdx", ctypes.c_uint32),
        ("statusFlags", ctypes.c_uint32),
    ]


class _RATIONAL(ctypes.Structure):
    _fields_ = [("n", ctypes.c_uint32), ("d", ctypes.c_uint32)]


class _PATH_TARGET(ctypes.Structure):
    # 48 字节。**末尾的 targetAvailable / statusFlags 不能省**：少了它们 TARGET 只有
    # 40 字节，PATH_INFO 变 64，而真实步长是 72 —— QueryDisplayConfig 按 72 写入、
    # 我们按 64 读，越界 8 字节/条，堆被踩（错的不是读数，是后续随机段错误）。
    _fields_ = [
        ("adapterId", _LUID),
        ("id", ctypes.c_uint32),
        ("modeInfoIdx", ctypes.c_uint32),
        ("outputTechnology", ctypes.c_uint32),
        ("rotation", ctypes.c_uint32),
        ("scaling", ctypes.c_uint32),
        ("refreshRate", _RATIONAL),
        ("scanLineOrdering", ctypes.c_uint32),
        ("targetAvailable", ctypes.c_int32),
        ("statusFlags", ctypes.c_uint32),
    ]


class _PATH_INFO(ctypes.Structure):
    # 20 + 48 + 4 = 72，等于真实的 sizeof(DISPLAYCONFIG_PATH_INFO)。
    # 这个尺寸是硬约束：小了 → 堆被踩（随机段错误），大了 → ERROR_INSUFFICIENT_BUFFER。
    _fields_ = [
        ("sourceInfo", _PATH_SOURCE),
        ("targetInfo", _PATH_TARGET),
        ("flags", ctypes.c_uint32),
    ]


class _MODE_INFO(ctypes.Structure):
    # DISPLAYCONFIG_MODE_INFO 是 **80** 字节：infoType(4) + id(4) + adapterId(8) +
    # 联合体(64)（targetMode / sourceMode / desktopImageInfo 里最大的是
    # DISPLAYCONFIG_TARGET_MODE，其 videoSignalInfo 本身就 64）。
    # 按 64 字节分配 → Windows 每次写越界 16 字节 → 堆被踩，**第一次调用正常、
    # 之后随便一次 ctypes 调用随机段错误**（表现像"随机崩溃"，其实很确定）。
    _fields_ = [("pad", ctypes.c_byte * 80)]


class _DEVICE_INFO_HEADER(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("adapterId", _LUID),
        ("id", ctypes.c_uint32),
    ]


class _ADVANCED_COLOR_INFO(ctypes.Structure):
    _fields_ = [
        ("header", _DEVICE_INFO_HEADER),
        ("value", ctypes.c_uint32),
        ("colorEncoding", ctypes.c_uint32),
        ("bitsPerColorChannel", ctypes.c_uint32),
    ]


_QDC_ONLY_ACTIVE_PATHS = 0x2
_GET_ADVANCED_COLOR_INFO = 9
_USER32 = None

# 尺寸自检：这两个数错了就是踩堆（随机段错误）或 INSUFFICIENT_BUFFER，且都是
# "看起来系统不支持 ACM"，极难从现象倒推。宁可导入时大声失败。
#   DISPLAYCONFIG_PATH_INFO = 72、DISPLAYCONFIG_MODE_INFO = 80
_EXPECTED_SIZES = {
    "_PATH_INFO": (_PATH_INFO, 72),
    "_MODE_INFO": (_MODE_INFO, 80),
    "_ADVANCED_COLOR_INFO": (_ADVANCED_COLOR_INFO, 32),
}


def _check_struct_sizes() -> None:
    bad = [
        f"{name}={ctypes.sizeof(obj)}（应为 {want}）"
        for name, (obj, want) in _EXPECTED_SIZES.items()
        if ctypes.sizeof(obj) != want
    ]
    if bad:
        logger.error("display config structs mis-sized: %s", ", ".join(bad))


def _user32():
    """缓存 user32 句柄并只设一次 argtypes。

    每次调用都新建 `ctypes.WinDLL` 会在解释器退出时让 LoadLibrary/FreeLibrary 交错，
    表现是"程序跑完、退出码 139"，而且只在别的 DLL 也参与时复现（查了很久）。
    """
    global _USER32
    if _USER32 is not None:
        return _USER32
    _check_struct_sizes()
    u = ctypes.WinDLL("user32", use_last_error=True)
    # 必须显式给 argtypes：不给的话 ctypes 把数组参数按 int 传，64 位下指针被截断。
    u.GetDisplayConfigBufferSizes.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    u.GetDisplayConfigBufferSizes.restype = ctypes.c_long
    u.QueryDisplayConfig.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(_PATH_INFO),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(_MODE_INFO),
        ctypes.c_void_p,
    ]
    u.QueryDisplayConfig.restype = ctypes.c_long
    u.DisplayConfigGetDeviceInfo.argtypes = [ctypes.POINTER(_ADVANCED_COLOR_INFO)]
    u.DisplayConfigGetDeviceInfo.restype = ctypes.c_long
    u.GetDC.argtypes = [ctypes.c_void_p]
    u.GetDC.restype = ctypes.c_void_p
    u.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    u.ReleaseDC.restype = ctypes.c_int
    _USER32 = u
    return u


def acm_state() -> Optional[dict]:
    """返回 {'supported','enabled','bits'}；拿不到返回 None（调用方按「不支持」处理）。"""
    if sys.platform != "win32":
        return None
    try:
        user32 = _user32()

        n_path = ctypes.c_uint32(0)
        n_mode = ctypes.c_uint32(0)
        # 这几个 API 返回 0 == ERROR_SUCCESS。写成 `if not rc` 会把成功当失败，
        # 而失败分支是"不自管"——于是 P0 静默永不生效（就是被这一条坑过一次）。
        if user32.GetDisplayConfigBufferSizes(
            _QDC_ONLY_ACTIVE_PATHS, ctypes.byref(n_path), ctypes.byref(n_mode)
        ) != 0:
            return None
        if not (0 < n_path.value <= 64 and 0 <= n_mode.value <= 1024):
            return None
        paths = (_PATH_INFO * max(1, n_path.value))()
        modes = (_MODE_INFO * max(1, n_mode.value))()
        if user32.QueryDisplayConfig(
            _QDC_ONLY_ACTIVE_PATHS,
            ctypes.byref(n_path),
            paths,
            ctypes.byref(n_mode),
            modes,
            None,
        ) != 0:
            return None
        for p in paths[: n_path.value]:
            info = _ADVANCED_COLOR_INFO()
            info.header.type = _GET_ADVANCED_COLOR_INFO
            info.header.size = ctypes.sizeof(info)
            info.header.adapterId = p.targetInfo.adapterId
            info.header.id = p.targetInfo.id
            if user32.DisplayConfigGetDeviceInfo(ctypes.byref(info)) != 0:
                continue
            return {
                "supported": bool(info.value & 0x1),
                "enabled": bool(info.value & 0x2),
                "bits": int(info.bitsPerColorChannel),
            }
    except Exception as exc:  # noqa: BLE001
        logger.debug("ACM query failed: %s", exc)
    return None


def managed_default() -> Tuple[bool, str]:
    """默认要不要自己管显示端。返回 (managed, 理由)。

    ACM 开着 → 系统已在做，我们退出（否则双重转换）。
    ACM 关着 → 自己管。
    探测失败 → 退出（保守，等于今天的行为，宁可不管也别管错）。
    """
    st = acm_state()
    if st is None:
        return False, "无法读取系统色彩管理状态"
    if st["enabled"]:
        return False, "系统已开启自动色彩管理（ACM），由系统负责转换"
    return True, ""


# --- 与 lcms 的对拍辅助（验收脚本用，产品不调） ----------------------------------------


def encode_linear(profile: DisplayProfile, lin: np.ndarray) -> np.ndarray:
    """线性 → 设备值（0..1）。GPU 查表的烘焙入口，也是纯 numpy 参考。"""
    return _trc_encode(profile.trc, lin)


def lut_data(profile: DisplayProfile, n: int = 1024) -> np.ndarray:
    """烘成 (1, n, 4) 的 RGBA 表，三通道各填一条曲线（显示器通常三通道一致）。"""
    lin = np.linspace(0.0, 1.0, n)
    enc = np.clip(_trc_encode(profile.trc, lin), 0.0, 1.0).astype(np.float32)
    data = np.zeros((1, n, 4), np.float32)
    data[0, :, 0] = enc
    data[0, :, 1] = enc
    data[0, :, 2] = enc
    data[0, :, 3] = 1.0
    return data


def to_srgb8(profile: DisplayProfile, lin_p3: np.ndarray) -> np.ndarray:
    """把线性 P3 数组送到该 profile 对应的 8-bit 设备值。纯 numpy 参考实现。"""
    lin_disp = np.maximum(np.asarray(lin_p3, dtype=np.float64), 0.0) @ profile.matrix.T
    enc = _trc_encode(profile.trc, lin_disp)
    return np.clip(enc * 255.0 + 0.5, 0, 255).astype(np.uint8)
