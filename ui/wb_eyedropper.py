"""色温吸管：芯片按钮 + 吸管光标。

芯片挂在 `AdobeScrubSlider.add_header_widget()`（与曝光行的 AU 芯片同一个插入位），
所以它的左边缘和 AU 对齐。素材是深灰单色 PNG，深色主题下几乎看不见，因此图标按主题
重着色：常态跟随正文色，激活态用主题色（琥珀金）。

激活态是**界面模式**，不是参数：不进 .sdc、不进撤销栈，切图/关会话都必须退出。
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QCursor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QPushButton, QWidget

from core.path_utils import get_icon_path

ACCENT = "#f59e0b"
ICON_NAME = "eyedropper.png"

# 素材 200x200，尖端在 (35,161)（多阈值 alpha 极值实测稳定在 ±1 px）。
TIP_FRACTION = (0.175, 0.805)

def _mix(fg: str, bg: str, t: float) -> str:
    """把 fg 按 t 混到 bg 上，返回 #rrggbb。

    Qt 样式表的 rgba() alpha 在 QPushButton 背景上不可靠（实测被画成不透明），
    这里自己算混合色，结果可预测。
    """
    def rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    fr, fg_, fb = rgb(fg)
    br, bg_, bb = rgb(bg)
    t = max(0.0, min(1.0, float(t)))
    return "#%02x%02x%02x" % (int(round(fr * t + br * (1 - t))),
                              int(round(fg_ * t + bg_ * (1 - t))),
                              int(round(fb * t + bb * (1 - t))))


_CURSOR_CACHE = {}
_ICON_CACHE = {}


def _normal_color(is_dark: bool) -> str:
    return "#cbd5e1" if is_dark else "#1e293b"


def _tinted(path: str, color: str, size: int) -> QPixmap:
    """把单色 PNG 渲染成指定颜色与尺寸（用 alpha 当蒙版）。"""
    key = (path, color, size)
    hit = _ICON_CACHE.get(key)
    if hit is not None:
        return hit
    src = QPixmap(path)
    if src.isNull():
        return QPixmap()
    scaled = src.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
    out = QPixmap(scaled.size())
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.drawPixmap(0, 0, scaled)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(out.rect(), QColor(color))
    p.end()
    _ICON_CACHE[key] = out
    return out


def eyedropper_cursor(size: int = 24) -> QCursor:
    """吸管光标：热点落在吸管左下尖端，而不是图像中心。"""
    path = get_icon_path(ICON_NAME)
    key = (path, size)
    hit = _CURSOR_CACHE.get(key)
    if hit is not None:
        return hit
    # 光标不受主题色影响：白边 + 深色主体，任何底色上都看得见。
    pix = _tinted(path, "#f8fafc", size + 2)
    if pix.isNull():
        return QCursor(Qt.CursorShape.CrossCursor)
    body = _tinted(path, "#1f2937", size)
    canvas = QPixmap(pix.size())
    canvas.fill(Qt.GlobalColor.transparent)
    p = QPainter(canvas)
    p.drawPixmap(1, 1, body)
    p.end()
    p2 = QPainter(canvas)
    p2.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
    p2.drawPixmap(0, 0, pix)
    p2.end()
    hx = int(round(1 + TIP_FRACTION[0] * size))
    hy = int(round(1 + TIP_FRACTION[1] * size))
    hx = max(0, min(canvas.width() - 1, hx))
    hy = max(0, min(canvas.height() - 1, hy))
    cur = QCursor(canvas, hx, hy)
    _CURSOR_CACHE[key] = cur
    return cur


class EyedropperChip(QPushButton):
    """色温行标题后面的吸管按钮：无底无框，只有图标本身。

    hover 提亮，激活态整枚图标转主题色。按钮比 AU 芯片大，靠 `SlotContainer`
    与 AU 共用同一个 44 px 宽的水平槽位，两者中心对齐。
    """

    SIZE = QSize(24, 24)
    ICON_PX = 22

    def __init__(self, parent=None, tooltip: str = ""):
        super().__init__(parent)
        self.setFixedSize(self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._active = False
        self._hover = False
        self.setToolTip(tooltip)
        self._theme_hooked = False
        self.apply_theme()
        try:
            from ui.theme_manager import theme_manager
            theme_manager.themeChanged.connect(self._on_theme_changed)
            self._theme_hooked = True
        except Exception:
            pass

    # -- state ---------------------------------------------------------------
    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool):
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        self.apply_theme()

    def _on_theme_changed(self, *args):
        self.apply_theme()

    # -- paint ---------------------------------------------------------------
    def enterEvent(self, event):
        self._hover = True
        self.apply_theme()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.apply_theme()
        super().leaveEvent(event)

    def apply_theme(self):
        try:
            from ui.theme_manager import theme_manager
            is_dark = theme_manager.is_dark()
        except Exception:
            is_dark = True
        # 永远不画底：状态只靠**整枚图标**的颜色表达，别在参数面板里加方块，
        # 也别让图标尺寸变化去撑高行距。
        if self._active:
            icon_color = ACCENT                 # 琥珀 500：已进入取色模式
        elif self._hover:
            icon_color = "#fcd34d"              # 琥珀 300：悬停可点，比常态明显变色
        else:
            icon_color = _normal_color(is_dark)
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; }")
        pix = _tinted(get_icon_path(ICON_NAME), icon_color, self.ICON_PX)
        self.setIcon(QIcon(pix) if not pix.isNull() else QIcon())
        self.setIconSize(QSize(self.ICON_PX, self.ICON_PX))


class ChipSlot(QWidget):
    """把一颗芯片放进固定宽度的透明槽位并水平居中。

    子控件不能画出父控件边界（实测会被裁掉），所以"图标比 AU 大、又要和 AU 同心"
    只能靠两边共用同一个槽宽：AU（30 宽）与吸管（44 宽）都以槽中心对齐。
    """

    WIDTH = 44

    def __init__(self, child: QWidget, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.setFixedHeight(max(1, child.height()))
        self.setStyleSheet("background: transparent; border: none;")
        child.setParent(self)
        child.move(max(0, (self.WIDTH - child.width()) // 2), 0)
        child.show()
        self._child = child

    def child(self) -> QWidget:
        return self._child
