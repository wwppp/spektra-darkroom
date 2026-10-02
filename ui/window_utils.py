"""window_utils.py - Window layout, DWM titlebar, and shared UI styling utilities
"""

import os
import sys
import ctypes
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QIcon, QPixmap, QPainter
from path_utils import get_icon_path

_ICON_CACHE = {}


def get_icon(name: str, target_size=None, scale: float = 1.0) -> QIcon:
    """Universal normalized icon loader for Darkroom UI.
    Automatically trims transparent padding and rescales icons onto a standardized 
    200x200 canvas with balanced visual weight, ensuring every menu item icon renders
    at precisely identical, crisp visual size.
    """
    if not name:
        return QIcon()
    path = get_icon_path(name) if not os.path.isabs(name) else name
    if not os.path.exists(path):
        return QIcon()

    cache_key = (path, target_size, scale)
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    pix = QPixmap(path)
    if pix.isNull():
        icon = QIcon(path)
        _ICON_CACHE[cache_key] = icon
        return icon

    img = pix.toImage()
    w, h = img.width(), img.height()
    if w <= 0 or h <= 0:
        icon = QIcon(pix)
        _ICON_CACHE[cache_key] = icon
        return icon

    # Extract alpha channel to detect visual bounding box
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape((h, w, 4))
    alpha = arr[:, :, 3]
    y_idx, x_idx = np.where(alpha > 15)
    if len(x_idx) == 0:
        icon = QIcon(pix)
        _ICON_CACHE[cache_key] = icon
        return icon

    min_x, max_x = int(x_idx.min()), int(x_idx.max())
    min_y, max_y = int(y_idx.min()), int(y_idx.max())
    content_w = max_x - min_x + 1
    content_h = max_y - min_y + 1

    target_canvas = 200
    base_visual = 138.0

    # Visual weight balance
    aspect = max(content_w / max(1, content_h), content_h / max(1, content_w))
    visual_target = base_visual * (1.0 + min(0.12, (aspect - 1.0) * 0.08))

    if target_size is not None:
        tw = target_size if isinstance(target_size, (int, float)) else target_size[0]
        visual_target = visual_target * (tw / 16.0)
    if scale != 1.0:
        visual_target *= scale

    calc_scale = visual_target / max(content_w, content_h)
    new_w = max(1, int(content_w * calc_scale))
    new_h = max(1, int(content_h * calc_scale))

    cropped = pix.copy(min_x, min_y, content_w, content_h)
    scaled_pix = cropped.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    canvas = QPixmap(target_canvas, target_canvas)
    canvas.fill(Qt.GlobalColor.transparent)
    p = QPainter(canvas)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    ox = (target_canvas - scaled_pix.width()) // 2
    oy = (target_canvas - scaled_pix.height()) // 2
    p.drawPixmap(ox, oy, scaled_pix)
    p.end()

    icon = QIcon(canvas)
    _ICON_CACHE[cache_key] = icon
    return icon



def apply_dark_titlebar(widget: QWidget):
    """Apply native Windows 10/11 immersive dark mode and custom dark caption bar."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
        set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute

        # 1. DWMWA_USE_IMMERSIVE_DARK_MODE: 20 (Win11 / Win10 20H1+), fallback 19 (Win10 1809+)
        val = ctypes.c_int(1)
        res = set_window_attribute(hwnd, 20, ctypes.byref(val), ctypes.sizeof(val))
        if res != 0:
            set_window_attribute(hwnd, 19, ctypes.byref(val), ctypes.sizeof(val))

        # 2. Windows 11 Build 22000+: Explicitly enforce dark titlebar background and text colors
        # DWMWA_CAPTION_COLOR = 35 (BGR: 0x001A1514 -> #14151a)
        # DWMWA_TEXT_COLOR = 36 (BGR: 0x00F0E8E2 -> #e2e8f0)
        try:
            caption_color = ctypes.c_uint(0x001A1514)
            set_window_attribute(hwnd, 35, ctypes.byref(caption_color), ctypes.sizeof(caption_color))
            text_color = ctypes.c_uint(0x00F0E8E2)
            set_window_attribute(hwnd, 36, ctypes.byref(text_color), ctypes.sizeof(text_color))
        except Exception:
            pass
    except Exception:
        pass


def show_dark_message_box(parent, title: str, text: str, icon=None, buttons=None):
    """Universal dark-themed QMessageBox with native dark titlebar and amber hover/pressed effects."""
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtGui import QIcon
    from path_utils import get_resource_dir
    mb = QMessageBox(parent)
    dlg_icon_path = os.path.join(get_resource_dir(), "icons", "dlg_info.png")
    if os.path.exists(dlg_icon_path):
        mb.setWindowIcon(QIcon(dlg_icon_path))
    elif parent and hasattr(parent, "windowIcon"):
        mb.setWindowIcon(parent.windowIcon())
    mb.setWindowTitle(title)
    mb.setText(text)
    if icon is not None:
        mb.setIcon(icon)
    else:
        mb.setIcon(QMessageBox.Icon.Information)
    if buttons is not None:
        mb.setStandardButtons(buttons)
    mb.setStyleSheet("""
        QMessageBox {
            background: #14151a;
            color: #e2e8f0;
        }
        QLabel {
            color: #e2e8f0;
            font-size: 12px;
        }
        QPushButton {
            background: #20222b;
            color: #e2e8f0;
            border: 1px solid #333647;
            border-radius: 4px;
            padding: 5px 16px;
            font-size: 12px;
            min-width: 60px;
        }
        QPushButton:hover {
            background: #2a2d3a;
            border-color: #f59e0b;
            color: #f59e0b;
        }
        QPushButton:pressed {
            background: #14151a;
            color: #d97706;
            border-color: #d97706;
        }
    """)
    def _dark_show_event(ev):
        apply_dark_titlebar(mb)
        return QMessageBox.showEvent(mb, ev)
    mb.showEvent = _dark_show_event

    apply_dark_titlebar(mb)
    return mb.exec()


def get_darkroom_menu_style():
    """Universal floating-pill context menu stylesheet with vertical center text and icon alignment."""
    return """
        QMenu {
            background: #18191f;
            color: #e2e8f0;
            border: 1px solid #2f3240;
            border-radius: 6px;
            padding: 5px;
            font-size: 12px;
        }
        QMenu::icon {
            padding-left: 5px;
            width: 14px;
            height: 14px;
        }
        QMenu::item {
            padding: 5px 18px 5px 4px;
            margin: 1px 4px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background: #f59e0b;
            color: #111113;
            font-weight: bold;
        }
        QMenu::item:disabled {
            color: #555869;
        }
        QMenu::separator {
            height: 1px;
            background: #2a2d3a;
            margin: 4px 6px;
        }
    """
