"""window_utils.py - Window layout, DWM titlebar, and shared UI styling utilities
"""

import os
import sys
import ctypes
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from core.path_utils import get_icon_path

_ICON_CACHE = {}


def get_icon(name: str, target_size=None, scale: float = 1.0, is_dark: bool = None) -> QIcon:
    """Universal normalized icon loader for Darkroom UI.
    Automatically trims transparent padding and rescales icons onto a standardized 
    200x200 canvas with balanced visual weight, ensuring every menu item icon renders
    at precisely identical, crisp visual size.
    Dynamically inverts/tints icons for light mode (#1e293b) while preserving alpha antialiasing.
    """
    if not name:
        return QIcon()
    path = get_icon_path(name) if not os.path.isabs(name) else name
    if not os.path.exists(path):
        return QIcon()

    if is_dark is None:
        try:
            from ui.theme_manager import theme_manager
            is_dark = theme_manager.is_dark()
        except Exception:
            is_dark = True

    cache_key = (path, target_size, scale, bool(is_dark))
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
    if not is_dark:
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(canvas.rect(), QColor("#1e293b"))
    p.end()

    icon = QIcon(canvas)
    dimmed_canvas = QPixmap(canvas.size())
    dimmed_canvas.fill(Qt.GlobalColor.transparent)
    dp = QPainter(dimmed_canvas)
    dp.setOpacity(0.32)
    dp.drawPixmap(0, 0, canvas)
    dp.end()
    icon.addPixmap(dimmed_canvas, QIcon.Mode.Disabled, QIcon.State.Off)
    icon.addPixmap(dimmed_canvas, QIcon.Mode.Disabled, QIcon.State.On)

    _ICON_CACHE[cache_key] = icon
    return icon



def apply_dark_titlebar(widget: QWidget, is_dark: bool = None):
    """Apply native Windows 10/11 immersive titlebar and custom caption bar matching theme."""
    if sys.platform != "win32":
        return
    try:
        from ui.theme_manager import theme_manager
        if is_dark is None:
            is_dark = theme_manager.is_dark()

        hwnd = int(widget.winId())
        set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute

        # 1. DWMWA_USE_IMMERSIVE_DARK_MODE: 20 (Win11 / Win10 20H1+), fallback 19 (Win10 1809+)
        val = ctypes.c_int(1 if is_dark else 0)
        res = set_window_attribute(hwnd, 20, ctypes.byref(val), ctypes.sizeof(val))
        if res != 0:
            set_window_attribute(hwnd, 19, ctypes.byref(val), ctypes.sizeof(val))

        # 2. Windows 11 Build 22000+: Enforce titlebar background and text colors
        try:
            if is_dark:
                caption_color = ctypes.c_uint(0x001A1514)  # BGR: #14151a
                text_color = ctypes.c_uint(0x00F0E8E2)     # BGR: #e2e8f0
            else:
                caption_color = ctypes.c_uint(0x00F6F2F0)  # BGR: #f0f2f6
                text_color = ctypes.c_uint(0x003B291E)     # BGR: #1e293b

            set_window_attribute(hwnd, 35, ctypes.byref(caption_color), ctypes.sizeof(caption_color))
            set_window_attribute(hwnd, 36, ctypes.byref(text_color), ctypes.sizeof(text_color))
        except Exception:
            pass
    except Exception:
        pass


def show_dark_message_box(parent, title: str, text: str, icon=None, buttons=None):
    """Universal themed QMessageBox with native titlebar and amber hover/pressed effects."""
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtGui import QIcon
    from core.path_utils import get_resource_dir
    from ui.theme_manager import theme_manager

    is_dark = theme_manager.is_dark()
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

    if is_dark:
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
    else:
        mb.setStyleSheet("""
            QMessageBox {
                background: #f8f9fc;
                color: #1e293b;
            }
            QLabel {
                color: #1e293b;
                font-size: 12px;
            }
            QPushButton {
                background: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 5px 16px;
                font-size: 12px;
                min-width: 60px;
            }
            QPushButton:hover {
                background: #f1f4f9;
                border-color: #f59e0b;
                color: #d97706;
            }
            QPushButton:pressed {
                background: #e2e8f0;
                color: #b45309;
                border-color: #d97706;
            }
        """)

    def _themed_show_event(ev):
        apply_dark_titlebar(mb, is_dark=is_dark)
        return QMessageBox.showEvent(mb, ev)
    mb.showEvent = _themed_show_event

    apply_dark_titlebar(mb, is_dark=is_dark)
    return mb.exec()


def get_darkroom_menu_style():
    """Universal floating-pill context menu stylesheet matching current theme."""
    from ui.theme_manager import theme_manager
    return theme_manager.get_menu_stylesheet()


def setup_solid_combobox(combo):
    """Enforces completely opaque solid background on QComboBox popup window, preventing layered transparent bleedthrough."""
    from PySide6.QtWidgets import QListView
    from PySide6.QtCore import Qt
    view = QListView()
    combo.setView(view)
    view.setAutoFillBackground(True)
    if hasattr(view, "window"):
        view.window().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        view.window().setAutoFillBackground(True)

