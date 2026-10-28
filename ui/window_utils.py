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
    """Enforces modern Darkroom popup styling on QComboBox:
    - Opaque background preventing layered transparent bleedthrough
    - Smooth inertial mouse-wheel scrolling (DarkroomComboListView with ScrollPerPixel)
    - Custom item delegate with hover feedback & left amber accent bar (DarkroomComboItemDelegate)
    - Adaptive popup width based on content text length
    - Smart screen-edge upward/downward direction detection
    - Smooth opening expansion & opacity fade animation
    """
    from PySide6.QtWidgets import QListView
    from PySide6.QtCore import Qt, QPoint, QRect, QVariantAnimation, QEasingCurve
    from PySide6.QtGui import QGuiApplication
    from ui.adobe_scrub_slider import DarkroomComboListView, DarkroomComboItemDelegate, DarkroomComboBox

    if not isinstance(combo.view(), DarkroomComboListView):
        view = DarkroomComboListView()
        delegate = DarkroomComboItemDelegate(combo)
        view.delegate = delegate
        combo.setView(view)
        combo.setItemDelegate(delegate)
        view.setAutoFillBackground(True)
        if hasattr(view, "window"):
            view.window().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            view.window().setAutoFillBackground(True)
        combo.setCursor(Qt.CursorShape.PointingHandCursor)
        combo.wheelEvent = lambda ev: ev.ignore()

    def _apply_solid_combo_tooltip():
        # Tooltip styling is global now (theme_manager.get_tooltip_stylesheet applied on
        # QApplication). A QToolTip rule on the popup view would shadow that global rule and
        # keep the previous theme's colours after a theme switch.
        try:
            if combo and combo.view():
                combo.view().setStyleSheet("")
        except RuntimeError:
            pass

    _apply_solid_combo_tooltip()
    try:
        from ui.theme_manager import theme_manager
        theme_manager.themeChanged.connect(lambda _: _apply_solid_combo_tooltip())
    except Exception:
        pass

    if not isinstance(combo, DarkroomComboBox) and not getattr(combo, "_setup_solid_hooked", False):
        combo._setup_solid_hooked = True
        orig_show_popup = combo.showPopup
        orig_hide_popup = combo.hidePopup

        def _hooked_show_popup():
            if combo.count() == 0:
                orig_show_popup()
                return

            fm = combo.fontMetrics()
            max_text_w = 0
            for i in range(combo.count()):
                w = fm.horizontalAdvance(combo.itemText(i))
                if w > max_text_w:
                    max_text_w = w
            target_width = max(combo.width(), max_text_w + 36)

            screen = combo.screen() or QGuiApplication.primaryScreen()
            avail = screen.availableGeometry() if screen else QRect(0, 0, 1920, 1080)
            p_global = combo.mapToGlobal(QPoint(0, 0))

            # Clamp width to maximum screen width
            max_allowed_w = max(combo.width(), avail.width() - 24)
            target_width = min(target_width, max_allowed_w)

            # Adaptive horizontal positioning: if expanding rightwards overflows screen edge, expand to the left!
            target_x = p_global.x()
            if target_x + target_width + 12 > avail.right():
                right_edge = min(avail.right() - 8, p_global.x() + combo.width())
                target_x = max(avail.left() + 8, right_edge - target_width)

            combo.view().setMinimumWidth(target_width)
            combo.view().setMaximumWidth(target_width)

            item_h = 24
            space_below = avail.bottom() - (p_global.y() + combo.height())
            space_above = p_global.y() - avail.top()
            max_space = max(space_below, space_above)
            fit_items = max(6, min(24, int((max_space - 20) / item_h)))
            combo.setMaxVisibleItems(min(combo.count(), fit_items))

            # Zero opacity before orig_show_popup() to completely eliminate initial-frame flash
            container = combo.view().window() if combo.view() else None
            if container:
                container.setWindowOpacity(0.0)

            combo._orig_sb_policy = combo.view().verticalScrollBarPolicy()
            combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

            orig_show_popup()

            if container:
                orig_geo = container.geometry()
                is_upward = orig_geo.y() < p_global.y()
                if hasattr(combo, '_popup_anim') and combo._popup_anim:
                    combo._popup_anim.stop()

                # Ensure final dest_geo uses adaptive target_x and target_width!
                dest_geo = QRect(target_x, orig_geo.y(), target_width, orig_geo.height())

                if is_upward:
                    start_geo = QRect(target_x, dest_geo.bottom(), target_width, 1)
                else:
                    start_geo = QRect(target_x, dest_geo.y(), target_width, 1)
                container.setGeometry(start_geo)

                combo._popup_anim = QVariantAnimation(combo)
                combo._popup_anim.setDuration(120)
                combo._popup_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
                combo._popup_anim.setStartValue(start_geo)
                combo._popup_anim.setEndValue(dest_geo)

                def _on_step(r):
                    container.setGeometry(r)
                    prog = combo._popup_anim.currentTime() / max(1, combo._popup_anim.duration())
                    container.setWindowOpacity(min(1.0, 0.1 + 0.9 * prog))

                def _on_finished():
                    container.setGeometry(dest_geo)
                    container.setWindowOpacity(1.0)
                    if hasattr(combo, '_orig_sb_policy') and combo._orig_sb_policy is not None:
                        combo.view().setVerticalScrollBarPolicy(combo._orig_sb_policy)

                combo._popup_anim.valueChanged.connect(_on_step)
                combo._popup_anim.finished.connect(_on_finished)
                combo._popup_anim.start()

        def _hooked_hide_popup():
            if hasattr(combo, '_popup_anim') and combo._popup_anim:
                combo._popup_anim.stop()
            if hasattr(combo, '_orig_sb_policy') and combo._orig_sb_policy is not None:
                combo.view().setVerticalScrollBarPolicy(combo._orig_sb_policy)
            container = combo.view().window() if combo.view() else None
            if container:
                container.setWindowOpacity(0.0)
            orig_hide_popup()

        combo.showPopup = _hooked_show_popup
        combo.hidePopup = _hooked_hide_popup


def get_standard_checkbox_style(is_dark: bool = None) -> str:
    """Universal standard QCheckBox stylesheet aligned with Export Dialog design."""
    if is_dark is None:
        try:
            from ui.theme_manager import theme_manager
            is_dark = theme_manager.is_dark()
        except Exception:
            is_dark = True

    from core.path_utils import get_resource_dir
    chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")
    from ui.theme_manager import theme_manager
    tt_ss = theme_manager.widget_tooltip_stylesheet()

    if is_dark:
        return f"""
            QCheckBox {{
                color: #c4c4cc;
                spacing: 8px;
                font-size: 11px;
                font-weight: 500;
                outline: none;
                background: transparent;
            }}
            QCheckBox:hover {{
                color: #ffffff;
            }}
            QCheckBox:focus {{
                outline: none;
            }}
            QCheckBox::indicator {{
                width: 12px;
                height: 12px;
                border: 1px solid #8b92a5;
                border-radius: 2px;
                background: transparent;
                outline: none;
            }}
            QCheckBox::indicator:hover {{
                border-color: #f59e0b;
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid #f59e0b;
                background: #14151a;
                image: url("{chk_amber}");
            }}
            QCheckBox:disabled {{
                color: #4b5563;
            }}
            QCheckBox::indicator:disabled {{
                border: 1px solid #282a36;
                background: #0f1013;
            }}
            {tt_ss}
        """
    else:
        return f"""
            QCheckBox {{
                color: #334155;
                spacing: 8px;
                font-size: 11px;
                font-weight: 500;
                outline: none;
                background: transparent;
            }}
            QCheckBox:hover {{
                color: #0f172a;
            }}
            QCheckBox:focus {{
                outline: none;
            }}
            QCheckBox::indicator {{
                width: 12px;
                height: 12px;
                border: 1px solid #94a3b8;
                border-radius: 2px;
                background: #ffffff;
                outline: none;
            }}
            QCheckBox::indicator:hover {{
                border-color: #f59e0b;
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid #f59e0b;
                background: #ffffff;
                image: url("{chk_amber}");
            }}
            QCheckBox:disabled {{
                color: #94a3b8;
            }}
            QCheckBox::indicator:disabled {{
                border: 1px solid #cbd5e1;
                background: #f1f5f9;
            }}
            {tt_ss}
        """

