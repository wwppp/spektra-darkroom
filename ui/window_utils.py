"""window_utils.py - Window layout, DWM titlebar, and shared UI styling utilities
"""

import sys
import ctypes
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


def apply_dark_titlebar(widget: QWidget):
    """Apply native Windows 11 dark mode theme to the window caption/titlebar."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
        # DWMWA_USE_IMMERSIVE_DARK_MODE: 20 (Windows 11 / Windows 10 20H1+), 19 (Windows 10 1809+)
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        val = ctypes.c_int(1)
        res = set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(val), ctypes.sizeof(val))
        if res != 0:
            # Fallback for older Windows 10 builds
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
            set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:
        pass


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
            padding-left: 8px;
            width: 14px;
            height: 14px;
        }
        QMenu::item {
            padding: 6px 22px 6px 30px;
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
