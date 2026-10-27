"""ui/theme_manager.py - Centralized UI Theme Management for SpektraDarkroom.
Supports 'dark', 'light', and 'system' (跟随系统) modes with dynamic real-time theme switching.
Maintains brand amber gold (#f59e0b) accent color strictly invariant across all themes.
"""

import sys
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QGuiApplication
from core import config_manager

DARK_COLORS = {
    "app_bg": "#121316",
    "root_border": "#282a34",
    "titlebar_bg": "#141518",
    "titlebar_border": "#22242c",
    "titlebar_text": "#f59e0b",
    "panel_bg": "#141518",
    "subpanel_bg": "#15161a",
    "panel_border": "#262834",
    "card_bg": "#15161c",
    "card_border": "#282a36",
    "card_hover_bg": "#1b1d24",
    "card_hover_border": "#4b5168",
    "card_active_bg": "rgba(245, 158, 11, 0.08)",
    "card_active_border": "#f59e0b",
    "card_thumb_bg": "#18191f",
    "text_primary": "#e2e8f0",
    "text_secondary": "#cbd5e1",
    "text_muted": "#94a3b8",
    "text_dim": "#64748b",
    "input_bg": "#171821",
    "input_border": "#282a38",
    "input_focus_bg": "#1c1d28",
    "input_focus_border": "#f59e0b",
    "splitter_bg": "#121316",
    "splitter_handle": "#cbd5e1",
    "filmstrip_bg": "#0d0e12",
    "track_rim": "#1c1e28",
    "view_toolbar_bg": "#141518",
    "view_toolbar_btn_bg": "#1c1d24",
    "view_toolbar_btn_border": "#333644",
    "view_toolbar_btn_hover": "#2a2c38",
    "scrollbar_handle": "#3f4350",
    "scrollbar_hover": "#cbd5e1",
    "scrollbar_pressed": "#f59e0b",
    "menu_bg": "#18191f",
    "menu_border": "#2f3240",
    "menu_separator": "#2a2d3a",
    "accent": "#f59e0b",
    "accent_hover": "#fbbf24",
    "accent_pressed": "#d97706",
}

LIGHT_COLORS = {
    "app_bg": "#ebedf2",
    "root_border": "#c8ccd8",
    "titlebar_bg": "#f0f2f6",
    "titlebar_border": "#d5d9e4",
    "titlebar_text": "#d97706",
    "panel_bg": "#f8f9fc",
    "subpanel_bg": "#ffffff",
    "panel_border": "#d5d9e4",
    "card_bg": "#ffffff",
    "card_border": "#d8dce6",
    "card_hover_bg": "#f1f4f9",
    "card_hover_border": "#94a3b8",
    "card_active_bg": "rgba(245, 158, 11, 0.12)",
    "card_active_border": "#f59e0b",
    "card_thumb_bg": "#e6eaf0",
    "text_primary": "#1e293b",
    "text_secondary": "#334155",
    "text_muted": "#64748b",
    "text_dim": "#94a3b8",
    "input_bg": "#ffffff",
    "input_border": "#cbd5e1",
    "input_focus_bg": "#ffffff",
    "input_focus_border": "#f59e0b",
    "splitter_bg": "#ebedf2",
    "splitter_handle": "#64748b",
    "filmstrip_bg": "#e0e3ea",
    "track_rim": "#c2c8d4",
    "view_toolbar_bg": "#f0f2f6",
    "view_toolbar_btn_bg": "#ffffff",
    "view_toolbar_btn_border": "#cbd5e1",
    "view_toolbar_btn_hover": "#e2e8f0",
    "scrollbar_handle": "#94a3b8",
    "scrollbar_hover": "#64748b",
    "scrollbar_pressed": "#f59e0b",
    "menu_bg": "#ffffff",
    "menu_border": "#d8dce6",
    "menu_separator": "#e2e8f0",
    "accent": "#f59e0b",
    "accent_hover": "#fbbf24",
    "accent_pressed": "#d97706",
}


def detect_system_theme() -> str:
    """Detect whether Windows or system is currently running in light or dark mode."""
    try:
        app = QGuiApplication.instance()
        if app:
            hints = QGuiApplication.styleHints()
            if hints and hasattr(hints, "colorScheme"):
                cs = hints.colorScheme()
                if cs == Qt.ColorScheme.Light:
                    return "light"
                elif cs == Qt.ColorScheme.Dark:
                    return "dark"
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
                val, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
                if val == 1:
                    return "light"
                elif val == 0:
                    return "dark"
        except Exception:
            pass

    return "dark"


class ThemeManager(QObject):
    """Centralized singleton managing application theme state and change notifications."""
    themeChanged = Signal(str)  # Emits effective theme: "dark" or "light"

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = config_manager.get_preference("ui_theme", "system")
        if self._mode not in ("dark", "light", "system"):
            self._mode = "system"
        self._effective_theme = self._resolve_theme(self._mode)
        self._apply_app_palette()

        # Hook to Qt system color scheme changes if available
        try:
            hints = QGuiApplication.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        except Exception:
            pass

    def _apply_app_palette(self):
        app = QGuiApplication.instance()
        if not app:
            return
        from PySide6.QtGui import QPalette, QColor
        palette = QPalette()
        if self.is_dark():
            palette.setColor(QPalette.ColorRole.Window, QColor(20, 21, 26))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(226, 232, 240))
            palette.setColor(QPalette.ColorRole.Base, QColor(18, 19, 24))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(28, 30, 38))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(36, 38, 48))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor(240, 240, 240))
            palette.setColor(QPalette.ColorRole.Text, QColor(226, 232, 240))
            palette.setColor(QPalette.ColorRole.Button, QColor(31, 33, 42))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(226, 232, 240))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(245, 158, 11))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        else:
            palette.setColor(QPalette.ColorRole.Window, QColor(235, 237, 242))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(30, 41, 59))
            palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(241, 244, 249))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor(30, 41, 59))
            palette.setColor(QPalette.ColorRole.Text, QColor(30, 41, 59))
            palette.setColor(QPalette.ColorRole.Button, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(30, 41, 59))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(245, 158, 11))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(17, 17, 17))
        app.setPalette(palette)
        if hasattr(app, "setStyleSheet"):
            # Kept for the few tooltips Qt still renders itself (menu item tips); widget tooltips
            # are painted by ui.tooltip.TooltipBubble, which reads the theme at paint time.
            app.setStyleSheet(self.get_tooltip_stylesheet())

    def get_tooltip_stylesheet(self) -> str:
        """Universal floating tooltip bubble stylesheet matching active theme.

        Apply this once to QApplication only (done automatically in _apply_app_palette) and
        never inject it into a child widget's stylesheet: Qt resolves QToolTip styling from the
        hovered widget's stylesheet chain, so a widget-level copy shadows the refreshed global
        rule and keeps the previous theme's colours after a theme switch.
        """
        is_dark = self.is_dark()
        tooltip_bg = "rgba(22, 24, 30, 0.95)" if is_dark else "#ffffff"
        tooltip_fg = "#f1f5f9" if is_dark else "#1e293b"
        tooltip_border = "1px solid #333644" if is_dark else "1px solid #cbd5e1"
        return f"""
            QToolTip {{
                background-color: {tooltip_bg};
                color: {tooltip_fg};
                border: {tooltip_border};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 400;
                line-height: 1.4;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
            }}
        """

    def widget_tooltip_stylesheet(self) -> str:
        """Legacy no-op for widget-level call sites - returns an empty fragment.

        Tooltip styling is global now. Injecting QToolTip rules into individual widget
        stylesheets made them shadow the theme-refreshed global rule, which left tooltips
        stuck in the old theme's colours.
        """
        return ""

    def _on_system_scheme_changed(self):
        if self._mode == "system":
            new_eff = self._resolve_theme("system")
            if new_eff != self._effective_theme:
                self._effective_theme = new_eff
                self._apply_app_palette()
                self.themeChanged.emit(new_eff)

    def _resolve_theme(self, mode: str) -> str:
        if mode == "system":
            return detect_system_theme()
        return "light" if mode == "light" else "dark"

    def get_theme_mode(self) -> str:
        """Returns configured mode: 'dark', 'light', or 'system'."""
        return self._mode

    def get_effective_theme(self) -> str:
        """Returns currently rendered theme: 'dark' or 'light'."""
        return self._effective_theme

    def is_dark(self) -> bool:
        """Convenience method returning True if dark theme is currently effective."""
        return self._effective_theme == "dark"

    def set_theme_mode(self, mode: str):
        """Set theme mode ('dark', 'light', or 'system'), persist to preferences, and notify listeners."""
        if mode not in ("dark", "light", "system"):
            mode = "dark"
        old_mode = self._mode
        old_eff = self._effective_theme
        self._mode = mode
        config_manager.set_preference("ui_theme", mode)

        new_eff = self._resolve_theme(mode)
        self._effective_theme = new_eff
        self._apply_app_palette()

        if new_eff != old_eff or mode != old_mode:
            self.themeChanged.emit(new_eff)

    def get_colors(self) -> dict:
        """Return the palette dictionary for the current effective theme."""
        return DARK_COLORS if self.is_dark() else LIGHT_COLORS

    def color(self, key: str) -> str:
        """Get a single hex color from the active theme palette."""
        colors = self.get_colors()
        return colors.get(key, "#f59e0b")

    def get_menu_stylesheet(self) -> str:
        """Universal floating-pill context menu & menubar stylesheet matching current theme."""
        if self.is_dark():
            return """
                QMenuBar {
                    background: transparent;
                    color: #e2e8f0;
                    font-size: 12px;
                    padding-left: 4px;
                    border: none;
                    margin: 0px;
                }
                QMenuBar::item {
                    background: transparent;
                    color: #e2e8f0;
                    padding: 4px 9px;
                    border-radius: 4px;
                }
                QMenuBar::item:selected {
                    background: #232530;
                    color: #ffffff;
                }
                QMenuBar::item:disabled {
                    color: #4b5563;
                    background: transparent;
                }
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
                    font-weight: normal;
                }
                QMenu::item:disabled {
                    color: #4b5563;
                    background: transparent;
                }
                QMenu::item:disabled:selected {
                    color: #4b5563;
                    background: transparent;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2a2d3a;
                    margin: 4px 6px;
                }
            """
        else:
            return """
                QMenuBar {
                    background: transparent;
                    color: #1e293b;
                    font-size: 12px;
                    padding-left: 4px;
                    border: none;
                    margin: 0px;
                }
                QMenuBar::item {
                    background: transparent;
                    color: #1e293b;
                    padding: 4px 9px;
                    border-radius: 4px;
                }
                QMenuBar::item:selected {
                    background: #e2e8f0;
                    color: #0f172a;
                }
                QMenuBar::item:disabled {
                    color: #94a3b8;
                    background: transparent;
                }
                QMenu {
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #d5d9e4;
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
                    font-weight: normal;
                }
                QMenu::item:disabled {
                    color: #94a3b8;
                    background: transparent;
                }
                QMenu::item:disabled:selected {
                    color: #94a3b8;
                    background: transparent;
                }
                QMenu::separator {
                    height: 1px;
                    background: #e2e8f0;
                    margin: 4px 6px;
                }
            """


# Global singleton instance
theme_manager = ThemeManager.instance()
