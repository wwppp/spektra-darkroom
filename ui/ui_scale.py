"""ui/ui_scale.py - global interface font scaling (小 / 中 / 大).

Every widget in this app carries its own inline stylesheet, and almost all of them spell the
text size as ``font-size: 11px``.  Qt has no global knob for that: ``QApplication.setFont``
loses to any explicit ``font-size`` rule, and QSS has no relative unit that resolves against a
preference.  So the scale is applied where the strings are handed to Qt -- ``setStyleSheet`` is
wrapped once at startup, the ``font-size: Npx`` declarations are rewritten, and the original
string is remembered so the whole app can be restyled live when the preference changes.

Only font sizes are scaled (not paddings, heights or icons), which is what the preference
promises: the text gets bigger, the layout stays.
"""

import re
import weakref

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QWidget

# 1.0 is the size the UI was designed at; the other two are the usual "a bit bigger" steps.
STEPS = {
    "small": 1.0,
    "medium": 1.12,
    "large": 1.26,
}
LABELS = {"small": "小", "medium": "中", "large": "大"}
DEFAULT_KEY = "small"

_FONT_SIZE_RE = re.compile(r"(font-size\s*:\s*)([0-9]+(?:\.[0-9]+)?)(px)")

_scale = 1.0
_key = DEFAULT_KEY
_installed = False
_base_font_pt = None
_app_css = None

# widget -> the stylesheet string as the caller wrote it (unscaled)
_records = weakref.WeakKeyDictionary()

_qt_widget_set = None
_qt_app_set = None


class _ScaleBus(QObject):
    """Fires after every restyle, for widgets whose *size* has to follow the text size
    (a fixed-width numeric box, for instance)."""
    changed = Signal()


bus = _ScaleBus()


def _rescale_css(css: str) -> str:
    if not css or _scale == 1.0:
        return css

    def _sub(m):
        value = float(m.group(2)) * _scale
        text = f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{m.group(1)}{text}{m.group(3)}"

    return _FONT_SIZE_RE.sub(_sub, css)


def _patched_set_style_sheet(self, css):
    try:
        _records[self] = css
    except TypeError:
        pass
    _qt_widget_set(self, _rescale_css(css))


def _patched_app_set_style_sheet(self, css):
    global _app_css
    _app_css = css
    _qt_app_set(self, _rescale_css(css))


def install() -> str:
    """Patch Qt once and apply the saved preference.  Returns the active key."""
    global _installed, _qt_widget_set, _qt_app_set, _base_font_pt, _key
    if not _installed:
        _qt_widget_set = QWidget.setStyleSheet
        _qt_app_set = QApplication.setStyleSheet
        QWidget.setStyleSheet = _patched_set_style_sheet
        QApplication.setStyleSheet = _patched_app_set_style_sheet
        _installed = True

    app = QApplication.instance()
    if app is not None and _base_font_pt is None:
        _base_font_pt = float(app.font().pointSizeF())
        if _base_font_pt <= 0:
            _base_font_pt = 9.0

    try:
        from core import config_manager
        _key = config_manager.get_preference("ui_font_size", DEFAULT_KEY)
    except Exception:
        _key = DEFAULT_KEY
    if _key not in STEPS:
        _key = DEFAULT_KEY
    _apply(_key, repaint=False)
    return _key


def current_key() -> str:
    return _key


def current_scale() -> float:
    return _scale


def _apply(key: str, repaint: bool = True):
    global _scale, _key
    if key not in STEPS:
        key = DEFAULT_KEY
    _key = key
    _scale = STEPS[key]

    app = QApplication.instance()
    if app is not None and _base_font_pt:
        # Widgets without an explicit font-size follow the application font.
        font = QFont(app.font())
        font.setPointSizeF(max(1.0, _base_font_pt * _scale))
        app.setFont(font)

    if repaint:
        _restyle_all()
        bus.changed.emit()


def set_font_scale(key: str) -> None:
    """Apply and persist a new scale (小/中/大)."""
    if key not in STEPS:
        key = DEFAULT_KEY
    _apply(key, repaint=True)
    try:
        from core import config_manager
        config_manager.set_preference("ui_font_size", key)
    except Exception:
        pass


def _restyle_all() -> None:
    dead = []
    for widget, css in list(_records.items()):
        try:
            _qt_widget_set(widget, _rescale_css(css))
        except (RuntimeError, ReferenceError):
            dead.append(widget)
    for widget in dead:
        _records.pop(widget, None)

    app = QApplication.instance()
    if app is not None and _app_css is not None:
        _qt_app_set(app, _rescale_css(_app_css))
