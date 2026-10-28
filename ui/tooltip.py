"""ui/tooltip.py - Self-rendered tooltip bubbles.

Qt's QToolTip (QTipLabel) is reused across tooltips, sizes itself with the hovered widget's
inherited font before any widget-level style applies, and resolves its colours from the hovered
widget's stylesheet chain. Patching that late (setting a stylesheet on the tip when it shows)
produced two visible defects: text clipped at the bottom (bubble sized for the old font) and
occasionally un-themed black bubbles (first frame painted before the patch landed).

This module replaces that path with a bubble we paint ourselves: the size is computed from the
same font and wrapping used for painting, and the colours are resolved at paint time from the
live theme. Tooltip events are consumed, so Qt no longer creates QTipLabel for widget tooltips.
"""

from PySide6.QtCore import QEvent, QObject, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from ui.theme_manager import theme_manager

_MAX_TEXT_WIDTH = 360   # logical px before wrapping
_PAD_X = 10
_PAD_Y = 7
_RADIUS = 6


class TooltipBubble(QWidget):
    """Frameless, click-through card that paints one tooltip message."""

    def __init__(self):
        flags = (Qt.WindowType.ToolTip
                 | Qt.WindowType.FramelessWindowHint
                 | Qt.WindowType.NoDropShadowWindowHint
                 | Qt.WindowType.WindowStaysOnTopHint)
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._text = ""
        self._text_size = QRect(0, 0, 0, 0)

    @staticmethod
    def _bubble_font() -> QFont:
        font = QFont(QApplication.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(font.pointSizeF(), 8.5))
        return font

    def prepare(self, text: str):
        """Measure exactly the way paintEvent draws, so the text can never be clipped."""
        self._text = text
        metrics = QFontMetrics(self._bubble_font())
        self._text_size = metrics.boundingRect(
            QRect(0, 0, _MAX_TEXT_WIDTH, 10000),
            int(Qt.TextFlag.TextWordWrap),
            text,
        )
        self.setFixedSize(self._text_size.width() + 2 * _PAD_X,
                          self._text_size.height() + 2 * _PAD_Y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        is_dark = theme_manager.is_dark()
        bg = QColor(22, 24, 30, 242) if is_dark else QColor(255, 255, 255, 250)
        fg = QColor("#f1f5f9") if is_dark else QColor("#1e293b")
        border = QColor("#333644") if is_dark else QColor("#cbd5e1")
        body = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(border, 1))
        painter.setBrush(bg)
        painter.drawRoundedRect(body, _RADIUS, _RADIUS)
        painter.setPen(fg)
        painter.setFont(self._bubble_font())
        painter.drawText(
            QRect(_PAD_X, _PAD_Y, self._text_size.width(), self._text_size.height()),
            int(Qt.TextFlag.TextWordWrap) | int(Qt.AlignmentFlag.AlignLeft),
            self._text,
        )


class TooltipController(QObject):
    """Application-wide filter that turns QEvent.ToolTip into our own bubble."""

    def __init__(self, app):
        super().__init__(app)
        self._bubble = TooltipBubble()
        self._visible = False

    def hide_bubble(self):
        if self._visible:
            self._visible = False
            self._bubble.hide()

    @staticmethod
    def _is_combobox_popup(obj) -> bool:
        curr = obj if isinstance(obj, QWidget) else None
        while curr is not None:
            cname = curr.metaObject().className()
            if cname in ("QComboBoxPrivateContainer", "DarkroomComboListView", "AdobeComboListView"):
                return True
            if hasattr(curr, "delegate") and hasattr(getattr(curr, "delegate", None), "combo"):
                return True
            curr = curr.parent()
        return False

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QEvent.Type.ToolTip:
            try:
                if self._is_combobox_popup(obj):
                    self.hide_bubble()
                    return True

                widget = obj if isinstance(obj, QWidget) else None
                text = widget.toolTip() if widget is not None else ""
                if text:
                    self._show(text, event.globalPos())
                    return True      # consume: Qt must not create a QTipLabel
            except Exception:
                pass
            return False
        if etype in (QEvent.Type.Leave, QEvent.Type.MouseButtonPress, QEvent.Type.Wheel,
                     QEvent.Type.KeyPress, QEvent.Type.Hide, QEvent.Type.WindowDeactivate,
                     QEvent.Type.ApplicationDeactivate):
            self.hide_bubble()
        return False

    def _show(self, text: str, global_pos):
        bubble = self._bubble
        bubble.prepare(text)
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else QRect(0, 0, 1920, 1080)
        x = global_pos.x() + 14
        y = global_pos.y() + 22
        if x + bubble.width() > area.right():
            x = max(area.left(), global_pos.x() - bubble.width() - 8)
        if y + bubble.height() > area.bottom():
            y = max(area.top(), global_pos.y() - bubble.height() - 12)
        bubble.move(x, y)
        bubble.show()
        bubble.raise_()
        bubble.update()
        self._visible = True


def install_tooltip_bubble(app):
    """Install once per QApplication; safe to call repeatedly."""
    if getattr(app, "_spektra_tooltip", None) is not None:
        return app._spektra_tooltip
    controller = TooltipController(app)
    app.installEventFilter(controller)
    app._spektra_tooltip = controller
    return controller
