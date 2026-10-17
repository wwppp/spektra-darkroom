"""smooth_scroll.py - Universal smooth easing scroll area with top/bottom fade edge indicators
Provides silky smooth non-linear animated scrolling (OutCubic) and subtle overscroll edge cues.
"""

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRectF
from PySide6.QtGui import QPainter, QColor, QLinearGradient
from PySide6.QtWidgets import QScrollArea, QWidget

from ui.theme_manager import theme_manager


class SmoothScrollArea(QScrollArea):
    """QScrollArea enhanced with:
    1. Smooth animated non-linear scrolling driven by QEasingCurve.OutCubic.
    2. Subtle top/bottom edge fade indicators showing scroll limits.
    """
    def __init__(self, parent=None, is_horizontal=False, enable_edge_fade=None):
        super().__init__(parent)
        self.is_horizontal = is_horizontal
        self.enable_edge_fade = (not is_horizontal) if enable_edge_fade is None else enable_edge_fade
        self.setWidgetResizable(True)
        self._target_value = None

        # Animated property on scrollbar
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        self._anim = QPropertyAnimation(sb, b"value", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Overlay widget on top of viewport child widgets (only when edge fade is enabled)
        if self.enable_edge_fade:
            self._overlay = _ScrollEdgeOverlay(self)
            sb.valueChanged.connect(self._on_scroll_value_changed)
        else:
            self._overlay = None

        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

    def apply_theme(self, theme_name=None):
        is_dark = theme_manager.is_dark()
        if is_dark:
            self.setStyleSheet("""
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:vertical {
                    background: rgba(18, 19, 24, 0.75);
                    width: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical {
                    background: #3f4350;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical:hover {
                    background: #cbd5e1;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    border: none;
                }
                QScrollBar:horizontal {
                    background: rgba(18, 19, 24, 0.75);
                    height: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:horizontal {
                    background: #3f4350;
                    min-width: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:horizontal:hover {
                    background: #cbd5e1;
                    border-radius: 3px;
                }
                QScrollBar::handle:horizontal:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                    background: none;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:vertical {
                    background: transparent;
                    width: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical {
                    background: #94a3b8;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical:hover {
                    background: #64748b;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    border: none;
                }
                QScrollBar:horizontal {
                    background: transparent;
                    height: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:horizontal {
                    background: #94a3b8;
                    min-width: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:horizontal:hover {
                    background: #64748b;
                    border-radius: 3px;
                }
                QScrollBar::handle:horizontal:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                    background: none;
                    border: none;
                }
            """)
        if hasattr(self, '_overlay') and self._overlay:
            self._sync_overlay()

    def _sync_overlay(self):
        vp = self.viewport()
        if vp and getattr(self, '_overlay', None) is not None:
            self._overlay.setGeometry(vp.rect())
            self._overlay.raise_()
            self._overlay.update()

    def _on_scroll_value_changed(self, val):
        self._sync_overlay()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_overlay()

    def wheelEvent(self, event):
        """Intercept mouse wheel to drive smooth interpolation animation."""
        p = self.parent()
        if p and hasattr(p, "_cancel_pending_scroll_timers"):
            try:
                p._cancel_pending_scroll_timers()
            except Exception:
                pass

        if self.is_horizontal:
            delta = event.angleDelta().x() or event.angleDelta().y()
            sb = self.horizontalScrollBar()
        else:
            delta = event.angleDelta().y()
            sb = self.verticalScrollBar()

        if delta == 0 or sb.maximum() <= 0:
            super().wheelEvent(event)
            return

        step = int(delta * 1.25)
        current_val = sb.value()

        if self._anim.state() == QPropertyAnimation.State.Running and self._target_value is not None:
            base_val = self._target_value
        else:
            base_val = current_val

        target = max(0, min(sb.maximum(), base_val - step))
        self._target_value = target

        self._anim.stop()
        self._anim.setStartValue(current_val)
        self._anim.setEndValue(target)
        self._anim.start()
        event.accept()

    def scroll_to(self, target_value: int, animate: bool = True):
        """Scroll to target position either with OutCubic easing or immediately."""
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if not sb:
            return
        target = max(sb.minimum(), min(sb.maximum(), int(target_value)))
        if not animate:
            if self._anim.state() == QPropertyAnimation.State.Running:
                self._anim.stop()
            self._target_value = target
            sb.setValue(target)
            return

        current_val = sb.value()
        if current_val == target:
            return

        self._target_value = target
        self._anim.stop()
        self._anim.setStartValue(current_val)
        self._anim.setEndValue(target)
        self._anim.start()


class _ScrollEdgeOverlay(QWidget):
    """Transparent overlay widget floating above all viewport children to draw subtle edge fade shadows."""
    def __init__(self, parent_scroll_area: SmoothScrollArea):
        super().__init__(parent_scroll_area.viewport())
        self.scroll_area = parent_scroll_area
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        if not getattr(self.scroll_area, 'enable_edge_fade', True):
            return
        vp = self.scroll_area.viewport()
        if not vp:
            return

        w = vp.width()
        h = vp.height()
        painter = QPainter(self)
        is_dark = theme_manager.is_dark()
        if is_dark:
            fade_col = QColor(18, 19, 24, 210)
            trans_col = QColor(18, 19, 24, 0)
        else:
            bg_hex = theme_manager.color("panel_bg")
            base = QColor(bg_hex)
            fade_col = QColor(base.red(), base.green(), base.blue(), 220)
            trans_col = QColor(base.red(), base.green(), base.blue(), 0)

        if not self.scroll_area.is_horizontal:
            sb = self.scroll_area.verticalScrollBar()
            if sb.maximum() > 0:
                # Top edge indicator: if scrolled down from 0
                if sb.value() > 0:
                    grad_top = QLinearGradient(0, 0, 0, 16)
                    grad_top.setColorAt(0.0, fade_col)
                    grad_top.setColorAt(1.0, trans_col)
                    painter.fillRect(QRectF(0, 0, w, 16), grad_top)

                # Bottom edge indicator: if not yet at maximum
                if sb.value() < sb.maximum():
                    grad_bottom = QLinearGradient(0, h - 16, 0, h)
                    grad_bottom.setColorAt(0.0, trans_col)
                    grad_bottom.setColorAt(1.0, fade_col)
                    painter.fillRect(QRectF(0, h - 16, w, 16), grad_bottom)
        else:
            sb = self.scroll_area.horizontalScrollBar()
            if sb.maximum() > 0:
                if sb.value() > 0:
                    grad_left = QLinearGradient(0, 0, 16, 0)
                    grad_left.setColorAt(0.0, fade_col)
                    grad_left.setColorAt(1.0, trans_col)
                    painter.fillRect(QRectF(0, 0, 16, h), grad_left)

                if sb.value() < sb.maximum():
                    grad_right = QLinearGradient(w - 16, 0, w, 0)
                    grad_right.setColorAt(0.0, trans_col)
                    grad_right.setColorAt(1.0, fade_col)
                    painter.fillRect(QRectF(w - 16, 0, 16, h), grad_right)
        painter.end()
