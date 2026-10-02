"""smooth_scroll.py - Universal smooth easing scroll area with top/bottom fade edge indicators
Provides silky smooth non-linear animated scrolling (OutCubic) and subtle overscroll edge cues.
"""

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRectF
from PySide6.QtGui import QPainter, QColor, QLinearGradient
from PySide6.QtWidgets import QScrollArea, QWidget


class SmoothScrollArea(QScrollArea):
    """QScrollArea enhanced with:
    1. Smooth animated non-linear scrolling driven by QEasingCurve.OutCubic.
    2. Subtle top/bottom edge fade indicators showing scroll limits.
    """
    def __init__(self, parent=None, is_horizontal=False):
        super().__init__(parent)
        self.is_horizontal = is_horizontal
        self.setWidgetResizable(True)
        self._target_value = None

        # Unified Darkroom Industrial Scrollbar Stylesheet
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
                background: #474d61;
                min-height: 24px;
                border-radius: 3px;
                border: none;
            }
            QScrollBar::handle:vertical:hover {
                background: #f59e0b;
            }
            QScrollBar::handle:vertical:pressed {
                background: #d97706;
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
                background: #474d61;
                min-width: 24px;
                border-radius: 3px;
                border: none;
            }
            QScrollBar::handle:horizontal:hover {
                background: #f59e0b;
            }
            QScrollBar::handle:horizontal:pressed {
                background: #d97706;
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

        # Animated property on scrollbar
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        self._anim = QPropertyAnimation(sb, b"value", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Repaint viewport to update edge indicators on scroll
        sb.valueChanged.connect(self._on_scroll_value_changed)

    def _on_scroll_value_changed(self, val):
        vp = self.viewport()
        if vp:
            vp.update()

    def wheelEvent(self, event):
        """Intercept mouse wheel to drive smooth interpolation animation."""
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

    def paintEvent(self, event):
        super().paintEvent(event)
        # Overlay subtle edge fade indicator in viewport
        vp = self.viewport()
        if not vp:
            return

        w = vp.width()
        h = vp.height()
        painter = QPainter(vp)

        if not self.is_horizontal:
            sb = self.verticalScrollBar()
            if sb.maximum() > 0:
                # Top edge indicator: if scrolled down from 0
                if sb.value() > 0:
                    grad_top = QLinearGradient(0, 0, 0, 14)
                    grad_top.setColorAt(0.0, QColor(0, 0, 0, 130))
                    grad_top.setColorAt(1.0, QColor(0, 0, 0, 0))
                    painter.fillRect(QRectF(0, 0, w, 14), grad_top)

                # Bottom edge indicator: if not yet at maximum
                if sb.value() < sb.maximum():
                    grad_bottom = QLinearGradient(0, h - 14, 0, h)
                    grad_bottom.setColorAt(0.0, QColor(0, 0, 0, 0))
                    grad_bottom.setColorAt(1.0, QColor(0, 0, 0, 130))
                    painter.fillRect(QRectF(0, h - 14, w, 14), grad_bottom)
        else:
            sb = self.horizontalScrollBar()
            if sb.maximum() > 0:
                if sb.value() > 0:
                    grad_left = QLinearGradient(0, 0, 14, 0)
                    grad_left.setColorAt(0.0, QColor(0, 0, 0, 130))
                    grad_left.setColorAt(1.0, QColor(0, 0, 0, 0))
                    painter.fillRect(QRectF(0, 0, 14, h), grad_left)

                if sb.value() < sb.maximum():
                    grad_right = QLinearGradient(w - 14, 0, w, 0)
                    grad_right.setColorAt(0.0, QColor(0, 0, 0, 0))
                    grad_right.setColorAt(1.0, QColor(0, 0, 0, 130))
                    painter.fillRect(QRectF(w - 14, 0, 14, h), grad_right)
        painter.end()
