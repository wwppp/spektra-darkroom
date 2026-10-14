"""SpektraDarkroom FadingStackedWidget
Smooth, GPU-friendly cross-fade transitions between application screens
(e.g., Startup Welcome Screen <-> Darkroom Workspace).
"""

import os
import logging
from PySide6.QtWidgets import QStackedWidget, QWidget
from PySide6.QtCore import Qt, Property, QPropertyAnimation, QEasingCurve, Signal
from PySide6.QtGui import QPainter, QPixmap

logger = logging.getLogger("SpektraDarkroom.FadingStackedWidget")


class FadeOverlay(QWidget):
    """Transparent overlay painting an outgoing snapshot with controllable opacity."""
    def __init__(self, parent: QWidget, pixmap: QPixmap):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.pixmap = pixmap
        self._opacity = 1.0

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, val: float):
        self._opacity = max(0.0, min(1.0, float(val)))
        self.update()

    opacity = Property(float, get_opacity, set_opacity)

    def paintEvent(self, event):
        if self.pixmap and not self.pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.setOpacity(self._opacity)
            painter.drawPixmap(self.rect(), self.pixmap)


class FadingStackedWidget(QStackedWidget):
    """A QStackedWidget subclass that supports smooth cross-fade page transitions
    without disrupting underlying GPU/Vulkan renderers or blurring text.
    """
    transitionStarted = Signal(int, int)   # from_idx, to_idx
    transitionFinished = Signal(int)       # current_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fade_overlay = None
        self._anim = None
        self._is_animating = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fade_overlay and self._fade_overlay.isVisible():
            self._fade_overlay.setGeometry(self.rect())

    def setCurrentIndexAnimated(self, next_idx: int, duration_ms: int = 250, on_finished=None):
        """Switches to next_idx using a smooth cross-fade animation.
        If FAST_TEST is set, duration_ms <= 0, or widget is hidden/unmapped, switches instantly.
        """
        curr_idx = self.currentIndex()
        if next_idx == curr_idx:
            if callable(on_finished):
                on_finished()
            return

        if next_idx < 0 or next_idx >= self.count():
            return

        # Instant fallback for automated tests, negative duration, or hidden widgets
        if os.environ.get("FAST_TEST") or duration_ms <= 0 or not self.isVisible():
            self._stop_current_animation()
            super().setCurrentIndex(next_idx)
            self.transitionFinished.emit(next_idx)
            if callable(on_finished):
                on_finished()
            return

        old_widget = self.currentWidget()
        if old_widget is None or old_widget.width() <= 0 or old_widget.height() <= 0:
            super().setCurrentIndex(next_idx)
            self.transitionFinished.emit(next_idx)
            if callable(on_finished):
                on_finished()
            return

        # Stop any active transition
        self._stop_current_animation()

        # Capture snapshot of the outgoing page
        try:
            pixmap = old_widget.grab()
        except Exception as e:
            logger.debug(f"Snapshot grab error: {e}")
            pixmap = None

        if pixmap is None or pixmap.isNull():
            super().setCurrentIndex(next_idx)
            self.transitionFinished.emit(next_idx)
            if callable(on_finished):
                on_finished()
            return

        # Create overlay covering the stacked widget
        self._fade_overlay = FadeOverlay(self, pixmap)
        self._fade_overlay.setGeometry(self.rect())
        self._fade_overlay.show()
        self._fade_overlay.raise_()

        # Switch underlying page immediately so it is laid out and rendered underneath
        self._is_animating = True
        self.transitionStarted.emit(curr_idx, next_idx)
        super().setCurrentIndex(next_idx)

        # Smoothly animate overlay opacity: 1.0 -> 0.0
        anim = QPropertyAnimation(self._fade_overlay, b"opacity", self)
        anim.setDuration(max(10, duration_ms))
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def _on_anim_finished():
            self._cleanup_overlay()
            self._is_animating = False
            self._anim = None
            self.transitionFinished.emit(next_idx)
            if callable(on_finished):
                on_finished()

        anim.finished.connect(_on_anim_finished)
        self._anim = anim
        anim.start()

    def _stop_current_animation(self):
        if self._anim:
            try:
                self._anim.stop()
            except Exception:
                pass
            self._anim = None
        self._cleanup_overlay()
        self._is_animating = False

    def _cleanup_overlay(self):
        if self._fade_overlay:
            try:
                self._fade_overlay.hide()
                self._fade_overlay.deleteLater()
            except Exception:
                pass
            self._fade_overlay = None
