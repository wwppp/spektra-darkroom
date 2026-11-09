"""smooth_scroll.py - Universal smooth easing scroll area with physics-based momentum and edge indicators
Provides silky smooth kinetic momentum scrolling (natural friction decay) and subtle overscroll edge cues.
"""

import time
from PySide6.QtCore import (Qt, QAbstractAnimation, QVariantAnimation, QEasingCurve,
                            QRectF, QTimer, QEvent)
from PySide6.QtGui import QPainter, QColor, QLinearGradient
from PySide6.QtWidgets import QScrollArea, QWidget

from ui.theme_manager import theme_manager
from ui.touch_scroll import install_touch_scroll


class _PixelSettleAnimation(QVariantAnimation):
    """Scroll-bar motion that does not stall for the last pixel.

    ``QPropertyAnimation`` on ``QScrollBar.value`` animates an **integer**, and every ease-out
    curve ends with zero velocity.  Together they mean the remaining half pixel is rounded away
    for the last third of the run: the bar sits perfectly still and then hops the final pixel --
    measured as 110 ms of stillness followed by one 1px step on a 280 ms settle, which is exactly
    the "one pixel twitch right before it stops" reported for every panel.

    Driving a *float* through ``round()`` fixes the arithmetic but not the cause: the curve has
    no speed left to spend.  So the curve keeps a small linear component, sized per run so the
    last frame still travels about one pixel::

        f(t) = (1 - v) * (1 - (1 - t) ** power) + v * t

    ``v = END_PX_PER_MS * duration / distance`` clamps the terminal speed to ~1 px per 60 Hz
    frame no matter how long the step is, so the settle decelerates all the way down and then
    simply stops, instead of freezing and twitching.
    """

    END_PX_PER_MS = 1.0 / 16.7     # fastest terminal crawl we can show: one pixel per frame

    def __init__(self, bar, parent=None):
        super().__init__(parent)
        self._bar = bar
        self._pos = 0.0
        self._mute = False
        self.setEasingCurve(QEasingCurve.Type.Linear)
        self.valueChanged.connect(self._apply)

    # -- public ---------------------------------------------------------
    @property
    def pos(self) -> float:
        """Where the bar is, including the fraction the scrollbar itself cannot hold."""
        return self._pos

    def move_to(self, start: float, target: float, duration: int, power: int = 4) -> None:
        start = float(start)
        target = float(target)
        # Rewriting the key values makes QVariantAnimation emit valueChanged for the *old* time
        # against the new keys (and stop() emits one for the old keys).  Those blips are not
        # positions the animation ever plays: they read as a 40px jump backwards for one frame,
        # so they must not reach the scrollbar.
        self._mute = True
        try:
            self.stop()
            if abs(target - start) < 0.75:
                self._pos = target
                self._bar.setValue(int(round(target)))
                return
            dist = abs(target - start)
            linear = min(0.5, self.END_PX_PER_MS * duration / dist)
            keys = []
            frames = max(2, int(round(duration / 16.7)))
            for i in range(frames + 1):
                t = i / frames
                f = (1.0 - linear) * (1.0 - (1.0 - t) ** power) + linear * t
                keys.append((t, start + (target - start) * f))
            self.setKeyValues(keys)
            self.setDuration(int(duration))
            self._pos = start
            self._bar.setValue(int(round(start)))
        finally:
            self._mute = False
        self.start()

    def _apply(self, value) -> None:
        # Only a running animation produces real positions.  `stop()` and `setKeyValues()` each
        # leave a queued update that arrives a moment later, already stopped, computed from a
        # half-built key table -- writing that would jerk the bar backwards for one frame.
        if self._mute or self.state() != QAbstractAnimation.State.Running:
            return
        self._pos = float(value)
        self._bar.setValue(int(round(self._pos)))


class SmoothScrollArea(QScrollArea):
    """QScrollArea enhanced with:
    1. Physics-based smooth momentum scrolling with natural inertia decay (惯性滑动收尾).
    2. Adaptive velocity calibration to prevent vertical lists from scrolling too fast.
    3. Seamless interruption on opposite-direction scroll, viewport click, or manual scrollbar grab.
    4. Subtle top/bottom or left/right edge fade indicators showing scroll limits.
    """
    def __init__(self, parent=None, is_horizontal=False, enable_edge_fade=None):
        super().__init__(parent)
        self.is_horizontal = is_horizontal
        self.enable_edge_fade = (not is_horizontal) if enable_edge_fade is None else enable_edge_fade
        self.setWidgetResizable(True)

        # Distance per wheel notch.  Vertical lists: 76px, horizontal filmstrip: 120px.
        self._step_size = 120.0 if self.is_horizontal else 76.0

        # Dynamic state.  The physics integrator is gone; these stay because the overlay,
        # the resize handler and scroll_to all read them.
        self._velocity = 0.0
        self._current_pos = 0.0
        self._is_internal_scroll = False
        self._last_wheel_time = 0.0

        sb0 = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        # Both drivers write the bar in whole pixels from a float position (see
        # _PixelSettleAnimation: an integer-valued animation freezes for the last pixel).
        self._anim = _PixelSettleAnimation(sb0, self)
        self._anim.finished.connect(self._on_anim_finished)

        # Wheel scrolling is a retargetable eased animation, not a velocity integrator.
        # Exponential friction decay has a near-linear tail and its reach depends on how fast
        # the notches arrive.  Aiming at a target with OutQuart instead gives the iOS feel --
        # a quick start and a long, gentle settle -- and extra notches just push the target
        # further out while the running animation retargets from wherever it is.
        self._wheel_anim = _PixelSettleAnimation(sb0, self)
        self._wheel_anim.finished.connect(self._on_anim_finished)
        self._wheel_target = 0
        self._wheel_power = 4

        # Connect scrollbar signals to maintain sync with manual interactions
        v_bar = self.verticalScrollBar()
        h_bar = self.horizontalScrollBar()
        if v_bar:
            v_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        if h_bar:
            h_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        v_bar.valueChanged.connect(self._on_scroll_value_changed)
        h_bar.valueChanged.connect(self._on_scroll_value_changed)
        v_bar.rangeChanged.connect(lambda min_val, max_val: self._sync_overlay())
        h_bar.rangeChanged.connect(lambda min_val, max_val: self._sync_overlay())
        v_bar.sliderPressed.connect(self._stop_physics)
        h_bar.sliderPressed.connect(self._stop_physics)

        # Install event filter on viewport to capture mouse press and wheel
        if self.viewport():
            self.viewport().installEventFilter(self)

        # Overlay widget on top of viewport child widgets (only when edge fade is enabled)
        if self.enable_edge_fade:
            self._overlay = _ScrollEdgeOverlay(self)
        else:
            self._overlay = None

        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

        # A touch panel has no wheel: a finger drag has to scroll, and still click on a tap.
        install_touch_scroll(self)

    def eventFilter(self, obj, event):
        if obj == self.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                self._stop_physics()
                if self._anim.state() == QAbstractAnimation.State.Running:
                    self._anim.stop()
            elif event.type() == QEvent.Type.Wheel:
                self.wheelEvent(event)
                return True
        return super().eventFilter(obj, event)

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
        overlay = getattr(self, '_overlay', None)
        if overlay is None:
            return
        # This runs on every scrollbar tick.  Re-setting the geometry and re-raising the overlay
        # (and the scrollbar) sixty times a second during a settle made the edge fade flicker and
        # cost a relayout per frame -- the last few pixels of a wheel animation looked like the
        # panel twitched.  Only touch the stacking when the viewport actually changed size.
        target = self.rect() if self.is_horizontal else self.viewport().rect()
        if getattr(self, "_overlay_geo", None) != target:
            self._overlay_geo = target
            overlay.setGeometry(target)
            if self.is_horizontal:
                overlay.raise_()
                hbar = self.horizontalScrollBar()
                if hbar:
                    hbar.raise_()
            else:
                overlay.raise_()
        overlay.update()

    def _is_animating(self) -> bool:
        return (self._anim.state() == QAbstractAnimation.State.Running
                or self._wheel_anim.state() == QAbstractAnimation.State.Running)

    def _on_scroll_value_changed(self, val):
        # While one of our own animations drives the bar the value change is not user input:
        # stopping the animation here is what used to make `scroll_to`, and then the eased
        # wheel, die after a single frame.
        if not self._is_internal_scroll and not self._is_animating():
            self._current_pos = float(val)
            self._stop_physics()
        self._sync_overlay()

    def _stop_physics(self):
        """Stop any in-flight scroll motion (wheel easing or programmatic animation)."""
        self._velocity = 0.0
        if getattr(self, "_wheel_anim", None) is not None:
            self._wheel_anim.stop()
        if getattr(self, "_anim", None) is not None:
            self._anim.stop()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if sb:
            self._current_pos = float(sb.value())
        self._sync_overlay()

    def wheelEvent(self, event):
        """Move the scrollbar with a retargetable eased animation (iOS-style settle)."""
        # Cancel any delayed filmstrip or layout alignment timers up the parent hierarchy
        p = self.parent()
        while p is not None:
            if hasattr(p, "_cancel_pending_scroll_timers"):
                try:
                    p._cancel_pending_scroll_timers()
                except Exception:
                    pass
                break
            p = p.parent()

        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if not sb or sb.maximum() <= sb.minimum():
            super().wheelEvent(event)
            return

        # Determine scroll delta
        if self.is_horizontal:
            ad_x = event.angleDelta().x()
            ad_y = event.angleDelta().y()
            delta = ad_x if abs(ad_x) > abs(ad_y) else ad_y
        else:
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.angleDelta().x()

        if delta == 0:
            super().wheelEvent(event)
            return

        min_val = float(sb.minimum())
        max_val = float(sb.maximum())
        current_sb_val = float(sb.value())

        # delta > 0 is scroll up / left (decrease position)
        # delta < 0 is scroll down / right (increase position)
        ticks = -(delta / 120.0)

        # If already at edge and trying to push beyond, ignore
        if (current_sb_val <= min_val and ticks < 0) or (current_sb_val >= max_val and ticks > 0):
            event.accept()
            return

        # Retarget from where the previous notch was heading, so a burst of notches stacks
        # instead of restarting from the current pixel every time.  Starting from the animation's
        # own float position (not the rounded bar value) keeps a retarget from looking like a
        # one-pixel jump.
        running = self._wheel_anim.state() == QAbstractAnimation.State.Running
        if running:
            base = float(self._wheel_target)
            start_pos = self._wheel_anim.pos
        else:
            base = current_sb_val
            start_pos = current_sb_val

        self._anim.stop()
        target = max(min_val, min(max_val, base + ticks * self._step_size))
        new_target = int(round(target))
        # Already on the way to exactly this position: restarting would drop the velocity back to
        # zero and re-accelerate, so a burst of notches arrived as a stutter instead of one long
        # settle.  This is also the case that produced the "twitch" right before the bottom.
        if running and new_target == self._wheel_target:
            event.accept()
            return
        self._wheel_target = new_target

        dist = abs(self._wheel_target - start_pos)
        if dist <= 0:
            event.accept()
            return
        # Longer for a bigger step, but never a snap: one 76px notch lands at ~310ms and the
        # OutQuart tail keeps the last few pixels crawling in.
        duration = int(min(620, max(220, 160 + dist * 2.0)))
        self._wheel_anim.move_to(start_pos, float(self._wheel_target), duration,
                                 power=self._wheel_power)
        self._current_pos = start_pos
        event.accept()

    def scroll_to(self, target_value: int, animate: bool = True):
        """Scroll to target position either with OutCubic easing or immediately."""
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if not sb:
            return
        target = max(sb.minimum(), min(sb.maximum(), int(target_value)))
        self._stop_physics()

        if not animate:
            if self._anim.state() == QAbstractAnimation.State.Running:
                self._anim.stop()
            self._current_pos = float(target)
            self._is_internal_scroll = True
            sb.setValue(target)
            self._is_internal_scroll = False
            self._sync_overlay()
            return

        current_val = sb.value()
        if current_val == target:
            return

        self._current_pos = float(current_val)
        self._anim.move_to(float(current_val), float(target), 220, power=3)

    def _on_anim_finished(self):
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if sb:
            self._current_pos = float(sb.value())
        self._sync_overlay()


class _ScrollEdgeOverlay(QWidget):
    """Transparent overlay widget floating above all viewport children to draw subtle edge fade shadows."""
    def __init__(self, parent_scroll_area: SmoothScrollArea):
        parent_w = parent_scroll_area if getattr(parent_scroll_area, 'is_horizontal', False) else parent_scroll_area.viewport()
        super().__init__(parent_w)
        self.scroll_area = parent_scroll_area
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        if not getattr(self.scroll_area, 'enable_edge_fade', True):
            return
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
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
            fade_w = 26.0
            if sb.maximum() > 0:
                from PySide6.QtGui import QPainterPath, QImage
                fw = int(fade_w)
                fh = max(1, int(h))
                cache_key = (fw, fh, is_dark)
                if getattr(self, "_h_fade_cache_key", None) != cache_key:
                    self._h_fade_cache_key = cache_key
                    left_img = QImage(fw, fh, QImage.Format.Format_ARGB32_Premultiplied)
                    left_img.fill(Qt.GlobalColor.transparent)
                    img_p = QPainter(left_img)
                    v_grad = QLinearGradient(0.0, 0.0, 0.0, float(fh))
                    if is_dark:
                        v_grad.setColorAt(0.0, QColor(6, 7, 10, 255))
                        v_grad.setColorAt(0.18, QColor(8, 9, 13, 255))
                        v_grad.setColorAt(0.85, QColor(10, 11, 16, 255))
                        v_grad.setColorAt(1.0, QColor(13, 14, 20, 255))
                    else:
                        v_grad.setColorAt(0.0, QColor(214, 219, 228, 255))
                        v_grad.setColorAt(0.18, QColor(224, 228, 236, 255))
                        v_grad.setColorAt(0.85, QColor(234, 238, 245, 255))
                        v_grad.setColorAt(1.0, QColor(242, 245, 250, 255))
                    img_p.fillRect(0, 0, fw, fh, v_grad)
                    img_p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                    h_mask = QLinearGradient(0.0, 0.0, float(fw), 0.0)
                    h_mask.setColorAt(0.0, QColor(0, 0, 0, 255))
                    h_mask.setColorAt(0.18, QColor(0, 0, 0, 235))
                    h_mask.setColorAt(0.55, QColor(0, 0, 0, 115))
                    h_mask.setColorAt(1.0, QColor(0, 0, 0, 0))
                    img_p.fillRect(0, 0, fw, fh, h_mask)
                    img_p.end()
                    self._left_fade_img = left_img
                    self._right_fade_img = left_img.mirrored(True, False)

                clip_path = QPainterPath()
                clip_path.addRoundedRect(QRectF(0.0, 0.0, float(w), float(h)), 4.5, 4.5)
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                painter.setClipPath(clip_path)

                if sb.value() > 0:
                    painter.drawImage(QRectF(0.0, 0.0, fade_w, float(h)), self._left_fade_img)
                if sb.value() < sb.maximum():
                    painter.drawImage(QRectF(float(w) - fade_w, 0.0, fade_w, float(h)), self._right_fade_img)

                painter.restore()
        painter.end()
