"""smooth_scroll.py - Universal smooth scrolling with edge indicators.

Wheel motion is driven by ``WheelFollower``: each notch moves a *target* position and one
shared velocity servo carries the bar there, so speed is continuous across notches instead
of restarting per notch.  See the class docstring for the measurements behind that.
"""

import math
from PySide6.QtCore import (Qt, QAbstractAnimation, QEasingCurve, QElapsedTimer, QObject,
                            QRectF, QTimer, QEvent, QVariantAnimation, Signal)
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


class WheelFollower(QObject):
    """Carry a scrollbar to the wheel's target with continuous velocity.

    The previous driver restarted an OutQuart ease from the current pixel on every
    notch.  OutQuart leaves its start at four times the average slope, so a 76 px notch
    opened with a 15 px lurch in a single frame and decayed from there; a second notch
    mid-flight restarted that lurch, and a burst read as a sawtooth (measured: 11, 9, 8,
    13, 8, 9, 8, 14 px per frame at 110 ms spacing).  Only the position was continuous,
    never the speed -- which is what "one pulse, one twitch" describes.

    This is a velocity servo instead of a replayed curve.  The wheel adds to a *target*
    and the speed chases ``sqrt(2 * ACCEL * distance)`` -- the profile that can still
    brake to a stop exactly at the target -- with the speed itself rate-limited.  So:

    * five notches still travel five steps however fast they arrive, unlike a momentum
      integrator whose reach depends on cadence;
    * a retarget changes the *acceleration*, never the speed, so there is no seam: a
      burst ramps up monotonically and then holds, rather than pulsing per notch;
    * arrival is exact and the last fraction of a pixel is snapped, so there is no
      constant-speed crawl at the end (the old driver spent ~150 ms at 1 px per frame)
      and nothing to rebound.

    A critically damped spring was tried first: also seamless, but a second-order system
    driven at the notch rhythm pumps, which put a 6..13 px/frame ripple back into a
    110 ms burst.  The servo does not resonate because its speed target is a function of
    distance, not of history.
    """

    ACCEL = 6000.0        # px/s^2, and also the braking rate
    V_MAX = 2000.0        # px/s ceiling, so a violent spin cannot turn into a teleport
    SUBSTEP_S = 0.008     # integration step; keeps the profile stable if a frame is long
    FRAME_MS = 8          # twice the display rate, so a frame never sees a stale position
    MARGIN_PX = 0.02      # keep sqrt() off zero; the landing rule below does the real work
    SNAP_PX = 0.7
    SNAP_V = 90.0         # px/s

    finished = Signal()

    def __init__(self, bar, parent=None):
        super().__init__(parent)
        self._bar = bar
        self._pos = float(bar.value())
        self._vel = 0.0
        self._target = float(bar.value())
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # -- public ---------------------------------------------------------
    @property
    def pos(self) -> float:
        return self._pos

    @property
    def speed(self) -> float:
        """Current speed in px/s (diagnostics)."""
        return self._vel

    def is_running(self) -> bool:
        return self._timer.isActive()

    def reset(self, value=None) -> None:
        """Forget any motion and adopt ``value`` (or the bar's current value)."""
        self._timer.stop()
        self._vel = 0.0
        self._pos = float(self._bar.value() if value is None else value)
        self._target = self._pos

    def add(self, delta: float) -> None:
        """Push the target by ``delta`` px.  Repeated calls accumulate."""
        lo = float(self._bar.minimum())
        hi = float(self._bar.maximum())
        if hi <= lo:
            return
        if not self._timer.isActive():
            # Start from wherever the bar actually is: the user may have dragged it, or a
            # previous flight may have been interrupted.
            self._pos = float(self._bar.value())
            self._vel = 0.0
            self._target = self._pos
            self._clock.start()
            self._timer.start()
        self._target = max(lo, min(hi, self._target + delta))

    # -- internals ------------------------------------------------------
    def _tick(self) -> None:
        dt = (self._clock.restart() / 1000.0) if self._clock.isValid() else self.FRAME_MS / 1000.0
        dt = min(0.05, max(1e-4, dt))

        lo = float(self._bar.minimum())
        hi = float(self._bar.maximum())
        # The range can collapse under a flight -- a panel section folds, a dropdown is
        # repopulated, a stock swap resizes the sidebar.  Re-clamping the target keeps the
        # follower from chasing a position that no longer exists (it would sit at the
        # clamp with the snap test never coming true, i.e. spin forever).
        self._target = max(lo, min(hi, self._target))

        steps = max(1, int(math.ceil(dt / self.SUBSTEP_S)))
        h = dt / steps
        lim = self.ACCEL * h
        for _ in range(steps):
            d = self._target - self._pos
            if d == 0.0:
                self._vel = 0.0
                break
            # Land exactly instead of coasting past: the braking curve converges on the
            # target from above, and without this the last substep would step over it.
            if self._vel * d > 0.0 and abs(self._vel * h) >= abs(d):
                self._pos = self._target
                self._vel = 0.0
                break
            v_des = min(self.V_MAX,
                        math.sqrt(2.0 * self.ACCEL
                                  * max(0.0, abs(d) - self.MARGIN_PX)))
            v_des = math.copysign(v_des, d)
            dv = v_des - self._vel
            self._vel += lim if dv > lim else (-lim if dv < -lim else dv)
            self._pos += self._vel * h

        if self._pos <= lo:
            self._pos = lo
            self._vel = 0.0
        elif self._pos >= hi:
            self._pos = hi
            self._vel = 0.0

        if abs(self._target - self._pos) < self.SNAP_PX and abs(self._vel) < self.SNAP_V:
            self._pos = self._target
            self._vel = 0.0
            self._timer.stop()
            self._bar.setValue(int(round(self._pos)))
            self.finished.emit()
            return

        self._bar.setValue(int(round(self._pos)))


class SmoothScrollArea(QScrollArea):
    """QScrollArea with wheel motion carried by a ``WheelFollower`` and edge fades.

    The wheel is not handled per notch any more: each notch moves a target, and one
    shared servo (`WheelFollower`) animates the bar toward it, interrupts on click or
    scrollbar grab, and keeps the fade overlay in sync.  ``scroll_to`` keeps its own
    settle animation for programmatic jumps, where the distance is already known.
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

        # The wheel drives a target position through a spring rather than a replayed
        # ease-out curve; the class docstring has the measurements behind that.
        self._wheel = WheelFollower(sb0, self)
        self._wheel.finished.connect(self._on_anim_finished)

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
                or self._wheel.is_running())

    def _on_scroll_value_changed(self, val):
        # While one of our own animations drives the bar the value change is not user input:
        # stopping the animation here is what used to make `scroll_to`, and then the eased
        # wheel, die after a single frame.
        if not self._is_internal_scroll and not self._is_animating():
            self._current_pos = float(val)
            self._stop_physics()
        self._sync_overlay()

    def _stop_physics(self):
        """Stop any in-flight scroll motion (wheel follow or programmatic animation)."""
        self._velocity = 0.0
        if getattr(self, "_wheel", None) is not None:
            self._wheel.reset()
        if getattr(self, "_anim", None) is not None:
            self._anim.stop()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if sb:
            self._current_pos = float(sb.value())
        self._sync_overlay()

    def wheelEvent(self, event):
        """Add the notch to the wheel target; the spring carries the bar there."""
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

        # delta > 0 is scroll up / left (decrease position)
        # delta < 0 is scroll down / right (increase position)
        ticks = -(delta / 120.0)

        # The target is a running sum, so a notch is never lost to a clamp: at the edge the
        # difference is simply zero and the spring has nothing to do.
        self._anim.stop()
        self._wheel.add(ticks * self._step_size)
        self._current_pos = self._wheel.pos
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
