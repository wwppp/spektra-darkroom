"""smooth_scroll.py - Universal smooth scrolling with edge indicators.

Wheel motion is driven by ``WheelFollower``: each notch moves a *target* position and one
shared exponential glide carries the bar there -- full speed on the first frame, then a
long slow tail -- so speed is continuous across notches instead of restarting per notch.
See the class docstring for the measurements behind that.
"""

import math
from PySide6.QtCore import (Qt, QAbstractAnimation, QEasingCurve, QElapsedTimer, QObject,
                            QRectF, QTimer, QEvent, QVariantAnimation, Signal)
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QPainterPath, QBrush
from PySide6.QtWidgets import QScrollArea, QWidget

from ui.theme_manager import theme_manager
from ui.touch_scroll import install_touch_scroll


# The wheel curve, shared by everything a wheel drives (panel scroll, dropdown lists, and the
# canvas zoom -- which runs the same arithmetic on log-zoom).  One source of truth on purpose:
# two copies of "the feel" drift apart the first time either is tuned.
GLIDE_TAU = 0.14          # s; speed is (distance to target) / TAU, no acceleration limit
GLIDE_FRAME_MS = 8        # integration frame; twice the display rate

# The largest share of the *remaining* distance a single painted frame may consume.
#
# The curve is time-based, so the share per frame is set by the frame rate: 11 % at 60 fps,
# 26 % at 25 fps.  The panel scroll repaints at 125 fps and looks continuous; the canvas only
# manages ~22 fps (each repaint costs 35-60 ms inside Qt's OpenGL-widget composite -- our own
# shader is 1.5 ms of that, measured), and there the same curve hands the first frame a quarter
# of the zoom in one 44 ms step and then a crawl: the "twitch" reported for wheel zoom.
#
# 0.24 is one 25 fps frame's share: a surface painting at 25 fps or better is untouched, and a
# slower one gets even steps instead of a front-loaded jump.  It costs time -- a notch reaches
# 90 % at ~0.5 s instead of ~0.33 s -- because frames are the only unit the surface gives us.
GLIDE_MAX_FRAME_FRAC = 0.24


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
    """Carry a scrollbar to the wheel's target as an exponential glide.

    Wanted shape: full speed within one frame, then a long, slow bleed to a stop.

    The three earlier drivers all failed at the *start*, because each derived its speed
    from a position curve instead of setting it:

    * an OutQuart ease restarted per notch opened with a 4x-average lurch and read as a
      sawtooth under a burst (11, 9, 8, 13, 8, 9, 8, 14 px per frame at 110 ms spacing);
    * a critically damped spring pumped at the notch rhythm (6..13 px/frame ripple);
    * a ``sqrt(2 * ACCEL * distance)`` speed servo had no seam but was acceleration
      limited, and one 76 px notch needs only 160 ms at 6000 px/s^2 -- so the *whole*
      flight was spent accelerating: the first frame moved 0.2 px, half the distance went
      in the last two frames.  Smooth arithmetic, but it reads as "nothing, nothing, then
      it jumps", which is exactly the sluggish start this replaces.

    So the speed is *read* from the distance and applied immediately, ``v = d / TAU``,
    with no acceleration limit and no braking curve:

    * one notch is at ~8.7 px/frame in the very first frame (measured: 8.7 px of the 76 px
      step lands in frame one, against 0.2 px for the accel-limited servo), and ``add()``
      advances that first frame before returning, so the paint caused by the wheel event
      already shows the motion;
    * the tail is the same 140 ms time constant decaying: 50 % of the way at 100 ms, 90 %
      at 333 ms, still creeping at 500 ms, snapped shut only inside the last sub-pixel;
    * distance still sums, so five notches travel five steps whatever the cadence, and a
      faster/larger wheel delta simply means a longer distance ahead -- which *is* the
      speed, so force maps to speed with no extra knob;
    * the approach is monotone, so nothing overshoots and nothing rebounds.
    """

    TAU = GLIDE_TAU        # s, the decay time constant of ``v = d / TAU``
    V_MAX = 2400.0        # px/s ceiling, so a violent spin cannot turn into a teleport
    SUBSTEP_S = 0.008     # integration step; keeps the profile stable if a frame is long
    FRAME_MS = GLIDE_FRAME_MS  # twice the display rate, so a frame never sees a stale position
    SNAP_PX = 0.75        # sub-pixel: at this distance the bar has already rounded still
    #: A reverse gesture has to displace the target this far before it takes over the flight
    #: that is still in the air.  Below one notch (76 px) by a wide margin, so a real reversal
    #: feels immediate, but far above the 1-3 px of finger jitter a hard spin produces --
    #: those must never cancel the glide the user just asked for.
    REVERSE_TAKEOVER_PX = 12.0
    #: ...and only when there is a real flight left to cancel.  A glider that is already
    #: within a notch of its target just lands; taking over there would turn a quick
    #: forward/back pair into "it jumped an extra notch".
    REVERSE_MIN_REMAINING_PX = 60.0

    finished = Signal()

    def __init__(self, bar, parent=None):
        super().__init__(parent)
        self._bar = bar
        self._pos = float(bar.value())
        self._vel = 0.0
        self._target = float(bar.value())
        # The integer we last wrote into the bar.  Anything else in there means somebody
        # else moved it (see `_advance`); ``None`` means "unknown, don't judge yet".
        self._bar_written = None
        # Distance the user has pushed *against* the flight still in the air; see `add`.
        self._reverse_px = 0.0
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
        self._bar_written = None
        self._reverse_px = 0.0

    def add(self, delta: float) -> None:
        """Push the target by ``delta`` px.  Repeated calls accumulate."""
        lo = float(self._bar.minimum())
        hi = float(self._bar.maximum())
        if hi <= lo:
            return
        first_event = not self._timer.isActive()
        if first_event:
            # Start from wherever the bar actually is: the user may have dragged it, or a
            # previous flight may have been interrupted.
            self._pos = float(self._bar.value())
            self._vel = 0.0
            self._target = self._pos
            self._bar_written = int(round(self._pos))
            self._clock.start()
            self._timer.start()

        # A spin that is still on its way somewhere and a wheel turned the other way: the old
        # distance used to have to be paid back first, so a hard flick read as "it will not
        # reverse until it stops on its own".  A reverse gesture takes over instead -- the
        # unflown distance is dropped and the new direction starts from where the bar is now.
        #
        # This deliberately does *not* consult QWheelEvent.phase(): on Windows it is
        # NoScrollPhase for touchpads too (only macOS/Wayland report ScrollBegin/Update), so
        # keying the behaviour on it meant the takeover never fired for the very user this
        # was written for.
        remaining = self._target - self._pos
        if delta and remaining and (delta > 0.0) != (remaining > 0.0):
            self._reverse_px += abs(delta)
            if (self._reverse_px >= self.REVERSE_TAKEOVER_PX
                    and abs(remaining) >= self.REVERSE_MIN_REMAINING_PX):
                self._target = self._pos
                self._reverse_px = 0.0
        else:
            self._reverse_px = 0.0
        self._target = max(lo, min(hi, self._target + delta))
        # Spend the time already waited, synchronously.  The wheel event and the repaint it
        # triggers are the same frame, so waiting for the first timer tick would show a
        # still bar on the frame the user acted on -- that 8 ms of nothing is half of what
        # "the initial response lags" measures.  The time is *claimed*, not invented: the
        # next step (timer or wheel) only gets what has really elapsed since this one, so a
        # rapid high-resolution wheel cannot out-run the clock.
        self._advance(self._claim_time(first_event))

    # -- internals ------------------------------------------------------
    def _claim_time(self, first_event: bool = False) -> float:
        """Seconds of real time since the previous step (one frame at the start of a flight)."""
        if first_event or not self._clock.isValid():
            return self.FRAME_MS / 1000.0
        return min(0.05, self._clock.restart() / 1000.0)

    def _write_bar(self) -> None:
        """Write our float position into the integer scrollbar, remembering what we wrote."""
        value = int(round(self._pos))
        self._bar_written = value
        self._bar.setValue(value)

    def _tick(self) -> None:
        self._advance(self._claim_time())

    def _advance(self, dt: float) -> bool:
        """Move for ``dt`` seconds.  Returns True while there is motion left."""
        # Somebody else wrote the bar mid-flight: touch inertia (16 ms steps), the export
        # queue restoring its position after a card reorder, `ensureVisible` from a focus
        # change, or Qt clamping the value when the range shrank.  Those writers own the
        # position now.  Carrying on would put the bar back on our own `_pos` on the very
        # next frame and both sides would pull at each other every 8-16 ms -- which is the
        # panel that "twitches" until the user touches something else.  Standing down costs
        # at most the rest of one glide; fighting costs a visible shake.
        if self._bar_written is not None and self._bar.value() != self._bar_written:
            self.reset()
            return False

        lo = float(self._bar.minimum())
        hi = float(self._bar.maximum())
        # The range can collapse under a flight -- a panel section folds, a dropdown is
        # repopulated, a stock swap resizes the sidebar.  Re-clamping the target keeps the
        # follower from chasing a position that no longer exists (it would sit at the
        # clamp with the snap test never coming true, i.e. spin forever).
        self._target = max(lo, min(hi, self._target))

        steps = max(1, int(math.ceil(dt / self.SUBSTEP_S)))
        h = dt / steps
        rate = 1.0 / self.TAU
        for _ in range(steps):
            d = self._target - self._pos
            if d == 0.0:
                self._vel = 0.0
                break
            v = d * rate
            if v > self.V_MAX:
                v = self.V_MAX
            elif v < -self.V_MAX:
                v = -self.V_MAX
            # Only reachable once the exponential has flattened out; it stops the last
            # float crumbs from spinning the timer forever.
            if abs(v * h) >= abs(d):
                self._pos = self._target
                self._vel = 0.0
                break
            self._vel = v
            self._pos += v * h

        if self._pos <= lo:
            self._pos = lo
            self._vel = 0.0
        elif self._pos >= hi:
            self._pos = hi
            self._vel = 0.0

        if abs(self._target - self._pos) < self.SNAP_PX:
            self._pos = self._target
            self._vel = 0.0
            self._timer.stop()
            self._write_bar()
            self.finished.emit()
            return False

        self._write_bar()
        return True


class SmoothScrollArea(QScrollArea):
    """QScrollArea with wheel motion carried by a ``WheelFollower`` and edge fades.

    The wheel is not handled per notch any more: each notch moves a target, and one
    shared ``WheelFollower`` glide animates the bar toward it, interrupts on click or
    scrollbar grab, and keeps the fade overlay in sync.  ``scroll_to`` keeps its own
    settle animation for programmatic jumps, where the distance is already known.
    """
    def __init__(self, parent=None, is_horizontal=False, enable_edge_fade=None):
        super().__init__(parent)
        self.is_horizontal = is_horizontal
        self.enable_edge_fade = (not is_horizontal) if enable_edge_fade is None else enable_edge_fade
        # 渐隐带的底色。默认 None = 按主题取 panel_bg；卡片墙这类"面板里还有一层托盘"的
        # 场景必须显式设成托盘色，否则渐隐带会和真实背景差好几个色阶，看着像一块白雾。
        self._fade_base_color = None
        self._fade_height = 16
        # 渐隐带左右内缩 + 圆角：内容卡片比 viewport 窄时（右侧留滚动条缝、卡片圆角），
        # 满宽画出来会糊出卡片边缘，变成一条压在留白上的白带。
        self._fade_inset_l = 0
        self._fade_inset_r = 0
        self._fade_radius = 0.0
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

        # The wheel drives a target position through an exponential glide rather than a
        # replayed ease-out curve; the class docstring has the measurements behind that.
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

    def set_fade_base_color(self, color, height=None):
        """指定边缘渐隐带的底色（默认按主题取 panel_bg）。

        渐隐是"把内容淡出到底色"，底色必须等于控件背后真实的填充色。卡片墙外面套了
        托盘之后，背后是 subpanel_bg 而不是 panel_bg；不设这个，浅色下渐隐带会比背景
        亮 6-7 级，看上去是一块发白的雾压在卡片上。
        """
        self._fade_base_color = QColor(color) if color is not None else None
        if height is not None:
            self._fade_height = max(4, int(height))
        self._sync_overlay()

    def set_fade_geometry(self, inset_left=0, inset_right=0, radius=0.0):
        """把渐隐带限制在卡片宽度内（inset 是左右内缩，radius 对齐卡片圆角）。"""
        self._fade_inset_l = max(0, int(inset_left))
        self._fade_inset_r = max(0, int(inset_right))
        self._fade_radius = max(0.0, float(radius))
        self._sync_overlay()

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
            _track = theme_manager.color("subpanel_bg")
            _qss = """
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:vertical {
                    background: __TRACK__;
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
            """
            self.setStyleSheet(_qss.replace("__TRACK__", _track))
        else:
            _track = theme_manager.color("subpanel_bg")
            _qss = """
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:vertical {
                    background: __TRACK__;
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
            """
            self.setStyleSheet(_qss.replace("__TRACK__", _track))
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

        # The fade is a two-state indicator per axis: it only appears when the bar is off an
        # edge.  Repainting on every tick meant ~125 full-viewport composites per second for
        # an image that is almost always identical.  Compare the only inputs `paintEvent`
        # reads (size, range, both edge predicates, theme, enable flag) and repaint on change.
        bar = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if bar is None:
            overlay.update()
            return
        state = (
            target.width(), target.height(),
            bar.minimum(), bar.maximum(),
            bar.value() > 0, bar.value() < bar.maximum(),
            bool(getattr(self, "enable_edge_fade", True)),
            theme_manager.is_dark(),
        )
        if getattr(self, "_overlay_state", None) != state:
            self._overlay_state = state
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
        """Add the notch to the wheel target; the glide carries the bar there."""
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
        # difference is simply zero and the glide has nothing to do.
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
        base = getattr(self.scroll_area, "_fade_base_color", None)
        if base is not None:
            fade_col = QColor(base)
            fade_col.setAlpha(220)
            trans_col = QColor(base)
            trans_col.setAlpha(0)
        elif is_dark:
            fade_col = QColor(18, 19, 24, 210)
            trans_col = QColor(18, 19, 24, 0)
        else:
            bg_hex = theme_manager.color("panel_bg")
            base = QColor(bg_hex)
            fade_col = QColor(base.red(), base.green(), base.blue(), 220)
            trans_col = QColor(base.red(), base.green(), base.blue(), 0)

        if not self.scroll_area.is_horizontal:
            sb = self.scroll_area.verticalScrollBar()
            fh = max(4, int(getattr(self.scroll_area, "_fade_height", 16)))
            li = max(0, int(getattr(self.scroll_area, "_fade_inset_l", 0)))
            ri = max(0, int(getattr(self.scroll_area, "_fade_inset_r", 0)))
            band_w = max(1, w - li - ri)
            radius = min(float(getattr(self.scroll_area, "_fade_radius", 0.0)), band_w / 2.0, fh / 2.0)
            if sb.maximum() > 0:
                # Top edge indicator: if scrolled down from 0
                if sb.value() > 0:
                    grad_top = QLinearGradient(0, 0, 0, fh)
                    grad_top.setColorAt(0.0, fade_col)
                    grad_top.setColorAt(1.0, trans_col)
                    if radius > 0:
                        path = QPainterPath()
                        path.addRoundedRect(QRectF(li, 0, band_w, fh), radius, radius)
                        painter.fillPath(path, QBrush(grad_top))
                    else:
                        painter.fillRect(QRectF(li, 0, band_w, fh), grad_top)

                # Bottom edge indicator: if not yet at maximum
                if sb.value() < sb.maximum():
                    grad_bottom = QLinearGradient(0, h - fh, 0, h)
                    grad_bottom.setColorAt(0.0, trans_col)
                    grad_bottom.setColorAt(1.0, fade_col)
                    if radius > 0:
                        path = QPainterPath()
                        path.addRoundedRect(QRectF(li, h - fh, band_w, fh), radius, radius)
                        painter.fillPath(path, QBrush(grad_bottom))
                    else:
                        painter.fillRect(QRectF(li, h - fh, band_w, fh), grad_bottom)
        else:
            sb = self.scroll_area.horizontalScrollBar()
            fade_w = 26.0
            if sb.maximum() > 0:
                from PySide6.QtGui import QImage
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
