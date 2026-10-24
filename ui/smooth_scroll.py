"""smooth_scroll.py - Universal smooth easing scroll area with physics-based momentum and edge indicators
Provides silky smooth kinetic momentum scrolling (natural friction decay) and subtle overscroll edge cues.
"""

import time
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRectF, QTimer, QEvent
from PySide6.QtGui import QPainter, QColor, QLinearGradient
from PySide6.QtWidgets import QScrollArea, QWidget

from ui.theme_manager import theme_manager


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

        # Physics simulation tuning
        # Vertical lists: 38px/notch (fine control, comfortable speed)
        # Horizontal filmstrip: 60px/notch (natural thumbnail navigation)
        self._step_size = 60.0 if self.is_horizontal else 38.0
        self._friction = 0.91 if self.is_horizontal else 0.895
        self._max_velocity = 55.0 if self.is_horizontal else 40.0
        self._threshold = 0.30

        # Dynamic physics state
        self._velocity = 0.0
        self._current_pos = 0.0
        self._is_internal_scroll = False
        self._last_wheel_time = 0.0

        # 60 FPS physics integration timer (16ms)
        self._physics_timer = QTimer(self)
        self._physics_timer.setInterval(16)
        self._physics_timer.timeout.connect(self._on_physics_tick)

        # Programmatic animation on scrollbar (used by scroll_to)
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        self._anim = QPropertyAnimation(sb, b"value", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._on_anim_finished)

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

    def eventFilter(self, obj, event):
        if obj == self.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                self._stop_physics()
                if self._anim.state() == QPropertyAnimation.State.Running:
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
        if getattr(self, '_overlay', None) is not None:
            if self.is_horizontal:
                self._overlay.setGeometry(self.rect())
                self._overlay.raise_()
                hbar = self.horizontalScrollBar()
                if hbar:
                    hbar.raise_()
            else:
                vp = self.viewport()
                if vp:
                    self._overlay.setGeometry(vp.rect())
                    self._overlay.raise_()
            self._overlay.update()

    def _on_scroll_value_changed(self, val):
        if not self._is_internal_scroll:
            # Value changed externally (manual thumb drag, track click, keyboard)
            self._current_pos = float(val)
            self._stop_physics()
            if self._anim.state() == QPropertyAnimation.State.Running:
                self._anim.stop()
        self._sync_overlay()

    def _stop_physics(self):
        if hasattr(self, '_physics_timer') and self._physics_timer.isActive():
            self._physics_timer.stop()
        self._velocity = 0.0

    def _on_physics_tick(self):
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if not sb or sb.maximum() <= sb.minimum():
            self._stop_physics()
            return

        min_val = float(sb.minimum())
        max_val = float(sb.maximum())

        # Advance continuous floating-point position
        self._current_pos += self._velocity

        # Boundary clamping with immediate velocity kill
        hit_boundary = False
        if self._current_pos <= min_val:
            self._current_pos = min_val
            self._velocity = 0.0
            hit_boundary = True
        elif self._current_pos >= max_val:
            self._current_pos = max_val
            self._velocity = 0.0
            hit_boundary = True

        # Apply integer value to scrollbar
        target_int = int(round(self._current_pos))
        if sb.value() != target_int:
            self._is_internal_scroll = True
            sb.setValue(target_int)
            self._is_internal_scroll = False

        if hit_boundary:
            self._stop_physics()
            self._sync_overlay()
            return

        # Natural friction decay (惯性滑动收尾)
        self._velocity *= self._friction
        if abs(self._velocity) < self._threshold:
            self._stop_physics()
            self._sync_overlay()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if sb:
            self._current_pos = float(sb.value())
        self._sync_overlay()

    def wheelEvent(self, event):
        """Intercept mouse wheel to drive smooth physical momentum animation."""
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
        ticks = - (delta / 120.0)

        # If already at edge and trying to push beyond, ignore
        if (current_sb_val <= min_val and ticks < 0) or (current_sb_val >= max_val and ticks > 0):
            event.accept()
            return

        # If programmatic animation was running, interrupt it immediately
        if self._anim.state() == QPropertyAnimation.State.Running:
            self._anim.stop()
            self._current_pos = current_sb_val

        # Sync position if physics was idle
        if not self._physics_timer.isActive():
            self._current_pos = current_sb_val

        # Responsive acceleration for continuous fast flicking
        now = time.monotonic()
        dt = now - self._last_wheel_time if self._last_wheel_time > 0 else 1.0
        self._last_wheel_time = now

        accel = 1.0
        if dt < 0.12 and (self._velocity * ticks > 0):
            speed_ratio = max(0.0, (0.12 - dt) / 0.12)
            accel = 1.0 + speed_ratio * 0.75

        # Impulse calibrated so single notch moves approx self._step_size pixels
        impulse = ticks * self._step_size * (1.0 - self._friction) * accel

        # Direction change handling:
        # If user rolls opposite to current motion, immediately kill old momentum and reverse
        if (self._velocity > 0 and ticks < 0) or (self._velocity < 0 and ticks > 0):
            self._velocity = impulse
        else:
            self._velocity = self._velocity + impulse

        # Clamp velocity
        self._velocity = max(-self._max_velocity, min(self._max_velocity, self._velocity))

        if not self._physics_timer.isActive():
            self._physics_timer.start(16)

        event.accept()

    def scroll_to(self, target_value: int, animate: bool = True):
        """Scroll to target position either with OutCubic easing or immediately."""
        sb = self.horizontalScrollBar() if self.is_horizontal else self.verticalScrollBar()
        if not sb:
            return
        target = max(sb.minimum(), min(sb.maximum(), int(target_value)))
        self._stop_physics()

        if not animate:
            if self._anim.state() == QPropertyAnimation.State.Running:
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
        self._anim.stop()
        self._anim.setTargetObject(sb)
        self._anim.setStartValue(current_val)
        self._anim.setEndValue(target)
        self._anim.start()

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
