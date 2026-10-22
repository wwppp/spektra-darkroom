import re
import os
from PySide6.QtCore import Qt, Signal, QPoint, QSize, QRect, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QCursor, QFont, QIcon, QGuiApplication, QPainter, QColor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QLineEdit, QPushButton, QSizePolicy, QComboBox, QApplication, QListView, QStyledItemDelegate, QAbstractItemView, QStyle
)

from core.path_utils import get_icon_path
from ui.theme_manager import theme_manager


class ScrubLabel(QLabel):
    """Adobe Camera Raw / Premiere style scrubbable parameter title:
    - Left-click and drag horizontally to smoothly scrub values.
    - Global mouse grab ensures dragging continues even if cursor leaves widget.
    - Seamless cursor wrapping across screen boundaries for infinite continuous dragging.
    - Modifier keys: Ctrl/Alt for ultra-fine (0.2x), Shift for accelerated (2.5x).
    - Double-click resets to default value.
    """
    scrubDelta = Signal(float)
    scrubFinished = Signal()
    doubleClicked = Signal()

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._dragging = False
        self._last_global_x = 0.0
        self.setMinimumWidth(10)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

    def apply_theme(self, theme_name=None):
        is_dark = theme_manager.is_dark()
        text_color = "#c8c8c8" if is_dark else "#334155"
        disabled_color = "#4b5563" if is_dark else "#94a3b8"
        self.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                border: none;
                color: {text_color};
                font-size: 11px;
                font-weight: 500;
                padding: 1px 0px;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
            }}
            QLabel:hover {{
                color: #f59e0b;
                background: transparent;
            }}
            QLabel:disabled {{
                color: {disabled_color};
                background: transparent;
            }}
        """)

    def minimumSizeHint(self):
        return QSize(30, 16)

    def mousePressEvent(self, event):
        if not self.isEnabled():
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_global_x = float(event.globalPosition().x())
            try:
                self.grabMouse(Qt.CursorShape.SizeHorCursor)
            except Exception:
                pass
            QApplication.setOverrideCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            cur_global_pos = event.globalPosition()
            cur_x = float(cur_global_pos.x())
            delta = cur_x - self._last_global_x
            if delta != 0.0:
                modifiers = event.modifiers()
                if (modifiers & Qt.KeyboardModifier.ControlModifier) or (modifiers & Qt.KeyboardModifier.AltModifier):
                    speed_mult = 0.20   # Ultra-fine precision
                elif modifiers & Qt.KeyboardModifier.ShiftModifier:
                    speed_mult = 2.50   # Accelerated range sweep
                else:
                    speed_mult = 1.00   # Standard smooth tactile micro-adjustment

                self.scrubDelta.emit(delta * speed_mult)

                # Seamless cursor wrapping across screen boundaries for infinite continuous dragging
                screen = QApplication.screenAt(cur_global_pos.toPoint()) or QApplication.primaryScreen()
                if screen:
                    geo = screen.geometry()
                    margin = 4
                    warp_to_x = None
                    if cur_x >= geo.right() - margin:
                        warp_to_x = geo.left() + margin + 1
                    elif cur_x <= geo.left() + margin:
                        warp_to_x = geo.right() - margin - 1

                    if warp_to_x is not None:
                        target_pt = QPoint(int(warp_to_x), int(cur_global_pos.y()))
                        QCursor.setPos(target_pt)
                        self._last_global_x = float(warp_to_x)
                    else:
                        self._last_global_x = cur_x
                else:
                    self._last_global_x = cur_x
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging:
                self._dragging = False
                try:
                    self.releaseMouse()
                except Exception:
                    pass
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass
                self.scrubFinished.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def hideEvent(self, event):
        if self._dragging:
            self._dragging = False
            try:
                self.releaseMouse()
            except Exception:
                pass
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
        super().hideEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


from PySide6.QtCore import Qt, Signal, QPoint, QSize, QRectF, QPointF
from PySide6.QtGui import QCursor, QFont, QIcon, QPainter, QColor, QPen


class RoundScrubSlider(QSlider):
    """Horizontal slider with custom vector anti-aliased circular knob,
    infinitely continuous smooth visual knob movement (no jerky stepping feel),
    stepped discrete value quantization, optional color gradient track,
    and center detent indicator.
    """
    doubleClicked = Signal()

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._hover = False
        self._pressed = False
        self._smooth_x = None
        self.track_gradient = None
        self.has_center_detent = False
        self.setFixedHeight(18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        theme_manager.themeChanged.connect(lambda _: self.update())

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def _update_from_mouse_x(self, mouse_x):
        w = self.width()
        r = 6.0
        x_start = r + 2.0
        x_end = w - r - 2.0
        span = max(1.0, x_end - x_start)

        # 1. Continuous smooth visual position (sub-pixel precision)
        clamped_x = max(x_start, min(x_end, float(mouse_x)))
        self._smooth_x = clamped_x

        # 2. Stepped integer value quantization
        frac = (clamped_x - x_start) / span
        step_val = self.minimum() + frac * (self.maximum() - self.minimum())
        quantized = int(round(step_val))
        if quantized != self.value():
            self.setValue(quantized)
        self.update()

    def set_continuous_ratio(self, ratio: float):
        """Set continuous floating visual knob position from external continuous ratio [0.0..1.0]."""
        w = self.width()
        r = 6.0
        x_start = r + 2.0
        x_end = w - r - 2.0
        span = max(1.0, x_end - x_start)
        clamped_ratio = max(0.0, min(1.0, float(ratio)))
        self._smooth_x = x_start + clamped_ratio * span
        self.update()

    def clear_continuous(self):
        """Clear external continuous visual knob position and snap to quantized value."""
        self._smooth_x = None
        self.update()

    def mousePressEvent(self, event):
        if not self.isEnabled():
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self._update_from_mouse_x(event.position().x())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.isEnabled():
            event.ignore()
            return
        if self._pressed:
            self._update_from_mouse_x(event.position().x())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = False
            self._smooth_x = None
            self.update()
            self.sliderReleased.emit()
            event.accept()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self.isEnabled():
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        event.ignore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        is_dark = theme_manager.is_dark()
        w = self.width()
        h = self.height()
        cy = h / 2.0
        r = 6.0  # knob radius = 6px (12px diameter)

        x_start = r + 2.0
        x_end = w - r - 2.0
        span = max(1.0, x_end - x_start)

        # Knob center X calculation: continuous if dragging/scrubbing, snapped if idle
        if self._smooth_x is not None:
            knob_cx = max(x_start, min(x_end, self._smooth_x))
        else:
            frac = 0.0
            if self.maximum() > self.minimum():
                frac = (self.value() - self.minimum()) / float(self.maximum() - self.minimum())
                frac = max(0.0, min(1.0, frac))
            knob_cx = x_start + frac * span

        # Disabled rendering: flat muted slot + dimmed knob
        if not self.isEnabled():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(25, 27, 33) if is_dark else QColor(226, 232, 240))
            painter.drawRoundedRect(QRectF(x_start, cy - 1.5, span, 3.0), 1.5, 1.5)
            if self.has_center_detent:
                mid_x = x_start + span * 0.5
                painter.setPen(QColor(255, 255, 255, 25) if is_dark else QColor(0, 0, 0, 30))
                painter.drawLine(QPointF(mid_x, cy - 2.0), QPointF(mid_x, cy + 2.0))
                painter.setPen(Qt.PenStyle.NoPen)
            knob_color = QColor(55, 59, 70) if is_dark else QColor(190, 195, 205)
            border_pen = QPen(QColor(32, 34, 40) if is_dark else QColor(203, 213, 225), 1.0)
            painter.setPen(border_pen)
            painter.setBrush(knob_color)
            painter.drawEllipse(QPointF(knob_cx, cy), r, r)
            return

        # 1. Background Groove (Strict 3.0px height as requested)
        painter.setPen(Qt.PenStyle.NoPen)
        if self.track_gradient:
            from PySide6.QtGui import QLinearGradient
            grad = QLinearGradient(x_start, cy, x_end, cy)
            for stop, color in self.track_gradient:
                grad.setColorAt(stop, color)
            painter.setBrush(grad)
            painter.drawRoundedRect(QRectF(x_start, cy - 1.5, span, 3.0), 1.5, 1.5)
            # Subtle 0.5px border to ensure contrast
            painter.setPen(QColor(0, 0, 0, 100) if is_dark else QColor(0, 0, 0, 40))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(x_start, cy - 1.5, span, 3.0), 1.5, 1.5)
            painter.setPen(Qt.PenStyle.NoPen)
        else:
            groove_bg = QColor(42, 44, 52) if is_dark else QColor(218, 222, 230)
            active_bg = QColor(90, 95, 110) if is_dark else QColor(140, 146, 160)
            painter.setBrush(groove_bg)
            painter.drawRoundedRect(QRectF(x_start, cy - 1.5, span, 3.0), 1.5, 1.5)
            # Active subpage track for normal uncolored slider
            painter.setBrush(active_bg)
            painter.drawRoundedRect(QRectF(x_start, cy - 1.5, max(0.0, knob_cx - x_start), 3.0), 1.5, 1.5)

        # 2. Optional Center Detent tick mark (1px clean mark for center-balanced sliders)
        if self.has_center_detent:
            mid_x = x_start + span * 0.5
            detent_color = QColor(255, 255, 255, 150) if is_dark else QColor(71, 85, 105, 180)
            painter.setPen(detent_color)
            painter.drawLine(QPointF(mid_x, cy - 2.5), QPointF(mid_x, cy + 2.5))
            painter.setPen(Qt.PenStyle.NoPen)

        # 3. Vector circular knob
        if self._pressed:
            knob_color = QColor(217, 119, 6)   # deep amber
            border_pen = QPen(QColor(245, 158, 11), 1.5)
        elif self._hover:
            knob_color = QColor(245, 158, 11)  # bright amber
            border_pen = QPen(QColor(255, 255, 255), 1.5)
        else:
            knob_color = QColor(228, 230, 238) if is_dark else QColor(255, 255, 255)
            border_pen = QPen(QColor(30, 32, 38) if is_dark else QColor(148, 163, 184), 1.2)

        painter.setPen(border_pen)
        painter.setBrush(knob_color)
        painter.drawEllipse(QPointF(knob_cx, cy), r, r)


class ParamScrubSlider(QWidget):
    """Complete scrubbable parameter slider module:
    - Left: Scrubbable label (drag to adjust, double-click to reset, detailed tooltip)
    - Center: Custom sleek dark slider with round knob and double-click reset
    - Right: Numeric line edit + reset button with light reset icon
    """
    valueChanged = Signal(float)
    sliderReleased = Signal()

    def __init__(self, label_text, min_val, max_val, default_val, step=0.01, unit="", decimals=2, tooltip="",
                 track_gradient=None, has_center_detent=False, parent=None):
        super().__init__(parent)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.default_val = float(default_val)
        self.current_val = float(default_val)
        self.step = float(step)
        self.unit = unit
        self.decimals = decimals
        self.tooltip_text = tooltip.strip() if tooltip else ""
        self.default_tooltip = self.tooltip_text
        self.track_gradient = track_gradient
        self.has_center_detent = has_center_detent
        self._is_updating = False

        self.steps_total = int(round((self.max_val - self.min_val) / self.step))

        self._init_ui(label_text)

    def _init_ui(self, label_text):
        self.setStyleSheet("ParamScrubSlider, AdobeScrubSlider { background: transparent; border: none; }")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 3, 0, 4)
        main_layout.setSpacing(2)

        # Header row: Label + Value edit + Reset button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self.label = ScrubLabel(label_text)
        if self.tooltip_text:
            self.label.setToolTip(self.tooltip_text)
            self.setToolTip(self.tooltip_text)
        self.label.scrubDelta.connect(self._on_scrub)
        self.label.scrubFinished.connect(self._on_scrub_finished)
        self.label.doubleClicked.connect(self.reset)
        header_layout.addWidget(self.label, 1)

        self.value_edit = QLineEdit()
        self.value_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.value_edit.setFixedWidth(52)
        self.value_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.value_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_edit.setToolTip("点击输入精确数值，Enter 确认")
        self.value_edit.returnPressed.connect(self._on_text_edited)
        self.value_edit.editingFinished.connect(self._on_text_edited)
        header_layout.addWidget(self.value_edit)

        self.reset_btn = QPushButton()
        self.reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.reset_btn.setFixedSize(16, 16)
        self.reset_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setToolTip("默认值")
        self._icon_reset_disabled = None
        self._icon_reset_active = None
        path_dis = get_icon_path("reset_disabled.png")
        path_act = get_icon_path("reset_active.png")
        if os.path.exists(path_dis) and os.path.exists(path_act):
            self._icon_reset_disabled = QIcon(path_dis)
            self._icon_reset_active = QIcon(path_act)
            self.reset_btn.setIcon(self._icon_reset_disabled)
            self.reset_btn.setIconSize(QSize(11, 11))
        else:
            reset_icon_path = get_icon_path("reset.png")
            if os.path.exists(reset_icon_path):
                self.reset_btn.setIcon(QIcon(reset_icon_path))
                self.reset_btn.setIconSize(QSize(11, 11))
            else:
                self.reset_btn.setText("R")
        self.reset_btn.clicked.connect(self.reset)
        header_layout.addWidget(self.reset_btn)

        main_layout.addLayout(header_layout)

        # Slider track with perfectly round knob
        self.slider = RoundScrubSlider(Qt.Orientation.Horizontal)
        self.slider.track_gradient = self.track_gradient
        self.slider.has_center_detent = self.has_center_detent
        self.slider.setMinimumWidth(30)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.setRange(0, self.steps_total)
        if self.tooltip_text:
            self.slider.setToolTip(self.tooltip_text)
        self.slider.setStyleSheet("QSlider { background: transparent; border: none; }")
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.doubleClicked.connect(self.reset)
        main_layout.addWidget(self.slider)

        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

    def apply_theme(self, theme_name=None):
        is_dark = theme_manager.is_dark()
        input_bg = "#141414" if is_dark else "#ffffff"
        input_fg = "#e0e0e0" if is_dark else "#1e293b"
        input_border = "#363636" if is_dark else "#cbd5e1"
        input_dis_bg = "#111216" if is_dark else "#f1f5f9"
        input_dis_fg = "#4b5563" if is_dark else "#94a3b8"
        input_dis_border = "#20222a" if is_dark else "#e2e8f0"
        btn_hover = "#282a36" if is_dark else "#e2e8f0"
        btn_pressed = "#1e202a" if is_dark else "#cbd5e1"

        if hasattr(self, 'value_edit') and self.value_edit:
            self.value_edit.setStyleSheet(f"""
                QLineEdit {{
                    background: {input_bg};
                    color: {input_fg};
                    border: 1px solid {input_border};
                    border-radius: 3px;
                    padding: 1px 3px;
                    font-size: 11px;
                    font-family: "Segoe UI", "Microsoft YaHei UI", Consolas, monospace;
                }}
                QLineEdit:focus {{
                    border: 1px solid #f59e0b;
                    background: {input_bg};
                }}
                QLineEdit:disabled {{
                    background: {input_dis_bg};
                    color: {input_dis_fg};
                    border: 1px solid {input_dis_border};
                }}
            """)
        if hasattr(self, 'reset_btn') and self.reset_btn:
            self.reset_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    padding: 1px;
                    border-radius: 2px;
                }}
                QPushButton:hover {{
                    background: {btn_hover};
                }}
                QPushButton:pressed {{
                    background: {btn_pressed};
                }}
                QPushButton:disabled {{
                    opacity: 0.25;
                }}
            """)
        if hasattr(self, 'label') and self.label:
            self.label.apply_theme()
        if hasattr(self, 'slider') and self.slider:
            self.slider.update()

    def setToolTip(self, text):
        super().setToolTip(text)
        if hasattr(self, 'label') and self.label:
            self.label.setToolTip(text)
        if hasattr(self, 'slider') and self.slider:
            self.slider.setToolTip(text)

    def set_track_gradient(self, gradient, center_detent=False):
        self.track_gradient = gradient
        self.has_center_detent = center_detent
        if hasattr(self, 'slider'):
            self.slider.track_gradient = gradient
            self.slider.has_center_detent = center_detent
            self.slider.update()

        self._update_display(self.default_val)

    def _val_to_slider(self, val):
        ratio = (val - self.min_val) / (self.max_val - self.min_val)
        return int(round(ratio * self.steps_total))

    def _slider_to_val(self, slider_pos):
        ratio = slider_pos / float(self.steps_total) if self.steps_total > 0 else 0.0
        val = self.min_val + ratio * (self.max_val - self.min_val)
        return round(val, self.decimals)

    def _update_display(self, val, sync_precise=True):
        self._is_updating = True
        self.current_val = val
        if sync_precise:
            self._precise_val = float(val)
        self.slider.setValue(self._val_to_slider(val))
        
        # Format text with +/- and unit
        prefix = "+" if (val > 0 and self.min_val < 0) else ""
        if self.decimals == 0:
            formatted = f"{prefix}{int(val)}{self.unit}"
        else:
            formatted = f"{prefix}{val:.{self.decimals}f}{self.unit}"
        self.value_edit.setText(formatted)

        # Update reset button state: grey for default, yellow for edited
        if hasattr(self, 'reset_btn') and hasattr(self, '_icon_reset_active') and self._icon_reset_active:
            is_modified = abs(val - self.default_val) > (self.step * 0.4)
            if is_modified:
                self.reset_btn.setIcon(self._icon_reset_active)
                self.reset_btn.setToolTip("已修改：点击重置为默认值")
            else:
                self.reset_btn.setIcon(self._icon_reset_disabled)
                self.reset_btn.setToolTip("默认值")

        self._is_updating = False

    def _on_slider_changed(self, pos):
        if self._is_updating:
            return
        val = self._slider_to_val(pos)
        self.current_val = val
        self._precise_val = float(val)
        self._update_display(val, sync_precise=True)
        self.valueChanged.emit(val)

    def _on_slider_released(self):
        self.sliderReleased.emit()

    def _on_scrub(self, delta_x):
        if not hasattr(self, "_precise_val") or self._precise_val is None:
            self._precise_val = float(self.current_val)

        # Micro-adjustment rate: 0.10 * step per pixel
        step_factor = self.step * 0.10
        self._precise_val += (delta_x * step_factor)
        self._precise_val = max(self.min_val, min(self.max_val, self._precise_val))

        # Drive visual continuous slider knob smoothly without jumping
        val_range = float(self.max_val - self.min_val)
        if val_range > 0:
            ratio = (self._precise_val - self.min_val) / val_range
            self.slider.set_continuous_ratio(ratio)

        # Discrete quantization for value and engine updates
        quantized_steps = round((self._precise_val - self.min_val) / self.step)
        rounded_val = self.min_val + quantized_steps * self.step
        rounded_val = round(max(self.min_val, min(self.max_val, rounded_val)), self.decimals)

        if rounded_val != self.current_val:
            self.current_val = rounded_val
            self._update_display(rounded_val, sync_precise=False)
            self.valueChanged.emit(self.current_val)

    def _on_scrub_finished(self):
        self._precise_val = float(self.current_val)
        self.slider.clear_continuous()
        self.sliderReleased.emit()

    def _on_text_edited(self):
        text = self.value_edit.text().replace(self.unit, "").replace("+", "").strip()
        try:
            val = float(text)
            val = max(self.min_val, min(self.max_val, val))
            self._precise_val = float(val)
            self._update_display(round(val, self.decimals))
            self.valueChanged.emit(self.current_val)
            self.sliderReleased.emit()
        except ValueError:
            self._update_display(self.current_val)

    def value(self):
        return self.current_val

    def setValue(self, val):
        val = max(self.min_val, min(self.max_val, float(val)))
        self.current_val = round(val, self.decimals)
        self._precise_val = float(self.current_val)
        self._update_display(self.current_val)
        self.valueChanged.emit(self.current_val)

    set_value = setValue

    def reset(self):
        # setValue() already formats display and emits valueChanged once
        self.setValue(self.default_val)
        self.sliderReleased.emit()

    def set_default_value(self, val):
        val = float(val)
        self.default_val = val
        if val < self.min_val:
            self.min_val = val
            self.steps_total = int(round((self.max_val - self.min_val) / self.step))
            self.slider.setRange(0, self.steps_total)
        elif val > self.max_val:
            self.max_val = val
            self.steps_total = int(round((self.max_val - self.min_val) / self.step))
            self.slider.setRange(0, self.steps_total)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        if hasattr(self, 'label') and self.label:
            self.label.setEnabled(enabled)
        if hasattr(self, 'slider') and self.slider:
            self.slider.setEnabled(enabled)
            self.slider.update()
        if hasattr(self, 'value_edit') and self.value_edit:
            self.value_edit.setEnabled(enabled)
        if hasattr(self, 'reset_btn') and self.reset_btn:
            self.reset_btn.setEnabled(enabled)
        self.update()

    def wheelEvent(self, event):
        event.ignore()


# Backward compatibility alias
AdobeScrubSlider = ParamScrubSlider


class DarkroomComboItemDelegate(QStyledItemDelegate):
    """Custom item delegate for DarkroomComboBox popup:
    - Fixed 24px item height matching unexpanded combobox
    - 8px left margin matching unexpanded combobox text padding
    - Clear visual feedback on hover (#262936 dark / #f1f5f9 light)
    - Selected row highlight: left 3px amber accent bar (#f59e0b) and amber bold text
    """
    def __init__(self, combo, parent=None):
        super().__init__(parent)
        self.combo = combo
        self.hovered_row = -1

    def sizeHint(self, option, index):
        w = option.rect.width() if (option and hasattr(option, "rect")) else 100
        return QSize(w, 24)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = option.rect
        row = index.row()
        is_selected = (row == self.combo.currentIndex())
        is_hovered = (row == self.hovered_row)

        is_dark = theme_manager.is_dark()
        if is_selected and is_hovered:
            bg = QColor("#2e2a22") if is_dark else QColor("#fef3c7")
        elif is_selected:
            bg = QColor("#24211a") if is_dark else QColor("#fffbeb")
        elif is_hovered:
            bg = QColor("#262936") if is_dark else QColor("#f1f5f9")
        else:
            bg = QColor("#181920") if is_dark else QColor("#ffffff")
        painter.fillRect(r, bg)

        # Left 3px amber accent bar for selected item
        if is_selected:
            bar_color = QColor("#f59e0b")
            painter.fillRect(QRect(r.x(), r.y() + 2, 3, r.height() - 4), bar_color)

        # Text: 8px left margin matching the unexpanded combobox padding
        txt_rect = QRect(r.x() + 8, r.y(), r.width() - 16, r.height())
        if is_selected:
            painter.setPen(QColor("#f59e0b") if is_dark else QColor("#d97706"))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
        elif is_hovered:
            painter.setPen(QColor("#ffffff") if is_dark else QColor("#0f172a"))
        else:
            painter.setPen(QColor("#cbd5e1") if is_dark else QColor("#334155"))

        raw_text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        elided = option.fontMetrics.elidedText(raw_text, Qt.TextElideMode.ElideRight, txt_rect.width())
        painter.drawText(txt_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)
        painter.restore()


# Backward compatibility alias
AdobeComboItemDelegate = DarkroomComboItemDelegate


class DarkroomComboListView(QListView):
    """Custom popup listview with smooth inertial scrolling and row hover tracking."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scroll_anim = QVariantAnimation(self)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim.valueChanged.connect(self._on_scroll_anim_step)
        self._target_scroll = 0
        self.delegate = None

    def _on_scroll_anim_step(self, val):
        self.verticalScrollBar().setValue(int(val))

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            return
        sb = self.verticalScrollBar()
        if sb.maximum() <= 0:
            event.ignore()
            return
        step = -int((delta / 120.0) * 72)
        if self._scroll_anim.state() == QVariantAnimation.State.Running:
            base = self._target_scroll
        else:
            base = sb.value()
        target = max(0, min(sb.maximum(), base + step))
        self._target_scroll = target
        curr = sb.value()
        dist = abs(target - curr)
        if dist > 0:
            duration = min(320, max(160, int(dist * 2.2)))
            self._scroll_anim.stop()
            self._scroll_anim.setDuration(duration)
            self._scroll_anim.setStartValue(curr)
            self._scroll_anim.setEndValue(target)
            self._scroll_anim.start()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.delegate:
            idx = self.indexAt(event.pos())
            row = idx.row() if idx.isValid() else -1
            if self.delegate.hovered_row != row:
                self.delegate.hovered_row = row
                self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.delegate and self.delegate.hovered_row != -1:
            self.delegate.hovered_row = -1
            self.viewport().update()
        super().leaveEvent(event)


# Backward compatibility alias
AdobeComboListView = DarkroomComboListView


class DarkroomComboBox(QComboBox):
    """Custom darkroom styled combobox featuring:
    - Screen-edge awareness (smart upward/downward popup direction)
    - Dynamic visible item count (scales from 6 up to 24 items depending on available screen height)
    - Compact bounded popup list width matching unexpanded control with tooltip for overflow
    - Smooth opening animation (OutCubic expansion and opacity fade)
    - Smooth inertial mouse-wheel scrolling
    - Full text tooltip on hover for collapsed box and popup list items
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_view = DarkroomComboListView()
        self.combo_delegate = DarkroomComboItemDelegate(self)
        self.combo_view.delegate = self.combo_delegate
        self.setView(self.combo_view)
        self.setItemDelegate(self.combo_delegate)
        self.combo_view.setAutoFillBackground(True)
        if hasattr(self.combo_view, "window"):
            self.combo_view.window().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.combo_view.window().setAutoFillBackground(True)
            self.combo_view.window().setWindowOpacity(0.0)
        self.wheelEvent = lambda ev: ev.ignore()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._popup_anim = None
        self._orig_sb_policy = None
        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

    def apply_theme(self, theme_name=None):
        is_dark = theme_manager.is_dark()
        tooltip_bg = "#ffffff" if not is_dark else "rgba(22, 24, 30, 0.95)"
        tooltip_fg = "#1e293b" if not is_dark else "#f1f5f9"
        tooltip_border = "#cbd5e1" if not is_dark else "#333644"
        if hasattr(self, "combo_view") and self.combo_view:
            self.combo_view.setStyleSheet(f"""
                QToolTip {{
                    background-color: {tooltip_bg};
                    color: {tooltip_fg};
                    border: 1px solid {tooltip_border};
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-size: 11px;
                    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
                }}
            """)

    def showPopup(self):
        if self.count() == 0:
            super().showPopup()
            return

        # 1. Adapt popup width to content text length (8px left + 8px right + 20px scrollbar/border = 36px)
        fm = self.fontMetrics()
        max_text_w = 0
        for i in range(self.count()):
            w = fm.horizontalAdvance(self.itemText(i))
            if w > max_text_w:
                max_text_w = w
        target_width = max(self.width(), max_text_w + 36)

        # 2. Measure screen available geometry
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QRect(0, 0, 1920, 1080)
        p_global = self.mapToGlobal(QPoint(0, 0))

        # Clamp width to maximum screen width
        max_allowed_w = max(self.width(), avail.width() - 24)
        target_width = min(target_width, max_allowed_w)

        # Adaptive horizontal positioning: if expanding rightwards overflows screen edge, expand to the left!
        target_x = p_global.x()
        if target_x + target_width + 12 > avail.right():
            right_edge = min(avail.right() - 8, p_global.x() + self.width())
            target_x = max(avail.left() + 8, right_edge - target_width)

        self.combo_view.setMinimumWidth(target_width)
        self.combo_view.setMaximumWidth(target_width)

        # 3. Dynamic visible items and direction detection
        space_below = avail.bottom() - (p_global.y() + self.height())
        space_above = p_global.y() - avail.top()

        item_h = 24
        max_space = max(space_below, space_above)
        fit_items = max(6, min(24, int((max_space - 20) / item_h)))
        self.setMaxVisibleItems(min(self.count(), fit_items))

        # 4. Zero opacity before super().showPopup() to completely eliminate initial-frame flash
        container = self.combo_view.window()
        if container:
            container.setWindowOpacity(0.0)

        self._orig_sb_policy = self.combo_view.verticalScrollBarPolicy()
        self.combo_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        super().showPopup()

        # 5. Opening animation (smooth expansion from height 1 + opacity fade)
        if container:
            orig_geo = container.geometry()
            is_upward = orig_geo.y() < p_global.y()
            if hasattr(self, '_popup_anim') and self._popup_anim:
                self._popup_anim.stop()

            # Ensure final dest_geo uses adaptive target_x and target_width!
            dest_geo = QRect(target_x, orig_geo.y(), target_width, orig_geo.height())

            if is_upward:
                start_geo = QRect(target_x, dest_geo.bottom(), target_width, 1)
            else:
                start_geo = QRect(target_x, dest_geo.y(), target_width, 1)
            container.setGeometry(start_geo)

            self._popup_anim = QVariantAnimation(self)
            self._popup_anim.setDuration(120)
            self._popup_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._popup_anim.setStartValue(start_geo)
            self._popup_anim.setEndValue(dest_geo)

            def _on_step(r):
                container.setGeometry(r)
                prog = self._popup_anim.currentTime() / max(1, self._popup_anim.duration())
                container.setWindowOpacity(min(1.0, 0.1 + 0.9 * prog))

            def _on_finished():
                container.setGeometry(dest_geo)
                container.setWindowOpacity(1.0)
                if hasattr(self, '_orig_sb_policy') and self._orig_sb_policy is not None:
                    self.combo_view.setVerticalScrollBarPolicy(self._orig_sb_policy)

            self._popup_anim.valueChanged.connect(_on_step)
            self._popup_anim.finished.connect(_on_finished)
            self._popup_anim.start()

    def hidePopup(self):
        if hasattr(self, '_popup_anim') and self._popup_anim:
            self._popup_anim.stop()
        if hasattr(self, '_orig_sb_policy') and self._orig_sb_policy is not None:
            self.combo_view.setVerticalScrollBarPolicy(self._orig_sb_policy)
        container = self.combo_view.window() if hasattr(self, 'combo_view') else None
        if container:
            container.setWindowOpacity(0.0)
        super().hidePopup()


# Backward compatibility alias
AdobeComboBox = DarkroomComboBox


class ComboBoxRow(QWidget):
    """Modern horizontal parameter row with label and drop-down selection."""
    valueChanged = Signal(object)

    def __init__(self, label_text, items, default_val=None, tooltip="", parent=None):
        super().__init__(parent)
        self.default_val = default_val
        self.default_tooltip = tooltip.strip() if tooltip else ""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        self.label = QLabel(label_text)
        self.label.setFixedWidth(74)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if self.default_tooltip:
            self.setToolTip(self.default_tooltip)
            self.label.setToolTip(self.default_tooltip)
        layout.addWidget(self.label)

        self.combo = DarkroomComboBox()
        self.combo.setFixedHeight(24)
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo.setMinimumWidth(100)
        if self.default_tooltip:
            self.combo.setToolTip(self.default_tooltip)

        for idx, (text, val) in enumerate(items):
            self.combo.addItem(text, val)
            self.combo.setItemData(idx, text, Qt.ItemDataRole.ToolTipRole)
        if default_val is not None:
            self.set_value(default_val)
        elif self.combo.currentIndex() >= 0:
            self.combo.setToolTip(self.combo.currentText())
        self.combo.currentIndexChanged.connect(self._on_index_changed)
        layout.addWidget(self.combo)

        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

    def apply_theme(self, theme_name=None):
        is_dark = theme_manager.is_dark()
        label_fg = "#c8c8c8" if is_dark else "#334155"
        label_dis_fg = "#4b5563" if is_dark else "#94a3b8"
        self.label.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                border: none;
                color: {label_fg};
                font-size: 11px;
                font-weight: 500;
                padding: 1px 0px;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
            }}
            QLabel:disabled {{
                color: {label_dis_fg};
                background: transparent;
            }}
        """)

        arrow_icon = get_icon_path("arrow_down.png").replace("\\", "/")
        arrow_disabled = get_icon_path("arrow_down_disabled.png").replace("\\", "/")

        combo_bg = "#14151a" if is_dark else "#ffffff"
        combo_fg = "#e2e8f0" if is_dark else "#1e293b"
        combo_border = "#2d303e" if is_dark else "#cbd5e1"
        combo_dis_bg = "#111216" if is_dark else "#f1f5f9"
        combo_dis_fg = "#4b5563" if is_dark else "#94a3b8"
        combo_dis_border = "#1e2029" if is_dark else "#e2e8f0"
        dropdown_border = "#242632" if is_dark else "#e2e8f0"
        dropdown_dis_border = "#1a1b22" if is_dark else "#e2e8f0"
        popup_bg = "#181920" if is_dark else "#ffffff"
        popup_fg = "#e2e8f0" if is_dark else "#1e293b"
        popup_border = "#333644" if is_dark else "#cbd5e1"
        sel_fg = "#111111" if is_dark else "#ffffff"

        tooltip_bg = "#ffffff" if not is_dark else "rgba(22, 24, 30, 0.95)"
        tooltip_fg = "#1e293b" if not is_dark else "#f1f5f9"
        tooltip_border = "#cbd5e1" if not is_dark else "#333644"

        self.combo.setStyleSheet(f"""
            QComboBox {{
                background: {combo_bg};
                color: {combo_fg};
                border: 1px solid {combo_border};
                border-radius: 4px;
                padding: 3px 22px 3px 8px;
                font-size: 11px;
                min-width: 100px;
            }}
            QComboBox:hover {{
                border-color: #f59e0b;
            }}
            QComboBox:focus {{
                border-color: #f59e0b;
            }}
            QComboBox:disabled {{
                background: {combo_dis_bg};
                color: {combo_dis_fg};
                border-color: {combo_dis_border};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid {dropdown_border};
            }}
            QComboBox::drop-down:disabled {{
                border-left-color: {dropdown_dis_border};
            }}
            QComboBox::down-arrow {{
                image: url("{arrow_icon}");
                width: 9px;
                height: 6px;
                margin-right: 2px;
            }}
            QComboBox::down-arrow:disabled {{
                image: url("{arrow_disabled}");
                width: 9px;
                height: 6px;
                margin-right: 2px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {popup_bg};
                background: {popup_bg};
                color: {popup_fg};
                border: 1px solid {popup_border};
                border-radius: 4px;
                padding: 2px 0px;
                outline: 0px;
            }}
            QToolTip {{
                background-color: {tooltip_bg};
                color: {tooltip_fg};
                border: 1px solid {tooltip_border};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
            }}
        """)
        if hasattr(self.combo, "view") and self.combo.view():
            self.combo.view().setStyleSheet(f"""
                QToolTip {{
                    background-color: {tooltip_bg};
                    color: {tooltip_fg};
                    border: 1px solid {tooltip_border};
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-size: 11px;
                    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
                }}
            """)

    def setToolTip(self, text):
        super().setToolTip(text)
        if hasattr(self, 'label') and self.label:
            self.label.setToolTip(text)
        if hasattr(self, 'combo') and self.combo:
            self.combo.setToolTip(text)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        if hasattr(self, 'label') and self.label:
            self.label.setEnabled(enabled)
        if hasattr(self, 'combo') and self.combo:
            self.combo.setEnabled(enabled)

    def _on_index_changed(self, idx):
        if idx >= 0:
            txt = self.combo.currentText()
            self.combo.setToolTip(txt if txt else self.default_tooltip)
        self.valueChanged.emit(self.combo.currentData())

    def current_value(self):
        return self.combo.currentData()

    def set_value(self, val):
        idx = self.combo.findData(val)
        if idx >= 0:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(idx)
            txt = self.combo.currentText()
            self.combo.setToolTip(txt if txt else self.default_tooltip)
            self.combo.blockSignals(False)

    def reset_default(self):
        if self.default_val is not None:
            self.set_value(self.default_val)

    def set_items(self, items, default_val=None):
        self.combo.blockSignals(True)
        self.combo.clear()
        for idx, (text, val) in enumerate(items):
            self.combo.addItem(text, val)
            self.combo.setItemData(idx, text, Qt.ItemDataRole.ToolTipRole)
        if default_val is not None:
            idx = self.combo.findData(default_val)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
            elif self.combo.count() > 0:
                self.combo.setCurrentIndex(0)
        elif self.combo.count() > 0:
            self.combo.setCurrentIndex(0)
        if self.combo.currentIndex() >= 0:
            txt = self.combo.currentText()
            self.combo.setToolTip(txt if txt else self.default_tooltip)
        self.combo.blockSignals(False)

