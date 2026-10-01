import re
import os
from PySide6.QtCore import Qt, Signal, QPoint, QSize
from PySide6.QtGui import QCursor, QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QLineEdit, QPushButton, QSizePolicy, QComboBox, QApplication
)

from path_utils import get_icon_path


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
        self.setStyleSheet("""
            QLabel {
                color: #c8c8c8;
                font-size: 11.5px;
                font-weight: 500;
                padding: 1px 0px;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QLabel:hover {
                color: #f59e0b;
            }
        """)

    def minimumSizeHint(self):
        return QSize(30, 16)

    def mousePressEvent(self, event):
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
    immediate click-to-reposition, double-click to reset,
    and ignored wheel event to prevent sidebar scrolling conflict.
    """
    doubleClicked = Signal()

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._hover = False
        self._pressed = False
        self.setFixedHeight(18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            w = max(1, self.width() - 14)
            val = self.minimum() + ((event.position().x() - 7) / float(w)) * (self.maximum() - self.minimum())
            self.setValue(int(round(max(self.minimum(), min(self.maximum(), val)))))
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pressed:
            w = max(1, self.width() - 14)
            val = self.minimum() + ((event.position().x() - 7) / float(w)) * (self.maximum() - self.minimum())
            self.setValue(int(round(max(self.minimum(), min(self.maximum(), val)))))
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
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

        w = self.width()
        h = self.height()
        cy = h / 2.0
        r = 6.0  # knob radius = 6px (12px diameter)

        x_start = r + 2.0
        x_end = w - r - 2.0
        span = max(1.0, x_end - x_start)

        frac = 0.0
        if self.maximum() > self.minimum():
            frac = (self.value() - self.minimum()) / float(self.maximum() - self.minimum())
            frac = max(0.0, min(1.0, frac))

        knob_cx = x_start + frac * span

        # 1. Background Groove
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(42, 44, 52))
        painter.drawRoundedRect(QRectF(x_start, cy - 1.5, span, 3.0), 1.5, 1.5)

        # 2. Active subpage track
        painter.setBrush(QColor(90, 95, 110))
        painter.drawRoundedRect(QRectF(x_start, cy - 1.5, max(0.0, knob_cx - x_start), 3.0), 1.5, 1.5)

        # 3. Vector circular knob
        if self._pressed:
            knob_color = QColor(217, 119, 6)   # deep amber
            border_pen = QPen(QColor(245, 158, 11), 1.5)
        elif self._hover:
            knob_color = QColor(245, 158, 11)  # bright amber
            border_pen = QPen(QColor(255, 255, 255), 1.5)
        else:
            knob_color = QColor(228, 230, 238) # clean light silver
            border_pen = QPen(QColor(30, 32, 38), 1.2)

        painter.setPen(border_pen)
        painter.setBrush(knob_color)
        painter.drawEllipse(QPointF(knob_cx, cy), r, r)


class AdobeScrubSlider(QWidget):
    """Complete Adobe Photoshop / Camera Raw slider module:
    - Left: Scrubbable label (drag to adjust, double-click to reset, detailed tooltip)
    - Center: Custom sleek dark slider with round knob and double-click reset
    - Right: Numeric line edit + reset button with light reset icon
    """
    valueChanged = Signal(float)
    sliderReleased = Signal()

    def __init__(self, label_text, min_val, max_val, default_val, step=0.01, unit="", decimals=2, tooltip="", parent=None):
        super().__init__(parent)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.default_val = float(default_val)
        self.current_val = float(default_val)
        self.step = float(step)
        self.unit = unit
        self.decimals = decimals
        self.tooltip_text = tooltip
        self._is_updating = False

        self.steps_total = int(round((self.max_val - self.min_val) / self.step))

        self._init_ui(label_text)

    def _init_ui(self, label_text):
        self.setStyleSheet("AdobeScrubSlider { background: transparent; border: none; }")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 3, 0, 4)
        main_layout.setSpacing(2)

        # Header row: Label + Value edit + Reset button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self.label = ScrubLabel(label_text)
        clean_tip = re.sub(r'[\(（][^()（）]*[\)）]', '', self.tooltip_text).strip() if self.tooltip_text else ""
        clean_tip = re.sub(r'\s{2,}', ' ', clean_tip)
        self.label.setToolTip(clean_tip)
        self.label.scrubDelta.connect(self._on_scrub)
        self.label.scrubFinished.connect(self._on_scrub_finished)
        self.label.doubleClicked.connect(self.reset)
        header_layout.addWidget(self.label, 1)

        self.value_edit = QLineEdit()
        self.value_edit.setFixedWidth(52)
        self.value_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.value_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_edit.setToolTip("点击输入精确数值，Enter 确认")
        self.value_edit.setStyleSheet("""
            QLineEdit {
                background: #141414;
                color: #e0e0e0;
                border: 1px solid #363636;
                border-radius: 3px;
                padding: 1px 3px;
                font-size: 11px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Consolas, monospace;
            }
            QLineEdit:focus {
                border: 1px solid #f59e0b;
                background: #1c1c1c;
            }
        """)
        self.value_edit.returnPressed.connect(self._on_text_edited)
        self.value_edit.editingFinished.connect(self._on_text_edited)
        header_layout.addWidget(self.value_edit)

        self.reset_btn = QPushButton()
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
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                padding: 1px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background: #282a36;
            }
        """)
        self.reset_btn.clicked.connect(self.reset)
        header_layout.addWidget(self.reset_btn)

        main_layout.addLayout(header_layout)

        # Slider track with perfectly round knob
        self.slider = RoundScrubSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimumWidth(30)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.setRange(0, self.steps_total)
        self.slider.setToolTip(clean_tip)
        self.slider.setStyleSheet("QSlider { background: transparent; border: none; }")
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.doubleClicked.connect(self.reset)
        main_layout.addWidget(self.slider)

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

        # Micro-adjustment rate calibrated: 0.05 * step per pixel
        # Eliminates rapid jumping and ensures smooth, continuous analog response
        step_factor = self.step * 0.05
        self._precise_val += (delta_x * step_factor)
        self._precise_val = max(self.min_val, min(self.max_val, self._precise_val))

        rounded_val = round(self._precise_val, self.decimals)
        if rounded_val != self.current_val:
            self.current_val = rounded_val
            self._update_display(rounded_val, sync_precise=False)
            self.valueChanged.emit(self.current_val)

    def _on_scrub_finished(self):
        self._precise_val = float(self.current_val)
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
        self.setValue(self.default_val)
        self.valueChanged.emit(self.default_val)
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

    def get_default_value(self):
        return self.default_val

    def wheelEvent(self, event):
        event.ignore()


class ComboBoxRow(QWidget):
    """Adobe-styled horizontal parameter row with label and drop-down selection."""
    valueChanged = Signal(object)

    def __init__(self, label_text, items, default_val=None, tooltip="", parent=None):
        super().__init__(parent)
        self.default_val = default_val
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(6)

        self.label = QLabel(label_text)
        self.label.setStyleSheet("""
            QLabel {
                color: #c8c8c8;
                font-size: 11.5px;
                font-weight: 500;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
            }
        """)
        if tooltip:
            cleaned_tip = re.sub(r'[\(（][^()（）]*[\)）]', '', tooltip).strip()
            cleaned_tip = re.sub(r'\s{2,}', ' ', cleaned_tip)
            self.setToolTip(cleaned_tip)
            self.label.setToolTip(cleaned_tip)
        layout.addWidget(self.label)
        layout.addStretch()

        self.combo = QComboBox()
        self.combo.wheelEvent = lambda ev: ev.ignore()
        self.combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo.setStyleSheet("""
            QComboBox {
                background: #14151a;
                color: #e2e8f0;
                border: 1px solid #2d303e;
                border-radius: 4px;
                padding: 3px 20px 3px 8px;
                font-size: 10.5px;
                min-width: 115px;
                max-width: 180px;
            }
            QComboBox:hover {
                border-color: #f59e0b;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #242632;
            }
            QComboBox QAbstractItemView {
                background: #181920;
                color: #e2e8f0;
                selection-background-color: #f59e0b;
                selection-color: #111;
                border: 1px solid #333644;
            }
        """)
        for text, val in items:
            self.combo.addItem(text, val)
        if default_val is not None:
            self.set_value(default_val)
        self.combo.currentIndexChanged.connect(self._on_index_changed)
        layout.addWidget(self.combo)

    def _on_index_changed(self, idx):
        self.valueChanged.emit(self.combo.currentData())

    def current_value(self):
        return self.combo.currentData()

    def set_value(self, val):
        idx = self.combo.findData(val)
        if idx >= 0:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(idx)
            self.combo.blockSignals(False)

    def reset_default(self):
        if self.default_val is not None:
            self.set_value(self.default_val)

