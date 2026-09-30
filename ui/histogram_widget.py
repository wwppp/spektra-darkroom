"""
histogram_widget.py - Professional Adobe/Lightroom-style Real-time RGB & Luma Histogram
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import Qt, QRectF
import numpy as np


class HistogramWidget(QWidget):
    """Real-time RGB and Luminance Histogram:
    - 256-bin fast histogram calculation with 99.5% peak clipping mitigation
    - Antialiased overlapping Red, Green, Blue, and Luma paths
    - Additive/semi-transparent gradients
    - Highlight & Shadow clipping indicators (over/under exposure)
    - Darkroom theme styling with IRE reference grid lines
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(88)
        self.setMinimumWidth(220)
        self.setStyleSheet("background: transparent;")

        # 256-bin normalized curves (float 0.0 ~ 1.0)
        self._r_hist = np.zeros(256, dtype=np.float32)
        self._g_hist = np.zeros(256, dtype=np.float32)
        self._b_hist = np.zeros(256, dtype=np.float32)
        self._l_hist = np.zeros(256, dtype=np.float32)

        self._has_data = False
        self._shadow_clip = False
        self._highlight_clip = False

    def clear(self):
        self._r_hist.fill(0)
        self._g_hist.fill(0)
        self._b_hist.fill(0)
        self._l_hist.fill(0)
        self._has_data = False
        self._shadow_clip = False
        self._highlight_clip = False
        self.update()

    def update_from_rgb(self, rgb_array):
        """Compute 256-bin histogram from thumbnail RGB array [H, W, 3] uint8 or float32."""
        if rgb_array is None or rgb_array.size == 0:
            self.clear()
            return

        try:
            if rgb_array.dtype != np.uint8:
                arr_u8 = np.clip(rgb_array * 255.0, 0, 255).astype(np.uint8)
            else:
                arr_u8 = rgb_array

            r = arr_u8[..., 0].ravel()
            g = arr_u8[..., 1].ravel()
            b = arr_u8[..., 2].ravel()

            # Fast 256-bin counting
            r_counts = np.bincount(r, minlength=256)[:256].astype(np.float32)
            g_counts = np.bincount(g, minlength=256)[:256].astype(np.float32)
            b_counts = np.bincount(b, minlength=256)[:256].astype(np.float32)

            # Luma (ITU-R BT.601 / Rec.709 approximation)
            luma = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
            l_counts = np.bincount(luma, minlength=256)[:256].astype(np.float32)

            total_px = float(len(r))
            if total_px > 0:
                self._shadow_clip = (l_counts[0] / total_px) > 0.015
                self._highlight_clip = (l_counts[255] / total_px) > 0.015

            # 99.2th percentile peak clipping so solid backdrop does not crush contrast
            all_mid = np.concatenate([r_counts[1:255], g_counts[1:255], b_counts[1:255]])
            if len(all_mid) > 0:
                peak = np.percentile(all_mid, 99.2)
                if peak <= 0:
                    peak = np.max(all_mid) if np.max(all_mid) > 0 else 1.0
            else:
                peak = 1.0

            # 3-tap moving average box filter for smooth photographic curves
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            self._r_hist = np.clip(np.convolve(r_counts / peak, kernel, mode='same'), 0.0, 1.0)
            self._g_hist = np.clip(np.convolve(g_counts / peak, kernel, mode='same'), 0.0, 1.0)
            self._b_hist = np.clip(np.convolve(b_counts / peak, kernel, mode='same'), 0.0, 1.0)
            self._l_hist = np.clip(np.convolve(l_counts / peak, kernel, mode='same'), 0.0, 1.0)

            self._has_data = True
            self.update()
        except Exception:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())
        pad_x = 8.0
        pad_y = 6.0
        plot_w = w - pad_x * 2.0
        plot_h = h - pad_y * 2.0

        # Background card
        painter.setPen(QPen(QColor("#262832"), 1.0))
        painter.setBrush(QBrush(QColor("#131417")))
        painter.drawRoundedRect(QRectF(pad_x, pad_y, plot_w, plot_h), 4.0, 4.0)

        # Reference grid lines (25%, 50%, 75% Exposure Zones)
        painter.setPen(QPen(QColor(255, 255, 255, 18), 1.0, Qt.PenStyle.DashLine))
        for frac in (0.25, 0.50, 0.75):
            gx = pad_x + plot_w * frac
            painter.drawLine(gx, pad_y + 2, gx, pad_y + plot_h - 2)

        if not self._has_data:
            painter.setPen(QColor("#4e5264"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(pad_x, pad_y, plot_w, plot_h), Qt.AlignmentFlag.AlignCenter, "直方图 (载入底片后显示)")
            return

        # Helper to construct QPainterPath
        def make_path(data):
            path = QPainterPath()
            base_y = pad_y + plot_h
            path.moveTo(pad_x, base_y)
            for i in range(256):
                x = pad_x + (float(i) / 255.0) * plot_w
                y = base_y - float(data[i]) * (plot_h - 4.0)
                path.lineTo(x, y)
            path.lineTo(pad_x + plot_w, base_y)
            path.closeSubpath()
            return path

        def make_stroke_path(data):
            path = QPainterPath()
            base_y = pad_y + plot_h
            path.moveTo(pad_x, base_y - float(data[0]) * (plot_h - 4.0))
            for i in range(1, 256):
                x = pad_x + (float(i) / 255.0) * plot_w
                y = base_y - float(data[i]) * (plot_h - 4.0)
                path.lineTo(x, y)
            return path

        # Channels rendering with semi-transparent additive look
        channels = [
            (self._r_hist, QColor(239, 68, 68, 55), QColor(239, 68, 68, 190)),   # Red
            (self._g_hist, QColor(16, 185, 129, 55), QColor(16, 185, 129, 190)), # Green
            (self._b_hist, QColor(59, 130, 246, 55), QColor(59, 130, 246, 190)), # Blue
            (self._l_hist, QColor(255, 255, 255, 30), QColor(226, 232, 240, 220))# Luma
        ]

        # 1. Fill areas
        painter.setPen(Qt.PenStyle.NoPen)
        for data, fill_color, _ in channels:
            fill_path = make_path(data)
            painter.setBrush(QBrush(fill_color))
            painter.drawPath(fill_path)

        # 2. Outline strokes
        for data, _, stroke_color in channels:
            stroke_path = make_stroke_path(data)
            painter.setPen(QPen(stroke_color, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(stroke_path)

        # 3. Clipping warnings
        if self._shadow_clip:
            painter.setBrush(QBrush(QColor("#38bdf8")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(pad_x + 5.0, pad_y + 5.0, 5.0, 5.0))

        if self._highlight_clip:
            painter.setBrush(QBrush(QColor("#ef4444")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(pad_x + plot_w - 10.0, pad_y + 5.0, 5.0, 5.0))
