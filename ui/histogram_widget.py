"""
histogram_widget.py - Professional Adobe/Lightroom-style Real-time RGB & Luma Histogram
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import Qt, QRectF
import numpy as np
from ui.theme_manager import theme_manager


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
        theme_manager.themeChanged.connect(lambda _: self.update())

        # 256-bin normalized curves (float 0.0 ~ 1.0)
        self._r_hist = np.zeros(256, dtype=np.float32)
        self._g_hist = np.zeros(256, dtype=np.float32)
        self._b_hist = np.zeros(256, dtype=np.float32)
        self._l_hist = np.zeros(256, dtype=np.float32)

        self._has_data = False
        self._shadow_clip = False
        self._highlight_clip = False
        self._last_rgb_array = None

        # Scale mode: "log" (Fotufilm/DaVinci standard) or "linear"
        self._scale_mode = "log"

    def set_scale_mode(self, mode: str):
        """Set scale mode to 'log' or 'linear'."""
        if mode in ("log", "linear") and self._scale_mode != mode:
            self._scale_mode = mode
            if self._last_rgb_array is not None:
                self.update_from_rgb(self._last_rgb_array)
            else:
                self.update()

    def toggle_scale_mode(self):
        """Toggle between Log and Linear scale modes."""
        self.set_scale_mode("linear" if self._scale_mode == "log" else "log")

    def mousePressEvent(self, event):
        """Click on the histogram to toggle between Log and Linear views."""
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_scale_mode()

    def clear(self):
        self._r_hist.fill(0)
        self._g_hist.fill(0)
        self._b_hist.fill(0)
        self._l_hist.fill(0)
        self._has_data = False
        self._shadow_clip = False
        self._highlight_clip = False
        self._last_rgb_array = None
        self.update()

    def update_from_rgb(self, rgb_array):
        """Compute 256-bin histogram from thumbnail RGB array [H, W, 3] uint8 or float32."""
        if rgb_array is None or rgb_array.size == 0:
            self.clear()
            return

        self._last_rgb_array = rgb_array

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
                peak = float(np.percentile(all_mid, 99.2))
                if peak <= 0:
                    peak = float(np.max(all_mid)) if np.max(all_mid) > 0 else 1.0
            else:
                peak = 1.0

            # Scale normalization: Logarithmic (Fotufilm/Resolve) vs Linear
            if self._scale_mode == "log":
                log_peak = float(np.log1p(peak))
                if log_peak <= 0:
                    log_peak = 1.0
                r_norm = np.log1p(r_counts) / log_peak
                g_norm = np.log1p(g_counts) / log_peak
                b_norm = np.log1p(b_counts) / log_peak
                l_norm = np.log1p(l_counts) / log_peak
            else:
                r_norm = r_counts / peak
                g_norm = g_counts / peak
                b_norm = b_counts / peak
                l_norm = l_counts / peak

            # Fotufilm standard 5-tap Gaussian smoothing kernel [1, 4, 6, 4, 1] / 16.0
            kernel = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float32)
            self._r_hist = np.clip(np.convolve(r_norm, kernel, mode='same'), 0.0, 1.0)
            self._g_hist = np.clip(np.convolve(g_norm, kernel, mode='same'), 0.0, 1.0)
            self._b_hist = np.clip(np.convolve(b_norm, kernel, mode='same'), 0.0, 1.0)
            self._l_hist = np.clip(np.convolve(l_norm, kernel, mode='same'), 0.0, 1.0)

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

        is_dark = theme_manager.is_dark()

        # Background card
        card_border = QColor("#262832") if is_dark else QColor("#d8dce6")
        card_bg = QColor("#131417") if is_dark else QColor("#ffffff")
        painter.setPen(QPen(card_border, 1.0))
        painter.setBrush(QBrush(card_bg))
        painter.drawRoundedRect(QRectF(pad_x, pad_y, plot_w, plot_h), 4.0, 4.0)

        # Reference grid lines (25%, 50%, 75% Exposure Zones)
        grid_pen = QColor(255, 255, 255, 18) if is_dark else QColor(0, 0, 0, 30)
        painter.setPen(QPen(grid_pen, 1.0, Qt.PenStyle.DashLine))
        for frac in (0.25, 0.50, 0.75):
            gx = pad_x + plot_w * frac
            painter.drawLine(gx, pad_y + 2, gx, pad_y + plot_h - 2)

        if not self._has_data:
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
        luma_fill = QColor(255, 255, 255, 30) if is_dark else QColor(30, 41, 59, 25)
        luma_stroke = QColor(226, 232, 240, 220) if is_dark else QColor(51, 65, 85, 200)
        channels = [
            (self._r_hist, QColor(239, 68, 68, 55), QColor(239, 68, 68, 190)),   # Red
            (self._g_hist, QColor(16, 185, 129, 55), QColor(16, 185, 129, 190)), # Green
            (self._b_hist, QColor(59, 130, 246, 55), QColor(59, 130, 246, 190)), # Blue
            (self._l_hist, luma_fill, luma_stroke)                                 # Luma
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

        # 4. Mode badge (LOG / LIN)
        badge_text = "LOG" if self._scale_mode == "log" else "LIN"
        badge_color = QColor(255, 255, 255, 100) if is_dark else QColor(30, 41, 59, 120)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(badge_color))
        painter.drawText(QRectF(pad_x + plot_w - 32.0, pad_y + 4.0, 26.0, 14.0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, badge_text)

