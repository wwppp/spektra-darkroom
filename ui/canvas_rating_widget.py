"""Canvas Rating Widget for SpektraDarkroom
Interactive 5-star rating widget embedded directly in the canvas view toolbar.
Supports mouse hover preview, click-to-rate, click-again-to-clear, and dark/light theme styling.
"""

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QFont, QCursor

from ui.theme_manager import theme_manager


class CanvasRatingWidget(QWidget):
    """Interactive 5-star rating widget in the view toolbar.
    - Displays active photo rating (0-5 stars)
    - Hover preview highlights stars up to mouse cursor
    - Click sets rating (or clears if clicking currently active star)
    """
    ratingChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rating = 0
        self._hover_rating = 0
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(92, 22)
        self.setToolTip("底片评级 (快捷键 1-5 评分，0 清除)")
        theme_manager.themeChanged.connect(lambda _: self.update())

    def get_rating(self) -> int:
        return self._rating

    def set_rating(self, rating: int):
        clean_r = max(0, min(5, int(rating or 0)))
        if self._rating != clean_r:
            self._rating = clean_r
            self.update()

    def _get_star_index_at(self, x: float) -> int:
        w = self.width()
        star_w = w / 5.0
        idx = int(x / star_w) + 1
        return max(1, min(5, idx))

    def mouseMoveEvent(self, event):
        idx = self._get_star_index_at(event.position().x())
        if self._hover_rating != idx:
            self._hover_rating = idx
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover_rating != 0:
            self._hover_rating = 0
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._get_star_index_at(event.position().x())
            if idx == self._rating:
                new_rating = 0  # Toggle off
            else:
                new_rating = idx
            self.set_rating(new_rating)
            self.ratingChanged.emit(new_rating)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        is_dark = theme_manager.is_dark()
        w = self.width()
        h = self.height()
        star_w = w / 5.0

        # Colors
        active_color = QColor("#f59e0b")  # Amber gold
        hover_color = QColor("#fbbf24") if is_dark else QColor("#d97706")
        empty_color = QColor("#3b4252") if is_dark else QColor("#cbd5e1")

        font = QFont("Segoe UI Symbol", 11)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        painter.setFont(font)

        eff_rating = self._hover_rating if self._hover_rating > 0 else self._rating

        for i in range(5):
            star_idx = i + 1
            rect = QRectF(i * star_w, 0, star_w, h)
            if self._hover_rating > 0:
                if star_idx <= self._hover_rating:
                    painter.setPen(hover_color)
                else:
                    painter.setPen(empty_color)
            else:
                if star_idx <= self._rating:
                    painter.setPen(active_color)
                else:
                    painter.setPen(empty_color)

            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "★")

        painter.end()
