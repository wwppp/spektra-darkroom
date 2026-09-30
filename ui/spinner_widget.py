from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget


class SpinnerWidget(QWidget):
    """Sleek Adobe-style rotating loading spinner circle widget.
    - When active (loading/processing/rendering): rotates an amber arc smoothly at 30 FPS.
    - When idle (ready): displays a subtle calm green dot indicator.
    """
    def __init__(self, size=14, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self._angle = 0
        self._is_spinning = False

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._on_tick)

    def start(self):
        """Start spinning animation."""
        self._is_spinning = True
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def stop(self):
        """Stop spinning animation and return to idle green indicator."""
        self._is_spinning = False
        if self._timer.isActive():
            self._timer.stop()
        self.update()

    def is_spinning(self) -> bool:
        return self._is_spinning

    def _on_tick(self):
        self._angle = (self._angle + 16) % 360
        self.update()

    def paintEvent(self, event):
        if not self._is_spinning:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        m = 2.0
        rect = QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)

        # Background track ring
        pen_bg = QPen(QColor("#2b2b2b"), 1.8)
        painter.setPen(pen_bg)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(rect, 0, 360 * 16)

        # Rotating highlighted arc (Amber / Gold)
        pen_arc = QPen(QColor("#f59e0b"), 2.0)
        pen_arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arc)
        painter.drawArc(rect, int(-self._angle * 16), int(110 * 16))
