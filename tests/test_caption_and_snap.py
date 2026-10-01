import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import unittest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QMouseEvent, QPainter, QImage
import sys

from app_core import SpektraEngine
from ui.main_window import WindowCaptionButton, DraggableTitleBar, DarkroomMainWindow

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

class TestCaptionAndSnap(unittest.TestCase):
    def setUp(self):
        self.btn = WindowCaptionButton("max")
        self.btn.resize(46, 32)

    def test_caption_button_paint_without_ghosting(self):
        """Verify paintEvent executes cleanly and renders into pixel buffer without errors."""
        img = QImage(46, 32, QImage.Format.Format_ARGB32)
        img.fill(0)
        self.btn.render(img)

        # Check maximized toggle
        self.btn.set_maximized_state(True)
        self.assertTrue(self.btn.is_maximized_state)
        
        img2 = QImage(46, 32, QImage.Format.Format_ARGB32)
        self.btn.render(img2)

    def test_draggable_titlebar_snap_protection(self):
        """Verify dragging to screen top does not trigger accidental un-maximize."""
        engine = SpektraEngine()
        mw = DarkroomMainWindow(engine)
        mw.resize(1200, 800)
        tb = mw.title_bar

        # 1. Normal drag starting when window is NOT maximized
        self.assertFalse(mw.isMaximized())
        press_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(100, 15),
            QPointF(100, 15),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        tb.mousePressEvent(press_event)
        self.assertFalse(tb._drag_started_while_maximized)

        # 2. Simulate Windows Aero Snap maximizing the window during drag
        mw.showMaximized()
        self.assertTrue(mw.isMaximized())

        # 3. mouseMoveEvent occurs while window is maximized, but drag started while normal
        move_event = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(100, 45),
            QPointF(100, 45),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        tb.mouseMoveEvent(move_event)

        # Window MUST REMAIN MAXIMIZED (protection against Aero Snap shrink bug)
        self.assertTrue(mw.isMaximized())

        # Release mouse
        release_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(100, 45),
            QPointF(100, 45),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier
        )
        tb.mouseReleaseEvent(release_event)
        self.assertIsNone(tb._press_pos)
        self.assertFalse(tb._drag_started_while_maximized)

        mw.close()

if __name__ == '__main__':
    unittest.main()
