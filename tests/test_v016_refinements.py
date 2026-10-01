import unittest
import os
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint

from version import VERSION_STRING, BUILD_NUMBER
from ui.adobe_scrub_slider import ScrubLabel, AdobeScrubSlider


app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)


class TestV016Refinements(unittest.TestCase):

    def test_version_bumped_to_016(self):
        self.assertTrue(VERSION_STRING in ("0.1.6", "0.1.7", "0.1.8", "0.1.9"))
        self.assertGreaterEqual(BUILD_NUMBER, 7)

    def test_scrub_label_properties_and_modifiers(self):
        label = ScrubLabel("曝光")
        self.assertEqual(label.text(), "曝光")
        self.assertEqual(label.cursor().shape(), Qt.CursorShape.SizeHorCursor)

        received_deltas = []
        label.scrubDelta.connect(lambda d: received_deltas.append(d))

        # Test normal delta emission
        label._dragging = True
        label._last_global_x = 100.0

        # Simulate move without modifiers
        from PySide6.QtGui import QMouseEvent
        event_norm = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPoint(50, 10),
            QPoint(120, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        label.mouseMoveEvent(event_norm)
        self.assertEqual(len(received_deltas), 1)
        self.assertAlmostEqual(received_deltas[-1], 20.0, places=2)

        # Simulate move with Ctrl (Ultra-fine 0.20x)
        event_ctrl = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPoint(50, 10),
            QPoint(140, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier
        )
        label.mouseMoveEvent(event_ctrl)
        self.assertEqual(len(received_deltas), 2)
        # delta = 140 - 120 = 20, 20 * 0.20 = 4.0
        self.assertAlmostEqual(received_deltas[-1], 4.0, places=2)

        # Simulate move with Shift (Accelerated 2.50x)
        event_shift = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPoint(50, 10),
            QPoint(160, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier
        )
        label.mouseMoveEvent(event_shift)
        self.assertEqual(len(received_deltas), 3)
        # delta = 160 - 140 = 20, 20 * 2.50 = 50.0
        self.assertAlmostEqual(received_deltas[-1], 50.0, places=2)

        label._dragging = False

    def test_adobe_scrub_slider_smooth_accumulator_and_calibrated_rate(self):
        slider = AdobeScrubSlider("曝光", -3.0, 3.0, 0.0, step=0.05, unit=" EV", decimals=2)
        self.assertEqual(slider.value(), 0.0)

        # Micro-adjustment rate calibrated: 0.05 * step per pixel
        # Moving 20 pixels gives 20 * 0.05 * 0.05 = 0.05 EV
        slider._on_scrub(20.0)
        self.assertAlmostEqual(slider.value(), 0.05, places=2)

        # High-precision small delta accumulation without precision truncation
        # Moving 2 pixels gives 2 * 0.05 * 0.05 = 0.005 EV
        slider._on_scrub(2.0)
        self.assertAlmostEqual(slider._precise_val, 0.055, places=3)
        # Moving another 2 pixels: total +0.01 EV
        slider._on_scrub(2.0)
        self.assertAlmostEqual(slider.value(), 0.06, places=2)


if __name__ == "__main__":
    unittest.main()
