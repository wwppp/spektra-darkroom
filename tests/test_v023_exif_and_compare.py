import os
import sys
import unittest
import numpy as np
from PIL import Image
import tifffile

# Set headless platform before Qt import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import app_core
from PySide6.QtWidgets import QApplication, QMainWindow, QLineEdit, QWidget
from PySide6.QtCore import Qt, QEvent, QTimer, QPointF
from PySide6.QtGui import QKeyEvent, QMouseEvent
from ui.main_window import CompareButton, DarkroomMainWindow
from ui.canvas_viewport import DarkroomGLCanvas

app = QApplication.instance() or QApplication(sys.argv)

TEST_RAW = os.path.abspath("测试底片/DSC02539.ARW")


class TestExifExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = app_core.SpektraEngine()
        cls.temp_dir = os.path.abspath("test_exif_out")
        os.makedirs(cls.temp_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_extract_exif_bundle_from_raw(self):
        self.assertTrue(os.path.exists(TEST_RAW), f"Test RAW file not found: {TEST_RAW}")
        pil_exif, exif_bytes, tiff_tags = app_core.extract_exif_bundle(TEST_RAW)
        self.assertIsNotNone(pil_exif)
        self.assertIsNotNone(exif_bytes)
        self.assertTrue(len(tiff_tags) > 0)

        # Check orientation is normalized to 1
        self.assertEqual(pil_exif.get(0x0112), 1)
        orient_tag = next((t for t in tiff_tags if t[0] == 274), None)
        self.assertIsNotNone(orient_tag)
        self.assertEqual(orient_tag[1], 'I')
        self.assertEqual(orient_tag[3], 1)

        # Check sub IFD camera data
        sub = pil_exif.get_ifd(0x8769)
        self.assertEqual(pil_exif.get(0x010f), "SONY")
        self.assertEqual(pil_exif.get(0x0110), "ILCE-7C")
        self.assertEqual(sub.get(0x8827), 100) # ISO
        self.assertIn("17-35mm", sub.get(0xa434, "")) # LensModel

    def test_02_export_jpeg_preserves_exif(self):
        out_jpg = os.path.join(self.temp_dir, "test_export.jpg")
        params = {"film_profile": "none", "paper_profile": "none", "jpeg_quality": 92}
        ret = self.engine.export_image(
            params, out_jpg, format_type="jpg", source_path=TEST_RAW
        )
        self.assertTrue(ret.get("success"), f"Export failed: {ret.get('error')}")
        self.assertTrue(os.path.exists(out_jpg))

        with Image.open(out_jpg) as im:
            ex = im.getexif()
            self.assertEqual(ex.get(0x010f), "SONY")
            self.assertEqual(ex.get(0x0110), "ILCE-7C")
            self.assertEqual(ex.get(0x0112), 1) # Orientation normalized
            sub = ex.get_ifd(0x8769)
            self.assertEqual(sub.get(0x8827), 100)
            self.assertIn("17-35mm", sub.get(0xa434, ""))

    def test_03_export_png_preserves_exif(self):
        out_png = os.path.join(self.temp_dir, "test_export.png")
        params = {"film_profile": "none", "paper_profile": "none", "png_compression": 1}
        ret = self.engine.export_image(
            params, out_png, format_type="png", source_path=TEST_RAW
        )
        self.assertTrue(ret.get("success"), f"Export failed: {ret.get('error')}")
        self.assertTrue(os.path.exists(out_png))

        with Image.open(out_png) as im:
            ex = im.getexif()
            self.assertEqual(ex.get(0x010f), "SONY")
            self.assertEqual(ex.get(0x0110), "ILCE-7C")
            sub = ex.get_ifd(0x8769)
            self.assertEqual(sub.get(0x8827), 100)
            self.assertIn("17-35mm", sub.get(0xa434, ""))

    def test_04_export_tiff_16bit_preserves_exif(self):
        out_tif16 = os.path.join(self.temp_dir, "test_export_16.tif")
        params = {"film_profile": "none", "paper_profile": "none", "bit_depth": 16}
        ret = self.engine.export_image(
            params, out_tif16, format_type="tif", source_path=TEST_RAW
        )
        self.assertTrue(ret.get("success"), f"Export failed: {ret.get('error')}")
        self.assertTrue(os.path.exists(out_tif16))

        with tifffile.TiffFile(out_tif16) as tf:
            page = tf.pages[0]
            self.assertEqual(page.tags.get(271).value, "SONY")
            self.assertEqual(page.tags.get(272).value, "ILCE-7C")
            self.assertEqual(page.tags.get(34855).value, 100) # ISO
            self.assertIn("17-35mm", page.tags.get(42036).value) # LensModel

    def test_05_export_tiff_32bit_preserves_exif(self):
        out_tif32 = os.path.join(self.temp_dir, "test_export_32.tif")
        params = {"film_profile": "none", "paper_profile": "none", "bit_depth": 32}
        ret = self.engine.export_image(
            params, out_tif32, format_type="tif", source_path=TEST_RAW
        )
        self.assertTrue(ret.get("success"), f"Export failed: {ret.get('error')}")
        self.assertTrue(os.path.exists(out_tif32))

        with tifffile.TiffFile(out_tif32) as tf:
            page = tf.pages[0]
            self.assertEqual(page.tags.get(271).value, "SONY")
            self.assertEqual(page.tags.get(272).value, "ILCE-7C")
            self.assertEqual(page.tags.get(34855).value, 100)

    def test_06_export_tiff_8bit_preserves_exif(self):
        out_tif8 = os.path.join(self.temp_dir, "test_export_8.tif")
        params = {"film_profile": "none", "paper_profile": "none", "bit_depth": 8}
        ret = self.engine.export_image(
            params, out_tif8, format_type="tif", source_path=TEST_RAW
        )
        self.assertTrue(ret.get("success"), f"Export failed: {ret.get('error')}")
        self.assertTrue(os.path.exists(out_tif8))

        with tifffile.TiffFile(out_tif8) as tf:
            page = tf.pages[0]
            self.assertEqual(page.tags.get(271).value, "SONY")
            self.assertEqual(page.tags.get(272).value, "ILCE-7C")
            self.assertEqual(page.tags.get(34855).value, 100)

        with Image.open(out_tif8) as im:
            ex = im.getexif()
            self.assertEqual(ex.get(0x010f), "SONY")
            self.assertEqual(ex.get(0x0110), "ILCE-7C")
            self.assertEqual(ex.get(0x8827), 100)


class TestCompareButtonAndBackslash(unittest.TestCase):
    def setUp(self):
        os.environ["FAST_TEST"] = "1"
        self.engine = app_core.SpektraEngine()
        self.window = DarkroomMainWindow(self.engine)
        self.window._suppress_close_confirm = True
        self.window.show()

    def tearDown(self):
        self.window.close()

    def test_01_compare_button_short_click_emits_clicked(self):
        btn = CompareButton()
        events = []
        btn.clicked.connect(lambda: events.append("clicked"))
        btn.longPressStarted.connect(lambda: events.append("long_start"))
        btn.longPressFinished.connect(lambda: events.append("long_finish"))

        # Simulate quick press & release (50ms < 180ms)
        press_ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(10, 10), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        btn.mousePressEvent(press_ev)
        self.assertTrue(btn._press_timer.isActive())

        release_ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(10, 10), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        btn.mouseReleaseEvent(release_ev)

        self.assertFalse(btn._press_timer.isActive())
        self.assertEqual(events, ["clicked"])

    def test_02_compare_button_long_press_emits_signals_and_swallows_clicked(self):
        btn = CompareButton()
        events = []
        btn.clicked.connect(lambda: events.append("clicked"))
        btn.longPressStarted.connect(lambda: events.append("long_start"))
        btn.longPressFinished.connect(lambda: events.append("long_finish"))

        # Mouse press
        press_ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(10, 10), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        btn.mousePressEvent(press_ev)

        # Trigger timer timeout (simulating hold >= 180ms)
        btn._press_timer.timeout.emit()
        self.assertIn("long_start", events)
        self.assertTrue(btn._is_long_pressing)

        # Release mouse
        release_ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(10, 10), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        btn.mouseReleaseEvent(release_ev)

        self.assertIn("long_finish", events)
        # Verify clicked is NOT emitted so split view mode is not accidentally toggled!
        self.assertNotIn("clicked", events)
        self.assertFalse(btn._is_long_pressing)

    def test_03_compare_button_focus_out_during_long_press_recovers(self):
        btn = CompareButton()
        events = []
        btn.longPressStarted.connect(lambda: events.append("long_start"))
        btn.longPressFinished.connect(lambda: events.append("long_finish"))

        btn.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(10, 10), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        btn._press_timer.timeout.emit()
        self.assertTrue(btn._is_long_pressing)

        # Focus out
        from PySide6.QtGui import QFocusEvent
        btn.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
        self.assertFalse(btn._is_long_pressing)
        self.assertIn("long_finish", events)

    def test_04_backslash_short_tap_toggles_split_view(self):
        self.window._current_photo_path = TEST_RAW
        initial_mode = self.window.canvas.params.get("view_mode", 0)

        ev_press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier)
        ev_release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier)

        app.sendEvent(self.window, ev_press)
        self.assertTrue(self.window._backslash_pressed)
        app.sendEvent(self.window, ev_release)
        self.assertFalse(self.window._backslash_pressed)

        new_mode = self.window.canvas.params.get("view_mode", 0)
        self.assertEqual(new_mode, (initial_mode + 1) % 3)

    def test_05_backslash_long_press_enters_mode_3_and_reverts(self):
        self.window._current_photo_path = TEST_RAW
        self.window.set_view_mode(1) # Start in split view
        self.assertEqual(self.window.canvas.params.get("view_mode"), 1)

        ev_press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier)
        ev_release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier)

        app.sendEvent(self.window, ev_press)
        self.assertTrue(self.window._backslash_pressed)

        # Simulate timeout >= 180ms
        self.window._backslash_timer.timeout.emit()
        self.assertTrue(self.window._backslash_long_pressed)
        # Should now be in view_mode 3 (Before preview)
        self.assertEqual(self.window.canvas.params.get("view_mode"), 3)

        # Release key
        app.sendEvent(self.window, ev_release)
        self.assertFalse(self.window._backslash_pressed)
        self.assertFalse(self.window._backslash_long_pressed)
        # Restored to split view mode 1
        self.assertEqual(self.window.canvas.params.get("view_mode"), 1)

    def test_06_backslash_in_lineedit_is_ignored(self):
        edit = QLineEdit(self.window)
        edit.show()
        initial_mode = self.window.canvas.params.get("view_mode", 0)

        ev_press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier, "\\")
        ev_release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier)

        app.sendEvent(edit, ev_press)
        self.assertFalse(self.window._backslash_pressed)
        app.sendEvent(edit, ev_release)
        # View mode should remain unchanged
        self.assertEqual(self.window.canvas.params.get("view_mode", 0), initial_mode)

    def test_07_canvas_view_mode_3_hud_badge(self):
        canvas = self.window.canvas
        canvas._has_image = True
        canvas.resize(800, 600)
        canvas.set_view_mode(3)
        self.assertEqual(canvas.params.get("view_mode"), 3)
        canvas._update_badges()
        self.assertFalse(canvas.badge_before.isHidden())
        self.assertEqual(canvas.badge_before.text(), "原片 (Before)")
        self.assertTrue(canvas.badge_after.isHidden())


if __name__ == "__main__":
    unittest.main()
