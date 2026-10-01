import os
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from ui.main_window import DarkroomMainWindow
from ui.canvas_viewport import DarkroomGLCanvas
from app_core import SpektraEngine

app = QApplication.instance() or QApplication([])

class TestGrainSizeControl(unittest.TestCase):
    def setUp(self):
        self.engine = SpektraEngine()
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()

    def test_slider_grain_size_exists_and_defaults(self):
        """Verify slider_grain_size exists on MainWindow with correct defaults."""
        self.assertTrue(hasattr(self.win, "slider_grain_size"))
        self.assertEqual(self.win.slider_grain_size.value(), 1.0)
        self.assertEqual(self.win.current_params.get("grain_size"), 1.0)

    def test_slider_grain_size_updates_params(self):
        """Verify moving grain size slider updates current_params and canvas."""
        self.win.slider_grain_size.set_value(1.75)
        self.assertEqual(self.win.current_params.get("grain_size"), 1.75)
        self.assertEqual(self.win.canvas.params.get("grain_size"), 1.75)

    def test_reset_track_resets_grain_size(self):
        """Verify reset_track('texture') resets grain_size to 1.0."""
        self.win.slider_grain_size.set_value(2.2)
        self.win.reset_track("texture")
        self.assertEqual(self.win.slider_grain_size.value(), 1.0)
        self.assertEqual(self.win.current_params.get("grain_size"), 1.0)

    def test_export_image_with_grain_size(self):
        """Verify export_image handles grain_size parameter smoothly."""
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        try:
            self.engine.raw_preview = np.ones((50, 50, 3), dtype=np.float32) * 0.5
            self.engine.current_file_path = "D:/dummy/test_grain_size.arw"

            params = {
                "exposure_ev": 0.0,
                "film_stock": "kodak_portra_400",
                "paper_stock": "kodak_2383",
                "grain": 0.6,
                "grain_size": 1.8,
                "grain_cloud_blur": 1.2
            }
            out_file = os.path.join(tmp_dir, "test_out.jpg")
            res = self.engine.export_image(params, out_file, format_type="jpeg")
            self.assertTrue(res.get("success"), f"Export failed: {res.get('error')}")
            self.assertTrue(os.path.exists(out_file))
            if os.path.exists(out_file):
                os.remove(out_file)
        finally:
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)


if __name__ == '__main__':
    unittest.main()
