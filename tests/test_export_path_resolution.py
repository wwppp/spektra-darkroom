import os
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget
from ui.export_dialog import ExportImageDialog
from app_core import SpektraEngine


app = QApplication.instance() or QApplication([])


class TestExportPathResolution(unittest.TestCase):
    def setUp(self):
        self.dummy_widget = QWidget()
        self.engine = SpektraEngine()

    def tearDown(self):
        self.dummy_widget.deleteLater()

    def test_dialog_init_parent_as_second_argument(self):
        """Verify passing parent as second positional argument does not trigger batch mode."""
        dlg = ExportImageDialog("default.jpg", self.dummy_widget)
        self.assertFalse(dlg.is_batch)
        self.assertEqual(dlg.windowTitle(), "导出图像")
        self.assertEqual(dlg.parent(), self.dummy_widget)
        dlg.close()

    def test_dialog_batch_init_explicit(self):
        """Verify explicit is_batch=True works as expected."""
        dlg = ExportImageDialog("D:/photos", is_batch=True, batch_count=5, parent=self.dummy_widget)
        self.assertTrue(dlg.is_batch)
        self.assertEqual(dlg.batch_count, 5)
        self.assertIn("冲印参数设置", dlg.windowTitle())
        dlg.close()

    def test_get_export_config_resolves_directory_to_file(self):
        """Verify that typing or selecting a directory path automatically appends filename."""
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        try:
            dlg = ExportImageDialog("my_photo_developed.jpg", is_batch=False)
            dlg.edit_path.setText(tmp_dir)
            cfg = dlg.get_export_config()
            expected = os.path.join(tmp_dir, "my_photo_developed.jpg")
            self.assertEqual(os.path.normpath(cfg["path"]), os.path.normpath(expected))
            dlg.close()
        finally:
            os.rmdir(tmp_dir)

    def test_export_image_to_directory_path(self):
        """Verify export_image handles a directory path without throwing PermissionError."""
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create a small dummy raw_full image in engine
            self.engine.raw_preview = np.ones((50, 50, 3), dtype=np.float32) * 0.5
            self.engine.current_file_path = "D:/dummy/sample_pic.arw"

            params = {
                "exposure_ev": 0.0,
                "film_stock": "kodak_portra_400",
                "paper_stock": "kodak_2383"
            }
            # Pass tmp_dir directly as output_path
            res = self.engine.export_image(params, tmp_dir, format_type="jpeg")
            self.assertTrue(res.get("success"), f"Failed: {res.get('error')}")
            expected_file = os.path.join(tmp_dir, "sample_pic_developed.jpg")
            self.assertTrue(os.path.exists(expected_file))
        finally:
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_portrait_export_no_black_bars(self):
        """Verify that portrait RAW images export with correct orientation and no pillarbox black bars."""
        import glob
        from PIL import Image
        arws = glob.glob("**/*DSC02461.ARW", recursive=True)
        if not arws:
            return
        arw_path = arws[0]
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        try:
            out_file = os.path.join(tmp_dir, "test_portrait_developed.jpg")
            params = {
                "film_stock": "none",
                "paper_stock": "none",
                "exposure_ev": 0.0,
                "color_temp": 5500.0,
                "tint": 1.0,
                "scale_pct": 25  # Fast 25% scale down for test
            }
            res = self.engine.export_image(params, out_file, format_type="jpeg", source_path=arw_path)
            self.assertTrue(res.get("success"), f"Export failed: {res.get('error')}")
            self.assertTrue(os.path.exists(out_file))

            with Image.open(out_file) as im:
                w, h = im.size
                # Portrait orientation: height must be strictly greater than width (aspect ratio ~ 2:3)
                self.assertGreater(h, w, f"Expected portrait image (h > w), got {w}x{h}")
                self.assertAlmostEqual(h / float(w), 6024.0 / 4024.0, delta=0.05)

                arr = np.array(im)
                # Verify that the left, right, and top margins are NOT pure black borders
                left_col_mean = arr[:, :5].mean()
                right_col_mean = arr[:, -5:].mean()
                self.assertGreater(left_col_mean, 5.0, "Left column is pure black border!")
                self.assertGreater(right_col_mean, 5.0, "Right column is pure black border!")
        finally:
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

