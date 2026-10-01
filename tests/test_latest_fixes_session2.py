import os
import sys
import unittest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Ensure QApplication exists
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow
from ui.preferences_dialog import PreferencesDialog
from ui.stock_manager_dialog import StockManagerDialog
import config_manager
from path_utils import get_resource_dir


class TestLatestFixesSession2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res_dir = get_resource_dir()
        cls.engine = SpektraEngine(resources_dir=cls.res_dir)
        cls.win = DarkroomMainWindow(cls.engine)

    def test_01_start_page_text_no_jpg(self):
        """Verify that JPG is removed from start view supported formats."""
        # Find lbl_sub inside dropzone
        lbl_text = None
        for child in self.win.dropzone.children():
            if hasattr(child, "text") and "支持格式" in child.text():
                lbl_text = child.text()
                break
        self.assertIsNotNone(lbl_text, "Start view format label not found")
        self.assertNotIn("JPG", lbl_text, "JPG should not be listed in start view formats")
        self.assertIn("TIFF", lbl_text)
        self.assertIn("ARW", lbl_text)

    def test_02_32bit_float_tiff_loading(self):
        """Verify 32-bit linear float TIFF decodes without becoming pitch black."""
        tif_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "测试底片", "portrait_leaves_32bit_linear_prophoto_rgb.tif")
        if os.path.exists(tif_path):
            res = self.engine.load_image(tif_path)
            self.assertTrue(res.get("success"), f"Failed to load TIFF: {res.get('error')}")
            self.assertGreater(self.engine.raw_preview.max(), 0.5, "TIFF float values were crushed to black")
            th = self.engine.session_photos[-1]["thumbnail_rgb"]
            self.assertGreater(th.max(), 100, "Thumbnail was crushed to black")

    def test_03_stock_manager_redundant_button_removed(self):
        from PySide6.QtWidgets import QPushButton
        dlg = StockManagerDialog(mode="film", engine=self.engine)
        btn_texts = [b.text() for b in dlg.findChildren(QPushButton)]
        for t in btn_texts:
            self.assertNotIn("一键导出全部", t, "Redundant 一键导出全部 button still exists")
        dlg.close()

    def test_04_action_clear_photos_does_not_return_to_start_view(self):
        """Verify clearing filmstrip keeps user on darkroom screen and cleans canvas & state."""
        self.win.stack.setCurrentIndex(1)
        self.win.action_clear_photos()
        self.assertEqual(self.win.stack.currentIndex(), 1, "Should not return to start view (index 0)")
        self.assertEqual(len(self.win.photos), 0)
        self.assertFalse(self.win.canvas._has_image, "Canvas should not have image after clear")
        self.assertFalse(self.win.btn_title_export.isVisible())
        files, active = config_manager.get_session_state()
        self.assertEqual(len(files), 0, "Saved session files should be cleared")

    def test_05_preferences_dialog_options(self):
        """Verify exit confirmation checkbox is removed and restore_last_files checkbox is added."""
        dlg = PreferencesDialog(self.win)
        self.assertFalse(hasattr(dlg, "chk_confirm_exit"), "chk_confirm_exit should be removed")
        self.assertTrue(hasattr(dlg, "chk_restore_last_files"), "chk_restore_last_files should be present")
        self.assertEqual(dlg.chk_restore_last_files.text(), "启动时恢复上一次会话工程 (.sdss)")
        dlg.close()


if __name__ == "__main__":
    unittest.main()
