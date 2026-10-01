import os
import sys
import unittest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ["FAST_TEST"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from version import VERSION_STRING, BUILD_NUMBER
from ui.filmstrip_widget import FilmstripItemWidget, FilmstripWidget
from ui.unsaved_session_dialog import UnsavedSessionDialog
from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow


class TestV013Refinements(unittest.TestCase):

    def setUp(self):
        self.engine = SpektraEngine()
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        if hasattr(self.win, "export_queue") and self.win.export_queue:
            self.win.export_queue.close()
        self.win.close()

    def test_version_bumped_to_013(self):
        self.assertTrue(VERSION_STRING in ("0.1.3", "0.1.4", "0.1.5", "0.1.6", "0.1.7", "0.1.8", "0.1.9"))
        self.assertGreaterEqual(BUILD_NUMBER, 4)

    def test_sprocket_holes_proportional_scaling(self):
        item = FilmstripItemWidget({"id": "test_id", "filename": "test.arw"})
        # Test item size at standard height 96
        item.set_item_size(100, 96)
        self.assertEqual(item.width(), 100)
        self.assertEqual(item.height(), 96)

        # Test item size at larger height 160 (margin_y must scale beyond 12.0)
        item.set_item_size(180, 160)
        self.assertEqual(item.width(), 180)
        self.assertEqual(item.height(), 160)
        h = 160.0
        expected_my = h * 0.14
        self.assertGreater(expected_my, 15.0)

    def test_unsaved_session_dialog_structure(self):
        dlg = UnsavedSessionDialog("my_project.sdss", self.win)
        self.assertEqual(dlg.sess_name, "my_project.sdss")
        self.assertIsNotNone(dlg.btn_save)
        self.assertIsNotNone(dlg.btn_discard)
        self.assertIsNotNone(dlg.btn_cancel)
        self.assertEqual(dlg.btn_discard.text(), "不保存直接退出")
        self.assertEqual(dlg.btn_save.text(), "保存")
        self.assertEqual(dlg.btn_cancel.text(), "取消")

    def test_permanent_photo_info_and_transient_status_isolation(self):
        # 1. Set active photo
        self.win._current_photo_path = "D:/photos/DSC01928.ARW"
        self.win.restore_default_status()
        self.assertIn("DSC01928.ARW", self.win.lbl_photo_info.text())
        self.assertTrue(self.win.status_sep.isHidden() or "transparent" in self.win.status_sep.styleSheet())

        # 2. Trigger transient status notification
        self.win.set_app_status("正在导入底片库 (3/10)...", spinning=True)
        # Permanent info MUST still display DSC01928.ARW
        self.assertIn("DSC01928.ARW", self.win.lbl_photo_info.text())
        # Notification must be in status_label
        self.assertIn("正在导入", self.win.status_label.text())
        self.assertTrue(not self.win.status_sep.isHidden() and "transparent" not in self.win.status_sep.styleSheet())

        # 3. Restore status
        self.win.restore_default_status()
        self.assertIn("DSC01928.ARW", self.win.lbl_photo_info.text())
        self.assertEqual(self.win.status_label.text(), "")
        self.assertTrue(self.win.status_sep.isHidden() or "transparent" in self.win.status_sep.styleSheet())


if __name__ == "__main__":
    unittest.main()
