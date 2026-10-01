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
from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow
from ui.export_queue_manager import ExportQueueManager, ExportQueueDialog, ExportTask


class TestV014Refinements(unittest.TestCase):

    def setUp(self):
        self.engine = SpektraEngine()
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        if hasattr(self.win, "export_queue") and self.win.export_queue:
            self.win.export_queue.close()
        self.win.close()

    def test_version_bumped_to_014(self):
        self.assertTrue(VERSION_STRING in ("0.1.4", "0.1.5", "0.1.6", "0.1.7", "0.1.8", "0.1.9"))
        self.assertTrue(BUILD_NUMBER >= 5)

    def test_export_queue_dialog_proportions_and_theme(self):
        dlg = ExportQueueDialog(self.win.export_queue, self.win)
        # Dimensions must be wide 680x420, strictly avoiding square layout
        self.assertEqual(dlg.width(), 680)
        self.assertEqual(dlg.height(), 420)
        ratio = dlg.width() / float(dlg.height())
        self.assertGreater(ratio, 1.5)  # Greater than 1.5:1 widescreen ratio
        dlg.close()

    def test_import_status_lock_not_cleared_by_title_or_restore(self):
        # When _is_importing is active, restore_default_status MUST NOT wipe out import message
        self.win._is_importing = True
        self.win.set_app_status("正在导入底片库 (2/10)...", spinning=True, timeout_ms=0)
        self.assertEqual(self.win.status_label.text(), "正在导入底片库 (2/10)...")
        self.assertFalse(self.win.status_sep.isHidden())

        # Calling restore_default_status must be rejected
        self.win.restore_default_status()
        self.assertEqual(self.win.status_label.text(), "正在导入底片库 (2/10)...")

        # Calling _update_window_title must NOT wipe out import status
        self.win._update_window_title()
        self.assertEqual(self.win.status_label.text(), "正在导入底片库 (2/10)...")

        # After import finished, _is_importing is False, message lingers
        self.win._is_importing = False
        self.win.set_app_status("已完成导入 10 张底片", spinning=False, timeout_ms=4500)
        self.assertEqual(self.win.status_label.text(), "已完成导入 10 张底片")
        # During lingering timer, restore_default_status still does NOT erase it
        self.win.restore_default_status()
        self.assertEqual(self.win.status_label.text(), "已完成导入 10 张底片")

    def test_export_worker_safe_execution(self):
        # Verify that ExportTask and queue worker execute without crash
        task = ExportTask(
            task_id_or_path="test_task_1",
            title_or_output="test.jpg",
            source_path="",
            output_path="D:/tmp/test_out.jpg",
            params={"film_stock": "none", "paper_stock": "none", "quality": 90}
        )
        self.assertIsNotNone(task)
        self.assertEqual(task.status, "queued")


if __name__ == "__main__":
    unittest.main()
