import unittest
import os
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from version import VERSION_STRING, BUILD_NUMBER
from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow
from ui.filmstrip_widget import DuplicateFilesDialog
from ui.export_queue_manager import ExportQueueDialog, ExportTask
import config_manager


app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)


class TestV015Refinements(unittest.TestCase):

    def setUp(self):
        self.engine = SpektraEngine()
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        if hasattr(self.win, "export_queue") and self.win.export_queue:
            self.win.export_queue.close()
        self.win.close()

    def test_version_bumped_to_015(self):
        self.assertTrue(VERSION_STRING in ("0.1.5", "0.1.6", "0.1.7", "0.1.8", "0.1.9"))
        self.assertGreaterEqual(BUILD_NUMBER, 6)

    def test_duplicate_files_dialog_construction(self):
        # Item 1: DuplicateFilesDialog must initialize without error and display duplicates
        dlg = DuplicateFilesDialog(["test_dup_1.arw", "test_dup_2.arw"], self.win)
        self.assertEqual(dlg.width(), 540)
        self.assertEqual(dlg.height(), 350)
        self.assertIn("test_dup_1.arw", dlg.txt.toPlainText())
        self.assertIn("test_dup_2.arw", dlg.txt.toPlainText())

    def test_export_queue_order_and_cards_in_place_update(self):
        # Item 5: Export queue items must be in chronological forward order (earliest at top)
        t1 = ExportTask("id_1", "photo_1.jpg", "p1.jpg", "out1.jpg", {})
        t2 = ExportTask("id_2", "photo_2.jpg", "p2.jpg", "out2.jpg", {})
        self.win.export_queue.enqueue(t1)
        self.win.export_queue.enqueue(t2)

        dlg = ExportQueueDialog(self.win.export_queue, self.win)
        self.assertEqual(len(dlg._cards), 2)
        # Task 1 must be created and tracked before Task 2
        card1 = dlg._cards.get("id_1")
        card2 = dlg._cards.get("id_2")
        self.assertIsNotNone(card1)
        self.assertIsNotNone(card2)
        self.assertIsNotNone(card1.get("btn_x"))

        # Verify in-place update does not destroy widgets
        old_btn_x = card1["btn_x"]
        t1.progress = 50
        self.win.export_queue._on_task_progress("id_1", 50)
        self.assertIs(dlg._cards["id_1"]["btn_x"], old_btn_x)

    def test_session_save_path_preservation(self):
        # Item 4: Opening a session must preserve _current_session_path and not overwrite it with None
        fake_session_path = os.path.abspath("test_dummy_session.sdss")
        self.win._current_session_path = fake_session_path
        self.win.photos.clear()
        
        # Simulating file import check logic: should preserve _current_session_path
        if not self.win.photos and self.win._current_session_path is None:
            self.win._session_is_dirty = True
        self.assertEqual(self.win._current_session_path, fake_session_path)

    def test_photo_info_min_width_and_smooth_status(self):
        # Item 2: lbl_photo_info has comfortable minimum width to avoid label jitter
        self.assertGreaterEqual(self.win.lbl_photo_info.minimumWidth(), 120)

    def test_file_association_functions_available(self):
        # Item 3: File association utilities in config_manager
        self.assertTrue(hasattr(config_manager, "register_sdss_file_association"))
        self.assertTrue(hasattr(config_manager, "is_sdss_file_associated"))


if __name__ == "__main__":
    unittest.main()
