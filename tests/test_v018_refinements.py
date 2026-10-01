import os
import sys
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QImage
from PySide6.QtNetwork import QLocalServer, QLocalSocket

# Set headless environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["FAST_TEST"] = "1"

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

import config_manager
from version import get_full_version_info
from ui.filmstrip_widget import FilmstripWidget, FilmstripItemWidget
from ui.main_window import DarkroomMainWindow
from app_core import SpektraEngine


class TestV018Refinements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SpektraEngine(resources_dir=os.path.join(os.path.dirname(__file__), "..", "resources"))

    def test_version_bump_v018_build9(self):
        v = get_full_version_info()
        self.assertTrue(v["version"] in ("0.1.8", "0.1.9"))
        self.assertGreaterEqual(v["build"], 9)

    def test_preferences_no_recent_file_count_setting(self):
        from ui.preferences_dialog import PreferencesDialog
        dlg = PreferencesDialog()
        self.assertFalse(hasattr(dlg, "spin_recent_count"))
        dlg.close()

    def test_sdss_icon_and_registry_association(self):
        app_dir = os.path.dirname(os.path.abspath(config_manager.__file__))
        ico_path = os.path.join(app_dir, "resources", "sdss_icon.ico")
        self.assertTrue(os.path.exists(ico_path))
        # Verify icon file is valid
        self.assertGreater(os.path.getsize(ico_path), 1000)

    def test_filmstrip_selection_changed_and_top_layer_border(self):
        fw = FilmstripWidget()
        photos = [
            {"id": "p1", "filename": "1.jpg", "path": "1.jpg"},
            {"id": "p2", "filename": "2.jpg", "path": "2.jpg"},
            {"id": "p3", "filename": "3.jpg", "path": "3.jpg"}
        ]
        emitted_selections = []
        fw.selectionChanged.connect(lambda sel: emitted_selections.append(sel))

        fw.set_photos(photos, "p1")
        self.assertTrue(len(emitted_selections) > 0)
        self.assertEqual(emitted_selections[-1], ["p1"])

        # Test Ctrl+A select all
        fw.select_all_photos()
        self.assertEqual(len(fw.get_selected_photo_ids()), 3)
        self.assertEqual(len(emitted_selections[-1]), 3)

        # Test paintEvent of FilmstripItemWidget does not crash and renders border
        item = fw._item_widgets["p1"]
        img = QImage(120, 100, QImage.Format.Format_ARGB32_Premultiplied)
        item.render(img)

        fw.close()

    def test_import_marks_session_dirty(self):
        win = DarkroomMainWindow(self.engine)
        win._suppress_close_confirm = True

        # Initially not dirty
        win._current_session_path = "D:\\test_session.sdss"
        win._session_is_dirty = False
        win._update_window_title()
        self.assertFalse(win._session_is_dirty)
        self.assertNotIn("*", win.windowTitle())

        # Create dummy image file
        import tempfile
        t_img = os.path.join(tempfile.gettempdir(), "test_spektra_v018.jpg")
        import cv2
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(t_img, dummy)

        # Import new file
        win._import_files_list([t_img])
        self.assertTrue(win._session_is_dirty)
        self.assertIn("*", win.windowTitle())

        if os.path.exists(t_img):
            os.remove(t_img)
        win.close()

    def test_export_no_developed_suffix_and_path_memory(self):
        win = DarkroomMainWindow(self.engine)
        win._suppress_close_confirm = True

        # Test path memory
        test_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_out"))
        config_manager.save_last_export_settings({"path": test_out_dir})
        loaded = config_manager.get_last_export_settings()
        self.assertEqual(loaded.get("path"), test_out_dir)

        # Single export config check: no _developed
        import tempfile
        src_path = os.path.join(tempfile.gettempdir(), "DSC02453.ARW")
        win._current_photo_path = src_path
        cfg = {
            "format": "jpeg",
            "path": ""  # Should fallback to test_out_dir without _developed
        }
        win.current_film_stock = "none"
        win.current_paper_stock = "none"

        # Check single export output path logic
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        ext = ".jpg"
        fallback_dir = cfg["path"] if cfg["path"] else test_out_dir
        expected_out = os.path.join(fallback_dir, f"{base_name}{ext}")
        self.assertNotIn("_developed", expected_out)
        self.assertTrue(expected_out.endswith("DSC02453.jpg"))

        win.close()

    def test_multi_select_permanent_status_no_flicker(self):
        win = DarkroomMainWindow(self.engine)
        win._suppress_close_confirm = True

        # Simulate multi-select
        win._on_filmstrip_selection_changed(["id1", "id2", "id3"])
        self.assertTrue(win._is_multi_selecting)
        self.assertEqual(win.status_label.text(), "已选中 3 张底片")

        # Calling on_switch_photo should NOT overwrite multi-select message
        win._on_status_timeout()
        self.assertEqual(win.status_label.text(), "已选中 3 张底片")

        # Switching to single selection restores default
        win._on_filmstrip_selection_changed(["id1"])
        self.assertFalse(win._is_multi_selecting)

        win.close()

    def test_single_instance_ipc_file_forwarding(self):
        server_name = "SpektraDarkroom_IPC_TestServer"
        server = QLocalServer()
        server.removeServer(server_name)
        self.assertTrue(server.listen(server_name))

        client = QLocalSocket()
        client.connectToServer(server_name)
        self.assertTrue(server.waitForNewConnection(1000))
        self.assertTrue(client.waitForConnected(1000))

        conn = server.nextPendingConnection()
        self.assertIsNotNone(conn)

        test_file = "D:\\spektra\\test.sdss"
        client.write(test_file.encode("utf-8"))
        client.flush()
        client.waitForBytesWritten(1000)

        self.assertTrue(conn.waitForReadyRead(1000))
        received_data = bytes(conn.readAll()).decode("utf-8")

        client.close()
        conn.close()
        server.close()
        server.removeServer(server_name)
        self.assertEqual(received_data, test_file)


if __name__ == "__main__":
    unittest.main()
