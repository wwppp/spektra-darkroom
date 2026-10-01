import os
import sys
import unittest
import tempfile
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QMouseEvent

# Set headless environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["FAST_TEST"] = "1"

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

import config_manager
import session_cache_manager
from version import get_full_version_info
from ui.filmstrip_widget import FilmstripWidget
from ui.main_window import DarkroomMainWindow
from ui.preferences_dialog import PreferencesDialog
from app_core import SpektraEngine


class TestV019Refinements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SpektraEngine(resources_dir=os.path.join(os.path.dirname(__file__), "..", "resources"))

    def test_version_bump_v019_build10(self):
        v = get_full_version_info()
        self.assertEqual(v["version"], "0.1.9")
        self.assertEqual(v["build"], 10)

    def test_ctrl_click_does_not_switch_active_photo(self):
        fw = FilmstripWidget()
        photos = [
            {"id": "p1", "filename": "1.jpg", "path": "1.jpg"},
            {"id": "p2", "filename": "2.jpg", "path": "2.jpg"},
            {"id": "p3", "filename": "3.jpg", "path": "3.jpg"}
        ]
        selected_switches = []
        fw.photoSelected.connect(lambda pid: selected_switches.append(pid))
        fw.set_photos(photos, "p1")
        self.assertEqual(fw._active_id, "p1")
        self.assertEqual(len(selected_switches), 0)

        # Create simulated Ctrl+Left Click event on p2
        evt_ctrl = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            QPointF(10, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier
        )

        fw._on_item_clicked("p2", evt_ctrl)

        # Crucial check: photoSelected MUST NOT be emitted when Ctrl-clicking
        self.assertEqual(len(selected_switches), 0, "Holding Ctrl MUST NOT emit photoSelected")
        self.assertEqual(fw._active_id, "p1", "Holding Ctrl MUST NOT switch active photo")
        self.assertIn("p1", fw.get_selected_photo_ids())
        self.assertIn("p2", fw.get_selected_photo_ids())
        self.assertEqual(len(fw.get_selected_photo_ids()), 2)

        # Ctrl+Click on p3
        fw._on_item_clicked("p3", evt_ctrl)
        self.assertEqual(len(selected_switches), 0, "Holding Ctrl MUST NOT emit photoSelected on further selections")
        self.assertEqual(fw._active_id, "p1")
        self.assertEqual(len(fw.get_selected_photo_ids()), 3)

        # Normal click on p2 (no Ctrl) should switch active photo and emit photoSelected
        evt_normal = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            QPointF(10, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        fw._on_item_clicked("p2", evt_normal)
        self.assertEqual(len(selected_switches), 1)
        self.assertEqual(selected_switches[-1], "p2")
        self.assertEqual(fw._active_id, "p2")
        self.assertEqual(fw.get_selected_photo_ids(), ["p2"])

        fw.close()

    def test_last_session_path_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".sdss", delete=False) as f:
            temp_sdss = f.name

        try:
            config_manager.save_last_session_path(temp_sdss)
            self.assertEqual(config_manager.get_last_session_path(), os.path.abspath(temp_sdss))

            config_manager.save_last_session_path(None)
            self.assertIsNone(config_manager.get_last_session_path())
        finally:
            if os.path.exists(temp_sdss):
                os.remove(temp_sdss)

    def test_no_photo_doubling_when_opening_session_file(self):
        # 1. Create dummy image files
        t_dir = tempfile.gettempdir()
        img1 = os.path.join(t_dir, "test_v019_1.jpg")
        img2 = os.path.join(t_dir, "test_v019_2.jpg")
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img1, dummy)
        cv2.imwrite(img2, dummy)

        # 2. Pre-cache thumbnails as normal session cache does
        p_data = {
            "width": 100,
            "height": 100,
            "thumbnail_rgb": np.zeros((80, 80, 3), dtype=np.uint8),
            "exif": {},
            "auto_ev": 0.0
        }
        session_cache_manager.save_photo_cache(img1, p_data)
        session_cache_manager.save_photo_cache(img2, p_data)

        # 3. Create dummy session file
        sdss_path = os.path.join(t_dir, "test_session_v019.sdss")
        session_data = {
            "format": "SpektraDarkroomSession",
            "version": "0.1.9",
            "active_photo_path": img1,
            "photos": [
                {"id": "p1", "filename": "test_v019_1.jpg", "path": img1, "params": {}},
                {"id": "p2", "filename": "test_v019_2.jpg", "path": img2, "params": {}}
            ]
        }
        session_cache_manager.save_session_file(sdss_path, session_data)

        try:
            # First launch opening the session
            win1 = DarkroomMainWindow(self.engine, initial_session=sdss_path)
            win1._suppress_close_confirm = True
            win1._open_session_by_path(sdss_path)
            if win1._import_worker and win1._import_worker.isRunning():
                win1._import_worker.wait(1000)
            self.assertEqual(len(win1.photos), 2, "Session should have exactly 2 photos")

            # Simulate closing window (which saves last_session_path, NOT raw files)
            win1.close()

            # Second launch opening the session again
            win2 = DarkroomMainWindow(self.engine, initial_session=sdss_path)
            win2._suppress_close_confirm = True
            win2._open_session_by_path(sdss_path)
            if win2._import_worker and win2._import_worker.isRunning():
                win2._import_worker.wait(1000)
            self.assertEqual(len(win2.photos), 2, "Reopening session MUST NOT double photo count to 4")

            win2.close()
        finally:
            for p in [img1, img2, sdss_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def test_preferences_dialog_restore_session_setting(self):
        dlg = PreferencesDialog()
        self.assertTrue(hasattr(dlg, "chk_restore_last_files"))
        self.assertIn("会话工程", dlg.chk_restore_last_files.text())
        dlg.close()


if __name__ == "__main__":
    unittest.main()
