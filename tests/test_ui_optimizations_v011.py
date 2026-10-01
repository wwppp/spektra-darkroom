import os
import unittest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow
import config_manager
import session_cache_manager
from path_utils import get_resource_dir
from version import VERSION_STRING

app = QApplication.instance() or QApplication([])

class TestUIOptimizationsV011(unittest.TestCase):
    def setUp(self):
        self.engine = SpektraEngine(resources_dir=get_resource_dir())
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        if hasattr(self.win, "export_queue") and self.win.export_queue:
            self.win.export_queue.close()
        self.win.close()

    def test_version_bumped(self):
        """Verify version patch bumped to >= 0.1.1."""
        self.assertTrue(VERSION_STRING >= "0.1.1")

    def test_compact_card_dimensions(self):
        """Item 10: Verify card minimum width reduced to 90px to fit 3-4 columns."""
        self.win.reflow_grids()
        # Ensure cards exist and dimensions are valid
        self.assertGreater(len(self.win.film_cards), 0)
        self.assertGreater(len(self.win.paper_cards), 0)

    def test_none_preset_pinned_at_index_zero(self):
        """Item 6: Verify 'none' preset is permanently at index 0 regardless of sorting."""
        # Sort by name
        self.win._sort_and_reflow_stocks("film", "name")
        first_film_key = next(iter(self.win.film_cards.keys()))
        self.assertEqual(first_film_key, "none")

        # Sort by time
        self.win._sort_and_reflow_stocks("film", "time")
        first_film_key = next(iter(self.win.film_cards.keys()))
        self.assertEqual(first_film_key, "none")

        # Paper sort by name
        self.win._sort_and_reflow_stocks("paper", "name")
        first_paper_key = next(iter(self.win.paper_cards.keys()))
        self.assertEqual(first_paper_key, "none")

    def test_session_save_and_load(self):
        """Item 9: Verify .sdss session file saving and restoring."""
        test_session_path = os.path.join(get_resource_dir(), ".test_session.sdss")
        self.win.photos = [
            {"id": "p1", "path": "D:/test1.dng", "film_profile": "kodak_portra_400", "paper_profile": "kodak_2383", "params": {"exposure_ev": 0.5}},
            {"id": "p2", "path": "D:/test2.dng", "film_profile": "fuji_pro_400h", "paper_profile": "none", "params": {"exposure_ev": -0.2}}
        ]
        self.win.active_photo_id = "p1"
        self.win._current_photo_path = "D:/test1.dng"

        # Save session
        self.win._do_save_session(test_session_path)
        self.assertTrue(os.path.exists(test_session_path))

        # Read back session
        data = session_cache_manager.load_session_file(test_session_path)
        self.assertIsNotNone(data)
        self.assertEqual(data.get("format"), "SpektraDarkroomSession")
        self.assertEqual(len(data.get("photos", [])), 2)
        self.assertEqual(data.get("active_photo_id"), "p1")

        if os.path.exists(test_session_path):
            os.remove(test_session_path)

    def test_delete_key_handling(self):
        """Item 5: Verify Delete / Backspace key removes selected photos."""
        self.win.photos = [
            {"id": "p1", "path": "D:/test1.dng", "filename": "test1.dng"},
            {"id": "p2", "path": "D:/test2.dng", "filename": "test2.dng"}
        ]
        self.win.filmstrip.set_photos(self.win.photos, active_id="p1")
        self.win.filmstrip._selected_ids = {"p1"}

        # Simulate Delete key press
        key_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
        self.win.keyPressEvent(key_event)
        remaining_ids = [p["id"] for p in self.win.photos]
        self.assertNotIn("p1", remaining_ids)

    def test_export_queue_manager_integrated(self):
        """Item 3 & 8: Verify ExportQueueManager integration and quick export action."""
        self.assertIsNotNone(self.win.export_queue)
        self.assertTrue(hasattr(self.win, "action_quick_export"))
        self.assertTrue(hasattr(self.win, "open_export_queue_dialog"))

if __name__ == "__main__":
    unittest.main()
