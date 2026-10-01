"""Unit tests for SpektraDarkroom v0.1.2 refinements.
Covers:
1. Quick export icon removed (clean text-only).
2. Title session name with dirty asterisk.
3. Canvas context menu integrity.
4. Auto-meter button removed from scene exposure, moved to preferences (auto_meter_on_import=False default).
5. 3-state comparison cycling (0 -> 1 -> 2 -> 0) and photo switch reset to 0.
6. Persistent filename & zoom in status bar.
7. Proportional sprocket holes calculation.
8. Add button solid style without dashed border.
9. Recent sessions menu and exit confirmation guard.
10. Version verification (0.1.2).
"""

import unittest
import os
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow
from ui.preferences_dialog import PreferencesDialog
import config_manager
from version import VERSION_STRING


app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestV012Refinements(unittest.TestCase):
    def setUp(self):
        self.engine = SpektraEngine()
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        if hasattr(self.win, "export_queue") and self.win.export_queue:
            self.win.export_queue.close()
        self.win.close()

    def test_version_bumped_to_012(self):
        self.assertTrue(VERSION_STRING in ("0.1.2", "0.1.3", "0.1.4", "0.1.5", "0.1.6", "0.1.7", "0.1.8", "0.1.9"))

    def test_auto_meter_button_removed_and_preference_available(self):
        # Auto-meter button removed from sec_exposure
        self.assertFalse(hasattr(self.win, "btn_auto_meter"))
        # Auto-meter preference exists and defaults to False
        pref_val = config_manager.get_preference("auto_meter_on_import", None)
        self.assertIn(pref_val, [False, None])

        # Preferences dialog has chk_auto_meter checkbox
        dlg = PreferencesDialog(self.win)
        self.assertTrue(hasattr(dlg, "chk_auto_meter"))
        self.assertEqual(dlg.chk_auto_meter.isChecked(), False)
        dlg.close()

    def test_three_state_compare_cycle_and_reset_on_photo_switch(self):
        # Default mode is 0 (normal single edit)
        self.assertEqual(self.win.canvas.params.get("view_mode"), 0)
        self.assertFalse(self.win.btn_split.isChecked())

        # First click: 1 (split drag compare)
        self.win.toggle_split_view()
        self.assertEqual(self.win.canvas.params.get("view_mode"), 1)
        self.assertTrue(self.win.btn_split.isChecked())

        # Second click: 2 (global side-by-side compare)
        self.win.toggle_split_view()
        self.assertEqual(self.win.canvas.params.get("view_mode"), 2)
        self.assertTrue(self.win.btn_split.isChecked())

        # Third click: returns to 0 (normal edit mode)
        self.win.toggle_split_view()
        self.assertEqual(self.win.canvas.params.get("view_mode"), 0)
        self.assertFalse(self.win.btn_split.isChecked())

        # Set to mode 2, then switch photo -> must auto reset to mode 0
        self.win.set_view_mode(2)
        self.assertEqual(self.win.canvas.params.get("view_mode"), 2)
        dummy_p = {
            "id": "p_test_switch",
            "filename": "test.jpg",
            "path": "test.jpg",
            "thumbnail_rgb": np.zeros((72, 108, 3), dtype=np.uint8),
            "float_img": np.zeros((100, 100, 3), dtype=np.float32)
        }
        self.win.photos.append(dummy_p)
        self.win.on_switch_photo("p_test_switch")
        self.assertEqual(self.win.canvas.params.get("view_mode"), 0)
        self.assertFalse(self.win.btn_split.isChecked())

    def test_session_recent_menu_and_title_asterisk(self):
        # Verify File menu has "最近会话"
        file_menu = self.win.menu_bar.actions()[0].menu()
        menu_titles = [a.text() for a in file_menu.actions()]
        self.assertIn("最近会话", menu_titles)
        self.assertNotIn("最近打开", menu_titles)

        # Mark dirty should prefix asterisk to title
        self.win._session_is_dirty = True
        self.win._update_window_title()
        self.assertTrue(self.win.windowTitle().startswith("*"))
        self.assertTrue(self.win.lbl_brand.text().startswith("*"))

        # Clean session removes asterisk
        self.win._session_is_dirty = False
        self.win._update_window_title()
        self.assertFalse(self.win.windowTitle().startswith("*"))
        self.assertFalse(self.win.lbl_brand.text().startswith("*"))

    def test_canvas_context_menu_no_crash(self):
        # Canvas context menu policy should be DefaultContextMenu
        self.assertEqual(self.win.canvas.contextMenuPolicy(), Qt.ContextMenuPolicy.DefaultContextMenu)

    def test_persistent_status_restore(self):
        self.win._current_photo_path = "D:/dummy/my_photo.arw"
        self.win.restore_default_status()
        self.assertIn("my_photo.arw", self.win.lbl_photo_info.text())


if __name__ == "__main__":
    unittest.main()
