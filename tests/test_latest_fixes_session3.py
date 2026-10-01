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
from ui.export_dialog import ExportImageDialog
from ui.adobe_scrub_slider import AdobeScrubSlider
from path_utils import get_resource_dir


class TestLatestFixesSession3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res_dir = get_resource_dir()
        cls.engine = SpektraEngine(resources_dir=cls.res_dir)
        cls.win = DarkroomMainWindow(cls.engine)

    def test_01_all_28_stocks_integrated(self):
        """Verify that all 28 official film and paper profiles are loaded."""
        films = self.engine.get_film_stocks()
        all_film_ids = []
        for g in films:
            for s in g.get("stocks", []):
                all_film_ids.append(s["id"])
        
        papers = self.engine.get_paper_stocks()
        all_paper_ids = [p["id"] for p in papers]

        film_stocks_only = [fid for fid in all_film_ids if fid != "none"]
        paper_stocks_only = [pid for pid in all_paper_ids if pid != "none"]
        self.assertEqual(len(film_stocks_only), 20, f"Expected 20 films, got {len(film_stocks_only)}")
        self.assertEqual(len(paper_stocks_only), 8, f"Expected 8 papers, got {len(paper_stocks_only)}")
        self.assertEqual(len(film_stocks_only) + len(paper_stocks_only), 28)
        self.assertIn("none", all_film_ids)
        self.assertIn("none", all_paper_ids)
        self.assertIn("kodak_kodachrome_64", all_film_ids)
        self.assertIn("kodak_verita_200d", all_film_ids)
        self.assertIn("kodak_2393", all_paper_ids)
        self.assertIn("kodak_supra_endura", all_paper_ids)

    def test_02_scrub_slider_reset_icon_color_states(self):
        """Verify reset button icon toggles between disabled (grey) and active (yellow)."""
        slider = AdobeScrubSlider("曝光", -3.0, 3.0, 0.0, step=0.05)
        # Initially at default
        self.assertEqual(slider.reset_btn.toolTip(), "默认值")
        # Move away from default
        slider.set_value(1.5)
        self.assertEqual(slider.reset_btn.toolTip(), "已修改：点击重置为默认值")
        # Reset back
        slider.reset()
        self.assertEqual(slider.value(), 0.0)
        self.assertEqual(slider.reset_btn.toolTip(), "默认值")

    def test_03_export_dialog_scale_options_simplified(self):
        """Verify export dialog scale dropdown has simplified percentage options and no redundant header icon."""
        dlg = ExportImageDialog("test.jpg", self.win)
        items = [dlg.combo_scale.itemText(i) for i in range(dlg.combo_scale.count())]
        self.assertIn("100% (原始尺寸)", items[0])
        self.assertIn("80%", items[1])
        self.assertIn("50%", items[3])
        self.assertFalse(hasattr(dlg, "lbl_icon"), "Redundant header icon should be removed")
        dlg.close()

    def test_04_preferences_dialog_simplified_labels(self):
        """Verify hardware acceleration and preview resolution strings are concise."""
        dlg = PreferencesDialog(self.win)
        hw_items = [dlg.combo_hw_accel.itemText(i) for i in range(dlg.combo_hw_accel.count())]
        self.assertTrue(any("仅缩略图与视口" in it for it in hw_items))
        self.assertTrue(any("全局 (GPU 加速导出)" in it for it in hw_items))

        prev_items = [dlg.combo_preview_res.itemText(i) for i in range(dlg.combo_preview_res.count())]
        self.assertTrue(any("小 (1080P)" in it for it in prev_items))
        self.assertTrue(any("中 (2K - 推荐)" in it for it in prev_items))
        self.assertFalse(hasattr(dlg, "spin_recent_count"))
        dlg.close()

    def test_05_undo_no_duplicates_immediate_response(self):
        """Verify undo does not store duplicate states and takes effect on first call."""
        self.win._undo_stack.clear()
        self.win._redo_stack.clear()
        self.win.current_params["exposure_ev"] = 0.0
        self.win.push_undo_state()
        self.assertEqual(len(self.win._undo_stack), 1)

        # Duplicate push should be ignored
        self.win.push_undo_state()
        self.assertEqual(len(self.win._undo_stack), 1)

        # Push different state
        self.win.current_params["exposure_ev"] = 1.0
        self.win.push_undo_state()
        self.assertEqual(len(self.win._undo_stack), 2)

        # Current is 1.0, undo should restore 0.0 on the very first undo call
        self.win.undo()
        self.assertEqual(self.win.current_params["exposure_ev"], 0.0)

    def test_06_copy_paste_params_and_select_all(self):
        """Verify Ctrl+C copies darkroom parameters and Ctrl+V pastes them to selected photos."""
        # Setup mock photos in window
        self.win.photos = [
            {"id": "p1", "path": "p1.tif", "filename": "p1.tif", "is_dirty": False, "meta": {}},
            {"id": "p2", "path": "p2.tif", "filename": "p2.tif", "is_dirty": False, "meta": {}},
            {"id": "p3", "path": "p3.tif", "filename": "p3.tif", "is_dirty": False, "meta": {}},
        ]
        self.win.active_photo_id = "p1"
        self.win._current_photo_path = "p1.tif"
        self.win.current_film_stock = "kodak_gold_200"
        self.win.current_paper_stock = "kodak_ultra_endura"
        self.win.current_params = {"exposure_ev": 0.85, "halation": 0.75}

        # 1. Copy
        self.win.action_copy_params()
        self.assertIsNotNone(self.win._copied_darkroom_profile)
        self.assertEqual(self.win._copied_darkroom_profile["film"], "kodak_gold_200")

        # 2. Select All
        self.win.filmstrip.set_photos(self.win.photos, "p1")
        self.win.action_select_all_photos()
        self.assertEqual(len(self.win.filmstrip.get_selected_photo_ids()), 3)

        # 3. Paste
        self.win.action_paste_params()
        for p in self.win.photos:
            self.assertEqual(p.get("film_profile"), "kodak_gold_200")
            self.assertEqual(p.get("paper_profile"), "kodak_ultra_endura")
            self.assertEqual(p.get("params", {}).get("exposure_ev"), 0.85)
            self.assertTrue(p.get("is_dirty"))


if __name__ == "__main__":
    unittest.main()
