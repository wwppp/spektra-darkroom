import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication, QMenu
from PySide6.QtCore import Qt, QPoint

app = QApplication.instance() or QApplication(sys.argv)

import config_manager
import sdc_manager
from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow, FilmGridCard
from ui.export_dialog import ExportImageDialog, ExportProgressDialog
from ui.stock_manager_dialog import StockManagerDialog, export_stock_lut
from ui.preferences_dialog import PreferencesDialog
from ui.window_utils import get_darkroom_menu_style


class TestNew14Features(unittest.TestCase):
    def setUp(self):
        self.engine = SpektraEngine()
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        self.win.close()

    def test_item_1_dirty_tracking_and_exit_handling(self):
        # 1. Dirty tracking
        dummy_photo = {
            "id": "photo_dirty_test",
            "filename": "dirty_test.jpg",
            "path": "dirty_test.jpg",
            "width": 1000,
            "height": 800,
            "thumbnail_rgb": np.zeros((72, 108, 3), dtype=np.uint8),
            "float_img": np.zeros((800, 1000, 3), dtype=np.float32),
            "exif": {"color_temp": 5500.0, "tint": 1.0}
        }
        self.win.photos.append(dummy_photo)
        self.win.filmstrip.append_photo(dummy_photo)
        self.win.on_switch_photo("photo_dirty_test")
        self.assertFalse(self.win._photo_is_dirty)

        # Modifying exposure marks dirty
        self.win._on_slider_live("exposure_ev", 0.5)
        self.assertTrue(self.win._photo_is_dirty)
        self.assertTrue(dummy_photo.get("is_dirty"))
        self.assertIn("dirty_test.jpg", self.win.windowTitle())

        # Save resets dirty
        self.win.action_save_current()
        self.assertFalse(self.win._photo_is_dirty)
        self.assertFalse(dummy_photo.get("is_dirty"))
        self.assertFalse(self.win.filmstrip._item_widgets["photo_dirty_test"].is_dirty)

    def test_item_2_and_3_export_concise_and_cancel_button(self):
        dlg = ExportImageDialog("test.jpg")
        cfg = dlg.get_export_config()
        self.assertEqual(cfg["dpi"], 300)
        self.assertEqual(cfg["format"], "jpeg")
        dlg.close()

        prog = ExportProgressDialog("批量导出")
        self.assertIsNotNone(prog.btn_cancel)
        self.assertEqual(prog.btn_cancel.text(), "取消")
        prog.close()

    def test_item_5_6_7_context_menus_film_and_paper(self):
        # Film and paper card right-click signals exist
        card = self.win.film_cards.get("kodak_portra_400")
        self.assertIsNotNone(card)
        self.assertTrue(hasattr(card, "exportRequested"))
        self.assertTrue(hasattr(card, "removeRequested"))

        # Film & paper scroll blank context menu policies
        self.assertEqual(self.win.film_scroll.contextMenuPolicy(), Qt.ContextMenuPolicy.CustomContextMenu)
        self.assertEqual(self.win.paper_scroll.contextMenuPolicy(), Qt.ContextMenuPolicy.CustomContextMenu)

    def test_item_8_stock_manager_dialog(self):
        dlg_film = StockManagerDialog(mode="film", engine=self.engine)
        self.assertEqual(dlg_film.windowTitle(), "胶卷库管理")
        self.assertGreater(dlg_film.table.rowCount(), 0)
        dlg_film.close()

        dlg_paper = StockManagerDialog(mode="paper", engine=self.engine)
        self.assertEqual(dlg_paper.windowTitle(), "相纸库管理")
        self.assertGreater(dlg_paper.table.rowCount(), 0)
        dlg_paper.close()

    def test_item_9_menu_style_no_clipping(self):
        style = get_darkroom_menu_style()
        self.assertIn("margin: 1px 4px;", style)
        self.assertIn("border-radius: 4px;", style)

    def test_item_10_and_11_menu_bar_structure(self):
        # File menu has "退出" and "保存修改"
        file_actions = [a.text() for a in self.win.menu_bar.actions()[0].menu().actions() if not a.isSeparator()]
        self.assertIn("退出", file_actions)
        self.assertIn("保存修改", file_actions)

        # Edit menu has "胶卷库...", "相纸库...", "首选项..."
        edit_actions = [a.text() for a in self.win.menu_bar.actions()[1].menu().actions() if not a.isSeparator()]
        self.assertIn("胶卷库...", edit_actions)
        self.assertIn("相纸库...", edit_actions)
        self.assertIn("首选项...", edit_actions)

    def test_item_12_preferences_dialog(self):
        prefs_dlg = PreferencesDialog()
        self.assertEqual(prefs_dlg.windowTitle(), "首选项 (Preferences)")
        self.assertTrue(hasattr(prefs_dlg, "combo_hw_accel"))
        self.assertTrue(hasattr(prefs_dlg, "combo_preview_res"))
        self.assertEqual(prefs_dlg.combo_hw_accel.count(), 3)
        prefs_dlg.close()

    def test_item_13_split_view_badges(self):
        # View mode 1 enables split compare
        self.win.canvas.set_view_mode(1)
        self.assertEqual(self.win.canvas.params["view_mode"], 1)

    def test_item_14_filmstrip_3_tier_header(self):
        fs = self.win.filmstrip
        self.assertEqual(fs.lbl_lib_title.text(), "胶片库")
        self.assertIn("0", fs.count_label.text())
        self.assertEqual(fs.add_btn.text(), "＋ 添加")


    def test_item_15_compare_reset_on_photo_switch(self):
        # Put into split compare mode
        self.win.set_view_mode(1)
        self.assertEqual(self.win.canvas.params["view_mode"], 1)
        # Calling on_switch_photo should reset compare view mode to 0
        self.win.on_switch_photo("non_existent_id")
        self.assertEqual(self.win.canvas.params["view_mode"], 0)

    def test_item_16_jpeg_scale_percentage_option(self):
        from ui.export_dialog import ExportImageDialog
        dlg = ExportImageDialog("test_out.jpg")
        self.assertTrue(hasattr(dlg, "combo_scale"))
        self.assertGreaterEqual(dlg.combo_scale.count(), 6)
        cfg = dlg.get_export_config()
        self.assertIn("scale_pct", cfg)
        self.assertEqual(cfg["scale_pct"], 100)
        dlg.close()

    def test_item_17_canvas_empty_overlay(self):
        self.assertTrue(hasattr(self.win.canvas, "empty_overlay"))
        self.assertTrue(hasattr(self.win.canvas, "importRequested"))


if __name__ == "__main__":
    unittest.main()
