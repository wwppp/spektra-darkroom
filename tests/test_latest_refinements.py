"""
test_latest_refinements.py - Unit tests for latest refinements and bug fixes
"""

import os
import tempfile
import json
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt, QPoint, QPointF

from app_core import SpektraEngine
from ui.canvas_viewport import DarkroomGLCanvas
from ui.filmstrip_widget import FilmstripWidget
from ui.export_dialog import ExportImageDialog, ExportProgressDialog, SingleExportWorker
from ui.stock_manager_dialog import StockManagerDialog, export_stock_profile_json
from ui.main_window import DarkroomMainWindow, FilmGridCard

os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication.instance() or QApplication([])


class TestLatestRefinements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SpektraEngine()

    def setUp(self):
        self.win = DarkroomMainWindow(self.engine)

    def tearDown(self):
        self.win.close()

    def test_canvas_floating_badges(self):
        canvas = self.win.canvas
        self.assertIsInstance(canvas.badge_before, QLabel)
        self.assertIsInstance(canvas.badge_after, QLabel)
        self.assertEqual(canvas.badge_before.text(), "原图")
        self.assertEqual(canvas.badge_after.text(), "已修改")

        # Switching to split mode displays badges
        canvas._has_image = True
        canvas.set_view_mode(1)
        self.assertFalse(canvas.badge_before.isHidden())
        self.assertFalse(canvas.badge_after.isHidden())

        # Switching back to single mode hides badges
        canvas.set_view_mode(0)
        self.assertTrue(canvas.badge_before.isHidden())
        self.assertTrue(canvas.badge_after.isHidden())

    def test_export_options_sorted_low_to_high(self):
        dlg = ExportImageDialog("test.tif")
        
        # 1. DPI sorted: 72 -> 300 -> 600
        dpis = [dlg.combo_dpi.itemData(i) for i in range(dlg.combo_dpi.count())]
        self.assertEqual(dpis, [72, 300, 600])
        self.assertEqual(dlg.combo_dpi.currentData(), 300)

        # 2. TIFF Bit Depths sorted: 8 -> 16 -> 32
        dlg.combo_format.setCurrentIndex(1) # TIFF
        depths = [dlg.combo_depth.itemData(i) for i in range(dlg.combo_depth.count())]
        self.assertEqual(depths, [8, 16, 32])
        self.assertEqual(dlg.combo_depth.currentData(), 16)
        dlg.close()

    def test_export_worker_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "cancel_test.jpg")
            params = {
                "source_path": None,
                "exposure_ev": 0.0,
                "color_temp": 5500.0,
                "tint": 1.0
            }
            worker = SingleExportWorker(self.engine, params, out_path, "jpeg", 9, None)
            worker.cancel()
            self.assertTrue(worker._is_cancelled)

    def test_filmstrip_right_action_layout(self):
        fs = self.win.filmstrip
        self.assertEqual(fs.lbl_lib_title.text(), "胶片库")
        self.assertEqual(fs.add_btn.text(), "＋ 添加")
        self.assertEqual(fs.action_container.width(), 66)

    def test_stock_manager_sizing_and_terminology(self):
        dlg_film = StockManagerDialog(mode="film", engine=self.engine)
        self.assertEqual(dlg_film.width(), 980)
        self.assertEqual(dlg_film.height(), 620)
        self.assertEqual(dlg_film.term, "胶卷")
        self.assertIn("胶卷", dlg_film.windowTitle())

        headers = [dlg_film.table.horizontalHeaderItem(i).text() for i in range(dlg_film.table.columnCount())]
        self.assertIn("胶卷名称", headers)
        dlg_film.close()

        dlg_paper = StockManagerDialog(mode="paper", engine=self.engine)
        self.assertEqual(dlg_paper.term, "相纸")
        self.assertIn("相纸", dlg_paper.windowTitle())
        headers_p = [dlg_paper.table.horizontalHeaderItem(i).text() for i in range(dlg_paper.table.columnCount())]
        self.assertIn("相纸名称", headers_p)
        dlg_paper.close()

    def test_export_physical_profile_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = os.path.join(tmp_dir, "kodak_portra_400.json")
            res = export_stock_profile_json(self.engine, "kodak_portra_400", out_json)
            self.assertTrue(res)
            self.assertTrue(os.path.exists(out_json))
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("info", data)
            self.assertEqual(data["info"]["stock"], "kodak_portra_400")

    def test_film_card_full_brand_name_and_context_actions(self):
        card = self.win.film_cards.get("kodak_portra_400")
        self.assertIsNotNone(card)
        self.assertEqual(card.clean_name, "Kodak Portra 400")
        self.assertEqual(card.lbl_name.text(), "Kodak Portra 400")

    def test_concise_diffusion_options(self):
        combo = self.win.combo_diff_strength
        labels = [combo.combo.itemText(i) for i in range(combo.combo.count())]
        self.assertIn("0 (无滤镜)", labels)
        self.assertIn("1/8 档", labels)
        self.assertIn("1/4 档", labels)
        self.assertIn("1/2 档", labels)
        self.assertIn("1 档", labels)
        self.assertIn("2 档", labels)


    def test_version_0_1_0(self):
        from version import VERSION_STRING
        self.assertEqual(VERSION_STRING, "0.1.0")

    def test_pinned_histogram_and_view_reset_layout(self):
        self.assertTrue(hasattr(self.win, "histogram_widget"))
        self.assertTrue(hasattr(self.win, "reset_default_layout"))
        # Verify action exists in View menu
        view_actions = [a.text() for a in self.win.menu_bar.actions()[2].menu().actions() if not a.isSeparator()]
        self.assertIn("恢复默认布局", view_actions)

    def test_combobox_row_ignores_wheel_event(self):
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtCore import QPointF
        combo_row = self.win.combo_diffusion
        ev = QWheelEvent(QPointF(0, 0), QPointF(0, 0), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        combo_row.combo.wheelEvent(ev)
        self.assertFalse(ev.isAccepted())


if __name__ == "__main__":
    unittest.main()
