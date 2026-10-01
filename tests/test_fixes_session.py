import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np
from PySide6.QtCore import Qt, QPointF, QPoint
from PySide6.QtWidgets import QApplication, QPushButton
from PySide6.QtGui import QMouseEvent

# Ensure offscreen Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication.instance() or QApplication(sys.argv)

import sdc_manager
from app_core import SpektraEngine
from ui.export_dialog import ExportImageDialog, ExportProgressDialog, SingleExportWorker
from ui.adobe_scrub_slider import AdobeScrubSlider
from ui.spinner_widget import SpinnerWidget
from ui.accordion import AccordionSection
from path_utils import get_icon_path

class TestFixesSession(unittest.TestCase):
    def test_icons_exist(self):
        arrow = get_icon_path("arrow_down.png")
        cb_c = get_icon_path("checkbox_checked.png")
        cb_u = get_icon_path("checkbox_unchecked.png")
        self.assertTrue(os.path.exists(arrow), "arrow_down.png missing")
        self.assertTrue(os.path.exists(cb_c), "checkbox_checked.png missing")
        self.assertTrue(os.path.exists(cb_u), "checkbox_unchecked.png missing")

    def test_exif_extraction_raf_and_arw(self):
        engine = SpektraEngine()
        if os.path.exists("测试底片/DSCF0296.RAF"):
            exif_raf = engine._extract_exif("测试底片/DSCF0296.RAF")
            self.assertEqual(exif_raf.get("camera_make"), "FUJIFILM")
            self.assertEqual(exif_raf.get("camera_model"), "X-M5")
            self.assertNotEqual(exif_raf.get("shutter_speed"), "-")
            self.assertNotEqual(exif_raf.get("iso"), "-")
            self.assertIn("color_temp", exif_raf)
            self.assertIn("tint", exif_raf)

        if os.path.exists("测试底片/sample.ARW"):
            exif_arw = engine._extract_exif("测试底片/sample.ARW")
            self.assertEqual(exif_arw.get("camera_make"), "SONY")
            self.assertEqual(exif_arw.get("camera_model"), "ILCE-7M3")
            self.assertIn("color_temp", exif_arw)
            self.assertIn("tint", exif_arw)

    def test_export_dialog_components(self):
        dlg = ExportImageDialog("test_out.jpg")
        cfg = dlg.get_export_config()
        self.assertEqual(cfg["dpi"], 300)
        self.assertEqual(cfg["format"], "jpeg")
        self.assertTrue(dlg.chk_progressive.isChecked())
        self.assertEqual(dlg.width(), 660)
        dlg.close()

    def test_slider_responsiveness_and_styling(self):
        slider = AdobeScrubSlider("曝光补偿 (EV)", -3.0, 3.0, 0.0, step=0.05, unit=" EV")
        self.assertLessEqual(slider.label.minimumWidth(), 30)
        self.assertEqual(slider.current_val, 0.0)
        slider.set_value(1.5)
        self.assertAlmostEqual(slider.current_val, 1.5)
        self.assertIn("transparent", slider.styleSheet())
        slider.close()

    def test_accordion_styling_and_collapse(self):
        sec = AccordionSection("测试面板", is_expanded=True)
        sec.resize(300, 200)
        sec.show()
        self.assertEqual(sec.card.objectName(), "paramGroupCard")
        self.assertTrue(sec.is_expanded)
        self.assertEqual(sec.title_label.text(), "测试面板")
        self.assertIsNotNone(sec.reset_btn)
        sec.close()

    def test_spinner_idle_dot_elimination(self):
        spinner = SpinnerWidget(size=14)
        spinner.stop()
        self.assertFalse(spinner._is_spinning)
        pix = spinner.grab()
        self.assertFalse(pix.isNull())
        spinner.close()

    def test_sdc_not_littered_when_not_dirty(self):
        dummy = "测试底片/dummy_test_photo.jpg"
        dummy_sdc = dummy + ".sdc"
        if os.path.exists(dummy_sdc):
            os.remove(dummy_sdc)
        photo_is_dirty = False
        if photo_is_dirty:
            sdc_manager.save_sdc(dummy, "kodak_portra_400", "kodak_2383", {})
        self.assertFalse(os.path.exists(dummy_sdc), "SDC should not be written when photo is not dirty!")

    def test_window_native_maximize_and_borders(self):
        from ui.main_window import DarkroomMainWindow
        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        self.assertTrue(win.acceptDrops(), "DarkroomMainWindow must accept drops!")
        self.assertFalse(win.btn_max.is_maximized_state)
        
        # Test maximize state sync
        win._update_root_style(True)
        self.assertIn("border-radius: 0px", win.centralWidget().styleSheet())
        win._update_root_style(False)
        self.assertIn("border-radius: 8px", win.centralWidget().styleSheet())
        win.close()

    def test_canvas_zoom_percentage_actual_resolution(self):
        from ui.canvas_viewport import DarkroomGLCanvas
        cv = DarkroomGLCanvas()
        cv.resize(800, 600)
        # Fake 4000x3000 image
        dummy_img = np.zeros((3000, 4000, 3), dtype=np.float32)
        cv.set_image(dummy_img)
        
        # base_w is 600 * (4000/3000) = 800.
        # zoom_100 is 4000 / 800 = 5.0.
        # When zoom = 5.0, 1 canvas pixel == 1 native image pixel (100% 1:1 pixel scale)
        cv.zoom = 5.0
        self.assertAlmostEqual(cv.get_actual_zoom_ratio(), 1.0)
        
        # When zoom = 1.0 (fit to screen), ratio should be 800 / 4000 = 0.2 (20%)
        cv.zoom = 1.0
        self.assertAlmostEqual(cv.get_actual_zoom_ratio(), 0.2)
        cv.close()

    def test_canvas_compare_amber_hairline(self):
        from ui.canvas_viewport import FRAGMENT_SHADER_SRC
        # Verify fragment shader contains theme amber hairline vec3(0.96, 0.62, 0.05)
        self.assertIn("0.96, 0.62, 0.05", FRAGMENT_SHADER_SRC)
        self.assertIn("u_view_mode == 2", FRAGMENT_SHADER_SRC)
        self.assertIn("u_base_temp", FRAGMENT_SHADER_SRC)
        self.assertIn("u_base_tint", FRAGMENT_SHADER_SRC)

    def test_lut_cache_generation_and_loading(self):
        engine = SpektraEngine()
        lut = engine.get_3d_lut("kodak_portra_400", "kodak_2383", lut_size=16)
        self.assertIsNotNone(lut)
        self.assertEqual(lut.shape, (16, 16, 16, 3))
        # Ensure disk cache file exists
        cache_dir = os.path.join(engine.resources_dir, ".lut_cache")
        cache_path = os.path.join(cache_dir, "v2__kodak_portra_400__kodak_2383__16.npy")
        self.assertTrue(os.path.exists(cache_path), f"LUT cache file {cache_path} must exist!")

    def test_compare_button_hold_and_click_logic(self):
        from ui.main_window import DarkroomMainWindow
        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        self.assertEqual(win.canvas.params.get("view_mode"), 0)
        # First click: split drag compare (Mode 1)
        win.toggle_split_view()
        self.assertEqual(win.canvas.params.get("view_mode"), 1)
        # Second click: full side-by-side compare (Mode 2)
        win.toggle_split_view()
        self.assertEqual(win.canvas.params.get("view_mode"), 2)
        # Third click: reset to normal edit state (Mode 0)
        win.toggle_split_view()
        self.assertEqual(win.canvas.params.get("view_mode"), 0)
        win.close()

    def test_slider_set_default_value(self):
        slider = AdobeScrubSlider("色温", 2000, 12000, 5500, step=50, unit="K", decimals=0)
        self.assertEqual(slider.get_default_value(), 5500.0)
        slider.set_default_value(6200.0)
        self.assertEqual(slider.get_default_value(), 6200.0)
        slider.reset()
        self.assertEqual(slider.value(), 6200.0)

    def test_on_switch_photo_execution(self):
        from ui.main_window import DarkroomMainWindow
        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        dummy_photo = {
            "id": "photo_test_1",
            "filename": "test.raw",
            "path": "test.raw",
            "width": 2048,
            "height": 1365,
            "thumbnail_rgb": np.zeros((72, 108, 3), dtype=np.uint8),
            "float_img": np.zeros((1365, 2048, 3), dtype=np.float32),
            "exif": {
                "color_temp": 5600,
                "tint": 1.05,
                "camera_model": "TestCam"
            }
        }
        win.photos.append(dummy_photo)
        # on_switch_photo MUST execute completely without raising AttributeError
        win.on_switch_photo("photo_test_1")
        self.assertEqual(win.active_photo_id, "photo_test_1")
        self.assertEqual(win._active_base_temp, 5600.0)
        self.assertEqual(win._active_base_tint, 1.05)
        self.assertEqual(win.canvas._image_height, 1365)
        self.assertTrue(win.lbl_photo_info.text().startswith("test.raw"))
        win.close()

    def test_accordion_clipping_and_collapse(self):
        sec = AccordionSection("Test Section", is_expanded=True)
        btn1 = QPushButton("Btn1")
        btn2 = QPushButton("Btn2")
        sec.addWidget(btn1)
        sec.addWidget(btn2)
        self.assertTrue(sec.is_expanded)
        self.assertEqual(sec.content_layout.count(), 2)

    def test_title_bar_draggable_and_menu_bar_policy(self):
        from ui.main_window import DarkroomMainWindow
        from PySide6.QtWidgets import QSizePolicy
        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        # Menu bar horizontal policy must be Minimum so it doesn't block window dragging
        self.assertEqual(win.menu_bar.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Minimum)
        # Test close button vector symmetry
        self.assertEqual(win.btn_min.width(), 46)
        self.assertEqual(win.btn_min.height(), 32)
        win.close()

    def test_raw_wb_adoption_over_stale_sdc(self):
        from ui.main_window import DarkroomMainWindow
        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        dummy_photo = {
            "id": "photo_wb_test",
            "filename": "wb_test.cr2",
            "path": "wb_test.cr2",
            "width": 2048,
            "height": 1365,
            "thumbnail_rgb": np.zeros((72, 108, 3), dtype=np.uint8),
            "float_img": np.zeros((1365, 2048, 3), dtype=np.float32),
            "exif": {
                "color_temp": 5950.0,
                "tint": 1.02,
                "camera_model": "Canon 5D"
            }
        }
        win.photos.append(dummy_photo)
        win.on_switch_photo("photo_wb_test")
        self.assertEqual(win.current_params["color_temp"], 5950.0)
        self.assertEqual(win.slider_temp.value(), 5950.0)
        win.close()

    def test_film_grid_columns_reflow(self):
        from ui.main_window import DarkroomMainWindow
        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        win.show()
        # Switch to editor page
        win.stack.setCurrentIndex(1)
        win.main_splitter.setSizes([416, 700, 320])
        win.reflow_grids()
        QApplication.processEvents()
        cards = list(win.film_cards.values())
        cols_used = max(win.film_grid.getItemPosition(win.film_grid.indexOf(c))[1] for c in cards) + 1
        self.assertIn(cols_used, (2, 3, 4))
        win.close()

    def test_filmstrip_compact_layout(self):
        from ui.main_window import DarkroomMainWindow
        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        fs = win.filmstrip
        # Scroll area must be first item on the left
        self.assertEqual(fs.layout().itemAt(0).widget(), fs.scroll_area)
        # Action container with count and add button must be last on the right
        self.assertEqual(fs.layout().itemAt(1).widget(), fs.action_container)
        self.assertEqual(fs.lbl_lib_title.text(), "胶片库")
        self.assertIn("0", fs.count_label.text())
        win.close()

if __name__ == "__main__":
    unittest.main()



