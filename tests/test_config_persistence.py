import os
import sys
import tempfile
import unittest
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

import config_manager
from ui.accordion import AccordionSection


class TestConfigPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self._orig_get_config_path = config_manager.get_config_path
        self.test_cfg_path = os.path.join(self.test_dir, "test_config.json")
        config_manager.get_config_path = lambda: self.test_cfg_path
        config_manager._cached_config = None

    def tearDown(self):
        config_manager.get_config_path = self._orig_get_config_path
        config_manager._cached_config = None
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_config_structure(self):
        cfg = config_manager.load_config()
        self.assertIn("version", cfg)
        self.assertIn("window", cfg)
        self.assertIn("ui_state", cfg)
        self.assertIn("preferences", cfg)
        self.assertIn("recent_files", cfg)

        self.assertEqual(cfg["window"]["splitter_main"], [320, 800, 320])
        self.assertTrue(cfg["ui_state"]["sections_expanded"]["exposure"])
        self.assertTrue(cfg["preferences"]["gpu_acceleration"])

    def test_save_and_load_window_config(self):
        config_manager.save_window_config(
            width=1600,
            height=1000,
            x=150,
            y=120,
            is_maximized=False,
            splitter_main=[340, 900, 360],
            splitter_left=[280, 200, 220],
            splitter_center=[700, 110]
        )

        # Clear cached config and reload from disk
        config_manager._cached_config = None
        win = config_manager.get_window_config()
        self.assertEqual(win["width"], 1600)
        self.assertEqual(win["height"], 1000)
        self.assertEqual(win["x"], 150)
        self.assertEqual(win["y"], 120)
        self.assertFalse(win["is_maximized"])
        self.assertEqual(win["splitter_main"], [340, 900, 360])
        self.assertEqual(win["splitter_left"], [280, 200, 220])
        self.assertEqual(win["splitter_center"], [700, 110])

    def test_save_and_load_ui_state(self):
        config_manager.save_ui_state(
            sections_expanded={"exposure": False, "enlarger": True, "optics": False},
            last_film_stock="fuji_velvia_100",
            last_paper_stock="kodak_portra_endura",
            last_active_tab=1
        )

        config_manager._cached_config = None
        ui_st = config_manager.get_ui_state()
        self.assertFalse(ui_st["sections_expanded"]["exposure"])
        self.assertTrue(ui_st["sections_expanded"]["enlarger"])
        self.assertFalse(ui_st["sections_expanded"]["optics"])
        self.assertEqual(ui_st["last_film_stock"], "fuji_velvia_100")
        self.assertEqual(ui_st["last_paper_stock"], "kodak_portra_endura")
        self.assertEqual(ui_st["last_active_tab"], 1)

    def test_preferences_get_and_set(self):
        config_manager.set_preference("export_quality", 98)
        config_manager.set_preference("default_export_dir", "D:/Exports")

        config_manager._cached_config = None
        self.assertEqual(config_manager.get_preference("export_quality"), 98)
        self.assertEqual(config_manager.get_preference("default_export_dir"), "D:/Exports")
        self.assertEqual(config_manager.get_preference("non_existent", "default_val"), "default_val")

    def test_parameter_group_card(self):
        sec = AccordionSection("测试面板")
        self.assertTrue(sec.is_expanded)
        self.assertEqual(sec.title_label.text(), "测试面板")
        self.assertIsNotNone(sec.reset_btn)

    def test_main_window_save_and_restore_cycle(self):
        from app_core import SpektraEngine
        from ui.main_window import DarkroomMainWindow
        from PySide6.QtGui import QCloseEvent

        # 1. First run: preset some custom window values in config
        config_manager.save_window_config(
            width=1500,
            height=950,
            x=100,
            y=100,
            is_maximized=False,
            splitter_main=[350, 750, 350],
            splitter_left=[290, 190, 210],
            splitter_center=[620, 100]
        )

        engine = SpektraEngine()
        win = DarkroomMainWindow(engine)
        win.show()
        app.processEvents()

        self.assertEqual(win.width(), 1500)
        self.assertEqual(win.height(), 950)
        self.assertIsNotNone(win.sec_exposure)
        self.assertIsNotNone(win.sec_enlarger)
        self.assertIsNotNone(win.sec_optical)

        # 2. Change state and trigger closeEvent (in editor workspace)
        win.stack.setCurrentIndex(1)
        app.processEvents()
        win.main_splitter.setSizes([380, 700, 370])
        app.processEvents()
        actual_sizes = win.main_splitter.sizes()
        ev = QCloseEvent()
        win.closeEvent(ev)

        # 3. Verify disk config was updated
        config_manager._cached_config = None
        win_cfg = config_manager.get_window_config()

        self.assertEqual(win_cfg["splitter_main"], actual_sizes)
        win.deleteLater()


if __name__ == "__main__":
    unittest.main()
