import os
import sys
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

import app_core
from app_core import (
    SpektraEngine, ACES_TO_PROPHOTO, SRGB_TO_PROPHOTO, PROPHOTO_TO_SRGB,
    prophoto_to_srgb, srgb_to_linear
)
from ui.main_window import DarkroomMainWindow
from ui.preferences_dialog import PreferencesDialog
from ui.canvas_viewport import DarkroomGLCanvas
from path_utils import get_resource_dir


class TestColorAndScanMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res_dir = get_resource_dir()
        cls.engine = SpektraEngine(resources_dir=cls.res_dir)
        cls.win = DarkroomMainWindow(cls.engine)

    def test_01_color_space_matrices_and_cctf(self):
        """Verify color conversion matrices and transfer functions."""
        # Check matrix dimensions
        self.assertEqual(ACES_TO_PROPHOTO.shape, (3, 3))
        self.assertEqual(SRGB_TO_PROPHOTO.shape, (3, 3))
        self.assertEqual(PROPHOTO_TO_SRGB.shape, (3, 3))

        # Check inverse relationship between SRGB_TO_PROPHOTO and PROPHOTO_TO_SRGB
        identity_test = SRGB_TO_PROPHOTO @ PROPHOTO_TO_SRGB
        np.testing.assert_allclose(identity_test, np.eye(3), atol=1e-3)

        # Check transfer function bounds
        dummy = np.array([[[0.0, 0.18, 1.0]]], dtype=np.float32)
        srgb_out = prophoto_to_srgb(dummy)
        self.assertEqual(srgb_out.shape, (1, 1, 3))
        self.assertGreaterEqual(float(srgb_out.min()), 0.0)
        self.assertLessEqual(float(srgb_out.max()), 1.0)

    def test_02_auto_metering_and_exposure_baseline(self):
        """Verify auto metering in linear ProPhoto and base_ev handling."""
        # 18% gray in linear ProPhoto
        mid_gray = np.full((100, 100, 3), 0.18, dtype=np.float32)
        ev_mid = self.engine.calculate_auto_exposure_ev(mid_gray)
        self.assertAlmostEqual(ev_mid, 0.0, delta=0.1)

        # Dark image -> positive EV compensation
        dark_img = np.full((100, 100, 3), 0.045, dtype=np.float32)
        ev_dark = self.engine.calculate_auto_exposure_ev(dark_img)
        self.assertGreater(ev_dark, 1.5)

    def test_03_preferences_dialog_no_auto_exposure(self):
        """Verify auto exposure toggle was deleted from PreferencesDialog."""
        dlg = PreferencesDialog(self.win)
        self.assertFalse(hasattr(dlg, "chk_auto_meter"), "Auto meter checkbox should be removed from preferences")
        dlg.close()

    def test_04_canvas_viewport_shader_and_lut_override(self):
        """Verify canvas shader contains prophoto_to_srgb and supports lut_override."""
        from ui.canvas_viewport import FRAGMENT_SHADER_SRC
        self.assertIn("prophoto_to_srgb", FRAGMENT_SHADER_SRC)
        self.assertIn("2.03649", FRAGMENT_SHADER_SRC)

        # Test render_offscreen signature
        import inspect
        sig = inspect.signature(self.win.canvas.render_offscreen)
        self.assertIn("lut_override", sig.parameters)

    def test_05_optical_and_scan_3d_lut_generation(self):
        """Verify 3D LUT generation works for both optical and scan print modes."""
        lut_optical = self.engine.get_3d_lut("kodak_portra_400", "kodak_2383", lut_size=17, print_mode="optical")
        self.assertEqual(lut_optical.shape, (17, 17, 17, 3))
        self.assertGreater(float(lut_optical.mean()), 0.1)

        lut_scan = self.engine.get_3d_lut("kodak_portra_400", "kodak_2383", lut_size=17, print_mode="scan")
        self.assertEqual(lut_scan.shape, (17, 17, 17, 3))
        self.assertGreater(float(lut_scan.mean()), 0.1)

        # Scan and optical LUTs should produce distinctly different density mappings
        diff = np.abs(lut_optical - lut_scan).mean()
        self.assertGreater(diff, 0.02, "Optical and scan mode LUTs should differ significantly")

    def test_06_ui_print_mode_switch(self):
        """Verify UI print mode switches properly when changing paper cards (Item 9)."""
        # Switching paper to 'none' engages scan mode
        self.win.on_switch_paper("none", push_undo=False)
        self.assertEqual(self.win.current_print_mode, "scan")

        # Switching paper to physical stock engages optical mode
        self.win.on_switch_paper("kodak_2383", push_undo=False)
        self.assertEqual(self.win.current_print_mode, "optical")


if __name__ == "__main__":
    unittest.main()
