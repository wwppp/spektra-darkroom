import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow

app = QApplication.instance() or QApplication([])

class TestNonePresetAndCorrectedMeter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SpektraEngine()

    def test_none_stocks_present_and_default(self):
        """Test that 'none' exists in film and paper stocks and MainWindow defaults to 'none'."""
        films = self.engine.get_film_stocks()
        first_film_category = films[0]
        self.assertEqual(first_film_category["stocks"][0]["id"], "none")

        papers = self.engine.get_paper_stocks()
        self.assertEqual(papers[0]["id"], "none")

        # Identity LUT for none + none
        lut = self.engine.get_3d_lut("none", "none", lut_size=17)
        self.assertEqual(lut.shape, (17, 17, 17, 3))
        # Corners should match exact lattice values (e.g. 0.0 at 0, 1.0 at -1)
        np.testing.assert_allclose(lut[0, 0, 0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(lut[-1, -1, -1], [1.0, 1.0, 1.0], atol=1e-5)

        # MainWindow defaults to none + none
        win = DarkroomMainWindow(self.engine)
        self.assertEqual(win.current_film_stock, "none")
        self.assertEqual(win.current_paper_stock, "none")
        win.close()

    def test_corrected_autoexposure_meter(self):
        """Test that autoexposure uses linear cctf_decoding=False and does not blow out daylight images."""
        # Daytime linear image (simulating DSC02432 with sky ~0.7-0.9)
        daylight_img = np.full((100, 100, 3), 0.70, dtype=np.float32)
        daylight_img[50:, :] = 0.15 # lower half roof shadows
        ev = self.engine.calculate_auto_exposure_ev(daylight_img)
        # Should NOT be huge overexposure like +1.95 EV
        self.assertLess(ev, 0.5)

if __name__ == '__main__':
    unittest.main()
