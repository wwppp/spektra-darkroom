import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow

app = QApplication.instance() or QApplication([])

class TestAutoexposureMeter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SpektraEngine()

    def test_calculate_auto_exposure_ev(self):
        """Test calculation of 18% middle-gray autoexposure EV."""
        # 1. 18.4% middle-gray linear image should yield approx 0 EV compensation
        mid_gray = np.full((100, 100, 3), 0.184, dtype=np.float32)
        ev_mid = self.engine.calculate_auto_exposure_ev(mid_gray)
        self.assertAlmostEqual(ev_mid, 0.0, delta=0.2)

        # 2. Under-exposed raw sensor data (e.g. mean ~ 0.05) should yield positive compensation
        dark_img = np.full((100, 100, 3), 0.04, dtype=np.float32)
        ev_dark = self.engine.calculate_auto_exposure_ev(dark_img)
        self.assertGreater(ev_dark, 0.5)

    def test_auto_meter_action_in_main_window(self):
        """Test UI action for applying 18% middle-gray autoexposure."""
        win = DarkroomMainWindow(self.engine)
        
        # Mock active photo with dark preview
        fake_photo = {
            "id": "test_p1",
            "path": "test_sample.arw",
            "filename": "test_sample.arw",
            "float_img": np.full((128, 128, 3), 0.05, dtype=np.float32),
            "thumbnail_rgb": np.zeros((72, 72, 3), dtype=np.uint8),
            "auto_ev": 1.25,
            "params": {}
        }
        win.photos = [fake_photo]
        win._current_photo_path = fake_photo["path"]

        # Trigger auto meter action
        win._action_apply_auto_meter()

        # Verify slider and params updated to 1.25 EV
        self.assertAlmostEqual(win.current_params.get("exposure_ev"), 1.25, places=2)
        self.assertAlmostEqual(win.slider_ev.current_val, 1.25, places=2)
        win.close()

if __name__ == '__main__':
    unittest.main()
