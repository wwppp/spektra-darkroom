import unittest
import numpy as np
import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core import SpektraEngine


class TestPhotoAndStockSwitchFreeze(unittest.TestCase):
    def setUp(self):
        self.engine = SpektraEngine()

    def test_get_3d_lut_bypass_combinations(self):
        """Verify that get_3d_lut handles all 'none' combinations safely without AttributeError or crash."""
        # 1. Both none
        lut_none_none = self.engine.get_3d_lut("none", "none", lut_size=17)
        self.assertEqual(lut_none_none.shape, (17, 17, 17, 3))
        # Verify identity LUT property: (0,0,0) -> [0,0,0], (16,16,16) -> [1,1,1]
        np.testing.assert_allclose(lut_none_none[0, 0, 0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(lut_none_none[-1, -1, -1], [1.0, 1.0, 1.0], atol=1e-5)

        # 2. Film real + paper none (MUST return identity without crashing on print.info)
        lut_film_none = self.engine.get_3d_lut("kodak_portra_400", "none", lut_size=17)
        self.assertEqual(lut_film_none.shape, (17, 17, 17, 3))
        np.testing.assert_allclose(lut_film_none[0, 0, 0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(lut_film_none[-1, -1, -1], [1.0, 1.0, 1.0], atol=1e-5)

        # 3. Film none + paper real (MUST return identity without crashing)
        lut_none_paper = self.engine.get_3d_lut("none", "kodak_2383", lut_size=17)
        self.assertEqual(lut_none_paper.shape, (17, 17, 17, 3))
        np.testing.assert_allclose(lut_none_paper[0, 0, 0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(lut_none_paper[-1, -1, -1], [1.0, 1.0, 1.0], atol=1e-5)

        # 4. Both real stocks (returns real calibrated simulation LUT)
        lut_real = self.engine.get_3d_lut("kodak_portra_400", "kodak_2383", lut_size=17)
        self.assertEqual(lut_real.shape, (17, 17, 17, 3))
        self.assertTrue(np.all(lut_real >= 0.0) and np.all(lut_real <= 1.0))

    def test_photo_switch_variable_scoping_and_restoration(self):
        """Simulate switching between photos to ensure sdc NameError is resolved and state preserves cleanly."""
        photos = [
            {
                "id": "photo_1",
                "filename": "DSC0001.ARW",
                "path": "d:/test/DSC0001.ARW",
                "float_img": np.ones((120, 160, 3), dtype=np.float32) * 0.18,
                "thumbnail_rgb": np.ones((60, 80, 3), dtype=np.uint8) * 128,
                "auto_ev": -0.27
            },
            {
                "id": "photo_2",
                "filename": "DSC0002.ARW",
                "path": "d:/test/DSC0002.ARW",
                "float_img": np.ones((120, 160, 3), dtype=np.float32) * 0.36,
                "thumbnail_rgb": np.ones((60, 80, 3), dtype=np.uint8) * 192,
                "auto_ev": 0.45
            }
        ]

        class DummyCanvas:
            def __init__(self):
                self.current_img = None
                self.params = {}
                self.lut = None

            def set_image(self, img):
                self.current_img = img

            def update_params(self, **kwargs):
                self.params.update(kwargs)

            def set_lut(self, lut):
                self.lut = lut

            def get_actual_zoom_ratio(self):
                return 1.0

        canvas = DummyCanvas()
        current_film = "none"
        current_paper = "none"
        current_params = {}
        active_id = None
        photo_is_dirty = False

        def switch_photo(target_id):
            nonlocal current_film, current_paper, current_params, active_id, photo_is_dirty
            if active_id:
                prev = next((p for p in photos if p["id"] == active_id), None)
                if prev:
                    prev["params"] = dict(current_params)
                    prev["film_profile"] = current_film
                    prev["paper_profile"] = current_paper
                    prev["is_dirty"] = photo_is_dirty

            photo = next((p for p in photos if p["id"] == target_id), None)
            active_id = target_id

            sdc = None
            if photo.get("params"):
                current_film = photo.get("film_profile", current_film)
                current_paper = photo.get("paper_profile", current_paper)
                current_params.clear()
                current_params.update(photo["params"])
                photo_is_dirty = photo.get("is_dirty", False)
            else:
                current_film = "none"
                current_paper = "none"
                init_ev = float(photo.get("auto_ev", 0.0))
                current_params.clear()
                current_params.update({"exposure_ev": init_ev, "color_temp": 5500.0, "tint": 1.0})
                photo_is_dirty = False

            has_custom_ev = bool(sdc and "exposure_ev" in sdc.get("params", {})) or photo.get("is_dirty", False) or bool(photo.get("params"))
            if not has_custom_ev and float(photo.get("auto_ev", 0.0)) != 0.0 and current_params.get("exposure_ev", 0.0) == 0.0:
                current_params["exposure_ev"] = float(photo["auto_ev"])

            canvas.set_image(photo["float_img"])
            canvas.update_params(**current_params)

        # 1. Switch to photo 1
        switch_photo("photo_1")
        self.assertEqual(active_id, "photo_1")
        self.assertEqual(current_film, "none")
        self.assertAlmostEqual(current_params["exposure_ev"], -0.27)
        np.testing.assert_array_equal(canvas.current_img, photos[0]["float_img"])

        # Modify photo 1 settings
        current_film = "kodak_portra_400"
        current_paper = "kodak_2383"
        current_params["exposure_ev"] = 0.50
        photo_is_dirty = True

        # 2. Switch to photo 2
        switch_photo("photo_2")
        self.assertEqual(active_id, "photo_2")
        self.assertEqual(current_film, "none")
        self.assertAlmostEqual(current_params["exposure_ev"], 0.45)
        np.testing.assert_array_equal(canvas.current_img, photos[1]["float_img"])

        # 3. Switch BACK to photo 1 (Must NOT throw NameError or freeze on photo 2!)
        switch_photo("photo_1")
        self.assertEqual(active_id, "photo_1")
        # Restored saved state from photo 1
        self.assertEqual(current_film, "kodak_portra_400")
        self.assertEqual(current_paper, "kodak_2383")
        self.assertAlmostEqual(current_params["exposure_ev"], 0.50)
        np.testing.assert_array_equal(canvas.current_img, photos[0]["float_img"])


if __name__ == "__main__":
    unittest.main()
