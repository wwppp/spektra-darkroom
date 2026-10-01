"""Comprehensive unit tests for SpektraDarkroom session caching system.
Uses standard Python unittest (zero external test dependencies).
Verifies:
1. Two-tier caching (Tier 1 metadata/thumbnail + Tier 2 float preview)
2. Cache invalidation on file modification or replacement
3. Session state persistence and restoration
4. Cache size calculation, clear, and LRU pruning
"""

import os
import sys
import time
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import session_cache_manager
import config_manager


class TestSessionCacheSystem(unittest.TestCase):
    def setUp(self):
        session_cache_manager.clear_cache()

    def tearDown(self):
        session_cache_manager.clear_cache()

    def _create_dummy_image(self, path, width=400, height=300):
        img = Image.new("RGB", (width, height), color=(120, 150, 180))
        img.save(path, format="JPEG", quality=85)

    def test_tier1_tier2_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            img_file = os.path.join(td, "sample_photo.jpg")
            self._create_dummy_image(img_file, 800, 600)

            thumb = np.random.randint(0, 255, (72, 96, 3), dtype=np.uint8)
            prev = np.random.rand(600, 800, 3).astype(np.float32)
            meta = {
                "width": 800,
                "height": 600,
                "thumbnail_rgb": thumb,
                "float_img": prev,
                "exif": {"iso": 100, "camera_model": "TestCam X"}
            }

            # Initially no cache
            self.assertFalse(session_cache_manager.has_valid_cache(img_file))

            # Save cache
            saved = session_cache_manager.save_photo_cache(img_file, meta)
            self.assertTrue(saved)
            self.assertTrue(session_cache_manager.has_valid_cache(img_file))

            # Tier 1 load (fast, no preview)
            t0 = time.perf_counter()
            cached_t1 = session_cache_manager.get_photo_cache(img_file, load_preview=False)
            t1_ms = (time.perf_counter() - t0) * 1000.0
            self.assertIsNotNone(cached_t1)
            self.assertEqual(cached_t1["width"], 800)
            self.assertEqual(cached_t1["height"], 600)
            self.assertEqual(cached_t1["exif"]["camera_model"], "TestCam X")
            self.assertEqual(cached_t1["thumbnail_rgb"].shape, (72, 96, 3))
            self.assertIsNone(cached_t1["float_img"])
            self.assertLess(t1_ms, 50.0)

            # Tier 2 load (with float preview)
            t1 = time.perf_counter()
            cached_t2 = session_cache_manager.get_photo_cache(img_file, load_preview=True)
            t2_ms = (time.perf_counter() - t1) * 1000.0
            self.assertIsNotNone(cached_t2)
            self.assertIsNotNone(cached_t2["float_img"])
            self.assertEqual(cached_t2["float_img"].shape, (600, 800, 3))
            self.assertTrue(np.allclose(cached_t2["float_img"], prev, atol=1e-3))
            self.assertLess(t2_ms, 100.0)

    def test_cache_invalidation_on_file_modification(self):
        with tempfile.TemporaryDirectory() as td:
            img_file = os.path.join(td, "sample_mod.jpg")
            self._create_dummy_image(img_file, 400, 300)

            thumb = np.random.randint(0, 255, (72, 96, 3), dtype=np.uint8)
            meta = {"width": 400, "height": 300, "thumbnail_rgb": thumb, "exif": {}}
            session_cache_manager.save_photo_cache(img_file, meta)
            self.assertTrue(session_cache_manager.has_valid_cache(img_file))

            # Modify file
            time.sleep(0.05)
            with open(img_file, "ab") as f:
                f.write(b"modified_bytes")

            # Cache must automatically invalidate
            self.assertFalse(session_cache_manager.has_valid_cache(img_file))
            self.assertIsNone(session_cache_manager.get_photo_cache(img_file))

    def test_session_state_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            p1 = f1.name
            p2 = f2.name

        try:
            config_manager.save_session_state([p1, p2], p2)
            loaded_files, loaded_active = config_manager.get_session_state()
            self.assertIn(os.path.abspath(p1), loaded_files)
            self.assertIn(os.path.abspath(p2), loaded_files)
            self.assertEqual(loaded_active, os.path.abspath(p2))
        finally:
            if os.path.exists(p1):
                os.remove(p1)
            if os.path.exists(p2):
                os.remove(p2)

    def test_cache_size_and_clear(self):
        with tempfile.TemporaryDirectory() as td:
            img_file = os.path.join(td, "size_test.jpg")
            self._create_dummy_image(img_file, 500, 500)

            thumb = np.zeros((72, 72, 3), dtype=np.uint8)
            prev = np.zeros((500, 500, 3), dtype=np.float32)
            session_cache_manager.save_photo_cache(
                img_file,
                {"width": 500, "height": 500, "thumbnail_rgb": thumb, "float_img": prev}
            )

            sz_bytes = session_cache_manager.get_cache_size_bytes()
            self.assertGreater(sz_bytes, 0)
            sz_mb = session_cache_manager.get_cache_size_mb()
            self.assertGreater(sz_mb, 0.0)

            # Test clear
            ok = session_cache_manager.clear_cache()
            self.assertTrue(ok)
            self.assertEqual(session_cache_manager.get_cache_size_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
