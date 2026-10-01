import os
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from ui.canvas_viewport import DarkroomGLCanvas, FRAGMENT_SHADER_SRC
from ui.main_window import DarkroomMainWindow
from app_core import SpektraEngine

app = QApplication.instance() or QApplication([])

class TestGrainAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SpektraEngine()
        cls.win = DarkroomMainWindow(cls.engine)
        cls.win.resize(900, 600)
        cls.win.show()
        cls.win.canvas.grabFramebuffer()
        app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls.win.deleteLater()
        app.processEvents()

    def test_shader_film_space_alignment_code(self):
        """Verify the fragment shader source code implements film-space grain anchoring and downsample integration."""
        self.assertIn("uniform vec2 u_master_resolution;", FRAGMENT_SHADER_SRC)
        self.assertIn("sample_uv * master_res", FRAGMENT_SHADER_SRC)
        self.assertIn("fwidth(gcoord)", FRAGMENT_SHADER_SRC)
        self.assertIn("crystal_att", FRAGMENT_SHADER_SRC)
        self.assertIn("cloud_att", FRAGMENT_SHADER_SRC)

    def test_set_image_master_dimensions(self):
        """Verify set_image correctly records master_width and master_height."""
        preview_img = np.ones((200, 300, 3), dtype=np.float32) * 0.5
        self.win.canvas.set_image(preview_img, master_width=6024, master_height=4024)
        
        self.assertEqual(self.win.canvas._image_width, 300)
        self.assertEqual(self.win.canvas._image_height, 200)
        self.assertEqual(self.win.canvas._master_width, 6024)
        self.assertEqual(self.win.canvas._master_height, 4024)

    def test_render_offscreen_grain_alignment(self):
        """Verify offscreen rendering with aligned grain runs cleanly."""
        if not self.win.canvas.isValid():
            self.skipTest("OpenGL widget not exposed by windowing system in headless test runner")
        img = np.ones((100, 150, 3), dtype=np.float32) * 0.4
        self.win.canvas.set_image(img, master_width=3000, master_height=2000)
        self.win.canvas.update_params(grain=0.8, grain_size=1.2, film_format_mm=35.0)

        # Offscreen render
        rendered = self.win.canvas.render_offscreen(target_w=150, target_h=100, float_img_rgb=img)
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.shape, (100, 150, 3))
        self.assertEqual(rendered.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
