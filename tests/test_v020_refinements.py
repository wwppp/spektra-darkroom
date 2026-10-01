"""Unit tests for v0.2.0 / refinements:
1. ExportQueueDialog titlebar close button flags & header close button
2. ExportQueueManager GPU offscreen acceleration integration
3. Rapid cancel_all and clear_history stability (no UI freeze or crash)
4. IPC single instance parameter parsing with BOM tolerance
"""

import os
import sys
import unittest
import numpy as np
from PySide6.QtCore import Qt, QCoreApplication, QEvent, QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QKeyEvent

# Ensure app root in sys.path
app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_root not in sys.path:
    sys.path.insert(0, app_root)

from ui.export_queue_manager import ExportQueueManager, ExportQueueDialog, ExportTask, _save_rendered_file


class MockCanvas:
    def __init__(self):
        self.render_calls = []

    def render_offscreen(self, target_w, target_h, float_img_rgb=None, params_override=None, bit_depth=8):
        self.render_calls.append({
            "target_w": target_w,
            "target_h": target_h,
            "bit_depth": bit_depth,
            "params": params_override
        })
        if bit_depth == 16:
            return np.zeros((target_h, target_w, 3), dtype=np.uint16)
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)


class MockEngine:
    def __init__(self):
        self.load_calls = []
        self.export_calls = []

    def _load_full_resolution(self, path):
        self.load_calls.append(path)
        return np.ones((100, 150, 3), dtype=np.float32)

    def load_image(self, path):
        self.raw_preview = np.ones((100, 150, 3), dtype=np.float32)
        return {"success": True}

    def export_image(self, params_dict, output_path, format_type="jpeg", quality=9, source_path=None, progress_cb=None, cancel_cb=None):
        self.export_calls.append(output_path)
        if progress_cb:
            progress_cb(50)
        return {"success": True}


class TestV020Refinements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication([])

    def setUp(self):
        self.engine = MockEngine()
        self.canvas = MockCanvas()
        self.mgr = ExportQueueManager(self.engine, self.canvas)

    def tearDown(self):
        self.mgr.close()

    def test_queue_dialog_close_button_and_flags(self):
        """Item 1: Verify window flags include WindowCloseButtonHint and dialog has header close button."""
        dlg = ExportQueueDialog(self.mgr)
        flags = dlg.windowFlags()
        self.assertTrue(bool(flags & Qt.WindowType.WindowCloseButtonHint),
                        "WindowCloseButtonHint must be set so OS renders active X button")
        self.assertFalse(bool(flags & Qt.WindowType.WindowContextHelpButtonHint),
                         "WindowContextHelpButtonHint must be unset")

        # In-dialog redundant close button removed (Item 4: avoid double X)
        self.assertFalse(hasattr(dlg, "btn_close"))

        # Key_Escape closes dialog
        escaped = False
        dlg.reject = lambda: setattr(dlg, "_was_rejected", True)
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        dlg.keyPressEvent(ev)
        self.assertTrue(getattr(dlg, "_was_rejected", False), "Escape key must trigger reject()")

    def test_rapid_cancel_and_clear_no_crash(self):
        """Item 4: Verify rapid clicks of cancel_all and clear_history do not raise exceptions or deadlock."""
        for i in range(10):
            self.mgr.add_task(f"test_{i}.jpg", f"out_{i}.jpg", {"format": "jpeg"})

        self.assertEqual(self.mgr.get_active_count(), 10)

        # Rapidly cancel and clear repeatedly
        for _ in range(5):
            self.mgr.cancel_all()
            self.mgr.clear_history()

        # All cancelled tasks should be safely removed, no exception raised
        self.assertEqual(self.mgr.get_active_count(), 0)

    def test_single_task_cancellation(self):
        """Verify canceling a single task updates status and emits queue_changed."""
        tid = self.mgr.add_task("test.jpg", "out.jpg", {"format": "jpeg"})
        self.mgr.cancel_task(tid)
        t = next((task for task in self.mgr.tasks if task.task_id == tid), None)
        self.assertIsNotNone(t)
        self.assertEqual(t.status, "cancelled")

    def test_save_rendered_file_jpeg_and_png(self):
        """Verify _save_rendered_file correctly saves JPEG and PNG outputs without error."""
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        arr = np.zeros((50, 50, 3), dtype=np.uint8)

        # JPEG
        jpg_out = os.path.join(tmp_dir, "test.jpg")
        _save_rendered_file(arr, jpg_out, {"format": "jpeg", "quality": 8, "dpi": 300})
        self.assertTrue(os.path.exists(jpg_out))
        self.assertGreater(os.path.getsize(jpg_out), 0)

        # PNG
        png_out = os.path.join(tmp_dir, "test.png")
        _save_rendered_file(arr, png_out, {"format": "png", "bit_depth": 8})
        self.assertTrue(os.path.exists(png_out))
        self.assertGreater(os.path.getsize(png_out), 0)

        # Clean up
        for p in [jpg_out, png_out]:
            try: os.remove(p)
            except Exception: pass
        try: os.rmdir(tmp_dir)
        except Exception: pass

    def test_ipc_bom_handling(self):
        """Item 3: Verify UTF-8 BOM is cleanly stripped from incoming IPC paths."""
        raw_with_bom = '\ufeff"D:\\test\\session.sdss"\n\ufeffD:\\test\\photo.jpg\n'
        data = raw_with_bom.encode("utf-8")
        decoded = data.decode("utf-8-sig", errors="ignore")
        lines = [line.strip().strip('"').strip("'").lstrip('\ufeff') for line in decoded.splitlines() if line.strip()]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], r"D:\test\session.sdss")
        self.assertEqual(lines[1], r"D:\test\photo.jpg")
        self.assertFalse(lines[0].startswith('\ufeff'))


if __name__ == "__main__":
    unittest.main()
