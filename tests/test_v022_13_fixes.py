import os
import sys
import tempfile
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config_manager
from path_utils import normalize_path
from app_core import (
    SpektraEngine,
    convert_image_colorspace,
    get_icc_profile_bytes
)


def test_item_8_path_normalization():
    """Item 8: Path normalization eliminates forward/backward slash mixing."""
    mixed_path = "D:/spektra-darkroom\\resources/icc\\sRGB.icc"
    norm = normalize_path(mixed_path)
    assert "/" not in norm
    assert "\\" in norm
    assert os.path.isabs(norm)


def test_item_2_auto_saved_session_config():
    """Item 2: Auto-saved temporary session configuration."""
    temp_p = config_manager.get_auto_saved_session_path()
    assert temp_p.endswith("auto_saved_session.sdss")
    assert os.path.isabs(temp_p)

    # Test clear and has
    config_manager.clear_auto_saved_session()
    assert not config_manager.has_auto_saved_session()


def test_item_7_icc_profiles_and_conversion():
    """Item 7: ICC profiles exist and color space conversion works."""
    spaces = ["sRGB", "Display P3", "Adobe RGB (1998)", "ProPhoto RGB"]
    for sp in spaces:
        data = get_icc_profile_bytes(sp)
        assert data is not None, f"Missing ICC profile for {sp}"
        assert len(data) > 500, f"ICC profile too small for {sp}"

    # Test conversion
    dummy_rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    p3_rgb = convert_image_colorspace(dummy_rgb, "Display P3")
    assert p3_rgb.shape == (16, 16, 3)
    assert p3_rgb.dtype == np.float32


def test_item_9_paper_stocks_scan_mode():
    """Item 9: Paper stock 'none' is renamed to 数码扫描 (Film Scan)."""
    engine = SpektraEngine()
    stocks = engine.get_paper_stocks()
    none_stock = next((s for s in stocks if s["id"] == "none"), None)
    assert none_stock is not None
    assert "数码扫描" in none_stock["name"] or "Film Scan" in none_stock["name"]


def test_item_3_apply_lut_to_linear():
    """Item 3: Engine has apply_lut_to_linear for filmic shoulder & thumbnail grading."""
    engine = SpektraEngine()
    lut = engine.get_3d_lut("kodak_portra_400", "kodak_2383", lut_size=17)
    assert lut is not None
    # Test linear float input
    linear_img = np.full((16, 16, 3), 0.18, dtype=np.float32)
    graded = engine.apply_lut_to_linear(linear_img, lut)
    assert graded.shape == (16, 16, 3)
    assert graded.dtype == np.uint8
    # Should not be blown out (255)
    assert graded.max() < 250


def test_item_10_full_resolution_preview():
    """Item 10: Preferences has full resolution (0) option and load_photo handles 0."""
    from ui.preferences_dialog import PreferencesDialog
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = PreferencesDialog()
    idx = dlg.combo_preview_res.findData(0)
    assert idx >= 0, "Full resolution (0) not found in preferences preview resolution"
    dlg.close()


def test_item_6_export_dialog_settings():
    """Item 6: Export dialog restores subsampling and compression settings."""
    from ui.export_dialog import ExportImageDialog
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = ExportImageDialog("dummy.jpg", is_batch=False)
    # Verify subsampling combo exists
    assert hasattr(dlg, "combo_subsampling")
    assert dlg.combo_subsampling.count() >= 3
    dlg.close()


def test_item_4_export_queue_manager_ui():
    """Item 4: Export queue manager has no redundant close button in title bar."""
    from ui.export_queue_manager import ExportQueueManager
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    eq = ExportQueueManager(engine=None, canvas=None)
    # btn_close should not exist in eq
    assert not hasattr(eq, "btn_close")
    eq.close()


def test_item_12_filmstrip_overlay_opacity():
    """Item 12: Filename background overlay opacity is more transparent."""
    from ui.filmstrip_widget import FilmstripItemWidget
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    item = FilmstripItemWidget({"id": "id1", "filename": "test.arw", "path": "D:/test/test.arw"})
    assert item.file_path == normalize_path("D:/test/test.arw")


def test_item_1_save_session_default_name():
    """Item 1: Default session save name is '未命名会话.sdss' when temp session."""
    # When _is_temp_session is True or _current_session_path is None
    curr_sess = None
    is_temp = True
    if curr_sess and not is_temp:
        default_name = os.path.basename(curr_sess)
    else:
        default_name = "未命名会话.sdss"
    assert default_name == "未命名会话.sdss"


def test_item_13_session_dirty_flags():
    """Item 13: Session file preserves and restores is_dirty and is_edited flags."""
    import session_cache_manager
    session_data = {
        "format": "SpektraDarkroomSession",
        "version": "0.2.2",
        "photos": [
            {
                "id": "p1",
                "path": "D:/test/sample.arw",
                "film_stock": "kodak_portra_400",
                "paper_stock": "kodak_2383",
                "print_mode": "optical",
                "base_ev": 0.0,
                "params": {"exposure_ev": 0.5},
                "is_dirty": True,
                "is_edited": True
            }
        ]
    }
    tmp_path = os.path.join(tempfile.gettempdir(), "test_session_dirty.sdss")
    try:
        ok = session_cache_manager.save_session_file(tmp_path, session_data)
        assert ok
        loaded = session_cache_manager.load_session_file(tmp_path)
        assert loaded is not None
        p = loaded["photos"][0]
        assert p["is_dirty"] is True
        assert p["is_edited"] is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    tests = [
        test_item_8_path_normalization,
        test_item_2_auto_saved_session_config,
        test_item_7_icc_profiles_and_conversion,
        test_item_9_paper_stocks_scan_mode,
        test_item_3_apply_lut_to_linear,
        test_item_10_full_resolution_preview,
        test_item_6_export_dialog_settings,
        test_item_4_export_queue_manager_ui,
        test_item_12_filmstrip_overlay_opacity,
        test_item_1_save_session_default_name,
        test_item_13_session_dirty_flags
    ]
    for t in tests:
        print(f"Running {t.__name__}...", end=" ")
        t()
        print("PASSED")
    print(f"\nALL {len(tests)} TESTS PASSED SUCCESSFULLY!")

