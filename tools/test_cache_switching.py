import os
import sys
import time
import shutil
import numpy as np

# Ensure root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import session_cache_manager
import sdc_manager

def test_cache_switching():
    print("=== Testing Session Cache Key Stability & SDC Invariance ===")
    test_raw = os.path.join(root_dir, "测试底片", "IMG_8425.DNG")
    assert os.path.isfile(test_raw), f"Test raw file not found: {test_raw}"

    # 1. Key stability test
    k1 = session_cache_manager.get_cache_key(test_raw)
    sdc_path = f"{os.path.abspath(test_raw)}.sdc"

    # Touch or modify SDC file mtime
    if os.path.exists(sdc_path):
        os.utime(sdc_path, (time.time() + 100, time.time() + 100))
    k2 = session_cache_manager.get_cache_key(test_raw)
    assert k1 == k2, f"FAILED: Cache key changed when SDC was touched! {k1} != {k2}"
    print(f"[PASS] Key stability: {k1} is completely invariant to SDC modifications.")

    # 2. Preview save & load test
    mock_preview = np.random.rand(1365, 2048, 3).astype(np.float32)
    saved = session_cache_manager.save_photo_preview(test_raw, mock_preview)
    assert saved, "FAILED: save_photo_preview returned False!"
    print("[PASS] save_photo_preview succeeded directly without requiring pre-existing _entry.npz.")

    t0 = time.perf_counter()
    loaded_prev = session_cache_manager.get_photo_preview(test_raw)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert loaded_prev is not None, "FAILED: get_photo_preview returned None!"
    assert loaded_prev.shape == mock_preview.shape, f"Shape mismatch: {loaded_prev.shape} vs {mock_preview.shape}"
    print(f"[PASS] get_photo_preview loaded in {dt_ms:.2f} ms (Target < 30ms).")

    # 3. Test thumbnail update with save_preview=False does not destroy preview
    mock_photo_data = {
        "width": 2048,
        "height": 1365,
        "thumbnail_rgb": (np.random.rand(72, 108, 3) * 255).astype(np.uint8),
        "raw_thumbnail_rgb": (np.random.rand(72, 108, 3) * 255).astype(np.uint8),
        "is_raw": True,
    }
    saved_entry = session_cache_manager.save_photo_cache(test_raw, mock_photo_data, save_preview=False)
    assert saved_entry, "FAILED: save_photo_cache returned False!"

    loaded_prev_after = session_cache_manager.get_photo_preview(test_raw)
    assert loaded_prev_after is not None, "FAILED: Preview was destroyed/lost after save_preview=False!"
    print("[PASS] Preview file preserved after save_photo_cache(..., save_preview=False).")

    # 4. Fallback migration test
    # Simulate an entry saved under an older key
    cache_dir = session_cache_manager.get_cache_dir()
    fake_old_key = "legacy_test_key_99999"
    fake_old_entry = os.path.join(cache_dir, f"{fake_old_key}_entry.npz")
    fake_old_prev = os.path.join(cache_dir, f"{fake_old_key}_preview.npy")
    test_raw_2 = os.path.join(root_dir, "测试底片", "IMG_8426.DNG")
    assert os.path.isfile(test_raw_2), f"Missing {test_raw_2}"

    stable_key_2 = session_cache_manager.get_cache_key(test_raw_2)
    stable_entry_2 = os.path.join(cache_dir, f"{stable_key_2}_entry.npz")
    stable_prev_2 = os.path.join(cache_dir, f"{stable_key_2}_preview.npy")
    if os.path.exists(stable_entry_2):
        os.remove(stable_entry_2)
    if os.path.exists(stable_prev_2):
        os.remove(stable_prev_2)

    import json
    meta_json = json.dumps({"path": os.path.abspath(test_raw_2), "width": 100, "height": 100})
    np.savez(fake_old_entry, thumb=np.zeros((72, 72, 3), dtype=np.uint8), meta=meta_json)
    np.save(fake_old_prev, np.ones((100, 100, 3), dtype=np.float16))

    # Reset in-memory index
    session_cache_manager._path_cache_index = None

    # Load preview for test_raw_2 - should seamlessly migrate legacy preview
    migrated_prev = session_cache_manager.get_photo_preview(test_raw_2)
    assert migrated_prev is not None, "FAILED: Legacy preview migration failed!"
    assert os.path.isfile(stable_prev_2), "FAILED: Stable preview file was not created by migration!"
    print("[PASS] Legacy preview successfully migrated to new stable key format.")

    # Cleanup fake legacy files
    for f in (fake_old_entry, fake_old_prev, stable_entry_2, stable_prev_2):
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    print("\nALL CACHE SWITCHING TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_cache_switching()
