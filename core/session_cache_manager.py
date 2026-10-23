"""Session Cache Manager for SpektraDarkroom
High-performance two-tier disk cache for imported negative sessions:
Tier 1: Instant filmstrip thumbnails and EXIF metadata (.npz, ~15KB, loads in <1ms)
Tier 2: High-fidelity viewport preview textures (.npy float16, loads in ~30ms)

Provides sub-100ms session restoration without re-decoding heavy camera RAW files.
Includes SHA256-based cache invalidation (path + mtime + size) to automatically
refresh if original media files are modified or replaced.
"""

import os
import sys
import time
import json
import hashlib
import logging
import shutil
import threading
import numpy as np

logger = logging.getLogger("SpektraDarkroom.SessionCache")

from .format_constants import RAW_EXTENSIONS


def is_dummy_or_corrupted_thumbnail(arr) -> bool:
    """
    Determines if a thumbnail array is a dummy placeholder or corrupted frame.
    Returns True if the array is None, not an ndarray, empty, uninitialized dark frame,
    or solid uniform placeholder color (e.g., 0 or 45).
    """
    if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
        return True
    if len(arr.shape) < 2 or min(arr.shape[:2]) < 8:
        return True
    if arr.max() <= 1:
        return True
    if arr.mean() < 0.5 and arr.max() < 3:
        return True
    # Catch any solid uniform placeholder color (e.g. 0, 45, 100, 255)
    if arr.min() == arr.max():
        return True
    # Catch near-zero variance flat noise (std < 0.5)
    if float(arr.std()) < 0.5:
        return True
    return False


def is_valid_thumbnail(arr) -> bool:
    """Convenience inverse of is_dummy_or_corrupted_thumbnail."""
    return not is_dummy_or_corrupted_thumbnail(arr)


_CACHED_SESSION_DIR = None
_CACHE_DIR_LOCK = threading.Lock()


def reset_cache_dir():
    """Resets cached session directory memoization to pick up updated configuration."""
    global _CACHED_SESSION_DIR
    with _CACHE_DIR_LOCK:
        _CACHED_SESSION_DIR = None


def get_cache_dir() -> str:
    """Returns path to the session cache folder, creating it if needed.
    Memoizes the validated path to eliminate multi-threaded filesystem race conditions.
    """
    global _CACHED_SESSION_DIR
    if _CACHED_SESSION_DIR is not None and os.path.isdir(_CACHED_SESSION_DIR):
        return _CACHED_SESSION_DIR

    with _CACHE_DIR_LOCK:
        if _CACHED_SESSION_DIR is not None and os.path.isdir(_CACHED_SESSION_DIR):
            return _CACHED_SESSION_DIR

        from .path_utils import get_user_session_cache_dir
        from . import config_manager
        app_dir = config_manager.get_app_dir()
        portable_marker = os.path.join(app_dir, "portable.txt")

        # 1. Portable mode if marker exists
        if os.path.exists(portable_marker):
            cache_path = os.path.join(app_dir, ".session_cache")
        else:
            # 2. Standard user local cache (%LOCALAPPDATA%/SpektraDarkroom/session_cache)
            cache_path = get_user_session_cache_dir()

        try:
            os.makedirs(cache_path, exist_ok=True)
        except Exception:
            pass

        _CACHED_SESSION_DIR = cache_path
        return _CACHED_SESSION_DIR


def get_cache_key(file_path: str, normcase: bool = True) -> str:
    """Generates a stable cache key based on absolute path, file modification time, and size.
    Notice: SDC sidecar mtime is intentionally excluded because RAW sensor demosaicing data
    (float_img / raw_preview) is strictly invariant to SDC grading parameter changes.
    """
    abs_p = os.path.abspath(file_path)
    norm_path = os.path.normcase(abs_p) if normcase else abs_p
    try:
        st = os.stat(abs_p)
        mtime = st.st_mtime
        size = st.st_size
    except OSError:
        mtime = 0.0
        size = 0
    max_edge = 2048
    try:
        from . import config_manager
        max_edge = int(config_manager.get_preferences().get("preview_max_edge", 2048))
    except Exception:
        max_edge = 2048
    token = f"prophoto_v3|{norm_path}|{mtime}|{size}|res_{max_edge}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:24]


_path_cache_index = None

def _ensure_path_cache_index():
    global _path_cache_index
    if _path_cache_index is None:
        _path_cache_index = {}
        cache_dir = get_cache_dir()
        try:
            if os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    if f.endswith("_entry.npz"):
                        f_path = os.path.join(cache_dir, f)
                        k = f[:-10]
                        prev_path = os.path.join(cache_dir, f"{k}_preview.npy")
                        has_prev = os.path.isfile(prev_path)
                        try:
                            with np.load(f_path, allow_pickle=False) as npz:
                                meta = json.loads(str(npz["meta"]))
                                p = meta.get("path")
                                if p:
                                    ap = os.path.abspath(p)
                                    # Prioritize key that already has preview.npy on disk
                                    if ap not in _path_cache_index or (has_prev and not os.path.isfile(os.path.join(cache_dir, f"{_path_cache_index[ap]}_preview.npy"))):
                                        _path_cache_index[ap] = k
                        except Exception:
                            pass
        except Exception:
            pass


def _migrate_preview_from_index(norm_path: str, primary_prev: str, cache_dir: str):
    """If primary_prev is missing, attempts to migrate preview array from an older key matching norm_path."""
    _ensure_path_cache_index()
    old_key = _path_cache_index.get(norm_path)
    if old_key:
        old_prev = os.path.join(cache_dir, f"{old_key}_preview.npy")
        if os.path.isfile(old_prev) and not os.path.isfile(primary_prev):
            try:
                shutil.copy2(old_prev, primary_prev)
            except Exception:
                pass


def _get_entry_path_with_fallback(file_path: str) -> tuple[str | None, str | None]:
    """Returns (entry_path, key) for a file, supporting seamless fallback to legacy/alternate keys."""
    if not file_path or not os.path.isfile(file_path):
        return None, None
    key = get_cache_key(file_path, normcase=True)
    cache_dir = get_cache_dir()
    primary_entry = os.path.join(cache_dir, f"{key}_entry.npz")
    primary_prev = os.path.join(cache_dir, f"{key}_preview.npy")
    norm_path = os.path.abspath(file_path)

    if os.path.isfile(primary_entry):
        if not os.path.isfile(primary_prev):
            _migrate_preview_from_index(norm_path, primary_prev, cache_dir)
        return primary_entry, key

    # Check un-normcased legacy key if case differed on Windows
    legacy_key = get_cache_key(file_path, normcase=False)
    if legacy_key != key:
        legacy_entry = os.path.join(cache_dir, f"{legacy_key}_entry.npz")
        if os.path.isfile(legacy_entry):
            try:
                shutil.copy2(legacy_entry, primary_entry)
                legacy_prev = os.path.join(cache_dir, f"{legacy_key}_preview.npy")
                if os.path.isfile(legacy_prev) and not os.path.isfile(primary_prev):
                    shutil.copy2(legacy_prev, primary_prev)
            except Exception:
                pass
            return primary_entry, key

    # Check fallback index for existing cached entry matching path
    _ensure_path_cache_index()
    old_key = _path_cache_index.get(norm_path)
    if old_key:
        old_entry = os.path.join(cache_dir, f"{old_key}_entry.npz")
        if os.path.isfile(old_entry):
            try:
                shutil.copy2(old_entry, primary_entry)
                old_prev = os.path.join(cache_dir, f"{old_key}_preview.npy")
                if os.path.isfile(old_prev) and not os.path.isfile(primary_prev):
                    shutil.copy2(old_prev, primary_prev)
            except Exception:
                pass
            return primary_entry, key

    return None, None


def has_valid_cache(file_path: str) -> bool:
    """Checks whether a valid, non-stale cache entry exists for the given file."""
    entry_path, _ = _get_entry_path_with_fallback(file_path)
    return entry_path is not None and os.path.isfile(entry_path)


def has_cached_preview(file_path: str) -> bool:
    """Checks whether float preview (.npy) exists on disk."""
    if not file_path:
        return False
    key = get_cache_key(file_path)
    cache_dir = get_cache_dir()
    preview_path = os.path.join(cache_dir, f"{key}_preview.npy")
    return os.path.isfile(preview_path)


def save_photo_cache(file_path: str, photo_data: dict, preview_array: np.ndarray = None, save_preview: bool = True) -> bool:
    """Saves photo metadata, thumbnail, and optional float preview to disk cache incrementally."""
    if not file_path or not os.path.isfile(file_path):
        return False

    try:
        cache_dir = get_cache_dir()
        key = get_cache_key(file_path)
        entry_path = os.path.join(cache_dir, f"{key}_entry.npz")
        preview_path = os.path.join(cache_dir, f"{key}_preview.npy")

        # Ensure width and height orientation matches thumbnail
        pw = int(photo_data.get("width", 0))
        ph = int(photo_data.get("height", 0))
        if (pw <= 0 or ph <= 0) and file_path and os.path.isfile(file_path):
            dw, dh = get_image_dimensions(file_path)
            if dw > 0 and dh > 0:
                pw, ph = dw, dh

        ref_thumb = photo_data.get("thumbnail_rgb") if photo_data.get("thumbnail_rgb") is not None else photo_data.get("raw_thumbnail_rgb")
        if ref_thumb is not None and hasattr(ref_thumb, "shape") and len(ref_thumb.shape) >= 2:
            th_h, th_w = ref_thumb.shape[:2]
            if (th_h > th_w and pw > ph) or (th_w > th_h and ph > pw):
                pw, ph = ph, pw

        # 1. Prepare metadata
        meta = {
            "path": os.path.abspath(file_path),
            "filename": os.path.basename(file_path),
            "width": pw,
            "height": ph,
            "exif": photo_data.get("exif", {}),
            "auto_ev": float(photo_data.get("auto_ev", 0.0)),
            "base_ev": float(photo_data.get("base_ev", 0.0)),
            "is_raw": bool(photo_data.get("is_raw", False))
        }
        meta_json = json.dumps(meta, default=str, ensure_ascii=False)

        # 2. Thumbnail handling: Raw base vs Graded
        thumb = photo_data.get("thumbnail_rgb")
        raw_thumb = photo_data.get("raw_thumbnail_rgb")
        if raw_thumb is None or (isinstance(raw_thumb, np.ndarray) and raw_thumb.size == 0):
            raw_thumb = photo_data.get("raw_thumbnail")
        if raw_thumb is None or (isinstance(raw_thumb, np.ndarray) and raw_thumb.size == 0):
            if not photo_data.get("is_edited"):
                raw_thumb = np.copy(thumb) if thumb is not None else None
            else:
                # Attempt healing from preview array
                prev = preview_array if preview_array is not None else photo_data.get("float_img")
                if prev is None:
                    prev = get_photo_preview(file_path)
                if prev is not None and isinstance(prev, np.ndarray) and prev.size > 0:
                    try:
                        import cv2
                        from .app_core import prophoto_to_srgb_filmic
                        th_h = 72
                        th_w = max(48, int(prev.shape[1] * (th_h / float(prev.shape[0]))))
                        th_img = cv2.resize(prev, (th_w, th_h), interpolation=cv2.INTER_AREA)
                        raw_thumb = (prophoto_to_srgb_filmic(th_img) * 255.0).astype(np.uint8)
                    except Exception:
                        pass
        graded_thumb = photo_data.get("graded_thumbnail_rgb")

        # Guard: Never write dummy arrays or uninitialized black frames to disk cache
        if is_dummy_or_corrupted_thumbnail(thumb):
            healed = extract_fast_thumbnail(file_path)
            if healed is not None and not is_dummy_or_corrupted_thumbnail(healed):
                thumb = healed
        if is_dummy_or_corrupted_thumbnail(raw_thumb):
            if not is_dummy_or_corrupted_thumbnail(thumb):
                raw_thumb = thumb
            else:
                healed = extract_fast_thumbnail(file_path)
                if healed is not None and not is_dummy_or_corrupted_thumbnail(healed):
                    raw_thumb = healed
                    thumb = healed

        def _to_u8(arr):
            if arr is None:
                return None
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            if np.issubdtype(arr.dtype, np.floating):
                return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            return arr.astype(np.uint8)

        thumb_u8 = _to_u8(thumb) if thumb is not None else np.zeros((72, 72, 3), dtype=np.uint8)
        raw_u8 = _to_u8(raw_thumb) if raw_thumb is not None else thumb_u8
        graded_u8 = _to_u8(graded_thumb) if (graded_thumb is not None and graded_thumb.size > 0) else np.zeros((0,), dtype=np.uint8)

        # Atomic write entry
        import threading
        nonce = f"{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
        tmp_entry = os.path.join(cache_dir, f"{key}_entry_tmp_{nonce}.npz")
        np.savez(tmp_entry, thumb=thumb_u8, raw_thumb=raw_u8, graded_thumb=graded_u8, meta=meta_json)
        if os.path.exists(entry_path):
            try:
                os.remove(entry_path)
            except Exception:
                pass
        try:
            os.replace(tmp_entry, entry_path)
        except Exception:
            if os.path.exists(tmp_entry):
                try:
                    os.remove(tmp_entry)
                except Exception:
                    pass

        # 3. Save preview if enabled and available
        if save_preview:
            preview = preview_array if preview_array is not None else photo_data.get("float_img")
            if preview is not None and isinstance(preview, np.ndarray):
                tmp_prev = os.path.join(cache_dir, f"{key}_preview_tmp_{nonce}.npy")
                np.save(tmp_prev, preview.astype(np.float16))
                if os.path.exists(preview_path):
                    try:
                        os.remove(preview_path)
                    except Exception:
                        pass
                try:
                    os.replace(tmp_prev, preview_path)
                except Exception:
                    if os.path.exists(tmp_prev):
                        try:
                            os.remove(tmp_prev)
                        except Exception:
                            pass
        else:
            if not os.path.isfile(preview_path):
                norm_p = os.path.abspath(file_path)
                _migrate_preview_from_index(norm_p, preview_path, cache_dir)

        global _path_cache_index
        if _path_cache_index is not None:
            _path_cache_index[os.path.abspath(file_path)] = key

        return True
    except Exception as e:
        logger.warning(f"Failed to save session cache for {file_path}: {e}")
        return False


def get_image_dimensions(file_path: str) -> tuple[int, int]:
    """Fast, lightweight dimension extraction for RAW and standard image files in sub-2ms."""
    if not file_path or not os.path.isfile(file_path):
        return 0, 0
    ext = os.path.splitext(file_path)[1].lower()
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
            with rawpy.imread(file_path) as raw:
                rw, rh = raw.sizes.width, raw.sizes.height
                flip = getattr(raw.sizes, "flip", 0)
                if flip in (4, 5, 6, 7):
                    rw, rh = rh, rw
                if rw > 0 and rh > 0:
                    return int(rw), int(rh)
        except Exception:
            pass
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            return int(img.width), int(img.height)
    except Exception:
        pass
    return 0, 0


def extract_fast_thumbnail(file_path: str, target_h: int = 72) -> np.ndarray | None:
    """Extracts high-quality camera embedded JPEG thumbnail or loads image thumbnail in sub-15ms.
    Guarantees correct orientation and avoids expensive LibRaw demosaicing.
    """
    if not file_path or not os.path.isfile(file_path):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    try:
        import cv2
        if ext in RAW_EXTENSIONS:
            import rawpy
            with rawpy.imread(file_path) as raw:
                rw, rh = raw.sizes.width, raw.sizes.height
                flip = getattr(raw.sizes, "flip", 0)
                if flip in (4, 5, 6, 7):
                    w, h = rh, rw
                else:
                    w, h = rw, rh
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    file_bytes = np.frombuffer(thumb.data, dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        th_h_raw, th_w_raw = img_bgr.shape[:2]
                        if (h > w and th_w_raw > th_h_raw):
                            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                        elif (w > h and th_h_raw > th_w_raw):
                            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        th_h = target_h
                        th_w = max(48, int(rgb.shape[1] * (th_h / float(rgb.shape[0]))))
                        return cv2.resize(rgb, (th_w, th_h), interpolation=cv2.INTER_AREA)
        else:
            with open(file_path, "rb") as fp:
                file_bytes = np.frombuffer(fp.read(), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                th_h = target_h
                th_w = max(48, int(rgb.shape[1] * (th_h / float(rgb.shape[0]))))
                return cv2.resize(rgb, (th_w, th_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.debug(f"extract_fast_thumbnail failed for {file_path}: {e}")
    return None


def get_photo_cache(file_path: str, load_preview: bool = False) -> dict | None:
    """Loads cached photo metadata, thumbnail, and optionally the viewport preview array."""
    entry_path, key = _get_entry_path_with_fallback(file_path)
    if not entry_path or not os.path.isfile(entry_path):
        return None

    try:
        try:
            os.utime(entry_path, None)
        except Exception:
            pass
        with np.load(entry_path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            thumb = np.copy(npz["thumb"])
            raw_thumb = np.copy(npz["raw_thumb"]) if "raw_thumb" in npz else np.copy(thumb)
            graded_thumb = np.copy(npz["graded_thumb"]) if ("graded_thumb" in npz and npz["graded_thumb"].size > 0) else None

        # Auto-heal inverted dimensions from old cache entries
        w_cached = int(meta.get("width", 0))
        h_cached = int(meta.get("height", 0))
        if (w_cached <= 0 or h_cached <= 0) and file_path and os.path.isfile(file_path):
            dw, dh = get_image_dimensions(file_path)
            if dw > 0 and dh > 0:
                w_cached, h_cached = dw, dh
                meta["width"] = w_cached
                meta["height"] = h_cached

        ref_th = raw_thumb if (raw_thumb is not None and raw_thumb.size > 0) else thumb
        if ref_th is not None and hasattr(ref_th, "shape") and len(ref_th.shape) >= 2:
            th_h, th_w = ref_th.shape[:2]
            if (th_h > th_w and w_cached > h_cached) or (th_w > th_h and h_cached > w_cached):
                w_cached, h_cached = h_cached, w_cached
                meta["width"] = w_cached
                meta["height"] = h_cached

        float_img = None
        if load_preview:
            float_img = get_photo_preview(file_path)

        # Healing 1: Auto-heal dummy solid placeholder arrays or corrupted frames
        if is_dummy_or_corrupted_thumbnail(raw_thumb) or is_dummy_or_corrupted_thumbnail(thumb):
            healed = extract_fast_thumbnail(file_path)
            if healed is not None and not is_dummy_or_corrupted_thumbnail(healed):
                raw_thumb = healed
                thumb = healed
                if is_dummy_or_corrupted_thumbnail(graded_thumb):
                    graded_thumb = None
                # Persist healed thumbnail back to disk cache
                try:
                    save_photo_cache(file_path, {
                        "width": meta.get("width", 0),
                        "height": meta.get("height", 0),
                        "thumbnail_rgb": thumb,
                        "raw_thumbnail_rgb": raw_thumb,
                        "graded_thumbnail_rgb": graded_thumb,
                        "exif": meta.get("exif", {}),
                        "auto_ev": meta.get("auto_ev", 0.0),
                        "base_ev": meta.get("base_ev", 0.0),
                        "is_raw": meta.get("is_raw", False)
                    }, save_preview=False)
                except Exception:
                    pass

        # Healing 2: If raw_thumb is missing, empty or dummy, recover from float preview
        if is_dummy_or_corrupted_thumbnail(raw_thumb):
            preview_src = float_img if float_img is not None else get_photo_preview(file_path)
            if preview_src is not None and preview_src.size > 0:
                try:
                    import cv2
                    from .app_core import prophoto_to_srgb_filmic
                    th_h = 72
                    th_w = max(48, int(preview_src.shape[1] * (th_h / float(preview_src.shape[0]))))
                    th_img = cv2.resize(preview_src, (th_w, th_h), interpolation=cv2.INTER_AREA)
                    raw_thumb = (prophoto_to_srgb_filmic(th_img) * 255.0).astype(np.uint8)
                    if is_dummy_or_corrupted_thumbnail(thumb):
                        thumb = raw_thumb
                except Exception:
                    pass

        effective_thumb = graded_thumb if (graded_thumb is not None and graded_thumb.size > 0 and not is_dummy_or_corrupted_thumbnail(graded_thumb)) else raw_thumb
        if is_dummy_or_corrupted_thumbnail(effective_thumb) and not is_dummy_or_corrupted_thumbnail(thumb):
            effective_thumb = thumb

        return {
            "path": meta.get("path", file_path),
            "filename": meta.get("filename", os.path.basename(file_path)),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "thumbnail_rgb": effective_thumb,
            "raw_thumbnail_rgb": raw_thumb,
            "raw_thumbnail": raw_thumb,
            "graded_thumbnail_rgb": graded_thumb,
            "float_img": float_img,
            "exif": meta.get("exif", {}),
            "auto_ev": float(meta.get("auto_ev", 0.0)),
            "base_ev": float(meta.get("base_ev", 0.0)),
            "is_raw": bool(meta.get("is_raw", False))
        }
    except Exception as e:
        logger.warning(f"Failed to load session cache for {file_path}: {e}")
        return None


def get_photo_preview(file_path: str) -> np.ndarray | None:
    """Loads only the viewport float preview array (float32) from cache."""
    if not file_path:
        return None
    key = get_cache_key(file_path)
    cache_dir = get_cache_dir()
    preview_path = os.path.join(cache_dir, f"{key}_preview.npy")

    if not os.path.isfile(preview_path):
        norm_path = os.path.abspath(file_path)
        _migrate_preview_from_index(norm_path, preview_path, cache_dir)

    if not os.path.isfile(preview_path):
        return None

    try:
        arr16 = np.load(preview_path, allow_pickle=False)
        try:
            os.utime(preview_path, None)
        except Exception:
            pass
        return arr16.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load preview cache for {file_path}: {e}")
        return None


def save_photo_preview(file_path: str, preview_array: np.ndarray) -> bool:
    """Saves only the viewport float preview array (float16 on disk) into cache."""
    if not file_path or preview_array is None or not isinstance(preview_array, np.ndarray) or preview_array.size == 0:
        return False
    key = get_cache_key(file_path)
    if not key:
        return False
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    preview_path = os.path.join(cache_dir, f"{key}_preview.npy")
    import threading
    nonce = f"{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
    tmp_prev = os.path.join(cache_dir, f"{key}_preview_tmp_{nonce}.npy")
    try:
        np.save(tmp_prev, preview_array.astype(np.float16))
        if os.path.exists(preview_path):
            try:
                os.remove(preview_path)
            except Exception:
                pass
        os.replace(tmp_prev, preview_path)
        global _path_cache_index
        if _path_cache_index is not None:
            _path_cache_index[os.path.abspath(file_path)] = key
        return True
    except Exception as e:
        logger.warning(f"Failed to save preview cache for {file_path}: {e}")
        if os.path.exists(tmp_prev):
            try:
                os.remove(tmp_prev)
            except Exception:
                pass
        return False


def invalidate_photo_cache(file_path: str):
    """Invalidates and purges memory fallback index for a given image file upon SDC removal."""
    if not file_path:
        return
    norm_path = os.path.abspath(file_path)
    global _path_cache_index
    if _path_cache_index is not None and norm_path in _path_cache_index:
        _path_cache_index.pop(norm_path, None)


def get_cache_size_bytes() -> int:
    """Calculates total disk usage of the session cache."""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return 0
    total = 0
    try:
        for f in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def get_cache_size_mb() -> float:
    """Returns cache size in Megabytes."""
    return get_cache_size_bytes() / (1024.0 * 1024.0)


def clear_cache() -> bool:
    """Removes all cached entries and previews."""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return True
    try:
        for f in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, f)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass
        return True
    except Exception as e:
        logger.warning(f"Error clearing session cache: {e}")
        return False


def prune_cache_by_age(max_days: float = 7.0) -> int:
    """Removes cache files older than max_days. Returns count of deleted files."""
    if max_days <= 0:
        return 0
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return 0
    cutoff = time.time() - (float(max_days) * 86400.0)
    deleted = 0
    try:
        for f in os.listdir(cache_dir):
            if f.endswith("_entry.npz") or f.endswith("_preview.npy"):
                fp = os.path.join(cache_dir, f)
                try:
                    if os.path.isfile(fp) and os.stat(fp).st_mtime < cutoff:
                        os.remove(fp)
                        deleted += 1
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error during prune_cache_by_age: {e}")
    return deleted


def prune_cache_by_size(max_size_bytes: int) -> int:
    """Prunes oldest cache files (LRU) if cache exceeds max_size_bytes.
    Tiered strategy: Prioritizes deleting heavy .npy viewport preview textures (~35MB)
    first, while fiercely preserving lightweight .npz filmstrip thumbnails (~14KB).
    """
    if max_size_bytes <= 0:
        return 0
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return 0

    deleted = 0
    try:
        preview_files = []
        entry_files = []
        total_size = 0

        for f in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, f)
            if os.path.isfile(fp):
                try:
                    st = os.stat(fp)
                    sz = st.st_size
                    total_size += sz
                    if f.endswith(".npy"):
                        preview_files.append((fp, st.st_mtime, sz))
                    elif f.endswith(".npz"):
                        entry_files.append((fp, st.st_mtime, sz))
                    else:
                        preview_files.append((fp, st.st_mtime, sz))
                except OSError:
                    pass

        if total_size <= max_size_bytes:
            return 0

        target_size = int(max_size_bytes * 0.85)

        # 1. First Tier: Oldest preview textures (.npy) first
        preview_files.sort(key=lambda x: x[1])
        for fp, _, sz in preview_files:
            try:
                os.remove(fp)
                total_size -= sz
                deleted += 1
                if total_size <= target_size:
                    return deleted
            except Exception:
                pass

        # 2. Second Tier: If still exceeding (rare), prune oldest entries (.npz)
        entry_files.sort(key=lambda x: x[1])
        for fp, _, sz in entry_files:
            try:
                os.remove(fp)
                total_size -= sz
                deleted += 1
                if total_size <= target_size:
                    return deleted
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Error during prune_cache_by_size: {e}")
    return deleted


def prune_cache_if_needed(max_size_bytes: int = None):
    """Wrapper for backward compatibility and auto-pruning."""
    if max_size_bytes is None:
        try:
            from . import config_manager
            prefs = config_manager.get_preferences()
            max_gb = float(prefs.get("session_cache_max_gb", 2.0))
            preview_res = int(prefs.get("preview_max_edge", 2048))
            if preview_res == 0 and max_gb < 8.0:
                max_gb = max(8.0, max_gb)
            if max_gb <= 0:
                return  # No limit
            max_size_bytes = int(max_gb * 1024 * 1024 * 1024)
        except Exception:
            max_size_bytes = 2 * 1024 * 1024 * 1024
    prune_cache_by_size(max_size_bytes)


def check_and_auto_clean():
    """Performs scheduled cache cleanup based on user preferences."""
    try:
        from . import config_manager
        prefs = config_manager.get_preferences()
        interval_days = int(prefs.get("session_cache_clean_interval_days", 7))
        last_clean = float(prefs.get("session_cache_last_clean_time", 0.0))
        now = time.time()

        # 1. Clean by age if scheduled interval reached
        if interval_days > 0:
            if now - last_clean >= interval_days * 86400.0:
                prune_cache_by_age(interval_days)
                config_manager.set_preference("session_cache_last_clean_time", now)

        # 2. Check total size limit
        max_gb = float(prefs.get("session_cache_max_gb", 2.0))
        preview_res = int(prefs.get("preview_max_edge", 2048))
        if preview_res == 0 and max_gb < 8.0:
            max_gb = max(8.0, max_gb)
        if max_gb > 0:
            max_bytes = int(max_gb * 1024 * 1024 * 1024)
            prune_cache_by_size(max_bytes)
    except Exception as e:
        logger.warning(f"Error in check_and_auto_clean: {e}")


def save_session_file(file_path: str, session_data: dict) -> bool:
    """Saves session metadata, loaded photos, and edit states to a .sdss file.
    v2.0 Architecture: Strips redundant per-photo params to ensure .sdc remains
    the Single Source of Truth for photo development profiles.
    """
    try:
        dir_name = os.path.dirname(os.path.abspath(file_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        clean_photos = []
        raw_photos = session_data.get("photos", [])
        for item in raw_photos:
            if isinstance(item, str):
                clean_photos.append({
                    "path": os.path.abspath(item),
                    "filename": os.path.basename(item)
                })
            elif isinstance(item, dict) and "path" in item:
                entry = {
                    "path": os.path.abspath(item["path"]),
                    "filename": item.get("filename") or os.path.basename(item["path"])
                }
                if "rel_path" in item and item["rel_path"]:
                    entry["rel_path"] = item["rel_path"]
                clean_photos.append(entry)

        clean_data = {
            "format": "SpektraDarkroomSession",
            "version": session_data.get("version", "2.1"),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "active_photo_path": session_data.get("active_photo_path"),
            "active_photo_rel_path": session_data.get("active_photo_rel_path"),
            "active_photo_filename": session_data.get("active_photo_filename"),
            "active_photo_id": session_data.get("active_photo_id"),
            "view_mode": session_data.get("view_mode", 0),
            "split_x": session_data.get("split_x", 0.5),
            "photos": clean_photos
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save session file to {file_path}: {e}")
        return False


def load_session_file(file_path: str) -> dict | None:
    """Loads session data from a .sdss file.
    Seamless backward compatibility:
    - Supports v2.0/v2.1 clean path and relative path array format;
    - If legacy v1.0 file with embedded params is detected, auto-migrates missing .sdc sidecars.
    """
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        # Normalize photos array
        raw_photos = data.get("photos", [])
        norm_photos = []
        from . import sdc_manager

        for item in raw_photos:
            if isinstance(item, str):
                norm_photos.append({
                    "path": os.path.abspath(item),
                    "filename": os.path.basename(item)
                })
            elif isinstance(item, dict) and "path" in item:
                photo_path = os.path.abspath(item["path"])
                # Migration: If session has recipe params and .sdc does not exist, save sidecar
                params = item.get("params")
                film = item.get("film_profile") or item.get("film_stock", "none")
                paper = item.get("paper_profile") or item.get("paper_stock", "none")
                pmode = item.get("print_mode", "optical")
                base_ev = item.get("base_ev", 0.0)
                if (params or film != "none" or paper != "none") and not sdc_manager.has_sdc(photo_path):
                    try:
                        sdc_manager.save_sdc(photo_path, film, paper, params or {}, print_mode=pmode)
                    except Exception:
                        pass
                entry = {
                    "path": photo_path,
                    "filename": item.get("filename") or os.path.basename(photo_path)
                }
                if "rel_path" in item and item["rel_path"]:
                    entry["rel_path"] = item["rel_path"]
                norm_photos.append(entry)

        data["photos"] = norm_photos
        return data
    except Exception as e:
        logger.error(f"Failed to load session file from {file_path}: {e}")
    return None

