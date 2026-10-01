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
import json
import hashlib
import logging
import shutil
import numpy as np

logger = logging.getLogger("SpektraDarkroom.SessionCache")


def get_cache_dir() -> str:
    """Returns path to the session cache folder, creating it if needed."""
    try:
        import config_manager
        app_dir = config_manager.get_app_dir()
        local_cache = os.path.join(app_dir, ".session_cache")
        os.makedirs(local_cache, exist_ok=True)
        # Test writeability
        test_p = os.path.join(local_cache, ".perm_test")
        with open(test_p, "w") as f:
            f.write("1")
        if os.path.exists(test_p):
            os.remove(test_p)
        return local_cache
    except Exception:
        pass

    # Fallback to User AppData / Cache directory
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~")
        cache_dir = os.path.join(base, "SpektraDarkroom", "session_cache")
    else:
        base = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base, "spektra_darkroom", "session_cache")

    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    return cache_dir


def get_cache_key(file_path: str) -> str:
    """Generates a stable cache key based on absolute path, file modification time, and size."""
    norm_path = os.path.abspath(file_path)
    try:
        st = os.stat(norm_path)
        mtime = st.st_mtime
        size = st.st_size
    except OSError:
        mtime = 0.0
        size = 0
    token = f"{norm_path}|{mtime}|{size}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:24]


def has_valid_cache(file_path: str) -> bool:
    """Checks whether a valid, non-stale cache entry exists for the given file."""
    if not file_path or not os.path.isfile(file_path):
        return False
    key = get_cache_key(file_path)
    entry_path = os.path.join(get_cache_dir(), f"{key}_entry.npz")
    return os.path.isfile(entry_path)


def save_photo_cache(file_path: str, photo_data: dict, preview_array: np.ndarray = None, save_preview: bool = True) -> bool:
    """Saves photo metadata, thumbnail, and optional float preview to disk cache incrementally."""
    if not file_path or not os.path.isfile(file_path):
        return False

    try:
        cache_dir = get_cache_dir()
        key = get_cache_key(file_path)
        entry_path = os.path.join(cache_dir, f"{key}_entry.npz")
        preview_path = os.path.join(cache_dir, f"{key}_preview.npy")

        # 1. Prepare metadata
        meta = {
            "path": os.path.abspath(file_path),
            "filename": os.path.basename(file_path),
            "width": int(photo_data.get("width", 0)),
            "height": int(photo_data.get("height", 0)),
            "exif": photo_data.get("exif", {}),
            "auto_ev": float(photo_data.get("auto_ev", 0.0))
        }
        meta_json = json.dumps(meta, default=str, ensure_ascii=False)

        # 2. Thumbnail
        thumb = photo_data.get("thumbnail_rgb")
        if thumb is None:
            thumb = np.zeros((72, 72, 3), dtype=np.uint8)
        elif not isinstance(thumb, np.ndarray):
            thumb = np.array(thumb, dtype=np.uint8)
        elif thumb.dtype != np.uint8:
            if np.issubdtype(thumb.dtype, np.floating):
                thumb = (np.clip(thumb, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                thumb = thumb.astype(np.uint8)

        # Atomic write entry
        pid = os.getpid()
        tmp_entry = os.path.join(cache_dir, f"{key}_entry_tmp_{pid}.npz")
        np.savez(tmp_entry, thumb=thumb, meta=meta_json)
        if os.path.exists(entry_path):
            try:
                os.remove(entry_path)
            except Exception:
                pass
        os.replace(tmp_entry, entry_path)

        # 3. Save preview if enabled and available
        if save_preview:
            preview = preview_array if preview_array is not None else photo_data.get("float_img")
            if preview is not None and isinstance(preview, np.ndarray):
                tmp_prev = os.path.join(cache_dir, f"{key}_preview_tmp_{pid}.npy")
                np.save(tmp_prev, preview.astype(np.float16))
                if os.path.exists(preview_path):
                    try:
                        os.remove(preview_path)
                    except Exception:
                        pass
                os.replace(tmp_prev, preview_path)

        return True
    except Exception as e:
        logger.warning(f"Failed to save session cache for {file_path}: {e}")
        return False


def save_photo_preview(file_path: str, preview_array: np.ndarray) -> bool:
    """Saves only the viewport preview array in float16 format."""
    if not file_path or not os.path.isfile(file_path) or preview_array is None:
        return False
    try:
        cache_dir = get_cache_dir()
        key = get_cache_key(file_path)
        preview_path = os.path.join(cache_dir, f"{key}_preview.npy")
        pid = os.getpid()
        tmp_prev = os.path.join(cache_dir, f"{key}_preview_tmp_{pid}.npy")
        np.save(tmp_prev, preview_array.astype(np.float16))
        if os.path.exists(preview_path):
            try:
                os.remove(preview_path)
            except Exception:
                pass
        os.replace(tmp_prev, preview_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to save preview cache for {file_path}: {e}")
        return False


def get_photo_cache(file_path: str, load_preview: bool = False) -> dict | None:
    """Loads cached photo metadata, thumbnail, and optionally the viewport preview array."""
    if not file_path or not os.path.isfile(file_path):
        return None

    key = get_cache_key(file_path)
    entry_path = os.path.join(get_cache_dir(), f"{key}_entry.npz")
    if not os.path.isfile(entry_path):
        return None

    try:
        with np.load(entry_path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            thumb = np.copy(npz["thumb"])

        float_img = None
        if load_preview:
            float_img = get_photo_preview(file_path)

        return {
            "path": meta.get("path", file_path),
            "filename": meta.get("filename", os.path.basename(file_path)),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "thumbnail_rgb": thumb,
            "float_img": float_img,
            "exif": meta.get("exif", {}),
            "auto_ev": meta.get("auto_ev", 0.0)
        }
    except Exception as e:
        logger.warning(f"Failed to load session cache for {file_path}: {e}")
        return None


def get_photo_preview(file_path: str) -> np.ndarray | None:
    """Loads only the viewport float preview array (float32) from cache."""
    if not file_path or not os.path.isfile(file_path):
        return None

    key = get_cache_key(file_path)
    preview_path = os.path.join(get_cache_dir(), f"{key}_preview.npy")
    if not os.path.isfile(preview_path):
        return None

    try:
        arr16 = np.load(preview_path, allow_pickle=False)
        return arr16.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load preview cache for {file_path}: {e}")
        return None


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


def prune_cache_if_needed(max_size_bytes: int = 2 * 1024 * 1024 * 1024):
    """Prunes oldest cache files (LRU) if cache directory exceeds max_size_bytes."""
    try:
        cache_dir = get_cache_dir()
        if not os.path.exists(cache_dir):
            return

        files = []
        total_size = 0
        for f in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, f)
            if os.path.isfile(fp):
                try:
                    st = os.stat(fp)
                    total_size += st.st_size
                    files.append((fp, st.st_mtime, st.st_size))
                except OSError:
                    pass

        if total_size <= max_size_bytes:
            return

        # Sort by mtime ascending (oldest first)
        files.sort(key=lambda x: x[1])
        target_size = int(max_size_bytes * 0.75)
        for fp, _, sz in files:
            try:
                os.remove(fp)
                total_size -= sz
                if total_size <= target_size:
                    break
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Error during cache pruning: {e}")
