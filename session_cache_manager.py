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
    max_edge = 2048
    try:
        import config_manager
        max_edge = int(config_manager.get_preferences().get("preview_max_edge", 2048))
    except Exception:
        max_edge = 2048
    token = f"prophoto_v3|{norm_path}|{mtime}|{size}|res_{max_edge}".encode("utf-8")
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
            "auto_ev": float(photo_data.get("auto_ev", 0.0)),
            "base_ev": float(photo_data.get("base_ev", 0.0)),
            "is_raw": bool(photo_data.get("is_raw", False))
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
        import threading
        nonce = f"{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
        tmp_entry = os.path.join(cache_dir, f"{key}_entry_tmp_{nonce}.npz")
        np.savez(tmp_entry, thumb=thumb, meta=meta_json)
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

        return True
    except Exception as e:
        logger.warning(f"Failed to save session cache for {file_path}: {e}")
        return False


def save_photo_preview(file_path: str, preview_array: np.ndarray) -> bool:
    """Saves only the viewport preview array in float16 format."""
    if not file_path or not os.path.isfile(file_path) or preview_array is None:
        return False
    try:
        import threading
        nonce = f"{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
        cache_dir = get_cache_dir()
        key = get_cache_key(file_path)
        preview_path = os.path.join(cache_dir, f"{key}_preview.npy")
        tmp_prev = os.path.join(cache_dir, f"{key}_preview_tmp_{nonce}.npy")
        np.save(tmp_prev, preview_array.astype(np.float16))
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
            "auto_ev": float(meta.get("auto_ev", 0.0)),
            "base_ev": float(meta.get("base_ev", 0.0)),
            "is_raw": bool(meta.get("is_raw", False))
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
            import config_manager
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
        import config_manager
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
                clean_photos.append(os.path.abspath(item))
            elif isinstance(item, dict) and "path" in item:
                clean_photos.append(os.path.abspath(item["path"]))

        clean_data = {
            "format": "SpektraDarkroomSession",
            "version": "2.0",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "active_photo_path": session_data.get("active_photo_path"),
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
    - Supports v2.0 clean path array format;
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
        import sdc_manager

        for item in raw_photos:
            if isinstance(item, str):
                norm_photos.append({"path": os.path.abspath(item)})
            elif isinstance(item, dict) and "path" in item:
                photo_path = os.path.abspath(item["path"])
                # Legacy migration: If old session has params and .sdc does not exist, save sidecar
                params = item.get("params")
                film = item.get("film_stock", "none")
                paper = item.get("paper_stock", "none")
                if (params or film != "none" or paper != "none") and not sdc_manager.has_sdc(photo_path):
                    try:
                        sdc_manager.save_sdc(photo_path, film, paper, params or {})
                    except Exception:
                        pass
                norm_photos.append({"path": photo_path})

        data["photos"] = norm_photos
        return data
    except Exception as e:
        logger.error(f"Failed to load session file from {file_path}: {e}")
    return None

