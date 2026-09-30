"""Config and Preference Manager for SpektraDarkroom.
Manages persistent user preferences, window layout/geometry, UI state, and recent files.
Supports both standard user directory (%APPDATA%/SpektraDarkroom) and portable mode (local config.json).
"""

import os
import sys
import json
import copy
import logging

logger = logging.getLogger("SpektraDarkroom.Config")

DEFAULT_CONFIG = {
    "version": 1,
    "window": {
        "width": 1440,
        "height": 900,
        "x": None,
        "y": None,
        "is_maximized": False,
        "splitter_main": [320, 800, 320],
        "splitter_left": [300, 180, 200],
        "splitter_center": [600, 96],
    },
    "ui_state": {
        "sections_expanded": {
            "exposure": True,
            "enlarger": True,
            "optics": True,
        },
        "last_film_stock": "kodak_portra_400",
        "last_paper_stock": "kodak_2383",
        "last_active_tab": 0,
    },
    "preferences": {
        "gpu_acceleration": True,
        "theme": "adobe_dark",
        "default_export_dir": "",
        "export_format": "jpeg",
        "export_quality": 95,
        "export_dpi": 300,
        "auto_save_sdc": True,
        "restore_last_files": True,
        "hardware_acceleration_mode": "global",
        "preview_max_edge": 2048,
        "color_space": "sRGB",
        "preview_quality": "full",
    },
    "recent_files": [],
    "last_session": {
        "files": [],
        "active_file": None,
    },
}

MAX_RECENT_FILES = 16
_cached_config = None


def get_app_dir() -> str:
    """Returns application root directory."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_config_path() -> str:
    """Returns the config file path, prioritizing the program root directory.
    Falls back to user AppData directory only if the program root is read-only.
    """
    app_dir = get_app_dir()
    local_config = os.path.join(app_dir, "config.json")

    # Check if local config already exists or if app_dir is writable
    try:
        if os.path.exists(local_config):
            return local_config
        # Test write permission in app_dir
        test_file = os.path.join(app_dir, ".perm_test")
        with open(test_file, "w") as f:
            f.write("1")
        if os.path.exists(test_file):
            os.remove(test_file)
        return local_config
    except Exception:
        # Fallback to User AppData if app_dir is read-only (e.g. Program Files)
        if sys.platform == "win32":
            app_data = os.getenv("APPDATA")
            if app_data:
                user_config_dir = os.path.join(app_data, "SpektraDarkroom")
            else:
                user_config_dir = os.path.join(os.path.expanduser("~"), ".spektra_darkroom")
        else:
            xdg_config = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
            user_config_dir = os.path.join(xdg_config, "spektra_darkroom")

        try:
            os.makedirs(user_config_dir, exist_ok=True)
        except Exception:
            pass

        return os.path.join(user_config_dir, "config.json")


def _deep_merge(target: dict, source: dict) -> dict:
    """Recursively merges source into a deep copy of target."""
    result = copy.deepcopy(target)
    for k, v in source.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config() -> dict:
    """Loads configuration with fallback defaults and legacy migration."""
    global _cached_config
    config_file = get_config_path()

    loaded_data = None
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading config from {config_file}: {e}")

    # Check for legacy config in home directory if new config doesn't exist
    if loaded_data is None:
        legacy_path = os.path.join(os.path.expanduser("~"), ".spektra_darkroom_config.json")
        if os.path.isfile(legacy_path):
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    legacy_data = json.load(f)
                loaded_data = legacy_data
                logger.info(f"Migrated legacy config from {legacy_path}")
            except Exception as e:
                logger.warning(f"Error reading legacy config: {e}")

    if loaded_data and isinstance(loaded_data, dict):
        merged = _deep_merge(DEFAULT_CONFIG, loaded_data)
    else:
        merged = copy.deepcopy(DEFAULT_CONFIG)

    _cached_config = merged
    # If the file does not exist on disk yet, save it immediately so it is visible
    if not os.path.isfile(config_file):
        save_config(merged)

    return _cached_config


def save_config(cfg: dict = None):
    """Atomically saves configuration to the determined path."""
    global _cached_config
    if cfg is not None:
        _cached_config = cfg
    elif _cached_config is None:
        _cached_config = copy.deepcopy(DEFAULT_CONFIG)

    config_file = get_config_path()
    try:
        parent_dir = os.path.dirname(config_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        temp_file = config_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(_cached_config, f, indent=2, ensure_ascii=False)

        # Atomic replacement on Windows / POSIX
        if os.path.exists(config_file):
            try:
                os.replace(temp_file, config_file)
            except OSError:
                os.remove(config_file)
                os.rename(temp_file, config_file)
        else:
            os.rename(temp_file, config_file)
    except Exception as e:
        logger.warning(f"Error saving config to {config_file}: {e}")


# =========================================================================
# Window Layout & Geometry APIs
# =========================================================================

def get_window_config() -> dict:
    cfg = load_config()
    return cfg.get("window", DEFAULT_CONFIG["window"])


def save_window_config(
    width: int,
    height: int,
    x: int = None,
    y: int = None,
    is_maximized: bool = False,
    splitter_main: list = None,
    splitter_left: list = None,
    splitter_center: list = None
):
    cfg = load_config()
    win = cfg.setdefault("window", {})
    win["width"] = int(width)
    win["height"] = int(height)
    if x is not None and y is not None:
        win["x"] = int(x)
        win["y"] = int(y)
    win["is_maximized"] = bool(is_maximized)
    if splitter_main:
        win["splitter_main"] = [int(s) for s in splitter_main]
    if splitter_left:
        win["splitter_left"] = [int(s) for s in splitter_left]
    if splitter_center:
        win["splitter_center"] = [int(s) for s in splitter_center]
    save_config(cfg)


# =========================================================================
# UI State APIs
# =========================================================================

def get_ui_state() -> dict:
    cfg = load_config()
    return cfg.get("ui_state", DEFAULT_CONFIG["ui_state"])


def save_ui_state(
    sections_expanded: dict = None,
    last_film_stock: str = None,
    last_paper_stock: str = None,
    last_active_tab: int = None
):
    cfg = load_config()
    ui_st = cfg.setdefault("ui_state", {})
    if sections_expanded is not None:
        ui_st["sections_expanded"] = sections_expanded
    if last_film_stock:
        ui_st["last_film_stock"] = last_film_stock
    if last_paper_stock:
        ui_st["last_paper_stock"] = last_paper_stock
    if last_active_tab is not None:
        ui_st["last_active_tab"] = int(last_active_tab)
    save_config(cfg)


# =========================================================================
# General Preferences APIs (for Future Settings Page)
# =========================================================================

def get_preference(key: str, default=None):
    cfg = load_config()
    return cfg.get("preferences", {}).get(key, default)


def get_preferences() -> dict:
    cfg = load_config()
    return cfg.get("preferences", DEFAULT_CONFIG["preferences"])


def set_preference(key: str, value):
    cfg = load_config()
    prefs = cfg.setdefault("preferences", {})
    prefs[key] = value
    save_config(cfg)


def set_preferences(prefs_dict: dict):
    cfg = load_config()
    prefs = cfg.setdefault("preferences", {})
    prefs.update(prefs_dict)
    save_config(cfg)


# =========================================================================
# Recent Files APIs
# =========================================================================

def get_recent_files() -> list:
    cfg = load_config()
    recent = cfg.get("recent_files", [])
    return [p for p in recent if os.path.exists(p)]


def add_recent_file(path: str):
    if not path or not os.path.exists(path):
        return
    norm_path = os.path.abspath(path)
    cfg = load_config()
    recent = cfg.get("recent_files", [])
    if norm_path in recent:
        recent.remove(norm_path)
    recent.insert(0, norm_path)
    cfg["recent_files"] = recent[:MAX_RECENT_FILES]
    save_config(cfg)


def clear_recent_files():
    cfg = load_config()
    cfg["recent_files"] = []
    save_config(cfg)


# =========================================================================
# Session State APIs (Restore last opened files on startup)
# =========================================================================

def save_session_state(file_paths: list, active_file: str = None):
    cfg = load_config()
    norm_paths = [os.path.abspath(p) for p in file_paths if p and os.path.exists(p)]
    norm_active = os.path.abspath(active_file) if active_file and os.path.exists(active_file) else None
    cfg["last_session"] = {
        "files": norm_paths,
        "active_file": norm_active
    }
    save_config(cfg)


def get_session_state() -> tuple:
    """Returns (files_list, active_file)."""
    cfg = load_config()
    sess = cfg.get("last_session", {})
    files = [p for p in sess.get("files", []) if os.path.exists(p)]
    active = sess.get("active_file")
    if active and not os.path.exists(active):
        active = None
    return files, active

