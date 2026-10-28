"""Config and Preference Manager for SpektraDarkroom.
Manages persistent user preferences, window layout/geometry, UI state, and recent files.
Supports both standard user directory (%APPDATA%/SpektraDarkroom) and portable mode (local config.json).
"""

import os
import sys
import json
import copy
import ctypes
import logging
import time

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
        "last_film_stock": "none",
        "last_paper_stock": "none",
        "last_active_tab": 0,
    },
    "preferences": {
        "gpu_acceleration": True,
        "theme": "dark",
        "ui_theme": "system",
        "canvas_background": "dark_gray",
        "default_export_dir": "",
        "export_format": "jpeg",
        "export_quality": 95,
        "export_dpi": 300,
        "auto_save_sdc": True,
        "restore_last_session": True,
        "restore_last_files": True,
        "hardware_acceleration_mode": "global",
        "preview_max_edge": 2048,
        "color_space": "sRGB",
        "preview_quality": "full",
        "auto_meter_on_import": False,
        "session_cache_max_gb": 2.0,
        "session_cache_clean_interval_days": 7,
        "session_cache_last_clean_time": 0.0,
        "custom_cache_dir": "",
    },
    "recent_files": [],
    "recent_sessions": [],
    "last_session_path": None,
    "last_session": {
        "files": [],
        "active_file": None,
    },
}

MAX_RECENT_FILES = 16
_cached_config = None
_cached_config_stamp = None


def get_app_dir() -> str:
    """Returns application root directory."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    core_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(core_dir)


def get_config_path() -> str:
    """Returns the config file path.
    Prioritizes the standard user AppData directory (%APPDATA%/SpektraDarkroom/config.json).
    Supports portable mode if portable.txt or local config marker exists in app_dir.
    """
    from .path_utils import get_user_data_dir
    app_dir = get_app_dir()
    local_config = os.path.join(app_dir, "config.json")
    portable_marker = os.path.join(app_dir, "portable.txt")

    # If portable.txt explicitly exists, stay in portable mode
    if os.path.exists(portable_marker):
        return local_config

    user_dir = get_user_data_dir()
    user_config = os.path.join(user_dir, "config.json")

    # Seamless migration: If user config does not exist yet but local config exists, copy it over
    if not os.path.exists(user_config) and os.path.exists(local_config):
        try:
            import shutil
            shutil.copy2(local_config, user_config)
        except Exception:
            pass

    if os.path.exists(user_config) or not os.path.exists(local_config):
        return user_config
    return local_config


_needs_repair = False


def _deep_merge(target: dict, source: dict) -> dict:
    """Recursively merges source into a deep copy of target.

    Type mismatches are treated as corruption, not as an override: a ``null`` or a string
    where the defaults expect a dict is dropped and the config is flagged for rewrite,
    instead of poisoning every later ``cfg.get`` call.
    """
    global _needs_repair
    result = copy.deepcopy(target)
    for k, v in source.items():
        if k in result:
            expected = result[k]
            if isinstance(expected, dict):
                if isinstance(v, dict):
                    result[k] = _deep_merge(expected, v)
                else:
                    _needs_repair = True
                    logger.warning(f"Config key '{k}' should be an object, got {type(v).__name__}; using defaults.")
                continue
            if isinstance(expected, list) and not isinstance(v, list):
                _needs_repair = True
                logger.warning(f"Config key '{k}' should be a list, got {type(v).__name__}; using defaults.")
                continue
            if isinstance(expected, bool) and not isinstance(v, bool):
                _needs_repair = True
                logger.warning(f"Config key '{k}' should be a boolean, got {type(v).__name__}; using defaults.")
                continue
        result[k] = copy.deepcopy(v)
    return result


def _config_stamp(config_file: str):
    """Cheap identity for the on-disk config: (path, mtime, size)."""
    try:
        st = os.stat(config_file)
        return (config_file, st.st_mtime, st.st_size)
    except OSError:
        return (config_file, None, None)


def load_config() -> dict:
    """Loads configuration with fallback defaults and legacy migration."""
    global _cached_config, _cached_config_stamp
    config_file = get_config_path()

    # Every get_xxx() in this module goes through load_config(); without this guard the
    # file was re-read and the defaults deep-copied on every single accessor (1.2 ms each,
    # i.e. seconds across a large library import).
    stamp = _config_stamp(config_file)
    if _cached_config is not None and _cached_config_stamp == stamp:
        return _cached_config

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
        global _needs_repair
        _needs_repair = False
        merged = _deep_merge(DEFAULT_CONFIG, loaded_data)
    else:
        merged = copy.deepcopy(DEFAULT_CONFIG)

    _cached_config = merged
    # Write back when the file is missing, or when corrupt values were dropped above:
    # otherwise a single bad key keeps breaking startup forever.
    if not os.path.isfile(config_file) or _needs_repair:
        save_config(merged)
    _cached_config_stamp = _config_stamp(config_file)

    return _cached_config


def save_config(cfg: dict = None):
    """Atomically saves configuration to the determined path."""
    global _cached_config, _cached_config_stamp
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

        # Atomic replacement only. Never delete the live config first: on Windows the
        # target file is routinely locked for a few milliseconds by AV scanners or the
        # indexer, and "remove then rename" turns that temporary failure into data loss.
        last_err = None
        for attempt in range(4):
            try:
                os.replace(temp_file, config_file)
                last_err = None
                break
            except OSError as ex:
                last_err = ex
                time.sleep(0.05 * (attempt + 1))
        if last_err is not None:
            logger.warning(
                f"Config replace failed after retries ({last_err}); "
                f"kept existing config and temp copy at {temp_file}")
            # Disk still holds the old bytes: drop the stamp so the next read reflects it.
            _cached_config_stamp = None
        else:
            _cached_config_stamp = _config_stamp(config_file)
    except Exception as e:
        _cached_config_stamp = None
        logger.warning(f"Error saving config to {config_file}: {e}")


# =========================================================================
# Window Layout & Geometry APIs
# =========================================================================

def get_window_config() -> dict:
    cfg = load_config()
    win = cfg.get("window", DEFAULT_CONFIG["window"])
    # Defence in depth: _deep_merge already drops type-mismatched values, but a caller
    # reading a config loaded before the repair landed must still get a usable dict.
    return win if isinstance(win, dict) else copy.deepcopy(DEFAULT_CONFIG["window"])


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
    if last_film_stock is not None:
        ui_st["last_film_stock"] = last_film_stock
    if last_paper_stock is not None:
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
# Recent Sessions APIs
# =========================================================================

def get_recent_sessions() -> list:
    cfg = load_config()
    recent = cfg.get("recent_sessions", [])
    return [p for p in recent if os.path.exists(p)]


def add_recent_session(path: str):
    if not path or not os.path.exists(path):
        return
    norm_path = os.path.abspath(path)
    cfg = load_config()
    recent = cfg.get("recent_sessions", [])
    if norm_path in recent:
        recent.remove(norm_path)
    recent.insert(0, norm_path)
    cfg["recent_sessions"] = recent[:MAX_RECENT_FILES]
    save_config(cfg)


def clear_recent_sessions():
    cfg = load_config()
    cfg["recent_sessions"] = []
    save_config(cfg)


def get_auto_saved_session_path() -> str:
    """Returns the persistent path for auto-saving temporary unsaved darkroom sessions."""
    app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    sess_dir = os.path.join(app_data, "SpektraDarkroom")
    os.makedirs(sess_dir, exist_ok=True)
    return os.path.normpath(os.path.join(sess_dir, "auto_saved_session.sdss"))


def has_auto_saved_session() -> bool:
    """Returns True if a valid auto-saved temporary session exists on disk."""
    p = get_auto_saved_session_path()
    return os.path.exists(p) and os.path.getsize(p) > 20


def clear_auto_saved_session():
    """Removes the auto-saved temporary session file."""
    p = get_auto_saved_session_path()
    if os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass


def save_last_session_path(session_path: str = None):
    """Saves the last opened or saved .sdss session file path."""
    cfg = load_config()
    if session_path and os.path.exists(session_path):
        cfg["last_session_path"] = os.path.abspath(session_path)
    else:
        cfg["last_session_path"] = None
    save_config(cfg)


def get_last_session_path() -> str:
    """Returns the last opened or saved .sdss session file path if it exists on disk,
    or None if no session was active or record is empty.
    """
    cfg = load_config()
    p = cfg.get("last_session_path")
    if p and os.path.exists(p):
        return os.path.abspath(p)
    return None


def save_session_state(file_paths: list = None, active_file: str = None):
    cfg = load_config()
    if file_paths is None:
        file_paths = []
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


def get_last_export_settings() -> dict:
    """Returns last used export settings dictionary."""
    cfg = load_config()
    return cfg.get("preferences", {}).get("last_export_settings", {
        "engine": "wysiwyg",
        "format": "jpeg",
        "bit_depth": 8,
        "quality": 95,
        "scale_pct": 100,
        "out_long_edge": 0,
        "color_space": "sRGB",
        "sharpen": False,
        "sharpen_strength": 0.5,
        "output_dir": ""
    })


def save_last_export_settings(settings: dict):
    """Persists last export settings dictionary."""
    cfg = load_config()
    if "preferences" not in cfg:
        cfg["preferences"] = {}
    current = cfg["preferences"].get("last_export_settings", {})
    current.update(settings)
    cfg["preferences"]["last_export_settings"] = current
    save_config(cfg)


def get_last_export_dir() -> str:
    """Returns the last used export directory. Always returns a directory path or empty string."""
    cfg = load_config()
    pref = cfg.get("preferences", {})
    last_settings = pref.get("last_export_settings", {})
    candidates = [
        pref.get("last_export_dir"),
        last_settings.get("output_dir"),
        last_settings.get("path"),
        pref.get("default_export_dir")
    ]
    for c in candidates:
        if c and isinstance(c, str):
            c_clean = c.strip()
            if os.path.isdir(c_clean):
                return c_clean
            parent = os.path.dirname(c_clean)
            if parent and os.path.isdir(parent):
                return parent
    return ""


def set_last_export_dir(dir_or_file_path: str):
    """Persists last used export directory."""
    if not dir_or_file_path or not isinstance(dir_or_file_path, str):
        return
    clean = dir_or_file_path.strip()
    target_dir = clean if os.path.isdir(clean) else os.path.dirname(clean)
    if not target_dir or not os.path.exists(target_dir):
        return
    cfg = load_config()
    if "preferences" not in cfg:
        cfg["preferences"] = {}
    cfg["preferences"]["last_export_dir"] = target_dir
    if "last_export_settings" not in cfg["preferences"]:
        cfg["preferences"]["last_export_settings"] = {}
    cfg["preferences"]["last_export_settings"]["output_dir"] = target_dir
    save_config(cfg)


def register_sdss_file_association() -> bool:
    """Item 3: Associate .sdss session files with SpektraDarkroom in HKCU registry.
    Requires no administrator elevation because it modifies HKCU\\Software\\Classes.
    """
    if sys.platform != "win32":
        return False
    try:
        import winreg
        import ctypes

        app_dir = get_app_dir()
        exe_path = os.path.join(app_dir, "SpektraDarkroom.exe")
        if not os.path.exists(exe_path):
            exe_path = sys.executable

        ico_path = os.path.join(app_dir, "resources", "sdss_icon.ico")
        if not os.path.exists(ico_path):
            ico_path = os.path.join(app_dir, "resources", "sdss_icon.png")
        if not os.path.exists(ico_path):
            ico_path = os.path.join(app_dir, "resources", "app_icon.ico")

        prog_id = "SpektraDarkroom.Session"
        file_desc = "SpektraDarkroom 暗房会话工程"

        # 1. Register .sdss extension under HKCU\Software\Classes\.sdss
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.sdss") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, prog_id)

        # 2. Register ProgID under HKCU\Software\Classes\SpektraDarkroom.Session
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{prog_id}") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, file_desc)

        if os.path.exists(ico_path):
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{prog_id}\DefaultIcon") as k:
                winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f'"{ico_path}",0')

        # Open command: "exe_path" "%1"
        cmd_str = f'"{exe_path}" "%1"'
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{prog_id}\shell\open\command") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, cmd_str)

        # 3. Notify Windows Shell of file association change
        try:
            SHCNE_ASSOCCHANGED = 0x08000000
            SHCNF_IDLIST = 0x0000
            ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
        except Exception:
            pass

        return True
    except Exception as e:
        logger.warning(f"Failed to register .sdss file association: {e}")
        return False


def is_sdss_file_associated() -> bool:
    """Checks if .sdss is registered to SpektraDarkroom in HKCU."""
    if sys.platform != "win32":
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.sdss") as k:
            val, _ = winreg.QueryValueEx(k, "")
            return val == "SpektraDarkroom.Session"
    except Exception:
        return False


def unregister_sdss_file_association() -> bool:
    """Removes .sdss file association and ProgID from HKCU registry."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        def delete_key_tree(root_key, subkey):
            try:
                with winreg.OpenKey(root_key, subkey, 0, winreg.KEY_ALL_ACCESS) as hkey:
                    while True:
                        try:
                            child = winreg.EnumKey(hkey, 0)
                            delete_key_tree(hkey, child)
                        except OSError:
                            break
                winreg.DeleteKey(root_key, subkey)
            except FileNotFoundError:
                pass

        # 1. Delete HKCU\Software\Classes\.sdss
        delete_key_tree(winreg.HKEY_CURRENT_USER, r"Software\Classes\.sdss")

        # 2. Delete HKCU\Software\Classes\SpektraDarkroom.Session
        delete_key_tree(winreg.HKEY_CURRENT_USER, r"Software\Classes\SpektraDarkroom.Session")

        # 3. Notify Windows Shell of association change
        try:
            SHCNE_ASSOCCHANGED = 0x08000000
            SHCNF_IDLIST = 0x0000
            ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
        except Exception:
            pass

        return True
    except Exception as e:
        logger.warning(f"Failed to unregister .sdss file association: {e}")
        return False


def get_last_dialog_dir(category: str = "photo_import") -> str:
    """Retrieve remembered directory for file dialogs by category, falling back to Pictures/User directory."""
    cfg = load_config()
    dirs = cfg.get("last_dialog_dirs", {})
    last_dir = dirs.get(category, "")
    if last_dir and os.path.isdir(last_dir):
        return last_dir
    # Check general last photo or session path
    if category == "session":
        last_sess = cfg.get("last_session_path", "")
        if last_sess and os.path.exists(last_sess):
            d = os.path.dirname(os.path.abspath(last_sess))
            if os.path.isdir(d):
                return d
    elif category == "photo_import":
        recents = cfg.get("recent_files", [])
        if recents:
            for r in recents:
                if os.path.exists(r):
                    d = os.path.dirname(os.path.abspath(r))
                    if os.path.isdir(d):
                        return d
    # Standard fallbacks
    pictures = os.path.join(os.path.expanduser("~"), "Pictures")
    if os.path.isdir(pictures):
        return pictures
    return os.path.expanduser("~")


def set_last_dialog_dir(category: str, dir_or_file_path: str):
    """Persist remembered directory for file dialogs."""
    if not dir_or_file_path:
        return
    target_dir = dir_or_file_path
    if os.path.isfile(target_dir):
        target_dir = os.path.dirname(target_dir)
    if not os.path.isdir(target_dir):
        return
    cfg = load_config()
    if "last_dialog_dirs" not in cfg:
        cfg["last_dialog_dirs"] = {}
    cfg["last_dialog_dirs"][category] = os.path.abspath(target_dir)
    save_config(cfg)
def get_preference(key: str, default=None):
    """Retrieve a preference value from configuration."""
    cfg = load_config()
    return cfg.get("preferences", {}).get(key, default)


def set_preference(key: str, value):
    """Save a single preference value to configuration."""
    cfg = load_config()
    prefs = cfg.setdefault("preferences", {})
    prefs[key] = value
    save_config(cfg)


