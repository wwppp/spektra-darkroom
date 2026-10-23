import os
import sys

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    core_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(core_dir)

def get_resource_dir():
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        exe_dir = os.path.dirname(sys.executable)
        for cand in [
            os.path.join(base, "resources"),
            os.path.join(exe_dir, "resources"),
            os.path.join(base, "_internal", "resources"),
            os.path.join(exe_dir, "_internal", "resources"),
        ]:
            if os.path.exists(cand):
                return cand
        return os.path.join(base, "resources")
    return os.path.join(get_base_dir(), "resources")

def get_icon_dir():
    res = get_resource_dir()
    return os.path.join(res, "icons")

def get_icon_path(name: str):
    return os.path.join(get_icon_dir(), name)


def get_user_data_dir() -> str:
    """Return user data directory (%APPDATA%/SpektraDarkroom on Windows).
    Used for permanent user settings, lens profiles, and configurations.
    """
    if sys.platform == "win32":
        base = os.getenv("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "SpektraDarkroom")
    elif sys.platform == "darwin":
        d = os.path.join(os.path.expanduser("~/Library/Application Support"), "SpektraDarkroom")
    else:
        xdg_config = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        d = os.path.join(xdg_config, "spektra_darkroom")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def get_default_cache_dir() -> str:
    """Return default system user cache directory (%LOCALAPPDATA%/SpektraDarkroom on Windows)."""
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "SpektraDarkroom")
    elif sys.platform == "darwin":
        d = os.path.join(os.path.expanduser("~/Library/Caches"), "SpektraDarkroom")
    else:
        xdg_cache = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        d = os.path.join(xdg_cache, "spektra_darkroom")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def get_user_cache_dir() -> str:
    """Return user cache directory (custom configured or default %LOCALAPPDATA%/SpektraDarkroom).
    Used for high-speed scratch caches, session images, and compiled textures.
    """
    try:
        from . import config_manager
        custom_d = config_manager.get_preference("custom_cache_dir", "")
        if custom_d and isinstance(custom_d, str):
            custom_d = custom_d.strip()
            if custom_d:
                try:
                    os.makedirs(custom_d, exist_ok=True)
                    if os.path.isdir(custom_d):
                        return custom_d
                except Exception:
                    pass
    except Exception:
        pass
    return get_default_cache_dir()


def get_user_lens_profiles_dir() -> str:
    """Return directory for custom user lens profiles (%APPDATA%/SpektraDarkroom/LensProfiles)."""
    d = os.path.join(get_user_data_dir(), "LensProfiles")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def get_user_session_cache_dir() -> str:
    """Return directory for temporary session raw/float image cache."""
    d = os.path.join(get_user_cache_dir(), "session_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def get_user_lut_cache_dir() -> str:
    """Return directory for runtime compiled 3D LUT cache."""
    d = os.path.join(get_user_cache_dir(), "lut_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def normalize_path(path: str) -> str:
    """Normalize file or directory path to native OS format, resolving mixed slashes and quotes."""
    if not path:
        return ""
    clean = str(path).strip().strip('"').strip("'")
    return os.path.normpath(clean)


def send_to_recycle_bin(paths) -> bool:
    """Send one or more files to the system Recycle Bin (Windows Shell SHFileOperationW).
    Safely preserves files in the user's Recycle Bin instead of permanent deletion.
    """
    if not paths:
        return True
    if isinstance(paths, str):
        paths = [paths]
    valid_paths = [os.path.abspath(p) for p in paths if os.path.exists(p)]
    if not valid_paths:
        return True

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            FO_DELETE = 3
            FOF_ALLOWUNDO = 0x0040
            FOF_NOCONFIRMATION = 0x0010
            FOF_SILENT = 0x0004

            class SHFILEOPSTRUCTW(ctypes.Structure):
                _fields_ = [
                    ("hwnd", wintypes.HWND),
                    ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort),
                    ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", wintypes.LPVOID),
                    ("lpszProgressTitle", wintypes.LPCWSTR),
                ]

            p_from = "\0".join(valid_paths) + "\0\0"
            file_op = SHFILEOPSTRUCTW()
            file_op.hwnd = None
            file_op.wFunc = FO_DELETE
            file_op.pFrom = p_from
            file_op.pTo = None
            file_op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
            res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(file_op))
            return (res == 0) and not file_op.fAnyOperationsAborted
        except Exception:
            pass

    # Fallback to standard delete if shell32 not available or non-windows
    try:
        for p in valid_paths:
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
        return True
    except Exception:
        return False


# Re-export centralized format constants
from .format_constants import (
    RAW_EXTENSIONS,
    STANDARD_IMAGE_EXTENSIONS,
    ALL_SUPPORTED_IMAGE_EXTENSIONS,
    FILE_DIALOG_IMAGE_FILTER,
    get_file_extension,
    is_raw_file,
    is_supported_image
)
