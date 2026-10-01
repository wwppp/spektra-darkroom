import os
import sys

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

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
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

def get_icon_dir():
    res = get_resource_dir()
    return os.path.join(res, "icons")

def get_icon_path(name: str):
    return os.path.join(get_icon_dir(), name)

def normalize_path(path: str) -> str:
    """Normalize file or directory path to native OS format, resolving mixed slashes and quotes."""
    if not path:
        return ""
    clean = str(path).strip().strip('"').strip("'")
    return os.path.normpath(clean)
