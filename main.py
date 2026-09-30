import os
import sys
import warnings

# Suppress harmless third-party library warnings
warnings.filterwarnings("ignore", message=".*Matplotlib.*")
try:
    import cv2
    cv2.setLogLevel(0)  # Silence OpenCV libtiff metadata tag warnings
except Exception:
    pass

# Ensure the executable or unpack directory is at the head of sys.path
if getattr(sys, 'frozen', False):
    _app_root = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    if _app_root not in sys.path:
        sys.path.insert(0, _app_root)
    _exe_dir = os.path.dirname(sys.executable)
    if _exe_dir not in sys.path:
        sys.path.insert(1, _exe_dir)
else:
    _app_root = os.path.dirname(os.path.abspath(__file__))
    if _app_root not in sys.path:
        sys.path.insert(0, _app_root)
    _spektrafilm_src = os.path.join(_app_root, "spektrafilm-repo", "src")
    if os.path.exists(_spektrafilm_src) and _spektrafilm_src not in sys.path:
        sys.path.insert(0, _spektrafilm_src)

import ctypes
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor

from app_core import SpektraEngine
from ui.main_window import DarkroomMainWindow
from version import get_app_title


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SpektraDarkroom.Vulkan.App.1.0")
        except Exception:
            pass

    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    app.setApplicationName(get_app_title())

    from path_utils import get_resource_dir
    from PySide6.QtGui import QIcon
    icon_path = os.path.join(get_resource_dir(), "app_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Item 19: Single instance check
    mutex_name = "SpektraDarkroom_Vulkan_SingleInstance_Mutex"
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
    last_error = ctypes.windll.kernel32.GetLastError()
    if last_error == 183:  # ERROR_ALREADY_EXISTS
        msg = QMessageBox()
        msg.setWindowTitle("SpektraDarkroom")
        msg.setText("SpektraDarkroom 已经在运行中，请勿重复打开。")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(0)

    # Set Adobe Pro Dark Theme Palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(24, 24, 24))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(225, 225, 225))
    palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 18))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(28, 28, 28))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(36, 36, 36))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.Text, QColor(225, 225, 225))
    palette.setColor(QPalette.ColorRole.Button, QColor(34, 34, 34))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(225, 225, 225))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(245, 158, 11))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # Set modern sans-serif UI font globally
    font = QFont()
    font.setFamilies([
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "Segoe UI",
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "sans-serif"
    ])
    font.setPointSize(9)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    from path_utils import get_resource_dir
    resources_dir = get_resource_dir()
    engine = SpektraEngine(resources_dir=resources_dir)

    win = DarkroomMainWindow(engine)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
