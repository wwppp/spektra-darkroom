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
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QPixmap

from version import VERSION_STRING, get_app_title


def main():
    launched_from_exe = "--launched-from-exe" in sys.argv
    if launched_from_exe:
        sys.argv.remove("--launched-from-exe")

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
    ico_path = os.path.join(get_resource_dir(), "app_icon.ico")
    png_path = os.path.join(get_resource_dir(), "app_icon.png")
    icon_file = ico_path if os.path.exists(ico_path) else png_path
    if os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))

    # Single-instance IPC: forward arguments to running instance or start local server
    from PySide6.QtNetwork import QLocalSocket, QLocalServer
    ipc_name = "SpektraDarkroom_IPC_SingleInstance"

    ipc_sock = QLocalSocket()
    ipc_sock.connectToServer(ipc_name)
    if ipc_sock.waitForConnected(400):
        # Already running: pass all path arguments to the primary instance and exit immediately
        args_to_send = [os.path.abspath(a) for a in sys.argv[1:] if a and not a.startswith("--")]
        payload = "\n".join(args_to_send).encode("utf-8")
        ipc_sock.write(payload)
        ipc_sock.flush()
        ipc_sock.waitForBytesWritten(1000)
        ipc_sock.close()
        sys.exit(0)

    # Primary instance: create server
    local_server = QLocalServer()
    local_server.removeServer(ipc_name)
    local_server.listen(ipc_name)

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

    splash = None
    if not launched_from_exe:
        splash_path = os.path.join(get_resource_dir(), "splash.png")
        if os.path.exists(splash_path):
            pix = QPixmap(splash_path)
            try:
                painter = QPainter(pix)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                ver_text = f"v{VERSION_STRING}"
                font_ver = QFont("Segoe UI", 16, QFont.Weight.Bold)
                painter.setFont(font_ver)
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(ver_text)
                th = fm.height()
                bw = tw + 24
                bh = th + 8
                bx = 898
                by = 233
                painter.setBrush(QBrush(QColor(20, 22, 28, 170)))
                painter.setPen(QPen(QColor(245, 158, 11, 210), 2.0))
                painter.drawRoundedRect(bx, by, bw, bh, 7, 7)
                painter.setPen(QPen(QColor(245, 158, 11)))
                painter.drawText(bx, by, bw, bh, Qt.AlignmentFlag.AlignCenter, ver_text)
                painter.end()
            except Exception:
                pass
            pix.setDevicePixelRatio(2.0)
            splash = QSplashScreen(pix, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
            splash.show()
            app.processEvents()

    def notify_status(msg):
        print(f"[SPLASH] {msg}", flush=True)
        if splash and not launched_from_exe:
            splash.showMessage(f"  {msg}", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, QColor("#cbd5e1"))
            app.processEvents()

    notify_status("正在解析 Qt 核心组件与图形环境...")

    notify_status("正在装载 Rawpy 图像解码动态库...")
    try:
        import rawpy
    except Exception:
        pass

    notify_status("正在装载 OpenCV 图像处理引擎...")
    try:
        import cv2
    except Exception:
        pass

    notify_status("正在装载胶卷与相纸光谱配置文件...")
    from app_core import SpektraEngine
    resources_dir = get_resource_dir()
    engine = SpektraEngine(resources_dir=resources_dir)

    # Check CLI arguments for .sdss session file or image files passed via OS association
    target_session = None
    target_files = []
    for arg in sys.argv[1:]:
        if arg and os.path.exists(arg):
            if arg.lower().endswith(".sdss"):
                target_session = os.path.abspath(arg)
            else:
                target_files.append(os.path.abspath(arg))

    notify_status("正在构建暗房工作台界面...")
    from ui.main_window import DarkroomMainWindow
    win = DarkroomMainWindow(engine, initial_session=target_session, initial_files=target_files)

    notify_status("就绪")
    win.show()

    def handle_ipc_connection():
        while local_server.hasPendingConnections():
            client = local_server.nextPendingConnection()
            if not client:
                continue

            def process_incoming_args(sock=client):
                try:
                    data = bytes(sock.readAll())
                    raw = data.decode("utf-8-sig", errors="ignore")
                    lines = [line.strip().strip('"').strip("'").lstrip('\ufeff') for line in raw.splitlines() if line.strip()]
                    target_sess = None
                    target_imgs = []
                    for p in lines:
                        if not p:
                            continue
                        norm_p = os.path.abspath(p)
                        if os.path.exists(norm_p):
                            if norm_p.lower().endswith(".sdss"):
                                target_sess = norm_p
                            else:
                                target_imgs.append(norm_p)
                    if target_sess:
                        win._open_session_by_path(target_sess)
                    elif target_imgs:
                        win._import_files_list(target_imgs)

                    # Bring window to foreground
                    win.setWindowState(win.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
                    win.show()
                    win.raise_()
                    win.activateWindow()
                    if sys.platform == "win32":
                        try:
                            import ctypes
                            hwnd = int(win.winId())
                            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                            ctypes.windll.user32.SetForegroundWindow(hwnd)
                        except Exception:
                            pass
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass

            client.readyRead.connect(process_incoming_args)
            if client.bytesAvailable() > 0:
                process_incoming_args()

    local_server.newConnection.connect(handle_ipc_connection)
    app._ipc_server = local_server

    if splash and not launched_from_exe:
        splash.finish(win)

    print("[READY]", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
