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

import ctypes
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PySide6.QtCore import Qt, QRect, QRectF
from PySide6.QtGui import QFont, QFontMetrics, QPalette, QColor, QPainter, QPen, QBrush, QPixmap, QImage

from version import VERSION_STRING, get_app_title


class DarkroomSplashScreen(QSplashScreen):
    """自定义启动画面：支持高 DPI 视网膜渲染与底部状态文案几何居中"""

    def __init__(self, pixmap: QPixmap):
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self._status_text = ""

    def set_status(self, text: str):
        self._status_text = text
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(self.rect(), self.pixmap())
        self.drawContents(painter)

    def drawContents(self, painter: QPainter):
        if self._status_text:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
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
            font.setPixelSize(11)
            painter.setFont(font)
            painter.setPen(QColor("#a0aec0"))
            # 物理 1280x720, pix.setDevicePixelRatio(2.0), 逻辑尺寸 640x360
            # 物理底栏高度 68px (逻辑 34px), 物理左边距 48px (逻辑 24px)
            # +1 针对汉字光学基线绝对垂直对齐
            text_rect = QRect(24, int(self.height() - 34) + 1, int(self.width() - 200), 34)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._status_text)
            painter.restore()


def select_splash_pixmap(resource_dir: str, app: QApplication, version_string: str = VERSION_STRING) -> QPixmap | None:
    """
    根据当前主显示器物理分辨率与 DPI 缩放比例自适应挑选最佳启动画面底图，
    按目标物理栅格进行高质量 Lanczos 重采样，
    在内存中以目标物理分辨率动态测量并合成版本号胶囊药丸徽章（解耦静态底图文件），
    并设置精确的 devicePixelRatio 保持 640x360 视网膜清晰视口，消除运行时点采样损失。
    """
    screen = app.primaryScreen() if app else None
    dpr = float(screen.devicePixelRatio()) if screen else 1.0

    # 目标物理像素尺寸（逻辑 640x360 * dpr）
    target_w = max(640, int(round(640 * dpr)))
    target_h = max(360, int(round(360 * dpr)))

    # 多分辨率候选梯队：选择能够覆盖目标物理宽度的底图
    candidates = [
        (1280, "splash.png"),
        (1920, "splash@1.5x.png"),
        (2560, "splash@2x.png"),
    ]

    chosen_path = None
    for cand_w, fname in candidates:
        cand_path = os.path.join(resource_dir, fname)
        if os.path.exists(cand_path):
            chosen_path = cand_path
            if cand_w >= target_w:
                break

    if not chosen_path or not os.path.exists(chosen_path):
        chosen_path = os.path.join(resource_dir, "splash.png")

    if not os.path.exists(chosen_path):
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.open(chosen_path).convert("RGBA")

        # 1. 高质量 Lanczos 预缩放到当前屏幕的实际物理像素栅格
        if (img.width, img.height) != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 2. 动态测量与绘制版本号全圆角药丸胶囊 (在目标物理像素栅格上以 1:1 像素精度绘制)
        if version_string:
            ver_text = f"v{version_string}"
            scale = target_w / 1280.0
            font_size = max(10, int(round(16 * scale)))
            font_path = "C:/Windows/Fonts/segoeuib.ttf"
            if not os.path.exists(font_path):
                font_path = "C:/Windows/Fonts/msyh.ttc"
            f_ver = ImageFont.truetype(font_path, font_size)
            b = f_ver.getbbox(ver_text)
            vw, vh = b[2] - b[0], b[3] - b[1]
            badge_h = max(16, int(round(30 * scale)))
            pad_x = max(10, int(round(22 * scale)))
            badge_w = vw + pad_x
            badge_r = badge_h // 2
            badge_x = int(round((867 + 18) * scale))
            title_center_y = int(round(291 * scale))
            badge_y = int(round(title_center_y - badge_h / 2.0))

            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle(
                [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                radius=badge_r,
                fill=(22, 24, 30, 255),
                outline=(245, 158, 11, 255),
                width=max(1, int(round(1 * scale)))
            )
            badge_tx = badge_x + (badge_w - vw) // 2 - b[0]
            badge_ty = badge_y + (badge_h - vh) // 2 - b[1]
            draw.text((badge_tx, badge_ty), ver_text, fill=(245, 158, 11, 255), font=f_ver)

        raw_bytes = img.tobytes("raw", "RGBA")
        qimg = QImage(raw_bytes, img.width, img.height, img.width * 4, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
    except Exception:
        pix = QPixmap(chosen_path)

    # 依据底图物理像素宽度自适应计算设备像素比，始终输出 640x360 黄金比例逻辑视口
    target_logical_w = 640.0
    scale_ratio = pix.width() / target_logical_w
    pix.setDevicePixelRatio(scale_ratio)
    return pix


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
    app.setStyleSheet("""
        QToolTip {
            background-color: rgba(22, 24, 30, 0.88);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 7px 11px;
            font-size: 11px;
            font-weight: 400;
            line-height: 1.4;
            font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
        }
    """)

    splash = None
    if not launched_from_exe:
        pix = select_splash_pixmap(get_resource_dir(), app, VERSION_STRING)
        if pix and not pix.isNull():
            splash = DarkroomSplashScreen(pix)
            splash.show()
            app.processEvents()

    def notify_status(msg):
        print(f"[SPLASH] {msg}", flush=True)
        if splash and not launched_from_exe:
            splash.set_status(msg)
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
