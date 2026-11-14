"""SpektraDarkroom Main Window - GPU Real-Time Accelerated Darkroom
Provides:
- Windows 11 Fluent frameless UI with native DWM rounded corners and native WM_NCHITTEST dragging
- Non-destructive .sdc sidecar parameter management per image file
- 3-column adjustable QSplitters with min/max constraints and outer padding
- 2:3 aspect ratio large photographic preview cards for Film and Paper libraries
- Resizable EXIF panel via vertical QSplitter with internal smooth scrolling
- Universal clean modern scrollbars without checkerboard artifacts
- View toolbar with status on left, view controls on right
- Bottom 35mm filmstrip with multi-selection, horizontal scroll, right-click batch export
- Full Undo/Redo stack with standard Ctrl+Z / Ctrl+Y shortcuts
- Duplicate file import detection and filtering dialog
- Professional export workflow with DPI, color space, bit depth, and compression options
"""

import os
import sys
import copy
import ctypes
import functools
import queue
import threading
import time
from collections import deque
from ctypes import c_int, byref
from datetime import datetime
import logging

logger = logging.getLogger("SpektraDarkroom.MainWindow")

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import (
    Qt, QPoint, QPointF, QSize, QRect, QRectF, Signal, QTimer, QEvent, QThread, QObject,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
)
if sys.platform == "win32":
    from PySide6.QtCore import QAbstractNativeEventFilter
from PySide6.QtGui import (
    QAction, QFont, QFontMetrics, QIcon, QKeySequence, QPainter, QPainterPath, QColor, QPen, QBrush,
    QPixmap, QImage, QLinearGradient, QCursor, QShortcut
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QStackedLayout, QGraphicsOpacityEffect, QMenuBar, QMenu, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSplitter, QSplitterHandle, QGridLayout, QSizePolicy, QApplication,
    QDialog, QLineEdit, QTextEdit, QCheckBox, QButtonGroup
)

from core.path_utils import get_icon_path, get_resource_dir, normalize_path
from core.format_constants import (
    RAW_EXTENSIONS,
    IMPORTABLE_IMAGE_EXTENSIONS,
    LOSSY_IMAGE_EXTENSIONS,
    FILE_DIALOG_IMAGE_FILTER,
    is_importable_image,
    is_lossy_image
)
from core.version import get_app_title, get_full_version_info
from core.app_core import SpektraEngine
from core.halation_model import (
    HALATION_MAX_STOPS,
    HALATION_OFF_STOPS,
    scale_from_stops,
    stops_from_scale,
)
from core import sdc_manager
from core import config_manager

from core import session_cache_manager
from ui.export_queue_manager import ExportQueueManager, ExportQueueDialog, ExportTask
from ui.unsaved_session_dialog import UnsavedSessionDialog
from ui.thumbnail_controller import ThumbnailController, StockThumbnailWorker
from ui.film_filter_header import FilmFilterHeader

if sys.platform == "win32":
    from ctypes import wintypes
    class POINT(ctypes.Structure):
        _fields_ = [
            ("x", wintypes.LONG),
            ("y", wintypes.LONG)
        ]
    class MINMAXINFO(ctypes.Structure):
        _fields_ = [
            ("ptReserved", POINT),
            ("ptMaxSize", POINT),
            ("ptMaxPosition", POINT),
            ("ptMinTrackSize", POINT),
            ("ptMaxTrackSize", POINT),
        ]
    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG)
        ]
    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD)
        ]
    class NCCALCSIZE_PARAMS(ctypes.Structure):
        _fields_ = [
            ("rgrc", RECT * 3),
            ("lppos", ctypes.c_void_p)
        ]

from ui.window_utils import (apply_dark_titlebar, get_darkroom_menu_style, get_icon,
                             get_standard_checkbox_style, show_dark_message_box)
from ui.smooth_scroll import SmoothScrollArea
from ui.preferences_dialog import PreferencesDialog
from ui.about_dialog import AboutDialog
from ui.canvas_viewport import CanvasViewport, DarkroomGLCanvas
from ui.accordion import AccordionSection
from ui.crop_overlay import CropSidebarWidget
from ui.adobe_scrub_slider import AdobeScrubSlider, ComboBoxRow
from ui.export_dialog import (
    SingleExportDialog, BatchExportDialog, ExportImageDialog, ExportProgressDialog, SingleExportWorker, BatchExportWorker
)
from ui.filmstrip_widget import FilmstripWidget, DuplicateFilesDialog
from ui.canvas_rating_widget import CanvasRatingWidget
from ui.spinner_widget import SpinnerWidget
from ui.histogram_widget import HistogramWidget, HistogramExifBar
from ui.fading_stacked_widget import FadingStackedWidget
from ui.theme_manager import theme_manager
from ui import ui_scale
from ui.tooltip import block_tooltips_under


# 参数面板的悬停提示统一关掉：滑块的长说明、以及「为什么这个参数现在不可用」的原因，
# 都会变成气泡压在参数本身上面，拖动时还反复弹出。文案本身保留在控件里（可用性刷新
# 仍在写 getattr(w, "default_tooltip") 那一套），只是不再显示，随时可以改回 True。
PARAM_TOOLTIPS_ENABLED = False


# 关闭窗口时在屏幕上多停的这一下（毫秒）。保存/销毁本来几百毫秒内就做完了，窗口在同一
# 毫秒里隐藏，看起来"秒响应"；这个停留让关闭有一个可感知的过程。
# 现在这一段是**真的占住 GUI 线程**（窗口不响应、指针转圈），所以时长按"看起来在处理"来定，
# 不再按"动画"来定（用户 2026-11-10：先要 0.5-1 s，加到 1.4 s 后又说"转圈稍微长了点"）。
CLOSE_DWELL_MS = 1100


# Opt-in window-frame diagnostics: set SPEKTRA_FRAME_DEBUG=1 to log every non-client message and
# window-state transition to %TEMP%\spektra_frame.log. Used to pin down the classic-titlebar
# flash if it ever comes back.
_FRAME_DEBUG = bool(os.environ.get("SPEKTRA_FRAME_DEBUG"))
_FRAME_LOG = os.path.join(os.environ.get("TEMP", "."), "spektra_frame.log")
# Window frame strategy (SPEKTRA_FRAME_MODE):
#   anim (default) - keep WS_CAPTION so DWM keeps animating minimise/maximise. The caption is
#                    kept invisible by taking the whole window as client area in WM_NCCALCSIZE;
#                    the DWM non-client rendering policy is chosen by SPEKTRA_NC_POLICY below.
#   legacy         - same styles, leaves the DWM policy at its default (see SPEKTRA_NC_POLICY=0).
#   clean          - drop WS_CAPTION entirely: no caption can ever be drawn, but the DWM zoom
#                    animation is gone too (it needs a frame).
_FRAME_MODE = os.environ.get("SPEKTRA_FRAME_MODE", "anim").strip().lower()

# DWM non-client rendering policy for the "anim" frame (SPEKTRA_NC_POLICY, 0..2):
#   0 - USEWINDOWSTYLE (default). DWM knows the window has a frame, so
#       DWMWA_EXTENDED_FRAME_BOUNDS resolves to the work area. Screenshot / window-picker
#       tools (WeChat, Snipping Tool) read that attribute, so they frame the window correctly.
#   2 - ENABLED. Same frame bounds as 0.
#   1 - DISABLED. Stops DWM from ever painting the classic caption (v0.4.54 set this to kill
#       the title-bar flash) but it also erases the frame from DWM's model, so
#       EXTENDED_FRAME_BOUNDS degrades to the raw window rect -- work area + the 8 px invisible
#       resize border ring, which hangs over the taskbar (see docs). Only keep it if the
#       caption flash returns and cannot be fixed any other way.
try:
    _NC_POLICY = int(os.environ.get("SPEKTRA_NC_POLICY", "0"))
except ValueError:
    _NC_POLICY = 0
if _NC_POLICY not in (0, 1, 2):
    _NC_POLICY = 0


if sys.platform == "win32":
    class _MaximizeScreenFilter(QAbstractNativeEventFilter):
        """Maximize into the monitor the window is actually on.

        The window is frameless, and for that case the maximize rectangle does *not* come
        from "wherever the window happens to be": Qt answers ``WM_GETMINMAXINFO`` from the
        platform window's screen, which for this window stays the primary monitor.  So
        maximizing while the window sat on the second monitor teleported it across --
        measured ``(2020,184,1200x760)`` -> ``(0,0,1920x1032)`` (the primary work area)
        instead of ``(1920,124,1600x852)``.  The same jump happened even through a raw
        ``ShowWindow(SW_MAXIMIZE)``, which is what pinned the cause on this message rather
        than on ``showMaximized``.

        Answering the message ourselves -- and returning True so Qt cannot overwrite the
        values -- makes maximize use the work area of the monitor the window is on, which
        is also what keeps the taskbar strip clear.
        """
        WM_GETMINMAXINFO = 0x0024
        MONITOR_DEFAULTTONEAREST = 2

        def __init__(self, window):
            super().__init__()
            self._window = window

        def nativeEventFilter(self, event_type, message):
            if event_type not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
                return False, 0
            try:
                msg = wintypes.MSG.from_address(int(message))
                if msg.message != self.WM_GETMINMAXINFO:
                    return False, 0
                hwnd = int(self._window.winId())
                if int(msg.hWnd or 0) != hwnd:
                    # Dialogs, menus and tooltips keep the default behaviour.
                    return False, 0
                monitor = ctypes.windll.user32.MonitorFromWindow(hwnd, self.MONITOR_DEFAULTTONEAREST)
                info = MONITORINFO()
                info.cbSize = ctypes.sizeof(MONITORINFO)
                if not ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                    return False, 0
                mon, work = info.rcMonitor, info.rcWork
                mmi = MINMAXINFO.from_address(int(msg.lParam))
                mmi.ptMaxPosition.x = work.left - mon.left
                mmi.ptMaxPosition.y = work.top - mon.top
                mmi.ptMaxSize.x = work.right - work.left
                mmi.ptMaxSize.y = work.bottom - work.top
                return True, 0
            except Exception:
                return False, 0


def _frame_log(line):
    if not _FRAME_DEBUG:
        return
    try:
        with open(_FRAME_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} {line}\n")
    except Exception:
        pass


class _LutBakeBridge(QObject):
    """Carries a finished background LUT build back to the GUI thread.

    Emitted from a plain worker thread, delivered as a queued connection, so the canvas
    texture upload always happens on the GUI thread.
    """

    baked = Signal(object, object, object)  # lut_3d, key, chem_snapshot


class WindowCaptionButton(QPushButton):
    """Vector-drawn Windows 11 style window control button."""
    def __init__(self, btn_type="close", parent=None):
        super().__init__(parent)
        self.btn_type = btn_type
        self.setFixedSize(46, 38)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.is_maximized_state = False
        theme_manager.themeChanged.connect(lambda _: self.update())

    def set_maximized_state(self, is_max):
        if self.is_maximized_state != is_max:
            self.is_maximized_state = is_max
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        is_dark = theme_manager.is_dark()

        # 1. 始终彻底填充背景，防止重绘/DPI变换时的残影与双边伪影
        if self.isDown():
            bg = QColor(241, 112, 122) if self.btn_type == "close" else (QColor(38, 40, 50) if is_dark else QColor(218, 222, 230))
        elif self.underMouse():
            bg = QColor(232, 17, 35) if self.btn_type == "close" else (QColor(48, 50, 60) if is_dark else QColor(226, 232, 240))
        else:
            bg = QColor(20, 21, 24) if is_dark else QColor(240, 242, 246)

        win = self.window()
        is_max = self.is_maximized_state or (win.isMaximized() if win else False)
        if not is_max and self.btn_type == "close":
            from PySide6.QtGui import QPainterPath
            path = QPainterPath()
            r = 8.0
            w_btn = float(self.width())
            h_btn = float(self.height())
            path.moveTo(0, 0)
            path.lineTo(w_btn - r, 0)
            path.quadTo(w_btn, 0, w_btn, r)
            path.lineTo(w_btn, h_btn)
            path.lineTo(0, h_btn)
            path.closeSubpath()
            painter.setClipPath(path)

        painter.fillRect(self.rect(), bg)

        icon_color = QColor(255, 255, 255) if (self.underMouse() and self.btn_type == "close") else (QColor(220, 222, 230) if is_dark else QColor(30, 41, 59))
        cx, cy = self.width() // 2, self.height() // 2

        if self.btn_type == "min":
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            pen = QPen(icon_color, 1.0)
            painter.setPen(pen)
            painter.drawLine(cx - 5, cy, cx + 4, cy)
        elif self.btn_type == "max":
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            pen = QPen(icon_color, 1.0)
            painter.setPen(pen)
            if not self.is_maximized_state:
                # 绘制单框 (精准 10x10 像素矢量方框: [cx-5 .. cx+4], [cy-5 .. cy+4])
                painter.drawRect(cx - 5, cy - 5, 9, 9)
            else:
                # 绘制双框还原图标 (后框 + 前框带背景遮罩，各精准 8x8 像素)
                painter.drawRect(cx - 3, cy - 6, 7, 7)
                painter.fillRect(cx - 6, cy - 3, 8, 8, bg)
                painter.drawRect(cx - 6, cy - 3, 7, 7)
        elif self.btn_type == "close":
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(icon_color, 1.1)
            painter.setPen(pen)
            painter.drawLine(QPointF(cx - 4.5, cy - 4.5), QPointF(cx + 4.5, cy + 4.5))
            painter.drawLine(QPointF(cx + 4.5, cy - 4.5), QPointF(cx - 4.5, cy + 4.5))


class DraggableTitleBar(QFrame):
    """Custom title bar supporting system window drag and double-click maximize/restore."""
    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self.main_window = main_window
        self._press_pos = None
        self._drag_started_while_maximized = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._drag_started_while_maximized = bool(self.main_window.isMaximized())
            if not self._drag_started_while_maximized:
                handle = self.main_window.windowHandle()
                if handle:
                    handle.startSystemMove()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            # 只有当拖拽起始于【已最大化状态】且明确向下拉动时，才触发还原
            # 普通窗口拖到屏幕顶部被 Windows Aero Snap 最大化时不会误触发
            if self._drag_started_while_maximized and self.main_window.isMaximized():
                delta = event.position().toPoint() - self._press_pos
                if delta.y() > 10:
                    norm_geo = getattr(self.main_window, '_saved_normal_geo', None)
                    norm_w = norm_geo.width() if norm_geo else 1440
                    norm_h = norm_geo.height() if norm_geo else 900
                    ratio_x = self._press_pos.x() / float(max(1, self.width()))
                    self.main_window.showNormal()
                    self.main_window.resize(norm_w, norm_h)
                    new_x = int(event.globalPosition().x() - norm_w * ratio_x)
                    new_y = int(event.globalPosition().y() - self._press_pos.y())
                    self.main_window.move(new_x, new_y)
                    self._press_pos = None
                    self._drag_started_while_maximized = False
                    handle = self.main_window.windowHandle()
                    if handle:
                        handle.startSystemMove()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        self._drag_started_while_maximized = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = None
            self._drag_started_while_maximized = False
            self.main_window.toggle_maximize()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class SinglePhotoLoadWorker(QThread):
    """Background worker that decodes a single high-resolution RAW photo
    without freezing the GUI, ensuring status spinners rotate smoothly.
    Also extracts fast camera-embedded JPEG preview (~10-15ms) for progressive clarity progression.
    """
    photoLoaded = Signal(str, str, dict)  # photo_id, photo_path, result_dict
    previewReady = Signal(str, str, object)  # photo_id, photo_path, preview_rgb (np.ndarray)

    def __init__(self, photo_id: str, path: str, engine, parent=None):
        super().__init__(parent)
        self.photo_id = photo_id
        self.path = path
        self.engine = engine
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            # 0. 磁盘会话缓存优先：有 2048 px 预览就不必再解一遍 RAW。
            #    这一步原本在主线程里做（`on_switch_photo` 同步 `get_photo_preview`），
            #    读一次 .npy 实测 60-70 ms —— 那是"有缓存的反而更卡"的主因之一。
            try:
                cached = session_cache_manager.get_photo_preview(self.path)
            except Exception:
                cached = None
            if cached is not None and not self._stopped:
                self.photoLoaded.emit(self.photo_id, self.path,
                                      {"success": True, "float_img": cached, "from_cache": True})
                return

            # 1. Fast camera-embedded JPEG preview extraction (~10-15ms)
            # Provides progressive multi-stage clarity: micro-thumb (0ms) -> 1080p preview (15ms) -> master (300ms)
            try:
                ext = os.path.splitext(self.path)[1].lower()
                if ext in RAW_EXTENSIONS:
                    import rawpy
                    with rawpy.imread(self.path) as raw:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            import cv2
                            emb_bgr = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
                            if emb_bgr is not None:
                                rw, rh = raw.sizes.width, raw.sizes.height
                                flip = getattr(raw.sizes, "flip", 0)
                                if flip in (4, 5, 6, 7):
                                    w, h = rh, rw
                                else:
                                    w, h = rw, rh
                                th_h_raw, th_w_raw = emb_bgr.shape[:2]
                                if (h > w and th_w_raw > th_h_raw) or (w > h and th_h_raw > th_w_raw):
                                    emb_bgr = cv2.rotate(emb_bgr, cv2.ROTATE_90_CLOCKWISE)
                                emb_rgb = cv2.cvtColor(emb_bgr, cv2.COLOR_BGR2RGB)
                                if not self._stopped:
                                    self.previewReady.emit(self.photo_id, self.path, emb_rgb)
            except Exception as e:
                logger.debug(f"Fast embedded preview extraction in SinglePhotoLoadWorker: {e}")

            if self._stopped:
                return

            # 2. Full demosaiced floating-point master load (~200-300ms)
            res = self.engine.load_image(self.path)
            if res and res.get("success") and not self._stopped:
                f_img = res.get("float_img")
                if f_img is None:
                    f_img = res.get("raw_preview")
                if f_img is not None:
                    try:
                        session_cache_manager.save_photo_preview(self.path, f_img)
                    except Exception:
                        pass
            if not self._stopped:
                self.photoLoaded.emit(self.photo_id, self.path, res if isinstance(res, dict) else {"success": False})
        except Exception as e:
            if not self._stopped:
                self.photoLoaded.emit(self.photo_id, self.path, {"success": False, "error": str(e)})


class BatchImportWorker(QThread):
    """Background worker that incrementally loads RAW and image files using
    a high-concurrency multi-threaded CPU pool, streaming results to the GUI
    in strict sequential order without freezing the main thread.
    """
    photoLoaded = Signal(dict, int, int)
    importFinished = Signal(int)

    def __init__(self, file_paths, engine, start_idx=0, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.engine = engine
        self.start_idx = start_idx
        self._stopped = False
        self._import_finished_emitted = False

    def stop(self):
        self._stopped = True

    def run(self):
        """Decode the batch, always emitting importFinished exactly once.

        The window clears its "importing" lock (and with it the status bar lock) only when
        importFinished arrives, so an exception escaping the decode loop would freeze the
        status bar — and every later "saved"/"exported" message — for the whole session.
        """
        loaded = 0
        try:
            loaded = self._run_import()
        except Exception as ex:
            logger.warning(f"BatchImportWorker aborted: {ex}")
        finally:
            if not self._import_finished_emitted:
                self._import_finished_emitted = True
                self.importFinished.emit(int(loaded or 0))

    def _run_import(self):
        total = len(self.file_paths)
        if total == 0:
            return 0

        import os
        import concurrent.futures

        preview_max_edge = 2048
        try:
            from core import config_manager
            preview_max_edge = int(config_manager.get_preferences().get("preview_max_edge", 2048))
        except Exception:
            preview_max_edge = 2048

        # Multi-core CPU concurrency pool capped to safe threshold (avoids LibRaw OOM and UI starvation)
        max_workers = min(3, max(2, (os.cpu_count() or 4) // 2))
        logger.info(f"BatchImportWorker starting with {max_workers} worker threads for {total} files")

        loaded_count = 0

        def _decode_item(idx, file_path):
            if self._stopped:
                return idx, None
            try:
                import time
                cached = session_cache_manager.get_photo_cache(file_path, load_preview=False)
                if cached is not None:
                    trgb = cached.get("thumbnail_rgb")
                    if session_cache_manager.is_dummy_or_corrupted_thumbnail(trgb):
                        cached = None
                if cached is not None:
                    pid = f"photo_{self.start_idx + idx}_{os.path.basename(file_path)}"
                    entry = {
                        "id": pid,
                        "filename": os.path.basename(file_path),
                        "path": file_path,
                        "width": cached.get("width", 0),
                        "height": cached.get("height", 0),
                        "thumbnail_rgb": cached.get("thumbnail_rgb"),
                        "raw_thumbnail_rgb": cached.get("raw_thumbnail_rgb") if cached.get("raw_thumbnail_rgb") is not None else cached.get("thumbnail_rgb"),
                        "graded_thumbnail_rgb": cached.get("graded_thumbnail_rgb"),
                        "float_img": None,
                        "raw_preview": None,
                        "exif": cached.get("exif", {}),
                        "auto_ev": cached.get("auto_ev", 0.0),
                        # Session caches written by older builds carry the auto-metered value in
                        # base_ev; the meter no longer ships one (it stays in auto_ev, applied
                        # only through the auto-exposure button), and the photo-load chain
                        # consults the sdc sidecar for explicitly saved values.
                        "base_ev": 0.0,
                        "is_raw": cached.get("is_raw", False)
                    }
                    time.sleep(0.001)
                    return idx, entry

                # 2. Parallel thread-safe decode:
                # Fast path: Extract camera-embedded JPEG preview in sub-15ms for all photos!
                # (The active photo gets full 16-bit linear demosaic progressively via SinglePhotoLoadWorker)
                entry = self.engine.decode_photo_data(
                    file_path,
                    preview_max_edge=preview_max_edge,
                    is_first=False,
                    fast_thumbnail=True
                )
                if entry and entry.get("success"):
                    pid = f"photo_{self.start_idx + idx}_{os.path.basename(file_path)}"
                    entry["id"] = pid
                    if sdc_manager.has_sdc(file_path):
                        sdc_data = sdc_manager.load_sdc(file_path)
                        if sdc_data:
                            if abs(float(entry.get("base_ev", 0.0))) < 0.0001 and sdc_data.get("params", {}).get("base_ev"):
                                entry["base_ev"] = float(sdc_data["params"]["base_ev"])
                    entry["rating"] = 0
                    entry["raw_thumbnail_rgb"] = entry.get("raw_thumbnail_rgb") if entry.get("raw_thumbnail_rgb") is not None else entry.get("thumbnail_rgb")
                    entry["graded_thumbnail_rgb"] = entry.get("graded_thumbnail_rgb")
                    entry["float_img"] = None
                    entry["raw_preview"] = None
                    time.sleep(0.001)
                    return idx, entry
                else:
                    logger.warning(f"Failed to decode {file_path}: {entry.get('error') if entry else 'Unknown error'}")
            except Exception as ex:
                logger.warning(f"Error decoding {file_path}: {ex}")

            # Fallback placeholder: attempt emergency standalone fast thumbnail extraction
            try:
                fast_th = session_cache_manager.extract_fast_thumbnail(file_path)
                dw, dh = session_cache_manager.get_image_dimensions(file_path)
                if dw <= 0 or dh <= 0:
                    dw, dh = 6000, 4000
                if fast_th is None or not isinstance(fast_th, np.ndarray) or fast_th.size == 0:
                    fast_th = np.full((72, 76, 3), 32, dtype=np.uint8)
                    fast_th[::4, ::4] = 48
                fallback_entry = {
                    "id": f"photo_{self.start_idx + idx}_{os.path.basename(file_path)}",
                    "filename": os.path.basename(file_path),
                    "path": file_path,
                    "width": dw,
                    "height": dh,
                    "rating": 0,
                    "thumbnail_rgb": fast_th,
                    "raw_thumbnail_rgb": fast_th,
                    "graded_thumbnail_rgb": None,
                    "float_img": None,
                    "raw_preview": None,
                    "exif": {"filename": os.path.basename(file_path), "path": file_path, "width": dw, "height": dh},
                    "auto_ev": 0.0,
                    "base_ev": 0.0,
                    "is_raw": True,
                    "decode_error": True
                }
                return idx, fallback_entry
            except Exception:
                dummy_th = np.full((72, 76, 3), 32, dtype=np.uint8)
                dummy_th[::4, ::4] = 48
                fallback_entry = {
                    "id": f"photo_{self.start_idx + idx}_{os.path.basename(file_path)}",
                    "filename": os.path.basename(file_path),
                    "path": file_path,
                    "width": 6000,
                    "height": 4000,
                    "rating": 0,
                    "thumbnail_rgb": dummy_th,
                    "raw_thumbnail_rgb": dummy_th,
                    "graded_thumbnail_rgb": None,
                    "float_img": None,
                    "raw_preview": None,
                    "exif": {"filename": os.path.basename(file_path), "path": file_path, "width": 6000, "height": 4000},
                    "auto_ev": 0.0,
                    "base_ev": 0.0,
                    "is_raw": True,
                    "decode_error": True
                }
                return idx, fallback_entry

        # 1. Priority fast-lane for the very first photo (index 0):
        # Decode index 0 immediately before launching the remaining pool so the canvas and status bar
        # update to (1/total) within ~15ms with zero thread contention!
        import time
        if total > 0 and not self._stopped:
            _, first_entry = _decode_item(0, self.file_paths[0])
            if first_entry is not None and not self._stopped:
                loaded_count += 1
                self.photoLoaded.emit(first_entry, loaded_count, total)

        # 2. Execute remaining files concurrently with paced emission so the GUI thread stays 60 FPS responsive
        if total > 1 and not self._stopped:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_decode_item, i, self.file_paths[i]): i for i in range(1, total)}
                for future in concurrent.futures.as_completed(future_map):
                    if self._stopped:
                        break
                    idx, entry = future.result()
                    # 不保序发射。底片栏的格子已经按文件顺序摆好了（append_placeholders），
                    # 主线程按 id 原地填图，所以"到货顺序"不再影响显示顺序。以前为了保序
                    # 在这里攒结果，只要有一张慢（实测偶发 1.5 s）后面的就全压着 —— 用户
                    # 看到的是"两秒毫无变化，然后一下全出来"。
                    if entry is not None and not self._stopped:
                        loaded_count += 1
                        self.photoLoaded.emit(entry, loaded_count, total)
                        if loaded_count % 4 == 0:
                            time.sleep(0.004)

        self._emit_import_finished(loaded_count)
        return loaded_count

    def _emit_import_finished(self, loaded_count):
        self._import_finished_emitted = True
        self.importFinished.emit(int(loaded_count or 0))


class ThumbPrimeWorker(QThread):
    """给底片栏第一格先抓一张内嵌 JPEG。

    第一张在导入/恢复 worker 里走"全分辨率优先"，实测 1.5-3 s 才回来，而这一格正好是
    用户第一眼盯的位置 —— 空上几秒看起来就是"第一张缩略图坏了"。内嵌 JPEG 本身不贵，
    但 `extract_fast_thumbnail` 要开 rawpy（实测 190 ms），不能放主线程。
    """
    primed = Signal(str, object)   # photo_id, rgb uint8

    def __init__(self, photo_id, path, parent=None):
        super().__init__(parent)
        self._pid = photo_id
        self._path = path
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        # 只跑 15 ms 的一件小事，但导入/恢复那几条解码线程会把 CPU 吃满 —— 实测不提高
        # 优先级时这一格要 1.5 s 才亮，提上去就是几十毫秒（给用户第一眼的位置用）。
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)   # ABOVE_NORMAL
        except Exception:
            pass
        try:
            th = session_cache_manager.extract_fast_thumbnail(self._path)
        except Exception:
            th = None
        if th is not None and not self._stopped:
            self.primed.emit(self._pid, th)


class ThumbGradeWorker(QThread):
    """导入期把 `.sdc` 缩略图的 3D LUT 现烤挪出主线程。

    每张 `.sdc` 带自己的一套参数，LUT 缓存键因此张张不同、几乎全 miss：一次 17³ 烘焙
    120-730 ms，26 张就是 4.4 s，全部压在主线程上 —— 那是"导入全程点不动"的头号来源
    （实测单条 `_on_single_photo_imported` 最长 1890 ms，其中 726 ms 是这一项）。
    `engine.get_3d_lut` 自带 `_lut_bake_lock`，decode 线程本来就在并发用同一个 engine，
    所以这里只是把同样的活换到另一个线程，语义不变。
    """
    thumbGraded = Signal(str, object)  # photo_id, graded uint8 thumbnail

    def __init__(self, thumb_controller, parent=None):
        super().__init__(parent)
        self._tc = thumb_controller
        self._jobs = queue.Queue()
        self._stopped = False

    def enqueue(self, photo_id, raw_thumb, film, paper, params):
        if self._stopped or raw_thumb is None:
            return
        self._jobs.put((photo_id, raw_thumb, film, paper, dict(params or {})))

    def pending(self):
        return self._jobs.qsize()

    def stop(self):
        self._stopped = True
        self._jobs.put(None)

    def run(self):
        while not self._stopped:
            job = self._jobs.get()
            if job is None or self._stopped:
                break
            pid, raw, film, paper, params = job
            try:
                graded = self._tc.grade_thumbnail_fast(raw, film=film, paper=paper, params=params)
            except Exception as ex:
                logger.warning(f"Deferred thumbnail grade failed for {pid}: {ex}")
                graded = None
            if graded is not None and not self._stopped:
                self.thumbGraded.emit(pid, graded)


# 导入期把 worker 的结果按帧预算投放到主线程：单条最重的可能是 700-1800 ms 的调用，
# 而 worker 一秒钟能吐出几十条，一次事件循环把十几条连跑完就是"整段卡死"。
IMPORT_DRAIN_TICK_MS = 16
IMPORT_DRAIN_BUDGET_MS = 16.0
IMPORT_DRAIN_MAX_ITEMS = 12
# 导入期首图只铺内嵌 JPEG，全分辨率链等导入结束再跑（那一次 1.4-1.9 s，见
# `_show_import_placeholder`）；留一拍先让状态栏和底片条画出来。
IMPORT_DEFERRED_SWITCH_DELAY_MS = 120


# A measurement is only complete when all four keys are present: sidecars written before the
# constant-magnification term existed carry only the two polynomial triplets, and reusing them
# would silently apply the old (corner-only) model.
_AUTO_CA_KEYS = ("auto_ca_r", "auto_ca_b", "auto_ca_scale_r", "auto_ca_scale_b")

# Status-line copy shared by several call sites (see docs/status-text-inventory.md §5).
STATUS_IMPORTING = "正在导入底片（{cur}/{tot}）…"
STATUS_LOSSY_SKIPPED = "已跳过 {n} 个有损压缩文件（JPG/WebP），仅支持 RAW 与无损图像"


class CAEstimateWorker(QThread):
    """Measures lateral CA for the current frame off the GUI thread.

    The estimate costs ~1 s on a 2048 px preview (it fits a radial model over a few
    hundred thousand edge pixels), which is far too long to run inside a checkbox slot.
    The result is cached in the sidecar, so this runs once per photo at most.
    """
    caReady = Signal(str, dict)  # photo_id, result dict

    def __init__(self, photo_id: str, image, parent=None):
        super().__init__(parent)
        self.photo_id = photo_id
        self.image = image
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            from core.ca_estimate import estimate_lateral_ca
            res = estimate_lateral_ca(self.image)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"自动色差估计失败: {e}")
            res = {"applied": False, "confidence": 0.0,
                   "ca_r_coeffs": (0.0, 0.0, 0.0), "ca_b_coeffs": (0.0, 0.0, 0.0),
                   "r_shift_px": 0.0, "b_shift_px": 0.0}
        self.image = None
        if not self._stopped:
            self.caReady.emit(self.photo_id, res)


def _make_thread_low_priority():
    """A callable that drops the *calling* thread to below-normal priority, once.

    Decoding RAW is a long C loop.  Sixteen of those at normal priority starve the GUI
    thread on every repaint: the main thread is technically alive but cannot get a time
    slice, so the status bar's cancel button reads as dead.  Threads are pooled, so the
    flag lives on a thread-local.
    """
    state = threading.local()
    THREAD_PRIORITY_BELOW_NORMAL = -1

    def apply():
        if getattr(state, "done", False):
            return
        state.done = True
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(),
                                       THREAD_PRIORITY_BELOW_NORMAL)
        except Exception:
            pass

    return apply


class SlotDecodeWorker(QThread):
    """Background worker for session restoration: decodes uncached photos directly
    into pre-allocated slot indices, streaming thumbnail pixmaps without affecting list order.
    """
    slotDecoded = Signal(int, dict)
    progress = Signal(int, int)
    finished = Signal()

    def __init__(self, uncached_items, engine, parent=None):
        super().__init__(parent)
        self.uncached_items = uncached_items  # list of (slot_idx, file_path)
        self.engine = engine
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        if not self.uncached_items:
            self.finished.emit()
            return
        import concurrent.futures
        preview_max_edge = 2048
        try:
            from core import config_manager
            preview_max_edge = int(config_manager.get_preferences().get("preview_max_edge", 2048))
        except Exception:
            preview_max_edge = 2048

        # 这个池子和导入是同一件事（解 RAW），并发也该一致：16 个线程会把每一核都占满，
        # 主线程连一次重绘都要排队，"取消导入"于是看着像点不动。3 个线程足够喂满磁盘，
        # 又留得出余量给 UI。
        max_workers = min(3, max(2, (os.cpu_count() or 4) // 2))
        total_items = len(self.uncached_items)
        done_count = 0
        low_priority = _make_thread_low_priority()

        def _decode(slot_idx, file_path):
            if self._stopped:
                return slot_idx, None
            low_priority()
            try:
                entry = self.engine.decode_photo_data(file_path, preview_max_edge=preview_max_edge, is_first=False, fast_thumbnail=True)
                if entry and entry.get("success"):
                    return slot_idx, entry
            except Exception as e:
                logger.warning(f"Error decoding slot {slot_idx} ({file_path}): {e}")
            return slot_idx, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_decode, i, p): i for (i, p) in self.uncached_items}
            for future in concurrent.futures.as_completed(future_map):
                if self._stopped:
                    break
                slot_idx, entry = future.result()
                done_count += 1
                self.progress.emit(done_count, total_items)
                if entry is not None and not self._stopped:
                    self.slotDecoded.emit(slot_idx, entry)

        if not self._stopped:
            self.finished.emit()


class SessionRestoreWorker(QThread):
    """Background worker for ultra-fast asynchronous session restoration.
    - Decouples disk I/O (.npz cache unzipping, .sdc JSON reading, CPU LUT grading) from Qt GUI thread.
    - Emits first active photo immediately (<30ms) for instantaneous canvas rendering.
    - Concurrent multi-worker extraction for remaining photos with in-order assembly.
    - Zero GUI freezing during session transitions.
    """
    activePhotoReady = Signal(int, dict)       # target_idx, target_photo_entry
    # 每处理完一张就发一张（乱序）。底片栏的格子已经按文件顺序摆好，收到就按 id 填，
    # 所以恢复过程能像导入那样一张张长出来，而不是等 restoreFinished 一次性冒出来。
    photoReady = Signal(int, dict)             # idx, photo_entry
    progress = Signal(int, int)                # current, total
    restoreFinished = Signal(list, list, str)  # all_photos, uncached_slots, target_path
    failed = Signal(str)

    def __init__(self, existing_files, active_file=None, thumb_controller=None, engine=None,
                 is_params_modified_fn=None, ratings_map=None, default_paper=None,
                 default_ffchain_paper=None, parent=None):
        super().__init__(parent)
        self.existing_files = list(existing_files or [])
        self.active_file = active_file
        self.thumb_controller = thumb_controller
        self.engine = engine
        self.is_params_modified_fn = is_params_modified_fn
        self.ratings_map = dict(ratings_map or {})
        # 两条链各一个默认相纸；负片且 .sdc 没记相纸的缩略图按它自己的卷取，避免整条胶卷条
        # 都是橙色负像（works on a copy so the worker does not read config from a thread）
        self.default_paper = default_paper or "none"
        self.default_ffchain_paper = default_ffchain_paper or "kodak_2383"
        self._stopped = False

    def _default_paper_for(self, film_profile):
        """这张照片的卷该落到哪张默认纸（与主窗口同一套链条判定）。

        以前整个 worker 只有一个 `default_paper`，会话里只要混着一条转换卷，它的缩略图就会
        拿原生链那张纸去烘 —— 烘焙时被 clamp 成别的纸，卡片文字和画面不是一回事。
        """
        if not film_profile or film_profile == "none":
            return self.default_paper
        try:
            if self.engine is not None and self.engine.paper_scope(film_profile) == "ffchain":
                paper = self.default_ffchain_paper
                try:
                    if not self.engine.paper_available(film_profile, paper):
                        return "kodak_2383"
                except Exception:
                    pass
                return paper
        except Exception:
            pass
        return self.default_paper

    def stop(self):
        self._stopped = True

    def _process_single_photo(self, idx: int, p: str, is_target: bool) -> tuple:
        """Processes a single photo entry: sdc, cache, thumbnails, dimensions."""
        from core import sdc_manager
        pid = f"photo_{idx}_{os.path.basename(p)}"
        sdc = sdc_manager.load_sdc(p)
        norm_p = os.path.normpath(os.path.abspath(p))
        if self.ratings_map and norm_p in self.ratings_map:
            rating = int(self.ratings_map[norm_p])
        else:
            rating = 0
        film_profile = sdc.get("film_profile", "none") if sdc else "none"
        paper_profile = sdc.get("paper_profile", "none") if sdc else "none"
        paper_by_scope = dict((sdc or {}).get("paper_by_scope") or {})
        print_mode = sdc.get("print_mode", sdc.get("params", {}).get("print_mode", "optical")) if sdc else "scan"
        params = dict(sdc.get("params", {})) if sdc else None
        is_dirty = False

        params_modified = False
        if callable(self.is_params_modified_fn):
            params_modified = bool(self.is_params_modified_fn(params))
        elif params:
            params_modified = any(abs(float(params.get(k, 0.0))) > 0.001 for k in ("exposure_ev", "highlights", "shadows"))

        is_edited = bool(sdc is not None and (
            film_profile not in ("none", "", None) or
            paper_profile not in ("none", "", None) or
            params_modified
        ))

        cache_item = session_cache_manager.get_photo_cache(p, load_preview=is_target)

        raw_th = None
        graded_th = None
        if cache_item:
            for k in ("raw_thumbnail_rgb", "raw_thumbnail", "thumbnail_rgb"):
                cand = cache_item.get(k)
                if session_cache_manager.is_valid_thumbnail(cand):
                    raw_th = cand
                    break

        if not session_cache_manager.is_valid_thumbnail(raw_th) and os.path.isfile(p):
            try:
                extracted = session_cache_manager.extract_fast_thumbnail(p)
                if session_cache_manager.is_valid_thumbnail(extracted):
                    raw_th = extracted
            except Exception:
                pass

        th_rgb = None
        # 负片但 .sdc 没记相纸（旧数据 / 直扫遗留）：缩略图按首选项的默认相纸渲染。
        # 内嵌缩略图是那张"直扫橙片"，这里必须跳过它重新 grade，否则整条胶卷条还是橙色。
        render_paper = paper_profile
        implicit_paper = False
        explicit_paper = bool(sdc and sdc.get("paper_explicit"))
        chain_paper = self._default_paper_for(film_profile)

        # 链归属：.sdc 里的 paper_profile 可能是另一条链存的（旧档、或用户之后改过卷），
        # 直接拿去 grade 时引擎会静默换成该链默认纸 —— 缩略图和面板从此对不上。
        # 先看旁挂记忆能不能补一张本链的纸，补不到再落该链默认纸。这两种情况都不算
        # "用户在这条链上选过"，所以内嵌缩略图（旧链渲的）也要丢掉重烘。
        scope = None
        if self.engine is not None and film_profile not in ("none", "", None):
            try:
                scope = self.engine.paper_scope(film_profile)
            except Exception:                          # noqa: BLE001
                scope = None
            if scope not in ("native", "ffchain"):
                scope = None
        explicit_out = explicit_paper
        if scope and paper_profile not in ("", "none", None):
            paper_ok = True
            try:
                paper_ok = bool(self.engine.paper_available(film_profile, paper_profile))
            except Exception:                          # noqa: BLE001
                paper_ok = True
            if not paper_ok:
                restored = paper_by_scope.get(scope)
                if restored is not None and self.engine.paper_available(film_profile, restored):
                    paper_profile = restored
                    explicit_out = True
                else:
                    paper_profile = chain_paper or "none"
                    explicit_out = False
                render_paper = paper_profile
                implicit_paper = True
        if (not implicit_paper
                and film_profile not in ("none", "", None)
                and paper_profile in ("", "none", None)
                and not explicit_paper
                and chain_paper not in ("none", "", None)):
            try:
                is_pos = bool(self.engine and self.engine.is_positive_film(film_profile))
            except Exception:
                is_pos = False
            if not is_pos:
                render_paper = chain_paper
                implicit_paper = True
                explicit_out = False

        if sdc and not implicit_paper:
            sdc_th = sdc.get("thumbnail_rgb")
            if session_cache_manager.is_valid_thumbnail(sdc_th):
                th_rgb = sdc_th
                graded_th = sdc_th
            elif raw_th is not None:
                if self.thumb_controller and (film_profile != "none" or render_paper != "none" or (params and abs(params.get("exposure_ev", 0.0)) > 0.001)):
                    try:
                        graded_th = self.thumb_controller.grade_thumbnail_fast(raw_th, film_profile, render_paper, params)
                    except Exception:
                        graded_th = raw_th
                    th_rgb = graded_th
                else:
                    th_rgb = raw_th
        elif implicit_paper and raw_th is not None and self.thumb_controller:
            try:
                graded_th = self.thumb_controller.grade_thumbnail_fast(raw_th, film_profile, render_paper, params)
            except Exception:
                graded_th = raw_th
            th_rgb = graded_th
        else:
            th_rgb = raw_th
            graded_th = None

        is_uncached = False
        crop_data = params.get("crop") if (params and isinstance(params, dict)) else None
        # base_ev is deliberately not restored from sessions or caches: it used to carry the
        # auto-metered value, pinning every restore to a stale exposure.  The sdc sidecar
        # stays authoritative (re-consulted at photo-load time); the metered suggestion lives
        # in auto_ev behind the auto-exposure button.
        b_ev = 0.0

        ext_p = os.path.splitext(p)[1].lower()
        is_raw_p = ext_p in RAW_EXTENSIONS

        if cache_item is not None:
            photo_entry = {
                "id": pid,
                "filename": os.path.basename(p),
                "path": p,
                "width": cache_item.get("width", 0),
                "height": cache_item.get("height", 0),
                "thumbnail_rgb": th_rgb,
                "raw_thumbnail_rgb": np.copy(raw_th) if raw_th is not None else None,
                "raw_thumbnail": np.copy(raw_th) if raw_th is not None else None,
                "graded_thumbnail_rgb": graded_th,
                "float_img": cache_item.get("float_img") if is_target else None,
                "exif": cache_item.get("exif", {}),
                "auto_ev": cache_item.get("auto_ev", b_ev),
                "base_ev": b_ev,
                "is_raw": cache_item.get("is_raw", is_raw_p),
                "film_profile": film_profile,
                "paper_profile": paper_profile,
                # 相纸是不是用户在这条链上选的，以及另一条链的选择。缺了前者，
                # 空串/默认纸会被读成"用户选过"；缺了后者，跨链往返会丢选择。
                "paper_explicit": bool(explicit_out),
                "paper_by_scope": dict(paper_by_scope),
                "print_mode": print_mode,
                "params": params,
                "crop": crop_data,
                "rating": rating,
                "is_dirty": is_dirty,
                "is_edited": is_edited,
            }
        else:
            dw, dh = session_cache_manager.get_image_dimensions(p)
            if dw > 0 and dh > 0 and session_cache_manager.is_valid_thumbnail(raw_th):
                try:
                    session_cache_manager.save_photo_cache(p, {
                        "path": p,
                        "filename": os.path.basename(p),
                        "width": dw,
                        "height": dh,
                        "thumbnail_rgb": th_rgb if th_rgb is not None else raw_th,
                        "raw_thumbnail_rgb": raw_th,
                        "graded_thumbnail_rgb": graded_th,
                        "exif": {"filename": os.path.basename(p), "path": p, "width": dw, "height": dh},
                        "auto_ev": b_ev,
                        "base_ev": b_ev,
                        "is_raw": is_raw_p
                    }, save_preview=False)
                    is_uncached = False
                except Exception:
                    is_uncached = True
            else:
                is_uncached = True

            photo_entry = {
                "id": pid,
                "filename": os.path.basename(p),
                "path": p,
                "width": dw,
                "height": dh,
                "thumbnail_rgb": th_rgb,
                "raw_thumbnail_rgb": np.copy(raw_th) if raw_th is not None else None,
                "raw_thumbnail": np.copy(raw_th) if raw_th is not None else None,
                "graded_thumbnail_rgb": graded_th,
                "float_img": None,
                "exif": {"filename": os.path.basename(p), "path": p, "width": dw, "height": dh},
                "auto_ev": b_ev,
                "base_ev": b_ev,
                "is_raw": is_raw_p,
                "film_profile": film_profile,
                "paper_profile": paper_profile,
                # 相纸是不是用户在这条链上选的，以及另一条链的选择。缺了前者，
                # 空串/默认纸会被读成"用户选过"；缺了后者，跨链往返会丢选择。
                "paper_explicit": bool(explicit_out),
                "paper_by_scope": dict(paper_by_scope),
                "print_mode": print_mode,
                "params": params,
                "crop": crop_data,
                "rating": rating,
                "is_dirty": is_dirty,
                "is_edited": is_edited,
            }

        return photo_entry, is_uncached

    def run(self):
        try:
            if not self.existing_files:
                self.restoreFinished.emit([], [], None)
                return

            total_files = len(self.existing_files)
            all_photos = [None] * total_files
            uncached_slots = []

            target_path = self.active_file if (self.active_file and self.active_file in self.existing_files) else self.existing_files[0]
            try:
                target_idx = self.existing_files.index(target_path)
            except ValueError:
                target_idx = 0
                target_path = self.existing_files[0]

            # 1. First photo priority (<30ms)
            target_entry, is_target_uncached = self._process_single_photo(target_idx, target_path, is_target=True)
            all_photos[target_idx] = target_entry
            if is_target_uncached:
                uncached_slots.append((target_idx, target_path))

            if not self._stopped:
                self.activePhotoReady.emit(target_idx, target_entry)
                self.photoReady.emit(target_idx, target_entry)
                self.progress.emit(1, total_files)

            if total_files == 1:
                if not self._stopped:
                    self.restoreFinished.emit(all_photos, uncached_slots, target_path)
                return

            # 2. Concurrent processing for remaining photos
            import concurrent.futures
            remaining_indices = [i for i in range(total_files) if i != target_idx]
            max_workers = min(8, max(2, (os.cpu_count() or 4)))
            completed_count = 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(self._process_single_photo, idx, self.existing_files[idx], False): idx
                    for idx in remaining_indices
                }
                for future in concurrent.futures.as_completed(future_map):
                    if self._stopped:
                        break
                    idx = future_map[future]
                    try:
                        entry, is_uncached = future.result()
                        all_photos[idx] = entry
                        if is_uncached:
                            uncached_slots.append((idx, self.existing_files[idx]))
                        if not self._stopped:
                            self.photoReady.emit(idx, entry)
                    except Exception as e:
                        logger.warning(f"Error restoring photo at idx {idx}: {e}")
                    completed_count += 1
                    if completed_count % 5 == 0 or completed_count == total_files:
                        self.progress.emit(completed_count, total_files)

            if not self._stopped:
                uncached_slots.sort(key=lambda x: x[0])
                self.restoreFinished.emit(all_photos, uncached_slots, target_path)
        except Exception as e:
            logger.error(f"SessionRestoreWorker failed: {e}", exc_info=True)
            self.failed.emit(str(e))

    def run_sync(self):
        self.run()


def _thumb_clip_path(w: float, h: float, top_r: float = 5.0) -> QPainterPath:
    """卡片缩略图的裁切形状：只圆上面两角，下缘直角接文字条。

    同心的前提是 `R内 = R外 - 内缩`：外框 6、边框 1 → 内圈只能 5，而且缩略图必须贴到
    边框内侧（内缩 1）。早先"6px 白边 + 内圆角 4"是 6-7 = -1，数学上永远落不到同一点。
    """
    r = max(0.0, min(top_r, w / 2.0, h / 2.0))
    path = QPainterPath()
    path.moveTo(0.0, h)
    path.lineTo(0.0, r)
    path.quadTo(0.0, 0.0, r, 0.0)
    path.lineTo(w - r, 0.0)
    path.quadTo(w, 0.0, w, r)
    path.lineTo(w, h)
    path.closeSubpath()
    return path


class _ElidedNameLabel(QLabel):
    """两行折行 + 末行省略号。

    QLabel 的 wordWrap 只会硬裁：第二行放不下的字直接消失，看不出"还有下文"。
    这里自己按宽度折行，超出行数就把最后一行 elide 成 "...".
    字体由 `setFont()` 决定（QSS 的 font-size 不进 QFont，量出来的宽度会和实际绘制对不上）。
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._source_text = text or ""
        self._text_color = None
        self.setWordWrap(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTextFormat(Qt.TextFormat.PlainText)

    def setText(self, text):  # noqa: N802 - Qt 命名
        self._source_text = text or ""
        self.update()

    def text(self):
        return self._source_text

    def set_text_color(self, color):
        self._text_color = QColor(color) if color is not None else None
        self.update()

    def _break_at(self, fm, text, start, width):
        """返回从 start 起能塞进 width 的字符边界（优先在空格处断，CJK 逐字可断）。"""
        n = len(text)
        last_space = -1
        acc = 0.0
        i = start
        while i < n:
            w = fm.horizontalAdvance(text[i])
            if acc + w > width and i > start:
                return last_space if last_space > start else i
            acc += w
            if text[i] == " ":
                last_space = i
            i += 1
        return n

    def _compute_lines(self):
        """返回 (lines, line_h, rect)。放不下就省略号结尾，不硬裁。"""
        rect = self.contentsRect()
        fm = QFontMetrics(self.font())
        line_h = max(1, fm.lineSpacing())
        max_lines = max(1, int(rect.height() // line_h))
        text = self._source_text.strip()
        if not text:
            return [], line_h, rect

        lines = []
        pos = 0
        n = len(text)
        while pos < n and len(lines) < max_lines:
            cut = max(pos + 1, self._break_at(fm, text, pos, rect.width()))
            line_start = pos
            line = text[line_start:cut].strip()
            next_pos = cut
            while next_pos < n and text[next_pos] == " ":
                next_pos += 1
            if next_pos < n and len(lines) == max_lines - 1:
                # 还有剩：最后一行放不下就整体省略，别直接截断
                lines.append(fm.elidedText(text[line_start:], Qt.TextElideMode.ElideRight, rect.width()))
                pos = n
                break
            lines.append(line)
            pos = next_pos
        return lines, line_h, rect

    def paintEvent(self, event):  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        color = self._text_color
        if color is None:
            color = self.palette().color(self.foregroundRole())
        painter.setPen(color)
        painter.setFont(self.font())

        lines, line_h, rect = self._compute_lines()
        self._last_lines = list(lines)
        if not lines:
            return
        y = rect.top() + (rect.height() - len(lines) * line_h) / 2.0
        for line in lines:
            painter.drawText(QRectF(rect.left(), y, rect.width(), line_h),
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, line)
            y += line_h


class FilmGridCard(QFrame):
    """Item 6: Large 2:3 aspect ratio photographic card (114x171px):
    - 2:3 ratio vertical photo preview of the active image
    - Clean film/paper name overlaid at the bottom
    - Active amber glowing border
    - Hover tooltip explaining optical characteristics
    """
    selected = Signal(str)
    exportRequested = Signal(str)
    sortRequested = Signal(str, str)

    SWATCH_COLORS = {
        "kodak_portra_400": ("#eab308", "#f97316"),
        "kodak_portra_160": ("#fde047", "#f59e0b"),
        "kodak_gold_200": ("#f59e0b", "#d97706"),
        "kodak_ektar_100": ("#ef4444", "#b91c1c"),
        "kodak_ultramax_400": ("#3b82f6", "#eab308"),
        "kodak_vision3_500t": ("#06b6d4", "#e11d48"),
        "kodak_vision3_250d": ("#0284c7", "#f59e0b"),
        "kodak_vision3_50d": ("#0ea5e9", "#10b981"),
        "fujifilm_pro_400h": ("#10b981", "#ec4899"),
        "fujifilm_c200": ("#22c55e", "#f59e0b"),
        "fujifilm_xtra_400": ("#059669", "#dc2626"),
        "fujifilm_velvia_100": ("#84cc16", "#7c3aed"),
        "fujifilm_provia_100f": ("#38bdf8", "#64748b"),
        "kodak_ektachrome_100": ("#2563eb", "#38bdf8"),
        "kodak_2383": ("#d97706", "#b45309"),
        "kodak_portra_endura": ("#f59e0b", "#d97706"),
        "fujifilm_crystal_archive_typeii": ("#10b981", "#059669"),
        "kodak_ultra_endura": ("#ef4444", "#b91c1c"),
    }

    CLEAN_NAMES = {
        "kodak_portra_400": "Portra 400",
        "kodak_portra_160": "Portra 160",
        "kodak_gold_200": "Gold 200",
        "kodak_ektar_100": "Ektar 100",
        "kodak_ultramax_400": "UltraMax 400",
        "kodak_vision3_500t": "Vision3 500T",
        "kodak_vision3_250d": "Vision3 250D",
        "kodak_vision3_50d": "Vision3 50D",
        "fujifilm_pro_400h": "Pro 400H",
        "fujifilm_c200": "Fuji C200",
        "fujifilm_xtra_400": "Superia 400",
        "fujifilm_velvia_100": "Velvia 100",
        "fujifilm_provia_100f": "Provia 100F",
        "kodak_ektachrome_100": "Ektachrome 100",
        "kodak_2383": "2383 电影放映",
        "kodak_portra_endura": "Portra 展览相纸",
        "fujifilm_crystal_archive_typeii": "水晶二代相纸",
        "kodak_ultra_endura": "Ultra 高反差相纸",
        "fujifilm_eterna_cp": "Eterna CP 发行",
        "fujifilm_eterna_cp_3521xd": "Eterna CP 3521XD",
        "kodak_exr_5386": "Vision 5386",
        "agfa_cp30": "Agfa CP30"
    }

    # 卡片几何预算。QSS 的 1px 边框会让 contentsRect 比 widget 小 2px，任何按 widget
    # 宽度直接算内件的写法都会偏 2px（旧版缩略图左边 7 右边 5 就是这么来的）。
    BORDER = 2
    NAME_GAP = 2
    BOTTOM_PAD = 2
    # 名称字号的设计值（9.5px），实际按首选项的字号比例缩放
    NAME_PT = 7.1

    def __init__(self, stock_info, is_active=False, is_custom=False, stock_type="film", parent=None):
        super().__init__(parent)
        self.setObjectName("filmGridCard")
        self.stock_info = stock_info
        self.stock_id = stock_info["id"]
        self.stock_name = stock_info["name"]
        self.stock_type = stock_type
        self.is_active = is_active
        self.is_custom = is_custom
        self.is_filter_matched = True
        self._thumb_pixmap = None

        # Compact photographic card dynamically resizable
        self._apply_name_font()
        self.card_w = 110
        self.card_h = self._height_for_width(110)
        self.setFixedSize(self.card_w, self.card_h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Full official brand name
        self.clean_name = self.stock_name

        self.original_tooltip = f"{self.stock_name}\n{stock_info.get('badge', '')}\n{stock_info.get('desc', '')}".strip()

        self._init_ui()
        theme_manager.themeChanged.connect(lambda _: self._update_appearance())
        # 名称是自绘的（QFont 不走 QSS 的 font-size 缩放），字号变了得自己跟着改高度
        ui_scale.bus.changed.connect(self._on_ui_scale_changed)

    def setToolTip(self, tip: str):
        super().setToolTip(tip)
        if hasattr(self, "lbl_thumb") and self.lbl_thumb:
            self.lbl_thumb.setToolTip(tip)
        if hasattr(self, "lbl_name") and self.lbl_name:
            self.lbl_name.setToolTip(tip)

    def set_card_tooltip(self, tip: str):
        self.setToolTip(tip)

    @classmethod
    def _thumb_size_for_width(cls, card_w: int):
        tw = max(10, int(card_w) - cls.BORDER)
        th = max(8, int(round(tw * (2.0 / 3.0))))
        return tw, th

    def _height_for_width(self, card_w: int) -> int:
        _, th = self._thumb_size_for_width(card_w)
        return th + self.BORDER + self.NAME_GAP + self.name_h + self.BOTTOM_PAD

    def _apply_name_font(self):
        """名称字号 = 设计值 × 首选项比例；两行高度跟着字号走，别写死 28。"""
        font = QFont(self.font())
        font.setPointSizeF(self.NAME_PT * ui_scale.current_scale())
        font.setWeight(QFont.Weight.Medium)
        self.name_font = font
        fm = QFontMetrics(font)
        self.name_line_h = max(1, fm.lineSpacing())
        self.name_h = self.name_line_h * 2 + 4
        if hasattr(self, "lbl_name") and self.lbl_name:
            self.lbl_name.setFont(font)
            self.lbl_name.setFixedHeight(self.name_h)

    def _on_ui_scale_changed(self):
        self._apply_name_font()
        tw, th = self._thumb_size_for_width(self.card_w)
        self.card_h = self._height_for_width(self.card_w)
        self.setFixedSize(self.card_w, self.card_h)
        self.lbl_thumb.setFixedSize(tw, th)
        if self._thumb_pixmap and not self._thumb_pixmap.isNull():
            self.set_thumbnail(self._thumb_pixmap)
        else:
            self._draw_color_pill()
        self.update()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        # 缩略图贴到边框内侧（方案 C：满幅，只圆上两角），间距只留文字条那 2px + 4px 下边距
        layout.setContentsMargins(0, 0, 0, self.BOTTOM_PAD)
        layout.setSpacing(self.NAME_GAP)

        # Image preview: 满幅贴边，圆角由 pixmap 自己裁（外 6 内 5，同心）
        self.lbl_thumb = QLabel()
        self.lbl_thumb.setObjectName("filmCardThumb")
        self.lbl_thumb.setFixedSize(*self._thumb_size_for_width(self.card_w))
        self.lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_thumb.setStyleSheet("QLabel#filmCardThumb { background: transparent; border: none; }")
        layout.addWidget(self.lbl_thumb, 0, Qt.AlignmentFlag.AlignHCenter)

        # Bottom label container（自己折行 + 省略号，见 _ElidedNameLabel）
        self.lbl_name = _ElidedNameLabel(self.clean_name)
        self.lbl_name.setObjectName("filmCardName")
        self.lbl_name.setFixedHeight(self.name_h)
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_name.setFont(self.name_font)
        layout.addWidget(self.lbl_name)

        self._update_appearance()
        self.set_card_tooltip(self.original_tooltip)

    def update_card_dimensions(self, new_w):
        new_w = max(96, min(220, int(new_w)))
        thumb_w, thumb_h = self._thumb_size_for_width(new_w)
        new_h = self._height_for_width(new_w)
        if new_w != self.card_w or new_h != self.card_h:
            self.card_w = new_w
            self.card_h = new_h
            self.setFixedSize(new_w, new_h)
            self.lbl_thumb.setFixedSize(thumb_w, thumb_h)
            if self._thumb_pixmap and not self._thumb_pixmap.isNull():
                self.set_thumbnail(self._thumb_pixmap)
            else:
                self._draw_color_pill()

    @property
    def thumbnail_pixmap(self):
        return self._thumb_pixmap

    def set_thumbnail(self, pixmap):
        self._thumb_pixmap = pixmap
        tw, th = self.lbl_thumb.width(), self.lbl_thumb.height()
        if pixmap and not pixmap.isNull():
            # 满幅：按覆盖方式放大后居中裁切，不留 letterbox 白条
            scaled = pixmap.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                   Qt.TransformationMode.SmoothTransformation)
            cx = max(0, (scaled.width() - tw) // 2)
            cy = max(0, (scaled.height() - th) // 2)
            cropped = scaled.copy(cx, cy, tw, th)
            out_pm = QPixmap(tw, th)
            out_pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(out_pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.setClipPath(_thumb_clip_path(tw, th, top_r=5.0))
            p.drawPixmap(0, 0, cropped)
            p.end()
            self.lbl_thumb.setPixmap(out_pm)
            p.end()
            self.lbl_thumb.setPixmap(out_pm)
        else:
            self._draw_color_pill()

    def _draw_color_pill(self):
        tw, th = max(10, self.lbl_thumb.width()), max(10, self.lbl_thumb.height())
        pm = QPixmap(tw, th)
        pm.fill(QColor("#131418"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        s_id = str(self.stock_id).lower()

        if s_id in ("none", ""):
            # 1. Digital RAW Bypass / Direct Scan
            pad = 8.0
            card_rect = QRectF(pad, pad, tw - pad * 2, th - pad * 2)
            p.setPen(QPen(QColor("#2d313e"), 1.0))
            p.setBrush(QBrush(QColor("#181920")))
            p.drawRoundedRect(card_rect, 4.0, 4.0)

            # Silicon sensor grid icon
            gx = tw / 2.0 - 12.0
            gy = th / 2.0 - 13.0
            gw, gh = 24.0, 15.0
            p.setPen(QPen(QColor("#475569"), 0.8))
            p.setBrush(QBrush(QColor("#1e293b")))
            p.drawRect(QRectF(gx, gy, gw, gh))
            p.setPen(QPen(QColor("#38bdf8"), 0.8, Qt.PenStyle.DashLine))
            p.drawLine(gx + gw * 0.33, gy, gx + gw * 0.33, gy + gh)
            p.drawLine(gx + gw * 0.66, gy, gx + gw * 0.66, gy + gh)
            p.drawLine(gx, gy + gh * 0.5, gx + gw, gy + gh * 0.5)

            p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            p.setPen(QColor("#94a3b8"))
            p.drawText(QRectF(0, gy + gh + 3.0, tw, 12.0), Qt.AlignmentFlag.AlignCenter, "RAW BYPASS" if self.stock_type == "film" else "DIRECT SCAN")

        elif self.stock_type == "paper":
            # 2. Darkroom Enlarging Paper Sheet on Easel
            pw = tw * 0.68
            ph = th * 0.76
            px = (tw - pw) / 2.0
            py = (th - ph) / 2.0

            # Easel metal base
            p.setPen(QPen(QColor("#282b36"), 1.0))
            p.setBrush(QBrush(QColor("#1c1e25")))
            p.drawRoundedRect(QRectF(px - 3, py - 3, pw + 6, ph + 6), 3.0, 3.0)

            # Photo paper white base
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor("#f1f5f9")))
            p.drawRect(QRectF(px, py, pw, ph))

            # Paper emulsion tone swatch
            emul_grad = QLinearGradient(px + 4, py + 4, px + pw - 4, py + ph - 4)
            if "2383" in s_id or "2393" in s_id:
                emul_grad.setColorAt(0.0, QColor("#1e293b"))
                emul_grad.setColorAt(1.0, QColor("#020617"))
            elif "premier" in s_id or "endura" in s_id:
                emul_grad.setColorAt(0.0, QColor("#475569"))
                emul_grad.setColorAt(0.5, QColor("#d97706"))
                emul_grad.setColorAt(1.0, QColor("#1e293b"))
            elif "crystal" in s_id:
                emul_grad.setColorAt(0.0, QColor("#334155"))
                emul_grad.setColorAt(0.5, QColor("#059669"))
                emul_grad.setColorAt(1.0, QColor("#0f172a"))
            else:
                emul_grad.setColorAt(0.0, QColor("#475569"))
                emul_grad.setColorAt(1.0, QColor("#1e293b"))

            p.setBrush(emul_grad)
            p.drawRect(QRectF(px + 3, py + 3, pw - 6, ph - 6))

            # Paper typography
            p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            p.setPen(QColor("#f8fafc"))
            p.drawText(QRectF(px, py + ph - 13, pw, 10), Qt.AlignmentFlag.AlignCenter, "AGX PAPER")

        else:
            # 3. Authentic 35mm Metal Film Canister (拟真暗盒)
            can_w = 32.0
            can_h = 48.0
            cx = (tw - can_w) / 2.0 - 2.0
            cy = (th - can_h) / 2.0 + 1.0

            # 3.1 Top spool hub (金属轴心)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#334155"))
            p.drawRoundedRect(QRectF(cx + (can_w - 8) / 2, cy - 4.5, 8, 5), 1.5, 1.5)

            # 3.2 Top & Bottom metal end caps
            cap_grad = QLinearGradient(cx - 2, cy, cx + can_w + 2, cy)
            cap_grad.setColorAt(0.0, QColor("#475569"))
            cap_grad.setColorAt(0.35, QColor("#94a3b8"))
            cap_grad.setColorAt(0.75, QColor("#475569"))
            cap_grad.setColorAt(1.0, QColor("#1e293b"))
            p.setBrush(cap_grad)
            p.drawRoundedRect(QRectF(cx - 2, cy - 0.5, can_w + 4, 3.5), 1.5, 1.5)
            p.drawRoundedRect(QRectF(cx - 2, cy + can_h - 3.0, can_w + 4, 3.5), 1.5, 1.5)

            # 3.3 Main canister cylinder body
            body_grad = QLinearGradient(cx, cy, cx + can_w, cy)
            body_grad.setColorAt(0.0, QColor("#1e222a"))
            body_grad.setColorAt(0.25, QColor("#333846"))
            body_grad.setColorAt(0.65, QColor("#181a20"))
            body_grad.setColorAt(1.0, QColor("#0f1115"))
            p.setBrush(body_grad)
            p.drawRect(QRectF(cx, cy + 3, can_w, can_h - 6))

            # 3.4 Film leader tongue protruding on right
            p.setBrush(QColor("#0a0d14"))
            p.drawRoundedRect(QRectF(cx + can_w - 0.5, cy + 18, 9, 13), 2, 2)
            p.setPen(QPen(QColor("#2d3342"), 0.8))
            p.drawLine(cx + can_w + 2, cy + 19, cx + can_w + 2, cy + 30)
            p.setPen(Qt.PenStyle.NoPen)

            # 3.5 Brand Identity Label Band
            band_y = cy + 12
            band_h = 24

            c_main = QColor("#f59e0b")
            c_sub = QColor("#dc2626")
            text_color = QColor("#ffffff")
            short_code = "FILM"

            if "gold" in s_id:
                c_main = QColor("#d97706")
                c_sub = QColor("#b91c1c")
                short_code = "200"
            elif "portra_400" in s_id:
                c_main = QColor("#475569")
                c_sub = QColor("#cbd5e1")
                short_code = "400"
            elif "portra_160" in s_id:
                c_main = QColor("#334155")
                c_sub = QColor("#94a3b8")
                short_code = "160"
            elif "portra_800" in s_id:
                c_main = QColor("#1e293b")
                c_sub = QColor("#e2e8f0")
                short_code = "800"
            elif "ektar" in s_id:
                c_main = QColor("#991b1b")
                c_sub = QColor("#f59e0b")
                short_code = "100"
            elif "ultramax" in s_id:
                c_main = QColor("#1d4ed8")
                c_sub = QColor("#facc15")
                short_code = "400"
            elif "vision" in s_id:
                c_main = QColor("#09090b")
                c_sub = QColor("#eab308")
                short_code = "500T" if "500t" in s_id else ("250D" if "250d" in s_id else "50D")
            elif "c200" in s_id:
                c_main = QColor("#047857")
                c_sub = QColor("#dc2626")
                short_code = "C200"
            elif "400h" in s_id:
                c_main = QColor("#0f766e")
                c_sub = QColor("#ec4899")
                short_code = "400H"
            elif "xtra" in s_id or "superia" in s_id:
                c_main = QColor("#065f46")
                c_sub = QColor("#ef4444")
                short_code = "X-TRA"
            elif "velvia" in s_id:
                c_main = QColor("#4c1d95")
                c_sub = QColor("#059669")
                short_code = "50" if "50" in s_id else "100"
            elif "provia" in s_id:
                c_main = QColor("#1e3a8a")
                c_sub = QColor("#38bdf8")
                short_code = "100F"
            elif "cinestill" in s_id:
                c_main = QColor("#090d16")
                c_sub = QColor("#f43f5e")
                short_code = "800T" if "800" in s_id else "400D"
            elif any(bw in s_id for bw in ("hp5", "delta", "tri_x", "tmax", "acros", "double_x", "fp4")):
                c_main = QColor("#09090b")
                c_sub = QColor("#f8fafc")
                short_code = "B&W"
            elif "kodachrome" in s_id:
                c_main = QColor("#b45309")
                c_sub = QColor("#dc2626")
                short_code = "K-64" if "64" in s_id else "K-25"
            elif "agfa" in s_id:
                c_main = QColor("#c2410c")
                c_sub = QColor("#ea580c")
                short_code = "AGFA"

            p.fillRect(QRectF(cx, band_y, can_w, band_h), c_main)
            p.fillRect(QRectF(cx, band_y, can_w, 2.5), c_sub)

            p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            p.setPen(text_color)
            p.drawText(QRectF(cx, band_y + 3, can_w, band_h - 3), Qt.AlignmentFlag.AlignCenter, short_code)

        # Subtle card thumbnail outer frame（跟着裁切形状走，不再画四角全圆的框）
        clip_path = _thumb_clip_path(tw, th, top_r=5.0)
        p.setPen(QPen(QColor("#222530"), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(clip_path)
        p.end()

        # 圆角裁切：上两角 5、下缘直角，与外框同心
        masked = QPixmap(tw, th)
        masked.fill(Qt.GlobalColor.transparent)
        mp = QPainter(masked)
        mp.setRenderHint(QPainter.RenderHint.Antialiasing)
        mp.setClipPath(clip_path)
        mp.drawPixmap(0, 0, pm)
        mp.end()
        self.lbl_thumb.setPixmap(masked)


    def set_active(self, is_active):
        if self.is_active == is_active:
            return
        self.is_active = is_active
        self._update_appearance()

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._update_appearance()

    def _update_appearance(self):
        is_dark = theme_manager.is_dark()
        tooltip_bubble_style = theme_manager.widget_tooltip_stylesheet()
        thumb_bg = "transparent"

        if hasattr(self, "lbl_thumb") and self.lbl_thumb:
            self.lbl_thumb.setStyleSheet("QLabel#filmCardThumb { border: none; background: transparent; }")

        if not self.isEnabled():
            dis_bg = "#101114" if is_dark else "#f1f5f9"
            dis_border = "#1c1d22" if is_dark else "#e2e8f0"
            dis_name = "#404452" if is_dark else "#94a3b8"
            self.setStyleSheet(f"""
                QFrame#filmGridCard {{
                    background: {dis_bg};
                    border: 1px solid {dis_border};
                    border-radius: 6px;
                }}
                QLabel#filmCardThumb {{ border: none; background: transparent; }}
                QLabel#filmCardName {{ border: none; background: transparent; color: {dis_name}; }}
                {tooltip_bubble_style}
            """)
            self.lbl_name.setStyleSheet(f"QLabel#filmCardName {{ color: {dis_name}; border: none; background: transparent; }}")
            self.lbl_name.set_text_color(dis_name)
            return
        if self.is_active:
            act_bg = "rgba(245, 158, 11, 0.08)" if is_dark else "rgba(245, 158, 11, 0.12)"
            self.setStyleSheet(f"""
                QFrame#filmGridCard {{
                    background: {act_bg};
                    border: 1px solid #f59e0b;
                    border-radius: 6px;
                }}
                QLabel#filmCardThumb {{ border: none; background: transparent; }}
                QLabel#filmCardName {{ border: none; background: transparent; color: #f59e0b; }}
                {tooltip_bubble_style}
            """)
            self.lbl_name.setStyleSheet("QLabel#filmCardName { color: #f59e0b; border: none; background: transparent; }")
            self.lbl_name.set_text_color("#f59e0b")
        else:
            idle_bg = "#15161c" if is_dark else "#ffffff"
            idle_border = "#282a36" if is_dark else "#d8dce6"
            hover_bg = "#1b1d24" if is_dark else "#f1f4f9"
            hover_border = "#4b5168" if is_dark else "#94a3b8"
            name_color = "#cbd5e1" if is_dark else "#1e293b"
            self.setStyleSheet(f"""
                QFrame#filmGridCard {{
                    background: {idle_bg};
                    border: 1px solid {idle_border};
                    border-radius: 6px;
                }}
                QFrame#filmGridCard:hover {{
                    border-color: {hover_border};
                    background: {hover_bg};
                }}
                QLabel#filmCardThumb {{ border: none; background: transparent; }}
                QLabel#filmCardName {{ border: none; background: transparent; color: {name_color}; }}
                {tooltip_bubble_style}
            """)
            self.lbl_name.setStyleSheet(f"QLabel#filmCardName {{ color: {name_color}; border: none; background: transparent; }}")
            self.lbl_name.set_text_color(name_color)

    def highlight_now(self) -> None:
        """把选中框立刻挪到这张卡上并同步重绘。

        `selected` 是直连信号，接收方 `on_switch_film` 会在同一个调用里同步烘完整条 LUT
        （原生→原生首次命中实测 690 ms）。排队重绘要等它返回，视觉上就是"点完要搁一拍
        才亮"。这里先切样式、再 repaint() 同步画一帧，把高亮挤在阻塞之前。
        """
        group = getattr(self, "sibling_cards", None)
        targets = list(group.items()) if group else [(self.stock_id, self)]
        repaint = []
        for sid, card in targets:
            before = card.is_active
            card.set_active(sid == self.stock_id)
            if card.is_active != before:
                repaint.append(card)
        for card in repaint:
            card.repaint()

    def mousePressEvent(self, event):
        if not self.isEnabled():
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.highlight_now()
            self.selected.emit(self.stock_id)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            menu.setStyleSheet(get_darkroom_menu_style())
            act_apply = menu.addAction("应用预设")
            act_apply.triggered.connect(lambda: (self.highlight_now(), self.selected.emit(self.stock_id)))

            menu.addSeparator()

            # Submenu: 整理排列 (regardless of clicking on blank or card)
            sort_menu = menu.addMenu("整理排列")
            sort_menu.setStyleSheet(get_darkroom_menu_style())
            act_by_name = sort_menu.addAction("按名称排序 (A-Z)")
            act_by_name.triggered.connect(lambda: self.sortRequested.emit(self.stock_type, "name"))
            act_by_time = sort_menu.addAction("按添加时间排序 (最新在前)")
            act_by_time.triggered.connect(lambda: self.sortRequested.emit(self.stock_type, "time"))

            menu.exec(event.globalPosition().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)


class CompactExifWidget(QWidget):
    """Item 7: High-density compact EXIF property grid with pinned header and anti-jitter margins."""

    # Emitted whenever the rows (or their font) change height, so the pane holding this
    # widget can re-pin its maximum to "exactly tall enough to show every row".
    contentResized = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.key_labels = []
        self._init_ui()
        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.card = QFrame()
        self.card.setObjectName("exifCard")
        c_layout = QVBoxLayout(self.card)
        c_layout.setContentsMargins(8, 6, 4, 6)
        c_layout.setSpacing(4)

        # Fixed Header (Issue 4: Title pinned, never scrolls)
        self.lbl_title = QLabel("EXIF")
        c_layout.addWidget(self.lbl_title)

        # Scroll area for metadata rows only
        self.scroll_area = SmoothScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Content widget
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        grid = QGridLayout(content)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setContentsMargins(0, 2, 6, 2)
        grid.setColumnMinimumWidth(0, 68)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        self.labels = {}
        fields = [
            ("拍摄日期", "capture_time"),
            ("分辨率", "resolution"),
            ("大小", "file_size"),
            ("相机制造商", "camera_make"),
            ("相机型号", "camera_model"),
            ("相机序列号", "serial_number"),
            ("镜头制造商", "lens_make"),
            ("镜头型号", "lens_model"),
            ("白平衡", "white_balance"),
            ("ISO 速度", "iso"),
            ("光圈值", "aperture"),
            ("曝光时间", "shutter_speed"),
            ("曝光补偿", "exposure_bias"),
            ("曝光程序", "exposure_program"),
            ("测光模式", "metering_mode"),
            ("闪光灯", "flash"),
            ("焦距", "focal_length"),
            ("35mm 焦距", "focal_length_35mm"),
            ("作者", "artist"),
            ("版权", "copyright"),
            ("GPS 坐标", "gps_coords"),
            ("GPS 基准", "gps_datum"),
        ]

        for row, (label_text, key) in enumerate(fields):
            lbl_k = QLabel(label_text)
            lbl_k.setFixedWidth(68)
            self.key_labels.append(lbl_k)
            lbl_v = QLabel("-")
            lbl_v.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            lbl_v.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            lbl_v.setWordWrap(True)
            grid.addWidget(lbl_k, row, 0)
            grid.addWidget(lbl_v, row, 1)
            self.labels[key] = lbl_v

        self.scroll_area.setWidget(content)
        c_layout.addWidget(self.scroll_area, 1)

        layout.addWidget(self.card)

    def apply_theme(self, theme_name=None):
        is_dark = theme_manager.is_dark()
        card_bg = "#15161a" if is_dark else "#ffffff"
        card_border = "#262834" if is_dark else "#d8dce6"
        key_color = "#7b8092" if is_dark else "#64748b"
        # 渐隐带底色必须等于卡片色，否则 EXIF 的下边缘会出现一条和胶卷卡片不同色的带子
        if hasattr(self, "scroll_area") and self.scroll_area is not None:
            self.scroll_area.set_fade_base_color(card_bg, height=16)
        val_color = "#e2e8f0" if is_dark else "#1e293b"
        title_color = "#f59e0b" if is_dark else "#d97706"

        if hasattr(self, 'card'):
            self.card.setStyleSheet(f"""
                QFrame#exifCard {{
                    background: {card_bg};
                    border: 1px solid {card_border};
                    border-radius: 6px;
                }}
                QLabel {{
                    border: none;
                    background: transparent;
                }}
            """)
        if hasattr(self, 'lbl_title'):
            self.lbl_title.setStyleSheet(f"color: {title_color}; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;")
        if hasattr(self, 'key_labels'):
            for lbl in self.key_labels:
                lbl.setStyleSheet(f"color: {key_color}; font-size: 10px;")
        if hasattr(self, 'labels'):
            for lbl in self.labels.values():
                lbl.setStyleSheet(f"color: {val_color}; font-size: 10px; font-weight: 500;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Values wrap, so a width change changes how many rows are two lines tall. Re-report
        # after the layout settles; the receiver ignores values it already applied, which is
        # what stops this from looping.
        if not getattr(self, "_resize_report_pending", False):
            self._resize_report_pending = True
            QTimer.singleShot(0, self._report_content_height)

    def _report_content_height(self):
        self._resize_report_pending = False
        self.contentResized.emit()

    def natural_height(self) -> int:
        """Height that shows every row without scrolling and without dead space.

        Measured with the live viewport width because values wrap: ``iPhone XS Max back
        camera 4.25mm f/1.8`` takes two lines in a narrow column, so a constant row count
        would clip it.
        """
        content = self.scroll_area.widget()
        if content is None:
            return 0
        width = max(1, self.scroll_area.viewport().width())
        grid = content.layout()
        rows_h = -1
        if grid is not None and grid.hasHeightForWidth():
            rows_h = grid.heightForWidth(width)
        if rows_h < 0:
            rows_h = content.sizeHint().height()

        chrome = 0
        card_layout = self.card.layout()
        if card_layout is not None:
            margins = card_layout.contentsMargins()
            chrome += margins.top() + margins.bottom() + card_layout.spacing()
        chrome += self.lbl_title.sizeHint().height()
        outer = self.layout()
        if outer is not None:
            margins = outer.contentsMargins()
            chrome += margins.top() + margins.bottom()
        return int(rows_h + chrome + 2)

    def set_exif_data(self, data):
        for k, lbl in self.labels.items():
            raw_val = data.get(k, "-")
            val = str(raw_val).strip() if raw_val is not None and str(raw_val).strip() != "" else "-"
            lbl.setText(val)
            lbl.setToolTip("")
        self.contentResized.emit()


# Splitter grab indicator: 2 px of brand amber instead of a 1 px theme-grey hairline. The
# alpha stops keep the ends soft so a 2 px bar still reads as a hairline, not a hard rule.
_SPLIT_INDICATOR_PX = 2
_SPLIT_INDICATOR_STOPS = ((0.0, 0), (0.08, 190), (0.5, 255), (0.92, 190), (1.0, 0))


def _restore_splitter_sizes(splitter, saved) -> bool:
    """Apply stored pane sizes, but only when they still describe this splitter.

    The left column once held three panes; it holds two now, and the old restore demanded
    three, so the stored sizes were silently discarded and the default came back on every
    launch. Matching against the live pane count keeps the check honest as panes come and go.
    """
    if not saved or not splitter:
        return False
    if len(saved) != splitter.count():
        return False
    if any(s <= 0 for s in saved):
        return False
    splitter.setSizes([int(s) for s in saved])
    return True


class FadingSplitterHandle(QSplitterHandle):
    """Refined splitter handle with an amber grab bar that fades at both ends."""
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._hovered = False
        # 指示线贴哪一侧：'trailing' = 竖 handle 贴右、横 handle 贴下（默认，贴着后面那个
        # pane）；'leading' = 竖 handle 贴左、横 handle 贴上。由 FadingSplitter 按索引下发。
        self.indicator_side = "trailing"
        theme_manager.themeChanged.connect(lambda _: self.update())

    def enterEvent(self, ev):
        self._hovered = True
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hovered = False
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        is_dark = theme_manager.is_dark()
        w, h = self.width(), self.height()
        bg_col = QColor('#121316') if is_dark else QColor('#ebedf2')
        painter.fillRect(0, 0, w, h, bg_col)

        # The indicator is a hover affordance only: it stays invisible until the pointer is
        # actually on the handle, then runs almost edge to edge so it reads as a grab bar
        # rather than a stray dash in the middle of the pane gap.
        if not self._hovered:
            return

        # Brand amber, 2 px, hugging the pane it belongs to instead of floating in the middle
        # of the gutter: a vertical handle draws on its trailing edge (against the pane to the
        # right), a horizontal one on its bottom edge (against the pane below). The ends still
        # fade out so it reads as a hairline rather than a hard rule.
        accent = QColor(theme_manager.color("accent"))
        leading = (getattr(self, "indicator_side", "trailing") == "leading")
        if self.orientation() == Qt.Orientation.Vertical:
            # Horizontal bar separating vertically stacked panes
            y = 0 if leading else max(0, h - _SPLIT_INDICATOR_PX)
            start_x, end_x = 8, max(8 + 24, w - 8)
            grad = QLinearGradient(start_x, 0, end_x, 0)
            for stop, alpha in _SPLIT_INDICATOR_STOPS:
                c = QColor(accent)
                c.setAlpha(alpha)
                grad.setColorAt(stop, c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(start_x, y, end_x - start_x, _SPLIT_INDICATOR_PX)
        else:
            # Vertical bar separating horizontally arranged panes
            x = 0 if leading else max(0, w - _SPLIT_INDICATOR_PX)
            start_y, end_y = 8, max(8 + 32, h - 8)
            grad = QLinearGradient(0, start_y, 0, end_y)
            for stop, alpha in _SPLIT_INDICATOR_STOPS:
                c = QColor(accent)
                c.setAlpha(alpha)
                grad.setColorAt(stop, c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(x, start_y, _SPLIT_INDICATOR_PX, end_y - start_y)


class FadingSplitter(QSplitter):
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(4 if orientation == Qt.Orientation.Vertical else 5)
        self.setChildrenCollapsible(False)
        self.setStyleSheet("QSplitter { background: transparent; }")
        self._indicator_sides = {}

    def createHandle(self):
        return FadingSplitterHandle(self.orientation(), self)

    def set_indicator_side(self, index, side):
        """把第 index 条缝的指示线钉到某一侧（'leading' / 'trailing'）。"""
        self._indicator_sides[int(index)] = side
        self._sync_indicator_sides()

    def _sync_indicator_sides(self):
        if not self._indicator_sides:
            return
        n = self.count() - 1
        if n <= 0:
            return
        handles = [h for h in self.findChildren(FadingSplitterHandle) if h.isVisible()]
        if len(handles) < n:
            return
        # findChildren 里可能混有构造期残留的 handle，只取几何参与布局的那些
        if self.orientation() == Qt.Orientation.Horizontal:
            handles.sort(key=lambda h: h.x())
        else:
            handles.sort(key=lambda h: h.y())
        handles = handles[-n:]
        for i, h in enumerate(handles):
            side = self._indicator_sides.get(i, "trailing")
            if getattr(h, "indicator_side", "trailing") != side:
                h.indicator_side = side
                h.update()

    def showEvent(self, ev):  # noqa: N802 - Qt 命名
        super().showEvent(ev)
        self._sync_indicator_sides()

    def resizeEvent(self, ev):  # noqa: N802 - Qt 命名
        super().resizeEvent(ev)
        self._sync_indicator_sides()


class ViewportResizeFilter(QObject):
    """Filters Resize events on QScrollArea widgets to trigger immediate, synchronized grid reflows."""
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.callback = callback
        self._last_w = -1

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize:
            new_w = event.size().width()
            old_w = event.oldSize().width()
            # In a vertical wrapping card grid, column layout is solely determined by width.
            # Strictly ignore height-only resizes or identical width to prevent redundant reflows.
            if old_w != -1 and new_w == old_w:
                return super().eventFilter(watched, event)
            if new_w == self._last_w:
                return super().eventFilter(watched, event)
            self._last_w = new_w
            self.callback()
        return super().eventFilter(watched, event)



class CompareButton(QPushButton):
    """Compare button supporting:
    - Click (< 180ms): toggle split view mode (cycles 0 -> 1 -> 2 -> 0)
    - Long-press (>= 180ms): preview full original (Before) image, reverting to previous view on release.
    """
    longPressStarted = Signal()
    longPressFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._press_timer = QTimer(self)
        self._press_timer.setSingleShot(True)
        self._press_timer.setInterval(180)
        self._press_timer.timeout.connect(self._on_long_press_timeout)
        self._is_long_pressing = False

    def _on_long_press_timeout(self):
        self._is_long_pressing = True
        self.longPressStarted.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_long_pressing = False
            self._press_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._press_timer.isActive():
                self._press_timer.stop()
            if self._is_long_pressing:
                self._is_long_pressing = False
                self.longPressFinished.emit()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def focusOutEvent(self, event):
        if self._press_timer.isActive():
            self._press_timer.stop()
        if self._is_long_pressing:
            self._is_long_pressing = False
            self.longPressFinished.emit()
        super().focusOutEvent(event)

    def hideEvent(self, event):
        if self._press_timer.isActive():
            self._press_timer.stop()
        if self._is_long_pressing:
            self._is_long_pressing = False
            self.longPressFinished.emit()
        super().hideEvent(event)


class StatusSlotWidget(QWidget):
    """Dynamic cross-fading status slot that unifies permanent photo information (filename & zoom)
    and transient application notifications / import progress into a single compact area.

    When a status notification or import occurs, it smoothly fades out the photo title and fades in
    the notification / progress in-place ('直接顶掉标题'). When the notification expires, it smoothly
    cross-fades back to the photo title & zoom percentage ('循环显示').
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._stack = QStackedLayout(self)
        self._stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self._stack.setContentsMargins(0, 0, 0, 0)

        # Layer 0: Photo Info Layer (Default / Persistent)
        self.w_info = QWidget(self)
        self.w_info.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        l_info = QHBoxLayout(self.w_info)
        l_info.setContentsMargins(0, 0, 0, 0)
        l_info.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.lbl_photo_info = QLabel("")
        self.lbl_photo_info.setObjectName("lbl_photo_info")
        self.lbl_photo_info.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.lbl_photo_info.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.lbl_photo_info.setStyleSheet("QLabel#lbl_photo_info { color: #cbd5e1; font-size: 11px; font-weight: 500; }")
        l_info.addWidget(self.lbl_photo_info)

        # Layer 1: Transient Status / Notification / Import Layer
        self.w_status = QWidget(self)
        self.w_status.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        l_status = QHBoxLayout(self.w_status)
        l_status.setContentsMargins(0, 0, 0, 0)
        l_status.setSpacing(5)
        l_status.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.spinner = SpinnerWidget(size=13, auto_hide=True)
        self.spinner.setVisible(False)
        l_status.addWidget(self.spinner)

        self.status_label = QLabel("")
        self.status_label.setObjectName("status_label")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.status_label.setStyleSheet("QLabel#status_label { color: #94a3b8; font-size: 11px; }")
        l_status.addWidget(self.status_label)

        self.btn_cancel_import = QPushButton("✕ 取消导入")
        self.btn_cancel_import.setVisible(False)
        self.btn_cancel_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel_import.setToolTip("中断当前批量导入")
        l_status.addWidget(self.btn_cancel_import)
        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

        # Graphics Opacity Effects for smooth cross-fading
        self.eff_info = QGraphicsOpacityEffect(self.w_info)
        self.w_info.setGraphicsEffect(self.eff_info)
        self.eff_status = QGraphicsOpacityEffect(self.w_status)
        self.w_status.setGraphicsEffect(self.eff_status)

        self._stack.addWidget(self.w_info)
        self._stack.addWidget(self.w_status)

        self.eff_info.setOpacity(1.0)
        self.eff_status.setOpacity(0.0)
        self._mode = "info"
        self._apply_slot_hit_test("info")

    def _apply_slot_hit_test(self, mode: str) -> None:
        """Let only the visible layer take clicks.

        Both layers are laid out over the same rectangle (StackAll), so the hidden one still
        sits in the hit-test chain.  The cancel chip lives in the status layer: while the info
        layer was accepting mouse events, a real click on the chip landed on the photo title
        instead and nothing happened -- `QTest.mouseClick` sent the event straight to the
        button, which is why it looked like it worked.
        """
        if hasattr(self, "w_info") and self.w_info is not None:
            self.w_info.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                                    mode != "info")
        if hasattr(self, "w_status") and self.w_status is not None:
            self.w_status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                                      mode != "status")

    def apply_theme(self, theme_name=None):
        """Repaint the status texts and the cancel-import chip for the active theme.

        The chip used to carry a hard-coded dark red sheet, so in light mode it stayed a
        near-black pill on a near-white bar.  Callers that restyle `status_label` per message
        still win, because this only runs on construction and on a theme switch.
        """
        is_dark = theme_manager.is_dark()
        info_fg = "#cbd5e1" if is_dark else "#334155"
        status_fg = "#94a3b8" if is_dark else "#64748b"
        if hasattr(self, "lbl_photo_info") and self.lbl_photo_info:
            self.lbl_photo_info.setStyleSheet(
                f"QLabel#lbl_photo_info {{ color: {info_fg}; font-size: 11px; "
                f"font-weight: 500; }}")
        if hasattr(self, "status_label") and self.status_label:
            self.status_label.setStyleSheet(
                f"QLabel#status_label {{ color: {status_fg}; font-size: 11px; }}")
        if not hasattr(self, "btn_cancel_import") or not self.btn_cancel_import:
            return
        if is_dark:
            self.btn_cancel_import.setStyleSheet("""
                QPushButton {
                    background: #2a1515;
                    color: #ef4444;
                    border: 1px solid #7f1d1d;
                    border-radius: 3px;
                    padding: 1px 7px;
                    font-size: 11px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #3f1d1d;
                    color: #f87171;
                    border-color: #ef4444;
                }
                QPushButton:pressed {
                    background: #1f0d0d;
                }
            """)
        else:
            self.btn_cancel_import.setStyleSheet("""
                QPushButton {
                    background: #fdeceb;
                    color: #dc2626;
                    border: 1px solid #f3b8b3;
                    border-radius: 3px;
                    padding: 1px 7px;
                    font-size: 11px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #fbdbd7;
                    color: #b91c1c;
                    border-color: #dc2626;
                }
                QPushButton:pressed {
                    background: #f6c8c3;
                }
            """)
        self._anim_group = None

    def sizeHint(self):
        if self._mode == "status":
            sh = self.w_status.sizeHint()
            return QSize(sh.width(), 28)
        sh = self.w_info.sizeHint()
        return QSize(sh.width(), 28)

    def current_mode(self) -> str:
        return self._mode

    def transition_to(self, target_mode: str, duration_ms: int = 180, on_finished=None):
        # 命中测试跟着可见性走：否则隐藏那层会挡在按钮前面（见 _apply_slot_hit_test）。
        self._apply_slot_hit_test(target_mode)
        if target_mode == self._mode and self._anim_group is None:
            if callable(on_finished):
                on_finished()
            return

        self._mode = target_mode
        self.updateGeometry()

        if self._anim_group:
            try:
                self._anim_group.stop()
            except Exception:
                pass
            self._anim_group = None

        if os.environ.get("FAST_TEST") or duration_ms <= 0 or not self.isVisible():
            self.eff_info.setOpacity(1.0 if target_mode == "info" else 0.0)
            self.eff_status.setOpacity(1.0 if target_mode == "status" else 0.0)
            if callable(on_finished):
                on_finished()
            return

        group = QParallelAnimationGroup(self)

        a_info = QPropertyAnimation(self.eff_info, b"opacity", self)
        a_info.setDuration(duration_ms)
        a_info.setEasingCurve(QEasingCurve.Type.InOutCubic)
        a_info.setStartValue(self.eff_info.opacity())
        a_info.setEndValue(1.0 if target_mode == "info" else 0.0)

        a_status = QPropertyAnimation(self.eff_status, b"opacity", self)
        a_status.setDuration(duration_ms)
        a_status.setEasingCurve(QEasingCurve.Type.InOutCubic)
        a_status.setStartValue(self.eff_status.opacity())
        a_status.setEndValue(1.0 if target_mode == "status" else 0.0)

        group.addAnimation(a_info)
        group.addAnimation(a_status)

        def _on_finish():
            self._anim_group = None
            if callable(on_finished):
                on_finished()

        group.finished.connect(_on_finish)
        self._anim_group = group
        group.start()


class CenteredViewToolbar(QFrame):
    """View toolbar frame that keeps its center_widget (canvas rating) at the absolute geometric center,
    automatically resolving collision with status messages when the window is narrow.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.center_widget = None
        self.status_slot = None
        self.status_label = None
        self.right_widget = None
        self._rating_enabled = False

    def set_center_widget(self, w):
        self.center_widget = w
        if w:
            w.setParent(self)
            self._update_center()

    def set_right_widget(self, w):
        """Widget pinned to the right end of the bar (the scan/export capsule).  The centred
        rating stars have to yield to it as well as to the left-hand status text."""
        self.right_widget = w
        self._update_center()

    def set_status_slot(self, slot):
        self.status_slot = slot
        self._update_center()

    def set_status_label(self, lbl):
        self.status_label = lbl

    def set_rating_enabled(self, enabled: bool):
        self._rating_enabled = enabled
        self._update_center()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_center()

    def _update_center(self):
        if not self.center_widget:
            return

        w_hint = self.center_widget.sizeHint().width() if hasattr(self.center_widget, 'sizeHint') else -1
        cw = w_hint if w_hint > 0 else (self.center_widget.width() if self.center_widget.width() > 0 else 92)
        h_hint = self.center_widget.sizeHint().height() if hasattr(self.center_widget, 'sizeHint') else -1
        ch = h_hint if h_hint > 0 else (self.center_widget.height() if self.center_widget.height() > 0 else 22)

        x = (self.width() - cw) // 2
        y = (self.height() - ch) // 2
        self.center_widget.setGeometry(x, y, cw, ch)
        self.center_widget.raise_()

        if not self._rating_enabled:
            self.center_widget.setVisible(False)
            return

        # Check collision with left-side status slot
        should_hide = False
        if self.status_slot and not self.status_slot.isHidden():
            slot_r = self.status_slot.x() + self.status_slot.sizeHint().width()
            if slot_r > 0 and slot_r + 14 >= x:
                should_hide = True
        elif self.status_label and not self.status_label.isHidden():
            txt = self.status_label.text()
            if txt and txt.strip():
                fm = self.status_label.fontMetrics()
                txt_w = fm.horizontalAdvance(txt)
                lbl_x = self.status_label.x()
                status_right = lbl_x + txt_w
                if status_right + 12 >= x:
                    should_hide = True

        # ...and with the right-hand scan capsule.  A narrow window used to let "扫描中 (1/1)"
        # run straight over the stars.
        if not should_hide and self.right_widget is not None and not self.right_widget.isHidden():
            right_left = self.right_widget.x()
            if right_left > 0 and x + cw + 14 >= right_left:
                should_hide = True

        self.center_widget.setVisible(not should_hide)


class DarkroomMainWindow(QMainWindow):
    """SpektraDarkroom Complete Application Window."""

    # 推拉（P4）暂时不提供 UI 入口。fotufilm 点式链不消费 push_ev，而 65³ 退役后也没有
    # 回落路径，于是 19 支"同时有点式表和推拉表"的卷会选了档位毫无反应；原生卷虽然还能推，
    # 但一个只在部分卷生效、又没有任何提示的下拉，比没有这个入口更糟。
    # 逻辑全部保留（控件、参数、push 档案都在），P4 落地后把这里改回 True 即可。
    PUSH_UI_ENABLED = False

    # 「光晕空间尺度」不再提供 UI 入口。它当年存在，是因为所有卷共用一条几何推导出来的
    # 168.5/238.3/292.2 µm 梯度 —— 想让 cinestill 的 425 µm 光环出现，只能用这个滑块去拉。
    # 现在半径由每支卷自己的支撑层几何 + profile 拟合出来（core/halation_kernel.py），
    # 引擎的设计也是"改强度不改半径"（`halationLookScale` 作用在返光比例上），再留一个
    # 会把物理半径拉歪的旋钮只会制造"预览对、导出不对"。参数与 .sdc 字段全部保留，
    # 但 halation 路径读取时强制为 1.0；要恢复入口把这里改回 True 并放开 canvas 的钳制。
    HALATION_SPATIAL_UI_ENABLED = False

    # 负片默认落在哪张纸上，完全跟首选项的"负片默认相纸"走（用户 2026-11-10 指定出厂值为
    # "胶片直扫"：新底片先看未反相的原始负像）。这里只是配置缺失或 id 失效时的兜底。
    DEFAULT_NEGATIVE_PAPER = "none"
    # 转换卷（fotufilm 点式链）的默认纸。点式链一直落到 2383（电影放映正片），这里把它
    # 提成首选项的第二条默认值，配置缺失/失效时仍回这个值，行为与拆分之前一致。
    DEFAULT_FFCHAIN_PAPER = "kodak_2383"

    def __init__(self, engine, initial_session=None, initial_files=None):
        super().__init__()
        self.engine = engine
        # 旧配置只有一个「负片默认相纸」，可能设的是一张只有点式链才有的纸。改成两条链
        # 各一个默认值之后必须先迁移，否则原生卷那边会读到越界值（然后被静默换掉）。
        try:
            from core import config_manager
            config_manager.migrate_default_papers(
                getattr(engine, "FFCHAIN_PAPERS", ()),
                getattr(engine, "NATIVE_PAPERS", ()),
                self.DEFAULT_FFCHAIN_PAPER)
        except Exception:                              # noqa: BLE001 - 迁移不能挡启动
            pass
        self._initial_session = initial_session
        self._initial_files = initial_files or []
        self._has_cli_targets = bool(initial_session or (initial_files and len(initial_files) > 0))
        self.setWindowTitle(get_app_title())
        ico_path = os.path.join(get_resource_dir(), "app_icon.ico")
        png_path = os.path.join(get_resource_dir(), "app_icon.png")
        icon_file = ico_path if os.path.exists(ico_path) else png_path
        if os.path.exists(icon_file):
            self.setWindowIcon(QIcon(icon_file))
        self.resize(1440, 900)
        self.setMinimumSize(1024, 680)

        # Frameless window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)

        # Enable native Drag and Drop
        self.setAcceptDrops(True)

        # State tracking
        self.photos = []
        self.active_photo_id = None
        self._current_photo_path = None
        self._photo_is_dirty = False
        self._current_worker = None
        self._import_worker = None
        self._single_photo_loader = None
        self._slot_decode_worker = None
        self._ca_worker = None
        self.current_film_stock = "none"
        self.current_paper_stock = "none"
        self.current_print_mode = "optical"
        self.current_params = {
            "print_mode": "optical",
            "exposure_ev": 0.0,
            "highlights": 0.0,
            "shadows": 0.0,
            "color_temp": 5500.0,
            "tint": 1.0,
            "base_temp": 5500.0,
            "base_tint": 1.0,
            "film_format_mm": 35.0,
            "diffusion_family": "none",
            "diffusion_strength": 0.0,
            "diffusion_warmth": 0.0,
            "dir_amount": 1.0,
            "dir_interlayer": 1.0,
            "dir_samelayer": 1.0,
            "enlarger_illuminant": "TH-KG3",
            "enlarger_cyan": 0.0,
            "enlarger_magenta": 0.0,
            "enlarger_yellow": 0.0,
            "print_exposure": 1.0,
            "pre_flash": 0.0,
            "pre_flash_camera": 0.0,
            "black_lift": 0.0,
            "morph_gamma": 1.0,
            "developer_exhaustion": 0.0,
            "halation": 1.0,
            "halation_amount": 1.0,
            "halation_spatial_scale": 1.0,
            "scatter_amount": 1.0,
            "boost_ev": 0.0,
            "halation_boost": 0.0,
            "grain": 0.4,
            "grain_size": 1.0,
            "grain_colour": 1.0,
            "scan_noise": 0.0,
            "push_ev": 0,
            # Push is a development condition; "shot N stops under" is a capture decision. They
            # are linked only when the user asks for the whole workflow (see 模拟欠曝).
            "push_underexpose": False,
            "lens": self._get_default_lens_params(),
            "split_x": 0.5,
            "view_mode": 0
        }
        self._active_base_temp = 5500.0
        self._active_base_tint = 1.0
        self._active_base_ev = 0.0

        # Undo / Redo history
        self._undo_stack = []
        self._redo_stack = []

        # Window Geometry state tracking
        self._saved_normal_geo = None
        self._current_photo_thumb = None
        self._thumb_worker = None

        # Central canvas & widgets
        self.thumb_controller = ThumbnailController(self.engine, self)
        self.canvas = DarkroomGLCanvas(self)
        # One shared timestamp of the user's most recent input.  The candidate-card wall reads it
        # between cards and stands aside while a click/drag is in flight -- the wall is cosmetic,
        # the picture is not.  Same list object as the controller's, so no signal plumbing.
        self._interaction_watch = self.thumb_controller._interaction_watch
        self.canvas._interaction_watch = self._interaction_watch
        # Surface a GPU->CPU fallback in the status bar: it is not an error (the export still
        # succeeds) but it changes the wait from seconds to minutes, and the console line was
        # invisible from inside the app.
        self.canvas.gpuFallback.connect(
            lambda msg: self.set_app_status(msg, timeout_ms=6000))
        self.film_cards = {}
        self.paper_cards = {}
        self._current_session_path = None
        self._is_temp_session = False
        self._pending_session_map = None
        self.export_queue = ExportQueueManager(self.engine, self.canvas, self)
        self._is_importing = False
        self._pending_import_queue = []
        self._current_import_batch = []
        self._import_session_completed = 0
        self._import_session_target_total = 0
        self._photo_id_counter = 0
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self._on_status_timeout)
        self._export_status_timer = QTimer(self)
        self._export_status_timer.setSingleShot(True)
        self._export_status_timer.timeout.connect(self._on_export_status_timeout)

        # Backslash (\) key compare shortcut & long-press tracking
        self._backslash_timer = QTimer(self)
        self._backslash_timer.setSingleShot(True)
        self._backslash_timer.setInterval(180)
        self._backslash_timer.timeout.connect(self._on_backslash_timeout)
        self._backslash_pressed = False
        self._backslash_long_pressed = False

        # Real-time throttled thumbnail & card preview update timer (60ms) during slider dragging & scrubbing
        self._live_thumb_timer = QTimer(self)
        self._live_thumb_timer.setSingleShot(True)
        self._live_thumb_timer.setInterval(60)
        self._live_thumb_timer.timeout.connect(self._update_active_thumbnails)

        # Debounced full stock cards update timer (220ms after slider dragging settles)
        self._cards_debounce_timer = QTimer(self)
        self._cards_debounce_timer.setSingleShot(True)
        self._cards_debounce_timer.setInterval(220)
        self._cards_debounce_timer.timeout.connect(self._refresh_card_previews)

        # Debounced film-only cards update timer (120ms when switching film category pills)
        self._film_filter_debounce_timer = QTimer(self)
        self._film_filter_debounce_timer.setSingleShot(True)
        self._film_filter_debounce_timer.setInterval(120)
        self._film_filter_debounce_timer.timeout.connect(lambda: self._refresh_card_previews(film_only=True))

        QApplication.instance().installEventFilter(self)
        if sys.platform == "win32":
            # Keeps maximize on the monitor the window is on; see _MaximizeScreenFilter.
            self._maximize_screen_filter = _MaximizeScreenFilter(self)
            QApplication.instance().installNativeEventFilter(self._maximize_screen_filter)

        self._init_ui()
        self.export_queue.status_changed.connect(self._on_export_queue_status_changed)
        self._init_menu_and_actions()
        self._wire_canvas_events()
        self._restore_saved_state()
        self._apply_hq_preview_pref()
        self._on_theme_changed()
        theme_manager.themeChanged.connect(self._on_theme_changed)

        # If CLI session or files provided, queue their loading
        if self._initial_session:
            QTimer.singleShot(20, lambda: self._open_session_by_path(self._initial_session))
        elif self._initial_files:
            QTimer.singleShot(20, lambda: self._import_files_list(self._initial_files))

    def _restore_saved_state(self):
        """Restores window geometry, splitter sizes, accordion expansion, and stock selections."""
        try:
            win_cfg = config_manager.get_window_config()
            ui_cfg = config_manager.get_ui_state()

            # 1. Window geometry
            saved_w = win_cfg.get("width", 1440)
            saved_h = win_cfg.get("height", 900)
            saved_x = win_cfg.get("x")
            saved_y = win_cfg.get("y")
            is_maximized = win_cfg.get("is_maximized", False)

            self.resize(max(1024, saved_w), max(680, saved_h))
            if saved_x is not None and saved_y is not None:
                # Ask which monitor the saved rect sits on instead of `self.screen()`: before
                # the window is shown, `self.screen()` still reports the primary monitor, so a
                # geometry saved on a second monitor was judged off-screen, silently dropped,
                # and the window opened on the primary one.
                probe = QPoint(int(saved_x) + min(300, saved_w) // 2, int(saved_y) + min(200, saved_h) // 2)
                screen = QApplication.screenAt(probe) or QApplication.primaryScreen()
                if screen:
                    screen_geo = screen.availableGeometry()
                    if screen_geo.intersects(QRect(saved_x, saved_y, min(300, saved_w), min(200, saved_h))):
                        self.move(saved_x, saved_y)

            # 2. Splitter proportions are NOT restored here: none of the splitters exist yet
            # (they are built further down __init__). _restore_splitter_layout() does it once
            # the panes are live -- see its call at the end of the editor layout.

            # 3. Right accordion panels
            sections_expanded = ui_cfg.get("sections_expanded", {})
            if "exposure" in sections_expanded and hasattr(self, "sec_exposure"):
                self.sec_exposure.set_expanded(sections_expanded["exposure"])
            if "optics_diff" in sections_expanded and hasattr(self, "sec_optics"):
                self.sec_optics.set_expanded(sections_expanded["optics_diff"])
            if "chemistry" in sections_expanded and hasattr(self, "sec_chemistry"):
                self.sec_chemistry.set_expanded(sections_expanded["chemistry"])
            if "enlarger" in sections_expanded and hasattr(self, "sec_enlarger"):
                self.sec_enlarger.set_expanded(sections_expanded["enlarger"])
            if "optics" in sections_expanded and hasattr(self, "sec_optical"):
                self.sec_optical.set_expanded(sections_expanded["optics"])

            # 4. Default to 'none' (Bypass mode: 无胶卷与无相纸透传) for all photos
            self.current_film_stock = "none"
            self.current_paper_stock = "none"
            for sid, card in getattr(self, "film_cards", {}).items():
                card.set_active(sid == "none")
            for sid, card in getattr(self, "paper_cards", {}).items():
                card.set_active(sid == "none")

            if is_maximized:
                self._restore_as_maximized = True

            # 4. Restore film & paper stock sort order
            film_sort = ui_cfg.get("film_sort_order", "name")
            self._sort_and_reflow_stocks("film", film_sort, persist=False)
            paper_sort = ui_cfg.get("paper_sort_order", "name")
            self._sort_and_reflow_stocks("paper", paper_sort, persist=False)

            # 5. Restore last session if preference is enabled and not launched with CLI session/files
            prefs = config_manager.get_preferences()
            restore_session_enabled = prefs.get("restore_last_session", prefs.get("restore_last_files", True))
            if not os.environ.get("FAST_TEST") and restore_session_enabled and not getattr(self, "_has_cli_targets", False) and not self.photos and not self._current_session_path:
                last_session = config_manager.get_last_session_path()
                if last_session and os.path.exists(last_session):
                    auto_temp = config_manager.get_auto_saved_session_path()
                    is_temp = (normalize_path(last_session).lower() == normalize_path(auto_temp).lower())
                    if is_temp:
                        sess_data = session_cache_manager.load_session_file(last_session)
                        if not sess_data or not sess_data.get("photos"):
                            # Empty auto-saved session should not restore into workbench
                            config_manager.save_last_session_path(None)
                            last_session = None
                    if last_session:
                        self._pending_startup_restore_timer = QTimer(self)
                        self._pending_startup_restore_timer.setSingleShot(True)
                        self._pending_startup_restore_timer.timeout.connect(
                            lambda: (not getattr(self, "_is_importing", False) and not self.photos and not self._current_session_path) and self._open_session_by_path(last_session)
                        )
                        self._pending_startup_restore_timer.start(40)
                    else:
                        self.stack.setCurrentIndex(0)
                        self.btn_title_export.setVisible(False)
                        self._update_window_title()
                else:
                    self.stack.setCurrentIndex(0)
                    self.btn_title_export.setVisible(False)
                    self._update_window_title()
            else:
                if not getattr(self, "_has_cli_targets", False) and not self.photos:
                    self.stack.setCurrentIndex(0)
                    self.btn_title_export.setVisible(False)
                    self._update_window_title()

            # 6. Silent session cache auto-cleaning check
            if not os.environ.get("FAST_TEST"):
                QTimer.singleShot(2500, self._auto_clean_cache_soon)
        except Exception as e:
            logger.warning(f"Error restoring application state: {e}")

    def _restore_saved_session(self, existing_files, active_file=None, session_name="", path_updated=False, ratings_map=None):
        """Asynchronously restores session photos using SessionRestoreWorker and FadingStackedWidget.
        Pre-allocates slots for all files to guarantee order and count stability (100 remains 100).
        Transitions smoothly to editor workspace while restoring active photo and thumbnails in background.
        """
        if not existing_files:
            return

        total_files = len(existing_files)
        self.photos = [None] * total_files
        self._is_importing = True
        self._import_lock_time = time.time()
        self._set_import_hq_suspend(True)
        self._set_cancel_chip(True, "恢复")
        self._seed_restore_placeholders(existing_files, active_file)

        target_path = active_file if (active_file and active_file in existing_files) else existing_files[0]
        self._target_initial_active_path = target_path

        # 1. Switch to darkroom workspace smoothly
        self.stack.setCurrentIndexAnimated(1, duration_ms=250)
        self.btn_title_export.setVisible(True)
        if session_name:
            self.set_app_status(f"正在载入会话：{session_name}…", spinning=True, timeout_ms=0)
        else:
            self.set_app_status("正在载入会话…", spinning=True, timeout_ms=0)

        # 2. Launch background worker
        worker = SessionRestoreWorker(
            existing_files,
            active_file=target_path,
            thumb_controller=self.thumb_controller,
            engine=self.engine,
            is_params_modified_fn=self._is_params_modified,
            ratings_map=ratings_map,
            default_paper=self._default_negative_paper(),
            default_ffchain_paper=self._default_ffchain_paper(),
            parent=self
        )
        self._session_restore_worker = worker

        def _on_active_photo_ready(target_idx, target_entry):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            if 0 <= target_idx < len(self.photos):
                self.photos[target_idx] = target_entry
            target_id = target_entry.get("id")
            if target_id:
                th = target_entry.get("raw_thumbnail_rgb")
                if th is None:
                    th = target_entry.get("thumbnail_rgb")
                # 恢复会话先铺缩略图占位。完整链一次 1.4-2.2 s（33³ LUT 现烘 + 全分辨率
                # 纹理上传），正好压在用户最想点"取消"的那几秒上；占位是 0 ms，完整链
                # 等恢复收尾再跑（用户已经接受了导入期的"先粗糙后精细"）。
                if th is not None and hasattr(self.canvas, "set_placeholder_image"):
                    self._show_import_placeholder(target_entry)
                else:
                    try:
                        self.on_switch_photo(target_id)
                    except Exception as e:
                        logger.error(f"Immediate active photo activation failed: {e}", exc_info=True)
            self._update_window_title()

        def _on_progress(current, total):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            if getattr(self, "_is_importing", False):
                self.set_app_status(f"正在载入会话（{current}/{total}）…", spinning=True, timeout_ms=0)

        def _on_restore_finished(all_photos, uncached_slots, final_target_path):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            self._session_restore_worker = None
            self.photos = all_photos
            self.canvas.set_has_library_photos(bool(self.photos))

            target_entry = next((p for p in self.photos if p and p.get("path") == final_target_path), self.photos[0] if self.photos else None)
            target_id = target_entry["id"] if target_entry else None
            # `activePhotoReady` 已经切过一次了。同 target 且 float_img 已经就位就不要再来一遍
            # —— 一次 on_switch_photo 实测 1.4-2.2 s（33³ LUT 现烘 + 24 MP 纹理上传）。
            already_loaded = bool(
                target_id
                and self.active_photo_id == target_id
                and (target_entry or {}).get("float_img") is not None
            )

            # 底片栏在恢复过程中已经按占位顺序一张张长好了（photoReady），这里只补漏 +
            # 定光标。别再用 set_photos：那会重建所有格子，用户刚看着长出来的会闪一下。
            missing = [p for p in self.photos if p and not self.filmstrip.has_photo(p.get("id"))]
            if missing:
                self.filmstrip.append_photos_batch(missing)
            if target_id:
                self.filmstrip.set_active_photo(target_id)

            # Mark dirty/edited badges
            for entry in self.photos:
                if entry and (entry.get("is_dirty") or entry.get("is_edited")):
                    self.filmstrip.set_photo_dirty(entry["id"], True)

            # Force activation of target photo so filmstrip, canvas, sidebar, cards, and histogram are all completely initialized
            deferred_pending = bool(getattr(self, "_deferred_switch_id", None))
            if deferred_pending:
                # 首图现在是占位缩略图（见 _on_active_photo_ready）。还有未缓存的槽位要解
                # 就把完整链留到 _on_all_slots_decoded，否则现在就补上。
                if not uncached_slots:
                    QTimer.singleShot(IMPORT_DEFERRED_SWITCH_DELAY_MS, self._activate_deferred_photo)
            elif target_id and not already_loaded:
                try:
                    self.on_switch_photo(target_id, force=True)
                except Exception as e:
                    logger.error(f"Error activating photo on session restore: {e}", exc_info=True)
            else:
                if hasattr(self, '_cards_debounce_timer'):
                    self._cards_debounce_timer.start(100)
                self._schedule_histogram_update()

            # 底片栏**不**在恢复过程中跟随：那样一打开就居中到上次停留的那张，左边全空。
            # 等这轮彻底加载完再滚过去（见 `_scroll_to_active_after_restore`）。
            if not uncached_slots:
                self._scroll_to_active_after_restore(
                    320 + (IMPORT_DEFERRED_SWITCH_DELAY_MS if getattr(self, "_deferred_switch_id", None) else 0))

            # Trigger a post-animation polish pass once FadingStackedWidget transition reaches 100%
            QTimer.singleShot(280, lambda: (
                getattr(self, 'canvas', None) and self.canvas.update(),
                self._schedule_histogram_update(),
                self._refresh_card_previews()
            ))

            self._update_window_title()

            # Uncached slots handling
            if uncached_slots:
                self._is_importing = True
                self._import_lock_time = time.time()
                self._set_cancel_chip(True, "恢复")
                self.set_app_status(f"正在恢复底片库（0/{len(uncached_slots)}）…", spinning=True, timeout_ms=0)
                self._reset_import_delivery()
                self._slot_decode_worker = SlotDecodeWorker(uncached_slots, self.engine, parent=self)
                # 16 个解码线程一秒能吐上百条，和导入一样必须按帧投放；收尾也排队，
                # 否则 `finished` 会比最后几条结果早到，状态栏先说"就绪"而底片条还在补图。
                self._slot_decode_worker.slotDecoded.connect(self._enqueue_slot_decoded)
                self._slot_decode_worker.progress.connect(self._on_slot_decode_progress)
                self._slot_decode_worker.finished.connect(self._on_slot_stream_finished)
                self._slot_decode_worker.start()
            else:
                if self._start_next_pending_import_batch():
                    return
                self._is_importing = False
                self._set_import_hq_suspend(False, hold=(total_files > 1))
                self._set_cancel_chip(False)
                self._sync_filmstrip_thumbnails("恢复收尾")
                if session_name:
                    if path_updated:
                        self.set_app_status(f"会话已恢复：{session_name}（{total_files} 张），底片路径已更新", spinning=False, timeout_ms=3000)
                    else:
                        self.set_app_status(f"会话已恢复：{session_name}（{total_files} 张）", spinning=False, timeout_ms=2500)
                else:
                    self.set_app_status(f"底片库已就绪（{total_files} 张）", spinning=False, timeout_ms=2500)

        def _on_failed(err_msg):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            self._session_restore_worker = None
            if self._start_next_pending_import_batch():
                return
            self._is_importing = False
            self._set_import_hq_suspend(False, hold=False)
            self._set_cancel_chip(False)
            self.set_app_status(f"载入会话失败：{err_msg}", spinning=False, timeout_ms=4000)

        worker.photoReady.connect(self._enqueue_restore_photo)
        worker.activePhotoReady.connect(_on_active_photo_ready)
        worker.progress.connect(_on_progress)
        worker.restoreFinished.connect(_on_restore_finished)
        worker.failed.connect(_on_failed)

        if os.environ.get("FAST_TEST"):
            worker.run_sync()
        else:
            worker.start()

    def _extend_dwm_frame(self):
        """Ask DWM to composite the whole window (smooth maximise/restore animation + native
        drop shadow). Kept separate from the style setup so it can be re-asserted cheaply."""
        if sys.platform != "win32":
            return
        try:
            class MARGINS(ctypes.Structure):
                _fields_ = [
                    ("cxLeftWidth", ctypes.c_int),
                    ("cxRightWidth", ctypes.c_int),
                    ("cyTopHeight", ctypes.c_int),
                    ("cyBottomHeight", ctypes.c_int),
                ]
            margins = MARGINS(1, 1, 1, 1)
            ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(int(self.winId()), byref(margins))
        except Exception:
            pass

    def _release_native_filters(self):
        """Drop the app-wide native message filter before teardown.

        It is installed on the QApplication, so it would otherwise keep being called (for
        every Windows message, for every window) while this window's HWND is halfway gone.
        """
        filter_ = getattr(self, "_maximize_screen_filter", None)
        if filter_ is None:
            return
        try:
            QApplication.instance().removeNativeEventFilter(filter_)
        except Exception:
            pass
        self._maximize_screen_filter = None

    def _setup_native_window_frame(self):
        """Configure Win32 window styles, DWM client frame extension, and Aero Snap once on show."""
        if sys.platform != "win32":
            return
        # One-shot: re-applying GWL_STYLE / SWP_FRAMECHANGED on every show (restore from
        # minimise, re-show) makes Windows recompute and repaint the non-client area, which is
        # what flashes the classic WS_CAPTION title bar over the custom one.
        if getattr(self, "_native_frame_ready", False):
            self._extend_dwm_frame()
            return
        try:
            hwnd = int(self.winId())
            GWL_STYLE = -16
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_MAXIMIZEBOX = 0x00010000
            WS_MINIMIZEBOX = 0x00020000
            WS_SYSMENU = 0x00080000
            # Frame strategy (see _FRAME_MODE at module level):
            # WS_CAPTION is what lets DWM animate minimise/maximise, but it is also the single
            # piece of state that lets DWM / the shell paint the classic title bar on every frame
            # re-validation - a paint that never reaches our window procedure, so no WM_NC*
            # interception can stop it. "clean" removes the style bit (no animation);
            # "anim" keeps it but turns off DWM non-client rendering (no caption, keeps zoom).
            if _FRAME_MODE == "clean":
                new_style = (style | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU) & ~WS_CAPTION
            else:
                new_style = style | WS_CAPTION | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)
            if _FRAME_MODE == "anim":
                # DWMWA_NCRENDERING_POLICY. DWMNCRP_DISABLED (1) would guarantee no classic
                # caption is ever painted, but it also makes DWMWA_EXTENDED_FRAME_BOUNDS report
                # the raw window rect (work area + 8 px invisible border) instead of the visible
                # picture, so window-picker tools frame us 8 px too large and the bottom edge
                # lands on the taskbar. USEWINDOWSTYLE (0) keeps the caption out of sight anyway
                # -- WM_NCCALCSIZE hands the whole window to the client area, so there is no
                # non-client strip left for DWM to paint -- and keeps the frame bounds honest.
                policy = ctypes.c_int(_NC_POLICY)
                try:
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, 2, ctypes.byref(policy), ctypes.sizeof(policy))
                except Exception:
                    pass
            _frame_log(f"frame mode={_FRAME_MODE} nc_policy={_NC_POLICY} "
                       f"style=0x{new_style & 0xFFFFFFFF:08X} "
                       f"caption={int((new_style & WS_CAPTION) != 0)}")
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_APPWINDOW)

            # Extend DWM frame into client area so DWM composites entire window smoothly during
            # maximize/restore animation without black borders and provides native drop shadow
            self._extend_dwm_frame()

            SWP_NOZORDER = 0x0004
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_FRAMECHANGED = 0x0020
            ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)
            self._native_frame_ready = True
        except Exception:
            pass

    def _set_native_caption(self, enabled: bool):
        """Add/remove WS_CAPTION at runtime (used while closing, see closeEvent)."""
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            GWL_STYLE = -16
            WS_CAPTION = 0x00C00000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            new_style = (style | WS_CAPTION) if enabled else (style & ~WS_CAPTION)
            if new_style != style:
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)
            _frame_log(f"caption set to {int(enabled)}")
        except Exception:
            pass

    def _apply_win11_corners(self, is_max=False):
        """Update Windows 11 rounded corner preference (ROUND when normal, DONOTROUND when maximized).
        Never issues SWP_FRAMECHANGED here to avoid interrupting the DWM maximize/restore animation."""
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            corner_pref = c_int(1 if is_max else 2)  # 1 = DONOTROUND, 2 = ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, byref(corner_pref), ctypes.sizeof(corner_pref)
            )
        except Exception:
            pass

    def _apply_native_window_icon(self):
        if sys.platform != "win32":
            return
        # One-shot: WM_SETICON / SetClassLongPtr make Windows invalidate and repaint the
        # non-client caption (the icon lives in the caption). Running this on every activation
        # change - as it used to - is exactly what made the classic title bar flash when
        # switching to or from another application.
        if getattr(self, "_native_icon_ready", False):
            return
        try:
            hwnd = int(self.winId())

            ico_path = os.path.join(get_resource_dir(), "app_icon.ico")
            if not os.path.exists(ico_path):
                return
            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x00000010
            LR_DEFAULTSIZE = 0x00000040
            hicon_big = ctypes.windll.user32.LoadImageW(None, ico_path, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
            if not hicon_big:
                hicon_big = ctypes.windll.user32.LoadImageW(None, ico_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
            hicon_small = ctypes.windll.user32.LoadImageW(None, ico_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
            WM_SETICON = 0x0080
            if hicon_big:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 1, hicon_big)
            if hicon_small:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 0, hicon_small)

            # Set Class Icon (GCLP_HICON = -14, GCLP_HICONSM = -34) for persistent Windows Taskbar grouping
            set_class_long = getattr(ctypes.windll.user32, "SetClassLongPtrW", None) or getattr(ctypes.windll.user32, "SetClassLongW", None)
            if set_class_long:
                if hicon_big:
                    try:
                        set_class_long(hwnd, -14, hicon_big)
                    except Exception:
                        pass
                if hicon_small:
                    try:
                        set_class_long(hwnd, -34, hicon_small)
                    except Exception:
                        pass

            # Ensure OS recognises this as a standard top-level application window (Taskbar /
            # Task Manager identity). WS_CAPTION and friends are set once in
            # _setup_native_window_frame: re-applying them on every activation change made
            # Windows recompute the non-client area, which flashed the classic title bar.
            if not getattr(self, "_taskbar_identity_ready", False):
                GWL_EXSTYLE = -20
                WS_EX_APPWINDOW = 0x00040000
                WS_EX_TOOLWINDOW = 0x00000080
                ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, (ex_style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW)
                self._taskbar_identity_ready = True
            self._native_icon_ready = True
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        self._setup_native_window_frame()
        self._apply_win11_corners(self.isMaximized())
        self._apply_native_window_icon()
        _frame_log(f"showEvent visible={int(self.isVisible())} max={int(self.isMaximized())}")
        if self._saved_normal_geo is None and not self.isMaximized():
            self._saved_normal_geo = self.geometry()
        if getattr(self, '_restore_as_maximized', False):
            self._restore_as_maximized = False
            self.showMaximized()
        # Ensure film & paper grids are reflowed with actual rendered layout width
        QTimer.singleShot(0, self.reflow_grids)
        QTimer.singleShot(80, self.reflow_grids)

    def changeEvent(self, event):
        """Keep maximize/restore button state in 100% sync with window state."""
        if event.type() == QEvent.Type.WindowStateChange:
            is_max = self.isMaximized()
            _frame_log(f"changeEvent WindowStateChange max={int(is_max)}")
            if hasattr(self, 'btn_max'):
                self.btn_max.set_maximized_state(is_max)
                if hasattr(self, 'btn_close'):
                    self.btn_close.set_maximized_state(is_max)
            self._apply_win11_corners(is_max)
            # Defer root style update and icon recheck to avoid blocking the GUI thread during DWM maximize animation
            QTimer.singleShot(0, lambda: self._on_window_state_changed_deferred(is_max))
        elif event.type() == QEvent.Type.ActivationChange:
            _frame_log(f"changeEvent ActivationChange active={int(self.isActiveWindow())}")
            self._apply_native_window_icon()
        super().changeEvent(event)

    def _on_window_state_changed_deferred(self, is_max):
        self._update_root_style(is_max)
        self._apply_native_window_icon()

    def nativeEvent(self, eventType, message):
        """Win32 native event handling for frameless window:
        - WM_NCCALCSIZE: Hides default non-client frame/titlebar while preserving DWM animations
        - WM_NCHITTEST: Native hardware-accelerated edge resizing with accurate cursor feedback
        """
        if sys.platform == "win32":
            try:
                from ctypes import wintypes
                class MSG(ctypes.Structure):
                    _fields_ = [
                        ("hwnd", wintypes.HWND),
                        ("message", wintypes.UINT),
                        ("wParam", wintypes.WPARAM),
                        ("lParam", wintypes.LPARAM),
                        ("time", wintypes.DWORD),
                        ("pt", wintypes.POINT)
                    ]
                msg = MSG.from_address(int(message))

                if _FRAME_DEBUG and msg.message in (0x0047, 0x007D, 0x0083, 0x0085, 0x0086, 0x00AE, 0x00AF,
                                                    0x0014, 0x001A, 0x001C, 0x0024, 0x031E):
                    try:
                        st = ctypes.windll.user32.GetWindowLongW(msg.hwnd, -16)
                        _frame_log(f"nc msg=0x{msg.message:04X} wparam={msg.wParam} "
                                   f"style=0x{st & 0xFFFFFFFF:08X} caption={int(bool(st & 0x00C00000))} "
                                   f"zoomed={int(ctypes.windll.user32.IsZoomed(msg.hwnd))} "
                                   f"visible={int(ctypes.windll.user32.IsWindowVisible(msg.hwnd))}")
                    except Exception:
                        pass

                # Suppress GDI background erase to prevent black flash during window resizing/animation
                if msg.message == 0x0014:  # WM_ERASEBKGND
                    return True, 1

                # 0. WM_GETMINMAXINFO (0x0024): Constrain maximized window strictly to work area (excluding taskbar)
                if msg.message == 0x0024:
                    h_mon = ctypes.windll.user32.MonitorFromWindow(msg.hwnd, 2)  # MONITOR_DEFAULTTONEAREST
                    mi = MONITORINFO()
                    mi.cbSize = ctypes.sizeof(MONITORINFO)
                    if ctypes.windll.user32.GetMonitorInfoW(h_mon, ctypes.byref(mi)):
                        mmi = MINMAXINFO.from_address(msg.lParam)
                        w = mi.rcWork.right - mi.rcWork.left
                        h = mi.rcWork.bottom - mi.rcWork.top
                        bx = ctypes.windll.user32.GetSystemMetrics(32) + ctypes.windll.user32.GetSystemMetrics(92)  # SM_CXFRAME + SM_CXPADDEDBORDER
                        by = ctypes.windll.user32.GetSystemMetrics(33) + ctypes.windll.user32.GetSystemMetrics(92)  # SM_CYFRAME + SM_CXPADDEDBORDER
                        mmi.ptMaxPosition.x = mi.rcWork.left - bx
                        mmi.ptMaxPosition.y = mi.rcWork.top - by
                        mmi.ptMaxSize.x = w + 2 * bx
                        mmi.ptMaxSize.y = h + 2 * by

                        # Enforce hardware-level minimum tracking size (prevents window squishing below minimum dimensions)
                        dpr = self.devicePixelRatio() if hasattr(self, 'devicePixelRatio') else 1.0
                        min_w = int(max(960, self.minimumWidth()) * dpr)
                        min_h = int(max(600, self.minimumHeight()) * dpr)
                        mmi.ptMinTrackSize.x = min_w
                        mmi.ptMinTrackSize.y = min_h
                    return True, 0

                # Non-client painting / activation redraws.
                #   WM_NCPAINT must be allowed through: it is how the system announces the frame to
                #   DWM. Swallowing it (which this code did since v0.4.54) makes DWM forget the
                #   window has a frame at all, so DWMWA_EXTENDED_FRAME_BOUNDS -- the rectangle every
                #   window picker (WeChat, Snipping Tool) frames us with -- degrades from the work
                #   area to the raw window rect: work area + the 8 px invisible resize ring, with
                #   the bottom edge hanging over the taskbar. Measured: swallow -> accurate on the
                #   first maximize only, then work+8 forever; pass through -> accurate every cycle.
                #   Painting is a no-op anyway: WM_NCCALCSIZE hands the whole window to the client
                #   area, so there is no non-client strip left to draw into (grab is pixel-identical
                #   either way).
                #   WM_NCACTIVATE and the WM_NCUAHDRAW* messages are still swallowed -- those are
                #   what repaint the classic caption on activation on some Win10/11 builds.
                if msg.message == 0x0086:  # WM_NCACTIVATE
                    return True, 1
                if msg.message in (0x00AE, 0x00AF):  # WM_NCUAHDRAWCAPTION, WM_NCUAHDRAWFRAME
                    return True, 0

                # 0x031E WM_DWMCOMPOSITIONCHANGED: composition toggled -> re-assert the DWM
                # frame extension, otherwise Windows falls back to painting the classic
                # non-client title bar. Fall through to Qt afterwards.
                if msg.message == 0x031E:
                    self._extend_dwm_frame()

                # 1. WM_NCCALCSIZE (0x0083): Remove standard title bar and frame while keeping DWM animations
                if msg.message == 0x0083:
                    if msg.wParam:
                        if ctypes.windll.user32.IsZoomed(msg.hwnd):
                            bx = ctypes.windll.user32.GetSystemMetrics(32) + ctypes.windll.user32.GetSystemMetrics(92)
                            by = ctypes.windll.user32.GetSystemMetrics(33) + ctypes.windll.user32.GetSystemMetrics(92)
                            params = NCCALCSIZE_PARAMS.from_address(msg.lParam)
                            params.rgrc[0].left += bx
                            params.rgrc[0].top += by
                            params.rgrc[0].right -= bx
                            params.rgrc[0].bottom -= by
                        return True, 0
                    return True, 0

                # 2. WM_NCHITTEST (0x0084): Handle border resize hover and dragging with physical coords
                if msg.message == 0x0084 and not self.isMaximized():
                    rect = wintypes.RECT()
                    ctypes.windll.user32.GetWindowRect(msg.hwnd, ctypes.byref(rect))
                    x = ctypes.c_short(msg.lParam & 0xFFFF).value
                    y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value

                    rx = x - rect.left
                    ry = y - rect.top
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top

                    # Dynamic border width taking DPI into account
                    dpr = self.devicePixelRatio() if hasattr(self, 'devicePixelRatio') else 1.0
                    border_width = max(8, int(8 * dpr))

                    if 0 <= rx <= w and 0 <= ry <= h:
                        left = rx < border_width
                        right = rx > (w - border_width)
                        top = ry < border_width
                        bottom = ry > (h - border_width)

                        if top and left: return True, 13      # HTTOPLEFT
                        if top and right: return True, 14     # HTTOPRIGHT
                        if bottom and left: return True, 16   # HTBOTTOMLEFT
                        if bottom and right: return True, 17  # HTBOTTOMRIGHT
                        if left: return True, 10              # HTLEFT
                        if right: return True, 11             # HTRIGHT
                        if top: return True, 12               # HTTOP
                        if bottom: return True, 15            # HTBOTTOM
            except Exception:
                pass
        return super().nativeEvent(eventType, message)

    def _set_dropzone_active(self, is_active):
        if hasattr(self, 'dropzone'):
            is_dark = theme_manager.is_dark()
            if is_active:
                bg = "rgba(245, 158, 11, 0.08)" if is_dark else "rgba(245, 158, 11, 0.12)"
                self.dropzone.setStyleSheet(f"""
                    QFrame#welcomeDropzone {{
                        border: 2px dashed #f59e0b;
                        border-radius: 14px;
                        background: {bg};
                    }}
                    QLabel {{ border: none; background: transparent; }}
                """)
            else:
                border_color = "#353846" if is_dark else "#cbd5e1"
                bg_color = "rgba(255, 255, 255, 0.015)" if is_dark else "#ffffff"
                hover_bg = "rgba(245, 158, 11, 0.03)" if is_dark else "rgba(245, 158, 11, 0.06)"
                self.dropzone.setStyleSheet(f"""
                    QFrame#welcomeDropzone {{
                        border: 1.5px dashed {border_color};
                        border-radius: 14px;
                        background: {bg_color};
                    }}
                    QFrame#welcomeDropzone:hover {{
                        border-color: #f59e0b;
                        background: {hover_bg};
                    }}
                    QLabel {{ border: none; background: transparent; }}
                """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_dropzone_active(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_dropzone_active(False)
        event.accept()

    def dropEvent(self, event):
        self._set_dropzone_active(False)
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        from core.path_utils import normalize_path
        valid_exts = IMPORTABLE_IMAGE_EXTENSIONS
        sdss_files = []
        img_files = []
        rejected_files = []
        empty_dirs = []

        for url in event.mimeData().urls():
            if url.isLocalFile():
                local_path = url.toLocalFile()
                if os.path.exists(local_path):
                    if os.path.isdir(local_path):
                        usable = 0
                        for root, _, fnames in os.walk(local_path):
                            for f in fnames:
                                ext = os.path.splitext(f)[1].lower()
                                if ext == ".sdss":
                                    sdss_files.append(os.path.join(root, f))
                                    usable += 1
                                elif ext in valid_exts:
                                    img_files.append(os.path.join(root, f))
                                    usable += 1
                        if not usable:
                            empty_dirs.append(local_path)
                    else:
                        ext = os.path.splitext(local_path)[1].lower()
                        if ext == ".sdss":
                            sdss_files.append(local_path)
                        elif ext in valid_exts:
                            img_files.append(local_path)
                        else:
                            rejected_files.append(local_path)
                else:
                    rejected_files.append(local_path)

        if sdss_files:
            event.acceptProposedAction()
            target_sdss = normalize_path(sdss_files[0])
            # 若是拖入了已打开的会话文件，直接不管
            if self._current_session_path and target_sdss.lower() == normalize_path(self._current_session_path).lower():
                return
            # 如有当前未保存的项目，提示保存待用户决定保存还是不保存后再打开新拖入的
            if not self._prompt_save_if_session_dirty():
                return
            self._open_session_by_path(target_sdss)
            return

        if img_files:
            event.acceptProposedAction()
            self._import_files_list(img_files)
        else:
            event.ignore()
            self._report_unsupported_drop(rejected_files, empty_dirs)

    def _unsupported_drop_message(self, rejected_files, empty_dirs) -> str:
        lines = []
        if rejected_files:
            shown = rejected_files[:6]
            lines.extend(f"  {os.path.basename(p)}" for p in shown)
            if len(rejected_files) > len(shown):
                lines.append(f"  …另有 {len(rejected_files) - len(shown)} 个文件")
        if empty_dirs:
            for d in empty_dirs[:2]:
                lines.append(f"  文件夹「{os.path.basename(d) or d}」里没有可导入的文件")
            if len(empty_dirs) > 2:
                lines.append(f"  …另有 {len(empty_dirs) - 2} 个空文件夹")
        return ("暗房只接受相机 RAW 与无损位图（PNG / TIFF）。\n\n"
                + ("未导入：\n" + "\n".join(lines) if lines else "拖入的内容里没有可导入的文件。"))

    def _report_unsupported_drop(self, rejected_files, empty_dirs):
        """Tell the user why a drop did nothing.

        Lossy containers (JPG/WebP) are refused on purpose and previously vanished without a
        word -- dropping a JPEG looked like the app had simply ignored the gesture.
        """
        from ui.confirm_dialog import show_darkroom_info

        show_darkroom_info(
            self,
            "不支持的文件",
            "这些文件无法导入",
            self._unsupported_drop_message(rejected_files, empty_dirs),
            ok_text="知道了",
            badge_symbol="!",
            width=470,
        )

    def _update_root_style(self, is_max):
        is_dark = theme_manager.is_dark()
        app_bg = "#121316" if is_dark else "#ebedf2"
        root_border_color = "#282a34" if is_dark else "#c8ccd8"
        tb_bg = "#141518" if is_dark else "#f0f2f6"
        scroll_handle = "#3f4350" if is_dark else "#94a3b8"
        scroll_hover = "#cbd5e1" if is_dark else "#64748b"

        radius = "0px" if is_max else "8px"
        border = "none" if is_max else f"1px solid {root_border_color}"
        tb_radius = "0px" if is_max else "8px"
        cw = self.centralWidget()
        if cw:
            if cw.layout():
                cw.layout().setContentsMargins(0, 0, 0, 0)
            cw.setStyleSheet(f"""
                QWidget#rootContainer {{
                    background: {app_bg};
                    border: {border};
                    border-radius: {radius};
                }}
                QScrollBar:vertical {{
                    background: transparent;
                    width: 6px;
                    margin: 0px;
                    border: none;
                }}
                QScrollBar::handle:vertical {{
                    background: {scroll_handle};
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: {scroll_hover};
                    border-radius: 3px;
                }}
                QScrollBar::handle:vertical:pressed {{
                    background: #f59e0b;
                    border-radius: 3px;
                }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }}
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                    background: none;
                    border: none;
                }}
                QScrollBar:horizontal {{
                    background: transparent;
                    height: 6px;
                    margin: 0px;
                    border: none;
                }}
                QScrollBar::handle:horizontal {{
                    background: {scroll_handle};
                    min-width: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:horizontal:hover {{
                    background: {scroll_hover};
                    border-radius: 3px;
                }}
                QScrollBar::handle:horizontal:pressed {{
                    background: #f59e0b;
                    border-radius: 3px;
                }}
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }}
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                    background: none;
                    border: none;
                }}
            """)
        if hasattr(self, 'title_bar'):
            self.title_bar.setStyleSheet(f"background: {tb_bg}; border: none; border-top-left-radius: {tb_radius}; border-top-right-radius: {tb_radius};")

    def _apply_tray_style(self, tray, scroll=None):
        """卡片墙托盘的底色/描边 + 同色边缘渐隐。主题切换时也要重调。"""
        if tray is None:
            return
        colors = theme_manager.get_colors()
        # 与 CompactExifWidget 的 exifCard 同一套配色：subpanel_bg + panel_border + R6
        card_bg = colors["subpanel_bg"]
        tray.setStyleSheet(
            f"QFrame#{tray.objectName()} {{"
            f" background: {card_bg};"
            f" border: 1px solid {colors['panel_border']};"
            f" border-radius: 6px; }}"
        )
        if scroll is not None and hasattr(scroll, "set_fade_base_color"):
            scroll.set_fade_base_color(card_bg, height=16)

    def _on_theme_changed(self, theme_name=None):
        is_dark = theme_manager.is_dark()
        colors = theme_manager.get_colors()

        # 1. Update native titlebar
        apply_dark_titlebar(self, is_dark=is_dark)

        # 2. Update rootContainer stylesheet
        self._update_root_style(self.isMaximized())

        # 3. Update Menu Bar
        if hasattr(self, 'menu_bar'):
            self.menu_bar.setStyleSheet(theme_manager.get_menu_stylesheet())

        # 4. Brand title & header titles
        if hasattr(self, 'lbl_brand'):
            brand_col = "#f59e0b" if is_dark else "#d97706"
            self.lbl_brand.setStyleSheet(f"color: {brand_col}; font-size: 13px; font-weight: bold; letter-spacing: 0.5px; border: none;")
        if hasattr(self, 'lbl_paper_title'):
            paper_col = "#f59e0b" if is_dark else "#d97706"
            self.lbl_paper_title.setStyleSheet(f"color: {paper_col}; font-size: 11px; font-weight: bold;")

        # 5. Right Sidebar Container
        if hasattr(self, 'sidebar_container'):
            self.sidebar_container.setStyleSheet(f"background: {colors['app_bg']};")

        # 5b. 卡片墙托盘（底色 + 渐隐色一起走，浅色下渐隐带必须等于托盘色）
        if hasattr(self, 'film_tray'):
            self._apply_tray_style(self.film_tray, getattr(self, 'film_scroll', None))

        # 6. View Toolbar & Status Bar Controls
        self._update_view_toolbar_style()

        # 7. Welcome View
        self._update_welcome_view_style()

        # 8. Editor View Background
        if hasattr(self, 'editor_widget'):
            self.editor_widget.setStyleSheet(f"background: {colors['app_bg']};")

        # 9. Stock cards reflow / refresh appearance
        for card in getattr(self, 'film_cards', {}).values():
            if hasattr(card, '_update_appearance'):
                card._update_appearance()
        for card in getattr(self, 'paper_cards', {}).values():
            if hasattr(card, '_update_appearance'):
                card._update_appearance()

        # 10. Update menu action icons
        actions_icons = [
            ('act_open_file', "open_file.png"),
            ('act_open_folder', "open_folder.png"),
            ('act_open_session', "open_session.png"),
            ('act_save_session', "save.png"),
            ('act_quick_export', "quick_export.png"),
            ('act_export', "export.png"),
            ('act_undo', "undo.png"),
            ('act_redo', "redo.png"),
            ('act_crop', "crop.png"),
            ('act_reset_all', "reset.png"),
            ('act_prefs', "preferences.png"),
            ('act_fit_img', "fit_window.png"),
            ('act_compare', "compare.png"),
            ('act_fit_win', "fit_window.png"),
            ('act_100', "zoom_100.png"),
        ]
        for attr, icon_name in actions_icons:
            act = getattr(self, attr, None)
            if act:
                act.setIcon(get_icon(icon_name, is_dark=is_dark))
        if hasattr(self, 'act_clear_sdc') and self.act_clear_sdc:
            self.act_clear_sdc.setIcon(QIcon())
        if hasattr(self, 'act_clear_crop') and self.act_clear_crop:
            self.act_clear_crop.setIcon(QIcon())

        # 11. Optics section styling
        self._update_optics_ui_style()

        # 12. Force update
        self.update()

    def _update_view_toolbar_style(self):
        is_dark = theme_manager.is_dark()
        colors = theme_manager.get_colors()
        if hasattr(self, 'view_toolbar'):
            self.view_toolbar.setStyleSheet(f"background: {colors['view_toolbar_bg']}; border: none;")
        if hasattr(self, 'lbl_photo_info'):
            self.lbl_photo_info.setStyleSheet(f"QLabel#lbl_photo_info {{ color: {colors['text_secondary']}; font-size: 11px; font-weight: 500; }}")
        if hasattr(self, 'status_label'):
            self.status_label.setStyleSheet(f"QLabel#status_label {{ color: {colors['text_muted']}; font-size: 11px; }}")

        btn_bg = colors['view_toolbar_btn_bg']
        btn_border = colors['view_toolbar_btn_border']
        btn_hover = colors['view_toolbar_btn_hover']
        btn_pressed = "#14151a" if is_dark else "#cbd5e1"
        tt_ss = theme_manager.widget_tooltip_stylesheet()
        btn_style = f"""
            QPushButton {{
                background: {btn_bg};
                border: 1px solid {btn_border};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background: {btn_hover};
                border-color: #f59e0b;
            }}
            QPushButton:pressed {{
                background: {btn_pressed};
                border-color: {btn_border};
            }}
            QPushButton:checked {{
                background: #f59e0b;
                border-color: #f59e0b;
            }}
            {tt_ss}
        """
        for btn in (getattr(self, 'btn_split', None), getattr(self, 'btn_crop', None), getattr(self, 'btn_fit', None), getattr(self, 'btn_100', None)):
            if btn:
                btn.setStyleSheet(btn_style)

        if getattr(self, 'crop_toolbar_sep', None):
            sep_color = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(0, 0, 0, 0.12)"
            self.crop_toolbar_sep.setStyleSheet(f"background-color: {sep_color}; border: none;")

        if getattr(self, 'btn_split', None):
            self.btn_split.setIcon(get_icon("compare.png", is_dark=is_dark))
        if getattr(self, 'btn_crop', None):
            self.btn_crop.setIcon(get_icon("crop.png", is_dark=is_dark))
        if getattr(self, 'btn_fit', None):
            self.btn_fit.setIcon(get_icon("fit_window.png", is_dark=is_dark))
        if getattr(self, 'btn_100', None):
            self.btn_100.setIcon(get_icon("zoom_100.png", is_dark=is_dark))
        if getattr(self, 'act_crop', None):
            self.act_crop.setIcon(get_icon("crop.png", is_dark=is_dark))

    def _update_optics_ui_style(self):
        is_dark = theme_manager.is_dark()
        tt_ss = theme_manager.widget_tooltip_stylesheet()
        chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")

        chk_ss = get_standard_checkbox_style(is_dark)

        for chk in (getattr(self, 'chk_remove_ca', None), getattr(self, 'chk_lens_profile_enabled', None),
                    getattr(self, 'chk_push_under', None)):
            if chk:
                chk.setStyleSheet(chk_ss)

        # Tab button stylesheet matching Preferences Dialog tabs style
        tab_bg = "#121317" if is_dark else "#f1f5f9"
        tab_normal = "#94a3b8" if is_dark else "#64748b"
        tab_hover = "#ffffff" if is_dark else "#1e293b"
        tab_hover_bg = "#1c1d25" if is_dark else "#e2e8f0"
        tab_checked_bg = "#181920" if is_dark else "#ffffff"
        tab_checked_fg = "#f59e0b"

        tab_ss = f"""
            QPushButton#lensTabBtn {{
                background: {tab_bg};
                border: none;
                border-bottom: 2px solid transparent;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                color: {tab_normal};
                font-size: 11px;
                font-weight: 500;
                padding: 4px 12px;
            }}
            QPushButton#lensTabBtn:hover:!checked {{
                color: {tab_hover};
                background: {tab_hover_bg};
                border-bottom: 2px solid transparent;
            }}
            QPushButton#lensTabBtn:checked {{
                background: {tab_checked_bg};
                border: none;
                border-bottom: 2px solid #f59e0b;
                color: {tab_checked_fg};
                font-weight: bold;
            }}
            {tt_ss}
        """
        for btn in (getattr(self, 'btn_lens_tab_profile', None), getattr(self, 'btn_lens_tab_manual', None)):
            if btn:
                btn.setStyleSheet(tab_ss)

        if getattr(self, 'lens_tab_frame', None):
            line_col = "#282a36" if is_dark else "#cbd5e1"
            self.lens_tab_frame.setStyleSheet(f"QFrame#lensTabFrame {{ border: none; border-bottom: 1px solid {line_col}; background: transparent; }}\n{tt_ss}")

        # Sub-header labels
        header_color = "#71717a" if is_dark else "#64748b"
        header_ss = f"""
            QLabel#subSectionHeader {{
                color: {header_color};
                font-size: 11px;
                font-weight: 600;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
                padding: 6px 0px 2px 0px;
                border: none;
                background: transparent;
            }}
            {tt_ss}
        """
        for lbl in (getattr(self, 'lbl_profile_header', None), getattr(self, 'lbl_amount_header', None), getattr(self, 'lbl_vignette_header', None), getattr(self, 'lbl_diff_header', None)):
            if lbl:
                lbl.setStyleSheet(header_ss)

        # Container panels for profile and manual
        if getattr(self, 'panel_lens_profile', None):
            self.panel_lens_profile.setStyleSheet(f"background: transparent; border: none;\n{tt_ss}")
        if getattr(self, 'panel_lens_manual', None):
            self.panel_lens_manual.setStyleSheet(f"background: transparent; border: none;\n{tt_ss}")

        # Divider line
        div_col = "#262834" if is_dark else "#e2e8f0"
        if getattr(self, 'optics_section_divider', None):
            self.optics_section_divider.setStyleSheet(f"background-color: {div_col}; border: none;")

    def _update_welcome_view_style(self):
        is_dark = theme_manager.is_dark()
        colors = theme_manager.get_colors()
        if hasattr(self, 'welcome_widget'):
            self.welcome_widget.setStyleSheet(f"background: {colors['app_bg']};")
        if hasattr(self, 'dropzone'):
            self._set_dropzone_active(False)
        if hasattr(self, 'lbl_drop'):
            self.lbl_drop.setStyleSheet(f"color: {colors['text_primary']}; font-size: 16px; font-weight: 600;")
        if hasattr(self, 'lbl_sub'):
            self.lbl_sub.setStyleSheet(f"color: {colors['text_muted']}; font-size: 12px;")
        if hasattr(self, 'btn_import_folder'):
            folder_bg = "#252732" if is_dark else "#ffffff"
            folder_fg = "#e2e8f0" if is_dark else "#1e293b"
            folder_border = "#383c4c" if is_dark else "#cbd5e1"
            folder_hover = "#303342" if is_dark else "#f1f4f9"
            folder_pressed = "#1e2029" if is_dark else "#e2e8f0"
            self.btn_import_folder.setStyleSheet(f"""
                QPushButton {{
                    background: {folder_bg};
                    color: {folder_fg};
                    font-size: 13px;
                    font-weight: 500;
                    border: 1px solid {folder_border};
                    border-radius: 6px;
                }}
                QPushButton:hover {{
                    background: {folder_hover};
                    border-color: #555b70;
                }}
                QPushButton:pressed {{
                    background: {folder_pressed};
                    border-color: {folder_border};
                }}
            """)

    def _init_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("rootContainer")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Custom Title Bar (Item 3, 7: Seamless drag, rounded top corners)
        self.title_bar = DraggableTitleBar(self)
        self.title_bar.setFixedHeight(38)
        self._update_root_style(False)
        tb_layout = QHBoxLayout(self.title_bar)
        tb_layout.setContentsMargins(12, 0, 0, 0)
        tb_layout.setSpacing(10)

        # App Icon
        self.lbl_title_icon = QLabel()
        self.lbl_title_icon.setFixedSize(22, 22)
        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            pix = QPixmap(icon_path).scaled(22, 22, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_title_icon.setPixmap(pix)
        tb_layout.addWidget(self.lbl_title_icon, 0, Qt.AlignmentFlag.AlignVCenter)

        # Menu Bar (靠左紧跟应用图标)
        self.menu_bar = QMenuBar()
        self.menu_bar.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self.menu_bar.setStyleSheet("""
            QMenuBar {
                background: transparent;
                color: #e2e8f0;
                font-size: 12px;
                padding-left: 4px;
                border: none;
                margin: 0px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 9px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: #232530;
                color: #ffffff;
            }
            QMenuBar::item:disabled {
                color: #4b5563;
                background: transparent;
            }
            QMenu {
                background: #18191f;
                color: #e2e8f0;
                border: 1px solid #2f3240;
                border-radius: 6px;
                padding: 4px;
                font-size: 12px;
            }
            QMenu::icon {
                padding-left: 5px;
                width: 14px;
                height: 14px;
            }
            QMenu::item {
                padding: 5px 18px 5px 4px;
                margin: 1px 4px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #f59e0b;
                color: #111113;
                font-weight: bold;
            }
            QMenu::item:disabled {
                color: #4b5563;
                background: transparent;
            }
            QMenu::item:disabled:selected {
                color: #4b5563;
                background: transparent;
            }
            QMenu::separator {
                height: 1px;
                background: #2a2d3a;
                margin: 4px 6px;
            }
        """)
        tb_layout.addWidget(self.menu_bar, 0, Qt.AlignmentFlag.AlignVCenter)

        tb_layout.addStretch(1)

        # App Brand Title (居中放置，鼠标穿透可拖动窗口)
        self.lbl_brand = QLabel("SpektraDarkroom")
        self.lbl_brand.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.lbl_brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_brand.setStyleSheet("color: #f59e0b; font-size: 13px; font-weight: bold; letter-spacing: 0.5px; border: none;")
        tb_layout.addWidget(self.lbl_brand, 0, Qt.AlignmentFlag.AlignVCenter)

        tb_layout.addStretch(1)

        # Amber Export Button (Scans / Ingestion)
        self.btn_title_export = QPushButton()
        self.btn_title_export.setFixedHeight(26)
        self.btn_title_export.setFixedWidth(68)
        self.btn_title_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_title_export.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #d97706;
            }
            QPushButton:pressed {
                background: #b45309;
            }
        """)

        btn_export_layout = QHBoxLayout(self.btn_title_export)
        btn_export_layout.setContentsMargins(0, 0, 1, 0)
        btn_export_layout.setSpacing(0)

        self.lbl_title_export_icon = QLabel()
        self.lbl_title_export_icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.lbl_title_export_icon.setStyleSheet("background: transparent; border: none;")
        save_icon = os.path.join(get_resource_dir(), "icons", "export_dark.png")
        if os.path.exists(save_icon):
            pix = QPixmap(save_icon).scaled(13, 13, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_title_export_icon.setPixmap(pix)
        self.lbl_title_export_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_title_export_text = QLabel("扫描")
        self.lbl_title_export_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.lbl_title_export_text.setStyleSheet("background: transparent; border: none; color: #111113; font-weight: bold; font-size: 11px;")
        self.lbl_title_export_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_export_layout.addStretch(1)
        btn_export_layout.addWidget(self.lbl_title_export_icon, 0, Qt.AlignmentFlag.AlignCenter)
        btn_export_layout.addSpacing(4)
        btn_export_layout.addWidget(self.lbl_title_export_text, 0, Qt.AlignmentFlag.AlignCenter)
        btn_export_layout.addStretch(1)

        self.btn_title_export.clicked.connect(self.action_export_image)
        self.btn_title_export.setVisible(False)
        tb_layout.addWidget(self.btn_title_export, 0, Qt.AlignmentFlag.AlignVCenter)
        tb_layout.addSpacing(6)

        # Caption Buttons (Win11 Vector) contiguous box extending to top of titlebar
        caption_box = QWidget()
        caption_box.setFixedHeight(38)
        caption_layout = QHBoxLayout(caption_box)
        caption_layout.setContentsMargins(0, 0, 0, 0)
        caption_layout.setSpacing(0)

        self.btn_min = WindowCaptionButton("min", self)
        self.btn_min.clicked.connect(self.showMinimized)
        caption_layout.addWidget(self.btn_min)

        self.btn_max = WindowCaptionButton("max", self)
        self.btn_max.clicked.connect(self.toggle_maximize)
        caption_layout.addWidget(self.btn_max)

        self.btn_close = WindowCaptionButton("close", self)
        self.btn_close.clicked.connect(self.close)
        caption_layout.addWidget(self.btn_close)

        tb_layout.addWidget(caption_box, 0, Qt.AlignmentFlag.AlignTop)

        root_layout.addWidget(self.title_bar)

        # 2. Main Stack (Welcome View & Editor View)
        self.stack = FadingStackedWidget()
        root_layout.addWidget(self.stack, 1)

        self._init_welcome_view()
        self._init_editor_view()

    def _init_welcome_view(self):
        self.welcome_widget = QWidget()
        self.welcome_widget.setStyleSheet(f"background: {theme_manager.color('app_bg')};")
        v_layout = QVBoxLayout(self.welcome_widget)
        v_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        c_widget = QWidget()
        c_layout = QVBoxLayout(c_widget)
        c_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_layout.setSpacing(24)

        # Single Clean Dropzone Card (Item 1)
        self.dropzone = QFrame()
        self.dropzone.setObjectName("welcomeDropzone")
        self.dropzone.setFixedSize(580, 260)
        self.dropzone.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_dropzone_active(False)
        drop_layout = QVBoxLayout(self.dropzone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.setSpacing(14)

        # Crisp Light Film Canister Icon
        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            lbl_icon = QLabel()
            lbl_icon.setFixedSize(56, 56)
            pix = QPixmap(icon_path).scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_icon.setPixmap(pix)
            drop_layout.addWidget(lbl_icon, 0, Qt.AlignmentFlag.AlignCenter)

        self.lbl_drop = QLabel("点击或拖入照片 / RAW 负片文件")
        self.lbl_drop.setStyleSheet(f"color: {theme_manager.color('text_primary')}; font-size: 16px; font-weight: 600;")
        self.lbl_drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self.lbl_drop)

        self.lbl_sub = QLabel("支持格式: ARW · CR2 · CR3 · NEF · RAF · DNG · TIFF · PNG")
        self.lbl_sub.setStyleSheet(f"color: {theme_manager.color('text_muted')}; font-size: 12px;")
        self.lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self.lbl_sub)

        self.dropzone.mousePressEvent = lambda e: self.action_open_files()
        c_layout.addWidget(self.dropzone)

        # Import Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)

        btn_import_file = QPushButton("导入文件")
        btn_import_file.setFixedSize(160, 38)
        btn_import_file.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import_file.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #111;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #d97706;
            }
            QPushButton:pressed {
                background: #b45309;
            }
        """)
        btn_import_file.clicked.connect(self.action_open_files)
        btn_layout.addWidget(btn_import_file)

        self.btn_import_folder = QPushButton("导入文件夹")
        self.btn_import_folder.setFixedSize(160, 38)
        self.btn_import_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import_folder.setStyleSheet("""
            QPushButton {
                background: #252732;
                color: #e2e8f0;
                font-size: 13px;
                font-weight: 500;
                border: 1px solid #383c4c;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #303342;
                border-color: #555b70;
            }
            QPushButton:pressed {
                background: #1e2029;
                border-color: #383c4c;
            }
        """)
        self.btn_import_folder.clicked.connect(self.action_open_folder)
        btn_layout.addWidget(self.btn_import_folder)

        c_layout.addLayout(btn_layout)
        v_layout.addWidget(c_widget)
        self.stack.addWidget(self.welcome_widget)

    def _init_editor_view(self):
        self.editor_widget = QWidget()
        self.editor_widget.setStyleSheet(f"background: {theme_manager.color('app_bg')};")
        e_layout = QVBoxLayout(self.editor_widget)
        
        # Item 2: Margins and Padding around work area
        e_layout.setContentsMargins(6, 6, 6, 6)
        e_layout.setSpacing(6)

        # Item 2 & 11: 3-Column Splitter with strict limits
        self.main_splitter = FadingSplitter(Qt.Orientation.Horizontal)

        # -------------------------------------------------------------
        # 1. Left Sidebar: 3-Tier Resizable Splitter (Films, Papers, EXIF)
        # -------------------------------------------------------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(160)
        left_panel.setMaximumWidth(460)
        lp_layout = QVBoxLayout(left_panel)
        # 不额外内缩：卡片/托盘的边缘就是 splitter 的边缘，这样左栏的上下左右和
        # 画布、底片槽、窗口边距才对得齐（曾经内缩 4px，四处各差 4px）
        lp_layout.setContentsMargins(0, 0, 0, 0)
        lp_layout.setSpacing(4)

        # 3-tier vertical splitter (Delicate amber hairline fading at both ends)
        self.left_splitter = FadingSplitter(Qt.Orientation.Vertical)

        # Sub-window 1: 胶卷型号 (Films)
        film_subwin = QWidget()
        film_layout = QVBoxLayout(film_subwin)
        film_layout.setContentsMargins(0, 0, 0, 0)
        film_layout.setSpacing(0)

        # 托盘包住整个窗格（标题/搜索/分类胶囊 + 卡片列表）。只包卡片列表的话，上面是
        # 面板底、下面是托盘底，中间会横着一道换底色的接缝。
        self.film_tray = QFrame()
        self.film_tray.setObjectName("filmTray")
        tray_layout = QVBoxLayout(self.film_tray)
        tray_layout.setContentsMargins(6, 6, 6, 6)
        tray_layout.setSpacing(6)

        self.film_header = FilmFilterHeader()
        # 这块必须显式透明：它自己会按 palette 的 Window 色铺一层底（暗 #121316 / 亮
        # #ebedf2）。以前它直接坐在面板底色上，看不出来；挪进托盘后就在标题和分类
        # 胶囊后面留下一块和托盘不同色的方块。
        self.film_header.setStyleSheet("background: transparent;")
        self.lbl_film_title = self.film_header.lbl_title
        self.film_header.searchChanged.connect(self._on_film_search_changed)
        self.film_header.categoryChanged.connect(self._on_film_category_changed)
        tray_layout.addWidget(self.film_header)

        self.film_scroll = SmoothScrollArea()
        self.film_scroll.setWidgetResizable(True)
        self.film_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.film_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.film_container = QWidget()
        self.film_grid = QGridLayout(self.film_container)
        self.film_grid.setContentsMargins(4, 4, 4, 4)
        self.film_grid.setHorizontalSpacing(6)
        self.film_grid.setVerticalSpacing(6)
        self.film_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.film_scroll.setWidget(self.film_container)

        # 卡片背后不能再有一层灰：QWidget 默认会按 palette 的 Window 色铺底，托盘的白底
        # 只在 padding 那一圈可见，中间整块是 app_bg。容器和视口都要显式透明。
        self.film_container.setStyleSheet("background: transparent;")
        if self.film_scroll.viewport():
            self.film_scroll.viewport().setStyleSheet("background: transparent;")
            self.film_scroll.viewport().setAutoFillBackground(False)

        self.film_scroll.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.film_scroll.customContextMenuRequested.connect(self._on_film_blank_context_menu)
        self.film_container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.film_container.customContextMenuRequested.connect(self._on_film_blank_context_menu)

        self._film_vp_filter = ViewportResizeFilter(self.reflow_grids, self)
        self.film_scroll.installEventFilter(self._film_vp_filter)

        # 有了托盘这一层，上下边缘渐隐才有地方"化开"，视觉上也和右侧参数窗格对齐
        tray_layout.addWidget(self.film_scroll, 1)
        self._apply_tray_style(self.film_tray, self.film_scroll)
        film_layout.addWidget(self.film_tray, 1)

        self._populate_film_grid()
        self.left_splitter.addWidget(film_subwin)

        # Sub-window 2: 放大相纸 (Papers)
        paper_subwin = QWidget()
        paper_layout = QVBoxLayout(paper_subwin)
        paper_layout.setContentsMargins(0, 0, 0, 0)
        paper_layout.setSpacing(4)

        p_header = QHBoxLayout()
        p_header.setContentsMargins(4, 2, 4, 0)
        self.lbl_paper_title = QLabel("放大相纸 (Papers)")
        self.lbl_paper_title.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")
        p_header.addWidget(self.lbl_paper_title)
        p_header.addStretch()
        paper_layout.addLayout(p_header)


        self.paper_scroll = SmoothScrollArea()
        self.paper_scroll.setWidgetResizable(True)
        self.paper_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.paper_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.paper_container = QWidget()
        self.paper_grid = QGridLayout(self.paper_container)
        self.paper_grid.setContentsMargins(4, 4, 4, 4)
        self.paper_grid.setHorizontalSpacing(6)
        self.paper_grid.setVerticalSpacing(6)
        self.paper_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.paper_scroll.setWidget(self.paper_container)

        self.paper_scroll.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.paper_scroll.customContextMenuRequested.connect(self._on_paper_blank_context_menu)
        self.paper_container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.paper_container.customContextMenuRequested.connect(self._on_paper_blank_context_menu)

        self._paper_vp_filter = ViewportResizeFilter(self.reflow_grids, self)
        self.paper_scroll.installEventFilter(self._paper_vp_filter)

        paper_layout.addWidget(self.paper_scroll, 1)

        self._populate_paper_grid()
        # 相纸卡片墙已迁到右侧“放大机与相纸显影”面板的下拉框，这里不再入布局。
        # 仍然保留实例引用：paper_cards / paper_grid 被缩略图与状态同步逻辑复用，
        # 且 lbl_paper_title 会在换主题时被访问，无父无引用的 QWidget 会先被回收。
        self.paper_subwin = paper_subwin
        paper_subwin.setVisible(False)

        # Sub-window 3: 元数据属性 (EXIF)
        exif_subwin = QWidget()
        exif_layout = QVBoxLayout(exif_subwin)
        exif_layout.setContentsMargins(0, 0, 0, 0)
        exif_layout.setSpacing(0)

        self.exif_widget = CompactExifWidget()
        exif_layout.addWidget(self.exif_widget, 1)

        # Keep the pane from being dragged taller than its content: below the last row there
        # is nothing to show, only a card edge stretching into empty space. The cap follows
        # the wrapped rows, so a long lens name still gets its second line.
        self.exif_subwin = exif_subwin
        self.exif_widget.contentResized.connect(self._sync_exif_max_height)
        QTimer.singleShot(0, self._sync_exif_max_height)

        self.left_splitter.addWidget(exif_subwin)

        self.left_splitter.setStretchFactor(0, 4)
        self.left_splitter.setStretchFactor(1, 3)
        self.left_splitter.setSizes([300, 200])

        lp_layout.addWidget(self.left_splitter)
        self.main_splitter.addWidget(left_panel)
        left_panel.resizeEvent = lambda ev: (QWidget.resizeEvent(left_panel, ev), self.reflow_grids())

        # -------------------------------------------------------------
        # 2. Center Column: Canvas + View Toolbar + Filmstrip (Vertical Splitter)
        # -------------------------------------------------------------
        self.center_splitter = FadingSplitter(Qt.Orientation.Vertical)
        self.center_splitter.setMinimumWidth(450)

        # Canvas with View Toolbar
        canvas_container = QWidget()
        cc_layout = QVBoxLayout(canvas_container)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        cc_layout.setSpacing(0)
        cc_layout.addWidget(self.canvas, 1)

        # View Toolbar (Item 6: Status on Left, Buttons on Right)
        # View Toolbar (Status on Left, Buttons on Right - Narrow 28px, clean borderless background)
        self.view_toolbar = CenteredViewToolbar()
        self.view_toolbar.setFixedHeight(28)
        self.view_toolbar.setStyleSheet(f"background: {theme_manager.color('view_toolbar_bg')}; border: none;")
        tb_layout = QHBoxLayout(self.view_toolbar)
        # Symmetric vertical margins: the old 2/1 split pushed the whole row half a pixel up,
        # which is what made the right-hand capsule sit slightly above the filename on the left.
        tb_layout.setContentsMargins(10, 0, 10, 0)
        tb_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        tb_layout.setSpacing(5)

        # Left: Dynamic Unified Cross-Fading Status Slot (Unifies photo filename & zoom with transient notifications)
        self.status_slot = StatusSlotWidget(self.view_toolbar)
        self.lbl_photo_info = self.status_slot.lbl_photo_info
        self.status_label = self.status_slot.status_label
        self.spinner = self.status_slot.spinner
        self.btn_cancel_import = self.status_slot.btn_cancel_import
        self.btn_cancel_import.clicked.connect(lambda: self._cancel_active_import(notify_user=True))
        self.status_label.mousePressEvent = self._on_status_label_clicked
        tb_layout.addWidget(self.status_slot)

        tb_layout.addStretch(1)

        # Center: Canvas Star Rating Widget (placed in CenteredViewToolbar at absolute geometric center)
        self.canvas_rating = CanvasRatingWidget(self.view_toolbar)
        self.canvas_rating.setVisible(False)
        self.canvas_rating.ratingChanged.connect(self._on_canvas_rating_changed)
        self.view_toolbar.set_center_widget(self.canvas_rating)
        self.view_toolbar.set_status_slot(self.status_slot)
        self.view_toolbar.set_status_label(self.status_label)
        # Right: Dedicated Export Status Prompt (Pure text + underline, matching left status bar)
        self.export_status_container = QWidget()
        self.export_status_container.setObjectName("export_status_container")
        self.export_status_container.setVisible(False)
        self.export_status_container.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_status_container.setToolTip("点击查看扫描队列详情（快捷键 Ctrl+J）")
        self.export_status_container.mousePressEvent = self._on_export_status_clicked
        export_layout = QHBoxLayout(self.export_status_container)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        export_layout.setSpacing(6)

        self.export_spinner = SpinnerWidget(size=12)
        export_layout.addWidget(self.export_spinner)

        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setObjectName("lbl_export_status")
        self.lbl_export_status.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.lbl_export_status.setStyleSheet(
            "QLabel#lbl_export_status { color: #f59e0b; font-size: 11px; "
            "text-decoration: underline; background: transparent; border: none; padding: 0; }")
        export_layout.addWidget(self.lbl_export_status)

        self.export_status_container.setStyleSheet("""
            QWidget#export_status_container {
                background: transparent;
                border: none;
                padding: 0;
            }
        """)
        tb_layout.addWidget(self.export_status_container)
        self.view_toolbar.set_right_widget(self.export_status_container)

        self.export_sep = QLabel("│")
        self.export_sep.setObjectName("export_sep")
        self.export_sep.setFixedWidth(12)
        self.export_sep.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter)
        self.export_sep.setStyleSheet("QLabel#export_sep { color: #2e3444; font-size: 11px; padding: 0; }")
        self.export_sep.setVisible(False)
        tb_layout.addWidget(self.export_sep)

        # Right: View Control Buttons (Item 6 & 9)
        btn_style = """
            QPushButton {
                background: #1c1d24;
                border: 1px solid #333644;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #2a2c38;
                border-color: #555b70;
            }
            QPushButton:pressed {
                background: #14151a;
                border-color: #333644;
            }
            QPushButton:checked {
                background: #f59e0b;
                border-color: #f59e0b;
            }
        """

        self.btn_crop = QPushButton()
        self.btn_crop.setFixedSize(28, 22)
        self.btn_crop.setIcon(get_icon("crop.png"))
        self.btn_crop.setIconSize(QSize(14, 14))
        self.btn_crop.setCheckable(True)
        self.btn_crop.setToolTip("裁切与校平（快捷键 C）")
        self.btn_crop.setStyleSheet(btn_style)
        self.btn_crop.clicked.connect(self.toggle_crop_mode)
        tb_layout.addWidget(self.btn_crop)

        self.crop_toolbar_sep = QFrame()
        self.crop_toolbar_sep.setFrameShape(QFrame.Shape.VLine)
        self.crop_toolbar_sep.setFrameShadow(QFrame.Shadow.Plain)
        self.crop_toolbar_sep.setFixedWidth(1)
        self.crop_toolbar_sep.setFixedHeight(14)
        is_dark_init = theme_manager.is_dark()
        sep_color = "rgba(255, 255, 255, 0.12)" if is_dark_init else "rgba(0, 0, 0, 0.12)"
        self.crop_toolbar_sep.setStyleSheet(f"background-color: {sep_color}; border: none;")
        tb_layout.addWidget(self.crop_toolbar_sep)

        self.btn_split = QPushButton(self)
        self.btn_split.setFixedSize(28, 22)
        self.btn_split.setIcon(get_icon("compare.png"))
        self.btn_split.setIconSize(QSize(14, 14))
        self.btn_split.setCheckable(True)
        self.btn_split.setToolTip("对比视图（快捷键 \\）：点击切换分屏对比")
        self.btn_split.setStyleSheet(btn_style)
        self.btn_split.clicked.connect(self.toggle_split_view)
        tb_layout.addWidget(self.btn_split)

        self.btn_fit = QPushButton()
        self.btn_fit.setFixedSize(28, 22)
        self.btn_fit.setIcon(get_icon("fit_window.png"))
        self.btn_fit.setIconSize(QSize(14, 14))
        self.btn_fit.setCheckable(True)
        self.btn_fit.setToolTip("适应图像全貌（快捷键 Ctrl+0）")
        self.btn_fit.setStyleSheet(btn_style)
        self.btn_fit.clicked.connect(self._on_btn_fit_clicked)
        tb_layout.addWidget(self.btn_fit)

        self.btn_100 = QPushButton()
        self.btn_100.setFixedSize(28, 22)
        self.btn_100.setIcon(get_icon("zoom_100.png"))
        self.btn_100.setIconSize(QSize(14, 14))
        self.btn_100.setCheckable(True)
        self.btn_100.setToolTip("1:1 实际像素（快捷键 Ctrl+1）")
        self.btn_100.setStyleSheet(btn_style)
        self.btn_100.clicked.connect(self._on_btn_100_clicked)
        tb_layout.addWidget(self.btn_100)

        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        cc_layout.addWidget(self.view_toolbar)
        self.center_splitter.addWidget(canvas_container)

        # Bottom 35mm Filmstrip (Item 8, 16)
        self.filmstrip = FilmstripWidget()
        self.filmstrip.photoSelected.connect(self.on_switch_photo)
        self.filmstrip.photoRemoved.connect(self.on_remove_photo)
        self.filmstrip.photosRemoved.connect(self.on_remove_multiple_photos)
        self.filmstrip.photosDeleted.connect(self.on_delete_multiple_photos)
        self.filmstrip.clearRequested.connect(self.action_clear_photos)
        self.filmstrip.clearSdcRequested.connect(self.on_clear_sdc_config)
        self.filmstrip.clearCropRequested.connect(self.on_clear_crop_photos)
        self.filmstrip.batchExportRequested.connect(self._batch_export_photos)
        self.filmstrip.quickExportRequested.connect(self.action_quick_export)
        self.filmstrip.addRequested.connect(self.action_open_files)
        self.filmstrip.selectionChanged.connect(self._on_filmstrip_selection_changed)
        self.filmstrip.stepPhotoRequested.connect(self._on_step_photo)
        self.filmstrip.navigateFirstRequested.connect(self._on_navigate_first_photo)
        self.filmstrip.navigateLastRequested.connect(self._on_navigate_last_photo)
        self.filmstrip.sortRequested.connect(self._sort_photos_by)
        self.filmstrip.copyParamsRequested.connect(self.action_copy_params)
        self.filmstrip.pasteParamsRequested.connect(self.action_paste_params)
        self.filmstrip.photoRated.connect(self._on_filmstrip_photo_rated)
        self.filmstrip.batchPhotoRated.connect(self._on_filmstrip_batch_rated)
        self.filmstrip.filterChanged.connect(self._on_filter_changed)
        self.center_splitter.addWidget(self.filmstrip)

        self.center_splitter.setStretchFactor(0, 1)
        self.center_splitter.setStretchFactor(1, 0)
        self.center_splitter.setSizes([600, 100])

        self.main_splitter.addWidget(self.center_splitter)

        # -------------------------------------------------------------
        # 3. Right Sidebar: Pinned Histogram + Adjustments Scroll
        # -------------------------------------------------------------
        self.sidebar_container = QWidget()
        self.sidebar_container.setMinimumWidth(260)
        self.sidebar_container.setMaximumWidth(420)
        self.sidebar_container.setStyleSheet(f"background: {theme_manager.color('app_bg')};")
        sb_outer_layout = QVBoxLayout(self.sidebar_container)
        # Right margin is 4 (not 8) so the 6px scrollbar sits centred in the 14px gutter between
        # the parameter cards and the panel edge (4 | 6 | 4) instead of touching the cards.
        # 不留内缩：右栏卡片的顶边要对齐左侧胶卷卡片，底边对齐底片栏
        sb_outer_layout.setContentsMargins(0, 0, 0, 0)
        sb_outer_layout.setSpacing(6)

        # Top Pinned Real-time RGB & Luma Histogram (始终置顶不跟随滚动)
        self.histogram_widget = HistogramWidget(self)
        sb_outer_layout.addWidget(self.histogram_widget)
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.setInterval(40)
        self._hist_timer.timeout.connect(self._compute_and_update_histogram)

        # Compact EXIF Summary Bar (上下居中，左右对齐)
        self.histogram_exif_bar = HistogramExifBar(self)
        sb_outer_layout.addWidget(self.histogram_exif_bar)

        sidebar_scroll = SmoothScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # 渐隐带只铺在参数卡片宽度内：卡片右侧留了 4px 滚动条缝、卡片本身是 R6 圆角，
        # 满宽铺会糊出卡片边缘，在留白上留一条白带
        sidebar_scroll.set_fade_geometry(inset_left=0, inset_right=4, radius=6.0)
        self.sidebar_scroll = sidebar_scroll

        self.sidebar_content = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_content)
        # 4px on the right keeps the cards' absolute right edge where it was while the scrollbar
        # (now 4px closer to the panel edge) gains 4px of breathing room on its left.
        self.sidebar_layout.setContentsMargins(0, 0, 4, 0)
        self.sidebar_layout.setSpacing(6)

        # Sections (Comprehensive Physical Analog Pipeline)
        self._init_exposure_section()
        self._init_optics_section()
        self._init_chemistry_section()
        self._init_enlarger_section()
        self._init_halation_section()
        self.sidebar_layout.addStretch(1)

        sidebar_scroll.setWidget(self.sidebar_content)

        self.sidebar_stack = QStackedWidget()
        self.sidebar_stack.addWidget(sidebar_scroll)
        self.crop_sidebar = CropSidebarWidget(self.canvas.crop_overlay, self)
        self.sidebar_stack.addWidget(self.crop_sidebar)
        sb_outer_layout.addWidget(self.sidebar_stack, 1)

        # 见模块顶 PARAM_TOOLTIPS_ENABLED：整棵参数面板（含裁切面板）的气泡静音。
        if not PARAM_TOOLTIPS_ENABLED:
            block_tooltips_under(self.sidebar_content)
            block_tooltips_under(self.crop_sidebar)

        if hasattr(self.canvas, 'crop_overlay'):
            self.canvas.crop_overlay.cropApplied.connect(self._on_crop_applied)
            self.canvas.crop_overlay.cropCancelled.connect(self._on_crop_cancelled)
            self.canvas.crop_overlay.cropChanged.connect(self._on_crop_live)

        self.main_splitter.addWidget(self.sidebar_container)
        self.main_splitter.setCollapsible(2, False)
        # 参数面板那条竖缝的指示线贴在**画布视口右侧**（即缝的左缘），而不是参数面板左侧：
        # 视觉上指示线属于它正在标记的边界，画布与参数面板之间读作"画布的右边界"。
        self.main_splitter.set_indicator_side(1, "leading")

        # Splitter proportion defaults: Left panel 320px (comfortably fits 3 columns), Center 800px, Right 320px
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([320, 800, 320])
        self.main_splitter.splitterMoved.connect(lambda pos, idx: self.reflow_grids())

        e_layout.addWidget(self.main_splitter)
        self.stack.addWidget(self.editor_widget)
        self.stack.currentChanged.connect(self._on_stack_page_changed)

        # _restore_saved_state() runs before any of the splitters exist, and the two default
        # setSizes() calls above would clobber it anyway, so the stored layout has to be
        # re-applied once the panes are live. The zero-delay shot lands after the first
        # layout pass, when the panes finally have real widths to scale against.
        self._restore_splitter_layout()
        QTimer.singleShot(0, self._restore_splitter_layout)

        # Persist on drag settle as well as on close: closeEvent already does it, but any exit
        # that skips it (crash, task manager, power loss) brought the defaults back.
        self._splitter_save_timer = QTimer(self)
        self._splitter_save_timer.setSingleShot(True)
        self._splitter_save_timer.setInterval(500)
        self._splitter_save_timer.timeout.connect(self._persist_splitter_layout)
        for _splitter in (self.main_splitter, self.left_splitter, self.center_splitter):
            _splitter.splitterMoved.connect(self._schedule_splitter_save)

    def _schedule_splitter_save(self, *_args):
        timer = getattr(self, "_splitter_save_timer", None)
        if timer is not None:
            timer.start()

    def _persist_splitter_layout(self):
        """Write the pane sizes only, leaving geometry and maximise state untouched."""
        if not hasattr(self, "main_splitter"):
            return
        stack = getattr(self, "stack", None)
        if stack is not None and stack.currentIndex() != 1:
            return                       # the import page has its own layout
        if sum(self.main_splitter.sizes()) <= 500:
            return                       # not laid out yet
        try:
            cfg = config_manager.load_config()
            win = cfg.setdefault("window", {})
            win["splitter_main"] = [int(s) for s in self.main_splitter.sizes()]
            if hasattr(self, "left_splitter"):
                win["splitter_left"] = [int(s) for s in self.left_splitter.sizes()]
            if hasattr(self, "center_splitter"):
                win["splitter_center"] = [int(s) for s in self.center_splitter.sizes()]
            config_manager.save_config(cfg)
        except Exception as exc:
            logger.debug("splitter layout save failed: %s", exc)

    def _restore_splitter_layout(self):
        """Apply the stored pane sizes to whichever splitters already exist."""
        win_cfg = config_manager.get_window_config() or {}
        for attr, key in (("main_splitter", "splitter_main"),
                          ("left_splitter", "splitter_left"),
                          ("center_splitter", "splitter_center")):
            splitter = getattr(self, attr, None)
            if splitter is not None and key in win_cfg:
                _restore_splitter_sizes(splitter, win_cfg[key])

    def _sync_exif_max_height(self):
        """Pin the EXIF pane to the height of its rows (see CompactExifWidget.natural_height)."""
        if not hasattr(self, "exif_subwin") or not hasattr(self, "exif_widget"):
            return
        try:
            target = self.exif_widget.natural_height()
        except Exception:
            return                      # mid-teardown, or fonts not resolved yet
        if target <= 0:
            return
        if self.exif_subwin.maximumHeight() != target:
            self.exif_subwin.setMaximumHeight(target)

    def _on_stack_page_changed(self, index):
        if index == 1:
            # Editor page activated: restore persisted splitter layout and reflow grids
            self._restore_splitter_layout()
            self.reflow_grids()
            QTimer.singleShot(0, self.reflow_grids)
            QTimer.singleShot(60, self.reflow_grids)

    def reflow_grids(self):
        if getattr(self, "_is_reflowing", False):
            return
        self._is_reflowing = True
        try:
            # Dynamic responsive layout: unified minimum card width 96px to prevent clipping
            min_card_w = 96
            spacing = 6
            margins = 8

            # Helper function for grid reflow with dynamic card resizing
            def _reflow_container(scroll_widget, cards_dict, grid_layout):
                if not scroll_widget or not cards_dict:
                    return
                # Calculate stable width from scroll area itself, reserving 6px for vertical scrollbar
                # This guarantees that card widths and layout do not oscillate when a scrollbar appears or disappears.
                container_w = scroll_widget.width()
                if container_w <= 0:
                    vp = scroll_widget.viewport()
                    container_w = vp.width() if vp else 0
                if container_w <= 0 and hasattr(self, "left_splitter"):
                    container_w = self.left_splitter.width() - 16
                avail_w = max(96, container_w - margins - 6)

                # Strict column thresholds based on min_card_w=96, spacing=6, margins=8:
                # 4 cols: 96 * 4 + 6 * 3 + 8 = 410px
                # 3 cols: 96 * 3 + 6 * 2 + 8 = 308px
                # 2 cols: 96 * 2 + 6 * 1 + 8 = 206px
                if avail_w >= 410:
                    cols = 4
                elif avail_w >= 308:
                    cols = 3
                elif avail_w >= 206:
                    cols = 2
                else:
                    cols = 1

                # Dynamically scale card width to fill row up to 220px max
                target_card_w = int((avail_w - (cols - 1) * spacing) / cols)
                target_card_w = max(min_card_w, min(220, target_card_w))

                visible_cards = [c for c in cards_dict.values() if getattr(c, "is_filter_matched", True)]

                # Check if repack is completely redundant (same visible cards, same columns, same card width)
                current_keys = (tuple(id(c) for c in visible_cards), cols, target_card_w)
                if getattr(grid_layout, "_last_reflow_keys", None) == current_keys:
                    return
                grid_layout._last_reflow_keys = current_keys

                scroll_widget.setUpdatesEnabled(False)
                try:
                    for card in visible_cards:
                        card.update_card_dimensions(target_card_w)

                    # Only rebuild grid layout slots if card membership or column count changed
                    pos_keys = (tuple(id(c) for c in visible_cards), cols)
                    if getattr(grid_layout, "_last_pos_keys", None) != pos_keys:
                        for card in cards_dict.values():
                            grid_layout.removeWidget(card)

                        for i, card in enumerate(visible_cards):
                            grid_layout.addWidget(card, i // cols, i % cols)
                            card.setVisible(True)

                        for card in cards_dict.values():
                            if not getattr(card, "is_filter_matched", True):
                                card.setVisible(False)

                        grid_layout._last_pos_keys = pos_keys

                    total_rows = (len(visible_cards) + cols - 1) // cols if visible_cards else 0
                    for r in range(max(1, total_rows)):
                        grid_layout.setRowStretch(r, 0)
                    grid_layout.setRowStretch(total_rows, 1)
                finally:
                    scroll_widget.setUpdatesEnabled(True)

            if hasattr(self, "film_scroll") and self.film_cards:
                _reflow_container(self.film_scroll, self.film_cards, self.film_grid)

            if hasattr(self, "paper_scroll") and self.paper_cards:
                _reflow_container(self.paper_scroll, self.paper_cards, self.paper_grid)
        finally:
            self._is_reflowing = False

    def _on_film_search_changed(self, query: str):
        self._film_search_query = query.strip().lower()
        self._apply_film_filters()

    def _on_film_category_changed(self, category: str):
        self._film_category_filter = category
        self._apply_film_filters()
        # Each category opens at its own top: the offset used to be inherited, so a short
        # category could open half-way down (or apparently empty).  If the active stock lives
        # in this category, land on it instead -- that is the one row worth scrolling to.
        if hasattr(self, "film_scroll"):
            self.film_scroll.scroll_to(0, animate=False)
            card = self.film_cards.get(getattr(self, "current_film_stock", None))
            if card is not None and getattr(card, "is_filter_matched", False):
                QTimer.singleShot(0, lambda c=card: self.film_scroll.ensureWidgetVisible(c, 0, 12))

    def _apply_film_filters(self):
        query = getattr(self, "_film_search_query", "").strip().lower()
        cat = getattr(self, "_film_category_filter", "all")
        visible_count = 0
        total_count = len(self.film_cards)

        for sid, card in self.film_cards.items():
            info = getattr(card, "stock_info", {})
            model = info.get("channel_model", "color")
            ftype = info.get("type", "negative")
            use = info.get("use", "still")
            badge = (info.get("badge") or "").lower()

            # 1. Category match
            cat_match = False
            if cat == "all":
                cat_match = True
            elif cat == "color":
                cat_match = (model == "color")
            elif cat == "bw":
                cat_match = (model == "bw")
            elif cat == "negative":
                cat_match = (ftype == "negative")
            elif cat == "positive":
                cat_match = (ftype == "positive")
            elif cat == "cine":
                cat_match = (use == "cine" or "电影" in badge or "cinestill" in sid.lower() or "vision3" in sid.lower() or "eterna" in sid.lower())

            # 2. Query match
            query_match = True
            if query:
                s_name = card.stock_name.lower()
                s_desc = (info.get("desc") or "").lower()
                s_id = sid.lower()
                query_match = (query in s_name or query in badge or query in s_desc or query in s_id)

            # Special case for "none" bypass stock
            if sid == "none":
                is_visible = (cat == "all") and (not query or query in "无原生bypass数码")
            else:
                is_visible = cat_match and query_match

            card.is_filter_matched = is_visible
            if card.parent() is not None:
                card.setVisible(is_visible)
            if is_visible:
                visible_count += 1

        if hasattr(self, "film_header"):
            self.film_header.set_title_count(visible_count, total_count)
        elif hasattr(self, "lbl_film_title"):
            self.lbl_film_title.setText(f"胶卷型号 ({visible_count}/{total_count} 卷)")

        self.reflow_grids()
        if hasattr(self, '_film_filter_debounce_timer'):
            self._film_filter_debounce_timer.start(120)

    def _populate_film_grid(self):
        stocks = self.engine.get_film_stocks()
        parent_w = getattr(self, "film_container", None)
        for group in stocks:
            for item in group.get("stocks", group.get("items", [])):
                card = FilmGridCard(item, is_active=(item["id"] == self.current_film_stock), is_custom=False, stock_type="film", parent=parent_w)
                card.selected.connect(self.on_switch_film)
                card.sortRequested.connect(lambda st, sb: self._sort_and_reflow_stocks(st, sb))
                card.sibling_cards = self.film_cards
                self.film_cards[item["id"]] = card
        if hasattr(self, 'film_header'):
            self.film_header.set_title_count(len(self.film_cards), len(self.film_cards))
        elif hasattr(self, 'lbl_film_title'):
            self.lbl_film_title.setText(f"胶卷型号 ({len(self.film_cards)} 卷)")
        self.reflow_grids()

    def _populate_paper_grid(self):
        papers = self.engine.get_paper_stocks()
        parent_w = getattr(self, "paper_container", None)
        for item in papers:
            card = FilmGridCard(item, is_active=(item["id"] == self.current_paper_stock), is_custom=False, stock_type="paper", parent=parent_w)
            card.selected.connect(self.on_switch_paper)
            card.sortRequested.connect(lambda st, sb: self._sort_and_reflow_stocks(st, sb))
            card.sibling_cards = self.paper_cards
            self.paper_cards[item["id"]] = card
        if hasattr(self, 'lbl_paper_title'):
            self.lbl_paper_title.setText(f"放大相纸 ({len(self.paper_cards)} 张)")
        self.reflow_grids()

    def _on_auto_exposure(self):
        """One-tap metered exposure: apply the active photo's metered EV suggestion.

        The meter runs on every decode and lands in ``auto_ev``; nothing applies it
        automatically (the default chain stays at 0 EV, matching the reference engine).
        """
        ev = 0.0
        photo = None
        pid = getattr(self, "active_photo_id", None)
        photos = getattr(self, "photos", [])
        if pid is not None:
            photo = next((p for p in photos if p and p.get("id") == pid), None)
        if photo is None:
            path = getattr(self, "_current_photo_path", None)
            photo = next((p for p in photos if p and p.get("path") == path), None)
        if photo is not None:
            ev = float(photo.get("auto_ev", 0.0) or 0.0)
        if abs(ev) < 1e-4 and getattr(self, "_current_photo_path", None):
            try:
                c_chk = session_cache_manager.get_photo_cache(self._current_photo_path)
                if c_chk:
                    ev = float(c_chk.get("auto_ev", 0.0) or 0.0)
            except Exception:
                pass
        # Last resort: the meter is idempotent, so measuring the already-loaded linear
        # ProPhoto preview here gives the same answer the decode worker would have
        # stored (covers previews cached by builds that predate the auto_ev field).
        if abs(ev) < 1e-4 and photo is not None and photo.get("float_img") is not None:
            try:
                ev = float(SpektraEngine.calculate_auto_exposure_ev(photo["float_img"]))
                photo["auto_ev"] = ev
            except Exception:
                pass
        if abs(ev) < 1e-4:
            self.set_app_status("测光建议 0 EV，无需调整", timeout_ms=2500)
            return
        ev = max(-3.0, min(3.0, ev))
        self.slider_ev.setValue(ev)
        self._on_slider_committed()
        self.set_app_status(f"已按测光建议设置曝光 {ev:+.2f} EV", timeout_ms=2500)

    def _init_exposure_section(self):
        self.sec_exposure = AccordionSection("场景曝光 (Exposure)")
        self.sec_exposure.resetClicked.connect(lambda: self.reset_track("exposure"))

        self.slider_ev = AdobeScrubSlider(
            "曝光", -3.0, 3.0, 0.0, step=0.05, unit=" EV", decimals=2,
            tooltip="线性增减场景总进光量 (EV)",
            track_gradient=[(0.0, QColor("#111317")), (0.5, QColor("#5a6170")), (1.0, QColor("#f8fafc"))],
            has_center_detent=True
        )
        self.slider_ev.valueChanged.connect(lambda v: self._on_slider_live("exposure_ev", v))
        self.slider_ev.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_ev)

        # One-tap metered exposure, mirroring the reference's Auto Adjust: the meter runs on
        # every decode but nothing ships until this button applies its suggestion to the
        # exposure slider (the default chain stays at 0 EV -- see decode_photo_data).
        # Compact "AU" chip in the exposure slider's header row (user request: the old
        # full-width row button was oversized on tablets).
        self.btn_auto_ev = QPushButton("AU")
        self.btn_auto_ev.setToolTip("自动曝光（AU）：按 18% 中灰测光建议设置曝光，不改动其他参数")
        self.btn_auto_ev.setFixedSize(30, 18)
        self.btn_auto_ev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto_ev.setStyleSheet("""
            QPushButton {
                background: #2a2f3a;
                color: #cbd5e1;
                font-size: 10px;
                font-weight: 600;
                border: 1px solid #3b4252;
                border-radius: 4px;
                padding: 0px 2px;
            }
            QPushButton:hover {
                background: #374151;
                color: #f1f5f9;
            }
            QPushButton:pressed {
                background: #1f2937;
            }
        """)
        self.btn_auto_ev.clicked.connect(self._on_auto_exposure)
        self.slider_ev.add_header_widget(self.btn_auto_ev)

        self.slider_highlights = AdobeScrubSlider(
            "高光", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            tooltip="高光强光区域场景线性滚降压缩或提亮",
            track_gradient=[(0.0, QColor("#1e293b")), (0.5, QColor("#64748b")), (1.0, QColor("#f1f5f9"))],
            has_center_detent=True
        )
        self.slider_highlights.valueChanged.connect(lambda v: self._on_slider_live("highlights", v))
        self.slider_highlights.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_highlights)

        self.slider_shadows = AdobeScrubSlider(
            "阴影", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            tooltip="深层暗部阶调平滑提亮或下压，保护纯黑底阶",
            track_gradient=[(0.0, QColor("#090a0f")), (0.5, QColor("#475569")), (1.0, QColor("#94a3b8"))],
            has_center_detent=True
        )
        self.slider_shadows.valueChanged.connect(lambda v: self._on_slider_live("shadows", v))
        self.slider_shadows.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_shadows)

        self.slider_black_lift = AdobeScrubSlider(
            "黑场", -0.25, 0.25, 0.0, step=0.01, unit="", decimals=2,
            tooltip="黑场标定：向左扣除片基灰雾、压实纯黑，向右抬高黑位。"
                    "作用在 LUT 之后的显示域",
            has_center_detent=True
        )
        self.slider_black_lift.valueChanged.connect(lambda v: self._on_slider_live("black_lift", v))
        self.slider_black_lift.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_black_lift)

        self.slider_temp = AdobeScrubSlider(
            "色温", 2500, 9500, 5500, step=25, unit=" K", decimals=0,
            tooltip="调节场景相关色温 (CCT)，左偏冷蓝，右偏暖黄",
            track_gradient=[(0.0, QColor("#3b82f6")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#f59e0b"))],
            has_center_detent=True
        )
        self.slider_temp.valueChanged.connect(lambda v: self._on_slider_live("color_temp", v))
        self.slider_temp.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_temp)

        self.slider_tint = AdobeScrubSlider(
            "色调", 0.70, 1.30, 1.0, step=0.01, unit="", decimals=2,
            tooltip="微调洋红与绿色光色平衡 (Tint)，左偏品红，右偏绿色",
            track_gradient=[(0.0, QColor("#ec4899")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#10b981"))],
            has_center_detent=True
        )
        self.slider_tint.valueChanged.connect(lambda v: self._on_slider_live("tint", v))
        self.slider_tint.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_tint)

        self.combo_format = ComboBoxRow(
            "画幅规格",
            [
                ("35mm 标准全画幅", 35.0),
                ("16mm 独立电影卷", 16.0),
                ("120 中画幅 (6x6)", 60.0),
                ("4x5 大画幅", 100.0),
            ],
            default_val=35.0,
            tooltip="底片画幅尺寸，影响银盐晶体与微米级弥散圈物理相对尺度"
        )
        self.combo_format.valueChanged.connect(lambda v: self._on_combo_changed("film_format_mm", v))
        self.sec_exposure.addWidget(self.combo_format)

        self.sidebar_layout.addWidget(self.sec_exposure)

    def _init_optics_section(self):
        self.sec_optics = AccordionSection("光学 (Optics)")
        self.sec_optics.resetClicked.connect(lambda: self.reset_track("optics"))

        # 1. Segmented Tabs: [ 配置文件 ] [ 手动 ]
        self.lens_tab_frame = QFrame()
        self.lens_tab_frame.setObjectName("lensTabFrame")
        tab_layout = QHBoxLayout(self.lens_tab_frame)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(4)

        self.btn_lens_tab_profile = QPushButton("配置文件")
        self.btn_lens_tab_profile.setObjectName("lensTabBtn")
        self.btn_lens_tab_profile.setCheckable(True)
        self.btn_lens_tab_profile.setChecked(True)
        self.btn_lens_tab_profile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lens_tab_profile.setFixedHeight(26)

        self.btn_lens_tab_manual = QPushButton("手动")
        self.btn_lens_tab_manual.setObjectName("lensTabBtn")
        self.btn_lens_tab_manual.setCheckable(True)
        self.btn_lens_tab_manual.setChecked(False)
        self.btn_lens_tab_manual.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lens_tab_manual.setFixedHeight(26)

        self.lens_tab_group = QButtonGroup(self)
        self.lens_tab_group.setExclusive(True)
        self.lens_tab_group.addButton(self.btn_lens_tab_profile, 0)
        self.lens_tab_group.addButton(self.btn_lens_tab_manual, 1)
        self.lens_tab_group.idClicked.connect(self._on_lens_tab_changed)

        tab_layout.addWidget(self.btn_lens_tab_profile, 1)
        tab_layout.addWidget(self.btn_lens_tab_manual, 1)
        self.sec_optics.addWidget(self.lens_tab_frame)

        # 2. Checkbox: 删除色差 -- with an inline progress marker, and a "未找到" label
        # when the frame has no measurable lateral CA. The measured shifts are deliberately
        # not surfaced: the numbers were noise for every user who never tunes them, and the
        # row is now a single line that says what happened.
        chk_layout = QHBoxLayout()
        chk_layout.setContentsMargins(0, 2, 0, 2)
        self.chk_remove_ca = QCheckBox("删除色差")
        self.chk_remove_ca.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_remove_ca.setToolTip("校正横向色差（紫边/绿边）。\n"
                                      "系数由画面自动测量（LCP 未提供色差模型时）")
        self.chk_remove_ca.toggled.connect(self._on_lens_remove_ca_toggled)
        chk_layout.addWidget(self.chk_remove_ca)

        self.spinner_ca = SpinnerWidget(size=12, auto_hide=True)
        chk_layout.addWidget(self.spinner_ca)

        self.lbl_ca_status = QLabel("")
        self.lbl_ca_status.setStyleSheet(
            f"color: {theme_manager.color('text_muted')}; font-size: 11px;")
        self.lbl_ca_status.setVisible(False)
        chk_layout.addWidget(self.lbl_ca_status)
        chk_layout.addStretch(1)
        self.sec_optics.addLayout(chk_layout)

        # 3. Profile Panel Container
        self.panel_lens_profile = QWidget()
        self.panel_lens_profile.setStyleSheet(f"background: transparent; border: none;\n{theme_manager.widget_tooltip_stylesheet()}")
        self.panel_lens_profile.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        profile_layout = QVBoxLayout(self.panel_lens_profile)
        profile_layout.setContentsMargins(0, 2, 0, 0)
        profile_layout.setSpacing(4)

        chk_prof_layout = QHBoxLayout()
        chk_prof_layout.setContentsMargins(0, 2, 0, 2)
        self.chk_lens_profile_enabled = QCheckBox("使用配置文件校正")
        self.chk_lens_profile_enabled.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_lens_profile_enabled.toggled.connect(self._on_lens_profile_enabled_toggled)
        chk_prof_layout.addWidget(self.chk_lens_profile_enabled)
        profile_layout.addLayout(chk_prof_layout)

        self.combo_lens_setup = ComboBoxRow(
            "来源",
            [
                ("默认 (自动读取 EXIF)", "auto"),
                ("自定", "custom"),
                ("内置 (RAW 内)", "builtin"),
            ],
            default_val="auto",
            tooltip="校正配置来源：自动读取 EXIF、手动选取，"
                    "或使用 RAW 内嵌校正表（索尼 ARW）"
        )
        self.combo_lens_setup.valueChanged.connect(self._on_lens_setup_changed)
        profile_layout.addWidget(self.combo_lens_setup)

        self.lbl_profile_header = QLabel("镜头配置文件")
        self.lbl_profile_header.setObjectName("subSectionHeader")
        profile_layout.addWidget(self.lbl_profile_header)

        # Pre-populate available makes from Adobe LCP database
        from core.lens_correction import LCPDatabase
        db = LCPDatabase.get_instance()
        db.index_installed_profiles()
        makes = db.get_available_makes()
        make_items = [("无", "")] + [(m, m) for m in makes]

        self.combo_lens_make = ComboBoxRow(
            "制造商",
            make_items,
            default_val="",
            tooltip="镜头制造商品牌"
        )
        self.combo_lens_make.valueChanged.connect(self._on_lens_make_changed)
        profile_layout.addWidget(self.combo_lens_make)

        self.combo_lens_model = ComboBoxRow(
            "型号",
            [("无", "")],
            default_val="",
            tooltip="具体镜头型号"
        )
        self.combo_lens_model.valueChanged.connect(self._on_lens_model_changed)
        profile_layout.addWidget(self.combo_lens_model)

        self.combo_lens_profile = ComboBoxRow(
            "配置文件",
            [("无", ""), ("[+ 添加配置文件 (.lcp)...]", "__import_lcp__")],
            default_val="",
            tooltip="匹配的镜头校正配置文件（.lcp）"
        )
        self.combo_lens_profile.valueChanged.connect(self._on_lens_profile_changed)
        profile_layout.addWidget(self.combo_lens_profile)

        self.lbl_amount_header = QLabel("校正量")
        self.lbl_amount_header.setObjectName("subSectionHeader")
        profile_layout.addWidget(self.lbl_amount_header)

        self.slider_lens_distortion = AdobeScrubSlider(
            "扭曲度", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            has_center_detent=True,
            tooltip="畸变校正量（0 为配置文件原值，向左减弱，向右增强）"
        )
        self.slider_lens_distortion.valueChanged.connect(lambda v: self._on_lens_amount_changed("profile_distortion", v))
        self.slider_lens_distortion.sliderReleased.connect(self._on_slider_committed)
        profile_layout.addWidget(self.slider_lens_distortion)

        self.slider_lens_vignette = AdobeScrubSlider(
            "晕影", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            has_center_detent=True,
            tooltip="暗角补偿量（0 为配置文件原值，负值加深，正值消除）"
        )
        self.slider_lens_vignette.valueChanged.connect(lambda v: self._on_lens_amount_changed("profile_vignette", v))
        self.slider_lens_vignette.sliderReleased.connect(self._on_slider_committed)
        profile_layout.addWidget(self.slider_lens_vignette)

        self.sec_optics.addWidget(self.panel_lens_profile)

        # 4. Manual Panel Container
        self.panel_lens_manual = QWidget()
        self.panel_lens_manual.setStyleSheet(f"background: transparent; border: none;\n{theme_manager.widget_tooltip_stylesheet()}")
        self.panel_lens_manual.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.panel_lens_manual.setVisible(False)
        manual_layout = QVBoxLayout(self.panel_lens_manual)
        manual_layout.setContentsMargins(0, 2, 0, 0)
        manual_layout.setSpacing(4)

        self.slider_manual_distortion = AdobeScrubSlider(
            "扭曲度", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            has_center_detent=True,
            tooltip="手动几何畸变校正：负值校正枕形畸变，正值校正桶形畸变"
        )
        self.slider_manual_distortion.valueChanged.connect(lambda v: self._on_lens_amount_changed("manual_distortion", v))
        self.slider_manual_distortion.sliderReleased.connect(self._on_slider_committed)
        manual_layout.addWidget(self.slider_manual_distortion)

        self.lbl_vignette_header = QLabel("晕影")
        self.lbl_vignette_header.setObjectName("subSectionHeader")
        manual_layout.addWidget(self.lbl_vignette_header)

        self.slider_manual_vignette_amount = AdobeScrubSlider(
            "数量", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            has_center_detent=True,
            tooltip="暗角补偿强度：0 为无调整，负值加深暗角，正值增亮边缘"
        )
        self.slider_manual_vignette_amount.valueChanged.connect(lambda v: self._on_lens_amount_changed("manual_vignette_amount", v))
        self.slider_manual_vignette_amount.sliderReleased.connect(self._on_slider_committed)
        manual_layout.addWidget(self.slider_manual_vignette_amount)

        self.slider_manual_vignette_midpoint = AdobeScrubSlider(
            "中点", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            has_center_detent=True,
            tooltip="暗角羽化中点：0 为标准，负值向中心收紧，正值向边缘扩散"
        )
        self.slider_manual_vignette_midpoint.valueChanged.connect(lambda v: self._on_lens_amount_changed("manual_vignette_midpoint", v))
        self.slider_manual_vignette_midpoint.sliderReleased.connect(self._on_slider_committed)
        manual_layout.addWidget(self.slider_manual_vignette_midpoint)

        self.sec_optics.addWidget(self.panel_lens_manual)

        # 5. Divider & Diffusion Section (Moved to Bottom)
        self.optics_section_divider = QFrame()
        self.optics_section_divider.setObjectName("opticsSectionDivider")
        self.optics_section_divider.setFrameShape(QFrame.Shape.HLine)
        self.optics_section_divider.setFrameShadow(QFrame.Shadow.Plain)
        self.optics_section_divider.setFixedHeight(1)
        self.sec_optics.addWidget(self.optics_section_divider)

        self.lbl_diff_header = QLabel("柔光滤镜仿真")
        self.lbl_diff_header.setObjectName("subSectionHeader")
        self.sec_optics.addWidget(self.lbl_diff_header)

        self.combo_diffusion = ComboBoxRow(
            "柔光镜型号",
            [
                ("无", "none"),
                ("黑柔", "black_pro_mist"),
                ("白柔", "pro_mist"),
                ("微光", "glimmerglass"),
                ("电影漫射", "cinebloom"),
            ],
            default_val="none",
            tooltip="电影柔光镜仿真：高光周围形成漫射晕开"
        )
        self.combo_diffusion.valueChanged.connect(lambda v: self._on_combo_changed("diffusion_family", v))
        self.sec_optics.addWidget(self.combo_diffusion)

        self.combo_diff_strength = ComboBoxRow(
            "柔光挡位",
            [
                ("1/8 档", 0.125),
                ("1/4 档", 0.25),
                ("1/2 档", 0.5),
                ("1 档", 1.0),
                ("2 档", 2.0),
            ],
            default_val=0.125,
            tooltip="柔光镜散射密度（1/8～2 档）"
        )
        self.combo_diff_strength.valueChanged.connect(lambda v: self._on_combo_changed("diffusion_strength", v))
        self.slider_diff_strength = self.combo_diff_strength  # Backward compatibility alias
        self.sec_optics.addWidget(self.combo_diff_strength)

        self.slider_diff_warmth = AdobeScrubSlider(
            "光晕冷暖偏置", -1.5, 1.5, 0.0, step=0.05, unit="", decimals=2,
            tooltip="柔光光晕冷暖：正值偏暖金，负值偏冷青",
            track_gradient=[(0.0, QColor("#60a5fa")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#fb923c"))],
            has_center_detent=True
        )
        self.slider_diff_warmth.valueChanged.connect(lambda v: self._on_slider_live("diffusion_warmth", v))
        self.slider_diff_warmth.sliderReleased.connect(self._on_slider_committed)
        self.sec_optics.addWidget(self.slider_diff_warmth)

        # 6. Apply themes & states
        self._update_optics_ui_style()
        self._update_lens_ui_state()
        self._update_diffusion_ui_state()

        self.sidebar_layout.addWidget(self.sec_optics)

    DIR_UNAVAILABLE_TEXT = {
        "bypass": "当前为数码原片旁通模式，DIR 不可用",
        "positive": "当前为反转正片，DIR 不可用",
        "bw": "当前为黑白胶卷，DIR 不可用",
        "no_data": "该胶卷档案未提供 DIR 数据",
    }

    def _init_chemistry_section(self):
        self.sec_chemistry = AccordionSection("胶片化学 (DIR)")
        self.sec_chemistry.resetClicked.connect(lambda: self.reset_track("chemistry"))

        # 迫冲显影 sits here rather than with the print controls because it *is* film chemistry:
        # it rewrites the stock's characteristic curve and base fog, so it invalidates the LUT
        # exactly like the DIR sliders do.
        self.combo_push = ComboBoxRow(
            "迫冲显影",
            [("标准冲洗 (0)", 0)],
            default_val=0,
            tooltip="按胶卷的迫冲表改变特性曲线与灰雾：+1 档相当于提高一档感光度，"
                    "反差上升，暗部颗粒变粗"
        )
        self.combo_push.valueChanged.connect(lambda v: self._on_combo_changed("push_ev", v))
        self.sec_chemistry.addWidget(self.combo_push)

        # Push and underexposure are two different things.  The table is a *development* condition
        # (same exposure, longer development); "shot two stops under" is a capture decision that
        # happens before the film ever reaches the tank.  Linking them by default made every push
        # silently darken the frame -- the curve only gives back a fraction of the missing light --
        # so the link is now explicit and off by default.
        self.chk_push_under = QCheckBox("模拟欠曝（曝光随档位下调）")
        self.chk_push_under.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_push_under.setToolTip(
            "勾选后，选 +N 档会同时把场景曝光下调 N EV，即完整的「欠曝 N 档再迫冲」。\n"
            "显影只能补回一部分（0.2～0.6 EV/档），画面仍比标准冲洗更暗、反差更大。\n"
            "不勾选时只更换显影曲线，曝光不动")
        self.chk_push_under.toggled.connect(self._on_push_underexpose_toggled)
        self.chk_push_under.setStyleSheet(get_standard_checkbox_style())
        self.sec_chemistry.addWidget(self.chk_push_under)

        # 暂时藏起来（见类顶 PUSH_UI_ENABLED）。QWidget 一旦不可见，布局就不占空间，
        # 所以侧栏不会留空档；控件本身仍然存在，`_update_push_ui_state` / 会话恢复 /
        # 旧 .sdc 的 push_ev 全部照旧工作 —— 只是用户点不到，也就不会被"点了没反应"误导。
        if not self.PUSH_UI_ENABLED:
            for _w in (self.combo_push, self.chk_push_under):
                _w.setVisible(False)

        # DIR 只有一个自由度。fotufilm 上游只有 COUPLER_SCALE 一个量，没有"跨层/同层"的划分；
        # spektrafilm 那边的 dir_interlayer / dir_samelayer 也只是 dir_amount 的乘数。
        # 所以三个滑块合并成一个「总量」，另两个键保留在参数里固定为 1.0（旧文件仍可读）。
        self.slider_dir_amount = AdobeScrubSlider(
            "DIR 总量", 0.0, 1.8, 1.0, step=0.05, unit="", decimals=2,
            tooltip="显影抑制剂强度：中性灰不受影响，只改变有色彩差异处的密度分离"
        )
        self.slider_dir_amount.valueChanged.connect(lambda v: self._on_slider_live("dir_amount", v))
        self.slider_dir_amount.sliderReleased.connect(self._on_slider_committed)
        self.sec_chemistry.addWidget(self.slider_dir_amount)

        self.sidebar_layout.addWidget(self.sec_chemistry)
        self._update_push_ui_state()
        self._update_chemistry_ui_state()

    def _init_enlarger_section(self):
        self.sec_enlarger = AccordionSection("放大机与相纸显影")
        self.sec_enlarger.resetClicked.connect(lambda: self.reset_track("enlarger"))

        # 相纸：从左侧卡片墙挪到这里，下拉框选择（与光源下拉同一套行组件）
        paper_items = [(item.get("name") or item["id"], item["id"]) for item in self.engine.get_paper_stocks()]
        self.combo_paper = ComboBoxRow(
            "相纸",
            paper_items,
            default_val=getattr(self, "current_paper_stock", "none"),
            tooltip="相纸型号：决定染色层光谱响应与密度曲线。选「胶片直扫」则跳过印相，输出负像"
        )
        self.combo_paper.valueChanged.connect(self._on_paper_combo_changed)
        self.sec_enlarger.addWidget(self.combo_paper)

        self.combo_illuminant = ComboBoxRow(
            "光源光谱",
            [
                ("TH-KG3 卤素暗房灯 (3400K)", "TH-KG3"),
                ("TH-KG3-L 放大机镜头透过", "TH-KG3-L"),
                ("T 传统白炽灯泡 (Incandescent)", "T"),
                ("D65 标准日光 (6500K)", "D65"),
                ("D50 印刷标准看样 (5000K)", "D50"),
                ("A 标准钨丝光源 (2856K)", "A"),
            ],
            default_val="TH-KG3",
            tooltip="放大机光源：光谱分布决定相纸感光层响应"
        )
        self.combo_illuminant.valueChanged.connect(lambda v: self._on_combo_changed("enlarger_illuminant", v))
        self.sec_enlarger.addWidget(self.combo_illuminant)

        self.slider_magenta = AdobeScrubSlider(
            "品红 / 绿色", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="品红（M）：左偏品红，右偏绿",
            track_gradient=[(0.0, QColor("#f43f5e")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#10b981"))],
            has_center_detent=True
        )
        self.slider_magenta.valueChanged.connect(lambda v: self._on_slider_live("enlarger_magenta", v))
        self.slider_magenta.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_magenta)

        self.slider_yellow = AdobeScrubSlider(
            "黄色 / 蓝色", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="黄色（Y）：左偏黄，右偏蓝",
            track_gradient=[(0.0, QColor("#eab308")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#38bdf8"))],
            has_center_detent=True
        )
        self.slider_yellow.valueChanged.connect(lambda v: self._on_slider_live("enlarger_yellow", v))
        self.slider_yellow.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_yellow)

        self.slider_cyan = AdobeScrubSlider(
            "青色 / 红色", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="青色（C）：左偏青，右偏红",
            track_gradient=[(0.0, QColor("#06b6d4")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#f87171"))],
            has_center_detent=True
        )
        self.slider_cyan.valueChanged.connect(lambda v: self._on_slider_live("enlarger_cyan", v))
        self.slider_cyan.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_cyan)

        # Range is +-2 stops, not the old +1 / -0.58: this slider is the *only* correct way to make
        # a pushed frame brighter again (scene exposure would cancel the underexposure and turn the
        # recipe into plain over-development), so it has to have the room to do that.
        self.slider_print_exp = AdobeScrubSlider(
            "相纸曝光", 0.25, 4.0, 1.0, step=0.02, unit="x", decimals=2,
            tooltip="相纸曝光：整体显影浓度（0.25x～4x，约 ±2 档）。\n"
                    "迫冲后提亮用这一档，不要动场景曝光，否则抵消欠曝",
            track_gradient=[(0.0, QColor("#14161d")), (1.0, QColor("#f1f5f9"))],
            has_center_detent=True
        )
        self.slider_print_exp.valueChanged.connect(lambda v: self._on_slider_live("print_exposure", v))
        self.slider_print_exp.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_print_exp)

        self.slider_preflash_camera = AdobeScrubSlider(
            "胶片预闪", 0.0, 0.25, 0.0, step=0.01, unit="", decimals=2,
            tooltip="在显影前给底片叠加一层均匀曝光（相对计量中灰）。\n"
                    "只抬暗部趾部、几乎不动高光，用来把死黑拉出细节；与下面的相纸预闪是两回事"
        )
        self.slider_preflash_camera.valueChanged.connect(lambda v: self._on_slider_live("pre_flash_camera", v))
        self.slider_preflash_camera.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_preflash_camera)

        self.slider_preflash = AdobeScrubSlider(
            "相纸预闪", 0.0, 0.25, 0.0, step=0.01, unit="", decimals=2,
            tooltip="在相纸曝光域叠加微量基础底光，柔化高光反差并保留亮部层次"
        )
        self.slider_preflash.valueChanged.connect(lambda v: self._on_slider_live("pre_flash", v))
        self.slider_preflash.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_preflash)

        self.slider_morph_gamma = AdobeScrubSlider(
            "显影形态反差", 0.70, 1.30, 1.0, step=0.02, unit="", decimals=2,
            tooltip="相纸显影斜率：决定画面反差"
        )
        self.slider_morph_gamma.valueChanged.connect(lambda v: self._on_slider_live("morph_gamma", v))
        self.slider_morph_gamma.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_morph_gamma)

        self.slider_exhaustion = AdobeScrubSlider(
            "显影液疲劳度", 0.0, 1.00, 0.0, step=0.02, unit="", decimals=2,
            tooltip="显影液疲劳：对相纸肩部做轻微塑形"
        )
        self.slider_exhaustion.valueChanged.connect(lambda v: self._on_slider_live("developer_exhaustion", v))
        self.slider_exhaustion.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_exhaustion)
        # Backward compatibility aliases
        self.slider_enlarger_c = self.slider_cyan
        self.slider_enlarger_m = self.slider_magenta
        self.slider_enlarger_y = self.slider_yellow

        self.sidebar_layout.addWidget(self.sec_enlarger)

    def _init_halation_section(self):
        self.sec_halation = AccordionSection("物理光晕 (Halation)")
        self.sec_halation.resetClicked.connect(lambda: self.reset_track("halation"))

        # The track is the engine's own: Stops about the stock's authored look, -6..+6,
        # with the bottom stop meaning off. The stored parameter stays the linear
        # multiple (`halation_amount`), so exports, the GPU chain and old sessions are
        # untouched -- this row only changes what the number means to a photographer.
        self.slider_halation_amount = AdobeScrubSlider(
            "光晕档位", HALATION_OFF_STOPS, HALATION_MAX_STOPS, 0.0, step=0.05,
            unit=" EV", decimals=2,
            tooltip="相对该卷本征返光比例的档位（引擎的 halationAmount）。\n"
                    "0 = 这支卷自己的标定值；上端 +6 档 = 64 倍，下端为关闭。\n"
                    "光环亮度按返回光占直射光的比例 s/(1+s) 饱和：往右推是越来越厚，"
                    "但不会无限亮；半径由片基厚度决定，不随档位变。\n"
                    "去掉 remjet 的 CineStill 通常在 +2 ~ +4 档才接近实拍观感",
            track_gradient=[(0.0, QColor("#111317")), (0.5, QColor("#5a6170")),
                            (1.0, QColor("#ffb37a"))],
            has_center_detent=True
        )
        self.slider_halation_amount.valueChanged.connect(
            lambda v: self._on_slider_live("halation_amount", scale_from_stops(v)))
        self.slider_halation_amount.sliderReleased.connect(self._on_slider_committed)
        self.sec_halation.addWidget(self.slider_halation_amount)

        # 下线入口（见类顶 HALATION_SPATIAL_UI_ENABLED）：参数仍然存在，只是没人能改。
        self.slider_halation_spatial_scale = AdobeScrubSlider(
            "光晕空间尺度", 0.1, 3.0, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="光晕扩散核的空间尺度倍率"
        )
        self.slider_halation_spatial_scale.valueChanged.connect(lambda v: self._on_slider_live("halation_spatial_scale", v))
        self.slider_halation_spatial_scale.sliderReleased.connect(self._on_slider_committed)
        self.sec_halation.addWidget(self.slider_halation_spatial_scale)
        if not self.HALATION_SPATIAL_UI_ENABLED:
            self.slider_halation_spatial_scale.setVisible(False)

        self.slider_scatter_amount = AdobeScrubSlider(
            "乳剂内散射强度", 0.0, 2.0, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="前向光散射（Light Wrap）强度。\n"
                    "散射核尺寸取引擎标定值；现有胶卷档案均未提供实测散射数据"
        )
        self.slider_scatter_amount.valueChanged.connect(lambda v: self._on_slider_live("scatter_amount", v))
        self.slider_scatter_amount.sliderReleased.connect(self._on_slider_committed)
        self.sec_halation.addWidget(self.slider_scatter_amount)

        self.slider_boost_ev = AdobeScrubSlider(
            "高光能量重建", 0.0, 4.0, 0.0, step=0.05, unit=" EV", decimals=2,
            tooltip="把画面最亮像素再抬高 N 档后交给光晕/散射，重建被扫描截断的高光能量。\n"
                    "保护点在中灰以上 4 档：常规曝光的底片整幅最大值达不到该点，滑块不生效；"
                    "需强过曝或 +EV 曝光补偿后才生效"
        )
        self.slider_boost_ev.valueChanged.connect(lambda v: self._on_slider_live("boost_ev", v))
        self.slider_boost_ev.sliderReleased.connect(self._on_slider_committed)
        self.sec_halation.addWidget(self.slider_boost_ev)

        # Backward compatibility aliases
        self.slider_halation = self.slider_halation_amount
        self.slider_halation_boost = self.slider_boost_ev

        self.sidebar_layout.addWidget(self.sec_halation)
        self.sec_texture = self.sec_halation  # Backward compatibility alias
        self.sec_optical = self.sec_halation  # Backward compatibility alias
        self._init_grain_section()

    def _init_grain_section(self):
        """M5: 银盐颗粒面板（`347b5cb` 删除后重建）。

        四个滑块的语义（前三个 2026-11-05 用户拍板，扫描噪声为 2026-11-11 实验）：
        - 强度：倍率式 0–3，默认 1.0 = 标定参考；内部写 `grain` 键 = 倍率 × 0.4
          （M4 映射 u_grain_strength = grain / 0.4，0.4 是 gold200 锚点的参考强度）。
        - 大小：粒子面积倍率 0.5–2.0，直接写 `grain_size`。
        - 彩色：乘性混合系数 0–1，直接写 `grain_colour`，与档案域 bundle.colour_grain
          相乘（黑白卷档案已强制 0，滑块随黑白胶卷自动禁用）。
        颗粒在 LUT 之外（shader 域），拖动只走 uniform，不触发 LUT 重烘焙。
        """
        self.sec_grain = AccordionSection("银盐颗粒 (Grain)")
        self.sec_grain.resetClicked.connect(lambda: self.reset_track("grain"))

        self.slider_grain_strength = AdobeScrubSlider(
            "颗粒强度", 0.0, 3.0, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="颗粒强度：1.0 为 gold200 标定参考，0 无颗粒，2.0 噪声幅度加倍。\n"
                    "速率与倍率平方反比，倍率越高颗粒越细密"
        )
        self.slider_grain_strength.valueChanged.connect(
            lambda v: self._on_slider_live("grain", round(0.4 / (v * v), 6) if v > 1e-3 else 0.0))
        self.slider_grain_strength.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_grain_strength)

        self.slider_grain_size = AdobeScrubSlider(
            "颗粒大小", 0.5, 2.0, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="颗粒空间尺度：缩放整个颗粒图案，团簇随倍率变化。\n"
                    "1.0 为档案标定几何；噪声幅度由「颗粒强度」控制"
        )
        self.slider_grain_size.valueChanged.connect(lambda v: self._on_slider_live("grain_size", v))
        self.slider_grain_size.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_grain_size)

        self.slider_grain_colour = AdobeScrubSlider(
            "彩色颗粒", 0.0, 1.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="彩色颗粒混合：1.0 为档案标定（三染料云独立），0 为共用亮度颗粒（类似黑白）。\n"
                    "黑白胶卷无效"
        )
        self.slider_grain_colour.valueChanged.connect(lambda v: self._on_slider_live("grain_colour", v))
        self.slider_grain_colour.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_grain_colour)

        # 实验：扫描仪读出噪声。默认 0 = 完全关闭，现有出图逐字节不变；用户手动拉才触发。
        # 与乳剂颗粒是独立噪声源（暗部不再干净），只走 shader uniform，不进 LUT 缓存 key。
        self.slider_scan_noise = AdobeScrubSlider(
            "扫描噪声 (实验)", 0.0, 1.0, 0.0, step=0.01, unit="", decimals=2,
            tooltip="模拟欠曝底片被强行扫描提亮时放大的读出噪声。\n"
                    "0 = 关闭；拉起后暗部出现颗粒，亮部影响很小；配合「黑场」抬灰即发灰的冲扫件。\n"
                    "仅在 GPU 预览与 GPU 导出生效，CPU 导出不含颗粒。"
        )
        self.slider_scan_noise.valueChanged.connect(lambda v: self._on_slider_live("scan_noise", v))
        self.slider_scan_noise.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_scan_noise)

        self.sidebar_layout.addWidget(self.sec_grain)

    _init_texture_section = _init_halation_section
    _init_optical_section = _init_halation_section

    def _init_menu_and_actions(self):
        # 1. File Menu
        file_menu = self.menu_bar.addMenu("文件 (F)")
        
        self.act_open_file = QAction("导入底片...", self)
        self.act_open_file.setIcon(get_icon("open_file.png"))
        self.act_open_file.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open_file.triggered.connect(self.action_open_files)
        file_menu.addAction(self.act_open_file)

        self.act_open_folder = QAction("导入文件夹...", self)
        self.act_open_folder.setIcon(get_icon("open_folder.png"))
        self.act_open_folder.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_open_folder.triggered.connect(self.action_open_folder)
        file_menu.addAction(self.act_open_folder)

        self.act_open_session = QAction("打开会话工程 (.sdss)...", self)
        self.act_open_session.setIcon(get_icon("open_session.png"))
        self.act_open_session.triggered.connect(self.action_open_session)
        file_menu.addAction(self.act_open_session)

        # Recent Sessions Submenu (Item 8 & 11)
        self.recent_menu = file_menu.addMenu("最近会话")
        self._update_recent_menu()

        file_menu.addSeparator()

        # Item 9: Session Save & Save As
        self.act_save_session = QAction("保存当前会话 (.sdss)", self)
        self.act_save_session.setIcon(get_icon("save.png"))
        self.act_save_session.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_session.setEnabled(False)
        self.act_save_session.triggered.connect(self.action_save_session)
        file_menu.addAction(self.act_save_session)

        self.act_save_session_as = QAction("另存为会话文件 (.sdss)...", self)
        self.act_save_session_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_session_as.setEnabled(False)
        self.act_save_session_as.triggered.connect(self.action_save_session_as)
        file_menu.addAction(self.act_save_session_as)

        self.act_close_session = QAction("关闭当前会话", self)
        self.act_close_session.setShortcut(QKeySequence("Ctrl+W"))
        self.act_close_session.triggered.connect(self.action_close_session)
        self.act_close_session.setEnabled(False)
        file_menu.addAction(self.act_close_session)

        file_menu.addSeparator()

        # Item 4: Quick Export with last settings
        self.act_quick_export = QAction("快速扫描 (上次参数)", self)
        self.act_quick_export.setIcon(get_icon("quick_export.png"))
        self.act_quick_export.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.act_quick_export.triggered.connect(lambda: self.action_quick_export())
        self.act_quick_export.setEnabled(False)
        file_menu.addAction(self.act_quick_export)

        self.act_export = QAction("扫描...", self)
        self.act_export.setIcon(get_icon("export.png"))
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.triggered.connect(self.action_export_image)
        self.act_export.setEnabled(False)
        file_menu.addAction(self.act_export)

        act_queue = QAction("扫描队列管理...", self)
        act_queue.triggered.connect(self.open_export_queue_dialog)
        file_menu.addAction(act_queue)

        self.act_export_lut = QAction("导出 3D LUT (.cube)...", self)
        self.act_export_lut.triggered.connect(self.action_export_darkroom_lut)
        self.act_export_lut.setEnabled(False)
        file_menu.addAction(self.act_export_lut)

        file_menu.addSeparator()

        # 维护入口。A1 之前改中性滤色或修档案都靠手工删 %LOCALAPPDATA%，很容易把「缓存没清」
        # 当成「代码没生效」。这里给一个正经按钮，只删这两个目录，不做通配删除。
        self.act_rebuild_caches = QAction("重建渲染缓存...", self)
        self.act_rebuild_caches.setToolTip(
            "删除 LUT 与颗粒瓦片缓存，下次渲染重建。改动档案或印片滤色后使用"
        )
        self.act_rebuild_caches.triggered.connect(self.rebuild_render_caches)
        file_menu.addAction(self.act_rebuild_caches)

        file_menu.addSeparator()

        act_exit = QAction("退出", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # 2. Edit Menu (Item 5: Simplified concise labels, 13px icons)
        edit_menu = self.menu_bar.addMenu("编辑 (E)")
        self.edit_menu = edit_menu
        
        self.act_undo = QAction("撤销", self)
        self.act_undo.setIcon(get_icon("undo.png"))
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self.undo)
        edit_menu.addAction(self.act_undo)

        self.act_redo = QAction("重做", self)
        self.act_redo.setIcon(get_icon("redo.png"))
        self.act_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.act_redo.triggered.connect(self.redo)
        edit_menu.addAction(self.act_redo)

        edit_menu.addSeparator()

        self.act_crop = QAction("裁切", self)
        self.act_crop.setIcon(get_icon("crop.png"))
        self.act_crop.setShortcut(QKeySequence("C"))
        self.act_crop.triggered.connect(self.toggle_crop_mode)
        edit_menu.addAction(self.act_crop)

        edit_menu.addSeparator()

        self.act_copy_params = QAction("复制所有参数", self)
        self.act_copy_params.setShortcut(QKeySequence("Ctrl+C"))
        self.act_copy_params.triggered.connect(self.action_copy_params)
        edit_menu.addAction(self.act_copy_params)

        self.act_paste_params = QAction("粘贴所有参数", self)
        self.act_paste_params.setShortcut(QKeySequence("Ctrl+V"))
        self.act_paste_params.triggered.connect(self.action_paste_params)
        edit_menu.addAction(self.act_paste_params)

        self.act_select_all = QAction("全选底片", self)
        self.act_select_all.setShortcut(QKeySequence("Ctrl+A"))
        self.act_select_all.triggered.connect(self.action_select_all_photos)
        edit_menu.addAction(self.act_select_all)

        self.act_invert_selection = QAction("反选底片", self)
        self.act_invert_selection.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self.act_invert_selection.triggered.connect(self.action_invert_selection)
        edit_menu.addAction(self.act_invert_selection)

        edit_menu.addSeparator()

        self.act_reset_all = QAction("重置所有参数", self)
        self.act_reset_all.setIcon(get_icon("reset.png"))
        self.act_reset_all.setShortcut(QKeySequence("Ctrl+R"))
        self.act_reset_all.triggered.connect(self.reset_all_params)
        edit_menu.addAction(self.act_reset_all)

        self.act_clear_crop = QAction("清除裁切", self)
        self.act_clear_crop.triggered.connect(self.action_clear_active_crop)
        edit_menu.addAction(self.act_clear_crop)

        self.act_clear_sdc = QAction("删除暗房配置 (.sdc)", self)
        self.act_clear_sdc.setShortcut(QKeySequence("Ctrl+Alt+R"))
        self.act_clear_sdc.triggered.connect(self.action_clear_sdc_active)
        edit_menu.addAction(self.act_clear_sdc)

        edit_menu.addSeparator()

        self.act_prefs = QAction("首选项...", self)
        self.act_prefs.setIcon(get_icon("preferences.png"))
        self.act_prefs.setShortcut(QKeySequence("Ctrl+,"))
        self.act_prefs.triggered.connect(self.open_preferences)
        edit_menu.addAction(self.act_prefs)

        edit_menu.aboutToShow.connect(self._update_menu_actions_state)

        # 3. View Menu
        view_menu = self.menu_bar.addMenu("视图 (V)")
        self.view_menu = view_menu
        
        self.act_fit_img = QAction("适应图像", self)
        self.act_fit_img.setIcon(get_icon("fit_window.png"))
        self.act_fit_img.setShortcut(QKeySequence("Ctrl+0"))
        self.act_fit_img.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(self.act_fit_img)

        self.act_compare = QAction("对比\t\\", self)
        self.act_compare.setIcon(get_icon("compare.png"))
        self.act_compare.triggered.connect(self.toggle_split_view)
        view_menu.addAction(self.act_compare)

        # Export-matched preview lives in 首选项 → 图像与渲染 (a multi-second rebuild is a
        # render policy, not a view action).  Only the resample kernel is pinned here.
        self._hq_resample = self._HQ_RESAMPLE

        self.act_fit_win = QAction("适应窗口", self)
        self.act_fit_win.setIcon(get_icon("fit_window.png"))
        self.act_fit_win.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(self.act_fit_win)

        self.act_100 = QAction("1:1 实际像素", self)
        self.act_100.setIcon(get_icon("zoom_100.png"))
        self.act_100.setShortcut(QKeySequence("Ctrl+1"))
        self.act_100.triggered.connect(self.canvas.set_zoom_100)
        view_menu.addAction(self.act_100)

        def _seqs(*specs):
            """Build a shortcut list with duplicates removed.

            Qt's shortcut map treats a repeated sequence as *ambiguous* even when both copies
            belong to the same action and reject it silently: registering StandardKey.ZoomOut
            (which already is Ctrl+-) next to the literal "Ctrl+-" is what made Ctrl+- do
            nothing while Ctrl+Shift+- still worked.
            """
            out, seen = [], set()
            for spec in specs:
                seq = QKeySequence(spec)
                key = seq.toString()
                if key and key not in seen:
                    seen.add(key)
                    out.append(seq)
            return out

        self.act_zoom_in = QAction("放大", self)
        self.act_zoom_in.setShortcuts(_seqs(
            QKeySequence.StandardKey.ZoomIn,
            "Ctrl+=",            # plain "=" -- what people actually press
            "Ctrl+Shift+=",      # "Ctrl++" as Qt spells it
        ))
        self.act_zoom_in.triggered.connect(lambda: self.canvas.zoom_in(animate=True))
        view_menu.addAction(self.act_zoom_in)
        self.addAction(self.act_zoom_in)

        self.act_zoom_out = QAction("缩小", self)
        self.act_zoom_out.setShortcuts(_seqs(
            QKeySequence.StandardKey.ZoomOut,
            "Ctrl+Shift+-",      # the keyboard's "-" with Shift, and numpad minus
        ))
        self.act_zoom_out.triggered.connect(lambda: self.canvas.zoom_out(animate=True))
        view_menu.addAction(self.act_zoom_out)
        self.addAction(self.act_zoom_out)

        view_menu.addSeparator()

        # 校样（P1）。快捷键刻意**不用** QAction.setShortcut：
        # QAction 的快捷键走 QShortcutMap，按住不放会随系统键盘重复率反复触发
        # triggered/toggled —— 用户按住想"保持开启"，实际看到的是每秒几十次开关。
        # QShortcut 可以 setAutoRepeat(False)，按住只发一次；菜单里的快捷键文字用
        # tab 手写在文本里（Qt 会把 tab 后半段右对齐渲染成快捷键列）。
        self.act_proof = QAction("校样\tCtrl+Shift+Y", self)
        self.act_proof.setCheckable(True)
        self.act_proof.setStatusTip("按选定的输出色彩空间模拟显示")
        self.act_proof.toggled.connect(self._on_proof_toggled)
        view_menu.addAction(self.act_proof)
        self._sc_proof = QShortcut(QKeySequence("Ctrl+Shift+Y"), self)
        self._sc_proof.setAutoRepeat(False)
        self._sc_proof.setContext(Qt.ShortcutContext.WindowShortcut)
        self._sc_proof.activated.connect(self.act_proof.toggle)

        self.act_proof_warn = QAction("色域警告\tCtrl+Alt+Y", self)
        self.act_proof_warn.setCheckable(True)
        self.act_proof_warn.setStatusTip("把超出校样色域的像素叠成品红")
        self.act_proof_warn.toggled.connect(self._on_proof_warn_toggled)
        view_menu.addAction(self.act_proof_warn)
        self._sc_proof_warn = QShortcut(QKeySequence("Ctrl+Alt+Y"), self)
        self._sc_proof_warn.setAutoRepeat(False)
        self._sc_proof_warn.setContext(Qt.ShortcutContext.WindowShortcut)
        self._sc_proof_warn.activated.connect(self.act_proof_warn.toggle)

        self.act_proof_setup = QAction("校样设置…", self)
        self.act_proof_setup.triggered.connect(self.open_proof_setup)
        view_menu.addAction(self.act_proof_setup)

        # 打开时按已保存的偏好同步菜单勾选状态（此时 canvas 已经建好）
        self._sync_proof_actions()

        view_menu.addSeparator()

        self.menu_star_filter = view_menu.addMenu("星标筛选")
        self.menu_star_filter.setStyleSheet(get_darkroom_menu_style())
        self.star_filter_actions = {}
        for key, text in [
            ("all", "全部"),
            ("separator", ""),
            ("5", "★★★★★"),
            ("4", "★★★★"),
            ("3", "★★★"),
            ("2", "★★"),
            ("1", "★"),
            ("separator", ""),
            ("0", "无评分"),
        ]:
            if key == "separator":
                self.menu_star_filter.addSeparator()
            else:
                act = QAction(text, self)
                act.setCheckable(True)
                act.triggered.connect(lambda chk=False, k=key: self._on_menu_filter_selected(k))
                self.menu_star_filter.addAction(act)
                self.star_filter_actions[key] = act

        view_menu.addSeparator()

        self.act_reset_layout = QAction("恢复默认布局", self)
        self.act_reset_layout.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.act_reset_layout.triggered.connect(self.reset_default_layout)
        view_menu.addAction(self.act_reset_layout)

        view_menu.aboutToShow.connect(self._update_menu_actions_state)
        self._update_menu_actions_state()

        # 4. Help Menu (Item 8: Version in About)
        help_menu = self.menu_bar.addMenu("帮助 (H)")
        act_about = QAction("关于 SpektraDarkroom", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _update_menu_actions_state(self):
        """Dynamically enable/disable Edit and View menu actions based on photo selection and history."""
        is_startup = (hasattr(self, 'stack') and self.stack.currentIndex() == 0)
        has_active_photo = bool(self._current_photo_path) and bool(self.photos) and not is_startup and bool(self.active_photo_id)
        has_any_photos = bool(self.photos) and not is_startup

        # Multi-photo / target photo modification states
        is_edited = False
        has_crop = False
        has_sdc = False
        if has_active_photo:
            sel_ids = self.filmstrip.get_selected_photo_ids() if hasattr(self, 'filmstrip') else []
            if not sel_ids and self.active_photo_id:
                sel_ids = [self.active_photo_id]
            target_photos = [p for p in self.photos if p and p.get("id") in set(sel_ids)]
            if not target_photos and self.active_photo_id:
                target_photos = [p for p in self.photos if p and p.get("id") == self.active_photo_id]

            if getattr(self.canvas, "is_crop_editing", False):
                overlay = getattr(self.canvas, "crop_overlay", None)
                if overlay:
                    has_crop = self._is_crop_modified(overlay.get_crop_dict())
                is_edited = has_crop
            else:
                # 1. Check if crop is modified for any target photo
                for p in target_photos:
                    c = p.get("crop")
                    if not c and isinstance(p.get("params"), dict):
                        c = p["params"].get("crop")
                    if not c and p.get("id") == self.active_photo_id:
                        c = self.current_params.get("crop")
                    if self._is_crop_modified(c):
                        has_crop = True
                        break

                # 2. Check if any target photo is edited or dirty
                for p in target_photos:
                    if p.get("id") == self.active_photo_id:
                        if getattr(self, "_photo_is_dirty", False):
                            is_edited = True
                            break
                    if p.get("is_dirty") or self._is_photo_edited(p):
                        is_edited = True
                        break

                # 3. Check if any target photo has .sdc on disk
                from core import sdc_manager
                for p in target_photos:
                    p_path = p.get("path") or (self._current_photo_path if p.get("id") == self.active_photo_id else None)
                    if p_path and sdc_manager.has_sdc(p_path):
                        has_sdc = True
                        break

        # 1. Edit menu: Undo / Redo
        can_undo = False
        can_redo = False
        if not is_startup:
            if getattr(self.canvas, "is_crop_editing", False):
                overlay = getattr(self.canvas, "crop_overlay", None)
                can_undo = bool(overlay and len(getattr(overlay, "_crop_undo_stack", [])) > 1)
                can_redo = bool(overlay and len(getattr(overlay, "_crop_redo_stack", [])) > 0)
            else:
                can_undo = len(self._undo_stack) > 1
                can_redo = len(self._redo_stack) > 0

        if hasattr(self, 'act_undo'):
            self.act_undo.setEnabled(can_undo)
        if hasattr(self, 'act_redo'):
            self.act_redo.setEnabled(can_redo)

        # 2. Edit menu: Photo-dependent actions
        if hasattr(self, 'act_crop'):
            self.act_crop.setEnabled(has_active_photo)
        # Copy params is only enabled when photo has actual edits to copy
        if hasattr(self, 'act_copy_params'):
            self.act_copy_params.setEnabled(has_active_photo and is_edited)
        if hasattr(self, 'act_paste_params'):
            has_clipboard = bool(getattr(self, "_copied_darkroom_profile", None))
            self.act_paste_params.setEnabled(has_active_photo and has_clipboard)
        if hasattr(self, 'act_select_all'):
            self.act_select_all.setEnabled(has_any_photos)
        if hasattr(self, 'act_invert_selection'):
            self.act_invert_selection.setEnabled(has_any_photos)

        if hasattr(self, 'act_reset_all'):
            if getattr(self.canvas, "is_crop_editing", False):
                self.act_reset_all.setEnabled(has_crop)
            else:
                self.act_reset_all.setEnabled(has_active_photo and is_edited)

        if hasattr(self, 'act_clear_crop'):
            self.act_clear_crop.setEnabled(has_active_photo and has_crop)

        if hasattr(self, 'act_clear_sdc'):
            self.act_clear_sdc.setEnabled(has_active_photo and (has_sdc or is_edited))

        # 3. View menu: Canvas & Viewport navigation actions
        if hasattr(self, 'act_fit_img'):
            self.act_fit_img.setEnabled(has_active_photo)
        # Compare mode is only enabled when photo is actually edited
        if hasattr(self, 'act_compare'):
            self.act_compare.setEnabled(has_active_photo and is_edited)
        if hasattr(self, 'btn_split'):
            self.btn_split.setEnabled(has_active_photo and is_edited)
        if hasattr(self, 'act_fit_win'):
            self.act_fit_win.setEnabled(has_active_photo)
        if hasattr(self, 'act_100'):
            self.act_100.setEnabled(has_active_photo)
        if hasattr(self, 'act_zoom_in'):
            self.act_zoom_in.setEnabled(has_active_photo)
        if hasattr(self, 'act_zoom_out'):
            self.act_zoom_out.setEnabled(has_active_photo)
        if hasattr(self, 'menu_star_filter'):
            self.menu_star_filter.setEnabled(has_any_photos)
            curr_f = getattr(self.filmstrip, "_rating_filter", "all") if hasattr(self, "filmstrip") else "all"
            if hasattr(self, "filmstrip"):
                counts = self.filmstrip.get_filter_counts()
                labels_map = {
                    "all": f"全部（{counts['all']}张）",
                    "5": f"★★★★★（{counts['5']}张）",
                    "4": f"★★★★（{counts['4']}张）",
                    "3": f"★★★（{counts['3']}张）",
                    "2": f"★★（{counts['2']}张）",
                    "1": f"★（{counts['1']}张）",
                    "0": f"无评分（{counts['0']}张）",
                }
                for k, act in getattr(self, "star_filter_actions", {}).items():
                    if k in labels_map:
                        act.setText(labels_map[k])
                    act.setChecked(k == curr_f)
            else:
                for k, act in getattr(self, "star_filter_actions", {}).items():
                    act.setChecked(k == curr_f)

    def reset_default_layout(self):
        """Restore all splitters to optimal default layout."""
        if hasattr(self, "main_splitter"):
            self.main_splitter.setSizes([320, 800, 320])
        if hasattr(self, "left_splitter"):
            self.left_splitter.setSizes([300, 200])
        if hasattr(self, "center_splitter"):
            self.center_splitter.setSizes([600, 100])
        config_manager.save_window_config({
            "splitter_main": [320, 800, 320],
            "splitter_left": [300, 180, 200],
            "splitter_center": [600, 106]
        })
        self.reflow_grids()
        self.canvas.fit_to_view()

    def _wire_canvas_events(self):
        self.canvas.requestUndo.connect(self.undo)
        self.canvas.requestRedo.connect(self.redo)
        self.canvas.requestReset.connect(self.reset_all_params)
        self.canvas.requestResetToOpen.connect(self.on_reset_to_open)
        self.canvas.requestClearCrop.connect(self.action_clear_active_crop)
        self.canvas.requestExport.connect(self.action_export_image)
        self.canvas.requestQuickExport.connect(self.action_quick_export)
        self.canvas.requestToggleSplit.connect(self.toggle_split_view)
        self.canvas.requestCompareStart.connect(self._on_compare_long_press_start)
        self.canvas.requestCompareEnd.connect(self._on_compare_long_press_end)
        self.canvas.importRequested.connect(self.action_open_files)

    def _update_recent_menu(self):
        self.recent_menu.clear()
        recent_sessions = config_manager.get_recent_sessions()
        if not recent_sessions:
            act_none = self.recent_menu.addAction("暂无最近会话")
            act_none.setEnabled(False)
            return

        for path in recent_sessions:
            act = self.recent_menu.addAction(os.path.basename(path))
            act.setToolTip(path)
            act.triggered.connect(lambda checked=False, p=path: self._open_recent_session(p))

        self.recent_menu.addSeparator()
        act_clear = self.recent_menu.addAction("清除最近记录")
        act_clear.triggered.connect(lambda: (config_manager.clear_recent_sessions(), self._update_recent_menu()))

    def _open_recent_session(self, path):
        from core.path_utils import normalize_path
        if self._current_session_path and normalize_path(path).lower() == normalize_path(self._current_session_path).lower():
            return
        if not self._prompt_save_if_session_dirty():
            return
        self._open_session_by_path(path)

    def _mark_current_photo_dirty(self):
        if getattr(self, '_is_updating_sliders', False) or getattr(self, '_is_switching_photo', False):
            return
        was_dirty = bool(getattr(self, '_photo_is_dirty', False))
        self._photo_is_dirty = True
        if self.active_photo_id:
            photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if photo:
                photo["is_dirty"] = True
                photo["film_profile"] = self.current_film_stock
                # 自动套上的默认相纸不写成实值（见 _save_current_photo_sdc），否则编辑一下
                # 就把这张照片变成"已选过相纸"，首选项改默认纸时它不再跟随。
                photo["paper_profile"] = (
                    self.current_paper_stock if self._photo_paper_is_explicit(photo) else "")
                photo["params"] = dict(self.current_params)
            if not was_dirty:
                self.filmstrip.set_photo_dirty(self.active_photo_id, True)
                self._update_window_title()

    def _update_window_title(self):
        base_title = get_app_title()
        is_startup = (hasattr(self, 'stack') and self.stack.currentIndex() == 0)
        has_named_session = bool(self._current_session_path and not getattr(self, "_is_temp_session", False))
        has_session = bool(has_named_session or self.photos or (getattr(self, "_is_temp_session", False) and not is_startup))
        has_photo = bool(self._current_photo_path) and bool(self.photos) and not is_startup

        if hasattr(self, 'act_export'):
            self.act_export.setEnabled(has_photo)
        if hasattr(self, 'act_quick_export'):
            self.act_quick_export.setEnabled(has_photo)
        if hasattr(self, 'act_export_lut'):
            self.act_export_lut.setEnabled(has_photo)
        if hasattr(self, 'act_save_session'):
            self.act_save_session.setEnabled(has_session)
        if hasattr(self, 'act_save_session_as'):
            self.act_save_session.setEnabled(has_session)
            self.act_save_session_as.setEnabled(has_session)
        if hasattr(self, 'act_close_session'):
            self.act_close_session.setEnabled(has_session)
        self._update_menu_actions_state()

        # Only display pure app brand on the startup welcome screen when NO session is loaded
        if not self.photos and not self._current_session_path and is_startup:
            if self.windowTitle() != base_title:
                self.setWindowTitle(base_title)
            if hasattr(self, 'lbl_brand'):
                self.lbl_brand.setText("SpektraDarkroom")
            self._update_photo_info_display()
            return

        is_temp = getattr(self, "_is_temp_session", False)
        session_name = os.path.basename(self._current_session_path) if (self._current_session_path and not is_temp) else "未命名会话"
        dirty_flag = "*" if getattr(self, "_session_is_dirty", False) else ""
        display_title = f"{dirty_flag}{session_name}"

        full_title = f"{display_title} - {base_title}"
        if self.windowTitle() != full_title:
            self.setWindowTitle(full_title)
        if hasattr(self, 'lbl_brand'):
            self.lbl_brand.setText(display_title)
        self._update_photo_info_display()

    def _update_photo_info_display(self):
        """Item 5: Update persistent status indicator showing active photo filename & actual zoom percentage."""
        if not hasattr(self, "lbl_photo_info"):
            return
        if self._current_photo_path:
            pct = int(round(self.canvas.get_actual_zoom_ratio() * 100))
            fn = os.path.basename(self._current_photo_path)
            self.lbl_photo_info.setText(f"{fn}  |  {pct}%")
            self.lbl_photo_info.setToolTip(f"{fn}  |  {pct}%")
        else:
            self.lbl_photo_info.setText("")
            self.lbl_photo_info.setToolTip("")
        if hasattr(self, "status_slot"):
            self.status_slot.updateGeometry()
        if hasattr(self, 'view_toolbar') and hasattr(self.view_toolbar, '_update_center'):
            self.view_toolbar._update_center()

    def _on_filmstrip_selection_changed(self, selected_ids):
        """Item 10: Permanent multi-selection notification without blinking or auto-hide."""
        count = len(selected_ids)
        if count > 1:
            self._is_multi_selecting = True
            self.set_app_status(f"已选择 {count} 张底片", spinning=False, timeout_ms=0)
        else:
            if getattr(self, "_is_multi_selecting", False):
                self._is_multi_selecting = False
                self.restore_default_status()
        self._update_menu_actions_state()

    def restore_default_status(self):
        """Restores transient status to default or clears it while preserving permanent photo info."""
        self._update_photo_info_display()
        if self._stale_import_lock():
            self._is_importing = False
        if getattr(self, "_is_importing", False) or getattr(self, "_is_multi_selecting", False):
            return
        if hasattr(self, "export_queue") and self.export_queue and self.export_queue.has_active_tasks():
            return
        if hasattr(self, "_status_timer") and self._status_timer.isActive():
            return
        if hasattr(self, 'status_slot'):
            self.status_slot.transition_to("info", duration_ms=180, on_finished=self._clear_status_text)
        else:
            self._clear_status_text()

    def _clear_status_text(self):
        if self._stale_import_lock():
            self._is_importing = False
        if getattr(self, "_is_importing", False) or getattr(self, "_is_multi_selecting", False):
            return
        self.status_label.setText("")
        self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
        colors = theme_manager.get_colors() if hasattr(theme_manager, 'get_colors') else {'text_muted': '#94a3b8'}
        self.status_label.setStyleSheet(f"QLabel#status_label {{ color: {colors['text_muted']}; font-size: 11px; }}")
        self.status_label.setToolTip("")
        self.spinner.stop()
        self.spinner.setVisible(False)
        if hasattr(self, 'view_toolbar') and hasattr(self.view_toolbar, '_update_center'):
            self.view_toolbar._update_center()

    def action_save_current(self):
        if self._current_photo_path:
            sdc_manager.save_sdc(
                self._current_photo_path,
                self.current_film_stock,
                self.current_paper_stock,
                self.current_params,
                print_mode=getattr(self, "current_print_mode", "optical")
            )
            self._photo_is_dirty = False
            if self.active_photo_id:
                photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                if photo:
                    photo["is_dirty"] = False
                self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            self._update_window_title()
            self.set_app_status(f"已保存更改：{os.path.basename(self._current_photo_path)}", timeout_ms=4000)

    def _on_backslash_timeout(self):
        if getattr(self, "_backslash_pressed", False):
            self._backslash_long_pressed = True
            self._on_compare_long_press_start()

    def eventFilter(self, watched, event):
        """Handle global backslash key, navigation shortcuts, and scrollbar context menus."""
        et = event.type()

        # Suppress context menus and right-clicks on all scrollbars app-wide
        if et in (QEvent.Type.ContextMenu, QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            from PySide6.QtWidgets import QScrollBar
            if isinstance(watched, QScrollBar) or (hasattr(watched, "parent") and isinstance(watched.parent(), QScrollBar)):
                if et == QEvent.Type.ContextMenu or (hasattr(event, "button") and event.button() == Qt.MouseButton.RightButton):
                    return True

        if et in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            # Do not intercept if a dialog or child modal is active and event belongs to it
            if watched and hasattr(watched, "window") and watched.window() != self:
                return super().eventFilter(watched, event)
            # Do not intercept if user is typing in a text/input field, or driving a combo
            # box: ←/→ inside a focused dropdown means "previous/next option", never
            # "previous/next photo".
            from PySide6.QtWidgets import (
                QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox)
            input_types = (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox)
            focus_w = QApplication.focusWidget()

            def _is_input_widget(w):
                if w is None:
                    return False
                if isinstance(w, input_types):
                    return True
                # The editable part of a combo box reports its own viewport as focus widget.
                # QWindow (a native dialog's focus object) has no parentWidget -- asking it
                # raised AttributeError inside the event filter on every key press.
                parent = w.parentWidget() if isinstance(w, QWidget) else None
                return parent is not None and isinstance(parent, input_types)

            if _is_input_widget(watched) or _is_input_widget(focus_w):
                return super().eventFilter(watched, event)

            if et == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Left:
                    self._on_step_photo(-1)
                    return True
                elif event.key() == Qt.Key.Key_Right:
                    self._on_step_photo(1)
                    return True
                elif event.key() == Qt.Key.Key_Home:
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        if hasattr(self, "filmstrip"):
                            self.filmstrip._select_range_to_boundary(to_start=True)
                    else:
                        self._on_navigate_first_photo()
                    return True
                elif event.key() == Qt.Key.Key_End:
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        if hasattr(self, "filmstrip"):
                            self.filmstrip._select_range_to_boundary(to_start=False)
                    else:
                        self._on_navigate_last_photo()
                    return True

            if event.key() == Qt.Key.Key_Backslash:
                if et == QEvent.Type.KeyPress:
                    if event.isAutoRepeat():
                        return True
                    self._backslash_pressed = True
                    self._backslash_long_pressed = False
                    self._backslash_timer.start(180)
                    return True
                elif et == QEvent.Type.KeyRelease:
                    if event.isAutoRepeat():
                        return True
                    if getattr(self, "_backslash_pressed", False):
                        self._backslash_pressed = False
                        if hasattr(self, "_backslash_timer") and self._backslash_timer.isActive():
                            self._backslash_timer.stop()
                        if getattr(self, "_backslash_long_pressed", False):
                            self._backslash_long_pressed = False
                            self._on_compare_long_press_end()
                        else:
                            self.toggle_split_view()
                        return True

        elif et in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut):
            if watched == self and getattr(self, "_backslash_pressed", False):
                self._backslash_pressed = False
                if hasattr(self, "_backslash_timer") and self._backslash_timer.isActive():
                    self._backslash_timer.stop()
                if getattr(self, "_backslash_long_pressed", False):
                    self._backslash_long_pressed = False
                    self._on_compare_long_press_end()

        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        """Item 5: Support Delete / Backspace keys to delete selected photo(s) from filmstrip.
        Item 7: Support Left / Right arrow keys to step previous / next photo in filmstrip.
        Support Home / End keys to navigate to first / last photo in filmstrip.
        """
        # Esc 中止导入/恢复。放在最前面，因为导入期间焦点可能在任意控件上，而按钮本身
        # 会被主线程的卡顿拖慢；键盘事件只要窗口还活着就一定到得了。
        if event.key() == Qt.Key.Key_Escape and getattr(self, "_is_importing", False):
            if not self._stale_import_lock():
                self._cancel_active_import(notify_user=True)
                event.accept()
                return

        focus_w = QApplication.focusWidget()
        from PySide6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox
        is_typing = isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox))

        if not is_typing:
            mods = event.modifiers()
            has_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            has_alt = bool(mods & Qt.KeyboardModifier.AltModifier)
            # "Ctrl++" arrives as Ctrl+Shift+= on a normal layout and as Ctrl+Shift+numpad+
            # on the keypad, so the Shift bit must be ignored for the zoom chords -- otherwise
            # they only fired when the user pressed Ctrl+Shift manually.  Every other shortcut
            # below still sees the real modifiers.
            zoom_ctrl = bool((mods & ~Qt.KeyboardModifier.ShiftModifier) & Qt.KeyboardModifier.ControlModifier)

            # Zoom shortcuts: Ctrl + Plus / Equal / Minus / Underscore / 0 / 1
            if zoom_ctrl and not has_alt:
                if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                    if hasattr(self, "canvas") and self.canvas._has_image:
                        self.canvas.zoom_in(animate=True)
                        event.accept()
                        return
                elif event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
                    if hasattr(self, "canvas") and self.canvas._has_image:
                        self.canvas.zoom_out(animate=True)
                        event.accept()
                        return
                elif event.key() == Qt.Key.Key_0:
                    if hasattr(self, "canvas") and self.canvas._has_image:
                        self.canvas.fit_to_view(animate=True)
                        event.accept()
                        return
                elif event.key() == Qt.Key.Key_1:
                    if hasattr(self, "canvas") and self.canvas._has_image:
                        self.canvas.set_zoom_100()
                        event.accept()
                        return

            # Reset to open: Ctrl + Alt + R
            if has_ctrl and has_alt and event.key() == Qt.Key.Key_R:
                self.on_reset_to_open()
                event.accept()
                return

            # Reset crop (in crop mode) or reset params (outside): Ctrl + R
            if has_ctrl and not has_alt and event.key() == Qt.Key.Key_R:
                if getattr(self.canvas, "is_crop_editing", False):
                    self.canvas.crop_overlay.reset_crop()
                else:
                    self.reset_all_params()
                event.accept()
                return

            # Rating shortcuts: Number keys 0-5 (0 to clear, 1-5 to rate)
            if not has_ctrl and not has_alt:
                if event.key() in (Qt.Key.Key_0, Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5):
                    target_rating = event.key() - Qt.Key.Key_0
                    self.action_set_active_rating(target_rating)
                    event.accept()
                    return

            if event.key() == Qt.Key.Key_C and not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
                self.toggle_crop_mode()
                event.accept()
                return

            if event.key() == Qt.Key.Key_Left:
                self._on_step_photo(-1)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Right:
                self._on_step_photo(1)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Home:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    if hasattr(self, "filmstrip"):
                        self.filmstrip._select_range_to_boundary(to_start=True)
                else:
                    self._on_navigate_first_photo()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_End:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    if hasattr(self, "filmstrip"):
                        self.filmstrip._select_range_to_boundary(to_start=False)
                else:
                    self._on_navigate_last_photo()
                event.accept()
                return

        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and not is_typing:
            if hasattr(self, "filmstrip") and self.filmstrip._selected_ids:
                self.on_remove_multiple_photos(list(self.filmstrip._selected_ids))
                event.accept()
                return
            elif self.active_photo_id:
                self.on_remove_photo(self.active_photo_id)
                event.accept()
                return
        super().keyPressEvent(event)

    def _stale_import_lock(self) -> bool:
        """True when the import lock outlived its worker (crash, killed batch, lost signal).

        The lock freezes every status message for the whole session, so a lost signal would
        leave the spinner turning forever and hide messages such as "会话已保存".
        """
        if not getattr(self, "_is_importing", False):
            return False
        worker = getattr(self, "_import_worker", None)
        if worker is not None and worker.isRunning():
            return False
        if getattr(self, "_pending_import_queue", None):
            return False
        return time.time() - float(getattr(self, "_import_lock_time", 0.0)) > 10.0

    def set_app_status(self, text: str, spinning: bool = False, timeout_ms: int = 0, is_export_link: bool = False):
        """Centralized status indicator with animated spinner and comfortable lingering timer."""
        # If legacy call routed export link here, redirect directly to right-side capsule
        if is_export_link:
            self._set_export_status(text, state="busy" if spinning else "success", timeout_ms=timeout_ms)
            return

        # Lock status during batch import so auxiliary switch events don't wipe it
        if getattr(self, "_is_importing", False):
            if self._stale_import_lock():
                logger.warning("Clearing stale import status lock")
                self._is_importing = False
            else:
                is_import_msg = any(text.startswith(prefix) for prefix in [
                    "正在导入", "导入底片", "正在恢复", "底片库", "会话底片库", "已导入", "已取消",
                    "正在载入", "载入会话", "载入底片", "会话已恢复", "会话",
                    "已跳过", "所选",
                    "正在保存会话"
                ])
                if not is_import_msg:
                    return
        # Lock status during multi-selection so photo click doesn't wipe selection count
        if getattr(self, "_is_multi_selecting", False) and not spinning:
            if not text.startswith(("已选择", "已选中", "已全选")):
                return

        if hasattr(self, "_status_timer"):
            self._status_timer.stop()

        if spinning:
            self.spinner.setVisible(True)
            self.spinner.start()
        else:
            self.spinner.stop()
            self.spinner.setVisible(False)

        self.status_label.setText(text)
        self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
        colors = theme_manager.get_colors() if hasattr(theme_manager, 'get_colors') else {'text_muted': '#94a3b8'}
        self.status_label.setStyleSheet(f"QLabel#status_label {{ color: {colors['text_muted']}; font-size: 11px; }}")
        self.status_label.setToolTip("")

        if text and text.strip():
            if hasattr(self, 'status_slot'):
                self.status_slot.transition_to("status", duration_ms=180)
            if timeout_ms > 0:
                if hasattr(self, "_status_timer"):
                    self._status_timer.start(timeout_ms)
            elif not spinning and not getattr(self, "_is_importing", False) and not getattr(self, "_is_multi_selecting", False):
                # Default lingering duration: 3.5 seconds
                if hasattr(self, "_status_timer"):
                    self._status_timer.start(3500)
        else:
            if hasattr(self, 'status_slot'):
                self.status_slot.transition_to("info", duration_ms=180, on_finished=self._clear_status_text)
            else:
                self._clear_status_text()

        if hasattr(self, 'view_toolbar') and hasattr(self.view_toolbar, '_update_center'):
            self.view_toolbar._update_center()

    def _on_status_timeout(self):
        if getattr(self, "_is_importing", False) or getattr(self, "_is_multi_selecting", False):
            return
        if hasattr(self, 'status_slot'):
            self.status_slot.transition_to("info", duration_ms=180, on_finished=self._clear_status_text)
        else:
            self._clear_status_text()

    def _on_status_label_clicked(self, event):
        pass

    def _set_export_status(self, text: str, state: str = "idle", timeout_ms: int = 0):
        """
        Dedicated export status indicator on the right side of the status bar.
        Completely decoupled from left-side import and general application status.
        state: 'busy' (spinning, cyan), 'success' (green check), 'error' (red), 'idle' (hidden)
        """
        if not hasattr(self, "export_status_container"):
            return

        if hasattr(self, "_export_status_timer"):
            self._export_status_timer.stop()

        if state == "idle" or not text:
            self.export_spinner.stop()
            self.lbl_export_status.setText("")
            self.export_status_container.setVisible(False)
            if hasattr(self, "export_sep"):
                self.export_sep.setVisible(False)
            if hasattr(self, "view_toolbar") and hasattr(self.view_toolbar, "_update_center"):
                self.view_toolbar._update_center()
            return

        self.export_status_container.setVisible(True)
        if hasattr(self, "export_sep"):
            self.export_sep.setVisible(True)
        if hasattr(self, "view_toolbar") and hasattr(self.view_toolbar, "_update_center"):
            self.view_toolbar._update_center()

        self.lbl_export_status.setText(text)

        if state == "busy":
            self.export_spinner.start()
            self.lbl_export_status.setStyleSheet("QLabel#lbl_export_status { color: #f59e0b; font-size: 11px; text-decoration: underline; background: transparent; border: none; }")
            self.export_status_container.setToolTip("正在后台扫描，点击查看扫描队列（快捷键 Ctrl+J）")
        elif state == "success":
            self.export_spinner.stop()
            self.lbl_export_status.setStyleSheet("QLabel#lbl_export_status { color: #f59e0b; font-size: 11px; text-decoration: underline; background: transparent; border: none; }")
            self.export_status_container.setToolTip("扫描完成，点击查看扫描队列（快捷键 Ctrl+J）")
            # Permanent display on success as requested by user
            timeout_ms = 0
        elif state == "error":
            self.export_spinner.stop()
            self.lbl_export_status.setStyleSheet("QLabel#lbl_export_status { color: #ef4444; font-size: 11px; text-decoration: underline; background: transparent; border: none; }")
            self.export_status_container.setToolTip("扫描出错，点击查看扫描队列（快捷键 Ctrl+J）")
            if timeout_ms <= 0:
                timeout_ms = 6000

        if timeout_ms > 0 and hasattr(self, "_export_status_timer"):
            self._export_status_timer.start(timeout_ms)

    def _on_export_status_timeout(self):
        """Fades out/hides the export status capsule once lingering time expires."""
        if hasattr(self, "export_queue") and self.export_queue and self.export_queue.has_active_tasks():
            return
        if hasattr(self, "export_status_container"):
            self.export_spinner.stop()
            self.lbl_export_status.setText("")
            self.export_status_container.setVisible(False)
            if hasattr(self, "export_sep"):
                self.export_sep.setVisible(False)

    def _on_export_status_clicked(self, event):
        """Clicking the right-side export capsule opens the export queue dialog."""
        self.open_export_queue_dialog()

    def open_export_queue_dialog(self):
        """Item 8: Opens the Export Queue Management dialog."""
        if hasattr(self, "export_queue") and self.export_queue:
            dlg = ExportQueueDialog(self.export_queue, self)
            dlg.exec()

    def _on_export_queue_status_changed(self, active_count, total_count):
        """Item 8: Live status updater for background export queue in right capsule."""
        if active_count > 0:
            done_count = total_count - active_count
            self._set_export_status(f"扫描中 ({done_count + 1}/{total_count})", state="busy")
        else:
            if total_count > 0:
                self._set_export_status(f"扫描完成 ({total_count}/{total_count})", state="success", timeout_ms=0)
            else:
                self._set_export_status("", state="idle")

    def action_save_session(self):
        """Item 9 & 11: Saves current editing session (.sdss). Returns True if saved."""
        if not self.photos and not self._current_session_path:
            self.set_app_status("底片库为空，无需保存", timeout_ms=3000)
            return False

        if not self._current_session_path or getattr(self, "_is_temp_session", False):
            return self.action_save_session_as()

        return self._do_save_session(self._current_session_path)

    def action_save_session_as(self):
        """Item 9 & 11: Prompts user to select a .sdss file path and saves session. Returns True if saved."""
        if not self.photos and not self._current_session_path:
            self.set_app_status("底片库为空，无需保存", timeout_ms=3000)
            return False

        # Item 1: Suggest title session name rather than single image filename
        if self._current_session_path and not getattr(self, "_is_temp_session", False):
            default_name = os.path.basename(self._current_session_path)
            if not default_name.lower().endswith(".sdss"):
                default_name = f"{default_name}.sdss"
        else:
            default_name = "未命名会话.sdss"

        last_d = config_manager.get_last_dialog_dir("session")
        initial_p = os.path.join(last_d, default_name) if last_d else default_name

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存当前暗房会话工程", initial_p, "SpektraDarkroom 会话工程 (*.sdss);;所有文件 (*.*)"
        )
        if file_path:
            config_manager.set_last_dialog_dir("session", file_path)
            from core.path_utils import normalize_path
            file_path = normalize_path(file_path)
            self._current_session_path = file_path
            self._is_temp_session = False
            return self._do_save_session(file_path)
        return False

    def _schedule_session_focus_save(self, delay_ms: int = 300):
        """Debounced session focus saving so rapid photo switching does not synchronously read/write disk."""
        if not hasattr(self, "_focus_save_timer"):
            self._focus_save_timer = QTimer(self)
            self._focus_save_timer.setSingleShot(True)
            self._focus_save_timer.timeout.connect(self._update_session_focus_on_disk)
        self._focus_save_timer.start(delay_ms)

    def _update_session_focus_on_disk(self):
        """Silently persist the current active photo focus to session storage without marking the session dirty."""
        if not self.photos or not self._current_photo_path:
            return
        try:
            from core.path_utils import normalize_path
            curr_path = self._current_photo_path
            curr_norm = normalize_path(curr_path)
            # 1. Update config_manager last_session state so fallback restore knows the exact active photo
            all_paths = [normalize_path(p["path"]) for p in self.photos if p and p.get("path")]
            config_manager.save_session_state(all_paths, curr_norm)

            # 2. Update .sdss session file on disk
            if self._current_session_path and os.path.exists(self._current_session_path) and not getattr(self, "_is_temp_session", False):
                session_data = session_cache_manager.load_session_file(self._current_session_path)
                if session_data and isinstance(session_data, dict):
                    session_dir = os.path.dirname(os.path.abspath(self._current_session_path))
                    session_data["active_photo_id"] = self.active_photo_id
                    session_data["active_photo_path"] = curr_norm
                    session_data["active_photo_filename"] = os.path.basename(curr_path)
                    try:
                        session_data["active_photo_rel_path"] = normalize_path(os.path.relpath(os.path.abspath(curr_path), session_dir))
                    except Exception:
                        session_data["active_photo_rel_path"] = None
                    session_cache_manager.save_session_file(self._current_session_path, session_data)
            else:
                self._save_auto_temp_session()
        except Exception as e:
            logger.debug(f"Silent session focus update: {e}")

    def _save_auto_temp_session(self):
        """Item 2: Auto-save temporary session state to auto_saved_session.sdss so work is preserved across restarts."""
        if not self.photos:
            config_manager.clear_auto_saved_session()
            return
        if self._current_session_path and not getattr(self, "_is_temp_session", False):
            return
        try:
            from core.path_utils import normalize_path
            if self.active_photo_id:
                photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                if photo:
                    if photo.get("is_edited", False) or getattr(self, "_photo_is_dirty", False) or self._is_photo_edited(photo):
                        photo["params"] = dict(self.current_params)
                        photo["film_profile"] = self.current_film_stock
                        photo["paper_profile"] = self.current_paper_stock
                        photo["print_mode"] = getattr(self, "current_print_mode", "optical")
                        if getattr(self, "_photo_is_dirty", False) and photo.get("path"):
                            sdc_manager.save_sdc(
                                photo["path"],
                                self.current_film_stock,
                                self.current_paper_stock,
                                self.current_params,
                                print_mode=getattr(self, "current_print_mode", "optical")
                            )
                            self._photo_is_dirty = False

            session_photos = []
            for p in self.photos:
                if not p or not p.get("path"):
                    continue
                p_path = normalize_path(p["path"])
                session_photos.append({
                    "path": p_path,
                    "filename": os.path.basename(p_path),
                    "rating": p.get("rating", 0)
                })

            from core.version import VERSION_STRING
            session_data = {
                "format": "SpektraDarkroomSession",
                "version": VERSION_STRING,
                "saved_at": datetime.now().isoformat(),
                "active_photo_id": self.active_photo_id,
                "active_photo_path": normalize_path(self._current_photo_path) if self._current_photo_path else None,
                "active_photo_filename": os.path.basename(self._current_photo_path) if self._current_photo_path else None,
                "photos": session_photos
            }
            temp_path = config_manager.get_auto_saved_session_path()
            session_cache_manager.save_session_file(temp_path, session_data)
        except Exception as e:
            logger.warning(f"Error auto-saving temporary session: {e}")

    def _do_save_session(self, file_path):
        self.set_app_status("正在保存会话…", spinning=True)
        try:
            return self._write_session(file_path)
        except Exception as e:
            logger.error(f"Failed to save session {file_path}: {e}")
            self.set_app_status(f"会话保存失败：{e}", timeout_ms=4000)
            return False

    def _write_session(self, file_path):
        # Flush current dirty photo parameters to .sdc sidecar
        if self._photo_is_dirty and self._current_photo_path:
            curr_target_p = next((x for x in self.photos if x and x.get("id") == self.active_photo_id), None)
            try:
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    self.current_paper_stock,
                    self.current_params,
                    print_mode=getattr(self, "current_print_mode", "optical")
                )
            except Exception as e:
                logger.warning(f"Could not flush sidecar for {self._current_photo_path}: {e}")
            self._photo_is_dirty = False
            if self.active_photo_id:
                if curr_target_p:
                    curr_target_p["is_dirty"] = False
                    curr_target_p["is_edited"] = True
            self.filmstrip.set_photo_dirty(self.active_photo_id, True)

        # Flush any other dirty photos in session to their respective .sdc sidecars
        for p in self.photos:
            if not p or not p.get("path"):
                continue
            if p.get("is_dirty"):
                try:
                    sdc_manager.save_sdc(
                        p["path"],
                        p.get("film_profile", "none"),
                        p.get("paper_profile", "none"),
                        p.get("params", {}),
                        print_mode=p.get("print_mode", "optical")
                    )
                    p["is_dirty"] = False
                    p["is_edited"] = True
                except Exception as e:
                    logger.warning(f"Could not flush sidecar for {p.get('path')}: {e}")

        from core.path_utils import normalize_path
        session_dir = os.path.dirname(os.path.abspath(file_path))
        session_photos = []
        for p in self.photos:
            if not p or not p.get("path"):
                continue
            p_abs = os.path.abspath(p["path"])
            rel_p = None
            try:
                rel_p = normalize_path(os.path.relpath(p_abs, session_dir))
            except Exception:
                rel_p = None

            session_photos.append({
                "path": normalize_path(p["path"]),
                "rel_path": rel_p,
                "filename": os.path.basename(p["path"]),
                "rating": p.get("rating", 0)
            })

        active_rel = None
        if self._current_photo_path:
            try:
                active_rel = normalize_path(os.path.relpath(os.path.abspath(self._current_photo_path), session_dir))
            except Exception:
                active_rel = None

        session_data = {
            "format": "SpektraDarkroomSession",
            "version": "2.1",
            "saved_at": datetime.now().isoformat(),
            "active_photo_id": self.active_photo_id,
            "active_photo_path": normalize_path(self._current_photo_path) if self._current_photo_path else None,
            "active_photo_rel_path": active_rel,
            "active_photo_filename": os.path.basename(self._current_photo_path) if self._current_photo_path else None,
            "photos": session_photos
        }

        ok = session_cache_manager.save_session_file(file_path, session_data)
        if ok:
            fn = os.path.basename(file_path)
            self.set_app_status(f"会话已保存：{fn}", timeout_ms=3000)
            self._session_is_dirty = False
            self._is_temp_session = False
            self._update_window_title()
            config_manager.add_recent_session(file_path)
            config_manager.save_last_session_path(file_path)
            self._update_recent_menu()
            return True
        else:
            self.set_app_status("会话保存失败：未能写入会话文件", timeout_ms=4000)
            return False

    def _close_needs_prompt(self) -> bool:
        """True when closing/leaving would throw away unsaved session work."""
        if not getattr(self, "_session_is_dirty", False):
            return False
        if not (self.photos or self._current_session_path):
            return False
        if getattr(self, "_suppress_close_confirm", False):
            return False
        if bool(os.environ.get("FAST_TEST")) or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return False
        return True

    def _prompt_save_if_session_dirty(self) -> bool:
        """Prompt user to save if current session has unsaved modifications.
        Returns True if safe to proceed (clean, saved, or discarded), False if user cancelled.
        """
        if not self._close_needs_prompt():
            return True
        sess_name = os.path.basename(self._current_session_path) if self._current_session_path else "未命名会话"
        dialog = UnsavedSessionDialog(sess_name, self)
        res = dialog.exec()
        if res == UnsavedSessionDialog.CANCEL:
            return False
        elif res == UnsavedSessionDialog.SAVE:
            # 只有真的写成功了才算"已保存"。用户可能在 Windows 的"另存为"里点取消
            # （action_save_session 返回 False），这种情况必须当成取消关闭，不能因为
            # 别处顺手清掉了 dirty 标志就把会话连同窗口一起丢掉。
            if not self.action_save_session():
                return False
            if getattr(self, "_session_is_dirty", False):
                return False
        # elif res == UnsavedSessionDialog.DISCARD: proceed
        return True

    def _run_close_prompt(self):
        """Ask the unsaved-session question outside the WM_CLOSE call stack, then close."""
        caption_was_enabled = _FRAME_MODE != "clean"
        if caption_was_enabled:
            self._set_native_caption(False)
        try:
            proceed = self._prompt_save_if_session_dirty()
        finally:
            self._close_prompt_running = False

        if not proceed:
            if caption_was_enabled:
                self._set_native_caption(True)
            # 取消保存之后窗口可能已经被系统停用（原生保存对话框关掉时主窗口不再有焦点）。
            # 显式拉回前台，否则看起来就像"应用被关掉了"，下一次启动还找不到这次的工作。
            if not self.isVisible():
                self.show()
            self.raise_()
            self.activateWindow()
            # The next close attempt must dwell again; this one never completed.
            self._close_dwell_done = False
            return

        self._close_approved = True
        try:
            self.close()
        finally:
            self._close_approved = False

    def action_close_session(self):
        """Item 4: Closes the current session, prompts if dirty, and returns cleanly to welcome page."""
        if not self.photos and not self._current_session_path and getattr(self, 'stack', None) and self.stack.currentIndex() == 0:
            return
        if not self._prompt_save_if_session_dirty():
            return
        self._perform_close_session()

    def _perform_close_session(self):
        """Executes full clean teardown of current session and returns to startup welcome view."""
        # 1. Silently persist individual photo parameters to .sdc
        try:
            if self._photo_is_dirty and self._current_photo_path:
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    self.current_paper_stock,
                    self.current_params,
                    print_mode=getattr(self, "current_print_mode", "optical")
                )
                self._photo_is_dirty = False
            for p in self.photos:
                if not p:
                    continue
                if p.get("is_dirty") and p.get("path"):
                    sdc_manager.save_sdc(
                        p["path"],
                        p.get("film_profile", self.current_film_stock),
                        p.get("paper_profile", self.current_paper_stock),
                        p.get("params", self.current_params),
                        print_mode=p.get("print_mode", getattr(self, "current_print_mode", "optical"))
                    )
                    p["is_dirty"] = False
        except Exception as e:
            logger.warning(f"Auto-saving .sdc on close session: {e}")

        # 0. Cancel any active background import worker and slot workers immediately
        self._cancel_active_import(notify_user=False)

        # 2. Clear engine and photo arrays
        if hasattr(self, 'engine'):
            self.engine.clear_all_photos()
        self.photos.clear()
        self.active_photo_id = None
        self._current_photo_path = None
        self._current_photo_thumb = None
        self.filmstrip.set_photos([], None)
        self.canvas.set_image(None)

        # 3. Reset session states
        self._current_session_path = None
        self._is_temp_session = False
        self._session_is_dirty = False
        self._photo_is_dirty = False

        # 4. Clear config last session path so restart shows welcome screen
        config_manager.save_last_session_path("")

        # 5. Reset UI components and action states
        self.btn_title_export.setVisible(False)
        if hasattr(self, 'act_export'):
            self.act_export.setEnabled(False)
        if hasattr(self, 'act_quick_export'):
            self.act_quick_export.setEnabled(False)
        if hasattr(self, 'act_export_lut'):
            self.act_export_lut.setEnabled(False)
        if hasattr(self, 'act_save_session'):
            self.act_save_session.setEnabled(False)
        if hasattr(self, 'act_save_session_as'):
            self.act_save_session_as.setEnabled(False)
        if hasattr(self, 'act_close_session'):
            self.act_close_session.setEnabled(False)

        if hasattr(self, 'exif_widget'):
            self.exif_widget.set_exif_data({})
        if hasattr(self, 'histogram_exif_bar'):
            self.histogram_exif_bar.set_exif_data({})
        if hasattr(self, 'histogram_widget'):
            self.histogram_widget.clear()
        elif hasattr(self, 'histogram'):
            self.histogram.clear()

        for card in self.film_cards.values():
            card.set_thumbnail(None)
        for card in self.paper_cards.values():
            card.set_thumbnail(None)

        # 6. Switch back to startup welcome screen
        self.stack.setCurrentIndexAnimated(0, duration_ms=250)
        self._update_window_title()
        self.set_app_status("已关闭当前会话", timeout_ms=3000)

    def action_open_session(self):
        """Loads and restores session from a .sdss file with dirty check."""
        if not self._prompt_save_if_session_dirty():
            return
        last_d = config_manager.get_last_dialog_dir("session")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开暗房会话工程", last_d, "SpektraDarkroom 会话工程 (*.sdss);;所有文件 (*.*)"
        )
        if file_path:
            config_manager.set_last_dialog_dir("session", file_path)
            from core.path_utils import normalize_path
            if self._current_session_path and normalize_path(file_path).lower() == normalize_path(self._current_session_path).lower():
                return
            self._open_session_by_path(file_path)

    def _open_session_by_path(self, file_path):
        if not file_path or not os.path.exists(file_path):
            self.set_app_status("载入会话失败：会话文件不存在", timeout_ms=4000)
            return

        self.set_app_status("正在载入会话…", spinning=True)
        session_data = session_cache_manager.load_session_file(file_path)
        if not session_data or "photos" not in session_data:
            self.set_app_status("载入会话失败：会话文件已损坏", timeout_ms=4000)
            return

        from core.path_utils import normalize_path
        session_dir = os.path.dirname(os.path.abspath(file_path))
        photo_entries = session_data["photos"]

        resolved_files = []
        missing_entries = []
        path_updated = False

        for p in photo_entries:
            # base_ev is re-derived from the file on every load -- it is a metering result,
            # not an edit. Sessions written by an older build may still carry one, and
            # restoring it would pin the exposure to a stale value forever.
            if isinstance(p.get("params"), dict):
                p["params"].pop("base_ev", None)
            orig_p = p.get("path", "")
            rel_p = p.get("rel_path")
            fname = p.get("filename") or (os.path.basename(orig_p) if orig_p else "")

            resolved = None
            # 1. Direct absolute path check
            if orig_p and os.path.exists(orig_p):
                resolved = normalize_path(orig_p)
            # 2. Relative path relative to .sdss folder
            elif rel_p:
                cand = os.path.abspath(os.path.join(session_dir, rel_p))
                if os.path.exists(cand):
                    resolved = normalize_path(cand)
                    path_updated = True

            # 3. Same directory as .sdss
            if not resolved and fname:
                cand = os.path.abspath(os.path.join(session_dir, fname))
                if os.path.exists(cand):
                    resolved = normalize_path(cand)
                    path_updated = True

            # 4. Standard subdirectories in session folder
            if not resolved and fname:
                for sub in ("Capture", "capture", "raw", "RAW", "images", "photos"):
                    cand = os.path.abspath(os.path.join(session_dir, sub, fname))
                    if os.path.exists(cand):
                        resolved = normalize_path(cand)
                        path_updated = True
                        break

            if resolved:
                resolved_files.append(resolved)
                p["path"] = resolved
            else:
                missing_entries.append({
                    "filename": fname,
                    "orig_path": orig_p,
                    "resolved_path": None,
                    "entry": p
                })

        # If there are missing files, prompt user with RelinkMediaDialog
        if missing_entries:
            from ui.relink_dialog import RelinkMediaDialog
            dlg = RelinkMediaDialog(missing_entries, total_photos_count=len(photo_entries), parent=self)
            dlg.exec()
            if dlg.action_result == RelinkMediaDialog.STATUS_RELINKED:
                mapping = dlg.get_relinked_mapping()
                for m in missing_entries:
                    new_p = mapping.get(m["orig_path"]) or m.get("resolved_path")
                    if new_p and os.path.exists(new_p):
                        norm_p = normalize_path(new_p)
                        resolved_files.append(norm_p)
                        m["entry"]["path"] = norm_p
                        path_updated = True
                if hasattr(dlg, "get_removed_orig_paths") and dlg.get_removed_orig_paths():
                    removed_paths = {normalize_path(p) for p in dlg.get_removed_orig_paths()}
                    photo_entries = [p for p in photo_entries if normalize_path(p.get("path", "")) not in removed_paths]
                    session_data["photos"] = photo_entries
                    path_updated = True
            elif dlg.action_result == RelinkMediaDialog.STATUS_OFFLINE:
                # User chose to keep offline and open with only found photos
                if hasattr(dlg, "get_removed_orig_paths") and dlg.get_removed_orig_paths():
                    removed_paths = {normalize_path(p) for p in dlg.get_removed_orig_paths()}
                    photo_entries = [p for p in photo_entries if normalize_path(p.get("path", "")) not in removed_paths]
                    session_data["photos"] = photo_entries
                    path_updated = True
                    try:
                        if is_temp and not photo_entries:
                            config_manager.clear_auto_saved_session()
                            config_manager.save_last_session_path(None)
                        else:
                            session_cache_manager.save_session_file(file_path, session_data)
                    except Exception as e:
                        logger.warning(f"Failed to update session file after removing offline media: {e}")
            elif dlg.action_result == RelinkMediaDialog.STATUS_REMOVE_OFFLINE:
                # User explicitly chose to remove offline media from the session!
                mapping = dlg.get_relinked_mapping()
                # 1. Any photos that were successfully relinked in the dialog should be kept and loaded
                for m in missing_entries:
                    new_p = mapping.get(m["orig_path"]) or m.get("resolved_path")
                    if new_p and os.path.exists(new_p):
                        norm_p = normalize_path(new_p)
                        resolved_files.append(norm_p)
                        m["entry"]["path"] = norm_p

                # 2. Determine which entries are still missing/offline
                unresolved_orig_paths = {
                    normalize_path(m["orig_path"]) for m in missing_entries
                    if not (mapping.get(m["orig_path"]) or m.get("resolved_path"))
                }
                if hasattr(dlg, "get_removed_orig_paths"):
                    unresolved_orig_paths.update(normalize_path(p) for p in dlg.get_removed_orig_paths())

                # 3. Filter photo_entries to remove the offline ones
                photo_entries = [p for p in photo_entries if normalize_path(p.get("path", "")) not in unresolved_orig_paths]
                path_updated = True

                # 4. Save the updated session file immediately so this dialog NEVER pops up again for these photos!
                session_data["photos"] = photo_entries
                try:
                    if is_temp and not photo_entries:
                        config_manager.clear_auto_saved_session()
                        config_manager.save_last_session_path(None)
                    else:
                        session_cache_manager.save_session_file(file_path, session_data)
                except Exception as e:
                    logger.warning(f"Failed to update session file after removing offline media: {e}")

                self.set_app_status(f"已从会话中移除 {len(unresolved_orig_paths)} 张脱机底片", timeout_ms=3000)
            else:
                # User cancelled opening session
                self.set_app_status("已取消载入会话", timeout_ms=2500)
                return

        if not resolved_files and photo_entries:
            self.set_app_status("载入会话失败：所有底片均已丢失且未关联", timeout_ms=4000)
            return

        # Cancel any active imports from previous session immediately
        self._cancel_active_import(notify_user=False)

        # 1. Clean previous session photos and workspace completely to prevent photo stacking
        if hasattr(self, 'engine'):
            self.engine.clear_all_photos()
        self.photos.clear()
        self.active_photo_id = None
        self._current_photo_path = None
        self.filmstrip.set_photos([], None)
        self.canvas.set_image(None)

        auto_temp = config_manager.get_auto_saved_session_path()
        is_temp = (normalize_path(file_path).lower() == normalize_path(auto_temp).lower())
        self._is_temp_session = is_temp
        if is_temp:
            self._current_session_path = None
        else:
            self._current_session_path = normalize_path(file_path)

        # Active path resolution
        active_path = session_data.get("active_photo_path")
        if not active_path:
            afn = session_data.get("active_photo_filename")
            if afn:
                for rf in resolved_files:
                    if os.path.basename(rf).lower() == afn.lower():
                        active_path = rf
                        break
        if not active_path:
            _, cfg_active = config_manager.get_session_state()
            if cfg_active and cfg_active in resolved_files:
                active_path = cfg_active
        if active_path:
            if not os.path.exists(active_path):
                active_fn = os.path.basename(active_path).lower()
                for rf in resolved_files:
                    if os.path.basename(rf).lower() == active_fn:
                        active_path = rf
                        break
        if not active_path or not os.path.exists(active_path):
            active_path = resolved_files[0] if resolved_files else None

        fn = "未命名会话" if is_temp else os.path.basename(file_path)

        if not is_temp:
            config_manager.add_recent_session(file_path)
            config_manager.save_last_session_path(file_path)
            self._update_recent_menu()

        # If any path was relinked/changed, mark session dirty so user can re-save it with new paths
        self._session_is_dirty = path_updated
        self._update_window_title()

        # Build ratings mapping from session photo entries (ratings belong to session)
        ratings_map = {}
        for pe in photo_entries:
            if isinstance(pe, dict) and pe.get("path"):
                ratings_map[normalize_path(pe["path"])] = max(0, min(5, int(pe.get("rating", 0) or 0)))

        if resolved_files:
            self._restore_saved_session(resolved_files, active_file=active_path, session_name=fn, path_updated=path_updated, ratings_map=ratings_map)
        else:
            if is_temp:
                # If opening an empty temporary/auto-saved session, stay on the welcome start screen
                self.stack.setCurrentIndex(0)
                self.canvas.set_has_library_photos(False)
                self.canvas.set_image(None)
                self.filmstrip.set_photos([], None)
                self.btn_title_export.setVisible(False)
                self._current_session_path = None
                self._is_temp_session = False
                config_manager.save_last_session_path(None)
                self._update_window_title()
            else:
                self.stack.setCurrentIndexAnimated(1, duration_ms=250)
                self.canvas.set_has_library_photos(False)
                self.canvas.set_image(None)
                self.filmstrip.set_photos([], None)
                self.btn_title_export.setVisible(False)
                self.thumb_controller.reset_all_cards_to_color_pills(self.film_cards, self.paper_cards)
                self.set_app_status(f"会话已恢复：{fn}（空工程）", timeout_ms=2500)
                self._update_window_title()

    def action_quick_export(self, target_ids=None):
        """Item 4: Quick export selected or active photo(s) using remembered last export settings."""
        last_cfg = config_manager.get_last_export_settings()
        if not last_cfg:
            self.set_app_status("未记录上次扫描参数，请先执行一次常规扫描", timeout_ms=3000)
            self.action_export_image()
            return

        if target_ids:
            pids = target_ids
        elif hasattr(self, "filmstrip") and self.filmstrip._selected_ids:
            pids = list(self.filmstrip._selected_ids)
        elif self.active_photo_id:
            pids = [self.active_photo_id]
        else:
            return

        photos_to_export = [p for p in self.photos if p and p.get("id") in pids]
        if not photos_to_export:
            return

        fmt = last_cfg.get("format", "jpeg")
        fmt_str = str(fmt).lower()
        ext = ".tif" if "tif" in fmt_str else (".jxl" if "jxl" in fmt_str else (".png" if "png" in fmt_str else ".jpg"))
        out_dir = config_manager.get_last_export_dir()
        if not out_dir or not os.path.isdir(out_dir):
            if self._current_photo_path:
                out_dir = os.path.dirname(self._current_photo_path)
            else:
                out_dir = os.path.expanduser("~")

        enqueued_count = 0
        new_tasks = []
        import uuid
        used_targets = set()
        from ui.export_dialog import (
            EXPORT_SIZE_MODE_KEY, EXPORT_SIZE_PREF_KEY, resolve_output_pixels)
        overwrite = bool(last_cfg.get("overwrite", True))

        for p in photos_to_export:
            src = p["path"]
            base_name = os.path.splitext(os.path.basename(src))[0]
            out_fn = f"{base_name}{ext}"
            out_file = os.path.join(out_dir, out_fn)

            if not overwrite:
                out_file = self._resolve_export_target(
                    out_file, src_path=src, used=used_targets, allow_overwrite=False)
            else:
                out_file = self._resolve_export_target(
                    out_file, src_path=src, used=used_targets, allow_overwrite=True)
            out_fn = os.path.basename(out_file)

            # Start every photo from its own defaults: quick export must never inherit the
            # recipe of whatever happens to be on the canvas. A frame that was never edited
            # therefore exports through the digital bypass path exactly as shot.
            sdc = sdc_manager.load_sdc(src)
            exif_d = p.get("exif", {})
            p_base_temp = float(exif_d.get("color_temp", 5500.0) if exif_d else 5500.0)
            p_base_tint = float(exif_d.get("tint", 1.0) if exif_d else 1.0)
            task_params = self.get_default_params(p_base_temp, p_base_tint)
            if p.get("film_profile"):
                film = p.get("film_profile")
                paper = p.get("paper_profile", "none")
                task_params.update(p.get("params") or {})
            elif sdc:
                film = sdc.get("film_profile", "none")
                paper = sdc.get("paper_profile", "none")
                if "params" in sdc and isinstance(sdc["params"], dict):
                    task_params.update(sdc["params"])
            elif src == self._current_photo_path:
                # Only the active frame has a live, unsaved recipe on the canvas.
                film = self.current_film_stock
                paper = self.current_paper_stock
                task_params.update(self.current_params)
            else:
                film = "none"
                paper = "none"

            task_params.update(last_cfg)
            # Drop the geometry captured for a *different* photo, then resolve it again here.
            task_params.pop("out_size", None)
            task_params.pop("out_long_edge", None)
            task_params.pop("path", None)
            crop_cfg = (p.get("params") or {}).get("crop") if isinstance(p.get("params"), dict) else None
            if crop_cfg is None and src == self._current_photo_path:
                crop_cfg = self.current_params.get("crop")
            out_size = resolve_output_pixels(
                last_cfg.get(EXPORT_SIZE_MODE_KEY, "original"),
                last_cfg.get(EXPORT_SIZE_PREF_KEY, 0),
                p.get("width") or 0, p.get("height") or 0, crop_cfg)
            task_params["out_size"] = list(out_size)
            task_params["out_long_edge"] = max(out_size)
            task_params["scale_pct"] = 100
            task_params["source_path"] = src
            task_params["film_stock"] = film
            task_params["paper_stock"] = paper
            task_params["print_mode"] = p.get("print_mode", getattr(self, "current_print_mode", "optical"))

            base_ev = float(p.get("base_ev", 0.0))
            if abs(base_ev) < 0.0001 and hasattr(self, "engine"):
                matched_ep = next((ep for ep in self.engine.session_photos if ep.get("path") == src), None)
                if matched_ep and "base_ev" in matched_ep:
                    base_ev = float(matched_ep["base_ev"])
            if abs(base_ev) < 0.0001 and src == self._current_photo_path:
                base_ev = float(getattr(self, "_active_base_ev", 0.0))
            task_params["base_ev"] = base_ev
            # White-balance reference belongs to the photo, not to the canvas that happens
            # to be open. Only the active frame legitimately borrows the live base values.
            if src == self._current_photo_path and not p.get("film_profile") and not sdc:
                task_params["base_temp"] = float(getattr(self, "_active_base_temp", p_base_temp))
                task_params["base_tint"] = float(getattr(self, "_active_base_tint", p_base_tint))
            else:
                task_params["base_temp"] = p_base_temp
                task_params["base_tint"] = p_base_tint
            task_params["engine"] = last_cfg.get("engine", "wysiwyg")

            pix = None
            if p.get("thumbnail_rgb") is not None:
                trgb = p["thumbnail_rgb"]
                th, tw = trgb.shape[:2]
                qimg = QImage(trgb.data, tw, th, tw * 3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg)

            tid = f"task_{uuid.uuid4().hex[:8]}"
            task = ExportTask(tid, os.path.basename(out_file), src, out_file, task_params, thumbnail=pix)
            new_tasks.append(task)
            enqueued_count += 1

        if hasattr(self.export_queue, "enqueue_batch"):
            self.export_queue.enqueue_batch(new_tasks)
        else:
            for t in new_tasks:
                self.export_queue.enqueue(t)

        self._set_export_status(f"已加入扫描队列 ({enqueued_count} 张)", state="busy")

    def push_undo_state(self, action: str = None):
        state = {
            "type": "single_photo_state",
            "photo_id": self.active_photo_id,
            "photo_path": self._current_photo_path,
            "film": getattr(self, "current_film_stock", "none"),
            "paper": getattr(self, "current_paper_stock", "none"),
            "print_mode": getattr(self, "current_print_mode", "optical"),
            "params": dict(self.current_params)
        }
        if action:
            state["action"] = action
        # Avoid pushing duplicate states that match the top of the undo stack
        if self._undo_stack:
            top = self._undo_stack[-1]
            top_pid = top.get("photo_id") if top.get("type") == "single_photo_state" else None
            if top.get("type") in ("batch_photos_state", "library_photos_state") or (top_pid and top_pid != self.active_photo_id):
                baseline = getattr(self, "_active_photo_baseline_state", None)
                if baseline and baseline.get("photo_id") == self.active_photo_id:
                    if not (baseline.get("film") == state["film"] and
                            baseline.get("paper") == state["paper"] and
                            baseline.get("print_mode") == state["print_mode"] and
                            baseline.get("params") == state["params"]):
                        self._undo_stack.append(baseline)
            elif (top.get("film") == state["film"] and
                  top.get("paper") == state["paper"] and
                  top.get("print_mode") == state["print_mode"] and
                  top.get("params") == state["params"]):
                return

        self._undo_stack.append(state)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_menu_actions_state()

    def _push_raw_state(self, state: dict):
        """Pushes a custom state (e.g. batch operation snapshot) to the undo stack."""
        self._undo_stack.append(state)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_menu_actions_state()

    def undo(self):
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay.undo():
                self.set_app_status("已撤销裁切", timeout_ms=2000)
            else:
                self.set_app_status("已回到最初的裁切状态", timeout_ms=2000)
            self._update_menu_actions_state()
            return

        if len(self._undo_stack) <= 1:
            self.set_app_status("已回到最初的历史状态", timeout_ms=2000)
            return

        curr_state = self._undo_stack.pop()
        self._redo_stack.append(curr_state)
        if curr_state.get("type") == "photo_rating_state":
            old_ratings = curr_state.get("old_ratings")
            if old_ratings:
                self._apply_rating_history_state({"ratings": old_ratings})
        prev_state = self._undo_stack[-1]
        self._apply_history_state(prev_state)

        act = curr_state.get("action") or prev_state.get("action")
        if act == "clear_sdc":
            self.set_app_status("已撤销删除暗房配置", timeout_ms=2000)
        elif act == "paste_params":
            self.set_app_status("已撤销粘贴暗房配方", timeout_ms=2000)
        elif act == "clear_crop":
            self.set_app_status("已撤销清除裁切", timeout_ms=2000)
        elif act in ("remove_photos", "delete_photos"):
            self.set_app_status("已撤销移除底片", timeout_ms=2000)
        elif act == "clear_photos":
            self.set_app_status("已撤销清空底片库", timeout_ms=2000)
        elif act == "sort_photos":
            self.set_app_status("已撤销底片排序", timeout_ms=2000)
        elif act == "rate_photos":
            self.set_app_status("已撤销底片评分", timeout_ms=2000)
        elif act == "reset_all_params":
            self.set_app_status("已撤销重置所有参数", timeout_ms=2000)
        elif act == "reset_to_open":
            self.set_app_status("已撤销恢复至原始底片", timeout_ms=2000)
        elif act and act.startswith("reset_track_"):
            track_names = {
                "exposure": "曝光",
                "optics": "光学",
                "chemistry": "化学",
                "enlarger": "放大机",
                "halation": "光晕",
                "grain": "颗粒"
            }
            tname = track_names.get(act[12:], "模块")
            self.set_app_status(f"已撤销重置{tname}模块", timeout_ms=2000)
        elif act == "auto_meter":
            self.set_app_status("已撤销基准测光", timeout_ms=2000)
        else:
            self.set_app_status("已撤销", timeout_ms=2000)
        self._update_menu_actions_state()

    def redo(self):
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay.redo():
                self.set_app_status("已重做裁切", timeout_ms=2000)
            else:
                self.set_app_status("已回到最新的裁切状态", timeout_ms=2000)
            self._update_menu_actions_state()
            return

        if not self._redo_stack:
            self.set_app_status("已回到最新的历史状态", timeout_ms=2000)
            return

        nxt_state = self._redo_stack.pop()
        self._undo_stack.append(nxt_state)
        self._apply_history_state(nxt_state)

        act = nxt_state.get("action")
        if act == "clear_sdc":
            self.set_app_status("已重做删除暗房配置", timeout_ms=2000)
        elif act == "paste_params":
            self.set_app_status("已重做粘贴暗房配方", timeout_ms=2000)
        elif act == "clear_crop":
            self.set_app_status("已重做清除裁切", timeout_ms=2000)
        elif act in ("remove_photos", "delete_photos"):
            self.set_app_status("已重做移除底片", timeout_ms=2000)
        elif act == "clear_photos":
            self.set_app_status("已重做清空底片库", timeout_ms=2000)
        elif act == "sort_photos":
            self.set_app_status("已重做底片排序", timeout_ms=2000)
        elif act == "rate_photos":
            self.set_app_status("已重做底片评分", timeout_ms=2000)
        elif act == "reset_all_params":
            self.set_app_status("已重做重置所有参数", timeout_ms=2000)
        elif act == "reset_to_open":
            self.set_app_status("已重做恢复至原始底片", timeout_ms=2000)
        elif act and act.startswith("reset_track_"):
            track_names = {
                "exposure": "曝光",
                "optics": "光学",
                "chemistry": "化学",
                "enlarger": "放大机",
                "halation": "光晕",
                "grain": "颗粒"
            }
            tname = track_names.get(act[12:], "模块")
            self.set_app_status(f"已重做重置{tname}模块", timeout_ms=2000)
        elif act == "auto_meter":
            self.set_app_status("已重做基准测光", timeout_ms=2000)
        else:
            self.set_app_status("已重做", timeout_ms=2000)
        self._update_menu_actions_state()

    def _is_photo_edited(self, photo_entry=None) -> bool:
        """Determines whether the target or current active photo has been modified."""
        p = photo_entry
        if p is None and self.active_photo_id:
            p = next((x for x in self.photos if x.get("id") == self.active_photo_id), None)
        if not p:
            return False

        # 1. Direct dirty or edited flags
        if p.get("is_dirty") or p.get("is_edited"):
            return True
        if p.get("id") == self.active_photo_id and getattr(self, "_photo_is_dirty", False):
            return True

        # 2. Check if film or paper stock is selected
        film = p.get("film_profile") or (self.current_film_stock if p.get("id") == self.active_photo_id else "none")
        paper = p.get("paper_profile") or (self.current_paper_stock if p.get("id") == self.active_photo_id else "none")
        if film not in ("none", "", None) or paper not in ("none", "", None):
            return True

        # 3. Check if an SDC file exists on disk and contains actual edits
        p_path = p.get("path")
        if p_path and sdc_manager.has_sdc(p_path):
            sdc = sdc_manager.load_sdc(p_path)
            if sdc:
                s_film = sdc.get("film_profile", "none")
                s_paper = sdc.get("paper_profile", "none")
                if s_film not in ("none", "", None) or s_paper not in ("none", "", None):
                    return True
                if self._is_params_modified(sdc.get("params", {}), p):
                    return True
            else:
                return True

        # 4. Check if parameters deviate from default baseline
        params = p.get("params") or (self.current_params if p.get("id") == self.active_photo_id else {})
        if self._is_params_modified(params, p):
            return True

        return False

    def _is_crop_modified(self, crop_data) -> bool:
        """Helper to determine whether crop settings deviate from the default uncropped rectangle."""
        if not crop_data:
            return False
        if isinstance(crop_data, dict):
            if crop_data.get("enabled", False):
                return True
            w = float(crop_data.get("w", 1.0))
            h = float(crop_data.get("h", 1.0))
            x = float(crop_data.get("x", 0.0))
            y = float(crop_data.get("y", 0.0))
            rot = int(crop_data.get("rot_90", 0)) % 360
            ang = float(crop_data.get("angle", 0.0))
            if (abs(w - 1.0) > 0.001 or abs(h - 1.0) > 0.001 or
                abs(x) > 0.001 or abs(y) > 0.001 or rot != 0 or abs(ang) > 0.01 or
                crop_data.get("flip_h") or crop_data.get("flip_v")):
                return True
        elif isinstance(crop_data, bool):
            return crop_data
        return False

    def _is_params_modified(self, params_dict: dict, photo_entry=None) -> bool:
        """Checks if a parameter dictionary has any non-default values."""
        if not params_dict:
            return False
        base_temp = 5500.0
        base_tint = 1.0
        if photo_entry:
            exif = photo_entry.get("meta") or photo_entry.get("exif") or {}
            base_temp = float(exif.get("color_temp", getattr(self, "_active_base_temp", 5500.0)))
            base_tint = float(exif.get("tint", getattr(self, "_active_base_tint", 1.0)))
        defaults = self.get_default_params(base_temp, base_tint)

        # Check crop
        if self._is_crop_modified(params_dict.get("crop")):
            return True

        # Check lens optical modifications separately
        lens_cfg = params_dict.get("lens")
        if isinstance(lens_cfg, dict):
            if (
                bool(lens_cfg.get("profile_enabled", False)) or
                bool(lens_cfg.get("remove_ca", False)) or
                abs(float(lens_cfg.get("profile_distortion", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("profile_vignette", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("manual_distortion", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("manual_vignette_amount", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("manual_vignette_midpoint", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("k1", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("k2", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("k3", 0.0) or 0.0)) > 0.01 or
                abs(float(lens_cfg.get("vignette_amount", 0.0) or 0.0)) > 0.01
            ):
                return True

        # Ignore internal non-color keys, lens (checked above), and viewport flags
        ignored_keys = {
            "print_mode", "custom_wb", "film_format_mm", "base_temp", "base_tint", "base_ev",
            "crop", "lens", "split_x", "view_mode"
        }
        if not params_dict.get("custom_wb", False):
            ignored_keys.add("color_temp")
            ignored_keys.add("tint")
        for k, def_val in defaults.items():
            if k in ignored_keys:
                continue
            cur_val = params_dict.get(k)
            if cur_val is None:
                continue
            if isinstance(def_val, float):
                if abs(float(cur_val) - def_val) >= 1e-4:
                    return True
            elif cur_val != def_val:
                return True
        return False

    def action_select_all_photos(self):
        """Ctrl+A handler: Select all photos in the filmstrip."""
        if hasattr(self, 'filmstrip'):
            self.filmstrip.select_all_photos()
            sel_count = len(self.filmstrip.get_selected_photo_ids())
            if sel_count > 1:
                self._is_multi_selecting = True
                self.set_app_status(f"已选择 {sel_count} 张底片", timeout_ms=0)
            elif sel_count == 1:
                self._is_multi_selecting = False
                self.restore_default_status()

    def action_invert_selection(self):
        """Ctrl+Shift+I handler: Invert selection in the filmstrip."""
        if hasattr(self, 'filmstrip'):
            self.filmstrip.invert_selection()
            sel_count = len(self.filmstrip.get_selected_photo_ids())
            if sel_count > 1:
                self._is_multi_selecting = True
                self.set_app_status(f"已选择 {sel_count} 张底片", timeout_ms=0)
            elif sel_count == 1:
                self._is_multi_selecting = False
                self.restore_default_status()

    def action_copy_params(self, photo_id=None):
        """Ctrl+C handler: Copy all physical darkroom parameters from the currently viewed or specified photo."""
        if hasattr(self, "filmstrip") and hasattr(self.filmstrip, "flush_pending_selection"):
            self.filmstrip.flush_pending_selection()

        target_id = photo_id if (photo_id and isinstance(photo_id, str)) else self.active_photo_id
        if not target_id:
            return

        target_p = next((p for p in self.photos if p and p.get("id") == target_id), None)
        if not target_p or not self._is_photo_edited(target_p):
            self.set_app_status("当前底片为原生底片，没有可复制的暗房配方", timeout_ms=2500)
            return

        exif = target_p.get("meta") or target_p.get("exif") or {}
        src_base_temp = float(exif.get("color_temp", 5500.0))
        src_base_tint = float(exif.get("tint", 1.0))

        if target_id == self.active_photo_id:
            film = self.current_film_stock
            paper = self.current_paper_stock
            pmode = getattr(self, "current_print_mode", "optical")
            raw_params = copy.deepcopy(self.current_params)
        else:
            film = target_p.get("film_profile", "none")
            paper = target_p.get("paper_profile", "none")
            pmode = target_p.get("print_mode", "optical")
            if target_p.get("params"):
                raw_params = copy.deepcopy(target_p["params"])
            elif target_p.get("path"):
                sdc = sdc_manager.load_sdc(target_p["path"])
                if sdc:
                    film = sdc.get("film_profile", film)
                    paper = sdc.get("paper_profile", paper)
                    pmode = sdc.get("print_mode", pmode)
                    raw_params = copy.deepcopy(sdc.get("params", {}))
                else:
                    raw_params = {}
            else:
                raw_params = {}

        params = self.get_default_params(src_base_temp, src_base_tint)
        params.update(raw_params)
        if "halation_amount" not in raw_params and "halation" in raw_params:
            params["halation_amount"] = raw_params["halation"]
        if "boost_ev" not in raw_params and "halation_boost" in raw_params:
            params["boost_ev"] = raw_params["halation_boost"]
        if "crop" not in raw_params and isinstance(target_p.get("crop"), dict):
            params["crop"] = copy.deepcopy(target_p["crop"])
        params.pop("split_x", None)
        params.pop("view_mode", None)

        self._copied_darkroom_profile = {
            "film": film,
            "paper": paper,
            "print_mode": pmode,
            "params": params,
            "is_edited": True
        }
        if hasattr(self, 'filmstrip'):
            self.filmstrip.set_can_paste(True)
        film_desc = "数码原生" if film == "none" else f"胶卷 ({film})"
        self.set_app_status(f"已复制暗房配方：{film_desc}", timeout_ms=2500)
        self._update_menu_actions_state()

    def action_paste_params(self):
        """Ctrl+V handler: Paste copied physical darkroom parameters to all selected photos."""
        if hasattr(self, "filmstrip") and hasattr(self.filmstrip, "flush_pending_selection"):
            self.filmstrip.flush_pending_selection()

        copied = getattr(self, '_copied_darkroom_profile', None)
        if not copied or not copied.get("is_edited"):
            self.set_app_status("剪贴板中没有可用的暗房配方", timeout_ms=2500)
            return

        copied_film = copied["film"]
        copied_paper = copied["paper"]
        copied_print_mode = copied.get("print_mode", "optical")
        copied_params = copy.deepcopy(copied["params"])

        selected_ids = self.filmstrip.get_selected_photo_ids() if hasattr(self, 'filmstrip') else []
        if not selected_ids and self.active_photo_id:
            selected_ids = [self.active_photo_id]

        if not selected_ids:
            return

        default_crop = {
            "enabled": False,
            "x": 0.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "angle": 0.0,
            "rot_90": 0,
            "flip_h": False,
            "flip_v": False,
            "aspect_ratio": "original",
            "is_portrait": False
        }

        # 1. Snapshot pre-paste state for all affected photos
        pre_photos = {}
        for pid in selected_ids:
            p = next((x for x in self.photos if x.get("id") == pid), None)
            if not p:
                continue
            if pid == self.active_photo_id:
                p_film = getattr(self, "current_film_stock", "none")
                p_paper = getattr(self, "current_paper_stock", "none")
                p_pmode = getattr(self, "current_print_mode", "optical")
                p_params = copy.deepcopy(self.current_params)
                p_crop = copy.deepcopy(self.current_params.get("crop") or p.get("crop") or default_crop)
                p_dirty = bool(getattr(self, "_photo_is_dirty", False) or p.get("is_dirty", False))
                p_edited = bool(self._is_photo_edited(p))
            else:
                p_film = p.get("film_profile", "none")
                p_paper = p.get("paper_profile", "none")
                p_pmode = p.get("print_mode", "optical")
                p_params = copy.deepcopy(p.get("params") or {})
                p_crop = copy.deepcopy(p.get("crop") or p_params.get("crop") or default_crop)
                p_dirty = bool(p.get("is_dirty", False))
                p_edited = bool(self._is_photo_edited(p))

            pre_photos[pid] = {
                "film": p_film,
                "paper": p_paper,
                "print_mode": p_pmode,
                "params": p_params,
                "crop": p_crop,
                "is_dirty": p_dirty,
                "is_edited": p_edited,
                "path": p.get("path"),
            }

        pre_state = {
            "type": "batch_photos_state",
            "action": "paste_params",
            "active_photo_id": self.active_photo_id,
            "photos": pre_photos,
        }
        self._push_raw_state(pre_state)

        affected_count = 0
        active_adapted_params = None
        for pid in selected_ids:
            photo = next((p for p in self.photos if p and p.get("id") == pid), None)
            if not photo:
                continue

            exif = photo.get("meta") or photo.get("exif") or {}
            p_base_temp = float(exif.get("color_temp", 5500.0))
            p_base_tint = float(exif.get("tint", 1.0))
            p_base_ev = float(photo.get("base_ev", 0.0))
            if abs(p_base_ev) < 0.0001 and isinstance(photo.get("params"), dict) and photo["params"].get("base_ev"):
                p_base_ev = float(photo["params"]["base_ev"])

            target_params = self.get_default_params(p_base_temp, p_base_tint)
            target_params.update(copy.deepcopy(copied_params))
            if "halation_amount" not in copied_params and "halation" in copied_params:
                target_params["halation_amount"] = copied_params["halation"]
            if "boost_ev" not in copied_params and "halation_boost" in copied_params:
                target_params["boost_ev"] = copied_params["halation_boost"]

            # Preserve target photo's sensor baseline exposure and camera EXIF white balance baseline
            target_params["base_temp"] = p_base_temp
            target_params["base_tint"] = p_base_tint
            target_params["base_ev"] = p_base_ev
            target_params["print_mode"] = copied_print_mode

            # Adapt white balance relative to target photo baseline EXIF if not custom
            if not target_params.get("custom_wb", False):
                target_params["color_temp"] = p_base_temp
                target_params["tint"] = p_base_tint

            # Synchronize crop state
            crop_data = copy.deepcopy(target_params.get("crop")) if isinstance(target_params.get("crop"), dict) else dict(default_crop)
            target_params["crop"] = dict(crop_data)
            photo["crop"] = dict(crop_data)

            # Recompute effective lens params for target photo focal length / aperture
            lens_cfg = target_params.setdefault("lens", self._get_default_lens_params())
            if isinstance(lens_cfg, dict):
                from core.lens_correction import compute_effective_lens_params
                eff_lens = compute_effective_lens_params(
                    lens_cfg,
                    focal_length=exif.get("focal_length"),
                    aperture=exif.get("aperture"),
aspect_ratio=self._lens_frame_aspect(photo),
                    source_path=(photo or {}).get("path")
                )
                lens_cfg.update(eff_lens)

            photo["film_profile"] = copied_film
            photo["paper_profile"] = copied_paper
            photo["print_mode"] = copied_print_mode
            photo["params"] = target_params
            photo["is_dirty"] = True
            photo["is_edited"] = True

            if pid == self.active_photo_id:
                active_adapted_params = copy.deepcopy(target_params)

            # Apply and refresh thumbnail immediately via ThumbnailController
            raw_thumb = self.thumb_controller.get_raw_thumbnail(photo)
            if raw_thumb is not None:
                graded_thumb = self.thumb_controller.grade_thumbnail_fast(
                    raw_thumb,
                    copied_film,
                    copied_paper,
                    target_params
                )
                self.thumb_controller.update_graded_thumbnail(photo, graded_thumb, self.filmstrip)

            # Save to SDC sidecar immediately
            try:
                sdc_manager.save_sdc(
                    photo["path"],
                    copied_film,
                    copied_paper,
                    target_params,
                    thumbnail_rgb=photo.get("thumbnail_rgb"),
                    print_mode=copied_print_mode
                )
            except Exception:
                pass

            if hasattr(self, "filmstrip"):
                self.filmstrip.set_photo_crop(pid, dict(crop_data))
                self.filmstrip.set_photo_dirty(pid, True)
            affected_count += 1

        # If only a single photo was targeted (e.g. via right-click context menu on a new filmstrip item)
        # and it is not yet the active photo, switch to it so the canvas and right-hand sliders reflect it immediately
        if len(selected_ids) == 1 and selected_ids[0] != self.active_photo_id:
            self.on_switch_photo(selected_ids[0])
        elif self.active_photo_id in selected_ids:
            self.current_film_stock = copied_film
            self.current_paper_stock = copied_paper
            self.current_print_mode = copied_print_mode
            if active_adapted_params is not None:
                self.current_params.clear()
                self.current_params.update(active_adapted_params)
            else:
                self.current_params = copy.deepcopy(copied_params)
            self.current_params["print_mode"] = copied_print_mode
            self._flatten_dir_params(self.current_params)

            # Enable sidebar controls BEFORE updating slider/combo values so Qt schedules repaints properly
            self._update_stock_mode_ui()

            # Sync crop workspace and overlay
            act_crop = dict(self.current_params.get("crop") or default_crop)
            if getattr(self.canvas, "is_crop_editing", False):
                self._exit_crop_workspace()
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                if not self._is_crop_modified(act_crop):
                    self.canvas.crop_overlay.reset_all_crop()
                elif self.canvas.crop_overlay.isVisible():
                    self.canvas.crop_overlay.start_crop_session(dict(act_crop))
            if hasattr(self, "crop_sidebar") and self.crop_sidebar:
                self.crop_sidebar.sync_from_overlay(dict(act_crop))

            # Auto-detect lens profile for active photo if lens setup is auto
            lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
            if isinstance(lens_cfg, dict) and lens_cfg.get("setup", "auto") == "auto":
                self._auto_detect_lens_profile()

            self.on_switch_print_mode(self.current_print_mode, push_undo=False)
            self.on_switch_film(self.current_film_stock, push_undo=False)
            if self.current_print_mode == "optical":
                self.on_switch_paper(self.current_paper_stock, push_undo=False)
            else:
                for pid, card in self.paper_cards.items():
                    card.set_active(pid == "none")

            # Refresh all sliders/combos/checkboxes AFTER stock & film switches so nothing overwrites them
            self._update_all_sliders_from_params()
            if hasattr(self, "sidebar_content") and self.sidebar_content:
                self.sidebar_content.update()

            self._sync_full_params_to_pipeline(fit_crop_if_changed=True)
            self._mark_current_photo_dirty()

        # 2. Snapshot post-paste state for all affected photos
        post_photos = {}
        for pid in selected_ids:
            photo = next((p for p in self.photos if p and p.get("id") == pid), None)
            if not photo:
                continue
            p_params_post = copy.deepcopy(photo.get("params") or {})
            post_photos[pid] = {
                "film": photo.get("film_profile", copied_film),
                "paper": photo.get("paper_profile", copied_paper),
                "print_mode": photo.get("print_mode", copied_print_mode),
                "params": p_params_post,
                "crop": copy.deepcopy(photo.get("crop") or p_params_post.get("crop") or default_crop),
                "is_dirty": True,
                "is_edited": True,
                "path": photo.get("path"),
            }

        post_state = {
            "type": "batch_photos_state",
            "action": "paste_params",
            "active_photo_id": self.active_photo_id,
            "photos": post_photos,
        }
        self._push_raw_state(post_state)

        self.set_app_status(f"已应用暗房配方：{affected_count} 张底片", timeout_ms=3000)

    def _apply_history_state(self, state):
        if not state:
            return

        if state.get("type") == "batch_photos_state":
            self._apply_batch_history_state(state)
            return

        if state.get("type") == "library_photos_state":
            self._apply_library_history_state(state)
            return

        if state.get("type") == "photo_rating_state":
            self._apply_rating_history_state(state)
            if not state.get("params"):
                return

        target_pid = state.get("photo_id")
        if target_pid and target_pid != self.active_photo_id:
            try:
                self.on_switch_photo(target_pid)
            except Exception as e:
                logger.warning(f"Failed switching to {target_pid} during undo: {e}")

        self.current_film_stock = state.get("film", "none")
        self.current_paper_stock = state.get("paper", "none")
        self.current_print_mode = state.get("print_mode", "optical")
        self.current_params = dict(state.get("params") or {})
        self.current_params["print_mode"] = self.current_print_mode
        self._flatten_dir_params(self.current_params)

        default_crop = {
            "enabled": False, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
            "angle": 0.0, "rot_90": 0, "flip_h": False, "flip_v": False,
            "aspect_ratio": "original", "is_portrait": False
        }
        crop_data = self.current_params.get("crop")
        if not isinstance(crop_data, dict):
            crop_data = dict(default_crop)
            self.current_params["crop"] = dict(default_crop)

        self._update_all_sliders_from_params()
        self.on_switch_print_mode(self.current_print_mode, push_undo=False)
        self.on_switch_film(self.current_film_stock, push_undo=False)
        if self.current_print_mode == "optical":
            self.on_switch_paper(self.current_paper_stock, push_undo=False)
        else:
            for pid, card in self.paper_cards.items():
                card.set_active(pid == "none")

        if getattr(self.canvas, "is_crop_editing", False):
            self._exit_crop_workspace()

        if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
            if not self._is_crop_modified(crop_data):
                self.canvas.crop_overlay.reset_all_crop()
            elif self.canvas.crop_overlay.isVisible():
                self.canvas.crop_overlay.start_crop_session(dict(crop_data))

        if hasattr(self, "crop_sidebar") and self.crop_sidebar:
            self.crop_sidebar.sync_from_overlay(dict(crop_data))

        self._sync_full_params_to_pipeline(fit_crop_if_changed=True)

        # Update current photo metadata
        if self._current_photo_path:
            curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if curr_p:
                curr_p["film_profile"] = self.current_film_stock
                curr_p["paper_profile"] = self.current_paper_stock
                curr_p["print_mode"] = self.current_print_mode
                curr_p["params"] = dict(self.current_params)
                curr_p["crop"] = dict(crop_data)
                is_reverted_to_raw = (self.current_film_stock == "none" and self.current_paper_stock == "none" and not self._is_params_modified(self.current_params, curr_p))
                curr_p["is_dirty"] = not is_reverted_to_raw
                curr_p["is_edited"] = not is_reverted_to_raw
                self._photo_is_dirty = not is_reverted_to_raw
                if hasattr(self, 'filmstrip'):
                    self.filmstrip.set_photo_dirty(self.active_photo_id, not is_reverted_to_raw)
                    self.filmstrip.set_photo_crop(self.active_photo_id, dict(crop_data))
            self._save_current_photo_sdc()
            self._update_active_thumbnails()
            if hasattr(self, '_cards_debounce_timer'):
                self._cards_debounce_timer.start(100)
            else:
                self._refresh_card_previews()

    def _apply_batch_history_state(self, state: dict):
        """Restores a batch operation snapshot across multiple photos (e.g. Select All -> Delete SDC, Clear Crop, or Paste)."""
        if not state or not isinstance(state, dict):
            return

        photos_data = state.get("photos", {})
        if not photos_data:
            return

        default_crop = {
            "enabled": False, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
            "angle": 0.0, "rot_90": 0, "flip_h": False, "flip_v": False,
            "aspect_ratio": "original", "is_portrait": False
        }

        active_restored = False
        for pid, pdata in photos_data.items():
            p = next((x for x in self.photos if x.get("id") == pid), None)
            if not p:
                continue

            film = pdata.get("film", "none")
            paper = pdata.get("paper", "none")
            print_mode = pdata.get("print_mode", "optical")
            params = dict(pdata.get("params") or {})

            crop_data = pdata.get("crop")
            if crop_data is None and isinstance(params.get("crop"), dict):
                crop_data = dict(params["crop"])
            elif crop_data is None:
                crop_data = dict(default_crop)
            else:
                crop_data = dict(crop_data)

            is_dirty = bool(pdata.get("is_dirty", False))
            is_edited = bool(pdata.get("is_edited", False))
            p_path = p.get("path")

            p["film_profile"] = film
            p["paper_profile"] = paper
            p["print_mode"] = print_mode
            p["crop"] = dict(crop_data)
            if params:
                params["crop"] = dict(crop_data)

            p["params"] = dict(params) if (is_edited or film not in ("none", "", None) or paper not in ("none", "", None) or self._is_params_modified(params, p)) else None
            p["is_dirty"] = is_dirty
            p["is_edited"] = is_edited

            # 1. Disk SDC persistence
            if p_path:
                if is_edited or film not in ("none", "", None) or paper not in ("none", "", None) or self._is_params_modified(params, p):
                    try:
                        cur_thumb = p.get("graded_thumbnail_rgb")
                        if cur_thumb is None:
                            cur_thumb = p.get("thumbnail_rgb")
                        sdc_manager.save_sdc(
                            p_path,
                            film,
                            paper,
                            p["params"] or params,
                            thumbnail_rgb=cur_thumb,
                            print_mode=print_mode
                        )
                    except Exception:
                        pass
                else:
                    try:
                        sdc_manager.delete_sdc(p_path)
                    except Exception:
                        pass

            # 2. Thumbnail rendering
            if is_edited or film not in ("none", "", None) or paper not in ("none", "", None) or self._is_params_modified(params, p):
                if hasattr(self, 'thumb_controller'):
                    raw_thumb = self.thumb_controller.get_raw_thumbnail(p)
                    if raw_thumb is not None:
                        graded = self.thumb_controller.grade_thumbnail_fast(raw_thumb, film, paper, p["params"] or params)
                        self.thumb_controller.update_graded_thumbnail(p, graded, self.filmstrip)
            else:
                if hasattr(self, 'thumb_controller'):
                    self.thumb_controller.revert_photo_to_raw(p, self.filmstrip)

            if hasattr(self, 'filmstrip'):
                self.filmstrip.set_photo_crop(pid, dict(crop_data))
                self.filmstrip.set_photo_dirty(pid, is_dirty)

            if pid == self.active_photo_id:
                active_restored = True

        self._update_window_title()

        # 3. Synchronize viewport if active photo was in the batch
        if active_restored and self.active_photo_id in photos_data:
            act_data = photos_data[self.active_photo_id]
            self.current_film_stock = act_data.get("film", "none")
            self.current_paper_stock = act_data.get("paper", "none")
            self.current_print_mode = act_data.get("print_mode", "optical")
            act_params = dict(act_data.get("params") or {})
            if not act_params:
                base_temp = float(getattr(self, "_active_base_temp", 5500.0))
                base_tint = float(getattr(self, "_active_base_tint", 1.0))
                act_params = self.get_default_params(base_temp, base_tint)

            act_crop = act_data.get("crop")
            if act_crop is None and isinstance(act_params.get("crop"), dict):
                act_crop = dict(act_params["crop"])
            elif act_crop is None:
                act_crop = dict(default_crop)
            else:
                act_crop = dict(act_crop)

            act_params["crop"] = dict(act_crop)
            self.current_params = act_params
            self.current_params["print_mode"] = self.current_print_mode
            self._flatten_dir_params(self.current_params)
            self._photo_is_dirty = bool(act_data.get("is_dirty", False))

            if getattr(self.canvas, "is_crop_editing", False):
                self._exit_crop_workspace()

            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                if not self._is_crop_modified(act_crop):
                    self.canvas.crop_overlay.reset_all_crop()
                elif self.canvas.crop_overlay.isVisible():
                    self.canvas.crop_overlay.start_crop_session(dict(act_crop))

            if hasattr(self, "crop_sidebar") and self.crop_sidebar:
                self.crop_sidebar.sync_from_overlay(dict(act_crop))

            self._update_stock_mode_ui()
            self.on_switch_print_mode(self.current_print_mode, push_undo=False)
            self.on_switch_film(self.current_film_stock, push_undo=False)
            if self.current_print_mode == "optical":
                self.on_switch_paper(self.current_paper_stock, push_undo=False)
            else:
                for pid_card, card in self.paper_cards.items():
                    card.set_active(pid_card == "none")
            self._update_all_sliders_from_params()
            if hasattr(self, "sidebar_content") and self.sidebar_content:
                self.sidebar_content.update()

            self._sync_full_params_to_pipeline(fit_crop_if_changed=True)
            self._update_reset_buttons_state()
            self._update_active_thumbnails()
            if hasattr(self, '_cards_debounce_timer'):
                self._cards_debounce_timer.start(100)
            else:
                self._refresh_card_previews()
        elif not active_restored and state.get("active_photo_id"):
            target_pid = state.get("active_photo_id")
            if target_pid in photos_data and target_pid != self.active_photo_id:
                try:
                    self.on_switch_photo(target_pid)
                except Exception:
                    pass

    def _snapshot_photo_entry(self, p: dict) -> dict:
        snap = dict(p)
        if "params" in p and isinstance(p["params"], dict):
            snap["params"] = copy.deepcopy(p["params"])
        if "crop" in p and isinstance(p["crop"], dict):
            snap["crop"] = copy.deepcopy(p["crop"])
        # 旁挂的跨链相纸记忆也必须是副本：撤销回退后继续切卷会原地改这份 dict，
        # 浅拷贝会让快照跟着一起变（重做就回不到原来的选择）。
        if "paper_by_scope" in p and isinstance(p["paper_by_scope"], dict):
            snap["paper_by_scope"] = dict(p["paper_by_scope"])
        return snap

    def _apply_library_history_state(self, state: dict):
        """Restores a filmstrip library snapshot (e.g. undo/redo photo removal or library clearing)."""
        if not state or not isinstance(state, dict):
            return

        photos_data = state.get("photos", [])
        active_id = state.get("active_photo_id")

        if not photos_data:
            # Reached empty library state (e.g. redo clear or redo remove all)
            self.action_clear_photos(push_undo=False)
            return

        # 1. Restore self.photos list
        self.photos = [self._snapshot_photo_entry(p) for p in photos_data]
        self._photo_id_counter = max(getattr(self, "_photo_id_counter", 0), len(self.photos))

        # 2. Synchronize engine session photos
        if hasattr(self, "engine") and hasattr(self.engine, "session_photos"):
            self.engine.session_photos.clear()
            for photo_entry in self.photos:
                ep_dict = {
                    "id": photo_entry["id"],
                    "path": photo_entry.get("path"),
                    "filename": photo_entry.get("filename"),
                    "width": photo_entry.get("width", 0),
                    "height": photo_entry.get("height", 0),
                    "thumbnail_rgb": photo_entry.get("thumbnail_rgb"),
                    "raw_preview": None,
                    "meta": dict(photo_entry.get("exif") or {}),
                    "is_raw": photo_entry.get("is_raw", False),
                    "base_ev": photo_entry.get("base_ev", 0.0),
                    "auto_ev": photo_entry.get("auto_ev", 0.0)
                }
                self.engine.session_photos.append(ep_dict)
            # The path -> index map is order-sensitive; force a rebuild on next use.
            self._engine_photos_path_sig = None

        # 3. Synchronize filmstrip widget
        if hasattr(self, "filmstrip") and self.filmstrip:
            self.filmstrip.set_photos(self.photos, active_id=active_id)
            for p in self.photos:
                pid = p.get("id")
                if pid:
                    if p.get("is_dirty"):
                        self.filmstrip.set_photo_dirty(pid, True)
                    if p.get("crop"):
                        self.filmstrip.set_photo_crop(pid, p["crop"])

        # 4. Restore session flags & title
        self._session_is_dirty = state.get("session_is_dirty", True)
        if state.get("session_path") is not None:
            self._current_session_path = state.get("session_path")
        self._update_window_title()

        # 5. Restore canvas visibility & activate target photo
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.set_has_library_photos(True)
        if hasattr(self, "btn_title_export"):
            self.btn_title_export.setVisible(True)

        target_id = active_id if (active_id and any(p.get("id") == active_id for p in self.photos)) else (self.photos[0]["id"] if self.photos else None)
        if target_id:
            self.on_switch_photo(target_id, force=True)

        self._update_menu_actions_state()

    def _apply_rating_history_state(self, state: dict):
        """Restores photo rating state across one or multiple photos."""
        ratings = state.get("ratings", {})
        if not ratings:
            return
        for pid, r in ratings.items():
            p = next((x for x in self.photos if x and x.get("id") == pid), None)
            if p:
                p["rating"] = r
            if hasattr(self, 'filmstrip'):
                self.filmstrip.set_photo_rating(pid, r)
            if pid == self.active_photo_id and hasattr(self, 'canvas_rating'):
                self.canvas_rating.set_rating(r)
        self._session_is_dirty = True
        self._update_window_title()
        self._save_auto_temp_session()
        self._update_menu_actions_state()

    def _init_histogram_widget(self):
        pass

    def _schedule_histogram_update(self):
        if hasattr(self, '_hist_timer') and not self._hist_timer.isActive():
            self._hist_timer.start(40)

    def _compute_and_update_histogram(self):
        if not hasattr(self, 'histogram_widget'):
            return
        if self._current_photo_path is None and self._current_photo_thumb is None:
            self.histogram_widget.clear()
            return
        try:
            # 1. 100% WYSIWYG Real-time GPU offscreen FBO sampling across all 20+ parameters
            if hasattr(self, 'canvas') and self.canvas and getattr(self.canvas, '_has_image', False):
                # noise to the readout without telling anyone anything.
                sampled = self.canvas.render_offscreen(
                    target_w=256, target_h=170, bit_depth=8)
                if sampled is not None and sampled.size > 0:
                    self.histogram_widget.update_from_rgb(sampled)
                    return

            # 2. Resilient fallback: CPU log-shaper LUT on linear thumbnail
            if self._current_photo_linear is not None:
                lut_3d = self.engine.get_3d_lut(
                    self.current_film_stock,
                    self.current_paper_stock,
                    lut_size=17,
                    params_dict=self.current_params
                )
                # 这条兜底以前直接拿 `_current_photo_linear` 算 —— 那是**主图**（24 MP
                # float32），`apply_lut_to_linear` 的逐像素三线性插值得跑 14.7 s（实测，
                # 会话恢复期间画布还没上纹理时就会走到这里）。直方图只要 256 级统计，
                # 抽稀到 ~0.5 MP 足够，代价降到几十毫秒。
                src_lin = self._current_photo_linear
                _h, _w = src_lin.shape[:2]
                if _h * _w > 64_000:
                    _step = int((_h * _w / 64_000) ** 0.5) + 1
                    src_lin = np.ascontiguousarray(src_lin[::_step, ::_step])
                eff_exp = float(self.current_params.get("exposure_ev", 0.0)) + float(self.current_params.get("base_ev", 0.0))
                if abs(eff_exp) > 0.001:
                    src_lin = src_lin * (2.0 ** eff_exp)
                graded = self.engine.apply_lut_to_linear(src_lin, lut_3d)
                self.histogram_widget.update_from_rgb(graded)
            elif self._current_photo_thumb is not None:
                self.histogram_widget.update_from_rgb(self._current_photo_thumb)
        except Exception:
            pass

    def get_default_params(self, base_temp=5500.0, base_tint=1.0):
        return {
            "print_mode": "optical",
            "exposure_ev": 0.0,
            "base_ev": 0.0,
            "highlights": 0.0,
            "shadows": 0.0,
            "color_temp": float(base_temp),
            "tint": float(base_tint),
            "base_temp": float(base_temp),
            "base_tint": float(base_tint),
            "custom_wb": False,
            "film_format_mm": 35.0,
            "diffusion_family": "none",
            "diffusion_strength": 0.0,
            "diffusion_warmth": 0.0,
            "dir_amount": 1.0,
            "dir_interlayer": 1.0,
            "dir_samelayer": 1.0,
            "enlarger_illuminant": "TH-KG3",
            "enlarger_cyan": 0.0,
            "enlarger_magenta": 0.0,
            "enlarger_yellow": 0.0,
            "print_exposure": 1.0,
            "pre_flash": 0.0,
            "pre_flash_camera": 0.0,
            "black_lift": 0.0,
            "morph_gamma": 1.0,
            "developer_exhaustion": 0.0,
            "halation": 1.0,
            "halation_amount": 1.0,
            "halation_spatial_scale": 1.0,
            "scatter_amount": 1.0,
            "boost_ev": 0.0,
            "halation_boost": 0.0,
            "grain": 0.4,
            "grain_size": 1.0,
            "grain_colour": 1.0,
            "scan_noise": 0.0,
            "lens": self._get_default_lens_params(),
            "split_x": 0.5,
            "view_mode": 0
        }

    # Parameters that only exist because of film / paper chemistry. In bypass mode
    # (film_stock == "none") those panels stay disabled, so any stray signal must be dropped
    # instead of being written into current_params and the .sdc sidecar / export payload.
    # 场景曝光 (exposure_ev / highlights / shadows / color_temp / tint) and 光学 (lens_* /
    # diffusion_*) are deliberately NOT listed: they remain fully live in bypass.
    BYPASS_INERT_PARAMS = frozenset({
        # 胶片化学 (DIR)
        "dir_amount", "dir_interlayer", "dir_samelayer", "push_ev",
        # 放大机与相纸显影
        "enlarger_illuminant", "enlarger_cyan", "enlarger_magenta", "enlarger_yellow",
        "print_exposure", "pre_flash", "pre_flash_camera",
        "morph_gamma", "developer_exhaustion",
        # 物理光晕
        "halation", "halation_amount", "halation_boost", "boost_ev",
        "halation_spatial_scale", "scatter_amount",
        # 银盐颗粒
        "grain", "grain_size", "grain_colour", "scan_noise",
        # 画幅规格只用于推导银盐晶体 / 弥散圈的物理相对尺度，无胶片时无意义
        "film_format_mm",
    })

    def _bypass_blocks_param(self, param_key):
        """True when the parameter is inert in 数码原片旁通 mode and must be ignored."""
        return (getattr(self, "current_film_stock", "none") == "none"
                and param_key in self.BYPASS_INERT_PARAMS)

    def _slider_is_dragging(self) -> bool:
        """Whether this value change comes from an ongoing drag rather than a single click.

        `canvas.begin_interaction()` is not free: it makes the canvas paint a cheap frame right
        away and a second, full-quality frame once the gesture has been idle for 160 ms.  A
        click commits once and is over immediately, so arming it pays two full-frame composites
        (~75 ms each here) for one click -- and that second frame is still rendering when the
        next click arrives, which is exactly what reads as "the second click sticks".
        """
        src = self.sender()
        if src is not None and hasattr(src, "is_scrubbing"):
            return bool(src.is_scrubbing())
        try:
            from PySide6.QtWidgets import QApplication
            return bool(QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        except Exception:                       # noqa: BLE001 - never break a slider over this
            return True

    def _on_slider_live(self, param_key, val):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        if self._bypass_blocks_param(param_key):
            return
        if getattr(self, "_interaction_watch", None):
            self._interaction_watch[0] = time.perf_counter()
        if self._slider_is_dragging():
            began = getattr(self.canvas, "begin_interaction", None)
            if began is not None:
                began()
        self.current_params[param_key] = val
        if param_key == "halation_amount":
            self.current_params["halation"] = val
        elif param_key == "boost_ev":
            self.current_params["halation_boost"] = val
        elif param_key == "halation":
            self.current_params["halation_amount"] = val
        elif param_key == "halation_boost":
            self.current_params["boost_ev"] = val
        if param_key in ("color_temp", "tint"):
            self.current_params["custom_wb"] = True
        if param_key == "exposure_ev" and int(round(float(self.current_params.get("push_ev", 0) or 0))):
            # The 迫冲 readout is arithmetic on the exposure, so it has to follow it.
            self._refresh_push_readout()
        # DIR / morph / exhaustion are baked into the 3D LUT: preview them by rebaking the same
        # full-resolution LUT in the background, everything else is applied post-LUT by the shader.
        # DIR 是例外——点式链接手后它是逐像素算的，滑块只改一个 uniform。
        if param_key in ("dir_amount", "dir_interlayer", "dir_samelayer",
                         "morph_gamma", "developer_exhaustion"):
            ff = getattr(self.canvas, "_ffchain", None)
            if param_key == "dir_amount" and getattr(ff, "ready", False):
                self.canvas.update_params(dir_amount=val)
            else:
                self._pending_lut_refresh = True
                self._schedule_live_lut_preview()
        else:
            self.canvas.update_params(**{param_key: val})
        self._schedule_histogram_update()
        self._mark_current_photo_dirty()
        self._update_reset_buttons_state()

    LUT_CHEM_KEYS = ("dir_amount", "dir_interlayer", "dir_samelayer",
                     "morph_gamma", "developer_exhaustion", "push_ev")

    @staticmethod
    def _flatten_dir_params(params):
        """DIR 的三个滑块已合并成一个「总量」，跨层/同层固定 1.0。

        fotufilm 上游只有 COUPLER_SCALE 一个量，没有跨层/同层的划分；spektrafilm 那边
        dir_interlayer / dir_samelayer 也只是 dir_amount 的乘数。键继续保留（旧 .sdc 不报错、
        照常读），但不再让它们生效 —— 否则"同一个总量"在不同工程里会给出不同的图。
        """
        if isinstance(params, dict):
            params["dir_interlayer"] = 1.0
            params["dir_samelayer"] = 1.0
        return params

    def _apply_lut(self, lut_3d):
        """Upload a committed LUT and record which parameter key it represents, so the live
        preview can tell "already showing this" from "still stale"."""
        self.canvas.set_lut(lut_3d)
        self.canvas.set_grain_bundle(self._grain_bundle_for_current())
        self.canvas.set_ffchain_bundle(self._ffchain_bundle_for_current())
        self._lut_applied_key = self._current_lut_key()
        queued = getattr(self, "_lut_job", None)
        if queued is not None and queued[0] != self._lut_applied_key:
            self._lut_job = None       # superseded: never spend a bake on outdated parameters

    def _on_ffchain_now(self):
        """当前参数是不是走 fotufilm 点式链（决定哪些 spektrafilm 参数已失效）。"""
        try:
            return self._ffchain_bundle_for_current() is not None
        except Exception:                           # noqa: BLE001 - 只用于 UI 状态
            return False

    def _ffchain_bundle_for_current(self):
        """fotufilm 点式链的表；不适用返回 None，画布回落到单张 65³。"""
        eng = getattr(self, "engine", None)
        film = getattr(self, "current_film_stock", "none")
        if eng is None or film in (None, "", "none"):
            return None
        params = dict(self.current_params)
        try:
            return eng.get_ffchain_point_bundle(
                film, getattr(self, "current_paper_stock", "kodak_2383"),
                getattr(self, "current_print_mode", "optical"), params)
        except Exception as exc:                    # noqa: BLE001 - never break a paint
            print(f"[FFChain] 点式链获取失败: {exc}")
            return None

    def _grain_bundle_for_current(self):
        """The two density tables for the LUT on screen, or ``None`` when that bake has none.

        A table read from disk has no Simulator behind it, so the print cube cannot be produced
        and the shader falls back to the single RGB table.  Saying so is the honest answer --
        the alternative is grain indexed by a density from a different print.
        """
        eng = getattr(self, "engine", None)
        film = getattr(self, "current_film_stock", "none")
        if eng is None or film in (None, "", "none"):
            return None
        # 底座用画布自己的 params，不用 current_params：渲染是 `dict(self.params)` 起头的
        # （会话恢复还会把上张图的编辑参数写进去）。两边的化学参数一旦分叉，算出的 LUT key
        # 就不同，`GLCubeGrain.bind` 的 expect_key 会判"不是同一套烘"并把颗粒静默关掉 ——
        # 表现就是预览有颗粒、导出没有，且换卷才复现。
        canvas_params = getattr(self.canvas, "params", None)
        params = (dict(canvas_params) if isinstance(canvas_params, dict) and canvas_params
                  else dict(self.current_params))
        # 只在画布没带 print_mode 时兜底。原来是无条件覆盖成 current_print_mode，
        # 而反转片的 UI 状态（optical）与它实际的介质（screen/scan）会分叉 ——
        # 覆盖后算出的 key 与渲染那帧不同，expect_key 判"不是同一套烘"，颗粒静默消失。
        if "print_mode" not in params:
            params["print_mode"] = getattr(self, "current_print_mode", "optical")
        try:
            return eng.get_grain_bundle(film, getattr(self, "current_paper_stock", "kodak_2383"),
                                        17, params)
        except Exception as exc:                    # noqa: BLE001 - never break a paint
            print(f"[SpektraGrain] 双立方体获取失败: {exc}")
            return None

    def _ensure_lut_bake_bridge(self):
        if hasattr(self, "_lut_bake_bridge"):
            return
        self._lut_bake_bridge = _LutBakeBridge()
        self._lut_bake_bridge.baked.connect(self._on_async_lut_ready)
        self._lut_worker_busy = False
        self._lut_job = None            # (key, snapshot, film, paper, mode) awaiting a worker
        self._lut_inflight_key = None
        self._lut_applied_key = getattr(self, "_lut_applied_key", None)
        self._live_lut_timer = QTimer(self)
        self._live_lut_timer.setSingleShot(True)
        self._live_lut_timer.timeout.connect(self._request_async_lut)

    def _current_lut_key(self):
        """Everything that changes the baked LUT, rounded to the engine's own cache precision."""
        p = self.current_params
        return (
            getattr(self, "current_film_stock", "none"),
            getattr(self, "current_paper_stock", "none"),
            getattr(self, "current_print_mode", "optical"),
            round(float(p.get("dir_amount", 1.0)), 2),
            round(float(p.get("dir_interlayer", 1.0)), 2),
            round(float(p.get("dir_samelayer", 1.0)), 2),
            round(float(p.get("morph_gamma", 1.0)), 2),
            round(float(p.get("developer_exhaustion", 0.0)), 2),
            int(round(float(p.get("push_ev", 0) or 0))),
        )

    def _schedule_live_lut_preview(self):
        """Live preview of LUT-baked parameters (DIR / morph / exhaustion).

        Uses the exact same 33^3 build as the committed bake, on a worker thread, so the
        image shown while dragging is what the user keeps on release. The previous 17^3
        preview was a different approximation and produced a visible jump on release.
        """
        if getattr(self, "current_film_stock", "none") == "none":
            return
        self._ensure_lut_bake_bridge()
        if not self._live_lut_timer.isActive():
            self._live_lut_timer.start(60)

    def _request_async_lut(self):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        if getattr(self, "current_film_stock", "none") == "none":
            return
        # 点式链能接手时，65³ 根本不会被采样（u_ff_on 的分支优先于 u_has_lut），
        # 所以为它烘一次表纯属白跑 25 s。这里只换 bundle 并记账，不启动烘焙。
        ff_bundle = self._ffchain_bundle_for_current()
        if ff_bundle is not None:
            self.canvas.set_ffchain_bundle(ff_bundle)
            self._lut_applied_key = self._current_lut_key()
            self._lut_job = None
            return
        self._ensure_lut_bake_bridge()
        key = self._current_lut_key()
        if key == self._lut_applied_key:
            return
        self._lut_job = (
            key,
            dict(self.current_params),
            getattr(self, "current_film_stock", "none"),
            getattr(self, "current_paper_stock", "none"),
            getattr(self, "current_print_mode", "optical"),
        )
        self._pump_lut_worker()

    def _pump_lut_worker(self):
        """At most one background bake at a time; a newer request replaces the queued job."""
        if getattr(self, "_lut_worker_busy", False) or getattr(self, "_lut_job", None) is None:
            return
        key, snapshot, film, paper, mode = self._lut_job
        self._lut_job = None
        self._lut_worker_busy = True
        self._lut_inflight_key = key
        engine = self.engine
        bridge = self._lut_bake_bridge
        chem = {k: snapshot.get(k, 1.0) for k in self.LUT_CHEM_KEYS}

        def _run():
            try:
                lut = engine.get_3d_lut(film, paper, lut_size=33, params_dict=snapshot, print_mode=mode)
            except Exception:
                lut = None
            try:
                bridge.baked.emit(lut, key, chem)
            except RuntimeError:
                pass  # window already destroyed

        threading.Thread(target=_run, name="SpektraLutBake", daemon=True).start()

    def _on_async_lut_ready(self, lut, key, chem):
        self._lut_worker_busy = False
        self._lut_inflight_key = None
        # Drop results whose parameters are no longer current (slider moved on, photo
        # switched, undo/redo, preset applied): the same key always yields the same LUT.
        if lut is not None and key == self._current_lut_key():
            try:
                if isinstance(getattr(self.canvas, "params", None), dict):
                    for k, v in chem.items():
                        self.canvas.params[k] = v
                self.canvas.set_lut(lut)
                self.canvas.set_grain_bundle(self._grain_bundle_for_current())
            except RuntimeError:
                return  # canvas / window already torn down
            self._lut_applied_key = key
        self._pump_lut_worker()

    def _on_combo_changed(self, param_key, val):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        if self._bypass_blocks_param(param_key):
            return
        began = getattr(self.canvas, "begin_interaction", None)
        if began is not None:
            began()
        # Read before the write: the push branch below needs to know which stop count we came from.
        prev_push = int(round(float(self.current_params.get("push_ev", 0) or 0)))
        self.current_params[param_key] = val
        self.canvas.update_params(**{param_key: val})
        if param_key == "push_ev":
            # The development condition itself never touches the scene exposure.  Only the linked
            # workflow does, and only when 模拟欠曝 is on: raising the push by N stops means the
            # film was *shot* N stops under, so the exposure takes that offset now.
            # Deliberately not a standing constraint: the exposure slider stays free afterwards,
            # and `push_ref_exposure` remembers where it landed so the readout can say how much of
            # the underexposure the user has since given back.
            delta = int(round(float(val))) - prev_push
            if delta and self.current_params.get("push_underexpose"):
                self._apply_push_exposure_offset(-delta)
            # A different development condition is a different film: the LUT has to be rebaked,
            # not just re-shaded. Same path the DIR sliders take.
            self._pending_lut_refresh = True
            self._schedule_live_lut_preview()
            self._schedule_histogram_update()
            self._mark_current_photo_dirty()
            self._update_reset_buttons_state()
            self._refresh_push_readout()
            return
        if param_key == "diffusion_family":
            # The strength combo no longer carries a 0 marker ("no filter" is expressed by
            # the family itself), so picking a family always lands on a real density.
            if val != "none":
                cur = float(self.current_params.get("diffusion_strength", 0.0) or 0.0)
                if cur <= 0.0:
                    cur = 0.125
                    self.current_params["diffusion_strength"] = cur
                    self.canvas.update_params(diffusion_strength=cur)
                if hasattr(self, 'combo_diff_strength'):
                    self.combo_diff_strength.set_value(cur)
            else:
                self.current_params["diffusion_strength"] = 0.0
                self.canvas.update_params(diffusion_strength=0.0)
        elif param_key == "diffusion_strength":
            # A strength can only be chosen once a family is active (the row is disabled
            # otherwise); fall back to the family the combo is already showing.
            if val > 0.0 and self.current_params.get("diffusion_family", "none") == "none":
                self.current_params["diffusion_family"] = "black_pro_mist"
                self.canvas.update_params(diffusion_family="black_pro_mist")
                if hasattr(self, 'combo_diffusion'):
                    self.combo_diffusion.set_value("black_pro_mist")
        if param_key in ("diffusion_family", "diffusion_strength"):
            self._update_diffusion_ui_state()
        elif param_key == "enlarger_illuminant":
            self._refresh_lut()
        self._schedule_histogram_update()
        self.push_undo_state()
        self._mark_current_photo_dirty()
        self._update_reset_buttons_state()
        self._save_current_photo_sdc()
        self._update_active_thumbnails()

    def _save_current_photo_sdc(self):
        """Immediately persist the active photo's film, paper, print mode, and parameters to .sdc sidecar.
        If the photo has reverted to completely unedited pristine RAW, safely removes obsolete .sdc file.
        """
        if not self._current_photo_path or not self.active_photo_id:
            return
        curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
        if not curr_p:
            return

        is_film_custom = (self.current_film_stock not in ("none", "", None))
        is_paper_custom = (self.current_paper_stock not in ("none", "", None))
        is_params_custom = self._is_params_modified(self.current_params, curr_p)

        if is_film_custom or is_paper_custom or is_params_custom:
            # Genuine darkroom edits exist -> Save .sdc
            try:
                cur_thumb = curr_p.get("graded_thumbnail_rgb")
                if cur_thumb is None:
                    cur_thumb = curr_p.get("thumbnail_rgb")
                self.current_params["print_mode"] = getattr(self, "current_print_mode", "optical")
                # 没选过相纸的照片不把默认纸写进 .sdc：写了就等于"选过"，之后在首选项
                # 换默认相纸时这张照片就不跟了。空串是 .sdc 里表示"未选择"的既有写法。
                paper_for_sdc = (self.current_paper_stock
                                 if self._photo_paper_is_explicit(curr_p) else "")
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    paper_for_sdc,
                    self.current_params,
                    thumbnail_rgb=cur_thumb,
                    print_mode=getattr(self, "current_print_mode", "optical"),
                    paper_explicit=self._photo_paper_is_explicit(curr_p),
                    paper_by_scope=self._paper_by_scope(curr_p)
                )
                curr_p["film_profile"] = self.current_film_stock
                curr_p["paper_profile"] = paper_for_sdc
                curr_p["print_mode"] = getattr(self, "current_print_mode", "optical")
                curr_p["params"] = dict(self.current_params)
                curr_p["crop"] = dict(self.current_params.get("crop", {}))
                curr_p["is_dirty"] = True
                curr_p["is_edited"] = True
                self._photo_is_dirty = True
                if hasattr(self, "filmstrip"):
                    self.filmstrip.set_photo_dirty(self.active_photo_id, True)
                    self.filmstrip.set_can_paste(True)
            except Exception as e:
                logger.warning(f"Failed saving SDC for {self._current_photo_path}: {e}")
        else:
            # Completely pristine RAW (no film, no paper, default params)
            if sdc_manager.has_sdc(self._current_photo_path):
                sdc_manager.delete_sdc(self._current_photo_path)
            curr_p["film_profile"] = "none"
            curr_p["paper_profile"] = "none"
            curr_p["print_mode"] = "scan"
            curr_p["params"] = None
            curr_p["crop"] = {
                "enabled": False, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
                "angle": 0.0, "rot_90": 0, "flip_h": False, "flip_v": False,
                "aspect_ratio": "original", "is_portrait": False
            }
            curr_p["is_dirty"] = False
            curr_p["is_edited"] = False
            self._photo_is_dirty = False
            if hasattr(self, "filmstrip"):
                self.filmstrip.set_photo_dirty(self.active_photo_id, False)

    def _refresh_lut(self):
        import time
        self._ensure_lut_bake_bridge()
        if self._live_lut_timer.isActive():
            self._live_lut_timer.stop()
        self._lut_job = None          # a committed bake supersedes any queued preview
        t0 = time.time()
        lut_3d = self.engine.get_3d_lut(
            self.current_film_stock,
            self.current_paper_stock,
            lut_size=33,
            params_dict=self.current_params,
            print_mode=getattr(self, "current_print_mode", "optical"),
        )
        for k in self.LUT_CHEM_KEYS:
            if k in self.current_params:
                self.canvas.params[k] = self.current_params[k]
        self._apply_lut(lut_3d)
        dur = (time.time() - t0) * 1000.0
        self.set_app_status(f"LUT 烘焙完成（{dur:.0f} ms）", timeout_ms=2000)

    def _commit_lut_preview(self):
        """Finish a DIR / morph drag.

        If a preview of exactly these parameters is already queued or running, let it land
        (it is the same 33^3 bake); otherwise bake synchronously so the committed image is
        exact and never a leftover approximation.
        """
        self._ensure_lut_bake_bridge()
        key = self._current_lut_key()
        if key == self._lut_applied_key:
            return
        # 点式链接手时这张 33³ 不会被采样，别为它同步烘一次（会卡住松手）。
        if self._on_ffchain_now():
            self._lut_applied_key = key
            self._lut_job = None
            return
        if key == self._lut_inflight_key:
            return
        if self._lut_job is not None and self._lut_job[0] == key:
            return
        self._refresh_lut()

    def _on_slider_committed(self, action: str = None):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        if getattr(self, "_interaction_watch", None):
            self._interaction_watch[0] = time.perf_counter()
        if hasattr(self, '_live_lut_timer') and self._live_lut_timer.isActive():
            self._live_lut_timer.stop()
        if getattr(self, "_pending_lut_refresh", False):
            self._pending_lut_refresh = False
            if getattr(self, "current_film_stock", "none") != "none":
                self._commit_lut_preview()
        self.push_undo_state(action=action)
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self._schedule_preview_refresh()

    def _schedule_preview_refresh(self, thumb_ms=60, cards_ms=220):
        """缩略图与候选卡预览走已有的防抖，而不是在提交时同步烘。

        这两样都是"看一眼就行"的产物，但一次全量刷新并不便宜：`_update_active_thumbnails`
        要离屏渲染 + 落盘缓存（~10 ms，缩略图缓存命中率低时更多），`_refresh_card_previews`
        要重启候选预览线程、并且每次为 53 张卡重算配对相纸（其中 23 次会重读偏好文件）。
        双击复位是两个提交事件叠在一起（第一次 release 提交点击位置、双击再复位），以前
        每个都同步跑一遍整条链 —— 用户看到的就是"卡一下"。

        用 60 / 220 ms 的单次定时器之后，连着的两个提交只烘最后一遍；数值与画布仍然在
        valueChanged 里立即更新，所以拖动手感不受影响。
        """
        if hasattr(self, '_live_thumb_timer'):
            self._live_thumb_timer.start(thumb_ms)
        if hasattr(self, '_cards_debounce_timer'):
            self._cards_debounce_timer.start(cards_ms)

    def _action_apply_auto_meter(self):
        """Reset exposure compensation to the calibrated 18% middle-gray baseline (0.00 EV)."""
        if not self._current_photo_path:
            return
        photo = next((p for p in self.photos if p and p.get("path") == self._current_photo_path), None)
        if not photo:
            return
        ev_val = 0.0
        self.current_params["exposure_ev"] = ev_val
        if hasattr(self, 'slider_ev'):
            self.slider_ev.set_value(ev_val)
        self.canvas.update_params(exposure_ev=ev_val)
        self._on_slider_committed(action="auto_meter")
        base_ev = float(photo.get("base_ev", 0.0))
        self.set_app_status(f"基准测光：18% 中灰（基准 {base_ev:+.2f} EV）", timeout_ms=2500)

    def _get_track_defaults(self):
        def_temp = getattr(self.slider_temp, "default_val", 5500.0) if hasattr(self, 'slider_temp') else 5500.0
        def_tint = getattr(self.slider_tint, "default_val", 1.0) if hasattr(self, 'slider_tint') else 1.0
        return {
            "exposure": {
                "exposure_ev": 0.0, "highlights": 0.0, "shadows": 0.0, "black_lift": 0.0,
                "color_temp": def_temp, "tint": def_tint,
                "film_format_mm": 35.0, "custom_wb": False
            },
            "optics": {
                "diffusion_family": "none", "diffusion_strength": 0.0, "diffusion_warmth": 0.0
            },
            "chemistry": {
                "dir_amount": 1.0, "dir_interlayer": 1.0, "dir_samelayer": 1.0,
                # Push is a development condition, so it belongs to this track's reset: dragging
                # "复原" here must return the film to standard development, not leave a +2 baked in.
                "push_ev": 0, "push_underexpose": False,
                # 胶片预闪也是胶片侧的曝光条件（显影前叠加的光），不属于放大机与相纸那一组。
                "pre_flash_camera": 0.0,
            },
            "enlarger": {
                "enlarger_illuminant": "TH-KG3",
                "enlarger_cyan": 0.0, "enlarger_magenta": 0.0, "enlarger_yellow": 0.0,
                "print_exposure": 1.0, "pre_flash": 0.0, "pre_flash_camera": 0.0,
                "morph_gamma": 1.0, "developer_exhaustion": 0.0
            },
            "halation": {
                "halation_amount": 1.0, "halation_spatial_scale": 1.0,
                "scatter_amount": 1.0, "boost_ev": 0.0,
                "halation": 1.0, "halation_boost": 0.0
            },
            "texture": {
                "halation_amount": 1.0, "halation_spatial_scale": 1.0,
                "scatter_amount": 1.0, "boost_ev": 0.0,
                "halation": 1.0, "halation_boost": 0.0
            },
            "grain": {
                # grain 存的是引擎键（0.4 = 标定参考），滑块显示倍率 = grain / 0.4
                "grain": 0.4, "grain_size": 1.0, "grain_colour": 1.0,
                "scan_noise": 0.0
            }
        }

    def _update_reset_buttons_state(self):
        """Item 5: Gray out and disable reset button when the section is at baseline defaults."""
        is_none = (getattr(self, "current_film_stock", "none") == "none")
        tracks = self._get_track_defaults()
        section_map = {
            "exposure": getattr(self, "sec_exposure", None),
            "optics": getattr(self, "sec_optics", None),
            "chemistry": getattr(self, "sec_chemistry", None),
            "enlarger": getattr(self, "sec_enlarger", None),
            "halation": getattr(self, "sec_halation", None),
            "grain": getattr(self, "sec_grain", None),
            "texture": getattr(self, "sec_texture", None),
        }
        for track_name, sec in section_map.items():
            if not sec or not hasattr(sec, "set_reset_enabled"):
                continue
            track_defs = tracks.get(track_name, {})
            is_default = all(
                self.current_params.get(k, def_v) == def_v or (isinstance(def_v, float) and abs(float(self.current_params.get(k, def_v) or 0) - def_v) < 1e-4)
                for k, def_v in track_defs.items()
                if k != "custom_wb"
            )
            if track_name == "optics":
                lens_cfg = self.current_params.get("lens", {})
                if isinstance(lens_cfg, dict):
                    is_lens_modified = (
                        bool(lens_cfg.get("profile_enabled", False)) or
                        bool(lens_cfg.get("remove_ca", False)) or
                        abs(float(lens_cfg.get("profile_distortion", 0.0))) > 0.01 or
                        abs(float(lens_cfg.get("profile_vignette", 0.0))) > 0.01 or
                        abs(float(lens_cfg.get("manual_distortion", 0.0))) > 0.01 or
                        abs(float(lens_cfg.get("manual_vignette_amount", 0.0))) > 0.01 or
                        abs(float(lens_cfg.get("manual_vignette_midpoint", 0.0))) > 0.01
                    )
                    if is_lens_modified:
                        is_default = False
            if track_name == "chemistry" and (is_none or not self._dir_available()):
                is_default = True   # stock carries no DIR chemistry: keep the section's reset off
            if track_name == "enlarger" and (is_none or not self._enlarger_available()):
                is_default = True   # enlarger is not available: keep reset off
            if track_name in ("halation", "texture") and is_none:
                is_default = True
            if track_name == "grain" and is_none:
                is_default = True
            sec.set_reset_enabled(not is_default)

    def rebuild_render_caches(self):
        """Delete the LUT and grain-tile caches, then drop the in-memory ones.

        Two directories, nothing else: LUT tables (``.npy``) and laid film tiles.  No wildcards on
        user data -- if a path does not look like the cache it is skipped and reported.
        """
        from ui.confirm_dialog import show_darkroom_confirm, show_darkroom_info
        try:
            from core.path_utils import get_user_lut_cache_dir, get_user_cache_dir
            lut_dir = str(get_user_lut_cache_dir())
            grain_dir = str(os.path.join(str(get_user_cache_dir()), "film_grain"))
        except Exception as exc:                       # noqa: BLE001
            show_darkroom_info(self, "重建渲染缓存", "找不到缓存目录", str(exc), badge_symbol="!")
            return

        sizes = {}
        for path in (lut_dir, grain_dir):
            total = 0
            for name in (os.listdir(path) if os.path.isdir(path) else []):
                full = os.path.join(path, name)
                if os.path.isfile(full):
                    total += os.path.getsize(full)
            sizes[path] = total
        total_mb = sum(sizes.values()) / (1024 * 1024)
        detail = "\n".join(f"  {os.path.basename(p)}: {s / 1024 / 1024:.1f} MB"
                           for p, s in sizes.items())
        if not show_darkroom_confirm(
                self, "重建渲染缓存", "删除并重建渲染缓存",
                f"将删除以下缓存（{total_mb:.0f} MB）：\n{detail}\n\n"
                "下次渲染会重新烘焙，第一次会慢一些。",
                confirm_text="删除并重建", cancel_text="取消",
                badge_symbol="!", is_danger=True):
            return

        removed, failed = 0, []
        for path in (lut_dir, grain_dir):
            if not os.path.isdir(path):
                continue
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if not os.path.isfile(full):
                    continue
                try:
                    os.remove(full)
                    removed += 1
                except Exception as exc:               # noqa: BLE001
                    failed.append(f"{name}: {exc}")

        engine = getattr(self, "engine", None)
        if engine is not None:
            for attr in ("_lut_cache", "_lut_sim_cache", "_lut_loge_cache",
                         "_film_density_cache", "_neutral_push_cache"):
                cache = getattr(engine, attr, None)
                if isinstance(cache, dict):
                    cache.clear()
        if hasattr(self, "canvas"):
            self.canvas.update()
        message = f"已删除 {removed} 个缓存文件。"
        if failed:
            message += "\n以下文件删不掉（可能被占用）：\n" + "\n".join(failed[:8])
        show_darkroom_info(self, "重建渲染缓存", "缓存已重建", message, badge_symbol="✓")

    def reset_track(self, track_name):
        is_none = (getattr(self, "current_film_stock", "none") == "none")
        if is_none and track_name in ("chemistry", "enlarger", "halation", "grain", "texture"):
            return
        if track_name == "chemistry" and not self._dir_available():
            return
        if track_name == "enlarger" and not self._enlarger_available():
            return
        defaults = self._get_track_defaults()
        if track_name not in defaults:
            return
        changed = False
        if track_name == "optics":
            self.current_params["lens"] = self._get_default_lens_params()
            if hasattr(self, 'combo_lens_setup'):
                self._auto_detect_lens_profile()
            self._apply_lens_to_canvas()
            changed = True
        for k, v in defaults[track_name].items():
            cur_v = self.current_params.get(k)
            if cur_v != v and (not isinstance(v, float) or abs(float(cur_v or 0) - v) >= 1e-4):
                changed = True
                self.current_params[k] = v
                self.canvas.update_params(**{k: v})
        if not changed:
            self._update_reset_buttons_state()
            return

        # Tracks that carry LUT-baked chemistry (DIR / morph / exhaustion) must rebake, not just
        # move uniforms - the shader no longer approximates DIR.
        lut_keys_changed = any(k in self.LUT_CHEM_KEYS for k in defaults[track_name])
        if "enlarger_illuminant" in defaults[track_name] or lut_keys_changed:
            self._refresh_lut()
        self._update_all_sliders_from_params()
        self._schedule_histogram_update()

        # Item 5: Resetting an unedited photo never marks it dirty!
        if getattr(self, "_photo_is_dirty", False):
            all_defs = {}
            for d in defaults.values():
                all_defs.update(d)
            all_default = all(
                self.current_params.get(k, dv) == dv or (isinstance(dv, float) and abs(float(self.current_params.get(k, dv) or 0) - dv) < 1e-4)
                for k, dv in all_defs.items()
                if k != "custom_wb"
            )
            if all_default and self.current_film_stock == "none" and self.current_paper_stock == "none":
                self._photo_is_dirty = False
                if self.active_photo_id:
                    photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                    if photo:
                        photo["is_dirty"] = False
                    self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            else:
                self._mark_current_photo_dirty()
                self._save_current_photo_sdc()
        self._update_reset_buttons_state()
        self._update_active_thumbnails()
        self._refresh_card_previews()
        self.push_undo_state(action=f"reset_track_{track_name}")
        self.setFocus()

    def reset_all_params(self):
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                self.canvas.crop_overlay.reset_crop()
                self.set_app_status("已重置裁切", timeout_ms=2000)
            return

        defaults = self._get_track_defaults()
        all_defs = {}
        for d in defaults.values():
            all_defs.update(d)
        changed = False
        for k, v in all_defs.items():
            cur_v = self.current_params.get(k)
            if cur_v != v and (not isinstance(v, float) or abs(float(cur_v or 0) - v) >= 1e-4):
                changed = True
                self.current_params[k] = v
                self.canvas.update_params(**{k: v})
        # Even when every parameter is already at its default, a reset still has to put the
        # paper back on the preference default -- that is the whole point of calling reset.
        paper_moved = self._restore_preference_paper()
        if not changed:
            if paper_moved:
                # The paper decides which LUT is baked, so a reset that only moved the paper
                # still has to rebake and refresh what shows the old one.
                self._refresh_lut()
                self._update_stock_mode_ui()
                self._update_active_thumbnails()
                self._refresh_card_previews()
            self._update_reset_buttons_state()
            return
        self._refresh_lut()
        self._update_all_sliders_from_params()
        self._schedule_histogram_update()
        if getattr(self, "_photo_is_dirty", False):
            if self.current_film_stock == "none" and self.current_paper_stock == "none":
                self._photo_is_dirty = False
                if self.active_photo_id:
                    photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                    if photo:
                        photo["is_dirty"] = False
                    self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            else:
                self._mark_current_photo_dirty()
                self._save_current_photo_sdc()
        self._update_reset_buttons_state()
        self._update_active_thumbnails()
        self._refresh_card_previews()
        self.push_undo_state(action="reset_all_params")

    def on_reset_to_open(self):
        """Dispatch '重置为打开状态' (Ctrl+Alt+R) according to workspace mode."""
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                self.canvas.crop_overlay.reset_to_initial_state()
                self.set_app_status("已恢复至打开时的裁切状态", timeout_ms=2000)
        else:
            self.reset_to_open_state()

    def _restore_preference_paper(self):
        """Re-seat a photo that never picked a paper on the preference default.

        ``reset`` is supposed to return the *parameters* to their defaults and nothing else, so
        a paper the user chose is left exactly where it is.  But a photo that never chose one
        owes its paper to the preference -- and any path that lets it fall to "none" makes the
        panel read 胶片直扫 and the render fall back to a flat scan, which is not a reset of
        anything the user did.  Called from every reset entry point; returns True when it
        actually changed the paper (the caller then has to rebake the LUT).
        """
        photo = None
        if self.active_photo_id:
            photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
        if photo is not None and self._photo_paper_is_explicit(photo):
            return
        paper = self._default_paper_for(self.current_film_stock)
        # `_default_paper_for` 已经按链条取值（转换卷落点式链默认、原生卷落首选项），所以
        # 这里不再需要"越界了就退回另一张"的分支。旧写法是
        # ``default_paper_for(...) or paper`` —— 原生卷那边 `default_paper_for` 返回 None，
        # `or` 于是把越界的首选项值原样留下，重置完停在一张置灰的纸上（实测）。
        mode = "scan" if str(paper or "none") in ("none", "") else "optical"
        if paper == self.current_paper_stock and mode == self.current_print_mode:
            return False
        self.current_paper_stock = paper
        self.current_print_mode = mode
        self.current_params["print_mode"] = mode
        if hasattr(self.canvas, "update_params"):
            self.canvas.update_params(paper_profile=paper, print_mode=mode)
        return True

    def reset_to_open_state(self):
        """Fully restore photo to virgin unedited RAW state, completely wiping all parameters AND crop."""
        import copy
        default_crop = {
            "enabled": False,
            "x": 0.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "angle": 0.0,
            "rot_90": 0,
            "flip_h": False,
            "flip_v": False,
            "aspect_ratio": "original",
            "is_portrait": False
        }
        self.current_params["crop"] = default_crop
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = default_crop
        if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
            self.canvas.crop_overlay.reset_all_crop()

        # The film goes back to bypass, but the *paper* lands on the preference default, not on
        # "胶片直扫".  Writing "none" here made every reset look like it had also thrown the
        # paper away, so the next film picked started from a blank sheet and the panel read
        # 胶片直扫 even though a default paper is configured (asked for explicitly).  Same rule
        # as the clear-.sdc path: bypass renders through its own table, the paper only decides
        # where the panel sits and what the next stock selection inherits.
        self.current_film_stock = "none"
        # 旁通没有相纸语义，这里写的是"下一次选卷要继承的那张"——所以取原生链的默认纸
        # （转换卷的那张纸在旁通面板上没有意义）。选卷时 on_switch_film 会按链条重新落。
        self.current_paper_stock = self._default_paper_for("none")
        self.current_print_mode = (
            "scan" if str(self.current_paper_stock or "none") in ("none", "") else "optical")
        self.current_params["print_mode"] = self.current_print_mode

        defaults = self._get_track_defaults()
        all_defs = {}
        for d in defaults.values():
            all_defs.update(d)
        for k, v in all_defs.items():
            self.current_params[k] = v
            self.canvas.update_params(**{k: v})

        self._refresh_lut()
        self._update_all_sliders_from_params()
        self._schedule_histogram_update()
        self._update_stock_mode_ui()
        self.canvas.update_params(crop=dict(default_crop))
        self.canvas.fit_to_view(animate=False)
        self.canvas.update()
        self._update_reset_buttons_state()
        self._update_active_thumbnails()
        self._refresh_card_previews()
        self.push_undo_state(action="reset_to_open")

        self._photo_is_dirty = False
        if self.active_photo_id:
            photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if photo:
                photo["is_dirty"] = False
                photo["is_edited"] = False
                photo["film_profile"] = "none"
                # "" = "never picked", so this photo keeps following the preference default.
                photo["paper_profile"] = ""
                photo["paper_explicit"] = False
                # 重置是"这张照片回到初始"：另一条链的那份选择也一起清掉，否则之后切卷往返
                # 会把重置前的那张纸又恢复出来，和"已恢复至原始底片状态"自相矛盾。
                photo.pop("paper_by_scope", None)
                photo["print_mode"] = self.current_print_mode
                photo["params"] = copy.deepcopy(self.current_params)
                photo["crop"] = dict(default_crop)
                p_path = photo.get("path", "")
                if p_path and sdc_manager.has_sdc(p_path):
                    sdc_manager.delete_sdc(p_path)
            self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            self.filmstrip.set_photo_crop(self.active_photo_id, False)

        self.set_app_status("已恢复至原始底片状态", timeout_ms=2000)

    def _push_speed_gain(self, stops):
        """EV the pushed curve actually gives back on this stock (0 when unknown)."""
        engine = getattr(self, "engine", None)
        if engine is None or not stops:
            return 0.0
        try:
            return float(engine.push_speed_gain(getattr(self, "current_film_stock", "none"), stops))
        except Exception:                           # noqa: BLE001 - never break a paint
            return 0.0

    def _push_underexposure_outstanding(self):
        """How much of the N-stop underexposure the user has not given back yet, in EV."""
        current = int(round(float(self.current_params.get("push_ev", 0) or 0)))
        if not current:
            return 0.0
        exposure = float(self.current_params.get("exposure_ev", 0.0) or 0.0)
        ref = self.current_params.get("push_ref_exposure")
        if ref is None:
            return float(current)
        return max(0.0, float(current) - (exposure - float(ref)))

    def _push_readout(self, stops):
        """Tooltip text for the 迫冲显影 row: provenance, and what the setting does to exposure.

        Two different statements live here, and conflating them is what made the control feel
        mysterious.  Unlinked (the default) the row only swaps the development curve, so the
        frame keeps its exposure and gains contrast.  Linked, it also models "shot N stops under",
        and then the honest number is *not* "+N stops of speed": measured tables give back only
        0.2-0.6 EV per stop, which is exactly why a pushed frame comes out darker.
        """
        if not stops:
            return "该胶卷没有迫冲数据（材料上无通行迫冲表，或未收录）"
        engine = getattr(self, "engine", None)
        label = ""
        if engine is not None:
            try:
                label = engine.push_status(getattr(self, "current_film_stock", "none"))[1]
            except Exception:                       # noqa: BLE001 - never break a paint
                label = ""
        current = int(round(float(self.current_params.get("push_ev", 0) or 0)))
        if not current:
            return label
        gain = self._push_speed_gain(current)
        linked = bool(self.current_params.get("push_underexpose"))

        if not linked:
            # Nothing is applied on top of the measured curve any more: no lifted black, no
            # grain multiplier.  A pushed negative is denser, and both the tone and the grain
            # follow from that -- which is also why fotufilm can be used as a reference.
            body = (f"当前：显影 +{current} 档，曝光不联动（只换特性曲线）\n"
                    f"实测：显影补回约 {gain:.1f} EV，暗部更密、颗粒自然变粗\n"
                    f"要模拟“欠曝 {current} 档再迫冲”请勾选「模拟欠曝」")
            return f"{label}\n{body}" if label else body

        exposure = float(self.current_params.get("exposure_ev", 0.0) or 0.0)
        ref = self.current_params.get("push_ref_exposure")
        if ref is None:
            # No record of where the exposure landed (sidecar from before this feature): take "now"
            # as the reference, which reads as a clean +N underexposure and never replays an offset.
            ref = exposure
            self.current_params["push_ref_exposure"] = ref
        given_back = exposure - float(ref)
        underexposed = current - given_back
        if underexposed <= 0.0:
            body = f"当前：欠曝 0 EV，显影 +{current} 档 —— 属过显影，不是迫冲"
        else:
            net = max(0.0, underexposed - gain)
            body = (f"当前：欠曝 {underexposed:.1f} EV，显影 +{current} 档"
                    f"（实测补回 {gain:.1f} EV，净暗约 {net:.1f} 档）")
            if abs(given_back) >= 0.05:
                body += f"\n曝光已手动补回 {given_back:+.1f} EV"
        return f"{label}\n{body}" if label else body

    def _apply_push_exposure_offset(self, dv):
        """Move the scene exposure by ``dv`` EV for a linked push change."""
        if not dv:
            return
        ev = round(float(self.current_params.get("exposure_ev", 0.0) or 0.0) + dv, 2)
        ev = max(-3.0, min(3.0, ev))
        self.current_params["exposure_ev"] = ev
        self.current_params["push_ref_exposure"] = ev
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.update_params(exposure_ev=ev)
        if hasattr(self, "slider_ev"):
            was_quiet = getattr(self, "_is_updating_sliders", False)
            self._is_updating_sliders = True
            try:
                self.slider_ev.set_value(ev)
            finally:
                self._is_updating_sliders = was_quiet

    def _on_push_underexpose_toggled(self, checked):
        """Linked workflow on/off. Turning it off gives back what is still owed, nothing more."""
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        self.current_params["push_underexpose"] = bool(checked)
        if checked:
            # Start the workflow from "the film was shot N stops under" if we have not already.
            if self.current_params.get("push_ref_exposure") is None:
                stops = int(round(float(self.current_params.get("push_ev", 0) or 0)))
                if stops:
                    self._apply_push_exposure_offset(-stops)
        else:
            owed = self._push_underexposure_outstanding()
            if owed > 0.01:
                self._apply_push_exposure_offset(owed)
        self._refresh_push_readout()
        self._mark_current_photo_dirty()

    def _apply_push_tooltip(self, stops):
        combo = getattr(self, "combo_push", None)
        if combo is None:
            return
        text = self._push_readout(stops)
        combo.setToolTip(text)
        combo.combo.setToolTip(text)
        combo.label.setToolTip(text)

    def _refresh_push_readout(self):
        combo = getattr(self, "combo_push", None)
        if combo is None:
            return
        stops = [combo.combo.itemData(i) for i in range(combo.combo.count())]
        self._apply_push_tooltip([s for s in stops if s])

    def _update_push_ui_state(self):
        """Rebuild the 迫冲显影 row for the current stock.

        The option list is the stock's actual table, never a fixed 0/+1/+2/+3: offering +3 on a
        +1/+2 stock would silently render +2 (the label would lie), and offering anything at all on
        a stock with no table would invent data. When there is no table the row is disabled and
        says why.
        """
        combo = getattr(self, "combo_push", None)
        if combo is None:
            return
        stock = getattr(self, "current_film_stock", "none")
        stops = []
        engine = getattr(self, "engine", None)
        if engine is not None and stock not in (None, "", "none"):
            try:
                stops = list(engine.push_stops(stock))
            except Exception:                       # noqa: BLE001 - never break a film switch
                stops = []
        current = int(round(float(self.current_params.get("push_ev", 0) or 0)))
        if current not in stops:
            current = 0
        combo.set_items([("标准冲洗 (0)", 0)] + [(f"+{s} 档", s) for s in stops],
                        default_val=current)
        combo.setEnabled(bool(stops))
        combo.label.setEnabled(bool(stops))
        # The switch is meaningless without a table, and it is the thing that explains the
        # darkening -- leaving it enabled on a stock with no push would offer a lie.
        chk = getattr(self, "chk_push_under", None)
        if chk is not None:
            chk.setEnabled(bool(stops))
            was_quiet = getattr(self, "_is_updating_sliders", False)
            self._is_updating_sliders = True
            try:
                chk.setChecked(bool(self.current_params.get("push_underexpose")))
            finally:
                self._is_updating_sliders = was_quiet
        self._apply_push_tooltip(stops)
        if int(round(float(self.current_params.get("push_ev", 0) or 0))) != current:
            # Losing the stop count (film switch to a stock without that table, or to none) also
            # ends the linked underexposure: what is still owed goes back instead of being
            # stranded on the exposure slider with nothing left to explain it.
            if self.current_params.get("push_underexpose"):
                owed = self._push_underexposure_outstanding()
                if owed > 0.01:
                    self._apply_push_exposure_offset(owed)
            self.current_params["push_ev"] = current
            canvas = getattr(self, "canvas", None)
            if canvas is not None:
                canvas.update_params(push_ev=current)

    def _update_all_sliders_from_params(self):
        self._is_updating_sliders = True
        try:
            p = self.current_params
            if hasattr(self, 'slider_ev'): self.slider_ev.set_value(p.get("exposure_ev", 0.0))
            if hasattr(self, 'slider_highlights'): self.slider_highlights.set_value(p.get("highlights", 0.0))
            if hasattr(self, 'slider_shadows'): self.slider_shadows.set_value(p.get("shadows", 0.0))
            if hasattr(self, 'slider_temp'): self.slider_temp.set_value(p.get("color_temp", 5500.0))
            if hasattr(self, 'slider_tint'): self.slider_tint.set_value(p.get("tint", 1.0))
            if hasattr(self, 'combo_format'): self.combo_format.set_value(p.get("film_format_mm", 35.0))

            if hasattr(self, 'combo_diffusion'): self.combo_diffusion.set_value(p.get("diffusion_family", "none"))
            # Older sessions stored strength 0 as "no filter". The combo has no 0 entry any
            # more and set_value() silently ignores unknown values, so an unclamped 0 would
            # leave the row showing a stale density next to an active family.
            if hasattr(self, 'combo_diff_strength'):
                _ds = float(p.get("diffusion_strength", 0.0) or 0.0)
                self.combo_diff_strength.set_value(_ds if _ds > 0.0 else 0.125)
            if hasattr(self, 'slider_diff_strength'):
                _ds = float(p.get("diffusion_strength", 0.0) or 0.0)
                self.slider_diff_strength.set_value(_ds if _ds > 0.0 else 0.125)
            if hasattr(self, 'slider_diff_warmth'): self.slider_diff_warmth.set_value(p.get("diffusion_warmth", 0.0))

            if hasattr(self, 'slider_dir_amount'): self.slider_dir_amount.set_value(p.get("dir_amount", 1.0))
            if hasattr(self, 'combo_push'): self._update_push_ui_state()

            if hasattr(self, 'combo_illuminant'): self.combo_illuminant.set_value(p.get("enlarger_illuminant", "TH-KG3"))
            if hasattr(self, 'slider_magenta'): self.slider_magenta.set_value(p.get("enlarger_magenta", 0.0))
            if hasattr(self, 'slider_yellow'): self.slider_yellow.set_value(p.get("enlarger_yellow", 0.0))
            if hasattr(self, 'slider_cyan'): self.slider_cyan.set_value(p.get("enlarger_cyan", 0.0))
            if hasattr(self, 'slider_print_exp'): self.slider_print_exp.set_value(p.get("print_exposure", 1.0))
            if hasattr(self, 'slider_preflash_camera'): self.slider_preflash_camera.set_value(p.get("pre_flash_camera", 0.0))
            if hasattr(self, 'slider_preflash'): self.slider_preflash.set_value(p.get("pre_flash", 0.0))
            if hasattr(self, 'slider_black_lift'): self.slider_black_lift.set_value(p.get("black_lift", 0.0))
            if hasattr(self, 'slider_morph_gamma'): self.slider_morph_gamma.set_value(p.get("morph_gamma", 1.0))
            if hasattr(self, 'slider_exhaustion'): self.slider_exhaustion.set_value(p.get("developer_exhaustion", 0.0))

            # The parameter is the linear multiple; this row draws stops (0 = the stock).
            _hal_scale = p.get("halation_amount", p.get("halation", 1.0))
            _hal_stops = stops_from_scale(_hal_scale)
            if hasattr(self, 'slider_halation_amount'): self.slider_halation_amount.set_value(_hal_stops)
            if hasattr(self, 'slider_halation_spatial_scale'): self.slider_halation_spatial_scale.set_value(p.get("halation_spatial_scale", 1.0))
            if hasattr(self, 'slider_scatter_amount'): self.slider_scatter_amount.set_value(p.get("scatter_amount", 1.0))
            if hasattr(self, 'slider_boost_ev'): self.slider_boost_ev.set_value(p.get("boost_ev", p.get("halation_boost", 0.0)))
            if hasattr(self, 'slider_halation'): self.slider_halation.set_value(_hal_stops)
            if hasattr(self, 'slider_halation_boost'): self.slider_halation_boost.set_value(p.get("boost_ev", p.get("halation_boost", 0.0)))

            # 银盐颗粒：强度滑块显示 σ 倍率，内部 grain = 0.4/σ²（Selwyn 反解；grain=0 → 无颗粒 → 滑块 0）
            _g = float(p.get("grain", 0.4))
            if hasattr(self, 'slider_grain_strength'):
                self.slider_grain_strength.set_value((0.4 / _g) ** 0.5 if _g > 1e-6 else 0.0)
            if hasattr(self, 'slider_grain_size'): self.slider_grain_size.set_value(p.get("grain_size", 1.0))
            if hasattr(self, 'slider_grain_colour'): self.slider_grain_colour.set_value(p.get("grain_colour", 1.0))
            if hasattr(self, 'slider_scan_noise'): self.slider_scan_noise.set_value(p.get("scan_noise", 0.0))

            # Lens correction controls
            lens_cfg = p.get("lens")
            if not isinstance(lens_cfg, dict):
                lens_cfg = self._get_default_lens_params()
                p["lens"] = lens_cfg

            act_tab = lens_cfg.get("active_tab", "profile")
            if hasattr(self, 'btn_lens_tab_profile') and hasattr(self, 'btn_lens_tab_manual'):
                self.btn_lens_tab_profile.setChecked(act_tab != "manual")
                self.btn_lens_tab_manual.setChecked(act_tab == "manual")
            if hasattr(self, 'panel_lens_profile') and hasattr(self, 'panel_lens_manual'):
                self.panel_lens_profile.setVisible(act_tab != "manual")
                self.panel_lens_manual.setVisible(act_tab == "manual")

            if hasattr(self, 'chk_remove_ca'):
                self.chk_remove_ca.setChecked(bool(lens_cfg.get("remove_ca", False)))
            if hasattr(self, 'chk_lens_profile_enabled'):
                self.chk_lens_profile_enabled.setChecked(bool(lens_cfg.get("profile_enabled", False)))
            if hasattr(self, 'combo_lens_setup'):
                self.combo_lens_setup.set_value(lens_cfg.get("setup", "auto"))

            if hasattr(self, 'combo_lens_make'):
                self._sync_lens_combos_from_cfg(lens_cfg)

            if hasattr(self, 'slider_lens_distortion'):
                self.slider_lens_distortion.set_value(lens_cfg.get("profile_distortion", 0.0))
            if hasattr(self, 'slider_lens_vignette'):
                self.slider_lens_vignette.set_value(lens_cfg.get("profile_vignette", 0.0))

            if hasattr(self, 'slider_manual_distortion'):
                self.slider_manual_distortion.set_value(lens_cfg.get("manual_distortion", 0.0))
            if hasattr(self, 'slider_manual_vignette_amount'):
                self.slider_manual_vignette_amount.set_value(lens_cfg.get("manual_vignette_amount", 0.0))
            if hasattr(self, 'slider_manual_vignette_midpoint'):
                self.slider_manual_vignette_midpoint.set_value(lens_cfg.get("manual_vignette_midpoint", 0.0))
        finally:
            self._is_updating_sliders = False
        self._update_reset_buttons_state()
        self._update_auto_ca_ui()
        self._ensure_auto_ca()
        self._update_diffusion_ui_state()
        self._update_lens_ui_state()

    def _sync_canvas_params(self, **extra):
        """Safely forwards current_params and active stock/print_mode state to canvas without keyword duplication."""
        merged = dict(self.current_params)
        merged["film_stock"] = getattr(self, "current_film_stock", "none")
        merged["paper_profile"] = getattr(self, "current_paper_stock", "none")
        merged["print_mode"] = getattr(self, "current_print_mode", "optical")
        merged.update(extra)
        self.canvas.update_params(**merged)

    def _sync_full_params_to_pipeline(self, fit_crop_if_changed: bool = False, refresh_lut: bool = True):
        """Unified entry point to push self.current_params to canvas shader uniforms, 3D LUT, and UI indicators."""
        old_crop = getattr(self.canvas, "params", {}).get("crop", {})
        self._sync_canvas_params()
        if fit_crop_if_changed:
            new_crop = self.current_params.get("crop", {})
            if old_crop != new_crop and hasattr(self, 'canvas') and self.canvas._has_image:
                self.canvas.fit_to_view(animate=False)
        if refresh_lut:
            lut_params = dict(self.current_params)
            lut_params["print_mode"] = self.current_print_mode
            eff_paper = self.current_paper_stock if self.current_print_mode == "optical" else "none"
            lut_3d = self.engine.get_3d_lut(self.current_film_stock, eff_paper, lut_size=33, params_dict=lut_params)
            self._apply_lut(lut_3d)
        self._schedule_histogram_update()
        self._update_diffusion_ui_state()
        self._update_stock_mode_ui()
        self._update_active_thumbnails()
        self._refresh_card_previews()

    def toggle_crop_mode(self):
        if not hasattr(self, 'canvas') or not self.canvas._has_image:
            if hasattr(self, 'btn_crop'):
                self.btn_crop.setChecked(False)
            return

        is_editing = getattr(self.canvas, "is_crop_editing", False)
        if is_editing:
            self.canvas.crop_overlay.apply_crop()
        else:
            self._enter_crop_workspace()

    # 裁切的"未裁切"基准。切换照片、清除裁切、退出裁切工作区都拿它对齐，避免各处各写一份
    # 字面量而后漏掉某个键（aspect_ratio 漏了就会沿用上一张的比例）。
    @staticmethod
    def _default_crop_dict():
        return {
            "enabled": False,
            "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
            "angle": 0.0, "rot_90": 0,
            "flip_h": False, "flip_v": False,
            "aspect_ratio": "original",
            "custom_ratio": [5.0, 7.0],
            "is_portrait": False,
        }

    @classmethod
    def _normalize_crop(cls, crop):
        """把任意来源的裁切数据收敛成一份完整的、可直接下发的 dict。

        ``None`` / 非 dict / 空 dict 一律视为"未裁切"；``enabled`` 为假时几何也清零，因为
        "enabled=False 但 x/y/w/h 还是上一张的"这种脏数据会被画布和侧栏当成真裁切读出来。
        """
        out = cls._default_crop_dict()
        if isinstance(crop, dict):
            out.update({k: v for k, v in crop.items() if k in out or k == "custom_ratio"})
        if not out.get("enabled", False):
            out.update(x=0.0, y=0.0, w=1.0, h=1.0, angle=0.0, rot_90=0,
                       flip_h=False, flip_v=False, aspect_ratio="original", is_portrait=False)
        return out

    def _apply_photo_crop_state(self, photo):
        """把画布 / 遮罩 / 侧栏的裁切对齐到 ``photo`` 自己。

        来源优先级：内存里的 current_params（刚才由 photo["params"] 或 .sdc 合并进来）
        → photo["crop"] → 整幅未裁切。存在 photo["crop"] 但 params 里没有的情况（旧会话、
        只写了裁切字段的路径），所以这里补一层兜底。
        """
        crop = self._normalize_crop(self.current_params.get("crop") if isinstance(self.current_params.get("crop"), dict) else None)
        if not crop["enabled"] and photo is not None:
            alt = photo.get("crop")
            if isinstance(alt, dict) and alt.get("enabled"):
                crop = self._normalize_crop(alt)
        # 若切换发生在裁切工作区内，先真正退出（否则遮罩还挂在这张照片上）
        if getattr(self.canvas, "is_crop_editing", False):
            self._exit_crop_workspace()
        self.current_params["crop"] = dict(crop)
        self.canvas.set_crop_state(dict(crop))
        if hasattr(self, "crop_sidebar") and self.crop_sidebar:
            self.crop_sidebar.sync_from_overlay(dict(crop))

    def _enter_crop_workspace(self):
        if hasattr(self, 'btn_crop'):
            self.btn_crop.setChecked(True)
        if hasattr(self, 'act_crop'):
            self.act_crop.setText("退出裁切")
        if hasattr(self, 'act_reset_to_open'):
            self.act_reset_to_open.setText("重置为打开时状态")

        # Mutual exclusion: Exit compare mode and disable compare buttons while in crop mode
        if self.current_params.get("view_mode", 0) != 0:
            self.set_view_mode(0)
        if hasattr(self, 'btn_split'):
            self.btn_split.setChecked(False)
            self.btn_split.setEnabled(False)
        if hasattr(self, 'act_compare'):
            self.act_compare.setEnabled(False)

        # 1. Switch right sidebar stack to Crop panel (Page 1)
        if hasattr(self, 'sidebar_stack'):
            self.sidebar_stack.setCurrentIndex(1)

        # 2. Start crop overlay session & sync controls
        cur_crop = self._normalize_crop(self.current_params.get("crop"))
        self._pre_crop_backup = copy.deepcopy(cur_crop)
        self.canvas.start_crop_mode(cur_crop)
        if hasattr(self, 'crop_sidebar'):
            self.crop_sidebar.sync_from_overlay(self.canvas.crop_overlay.get_crop_dict())
        self._update_menu_actions_state()

    def _exit_crop_workspace(self):
        if hasattr(self, 'btn_crop'):
            self.btn_crop.setChecked(False)
        if hasattr(self, 'act_crop'):
            self.act_crop.setText("裁切")
        if hasattr(self, 'act_reset_to_open'):
            self.act_reset_to_open.setText("重置为打开状态")

        # Re-enable compare buttons
        if hasattr(self, 'btn_split'):
            self.btn_split.setEnabled(True)
        if hasattr(self, 'act_compare'):
            self.act_compare.setEnabled(True)

        # 1. Switch right sidebar stack back to Develop cards (Page 0)
        if hasattr(self, 'sidebar_stack'):
            self.sidebar_stack.setCurrentIndex(0)

        # 2. Exit canvas crop mode & refresh
        self.canvas.exit_crop_mode()
        self.canvas.update()
        self._update_menu_actions_state()

    def _on_crop_live(self, crop_dict):
        self.current_params["crop"] = crop_dict
        if hasattr(self, 'canvas'):
            self.canvas.params["crop"] = crop_dict
            self.canvas.update()

    def _on_crop_applied(self, crop_dict):
        self._exit_crop_workspace()
        self.current_params["crop"] = crop_dict
        self._mark_current_photo_dirty()
        self.canvas.update_params(crop=crop_dict)
        # Animated, not snapped: ``_exit_crop_workspace`` already started the glide back to fit,
        # and ``fit_to_view(animate=False)`` stops glides by design -- so the exit animation was
        # cancelled on the line right after it started (reported as "退出裁切的缩放动画丢失").
        self.canvas.fit_to_view(animate=True)
        self._save_current_photo_sdc()
        self.push_undo_state()
        if hasattr(self, "filmstrip") and self.active_photo_id:
            self.filmstrip.set_photo_crop(self.active_photo_id, crop_dict)
        target_photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
        if target_photo:
            target_photo["crop"] = dict(crop_dict)
            if "params" in target_photo and isinstance(target_photo["params"], dict):
                target_photo["params"]["crop"] = dict(crop_dict)
        self._update_active_thumbnails()
        if hasattr(self, '_cards_debounce_timer'):
            self._cards_debounce_timer.start(100)
        else:
            self._refresh_card_previews()

    def _on_crop_cancelled(self):
        self._exit_crop_workspace()
        reverted_crop = getattr(self, "_pre_crop_backup", {})
        self.current_params["crop"] = copy.deepcopy(reverted_crop)
        if hasattr(self, 'canvas') and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = copy.deepcopy(reverted_crop)
            self.canvas.update()
        # Animated for the same reason as ``_on_crop_applied``: the snap would cancel the glide
        # that ``_exit_crop_workspace`` just started.
        self.canvas.fit_to_view(animate=True)
        if hasattr(self, "filmstrip") and self.active_photo_id:
            self.filmstrip.set_photo_crop(self.active_photo_id, reverted_crop)
        self._update_active_thumbnails()
        if hasattr(self, '_cards_debounce_timer'):
            self._cards_debounce_timer.start(100)
        else:
            self._refresh_card_previews()

    def _default_negative_paper(self):
        """首选项里的「原生链默认相纸」（spektrafilm 单表）。

        "none" 是合法值（胶片直扫，不印相），出厂默认就是它。配置里可能是旧 id、被删掉的
        相纸或空串，这几种退回类常量，免得新建照片落到一个不存在的纸 id 上。每次读配置而
        不是缓存：首选项改完要立刻生效。

        2026-11-13 起这里只管原生链：只有点式链才有的纸（Eterna CP / Agfa CP30 …）不再是
        合法的原生默认值，读到就退回 `DEFAULT_NEGATIVE_PAPER`。旧配置的迁移见
        `config_manager.migrate_default_papers()`。
        """
        fallback = self.DEFAULT_NEGATIVE_PAPER
        try:
            from core import config_manager
            pid = config_manager.get_preference("default_negative_paper", fallback)
        except Exception:
            return fallback
        if pid == "none":
            return "none"
        if not pid or pid in ("", "screen"):
            return fallback
        ffchain = set(getattr(self.engine, "FFCHAIN_PAPERS", ()) or ())
        native = set(getattr(self.engine, "NATIVE_PAPERS", ()) or ())
        # 纯点式纸不合法；自定义纸（不在 FFCHAIN_PAPERS 里）照旧放行 —— 用户导入的相纸
        # 走原生链，把它判掉等于让「默认相纸」对自定义纸失效。
        if native and pid in ffchain and pid not in native:
            return fallback
        try:
            ids = {str(item.get("id")) for item in self.engine.get_paper_stocks()}
        except Exception:
            ids = set()
        return pid if (not ids or pid in ids) else fallback

    def _default_ffchain_paper(self):
        """首选项里的「转换卷默认相纸」（fotufilm 点式链）。

        转换卷一定能印相，所以这里不接 `none`：读到空/直扫/退役的纸都退回结构默认
        （`kodak_2383`，也就是点式链一直以来的落点，行为不变）。
        """
        fallback = self.DEFAULT_FFCHAIN_PAPER
        try:
            from core import config_manager
            pid = str(config_manager.get_preference("default_ffchain_paper", fallback) or "")
        except Exception:
            return fallback
        if not pid or pid in ("none", "screen"):
            return fallback
        ffchain = set(getattr(self.engine, "FFCHAIN_PAPERS", ()) or ())
        return pid if (not ffchain or pid in ffchain) else fallback

    def _default_paper_for(self, film_stock=None):
        """这张卷该落到哪张默认纸。所有落点入口都走这一个函数。

        以前每个入口各自读一遍 `default_negative_paper` 再各自兜底（其中一处还硬编码
        Ektacolor Edge、另一处 `or paper` 把越界值原样留下），于是同一个"默认"在原生链和
        转换卷上落成不同的纸，面板和出图还会分叉。链条归属只在这里判一次。

        `film=none`（旁通/刚刚重置）返回原生链的默认纸：相纸是"下一次选卷要继承"的面板
        状态，不能因为旁通就写成直扫。
        """
        film = getattr(self, "current_film_stock", "none") if film_stock is None else film_stock
        if self._paper_scope_key(film) == "ffchain":
            return self._default_ffchain_paper()
        return self._default_negative_paper()

    def _paper_scope_key(self, film_stock=None):
        """`'ffchain'` 或 `'native'` —— `_default_paper_for` 只分这两叉。"""
        film = getattr(self, "current_film_stock", "none") if film_stock is None else film_stock
        try:
            scope = self.engine.paper_scope(film)
        except Exception:                              # noqa: BLE001 - 判不出来按原生
            scope = "native"
        return "ffchain" if scope == "ffchain" else "native"

    def _default_paper_pair(self):
        """两条链的默认纸各取一次，供"一轮里要问很多张卡"的调用方复用。

        `_default_negative_paper()` 以前每次都枚举整个 `profiles/` 来核一个 id（122 ms），
        卡片墙按卡片逐个问就成了 53 × 122 ms —— 见 `_custom_paper_stocks` 的缓存与
        `_update_card_live_thumbnails` 的用法。
        """
        return {"native": self._default_negative_paper(),
                "ffchain": self._default_ffchain_paper()}

    def _film_scope(self, film):
        """这支卷属于哪条链：`native`（spektrafilm 单表）、`ffchain`（fotufilm 点式链），
        或 `None`（旁通 / 反转 —— 引擎判 inert，根本不过纸）。

        旁通与反转返回 None 是有意的：它们的相纸只是"下面那张负片"的携带状态，
        跨过去再回来不该改写那份记忆，也不该被另一条链的记忆顶掉。
        """
        try:
            scope = self.engine.paper_scope(film)
        except Exception:                              # noqa: BLE001 - 判不出来按原生
            return "native"
        return scope if scope in ("native", "ffchain") else None

    @staticmethod
    def _paper_by_scope(photo):
        """照片旁挂的另一条链的相纸选择（`{"native": id, "ffchain": id}`）。"""
        by = photo.get("paper_by_scope") if photo else None
        return dict(by) if isinstance(by, dict) else {}

    def _set_paper_by_scope(self, photo, scope, paper):
        """把一次显式选择记到 `scope` 这条链上。

        `none`（胶片直扫）也要记：那是用户在这条链上的选择，切回来必须还是直扫。
        空串表示"这条链上没有显式选择"，清掉。
        """
        if not photo or not scope:
            return
        by = self._paper_by_scope(photo)
        paper = str(paper or "")
        if paper:
            by[scope] = paper
        else:
            by.pop(scope, None)
        if by:
            photo["paper_by_scope"] = by
        else:
            photo.pop("paper_by_scope", None)

    @staticmethod
    def _photo_paper_is_explicit(photo):
        """这张照片的相纸是用户主动选的，还是只是"没记录"。

        ``paper_explicit`` 是内存里的标记，由 on_switch_paper 这类用户动作写入；从 .sdc
        恢复的照片没有该键，此时看 paper_profile 是否为空 —— .sdc 会把"没选过"写成空串，
        所以空串 / "none" / 缺键都算没选过，这类照片跟随首选项里的默认相纸。
        """
        if not photo:
            return False
        flag = photo.get("paper_explicit")
        if flag is not None:
            return bool(flag)
        paper = photo.get("paper_profile")
        return bool(paper) and str(paper) not in ("none", "")

    def _effective_paper_for_photo(self, film, paper):
        """照片没有相纸记录时该用哪张纸。

        只有"负片 + 相纸未记录"才落到默认相纸；旁通（film=none）、反转片（引擎固定介质）
        和用户已经选过的相纸都原样返回，绝不覆盖。UI 的相纸下拉本身没有"无"选项，所以
        负片的 paper=none 只可能来自旧数据/直扫遗留，不是用户的选择。
        """
        film = str(film or "none")
        paper = str(paper or "none")
        if film == "none" or paper not in ("", "none"):
            return paper
        try:
            if self.engine.is_positive_film(film):
                return paper
        except Exception:
            pass
        return self._default_paper_for(film)

    def on_default_negative_paper_changed(self):
        """首选项换了「原生链默认相纸」：只重刷这条链上没选过相纸的负片。"""
        self._refresh_default_paper_thumbnails("native")

    def on_default_ffchain_paper_changed(self):
        """首选项换了「转换卷默认相纸」：只重刷转换卷上没选过相纸的负片。"""
        self._refresh_default_paper_thumbnails("ffchain")

    def _refresh_default_paper_thumbnails(self, scope=None):
        """默认相纸换了：重刷这条链上"没选过相纸"的负片，选过的原样保留。

        每张缩略图要烘一次 17³ LUT（几十毫秒），几十张就是几秒，所以排队交给定时器分批
        处理，保存完首选项立刻能返回。当前照片同步重贴 LUT，画布与缩略图保持一致。

        `scope` 是这条链（`native` / `ffchain`）。以前这里不分链，一次改默认纸会把两条链
        的照片全刷一遍，而且用的是同一张纸 id —— 转换卷那边只能靠烘焙时的 clamp 兜，
        烘出来的其实是另一张纸（卡片文字和画面不是一回事）。
        """
        queue = []
        for photo in getattr(self, "photos", []):
            if not photo or self._photo_paper_is_explicit(photo):
                continue
            film = photo.get("film_profile", "none")
            if film in ("none", "", None):
                continue
            try:
                if self.engine.is_positive_film(film):
                    continue
            except Exception:
                pass
            if scope is not None:
                try:
                    if self.engine.paper_scope(film) != scope:
                        continue
                except Exception:
                    pass
            queue.append(photo)

        self._default_paper_queue = queue
        self._default_paper_scope = scope
        self._default_paper_refreshed = 0
        if not hasattr(self, "_default_paper_timer"):
            self._default_paper_timer = QTimer(self)
            self._default_paper_timer.setInterval(0)
            self._default_paper_timer.timeout.connect(self._drain_default_paper_queue)
        if queue and not self._default_paper_timer.isActive():
            self._default_paper_timer.start()

        active = next((p for p in getattr(self, "photos", [])
                       if p and p.get("id") == self.active_photo_id), None)
        if (active is not None and not self._photo_paper_is_explicit(active)
                and self.current_film_stock not in ("none", "", None)):
            try:
                is_pos = self.engine.is_positive_film(self.current_film_stock)
            except Exception:
                is_pos = False
            if not is_pos and self._paper_allowed_now(
                    self._default_paper_for(self.current_film_stock)):
                self.on_switch_paper(self._default_paper_for(self.current_film_stock),
                                     push_undo=False)

        # 左侧胶卷卡片是按"默认相纸"配对的，换纸后必须重排一次；否则要等切换分类/
        # 换底片才刷新（用户看到的正是"设置默认相纸后缩略图还是橙色负片"）。
        if hasattr(self, "_cards_debounce_timer"):
            self._cards_debounce_timer.start(120)
        else:
            self._refresh_card_previews(film_only=True)

    def _drain_default_paper_queue(self):
        """分批重刷 _refresh_default_paper_thumbnails 排下的缩略图（每 tick 4 张）。"""
        queue = getattr(self, "_default_paper_queue", None) or []
        done = 0
        while queue and done < 4:
            photo = queue.pop(0)
            done += 1
            raw = self.thumb_controller.get_raw_thumbnail(photo)
            if raw is None or getattr(raw, "size", 0) == 0:
                continue
            # 每张照片按自己的链条取默认纸，不能拿"刚改的那个值"套到所有照片上。
            paper = self._default_paper_for(photo.get("film_profile", "none"))
            try:
                graded = self.thumb_controller.grade_thumbnail_fast(
                    raw, photo.get("film_profile", "none"), paper, photo.get("params") or {})
                self.thumb_controller.update_graded_thumbnail(photo, graded, self.filmstrip)
                self._default_paper_refreshed = getattr(self, "_default_paper_refreshed", 0) + 1
            except Exception:
                continue
        if not queue:
            self._default_paper_timer.stop()
            n = getattr(self, "_default_paper_refreshed", 0)
            if n:
                self.set_app_status(f"默认相纸已更新，{n} 张底片已更新", timeout_ms=3000)

    def _photo_effective_paper(self, photo):
        """给某张照片算它实际该用的相纸（见 _effective_paper_for_photo + 显式标记）。

        显式选过的纸也要过一道链归属：`.sdc` 可能是另一支卷存的、撤销可能退回跨链状态，
        直接拿去烘时引擎的 `_clamp_paper` 会静默换成该链默认纸 —— 缩略图看着是 A 纸、
        面板写着 B 纸。夹在这里，缩略图 / 画布 / 导出用的是同一个 id。
        """
        if not photo:
            return "none"
        film = photo.get("film_profile") or "none"
        if self._photo_paper_is_explicit(photo):
            paper = str(photo.get("paper_profile") or "none")
        else:
            paper = self._effective_paper_for_photo(film, photo.get("paper_profile") or "none")
        if not self._paper_allowed_now(paper, film=film):
            paper = self._default_paper_for(film)
        return paper

    def on_switch_film(self, stock_id, push_undo=True):
        # Retired push stocks arrive here from old sessions and dropped folders: "Portra 800 Push 1"
        # is Portra 800 developed +1 now, so the id is rewritten before anything touches the LUT.
        from core.sdc_manager import LEGACY_PUSH_PROFILES
        if stock_id in LEGACY_PUSH_PROFILES:
            base, stops = LEGACY_PUSH_PROFILES[stock_id]
            self.current_params["push_ev"] = stops
            stock_id = base
        if hasattr(self, "_hover_timer"):
            self._hover_timer.stop()
        self._hover_pending_stock = None
        self._saved_film_stock = None
        self._saved_paper_stock = None
        self._saved_print_mode = None

        if push_undo and stock_id != self.current_film_stock:
            self._mark_current_photo_dirty()

        # 跨链切卷要保住用户在另一条链上选过的纸。`paper_profile` 永远存**当前链**的合法值，
        # 另一条链的那份旁挂在 `paper_by_scope` 里 —— 只用一个字段时切卷会把它覆写成新链的
        # 默认纸，切回去也回不来（实测：一次往返就永久丢掉用户的选择）。
        prev_film = getattr(self, "current_film_stock", "none")
        prev_paper = getattr(self, "current_paper_stock", "none")
        prev_scope = self._film_scope(prev_film)
        new_scope = self._film_scope(stock_id)
        scope_changed = bool(new_scope and prev_scope != new_scope)
        cur_photo = None
        restored_paper = None
        if self._current_photo_path:
            cur_photo = next((p for p in getattr(self, "photos", [])
                              if p and p.get("path") == self._current_photo_path), None)
        if cur_photo is not None and scope_changed:
            if prev_scope and self._photo_paper_is_explicit(cur_photo):
                self._set_paper_by_scope(cur_photo, prev_scope, prev_paper)
            restored_paper = self._paper_by_scope(cur_photo).get(new_scope)

        self.current_film_stock = stock_id
        self._update_push_ui_state()

        # Selecting "none" film (Bypass / original raw) locks paper to "none" (scan / bypass).
        is_pos = self.engine.is_positive_film(stock_id) if hasattr(self.engine, 'is_positive_film') else False
        # Bypass (no film medium at all) still collapses paper and print mode: there is nothing
        # left that could be printed.
        #
        # A reversal slide is *not* the same case.  The paper choice belongs to the negative the
        # user was working on, and wiping it here meant 负片 → 正片 → 负片 lost the paper and
        # asked for it a second time; the panel also showed 直扫 while the paper cards sat
        # behind it, which is the desync.  The engine already ignores paper for positives
        # (`eff_paper` collapses to `screen`, measured bit-identical LUT and grain table with
        # paper=none/scan), so the state is simply carried and the paper controls are greyed out
        # by _update_stock_mode_ui.  The enlarger parameters are carried for the same reason.
        if stock_id == "none":
            self.current_paper_stock = "none"
            self.current_print_mode = "scan"
            self.current_params["print_mode"] = "scan"
            self.current_params["enlarger_cyan"] = 0.0
            self.current_params["enlarger_magenta"] = 0.0
            self.current_params["enlarger_yellow"] = 0.0
            self.current_params["print_exposure"] = 1.0
            self.current_params["pre_flash"] = 0.0
            self.current_params["pre_flash_camera"] = 0.0
            self.current_params["developer_exhaustion"] = 0.0
            self.current_params["morph_gamma"] = 1.0
            for pid, card in self.paper_cards.items():
                card.set_active(pid == "none")
            self._update_all_sliders_from_params()
        elif not is_pos:
            # 负片选中时接上这条链的默认相纸（原生链 / 转换卷各一个）。原生链的默认值可能是
            # "胶片直扫"（none）——那就要把印相模式一起摆到 scan，否则会出现"纸=直扫、模式=印相"
            # 的不一致状态，面板显示的和实际烘出来的 LUT 不是一回事。
            #
            # 跨链往返时先看旁挂记忆：用户在这条链上选过就恢复它（包括主动选的直扫），
            # 没有才落默认纸。判断用 `is not None`，"none" 是合法记录。
            if restored_paper is not None and self._paper_allowed_now(restored_paper, film=stock_id):
                self.current_paper_stock = restored_paper
                restored_is_scan = str(restored_paper or "none") in ("none", "")
                self.current_print_mode = "scan" if restored_is_scan else "optical"
                self.current_params["print_mode"] = self.current_print_mode
                if hasattr(self, "combo_paper"):
                    self.combo_paper.set_value(restored_paper)
                for pid, card in self.paper_cards.items():
                    card.set_active(pid == restored_paper)
            elif getattr(self, "current_paper_stock", "none") in (None, "", "none"):
                default_paper = self._default_paper_for(stock_id)
                self.current_paper_stock = default_paper
                default_is_scan = str(default_paper or "none") in ("none", "")
                self.current_print_mode = "scan" if default_is_scan else "optical"
                self.current_params["print_mode"] = self.current_print_mode
                if hasattr(self, "combo_paper"):
                    self.combo_paper.set_value(default_paper)
                for pid, card in self.paper_cards.items():
                    card.set_active(pid == default_paper)
            # 换了链条（原生 ↔ fotufilm）时，手里那张纸可能不属于新链条：落到新链条的默认纸。
            # 必须在下面烘 LUT 之前做，否则这一帧会先用一张不支持的纸去烘焙（引擎会 clamp，
            # 于是面板写着 A 纸、画面出的是 B 纸）。
            self._ensure_paper_allowed(stock_id)

        for sid, card in self.film_cards.items():
            card.set_active(sid == stock_id)

        # Save to current photo
        if self._current_photo_path:
            photo = next((p for p in self.photos if p and p.get("path") == self._current_photo_path), None)
            if photo:
                photo["film_profile"] = stock_id
                if stock_id == "none":
                    photo["paper_profile"] = "none"
                    photo["print_mode"] = "scan"
                else:
                    # The paper (or whatever the user had) is part of the photo state, otherwise
                    # reopening it would fall back to 胶片直扫.  For a reversal slide this keeps
                    # the negative's paper choice intact for when the user switches back.
                    # 自动套上的默认纸不写实值：写作空串，这张照片仍算"没选过相纸"，
                    # 首选项换默认纸时会跟着变（.sdc 的空串就是"未选择"）。
                    #
                    # 跨链时"显式"由旁挂记忆决定：恢复了记忆 => 用户选的；落默认纸 => 自动值。
                    # 同链切卷沿用原标记。
                    explicit_here = (restored_paper is not None if scope_changed
                                     else self._photo_paper_is_explicit(photo))
                    photo["paper_explicit"] = bool(explicit_here)
                    photo["paper_profile"] = (
                        self.current_paper_stock if explicit_here else "")
                    photo["print_mode"] = self.current_print_mode
                if stock_id == "none":
                    photo["graded_thumbnail_rgb"] = None
                    self.thumb_controller.revert_photo_to_raw(photo, self.filmstrip)

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = self.current_print_mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self._apply_lut(lut_3d)
        self.canvas.update_params(paper_profile=self.current_paper_stock, print_mode=self.current_print_mode)
        hal_preset = self.engine.get_film_halation_preset(stock_id) if hasattr(self.engine, 'get_film_halation_preset') else None
        self.canvas.update_params(film_stock=stock_id, _halation_preset=hal_preset)
        # The LUT above was applied while canvas.params still held the previous paper/mode, so the
        # bundle staged alongside it belongs to that bake.  The preview now refuses a mismatched
        # bundle instead of painting it, but restaging here means the grain comes back on this
        # frame rather than on the next switch.  Only in 印相 mode: 直扫 has no print cube and
        # its density table is not interchangeable with the LUT, so scan keeps the LUT path.
        if self.current_print_mode == "optical":
            self.canvas.set_grain_bundle(self._grain_bundle_for_current())
        self._schedule_histogram_update()
        if self._current_photo_path:
            self._update_active_thumbnails()
            self._refresh_card_previews(paper_only=True)
        self._update_stock_mode_ui()
        if push_undo:
            self.push_undo_state()
            self._save_current_photo_sdc()
            film_desc = "无" if stock_id == "none" else f"{stock_id}"
            self.set_app_status(f"胶卷：{film_desc}", timeout_ms=2500)
            # 跨链换纸一定要说出来：这一整轮 bug 的共同形状都是"静默换纸"（面板 A、出图 B），
            # 而用户看不出缩略图用的是哪张纸。恢复记忆时不提示 —— 那是用户自己的选择，
            # 面板已经显示回来了。
            if (scope_changed and restored_paper is None
                    and str(prev_paper or "none") != str(self.current_paper_stock or "none")
                    and str(prev_paper or "none") not in ("", "none")):
                prev_label = FilmGridCard.CLEAN_NAMES.get(str(prev_paper), str(prev_paper))
                now_label = FilmGridCard.CLEAN_NAMES.get(str(self.current_paper_stock),
                                                         str(self.current_paper_stock))
                self.set_app_status(
                    f"{prev_label} 只用于另一条链，已切到 {now_label}；切回原来的卷会自动恢复",
                    timeout_ms=4500)

    def on_switch_print_mode(self, mode, push_undo=True):
        if mode not in ("optical", "scan"):
            mode = "optical"
        # 反转片只有"直接看片"一条路：相纸印相在物理上不存在（引擎里它的介质永远是
        # Digital Reference）。但"当前已经是 optical"要放行 —— 撤销/会话恢复会用当前模式
        # 重放这一路来重新贴 LUT，拦住它等于让反转片的画面停在上一张。
        if (mode == "optical" and mode != getattr(self, "current_print_mode", "optical")
                and hasattr(self.engine, "is_positive_film")
                and self.engine.is_positive_film(getattr(self, "current_film_stock", "none"))):
            self.set_app_status("反转片直接查看，不经过相纸印相", timeout_ms=2500)
            return
        if push_undo and mode != getattr(self, "current_print_mode", "optical"):
            self._mark_current_photo_dirty()

        self.current_print_mode = mode
        self.current_params["print_mode"] = mode

        # Save to current photo
        if push_undo and self._current_photo_path:
            photo = next((p for p in self.photos if p and p.get("path") == self._current_photo_path), None)
            if photo:
                photo["print_mode"] = mode
                if mode == "scan":
                    # 用户主动选了"胶片直扫"：这张片就不该再被默认相纸接管，标记成已选择。
                    photo["paper_explicit"] = True

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self._apply_lut(lut_3d)
        # This path never told the canvas which mode it is in, so 相纸印相 / 胶片直扫 could keep
        # painting the previous one.  Params go in after the bake (matching the paper and film
        # switches, where the bundle is chosen from the params still on screen), then the bundle
        # is restaged so the grain matches the frame instead of being dropped by the guard.
        self.canvas.update_params(print_mode=mode)
        if mode == "optical":
            self.canvas.set_grain_bundle(self._grain_bundle_for_current())
        self._schedule_histogram_update()
        if push_undo and self._current_photo_path:
            self._update_active_thumbnails()
            self._refresh_card_previews()
        self._update_stock_mode_ui()
        if push_undo:
            self.push_undo_state()
            self._save_current_photo_sdc()
            mode_name = "相纸印相" if mode == "optical" else "胶片直扫"
            self.set_app_status(f"印相模式：{mode_name}", timeout_ms=2500)

    def on_switch_paper(self, stock_id, push_undo=True):
        if hasattr(self, "_hover_timer"):
            self._hover_timer.stop()
        self._hover_pending_stock = None
        self._saved_film_stock = None
        self._saved_paper_stock = None
        self._saved_print_mode = None

        is_pos = self.engine.is_positive_film(self.current_film_stock) if hasattr(self.engine, 'is_positive_film') else False
        if is_pos and stock_id != "none":
            # 反转片不经过相纸，但"选中的是哪张纸"还是要显示出来（撤销/会话恢复会带着它
            # 走这条路）。只同步显示，不重烘 LUT —— 反转片的介质由引擎固定，与纸无关。
            self.current_paper_stock = stock_id
            for sid, card in getattr(self, "paper_cards", {}).items():
                card.set_active(sid == stock_id)
            if hasattr(self, "combo_paper"):
                self.combo_paper.set_value(stock_id)
            self.set_app_status("反转片不使用相纸，当前选择已保留给负片", timeout_ms=2500)
            return

        if stock_id not in ("", "none") and not self._paper_allowed_now(
                stock_id, self.paper_cards.get(stock_id)):
            # 链条不支持这张纸：不动状态，只解释。卡片是置灰的，走到这里通常是会话/快捷键
            # 或旧配置，硬切过去只会让烘焙回落到另一条链。
            self.set_app_status(f"{stock_id} 与当前胶卷不属同一条链，未切换", timeout_ms=2500)
            if hasattr(self, "combo_paper"):
                self.combo_paper.set_value(getattr(self, "current_paper_stock", "none"))
            # 卡片在点击时已经把高亮挪过去了（FilmGridCard.highlight_now，为了不等烘焙），
            # 这里既然拒绝了切换，得把高亮挪回真正的当前纸，否则高亮停在没生效的那张上。
            for pid, card in getattr(self, "paper_cards", {}).items():
                card.set_active(str(pid) == str(getattr(self, "current_paper_stock", "none")))
            return

        prev_paper = self.current_paper_stock
        if push_undo and stock_id != self.current_paper_stock:
            self._mark_current_photo_dirty()
        self.current_paper_stock = stock_id

        # The print mode follows the paper unconditionally.  It used to be updated only when
        # a user-driven switch came through `push_undo`, so the first pick made through
        # another path left the render in 胶片直扫 while the panel showed a paper -- which
        # reads exactly like "the first paper switch did nothing".
        self.current_print_mode = "scan" if stock_id == "none" else "optical"
        self.current_params["print_mode"] = self.current_print_mode

        for sid, card in self.paper_cards.items():
            card.set_active(sid == stock_id)
        if hasattr(self, "combo_paper"):
            self.combo_paper.set_value(stock_id)

        # Save to current photo
        if push_undo and self._current_photo_path:
            photo = next((p for p in self.photos if p and p.get("path") == self._current_photo_path), None)
            if photo:
                photo["paper_profile"] = stock_id
                photo["print_mode"] = self.current_print_mode
                # 记到当前链上：跨链切回来时靠这份记忆恢复（`paper_profile` 只存当前链的值，
                # 单靠它在往返时会把另一条链的选择擦掉）。
                scope_now = self._film_scope(self.current_film_stock)
                if scope_now:
                    self._set_paper_by_scope(photo, scope_now, stock_id)
                # 只有真的换过纸才算"用户选过"：这个标记让首选项改默认相纸时，已经选过
                # 相纸的底片不被覆盖，而没选过的继续跟随默认值。
                if stock_id != prev_paper:
                    photo["paper_explicit"] = True

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = self.current_print_mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self._apply_lut(lut_3d)
        self.canvas.update_params(paper_profile=stock_id, print_mode=self.current_print_mode)
        # Restage the grain for the bake that is actually on screen (see on_switch_film).
        if self.current_print_mode == "optical":
            self.canvas.set_grain_bundle(self._grain_bundle_for_current())
        self._schedule_histogram_update()
        if push_undo and self._current_photo_path:
            self._update_active_thumbnails()
            # 相纸换了，左侧胶卷卡片的预览必须重渲染；否则要等切分类标签才刷新
            self._refresh_card_previews(film_only=True)
        self._update_stock_mode_ui()
        if push_undo:
            self.push_undo_state()
            self._save_current_photo_sdc()
            mode_desc = "胶片直扫" if stock_id == "none" else f"{stock_id}"
            self.set_app_status(f"相纸：{mode_desc}", timeout_ms=2500)

    def _on_paper_combo_changed(self, paper_id):
        """右侧“相纸”下拉框：与旧卡片墙走同一条切换路径。"""
        if not paper_id:
            return
        if paper_id == getattr(self, "current_paper_stock", None):
            return
        self.on_switch_paper(paper_id)

    def set_view_mode(self, mode):
        self.canvas.set_view_mode(mode)
        if hasattr(self, 'btn_split'):
            self.btn_split.setChecked(mode != 0)
            if mode == 0:
                self.btn_split.setToolTip("对比视图（快捷键 \\）：点击切换分屏对比")
            elif mode == 1:
                self.btn_split.setToolTip("对比视图（快捷键 \\）：分屏对比中，再次点击切换为左右双屏")
            elif mode == 2:
                self.btn_split.setToolTip("对比视图（快捷键 \\）：左右双屏中，再次点击恢复编辑")
            elif mode == 3:
                self.btn_split.setToolTip("对比视图：正在查看原始图像（松开恢复）")

    def _on_compare_long_press_start(self):
        if not self._current_photo_path:
            return
        self._saved_view_mode_before_long_press = self.canvas.params.get("view_mode", 0)
        self.set_view_mode(3)
        self.set_app_status("原片对比（松开恢复）", timeout_ms=0)

    def _on_compare_long_press_end(self):
        if hasattr(self, "_saved_view_mode_before_long_press"):
            prev_mode = self._saved_view_mode_before_long_press
            delattr(self, "_saved_view_mode_before_long_press")
            self.set_view_mode(prev_mode)
            self.restore_default_status()

    # Resample kernel for the HQ frame is pinned to cubic (the Windows-photos-like one);
    # the option was scaffolding while the mode was being measured, not a user choice.
    _HQ_RESAMPLE = "cubic"

    def _apply_hq_preview_pref(self, *, announce=False):
        """Push 首选项 → 高质量预览 into the canvas.

        The switch used to be a checkable View-menu action; it now only lives in the
        preferences dialog, so both the startup path and the dialog's OK path read the
        pref instead of an action's checked state.  Off by default: the rebuild costs
        seconds at full resolution.
        """
        try:
            want = bool(config_manager.get_preference("hq_preview_export", False))
        except Exception:
            want = False
        if not hasattr(self, "canvas"):
            return want
        self.canvas.set_hq_preview(want, resample=self._hq_resample)
        if announce:
            if not want:
                self.set_app_status("高质量预览：关", timeout_ms=2200)
            elif self.canvas._hq_fit_view():
                self.set_app_status("高质量预览：开 · 按导出分辨率渲染后缩小", timeout_ms=2200)
            else:
                self.set_app_status("高质量预览：开 · 回到「适应窗口」才生效", timeout_ms=2600)
        return want

    def _apply_display_cm_pref(self):
        """Push 首选项 → 视口色彩管理 into the canvas（P0 显示端）。

        关掉时 canvas 会把 shader 尾段退回恒 sRGB —— 也就是接入 P0 之前的行为，
        留给「ACM 用户」和排查用。改动只影响上屏，导出永远走 sRGB。
        """
        try:
            want = bool(config_manager.get_preference("viewport_color_management", True))
        except Exception:
            want = True
        if hasattr(self, "canvas") and hasattr(self.canvas, "set_viewport_color_management"):
            self.canvas.set_viewport_color_management(want)
        return want

    # --- P1 校样 ------------------------------------------------------------------------
    def _sync_proof_actions(self):
        """把 canvas 的校样状态推回菜单勾选（不要触发 toggled 再走一圈）。"""
        try:
            st = self.canvas.proof_settings()
        except Exception:  # noqa: BLE001
            return
        for act, val in ((getattr(self, "act_proof", None), st.on),
                         (getattr(self, "act_proof_warn", None), st.warn)):
            if act is None:
                continue
            act.blockSignals(True)
            act.setChecked(bool(val))
            act.blockSignals(False)

    def _on_proof_toggled(self, on: bool):
        st = self.canvas.set_proof(on=bool(on))
        if on:
            self._sync_export_colorspace(st)
            self.set_app_status(f"校样：{st.canonical_target()}", timeout_ms=1500)
        else:
            self.set_app_status("校样已关闭", timeout_ms=1200)

    def _on_proof_warn_toggled(self, on: bool):
        self.canvas.set_proof(warn=bool(on))

    def open_proof_setup(self):
        from ui.proof_dialog import ProofSetupDialog

        st = self.canvas.proof_settings()
        dlg = ProofSetupDialog(st, parent=self)
        new = dlg.exec()
        if new is None:
            return
        self.canvas.set_proof(on=new.on, target=new.target, warn=new.warn)
        self._sync_proof_actions()
        self._sync_export_colorspace(new)

    def _sync_export_colorspace(self, settings):
        """校样 → 导出联动（P1 的「所见即所得」闭环）。

        只在**校样开着**时写回最近一次导出设置：用户按 Ctrl+Y 是想「按这个空间交付」，
        导出默认跟着走才不误导。校样关掉不动导出设置 —— 关校样只是停止模拟。
        """
        if not settings.on:
            return
        try:
            cfg = dict(config_manager.get_last_export_settings() or {})
            if cfg.get("colorspace") != settings.canonical_target():
                # save_* 是 merge 语义，只写这一个键，别把其它导出设置覆盖掉
                config_manager.save_last_export_settings(
                    {"colorspace": settings.canonical_target()})
        except Exception:  # noqa: BLE001
            pass

    def _sync_proof_from_export(self, colorspace):
        """导出 → 校样（双向联动的另一半）。

        校样开着 = 用户正在按某个空间交付；导出换了色彩空间就该同步，
        否则回到「预览一个样、文件另一个样」——这正是 P1 要消灭的问题。
        校样关掉时不动它（关校样只是停止模拟，不代表要改目标）。
        """
        if not colorspace or not hasattr(self, "canvas"):
            return
        try:
            st = self.canvas.proof_settings()
        except Exception:  # noqa: BLE001
            return
        if not st.on:
            return
        new = self.canvas.set_proof(target=colorspace)
        if new.canonical_target() != st.canonical_target():
            self._sync_proof_actions()
            self.set_app_status(f"校样跟随导出：{new.canonical_target()}", timeout_ms=1800)

    def toggle_split_view(self):
        if not self._is_photo_edited():
            self.set_app_status("原生未调色底片无前后对比", timeout_ms=1500)
            return
        curr_mode = self.canvas.params.get("view_mode", 0)
        if curr_mode == 3:
            curr_mode = getattr(self, "_saved_view_mode_before_long_press", 0)
        new_mode = (curr_mode + 1) % 3
        self.set_view_mode(new_mode)

    def _on_btn_fit_clicked(self):
        if not hasattr(self, 'canvas') or not getattr(self.canvas, '_has_image', False):
            if hasattr(self, 'btn_fit'):
                self.btn_fit.setChecked(False)
            return
        self.canvas.fit_to_view(animate=True)
        self._update_zoom_buttons()

    def _on_btn_100_clicked(self):
        if not hasattr(self, 'canvas') or not getattr(self.canvas, '_has_image', False):
            if hasattr(self, 'btn_100'):
                self.btn_100.setChecked(False)
            return
        self.canvas.set_zoom_100(animate=True)
        self._update_zoom_buttons()

    def _update_zoom_buttons(self):
        if not hasattr(self, 'btn_fit') or not hasattr(self, 'btn_100') or not hasattr(self, 'canvas'):
            return
        if not getattr(self.canvas, '_has_image', False):
            self.btn_fit.setChecked(False)
            self.btn_100.setChecked(False)
            return

        is_fit = abs(self.canvas.zoom - 1.0) < 0.005 and abs(self.canvas.pan_x) < 0.005 and abs(self.canvas.pan_y) < 0.005
        actual_ratio = self.canvas.get_actual_zoom_ratio()
        is_100 = abs(actual_ratio - 1.0) < 0.015

        if is_fit and is_100:
            is_100 = False

        self.btn_fit.setChecked(is_fit)
        self.btn_100.setChecked(is_100)

    def _on_zoom_changed(self, zoom_val):
        self._update_photo_info_display()
        self._update_zoom_buttons()

    def action_open_files(self):
        filt = FILE_DIALOG_IMAGE_FILTER
        last_d = config_manager.get_last_dialog_dir("photo")
        # Decouple native dialog parent from frameless HWND to prevent ghost window artifacts
        files, _ = QFileDialog.getOpenFileNames(None, "导入照片 / RAW 底片", last_d, filt)
        if files:
            config_manager.set_last_dialog_dir("photo", files[0])
            QTimer.singleShot(20, lambda fl=list(files): self._import_files_list(fl))

    def action_open_folder(self):
        last_d = config_manager.get_last_dialog_dir("photo")
        # Decouple native dialog parent from frameless HWND to prevent ghost window artifacts
        folder = QFileDialog.getExistingDirectory(None, "选择导入照片文件夹", last_d)
        if folder:
            config_manager.set_last_dialog_dir("photo", folder)
            QTimer.singleShot(20, lambda f=folder: self._scan_and_import_folder(f))

    def _scan_and_import_folder(self, folder):
        if not folder or not os.path.exists(folder):
            return
        files = []
        skipped_lossy = 0
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMPORTABLE_IMAGE_EXTENSIONS:
                    files.append(os.path.join(root, f))
                elif ext in LOSSY_IMAGE_EXTENSIONS:
                    skipped_lossy += 1
        if files:
            self._import_files_list(files)
            if skipped_lossy:
                self.set_app_status(
                    STATUS_LOSSY_SKIPPED.format(n=skipped_lossy),
                    timeout_ms=4000,
                )
        else:
            # Silent by policy: no dialog for rejected formats, only a status-bar note.
            self.set_app_status(
                f"所选文件夹内 {skipped_lossy} 个文件均为有损压缩（JPG/WebP），未导入" if skipped_lossy
                else "所选文件夹内未找到受支持的图像文件",
                timeout_ms=4000,
            )

    def open_single_file(self, file_path):
        if os.path.exists(file_path):
            self._import_files_list([file_path])

    def _seed_import_placeholders(self, paths, start_idx):
        """先把这一批的格子按顺序摆进底片栏，缩略图等解码回来原地填。

        worker 的解码是乱序完成的、发射必须保序才能保持文件顺序，只要有一张慢（实测偶发
        1.5 s）整条栏就僵在"只有一两张"，然后突然全冒出来。先摆格子以后，用户看到的是
        一排灰格逐个变亮 —— 顺序还天然是对的。
        """
        strip = getattr(self, "filmstrip", None)
        if strip is None or not paths:
            return
        entries = [{
            "id": f"photo_{start_idx + i}_{os.path.basename(p)}",
            "filename": os.path.basename(p),
            "path": p,
            "rating": 0,
        } for i, p in enumerate(paths)]
        try:
            strip.append_placeholders(entries)
        except Exception as ex:
            logger.warning(f"Placeholder seeding failed: {ex}")
            return
        # 解析内嵌 JPEG 要 15-30 ms，别和"摆 40 个格子"叠在同一帧里 —— 挪到下一轮事件循环，
        # 用户看到的是格子先排好、紧接着第一格亮起来。
        try:
            QTimer.singleShot(0, lambda s=strip, e=entries[0]: self._prime_first_placeholder(s, e))
        except Exception:
            self._prime_first_placeholder(strip, entries[0])

    def _prime_first_placeholder(self, strip, entry):
        """第一格先挂上内嵌 JPEG（在后台抓，主线程只做一次换图）。

        worker 里第一张走的是"全分辨率优先"（要给画布用），实测 1.5-3 s 才回来，而这一格
        正好是用户第一眼盯的位置 —— 前面几秒就它一个是空的，看起来就是"第一张坏了"。
        """
        if not entry or not entry.get("path"):
            return
        pid = entry.get("id")
        if not pid:
            return
        try:
            if getattr(self, "_thumb_prime_worker", None) is not None:
                old = self._thumb_prime_worker
                self._thumb_prime_worker = None
                try:
                    old.primed.disconnect()
                except Exception:
                    pass
                if old.isRunning():
                    old.stop()
                    return          # 上一张还在抓，让位，别开两条 rawpy
            w = ThumbPrimeWorker(pid, entry["path"], parent=self)
            w.primed.connect(self._on_prime_thumbnail)
            self._thumb_prime_worker = w
            w.start()
        except Exception as ex:
            logger.debug(f"Prime first placeholder failed: {ex}")

    def _on_prime_thumbnail(self, pid, rgb):
        strip = getattr(self, "filmstrip", None)
        if strip is None or rgb is None:
            return
        try:
            strip.fill_placeholder(pid, {
                "id": pid,
                "thumbnail_rgb": rgb,
                "raw_thumbnail_rgb": rgb,
            })
        except Exception as ex:
            logger.debug(f"Prime fill failed: {ex}")

    def _seed_restore_placeholders(self, files, active_path=None):
        """恢复会话也先把格子按顺序摆出来（id 规则与 SessionRestoreWorker 一致）。"""
        strip = getattr(self, "filmstrip", None)
        if strip is None or not files:
            return
        entries = [{
            "id": f"photo_{i}_{os.path.basename(p)}",
            "filename": os.path.basename(p),
            "path": p,
            "rating": 0,
        } for i, p in enumerate(files)]
        try:
            strip.append_placeholders(entries)
        except Exception as ex:
            logger.warning(f"Restore placeholder seeding failed: {ex}")
            return
        # 恢复时活动那张排在最前面（或用户指定的那张），同样先挂内嵌 JPEG。
        first = next((e for e in entries if e.get("path") == active_path), entries[0] if entries else None)
        if first is not None:
            try:
                QTimer.singleShot(0, lambda s=strip, e=first: self._prime_first_placeholder(s, e))
            except Exception:
                self._prime_first_placeholder(strip, first)
        # 恢复期间不许自动滚动：光标会定在用户上次停留的那张，一旦自动居中，用户看到的
        # 就是"底片栏从中间那格开始"，左边一片空白。等他自己点/滚，或者恢复完再说。
        try:
            strip.set_autoscroll_enabled(False)
            QTimer.singleShot(0, strip.scroll_to_start)
        except Exception:
            pass

    def _enqueue_restore_photo(self, idx, entry):
        self._queue_ui_work(functools.partial(self._on_restore_photo_ready, idx, entry))

    def _on_restore_photo_ready(self, idx, entry):
        """worker 每解完一张就填一格 —— 恢复期间底片栏是一张张长出来的。"""
        if 0 <= idx < len(self.photos):
            self.photos[idx] = entry
        pid = (entry or {}).get("id")
        strip = getattr(self, "filmstrip", None)
        if not pid or strip is None:
            return
        if not strip.fill_placeholder(pid, entry):
            strip.append_photos_batch([entry])

    def _import_files_list(self, file_paths):
        if hasattr(self, "_pending_startup_restore_timer") and self._pending_startup_restore_timer is not None:
            self._pending_startup_restore_timer.stop()

        # 0. Lossless-only gate, applied once for every entry point (dialog, drag & drop,
        # folder scan, CLI args, recent files). Lossy containers are dropped silently and
        # reported only in the status bar -- no dialog, no per-file prompt.
        # Session restore does NOT pass through here (it resolves paths by existence), so
        # old .sdss projects that still reference JPEG files keep loading untouched.
        if file_paths:
            importable = []
            dropped_lossy = 0
            for p in file_paths:
                if is_importable_image(p):
                    importable.append(p)
                elif is_lossy_image(p):
                    dropped_lossy += 1
            if dropped_lossy:
                self.set_app_status(
                    STATUS_LOSSY_SKIPPED.format(n=dropped_lossy),
                    timeout_ms=4000,
                )
            file_paths = importable
            if not file_paths:
                return

        # 1. Three-tier deduplication:
        # Tier 1: Photos already loaded in library
        existing_norm_paths = {os.path.normcase(os.path.abspath(p["path"])) for p in self.photos if p and "path" in p}
        # Tier 2: Photos currently being decoded in-flight by the running worker
        in_flight_norm_paths = {os.path.normcase(os.path.abspath(p)) for p in getattr(self, "_current_import_batch", [])}
        # Tier 3: Photos currently waiting in the pending FIFO import queue
        queued_norm_paths = {os.path.normcase(os.path.abspath(p)) for p in getattr(self, "_pending_import_queue", [])}

        all_known_paths = existing_norm_paths | in_flight_norm_paths | queued_norm_paths

        new_paths = []
        dup_names = []
        seen_in_batch = set()

        for fp in file_paths:
            norm_p = os.path.normcase(os.path.abspath(fp))
            if norm_p in all_known_paths or norm_p in seen_in_batch:
                dup_names.append(os.path.basename(fp))
            else:
                seen_in_batch.add(norm_p)
                new_paths.append(os.path.abspath(fp))

        # Duplicate detection dialog (non-blocking status notifications)
        if dup_names:
            if not new_paths:
                logger.info(f"All {len(dup_names)} selected files are already in library")
                self.set_app_status(f"所选 {len(dup_names)} 张底片已在底片库中，无需重复导入", timeout_ms=3000)
                return
            else:
                logger.info(f"Skipping {len(dup_names)} duplicate files already in library")
                self.set_app_status(f"已跳过 {len(dup_names)} 张重复底片，正在导入新底片…", timeout_ms=3000)

        if not new_paths:
            return

        # Only mark as temporary session if there is NO existing named session being opened or active
        if not self._current_session_path and not getattr(self, "_pending_session_map", None):
            self._is_temp_session = True
            self._current_session_path = None

        # Only mark session dirty when actively importing new user photos, not restoring saved session
        if not getattr(self, "_pending_session_map", None):
            self._session_is_dirty = True
            self._update_window_title()

        # 1. Immediately switch to edit workspace
        self.stack.setCurrentIndexAnimated(1, duration_ms=250)
        self.btn_title_export.setVisible(True)

        # 2. Add to recent files（一次读写，别在导入循环里逐张存配置）
        config_manager.add_recent_files(new_paths)

        # 3. Ensure cancel button is visible
        self._set_cancel_chip(True, "导入")

        # 4. Check if an import or restore worker is actively running
        active_worker_running = (
            (getattr(self, "_import_worker", None) is not None and self._import_worker.isRunning()) or
            (getattr(self, "_slot_decode_worker", None) is not None and self._slot_decode_worker.isRunning()) or
            (getattr(self, "_session_restore_worker", None) is not None and self._session_restore_worker.isRunning())
        )

        if getattr(self, "_is_importing", False):
            # Enqueue new files into pending FIFO import queue
            if not hasattr(self, "_pending_import_queue") or self._pending_import_queue is None:
                self._pending_import_queue = []
            self._pending_import_queue.extend(new_paths)
            self._import_session_target_total = getattr(self, "_import_session_target_total", 0) + len(new_paths)

            cur = getattr(self, "_import_session_completed", 0)
            tot = self._import_session_target_total
            self.set_app_status(STATUS_IMPORTING.format(cur=cur, tot=tot), spinning=True, timeout_ms=0)
            logger.info(f"Import queue: appended {len(new_paths)} files, pending queue size: {len(self._pending_import_queue)}, target total: {tot}")

            if not active_worker_running:
                self._start_next_pending_import_batch()
            return

        # 5. No worker running: initialize and start fresh import session
        self._is_importing = True
        self._import_lock_time = time.time()
        self._pending_import_queue = []
        self._current_import_batch = list(new_paths)
        self._import_session_completed = 0
        self._import_session_target_total = len(new_paths)
        self._last_import_status_time = 0.0

        self.set_app_status(STATUS_IMPORTING.format(cur=0, tot=self._import_session_target_total), spinning=True, timeout_ms=0)

        # Asynchronous streaming import worker
        start_idx = max(getattr(self, "_photo_id_counter", 0), len(self.photos))
        self._photo_id_counter = start_idx + len(new_paths)

        self._seed_import_placeholders(new_paths, start_idx)
        self._import_filmstrip_buffer = []
        self._reset_import_delivery()
        self._set_import_hq_suspend(True)
        self._import_worker = BatchImportWorker(new_paths, self.engine, start_idx=start_idx, parent=self)
        self._wire_import_worker(self._import_worker)
        self._import_worker.start(QThread.Priority.LowPriority)

    def _start_next_pending_import_batch(self):
        """Dispatches the next queued batch of photos from _pending_import_queue to a new BatchImportWorker."""
        if not getattr(self, "_pending_import_queue", None):
            return False

        next_batch = list(self._pending_import_queue)
        self._pending_import_queue.clear()
        self._current_import_batch = next_batch

        start_idx = max(getattr(self, "_photo_id_counter", 0), len(self.photos))
        self._photo_id_counter = start_idx + len(next_batch)

        self._is_importing = True
        self._import_lock_time = time.time()
        self._set_cancel_chip(True, "导入")

        logger.info(f"Import queue: starting next batch with {len(next_batch)} files (start_idx={start_idx})")
        self._seed_import_placeholders(next_batch, start_idx)
        self._reset_import_delivery()
        self._set_import_hq_suspend(True)
        self._import_worker = BatchImportWorker(next_batch, self.engine, start_idx=start_idx, parent=self)
        self._wire_import_worker(self._import_worker)
        self._import_worker.start(QThread.Priority.LowPriority)
        return True

    # ------------------------------------------------------------------ 导入投放节流
    def _reset_import_delivery(self):
        """换批次/开始新导入时清空投放队列与收尾标记。"""
        if getattr(self, "_import_deliver_timer", None) is not None:
            self._import_deliver_timer.stop()
        self._import_deliver_queue = deque()
        self._deliver_finalizer = None
        self._deferred_switch_id = None

    def _wire_import_worker(self, worker):
        """worker 不再直连主线程槽，改为进队列由定时器按帧预算投放。

        worker 一秒能吐几十条，而单条最重可以是 700-1800 ms；直连时一次事件循环会把排队
        的信号连跑十几条，用户看到的就是"整段卡死、取消键也点不动"。
        """
        worker.photoLoaded.connect(self._enqueue_imported_photo)
        worker.importFinished.connect(self._on_import_stream_finished)

    def _queue_ui_work(self, fn):
        """把一件主线程工作排进投放队列：每个事件循环周期最多做 IMPORT_DRAIN_BUDGET_MS。

        队列里放的是**可调用对象**而不是"照片条目"，导入（photoLoaded）和会话恢复
        （slotDecoded）因此共用同一套节流，不必维护两份几乎一样的分帧逻辑。
        """
        q = getattr(self, "_import_deliver_queue", None)
        if q is None:
            q = deque()
            self._import_deliver_queue = q
        q.append(fn)
        if getattr(self, "_import_deliver_timer", None) is None:
            self._import_deliver_timer = QTimer(self)
            self._import_deliver_timer.setInterval(IMPORT_DRAIN_TICK_MS)
            self._import_deliver_timer.timeout.connect(self._drain_import_queue)
        if not self._import_deliver_timer.isActive():
            self._import_deliver_timer.start(IMPORT_DRAIN_TICK_MS)

    def _enqueue_imported_photo(self, photo_entry, current_idx, total_count):
        self._queue_ui_work(functools.partial(
            self._on_single_photo_imported, photo_entry, current_idx, total_count))

    def _enqueue_slot_decoded(self, slot_idx, decoded_entry):
        self._queue_ui_work(functools.partial(
            self._on_slot_photo_decoded, slot_idx, decoded_entry))

    def _finalize_after_delivery(self, fn):
        """让 worker 的"完成"信号也走队列：队列里还有货时不能先收尾。

        直连的 finished 会比最后几条结果早到 —— 导入会变成"已导入 N 张"却少几张，
        恢复会话会变成状态栏说"底片库已就绪"而底片条还在补图。
        """
        self._deliver_finalizer = fn
        self._drain_import_queue()

    def _on_import_stream_finished(self, total_loaded):
        self._finalize_after_delivery(functools.partial(
            self._on_all_photos_imported, int(total_loaded or 0)))

    def _on_slot_stream_finished(self):
        self._finalize_after_delivery(self._on_all_slots_decoded)

    def _drain_import_queue(self):
        q = getattr(self, "_import_deliver_queue", None)
        if q is None:
            q = deque()
            self._import_deliver_queue = q

        t0 = time.perf_counter()
        n = 0
        while q and n < IMPORT_DRAIN_MAX_ITEMS:
            item = q.popleft()
            n += 1
            try:
                item()
            except Exception as ex:
                logger.error(f"Deferred UI work failed: {ex}", exc_info=True)
            # 预算只在"整条做完"之后判：中途打断一条会把照片留在半初始化的状态。
            if (time.perf_counter() - t0) * 1000 >= IMPORT_DRAIN_BUDGET_MS:
                break

        if not q and getattr(self, "_deliver_finalizer", None) is not None:
            fn = self._deliver_finalizer
            self._deliver_finalizer = None
            if getattr(self, "_import_deliver_timer", None) is not None:
                self._import_deliver_timer.stop()
            fn()

    def _thumb_grade_worker(self):
        """惰性创建导入期的缩略图烘焙线程（一次 120-730 ms 的 LUT 现烤）。"""
        w = getattr(self, "_thumb_grade_worker_ref", None)
        if w is not None:
            return w
        tc = getattr(self, "thumb_controller", None)
        if tc is None:
            return None
        w = ThumbGradeWorker(tc, parent=self)
        w.thumbGraded.connect(self._on_thumb_graded)
        w.start(QThread.Priority.LowPriority)
        self._thumb_grade_worker_ref = w
        return w

    def _grade_thumb_or_defer(self, photo_entry, raw_th, film, paper, params):
        """导入期把烘焙丢给 worker，先返回原片缩略图（卡片"先粗糙"）。

        非导入路径（打开单张、手动重渲染）照旧同步跑，避免为了省一次 170 ms 把交互
        逻辑也改成异步。
        """
        worker = self._thumb_grade_worker() if getattr(self, "_is_importing", False) else None
        if worker is None:
            return self.thumb_controller.grade_thumbnail_fast(
                raw_th, film=film, paper=paper, params=params)
        pid = photo_entry.get("id")
        if not pid:
            return raw_th
        pending = getattr(self, "_pending_grade_ids", None)
        if pending is None:
            pending = set()
            self._pending_grade_ids = pending
        pending.add(pid)
        worker.enqueue(pid, raw_th, film, paper, params)
        return raw_th

    def _on_thumb_graded(self, photo_id, graded):
        pending = getattr(self, "_pending_grade_ids", None)
        if pending:
            pending.discard(photo_id)
        photo = next((p for p in self.photos if p and p.get("id") == photo_id), None)
        if photo is None or graded is None:
            return
        try:
            self.thumb_controller.update_graded_thumbnail(photo, graded, self.filmstrip)
        except Exception as ex:
            logger.warning(f"Applying deferred thumbnail failed for {photo_id}: {ex}")
            return
        # 导入已经收尾了才轮到它的：状态栏把"补齐缩略图"的尾巴交代清楚。
        if not getattr(self, "_is_importing", False):
            left = self._thumb_grade_worker().pending() if getattr(
                self, "_thumb_grade_worker_ref", None) is not None else 0
            if left <= 0:
                done = getattr(self, "_import_session_completed", len(self.photos))
                self.set_app_status(f"已导入 {done} 张底片", spinning=False, timeout_ms=3000)
            else:
                self.set_app_status(f"正在补齐缩略图（剩余 {left} 张）…", spinning=True, timeout_ms=0)

    def _should_defer_first_switch(self):
        """批量导入才延后首图，单张打开照旧（否则打开一张也要等导入"结束"才出图）。"""
        if not getattr(self, "_is_importing", False):
            return False
        if len(getattr(self, "_current_import_batch", None) or []) <= 1:
            return False
        canvas = getattr(self, "canvas", None)
        return canvas is not None and hasattr(canvas, "set_placeholder_image")

    def _show_import_placeholder(self, photo_entry):
        """导入期首图：直接铺相机内嵌 JPEG（sRGB bypass），完整链留到导入结束。

        取舍是用户拍板的"先粗糙后精细"：导入期间画布上是未调色的原片，导完再变成
        调色后的画面。换来的是导入一开始就能点按钮、能翻底片条。
        """
        pid = photo_entry.get("id")
        if not pid:
            return
        self.active_photo_id = pid
        self._current_photo_path = photo_entry.get("path")
        self._deferred_switch_id = pid
        # 参数面板里还是上一张的内容：这里绝不能报脏，否则切回首图时会把上一张的参数
        # 写进这张的 .sdc。
        self._photo_is_dirty = False
        exif = photo_entry.get("exif") or {}
        try:
            self.exif_widget.set_exif_data(exif)
            if hasattr(self, "histogram_exif_bar") and self.histogram_exif_bar:
                self.histogram_exif_bar.set_exif_data(exif)
        except Exception:
            pass
        th = photo_entry.get("raw_thumbnail_rgb")
        if th is None:
            th = photo_entry.get("thumbnail_rgb")
        try:
            self.canvas.set_has_library_photos(True)
            if th is not None:
                self.canvas.set_placeholder_image(
                    th,
                    master_width=int(photo_entry.get("width") or 0) or None,
                    master_height=int(photo_entry.get("height") or 0) or None,
                    photo_id=pid,
                )
        except Exception as ex:
            logger.warning(f"Import placeholder failed for {pid}: {ex}")

    def _activate_deferred_photo(self):
        """导入/取消结束后，把首图从内嵌 JPEG 换成真正调色后的画面。"""
        pid = getattr(self, "_deferred_switch_id", None)
        self._deferred_switch_id = None
        if not pid or pid != getattr(self, "active_photo_id", None):
            return  # 用户在导入期间已经自己点了别的照片
        if not any(p and p.get("id") == pid for p in self.photos):
            return
        strip = getattr(self, "filmstrip", None)
        prev_autoscroll = None
        if strip is not None:
            # 这一次切图不是为了"带用户去看"，别让它把底片栏从左边第一格拽走。
            try:
                prev_autoscroll = getattr(strip, "_autoscroll_enabled", True)
                strip.set_autoscroll_enabled(False)
            except Exception:
                prev_autoscroll = None
        try:
            self.on_switch_photo(pid, force=True)
        except Exception as ex:
            logger.error(f"Deferred first-photo switch failed: {ex}", exc_info=True)
        finally:
            if strip is not None and prev_autoscroll is not None:
                try:
                    strip.set_autoscroll_enabled(prev_autoscroll)
                except Exception:
                    pass
        # 完整链会 `invalidate_hq_preview()` 把 hold 清掉；这里再压一次，别让 2.2 s 的
        # 全分辨率读回紧跟着导入结束砸下来（用户这会儿正在翻底片条）。
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "rehold_hq_preview"):
            canvas.rehold_hq_preview()

    def _on_single_photo_imported(self, photo_entry, current_idx, total_count):
        # Defensive check: ensure photo_entry ID is strictly unique
        existing_ids = {p["id"] for p in self.photos if p and "id" in p}
        pid = photo_entry.get("id")
        # 这一格是不是我们自己刚摆的占位？是的话 id 必须保留 —— 改名会让
        # `fill_placeholder` 找不到格子，那张图就被追加到底片栏末尾，前面留一个黑格。
        seeded_placeholder = bool(
            pid and pid not in existing_ids
            and getattr(self, "filmstrip", None) is not None
            and self.filmstrip.has_photo(pid)
        )
        if (not pid or pid in existing_ids) and not seeded_placeholder:
            base_pid = pid or f"photo_{len(self.photos)}_{os.path.basename(photo_entry.get('path', ''))}"
            unique_pid = base_pid
            counter = 1
            while unique_pid in existing_ids:
                unique_pid = f"{base_pid}_{counter}"
                counter += 1
            photo_entry["id"] = unique_pid

        try:
            raw_th = self.thumb_controller.get_raw_thumbnail(photo_entry)
            if raw_th is not None:
                photo_entry["raw_thumbnail_rgb"] = raw_th
                curr_th = photo_entry.get("thumbnail_rgb")
                if curr_th is None or session_cache_manager.is_dummy_or_corrupted_thumbnail(curr_th):
                    photo_entry["thumbnail_rgb"] = raw_th
        except Exception as e:
            logger.warning(f"Error resolving thumbnail for imported photo: {e}")

        if hasattr(self, "_pending_session_map") and self._pending_session_map:
            from core.path_utils import normalize_path
            s_p = self._pending_session_map.get(normalize_path(photo_entry.get("path", "")))
            if s_p:
                photo_entry["film_profile"] = s_p.get("film_stock", "none")
                photo_entry["paper_profile"] = s_p.get("paper_stock", "none")
                photo_entry["print_mode"] = s_p.get("print_mode", "optical")
                photo_entry["base_ev"] = s_p.get("base_ev", photo_entry.get("base_ev", 0.0))
                photo_entry["params"] = dict(s_p.get("params") or {})
                is_dirty = bool(s_p.get("is_dirty", False))
                is_edited = bool(s_p.get("is_edited", False))
                photo_entry["is_dirty"] = is_dirty
                photo_entry["is_edited"] = is_edited
                if (is_edited or photo_entry["film_profile"] != "none") and raw_th is not None:
                    graded = self._grade_thumb_or_defer(
                        photo_entry, raw_th,
                        photo_entry["film_profile"],
                        self._photo_effective_paper(photo_entry),
                        photo_entry["params"]
                    )
                    photo_entry["graded_thumbnail_rgb"] = graded
                    photo_entry["thumbnail_rgb"] = graded
        else:
            # Check for existing .sdc sidecar metadata for newly imported photo
            try:
                from core import sdc_manager
                sdc = sdc_manager.load_sdc(photo_entry.get("path", ""))
                if sdc:
                    photo_entry["film_profile"] = sdc.get("film_profile", "none")
                    photo_entry["paper_profile"] = sdc.get("paper_profile", "none")
                    photo_entry["print_mode"] = sdc.get("print_mode", sdc.get("params", {}).get("print_mode", "optical"))
                    photo_entry["params"] = dict(sdc.get("params") or {})
                    if "rating" not in photo_entry:
                        photo_entry["rating"] = 0
                    if "crop" in photo_entry["params"]:
                        photo_entry["crop"] = dict(photo_entry["params"]["crop"])
                    photo_entry["is_dirty"] = True
                    photo_entry["is_edited"] = True
                    # 负片 + .sdc 没记相纸：内嵌缩略图那张是直扫橙片，跳过它按默认相纸重渲。
                    th_paper = (photo_entry["paper_profile"] or "none")
                    if not sdc.get("paper_explicit"):
                        th_paper = self._effective_paper_for_photo(
                            photo_entry["film_profile"], photo_entry["paper_profile"])
                    wants_regrade = (th_paper != (photo_entry["paper_profile"] or "none"))
                    if sdc.get("thumbnail_rgb") is not None and not wants_regrade:
                        photo_entry["graded_thumbnail_rgb"] = sdc["thumbnail_rgb"]
                        photo_entry["thumbnail_rgb"] = sdc["thumbnail_rgb"]
                    elif raw_th is not None:
                        graded = self._grade_thumb_or_defer(
                            photo_entry, raw_th,
                            photo_entry["film_profile"],
                            th_paper,
                            photo_entry["params"]
                        )
                        photo_entry["graded_thumbnail_rgb"] = graded
                        photo_entry["thumbnail_rgb"] = graded
            except Exception:
                pass

        self.photos.append(photo_entry)
        is_first = (len(self.photos) == 1) or (self.active_photo_id is None)

        if is_first:
            # Fast-lane for the very first photo: display it immediately on the canvas!
            self.filmstrip.append_photo(photo_entry, is_active=True)
            if photo_entry.get("is_dirty") or photo_entry.get("is_edited"):
                self.filmstrip.set_photo_dirty(photo_entry["id"], True)
            if self._should_defer_first_switch():
                # 批量导入期间先只铺内嵌 JPEG：完整链那一次要 1.4-1.9 s（解码 + 烘 LUT +
                # 上传纹理），正好砸在导入刚开始、用户最想点取消的那一刻。
                self._show_import_placeholder(photo_entry)
            else:
                self.on_switch_photo(photo_entry["id"])
        else:
            # Buffer for batch insertion to eliminate main-thread layout thrashing
            if not hasattr(self, "_import_filmstrip_buffer") or self._import_filmstrip_buffer is None:
                self._import_filmstrip_buffer = []
            self._import_filmstrip_buffer.append(photo_entry)
            import time as _t_mod
            now_fl = _t_mod.time()
            last_fl = getattr(self, "_last_filmstrip_flush_time", 0.0)
            if len(self._import_filmstrip_buffer) >= 16 or (now_fl - last_fl >= 0.06 and len(self._import_filmstrip_buffer) >= 4):
                self._last_filmstrip_flush_time = now_fl
                self._flush_import_filmstrip_buffer()
            else:
                if not hasattr(self, "_import_buffer_timer") or self._import_buffer_timer is None:
                    self._import_buffer_timer = QTimer(self)
                    self._import_buffer_timer.setSingleShot(True)
                    self._import_buffer_timer.timeout.connect(self._flush_import_filmstrip_buffer)
                if not self._import_buffer_timer.isActive():
                    self._import_buffer_timer.start(45)

        # Synchronize with engine session photos for safe base_ev lookups with O(1) map.
        # The signature guard matters: undo/redo and session restore rebuild
        # engine.session_photos in a different order, and a stale path -> index map then
        # wrote base_ev into another photo's slot.
        if hasattr(self, "engine") and hasattr(self.engine, "session_photos"):
            sp_sig = tuple(ep.get("path") for ep in self.engine.session_photos)
            if getattr(self, "_engine_photos_path_sig", None) != sp_sig:
                self._engine_photos_path_map = {
                    p: idx for idx, p in enumerate(sp_sig) if p
                }
                self._engine_photos_path_sig = sp_sig
            elif not hasattr(self, "_engine_photos_path_map") or self._engine_photos_path_map is None:
                self._engine_photos_path_map = {
                    p: idx for idx, p in enumerate(sp_sig) if p
                }
            ep_dict = {
                "id": photo_entry["id"],
                "path": photo_entry["path"],
                "filename": photo_entry["filename"],
                "width": photo_entry.get("width", 0),
                "height": photo_entry.get("height", 0),
                "thumbnail_rgb": photo_entry.get("thumbnail_rgb"),
                "raw_preview": None,
                "meta": dict(photo_entry.get("exif") or {}),
                "is_raw": photo_entry.get("is_raw", False),
                "base_ev": photo_entry.get("base_ev", 0.0),
                "auto_ev": photo_entry.get("auto_ev", 0.0)
            }
            p_path = photo_entry["path"]
            if p_path in self._engine_photos_path_map:
                e_idx = self._engine_photos_path_map[p_path]
                if e_idx < len(self.engine.session_photos):
                    self.engine.session_photos[e_idx] = ep_dict
            else:
                self._engine_photos_path_map[p_path] = len(self.engine.session_photos)
                self.engine.session_photos.append(ep_dict)
                self._engine_photos_path_sig = tuple(sp_sig) + (p_path,)

        # Strictly synchronized progress: increment session completed count
        self._import_session_completed = getattr(self, "_import_session_completed", 0) + 1
        cur = self._import_session_completed
        if not getattr(self, "_import_session_target_total", 0) and total_count:
            self._import_session_target_total = total_count
        tot = max(getattr(self, "_import_session_target_total", 0), cur)
        status_text = STATUS_IMPORTING.format(cur=cur, tot=tot)

        import time
        now = time.time()
        last_t = getattr(self, "_last_import_status_time", 0.0)

        if (now - last_t >= 0.08) or cur >= tot or cur == 1:
            self._last_import_status_time = now
            self.set_app_status(status_text, spinning=True, timeout_ms=0)
        else:
            # Trailing timer to ensure the status bar ALWAYS catches up and never freezes behind
            if not hasattr(self, "_import_flush_timer") or self._import_flush_timer is None:
                self._import_flush_timer = QTimer(self)
                self._import_flush_timer.setSingleShot(True)
                self._import_flush_timer.timeout.connect(self._flush_import_status)
            self._import_flush_timer.start(50)

    def _flush_import_status(self):
        if getattr(self, "_is_importing", False):
            cur = getattr(self, "_import_session_completed", len(self.photos))
            tot = max(getattr(self, "_import_session_target_total", cur), cur)
            self.set_app_status(STATUS_IMPORTING.format(cur=cur, tot=tot), spinning=True, timeout_ms=0)

    def _flush_import_filmstrip_buffer(self):
        buf = getattr(self, "_import_filmstrip_buffer", None)
        if not buf:
            return
        to_append = list(buf)
        self._import_filmstrip_buffer.clear()
        self.filmstrip.append_photos_batch(to_append)
        for p in to_append:
            if p.get("is_dirty") or p.get("is_edited"):
                self.filmstrip.set_photo_dirty(p["id"], True)

    def _sync_filmstrip_thumbnails(self, why=""):
        """收尾兜底：底片栏里还停在空窗口的格子，从 `self.photos` 再取一次图。

        为什么需要：占位格只有"收到的条目里带缩略图"才会亮，而首图常常是**未缓存**的那张
        （要等后台解码），只要有一条路没把图补回去，那格就永远黑着 —— 用户看到的就是
        "第一张缩略图坏了"。这里统一扫一遍，顺便记一条日志，方便下次直接看到是哪种情况。
        """
        strip = getattr(self, "filmstrip", None)
        if strip is None:
            return
        try:
            pending = strip.pending_thumbnail_ids()
        except Exception:
            return
        if not pending:
            return
        by_id = {p.get("id"): p for p in self.photos if isinstance(p, dict)}
        fixed = 0
        for pid in pending:
            entry = by_id.get(pid)
            if entry is None:
                continue
            try:
                if strip.fill_placeholder(pid, entry):
                    fixed += 1
            except Exception as ex:
                logger.debug(f"fill placeholder failed for {pid}: {ex}")
        left = strip.pending_thumbnail_ids()
        if left:
            logger.info(f"[filmstrip] {why or '收尾'}: 仍有 {len(left)} 格没有缩略图 "
                        f"（{', '.join(str(x) for x in left[:4])}）")
        elif fixed:
            logger.info(f"[filmstrip] {why or '收尾'}: 补回 {fixed} 格缩略图")

    def _on_all_photos_imported(self, total_loaded):
        if getattr(self, "_import_worker", None):
            try:
                self._import_worker.photoLoaded.disconnect()
                self._import_worker.importFinished.disconnect()
            except Exception:
                pass
            self._import_worker = None
        self._current_import_batch = []

        # If there are files queued in the FIFO pipeline, start next batch immediately
        if self._start_next_pending_import_batch():
            return

        # All batches have finished: finalize session
        if hasattr(self, "_import_flush_timer") and self._import_flush_timer is not None:
            self._import_flush_timer.stop()
        if hasattr(self, "_import_buffer_timer") and self._import_buffer_timer is not None:
            self._import_buffer_timer.stop()
        self._flush_import_filmstrip_buffer()
        self._is_importing = False
        self._current_import_batch = []
        self._pending_import_queue = []
        self._set_cancel_chip(False)
        self._pending_session_map = None
        self._sync_filmstrip_thumbnails("导入收尾")
        self.canvas.set_has_library_photos(bool(self.photos))
        self._update_recent_menu()
        target_path = getattr(self, "_target_initial_active_path", None)
        if target_path:
            self._target_initial_active_path = None
            target_entry = next((p for p in self.photos if p and p.get("path") == target_path), None)
            if target_entry:
                self.on_switch_photo(target_entry["id"])

        completed_count = getattr(self, "_import_session_completed", len(self.photos))
        self.set_app_status(f"已导入 {completed_count} 张底片", spinning=False, timeout_ms=3000)
        self._set_import_hq_suspend(False, hold=(completed_count > 1))

        # 首图还停在导入期铺的内嵌 JPEG 上，现在才跑完整链（一次 0.3-1.9 s）。
        # 留一拍让"已导入 N 张"和底片条先画出来，别让用户以为收尾时又卡住了。
        if getattr(self, "_deferred_switch_id", None):
            QTimer.singleShot(IMPORT_DEFERRED_SWITCH_DELAY_MS, self._activate_deferred_photo)

        # Real-time session state save & cache maintenance
        try:
            if self._current_session_path and os.path.exists(self._current_session_path):
                config_manager.save_last_session_path(self._current_session_path)
            elif getattr(self, "_is_temp_session", False) or not self._current_session_path:
                if self.photos:
                    self._save_auto_temp_session()
                    config_manager.save_last_session_path(config_manager.get_auto_saved_session_path())
        except Exception:
            pass
        # 缓存修剪要遍历整个缓存目录，一次 300-400 ms（实测 prune_cache_by_size 351 ms），
        # 放在导入完成的当口正好砸在用户开始翻照片的时候。推后跑，并在真正的导入/恢复
        # 期间继续顺延 —— 宁可晚一点修剪，也不要再添一次卡顿。
        QTimer.singleShot(5000, self._deferred_prune_cache)

    def _deferred_prune_cache(self):
        if getattr(self, "_is_importing", False):
            QTimer.singleShot(5000, self._deferred_prune_cache)
            return
        self._run_cache_maintenance("prune", session_cache_manager.prune_cache_if_needed)

    def _run_cache_maintenance(self, tag, fn):
        """缓存目录维护（遍历几千个文件 + stat，一次 300-500 ms）挪出主线程。

        这些操作只读目录、删的是可再生的缓存文件，失败一律容忍（下次缓存 miss 重算）。
        放在主线程上无论怎么排都会留下一次几百毫秒的冻结 —— 实测启动后 2.5 s 那次
        `check_and_auto_clean` 就占 487 ms，正撞在用户刚点完导入的当口。
        """
        if getattr(self, "_cache_maintenance_running", False):
            return
        self._cache_maintenance_running = True

        def _work():
            try:
                fn()
            except Exception as ex:
                logger.warning(f"Cache maintenance ({tag}) failed: {ex}")
            finally:
                self._cache_maintenance_running = False

        threading.Thread(target=_work, daemon=True, name=f"CacheMaint-{tag}").start()

    def _auto_clean_cache_soon(self):
        if getattr(self, "_is_importing", False):
            QTimer.singleShot(5000, self._auto_clean_cache_soon)
            return
        self._run_cache_maintenance("autoclean", session_cache_manager.check_and_auto_clean)

    def _set_import_hq_suspend(self, on, resume_delay=1200, hold=True):
        """导入 / 恢复会话期间挂起 HQ 预览重建。

        一次 HQ 重建要 2.2 s，其中 2.1 s 是 12 MP 的同步 glReadPixels —— 全在主线程上，
        期间事件循环不转，取消按钮点了也没反应。导入/恢复本来就不是看画质的时候，
        退回实时 shader 更合适。
        """
        canvas = getattr(self, "canvas", None)
        if canvas is None or not hasattr(canvas, "suspend_hq_preview"):
            return
        if on:
            if getattr(self, "_hq_import_suspended", False):
                return
            self._hq_import_suspended = True
            canvas.suspend_hq_preview()
        else:
            if not getattr(self, "_hq_import_suspended", False):
                return
            self._hq_import_suspended = False
            # 批量导入/恢复刚结束时**不**自动重建：那一帧要 0.8-2.4 s，正好砸在用户开始
            # 翻照片的时刻。等用户第一次操作（滚轮/平移/切图/改参数）再渲。
            # 单张打开没有"之后要翻什么"的情况，照常重建，否则 HQ 开了却一直不出帧。
            canvas.resume_hq_preview(delay=resume_delay, hold=hold)

    def _safely_terminate_worker(self, worker, wait_ms=1000):
        if not worker:
            return None
        try:
            if hasattr(worker, "stop"):
                worker.stop()
            elif hasattr(worker, "cancel"):
                worker.cancel()
            if hasattr(worker, "isRunning") and worker.isRunning():
                worker.wait(wait_ms)
            if hasattr(worker, "isRunning") and worker.isRunning():
                if not hasattr(self, "_retained_zombie_workers"):
                    self._retained_zombie_workers = []
                self._retained_zombie_workers.append(worker)
        except Exception:
            pass
        return None

    def _set_cancel_chip(self, visible, what="导入"):
        """状态栏那颗"取消"胶囊：导入和恢复会话共用，文案跟着阶段走。

        恢复会话也要能中止（几秒到一分钟），所以两条路都显示它。可点性靠
        StatusSlotWidget._apply_slot_hit_test 保证 —— 光 setVisible 是不够的，
        被隐藏层挡住时真实鼠标根本到不了。
        """
        btn = getattr(self, "btn_cancel_import", None)
        if btn is None:
            return
        btn.setText(f"✕ 取消{what}")
        btn.setToolTip(f"中断当前{what}（Esc）")
        btn.setVisible(bool(visible))

    def _cancel_active_import(self, notify_user=True):
        """Forcefully cancels active batch import worker or slot decode worker, freeing CPU and memory."""
        was_importing = getattr(self, "_is_importing", False) or bool(getattr(self, "_pending_import_queue", None))
        # 1. Clear pending queues immediately so no more batches trigger
        self._pending_import_queue = []
        self._current_import_batch = []
        self._import_session_completed = 0
        self._import_session_target_total = 0

        if hasattr(self, "_import_flush_timer") and self._import_flush_timer is not None:
            self._import_flush_timer.stop()
        if hasattr(self, "_import_buffer_timer") and self._import_buffer_timer is not None:
            self._import_buffer_timer.stop()
        # 投放队列里还没跑的条目一并丢掉，否则取消之后它们还会继续往底片库里塞
        if getattr(self, "_import_deliver_timer", None) is not None:
            self._import_deliver_timer.stop()
        if getattr(self, "_import_deliver_queue", None) is not None:
            self._import_deliver_queue.clear()
        self._deliver_finalizer = None
        self._flush_import_filmstrip_buffer()

        # 2. Terminate and disconnect batch import worker
        if getattr(self, "_import_worker", None):
            try:
                self._import_worker.photoLoaded.disconnect()
                self._import_worker.importFinished.disconnect()
            except Exception:
                pass
            self._safely_terminate_worker(self._import_worker, wait_ms=0)
            self._import_worker = None

        # 3. Terminate slot decode worker
        if getattr(self, "_slot_decode_worker", None):
            try:
                self._slot_decode_worker.slotDecoded.disconnect()
                self._slot_decode_worker.progress.disconnect()
                self._slot_decode_worker.finished.disconnect()
            except Exception:
                pass
            self._safely_terminate_worker(self._slot_decode_worker, wait_ms=0)
            self._slot_decode_worker = None

        # 4. Terminate session restore worker
        if getattr(self, "_session_restore_worker", None):
            try:
                self._session_restore_worker.activePhotoReady.disconnect()
                self._session_restore_worker.progress.disconnect()
                self._session_restore_worker.restoreFinished.disconnect()
                self._session_restore_worker.failed.disconnect()
            except Exception:
                pass
            self._safely_terminate_worker(self._session_restore_worker, wait_ms=0)
            self._session_restore_worker = None

        self._is_importing = False
        # 取消是用户主动打断的：HQ 晚一点再重建，别让他刚点完取消又卡 2 秒。
        self._set_import_hq_suspend(False, resume_delay=3000, hold=True)
        # 首图还停在导入期铺的内嵌 JPEG 上。取消后不能就这么留着 —— 画布上是"原片"，
        # 而参数面板里还是上一张的参数，拖滑块会拿低分辨率贴图当前景渲。
        # 也别在这儿补完整链：那一跑 300-500 ms，正好砸在用户刚点完取消的瞬间。
        # 直接退回空画布，用户点哪张就正常加载哪张。
        if getattr(self, "_deferred_switch_id", None):
            self._deferred_switch_id = None
            self.active_photo_id = None
            try:
                self.canvas.set_image(None)
            except Exception:
                pass
        self.canvas.set_has_library_photos(bool(self.photos))
        self._set_cancel_chip(False)
        self.restore_default_status()
        if notify_user and was_importing:
            self.set_app_status("已取消导入", timeout_ms=3000)

    def _on_slot_photo_decoded(self, slot_idx, decoded_entry):
        if slot_idx < 0 or slot_idx >= len(self.photos):
            return
        photo = self.photos[slot_idx]
        if photo is None:
            return
        photo["width"] = decoded_entry.get("width", 0)
        photo["height"] = decoded_entry.get("height", 0)
        photo["exif"] = decoded_entry.get("exif", {})
        photo["auto_ev"] = decoded_entry.get("auto_ev", 0.0)
        photo["is_raw"] = decoded_entry.get("is_raw", False)

        # Populate pristine raw thumbnail base
        raw_th = decoded_entry.get("raw_thumbnail_rgb") if decoded_entry.get("raw_thumbnail_rgb") is not None else decoded_entry.get("thumbnail_rgb")
        if raw_th is not None:
            photo["raw_thumbnail_rgb"] = np.copy(raw_th)
            photo["raw_thumbnail"] = np.copy(raw_th)

        # Preserve graded thumbnail if already provided by SDC, otherwise grade decoded thumbnail if photo has edits
        if photo.get("thumbnail_rgb") is None:
            p_params = photo.get("params") or {}
            th_film = photo.get("film_profile", "none")
            # 没选过相纸的负片用首选项的默认相纸出缩略图，否则这里是橙色负像。
            th_paper = self._photo_effective_paper(photo)
            if th_film != "none" or th_paper != "none" or abs(p_params.get("exposure_ev", 0.0)) > 0.001:
                # 和导入同一条路：恢复期间先把原片缩略图挂上去，LUT 现烤交给 worker
                # （每张 .sdc 参数不同 → 缓存全 miss，单张 120-730 ms 全压主线程）。
                graded = self._grade_thumb_or_defer(
                    photo,
                    raw_th,
                    th_film,
                    th_paper,
                    photo.get("params", {})
                )
                photo["graded_thumbnail_rgb"] = graded
                photo["thumbnail_rgb"] = graded
            else:
                photo["thumbnail_rgb"] = raw_th
                photo["graded_thumbnail_rgb"] = None

        if photo.get("thumbnail_rgb") is not None:
            trgb = photo["thumbnail_rgb"]
            h, w, c = trgb.shape
            qimg = QImage(trgb.data, w, h, w * c, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.filmstrip.update_photo_thumbnail(photo["id"], pix)

        self.filmstrip.update_photo_data(photo["id"], photo)

        # Synchronize with engine session photos for safe base_ev lookups
        if hasattr(self, "engine") and hasattr(self.engine, "session_photos"):
            matched_ep = next((ep for ep in self.engine.session_photos if ep.get("path") == photo["path"]), None)
            ep_dict = {
                "id": photo["id"],
                "path": photo["path"],
                "filename": photo["filename"],
                "width": photo.get("width", 0),
                "height": photo.get("height", 0),
                "thumbnail_rgb": photo.get("thumbnail_rgb"),
                "raw_preview": None,
                "meta": dict(photo.get("exif") or {}),
                "is_raw": photo.get("is_raw", False),
                "base_ev": photo.get("base_ev", 0.0),
                "auto_ev": photo.get("auto_ev", 0.0)
            }
            if matched_ep:
                self.engine.session_photos[self.engine.session_photos.index(matched_ep)] = ep_dict
            else:
                self.engine.session_photos.append(ep_dict)

        # 这一张刚解出来：底片栏那格换成真图（占位阶段它是灰的）。
        try:
            self.filmstrip.fill_placeholder(photo.get("id"), photo)
        except Exception:
            pass

        # If current active photo was uncached and just completed, refresh canvas display
        if self.active_photo_id == photo["id"]:
            if photo.get("float_img") is not None:
                self.canvas.set_image(photo["float_img"], master_width=photo.get("width"), master_height=photo.get("height"), photo_id=photo["id"])
                self._update_photo_info_display()
                if hasattr(self, 'exif_widget'):
                    self.exif_widget.set_exif_data(photo.get("exif") or {})
                if hasattr(self, 'histogram_exif_bar'):
                    self.histogram_exif_bar.set_exif_data(photo.get("exif") or {})

    def _on_slot_decode_progress(self, curr, total):
        if getattr(self, "_is_importing", False):
            self.set_app_status(f"正在恢复底片库（{curr}/{total}）…", spinning=True, timeout_ms=0)

    def _scroll_to_active_after_restore(self, delay_ms=300):
        """恢复/导入收尾后，把底片栏滚到当前那张。

        恢复期间故意不跟随（否则一打开就居中到上次停留的那张，左边全空），等**加载完**
        再滚过去。用户中途自己点过/滚过就不抢他的位置 —— 那时 `_autoscroll_enabled`
        已经被输入事件打开。
        """
        strip = getattr(self, "filmstrip", None)
        if strip is None:
            return

        def _go():
            pid = getattr(self, "active_photo_id", None)
            if not pid or getattr(self, "_is_importing", False):
                return
            if getattr(strip, "_autoscroll_enabled", True):
                return          # 用户自己动过手，别抢
            strip.set_autoscroll_enabled(True)
            try:
                strip.scroll_to_photo(pid, animate=True, force_center=True)
            except Exception as ex:
                logger.debug(f"Post-restore scroll failed: {ex}")

        QTimer.singleShot(int(delay_ms), _go)

    def _on_all_slots_decoded(self):
        if getattr(self, "_slot_decode_worker", None):
            try:
                self._slot_decode_worker.slotDecoded.disconnect()
                self._slot_decode_worker.progress.disconnect()
                self._slot_decode_worker.finished.disconnect()
            except Exception:
                pass
            self._slot_decode_worker = None

        if self._start_next_pending_import_batch():
            return

        self._is_importing = False
        self._set_import_hq_suspend(False, hold=(len(self.photos) > 1))
        self._set_cancel_chip(False)
        self._sync_filmstrip_thumbnails("恢复收尾")
        # 首图还停在占位缩略图上，把完整链补上（用户这会儿已经能随便点了）。
        if getattr(self, "_deferred_switch_id", None):
            QTimer.singleShot(IMPORT_DEFERRED_SWITCH_DELAY_MS, self._activate_deferred_photo)
        # 全部解完 = 这一轮加载结束，这时候才把底片栏滚到当前那张。
        self._scroll_to_active_after_restore(
            320 + (IMPORT_DEFERRED_SWITCH_DELAY_MS if getattr(self, "_deferred_switch_id", None) else 0))
        sess_name = os.path.basename(self._current_session_path) if getattr(self, "_current_session_path", None) else ""
        if sess_name:
            self.set_app_status(f"会话已恢复：{sess_name}（{len(self.photos)} 张）", spinning=False, timeout_ms=3000)
        else:
            self.set_app_status(f"底片库已就绪（{len(self.photos)} 张）", spinning=False, timeout_ms=2500)
        # 修剪要遍历整个缓存目录（实测 351-487 ms），别压在恢复完成的当口。
        self._run_cache_maintenance("prune", session_cache_manager.prune_cache_if_needed)

    def _sort_photos_by(self, sort_key):
        """Sort photos in filmstrip and session by criteria:
        'time_asc'/'exif_asc', 'time_desc'/'exif_desc', 'name_asc', 'name_desc', 'import'.
        Zero-blocking in-place widget reordering without undo/redo deep snapshots.
        """
        if not self.photos or len(self.photos) <= 1:
            return

        from datetime import datetime
        import re
        _dt_regex = re.compile(rb"((?:19|20)\d{2}:(?:0[1-9]|1[0-2]):(?:0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d)")

        def _parse_dt_str(dt_str):
            if not dt_str or dt_str == "-":
                return None
            cleaned = str(dt_str).strip()
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
                try:
                    return datetime.strptime(cleaned, fmt).timestamp()
                except Exception:
                    pass
            try:
                return float(cleaned)
            except Exception:
                return None

        def _get_photo_timestamp(photo):
            if not photo:
                return 0.0
            cached_ts = photo.get("_sort_timestamp")
            if cached_ts is not None:
                return float(cached_ts)

            exif = photo.get("exif") or {}
            dt_str = (
                exif.get("capture_time")
                or exif.get("DateTimeOriginal")
                or exif.get("DateTimeDigitized")
                or exif.get("DateTime")
                or photo.get("capture_time")
                or ""
            )

            ts = _parse_dt_str(dt_str)

            # Fast 64KB header scan for embedded EXIF DateTimeOriginal (no rawpy/exiftool overhead)
            fpath = photo.get("path")
            if ts is None and fpath:
                try:
                    with open(fpath, "rb") as _hfp:
                        _hdr = _hfp.read(65536)
                    _m = _dt_regex.search(_hdr)
                    if _m:
                        extracted_str = _m.group(1).decode("ascii", errors="ignore")
                        exif["capture_time"] = extracted_str
                        photo["exif"] = exif
                        ts = _parse_dt_str(extracted_str)
                except Exception:
                    pass

            # Fallback to cached mtime or filesystem modification time
            if ts is None:
                if exif.get("_mtime"):
                    ts = float(exif["_mtime"])
                elif fpath:
                    try:
                        ts = float(os.path.getmtime(fpath))
                    except Exception:
                        ts = 0.0
                else:
                    ts = 0.0

            photo["_sort_timestamp"] = float(ts)
            return float(ts)

        if sort_key in ("time_asc", "exif_asc"):
            self.photos.sort(key=_get_photo_timestamp)
            msg = "底片已按拍摄时间 (从早到晚) 排序"
        elif sort_key in ("time_desc", "exif_desc"):
            self.photos.sort(key=_get_photo_timestamp, reverse=True)
            msg = "底片已按拍摄时间 (从晚到早) 排序"
        elif sort_key == "name_asc":
            self.photos.sort(key=lambda p: (p.get("filename") or "").lower() if p else "")
            msg = "底片已按名称 (A 到 Z) 排序"
        elif sort_key == "name_desc":
            self.photos.sort(key=lambda p: (p.get("filename") or "").lower() if p else "", reverse=True)
            msg = "底片已按名称 (Z 到 A) 排序"
        elif sort_key == "import":
            def _get_id_idx(p):
                if not p:
                    return 0
                try:
                    parts = (p.get("id") or "").split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
                except Exception:
                    pass
                return 0
            self.photos.sort(key=_get_id_idx)
            msg = "底片已恢复初始导入顺序"
        else:
            return

        curr_active_id = self.active_photo_id
        self.filmstrip.reorder_photos(self.photos, curr_active_id)
        self._session_is_dirty = True
        self._update_window_title()
        self.set_app_status(msg, timeout_ms=3000)
        self._update_menu_actions_state()

    def _on_step_photo(self, delta):
        """Step previous (-1) or next (+1) photo in filmstrip with 0ms visual responsiveness."""
        if hasattr(self, 'filmstrip') and self.filmstrip:
            self.filmstrip.step_photo(delta)

    def _on_navigate_first_photo(self):
        """Navigate to the first photo in filmstrip, activate and select it (Home key)."""
        if not self.photos:
            return
        first_photo = self.photos[0]
        if first_photo and first_photo.get("id"):
            target_id = first_photo["id"]
            if hasattr(self, "filmstrip"):
                self.filmstrip.set_active_photo(target_id, reset_selection=True)
            if self.active_photo_id != target_id:
                self.on_switch_photo(target_id)
            if hasattr(self, "filmstrip"):
                self.filmstrip.scroll_to_start()
                self.filmstrip.scroll_to_active()

    def _on_navigate_last_photo(self):
        """Navigate to the last photo in filmstrip, activate and select it (End key)."""
        if not self.photos:
            return
        last_photo = self.photos[-1]
        if last_photo and last_photo.get("id"):
            target_id = last_photo["id"]
            if hasattr(self, "filmstrip"):
                self.filmstrip.set_active_photo(target_id, reset_selection=True)
            if self.active_photo_id != target_id:
                self.on_switch_photo(target_id)
            if hasattr(self, "filmstrip"):
                self.filmstrip.scroll_to_end()
                self.filmstrip.scroll_to_active()

    def _on_canvas_rating_changed(self, rating: int):
        """User clicked canvas status bar rating widget."""
        self.action_set_active_rating(rating)

    def _on_menu_filter_selected(self, filter_key: str):
        """User toggled a filter in the View menu."""
        if hasattr(self, 'filmstrip'):
            self.filmstrip.set_rating_filter(filter_key)

    def _on_filter_changed(self, filter_key: str):
        """Synchronize View menu checkmark state and counts when filmstrip filter changes."""
        if hasattr(self, 'star_filter_actions'):
            labels_map = {}
            if hasattr(self, "filmstrip"):
                counts = self.filmstrip.get_filter_counts()
                labels_map = {
                    "all": f"全部（{counts['all']}张）",
                    "5": f"★★★★★（{counts['5']}张）",
                    "4": f"★★★★（{counts['4']}张）",
                    "3": f"★★★（{counts['3']}张）",
                    "2": f"★★（{counts['2']}张）",
                    "1": f"★（{counts['1']}张）",
                    "0": f"无评分（{counts['0']}张）",
                }
            for k, act in self.star_filter_actions.items():
                if k in labels_map:
                    act.setText(labels_map[k])
                act.setChecked(k == filter_key)

    def _on_filmstrip_photo_rated(self, photo_id: str, rating: int):
        """Single photo rated directly in filmstrip item widget."""
        self._on_filmstrip_batch_rated([photo_id], rating)

    def _on_filmstrip_batch_rated(self, photo_ids: list, rating: int):
        """Batch rate one or multiple photos and persist to .sdc sidecars."""
        if not photo_ids or not self.photos:
            return
        clean_r = max(0, min(5, int(rating or 0)))
        target_set = set(photo_ids)

        old_ratings = {
            p["id"]: int(p.get("rating", 0) or 0)
            for p in self.photos
            if p and p.get("id") in target_set
        }
        if all(r == clean_r for r in old_ratings.values()):
            return

        rating_state = {
            "type": "photo_rating_state",
            "action": "rate_photos",
            "ratings": {pid: clean_r for pid in old_ratings},
            "old_ratings": old_ratings,
            "photo_id": self.active_photo_id,
            "film": getattr(self, "current_film_stock", "none"),
            "paper": getattr(self, "current_paper_stock", "none"),
            "print_mode": getattr(self, "current_print_mode", "optical"),
            "params": dict(self.current_params) if hasattr(self, "current_params") else {},
        }
        self._push_raw_state(rating_state)

        for p in self.photos:
            if p and p.get("id") in target_set:
                p["rating"] = clean_r
        if hasattr(self, 'filmstrip'):
            self.filmstrip.set_photos_rating(photo_ids, clean_r)
        if self.active_photo_id in target_set and hasattr(self, 'canvas_rating'):
            self.canvas_rating.set_rating(clean_r)

        self._session_is_dirty = True
        self._update_window_title()
        self._save_auto_temp_session()
        self._update_menu_actions_state()

        cnt = len(photo_ids)
        stars_str = "★" * clean_r + "☆" * (5 - clean_r)
        if clean_r > 0:
            msg = f"已评为 {stars_str}（{cnt} 张底片）" if cnt > 1 else f"已评为 {stars_str}"
        else:
            msg = f"已清除评级（{cnt} 张底片）" if cnt > 1 else "已清除评级"
        self.set_app_status(msg, timeout_ms=3000)

    def action_set_active_rating(self, rating: int):
        """Set rating for currently selected photo(s) or active photo via shortcut or UI."""
        if not self.photos:
            return
        clean_r = max(0, min(5, int(rating or 0)))
        target_ids = []
        if hasattr(self, 'filmstrip'):
            target_ids = self.filmstrip.get_selected_photo_ids()
        if not target_ids and self.active_photo_id:
            target_ids = [self.active_photo_id]
        if not target_ids:
            return
        self._on_filmstrip_batch_rated(target_ids, clean_r)

    def grade_thumbnail_fast(self, raw_thumb, film="none", paper="none", params=None):
        """Ultra-fast (2-4ms) CPU thumbnail grading applying exposure, highlights/shadows, and 3D LUT."""
        if hasattr(self, 'thumb_controller'):
            return self.thumb_controller.grade_thumbnail_fast(raw_thumb, film=film, paper=paper, params=params)
        return raw_thumb

    def _schedule_active_thumbnail_update(self, delay_ms: int = 150):
        """Debounces offscreen active thumbnail render so UI interaction stays 60 FPS silky smooth."""
        if not hasattr(self, "_active_thumb_timer"):
            self._active_thumb_timer = QTimer(self)
            self._active_thumb_timer.setSingleShot(True)
            self._active_thumb_timer.timeout.connect(self._update_active_thumbnails)
        self._active_thumb_timer.start(delay_ms)

    def _update_active_thumbnails(self):
        """Update active filmstrip thumbnail and active film/paper card thumbnails with WYSIWYG graded rendering,
        strictly maintaining the photo's genuine original aspect ratio (never stretching horizontal or vertical).
        """
        if not hasattr(self, 'canvas') or not getattr(self.canvas, '_has_image', False):
            return
        # Prevent rendering previous photo when switching to an un-decoded photo
        if getattr(self.canvas, '_loaded_photo_id', None) != self.active_photo_id:
            return
        try:
            # Prefer loaded GPU texture dimensions for exact rendering aspect ratio
            img_w = getattr(self.canvas, '_image_width', 0)
            img_h = getattr(self.canvas, '_image_height', 0)
            if img_w <= 1 or img_h <= 1:
                img_w = getattr(self.canvas, '_master_width', 0)
                img_h = getattr(self.canvas, '_master_height', 0)
            if img_w <= 1 or img_h <= 1:
                curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                if curr_p:
                    img_w = curr_p.get("width", 0)
                    img_h = curr_p.get("height", 0)
                    if not img_w or not img_h:
                        trgb = curr_p.get("thumbnail_rgb")
                        if trgb is not None and hasattr(trgb, "shape") and len(trgb.shape) >= 2:
                            img_h, img_w = trgb.shape[:2]

            img_w = max(1, int(img_w or 1))
            img_h = max(1, int(img_h or 1))

            # Determine target resolution maintaining exact photo or cropped aspect ratio without distortion
            crop_dict = self.current_params.get("crop", {})
            if isinstance(crop_dict, dict) and crop_dict.get("enabled", False):
                cw = max(0.01, float(crop_dict.get("w", 1.0))) * img_w
                ch = max(0.01, float(crop_dict.get("h", 1.0))) * img_h
                rot_90 = int(crop_dict.get("rot_90", 0)) % 360
                if rot_90 in (90, 270):
                    cw, ch = ch, cw
                asp = max(0.01, cw / ch)
            else:
                asp = max(0.01, float(img_w) / float(img_h))

            # Determine target resolution maintaining exact photo aspect ratio without any letterbox distortion:
            if asp >= 1.0:
                tgt_w = 120
                tgt_h = max(10, int(round(120.0 / asp)))
            else:
                tgt_h = 120
                tgt_w = max(10, int(round(120.0 * asp)))

            rendered_rgb = self.canvas.render_offscreen(target_w=tgt_w, target_h=tgt_h, bit_depth=8)
            if rendered_rgb is not None and rendered_rgb.size > 0:
                is_valid = bool(rendered_rgb.mean() > 1.0 or rendered_rgb.max() > 10)
                if not is_valid and not getattr(self.canvas, '_has_image', False):
                    return

                h, w = rendered_rgb.shape[:2]
                qimg = QImage(rendered_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
                pix = QPixmap.fromImage(qimg)

                # Bottom filmstrip active frame thumbnail
                if self.active_photo_id and is_valid:
                    curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                    if curr_p:
                        is_film = (self.current_film_stock not in ("none", "", None))
                        is_paper = (self.current_paper_stock not in ("none", "", None))
                        is_params = self._is_params_modified(self.current_params, curr_p)
                        has_recipe = is_film or is_paper or is_params or self._photo_is_dirty
                        if has_recipe:
                            self.thumb_controller.update_graded_thumbnail(curr_p, rendered_rgb, self.filmstrip)
                        else:
                            self.thumb_controller.revert_photo_to_raw(curr_p, self.filmstrip)
        except Exception as e:
            logger.debug(f"_update_active_thumbnails error: {e}")

    def _dir_available(self):
        try:
            return bool(self.engine.dir_status(self.current_film_stock)[0])
        except Exception:
            return True

    def _enlarger_status(self):
        """Returns (available: bool, reason: str) for the enlarger & paper development section."""
        film = getattr(self, "current_film_stock", "none")
        if film == "none":
            return False, "当前为数码原片旁通模式，无需放大与相纸显影"
        if hasattr(self.engine, "is_positive_film") and self.engine.is_positive_film(film):
            return False, "当前为反转正片，无放大机与相纸显影"
        paper = getattr(self, "current_paper_stock", "none")
        pmode = getattr(self, "current_print_mode", "optical")
        if paper == "none" or pmode == "scan":
            return False, "当前为胶片直扫（未反相），无放大机与相纸显影"
        return True, ""

    def _enlarger_available(self) -> bool:
        return self._enlarger_status()[0]

    def _update_chemistry_ui_state(self):
        """Grey out the DIR controls when the current stock carries no coupler chemistry and
        explain why on hover. DIR is baked into the LUT only, so without this the three sliders
        would simply look broken on reversal / B&W stocks (they are genuinely inert there)."""
        is_none = (getattr(self, "current_film_stock", "none") == "none")
        if is_none:
            available = False
            hint = "当前为数码原片旁通模式，无成色剂抑制化学"
        else:
            try:
                available, reason = self.engine.dir_status(getattr(self, "current_film_stock", "none"))
            except Exception:
                available, reason = True, ""
            hint = "" if available else self.DIR_UNAVAILABLE_TEXT.get(reason, "当前胶卷未提供 DIR 数据")
        for slider in (getattr(self, "slider_dir_amount", None),
                       getattr(self, "slider_dir_interlayer", None),
                       getattr(self, "slider_dir_samelayer", None)):
            if slider is None:
                continue
            slider.setEnabled(available)
            slider.setToolTip(getattr(slider, "default_tooltip", "") if available else hint)
            # Disabled Qt widgets do not reliably receive hover events, so let them fall through
            # to the card behind - the section's own tooltip then explains the state on hover.
            for w in [slider] + slider.findChildren(QWidget):
                w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not available)
        section = getattr(self, "sec_chemistry", None)
        if section is not None and hasattr(section, "set_available"):
            section.set_available(available, hint)
            if not available:
                section.set_reset_enabled(False)

    # -- 相纸的链条归属 ------------------------------------------------------------------------
    #
    # 转换卷（fotufilm 点式链）和原生卷（spektrafilm 单表）各有一组相纸，重叠只有四张。
    # 判定放在引擎里（`paper_available`），这里只负责把它翻译成界面状态。
    def _paper_allowed_now(self, paper_id, card=None, film=None):
        """这张纸在当前卷上能不能选。用户自己导入的相纸不参与链条判定。"""
        if card is not None and getattr(card, "is_custom", False):
            return True
        try:
            return self.engine.paper_available(
                film or getattr(self, "current_film_stock", "none"), paper_id)
        except Exception:                              # noqa: BLE001 - 判不出来就别禁用
            return True

    def _paper_unavailable_hint(self):
        scope = "native"
        try:
            scope = self.engine.paper_scope(getattr(self, "current_film_stock", "none"))
        except Exception:                              # noqa: BLE001
            pass
        if scope == "ffchain":
            return "这张相纸只用于原生链（spektrafilm）；当前是 fotufilm 转换卷，改用点式链相纸"
        return "这张相纸只有 fotufilm 点式链的表；当前是原生链胶卷，请改用原生相纸"

    def _refresh_paper_combo(self):
        """按当前卷重列相纸下拉框（只列能选的），保留仍在列表里的那个值。"""
        combo = getattr(self, "combo_paper", None)
        if combo is None or not hasattr(combo, "set_items"):
            return
        items = []
        for item in self.engine.get_paper_stocks():
            pid = item["id"]
            if pid in ("", "none"):
                items.append((item.get("name") or pid, pid))
                continue
            if self._paper_allowed_now(pid, self.paper_cards.get(pid)):
                items.append((item.get("name") or pid, pid))
        combo.set_items(items, keep_value=getattr(self, "current_paper_stock", "none"))

    def _ensure_paper_allowed(self, film=None):
        """当前相纸在这支卷上用不了时，落到该链的默认纸。返回是否改过。

        每个入口都要过这一道：切卷、激活照片、应用 .sdc、重置。`.sdc` 里可能存着一张
        当前卷不支持的纸，引擎虽然会 `_clamp_paper` 兜底，但面板会和出图对不上。
        """
        film = film or getattr(self, "current_film_stock", "none")
        paper = getattr(self, "current_paper_stock", "none")
        if self._paper_allowed_now(paper, film=film):
            return False
        landing = self._default_paper_for(film)
        if not landing or not self._paper_allowed_now(landing, film=film):
            # 两条链的默认值都在各自合法域内，这条分支理论上不可达；留一个最保守的落点。
            try:
                scope = self.engine.paper_scope(film)
            except Exception:                          # noqa: BLE001
                scope = "native"
            landing = "kodak_2383" if scope == "ffchain" else "kodak_ektacolor_edge"
        self.current_paper_stock = landing
        self.current_print_mode = (
            "scan" if str(landing or "none") in ("none", "") else "optical")
        self.current_params["print_mode"] = "optical"
        for pid, card in self.paper_cards.items():
            card.set_active(pid == landing)
        if hasattr(self, "combo_paper"):
            self.combo_paper.set_value(landing)
        if hasattr(self.canvas, "update_params"):
            self.canvas.update_params(paper_profile=landing, print_mode="optical")
        return True

    def _update_stock_mode_ui(self):
        """Update parameter availability based on film type (negative vs slide/reversal vs none/bypass) and print mode."""
        # 这个函数只看这三件事，但它会遍历上百个控件（还用 `findChildren` 逐个改
        # WA_TransparentForMouseEvents）。切一张照片走两次、实测 131-149 ms，而这三次
        # 值通常一模一样 —— 没变就整段跳过。
        _sig = (getattr(self, "current_film_stock", "none"),
                getattr(self, "current_paper_stock", "none"),
                getattr(self, "current_print_mode", "optical"))
        if _sig == getattr(self, "_stock_mode_ui_sig", None):
            return
        self._stock_mode_ui_sig = _sig
        is_none = (getattr(self, "current_film_stock", "none") == "none")
        is_pos = self.engine.is_positive_film(self.current_film_stock) if hasattr(self.engine, 'is_positive_film') else False

        if hasattr(self, 'sidebar_content') and self.sidebar_content is not None:
            self.sidebar_content.setEnabled(True)
            self.sidebar_content.setToolTip("")

        # 1. 场景曝光 (Exposure) - Always available
        sec_exposure = getattr(self, "sec_exposure", None)
        if sec_exposure is not None and hasattr(sec_exposure, "set_available"):
            sec_exposure.set_available(True, "")
        exposure_widgets = [
            getattr(self, 'slider_ev', None),
            getattr(self, 'slider_highlights', None),
            getattr(self, 'slider_shadows', None),
            getattr(self, 'slider_black_lift', None),
            getattr(self, 'slider_temp', None),
            getattr(self, 'slider_tint', None),
        ]
        for w in exposure_widgets:
            if w is not None:
                w.setEnabled(True)
                w.setToolTip(getattr(w, "default_tooltip", ""))
                for child in [w] + w.findChildren(QWidget):
                    child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # 画幅规格: 只决定银盐晶体与弥散圈的物理相对尺度，旁通态（无胶片）下无意义
        combo_format = getattr(self, "combo_format", None)
        if combo_format is not None:
            fmt_available = not is_none
            fmt_hint = "当前处于数码原片旁通模式，无胶片画幅与银盐晶体尺度" if is_none else ""
            combo_format.setEnabled(fmt_available)
            combo_format.setToolTip(getattr(combo_format, "default_tooltip", "") if fmt_available else fmt_hint)
            for child in [combo_format] + combo_format.findChildren(QWidget):
                child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not fmt_available)

        # 2. 光学 (Optics) - Always available (lens correction & optical diffusion)
        sec_optics = getattr(self, "sec_optics", None)
        if sec_optics is not None and hasattr(sec_optics, "set_available"):
            sec_optics.set_available(True, "")
        optics_widgets = [
            getattr(self, 'chk_lens_profile_enabled', None),
            getattr(self, 'chk_remove_ca', None),
            getattr(self, 'combo_lens_setup', None),
            getattr(self, 'combo_lens_make', None),
            getattr(self, 'combo_lens_model', None),
            getattr(self, 'combo_lens_profile', None),
            getattr(self, 'slider_lens_distortion', None),
            getattr(self, 'slider_lens_vignette', None),
            getattr(self, 'slider_manual_distortion', None),
            getattr(self, 'slider_manual_vignette_amount', None),
            getattr(self, 'slider_manual_vignette_midpoint', None),
            getattr(self, 'combo_diffusion', None),
        ]
        for w in optics_widgets:
            if w is not None:
                w.setEnabled(True)
                w.setToolTip(getattr(w, "default_tooltip", ""))
                for child in [w] + w.findChildren(QWidget):
                    child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._update_diffusion_ui_state()

        # 3. 胶片化学 (DIR) - Only available on films carrying DIR couplers
        self._update_chemistry_ui_state()

        # 4. 放大机与相纸显影 - Only available on negative film in optical print mode
        enlarger_available, enlarger_hint = self._enlarger_status()
        enlarger_and_paper_widgets = [
            getattr(self, 'combo_illuminant', None),
            getattr(self, 'slider_cyan', getattr(self, 'slider_enlarger_c', None)),
            getattr(self, 'slider_magenta', getattr(self, 'slider_enlarger_m', None)),
            getattr(self, 'slider_yellow', getattr(self, 'slider_enlarger_y', None)),
            getattr(self, 'slider_print_exp', None),
            getattr(self, 'slider_preflash', None),
            getattr(self, 'slider_morph_gamma', None),
            getattr(self, 'slider_exhaustion', None),
        ]
        for w in enlarger_and_paper_widgets:
            if w is not None:
                w.setEnabled(enlarger_available)
                w.setToolTip(getattr(w, "default_tooltip", "") if enlarger_available else enlarger_hint)
        # 胶片预闪只需要"有胶片介质"：反转片、直扫、黑白卷都有效（它发生在显影之前），
        # 所以不能跟放大机那一组一起置灰，只在旁通模式下禁用。
        cam_pf = getattr(self, 'slider_preflash_camera', None)
        if cam_pf is not None:
            cam_pf.setEnabled(not is_none)
            cam_pf.setToolTip(
                getattr(cam_pf, "default_tooltip", "") if not is_none
                else "当前为数码原片旁通模式：没有胶片介质，胶片预闪无效")
        # fotufilm 点式链没有 morph / 显影衰竭这两个自由度（上游配置里根本没有这两类槽）。
        # 置灰并说明，而不是让它们看着能拖、实际什么都不做。
        if enlarger_available and self._on_ffchain_now():
            for slider, reason in (
                    (getattr(self, 'slider_morph_gamma', None), "fotufilm 链不含形态学伽马"),
                    (getattr(self, 'slider_exhaustion', None), "fotufilm 链不含显影衰竭")):
                if slider is not None:
                    slider.setEnabled(False)
                    slider.setToolTip(f"{reason}，切回 spektrafilm 链后生效")
                for child in [w] + w.findChildren(QWidget):
                    child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not enlarger_available)

        sec_enlarger = getattr(self, "sec_enlarger", None)
        if sec_enlarger is not None and hasattr(sec_enlarger, "set_available"):
            sec_enlarger.set_available(enlarger_available, enlarger_hint)
            if not enlarger_available:
                sec_enlarger.set_reset_enabled(False)

        # 5. 物理光晕 (Halation) - Only available with real film media
        sec_halation = getattr(self, "sec_halation", None)
        hal_available = not is_none
        hal_hint = "" if hal_available else "当前为数码原片旁通模式，无片基背反射光晕"
        if sec_halation is not None and hasattr(sec_halation, "set_available"):
            sec_halation.set_available(hal_available, hal_hint)
            if not hal_available:
                sec_halation.set_reset_enabled(False)
        halation_sliders = [
            getattr(self, "slider_halation_amount", None),
            getattr(self, "slider_halation_spatial_scale", None),
            getattr(self, "slider_scatter_amount", None),
            getattr(self, "slider_boost_ev", None),
        ]
        for s in halation_sliders:
            if s is not None:
                s.setEnabled(hal_available)
                s.setToolTip(getattr(s, "default_tooltip", "") if hal_available else hal_hint)
                for child in [s] + s.findChildren(QWidget):
                    child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not hal_available)

        # 6. 银盐颗粒 (Grain) - 旁通态不可用；彩色滑块在黑白胶卷上无意义（档案强制共用银影）
        sec_grain = getattr(self, "sec_grain", None)
        grain_available = not is_none
        grain_hint = "" if grain_available else "当前为数码原片旁通模式，无银盐乳剂与颗粒"
        try:
            is_bw_film = (not is_none) and bool(self.engine.is_bw_stock(self.current_film_stock))
        except Exception:
            is_bw_film = False
        if sec_grain is not None and hasattr(sec_grain, "set_available"):
            sec_grain.set_available(grain_available, grain_hint)
            if not grain_available:
                sec_grain.set_reset_enabled(False)
        grain_sliders = [
            getattr(self, "slider_grain_strength", None),
            getattr(self, "slider_grain_size", None),
            getattr(self, "slider_scan_noise", None),
        ]
        for s in grain_sliders:
            if s is not None:
                s.setEnabled(grain_available)
                s.setToolTip(getattr(s, "default_tooltip", "") if grain_available else grain_hint)
                for child in [s] + s.findChildren(QWidget):
                    child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not grain_available)
        colour_slider = getattr(self, "slider_grain_colour", None)
        if colour_slider is not None:
            colour_ok = grain_available and not is_bw_film
            colour_slider.setEnabled(colour_ok)
            if not grain_available:
                colour_slider.setToolTip(grain_hint)
            elif is_bw_film:
                colour_slider.setToolTip("黑白胶卷只有单一银影，无彩色颗粒可调")
            else:
                colour_slider.setToolTip(getattr(colour_slider, "default_tooltip", ""))
            for child in [colour_slider] + colour_slider.findChildren(QWidget):
                child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not colour_ok)

        # Left panel paper cards: lock and dim when in bypass or positive reversal film.
        # On a reversal slide every paper is inert (the medium is Digital Reference), so all of
        # them are disabled -- including the outline of "which one is still selected", which
        # stays visible because the choice is kept for the next negative.
        if hasattr(self, 'paper_cards') and self.paper_cards:
            for pid, card in self.paper_cards.items():
                if is_none:
                    card.setEnabled(pid == "none")
                    if pid != "none":
                        card.setToolTip("数码原片旁通状态无需相纸印相")
                    else:
                        card.setToolTip(getattr(card, "original_tooltip", ""))
                elif is_pos:
                    card.setEnabled(False)
                    card.setToolTip("反转正片不经过相纸；当前选择保留，切回负片时继续使用")
                else:
                    # 两条链的相纸集合不通用（见 engine.paper_available）：没有（卷，纸）
                    # 配对表的纸在这里置灰，而不是让它落到另一条链上出图。
                    allowed = self._paper_allowed_now(pid, card)
                    card.setEnabled(allowed)
                    if allowed:
                        card.setToolTip(getattr(card, "original_tooltip", ""))
                    else:
                        card.setToolTip(self._paper_unavailable_hint())

        self._update_reset_buttons_state()

        # 右侧相纸下拉框跟随当前状态（切换底片/撤销/相纸变更都走这里）
        combo_paper = getattr(self, "combo_paper", None)
        if combo_paper is not None:
            # 下拉框只列当前卷能用的纸；被链条排除的纸留在左侧卡片墙上（置灰 + 说明），
            # 这样用户能看到"这张纸存在，只是换不了"，而不是它凭空消失。
            self._refresh_paper_combo()
            # 旁通态不再把相纸改写成"直扫"：删掉 .sdc 之后相纸要留在首选项的默认纸上
            # （用户指定），这里改写成 none 会立刻把它抹掉。旁通本来就不走印相（引擎对
            # film=none 直接出旁通表），控制已置灰，保留选择只是让面板状态说得通。
            # 反转正片同理：相纸选择属于下面那张负片，切回负片就能直接用。
            combo_paper.set_value(getattr(self, "current_paper_stock", "none"))
            paper_ok = not (is_pos or is_none)
            combo_paper.setEnabled(paper_ok)
            combo_paper.setToolTip(
                getattr(combo_paper, "default_tooltip", "") if paper_ok
                else ("反转正片不经过相纸；当前选择保留，切回负片时继续使用" if is_pos
                      else "数码原片旁通模式无相纸印相环节")
            )

    def _update_diffusion_ui_state(self):
        """Enable or gray out diffusion strength and warmth controls based on diffusion_family."""
        diff_family = str(self.current_params.get("diffusion_family", "none")).lower()
        is_diff_none = (diff_family == "none")

        if hasattr(self, 'combo_diff_strength') and self.combo_diff_strength is not None:
            self.combo_diff_strength.setEnabled(not is_diff_none)
            if is_diff_none:
                self.combo_diff_strength.setToolTip("选择柔光镜型号后可调节挡位")
            else:
                self.combo_diff_strength.setToolTip(getattr(self.combo_diff_strength, "default_tooltip", "柔光镜散射密度（1/8～2 档）"))

        if hasattr(self, 'slider_diff_warmth') and self.slider_diff_warmth is not None:
            self.slider_diff_warmth.setEnabled(not is_diff_none)
            if is_diff_none:
                self.slider_diff_warmth.setToolTip("选择柔光镜型号后可调节冷暖偏置")
            else:
                self.slider_diff_warmth.setToolTip(getattr(self.slider_diff_warmth, "default_tooltip", "柔光光晕冷暖：正值偏暖金，负值偏冷青"))

    @staticmethod
    def _lens_frame_aspect(photo):
        """Frame height/width, used for the vignette corner radius (3:2 landscape = 0.6667).

        A portrait 3:2 frame is 1.5, not 0.6667 — feeding the landscape value biased the
        corner r² and therefore the compensated gain for every upright photo.
        """
        if isinstance(photo, dict):
            w = photo.get("width") or (photo.get("meta") or {}).get("width")
            h = photo.get("height") or (photo.get("meta") or {}).get("height")
            try:
                if w and h and float(w) > 0:
                    return float(h) / float(w)
            except (TypeError, ValueError):
                pass
        return 0.6667

    def _get_default_lens_params(self):
        return {
            "profile_enabled": False,
            "setup": "auto",
            "make": "",
            "model": "",
            "profile_name": "",
            "profile_path": "",
            "profile_distortion": 0.0,
            "profile_vignette": 0.0,
            "remove_ca": False,
            "active_tab": "profile",
            "manual_enabled": True,
            "manual_distortion": 0.0,
            "manual_vignette_amount": 0.0,
            "manual_vignette_midpoint": 0.0,
            "enabled": False,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "vignette_amount": 0.0,
            "vignette_midpoint": 50.0,
        }

    def _update_lens_ui_state(self):
        """Enable or gray out profile controls depending on whether profile correction is enabled."""
        lens_cfg = self.current_params.get("lens", {})
        prof_en = bool(lens_cfg.get("profile_enabled", False))
        is_builtin = str(lens_cfg.get("setup", "")).lower() == "builtin" or \
            lens_cfg.get("profile_path") == "@builtin"
        if hasattr(self, 'combo_lens_setup'):
            self.combo_lens_setup.setEnabled(prof_en)
        # 内置 has nothing to choose from: the tables are per-frame inside the RAW. Show the
        # three rows grayed out reading "内置" instead of leaving a stale lens list the user
        # could mistake for the active correction.
        rows_enabled = prof_en and not is_builtin
        for name in ('combo_lens_make', 'combo_lens_model', 'combo_lens_profile'):
            if hasattr(self, name):
                getattr(self, name).setEnabled(rows_enabled)
        if is_builtin and hasattr(self, 'combo_lens_make'):
            self._is_updating_sliders = True
            try:
                self.combo_lens_make.set_items([("内置", "")], default_val="")
                self.combo_lens_model.set_items([("内置", "")], default_val="")
                self.combo_lens_profile.set_items(
                    [("内置（RAW 内）", "@builtin")], default_val="@builtin")
            finally:
                self._is_updating_sliders = False
        if hasattr(self, 'slider_lens_distortion'):
            self.slider_lens_distortion.setEnabled(prof_en)
        if hasattr(self, 'slider_lens_vignette'):
            self.slider_lens_vignette.setEnabled(prof_en)

    def _on_lens_tab_changed(self, tab_id):
        active_tab = "profile" if tab_id == 0 else "manual"
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["active_tab"] = active_tab
        if hasattr(self, 'panel_lens_profile'):
            self.panel_lens_profile.setVisible(tab_id == 0)
        if hasattr(self, 'panel_lens_manual'):
            self.panel_lens_manual.setVisible(tab_id == 1)
        self._apply_lens_to_canvas()

    def _on_lens_remove_ca_toggled(self, checked):
        if getattr(self, "_is_updating_sliders", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["remove_ca"] = checked
        self._update_auto_ca_ui()
        self._ensure_auto_ca()
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()

    # ------------------------------------------------------------------ 自动色差估计
    def _active_photo(self):
        if not self.active_photo_id:
            return None
        return next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)

    def _ensure_auto_ca(self, force: bool = False):
        """Measure lateral CA for the active frame if it has never been measured.

        Cached in the sidecar under ``auto_ca_r`` / ``auto_ca_b``; presence of the keys
        means "already measured", including a measured zero. Nothing happens unless
        删除色差 is on, so the cost is only ever paid by users who asked for it.
        """
        photo = self._active_photo()
        lens_cfg = self.current_params.get("lens", {})
        if not photo or not isinstance(lens_cfg, dict):
            return
        if not lens_cfg.get("remove_ca", False):
            self._update_auto_ca_ui()
            return
        if getattr(self, "_ca_worker", None) is not None and self._ca_worker.isRunning():
            return
        if not force and all(k in lens_cfg for k in _AUTO_CA_KEYS):
            self._update_auto_ca_ui()
            return
        img = getattr(self.engine, "raw_preview", None)
        if img is None or getattr(img, "ndim", 0) != 3 or max(img.shape[:2]) < 640:
            self._update_auto_ca_ui(no_image=True)
            return
        # The engine may still be holding the previous frame while the switch settles;
        # measuring that would cache one photo's CA onto another.
        cur_path = getattr(self.engine, "current_file_path", None)
        try:
            if cur_path and os.path.normcase(os.path.abspath(cur_path)) != \
                    os.path.normcase(os.path.abspath(photo.get("path", ""))):
                return
        except Exception:
            pass
        # Copy: the worker outlives this call, and the engine may rebind or mutate
        # raw_preview while the estimate runs (~33 MB at 2048 px, one-off).
        self._ca_worker = CAEstimateWorker(
            photo.get("id"), np.array(img, dtype=np.float32, copy=True))
        self._ca_worker.caReady.connect(self._on_auto_ca_ready)
        self._ca_pending_id = photo.get("id")
        # The row itself carries the progress: spinner next to the checkbox, "未找到"
        # once a measurement comes back empty.
        self.lbl_ca_status.setVisible(False)
        self.spinner_ca.start()
        self._ca_worker.start(QThread.Priority.LowPriority)

    def _on_reestimate_ca(self):
        lens_cfg = self.current_params.get("lens", {})
        if isinstance(lens_cfg, dict):
            for k in _AUTO_CA_KEYS + ("auto_ca_shift", "auto_ca_confidence"):
                lens_cfg.pop(k, None)
        self._ensure_auto_ca(force=True)

    def _on_auto_ca_ready(self, photo_id, result):
        photo = next((p for p in self.photos if p and p.get("id") == photo_id), None)
        if not photo:
            self._update_auto_ca_ui()
            return
        is_active = (photo_id == self.active_photo_id)
        params = self.current_params if is_active else (photo.get("params") or {})
        if not isinstance(params, dict):
            params = {}
        lens_cfg = params.get("lens")
        if not isinstance(lens_cfg, dict):
            lens_cfg = params.setdefault("lens", self._get_default_lens_params())

        lens_cfg["auto_ca_r"] = tuple(float(c) for c in result.get("ca_r_coeffs", (0.0, 0.0, 0.0)))
        lens_cfg["auto_ca_b"] = tuple(float(c) for c in result.get("ca_b_coeffs", (0.0, 0.0, 0.0)))
        lens_cfg["auto_ca_scale_r"] = round(float(result.get("ca_r_scale", 0.0)), 7)
        lens_cfg["auto_ca_scale_b"] = round(float(result.get("ca_b_scale", 0.0)), 7)
        lens_cfg["auto_ca_shift"] = (round(float(result.get("r_shift_px", 0.0)), 2),
                                     round(float(result.get("b_shift_px", 0.0)), 2))
        lens_cfg["auto_ca_confidence"] = round(float(result.get("confidence", 0.0)), 3)

        if is_active:
            self._apply_lens_to_canvas()
            self._update_auto_ca_ui()
            self._mark_current_photo_dirty()
        try:
            from core.sdc_manager import save_sdc
            save_sdc(photo["path"], photo.get("film_profile", self.current_film_stock),
                     photo.get("paper_profile") or "", params,
                     thumbnail_rgb=photo.get("thumbnail_rgb"),
                     print_mode=photo.get("print_mode", self.current_print_mode))
        except Exception as e:
            logger.debug(f"自动色差结果写入旁车失败: {e}")

    def _update_auto_ca_ui(self, no_image: bool = False):
        """Spinner while measuring, "未找到" when the frame has no lateral CA, else quiet.

        The measured shift used to be printed on a second line; it is not shown anywhere
        now -- the correction either ran or it did not.
        """
        if not hasattr(self, "lbl_ca_status"):
            return
        lens_cfg = self.current_params.get("lens", {})
        on = isinstance(lens_cfg, dict) and bool(lens_cfg.get("remove_ca", False))
        if not on:
            self.spinner_ca.stop()
            self.lbl_ca_status.setVisible(False)
            return

        worker = getattr(self, "_ca_worker", None)
        if worker is not None and worker.isRunning():
            self.spinner_ca.start()
            self.lbl_ca_status.setVisible(False)
            return

        self.spinner_ca.stop()
        if no_image:
            self.lbl_ca_status.setText("不可用")
            self.lbl_ca_status.setVisible(True)
            return
        if not all(k in lens_cfg for k in _AUTO_CA_KEYS):
            self.lbl_ca_status.setVisible(False)
            return
        shift = lens_cfg.get("auto_ca_shift") or (0.0, 0.0)
        if abs(float(shift[0])) < 0.05 and abs(float(shift[1])) < 0.05:
            self.lbl_ca_status.setText("未找到")
            self.lbl_ca_status.setVisible(True)
        else:
            self.lbl_ca_status.setVisible(False)

    def _on_lens_profile_enabled_toggled(self, checked):
        if getattr(self, "_is_updating_sliders", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["profile_enabled"] = checked
        self._update_lens_ui_state()
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

    def _on_lens_setup_changed(self, setup_mode):
        if getattr(self, "_is_updating_sliders", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["setup"] = setup_mode
        if hasattr(self, 'combo_lens_setup') and self.combo_lens_setup.current_value() != setup_mode:
            # The combo is the entry point for this handler, so this only fires when the
            # mode is set programmatically (session restore, tests) -- keep the display
            # and the saved state from drifting apart.
            self._is_updating_sliders = True
            try:
                self.combo_lens_setup.set_value(setup_mode)
            finally:
                self._is_updating_sliders = False
        if setup_mode == "builtin":
            # 内置 tables live inside the RAW file (sony_lens reads them per frame), so
            # there is no brand/model/profile to pick -- the three rows below go read-only
            # and say "内置" rather than offering a list that cannot apply.
            lens_cfg["make"] = ""
            lens_cfg["model"] = ""
            lens_cfg["profile_name"] = "内置（RAW 内）"
            lens_cfg["profile_path"] = "@builtin"
        elif lens_cfg.get("profile_path") == "@builtin":
            # Leaving 内置: drop the pseudo-profile so nothing stale keeps being applied.
            lens_cfg["profile_name"] = ""
            lens_cfg["profile_path"] = ""
            if setup_mode == "auto":
                self._auto_detect_lens_profile()
            else:
                self._sync_lens_combos_from_cfg(lens_cfg)
        elif setup_mode == "auto":
            self._auto_detect_lens_profile()
        self._update_lens_ui_state()
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

    def _on_lens_make_changed(self, make):
        if getattr(self, "_is_updating_sliders", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["make"] = make

        # If user manually selects a make while in auto mode, switch setup to custom
        if lens_cfg.get("setup") == "auto":
            lens_cfg["setup"] = "custom"
            if hasattr(self, 'combo_lens_setup'):
                self._is_updating_sliders = True
                self.combo_lens_setup.set_value("custom")
                self._is_updating_sliders = False

        from core.lens_correction import LCPDatabase
        db = LCPDatabase.get_instance()
        models = db.get_models_for_make(make) if make else []
        seen_models = set()
        model_items = [("无", "")]
        for m_name, _, _ in models:
            if m_name not in seen_models:
                seen_models.add(m_name)
                model_items.append((m_name, m_name))

        self._is_updating_sliders = True
        try:
            self.combo_lens_model.set_items(model_items, default_val="")
            self.combo_lens_profile.set_items([("无", ""), ("[+ 添加配置文件 (.lcp)...]", "__import_lcp__")], default_val="")
        finally:
            self._is_updating_sliders = False

        lens_cfg["model"] = ""
        lens_cfg["profile_name"] = ""
        lens_cfg["profile_path"] = ""
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

    def _on_lens_model_changed(self, model_name):
        if getattr(self, "_is_updating_sliders", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["model"] = model_name
        if lens_cfg.get("setup") == "auto":
            lens_cfg["setup"] = "custom"
            if hasattr(self, 'combo_lens_setup'):
                self._is_updating_sliders = True
                self.combo_lens_setup.set_value("custom")
                self._is_updating_sliders = False

        make = lens_cfg.get("make", "")
        from core.lens_correction import LCPDatabase
        db = LCPDatabase.get_instance()
        models = db.get_models_for_make(make) if make else []
        prof_items = [("无", "")]
        matching_paths = []
        for m_name, p_display, p_path in models:
            if m_name == model_name:
                prof_items.append((p_display, p_path))
                matching_paths.append((p_display, p_path))

        prof_items.append(("[+ 添加配置文件 (.lcp)...]", "__import_lcp__"))
        self._is_updating_sliders = True
        try:
            def_path = ""
            if matching_paths:
                active_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                is_raw = bool(active_p and active_p.get("ext", "").lower() in getattr(self.engine, "RAW_EXTS", ()))
                if is_raw:
                    raw_cand = next((p[1] for p in matching_paths if "raw" in p[1].lower()), None)
                    def_path = raw_cand or matching_paths[0][1]
                else:
                    def_path = matching_paths[0][1]
            self.combo_lens_profile.set_items(prof_items, default_val=def_path)
        finally:
            self._is_updating_sliders = False

        # Store the profile the combo is actually showing: it prefers the "- RAW" variant,
        # while this used to always save matching_paths[0], so the panel and the renderer
        # disagreed until the user re-picked the same entry by hand.
        if def_path:
            lens_cfg["profile_path"] = def_path
            lens_cfg["profile_name"] = next(
                (p[0] for p in matching_paths if p[1] == def_path), matching_paths[0][0])
        else:
            lens_cfg["profile_name"] = ""
            lens_cfg["profile_path"] = ""
        # New optics: a cached lateral-CA measurement no longer describes this frame.
        for k in _AUTO_CA_KEYS + ("auto_ca_shift", "auto_ca_confidence"):
            lens_cfg.pop(k, None)
        self._ensure_auto_ca()
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

    def _on_lens_profile_changed(self, profile_path):
        if getattr(self, "_is_updating_sliders", False):
            return
        if profile_path == "__import_lcp__":
            self._handle_import_lens_profile()
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["profile_path"] = profile_path
        if hasattr(self, 'combo_lens_profile') and hasattr(self.combo_lens_profile, 'combo'):
            lens_cfg["profile_name"] = self.combo_lens_profile.combo.currentText()
        # A different profile means different optics, so a cached CA measurement is no
        # longer evidence about this frame. This is also the only remaining re-estimate
        # trigger now that the "重新估计" button is gone.
        for k in _AUTO_CA_KEYS + ("auto_ca_shift", "auto_ca_confidence"):
            lens_cfg.pop(k, None)
        self._ensure_auto_ca()
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

    def _handle_import_lens_profile(self):
        """Handle user selecting '[+ 添加配置文件 (.lcp)...]' in profile dropdown."""
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        prev_path = lens_cfg.get("profile_path", "")

        last_d = config_manager.get_last_dialog_dir("lcp")
        f_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择镜头校正配置文件 (.lcp)",
            last_d,
            "Adobe 镜头配置文件 (*.lcp);;所有文件 (*.*)"
        )
        if not f_path:
            # Revert selection back to previous
            if hasattr(self, "combo_lens_profile"):
                self._is_updating_sliders = True
                try:
                    self.combo_lens_profile.set_value(prev_path)
                finally:
                    self._is_updating_sliders = False
            return

        config_manager.set_last_dialog_dir("lcp", f_path)

        from core.lens_correction import import_user_lcp_file
        cur_make = lens_cfg.get("make", "")
        success, make, model_name, prof_display, dest_path = import_user_lcp_file(f_path, target_make=cur_make)

        if not success:
            self._show_warning_dialog("导入失败", f"无法导入配置文件:\n{f_path}")
            if hasattr(self, "combo_lens_profile"):
                self._is_updating_sliders = True
                try:
                    self.combo_lens_profile.set_value(prev_path)
                finally:
                    self._is_updating_sliders = False
            return

        lens_cfg["make"] = make
        lens_cfg["model"] = model_name
        lens_cfg["profile_name"] = prof_display
        lens_cfg["profile_path"] = dest_path
        lens_cfg["setup"] = "custom"
        lens_cfg["enabled"] = True
        lens_cfg["profile_enabled"] = True

        if hasattr(self, "combo_lens_setup"):
            self.combo_lens_setup.set_value("custom")
        if hasattr(self, "chk_lens_profile_enabled"):
            self.chk_lens_profile_enabled.setChecked(True)

        self._sync_lens_combos_from_cfg(lens_cfg)
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

        self.set_app_status(f"已导入镜头配置文件：{prof_display}", timeout_ms=4000)

    def _on_lens_amount_changed(self, key, val):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg[key] = val
        fl = None
        ap = None
        photo = None
        if self.active_photo_id:
            photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if photo:
                exif = photo.get("exif", {})
                fl = exif.get("focal_length")
                ap = exif.get("aperture")
        from core.lens_correction import compute_effective_lens_params
        try:
            eff = compute_effective_lens_params(lens_cfg, focal_length=fl, aperture=ap,
aspect_ratio=self._lens_frame_aspect(photo),
                                                source_path=(photo or {}).get("path"))
            lens_cfg.update(eff)
            self.canvas.update_params(lens=eff)
        except Exception as e:
            logger.error(f"Error in _on_lens_amount_changed: {e}", exc_info=True)
        self._update_reset_buttons_state()

    def _apply_lens_to_canvas(self):
        """Compute effective shader coefficients and forward to canvas."""
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        fl = None
        ap = None
        photo = None
        if self.active_photo_id:
            photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if photo:
                exif = photo.get("exif", {})
                fl = exif.get("focal_length")
                ap = exif.get("aperture")
        from core.lens_correction import compute_effective_lens_params
        try:
            eff = compute_effective_lens_params(lens_cfg, focal_length=fl, aperture=ap,
aspect_ratio=self._lens_frame_aspect(photo),
                                                source_path=(photo or {}).get("path"))
            lens_cfg.update(eff)
            self.canvas.update_params(lens=eff)
        except Exception as e:
            logger.error(f"Error in _apply_lens_to_canvas: {e}", exc_info=True)
        self._update_reset_buttons_state()

    def _sync_lens_combos_from_cfg(self, lens_cfg):
        self._is_updating_sliders = True
        try:
            from core.lens_correction import LCPDatabase
            db = LCPDatabase.get_instance()
            make = lens_cfg.get("make", "")
            model = lens_cfg.get("model", "")
            prof_path = lens_cfg.get("profile_path", "")

            # 内置 is a mode of the 设置 row, not a brand: reflect it there and leave the
            # three dependent rows reading "内置" (they are also disabled by
            # _update_lens_ui_state).
            is_builtin = (str(lens_cfg.get("setup", "")).lower() == "builtin"
                          or prof_path == "@builtin")
            if is_builtin:
                lens_cfg["setup"] = "builtin"
                if hasattr(self, 'combo_lens_setup'):
                    self.combo_lens_setup.set_value("builtin")
                self.combo_lens_make.set_items([("内置", "")], default_val="")
                self.combo_lens_model.set_items([("内置", "")], default_val="")
                self.combo_lens_profile.set_items(
                    [("内置（RAW 内）", "@builtin")], default_val="@builtin")
                return

            # 1. Populate makes
            makes = db.get_available_makes()
            make_items = [("无", "")] + [(m, m) for m in makes]
            if make and make not in [m for _, m in make_items]:
                make_items.append((make, make))
            self.combo_lens_make.set_items(make_items, default_val=make)

            # 2. Populate models
            models = db.get_models_for_make(make) if make else []
            seen_models = set()
            model_items = [("无", "")]
            for m_name, _, _ in models:
                if m_name not in seen_models:
                    seen_models.add(m_name)
                    model_items.append((m_name, m_name))
            if model and model not in [m for _, m in model_items]:
                model_items.append((model, model))
            self.combo_lens_model.set_items(model_items, default_val=model)

            # 3. Populate profiles
            prof_items = [("无", "")]
            for m_name, p_name, p_p in models:
                if m_name == model:
                    prof_items.append((p_name, p_p))
            if prof_path and prof_path not in [p for _, p in prof_items]:
                p_display = lens_cfg.get("profile_name") or "已载入配置文件"
                prof_items.append((p_display, prof_path))
            prof_items.append(("[+ 添加配置文件 (.lcp)...]", "__import_lcp__"))
            self.combo_lens_profile.set_items(prof_items, default_val=prof_path)
        finally:
            self._is_updating_sliders = False

    def _auto_detect_lens_profile(self):
        """Auto-detect camera/lens from photo EXIF metadata and match Adobe LCP profile."""
        if not self.active_photo_id:
            return
        photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
        if not photo:
            return

        from core.lens_correction import LCPDatabase, read_lens_metadata, detect_lens_brand
        exif = photo.get("exif", {}) or {}
        has_rich_exif = (exif.get("camera_model") and exif.get("camera_model") != "-") or (exif.get("lens_model") and exif.get("lens_model") != "-")
        meta = {} if has_rich_exif else read_lens_metadata(photo["path"])

        # Prioritize rich EXIF from SpektraEngine._extract_exif stored in photo['exif']
        cam_make = (exif.get("camera_make") if exif.get("camera_make") != "-" else "") or meta.get("camera_make", "")
        cam_model = (exif.get("camera_model") if exif.get("camera_model") != "-" else "") or meta.get("camera_model", "")
        lens_model = (exif.get("lens_model") if exif.get("lens_model") != "-" else "") or meta.get("lens_model", "")
        fl = exif.get("focal_length") or meta.get("focal_length")

        db = LCPDatabase.get_instance()
        match = db.match_lens_from_exif(cam_make, cam_model, lens_model, fl)
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())

        if match:
            m, mod_name, prof_name, p_path = match
            lens_cfg["make"] = m
            lens_cfg["model"] = mod_name
            lens_cfg["profile_name"] = prof_name
            lens_cfg["profile_path"] = p_path
        else:
            suspected_make = detect_lens_brand(lens_model, cam_make)
            lens_cfg["make"] = suspected_make or cam_make or ""
            lens_cfg["model"] = lens_model or cam_model or "未找到匹配镜头"
            lens_cfg["profile_name"] = "无配置文件"
            lens_cfg["profile_path"] = ""

        if hasattr(self, 'combo_lens_make'):
            self._sync_lens_combos_from_cfg(lens_cfg)
            self._update_lens_ui_state()

    def _manage_photo_memory_lru(self, active_photo_id: str):
        """Strict LRU memory management: Keeps at most 2 photos' float_img in RAM
        (active photo + most recent previous photo, ~580MB max).
        All other photos rely on ultra-fast (20ms) disk session cache.
        Evicts float arrays from self.photos, engine.session_photos, and runs gc.collect().
        """
        if not hasattr(self, "_loaded_photos_lru"):
            self._loaded_photos_lru = []
        if active_photo_id in self._loaded_photos_lru:
            self._loaded_photos_lru.remove(active_photo_id)
        self._loaded_photos_lru.append(active_photo_id)

        evicted = False
        while len(self._loaded_photos_lru) > 2:
            old_pid = self._loaded_photos_lru.pop(0)
            if old_pid != active_photo_id:
                old_p = next((p for p in self.photos if p and p.get("id") == old_pid), None)
                if old_p and old_p.get("float_img") is not None:
                    old_p["float_img"] = None
                    evicted = True
                if hasattr(self, 'engine') and hasattr(self.engine, 'session_photos'):
                    for ep in self.engine.session_photos:
                        if ep.get("id") == old_pid and ep.get("raw_preview") is not None:
                            ep["raw_preview"] = None
                            evicted = True

        # Safety sweep: Purge any non-LRU photos that inadvertently hold float_img
        lru_set = set(self._loaded_photos_lru)
        for p in self.photos:
            if not p:
                continue
            if p.get("id") not in lru_set and p.get("float_img") is not None:
                p["float_img"] = None
                evicted = True
        if hasattr(self, 'engine') and hasattr(self.engine, 'session_photos'):
            for ep in self.engine.session_photos:
                if not ep:
                    continue
                if ep.get("id") not in lru_set and ep.get("raw_preview") is not None:
                    ep["raw_preview"] = None
                    evicted = True

        if evicted:
            import gc
            gc.collect()

    def on_switch_photo(self, photo_id, force=False):
        # 用户自己在导入期间点了照片：作废"导入结束后再切回首图"的约定。
        # `_activate_deferred_photo` 是先把 id 取走再调进来的，所以这里清空不会打断它自己。
        self._deferred_switch_id = None
        # Stop any lingering live/debounce timers from previous photo
        if hasattr(self, '_live_thumb_timer') and self._live_thumb_timer.isActive():
            self._live_thumb_timer.stop()
        if hasattr(self, '_cards_debounce_timer') and self._cards_debounce_timer.isActive():
            self._cards_debounce_timer.stop()
        if hasattr(self, 'thumb_controller') and self.thumb_controller:
            self.thumb_controller.cancel_card_preview_worker()

        # If user clicks the currently active photo and it is already fully loaded, avoid redundant reload
        if not force and photo_id == self.active_photo_id:
            curr_photo = next((p for p in self.photos if p and p.get("id") == photo_id), None)
            if curr_photo and curr_photo.get("float_img") is not None:
                return
            if self._single_photo_loader and self._single_photo_loader.isRunning():
                return

        # Auto-reset compare / split-screen mode to normal edit view when switching to another photo
        self.set_view_mode(0)

        self._is_switching_photo = True
        if hasattr(self, 'thumb_controller') and self.thumb_controller:
            self.thumb_controller.cancel_card_preview_worker()
        try:
            # 1. Update in-memory state and ensure SDC is saved for previous photo ONLY if genuinely dirty
            if self.active_photo_id:
                prev_photo = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                if prev_photo and getattr(self, "_photo_is_dirty", False):
                    # If in crop workspace, user has not clicked apply, discard unapplied crop for previous photo
                    if getattr(self.canvas, "is_crop_editing", False):
                        old_c = prev_photo.get("crop")
                        if not old_c and isinstance(prev_photo.get("params"), dict):
                            old_c = prev_photo["params"].get("crop")
                        if old_c:
                            self.current_params["crop"] = dict(old_c)
                        else:
                            self.current_params.pop("crop", None)
                    self._save_current_photo_sdc()

            # 2. Find photo entry
            photo = next((p for p in self.photos if p and p.get("id") == photo_id), None)
            if not photo:
                return

            self.active_photo_id = photo_id
            self._current_photo_path = photo["path"]
            if "rating" not in photo:
                photo["rating"] = 0
            if hasattr(self, 'canvas_rating'):
                self.canvas_rating.set_rating(photo.get("rating", 0))
                if hasattr(self, 'view_toolbar') and hasattr(self.view_toolbar, 'set_rating_enabled'):
                    self.view_toolbar.set_rating_enabled(True)
                else:
                    self.canvas_rating.setVisible(True)
            if getattr(self.filmstrip, "_active_id", None) != photo_id:
                self.filmstrip.set_active_photo(photo_id, scroll_to_visible=True, reset_selection=True)
            self.canvas.set_has_library_photos(bool(self.photos))

            # Automatically return parameter sidebar to top (exposure pipeline start)
            if hasattr(self, "sidebar_scroll") and self.sidebar_scroll:
                self.sidebar_scroll.verticalScrollBar().setValue(0)

            # 3. Ensure EXIF metadata (camera WB, model, etc.)
            # If photo will be loaded asynchronously via SinglePhotoLoadWorker (not FAST_TEST and no float_img yet),
            # defer heavy disk EXIF parsing to the background worker thread so switching/importing never blocks the GUI thread!
            exif_data = photo.get("exif")
            has_full_exif = bool(exif_data and exif_data.get("camera_model", "-") != "-")
            if not has_full_exif and (photo.get("float_img") is not None or os.environ.get("FAST_TEST")):
                try:
                    extracted = self.engine._extract_exif(photo["path"])
                    if photo.get("exif"):
                        photo["exif"].update(extracted)
                    else:
                        photo["exif"] = extracted
                    exif_data = photo["exif"]
                except Exception:
                    pass
            self.exif_widget.set_exif_data(exif_data or {})
            if hasattr(self, 'histogram_exif_bar'):
                self.histogram_exif_bar.set_exif_data(exif_data or {})

            base_temp = float(exif_data.get("color_temp", 5500.0) if exif_data else 5500.0)
            base_tint = float(exif_data.get("tint", 1.0) if exif_data else 1.0)
            base_ev = float(photo.get("base_ev", 0.0))
            if abs(base_ev) < 0.0001:
                if photo.get("params") and "base_ev" in photo["params"] and abs(float(photo["params"]["base_ev"])) > 0.0001:
                    base_ev = float(photo["params"]["base_ev"])
                    photo["base_ev"] = base_ev
                elif sdc_manager.has_sdc(photo["path"]):
                    sdc_chk = sdc_manager.load_sdc(photo["path"])
                    if sdc_chk and sdc_chk.get("params", {}).get("base_ev"):
                        base_ev = float(sdc_chk["params"]["base_ev"])
                        photo["base_ev"] = base_ev
                # The session cache is deliberately NOT consulted for base_ev: legacy entries
                # carry the old auto-metered value and the meter itself no longer ships one
                # (decode keeps it in auto_ev; the explicit auto-exposure button applies it).
                # An sdc sidecar stays authoritative because the user saved it deliberately.
            self._active_base_temp = base_temp
            self._active_base_tint = base_tint
            self._active_base_ev = base_ev
            self.slider_temp.set_default_value(base_temp)
            self.slider_tint.set_default_value(base_tint)

            # 4. Check for in-memory edits or SDC sidecar configuration
            sdc = None
            if photo.get("params"):
                base_dict = self.get_default_params(base_temp, base_tint)
                base_dict.update(photo["params"])
                if "halation_amount" not in photo["params"] and "halation" in photo["params"]:
                    base_dict["halation_amount"] = photo["params"]["halation"]
                if "boost_ev" not in photo["params"] and "halation_boost" in photo["params"]:
                    base_dict["boost_ev"] = photo["params"]["halation_boost"]
                self.current_film_stock = photo.get("film_profile") or "none"
                self.current_paper_stock = photo.get("paper_profile") or "none"
                self.current_print_mode = photo.get("print_mode") or base_dict.get("print_mode", "optical")
                base_dict["print_mode"] = self.current_print_mode
                base_dict["base_temp"] = base_temp
                base_dict["base_tint"] = base_tint
                if "base_ev" in photo["params"] and abs(float(photo["params"]["base_ev"])) > 0.0001:
                    base_ev = float(photo["params"]["base_ev"])
                base_dict["base_ev"] = base_ev
                self.current_params.clear()
                self.current_params.update(base_dict)
                self._photo_is_dirty = photo.get("is_dirty", False)
            else:
                sdc = sdc_manager.load_sdc(self._current_photo_path)
                if sdc:
                    base_dict = self.get_default_params(base_temp, base_tint)
                    loaded_params = sdc.get("params", {})
                    base_dict.update(loaded_params)
                    if "halation_amount" not in loaded_params and "halation" in loaded_params:
                        base_dict["halation_amount"] = loaded_params["halation"]
                    if "boost_ev" not in loaded_params and "halation_boost" in loaded_params:
                        base_dict["boost_ev"] = loaded_params["halation_boost"]
                    self.current_film_stock = sdc.get("film_profile") or "none"
                    self.current_paper_stock = sdc.get("paper_profile") or "none"
                    if "paper_explicit" in sdc:
                        photo["paper_explicit"] = bool(sdc.get("paper_explicit"))
                    self.current_print_mode = sdc.get("print_mode") or loaded_params.get("print_mode", "optical")
                    base_dict["print_mode"] = self.current_print_mode
                    if "color_temp" not in loaded_params or (loaded_params.get("color_temp") == 5500.0 and base_temp != 5500.0 and not loaded_params.get("custom_wb")):
                        base_dict["color_temp"] = base_temp
                    if "tint" not in loaded_params or (loaded_params.get("tint") == 1.0 and base_tint != 1.0 and not loaded_params.get("custom_wb")):
                        base_dict["tint"] = base_tint
                    base_dict["base_temp"] = base_temp
                    base_dict["base_tint"] = base_tint
                    if "base_ev" in loaded_params and abs(float(loaded_params["base_ev"])) > 0.0001:
                        base_ev = float(loaded_params["base_ev"])
                    base_dict["base_ev"] = base_ev
                    self.current_params.clear()
                    self.current_params.update(base_dict)
                    self._photo_is_dirty = False
                else:
                    self.current_film_stock = "none"
                    self.current_paper_stock = "none"
                    self.current_print_mode = "scan"
                    defaults = self.get_default_params(base_temp, base_tint)
                    defaults["print_mode"] = "scan"
                    defaults["base_ev"] = base_ev
                    self.current_params.clear()
                    self.current_params.update(defaults)
                    self._photo_is_dirty = False

            # 负片但相纸没记录（旧 .sdc、直扫遗留、旁通退回）：直接落到首选项的默认相纸，
            # 否则画面和缩略图都是橙色负像，而面板显示"胶片直扫"。用户选过的相纸不动，
            # 也不写回 photo：这张照片仍算"没选过"，首选项换纸时它会跟着变。
            eff_paper = self._photo_effective_paper(photo)
            if eff_paper != self.current_paper_stock:
                self.current_paper_stock = eff_paper
                self.current_print_mode = "optical"
                self.current_params["print_mode"] = "optical"
                # 记成"自动值"而不是"用户选的"：面板/缩略图会写回 photo，没有这个标记的话
                # 下次就变成"选过相纸"，首选项再改默认纸这张照片就不跟了。
                if photo is not None and not self._photo_paper_is_explicit(photo):
                    photo["paper_explicit"] = False

            # .sdc / 旧会话里可能存着这张卷不支持的纸（例如在原生卷上存了 Portra Endura，
            # 再拿同一份配置去开一张转换卷）。落点跟切卷同一条规则，别让面板和出图分叉。
            self._ensure_paper_allowed(self.current_film_stock)

            self.filmstrip.set_photo_dirty(photo_id, self._photo_is_dirty)
            self._update_window_title()

            # 5. Upload photo to OpenGL canvas texture (lazy load preview or RAW if needed)
            fn = photo.get("filename", os.path.basename(photo["path"]))
            if photo.get("float_img") is None:
                if os.environ.get("FAST_TEST"):
                    res = self.engine.load_image(photo["path"])
                    if res.get("success"):
                        photo["float_img"] = res.get("float_img")
                        if photo.get("thumbnail_rgb") is None:
                            photo["thumbnail_rgb"] = res.get("thumbnail_rgb")
                        if "base_ev" in res:
                            photo["base_ev"] = res["base_ev"]
                        session_cache_manager.save_photo_preview(photo["path"], photo["float_img"])
                else:
                    # 磁盘缓存和解码都交给 worker，主线程只负责先把占位图铺上去。
                    # 以前"有缓存"走的是同步 `get_photo_preview`（读 .npy 60-70 ms）+ 同步
                    # 上传，于是有缓存的切起来反而比没缓存的卡；现在两条路合成一条。
                    if not getattr(self, "_is_importing", False) and not getattr(self, "_is_multi_selecting", False):
                        self.set_app_status(f"正在解析 RAW：{fn}…", spinning=True, timeout_ms=0)

                    # Priority 2: Visual seamless placeholder in Direct sRGB Bypass mode (u_is_bypass = 2)
                    # Immediately present the bottom filmstrip or cached thumbnail with 0ms response,
                    # zero black flash, zero double grading, and zero empty state banner!
                    placeholder_img = photo.get("graded_thumbnail_rgb")
                    if placeholder_img is None or not isinstance(placeholder_img, np.ndarray) or placeholder_img.size == 0:
                        placeholder_img = photo.get("thumbnail_rgb")
                    if placeholder_img is None or not isinstance(placeholder_img, np.ndarray) or placeholder_img.size == 0:
                        placeholder_img = photo.get("raw_thumbnail_rgb")
                    if placeholder_img is None or not isinstance(placeholder_img, np.ndarray) or placeholder_img.size == 0:
                        placeholder_img = photo.get("raw_thumbnail")
                    m_w = photo.get("width") or (photo.get("meta", {}).get("width") if isinstance(photo.get("meta"), dict) else None)
                    m_h = photo.get("height") or (photo.get("meta", {}).get("height") if isinstance(photo.get("meta"), dict) else None)
                    if placeholder_img is not None and isinstance(placeholder_img, np.ndarray) and placeholder_img.size > 0:
                        self.canvas.set_placeholder_image(placeholder_img, master_width=m_w, master_height=m_h, photo_id=photo_id)
                    else:
                        self.canvas.set_image(None, photo_id=photo_id)

                    if self._single_photo_loader and self._single_photo_loader.isRunning():
                        try:
                            self._single_photo_loader.photoLoaded.disconnect()
                            self._single_photo_loader.previewReady.disconnect()
                        except Exception:
                            pass
                        if hasattr(self._single_photo_loader, "stop"):
                            self._single_photo_loader.stop()

                    self._single_photo_loader = SinglePhotoLoadWorker(photo_id, photo["path"], self.engine, self)
                    self._single_photo_loader.previewReady.connect(self._on_single_photo_preview_ready)
                    self._single_photo_loader.photoLoaded.connect(self._on_single_photo_loaded)
                    self._single_photo_loader.start()


            if photo.get("float_img") is not None:
                m_w = photo.get("width") or (photo.get("meta", {}).get("width") if isinstance(photo.get("meta"), dict) else None)
                m_h = photo.get("height") or (photo.get("meta", {}).get("height") if isinstance(photo.get("meta"), dict) else None)
                self.canvas.set_image(photo["float_img"], master_width=m_w, master_height=m_h, photo_id=photo_id)
                if not getattr(self, "_is_multi_selecting", False):
                    self.set_app_status(f"底片已载入：{fn}", spinning=False, timeout_ms=2000)

            self._manage_photo_memory_lru(photo_id)

            # Ensure lens configuration is present and auto-detected if in auto mode
            lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
            if lens_cfg.get("setup", "auto") == "auto":
                if photo.get("float_img") is not None or has_full_exif or os.environ.get("FAST_TEST"):
                    self._auto_detect_lens_profile()
            fl = exif_data.get("focal_length") if exif_data else None
            ap = exif_data.get("aperture") if exif_data else None
            from core.lens_correction import compute_effective_lens_params
            eff_lens = compute_effective_lens_params(lens_cfg, focal_length=fl, aperture=ap,
aspect_ratio=self._lens_frame_aspect(photo),
                                                     source_path=(photo or {}).get("path"))
            lens_cfg.update(eff_lens)

            self._update_stock_mode_ui()
            self._sync_canvas_params()
            self._update_all_sliders_from_params()
            # 裁切是逐张状态，不是全局状态：这里必须显式装载本张的裁切（没有就装载整幅），
            # 否则画布 uniform 和遮罩控件都会留着上一张的矩形与比例。
            self._apply_photo_crop_state(photo)

            # 6. Apply print mode, film & paper LUT in a single unified, instant pass
            eff_paper = self.current_paper_stock if self.current_print_mode == "optical" else "none"
            lut_params = dict(self.current_params)
            lut_params["print_mode"] = self.current_print_mode
            lut_3d = self.engine.get_3d_lut(self.current_film_stock, eff_paper, lut_size=33, params_dict=lut_params)
            self._apply_lut(lut_3d)

            hal_preset = self.engine.get_film_halation_preset(self.current_film_stock) if hasattr(self.engine, 'get_film_halation_preset') else None
            self.canvas.update_params(film_stock=self.current_film_stock, _halation_preset=hal_preset)

            # Sync active card highlights
            for sid, card in self.film_cards.items():
                card.set_active(sid == self.current_film_stock)
            for pid, card in self.paper_cards.items():
                card.set_active(pid == self.current_paper_stock)

            # 7. Update status text with 1:1 image native pixel scale percentage
            self._update_photo_info_display()
            self._update_zoom_buttons()

            # 8. Update live thumbnails on Film & Paper cards and filmstrip
            self._current_photo_thumb = photo.get("thumbnail_rgb")
            float_img = photo.get("float_img")
            if float_img is not None:
                if hasattr(self, 'engine'):
                    self.engine.raw_preview = float_img
                self._current_photo_linear = float_img
                self._schedule_active_thumbnail_update(100)
                # Active photo is already in RAM: refresh all 61 candidate cards in background after UI settles
                if hasattr(self, '_cards_debounce_timer'):
                    self._cards_debounce_timer.start(250)
            else:
                self._current_photo_linear = None
                if hasattr(self, 'engine'):
                    self.engine.raw_preview = None
                # Immediate (0ms) active card update with existing thumbnail (Important active card):
                th_display = photo.get("graded_thumbnail_rgb")
                raw_thumb = self.thumb_controller.get_raw_thumbnail(photo)
                if th_display is None and raw_thumb is not None:
                    th_paper = self._effective_paper_for_photo(self.current_film_stock, self.current_paper_stock)
                    if self.current_film_stock != "none" or th_paper != "none":
                        th_display = self.thumb_controller.grade_thumbnail_fast(
                            raw_thumb, self.current_film_stock, th_paper, self.current_params
                        )
                    else:
                        th_display = raw_thumb
                elif th_display is None:
                    th_display = photo.get("thumbnail_rgb")

                if th_display is not None and isinstance(th_display, np.ndarray) and th_display.size > 0:
                    th_c = np.ascontiguousarray(th_display, dtype=np.uint8)
                    th_h, th_w = th_c.shape[:2]
                    qimg = QImage(th_c.data, th_w, th_h, th_w * 3, QImage.Format.Format_RGB888).copy()
                    pix = QPixmap.fromImage(qimg)
                    self.thumb_controller.update_active_card_wysiwyg(
                        self.film_cards, self.paper_cards,
                        self.current_film_stock, self.current_paper_stock, pix
                    )
                else:
                    if self.film_cards and self.current_film_stock in self.film_cards:
                        self.film_cards[self.current_film_stock].set_thumbnail(None)
                    if self.paper_cards and self.current_paper_stock in self.paper_cards:
                        self.paper_cards[self.current_paper_stock].set_thumbnail(None)

                # Non-essential candidate cards (remaining 60 cards) are queued to refresh AFTER the negative is ready (_on_single_photo_loaded)

            self._schedule_histogram_update()
            self._update_stock_mode_ui()

            # 9. Track baseline state for newly switched photo without obliterating global undo/redo history
            self._active_photo_baseline_state = {
                "type": "single_photo_state",
                "photo_id": photo_id,
                "photo_path": photo.get("path"),
                "film": self.current_film_stock,
                "paper": self.current_paper_stock,
                "print_mode": getattr(self, "current_print_mode", "optical"),
                "params": dict(self.current_params)
            }
            if not self._undo_stack:
                self._undo_stack = [self._active_photo_baseline_state]
                self._redo_stack.clear()

            # Synchronize engine state
            if hasattr(self, 'engine'):
                self.engine.current_file_path = photo["path"]
                self.engine.active_photo_id = photo_id

            # 10. If currently in crop workspace, reset crop box to original ratio full screen for new photo
            if getattr(self.canvas, "is_crop_editing", False):
                default_crop = {
                    "enabled": False,
                    "x": 0.0,
                    "y": 0.0,
                    "w": 1.0,
                    "h": 1.0,
                    "angle": 0.0,
                    "rot_90": 0,
                    "flip_h": False,
                    "flip_v": False,
                    "aspect_ratio": "original",
                    "is_portrait": False
                }
                self.current_params["crop"] = dict(default_crop)
                self.canvas.params["crop"] = dict(default_crop)
                if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                    self.canvas.crop_overlay.start_crop_session(dict(default_crop))
                if hasattr(self, "crop_sidebar") and self.crop_sidebar:
                    self.crop_sidebar.sync_from_overlay(dict(default_crop))
                self.canvas.zoom = 0.90
                self.canvas.pan_x = 0.0
                self.canvas.pan_y = 0.0
                self.canvas.update()
        finally:
            self._is_switching_photo = False
            self._schedule_session_focus_save(300)

    def _on_single_photo_preview_ready(self, photo_id, photo_path, preview_rgb):
        if self.active_photo_id == photo_id and preview_rgb is not None:
            target_photo = next((p for p in self.photos if p and p.get("id") == photo_id), None)
            if target_photo and target_photo.get("float_img") is None:
                # Only upgrade to 1080p embedded preview if photo is unedited (film stock is "none")
                # so that raw visual clarity progresses smoothly: 0ms thumbnail -> 15ms 1080p -> 300ms float master
                is_unedited = (getattr(self, "current_film_stock", "none") == "none" and not getattr(self, "_photo_is_dirty", False))
                if is_unedited:
                    m_w = target_photo.get("width") or preview_rgb.shape[1]
                    m_h = target_photo.get("height") or preview_rgb.shape[0]
                    self.canvas.set_placeholder_image(preview_rgb, master_width=m_w, master_height=m_h, photo_id=photo_id)

    def _on_single_photo_loaded(self, photo_id, photo_path, res):
        if not res or not res.get("success"):
            if self.active_photo_id == photo_id:
                self.set_app_status(
                    f"底片载入失败：{os.path.basename(photo_path)}", spinning=False, timeout_ms=4000)
            return

        float_img = res.get("float_img")
        if float_img is None:
            float_img = res.get("raw_preview")
        if float_img is None:
            float_img = session_cache_manager.get_photo_preview(photo_path)
        if float_img is None:
            if self.active_photo_id == photo_id:
                self.set_app_status(
                    f"底片载入失败：{os.path.basename(photo_path)}（无可用的预览数据）",
                    spinning=False, timeout_ms=4000)
            return



        target_photo = next((p for p in self.photos if p and p.get("id") == photo_id), None)
        if target_photo:
            if self.active_photo_id == photo_id:
                target_photo["float_img"] = float_img
            else:
                target_photo["float_img"] = None
            if res.get("width") and res.get("height"):
                target_photo["width"] = res["width"]
                target_photo["height"] = res["height"]
            if res.get("exif"):
                target_photo["exif"] = res["exif"]
            if "base_ev" in res:
                target_photo["base_ev"] = res["base_ev"]
            curr_thumb = target_photo.get("thumbnail_rgb")
            thumb_is_dark = (curr_thumb is None or not isinstance(curr_thumb, np.ndarray) or curr_thumb.size == 0 or curr_thumb.mean() < 1.0)
            if thumb_is_dark and res.get("thumbnail_rgb") is not None:
                new_thumb = res.get("thumbnail_rgb")
                target_photo["raw_thumbnail_rgb"] = new_thumb
                if target_photo.get("is_edited") or target_photo.get("film_profile", "none") != "none":
                    graded_th = self.thumb_controller.grade_thumbnail_fast(
                        new_thumb,
                        target_photo.get("film_profile", "none"),
                        self._photo_effective_paper(target_photo),
                        target_photo.get("params") or self.current_params
                    )
                    self.thumb_controller.update_graded_thumbnail(target_photo, graded_th, self.filmstrip)
                else:
                    target_photo["thumbnail_rgb"] = new_thumb
                    th_c = np.ascontiguousarray(new_thumb, dtype=np.uint8)
                    th_qimg = QImage(th_c.data, th_c.shape[1], th_c.shape[0], th_c.shape[1] * 3, QImage.Format.Format_RGB888).copy()
                    self.filmstrip.update_photo_thumbnail(photo_id, QPixmap.fromImage(th_qimg))
            self.filmstrip.update_photo_data(photo_id, target_photo)
            try:
                session_cache_manager.save_photo_cache(photo_path, target_photo, save_preview=False)
            except Exception:
                pass

        # Only update GUI if this is still the active photo
        if self.active_photo_id == photo_id:
            fn = target_photo.get("filename", os.path.basename(photo_path)) if target_photo else os.path.basename(photo_path)
            try:
                if "base_ev" in res:
                    new_bev = float(res["base_ev"])
                    if "base_ev" not in self.current_params or abs(float(self.current_params.get("base_ev", 0.0))) < 0.0001:
                        self._active_base_ev = new_bev
                        self.current_params["base_ev"] = new_bev
                if res.get("exif"):
                    exif_meta = res["exif"]
                    if hasattr(self, "exif_widget"):
                        self.exif_widget.set_exif_data(exif_meta)
                    if hasattr(self, "histogram_exif_bar"):
                        self.histogram_exif_bar.set_exif_data(exif_meta)
                    self._active_base_temp = float(exif_meta.get("color_temp", 5500.0))
                    self._active_base_tint = float(exif_meta.get("tint", 1.0))
                    self.current_params["base_temp"] = self._active_base_temp
                    self.current_params["base_tint"] = self._active_base_tint
                    if not self.current_params.get("custom_wb", False):
                        if "color_temp" not in self.current_params or self.current_params.get("color_temp") == 5500.0:
                            self.current_params["color_temp"] = self._active_base_temp
                        if "tint" not in self.current_params or self.current_params.get("tint") == 1.0:
                            self.current_params["tint"] = self._active_base_tint
                        if hasattr(self, "slider_temp"):
                            self.slider_temp.set_default_value(self._active_base_temp)
                            self._is_updating_sliders = True
                            try:
                                self.slider_temp.set_value(self.current_params["color_temp"])
                            finally:
                                self._is_updating_sliders = False
                        if hasattr(self, "slider_tint"):
                            self.slider_tint.set_default_value(self._active_base_tint)
                            self._is_updating_sliders = True
                            try:
                                self.slider_tint.set_value(self.current_params["tint"])
                            finally:
                                self._is_updating_sliders = False
                    lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
                    if lens_cfg.get("setup", "auto") == "auto":
                        self._auto_detect_lens_profile()
                    from core.lens_correction import compute_effective_lens_params
                    eff_lens = compute_effective_lens_params(
                        lens_cfg,
                        focal_length=exif_meta.get("focal_length"),
                        aperture=exif_meta.get("aperture"),
aspect_ratio=self._lens_frame_aspect(target_photo),
                        source_path=(target_photo or {}).get("path")
                    )
                    lens_cfg.update(eff_lens)

                self._sync_canvas_params()

                m_w = target_photo.get("width") or (target_photo.get("meta", {}).get("width") if isinstance(target_photo.get("meta"), dict) else None) if target_photo else None
                m_h = target_photo.get("height") or (target_photo.get("meta", {}).get("height") if isinstance(target_photo.get("meta"), dict) else None) if target_photo else None
                self.canvas.set_image(float_img, master_width=m_w, master_height=m_h, reset_zoom=False, photo_id=photo_id)

                self._manage_photo_memory_lru(photo_id)

                # Update live 2:3 thumbnails on cards (background worker will downsample as needed)
                self._current_photo_thumb = target_photo.get("thumbnail_rgb") if target_photo else None
                self._current_photo_linear = float_img

                if hasattr(self, '_cards_debounce_timer'):
                    self._cards_debounce_timer.start(250)

                self._schedule_histogram_update()
                self._schedule_active_thumbnail_update(150)
                self._update_photo_info_display()
            except Exception as e:
                logger.error(f"Error updating GUI for loaded photo {photo_id}: {e}")
            finally:
                if not getattr(self, "_is_multi_selecting", False):
                    self.set_app_status(f"底片已载入：{fn}", spinning=False, timeout_ms=2000)

    def _card_paper_for(self, film, current_paper, default_papers=None):
        """这张胶卷卡片该配哪张纸：本卷用得了当前纸就用它，否则用这支卷所在链的默认纸。

        卡片墙是"换这支卷会是什么样"的橱窗，所以每张卡必须按它自己的链出图。以前是一个
        值发给所有卷：当前纸是 Portra Endura 时，整排转换卷卡片都烘成 2383（实测与 2383
        逐位相同，max|d| = 0.0），用户看到的是一排假预览。

        `default_papers`（`_default_paper_pair()`）由一轮刷新共用：当前纸是只对转换卷开放
        的那种（Eterna CP / CP30 …）时，每张原生卷卡片都会落到这里，逐张去读偏好配置会让
        一次点击卡 1 s 以上。
        """
        paper = str(current_paper or "none")
        if paper in ("none", ""):
            return paper or "none"
        if self._paper_allowed_now(paper, film=film):
            return paper
        if default_papers is not None:
            chain = self._paper_scope_key(film)
            if chain in default_papers:
                return default_papers[chain]
        return self._default_paper_for(film)

    def _update_card_live_thumbnails(self, thumb_data, do_films=True, do_papers=True):
        if thumb_data is None:
            return

        # Synchronize candidate card previews with active crop box and rotation
        crop = self.current_params.get("crop", {})
        if isinstance(crop, dict) and crop.get("enabled", False) and thumb_data is not None:
            try:
                x = max(0.0, min(0.99, float(crop.get("x", 0.0))))
                y = max(0.0, min(0.99, float(crop.get("y", 0.0))))
                w = max(0.01, min(1.0 - x, float(crop.get("w", 1.0))))
                h = max(0.01, min(1.0 - y, float(crop.get("h", 1.0))))
                ih, iw = thumb_data.shape[:2]
                x0 = int(round(x * iw))
                y0 = int(round(y * ih))
                w0 = max(1, int(round(w * iw)))
                h0 = max(1, int(round(h * ih)))
                thumb_data = thumb_data[y0:y0+h0, x0:x0+w0]
                rot_90 = int(crop.get("rot_90", 0)) % 360
                if rot_90 == 90:
                    thumb_data = cv2.rotate(thumb_data, cv2.ROTATE_90_CLOCKWISE)
                elif rot_90 == 180:
                    thumb_data = cv2.rotate(thumb_data, cv2.ROTATE_180)
                elif rot_90 == 270:
                    thumb_data = cv2.rotate(thumb_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if crop.get("flip_h", False):
                    thumb_data = cv2.flip(thumb_data, 1)
                if crop.get("flip_v", False):
                    thumb_data = cv2.flip(thumb_data, 0)
            except Exception as e:
                logger.error(f"Error cropping thumbnail for card previews: {e}")

        if do_films:
            # Prioritize filter-matched cards currently visible to user (e.g. 11 cine cards instead of all 51)
            matched_films = {k: v for k, v in self.film_cards.items() if getattr(v, "is_filter_matched", True)}
            film_cards_dict = matched_films if matched_films else self.film_cards
        else:
            film_cards_dict = {}
        # 置灰（跨链不可选）的相纸卡片不烘预览：那张图在物理上不存在，烘出来只会是
        # 引擎夹取后的另一张纸（实测：原生卷下四张点式纸的卡片画面逐位相同），
        # 既误导用户又白烧 LUT。保留它们上一帧的图，比给一张假的强。
        paper_cards_dict = ({k: v for k, v in self.paper_cards.items() if v.isEnabled()}
                            if do_papers else {})

        params = dict(self.current_params)
        params["base_temp"] = getattr(self, "_active_base_temp", 5500.0)
        params["base_tint"] = getattr(self, "_active_base_tint", 1.0)
        params["print_mode"] = getattr(self, "current_print_mode", "optical")

        # 胶卷卡片各自配一张纸来印。旁通（film=none）会把当前相纸强制成"直扫"，若照它
        # 去配每一支卷，整片网格都是橙色负像 —— 即使首选项里明明选了默认相纸。只有用户
        # 明确选了直扫（paper_explicit）才保留直扫预览。
        preview_paper = self.current_paper_stock
        if str(preview_paper or "none") in ("none", ""):
            active = next((p for p in getattr(self, "photos", [])
                           if p and p.get("id") == self.active_photo_id), None)
            if active is None and self._current_photo_path:
                active = next((p for p in self.photos
                               if p and p.get("path") == self._current_photo_path), None)
            if not self._photo_paper_is_explicit(active):
                preview_paper = self._default_paper_for(self.current_film_stock)

        # 一轮刷新只读一次两条链的默认纸：卡片墙是逐张配纸的，以前每张卡都去枚举一遍
        # 相纸档案（122 ms），选中 Eterna CP 这类"只对转换卷开放"的纸时一次点击卡 1 s+。
        film_paper = {}
        if film_cards_dict:
            default_papers = self._default_paper_pair()
            film_paper = {sid: self._card_paper_for(sid, preview_paper, default_papers)
                          for sid in film_cards_dict}

        self.thumb_controller.trigger_candidate_card_previews(
            thumb_data,
            film_cards_dict,
            paper_cards_dict,
            self.current_film_stock,
            self.current_paper_stock,
            params,
            on_film_ready=self._on_film_thumb_ready,
            on_paper_ready=self._on_paper_thumb_ready,
            film_paper=film_paper
        )

    def _refresh_card_previews(self, film_only=False, paper_only=False):
        thumb_data = getattr(self, '_current_photo_linear', None)
        if thumb_data is None and self.active_photo_id:
            curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if curr_p:
                thumb_data = self.thumb_controller.get_raw_thumbnail(curr_p)
        if thumb_data is None:
            thumb_data = getattr(self, '_current_photo_thumb', None)
        if thumb_data is None:
            return
        do_films = not paper_only
        do_papers = not film_only
        self._update_card_live_thumbnails(thumb_data, do_films=do_films, do_papers=do_papers)

    def _on_film_thumb_ready(self, stock_id, pixmap):
        card = self.film_cards.get(stock_id)
        if not card:
            return
        card.set_thumbnail(pixmap)

    def _on_paper_thumb_ready(self, stock_id, pixmap):
        card = self.paper_cards.get(stock_id)
        if not card:
            return
        card.set_thumbnail(pixmap)

    def on_remove_photo(self, photo_id, push_undo=True):
        idx = next((i for i, p in enumerate(self.photos) if p and p.get("id") == photo_id), -1)
        if idx == -1:
            return

        # Synchronize current active photo state into self.photos first if dirty
        if self.active_photo_id:
            curr_target_p = next((x for x in self.photos if x and x.get("id") == self.active_photo_id), None)
            if curr_target_p:
                curr_target_p["film_profile"] = self.current_film_stock
                curr_target_p["paper_profile"] = self.current_paper_stock
                curr_target_p["print_mode"] = getattr(self, "current_print_mode", "optical")
                curr_target_p["params"] = copy.deepcopy(self.current_params)
                curr_target_p["is_dirty"] = getattr(self, "_photo_is_dirty", False)

        pre_state = None
        if push_undo:
            pre_state = {
                "type": "library_photos_state",
                "action": "remove_photos",
                "photos": [self._snapshot_photo_entry(p) for p in self.photos],
                "active_photo_id": self.active_photo_id,
                "session_path": self._current_session_path,
                "session_is_dirty": getattr(self, "_session_is_dirty", False),
            }
            self._push_raw_state(pre_state)

        # Determine next photo to activate:
        # Standard professional darkroom behavior: switch to the photo immediately following the deleted one.
        # If the deleted photo was the last one, switch to the new last photo (idx - 1).
        next_id = None
        if len(self.photos) > 1:
            if idx + 1 < len(self.photos) and self.photos[idx + 1]:
                next_id = self.photos[idx + 1]["id"]
            elif idx > 0 and self.photos[idx - 1]:
                next_id = self.photos[idx - 1]["id"]

        self.photos = [p for p in self.photos if p and p.get("id") != photo_id]
        if hasattr(self, 'engine'):
            self.engine.remove_photo(photo_id)
        self._session_is_dirty = True
        self._update_window_title()

        if not self.photos:
            self.action_clear_photos(push_undo=False)
        else:
            self.filmstrip.remove_photo(photo_id, next_active_id=next_id)
            self.on_switch_photo(next_id)

        if push_undo and pre_state:
            post_state = {
                "type": "library_photos_state",
                "action": "remove_photos",
                "photos": [self._snapshot_photo_entry(p) for p in self.photos],
                "active_photo_id": self.active_photo_id,
                "session_path": self._current_session_path,
                "session_is_dirty": getattr(self, "_session_is_dirty", False),
            }
            self._push_raw_state(post_state)
        self._update_menu_actions_state()

    def on_remove_multiple_photos(self, photo_ids, push_undo=True):
        id_set = set(photo_ids)
        if not id_set:
            return

        # Synchronize current active photo state into self.photos first if dirty
        if self.active_photo_id:
            curr_target_p = next((x for x in self.photos if x and x.get("id") == self.active_photo_id), None)
            if curr_target_p:
                curr_target_p["film_profile"] = self.current_film_stock
                curr_target_p["paper_profile"] = self.current_paper_stock
                curr_target_p["print_mode"] = getattr(self, "current_print_mode", "optical")
                curr_target_p["params"] = copy.deepcopy(self.current_params)
                curr_target_p["is_dirty"] = getattr(self, "_photo_is_dirty", False)

        pre_state = None
        if push_undo:
            pre_state = {
                "type": "library_photos_state",
                "action": "remove_photos",
                "photos": [self._snapshot_photo_entry(p) for p in self.photos],
                "active_photo_id": self.active_photo_id,
                "session_path": self._current_session_path,
                "session_is_dirty": getattr(self, "_session_is_dirty", False),
            }
            self._push_raw_state(pre_state)

        # Find surviving candidate to activate following deletion:
        # 1. If currently active photo is NOT in id_set and survives, retain it as active.
        # 2. If currently active photo IS in id_set (or no active photo):
        #    Anchor at active photo index (or the last deleted photo index) and search forward
        #    for the first surviving photo following it.
        # 3. If no surviving photo follows the anchor, search backward for the closest surviving photo.
        active_idx = next((i for i, p in enumerate(self.photos) if p and p.get("id") == self.active_photo_id), -1)

        next_id = None
        if self.active_photo_id and self.active_photo_id not in id_set:
            next_id = self.active_photo_id
        else:
            del_indices = [i for i, p in enumerate(self.photos) if p and p.get("id") in id_set]
            anchor_idx = active_idx if active_idx != -1 else (max(del_indices) if del_indices else -1)

            if anchor_idx != -1:
                # Search forward for the first photo not being deleted
                for p in self.photos[anchor_idx + 1:]:
                    if p and p.get("id") not in id_set:
                        next_id = p["id"]
                        break
                # If no photo after anchor survives, search backward
                if not next_id:
                    for p in reversed(self.photos[:anchor_idx]):
                        if p and p.get("id") not in id_set:
                            next_id = p["id"]
                            break

        self.photos = [p for p in self.photos if p and p.get("id") not in id_set]
        if hasattr(self, 'engine'):
            for pid in photo_ids:
                self.engine.remove_photo(pid)
        self._session_is_dirty = True
        self._update_window_title()

        if not self.photos:
            self.action_clear_photos(push_undo=False)
        else:
            if not next_id and self.photos and self.photos[0]:
                next_id = self.photos[0]["id"]
            self.filmstrip.remove_multiple_photos(photo_ids, next_active_id=next_id)
            self.on_switch_photo(next_id)

        if push_undo and pre_state:
            post_state = {
                "type": "library_photos_state",
                "action": "remove_photos",
                "photos": [self._snapshot_photo_entry(p) for p in self.photos],
                "active_photo_id": self.active_photo_id,
                "session_path": self._current_session_path,
                "session_is_dirty": getattr(self, "_session_is_dirty", False),
            }
            self._push_raw_state(post_state)
        self._update_menu_actions_state()

    def on_delete_multiple_photos(self, photo_ids):
        """Prompt user with confirmation warning and delete selected photo files to Recycle Bin."""
        if not photo_ids:
            return
        target_photos = [p for p in self.photos if p and p.get("id") in photo_ids]
        if not target_photos:
            return

        target_names = [p.get("filename") or os.path.basename(p.get("path", "")) for p in target_photos]

        from ui.delete_photo_dialog import DeletePhotoConfirmDialog
        dlg = DeletePhotoConfirmDialog(target_names, self)
        if dlg.exec() != DeletePhotoConfirmDialog.CONFIRMED:
            return

        from core.path_utils import send_to_recycle_bin
        files_to_recycle = []
        for p in target_photos:
            fpath = p.get("path")
            if fpath and os.path.isfile(fpath):
                files_to_recycle.append(fpath)
                sdc_path = fpath + ".sdc"
                if os.path.isfile(sdc_path):
                    files_to_recycle.append(sdc_path)

        recycled_ok = send_to_recycle_bin(files_to_recycle)
        if not recycled_ok:
            self.set_app_status("删除失败：无法移入回收站", timeout_ms=3000)
            return

        from core import session_cache_manager
        for p in target_photos:
            fpath = p.get("path")
            if fpath:
                session_cache_manager.invalidate_photo_cache(fpath)

        count = len(target_photos)
        self.on_remove_multiple_photos(photo_ids, push_undo=False)

        # Purge deleted photos from prior undo/redo history so undo cannot revive recycled files
        id_set = set(photo_ids)
        new_undo = []
        for s in self._undo_stack:
            stype = s.get("type")
            if stype == "single_photo_state" and s.get("photo_id") in id_set:
                continue
            if stype == "library_photos_state":
                s["photos"] = [p for p in s.get("photos", []) if p.get("id") not in id_set]
            elif stype == "batch_photos_state":
                for pid in id_set:
                    s.get("photos", {}).pop(pid, None)
            elif stype == "photo_rating_state":
                for pid in id_set:
                    s.get("ratings", {}).pop(pid, None)
                    s.get("old_ratings", {}).pop(pid, None)
            new_undo.append(s)
        self._undo_stack = new_undo
        self._redo_stack.clear()
        if not self._undo_stack and self.active_photo_id and getattr(self, "_active_photo_baseline_state", None):
            self._undo_stack = [self._active_photo_baseline_state]
        self._update_menu_actions_state()
        self.set_app_status(f"已将 {count} 个文件移入回收站", timeout_ms=3000)

    def action_clear_sdc_active(self):
        """Action handler for menu bar: Delete SDC sidecar recipe and revert photo(s) to pristine RAW."""
        sel_ids = self.filmstrip.get_selected_photo_ids() if hasattr(self, 'filmstrip') else []
        if not sel_ids and self.active_photo_id:
            sel_ids = [self.active_photo_id]
        if sel_ids:
            self.on_clear_sdc_config(sel_ids)

    def on_clear_sdc_config(self, photo_ids):
        """Delete .sdc configuration sidecar, revert filmstrip thumbnails to raw base,
        and thoroughly reset sidebar cards to clean brand color pills.
        Supports full undo/redo across single or batch/all selected photos.
        """
        id_set = set(photo_ids)
        target_photos = [p for p in self.photos if p and p.get("id") in id_set]
        if not target_photos:
            return

        # 1. Capture comprehensive pre-clear state snapshot for all target photos
        pre_photos = {}
        for p in target_photos:
            pid = p["id"]
            if pid == self.active_photo_id:
                p_film = getattr(self, "current_film_stock", "none")
                p_paper = getattr(self, "current_paper_stock", "none")
                p_pmode = getattr(self, "current_print_mode", "optical")
                p_params = copy.deepcopy(self.current_params)
                p_dirty = bool(getattr(self, "_photo_is_dirty", False) or p.get("is_dirty", False))
                p_edited = bool(self._is_photo_edited(p))
            else:
                p_film = p.get("film_profile", "none")
                p_paper = p.get("paper_profile", "none")
                p_pmode = p.get("print_mode", "optical")
                p_params = copy.deepcopy(p.get("params") or {})
                if not p_params and p.get("path"):
                    sdc = sdc_manager.load_sdc(p["path"])
                    if sdc:
                        p_film = sdc.get("film_profile", p_film)
                        p_paper = sdc.get("paper_profile", p_paper)
                        p_pmode = sdc.get("print_mode", p_pmode)
                        p_params = dict(sdc.get("params") or {})
                p_dirty = bool(p.get("is_dirty", False))
                p_edited = bool(self._is_photo_edited(p))

            pre_photos[pid] = {
                "film": p_film,
                "paper": p_paper,
                "print_mode": p_pmode,
                "params": p_params,
                "is_dirty": p_dirty,
                "is_edited": p_edited,
                "path": p.get("path"),
            }

        pre_state = {
            "type": "batch_photos_state",
            "action": "clear_sdc",
            "active_photo_id": self.active_photo_id,
            "photos": pre_photos,
        }
        self._push_raw_state(pre_state)

        # 2. Clear .sdc and reset state for all target photos
        default_crop = {
            "enabled": False,
            "x": 0.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "angle": 0.0,
            "rot_90": 0,
            "flip_h": False,
            "flip_v": False,
            "aspect_ratio": "original",
            "is_portrait": False
        }
        active_cleared = False
        post_photos = {}
        # 清掉 .sdc 后胶卷回到"原生底片"，相纸不该跟着掉到"胶片直扫"：它跟首选项的负片
        # 默认相纸走（用户指定）。film=none 走旁通表，相纸在这里不影响渲染，但面板状态和
        # 接下来选胶卷时的落点都按这张纸来。
        reset_paper = self._default_paper_for("none")
        reset_mode = "scan" if str(reset_paper or "none") in ("none", "") else "optical"
        for p in target_photos:
            pid = p["id"]
            p_path = p.get("path")
            if p_path:
                sdc_manager.delete_sdc(p_path)
            p.pop("params", None)
            p["film_profile"] = "none"
            p["crop"] = dict(default_crop)
            p["is_dirty"] = False
            p["is_edited"] = False

            # Instantly revert filmstrip thumbnail to raw base
            if hasattr(self, 'thumb_controller'):
                self.thumb_controller.revert_photo_to_raw(p, self.filmstrip)
            if hasattr(self, 'filmstrip'):
                self.filmstrip.set_photo_dirty(pid, False)
                self.filmstrip.set_photo_crop(pid, False)

            # revert_photo_to_raw 会把相纸写成 "none"，这里再按首选项落一次。空串表示
            # "没选过相纸"，以后改首选项这张照片还会跟着变。
            p["paper_profile"] = ""
            p["print_mode"] = reset_mode

            if pid == self.active_photo_id:
                active_cleared = True

            post_photos[pid] = {
                "film": "none",
                "paper": "",
                "print_mode": reset_mode,
                "params": {},
                "is_dirty": False,
                "is_edited": False,
                "path": p_path,
            }

        self._update_window_title()

        # 3. If active photo was cleared, reset live controls, canvas, and cards
        if active_cleared and self.active_photo_id:
            if getattr(self.canvas, "is_crop_editing", False):
                self._exit_crop_workspace()

            self._photo_is_dirty = False
            self.current_film_stock = "none"
            self.current_paper_stock = reset_paper
            self.current_print_mode = reset_mode

            base_temp = float(getattr(self, "_active_base_temp", 5500.0))
            base_tint = float(getattr(self, "_active_base_tint", 1.0))
            defaults = self.get_default_params(base_temp, base_tint)
            self.current_params.clear()
            self.current_params.update(defaults)
            self.current_params["print_mode"] = reset_mode
            self.current_params["crop"] = dict(default_crop)
            # Clearing the sidecar must not throw away the metering anchor: base_ev is derived
            # from the file, not a saved edit, and dropping it here silently shifted the
            # exposure by up to 3 stops (that is why "delete the .sdc" appeared to fix a
            # brightness problem that had nothing to do with the sidecar).
            self.current_params["base_ev"] = float(getattr(self, "_active_base_ev", 0.0))

            active_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if active_p:
                active_p["params"] = None
                active_p["film_profile"] = "none"
                active_p["paper_profile"] = ""
                active_p["print_mode"] = reset_mode
                active_p["crop"] = dict(default_crop)
                active_p["is_dirty"] = False
                active_p["is_edited"] = False

            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                self.canvas.crop_overlay.reset_all_crop()

            self.canvas.update_params(crop=dict(default_crop))
            self.canvas.fit_to_view(animate=False)

            if self.active_photo_id in post_photos:
                post_photos[self.active_photo_id]["params"] = None

            self._update_all_sliders_from_params()
            self._update_reset_buttons_state()
            self._sync_full_params_to_pipeline(fit_crop_if_changed=True)
            self._update_window_title()

            for sid, card in self.film_cards.items():
                card.set_active(sid == "none")
            for sid, card in self.paper_cards.items():
                card.set_active(sid == reset_paper)
                if sid != reset_paper:
                    card.set_thumbnail(None)

            self._update_active_thumbnails()
            self._refresh_card_previews()
            self._update_stock_mode_ui()

        # 4. Push post-clear state snapshot
        post_state = {
            "type": "batch_photos_state",
            "action": "clear_sdc",
            "active_photo_id": self.active_photo_id,
            "photos": post_photos,
        }
        self._push_raw_state(post_state)
        self._session_is_dirty = True
        self._update_window_title()
        self._save_auto_temp_session()

        if active_cleared and len(id_set) == 1:
            self.set_app_status("已清除暗房配置，恢复为原生底片", timeout_ms=3000)
        else:
            self.set_app_status(f"已清除 {len(id_set)} 张底片的暗房配置，恢复为原生底片", timeout_ms=3000)

    def action_clear_active_crop(self):
        """Clear crop on the currently active photo or selected photo(s)."""
        sel_ids = self.filmstrip.get_selected_photo_ids() if hasattr(self, 'filmstrip') else []
        if not sel_ids and self.active_photo_id:
            sel_ids = [self.active_photo_id]
        if sel_ids:
            self.on_clear_crop_photos(sel_ids)

    def on_clear_crop_photos(self, photo_ids):
        """Clear crop configuration for target photo(s) without altering develop parameters."""
        id_set = set(photo_ids)
        target_photos = [p for p in self.photos if p and p.get("id") in id_set]
        if not target_photos:
            return

        default_crop = {
            "enabled": False,
            "x": 0.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "angle": 0.0,
            "rot_90": 0,
            "flip_h": False,
            "flip_v": False,
            "aspect_ratio": "original",
            "is_portrait": False
        }

        # Check if at least one photo has an active/non-default crop
        has_any_crop = False
        for p in target_photos:
            c = p.get("crop")
            if not c and isinstance(p.get("params"), dict):
                c = p["params"].get("crop")
            if not c and p.get("id") == self.active_photo_id:
                c = self.current_params.get("crop")
            if self._is_crop_modified(c):
                has_any_crop = True
                break
        if not has_any_crop:
            return

        # 1. Capture comprehensive pre-clear state snapshot for all target photos
        pre_photos = {}
        for p in target_photos:
            pid = p["id"]
            if pid == self.active_photo_id:
                p_film = getattr(self, "current_film_stock", "none")
                p_paper = getattr(self, "current_paper_stock", "none")
                p_pmode = getattr(self, "current_print_mode", "optical")
                p_params = dict(self.current_params)
                p_crop = dict(self.current_params.get("crop") or p.get("crop") or default_crop)
                p_params["crop"] = dict(p_crop)
                p_dirty = bool(getattr(self, "_photo_is_dirty", False) or p.get("is_dirty", False))
                p_edited = bool(self._is_photo_edited(p))
            else:
                p_film = p.get("film_profile", "none")
                p_paper = p.get("paper_profile", "none")
                p_pmode = p.get("print_mode", "optical")
                p_params = dict(p.get("params") or {})
                if not p_params and p.get("path"):
                    sdc = sdc_manager.load_sdc(p["path"])
                    if sdc:
                        p_film = sdc.get("film_profile", p_film)
                        p_paper = sdc.get("paper_profile", p_paper)
                        p_pmode = sdc.get("print_mode", p_pmode)
                        p_params = dict(sdc.get("params") or {})
                p_crop = dict(p_params.get("crop") or p.get("crop") or default_crop)
                p_params["crop"] = dict(p_crop)
                p_dirty = bool(p.get("is_dirty", False))
                p_edited = bool(self._is_photo_edited(p))

            pre_photos[pid] = {
                "film": p_film,
                "paper": p_paper,
                "print_mode": p_pmode,
                "params": p_params,
                "crop": p_crop,
                "is_dirty": p_dirty,
                "is_edited": p_edited,
                "path": p.get("path"),
            }

        pre_state = {
            "type": "batch_photos_state",
            "action": "clear_crop",
            "active_photo_id": self.active_photo_id,
            "photos": pre_photos,
        }
        self._push_raw_state(pre_state)

        # 2. Clear crop for all target photos
        active_cleared = False
        post_photos = {}
        for p in target_photos:
            pid = p["id"]
            if "params" not in p or not isinstance(p["params"], dict):
                p["params"] = {}
            p["params"]["crop"] = dict(default_crop)
            p["crop"] = dict(default_crop)

            is_film = p.get("film_profile", "none") not in ("none", "", None)
            is_paper = p.get("paper_profile", "none") not in ("none", "", None)
            is_params = self._is_params_modified(p["params"], p)
            has_other_edits = is_film or is_paper or is_params

            p_path = p.get("path", "")
            if has_other_edits:
                p["is_dirty"] = True
                p["is_edited"] = True
                if p_path:
                    try:
                        cur_thumb = p.get("graded_thumbnail_rgb")
                        if cur_thumb is None:
                            cur_thumb = p.get("thumbnail_rgb")
                        sdc_manager.save_sdc(
                            p_path,
                            p.get("film_profile", "none"),
                            p.get("paper_profile", "none"),
                            p["params"],
                            thumbnail_rgb=cur_thumb,
                            print_mode=p.get("print_mode", "optical")
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save SDC for {p_path}: {e}")
            else:
                p["is_dirty"] = False
                p["is_edited"] = False
                if p_path and sdc_manager.has_sdc(p_path):
                    try:
                        sdc_manager.delete_sdc(p_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete SDC for {p_path}: {e}")

            if hasattr(self, "filmstrip"):
                self.filmstrip.set_photo_crop(pid, dict(default_crop))
                self.filmstrip.set_photo_dirty(pid, p["is_dirty"])

            if pid == self.active_photo_id:
                active_cleared = True
            else:
                if hasattr(self, "thumb_controller"):
                    if not has_other_edits:
                        self.thumb_controller.revert_photo_to_raw(p, self.filmstrip)
                    else:
                        raw_thumb = self.thumb_controller.get_raw_thumbnail(p)
                        if raw_thumb is not None and raw_thumb.size > 0:
                            graded = self.thumb_controller.grade_thumbnail_fast(
                                raw_thumb,
                                film=p.get("film_profile", "none"),
                                paper=p.get("paper_profile", "none"),
                                params=p.get("params", {})
                            )
                            self.thumb_controller.update_graded_thumbnail(p, graded, self.filmstrip)

            post_photos[pid] = {
                "film": p.get("film_profile", "none"),
                "paper": p.get("paper_profile", "none"),
                "print_mode": p.get("print_mode", "optical"),
                "params": dict(p.get("params") or {}),
                "crop": dict(default_crop),
                "is_dirty": p.get("is_dirty", False),
                "is_edited": p.get("is_edited", False),
                "path": p_path,
            }

        # 3. If active photo was cleared, synchronize live controls, canvas, and cards
        if active_cleared:
            if getattr(self.canvas, "is_crop_editing", False):
                self._exit_crop_workspace()

            self.current_params["crop"] = dict(default_crop)
            if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
                self.canvas.params["crop"] = dict(default_crop)
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                self.canvas.crop_overlay.reset_all_crop()

            self.canvas.update_params(crop=dict(default_crop))
            self.canvas.fit_to_view(animate=False)

            active_p = next((p for p in self.photos if p.get("id") == self.active_photo_id), None)
            if active_p:
                self._photo_is_dirty = active_p.get("is_dirty", False)
                if self.active_photo_id in post_photos:
                    post_photos[self.active_photo_id]["is_dirty"] = self._photo_is_dirty
                    post_photos[self.active_photo_id]["is_edited"] = active_p.get("is_edited", False)
                    post_photos[self.active_photo_id]["params"] = dict(self.current_params)

            self._save_current_photo_sdc()
            self._update_active_thumbnails()
            if hasattr(self, '_cards_debounce_timer'):
                self._cards_debounce_timer.start(100)
            else:
                self._refresh_card_previews()

        # 4. Push post-clear state snapshot
        post_state = {
            "type": "batch_photos_state",
            "action": "clear_crop",
            "active_photo_id": self.active_photo_id,
            "photos": post_photos,
        }
        self._push_raw_state(post_state)

        count = len(target_photos)
        msg = "已清除裁切" if count == 1 else f"已清除 {count} 张底片的裁切"
        self.set_app_status(msg, timeout_ms=2000)

    def action_clear_photos(self, push_undo=True):
        """Completely flush bottom filmstrip, engine session, and canvas without exiting to start screen."""
        self._cancel_active_import(notify_user=False)
        self._photo_id_counter = 0

        pre_state = None
        if push_undo and self.photos:
            if self.active_photo_id:
                curr_target_p = next((x for x in self.photos if x and x.get("id") == self.active_photo_id), None)
                if curr_target_p:
                    curr_target_p["film_profile"] = self.current_film_stock
                    curr_target_p["paper_profile"] = self.current_paper_stock
                    curr_target_p["print_mode"] = getattr(self, "current_print_mode", "optical")
                    curr_target_p["params"] = copy.deepcopy(self.current_params)
                    curr_target_p["is_dirty"] = getattr(self, "_photo_is_dirty", False)

            pre_state = {
                "type": "library_photos_state",
                "action": "clear_photos",
                "photos": [self._snapshot_photo_entry(p) for p in self.photos],
                "active_photo_id": self.active_photo_id,
                "session_path": self._current_session_path,
                "session_is_dirty": getattr(self, "_session_is_dirty", False),
            }
            self._push_raw_state(pre_state)

        if self._photo_is_dirty and self._current_photo_path:
            sdc_manager.save_sdc(
                self._current_photo_path,
                self.current_film_stock,
                self.current_paper_stock,
                self.current_params,
                print_mode=getattr(self, "current_print_mode", "optical")
            )
        self._photo_is_dirty = False
        if hasattr(self, 'engine'):
            self.engine.clear_all_photos()
        self.photos.clear()
        self.active_photo_id = None
        self._current_photo_path = None

        has_named_session = bool(self._current_session_path and not getattr(self, "_is_temp_session", False))
        if has_named_session:
            # Retain current session identity and mark dirty (clearing photos modifies the session)
            self._session_is_dirty = True
            config_manager.save_last_session_path(self._current_session_path)
        else:
            self._current_session_path = None
            self._session_is_dirty = False
            self._is_temp_session = True
            config_manager.clear_auto_saved_session()
            config_manager.save_last_session_path(None)
            config_manager.save_session_state([], None)

        # Retain darkroom workspace, do not return to start page
        self.canvas.set_has_library_photos(False)
        self.canvas.set_image(None)
        self.filmstrip.set_photos([], None)
        if hasattr(self, 'canvas_rating'):
            if hasattr(self, 'view_toolbar') and hasattr(self.view_toolbar, 'set_rating_enabled'):
                self.view_toolbar.set_rating_enabled(False)
            else:
                self.canvas_rating.setVisible(False)
            self.canvas_rating.set_rating(0)

        self.btn_title_export.setVisible(False)
        self._update_window_title()
        if hasattr(self, 'exif_widget'):
            self.exif_widget.set_exif_data({})
        if hasattr(self, 'histogram_exif_bar'):
            self.histogram_exif_bar.set_exif_data({})
        if hasattr(self, 'histogram_widget'):
            self.histogram_widget.clear()
        elif hasattr(self, 'histogram'):
            self.histogram.clear()

        # Item 6: Reset cards thumbnails to default swatch
        self._current_photo_thumb = None
        self._current_photo_linear = None
        if hasattr(self, "_loaded_photos_lru"):
            self._loaded_photos_lru.clear()
        self.thumb_controller.reset_all_cards_to_color_pills(self.film_cards, self.paper_cards)
        self.restore_default_status()
        import gc
        gc.collect()

        if push_undo and pre_state:
            post_state = {
                "type": "library_photos_state",
                "action": "clear_photos",
                "photos": [],
                "active_photo_id": None,
                "session_path": self._current_session_path,
                "session_is_dirty": getattr(self, "_session_is_dirty", False),
            }
            self._push_raw_state(post_state)
        self._update_menu_actions_state()

    def action_export_image(self):
        # 1. Check filmstrip multi-selection
        selected_ids = self.filmstrip.get_selected_ids() if hasattr(self, "filmstrip") else []
        if len(selected_ids) > 1:
            self._batch_export_photos(selected_ids)
            return

        # 2. Single export
        if not self._current_photo_path:
            return

        last_dir = config_manager.get_last_export_dir()
        if not last_dir or not os.path.exists(last_dir):
            last_dir = os.path.dirname(self._current_photo_path) if self._current_photo_path else os.path.expanduser("~")

        last_cfg = config_manager.get_last_export_settings()
        fmt = str(last_cfg.get("format", "jpeg")).lower() if last_cfg else "jpeg"
        ext = ".tif" if "tif" in fmt else (".jxl" if "jxl" in fmt else (".png" if "png" in fmt else ".jpg"))

        base_name = os.path.splitext(os.path.basename(self._current_photo_path))[0]
        default_out = os.path.join(last_dir, f"{base_name}{ext}")

        dlg = SingleExportDialog(default_out, parent=self, source_path=self._current_photo_path)
        self._seed_export_size_dialog(dlg, self._current_photo_path)
        if dlg.exec():
            cfg = dlg.get_export_config()
            self._sync_proof_from_export(cfg.get("colorspace"))
            self._do_export_single(cfg)

    def _resolve_export_target(self, out_path, src_path=None, used=None, allow_overwrite=True):
        """Resolve the final path an export may write to.

        Three guarantees, in order of importance:
        1. The digital negative is never overwritten. When the computed target would
           coincide with ``src_path``, a ``_dev`` suffix is appended first.
        2. No two exports in the same batch share a path (``used`` is updated in place).
        3. With ``allow_overwrite=False`` an already existing file is skipped by
           appending 1, 2, 3 ... before the extension.
        """
        if not out_path:
            return out_path
        used = used if used is not None else set()
        src_key = os.path.normcase(os.path.abspath(src_path)) if src_path else None

        base, ext = os.path.splitext(out_path)
        candidate = out_path
        if src_key and os.path.normcase(os.path.abspath(candidate)) == src_key:
            base = f"{base}_dev"
            candidate = f"{base}{ext}"

        counter = 1
        while True:
            cand_key = os.path.normcase(os.path.abspath(candidate))
            collision = cand_key in used
            if not collision and src_key and cand_key == src_key:
                collision = True
            if not collision and not allow_overwrite and os.path.exists(candidate):
                collision = True
            if not collision:
                break
            candidate = f"{base}_{counter}{ext}"
            counter += 1

        used.add(os.path.normcase(os.path.abspath(candidate)))
        return candidate

    def _seed_export_size_dialog(self, dlg, photo_path):
        """Give the export dialog the master resolution and crop of the photo(s) involved,
        so the size selector can show real output pixels instead of a bare percentage."""
        try:
            photo = next((p for p in self.photos if p and p.get("path") == photo_path), None)
            width = photo.get("width") if photo else None
            height = photo.get("height") if photo else None
            if not width or not height:
                if photo_path == self._current_photo_path and self.canvas._has_image:
                    width, height = self.canvas._master_width, self.canvas._master_height
            if not width or not height:
                # Non-RAW sources such as a freshly imported JPEG carry no cached master size.
                try:
                    from core.session_cache_manager import get_image_dimensions
                    dw, dh = get_image_dimensions(photo_path)
                    if dw > 0 and dh > 0:
                        width, height = dw, dh
                except Exception:
                    pass
            crop = None
            if photo_path == self._current_photo_path:
                crop = self.current_params.get("crop")
            elif isinstance(photo, dict) and isinstance(photo.get("params"), dict):
                crop = photo["params"].get("crop")
            if hasattr(dlg, "set_source_context"):
                dlg.set_source_context(width, height, crop)
        except Exception as e:
            logger.debug(f"Export size context unavailable: {e}")

    def _do_export_single(self, cfg, photo_path=None):
        """Queue one photo for export.

        ``photo_path`` lets a caller export a frame that is not the one on the canvas
        (right-click → 扫描当前底片 on a non-active thumbnail). The path stays a local
        here: mutating ``_current_photo_path`` used to desync the sidecar autosave, which
        then wrote the canvas recipe into the other photo's .sdc.
        """
        src_path = photo_path or self._current_photo_path
        if not src_path:
            return
        is_active = (src_path == self._current_photo_path)

        curr_p = next((p for p in self.photos if p and p.get("path") == src_path), None)
        base_ev = float(curr_p.get("base_ev", 0.0)) if curr_p else 0.0
        if abs(base_ev) < 0.0001 and is_active:
            base_ev = float(getattr(self, "_active_base_ev", 0.0))
        if abs(base_ev) < 0.0001 and hasattr(self, "engine"):
            matched_ep = next((p for p in self.engine.session_photos if p.get("path") == src_path), None)
            if matched_ep and "base_ev" in matched_ep:
                base_ev = float(matched_ep["base_ev"])

        if is_active:
            export_params = dict(self.current_params)
            film_id = self.current_film_stock
            paper_id = self.current_paper_stock
            base_temp = float(getattr(self, "_active_base_temp", 5500.0))
            base_tint = float(getattr(self, "_active_base_tint", 1.0))
        else:
            sdc_cfg = sdc_manager.load_sdc(src_path)
            exif_d = (curr_p or {}).get("exif") or {}
            base_temp = float(exif_d.get("color_temp", 5500.0) if exif_d else 5500.0)
            base_tint = float(exif_d.get("tint", 1.0) if exif_d else 1.0)
            export_params = self.get_default_params(base_temp, base_tint)
            if curr_p and curr_p.get("film_profile"):
                film_id = curr_p.get("film_profile")
                paper_id = curr_p.get("paper_profile", "none")
                export_params.update(curr_p.get("params") or {})
            elif sdc_cfg:
                film_id = sdc_cfg.get("film_profile", "none")
                paper_id = sdc_cfg.get("paper_profile", "none")
                export_params.update(sdc_cfg.get("params") or {})
            else:
                film_id, paper_id = "none", "none"

        export_params.update({
            "source_path": src_path,
            "film_stock": film_id,
            "paper_stock": paper_id,
            "print_mode": (curr_p or {}).get("print_mode") or getattr(self, "current_print_mode", "optical"),
            "base_ev": base_ev,
            "base_temp": base_temp,
            "base_tint": base_tint,
            "format": cfg.get("format", "jpeg"),
            "engine": cfg.get("engine", "wysiwyg"),
            "bit_depth": cfg.get("bit_depth", 8),
            "colorspace": cfg.get("colorspace", "sRGB"),
            "dpi": cfg.get("dpi", 300),
            "strip_gps": cfg.get("strip_gps", True),
            "quality": cfg.get("quality", 9),
            "scale_pct": cfg.get("scale_pct", 100),
            "out_size": list(cfg.get("out_size") or [0, 0]),
            "out_long_edge": cfg.get("out_long_edge", 0),
            "subsampling": cfg.get("subsampling", "4:4:4"),
            "progressive": cfg.get("progressive", True),
            "tiff_compression": cfg.get("tiff_compression", "lzw"),
            "png_level": cfg.get("png_level", 6),
            "jxl_mode": cfg.get("jxl_mode", "visual_10"),
            "jxl_lossless": cfg.get("jxl_lossless", False),
            "jxl_distance": cfg.get("jxl_distance", 1.0),
            "jxl_effort": cfg.get("jxl_effort", 7),
            # Viewport-only state must never reach a render: exporting while the compare
            # split is open used to bake a split-screen frame into the output file.
            "view_mode": 0,
            "is_thumbnail": False,
        })

        out_path = cfg["path"].strip() if cfg.get("path") else ""
        if str(cfg.get("path_mode", "custom")) == "source":
            # 输出到源文件所在目录，文件名沿用原名：路径由真实源文件算，不用对话框里的副本。
            base_name = os.path.splitext(os.path.basename(src_path))[0] if src_path else "image"
            fmt_str = str(cfg.get("format", "jpeg")).lower()
            ext = ".tif" if "tif" in fmt_str else (".jxl" if "jxl" in fmt_str else (".png" if "png" in fmt_str else ".jpg"))
            out_path = os.path.join(os.path.dirname(src_path) or os.path.expanduser("~"),
                                    f"{base_name}{ext}")
        if not out_path or os.path.isdir(out_path) or out_path.endswith(('/', '\\')):
            base_name = os.path.splitext(os.path.basename(src_path))[0] if src_path else "image"
            fmt_str = str(cfg.get("format", "jpeg")).lower()
            ext = ".tif" if "tif" in fmt_str else (".jxl" if "jxl" in fmt_str else (".png" if "png" in fmt_str else ".jpg"))
            fallback_dir = out_path if out_path else (os.path.dirname(src_path) if src_path else os.path.expanduser("~"))
            out_path = os.path.join(fallback_dir, f"{base_name}{ext}")

        # Remember output directory for subsequent exports
        try:
            config_manager.set_last_export_dir(os.path.dirname(out_path))
        except Exception:
            pass

        _requested_path = out_path
        out_path = self._resolve_export_target(
            out_path, src_path=src_path,
            allow_overwrite=bool(cfg.get("overwrite", True)))
        if out_path != _requested_path:
            self.set_app_status(
                f"已另存为 {os.path.basename(out_path)}，未覆盖原始底片", timeout_ms=4000)

        pix = None
        thumb_arr = self._current_photo_thumb if is_active else (curr_p or {}).get("thumbnail_rgb")
        if thumb_arr is not None:
            trgb = thumb_arr
            th, tw = trgb.shape[:2]
            qimg = QImage(trgb.data, tw, th, tw * 3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

        import uuid
        tid = f"task_{uuid.uuid4().hex[:8]}"
        task = ExportTask(tid, os.path.basename(out_path),
                          src_path, out_path, export_params, thumbnail=pix)
        self.export_queue.enqueue(task)
        self._set_export_status("已加入扫描队列，正在后台扫描...", state="busy")

    def action_export_darkroom_lut(self):
        """Export full darkroom physical color profile into standard 3D .cube LUT."""
        suggested_name = f"{self.current_film_stock}_{self.current_paper_stock}.cube"
        last_d = config_manager.get_last_dialog_dir("lut")
        initial_p = os.path.join(last_d, suggested_name) if last_d else suggested_name
        out_file, _ = QFileDialog.getSaveFileName(
            self, "导出当前暗房 3D LUT", initial_p, "3D LUT (*.cube)"
        )
        if not out_file:
            return
        config_manager.set_last_dialog_dir("lut", out_file)

        lut_params = dict(self.current_params)
        lut_params["film_stock"] = self.current_film_stock
        lut_params["paper_stock"] = self.current_paper_stock
        try:
            self.engine.export_lut(lut_params, out_file, lut_size=33)
            self._show_info_dialog("导出完成", f"当前暗房完整 3D LUT 已成功烘焙导出至:\n{out_file}")
        except Exception as e:
            self._show_warning_dialog("导出失败", f"导出 3D LUT 失败: {e}")

    def _batch_export_photos(self, photo_ids):
        """Batch export multiple photos with background export queue.

        One photo and many photos use the same dialog on purpose: the single-selection
        shortcut used to open 单片扫描 (exact path) while two or more opened 批量扫描
        (prefix/suffix), so the same menu item changed its own meaning with selection size.
        """
        if not photo_ids:
            return

        last_dir = config_manager.get_last_export_dir()
        if not last_dir or not os.path.exists(last_dir):
            first_photo = next((p for p in self.photos if p and p.get("id") in photo_ids), None)
            last_dir = os.path.dirname(first_photo["path"]) if first_photo else os.path.expanduser("~")

        first_photo = next((p for p in self.photos if p and p.get("id") in photo_ids), None)
        default_stem = ""
        if len(photo_ids) == 1 and first_photo and first_photo.get("path"):
            default_stem = os.path.splitext(os.path.basename(first_photo["path"]))[0]

        source_dir = os.path.dirname(first_photo["path"]) if first_photo and first_photo.get("path") else last_dir
        dialog = BatchExportDialog(
            last_dir, batch_count=len(photo_ids), parent=self, default_stem=default_stem,
            source_dir=source_dir)
        if first_photo:
            self._seed_export_size_dialog(dialog, first_photo.get("path"))
        if not dialog.exec():
            return

        cfg = dialog.get_export_config()
        self._sync_proof_from_export(cfg.get("colorspace"))
        use_source_dir = str(cfg.get("path_mode", "custom")) == "source"
        out_dir = cfg.get("path", "").strip()
        if not use_source_dir:
            if not out_dir:
                return

            try:
                os.makedirs(out_dir, exist_ok=True)
                config_manager.set_last_export_dir(out_dir)
            except Exception:
                mb = QMessageBox(self)
                mb.setWindowIcon(self.windowIcon())
                mb.setWindowTitle("路径无效")
                mb.setText("无法访问或创建所选的批量导出目标文件夹。")
                mb.setIcon(QMessageBox.Icon.Warning)
                mb.exec()
                return

        fmt = cfg.get("format", "jpeg")
        if fmt == "tiff":
            ext = ".tif"
        elif fmt == "jxl":
            ext = ".jxl"
        elif fmt == "png":
            ext = ".png"
        else:
            ext = ".jpg"

        prefix = cfg.get("prefix", "").strip()
        suffix = cfg.get("suffix", "").strip()
        # Single-photo scan: the dialog collected an exact filename instead of a rule.
        explicit_name = (cfg.get("filename") or "").strip() if len(photo_ids) == 1 else ""
        overwrite = bool(cfg.get("overwrite", True))

        import uuid
        enqueued_count = 0
        missing_count = 0
        new_tasks = []
        used_targets = set()
        for i, pid in enumerate(photo_ids):
            photo = next((p for p in self.photos if p and p.get("id") == pid), None)
            if not photo:
                # Never drop a requested photo silently: the completion toast used to report
                # "N photos queued" with N less than what the user had selected.
                missing_count += 1
                continue

            file_p = photo["path"]
            sdc = sdc_manager.load_sdc(file_p)
            if photo.get("film_profile"):
                film = photo.get("film_profile")
                paper = photo.get("paper_profile", "none")
                params = photo.get("params") or {}
            elif sdc:
                film = sdc.get("film_profile", "none")
                paper = sdc.get("paper_profile", "none")
                params = sdc.get("params") or {}
            elif file_p == self._current_photo_path:
                film = self.current_film_stock
                paper = self.current_paper_stock
                params = dict(self.current_params)
            else:
                film = "none"
                paper = "none"
                params = {}

            exif_d = photo.get("exif", {})
            b_temp = float(exif_d.get("color_temp", 5500.0) if exif_d else 5500.0)
            b_tint = float(exif_d.get("tint", 1.0) if exif_d else 1.0)

            b_ev = float(photo.get("base_ev", 0.0))
            if abs(b_ev) < 0.0001 and hasattr(self, "engine"):
                matched_ep = next((p for p in self.engine.session_photos if p.get("path") == file_p), None)
                if matched_ep and "base_ev" in matched_ep:
                    b_ev = float(matched_ep["base_ev"])

            # Start from the photo's full parameter set instead of re-listing keys by hand:
            # every value the renderer reads has to survive this hop, including the nested
            # lens / crop blocks and the chemistry values baked into the LUT. Defaults only
            # fill in what an older sidecar may be missing.
            export_params = self.get_default_params(b_temp, b_tint)
            if isinstance(params, dict):
                export_params.update(params)
            export_params.update({
                "source_path": file_p,
                "film_stock": film,
                "paper_stock": paper,
                "print_mode": photo.get("print_mode") or params.get("print_mode") or "optical",
                "base_ev": b_ev,
                "base_temp": b_temp,
                "base_tint": b_tint,
                "halation": export_params.get("halation_amount", export_params.get("halation", 1.0)),
                "halation_amount": export_params.get("halation_amount", export_params.get("halation", 1.0)),
                "boost_ev": export_params.get("boost_ev", export_params.get("halation_boost", 0.0)),
                "halation_boost": export_params.get("boost_ev", export_params.get("halation_boost", 0.0)),
                "crop": dict(export_params.get("crop") or {}),
                "lens": dict(export_params.get("lens") or self._get_default_lens_params()),
                # Viewport-only state must never reach a render: a leftover compare view
                # would otherwise turn the exported frame into a split-screen image.
                "view_mode": 0,
                "is_thumbnail": False,
                "engine": cfg.get("engine", "wysiwyg"),
                "format": fmt,
                "quality": cfg.get("quality", 9),
                "dpi": cfg.get("dpi", 300),
                "strip_gps": cfg.get("strip_gps", True),
                "bit_depth": cfg.get("bit_depth", 8),
                "colorspace": cfg.get("colorspace", "sRGB"),
                "subsampling": cfg.get("subsampling", "4:4:4"),
                "progressive": cfg.get("progressive", True),
                "tiff_compression": cfg.get("tiff_compression", "lzw"),
                "png_level": cfg.get("png_level", 6),
                "jxl_mode": cfg.get("jxl_mode", "visual_10"),
                "jxl_lossless": cfg.get("jxl_lossless", False),
                "jxl_distance": cfg.get("jxl_distance", 1.0),
                "jxl_effort": cfg.get("jxl_effort", 7),
                "scale_pct": cfg.get("scale_pct", 100)
            })

            stem = os.path.splitext(os.path.basename(file_p))[0]
            base_fn = explicit_name if explicit_name else f"{prefix}{stem}{suffix}"
            out_fn = f"{base_fn}{ext}"
            target_dir = os.path.dirname(os.path.abspath(file_p)) if use_source_dir else out_dir
            out_target = os.path.join(target_dir, out_fn)
            out_target = self._resolve_export_target(
                out_target, src_path=file_p, used=used_targets,
                allow_overwrite=overwrite)
            out_fn = os.path.basename(out_target)

            pix = None
            if photo.get("thumbnail_rgb") is not None:
                trgb = photo["thumbnail_rgb"]
                th, tw = trgb.shape[:2]
                qimg = QImage(trgb.data, tw, th, tw * 3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg)

            tid = f"task_{uuid.uuid4().hex[:8]}"
            task = ExportTask(tid, os.path.basename(out_target), file_p, out_target, export_params, thumbnail=pix)
            new_tasks.append(task)
            enqueued_count += 1

        if hasattr(self.export_queue, "enqueue_batch"):
            self.export_queue.enqueue_batch(new_tasks)
        else:
            for t in new_tasks:
                self.export_queue.enqueue(t)

        if missing_count:
            self._set_export_status(
                f"已将 {enqueued_count} 张底片加入扫描队列（{missing_count} 张已不在底片库中，已跳过）",
                state="busy")
        else:
            self._set_export_status(f"已将 {enqueued_count} 张底片加入扫描队列", state="busy")

    def _show_info_dialog(self, title, text):
        show_dark_message_box(self, title, text, QMessageBox.Icon.Information,
                              QMessageBox.StandardButton.Ok)

    def _show_warning_dialog(self, title, text):
        show_dark_message_box(self, title, text, QMessageBox.Icon.Warning,
                              QMessageBox.StandardButton.Ok)

    def _show_question_dialog(self, title, text):
        # 统一走 window_utils 那套按主题切换的消息框：这里原来写死深色，浅色主题下
        # 导出/导入的提示是深色卡片，跟应用对不上。
        res = show_dark_message_box(
            self, title, text, QMessageBox.Icon.Question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return res == QMessageBox.StandardButton.Yes

    def _on_film_blank_context_menu(self, pos):
        sender = self.sender()
        global_pos = sender.mapToGlobal(pos) if sender else QCursor.pos()
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        sort_menu = menu.addMenu("整理排列")
        sort_menu.setStyleSheet(get_darkroom_menu_style())
        act_by_name = sort_menu.addAction("按名称排序 (A-Z)")
        act_by_name.triggered.connect(lambda: self._sort_and_reflow_stocks("film", "name"))
        act_by_time = sort_menu.addAction("按添加时间排序 (最新在前)")
        act_by_time.triggered.connect(lambda: self._sort_and_reflow_stocks("film", "time"))

        menu.exec(global_pos)

    def _on_paper_blank_context_menu(self, pos):
        sender = self.sender()
        global_pos = sender.mapToGlobal(pos) if sender else QCursor.pos()
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        sort_menu = menu.addMenu("整理排列")
        sort_menu.setStyleSheet(get_darkroom_menu_style())
        act_by_name = sort_menu.addAction("按名称排序 (A-Z)")
        act_by_name.triggered.connect(lambda: self._sort_and_reflow_stocks("paper", "name"))
        act_by_time = sort_menu.addAction("按添加时间排序 (最新在前)")
        act_by_time.triggered.connect(lambda: self._sort_and_reflow_stocks("paper", "time"))

        menu.exec(global_pos)

    def _sort_and_reflow_stocks(self, mode="film", sort_by="name", persist=True):
        cards_dict = self.film_cards if mode == "film" else self.paper_cards
        grid = self.film_grid if mode == "film" else self.paper_grid
        items = list(cards_dict.items())

        # Item 6: "无 (None)" preset permanently stays at index 0 regardless of sorting
        def _sort_key(item):
            sid, card = item
            is_none = (sid == "none" or getattr(card, "clean_name", "").strip() in ("无 (None)", "无", "None"))
            if is_none:
                return (0, "")
            if sort_by == "name":
                return (1, card.clean_name.lower())
            else:
                return (1, not card.is_custom, sid)

        items.sort(key=_sort_key)

        # Re-populate in sorted order
        for card in cards_dict.values():
            grid.removeWidget(card)
        cards_dict.clear()
        for k, v in items:
            cards_dict[k] = v
        self.reflow_grids()

        if persist:
            try:
                config_manager.set_ui_state(f"{mode}_sort_order", sort_by)
            except Exception:
                pass

    def open_preferences(self):
        old_res = config_manager.get_preferences().get("preview_max_edge", 2048)
        dlg = PreferencesDialog(parent=self)
        if dlg.exec():
            self._apply_hq_preview_pref(announce=True)
            self._apply_display_cm_pref()
            new_res = config_manager.get_preferences().get("preview_max_edge", 2048)
            if old_res != new_res:
                for p in self.photos:
                    if p:
                        p["float_img"] = None
                session_cache_manager.clear_cache()
                if self._current_photo_path:
                    photo = next((p for p in self.photos if p and p.get("path") == self._current_photo_path), None)
                    if photo:
                        res = self.engine.load_image(photo["path"])
                        if res.get("success"):
                            photo["float_img"] = res.get("float_img")
                            session_cache_manager.save_photo_preview(photo["path"], photo["float_img"])
                            m_w = photo.get("width") or (photo.get("meta", {}).get("width") if isinstance(photo.get("meta"), dict) else None)
                            m_h = photo.get("height") or (photo.get("meta", {}).get("height") if isinstance(photo.get("meta"), dict) else None)
                            self.canvas.set_image(photo["float_img"], master_width=m_w, master_height=m_h, reset_zoom=False)
                            self._update_photo_info_display()
                            self.set_app_status("已按首选项重新载入全尺寸纹理", timeout_ms=3000)
            if hasattr(self, 'canvas'):
                self.canvas.update()

    def _show_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self._saved_normal_geo = self.geometry()
            self.showMaximized()
        if hasattr(self, 'btn_max'):
            is_max = self.isMaximized()
            self.btn_max.set_maximized_state(is_max)
            if hasattr(self, 'btn_close'):
                self.btn_close.set_maximized_state(is_max)

    def showMinimized(self):
        # Native DWM minimize animation
        super().showMinimized()

    def _close_after_dwell(self):
        """Finish a close that closeEvent held on screen for CLOSE_DWELL_MS."""
        self._close_approved = True
        try:
            self.close()
        finally:
            self._close_approved = False

    def _run_close_dwell(self):
        """Look like the app is finishing work, then finish it.

        A timer-driven wait left the window fully live: the user saw a normal, responsive
        window for a beat and then it vanished.  What was asked for is the opposite -- a window
        that has stopped responding, with the system's spinner over it -- so this blocks the GUI
        thread on purpose.  One round trip first, because the cursor change and the last repaint
        have to reach the screen *before* the block, or the whole dwell happens off-camera.
        """
        app = QApplication.instance()
        if app is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            app.processEvents()
        try:
            time.sleep(CLOSE_DWELL_MS / 1000.0)
        finally:
            if app is not None:
                QApplication.restoreOverrideCursor()
            self._close_after_dwell()

    def closeEvent(self, event):
        _frame_log("closeEvent enter")
        approved = getattr(self, "_close_approved", False)
        if not approved:
            # 未保存的会话必须在提问之后才真正关闭，而提问里会弹 Windows 原生"另存为"。
            # 在 WM_CLOSE 的同步调用栈里开这种模态原生窗口，Windows 的对话框消息循环会
            # 重入关闭流程：用户在对话框里点取消后事件链被吞掉，窗口反而被销毁（会话和
            # 进程一起没了）。所以这里先 ignore，下一轮事件循环再问；问过了才 close。
            if getattr(self, "_close_prompt_running", False):
                event.ignore()
                return
            if self._close_needs_prompt():
                # 标记在排队前就置上：连点关闭按钮时第二次 closeEvent 只会被忽略，
                # 不会排出第二个提问。
                self._close_prompt_running = True
                event.ignore()
                QTimer.singleShot(0, self._run_close_prompt)
                return

        # Hold the window on screen for a beat before it vanishes. Nothing below is slow --
        # teardown hides the window in the same millisecond it starts -- so a close used to be
        # instantaneous in a way that reads as "it did not even try to save"; a 0.7 s dwell is
        # the price of the app looking like it shut down deliberately (asked for explicitly).
        # The dwell runs before the caption drop so the frame is still a normal window while it
        # waits; only the final close() reaches the teardown below.
        if not getattr(self, "_close_dwell_done", False):
            self._close_dwell_done = True
            event.ignore()
            # Not a bare ``self.close``: the prompt path clears ``_close_approved`` in a
            # ``finally`` around its own close(), so a deferred close would come back here with
            # the flag down and pop the same save dialog a second time.
            QTimer.singleShot(0, self._run_close_dwell)
            return

        # Drop the caption for the whole closing sequence (prompt included). Window teardown is
        # when Windows repaints the frame; without the style bit there is nothing to paint, so
        # the classic title bar cannot flash while the save prompt / .sdc writes / worker
        # teardown run. Restored if the user cancels the close.
        caption_was_enabled = _FRAME_MODE != "clean"
        if caption_was_enabled:
            self._set_native_caption(False)

        # Capture the closing state and hide IMMEDIATELY. Every step below (.sdc writes for each
        # dirty photo including thumbnails, worker teardown with 1 s waits, config/session
        # writes) runs synchronously on the GUI thread. A still-visible window that stops
        # answering messages for ~5 s makes DWM draw its ghost placeholder - the classic
        # "Win7 style" caption bar that no WM_NC* interception can suppress, because the shell
        # paints it, not our window procedure.
        try:
            self._close_is_maximized = self.isMaximized()
            self._close_geometry = self.geometry()
            self.hide()
            self._release_native_filters()
            QApplication.processEvents()
            _frame_log("closeEvent hidden")
        except Exception:
            pass

        # Silently ensure any modified parameters are saved to .sdc
        try:
            if self._photo_is_dirty and self._current_photo_path:
                curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                cur_thumb = curr_p.get("thumbnail_rgb") if curr_p else None
                paper_val = (self.current_paper_stock
                             if (curr_p is not None and self._photo_paper_is_explicit(curr_p)) else "")
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    paper_val,
                    self.current_params,
                    thumbnail_rgb=cur_thumb,
                    print_mode=getattr(self, "current_print_mode", "optical"),
                    paper_explicit=(curr_p is not None and self._photo_paper_is_explicit(curr_p))
                )
                self._photo_is_dirty = False
            for p in self.photos:
                if p and p.get("is_dirty") and p.get("path"):
                    sdc_manager.save_sdc(
                        p["path"],
                        p.get("film_profile", self.current_film_stock),
                        p.get("paper_profile", self.current_paper_stock),
                        p.get("params", self.current_params),
                        thumbnail_rgb=p.get("thumbnail_rgb"),
                        print_mode=p.get("print_mode", getattr(self, "current_print_mode", "optical"))
                    )
                    p["is_dirty"] = False
        except Exception as e:
            logger.warning(f"Auto-saving .sdc on close: {e}")

        try:
            if hasattr(self, 'thumb_controller'):
                self.thumb_controller.cancel_card_preview_worker()
            if getattr(self, "_thumb_grade_worker_ref", None) is not None:
                self._thumb_grade_worker_ref.stop()
                self._safely_terminate_worker(self._thumb_grade_worker_ref, wait_ms=1000)
                self._thumb_grade_worker_ref = None
            if getattr(self, "_thumb_prime_worker", None) is not None:
                self._thumb_prime_worker.stop()
                self._safely_terminate_worker(self._thumb_prime_worker, wait_ms=1000)
                self._thumb_prime_worker = None
            if hasattr(self, '_single_photo_loader') and self._single_photo_loader:
                self._safely_terminate_worker(self._single_photo_loader, wait_ms=1000)
                self._single_photo_loader = None
            if hasattr(self, '_session_restore_worker') and self._session_restore_worker:
                self._safely_terminate_worker(self._session_restore_worker, wait_ms=1000)
                self._session_restore_worker = None
            if getattr(self, '_ca_worker', None):
                self._safely_terminate_worker(self._ca_worker, wait_ms=2000)
                self._ca_worker = None
            # Workers that overran the 1 s budget were parked in _retained_zombie_workers and
            # never waited on again, so the window could be destroyed while they were still
            # running ("QThread: Destroyed while thread is still running"). Give them one
            # last, longer grace period here — closeEvent is the final chance.
            zombies = list(getattr(self, "_retained_zombie_workers", []) or [])
            for zb in zombies:
                try:
                    if hasattr(zb, "isRunning") and zb.isRunning():
                        zb.wait(3000)
                except Exception:
                    pass
            self._retained_zombie_workers = [
                zb for zb in zombies
                if hasattr(zb, "isRunning") and zb.isRunning()
            ]
        except Exception:
            pass

        try:
            # Prefer the snapshot taken before hide(): windowState/geometry can change once the
            # window is hidden.
            is_max = getattr(self, "_close_is_maximized", self.isMaximized())
            geo = getattr(self, "_close_geometry", None) or self.geometry()
            if is_max and self._saved_normal_geo:
                norm_w = self._saved_normal_geo.width()
                norm_h = self._saved_normal_geo.height()
                norm_x = self._saved_normal_geo.x()
                norm_y = self._saved_normal_geo.y()
            else:
                norm_w = geo.width()
                norm_h = geo.height()
                norm_x = geo.x()
                norm_y = geo.y()

            if hasattr(self, "main_splitter") and self.stack.currentIndex() == 1 and sum(self.main_splitter.sizes()) > 500:
                splitter_main = self.main_splitter.sizes()
                splitter_left = self.left_splitter.sizes() if hasattr(self, "left_splitter") else None
                splitter_center = self.center_splitter.sizes() if hasattr(self, "center_splitter") else None
            else:
                prev_win = config_manager.get_window_config()
                splitter_main = prev_win.get("splitter_main")
                splitter_left = prev_win.get("splitter_left")
                splitter_center = prev_win.get("splitter_center")

            config_manager.save_window_config(
                width=norm_w,
                height=norm_h,
                x=norm_x,
                y=norm_y,
                is_maximized=is_max,
                splitter_main=splitter_main,
                splitter_left=splitter_left,
                splitter_center=splitter_center,
            )

            sections_expanded = {}
            if hasattr(self, "sec_exposure"):
                sections_expanded["exposure"] = self.sec_exposure.is_expanded
            if hasattr(self, "sec_optics"):
                sections_expanded["optics_diff"] = self.sec_optics.is_expanded
            if hasattr(self, "sec_chemistry"):
                sections_expanded["chemistry"] = self.sec_chemistry.is_expanded
            if hasattr(self, "sec_enlarger"):
                sections_expanded["enlarger"] = self.sec_enlarger.is_expanded
            if hasattr(self, "sec_optical"):
                sections_expanded["optics"] = self.sec_optical.is_expanded

            config_manager.save_ui_state(
                sections_expanded=sections_expanded,
                last_film_stock=self.current_film_stock,
                last_paper_stock=self.current_paper_stock,
            )

            # Ensure active photo focus is persisted to session storage
            self._update_session_focus_on_disk()

            # Remember active session path ONLY if it is a saved session on disk
            if self._current_session_path and os.path.exists(self._current_session_path) and not getattr(self, "_is_temp_session", False):
                config_manager.save_last_session_path(self._current_session_path)
            else:
                config_manager.save_last_session_path(None)
                if self.photos:
                    self._save_auto_temp_session()
        except Exception as e:
            logger.warning(f"Error saving config on close: {e}")

        self._cancel_active_import(notify_user=False)

        if hasattr(self, 'export_queue') and self.export_queue:
            self.export_queue.close()

        if self._photo_is_dirty and self._current_photo_path:
            sdc_manager.save_sdc(
                self._current_photo_path,
                self.current_film_stock,
                self.current_paper_stock,
                self.current_params,
                print_mode=getattr(self, "current_print_mode", "optical")
            )
        self._photo_is_dirty = False
        event.accept()
