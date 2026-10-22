"""SpektraDarkroom Main Window - Vulkan GPU Real-Time Accelerated Darkroom
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
from PySide6.QtGui import (
    QAction, QFont, QIcon, QKeySequence, QPainter, QColor, QPen, QBrush, QPixmap, QImage, QLinearGradient, QCursor
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QMenuBar, QMenu, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSplitter, QSplitterHandle, QGridLayout, QSizePolicy, QApplication,
    QDialog, QLineEdit, QTextEdit, QCheckBox, QButtonGroup
)

from core.path_utils import get_icon_path, get_resource_dir, normalize_path
from core.format_constants import (
    RAW_EXTENSIONS,
    ALL_SUPPORTED_IMAGE_EXTENSIONS,
    FILE_DIALOG_IMAGE_FILTER
)
from core.version import get_app_title, get_full_version_info
from core.app_core import SpektraEngine
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

from ui.window_utils import apply_dark_titlebar, get_darkroom_menu_style, get_icon, get_standard_checkbox_style
from ui.smooth_scroll import SmoothScrollArea
from ui.stock_manager_dialog import StockManagerDialog, export_stock_lut
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
from ui.spinner_widget import SpinnerWidget
from ui.histogram_widget import HistogramWidget
from ui.fading_stacked_widget import FadingStackedWidget
from ui.theme_manager import theme_manager


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

    def stop(self):
        self._stopped = True

    def run(self):
        total = len(self.file_paths)
        if total == 0:
            self.importFinished.emit(0)
            return

        import os
        import concurrent.futures

        preview_max_edge = 2048
        try:
            from core import config_manager
            preview_max_edge = int(config_manager.get_preferences().get("preview_max_edge", 2048))
        except Exception:
            preview_max_edge = 2048

        # Multi-core CPU concurrency pool capped to safe threshold (avoids LibRaw OOM)
        max_workers = min(6, max(2, (os.cpu_count() or 4) // 2))
        logger.info(f"BatchImportWorker starting with {max_workers} worker threads for {total} files")

        results = {}
        next_to_emit = 0
        loaded_count = 0

        def _decode_item(idx, file_path):
            if self._stopped:
                return idx, None
            try:
                is_first = (idx == 0 and self.start_idx == 0)
                cached = session_cache_manager.get_photo_cache(file_path, load_preview=is_first)
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
                        "float_img": cached.get("float_img") if is_first else None,
                        "raw_preview": None,
                        "exif": cached.get("exif", {}),
                        "auto_ev": cached.get("auto_ev", 0.0),
                        "base_ev": cached.get("base_ev", 0.0),
                        "is_raw": cached.get("is_raw", False)
                    }
                    return idx, entry

                # 2. Parallel thread-safe decode:
                # First photo gets full demosaic for immediate viewport presentation;
                # all subsequent photos use fast_thumbnail=True (embedded JPEG, sub-15ms)
                use_fast = not is_first
                entry = self.engine.decode_photo_data(
                    file_path,
                    preview_max_edge=preview_max_edge,
                    is_first=is_first,
                    fast_thumbnail=use_fast
                )
                if entry and entry.get("success"):
                    pid = f"photo_{self.start_idx + idx}_{os.path.basename(file_path)}"
                    entry["id"] = pid
                    if abs(float(entry.get("base_ev", 0.0))) < 0.0001:
                        if sdc_manager.has_sdc(file_path):
                            sdc_data = sdc_manager.load_sdc(file_path)
                            if sdc_data and sdc_data.get("params", {}).get("base_ev"):
                                entry["base_ev"] = float(sdc_data["params"]["base_ev"])
                    entry["raw_thumbnail_rgb"] = entry.get("raw_thumbnail_rgb") if entry.get("raw_thumbnail_rgb") is not None else entry.get("thumbnail_rgb")
                    entry["graded_thumbnail_rgb"] = entry.get("graded_thumbnail_rgb")
                    if not is_first:
                        entry["float_img"] = None
                        entry["raw_preview"] = None
                    return idx, entry
                else:
                    logger.warning(f"Failed to decode {file_path}: {entry.get('error') if entry else 'Unknown error'}")
            except Exception as ex:
                logger.warning(f"Error decoding {file_path}: {ex}")

            # Fallback placeholder: attempt emergency standalone fast thumbnail extraction
            try:
                fast_th = session_cache_manager.extract_fast_thumbnail(file_path)
                dw, dh = session_cache_manager.get_image_dimensions(file_path)
                fallback_entry = {
                    "id": f"photo_{self.start_idx + idx}_{os.path.basename(file_path)}",
                    "filename": os.path.basename(file_path),
                    "path": file_path,
                    "width": dw,
                    "height": dh,
                    "thumbnail_rgb": fast_th,
                    "raw_thumbnail_rgb": fast_th,
                    "graded_thumbnail_rgb": None,
                    "float_img": None,
                    "raw_preview": None,
                    "exif": {"filename": os.path.basename(file_path), "path": file_path, "width": dw, "height": dh},
                    "auto_ev": 0.0,
                    "base_ev": 0.0,
                    "is_raw": True,
                    "decode_error": (fast_th is None)
                }
                return idx, fallback_entry
            except Exception:
                return idx, None

        # Execute concurrently with high CPU utilization
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_decode_item, i, p): i for i, p in enumerate(self.file_paths)}
            for future in concurrent.futures.as_completed(future_map):
                if self._stopped:
                    break
                idx, entry = future.result()
                results[idx] = entry

                # In-order streaming emission so filmstrip keeps exact file sequence
                while next_to_emit in results:
                    emit_entry = results.pop(next_to_emit)
                    if emit_entry is not None:
                        loaded_count += 1
                        self.photoLoaded.emit(emit_entry, loaded_count, total)
                    next_to_emit += 1

        # Drain any remaining results
        while next_to_emit in results:
            emit_entry = results.pop(next_to_emit)
            if emit_entry is not None:
                loaded_count += 1
                self.photoLoaded.emit(emit_entry, loaded_count, total)
            next_to_emit += 1

        self.importFinished.emit(loaded_count)


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

        max_workers = min(16, max(4, os.cpu_count() or 4))
        total_items = len(self.uncached_items)
        done_count = 0

        def _decode(slot_idx, file_path):
            if self._stopped:
                return slot_idx, None
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
    progress = Signal(int, int)                # current, total
    restoreFinished = Signal(list, list, str)  # all_photos, uncached_slots, target_path
    failed = Signal(str)

    def __init__(self, existing_files, active_file=None, thumb_controller=None, engine=None, is_params_modified_fn=None, parent=None):
        super().__init__(parent)
        self.existing_files = list(existing_files or [])
        self.active_file = active_file
        self.thumb_controller = thumb_controller
        self.engine = engine
        self.is_params_modified_fn = is_params_modified_fn
        self._stopped = False

    def stop(self):
        self._stopped = True

    def _process_single_photo(self, idx: int, p: str, is_target: bool) -> tuple:
        """Processes a single photo entry: sdc, cache, thumbnails, dimensions."""
        from core import sdc_manager
        pid = f"photo_{idx}_{os.path.basename(p)}"
        sdc = sdc_manager.load_sdc(p)
        film_profile = sdc.get("film_profile", "none") if sdc else "none"
        paper_profile = sdc.get("paper_profile", "none") if sdc else "none"
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
        if sdc:
            sdc_th = sdc.get("thumbnail_rgb")
            if session_cache_manager.is_valid_thumbnail(sdc_th):
                th_rgb = sdc_th
                graded_th = sdc_th
            elif raw_th is not None:
                if self.thumb_controller and (film_profile != "none" or paper_profile != "none" or (params and abs(params.get("exposure_ev", 0.0)) > 0.001)):
                    try:
                        graded_th = self.thumb_controller.grade_thumbnail_fast(raw_th, film_profile, paper_profile, params)
                    except Exception:
                        graded_th = raw_th
                    th_rgb = graded_th
                else:
                    th_rgb = raw_th
        else:
            th_rgb = raw_th
            graded_th = None

        is_uncached = False
        crop_data = params.get("crop") if (params and isinstance(params, dict)) else None
        b_ev = 0.0
        if params and "base_ev" in params and abs(float(params["base_ev"])) > 0.0001:
            b_ev = float(params["base_ev"])
        elif cache_item is not None:
            b_ev = float(cache_item.get("base_ev", 0.0))

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
                "print_mode": print_mode,
                "params": params,
                "crop": crop_data,
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
                "print_mode": print_mode,
                "params": params,
                "crop": crop_data,
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
        "kodak_ultra_endura": "Ultra 高反差相纸"
    }

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
        self.card_w = 110
        self.card_h = 118
        self.setFixedSize(self.card_w, self.card_h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Full official brand name
        self.clean_name = self.stock_name

        self.original_tooltip = f"{self.stock_name}\n{stock_info.get('badge', '')}\n{stock_info.get('desc', '')}".strip()

        self._init_ui()
        theme_manager.themeChanged.connect(lambda _: self._update_appearance())

    def setToolTip(self, tip: str):
        super().setToolTip(tip)
        if hasattr(self, "lbl_thumb") and self.lbl_thumb:
            self.lbl_thumb.setToolTip(tip)
        if hasattr(self, "lbl_name") and self.lbl_name:
            self.lbl_name.setToolTip(tip)

    def set_card_tooltip(self, tip: str):
        self.setToolTip(tip)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)

        # Image preview frame (centered, rounded corners)
        self.lbl_thumb = QLabel()
        self.lbl_thumb.setObjectName("filmCardThumb")
        self.lbl_thumb.setFixedSize(98, 70)
        self.lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_thumb.setStyleSheet("QLabel#filmCardThumb { border-radius: 4px; background: #18191f; border: none; }")
        layout.addWidget(self.lbl_thumb, 0, Qt.AlignmentFlag.AlignCenter)

        # Bottom label container
        self.lbl_name = QLabel(self.clean_name)
        self.lbl_name.setObjectName("filmCardName")
        self.lbl_name.setFixedHeight(28)
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_name)

        self._update_appearance()
        self.set_card_tooltip(self.original_tooltip)

    def update_card_dimensions(self, new_w):
        new_w = max(96, min(220, int(new_w)))
        thumb_w = new_w - 12
        thumb_h = int(round(thumb_w * (66.0 / 98.0)))
        new_h = thumb_h + 40
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
            scaled = pixmap.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_thumb.setPixmap(scaled)
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

        # Subtle card thumbnail outer frame
        p.setPen(QPen(QColor("#222530"), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, tw - 1.0, th - 1.0), 4.0, 4.0)
        p.end()
        self.lbl_thumb.setPixmap(pm)


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
        tooltip_bubble_style = theme_manager.get_tooltip_stylesheet()
        thumb_bg = "#18191f" if is_dark else "#e6eaf0"

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
                QLabel#filmCardThumb {{ border: none; background: {thumb_bg}; border-radius: 4px; }}
                QLabel#filmCardName {{ border: none; background: transparent; color: {dis_name}; font-size: 9.5px; }}
                {tooltip_bubble_style}
            """)
            self.lbl_name.setStyleSheet(f"QLabel#filmCardName {{ color: {dis_name}; font-size: 9.5px; border: none; background: transparent; }}")
            return
        if self.is_active:
            act_bg = "rgba(245, 158, 11, 0.08)" if is_dark else "rgba(245, 158, 11, 0.12)"
            self.setStyleSheet(f"""
                QFrame#filmGridCard {{
                    background: {act_bg};
                    border: 1px solid #f59e0b;
                    border-radius: 6px;
                }}
                QLabel#filmCardThumb {{ border: none; background: {thumb_bg}; border-radius: 4px; }}
                QLabel#filmCardName {{ border: none; background: transparent; color: #f59e0b; font-size: 9.5px; font-weight: bold; }}
                {tooltip_bubble_style}
            """)
            self.lbl_name.setStyleSheet("QLabel#filmCardName { color: #f59e0b; font-size: 9.5px; font-weight: bold; border: none; background: transparent; }")
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
                QLabel#filmCardThumb {{ border: none; background: {thumb_bg}; border-radius: 4px; }}
                QLabel#filmCardName {{ border: none; background: transparent; color: {name_color}; font-size: 9.5px; font-weight: 500; }}
                {tooltip_bubble_style}
            """)
            self.lbl_name.setStyleSheet(f"QLabel#filmCardName {{ color: {name_color}; font-size: 9.5px; font-weight: 500; border: none; background: transparent; }}")

    def mousePressEvent(self, event):
        if not self.isEnabled():
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.stock_id)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            menu.setStyleSheet(get_darkroom_menu_style())
            act_apply = menu.addAction("应用预设")
            act_apply.triggered.connect(lambda: self.selected.emit(self.stock_id))

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


class ImportStockDialog(QDialog):
    """Adobe-styled darkroom dialog for entering stock name and optical description upon JSON import."""
    def __init__(self, stock_type="film", default_name="", parent=None):
        super().__init__(parent)
        type_str = "胶卷" if stock_type == "film" else "放大相纸"
        self.setWindowTitle(f"导入{type_str}物理配置")
        self.setFixedSize(380, 260)
        self.setStyleSheet("""
            QDialog {
                background: #14151a;
                border: 1px solid #333644;
                border-radius: 8px;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 12px;
                font-weight: 500;
            }
            QLineEdit, QTextEdit {
                background: #1e2029;
                color: #f8fafc;
                border: 1px solid #333644;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #f59e0b;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        lbl_name = QLabel("预设名称:")
        layout.addWidget(lbl_name)
        self.txt_name = QLineEdit(default_name)
        layout.addWidget(self.txt_name)

        lbl_desc = QLabel(f"{type_str}描述 / 光学特征 (可选):")
        layout.addWidget(lbl_desc)
        self.txt_desc = QTextEdit()
        self.txt_desc.setPlaceholderText("例如：微暖肤色、高宽容度、柔和高光滚降...")
        layout.addWidget(self.txt_desc)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        btn_cancel = QPushButton("取消")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: #232530;
                color: #94a3b8;
                border: 1px solid #333644;
                border-radius: 4px;
                padding: 5px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #2c2e3c;
                color: #f1f5f9;
            }
        """)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_confirm = QPushButton("确认导入")
        btn_confirm.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #000000;
                border: 1px solid #d97706;
                border-radius: 4px;
                padding: 5px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #fbbf24;
            }
            QPushButton:pressed {
                background: #d97706;
            }
        """)
        btn_confirm.clicked.connect(self.accept)
        btn_layout.addWidget(btn_confirm)

        layout.addLayout(btn_layout)

    def get_data(self):
        name = self.txt_name.text().strip()
        desc = self.txt_desc.toPlainText().strip()
        return name, desc



class CompactExifWidget(QWidget):
    """Item 7: High-density compact EXIF property grid with pinned header and anti-jitter margins."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.key_labels = []
        self._init_ui()
        self.apply_theme()
        theme_manager.themeChanged.connect(self.apply_theme)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
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

    def set_exif_data(self, data):
        for k, lbl in self.labels.items():
            raw_val = data.get(k, "-")
            val = str(raw_val).strip() if raw_val is not None and str(raw_val).strip() != "" else "-"
            lbl.setText(val)
            lbl.setToolTip("")


class FadingSplitterHandle(QSplitterHandle):
    """Refined splitter handle with delicate theme-matched hairline fading at both ends."""
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._hovered = False
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

        if self.orientation() == Qt.Orientation.Vertical:
            # Horizontal line separating vertical tiers (left sidebar, center splitter)
            cy = h // 2
            # User requirement: 10% length centered, gray-white subtle indicator
            line_w = max(24, int(w * 0.10))
            start_x = (w - line_w) // 2
            end_x = start_x + line_w

            grad = QLinearGradient(start_x, cy, end_x, cy)
            if self._hovered:
                if is_dark:
                    grad.setColorAt(0.0, QColor(248, 250, 252, 20))
                    grad.setColorAt(0.2, QColor(248, 250, 252, 180))
                    grad.setColorAt(0.5, QColor(255, 255, 255, 240))
                    grad.setColorAt(0.8, QColor(248, 250, 252, 180))
                    grad.setColorAt(1.0, QColor(248, 250, 252, 20))
                else:
                    grad.setColorAt(0.0, QColor(51, 65, 85, 20))
                    grad.setColorAt(0.2, QColor(51, 65, 85, 180))
                    grad.setColorAt(0.5, QColor(15, 23, 42, 240))
                    grad.setColorAt(0.8, QColor(51, 65, 85, 180))
                    grad.setColorAt(1.0, QColor(51, 65, 85, 20))
            else:
                if is_dark:
                    grad.setColorAt(0.0, QColor(203, 213, 225, 0))
                    grad.setColorAt(0.2, QColor(203, 213, 225, 60))
                    grad.setColorAt(0.5, QColor(226, 232, 240, 110))
                    grad.setColorAt(0.8, QColor(203, 213, 225, 60))
                    grad.setColorAt(1.0, QColor(203, 213, 225, 0))
                else:
                    grad.setColorAt(0.0, QColor(100, 116, 139, 0))
                    grad.setColorAt(0.2, QColor(100, 116, 139, 80))
                    grad.setColorAt(0.5, QColor(71, 85, 105, 140))
                    grad.setColorAt(0.8, QColor(100, 116, 139, 80))
                    grad.setColorAt(1.0, QColor(100, 116, 139, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(start_x, cy, line_w, 1)
        else:
            # Vertical line separating horizontal columns
            cx = w // 2
            # User requirement: 10% length centered, gray-white subtle indicator
            line_h = max(32, int(h * 0.10))
            start_y = (h - line_h) // 2
            end_y = start_y + line_h

            grad = QLinearGradient(cx, start_y, cx, end_y)
            if self._hovered:
                if is_dark:
                    grad.setColorAt(0.0, QColor(248, 250, 252, 20))
                    grad.setColorAt(0.2, QColor(248, 250, 252, 180))
                    grad.setColorAt(0.5, QColor(255, 255, 255, 240))
                    grad.setColorAt(0.8, QColor(248, 250, 252, 180))
                    grad.setColorAt(1.0, QColor(248, 250, 252, 20))
                else:
                    grad.setColorAt(0.0, QColor(51, 65, 85, 20))
                    grad.setColorAt(0.2, QColor(51, 65, 85, 180))
                    grad.setColorAt(0.5, QColor(15, 23, 42, 240))
                    grad.setColorAt(0.8, QColor(51, 65, 85, 180))
                    grad.setColorAt(1.0, QColor(51, 65, 85, 20))
            else:
                if is_dark:
                    grad.setColorAt(0.0, QColor(203, 213, 225, 0))
                    grad.setColorAt(0.2, QColor(203, 213, 225, 60))
                    grad.setColorAt(0.5, QColor(226, 232, 240, 110))
                    grad.setColorAt(0.8, QColor(203, 213, 225, 60))
                    grad.setColorAt(1.0, QColor(203, 213, 225, 0))
                else:
                    grad.setColorAt(0.0, QColor(100, 116, 139, 0))
                    grad.setColorAt(0.2, QColor(100, 116, 139, 80))
                    grad.setColorAt(0.5, QColor(71, 85, 105, 140))
                    grad.setColorAt(0.8, QColor(100, 116, 139, 80))
                    grad.setColorAt(1.0, QColor(100, 116, 139, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(cx, start_y, 1, line_h)


class FadingSplitter(QSplitter):
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(4 if orientation == Qt.Orientation.Vertical else 5)
        self.setChildrenCollapsible(False)
        self.setStyleSheet("QSplitter { background: transparent; }")

    def createHandle(self):
        return FadingSplitterHandle(self.orientation(), self)


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


class DarkroomMainWindow(QMainWindow):
    """SpektraDarkroom Complete Application Window."""
    def __init__(self, engine, initial_session=None, initial_files=None):
        super().__init__()
        self.engine = engine
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
            "grain_cloud_blur": 1.0,
            "grain_colour": 1.0,
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

        self._init_ui()
        self.export_queue.status_changed.connect(self._on_export_queue_status_changed)
        self._init_menu_and_actions()
        self._wire_canvas_events()
        self._restore_saved_state()
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
                screen = self.screen() or QApplication.primaryScreen()
                if screen:
                    screen_geo = screen.availableGeometry()
                    if screen_geo.intersects(QRect(saved_x, saved_y, min(300, saved_w), min(200, saved_h))):
                        self.move(saved_x, saved_y)

            # 2. Splitter proportions
            if hasattr(self, "main_splitter") and "splitter_main" in win_cfg:
                sm = win_cfg["splitter_main"]
                if len(sm) == 3 and all(s > 0 for s in sm):
                    self.main_splitter.setSizes(sm)

            if hasattr(self, "left_splitter") and "splitter_left" in win_cfg:
                sl = win_cfg["splitter_left"]
                if len(sl) == 3 and all(s > 0 for s in sl):
                    self.left_splitter.setSizes(sl)

            if hasattr(self, "center_splitter") and "splitter_center" in win_cfg:
                sc = win_cfg["splitter_center"]
                if len(sc) == 2 and all(s > 0 for s in sc):
                    self.center_splitter.setSizes(sc)

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
                        QTimer.singleShot(40, lambda: (not self.photos and not self._current_session_path) and self._open_session_by_path(last_session))
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
                QTimer.singleShot(2500, session_cache_manager.check_and_auto_clean)
        except Exception as e:
            logger.warning(f"Error restoring application state: {e}")

    def _restore_saved_session(self, existing_files, active_file=None, session_name="", path_updated=False):
        """Asynchronously restores session photos using SessionRestoreWorker and FadingStackedWidget.
        Pre-allocates slots for all files to guarantee order and count stability (100 remains 100).
        Transitions smoothly to editor workspace while restoring active photo and thumbnails in background.
        """
        if not existing_files:
            return

        total_files = len(existing_files)
        self.photos = [None] * total_files
        self._is_importing = True
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(True)

        target_path = active_file if (active_file and active_file in existing_files) else existing_files[0]
        self._target_initial_active_path = target_path

        # 1. Switch to darkroom workspace smoothly
        self.stack.setCurrentIndexAnimated(1, duration_ms=250)
        self.btn_title_export.setVisible(True)
        if session_name:
            self.set_app_status(f"正在载入会话: {session_name}...", spinning=True, timeout_ms=0)
        else:
            self.set_app_status("正在载入会话...", spinning=True, timeout_ms=0)

        # 2. Launch background worker
        worker = SessionRestoreWorker(
            existing_files,
            active_file=target_path,
            thumb_controller=self.thumb_controller,
            engine=self.engine,
            is_params_modified_fn=self._is_params_modified,
            parent=self
        )
        self._session_restore_worker = worker

        def _on_active_photo_ready(target_idx, target_entry):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            if 0 <= target_idx < len(self.photos):
                self.photos[target_idx] = target_entry
            target_id = target_entry.get("id")
            if target_id and target_entry.get("float_img") is not None:
                try:
                    self.on_switch_photo(target_id)
                except Exception as e:
                    logger.debug(f"Immediate active photo activation: {e}")
            self._update_window_title()

        def _on_progress(current, total):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            if getattr(self, "_is_importing", False):
                self.set_app_status(f"正在载入会话 ({current}/{total} 张)...", spinning=True, timeout_ms=0)

        def _on_restore_finished(all_photos, uncached_slots, final_target_path):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            self._session_restore_worker = None
            self.photos = all_photos
            self.canvas.set_has_library_photos(bool(self.photos))

            target_entry = next((p for p in self.photos if p and p.get("path") == final_target_path), self.photos[0] if self.photos else None)
            target_id = target_entry["id"] if target_entry else None

            # Populate filmstrip with full session photos
            self.filmstrip.set_photos(self.photos, target_id)

            # Mark dirty/edited badges
            for entry in self.photos:
                if entry and (entry.get("is_dirty") or entry.get("is_edited")):
                    self.filmstrip.set_photo_dirty(entry["id"], True)

            # Activate target photo if not yet active or canvas empty
            if target_id and (self.active_photo_id != target_id or not getattr(self.canvas, '_has_image', False)):
                try:
                    self.on_switch_photo(target_id)
                except Exception as e:
                    logger.error(f"Error activating photo on session restore: {e}")

            # Ensure filmstrip jumps smoothly to the focus photo upon session restore
            if target_id and hasattr(self.filmstrip, 'scroll_to_photo'):
                self.filmstrip.scroll_to_photo(target_id, animate=False, force_center=True)
                QTimer.singleShot(60, lambda: self.filmstrip.scroll_to_photo(target_id, animate=False, force_center=True))
                QTimer.singleShot(260, lambda: self.filmstrip.scroll_to_photo(target_id, animate=False, force_center=True))

            self._update_window_title()

            # Uncached slots handling
            if uncached_slots:
                self._is_importing = True
                if hasattr(self, "btn_cancel_import"):
                    self.btn_cancel_import.setVisible(True)
                self.set_app_status(f"正在恢复底片库 (0/{len(uncached_slots)} 张)...", spinning=True, timeout_ms=0)
                self._slot_decode_worker = SlotDecodeWorker(uncached_slots, self.engine, parent=self)
                self._slot_decode_worker.slotDecoded.connect(self._on_slot_photo_decoded)
                self._slot_decode_worker.progress.connect(self._on_slot_decode_progress)
                self._slot_decode_worker.finished.connect(self._on_all_slots_decoded)
                self._slot_decode_worker.start()
            else:
                if self._start_next_pending_import_batch():
                    return
                self._is_importing = False
                if hasattr(self, "btn_cancel_import"):
                    self.btn_cancel_import.setVisible(False)
                if session_name:
                    if path_updated:
                        self.set_app_status(f"会话已恢复 (底片路径已更新): {session_name} ({total_files} 张)", spinning=False, timeout_ms=3000)
                    else:
                        self.set_app_status(f"会话已恢复: {session_name} ({total_files} 张)", spinning=False, timeout_ms=2500)
                else:
                    self.set_app_status(f"底片库已就绪 ({total_files} 张)", spinning=False, timeout_ms=2500)

        def _on_failed(err_msg):
            if getattr(self, "_session_restore_worker", None) != worker:
                return
            self._session_restore_worker = None
            if self._start_next_pending_import_batch():
                return
            self._is_importing = False
            if hasattr(self, "btn_cancel_import"):
                self.btn_cancel_import.setVisible(False)
            self.set_app_status(f"载入会话失败: {err_msg}", spinning=False, timeout_ms=3500)

        worker.activePhotoReady.connect(_on_active_photo_ready)
        worker.progress.connect(_on_progress)
        worker.restoreFinished.connect(_on_restore_finished)
        worker.failed.connect(_on_failed)

        if os.environ.get("FAST_TEST"):
            worker.run_sync()
        else:
            worker.start()

    def _apply_win11_corners(self, is_max=False):
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            # When maximized, disable rounding to eliminate edge and corner gaps completely
            corner_pref = c_int(1 if is_max else 2)  # 1 = DONOTROUND, 2 = ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, byref(corner_pref), ctypes.sizeof(corner_pref)
            )
            # Enable Windows native Aero Snap, edge snapping, system menu and native minimize/maximize animations
            GWL_STYLE = -16
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_MAXIMIZEBOX = 0x00010000
            WS_MINIMIZEBOX = 0x00020000
            WS_SYSMENU = 0x00080000
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_STYLE, style | WS_CAPTION | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU
            )
            # Ensure window is registered as top-level app window in taskbar
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_APPWINDOW)

            SWP_NOZORDER = 0x0004
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_FRAMECHANGED = 0x0020
            ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)
        except Exception:
            pass

    def _apply_native_window_icon(self):
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            from core.version import get_app_title
            ctypes.windll.user32.SetWindowTextW(hwnd, get_app_title())

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

            # Ensure OS recognizes this as a standard top-level application window with Taskbar/Task Manager identity
            GWL_STYLE = -16
            WS_SYSMENU = 0x00080000
            WS_MINIMIZEBOX = 0x00020000
            WS_MAXIMIZEBOX = 0x00010000
            WS_CAPTION = 0x00C00000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_CAPTION)

            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, (ex_style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW)

            ctypes.windll.user32.SetWindowTextW(hwnd, self.windowTitle() or get_app_title())
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_win11_corners(self.isMaximized())
        self._apply_native_window_icon()
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
            if hasattr(self, 'btn_max'):
                self.btn_max.set_maximized_state(is_max)
                if hasattr(self, 'btn_close'):
                    self.btn_close.set_maximized_state(is_max)
            self._update_root_style(is_max)
            self._apply_win11_corners(is_max)
            self._apply_native_window_icon()
        elif event.type() == QEvent.Type.ActivationChange:
            self._apply_native_window_icon()
        super().changeEvent(event)

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
                    return True, 0

                # 1. WM_NCCALCSIZE (0x0083): Remove standard title bar and frame while keeping DWM animations
                if msg.message == 0x0083:
                    if msg.wParam and ctypes.windll.user32.IsZoomed(msg.hwnd):
                        h_mon = ctypes.windll.user32.MonitorFromWindow(msg.hwnd, 2)
                        mi = MONITORINFO()
                        mi.cbSize = ctypes.sizeof(MONITORINFO)
                        if ctypes.windll.user32.GetMonitorInfoW(h_mon, ctypes.byref(mi)):
                            params = NCCALCSIZE_PARAMS.from_address(msg.lParam)
                            params.rgrc[0].left = mi.rcWork.left
                            params.rgrc[0].top = mi.rcWork.top
                            params.rgrc[0].right = mi.rcWork.right
                            params.rgrc[0].bottom = mi.rcWork.bottom
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
        valid_exts = ALL_SUPPORTED_IMAGE_EXTENSIONS
        sdss_files = []
        img_files = []

        for url in event.mimeData().urls():
            if url.isLocalFile():
                local_path = url.toLocalFile()
                if os.path.exists(local_path):
                    if os.path.isdir(local_path):
                        for root, _, fnames in os.walk(local_path):
                            for f in fnames:
                                ext = os.path.splitext(f)[1].lower()
                                if ext == ".sdss":
                                    sdss_files.append(os.path.join(root, f))
                                elif ext in valid_exts:
                                    img_files.append(os.path.join(root, f))
                    else:
                        ext = os.path.splitext(local_path)[1].lower()
                        if ext == ".sdss":
                            sdss_files.append(local_path)
                        elif ext in valid_exts:
                            img_files.append(local_path)

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

    def _update_root_style(self, is_max):
        is_dark = theme_manager.is_dark()
        app_bg = "#121316" if is_dark else "#ebedf2"
        root_border_color = "#282a34" if is_dark else "#c8ccd8"
        tb_bg = "#141518" if is_dark else "#f0f2f6"
        scroll_handle = "#3f4350" if is_dark else "#94a3b8"
        scroll_hover = "#cbd5e1" if is_dark else "#64748b"
        tooltip_bg = "rgba(22, 24, 30, 0.95)" if is_dark else "#ffffff"
        tooltip_fg = "#f1f5f9" if is_dark else "#1e293b"
        tooltip_border = "1px solid #333644" if is_dark else "1px solid #cbd5e1"

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
                QToolTip {{
                    background-color: {tooltip_bg};
                    color: {tooltip_fg};
                    border: {tooltip_border};
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-size: 11px;
                    font-weight: 400;
                    line-height: 1.4;
                    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
                }}
            """)
        if hasattr(self, 'title_bar'):
            self.title_bar.setStyleSheet(f"background: {tb_bg}; border: none; border-top-left-radius: {tb_radius}; border-top-right-radius: {tb_radius};")

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
            self.lbl_paper_title.setStyleSheet(f"color: {paper_col}; font-size: 11.5px; font-weight: bold;")

        # 5. Right Sidebar Container
        if hasattr(self, 'sidebar_container'):
            self.sidebar_container.setStyleSheet(f"background: {colors['panel_bg']};")

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
            ('act_save_session', "save.png"),
            ('act_quick_export', "quick_export.png"),
            ('act_export', "export.png"),
            ('act_undo', "undo.png"),
            ('act_redo', "redo.png"),
            ('act_crop', "crop.png"),
            ('act_reset_all', "reset.png"),
            ('act_clear_crop', "crop.png"),
            ('act_clear_sdc', "reset.png"),
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
            self.lbl_photo_info.setStyleSheet(f"QLabel#lbl_photo_info {{ color: {colors['text_secondary']}; font-size: 11px; font-weight: 500; padding-top: 1px; }}")
        if hasattr(self, 'status_label'):
            self.status_label.setStyleSheet(f"QLabel#status_label {{ color: {colors['text_muted']}; font-size: 11px; padding-top: 1px; }}")

        btn_bg = colors['view_toolbar_btn_bg']
        btn_border = colors['view_toolbar_btn_border']
        btn_hover = colors['view_toolbar_btn_hover']
        btn_pressed = "#14151a" if is_dark else "#cbd5e1"
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
        chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")

        chk_ss = get_standard_checkbox_style(is_dark)

        for chk in (getattr(self, 'chk_remove_ca', None), getattr(self, 'chk_lens_profile_enabled', None)):
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
                font-size: 11.5px;
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
        """
        for btn in (getattr(self, 'btn_lens_tab_profile', None), getattr(self, 'btn_lens_tab_manual', None)):
            if btn:
                btn.setStyleSheet(tab_ss)

        if getattr(self, 'lens_tab_frame', None):
            line_col = "#282a36" if is_dark else "#cbd5e1"
            self.lens_tab_frame.setStyleSheet(f"QFrame#lensTabFrame {{ border: none; border-bottom: 1px solid {line_col}; background: transparent; }}")

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
        """
        for lbl in (getattr(self, 'lbl_profile_header', None), getattr(self, 'lbl_amount_header', None), getattr(self, 'lbl_vignette_header', None), getattr(self, 'lbl_diff_header', None)):
            if lbl:
                lbl.setStyleSheet(header_ss)

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
        self.lbl_title_export_text.setStyleSheet("background: transparent; border: none; color: #111113; font-weight: bold; font-size: 11.5px;")
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
        lp_layout.setContentsMargins(4, 4, 4, 4)
        lp_layout.setSpacing(4)

        # 3-tier vertical splitter (Delicate amber hairline fading at both ends)
        self.left_splitter = FadingSplitter(Qt.Orientation.Vertical)

        # Sub-window 1: 胶卷型号 (Films)
        film_subwin = QWidget()
        film_layout = QVBoxLayout(film_subwin)
        film_layout.setContentsMargins(0, 0, 0, 0)
        film_layout.setSpacing(4)

        self.film_header = FilmFilterHeader()
        self.lbl_film_title = self.film_header.lbl_title
        self.film_header.searchChanged.connect(self._on_film_search_changed)
        self.film_header.categoryChanged.connect(self._on_film_category_changed)
        film_layout.addWidget(self.film_header)

        self.film_scroll = SmoothScrollArea()
        self.film_scroll.setWidgetResizable(True)
        self.film_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.film_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.film_container = QWidget()
        self.film_grid = QGridLayout(self.film_container)
        self.film_grid.setContentsMargins(4, 4, 4, 4)
        self.film_grid.setHorizontalSpacing(6)
        self.film_grid.setVerticalSpacing(6)
        self.film_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.film_scroll.setWidget(self.film_container)

        self.film_scroll.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.film_scroll.customContextMenuRequested.connect(self._on_film_blank_context_menu)
        self.film_container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.film_container.customContextMenuRequested.connect(self._on_film_blank_context_menu)

        self._film_vp_filter = ViewportResizeFilter(self.reflow_grids, self)
        self.film_scroll.installEventFilter(self._film_vp_filter)

        film_layout.addWidget(self.film_scroll, 1)

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
        self.lbl_paper_title.setStyleSheet("color: #f59e0b; font-size: 11.5px; font-weight: bold;")
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
        self.left_splitter.addWidget(paper_subwin)

        # Sub-window 3: 元数据属性 (EXIF)
        exif_subwin = QWidget()
        exif_layout = QVBoxLayout(exif_subwin)
        exif_layout.setContentsMargins(0, 0, 0, 0)
        exif_layout.setSpacing(0)

        self.exif_widget = CompactExifWidget()
        exif_layout.addWidget(self.exif_widget, 1)

        self.left_splitter.addWidget(exif_subwin)

        self.left_splitter.setStretchFactor(0, 4)
        self.left_splitter.setStretchFactor(1, 3)
        self.left_splitter.setStretchFactor(2, 3)
        self.left_splitter.setSizes([300, 180, 200])

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
        self.view_toolbar = QFrame()
        self.view_toolbar.setFixedHeight(28)
        self.view_toolbar.setStyleSheet(f"background: {theme_manager.color('view_toolbar_bg')}; border: none;")
        tb_layout = QHBoxLayout(self.view_toolbar)
        tb_layout.setContentsMargins(10, 2, 10, 1)
        tb_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        tb_layout.setSpacing(6)

        # Permanent photo filename & zoom percentage display (never erased or overwritten by transient notifications)
        self.lbl_photo_info = QLabel("")
        self.lbl_photo_info.setObjectName("lbl_photo_info")
        self.lbl_photo_info.setFixedWidth(160)
        self.lbl_photo_info.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.lbl_photo_info.setStyleSheet("QLabel#lbl_photo_info { color: #cbd5e1; font-size: 11px; font-weight: 500; padding-top: 1px; }")
        tb_layout.addWidget(self.lbl_photo_info)

        # Subtle separator between permanent photo info and transient system notification
        self.status_sep = QLabel("│")
        self.status_sep.setObjectName("status_sep")
        self.status_sep.setFixedWidth(10)
        self.status_sep.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter)
        self.status_sep.setStyleSheet("QLabel#status_sep { color: transparent; font-size: 11px; padding-top: 1px; }")
        tb_layout.addWidget(self.status_sep)

        # Left: Spinner & Status Label (Item 6 & 8)
        self.spinner = SpinnerWidget(size=13)
        tb_layout.addWidget(self.spinner)

        self.status_label = QLabel("")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.status_label.setStyleSheet("QLabel#status_label { color: #94a3b8; font-size: 11px; padding-top: 1px; }")
        self.status_label.mousePressEvent = self._on_status_label_clicked
        tb_layout.addWidget(self.status_label)

        self.btn_cancel_import = QPushButton("✕ 取消导入")
        self.btn_cancel_import.setVisible(False)
        self.btn_cancel_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel_import.setToolTip("中断当前正在进行的底片批量导入流程")
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
        self.btn_cancel_import.clicked.connect(lambda: self._cancel_active_import(notify_user=True))
        tb_layout.addWidget(self.btn_cancel_import)

        tb_layout.addStretch(1)

        # Right: Dedicated Export Status Prompt (Pure text + underline, matching left status bar)
        self.export_status_container = QWidget()
        self.export_status_container.setObjectName("export_status_container")
        self.export_status_container.setVisible(False)
        self.export_status_container.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_status_container.setToolTip("点击查看扫描队列详情 (Ctrl+J)")
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
        self.lbl_export_status.setStyleSheet("QLabel#lbl_export_status { color: #f59e0b; font-size: 11px; text-decoration: underline; padding-top: 1px; }")
        export_layout.addWidget(self.lbl_export_status)

        self.export_status_container.setStyleSheet("""
            QWidget#export_status_container {
                background: transparent;
                border: none;
                padding: 0;
            }
        """)
        tb_layout.addWidget(self.export_status_container)

        self.export_sep = QLabel("│")
        self.export_sep.setObjectName("export_sep")
        self.export_sep.setFixedWidth(12)
        self.export_sep.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter)
        self.export_sep.setStyleSheet("QLabel#export_sep { color: #2e3444; font-size: 11px; padding-top: 1px; }")
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
        self.btn_crop.setToolTip("裁切与校平 (快捷键 C)")
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

        self.btn_split = CompareButton(self)
        self.btn_split.setFixedSize(28, 22)
        self.btn_split.setIcon(get_icon("compare.png"))
        self.btn_split.setIconSize(QSize(14, 14))
        self.btn_split.setCheckable(True)
        self.btn_split.setToolTip("对比视图 (快捷键 \\): 点击切换分屏对比，长按查看原片")
        self.btn_split.setStyleSheet(btn_style)
        self.btn_split.clicked.connect(self.toggle_split_view)
        self.btn_split.longPressStarted.connect(self._on_compare_long_press_start)
        self.btn_split.longPressFinished.connect(self._on_compare_long_press_end)
        tb_layout.addWidget(self.btn_split)

        self.btn_fit = QPushButton()
        self.btn_fit.setFixedSize(28, 22)
        self.btn_fit.setIcon(get_icon("fit_window.png"))
        self.btn_fit.setIconSize(QSize(14, 14))
        self.btn_fit.setCheckable(True)
        self.btn_fit.setToolTip("适应图像全貌 (Ctrl+0)")
        self.btn_fit.setStyleSheet(btn_style)
        self.btn_fit.clicked.connect(self._on_btn_fit_clicked)
        tb_layout.addWidget(self.btn_fit)

        self.btn_100 = QPushButton()
        self.btn_100.setFixedSize(28, 22)
        self.btn_100.setIcon(get_icon("zoom_100.png"))
        self.btn_100.setIconSize(QSize(14, 14))
        self.btn_100.setCheckable(True)
        self.btn_100.setToolTip("1:1 实际像素 (Ctrl+1)")
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
        self.sidebar_container.setStyleSheet(f"background: {theme_manager.color('panel_bg')};")
        sb_outer_layout = QVBoxLayout(self.sidebar_container)
        sb_outer_layout.setContentsMargins(8, 8, 8, 8)
        sb_outer_layout.setSpacing(6)

        # Top Pinned Real-time RGB & Luma Histogram (始终置顶不跟随滚动)
        self.histogram_widget = HistogramWidget(self)
        sb_outer_layout.addWidget(self.histogram_widget)
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.setInterval(40)
        self._hist_timer.timeout.connect(self._compute_and_update_histogram)

        sidebar_scroll = SmoothScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.sidebar_scroll = sidebar_scroll

        self.sidebar_content = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_content)
        self.sidebar_layout.setContentsMargins(0, 4, 0, 0)
        self.sidebar_layout.setSpacing(6)

        # Sections (Comprehensive Physical Analog Pipeline)
        self._init_exposure_section()
        self._init_optics_section()
        self._init_chemistry_section()
        self._init_enlarger_section()
        self._init_halation_section()
        self._init_grain_section()
        self.sidebar_layout.addStretch(1)

        sidebar_scroll.setWidget(self.sidebar_content)

        self.sidebar_stack = QStackedWidget()
        self.sidebar_stack.addWidget(sidebar_scroll)
        self.crop_sidebar = CropSidebarWidget(self.canvas.crop_overlay, self)
        self.sidebar_stack.addWidget(self.crop_sidebar)
        sb_outer_layout.addWidget(self.sidebar_stack, 1)

        if hasattr(self.canvas, 'crop_overlay'):
            self.canvas.crop_overlay.cropApplied.connect(self._on_crop_applied)
            self.canvas.crop_overlay.cropCancelled.connect(self._on_crop_cancelled)
            self.canvas.crop_overlay.cropChanged.connect(self._on_crop_live)

        self.main_splitter.addWidget(self.sidebar_container)
        self.main_splitter.setCollapsible(2, False)

        # Splitter proportion defaults: Left panel 320px (comfortably fits 3 columns), Center 800px, Right 320px
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([320, 800, 320])
        self.main_splitter.splitterMoved.connect(lambda pos, idx: self.reflow_grids())

        e_layout.addWidget(self.main_splitter)
        self.stack.addWidget(self.editor_widget)
        self.stack.currentChanged.connect(self._on_stack_page_changed)

    def _on_stack_page_changed(self, index):
        if index == 1:
            # Editor page activated: restore persisted splitter layout and reflow grids
            win_cfg = config_manager.get_window_config()
            if hasattr(self, "main_splitter") and "splitter_main" in win_cfg:
                sm = win_cfg["splitter_main"]
                if len(sm) == 3 and all(s > 0 for s in sm):
                    self.main_splitter.setSizes(sm)
            if hasattr(self, "left_splitter") and "splitter_left" in win_cfg:
                sl = win_cfg["splitter_left"]
                if len(sl) == 3 and all(s > 0 for s in sl):
                    self.left_splitter.setSizes(sl)
            if hasattr(self, "center_splitter") and "splitter_center" in win_cfg:
                sc = win_cfg["splitter_center"]
                if len(sc) == 2 and all(s > 0 for s in sc):
                    self.center_splitter.setSizes(sc)
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
        for group in stocks:
            for item in group.get("stocks", group.get("items", [])):
                card = FilmGridCard(item, is_active=(item["id"] == self.current_film_stock), is_custom=False, stock_type="film")
                card.selected.connect(self.on_switch_film)
                card.sortRequested.connect(lambda st, sb: self._sort_and_reflow_stocks(st, sb))
                self.film_cards[item["id"]] = card
        if hasattr(self, 'film_header'):
            self.film_header.set_title_count(len(self.film_cards), len(self.film_cards))
        elif hasattr(self, 'lbl_film_title'):
            self.lbl_film_title.setText(f"胶卷型号 ({len(self.film_cards)} 卷)")
        self.reflow_grids()

    def _populate_paper_grid(self):
        papers = self.engine.get_paper_stocks()
        for item in papers:
            card = FilmGridCard(item, is_active=(item["id"] == self.current_paper_stock), is_custom=False, stock_type="paper")
            card.selected.connect(self.on_switch_paper)
            card.sortRequested.connect(lambda st, sb: self._sort_and_reflow_stocks(st, sb))
            self.paper_cards[item["id"]] = card
        if hasattr(self, 'lbl_paper_title'):
            self.lbl_paper_title.setText(f"放大相纸 ({len(self.paper_cards)} 张)")
        self.reflow_grids()

    def _init_exposure_section(self):
        self.sec_exposure = AccordionSection("场景曝光")
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
        self.sec_optics = AccordionSection("光学")
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

        # 2. Checkbox: 删除色差
        chk_layout = QHBoxLayout()
        chk_layout.setContentsMargins(0, 2, 0, 2)
        self.chk_remove_ca = QCheckBox("删除色差")
        self.chk_remove_ca.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_remove_ca.setToolTip("通过反向色差径向位移算法校正横向色差（紫边/绿边）")
        self.chk_remove_ca.toggled.connect(self._on_lens_remove_ca_toggled)
        chk_layout.addWidget(self.chk_remove_ca)
        self.sec_optics.addLayout(chk_layout)

        # 3. Profile Panel Container
        self.panel_lens_profile = QWidget()
        self.panel_lens_profile.setStyleSheet("background: transparent; border: none;")
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
            "设置",
            [
                ("默认 (自动读取 EXIF)", "auto"),
                ("自定", "custom"),
            ],
            default_val="auto",
            tooltip="镜头校正配置文件设置来源：自动读取 EXIF 或自定义手动选取"
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
            "品牌",
            make_items,
            default_val="",
            tooltip="镜头制造商品牌"
        )
        self.combo_lens_make.valueChanged.connect(self._on_lens_make_changed)
        profile_layout.addWidget(self.combo_lens_make)

        self.combo_lens_model = ComboBoxRow(
            "机型",
            [("无", "")],
            default_val="",
            tooltip="具体镜头型号"
        )
        self.combo_lens_model.valueChanged.connect(self._on_lens_model_changed)
        profile_layout.addWidget(self.combo_lens_model)

        self.combo_lens_profile = ComboBoxRow(
            "配置文件",
            [("无", "")],
            default_val="",
            tooltip="匹配的标准镜头校正配置文件 (.lcp)"
        )
        self.combo_lens_profile.valueChanged.connect(self._on_lens_profile_changed)
        profile_layout.addWidget(self.combo_lens_profile)

        self.lbl_amount_header = QLabel("校正量")
        self.lbl_amount_header.setObjectName("subSectionHeader")
        profile_layout.addWidget(self.lbl_amount_header)

        self.slider_lens_distortion = AdobeScrubSlider(
            "扭曲度", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            has_center_detent=True,
            tooltip="配置文件几何畸变校正量 (0 为标准出厂值，向左减弱，向右增强)"
        )
        self.slider_lens_distortion.valueChanged.connect(lambda v: self._on_lens_amount_changed("profile_distortion", v))
        self.slider_lens_distortion.sliderReleased.connect(self._on_slider_committed)
        profile_layout.addWidget(self.slider_lens_distortion)

        self.slider_lens_vignette = AdobeScrubSlider(
            "晕影", -100.0, 100.0, 0.0, step=1.0, unit="", decimals=0,
            has_center_detent=True,
            tooltip="配置文件暗角光学补偿量 (0 为标准出厂值，负值加深暗角，正值增亮消除暗角)"
        )
        self.slider_lens_vignette.valueChanged.connect(lambda v: self._on_lens_amount_changed("profile_vignette", v))
        self.slider_lens_vignette.sliderReleased.connect(self._on_slider_committed)
        profile_layout.addWidget(self.slider_lens_vignette)

        self.sec_optics.addWidget(self.panel_lens_profile)

        # 4. Manual Panel Container
        self.panel_lens_manual = QWidget()
        self.panel_lens_manual.setStyleSheet("background: transparent; border: none;")
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
            tooltip="暗角羽化过渡中点：0 为标准中点，负值向中心收缩，正值向边缘扩展"
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
                ("无柔光滤镜", "none"),
                ("Black Pro-Mist (黑柔)", "black_pro_mist"),
                ("Glimmerglass (微光)", "glimmerglass"),
                ("Pro-Mist (经典白柔)", "pro_mist"),
                ("CineBloom (电影漫射)", "cinebloom"),
            ],
            default_val="none",
            tooltip="经典电影柔光滤镜仿真，在高光周围形成漫射晕开质感"
        )
        self.combo_diffusion.valueChanged.connect(lambda v: self._on_combo_changed("diffusion_family", v))
        self.sec_optics.addWidget(self.combo_diffusion)

        self.combo_diff_strength = ComboBoxRow(
            "柔光挡位",
            [
                ("0 (无滤镜)", 0.0),
                ("1/8 档", 0.125),
                ("1/4 档", 0.25),
                ("1/2 档", 0.5),
                ("1 档", 1.0),
                ("2 档", 2.0),
            ],
            default_val=0.0,
            tooltip="柔光镜散射密度挡位 (1/8 ~ 2 档)"
        )
        self.combo_diff_strength.valueChanged.connect(lambda v: self._on_combo_changed("diffusion_strength", v))
        self.slider_diff_strength = self.combo_diff_strength  # Backward compatibility alias
        self.sec_optics.addWidget(self.combo_diff_strength)

        self.slider_diff_warmth = AdobeScrubSlider(
            "光晕冷暖偏置", -1.5, 1.5, 0.0, step=0.05, unit="", decimals=2,
            tooltip="柔光光晕波长能量再分配，正值偏复古暖金，负值偏清冷晨光",
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

    def _init_chemistry_section(self):
        self.sec_chemistry = AccordionSection("胶片乳剂与化学 (DIR)")
        self.sec_chemistry.resetClicked.connect(lambda: self.reset_track("chemistry"))

        self.slider_dir_amount = AdobeScrubSlider(
            "抑制剂总量", 0.0, 2.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="DIR 显影抑制耦合剂全局强度乘数 (1.0 为出厂标准值)"
        )
        self.slider_dir_amount.valueChanged.connect(lambda v: self._on_slider_live("dir_amount", v))
        self.slider_dir_amount.sliderReleased.connect(self._on_slider_committed)
        self.sec_chemistry.addWidget(self.slider_dir_amount)

        self.slider_dir_interlayer = AdobeScrubSlider(
            "跨层色彩纯度", 0.0, 2.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="相邻感光层相互抑制释放，形成高纯度通透色彩分离"
        )
        self.slider_dir_interlayer.valueChanged.connect(lambda v: self._on_slider_live("dir_interlayer", v))
        self.slider_dir_interlayer.sliderReleased.connect(self._on_slider_committed)
        self.sec_chemistry.addWidget(self.slider_dir_interlayer)

        self.slider_dir_samelayer = AdobeScrubSlider(
            "同层边缘反差", 0.0, 2.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="单色层自抑制微反差控制，防止大光比下色彩密度堆积过厚"
        )
        self.slider_dir_samelayer.valueChanged.connect(lambda v: self._on_slider_live("dir_samelayer", v))
        self.slider_dir_samelayer.sliderReleased.connect(self._on_slider_committed)
        self.sec_chemistry.addWidget(self.slider_dir_samelayer)

        self.sidebar_layout.addWidget(self.sec_chemistry)

    def _init_enlarger_section(self):
        self.sec_enlarger = AccordionSection("放大机与相纸显影")
        self.sec_enlarger.resetClicked.connect(lambda: self.reset_track("enlarger"))

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
            tooltip="放大机投光光源光谱分布，影响相纸感光层综合响应"
        )
        self.combo_illuminant.valueChanged.connect(lambda v: self._on_combo_changed("enlarger_illuminant", v))
        self.sec_enlarger.addWidget(self.combo_illuminant)

        self.slider_magenta = AdobeScrubSlider(
            "品红 / 绿色", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="放大机滤色镜品红 (M) 调节，左偏品红，右偏绿色",
            track_gradient=[(0.0, QColor("#f43f5e")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#10b981"))],
            has_center_detent=True
        )
        self.slider_magenta.valueChanged.connect(lambda v: self._on_slider_live("enlarger_magenta", v))
        self.slider_magenta.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_magenta)

        self.slider_yellow = AdobeScrubSlider(
            "黄色 / 蓝色", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="放大机滤色镜黄色 (Y) 调节，左偏黄色，右偏蓝色",
            track_gradient=[(0.0, QColor("#eab308")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#38bdf8"))],
            has_center_detent=True
        )
        self.slider_yellow.valueChanged.connect(lambda v: self._on_slider_live("enlarger_yellow", v))
        self.slider_yellow.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_yellow)

        self.slider_cyan = AdobeScrubSlider(
            "青色 / 红色", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="放大机滤色镜青色 (C) 调节，左偏青色，右偏红色",
            track_gradient=[(0.0, QColor("#06b6d4")), (0.5, QColor("#cbd5e1")), (1.0, QColor("#f87171"))],
            has_center_detent=True
        )
        self.slider_cyan.valueChanged.connect(lambda v: self._on_slider_live("enlarger_cyan", v))
        self.slider_cyan.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_cyan)

        self.slider_print_exp = AdobeScrubSlider(
            "相纸曝光", 0.5, 2.5, 1.0, step=0.02, unit="x", decimals=2,
            tooltip="放大机相纸曝光时间，调节相纸整体显影浓度",
            track_gradient=[(0.0, QColor("#14161d")), (1.0, QColor("#f1f5f9"))],
            has_center_detent=False
        )
        self.slider_print_exp.valueChanged.connect(lambda v: self._on_slider_live("print_exposure", v))
        self.slider_print_exp.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_print_exp)

        self.slider_preflash = AdobeScrubSlider(
            "预闪反差", 0.0, 0.25, 0.0, step=0.01, unit="", decimals=2,
            tooltip="暗房预闪光技术 (Pre-flash)，降低相纸反差并显现极暗部细节"
        )
        self.slider_preflash.valueChanged.connect(lambda v: self._on_slider_live("pre_flash", v))
        self.slider_preflash.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_preflash)

        self.slider_morph_gamma = AdobeScrubSlider(
            "显影形态反差", 0.70, 1.30, 1.0, step=0.02, unit="", decimals=2,
            tooltip="相纸 D-logH 显影曲线斜率形态，影响中间调与暗部陡峭度"
        )
        self.slider_morph_gamma.valueChanged.connect(lambda v: self._on_slider_live("morph_gamma", v))
        self.slider_morph_gamma.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_morph_gamma)

        self.slider_exhaustion = AdobeScrubSlider(
            "显影液疲劳度", 0.0, 0.50, 0.0, step=0.02, unit="", decimals=2,
            tooltip="模拟显影液氧化衰老，柔化死黑黑位并呈现复古相纸暗部质感"
        )
        self.slider_exhaustion.valueChanged.connect(lambda v: self._on_slider_live("developer_exhaustion", v))
        self.slider_exhaustion.sliderReleased.connect(self._on_slider_committed)
        # Backward compatibility aliases
        self.slider_enlarger_c = self.slider_cyan
        self.slider_enlarger_m = self.slider_magenta
        self.slider_enlarger_y = self.slider_yellow

        self.sidebar_layout.addWidget(self.sec_enlarger)

    def _init_halation_section(self):
        self.sec_halation = AccordionSection("胶片物理光晕 (Halation)")
        self.sec_halation.resetClicked.connect(lambda: self.reset_track("halation"))

        self.slider_halation_amount = AdobeScrubSlider(
            "光晕物理倍率", 0.0, 3.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="背反射物理光晕倍率，按防光晕层实测系数缩放各通道振幅"
        )
        self.slider_halation_amount.valueChanged.connect(lambda v: self._on_slider_live("halation_amount", v))
        self.slider_halation_amount.sliderReleased.connect(self._on_slider_committed)
        self.sec_halation.addWidget(self.slider_halation_amount)

        self.slider_halation_spatial_scale = AdobeScrubSlider(
            "光晕空间尺度", 0.1, 3.0, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="光晕物理扩散核空间尺度倍率"
        )
        self.slider_halation_spatial_scale.valueChanged.connect(lambda v: self._on_slider_live("halation_spatial_scale", v))
        self.slider_halation_spatial_scale.sliderReleased.connect(self._on_slider_committed)
        self.sec_halation.addWidget(self.slider_halation_spatial_scale)

        self.slider_scatter_amount = AdobeScrubSlider(
            "乳剂内散射强度", 0.0, 2.0, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="感光乳剂层内前向光散射强度，模拟高光微粒漫射 (Light Wrap)"
        )
        self.slider_scatter_amount.valueChanged.connect(lambda v: self._on_slider_live("scatter_amount", v))
        self.slider_scatter_amount.sliderReleased.connect(self._on_slider_committed)
        self.sec_halation.addWidget(self.slider_scatter_amount)

        self.slider_boost_ev = AdobeScrubSlider(
            "高光能量重建", 0.0, 4.0, 0.0, step=0.05, unit=" EV", decimals=2,
            tooltip="超大光比高光点能量重构 (EV)，呈现自然真实的胶片逆光包容感"
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

    def _init_grain_section(self):
        self.sec_grain = AccordionSection("胶片银盐颗粒 (Grain)")
        self.sec_grain.resetClicked.connect(lambda: self.reset_track("grain"))

        self.slider_grain = AdobeScrubSlider(
            "银盐颗粒感", 0.0, 1.0, 0.4, step=0.02, unit="", decimals=2,
            tooltip="卤化银晶体显影后的微观有机颗粒结构强度"
        )
        self.slider_grain.valueChanged.connect(lambda v: self._on_slider_live("grain", v))
        self.slider_grain.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_grain)

        self.slider_grain_size = AdobeScrubSlider(
            "颗粒物理尺寸", 0.5, 2.5, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="卤化银微晶与发色染料微云物理尺度倍率"
        )
        self.slider_grain_size.valueChanged.connect(lambda v: self._on_slider_live("grain_size", v))
        self.slider_grain_size.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_grain_size)

        self.slider_grain_cloud = AdobeScrubSlider(
            "染料云团柔化", 0.5, 2.5, 1.0, step=0.05, unit=" um", decimals=2,
            tooltip="柔化噪点边缘，呈现彩色胶片发色染料云团 (Dye Clouds) 真实形态"
        )
        self.slider_grain_cloud.valueChanged.connect(lambda v: self._on_slider_live("grain_cloud_blur", v))
        self.slider_grain_cloud.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_grain_cloud)

        self.slider_grain_colour = AdobeScrubSlider(
            "色彩微粒 (彩噪)", 0.0, 1.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="0.0为纯单色明度颗粒，1.0为真实彩色胶片独立多层染料云(彩噪)"
        )
        self.slider_grain_colour.valueChanged.connect(lambda v: self._on_slider_live("grain_colour", v))
        self.slider_grain_colour.sliderReleased.connect(self._on_slider_committed)
        self.sec_grain.addWidget(self.slider_grain_colour)

        self.sidebar_layout.addWidget(self.sec_grain)

    _init_texture_section = _init_halation_section
    _init_optical_section = _init_halation_section

    def _init_menu_and_actions(self):
        # 1. File Menu
        file_menu = self.menu_bar.addMenu("文件 (F)")
        
        self.act_open_file = QAction("打开文件...", self)
        self.act_open_file.setIcon(get_icon("open_file.png"))
        self.act_open_file.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open_file.triggered.connect(self.action_open_files)
        file_menu.addAction(self.act_open_file)

        act_open_folder = QAction("打开文件夹...", self)
        act_open_folder.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_open_folder.triggered.connect(self.action_open_folder)
        file_menu.addAction(act_open_folder)

        act_open_session = QAction("打开会话工程 (.sdss)...", self)
        act_open_session.triggered.connect(self.action_open_session)
        file_menu.addAction(act_open_session)

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

        act_exit = QAction("退出", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # 2. Edit Menu (Item 5: Simplified concise labels, 13px icons)
        edit_menu = self.menu_bar.addMenu("编辑 (E)")
        
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

        act_copy_params = QAction("复制所有参数", self)
        act_copy_params.setShortcut(QKeySequence("Ctrl+C"))
        act_copy_params.triggered.connect(self.action_copy_params)
        edit_menu.addAction(act_copy_params)

        act_paste_params = QAction("粘贴所有参数", self)
        act_paste_params.setShortcut(QKeySequence("Ctrl+V"))
        act_paste_params.triggered.connect(self.action_paste_params)
        edit_menu.addAction(act_paste_params)

        act_select_all = QAction("全选底片", self)
        act_select_all.setShortcut(QKeySequence("Ctrl+A"))
        act_select_all.triggered.connect(self.action_select_all_photos)
        edit_menu.addAction(act_select_all)

        edit_menu.addSeparator()

        self.act_reset_all = QAction("重置所有参数", self)
        self.act_reset_all.setIcon(get_icon("reset.png"))
        self.act_reset_all.setShortcut(QKeySequence("Ctrl+R"))
        self.act_reset_all.triggered.connect(self.reset_all_params)
        edit_menu.addAction(self.act_reset_all)

        self.act_clear_crop = QAction("清除裁切", self)
        self.act_clear_crop.setIcon(get_icon("crop.png"))
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

        # 3. View Menu
        view_menu = self.menu_bar.addMenu("视图 (V)")
        
        self.act_fit_img = QAction("适应图像", self)
        self.act_fit_img.setIcon(get_icon("fit_window.png"))
        self.act_fit_img.setShortcut(QKeySequence("Ctrl+0"))
        self.act_fit_img.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(self.act_fit_img)

        self.act_compare = QAction("对比\t\\", self)
        self.act_compare.setIcon(get_icon("compare.png"))
        self.act_compare.triggered.connect(self.toggle_split_view)
        view_menu.addAction(self.act_compare)

        self.act_fit_win = QAction("适应窗口", self)
        self.act_fit_win.setIcon(get_icon("fit_window.png"))
        self.act_fit_win.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(self.act_fit_win)

        self.act_100 = QAction("1:1 实际像素", self)
        self.act_100.setIcon(get_icon("zoom_100.png"))
        self.act_100.setShortcut(QKeySequence("Ctrl+1"))
        self.act_100.triggered.connect(self.canvas.set_zoom_100)
        view_menu.addAction(self.act_100)

        act_zoom_in = QAction("放大", self)
        act_zoom_in.setShortcuts([
            QKeySequence.StandardKey.ZoomIn,
            QKeySequence("Ctrl+="),
            QKeySequence("Ctrl++")
        ])
        act_zoom_in.triggered.connect(lambda: self.canvas.zoom_in(animate=True))
        view_menu.addAction(act_zoom_in)
        self.addAction(act_zoom_in)

        act_zoom_out = QAction("缩小", self)
        act_zoom_out.setShortcuts([
            QKeySequence.StandardKey.ZoomOut,
            QKeySequence("Ctrl+-")
        ])
        act_zoom_out.triggered.connect(lambda: self.canvas.zoom_out(animate=True))
        view_menu.addAction(act_zoom_out)
        self.addAction(act_zoom_out)

        view_menu.addSeparator()

        act_reset_layout = QAction("恢复默认布局", self)
        act_reset_layout.setShortcut(QKeySequence("Ctrl+Shift+R"))
        act_reset_layout.triggered.connect(self.reset_default_layout)
        view_menu.addAction(act_reset_layout)

        # 4. Help Menu (Item 8: Version in About)
        help_menu = self.menu_bar.addMenu("帮助 (H)")
        act_about = QAction("关于 SpektraDarkroom", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def reset_default_layout(self):
        """Restore all splitters to optimal default layout."""
        if hasattr(self, "main_splitter"):
            self.main_splitter.setSizes([320, 800, 320])
        if hasattr(self, "left_splitter"):
            self.left_splitter.setSizes([300, 180, 200])
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
        self._photo_is_dirty = True
        if self.active_photo_id:
            photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
            if photo:
                photo["is_dirty"] = True
                photo["film_profile"] = self.current_film_stock
                photo["paper_profile"] = self.current_paper_stock
                photo["params"] = dict(self.current_params)
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
            self.act_save_session_as.setEnabled(has_session)
        if hasattr(self, 'act_close_session'):
            self.act_close_session.setEnabled(has_session)

        # Only display pure app brand on the startup welcome screen when NO session is loaded
        if not self.photos and not self._current_session_path and is_startup:
            self.setWindowTitle(base_title)
            if sys.platform == "win32":
                try:
                    import ctypes
                    hwnd = int(self.winId())
                    ctypes.windll.user32.SetWindowTextW(hwnd, base_title)
                except Exception:
                    pass
            if hasattr(self, 'lbl_brand'):
                self.lbl_brand.setText("SpektraDarkroom")
            self._update_photo_info_display()
            return

        is_temp = getattr(self, "_is_temp_session", False)
        session_name = os.path.basename(self._current_session_path) if (self._current_session_path and not is_temp) else "未命名会话"
        dirty_flag = "*" if getattr(self, "_session_is_dirty", False) else ""
        display_title = f"{dirty_flag}{session_name}"

        full_title = f"{display_title} - {base_title}"
        self.setWindowTitle(full_title)
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = int(self.winId())
                ctypes.windll.user32.SetWindowTextW(hwnd, full_title)
            except Exception:
                pass
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

    def _on_filmstrip_selection_changed(self, selected_ids):
        """Item 10: Permanent multi-selection notification without blinking or auto-hide."""
        count = len(selected_ids)
        if count > 1:
            self._is_multi_selecting = True
            self.set_app_status(f"已选中 {count} 张底片", spinning=False, timeout_ms=0)
        else:
            if getattr(self, "_is_multi_selecting", False):
                self._is_multi_selecting = False
                self.restore_default_status()

    def restore_default_status(self):
        """Restores transient status to default or clears it while preserving permanent photo info."""
        self._update_photo_info_display()
        if getattr(self, "_is_importing", False) or getattr(self, "_is_multi_selecting", False):
            return
        if hasattr(self, "export_queue") and self.export_queue and self.export_queue.has_active_tasks():
            return
        if hasattr(self, "_status_timer") and self._status_timer.isActive():
            return
        self.status_label.setText("")
        self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.status_label.setToolTip("")
        self.spinner.stop()
        if hasattr(self, "status_sep"):
            self.status_sep.setStyleSheet("color: transparent; font-size: 11px;")

    def action_save_current(self):
        if self._current_photo_path:
            sdc_manager.save_sdc(
                self._current_photo_path,
                self.current_film_stock,
                self.current_paper_stock,
                self.current_params
            )
            self._photo_is_dirty = False
            if self.active_photo_id:
                photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
                if photo:
                    photo["is_dirty"] = False
                self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            self._update_window_title()
            self.set_app_status(f"已保存更改: {os.path.basename(self._current_photo_path)}", timeout_ms=4000)

    def _on_backslash_timeout(self):
        if getattr(self, "_backslash_pressed", False):
            self._backslash_long_pressed = True
            self._on_compare_long_press_start()

    def eventFilter(self, watched, event):
        """Handle global backslash key and navigation shortcuts."""
        et = event.type()

        if et in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            # Do not intercept if a dialog or child modal is active and event belongs to it
            if watched and hasattr(watched, "window") and watched.window() != self:
                return super().eventFilter(watched, event)
            # Do not intercept if user is typing in a text/input field
            from PySide6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox
            input_types = (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)
            focus_w = QApplication.focusWidget()
            if isinstance(watched, input_types) or isinstance(focus_w, input_types):
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
        focus_w = QApplication.focusWidget()
        from PySide6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox
        is_typing = isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox))

        if not is_typing:
            mods = event.modifiers()
            has_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            has_alt = bool(mods & Qt.KeyboardModifier.AltModifier)

            # Zoom shortcuts: Ctrl + Plus / Equal / Minus / Underscore / 0 / 1
            if has_ctrl and not has_alt:
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

    def set_app_status(self, text: str, spinning: bool = False, timeout_ms: int = 0, is_export_link: bool = False):
        """Centralized status indicator with animated spinner and comfortable lingering timer."""
        # If legacy call routed export link here, redirect directly to right-side capsule
        if is_export_link:
            self._set_export_status(text, state="busy" if spinning else "success", timeout_ms=timeout_ms)
            return

        # Lock status during batch import so auxiliary switch events don't wipe it
        if getattr(self, "_is_importing", False):
            is_import_msg = any(text.startswith(prefix) for prefix in [
                "正在导入", "导入底片", "正在恢复", "底片库", "会话底片库", "已导入", "已取消",
                "正在载入", "载入会话", "载入底片", "正在加载", "加载会话", "会话已恢复", "会话"
            ])
            if not is_import_msg:
                return
        # Lock status during multi-selection so photo click doesn't wipe selection count
        if getattr(self, "_is_multi_selecting", False) and not spinning:
            if not text.startswith("已选中") and not text.startswith("已全选"):
                return

        if hasattr(self, "_status_timer"):
            self._status_timer.stop()

        if spinning:
            self.spinner.start()
        else:
            self.spinner.stop()

        self.status_label.setText(text)
        if hasattr(self, "status_sep"):
            if text or spinning:
                self.status_sep.setStyleSheet("QLabel#status_sep { color: #333644; font-size: 11px; }")
            else:
                self.status_sep.setStyleSheet("QLabel#status_sep { color: transparent; font-size: 11px; }")

        self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.status_label.setStyleSheet("QLabel#status_label { color: #94a3b8; font-size: 11px; }")
        self.status_label.setToolTip("")

        if timeout_ms > 0:
            if hasattr(self, "_status_timer"):
                self._status_timer.start(timeout_ms)
        elif not spinning and text and not getattr(self, "_is_importing", False) and not getattr(self, "_is_multi_selecting", False):
            # Default lingering duration: 4.0 seconds without flickering or premature wipeout
            if hasattr(self, "_status_timer"):
                self._status_timer.start(4000)

    def _on_status_timeout(self):
        if getattr(self, "_is_importing", False) or getattr(self, "_is_multi_selecting", False):
            return
        self.status_label.setText("")
        self.spinner.stop()
        if hasattr(self, "status_sep"):
            self.status_sep.setStyleSheet("QLabel#status_sep { color: transparent; font-size: 11px; }")

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
            return

        self.export_status_container.setVisible(True)
        if hasattr(self, "export_sep"):
            self.export_sep.setVisible(True)

        self.lbl_export_status.setText(text)

        if state == "busy":
            self.export_spinner.start()
            self.lbl_export_status.setStyleSheet("QLabel#lbl_export_status { color: #f59e0b; font-size: 11px; text-decoration: underline; background: transparent; border: none; }")
            self.export_status_container.setToolTip("正在后台扫描，点击查看扫描队列 (Ctrl+J)")
        elif state == "success":
            self.export_spinner.stop()
            self.lbl_export_status.setStyleSheet("QLabel#lbl_export_status { color: #f59e0b; font-size: 11px; text-decoration: underline; background: transparent; border: none; }")
            self.export_status_container.setToolTip("扫描完成，点击查看扫描队列 (Ctrl+J)")
            # Permanent display on success as requested by user
            timeout_ms = 0
        elif state == "error":
            self.export_spinner.stop()
            self.lbl_export_status.setStyleSheet("QLabel#lbl_export_status { color: #ef4444; font-size: 11px; text-decoration: underline; background: transparent; border: none; }")
            self.export_status_container.setToolTip("扫描发生错误，点击查看扫描队列 (Ctrl+J)")
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
            self.set_app_status("底片库为空，无需保存会话", timeout_ms=2500)
            return False

        if not self._current_session_path or getattr(self, "_is_temp_session", False):
            return self.action_save_session_as()

        return self._do_save_session(self._current_session_path)

    def action_save_session_as(self):
        """Item 9 & 11: Prompts user to select a .sdss file path and saves session. Returns True if saved."""
        if not self.photos and not self._current_session_path:
            self.set_app_status("底片库为空，无需保存会话", timeout_ms=2500)
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
                photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
                if photo:
                    if photo.get("is_edited", False) or getattr(self, "_photo_is_dirty", False) or self._is_photo_edited(photo):
                        photo["params"] = dict(self.current_params)
                        photo["film_profile"] = self.current_film_stock
                        photo["paper_profile"] = self.current_paper_stock
                        photo["print_mode"] = getattr(self, "current_print_mode", "optical")
                    else:
                        photo["params"] = None
                        photo["film_profile"] = "none"
                        photo["paper_profile"] = "none"
                        photo["print_mode"] = "scan"

            session_photos = []
            for p in self.photos:
                session_photos.append({
                    "id": p["id"],
                    "path": normalize_path(p["path"]),
                    "filename": os.path.basename(p["path"]),
                    "film_stock": p.get("film_profile", "none"),
                    "paper_stock": p.get("paper_profile", "none"),
                    "print_mode": p.get("print_mode", "optical"),
                    "base_ev": p.get("base_ev", 0.0),
                    "params": p.get("params", {}),
                    "is_dirty": p.get("is_dirty", False),
                    "is_edited": p.get("is_edited", False)
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
        self.set_app_status("正在保存会话...", spinning=True)
        # Flush current dirty photo parameters to .sdc sidecar
        if self._photo_is_dirty and self._current_photo_path:
            sdc_manager.save_sdc(
                self._current_photo_path,
                self.current_film_stock,
                self.current_paper_stock,
                self.current_params
            )
            self._photo_is_dirty = False
            if self.active_photo_id:
                p = next((x for x in self.photos if x["id"] == self.active_photo_id), None)
                if p:
                    p["is_dirty"] = False
                    p["is_edited"] = True
            self.filmstrip.set_photo_dirty(self.active_photo_id, True)

        from core.path_utils import normalize_path
        session_dir = os.path.dirname(os.path.abspath(file_path))
        session_photos = []
        for p in self.photos:
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
                "base_ev": float(p.get("base_ev", 0.0))
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
            self.set_app_status(f"会话已保存: {fn}", timeout_ms=3000)
            self._session_is_dirty = False
            self._is_temp_session = False
            self._update_window_title()
            config_manager.add_recent_session(file_path)
            config_manager.save_last_session_path(file_path)
            self._update_recent_menu()
            return True
        else:
            self.set_app_status("会话保存失败", timeout_ms=3000)
            return False

    def _prompt_save_if_session_dirty(self) -> bool:
        """Prompt user to save if current session has unsaved modifications.
        Returns True if safe to proceed (clean, saved, or discarded), False if user cancelled.
        """
        is_suppressed = getattr(self, '_suppress_close_confirm', False) or bool(os.environ.get("FAST_TEST")) or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        if getattr(self, '_session_is_dirty', False) and (self.photos or self._current_session_path) and not is_suppressed:
            sess_name = os.path.basename(self._current_session_path) if self._current_session_path else "未命名会话"
            dialog = UnsavedSessionDialog(sess_name, self)
            res = dialog.exec()
            if res == UnsavedSessionDialog.CANCEL:
                return False
            elif res == UnsavedSessionDialog.SAVE:
                self.action_save_session()
                if getattr(self, '_session_is_dirty', False):
                    return False
            # elif res == UnsavedSessionDialog.DISCARD: proceed
        return True

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
                    self.current_params
                )
                self._photo_is_dirty = False
            for p in self.photos:
                if p.get("is_dirty") and p.get("path"):
                    sdc_manager.save_sdc(
                        p["path"],
                        p.get("film_profile", self.current_film_stock),
                        p.get("paper_profile", self.current_paper_stock),
                        p.get("params", self.current_params)
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
            self.set_app_status("会话文件不存在", timeout_ms=3000)
            return

        self.set_app_status("正在加载会话...", spinning=True)
        session_data = session_cache_manager.load_session_file(file_path)
        if not session_data or "photos" not in session_data:
            self.set_app_status("会话文件损坏或无效", timeout_ms=3000)
            return

        from core.path_utils import normalize_path
        session_dir = os.path.dirname(os.path.abspath(file_path))
        photo_entries = session_data["photos"]

        resolved_files = []
        missing_entries = []
        path_updated = False

        for p in photo_entries:
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
            elif dlg.action_result == RelinkMediaDialog.STATUS_OFFLINE:
                # User chose to keep offline and open with only found photos
                pass
            else:
                # User cancelled opening session
                self.set_app_status("已取消载入会话", timeout_ms=2500)
                return

        if not resolved_files and photo_entries:
            self.set_app_status("底片均丢失且未关联，无法载入会话", timeout_ms=3000)
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

        if resolved_files:
            self._restore_saved_session(resolved_files, active_file=active_path, session_name=fn, path_updated=path_updated)
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
                self.set_app_status(f"会话已恢复 (空工程): {fn}", timeout_ms=2500)
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

        photos_to_export = [p for p in self.photos if p["id"] in pids]
        if not photos_to_export:
            return

        fmt = last_cfg.get("format", "jpeg")
        ext = ".tif" if "tif" in str(fmt).lower() else (".png" if "png" in str(fmt).lower() else ".jpg")
        out_dir = config_manager.get_last_export_dir()
        if not out_dir or not os.path.isdir(out_dir):
            if self._current_photo_path:
                out_dir = os.path.dirname(self._current_photo_path)
            else:
                out_dir = os.path.expanduser("~")

        enqueued_count = 0
        new_tasks = []
        import uuid
        for p in photos_to_export:
            src = p["path"]
            base_name = os.path.splitext(os.path.basename(src))[0]
            out_file = os.path.join(out_dir, f"{base_name}{ext}")

            task_params = dict(self.current_params)
            sdc = sdc_manager.load_sdc(src)
            film = sdc.get("film_profile", self.current_film_stock) if sdc else self.current_film_stock
            paper = sdc.get("paper_profile", self.current_paper_stock) if sdc else self.current_paper_stock
            if sdc and "params" in sdc:
                task_params.update(sdc["params"])

            task_params.update(last_cfg)
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
            task_params["base_temp"] = getattr(self, "_active_base_temp", 5500.0)
            task_params["base_tint"] = getattr(self, "_active_base_tint", 1.0)
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

    def push_undo_state(self):
        state = {
            "type": "single_photo_state",
            "photo_id": self.active_photo_id,
            "photo_path": self._current_photo_path,
            "film": getattr(self, "current_film_stock", "none"),
            "paper": getattr(self, "current_paper_stock", "none"),
            "print_mode": getattr(self, "current_print_mode", "optical"),
            "params": dict(self.current_params)
        }
        # Avoid pushing duplicate states that match the top of the undo stack
        if self._undo_stack:
            top = self._undo_stack[-1]
            top_pid = top.get("photo_id") if top.get("type") == "single_photo_state" else None
            if top.get("type") == "batch_photos_state" or (top_pid and top_pid != self.active_photo_id):
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

    def _push_raw_state(self, state: dict):
        """Pushes a custom state (e.g. batch operation snapshot) to the undo stack."""
        self._undo_stack.append(state)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay.undo():
                self.set_app_status("已撤销裁切操作", timeout_ms=1500)
            else:
                self.set_app_status("已到最初裁切状态", timeout_ms=1500)
            return

        if len(self._undo_stack) <= 1:
            self.set_app_status("已到最初历史状态", timeout_ms=1500)
            return

        curr_state = self._undo_stack.pop()
        self._redo_stack.append(curr_state)
        prev_state = self._undo_stack[-1]
        self._apply_history_state(prev_state)

        act = curr_state.get("action") or prev_state.get("action")
        if act == "clear_sdc":
            self.set_app_status("已撤销删除暗房配置", timeout_ms=2000)
        elif act == "paste_params":
            self.set_app_status("已撤销粘贴暗房配方", timeout_ms=2000)
        elif act == "clear_crop":
            self.set_app_status("已撤销清除裁切", timeout_ms=2000)
        else:
            self.set_app_status("已撤销", timeout_ms=1500)

    def redo(self):
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay.redo():
                self.set_app_status("已重做裁切操作", timeout_ms=1500)
            else:
                self.set_app_status("已到最新裁切状态", timeout_ms=1500)
            return

        if not self._redo_stack:
            self.set_app_status("已到最新历史状态", timeout_ms=1500)
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
        else:
            self.set_app_status("已重做", timeout_ms=1500)

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

        # Ignore internal non-color keys
        ignored_keys = {"print_mode", "custom_wb", "film_format_mm", "base_temp", "base_tint", "crop"}
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

    def action_copy_params(self, photo_id=None):
        """Ctrl+C handler: Copy all physical darkroom parameters from the currently viewed or specified photo."""
        target_id = photo_id if (photo_id and isinstance(photo_id, str)) else self.active_photo_id
        if not target_id:
            return

        target_p = next((p for p in self.photos if p.get("id") == target_id), None)
        if not target_p or not self._is_photo_edited(target_p):
            self.set_app_status("当前底片为原生底片（未做任何暗房修改），无暗房配方可复制", timeout_ms=2500)
            return

        if target_id == self.active_photo_id:
            film = self.current_film_stock
            paper = self.current_paper_stock
            pmode = getattr(self, "current_print_mode", "optical")
            params = dict(self.current_params)
        else:
            film = target_p.get("film_profile", "none")
            paper = target_p.get("paper_profile", "none")
            pmode = target_p.get("print_mode", "optical")
            if target_p.get("params"):
                params = dict(target_p["params"])
            elif target_p.get("path"):
                sdc = sdc_manager.load_sdc(target_p["path"])
                if sdc:
                    film = sdc.get("film_profile", film)
                    paper = sdc.get("paper_profile", paper)
                    pmode = sdc.get("print_mode", pmode)
                    params = dict(sdc.get("params", {}))
                else:
                    params = {}
            else:
                params = {}

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
        self.set_app_status(f"已复制暗房配方 [{film_desc}]", timeout_ms=2000)

    def action_paste_params(self):
        """Ctrl+V handler: Paste copied physical darkroom parameters to all selected photos."""
        copied = getattr(self, '_copied_darkroom_profile', None)
        if not copied or not copied.get("is_edited"):
            self.set_app_status("剪贴板中无有效暗房配方可粘贴", timeout_ms=2500)
            return

        copied_film = copied["film"]
        copied_paper = copied["paper"]
        copied_print_mode = copied.get("print_mode", "optical")
        copied_params = dict(copied["params"])

        selected_ids = self.filmstrip.get_selected_photo_ids() if hasattr(self, 'filmstrip') else []
        if not selected_ids and self.active_photo_id:
            selected_ids = [self.active_photo_id]

        if not selected_ids:
            return

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
                p_params = dict(self.current_params)
                p_dirty = bool(getattr(self, "_photo_is_dirty", False) or p.get("is_dirty", False))
                p_edited = bool(self._is_photo_edited(p))
            else:
                p_film = p.get("film_profile", "none")
                p_paper = p.get("paper_profile", "none")
                p_pmode = p.get("print_mode", "optical")
                p_params = dict(p.get("params") or {})
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
            "action": "paste_params",
            "active_photo_id": self.active_photo_id,
            "photos": pre_photos,
        }
        self._push_raw_state(pre_state)

        affected_count = 0
        for pid in selected_ids:
            photo = next((p for p in self.photos if p["id"] == pid), None)
            if not photo:
                continue

            target_params = dict(copied_params)
            # Adapt white balance relative to target photo baseline EXIF if not custom
            if not target_params.get("custom_wb", False):
                exif = photo.get("meta") or photo.get("exif") or {}
                target_params["color_temp"] = float(exif.get("color_temp", 5500.0))
                target_params["tint"] = float(exif.get("tint", 1.0))

            photo["film_profile"] = copied_film
            photo["paper_profile"] = copied_paper
            photo["print_mode"] = copied_print_mode
            photo["params"] = target_params
            photo["is_dirty"] = True
            photo["is_edited"] = True

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

            self.filmstrip.set_photo_dirty(pid, True)
            affected_count += 1

        # If current active photo is in selected targets, apply immediately to viewport
        if self.active_photo_id in selected_ids:
            self.current_film_stock = copied_film
            self.current_paper_stock = copied_paper
            self.current_print_mode = copied_print_mode
            self.current_params = dict(copied_params)
            self.current_params["print_mode"] = copied_print_mode

            self._update_all_sliders_from_params()
            self.on_switch_print_mode(self.current_print_mode, push_undo=False)
            self.on_switch_film(self.current_film_stock, push_undo=False)
            if self.current_print_mode == "optical":
                self.on_switch_paper(self.current_paper_stock, push_undo=False)
            else:
                for pid, card in self.paper_cards.items():
                    card.set_active(pid == "none")

            self._sync_full_params_to_pipeline(fit_crop_if_changed=True)
            self._mark_current_photo_dirty()

        # 2. Snapshot post-paste state for all affected photos
        post_photos = {}
        for pid in selected_ids:
            photo = next((p for p in self.photos if p["id"] == pid), None)
            if not photo:
                continue
            post_photos[pid] = {
                "film": photo.get("film_profile", copied_film),
                "paper": photo.get("paper_profile", copied_paper),
                "print_mode": photo.get("print_mode", copied_print_mode),
                "params": dict(photo.get("params") or {}),
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

        self.set_app_status(f"已应用暗房配方至 {affected_count} 张底片", timeout_ms=2500)

    def _apply_history_state(self, state):
        if not state:
            return

        if state.get("type") == "batch_photos_state":
            self._apply_batch_history_state(state)
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

            self._update_all_sliders_from_params()
            self.on_switch_print_mode(self.current_print_mode, push_undo=False)
            self.on_switch_film(self.current_film_stock, push_undo=False)
            if self.current_print_mode == "optical":
                self.on_switch_paper(self.current_paper_stock, push_undo=False)
            else:
                for pid_card, card in self.paper_cards.items():
                    card.set_active(pid_card == "none")

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
                sampled = self.canvas.render_offscreen(target_w=256, target_h=170, bit_depth=8)
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
                eff_exp = float(self.current_params.get("exposure_ev", 0.0)) + float(self.current_params.get("base_ev", 0.0))
                src_lin = self._current_photo_linear * (2.0 ** eff_exp) if abs(eff_exp) > 0.001 else self._current_photo_linear
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
            "grain_cloud_blur": 1.0,
            "grain_colour": 1.0,
            "lens": self._get_default_lens_params(),
            "split_x": 0.5,
            "view_mode": 0
        }

    def _on_slider_live(self, param_key, val):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        if getattr(self, "current_film_stock", "none") == "none":
            return
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
        if param_key in ("dir_amount", "dir_interlayer", "dir_samelayer", "morph_gamma", "developer_exhaustion", "pre_flash", "print_exposure", "enlarger_cyan", "enlarger_magenta", "enlarger_yellow"):
            self._pending_lut_refresh = True
        self.canvas.update_params(**{param_key: val})
        self._schedule_histogram_update()
        self._mark_current_photo_dirty()
        self._update_reset_buttons_state()

    def _on_combo_changed(self, param_key, val):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        if getattr(self, "current_film_stock", "none") == "none":
            return
        self.current_params[param_key] = val
        self.canvas.update_params(**{param_key: val})
        if param_key == "diffusion_family":
            if val != "none" and self.current_params.get("diffusion_strength", 0.0) == 0.0:
                self.current_params["diffusion_strength"] = 0.25
                self.canvas.update_params(diffusion_strength=0.25)
                if hasattr(self, 'combo_diff_strength'):
                    self.combo_diff_strength.set_value(0.25)
            elif val == "none":
                self.current_params["diffusion_strength"] = 0.0
                self.canvas.update_params(diffusion_strength=0.0)
                if hasattr(self, 'combo_diff_strength'):
                    self.combo_diff_strength.set_value(0.0)
        elif param_key == "diffusion_strength":
            if val > 0.0 and self.current_params.get("diffusion_family", "none") == "none":
                self.current_params["diffusion_family"] = "black_pro_mist"
                self.canvas.update_params(diffusion_family="black_pro_mist")
                if hasattr(self, 'combo_diffusion'):
                    self.combo_diffusion.set_value("black_pro_mist")
            elif val == 0.0 and self.current_params.get("diffusion_family") != "none":
                self.current_params["diffusion_family"] = "none"
                self.canvas.update_params(diffusion_family="none")
                if hasattr(self, 'combo_diffusion'):
                    self.combo_diffusion.set_value("none")
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
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    self.current_paper_stock,
                    self.current_params,
                    thumbnail_rgb=cur_thumb,
                    print_mode=getattr(self, "current_print_mode", "optical")
                )
                curr_p["film_profile"] = self.current_film_stock
                curr_p["paper_profile"] = self.current_paper_stock
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
        t0 = time.time()
        lut_3d = self.engine.get_3d_lut(
            self.current_film_stock,
            self.current_paper_stock,
            lut_size=33,
            params_dict=self.current_params
        )
        self.canvas.set_lut(lut_3d)
        dur = (time.time() - t0) * 1000.0
        self.set_app_status(f"LUT 烘焙完成 ({dur:.0f}ms)", timeout_ms=2000)

    def _on_slider_committed(self):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        if hasattr(self, '_live_thumb_timer') and self._live_thumb_timer.isActive():
            self._live_thumb_timer.stop()
        if hasattr(self, '_cards_debounce_timer') and self._cards_debounce_timer.isActive():
            self._cards_debounce_timer.stop()
        if getattr(self, "_pending_lut_refresh", False):
            self._pending_lut_refresh = False
            if getattr(self, "current_film_stock", "none") != "none":
                self._refresh_lut()
        self.push_undo_state()
        self._mark_current_photo_dirty()
        self._update_active_thumbnails()
        self._save_current_photo_sdc()
        self._refresh_card_previews()

    def _action_apply_auto_meter(self):
        """Reset exposure compensation to the calibrated 18% middle-gray baseline (0.00 EV)."""
        if not self._current_photo_path:
            return
        photo = next((p for p in self.photos if p.get("path") == self._current_photo_path), None)
        if not photo:
            return
        ev_val = 0.0
        self.current_params["exposure_ev"] = ev_val
        if hasattr(self, 'slider_ev'):
            self.slider_ev.set_value(ev_val)
        self.canvas.update_params(exposure_ev=ev_val)
        self._on_slider_committed()
        base_ev = float(photo.get("base_ev", 0.0))
        self.set_app_status(f"基准测光: 18% 中灰 (基准 {base_ev:+.2f} EV)", timeout_ms=2500)

    def _get_track_defaults(self):
        def_temp = getattr(self.slider_temp, "default_val", 5500.0) if hasattr(self, 'slider_temp') else 5500.0
        def_tint = getattr(self.slider_tint, "default_val", 1.0) if hasattr(self, 'slider_tint') else 1.0
        return {
            "exposure": {
                "exposure_ev": 0.0, "highlights": 0.0, "shadows": 0.0,
                "color_temp": def_temp, "tint": def_tint,
                "film_format_mm": 35.0, "custom_wb": False
            },
            "optics": {
                "diffusion_family": "none", "diffusion_strength": 0.0, "diffusion_warmth": 0.0
            },
            "chemistry": {
                "dir_amount": 1.0, "dir_interlayer": 1.0, "dir_samelayer": 1.0
            },
            "enlarger": {
                "enlarger_illuminant": "TH-KG3",
                "enlarger_cyan": 0.0, "enlarger_magenta": 0.0, "enlarger_yellow": 0.0,
                "print_exposure": 1.0, "pre_flash": 0.0,
                "morph_gamma": 1.0, "developer_exhaustion": 0.0
            },
            "halation": {
                "halation_amount": 1.0, "halation_spatial_scale": 1.0,
                "scatter_amount": 1.0, "boost_ev": 0.0,
                "halation": 1.0, "halation_boost": 0.0
            },
            "grain": {
                "grain": 0.4, "grain_size": 1.0, "grain_cloud_blur": 1.0, "grain_colour": 1.0
            },
            "texture": {
                "halation_amount": 1.0, "halation_spatial_scale": 1.0,
                "scatter_amount": 1.0, "boost_ev": 0.0,
                "halation": 1.0, "halation_boost": 0.0,
                "grain": 0.4, "grain_size": 1.0, "grain_cloud_blur": 1.0, "grain_colour": 1.0
            }
        }

    def _update_reset_buttons_state(self):
        """Item 5: Gray out and disable reset button when the section is at baseline defaults."""
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
            sec.set_reset_enabled(not is_default)

    def reset_track(self, track_name):
        if getattr(self, "current_film_stock", "none") == "none":
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

        if "enlarger_illuminant" in defaults[track_name]:
            self._refresh_lut()
        self._update_all_sliders_from_params()
        self._schedule_histogram_update()

        # Item 5: Resetting an unedited photo never marks it dirty!
        if getattr(self, "_photo_is_dirty", False):
            self.push_undo_state()
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
                    photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
                    if photo:
                        photo["is_dirty"] = False
                    self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            else:
                self._mark_current_photo_dirty()
                self._save_current_photo_sdc()
        self._update_reset_buttons_state()
        self._update_active_thumbnails()
        self._refresh_card_previews()
        self.push_undo_state()
        self.setFocus()

    def reset_all_params(self):
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                self.canvas.crop_overlay.reset_crop()
                self.set_app_status("已重置裁切", timeout_ms=1500)
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
        if not changed:
            self._update_reset_buttons_state()
            return
        self._refresh_lut()
        self._update_all_sliders_from_params()
        self._schedule_histogram_update()
        if getattr(self, "_photo_is_dirty", False):
            if self.current_film_stock == "none" and self.current_paper_stock == "none":
                self._photo_is_dirty = False
                if self.active_photo_id:
                    photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
                    if photo:
                        photo["is_dirty"] = False
                    self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            else:
                self._mark_current_photo_dirty()
                self._save_current_photo_sdc()
        self._update_reset_buttons_state()
        self._update_active_thumbnails()
        self._refresh_card_previews()
        self.push_undo_state()

    def on_reset_to_open(self):
        """Dispatch '重置为打开状态' (Ctrl+Alt+R) according to workspace mode."""
        if getattr(self.canvas, "is_crop_editing", False):
            if hasattr(self.canvas, "crop_overlay") and self.canvas.crop_overlay:
                self.canvas.crop_overlay.reset_to_initial_state()
                self.set_app_status("已恢复至打开时裁切状态", timeout_ms=1500)
        else:
            self.reset_to_open_state()

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

        # Reset stocks to none (virgin bypass)
        self.current_film_stock = "none"
        self.current_paper_stock = "none"
        self.current_print_mode = "scan"

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
        self.push_undo_state()

        self._photo_is_dirty = False
        if self.active_photo_id:
            photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
            if photo:
                photo["is_dirty"] = False
                photo["is_edited"] = False
                photo["film_profile"] = "none"
                photo["paper_profile"] = "none"
                photo["params"] = copy.deepcopy(self.current_params)
                photo["crop"] = dict(default_crop)
                p_path = photo.get("path", "")
                if p_path and sdc_manager.has_sdc(p_path):
                    sdc_manager.delete_sdc(p_path)
            self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            self.filmstrip.set_photo_crop(self.active_photo_id, False)

        self.set_app_status("已恢复至原始底片状态", timeout_ms=1500)

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
            if hasattr(self, 'combo_diff_strength'): self.combo_diff_strength.set_value(p.get("diffusion_strength", 0.0))
            if hasattr(self, 'slider_diff_strength'): self.slider_diff_strength.set_value(p.get("diffusion_strength", 0.0))
            if hasattr(self, 'slider_diff_warmth'): self.slider_diff_warmth.set_value(p.get("diffusion_warmth", 0.0))

            if hasattr(self, 'slider_dir_amount'): self.slider_dir_amount.set_value(p.get("dir_amount", 1.0))
            if hasattr(self, 'slider_dir_interlayer'): self.slider_dir_interlayer.set_value(p.get("dir_interlayer", 1.0))
            if hasattr(self, 'slider_dir_samelayer'): self.slider_dir_samelayer.set_value(p.get("dir_samelayer", 1.0))

            if hasattr(self, 'combo_illuminant'): self.combo_illuminant.set_value(p.get("enlarger_illuminant", "TH-KG3"))
            if hasattr(self, 'slider_magenta'): self.slider_magenta.set_value(p.get("enlarger_magenta", 0.0))
            if hasattr(self, 'slider_yellow'): self.slider_yellow.set_value(p.get("enlarger_yellow", 0.0))
            if hasattr(self, 'slider_cyan'): self.slider_cyan.set_value(p.get("enlarger_cyan", 0.0))
            if hasattr(self, 'slider_print_exp'): self.slider_print_exp.set_value(p.get("print_exposure", 1.0))
            if hasattr(self, 'slider_preflash'): self.slider_preflash.set_value(p.get("pre_flash", 0.0))
            if hasattr(self, 'slider_morph_gamma'): self.slider_morph_gamma.set_value(p.get("morph_gamma", 1.0))
            if hasattr(self, 'slider_exhaustion'): self.slider_exhaustion.set_value(p.get("developer_exhaustion", 0.0))

            if hasattr(self, 'slider_halation_amount'): self.slider_halation_amount.set_value(p.get("halation_amount", p.get("halation", 1.0)))
            if hasattr(self, 'slider_halation_spatial_scale'): self.slider_halation_spatial_scale.set_value(p.get("halation_spatial_scale", 1.0))
            if hasattr(self, 'slider_scatter_amount'): self.slider_scatter_amount.set_value(p.get("scatter_amount", 1.0))
            if hasattr(self, 'slider_boost_ev'): self.slider_boost_ev.set_value(p.get("boost_ev", p.get("halation_boost", 0.0)))
            if hasattr(self, 'slider_halation'): self.slider_halation.set_value(p.get("halation_amount", p.get("halation", 1.0)))
            if hasattr(self, 'slider_halation_boost'): self.slider_halation_boost.set_value(p.get("boost_ev", p.get("halation_boost", 0.0)))
            if hasattr(self, 'slider_grain'): self.slider_grain.set_value(p.get("grain", 0.4))
            if hasattr(self, 'slider_grain_size'): self.slider_grain_size.set_value(p.get("grain_size", 1.0))
            if hasattr(self, 'slider_grain_cloud'): self.slider_grain_cloud.set_value(p.get("grain_cloud_blur", 1.0))
            if hasattr(self, 'slider_grain_colour'): self.slider_grain_colour.set_value(p.get("grain_colour", 1.0))

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
            self.canvas.set_lut(lut_3d)
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
        cur_crop = self.current_params.get("crop", {})
        self._pre_crop_backup = copy.deepcopy(cur_crop) if isinstance(cur_crop, dict) else {}
        self.canvas.start_crop_mode(cur_crop)
        if hasattr(self, 'crop_sidebar'):
            self.crop_sidebar.sync_from_overlay(self.canvas.crop_overlay.get_crop_dict())

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
        self.canvas.fit_to_view(animate=False)
        self._save_current_photo_sdc()
        self.push_undo_state()
        if hasattr(self, "filmstrip") and self.active_photo_id:
            self.filmstrip.set_photo_crop(self.active_photo_id, crop_dict)
        target_photo = next((p for p in self.photos if p.get("id") == self.active_photo_id), None)
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
        self.canvas.fit_to_view(animate=False)
        if hasattr(self, "filmstrip") and self.active_photo_id:
            self.filmstrip.set_photo_crop(self.active_photo_id, reverted_crop)
        self._update_active_thumbnails()
        if hasattr(self, '_cards_debounce_timer'):
            self._cards_debounce_timer.start(100)
        else:
            self._refresh_card_previews()

    def on_switch_film(self, stock_id, push_undo=True):
        if hasattr(self, "_hover_timer"):
            self._hover_timer.stop()
        self._hover_pending_stock = None
        self._saved_film_stock = None
        self._saved_paper_stock = None
        self._saved_print_mode = None

        if push_undo and stock_id != self.current_film_stock:
            self._mark_current_photo_dirty()
        self.current_film_stock = stock_id

        # Selecting "none" film (Bypass / original raw) or positive reversal slide film locks paper to "none" (scan / bypass)
        is_pos = self.engine.is_positive_film(stock_id) if hasattr(self.engine, 'is_positive_film') else False
        if (stock_id == "none" or is_pos) and push_undo:
            self.current_paper_stock = "none"
            self.current_print_mode = "scan"
            self.current_params["print_mode"] = "scan"
            self.current_params["enlarger_cyan"] = 0.0
            self.current_params["enlarger_magenta"] = 0.0
            self.current_params["enlarger_yellow"] = 0.0
            self.current_params["print_exposure"] = 1.0
            self.current_params["pre_flash"] = 0.0
            self.current_params["developer_exhaustion"] = 0.0
            self.current_params["morph_gamma"] = 1.0
            for pid, card in self.paper_cards.items():
                card.set_active(pid == "none")
            self._update_all_sliders_from_params()

        for sid, card in self.film_cards.items():
            card.set_active(sid == stock_id)

        # Save to current photo
        if self._current_photo_path:
            photo = next((p for p in self.photos if p.get("path") == self._current_photo_path), None)
            if photo:
                photo["film_profile"] = stock_id
                if stock_id == "none" or is_pos:
                    photo["paper_profile"] = "none"
                    photo["print_mode"] = "scan"
                if stock_id == "none":
                    photo["graded_thumbnail_rgb"] = None
                    self.thumb_controller.revert_photo_to_raw(photo, self.filmstrip)

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = self.current_print_mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self.canvas.set_lut(lut_3d)
        self.canvas.update_params(paper_profile=stock_id, print_mode=self.current_print_mode)
        hal_preset = self.engine.get_film_halation_preset(stock_id) if hasattr(self.engine, 'get_film_halation_preset') else None
        grain_preset = self.engine.get_film_grain_preset(stock_id) if hasattr(self.engine, 'get_film_grain_preset') else None
        self.canvas.update_params(film_stock=stock_id, _halation_preset=hal_preset, _grain_preset=grain_preset)
        self._schedule_histogram_update()
        if self._current_photo_path:
            self._update_active_thumbnails()
            self._refresh_card_previews(paper_only=False)
        self._update_stock_mode_ui()
        if push_undo:
            self.push_undo_state()
            self._save_current_photo_sdc()
            film_desc = "无" if stock_id == "none" else f"胶卷 ({stock_id})"
            self.set_app_status(f"已选用: {film_desc}", timeout_ms=2500)

    def on_switch_print_mode(self, mode, push_undo=True):
        if mode not in ("optical", "scan"):
            mode = "optical"
        if push_undo and mode != getattr(self, "current_print_mode", "optical"):
            self._mark_current_photo_dirty()

        self.current_print_mode = mode
        self.current_params["print_mode"] = mode

        # Save to current photo
        if push_undo and self._current_photo_path:
            photo = next((p for p in self.photos if p.get("path") == self._current_photo_path), None)
            if photo:
                photo["print_mode"] = mode

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self.canvas.set_lut(lut_3d)
        self._schedule_histogram_update()
        if push_undo and self._current_photo_path:
            self._update_active_thumbnails()
            self._refresh_card_previews()
        self._update_stock_mode_ui()
        if push_undo:
            self.push_undo_state()
            self._save_current_photo_sdc()
            mode_name = "相纸印相" if mode == "optical" else "数码扫描"
            self.set_app_status(f"模式: {mode_name}", timeout_ms=2500)

    def on_switch_paper(self, stock_id, push_undo=True):
        if hasattr(self, "_hover_timer"):
            self._hover_timer.stop()
        self._hover_pending_stock = None
        self._saved_film_stock = None
        self._saved_paper_stock = None
        self._saved_print_mode = None

        is_pos = self.engine.is_positive_film(self.current_film_stock) if hasattr(self.engine, 'is_positive_film') else False
        if is_pos and stock_id != "none":
            self.set_app_status("反转底片无需相纸放大", timeout_ms=2500)
            return

        if push_undo and stock_id != self.current_paper_stock:
            self._mark_current_photo_dirty()
        self.current_paper_stock = stock_id

        # Item 9: Selecting "none" (数码扫描 Film Scan) engages scan mode; selecting physical paper engages Optical Print mode
        if push_undo:
            if stock_id == "none":
                self.current_print_mode = "scan"
                self.current_params["print_mode"] = "scan"
            else:
                self.current_print_mode = "optical"
                self.current_params["print_mode"] = "optical"

        for sid, card in self.paper_cards.items():
            card.set_active(sid == stock_id)

        # Save to current photo
        if push_undo and self._current_photo_path:
            photo = next((p for p in self.photos if p.get("path") == self._current_photo_path), None)
            if photo:
                photo["paper_profile"] = stock_id
                photo["print_mode"] = self.current_print_mode

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = self.current_print_mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self.canvas.set_lut(lut_3d)
        self.canvas.update_params(paper_profile=stock_id, print_mode=self.current_print_mode)
        self._schedule_histogram_update()
        if push_undo and self._current_photo_path:
            self._refresh_card_previews(paper_only=False)
            self._update_active_thumbnails()
        self._update_stock_mode_ui()
        if push_undo:
            self.push_undo_state()
            self._save_current_photo_sdc()
            mode_desc = "数码扫描" if stock_id == "none" else f"相纸印相 ({stock_id})"
            self.set_app_status(f"已选用: {mode_desc}", timeout_ms=2500)

    def set_view_mode(self, mode):
        self.canvas.set_view_mode(mode)
        if hasattr(self, 'btn_split'):
            self.btn_split.setChecked(mode != 0)
            if mode == 0:
                self.btn_split.setToolTip("对比视图 (快捷键 \\): 点击切换分屏对比，长按查看原片")
            elif mode == 1:
                self.btn_split.setToolTip("对比视图 (快捷键 \\): 拖动分屏对比中，再次点击切换为左右全局双屏对比，长按查看原片")
            elif mode == 2:
                self.btn_split.setToolTip("对比视图 (快捷键 \\): 左右全局双屏对比中，再次点击恢复编辑模式，长按查看原片")
            elif mode == 3:
                self.btn_split.setToolTip("对比视图: 正在查看原始图像 (松开恢复)")

    def _on_compare_long_press_start(self):
        if not self._current_photo_path:
            return
        self._saved_view_mode_before_long_press = self.canvas.params.get("view_mode", 0)
        self.set_view_mode(3)
        self.set_app_status("原片对比 (松开恢复)", timeout_ms=0)

    def _on_compare_long_press_end(self):
        if hasattr(self, "_saved_view_mode_before_long_press"):
            prev_mode = self._saved_view_mode_before_long_press
            delattr(self, "_saved_view_mode_before_long_press")
            self.set_view_mode(prev_mode)
            self.restore_default_status()

    def toggle_split_view(self):
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
        files, _ = QFileDialog.getOpenFileNames(self, "导入照片 / RAW 负片", last_d, filt)
        if files:
            config_manager.set_last_dialog_dir("photo", files[0])
            self._import_files_list(files)

    def action_open_folder(self):
        last_d = config_manager.get_last_dialog_dir("photo")
        folder = QFileDialog.getExistingDirectory(self, "选择导入照片文件夹", last_d)
        if folder:
            config_manager.set_last_dialog_dir("photo", folder)
            valid_exts = ALL_SUPPORTED_IMAGE_EXTENSIONS
            files = []
            for root, _, fnames in os.walk(folder):
                for f in fnames:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        files.append(os.path.join(root, f))
            if files:
                self._import_files_list(files)
            else:
                self._show_info_dialog("提示", "所选文件夹中未找到受支持的图像文件。")

    def open_single_file(self, file_path):
        if os.path.exists(file_path):
            self._import_files_list([file_path])

    def _import_files_list(self, file_paths):
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

        # Duplicate detection dialog
        if dup_names:
            dlg = DuplicateFilesDialog(dup_names, self)
            dlg.exec()

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

        # 2. Add to recent files
        for p in new_paths:
            config_manager.add_recent_file(p)

        # 3. Ensure cancel button is visible
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(True)

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
            self.set_app_status(f"正在导入底片 ({cur}/{tot})...", spinning=True, timeout_ms=0)
            logger.info(f"Import queue: appended {len(new_paths)} files, pending queue size: {len(self._pending_import_queue)}, target total: {tot}")

            if not active_worker_running:
                self._start_next_pending_import_batch()
            return

        # 5. No worker running: initialize and start fresh import session
        self._is_importing = True
        self._pending_import_queue = []
        self._current_import_batch = list(new_paths)
        self._import_session_completed = 0
        self._import_session_target_total = len(new_paths)
        self._last_import_status_time = 0.0

        self.set_app_status(f"正在导入底片 (0/{self._import_session_target_total})...", spinning=True, timeout_ms=0)

        # Asynchronous streaming import worker
        start_idx = max(getattr(self, "_photo_id_counter", 0), len(self.photos))
        self._photo_id_counter = start_idx + len(new_paths)

        self._import_worker = BatchImportWorker(new_paths, self.engine, start_idx=start_idx, parent=self)
        self._import_worker.photoLoaded.connect(self._on_single_photo_imported)
        self._import_worker.importFinished.connect(self._on_all_photos_imported)
        self._import_worker.start()

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
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(True)

        logger.info(f"Import queue: starting next batch with {len(next_batch)} files (start_idx={start_idx})")
        self._import_worker = BatchImportWorker(next_batch, self.engine, start_idx=start_idx, parent=self)
        self._import_worker.photoLoaded.connect(self._on_single_photo_imported)
        self._import_worker.importFinished.connect(self._on_all_photos_imported)
        self._import_worker.start()
        return True

    def _on_single_photo_imported(self, photo_entry, current_idx, total_count):
        # Defensive check: ensure photo_entry ID is strictly unique
        existing_ids = {p["id"] for p in self.photos if p and "id" in p}
        pid = photo_entry.get("id")
        if not pid or pid in existing_ids:
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
                if (is_edited or photo_entry["film_profile"] != "none" or photo_entry["paper_profile"] != "none") and raw_th is not None:
                    graded = self.thumb_controller.grade_thumbnail_fast(
                        raw_th,
                        photo_entry["film_profile"],
                        photo_entry["paper_profile"],
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
                    if "crop" in photo_entry["params"]:
                        photo_entry["crop"] = dict(photo_entry["params"]["crop"])
                    photo_entry["is_dirty"] = True
                    photo_entry["is_edited"] = True
                    if sdc.get("thumbnail_rgb") is not None:
                        photo_entry["graded_thumbnail_rgb"] = sdc["thumbnail_rgb"]
                        photo_entry["thumbnail_rgb"] = sdc["thumbnail_rgb"]
                    elif raw_th is not None:
                        graded = self.thumb_controller.grade_thumbnail_fast(
                            raw_th,
                            photo_entry["film_profile"],
                            photo_entry["paper_profile"],
                            photo_entry["params"]
                        )
                        photo_entry["graded_thumbnail_rgb"] = graded
                        photo_entry["thumbnail_rgb"] = graded
            except Exception:
                pass

        self.photos.append(photo_entry)
        is_first = (len(self.photos) == 1) or (self.active_photo_id is None)

        # Incrementally stream into bottom filmstrip
        self.filmstrip.append_photo(photo_entry, is_active=is_first)
        if photo_entry.get("is_dirty") or photo_entry.get("is_edited"):
            self.filmstrip.set_photo_dirty(photo_entry["id"], True)

        # If this is the first photo, display it immediately on the canvas!
        if is_first:
            self.on_switch_photo(photo_entry["id"])

        # Synchronize with engine session photos for safe base_ev lookups
        if hasattr(self, "engine") and hasattr(self.engine, "session_photos"):
            matched_ep = next((ep for ep in self.engine.session_photos if ep.get("path") == photo_entry["path"]), None)
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
            if matched_ep:
                self.engine.session_photos[self.engine.session_photos.index(matched_ep)] = ep_dict
            else:
                self.engine.session_photos.append(ep_dict)

        # Strictly synchronized progress: increment session completed count
        self._import_session_completed = getattr(self, "_import_session_completed", 0) + 1
        cur = self._import_session_completed
        if not getattr(self, "_import_session_target_total", 0) and total_count:
            self._import_session_target_total = total_count
        tot = max(getattr(self, "_import_session_target_total", 0), cur)
        status_text = f"正在导入底片 ({cur}/{tot})..."

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
            self.set_app_status(f"正在导入底片 ({cur}/{tot})...", spinning=True, timeout_ms=0)

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
        self._is_importing = False
        self._current_import_batch = []
        self._pending_import_queue = []
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(False)
        self._pending_session_map = None
        self.canvas.set_has_library_photos(bool(self.photos))
        self._update_recent_menu()
        target_path = getattr(self, "_target_initial_active_path", None)
        if target_path:
            self._target_initial_active_path = None
            target_entry = next((p for p in self.photos if p.get("path") == target_path), None)
            if target_entry:
                self.on_switch_photo(target_entry["id"])

        completed_count = getattr(self, "_import_session_completed", len(self.photos))
        self.set_app_status(f"底片库导入完成 (共 {completed_count} 张)", spinning=False, timeout_ms=3000)

        # Real-time session state save & cache maintenance
        try:
            if self._current_session_path and os.path.exists(self._current_session_path):
                config_manager.save_last_session_path(self._current_session_path)
            elif getattr(self, "_is_temp_session", False) or not self._current_session_path:
                if self.photos:
                    self._save_auto_temp_session()
                    config_manager.save_last_session_path(config_manager.get_auto_saved_session_path())
            session_cache_manager.prune_cache_if_needed()
        except Exception:
            pass

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

        # 2. Terminate and disconnect batch import worker
        if getattr(self, "_import_worker", None):
            try:
                self._import_worker.photoLoaded.disconnect()
                self._import_worker.importFinished.disconnect()
            except Exception:
                pass
            if hasattr(self._import_worker, "isRunning") and self._import_worker.isRunning():
                if hasattr(self._import_worker, "stop"):
                    self._import_worker.stop()
                if hasattr(self._import_worker, "wait"):
                    self._import_worker.wait(150)
            self._import_worker = None

        # 3. Terminate slot decode worker
        if getattr(self, "_slot_decode_worker", None):
            try:
                self._slot_decode_worker.slotDecoded.disconnect()
                self._slot_decode_worker.progress.disconnect()
                self._slot_decode_worker.finished.disconnect()
            except Exception:
                pass
            if hasattr(self._slot_decode_worker, "isRunning") and self._slot_decode_worker.isRunning():
                if hasattr(self._slot_decode_worker, "stop"):
                    self._slot_decode_worker.stop()
                if hasattr(self._slot_decode_worker, "wait"):
                    self._slot_decode_worker.wait(150)
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
            if hasattr(self._session_restore_worker, "isRunning") and self._session_restore_worker.isRunning():
                if hasattr(self._session_restore_worker, "stop"):
                    self._session_restore_worker.stop()
                if hasattr(self._session_restore_worker, "wait"):
                    self._session_restore_worker.wait(150)
            self._session_restore_worker = None

        self._is_importing = False
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(False)
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
            if photo.get("film_profile", "none") != "none" or photo.get("paper_profile", "none") != "none" or abs(p_params.get("exposure_ev", 0.0)) > 0.001:
                graded = self.grade_thumbnail_fast(
                    raw_th,
                    photo.get("film_profile", "none"),
                    photo.get("paper_profile", "none"),
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

        # If current active photo was uncached and just completed, refresh canvas display
        if self.active_photo_id == photo["id"]:
            if photo.get("float_img") is not None:
                self.canvas.set_image(photo["float_img"], master_width=photo.get("width"), master_height=photo.get("height"), photo_id=photo["id"])
                self._update_photo_info_display()
                if hasattr(self, 'exif_widget'):
                    self.exif_widget.set_exif_data(photo.get("exif") or {})

    def _on_slot_decode_progress(self, curr, total):
        if getattr(self, "_is_importing", False):
            self.set_app_status(f"正在恢复底片库 ({curr}/{total} 张)...", spinning=True, timeout_ms=0)

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
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(False)
        sess_name = os.path.basename(self._current_session_path) if getattr(self, "_current_session_path", None) else ""
        if sess_name:
            self.set_app_status(f"会话底片库已就绪 ({len(self.photos)} 张): {sess_name}", spinning=False, timeout_ms=3000)
        else:
            self.set_app_status(f"底片库已就绪 ({len(self.photos)} 张)", spinning=False, timeout_ms=3000)
        try:
            session_cache_manager.prune_cache_if_needed()
        except Exception:
            pass

    def _sort_photos_by(self, sort_key):
        """Sort photos in filmstrip and session by criteria:
        'time_asc'/'exif_asc', 'time_desc'/'exif_desc', 'name_asc', 'name_desc', 'import'
        """
        if not self.photos or len(self.photos) <= 1:
            return

        def _get_photo_timestamp(photo):
            if not photo:
                return 0.0
            exif = photo.get("exif") or {}
            dt_str = (
                exif.get("capture_time")
                or exif.get("DateTimeOriginal")
                or exif.get("DateTimeDigitized")
                or exif.get("DateTime")
                or photo.get("capture_time")
                or ""
            )

            # If not in photo dict, check cached metadata
            if (not dt_str or dt_str == "-") and photo.get("path"):
                try:
                    from core import session_cache_manager
                    cached = session_cache_manager.get_photo_cache(photo["path"], load_preview=False)
                    if cached and cached.get("exif"):
                        c_exif = cached["exif"]
                        dt_str = (
                            c_exif.get("capture_time")
                            or c_exif.get("DateTimeOriginal")
                            or c_exif.get("DateTimeDigitized")
                            or c_exif.get("DateTime")
                            or ""
                        )
                except Exception:
                    pass

            # If still missing, attempt direct EXIF extraction
            if (not dt_str or dt_str == "-") and photo.get("path") and os.path.exists(photo["path"]):
                try:
                    meta = self.engine._extract_exif(photo["path"])
                    dt_str = meta.get("capture_time", "")
                except Exception:
                    pass

            # Convert date/time string to float epoch timestamp
            if dt_str and dt_str != "-":
                from datetime import datetime
                cleaned = str(dt_str).strip()
                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(cleaned, fmt).timestamp()
                    except Exception:
                        pass
                try:
                    return float(cleaned)
                except Exception:
                    pass

            # Fallback to filesystem modification time
            if photo.get("path") and os.path.exists(photo["path"]):
                try:
                    return float(os.path.getmtime(photo["path"]))
                except Exception:
                    pass
            return 0.0

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
        self.filmstrip.set_photos(self.photos, curr_active_id)
        for p in self.photos:
            if p and (p.get("is_dirty") or p.get("is_edited")):
                self.filmstrip.set_photo_dirty(p["id"], True)
        self.filmstrip.scroll_to_active()
        self._session_is_dirty = True
        self._update_window_title()
        self.set_app_status(msg, timeout_ms=3000)

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

            rendered_rgb = self.canvas.render_offscreen(target_w=tgt_w, target_h=tgt_h, params_override={"grain": 0.0}, bit_depth=8)
            if rendered_rgb is not None and rendered_rgb.size > 0:
                is_valid = bool(rendered_rgb.mean() > 1.0 or rendered_rgb.max() > 10)
                if not is_valid and not getattr(self.canvas, '_has_image', False):
                    return

                h, w = rendered_rgb.shape[:2]
                qimg = QImage(rendered_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
                pix = QPixmap.fromImage(qimg)

                # 1. Left active film and paper cards (WYSIWYG)
                self.thumb_controller.update_active_card_wysiwyg(
                    self.film_cards, self.paper_cards,
                    self.current_film_stock, self.current_paper_stock, pix
                )

                # 2. Bottom filmstrip active frame thumbnail
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

    def _update_stock_mode_ui(self):
        """Update parameter availability based on film type (negative vs slide/reversal vs none/bypass) and print mode."""
        is_none = (self.current_film_stock == "none")
        if hasattr(self, 'sidebar_content') and self.sidebar_content is not None:
            self.sidebar_content.setEnabled(not is_none)
            if is_none:
                self.sidebar_content.setToolTip("当前处于数码原片旁通模式 (无胶片介质)，所有胶片物理调色参数已禁用")
            else:
                self.sidebar_content.setToolTip("")

        stock_combos = [
            getattr(self, 'combo_format', None),
            getattr(self, 'combo_diffusion', None),
            getattr(self, 'combo_diff_strength', None),
            getattr(self, 'combo_illuminant', None),
        ]
        for c in stock_combos:
            if c is not None:
                c.setEnabled(not is_none)

        if is_none:
            # When in pure digital RAW bypass mode, paper cards are locked to "none"
            if hasattr(self, 'paper_cards') and self.paper_cards:
                for pid, card in self.paper_cards.items():
                    card.setEnabled(pid == "none")
                    if pid != "none":
                        card.setToolTip("数码原片旁通状态无需相纸印相")
                    else:
                        card.setToolTip(getattr(card, "original_tooltip", ""))
            self._update_diffusion_ui_state()
            return

        is_pos = self.engine.is_positive_film(self.current_film_stock) if hasattr(self.engine, 'is_positive_film') else False
        is_scan = (getattr(self, "current_print_mode", "optical") == "scan") or (self.current_paper_stock == "none")

        disable_paper_sliders = is_pos or is_scan
        paper_sliders = [
            getattr(self, 'slider_morph_gamma', None),
            getattr(self, 'slider_exhaustion', None),
        ]
        for s in paper_sliders:
            if s is not None:
                s.setEnabled(not disable_paper_sliders)
                if disable_paper_sliders:
                    s.setToolTip("当前选用反转正片或数码扫描模式，相纸显影特性自动旁路")
                else:
                    s.setToolTip(getattr(s, "default_tooltip", ""))

        enlarger_sliders = [
            getattr(self, 'slider_cyan', getattr(self, 'slider_enlarger_c', None)),
            getattr(self, 'slider_magenta', getattr(self, 'slider_enlarger_m', None)),
            getattr(self, 'slider_yellow', getattr(self, 'slider_enlarger_y', None)),
            getattr(self, 'slider_print_exp', None),
            getattr(self, 'slider_preflash', None),
        ]
        for s in enlarger_sliders:
            if s is not None:
                s.setEnabled(not is_pos)
                if is_pos:
                    s.setToolTip("反转正片无暗房放大机滤镜及曝光预闪环节")
                else:
                    s.setToolTip(getattr(s, "default_tooltip", ""))

        if hasattr(self, 'combo_illuminant') and self.combo_illuminant is not None:
            self.combo_illuminant.setEnabled(not is_pos)

        # Left panel paper cards: lock and dim when positive reversal film is active
        if hasattr(self, 'paper_cards') and self.paper_cards:
            for pid, card in self.paper_cards.items():
                if is_pos:
                    card.setEnabled(pid == "none")
                    if pid != "none":
                        card.setToolTip("反转正片无需相纸放大环节")
                    else:
                        card.setToolTip(getattr(card, "original_tooltip", ""))
                else:
                    card.setEnabled(True)
                    card.setToolTip(getattr(card, "original_tooltip", ""))

        self._update_diffusion_ui_state()

    def _update_diffusion_ui_state(self):
        """Enable or gray out diffusion strength and warmth controls based on diffusion_family."""
        is_film_none = (getattr(self, "current_film_stock", "none") == "none")
        diff_family = str(self.current_params.get("diffusion_family", "none")).lower()
        is_diff_none = (diff_family == "none") or is_film_none

        if hasattr(self, 'combo_diff_strength') and self.combo_diff_strength is not None:
            self.combo_diff_strength.setEnabled(not is_diff_none)
            if is_diff_none:
                self.combo_diff_strength.setToolTip("请先在上方选择柔光镜型号以启用挡位调节")
            else:
                self.combo_diff_strength.setToolTip(getattr(self.combo_diff_strength, "default_tooltip", "柔光镜散射密度挡位 (1/8 ~ 2 档)"))

        if hasattr(self, 'slider_diff_warmth') and self.slider_diff_warmth is not None:
            self.slider_diff_warmth.setEnabled(not is_diff_none)
            if is_diff_none:
                self.slider_diff_warmth.setToolTip("请先在上方选择柔光镜型号以启用冷暖偏置调节")
            else:
                self.slider_diff_warmth.setToolTip(getattr(self.slider_diff_warmth, "default_tooltip", "柔光光晕波长能量再分配，正值偏复古暖金，负值偏清冷晨光"))

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
        if hasattr(self, 'combo_lens_setup'):
            self.combo_lens_setup.setEnabled(prof_en)
        if hasattr(self, 'combo_lens_make'):
            self.combo_lens_make.setEnabled(prof_en)
        if hasattr(self, 'combo_lens_model'):
            self.combo_lens_model.setEnabled(prof_en)
        if hasattr(self, 'combo_lens_profile'):
            self.combo_lens_profile.setEnabled(prof_en)
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
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

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
        if setup_mode == "auto":
            self._auto_detect_lens_profile()
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
        model_items = [("无", "")] + [(m_name, m_name) for m_name, _, _ in models]

        self._is_updating_sliders = True
        try:
            self.combo_lens_model.set_items(model_items, default_val="")
            self.combo_lens_profile.set_items([("无", "")], default_val="")
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

        self._is_updating_sliders = True
        try:
            def_path = matching_paths[0][1] if matching_paths else ""
            self.combo_lens_profile.set_items(prof_items, default_val=def_path)
        finally:
            self._is_updating_sliders = False

        if matching_paths:
            lens_cfg["profile_name"] = matching_paths[0][0]
            lens_cfg["profile_path"] = matching_paths[0][1]
        else:
            lens_cfg["profile_name"] = ""
            lens_cfg["profile_path"] = ""
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

    def _on_lens_profile_changed(self, profile_path):
        if getattr(self, "_is_updating_sliders", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg["profile_path"] = profile_path
        if hasattr(self, 'combo_lens_profile') and hasattr(self.combo_lens_profile, 'combo'):
            lens_cfg["profile_name"] = self.combo_lens_profile.combo.currentText()
        self._apply_lens_to_canvas()
        self._mark_current_photo_dirty()
        self._save_current_photo_sdc()
        self.push_undo_state()

    def _on_lens_amount_changed(self, key, val):
        if getattr(self, "_is_updating_sliders", False) or getattr(self, "_is_switching_photo", False):
            return
        lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
        lens_cfg[key] = val
        fl = None
        ap = None
        if self.active_photo_id:
            photo = next((p for p in self.photos if p.get("id") == self.active_photo_id), None)
            if photo:
                exif = photo.get("exif", {})
                fl = exif.get("focal_length")
                ap = exif.get("aperture")
        from core.lens_correction import compute_effective_lens_params
        try:
            eff = compute_effective_lens_params(lens_cfg, focal_length=fl, aperture=ap)
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
        if self.active_photo_id:
            photo = next((p for p in self.photos if p.get("id") == self.active_photo_id), None)
            if photo:
                exif = photo.get("exif", {})
                fl = exif.get("focal_length")
                ap = exif.get("aperture")
        from core.lens_correction import compute_effective_lens_params
        try:
            eff = compute_effective_lens_params(lens_cfg, focal_length=fl, aperture=ap)
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

            # 1. Populate makes
            makes = db.get_available_makes()
            make_items = [("无", "")] + [(m, m) for m in makes]
            if make and make not in [m for _, m in make_items]:
                make_items.append((make, make))
            self.combo_lens_make.set_items(make_items, default_val=make)

            # 2. Populate models
            models = db.get_models_for_make(make) if make else []
            model_items = [("无", "")] + [(m_name, m_name) for m_name, _, _ in models]
            if model and model not in [m for _, m in model_items]:
                model_items.append((model, model))
            self.combo_lens_model.set_items(model_items, default_val=model)

            # 3. Populate profiles
            prof_items = [("无", "")]
            for m_name, p_name, p_p in models:
                if m_name == model:
                    prof_items.append((p_name, p_p))
            if prof_path and prof_path not in [p for _, p in prof_items]:
                p_display = lens_cfg.get("profile_name") or "已加载配置文件"
                prof_items.append((p_display, prof_path))
            self.combo_lens_profile.set_items(prof_items, default_val=prof_path)
        finally:
            self._is_updating_sliders = False

    def _auto_detect_lens_profile(self):
        """Auto-detect camera/lens from photo EXIF metadata and match Adobe LCP profile."""
        if not self.active_photo_id:
            return
        photo = next((p for p in self.photos if p.get("id") == self.active_photo_id), None)
        if not photo:
            return

        from core.lens_correction import LCPDatabase, read_lens_metadata
        exif = photo.get("exif", {})
        meta = read_lens_metadata(photo["path"])

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
            lens_cfg["make"] = cam_make or ""
            lens_cfg["model"] = lens_model or cam_model or "未找到匹配镜头"
            lens_cfg["profile_name"] = "无配置文件"
            lens_cfg["profile_path"] = ""

        if hasattr(self, 'combo_lens_make'):
            self._sync_lens_combos_from_cfg(lens_cfg)

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
                old_p = next((p for p in self.photos if p.get("id") == old_pid), None)
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
            if p.get("id") not in lru_set and p.get("float_img") is not None:
                p["float_img"] = None
                evicted = True
        if hasattr(self, 'engine') and hasattr(self.engine, 'session_photos'):
            for ep in self.engine.session_photos:
                if ep.get("id") not in lru_set and ep.get("raw_preview") is not None:
                    ep["raw_preview"] = None
                    evicted = True

        if evicted:
            import gc
            gc.collect()

    def on_switch_photo(self, photo_id):
        # Stop any lingering live/debounce timers from previous photo
        if hasattr(self, '_live_thumb_timer') and self._live_thumb_timer.isActive():
            self._live_thumb_timer.stop()
        if hasattr(self, '_cards_debounce_timer') and self._cards_debounce_timer.isActive():
            self._cards_debounce_timer.stop()
        if hasattr(self, 'thumb_controller') and self.thumb_controller:
            self.thumb_controller.cancel_card_preview_worker()

        # If user clicks the currently active photo and it is already fully loaded, avoid redundant reload
        if photo_id == self.active_photo_id:
            curr_photo = next((p for p in self.photos if p.get("id") == photo_id), None)
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
                prev_photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
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
            photo = next((p for p in self.photos if p["id"] == photo_id), None)
            if not photo:
                return

            self.active_photo_id = photo_id
            self._current_photo_path = photo["path"]
            if getattr(self.filmstrip, "_active_id", None) != photo_id:
                self.filmstrip.set_active_photo(photo_id, scroll_to_visible=True, reset_selection=True)
            self.canvas.set_has_library_photos(bool(self.photos))

            # Automatically return parameter sidebar to top (exposure pipeline start)
            if hasattr(self, "sidebar_scroll") and self.sidebar_scroll:
                self.sidebar_scroll.verticalScrollBar().setValue(0)

            # 3. Ensure EXIF metadata (camera WB, model, etc.)
            exif_data = photo.get("exif")
            if not exif_data or exif_data.get("camera_model", "-") == "-":
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
                if abs(base_ev) < 0.0001:
                    c_chk = session_cache_manager.get_photo_cache(photo["path"])
                    if c_chk and c_chk.get("base_ev"):
                        base_ev = float(c_chk["base_ev"])
                        photo["base_ev"] = base_ev
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

            self.filmstrip.set_photo_dirty(photo_id, self._photo_is_dirty)
            self._update_window_title()

            # 5. Upload photo to OpenGL canvas texture (lazy load preview or RAW if needed)
            fn = photo.get("filename", os.path.basename(photo["path"]))
            if photo.get("float_img") is None:
                cached_prev = session_cache_manager.get_photo_preview(photo["path"])
                if cached_prev is not None:
                    photo["float_img"] = cached_prev
                    if abs(base_ev) < 0.0001 and photo.get("is_raw", True):
                        base_ev = SpektraEngine.calculate_auto_exposure_ev(cached_prev)
                        photo["base_ev"] = base_ev
                        self._active_base_ev = base_ev
                        self.current_params["base_ev"] = base_ev
                        session_cache_manager.save_photo_cache(photo["path"], photo, save_preview=False)
                else:
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
                        # Asynchronous load so that the spinner rotates smoothly at 30 FPS without freezing
                        if not getattr(self, "_is_importing", False) and not getattr(self, "_is_multi_selecting", False):
                            self.set_app_status(f"解析 RAW: {fn}...", spinning=True, timeout_ms=0)

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
                    self.set_app_status(f"底片已就绪: {fn}", spinning=False, timeout_ms=2000)

            self._manage_photo_memory_lru(photo_id)

            # Ensure lens configuration is present and auto-detected if in auto mode
            lens_cfg = self.current_params.setdefault("lens", self._get_default_lens_params())
            if lens_cfg.get("setup", "auto") == "auto":
                self._auto_detect_lens_profile()
            fl = exif_data.get("focal_length") if exif_data else None
            ap = exif_data.get("aperture") if exif_data else None
            from core.lens_correction import compute_effective_lens_params
            eff_lens = compute_effective_lens_params(lens_cfg, focal_length=fl, aperture=ap)
            lens_cfg.update(eff_lens)

            self._sync_canvas_params()
            self._update_all_sliders_from_params()

            # 6. Apply print mode, film & paper LUT in a single unified, instant pass
            eff_paper = self.current_paper_stock if self.current_print_mode == "optical" else "none"
            lut_params = dict(self.current_params)
            lut_params["print_mode"] = self.current_print_mode
            lut_3d = self.engine.get_3d_lut(self.current_film_stock, eff_paper, lut_size=33, params_dict=lut_params)
            self.canvas.set_lut(lut_3d)

            hal_preset = self.engine.get_film_halation_preset(self.current_film_stock) if hasattr(self.engine, 'get_film_halation_preset') else None
            grain_preset = self.engine.get_film_grain_preset(self.current_film_stock) if hasattr(self.engine, 'get_film_grain_preset') else None
            self.canvas.update_params(film_stock=self.current_film_stock, _halation_preset=hal_preset, _grain_preset=grain_preset)

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
                    if self.current_film_stock != "none" or self.current_paper_stock != "none":
                        th_display = self.thumb_controller.grade_thumbnail_fast(
                            raw_thumb, self.current_film_stock, self.current_paper_stock, self.current_params
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
            target_photo = next((p for p in self.photos if p["id"] == photo_id), None)
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
                self.set_app_status("底片解析失败", spinning=False, timeout_ms=3000)
            return

        float_img = res.get("float_img")
        if float_img is None:
            float_img = res.get("raw_preview")
        if float_img is None:
            float_img = session_cache_manager.get_photo_preview(photo_path)
        if float_img is None:
            if self.active_photo_id == photo_id:
                self.set_app_status("底片解析失败", spinning=False, timeout_ms=3000)
            return



        target_photo = next((p for p in self.photos if p["id"] == photo_id), None)
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
                    self.set_app_status(f"底片已就绪: {fn}", spinning=False, timeout_ms=2000)

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
        paper_cards_dict = self.paper_cards if do_papers else {}

        params = dict(self.current_params)
        params["base_temp"] = getattr(self, "_active_base_temp", 5500.0)
        params["base_tint"] = getattr(self, "_active_base_tint", 1.0)
        params["print_mode"] = getattr(self, "current_print_mode", "optical")

        self.thumb_controller.trigger_candidate_card_previews(
            thumb_data,
            film_cards_dict,
            paper_cards_dict,
            self.current_film_stock,
            self.current_paper_stock,
            params,
            on_film_ready=self._on_film_thumb_ready,
            on_paper_ready=self._on_paper_thumb_ready
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
        if stock_id == self.current_film_stock and card.thumbnail_pixmap is not None:
            return
        card.set_thumbnail(pixmap)

    def _on_paper_thumb_ready(self, stock_id, pixmap):
        card = self.paper_cards.get(stock_id)
        if not card:
            return
        if stock_id == self.current_paper_stock and card.thumbnail_pixmap is not None:
            return
        card.set_thumbnail(pixmap)

    def on_remove_photo(self, photo_id):
        idx = next((i for i, p in enumerate(self.photos) if p["id"] == photo_id), -1)
        if idx == -1:
            return

        # Determine next photo to activate:
        # Standard professional darkroom behavior: switch to the photo immediately following the deleted one.
        # If the deleted photo was the last one, switch to the new last photo (idx - 1).
        next_id = None
        if len(self.photos) > 1:
            if idx + 1 < len(self.photos):
                next_id = self.photos[idx + 1]["id"]
            else:
                next_id = self.photos[idx - 1]["id"]

        self.photos = [p for p in self.photos if p["id"] != photo_id]
        if hasattr(self, 'engine'):
            self.engine.remove_photo(photo_id)
        self._session_is_dirty = True
        self._update_window_title()

        if not self.photos:
            self.action_clear_photos()
        else:
            self.filmstrip.remove_photo(photo_id, next_active_id=next_id)
            self.on_switch_photo(next_id)

    def on_remove_multiple_photos(self, photo_ids):
        id_set = set(photo_ids)
        if not id_set:
            return

        # Find surviving candidate to activate following deletion:
        # 1. If currently active photo is NOT in id_set and survives, retain it as active.
        # 2. If currently active photo IS in id_set (or no active photo):
        #    Anchor at active photo index (or the last deleted photo index) and search forward
        #    for the first surviving photo following it.
        # 3. If no surviving photo follows the anchor, search backward for the closest surviving photo.
        active_idx = next((i for i, p in enumerate(self.photos) if p["id"] == self.active_photo_id), -1)

        next_id = None
        if self.active_photo_id and self.active_photo_id not in id_set:
            next_id = self.active_photo_id
        else:
            del_indices = [i for i, p in enumerate(self.photos) if p["id"] in id_set]
            anchor_idx = active_idx if active_idx != -1 else (max(del_indices) if del_indices else -1)

            if anchor_idx != -1:
                # Search forward for the first photo not being deleted
                for p in self.photos[anchor_idx + 1:]:
                    if p["id"] not in id_set:
                        next_id = p["id"]
                        break
                # If no photo after anchor survives, search backward
                if not next_id:
                    for p in reversed(self.photos[:anchor_idx]):
                        if p["id"] not in id_set:
                            next_id = p["id"]
                            break

        self.photos = [p for p in self.photos if p["id"] not in id_set]
        if hasattr(self, 'engine'):
            for pid in photo_ids:
                self.engine.remove_photo(pid)
        self._session_is_dirty = True
        self._update_window_title()

        if not self.photos:
            self.action_clear_photos()
        else:
            if not next_id and self.photos:
                next_id = self.photos[0]["id"]
            self.filmstrip.remove_multiple_photos(photo_ids, next_active_id=next_id)
            self.on_switch_photo(next_id)

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
            self.set_app_status("删除失败：无法将文件移至回收站", timeout_ms=3000)
            return

        from core import session_cache_manager
        for p in target_photos:
            fpath = p.get("path")
            if fpath:
                session_cache_manager.invalidate_photo_cache(fpath)

        count = len(target_photos)
        self.on_remove_multiple_photos(photo_ids)
        self.set_app_status(f"已将 {count} 个底片文件移至回收站", timeout_ms=3000)

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
        target_photos = [p for p in self.photos if p.get("id") in id_set]
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
                p_params = dict(self.current_params)
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
        for p in target_photos:
            pid = p["id"]
            p_path = p.get("path")
            if p_path:
                sdc_manager.delete_sdc(p_path)
            p.pop("params", None)
            p["film_profile"] = "none"
            p["paper_profile"] = "none"
            p["print_mode"] = "scan"
            p["crop"] = dict(default_crop)
            p["is_dirty"] = False
            p["is_edited"] = False

            # Instantly revert filmstrip thumbnail to raw base
            if hasattr(self, 'thumb_controller'):
                self.thumb_controller.revert_photo_to_raw(p, self.filmstrip)
            if hasattr(self, 'filmstrip'):
                self.filmstrip.set_photo_dirty(pid, False)
                self.filmstrip.set_photo_crop(pid, False)

            if pid == self.active_photo_id:
                active_cleared = True

            post_photos[pid] = {
                "film": "none",
                "paper": "none",
                "print_mode": "scan",
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
            self.current_paper_stock = "none"
            self.current_print_mode = "scan"

            base_temp = float(getattr(self, "_active_base_temp", 5500.0))
            base_tint = float(getattr(self, "_active_base_tint", 1.0))
            defaults = self.get_default_params(base_temp, base_tint)
            self.current_params.clear()
            self.current_params.update(defaults)
            self.current_params["print_mode"] = "scan"
            self.current_params["crop"] = dict(default_crop)

            active_p = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
            if active_p:
                active_p["params"] = None
                active_p["film_profile"] = "none"
                active_p["paper_profile"] = "none"
                active_p["print_mode"] = "scan"
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
                card.set_active(sid == "none")
                if sid != "none":
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

        if active_cleared and len(id_set) == 1:
            self.set_app_status("已清除暗房配置并重置为原生底片", timeout_ms=3000)
        else:
            self.set_app_status(f"已清除 {len(id_set)} 项底片配置并重置为原生底片", timeout_ms=3000)

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
        target_photos = [p for p in self.photos if p.get("id") in id_set]
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
        msg = "已清除裁切" if count == 1 else f"已清除 {count} 项底片裁切"
        self.set_app_status(msg, timeout_ms=1500)

    def action_clear_photos(self):
        """Completely flush bottom filmstrip, engine session, and canvas without exiting to start screen."""
        self._cancel_active_import(notify_user=False)
        self._photo_id_counter = 0
        if self._photo_is_dirty and self._current_photo_path:
            sdc_manager.save_sdc(
                self._current_photo_path,
                self.current_film_stock,
                self.current_paper_stock,
                self.current_params
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

        self.btn_title_export.setVisible(False)
        self._update_window_title()
        if hasattr(self, 'exif_widget'):
            self.exif_widget.set_exif_data({})
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
        ext = ".tif" if "tif" in fmt else (".png" if "png" in fmt else ".jpg")

        base_name = os.path.splitext(os.path.basename(self._current_photo_path))[0]
        default_out = os.path.join(last_dir, f"{base_name}{ext}")

        dlg = SingleExportDialog(default_out, parent=self)
        if dlg.exec():
            cfg = dlg.get_export_config()
            self._do_export_single(cfg)

    def _do_export_single(self, cfg):
        if not self._current_photo_path:
            return

        curr_p = next((p for p in self.photos if p.get("path") == self._current_photo_path), None)
        base_ev = float(curr_p.get("base_ev", 0.0)) if curr_p else 0.0
        if abs(base_ev) < 0.0001:
            base_ev = float(getattr(self, "_active_base_ev", 0.0))
        if abs(base_ev) < 0.0001 and hasattr(self, "engine"):
            matched_ep = next((p for p in self.engine.session_photos if p.get("path") == self._current_photo_path), None)
            if matched_ep and "base_ev" in matched_ep:
                base_ev = float(matched_ep["base_ev"])
        export_params = dict(self.current_params)
        export_params.update({
            "source_path": self._current_photo_path,
            "film_stock": self.current_film_stock,
            "paper_stock": self.current_paper_stock,
            "print_mode": getattr(self, "current_print_mode", "optical"),
            "base_ev": base_ev,
            "base_temp": getattr(self, "_active_base_temp", 5500.0),
            "base_tint": getattr(self, "_active_base_tint", 1.0),
            "format": cfg.get("format", "jpeg"),
            "engine": cfg.get("engine", "wysiwyg"),
            "bit_depth": cfg.get("bit_depth", 8),
            "colorspace": cfg.get("colorspace", "sRGB"),
            "dpi": cfg.get("dpi", 300),
            "quality": cfg.get("quality", 9),
            "scale_pct": cfg.get("scale_pct", 100),
            "subsampling": cfg.get("subsampling", "4:4:4"),
            "progressive": cfg.get("progressive", True),
            "tiff_compression": cfg.get("tiff_compression", "lzw"),
            "png_level": cfg.get("png_level", 6)
        })

        out_path = cfg["path"].strip() if cfg.get("path") else ""
        if not out_path or os.path.isdir(out_path) or out_path.endswith(('/', '\\')):
            base_name = os.path.splitext(os.path.basename(self._current_photo_path))[0] if self._current_photo_path else "image"
            fmt_str = str(cfg.get("format", "jpeg")).lower()
            ext = ".tif" if "tif" in fmt_str else (".png" if "png" in fmt_str else ".jpg")
            fallback_dir = out_path if out_path else (os.path.dirname(self._current_photo_path) if self._current_photo_path else os.path.expanduser("~"))
            out_path = os.path.join(fallback_dir, f"{base_name}{ext}")

        # Remember output directory for subsequent exports
        try:
            config_manager.set_last_export_dir(os.path.dirname(out_path))
        except Exception:
            pass

        pix = None
        if self._current_photo_thumb is not None:
            trgb = self._current_photo_thumb
            th, tw = trgb.shape[:2]
            qimg = QImage(trgb.data, tw, th, tw * 3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

        import uuid
        tid = f"task_{uuid.uuid4().hex[:8]}"
        task = ExportTask(tid, os.path.basename(out_path),
                          self._current_photo_path, out_path, export_params, thumbnail=pix)
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
        """Batch export multiple photos with background export queue."""
        if not photo_ids:
            return

        # If only 1 photo was passed, route to single photo export
        if len(photo_ids) == 1:
            pid = photo_ids[0]
            photo = next((p for p in self.photos if p["id"] == pid), None)
            if photo and photo.get("path"):
                self._current_photo_path = photo["path"]
                last_dir = config_manager.get_last_export_dir()
                if not last_dir or not os.path.exists(last_dir):
                    last_dir = os.path.dirname(self._current_photo_path) if self._current_photo_path else os.path.expanduser("~")
                last_cfg = config_manager.get_last_export_settings()
                fmt = str(last_cfg.get("format", "jpeg")).lower() if last_cfg else "jpeg"
                ext = ".tif" if "tif" in fmt else (".png" if "png" in fmt else ".jpg")
                base_name = os.path.splitext(os.path.basename(self._current_photo_path))[0]
                default_out = os.path.join(last_dir, f"{base_name}{ext}")
                dlg = SingleExportDialog(default_out, parent=self)
                if dlg.exec():
                    cfg = dlg.get_export_config()
                    self._do_export_single(cfg)
            return

        last_dir = config_manager.get_last_export_dir()
        if not last_dir or not os.path.exists(last_dir):
            first_photo = next((p for p in self.photos if p["id"] in photo_ids), None)
            last_dir = os.path.dirname(first_photo["path"]) if first_photo else os.path.expanduser("~")

        dialog = BatchExportDialog(last_dir, batch_count=len(photo_ids), parent=self)
        if not dialog.exec():
            return

        cfg = dialog.get_export_config()
        out_dir = cfg.get("path", "").strip()
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
        elif fmt == "png":
            ext = ".png"
        else:
            ext = ".jpg"

        prefix = cfg.get("prefix", "").strip()
        suffix = cfg.get("suffix", "").strip()

        import uuid
        enqueued_count = 0
        new_tasks = []
        for i, pid in enumerate(photo_ids):
            photo = next((p for p in self.photos if p["id"] == pid), None)
            if not photo:
                continue

            file_p = photo["path"]
            sdc = sdc_manager.load_sdc(file_p)
            film = sdc.get("film_profile", "kodak_portra_400") if sdc else "kodak_portra_400"
            paper = sdc.get("paper_profile", "kodak_2383") if sdc else "kodak_2383"
            params = sdc.get("params", {}) if sdc else {}

            exif_d = photo.get("exif", {})
            b_temp = float(exif_d.get("color_temp", 5500.0) if exif_d else 5500.0)
            b_tint = float(exif_d.get("tint", 1.0) if exif_d else 1.0)

            b_ev = float(photo.get("base_ev", 0.0))
            if abs(b_ev) < 0.0001 and hasattr(self, "engine"):
                matched_ep = next((p for p in self.engine.session_photos if p.get("path") == file_p), None)
                if matched_ep and "base_ev" in matched_ep:
                    b_ev = float(matched_ep["base_ev"])

            export_params = {
                "source_path": file_p,
                "film_stock": film,
                "paper_stock": paper,
                "print_mode": photo.get("print_mode", params.get("print_mode", "optical")),
                "base_ev": b_ev,
                "base_temp": b_temp,
                "base_tint": b_tint,
                "exposure_ev": params.get("exposure_ev", 0.0),
                "highlights": params.get("highlights", 0.0),
                "shadows": params.get("shadows", 0.0),
                "color_temp": params.get("color_temp", b_temp),
                "tint": params.get("tint", b_tint),
                "enlarger_cyan": params.get("enlarger_cyan", 0.0),
                "enlarger_magenta": params.get("enlarger_magenta", 0.0),
                "enlarger_yellow": params.get("enlarger_yellow", 0.0),
                "print_exposure": params.get("print_exposure", 1.0),
                "pre_flash": params.get("pre_flash", 0.0),
                "halation": params.get("halation_amount", params.get("halation", 1.0)),
                "halation_amount": params.get("halation_amount", params.get("halation", 1.0)),
                "halation_spatial_scale": params.get("halation_spatial_scale", 1.0),
                "scatter_amount": params.get("scatter_amount", 1.0),
                "boost_ev": params.get("boost_ev", params.get("halation_boost", 0.0)),
                "halation_boost": params.get("boost_ev", params.get("halation_boost", 0.0)),
                "grain": params.get("grain", 0.4),
                "grain_size": params.get("grain_size", 1.0),
                "grain_cloud_blur": params.get("grain_cloud_blur", 1.0),
                "grain_colour": params.get("grain_colour", 1.0),
                "engine": cfg.get("engine", "wysiwyg"),
                "format": fmt,
                "quality": cfg.get("quality", 9),
                "dpi": cfg.get("dpi", 300),
                "bit_depth": cfg.get("bit_depth", 8),
                "colorspace": cfg.get("colorspace", "sRGB"),
                "subsampling": cfg.get("subsampling", "4:4:4"),
                "progressive": cfg.get("progressive", True),
                "tiff_compression": cfg.get("tiff_compression", "lzw"),
                "png_level": cfg.get("png_level", 6),
                "scale_pct": cfg.get("scale_pct", 100)
            }

            stem = os.path.splitext(os.path.basename(file_p))[0]
            out_fn = f"{prefix}{stem}{suffix}{ext}"
            out_target = os.path.join(out_dir, out_fn)

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

        self._set_export_status(f"已将 {enqueued_count} 张底片加入扫描队列", state="busy")

    def _create_dark_message_box(self, title, text, icon=QMessageBox.Icon.Information):
        mb = QMessageBox(self)
        dlg_icon_path = os.path.join(get_resource_dir(), "icons", "dlg_info.png")
        if os.path.exists(dlg_icon_path):
            mb.setWindowIcon(QIcon(dlg_icon_path))
        else:
            mb.setWindowIcon(self.windowIcon())
        mb.setWindowTitle(title)
        mb.setText(text)
        mb.setIcon(icon)
        mb.setStyleSheet("""
            QMessageBox {
                background: #14151a;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 12px;
            }
            QPushButton {
                background: #20222b;
                color: #e2e8f0;
                border: 1px solid #333647;
                border-radius: 4px;
                padding: 5px 16px;
                font-size: 12px;
                min-width: 60px;
            }
            QPushButton:hover {
                background: #2a2d3a;
                border-color: #f59e0b;
                color: #f59e0b;
            }
            QPushButton:pressed {
                background: #14151a;
                color: #d97706;
                border-color: #d97706;
            }
        """)
        apply_dark_titlebar(mb)
        return mb

    def _show_info_dialog(self, title, text):
        mb = self._create_dark_message_box(title, text, QMessageBox.Icon.Information)
        mb.exec()

    def _show_warning_dialog(self, title, text):
        mb = self._create_dark_message_box(title, text, QMessageBox.Icon.Warning)
        mb.exec()

    def _show_question_dialog(self, title, text):
        mb = self._create_dark_message_box(title, text, QMessageBox.Icon.Question)
        mb.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        mb.setDefaultButton(QMessageBox.StandardButton.No)
        return mb.exec() == QMessageBox.StandardButton.Yes

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

    def open_stock_manager(self, mode="film"):
        dlg = StockManagerDialog(mode=mode, engine=self.engine, parent=self)
        dlg.stocksChanged.connect(self._on_stocks_changed)
        dlg.exec()

    def _on_stocks_changed(self):
        for card in list(self.film_cards.values()):
            self.film_grid.removeWidget(card)
            card.deleteLater()
        self.film_cards.clear()
        self._populate_film_grid()

        for card in list(self.paper_cards.values()):
            self.paper_grid.removeWidget(card)
            card.deleteLater()
        self.paper_cards.clear()
        self._populate_paper_grid()
        self.reflow_grids()

    def open_preferences(self):
        old_res = config_manager.get_preferences().get("preview_max_edge", 2048)
        dlg = PreferencesDialog(parent=self)
        if dlg.exec():
            new_res = config_manager.get_preferences().get("preview_max_edge", 2048)
            if old_res != new_res:
                for p in self.photos:
                    p["float_img"] = None
                session_cache_manager.clear_cache()
                if self._current_photo_path:
                    photo = next((p for p in self.photos if p.get("path") == self._current_photo_path), None)
                    if photo:
                        res = self.engine.load_image(photo["path"])
                        if res.get("success"):
                            photo["float_img"] = res.get("float_img")
                            session_cache_manager.save_photo_preview(photo["path"], photo["float_img"])
                            m_w = photo.get("width") or (photo.get("meta", {}).get("width") if isinstance(photo.get("meta"), dict) else None)
                            m_h = photo.get("height") or (photo.get("meta", {}).get("height") if isinstance(photo.get("meta"), dict) else None)
                            self.canvas.set_image(photo["float_img"], master_width=m_w, master_height=m_h, reset_zoom=False)
                            self._update_photo_info_display()
                            self.set_app_status("已按新首选项重新加载底片全尺寸纹理", timeout_ms=3000)
            if hasattr(self, 'canvas'):
                self.canvas.update()

    def _export_film_lut(self, stock_id):
        last_d = config_manager.get_last_dialog_dir("lut")
        initial_p = os.path.join(last_d, f"{stock_id}.cube") if last_d else f"{stock_id}.cube"
        out_file, _ = QFileDialog.getSaveFileName(self, "导出胶卷 3D LUT", initial_p, "3D LUT (*.cube)")
        if out_file:
            config_manager.set_last_dialog_dir("lut", out_file)
            succ = export_stock_lut(self.engine, stock_id, out_file, is_film=True)
            if succ:
                self._show_info_dialog("导出完成", f"胶卷预设已成功导出至:\n{out_file}")
            else:
                self._show_warning_dialog("导出失败", f"导出胶卷预设 {stock_id} 失败，请检查文件写入权限。")

    def _export_paper_lut(self, stock_id):
        last_d = config_manager.get_last_dialog_dir("lut")
        initial_p = os.path.join(last_d, f"{stock_id}.cube") if last_d else f"{stock_id}.cube"
        out_file, _ = QFileDialog.getSaveFileName(self, "导出相纸 3D LUT", initial_p, "3D LUT (*.cube)")
        if out_file:
            config_manager.set_last_dialog_dir("lut", out_file)
            succ = export_stock_lut(self.engine, stock_id, out_file, is_film=False)
            if succ:
                self._show_info_dialog("导出完成", f"相纸预设已成功导出至:\n{out_file}")
            else:
                self._show_warning_dialog("导出失败", f"导出相纸预设 {stock_id} 失败，请检查文件写入权限。")

    def _import_film_lut(self):
        last_d = config_manager.get_last_dialog_dir("profile")
        f, _ = QFileDialog.getOpenFileName(self, "导入胶卷物理配置", last_d, "JSON 配置 (*.json);;所有文件 (*.*)")
        if f:
            config_manager.set_last_dialog_dir("profile", f)
            default_name = os.path.splitext(os.path.basename(f))[0]
            dlg = ImportStockDialog(stock_type="film", default_name=default_name, parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                custom_name, custom_desc = dlg.get_data()
                dest = self.engine.import_custom_stock("film", f, custom_name, custom_desc)
                if dest:
                    self._on_stocks_changed()
                    self._show_info_dialog("导入成功", f"胶卷物理配置已成功存入:\n{dest}")
                else:
                    self._show_warning_dialog("导入失败", "导入胶卷配置失败，请检查文件格式。")

    def _import_paper_lut(self):
        last_d = config_manager.get_last_dialog_dir("profile")
        f, _ = QFileDialog.getOpenFileName(self, "导入相纸物理配置", last_d, "JSON 配置 (*.json);;所有文件 (*.*)")
        if f:
            config_manager.set_last_dialog_dir("profile", f)
            default_name = os.path.splitext(os.path.basename(f))[0]
            dlg = ImportStockDialog(stock_type="paper", default_name=default_name, parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                custom_name, custom_desc = dlg.get_data()
                dest = self.engine.import_custom_stock("paper", f, custom_name, custom_desc)
                if dest:
                    self._on_stocks_changed()
                    self._show_info_dialog("导入成功", f"相纸物理配置已成功存入:\n{dest}")
                else:
                    self._show_warning_dialog("导入失败", "导入相纸配置失败，请检查文件格式。")

    def _remove_film_lut(self, stock_id):
        if hasattr(self.engine, 'BUILTIN_FILM_IDS') and stock_id in self.engine.BUILTIN_FILM_IDS:
            self._show_warning_dialog("官方内置不可删", f"'{stock_id}' 为官方内置胶卷配置，不可丢弃。")
            return
        if self._show_question_dialog("确认丢弃", f"确定要永久删除并丢弃自定义胶卷配置 '{stock_id}' 吗？\n相关配置文件将从磁盘中物理移除。"):
            ok = self.engine.delete_custom_stock("film", stock_id)
            if stock_id in self.film_cards:
                card = self.film_cards.pop(stock_id)
                self.film_grid.removeWidget(card)
                card.deleteLater()
                self.reflow_grids()
            self._on_stocks_changed()
            self._show_info_dialog("已丢弃", f"自定义胶卷配置 '{stock_id}' 已从系统和磁盘中物理移除。")

    def _remove_paper_lut(self, stock_id):
        if hasattr(self.engine, 'BUILTIN_PAPER_IDS') and stock_id in self.engine.BUILTIN_PAPER_IDS:
            self._show_warning_dialog("官方内置不可删", f"'{stock_id}' 为官方内置相纸配置，不可丢弃。")
            return
        if self._show_question_dialog("确认丢弃", f"确定要永久删除并丢弃自定义相纸配置 '{stock_id}' 吗？\n相关配置文件将从磁盘中物理移除。"):
            ok = self.engine.delete_custom_stock("paper", stock_id)
            if stock_id in self.paper_cards:
                card = self.paper_cards.pop(stock_id)
                self.paper_grid.removeWidget(card)
                card.deleteLater()
                self.reflow_grids()
            self._on_stocks_changed()
            self._show_info_dialog("已丢弃", f"自定义相纸配置 '{stock_id}' 已从系统和磁盘中物理移除。")

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

    def closeEvent(self, event):
        # Prompt user to save if current session has unsaved modifications
        if not self._prompt_save_if_session_dirty():
            event.ignore()
            return

        # Silently ensure any modified parameters are saved to .sdc
        try:
            if self._photo_is_dirty and self._current_photo_path:
                curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                cur_thumb = curr_p.get("thumbnail_rgb") if curr_p else None
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    self.current_paper_stock,
                    self.current_params,
                    thumbnail_rgb=cur_thumb
                )
                self._photo_is_dirty = False
            for p in self.photos:
                if p and p.get("is_dirty") and p.get("path"):
                    sdc_manager.save_sdc(
                        p["path"],
                        p.get("film_profile", self.current_film_stock),
                        p.get("paper_profile", self.current_paper_stock),
                        p.get("params", self.current_params),
                        thumbnail_rgb=p.get("thumbnail_rgb")
                    )
                    p["is_dirty"] = False
        except Exception as e:
            logger.warning(f"Auto-saving .sdc on close: {e}")

        try:
            if hasattr(self, 'thumb_controller'):
                self.thumb_controller.cancel_card_preview_worker()
            if hasattr(self, '_single_photo_loader') and self._single_photo_loader and self._single_photo_loader.isRunning():
                self._single_photo_loader.stop()
                self._single_photo_loader.wait(100)
            if hasattr(self, '_session_restore_worker') and self._session_restore_worker and self._session_restore_worker.isRunning():
                self._session_restore_worker.stop()
                self._session_restore_worker.wait(100)
        except Exception:
            pass

        try:
            is_max = self.isMaximized()
            if is_max and self._saved_normal_geo:
                norm_w = self._saved_normal_geo.width()
                norm_h = self._saved_normal_geo.height()
                norm_x = self._saved_normal_geo.x()
                norm_y = self._saved_normal_geo.y()
            else:
                norm_w = self.width()
                norm_h = self.height()
                norm_x = self.x()
                norm_y = self.y()

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
                self.current_params
            )
        self._photo_is_dirty = False
        event.accept()
