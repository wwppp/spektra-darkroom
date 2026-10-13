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
import ctypes
from ctypes import c_int, byref
from datetime import datetime
import logging

logger = logging.getLogger("SpektraDarkroom.MainWindow")

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import (
    Qt, QPoint, QPointF, QSize, QRect, QRectF, Signal, QTimer, QEvent, QThread,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
)
from PySide6.QtGui import (
    QAction, QFont, QIcon, QKeySequence, QPainter, QColor, QPen, QBrush, QPixmap, QImage, QLinearGradient, QCursor
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QMenuBar, QMenu, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSplitter, QSplitterHandle, QGridLayout, QSizePolicy, QApplication,
    QDialog, QLineEdit, QTextEdit
)

from path_utils import get_icon_path, get_resource_dir
from version import get_app_title, get_full_version_info
import sdc_manager
import config_manager
import session_cache_manager
from ui.export_queue_manager import ExportQueueManager, ExportQueueDialog, ExportTask
from ui.unsaved_session_dialog import UnsavedSessionDialog
from ui.thumbnail_controller import ThumbnailController, StockThumbnailWorker

if sys.platform == "win32":
    from ctypes import wintypes
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

from ui.window_utils import apply_dark_titlebar, get_darkroom_menu_style, get_icon
from ui.smooth_scroll import SmoothScrollArea
from ui.stock_manager_dialog import StockManagerDialog, export_stock_lut
from ui.preferences_dialog import PreferencesDialog
from ui.about_dialog import AboutDialog
from ui.canvas_viewport import CanvasViewport, DarkroomGLCanvas
from ui.accordion import AccordionSection
from ui.adobe_scrub_slider import AdobeScrubSlider, ComboBoxRow
from ui.export_dialog import (
    ExportImageDialog, ExportProgressDialog, SingleExportWorker, BatchExportWorker
)
from ui.filmstrip_widget import FilmstripWidget, DuplicateFilesDialog
from ui.spinner_widget import SpinnerWidget
from ui.histogram_widget import HistogramWidget


class WindowCaptionButton(QPushButton):
    """Vector-drawn Windows 11 style window control button."""
    def __init__(self, btn_type="close", parent=None):
        super().__init__(parent)
        self.btn_type = btn_type
        self.setFixedSize(46, 38)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.is_maximized_state = False

    def set_maximized_state(self, is_max):
        if self.is_maximized_state != is_max:
            self.is_maximized_state = is_max
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # 1. 始终彻底填充背景，防止重绘/DPI变换时的残影与双边伪影
        if self.isDown():
            bg = QColor(241, 112, 122) if self.btn_type == "close" else QColor(38, 40, 50)
        elif self.underMouse():
            bg = QColor(232, 17, 35) if self.btn_type == "close" else QColor(48, 50, 60)
        else:
            bg = QColor(20, 21, 24)

        if not self.is_maximized_state and self.btn_type == "close":
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

        icon_color = QColor(255, 255, 255) if (self.underMouse() and self.btn_type == "close") else QColor(220, 222, 230)
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
                raw_exts = {
                    ".arw", ".srf", ".sr2", ".arq",
                    ".cr2", ".cr3", ".crw",
                    ".nef", ".nrw",
                    ".raf",
                    ".dng",
                    ".rw2",
                    ".orf", ".ori",
                    ".pef", ".ptx",
                    ".3fr", ".fff",
                    ".x3f",
                }
                ext = os.path.splitext(self.path)[1].lower()
                if ext in raw_exts:
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
            import config_manager
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
            import config_manager
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


class FilmGridCard(QFrame):
    """Item 6: Large 2:3 aspect ratio photographic card (114x171px):
    - 2:3 ratio vertical photo preview of the active image
    - Clean film/paper name overlaid at the bottom
    - Active amber glowing border
    - Hover tooltip explaining optical characteristics
    """
    selected = Signal(str)
    exportRequested = Signal(str)
    removeRequested = Signal(str)
    hoverEntered = Signal(str, str)
    sortRequested = Signal(str, str)
    importRequested = Signal(str)

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
        self.stock_id = stock_info["id"]
        self.stock_name = stock_info["name"]
        self.stock_type = stock_type
        self.is_active = is_active
        self.is_custom = is_custom
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
        self.lbl_thumb.setFixedSize(98, 70)
        self.lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_thumb.setStyleSheet("border-radius: 4px; background: #18191f;")
        layout.addWidget(self.lbl_thumb, 0, Qt.AlignmentFlag.AlignCenter)

        # Bottom label container
        self.lbl_name = QLabel(self.clean_name)
        self.lbl_name.setFixedHeight(28)
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_name)

        self._update_appearance()
        self.set_card_tooltip(self.original_tooltip)

    def update_card_dimensions(self, new_w):
        new_w = max(96, min(160, int(new_w)))
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
        c1_str, c2_str = self.SWATCH_COLORS.get(self.stock_id, ("#f59e0b", "#d97706"))
        pm = QPixmap(tw, th)
        pm.fill(QColor(22, 23, 28))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(c1_str))
        p.drawRoundedRect(QRectF(tw * 0.15, th * 0.14, tw * 0.70, th * 0.72), 6, 6)
        p.end()
        self.lbl_thumb.setPixmap(pm)

    def set_active(self, is_active):
        self.is_active = is_active
        self._update_appearance()

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._update_appearance()

    def _update_appearance(self):
        if not self.isEnabled():
            self.setStyleSheet("""
                QFrame#filmGridCard {
                    background: #101114;
                    border: 1px solid #1c1d22;
                    border-radius: 6px;
                }
                QLabel { border: none; background: transparent; color: #404452; }
            """)
            self.lbl_name.setStyleSheet("color: #404452; font-size: 9.5px;")
            return
        if self.is_active:
            self.setStyleSheet("""
                QFrame#filmGridCard {
                    background: rgba(245, 158, 11, 0.08);
                    border: 1px solid #f59e0b;
                    border-radius: 6px;
                }
                QLabel { border: none; background: transparent; }
            """)
            self.lbl_name.setStyleSheet("color: #f59e0b; font-size: 9.5px; font-weight: bold;")
        else:
            self.setStyleSheet("""
                QFrame#filmGridCard {
                    background: #15161c;
                    border: 1px solid #282a36;
                    border-radius: 6px;
                }
                QFrame#filmGridCard:hover {
                    border-color: #4b5168;
                    background: #1b1d24;
                }
                QLabel { border: none; background: transparent; }
            """)
            self.lbl_name.setStyleSheet("color: #cbd5e1; font-size: 9.5px; font-weight: 500;")

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

            act_imp = menu.addAction(f"导入{'胶卷' if self.stock_type == 'film' else '相纸'}配置...")
            icon_imp = os.path.join(get_resource_dir(), "icons", "dlg_import.png")
            if os.path.exists(icon_imp):
                act_imp.setIcon(QIcon(icon_imp))
            act_imp.triggered.connect(lambda: self.importRequested.emit(self.stock_type))

            menu.addSeparator()

            act_del = menu.addAction("丢弃")
            icon_del = os.path.join(get_resource_dir(), "icons", "dlg_discard.png")
            if os.path.exists(icon_del):
                act_del.setIcon(QIcon(icon_del))
            if not self.is_custom:
                act_del.setEnabled(False)
                act_del.setText("丢弃 (官方内置不可删)")
            else:
                act_del.triggered.connect(lambda: self.removeRequested.emit(self.stock_id))

            menu.exec(event.globalPosition().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.hoverEntered.emit(self.stock_type, self.stock_id)


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
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        card = QFrame()
        card.setObjectName("exifCard")
        card.setStyleSheet("""
            QFrame#exifCard {
                background: #15161a;
                border: 1px solid #262834;
                border-radius: 6px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(8, 6, 4, 6)
        c_layout.setSpacing(4)

        # Fixed Header (Issue 4: Title pinned, never scrolls)
        title = QLabel("EXIF")
        title.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;")
        c_layout.addWidget(title)

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
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        grid.setContentsMargins(0, 2, 8, 2)
        grid.setColumnMinimumWidth(0, 75)
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
            lbl_k.setFixedWidth(75)
            lbl_k.setStyleSheet("color: #7b8092; font-size: 10px;")
            lbl_v = QLabel("-")
            lbl_v.setStyleSheet("color: #e2e8f0; font-size: 10px; font-weight: 500;")
            lbl_v.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            lbl_v.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            lbl_v.setWordWrap(True)
            grid.addWidget(lbl_k, row, 0)
            grid.addWidget(lbl_v, row, 1)
            self.labels[key] = lbl_v

        self.scroll_area.setWidget(content)
        c_layout.addWidget(self.scroll_area, 1)

        layout.addWidget(card)

    def set_exif_data(self, data):
        for k, lbl in self.labels.items():
            raw_val = data.get(k, "-")
            val = str(raw_val).strip() if raw_val is not None and str(raw_val).strip() != "" else "-"
            lbl.setText(val)
            lbl.setToolTip(val if val != "-" else "")


class FadingSplitterHandle(QSplitterHandle):
    """Refined splitter handle with delicate theme-matched hairline fading at both ends."""
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._hovered = False

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
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor('#121316'))

        if self.orientation() == Qt.Orientation.Vertical:
            # Horizontal line separating vertical tiers (left sidebar, center splitter)
            cy = h // 2
            margin_x = 4
            line_w = max(0, w - margin_x * 2)
            grad = QLinearGradient(margin_x, cy, max(margin_x + 1, w - margin_x), cy)
            if self._hovered:
                grad.setColorAt(0.0, QColor(245, 158, 11, 40))
                grad.setColorAt(0.03, QColor(245, 158, 11, 200))
                grad.setColorAt(0.5, QColor(251, 191, 36, 255))
                grad.setColorAt(0.97, QColor(245, 158, 11, 200))
                grad.setColorAt(1.0, QColor(245, 158, 11, 40))
            else:
                grad.setColorAt(0.0, QColor(217, 119, 6, 30))
                grad.setColorAt(0.04, QColor(217, 119, 6, 150))
                grad.setColorAt(0.5, QColor(217, 119, 6, 185))
                grad.setColorAt(0.96, QColor(217, 119, 6, 150))
                grad.setColorAt(1.0, QColor(217, 119, 6, 30))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(margin_x, cy, line_w, 1)
        else:
            # Vertical line separating horizontal columns
            cx = w // 2
            margin_y = 4
            line_h = max(0, h - margin_y * 2)
            grad = QLinearGradient(cx, margin_y, cx, max(margin_y + 1, h - margin_y))
            if self._hovered:
                grad.setColorAt(0.0, QColor(245, 158, 11, 40))
                grad.setColorAt(0.03, QColor(245, 158, 11, 200))
                grad.setColorAt(0.5, QColor(251, 191, 36, 255))
                grad.setColorAt(0.97, QColor(245, 158, 11, 200))
                grad.setColorAt(1.0, QColor(245, 158, 11, 40))
            else:
                grad.setColorAt(0.0, QColor(217, 119, 6, 30))
                grad.setColorAt(0.04, QColor(217, 119, 6, 140))
                grad.setColorAt(0.5, QColor(217, 119, 6, 180))
                grad.setColorAt(0.96, QColor(217, 119, 6, 140))
                grad.setColorAt(1.0, QColor(217, 119, 6, 30))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(cx, margin_y, 1, line_h)


class FadingSplitter(QSplitter):
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(7)
        self.setChildrenCollapsible(False)
        self.setStyleSheet("QSplitter { background: transparent; }")

    def createHandle(self):
        return FadingSplitterHandle(self.orientation(), self)

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

        # Film & Paper 1s hover preview state machine
        self._hover_preview_active = False
        self._hover_pending_stock = None
        self._saved_film_stock = None
        self._saved_paper_stock = None
        self._saved_print_mode = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._on_hover_timer_timeout)

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

        QApplication.instance().installEventFilter(self)

        self._init_ui()
        self.export_queue.status_changed.connect(self._on_export_queue_status_changed)
        self._init_menu_and_actions()
        self._wire_canvas_events()
        self._restore_saved_state()

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
                    QTimer.singleShot(40, lambda: (not self.photos and not self._current_session_path) and self._open_session_by_path(last_session))
                else:
                    self.stack.setCurrentIndex(0)
                    self.btn_title_export.setVisible(False)
            else:
                if not getattr(self, "_has_cli_targets", False) and not self.photos:
                    self.stack.setCurrentIndex(0)
                    self.btn_title_export.setVisible(False)

            # 6. Silent session cache auto-cleaning check
            if not os.environ.get("FAST_TEST"):
                QTimer.singleShot(2500, session_cache_manager.check_and_auto_clean)
        except Exception as e:
            logger.warning(f"Error restoring application state: {e}")

    def _restore_saved_session(self, existing_files, active_file=None):
        """Ultra-fast session restoration: loads cached filmstrip thumbnails and EXIF in <100ms.
        Pre-allocates slots for all files to guarantee order and count stability (100 remains 100).
        Any uncached files are decoded directly into their slot indices in the background.
        """
        if not existing_files:
            return

        total_files = len(existing_files)
        self.photos = [None] * total_files
        uncached_slots = []

        target_path = active_file if (active_file and active_file in existing_files) else existing_files[0]
        self._target_initial_active_path = target_path

        import sdc_manager

        for idx, p in enumerate(existing_files):
            pid = f"photo_{idx}_{os.path.basename(p)}"
            sdc = sdc_manager.load_sdc(p)
            film_profile = sdc.get("film_profile", "none") if sdc else "none"
            paper_profile = sdc.get("paper_profile", "none") if sdc else "none"
            print_mode = sdc.get("print_mode", sdc.get("params", {}).get("print_mode", "optical")) if sdc else "scan"
            params = dict(sdc.get("params", {})) if sdc else None
            is_dirty = False
            is_edited = bool(sdc is not None and (film_profile not in ("none", "", None) or paper_profile not in ("none", "", None) or self._is_params_modified(params)))

            is_target = (p == target_path)
            cache_item = session_cache_manager.get_photo_cache(p, load_preview=is_target)

            # Strictly fetch immutable raw thumbnail from session cache
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
                    if (film_profile != "none" or paper_profile != "none" or abs(params.get("exposure_ev", 0.0)) > 0.001):
                        graded_th = self.thumb_controller.grade_thumbnail_fast(raw_th, film_profile, paper_profile, params)
                        th_rgb = graded_th
                    else:
                        th_rgb = raw_th
            else:
                th_rgb = raw_th
                graded_th = None

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
                    "auto_ev": cache_item.get("auto_ev", 0.0),
                    "base_ev": cache_item.get("base_ev", 0.0),
                    "is_raw": cache_item.get("is_raw", False),
                    "film_profile": film_profile,
                    "paper_profile": paper_profile,
                    "print_mode": print_mode,
                    "params": params,
                    "is_dirty": is_dirty,
                    "is_edited": is_edited,
                }
            else:
                dw, dh = session_cache_manager.get_image_dimensions(p)
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
                    "auto_ev": 0.0,
                    "base_ev": 0.0,
                    "is_raw": False,
                    "film_profile": film_profile,
                    "paper_profile": paper_profile,
                    "print_mode": print_mode,
                    "params": params,
                    "is_dirty": is_dirty,
                    "is_edited": is_edited,
                }
                uncached_slots.append((idx, p))

            self.photos[idx] = photo_entry

        # 1. Switch to darkroom workspace instantly
        self.stack.setCurrentIndex(1)
        self.btn_title_export.setVisible(True)
        self.canvas.set_has_library_photos(bool(self.photos))

        # 2. Populate filmstrip with all entries in exact order
        target_entry = next((p for p in self.photos if p and p.get("path") == target_path), self.photos[0])
        target_id = target_entry["id"] if target_entry else None
        self.filmstrip.set_photos(self.photos, target_id)

        # 3. Mark dirty/edited badges
        for entry in self.photos:
            if entry and (entry.get("is_dirty") or entry.get("is_edited")):
                self.filmstrip.set_photo_dirty(entry["id"], True)

        # 4. Activate target photo immediately
        self.active_photo_id = None
        if uncached_slots:
            self._is_importing = True
            if hasattr(self, "btn_cancel_import"):
                self.btn_cancel_import.setVisible(True)
            self.set_app_status(f"正在恢复底片库 (0/{len(uncached_slots)} 张)...", spinning=True, timeout_ms=0)
        else:
            self._is_importing = False
            if hasattr(self, "btn_cancel_import"):
                self.btn_cancel_import.setVisible(False)
            self.set_app_status(f"底片库已就绪 ({total_files} 张)", spinning=False, timeout_ms=3000)

        # 5. Decode any uncached slots in background without changing list length or order
        if uncached_slots:
            self._slot_decode_worker = SlotDecodeWorker(uncached_slots, self.engine, parent=self)
            self._slot_decode_worker.slotDecoded.connect(self._on_slot_photo_decoded)
            self._slot_decode_worker.progress.connect(self._on_slot_decode_progress)
            self._slot_decode_worker.finished.connect(self._on_all_slots_decoded)
            self._slot_decode_worker.start()

        if target_id:
            try:
                self.on_switch_photo(target_id)
            except Exception as e:
                logger.error(f"Error activating initial photo on session restore: {e}")

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
            from version import get_app_title
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

                # 1. WM_NCCALCSIZE (0x0083): Remove standard title bar and frame while keeping DWM animations
                if msg.message == 0x0083:
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
            if is_active:
                self.dropzone.setStyleSheet("""
                    QFrame#welcomeDropzone {
                        border: 2px dashed #f59e0b;
                        border-radius: 14px;
                        background: rgba(245, 158, 11, 0.08);
                    }
                    QLabel { border: none; background: transparent; }
                """)
            else:
                self.dropzone.setStyleSheet("""
                    QFrame#welcomeDropzone {
                        border: 1.5px dashed #353846;
                        border-radius: 14px;
                        background: rgba(255, 255, 255, 0.015);
                    }
                    QFrame#welcomeDropzone:hover {
                        border-color: #f59e0b;
                        background: rgba(245, 158, 11, 0.03);
                    }
                    QLabel { border: none; background: transparent; }
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

        from path_utils import normalize_path
        valid_exts = {
            ".arw", ".cr2", ".cr3", ".nef", ".raf", ".dng",
            ".tif", ".tiff", ".png"
        }
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
        radius = "0px" if is_max else "8px"
        border = "none" if is_max else "1px solid #282a34"
        tb_radius = "0px" if is_max else "8px"
        cw = self.centralWidget()
        if cw:
            if cw.layout():
                if is_max:
                    cw.layout().setContentsMargins(8, 8, 8, 8)
                else:
                    cw.layout().setContentsMargins(0, 0, 0, 0)
            cw.setStyleSheet(f"""
                QWidget#rootContainer {{
                    background: #121316;
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
                    background: #353846;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: #f59e0b;
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
                    background: #353846;
                    min-width: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:horizontal:hover {{
                    background: #f59e0b;
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
            self.title_bar.setStyleSheet(f"background: #141518; border: none; border-top-left-radius: {tb_radius}; border-top-right-radius: {tb_radius};")

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
        self.lbl_brand = QLabel(get_app_title())
        self.lbl_brand.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.lbl_brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_brand.setStyleSheet("color: #f59e0b; font-size: 13px; font-weight: bold; letter-spacing: 0.5px; border: none;")
        tb_layout.addWidget(self.lbl_brand, 0, Qt.AlignmentFlag.AlignVCenter)

        tb_layout.addStretch(1)

        # Amber Export Button (Item 3)
        self.btn_title_export = QPushButton("冲印")
        self.btn_title_export.setFixedHeight(26)
        self.btn_title_export.setCursor(Qt.CursorShape.PointingHandCursor)
        save_icon = os.path.join(get_resource_dir(), "icons", "export_dark.png")
        if os.path.exists(save_icon):
            self.btn_title_export.setIcon(QIcon(save_icon))
            self.btn_title_export.setIconSize(QSize(13, 13))
        self.btn_title_export.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #111113;
                font-weight: bold;
                font-size: 11.5px;
                border: none;
                border-radius: 4px;
                padding: 3px 14px;
                margin-right: 6px;
            }
            QPushButton:hover {
                background: #d97706;
            }
            QPushButton:pressed {
                background: #b45309;
            }
        """)
        self.btn_title_export.clicked.connect(self.action_export_image)
        self.btn_title_export.setVisible(False)
        tb_layout.addWidget(self.btn_title_export, 0, Qt.AlignmentFlag.AlignVCenter)

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
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, 1)

        self._init_welcome_view()
        self._init_editor_view()

    def _init_welcome_view(self):
        welcome_widget = QWidget()
        welcome_widget.setStyleSheet("background: #111215;")
        v_layout = QVBoxLayout(welcome_widget)
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
        self.dropzone.setStyleSheet("""
            QFrame#welcomeDropzone {
                border: 1.5px dashed #353846;
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.015);
            }
            QFrame#welcomeDropzone:hover {
                border-color: #f59e0b;
                background: rgba(245, 158, 11, 0.03);
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)
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

        lbl_drop = QLabel("点击或拖入照片 / RAW 负片文件")
        lbl_drop.setStyleSheet("color: #f1f5f9; font-size: 16px; font-weight: 600;")
        lbl_drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(lbl_drop)

        lbl_sub = QLabel("支持格式: ARW · CR2 · CR3 · NEF · RAF · DNG · TIFF · PNG")
        lbl_sub.setStyleSheet("color: #71717a; font-size: 12px;")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(lbl_sub)

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

        btn_import_folder = QPushButton("导入文件夹")
        btn_import_folder.setFixedSize(160, 38)
        btn_import_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import_folder.setStyleSheet("""
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
        btn_import_folder.clicked.connect(self.action_open_folder)
        btn_layout.addWidget(btn_import_folder)

        c_layout.addLayout(btn_layout)
        v_layout.addWidget(c_widget)
        self.stack.addWidget(welcome_widget)

    def _init_editor_view(self):
        editor_widget = QWidget()
        editor_widget.setStyleSheet("background: #121316;")
        e_layout = QVBoxLayout(editor_widget)
        
        # Item 2: Margins and Padding around work area
        e_layout.setContentsMargins(6, 6, 6, 6)
        e_layout.setSpacing(6)

        # Item 2 & 11: 3-Column Splitter with strict limits
        self.main_splitter = FadingSplitter(Qt.Orientation.Horizontal)

        # -------------------------------------------------------------
        # 1. Left Sidebar: 3-Tier Resizable Splitter (Films, Papers, EXIF)
        # -------------------------------------------------------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(220)
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

        f_header = QHBoxLayout()
        f_header.setContentsMargins(4, 2, 4, 0)
        self.lbl_film_title = QLabel("胶卷型号 (Films)")
        self.lbl_film_title.setStyleSheet("color: #f59e0b; font-size: 11.5px; font-weight: bold;")
        f_header.addWidget(self.lbl_film_title)
        f_header.addStretch()

        btn_import_film = QPushButton("＋导入")
        btn_import_film.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import_film.setToolTip("导入外部胶卷物理配置 (.json)")
        btn_import_film.setStyleSheet("""
            QPushButton {
                background: #1a1b22;
                color: #94a3b8;
                border: 1px solid #2b2e3c;
                border-radius: 3px;
                padding: 1px 6px;
                font-size: 10.5px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #252836;
                color: #f59e0b;
                border-color: #f59e0b;
            }
            QPushButton:pressed {
                background: #121317;
                color: #d97706;
                border-color: #d97706;
            }
        """)
        btn_import_film.clicked.connect(self._import_film_lut)
        f_header.addWidget(btn_import_film)
        film_layout.addLayout(f_header)

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

        self.film_scroll.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.film_scroll.customContextMenuRequested.connect(self._on_film_blank_context_menu)
        self.film_container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.film_container.customContextMenuRequested.connect(self._on_film_blank_context_menu)

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

        btn_import_paper = QPushButton("＋导入")
        btn_import_paper.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import_paper.setToolTip("导入外部相纸物理配置 (.json)")
        btn_import_paper.setStyleSheet("""
            QPushButton {
                background: #1a1b22;
                color: #94a3b8;
                border: 1px solid #2b2e3c;
                border-radius: 3px;
                padding: 1px 6px;
                font-size: 10.5px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #252836;
                color: #f59e0b;
                border-color: #f59e0b;
            }
            QPushButton:pressed {
                background: #121317;
                color: #d97706;
                border-color: #d97706;
            }
        """)
        btn_import_paper.clicked.connect(self._import_paper_lut)
        p_header.addWidget(btn_import_paper)
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
        self.paper_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.paper_scroll.setWidget(self.paper_container)

        self.paper_scroll.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.paper_scroll.customContextMenuRequested.connect(self._on_paper_blank_context_menu)
        self.paper_container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.paper_container.customContextMenuRequested.connect(self._on_paper_blank_context_menu)

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
        # View Toolbar (Status on Left, Buttons on Right - Narrow 26px, clean borderless background)
        view_toolbar = QFrame()
        view_toolbar.setFixedHeight(26)
        view_toolbar.setStyleSheet("background: #141518; border: none;")
        tb_layout = QHBoxLayout(view_toolbar)
        tb_layout.setContentsMargins(10, 0, 10, 0)
        tb_layout.setSpacing(6)

        # Permanent photo filename & zoom percentage display (never erased or overwritten by transient notifications)
        self.lbl_photo_info = QLabel("")
        self.lbl_photo_info.setFixedWidth(160)
        self.lbl_photo_info.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 500;")
        tb_layout.addWidget(self.lbl_photo_info)

        # Subtle separator between permanent photo info and transient system notification
        self.status_sep = QLabel("│")
        self.status_sep.setFixedWidth(10)
        self.status_sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_sep.setStyleSheet("color: transparent; font-size: 11px;")
        tb_layout.addWidget(self.status_sep)

        # Left: Spinner & Status Label (Item 6 & 8)
        self.spinner = SpinnerWidget(size=13)
        tb_layout.addWidget(self.spinner)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
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
        self.btn_cancel_import.clicked.connect(self._cancel_active_import)
        tb_layout.addWidget(self.btn_cancel_import)

        tb_layout.addStretch(1)

        # Right: Dedicated Export Status Prompt (Pure text + underline, matching left status bar)
        self.export_status_container = QWidget()
        self.export_status_container.setObjectName("export_status_container")
        self.export_status_container.setVisible(False)
        self.export_status_container.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_status_container.setToolTip("点击查看冲印队列详情 (Ctrl+J)")
        self.export_status_container.mousePressEvent = self._on_export_status_clicked
        export_layout = QHBoxLayout(self.export_status_container)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(6)

        self.export_spinner = SpinnerWidget(size=12)
        export_layout.addWidget(self.export_spinner)

        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setObjectName("lbl_export_status")
        self.lbl_export_status.setStyleSheet("color: #f59e0b; font-size: 11px; text-decoration: underline;")
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
        self.export_sep.setFixedWidth(12)
        self.export_sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.export_sep.setStyleSheet("color: #2e3444; font-size: 11px;")
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

        self.btn_split = CompareButton(self)
        self.btn_split.setFixedSize(28, 24)
        self.btn_split.setIcon(get_icon("compare.png"))
        self.btn_split.setIconSize(QSize(14, 14))
        self.btn_split.setCheckable(True)
        self.btn_split.setToolTip("对比视图 (快捷键 \\): 点击切换分屏对比，长按查看原片")
        self.btn_split.setStyleSheet(btn_style)
        self.btn_split.clicked.connect(self.toggle_split_view)
        self.btn_split.longPressStarted.connect(self._on_compare_long_press_start)
        self.btn_split.longPressFinished.connect(self._on_compare_long_press_end)
        tb_layout.addWidget(self.btn_split)

        btn_fit = QPushButton()
        btn_fit.setFixedSize(28, 24)
        btn_fit.setIcon(get_icon("fit_window.png"))
        btn_fit.setIconSize(QSize(14, 14))
        btn_fit.setToolTip("适应图像全貌 (Ctrl+0)")
        btn_fit.setStyleSheet(btn_style)
        btn_fit.clicked.connect(self.canvas.fit_to_view)
        tb_layout.addWidget(btn_fit)

        btn_100 = QPushButton()
        btn_100.setFixedSize(28, 24)
        btn_100.setIcon(get_icon("zoom_100.png"))
        btn_100.setIconSize(QSize(14, 14))
        btn_100.setToolTip("1:1 实际像素 (Ctrl+1)")
        btn_100.setStyleSheet(btn_style)
        btn_100.clicked.connect(self.canvas.set_zoom_100)
        tb_layout.addWidget(btn_100)

        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        cc_layout.addWidget(view_toolbar)
        self.center_splitter.addWidget(canvas_container)

        # Bottom 35mm Filmstrip (Item 8, 16)
        self.filmstrip = FilmstripWidget()
        self.filmstrip.photoSelected.connect(self.on_switch_photo)
        self.filmstrip.photoRemoved.connect(self.on_remove_photo)
        self.filmstrip.photosRemoved.connect(self.on_remove_multiple_photos)
        self.filmstrip.clearRequested.connect(self.action_clear_photos)
        self.filmstrip.clearSdcRequested.connect(self.on_clear_sdc_config)
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
        self.center_splitter.setSizes([600, 106])

        self.main_splitter.addWidget(self.center_splitter)

        # -------------------------------------------------------------
        # 3. Right Sidebar: Pinned Histogram + Adjustments Scroll
        # -------------------------------------------------------------
        sidebar_container = QWidget()
        sidebar_container.setMinimumWidth(260)
        sidebar_container.setMaximumWidth(420)
        sidebar_container.setStyleSheet("background: #141518;")
        sb_outer_layout = QVBoxLayout(sidebar_container)
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
        sb_outer_layout.addWidget(sidebar_scroll, 1)
        self.main_splitter.addWidget(sidebar_container)
        self.main_splitter.setCollapsible(2, False)

        # Splitter proportion defaults: Left panel 320px (comfortably fits 3 columns), Center 800px, Right 320px
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([320, 800, 320])
        self.main_splitter.splitterMoved.connect(lambda pos, idx: self.reflow_grids())

        e_layout.addWidget(self.main_splitter)
        self.stack.addWidget(editor_widget)
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
        # Dynamic responsive layout: compact card dimensions to accommodate more stocks
        min_card_w = 90
        spacing = 6
        margins = 8

        # Helper function for grid reflow with dynamic card resizing
        def _reflow_container(scroll_widget, cards_dict, grid_layout):
            if not scroll_widget or not cards_dict:
                return
            viewport_w = scroll_widget.viewport().width()
            if viewport_w <= 0 and hasattr(self, "left_splitter"):
                viewport_w = self.left_splitter.width() - 16
            avail_w = max(90, viewport_w - margins)

            # Determine number of columns (accommodates 3-4 columns in typical sidebar widths)
            if avail_w >= 378:
                cols = 4
            elif avail_w >= 282:
                cols = 3
            elif avail_w >= 186:
                cols = 2
            else:
                cols = 1

            # Dynamically scale card width to fill row up to 135px max
            target_card_w = int((avail_w - (cols - 1) * spacing) / cols)
            target_card_w = max(min_card_w, min(135, target_card_w))

            for card in cards_dict.values():
                card.update_card_dimensions(target_card_w)

            for i, (sid, card) in enumerate(cards_dict.items()):
                grid_layout.removeWidget(card)
                grid_layout.addWidget(card, i // cols, i % cols)
            total_rows = (len(cards_dict) + cols - 1) // cols
            for r in range(total_rows):
                grid_layout.setRowStretch(r, 0)
            grid_layout.setRowStretch(total_rows, 1)

        if hasattr(self, "film_scroll") and self.film_cards:
            _reflow_container(self.film_scroll, self.film_cards, self.film_grid)

        if hasattr(self, "paper_scroll") and self.paper_cards:
            _reflow_container(self.paper_scroll, self.paper_cards, self.paper_grid)

    def _populate_film_grid(self):
        stocks = self.engine.get_film_stocks()
        for group in stocks:
            for item in group.get("stocks", group.get("items", [])):
                is_custom = bool(item.get("is_custom", False) or (hasattr(self.engine, 'BUILTIN_FILM_IDS') and item["id"] not in self.engine.BUILTIN_FILM_IDS))
                card = FilmGridCard(item, is_active=(item["id"] == self.current_film_stock), is_custom=is_custom, stock_type="film")
                card.selected.connect(self.on_switch_film)
                card.hoverEntered.connect(self._on_card_hover_entered)
                card.exportRequested.connect(lambda sid: self._export_film_lut(sid))
                card.removeRequested.connect(lambda sid: self._remove_film_lut(sid))
                card.sortRequested.connect(lambda st, sb: self._sort_and_reflow_stocks(st, sb))
                card.importRequested.connect(lambda st: self._import_film_lut())
                self.film_cards[item["id"]] = card
        if hasattr(self, 'lbl_film_title'):
            self.lbl_film_title.setText(f"胶卷型号 ({len(self.film_cards)} 卷)")
        self.reflow_grids()

    def _populate_paper_grid(self):
        papers = self.engine.get_paper_stocks()
        for item in papers:
            is_custom = bool(item.get("is_custom", False) or (hasattr(self.engine, 'BUILTIN_PAPER_IDS') and item["id"] not in self.engine.BUILTIN_PAPER_IDS))
            card = FilmGridCard(item, is_active=(item["id"] == self.current_paper_stock), is_custom=is_custom, stock_type="paper")
            card.selected.connect(self.on_switch_paper)
            card.hoverEntered.connect(self._on_card_hover_entered)
            card.exportRequested.connect(lambda sid: self._export_paper_lut(sid))
            card.removeRequested.connect(lambda sid: self._remove_paper_lut(sid))
            card.sortRequested.connect(lambda st, sb: self._sort_and_reflow_stocks(st, sb))
            card.importRequested.connect(lambda st: self._import_paper_lut())
            self.paper_cards[item["id"]] = card
        if hasattr(self, 'lbl_paper_title'):
            self.lbl_paper_title.setText(f"放大相纸 ({len(self.paper_cards)} 张)")
        self.reflow_grids()

    def _on_card_hover_entered(self, stock_type, stock_id):
        """Item 3: 1s hover activates temporary preview; subsequent hovers preview immediately."""
        if not self._current_photo_path:
            return

        if not getattr(self, "_hover_preview_active", False):
            current_sel = self.current_film_stock if stock_type == "film" else self.current_paper_stock
            if stock_id == current_sel and getattr(self, "_saved_film_stock", None) is None:
                if hasattr(self, "_hover_timer") and self._hover_timer.isActive():
                    self._hover_timer.stop()
                self._hover_pending_stock = None
                return
            if getattr(self, "_saved_film_stock", None) is None:
                self._saved_film_stock = self.current_film_stock
                self._saved_paper_stock = self.current_paper_stock
                self._saved_print_mode = self.current_print_mode
            self._hover_pending_stock = (stock_type, stock_id)
            if hasattr(self, "_hover_timer"):
                self._hover_timer.stop()
                self._hover_timer.start(1000)
        else:
            self._apply_temp_preview(stock_type, stock_id)

    def _on_hover_timer_timeout(self):
        if getattr(self, "_hover_pending_stock", None):
            self._hover_preview_active = True
            st, sid = self._hover_pending_stock
            self._apply_temp_preview(st, sid)

    def _apply_temp_preview(self, stock_type, stock_id):
        if not self._current_photo_path:
            return
        if getattr(self, "_saved_film_stock", None) is None:
            self._saved_film_stock = self.current_film_stock
            self._saved_paper_stock = self.current_paper_stock
            self._saved_print_mode = self.current_print_mode

        if stock_type == "film":
            temp_film = stock_id
            temp_paper = self._saved_paper_stock
            temp_mode = self._saved_print_mode
            if stock_id == "none":
                temp_paper = "none"
                temp_mode = "scan"
        else:
            temp_film = self._saved_film_stock
            temp_paper = stock_id
            temp_mode = "scan" if stock_id == "none" else "optical"

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = temp_mode
        lut_3d = self.engine.get_3d_lut(temp_film, temp_paper, lut_size=33, params_dict=lut_params)
        self.canvas.set_lut(lut_3d)
        self._schedule_histogram_update()

        for sid, card in self.film_cards.items():
            card.set_active(sid == temp_film)
        for pid, card in self.paper_cards.items():
            card.set_active(pid == temp_paper)



    def _revert_temp_preview(self):
        if getattr(self, "_saved_film_stock", None) is not None:
            saved_film = self._saved_film_stock
            saved_paper = self._saved_paper_stock
            saved_mode = self._saved_print_mode
            self._saved_film_stock = None
            self._saved_paper_stock = None
            self._saved_print_mode = None

            lut_params = dict(self.current_params)
            lut_params["print_mode"] = saved_mode
            lut_3d = self.engine.get_3d_lut(saved_film, saved_paper, lut_size=33, params_dict=lut_params)
            self.canvas.set_lut(lut_3d)
            self._schedule_histogram_update()

            for sid, card in self.film_cards.items():
                card.set_active(sid == saved_film)
            for pid, card in self.paper_cards.items():
                card.set_active(pid == saved_paper)

    def _on_film_paper_pane_left(self):
        """Called when mouse leaves the left film & paper cards pane."""
        if hasattr(self, "_hover_timer") and self._hover_timer.isActive():
            self._hover_timer.stop()
        self._hover_pending_stock = None
        self._hover_preview_active = False
        self._revert_temp_preview()

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
        self.sec_optics = AccordionSection("镜头光学与柔光")
        self.sec_optics.resetClicked.connect(lambda: self.reset_track("optics"))

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

        self.sidebar_layout.addWidget(self.sec_grain)

    _init_texture_section = _init_halation_section
    _init_optical_section = _init_halation_section

    def _init_menu_and_actions(self):
        # 1. File Menu
        file_menu = self.menu_bar.addMenu("文件 (F)")
        
        act_open_file = QAction("打开文件...", self)
        act_open_file.setIcon(get_icon("open_file.png"))
        act_open_file.setShortcut(QKeySequence("Ctrl+O"))
        act_open_file.triggered.connect(self.action_open_files)
        file_menu.addAction(act_open_file)

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
        self.act_quick_export = QAction("快速冲印 (上次参数)", self)
        self.act_quick_export.setIcon(get_icon("quick_export.png"))
        self.act_quick_export.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.act_quick_export.triggered.connect(lambda: self.action_quick_export())
        self.act_quick_export.setEnabled(False)
        file_menu.addAction(self.act_quick_export)

        self.act_export = QAction("冲印...", self)
        self.act_export.setIcon(get_icon("export.png"))
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.triggered.connect(self.action_export_image)
        self.act_export.setEnabled(False)
        file_menu.addAction(self.act_export)

        act_queue = QAction("冲印队列管理...", self)
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
        
        act_undo = QAction("撤销", self)
        act_undo.setIcon(get_icon("undo.png"))
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self.undo)
        edit_menu.addAction(act_undo)

        act_redo = QAction("重做", self)
        act_redo.setIcon(get_icon("redo.png"))
        act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        act_redo.triggered.connect(self.redo)
        edit_menu.addAction(act_redo)

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

        act_reset_all = QAction("重置所有参数", self)
        act_reset_all.setIcon(get_icon("reset.png"))
        act_reset_all.setShortcut(QKeySequence("Ctrl+R"))
        act_reset_all.triggered.connect(self.reset_all_params)
        edit_menu.addAction(act_reset_all)

        edit_menu.addSeparator()

        act_prefs = QAction("首选项...", self)
        act_prefs.setIcon(get_icon("preferences.png"))
        act_prefs.setShortcut(QKeySequence("Ctrl+,"))
        act_prefs.triggered.connect(self.open_preferences)
        edit_menu.addAction(act_prefs)

        # 3. View Menu
        view_menu = self.menu_bar.addMenu("视图 (V)")
        
        act_fit_img = QAction("适应图像", self)
        act_fit_img.setIcon(get_icon("fit_window.png"))
        act_fit_img.setShortcut(QKeySequence("Ctrl+0"))
        act_fit_img.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(act_fit_img)

        act_compare = QAction("对比\t\\", self)
        act_compare.setIcon(get_icon("compare.png"))
        act_compare.triggered.connect(self.toggle_split_view)
        view_menu.addAction(act_compare)

        act_fit_win = QAction("适应窗口", self)
        act_fit_win.setIcon(get_icon("fit_window.png"))
        act_fit_win.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(act_fit_win)

        act_100 = QAction("1:1 实际像素", self)
        act_100.setIcon(get_icon("zoom_100.png"))
        act_100.setShortcut(QKeySequence("Ctrl+1"))
        act_100.triggered.connect(self.canvas.set_zoom_100)
        view_menu.addAction(act_100)

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
            self.center_splitter.setSizes([600, 106])
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
        from path_utils import normalize_path
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
        has_session = not is_startup and bool(
            (self._current_session_path and not getattr(self, "_is_temp_session", False)) or
            self.photos or
            getattr(self, "_is_temp_session", False)
        )
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

        if not self.photos and is_startup:
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
        """Handle global backslash key and mouse exit detection for film/paper hover preview."""
        et = event.type()

        # Film & paper pane hover exit detection on real mouse movement
        if et == QEvent.Type.MouseMove and self.isVisible() and hasattr(self, "left_splitter"):
            if getattr(self, "_hover_preview_active", False) or (hasattr(self, "_hover_timer") and self._hover_timer.isActive()):
                from PySide6.QtGui import QCursor
                pos_in_splitter = self.left_splitter.mapFromGlobal(QCursor.pos())
                if not self.left_splitter.rect().contains(pos_in_splitter):
                    self._on_film_paper_pane_left()

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
            self._on_film_paper_pane_left()
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
                "正在导入", "导入底片", "正在恢复", "底片库", "会话底片库", "已导入", "已取消"
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
                self.status_sep.setStyleSheet("color: #333644; font-size: 11px;")
            else:
                self.status_sep.setStyleSheet("color: transparent; font-size: 11px;")

        self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
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
            self.status_sep.setStyleSheet("color: transparent; font-size: 11px;")

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
            self.lbl_export_status.setStyleSheet("color: #f59e0b; font-size: 11px; text-decoration: underline; background: transparent; border: none;")
            self.export_status_container.setToolTip("正在后台冲印，点击查看冲印队列 (Ctrl+J)")
        elif state == "success":
            self.export_spinner.stop()
            self.lbl_export_status.setStyleSheet("color: #f59e0b; font-size: 11px; text-decoration: underline; background: transparent; border: none;")
            self.export_status_container.setToolTip("冲印完成，点击查看冲印队列 (Ctrl+J)")
            # Permanent display on success as requested by user
            timeout_ms = 0
        elif state == "error":
            self.export_spinner.stop()
            self.lbl_export_status.setStyleSheet("color: #ef4444; font-size: 11px; text-decoration: underline; background: transparent; border: none;")
            self.export_status_container.setToolTip("冲印发生错误，点击查看冲印队列 (Ctrl+J)")
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
            self._set_export_status(f"冲印中 ({done_count + 1}/{total_count})", state="busy")
        else:
            if total_count > 0:
                self._set_export_status(f"冲印完成 ({total_count}/{total_count})", state="success", timeout_ms=0)
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
            from path_utils import normalize_path
            file_path = normalize_path(file_path)
            self._current_session_path = file_path
            self._is_temp_session = False
            return self._do_save_session(file_path)
        return False

    def _save_auto_temp_session(self):
        """Item 2: Auto-save temporary session state to auto_saved_session.sdss so work is preserved across restarts."""
        if not self.photos:
            return
        if self._current_session_path and not getattr(self, "_is_temp_session", False):
            return
        try:
            from path_utils import normalize_path
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

            from version import VERSION_STRING
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

        from path_utils import normalize_path
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
                "filename": os.path.basename(p["path"])
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
        if not self.photos and getattr(self, 'stack', None) and self.stack.currentIndex() == 0:
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
        self._cancel_active_import()

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
        self.stack.setCurrentIndex(0)
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
            from path_utils import normalize_path
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

        from path_utils import normalize_path
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
        self._cancel_active_import()

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
        if active_path:
            if not os.path.exists(active_path):
                active_fn = os.path.basename(active_path).lower()
                for rf in resolved_files:
                    if os.path.basename(rf).lower() == active_fn:
                        active_path = rf
                        break
        if not active_path or not os.path.exists(active_path):
            active_path = resolved_files[0] if resolved_files else None

        if resolved_files:
            self._restore_saved_session(resolved_files, active_file=active_path)
        else:
            self.stack.setCurrentIndex(1)
            self.canvas.set_has_library_photos(False)
            self.canvas.set_image(None)
            self.filmstrip.set_photos([], None)
            self.btn_title_export.setVisible(False)
            self.thumb_controller.reset_all_cards_to_color_pills(self.film_cards, self.paper_cards)

        if not is_temp:
            config_manager.add_recent_session(file_path)
            config_manager.save_last_session_path(file_path)
            self._update_recent_menu()

        # If any path was relinked/changed, mark session dirty so user can re-save it with new paths
        self._session_is_dirty = path_updated
        self._update_window_title()

        fn = "未命名会话" if is_temp else os.path.basename(file_path)
        if not getattr(self, "_is_importing", False):
            if path_updated:
                self.set_app_status(f"会话已恢复 (底片路径已更新): {fn}", timeout_ms=3000)
            else:
                self.set_app_status(f"会话已恢复: {fn}", timeout_ms=2500)

    def action_quick_export(self, target_ids=None):
        """Item 4: Quick export selected or active photo(s) using remembered last export settings."""
        last_cfg = config_manager.get_last_export_settings()
        if not last_cfg:
            self.set_app_status("未记录上次冲印参数，请先执行一次常规冲印", timeout_ms=3000)
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
        out_dir = last_cfg.get("path", "")
        if not out_dir or not os.path.isdir(out_dir):
            if self._current_photo_path:
                out_dir = os.path.dirname(self._current_photo_path)
            else:
                out_dir = os.path.expanduser("~")

        enqueued_count = 0
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
            self.export_queue.enqueue(task)
            enqueued_count += 1

        self._set_export_status(f"已加入冲印队列 ({enqueued_count} 张)", state="busy")

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
        else:
            self.set_app_status("已撤销", timeout_ms=1500)

    def redo(self):
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

        # Ignore internal non-color keys
        ignored_keys = {"print_mode", "custom_wb", "film_format_mm", "base_temp", "base_tint"}
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

            self.canvas.update_params(**self.current_params)
            lut_params = dict(self.current_params)
            lut_params["print_mode"] = self.current_print_mode
            lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
            self.canvas.set_lut(lut_3d)

            self._schedule_histogram_update()
            self._update_diffusion_ui_state()
            self._update_stock_mode_ui()
            self._update_active_thumbnails()
            self._refresh_card_previews()
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

        self._update_all_sliders_from_params()
        self.on_switch_print_mode(self.current_print_mode, push_undo=False)
        self.on_switch_film(self.current_film_stock, push_undo=False)
        if self.current_print_mode == "optical":
            self.on_switch_paper(self.current_paper_stock, push_undo=False)
        else:
            for pid, card in self.paper_cards.items():
                card.set_active(pid == "none")

        self.canvas.update_params(**self.current_params)
        lut_params = dict(self.current_params)
        lut_params["print_mode"] = self.current_print_mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self.canvas.set_lut(lut_3d)

        self._schedule_histogram_update()
        self._update_diffusion_ui_state()
        self._update_stock_mode_ui()
        self._update_active_thumbnails()
        self._refresh_card_previews()

        # Update current photo metadata
        if self._current_photo_path:
            curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
            if curr_p:
                curr_p["film_profile"] = self.current_film_stock
                curr_p["paper_profile"] = self.current_paper_stock
                curr_p["print_mode"] = self.current_print_mode
                curr_p["params"] = dict(self.current_params)
                is_reverted_to_raw = (self.current_film_stock == "none" and self.current_paper_stock == "none" and not self._is_params_modified(self.current_params, curr_p))
                curr_p["is_dirty"] = not is_reverted_to_raw
                curr_p["is_edited"] = not is_reverted_to_raw
                self._photo_is_dirty = not is_reverted_to_raw
                self.filmstrip.set_photo_dirty(self.active_photo_id, not is_reverted_to_raw)
            self._save_current_photo_sdc()

    def _apply_batch_history_state(self, state: dict):
        """Restores a batch operation snapshot across multiple photos (e.g. Select All -> Delete SDC or Paste)."""
        if not state or not isinstance(state, dict):
            return

        photos_data = state.get("photos", {})
        if not photos_data:
            return

        active_restored = False
        for pid, pdata in photos_data.items():
            p = next((x for x in self.photos if x.get("id") == pid), None)
            if not p:
                continue

            film = pdata.get("film", "none")
            paper = pdata.get("paper", "none")
            print_mode = pdata.get("print_mode", "optical")
            params = dict(pdata.get("params") or {})
            is_dirty = bool(pdata.get("is_dirty", False))
            is_edited = bool(pdata.get("is_edited", False))
            p_path = p.get("path")

            p["film_profile"] = film
            p["paper_profile"] = paper
            p["print_mode"] = print_mode
            p["params"] = dict(params) if (is_edited or film not in ("none", "", None) or paper not in ("none", "", None) or self._is_params_modified(params, p)) else None
            p["is_dirty"] = is_dirty
            p["is_edited"] = is_edited

            # 1. Disk SDC persistence
            if p_path:
                if is_edited or film not in ("none", "", None) or paper not in ("none", "", None) or self._is_params_modified(params, p):
                    try:
                        sdc_manager.save_sdc(
                            p_path,
                            film,
                            paper,
                            params,
                            thumbnail_rgb=p.get("thumbnail_rgb"),
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
                        graded = self.thumb_controller.grade_thumbnail_fast(raw_thumb, film, paper, params)
                        self.thumb_controller.update_graded_thumbnail(p, graded, self.filmstrip)
            else:
                if hasattr(self, 'thumb_controller'):
                    self.thumb_controller.revert_photo_to_raw(p, self.filmstrip)

            if hasattr(self, 'filmstrip'):
                self.filmstrip.set_photo_dirty(pid, is_dirty)

            if pid == self.active_photo_id:
                active_restored = True

        self._session_is_dirty = True
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
            self.current_params = act_params
            self.current_params["print_mode"] = self.current_print_mode
            self._photo_is_dirty = bool(act_data.get("is_dirty", False))

            self._update_all_sliders_from_params()
            self.on_switch_print_mode(self.current_print_mode, push_undo=False)
            self.on_switch_film(self.current_film_stock, push_undo=False)
            if self.current_print_mode == "optical":
                self.on_switch_paper(self.current_paper_stock, push_undo=False)
            else:
                for pid_card, card in self.paper_cards.items():
                    card.set_active(pid_card == "none")

            self.canvas.update_params(**self.current_params)
            lut_params = dict(self.current_params)
            lut_params["print_mode"] = self.current_print_mode
            lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
            self.canvas.set_lut(lut_3d)

            self._schedule_histogram_update()
            self._update_diffusion_ui_state()
            self._update_stock_mode_ui()
            self._update_active_thumbnails()
            self._update_reset_buttons_state()
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
                graded = self.engine.apply_lut_to_linear(self._current_photo_linear, lut_3d)
                self.histogram_widget.update_from_rgb(graded)
            elif self._current_photo_thumb is not None:
                self.histogram_widget.update_from_rgb(self._current_photo_thumb)
        except Exception:
            pass

    def get_default_params(self, base_temp=5500.0, base_tint=1.0):
        return {
            "print_mode": "optical",
            "exposure_ev": 0.0,
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
                "grain": 0.4, "grain_size": 1.0, "grain_cloud_blur": 1.0
            },
            "texture": {
                "halation_amount": 1.0, "halation_spatial_scale": 1.0,
                "scatter_amount": 1.0, "boost_ev": 0.0,
                "halation": 1.0, "halation_boost": 0.0,
                "grain": 0.4, "grain_size": 1.0, "grain_cloud_blur": 1.0
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
            sec.set_reset_enabled(not is_default)

    def reset_track(self, track_name):
        if getattr(self, "current_film_stock", "none") == "none":
            return
        defaults = self._get_track_defaults()
        if track_name not in defaults:
            return
        changed = False
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
        finally:
            self._is_updating_sliders = False
        self._update_reset_buttons_state()
        self._update_diffusion_ui_state()

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
            for pid, card in self.paper_cards.items():
                card.set_active(pid == "none")

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
        hal_preset = self.engine.get_film_halation_preset(stock_id) if hasattr(self.engine, 'get_film_halation_preset') else None
        self.canvas.update_params(film_stock=stock_id, _halation_preset=hal_preset)
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
            self.set_app_status(f"冲印模式: {mode_name}", timeout_ms=2500)

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

    def _on_zoom_changed(self, zoom_val):
        self._update_photo_info_display()

    def action_open_files(self):
        filt = "图像文件 (*.arw *.cr2 *.cr3 *.nef *.raf *.dng *.tif *.tiff *.png);;所有文件 (*.*)"
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
            valid_exts = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".dng", ".tif", ".tiff", ".png"}
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
        existing_norm_paths = {os.path.normcase(os.path.abspath(p["path"])) for p in self.photos if p and "path" in p}
        new_paths = []
        dup_names = []
        seen_in_batch = set()

        for fp in file_paths:
            norm_p = os.path.normcase(os.path.abspath(fp))
            if norm_p in existing_norm_paths or norm_p in seen_in_batch:
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
        self.stack.setCurrentIndex(1)
        self.btn_title_export.setVisible(True)

        # 2. Add to recent files
        for p in new_paths:
            config_manager.add_recent_file(p)

        # 3. Start progress spinner & status text
        self._is_importing = True
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(True)
        self._last_import_status_time = 0.0
        total_count = len(new_paths)
        self.set_app_status(f"正在导入底片 (0/{total_count})...", spinning=True, timeout_ms=0)

        # 4. Asynchronous streaming import worker
        self._import_worker = BatchImportWorker(new_paths, self.engine, start_idx=len(self.photos), parent=self)
        self._import_worker.photoLoaded.connect(self._on_single_photo_imported)
        self._import_worker.importFinished.connect(self._on_all_photos_imported)
        self._import_worker.start()

    def _on_single_photo_imported(self, photo_entry, current_idx, total_count):
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
            from path_utils import normalize_path
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
                import sdc_manager
                sdc = sdc_manager.load_sdc(photo_entry.get("path", ""))
                if sdc:
                    photo_entry["film_profile"] = sdc.get("film_profile", "none")
                    photo_entry["paper_profile"] = sdc.get("paper_profile", "none")
                    photo_entry["print_mode"] = sdc.get("print_mode", sdc.get("params", {}).get("print_mode", "optical"))
                    photo_entry["params"] = dict(sdc.get("params") or {})
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
                "raw_preview": photo_entry.get("float_img"),
                "meta": dict(photo_entry.get("exif") or {}),
                "is_raw": photo_entry.get("is_raw", False),
                "base_ev": photo_entry.get("base_ev", 0.0),
                "auto_ev": photo_entry.get("auto_ev", 0.0)
            }
            if matched_ep:
                self.engine.session_photos[self.engine.session_photos.index(matched_ep)] = ep_dict
            else:
                self.engine.session_photos.append(ep_dict)

        # Status updates: strictly synchronized with actual imported count len(self.photos)
        imported_count = len(self.photos)
        import time
        now = time.time()
        last_t = getattr(self, "_last_import_status_time", 0.0)
        status_text = f"正在导入底片 ({imported_count}/{total_count})..."

        if (now - last_t >= 0.08) or imported_count == total_count or imported_count == 1:
            self._last_import_status_time = now
            self.set_app_status(status_text, spinning=True, timeout_ms=0)
        else:
            # Trailing timer to ensure the status bar ALWAYS catches up and never freezes behind
            if not hasattr(self, "_import_flush_timer") or self._import_flush_timer is None:
                self._import_flush_timer = QTimer(self)
                self._import_flush_timer.setSingleShot(True)
                self._import_flush_timer.timeout.connect(self._flush_import_status)
            self._pending_import_status = (imported_count, total_count)
            self._import_flush_timer.start(50)

    def _flush_import_status(self):
        if getattr(self, "_is_importing", False) and hasattr(self, "_pending_import_status"):
            cur = len(self.photos)
            _, tot = self._pending_import_status
            self.set_app_status(f"正在导入底片 ({cur}/{tot})...", spinning=True, timeout_ms=0)

    def _on_all_photos_imported(self, total_loaded):
        if hasattr(self, "_import_flush_timer") and self._import_flush_timer is not None:
            self._import_flush_timer.stop()
        self._is_importing = False
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
        self.set_app_status(f"底片库导入完成 ({total_loaded} 张)", spinning=False, timeout_ms=3000)

        # Real-time session state save & cache maintenance
        try:
            if self._current_session_path and os.path.exists(self._current_session_path):
                config_manager.save_last_session_path(self._current_session_path)
            elif getattr(self, "_is_temp_session", False) or not self._current_session_path:
                self._save_auto_temp_session()
                config_manager.save_last_session_path(config_manager.get_auto_saved_session_path())
            session_cache_manager.prune_cache_if_needed()
        except Exception:
            pass

    def _cancel_active_import(self):
        """Forcefully cancels active batch import worker or slot decode worker, freeing CPU and memory."""
        if getattr(self, "_import_worker", None) and self._import_worker.isRunning():
            try:
                self._import_worker.photoLoaded.disconnect()
                self._import_worker.importFinished.disconnect()
            except Exception:
                pass
            self._import_worker.stop()
            self._import_worker.wait(150)
            self._import_worker = None

        if getattr(self, "_slot_decode_worker", None) and self._slot_decode_worker.isRunning():
            try:
                self._slot_decode_worker.slotDecoded.disconnect()
                self._slot_decode_worker.progress.disconnect()
                self._slot_decode_worker.finished.disconnect()
            except Exception:
                pass
            self._slot_decode_worker.stop()
            self._slot_decode_worker.wait(150)
            self._slot_decode_worker = None

        self._is_importing = False
        if hasattr(self, "btn_cancel_import"):
            self.btn_cancel_import.setVisible(False)
        self.restore_default_status()
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
                    import session_cache_manager
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
        """Step previous (-1) or next (+1) photo in filmstrip."""
        if not self.photos or not self.active_photo_id:
            return
        idx = next((i for i, p in enumerate(self.photos) if p and p.get("id") == self.active_photo_id), -1)
        if idx == -1:
            return
        new_idx = idx + delta
        if 0 <= new_idx < len(self.photos):
            target = self.photos[new_idx]
            if target and target.get("id"):
                self.on_switch_photo(target["id"])
                self.filmstrip.scroll_to_active()

    def _on_navigate_first_photo(self):
        """Navigate to the first photo in filmstrip, activate and select it (Home key)."""
        if not self.photos:
            return
        first_photo = self.photos[0]
        if first_photo and first_photo.get("id"):
            target_id = first_photo["id"]
            if hasattr(self, "filmstrip"):
                self.filmstrip._selected_ids = {target_id}
                self.filmstrip.set_active_photo(target_id)
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
                self.filmstrip._selected_ids = {target_id}
                self.filmstrip.set_active_photo(target_id)
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

                # 1. Left active film and paper cards (WYSIWYG)
                self.thumb_controller.update_active_card_wysiwyg(
                    self.film_cards, self.paper_cards,
                    self.current_film_stock, self.current_paper_stock, pix
                )

                # 2. Bottom filmstrip active frame thumbnail
                if self.active_photo_id and is_valid:
                    curr_p = next((p for p in self.photos if p and p.get("id") == self.active_photo_id), None)
                    if curr_p:
                        if self.current_film_stock == "none":
                            self.thumb_controller.revert_photo_to_raw(curr_p, self.filmstrip)
                        else:
                            has_recipe = (self.current_film_stock != "none" or self.current_paper_stock != "none" or self._photo_is_dirty)
                            if has_recipe:
                                self.thumb_controller.update_graded_thumbnail(curr_p, rendered_rgb, self.filmstrip)
                            else:
                                self.filmstrip.update_photo_thumbnail(self.active_photo_id, pix)
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
            getattr(self, 'slider_preflash', None),
        ]
        for s in paper_sliders:
            if s is not None:
                s.setEnabled(not disable_paper_sliders)
                if disable_paper_sliders:
                    s.setToolTip("当前选用反转正片或数码扫描模式，相纸印相物理参数已自动禁用")
                else:
                    s.setToolTip(getattr(s, "default_tooltip", ""))

        enlarger_sliders = [
            getattr(self, 'slider_cyan', getattr(self, 'slider_enlarger_c', None)),
            getattr(self, 'slider_magenta', getattr(self, 'slider_enlarger_m', None)),
            getattr(self, 'slider_yellow', getattr(self, 'slider_enlarger_y', None)),
            getattr(self, 'slider_print_exp', None),
        ]
        for s in enlarger_sliders:
            if s is not None:
                s.setEnabled(not is_pos)
                if is_pos:
                    s.setToolTip("反转正片无暗房放大机滤镜环节")
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

    def on_switch_photo(self, photo_id):
        # Stop any lingering live/debounce timers from previous photo
        if hasattr(self, '_live_thumb_timer') and self._live_thumb_timer.isActive():
            self._live_thumb_timer.stop()
        if hasattr(self, '_cards_debounce_timer') and self._cards_debounce_timer.isActive():
            self._cards_debounce_timer.stop()

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
        try:
            # 1. Update in-memory state and ensure SDC is saved for previous photo
            if self.active_photo_id:
                prev_photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
                if prev_photo:
                    if getattr(self, "_photo_is_dirty", False) or prev_photo.get("is_edited", False):
                        self._save_current_photo_sdc()

            # 2. Find photo entry
            photo = next((p for p in self.photos if p["id"] == photo_id), None)
            if not photo:
                return

            self.active_photo_id = photo_id
            self._current_photo_path = photo["path"]
            self.filmstrip.set_active_photo(photo_id)
            self.canvas.set_has_library_photos(bool(self.photos))

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
            self._active_base_temp = base_temp
            self._active_base_tint = base_tint
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
                    self.current_params.clear()
                    self.current_params.update(base_dict)
                    self._photo_is_dirty = False
                else:
                    self.current_film_stock = "none"
                    self.current_paper_stock = "none"
                    self.current_print_mode = "scan"
                    defaults = self.get_default_params(base_temp, base_tint)
                    defaults["print_mode"] = "scan"
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
                            self._single_photo_loader.wait(300)

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

            # Memory LRU management: Keep only current and most recent photo float_img in RAM
            # to strictly prevent gigabyte memory bloat across dozens of RAW photos
            if not hasattr(self, "_loaded_photos_lru"):
                self._loaded_photos_lru = []
            if photo_id in self._loaded_photos_lru:
                self._loaded_photos_lru.remove(photo_id)
            self._loaded_photos_lru.append(photo_id)
            while len(self._loaded_photos_lru) > 2:
                old_pid = self._loaded_photos_lru.pop(0)
                if old_pid != photo_id:
                    old_p = next((p for p in self.photos if p.get("id") == old_pid), None)
                    if old_p:
                        old_p["float_img"] = None
            self.canvas.update_params(**self.current_params)
            self._update_all_sliders_from_params()

            # 6. Apply print mode, film & paper LUT
            self.on_switch_print_mode(self.current_print_mode, push_undo=False)
            self.on_switch_film(self.current_film_stock, push_undo=False)
            if self.current_print_mode == "optical":
                self.on_switch_paper(self.current_paper_stock, push_undo=False)
            else:
                self.on_switch_paper("none", push_undo=False)

            # 7. Update status text with 1:1 image native pixel scale percentage
            self._update_photo_info_display()

            # 8. Update live 2:3 thumbnails on Film & Paper cards with real film emulation
            self._current_photo_thumb = photo.get("thumbnail_rgb")
            float_img = photo.get("float_img")
            if float_img is not None:
                lh, lw = float_img.shape[:2]
                lscale = 120.0 / float(max(lh, lw))
                ltw = max(1, int(round(lw * lscale)))
                lth = max(1, int(round(lh * lscale)))
                self._current_photo_linear = cv2.resize(float_img, (ltw, lth), interpolation=cv2.INTER_AREA)
                self._update_card_live_thumbnails(self._current_photo_linear)
                self._update_active_thumbnails()
            else:
                self._current_photo_linear = None
                if hasattr(self, 'engine'):
                    self.engine.raw_preview = None
                # Immediate (0ms) active card update with existing thumbnail:
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

                # For candidate cards, strictly use the unedited RAW base negative thumbnail
                if raw_thumb is not None:
                    self._update_card_live_thumbnails(raw_thumb)

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
        finally:
            self._is_switching_photo = False

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

        try:
            session_cache_manager.save_photo_preview(photo_path, float_img)
        except Exception as e:
            logger.debug(f"save_photo_preview failed: {e}")

        target_photo = next((p for p in self.photos if p["id"] == photo_id), None)
        if target_photo:
            target_photo["float_img"] = float_img
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

        # Only update GUI if this is still the active photo
        if self.active_photo_id == photo_id:
            fn = target_photo.get("filename", os.path.basename(photo_path)) if target_photo else os.path.basename(photo_path)
            try:
                if res.get("exif"):
                    exif_meta = res["exif"]
                    self._active_base_temp = float(exif_meta.get("color_temp", 5500.0))
                    self._active_base_tint = float(exif_meta.get("tint", 1.0))
                    self.current_params["base_temp"] = self._active_base_temp
                    self.current_params["base_tint"] = self._active_base_tint
                    if not self.current_params.get("custom_wb", False):
                        self.current_params["color_temp"] = self._active_base_temp
                        self.current_params["tint"] = self._active_base_tint
                        if hasattr(self, "slider_temp"):
                            self.slider_temp.set_default_value(self._active_base_temp)
                            self._is_updating_sliders = True
                            try:
                                self.slider_temp.set_value(self._active_base_temp)
                            finally:
                                self._is_updating_sliders = False
                        if hasattr(self, "slider_tint"):
                            self.slider_tint.set_default_value(self._active_base_tint)
                            self._is_updating_sliders = True
                            try:
                                self.slider_tint.set_value(self._active_base_tint)
                            finally:
                                self._is_updating_sliders = False
                    self.canvas.update_params(**self.current_params)

                m_w = target_photo.get("width") or (target_photo.get("meta", {}).get("width") if isinstance(target_photo.get("meta"), dict) else None) if target_photo else None
                m_h = target_photo.get("height") or (target_photo.get("meta", {}).get("height") if isinstance(target_photo.get("meta"), dict) else None) if target_photo else None
                self.canvas.set_image(float_img, master_width=m_w, master_height=m_h, reset_zoom=False, photo_id=photo_id)

                # Update LRU queue
                if not hasattr(self, "_loaded_photos_lru"):
                    self._loaded_photos_lru = []
                if photo_id in self._loaded_photos_lru:
                    self._loaded_photos_lru.remove(photo_id)
                self._loaded_photos_lru.append(photo_id)
                while len(self._loaded_photos_lru) > 2:
                    old_pid = self._loaded_photos_lru.pop(0)
                    if old_pid != photo_id:
                        old_p = next((p for p in self.photos if p.get("id") == old_pid), None)
                        if old_p:
                            old_p["float_img"] = None

                # Update live 2:3 thumbnails on cards
                self._current_photo_thumb = target_photo.get("thumbnail_rgb") if target_photo else None
                lh, lw = float_img.shape[:2]
                lscale = 120.0 / float(max(lh, lw))
                ltw = max(1, int(round(lw * lscale)))
                lth = max(1, int(round(lh * lscale)))
                self._current_photo_linear = cv2.resize(float_img, (ltw, lth), interpolation=cv2.INTER_AREA)

                self._update_card_live_thumbnails(self._current_photo_linear)

                self._schedule_histogram_update()
                self._update_active_thumbnails()
                self._update_photo_info_display()
            except Exception as e:
                logger.error(f"Error updating GUI for loaded photo {photo_id}: {e}")
            finally:
                if not getattr(self, "_is_multi_selecting", False):
                    self.set_app_status(f"底片已就绪: {fn}", spinning=False, timeout_ms=2000)

    def _update_card_live_thumbnails(self, thumb_data, do_films=True, do_papers=True):
        if thumb_data is None:
            return

        film_cards_dict = self.film_cards if do_films else {}
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
            self.filmstrip.set_photos(self.photos, next_id)
            self.on_switch_photo(next_id)
            if hasattr(self.filmstrip, 'scroll_to_active'):
                self.filmstrip.scroll_to_active()

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
            self.filmstrip.set_photos(self.photos, next_id)
            self.on_switch_photo(next_id)
            if hasattr(self.filmstrip, 'scroll_to_active'):
                self.filmstrip.scroll_to_active()

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
            p["is_dirty"] = False
            p["is_edited"] = False

            # Instantly revert filmstrip thumbnail to raw base
            if hasattr(self, 'thumb_controller'):
                self.thumb_controller.revert_photo_to_raw(p, self.filmstrip)
            if hasattr(self, 'filmstrip'):
                self.filmstrip.set_photo_dirty(pid, False)

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

        self._session_is_dirty = True
        self._update_window_title()

        # 3. If active photo was cleared, reset live controls, canvas, and cards
        if active_cleared and self.active_photo_id:
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

            active_p = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
            if active_p:
                active_p["params"] = None
                active_p["film_profile"] = "none"
                active_p["paper_profile"] = "none"
                active_p["print_mode"] = "scan"
                active_p["is_dirty"] = False
                active_p["is_edited"] = False

            if self.active_photo_id in post_photos:
                post_photos[self.active_photo_id]["params"] = None

            self._update_all_sliders_from_params()
            self._update_reset_buttons_state()
            self._update_diffusion_ui_state()

            self.canvas.update_params(**self.current_params)
            lut_3d = self.engine.get_3d_lut("none", "none", lut_size=33, params_dict=self.current_params)
            self.canvas.set_lut(lut_3d)
            self._schedule_histogram_update()
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

    def action_clear_photos(self):
        """Completely flush bottom filmstrip, engine session, and canvas without exiting to start screen."""
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
        self.thumb_controller.reset_all_cards_to_color_pills(self.film_cards, self.paper_cards)
        self.restore_default_status()

    def action_export_image(self):
        if not self._current_photo_path:
            return

        last_cfg = config_manager.get_last_export_settings()
        last_dir = ""
        if last_cfg and "path" in last_cfg:
            p = last_cfg["path"]
            if os.path.isdir(p):
                last_dir = p
            elif p:
                last_dir = os.path.dirname(p)
        if not last_dir or not os.path.exists(last_dir):
            last_dir = os.path.dirname(self._current_photo_path) if self._current_photo_path else os.path.expanduser("~")

        base_name = os.path.splitext(os.path.basename(self._current_photo_path))[0]
        default_out = os.path.join(last_dir, f"{base_name}.jpg")

        dlg = ExportImageDialog(default_out, is_batch=False, parent=self)
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
            config_manager.save_last_export_settings({"path": os.path.dirname(out_path)})
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
        self._set_export_status("已加入冲印队列，正在后台冲印...", state="busy")

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
        """Item 16: Batch export multiple photos with background export queue."""
        if not photo_ids:
            return

        last_cfg = config_manager.get_last_export_settings()
        last_dir = ""
        if last_cfg and "path" in last_cfg:
            p = last_cfg["path"]
            if os.path.isdir(p):
                last_dir = p
            elif p:
                last_dir = os.path.dirname(p)
        if not last_dir or not os.path.exists(last_dir):
            first_photo = next((p for p in self.photos if p["id"] in photo_ids), None)
            last_dir = os.path.dirname(first_photo["path"]) if first_photo else os.path.expanduser("~")
        default_dir = last_dir

        dialog = ExportImageDialog(default_dir, is_batch=True, batch_count=len(photo_ids), parent=self)
        if not dialog.exec():
            return

        cfg = dialog.get_export_config()
        out_dir = cfg.get("path", "").strip()
        if not out_dir:
            return

        try:
            os.makedirs(out_dir, exist_ok=True)
            config_manager.save_last_export_settings({"path": out_dir})
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

        import uuid
        enqueued_count = 0
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

            out_fn = f"{os.path.splitext(os.path.basename(file_p))[0]}{ext}"
            out_target = os.path.join(out_dir, out_fn)

            pix = None
            if photo.get("thumbnail_rgb") is not None:
                trgb = photo["thumbnail_rgb"]
                th, tw = trgb.shape[:2]
                qimg = QImage(trgb.data, tw, th, tw * 3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg)

            tid = f"task_{uuid.uuid4().hex[:8]}"
            task = ExportTask(tid, os.path.basename(out_target), file_p, out_target, export_params, thumbnail=pix)
            self.export_queue.enqueue(task)
            enqueued_count += 1

        self._set_export_status(f"已将 {enqueued_count} 张底片加入冲印队列...", state="busy")

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
        
        act_imp = menu.addAction("导入胶卷配置...")
        icon_imp = os.path.join(get_resource_dir(), "icons", "dlg_import.png")
        if os.path.exists(icon_imp):
            act_imp.setIcon(QIcon(icon_imp))
        act_imp.triggered.connect(self._import_film_lut)

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
        
        act_imp = menu.addAction("导入相纸配置...")
        icon_imp = os.path.join(get_resource_dir(), "icons", "dlg_import.png")
        if os.path.exists(icon_imp):
            act_imp.setIcon(QIcon(icon_imp))
        act_imp.triggered.connect(self._import_paper_lut)

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
            self.btn_max.set_maximized_state(self.isMaximized())

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

            # Remember active session path ONLY if it is a saved session on disk
            if self._current_session_path and os.path.exists(self._current_session_path) and not getattr(self, "_is_temp_session", False):
                config_manager.save_last_session_path(self._current_session_path)
            else:
                config_manager.save_last_session_path(None)
                if self.photos:
                    self._save_auto_temp_session()
        except Exception as e:
            logger.warning(f"Error saving config on close: {e}")

        if self._import_worker and self._import_worker.isRunning():
            self._import_worker.stop()
            self._import_worker.wait(300)

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
