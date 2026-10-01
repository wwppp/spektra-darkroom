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
    QMessageBox, QScrollArea, QFrame, QSplitter, QSplitterHandle, QGridLayout, QSizePolicy, QApplication
)

from path_utils import get_icon_path, get_resource_dir
from version import get_app_title, get_full_version_info
import sdc_manager
import config_manager
import session_cache_manager
from ui.export_queue_manager import ExportQueueManager, ExportQueueDialog, ExportTask
from ui.unsaved_session_dialog import UnsavedSessionDialog

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

from ui.window_utils import apply_dark_titlebar, get_darkroom_menu_style
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


def get_icon(name, target_size=None):
    path = get_icon_path(name)
    if not os.path.exists(path):
        return QIcon()
    return QIcon(path)


class WindowCaptionButton(QPushButton):
    """Vector-drawn Windows 11 style window control button."""
    def __init__(self, btn_type="close", parent=None):
        super().__init__(parent)
        self.btn_type = btn_type
        self.setFixedSize(46, 32)
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



class StockThumbnailWorker(QThread):
    film_ready = Signal(str, QPixmap)
    paper_ready = Signal(str, QPixmap)

    def __init__(self, engine, linear_thumb, film_stocks, paper_stocks, current_film, current_paper, do_films=True, do_papers=True, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.linear_thumb = linear_thumb.copy() if linear_thumb is not None else None
        self.film_stocks = list(film_stocks)
        self.paper_stocks = list(paper_stocks)
        self.current_film = current_film
        self.current_paper = current_paper
        self.do_films = do_films
        self.do_papers = do_papers
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        if self.linear_thumb is None or self._stopped:
            return

        from app_core import filmic_shoulder_np, prophoto_to_srgb

        # If passed an 8-bit integer image by fallback, normalize to float32
        if self.linear_thumb.dtype == np.uint8:
            small_float = self.linear_thumb.astype(np.float32) / 255.0
        else:
            small_float = self.linear_thumb.astype(np.float32)

        # Scale down while strictly preserving aspect ratio so 3D LUT grading is ultra-fast
        h, w = small_float.shape[:2]
        max_edge = 120
        if max(h, w) > max_edge:
            scale = max_edge / float(max(h, w))
            target_w = max(1, int(round(w * scale)))
            target_h = max(1, int(round(h * scale)))
            small_linear = cv2.resize(small_float, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            small_linear = small_float.copy()

        # Item 3: Apply filmic soft highlight shoulder roll-off matching GLSL viewport shader
        rolled_linear = filmic_shoulder_np(np.maximum(0.0, small_linear), 0.75)

        # Reference stocks to pair with when current stock is 'none' (bypass mode)
        ref_paper = self.current_paper if self.current_paper != "none" else "kodak_2383"
        ref_film = self.current_film if self.current_film != "none" else "kodak_portra_400"

        # 1. Process Film stocks (paired with current paper or reference paper)
        if self.do_films:
            for sid in self.film_stocks:
                if self._stopped:
                    return
                try:
                    if sid == "none":
                        # Bypass mode: directly convert linear ProPhoto to display sRGB
                        srgb_u8 = (prophoto_to_srgb(small_linear) * 255.0).astype(np.uint8)
                        gh, gw, gc = srgb_u8.shape
                        qimg = QImage(srgb_u8.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.film_ready.emit(sid, pix)
                    else:
                        lut = self.engine.get_3d_lut(sid, ref_paper, lut_size=17)
                        graded = self.engine.apply_lut_to_linear(rolled_linear, lut)
                        gh, gw, gc = graded.shape
                        qimg = QImage(graded.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.film_ready.emit(sid, pix)
                except Exception:
                    pass

        # 2. Process Paper stocks (paired with current film or reference film)
        if self.do_papers:
            for sid in self.paper_stocks:
                if self._stopped:
                    return
                try:
                    if sid == "none":
                        # Scan mode / Bypass mode: digital scan directly from film
                        srgb_u8 = (prophoto_to_srgb(small_linear) * 255.0).astype(np.uint8)
                        gh, gw, gc = srgb_u8.shape
                        qimg = QImage(srgb_u8.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.paper_ready.emit(sid, pix)
                    else:
                        lut = self.engine.get_3d_lut(ref_film, sid, lut_size=17)
                        graded = self.engine.apply_lut_to_linear(rolled_linear, lut)
                        gh, gw, gc = graded.shape
                        qimg = QImage(graded.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.paper_ready.emit(sid, pix)
                except Exception:
                    pass


class BatchImportWorker(QThread):
    """Background worker that incrementally loads RAW and image files,
    streaming results to the GUI without freezing the main thread.
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
        loaded = 0
        for i, p in enumerate(self.file_paths, 1):
            if self._stopped:
                break
            try:
                # 1. Fast path: check valid session cache (preview loaded only for first photo)
                should_load_prev = (i == 1 and self.start_idx == 0)
                cached_data = session_cache_manager.get_photo_cache(p, load_preview=should_load_prev)
                if cached_data is not None:
                    pid = f"photo_{self.start_idx + loaded}_{os.path.basename(p)}"
                    photo_entry = {
                        "id": pid,
                        "filename": os.path.basename(p),
                        "path": p,
                        "width": cached_data.get("width", 0),
                        "height": cached_data.get("height", 0),
                        "thumbnail_rgb": cached_data.get("thumbnail_rgb"),
                        "float_img": cached_data.get("float_img"),
                        "exif": cached_data.get("exif", {}),
                        "auto_ev": cached_data.get("auto_ev", 0.0)
                    }
                    loaded += 1
                    self.photoLoaded.emit(photo_entry, i, total)
                    continue

                # 2. Slow path: decode via engine
                res = self.engine.load_image(p)
                if res.get("success"):
                    pid = f"photo_{self.start_idx + loaded}_{os.path.basename(p)}"
                    photo_entry = {
                        "id": pid,
                        "filename": os.path.basename(p),
                        "path": p,
                        "width": res.get("width", 0),
                        "height": res.get("height", 0),
                        "thumbnail_rgb": res.get("thumbnail_rgb"),
                        "float_img": res.get("float_img"),
                        "exif": res.get("exif", {}),
                        "auto_ev": 0.0,
                        "base_ev": res.get("base_ev", 0.0)
                    }
                    # Write-through: save full preview for first photo, fast thumbnail+exif for others
                    save_prev = (i == 1 and self.start_idx == 0)
                    session_cache_manager.save_photo_cache(p, photo_entry, save_preview=save_prev)
                    loaded += 1
                    self.photoLoaded.emit(photo_entry, i, total)
            except Exception as e:
                logger.warning(f"Error importing {p}: {e}")
        self.importFinished.emit(loaded)


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

    def __init__(self, stock_info, is_active=False, is_custom=False, parent=None):
        super().__init__(parent)
        self.setObjectName("filmGridCard")
        self.stock_id = stock_info["id"]
        self.stock_name = stock_info["name"]
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

        tip = f"{self.stock_name}\n{stock_info.get('badge', '')}\n{stock_info.get('desc', '')}".strip()
        self.setToolTip(tip)

        self._init_ui()

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

    def _update_appearance(self):
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
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.stock_id)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            menu.setStyleSheet(get_darkroom_menu_style())
            act_apply = menu.addAction("应用")
            act_apply.triggered.connect(lambda: self.selected.emit(self.stock_id))
            menu.addSeparator()
            act_del = menu.addAction("丢弃")
            icon_del = os.path.join(get_resource_dir(), "icons", "dlg_discard.png")
            if os.path.exists(icon_del):
                act_del.setIcon(QIcon(icon_del))
            act_del.triggered.connect(lambda: self.removeRequested.emit(self.stock_id))
            menu.exec(event.globalPosition().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)


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
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")

        # Content widget
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        grid = QGridLayout(content)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        grid.setContentsMargins(0, 2, 8, 2)
        grid.setColumnMinimumWidth(0, 70)
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
            ("白平衡", "white_balance"),
            ("ISO 速度", "iso"),
            ("光圈值", "aperture"),
            ("曝光时间", "shutter_speed"),
            ("曝光补偿", "exposure_bias"),
            ("曝光程序", "exposure_program"),
            ("测光模式", "metering_mode"),
            ("闪光灯", "flash"),
            ("焦距", "focal_length"),
            ("镜头型号", "lens_model"),
        ]

        for row, (label_text, key) in enumerate(fields):
            lbl_k = QLabel(label_text)
            lbl_k.setFixedWidth(70)
            lbl_k.setStyleSheet("color: #7b8092; font-size: 10px;")
            lbl_v = QLabel("-")
            lbl_v.setStyleSheet("color: #e2e8f0; font-size: 10px; font-weight: 500;")
            lbl_v.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            lbl_v.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            grid.addWidget(lbl_k, row, 0)
            grid.addWidget(lbl_v, row, 1)
            self.labels[key] = lbl_v

        self.scroll_area.setWidget(content)
        c_layout.addWidget(self.scroll_area, 1)

        layout.addWidget(card)

    def set_exif_data(self, data):
        for k, lbl in self.labels.items():
            raw_val = data.get(k, "-")
            val = str(raw_val) if raw_val is not None and str(raw_val).strip() != "" else "-"
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
            margin_x = 16
            line_w = max(0, w - margin_x * 2)
            grad = QLinearGradient(margin_x, cy, max(margin_x + 1, w - margin_x), cy)
            if self._hovered:
                grad.setColorAt(0.0, QColor(0, 0, 0, 0))
                grad.setColorAt(0.12, QColor(245, 158, 11, 70))
                grad.setColorAt(0.5, QColor(251, 191, 36, 230))
                grad.setColorAt(0.88, QColor(245, 158, 11, 70))
                grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            else:
                grad.setColorAt(0.0, QColor(0, 0, 0, 0))
                grad.setColorAt(0.15, QColor(50, 53, 68, 130))
                grad.setColorAt(0.5, QColor(217, 119, 6, 175))
                grad.setColorAt(0.85, QColor(50, 53, 68, 130))
                grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(margin_x, cy, line_w, 1)
        else:
            # Vertical line separating horizontal columns
            cx = w // 2
            margin_y = 10
            line_h = max(0, h - margin_y * 2)
            grad = QLinearGradient(cx, margin_y, cx, max(margin_y + 1, h - margin_y))
            if self._hovered:
                grad.setColorAt(0.0, QColor(0, 0, 0, 0))
                grad.setColorAt(0.1, QColor(245, 158, 11, 80))
                grad.setColorAt(0.5, QColor(251, 191, 36, 220))
                grad.setColorAt(0.9, QColor(245, 158, 11, 80))
                grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            else:
                grad.setColorAt(0.0, QColor(0, 0, 0, 0))
                grad.setColorAt(0.15, QColor(40, 42, 54, 110))
                grad.setColorAt(0.5, QColor(217, 119, 6, 130))
                grad.setColorAt(0.85, QColor(40, 42, 54, 110))
                grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRect(cx, margin_y, 1, line_h)


class FadingSplitter(QSplitter):
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(7)
        self.setChildrenCollapsible(False)
        self.setStyleSheet("QSplitter { background: transparent; }")

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
        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
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
        self.current_film_stock = "none"
        self.current_paper_stock = "none"
        self.current_print_mode = "optical"
        self.current_params = {
            "print_mode": "optical",
            "exposure_ev": 0.0,
            "color_temp": 5500.0,
            "tint": 1.0,
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
            "halation": 0.5,
            "halation_bounces": 2,
            "halation_decay": 0.5,
            "halation_boost": 0.0,
            "grain": 0.4,
            "grain_size": 1.0,
            "grain_cloud_blur": 1.0,
            "split_x": 0.5,
            "view_mode": 0
        }

        # Undo / Redo history
        self._undo_stack = []
        self._redo_stack = []

        # Window Geometry state tracking
        self._saved_normal_geo = None
        self._current_photo_thumb = None
        self._thumb_worker = None

        # Central canvas & widgets
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

        # Backslash (\) key compare shortcut & long-press tracking
        self._backslash_timer = QTimer(self)
        self._backslash_timer.setSingleShot(True)
        self._backslash_timer.setInterval(180)
        self._backslash_timer.timeout.connect(self._on_backslash_timeout)
        self._backslash_pressed = False
        self._backslash_long_pressed = False
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
            if restore_session_enabled and not getattr(self, "_has_cli_targets", False) and not self.photos and not self._current_session_path:
                last_session = config_manager.get_last_session_path()
                if last_session and os.path.exists(last_session):
                    QTimer.singleShot(40, lambda: self._open_session_by_path(last_session))
        except Exception as e:
            logger.warning(f"Error restoring application state: {e}")

    def _restore_saved_session(self, existing_files, active_file=None):
        """Ultra-fast session restoration: loads cached filmstrip thumbnails and EXIF in <100ms.
        Any uncached or changed files are streamed in background without freezing the UI.
        """
        if not existing_files:
            return

        cached_entries = []
        uncached_files = []

        target_path = active_file if (active_file and active_file in existing_files) else existing_files[0]
        self._target_initial_active_path = target_path

        for idx, p in enumerate(existing_files):
            is_target = (p == target_path)
            cache_item = session_cache_manager.get_photo_cache(p, load_preview=is_target)
            if cache_item is not None:
                pid = f"photo_{idx}_{os.path.basename(p)}"
                photo_entry = {
                    "id": pid,
                    "filename": os.path.basename(p),
                    "path": p,
                    "width": cache_item.get("width", 0),
                    "height": cache_item.get("height", 0),
                    "thumbnail_rgb": cache_item.get("thumbnail_rgb"),
                    "float_img": cache_item.get("float_img"),
                    "exif": cache_item.get("exif", {}),
                    "auto_ev": cache_item.get("auto_ev", 0.0)
                }
                cached_entries.append(photo_entry)
            else:
                uncached_files.append(p)

        if cached_entries:
            # Item 13: Merge session photo metadata (params, film, paper, dirty flags)
            if hasattr(self, "_pending_session_map") and self._pending_session_map:
                from path_utils import normalize_path
                for entry in cached_entries:
                    norm_p = normalize_path(entry.get("path", ""))
                    s_p = self._pending_session_map.get(norm_p)
                    if s_p:
                        entry["film_profile"] = s_p.get("film_stock", "none")
                        entry["paper_profile"] = s_p.get("paper_stock", "none")
                        entry["print_mode"] = s_p.get("print_mode", "optical")
                        entry["base_ev"] = s_p.get("base_ev", entry.get("base_ev", 0.0))
                        entry["params"] = dict(s_p.get("params") or {})
                        is_dirty = bool(s_p.get("is_dirty", False))
                        is_edited = bool(s_p.get("is_edited", False))
                        entry["is_dirty"] = is_dirty
                        entry["is_edited"] = is_edited

            # 1. Switch to darkroom workspace instantly
            self.stack.setCurrentIndex(1)
            self.btn_title_export.setVisible(True)

            # 2. Populate filmstrip with all cached thumbnails in one go (clean replacement)
            self.photos = list(cached_entries)
            target_entry = next((p for p in cached_entries if p.get("path") == target_path), cached_entries[0])
            target_id = target_entry["id"]
            self.filmstrip.set_photos(self.photos, target_id)

            # Item 13: Restore dirty indicators on filmstrip
            for entry in cached_entries:
                if entry.get("is_dirty") or entry.get("is_edited"):
                    self.filmstrip.set_photo_dirty(entry["id"], True)

            # 4. Activate target photo immediately
            self.active_photo_id = None
            self.on_switch_photo(target_id)
            self.set_app_status(f"已恢复底片库 ({len(cached_entries)}/{len(existing_files)} 张)", timeout_ms=3000)

        # 5. If some files are uncached or were modified, import them seamlessly in background
        if uncached_files:
            self._import_files_list(uncached_files)
        elif not cached_entries:
            # None was cached, full import fallback
            self._import_files_list(existing_files)

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
            # Enable Windows native Aero Snap, edge snapping and native minimize/maximize animations
            GWL_STYLE = -16
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_MAXIMIZEBOX = 0x00010000
            WS_MINIMIZEBOX = 0x00020000
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_STYLE, style | WS_CAPTION | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX
            )
            SWP_NOZORDER = 0x0004
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_FRAMECHANGED = 0x0020
            ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_win11_corners(self.isMaximized())
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
        if event.mimeData().hasUrls():
            valid_exts = {
                ".arw", ".cr2", ".cr3", ".nef", ".raf", ".dng",
                ".tif", ".tiff", ".png"
            }
            files = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    local_path = url.toLocalFile()
                    if os.path.exists(local_path):
                        if os.path.isdir(local_path):
                            for root, _, fnames in os.walk(local_path):
                                for f in fnames:
                                    if os.path.splitext(f)[1].lower() in valid_exts:
                                        files.append(os.path.join(root, f))
                        else:
                            if os.path.splitext(local_path)[1].lower() in valid_exts:
                                files.append(local_path)
            if files:
                event.acceptProposedAction()
                self._import_files_list(files)
            else:
                event.ignore()
        else:
            event.ignore()

    def _update_root_style(self, is_max):
        radius = "0px" if is_max else "8px"
        border = "none" if is_max else "1px solid #282a34"
        tb_radius = "0px" if is_max else "8px"
        cw = self.centralWidget()
        if cw:
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
            QMenu {
                background: #18191f;
                color: #e2e8f0;
                border: 1px solid #2f3240;
                border-radius: 6px;
                padding: 4px;
                font-size: 12px;
            }
            QMenu::icon {
                padding-left: 6px;
                width: 14px;
                height: 14px;
            }
            QMenu::item {
                padding: 5px 22px 5px 28px;
                margin: 1px 4px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #f59e0b;
                color: #111113;
                font-weight: bold;
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
        self.btn_title_export = QPushButton("导出")
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

        # Caption Buttons (Win11 Vector)
        self.btn_min = WindowCaptionButton("min", self)
        self.btn_min.clicked.connect(self.showMinimized)
        tb_layout.addWidget(self.btn_min, 0, Qt.AlignmentFlag.AlignVCenter)

        self.btn_max = WindowCaptionButton("max", self)
        self.btn_max.clicked.connect(self.toggle_maximize)
        tb_layout.addWidget(self.btn_max, 0, Qt.AlignmentFlag.AlignVCenter)

        self.btn_close = WindowCaptionButton("close", self)
        self.btn_close.clicked.connect(self.close)
        tb_layout.addWidget(self.btn_close, 0, Qt.AlignmentFlag.AlignVCenter)

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
        btn_import_file.setIcon(get_icon("open_file.png"))
        btn_import_file.setIconSize(QSize(16, 16))
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
        btn_import_folder.setIcon(get_icon("open_folder.png"))
        btn_import_folder.setIconSize(QSize(16, 16))
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
        self.film_scroll.setStyleSheet("background: transparent; border: none;")

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
        self.paper_scroll.setStyleSheet("background: transparent; border: none;")

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

        tb_layout.addStretch(1)

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
        self.center_splitter.addWidget(self.filmstrip)

        self.center_splitter.setStretchFactor(0, 1)
        self.center_splitter.setStretchFactor(1, 0)
        self.center_splitter.setSizes([600, 96])

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
        sidebar_scroll.setStyleSheet("background: transparent; border: none;")

        sidebar_content = QWidget()
        self.sidebar_layout = QVBoxLayout(sidebar_content)
        self.sidebar_layout.setContentsMargins(0, 4, 0, 0)
        self.sidebar_layout.setSpacing(6)

        # Sections (Comprehensive Physical Analog Pipeline)
        self._init_exposure_section()
        self._init_optics_section()
        self._init_chemistry_section()
        self._init_enlarger_section()
        self._init_texture_section()

        # Master Reset
        btn_reset_all = QPushButton("重置全部参数 (Ctrl+R)")
        btn_reset_all.setIcon(get_icon("reset.png"))
        btn_reset_all.setIconSize(QSize(14, 14))
        btn_reset_all.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_all.setStyleSheet("""
            QPushButton {
                background: #1c1d24;
                color: #f59e0b;
                border: 1px solid #333644;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                font-size: 11.5px;
            }
            QPushButton:hover {
                background: #262834;
                border-color: #f59e0b;
            }
            QPushButton:pressed {
                background: #14151a;
                border-color: #d97706;
            }
        """)
        btn_reset_all.clicked.connect(self.reset_all_params)
        self.sidebar_layout.addWidget(btn_reset_all)
        self.sidebar_layout.addStretch(1)

        sidebar_scroll.setWidget(sidebar_content)
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
                card = FilmGridCard(item, is_active=(item["id"] == self.current_film_stock))
                card.selected.connect(self.on_switch_film)
                card.exportRequested.connect(lambda sid: self._export_film_lut(sid))
                card.removeRequested.connect(lambda sid: self._remove_film_lut(sid))
                self.film_cards[item["id"]] = card
        if hasattr(self, 'lbl_film_title'):
            self.lbl_film_title.setText(f"胶卷型号 ({len(self.film_cards)} 卷)")
        self.reflow_grids()

    def _populate_paper_grid(self):
        papers = self.engine.get_paper_stocks()
        for item in papers:
            card = FilmGridCard(item, is_active=(item["id"] == self.current_paper_stock))
            card.selected.connect(self.on_switch_paper)
            card.exportRequested.connect(lambda sid: self._export_paper_lut(sid))
            card.removeRequested.connect(lambda sid: self._remove_paper_lut(sid))
            self.paper_cards[item["id"]] = card
        if hasattr(self, 'lbl_paper_title'):
            self.lbl_paper_title.setText(f"放大相纸 ({len(self.paper_cards)} 张)")
        self.reflow_grids()

    def _init_exposure_section(self):
        self.sec_exposure = AccordionSection("场景曝光")
        self.sec_exposure.resetClicked.connect(lambda: self.reset_track("exposure"))

        self.slider_ev = AdobeScrubSlider(
            "曝光", -3.0, 3.0, 0.0, step=0.05, unit=" EV", decimals=2,
            tooltip="模拟相机物理感光曝光调节，以真实 EV 档位线性增减画面总进光量"
        )
        self.slider_ev.valueChanged.connect(lambda v: self._on_slider_live("exposure_ev", v))
        self.slider_ev.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_ev)

        self.slider_temp = AdobeScrubSlider(
            "色温", 2500, 9500, 5500, step=25, unit=" K", decimals=0,
            tooltip="调整场景相关色温 (CCT)，向左偏冷调蓝光，向右偏温暖黄光"
        )
        self.slider_temp.valueChanged.connect(lambda v: self._on_slider_live("color_temp", v))
        self.slider_temp.sliderReleased.connect(self._on_slider_committed)
        self.sec_exposure.addWidget(self.slider_temp)

        self.slider_tint = AdobeScrubSlider(
            "色调", 0.70, 1.30, 1.0, step=0.01, unit="", decimals=2,
            tooltip="调整绿光与洋红平衡 (Tint)，微调中间调肤色偏向"
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
            tooltip="底片画幅尺寸：直接影响银盐晶体与微米级光学弥散圈的物理相对尺度"
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
            tooltip="经典电影镜头物理柔光镜仿真：在高光周围形成柔和漫射与呼吸感"
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
            tooltip="商业电影镜头标准柔光挡位 (0, 1/8, 1/4, 1/2, 1, 2 档)"
        )
        self.combo_diff_strength.valueChanged.connect(lambda v: self._on_combo_changed("diffusion_strength", v))
        self.slider_diff_strength = self.combo_diff_strength  # Backward compatibility alias
        self.sec_optics.addWidget(self.combo_diff_strength)

        self.slider_diff_warmth = AdobeScrubSlider(
            "光晕冷暖偏置", -1.5, 1.5, 0.0, step=0.05, unit="", decimals=2,
            tooltip="高光光晕能量在不同波长子层间的再分配，正值偏复古暖金，负值偏清冷晨光"
        )
        self.slider_diff_warmth.valueChanged.connect(lambda v: self._on_slider_live("diffusion_warmth", v))
        self.slider_diff_warmth.sliderReleased.connect(self._on_slider_committed)
        self.sec_optics.addWidget(self.slider_diff_warmth)

        self.sidebar_layout.addWidget(self.sec_optics)

    def _init_chemistry_section(self):
        self.sec_chemistry = AccordionSection("胶片乳剂与化学 (DIR)")
        self.sec_chemistry.resetClicked.connect(lambda: self.reset_track("chemistry"))

        self.slider_dir_amount = AdobeScrubSlider(
            "抑制剂总量", 0.0, 2.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="DIR 显影抑制耦合剂全局强度乘数，1.0 为胶卷出厂物理标准状态"
        )
        self.slider_dir_amount.valueChanged.connect(lambda v: self._on_slider_live("dir_amount", v))
        self.slider_dir_amount.sliderReleased.connect(self._on_slider_committed)
        self.sec_chemistry.addWidget(self.slider_dir_amount)

        self.slider_dir_interlayer = AdobeScrubSlider(
            "跨层色彩纯度", 0.0, 2.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="跨层抑制增强：相邻感光层相互抑制释放，形成胶片标志性的通透色彩分离与纯度，鲜艳而不溢出"
        )
        self.slider_dir_interlayer.valueChanged.connect(lambda v: self._on_slider_live("dir_interlayer", v))
        self.slider_dir_interlayer.sliderReleased.connect(self._on_slider_committed)
        self.sec_chemistry.addWidget(self.slider_dir_interlayer)

        self.slider_dir_samelayer = AdobeScrubSlider(
            "同层边缘反差", 0.0, 2.0, 1.0, step=0.05, unit="", decimals=2,
            tooltip="同层自抑制：控制单色层感光微反差，避免大光比下局部色彩密度堆积过厚"
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
            tooltip="放大机物理投光光源光谱分布，真实影响相纸感光各层的综合响应"
        )
        self.combo_illuminant.valueChanged.connect(lambda v: self._on_combo_changed("enlarger_illuminant", v))
        self.sec_enlarger.addWidget(self.combo_illuminant)

        self.slider_magenta = AdobeScrubSlider(
            "品红 / 绿", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="放大机滤色镜品红(M)调节，负值偏绿，正值偏品红"
        )
        self.slider_magenta.valueChanged.connect(lambda v: self._on_slider_live("enlarger_magenta", v))
        self.slider_magenta.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_magenta)

        self.slider_yellow = AdobeScrubSlider(
            "黄色 / 蓝", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="放大机滤色镜黄色(Y)调节，负值偏冷蓝，正值偏暖黄"
        )
        self.slider_yellow.valueChanged.connect(lambda v: self._on_slider_live("enlarger_yellow", v))
        self.slider_yellow.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_yellow)

        self.slider_cyan = AdobeScrubSlider(
            "青色", -35.0, 35.0, 0.0, step=0.5, unit="", decimals=1,
            tooltip="放大机青色滤镜微调，控制暗部阴影色调与对比"
        )
        self.slider_cyan.valueChanged.connect(lambda v: self._on_slider_live("enlarger_cyan", v))
        self.slider_cyan.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_cyan)

        self.slider_print_exp = AdobeScrubSlider(
            "相纸曝光", 0.5, 2.5, 1.0, step=0.02, unit="x", decimals=2,
            tooltip="调整相纸在放大机下的曝光时间，影响相纸整体显影浓度"
        )
        self.slider_print_exp.valueChanged.connect(lambda v: self._on_slider_live("print_exposure", v))
        self.slider_print_exp.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_print_exp)

        self.slider_preflash = AdobeScrubSlider(
            "预闪反差", 0.0, 0.25, 0.0, step=0.01, unit="", decimals=2,
            tooltip="暗房预闪光技术 (Pre-exposure)，降低相纸高反差以拉出极暗部阴影细节"
        )
        self.slider_preflash.valueChanged.connect(lambda v: self._on_slider_live("pre_flash", v))
        self.slider_preflash.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_preflash)

        self.slider_morph_gamma = AdobeScrubSlider(
            "显影形态反差", 0.70, 1.30, 1.0, step=0.02, unit="", decimals=2,
            tooltip="相纸感光 D-logH 显影曲线斜率形态，影响中间调与暗部的陡峭度"
        )
        self.slider_morph_gamma.valueChanged.connect(lambda v: self._on_slider_live("morph_gamma", v))
        self.slider_morph_gamma.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_morph_gamma)

        self.slider_exhaustion = AdobeScrubSlider(
            "显影液疲劳度", 0.0, 0.50, 0.0, step=0.02, unit="", decimals=2,
            tooltip="模拟暗房冲洗药水氧化衰老，略微提亮死黑黑位，呈现复古老相纸柔化暗部质感"
        )
        self.slider_exhaustion.valueChanged.connect(lambda v: self._on_slider_live("developer_exhaustion", v))
        self.slider_exhaustion.sliderReleased.connect(self._on_slider_committed)
        self.sec_enlarger.addWidget(self.slider_exhaustion)

        self.sidebar_layout.addWidget(self.sec_enlarger)

    def _init_texture_section(self):
        self.sec_texture = AccordionSection("质感与物理光晕")
        self.sec_optical = self.sec_texture  # Backward compatibility alias
        self.sec_texture.resetClicked.connect(lambda: self.reset_track("texture"))

        self.slider_halation = AdobeScrubSlider(
            "光晕总强度", 0.0, 1.0, 0.5, step=0.02, unit="", decimals=2,
            tooltip="胶片红光背反射物理光晕 (Anti-halation) 暖红泛光强度"
        )
        self.slider_halation.valueChanged.connect(lambda v: self._on_slider_live("halation", v))
        self.slider_halation.sliderReleased.connect(self._on_slider_committed)
        self.sec_texture.addWidget(self.slider_halation)

        self.slider_halation_bounces = AdobeScrubSlider(
            "片基弹跳次数", 1.0, 4.0, 2.0, step=1.0, unit=" 次", decimals=0,
            tooltip="光子在片基和底板间的多重反射弹跳次数，形成由浓到淡的环形扩散层"
        )
        self.slider_halation_bounces.valueChanged.connect(lambda v: self._on_slider_live("halation_bounces", int(v)))
        self.slider_halation_bounces.sliderReleased.connect(self._on_slider_committed)
        self.sec_texture.addWidget(self.slider_halation_bounces)

        self.slider_halation_decay = AdobeScrubSlider(
            "弹跳能量衰减", 0.20, 0.80, 0.50, step=0.05, unit="", decimals=2,
            tooltip="多次弹跳间的光子吸收与逸散衰减率"
        )
        self.slider_halation_decay.valueChanged.connect(lambda v: self._on_slider_live("halation_decay", v))
        self.slider_halation_decay.sliderReleased.connect(self._on_slider_committed)
        self.sec_texture.addWidget(self.slider_halation_decay)

        self.slider_halation_boost = AdobeScrubSlider(
            "高光泛红重建", 0.0, 2.5, 0.0, step=0.05, unit=" EV", decimals=2,
            tooltip="超大光比高光点预截断重构，让逆光光源呈现深邃温暖的胶片红晕包容感"
        )
        self.slider_halation_boost.valueChanged.connect(lambda v: self._on_slider_live("halation_boost", v))
        self.slider_halation_boost.sliderReleased.connect(self._on_slider_committed)
        self.sec_texture.addWidget(self.slider_halation_boost)

        self.slider_grain = AdobeScrubSlider(
            "银盐颗粒感", 0.0, 1.0, 0.4, step=0.02, unit="", decimals=2,
            tooltip="模拟感光乳剂中卤化银微粒晶体结构在显影后的自然有机颗粒质感"
        )
        self.slider_grain.valueChanged.connect(lambda v: self._on_slider_live("grain", v))
        self.slider_grain.sliderReleased.connect(self._on_slider_committed)
        self.sec_texture.addWidget(self.slider_grain)

        self.slider_grain_size = AdobeScrubSlider(
            "颗粒物理尺寸", 0.5, 2.5, 1.0, step=0.05, unit="x", decimals=2,
            tooltip="微调胶卷感光乳剂中卤化银微晶与发色染料微云的物理直径大小 (0.50x 细腻 ~ 2.50x 粗犷)"
        )
        self.slider_grain_size.valueChanged.connect(lambda v: self._on_slider_live("grain_size", v))
        self.slider_grain_size.sliderReleased.connect(self._on_slider_committed)
        self.sec_texture.addWidget(self.slider_grain_size)

        self.slider_grain_cloud = AdobeScrubSlider(
            "染料云团柔化", 0.5, 2.5, 1.0, step=0.05, unit=" um", decimals=2,
            tooltip="将数码点状噪点柔化晕开为真实的彩色胶片发色染料云团 (Dye Clouds)"
        )
        self.slider_grain_cloud.valueChanged.connect(lambda v: self._on_slider_live("grain_cloud_blur", v))
        self.slider_grain_cloud.sliderReleased.connect(self._on_slider_committed)
        self.sec_texture.addWidget(self.slider_grain_cloud)

        self.sidebar_layout.addWidget(self.sec_texture)

    _init_optical_section = _init_texture_section

    def _init_menu_and_actions(self):
        # 1. File Menu
        file_menu = self.menu_bar.addMenu("文件 (F)")
        
        act_open_file = QAction("打开文件...", self)
        act_open_file.setIcon(get_icon("open_file.png"))
        act_open_file.setShortcut(QKeySequence("Ctrl+O"))
        act_open_file.triggered.connect(self.action_open_files)
        file_menu.addAction(act_open_file)

        act_open_folder = QAction("打开文件夹...", self)
        act_open_folder.setIcon(get_icon("open_folder.png"))
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
        self.act_save_session.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_session.triggered.connect(self.action_save_session)
        file_menu.addAction(self.act_save_session)

        self.act_save_session_as = QAction("另存为会话文件 (.sdss)...", self)
        self.act_save_session_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_session_as.triggered.connect(self.action_save_session_as)
        file_menu.addAction(self.act_save_session_as)

        file_menu.addSeparator()

        # Item 4: Quick Export with last settings
        self.act_quick_export = QAction("快速导出 (上次参数)", self)
        self.act_quick_export.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.act_quick_export.triggered.connect(lambda: self.action_quick_export())
        self.act_quick_export.setEnabled(False)
        file_menu.addAction(self.act_quick_export)

        self.act_export = QAction("导出...", self)
        self.act_export.setIcon(get_icon("export.png"))
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.triggered.connect(self.action_export_image)
        self.act_export.setEnabled(False)
        file_menu.addAction(self.act_export)

        act_queue = QAction("导出队列管理...", self)
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
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self.undo)
        edit_menu.addAction(act_undo)

        act_redo = QAction("重做", self)
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

        act_film_lib = QAction("胶卷库...", self)
        act_film_lib.triggered.connect(lambda: self.open_stock_manager("film"))
        edit_menu.addAction(act_film_lib)

        act_paper_lib = QAction("相纸库...", self)
        act_paper_lib.triggered.connect(lambda: self.open_stock_manager("paper"))
        edit_menu.addAction(act_paper_lib)

        edit_menu.addSeparator()

        act_prefs = QAction("首选项...", self)
        act_prefs.setShortcut(QKeySequence("Ctrl+,"))
        act_prefs.triggered.connect(self.open_preferences)
        edit_menu.addAction(act_prefs)

        # 3. View Menu
        view_menu = self.menu_bar.addMenu("视图 (V)")
        
        act_fit_img = QAction("适应图像", self)
        act_fit_img.setShortcut(QKeySequence("Ctrl+0"))
        act_fit_img.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(act_fit_img)

        act_compare = QAction("对比\t\\", self)
        act_compare.setIcon(get_icon("compare.png"))
        act_compare.triggered.connect(self.toggle_split_view)
        view_menu.addAction(act_compare)

        act_fit_win = QAction("适应窗口", self)
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
            self.center_splitter.setSizes([600, 96])
        config_manager.save_window_config({
            "splitter_main": [320, 800, 320],
            "splitter_left": [300, 180, 200],
            "splitter_center": [600, 96]
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
            act.triggered.connect(lambda checked=False, p=path: self._open_session_by_path(p))

        self.recent_menu.addSeparator()
        act_clear = self.recent_menu.addAction("清除最近记录")
        act_clear.triggered.connect(lambda: (config_manager.clear_recent_sessions(), self._update_recent_menu()))

    def _mark_current_photo_dirty(self):
        if getattr(self, '_is_updating_sliders', False) or getattr(self, '_is_switching_photo', False):
            return
        self._photo_is_dirty = True
        self._session_is_dirty = True
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
        has_photo = bool(self._current_photo_path)
        if hasattr(self, 'act_export'):
            self.act_export.setEnabled(has_photo)
        if hasattr(self, 'act_quick_export'):
            self.act_quick_export.setEnabled(has_photo)
        if hasattr(self, 'act_export_lut'):
            self.act_export_lut.setEnabled(has_photo)

        is_temp = getattr(self, "_is_temp_session", False)
        session_name = os.path.basename(self._current_session_path) if (self._current_session_path and not is_temp) else "未命名会话"
        dirty_flag = "*" if getattr(self, "_session_is_dirty", False) else ""
        display_title = f"{dirty_flag}{session_name}"

        self.setWindowTitle(f"{display_title} - {base_title}")
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
            self._session_is_dirty = any(p.get("is_dirty", False) for p in self.photos)
            self._update_window_title()
            self.set_app_status(f"已保存更改: {os.path.basename(self._current_photo_path)}", timeout_ms=4000)

    def _on_backslash_timeout(self):
        if getattr(self, "_backslash_pressed", False):
            self._backslash_long_pressed = True
            self._on_compare_long_press_start()

    def eventFilter(self, watched, event):
        """Handle global backslash key for compare toggle / long-press preview."""
        et = event.type()
        if et in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == Qt.Key.Key_Backslash:
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
        """Item 5: Support Delete / Backspace keys to delete selected photo(s) from filmstrip."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            focus_w = QApplication.focusWidget()
            from PySide6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox
            if not isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)):
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
        # Lock status during batch import so auxiliary switch events don't wipe it
        if getattr(self, "_is_importing", False) and not text.startswith("正在导入") and not text.startswith("底片库导入完成"):
            return
        # Lock status during multi-selection so photo click doesn't wipe selection count
        if getattr(self, "_is_multi_selecting", False) and not is_export_link and not spinning:
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

        if is_export_link:
            self.status_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 11px; text-decoration: underline;")
            self.status_label.setToolTip("正在后台冲印，点击打开导出队列管理")
        else:
            self.status_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
            self.status_label.setToolTip("")

        if timeout_ms > 0:
            if hasattr(self, "_status_timer"):
                self._status_timer.start(timeout_ms)
        elif not spinning and text and not getattr(self, "_is_importing", False) and not getattr(self, "_is_multi_selecting", False) and not is_export_link:
            # Default lingering duration: 4.0 seconds without flickering or premature wipeout
            if hasattr(self, "_status_timer"):
                self._status_timer.start(4000)

    def _on_status_timeout(self):
        if getattr(self, "_is_importing", False) or getattr(self, "_is_multi_selecting", False):
            return
        if hasattr(self, "export_queue") and self.export_queue and self.export_queue.has_active_tasks():
            return
        self.status_label.setText("")
        self.spinner.stop()
        if hasattr(self, "status_sep"):
            self.status_sep.setStyleSheet("color: transparent; font-size: 11px;")

    def _on_status_label_clicked(self, event):
        if hasattr(self, "export_queue") and self.export_queue:
            self.open_export_queue_dialog()

    def open_export_queue_dialog(self):
        """Item 8: Opens the Export Queue Management dialog."""
        if hasattr(self, "export_queue") and self.export_queue:
            dlg = ExportQueueDialog(self.export_queue, self)
            dlg.exec()

    def _on_export_queue_status_changed(self, active_count, total_count):
        """Item 8: Live status updater for background export queue."""
        if active_count > 0:
            done_count = total_count - active_count
            self.set_app_status(f"正在冲印导出 ({done_count + 1}/{total_count})...", spinning=True, is_export_link=True)
        else:
            if total_count > 0:
                self.set_app_status(f"导出队列冲印完成 (共 {total_count} 项)", spinning=False, timeout_ms=4500)
            else:
                self.set_app_status("", spinning=False)

    def action_save_session(self):
        """Item 9 & 11: Saves current editing session (.sdss). Returns True if saved."""
        if not self.photos:
            self.set_app_status("当前没有打开的照片底片，无需保存会话", timeout_ms=4000)
            return False

        if not self._current_session_path or getattr(self, "_is_temp_session", False):
            return self.action_save_session_as()

        return self._do_save_session(self._current_session_path)

    def action_save_session_as(self):
        """Item 9 & 11: Prompts user to select a .sdss file path and saves session. Returns True if saved."""
        if not self.photos:
            self.set_app_status("当前没有打开的照片底片，无需保存会话", timeout_ms=3000)
            return False

        # Item 1: Suggest title session name rather than single image filename
        if self._current_session_path and not getattr(self, "_is_temp_session", False):
            default_name = os.path.basename(self._current_session_path)
            if not default_name.lower().endswith(".sdss"):
                default_name = f"{default_name}.sdss"
        else:
            default_name = "未命名会话.sdss"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存当前暗房会话工程", default_name, "SpektraDarkroom 会话工程 (*.sdss);;所有文件 (*.*)"
        )
        if file_path:
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
                    photo["params"] = dict(self.current_params)
                    photo["film_profile"] = self.current_film_stock
                    photo["paper_profile"] = self.current_paper_stock
                    photo["print_mode"] = getattr(self, "current_print_mode", "optical")

            session_photos = []
            for p in self.photos:
                session_photos.append({
                    "id": p["id"],
                    "path": normalize_path(p["path"]),
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
                "photos": session_photos
            }
            temp_path = config_manager.get_auto_saved_session_path()
            session_cache_manager.save_session_file(temp_path, session_data)
        except Exception as e:
            logger.warning(f"Error auto-saving temporary session: {e}")

    def _do_save_session(self, file_path):
        self.set_app_status("正在保存会话...", spinning=True)
        # Flush current dirty photo parameters to photo dict
        if self.active_photo_id:
            photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
            if photo:
                photo["params"] = dict(self.current_params)
                photo["film_profile"] = self.current_film_stock
                photo["paper_profile"] = self.current_paper_stock
                photo["print_mode"] = getattr(self, "current_print_mode", "optical")

        from path_utils import normalize_path
        session_photos = []
        for p in self.photos:
            session_photos.append({
                "id": p["id"],
                "path": normalize_path(p["path"]),
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
            "photos": session_photos
        }

        ok = session_cache_manager.save_session_file(file_path, session_data)
        if ok:
            fn = os.path.basename(file_path)
            self.set_app_status(f"已成功保存会话: {fn}", timeout_ms=3500)
            self._session_is_dirty = False
            self._is_temp_session = False
            self._update_window_title()
            config_manager.add_recent_session(file_path)
            config_manager.save_last_session_path(file_path)
            self._update_recent_menu()
            return True
        else:
            self.set_app_status("保存会话失败", timeout_ms=3500)
            return False

    def action_open_session(self):
        """Item 9 & 11: Loads and restores session from a .sdss file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开暗房会话工程", "", "SpektraDarkroom 会话工程 (*.sdss);;所有文件 (*.*)"
        )
        if file_path:
            self._open_session_by_path(file_path)

    def _open_session_by_path(self, file_path):
        if not file_path or not os.path.exists(file_path):
            self.set_app_status("会话文件不存在或已被移除", timeout_ms=3500)
            return

        self.set_app_status("正在加载会话工程...", spinning=True)
        session_data = session_cache_manager.load_session_file(file_path)
        if not session_data or "photos" not in session_data:
            self.set_app_status("会话文件格式无效或无法读取", timeout_ms=3500)
            return

        # 1. Clean previous session photos and workspace completely to prevent photo stacking
        if hasattr(self, 'engine'):
            self.engine.clear_all_photos()
        self.photos.clear()
        self.active_photo_id = None
        self._current_photo_path = None
        self.filmstrip.set_photos([], None)
        self.canvas.set_image(None)

        from path_utils import normalize_path
        auto_temp = config_manager.get_auto_saved_session_path()
        is_temp = (normalize_path(file_path).lower() == normalize_path(auto_temp).lower())
        self._is_temp_session = is_temp
        if is_temp:
            self._current_session_path = None
        else:
            self._current_session_path = normalize_path(file_path)

        photo_entries = session_data["photos"]
        files = [p["path"] for p in photo_entries if os.path.exists(p.get("path", ""))]
        if not files:
            self.set_app_status("会话中引用的照片文件在本地均不存在", timeout_ms=3500)
            return

        # Item 13: Build lookup map for session metadata restoration
        session_map = {}
        for s_p in photo_entries:
            fp = s_p.get("path")
            if fp:
                session_map[normalize_path(fp)] = s_p
        self._pending_session_map = session_map

        active_path = session_data.get("active_photo_path")
        self._restore_saved_session(files, active_file=active_path)

        for s_p in photo_entries:
            file_p = s_p.get("path")
            if not file_p:
                continue
            norm_fp = normalize_path(file_p)
            params = s_p.get("params") or {}
            film = s_p.get("film_stock", "none")
            paper = s_p.get("paper_stock", "none")
            p_mode = s_p.get("print_mode", "optical")
            b_ev = s_p.get("base_ev", 0.0)
            is_dirty = bool(s_p.get("is_dirty", False))
            is_edited = bool(s_p.get("is_edited", False))
            target_p = next((p for p in self.photos if normalize_path(p.get("path", "")) == norm_fp), None)
            if target_p:
                target_p["film_profile"] = film
                target_p["paper_profile"] = paper
                target_p["print_mode"] = p_mode
                target_p["base_ev"] = b_ev
                target_p["params"] = dict(params)
                target_p["is_dirty"] = is_dirty
                target_p["is_edited"] = is_edited
                if is_dirty or is_edited:
                    self.filmstrip.set_photo_dirty(target_p["id"], True)
            if file_p and params:
                sdc_manager.save_sdc(file_p, film, paper, params)

        # Re-activate active photo if present to ensure canvas LUT and sliders are synced
        if self.active_photo_id:
            curr_id = self.active_photo_id
            self.active_photo_id = None
            self.on_switch_photo(curr_id)

        if not is_temp:
            config_manager.add_recent_session(file_path)
            config_manager.save_last_session_path(file_path)
            self._update_recent_menu()
            self._session_is_dirty = any(p.get("is_dirty", False) for p in self.photos)
        else:
            self._session_is_dirty = any(p.get("is_dirty", False) or p.get("is_edited", False) for p in self.photos)

        self._update_window_title()

        fn = "未命名会话" if is_temp else os.path.basename(file_path)
        self.set_app_status(f"已恢复会话工程: {fn}", timeout_ms=3500)

    def action_quick_export(self, target_ids=None):
        """Item 4: Quick export selected or active photo(s) using remembered last export settings."""
        last_cfg = config_manager.get_last_export_settings()
        if not last_cfg:
            self.set_app_status("未记录上次导出参数，请先执行一次常规导出", timeout_ms=3000)
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
            task_params["base_ev"] = float(p.get("base_ev", 0.0))

            pix = None
            if p.get("thumbnail_rgb") is not None:
                trgb = p["thumbnail_rgb"]
                th, tw = trgb.shape[:2]
                qimg = QImage(trgb.data, tw, th, tw * 3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg)

            tid = f"task_{uuid.uuid4().hex[:8]}"
            task = ExportTask(tid, os.path.basename(src), src, out_file, task_params, thumbnail=pix)
            self.export_queue.enqueue(task)
            enqueued_count += 1

        self.set_app_status(f"⚡ 快速导出：已将 {enqueued_count} 张底片加入冲印队列", spinning=True, timeout_ms=3000, is_export_link=True)

    def push_undo_state(self):
        state = {
            "film": self.current_film_stock,
            "paper": self.current_paper_stock,
            "params": dict(self.current_params)
        }
        # Avoid pushing duplicate states that match the top of the undo stack
        if self._undo_stack:
            top = self._undo_stack[-1]
            if top["film"] == state["film"] and top["paper"] == state["paper"] and top["params"] == state["params"]:
                return

        self._undo_stack.append(state)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        curr_state = {
            "film": self.current_film_stock,
            "paper": self.current_paper_stock,
            "params": dict(self.current_params)
        }
        # Pop from undo stack until we find a state genuinely different from curr_state
        prev = None
        while self._undo_stack:
            cand = self._undo_stack.pop()
            if cand["film"] != curr_state["film"] or cand["paper"] != curr_state["paper"] or cand["params"] != curr_state["params"]:
                prev = cand
                break

        if prev is not None:
            self._redo_stack.append(curr_state)
            self._apply_history_state(prev)
            self._mark_current_photo_dirty()

    def redo(self):
        if not self._redo_stack:
            return
        curr_state = {
            "film": self.current_film_stock,
            "paper": self.current_paper_stock,
            "params": dict(self.current_params)
        }
        nxt = self._redo_stack.pop()
        self._undo_stack.append(curr_state)
        self._apply_history_state(nxt)
        self._mark_current_photo_dirty()

    def action_select_all_photos(self):
        """Ctrl+A handler: Select all photos in the filmstrip."""
        if hasattr(self, 'filmstrip'):
            self.filmstrip.select_all_photos()
            sel_count = len(self.filmstrip.get_selected_photo_ids())
            if sel_count > 1:
                self._is_multi_selecting = True
                self.set_app_status(f"已全选 {sel_count} 张底片", timeout_ms=0)
            elif sel_count == 1:
                self._is_multi_selecting = False
                self.restore_default_status()

    def action_copy_params(self):
        """Ctrl+C handler: Copy all physical darkroom parameters from the currently viewed photo."""
        if not self._current_photo_path:
            return
        self._copied_darkroom_profile = {
            "film": self.current_film_stock,
            "paper": self.current_paper_stock,
            "params": dict(self.current_params)
        }
        self.set_app_status("已复制暗房所有物理参数 (胶卷/相纸/曝光/冲印/颗粒)", timeout_ms=2500)

    def action_paste_params(self):
        """Ctrl+V handler: Paste copied physical darkroom parameters to all selected photos."""
        if not getattr(self, '_copied_darkroom_profile', None):
            self.set_app_status("剪贴板中无复制的暗房参数，请先按 Ctrl+C 复制", timeout_ms=2500)
            return

        copied = self._copied_darkroom_profile
        copied_film = copied["film"]
        copied_paper = copied["paper"]
        copied_params = dict(copied["params"])

        selected_ids = self.filmstrip.get_selected_photo_ids() if hasattr(self, 'filmstrip') else []
        if not selected_ids and self.active_photo_id:
            selected_ids = [self.active_photo_id]

        if not selected_ids:
            return

        # If current active photo is among targets, push undo state first
        if self.active_photo_id in selected_ids:
            self.push_undo_state()

        affected_count = 0
        for pid in selected_ids:
            photo = next((p for p in self.photos if p["id"] == pid), None)
            if not photo:
                continue

            target_params = dict(copied_params)
            # Adapt white balance relative to target photo baseline EXIF if not custom
            if not target_params.get("custom_wb", False):
                exif = photo.get("meta", {})
                target_params["color_temp"] = float(exif.get("color_temp", 5500.0))
                target_params["tint"] = float(exif.get("tint", 1.0))

            photo["film_profile"] = copied_film
            photo["paper_profile"] = copied_paper
            photo["params"] = target_params
            photo["is_dirty"] = True

            # Save to SDC sidecar immediately
            try:
                sdc_manager.save_sdc(
                    photo["path"],
                    copied_film,
                    copied_paper,
                    target_params
                )
            except Exception:
                pass

            self.filmstrip.set_photo_dirty(pid, True)
            affected_count += 1

        # If current active photo is in selected targets, apply immediately to viewport
        if self.active_photo_id in selected_ids:
            self.current_film_stock = copied_film
            self.current_paper_stock = copied_paper
            self.current_params = dict(copied_params)
            self._update_all_sliders_from_params()
            self.on_switch_film(self.current_film_stock, push_undo=False)
            self.on_switch_paper(self.current_paper_stock, push_undo=False)
            self.canvas.update_params(**self.current_params)
            self._schedule_histogram_update()
            self._mark_current_photo_dirty()

        self.set_app_status(f"已将暗房参数粘贴至 {affected_count} 张选中的底片", timeout_ms=3000)

    def _apply_history_state(self, state):
        self.current_film_stock = state["film"]
        self.current_paper_stock = state["paper"]
        self.current_params = dict(state["params"])
        self._update_all_sliders_from_params()
        self.on_switch_film(self.current_film_stock, push_undo=False)
        self.on_switch_paper(self.current_paper_stock, push_undo=False)
        self.canvas.update_params(**self.current_params)
        self._schedule_histogram_update()

    def _init_histogram_widget(self):
        pass

    def _schedule_histogram_update(self):
        if hasattr(self, '_hist_timer') and not self._hist_timer.isActive():
            self._hist_timer.start(40)

    def _compute_and_update_histogram(self):
        if not hasattr(self, 'histogram_widget'):
            return
        if self._current_photo_thumb is None:
            self.histogram_widget.clear()
            return
        try:
            lut_3d = self.engine.get_3d_lut(
                self.current_film_stock,
                self.current_paper_stock,
                lut_size=17,
                params_dict=self.current_params
            )
            graded = self.engine.apply_lut_to_rgb(self._current_photo_thumb, lut_3d)
            ev = float(self.current_params.get("exposure_ev", 0.0))
            if abs(ev) > 0.01:
                mult = 2.0 ** ev
                graded = np.clip(graded.astype(np.float32) * mult, 0, 255).astype(np.uint8)
            self.histogram_widget.update_from_rgb(graded)
        except Exception:
            pass

    def _on_slider_live(self, param_key, val):
        self.current_params[param_key] = val
        if param_key in ("color_temp", "tint"):
            self.current_params["custom_wb"] = True
        self.canvas.update_params(**{param_key: val})
        self._schedule_histogram_update()
        self._mark_current_photo_dirty()

    def _on_combo_changed(self, param_key, val):
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
        elif param_key == "enlarger_illuminant":
            self._refresh_lut()
        self._schedule_histogram_update()
        self.push_undo_state()
        self._mark_current_photo_dirty()
        if self._current_photo_path:
            try:
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    self.current_paper_stock,
                    self.current_params
                )
            except Exception:
                pass

    def _refresh_lut(self):
        lut_3d = self.engine.get_3d_lut(
            self.current_film_stock,
            self.current_paper_stock,
            lut_size=33,
            params_dict=self.current_params
        )
        self.canvas.set_lut(lut_3d)

    def _on_slider_committed(self):
        self.push_undo_state()
        self._mark_current_photo_dirty()
        if self._current_photo_path:
            try:
                sdc_manager.save_sdc(
                    self._current_photo_path,
                    self.current_film_stock,
                    self.current_paper_stock,
                    self.current_params
                )
            except Exception:
                pass

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
        self.set_app_status(f"已校准至 18% 中灰基准测光 (0.00 EV，底片物理基准: {base_ev:+.2f} EV)", timeout_ms=3000)

    def reset_track(self, track_name):
        def_temp = getattr(self.slider_temp, "default_val", 5500.0)
        def_tint = getattr(self.slider_tint, "default_val", 1.0)
        defaults = {
            "exposure": {
                "exposure_ev": 0.0, "color_temp": def_temp, "tint": def_tint,
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
            "texture": {
                "halation": 0.5, "halation_bounces": 2, "halation_decay": 0.5, "halation_boost": 0.0,
                "grain": 0.4, "grain_size": 1.0, "grain_cloud_blur": 1.0
            },
            "optical": {
                "halation": 0.5, "halation_bounces": 2, "halation_decay": 0.5, "halation_boost": 0.0,
                "grain": 0.4, "grain_size": 1.0, "grain_cloud_blur": 1.0
            }
        }
        if track_name in defaults:
            for k, v in defaults[track_name].items():
                self.current_params[k] = v
                self.canvas.update_params(**{k: v})
            if "enlarger_illuminant" in defaults[track_name]:
                self._refresh_lut()
            self._update_all_sliders_from_params()
            self._on_slider_committed()
            self._schedule_histogram_update()

    def reset_all_params(self):
        def_temp = getattr(self.slider_temp, "default_val", 5500.0)
        def_tint = getattr(self.slider_tint, "default_val", 1.0)
        defaults = {
            "exposure_ev": 0.0, "color_temp": def_temp, "tint": def_tint,
            "film_format_mm": 35.0,
            "diffusion_family": "none", "diffusion_strength": 0.0, "diffusion_warmth": 0.0,
            "dir_amount": 1.0, "dir_interlayer": 1.0, "dir_samelayer": 1.0,
            "enlarger_illuminant": "TH-KG3",
            "enlarger_cyan": 0.0, "enlarger_magenta": 0.0, "enlarger_yellow": 0.0,
            "print_exposure": 1.0, "pre_flash": 0.0,
            "morph_gamma": 1.0, "developer_exhaustion": 0.0,
            "halation": 0.5, "halation_bounces": 2, "halation_decay": 0.5, "halation_boost": 0.0,
            "grain": 0.4, "grain_size": 1.0, "grain_cloud_blur": 1.0,
            "custom_wb": False
        }
        self.current_params.update(defaults)
        self.canvas.update_params(**defaults)
        self._refresh_lut()
        self._update_all_sliders_from_params()
        self._on_slider_committed()
        self._schedule_histogram_update()

    def _update_all_sliders_from_params(self):
        self._is_updating_sliders = True
        try:
            p = self.current_params
            if hasattr(self, 'slider_ev'): self.slider_ev.set_value(p.get("exposure_ev", 0.0))
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

            if hasattr(self, 'slider_halation'): self.slider_halation.set_value(p.get("halation", 0.5))
            if hasattr(self, 'slider_halation_bounces'): self.slider_halation_bounces.set_value(p.get("halation_bounces", 2))
            if hasattr(self, 'slider_halation_decay'): self.slider_halation_decay.set_value(p.get("halation_decay", 0.5))
            if hasattr(self, 'slider_halation_boost'): self.slider_halation_boost.set_value(p.get("halation_boost", 0.0))
            if hasattr(self, 'slider_grain'): self.slider_grain.set_value(p.get("grain", 0.4))
            if hasattr(self, 'slider_grain_size'): self.slider_grain_size.set_value(p.get("grain_size", 1.0))
            if hasattr(self, 'slider_grain_cloud'): self.slider_grain_cloud.set_value(p.get("grain_cloud_blur", 1.0))
        finally:
            self._is_updating_sliders = False

    def on_switch_film(self, stock_id, push_undo=True):
        if push_undo and stock_id != self.current_film_stock:
            self.push_undo_state()
            self._mark_current_photo_dirty()
        self.current_film_stock = stock_id

        # Intelligent pairing for bypass "none" stock
        if stock_id == "none":
            self.current_paper_stock = "none"
            for pid, card in self.paper_cards.items():
                card.set_active(pid == "none")
        elif self.current_paper_stock == "none":
            self.current_paper_stock = "kodak_2383"
            for pid, card in self.paper_cards.items():
                card.set_active(pid == "kodak_2383")

        for sid, card in self.film_cards.items():
            card.set_active(sid == stock_id)

        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=self.current_params)
        self.canvas.set_lut(lut_3d)
        self._schedule_histogram_update()
        if push_undo and self._current_photo_path:
            self._refresh_card_previews(paper_only=True)

    def on_switch_print_mode(self, mode, push_undo=True):
        if mode not in ("optical", "scan"):
            mode = "optical"
        if push_undo and mode != getattr(self, "current_print_mode", "optical"):
            self.push_undo_state()
            self._mark_current_photo_dirty()

        self.current_print_mode = mode
        self.current_params["print_mode"] = mode

        # Save to current photo
        if self._current_photo_path:
            photo = next((p for p in self.photos if p.get("path") == self._current_photo_path), None)
            if photo:
                photo["print_mode"] = mode

        lut_params = dict(self.current_params)
        lut_params["print_mode"] = mode
        lut_3d = self.engine.get_3d_lut(self.current_film_stock, self.current_paper_stock, lut_size=33, params_dict=lut_params)
        self.canvas.set_lut(lut_3d)
        self._schedule_histogram_update()
        if push_undo:
            mode_name = "相纸印相模式 (Optical Print)" if mode == "optical" else "数码扫描模式 (Film Scan)"
            self.set_app_status(f"已切换冲印模式: {mode_name}", timeout_ms=3000)

    def on_switch_paper(self, stock_id, push_undo=True):
        if push_undo and stock_id != self.current_paper_stock:
            self.push_undo_state()
            self._mark_current_photo_dirty()
        self.current_paper_stock = stock_id

        # Item 9: Selecting "none" (数码扫描 Film Scan) engages scan mode; selecting physical paper engages Optical Print mode
        if stock_id == "none":
            self.current_print_mode = "scan"
            self.current_params["print_mode"] = "scan"
        else:
            self.current_print_mode = "optical"
            self.current_params["print_mode"] = "optical"

        for sid, card in self.paper_cards.items():
            card.set_active(sid == stock_id)

        # Save to current photo
        if self._current_photo_path:
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
            self._refresh_card_previews(film_only=True)
        if push_undo:
            mode_desc = "数码扫描 (Film Scan)" if stock_id == "none" else f"相纸印相 ({stock_id})"
            self.set_app_status(f"已选用: {mode_desc}", timeout_ms=3000)

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
        self.set_app_status("对比原片中 (松开恢复)", timeout_ms=0)

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
        files, _ = QFileDialog.getOpenFileNames(self, "导入照片 / RAW 负片", "", filt)
        if files:
            self._import_files_list(files)

    def action_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择导入照片文件夹")
        if folder:
            valid_exts = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".dng", ".tif", ".tiff", ".png"}
            files = []
            for root, _, fnames in os.walk(folder):
                for f in fnames:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        files.append(os.path.join(root, f))
            if files:
                self._import_files_list(files)
            else:
                QMessageBox.information(self, "提示", "所选文件夹中未找到受支持的图像文件。")

    def open_single_file(self, file_path):
        if os.path.exists(file_path):
            self._import_files_list([file_path])

    def _import_files_list(self, file_paths):
        existing_paths = {p["path"] for p in self.photos}
        new_paths = []
        dup_names = []

        for fp in file_paths:
            norm_p = os.path.abspath(fp)
            if norm_p in existing_paths:
                dup_names.append(os.path.basename(norm_p))
            else:
                new_paths.append(norm_p)

        # Item 12: Duplicate detection dialog
        if dup_names:
            dlg = DuplicateFilesDialog(dup_names, self)
            dlg.exec()

        if not new_paths:
            return

        # If this is a new batch of photos without an active named session, mark as temporary session
        if not self.photos or getattr(self, "_is_temp_session", False):
            self._is_temp_session = True
            self._current_session_path = None

        # Always mark session dirty whenever new photos are imported
        self._session_is_dirty = True
        self._update_window_title()

        # 1. Immediately switch to edit workspace (Option B: instant switch)
        self.stack.setCurrentIndex(1)
        self.btn_title_export.setVisible(True)

        # 2. Add to recent files
        for p in new_paths:
            config_manager.add_recent_file(p)

        # 3. Start progress spinner & status text
        self._is_importing = True
        self._last_import_status_time = 0.0
        total_count = len(new_paths)
        self.set_app_status(f"正在导入底片库 (0/{total_count})...", spinning=True, timeout_ms=0)

        # 4. Asynchronous streaming import worker
        self._import_worker = BatchImportWorker(new_paths, self.engine, start_idx=len(self.photos), parent=self)
        self._import_worker.photoLoaded.connect(self._on_single_photo_imported)
        self._import_worker.importFinished.connect(self._on_all_photos_imported)
        self._import_worker.start()

    def _on_single_photo_imported(self, photo_entry, current_idx, total_count):
        # Item 13: Restore session parameters if loading via session map
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

        self.photos.append(photo_entry)
        is_first = (len(self.photos) == 1) or (self.active_photo_id is None)

        # Incrementally stream into bottom filmstrip
        self.filmstrip.append_photo(photo_entry, is_active=is_first)
        if photo_entry.get("is_dirty") or photo_entry.get("is_edited"):
            self.filmstrip.set_photo_dirty(photo_entry["id"], True)

        # If this is the first photo, display it immediately on the canvas!
        if is_first:
            self.on_switch_photo(photo_entry["id"])

        # Throttled status updates (every 120ms or first/last item) to eliminate label shaking/jitter
        import time
        now = time.time()
        last_t = getattr(self, "_last_import_status_time", 0.0)
        if (now - last_t >= 0.12) or current_idx == total_count or current_idx == 1:
            self._last_import_status_time = now
            self.set_app_status(f"正在导入底片库 ({current_idx}/{total_count})...", spinning=True, timeout_ms=0)

    def _on_all_photos_imported(self, total_loaded):
        self._is_importing = False
        self._pending_session_map = None
        self._update_recent_menu()
        target_path = getattr(self, "_target_initial_active_path", None)
        if target_path:
            self._target_initial_active_path = None
            target_entry = next((p for p in self.photos if p.get("path") == target_path), None)
            if target_entry:
                self.on_switch_photo(target_entry["id"])
        self.set_app_status(f"底片库导入完成，共 {total_loaded} 张", spinning=False, timeout_ms=5000)

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
            pass

    def on_switch_photo(self, photo_id):
        # Auto-reset compare / split-screen mode to normal edit view when switching to another photo
        self.set_view_mode(0)

        self._is_switching_photo = True
        try:
            # 1. Update in-memory state for previous photo
            if self.active_photo_id:
                prev_photo = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
                if prev_photo:
                    prev_photo["params"] = dict(self.current_params)
                    prev_photo["film_profile"] = self.current_film_stock
                    prev_photo["paper_profile"] = self.current_paper_stock
                    prev_photo["is_dirty"] = self._photo_is_dirty

            # 2. Find photo entry
            photo = next((p for p in self.photos if p["id"] == photo_id), None)
            if not photo:
                return

            self.active_photo_id = photo_id
            self._current_photo_path = photo["path"]
            self.filmstrip.set_active_photo(photo_id)

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
                self.current_film_stock = photo.get("film_profile", self.current_film_stock)
                self.current_paper_stock = photo.get("paper_profile", self.current_paper_stock)
                self.current_print_mode = photo.get("print_mode", photo["params"].get("print_mode", "optical"))
                self.current_params.clear()
                self.current_params.update(photo["params"])
                self.current_params["print_mode"] = self.current_print_mode
                self._photo_is_dirty = photo.get("is_dirty", False)
            else:
                sdc = sdc_manager.load_sdc(self._current_photo_path)
                if sdc:
                    self.current_film_stock = sdc.get("film_profile", self.current_film_stock)
                    self.current_paper_stock = sdc.get("paper_profile", self.current_paper_stock)
                    self.current_print_mode = sdc.get("print_mode", sdc.get("params", {}).get("print_mode", "optical"))
                    loaded_params = sdc.get("params", {})
                    self.current_params.clear()
                    self.current_params.update(loaded_params)
                    self.current_params["print_mode"] = self.current_print_mode
                    if "color_temp" not in loaded_params or (loaded_params.get("color_temp") == 5500.0 and base_temp != 5500.0 and not loaded_params.get("custom_wb")):
                        self.current_params["color_temp"] = base_temp
                    if "tint" not in loaded_params or (loaded_params.get("tint") == 1.0 and base_tint != 1.0 and not loaded_params.get("custom_wb")):
                        self.current_params["tint"] = base_tint
                else:
                    self.current_film_stock = "none"
                    self.current_paper_stock = "none"
                    self.current_print_mode = "optical"
                    defaults = {
                        "exposure_ev": 0.0, "color_temp": base_temp, "tint": base_tint,
                        "enlarger_cyan": 0.0, "enlarger_magenta": 0.0, "enlarger_yellow": 0.0,
                        "print_exposure": 1.0, "pre_flash": 0.0, "halation": 0.5, "grain": 0.4,
                        "print_mode": "optical"
                    }
                    self.current_params.clear()
                    self.current_params.update(defaults)
                self._photo_is_dirty = False

            self.filmstrip.set_photo_dirty(photo_id, self._photo_is_dirty)
            self._update_window_title()

            # 5. Upload photo to OpenGL canvas texture (lazy load preview or RAW if needed)
            if photo.get("float_img") is None:
                cached_prev = session_cache_manager.get_photo_preview(photo["path"])
                if cached_prev is not None:
                    photo["float_img"] = cached_prev
                else:
                    fn = photo.get("filename", os.path.basename(photo["path"]))
                    if not getattr(self, "_is_importing", False) and not getattr(self, "_is_multi_selecting", False):
                        self.set_app_status(f"正在载入底片并解析 RAW ({fn})...", spinning=True, timeout_ms=0)
                    QApplication.processEvents()
                    res = self.engine.load_image(photo["path"])
                    if res.get("success"):
                        photo["float_img"] = res.get("float_img")
                        if photo.get("thumbnail_rgb") is None:
                            photo["thumbnail_rgb"] = res.get("thumbnail_rgb")
                        if "base_ev" in res:
                            photo["base_ev"] = res["base_ev"]
                        session_cache_manager.save_photo_preview(photo["path"], photo["float_img"])
                    if not getattr(self, "_is_importing", False) and not getattr(self, "_is_multi_selecting", False):
                        self.set_app_status(f"底片已载入: {fn}", spinning=False, timeout_ms=3000)

            if photo.get("float_img") is not None:
                m_w = photo.get("width") or (photo.get("meta", {}).get("width") if isinstance(photo.get("meta"), dict) else None)
                m_h = photo.get("height") or (photo.get("meta", {}).get("height") if isinstance(photo.get("meta"), dict) else None)
                self.canvas.set_image(photo["float_img"], master_width=m_w, master_height=m_h)
            self.canvas.update_params(base_temp=base_temp, base_tint=base_tint, **self.current_params)
            self._update_all_sliders_from_params()

            # 6. Apply print mode, film & paper LUT
            self.on_switch_print_mode(self.current_print_mode, push_undo=False)
            self.on_switch_film(self.current_film_stock, push_undo=False)
            if self.current_print_mode == "optical":
                self.on_switch_paper(self.current_paper_stock, push_undo=False)

            # 7. Update status text with 1:1 image native pixel scale percentage
            self._update_photo_info_display()

            # 8. Update live 2:3 thumbnails on Film & Paper cards with real film emulation
            self._current_photo_thumb = photo.get("thumbnail_rgb")
            float_img = photo.get("float_img")
            if float_img is None and hasattr(self, 'engine'):
                float_img = getattr(self.engine, 'raw_preview', None)
            if float_img is not None:
                lh, lw = float_img.shape[:2]
                lscale = 120.0 / float(max(lh, lw))
                ltw = max(1, int(round(lw * lscale)))
                lth = max(1, int(round(lh * lscale)))
                self._current_photo_linear = cv2.resize(float_img, (ltw, lth), interpolation=cv2.INTER_AREA)
            else:
                self._current_photo_linear = None

            thumb_to_grade = self._current_photo_linear if self._current_photo_linear is not None else self._current_photo_thumb
            self._update_card_live_thumbnails(thumb_to_grade)
            self._schedule_histogram_update()

            # 9. Reset undo/redo history for newly switched photo with initial baseline state
            self._undo_stack = [{
                "film": self.current_film_stock,
                "paper": self.current_paper_stock,
                "params": dict(self.current_params)
            }]
            self._redo_stack.clear()

            # Synchronize engine state
            if hasattr(self, 'engine'):
                self.engine.current_file_path = photo["path"]
                self.engine.active_photo_id = photo_id
        finally:
            self._is_switching_photo = False

    def _update_card_live_thumbnails(self, thumb_data, do_films=True, do_papers=True):
        if thumb_data is None:
            return

        from app_core import prophoto_to_srgb
        if thumb_data.dtype != np.uint8:
            base_srgb = (prophoto_to_srgb(thumb_data) * 255.0).astype(np.uint8)
        else:
            base_srgb = thumb_data

        h, w, c = base_srgb.shape
        qimg = QImage(base_srgb.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
        base_pix = QPixmap.fromImage(qimg)

        # 1. Immediately show base thumbnail so cards display without delay
        if do_films:
            for card in self.film_cards.values():
                card.set_thumbnail(base_pix)
        if do_papers:
            for card in self.paper_cards.values():
                card.set_thumbnail(base_pix)

        # 2. Stop any existing background worker and disconnect signals to avoid race condition
        if getattr(self, '_thumb_worker', None) is not None and self._thumb_worker.isRunning():
            self._thumb_worker.stop()
            try:
                self._thumb_worker.film_ready.disconnect()
                self._thumb_worker.paper_ready.disconnect()
            except Exception:
                pass
            self._thumb_worker.wait(300)

        # 3. Dedicated QThread worker applying 3D LUT asynchronously in linear space
        self._thumb_worker = StockThumbnailWorker(
            self.engine, thumb_data,
            list(self.film_cards.keys()),
            list(self.paper_cards.keys()),
            self.current_film_stock,
            self.current_paper_stock,
            do_films=do_films,
            do_papers=do_papers,
            parent=self
        )
        self._thumb_worker.film_ready.connect(self._on_film_thumb_ready)
        self._thumb_worker.paper_ready.connect(self._on_paper_thumb_ready)
        self._thumb_worker.start()

    def _refresh_card_previews(self, film_only=False, paper_only=False):
        thumb_data = getattr(self, '_current_photo_linear', None)
        if thumb_data is None:
            thumb_data = self._current_photo_thumb
        if thumb_data is None:
            return
        do_films = not paper_only
        do_papers = not film_only
        self._update_card_live_thumbnails(thumb_data, do_films=do_films, do_papers=do_papers)

    def _on_film_thumb_ready(self, stock_id, pixmap):
        card = self.film_cards.get(stock_id)
        if card:
            card.set_thumbnail(pixmap)

    def _on_paper_thumb_ready(self, stock_id, pixmap):
        card = self.paper_cards.get(stock_id)
        if card:
            card.set_thumbnail(pixmap)

    def on_remove_photo(self, photo_id):
        self.photos = [p for p in self.photos if p["id"] != photo_id]
        if hasattr(self, 'engine'):
            self.engine.remove_photo(photo_id)
        if not self.photos:
            self.action_clear_photos()
        else:
            next_id = self.photos[-1]["id"]
            self.filmstrip.set_photos(self.photos, next_id)
            self.on_switch_photo(next_id)

    def on_remove_multiple_photos(self, photo_ids):
        id_set = set(photo_ids)
        self.photos = [p for p in self.photos if p["id"] not in id_set]
        if hasattr(self, 'engine'):
            for pid in photo_ids:
                self.engine.remove_photo(pid)
        if not self.photos:
            self.action_clear_photos()
        else:
            next_id = self.photos[-1]["id"]
            self.filmstrip.set_photos(self.photos, next_id)
            self.on_switch_photo(next_id)

    def on_clear_sdc_config(self, photo_ids):
        """Delete .sdc configuration sidecar and reset adjustments to baseline defaults."""
        id_set = set(photo_ids)
        active_cleared = False
        for p in self.photos:
            if p["id"] in id_set:
                p_path = p.get("path")
                if p_path:
                    sdc_manager.delete_sdc(p_path)
                p.pop("params", None)
                p["film_profile"] = "none"
                p["paper_profile"] = "none"
                p["print_mode"] = "scan"
                p["is_dirty"] = False
                p["is_edited"] = False
                self.filmstrip.set_photo_dirty(p["id"], False)
                if p["id"] == self.active_photo_id:
                    active_cleared = True

        if active_cleared and self.active_photo_id:
            # Item 11: Thoroughly reset active photo editing state, sliders, stocks, and canvas
            self._photo_is_dirty = False
            self.reset_all_params()
            self.on_switch_film("none", push_undo=False)
            self.on_switch_paper("none", push_undo=False)
            active_p = next((p for p in self.photos if p["id"] == self.active_photo_id), None)
            if active_p:
                active_p["params"] = dict(self.current_params)
                active_p["film_profile"] = "none"
                active_p["paper_profile"] = "none"
                active_p["print_mode"] = "scan"
                active_p["is_dirty"] = False
                active_p["is_edited"] = False
            self.filmstrip.set_photo_dirty(self.active_photo_id, False)
            self._session_is_dirty = any(p.get("is_dirty", False) for p in self.photos)
            self._update_window_title()
            self.set_app_status("已清除暗房配置并重置为原生底片", timeout_ms=3000)

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
        self._current_session_path = None
        self._session_is_dirty = False
        self._is_temp_session = False

        config_manager.clear_auto_saved_session()
        config_manager.save_last_session_path(None)
        config_manager.save_session_state([], None)

        # Retain darkroom workspace, do not return to start page
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
        for card in self.film_cards.values():
            card.set_thumbnail(None)
        for card in self.paper_cards.values():
            card.set_thumbnail(None)
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
        task = ExportTask(tid, os.path.basename(self._current_photo_path),
                          self._current_photo_path, out_path, export_params, thumbnail=pix)
        self.export_queue.enqueue(task)
        self.set_app_status("已加入导出队列，正在后台冲印...", spinning=True, is_export_link=True)

    def action_export_darkroom_lut(self):
        """Export full darkroom physical color profile into standard 3D .cube LUT."""
        suggested_name = f"{self.current_film_stock}_{self.current_paper_stock}.cube"
        out_file, _ = QFileDialog.getSaveFileName(
            self, "导出当前暗房 3D LUT", suggested_name, "3D LUT (*.cube)"
        )
        if not out_file:
            return

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

            export_params = {
                "source_path": file_p,
                "film_stock": film,
                "paper_stock": paper,
                "print_mode": photo.get("print_mode", params.get("print_mode", "optical")),
                "base_ev": float(photo.get("base_ev", 0.0)),
                "base_temp": b_temp,
                "base_tint": b_tint,
                "exposure_ev": params.get("exposure_ev", 0.0),
                "color_temp": params.get("color_temp", b_temp),
                "tint": params.get("tint", b_tint),
                "enlarger_cyan": params.get("enlarger_cyan", 0.0),
                "enlarger_magenta": params.get("enlarger_magenta", 0.0),
                "enlarger_yellow": params.get("enlarger_yellow", 0.0),
                "print_exposure": params.get("print_exposure", 1.0),
                "pre_flash": params.get("pre_flash", 0.0),
                "halation": params.get("halation", 0.5),
                "grain": params.get("grain", 0.4),
                "grain_size": params.get("grain_size", 1.0),
                "grain_cloud_blur": params.get("grain_cloud_blur", 1.0),
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
            task = ExportTask(tid, os.path.basename(file_p), file_p, out_target, export_params, thumbnail=pix)
            self.export_queue.enqueue(task)
            enqueued_count += 1

        self.set_app_status(f"已将 {enqueued_count} 张底片加入冲印队列", spinning=True, is_export_link=True)

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

        menu.addSeparator()
        act_lib = menu.addAction("胶卷库管理...")
        icon_film = os.path.join(get_resource_dir(), "icons", "dlg_film.png")
        if os.path.exists(icon_film):
            act_lib.setIcon(QIcon(icon_film))
        act_lib.triggered.connect(lambda: self.open_stock_manager("film"))
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

        menu.addSeparator()
        act_lib = menu.addAction("相纸库管理...")
        icon_paper = os.path.join(get_resource_dir(), "icons", "dlg_paper.png")
        if os.path.exists(icon_paper):
            act_lib.setIcon(QIcon(icon_paper))
        act_lib.triggered.connect(lambda: self.open_stock_manager("paper"))
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
        dlg = PreferencesDialog(parent=self)
        if dlg.exec():
            if hasattr(self, 'canvas'):
                self.canvas.update()

    def _export_film_lut(self, stock_id):
        out_file, _ = QFileDialog.getSaveFileName(self, "导出胶卷 3D LUT", f"{stock_id}.cube", "3D LUT (*.cube)")
        if out_file:
            succ = export_stock_lut(self.engine, stock_id, out_file, is_film=True)
            if succ:
                self._show_info_dialog("导出完成", f"胶卷预设已成功导出至:\n{out_file}")
            else:
                self._show_warning_dialog("导出失败", f"导出胶卷预设 {stock_id} 失败，请检查文件写入权限。")

    def _export_paper_lut(self, stock_id):
        out_file, _ = QFileDialog.getSaveFileName(self, "导出相纸 3D LUT", f"{stock_id}.cube", "3D LUT (*.cube)")
        if out_file:
            succ = export_stock_lut(self.engine, stock_id, out_file, is_film=False)
            if succ:
                self._show_info_dialog("导出完成", f"相纸预设已成功导出至:\n{out_file}")
            else:
                self._show_warning_dialog("导出失败", f"导出相纸预设 {stock_id} 失败，请检查文件写入权限。")

    def _import_film_lut(self):
        f, _ = QFileDialog.getOpenFileName(self, "导入胶卷物理配置", "", "JSON 配置 (*.json);;所有文件 (*.*)")
        if f:
            dest = self.engine.import_film_lut(f)
            self._on_stocks_changed()
            self._show_info_dialog("导入成功", f"胶卷物理配置已成功存入:\n{dest}")

    def _import_paper_lut(self):
        f, _ = QFileDialog.getOpenFileName(self, "导入相纸物理配置", "", "JSON 配置 (*.json);;所有文件 (*.*)")
        if f:
            dest = self.engine.import_paper_lut(f)
            self._on_stocks_changed()
            self._show_info_dialog("导入成功", f"相纸物理配置已成功存入:\n{dest}")

    def _remove_film_lut(self, stock_id):
        if self._show_question_dialog("确认移除", f"确定要从当前胶卷库中丢弃/移除预设 '{stock_id}' 吗？"):
            if stock_id in self.film_cards:
                card = self.film_cards.pop(stock_id)
                self.film_grid.removeWidget(card)
                card.deleteLater()
                self.reflow_grids()
            self._show_info_dialog("提示", f"已从当前预设库中移除 {stock_id}")

    def _remove_paper_lut(self, stock_id):
        if self._show_question_dialog("确认移除", f"确定要从当前相纸库中丢弃/移除预设 '{stock_id}' 吗？"):
            if stock_id in self.paper_cards:
                card = self.paper_cards.pop(stock_id)
                self.paper_grid.removeWidget(card)
                card.deleteLater()
                self.reflow_grids()
            self._show_info_dialog("提示", f"已从当前预设库中移除 {stock_id}")

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
        is_test_env = bool(os.environ.get("FAST_TEST")) or getattr(self, '_suppress_close_confirm', False)
        if getattr(self, '_session_is_dirty', False) and self.photos and not is_test_env:
            sess_name = os.path.basename(self._current_session_path) if self._current_session_path else "未命名会话"
            dialog = UnsavedSessionDialog(sess_name, self)
            res = dialog.exec()
            if res == UnsavedSessionDialog.CANCEL:
                event.ignore()
                return
            elif res == UnsavedSessionDialog.SAVE:
                self.action_save_session()
                if getattr(self, '_session_is_dirty', False):
                    event.ignore()
                    return
            # elif res == UnsavedSessionDialog.DISCARD: proceed to exit

        # Silently ensure any modified parameters are saved to .sdc
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
            logger.warning(f"Auto-saving .sdc on close: {e}")

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

            # Item 2: Remember active session path if saved on disk, or auto-save temporary session
            if self._current_session_path and os.path.exists(self._current_session_path) and not getattr(self, "_is_temp_session", False):
                config_manager.save_last_session_path(self._current_session_path)
            elif self.photos:
                self._save_auto_temp_session()
                config_manager.save_last_session_path(config_manager.get_auto_saved_session_path())
            else:
                config_manager.save_last_session_path(None)
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
