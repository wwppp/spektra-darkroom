import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPoint, QEvent
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen, QFontMetrics, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QMenu, QDialog, QTextEdit, QSizePolicy
)
import sdc_manager
import session_cache_manager
from session_cache_manager import is_dummy_or_corrupted_thumbnail, is_valid_thumbnail
from path_utils import get_icon_path, get_resource_dir
from ui.window_utils import apply_dark_titlebar, get_darkroom_menu_style, get_icon
from ui.smooth_scroll import SmoothScrollArea


class DuplicateFilesDialog(QDialog):
    """Dialog showing duplicate files that were skipped during import."""
    def __init__(self, duplicate_names, parent=None):
        super().__init__(parent)
        self.duplicate_names = list(duplicate_names or [])
        self.setWindowTitle("重复底片过滤提示")
        self.setFixedSize(540, 350)
        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setStyleSheet("""
            QDialog {
                background: #14151a;
                border: 1px solid #2b2e3b;
                border-radius: 8px;
            }
            QLabel {
                color: #e2e8f0;
                font-family: "Microsoft YaHei UI", sans-serif;
            }
            QTextEdit {
                background: #101115;
                color: #cbd5e1;
                border: 1px solid #282b37;
                border-radius: 6px;
                padding: 10px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 11.5px;
                line-height: 1.4;
            }
            QPushButton {
                background: #f59e0b;
                color: #111111;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 6px 24px;
                font-size: 12px;
                min-width: 72px;
            }
            QPushButton:hover {
                background: #fbbf24;
            }
            QPushButton:pressed {
                background: #d97706;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        # Header with icon and count
        hdr = QHBoxLayout()
        hdr.setSpacing(10)
        badge = QLabel("ℹ")
        badge.setFixedSize(28, 28)
        badge.setStyleSheet("""
            background: #261f14;
            border: 1px solid #f59e0b;
            border-radius: 6px;
            color: #f59e0b;
            font-size: 14px;
            font-weight: bold;
        """)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.addWidget(badge)

        count = len(self.duplicate_names)
        lbl_head = QLabel(f"已自动跳过 {count} 个已在暗房底片库中的重复文件：")
        lbl_head.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 13px;")
        hdr.addWidget(lbl_head, 1)
        layout.addLayout(hdr)

        # Files list
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setPlainText("\n".join(self.duplicate_names))
        layout.addWidget(self.txt, 1)

        # Footer
        btn_box = QHBoxLayout()
        lbl_hint = QLabel("底片库已保留原有底片，不会重复添加。")
        lbl_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        btn_box.addWidget(lbl_hint)
        btn_box.addStretch()

        btn_ok = QPushButton("确定")
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.clicked.connect(self.accept)
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)


class FilmstripItemWidget(QFrame):
    """Compact 35mm film negative frame:
    - 35mm Sprocket holes on top and bottom
    - Centered photo thumbnail keeping true original aspect ratio
    - Translucent filename overlay at the bottom
    - Multi-select border highlight
    """
    clicked = Signal(str, object)  # photo_id, mouse event for modifier keys
    removeRequested = Signal(str)
    contextMenuRequested = Signal(str, QPoint)

    def __init__(self, photo_data, is_active=False, is_selected=False, parent=None):
        super().__init__(parent)
        self.photo_id = photo_data.get("id", "")
        self.filename = photo_data.get("filename") or (os.path.basename(photo_data.get("path", "image")) if photo_data.get("path") else "image")
        from path_utils import normalize_path
        self.file_path = normalize_path(photo_data.get("path", ""))
        self.is_active = is_active
        self.is_selected = is_selected
        self.photo_data = photo_data

        self.setFixedSize(76, 82)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_tooltip()

        self._base_pixmap = None
        self._pixmap = None
        thumb_rgb = photo_data.get("thumbnail_rgb")

        if is_dummy_or_corrupted_thumbnail(thumb_rgb):
            raw_rgb = photo_data.get("raw_thumbnail_rgb")
            if is_valid_thumbnail(raw_rgb):
                thumb_rgb = raw_rgb
            else:
                raw_rgb = photo_data.get("raw_thumbnail")
                if is_valid_thumbnail(raw_rgb):
                    thumb_rgb = raw_rgb
                elif photo_data.get("path") and os.path.isfile(photo_data["path"]):
                    try:
                        extracted = session_cache_manager.extract_fast_thumbnail(photo_data["path"])
                        if is_valid_thumbnail(extracted):
                            thumb_rgb = extracted
                            photo_data["thumbnail_rgb"] = extracted
                            photo_data["raw_thumbnail_rgb"] = extracted
                    except Exception:
                        pass

        if thumb_rgb is not None and isinstance(thumb_rgb, np.ndarray) and thumb_rgb.size > 0:
            thumb_c = np.ascontiguousarray(thumb_rgb, dtype=np.uint8)
            h, w, c = thumb_c.shape
            qimg = QImage(thumb_c.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
            self._base_pixmap = QPixmap.fromImage(qimg)
            my = 82 * 0.14
            tw = max(10, int(76 - 8))
            th = max(10, int(82 - my * 2.0))
            self._pixmap = self._base_pixmap.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def set_item_size(self, target_w, target_h):
        self.setFixedSize(target_w, target_h)
        if (self._base_pixmap is None or self._base_pixmap.isNull()) and self.photo_data:
            thumb_rgb = self.photo_data.get("thumbnail_rgb") if self.photo_data.get("thumbnail_rgb") is not None else self.photo_data.get("raw_thumbnail_rgb")
            if not is_valid_thumbnail(thumb_rgb) and self.photo_data.get("path") and os.path.isfile(self.photo_data["path"]):
                try:
                    extracted = session_cache_manager.extract_fast_thumbnail(self.photo_data["path"])
                    if is_valid_thumbnail(extracted):
                        thumb_rgb = extracted
                        self.photo_data["thumbnail_rgb"] = extracted
                        self.photo_data["raw_thumbnail_rgb"] = extracted
                except Exception:
                    pass
            if thumb_rgb is not None and isinstance(thumb_rgb, np.ndarray) and thumb_rgb.size > 0:
                thumb_c = np.ascontiguousarray(thumb_rgb, dtype=np.uint8)
                h, w, c = thumb_c.shape
                qimg = QImage(thumb_c.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
                self._base_pixmap = QPixmap.fromImage(qimg)

        if self._base_pixmap and not self._base_pixmap.isNull():
            my = target_h * 0.14
            tw = max(10, int(target_w - 8))
            th = max(10, int(target_h - my * 2.0))
            self._pixmap = self._base_pixmap.scaled(
                tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        self.update()

    def set_thumbnail(self, pixmap):
        """Update thumbnail pixmap dynamically in real-time."""
        if pixmap and not pixmap.isNull():
            self._base_pixmap = pixmap
            my = self.height() * 0.14
            tw = max(10, int(self.width() - 8))
            th = max(10, int(self.height() - my * 2.0))
            self._pixmap = self._base_pixmap.scaled(
                tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.update()

    def set_active(self, is_active):
        self.is_active = is_active
        self.update()

    def set_selected(self, is_selected):
        self.is_selected = is_selected
        self.update()

    def update_photo_data(self, photo_data):
        """Update photo metadata and ensure tooltip is synchronized."""
        if not photo_data:
            return
        self.photo_data = photo_data
        if photo_data.get("filename"):
            self.filename = photo_data["filename"]
        if photo_data.get("path"):
            from path_utils import normalize_path
            self.file_path = normalize_path(photo_data["path"])
        self.update_tooltip()

    def update_tooltip(self):
        """Dynamically build tooltip ensuring resolution is accurate and never 0 x 0."""
        w = int(self.photo_data.get("width", 0))
        h = int(self.photo_data.get("height", 0))
        if (w <= 0 or h <= 0) and self.file_path and os.path.isfile(self.file_path):
            try:
                import session_cache_manager
                dw, dh = session_cache_manager.get_image_dimensions(self.file_path)
                if dw > 0 and dh > 0:
                    w, h = dw, dh
                    self.photo_data["width"] = w
                    self.photo_data["height"] = h
            except Exception:
                pass
        self.setToolTip(f"{self.filename}\n路径: {self.file_path}\n尺寸: {w} × {h}")

    def event(self, e):
        if e.type() == QEvent.Type.ToolTip:
            self.update_tooltip()
        return super().event(e)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.photo_id, event)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.contextMenuRequested.emit(self.photo_id, event.globalPosition().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        # 1. Base Film Negative Border & Background
        if self.is_active:
            bg_color = QColor(36, 30, 20)
            border_color = QColor(245, 158, 11)
            border_width = 1.8
        elif self.is_selected:
            bg_color = QColor(28, 30, 36)
            border_color = QColor(217, 119, 6)
            border_width = 1.5
        else:
            bg_color = QColor(20, 21, 25)
            border_color = QColor(44, 48, 58)
            border_width = 1.0

        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), 4.0, 4.0)

        # 2. Draw 35mm Sprocket Holes with strict proportional scaling (Item 9)
        sprocket_pen = QPen(QColor(36, 38, 48), 0.5)
        sprocket_brush = QBrush(QColor(10, 11, 14))
        painter.setPen(sprocket_pen)
        painter.setBrush(sprocket_brush)

        margin_y = h * 0.14
        hole_h = margin_y * 0.60
        hole_w = hole_h * 1.30
        hole_r = max(1.0, hole_h * 0.22)
        top_hy = (margin_y - hole_h) / 2.0
        bot_hy = h - margin_y + (margin_y - hole_h) / 2.0

        hole_pitch = max(12.0, hole_w * 2.2)
        num_holes = max(2, int(round((w - 10.0) / hole_pitch)))
        spacing = (w - 8.0) / float(num_holes)
        for i in range(num_holes):
            hx = 4.0 + i * spacing + (spacing - hole_w) / 2.0
            painter.drawRoundedRect(QRectF(hx, top_hy, hole_w, hole_h), hole_r, hole_r)
            painter.drawRoundedRect(QRectF(hx, bot_hy, hole_w, hole_h), hole_r, hole_r)

        # 3. Draw Thumbnail strictly inside leader bands
        thumb_rect = QRectF(4.0, margin_y, w - 8.0, h - margin_y * 2.0)
        painter.setClipRect(thumb_rect)
        if self._pixmap and not self._pixmap.isNull():
            px = thumb_rect.x() + (thumb_rect.width() - self._pixmap.width()) / 2.0
            py = thumb_rect.y() + (thumb_rect.height() - self._pixmap.height()) / 2.0
            painter.drawPixmap(int(px), int(py), self._pixmap)
        else:
            painter.fillRect(thumb_rect, QColor(20, 20, 22))

        # 4. Draw Filename Overlay at the bottom of the active frame (Light translucent glass)
        overlay_h = max(14.0, min(24.0, h * 0.20))
        overlay_rect = QRectF(thumb_rect.x(), thumb_rect.bottom() - overlay_h, thumb_rect.width(), overlay_h)
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 80))

        painter.setPen(QPen(QColor(230, 230, 230)))
        font = painter.font()
        font_size = max(7, min(10, int(round(h * 0.09))))
        font.setPointSize(font_size)
        painter.setFont(font)

        fm = QFontMetrics(font)
        elided = fm.elidedText(self.filename, Qt.TextElideMode.ElideMiddle, int(overlay_rect.width() - 4))
        painter.drawText(overlay_rect, Qt.AlignmentFlag.AlignCenter, elided)

        # 5. Draw Edited/Modified indicator badge (Item 1)
        painter.setClipping(False)
        if self.photo_data.get("is_dirty") or self.photo_data.get("is_edited"):
            badge_sz = max(5.0, min(9.0, h * 0.07))
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1.0))
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.drawEllipse(QRectF(margin_y * 0.6, margin_y * 1.1, badge_sz, badge_sz))

        # 6. Top layer: High-visibility Selection & Active Focus Border (Single and Multi-select)
        if self.is_active:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#f59e0b"), 2.2))
            painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), 4.0, 4.0)
        elif self.is_selected:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(245, 158, 11, 230), 2.0))
            painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), 4.0, 4.0)

    @property
    def is_dirty(self):
        return bool(self.photo_data.get("is_dirty"))

    def set_dirty(self, is_dirty=True):
        self.photo_data["is_dirty"] = is_dirty
        if is_dirty:
            self.photo_data["is_edited"] = True
        else:
            p_path = self.photo_data.get("path")
            self.photo_data["is_edited"] = bool(p_path and sdc_manager.has_sdc(p_path))
        self.update()


class FilmstripWidget(QWidget):
    """ACR-style bottom negative filmstrip with 35mm frames, multi-selection,
    horizontal wheel scrolling, and batch export right-click menu.
    """
    photoSelected = Signal(str)
    photoRemoved = Signal(str)
    photosRemoved = Signal(list)
    clearRequested = Signal()
    batchExportRequested = Signal(list)
    quickExportRequested = Signal(list)
    addRequested = Signal()
    clearSdcRequested = Signal(list)
    copyParamsRequested = Signal(str)
    pasteParamsRequested = Signal()
    selectionChanged = Signal(list)
    sortRequested = Signal(str)
    stepPhotoRequested = Signal(int)
    navigateFirstRequested = Signal()
    navigateLastRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setMaximumHeight(260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._item_widgets = {}
        self._photos_list = []
        self._active_id = None
        self._selected_ids = set()
        self._can_paste_params = False

        self._init_ui()

    def set_can_paste(self, can_paste: bool):
        """Update whether clipboard contains a valid darkroom recipe."""
        self._can_paste_params = bool(can_paste)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_ids:
                self.photosRemoved.emit(list(self._selected_ids))
                event.accept()
                return
        elif event.key() == Qt.Key.Key_Left:
            self.stepPhotoRequested.emit(-1)
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Right:
            self.stepPhotoRequested.emit(1)
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Home:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._select_range_to_boundary(to_start=True)
            else:
                self.navigate_first()
                self.navigateFirstRequested.emit()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_End:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._select_range_to_boundary(to_start=False)
            else:
                self.navigate_last()
                self.navigateLastRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 6, 10, 6)
        main_layout.setSpacing(8)
        self.setStyleSheet("background: #0d0e11; border-top: 1px solid #20222a;")

        # Scroll area for items (Smooth non-linear scrolling with edge indicators)
        self.scroll_area = SmoothScrollArea(self, is_horizontal=True)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:horizontal {
                background: rgba(18, 19, 24, 0.75);
                height: 6px;
                margin: 0px 4px 1px 4px;
                border-radius: 3px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #474d61;
                min-width: 24px;
                border-radius: 3px;
                border: none;
            }
            QScrollBar::handle:horizontal:hover {
                background: #f59e0b;
            }
            QScrollBar::handle:horizontal:pressed {
                background: #d97706;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                height: 0px;
                width: 0px;
                background: none;
                border: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
        """)

        self.scroll_content = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 7, 0, 7)
        self.scroll_layout.setSpacing(6)
        self.scroll_layout.addStretch()

        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scroll_area.customContextMenuRequested.connect(self._on_blank_context_menu)
        self.scroll_content.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scroll_content.customContextMenuRequested.connect(self._on_blank_context_menu)
        main_layout.addWidget(self.scroll_area, 1)

        # Right Action Container (Upper 50%: 1/4 胶片库 + 1/4 张数; Lower 50%: 2/4 铺满 ＋ 添加 button)
        self.action_container = QWidget()
        self.action_container.setFixedWidth(66)
        action_layout = QVBoxLayout(self.action_container)
        action_layout.setContentsMargins(1, 2, 1, 2)
        action_layout.setSpacing(2)

        # Upper container (50%): 1/4 胶片库 + 1/4 张数
        top_box = QWidget()
        top_layout = QVBoxLayout(top_box)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self.lbl_lib_title = QLabel("胶片库")
        self.lbl_lib_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_lib_title.setStyleSheet("color: #71717a; font-size: 10px; font-weight: bold; padding: 0px; margin: 0px;")
        top_layout.addWidget(self.lbl_lib_title, 1)

        self.count_label = QLabel("0 张")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_label.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold; padding: 0px; margin: 0px;")
        top_layout.addWidget(self.count_label, 1)
        action_layout.addWidget(top_box, 1)

        # Lower container (50%): ＋ 添加 按钮全部铺满
        self.add_btn = QPushButton("＋ 添加")
        self.add_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.setToolTip("添加更多底片 / RAW 文件")
        self.add_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background: #181920;
                color: #f59e0b;
                border: 1px solid #2d303e;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                outline: none;
            }
            QPushButton:hover {
                background: #232530;
                color: #fbbf24;
                border: 1px solid #f59e0b;
            }
            QPushButton:pressed {
                background: #121317;
                color: #d97706;
                border: 1px solid #d97706;
            }
            QPushButton:focus {
                outline: none;
                border: 1px solid #2d303e;
            }
        """)
        self.add_btn.clicked.connect(self.addRequested.emit)
        action_layout.addWidget(self.add_btn, 1)

        main_layout.addWidget(self.action_container)

    def set_photo_dirty(self, photo_id, is_dirty=True):
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].set_dirty(is_dirty)

    def _compute_item_dimensions(self):
        """Computes (item_w, item_h) leaving comfortable vertical padding
        above (so selection border doesn't touch the top dividing line)
        and below (so horizontal scrollbar doesn't obscure sprocket holes).
        """
        avail_h = self.height()
        item_h = max(48, min(240, avail_h - 26))
        item_w = int(round(item_h * 0.92))
        return item_w, item_h

    def resizeEvent(self, event):
        super().resizeEvent(event)
        item_w, item_h = self._compute_item_dimensions()
        for item in self._item_widgets.values():
            item.set_item_size(item_w, item_h)

    def wheelEvent(self, event):
        """Enable horizontal scrolling via standard mouse wheel over the filmstrip."""
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta != 0:
            h_bar = self.scroll_area.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)

    def set_photos(self, photos, active_id):
        self._photos_list = photos
        self._active_id = active_id
        if active_id:
            self._selected_ids = {active_id}
        else:
            self._selected_ids = set()

        # Clear existing
        for w in self._item_widgets.values():
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self._item_widgets.clear()

        # Update count
        self.count_label.setText(f"{len(photos)} 张")

        if len(photos) == 0:
            if not hasattr(self, '_empty_lbl') or self._empty_lbl is None:
                self._empty_lbl = QLabel("底片库为空，点击右侧「＋ 添加」或拖入文件开始导入")
                self._empty_lbl.setStyleSheet("color: #64748b; font-size: 11px; margin-left: 14px;")
            self.scroll_layout.insertWidget(0, self._empty_lbl)
            self.selectionChanged.emit([])
            return

        if hasattr(self, '_empty_lbl') and self._empty_lbl is not None:
            self.scroll_layout.removeWidget(self._empty_lbl)
            self._empty_lbl.deleteLater()
            self._empty_lbl = None

        # Re-add with current dynamic size
        item_w, item_h = self._compute_item_dimensions()
        for idx, p in enumerate(photos):
            pid = p["id"]
            is_active = (pid == active_id)
            is_selected = (pid in self._selected_ids)
            item = FilmstripItemWidget(p, is_active=is_active, is_selected=is_selected)
            item.set_item_size(item_w, item_h)
            item.clicked.connect(self._on_item_clicked)
            item.removeRequested.connect(self.photoRemoved.emit)
            item.contextMenuRequested.connect(self._show_context_menu)
            self._item_widgets[pid] = item
            self.scroll_layout.insertWidget(idx, item)
        self.selectionChanged.emit(list(self._selected_ids))

    def append_photo(self, photo, is_active=False):
        """Incrementally appends a single photo entry to the filmstrip without clearing."""
        if hasattr(self, '_empty_lbl') and self._empty_lbl is not None:
            self.scroll_layout.removeWidget(self._empty_lbl)
            self._empty_lbl.deleteLater()
            self._empty_lbl = None

        if photo not in self._photos_list:
            self._photos_list.append(photo)
        self.count_label.setText(f"{len(self._photos_list)} 张")

        item_w, item_h = self._compute_item_dimensions()
        pid = photo["id"]

        if is_active:
            self._active_id = pid
            self._selected_ids = {pid}

        item = FilmstripItemWidget(photo, is_active=is_active, is_selected=is_active)
        item.set_item_size(item_w, item_h)
        item.clicked.connect(self._on_item_clicked)
        item.removeRequested.connect(self.photoRemoved.emit)
        item.contextMenuRequested.connect(self._show_context_menu)
        self._item_widgets[pid] = item
        idx = max(0, len(self._item_widgets) - 1)
        self.scroll_layout.insertWidget(idx, item)
        if is_active:
            self.selectionChanged.emit(list(self._selected_ids))

    def set_active_photo(self, active_id):
        self._active_id = active_id
        if active_id and active_id not in self._selected_ids:
            self._selected_ids = {active_id}
        self._update_items_selection()
        self.selectionChanged.emit(list(self._selected_ids))

    def get_selected_photo_ids(self):
        return list(self._selected_ids) if self._selected_ids else ([self._active_id] if self._active_id else [])

    def select_all_photos(self):
        """Select all photos in filmstrip (Ctrl+A)."""
        self.select_all()

    def _update_items_selection(self):
        for pid, w in self._item_widgets.items():
            w.set_active(pid == self._active_id)
            w.set_selected(pid in self._selected_ids)

    def _on_item_clicked(self, photo_id, event):
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+Click: Toggle selection WITHOUT switching active canvas photo
            if photo_id in self._selected_ids:
                if len(self._selected_ids) > 1:
                    self._selected_ids.remove(photo_id)
                    # If active photo was deselected, update internal pointer without emitting photoSelected
                    if self._active_id == photo_id:
                        self._active_id = next(iter(self._selected_ids))
            else:
                self._selected_ids.add(photo_id)
                # Keep active photo intact so canvas never switches
                if not self._active_id:
                    self._active_id = photo_id

            self._update_items_selection()
            self.selectionChanged.emit(list(self._selected_ids))
            return
        elif mods & Qt.KeyboardModifier.ShiftModifier and self._active_id in self._item_widgets:
            # Shift+Click: Range selection
            pids = [p["id"] for p in self._photos_list]
            if self._active_id in pids and photo_id in pids:
                i1 = pids.index(self._active_id)
                i2 = pids.index(photo_id)
                start, end = min(i1, i2), max(i1, i2)
                self._selected_ids = set(pids[start:end+1])
                self.set_active_photo(photo_id)
                self.photoSelected.emit(photo_id)
        else:
            # Normal single click
            self._selected_ids = {photo_id}
            self.set_active_photo(photo_id)
            self.photoSelected.emit(photo_id)
        self.selectionChanged.emit(list(self._selected_ids))

    def update_photo_thumbnail(self, photo_id, pixmap):
        """Update thumbnail for specific photo item."""
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].set_thumbnail(pixmap)

    def update_photo_data(self, photo_id, photo_data):
        """Update photo metadata and tooltip for specific photo item."""
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].update_photo_data(photo_data)

    def scroll_to_active(self):
        """Smoothly ensure active photo item is centered in view."""
        if self._active_id in self._item_widgets:
            w = self._item_widgets[self._active_id]
            self.scroll_area.ensureWidgetVisible(w, 50, 0)

    def scroll_to_start(self):
        """Scroll filmstrip to the very beginning."""
        if hasattr(self, "scroll_area") and self.scroll_area:
            hbar = self.scroll_area.horizontalScrollBar()
            if hbar:
                hbar.setValue(hbar.minimum())

    def scroll_to_end(self):
        """Scroll filmstrip to the very end."""
        if hasattr(self, "scroll_area") and self.scroll_area:
            hbar = self.scroll_area.horizontalScrollBar()
            if hbar:
                hbar.setValue(hbar.maximum())

    def navigate_first(self):
        """Navigate to the first photo in filmstrip, activate and select it (Home key)."""
        if not self._photos_list:
            return
        first_id = self._photos_list[0].get("id")
        if first_id:
            self._selected_ids = {first_id}
            self.set_active_photo(first_id)
            self.photoSelected.emit(first_id)
            self.scroll_to_start()
            self.scroll_to_active()

    def navigate_last(self):
        """Navigate to the last photo in filmstrip, activate and select it (End key)."""
        if not self._photos_list:
            return
        last_id = self._photos_list[-1].get("id")
        if last_id:
            self._selected_ids = {last_id}
            self.set_active_photo(last_id)
            self.photoSelected.emit(last_id)
            self.scroll_to_end()
            self.scroll_to_active()

    def _select_range_to_boundary(self, to_start=True):
        """Handle Shift+Home or Shift+End range selection to the start or end."""
        if not self._photos_list:
            return
        pids = [p["id"] for p in self._photos_list if p and "id" in p]
        if not pids:
            return
        curr_idx = pids.index(self._active_id) if self._active_id in pids else (0 if to_start else len(pids) - 1)
        target_idx = 0 if to_start else len(pids) - 1
        start, end = min(curr_idx, target_idx), max(curr_idx, target_idx)
        self._selected_ids = set(pids[start:end + 1])
        target_id = pids[target_idx]
        self.set_active_photo(target_id)
        self.photoSelected.emit(target_id)
        if to_start:
            self.scroll_to_start()
        else:
            self.scroll_to_end()
        self.scroll_to_active()

    def _show_context_menu(self, target_id, global_pos):
        # Decouple right-click: ensure target_id is selected for action scope, but DO NOT switch active canvas photo
        if target_id not in self._selected_ids:
            self._selected_ids = {target_id}
            self._update_items_selection()
            self.selectionChanged.emit(list(self._selected_ids))

        sel_count = len(self._selected_ids)
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        quick_label = "快速冲印 (上次参数)" if sel_count == 1 else f"快速冲印选中底片 ({sel_count} 张)"
        act_quick = menu.addAction(quick_label)
        act_quick.setIcon(get_icon("quick_export.png"))
        act_quick.triggered.connect(lambda: self.quickExportRequested.emit(list(self._selected_ids)))

        export_label = f"冲印当前底片..." if sel_count == 1 else f"批量冲印选中底片 ({sel_count} 张)..."
        act_export = menu.addAction(export_label)
        act_export.setIcon(get_icon("dlg_export.png"))
        act_export.triggered.connect(lambda: self.batchExportRequested.emit(list(self._selected_ids)))

        target_photo = next((p for p in self._photos_list if p.get("id") == target_id), None)
        target_path = target_photo.get("path", "") if target_photo else ""
        if target_path:
            act_reveal = menu.addAction("在文件资源管理器中显示")
            act_reveal.triggered.connect(lambda checked=False, p=target_path: self._reveal_in_explorer(p))

        remove_label = "从底片库移除" if sel_count == 1 else f"从底片库移除 ({sel_count} 项)"
        act_remove = menu.addAction(remove_label)
        act_remove.setIcon(get_icon("dlg_discard.png"))
        act_remove.triggered.connect(lambda: self.photosRemoved.emit(list(self._selected_ids)))

        menu.addSeparator()

        # Copy / Paste darkroom recipe actions
        target_edited = False
        if target_photo:
            p_path = target_photo.get("path", "")
            if target_photo.get("is_dirty") or target_photo.get("is_edited") or (p_path and sdc_manager.has_sdc(p_path)):
                target_edited = True
            elif target_photo.get("film_profile", "none") != "none" or target_photo.get("paper_profile", "none") != "none":
                target_edited = True

        act_copy = menu.addAction("复制暗房配方 (Ctrl+C)")
        act_copy.setEnabled(target_edited)
        act_copy.triggered.connect(lambda checked=False, pid=target_id: self.copyParamsRequested.emit(pid))

        paste_label = "粘贴暗房配方 (Ctrl+V)" if sel_count <= 1 else f"粘贴暗房配方至选中底片 ({sel_count} 张) (Ctrl+V)"
        act_paste = menu.addAction(paste_label)
        can_paste = bool(getattr(self, "_can_paste_params", False))
        act_paste.setEnabled(can_paste)
        act_paste.triggered.connect(self.pasteParamsRequested.emit)

        menu.addSeparator()

        # Check if any selected item is edited or has an existing .sdc file
        selected_photos = [p for p in self._photos_list if p["id"] in self._selected_ids]
        can_clear_sdc = False
        for p in selected_photos:
            p_path = p.get("path", "")
            if p.get("is_dirty") or p.get("is_edited") or (p_path and sdc_manager.has_sdc(p_path)):
                can_clear_sdc = True
                break

        sdc_label = "删除暗房配置 (.sdc)" if sel_count == 1 else f"删除暗房配置 (.sdc) ({sel_count} 项)"
        act_clear_sdc = menu.addAction(sdc_label)
        act_clear_sdc.setIcon(get_icon("reset.png"))
        act_clear_sdc.setEnabled(can_clear_sdc)
        act_clear_sdc.triggered.connect(lambda: self.clearSdcRequested.emit(list(self._selected_ids)))

        menu.addSeparator()

        # Submenu: 整理底片
        sort_menu = menu.addMenu("整理底片")
        sort_menu.setStyleSheet(get_darkroom_menu_style())
        act_s1 = sort_menu.addAction("按拍摄时间升序 (旧 → 新)")
        act_s1.triggered.connect(lambda: self.sortRequested.emit("time_asc"))
        act_s2 = sort_menu.addAction("按拍摄时间降序 (新 → 旧)")
        act_s2.triggered.connect(lambda: self.sortRequested.emit("time_desc"))
        sort_menu.addSeparator()
        act_s3 = sort_menu.addAction("按文件名升序 (A → Z)")
        act_s3.triggered.connect(lambda: self.sortRequested.emit("name_asc"))
        act_s4 = sort_menu.addAction("按文件名降序 (Z → A)")
        act_s4.triggered.connect(lambda: self.sortRequested.emit("name_desc"))
        sort_menu.addSeparator()
        act_s5 = sort_menu.addAction("按导入添加顺序")
        act_s5.triggered.connect(lambda: self.sortRequested.emit("import"))

        menu.addSeparator()

        act_select_all = menu.addAction("全选所有底片 (Ctrl+A)")
        act_select_all.triggered.connect(self.select_all)

        act_invert = menu.addAction("反选")
        act_invert.triggered.connect(self.invert_selection)

        menu.exec(global_pos)

    def _reveal_in_explorer(self, file_path):
        if not file_path:
            return
        import subprocess
        norm_p = os.path.abspath(file_path)
        if os.path.exists(norm_p):
            subprocess.Popen(f'explorer /select,"{norm_p}"')
        else:
            dir_p = os.path.dirname(norm_p)
            if os.path.exists(dir_p):
                os.startfile(dir_p)

    def _on_blank_context_menu(self, pos):
        sender = self.sender()
        global_pos = sender.mapToGlobal(pos) if sender else QCursor.pos()
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        act_add = menu.addAction("导入底片...")
        act_add.setIcon(get_icon("dlg_import.png"))
        act_add.triggered.connect(self.addRequested.emit)

        menu.addSeparator()

        # Submenu: 整理底片
        sort_menu = menu.addMenu("整理底片")
        sort_menu.setStyleSheet(get_darkroom_menu_style())
        act_s1 = sort_menu.addAction("按拍摄时间升序 (旧 → 新)")
        act_s1.triggered.connect(lambda: self.sortRequested.emit("time_asc"))
        act_s2 = sort_menu.addAction("按拍摄时间降序 (新 → 旧)")
        act_s2.triggered.connect(lambda: self.sortRequested.emit("time_desc"))
        sort_menu.addSeparator()
        act_s3 = sort_menu.addAction("按文件名升序 (A → Z)")
        act_s3.triggered.connect(lambda: self.sortRequested.emit("name_asc"))
        act_s4 = sort_menu.addAction("按文件名降序 (Z → A)")
        act_s4.triggered.connect(lambda: self.sortRequested.emit("name_desc"))
        sort_menu.addSeparator()
        act_s5 = sort_menu.addAction("按导入添加顺序")
        act_s5.triggered.connect(lambda: self.sortRequested.emit("import"))

        menu.addSeparator()

        act_select_all = menu.addAction("全选所有底片 (Ctrl+A)")
        act_select_all.triggered.connect(self.select_all)

        if self._photos_list:
            act_clear = menu.addAction("清空底片库")
            act_clear.setIcon(get_icon("dlg_discard.png"))
            act_clear.triggered.connect(self.clearRequested.emit)

        menu.exec(global_pos)

    def select_all(self):
        self._selected_ids = {p["id"] for p in self._photos_list}
        self._update_items_selection()
        self.selectionChanged.emit(list(self._selected_ids))

    def invert_selection(self):
        all_ids = {p["id"] for p in self._photos_list}
        self._selected_ids = all_ids - self._selected_ids
        self._update_items_selection()
        self.selectionChanged.emit(list(self._selected_ids))

    def get_selected_ids(self) -> list[str]:
        """Returns a list of currently selected photo IDs in the filmstrip."""
        return list(self._selected_ids)
