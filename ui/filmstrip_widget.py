import os
from PySide6.QtCore import Qt, Signal, QRectF, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen, QFontMetrics, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QMenu, QDialog, QTextEdit, QSizePolicy
)
import sdc_manager
from path_utils import get_icon_path, get_resource_dir
from ui.window_utils import apply_dark_titlebar, get_darkroom_menu_style
from ui.smooth_scroll import SmoothScrollArea


class DuplicateFilesDialog(QDialog):
    """Dialog showing duplicate files that were skipped during import."""
    def __init__(self, duplicate_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("重复文件过滤提示")
        self.setFixedSize(480, 320)
        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)
        self.setStyleSheet("""
            QDialog {
                background: #16171b;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 12px;
            }
            QTextEdit {
                background: #111215;
                color: #a1a1aa;
                border: 1px solid #2f323e;
                border-radius: 6px;
                padding: 8px;
                font-family: Consolas, monospace;
                font-size: 11.5px;
            }
            QPushButton {
                background: #f59e0b;
                color: #111;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 6px 22px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #d97706;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        lbl_head = QLabel(f"已自动过滤 {len(duplicate_names)} 个已打开的重复文件：")
        lbl_head.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 13px;")
        layout.addWidget(lbl_head)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText("\n".join(duplicate_names))
        layout.addWidget(txt, 1)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_ok = QPushButton("确定")
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.clicked.connect(self.accept)
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)


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
        self.photo_id = photo_data["id"]
        self.filename = photo_data["filename"]
        self.file_path = photo_data.get("path", "")
        self.is_active = is_active
        self.is_selected = is_selected
        self.photo_data = photo_data

        self.setFixedSize(76, 82)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{self.filename}\n尺寸: {photo_data.get('width', 0)} × {photo_data.get('height', 0)}")

        self._base_pixmap = None
        self._pixmap = None
        thumb_rgb = photo_data.get("thumbnail_rgb")
        if thumb_rgb is not None:
            h, w, c = thumb_rgb.shape
            qimg = QImage(thumb_rgb.data, w, h, w * c, QImage.Format.Format_RGB888)
            self._base_pixmap = QPixmap.fromImage(qimg)
            self._pixmap = self._base_pixmap.scaled(66, 52, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def set_item_size(self, target_w, target_h):
        self.setFixedSize(target_w, target_h)
        if self._base_pixmap and not self._base_pixmap.isNull():
            tw = max(10, target_w - 10)
            th = max(10, target_h - 20)
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

        w = self.width()
        h = self.height()

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
        painter.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 4, 4)

        # 2. Draw 35mm Sprocket Holes (4 holes on top, 4 on bottom)
        sprocket_pen = QPen(QColor(32, 34, 42), 0.5)
        sprocket_brush = QBrush(QColor(10, 11, 14))
        painter.setPen(sprocket_pen)
        painter.setBrush(sprocket_brush)

        hole_w = 6.0
        hole_h = 4.0
        spacing = (w - 8) / 4.0

        for i in range(4):
            hx = 5.0 + i * spacing + (spacing - hole_w) / 2.0
            # Top row
            painter.drawRoundedRect(QRectF(hx, 3.5, hole_w, hole_h), 1.0, 1.0)
            # Bottom row
            painter.drawRoundedRect(QRectF(hx, h - 7.5, hole_w, hole_h), 1.0, 1.0)

        # 3. Draw Thumbnail in the center
        thumb_rect = QRectF(5, 10, w - 10, h - 20)
        painter.setClipRect(thumb_rect)
        if self._pixmap and not self._pixmap.isNull():
            px = thumb_rect.x() + (thumb_rect.width() - self._pixmap.width()) / 2.0
            py = thumb_rect.y() + (thumb_rect.height() - self._pixmap.height()) / 2.0
            painter.drawPixmap(int(px), int(py), self._pixmap)
        else:
            painter.fillRect(thumb_rect, QColor(20, 20, 22))

        # 4. Draw Filename Overlay at the bottom
        overlay_h = 16.0
        overlay_rect = QRectF(thumb_rect.x(), thumb_rect.bottom() - overlay_h, thumb_rect.width(), overlay_h)
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 185))

        painter.setPen(QPen(QColor(230, 230, 230)))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        fm = QFontMetrics(font)
        elided = fm.elidedText(self.filename, Qt.TextElideMode.ElideMiddle, int(overlay_rect.width() - 4))
        painter.drawText(overlay_rect, Qt.AlignmentFlag.AlignCenter, elided)

        # 5. Draw Edited/Modified indicator badge (Item 1)
        if self.photo_data.get("is_dirty") or self.photo_data.get("is_edited"):
            painter.setClipping(False)
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1.0))
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.drawEllipse(QRectF(7.0, 12.0, 6.0, 6.0))

    @property
    def is_dirty(self):
        return bool(self.photo_data.get("is_dirty"))

    def set_dirty(self, is_dirty=True):
        self.photo_data["is_dirty"] = is_dirty
        if is_dirty:
            self.photo_data["is_edited"] = True
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
    addRequested = Signal()
    clearSdcRequested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setMaximumHeight(260)
        self._item_widgets = {}
        self._photos_list = []
        self._active_id = None
        self._selected_ids = set()

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 4, 10, 4)
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
                height: 5px;
                background: #0d0e11;
                border-radius: 2px;
            }
            QScrollBar::handle:horizontal {
                background: #2b2e38;
                border-radius: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #f59e0b;
            }
        """)

        self.scroll_content = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
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
                background: #141519;
                color: #f59e0b;
                border: 1px dashed #424656;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                outline: none;
            }
            QPushButton:hover {
                background: #1e2028;
                color: #fbbf24;
                border: 1px dashed #f59e0b;
            }
            QPushButton:pressed {
                background: #101114;
                color: #d97706;
                border: 1px solid #d97706;
            }
            QPushButton:focus {
                outline: none;
                border: 1px dashed #f59e0b;
            }
        """)
        self.add_btn.clicked.connect(self.addRequested.emit)
        action_layout.addWidget(self.add_btn, 1)

        main_layout.addWidget(self.action_container)

    def set_photo_dirty(self, photo_id, is_dirty=True):
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].set_dirty(is_dirty)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        avail_h = self.height()
        item_h = max(52, min(240, avail_h - 14))
        item_w = int(round(item_h * 0.92))
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
        if active_id and active_id not in self._selected_ids:
            self._selected_ids = {active_id}

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
            return

        if hasattr(self, '_empty_lbl') and self._empty_lbl is not None:
            self.scroll_layout.removeWidget(self._empty_lbl)
            self._empty_lbl.deleteLater()
            self._empty_lbl = None

        # Re-add with current dynamic size
        item_h = max(52, min(240, self.height() - 14))
        item_w = int(round(item_h * 0.92))
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

    def append_photo(self, photo, is_active=False):
        """Incrementally appends a single photo entry to the filmstrip without clearing."""
        if hasattr(self, '_empty_lbl') and self._empty_lbl is not None:
            self.scroll_layout.removeWidget(self._empty_lbl)
            self._empty_lbl.deleteLater()
            self._empty_lbl = None

        if photo not in self._photos_list:
            self._photos_list.append(photo)
        self.count_label.setText(f"{len(self._photos_list)} 张")

        item_h = max(52, min(240, self.height() - 14))
        item_w = int(round(item_h * 0.92))
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

    def set_active_photo(self, active_id):
        self._active_id = active_id
        if active_id not in self._selected_ids:
            self._selected_ids = {active_id}
        for pid, w in self._item_widgets.items():
            w.set_active(pid == active_id)
            w.set_selected(pid in self._selected_ids)

    def get_selected_photo_ids(self):
        return list(self._selected_ids) if self._selected_ids else ([self._active_id] if self._active_id else [])

    def select_all_photos(self):
        """Select all photos in filmstrip (Ctrl+A)."""
        self._selected_ids = {p["id"] for p in self._photos_list}
        for pid, w in self._item_widgets.items():
            w.set_selected(True)

    def _update_items_selection(self):
        for pid, w in self._item_widgets.items():
            w.set_active(pid == self._active_id)
            w.set_selected(pid in self._selected_ids)

    def _on_item_clicked(self, photo_id, event):
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+Click: Toggle selection (can deselect any item)
            if photo_id in self._selected_ids:
                if len(self._selected_ids) > 1:
                    self._selected_ids.remove(photo_id)
                    # If active photo was deselected, switch active pointer to another selected photo
                    if self._active_id == photo_id:
                        new_active = next(iter(self._selected_ids))
                        self._active_id = new_active
                        self.photoSelected.emit(new_active)
                    self._update_items_selection()
            else:
                self._selected_ids.add(photo_id)
                self._active_id = photo_id
                self._update_items_selection()
                self.photoSelected.emit(photo_id)
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

    def _show_context_menu(self, target_id, global_pos):
        # Decouple right-click: ensure target_id is selected for action scope, but DO NOT switch active canvas photo
        if target_id not in self._selected_ids:
            self._selected_ids = {target_id}
            for pid, w in self._item_widgets.items():
                w.set_selected(pid in self._selected_ids)

        sel_count = len(self._selected_ids)
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        icon_exp = os.path.join(get_resource_dir(), "icons", "dlg_export.png")
        icon_del = os.path.join(get_resource_dir(), "icons", "dlg_discard.png")

        export_label = f"导出此底片..." if sel_count == 1 else f"导出选中底片 ({sel_count} 张)..."
        act_export = menu.addAction(export_label)
        if os.path.exists(icon_exp):
            act_export.setIcon(QIcon(icon_exp))
        act_export.triggered.connect(lambda: self.batchExportRequested.emit(list(self._selected_ids)))

        remove_label = "从底片库移除" if sel_count == 1 else f"从底片库移除 ({sel_count} 项)"
        act_remove = menu.addAction(remove_label)
        if os.path.exists(icon_del):
            act_remove.setIcon(QIcon(icon_del))
        act_remove.triggered.connect(lambda: self.photosRemoved.emit(list(self._selected_ids)))

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
        act_clear_sdc.setEnabled(can_clear_sdc)
        act_clear_sdc.triggered.connect(lambda: self.clearSdcRequested.emit(list(self._selected_ids)))

        menu.addSeparator()

        act_select_all = menu.addAction("全选所有底片 (Ctrl+A)")
        act_select_all.triggered.connect(self.select_all)

        act_invert = menu.addAction("反选")
        act_invert.triggered.connect(self.invert_selection)

        menu.exec(global_pos)

    def _on_blank_context_menu(self, pos):
        sender = self.sender()
        global_pos = sender.mapToGlobal(pos) if sender else QCursor.pos()
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        icon_imp = os.path.join(get_resource_dir(), "icons", "dlg_import.png")
        icon_del = os.path.join(get_resource_dir(), "icons", "dlg_discard.png")

        act_add = menu.addAction("导入底片...")
        if os.path.exists(icon_imp):
            act_add.setIcon(QIcon(icon_imp))
        act_add.triggered.connect(self.addRequested.emit)

        menu.addSeparator()

        act_select_all = menu.addAction("全选所有底片 (Ctrl+A)")
        act_select_all.triggered.connect(self.select_all)

        if self._photos_list:
            act_clear = menu.addAction("清空底片库")
            if os.path.exists(icon_del):
                act_clear.setIcon(QIcon(icon_del))
            act_clear.triggered.connect(self.clearRequested.emit)

        menu.exec(global_pos)

    def select_all(self):
        self._selected_ids = {p["id"] for p in self._photos_list}
        for pid, w in self._item_widgets.items():
            w.set_selected(True)

    def invert_selection(self):
        all_ids = {p["id"] for p in self._photos_list}
        self._selected_ids = all_ids - self._selected_ids
        for pid, w in self._item_widgets.items():
            w.set_selected(pid in self._selected_ids)
