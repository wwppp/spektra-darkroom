"""
relink_dialog.py - Professional Relink Media Dialog for SpektraDarkroom
Handles missing/offline photos when opening sessions moved across directories or machines.
Supports cascaded auto-relink by scanning a user-specified root folder.
"""

import os
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QColor, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QAbstractItemView, QMessageBox
)

from core.path_utils import get_resource_dir, normalize_path
from ui.window_utils import apply_dark_titlebar
from ui.theme_manager import theme_manager


class RelinkMediaDialog(QDialog):
    """Dialog displayed when photos referenced in a session are not found at their recorded paths."""

    STATUS_OFFLINE = 0
    STATUS_RELINKED = 1
    STATUS_CANCEL = 2

    def __init__(self, missing_entries, total_photos_count=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("底片脱机 — 重新关联媒体文件")
        self.setFixedSize(700, 480)
        self.setModal(True)

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.missing_entries = missing_entries  # list of dicts: {"filename": ..., "orig_path": ..., "resolved_path": None}
        self.total_count = total_photos_count or len(missing_entries)
        self.relinked_mapping = {}  # {orig_path: new_path}
        self.action_result = self.STATUS_CANCEL

        self._init_ui()
        self._apply_style()
        self._populate_table()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(22, 20, 22, 18)
        main_layout.setSpacing(14)

        # Header Info Banner
        header = QHBoxLayout()
        header.setSpacing(14)

        is_dark = theme_manager.is_dark()
        lbl_badge = QLabel("!")
        lbl_badge.setFixedSize(36, 36)
        lbl_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_bg = "#2b1f11" if is_dark else "#fef3c7"
        badge_border = "#78350f" if is_dark else "#f59e0b"
        lbl_badge.setStyleSheet(f"""
            background: {badge_bg};
            color: #f59e0b;
            border: 1px solid {badge_border};
            border-radius: 18px;
            font-size: 18px;
            font-weight: bold;
        """)
        header.addWidget(lbl_badge)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)
        lbl_title = QLabel("检测到底片脱机 (媒体文件丢失)")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f59e0b;")
        text_layout.addWidget(lbl_title)

        missing_cnt = len(self.missing_entries)
        desc_col = "#94a3b8" if is_dark else "#475569"
        lbl_desc = QLabel(
            f"会话共记录 {self.total_count} 张底片，其中 {missing_cnt} 张在原存储路径未找到。\n"
            "若您已将底片文件夹移动至其他位置或驱动器，请指定底片所在文件夹执行自动级联重链。"
        )
        lbl_desc.setStyleSheet(f"font-size: 11.5px; color: {desc_col}; line-height: 1.4;")
        text_layout.addWidget(lbl_desc)

        header.addLayout(text_layout, 1)
        main_layout.addLayout(header)

        # Missing Photos Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["底片文件名", "最后已知路径", "当前状态"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)

        h_header = self.table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 160)
        self.table.setColumnWidth(2, 130)

        main_layout.addWidget(self.table, 1)

        # Bottom Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #10b981; font-size: 12px; font-weight: bold;")
        btn_row.addWidget(self.lbl_status)
        btn_row.addStretch()

        self.btn_browse = QPushButton("指定底片所在文件夹...")
        self.btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #111;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #fbbf24;
            }
            QPushButton:pressed {
                background: #d97706;
            }
        """)
        self.btn_browse.clicked.connect(self._on_browse_folder)
        btn_row.addWidget(self.btn_browse)

        self.btn_offline = QPushButton("保持脱机继续")
        self.btn_offline.setCursor(Qt.CursorShape.PointingHandCursor)
        if is_dark:
            self.btn_offline.setStyleSheet("""
                QPushButton {
                    background: #20222a;
                    color: #cbd5e1;
                    border: 1px solid #333647;
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: #2b2e3b;
                    color: #fff;
                }
            """)
        else:
            self.btn_offline.setStyleSheet("""
                QPushButton {
                    background: #ffffff;
                    color: #334155;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: #f1f5f9;
                    color: #0f172a;
                }
            """)
        self.btn_offline.clicked.connect(self._on_keep_offline)
        btn_row.addWidget(self.btn_offline)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        if is_dark:
            self.btn_cancel.setStyleSheet("""
                QPushButton {
                    background: #181920;
                    color: #94a3b8;
                    border: 1px solid #282a36;
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: #22242e;
                    color: #cbd5e1;
                }
            """)
        else:
            self.btn_cancel.setStyleSheet("""
                QPushButton {
                    background: #ffffff;
                    color: #64748b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: #f1f5f9;
                    color: #334155;
                }
            """)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        main_layout.addLayout(btn_row)

    def _apply_style(self):
        is_dark = theme_manager.is_dark()
        if is_dark:
            self.setStyleSheet("""
                QDialog {
                    background: #14151a;
                    color: #e2e8f0;
                }
                QTableWidget {
                    background: #181920;
                    color: #cbd5e1;
                    border: 1px solid #282a36;
                    border-radius: 4px;
                    gridline-color: #22242e;
                    font-size: 11.5px;
                }
                QTableWidget::item {
                    padding: 4px 8px;
                }
                QTableWidget::item:selected {
                    background: #2b2e3b;
                    color: #f59e0b;
                }
                QHeaderView::section {
                    background: #1f212a;
                    color: #94a3b8;
                    border: none;
                    border-bottom: 1px solid #2b2d38;
                    padding: 6px 8px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background: #ebedf2;
                    color: #1e293b;
                }
                QTableWidget {
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #d5d9e4;
                    border-radius: 4px;
                    gridline-color: #f1f5f9;
                    font-size: 11.5px;
                }
                QTableWidget::item {
                    padding: 4px 8px;
                }
                QTableWidget::item:selected {
                    background: #fef3c7;
                    color: #b45309;
                }
                QHeaderView::section {
                    background: #f1f4f9;
                    color: #475569;
                    border: none;
                    border-bottom: 1px solid #d5d9e4;
                    padding: 6px 8px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)

    def _populate_table(self):
        self.table.setRowCount(len(self.missing_entries))
        for row, entry in enumerate(self.missing_entries):
            fn = entry.get("filename", "")
            orig_p = entry.get("orig_path", "")
            resolved = entry.get("resolved_path")

            item_fn = QTableWidgetItem(fn)
            item_fn.setToolTip(orig_p)
            self.table.setItem(row, 0, item_fn)

            item_path = QTableWidgetItem(orig_p)
            item_path.setToolTip(orig_p)
            self.table.setItem(row, 1, item_path)

            if resolved and os.path.exists(resolved):
                item_st = QTableWidgetItem("已重新关联")
                item_st.setForeground(QColor("#10b981"))
            else:
                item_st = QTableWidgetItem("未找到")
                item_st.setForeground(QColor("#ef4444"))
            item_st.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, item_st)

    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择底片所在的根文件夹",
            ""
        )
        if not folder:
            return

        folder = normalize_path(folder)
        # Scan folder for missing filenames (cascaded matching)
        # Scan up to 3 levels deep for efficiency
        name_to_path = {}
        target_names = {entry["filename"].lower(): entry["filename"] for entry in self.missing_entries if not entry.get("resolved_path")}

        for root, dirs, files in os.walk(folder):
            # Limit scan depth to avoid hanging on entire drive roots
            depth = os.path.relpath(root, folder).count(os.sep)
            if depth > 3:
                del dirs[:]  # don't recurse deeper
                continue

            for f in files:
                f_lower = f.lower()
                if f_lower in target_names and f_lower not in name_to_path:
                    name_to_path[f_lower] = normalize_path(os.path.join(root, f))
                    if len(name_to_path) == len(target_names):
                        break
            if len(name_to_path) == len(target_names):
                break

        # Apply found paths
        relinked_count = 0
        for entry in self.missing_entries:
            fn_key = entry["filename"].lower()
            if fn_key in name_to_path:
                new_p = name_to_path[fn_key]
                entry["resolved_path"] = new_p
                self.relinked_mapping[entry["orig_path"]] = new_p
                relinked_count += 1

        self._populate_table()

        total_missing = len(self.missing_entries)
        if relinked_count == total_missing:
            self.lbl_status.setText(f"全部 {relinked_count} 张底片已成功重新关联！")
            self.btn_browse.setText("确定载入")
            self.btn_browse.clicked.disconnect()
            self.btn_browse.clicked.connect(self._on_confirm_relink)
        elif relinked_count > 0:
            self.lbl_status.setText(f"已找到并关联 {relinked_count}/{total_missing} 张底片")
            self.btn_browse.setText("继续载入已关联底片")
            self.btn_browse.clicked.disconnect()
            self.btn_browse.clicked.connect(self._on_confirm_relink)
        else:
            QMessageBox.information(
                self,
                "未找到匹配底片",
                f"在文件夹中未扫描到底片列表中对应的原始文件：\n{folder}\n请确认底片存放位置。"
            )

    def _on_confirm_relink(self):
        self.action_result = self.STATUS_RELINKED
        self.accept()

    def _on_keep_offline(self):
        self.action_result = self.STATUS_OFFLINE
        self.accept()

    def get_relinked_mapping(self):
        """Returns dict of {orig_path: new_path} for all successfully relinked photos."""
        return self.relinked_mapping
