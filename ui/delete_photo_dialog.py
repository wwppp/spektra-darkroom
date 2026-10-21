"""
delete_photo_dialog.py - Elegant Darkroom-Themed Delete Photo File Confirmation Dialog
Visually unified with UnsavedSessionDialog for seamless darkroom aesthetic.
"""

import os
from typing import List, Union
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)

from core.path_utils import get_resource_dir
from ui.window_utils import apply_dark_titlebar
from ui.theme_manager import theme_manager


class DeletePhotoConfirmDialog(QDialog):
    """Custom modal confirmation dialog matching UnsavedSessionDialog style."""

    CONFIRMED = 1
    CANCELLED = 0

    def __init__(self, target_names: Union[str, List[str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("删除底片文件")
        self.setFixedSize(450, 185)
        self.setModal(True)
        self.action_result = self.CANCELLED

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        if isinstance(target_names, str):
            self.target_names = [target_names]
        else:
            self.target_names = list(target_names)

        self._init_ui()

    def exec(self):
        super().exec()
        return self.action_result

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self):
        is_dark = theme_manager.is_dark()
        if is_dark:
            self.setStyleSheet("""
                QDialog {
                    background-color: #14151a;
                    color: #e2e8f0;
                }
                QLabel {
                    color: #cbd5e1;
                }
                QPushButton {
                    background: #1f212a;
                    color: #e2e8f0;
                    border: 1px solid #333647;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 500;
                    height: 28px;
                }
                QPushButton:hover {
                    background: #282b37;
                    border-color: #f59e0b;
                    color: #f59e0b;
                }
                QPushButton:pressed {
                    background: #14151a;
                }
                QPushButton#btnDelete {
                    background: #dc2626;
                    color: #ffffff;
                    border: none;
                    font-weight: bold;
                }
                QPushButton#btnDelete:hover {
                    background: #ef4444;
                }
                QPushButton#btnDelete:pressed {
                    background: #b91c1c;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: #ebedf2;
                    color: #1e293b;
                }
                QLabel {
                    color: #1e293b;
                }
                QPushButton {
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 500;
                    height: 28px;
                }
                QPushButton:hover {
                    background: #f8fafc;
                    border-color: #f59e0b;
                    color: #d97706;
                }
                QPushButton:pressed {
                    background: #e2e8f0;
                    color: #b45309;
                }
                QPushButton#btnDelete {
                    background: #dc2626;
                    color: #ffffff;
                    border: none;
                    font-weight: bold;
                }
                QPushButton#btnDelete:hover {
                    background: #ef4444;
                }
                QPushButton#btnDelete:pressed {
                    background: #b91c1c;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(18)

        # Content Row
        content_row = QHBoxLayout()
        content_row.setSpacing(18)
        content_row.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Elegant Amber Warning Badge
        badge = QLabel()
        badge.setFixedSize(40, 40)
        badge_bg = "#261f14" if is_dark else "#fef3c7"
        badge.setStyleSheet(f"""
            background: {badge_bg};
            border: 1px solid #f59e0b;
            border-radius: 8px;
            color: #f59e0b;
            font-size: 20px;
            font-weight: bold;
        """)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setText("!")
        content_row.addWidget(badge)

        # Text Container
        text_layout = QVBoxLayout()
        text_layout.setSpacing(6)

        title_col = "#f8fafc" if is_dark else "#0f172a"
        title_lbl = QLabel("删除底片文件")
        title_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {title_col};")
        text_layout.addWidget(title_lbl)

        count = len(self.target_names)
        if count == 1:
            fn = self.target_names[0]
            desc_text = f"确定要将底片文件 “{fn}” 移至回收站吗？\n此操作将从磁盘中删除该文件并从底片库中移除。"
        else:
            desc_text = f"确定要将选中的 {count} 个底片文件移至回收站吗？\n此操作将从磁盘中删除这些文件并从底片库中移除。"

        desc_col = "#94a3b8" if is_dark else "#475569"
        desc_lbl = QLabel(desc_text)
        desc_lbl.setStyleSheet(f"font-size: 12px; color: {desc_col}; line-height: 1.4;")
        desc_lbl.setWordWrap(True)
        text_layout.addWidget(desc_lbl)

        content_row.addLayout(text_layout, 1)
        layout.addLayout(content_row, 1)

        # Footer Buttons
        footer = QHBoxLayout()
        footer.setSpacing(10)
        footer.addStretch(1)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFixedWidth(75)
        self.btn_cancel.clicked.connect(self._on_cancel)
        footer.addWidget(self.btn_cancel)

        self.btn_delete = QPushButton("移至回收站", objectName="btnDelete")
        self.btn_delete.setFixedWidth(90)
        self.btn_delete.setDefault(True)
        self.btn_delete.clicked.connect(self._on_delete)
        footer.addWidget(self.btn_delete)

        layout.addLayout(footer)

    def _on_delete(self):
        self.action_result = self.CONFIRMED
        self.accept()

    def _on_cancel(self):
        self.action_result = self.CANCELLED
        self.reject()
