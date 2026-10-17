"""
unsaved_session_dialog.py - Elegant Darkroom-Themed Unsaved Session Exit Guard Dialog
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget
)

from path_utils import get_resource_dir
from ui.window_utils import apply_dark_titlebar
from ui.theme_manager import theme_manager


class UnsavedSessionDialog(QDialog):
    SAVE = 1
    DISCARD = 2
    CANCEL = 0

    def __init__(self, session_name="未命名会话", parent=None):
        super().__init__(parent)
        self.setWindowTitle("未保存的会话")
        self.setFixedSize(450, 185)
        self.setModal(True)
        self.action_result = self.CANCEL

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.sess_name = session_name
        self._init_ui(session_name)

    def exec(self):
        super().exec()
        return self.action_result

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self, session_name):
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
                QPushButton#btnSave {
                    background: #f59e0b;
                    color: #111111;
                    border: none;
                    font-weight: bold;
                }
                QPushButton#btnSave:hover {
                    background: #fbbf24;
                }
                QPushButton#btnSave:pressed {
                    background: #d97706;
                }
                QPushButton#btnDiscard:hover {
                    border-color: #ef4444;
                    color: #f87171;
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
                QPushButton#btnSave {
                    background: #f59e0b;
                    color: #111111;
                    border: none;
                    font-weight: bold;
                }
                QPushButton#btnSave:hover {
                    background: #fbbf24;
                }
                QPushButton#btnSave:pressed {
                    background: #d97706;
                }
                QPushButton#btnDiscard:hover {
                    background: #fef2f2;
                    border-color: #ef4444;
                    color: #dc2626;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(18)

        # Content Row
        content_row = QHBoxLayout()
        content_row.setSpacing(18)
        content_row.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Elegant Amber Warning Icon Badge
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
        title_lbl = QLabel("未保存的会话")
        title_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {title_col};")
        text_layout.addWidget(title_lbl)

        desc_col = "#94a3b8" if is_dark else "#475569"
        desc_lbl = QLabel(f"会话「{session_name}」有尚未保存的修改。\n是否在退出前保存当前会话工程？")
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
        self.btn_cancel.setFixedWidth(70)
        self.btn_cancel.clicked.connect(self._on_cancel)
        footer.addWidget(self.btn_cancel)

        self.btn_discard = QPushButton("不保存直接退出", objectName="btnDiscard")
        self.btn_discard.setFixedWidth(118)
        self.btn_discard.clicked.connect(self._on_discard)
        footer.addWidget(self.btn_discard)

        self.btn_save = QPushButton("保存", objectName="btnSave")
        self.btn_save.setFixedWidth(75)
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._on_save)
        footer.addWidget(self.btn_save)

        layout.addLayout(footer)

    def _on_save(self):
        self.action_result = self.SAVE
        self.accept()

    def _on_discard(self):
        self.action_result = self.DISCARD
        self.accept()

    def _on_cancel(self):
        self.action_result = self.CANCEL
        self.reject()
