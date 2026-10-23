"""
confirm_dialog.py - Elegant Darkroom-Themed Modal Confirmation & Info Dialog
Visually unified with UnsavedSessionDialog and DeletePhotoConfirmDialog.
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)

from core.path_utils import get_resource_dir
from ui.window_utils import apply_dark_titlebar
from ui.theme_manager import theme_manager


def _get_dialog_stylesheet(is_dark: bool) -> str:
    if is_dark:
        return """
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
                padding: 0 12px;
            }
            QPushButton:hover {
                background: #282b37;
                border-color: #f59e0b;
                color: #f59e0b;
            }
            QPushButton:pressed {
                background: #14151a;
            }
            QPushButton#btnPrimary {
                background: #f59e0b;
                color: #111111;
                border: none;
                font-weight: bold;
            }
            QPushButton#btnPrimary:hover {
                background: #fbbf24;
            }
            QPushButton#btnPrimary:pressed {
                background: #d97706;
            }
            QPushButton#btnDanger {
                background: #dc2626;
                color: #ffffff;
                border: none;
                font-weight: bold;
            }
            QPushButton#btnDanger:hover {
                background: #ef4444;
            }
            QPushButton#btnDanger:pressed {
                background: #b91c1c;
            }
        """
    else:
        return """
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
                padding: 0 12px;
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
            QPushButton#btnPrimary {
                background: #f59e0b;
                color: #111111;
                border: none;
                font-weight: bold;
            }
            QPushButton#btnPrimary:hover {
                background: #fbbf24;
            }
            QPushButton#btnPrimary:pressed {
                background: #d97706;
            }
            QPushButton#btnDanger {
                background: #dc2626;
                color: #ffffff;
                border: none;
                font-weight: bold;
            }
            QPushButton#btnDanger:hover {
                background: #ef4444;
            }
            QPushButton#btnDanger:pressed {
                background: #b91c1c;
            }
        """


class DarkroomConfirmDialog(QDialog):
    """Elegant Darkroom-themed confirmation dialog with Cancel and Primary action buttons."""

    CONFIRMED = 1
    CANCELLED = 0

    def __init__(self, title: str, heading: str, description: str,
                 confirm_text: str = "确定", cancel_text: str = "取消",
                 badge_symbol: str = "?", is_danger: bool = False,
                 parent=None, width: int = 470):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(width)
        self.setModal(True)
        self.action_result = self.CANCELLED

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.title_text = heading
        self.desc_text = description
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text
        self.badge_symbol = badge_symbol
        self.is_danger = is_danger

        self._init_ui()

    def exec(self):
        super().exec()
        return self.action_result

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self):
        is_dark = theme_manager.is_dark()
        self.setStyleSheet(_get_dialog_stylesheet(is_dark))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(18)

        # Content Row
        content_row = QHBoxLayout()
        content_row.setSpacing(18)
        content_row.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Badge
        badge = QLabel()
        badge.setFixedSize(40, 40)
        badge_bg = "#261f14" if is_dark else "#fef3c7"
        badge_border = "#dc2626" if self.is_danger else "#f59e0b"
        badge_color = "#ef4444" if self.is_danger else "#f59e0b"
        badge.setStyleSheet(f"""
            background: {badge_bg};
            border: 1px solid {badge_border};
            border-radius: 8px;
            color: {badge_color};
            font-size: 20px;
            font-weight: bold;
        """)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setText(self.badge_symbol)
        content_row.addWidget(badge)

        # Text Container
        text_layout = QVBoxLayout()
        text_layout.setSpacing(6)

        title_col = "#f8fafc" if is_dark else "#0f172a"
        title_lbl = QLabel(self.title_text)
        title_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {title_col};")
        text_layout.addWidget(title_lbl)

        desc_col = "#94a3b8" if is_dark else "#475569"
        desc_lbl = QLabel(self.desc_text)
        desc_lbl.setStyleSheet(f"font-size: 12px; color: {desc_col}; line-height: 1.4;")
        desc_lbl.setWordWrap(True)
        text_layout.addWidget(desc_lbl)

        content_row.addLayout(text_layout, 1)
        layout.addLayout(content_row, 1)

        # Footer Buttons
        footer = QHBoxLayout()
        footer.setSpacing(10)
        footer.addStretch(1)

        self.btn_cancel = QPushButton(self.cancel_text)
        self.btn_cancel.setMinimumWidth(72)
        self.btn_cancel.clicked.connect(self._on_cancel)
        footer.addWidget(self.btn_cancel)

        obj_name = "btnDanger" if self.is_danger else "btnPrimary"
        self.btn_confirm = QPushButton(self.confirm_text, objectName=obj_name)
        self.btn_confirm.setMinimumWidth(80)
        self.btn_confirm.setDefault(True)
        self.btn_confirm.clicked.connect(self._on_confirm)
        footer.addWidget(self.btn_confirm)

        layout.addLayout(footer)

    def _on_confirm(self):
        self.action_result = self.CONFIRMED
        self.accept()

    def _on_cancel(self):
        self.action_result = self.CANCELLED
        self.reject()


class DarkroomInfoDialog(QDialog):
    """Elegant Darkroom-themed informational dialog with single confirmation button."""

    def __init__(self, title: str, heading: str, description: str,
                 ok_text: str = "确定", badge_symbol: str = "!",
                 parent=None, width: int = 470):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(width)
        self.setModal(True)

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.title_text = heading
        self.desc_text = description
        self.ok_text = ok_text
        self.badge_symbol = badge_symbol

        self._init_ui()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self):
        is_dark = theme_manager.is_dark()
        self.setStyleSheet(_get_dialog_stylesheet(is_dark))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(18)

        # Content Row
        content_row = QHBoxLayout()
        content_row.setSpacing(18)
        content_row.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Badge
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
        badge.setText(self.badge_symbol)
        content_row.addWidget(badge)

        # Text Container
        text_layout = QVBoxLayout()
        text_layout.setSpacing(6)

        title_col = "#f8fafc" if is_dark else "#0f172a"
        title_lbl = QLabel(self.title_text)
        title_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {title_col};")
        text_layout.addWidget(title_lbl)

        desc_col = "#94a3b8" if is_dark else "#475569"
        desc_lbl = QLabel(self.desc_text)
        desc_lbl.setStyleSheet(f"font-size: 12px; color: {desc_col}; line-height: 1.4;")
        desc_lbl.setWordWrap(True)
        text_layout.addWidget(desc_lbl)

        content_row.addLayout(text_layout, 1)
        layout.addLayout(content_row, 1)

        # Footer Buttons
        footer = QHBoxLayout()
        footer.addStretch(1)

        self.btn_ok = QPushButton(self.ok_text, objectName="btnPrimary")
        self.btn_ok.setMinimumWidth(80)
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self.accept)
        footer.addWidget(self.btn_ok)

        layout.addLayout(footer)


def show_darkroom_confirm(parent, title: str, heading: str, description: str,
                          confirm_text: str = "确定", cancel_text: str = "取消",
                          badge_symbol: str = "?", is_danger: bool = False,
                          width: int = 470) -> bool:
    """Convenience helper to show a confirmation dialog matching UnsavedSessionDialog style."""
    dlg = DarkroomConfirmDialog(
        title=title,
        heading=heading,
        description=description,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
        badge_symbol=badge_symbol,
        is_danger=is_danger,
        parent=parent,
        width=width
    )
    return dlg.exec() == DarkroomConfirmDialog.CONFIRMED


def show_darkroom_info(parent, title: str, heading: str, description: str,
                       ok_text: str = "确定", badge_symbol: str = "!",
                       width: int = 470) -> None:
    """Convenience helper to show an informational dialog matching UnsavedSessionDialog style."""
    dlg = DarkroomInfoDialog(
        title=title,
        heading=heading,
        description=description,
        ok_text=ok_text,
        badge_symbol=badge_symbol,
        parent=parent,
        width=width
    )
    dlg.exec()
