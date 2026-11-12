"""校样设置对话框（P1）。

对齐 Photoshop 的「视图 → 校样设置」：选一个目标空间，视口里显示「导出成它会
是什么样」。RGB 目标 = 在目标域里 clip（相对比色），与导出用的是同一组矩阵，
所以「校样 = 导出」这个闭环成立，不是两套算法各算一遍。

不做的事：CMYK / 纸白模拟 / 可感知意图。理由见 `core/proof.py` 顶部与
`docs/viewport-colorspace-plan.md` §5：这三样都需要 LUT 型 CMM 路径，而这里
只有矩阵，硬做出来的是"看起来像校样"的东西。
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout,
)

from core import proof
from core.path_utils import get_resource_dir
from ui.theme_manager import theme_manager
from ui.window_utils import apply_dark_titlebar


class ProofSetupDialog(QDialog):
    """返回一个 ``proof.ProofSettings``（点取消则返回 None）。"""

    def __init__(self, settings: proof.ProofSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("校样设置")
        self.setModal(True)
        self.setFixedWidth(430)
        self._initial = settings
        self.result_settings = None

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._build_ui()
        self._apply_theme()

    # -- UI -------------------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        self.chk_on = QCheckBox("启用校样")
        self.chk_on.setChecked(self._initial.on)
        self.chk_on.setToolTip("快捷键 Ctrl+Shift+Y")
        root.addWidget(self.chk_on)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(20, 0, 0, 0)
        lbl = QLabel("校样目标")
        grid.addWidget(lbl, 0, 0)
        self.combo_target = QComboBox()
        for t in proof.TARGETS:
            self.combo_target.addItem(t, t)
        idx = self.combo_target.findData(self._initial.canonical_target())
        self.combo_target.setCurrentIndex(max(0, idx))
        self.combo_target.setToolTip("视口会显示「导出成这个色彩空间」的样子")
        grid.addWidget(self.combo_target, 0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        self.chk_warn = QCheckBox("色域警告（超出目标色域的像素叠品红）")
        self.chk_warn.setChecked(self._initial.warn)
        self.chk_warn.setToolTip("快捷键 Ctrl+Alt+Y")
        root.addWidget(self.chk_warn)

        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        root.addWidget(self.lbl_note)

        root.addStretch(1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFixedWidth(84)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("好")
        self.btn_ok.setFixedWidth(84)
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._accept)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        root.addLayout(btns)

        self.chk_on.toggled.connect(self._refresh_note)
        self.combo_target.currentIndexChanged.connect(self._refresh_note)
        self._refresh_note()

    def _refresh_note(self):
        on = self.chk_on.isChecked()
        self.combo_target.setEnabled(on)
        self.chk_warn.setEnabled(on)
        if not on:
            self.lbl_note.setText("校样关闭时，视口按显示器 profile 直接显示。")
            return
        t = self.combo_target.currentData()
        if t == "sRGB":
            self.lbl_note.setText(
                "sRGB 是最常见的交付目标；超出它的颜色会被夹到边界（相对比色）。")
        else:
            self.lbl_note.setText(
                f"「{t}」的色域之外的像素会被夹到边界。与导出该色彩空间用的是同一组矩阵，"
                "所以看到的和导出的一致。")

    def _apply_theme(self):
        if not theme_manager.is_dark():
            return
        self.setStyleSheet("""
            QDialog { background-color: #14151a; }
            QLabel { color: #cbd5e1; font-size: 12px; }
            QCheckBox { color: #e2e8f0; font-size: 12px; spacing: 8px; }
            QComboBox {
                background: #1f212a; color: #e2e8f0; border: 1px solid #333647;
                border-radius: 4px; padding: 4px 8px; min-height: 24px; font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background: #1f212a; color: #e2e8f0; selection-background-color: #3a3d4d;
            }
            QPushButton {
                background: #1f212a; color: #e2e8f0; border: 1px solid #333647;
                border-radius: 4px; height: 28px; font-size: 12px; font-weight: 500;
            }
            QPushButton:hover { background: #2a2d38; }
            QPushButton:default { border-color: #f59e0b; color: #fbbf24; }
        """)

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _accept(self):
        self.result_settings = proof.ProofSettings(
            on=self.chk_on.isChecked(),
            target=self.combo_target.currentData(),
            warn=self.chk_warn.isChecked(),
        )
        self.accept()

    def exec(self):
        super().exec()
        return self.result_settings
