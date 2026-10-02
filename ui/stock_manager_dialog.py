"""
stock_manager_dialog.py - Professional Film & Paper Profile Library Manager
Supports:
- Multi-selection of film/paper stock profiles
- Export selected to 3D LUT (.cube)
- Export selected to SpektraDarkroom physical profile (.json)
- Export all profiles to designated directory
- Import external SpektraDarkroom physical profiles (.json) and 3D LUTs (.cube)
- Discard/remove custom or hidden profiles
"""

import os
import json
import shutil
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QColor, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QFrame, QCheckBox,
    QWidget
)

from path_utils import get_icon_path, get_resource_dir
from ui.window_utils import apply_dark_titlebar


class StockManagerDialog(QDialog):
    """Full-featured stock library manager for film negative and print paper profiles."""
    stocksChanged = Signal()

    def __init__(self, mode="film", engine=None, parent=None):
        super().__init__(parent)
        self.mode = mode  # "film" or "paper"
        self.engine = engine
        self.is_film = (mode == "film")
        self.term = "胶卷" if self.is_film else "相纸"

        title = f"{self.term}库管理"
        self.setWindowTitle(title)
        self.resize(980, 620)
        self.setModal(True)

        icon_filename = "dlg_film.png" if self.is_film else "dlg_paper.png"
        icon_path = os.path.join(get_resource_dir(), "icons", icon_filename)
        if not os.path.exists(icon_path):
            icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._init_ui()
        self._load_stocks()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self):
        chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")

        self.setStyleSheet(f"""
            QDialog {{
                background: #14151a;
                color: #e2e8f0;
            }}
            QLabel {{
                color: #e2e8f0;
            }}
            QTableWidget {{
                background: #101115;
                color: #e2e8f0;
                gridline-color: #242735;
                border: 1px solid #282a36;
                border-radius: 6px;
                selection-background-color: #1e222d;
                selection-color: #f59e0b;
                font-size: 12px;
                outline: none;
            }}
            QTableWidget::item {{
                outline: none;
                border: none;
            }}
            QTableWidget::item:focus {{
                outline: none;
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: #1e222d;
                outline: none;
                border: none;
            }}
            QHeaderView::section {{
                background: #1a1c24;
                color: #94a3b8;
                border: none;
                border-bottom: 1px solid #2d303f;
                padding: 6px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton {{
                background: #1f212a;
                color: #e2e8f0;
                border: 1px solid #333647;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: #282b37;
                border-color: #f59e0b;
                color: #f59e0b;
            }}
            QPushButton:pressed {{
                background: #14151a;
                color: #d97706;
                border-color: #d97706;
            }}
            QPushButton#btnPrimary {{
                background: #f59e0b;
                color: #111;
                border: none;
                font-weight: bold;
            }}
            QPushButton#btnPrimary:hover {{
                background: #d97706;
            }}
            QPushButton#btnPrimary:pressed {{
                background: #b45309;
            }}
            QPushButton#btnDanger {{
                background: #201a1d;
                color: #f87171;
                border: 1px solid #4a282f;
            }}
            QPushButton#btnDanger:hover {{
                background: #2e1d23;
                border-color: #ef4444;
                color: #ffffff;
            }}
            QPushButton#btnDanger:pressed {{
                background: #3b1219;
                color: #ef4444;
                border-color: #ef4444;
            }}
            QCheckBox {{
                spacing: 6px;
                color: #cbd5e1;
                font-size: 12px;
                outline: none;
            }}
            QCheckBox:focus {{
                outline: none;
            }}
            QCheckBox::indicator {{
                width: 12px;
                height: 12px;
                border: 1px solid #8b92a5;
                border-radius: 2px;
                background: transparent;
                outline: none;
            }}
            QCheckBox::indicator:hover {{
                border-color: #f59e0b;
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid #f59e0b;
                background: #14151a;
                image: url("{chk_amber}");
            }}
            QScrollBar:vertical {{
                background: rgba(18, 19, 24, 0.75);
                width: 6px;
                margin: 0px;
                border-radius: 3px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: #474d61;
                min-height: 24px;
                border-radius: 3px;
                border: none;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #f59e0b;
            }}
            QScrollBar::handle:vertical:pressed {{
                background: #d97706;
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
                background: rgba(18, 19, 24, 0.75);
                height: 6px;
                margin: 0px;
                border-radius: 3px;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: #474d61;
                min-width: 24px;
                border-radius: 3px;
                border: none;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: #f59e0b;
            }}
            QScrollBar::handle:horizontal:pressed {{
                background: #d97706;
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Header Info
        header_row = QHBoxLayout()
        lbl_head = QLabel(f"{self.term}档案库")
        lbl_head.setStyleSheet("font-size: 15px; font-weight: bold; color: #f59e0b;")
        header_row.addWidget(lbl_head)

        self.lbl_count = QLabel(f"共 0 个{self.term}")
        self.lbl_count.setStyleSheet("color: #71717a; font-size: 11px; margin-left: 8px;")
        header_row.addWidget(self.lbl_count)
        header_row.addStretch()

        # Action Buttons (with user inverted dark-mode PNG icons)
        icon_import = os.path.join(get_resource_dir(), "icons", "dlg_import.png")
        icon_export = os.path.join(get_resource_dir(), "icons", "dlg_export.png")
        icon_save = os.path.join(get_resource_dir(), "icons", "dlg_save.png")

        btn_import = QPushButton(f"导入{self.term}配置 (.json)")
        if os.path.exists(icon_import):
            btn_import.setIcon(QIcon(icon_import))
            btn_import.setIconSize(QSize(14, 14))
        btn_import.setToolTip(f"导入 SpektraDarkroom 完整物理参数配置 (.json)")
        btn_import.clicked.connect(self._import_preset)
        header_row.addWidget(btn_import)

        btn_export_sel_json = QPushButton(f"导出所选{self.term}配置 (.json)")
        if os.path.exists(icon_save):
            btn_export_sel_json.setIcon(QIcon(icon_save))
            btn_export_sel_json.setIconSize(QSize(14, 14))
        btn_export_sel_json.setToolTip(f"导出所选{self.term}的完整物理感光曲线与化学反应光谱参数文件 (.json)")
        btn_export_sel_json.clicked.connect(self._export_selected_json)
        header_row.addWidget(btn_export_sel_json)

        layout.addLayout(header_row)

        # Stocks Table
        self.table = QTableWidget(0, 5)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setHorizontalHeaderLabels(["", f"{self.term}名称", f"{self.term}风格 / 标识", "特点与说明", "类型"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 36)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(1, 220)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(2, 130)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 75)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

        # Footer Actions
        footer = QHBoxLayout()
        btn_select_all = QPushButton("全选")
        btn_select_all.clicked.connect(self._toggle_select_all)
        footer.addWidget(btn_select_all)

        icon_discard = os.path.join(get_resource_dir(), "icons", "dlg_discard.png")
        btn_delete_sel = QPushButton("丢弃所选", objectName="btnDanger")
        if os.path.exists(icon_discard):
            btn_delete_sel.setIcon(QIcon(icon_discard))
            btn_delete_sel.setIconSize(QSize(14, 14))
        btn_delete_sel.clicked.connect(self._delete_selected)
        footer.addWidget(btn_delete_sel)

        footer.addStretch()

        btn_close = QPushButton("关闭", objectName="btnPrimary")
        btn_close.setFixedWidth(80)
        btn_close.clicked.connect(self.accept)
        footer.addWidget(btn_close)

        layout.addLayout(footer)

    def _create_dialog_msg(self, title, text, icon_type=QMessageBox.Icon.Information):
        mb = QMessageBox(self)
        apply_dark_titlebar(mb)
        mb.setWindowTitle(title)
        mb.setText(text)
        mb.setIcon(icon_type)
        icon_f = "dlg_film.png" if self.is_film else "dlg_paper.png"
        ip = os.path.join(get_resource_dir(), "icons", icon_f)
        if os.path.exists(ip):
            mb.setWindowIcon(QIcon(ip))
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
        return mb

    def _show_info(self, title, text):
        mb = self._create_dialog_msg(title, text, QMessageBox.Icon.Information)
        mb.exec()

    def _show_question(self, title, text):
        mb = self._create_dialog_msg(title, text, QMessageBox.Icon.Question)
        mb.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        mb.setDefaultButton(QMessageBox.StandardButton.No)
        return mb.exec() == QMessageBox.StandardButton.Yes

    def _load_stocks(self):
        self.table.setRowCount(0)
        if not self.engine:
            return

        if self.is_film:
            raw_groups = self.engine.get_film_stocks()
            stocks = []
            for g in raw_groups:
                stocks.extend(g.get("stocks", []))
        else:
            stocks = self.engine.get_paper_stocks()

        self.stocks_data = stocks
        self.lbl_count.setText(f"共 {len(stocks)} 个{self.term}")
        self.table.setRowCount(len(stocks))

        for row, s in enumerate(stocks):
            # Checkbox item centered without focus border
            chk = QCheckBox()
            chk.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            chk_wrap = QWidget()
            chk_wrap_layout = QHBoxLayout(chk_wrap)
            chk_wrap_layout.setContentsMargins(0, 0, 0, 0)
            chk_wrap_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk_wrap_layout.addWidget(chk)
            self.table.setCellWidget(row, 0, chk_wrap)

            # Name
            it_name = QTableWidgetItem(s.get("name", s.get("id", "")))
            it_name.setData(Qt.ItemDataRole.UserRole, s.get("id"))
            self.table.setItem(row, 1, it_name)

            # Badge
            it_badge = QTableWidgetItem(s.get("badge", "经典"))
            it_badge.setForeground(QColor("#f59e0b"))
            self.table.setItem(row, 2, it_badge)

            # Description
            it_desc = QTableWidgetItem(s.get("desc", ""))
            it_desc.setForeground(QColor("#94a3b8"))
            self.table.setItem(row, 3, it_desc)

            # Type (Built-in or Custom)
            is_custom = s.get("is_custom", False)
            it_type = QTableWidgetItem("自定义" if is_custom else "内置")
            it_type.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it_type.setForeground(QColor("#38bdf8") if is_custom else QColor("#64748b"))
            self.table.setItem(row, 4, it_type)

    def _toggle_select_all(self):
        all_checked = True
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            chk = w.findChild(QCheckBox) if w else None
            if chk and not chk.isChecked():
                all_checked = False
                break
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            chk = w.findChild(QCheckBox) if w else None
            if chk:
                chk.setChecked(not all_checked)

    def _get_selected_stock_ids(self):
        sel_ids = []
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            chk = w.findChild(QCheckBox) if w else None
            if chk and chk.isChecked():
                it = self.table.item(row, 1)
                if it:
                    sel_ids.append(it.data(Qt.ItemDataRole.UserRole))
        return sel_ids

    def _export_selected_json(self):
        sel_ids = self._get_selected_stock_ids()
        if not sel_ids:
            self._show_info("提示", f"请先勾选需要导出的{self.term}。")
            return

        if len(sel_ids) == 1:
            sid = sel_ids[0]
            out_file, _ = QFileDialog.getSaveFileName(self, f"导出{self.term}物理参数配置", f"{sid}.json", "JSON 配置 (*.json);;所有文件 (*.*)")
            if out_file:
                export_stock_profile_json(self.engine, sid, out_file)
                self._show_info("完成", f"{self.term}物理配置已成功导出至:\n{out_file}")
        else:
            folder = QFileDialog.getExistingDirectory(self, f"选择批量导出{self.term}物理配置目标文件夹")
            if folder:
                for sid in sel_ids:
                    out_file = os.path.join(folder, f"{sid}.json")
                    export_stock_profile_json(self.engine, sid, out_file)
                self._show_info("完成", f"已成功导出 {len(sel_ids)} 个{self.term}物理配置至目标文件夹。")

    def _export_all(self):
        folder = QFileDialog.getExistingDirectory(self, f"选择一键导出全部{self.term}的目标文件夹")
        if not folder:
            return
        total = 0
        for s in getattr(self, 'stocks_data', []):
            sid = s.get("id")
            if sid:
                out_json = os.path.join(folder, f"{sid}.json")
                export_stock_profile_json(self.engine, sid, out_json)
                total += 1
        self._show_info("完成", f"已成功一键导出全部 {total} 个{self.term}物理配置 (.json) 至:\n{folder}")

    def _import_preset(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            f"导入{self.term}物理配置",
            "",
            "JSON 配置 (*.json);;所有文件 (*.*)"
        )
        if not files:
            return

        imported = 0
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext == ".json":
                try:
                    with open(f, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    info = data.get("info", {})
                    stock_id = info.get("stock") or os.path.splitext(os.path.basename(f))[0]
                    dest = os.path.join(self.engine.profiles_dir, f"{stock_id}.json")
                    shutil.copy2(f, dest)
                    imported += 1
                except Exception as e:
                    print(f"[StockManager] 导入物理配置文件失败 {f}: {e}")

        self._load_stocks()
        self.stocksChanged.emit()
        self._show_info("导入完成", f"已成功导入 {imported} 个{self.term}物理配置。")

    def _delete_selected(self):
        sel_ids = self._get_selected_stock_ids()
        if not sel_ids:
            self._show_info("提示", f"请先勾选需要丢弃/移除的{self.term}。")
            return
        if self._show_question("确认丢弃", f"确定要从当前列表中丢弃所选的 {len(sel_ids)} 个{self.term}吗？"):
            for sid in sel_ids:
                if hasattr(self.parent(), '_remove_film_lut'):
                    self.parent()._remove_film_lut(sid)
                elif hasattr(self.parent(), '_remove_paper_lut'):
                    self.parent()._remove_paper_lut(sid)
            self._load_stocks()
            self.stocksChanged.emit()


def write_cube_file(out_path, lut_3d, title="LUT"):
    size = lut_3d.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    val = lut_3d[b, g, r]
                    f.write(f"{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}\n")


def export_stock_lut(engine, stock_id, out_file, is_film=True):
    try:
        if is_film:
            lut = engine.get_3d_lut(stock_id, "kodak_2383", lut_size=33)
        else:
            lut = engine.get_3d_lut("kodak_portra_400", stock_id, lut_size=33)
        write_cube_file(out_file, lut, title=stock_id)
        return True
    except Exception as e:
        print(f"[StockManager] 导出 LUT 失败 {stock_id}: {e}")
        return False


def export_stock_profile_json(engine, stock_id, out_file):
    """Export complete SpektraDarkroom physical profile configuration file (.json)."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        # Check if local profile file exists
        src_json = os.path.join(engine.profiles_dir, f"{stock_id}.json")
        if os.path.exists(src_json):
            shutil.copy2(src_json, out_file)
            return True
        # Or dump from engine loaded profile
        prof = engine._load_profile(stock_id)
        if prof:
            from spektrafilm.profiles.io import profile_to_dict
            data = profile_to_dict(prof)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
    except Exception as e:
        print(f"[StockManager] 导出物理配置文件失败 {stock_id}: {e}")
    return False
