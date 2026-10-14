"""Adobe Camera Raw / Lightroom Style Professional Export Dialog System
Provides:
- BaseExportDialog: Shared render settings (Engine, Format, Color Space, Bit Depth, DPI, Codec parameters)
- SingleExportDialog: Tailored for single photo export with explicit file destination and save file picker
- BatchExportDialog: Tailored for batch multi-photo export with output directory, count badge, and naming rules
- ExportImageDialog: Compatibility factory mapping to SingleExportDialog or BatchExportDialog
- Zero layout shift, strictly decoupled file vs folder paths, concise professional typography.
"""

import os
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QLineEdit, QPushButton, QFileDialog, QFrame, QWidget,
    QStackedWidget, QCheckBox, QProgressBar
)

from path_utils import get_icon_path, get_resource_dir, normalize_path
from ui.window_utils import apply_dark_titlebar
import config_manager


class BaseExportDialog(QDialog):
    """Base dialog encapsulating shared darkroom rendering and color management settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)

        icon_path = os.path.join(get_resource_dir(), "icons", "dlg_export.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_settings_card(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("settingsCard")
        card.setStyleSheet("""
            QFrame#settingsCard {
                background: #1a1a1e;
                border: 1px solid #2b2d38;
                border-radius: 8px;
                padding: 14px;
            }
            QLabel {
                border: none;
                background: transparent;
                color: #c4c4cc;
                font-size: 12px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(11)

        # Row 0: Rendering Engine Selection
        r0 = QHBoxLayout()
        r0.setSpacing(12)
        lbl_eng = QLabel("冲印引擎:")
        lbl_eng.setFixedWidth(65)
        r0.addWidget(lbl_eng)
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("spektradarkroom（较快）", "wysiwyg")
        self.combo_engine.addItem("spektrafilm（极慢）", "official")
        self.combo_engine.setItemData(0, "多核并行硬件加速冲印，与视口画面 100% 视觉对齐", Qt.ItemDataRole.ToolTipRole)
        self.combo_engine.setItemData(1, "官方 spektrafilm 原生端到端物理光谱模拟管线", Qt.ItemDataRole.ToolTipRole)
        r0.addWidget(self.combo_engine, 1)
        card_layout.addLayout(r0)

        # Row 1: Format & Color Space
        r1 = QHBoxLayout()
        r1.setSpacing(12)

        lbl_fmt = QLabel("文件格式:")
        lbl_fmt.setFixedWidth(65)
        r1.addWidget(lbl_fmt)
        self.combo_format = QComboBox()
        self.combo_format.addItem("JPEG (*.jpg)", "jpeg")
        self.combo_format.addItem("TIFF (*.tif)", "tiff")
        self.combo_format.addItem("PNG (*.png)", "png")
        r1.addWidget(self.combo_format, 1)

        lbl_cs = QLabel("色彩空间:")
        lbl_cs.setFixedWidth(65)
        r1.addWidget(lbl_cs)
        self.combo_colorspace = QComboBox()
        self.combo_colorspace.addItem("sRGB", "sRGB")
        self.combo_colorspace.addItem("Display P3", "Display P3")
        self.combo_colorspace.addItem("Adobe RGB (1998)", "Adobe RGB (1998)")
        self.combo_colorspace.addItem("ProPhoto RGB", "ProPhoto RGB")
        r1.addWidget(self.combo_colorspace, 1)

        card_layout.addLayout(r1)

        # Row 2: Resolution (DPI) & Bit Depth
        r2 = QHBoxLayout()
        r2.setSpacing(12)
        lbl_dpi = QLabel("输出分辨率:")
        lbl_dpi.setFixedWidth(70)
        r2.addWidget(lbl_dpi)

        self.combo_dpi = QComboBox()
        self.combo_dpi.addItem("72 DPI", 72)
        self.combo_dpi.addItem("300 DPI", 300)
        self.combo_dpi.addItem("600 DPI", 600)
        self.combo_dpi.setCurrentIndex(1)
        r2.addWidget(self.combo_dpi, 1)

        lbl_bit = QLabel("输出位深:")
        lbl_bit.setFixedWidth(65)
        r2.addWidget(lbl_bit)
        self.combo_depth = QComboBox()
        r2.addWidget(self.combo_depth, 1)

        card_layout.addLayout(r2)

        # Row 3: Format-Specific Settings Stack
        self.stack_settings = QStackedWidget()
        self.stack_settings.setFixedHeight(82)

        # --- Stack Page 0: JPEG Options ---
        page_jpeg = QWidget()
        l_jpeg = QVBoxLayout(page_jpeg)
        l_jpeg.setContentsMargins(0, 2, 0, 0)
        l_jpeg.setSpacing(8)

        row_jpeg_opts = QHBoxLayout()
        row_jpeg_opts.setSpacing(10)

        lbl_scale = QLabel("输出尺寸:")
        lbl_scale.setFixedWidth(65)
        row_jpeg_opts.addWidget(lbl_scale)

        self.combo_scale = QComboBox()
        self.combo_scale.addItem("100% (原始尺寸)", 100)
        self.combo_scale.addItem("80%", 80)
        self.combo_scale.addItem("60%", 60)
        self.combo_scale.addItem("50%", 50)
        self.combo_scale.addItem("33%", 33)
        self.combo_scale.addItem("25%", 25)
        self.combo_scale.setCurrentIndex(0)
        self.combo_scale.setFixedWidth(130)
        row_jpeg_opts.addWidget(self.combo_scale)

        row_jpeg_opts.addSpacing(14)

        lbl_sub = QLabel("色彩抽样:")
        lbl_sub.setFixedWidth(60)
        row_jpeg_opts.addWidget(lbl_sub)

        self.combo_subsampling = QComboBox()
        self.combo_subsampling.addItem("4:4:4 (无损色度)", "4:4:4")
        self.combo_subsampling.addItem("4:2:2 (标准压缩)", "4:2:2")
        self.combo_subsampling.addItem("4:2:0 (高压缩)", "4:2:0")
        self.combo_subsampling.setFixedWidth(135)
        row_jpeg_opts.addWidget(self.combo_subsampling)

        row_jpeg_opts.addSpacing(12)

        self.chk_progressive = QCheckBox("渐进式")
        self.chk_progressive.setChecked(True)
        self.chk_progressive.setStyleSheet("color: #bbb; font-size: 11.5px;")
        row_jpeg_opts.addWidget(self.chk_progressive)
        row_jpeg_opts.addStretch()

        l_jpeg.addLayout(row_jpeg_opts)

        row_q = QHBoxLayout()
        row_q.setSpacing(10)
        lbl_q = QLabel("画质档位:")
        lbl_q.setFixedWidth(65)
        row_q.addWidget(lbl_q)

        self.slider_quality = QSlider(Qt.Orientation.Horizontal)
        self.slider_quality.setRange(1, 10)
        self.slider_quality.setValue(9)
        self.slider_quality.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_quality.setTickInterval(1)
        self.slider_quality.valueChanged.connect(self._on_quality_changed)
        row_q.addWidget(self.slider_quality, 1)

        self.lbl_quality_val = QLabel("9 档 (90%)")
        self.lbl_quality_val.setFixedWidth(75)
        self.lbl_quality_val.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")
        row_q.addWidget(self.lbl_quality_val)

        l_jpeg.addLayout(row_q)
        self.stack_settings.addWidget(page_jpeg)

        # --- Stack Page 1: TIFF Options ---
        page_tiff = QWidget()
        l_tiff = QVBoxLayout(page_tiff)
        l_tiff.setContentsMargins(0, 4, 0, 0)
        l_tiff.setSpacing(8)

        row_t_comp = QHBoxLayout()
        row_t_comp.setSpacing(10)
        lbl_t_comp = QLabel("压缩算法:")
        lbl_t_comp.setFixedWidth(65)
        row_t_comp.addWidget(lbl_t_comp)

        self.combo_tiff_comp = QComboBox()
        self.combo_tiff_comp.addItem("LZW 压缩 (无损推荐)", "lzw")
        self.combo_tiff_comp.addItem("Zip / Deflate 压缩 (小体积)", "deflate")
        self.combo_tiff_comp.addItem("无压缩 (Raw TIFF)", "none")
        self.combo_tiff_comp.setFixedWidth(200)
        row_t_comp.addWidget(self.combo_tiff_comp)
        row_t_comp.addStretch()
        l_tiff.addLayout(row_t_comp)

        lbl_t_hint = QLabel("TIFF 支持最高 32 位浮点线性渲染，适合送印大型相纸展演或二次排版。")
        lbl_t_hint.setStyleSheet("color: #888; font-size: 11px;")
        l_tiff.addWidget(lbl_t_hint)
        self.stack_settings.addWidget(page_tiff)

        # --- Stack Page 2: PNG Options ---
        page_png = QWidget()
        l_png = QVBoxLayout(page_png)
        l_png.setContentsMargins(0, 4, 0, 0)
        l_png.setSpacing(8)

        row_p_comp = QHBoxLayout()
        row_p_comp.setSpacing(10)
        lbl_p_comp = QLabel("压缩级别:")
        lbl_p_comp.setFixedWidth(65)
        row_p_comp.addWidget(lbl_p_comp)

        self.combo_png_level = QComboBox()
        self.combo_png_level.addItem("标准压缩 (Level 6)", 6)
        self.combo_png_level.addItem("极限压缩 (Level 9)", 9)
        self.combo_png_level.addItem("快速存储 (Level 1)", 1)
        self.combo_png_level.setFixedWidth(200)
        row_p_comp.addWidget(self.combo_png_level)
        row_p_comp.addStretch()
        l_png.addLayout(row_p_comp)

        lbl_p_hint = QLabel("PNG 支持 8位 与 16位 无损通道与广泛的网络兼容性。")
        lbl_p_hint.setStyleSheet("color: #888; font-size: 11px;")
        l_png.addWidget(lbl_p_hint)
        self.stack_settings.addWidget(page_png)

        card_layout.addWidget(self.stack_settings)
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)

        return card, card_layout

    def _on_format_changed(self, index):
        fmt = self.combo_format.currentData()
        self.combo_depth.clear()
        if fmt == "jpeg":
            self.stack_settings.setCurrentIndex(0)
            self.combo_depth.addItem("8 位 (8-bit)", 8)
        elif fmt == "tiff":
            self.stack_settings.setCurrentIndex(1)
            self.combo_depth.addItem("8 位 (8-bit)", 8)
            self.combo_depth.addItem("16 位 (16-bit)", 16)
            self.combo_depth.addItem("32 位浮点 (32-bit Float)", 32)
            self.combo_depth.setCurrentIndex(1)
        else:  # png
            self.stack_settings.setCurrentIndex(2)
            self.combo_depth.addItem("8 位 (8-bit)", 8)
            self.combo_depth.addItem("16 位 (16-bit)", 16)

    def _on_quality_changed(self, val):
        pct = int(val * 10)
        self.lbl_quality_val.setText(f"{val} 档 ({pct}%)")

    def _apply_last_settings(self):
        try:
            last = config_manager.get_last_export_settings()
            if not last:
                return
            eng = last.get("engine", "wysiwyg")
            idx_eng = self.combo_engine.findData(eng)
            if idx_eng >= 0:
                self.combo_engine.setCurrentIndex(idx_eng)

            fmt = last.get("format", "jpeg")
            idx = self.combo_format.findData(fmt)
            if idx >= 0:
                self.combo_format.setCurrentIndex(idx)
                self._on_format_changed(idx)

            cs = last.get("colorspace", "sRGB")
            idx_cs = self.combo_colorspace.findData(cs)
            if idx_cs >= 0:
                self.combo_colorspace.setCurrentIndex(idx_cs)

            bd = last.get("bit_depth", 8)
            idx_bd = self.combo_depth.findData(bd)
            if idx_bd >= 0:
                self.combo_depth.setCurrentIndex(idx_bd)

            q = last.get("quality", 9)
            if hasattr(self, "slider_quality"):
                self.slider_quality.setValue(min(10, max(1, int(q))))

            sc = last.get("scale_pct", 100)
            if hasattr(self, "combo_scale"):
                idx_sc = self.combo_scale.findData(sc)
                if idx_sc >= 0:
                    self.combo_scale.setCurrentIndex(idx_sc)

            sub = last.get("subsampling", "4:4:4")
            if hasattr(self, "combo_subsampling"):
                idx_sub = self.combo_subsampling.findData(sub)
                if idx_sub >= 0:
                    self.combo_subsampling.setCurrentIndex(idx_sub)

            if hasattr(self, "chk_progressive"):
                self.chk_progressive.setChecked(bool(last.get("progressive", True)))

            comp = last.get("tiff_compression", "lzw")
            if hasattr(self, "combo_tiff_comp"):
                idx_comp = self.combo_tiff_comp.findData(comp)
                if idx_comp >= 0:
                    self.combo_tiff_comp.setCurrentIndex(idx_comp)

            png_lvl = last.get("png_level", 6)
            if hasattr(self, "combo_png_level"):
                idx_png = self.combo_png_level.findData(png_lvl)
                if idx_png >= 0:
                    self.combo_png_level.setCurrentIndex(idx_png)

            dpi_val = last.get("dpi", 300)
            if hasattr(self, "combo_dpi"):
                idx_dpi = self.combo_dpi.findData(dpi_val)
                if idx_dpi >= 0:
                    self.combo_dpi.setCurrentIndex(idx_dpi)
        except Exception:
            pass

    def get_common_export_config(self) -> dict:
        try:
            dpi_val = int(str(self.combo_dpi.currentText()).split()[0])
        except Exception:
            dpi_val = 300

        return {
            "engine": self.combo_engine.currentData() if hasattr(self, "combo_engine") else "wysiwyg",
            "format": self.combo_format.currentData(),
            "colorspace": self.combo_colorspace.currentData(),
            "bit_depth": self.combo_depth.currentData(),
            "dpi": dpi_val,
            "quality": self.slider_quality.value(),
            "subsampling": self.combo_subsampling.currentData(),
            "progressive": self.chk_progressive.isChecked(),
            "tiff_compression": self.combo_tiff_comp.currentData(),
            "png_level": self.combo_png_level.currentData(),
            "scale_pct": int(self.combo_scale.currentData()) if hasattr(self, "combo_scale") and self.combo_format.currentData() == "jpeg" else 100,
        }

    def _apply_style(self):
        arrow_icon = get_icon_path("arrow_down.png").replace("\\", "/")
        chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")

        self.setStyleSheet(f"""
            QDialog {{
                background: #141416;
            }}
            QComboBox {{
                background: #14151a;
                color: #e2e8f0;
                border: 1px solid #323542;
                border-radius: 4px;
                padding: 4px 26px 4px 8px;
                font-size: 11.5px;
            }}
            QComboBox:hover {{
                border-color: #4b5168;
            }}
            QComboBox:focus {{
                border-color: #f59e0b;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 22px;
                border-left: 1px solid #282a36;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background: #181920;
            }}
            QComboBox::down-arrow {{
                image: url("{arrow_icon}");
                width: 10px;
                height: 7px;
            }}
            QComboBox QAbstractItemView {{
                background: #181920;
                color: #e2e8f0;
                selection-background-color: #f59e0b;
                selection-color: #111;
                border: 1px solid #333644;
                padding: 4px;
            }}
            QLineEdit {{
                background: #121215;
                color: #e2e8f0;
                border: 1px solid #323542;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 12px;
            }}
            QLineEdit:focus {{
                border: 1px solid #f59e0b;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: #2b2e3b;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: #d8d8e0;
                width: 14px;
                height: 14px;
                margin: -5px 0px;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #f59e0b;
            }}
            QCheckBox {{
                color: #c4c4cc;
                spacing: 8px;
                font-size: 11.5px;
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
            QPushButton {{
                background: #25262c;
                color: #bbb;
                border: 1px solid #3c3e4a;
                border-radius: 4px;
                padding: 6px 20px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: #32343e;
                color: #fff;
            }}
            QPushButton:pressed {{
                background: #181920;
                color: #d97706;
                border-color: #d97706;
            }}
        """)


class SingleExportDialog(BaseExportDialog):
    """Export dialog for single photo development with precise file destination."""

    def __init__(self, default_file_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("冲印照片")
        self.setFixedSize(660, 455)

        self._default_file_path = normalize_path(default_file_path) if default_file_path else ""
        self._init_ui()
        self._apply_style()
        self._apply_last_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # 1. Header (Concise typography without emoji)
        header_container = QWidget()
        header_container.setFixedHeight(30)
        h_layout = QHBoxLayout(header_container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(10)

        lbl_title = QLabel("冲印照片")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f59e0b;")
        h_layout.addWidget(lbl_title)

        if self._default_file_path:
            src_fname = os.path.basename(self._default_file_path)
            lbl_sub = QLabel(f"底片: {src_fname}")
            lbl_sub.setStyleSheet("font-size: 12px; color: #8b92a5;")
            h_layout.addWidget(lbl_sub)

        h_layout.addStretch()
        main_layout.addWidget(header_container)

        # 2. Shared Settings Card
        card, card_layout = self._init_settings_card()

        # Row 4: Exact Save Path for Single Photo
        row_path = QVBoxLayout()
        row_path.setSpacing(6)
        lbl_path = QLabel("保存路径:")
        lbl_path.setStyleSheet("color: #bbb; font-size: 12px;")
        row_path.addWidget(lbl_path)

        path_box = QHBoxLayout()
        path_box.setSpacing(8)
        self.edit_path = QLineEdit(self._default_file_path)
        path_box.addWidget(self.edit_path, 1)

        btn_browse = QPushButton("浏览...")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self._browse_save_path)
        path_box.addWidget(btn_browse)

        row_path.addLayout(path_box)
        card_layout.addLayout(row_path)
        main_layout.addWidget(card)

        # Initial format extension setup
        ext = os.path.splitext(self._default_file_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            self.combo_format.setCurrentIndex(0)
        elif ext in (".tif", ".tiff"):
            self.combo_format.setCurrentIndex(1)
        else:
            self.combo_format.setCurrentIndex(2)
        self._on_format_changed(self.combo_format.currentIndex())

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("取消")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_export = QPushButton("开始冲印")
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #111;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 6px 26px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #d97706;
            }
            QPushButton:pressed {
                background: #b45309;
                color: #000;
            }
        """)
        btn_export.clicked.connect(self.accept)
        btn_row.addWidget(btn_export)

        main_layout.addLayout(btn_row)

    def _on_format_changed(self, index):
        super()._on_format_changed(index)
        fmt = self.combo_format.currentData()
        cur_path = self.edit_path.text()
        if cur_path:
            base, _ = os.path.splitext(cur_path)
            new_ext = ".tif" if fmt == "tiff" else (".png" if fmt == "png" else ".jpg")
            self.edit_path.setText(base + new_ext)

    def _browse_save_path(self):
        fmt = self.combo_format.currentData()
        if fmt == "tiff":
            filt = "专业 TIFF (*.tif *.tiff)"
        elif fmt == "jpeg":
            filt = "高质量 JPEG (*.jpg *.jpeg)"
        else:
            filt = "便携 PNG (*.png)"

        start_path = self.edit_path.text().strip()
        path, _ = QFileDialog.getSaveFileName(self, "选择保存路径", start_path, filt)
        if path:
            self.edit_path.setText(normalize_path(path))

    def get_export_config(self) -> dict:
        cfg = self.get_common_export_config()
        raw_path = normalize_path(self.edit_path.text().strip())
        if raw_path and (os.path.isdir(raw_path) or raw_path.endswith(("/", "\\"))):
            cur_name = os.path.basename(self._default_file_path) or "developed.jpg"
            raw_path = os.path.join(raw_path, cur_name)
        cfg["path"] = normalize_path(raw_path)
        cfg["is_batch"] = False
        return cfg

    def accept(self):
        try:
            cfg = self.get_export_config()
            config_manager.save_last_export_settings(cfg)
            if cfg.get("path"):
                config_manager.set_last_export_dir(os.path.dirname(cfg["path"]))
        except Exception:
            pass
        super().accept()


class BatchExportDialog(BaseExportDialog):
    """Export dialog for multi-photo batch queue delivery with strictly directory destination."""

    def __init__(self, default_dir: str, batch_count: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量冲印")
        self.setFixedSize(660, 495)
        self.batch_count = batch_count

        # Ensure directory only, never a file
        clean_dir = normalize_path(default_dir) if default_dir else ""
        if clean_dir and not os.path.isdir(clean_dir):
            clean_dir = os.path.dirname(clean_dir)
        if not clean_dir or not os.path.exists(clean_dir):
            clean_dir = config_manager.get_last_export_dir() or os.path.expanduser("~")
        self._default_dir = clean_dir

        self._init_ui()
        self._apply_style()
        self._apply_last_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # 1. Header (Concise typography, badge without emoji)
        header_container = QWidget()
        header_container.setFixedHeight(30)
        h_layout = QHBoxLayout(header_container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(10)

        lbl_title = QLabel("批量冲印")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f59e0b;")
        h_layout.addWidget(lbl_title)

        lbl_badge = QLabel(f"批量：选中了 {self.batch_count} 张")
        lbl_badge.setStyleSheet("""
            QLabel {
                background: #25262c;
                border: 1px solid #3c3e4a;
                border-radius: 4px;
                color: #e2e8f0;
                padding: 2px 8px;
                font-size: 11.5px;
            }
        """)
        h_layout.addWidget(lbl_badge)
        h_layout.addStretch()
        main_layout.addWidget(header_container)

        # 2. Shared Settings Card
        card, card_layout = self._init_settings_card()

        # Row 4: File Naming Rules (Concise prefix and suffix)
        row_naming = QHBoxLayout()
        row_naming.setSpacing(10)

        lbl_name_tag = QLabel("文件命名:")
        lbl_name_tag.setFixedWidth(65)
        row_naming.addWidget(lbl_name_tag)

        lbl_pre = QLabel("前缀:")
        lbl_pre.setStyleSheet("color: #94a3b8; font-size: 11px;")
        row_naming.addWidget(lbl_pre)
        self.edit_prefix = QLineEdit()
        self.edit_prefix.setPlaceholderText("无")
        self.edit_prefix.setFixedWidth(110)
        row_naming.addWidget(self.edit_prefix)

        row_naming.addSpacing(10)

        lbl_suf = QLabel("后缀:")
        lbl_suf.setStyleSheet("color: #94a3b8; font-size: 11px;")
        row_naming.addWidget(lbl_suf)
        self.edit_suffix = QLineEdit()
        self.edit_suffix.setPlaceholderText("无")
        self.edit_suffix.setFixedWidth(110)
        row_naming.addWidget(self.edit_suffix)

        row_naming.addSpacing(10)
        lbl_hint = QLabel("留空则保持原文件名输出")
        lbl_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        row_naming.addWidget(lbl_hint)
        row_naming.addStretch()

        card_layout.addLayout(row_naming)

        # Row 5: Strictly Destination Directory
        row_path = QVBoxLayout()
        row_path.setSpacing(6)
        lbl_path = QLabel("目标输出文件夹:")
        lbl_path.setStyleSheet("color: #bbb; font-size: 12px;")
        row_path.addWidget(lbl_path)

        path_box = QHBoxLayout()
        path_box.setSpacing(8)
        self.edit_path = QLineEdit(self._default_dir)
        path_box.addWidget(self.edit_path, 1)

        btn_browse = QPushButton("浏览...")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self._browse_save_dir)
        path_box.addWidget(btn_browse)

        row_path.addLayout(path_box)
        card_layout.addLayout(row_path)
        main_layout.addWidget(card)

        # Initial format setup
        self._on_format_changed(self.combo_format.currentIndex())

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("取消")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_export = QPushButton(f"加入冲印队列 ({self.batch_count})")
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #111;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 6px 24px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #d97706;
            }
            QPushButton:pressed {
                background: #b45309;
                color: #000;
            }
        """)
        btn_export.clicked.connect(self.accept)
        btn_row.addWidget(btn_export)

        main_layout.addLayout(btn_row)

    def _browse_save_dir(self):
        start_dir = self.edit_path.text().strip()
        if not os.path.isdir(start_dir):
            start_dir = config_manager.get_last_export_dir() or os.path.expanduser("~")
        dir_chosen = QFileDialog.getExistingDirectory(self, "选择目标输出文件夹", start_dir)
        if dir_chosen:
            self.edit_path.setText(normalize_path(dir_chosen))

    def get_export_config(self) -> dict:
        cfg = self.get_common_export_config()
        out_dir = normalize_path(self.edit_path.text().strip())
        if out_dir and not os.path.isdir(out_dir):
            # Strip accidental file suffix if user typed filename
            out_dir = os.path.dirname(out_dir) if os.path.splitext(out_dir)[1] else out_dir
        cfg["path"] = normalize_path(out_dir)
        cfg["prefix"] = self.edit_prefix.text().strip()
        cfg["suffix"] = self.edit_suffix.text().strip()
        cfg["is_batch"] = True
        cfg["batch_count"] = self.batch_count
        return cfg

    def accept(self):
        try:
            cfg = self.get_export_config()
            config_manager.save_last_export_settings(cfg)
            if cfg.get("path"):
                config_manager.set_last_export_dir(cfg["path"])
        except Exception:
            pass
        super().accept()


class ExportImageDialog:
    """Compatibility factory dispatching to SingleExportDialog or BatchExportDialog."""

    def __new__(cls, default_path: str, is_batch: bool = False, batch_count: int = 0, parent=None):
        if is_batch:
            clean_dir = default_path
            if clean_dir and not os.path.isdir(clean_dir):
                clean_dir = os.path.dirname(clean_dir)
            return BatchExportDialog(clean_dir, batch_count=batch_count, parent=parent)
        else:
            return SingleExportDialog(default_path, parent=parent)


class ExportProgressDialog(QDialog):
    """Modern darkroom responsive export progress dialog with progress bar and cancel button."""
    cancelRequested = Signal()

    def __init__(self, title="冲印照片", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(440, 155)
        self.setModal(True)

        icon_path = os.path.join(get_resource_dir(), "icons", "dlg_export.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setStyleSheet("""
            QDialog {
                background: #14151a;
                border: 1px solid #2d303e;
                border-radius: 8px;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 12px;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            }
            QProgressBar {
                background: #101115;
                border: 1px solid #2a2d3b;
                border-radius: 4px;
                height: 16px;
                text-align: center;
                color: #fff;
                font-size: 10px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: #f59e0b;
                border-radius: 3px;
            }
            QPushButton {
                background: #20222a;
                color: #cbd5e1;
                border: 1px solid #353846;
                border-radius: 4px;
                font-size: 11.5px;
                padding: 4px 16px;
            }
            QPushButton:hover {
                background: #2d303e;
                color: #ef4444;
                border-color: #ef4444;
            }
            QPushButton:pressed {
                background: #1b171a;
                color: #dc2626;
                border-color: #dc2626;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        self.lbl_title = QLabel("正在冲印...")
        self.lbl_title.setStyleSheet("font-weight: bold; color: #f59e0b; font-size: 13px;")
        layout.addWidget(self.lbl_title)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        b_row = QHBoxLayout()
        self.lbl_detail = QLabel("准备冲印...")
        self.lbl_detail.setStyleSheet("color: #94a3b8; font-size: 11px;")
        b_row.addWidget(self.lbl_detail, 1)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFixedSize(68, 26)
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        b_row.addWidget(self.btn_cancel)
        layout.addLayout(b_row)

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _on_cancel_clicked(self):
        self.lbl_detail.setText("正在取消冲印...")
        self.btn_cancel.setEnabled(False)
        self.cancelRequested.emit()
        self.reject()

    def set_progress(self, pct, detail=""):
        self.progress_bar.setValue(pct)
        if detail:
            clean_detail = detail.replace("正在载入底片并准备物理色彩管线...", "准备底片与色彩管线...")
            clean_detail = clean_detail.replace("正在准备全分辨率冲印管线...", "准备管线...")
            clean_detail = clean_detail.replace("正在进行高精度冲印渲染与显影", "高精度冲印中")
            self.lbl_detail.setText(clean_detail)


class SingleExportWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(dict)

    def __init__(self, engine, params_dict, out_path, format_type, quality, source_path):
        super().__init__()
        self.engine = engine
        self.params_dict = params_dict
        self.out_path = out_path
        self.format_type = format_type
        self.quality = quality
        self.source_path = source_path
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            def on_prog(pct, msg):
                if not self._is_cancelled:
                    self.progress.emit(pct, msg)

            res = self.engine.render_full_res(
                self.params_dict,
                self.out_path,
                format_type=self.format_type,
                quality=self.quality,
                source_path=self.source_path,
                progress_cb=on_prog,
                cancel_cb=lambda: self._is_cancelled
            )
            if self._is_cancelled or (res and res.get("cancelled")):
                if os.path.exists(self.out_path):
                    try:
                        os.remove(self.out_path)
                    except Exception:
                        pass
                self.finished.emit({"success": False, "cancelled": True, "error": "用户已取消冲印"})
            else:
                self.finished.emit(res)
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class BatchExportWorker(QThread):
    itemProgress = Signal(int, int, str)
    finished = Signal(int, int, str)

    def __init__(self, engine, tasks, out_dir):
        super().__init__()
        self.engine = engine
        self.tasks = tasks
        self.out_dir = out_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        success_count = 0
        total = len(self.tasks)
        for i, task in enumerate(self.tasks):
            if self._is_cancelled:
                break
            file_p = task["source_path"]
            fname = os.path.basename(file_p)
            self.itemProgress.emit(i, total, f"冲印 ({i+1}/{total}): {fname}")
            try:
                fmt = task.get("format", "jpeg")
                r = self.engine.render_full_res(
                    task["export_params"],
                    task["out_target"],
                    format_type=fmt,
                    quality=task.get("quality", 9),
                    source_path=file_p,
                    cancel_cb=lambda: self._is_cancelled
                )
                if r.get("success"):
                    success_count += 1
            except Exception:
                pass
        self.finished.emit(success_count, total, self.out_dir)
