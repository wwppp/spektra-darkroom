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
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QSlider, QLineEdit, QPushButton, QFileDialog, QFrame, QWidget,
    QStackedWidget, QCheckBox, QProgressBar
)

from core.path_utils import get_icon_path, get_resource_dir, normalize_path
from ui.window_utils import setup_solid_combobox, apply_dark_titlebar, get_standard_checkbox_style
from ui.theme_manager import theme_manager
from core import config_manager


class BaseExportDialog(QDialog):
    """Base dialog encapsulating shared darkroom rendering and color management settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        from ui.window_utils import setup_solid_combobox, get_icon
        self.setWindowIcon(get_icon("dlg_export.png", is_dark=theme_manager.is_dark()))

    def showEvent(self, event):
        super().showEvent(event)
        is_dark = theme_manager.is_dark()
        apply_dark_titlebar(self, is_dark=is_dark)
        from ui.window_utils import setup_solid_combobox, get_icon
        self.setWindowIcon(get_icon("dlg_export.png", is_dark=is_dark))

    def _init_settings_card(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("settingsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        # Top 2 Rows: Format, Color Space, Resolution, Bit Depth in a unified QGridLayout
        grid_top = QGridLayout()
        grid_top.setContentsMargins(0, 0, 0, 0)
        grid_top.setHorizontalSpacing(12)
        grid_top.setVerticalSpacing(10)
        grid_top.setColumnStretch(0, 0)
        grid_top.setColumnStretch(1, 1)
        grid_top.setColumnStretch(2, 0)
        grid_top.setColumnStretch(3, 1)

        # Row 0: Format & Color Space
        lbl_fmt = QLabel("文件格式:")
        lbl_fmt.setFixedWidth(72)
        grid_top.addWidget(lbl_fmt, 0, 0)

        self.combo_format = QComboBox()
        setup_solid_combobox(self.combo_format)
        self.combo_format.addItem("JPEG (*.jpg)", "jpeg")
        self.combo_format.addItem("JPEG XL (*.jxl)", "jxl")
        self.combo_format.addItem("TIFF (*.tif)", "tiff")
        self.combo_format.addItem("PNG (*.png)", "png")
        grid_top.addWidget(self.combo_format, 0, 1)

        lbl_cs = QLabel("色彩空间:")
        lbl_cs.setFixedWidth(65)
        grid_top.addWidget(lbl_cs, 0, 2)

        self.combo_colorspace = QComboBox()
        setup_solid_combobox(self.combo_colorspace)
        self.combo_colorspace.addItem("sRGB", "sRGB")
        self.combo_colorspace.addItem("Display P3", "Display P3")
        self.combo_colorspace.addItem("Adobe RGB (1998)", "Adobe RGB (1998)")
        self.combo_colorspace.addItem("Rec. 2020", "Rec. 2020")
        self.combo_colorspace.addItem("ProPhoto RGB", "ProPhoto RGB")
        grid_top.addWidget(self.combo_colorspace, 0, 3)

        # Row 1: Resolution (DPI) & Bit Depth
        lbl_dpi = QLabel("输出分辨率:")
        lbl_dpi.setFixedWidth(72)
        grid_top.addWidget(lbl_dpi, 1, 0)

        self.combo_dpi = QComboBox()
        setup_solid_combobox(self.combo_dpi)
        self.combo_dpi.addItem("72 DPI", 72)
        self.combo_dpi.addItem("300 DPI", 300)
        self.combo_dpi.addItem("600 DPI", 600)
        self.combo_dpi.setCurrentIndex(1)
        grid_top.addWidget(self.combo_dpi, 1, 1)

        lbl_bit = QLabel("输出位深:")
        lbl_bit.setFixedWidth(65)
        grid_top.addWidget(lbl_bit, 1, 2)

        self.combo_depth = QComboBox()
        setup_solid_combobox(self.combo_depth)
        grid_top.addWidget(self.combo_depth, 1, 3)

        # Row 2: GPS Privacy Checkbox (Strictly Protects Photographer Location)
        self.chk_strip_gps = QCheckBox("去除 GPS 位置隐私 (保护个人安全)")
        self.chk_strip_gps.setChecked(True)
        self.chk_strip_gps.setToolTip("开启后将从导出文件的 EXIF 元数据中完全抹除 GPS 经纬度与海拔坐标，保留相机、镜头、光圈、快门与拍摄时间等专业参数。")
        grid_top.addWidget(self.chk_strip_gps, 2, 0, 1, 4)

        card_layout.addLayout(grid_top)

        # Row 3: Format-Specific Settings Stack
        self.stack_settings = QStackedWidget()
        self.stack_settings.setFixedHeight(64)

        # --- Stack Page 0: JPEG Options ---
        page_jpeg = QWidget()
        grid_jpeg = QGridLayout(page_jpeg)
        grid_jpeg.setContentsMargins(0, 0, 0, 0)
        grid_jpeg.setHorizontalSpacing(12)
        grid_jpeg.setVerticalSpacing(10)
        grid_jpeg.setColumnStretch(0, 0)
        grid_jpeg.setColumnStretch(1, 1)
        grid_jpeg.setColumnStretch(2, 0)
        grid_jpeg.setColumnStretch(3, 1)

        # Row 0: Scale & Subsampling
        lbl_scale = QLabel("输出尺寸:")
        lbl_scale.setFixedWidth(72)
        grid_jpeg.addWidget(lbl_scale, 0, 0)

        self.combo_scale = QComboBox()
        setup_solid_combobox(self.combo_scale)
        self.combo_scale.addItem("100% (原始尺寸)", 100)
        self.combo_scale.addItem("80%", 80)
        self.combo_scale.addItem("60%", 60)
        self.combo_scale.addItem("50%", 50)
        self.combo_scale.addItem("33%", 33)
        self.combo_scale.addItem("25%", 25)
        self.combo_scale.setCurrentIndex(0)
        grid_jpeg.addWidget(self.combo_scale, 0, 1)

        lbl_sub = QLabel("色彩抽样:")
        lbl_sub.setFixedWidth(65)
        grid_jpeg.addWidget(lbl_sub, 0, 2)

        self.combo_subsampling = QComboBox()
        setup_solid_combobox(self.combo_subsampling)
        self.combo_subsampling.addItem("4:4:4 (无损色度)", "4:4:4")
        self.combo_subsampling.addItem("4:2:2 (标准压缩)", "4:2:2")
        self.combo_subsampling.addItem("4:2:0 (高压缩)", "4:2:0")
        grid_jpeg.addWidget(self.combo_subsampling, 0, 3)

        # Row 1: Quality Slider & Progressive Checkbox
        lbl_q = QLabel("画质档位:")
        lbl_q.setFixedWidth(72)
        grid_jpeg.addWidget(lbl_q, 1, 0)

        box_q = QHBoxLayout()
        box_q.setContentsMargins(0, 0, 0, 0)
        box_q.setSpacing(8)

        self.slider_quality = QSlider(Qt.Orientation.Horizontal)
        self.slider_quality.setRange(1, 10)
        self.slider_quality.setValue(9)
        self.slider_quality.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_quality.setTickInterval(1)
        self.slider_quality.valueChanged.connect(self._on_quality_changed)
        box_q.addWidget(self.slider_quality, 1)

        self.lbl_quality_val = QLabel("9 档 (90%)")
        self.lbl_quality_val.setFixedWidth(68)
        self.lbl_quality_val.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")
        box_q.addWidget(self.lbl_quality_val)

        grid_jpeg.addLayout(box_q, 1, 1)

        self.chk_progressive = QCheckBox("渐进式")
        self.chk_progressive.setChecked(True)
        grid_jpeg.addWidget(self.chk_progressive, 1, 3, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.stack_settings.addWidget(page_jpeg)

        # --- Stack Page 1: TIFF Options ---
        page_tiff = QWidget()
        grid_tiff = QGridLayout(page_tiff)
        grid_tiff.setContentsMargins(0, 0, 0, 0)
        grid_tiff.setHorizontalSpacing(12)
        grid_tiff.setVerticalSpacing(10)
        grid_tiff.setColumnStretch(0, 0)
        grid_tiff.setColumnStretch(1, 1)
        grid_tiff.setColumnStretch(2, 0)
        grid_tiff.setColumnStretch(3, 1)

        lbl_t_comp = QLabel("压缩算法:")
        lbl_t_comp.setFixedWidth(72)
        grid_tiff.addWidget(lbl_t_comp, 0, 0)

        self.combo_tiff_comp = QComboBox()
        setup_solid_combobox(self.combo_tiff_comp)
        self.combo_tiff_comp.addItem("LZW 压缩 (无损推荐)", "lzw")
        self.combo_tiff_comp.addItem("Zip / Deflate 压缩 (小体积)", "deflate")
        self.combo_tiff_comp.addItem("无压缩 (Raw TIFF)", "none")
        grid_tiff.addWidget(self.combo_tiff_comp, 0, 1)

        lbl_t_hint = QLabel("TIFF 支持最高 32 位浮点线性渲染，适合送印大型相纸展演或二次排版。")
        lbl_t_hint.setStyleSheet("color: #888; font-size: 11px;")
        grid_tiff.addWidget(lbl_t_hint, 1, 0, 1, 4)

        self.stack_settings.addWidget(page_tiff)

        # --- Stack Page 2: PNG Options ---
        page_png = QWidget()
        grid_png = QGridLayout(page_png)
        grid_png.setContentsMargins(0, 0, 0, 0)
        grid_png.setHorizontalSpacing(12)
        grid_png.setVerticalSpacing(10)
        grid_png.setColumnStretch(0, 0)
        grid_png.setColumnStretch(1, 1)
        grid_png.setColumnStretch(2, 0)
        grid_png.setColumnStretch(3, 1)

        lbl_p_comp = QLabel("压缩级别:")
        lbl_p_comp.setFixedWidth(72)
        grid_png.addWidget(lbl_p_comp, 0, 0)

        self.combo_png_level = QComboBox()
        setup_solid_combobox(self.combo_png_level)
        self.combo_png_level.addItem("标准压缩 (Level 6)", 6)
        self.combo_png_level.addItem("极限压缩 (Level 9)", 9)
        self.combo_png_level.addItem("快速存储 (Level 1)", 1)
        grid_png.addWidget(self.combo_png_level, 0, 1)

        lbl_p_hint = QLabel("PNG 支持 8位 与 16位 无损通道与广泛的网络兼容性。")
        lbl_p_hint.setStyleSheet("color: #888; font-size: 11px;")
        grid_png.addWidget(lbl_p_hint, 1, 0, 1, 4)

        self.stack_settings.addWidget(page_png)

        # --- Stack Page 3: JPEG XL Options ---
        page_jxl = QWidget()
        grid_jxl = QGridLayout(page_jxl)
        grid_jxl.setContentsMargins(0, 0, 0, 0)
        grid_jxl.setHorizontalSpacing(12)
        grid_jxl.setVerticalSpacing(10)
        grid_jxl.setColumnStretch(0, 0)
        grid_jxl.setColumnStretch(1, 1)
        grid_jxl.setColumnStretch(2, 0)
        grid_jxl.setColumnStretch(3, 1)

        lbl_jxl_mode = QLabel("压缩模式:")
        lbl_jxl_mode.setFixedWidth(72)
        grid_jxl.addWidget(lbl_jxl_mode, 0, 0)

        self.combo_jxl_mode = QComboBox()
        setup_solid_combobox(self.combo_jxl_mode)
        self.combo_jxl_mode.addItem("视觉无损 (16位推荐 / 距离1.0)", "visual_10")
        self.combo_jxl_mode.addItem("超高保真 (距离0.5)", "visual_05")
        self.combo_jxl_mode.addItem("高质量平衡 (距离1.5)", "visual_15")
        self.combo_jxl_mode.addItem("数学完全无损 (Lossless 1:1)", "lossless")
        grid_jxl.addWidget(self.combo_jxl_mode, 0, 1)

        lbl_jxl_effort = QLabel("编码精度:")
        lbl_jxl_effort.setFixedWidth(65)
        grid_jxl.addWidget(lbl_jxl_effort, 0, 2)

        self.combo_jxl_effort = QComboBox()
        setup_solid_combobox(self.combo_jxl_effort)
        self.combo_jxl_effort.addItem("标准精度 (Effort 7)", 7)
        self.combo_jxl_effort.addItem("极限压缩 (Effort 9)", 9)
        self.combo_jxl_effort.addItem("极速导出 (Effort 5)", 5)
        grid_jxl.addWidget(self.combo_jxl_effort, 0, 3)

        lbl_jxl_hint = QLabel("JPEG XL (.jxl) 是新一代高保真格式，兼具超高压缩率与 16/32 位浮点广色域存储。")
        lbl_jxl_hint.setStyleSheet("color: #888; font-size: 11px;")
        grid_jxl.addWidget(lbl_jxl_hint, 1, 0, 1, 4)

        self.stack_settings.addWidget(page_jxl)

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
        elif fmt == "jxl":
            self.stack_settings.setCurrentIndex(3)
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

            if hasattr(self, "chk_strip_gps"):
                self.chk_strip_gps.setChecked(bool(last.get("strip_gps", True)))

            if hasattr(self, "combo_jxl_mode"):
                idx_jm = self.combo_jxl_mode.findData(last.get("jxl_mode", "visual_10"))
                if idx_jm >= 0:
                    self.combo_jxl_mode.setCurrentIndex(idx_jm)

            if hasattr(self, "combo_jxl_effort"):
                idx_je = self.combo_jxl_effort.findData(last.get("jxl_effort", 7))
                if idx_je >= 0:
                    self.combo_jxl_effort.setCurrentIndex(idx_je)
        except Exception:
            pass

    def get_common_export_config(self) -> dict:
        try:
            dpi_val = int(str(self.combo_dpi.currentText()).split()[0])
        except Exception:
            dpi_val = 300

        mode_val = self.combo_jxl_mode.currentData() if hasattr(self, "combo_jxl_mode") else "visual_10"
        is_lossless = (mode_val == "lossless")
        dist_map = {"visual_05": 0.5, "visual_10": 1.0, "visual_15": 1.5, "lossless": 0.0}
        dist_val = dist_map.get(mode_val, 1.0)
        effort_val = int(self.combo_jxl_effort.currentData()) if hasattr(self, "combo_jxl_effort") else 7

        return {
            "engine": "wysiwyg",
            "format": self.combo_format.currentData(),
            "colorspace": self.combo_colorspace.currentData(),
            "bit_depth": self.combo_depth.currentData(),
            "dpi": dpi_val,
            "strip_gps": self.chk_strip_gps.isChecked() if hasattr(self, "chk_strip_gps") else True,
            "quality": self.slider_quality.value(),
            "subsampling": self.combo_subsampling.currentData(),
            "progressive": self.chk_progressive.isChecked(),
            "tiff_compression": self.combo_tiff_comp.currentData(),
            "png_level": self.combo_png_level.currentData(),
            "jxl_mode": mode_val,
            "jxl_lossless": is_lossless,
            "jxl_distance": dist_val,
            "jxl_effort": effort_val,
            "scale_pct": int(self.combo_scale.currentData()) if hasattr(self, "combo_scale") and self.combo_format.currentData() in ("jpeg", "jxl") else 100,
        }

    def _apply_style(self):
        is_dark = theme_manager.is_dark()
        arrow_name = "arrow_down.png" if is_dark else "arrow_down_dark.png"
        arrow_icon = get_icon_path(arrow_name).replace("\\", "/")
        chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")

        if is_dark:
            self.setStyleSheet(f"""
                QDialog {{
                    background: #141416;
                }}
                QFrame#settingsCard {{
                    background: #1a1a1e;
                    border: 1px solid #2b2d38;
                    border-radius: 8px;
                    padding: 14px;
                }}
                QLabel {{
                    border: none;
                    background: transparent;
                    color: #c4c4cc;
                    font-size: 12px;
                }}
                QComboBox {{
                    background: #14151a;
                    color: #e2e8f0;
                    border: 1px solid #323542;
                    border-radius: 4px;
                    padding: 4px 26px 4px 8px;
                    font-size: 11px;
                }}
                QComboBox:hover {{
                    border-color: #f59e0b;
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
                    border: 1px solid #333644;
                    border-radius: 4px;
                    padding: 2px 0px;
                    outline: 0px;
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
{get_standard_checkbox_style(is_dark=True)}
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
        else:
            self.setStyleSheet(f"""
                QDialog {{
                    background: #ebedf2;
                }}
                QFrame#settingsCard {{
                    background: #ffffff;
                    border: 1px solid #d5d9e4;
                    border-radius: 8px;
                    padding: 14px;
                }}
                QLabel {{
                    border: none;
                    background: transparent;
                    color: #334155;
                    font-size: 12px;
                }}
                QComboBox {{
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 4px 26px 4px 8px;
                    font-size: 11px;
                }}
                QComboBox:hover {{
                    border-color: #f59e0b;
                }}
                QComboBox:focus {{
                    border-color: #f59e0b;
                }}
                QComboBox::drop-down {{
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 22px;
                    border-left: 1px solid #e2e8f0;
                    border-top-right-radius: 3px;
                    border-bottom-right-radius: 3px;
                    background: #f8fafc;
                }}
                QComboBox::down-arrow {{
                    image: url("{arrow_icon}");
                    width: 10px;
                    height: 7px;
                }}
                QComboBox QAbstractItemView {{
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 2px 0px;
                    outline: 0px;
                }}
                QLineEdit {{
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 5px 8px;
                    font-size: 12px;
                }}
                QLineEdit:focus {{
                    border: 1px solid #f59e0b;
                }}
                QSlider::groove:horizontal {{
                    height: 4px;
                    background: #cbd5e1;
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: #64748b;
                    width: 14px;
                    height: 14px;
                    margin: -5px 0px;
                    border-radius: 7px;
                }}
                QSlider::handle:horizontal:hover {{
                    background: #f59e0b;
                }}
{get_standard_checkbox_style(is_dark=False)}
                QPushButton {{
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 6px 20px;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background: #f8fafc;
                    border-color: #f59e0b;
                    color: #d97706;
                }}
                QPushButton:pressed {{
                    background: #e2e8f0;
                    color: #b45309;
                }}
            """)


class SingleExportDialog(BaseExportDialog):
    """Export dialog for single photo development with precise file destination."""

    def __init__(self, default_file_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("单片扫描")
        self.setFixedSize(660, 390)

        self._default_file_path = normalize_path(default_file_path) if default_file_path else ""
        self._init_ui()
        self._apply_style()
        self._apply_last_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # 1. Shared Settings Card
        card, card_layout = self._init_settings_card()

        # Row 4: Exact Save Path for Single Photo
        row_path = QVBoxLayout()
        row_path.setContentsMargins(0, 2, 0, 0)
        row_path.setSpacing(6)
        lbl_path = QLabel("保存路径:")
        lbl_path.setStyleSheet("font-size: 12px; background: transparent;")
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
            init_fmt = "jpeg"
        elif ext == ".jxl":
            init_fmt = "jxl"
        elif ext in (".tif", ".tiff"):
            init_fmt = "tiff"
        else:
            init_fmt = "png"
        idx_f = self.combo_format.findData(init_fmt)
        if idx_f >= 0:
            self.combo_format.setCurrentIndex(idx_f)
        self._on_format_changed(self.combo_format.currentIndex())

        main_layout.addStretch(1)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("取消")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_export = QPushButton("开始扫描")
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
            new_ext = ".tif" if fmt == "tiff" else (".jxl" if fmt == "jxl" else (".png" if fmt == "png" else ".jpg"))
            self.edit_path.setText(base + new_ext)

    def _browse_save_path(self):
        fmt = self.combo_format.currentData()
        if fmt == "tiff":
            filt = "专业 TIFF (*.tif *.tiff)"
        elif fmt == "jxl":
            filt = "新一代 JPEG XL (*.jxl)"
        elif fmt == "png":
            filt = "便携 PNG (*.png)"
        else:
            filt = "高质量 JPEG (*.jpg *.jpeg)"

        start_path = self.edit_path.text().strip()
        path, _ = QFileDialog.getSaveFileName(self, "选择保存路径", start_path, filt)
        if path:
            base, ext = os.path.splitext(path)
            valid_exts = {'jpeg': ('.jpg', '.jpeg'), 'jxl': ('.jxl',), 'tiff': ('.tif', '.tiff'), 'png': ('.png',)}.get(fmt, ())
            canonical_ext = {'jpeg': '.jpg', 'jxl': '.jxl', 'tiff': '.tif', 'png': '.png'}.get(fmt, '.jpg')
            if ext.lower() not in valid_exts:
                path = base + canonical_ext
            self.edit_path.setText(normalize_path(path))

    def get_export_config(self) -> dict:
        cfg = self.get_common_export_config()
        raw_path = normalize_path(self.edit_path.text().strip())
        fmt = str(cfg.get("format", "jpeg")).lower()
        if raw_path and (os.path.isdir(raw_path) or raw_path.endswith(("/", "\\"))):
            cur_name = os.path.basename(self._default_file_path) or "developed.jpg"
            raw_path = os.path.join(raw_path, cur_name)

        if raw_path:
            base, ext = os.path.splitext(raw_path)
            valid_exts = {
                "jpeg": (".jpg", ".jpeg"),
                "jxl": (".jxl",),
                "tiff": (".tif", ".tiff"),
                "png": (".png",),
            }.get(fmt, ())
            canonical_ext = {
                "jpeg": ".jpg",
                "jxl": ".jxl",
                "tiff": ".tif",
                "png": ".png",
            }.get(fmt, ".jpg")
            if ext.lower() not in valid_exts:
                raw_path = base + canonical_ext
                self.edit_path.setText(normalize_path(raw_path))

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
        self.batch_count = batch_count
        self.setWindowTitle(f"批量扫描：{self.batch_count}张")
        self.setFixedSize(660, 450)

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

        # 1. Shared Settings Card
        card, card_layout = self._init_settings_card()

        # Row 4: File Naming Rules (Concise prefix and suffix)
        row_naming = QHBoxLayout()
        row_naming.setSpacing(10)

        lbl_name_tag = QLabel("文件命名:")
        lbl_name_tag.setFixedWidth(72)
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
        row_path.setContentsMargins(0, 2, 0, 0)
        row_path.setSpacing(6)
        lbl_path = QLabel("目标输出文件夹:")
        lbl_path.setStyleSheet("font-size: 12px; background: transparent;")
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

        main_layout.addStretch(1)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("取消")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_export = QPushButton(f"加入扫描队列 ({self.batch_count})")
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

    def __init__(self, title="扫描照片", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(440, 155)
        self.setModal(True)

        is_dark = theme_manager.is_dark()
        from ui.window_utils import setup_solid_combobox, get_icon
        self.setWindowIcon(get_icon("dlg_export.png", is_dark=is_dark))
        if is_dark:
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
                    font-size: 11px;
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
        else:
            self.setStyleSheet("""
                QDialog {
                    background: #ebedf2;
                    border: 1px solid #d5d9e4;
                    border-radius: 8px;
                }
                QLabel {
                    color: #1e293b;
                    font-size: 12px;
                    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                }
                QProgressBar {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    height: 16px;
                    text-align: center;
                    color: #111;
                    font-size: 10px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QPushButton {
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    font-size: 11px;
                    padding: 4px 16px;
                }
                QPushButton:hover {
                    background: #fef2f2;
                    color: #ef4444;
                    border-color: #ef4444;
                }
                QPushButton:pressed {
                    background: #fee2e2;
                    color: #dc2626;
                    border-color: #dc2626;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        self.lbl_title = QLabel("正在扫描...")
        self.lbl_title.setStyleSheet("font-weight: bold; color: #f59e0b; font-size: 13px;")
        layout.addWidget(self.lbl_title)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        b_row = QHBoxLayout()
        self.lbl_detail = QLabel("准备扫描...")
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
        self.lbl_detail.setText("正在取消扫描...")
        self.btn_cancel.setEnabled(False)
        self.cancelRequested.emit()
        self.reject()

    def set_progress(self, pct, detail=""):
        self.progress_bar.setValue(pct)
        if detail:
            clean_detail = detail.replace("正在载入底片并准备物理色彩管线...", "准备底片与色彩管线...")
            clean_detail = clean_detail.replace("正在准备全分辨率冲印管线...", "准备管线...")
            clean_detail = clean_detail.replace("正在进行高精度冲印渲染与显影", "高精度扫描中")
            clean_detail = clean_detail.replace("正在准备全分辨率扫描管线...", "准备管线...")
            clean_detail = clean_detail.replace("正在进行高精度扫描渲染", "高精度扫描中")
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
                self.finished.emit({"success": False, "cancelled": True, "error": "用户已取消扫描"})
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
            self.itemProgress.emit(i, total, f"扫描 ({i+1}/{total}): {fname}")
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
