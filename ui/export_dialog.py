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
    QStackedWidget, QCheckBox, QProgressBar, QStyle, QStyleOptionComboBox
)

from core.path_utils import get_icon_path, get_resource_dir, normalize_path
from ui.window_utils import setup_solid_combobox, apply_dark_titlebar, get_standard_checkbox_style
from ui.theme_manager import theme_manager
from core import config_manager


# Output size presets. A preset pins the *long edge*; the short edge follows the source
# aspect ratio. Percentages were dropped because their physical meaning differs from file to
# file (50% of a 45 MP frame is 6000 px, of a 24 MP frame 3000 px) while photographers size an
# export by use: print at a paper size, or fit a pixel budget for the web.
#   key, label, long edge in px (0 = keep the source resolution), paper mm (w, h) or None
EXPORT_SIZE_PRESETS = (
    ("original", "原尺寸", 0, None),
    ("print_a4", "A4", 3508, (210, 297)),
    ("print_a3", "A3", 4961, (297, 420)),
    ("print_a2", "A2", 7016, (420, 594)),
    ("web_4k", "4K", 3840, None),
    ("web_2k", "2K", 2048, None),
    ("share_1080", "1080p", 1080, None),
    ("custom", "自定义…", -1, None),
)

EXPORT_SIZE_PREF_KEY = "export_size_long_edge"
EXPORT_SIZE_MODE_KEY = "export_size_preset"

# JPEG quality tiers, highest first: (slider/config value, human label)
EXPORT_QUALITY_LEVELS = (
    (10, "极高画质"),
    (9, "高画质 (推荐)"),
    (8, "较高画质"),
    (7, "标准偏上"),
    (6, "标准"),
    (5, "中等"),
    (4, "偏低"),
    (3, "低"),
    (2, "很低"),
    (1, "最低"),
)


def crop_source_pixels(src_w, src_h, crop=None):
    """Pixel rectangle a 100 % export produces: crop fractions applied, 90 deg swaps axes."""
    w, h = int(src_w or 0), int(src_h or 0)
    if isinstance(crop, dict) and crop.get("enabled"):
        try:
            w = int(round(w * float(crop.get("w", 1.0))))
            h = int(round(h * float(crop.get("h", 1.0))))
        except (TypeError, ValueError):
            pass
        if int(crop.get("rot_90", 0)) % 180 == 90:
            w, h = h, w
    return max(1, w), max(1, h)


def resolve_output_pixels(preset_key, custom_edge, src_w, src_h, crop=None):
    """(width, height) for a size preset, never upscaling past the source.

    Mirrors ExportSizeWidget.output_pixels() for callers that have no dialog (quick export).
    The preset pins the *long* edge, whichever axis that is.
    """
    w, h = crop_source_pixels(src_w, src_h, crop)
    edge = int(custom_edge or 0) if preset_key == "custom" else 0
    if preset_key != "custom":
        for name, _label, long_edge, _paper in EXPORT_SIZE_PRESETS:
            if name == preset_key:
                edge = long_edge
                break
    long_edge = max(w, h)
    if edge <= 0 or edge >= long_edge:
        return w, h
    scale = edge / float(long_edge)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


class ExportSizeWidget(QWidget):
    """Size selector shared by the single and batch dialogs.

    Shows the real output pixel dimensions for the photo being exported instead of a bare
    percentage, plus the physical print size at the chosen DPI, so the choice is concrete.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._src_w = 0
        self._src_h = 0
        self._crop = None
        self._dpi = 300

        # A single combo, never a second widget: when "自定义长边" is chosen the combo itself
        # becomes editable and takes the number, so the two control columns always end up
        # exactly the same length.
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.combo = QComboBox()
        setup_solid_combobox(self.combo)
        for key, label, edge, _paper in EXPORT_SIZE_PRESETS:
            self.combo.addItem(label, key)
        lay.addWidget(self.combo, 1)
        self.sub_row = lay  # kept for callers

        self.combo.currentIndexChanged.connect(self._on_preset_changed)

    # --- source context ------------------------------------------------------------------
    def set_source(self, width=None, height=None, crop=None):
        """Tell the widget the master pixel size and the crop rect (fractions of the master)."""
        try:
            w = int(width or 0)
            h = int(height or 0)
        except (TypeError, ValueError):
            w = h = 0
        if w > 0 and h > 0:
            self._src_w, self._src_h = w, h
        self._crop = crop if isinstance(crop, dict) and crop.get("enabled") else None
        self.refresh()

    def set_dpi(self, dpi):
        try:
            self._dpi = max(1, int(dpi))
        except (TypeError, ValueError):
            self._dpi = 300
        self._update_summary()

    def _print_label(self, key):
        for name, label, _long_edge, _paper in EXPORT_SIZE_PRESETS:
            if name == key:
                return label
        return ""

    def _max_short_edge_dpi(self, long_edge, paper):
        """DPI the *short* edge can sustain at this long edge (the binding constraint)."""
        pw, ph = paper
        short_mm = min(pw, ph)
        if short_mm <= 0 or long_edge <= 0:
            return 300
        return max(36, int(round(long_edge * short_mm / max(pw, ph) / (short_mm / 25.4))))

    # --- geometry ------------------------------------------------------------------------
    def source_pixels(self):
        """The pixel rectangle the renderer will actually produce at 100 % (crop applied)."""
        w, h = self._src_w, self._src_h
        if self._crop:
            try:
                w = int(round(w * float(self._crop.get("w", 1.0))))
                h = int(round(h * float(self._crop.get("h", 1.0))))
            except (TypeError, ValueError):
                pass
        if int(self._crop.get("rot_90", 0) if self._crop else 0) % 180 == 90:
            w, h = h, w
        return max(1, w), max(1, h)

    def _target_long_edge(self):
        key = self.combo.currentData()
        for name, _label, long_edge, _paper in EXPORT_SIZE_PRESETS:
            if name == key:
                if name == "custom":
                    return self._custom_edge_value()
                return long_edge
        return 0

    def output_pixels(self):
        """(width, height) the file will have, never upscaling past the source.

        The preset pins the *long* edge, whichever axis that is: a portrait frame must stay
        portrait instead of being forced to `edge` pixels wide.
        """
        src_w, src_h = self.source_pixels()
        edge = self._target_long_edge()
        long_edge = max(src_w, src_h)
        if edge <= 0 or edge >= long_edge:
            return src_w, src_h
        scale = edge / float(long_edge)
        return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))

    def scale_pct(self):
        src_w, src_h = self.source_pixels()
        out_w, out_h = self.output_pixels()
        src_long = max(src_w, src_h)
        if src_long <= 0:
            return 100
        return max(1, min(100, int(round(max(out_w, out_h) * 100.0 / src_long))))

    # --- state ---------------------------------------------------------------------------
    def _preset_row(self, key):
        for idx, (name, _label, _edge, _paper) in enumerate(EXPORT_SIZE_PRESETS):
            if name == key:
                return idx
        return 0

    def _on_preset_changed(self, idx):
        is_custom = self.combo.currentData() == "custom"
        self._set_editable(is_custom)
        if is_custom:
            # Show the edge only: no "自定义长边…" prefix, no parentheses, no unit suffix.
            edge = self._custom_edge_value()
            self.combo.setEditText(str(edge))
            edit = self.combo.lineEdit()
            if edit is not None:
                edit.setText(str(edge))
                edit.selectAll()
                self._align_custom_editor()
        else:
            # Re-selecting the row restores the preset's own label after a custom edit.
            self.combo.setCurrentIndex(idx)
        self._update_summary()

    def _set_editable(self, editable):
        if self.combo.isEditable() == editable:
            return
        self.combo.setEditable(editable)
        if editable:
            from PySide6.QtGui import QIntValidator
            edit = self.combo.lineEdit()
            if edit is not None:
                edit.setValidator(QIntValidator(64, 30000, self))
                edit.setPlaceholderText("长边 px")
                # A styled combo reserves the arrow through the drop-down subcontrol, which Qt
                # then subtracts from the *left* padding of the editor. Pin the editor's own
                # geometry so the digits start where the preset text starts.
                edit.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                edit.setStyleSheet(
                    "QLineEdit { padding: 0px 0px 0px 0px; margin: 0px; "
                    "border: none; background: transparent; }")
                edit.setContentsMargins(0, 0, 0, 0)
                edit.setTextMargins(0, 0, 0, 0)
                edit.editingFinished.connect(self._update_summary)
                edit.returnPressed.connect(self._update_summary)
                self._align_custom_editor()

    def _align_custom_editor(self):
        """Match the editable text inset to the preset text inset (8 px, as the combo style)."""
        edit = self.combo.lineEdit()
        if edit is None:
            return
        try:
            arrow = self.combo.style().subControlRect(
                QStyle.ComplexControl.CC_ComboBox,
                QStyleOptionComboBox(),
                QStyle.SubControl.SC_ComboBoxArrow,
                self.combo)
            inset = max(0, arrow.left())
        except Exception:
            inset = 0
        edit.setTextMargins(inset, 0, 0, 0)

    def showEvent(self, event):
        super().showEvent(event)
        if self.combo.isEditable():
            self._align_custom_editor()

    def _custom_edge_value(self):
        """Long edge the custom box currently holds, falling back to the source long edge."""
        raw = self.combo.currentText() if self.combo.isEditable() else ""
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        if digits:
            try:
                return max(64, min(30000, int(digits)))
            except ValueError:
                pass
        # No usable number yet: fall back to the source long edge (never to output_pixels(),
        # which would recurse back into this lookup).
        return max(1, max(self.source_pixels()))

    def set_custom_edge(self, edge):
        """Programmatically pick the custom row and set its long edge."""
        try:
            edge = max(64, min(30000, int(edge)))
        except (TypeError, ValueError):
            return
        if self.combo.currentData() != "custom":
            self.combo.setCurrentIndex(self._preset_row("custom"))
        else:
            self._set_editable(True)
        self.combo.setEditText(str(edge))
        self._update_summary()

    def _on_custom_edited(self):
        self._update_summary()

    def _base_labels(self):
        """Preset labels."""
        return {name: label for name, label, _edge, _paper in EXPORT_SIZE_PRESETS}

    def _update_summary(self):
        """Fold the real pixel size into the combo's own text so it needs no extra row."""
        out_w, out_h = self.output_pixels()
        labels = self._base_labels()
        if not self.combo.isEditable():
            current = self.combo.currentData()
            for name, text in labels.items():
                row = self._preset_row(name)
                if name == current and name != "custom":
                    self.combo.setItemText(row, f"{text} · {out_w}×{out_h}")
                else:
                    self.combo.setItemText(row, text)
        summary = f"{out_w} × {out_h} px"
        if self._dpi > 0 and self._src_w > 0:
            cm_w = out_w / self._dpi * 2.54
            cm_h = out_h / self._dpi * 2.54
            summary += f"  ·  {cm_w:.1f} × {cm_h:.1f} cm @ {self._dpi}dpi"
        self.setToolTip(summary)
        self.combo.setToolTip(summary)
        self.setProperty("summary", summary)
        # The owning dialogs render their own label; walk up the parent chain because Qt
        # re-parents the widget onto an internal page container.
        node = self.parent()
        for _ in range(6):
            if node is None:
                break
            if hasattr(node, "_on_export_size_changed"):
                node._on_export_size_changed(out_w, out_h)
                break
            node = node.parent()

    def set_scale_pct(self, pct):
        """Restore a previously used size (older configs stored a percentage)."""
        try:
            pct = int(pct)
        except (TypeError, ValueError):
            pct = 100
        if pct >= 100:
            self.combo.setCurrentIndex(self._preset_row("original"))
            return
        src_w, src_h = self.source_pixels()
        edge = max(1, int(round(max(src_w, src_h) * pct / 100.0)))
        self.set_custom_edge(edge)

    def select_preset(self, key):
        """Select a preset by its key (source-independent, used when restoring settings)."""
        row = self._preset_row(key)
        if row is None or self.combo.itemData(row) != key:
            return
        self.combo.setCurrentIndex(row)
        if self.combo.isEditable():
            self.combo.setEditable(False)
        self._update_summary()

    def restore_long_edge(self, edge):
        """Restore by absolute long edge (everything saved from now on)."""
        try:
            edge = int(edge or 0)
        except (TypeError, ValueError):
            edge = 0
        if edge <= 0:
            self.combo.setCurrentIndex(self._preset_row("original"))
            return
        for name, _label, preset_edge, _paper in EXPORT_SIZE_PRESETS:
            if name not in ("original", "custom") and preset_edge == edge:
                self.combo.setCurrentIndex(self._preset_row(name))
                return
        self.set_custom_edge(edge)

    def refresh(self):
        self._update_summary()


class ExportPathCombo(QComboBox):
    """输出位置下拉框：把"路径输入框"换成两档选择，同时保留手输路径的能力。

    - 源文件路径：输出到每个源文件所在目录（批量时各张回各自的文件夹），字段只读。
    - 自定义路径：字段即编辑器，默认填上次使用的目录；上次为空则填源路径。

    保留 ``text()`` / ``setText()``，调用方不必知道自己拿到的其实是一个下拉框——
    格式切换时改写扩展名的那几处逻辑照旧可用。
    """

    MODE_SOURCE = "source"
    MODE_CUSTOM = "custom"
    LABEL_SOURCE = "源文件路径"
    LABEL_CUSTOM = "自定义路径"

    modeChanged = Signal(str)

    def __init__(self, source_path: str = "", custom_path: str = "", parent=None):
        super().__init__(parent)
        self._source_path = normalize_path(source_path) if source_path else ""
        # 上次为空则回落到源路径
        self._custom_path = normalize_path(custom_path) if custom_path else self._source_path
        self._syncing = False

        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setFixedHeight(26)
        self.addItem(self.LABEL_SOURCE, self.MODE_SOURCE)
        self.addItem(self.LABEL_CUSTOM, self.MODE_CUSTOM)
        setup_solid_combobox(self)

        self.setCurrentIndex(max(0, self.findData(self.MODE_CUSTOM)))
        self._sync_field()
        self.currentIndexChanged.connect(self._on_index_changed)
        self.lineEdit().textEdited.connect(self._on_text_edited)

    # --- public -----------------------------------------------------------
    def mode(self) -> str:
        data = self.currentData()
        return data if data in (self.MODE_SOURCE, self.MODE_CUSTOM) else self.MODE_CUSTOM

    def is_source_mode(self) -> bool:
        return self.mode() == self.MODE_SOURCE

    def custom_path(self) -> str:
        return self._custom_path

    def text(self) -> str:
        """当前生效的路径。源文件模式下返回源路径（调用方通常还会按实际照片再算一次）。"""
        if self.is_source_mode():
            return self._source_path
        raw = self.lineEdit().text().strip()
        return normalize_path(raw) if raw else ""

    def setText(self, value: str):
        """写入路径 = 切到自定义模式。浏览对话框和扩展名改写都走这里。"""
        path = normalize_path(value) if value else ""
        self._custom_path = path
        if self.mode() != self.MODE_CUSTOM:
            self._syncing = True
            self.setCurrentIndex(max(0, self.findData(self.MODE_CUSTOM)))
            self._syncing = False
        self.lineEdit().setReadOnly(False)
        self.lineEdit().setText(path)
        self.modeChanged.emit(self.mode())

    def set_mode(self, mode: str):
        idx = self.findData(mode)
        if idx < 0:
            return
        self._syncing = True
        self.setCurrentIndex(idx)
        self._sync_field()
        self._syncing = False
        self.modeChanged.emit(self.mode())

    def set_source_path(self, source_path: str):
        self._source_path = normalize_path(source_path) if source_path else ""
        if self.is_source_mode():
            self._sync_field()
        elif not self._custom_path:
            self._custom_path = self._source_path
            self._sync_field()

    # --- internals --------------------------------------------------------
    def _sync_field(self):
        is_source = self.is_source_mode()
        edit = self.lineEdit()
        edit.setReadOnly(is_source)
        if is_source:
            edit.setText(self.LABEL_SOURCE)
        else:
            edit.setText(self._custom_path or self._source_path)

    def _on_index_changed(self, _index):
        if self._syncing:
            return
        self._syncing = True
        self._sync_field()
        self._syncing = False
        self.modeChanged.emit(self.mode())

    def _on_text_edited(self, text):
        # 源文件模式下字段是只读的，能走到这里说明用户在自定义模式里敲字。
        self._custom_path = normalize_path(text.strip()) if text.strip() else ""


class BaseExportDialog(QDialog):
    """Base dialog encapsulating shared darkroom rendering and color management settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        from ui.window_utils import setup_solid_combobox, get_icon
        self.setWindowIcon(get_icon("dlg_export.png", is_dark=theme_manager.is_dark()))

    def showEvent(self, event):
        super().showEvent(event)
        self._match_option_widths()
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
        grid_top.setVerticalSpacing(8)
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
        lbl_cs.setFixedWidth(72)
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
        lbl_bit.setFixedWidth(72)
        grid_top.addWidget(lbl_bit, 1, 2)

        self.combo_depth = QComboBox()
        setup_solid_combobox(self.combo_depth)
        grid_top.addWidget(self.combo_depth, 1, 3)

        # Shared output-size row: every format can be scaled, so it sits above the format
        # pages instead of inside one of them.
        lbl_size = QLabel("输出尺寸:")
        lbl_size.setFixedWidth(72)
        self.lbl_output_size = lbl_size
        grid_top.addWidget(lbl_size, 2, 0)

        self.size_widget = ExportSizeWidget(self)
        grid_top.addWidget(self.size_widget, 2, 1)

        lbl_sub = QLabel("色彩抽样:")
        lbl_sub.setFixedWidth(72)
        grid_top.addWidget(lbl_sub, 2, 2)

        self.combo_subsampling = QComboBox()
        setup_solid_combobox(self.combo_subsampling)
        self.combo_subsampling.addItem("4:4:4 (无损色度)", "4:4:4")
        self.combo_subsampling.addItem("4:2:2 (标准压缩)", "4:2:2")
        self.combo_subsampling.addItem("4:2:0 (高压缩)", "4:2:0")
        grid_top.addWidget(self.combo_subsampling, 2, 3)

        # No extra hint line: the resolved pixel size is folded into the size combo's own text,
        # so the form keeps four evenly spaced rows.
        self.lbl_size_summary = None

        # Shared GPS Privacy Checkbox (Placed cleanly below path input in specific dialogs)
        self.chk_strip_gps = QCheckBox("去除 GPS 位置隐私 (保护个人安全)")
        self.chk_strip_gps.setObjectName("chkStripGps")
        self.chk_strip_gps.setChecked(True)
        self.chk_strip_gps.setToolTip("开启后将从导出文件的 EXIF 元数据中完全抹除 GPS 经纬度与海拔坐标，保留相机、镜头、光圈、快门与拍摄时间等专业参数。")

        # Shared overwrite policy: checked (default) replaces a same-name file; unchecked makes
        # the exporter append 1, 2, 3 ... instead of overwriting.
        self.chk_overwrite = QCheckBox("替换重名文件")
        self.chk_overwrite.setObjectName("chkOverwrite")
        self.chk_overwrite.setChecked(True)
        self.chk_overwrite.setToolTip("勾选时同目录同名文件直接覆盖；取消勾选则自动改名为 xxx_1、xxx_2 等，保留原文件。")

        # Row 3: Format-Specific Settings (Directly in grid_top columns 0, 1, 2, 3)
        self.lbl_format_opt1 = QLabel("画质档位:")
        self.lbl_format_opt1.setFixedWidth(72)
        grid_top.addWidget(self.lbl_format_opt1, 3, 0)

        self.stack_format_opt1 = QStackedWidget()
        self.stack_format_opt1.setFixedHeight(26)

        # 0: JPEG Quality
        self.combo_quality = QComboBox()
        setup_solid_combobox(self.combo_quality)
        for q, desc in EXPORT_QUALITY_LEVELS:
            self.combo_quality.addItem(f"{q} · {desc}", q)
        self.combo_quality.setCurrentIndex(self.combo_quality.findData(9))
        self.combo_quality.setToolTip("JPEG 画质档位 1–10，数字越大画质越高、体积越大")
        self.stack_format_opt1.addWidget(self.combo_quality)

        # 1: TIFF Compression
        self.combo_tiff_comp = QComboBox()
        setup_solid_combobox(self.combo_tiff_comp)
        self.combo_tiff_comp.addItem("LZW 压缩", "lzw")
        self.combo_tiff_comp.addItem("Zip / Deflate", "deflate")
        self.combo_tiff_comp.addItem("无压缩", "none")
        self.stack_format_opt1.addWidget(self.combo_tiff_comp)

        # 2: PNG Level
        self.combo_png_level = QComboBox()
        setup_solid_combobox(self.combo_png_level)
        self.combo_png_level.addItem("标准 (Level 6)", 6)
        self.combo_png_level.addItem("极限 (Level 9)", 9)
        self.combo_png_level.addItem("快速 (Level 1)", 1)
        self.stack_format_opt1.addWidget(self.combo_png_level)

        # 3: JPEG XL Mode
        self.combo_jxl_mode = QComboBox()
        setup_solid_combobox(self.combo_jxl_mode)
        self.combo_jxl_mode.addItem("高质量 (1.5)", "visual_15")
        self.combo_jxl_mode.addItem("视觉无损 (1.0)", "visual_10")
        self.combo_jxl_mode.addItem("高保真 (0.5)", "visual_05")
        self.combo_jxl_mode.addItem("无损", "lossless")
        self.combo_jxl_mode.setCurrentIndex(self.combo_jxl_mode.findData("visual_10"))
        self.stack_format_opt1.addWidget(self.combo_jxl_mode)

        grid_top.addWidget(self.stack_format_opt1, 3, 1)

        # Row 3 Column 2: Format Option 2 Label (for JXL: "编码精度:")
        self.lbl_format_opt2 = QLabel("")
        self.lbl_format_opt2.setFixedWidth(72)
        grid_top.addWidget(self.lbl_format_opt2, 3, 2)

        self.stack_format_opt2 = QStackedWidget()
        self.stack_format_opt2.setFixedHeight(26)

        # 0: JPEG Progressive
        self.chk_progressive = QCheckBox("渐进式")
        self.chk_progressive.setObjectName("chkProgressive")
        self.chk_progressive.setChecked(True)
        self.chk_progressive.setToolTip("渐进式 JPEG 在网络加载时由模糊到清晰，文件体积略大")
        self.stack_format_opt2.addWidget(self.chk_progressive)

        # 1: TIFF empty placeholder
        self.stack_format_opt2.addWidget(QWidget())

        # 2: PNG empty placeholder
        self.stack_format_opt2.addWidget(QWidget())

        # 3: JPEG XL Effort
        self.combo_jxl_effort = QComboBox()
        setup_solid_combobox(self.combo_jxl_effort)
        self.combo_jxl_effort.addItem("标准 (Effort 7)", 7)
        self.combo_jxl_effort.addItem("极限 (9)", 9)
        self.combo_jxl_effort.addItem("极速 (5)", 5)
        self.stack_format_opt2.addWidget(self.combo_jxl_effort)

        grid_top.addWidget(self.stack_format_opt2, 3, 3)

        # Compatibility alias
        self.stack_settings = self.stack_format_opt1

        self.grid_top = grid_top
        card_layout.addLayout(grid_top)
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        if hasattr(self, "combo_dpi"):
            self.combo_dpi.currentIndexChanged.connect(self._on_dpi_changed)

        return card, card_layout

    def _make_page(self):
        """Deprecated: kept for backward compatibility."""
        return QWidget()

    def _match_option_widths(self):
        """No-op: All rows share self.grid_top, guaranteed identical column widths."""
        pass

    def _on_dpi_changed(self, _index):
        if hasattr(self, "size_widget"):
            self.size_widget.set_dpi(self.combo_dpi.currentData() or 300)

    def _on_export_size_changed(self, out_w, out_h):
        """Called by ExportSizeWidget whenever the resulting pixel size changes."""
        # The label stays 输出尺寸: the editable combo already shows the number when custom.
        return

    def _on_format_changed(self, index):
        fmt = self.combo_format.currentData()
        self.combo_depth.clear()
        # Subsampling is a JPEG-only concept; keep the row otherwise clean.
        self.combo_subsampling.setEnabled(fmt == "jpeg")
        self.combo_subsampling.setToolTip("" if fmt == "jpeg" else "色彩抽样仅对 JPEG 生效")
        # Stack pages: 0 JPEG, 1 TIFF, 2 PNG, 3 JPEG XL
        if fmt == "jpeg":
            self.lbl_format_opt1.setText("画质档位:")
            self.stack_format_opt1.setCurrentIndex(0)
            self.lbl_format_opt2.setText("")
            self.stack_format_opt2.setCurrentIndex(0)
            self.combo_depth.addItem("8 位 (8-bit)", 8)
        elif fmt == "tiff":
            self.lbl_format_opt1.setText("压缩算法:")
            self.stack_format_opt1.setCurrentIndex(1)
            self.lbl_format_opt2.setText("")
            self.stack_format_opt2.setCurrentIndex(1)
            self.combo_depth.addItem("8 位 (8-bit)", 8)
            self.combo_depth.addItem("16 位 (16-bit)", 16)
            self.combo_depth.addItem("32 位浮点 (32-bit Float)", 32)
            self.combo_depth.setCurrentIndex(1)
        elif fmt == "jxl":
            self.lbl_format_opt1.setText("压缩模式:")
            self.stack_format_opt1.setCurrentIndex(3)
            self.lbl_format_opt2.setText("编码精度:")
            self.stack_format_opt2.setCurrentIndex(3)
            self.combo_depth.addItem("8 位 (8-bit)", 8)
            self.combo_depth.addItem("16 位 (16-bit)", 16)
            self.combo_depth.addItem("32 位浮点 (32-bit Float)", 32)
            self.combo_depth.setCurrentIndex(1)
        else:  # png
            self.lbl_format_opt1.setText("压缩级别:")
            self.stack_format_opt1.setCurrentIndex(2)
            self.lbl_format_opt2.setText("")
            self.stack_format_opt2.setCurrentIndex(2)
            self.combo_depth.addItem("8 位 (8-bit)", 8)
            self.combo_depth.addItem("16 位 (16-bit)", 16)

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

            # 输出位置模式也沿用上次的选择；之前没有记录过就保持"自定义路径"。
            pm = last.get("path_mode")
            if pm and isinstance(getattr(self, "edit_path", None), ExportPathCombo):
                if pm == ExportPathCombo.MODE_SOURCE:
                    self.edit_path.set_mode(ExportPathCombo.MODE_SOURCE)

            cs = last.get("colorspace", "sRGB")
            idx_cs = self.combo_colorspace.findData(cs)
            if idx_cs >= 0:
                self.combo_colorspace.setCurrentIndex(idx_cs)

            bd = last.get("bit_depth", 8)
            idx_bd = self.combo_depth.findData(bd)
            if idx_bd >= 0:
                self.combo_depth.setCurrentIndex(idx_bd)

            q = min(10, max(1, int(last.get("quality", 9))))
            if hasattr(self, "combo_quality"):
                idx_q = self.combo_quality.findData(q)
                if idx_q >= 0:
                    self.combo_quality.setCurrentIndex(idx_q)

            if hasattr(self, "size_widget"):
                mode = last.get(EXPORT_SIZE_MODE_KEY)
                edge = last.get(EXPORT_SIZE_PREF_KEY)
                if mode == "custom":
                    self.size_widget.set_custom_edge(edge or 0)
                elif mode:
                    self.size_widget.select_preset(mode)
                elif edge:
                    self.size_widget.restore_long_edge(edge)
                else:
                    # Older configs only stored a percentage of the long edge.
                    self.size_widget.set_scale_pct(last.get("scale_pct", 100))

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
            if hasattr(self, "size_widget"):
                self.size_widget.set_dpi(dpi_val)

            if hasattr(self, "chk_strip_gps"):
                self.chk_strip_gps.setChecked(bool(last.get("strip_gps", True)))

            if hasattr(self, "chk_overwrite"):
                self.chk_overwrite.setChecked(bool(last.get("overwrite", True)))

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
            "quality": int(self.combo_quality.currentData()),
            "subsampling": self.combo_subsampling.currentData(),
            "progressive": self.chk_progressive.isChecked(),
            "tiff_compression": self.combo_tiff_comp.currentData(),
            "png_level": self.combo_png_level.currentData(),
            "jxl_mode": mode_val,
            "jxl_lossless": is_lossless,
            "jxl_distance": dist_val,
            "jxl_effort": effort_val,
            # Output size: exact (width, height) resolved against the source geometry and the
            # crop rect. A bare long edge is ambiguous for portrait frames - passing the width
            # as the long edge is what squashed a 4024x6024 scan into 4024x4024.
            "out_size": list(self.size_widget.output_pixels()) if hasattr(self, "size_widget") else [0, 0],
            "out_long_edge": max(self.size_widget.output_pixels()) if hasattr(self, "size_widget") else 0,
            "scale_pct": self.size_widget.scale_pct() if hasattr(self, "size_widget") else 100,
            EXPORT_SIZE_PREF_KEY: self.size_widget._target_long_edge() if hasattr(self, "size_widget") else 0,
            EXPORT_SIZE_MODE_KEY: self.size_widget.combo.currentData() if hasattr(self, "size_widget") else "original",
            "overwrite": self.chk_overwrite.isChecked() if hasattr(self, "chk_overwrite") else True,
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
                QComboBox:disabled {{
                    color: #5b5d68;
                    background: #1a1b20;
                    border-color: #2a2c36;
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
                    background: #14151a;
                    color: #e2e8f0;
                    border: 1px solid #323542;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                }}
                QLineEdit:focus {{
                    border: 1px solid #f59e0b;
                }}
                /* 可编辑下拉框内嵌的输入行：去掉自身边框，避免出现"框中框" */
                QComboBox QLineEdit {{
                    background: transparent;
                    border: none;
                    padding: 0px;
                    margin: 0px;
                    color: #e2e8f0;
                    font-size: 11px;
                }}
                QComboBox QLineEdit:read-only {{
                    color: #94a3b8;
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
                QCheckBox#chkStripGps, QCheckBox#chkOverwrite, QCheckBox#chkProgressive {{
                    color: #c4c4cc;
                    font-size: 12px;
                    font-weight: normal;
                    spacing: 7px;
                    background: transparent;
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
                QPushButton:disabled {{
                    background: #1c1d23;
                    color: #5b5d68;
                    border-color: #2a2c36;
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
                QComboBox:disabled {{
                    color: #a2a9b6;
                    background: #f1f3f7;
                    border-color: #dbe0ea;
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
                    padding: 4px 8px;
                    font-size: 11px;
                }}
                QLineEdit:focus {{
                    border: 1px solid #f59e0b;
                }}
                /* 可编辑下拉框内嵌的输入行：去掉自身边框，避免出现"框中框" */
                QComboBox QLineEdit {{
                    background: transparent;
                    border: none;
                    padding: 0px;
                    margin: 0px;
                    color: #1e293b;
                    font-size: 11px;
                }}
                QComboBox QLineEdit:read-only {{
                    color: #64748b;
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
                QCheckBox#chkStripGps, QCheckBox#chkOverwrite, QCheckBox#chkProgressive {{
                    color: #334155;
                    font-size: 12px;
                    font-weight: normal;
                    spacing: 7px;
                    background: transparent;
                }}
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
                QPushButton:disabled {{
                    background: #f1f3f7;
                    color: #a2a9b6;
                    border-color: #e2e8f0;
                }}
            """)


class SingleExportDialog(BaseExportDialog):
    """Export dialog for single photo development with precise file destination."""

    def __init__(self, default_file_path: str, parent=None, source_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("单片扫描")
        self.setFixedSize(620, 360)

        self._default_file_path = normalize_path(default_file_path) if default_file_path else ""
        # 源文件路径模式需要知道底片落在哪；缺省时按默认输出路径反推目录。
        self._source_path = normalize_path(source_path) if source_path else self._default_file_path
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
        lbl_path = QLabel("输出位置:")
        lbl_path.setStyleSheet("font-size: 12px; background: transparent;")
        row_path.addWidget(lbl_path)

        path_box = QHBoxLayout()
        path_box.setSpacing(8)
        self.edit_path = ExportPathCombo(
            source_path=self._source_path,
            custom_path=self._default_file_path,
            parent=self,
        )
        path_box.addWidget(self.edit_path, 1)

        btn_browse = QPushButton("浏览...")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self._browse_save_path)
        path_box.addWidget(btn_browse)
        self._btn_browse = btn_browse

        row_path.addLayout(path_box)

        self.lbl_path_hint = QLabel("")
        self.lbl_path_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        row_path.addWidget(self.lbl_path_hint)

        self.edit_path.modeChanged.connect(self._on_path_mode_changed)
        self._on_path_mode_changed(self.edit_path.mode())

        card_layout.addLayout(row_path)

        # Row 5: GPS Privacy Checkbox directly below path input box inside the card
        row_opts = QHBoxLayout()
        row_opts.setContentsMargins(0, 4, 0, 0)
        row_opts.setSpacing(18)
        row_opts.addWidget(self.chk_strip_gps)
        row_opts.addWidget(self.chk_overwrite)
        row_opts.addStretch()
        card_layout.addLayout(row_opts)

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

        # Buttons row (shifted leftward away from far-right border)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
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
        btn_row.addSpacing(24)

        main_layout.addLayout(btn_row)

    def _on_path_mode_changed(self, mode):
        is_source = mode == ExportPathCombo.MODE_SOURCE
        if hasattr(self, "_btn_browse"):
            self._btn_browse.setEnabled(not is_source)
            self._btn_browse.setCursor(
                Qt.CursorShape.PointingHandCursor if not is_source else Qt.CursorShape.ForbiddenCursor)
        if hasattr(self, "lbl_path_hint"):
            self.lbl_path_hint.setText(
                "输出到源文件所在目录，文件名沿用原文件名，扩展名随格式自动修正"
                if is_source else "可手动输入路径，或点“浏览...”选择保存位置")

    def _on_format_changed(self, index):
        super()._on_format_changed(index)
        fmt = self.combo_format.currentData()
        # 源文件模式下路径不归用户管，扩展名在导出时按格式派发，别在这里改写文本。
        if isinstance(self.edit_path, ExportPathCombo) and self.edit_path.is_source_mode():
            return
        cur_path = self.edit_path.text()
        if cur_path:
            base, _ = os.path.splitext(cur_path)
            new_ext = ".tif" if fmt == "tiff" else (".jxl" if fmt == "jxl" else (".png" if fmt == "png" else ".jpg"))
            self.edit_path.setText(base + new_ext)

    def set_source_context(self, width=None, height=None, crop=None):
        """Size the output selector from the photo about to be scanned."""
        if hasattr(self, "size_widget"):
            self.size_widget.set_dpi(self.combo_dpi.currentData() or 300)
            self.size_widget.set_source(width, height, crop)

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
        path_mode = self.edit_path.mode() if isinstance(self.edit_path, ExportPathCombo) else "custom"
        cfg["path_mode"] = path_mode
        if path_mode == ExportPathCombo.MODE_SOURCE:
            # 目标由调用方按真实源文件算（对话框只知道一个副本），这里不猜路径。
            cfg["path"] = ""
            cfg["is_batch"] = False
            return cfg

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

    def __init__(self, default_dir: str, batch_count: int = 0, parent=None, default_stem: str = "",
                 source_dir: str = ""):
        super().__init__(parent)
        self.batch_count = batch_count
        self._default_stem = default_stem or ""
        # One photo and many photos go through this same dialog on purpose: the batch
        # command used to swap in SingleExportDialog for a single selection, so the naming
        # fields silently changed meaning depending on how many frames were selected.
        if batch_count == 1:
            self.setWindowTitle("扫描当前底片")
            self.setFixedSize(620, 410)
        else:
            self.setWindowTitle(f"批量扫描：{self.batch_count}张")
            self.setFixedSize(620, 410)

        # Ensure directory only, never a file
        clean_dir = normalize_path(default_dir) if default_dir else ""
        if clean_dir and not os.path.isdir(clean_dir):
            clean_dir = os.path.dirname(clean_dir)
        if not clean_dir or not os.path.exists(clean_dir):
            clean_dir = config_manager.get_last_export_dir() or ""
        if not clean_dir:
            # 没有任何"上次路径"时回落到源文件目录，而不是家目录。
            clean_dir = normalize_path(source_dir) if source_dir else os.path.expanduser("~")
        self._default_dir = clean_dir
        # 源文件路径模式要把每张写回自己的目录；这里只留一个代表值给对话框显示。
        self._source_dir = normalize_path(source_dir) if source_dir else clean_dir

        self._init_ui()
        self._apply_style()
        self._apply_last_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # 1. Shared Settings Card
        card, card_layout = self._init_settings_card()

        # Row 4: File naming. A single frame gets an exact filename; a real batch gets the
        # prefix/suffix pair. Same dialog, same position, so switching selection size no
        # longer changes what the fields mean without warning.
        if self.batch_count == 1:
            lbl_name = QLabel("文件名:")
            lbl_name.setFixedWidth(72)
            self.grid_top.addWidget(lbl_name, 4, 0)

            self.edit_filename = QLineEdit(self._default_stem)
            self.edit_filename.setFixedHeight(26)
            self.edit_filename.setPlaceholderText("留空则使用原文件名")
            self.grid_top.addWidget(self.edit_filename, 4, 1, 1, 3)

            self.edit_prefix = None
            self.edit_suffix = None

            lbl_hint = QLabel("扩展名随格式自动修正")
            lbl_hint.setStyleSheet("color: #64748b; font-size: 11px;")
            self.grid_top.addWidget(lbl_hint, 5, 1, 1, 3)
        else:
            lbl_pre = QLabel("文件前缀:")
            lbl_pre.setFixedWidth(72)
            self.grid_top.addWidget(lbl_pre, 4, 0)

            self.edit_prefix = QLineEdit()
            self.edit_prefix.setFixedHeight(26)
            self.edit_prefix.setPlaceholderText("无")
            self.grid_top.addWidget(self.edit_prefix, 4, 1)

            lbl_suf = QLabel("文件后缀:")
            lbl_suf.setFixedWidth(72)
            self.grid_top.addWidget(lbl_suf, 4, 2)

            self.edit_suffix = QLineEdit()
            self.edit_suffix.setFixedHeight(26)
            self.edit_suffix.setPlaceholderText("无")
            self.grid_top.addWidget(self.edit_suffix, 4, 3)

            self.edit_filename = None

            # Row 5: Naming hint on next line
            lbl_hint = QLabel("留空则保持原文件名输出")
            lbl_hint.setStyleSheet("color: #64748b; font-size: 11px;")
            self.grid_top.addWidget(lbl_hint, 5, 1, 1, 3)

        # Row 6: Strictly Destination Directory
        row_path = QVBoxLayout()
        row_path.setContentsMargins(0, 2, 0, 0)
        row_path.setSpacing(6)
        lbl_path = QLabel("输出位置:")
        lbl_path.setStyleSheet("font-size: 12px; background: transparent;")
        row_path.addWidget(lbl_path)

        path_box = QHBoxLayout()
        path_box.setSpacing(8)
        self.edit_path = ExportPathCombo(
            source_path=self._source_dir,
            custom_path=self._default_dir,
            parent=self,
        )
        path_box.addWidget(self.edit_path, 1)

        btn_browse = QPushButton("浏览...")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self._browse_save_dir)
        path_box.addWidget(btn_browse)
        self._btn_browse = btn_browse

        row_path.addLayout(path_box)

        self.lbl_path_hint = QLabel("")
        self.lbl_path_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        row_path.addWidget(self.lbl_path_hint)

        self.edit_path.modeChanged.connect(self._on_path_mode_changed)
        self._on_path_mode_changed(self.edit_path.mode())

        card_layout.addLayout(row_path)

        # Row 6: GPS Privacy Checkbox directly below destination path inside the card
        row_opts = QHBoxLayout()
        row_opts.setContentsMargins(0, 4, 0, 0)
        row_opts.setSpacing(18)
        row_opts.addWidget(self.chk_strip_gps)
        row_opts.addWidget(self.chk_overwrite)
        row_opts.addStretch()
        card_layout.addLayout(row_opts)

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
        btn_row.addSpacing(24)

        main_layout.addLayout(btn_row)

    def _on_path_mode_changed(self, mode):
        is_source = mode == ExportPathCombo.MODE_SOURCE
        if hasattr(self, "_btn_browse"):
            self._btn_browse.setEnabled(not is_source)
            self._btn_browse.setCursor(
                Qt.CursorShape.PointingHandCursor if not is_source else Qt.CursorShape.ForbiddenCursor)
        if hasattr(self, "lbl_path_hint"):
            self.lbl_path_hint.setText(
                "每张输出到各自源文件所在目录，文件名沿用原名"
                if is_source else "全部输出到同一个文件夹，可手动输入或点“浏览...”选择")

    def _browse_save_dir(self):
        start_dir = self.edit_path.text().strip()
        if not os.path.isdir(start_dir):
            start_dir = config_manager.get_last_export_dir() or os.path.expanduser("~")
        dir_chosen = QFileDialog.getExistingDirectory(self, "选择目标输出文件夹", start_dir)
        if dir_chosen:
            self.edit_path.setText(normalize_path(dir_chosen))

    def set_source_context(self, width=None, height=None, crop=None):
        """Size the output selector from a representative photo of the batch."""
        self.size_widget.set_dpi(self.combo_dpi.currentData() or 300)
        self.size_widget.set_source(width, height, crop)

    def get_export_config(self) -> dict:
        cfg = self.get_common_export_config()
        path_mode = self.edit_path.mode() if isinstance(self.edit_path, ExportPathCombo) else "custom"
        cfg["path_mode"] = path_mode
        if path_mode == ExportPathCombo.MODE_SOURCE:
            # 目标目录按每张照片各自的源目录算，这里留空由调用方决定。
            cfg["path"] = ""
        else:
            out_dir = normalize_path(self.edit_path.text().strip())
            if out_dir and not os.path.isdir(out_dir):
                # Strip accidental file suffix if user typed filename
                out_dir = os.path.dirname(out_dir) if os.path.splitext(out_dir)[1] else out_dir
            cfg["path"] = normalize_path(out_dir)
        if self.batch_count == 1:
            raw_name = self.edit_filename.text().strip() if self.edit_filename is not None else ""
            if raw_name:
                # Drop any extension the user typed: it is re-derived from the format below.
                raw_name = os.path.splitext(os.path.basename(raw_name))[0]
            cfg["filename"] = raw_name
            cfg["prefix"] = ""
            cfg["suffix"] = ""
        else:
            cfg["filename"] = ""
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
