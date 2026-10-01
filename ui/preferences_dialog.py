"""
preferences_dialog.py - Professional Application Preferences Dialog
Settings:
- Hardware Acceleration (GPU OpenGL / Vulkan)
- Default Film Format (35mm / 16mm / 120 / 4x5)
- Exit confirmation on unsaved edits
- Viewport Antialiasing & Default Color Space
- LUT Disk Cache management and cleanup
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QComboBox,
    QPushButton, QTabWidget, QWidget, QFrame, QMessageBox, QSpinBox
)

from path_utils import get_resource_dir
from ui.window_utils import apply_dark_titlebar
import config_manager
import session_cache_manager


class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("首选项 (Preferences)")
        self.setFixedSize(540, 420)
        self.setModal(True)

        icon_path = os.path.join(get_resource_dir(), "icons", "dlg_prefs.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._init_ui()
        self._load_preferences()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self):
        chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")
        arrow_up = os.path.join(get_resource_dir(), "icons", "arrow_up.png").replace("\\", "/")
        arrow_down = os.path.join(get_resource_dir(), "icons", "arrow_down.png").replace("\\", "/")

        self.setStyleSheet(f"""
            QDialog {{
                background: #14151a;
                color: #e2e8f0;
            }}
            QTabWidget::pane {{
                border: 1px solid #282a36;
                background: #181920;
                border-radius: 6px;
                top: -1px;
            }}
            QTabBar::tab {{
                background: #121317;
                color: #94a3b8;
                padding: 8px 18px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
                font-size: 12px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: #181920;
                color: #f59e0b;
                border-bottom: 2px solid #f59e0b;
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                color: #ffffff;
                background: #1c1d25;
            }}
            QFrame#card {{
                background: transparent;
                border: none;
            }}
            QLabel {{
                color: #cbd5e1;
                font-size: 12px;
            }}
            QCheckBox {{
                color: #e2e8f0;
                font-size: 12px;
                spacing: 8px;
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
            QComboBox {{
                background: #1f212a;
                color: #e2e8f0;
                border: 1px solid #333647;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }}
            QSpinBox {{
                background: #181922;
                color: #f1f5f9;
                border: 1px solid #333647;
                border-radius: 4px;
                padding-left: 8px;
                padding-right: 24px;
                font-size: 12px;
                font-weight: 500;
                selection-background-color: #f59e0b;
                selection-color: #000000;
            }}
            QSpinBox:focus {{
                border-color: #f59e0b;
            }}
            QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px;
                height: 14px;
                border-left: 1px solid #333647;
                border-bottom: 1px solid #333647;
                border-top-right-radius: 3px;
                background: #252836;
            }}
            QSpinBox::up-button:hover {{
                background: #f59e0b;
            }}
            QSpinBox::up-button:pressed {{
                background: #d97706;
            }}
            QSpinBox::up-arrow {{
                image: url("{arrow_up}");
                width: 8px;
                height: 6px;
            }}
            QSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px;
                height: 14px;
                border-left: 1px solid #333647;
                border-bottom-right-radius: 3px;
                background: #252836;
            }}
            QSpinBox::down-button:hover {{
                background: #f59e0b;
            }}
            QSpinBox::down-button:pressed {{
                background: #d97706;
            }}
            QSpinBox::down-arrow {{
                image: url("{arrow_down}");
                width: 8px;
                height: 6px;
            }}
            QPushButton {{
                background: #1f212a;
                color: #e2e8f0;
                border: 1px solid #333647;
                border-radius: 4px;
                padding: 6px 18px;
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
            QPushButton#btnSave {{
                background: #f59e0b;
                color: #111;
                border: none;
                font-weight: bold;
            }}
            QPushButton#btnSave:hover {{
                background: #d97706;
            }}
            QPushButton#btnSave:pressed {{
                background: #b45309;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        self.tabs = QTabWidget()

        # Tab 1: 图像与渲染
        tab_render = QWidget()
        l_ren = QVBoxLayout(tab_render)
        l_ren.setContentsMargins(16, 16, 16, 16)
        l_ren.setSpacing(14)

        r_hw = QHBoxLayout()
        lbl_hw = QLabel("硬件加速模式:")
        lbl_hw.setFixedWidth(115)
        r_hw.addWidget(lbl_hw)
        self.combo_hw_accel = QComboBox()
        self.combo_hw_accel.addItem("关 (纯 CPU 渲染)", "off")
        self.combo_hw_accel.addItem("仅缩略图与视口", "preview_only")
        self.combo_hw_accel.addItem("全局 (GPU 加速导出)", "global")
        r_hw.addWidget(self.combo_hw_accel, 1)
        l_ren.addLayout(r_hw)

        r_res = QHBoxLayout()
        lbl_res = QLabel("视口实时预览分辨率:")
        lbl_res.setFixedWidth(115)
        r_res.addWidget(lbl_res)
        self.combo_preview_res = QComboBox()
        self.combo_preview_res.addItem("小 (1080P)", 1440)
        self.combo_preview_res.addItem("中 (2K - 推荐)", 2048)
        self.combo_preview_res.addItem("大 (4K 极致)", 3840)
        r_res.addWidget(self.combo_preview_res, 1)
        l_ren.addLayout(r_res)

        self.chk_hq_preview = QCheckBox("高画质视口双线性平滑与抗锯齿 (HQ Viewport Anti-Aliasing)")
        self.chk_hq_preview.setChecked(True)
        l_ren.addWidget(self.chk_hq_preview)

        r_cs = QHBoxLayout()
        lbl_cs = QLabel("默认工作色彩空间:")
        lbl_cs.setFixedWidth(115)
        r_cs.addWidget(lbl_cs)
        self.combo_default_cs = QComboBox()
        self.combo_default_cs.addItems(["sRGB (标准通用)", "Display P3 (广色域屏)", "Adobe RGB (1998)"])
        r_cs.addWidget(self.combo_default_cs, 1)
        l_ren.addLayout(r_cs)

        l_ren.addStretch()
        self.tabs.addTab(tab_render, "图像与渲染")

        # Tab 2: 暗房与工作流
        tab_darkroom = QWidget()
        l_dr = QVBoxLayout(tab_darkroom)
        l_dr.setContentsMargins(16, 16, 16, 16)
        l_dr.setSpacing(14)

        r_fmt = QHBoxLayout()
        r_fmt.addWidget(QLabel("默认底片画幅规格:"))
        self.combo_default_fmt = QComboBox()
        self.combo_default_fmt.addItem("35mm 标准全画幅", 35.0)
        self.combo_default_fmt.addItem("16mm 独立电影卷", 16.0)
        self.combo_default_fmt.addItem("120 中画幅 (6x6)", 60.0)
        self.combo_default_fmt.addItem("4x5 大画幅", 100.0)
        r_fmt.addWidget(self.combo_default_fmt, 1)
        l_dr.addLayout(r_fmt)

        self.chk_restore_last_files = QCheckBox("打开时回到上一次打开的文件")
        self.chk_restore_last_files.setChecked(True)
        l_dr.addWidget(self.chk_restore_last_files)

        self.chk_restore_geo = QCheckBox("启动时恢复上次窗口位置与工作区分割比例")
        self.chk_restore_geo.setChecked(True)
        l_dr.addWidget(self.chk_restore_geo)

        l_dr.addStretch()
        self.tabs.addTab(tab_darkroom, "暗房与工作流")

        # Tab 3: 性能与缓存
        tab_cache = QWidget()
        l_ca = QVBoxLayout(tab_cache)
        l_ca.setContentsMargins(16, 16, 16, 16)
        l_ca.setSpacing(14)

        r_rec = QHBoxLayout()
        r_rec.addWidget(QLabel("最近打开文件记录上限:"))
        self.spin_recent_count = QSpinBox()
        self.spin_recent_count.setFixedWidth(74)
        self.spin_recent_count.setFixedHeight(28)
        self.spin_recent_count.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.spin_recent_count.setRange(5, 50)
        self.spin_recent_count.setValue(15)
        r_rec.addWidget(self.spin_recent_count)
        r_rec.addStretch()
        l_ca.addLayout(r_rec)

        r_cache = QHBoxLayout()
        self.lbl_cache_size = QLabel("3D LUT 磁盘缓存占用: 计算中...")
        r_cache.addWidget(self.lbl_cache_size, 1)
        btn_clean_cache = QPushButton("清理 LUT 缓存")
        btn_clean_cache.clicked.connect(self._clean_lut_cache)
        r_cache.addWidget(btn_clean_cache)
        l_ca.addLayout(r_cache)

        r_sess = QHBoxLayout()
        self.lbl_sess_cache_size = QLabel("底片会话快显缓存: 计算中...")
        r_sess.addWidget(self.lbl_sess_cache_size, 1)
        btn_clean_sess = QPushButton("清理底片缓存")
        btn_clean_sess.clicked.connect(self._clean_sess_cache)
        r_sess.addWidget(btn_clean_sess)
        l_ca.addLayout(r_sess)

        l_ca.addStretch()
        self.tabs.addTab(tab_cache, "性能与缓存")

        layout.addWidget(self.tabs, 1)

        # Footer Buttons
        footer = QHBoxLayout()
        footer.addStretch()
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        footer.addWidget(btn_cancel)

        btn_save = QPushButton("保存设置", objectName="btnSave")
        btn_save.clicked.connect(self._save_preferences)
        footer.addWidget(btn_save)

        layout.addLayout(footer)

    def _load_preferences(self):
        cfg = config_manager.load_config()
        prefs = cfg.get("preferences", {})

        hw_mode = prefs.get("hardware_acceleration_mode")
        if not hw_mode:
            old_bool = prefs.get("hardware_acceleration", True)
            hw_mode = "global" if old_bool else "off"
        idx_hw = self.combo_hw_accel.findData(hw_mode)
        self.combo_hw_accel.setCurrentIndex(idx_hw if idx_hw >= 0 else 2)

        prev_res = prefs.get("preview_max_edge", 2048)
        idx_res = self.combo_preview_res.findData(prev_res)
        self.combo_preview_res.setCurrentIndex(idx_res if idx_res >= 0 else 1)

        self.chk_hq_preview.setChecked(prefs.get("hq_preview", True))
        self.chk_restore_last_files.setChecked(prefs.get("restore_last_files", True))
        self.chk_restore_geo.setChecked(prefs.get("restore_window_state", True))
        self.spin_recent_count.setValue(prefs.get("recent_files_max", 15))

        def_fmt = prefs.get("default_film_format", 35.0)
        for i in range(self.combo_default_fmt.count()):
            if abs(self.combo_default_fmt.itemData(i) - def_fmt) < 1.0:
                self.combo_default_fmt.setCurrentIndex(i)
                break

        self._refresh_cache_size()

    def _refresh_cache_size(self):
        try:
            cache_dir = os.path.join(get_resource_dir(), ".lut_cache")
            if os.path.exists(cache_dir):
                total_bytes = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f)))
                sz_mb = total_bytes / (1024.0 * 1024.0)
                self.lbl_cache_size.setText(f"3D LUT 磁盘缓存: {sz_mb:.1f} MB")
            else:
                self.lbl_cache_size.setText("3D LUT 磁盘缓存: 0.0 MB")
        except Exception:
            self.lbl_cache_size.setText("3D LUT 磁盘缓存: 0.0 MB")

        try:
            sess_mb = session_cache_manager.get_cache_size_mb()
            self.lbl_sess_cache_size.setText(f"底片会话快显缓存: {sess_mb:.1f} MB")
        except Exception:
            self.lbl_sess_cache_size.setText("底片会话快显缓存: 0.0 MB")

    def _clean_lut_cache(self):
        try:
            cache_dir = os.path.join(get_resource_dir(), ".lut_cache")
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
                os.makedirs(cache_dir, exist_ok=True)
            self._refresh_cache_size()
            QMessageBox.information(self, "完成", "3D LUT 物理缓存已清空。")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"清理失败: {e}")

    def _clean_sess_cache(self):
        try:
            session_cache_manager.clear_cache()
            self._refresh_cache_size()
            QMessageBox.information(self, "完成", "底片会话快显缓存已清空。")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"清理底片缓存失败: {e}")

    def _save_preferences(self):
        cfg = config_manager.load_config()
        if "preferences" not in cfg:
            cfg["preferences"] = {}

        prefs = cfg["preferences"]
        hw_mode = self.combo_hw_accel.currentData()
        prefs["hardware_acceleration_mode"] = hw_mode
        prefs["hardware_acceleration"] = (hw_mode != "off")
        prefs["preview_max_edge"] = int(self.combo_preview_res.currentData())
        prefs["hq_preview"] = self.chk_hq_preview.isChecked()
        prefs["restore_last_files"] = self.chk_restore_last_files.isChecked()
        prefs["restore_window_state"] = self.chk_restore_geo.isChecked()
        prefs["default_film_format"] = float(self.combo_default_fmt.currentData())
        prefs["recent_files_max"] = int(self.spin_recent_count.value())

        config_manager.save_config(cfg)
        self.accept()
