"""
preferences_dialog.py - Professional Application Preferences Dialog
Settings:
- Hardware Acceleration (GPU OpenGL)
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
    QPushButton, QTabWidget, QWidget, QFrame, QMessageBox, QSpinBox, QLineEdit
)
from ui.confirm_dialog import show_darkroom_confirm, show_darkroom_info

from core.path_utils import get_resource_dir, get_icon_path
from ui.window_utils import apply_dark_titlebar, show_dark_message_box, get_icon, setup_solid_combobox, get_standard_checkbox_style
from ui.theme_manager import theme_manager
from core import config_manager
from core import session_cache_manager


class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("首选项")
        self.setFixedSize(520, 460)
        self.setModal(True)

        is_dark = theme_manager.is_dark()
        self.setWindowIcon(get_icon("dlg_prefs.png", is_dark=is_dark))

        self._init_ui()
        self._load_preferences()
        theme_manager.themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme_name=None):
        self._apply_dialog_theme()
        is_dark = theme_manager.is_dark()
        self.setWindowIcon(get_icon("dlg_prefs.png", is_dark=is_dark))
        apply_dark_titlebar(self, is_dark=is_dark)
        if hasattr(self, "btn_assoc"):
            self._update_assoc_button()

    def showEvent(self, event):
        super().showEvent(event)
        is_dark = theme_manager.is_dark()
        self.setWindowIcon(get_icon("dlg_prefs.png", is_dark=is_dark))
        apply_dark_titlebar(self, is_dark=is_dark)

    def _apply_dialog_theme(self):
        chk_amber = os.path.join(get_resource_dir(), "icons", "chk_checked_amber.png").replace("\\", "/")
        arrow_up = os.path.join(get_resource_dir(), "icons", "arrow_up.png").replace("\\", "/")
        arrow_down = os.path.join(get_resource_dir(), "icons", "arrow_down.png").replace("\\", "/")
        arrow_up_dark = os.path.join(get_resource_dir(), "icons", "arrow_up_dark.png").replace("\\", "/")
        arrow_down_dark = os.path.join(get_resource_dir(), "icons", "arrow_down_dark.png").replace("\\", "/")
        if not os.path.exists(arrow_up_dark):
            arrow_up_dark = arrow_up
        if not os.path.exists(arrow_down_dark):
            arrow_down_dark = arrow_down

        is_dark = theme_manager.is_dark()
        arrow_name = "arrow_down.png" if is_dark else "arrow_down_dark.png"
        arrow_icon = get_icon_path(arrow_name).replace("\\", "/")

        self.setWindowIcon(get_icon("dlg_prefs.png", is_dark=is_dark))
        if hasattr(self, "btn_assoc"):
            self._update_assoc_button()
        if is_dark:
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
                QLabel#lblCachePath {{
                    color: #8b92a5;
                    font-size: 11px;
                    line-height: 1.3;
                }}
{get_standard_checkbox_style(is_dark=True)}
                QLineEdit {{
                    background: #14151a;
                    color: #e2e8f0;
                    border: 1px solid #323542;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                }}
                QLineEdit:hover {{
                    border-color: #f59e0b;
                }}
                QLineEdit:focus {{
                    border-color: #f59e0b;
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
                QFrame#sepAssoc, QFrame#sepLens {{
                    background: rgba(255, 255, 255, 0.08);
                    height: 1px;
                    max-height: 1px;
                    min-height: 1px;
                    border: none;
                    margin-top: 6px;
                    margin-bottom: 6px;
                }}
                QLabel#lblAssocTitle, QLabel#lblLensTitle {{
                    color: #e2e8f0;
                    font-size: 12px;
                    font-weight: 500;
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
                QScrollBar:vertical {{
                    background: rgba(18, 19, 24, 0.75);
                    width: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:vertical {{
                    background: #3f4350;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: #cbd5e1;
                    border-radius: 3px;
                }}
                QScrollBar::handle:vertical:pressed {{
                    background: #f59e0b;
                    border-radius: 3px;
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
            """)
        else:
            self.setStyleSheet(f"""
                QDialog {{
                    background: #ebedf2;
                    color: #1e293b;
                }}
                QTabWidget::pane {{
                    border: 1px solid #d5d9e4;
                    background: #ffffff;
                    border-radius: 6px;
                    top: -1px;
                }}
                QTabBar::tab {{
                    background: #e2e5ec;
                    color: #64748b;
                    padding: 8px 18px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    font-weight: 500;
                    font-size: 12px;
                    margin-right: 2px;
                }}
                QTabBar::tab:selected {{
                    background: #ffffff;
                    color: #d97706;
                    border-bottom: 2px solid #f59e0b;
                    font-weight: bold;
                }}
                QTabBar::tab:hover:!selected {{
                    color: #1e293b;
                    background: #edf0f6;
                }}
                QFrame#card {{
                    background: transparent;
                    border: none;
                }}
                QLabel {{
                    color: #334155;
                    font-size: 12px;
                }}
                QLabel#lblCachePath {{
                    color: #64748b;
                    font-size: 11px;
                    line-height: 1.3;
                }}
{get_standard_checkbox_style(is_dark=False)}
                QLineEdit {{
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                }}
                QLineEdit:hover {{
                    border-color: #f59e0b;
                }}
                QLineEdit:focus {{
                    border-color: #f59e0b;
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
                QFrame#sepAssoc, QFrame#sepLens {{
                    background: rgba(0, 0, 0, 0.08);
                    height: 1px;
                    max-height: 1px;
                    min-height: 1px;
                    border: none;
                    margin-top: 6px;
                    margin-bottom: 6px;
                }}
                QLabel#lblAssocTitle, QLabel#lblLensTitle {{
                    color: #1e293b;
                    font-size: 12px;
                    font-weight: 500;
                }}
                QSpinBox {{
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
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
                    border-left: 1px solid #cbd5e1;
                    border-bottom: 1px solid #cbd5e1;
                    border-top-right-radius: 3px;
                    background: #f1f4f9;
                }}
                QSpinBox::up-button:hover {{
                    background: #f59e0b;
                }}
                QSpinBox::up-button:pressed {{
                    background: #d97706;
                }}
                QSpinBox::up-arrow {{
                    image: url("{arrow_up_dark}");
                    width: 8px;
                    height: 6px;
                }}
                QSpinBox::down-button {{
                    subcontrol-origin: border;
                    subcontrol-position: bottom right;
                    width: 22px;
                    height: 14px;
                    border-left: 1px solid #cbd5e1;
                    border-bottom-right-radius: 3px;
                    background: #f1f4f9;
                }}
                QSpinBox::down-button:hover {{
                    background: #f59e0b;
                }}
                QSpinBox::down-button:pressed {{
                    background: #d97706;
                }}
                QSpinBox::down-arrow {{
                    image: url("{arrow_down_dark}");
                    width: 8px;
                    height: 6px;
                }}
                QPushButton {{
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 6px 18px;
                    font-size: 12px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background: #f8fafc;
                    border-color: #f59e0b;
                    color: #d97706;
                }}
                QPushButton:pressed {{
                    background: #e2e8f0;
                    color: #b45309;
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
                QScrollBar:vertical {{
                    background: rgba(235, 237, 242, 0.75);
                    width: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:vertical {{
                    background: #94a3b8;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: #64748b;
                    border-radius: 3px;
                }}
                QScrollBar::handle:vertical:pressed {{
                    background: #f59e0b;
                    border-radius: 3px;
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
            """)

    def _init_ui(self):
        self._apply_dialog_theme()

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
        lbl_hw = QLabel("硬件加速:")
        lbl_hw.setFixedWidth(100)
        r_hw.addWidget(lbl_hw)
        self.combo_hw_accel = QComboBox()
        self.combo_hw_accel.setFixedWidth(180)
        setup_solid_combobox(self.combo_hw_accel)
        self.combo_hw_accel.addItem("关闭", "off")
        self.combo_hw_accel.addItem("视口与缩略图", "preview_only")
        self.combo_hw_accel.addItem("全局", "global")
        r_hw.addWidget(self.combo_hw_accel)
        r_hw.addStretch(1)
        l_ren.addLayout(r_hw)

        r_res = QHBoxLayout()
        lbl_res = QLabel("预览分辨率:")
        lbl_res.setFixedWidth(100)
        r_res.addWidget(lbl_res)
        self.combo_preview_res = QComboBox()
        self.combo_preview_res.setFixedWidth(180)
        setup_solid_combobox(self.combo_preview_res)
        self.combo_preview_res.addItem("1080P", 1440)
        self.combo_preview_res.addItem("2K", 2048)
        self.combo_preview_res.addItem("4K", 3840)
        self.combo_preview_res.addItem("全分辨率", 0)
        r_res.addWidget(self.combo_preview_res)
        r_res.addStretch(1)
        l_ren.addLayout(r_res)

        self.chk_hq_preview = QCheckBox("视口平滑与抗锯齿")
        self.chk_hq_preview.setChecked(True)
        l_ren.addWidget(self.chk_hq_preview)

        r_cs = QHBoxLayout()
        lbl_cs = QLabel("工作色彩空间:")
        lbl_cs.setFixedWidth(100)
        r_cs.addWidget(lbl_cs)
        self.combo_default_cs = QComboBox()
        self.combo_default_cs.setFixedWidth(180)
        setup_solid_combobox(self.combo_default_cs)
        self.combo_default_cs.addItems(["sRGB", "Display P3", "Adobe RGB"])
        r_cs.addWidget(self.combo_default_cs)
        r_cs.addStretch(1)
        l_ren.addLayout(r_cs)

        l_ren.addStretch()
        self.tabs.addTab(tab_render, "图像与渲染")

        # Tab 2: 暗房与工作流
        tab_darkroom = QWidget()
        l_dr = QVBoxLayout(tab_darkroom)
        l_dr.setContentsMargins(16, 16, 16, 16)
        l_dr.setSpacing(14)

        r_theme = QHBoxLayout()
        lbl_theme = QLabel("界面主题:")
        lbl_theme.setFixedWidth(100)
        r_theme.addWidget(lbl_theme)
        self.combo_ui_theme = QComboBox()
        self.combo_ui_theme.setFixedWidth(180)
        setup_solid_combobox(self.combo_ui_theme)
        self.combo_ui_theme.addItem("浅色", "light")
        self.combo_ui_theme.addItem("深色", "dark")
        self.combo_ui_theme.addItem("跟随系统", "system")
        r_theme.addWidget(self.combo_ui_theme)
        r_theme.addStretch(1)
        l_dr.addLayout(r_theme)

        r_fmt = QHBoxLayout()
        lbl_fmt = QLabel("默认画幅:")
        lbl_fmt.setFixedWidth(100)
        r_fmt.addWidget(lbl_fmt)
        self.combo_default_fmt = QComboBox()
        self.combo_default_fmt.setFixedWidth(180)
        setup_solid_combobox(self.combo_default_fmt)
        self.combo_default_fmt.addItem("35mm", 35.0)
        self.combo_default_fmt.addItem("16mm", 16.0)
        self.combo_default_fmt.addItem("120 (6x6)", 60.0)
        self.combo_default_fmt.addItem("4x5", 100.0)
        r_fmt.addWidget(self.combo_default_fmt)
        r_fmt.addStretch(1)
        l_dr.addLayout(r_fmt)

        self.chk_restore_last_files = QCheckBox("启动时打开上次工程")
        self.chk_restore_last_files.setChecked(True)
        l_dr.addWidget(self.chk_restore_last_files)

        self.chk_restore_geo = QCheckBox("记住窗口位置与布局")
        self.chk_restore_geo.setChecked(True)
        l_dr.addWidget(self.chk_restore_geo)

        sep_assoc = QFrame()
        sep_assoc.setObjectName("sepAssoc")
        sep_assoc.setFrameShape(QFrame.Shape.NoFrame)
        sep_assoc.setFixedHeight(1)
        l_dr.addWidget(sep_assoc)

        r_assoc = QHBoxLayout()
        r_assoc.setSpacing(10)
        lbl_assoc_info = QLabel("文件关联:")
        lbl_assoc_info.setObjectName("lblAssocTitle")
        r_assoc.addWidget(lbl_assoc_info, 1)

        self.btn_assoc = QPushButton()
        self._update_assoc_button()
        self.btn_assoc.clicked.connect(self._on_toggle_association)
        r_assoc.addWidget(self.btn_assoc)
        l_dr.addLayout(r_assoc)

        # Lens Profiles / 镜头配置文件库
        sep_lens = QFrame()
        sep_lens.setObjectName("sepLens")
        sep_lens.setFrameShape(QFrame.Shape.NoFrame)
        sep_lens.setFixedHeight(1)
        l_dr.addWidget(sep_lens)

        r_lens = QHBoxLayout()
        r_lens.setSpacing(10)
        lbl_lens_info = QLabel("镜头配置文件:")
        lbl_lens_info.setObjectName("lblLensTitle")
        r_lens.addWidget(lbl_lens_info, 1)

        self.btn_import_acr = QPushButton("从 ACR 导入镜头库")
        self.btn_import_acr.setToolTip("检测系统 Adobe Camera Raw 镜头库并克隆到本地用户库 (%APPDATA%/SpektraDarkroom/LensProfiles)")
        self.btn_import_acr.clicked.connect(self._on_import_acr_clicked)
        r_lens.addWidget(self.btn_import_acr)

        self.btn_open_lens_dir = QPushButton("打开镜头库文件夹")
        self.btn_open_lens_dir.setToolTip("在资源管理器中打开用户镜头配置存放目录")
        self.btn_open_lens_dir.clicked.connect(self._on_open_lens_dir_clicked)
        r_lens.addWidget(self.btn_open_lens_dir)
        l_dr.addLayout(r_lens)

        l_dr.addStretch()
        self.tabs.addTab(tab_darkroom, "暗房与工作流")

        # Tab 3: 性能与缓存
        tab_cache = QWidget()
        l_ca = QVBoxLayout(tab_cache)
        l_ca.setContentsMargins(16, 16, 16, 16)
        l_ca.setSpacing(14)

        # 缓存存储目录组合
        v_cache_dir_grp = QVBoxLayout()
        v_cache_dir_grp.setSpacing(5)

        r_dir_hdr = QHBoxLayout()
        r_dir_hdr.setSpacing(10)
        lbl_dir = QLabel("缓存目录:")
        lbl_dir.setFixedWidth(100)
        r_dir_hdr.addWidget(lbl_dir)

        self.btn_change_cache_dir = QPushButton("更改...")
        self.btn_change_cache_dir.setToolTip("选择新的缓存存储目录")
        self.btn_change_cache_dir.clicked.connect(self._on_change_cache_dir_clicked)
        r_dir_hdr.addWidget(self.btn_change_cache_dir)

        self.btn_reset_cache_dir = QPushButton("默认")
        self.btn_reset_cache_dir.setToolTip("恢复为系统默认缓存目录 (%LOCALAPPDATA%/SpektraDarkroom)")
        self.btn_reset_cache_dir.clicked.connect(self._on_reset_cache_dir_clicked)
        r_dir_hdr.addWidget(self.btn_reset_cache_dir)

        r_dir_hdr.addStretch(1)
        v_cache_dir_grp.addLayout(r_dir_hdr)

        # 当前目录单独起一行在下面，纯文本展示，不设样式框
        r_dir_path = QHBoxLayout()
        r_dir_path.setContentsMargins(100, 0, 0, 2)
        self.lbl_cache_dir = QLabel()
        self.lbl_cache_dir.setObjectName("lblCachePath")
        self.lbl_cache_dir.setWordWrap(True)
        self.lbl_cache_dir.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        r_dir_path.addWidget(self.lbl_cache_dir, 1)
        v_cache_dir_grp.addLayout(r_dir_path)

        l_ca.addLayout(v_cache_dir_grp)

        # 兼容性别名
        self.txt_cache_dir = self.lbl_cache_dir

        r_max_size = QHBoxLayout()
        lbl_max_size = QLabel("缓存容量上限:")
        lbl_max_size.setFixedWidth(100)
        r_max_size.addWidget(lbl_max_size)
        self.combo_cache_max_size = QComboBox()
        self.combo_cache_max_size.setFixedWidth(180)
        setup_solid_combobox(self.combo_cache_max_size)
        self.combo_cache_max_size.addItem("500 MB", 0.5)
        self.combo_cache_max_size.addItem("1 GB", 1.0)
        self.combo_cache_max_size.addItem("2 GB", 2.0)
        self.combo_cache_max_size.addItem("5 GB", 5.0)
        self.combo_cache_max_size.addItem("10 GB", 10.0)
        self.combo_cache_max_size.addItem("不限制", 0.0)
        r_max_size.addWidget(self.combo_cache_max_size)
        r_max_size.addStretch(1)
        l_ca.addLayout(r_max_size)

        r_interval = QHBoxLayout()
        lbl_interval = QLabel("自动清理周期:")
        lbl_interval.setFixedWidth(100)
        r_interval.addWidget(lbl_interval)
        self.combo_clean_interval = QComboBox()
        self.combo_clean_interval.setFixedWidth(180)
        setup_solid_combobox(self.combo_clean_interval)
        self.combo_clean_interval.addItem("7 天", 7)
        self.combo_clean_interval.addItem("10 天", 10)
        self.combo_clean_interval.addItem("15 天", 15)
        self.combo_clean_interval.addItem("30 天", 30)
        self.combo_clean_interval.addItem("从不", 0)
        r_interval.addWidget(self.combo_clean_interval)
        r_interval.addStretch(1)
        l_ca.addLayout(r_interval)

        r_cache = QHBoxLayout()
        self.lbl_cache_size = QLabel("LUT 缓存: 计算中...")
        r_cache.addWidget(self.lbl_cache_size, 1)
        self.btn_clean_cache = QPushButton("清理")
        self.btn_clean_cache.clicked.connect(self._clean_lut_cache)
        r_cache.addWidget(self.btn_clean_cache)
        l_ca.addLayout(r_cache)

        r_sess = QHBoxLayout()
        self.lbl_sess_cache_size = QLabel("底片缓存: 计算中...")
        r_sess.addWidget(self.lbl_sess_cache_size, 1)
        self.btn_clean_sess = QPushButton("清理")
        self.btn_clean_sess.clicked.connect(self._clean_sess_cache)
        r_sess.addWidget(self.btn_clean_sess)
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

        btn_save = QPushButton("保存", objectName="btnSave")
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
        self.chk_restore_last_files.setChecked(prefs.get("restore_last_session", prefs.get("restore_last_files", True)))
        self.chk_restore_geo.setChecked(prefs.get("restore_window_state", True))

        def_fmt = prefs.get("default_film_format", 35.0)
        for i in range(self.combo_default_fmt.count()):
            if abs(self.combo_default_fmt.itemData(i) - def_fmt) < 1.0:
                self.combo_default_fmt.setCurrentIndex(i)
                break

        max_gb = float(prefs.get("session_cache_max_gb", 2.0))
        idx_max = -1
        for i in range(self.combo_cache_max_size.count()):
            if abs(float(self.combo_cache_max_size.itemData(i)) - max_gb) < 0.05:
                idx_max = i
                break
        self.combo_cache_max_size.setCurrentIndex(idx_max if idx_max >= 0 else 2)

        interval_days = int(prefs.get("session_cache_clean_interval_days", 7))
        idx_intv = self.combo_clean_interval.findData(interval_days)
        self.combo_clean_interval.setCurrentIndex(idx_intv if idx_intv >= 0 else 0)

        from core.path_utils import get_default_cache_dir
        custom_cache = prefs.get("custom_cache_dir", "")
        self._pending_cache_dir = custom_cache
        current_display = custom_cache if (custom_cache and os.path.isdir(custom_cache)) else get_default_cache_dir()
        self.txt_cache_dir.setText(current_display)
        self.txt_cache_dir.setToolTip(current_display)

        theme_mode = config_manager.get_preference("ui_theme", "system")
        idx_th = self.combo_ui_theme.findData(theme_mode)
        self.combo_ui_theme.setCurrentIndex(idx_th if idx_th >= 0 else 2)

        self.btn_clean_cache.setEnabled(True)
        self.btn_clean_cache.setText("清理")
        self.btn_clean_sess.setEnabled(True)
        self.btn_clean_sess.setText("清理")
        self._refresh_cache_size()

    def _update_assoc_button(self):
        is_dark = theme_manager.is_dark()
        if config_manager.is_sdss_file_associated():
            self.btn_assoc.setText("取消关联")
            self.btn_assoc.setToolTip("清除注册表中的 .sdss 文件关联")
            bg = "#2a1619" if is_dark else "#fee2e2"
            border = "#7f1d1d" if is_dark else "#fca5a5"
            fg = "#f87171" if is_dark else "#b91c1c"
            h_bg = "#3c1a20" if is_dark else "#fecaca"
            p_bg = "#200f13" if is_dark else "#f87171"
            self.btn_assoc.setStyleSheet(f"""
                QPushButton {{
                    background: {bg};
                    border: 1px solid {border};
                    color: {fg};
                    padding: 5px 14px;
                    border-radius: 4px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background: {h_bg};
                    border-color: #ef4444;
                    color: #ef4444;
                }}
                QPushButton:pressed {{
                    background: {p_bg};
                    border-color: #b91c1c;
                    color: #7f1d1d;
                }}
            """)
        else:
            self.btn_assoc.setText("关联 .sdss")
            self.btn_assoc.setToolTip("关联 .sdss 文件，支持双击直接打开")
            bg = "#1e202a" if is_dark else "#ffffff"
            border = "#3d4255" if is_dark else "#cbd5e1"
            fg = "#f59e0b" if is_dark else "#d97706"
            h_bg = "#282c3c" if is_dark else "#fef3c7"
            p_bg = "#14151a" if is_dark else "#fde68a"
            self.btn_assoc.setStyleSheet(f"""
                QPushButton {{
                    background: {bg};
                    border: 1px solid {border};
                    color: {fg};
                    padding: 5px 14px;
                    border-radius: 4px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background: {h_bg};
                    border-color: #f59e0b;
                    color: #f59e0b;
                }}
                QPushButton:pressed {{
                    background: {p_bg};
                    border-color: #d97706;
                    color: #d97706;
                }}
            """)

    def _on_toggle_association(self):
        if config_manager.is_sdss_file_associated():
            config_manager.unregister_sdss_file_association()
        else:
            config_manager.register_sdss_file_association()
        self._update_assoc_button()

    def _on_open_lens_dir_clicked(self):
        from core.lens_correction import get_user_lcp_directory
        d = get_user_lcp_directory()
        try:
            os.startfile(d)
        except Exception as e:
            show_dark_message_box("打开文件夹失败", f"无法打开目录:\n{d}\n{e}", parent=self)

    def _on_change_cache_dir_clicked(self):
        from PySide6.QtWidgets import QFileDialog
        from core.path_utils import get_user_cache_dir
        cur_d = self.lbl_cache_dir.text().strip() or get_user_cache_dir()
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择缓存存储目录",
            cur_d
        )
        if selected:
            selected = os.path.abspath(selected)
            self._pending_cache_dir = selected
            self.lbl_cache_dir.setText(selected)
            self.lbl_cache_dir.setToolTip(selected)
            try:
                os.makedirs(os.path.join(selected, "session_cache"), exist_ok=True)
                os.makedirs(os.path.join(selected, "lut_cache"), exist_ok=True)
            except Exception:
                pass
            self._refresh_cache_size()

    def _on_reset_cache_dir_clicked(self):
        from core.path_utils import get_default_cache_dir
        def_d = get_default_cache_dir()
        self._pending_cache_dir = ""
        self.lbl_cache_dir.setText(def_d)
        self.lbl_cache_dir.setToolTip(def_d)
        self._refresh_cache_size()

    def _on_import_acr_clicked(self):
        from core.lens_correction import find_system_lcp_directories, copy_system_acr_profiles_to_user_dir, get_user_lcp_directory
        acr_dirs = find_system_lcp_directories()
        if not acr_dirs:
            show_darkroom_info(
                self,
                title="未检测到 ACR 镜头库",
                heading="未检测到系统 ACR 镜头库",
                description="未在系统中检测到 Adobe Camera Raw / Lightroom 镜头配置文件目录。\n\n"
                            "提示：您可以免费安装官方 Adobe DNG Converter 即可自动获取数千款官方镜头库，\n"
                            "或直接点击【打开镜头库文件夹】手动放入第三方/厂商提供的 .lcp 文件。",
                ok_text="知道了",
                badge_symbol="!"
            )
            return

        confirmed = show_darkroom_confirm(
            self,
            title="确认导入镜头库",
            heading="导入 Adobe Camera Raw 镜头库",
            description="是否立即将系统 Adobe Camera Raw 镜头配置文件克隆至本软件的本地用户库？\n\n"
                        "导入后即使卸载 Adobe 软件亦可完全独立离线使用该镜头库。",
            confirm_text="开始导入",
            cancel_text="取消",
            badge_symbol="?"
        )
        if not confirmed:
            return

        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.btn_import_acr.setEnabled(False)
        self.btn_import_acr.setText("正在导入...")
        QApplication.processEvents()

        try:
            copied, skipped = copy_system_acr_profiles_to_user_dir()
            QApplication.restoreOverrideCursor()
            self.btn_import_acr.setText("从 ACR 导入镜头库")
            self.btn_import_acr.setEnabled(True)
            show_darkroom_info(
                self,
                title="导入完成",
                heading="镜头库导入成功",
                description=f"镜头库已成功同步至本地用户目录！\n\n"
                            f"• 已导入文件: {copied} 个\n"
                            f"• 已存在跳过: {skipped} 个\n\n"
                            f"存放位置:\n{get_user_lcp_directory()}",
                ok_text="完成",
                badge_symbol="✓"
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.btn_import_acr.setText("从 ACR 导入镜头库")
            self.btn_import_acr.setEnabled(True)
            show_darkroom_info(self, "导入发生异常", "导入镜头库失败", f"导入过程中发生错误:\n{e}", ok_text="确定", badge_symbol="!")

    def _refresh_cache_size(self):
        try:
            from core.path_utils import get_user_lut_cache_dir
            u_cache = get_user_lut_cache_dir()
            b_cache = os.path.join(get_resource_dir(), ".lut_cache")
            total_bytes = 0
            for d in (u_cache, b_cache):
                if os.path.isdir(d):
                    total_bytes += sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
            sz_mb = total_bytes / (1024.0 * 1024.0)
            self.lbl_cache_size.setText(f"LUT 缓存: {sz_mb:.1f} MB")
        except Exception:
            self.lbl_cache_size.setText("LUT 缓存: 0.0 MB")

        try:
            sess_mb = session_cache_manager.get_cache_size_mb()
            self.lbl_sess_cache_size.setText(f"底片缓存: {sess_mb:.1f} MB")
        except Exception:
            self.lbl_sess_cache_size.setText("底片缓存: 0.0 MB")

    def _clean_lut_cache(self):
        try:
            from core.path_utils import get_user_lut_cache_dir
            u_cache = get_user_lut_cache_dir()
            if os.path.isdir(u_cache):
                import shutil
                shutil.rmtree(u_cache, ignore_errors=True)
                os.makedirs(u_cache, exist_ok=True)
            self._refresh_cache_size()
            self.btn_clean_cache.setEnabled(False)
            self.btn_clean_cache.setText("已清理")
        except Exception:
            pass

    def _clean_sess_cache(self):
        try:
            session_cache_manager.clear_cache()
            self._refresh_cache_size()
            self.btn_clean_sess.setEnabled(False)
            self.btn_clean_sess.setText("已清理")
        except Exception:
            pass

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
        restore_val = self.chk_restore_last_files.isChecked()
        prefs["restore_last_session"] = restore_val
        prefs["restore_last_files"] = restore_val
        prefs["restore_window_state"] = self.chk_restore_geo.isChecked()
        prefs["default_film_format"] = float(self.combo_default_fmt.currentData())
        prefs["session_cache_max_gb"] = float(self.combo_cache_max_size.currentData())
        prefs["session_cache_clean_interval_days"] = int(self.combo_clean_interval.currentData())

        if hasattr(self, "_pending_cache_dir"):
            prefs["custom_cache_dir"] = self._pending_cache_dir
            session_cache_manager.reset_cache_dir()

        th_mode = self.combo_ui_theme.currentData()
        prefs["ui_theme"] = th_mode
        config_manager.set_preference("ui_theme", th_mode)
        theme_manager.set_theme_mode(th_mode)

        config_manager.save_config(cfg)
        self.accept()

