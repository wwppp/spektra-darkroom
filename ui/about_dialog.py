"""
about_dialog.py - Standard About Dialog for SpektraDarkroom
Displays formal software identity, technical features, upstream origin & credits, and licenses.
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTextBrowser
)

from core.path_utils import get_resource_dir
from ui.window_utils import apply_dark_titlebar
from ui.theme_manager import theme_manager
from core.version import get_full_version_info


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于 SpektraDarkroom")
        self.setFixedSize(560, 560)
        self.setModal(True)

        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._init_ui()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def _init_ui(self):
        info = get_full_version_info()
        is_dark = theme_manager.is_dark()

        if is_dark:
            self.setStyleSheet("""
                QDialog {
                    background: #14151a;
                    color: #e2e8f0;
                }
                QLabel {
                    border: none;
                    background: transparent;
                }
                QTextBrowser {
                    background: #181920;
                    border: 1px solid #282a36;
                    border-radius: 6px;
                    padding: 12px 14px;
                    color: #cbd5e1;
                    font-size: 11px;
                }
                QScrollBar:vertical {
                    width: 6px;
                    background: transparent;
                    margin: 0px;
                    border: none;
                }
                QScrollBar::handle:vertical {
                    background: #3f4350;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical:hover {
                    background: #cbd5e1;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    border: none;
                }
                QPushButton#btn_close {
                    background: #20222b;
                    color: #e2e8f0;
                    border: 1px solid #333647;
                    border-radius: 4px;
                    padding: 6px 22px;
                    font-size: 12px;
                    font-weight: 500;
                    min-width: 76px;
                }
                QPushButton#btn_close:hover {
                    background: #2a2d3a;
                    border-color: #f59e0b;
                    color: #f59e0b;
                }
                QPushButton#btn_close:pressed {
                    background: #14151a;
                    color: #d97706;
                    border-color: #d97706;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background: #ebedf2;
                    color: #1e293b;
                }
                QLabel {
                    border: none;
                    background: transparent;
                }
                QTextBrowser {
                    background: #ffffff;
                    border: 1px solid #d5d9e4;
                    border-radius: 6px;
                    padding: 12px 14px;
                    color: #334155;
                    font-size: 11px;
                }
                QScrollBar:vertical {
                    width: 6px;
                    background: transparent;
                    margin: 0px;
                    border: none;
                }
                QScrollBar::handle:vertical {
                    background: #94a3b8;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical:hover {
                    background: #64748b;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    border: none;
                }
                QPushButton#btn_close {
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 6px 22px;
                    font-size: 12px;
                    font-weight: 500;
                    min-width: 76px;
                }
                QPushButton#btn_close:hover {
                    background: #f8fafc;
                    border-color: #f59e0b;
                    color: #d97706;
                }
                QPushButton#btn_close:pressed {
                    background: #e2e8f0;
                    color: #b45309;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(14)

        # Header area: App Icon + App Name & Version
        header = QHBoxLayout()
        header.setSpacing(14)

        lbl_icon = QLabel()
        lbl_icon.setFixedSize(48, 48)
        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            pix = QPixmap(icon_path).scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_icon.setPixmap(pix)
        header.addWidget(lbl_icon, 0, Qt.AlignmentFlag.AlignTop)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(3)

        lbl_name = QLabel(f"SpektraDarkroom v{info['version']}")
        lbl_name.setStyleSheet("color: #f59e0b; font-size: 16px; font-weight: bold; letter-spacing: 0.5px;")
        info_layout.addWidget(lbl_name)

        lbl_build = QLabel(f"版本: {info['version']} (Build {info['build']})  |  发布时间: {info['release_date']}")
        lbl_build.setStyleSheet("color: #64748b; font-size: 11px;")
        info_layout.addWidget(lbl_build)

        lbl_engine = QLabel(f"加速引擎: {info['engine']}")
        lbl_engine.setStyleSheet("color: #64748b; font-size: 11px;" if not is_dark else "color: #94a3b8; font-size: 11px;")
        info_layout.addWidget(lbl_engine)

        header.addLayout(info_layout, 1)
        layout.addLayout(header)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep_col = "#262834" if is_dark else "#d5d9e4"
        sep.setStyleSheet(f"color: {sep_col}; background: {sep_col}; max-height: 1px;")
        layout.addWidget(sep)

        # Detailed description browser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        col_title = "#f1f5f9" if is_dark else "#0f172a"
        col_text = "#94a3b8" if is_dark else "#475569"
        col_bold = "#e2e8f0" if is_dark else "#1e293b"

        about_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.5;">
            <p style="margin-top: 0; margin-bottom: 8px; color: {col_title}; font-weight: 600; font-size: 12px;">
                专业级 35mm 胶片物理显影与放大暗房模拟工作站
            </p>
            <p style="margin-bottom: 12px; color: {col_text};">
                基于连续光谱模拟与光化学感光/显影模型，完整还原相机 RAW 负片曝光、DIR 发色剂抑制扩散、彩色放大机减色法滤色曝光、相纸光化学显影及真实胶片染料云微粒质感。
            </p>

            <p style="margin-bottom: 4px; color: {col_bold}; font-weight: bold;">核心特性：</p>
            <ul style="margin-top: 2px; margin-bottom: 12px; padding-left: 18px; color: {col_text};">
                <li><b>81 波段连续光谱物理模拟</b>：基于真实物理光谱测定，彻底告别普通 3D LUT 断阶。</li>
                <li><b>61 款经典物理档案</b>：内置 52 款经典负片/电影卷/反转片预设与 9 款相纸/拷贝片档案。</li>
                <li><b>物理暗房全模拟管线</b>：支持 CMY 减色滤色曝光、预闪光、相纸显影衰退与光晕扩散。</li>
                <li><b>真实染料云团微粒 (CLT)</b>：视口积分衰减模型，模拟微观 35mm 银盐染料云团真实颗粒。</li>
                <li><b>Modern OpenGL / Vulkan 硬件加速</b>：4K 实时交互反馈，支持无级缩放与快速对比。</li>
                <li><b>非破坏性暗房伴侣 (.sdc)</b>：调色扫描参数独立持久化写入，原片素材无损保留。</li>
            </ul>

            <p style="margin-bottom: 4px; color: {col_bold}; font-weight: bold;">致谢与开源基石：</p>
            <p style="margin-top: 2px; margin-bottom: 12px; color: {col_text};">
                • <b><a style="color: #f59e0b; text-decoration: none;" href="https://github.com/andreavolpato/spektrafilm">Andrea Volpato (spektrafilm)</a></b>：光谱感光乳剂模型、发色剂化学显影反应及感光测定数据集基石。<br>
                • <b><a style="color: #f59e0b; text-decoration: none;" href="https://github.com/fotufilm/fotufilm-engine">fotufilm-engine</a></b>：彩色染料云团（Dye Clouds）微粒算法模型与经典胶卷光谱配置扩展。
            </p>

            <p style="margin-bottom: 4px; color: {col_bold}; font-weight: bold;">开源许可协议：</p>
            <p style="margin-top: 2px; margin-bottom: 2px; color: {col_text};">
                • 代码部分遵循 <b>GNU General Public License v3.0 (GPLv3)</b> 开源协议。<br>
                • 物理配置文件与测定数据遵循 <b>CC BY-SA 4.0</b> 知识共享协议。
            </p>
        </div>
        """
        browser.setHtml(about_html)
        layout.addWidget(browser, 1)

        # Bottom Button Bar
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_close = QPushButton("确定")
        btn_close.setObjectName("btn_close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)
