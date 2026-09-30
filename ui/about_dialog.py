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

from path_utils import get_resource_dir
from ui.window_utils import apply_dark_titlebar
from version import get_full_version_info


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
                font-size: 11.5px;
            }
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #353846;
                min-height: 24px;
                border-radius: 3px;
                border: none;
            }
            QScrollBar::handle:vertical:hover {
                background: #f59e0b;
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
        lbl_build.setStyleSheet("color: #7b8092; font-size: 11px;")
        info_layout.addWidget(lbl_build)

        lbl_engine = QLabel(f"加速引擎: {info['engine']}")
        lbl_engine.setStyleSheet("color: #94a3b8; font-size: 11px;")
        info_layout.addWidget(lbl_engine)

        header.addLayout(info_layout, 1)
        layout.addLayout(header)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #262834; background: #262834; max-height: 1px;")
        layout.addWidget(sep)

        # Detailed description browser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        about_html = """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
            <p style="margin-top: 0; color: #f1f5f9; font-weight: 600; font-size: 12px;">
                专业级物理暗房模拟与胶片显影工作站
            </p>
            <p style="margin-bottom: 12px; color: #94a3b8;">
                SpektraDarkroom 基于真实物理化学感光与显影光学模型，还原从相机 RAW 负片曝光、化学显影（DIR抑制层扩散）、彩色放大机滤色曝光到相纸显影定影的完整暗房冲印体验。
            </p>

            <p style="margin-bottom: 4px; color: #e2e8f0; font-weight: bold;">核心特性：</p>
            <ul style="margin-top: 2px; margin-bottom: 12px; padding-left: 18px; color: #94a3b8;">
                <li><b>81 波段全连续光谱模拟</b>：超越传统数码 3D LUT，实现真实的物理感光与显影计算。</li>
                <li><b>28 款经典物理档案</b>：内置 20 款官方胶卷（负片/电影卷/正片）与 8 款放大相纸/放映片。</li>
                <li><b>全物理模拟管线</b>：支持减色法放大滤色镜（CMY）、预闪光、相纸显影衰退与银盐微粒。</li>
                <li><b>GPU 实时加速</b>：基于 Modern OpenGL 与 Vulkan 加速管线，毫秒级实时交互反馈。</li>
                <li><b>非破坏性暗房侧车</b>：编辑参数独立持久化写入 <code>.sdc</code> 文件，原片无损保留。</li>
            </ul>

            <p style="margin-bottom: 4px; color: #e2e8f0; font-weight: bold;">开源基石与致谢：</p>
            <p style="margin-top: 2px; margin-bottom: 12px; color: #94a3b8;">
                • 本软件物理化学仿真算法与光谱感光/相纸密度数据集基于 <b>Andrea Volpato</b> 发起的开源项目 
                <a style="color: #f59e0b; text-decoration: none;" href="https://github.com/andreavolpato/spektrafilm">andreavolpato/spektrafilm</a>。<br>
                • 感谢开源暗房模拟生态及各衍生实现提供的工程灵感与色彩科学研究。
            </p>

            <p style="margin-bottom: 4px; color: #e2e8f0; font-weight: bold;">开源许可协议：</p>
            <p style="margin-top: 2px; margin-bottom: 4px; color: #94a3b8;">
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
