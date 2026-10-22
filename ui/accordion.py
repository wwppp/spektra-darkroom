"""Parameter Group Card widget for SpektraDarkroom adjustments sidebar.
Permanently flat and tiled layout: clean headers with reset buttons,
and immediate vertical layout for sliders. Zero animation, zero collapsible lag.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)
from ui.theme_manager import theme_manager


class ParameterGroupCard(QWidget):
    """Clean, tiled parameter group card with title, reset button,
    and direct slider container. Completely flat and permanent.
    """
    resetClicked = Signal()

    def __init__(self, title, parent=None, is_expanded=True):
        super().__init__(parent)
        self.title_text = title
        self.is_expanded = True
        self._is_reset_enabled = False
        self._init_ui()
        theme_manager.themeChanged.connect(lambda _: self.apply_theme())

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 6)
        main_layout.setSpacing(0)

        # Card container
        self.card = QFrame()
        self.card.setObjectName("paramGroupCard")

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(8, 6, 8, 8)
        card_layout.setSpacing(6)

        # Header bar
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 2)
        header_layout.setSpacing(6)

        self.title_label = QLabel(self.title_text)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        header_layout.addWidget(self.title_label, 1)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.setObjectName("accordionResetBtn")
        self.reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.clicked.connect(self.resetClicked.emit)
        header_layout.addWidget(self.reset_btn)

        card_layout.addLayout(header_layout)

        # Sliders container
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(4)
        card_layout.addLayout(self.content_layout)

        main_layout.addWidget(self.card)
        self.apply_theme()

    def apply_theme(self):
        """Update card border, background, and reset button colors for active theme."""
        is_dark = theme_manager.is_dark()
        if is_dark:
            self.card.setStyleSheet("""
                QFrame#paramGroupCard {
                    background: #17181e;
                    border: 1px solid #262834;
                    border-radius: 6px;
                }
                QFrame#paramGroupCard:disabled {
                    background: #131418;
                    border: 1px solid #1c1d24;
                }
            """)
            self.title_label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    border: none;
                    color: #f59e0b;
                    font-size: 11px;
                    font-weight: bold;
                    letter-spacing: 0.5px;
                }
                QLabel:disabled {
                    background: transparent;
                    border: none;
                    color: #52525b;
                }
            """)
        else:
            self.card.setStyleSheet("""
                QFrame#paramGroupCard {
                    background: #ffffff;
                    border: 1px solid #d8dce6;
                    border-radius: 6px;
                }
                QFrame#paramGroupCard:disabled {
                    background: #f1f4f9;
                    border: 1px solid #e2e8f0;
                }
            """)
            self.title_label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    border: none;
                    color: #d97706;
                    font-size: 11px;
                    font-weight: bold;
                    letter-spacing: 0.5px;
                }
                QLabel:disabled {
                    background: transparent;
                    border: none;
                    color: #94a3b8;
                }
            """)
        self.set_reset_enabled(self._is_reset_enabled)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

    def set_reset_enabled(self, enabled: bool):
        """Enable or disable the section's reset button, lighting it up when modified."""
        self._is_reset_enabled = enabled
        self.reset_btn.setEnabled(enabled)
        is_dark = theme_manager.is_dark()
        tt_ss = theme_manager.get_tooltip_stylesheet()
        if enabled:
            self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.reset_btn.setStyleSheet(f"""
                #accordionResetBtn {{
                    background: transparent;
                    border: none;
                    color: #f59e0b !important;
                    font-size: 10.5px;
                    font-weight: 600;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                #accordionResetBtn:hover {{
                    color: #fde68a !important;
                    background: rgba(245, 158, 11, 0.15);
                }}
                #accordionResetBtn:pressed {{
                    color: #d97706 !important;
                    background: rgba(245, 158, 11, 0.25);
                }}
                {tt_ss}
            """)
        else:
            self.reset_btn.setCursor(Qt.CursorShape.ArrowCursor)
            dim_color = "#383a48" if is_dark else "#94a3b8"
            self.reset_btn.setStyleSheet(f"""
                #accordionResetBtn {{
                    background: transparent;
                    border: none;
                    color: {dim_color} !important;
                    font-size: 10.5px;
                    font-weight: normal;
                    padding: 2px 5px;
                }}
                {tt_ss}
            """)

    def set_expanded(self, expanded: bool, animate: bool = False):
        """No-op compatibility stub (groups are always tiled and expanded)."""
        pass


# Backward compatibility alias
AccordionSection = ParameterGroupCard
