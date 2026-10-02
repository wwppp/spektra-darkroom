"""Parameter Group Card widget for SpektraDarkroom adjustments sidebar.
Permanently flat and tiled layout: clean headers with reset buttons,
and immediate vertical layout for sliders. Zero animation, zero collapsible lag.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)


class ParameterGroupCard(QWidget):
    """Clean, tiled parameter group card with title, reset button,
    and direct slider container. Completely flat and permanent.
    """
    resetClicked = Signal()

    def __init__(self, title, parent=None, is_expanded=True):
        super().__init__(parent)
        self.title_text = title
        self.is_expanded = True
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 6)
        main_layout.setSpacing(0)

        # Card container
        self.card = QFrame()
        self.card.setObjectName("paramGroupCard")
        self.card.setStyleSheet("""
            QFrame#paramGroupCard {
                background: #17181e;
                border: 1px solid #262834;
                border-radius: 6px;
            }
        """)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(8, 6, 8, 8)
        card_layout.setSpacing(6)

        # Header bar
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 0, 2, 2)
        header_layout.setSpacing(6)

        self.title_label = QLabel(self.title_text)
        self.title_label.setStyleSheet("""
            color: #f59e0b;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 0.5px;
        """)
        header_layout.addWidget(self.title_label, 1)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #7b8092;
                font-size: 10.5px;
                padding: 2px 4px;
                border-radius: 3px;
            }
            QPushButton:hover {
                color: #f59e0b;
                background: rgba(245, 158, 11, 0.12);
            }
            QPushButton:pressed {
                color: #d97706;
                background: rgba(245, 158, 11, 0.22);
            }
        """)
        self.reset_btn.clicked.connect(self.resetClicked.emit)
        header_layout.addWidget(self.reset_btn)

        card_layout.addLayout(header_layout)

        # Sliders container
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(4)
        card_layout.addLayout(self.content_layout)

        main_layout.addWidget(self.card)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

    def set_reset_enabled(self, enabled: bool):
        """Enable or disable the section's reset button, graying it out when disabled."""
        self.reset_btn.setEnabled(enabled)
        if enabled:
            self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.reset_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    color: #7b8092;
                    font-size: 10.5px;
                    padding: 2px 4px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    color: #f59e0b;
                    background: rgba(245, 158, 11, 0.12);
                }
                QPushButton:pressed {
                    color: #d97706;
                    background: rgba(245, 158, 11, 0.22);
                }
            """)
        else:
            self.reset_btn.setCursor(Qt.CursorShape.ArrowCursor)
            self.reset_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    color: #383a48;
                    font-size: 10.5px;
                    padding: 2px 4px;
                }
            """)

    def set_expanded(self, expanded: bool, animate: bool = False):
        """No-op compatibility stub (groups are always tiled and expanded)."""
        pass


# Backward compatibility alias
AccordionSection = ParameterGroupCard
