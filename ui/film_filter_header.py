"""ui/film_filter_header.py - Responsive Film Section Header with Capsule Search & Category Filter Pills.
Supports dynamic wrapping, 2-row vs 3-row responsive breakpoint, live query filtering, and pill selection.
"""

import os
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize, QEvent
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QButtonGroup, QLayout, QSizePolicy
)
from core.path_utils import get_icon_path
from ui.theme_manager import theme_manager
from ui.window_utils import get_icon


class FlowLayout(QLayout):
    """Standard Qt FlowLayout that wraps child widgets onto new rows when horizontal space is exhausted."""
    def __init__(self, parent=None, margin=0, hspacing=4, vspacing=4):
        super().__init__(parent)
        self.item_list = []
        self.h_spacing = hspacing
        self.v_spacing = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.item_list.append(item)

    def horizontalSpacing(self):
        return self.h_spacing

    def verticalSpacing(self):
        return self.v_spacing

    def count(self):
        return len(self.item_list)

    def itemAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.item_list:
            size = size.expandedTo(item.sizeHint())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only):
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0

        for item in self.item_list:
            wid = item.widget()
            if wid and wid.isHidden():
                continue
            item_size = item.sizeHint()
            next_x = x + item_size.width() + self.h_spacing
            if next_x - self.h_spacing > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + self.v_spacing
                next_x = x + item_size.width() + self.h_spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))

            x = next_x
            line_height = max(line_height, item_size.height())

        return y + line_height - rect.y() + bottom


class FilmFilterHeader(QWidget):
    """Integrated Film Section Header:
    - Row 1: "胶卷型号 (52 卷)" title + capsule search box (in wide mode >= 290px)
    - Row 2: Category pills: 全部, 彩色, 黑白, 负片, 反转片, 电影
    - Responsive: when width < 290px, search box automatically moves to Row 2, pills to Row 3.
    """
    searchChanged = Signal(str)
    categoryChanged = Signal(str)

    CATEGORIES = [
        ("all", "全部"),
        ("color", "彩色"),
        ("bw", "黑白"),
        ("negative", "负片"),
        ("positive", "反转片"),
        ("cine", "电影")
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_cat = "all"
        self._current_query = ""
        self._is_compact_3row = False
        self._total_count = 0
        self._visible_count = 0
        self._init_ui()
        theme_manager.themeChanged.connect(lambda _: self.apply_theme())

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setMinimumHeight(44)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 2, 4, 2)
        self.main_layout.setSpacing(4)

        # Title Label
        self.lbl_title = QLabel("胶卷型号")
        self.lbl_title.setStyleSheet("color: #f59e0b; font-size: 11.5px; font-weight: bold;")

        # Capsule Search Box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("搜索胶卷...")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setFixedHeight(22)

        search_icon = get_icon("search.png", target_size=(13, 13))
        self.search_action = QAction(search_icon, "", self.search_box)
        self.search_box.addAction(self.search_action, QLineEdit.ActionPosition.LeadingPosition)

        self.search_box.setStyleSheet("""
            QLineEdit {
                background: #171821;
                border: 1px solid #282a38;
                border-radius: 11px;
                color: #e2e8f0;
                font-size: 10.5px;
                padding-left: 2px;
                padding-right: 20px;
            }
            QLineEdit:focus {
                border: 1px solid #f59e0b;
                background: #1c1d28;
            }
        """)
        self.search_box.textChanged.connect(self._on_search_changed)

        # Category Buttons Group
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.pills_dict = {}

        # Pills container using FlowLayout
        self.pills_container = QWidget()
        self.pills_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.pills_container.setMinimumHeight(18)
        self.pills_layout = FlowLayout(self.pills_container, margin=0, hspacing=3, vspacing=3)

        for cat_id, cat_name in self.CATEGORIES:
            btn = QPushButton(cat_name)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("cat_id", cat_id)
            btn.setStyleSheet("""
                QPushButton {
                    background: #171821;
                    color: #94a3b8;
                    border: 1px solid #282a38;
                    border-radius: 9px;
                    font-size: 10.5px;
                    font-weight: 500;
                    padding: 0px 9px;
                    min-height: 16px;
                    max-height: 16px;
                }
                QPushButton:hover {
                    color: #f8fafc;
                    border: 1px solid #45485c;
                    background: #1e202c;
                }
                QPushButton:checked {
                    background: rgba(245, 158, 11, 0.16);
                    color: #f59e0b;
                    border: 1px solid #f59e0b;
                    font-weight: bold;
                }
            """)
            if cat_id == "all":
                btn.setChecked(True)
            btn.clicked.connect(lambda checked=False, cid=cat_id: self._on_category_clicked(cid))
            self.btn_group.addButton(btn)
            self.pills_dict[cat_id] = btn
            self.pills_layout.addWidget(btn)

        # Row 1 container
        self.row1_widget = QWidget()
        self.row1_layout = QHBoxLayout(self.row1_widget)
        self.row1_layout.setContentsMargins(0, 0, 0, 0)
        self.row1_layout.setSpacing(6)
        self.row1_layout.addWidget(self.lbl_title)
        self.row1_layout.addStretch()

        # Dedicated Row 2 container for search when in compact mode
        self.row_search_widget = QWidget()
        self.row_search_layout = QHBoxLayout(self.row_search_widget)
        self.row_search_layout.setContentsMargins(0, 0, 0, 0)
        self.row_search_layout.setSpacing(0)
        self.row_search_widget.setVisible(False)

        # Assemble into main_layout
        self.main_layout.addWidget(self.row1_widget)
        self.main_layout.addWidget(self.row_search_widget)
        self.main_layout.addWidget(self.pills_container)

        # Default place search box in row 1
        self.search_box.setFixedWidth(130)
        self.row1_layout.addWidget(self.search_box)
        self.apply_theme()

    def apply_theme(self):
        """Update header search and category pill styles according to active theme."""
        is_dark = theme_manager.is_dark()
        if hasattr(self, "search_action"):
            self.search_action.setIcon(get_icon("search.png", target_size=(13, 13), is_dark=is_dark))
        if is_dark:
            self.lbl_title.setStyleSheet("color: #f59e0b; font-size: 11.5px; font-weight: bold;")
            self.search_box.setStyleSheet("""
                QLineEdit {
                    background: #171821;
                    border: 1px solid #282a38;
                    border-radius: 11px;
                    color: #e2e8f0;
                    font-size: 10.5px;
                    padding-left: 2px;
                    padding-right: 20px;
                }
                QLineEdit:focus {
                    border: 1px solid #f59e0b;
                    background: #1c1d28;
                }
            """)
            pill_style = """
                QPushButton {
                    background: #171821;
                    color: #94a3b8;
                    border: 1px solid #282a38;
                    border-radius: 9px;
                    font-size: 10.5px;
                    font-weight: 500;
                    padding: 0px 9px;
                    min-height: 16px;
                    max-height: 16px;
                }
                QPushButton:hover {
                    color: #f8fafc;
                    border: 1px solid #45485c;
                    background: #1e202c;
                }
                QPushButton:checked {
                    background: rgba(245, 158, 11, 0.16);
                    color: #f59e0b;
                    border: 1px solid #f59e0b;
                    font-weight: bold;
                }
            """
        else:
            self.lbl_title.setStyleSheet("color: #d97706; font-size: 11.5px; font-weight: bold;")
            self.search_box.setStyleSheet("""
                QLineEdit {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 11px;
                    color: #1e293b;
                    font-size: 10.5px;
                    padding-left: 2px;
                    padding-right: 20px;
                }
                QLineEdit:focus {
                    border: 1px solid #f59e0b;
                    background: #ffffff;
                }
            """)
            pill_style = """
                QPushButton {
                    background: #ffffff;
                    color: #64748b;
                    border: 1px solid #d8dce6;
                    border-radius: 9px;
                    font-size: 10.5px;
                    font-weight: 500;
                    padding: 0px 9px;
                    min-height: 16px;
                    max-height: 16px;
                }
                QPushButton:hover {
                    color: #1e293b;
                    border: 1px solid #94a3b8;
                    background: #f1f4f9;
                }
                QPushButton:checked {
                    background: rgba(245, 158, 11, 0.16);
                    color: #d97706;
                    border: 1px solid #f59e0b;
                    font-weight: bold;
                }
            """
        for btn in self.pills_dict.values():
            btn.setStyleSheet(pill_style)

    def _on_search_changed(self, text: str):
        self._current_query = text.strip()
        self.searchChanged.emit(self._current_query)

    def _on_category_clicked(self, cat_id: str):
        if self._current_cat != cat_id:
            self._current_cat = cat_id
            self.categoryChanged.emit(cat_id)

    def set_title_count(self, visible_count: int, total_count: int):
        self._visible_count = visible_count
        self._total_count = total_count
        if visible_count == total_count:
            self.lbl_title.setText(f"胶卷型号 ({total_count} 卷)")
        else:
            self.lbl_title.setText(f"胶卷型号 ({visible_count}/{total_count} 卷)")

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        p_margins = self.main_layout.contentsMargins()
        effective_w = max(50, width - p_margins.left() - p_margins.right())
        pills_h = self.pills_layout.heightForWidth(effective_w)
        r1_h = self.row1_widget.sizeHint().height()
        r_search_h = (self.row_search_widget.sizeHint().height() + self.main_layout.spacing()) if self._is_compact_3row else 0
        spacing = self.main_layout.spacing()
        return r1_h + r_search_h + pills_h + spacing + p_margins.top() + p_margins.bottom()

    def sizeHint(self):
        w = self.width() if self.width() > 50 else 240
        return QSize(w, max(44, self.heightForWidth(w)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        # Responsive breakpoint: < 290px -> 3-row layout, >= 290px -> 2-row layout
        should_compact = (w < 290)
        if should_compact != self._is_compact_3row:
            self._is_compact_3row = should_compact
            if should_compact:
                # Move search box to its own full-width row
                self.row1_layout.removeWidget(self.search_box)
                self.search_box.setMaximumWidth(16777215)
                self.search_box.setMinimumWidth(0)
                self.row_search_layout.addWidget(self.search_box)
                self.row_search_widget.setVisible(True)
            else:
                # Move search box back to row 1
                self.row_search_layout.removeWidget(self.search_box)
                self.row_search_widget.setVisible(False)
                self.search_box.setFixedWidth(130)
                self.row1_layout.addWidget(self.search_box)
            self.pills_container.updateGeometry()
            self.updateGeometry()
