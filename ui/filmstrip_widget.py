import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QRect, QRectF, QPoint, QEvent, QTimer, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen, QFont, QFontMetrics, QKeySequence, QIcon, QLinearGradient, QCursor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QMenu, QDialog, QTextEdit, QSizePolicy
)
from core import sdc_manager
from core import session_cache_manager
from core.session_cache_manager import is_dummy_or_corrupted_thumbnail, is_valid_thumbnail
from core.path_utils import get_icon_path, get_resource_dir
from ui.window_utils import apply_dark_titlebar, get_darkroom_menu_style, get_icon
from ui.theme_manager import theme_manager
from ui.smooth_scroll import SmoothScrollArea


_CROP_BADGE_PM = None

def _get_crop_badge_pixmap() -> QPixmap:
    global _CROP_BADGE_PM
    if _CROP_BADGE_PM is None:
        p = os.path.join(get_resource_dir(), "icons", "crop.png")
        if os.path.exists(p):
            _CROP_BADGE_PM = QPixmap(p)
        else:
            _CROP_BADGE_PM = QPixmap()
    return _CROP_BADGE_PM


class DuplicateFilesDialog(QDialog):
    """Dialog showing duplicate files that were skipped during import."""
    def __init__(self, duplicate_names, parent=None):
        super().__init__(parent)
        self.duplicate_names = list(duplicate_names or [])
        self.setWindowTitle("重复底片过滤提示")
        self.setFixedSize(540, 350)
        icon_path = os.path.join(get_resource_dir(), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        is_dark = theme_manager.is_dark()
        if is_dark:
            self.setStyleSheet("""
                QDialog {
                    background: #14151a;
                    border: 1px solid #2b2e3b;
                    border-radius: 8px;
                }
                QLabel {
                    color: #e2e8f0;
                    font-family: "Microsoft YaHei UI", sans-serif;
                }
                QTextEdit {
                    background: #101115;
                    color: #cbd5e1;
                    border: 1px solid #282b37;
                    border-radius: 6px;
                    padding: 10px;
                    font-family: Consolas, "Courier New", monospace;
                    font-size: 11px;
                    line-height: 1.4;
                }
                QPushButton {
                    background: #f59e0b;
                    color: #111111;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 24px;
                    font-size: 12px;
                    min-width: 72px;
                }
                QPushButton:hover {
                    background: #fbbf24;
                }
                QPushButton:pressed {
                    background: #d97706;
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
                    font-family: "Microsoft YaHei UI", sans-serif;
                }
                QTextEdit {
                    background: #ffffff;
                    color: #1e293b;
                    border: 1px solid #d5d9e4;
                    border-radius: 6px;
                    padding: 10px;
                    font-family: Consolas, "Courier New", monospace;
                    font-size: 11px;
                    line-height: 1.4;
                }
                QPushButton {
                    background: #f59e0b;
                    color: #111111;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 24px;
                    font-size: 12px;
                    min-width: 72px;
                }
                QPushButton:hover {
                    background: #fbbf24;
                }
                QPushButton:pressed {
                    background: #d97706;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        # Header with icon and count
        hdr = QHBoxLayout()
        hdr.setSpacing(10)
        badge = QLabel("ℹ")
        badge.setFixedSize(28, 28)
        badge_bg = "#261f14" if is_dark else "#fef3c7"
        badge.setStyleSheet(f"""
            background: {badge_bg};
            border: 1px solid #f59e0b;
            border-radius: 6px;
            color: #f59e0b;
            font-size: 14px;
            font-weight: bold;
        """)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.addWidget(badge)

        count = len(self.duplicate_names)
        head_col = "#f8fafc" if is_dark else "#0f172a"
        lbl_head = QLabel(f"已自动跳过 {count} 个已在暗房底片库中的重复文件：")
        lbl_head.setStyleSheet(f"color: {head_col}; font-weight: bold; font-size: 13px;")
        hdr.addWidget(lbl_head, 1)
        layout.addLayout(hdr)

        # Files list
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setPlainText("\n".join(self.duplicate_names))
        layout.addWidget(self.txt, 1)

        # Footer
        btn_box = QHBoxLayout()
        lbl_hint = QLabel("底片库已保留原有底片，不会重复添加。")
        lbl_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        btn_box.addWidget(lbl_hint)
        btn_box.addStretch()

        btn_ok = QPushButton("确定")
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.clicked.connect(self.accept)
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)


class FilmstripItemWidget(QFrame):
    """Compact 35mm film negative frame:
    - 35mm Sprocket holes on top and bottom
    - Centered photo thumbnail keeping true original aspect ratio
    - Translucent filename overlay at the bottom
    - Multi-select border highlight
    """
    clicked = Signal(str, object)  # photo_id, mouse event for modifier keys
    removeRequested = Signal(str)
    contextMenuRequested = Signal(str, QPoint)
    ratingChanged = Signal(str, int)  # photo_id, rating

    def __init__(self, photo_data, is_active=False, is_selected=False, parent=None):
        super().__init__(parent)
        self.photo_id = photo_data.get("id", "")
        self.filename = photo_data.get("filename") or (os.path.basename(photo_data.get("path", "image")) if photo_data.get("path") else "image")
        from core.path_utils import normalize_path
        self.file_path = normalize_path(photo_data.get("path", ""))
        self.is_active = is_active
        self.is_selected = is_selected
        self.photo_data = photo_data
        self.rating = int(photo_data.get("rating", 0) or 0)

        self.setFixedSize(76, 82)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_tooltip()
        self._apply_theme_style()
        theme_manager.themeChanged.connect(self._apply_theme_style)

        self._base_pixmap = None
        self._pixmap = None
        thumb_rgb = photo_data.get("thumbnail_rgb")

        if is_dummy_or_corrupted_thumbnail(thumb_rgb):
            raw_rgb = photo_data.get("raw_thumbnail_rgb")
            if is_valid_thumbnail(raw_rgb):
                thumb_rgb = raw_rgb
            else:
                raw_rgb = photo_data.get("raw_thumbnail")
                if is_valid_thumbnail(raw_rgb):
                    thumb_rgb = raw_rgb
                elif photo_data.get("path") and os.path.isfile(photo_data["path"]):
                    try:
                        extracted = session_cache_manager.extract_fast_thumbnail(photo_data["path"])
                        if is_valid_thumbnail(extracted):
                            thumb_rgb = extracted
                            photo_data["thumbnail_rgb"] = extracted
                            photo_data["raw_thumbnail_rgb"] = extracted
                    except Exception:
                        pass

        if thumb_rgb is not None and isinstance(thumb_rgb, np.ndarray) and thumb_rgb.size > 0:
            thumb_c = np.ascontiguousarray(thumb_rgb, dtype=np.uint8)
            h, w, c = thumb_c.shape
            qimg = QImage(thumb_c.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
            self._base_pixmap = QPixmap.fromImage(qimg)
            my = 82 * 0.14
            tw = max(10, int(76 - 8))
            th = max(10, int(82 - my * 2.0))
            self._pixmap = self._base_pixmap.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def set_item_size(self, target_w, target_h):
        self.setFixedSize(target_w, target_h)
        if (self._base_pixmap is None or self._base_pixmap.isNull()) and self.photo_data:
            thumb_rgb = self.photo_data.get("thumbnail_rgb") if self.photo_data.get("thumbnail_rgb") is not None else self.photo_data.get("raw_thumbnail_rgb")
            if not is_valid_thumbnail(thumb_rgb) and self.photo_data.get("path") and os.path.isfile(self.photo_data["path"]):
                try:
                    extracted = session_cache_manager.extract_fast_thumbnail(self.photo_data["path"])
                    if is_valid_thumbnail(extracted):
                        thumb_rgb = extracted
                        self.photo_data["thumbnail_rgb"] = extracted
                        self.photo_data["raw_thumbnail_rgb"] = extracted
                except Exception:
                    pass
            if thumb_rgb is not None and isinstance(thumb_rgb, np.ndarray) and thumb_rgb.size > 0:
                thumb_c = np.ascontiguousarray(thumb_rgb, dtype=np.uint8)
                h, w, c = thumb_c.shape
                qimg = QImage(thumb_c.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
                self._base_pixmap = QPixmap.fromImage(qimg)

        if self._base_pixmap and not self._base_pixmap.isNull():
            my = target_h * 0.14
            tw = max(10, int(target_w - 8))
            th = max(10, int(target_h - my * 2.0))
            self._pixmap = self._base_pixmap.scaled(
                tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        self.update()

    def set_thumbnail(self, pixmap):
        """Update thumbnail pixmap dynamically in real-time."""
        if pixmap and not pixmap.isNull():
            self._base_pixmap = pixmap
            my = self.height() * 0.14
            tw = max(10, int(self.width() - 8))
            th = max(10, int(self.height() - my * 2.0))
            self._pixmap = self._base_pixmap.scaled(
                tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.update()

    def set_active(self, is_active):
        if self.is_active != is_active:
            self.is_active = is_active
            self.update()

    def set_selected(self, is_selected):
        if self.is_selected != is_selected:
            self.is_selected = is_selected
            self.update()

    def set_rating(self, rating: int):
        clean_r = max(0, min(5, int(rating or 0)))
        if self.rating != clean_r:
            self.rating = clean_r
            self.photo_data["rating"] = clean_r
            self.update()

    def update_photo_data(self, photo_data):
        """Update photo metadata and ensure tooltip is synchronized."""
        if not photo_data:
            return
        self.photo_data = photo_data
        if "rating" in photo_data:
            self.rating = int(photo_data.get("rating", 0) or 0)
        if photo_data.get("filename"):
            self.filename = photo_data["filename"]
        if photo_data.get("path"):
            from core.path_utils import normalize_path
            self.file_path = normalize_path(photo_data["path"])
        self.update_tooltip()

    def _apply_theme_style(self, _theme=None):
        self.setStyleSheet(f"FilmstripItemWidget {{ background: transparent; border: none; }}\n{theme_manager.get_tooltip_stylesheet()}")

    def update_tooltip(self):
        """Dynamically build tooltip ensuring resolution is accurate and never 0 x 0."""
        w = int(self.photo_data.get("width", 0))
        h = int(self.photo_data.get("height", 0))
        if (w <= 0 or h <= 0) and self.file_path and os.path.isfile(self.file_path):
            try:
                from core import session_cache_manager
                dw, dh = session_cache_manager.get_image_dimensions(self.file_path)
                if dw > 0 and dh > 0:
                    w, h = dw, dh
                    self.photo_data["width"] = w
                    self.photo_data["height"] = h
            except Exception:
                pass
        self.setToolTip(f"{self.filename}\n路径: {self.file_path}\n尺寸: {w} × {h}")

    def event(self, e):
        if e.type() == QEvent.Type.ToolTip:
            self.update_tooltip()
        return super().event(e)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            h = float(self.height())
            can_fit_stars = (h >= 50)
            if can_fit_stars:
                margin_y = h * 0.14
                thumb_rect = QRectF(4.0, margin_y, self.width() - 8.0, h - margin_y * 2.0)
                overlay_h = max(22.0, min(28.0, h * 0.35))
                overlay_top = thumb_rect.bottom() - overlay_h
                fn_h = overlay_h * 0.48
                star_row_top = overlay_top + fn_h
                click_x = event.position().x()
                click_y = event.position().y()
                if overlay_top <= click_y <= thumb_rect.bottom():
                    start_x = thumb_rect.x() + 3.0
                    star_w = max(7.0, min(9.0, (thumb_rect.width() - 6.0) / 5.2))
                    star_area_end = start_x + 5.0 * star_w
                    if star_row_top - 2.0 <= click_y <= thumb_rect.bottom() and start_x <= click_x <= star_area_end + 4.0:
                        star_idx = int((click_x - start_x) / star_w) + 1
                        star_idx = max(1, min(5, star_idx))
                        new_r = 0 if star_idx == self.rating else star_idx
                        self.set_rating(new_r)
                        self.ratingChanged.emit(self.photo_id, new_r)
                        event.accept()
                        return
            self.clicked.emit(self.photo_id, event)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.contextMenuRequested.emit(self.photo_id, event.globalPosition().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        is_dark = theme_manager.is_dark()

        # 1. Base 35mm Film Negative Border & Background (真实胶片色深褐色 / Deep Authentic Film Acetate Base)
        # Real C-41 / silver halide color negative film has an iconic deep dark amber-brown / espresso base
        if self.is_active:
            # Active photo: deep luminous amber-brown illuminated by light table
            bg_grad = QLinearGradient(0, 0, 0, h)
            bg_grad.setColorAt(0.0, QColor(48, 26, 14))      # #301a0e
            bg_grad.setColorAt(0.45, QColor(34, 18, 10))     # #22120a
            bg_grad.setColorAt(1.0, QColor(24, 13, 7))       # #180d07
            border_color = QColor("#f59e0b")                 # Signature brand amber
            border_width = 2.0
            sprocket_edge_color = QColor(180, 83, 9, 180)
        elif self.is_selected:
            # Multi-selected photo: deep amber-brown with amber highlight border
            bg_grad = QLinearGradient(0, 0, 0, h)
            bg_grad.setColorAt(0.0, QColor(42, 22, 12))      # #2a160c
            bg_grad.setColorAt(0.45, QColor(30, 16, 8))      # #1e1008
            bg_grad.setColorAt(1.0, QColor(20, 11, 6))       # #140b06
            border_color = QColor(245, 158, 11, 230)
            border_width = 1.8
            sprocket_edge_color = QColor(180, 83, 9, 140)
        else:
            # Normal photo: classic deep dark amber-brown (深邃经典胶片深褐色)
            bg_grad = QLinearGradient(0, 0, 0, h)
            bg_grad.setColorAt(0.0, QColor(34, 18, 10))      # #22120a Deep dark amber sheen
            bg_grad.setColorAt(0.45, QColor(24, 13, 7))      # #180d07 Rich espresso film base
            bg_grad.setColorAt(1.0, QColor(17, 9, 5))        # #110905 Dense negative base
            border_color = QColor(48, 26, 14)                # #301a0e Subtle dark brown rim
            border_width = 1.0
            sprocket_edge_color = QColor(14, 7, 3, 220) if is_dark else QColor(38, 20, 10, 180)

        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), 3.5, 3.5)

        # 2. Draw 35mm Sprocket Holes punched into the brown film base
        # Holes reveal the underlying recessed track depth (dark or light groove)
        if is_dark:
            sprocket_brush = QBrush(QColor(8, 9, 13))
        else:
            sprocket_brush = QBrush(QColor(220, 224, 232))

        painter.setPen(QPen(sprocket_edge_color, 0.7))
        painter.setBrush(sprocket_brush)

        margin_y = h * 0.14
        hole_h = margin_y * 0.60
        hole_w = hole_h * 1.30
        hole_r = max(1.0, hole_h * 0.22)
        top_hy = (margin_y - hole_h) / 2.0
        bot_hy = h - margin_y + (margin_y - hole_h) / 2.0

        hole_pitch = max(12.0, hole_w * 2.2)
        num_holes = max(2, int(round((w - 10.0) / hole_pitch)))
        spacing = (w - 8.0) / float(num_holes)
        for i in range(num_holes):
            hx = 4.0 + i * spacing + (spacing - hole_w) / 2.0
            painter.drawRoundedRect(QRectF(hx, top_hy, hole_w, hole_h), hole_r, hole_r)
            painter.drawRoundedRect(QRectF(hx, bot_hy, hole_w, hole_h), hole_r, hole_r)

        # 3. Draw Thumbnail strictly inside leader bands with camera aperture frame border
        thumb_rect = QRectF(4.0, margin_y, w - 8.0, h - margin_y * 2.0)
        painter.setClipRect(thumb_rect)
        painter.fillRect(thumb_rect, QColor(10, 8, 6))
        if self._pixmap and not self._pixmap.isNull():
            px = thumb_rect.x() + (thumb_rect.width() - self._pixmap.width()) / 2.0
            py = thumb_rect.y() + (thumb_rect.height() - self._pixmap.height()) / 2.0
            painter.drawPixmap(int(px), int(py), self._pixmap)

        # 4. Draw camera aperture frame border around the exposed thumbnail
        painter.setClipping(False)
        painter.setPen(QPen(QColor(14, 7, 4, 220), 0.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(thumb_rect)

        # 5. Draw Filename & Star Rating Overlay at the bottom
        can_fit_stars = (h >= 50)
        if can_fit_stars:
            overlay_h = max(22.0, min(28.0, h * 0.35))
        else:
            overlay_h = max(13.0, min(16.0, h * 0.20))

        overlay_rect = QRectF(thumb_rect.x(), thumb_rect.bottom() - overlay_h, thumb_rect.width(), overlay_h)
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 115))

        if can_fit_stars:
            # Row 1: Filename (slightly smaller font, left-aligned with 3px margin)
            font = painter.font()
            fn_font_size = max(6.5, min(7.5, h * 0.085))
            font.setPointSizeF(fn_font_size)
            painter.setFont(font)
            fm = QFontMetrics(font)

            painter.setPen(QPen(QColor(230, 230, 230)))
            fn_h = overlay_h * 0.48
            fn_rect = QRectF(overlay_rect.x() + 3.0, overlay_rect.top() + 1.0, overlay_rect.width() - 6.0, fn_h)
            elided = fm.elidedText(self.filename, Qt.TextElideMode.ElideMiddle, int(fn_rect.width()))
            painter.drawText(fn_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)

            # Row 2: 5 Stars (Left-aligned with filename)
            star_font = QFont("Segoe UI Symbol")
            star_font.setStyleHint(QFont.StyleHint.SansSerif)
            star_font.setPointSizeF(max(6.5, min(8.0, h * 0.09)))
            painter.setFont(star_font)

            star_w = max(7.0, min(9.0, (overlay_rect.width() - 6.0) / 5.2))
            start_x = overlay_rect.x() + 3.0
            star_y = overlay_rect.top() + fn_h
            star_h = overlay_rect.bottom() - star_y
            active_col = QColor("#f59e0b")
            empty_col = QColor(100, 110, 125, 170) if is_dark else QColor(148, 163, 184, 190)

            for i in range(5):
                s_rect = QRectF(start_x + i * star_w, star_y, star_w, star_h)
                if i < self.rating:
                    painter.setPen(active_col)
                else:
                    painter.setPen(empty_col)
                painter.drawText(s_rect, Qt.AlignmentFlag.AlignCenter, "★")
        else:
            # Compact mode (h < 50): single line filename only (stars automatically hidden)
            font = painter.font()
            font_size = max(7, min(9, int(round(h * 0.09))))
            font.setPointSize(font_size)
            painter.setFont(font)
            fm = QFontMetrics(font)

            painter.setPen(QPen(QColor(230, 230, 230)))
            elided = fm.elidedText(self.filename, Qt.TextElideMode.ElideMiddle, int(overlay_rect.width() - 4))
            painter.drawText(overlay_rect, Qt.AlignmentFlag.AlignCenter, elided)

        # 6. Draw Edited/Modified indicator badge (Item 1)
        if self.photo_data.get("is_dirty") or self.photo_data.get("is_edited"):
            badge_sz = max(5.0, min(9.0, h * 0.07))
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1.0))
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.drawEllipse(QRectF(margin_y * 0.6, margin_y * 1.1, badge_sz, badge_sz))

        # 6.5. Draw Crop indicator badge at mirrored top-right corner
        if self.has_crop():
            c_sz = max(9.0, min(13.0, h * 0.11))
            cx = w - margin_y * 0.6 - c_sz
            cy = margin_y * 1.1
            badge_rect = QRectF(cx, cy, c_sz, c_sz)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
            painter.drawRoundedRect(badge_rect.adjusted(-1.5, -1.5, 1.5, 1.5), 2.5, 2.5)

            pm = _get_crop_badge_pixmap()
            if not pm.isNull():
                painter.drawPixmap(badge_rect.toRect(), pm)

        # 7. Top layer: High-visibility Selection & Active Focus Border
        if self.is_active:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#f59e0b"), 2.2))
            painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), 3.5, 3.5)
        elif self.is_selected:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(245, 158, 11, 230), 2.0))
            painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), 3.5, 3.5)

    def has_crop(self):
        crop = self.photo_data.get("crop")
        if not crop:
            params = self.photo_data.get("params")
            if isinstance(params, dict):
                crop = params.get("crop")
        if isinstance(crop, dict):
            if crop.get("enabled", False):
                return True
            w = float(crop.get("w", 1.0))
            h = float(crop.get("h", 1.0))
            x = float(crop.get("x", 0.0))
            y = float(crop.get("y", 0.0))
            rot = int(crop.get("rot_90", 0)) % 360
            angle = float(crop.get("angle", 0.0))
            return (abs(w - 1.0) > 0.001 or abs(h - 1.0) > 0.001 or
                    abs(x) > 0.001 or abs(y) > 0.001 or rot != 0 or abs(angle) > 0.01 or
                    crop.get("flip_h") or crop.get("flip_v"))
        elif isinstance(crop, bool):
            return crop
        return False

    def set_crop(self, crop_data):
        if isinstance(crop_data, dict):
            self.photo_data["crop"] = dict(crop_data)
            if "params" in self.photo_data and isinstance(self.photo_data["params"], dict):
                self.photo_data["params"]["crop"] = dict(crop_data)
        elif isinstance(crop_data, bool):
            if not crop_data:
                def_c = {
                    "enabled": False, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
                    "angle": 0.0, "rot_90": 0, "flip_h": False, "flip_v": False,
                    "aspect_ratio": "original", "is_portrait": False
                }
                self.photo_data["crop"] = def_c
                if "params" in self.photo_data and isinstance(self.photo_data["params"], dict):
                    self.photo_data["params"]["crop"] = def_c
            else:
                if "crop" not in self.photo_data or not isinstance(self.photo_data["crop"], dict):
                    self.photo_data["crop"] = {"enabled": True}
                else:
                    self.photo_data["crop"]["enabled"] = True
        self.update()

    @property
    def is_dirty(self):
        return bool(self.photo_data.get("is_dirty"))

    def set_dirty(self, is_dirty=True):
        self.photo_data["is_dirty"] = is_dirty
        if is_dirty:
            self.photo_data["is_edited"] = True
        else:
            p_path = self.photo_data.get("path")
            self.photo_data["is_edited"] = bool(p_path and sdc_manager.has_sdc(p_path))
        self.update()


class FilmTrackContainer(QFrame):
    """Concave physical film groove / guide-rail channel with rounded corners.
    Renders an authentic darkroom recessed track (长条凹形胶片条槽路) with subtle 3D sunken depth.
    Uses WA_TranslucentBackground so 4 outer corners are 100% transparent without rectangular artifacts.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        r = 6.0

        is_dark = theme_manager.is_dark()
        if is_dark:
            # Inset recessed groove gradient: sunken dark charcoal/black
            bg_grad = QLinearGradient(0, 0, 0, h)
            bg_grad.setColorAt(0.0, QColor(6, 7, 10))
            bg_grad.setColorAt(0.18, QColor(8, 9, 13))
            bg_grad.setColorAt(0.85, QColor(10, 11, 16))
            bg_grad.setColorAt(1.0, QColor(13, 14, 20))
            painter.setBrush(QBrush(bg_grad))

            # Outer groove rim border
            pen = QPen(QColor(28, 30, 40), 1.0)
            painter.setPen(pen)
            painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), r, r)

            # Subtle inner top shadow to accentuate the concave depth
            shadow_pen = QPen(QColor(0, 0, 0, 140), 1.0)
            painter.setPen(shadow_pen)
            painter.drawRoundedRect(QRectF(1.5, 1.5, w - 3.0, h - 3.0), r - 0.5, r - 0.5)
        else:
            # Inset recessed groove gradient: light metallic concave groove
            bg_grad = QLinearGradient(0, 0, 0, h)
            bg_grad.setColorAt(0.0, QColor(214, 219, 228))
            bg_grad.setColorAt(0.18, QColor(224, 228, 236))
            bg_grad.setColorAt(0.85, QColor(234, 238, 245))
            bg_grad.setColorAt(1.0, QColor(242, 245, 250))
            painter.setBrush(QBrush(bg_grad))

            # Outer groove rim border
            pen = QPen(QColor(194, 200, 212), 1.0)
            painter.setPen(pen)
            painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), r, r)

            # Subtle inner top shadow
            shadow_pen = QPen(QColor(0, 0, 0, 25), 1.0)
            painter.setPen(shadow_pen)
            painter.drawRoundedRect(QRectF(1.5, 1.5, w - 3.0, h - 3.0), r - 0.5, r - 0.5)


class FilmstripWidget(QWidget):
    """ACR-style bottom negative filmstrip with 35mm frames, multi-selection,
    horizontal wheel scrolling, and batch export right-click menu.
    """
    photoSelected = Signal(str)
    photoRemoved = Signal(str)
    photosRemoved = Signal(list)
    photosDeleted = Signal(list)
    clearRequested = Signal()
    batchExportRequested = Signal(list)
    quickExportRequested = Signal(list)
    addRequested = Signal()
    clearSdcRequested = Signal(list)
    clearCropRequested = Signal(list)
    copyParamsRequested = Signal(str)
    pasteParamsRequested = Signal()
    selectionChanged = Signal(list)
    sortRequested = Signal(str)
    stepPhotoRequested = Signal(int)
    navigateFirstRequested = Signal()
    navigateLastRequested = Signal()
    photoRated = Signal(str, int)  # photo_id, rating
    batchPhotoRated = Signal(list, int)  # photo_ids, rating
    filterChanged = Signal(str)  # filter_key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setMaximumHeight(260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._item_widgets = {}
        self._photos_list = []
        self._active_id = None
        self._selection_anchor_id = None
        self._selected_ids = set()
        self._rating_filter = "all"
        self._can_paste_params = False
        self._pending_scroll_timers = []
        self._select_debounce_timer = QTimer(self)
        self._select_debounce_timer.setSingleShot(True)
        self._select_debounce_timer.setInterval(25)
        self._select_debounce_timer.timeout.connect(self._dispatch_photo_selected)
        self._pending_select_id = None
        self._init_ui()
        theme_manager.themeChanged.connect(lambda _: self.apply_theme())

    def _schedule_photo_selected(self, photo_id):
        """Debounces photoSelected signal emission by 25ms so selection frame moves in 0ms without UI hitch."""
        self._pending_select_id = photo_id
        self._select_debounce_timer.start(25)

    def _dispatch_photo_selected(self):
        if self._pending_select_id:
            pid = self._pending_select_id
            self._pending_select_id = None
            self.photoSelected.emit(pid)

    def step_photo(self, delta):
        """Immediately moves the visual selection box (0ms) and smoothly navigates previous (-1) or next (+1) photo."""
        if not self._photos_list or not self._active_id:
            return
        pids = [p["id"] for p in self._photos_list if p and "id" in p]
        if self._active_id not in pids:
            return
        idx = pids.index(self._active_id)
        new_idx = idx + delta
        if 0 <= new_idx < len(pids):
            target_id = pids[new_idx]
            self.set_active_photo(target_id, scroll_to_visible=True, reset_selection=True)
            self._schedule_photo_selected(target_id)

    def _cancel_pending_scroll_timers(self):
        """Cancel any delayed scroll alignment timers to prevent jumping during user interactions."""
        timers = getattr(self, '_pending_scroll_timers', [])
        for timer in list(timers):
            try:
                timer.stop()
                timer.deleteLater()
            except Exception:
                pass
        self._pending_scroll_timers = []

    def _safe_single_shot(self, delay_ms, callback):
        """Spawn a cancellable single-shot timer tracked in self._pending_scroll_timers."""
        if not hasattr(self, '_pending_scroll_timers'):
            self._pending_scroll_timers = []
        timer = QTimer(self)
        timer.setSingleShot(True)

        def _wrapper():
            if timer in self._pending_scroll_timers:
                self._pending_scroll_timers.remove(timer)
            timer.deleteLater()
            try:
                callback()
            except Exception:
                pass

        timer.timeout.connect(_wrapper)
        self._pending_scroll_timers.append(timer)
        timer.start(delay_ms)
        return timer

    def set_can_paste(self, can_paste: bool):
        """Update whether clipboard contains a valid darkroom recipe."""
        self._can_paste_params = bool(can_paste)


    def _matches_filter(self, rating: int, filter_key: str) -> bool:
        if filter_key == "all":
            return True
        r = int(rating or 0)
        if filter_key in ("5", "4", "3", "2", "1", "0"):
            return r == int(filter_key)
        if filter_key.endswith("+"):
            return r == int(filter_key[:-1])
        return True

    def set_rating_filter(self, filter_key: str):
        self._rating_filter = filter_key
        labels_map = {
            "all": "★ 全部 ▾",
            "5": "★ 5星 ▾",
            "4": "★ 4星 ▾",
            "3": "★ 3星 ▾",
            "2": "★ 2星 ▾",
            "1": "★ 1星 ▾",
            "0": "★ 无评分 ▾"
        }
        if hasattr(self, 'filter_btn'):
            self.filter_btn.setText(labels_map.get(filter_key, "★ 筛选 ▾"))
            self.apply_theme()

        visible_pids = []
        for p in self._photos_list:
            pid = p.get("id")
            r = int(p.get("rating", 0) or 0)
            matches = self._matches_filter(r, filter_key)
            if pid in self._item_widgets:
                self._item_widgets[pid].setVisible(matches)
            if matches:
                visible_pids.append(pid)

        total = len(self._photos_list)
        if hasattr(self, 'count_label'):
            if filter_key == "all":
                self.count_label.setText(f"{total} 张")
            else:
                self.count_label.setText(f"{len(visible_pids)}/{total} 张")

        # If current active photo is hidden by filter, auto switch to first visible photo
        if self._active_id not in visible_pids and visible_pids:
            self.set_active_photo(visible_pids[0], reset_selection=True)
            self.photoSelected.emit(visible_pids[0])

        self.filterChanged.emit(filter_key)

    def set_photo_rating(self, photo_id: str, rating: int):
        clean_r = max(0, min(5, int(rating or 0)))
        for p in self._photos_list:
            if p.get("id") == photo_id:
                p["rating"] = clean_r
                break
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].set_rating(clean_r)
            matches = self._matches_filter(clean_r, self._rating_filter)
            self._item_widgets[photo_id].setVisible(matches)

        total = len(self._photos_list)
        if self._rating_filter == "all":
            self.count_label.setText(f"{total} 张")
        else:
            visible_cnt = sum(1 for p in self._photos_list if self._matches_filter(p.get("rating", 0), self._rating_filter))
            self.count_label.setText(f"{visible_cnt}/{total} 张")

    def set_photos_rating(self, photo_ids: list, rating: int):
        clean_r = max(0, min(5, int(rating or 0)))
        target_set = set(photo_ids)
        for p in self._photos_list:
            if p.get("id") in target_set:
                p["rating"] = clean_r
                pid = p["id"]
                if pid in self._item_widgets:
                    self._item_widgets[pid].set_rating(clean_r)
                    self._item_widgets[pid].setVisible(self._matches_filter(clean_r, self._rating_filter))

        total = len(self._photos_list)
        if self._rating_filter == "all":
            self.count_label.setText(f"{total} 张")
        else:
            visible_cnt = sum(1 for p in self._photos_list if self._matches_filter(p.get("rating", 0), self._rating_filter))
            self.count_label.setText(f"{visible_cnt}/{total} 张")

    def _on_item_rating_changed(self, photo_id: str, rating: int):
        self.photoRated.emit(photo_id, rating)

    def get_filter_counts(self) -> dict:
        total = len(self._photos_list)
        c5 = sum(1 for p in self._photos_list if int(p.get("rating", 0) or 0) == 5)
        c4 = sum(1 for p in self._photos_list if int(p.get("rating", 0) or 0) == 4)
        c3 = sum(1 for p in self._photos_list if int(p.get("rating", 0) or 0) == 3)
        c2 = sum(1 for p in self._photos_list if int(p.get("rating", 0) or 0) == 2)
        c1 = sum(1 for p in self._photos_list if int(p.get("rating", 0) or 0) == 1)
        c0 = sum(1 for p in self._photos_list if int(p.get("rating", 0) or 0) == 0)
        return {
            "all": total,
            "5": c5,
            "4": c4,
            "3": c3,
            "2": c2,
            "1": c1,
            "0": c0,
        }

    def get_filter_menu_items(self) -> list:
        counts = self.get_filter_counts()
        return [
            ("all", f"全部（{counts['all']}张）"),
            ("separator", ""),
            ("5", f"★★★★★（{counts['5']}张）"),
            ("4", f"★★★★（{counts['4']}张）"),
            ("3", f"★★★（{counts['3']}张）"),
            ("2", f"★★（{counts['2']}张）"),
            ("1", f"★（{counts['1']}张）"),
            ("separator", ""),
            ("0", f"无评分（{counts['0']}张）"),
        ]

    def _show_filter_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())
        for key, text in self.get_filter_menu_items():
            if key == "separator":
                menu.addSeparator()
                continue
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(self._rating_filter == key)
            act.triggered.connect(lambda chk=False, k=key: self.set_rating_filter(k))

        btn_global = self.filter_btn.mapToGlobal(QPoint(0, self.filter_btn.height() + 2))
        menu.setWindowOpacity(0.0)
        menu.popup(btn_global)

        dest_geo = menu.geometry()
        is_upward = dest_geo.bottom() <= self.filter_btn.mapToGlobal(QPoint(0, 0)).y() + 5
        if is_upward:
            start_geo = QRect(dest_geo.x(), dest_geo.bottom(), dest_geo.width(), 1)
        else:
            start_geo = QRect(dest_geo.x(), dest_geo.y(), dest_geo.width(), 1)

        menu.setGeometry(start_geo)
        anim = QVariantAnimation(menu)
        anim.setDuration(120)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(start_geo)
        anim.setEndValue(dest_geo)

        def _on_step(r):
            menu.setGeometry(r)
            prog = anim.currentTime() / max(1, anim.duration())
            menu.setWindowOpacity(min(1.0, 0.1 + 0.9 * prog))

        def _on_finish():
            menu.setGeometry(dest_geo)
            menu.setWindowOpacity(1.0)

        anim.valueChanged.connect(_on_step)
        anim.finished.connect(_on_finish)
        self._filter_menu_anim = anim
        self._active_filter_menu = menu
        anim.start()

    def keyPressEvent(self, event):
        mods = event.modifiers()
        if (mods & Qt.KeyboardModifier.ControlModifier) and (mods & Qt.KeyboardModifier.ShiftModifier) and event.key() == Qt.Key.Key_I:
            self.invert_selection()
            event.accept()
            return
        elif (mods & Qt.KeyboardModifier.ControlModifier) and event.key() == Qt.Key.Key_A:
            self.select_all()
            event.accept()
            return

        # 0-5 Star rating shortcuts
        if not (mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
            if event.key() in (Qt.Key.Key_0, Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5):
                target_r = event.key() - Qt.Key.Key_0
                targets = list(self._selected_ids) if self._selected_ids else ([self._active_id] if self._active_id else [])
                if targets:
                    self.batchPhotoRated.emit(targets, target_r)
                    event.accept()
                    return

        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_ids:
                self.photosRemoved.emit(list(self._selected_ids))
                event.accept()
                return
        elif event.key() == Qt.Key.Key_Left:
            self.step_photo(-1)
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Right:
            self.step_photo(1)
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Home:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._select_range_to_boundary(to_start=True)
            else:
                self.navigate_first()
                self.navigateFirstRequested.emit()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_End:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._select_range_to_boundary(to_start=False)
            else:
                self.navigate_last()
                self.navigateLastRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 2, 8, 2)
        main_layout.setSpacing(6)
        is_dark = theme_manager.is_dark()
        top_border = "#1c1e28" if is_dark else "#c2c8d4"
        bg = theme_manager.color('filmstrip_bg')
        self.setStyleSheet(f"FilmstripWidget {{ background: {bg}; border: none; border-top: 1px solid {top_border}; }}\n{theme_manager.get_tooltip_stylesheet()}")

        # 1. Long Concave Film Track Channel (长条凹形胶片条槽路)
        self.track_container = FilmTrackContainer(self)
        self.track_container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.track_container.customContextMenuRequested.connect(self._on_blank_context_menu)
        track_layout = QVBoxLayout(self.track_container)
        track_layout.setContentsMargins(6, 0, 6, 0)
        track_layout.setSpacing(0)

        # Scroll area for items (Smooth non-linear scrolling within the concave track)
        self.scroll_area = SmoothScrollArea(self.track_container, is_horizontal=True, enable_edge_fade=True)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        if self.scroll_area.viewport():
            self.scroll_area.viewport().setStyleSheet("background: transparent; border: none;")
            self.scroll_area.viewport().setAutoFillBackground(False)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
                border: none;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 6px;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #3f4350;
                min-width: 24px;
                border-radius: 3px;
                border: none;
            }
            QScrollBar::handle:horizontal:hover {
                background: #cbd5e1;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal:pressed {
                background: #f59e0b;
                border-radius: 3px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                height: 0px;
                width: 0px;
                background: none;
                border: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
        """)

        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background: transparent; border: none;")
        self.scroll_layout = QHBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(4, 0, 4, 0)
        self.scroll_layout.setSpacing(6)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._empty_lbl = QLabel("底片槽路为空，点击右侧「＋ 添加」或拖入 RAW / 照片文件")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500; border: none; background: transparent;")
        self.scroll_layout.addWidget(self._empty_lbl, 1, Qt.AlignmentFlag.AlignCenter)
        self.scroll_layout.addStretch()

        self.scroll_area.setWidget(self.scroll_content)
        hbar = self.scroll_area.horizontalScrollBar()
        if hbar:
            hbar.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            hbar.sliderMoved.connect(lambda _: self._cancel_pending_scroll_timers())
        self.scroll_area.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scroll_area.customContextMenuRequested.connect(self._on_blank_context_menu)
        self.scroll_content.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scroll_content.customContextMenuRequested.connect(self._on_blank_context_menu)
        orig_content_press = self.scroll_content.mousePressEvent
        def _content_mouse_press(ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                child = self.scroll_content.childAt(ev.position().toPoint())
                if child is None or child == self.scroll_content or child == self._empty_lbl:
                    if len(self._selected_ids) > 1 and self._active_id:
                        self.set_active_photo(self._active_id, reset_selection=True)
                        ev.accept()
                        return
            orig_content_press(ev)
        self.scroll_content.mousePressEvent = _content_mouse_press

        viewport = self.scroll_area.viewport()
        orig_vp_press = viewport.mousePressEvent
        def _vp_mouse_press(ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                pos_in_content = self.scroll_content.mapFrom(viewport, ev.position().toPoint())
                child = self.scroll_content.childAt(pos_in_content)
                if child is None or child == self.scroll_content or child == self._empty_lbl:
                    if len(self._selected_ids) > 1 and self._active_id:
                        self.set_active_photo(self._active_id, reset_selection=True)
                        ev.accept()
                        return
            orig_vp_press(ev)
        viewport.mousePressEvent = _vp_mouse_press

        orig_track_press = self.track_container.mousePressEvent
        def _track_mouse_press(ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                if len(self._selected_ids) > 1 and self._active_id:
                    self.set_active_photo(self._active_id, reset_selection=True)
                    ev.accept()
                    return
            orig_track_press(ev)
        self.track_container.mousePressEvent = _track_mouse_press

        track_layout.addWidget(self.scroll_area)
        main_layout.addWidget(self.track_container, 1)

        # 2. Right Action Container (Upper: 胶片库 + 张数; Lower: ＋ 添加 button)
        self.action_container = QWidget(self)
        self.action_container.setFixedWidth(66)
        self.action_container.setStyleSheet("background: transparent; border: none;")
        action_layout = QVBoxLayout(self.action_container)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)

        # Upper container: 胶片库 + 张数
        self.top_box = QFrame()
        self.top_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.top_box.setStyleSheet("background: #13141a; border: 1px solid #20222a; border-radius: 6px;")
        top_layout = QVBoxLayout(self.top_box)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self.lbl_lib_title = QLabel("胶片库")
        self.lbl_lib_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_lib_title.setStyleSheet("color: #71717a; font-size: 10px; font-weight: bold; padding: 0px; margin: 0px; border: none; background: transparent;")
        top_layout.addWidget(self.lbl_lib_title, 1)

        self.count_label = QLabel("0 张")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_label.setStyleSheet("color: #f59e0b; font-size: 10px; font-weight: bold; padding: 0px; margin: 0px; border: none; background: transparent;")
        top_layout.addWidget(self.count_label, 1)
        action_layout.addWidget(self.top_box, 1)

        # Middle container: ★ 筛选 按钮 (位于张数与添加按钮之间，三等分高度)
        self.filter_btn = QPushButton("★ 全部 ▾")
        self.filter_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.filter_btn.setToolTip("按底片星标评级筛选")
        self.filter_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.filter_btn.clicked.connect(self._show_filter_menu)
        action_layout.addWidget(self.filter_btn, 1)

        # Lower container: ＋ 添加 按钮全部铺满 (三等分高度)
        self.add_btn = QPushButton("＋ 添加")
        self.add_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.setToolTip("添加更多底片 / RAW 文件")
        self.add_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.add_btn.clicked.connect(self.addRequested.emit)
        action_layout.addWidget(self.add_btn, 1)

        main_layout.addWidget(self.action_container)
        self.apply_theme()

    def apply_theme(self):
        """Re-render filmstrip and track colors to match active UI theme (light / dark)."""
        is_dark = theme_manager.is_dark()
        top_border = "#1c1e28" if is_dark else "#c2c8d4"
        bg = theme_manager.color('filmstrip_bg')
        tt_ss = theme_manager.get_tooltip_stylesheet()
        self.setStyleSheet(f"FilmstripWidget {{ background: {bg}; border: none; border-top: 1px solid {top_border}; }}\n{tt_ss}")
        if hasattr(self, "track_container") and self.track_container:
            self.track_container.setStyleSheet(f"background: transparent; border: none;\n{tt_ss}")
            self.track_container.update()
        if hasattr(self, "action_container") and self.action_container:
            self.action_container.setStyleSheet(f"background: transparent; border: none;\n{tt_ss}")
        if hasattr(self, "scroll_area") and self.scroll_area:
            if self.scroll_area.viewport():
                self.scroll_area.viewport().setStyleSheet(f"background: transparent; border: none;\n{tt_ss}")
                self.scroll_area.viewport().setAutoFillBackground(False)
            hbar = self.scroll_area.horizontalScrollBar()
            if hbar:
                hbar.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        if hasattr(self, "scroll_content") and self.scroll_content:
            self.scroll_content.setStyleSheet(f"background: transparent; border: none;\n{tt_ss}")
        if is_dark:
            self.scroll_area.setStyleSheet(f"""
                QScrollArea {{
                    background: transparent;
                    border: none;
                }}
                QScrollArea > QWidget > QWidget {{
                    background: transparent;
                    border: none;
                }}
                QScrollBar:horizontal {{
                    background: transparent;
                    height: 6px;
                    margin: 0px;
                    border: none;
                }}
                QScrollBar::handle:horizontal {{
                    background: #3f4350;
                    min-width: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:horizontal:hover {{
                    background: #cbd5e1;
                    border-radius: 3px;
                }}
                QScrollBar::handle:horizontal:pressed {{
                    background: #f59e0b;
                    border-radius: 3px;
                }}
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }}
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                    background: none;
                    border: none;
                }}
                {tt_ss}
            """)
            if hasattr(self, "top_box"):
                self.top_box.setStyleSheet("background: #13141a; border: 1px solid #20222a; border-radius: 6px;")
            if hasattr(self, "lbl_lib_title"):
                self.lbl_lib_title.setStyleSheet("color: #71717a; font-size: 10px; font-weight: bold; padding: 0px; margin: 0px; border: none; background: transparent;")
            if hasattr(self, "count_label"):
                self.count_label.setStyleSheet("color: #f59e0b; font-size: 10px; font-weight: bold; padding: 0px; margin: 0px; border: none; background: transparent;")
            if hasattr(self, "filter_btn"):
                if self._rating_filter == "all":
                    self.filter_btn.setStyleSheet("""
                        QPushButton {
                            background: #15161c;
                            color: #cbd5e1;
                            border: 1px solid #282a36;
                            border-radius: 6px;
                            font-size: 10px;
                            font-weight: 500;
                            padding: 0px;
                            margin: 0px;
                            outline: none;
                        }
                        QPushButton:hover {
                            background: #232532;
                            border-color: #f59e0b;
                            color: #ffffff;
                        }
                        QPushButton:pressed {
                            background: #121317;
                            color: #d97706;
                            border: 1px solid #d97706;
                        }
                    """)
                else:
                    self.filter_btn.setStyleSheet("""
                        QPushButton {
                            background: #241d11;
                            color: #f59e0b;
                            border: 1px solid #f59e0b;
                            border-radius: 6px;
                            font-size: 10px;
                            font-weight: bold;
                            padding: 0px;
                            margin: 0px;
                            outline: none;
                        }
                        QPushButton:hover {
                            background: #332714;
                            color: #fbbf24;
                        }
                        QPushButton:pressed {
                            background: #18130a;
                            color: #d97706;
                            border: 1px solid #d97706;
                        }
                    """)
            if hasattr(self, "add_btn"):
                self.add_btn.setStyleSheet("""
                    QPushButton {
                        background: #181920;
                        color: #f59e0b;
                        border: 1px solid #2d303e;
                        border-radius: 6px;
                        font-size: 11px;
                        font-weight: bold;
                        padding: 0px;
                        margin: 0px;
                        outline: none;
                    }
                    QPushButton:hover {
                        background: #232530;
                        color: #fbbf24;
                        border: 1px solid #f59e0b;
                    }
                    QPushButton:pressed {
                        background: #121317;
                        color: #d97706;
                        border: 1px solid #d97706;
                    }
                    QPushButton:focus {
                        outline: none;
                        border: 1px solid #2d303e;
                    }
                """)
        else:
            self.scroll_area.setStyleSheet(f"""
                QScrollArea {{
                    background: transparent;
                    border: none;
                }}
                QScrollArea > QWidget > QWidget {{
                    background: transparent;
                    border: none;
                }}
                QScrollBar:horizontal {{
                    background: transparent;
                    height: 6px;
                    margin: 0px;
                    border: none;
                }}
                QScrollBar::handle:horizontal {{
                    background: #94a3b8;
                    min-width: 24px;
                    border-radius: 3px;
                    border: none;
                }}
                QScrollBar::handle:horizontal:hover {{
                    background: #64748b;
                    border-radius: 3px;
                }}
                QScrollBar::handle:horizontal:pressed {{
                    background: #f59e0b;
                    border-radius: 3px;
                }}
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }}
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                    background: none;
                    border: none;
                }}
                {tt_ss}
            """)
            if hasattr(self, "top_box"):
                self.top_box.setStyleSheet("background: #ffffff; border: 1px solid #d5d9e4; border-radius: 6px;")
            if hasattr(self, "lbl_lib_title"):
                self.lbl_lib_title.setStyleSheet("color: #64748b; font-size: 10px; font-weight: bold; padding: 0px; margin: 0px; border: none; background: transparent;")
            if hasattr(self, "count_label"):
                self.count_label.setStyleSheet("color: #d97706; font-size: 10px; font-weight: bold; padding: 0px; margin: 0px; border: none; background: transparent;")
            if hasattr(self, "filter_btn"):
                if self._rating_filter == "all":
                    self.filter_btn.setStyleSheet("""
                        QPushButton {
                            background: #ffffff;
                            color: #334155;
                            border: 1px solid #d5d9e4;
                            border-radius: 6px;
                            font-size: 10px;
                            font-weight: 500;
                            padding: 0px;
                            margin: 0px;
                            outline: none;
                        }
                        QPushButton:hover {
                            background: #f8fafc;
                            color: #0f172a;
                            border-color: #d97706;
                        }
                        QPushButton:pressed {
                            background: #e2e8f0;
                            color: #0f172a;
                            border: 1px solid #d97706;
                        }
                    """)
                else:
                    self.filter_btn.setStyleSheet("""
                        QPushButton {
                            background: #fef3c7;
                            color: #d97706;
                            border: 1px solid #d97706;
                            border-radius: 6px;
                            font-size: 10px;
                            font-weight: bold;
                            padding: 0px;
                            margin: 0px;
                            outline: none;
                        }
                        QPushButton:hover {
                            background: #fde68a;
                        }
                        QPushButton:pressed {
                            background: #fcd34d;
                            color: #92400e;
                            border: 1px solid #d97706;
                        }
                    """)
            if hasattr(self, "add_btn"):
                self.add_btn.setStyleSheet("""
                    QPushButton {
                        background: #ffffff;
                        color: #d97706;
                        border: 1px solid #d5d9e4;
                        border-radius: 6px;
                        font-size: 11px;
                        font-weight: bold;
                        padding: 0px;
                        margin: 0px;
                        outline: none;
                    }
                    QPushButton:hover {
                        background: #fef3c7;
                        color: #b45309;
                        border: 1px solid #f59e0b;
                    }
                    QPushButton:pressed {
                        background: #fde68a;
                        color: #92400e;
                        border: 1px solid #d97706;
                    }
                    QPushButton:focus {
                        outline: none;
                        border: 1px solid #d5d9e4;
                    }
                """)
        if hasattr(self, "track_container"):
            self.track_container.update()
        for w in self._item_widgets.values():
            if hasattr(w, "_apply_theme_style"):
                w._apply_theme_style()
            w.update()
        self.update()


    def set_photo_dirty(self, photo_id, is_dirty=True):
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].set_dirty(is_dirty)

    def set_photo_crop(self, photo_id, crop_data):
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].set_crop(crop_data)
        for p in self._photos_list:
            if p.get("id") == photo_id:
                if isinstance(crop_data, dict):
                    p["crop"] = dict(crop_data)
                    if "params" in p and isinstance(p["params"], dict):
                        p["params"]["crop"] = dict(crop_data)
                elif isinstance(crop_data, bool):
                    if not crop_data:
                        def_c = {
                            "enabled": False, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
                            "angle": 0.0, "rot_90": 0, "flip_h": False, "flip_v": False,
                            "aspect_ratio": "original", "is_portrait": False
                        }
                        p["crop"] = def_c
                        if "params" in p and isinstance(p["params"], dict):
                            p["params"]["crop"] = def_c
                    else:
                        if "crop" not in p or not isinstance(p["crop"], dict):
                            p["crop"] = {"enabled": True}
                        else:
                            p["crop"]["enabled"] = True
                break

    def _compute_item_dimensions(self):
        """Computes (item_w, item_h) leaving comfortable vertical padding
        inside the concave film track matching horizontal ends padding (10px).
        """
        avail_h = self.height()
        item_h = max(48, min(240, avail_h - 24))
        item_w = int(round(item_h * 0.92))
        return item_w, item_h

    def resizeEvent(self, event):
        """Anchor visible photos during resize (e.g. splitter dragging) to eliminate
        horizontal drift and preserve scroll stability.
        """
        if not hasattr(self, 'scroll_area') or not self.scroll_area:
            super().resizeEvent(event)
            return

        vp = self.scroll_area.viewport()
        hbar = self.scroll_area.horizontalScrollBar()
        if not vp or not hbar or not self._item_widgets:
            super().resizeEvent(event)
            item_w, item_h = self._compute_item_dimensions()
            for item in self._item_widgets.values():
                item.set_item_size(item_w, item_h)
            return

        # Cancel any pending delayed timers so they don't fire mid-resize
        self._cancel_pending_scroll_timers()

        vp_w_before = vp.width()
        vp_center = vp_w_before // 2

        was_at_min = (hbar.value() <= hbar.minimum())
        was_at_max = (hbar.value() >= hbar.maximum() and hbar.maximum() > 0)

        anchor_item = None
        if not was_at_min and not was_at_max:
            # Priority 1: active photo if its center is within the viewport [0, vp_w_before]
            if self._active_id and self._active_id in self._item_widgets:
                act_w = self._item_widgets[self._active_id]
                try:
                    act_pos = act_w.mapTo(vp, QPoint(act_w.width() // 2, 0)).x()
                    if 0 <= act_pos <= vp_w_before:
                        anchor_item = act_w
                except Exception:
                    pass

            # Priority 2: if active photo is outside viewport (scrolled elsewhere),
            # pick the item whose center is closest to viewport center
            if anchor_item is None:
                closest_dist = float('inf')
                for item in self._item_widgets.values():
                    try:
                        pos = item.mapTo(vp, QPoint(item.width() // 2, 0)).x()
                        dist = abs(pos - vp_center)
                        if dist < closest_dist:
                            closest_dist = dist
                            anchor_item = item
                    except Exception:
                        pass

        center_before = None
        ratio = 0.5
        if anchor_item is not None:
            try:
                center_before = anchor_item.mapTo(vp, QPoint(anchor_item.width() // 2, 0)).x()
                ratio = (center_before / vp_w_before) if vp_w_before > 0 else 0.5
            except Exception:
                pass

        super().resizeEvent(event)

        item_w, item_h = self._compute_item_dimensions()
        for item in self._item_widgets.values():
            item.set_item_size(item_w, item_h)

        if hasattr(self, 'scroll_content') and self.scroll_content:
            self.scroll_content.adjustSize()

        if was_at_min:
            target = hbar.minimum()
        elif was_at_max:
            target = hbar.maximum()
        elif anchor_item is not None and center_before is not None:
            vp_w_after = vp.width()
            desired_center = int(vp_w_after * ratio)
            new_target = anchor_item.x() + (anchor_item.width() // 2) - desired_center
            target = max(hbar.minimum(), min(hbar.maximum(), new_target))
        else:
            target = hbar.value()

        if hasattr(self.scroll_area, "scroll_to"):
            self.scroll_area.scroll_to(target, animate=False)
        else:
            hbar.setValue(target)

    def wheelEvent(self, event):
        """Enable horizontal scrolling via standard mouse wheel over the filmstrip."""
        self._cancel_pending_scroll_timers()
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta != 0:
            if hasattr(self.scroll_area, "wheelEvent"):
                self.scroll_area.wheelEvent(event)
            else:
                h_bar = self.scroll_area.horizontalScrollBar()
                h_bar.setValue(h_bar.value() - delta)
                event.accept()
        else:
            super().wheelEvent(event)

    def set_photos(self, photos, active_id):
        self._photos_list = photos
        self._active_id = active_id
        self._selection_anchor_id = active_id
        if active_id:
            self._selected_ids = {active_id}
        else:
            self._selected_ids = set()
        self._base_selection = set(self._selected_ids)

        # Clear existing
        for w in self._item_widgets.values():
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self._item_widgets.clear()

        # Update count
        self.count_label.setText(f"{len(photos)} 张")

        if len(photos) == 0:
            if not hasattr(self, '_empty_lbl') or self._empty_lbl is None:
                self._empty_lbl = QLabel("底片槽路为空，点击右侧「＋ 添加」或拖入 RAW / 照片文件开始扫描")
                self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._empty_lbl.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500; border: none; background: transparent;")
            self.scroll_layout.insertWidget(0, self._empty_lbl, 1, Qt.AlignmentFlag.AlignCenter)
            self.selectionChanged.emit([])
            return

        if hasattr(self, '_empty_lbl') and self._empty_lbl is not None:
            self.scroll_layout.removeWidget(self._empty_lbl)
            self._empty_lbl.deleteLater()
            self._empty_lbl = None

        # Re-add with current dynamic size
        item_w, item_h = self._compute_item_dimensions()
        for idx, p in enumerate(photos):
            pid = p["id"]
            is_active = (pid == active_id)
            is_selected = (pid in self._selected_ids)
            item = FilmstripItemWidget(p, is_active=is_active, is_selected=is_selected)
            item.set_item_size(item_w, item_h)
            item.clicked.connect(self._on_item_clicked)
            item.removeRequested.connect(self.photoRemoved.emit)
            item.contextMenuRequested.connect(self._show_context_menu)
            item.ratingChanged.connect(self._on_item_rating_changed)
            self._item_widgets[pid] = item
            self.scroll_layout.insertWidget(idx, item, 0, Qt.AlignmentFlag.AlignVCenter)
        self.selectionChanged.emit(list(self._selected_ids))

        # Apply active rating filter
        if self._rating_filter != "all":
            self.set_rating_filter(self._rating_filter)

        # Auto-scroll to active photo upon initial population with multi-phase layout calibration
        if active_id:
            self.scroll_to_active(animate=False, force_center=True)
            self._safe_single_shot(0, lambda: self.scroll_to_active(animate=False, force_center=True))
            self._safe_single_shot(60, lambda: self.scroll_to_active(animate=False, force_center=True))
            self._safe_single_shot(260, lambda: self.scroll_to_active(animate=False, force_center=True))

    def append_photo(self, photo, is_active=False):
        """Incrementally appends a single photo entry to the filmstrip without clearing."""
        if hasattr(self, '_empty_lbl') and self._empty_lbl is not None:
            self.scroll_layout.removeWidget(self._empty_lbl)
            self._empty_lbl.deleteLater()
            self._empty_lbl = None

        if photo not in self._photos_list:
            self._photos_list.append(photo)
        self.count_label.setText(f"{len(self._photos_list)} 张")

        item_w, item_h = self._compute_item_dimensions()
        pid = photo["id"]

        if is_active:
            self._active_id = pid
            self._selected_ids = {pid}

        item = FilmstripItemWidget(photo, is_active=is_active, is_selected=is_active)
        item.set_item_size(item_w, item_h)
        item.clicked.connect(self._on_item_clicked)
        item.removeRequested.connect(self.photoRemoved.emit)
        item.contextMenuRequested.connect(self._show_context_menu)
        item.ratingChanged.connect(self._on_item_rating_changed)
        item.setVisible(self._matches_filter(photo.get("rating", 0), self._rating_filter))
        self._item_widgets[pid] = item
        idx = max(0, len(self._item_widgets) - 1)
        self.scroll_layout.insertWidget(idx, item, 0, Qt.AlignmentFlag.AlignVCenter)
        if is_active:
            self.selectionChanged.emit(list(self._selected_ids))

    def remove_photo(self, photo_id, next_active_id=None):
        """Incrementally removes a single photo entry without flushing the entire layout,
        strictly preserving the horizontal scroll position.
        """
        w = self._item_widgets.pop(photo_id, None)
        if w is not None:
            self.scroll_layout.removeWidget(w)
            w.deleteLater()

        self._photos_list = [p for p in self._photos_list if p.get("id") != photo_id]
        self._selected_ids.discard(photo_id)
        self.count_label.setText(f"{len(self._photos_list)} 张")

        if len(self._photos_list) == 0:
            if not hasattr(self, '_empty_lbl') or self._empty_lbl is None:
                self._empty_lbl = QLabel("底片槽路为空，点击右侧「＋ 添加」或拖入 RAW / 照片文件开始扫描")
                self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._empty_lbl.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500; border: none; background: transparent;")
            self.scroll_layout.insertWidget(0, self._empty_lbl, 1, Qt.AlignmentFlag.AlignCenter)
            self._active_id = None
            self._selected_ids = set()
            self.selectionChanged.emit([])
            return

        if next_active_id:
            self.set_active_photo(next_active_id, scroll_to_visible=True)

    def remove_multiple_photos(self, photo_ids, next_active_id=None):
        """Incrementally removes multiple photo entries without flushing the entire layout."""
        id_set = set(photo_ids)
        for pid in id_set:
            w = self._item_widgets.pop(pid, None)
            if w is not None:
                self.scroll_layout.removeWidget(w)
                w.deleteLater()
            self._selected_ids.discard(pid)

        self._photos_list = [p for p in self._photos_list if p.get("id") not in id_set]
        self.count_label.setText(f"{len(self._photos_list)} 张")

        if len(self._photos_list) == 0:
            if not hasattr(self, '_empty_lbl') or self._empty_lbl is None:
                self._empty_lbl = QLabel("底片槽路为空，点击右侧「＋ 添加」或拖入 RAW / 照片文件开始扫描")
                self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._empty_lbl.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500; border: none; background: transparent;")
            self.scroll_layout.insertWidget(0, self._empty_lbl, 1, Qt.AlignmentFlag.AlignCenter)
            self._active_id = None
            self._selected_ids = set()
            self.selectionChanged.emit([])
            return

        if next_active_id:
            self.set_active_photo(next_active_id, scroll_to_visible=True)

    def set_active_photo(self, active_id, scroll_to_visible=False, reset_selection=False, new_selected_ids=None):
        old_active = self._active_id
        old_selected = set(self._selected_ids)
        self._active_id = active_id
        if new_selected_ids is not None:
            self._selected_ids = set(new_selected_ids)
        elif reset_selection or (active_id and active_id not in self._selected_ids):
            self._selected_ids = {active_id} if active_id else set()

        self._update_items_selection()

        self.selectionChanged.emit(list(self._selected_ids))
        if scroll_to_visible and active_id:
            self.scroll_to_photo(active_id, animate=True, force_center=False)

    def get_selected_photo_ids(self):
        return list(self._selected_ids) if self._selected_ids else ([self._active_id] if self._active_id else [])

    def select_all_photos(self):
        """Select all photos in filmstrip (Ctrl+A)."""
        self.select_all()

    def _update_items_selection(self):
        for pid, w in self._item_widgets.items():
            w.set_active(pid == self._active_id and pid in self._selected_ids)
            w.set_selected(pid in self._selected_ids)
            w.update()

    def _on_item_clicked(self, photo_id, event):
        mods = event.modifiers()
        is_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        is_shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if is_shift and self._item_widgets:
            # Shift+Click or Ctrl+Shift+Click: Range selection preserving previous Ctrl selections
            pids = [p["id"] for p in self._photos_list if p and "id" in p]
            anchor = self._selection_anchor_id if (self._selection_anchor_id in pids) else self._active_id
            if not anchor or anchor not in pids:
                anchor = photo_id
                self._selection_anchor_id = anchor
            if anchor in pids and photo_id in pids:
                i1 = pids.index(anchor)
                i2 = pids.index(photo_id)
                start, end = min(i1, i2), max(i1, i2)
                range_ids = set(pids[start:end+1])
                base_sel = getattr(self, "_base_selection", None)
                if base_sel is None:
                    base_sel = set(self._selected_ids)
                new_selection = set(base_sel) | range_ids
                # Activate clicked photo and set range selection without moving the anchor
                self.set_active_photo(photo_id, reset_selection=False, new_selected_ids=new_selection)
                self._schedule_photo_selected(photo_id)
        elif is_ctrl:
            # Ctrl+Click: Toggle selection WITHOUT switching active canvas photo
            if photo_id in self._selected_ids:
                if len(self._selected_ids) > 1:
                    self._selected_ids.remove(photo_id)
                    # If active photo was deselected, update internal pointer without emitting photoSelected
                    if self._active_id == photo_id:
                        self._active_id = next(iter(self._selected_ids))
            else:
                self._selected_ids.add(photo_id)
                # Keep active photo intact so canvas never switches
                if not self._active_id:
                    self._active_id = photo_id
            self._selection_anchor_id = photo_id
            self._base_selection = set(self._selected_ids)

            self._update_items_selection()
            self.selectionChanged.emit(list(self._selected_ids))
            return
        else:
            # Normal single click: cancels multi-selection, establishes new selection anchor and base selection
            self._selection_anchor_id = photo_id
            self._base_selection = {photo_id}
            self.set_active_photo(photo_id, reset_selection=True)
            self._schedule_photo_selected(photo_id)

    def update_photo_thumbnail(self, photo_id, pixmap):
        """Update thumbnail for specific photo item."""
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].set_thumbnail(pixmap)

    def update_photo_data(self, photo_id, photo_data):
        """Update photo metadata and tooltip for specific photo item."""
        if photo_id in self._item_widgets:
            self._item_widgets[photo_id].update_photo_data(photo_data)

    def scroll_to_photo(self, photo_id, animate=False, force_center=False):
        """Ensure the specified photo widget is visible in the scroll area.
        If force_center is True, or if the widget is outside the visible viewport,
        scroll it to the center smoothly or instantly.
        """
        if not photo_id or photo_id not in self._item_widgets:
            return

        w = self._item_widgets[photo_id]
        hbar = self.scroll_area.horizontalScrollBar()
        vp = self.scroll_area.viewport()
        if not hbar or not vp:
            return

        vp_w = vp.width()
        if vp_w <= 0:
            self._safe_single_shot(25, lambda: self.scroll_to_photo(photo_id, animate=animate, force_center=force_center))
            return

        self.scroll_content.adjustSize()
        w_x = w.x()
        w_w = w.width()

        # If w.x() is still 0 but it's not the first item, estimate from index
        if w_x == 0 and photo_id in [p.get("id") for p in self._photos_list]:
            idx = next((i for i, p in enumerate(self._photos_list) if p.get("id") == photo_id), 0)
            if idx > 0:
                item_w, _ = self._compute_item_dimensions()
                w_x = idx * (item_w + 6)
                w_w = item_w

        cur_val = hbar.value()
        is_visible = (w_x >= cur_val + 10) and (w_x + w_w <= cur_val + vp_w - 10)

        if not force_center and is_visible:
            return

        target_x = int(w_x - (vp_w - w_w) / 2)
        target_x = max(hbar.minimum(), min(hbar.maximum(), target_x))

        if hasattr(self.scroll_area, "scroll_to"):
            self.scroll_area.scroll_to(target_x, animate=animate)
        else:
            hbar.setValue(target_x)

    def scroll_to_active(self, animate=False, force_center=True):
        """Ensure active photo item is centered in view with multi-tick alignment."""
        if not self._active_id:
            return
        self.scroll_to_photo(self._active_id, animate=animate, force_center=force_center)
        # Double-check calibration on next tick to guarantee perfection after Qt layout pass
        self._safe_single_shot(30, lambda: self.scroll_to_photo(self._active_id, animate=False, force_center=force_center))

    def scroll_to_start(self):
        """Scroll filmstrip to the very beginning."""
        if hasattr(self, "scroll_area") and self.scroll_area:
            hbar = self.scroll_area.horizontalScrollBar()
            if hbar:
                hbar.setValue(hbar.minimum())

    def scroll_to_end(self):
        """Scroll filmstrip to the very end."""
        if hasattr(self, "scroll_area") and self.scroll_area:
            hbar = self.scroll_area.horizontalScrollBar()
            if hbar:
                hbar.setValue(hbar.maximum())

    def navigate_first(self):
        """Navigate to the first photo in filmstrip, activate and select it (Home key)."""
        if not self._photos_list:
            return
        first_id = self._photos_list[0].get("id")
        if first_id:
            self.set_active_photo(first_id, reset_selection=True)
            self._schedule_photo_selected(first_id)
            self.scroll_to_start()
            self.scroll_to_active()

    def navigate_last(self):
        """Navigate to the last photo in filmstrip, activate and select it (End key)."""
        if not self._photos_list:
            return
        last_id = self._photos_list[-1].get("id")
        if last_id:
            self.set_active_photo(last_id, reset_selection=True)
            self._schedule_photo_selected(last_id)
            self.scroll_to_end()
            self.scroll_to_active()

    def _select_range_to_boundary(self, to_start=True):
        """Handle Shift+Home or Shift+End range selection to the start or end."""
        if not self._photos_list:
            return
        pids = [p["id"] for p in self._photos_list if p and "id" in p]
        if not pids:
            return
        curr_idx = pids.index(self._active_id) if self._active_id in pids else (0 if to_start else len(pids) - 1)
        target_idx = 0 if to_start else len(pids) - 1
        start, end = min(curr_idx, target_idx), max(curr_idx, target_idx)
        range_ids = set(pids[start:end + 1])
        target_id = pids[target_idx]
        self.set_active_photo(target_id, new_selected_ids=range_ids)
        self._schedule_photo_selected(target_id)
        if to_start:
            self.scroll_to_start()
        else:
            self.scroll_to_end()
        self.scroll_to_active()

    def _show_context_menu(self, target_id, global_pos):
        # Decouple right-click: ensure target_id is selected for action scope, but DO NOT switch active canvas photo
        if target_id not in self._selected_ids:
            self._selected_ids = {target_id}
            self._update_items_selection()
            self.selectionChanged.emit(list(self._selected_ids))

        sel_count = len(self._selected_ids)
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        quick_label = "快速扫描 (上次参数)" if sel_count == 1 else f"快速扫描选中底片 ({sel_count} 张)"
        act_quick = menu.addAction(quick_label)
        act_quick.setIcon(get_icon("quick_export.png"))
        act_quick.triggered.connect(lambda: self.quickExportRequested.emit(list(self._selected_ids)))

        export_label = f"扫描当前底片..." if sel_count == 1 else f"批量扫描选中底片 ({sel_count} 张)..."
        act_export = menu.addAction(export_label)
        act_export.setIcon(get_icon("dlg_export.png"))
        act_export.triggered.connect(lambda: self.batchExportRequested.emit(list(self._selected_ids)))

        target_photo = next((p for p in self._photos_list if p.get("id") == target_id), None)
        target_path = target_photo.get("path", "") if target_photo else ""
        if target_path:
            act_reveal = menu.addAction("在文件资源管理器中显示")
            act_reveal.triggered.connect(lambda checked=False, p=target_path: self._reveal_in_explorer(p))

        menu.addSeparator()

        # Submenu: 底片评级 (Rating)
        target_r = int(target_photo.get("rating", 0) or 0) if target_photo else 0
        rate_menu = menu.addMenu("底片评级")
        rate_menu.setStyleSheet(get_darkroom_menu_style())
        for r_val, r_text in [
            (5, "★★★★★ 5 星\t5"),
            (4, "★★★★☆ 4 星\t4"),
            (3, "★★★☆☆ 3 星\t3"),
            (2, "★★☆☆☆ 2 星\t2"),
            (1, "★☆☆☆☆ 1 星\t1"),
            (0, "清除评级\t0"),
        ]:
            act_r = rate_menu.addAction(r_text)
            if r_val == target_r:
                act_r.setIcon(get_icon("check.png") if os.path.exists(os.path.join(get_resource_dir(), "icons", "check.png")) else QIcon())
            act_r.triggered.connect(lambda chk=False, r=r_val: self.batchPhotoRated.emit(list(self._selected_ids), r))

        # Submenu: 星标筛选 (Filter)
        filter_menu = menu.addMenu("星标筛选")
        filter_menu.setStyleSheet(get_darkroom_menu_style())
        for key, text in self.get_filter_menu_items():
            if key == "separator":
                filter_menu.addSeparator()
                continue
            act_f = filter_menu.addAction(text)
            act_f.setCheckable(True)
            act_f.setChecked(self._rating_filter == key)
            act_f.triggered.connect(lambda chk=False, k=key: self.set_rating_filter(k))

        menu.addSeparator()

        remove_label = "从底片库移除" if sel_count == 1 else f"从底片库移除 ({sel_count} 项)"
        act_remove = menu.addAction(remove_label)
        act_remove.setIcon(get_icon("dlg_discard.png"))
        act_remove.setShortcut(QKeySequence("Delete"))
        act_remove.triggered.connect(lambda: self.photosRemoved.emit(list(self._selected_ids)))

        delete_label = "删除文件" if sel_count == 1 else f"删除文件 ({sel_count} 项)"
        act_delete = menu.addAction(delete_label)
        act_delete.setShortcut(QKeySequence("Shift+Delete"))
        act_delete.triggered.connect(lambda: self.photosDeleted.emit(list(self._selected_ids)))

        menu.addSeparator()

        # Copy / Paste darkroom recipe actions
        target_edited = False
        if target_photo:
            p_path = target_photo.get("path", "")
            if target_photo.get("is_dirty") or target_photo.get("is_edited") or (p_path and sdc_manager.has_sdc(p_path)):
                target_edited = True
            elif target_photo.get("film_profile", "none") != "none" or target_photo.get("paper_profile", "none") != "none":
                target_edited = True

        act_copy = menu.addAction("复制暗房配方")
        act_copy.setShortcut(QKeySequence("Ctrl+C"))
        act_copy.setEnabled(target_edited)
        act_copy.triggered.connect(lambda checked=False, pid=target_id: self.copyParamsRequested.emit(pid))

        paste_label = "粘贴暗房配方" if sel_count <= 1 else f"粘贴暗房配方至选中底片 ({sel_count} 张)"
        act_paste = menu.addAction(paste_label)
        act_paste.setShortcut(QKeySequence("Ctrl+V"))
        can_paste = bool(getattr(self, "_can_paste_params", False))
        act_paste.setEnabled(can_paste)
        act_paste.triggered.connect(self.pasteParamsRequested.emit)

        menu.addSeparator()

        # Check if any selected item is edited or has an existing .sdc file
        selected_photos = [p for p in self._photos_list if p["id"] in self._selected_ids]
        can_clear_sdc = False
        for p in selected_photos:
            p_path = p.get("path", "")
            if p.get("is_dirty") or p.get("is_edited") or (p_path and sdc_manager.has_sdc(p_path)):
                can_clear_sdc = True
                break

        sdc_label = "删除暗房配置 (.sdc)" if sel_count == 1 else f"删除暗房配置 (.sdc) ({sel_count} 项)"
        act_clear_sdc = menu.addAction(sdc_label)
        act_clear_sdc.setIcon(get_icon("reset.png"))
        act_clear_sdc.setEnabled(can_clear_sdc)
        act_clear_sdc.triggered.connect(lambda: self.clearSdcRequested.emit(list(self._selected_ids)))

        # Check if any selected item has crop enabled
        can_clear_crop = False
        for p in selected_photos:
            c = p.get("crop")
            if not c:
                params = p.get("params")
                if isinstance(params, dict):
                    c = params.get("crop")
            if isinstance(c, dict):
                if c.get("enabled", False):
                    can_clear_crop = True
                    break
                w = float(c.get("w", 1.0))
                h = float(c.get("h", 1.0))
                x = float(c.get("x", 0.0))
                y = float(c.get("y", 0.0))
                rot = int(c.get("rot_90", 0)) % 360
                angle = float(c.get("angle", 0.0))
                if (abs(w - 1.0) > 0.001 or abs(h - 1.0) > 0.001 or
                    abs(x) > 0.001 or abs(y) > 0.001 or rot != 0 or abs(angle) > 0.01 or
                    c.get("flip_h") or c.get("flip_v")):
                    can_clear_crop = True
                    break
            elif isinstance(c, bool) and c:
                can_clear_crop = True
                break

        crop_label = "清除裁切" if sel_count == 1 else f"清除裁切 ({sel_count} 项)"
        act_clear_crop = menu.addAction(crop_label)
        act_clear_crop.setEnabled(can_clear_crop)
        act_clear_crop.triggered.connect(lambda: self.clearCropRequested.emit(list(self._selected_ids)))

        menu.addSeparator()

        # Submenu: 星标筛选
        b_filter_menu = menu.addMenu("星标筛选")
        b_filter_menu.setStyleSheet(get_darkroom_menu_style())
        b_filter_menu.setEnabled(bool(self._photos_list))
        for key, text in self.get_filter_menu_items():
            if key == "separator":
                b_filter_menu.addSeparator()
                continue
            act_bf = b_filter_menu.addAction(text)
            act_bf.setCheckable(True)
            act_bf.setChecked(self._rating_filter == key)
            act_bf.triggered.connect(lambda chk=False, k=key: self.set_rating_filter(k))

        # Submenu: 整理底片
        sort_menu = menu.addMenu("整理底片")
        sort_menu.setStyleSheet(get_darkroom_menu_style())
        act_s1 = sort_menu.addAction("按拍摄时间升序 (旧 → 新)")
        act_s1.triggered.connect(lambda: self.sortRequested.emit("time_asc"))
        act_s2 = sort_menu.addAction("按拍摄时间降序 (新 → 旧)")
        act_s2.triggered.connect(lambda: self.sortRequested.emit("time_desc"))
        sort_menu.addSeparator()
        act_s3 = sort_menu.addAction("按文件名升序 (A → Z)")
        act_s3.triggered.connect(lambda: self.sortRequested.emit("name_asc"))
        act_s4 = sort_menu.addAction("按文件名降序 (Z → A)")
        act_s4.triggered.connect(lambda: self.sortRequested.emit("name_desc"))
        sort_menu.addSeparator()
        act_s5 = sort_menu.addAction("按导入添加顺序")
        act_s5.triggered.connect(lambda: self.sortRequested.emit("import"))

        menu.addSeparator()

        act_select_all = menu.addAction("全选所有底片")
        act_select_all.setShortcut(QKeySequence("Ctrl+A"))
        act_select_all.triggered.connect(self.select_all)

        act_invert = menu.addAction("反选")
        act_invert.setShortcut(QKeySequence("Ctrl+Shift+I"))
        act_invert.triggered.connect(self.invert_selection)

        menu.exec(global_pos)

    def _reveal_in_explorer(self, file_path):
        if not file_path:
            return
        import subprocess
        norm_p = os.path.abspath(file_path)
        if os.path.exists(norm_p):
            subprocess.Popen(f'explorer /select,"{norm_p}"')
        else:
            dir_p = os.path.dirname(norm_p)
            if os.path.exists(dir_p):
                os.startfile(dir_p)

    def _on_blank_context_menu(self, pos):
        sender = self.sender()
        global_pos = sender.mapToGlobal(pos) if sender else QCursor.pos()
        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        act_add = menu.addAction("导入底片...")
        act_add.setIcon(get_icon("dlg_import.png"))
        act_add.triggered.connect(self.addRequested.emit)

        menu.addSeparator()

        # Submenu: 星标筛选
        filter_menu = menu.addMenu("星标筛选")
        filter_menu.setStyleSheet(get_darkroom_menu_style())
        filter_menu.setEnabled(bool(self._photos_list))
        for key, text in self.get_filter_menu_items():
            if key == "separator":
                filter_menu.addSeparator()
                continue
            act_f = filter_menu.addAction(text)
            act_f.setCheckable(True)
            act_f.setChecked(self._rating_filter == key)
            act_f.triggered.connect(lambda chk=False, k=key: self.set_rating_filter(k))

        # Submenu: 整理底片
        sort_menu = menu.addMenu("整理底片")
        sort_menu.setStyleSheet(get_darkroom_menu_style())
        act_s1 = sort_menu.addAction("按拍摄时间升序 (旧 → 新)")
        act_s1.triggered.connect(lambda: self.sortRequested.emit("time_asc"))
        act_s2 = sort_menu.addAction("按拍摄时间降序 (新 → 旧)")
        act_s2.triggered.connect(lambda: self.sortRequested.emit("time_desc"))
        sort_menu.addSeparator()
        act_s3 = sort_menu.addAction("按文件名升序 (A → Z)")
        act_s3.triggered.connect(lambda: self.sortRequested.emit("name_asc"))
        act_s4 = sort_menu.addAction("按文件名降序 (Z → A)")
        act_s4.triggered.connect(lambda: self.sortRequested.emit("name_desc"))
        sort_menu.addSeparator()
        act_s5 = sort_menu.addAction("按导入添加顺序")
        act_s5.triggered.connect(lambda: self.sortRequested.emit("import"))

        menu.addSeparator()

        act_select_all = menu.addAction("全选所有底片")
        act_select_all.setShortcut(QKeySequence("Ctrl+A"))
        act_select_all.triggered.connect(self.select_all)

        act_invert = menu.addAction("反选")
        act_invert.setShortcut(QKeySequence("Ctrl+Shift+I"))
        act_invert.triggered.connect(self.invert_selection)

        if self._photos_list:
            act_clear = menu.addAction("清空底片库")
            act_clear.setIcon(get_icon("dlg_discard.png"))
            act_clear.triggered.connect(self.clearRequested.emit)

        menu.exec(global_pos)

    def select_all(self):
        self._selected_ids = {p["id"] for p in self._photos_list if p and "id" in p}
        self._base_selection = set(self._selected_ids)
        self._update_items_selection()
        self.selectionChanged.emit(list(self._selected_ids))

    def invert_selection(self):
        all_ids = {p["id"] for p in self._photos_list if p and "id" in p}
        new_selected = all_ids - self._selected_ids
        new_active = self._active_id
        if new_selected:
            if self._active_id not in new_selected:
                new_active = next((p["id"] for p in self._photos_list if p and p.get("id") in new_selected), None)
        else:
            new_active = None

        self._selection_anchor_id = new_active
        self._base_selection = set(new_selected)
        self.set_active_photo(new_active, reset_selection=False, new_selected_ids=new_selected)
        if new_active:
            self._schedule_photo_selected(new_active)

    def get_selected_ids(self) -> list[str]:
        """Returns a list of currently selected photo IDs in the filmstrip."""
        return list(self._selected_ids)
