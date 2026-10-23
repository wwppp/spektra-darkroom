"""crop_overlay.py - Interactive Crop & Straighten Subsystem for SpektraDarkroom
ACR 16.0 standard architecture:
- Unified right-sidebar integration using AccordionSection and AdobeScrubSlider
- High-DPI custom cursors (Crop cursor and curved Rotation cursor)
- Full viewport dark dim mask (zero white edges or borders)
- Subtle white hairline crop boundary with solid white corner L-brackets (no yellow lines)
- True-direction live rotation with high-density millimeter leveling grid
- Stationary background canvas during crop box repositioning
- Edge-clamped translation without box shrinking
- Automatic user crop scale memory and expansion restoration
- Strict canvas zoom & pan coordinate locking
"""

import os
import math
from typing import Optional, Tuple
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QPoint
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPainterPath, QCursor, QFont,
    QPixmap, QPolygonF, QAction, QActionGroup, QKeySequence
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
    QFrame, QSlider, QSizePolicy, QMenu
)

from ui.theme_manager import theme_manager
from ui.window_utils import get_icon
from ui.accordion import AccordionSection
from ui.adobe_scrub_slider import AdobeScrubSlider, ComboBoxRow


ASPECT_RATIOS = [
    ("原始比例", "original"),
    ("自由比例", "free"),
    ("1:1", "1:1"),
    ("4:5", "4:5"),
    ("3:2", "3:2"),
    ("16:9", "16:9"),
    ("2.39:1", "2.39:1"),
]

GRID_TYPES = [
    ("三分法则", 0),
    ("黄金比例", 1),
    ("对角", 2),
    ("居中", 3),
    ("三角形", 4),
    ("黄金螺旋", 5),
    ("密织校平网格", 6),
    ("无参考线", 7),
]


def _create_rotate_cursor(alpha_deg: float) -> QCursor:
    """Generate crisp high-DPI curved rotation double-arrow cursor facing outward at alpha_deg angle (ACR style)."""
    size = 96
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    rad0 = math.radians(alpha_deg)
    ux = math.cos(rad0)
    uy = -math.sin(rad0)
    R = 38.0
    cx = 48.0 - R * ux
    cy = 48.0 - R * uy

    arc_span = 32.0  # +/- 32 deg -> 64 deg arc matching native cursors
    num_pts = 48
    pts = []
    for i in range(num_pts + 1):
        th = -arc_span + (2.0 * arc_span * i / num_pts)
        a = math.radians(alpha_deg + th)
        pts.append(QPointF(cx + R * math.cos(a), cy - R * math.sin(a)))

    p.setPen(QPen(QColor(0, 0, 0, 255), 6.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for i in range(len(pts) - 1):
        p.drawLine(pts[i], pts[i + 1])

    p.setPen(QPen(QColor(255, 255, 255, 255), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for i in range(len(pts) - 1):
        p.drawLine(pts[i], pts[i + 1])

    def draw_arrow(th_deg, pointing_increasing):
        a = math.radians(alpha_deg + th_deg)
        x = cx + R * math.cos(a)
        y = cy - R * math.sin(a)
        if pointing_increasing:
            tx = -math.sin(a)
            ty = -math.cos(a)
        else:
            tx = math.sin(a)
            ty = math.cos(a)
        nx = math.cos(a)
        ny = -math.sin(a)

        arrow_len = 15.0
        arrow_half_w = 7.5
        p_tip = QPointF(x + tx * 3.0, y + ty * 3.0)
        base_x = x - tx * (arrow_len - 3.0)
        base_y = y - ty * (arrow_len - 3.0)
        p_l = QPointF(base_x + nx * arrow_half_w, base_y + ny * arrow_half_w)
        p_r = QPointF(base_x - nx * arrow_half_w, base_y - ny * arrow_half_w)

        p.setPen(QPen(QColor(0, 0, 0, 255), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.MiterJoin))
        p.setBrush(QBrush(QColor(255, 255, 255, 255)))
        p.drawPolygon(QPolygonF([p_tip, p_l, p_r]))

    draw_arrow(arc_span, pointing_increasing=True)
    draw_arrow(-arc_span, pointing_increasing=False)
    p.end()

    pm.setDevicePixelRatio(2.0)
    return QCursor(pm, 24, 24)


def _get_rotate_cursor() -> QCursor:
    """Return default high-DPI curved rotation cursor at 45-degree angle."""
    return _create_rotate_cursor(45.0)


def _get_crop_cursor() -> QCursor:
    """Return crisp high-DPI ACR-style crop selection cursor."""
    icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "icons", "crop_cursor.png")
    if os.path.exists(icon_path):
        pm = QPixmap(icon_path)
        pm.setDevicePixelRatio(2.0)
        return QCursor(pm, 8, 8)

    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    p.setPen(QPen(QColor(0, 0, 0, 255), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
    p.drawRect(16, 16, 28, 28)
    p.setPen(QPen(QColor(255, 255, 255, 255), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
    p.drawRect(16, 16, 28, 28)

    p.setPen(QPen(QColor(0, 0, 0, 255), 4.5))
    p.drawLine(6, 16, 16, 16)
    p.drawLine(16, 6, 16, 16)
    p.setPen(QPen(QColor(255, 255, 255, 255), 2.0))
    p.drawLine(6, 16, 16, 16)
    p.drawLine(16, 6, 16, 16)

    p.end()
    pm.setDevicePixelRatio(2.0)
    return QCursor(pm, 8, 8)


class CropOverlayWidget(QWidget):
    """Interactive overlay placed directly over CanvasViewport during crop mode.

    Handles on-canvas mouse gestures:
    - 8-handle corner/edge dragging with stationary diagonal anchors
    - Marquee crop rectangle dragging
    - Full-canvas rotation in gray area outside the crop box (correct rotation direction)
    - High-density leveling grid during rotation
    - Stationary background photo during translation
    - Edge-clamped translation without box shrinking
    - User scale memory for automatic inscribe & expand restoration
    - Strict canvas pan & zoom locking
    """

    cropChanged = Signal(dict)       # Live parameter dict during dragging
    cropApplied = Signal(dict)       # Confirmed crop dict
    cropCancelled = Signal()         # Cancelled without changes

    HANDLE_SIZE = 14
    CORNER_EXTENT = 24
    ROTATE_MARGIN = 36

    def _draw_guide_line(self, painter: QPainter, p1: QPointF, p2: QPointF, is_fine: bool = False):
        """Draw high-contrast, bright hairline guide line matching ACR styling."""
        pen_w = 0.75 if is_fine else 1.0
        # 1. Subtle dark shadow offset by +1px for visibility on bright highlights
        painter.setPen(QPen(QColor(0, 0, 0, 50 if is_fine else 75), pen_w, Qt.PenStyle.SolidLine))
        painter.drawLine(QPointF(p1.x() + 1.0, p1.y() + 1.0), QPointF(p2.x() + 1.0, p2.y() + 1.0))
        # 2. Main bright translucent white hairline at 0px
        painter.setPen(QPen(QColor(255, 255, 255, 125 if is_fine else 165), pen_w, Qt.PenStyle.SolidLine))
        painter.drawLine(p1, p2)

    def _draw_guide_path(self, painter: QPainter, path: QPainterPath):
        """Draw high-contrast bright hairline guide path matching _draw_guide_line."""
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # 1. Subtle dark shadow offset by +1px
        offset_path = QPainterPath(path)
        offset_path.translate(1.0, 1.0)
        painter.setPen(QPen(QColor(0, 0, 0, 75), 1.0, Qt.PenStyle.SolidLine))
        painter.drawPath(offset_path)
        # 2. Main bright translucent white hairline at 0px
        painter.setPen(QPen(QColor(255, 255, 255, 165), 1.0, Qt.PenStyle.SolidLine))
        painter.drawPath(path)

    def __init__(self, canvas, parent=None):
        p = parent if parent is not None else (canvas if isinstance(canvas, QWidget) else None)
        super().__init__(p)
        self.canvas = canvas
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # High-DPI ACR cursors for 4 quadrants (outward facing arc)
        self._rotate_cursor = _get_rotate_cursor()
        self._rotate_tr = _create_rotate_cursor(45.0)
        self._rotate_tl = _create_rotate_cursor(135.0)
        self._rotate_bl = _create_rotate_cursor(225.0)
        self._rotate_br = _create_rotate_cursor(315.0)
        self._crop_cursor = _get_crop_cursor()

        # Crop rect in normalized image coords [0.0 ~ 1.0], top-left is (0, 0)
        self.crop_x = 0.0
        self.crop_y = 0.0
        self.crop_w = 1.0
        self.crop_h = 1.0
        self.crop_angle = 0.0  # In degrees (-45.0 to +45.0)
        self.rot_90 = 0  # 0, 90, 180, 270 degrees
        self.flip_h = False
        self.flip_v = False
        self.aspect_ratio_mode = "original"
        self.is_portrait_orientation = False  # Track orientation (landscape vs portrait)
        self.grid_type = 0  # 0: Rule-of-thirds, 1: Golden ratio, etc.
        self.show_guides = True
        self.grid_orientation = 0
        self.mask_alpha = 160

        # User scale memory: remembers the user's manual crop dimensions
        # When rotating, crop box shrinks to fit; when straightening back, it restores to user size
        self._user_crop_w = 1.0
        self._user_crop_h = 1.0

        # Initial backup for cancellation
        self._initial_crop = {}

        # Interaction state
        self._active_handle = None
        self._drag_start_pos = None
        self._drag_start_rect = None
        self._drag_start_angle = 0.0
        self._is_rotating = False
        self._is_panning = False
        self._space_held = False
        self._marquee_mode_enabled = False
        self._marquee_start_pos = None

        self._crop_undo_stack = []
        self._crop_redo_stack = []

        self.hide()

    def _get_effective_image_size(self) -> Tuple[float, float]:
        """Return the effective (width, height) of the image in pixels, respecting 90°/270° orientation."""
        w = float(getattr(self.canvas, "_master_width", 0) or getattr(self.canvas, "master_width", 0) or getattr(self.canvas, "_image_width", 0) or 6000)
        h = float(getattr(self.canvas, "_master_height", 0) or getattr(self.canvas, "master_height", 0) or getattr(self.canvas, "_image_height", 0) or 4000)
        if getattr(self, "rot_90", 0) in (90, 270):
            w, h = h, w
        return max(1.0, w), max(1.0, h)

    def set_mask_opacity(self, alpha: int):
        self.mask_alpha = max(0, min(255, int(alpha)))
        self.update()

    def toggle_show_guides(self):
        self.show_guides = not getattr(self, "show_guides", True)
        self.update()

    def cycle_grid_type(self):
        self.grid_type = (self.grid_type + 1) % len(GRID_TYPES)
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._sync_sidebar_grid()

    def cycle_grid_orientation(self):
        self.grid_orientation = (getattr(self, "grid_orientation", 0) + 1) % 4
        self.update()

    def _sync_sidebar(self):
        mw = self.window()
        if hasattr(mw, "crop_sidebar") and mw.crop_sidebar:
            mw.crop_sidebar.sync_from_overlay(self.get_crop_dict())

    def _capture_snapshot(self) -> dict:
        return {
            "x": round(self.crop_x, 5),
            "y": round(self.crop_y, 5),
            "w": round(self.crop_w, 5),
            "h": round(self.crop_h, 5),
            "angle": round(self.crop_angle, 1),
            "rot_90": int(self.rot_90) % 360,
            "flip_h": bool(self.flip_h),
            "flip_v": bool(self.flip_v),
            "aspect_ratio": str(self.aspect_ratio_mode),
            "is_portrait": bool(getattr(self, "is_portrait_orientation", False)),
            "_user_crop_w": round(self._user_crop_w, 5),
            "_user_crop_h": round(self._user_crop_h, 5),
            "pan_x": float(getattr(self.canvas, "pan_x", 0.0)),
            "pan_y": float(getattr(self.canvas, "pan_y", 0.0)),
            "zoom": float(getattr(self.canvas, "zoom", 1.0)),
        }

    def _is_snapshot_equal(self, s1: dict, s2: dict) -> bool:
        if not s1 or not s2:
            return False
        for k in ("x", "y", "w", "h", "angle", "rot_90", "flip_h", "flip_v", "aspect_ratio", "is_portrait", "pan_x", "pan_y", "zoom"):
            if s1.get(k) != s2.get(k):
                return False
        return True

    def _push_undo_snapshot(self):
        if not hasattr(self, "_crop_undo_stack"):
            self._crop_undo_stack = []
        if not hasattr(self, "_crop_redo_stack"):
            self._crop_redo_stack = []
        snap = self._capture_snapshot()
        if self._crop_undo_stack and self._is_snapshot_equal(self._crop_undo_stack[-1], snap):
            return
        self._crop_undo_stack.append(snap)
        if len(self._crop_undo_stack) > 50:
            self._crop_undo_stack.pop(0)
        self._crop_redo_stack.clear()

    def undo(self) -> bool:
        """Undo last crop manipulation within the crop session."""
        if not hasattr(self, "_crop_undo_stack") or len(self._crop_undo_stack) <= 1:
            return False
        curr = self._crop_undo_stack.pop()
        self._crop_redo_stack.append(curr)
        prev = self._crop_undo_stack[-1]
        self._apply_snapshot(prev)
        return True

    def redo(self) -> bool:
        """Redo last undone crop manipulation within the crop session."""
        if not hasattr(self, "_crop_redo_stack") or not self._crop_redo_stack:
            return False
        nxt = self._crop_redo_stack.pop()
        self._crop_undo_stack.append(nxt)
        self._apply_snapshot(nxt)
        return True

    def _apply_snapshot(self, snap: dict):
        self.crop_x = snap["x"]
        self.crop_y = snap["y"]
        self.crop_w = snap["w"]
        self.crop_h = snap["h"]
        self.crop_angle = snap["angle"]
        self.rot_90 = snap["rot_90"]
        self.flip_h = snap["flip_h"]
        self.flip_v = snap["flip_v"]
        self.aspect_ratio_mode = snap["aspect_ratio"]
        self.is_portrait_orientation = snap["is_portrait"]
        self._user_crop_w = snap.get("_user_crop_w", self.crop_w)
        self._user_crop_h = snap.get("_user_crop_h", self.crop_h)
        if "pan_x" in snap and hasattr(self.canvas, "pan_x"):
            self.canvas.pan_x = snap["pan_x"]
            self.canvas.pan_y = snap["pan_y"]
            self.canvas.zoom = snap.get("zoom", 1.0)

        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())

    def start_crop_session(self, initial_crop=None):
        """Enter crop mode with existing or default crop parameters."""
        self.setGeometry(self.canvas.rect())

        crop_dict = initial_crop or {}
        if crop_dict.get("enabled"):
            self.crop_x = float(crop_dict.get("x", 0.0))
            self.crop_y = float(crop_dict.get("y", 0.0))
            self.crop_w = float(crop_dict.get("w", 1.0))
            self.crop_h = float(crop_dict.get("h", 1.0))
            self.crop_angle = float(crop_dict.get("angle", 0.0))
            self.rot_90 = int(crop_dict.get("rot_90", 0)) % 360
            self.flip_h = bool(crop_dict.get("flip_h", False))
            self.flip_v = bool(crop_dict.get("flip_v", False))
            self.aspect_ratio_mode = str(crop_dict.get("aspect_ratio", "original"))
        else:
            self.crop_x = 0.0
            self.crop_y = 0.0
            self.crop_w = 1.0
            self.crop_h = 1.0
            self.crop_angle = 0.0
            self.rot_90 = 0
            self.flip_h = False
            self.flip_v = False
            self.aspect_ratio_mode = str(crop_dict.get("aspect_ratio", "original"))

        # Clamp initial rect
        self.crop_x = max(0.0, min(0.95, self.crop_x))
        self.crop_y = max(0.0, min(0.95, self.crop_y))
        self.crop_w = max(0.05, min(1.0 - self.crop_x, self.crop_w))
        self.crop_h = max(0.05, min(1.0 - self.crop_y, self.crop_h))

        img_w, img_h = self._get_effective_image_size()
        self.is_portrait_orientation = bool((self.crop_w * img_w) < (self.crop_h * img_h))

        # Initialize user scale baseline
        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h

        self._initial_crop = self.get_crop_dict()
        if abs(self.crop_angle) > 0.01:
            self._clamp_to_rotated_bounds()

        self._marquee_mode_enabled = False
        self._initial_pan = (self.canvas.pan_x, self.canvas.pan_y)
        self._initial_zoom = self.canvas.zoom
        self._crop_undo_stack = [self._capture_snapshot()]
        self._crop_redo_stack = []
        self.show()
        self.setFocus()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())

    def get_crop_dict(self) -> dict:
        is_modified = not (
            abs(self.crop_x) < 0.001 and
            abs(self.crop_y) < 0.001 and
            abs(self.crop_w - 1.0) < 0.001 and
            abs(self.crop_h - 1.0) < 0.001 and
            abs(self.crop_angle) < 0.01 and
            self.rot_90 == 0 and
            not self.flip_h and
            not self.flip_v
        )
        return {
            "enabled": is_modified,
            "x": round(self.crop_x, 5),
            "y": round(self.crop_y, 5),
            "w": round(self.crop_w, 5),
            "h": round(self.crop_h, 5),
            "angle": round(self.crop_angle, 1),
            "rot_90": self.rot_90,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "aspect_ratio": self.aspect_ratio_mode,
            "is_portrait": getattr(self, "is_portrait_orientation", False)
        }

    def rotate_90_ccw(self):
        """Rotate 90 degrees counter-clockwise (ACR action)."""
        self.rot_90 = (self.rot_90 - 90) % 360
        self.crop_x = 0.0
        self.crop_y = 0.0
        self.crop_w = 1.0
        self.crop_h = 1.0
        self._user_crop_w = 1.0
        self._user_crop_h = 1.0
        self._enforce_aspect_ratio()
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def rotate_90_cw(self):
        """Rotate 90 degrees clockwise (ACR action)."""
        self.rot_90 = (self.rot_90 + 90) % 360
        self.crop_x = 0.0
        self.crop_y = 0.0
        self.crop_w = 1.0
        self.crop_h = 1.0
        self._user_crop_w = 1.0
        self._user_crop_h = 1.0
        self._enforce_aspect_ratio()
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def toggle_flip_v(self):
        """Vertical flip (top-to-bottom)."""
        self.flip_v = not self.flip_v
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def toggle_flip_h(self):
        """Horizontal flip (left-to-right)."""
        self.flip_h = not self.flip_h
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def reset_all_crop(self):
        """Instant reset of all crop and transform parameters (triggered by Ctrl+R)."""
        self.canvas.pan_x = 0.0
        self.canvas.pan_y = 0.0
        self.canvas.zoom = 1.0
        self.crop_x = 0.0
        self.crop_y = 0.0
        self.crop_w = 1.0
        self.crop_h = 1.0
        self.crop_angle = 0.0
        self.rot_90 = 0
        self.flip_h = False
        self.flip_v = False
        img_w, img_h = self._get_effective_image_size()
        self.aspect_ratio_mode = "original"
        self.is_portrait_orientation = bool(img_w < img_h)
        self._user_crop_w = 1.0
        self._user_crop_h = 1.0
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def clear_crop_completely(self):
        """Reset crop box to full and immediately notify canvas to clear crop."""
        self.reset_all_crop()
        if hasattr(self.canvas, "requestClearCrop"):
            self.canvas.requestClearCrop.emit()
        else:
            self.apply_crop()

    def reset_crop(self):
        """Reset crop box to full image and straighten angle to 0°."""
        self.crop_x = 0.0
        self.crop_y = 0.0
        self.crop_w = 1.0
        self.crop_h = 1.0
        self.crop_angle = 0.0
        self.aspect_ratio_mode = "original"
        self._user_crop_w = 1.0
        self._user_crop_h = 1.0
        img_w, img_h = self._get_effective_image_size()
        self.is_portrait_orientation = bool(img_w < img_h)
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def reset_to_initial_state(self):
        """Restore crop to the state when crop overlay was opened (Ctrl+Alt+R)."""
        init_crop = getattr(self, "_initial_crop", {})
        if not init_crop:
            self.reset_crop()
            return
        self.crop_x = float(init_crop.get("x", 0.0))
        self.crop_y = float(init_crop.get("y", 0.0))
        self.crop_w = float(init_crop.get("w", 1.0))
        self.crop_h = float(init_crop.get("h", 1.0))
        self.crop_angle = float(init_crop.get("angle", 0.0))
        self.rot_90 = int(init_crop.get("rot_90", 0)) % 360
        self.flip_h = bool(init_crop.get("flip_h", False))
        self.flip_v = bool(init_crop.get("flip_v", False))
        self.aspect_ratio_mode = str(init_crop.get("aspect_ratio", "original"))
        self.is_portrait_orientation = bool(init_crop.get("is_portrait", False))
        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def cancel_crop(self):
        self.canvas.pan_x = 0.0
        self.canvas.pan_y = 0.0
        self.canvas.zoom = 1.0
        if hasattr(self.canvas, "zoomChanged"):
            self.canvas.zoomChanged.emit(self.canvas.get_actual_zoom_ratio())
        init_crop = getattr(self, "_initial_crop", {})
        self.crop_x = float(init_crop.get("x", 0.0))
        self.crop_y = float(init_crop.get("y", 0.0))
        self.crop_w = float(init_crop.get("w", 1.0))
        self.crop_h = float(init_crop.get("h", 1.0))
        self.crop_angle = float(init_crop.get("angle", 0.0))
        self.rot_90 = int(init_crop.get("rot_90", 0)) % 360
        self.flip_h = bool(init_crop.get("flip_h", False))
        self.flip_v = bool(init_crop.get("flip_v", False))
        self.aspect_ratio_mode = str(init_crop.get("aspect_ratio", "original"))
        self.is_portrait_orientation = bool(init_crop.get("is_portrait", False))
        self._crop_undo_stack = []
        self._crop_redo_stack = []
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = init_crop
            self.canvas.update()
        self.hide()
        self.cropCancelled.emit()

    def apply_crop(self):
        self.canvas.pan_x = 0.0
        self.canvas.pan_y = 0.0
        self.canvas.zoom = 1.0
        if hasattr(self.canvas, "zoomChanged"):
            self.canvas.zoomChanged.emit(self.canvas.get_actual_zoom_ratio())
        self._crop_undo_stack = []
        self._crop_redo_stack = []
        self.hide()
        self.cropApplied.emit(self.get_crop_dict())

    def enable_marquee_mode(self):
        """Allow user to click and drag to draw a brand new crop box."""
        self._marquee_mode_enabled = True
        self.setCursor(self._crop_cursor)

    def flip_aspect_ratio(self):
        """Swap crop orientation between landscape and portrait."""
        img_w, img_h = self._get_effective_image_size()

        cur_phys_w = self.crop_w * img_w
        cur_phys_h = self.crop_h * img_h
        cur_phys_ratio = cur_phys_w / max(1.0, cur_phys_h)

        self.is_portrait_orientation = not getattr(self, "is_portrait_orientation", False)

        if self.aspect_ratio_mode == "free":
            target_phys_ratio = 1.0 / max(0.001, cur_phys_ratio)
        else:
            base_r = self._get_base_preset_ratio(self.aspect_ratio_mode)
            if base_r is None:
                target_phys_ratio = 1.0 / max(0.001, cur_phys_ratio)
            else:
                r_land = max(base_r, 1.0 / base_r)
                target_phys_ratio = (1.0 / r_land) if self.is_portrait_orientation else r_land

        K = target_phys_ratio * (float(img_h) / float(img_w))
        S = max(self.crop_w, self.crop_h)
        if S >= 0.95:
            if K <= 1.0:
                new_w = K
                new_h = 1.0
            else:
                new_w = 1.0
                new_h = 1.0 / K
        else:
            if K <= 1.0:
                new_h = min(1.0, S)
                new_w = new_h * K
            else:
                new_w = min(1.0, S)
                new_h = new_w / K

        cx = self.crop_x + self.crop_w * 0.5
        cy = self.crop_y + self.crop_h * 0.5

        self.crop_w = max(0.05, min(1.0, new_w))
        self.crop_h = max(0.05, min(1.0, new_h))
        self.crop_x = max(0.0, min(1.0 - self.crop_w, cx - self.crop_w * 0.5))
        self.crop_y = max(0.0, min(1.0 - self.crop_h, cy - self.crop_h * 0.5))
        self.is_portrait_orientation = bool(self.crop_w * img_w < self.crop_h * img_h)
        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h

        if abs(self.crop_angle) > 0.01:
            self._clamp_to_rotated_bounds()

        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())
        self._push_undo_snapshot()

    def set_aspect_ratio_mode(self, mode: str):
        self.aspect_ratio_mode = mode
        if mode == "free":
            if abs(self.crop_angle) > 0.01:
                self._clamp_to_rotated_bounds()
            if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
                self.canvas.params["crop"] = self.get_crop_dict()
                self.canvas.update()
            self.update()
            self.cropChanged.emit(self.get_crop_dict())
            self._sync_sidebar()
            self._push_undo_snapshot()
            return
        self._apply_preset_ratio(mode)
        self._sync_sidebar()
        self._push_undo_snapshot()

    def _get_base_preset_ratio(self, mode: str) -> Optional[float]:
        img_w, img_h = self._get_effective_image_size()
        native_ratio = float(img_w) / float(img_h) if img_h > 0 else 1.5

        if mode == "original":
            return native_ratio
        elif mode == "1:1":
            return 1.0
        elif mode == "4:5":
            return 4.0 / 5.0
        elif mode == "3:2":
            return 3.0 / 2.0
        elif mode == "16:9":
            return 16.0 / 9.0
        elif mode == "2.39:1":
            return 2.39
        return None

    def _apply_preset_ratio(self, mode: str):
        base_r = self._get_base_preset_ratio(mode)
        if base_r is None:
            return
        img_w, img_h = self._get_effective_image_size()

        cur_is_portrait = (self.crop_w * img_w < self.crop_h * img_h)
        self.is_portrait_orientation = cur_is_portrait

        r_land = max(base_r, 1.0 / base_r)
        target_phys_ratio = (1.0 / r_land) if self.is_portrait_orientation else r_land

        K = target_phys_ratio * (float(img_h) / float(img_w))
        S = max(self.crop_w, self.crop_h)
        if S >= 0.95:
            if K <= 1.0:
                new_w, new_h = K, 1.0
            else:
                new_w, new_h = 1.0, 1.0 / K
        else:
            if K <= 1.0:
                new_h = min(1.0, S)
                new_w = new_h * K
            else:
                new_w = min(1.0, S)
                new_h = new_w / K

        cx = self.crop_x + self.crop_w * 0.5
        cy = self.crop_y + self.crop_h * 0.5
        self.crop_w = max(0.05, min(1.0, new_w))
        self.crop_h = max(0.05, min(1.0, new_h))
        self.crop_x = max(0.0, min(1.0 - self.crop_w, cx - self.crop_w * 0.5))
        self.crop_y = max(0.0, min(1.0 - self.crop_h, cy - self.crop_h * 0.5))
        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h

        if abs(self.crop_angle) > 0.01:
            self._clamp_to_rotated_bounds()
        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h
        if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
            self.canvas.params["crop"] = self.get_crop_dict()
            self.canvas.update()
        self.update()
        self.cropChanged.emit(self.get_crop_dict())

    def set_crop_angle(self, angle: float, push_undo: bool = False):
        clamped = max(-45.0, min(45.0, round(angle, 1)))
        if abs(self.crop_angle - clamped) > 0.05:
            self.crop_angle = clamped
            self._clamp_to_rotated_bounds()
            if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
                crop_p = self.canvas.params.setdefault("crop", {})
                crop_p["x"] = self.crop_x
                crop_p["y"] = self.crop_y
                crop_p["w"] = self.crop_w
                crop_p["h"] = self.crop_h
                crop_p["angle"] = self.crop_angle
                self.canvas.update()
            self.update()
            self.cropChanged.emit(self.get_crop_dict())
            if push_undo:
                self._push_undo_snapshot()

    def set_grid_type(self, gt: int):
        self.grid_type = gt
        self.show_guides = True
        self.update()
        self._sync_sidebar()

    def _get_target_ratio(self) -> Optional[float]:
        """Return target width / height aspect ratio based on current mode."""
        img_w, img_h = self._get_effective_image_size()
        native_ratio = float(img_w) / float(img_h) if img_h > 0 else 1.5

        mode = self.aspect_ratio_mode
        base_r = None
        if mode == "original":
            base_r = native_ratio
        elif mode == "free":
            return None
        elif mode == "1:1":
            return 1.0
        elif mode == "4:5":
            base_r = 4.0 / 5.0
        elif mode == "3:2":
            base_r = 3.0 / 2.0
        elif mode == "16:9":
            base_r = 16.0 / 9.0
        elif mode == "2.39:1":
            base_r = 2.39

        if base_r is None:
            return None

        # Standardize base_r to always be >= 1.0 (landscape baseline), then apply portrait flag
        r_landscape = max(base_r, 1.0 / base_r)
        if getattr(self, "is_portrait_orientation", False):
            return 1.0 / r_landscape
        else:
            return r_landscape

    def _enforce_aspect_ratio(self):
        """Apply target ratio symmetrically around current crop center."""
        if self.aspect_ratio_mode != "free":
            self._apply_preset_ratio(self.aspect_ratio_mode)

    def _clamp_to_rotated_bounds(self):
        """Constrain crop rectangle inside the image boundaries when rotated by crop_angle around (0.5, 0.5),
        while maintaining user scale memory.
        - Guarantees all 4 corners stay 100% inside [0, 1] x [0, 1] with zero black pixels.
        - When rotating back towards 0°, smoothly expands back to the user's manual size!
        """
        if not hasattr(self, '_user_crop_w') or self._user_crop_w <= 0.01:
            self._user_crop_w = max(0.05, self.crop_w)
            self._user_crop_h = max(0.05, self.crop_h)

        cx = self.crop_x + self.crop_w * 0.5
        cy = self.crop_y + self.crop_h * 0.5

        if abs(self.crop_angle) < 0.01:
            # Fully straightened: restore to user scale memory
            w = min(1.0, self._user_crop_w)
            h = min(1.0, self._user_crop_h)
            self.crop_w = w
            self.crop_h = h
            self.crop_x = max(0.0, min(1.0 - w, cx - w * 0.5))
            self.crop_y = max(0.0, min(1.0 - h, cy - h * 0.5))
            return

        img_w, img_h = self._get_effective_image_size()
        A = float(img_w) / float(img_h) if img_h > 0 else 1.5

        rad = math.radians(self.crop_angle)
        ca = math.cos(rad)
        sa = math.sin(-rad)

        uc = cx - 0.5
        vc = cy - 0.5
        u_prime = uc * ca - vc * (sa / A)
        v_prime = uc * A * sa + vc * ca

        # 1. Available envelope around current rotated center (preserves position without jumping)
        avail_x = max(0.1, 1.0 - 2.0 * abs(u_prime))
        avail_y = max(0.1, 1.0 - 2.0 * abs(v_prime))

        X_term = self._user_crop_w * abs(ca) + self._user_crop_h * (abs(sa) / A)
        Y_term = self._user_crop_w * (A * abs(sa)) + self._user_crop_h * abs(ca)
        scale = min(1.0, avail_x / max(1e-4, X_term), avail_y / max(1e-4, Y_term))
        w = max(0.05, self._user_crop_w * scale)
        h = max(0.05, self._user_crop_h * scale)

        # 2. Clamped center in rotated coordinate frame
        X_term_w = w * abs(ca) + h * (abs(sa) / A)
        Y_term_h = w * (A * abs(sa)) + h * abs(ca)
        M_x = max(0.0, (1.0 - X_term_w) * 0.5)
        M_y = max(0.0, (1.0 - Y_term_h) * 0.5)

        u_clamped = max(-M_x, min(M_x, u_prime))
        v_clamped = max(-M_y, min(M_y, v_prime))

        uc_new = u_clamped * ca + v_clamped * (sa / A)
        vc_new = -u_clamped * A * sa + v_clamped * ca

        cx_new = 0.5 + uc_new
        cy_new = 0.5 + vc_new

        self.crop_w = w
        self.crop_h = h
        self.crop_x = cx_new - w * 0.5
        self.crop_y = cy_new - h * 0.5

    # -------------------------------------------------------------
    # Coordinate Mapping Helpers (Normalized UV <-> Screen Pixels)
    # -------------------------------------------------------------
    def _get_screen_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate the pixel bounding rectangle of the full image on this widget using canvas geometry."""
        left, right, top, bottom = self.canvas.get_image_bounds_norm()
        vw = float(self.width())
        vh = float(self.height())
        bx = left * vw
        by = top * vh
        bw = max(1.0, (right - left) * vw)
        bh = max(1.0, (bottom - top) * vh)
        return bx, by, bw, bh

    def _get_crop_screen_rect(self) -> QRectF:
        bx, by, bw, bh = self._get_screen_bounds()
        return QRectF(
            bx + self.crop_x * bw,
            by + self.crop_y * bh,
            self.crop_w * bw,
            self.crop_h * bh
        )

    # -------------------------------------------------------------
    # Mouse Interaction & Hit Testing
    # -------------------------------------------------------------
    def _get_rotate_cursor_for_pos(self, pos: QPointF) -> QCursor:
        """Return the quadrant-appropriate rotation cursor curving outward around the crop box."""
        rect = self._get_crop_screen_rect()
        cx = rect.center().x()
        cy = rect.center().y()
        dx = pos.x() - cx
        dy = pos.y() - cy
        if dx >= 0:
            return self._rotate_tr if dy < 0 else self._rotate_br
        else:
            return self._rotate_tl if dy < 0 else self._rotate_bl

    def _hit_test(self, pos: QPointF) -> Optional[str]:
        rect = self._get_crop_screen_rect()
        h_gap = 4.0
        outer_rect = rect.adjusted(-h_gap, -h_gap, h_gap, h_gap)
        c_margin = 20.0   # Corner hit radius
        e_margin = 12.0   # Edge hit thickness

        px = pos.x()
        py = pos.y()

        # 1. Check 4 corners first (highest priority, tested on outer_rect where handles are)
        if abs(px - outer_rect.left()) <= c_margin and abs(py - outer_rect.top()) <= c_margin:
            return "tl"
        if abs(px - outer_rect.right()) <= c_margin and abs(py - outer_rect.top()) <= c_margin:
            return "tr"
        if abs(px - outer_rect.left()) <= c_margin and abs(py - outer_rect.bottom()) <= c_margin:
            return "bl"
        if abs(px - outer_rect.right()) <= c_margin and abs(py - outer_rect.bottom()) <= c_margin:
            return "br"

        # 2. Check 4 edges (tested on outer_rect where dashed line and pill handles are)
        if abs(py - outer_rect.top()) <= e_margin and (outer_rect.left() - c_margin) <= px <= (outer_rect.right() + c_margin):
            return "t"
        if abs(py - outer_rect.bottom()) <= e_margin and (outer_rect.left() - c_margin) <= px <= (outer_rect.right() + c_margin):
            return "b"
        if abs(px - outer_rect.left()) <= e_margin and (outer_rect.top() - c_margin) <= py <= (outer_rect.bottom() + c_margin):
            return "l"
        if abs(px - outer_rect.right()) <= e_margin and (outer_rect.top() - c_margin) <= py <= (outer_rect.bottom() + c_margin):
            return "r"

        # 3. Inside crop box
        if rect.contains(pos) or outer_rect.contains(pos):
            # If currently at 100% full uncropped image or marquee mode is primed, drag creates marquee
            is_full = (self.crop_w >= 0.98 and self.crop_h >= 0.98)
            if is_full or getattr(self, "_marquee_mode_enabled", False):
                return "marquee"
            return "inside"

        # 4. Anywhere outside crop rect (dimmed photo or gray canvas area) is rotation zone
        return "rotate"

    def mousePressEvent(self, event):
        # Support canvas panning with Middle Mouse Button or Space + Left Button
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and getattr(self, "_space_held", False)):
            self._is_panning = True
            self.canvas.mousePressEvent(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            handle = self._hit_test(pos)

            # Check if user forced marquee drag via Ctrl / Alt key
            force_marquee = bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier))
            if force_marquee or handle == "marquee":
                self._active_handle = "marquee"
                self._marquee_start_pos = pos
                self._drag_start_pos = pos
                self.setCursor(self._crop_cursor)
                self.update()
                return

            if handle:
                self._active_handle = handle
                self._drag_start_pos = pos
                self._drag_last_pos = pos
                self._drag_accum_angle = float(self.crop_angle)
                self._drag_start_rect = (self.crop_x, self.crop_y, self.crop_w, self.crop_h)
                self._drag_start_angle = self.crop_angle
                self._drag_start_pan = (self.canvas.pan_x, self.canvas.pan_y)
                self._stationary_crop_rect = self._get_crop_screen_rect()
                self._is_rotating = (handle == "rotate")
                if handle == "inside":
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                elif handle == "rotate":
                    self.setCursor(self._get_rotate_cursor_for_pos(pos))
                self.update()
                return

        super().mousePressEvent(event)

    def _apply_handle_drag(self, handle: str, dx: float, dy: float):
        """Mathematical fixed-anchor resize for corners and edges.
        When dragging any corner, the opposite corner remains strictly stationary.
        Automatically switches orientation (landscape/portrait) only when dragging corners past threshold.
        Stretches stop cleanly when any boundary is reached without breaking aspect ratio or protruding outside rotated quad.
        """
        sx, sy, sw, sh = self._drag_start_rect
        img_w, img_h = self._get_effective_image_size()
        native_ratio = float(img_w) / float(img_h) if img_h > 0 else 1.5

        ratio = self._get_target_ratio()
        K = (ratio / native_ratio) if ratio else None

        A = native_ratio
        rad = math.radians(self.crop_angle)
        ca = math.cos(rad)
        sa = math.sin(-rad)

        def get_anchor_w_max(ax, ay, sign_x, sign_y, k_ratio):
            u_0 = 0.5 + (ax - 0.5) * ca - (ay - 0.5) * (sa / A)
            v_0 = 0.5 + (ax - 0.5) * A * sa + (ay - 0.5) * ca

            bounds = [1.0]
            if sign_x > 0:
                bounds.append(1.0 - ax)
            else:
                bounds.append(ax)
            if sign_y > 0:
                bounds.append((1.0 - ay) * k_ratio)
            else:
                bounds.append(ay * k_ratio)

            if abs(self.crop_angle) > 0.01:
                corner_coeffs = [
                    (0.0, 0.0),
                    (float(sign_x), 0.0),
                    (0.0, float(sign_y)),
                    (float(sign_x), float(sign_y))
                ]
                for cx_i, cy_i in corner_coeffs:
                    Mu = cx_i * ca - cy_i * sa / (A * k_ratio)
                    Mv = cx_i * A * sa + cy_i * ca / k_ratio
                    if Mu > 1e-7:
                        bounds.append((1.0 - u_0) / Mu)
                    elif Mu < -1e-7:
                        bounds.append(-u_0 / Mu)
                    if Mv > 1e-7:
                        bounds.append((1.0 - v_0) / Mv)
                    elif Mv < -1e-7:
                        bounds.append(-v_0 / Mv)
            return max(0.05, min(bounds))

        def get_edge_h_max(cx, ay, sign_y, k_ratio):
            u_0 = 0.5 + (cx - 0.5) * ca - (ay - 0.5) * (sa / A)
            v_0 = 0.5 + (cx - 0.5) * A * sa + (ay - 0.5) * ca

            bounds = [1.0]
            if sign_y > 0:
                bounds.append(1.0 - ay)
            else:
                bounds.append(ay)
            max_w_sym = 2.0 * min(cx, 1.0 - cx)
            bounds.append(max_w_sym / k_ratio)

            if abs(self.crop_angle) > 0.01:
                corner_coeffs = [
                    (-0.5 * k_ratio, 0.0),
                    (0.5 * k_ratio, 0.0),
                    (-0.5 * k_ratio, float(sign_y)),
                    (0.5 * k_ratio, float(sign_y))
                ]
                for cx_i, cy_i in corner_coeffs:
                    Mu = cx_i * ca - cy_i * sa / A
                    Mv = cx_i * A * sa + cy_i * ca
                    if Mu > 1e-7:
                        bounds.append((1.0 - u_0) / Mu)
                    elif Mu < -1e-7:
                        bounds.append(-u_0 / Mu)
                    if Mv > 1e-7:
                        bounds.append((1.0 - v_0) / Mv)
                    elif Mv < -1e-7:
                        bounds.append(-v_0 / Mv)
            return max(0.05, min(bounds))

        def get_edge_w_max(ax, cy, sign_x, k_ratio):
            u_0 = 0.5 + (ax - 0.5) * ca - (cy - 0.5) * (sa / A)
            v_0 = 0.5 + (ax - 0.5) * A * sa + (cy - 0.5) * ca

            bounds = [1.0]
            if sign_x > 0:
                bounds.append(1.0 - ax)
            else:
                bounds.append(ax)
            max_h_sym = 2.0 * min(cy, 1.0 - cy)
            bounds.append(max_h_sym * k_ratio)

            if abs(self.crop_angle) > 0.01:
                corner_coeffs = [
                    (0.0, -0.5 / k_ratio),
                    (0.0, 0.5 / k_ratio),
                    (float(sign_x), -0.5 / k_ratio),
                    (float(sign_x), 0.5 / k_ratio)
                ]
                for cx_i, cy_i in corner_coeffs:
                    Mu = cx_i * ca - cy_i * sa / A
                    Mv = cx_i * A * sa + cy_i * ca
                    if Mu > 1e-7:
                        bounds.append((1.0 - u_0) / Mu)
                    elif Mu < -1e-7:
                        bounds.append(-u_0 / Mu)
                    if Mv > 1e-7:
                        bounds.append((1.0 - v_0) / Mv)
                    elif Mv < -1e-7:
                        bounds.append(-v_0 / Mv)
            return max(0.05, min(bounds))

        def clamp_free_corner(ax, ay, raw_tx, raw_ty, sign_x, sign_y):
            if sign_x < 0:
                tx_c = max(0.0, min(ax - 0.05, raw_tx))
            else:
                tx_c = min(1.0, max(ax + 0.05, raw_tx))
            if sign_y < 0:
                ty_c = max(0.0, min(ay - 0.05, raw_ty))
            else:
                ty_c = min(1.0, max(ay + 0.05, raw_ty))

            if abs(self.crop_angle) < 0.01:
                return tx_c, ty_c

            dx_c = tx_c - ax
            dy_c = ty_c - ay
            u_0 = 0.5 + (ax - 0.5) * ca - (ay - 0.5) * (sa / A)
            v_0 = 0.5 + (ax - 0.5) * A * sa + (ay - 0.5) * ca

            coeffs = [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]
            t_max = 1.0
            for cx_i, cy_i in coeffs:
                Mu = cx_i * dx_c * ca - cy_i * dy_c * (sa / A)
                Mv = cx_i * dx_c * A * sa + cy_i * dy_c * ca
                if Mu > 1e-7:
                    t_max = min(t_max, (1.0 - u_0) / Mu)
                elif Mu < -1e-7:
                    t_max = min(t_max, -u_0 / Mu)
                if Mv > 1e-7:
                    t_max = min(t_max, (1.0 - v_0) / Mv)
                elif Mv < -1e-7:
                    t_max = min(t_max, -v_0 / Mv)

            t_max = max(0.01, min(1.0, t_max))
            return ax + t_max * dx_c, ay + t_max * dy_c

        def check_orientation_swap(rw_val: float, rh_val: float):
            nonlocal ratio, K
            if self.aspect_ratio_mode in ("free", "1:1"):
                return
            if rw_val <= 0.01 or rh_val <= 0.01:
                return
            pw = rw_val * img_w
            ph = rh_val * img_h
            if pw < 10.0 or ph < 10.0:
                return
            mr = pw / ph
            changed = False
            if self.is_portrait_orientation:
                if mr > 1.10:
                    self.is_portrait_orientation = False
                    changed = True
            else:
                if mr < 0.90:
                    self.is_portrait_orientation = True
                    changed = True
            if changed:
                ratio = self._get_target_ratio()
                K = (ratio / native_ratio) if ratio else None

        if handle == "tl":
            # Anchor is bottom-right (sx + sw, sy + sh)
            ax, ay = sx + sw, sy + sh
            tx, ty = sx + dx, sy + dy
            rw = ax - tx
            rh = ay - ty
            check_orientation_swap(rw, rh)
            if K is None:
                new_x, new_y = clamp_free_corner(ax, ay, tx, ty, -1, -1)
                new_w = ax - new_x
                new_h = ay - new_y
            else:
                w_max = get_anchor_w_max(ax, ay, -1, -1, K)
                raw_w = max(0.05, (rw + rh * K) * 0.5)
                w = min(raw_w, w_max)
                new_w = max(0.05, w)
                new_h = max(0.05, w / K)
                new_x = ax - new_w
                new_y = ay - new_h
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        elif handle == "br":
            # Anchor is top-left (sx, sy)
            ax, ay = sx, sy
            tx, ty = sx + sw + dx, sy + sh + dy
            rw = tx - ax
            rh = ty - ay
            check_orientation_swap(rw, rh)
            if K is None:
                new_tx, new_ty = clamp_free_corner(ax, ay, tx, ty, 1, 1)
                new_w = new_tx - ax
                new_h = new_ty - ay
                new_x, new_y = ax, ay
            else:
                w_max = get_anchor_w_max(ax, ay, 1, 1, K)
                raw_w = max(0.05, (rw + rh * K) * 0.5)
                w = min(raw_w, w_max)
                new_w = max(0.05, w)
                new_h = max(0.05, w / K)
                new_x, new_y = ax, ay
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        elif handle == "tr":
            # Anchor is bottom-left (sx, sy + sh)
            ax, ay = sx, sy + sh
            tx, ty = sx + sw + dx, sy + dy
            rw = tx - ax
            rh = ay - ty
            check_orientation_swap(rw, rh)
            if K is None:
                new_tx, new_ty = clamp_free_corner(ax, ay, tx, ty, 1, -1)
                new_w = new_tx - ax
                new_h = ay - new_ty
                new_x = ax
                new_y = new_ty
            else:
                w_max = get_anchor_w_max(ax, ay, 1, -1, K)
                raw_w = max(0.05, (rw + rh * K) * 0.5)
                w = min(raw_w, w_max)
                new_w = max(0.05, w)
                new_h = max(0.05, w / K)
                new_x = ax
                new_y = ay - new_h
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        elif handle == "bl":
            # Anchor is top-right (sx + sw, sy)
            ax, ay = sx + sw, sy
            tx, ty = sx + dx, sy + sh + dy
            rw = ax - tx
            rh = ty - ay
            check_orientation_swap(rw, rh)
            if K is None:
                new_tx, new_ty = clamp_free_corner(ax, ay, tx, ty, -1, 1)
                new_w = ax - new_tx
                new_h = new_ty - ay
                new_x = new_tx
                new_y = ay
            else:
                w_max = get_anchor_w_max(ax, ay, -1, 1, K)
                raw_w = max(0.05, (rw + rh * K) * 0.5)
                w = min(raw_w, w_max)
                new_w = max(0.05, w)
                new_h = max(0.05, w / K)
                new_x = ax - new_w
                new_y = ay
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        elif handle == "t":
            ay = sy + sh
            if K is None:
                raw_h = ay - (sy + dy)
                bounds = [ay]
                if abs(self.crop_angle) > 0.01:
                    for x_c in (sx, sx + sw):
                        u_0 = 0.5 + (x_c - 0.5) * ca - (ay - 0.5) * (sa / A)
                        v_0 = 0.5 + (x_c - 0.5) * A * sa + (ay - 0.5) * ca
                        Mu = sa / A
                        Mv = -ca
                        if Mu > 1e-7:
                            bounds.append((1.0 - u_0) / Mu)
                        elif Mu < -1e-7:
                            bounds.append(-u_0 / Mu)
                        if Mv > 1e-7:
                            bounds.append((1.0 - v_0) / Mv)
                        elif Mv < -1e-7:
                            bounds.append(-v_0 / Mv)
                new_h = max(0.05, min(raw_h, min(bounds)))
                new_y = ay - new_h
                new_x, new_w = sx, sw
            else:
                cx = sx + sw * 0.5
                requested_h = ay - (sy + dy)
                if abs(self.crop_angle) < 0.01:
                    max_h_fit = min(1.0, 1.0 / K)
                    new_h = min(max(0.05, requested_h), max_h_fit)
                    new_w = new_h * K
                    new_y = max(0.0, min(1.0 - new_h, ay - requested_h))
                    new_x = max(0.0, min(1.0 - new_w, cx - new_w * 0.5))
                else:
                    h_max = get_edge_h_max(cx, ay, -1, K)
                    new_h = max(0.05, min(requested_h, h_max))
                    new_w = new_h * K
                    new_x = cx - new_w * 0.5
                    new_y = ay - new_h
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        elif handle == "b":
            ay = sy
            if K is None:
                raw_h = sy + sh + dy - ay
                bounds = [1.0 - ay]
                if abs(self.crop_angle) > 0.01:
                    for x_c in (sx, sx + sw):
                        u_0 = 0.5 + (x_c - 0.5) * ca - (ay - 0.5) * (sa / A)
                        v_0 = 0.5 + (x_c - 0.5) * A * sa + (ay - 0.5) * ca
                        Mu = -sa / A
                        Mv = ca
                        if Mu > 1e-7:
                            bounds.append((1.0 - u_0) / Mu)
                        elif Mu < -1e-7:
                            bounds.append(-u_0 / Mu)
                        if Mv > 1e-7:
                            bounds.append((1.0 - v_0) / Mv)
                        elif Mv < -1e-7:
                            bounds.append(-v_0 / Mv)
                new_h = max(0.05, min(raw_h, min(bounds)))
                new_y = ay
                new_x, new_w = sx, sw
            else:
                cx = sx + sw * 0.5
                requested_h = sy + sh + dy - ay
                if abs(self.crop_angle) < 0.01:
                    max_h_fit = min(1.0, 1.0 / K)
                    new_h = min(max(0.05, requested_h), max_h_fit)
                    new_w = new_h * K
                    ideal_bottom = ay + requested_h
                    bottom = min(1.0, ideal_bottom)
                    new_y = max(0.0, min(1.0 - new_h, bottom - new_h))
                    new_x = max(0.0, min(1.0 - new_w, cx - new_w * 0.5))
                else:
                    h_max = get_edge_h_max(cx, ay, 1, K)
                    new_h = max(0.05, min(requested_h, h_max))
                    new_w = new_h * K
                    new_x = cx - new_w * 0.5
                    new_y = ay
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        elif handle == "l":
            ax = sx + sw
            if K is None:
                raw_w = ax - (sx + dx)
                bounds = [ax]
                if abs(self.crop_angle) > 0.01:
                    for y_c in (sy, sy + sh):
                        u_0 = 0.5 + (ax - 0.5) * ca - (y_c - 0.5) * (sa / A)
                        v_0 = 0.5 + (ax - 0.5) * A * sa + (y_c - 0.5) * ca
                        Mu = -ca
                        Mv = -A * sa
                        if Mu > 1e-7:
                            bounds.append((1.0 - u_0) / Mu)
                        elif Mu < -1e-7:
                            bounds.append(-u_0 / Mu)
                        if Mv > 1e-7:
                            bounds.append((1.0 - v_0) / Mv)
                        elif Mv < -1e-7:
                            bounds.append(-v_0 / Mv)
                new_w = max(0.05, min(raw_w, min(bounds)))
                new_x = ax - new_w
                new_y, new_h = sy, sh
            else:
                cy = sy + sh * 0.5
                requested_w = ax - (sx + dx)
                if abs(self.crop_angle) < 0.01:
                    max_w_fit = min(1.0, 1.0 * K)
                    new_w = min(max(0.05, requested_w), max_w_fit)
                    new_h = new_w / K
                    new_x = max(0.0, min(1.0 - new_w, ax - requested_w))
                    new_y = max(0.0, min(1.0 - new_h, cy - new_h * 0.5))
                else:
                    w_max = get_edge_w_max(ax, cy, -1, K)
                    new_w = max(0.05, min(requested_w, w_max))
                    new_h = new_w / K
                    new_x = ax - new_w
                    new_y = cy - new_h * 0.5
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        elif handle == "r":
            ax = sx
            if K is None:
                raw_w = sx + sw + dx - ax
                bounds = [1.0 - ax]
                if abs(self.crop_angle) > 0.01:
                    for y_c in (sy, sy + sh):
                        u_0 = 0.5 + (ax - 0.5) * ca - (y_c - 0.5) * (sa / A)
                        v_0 = 0.5 + (ax - 0.5) * A * sa + (y_c - 0.5) * ca
                        Mu = ca
                        Mv = A * sa
                        if Mu > 1e-7:
                            bounds.append((1.0 - u_0) / Mu)
                        elif Mu < -1e-7:
                            bounds.append(-u_0 / Mu)
                        if Mv > 1e-7:
                            bounds.append((1.0 - v_0) / Mv)
                        elif Mv < -1e-7:
                            bounds.append(-v_0 / Mv)
                new_w = max(0.05, min(raw_w, min(bounds)))
                new_x = ax
                new_y, new_h = sy, sh
            else:
                cy = sy + sh * 0.5
                requested_w = sx + sw + dx - ax
                if abs(self.crop_angle) < 0.01:
                    max_w_fit = min(1.0, 1.0 * K)
                    new_w = min(max(0.05, requested_w), max_w_fit)
                    new_h = new_w / K
                    ideal_right = ax + requested_w
                    right = min(1.0, ideal_right)
                    new_x = max(0.0, min(1.0 - new_w, right - new_w))
                    new_y = max(0.0, min(1.0 - new_h, cy - new_h * 0.5))
                else:
                    w_max = get_edge_w_max(ax, cy, 1, K)
                    new_w = max(0.05, min(requested_w, w_max))
                    new_h = new_w / K
                    new_x = ax
                    new_y = cy - new_h * 0.5
            self.crop_x, self.crop_y, self.crop_w, self.crop_h = new_x, new_y, new_w, new_h

        # Update user scale baseline
        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h
        if self.aspect_ratio_mode == "free":
            self.is_portrait_orientation = bool((self.crop_w * img_w) < (self.crop_h * img_h))

    def _apply_marquee_drag(self, pos: QPointF):
        """Draw a new crop box dynamically from marquee start position to current mouse pos."""
        bx, by, bw, bh = self._get_screen_bounds()
        p0 = self._marquee_start_pos
        p1 = pos

        u0 = max(0.0, min(1.0, (p0.x() - bx) / bw))
        v0 = max(0.0, min(1.0, (p0.y() - by) / bh))
        u1 = max(0.0, min(1.0, (p1.x() - bx) / bw))
        v1 = max(0.0, min(1.0, (p1.y() - by) / bh))

        ratio = self._get_target_ratio()
        img_w, img_h = self._get_effective_image_size()
        native_ratio = float(img_w) / float(img_h) if img_h > 0 else 1.5

        raw_du = abs(u1 - u0)
        raw_dv = abs(v1 - v0)

        if self.aspect_ratio_mode not in ("free", "1:1"):
            pw = raw_du * img_w
            ph = raw_dv * img_h
            if pw >= 10.0 and ph >= 10.0:
                mr = pw / ph
                if self.is_portrait_orientation and mr > 1.10:
                    self.is_portrait_orientation = False
                    ratio = self._get_target_ratio()
                elif not self.is_portrait_orientation and mr < 0.90:
                    self.is_portrait_orientation = True
                    ratio = self._get_target_ratio()

        if ratio is not None and native_ratio > 0:
            K = ratio / native_ratio  # norm_w / norm_h
            du = max(0.05, (raw_du + raw_dv * K) * 0.5)
            dv = du / K

            s_u = 1.0 if u1 >= u0 else -1.0
            s_v = 1.0 if v1 >= v0 else -1.0

            end_u = u0 + s_u * du
            end_v = v0 + s_v * dv

            min_u = min(u0, end_u)
            max_u = max(u0, end_u)
            min_v = min(v0, end_v)
            max_v = max(v0, end_v)

            if min_u < 0.0:
                min_u = 0.0
                max_u = min(1.0, min_u + du)
            if max_u > 1.0:
                max_u = 1.0
                min_u = max(0.0, max_u - du)
            if min_v < 0.0:
                min_v = 0.0
                max_v = min(1.0, min_v + dv)
            if max_v > 1.0:
                max_v = 1.0
                min_v = max(0.0, max_v - dv)

            self.crop_x = min_u
            self.crop_y = min_v
            self.crop_w = max_u - min_u
            self.crop_h = max_v - min_v
        else:
            self.crop_x = min(u0, u1)
            self.crop_y = min(v0, v1)
            self.crop_w = max(0.05, abs(u1 - u0))
            self.crop_h = max(0.05, abs(v1 - v0))

        self._user_crop_w = self.crop_w
        self._user_crop_h = self.crop_h
        if abs(self.crop_angle) > 0.01:
            self._clamp_to_rotated_bounds()

    def mouseMoveEvent(self, event):
        pos = event.position()

        # Canvas pan tracking
        if getattr(self, "_is_panning", False):
            self.canvas.mouseMoveEvent(event)
            self.update()
            return

        if self._active_handle:
            bx, by, bw, bh = self._get_screen_bounds()
            if bw <= 1.0 or bh <= 1.0:
                return

            if self._active_handle == "marquee":
                self._apply_marquee_drag(pos)

            elif self._active_handle == "inside":
                # ACR-style Image Dragging: Mouse drags the photo underneath!
                # The crop frame on screen STAYS ROCK-SOLID at self._stationary_crop_rect!
                # The canvas pans the photo by delta_px!
                delta_px = pos - self._drag_start_pos
                cw = float(max(1, self.canvas.width()))
                ch = float(max(1, self.canvas.height()))

                pan_dx = (delta_px.x() / cw) * 2.0
                pan_dy = -(delta_px.y() / ch) * 2.0

                target_pan_x = self._drag_start_pan[0] + pan_dx
                target_pan_y = self._drag_start_pan[1] + pan_dy

                self.canvas.pan_x = target_pan_x
                self.canvas.pan_y = target_pan_y

                bx, by, bw, bh = self._get_screen_bounds()
                if bw > 1.0 and bh > 1.0:
                    S = self._stationary_crop_rect
                    target_crop_x = (S.left() - bx) / bw
                    target_crop_y = (S.top() - by) / bh
                    target_crop_w = S.width() / bw
                    target_crop_h = S.height() / bh

                    hw = target_crop_w * 0.5
                    hh = target_crop_h * 0.5
                    cx = target_crop_x + hw
                    cy = target_crop_y + hh

                    if abs(self.crop_angle) < 0.01:
                        clamped_crop_x = max(0.0, min(1.0 - target_crop_w, target_crop_x))
                        clamped_crop_y = max(0.0, min(1.0 - target_crop_h, target_crop_y))
                    else:
                        img_w, img_h = self._get_effective_image_size()
                        A = float(img_w) / float(img_h) if img_h > 0 else 1.5

                        rad = math.radians(self.crop_angle)
                        ca = math.cos(rad)
                        sa = math.sin(-rad)

                        # Rotated extents
                        X_ext = target_crop_w * abs(ca) + target_crop_h * (abs(sa) / A)
                        Y_ext = target_crop_w * (A * abs(sa)) + target_crop_h * abs(ca)

                        M_x = max(0.0, (1.0 - X_ext) * 0.5)
                        M_y = max(0.0, (1.0 - Y_ext) * 0.5)

                        uc = cx - 0.5
                        vc = cy - 0.5
                        u_prime = uc * ca - vc * (sa / A)
                        v_prime = uc * A * sa + vc * ca

                        u_clamped = max(-M_x, min(M_x, u_prime))
                        v_clamped = max(-M_y, min(M_y, v_prime))

                        uc_new = u_clamped * ca + v_clamped * (sa / A)
                        vc_new = -u_clamped * A * sa + v_clamped * ca

                        clamped_cx = 0.5 + uc_new
                        clamped_cy = 0.5 + vc_new

                        clamped_crop_x = clamped_cx - hw
                        clamped_crop_y = clamped_cy - hh

                    if abs(clamped_crop_x - target_crop_x) > 1e-5:
                        excess_px_x = (clamped_crop_x - target_crop_x) * bw
                        self.canvas.pan_x -= (excess_px_x / cw) * 2.0

                    if abs(clamped_crop_y - target_crop_y) > 1e-5:
                        excess_px_y = (clamped_crop_y - target_crop_y) * bh
                        self.canvas.pan_y += (excess_px_y / ch) * 2.0

                    self.crop_x = clamped_crop_x
                    self.crop_y = clamped_crop_y
                    self.crop_w = target_crop_w
                    self.crop_h = target_crop_h

            elif self._active_handle == "rotate":
                # Rotate around center: Clockwise mouse drag produces positive angle, rotating image clockwise!
                bx, by, bw, bh = self._get_screen_bounds()
                img_center = QPointF(bx + bw * 0.5, by + bh * 0.5)
                v_prev = getattr(self, "_drag_last_pos", self._drag_start_pos) - img_center
                v_curr = pos - img_center
                ang_prev = math.degrees(math.atan2(v_prev.y(), v_prev.x()))
                ang_curr = math.degrees(math.atan2(v_curr.y(), v_curr.x()))
                d_ang = (ang_curr - ang_prev + 180.0) % 360.0 - 180.0
                if not hasattr(self, "_drag_accum_angle"):
                    self._drag_accum_angle = float(self.crop_angle)
                self._drag_accum_angle += d_ang
                self._drag_last_pos = pos
                new_angle = max(-45.0, min(45.0, round(self._drag_accum_angle, 1)))
                self.crop_angle = new_angle
                self._clamp_to_rotated_bounds()
                self.setCursor(self._get_rotate_cursor_for_pos(pos))

            else:
                dx = (pos.x() - self._drag_start_pos.x()) / bw
                dy = (pos.y() - self._drag_start_pos.y()) / bh
                self._apply_handle_drag(self._active_handle, dx, dy)

            # Live canvas update
            if hasattr(self.canvas, "params") and isinstance(self.canvas.params, dict):
                crop_p = self.canvas.params.setdefault("crop", {})
                crop_p["x"] = self.crop_x
                crop_p["y"] = self.crop_y
                crop_p["w"] = self.crop_w
                crop_p["h"] = self.crop_h
                crop_p["angle"] = self.crop_angle
                self.canvas.update()

            self.update()
            self.cropChanged.emit(self.get_crop_dict())
            return

        # Hover Cursor Updates
        if getattr(self, "_space_held", False):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return

        handle = self._hit_test(pos)
        if handle in ("tl", "br"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in ("tr", "bl"):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in ("t", "b"):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle in ("l", "r"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif handle == "inside":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif handle == "marquee":
            self.setCursor(self._crop_cursor)
        elif handle == "rotate":
            self.setCursor(self._get_rotate_cursor_for_pos(pos))
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if getattr(self, "_is_panning", False):
            self._is_panning = False
            self.canvas.mouseReleaseEvent(event)
            self.update()
            if getattr(self, "_space_held", False):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._push_undo_snapshot()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._active_handle:
            if self._active_handle == "marquee":
                self._marquee_mode_enabled = False
                if self.crop_w < 0.04 or self.crop_h < 0.04:
                    # Tiny accidental click, reset to full
                    self.crop_x = 0.0
                    self.crop_y = 0.0
                    self.crop_w = 1.0
                    self.crop_h = 1.0
                    self._user_crop_w = 1.0
                    self._user_crop_h = 1.0

            self._active_handle = None
            self._is_rotating = False
            self.update()
            self.cropChanged.emit(self.get_crop_dict())
            self._push_undo_snapshot()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Prevent accidental crop commits on double-clicking inside the crop box;
        # user explicitly confirms via Enter key or Apply button.
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        # Strict image coordinate locking: forward wheel zoom to canvas viewport
        self.canvas.wheelEvent(event)
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.apply_crop()
            event.accept()
        elif key == Qt.Key.Key_Escape:
            self.cancel_crop()
            event.accept()
        elif key == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
            event.accept()
        elif key == Qt.Key.Key_Y and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.redo()
            event.accept()
        elif key == Qt.Key.Key_V:
            if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                self.cycle_grid_type()
            elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.cycle_grid_orientation()
            else:
                self.toggle_show_guides()
            event.accept()
        elif key == Qt.Key.Key_O:
            self.cycle_grid_type()
            event.accept()
        elif key == Qt.Key.Key_X:
            self.flip_aspect_ratio()
            event.accept()
        elif key == Qt.Key.Key_R:
            if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and (event.modifiers() & Qt.KeyboardModifier.AltModifier):
                self.reset_to_initial_state()
            elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.reset_crop()
            else:
                self.reset_crop()
            event.accept()
        elif key == Qt.Key.Key_C or key == Qt.Key.Key_M:
            self.enable_marquee_mode()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def contextMenuEvent(self, event):
        from ui.canvas_viewport import get_darkroom_menu_style

        menu = QMenu(self)
        menu.setStyleSheet(get_darkroom_menu_style())

        # 1. Reset options
        act_reset = menu.addAction("重置裁切")
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.triggered.connect(self.reset_crop)

        act_reset_all = menu.addAction("重置为打开时状态")
        act_reset_all.setShortcut(QKeySequence("Ctrl+Alt+R"))
        act_reset_all.triggered.connect(self.reset_to_initial_state)

        act_clear = menu.addAction("清除裁切")
        act_clear.setIcon(get_icon("crop.png"))
        act_clear.triggered.connect(self.clear_crop_completely)

        menu.addSeparator()

        # 2. Aspect Ratio Presets
        menu_presets = menu.addMenu("画幅预设")
        grp_ratio = QActionGroup(self)
        for label, mode in ASPECT_RATIOS:
            act_r = menu_presets.addAction(label)
            act_r.setCheckable(True)
            if self.aspect_ratio_mode == mode:
                act_r.setChecked(True)
            act_r.triggered.connect(lambda checked=False, m=mode: self.set_aspect_ratio_mode(m))
            grp_ratio.addAction(act_r)

        menu.addSeparator()

        # 3. Swap Aspect Ratio
        act_swap = menu.addAction("调换长宽比")
        act_swap.setShortcut(QKeySequence("X"))
        act_swap.triggered.connect(self.flip_aspect_ratio)

        menu.addSeparator()

        # 4. Mask Opacity
        menu_mask = menu.addMenu("裁切蒙版")
        grp_mask = QActionGroup(self)
        cur_alpha = getattr(self, "mask_alpha", 160)
        mask_options = [
            ("100% 不透明度", 255),
            ("80% 不透明度", 204),
            ("50% 不透明度", 128),
            ("20% 不透明度", 51),
        ]
        for m_lbl, m_alpha in mask_options:
            act_m = menu_mask.addAction(m_lbl)
            act_m.setCheckable(True)
            if abs(cur_alpha - m_alpha) <= 25:
                act_m.setChecked(True)
            act_m.triggered.connect(lambda checked=False, a=m_alpha: self.set_mask_opacity(a))
            grp_mask.addAction(act_m)

        # 5. Show Overlay / Guides
        act_show_guides = menu.addAction("显示叠加")
        act_show_guides.setCheckable(True)
        act_show_guides.setChecked(getattr(self, "show_guides", True))
        act_show_guides.setShortcut(QKeySequence("V"))
        act_show_guides.triggered.connect(self.toggle_show_guides)

        # 6. Overlay Styles
        menu_style = menu.addMenu("叠加样式")
        grp_grid = QActionGroup(self)
        for g_label, g_val in GRID_TYPES:
            act_g = menu_style.addAction(g_label)
            act_g.setCheckable(True)
            if self.grid_type == g_val:
                act_g.setChecked(True)
            act_g.triggered.connect(lambda checked=False, g=g_val: self.set_grid_type(g))
            grp_grid.addAction(act_g)

        menu_style.addSeparator()
        act_cycle = menu_style.addAction("循环切换裁剪重叠层")
        act_cycle.setShortcut(QKeySequence("Alt+V"))
        act_cycle.triggered.connect(self.cycle_grid_type)

        menu.exec(event.globalPos())
        event.accept()

    # -------------------------------------------------------------
    # Rendering (Dim Mask, Leveling Grid, Hairlines, Brackets, Badges)
    # -------------------------------------------------------------
    def paintEvent(self, event):
        rect = self._get_crop_screen_rect()
        if rect.width() <= 2 or rect.height() <= 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Dim Mask over entire canvas viewport, hole-punched at the crop rectangle
        dim_path = QPainterPath()
        dim_path.setFillRule(Qt.FillRule.OddEvenFill)
        dim_path.addRect(QRectF(self.rect()))
        dim_path.addRect(rect)

        painter.setPen(Qt.PenStyle.NoPen)
        mask_a = getattr(self, "mask_alpha", 160)
        painter.setBrush(QBrush(QColor(0, 0, 0, mask_a)))
        painter.drawPath(dim_path)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # 2. Reference & Leveling Grids
        inner_rect = rect

        draw_guides = getattr(self, "show_guides", True) or self._is_rotating
        # When crop box is shrunk small (< 64px), suppress interior guide lines to avoid visual clutter and overlapping artifacts
        if inner_rect.width() < 64.0 or inner_rect.height() < 64.0:
            draw_guides = False

        if draw_guides:
            if self._is_rotating or self.grid_type == 6:
                # High-density square millimeter leveling grid during live rotation
                step = 28.0
                cx = inner_rect.center().x()
                cy = inner_rect.center().y()

                l_lim = inner_rect.left() + 2.0
                r_lim = inner_rect.right() - 2.0
                t_lim = inner_rect.top()
                b_lim = inner_rect.bottom()

                # Vertical square lines
                if l_lim <= cx <= r_lim:
                    self._draw_guide_line(painter, QPointF(cx, t_lim), QPointF(cx, b_lim), is_fine=True)
                gx = cx + step
                while gx <= r_lim:
                    self._draw_guide_line(painter, QPointF(gx, t_lim), QPointF(gx, b_lim), is_fine=True)
                    gx += step
                gx = cx - step
                while gx >= l_lim:
                    self._draw_guide_line(painter, QPointF(gx, t_lim), QPointF(gx, b_lim), is_fine=True)
                    gx -= step

                # Horizontal square lines
                t_edge = inner_rect.top() + 2.0
                b_edge = inner_rect.bottom() - 2.0
                l_edge = inner_rect.left()
                r_edge = inner_rect.right()

                if t_edge <= cy <= b_edge:
                    self._draw_guide_line(painter, QPointF(l_edge, cy), QPointF(r_edge, cy), is_fine=True)
                gy = cy + step
                while gy <= b_edge:
                    self._draw_guide_line(painter, QPointF(l_edge, gy), QPointF(r_edge, gy), is_fine=True)
                    gy += step
                gy = cy - step
                while gy >= t_edge:
                    self._draw_guide_line(painter, QPointF(l_edge, gy), QPointF(r_edge, gy), is_fine=True)
                    gy -= step

            elif self.grid_type == 0:
                # 三分法则 (Rule of thirds)
                x1 = rect.left() + rect.width() / 3.0
                x2 = rect.left() + rect.width() * 2.0 / 3.0
                y1 = rect.top() + rect.height() / 3.0
                y2 = rect.top() + rect.height() * 2.0 / 3.0
                self._draw_guide_line(painter, QPointF(x1, inner_rect.top()), QPointF(x1, inner_rect.bottom()))
                self._draw_guide_line(painter, QPointF(x2, inner_rect.top()), QPointF(x2, inner_rect.bottom()))
                self._draw_guide_line(painter, QPointF(inner_rect.left(), y1), QPointF(inner_rect.right(), y1))
                self._draw_guide_line(painter, QPointF(inner_rect.left(), y2), QPointF(inner_rect.right(), y2))

            elif self.grid_type == 1:
                # 黄金比例 (Golden ratio / Phi Grid)
                phi = 0.61803398875
                x1 = rect.left() + rect.width() * (1.0 - phi)
                x2 = rect.left() + rect.width() * phi
                y1 = rect.top() + rect.height() * (1.0 - phi)
                y2 = rect.top() + rect.height() * phi
                self._draw_guide_line(painter, QPointF(x1, inner_rect.top()), QPointF(x1, inner_rect.bottom()))
                self._draw_guide_line(painter, QPointF(x2, inner_rect.top()), QPointF(x2, inner_rect.bottom()))
                self._draw_guide_line(painter, QPointF(inner_rect.left(), y1), QPointF(inner_rect.right(), y1))
                self._draw_guide_line(painter, QPointF(inner_rect.left(), y2), QPointF(inner_rect.right(), y2))

            elif self.grid_type == 2:
                # 对角 (Diagonal)
                self._draw_guide_line(painter, inner_rect.topLeft(), inner_rect.bottomRight())
                self._draw_guide_line(painter, inner_rect.topRight(), inner_rect.bottomLeft())

            elif self.grid_type == 3:
                # 居中十字 (Center crosshair)
                cx = inner_rect.center().x()
                cy = inner_rect.center().y()
                self._draw_guide_line(painter, QPointF(cx, inner_rect.top()), QPointF(cx, inner_rect.bottom()))
                self._draw_guide_line(painter, QPointF(inner_rect.left(), cy), QPointF(inner_rect.right(), cy))

            elif self.grid_type == 4:
                # 三角形 (Golden Triangle)
                diag_a = inner_rect.topLeft()
                diag_b = inner_rect.bottomRight()
                self._draw_guide_line(painter, diag_a, diag_b)
                w = inner_rect.width()
                h = inner_rect.height()
                denom = max(1.0, w * w + h * h)
                t1 = (w * w) / denom
                p1 = QPointF(diag_a.x() + t1 * w, diag_a.y() + t1 * h)
                self._draw_guide_line(painter, inner_rect.topRight(), p1)
                t2 = (h * h) / denom
                p2 = QPointF(diag_a.x() + t2 * w, diag_a.y() + t2 * h)
                self._draw_guide_line(painter, inner_rect.bottomLeft(), p2)

            elif self.grid_type == 5:
                # 黄金螺旋 (Golden Spiral / Fibonacci Spiral)
                ori = getattr(self, "grid_orientation", 0) % 4
                gx, gy = inner_rect.left(), inner_rect.top()
                gw, gh = inner_rect.width(), inner_rect.height()

                def map_pt(u_norm: float, v_norm: float) -> QPointF:
                    if ori == 0:
                        u, v = u_norm, v_norm
                    elif ori == 1:
                        u, v = 1.0 - u_norm, v_norm
                    elif ori == 2:
                        u, v = 1.0 - u_norm, 1.0 - v_norm
                    else:
                        u, v = u_norm, 1.0 - v_norm
                    return QPointF(gx + u * gw, gy + v * gh)

                # Strict clipping to inner_rect so no curve or hairline ever overshoots bounds
                painter.save()
                painter.setClipRect(inner_rect)

                phi = (math.sqrt(5.0) - 1.0) / 2.0  # 0.61803398875
                u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0
                div_lines = []
                arcs = []

                for step in range(8):
                    W = u1 - u0
                    H = v1 - v0
                    phase = step % 4
                    if phase == 0:
                        cut = u0 + W * phi
                        div_lines.append((cut, v0, cut, v1))
                        arcs.append(('p0', cut, v1, cut - u0, v1 - v0))
                        u0 = cut
                    elif phase == 1:
                        cut = v0 + H * phi
                        div_lines.append((u0, cut, u1, cut))
                        arcs.append(('p1', u0, cut, u1 - u0, cut - v0))
                        v0 = cut
                    elif phase == 2:
                        cut = u1 - W * phi
                        div_lines.append((cut, v0, cut, v1))
                        arcs.append(('p2', cut, v0, u1 - cut, v1 - v0))
                        u1 = cut
                    elif phase == 3:
                        cut = v1 - H * phi
                        div_lines.append((u0, cut, u1, cut))
                        arcs.append(('p3', u1, cut, u1 - u0, v1 - cut))
                        v1 = cut

                # 1. Golden ratio frame subdivision lines
                for u_start, v_start, u_end, v_end in div_lines:
                    self._draw_guide_line(painter, map_pt(u_start, v_start), map_pt(u_end, v_end), is_fine=True)

                # 2. Continuous authentic Fibonacci quarter arcs
                spiral_path = QPainterPath()
                first = True
                for kind, acx, acy, rx, ry in arcs:
                    for i in range(21):
                        t = (math.pi / 2.0) * (i / 20.0)
                        if kind == 'p0':
                            u = acx - rx * math.cos(t)
                            v = acy - ry * math.sin(t)
                        elif kind == 'p1':
                            u = acx + rx * math.sin(t)
                            v = acy - ry * math.cos(t)
                        elif kind == 'p2':
                            u = acx + rx * math.cos(t)
                            v = acy + ry * math.sin(t)
                        elif kind == 'p3':
                            u = acx - rx * math.sin(t)
                            v = acy + ry * math.cos(t)
                        pt = map_pt(u, v)
                        if first:
                            spiral_path.moveTo(pt)
                            first = False
                        else:
                            spiral_path.lineTo(pt)

                self._draw_guide_path(painter, spiral_path)
                painter.restore()

        # 3. Inner crop boundary (solid line removed per ACR style; rectangular dim mask defines the clean crop edge)

        # 4. ACR-Style Floating Control Frame (4.0px outer gap, 4 corner L-brackets and 4 midpoint pill bars, NO connecting lines)
        h_gap = 4.0
        outer_rect = rect.adjusted(-h_gap, -h_gap, h_gap, h_gap)

        ext = 13.0
        mid_ext = 11.0

        # Clamp ext if box is very small so corners never collide or overlap
        ext_x = min(ext, max(3.0, (outer_rect.width() * 0.5) - 3.0))
        ext_y = min(ext, max(3.0, (outer_rect.height() * 0.5) - 3.0))

        handle_shadow = QPen(QColor(0, 0, 0, 140), 3.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        handle_pen = QPen(QColor(255, 255, 255, 255), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

        corners_path = QPainterPath()
        # TL
        ox, oy = outer_rect.left(), outer_rect.top()
        corners_path.moveTo(ox + ext_x, oy)
        corners_path.lineTo(ox, oy)
        corners_path.lineTo(ox, oy + ext_y)

        # TR
        ox, oy = outer_rect.right(), outer_rect.top()
        corners_path.moveTo(ox - ext_x, oy)
        corners_path.lineTo(ox, oy)
        corners_path.lineTo(ox, oy + ext_y)

        # BR
        ox, oy = outer_rect.right(), outer_rect.bottom()
        corners_path.moveTo(ox - ext_x, oy)
        corners_path.lineTo(ox, oy)
        corners_path.lineTo(ox, oy - ext_y)

        # BL
        ox, oy = outer_rect.left(), outer_rect.bottom()
        corners_path.moveTo(ox + ext_x, oy)
        corners_path.lineTo(ox, oy)
        corners_path.lineTo(ox, oy - ext_y)

        cx = outer_rect.center().x()
        cy = outer_rect.center().y()

        def _draw_handles(p_pen):
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(p_pen)
            painter.drawPath(corners_path)
            # Midpoint pill tabs with RoundCap - auto-hidden when box is small to prevent overlap/artifacts
            if outer_rect.width() >= 64.0:
                painter.drawLine(QPointF(cx - mid_ext, outer_rect.top()), QPointF(cx + mid_ext, outer_rect.top()))
                painter.drawLine(QPointF(cx - mid_ext, outer_rect.bottom()), QPointF(cx + mid_ext, outer_rect.bottom()))
            if outer_rect.height() >= 64.0:
                painter.drawLine(QPointF(outer_rect.left(), cy - mid_ext), QPointF(outer_rect.left(), cy + mid_ext))
                painter.drawLine(QPointF(outer_rect.right(), cy - mid_ext), QPointF(outer_rect.right(), cy + mid_ext))

        _draw_handles(handle_shadow)
        _draw_handles(handle_pen)




class CropSidebarWidget(QFrame):
    """ACR-style Right Adjustment Panel for Crop & Straighten mode.
    Directly embedded into the right sidebar parameter stack beneath the histogram.
    Built with the EXACT same AccordionSection cards and AdobeScrubSlider controls
    as the main parameter panel, guaranteeing 100% unified visual styling across both themes.
    """

    def __init__(self, overlay: CropOverlayWidget, parent=None):
        super().__init__(parent)
        self.overlay = overlay
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setObjectName("crop_sidebar_widget")

        self._is_updating = False
        self._init_ui()

        # Connect live sync from overlay
        self.overlay.cropChanged.connect(self.sync_from_overlay)
        theme_manager.themeChanged.connect(lambda _: self.apply_theme_style())

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        # 1. Section: 画幅比例 (AccordionSection)
        self.sec_ratio = AccordionSection("画幅比例")
        self.sec_ratio.resetClicked.connect(self._on_reset_ratio)

        ratio_row = QHBoxLayout()
        ratio_row.setContentsMargins(0, 0, 0, 0)
        ratio_row.setSpacing(6)

        self.combo_ratio = ComboBoxRow("画幅预设", ASPECT_RATIOS, default_val="original")
        self.combo_ratio.valueChanged.connect(self._on_ratio_changed)
        ratio_row.addWidget(self.combo_ratio, 1)

        self.btn_swap_aspect = QPushButton()
        self.btn_swap_aspect.setFixedSize(26, 26)
        self.btn_swap_aspect.setToolTip("调换长宽比 (快捷键 X)")
        self.btn_swap_aspect.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_swap_aspect.clicked.connect(self.overlay.flip_aspect_ratio)
        ratio_row.addWidget(self.btn_swap_aspect)

        self.sec_ratio.addLayout(ratio_row)
        layout.addWidget(self.sec_ratio)

        # 2. Section: 角度 (Item 6: Renamed to 角度, Item 7: Merged reference lines)
        self.sec_angle = AccordionSection("角度")
        self.sec_angle.resetClicked.connect(self._on_reset_angle)

        # Item 8: 4 quick transform action buttons
        trans_layout = QHBoxLayout()
        trans_layout.setContentsMargins(0, 2, 0, 4)
        trans_layout.setSpacing(6)

        self.btn_rot_left = QPushButton()
        self.btn_rot_left.setFixedSize(32, 28)
        self.btn_rot_left.setToolTip("向左旋转 90°")
        self.btn_rot_left.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rot_left.clicked.connect(self.overlay.rotate_90_ccw)
        trans_layout.addWidget(self.btn_rot_left)

        self.btn_rot_right = QPushButton()
        self.btn_rot_right.setFixedSize(32, 28)
        self.btn_rot_right.setToolTip("向右旋转 90°")
        self.btn_rot_right.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rot_right.clicked.connect(self.overlay.rotate_90_cw)
        trans_layout.addWidget(self.btn_rot_right)

        self.btn_flip_v = QPushButton()
        self.btn_flip_v.setFixedSize(32, 28)
        self.btn_flip_v.setToolTip("上下翻转")
        self.btn_flip_v.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_flip_v.clicked.connect(self.overlay.toggle_flip_v)
        trans_layout.addWidget(self.btn_flip_v)

        self.btn_flip_h = QPushButton()
        self.btn_flip_h.setFixedSize(32, 28)
        self.btn_flip_h.setToolTip("左右翻转")
        self.btn_flip_h.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_flip_h.clicked.connect(self.overlay.toggle_flip_h)
        trans_layout.addWidget(self.btn_flip_h)

        trans_layout.addStretch()
        self.sec_angle.addLayout(trans_layout)

        # Angle scrub slider
        self.slider_angle = AdobeScrubSlider(
            "角度", -45.0, 45.0, 0.0, step=0.1, unit="°", decimals=1,
            tooltip="微调画面倾斜度 (-45° ~ +45°)，双击回零"
        )
        self.slider_angle.valueChanged.connect(self._on_slider_angle_changed)
        self.slider_angle.sliderReleased.connect(self._on_angle_committed)
        self.sec_angle.addWidget(self.slider_angle)

        # Item 7: Merged reference lines inside 角度 card
        self.combo_grid = ComboBoxRow("参考线", GRID_TYPES, default_val=0)
        self.combo_grid.valueChanged.connect(self._on_grid_changed)
        self.sec_angle.addWidget(self.combo_grid)

        layout.addWidget(self.sec_angle)

        # 3. Action buttons on the same row (replacing tips box)
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 4)
        btn_layout.setSpacing(8)

        self.btn_apply = QPushButton("裁切")
        self.btn_apply.setFixedHeight(32)
        self.btn_apply.setToolTip("套用裁切参数并返回暗房面板 (快捷键 Enter)")
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #000000;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #d97706;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #fbbf24;
                border-color: #f59e0b;
            }
            QPushButton:pressed {
                background: #d97706;
            }
        """)
        self.btn_apply.clicked.connect(self.overlay.apply_crop)
        btn_layout.addWidget(self.btn_apply, 1)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFixedHeight(32)
        self.btn_cancel.setToolTip("放弃裁切修改 (快捷键 Esc)")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.overlay.cancel_crop)
        btn_layout.addWidget(self.btn_cancel, 1)

        layout.addLayout(btn_layout)
        layout.addStretch(1)

        self.apply_theme_style()

    def apply_theme_style(self):
        is_dark = theme_manager.is_dark()

        # Update AccordionSection cards
        if hasattr(self, 'sec_ratio'):
            self.sec_ratio.apply_theme()
        if hasattr(self, 'sec_angle'):
            self.sec_angle.apply_theme()

        # Update icons to match theme
        if hasattr(self, 'btn_swap_aspect'):
            self.btn_swap_aspect.setIcon(get_icon("aspect_swap.png", is_dark=is_dark))
        if hasattr(self, 'btn_rot_left'):
            self.btn_rot_left.setIcon(get_icon("rotate_left.png", is_dark=is_dark))
        if hasattr(self, 'btn_rot_right'):
            self.btn_rot_right.setIcon(get_icon("rotate_right.png", is_dark=is_dark))
        if hasattr(self, 'btn_flip_v'):
            self.btn_flip_v.setIcon(get_icon("flip_v.png", is_dark=is_dark))
        if hasattr(self, 'btn_flip_h'):
            self.btn_flip_h.setIcon(get_icon("flip_h.png", is_dark=is_dark))

        if is_dark:
            btn_sec_style = """
                QPushButton {
                    background: #1c1d25;
                    color: #cbd5e1;
                    border: 1px solid #2a2c3a;
                    border-radius: 4px;
                    padding: 3px;
                }
                QPushButton:hover {
                    background: #262836;
                    border-color: #f59e0b;
                    color: #f59e0b;
                }
                QPushButton:pressed {
                    background: #14151b;
                }
            """
            cancel_style = """
                QPushButton {
                    background: #20222a;
                    color: #cbd5e1;
                    border: 1px solid #333647;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #2a2d39;
                    color: #f8fafc;
                    border-color: #4b5268;
                }
                QPushButton:pressed {
                    background: #181920;
                }
            """
        else:
            btn_sec_style = """
                QPushButton {
                    background: #f8fafc;
                    color: #334155;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    padding: 3px;
                }
                QPushButton:hover {
                    background: #f1f5f9;
                    border-color: #f59e0b;
                    color: #d97706;
                }
                QPushButton:pressed {
                    background: #e2e8f0;
                }
            """
            cancel_style = """
                QPushButton {
                    background: #ffffff;
                    color: #475569;
                    border: 1px solid #cbd5e1;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #f8fafc;
                    color: #0f172a;
                    border-color: #94a3b8;
                }
                QPushButton:pressed {
                    background: #f1f5f9;
                }
            """
        for btn in (getattr(self, 'btn_swap_aspect', None),
                    getattr(self, 'btn_rot_left', None),
                    getattr(self, 'btn_rot_right', None),
                    getattr(self, 'btn_flip_v', None),
                    getattr(self, 'btn_flip_h', None)):
            if btn:
                btn.setStyleSheet(btn_sec_style)

        if hasattr(self, 'btn_cancel'):
            self.btn_cancel.setStyleSheet(cancel_style)

        self.setStyleSheet("""
            QFrame#crop_sidebar_widget {
                background: transparent;
                border: none;
            }
        """)

    def _on_ratio_changed(self, val):
        if self._is_updating:
            return
        self.overlay.set_aspect_ratio_mode(val)

    def _on_reset_ratio(self):
        self.combo_ratio.set_value("original")
        self.overlay.aspect_ratio_mode = "original"
        img_w, img_h = self.overlay._get_effective_image_size()
        self.overlay.is_portrait_orientation = bool(img_w < img_h)
        self.overlay.crop_x = 0.0
        self.overlay.crop_y = 0.0
        self.overlay.crop_w = 1.0
        self.overlay.crop_h = 1.0
        self.overlay._user_crop_w = 1.0
        self.overlay._user_crop_h = 1.0
        if abs(self.overlay.crop_angle) > 0.01:
            self.overlay._clamp_to_rotated_bounds()
        if hasattr(self.overlay.canvas, "params") and isinstance(self.overlay.canvas.params, dict):
            self.overlay.canvas.params["crop"] = self.overlay.get_crop_dict()
            self.overlay.canvas.update()
        self.overlay.update()
        self.overlay.cropChanged.emit(self.overlay.get_crop_dict())
        self.overlay._push_undo_snapshot()

    def _on_slider_angle_changed(self, val):
        if self._is_updating:
            return
        self.overlay.set_crop_angle(float(val))

    def _on_angle_committed(self):
        if self._is_updating:
            return
        self.overlay._push_undo_snapshot()

    def _on_reset_angle(self):
        self.slider_angle.setValue(0.0)
        self.overlay.crop_angle = 0.0
        self.overlay.rot_90 = 0
        self.overlay.flip_h = False
        self.overlay.flip_v = False
        if hasattr(self.overlay.canvas, "params") and isinstance(self.overlay.canvas.params, dict):
            self.overlay.canvas.params["crop"] = self.overlay.get_crop_dict()
            self.overlay.canvas.update()
        self.overlay.update()
        self.overlay.cropChanged.emit(self.overlay.get_crop_dict())
        self.overlay._push_undo_snapshot()

    def _on_grid_changed(self, val):
        if self._is_updating:
            return
        self.overlay.set_grid_type(int(val))

    def sync_from_overlay(self, crop_dict: dict):
        self._is_updating = True
        try:
            # Sync ratio combo
            mode = crop_dict.get("aspect_ratio", "original")
            self.combo_ratio.set_value(mode)

            # Sync angle slider
            angle = float(crop_dict.get("angle", 0.0))
            self.slider_angle.setValue(angle)

            # Sync grid combo
            gt = getattr(self.overlay, "grid_type", 0)
            self.combo_grid.set_value(gt)

            # Item 3: Reset buttons light up reliably
            cx = float(crop_dict.get("x", 0.0))
            cy = float(crop_dict.get("y", 0.0))
            cw = float(crop_dict.get("w", 1.0))
            ch = float(crop_dict.get("h", 1.0))
            is_port = bool(crop_dict.get("is_portrait", False))
            is_box_modified = (abs(cx) > 0.001 or abs(cy) > 0.001 or abs(cw - 1.0) > 0.001 or abs(ch - 1.0) > 0.001)
            is_ratio_dirty = (mode != "original") or is_port or is_box_modified
            self.sec_ratio.set_reset_enabled(is_ratio_dirty)

            rot_90 = int(crop_dict.get("rot_90", 0)) % 360
            flip_h = bool(crop_dict.get("flip_h", False))
            flip_v = bool(crop_dict.get("flip_v", False))
            is_angle_dirty = (abs(angle) > 0.05) or (rot_90 != 0) or flip_h or flip_v
            self.sec_angle.set_reset_enabled(is_angle_dirty)

        finally:
            self._is_updating = False

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.overlay.apply_crop()
            event.accept()
        elif key == Qt.Key.Key_Escape:
            self.overlay.cancel_crop()
            event.accept()
        elif key == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.overlay.redo()
            else:
                self.overlay.undo()
            event.accept()
        elif key == Qt.Key.Key_Y and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.overlay.redo()
            event.accept()
        elif key == Qt.Key.Key_R:
            if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and (event.modifiers() & Qt.KeyboardModifier.AltModifier):
                self.overlay.reset_to_initial_state()
                event.accept()
            elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.overlay.reset_crop()
                event.accept()
        elif key == Qt.Key.Key_X:
            self.overlay.flip_aspect_ratio()
            event.accept()
        elif key == Qt.Key.Key_O:
            self.overlay.grid_type = (self.overlay.grid_type + 1) % len(GRID_TYPES)
            self.overlay.update()
            self.overlay.cropChanged.emit(self.overlay.get_crop_dict())
            event.accept()
        else:
            super().keyPressEvent(event)
