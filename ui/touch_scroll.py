"""touch_scroll.py - one-finger drag scrolls, a tap still clicks.

Qt on Windows gives a scroll area two ways in: the wheel and the scrollbar.  A touch panel has
neither, so a finger could only grab the 6 px scrollbar.  This filter accepts the touch sequence
on the viewport a finger lands on and turns it into the same scroll the wheel produces, with the
same inertia; a touch that never leaves the slop is replayed as a mouse click on the child under
it, so cards still select.

The filter is installed per scroll area and needs no cooperation from the content: the viewport
accepts touch events, and Qt sends a touch to the nearest ancestor that accepts it when the child
under the finger does not.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox,
    QLineEdit, QTextEdit,
)

#: Widgets that want the finger themselves: dragging a slider, typing, picking from a combo.  A
#: touch that lands on one is left to Qt's own touch-to-mouse translation.
_CONTROL_TYPES = (QAbstractSlider, QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit, QTextEdit)

SLOP_PX = 10.0          # a finger may wander this far and still be a tap
LONG_PRESS_MS = 550     # held still this long, the finger asks for the context menu
TAP_MS = 600            # and must lift within this long
FRICTION = 0.90         # per 16 ms tick, matching the wheel's momentum
STOP_VELOCITY = 0.4     # px/tick below which the fling is over
MAX_VELOCITY = 60.0     # px/tick; a fast flick cannot launch the list into orbit

_INSTALLED: set[int] = set()
_TOUCH_AREAS: set[int] = set()      # areas whose viewport took the touch ourselves


def _in_pan_area(widget) -> bool:
    """Whether ``widget`` sits inside a scroll area a finger drives, and not on a control."""
    node = widget
    while node is not None:
        if isinstance(node, _CONTROL_TYPES):
            return False
        if id(node) in _TOUCH_AREAS:
            return True
        node = node.parentWidget()
    return False


class _SynthesizedClickGuard(QObject):
    """Drop the mouse press Qt invents from a touch inside a pan area.

    A touch on a card is not accepted by the card, so Qt replays it as a mouse press on the card
    -- which selects it before the finger has moved far enough to decide between a tap and a
    drag.  The pan filter replays the tap itself once it knows, so the invented one is dropped.
    Events from a real mouse carry `MouseEventNotSynthesized` and pass straight through.
    """

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        kind = event.type()
        if kind not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
                        QEvent.Type.MouseButtonDblClick):
            return False
        try:
            if event.source() != Qt.MouseEventSource.MouseEventSynthesizedBySystem:
                return False
            if event.button() == Qt.MouseButton.RightButton:
                # A press-and-hold arrives from the system as a right click; that is a menu,
                # not a selection, and it must reach the card.
                return False
        except AttributeError:
            return False
        app = QApplication.instance()
        if app is None:
            return False
        return _in_pan_area(app.widgetAt(event.globalPosition().toPoint()))


def guard_touch_clicks(app) -> None:
    """Install the guard once, on the application, after it exists."""
    if getattr(app, "_touch_click_guard", None) is not None:
        return
    guard = _SynthesizedClickGuard(app)
    app.installEventFilter(guard)
    app._touch_click_guard = guard


class TouchPanFilter(QObject):
    """Scrolls its area on a one-finger drag, and clicks the child under a tap."""

    def __init__(self, area: QAbstractScrollArea, parent: QObject | None = None) -> None:
        super().__init__(parent or area)
        self._area = area
        self._viewport = area.viewport()
        if self._viewport is None:
            return
        self._viewport.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self._viewport.installEventFilter(self)

        self._press: QPointF | None = None
        self._last: QPointF | None = None
        self._press_ms = 0
        self._dragging = False
        self._horizontal = False
        self._velocity = 0.0
        self._position = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        # A finger held still is a press-and-hold: the menu it used to raise came from the touch
        # being turned into a right click, which the pan filter now swallows.  Raise it here.
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.setInterval(LONG_PRESS_MS)
        self._long_press_timer.timeout.connect(self._long_press)
        self._long_pressed = False

    # -- touch ------------------------------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if obj is not self._viewport:
            return super().eventFilter(obj, event)
        kind = event.type()
        if kind not in (QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd,
                        QEvent.Type.TouchCancel):
            return super().eventFilter(obj, event)
        points = event.points()
        if not points:
            return True
        point = points[0]
        position = QPointF(point.position())

        if kind == QEvent.Type.TouchBegin:
            self._stop()
            if self._control_at(position):
                # A slider or a field under the finger: the touch is not ours to scroll.
                self._press = None
                return False
            self._press = position
            self._last = position
            self._press_ms = event.timestamp()
            self._dragging = False
            self._long_pressed = False
            self._horizontal = self._scrolls_horizontally()
            self._long_press_timer.start()
            return True

        if kind == QEvent.Type.TouchUpdate:
            if self._press is None or self._last is None:
                return True
            if not self._dragging:
                dx = abs(position.x() - self._press.x())
                dy = abs(position.y() - self._press.y())
                if max(dx, dy) < SLOP_PX:
                    return True
                self._dragging = True
                self._velocity = 0.0
                self._long_press_timer.stop()
            delta = (position.x() - self._last.x()) if self._horizontal \
                else (position.y() - self._last.y())
            self._scroll_by(-delta)
            # Events arrive at the panel's own rate; treat one as a frame of the fling.
            self._velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, -delta))
            self._last = position
            return True

        # TouchEnd / TouchCancel
        self._long_press_timer.stop()
        was_drag = self._dragging
        was_long = self._long_pressed
        self._long_pressed = False
        press = self._press
        self._press = None
        self._last = None
        self._dragging = False
        if was_long:
            return True
        if not was_drag:
            if kind == QEvent.Type.TouchEnd and press is not None \
                    and event.timestamp() - self._press_ms <= TAP_MS:
                self._click_at(press)
            return True
        if abs(self._velocity) > STOP_VELOCITY:
            self._position = float(self._bar().value())
            self._timer.start()
        return True

    def _long_press(self) -> None:
        if self._press is None or self._dragging:
            return
        self._long_pressed = True
        target = self._viewport.childAt(self._press.toPoint())
        if target is None:
            return
        local = target.mapFrom(self._viewport, self._press.toPoint())
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, local,
                                  target.mapToGlobal(local))
        QApplication.sendEvent(target, event)
        # The gesture is spent: a finger that keeps travelling now must not scroll behind the
        # menu it just opened.
        self._press = None
        self._last = None

    def _control_at(self, position: QPointF) -> bool:
        node = self._viewport.childAt(position.toPoint())
        while node is not None and node is not self._viewport:
            if isinstance(node, _CONTROL_TYPES):
                return True
            node = node.parentWidget()
        return False

    # -- scrolling --------------------------------------------------------------------------
    def _bar(self):
        return (self._area.horizontalScrollBar() if self._horizontal
                else self._area.verticalScrollBar())

    def _scrolls_horizontally(self) -> bool:
        vertical = self._area.verticalScrollBar()
        horizontal = self._area.horizontalScrollBar()
        can_v = vertical is not None and vertical.maximum() > vertical.minimum()
        can_h = horizontal is not None and horizontal.maximum() > horizontal.minimum()
        return can_h and not can_v

    def _scroll_by(self, amount: float) -> None:
        bar = self._bar()
        if bar is None:
            return
        target = int(round(bar.value() + amount))
        bar.setValue(max(bar.minimum(), min(bar.maximum(), target)))

    def _tick(self) -> None:
        bar = self._bar()
        if bar is None or bar.maximum() <= bar.minimum():
            self._stop()
            return
        self._position += self._velocity
        if self._position <= bar.minimum():
            self._position = float(bar.minimum())
            self._velocity = 0.0
        elif self._position >= bar.maximum():
            self._position = float(bar.maximum())
            self._velocity = 0.0
        value = int(round(self._position))
        if bar.value() != value:
            bar.setValue(value)
        if self._velocity == 0.0:
            self._stop()
            return
        self._velocity *= FRICTION
        if abs(self._velocity) < STOP_VELOCITY:
            self._stop()

    def _stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self._velocity = 0.0

    # -- taps -------------------------------------------------------------------------------
    def _click_at(self, position: QPointF) -> None:
        """Replay a tap as a click on whatever child the finger is over."""
        point = QPoint(int(position.x()), int(position.y()))
        target = self._viewport.childAt(point)
        if target is None or not target.isEnabled():
            return
        local = target.mapFrom(self._viewport, point)
        global_pos = target.mapToGlobal(local)
        for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            event = QMouseEvent(kind, QPointF(local), QPointF(global_pos),
                                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(target, event)


def install_touch_scroll(area: QAbstractScrollArea | None) -> TouchPanFilter | None:
    """Give one scroll area finger scrolling.  Safe to call twice for the same widget."""
    if area is None or area.viewport() is None:
        return None
    key = id(area)
    if key in _INSTALLED:
        return None
    _INSTALLED.add(key)
    _TOUCH_AREAS.add(key)
    return TouchPanFilter(area)
