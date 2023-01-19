import enum
from dataclasses import dataclass

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel


class HandleDirection(enum.Enum):
    VERTICAL = enum.auto()
    HORIZONTAL = enum.auto()


class BBoxHandle(QLabel):
    SIZE = 25
    HALF_SIZE = int(SIZE / 2)
    V_ICON_PATH = "icons/vertical_handle.png"
    H_ICON_PATH = "icons/horizontal_handle.png"
    position = pyqtSignal(int)

    def __init__(self, parent, direction: HandleDirection):
        super().__init__(parent)
        self.direction = direction
        self.resize(BBoxHandle.SIZE, BBoxHandle.SIZE)
        self.offset = None
        self.icon = self.V_ICON_PATH if self.direction == HandleDirection.VERTICAL else self.H_ICON_PATH
        icon = QPixmap(self.icon)
        # resize icon to fit the label
        self.setPixmap(icon.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, event):
        self.offset = event.pos()

    def mouseMoveEvent(self, event):
        dest = self.mapToParent(event.pos() - self.offset)
        position = dest.y() if self.direction == HandleDirection.VERTICAL else dest.x()
        self.position.emit(position)


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float


class DraggableBbox:
    """ A non-widget class to help coordinate the 4 draggable handles"""

    def __init__(self, parent):
        self.parent = parent
        self.bbox = BBox(x0=2, y0=2, x1=636, y1=200)

        self.left = BBoxHandle(parent, HandleDirection.HORIZONTAL)
        self.left.position.connect(self.on_left_position_changed)
        self.top = BBoxHandle(parent, HandleDirection.VERTICAL)
        self.top.position.connect(self.on_top_position_changed)
        self.right = BBoxHandle(parent, HandleDirection.HORIZONTAL)
        self.right.position.connect(self.on_right_position_changed)
        self.bottom = BBoxHandle(parent, HandleDirection.VERTICAL)
        self.bottom.position.connect(self.on_bottom_position_changed)

        self.on_left_position_changed(self.bbox.x0)
        self.on_top_position_changed(self.bbox.y0)
        self.on_right_position_changed(self.bbox.x1)
        self.on_bottom_position_changed(self.bbox.y1)

    def on_left_position_changed(self, position):
        self.bbox.x0 = position
        self.parent.update()

    def on_top_position_changed(self, position):
        self.bbox.y0 = position
        self.parent.update()

    def on_right_position_changed(self, position):
        self.bbox.x1 = position
        self.parent.update()

    def on_bottom_position_changed(self, position):
        self.bbox.y1 = position
        self.parent.update()

    def get_rect_params(self):
        w = self.bbox.x1 - self.bbox.x0
        h = self.bbox.y1 - self.bbox.y0
        x0 = self.bbox.x0
        y0 = self.bbox.y0

        return x0, y0, w, h
