import enum
import os
import sys
from time import sleep

import numpy as np
# noinspection PyUnresolvedReferences
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QLibraryInfo, pyqtSignal, QThread, QObject
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QHBoxLayout

from counts_widget import CountsWidget
from game_logic import KiitosGame
from tracking_and_detection import NewCardDetector

# This is a problem caused by OpenCV
# https://stackoverflow.com/questions/68417682/
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

IMG_W = 640
IMG_H = 480


def array_to_qimg(annotated_image):
    return QImage(annotated_image.data, IMG_W, IMG_H, 3 * IMG_W, QImage.Format.Format_RGB888)


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


class DraggableBbox:
    """ A non-widget class to help coordinate the 4 draggable handles"""

    def __init__(self, parent):
        self.parent = parent
        self.bbox = np.array([
            [40, 100],
            [610, 400],
        ])

        self.left = BBoxHandle(parent, HandleDirection.HORIZONTAL)
        self.left.position.connect(self.on_left_position_changed)
        self.top = BBoxHandle(parent, HandleDirection.VERTICAL)
        self.top.position.connect(self.on_top_position_changed)
        self.right = BBoxHandle(parent, HandleDirection.HORIZONTAL)
        self.right.position.connect(self.on_right_position_changed)
        self.bottom = BBoxHandle(parent, HandleDirection.VERTICAL)
        self.bottom.position.connect(self.on_bottom_position_changed)

        self.on_left_position_changed(self.bbox[0, 0])
        self.on_top_position_changed(self.bbox[0, 1])
        self.on_right_position_changed(self.bbox[1, 0])
        self.on_bottom_position_changed(self.bbox[1, 1])

    def on_top_position_changed(self, position):
        self.bbox[0, 1] = position
        self.parent.update()

    def on_left_position_changed(self, position):
        self.bbox[0, 0] = position
        self.parent.update()

    def on_right_position_changed(self, position):
        self.bbox[1, 0] = position
        self.parent.update()

    def on_bottom_position_changed(self, position):
        self.bbox[1, 1] = position
        self.parent.update()

    def get_rect_params(self):
        w = self.bbox[1, 0] - self.bbox[0, 0]
        h = self.bbox[1, 1] - self.bbox[0, 1]
        x0 = self.bbox[0, 0]
        y0 = self.bbox[0, 1]

        return x0, y0, w, h


class ImageWidget(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.update_pixmap(array_to_qimg(np.ones([IMG_H, IMG_W, 3], dtype=np.uint8) * 128))
        self.draggable_bbox = DraggableBbox(self)

    def update_pixmap(self, q_img: QImage):
        pixmap = QPixmap(q_img)
        # Calling setPixmap will trigger a paintEvent which ensures the box is drawn on top of the image
        self.setPixmap(pixmap)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(3)
        pen.setColor(Qt.red)
        painter.setPen(pen)

        x0, y0, w, h = self.draggable_bbox.get_rect_params()

        # clip based on the size of the image
        # the order of these updates is important because we're overwriting values
        x1 = min(max(x0 + w, 0), self.width())
        y1 = min(max(y0 + h, 0), self.height())
        x0 = min(max(x0, 0), self.width())
        y0 = min(max(y0, 0), self.height())
        h = y1 - y0
        w = x1 - x0

        center_x = int(x0 + w / 2)
        center_y = int(y0 + h / 2)
        self.draggable_bbox.left.move(x0 - BBoxHandle.HALF_SIZE, center_y - BBoxHandle.HALF_SIZE)
        self.draggable_bbox.top.move(center_x - BBoxHandle.HALF_SIZE, y0 - BBoxHandle.HALF_SIZE)
        self.draggable_bbox.right.move(x1 - BBoxHandle.HALF_SIZE, center_y - BBoxHandle.HALF_SIZE)
        self.draggable_bbox.bottom.move(center_x - BBoxHandle.HALF_SIZE, y1 - BBoxHandle.HALF_SIZE)

        painter.drawRect(x0, y0, w, h)
        painter.end()


class CardDetectorWidget(QObject):
    new_detection_str = pyqtSignal(str)
    new_detection_img = pyqtSignal(QImage)

    def __init__(self, img_widget: ImageWidget, game: KiitosGame):
        super().__init__()
        self.game = game
        self.img_widget = img_widget
        self.ncd = NewCardDetector()
        self.done = False

    def run(self):
        while not self.done:
            new_card, annotated_image = self.ncd.detect(self.img_widget.draggable_bbox.bbox)
            q_img = array_to_qimg(annotated_image)
            self.new_detection_img.emit(q_img)
            if new_card is not None and self.game.is_valid_card(new_card):
                # pygame.mixer.Sound.play(notification_sound)
                # pygame.time.set_timer(UNDO_EXPIRED_EVENT, UNDO_EXPIRE_MILLIS, loops=1)
                self.game.on_new_valid_card(new_card)
        self.ncd.cap_manager.stop()


class KiitosUi(QtWidgets.QMainWindow):

    def __init__(self):
        super(KiitosUi, self).__init__()
        uic.loadUi('game.ui', self)

        self.game = KiitosGame()

        self.counts_widget = CountsWidget(self, self.game)
        self.img_widget = ImageWidget(self)
        self.detector_widget = CardDetectorWidget(self.img_widget, self.game)
        self.detector_widget.new_detection_img.connect(self.img_widget.update_pixmap)
        self.center_layout = self.findChild(QHBoxLayout, "center_layout")
        self.center_layout.addWidget(self.counts_widget)
        self.center_layout.addWidget(self.img_widget)

        self.camera_thread = QThread()
        self.detector_widget.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.detector_widget.run)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.start()

        self.show()

    def closeEvent(self, event):
        self.detector_widget.done = True
        while self.camera_thread.wait(10):
            pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = KiitosUi()
    app.exec()
