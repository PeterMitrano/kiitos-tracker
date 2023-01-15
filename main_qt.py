import enum
import os
import sys

import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QLibraryInfo, QTimer, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtWidgets import QLabel, QWidget

from game_logic import KiitosGame
from tracking_and_detection import NewCardDetector

# This is a problem caused by OpenCV
# https://stackoverflow.com/questions/68417682/
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

IMG_W = 640
IMG_H = 480


class HandleDirection(enum.Enum):
    VERTICAL = enum.auto()
    HORIZONTAL = enum.auto()


class BBoxHandle(QLabel):
    SIZE = 25
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
        dest.setX(max(0, dest.x()))
        dest.setY(max(0, dest.y()))
        position = dest.y() if self.direction == HandleDirection.VERTICAL else dest.x()
        self.position.emit(position)


class DraggableBbox(QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.workspace_bbox = np.array([
            [40, 100],
            [610, 400],
        ])

        self.left = BBoxHandle(parent, HandleDirection.HORIZONTAL)
        self.left.position.connect(self.on_left_position_changed)
        self.top = BBoxHandle(parent, HandleDirection.VERTICAL)
        self.top.position.connect(self.on_top_position_changed)

        self.on_left_position_changed(self.workspace_bbox[0, 0])
        self.on_top_position_changed(self.workspace_bbox[0, 1])

    def draw_box(self):
        painter = QPainter(self.parent().pixmap())
        pen = QPen()
        pen.setWidth(3)
        pen.setColor(Qt.red)
        painter.setPen(pen)

        workspace_w = self.workspace_bbox[1, 0] - self.workspace_bbox[0, 0]
        workspace_h = self.workspace_bbox[1, 1] - self.workspace_bbox[0, 1]
        workspace_x0 = self.workspace_bbox[0, 0]
        workspace_y0 = self.workspace_bbox[0, 1]

        painter.drawRect(workspace_x0, workspace_y0, workspace_w, workspace_h)
        painter.end()

    def on_top_position_changed(self, position):
        center_x = int(self.workspace_bbox[0, 0] + (self.workspace_bbox[1, 0] - self.workspace_bbox[0, 0]) / 2)
        self.top.move(center_x, position)

    def on_left_position_changed(self, position):
        center_y = int(self.workspace_bbox[0, 1] + (self.workspace_bbox[1, 1] - self.workspace_bbox[0, 1]) / 2)
        self.left.move(position, center_y)


class KiitosUi(QtWidgets.QMainWindow):
    def __init__(self):
        super(KiitosUi, self).__init__()
        uic.loadUi('game.ui', self)

        self.game = KiitosGame()
        self.annotated_image = np.ones([IMG_H, IMG_W, 3], dtype=np.uint8) * 128
        self.ncd = NewCardDetector()

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(100)

        self.counts_label = self.findChild(QLabel, 'counts')
        canvas = QPixmap(400, 300)
        canvas.fill(Qt.white)
        self.counts_label.setPixmap(canvas)

        self.img_label = self.findChild(QLabel, 'img')
        self.img_label.resize(IMG_W, IMG_H)
        self.update_img_pixmap()

        self.bbox_widget = DraggableBbox(self.img_label)

        self.show()

    def update_camera(self):
        # new_card, self.annotated_image = self.ncd.detect(self.workspace_bbox)
        # if new_card is not None and self.game.is_valid_card(new_card):
        #     # pygame.mixer.Sound.play(notification_sound)
        #     # pygame.time.set_timer(UNDO_EXPIRED_EVENT, UNDO_EXPIRE_MILLIS, loops=1)
        #     self.game.on_new_valid_card(new_card)

        self.draw_board()
        self.update_img_pixmap()
        self.bbox_widget.draw_box()
        self.update()

    def update_img_pixmap(self):
        q_img = QImage(self.annotated_image.data, IMG_W, IMG_H, 3 * IMG_W, QImage.Format.Format_RGB888)
        pixmap = QPixmap(q_img)
        self.img_label.setPixmap(pixmap)

    def draw_board(self):
        painter = QPainter(self.counts_label.pixmap())
        pen = QPen()
        pen.setWidth(3)
        pen.setColor(QColor("#376F9F"))
        painter.setPen(pen)
        painter.drawRoundedRect(120, 120, 100, 100, 50, 50)
        painter.end()

    def closeEvent(self, event):
        self.ncd.cap_manager.stop()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = KiitosUi()
    app.exec()
