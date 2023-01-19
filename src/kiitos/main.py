import os
import sys
from datetime import datetime

from PIL import Image
# noinspection PyUnresolvedReferences
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QLibraryInfo, pyqtSignal, QThread, QObject, QUrl
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import QLabel, QSizePolicy

from counts_widget import CountsWidget
from draggable_bbox import BBoxHandle, DraggableBbox
from game_logic import KiitosGame
from tracking import NewCardDetector

# This is a problem caused by OpenCV
# https://stackoverflow.com/questions/68417682/
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

IMG_W = 640
IMG_H = 480


def array_to_qimg(annotated_image):
    return QImage(annotated_image.data, IMG_W, IMG_H, 3 * IMG_W, QImage.Format.Format_RGB888)


class ImageWidget(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        init_pixmap = QPixmap(IMG_W, IMG_H)
        init_pixmap.fill(Qt.gray)
        self.setPixmap(init_pixmap)
        self.draggable_bbox = DraggableBbox(self)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.resize(IMG_H, IMG_W)
        self.setStyleSheet("border: 1px solid black;")

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

        self.effect = QSoundEffect()
        self.effect.setSource(QUrl.fromLocalFile("notification.wav"))
        self.effect.setVolume(0.25)

    def run(self):
        while not self.done:
            new_card, annotated_image = self.ncd.detect(self.img_widget.draggable_bbox.bbox)
            q_img = array_to_qimg(annotated_image)
            self.new_detection_img.emit(q_img)
            if new_card is not None and self.game.is_valid_card(new_card):
                self.effect.play()
                # pygame.time.set_timer(UNDO_EXPIRED_EVENT, UNDO_EXPIRE_MILLIS, loops=1)
                self.new_detection_str.emit(new_card)
        self.ncd.cap_manager.stop()


class KiitosUi(QtWidgets.QMainWindow):

    def __init__(self):
        super(KiitosUi, self).__init__()
        uic.loadUi('src/kiitos/game.ui', self)

        self.game = KiitosGame()

        margin = 10
        self.setContentsMargins(margin, margin, margin, margin)

        self.action_capture.triggered.connect(self.save_last_frame)
        self.action_capture.setShortcut("Ctrl+S")
        self.counts_widget = CountsWidget(self, self.game)
        self.img_widget = ImageWidget(self)
        self.detector_widget = CardDetectorWidget(self.img_widget, self.game)
        self.detector_widget.new_detection_img.connect(self.img_widget.update_pixmap)
        self.detector_widget.new_detection_str.connect(self.game.on_new_valid_card)
        self.detector_widget.new_detection_str.connect(self.new_detection)
        self.center_layout.setAlignment(Qt.AlignTop)
        self.center_layout.addWidget(self.counts_widget)
        self.center_layout.addWidget(self.img_widget)

        self.camera_thread = QThread()
        self.detector_widget.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.detector_widget.run)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.start()

        self.show()

    def save_last_frame(self, event):
        pil_img = Image.fromarray(self.detector_widget.ncd.cap_manager.last_frame)
        pil_img = pil_img.rotate(180)
        path = f'saved_from_live/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
        print(f"Saving {path}")
        pil_img.save(path)

    def new_detection(self):
        self.counts_widget.update()

    def closeEvent(self, event):
        self.detector_widget.done = True
        while self.camera_thread.wait(10):
            pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = KiitosUi()
    app.exec()
