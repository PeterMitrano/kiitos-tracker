import enum
import os
import sys

import numpy as np
# noinspection PyUnresolvedReferences
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QLibraryInfo, pyqtSignal, QRectF
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QTextOption, QFont
from PyQt5.QtWidgets import QLabel, QHBoxLayout

from game_logic import KiitosGame

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
        self.annotated_image = np.ones([IMG_H, IMG_W, 3], dtype=np.uint8) * 128
        self.update_pixmap()
        self.draggable_bbox = DraggableBbox(self)

    def update_pixmap(self):
        q_img = QImage(self.annotated_image.data, IMG_W, IMG_H, 3 * IMG_W, QImage.Format.Format_RGB888)
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


class CountsWidget(QLabel):
    CARD_FONT_SIZE = 28

    def __init__(self, parent, game):
        super().__init__(parent)
        self.game = game
        self.setFixedSize(650, 750)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(3)
        blue = QColor("#262f4e")
        pen.setColor(blue)
        painter.setPen(pen)

        frame_padding = 50
        card_padding = 25
        width = 75
        height = 100
        n_rows = 5
        top_padding = 5
        left_padding = 5
        n_cols = 5
        frame_width = frame_padding + n_cols * width + (n_cols + 1) * card_padding
        frame_height = frame_padding + n_rows * height + (n_rows + 1) * card_padding
        painter.drawRoundedRect(left_padding, top_padding, frame_width, frame_height, 40, 40)

        # set the font size of the painter
        font = QFont()
        font.setPointSize(self.CARD_FONT_SIZE)
        painter.setFont(font)

        for row in range(n_rows):
            for col in range(n_cols):
                card_left = left_padding + frame_padding + col * width + col * card_padding
                card_top = top_padding + frame_padding + row * height + row * card_padding
                painter.drawRoundedRect(card_left, card_top, width, height, 20, 20)

        text_option = QTextOption()
        text_option.setAlignment(Qt.AlignCenter)
        for letter in self.game.remaining_cards.keys():
            row = (ord(letter) - 65) // n_cols if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) // n_cols
            col = (ord(letter) - 65) % n_rows if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) % n_rows
            card_left = left_padding + frame_padding + col * width + col * card_padding
            card_top = top_padding + frame_padding + row * height + row * card_padding
            card_rect = QRectF(card_left, card_top, width, height)
            painter.drawText(card_rect, letter.upper(), option=text_option)
            # letter_size = font.size(letter)
            # value_size = font.size(str(self.remaining_cards[letter]))
            # text_left = card_left + (width - letter_size[0]) // 2
            # text_top = card_top + (height // 2 - letter_size[1]) // 2
            # value_left = card_left + (width - value_size[0]) // 2
            # value_top = card_top + height // 2 + (height // 2 - value_size[1]) // 2
            # letter_img = font.render(letter, True, GRAY)
            # value_img = font.render(str(self.remaining_cards[letter]), True, WHITE)
            # self.screen.blit(letter_img, (text_left, text_top))
            # self.screen.blit(value_img, (value_left, value_top))

        painter.end()


class KiitosUi(QtWidgets.QMainWindow):

    def __init__(self):
        super(KiitosUi, self).__init__()
        uic.loadUi('game.ui', self)

        self.game = KiitosGame()

        self.counts_widget = CountsWidget(self, self.game)
        self.img_widget = ImageWidget(self)
        self.center_layout = self.findChild(QHBoxLayout, "center_layout")
        self.center_layout.addWidget(self.counts_widget)
        self.center_layout.addWidget(self.img_widget)

        # self.ncd = NewCardDetector()

        # self.camera_thread = QThread()
        # self.camera_timer = QTimer()
        # self.camera_timer.timeout.connect(self.update_camera)
        # self.camera_timer.start(1000)  # this time is kinda arbitrary?
        # TODO: we have two while-loops here which is unncessary
        #  this thread and the camera manager are doing the same thing
        # self.camera_timer.moveToThread(self.camera_thread)
        # self.camera_thread.start()

        # FIXME: if we want to override paintEvent for the img QLabel we need to subclass it

        self.show()

    def update_camera(self):
        # new_card, self.annotated_image = self.ncd.detect(self.bbox_widget.bbox)
        # if new_card is not None and self.game.is_valid_card(new_card):
        # pygame.mixer.Sound.play(notification_sound)
        # pygame.time.set_timer(UNDO_EXPIRED_EVENT, UNDO_EXPIRE_MILLIS, loops=1)
        # self.game.on_new_valid_card(new_card)

        # self.img_widget.update_pixmap()
        pass

    def closeEvent(self, event):
        # self.ncd.cap_manager.stop()
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = KiitosUi()
    app.exec()
