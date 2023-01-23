import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtWidgets import QLabel, QSizePolicy

from kiitos.draggable_bbox import DraggableBbox, BBoxHandle

IMG_W = 640
IMG_H = 480


def array_to_qimg(annotated_image):
    return QImage(annotated_image.data, IMG_W, IMG_H, 3 * IMG_W, QImage.Format.Format_RGB888)


class ImageWidget(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.settings = parent.settings
        init_pixmap = QPixmap(IMG_W, IMG_H)
        init_pixmap.fill(Qt.gray)
        self.setPixmap(init_pixmap)
        self.draggable_bbox = DraggableBbox(self)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.resize(IMG_H, IMG_W)
        self.setStyleSheet("border: 1px solid black;")

    def on_new_annotated_image(self, annotated_image):
        # grey-out the areas outside the rect
        h, w, x0, x1, y0, y1 = self.get_bounded_rect_params()

        self.grey_outside_bbox(annotated_image, x0, x1, y0, y1)

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        q_img = array_to_qimg(annotated_image)
        pixmap = QPixmap(q_img)
        # Calling setPixmap will trigger a paintEvent which ensures the box is drawn on top of the image
        self.setPixmap(pixmap)

    def grey_outside_bbox(self, annotated_image, x0, x1, y0, y1):
        greyed = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY)
        greyed = np.expand_dims(greyed, axis=2)
        k = -120
        annotated_image[:y0, :] = (greyed[:y0, :] + k).clip(0, 255)
        annotated_image[y1:, :] = (greyed[y1:, :] + k).clip(0, 255)
        annotated_image[:, :x0] = (greyed[:, :x0] + k).clip(0, 255)
        annotated_image[:, x1:] = (greyed[:, x1:] + k).clip(0, 255)
        return annotated_image

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(3)
        pen.setColor(Qt.red)
        painter.setPen(pen)

        h, w, x0, x1, y0, y1 = self.get_bounded_rect_params()

        center_x = int(x0 + w / 2)
        center_y = int(y0 + h / 2)
        self.draggable_bbox.left.move(x0 - BBoxHandle.HALF_SIZE, center_y - BBoxHandle.HALF_SIZE)
        self.draggable_bbox.top.move(center_x - BBoxHandle.HALF_SIZE, y0 - BBoxHandle.HALF_SIZE)
        self.draggable_bbox.right.move(x1 - BBoxHandle.HALF_SIZE, center_y - BBoxHandle.HALF_SIZE)
        self.draggable_bbox.bottom.move(center_x - BBoxHandle.HALF_SIZE, y1 - BBoxHandle.HALF_SIZE)

        painter.drawRect(x0, y0, w, h)
        painter.end()

    def get_bounded_rect_params(self):
        x0, y0, w, h = self.draggable_bbox.get_rect_params()
        # clip based on the size of the image
        # the order of these updates is important because we're overwriting values
        x1 = min(max(x0 + w, 0), self.width())
        y1 = min(max(y0 + h, 0), self.height())
        x0 = min(max(x0, 0), self.width())
        y0 = min(max(y0, 0), self.height())
        h = y1 - y0
        w = x1 - x0
        return h, w, x0, x1, y0, y1

    def save_settings(self):
        self.draggable_bbox.save_settings()
