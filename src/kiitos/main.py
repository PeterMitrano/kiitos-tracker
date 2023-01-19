import os
import sys
from datetime import datetime

from PIL import Image
# noinspection PyUnresolvedReferences
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QLibraryInfo, QThread
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

from counts_widget import CountsWidget
from game_logic import KiitosGame
from kiitos.card_detector_widget import CardDetectorWidget
from kiitos.image_widget import ImageWidget
from kiitos.next_round_dialog import NextRoundDialog

# This is a problem caused by OpenCV
# https://stackoverflow.com/questions/68417682/
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

class KiitosUi(QtWidgets.QMainWindow):

    def __init__(self):
        super(KiitosUi, self).__init__()
        uic.loadUi('src/kiitos/game.ui', self)

        self.game = KiitosGame()

        margin = 10
        self.setContentsMargins(margin, margin, margin, margin)

        self.capture_action.triggered.connect(self.save_last_frame)
        self.capture_action.setShortcut("Ctrl+S")
        self.reset_action.triggered.connect(self.reset)
        self.reset_action.setShortcut("Ctrl+R")
        self.counts_widget = CountsWidget(self, self.game)
        # FIXME: should the detector widget own the image widget? or vise-versa?
        self.img_widget = ImageWidget(self)
        self.detector_widget = CardDetectorWidget(self.img_widget, self.game)
        self.detector_widget.new_annotated_image.connect(self.img_widget.on_new_annotated_image)
        self.detector_widget.new_detection_str.connect(self.game.on_new_valid_card)
        self.detector_widget.new_detection_str.connect(self.new_detection)
        self.game.next_round.connect(self.on_next_round)
        self.center_layout.setAlignment(Qt.AlignTop)
        self.center_layout.addWidget(self.counts_widget)
        self.center_layout.addWidget(self.img_widget)

        self.camera_thread = QThread()
        self.detector_widget.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.detector_widget.run)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.start()

        self.show()

    def on_next_round(self):
        dialog = NextRoundDialog()
        if dialog.exec():
            self.reset()

    def reset(self):
        self.game.reset()
        self.counts_widget.update()

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
