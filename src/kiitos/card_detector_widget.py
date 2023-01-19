import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QSoundEffect

from kiitos.game_logic import KiitosGame
from kiitos.image_widget import ImageWidget
from kiitos.tracking import NewCardDetector


class CardDetectorWidget(QObject):
    new_detection_str = pyqtSignal(str)
    new_annotated_image = pyqtSignal(np.ndarray)

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
            self.new_annotated_image.emit(annotated_image)
            if new_card is not None and self.game.is_valid_card(new_card):
                self.effect.play()
                self.new_detection_str.emit(new_card)
        self.ncd.cap_manager.stop()
