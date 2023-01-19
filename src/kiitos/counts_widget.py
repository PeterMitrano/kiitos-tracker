from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QTextOption
from PyQt5.QtWidgets import QLabel


class CountsWidget(QLabel):
    CARD_FONT_SIZE = 28

    def __init__(self, parent, game):
        super().__init__(parent)
        self.game = game
        self.setFixedSize(550, 650)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        blue = QColor("#262f4e")
        grayblue = QColor("#6d84a1")

        card_pen = QPen()
        card_pen.setWidth(4)
        card_pen.setColor(grayblue)

        letter_pen = QPen()
        letter_pen.setWidth(3)
        letter_pen.setColor(grayblue)

        count_pen = QPen()
        count_pen.setWidth(3)
        count_pen.setColor(blue)

        frame_padding = 25
        card_padding = 25
        width = 75
        height = 100
        n_rows = 5
        n_cols = 5

        font = QFont()
        font.setPointSize(self.CARD_FONT_SIZE)
        painter.setFont(font)

        for row in range(n_rows):
            for col in range(n_cols):
                card_left = frame_padding + col * width + col * card_padding
                card_top = frame_padding + row * height + row * card_padding
                painter.setPen(card_pen)
                painter.drawRoundedRect(card_left, card_top, width, height, 20, 20)

        text_option = QTextOption()
        text_option.setAlignment(Qt.AlignCenter)
        for letter, count in self.game.remaining_cards.items():
            row = (ord(letter) - 65) // n_cols if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) // n_cols
            col = (ord(letter) - 65) % n_rows if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) % n_rows
            card_left = frame_padding + col * width + col * card_padding
            card_top = frame_padding + row * height + row * card_padding
            letter_rect = QRectF(card_left, card_top, width, height / 2)
            count_rect = QRectF(card_left, card_top + height / 2, width, height / 2)

            painter.setPen(letter_pen)
            painter.drawText(letter_rect, letter.upper(), option=text_option)

            painter.setPen(count_pen)
            painter.drawText(count_rect, str(count), option=text_option)

        painter.end()
