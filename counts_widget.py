from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QTextOption
from PyQt5.QtWidgets import QLabel


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
