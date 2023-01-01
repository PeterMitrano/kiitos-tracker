import random
import time

import cv2
import numpy as np
import pygame

from ocr import GoogleOCR
from video_capture import CaptureManager

confidence_inc = 0.25
confidence_dec = confidence_inc / 5
motion_alpha = 0.5


class CardTracker:

    def __init__(self, letter, position):
        self.letter = letter
        self.position = position
        self.confidence = 0.1
        self.reported = False

    def __repr__(self):
        return f'{self.letter} {self.position} {self.confidence} {self.reported}'


class NewCardDetector:

    def __init__(self):
        self.card_trackers = []
        self.cap = cv2.VideoCapture(2)
        self.cap_manager = CaptureManager(self.cap)
        while self.cap_manager.last_frame is None:
            time.sleep(1)

        self.ocr = GoogleOCR()
        # self.ocr = None

    def detect(self):
        frame = self.cap_manager.last_frame

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        if self.ocr is not None:
            annotated_frame, detected_letters, detected_positions = self.ocr.detect(frame)
            cv2.imshow('OCR', annotated_frame)

            for letter, position in zip(detected_letters, detected_positions):
                tracker_found = False
                for card_tracker in self.card_trackers:
                    if card_tracker.letter == letter and np.linalg.norm(card_tracker.position - position) < 50:
                        tracker_found = True
                        break
                if not tracker_found:
                    self.card_trackers.append(CardTracker(letter, position))
                else:
                    card_tracker.position = motion_alpha * card_tracker.position + (1 - motion_alpha) * position
                    card_tracker.confidence = min(1, card_tracker.confidence + confidence_inc)

            to_remove = []
            for card_tracker in self.card_trackers:
                card_tracker.confidence -= confidence_dec
                if card_tracker.confidence <= 0:
                    to_remove.append(card_tracker)

            for to_remove_i in to_remove:
                self.card_trackers.remove(to_remove_i)

            for card_tracker in self.card_trackers:
                if card_tracker.confidence > 0.9 and not card_tracker.reported:
                    card_tracker.reported = True
                    return card_tracker.letter

        return None


def get_card_dict():
    return {
        'A': 5,
        'B': 2,
        'C': 3,
        'D': 3,
        'E': 8,
        'F': 2,
        'G': 2,
        'H': 2,
        'I': 6,
        'J': 1,
        'K': 1,
        'L': 3,
        'M': 2,
        'N': 4,
        'O': 5,
        'P': 2,
        'R': 5,
        'S': 5,
        'T': 4,
        'U': 2,
        'V': 1,
        'W': 1,
        'X': 1,
        'Y': 1,
        'Z': 1,
    }


def draw_rectangles(display, top_padding, left_padding=50, n_rows=5, n_cols=5, frame_padding=100, card_padding=50,
                    width=150, height=200):
    frame_width = frame_padding + n_cols * width + (n_cols + 1) * card_padding
    frame_height = frame_padding + n_rows * height + (n_rows + 1) * card_padding
    card_frame = pygame.Rect(left_padding, top_padding, frame_width, frame_height)
    pygame.draw.rect(display, (0, 0, 255), card_frame, width=20, border_radius=80)
    for row in range(n_rows):
        for col in range(n_cols):
            card_left = left_padding + frame_padding + col * width + col * card_padding
            card_top = top_padding + frame_padding + row * height + row * card_padding
            card_rect = pygame.Rect(card_left, card_top, width, height)
            pygame.draw.rect(display, (0, 0, 255), card_rect)


def draw_cards(display, font, cards, top_padding, left_padding=50, n_rows=5, n_cols=5, frame_padding=100,
               card_padding=50, width=150, height=200):
    for letter in cards.keys():
        letter_size = font.size(letter)
        value_size = font.size(str(cards[letter]))
        row = (ord(letter) - 65) // n_cols if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) // n_cols
        col = (ord(letter) - 65) % n_rows if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) % n_rows
        card_left = left_padding + frame_padding + col * width + col * card_padding
        card_top = top_padding + frame_padding + row * height + row * card_padding
        text_left = card_left + (width - letter_size[0]) // 2
        text_top = card_top + (height // 2 - letter_size[1]) // 2
        value_left = card_left + (width - value_size[0]) // 2
        value_top = card_top + height // 2 + (height // 2 - value_size[1]) // 2
        letter_img = font.render(letter, True, (255, 255, 255))
        value_img = font.render(str(cards[letter]), True, (255, 255, 255))
        display.blit(letter_img, (text_left, text_top))
        display.blit(value_img, (value_left, value_top))


def run_kiitos():
    ncd = NewCardDetector()
    random.seed(0)
    remaining_cards = get_card_dict()
    pygame.init()
    game_width = 1250
    game_height = 1800
    top_padding = 200
    screen = pygame.display.set_mode([game_width, game_height])
    background = pygame.image.load('background.jpg')
    background = pygame.transform.scale(background, (game_width + 500, game_height + 500))
    headline_font = pygame.font.SysFont(None, 128)
    headline_size = headline_font.size('Kiitos: Remaining letters!')
    card_font = pygame.font.SysFont(None, 100)
    headline = headline_font.render('Kiitos: Remaining letters!', True, (0, 0, 255))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print("Thanks for playing Kiitos with Peter and Andrea's card counter!")

        new_card = ncd.detect()
        if new_card is not None:
            if new_card in remaining_cards:
                on_new_valid_card(new_card, remaining_cards)
            else:
                print(f"bad detection! {new_card}")

        screen.fill((255, 255, 255))
        screen.blit(background, (0, 0))
        screen.blit(headline, ((game_width - headline_size[0]) // 2, (top_padding - headline_size[1]) // 2))
        draw_rectangles(screen, top_padding)
        draw_cards(screen, card_font, remaining_cards, top_padding)
        pygame.display.flip()

        cv2.waitKey(10)


def on_new_valid_card(new_card, remaining_cards):
    remaining_cards[new_card] -= 1
    print(f'new card! {new_card}')


if __name__ == '__main__':
    run_kiitos()
