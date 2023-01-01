import random

import cv2
import numpy as np
import pygame

from ocr import GoogleOCR
from video_capture import CaptureManager


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
        cap = cv2.VideoCapture(2)
        self.cap_manager = CaptureManager(cap)

        self.ocr = GoogleOCR()

    def detect(self):
        ret, frame = self.cap_manager.last_frame
        if not ret:
            print("Can't receive frame!")
            return None

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        cv2.imshow('input', frame)

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
                    card_tracker.position = 0.95 * card_tracker.position + 0.05 * position
                    card_tracker.confidence = min(1, card_tracker.confidence + 0.25)

            to_remove = []
            for card_tracker in self.card_trackers:
                card_tracker.confidence -= 0.02
                if card_tracker.confidence <= 0:
                    # print('removing', card_tracker)
                    to_remove.append(card_tracker)

            for to_remove_i in to_remove:
                self.card_trackers.remove(to_remove_i)

            # print(' '.join([t.letter for t in self.card_trackers]))
            for card_tracker in self.card_trackers:
                # print(card_tracker)
                if card_tracker.confidence > 0.9 and not card_tracker.reported:
                    card_tracker.reported = True
                    return card_tracker.letter

        cv2.waitKey(1)

        return None

    # cap = cv2.VideoCapture(2)
    # ocr = GoogleOCR()
    # all_positions = []
    # for _ in range(60):
    #     ret, frame = cap.read()
    #     positions = ocr.detect(frame)
    #     all_positions.extend(positions)
    #
    # all_positions = np.array(all_positions)
    # plt.scatter(all_positions[:, 0], all_positions[:, 1], s=1)
    # plt.xlim(0, 640)
    # plt.ylim(0, 480)
    # plt.show()


# if __name__ == '__main__':
#     main()


def get_card_dict():
    card_dict = dict()
    card_dict['A'] = 5
    card_dict['B'] = 2
    card_dict['C'] = 3
    card_dict['D'] = 3
    card_dict['E'] = 8
    card_dict['F'] = 2
    card_dict['G'] = 2
    card_dict['H'] = 2
    card_dict['I'] = 6
    card_dict['J'] = 1
    card_dict['K'] = 1
    card_dict['L'] = 3
    card_dict['M'] = 2
    card_dict['N'] = 4
    card_dict['O'] = 5
    card_dict['P'] = 2
    card_dict['R'] = 5
    card_dict['S'] = 5
    card_dict['T'] = 4
    card_dict['U'] = 2
    card_dict['V'] = 1
    card_dict['W'] = 1
    card_dict['X'] = 1
    card_dict['Y'] = 1
    card_dict['Z'] = 1
    return card_dict


def update_cards(cards, random_update=False):
    if random_update:
        random_letter = chr(random.randint(65, 90))
        while random_letter == 'Q' or cards[random_letter] == 0:
            random_letter = chr(random.randint(65, 90))
    else:
        pass


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
    for letter in list(cards.keys()):
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
            screen.fill((255, 255, 255))
            screen.blit(background, (0, 0))
            screen.blit(headline, ((game_width - headline_size[0]) // 2, (top_padding - headline_size[1]) // 2))

            new_card = ncd.detect()
            if new_card is not None:
                remaining_cards[new_card] -= 1
                print(f'new card! {new_card}')

            draw_rectangles(screen, top_padding)
            draw_cards(screen, card_font, remaining_cards, top_padding)
            pygame.display.flip()
            # pygame.time.wait(sleep_time)
            # time_count += sleep_time


if __name__ == '__main__':
    run_kiitos()
