import argparse
import enum
import random
import time

import numpy as np
import pygame

from tracking_and_detection import NewCardDetector


class GameState(enum.Enum):
    PLAYING = enum.auto()
    MANUAL_LETTER = enum.auto()
    CORRECTING = enum.auto()
    OVER = enum.auto()


class UndoState(enum.Enum):
    WAITING = enum.auto()
    SHOWING = enum.auto()


workspace_bbox_color = (205, 25, 25)
workspace_bbox = np.array([
    [50, 140],
    [600, 310],
])

pygame.init()
headline_font_size = 80
img_w = 640
img_h = 480
img_padding = 10
game_width = 625 + img_w + img_padding * 2
img_x0 = game_width - 2 * img_padding - img_w
img_y0 = img_padding + headline_font_size
game_height = 900
top_padding = 100
bottom_padding = 10
left_padding = 25
n_rows = 5
n_cols = 5
frame_padding = 50
card_padding = 25
width = 75
height = 100
countdown = 3
interval = 500
background = pygame.image.load('img/background.jpg')
background = pygame.transform.scale(background, (game_width + 500, game_height + 500))
headline_font = pygame.font.SysFont(None, headline_font_size)
instructions_font_size = 40
instruction_font = pygame.font.SysFont(None, instructions_font_size)
card_font = pygame.font.SysFont(None, 50)
WHITE = (255, 255, 255)
GRAY = (199, 201, 194)
BLACK = (0, 0, 0)
BLUE = (38, 47, 78)

notification_sound = pygame.mixer.Sound("notification.wav")
notification_sound.set_volume(0.25)

UNDO_EVENT = pygame.USEREVENT + 1


def reset_card_dict():
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


class Kiitos:

    def __init__(self, debug_vision):
        self.debug_vision = debug_vision
        self.screen = pygame.display.set_mode([game_width, game_height])
        self.remaining_cards = reset_card_dict()
        self.state = GameState.PLAYING
        self.undo_state = UndoState.WAITING
        self.round_count = 1
        self.annotated_image = np.zeros([img_h, img_w, 3])
        self.ncd = None
        self.latest_letter = None
        if self.debug_vision:
            self.ncd = NewCardDetector()

        self.undo_rect = pygame.Rect(2, 2, 50, 10)

    def run(self):
        while self.state != GameState.OVER:
            manual_letter = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = GameState.OVER
                    print("Thanks for playing Kiitos with Peter and Andrea's card counter!")
                elif event.type == pygame.KEYDOWN:
                    self.state = GameState.MANUAL_LETTER
                    manual_letter = event.unicode.upper()
                elif event.type == UNDO_EVENT:
                    print("undo timer is done!")
                    self.undo_state = UndoState.WAITING
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.undo_rect.collidepoint(*mouse_pos):
                        self.state = GameState.CORRECTING

            self.handle_state(manual_letter)

            self.draw_board()

            time.sleep(0.1)

        self.end_game()

    def handle_state(self, manual_letter):
        if self.state == GameState.PLAYING:
            if self.debug_vision:
                new_card, annotated_image = self.ncd.detect(workspace_bbox)
                if new_card is not None:
                    if new_card in self.remaining_cards:
                        self.on_new_valid_card(new_card, print_card=True)
                else:
                    print(f"bad detection! {new_card}")
            else:
                choice_var = random.randint(0, 99)
                if choice_var > 90:
                    random_letter = chr(random.randint(65, 90))
                    while random_letter == 'Q' or self.remaining_cards[random_letter] == 0:
                        random_letter = chr(random.randint(65, 90))
                    self.on_new_valid_card(random_letter)
            round_over = np.sum(np.asarray(list(self.remaining_cards.values()))) <= 0
            if round_over:
                self.state = GameState.OVER
        elif self.state == GameState.MANUAL_LETTER:
            if manual_letter in self.remaining_cards:
                self.on_new_valid_card(manual_letter)
            self.state = GameState.PLAYING
        elif self.state == GameState.CORRECTING:
            print("correcting!")
            pass
        elif self.state == GameState.OVER:
            if self.round_count < 3:
                self.draw_round_reset()
                self.round_count += 1
                self.remaining_cards = reset_card_dict()
            else:
                game_over = True

    def end_game(self):
        game_over_splash = True
        while game_over_splash:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over_splash = False
                    print("Thanks for playing Kiitos with Peter and Andrea's card counter!")
            self.draw_game_over()

    def draw_rectangles(self):
        frame_width = frame_padding + n_cols * width + (n_cols + 1) * card_padding
        frame_height = frame_padding + n_rows * height + (n_rows + 1) * card_padding
        card_frame = pygame.Rect(left_padding, top_padding, frame_width, frame_height)
        pygame.draw.rect(self.screen, BLUE, card_frame, width=20, border_radius=80)
        for row in range(n_rows):
            for col in range(n_cols):
                card_left = left_padding + frame_padding + col * width + col * card_padding
                card_top = top_padding + frame_padding + row * height + row * card_padding
                card_rect = pygame.Rect(card_left, card_top, width, height)
                pygame.draw.rect(self.screen, BLUE, card_rect, border_radius=20)

    def draw_cards(self, font):
        for letter in list(self.remaining_cards.keys()):
            letter_size = font.size(letter)
            value_size = font.size(str(self.remaining_cards[letter]))
            row = (ord(letter) - 65) // n_cols if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) // n_cols
            col = (ord(letter) - 65) % n_rows if ord(letter) < ord('Q') else (ord(letter) - 65 - 1) % n_rows
            card_left = left_padding + frame_padding + col * width + col * card_padding
            card_top = top_padding + frame_padding + row * height + row * card_padding
            text_left = card_left + (width - letter_size[0]) // 2
            text_top = card_top + (height // 2 - letter_size[1]) // 2
            value_left = card_left + (width - value_size[0]) // 2
            value_top = card_top + height // 2 + (height // 2 - value_size[1]) // 2
            letter_img = font.render(letter, True, GRAY)
            value_img = font.render(str(self.remaining_cards[letter]), True, WHITE)
            self.screen.blit(letter_img, (text_left, text_top))
            self.screen.blit(value_img, (value_left, value_top))

    def draw_board(self):
        self.screen.fill(WHITE)
        self.screen.blit(background, (0, 0))
        annotated_image = self.annotated_image.transpose([1, 0, 2])
        annotated_image_surf = pygame.surfarray.make_surface(annotated_image)
        self.screen.blit(annotated_image_surf, (img_x0, img_y0))
        self.draw_rectangles()
        headline_size = headline_font.size(f'Kiitos: Round {self.round_count}')
        headline = headline_font.render(f'Kiitos: Round {self.round_count}', True, BLUE)
        self.screen.blit(headline, ((game_width - headline_size[0]) // 2, (top_padding - headline_size[1]) // 2))
        self.draw_cards(card_font)
        workspace_w = workspace_bbox[1, 0] - workspace_bbox[0, 0]
        workspace_h = workspace_bbox[1, 1] - workspace_bbox[0, 1]
        workspace_rect = pygame.Rect(img_x0 + workspace_bbox[0, 0], img_y0 + workspace_bbox[0, 1], workspace_w,
                                     workspace_h)
        pygame.draw.rect(self.screen, workspace_bbox_color, workspace_rect, width=2)

        manual_instructions = instruction_font.render('Press any key to manually decrement the count', True, BLACK)
        self.screen.blit(manual_instructions, (left_padding, game_height - bottom_padding - 2 * instructions_font_size))

        if self.undo_state == UndoState.SHOWING:
            undo_instructions = instruction_font.render(f'You played {self.latest_letter}', True, BLACK)
            self.screen.blit(undo_instructions, (left_padding, game_height - bottom_padding - instructions_font_size))
            pygame.draw.rect(self.screen, BLACK, self.undo_rect, width=1, border_radius=1)

        pygame.display.flip()

    def draw_round_reset(self):
        round_over_size = instruction_font.size(f'Round {self.round_count} over! Reset the playing area.')
        round_over_text = instruction_font.render(f'Round {self.round_count} over! Reset the playing area.', True, BLUE)

        for i in range(countdown):
            self.draw_board()
            countdown_size = instruction_font.size(f'Starting the next round in...{countdown - i}')
            countdown_text = instruction_font.render(f'Starting the next round in...{countdown - i}', True, BLUE)
            self.screen.blit(round_over_text, ((game_width - round_over_size[0]) // 2,
                                               game_height - top_padding + (
                                                       top_padding // 2 - round_over_size[1]) // 2))
            self.screen.blit(countdown_text, ((game_width - countdown_size[0]) // 2,
                                              game_height - top_padding // 2 + (
                                                      top_padding // 2 - countdown_size[1]) // 2))
            pygame.self.screen.flip()
            pygame.time.wait(interval)

    def draw_game_over(self):
        self.screen.fill(GRAY)
        self.screen.blit(background, (0, 0))
        text_size = headline_font.size('Game Over!')
        text = headline_font.render('Game Over!', True, WHITE)
        text_padding = 40
        text_background = pygame.Rect(game_width // 2 - (text_size[0] + text_padding) // 2,
                                      game_height // 2 - (text_size[1] + text_padding) // 2,
                                      text_size[0] + text_padding,
                                      text_size[1] + text_padding)
        pygame.draw.rect(self.screen, BLUE, text_background, border_radius=40)
        self.screen.blit(text, (game_width // 2 - (text_size[0]) // 2, game_height // 2 - (text_size[1]) // 2))
        pygame.display.flip()

    def on_new_valid_card(self, new_card, print_card=False):
        self.latest_letter = new_card
        pygame.mixer.Sound.play(notification_sound)

        self.undo_state = UndoState.SHOWING
        pygame.time.set_timer(UNDO_EVENT, 3000, loops=1)

        if print_card:
            print(f'new card! {new_card}')
        if self.remaining_cards[new_card] > 0:
            self.remaining_cards[new_card] -= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-vision', action='store_true')

    args = parser.parse_args()

    k = Kiitos(args.debug_vision)
    k.run()


if __name__ == '__main__':
    main()
