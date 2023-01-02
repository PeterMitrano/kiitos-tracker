import argparse
import random

import numpy as np
import pygame

from tracking_and_detection import NewCardDetector

workspace_bbox_color = (205, 25, 25)

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
instruction_font = pygame.font.SysFont(None, 40)
card_font = pygame.font.SysFont(None, 50)
WHITE = (255, 255, 255)
GRAY = (199, 201, 194)
BLACK = (0, 0, 0)
BLUE = (38, 47, 78)


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


def draw_rectangles(display):
    frame_width = frame_padding + n_cols * width + (n_cols + 1) * card_padding
    frame_height = frame_padding + n_rows * height + (n_rows + 1) * card_padding
    card_frame = pygame.Rect(left_padding, top_padding, frame_width, frame_height)
    pygame.draw.rect(display, BLUE, card_frame, width=20, border_radius=80)
    for row in range(n_rows):
        for col in range(n_cols):
            card_left = left_padding + frame_padding + col * width + col * card_padding
            card_top = top_padding + frame_padding + row * height + row * card_padding
            card_rect = pygame.Rect(card_left, card_top, width, height)
            pygame.draw.rect(display, BLUE, card_rect, border_radius=20)


def draw_cards(display, font, cards):
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
        letter_img = font.render(letter, True, GRAY)
        value_img = font.render(str(cards[letter]), True, WHITE)
        display.blit(letter_img, (text_left, text_top))
        display.blit(value_img, (value_left, value_top))


def draw_board(screen, remaining_cards, round_count, workspace_bbox, annotated_image):
    screen.fill(WHITE)
    screen.blit(background, (0, 0))
    annotated_image = annotated_image.transpose([1, 0, 2])
    annotated_image_surf = pygame.surfarray.make_surface(annotated_image)
    screen.blit(annotated_image_surf, (img_x0, img_y0))
    draw_rectangles(screen)
    headline_size = headline_font.size(f'Kiitos: Round {round_count}')
    headline = headline_font.render(f'Kiitos: Round {round_count}', True, BLUE)
    screen.blit(headline, ((game_width - headline_size[0]) // 2, (top_padding - headline_size[1]) // 2))
    draw_cards(screen, card_font, remaining_cards)
    workspace_w = workspace_bbox[1, 0] - workspace_bbox[0, 0]
    workspace_h = workspace_bbox[1, 1] - workspace_bbox[0, 1]
    workspace_rect = pygame.Rect(img_x0 + workspace_bbox[0, 0], img_y0 + workspace_bbox[0, 1], workspace_w, workspace_h)
    pygame.draw.rect(screen, workspace_bbox_color, workspace_rect, width=2)

    pygame.display.flip()


def draw_round_reset(screen, remaining_cards, workspace_bbox, round_count, annotated_image):
    round_over_size = instruction_font.size(f'Round {round_count} over! Reset the playing area.')
    round_over_text = instruction_font.render(f'Round {round_count} over! Reset the playing area.', True, BLUE)

    for i in range(countdown):
        draw_board(screen, remaining_cards, workspace_bbox, round_count, annotated_image)
        countdown_size = instruction_font.size(f'Starting the next round in...{countdown - i}')
        countdown_text = instruction_font.render(f'Starting the next round in...{countdown - i}', True, BLUE)
        screen.blit(round_over_text, ((game_width - round_over_size[0]) // 2,
                                      game_height - top_padding + (top_padding // 2 - round_over_size[1]) // 2))
        screen.blit(countdown_text, ((game_width - countdown_size[0]) // 2,
                                     game_height - top_padding // 2 + (top_padding // 2 - countdown_size[1]) // 2))
        pygame.display.flip()
        pygame.time.wait(interval)


def draw_game_over(screen):
    screen.fill(GRAY)
    screen.blit(background, (0, 0))
    text_size = headline_font.size('Game Over!')
    text = headline_font.render('Game Over!', True, WHITE)
    text_padding = 40
    text_background = pygame.Rect(game_width // 2 - (text_size[0] + text_padding) // 2, game_height // 2 - (text_size[1] + text_padding) // 2, text_size[0] + text_padding, text_size[1] + text_padding)
    pygame.draw.rect(screen, BLUE, text_background, border_radius=40)
    screen.blit(text, (game_width // 2 - (text_size[0]) // 2, game_height // 2 - (text_size[1]) // 2))
    pygame.display.flip()


def on_new_valid_card(new_card, remaining_cards, print_card=False):
    remaining_cards[new_card] -= 1
    if print_card:
        print(f'new card! {new_card}')


def run_kiitos():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-vision', action='store_true')

    args = parser.parse_args()

    if args.debug_vision:
        ncd = NewCardDetector()
    remaining_cards = reset_card_dict()

    screen = pygame.display.set_mode([game_width, game_height])

    workspace_bbox = np.array([
        [50, 90],
        [600, 410],
    ])

    round_count = 1
    annotated_image = np.zeros([img_h, img_w, 3])
    game_over = False
    running = True
    while running and not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print("Thanks for playing Kiitos with Peter and Andrea's card counter!")

        if args.debug_vision:
            new_card, annotated_image = ncd.detect(workspace_bbox)
            if new_card is not None:
                if new_card in remaining_cards:
                    on_new_valid_card(new_card, remaining_cards, print_card=True)
                else:
                    print(f"bad detection! {new_card}")
        else:
            choice_var = random.randint(0, 99)
            if choice_var > 90:
                random_letter = chr(random.randint(65, 90))
                while random_letter == 'Q' or remaining_cards[random_letter] == 0:
                    random_letter = chr(random.randint(65, 90))
                on_new_valid_card(random_letter, remaining_cards)

        round_over = np.sum(np.asarray(list(remaining_cards.values()))) <= 0

        draw_board(screen, remaining_cards, round_count, workspace_bbox, annotated_image)

        if round_over:
            if round_count < 3:
                draw_round_reset(screen, remaining_cards, round_count, workspace_bbox, annotated_image)
                round_count += 1
                remaining_cards = reset_card_dict()
            else:
                game_over = True

        # cv2.waitKey(10)

    game_over_splash = True
    while game_over_splash:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over_splash = False
                print("Thanks for playing Kiitos with Peter and Andrea's card counter!")
        draw_game_over(screen)


if __name__ == '__main__':
    run_kiitos()
