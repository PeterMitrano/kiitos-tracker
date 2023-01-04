import enum

import pygame

instructions_font_size = 40
instruction_font = pygame.font.SysFont(None, instructions_font_size)
card_font = pygame.font.SysFont(None, 50)
WHITE = (255, 255, 255)
GRAY = (199, 201, 194)
BLACK = (0, 0, 0)
BLUE = (38, 47, 78)


class Align(enum.Enum):
    BEGIN = enum.auto()
    MIDDLE = enum.auto()
    END = enum.auto()


class TextBox:

    def __init__(self, text, x, y):
        self.text_color = WHITE
        self.text = text
        self.x = x
        self.y = y
        self.padding = 4
        self.x_align = Align.BEGIN
        self.y_align = Align.BEGIN
        self.text_surf = instruction_font.render(self.text, True, self.text_color)
        self.h = self.text_surf.get_height() + 2 * self.padding
        self.text_w = self.text_surf.get_width()
        self.rect = None
        self.update_align()
        self.update_rect()

    def draw(self, screen, color):
        self.update_align()
        self.update_rect()
        screen.fill(color, self.rect)
        screen.blit(self.text_surf, (self.x + self.align_x, self.y + self.align_y))

    def update_align(self):
        self.align_x = 0
        if self.x_align == Align.MIDDLE:
            self.align_x = -self.text_surf.get_width() / 2
        elif self.x_align == Align.END:
            self.align_x = -self.text_surf.get_width()
        self.align_y = 0
        if self.y_align == Align.MIDDLE:
            self.align_y = -self.text_surf.get_width() / 2
        elif self.y_align == Align.END:
            self.align_y = -self.text_surf.get_width()

    def update_rect(self):
        self.rect = pygame.Rect(self.x - self.padding + self.align_x, self.y - self.padding + self.align_y, self.text_w + 2 * self.padding, self.h + 2 * self.padding)

    def pressed(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(*mouse_pos):
            return True
        return False


class ButtonState(enum.Enum):
    HIDING = enum.auto()
    WAITING = enum.auto()
    SHOWING = enum.auto()


class Button(TextBox):

    def __init__(self, text, x, y):
        super().__init__(text, x, y)
        self.state = ButtonState.SHOWING

    def draw(self, screen):
        button_color = BLACK
        if self.state == ButtonState.WAITING:
            button_color = GRAY

        if self.state != ButtonState.HIDING:
            super().draw(screen, button_color)

    def set_state(self, state):
        self.state = state

    def pressed(self):
        pressed = super().pressed()
        if pressed:
            self.state = ButtonState.WAITING
            return True
        return False


class PopUpState(enum.Enum):
    HIDING = enum.auto()
    SHOWING = enum.auto()


class PopUp(TextBox):

    def __init__(self, text, x, y):
        super().__init__(text, x, y)
        self.state = PopUpState.HIDING
        self.background_color = BLACK

    def draw(self, screen):
        if self.state == PopUpState.SHOWING:
            super().draw(screen, self.background_color)

    def set_state(self, state):
        self.state = state
