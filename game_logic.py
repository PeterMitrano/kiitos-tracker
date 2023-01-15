def reset_card_dict():
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
        'Z': 1
    }


class KiitosGame:

    def __init__(self):
        self.latest_letter = None
        self.remaining_cards = reset_card_dict()

    def is_valid_card(self, card):
        return card in self.remaining_cards

    def on_new_valid_card(self, new_card, print_card=False):
        self.latest_letter = new_card
        if print_card:
            print(f'new card! {new_card}')

        if self.remaining_cards[new_card] > 0:
            self.remaining_cards[new_card] -= 1
