import os
import pickle

import numpy as np
import requests

# noinspection PyUnresolvedReferences
from kiitos.generate_trie import Node, get_possible_words, Index

API_KEY = os.getenv("MW_API_KEY")

BASE_URL = f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/"


class KiitosAgent:

    def __init__(self):
        self.current_substr = ""
        self.cards = []

        with open("data/trie.pkl", "rb") as trie_f:
            self.trie = pickle.load(trie_f)

    def reset(self):
        self.current_substr = ""
        self.cards = []

    def print_all_moves(self, current_substr):
        for candidate_substr in self.get_candidate_substrs(current_substr):
            is_valid, possible_words = self.is_valid(candidate_substr)
            if is_valid:
                print(candidate_substr, "is valid:")
                print(possible_words)

    def parse_cards(self, cards_str):
        self.cards = [c.lower() for c in cards_str]

    def is_valid(self, candidate_substr):
        node = self.trie
        for c in candidate_substr:
            if Node(c) in node.children:
                node = node.find_child_by_name(c)
            else:
                return False, None

        possible_words = np.unique(list(get_possible_words(node))).tolist()
        sorted_possible_words = sorted(possible_words, key=lambda w: len(w))
        return True, sorted_possible_words

    def get_candidate_substrs(self, current_substr):
        """assumes super-kiitos and sqeeze-in"""
        for c in self.cards:
            for insert_idx in range(len(current_substr) + 1):
                yield current_substr[:insert_idx] + c + current_substr[insert_idx:]


def is_valid_word(word):
    """Check if a word is valid using the Merriam-Webster API."""
    url = f"{BASE_URL}{word}?key={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching data from API")
        return False

    data = response.json()

    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], dict)
        and "meta" in data[0]
    ):
        return True

    return False


def main():
    agent = KiitosAgent()

    word = "aalii"

    while True:
        try:
            cards_str = input("Please enter your cards, no seperators:\n")
            # cards_str = "esnsetsi"

            agent.parse_cards(cards_str)

            current_substr = input("Please enter the current game state:\n")
            # current_substr = "lett"

            agent.print_all_moves(current_substr)
        except KeyboardInterrupt:
            pass
        print("Starting a new round")


if __name__ == "__main__":
    main()
