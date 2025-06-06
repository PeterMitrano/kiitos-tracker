import os
import random
import time

import requests
from tqdm import tqdm

API_KEY = os.getenv("MW_API_KEY")
BASE_URL = f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/"


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
    with open("data/nltk_words.txt") as f:
        words = [l.strip("\n") for l in f.readlines()][5015:]

        valid_words = []
        for word in tqdm(words):
            if is_valid_word(word):
                valid_words.append(word)
            time.sleep(15 + random.randint(1, 15))

    with open("validated_nltk_words.txt") as f:
        for word in valid_words:
            f.write(word + "\n")


if __name__ == "__main__":
    main()
