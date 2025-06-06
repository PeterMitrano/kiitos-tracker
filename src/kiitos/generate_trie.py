import json
import math
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Sequence, Union, Tuple, Dict, List

import rerun as rr
from tqdm import tqdm


def binomial_coefficient(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def staircase_number(n):
    return binomial_coefficient(math.floor((n + 1) / 2) + (n + 1), n)


@lru_cache
def triangle_number(n):
    return sum([x for x in range(n)])


def load_words(words_path=Path("data/1000-most-common-words.txt")):
    """reads a txt file of newline-seperated words"""
    with words_path.open() as f:
        return [w.strip() for w in f.readlines()]

@dataclass
class Node:
    children: Dict[str, "Node"] = None
    words: List[str] = None
    word_indices: List[Tuple[int]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.words is None:
            self.words = []
        if self.word_indices is None:
            self.word_indices = []

    def to_json(self):
        return {
            "children": {
                str(name): node.to_json()
                for name, node in self.children.items()
            },
            "words": self.words,
            "word_indices": self.word_indices,
        }


def gen_substrings(word: Sequence, word_idx: int):
    """
    yields all substrings of w, as well as its unique ID tuple (word_idx, suffix_idx)
    """
    word_len = len(word)
    suffix_idx = 0
    for suffix_len in range(1, word_len + 1):
        for start in range(word_len - suffix_len + 1):
            yield word[start : start + suffix_len], (word_idx, suffix_idx)
            suffix_idx += 1


def trie_to_graph(node: Node, vertices=(), labels=(), edges=(), vertex_idx=0):
    child_vertex_idx = vertex_idx + 1

    vertices += (vertex_idx,)
    labels += (node.get_name(),)
    for child in node.children.values():
        edges += ((vertex_idx, child_vertex_idx),)
        vertices, edges, labels, new_idx = trie_to_graph(
            child,
            vertices=vertices,
            labels=labels,
            edges=edges,
            vertex_idx=child_vertex_idx,
        )
        child_vertex_idx = new_idx
    return vertices, edges, labels, child_vertex_idx


def viz_trie(trie):
    vertices, edges, labels, _ = trie_to_graph(trie)

    rr.log(
        "trie",
        rr.GraphNodes(vertices, labels=labels),
        rr.GraphEdges(edges, graph_type="directed"),
    )


def count_substrings(words):
    # figure out how many suffixes exist in the dictionary
    n_suffixes = sum([triangle_number(len(w)) for w in words])
    print(f"Expecting {n_suffixes=:,d}")
    n_chars = sum([staircase_number(len(w)) for w in words])
    print(f"Which would have a total of {n_chars=:,d}")


def add_to_trie(trie: Node, substr, idx, original_word=None):
    node = trie
    for c in substr:
        if c not in node.children:
            node.children[c] = Node()
        node = node.children[c]  # Move deeper in the trie

    node.words.append(original_word)
    node.word_indices.append(idx)


def get_possible_words(node: Node):
    if node.original_word is not None:
        yield node.original_word
    for c_name, c_node in node.children.items():
        yield from get_possible_words(c_node)


def set_up_wordnet():
    from nltk.corpus import wordnet as wn

    all_common_words = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            name = lemma.name()
            if name[0].isupper():  # skip capitalized (likely proper nouns)
                continue
            if "-" in name:
                continue
            if "_" in name:
                continue
            if "'" in name:
                continue
            if "." in name:
                continue
            if re.match(r"\d+", name) is not None:
                continue
            if len(name) < 4:
                continue
            all_common_words.add(name)

    lines = sorted([f"{w}\n" for w in all_common_words])

    with Path("data/nltk_words.txt").open("w") as f:
        f.writelines(lines)
    print(f"Common words (excluding proper nouns): {len(all_common_words)}")


def main():
    words = load_words(Path("data/nltk_words.txt"))
    # words = list(filter(lambda w: len(w) <= 10, words))
    # words = [
    #     'banana',
    #     'anty',
    #     'band',
    # ]

    # Construct a trie using all possible substrings
    trie = Node()

    # rr.init("generate_trie")
    # rr.connect_tcp()

    SAVE_PERIOD = 25000
    trie_path = Path("data/trie.pkl")
    trie_json_path = Path("data/trie.json")
    trie_bt_path = Path("data/trie.bt")

    for word_idx, word in enumerate(tqdm(words)):
        for substr, idx in gen_substrings(word, word_idx):
            add_to_trie(trie, substr, idx, original_word=word)

    pass

    # print(f"Saving pkl {trie_path}")
    # with trie_path.open("wb") as trie_f:
    #     pickle.dump(trie, trie_f)
    #
    print(f"Saving json {trie_json_path}")
    trie_json = trie.to_json()

    with trie_json_path.open("w") as trie_json_f:
        json.dump(trie_json, trie_json_f)


if __name__ == "__main__":
    main()
