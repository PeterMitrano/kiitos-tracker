import json
import math
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Sequence, Union, Tuple, Dict

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
class Index:
    word_idx: int
    substring_idx: int

    def __repr__(self):
        return f"({self.word_idx},{self.substring_idx})"

    def __str__(self):
        return f"({self.word_idx},{self.substring_idx})"

    def __hash__(self):
        return 1_000_000 * self.word_idx + self.substring_idx


StrOrIdx = Union[str, Index]


@dataclass
class Node:
    name: StrOrIdx
    children: Dict[StrOrIdx, "Node"] = None
    original_word = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    def __eq__(self, other):
        return self.name == other.name

    def find_child_by_name(self, name):
        return self.children.get(name)

    def get_name(self):
        if self.original_word is not None:
            return f"{self.name} {self.original_word}"
        else:
            return self.name


def gen_substrings(word: Sequence, word_idx: int):
    """
    yields all substrings of w, as well as its unique ID tuple (word_idx, suffix_idx)
    """
    word_len = len(word)
    suffix_idx = 0
    for suffix_len in range(1, word_len + 1):
        for start in range(word_len - suffix_len + 1):
            yield word[start : start + suffix_len], Index(word_idx, suffix_idx)
            suffix_idx += 1


def trie_to_json(node: Node | StrOrIdx):
    if isinstance(node.name, Index):
        return node.original_word
    else:
        return {node.name: [trie_to_json(c) for c in node.children]}


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


# def add_to_trie(trie: Node, substr, original_word=None):
#     c0 = substr[0]
#     if c0 not in trie.children:
#         trie.children[c0] = Node(c0)
#         if isinstance(c0, Index):
#             trie.children[c0].original_word = original_word
#
#     if len(substr) > 1:
#         # recursively add the rest of the substr
#         add_to_trie(trie.children[c0], substr[1:], original_word=original_word)
#

def add_to_trie(trie: Node, substr, original_word=None):
    node = trie
    for c in substr:
        if c not in node.children:
            node.children[c] = Node(c)
        node = node.children[c]  # Move deeper in the trie
    node.original_word = original_word

def get_possible_words(node: Node):
    if node.original_word is not None:
        yield node.original_word
    for c in node.children:
        yield from get_possible_words(c)


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
    # words = [
    #     'banana',
    #     'anty',
    #     'band',
    # ]

    # Construct a trie using all possible substrings
    trie = Node("")

    rr.init("generate_trie")
    rr.connect_tcp()

    for word_idx, word in enumerate(tqdm(words)):
        for substr, idx in gen_substrings(word, word_idx):
            substr_seq = tuple(substr) + (idx,)
            add_to_trie(trie, substr_seq, original_word=word)

        # Save the data structure periodically
        if word_idx % 2500 == 0 and word_idx > 2500:
            trie_path = Path("data/trie.pkl")
            with trie_path.open("wb") as trie_f:
                pickle.dump(trie, trie_f)

    # viz_trie(trie)

    # trie_path = Path("data/trie.pkl")
    with trie_path.open("wb") as trie_f:
        pickle.dump(trie, trie_f)


if __name__ == "__main__":
    main()
