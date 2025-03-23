import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union, Tuple

import rerun as rr


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


CharLike = Union[str, Index]


@dataclass
class Node:
    name: str
    children: Tuple["Node", ...] = ()


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


def trie_to_graph(node: Node, vertices=(), labels=(), edges=(), vertex_idx=0):
    child_vertex_idx = vertex_idx + 1

    vertices += (vertex_idx,)
    labels += (node.name,)
    for child in node.children:
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


def main():
    # words = load_words()
    # words = load_words(Path("data/words_alpha_modified.txt"))
    #
    # words = [
    #     "then",
    #     "theme",
    #     "toon",
    #     "that",
    #     "moon",
    # ]
    #
    # # figure out how many suffixes exist in the dictionary
    # n_suffixes = sum([triangle_number(len(w)) for w in words])
    # print(f"Expecting {n_suffixes=:,d}")
    #
    # n_chars = sum([staircase_number(len(w)) for w in words])
    # print(f"Which would have a total of {n_chars=:,d}")
    #
    # # Construct a trie using all possible substrings
    # # trie: Mapping[CharLike, CharLike] = {}
    #
    # for word_idx, word in enumerate(words):
    #     for substr, idx in gen_substrings(word, word_idx):
    #         print(substr, idx)

    trie = Node(
        name="t",
        children=(
            Node(
                name="o",
                children=(Node(name="n"), Node(name="p")),
            ),
            Node(name="i", children=(Node(name="n"),)),
        ),
    )

    viz_trie(trie)

    rr.init("kiito_agent_trie")
    rr.connect_tcp()


if __name__ == "__main__":
    main()
