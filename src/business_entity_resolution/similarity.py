"""
similarity.py
=============
Self-contained string-similarity primitives.

No third-party fuzzy-matching library (python-Levenshtein, jellyfish,
rapidfuzz) is available in this environment, so Levenshtein edit distance
and Jaro-Winkler similarity are implemented here in pure Python/NumPy. Both
are small, dependency-free, deterministic, and unit-tested (see
tests/test_similarity.py).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Set


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # single-row DP
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def levenshtein_similarity(a: str, b: str) -> float:
    """1 - normalized edit distance, in [0, 1]. Empty/empty -> 1.0, empty/nonempty -> 0.0."""
    if a == "" and b == "":
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / max_len


def jaro_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    match_distance = max(la, lb) // 2 - 1
    match_distance = max(match_distance, 0)

    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0
    transpositions = 0

    for i in range(la):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, lb)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (matches / la + matches / lb + (matches - transpositions) / matches) / 3.0


def jaro_winkler_similarity(a: str, b: str, prefix_weight: float = 0.1, max_prefix: int = 4) -> float:
    jaro = jaro_similarity(a, b)
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
            if prefix_len == max_prefix:
                break
        else:
            break
    return jaro + prefix_len * prefix_weight * (1 - jaro)


def jaccard_similarity(set_a: Iterable, set_b: Iterable) -> float:
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def token_overlap_count(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> int:
    return len(set(tokens_a) & set(tokens_b))


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def common_suffix_len(a: str, b: str) -> int:
    return common_prefix_len(a[::-1], b[::-1])


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0
