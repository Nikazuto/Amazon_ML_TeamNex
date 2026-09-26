"""
blocking.py
===========
Hybrid candidate generation (Requirements Analysis §6, FR-3.1-3.4).

Candidate generation is the recall ceiling of the entire pipeline: any true
match not produced here can never be recovered later. We therefore generate
candidates as the UNION of several independent, complementary strategies
rather than a single blocking key, and we never use country as a hard
filter (country is unseen for France at test time -- FR-1.3).

Strategies implemented (all configurable via BlockingConfig):
  A. Exact normalized-name / token-sorted-name blocking (inverted index)
  B. Token-based blocking on rare/informative name tokens (inverted index,
     with a cap on how common a token may be before it's excluded as
     uninformative -- a token appearing in most records carries no
     blocking signal and would blow up candidate volume)
  C. Character n-gram TF-IDF nearest-neighbor retrieval on names
  D. Address-based blocking: shared normalized-address tokens (inverted
     index) plus a TF-IDF nearest-neighbor retrieval on addresses
  E. Combined name+address blocking is implicit: it is the union of A-D
  F. Fuzzy retrieval: character n-gram TF-IDF nearest-neighbor over the
     concatenated name+address representation, which tolerates typos and
     transliteration better than any exact-key method

The final candidate set for every S1 entity -- exactly what will be fed to
the model at inference -- is what this module returns; it becomes
candidate_pairs.tsv directly (FR-3.3).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .config import COL_ENTITY_ID, BlockingConfig
from .normalization import NormalizedAddress, NormalizedName, char_ngrams

logger = logging.getLogger(__name__)

COMMON_TOKEN_STOPSET = {
    "the", "and", "of", "group", "company", "corporation", "limited",
    "international", "national", "private", "incorporated",
}


class NormalizedCorpus:
    """Precomputed normalized name/address representations for a set of records, keyed by entity_id."""

    def __init__(self, df: pd.DataFrame, names: Dict[str, NormalizedName], addresses: Dict[str, NormalizedAddress]):
        self.ids: List[str] = df[COL_ENTITY_ID].tolist()
        self.names = names
        self.addresses = addresses


def _build_inverted_index_by_key(ids: List[str], key_fn) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for eid in ids:
        key = key_fn(eid)
        if key:
            index[key].append(eid)
    return index


def _build_token_index(ids: List[str], token_fn, min_len: int, max_block_size: int) -> Dict[str, List[str]]:
    raw_index: Dict[str, List[str]] = defaultdict(list)
    for eid in ids:
        for tok in token_fn(eid):
            if len(tok) < min_len or tok in COMMON_TOKEN_STOPSET:
                continue
            raw_index[tok].append(eid)
    return {tok: lst for tok, lst in raw_index.items() if len(lst) <= max_block_size}


def _tfidf_topk_neighbors(s1_texts: List[str], target_texts: List[str], char_range, top_k: int) -> np.ndarray:
    """
    Fit a char-ngram TF-IDF vectorizer on target_texts and retrieve, for each
    s1 text, the indices (into target_texts) of the top_k nearest neighbors
    by cosine similarity. Returns an array of shape (len(s1_texts), <=top_k).
    Fit only ever happens on the corpus being retrieved *from* (S2/S3 side),
    never on validation labels, so this introduces no leakage.
    """
    non_empty_idx = [i for i, t in enumerate(target_texts) if t]
    if not non_empty_idx:
        return np.array([[] for _ in s1_texts], dtype=object)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=char_range, min_df=1)
    target_matrix = vectorizer.fit_transform([target_texts[i] for i in non_empty_idx])

    s1_nonempty_mask = [bool(t) for t in s1_texts]
    results = [np.array([], dtype=int) for _ in s1_texts]
    if not any(s1_nonempty_mask):
        return np.array(results, dtype=object)

    s1_query_texts = [t for t, keep in zip(s1_texts, s1_nonempty_mask) if keep]
    s1_matrix = vectorizer.transform(s1_query_texts)

    k = min(top_k, target_matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(target_matrix)
    _, neighbor_idx = nn.kneighbors(s1_matrix)

    query_pos = 0
    for i, keep in enumerate(s1_nonempty_mask):
        if keep:
            local_indices = neighbor_idx[query_pos]
            results[i] = np.array([non_empty_idx[j] for j in local_indices], dtype=int)
            query_pos += 1
    return np.array(results, dtype=object)


def generate_candidates(
    s1_corpus: NormalizedCorpus,
    target_corpus: NormalizedCorpus,
    cfg: BlockingConfig,
) -> Dict[str, Set[str]]:
    """
    Returns: dict mapping each s1 entity_id -> set of candidate entity_ids
    drawn from target_corpus (which should be S2+S3 combined).
    """
    candidates: Dict[str, Set[str]] = {eid: set() for eid in s1_corpus.ids}

    # --- A. Exact normalized-name / token-sorted-name blocking ---
    if cfg.enable_exact_name_block:
        exact_index = _build_inverted_index_by_key(target_corpus.ids, lambda eid: target_corpus.names[eid].normalized)
        sorted_index = _build_inverted_index_by_key(target_corpus.ids, lambda eid: target_corpus.names[eid].token_sorted)
        for eid in s1_corpus.ids:
            nname = s1_corpus.names[eid]
            if nname.normalized:
                candidates[eid].update(exact_index.get(nname.normalized, []))
            if nname.token_sorted:
                candidates[eid].update(sorted_index.get(nname.token_sorted, []))

    # --- B. Token-based blocking on informative name tokens ---
    if cfg.enable_token_block:
        token_index = _build_token_index(
            target_corpus.ids,
            lambda eid: target_corpus.names[eid].dedup_tokens,
            cfg.min_token_length,
            cfg.max_token_block_size,
        )
        for eid in s1_corpus.ids:
            for tok in s1_corpus.names[eid].dedup_tokens:
                if len(tok) < cfg.min_token_length or tok in COMMON_TOKEN_STOPSET:
                    continue
                candidates[eid].update(token_index.get(tok, []))

    # --- D. Address-based blocking: shared normalized-address tokens ---
    if cfg.enable_address_block:
        addr_token_index = _build_token_index(
            target_corpus.ids,
            lambda eid: target_corpus.addresses[eid].tokens,
            min_len=cfg.address_min_token_length,
            max_block_size=cfg.max_token_block_size,
        )
        for eid in s1_corpus.ids:
            for tok in s1_corpus.addresses[eid].tokens:
                if len(tok) < cfg.address_min_token_length or tok in COMMON_TOKEN_STOPSET:
                    continue
                candidates[eid].update(addr_token_index.get(tok, []))

        # postal code exact match is a strong, cheap blocking key when present
        postal_index = _build_inverted_index_by_key(target_corpus.ids, lambda eid: target_corpus.addresses[eid].postal_code or "")
        for eid in s1_corpus.ids:
            pcode = s1_corpus.addresses[eid].postal_code
            if pcode:
                candidates[eid].update(postal_index.get(pcode, []))

    # --- C. Character n-gram TF-IDF nearest neighbor retrieval on names ---
    if cfg.enable_ngram_block and target_corpus.ids:
        s1_name_texts = [s1_corpus.names[eid].normalized for eid in s1_corpus.ids]
        target_name_texts = [target_corpus.names[eid].normalized for eid in target_corpus.ids]
        try:
            neighbor_sets = _tfidf_topk_neighbors(s1_name_texts, target_name_texts, cfg.ngram_char_range, cfg.ngram_top_k)
            for i, eid in enumerate(s1_corpus.ids):
                for idx in neighbor_sets[i]:
                    candidates[eid].add(target_corpus.ids[idx])
        except ValueError:
            logger.warning("name n-gram blocking skipped: empty vocabulary")

    # --- F. Fuzzy retrieval on combined name+address char n-grams ---
    if cfg.enable_fuzzy_block and target_corpus.ids:
        s1_combo = [
            f"{s1_corpus.names[eid].normalized} {s1_corpus.addresses[eid].normalized}".strip()
            for eid in s1_corpus.ids
        ]
        target_combo = [
            f"{target_corpus.names[eid].normalized} {target_corpus.addresses[eid].normalized}".strip()
            for eid in target_corpus.ids
        ]
        try:
            neighbor_sets = _tfidf_topk_neighbors(s1_combo, target_combo, cfg.fuzzy_char_range, cfg.fuzzy_top_k)
            for i, eid in enumerate(s1_corpus.ids):
                for idx in neighbor_sets[i]:
                    candidates[eid].add(target_corpus.ids[idx])
        except ValueError:
            logger.warning("fuzzy blocking skipped: empty vocabulary")

    # Safety valve: cap candidate volume per S1 entity if it explodes on a
    # pathologically common name. We keep the candidates whose normalized
    # names are closest in length to the S1 name as a cheap, model-free
    # primary ranking key.
    #
    # BUGFIX (reproducibility): `cand` is a Python set, and CPython's string
    # hashing is randomized per-process (PYTHONHASHSEED) unless pinned, so
    # iterating a set of entity-id strings is NOT guaranteed to produce the
    # same order across runs even with `seed_everything()` called (that only
    # seeds `random`/`numpy`, not hash-based container ordering). Since
    # `sorted()` is stable, ties on the length-diff key were previously
    # broken by whatever order the set happened to iterate in that process
    # -- i.e. which candidates survive the cap could silently change between
    # runs on identical input/config, violating the "deterministic,
    # reproducible" contract this module must uphold. We add the candidate
    # id itself as a secondary sort key so ties are broken the same way on
    # every run, on every machine, regardless of hash seed.
    for eid in s1_corpus.ids:
        cand = candidates[eid]
        if len(cand) > cfg.max_candidates_per_entity:
            s1_len = len(s1_corpus.names[eid].normalized)
            ranked = sorted(
                cand,
                key=lambda cid: (abs(len(target_corpus.names[cid].normalized) - s1_len), cid),
            )
            candidates[eid] = set(ranked[: cfg.max_candidates_per_entity])

    return candidates


def build_corpus(df: pd.DataFrame, normalized_names: Dict[str, NormalizedName], normalized_addresses: Dict[str, NormalizedAddress]) -> NormalizedCorpus:
    return NormalizedCorpus(df, normalized_names, normalized_addresses)
