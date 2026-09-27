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
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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


def _tfidf_topk_neighbors(
    s1_texts: List[str],
    target_texts: List[str],
    char_range,
    top_k: int,
    max_postings: int = 2000,
) -> np.ndarray:
    """
    Retrieve, for each s1 text, the indices (into target_texts) of the
    top_k nearest neighbors by TF-IDF cosine similarity.

    SCALABILITY FIX (2026-09-27): the previous implementation called
    sklearn.neighbors.NearestNeighbors(algorithm="brute") on the *entire*
    target matrix. Even though the TF-IDF transform itself is vectorized,
    brute-force cosine kneighbors still costs O(n_queries * n_targets):
    every query is compared against every target to find the nearest ones.
    At this project's real scale (S1 ~2.2M rows, S2+S3 targets ~10M rows)
    that is on the order of 10^13 comparisons -- months of wall-clock time,
    confirmed by direct benchmarking, not a "let it run longer" problem.

    The fix: cosine similarity between two TF-IDF vectors is exactly zero
    unless they share at least one character n-gram, so nearest neighbors
    can never come from outside that overlap set. We therefore build an
    inverted index (n-gram feature -> posting list of target rows) directly
    from the fitted TF-IDF matrix, look up only the candidate rows that
    share a feature with each query, and score cosine similarity on just
    that restricted (typically tiny) candidate set via a sparse dot
    product. This returns mathematically the same top-k neighbors as brute
    force (both are exact cosine top-k over the shared-feature candidate
    space), but the cost per query scales with how many targets share an
    n-gram with it, not with total corpus size.

    Postings for a feature that appears in more than `max_postings` target
    rows are dropped as uninformative (mirrors the existing
    COMMON_TOKEN_STOPSET / max_token_block_size treatment of overly common
    tokens elsewhere in this module) -- this is what keeps per-query cost
    bounded even for extremely common n-grams (e.g. "the", padding grams).

    Fit only ever happens on the corpus being retrieved *from* (S2/S3
    side), never on validation labels, so this introduces no leakage.
    """
    non_empty_idx = [i for i, t in enumerate(target_texts) if t]
    if not non_empty_idx:
        return np.array([[] for _ in s1_texts], dtype=object)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=char_range, min_df=1)
    target_matrix = vectorizer.fit_transform([target_texts[i] for i in non_empty_idx]).tocsr()

    s1_nonempty_mask = [bool(t) for t in s1_texts]
    results = [np.array([], dtype=int) for _ in s1_texts]
    if not any(s1_nonempty_mask):
        return np.array(results, dtype=object)

    s1_query_texts = [t for t, keep in zip(s1_texts, s1_nonempty_mask) if keep]
    s1_matrix = vectorizer.transform(s1_query_texts).tocsr()

    # Inverted index: feature index -> array of target row indices that
    # contain it, dropped entirely if it's too common to be informative.
    target_matrix_csc = target_matrix.tocsc()
    n_features = target_matrix.shape[1]
    feature_postings: List[Any] = [None] * n_features
    for f in range(n_features):
        start, end = target_matrix_csc.indptr[f], target_matrix_csc.indptr[f + 1]
        if end - start <= max_postings:
            feature_postings[f] = target_matrix_csc.indices[start:end]

    query_pos = 0
    for i, keep in enumerate(s1_nonempty_mask):
        if not keep:
            continue
        row = s1_matrix.getrow(query_pos)
        query_pos += 1

        cand_rows: Set[int] = set()
        for f in row.indices:
            postings = feature_postings[f]
            if postings is not None:
                cand_rows.update(postings.tolist())

        if cand_rows:
            cand_arr = np.fromiter(cand_rows, dtype=np.int64, count=len(cand_rows))
            sims = target_matrix[cand_arr].dot(row.T).toarray().ravel()
            k = min(top_k, len(cand_arr))
            if k < len(sims):
                top_local = np.argpartition(-sims, k - 1)[:k]
            else:
                top_local = np.arange(len(sims))
            top_local = top_local[np.argsort(-sims[top_local])]
            local_indices = cand_arr[top_local]
            results[i] = np.array([non_empty_idx[j] for j in local_indices], dtype=int)

    return np.array(results, dtype=object)


def _trim_to_cap(eid: str, cand: Set[str], s1_corpus: NormalizedCorpus, target_corpus: NormalizedCorpus, max_candidates: int) -> Set[str]:
    """
    Cap candidate volume per S1 entity if it explodes on a pathologically
    common name. We keep the candidates whose normalized names are closest
    in length to the S1 name as a cheap, model-free primary ranking key.

    BUGFIX (reproducibility): `cand` is a Python set, and CPython's string
    hashing is randomized per-process (PYTHONHASHSEED) unless pinned, so
    iterating a set of entity-id strings is NOT guaranteed to produce the
    same order across runs even with `seed_everything()` called (that only
    seeds `random`/`numpy`, not hash-based container ordering). Since
    `sorted()` is stable, ties on the length-diff key were previously
    broken by whatever order the set happened to iterate in that process --
    i.e. which candidates survive the cap could silently change between
    runs on identical input/config. We add the candidate id itself as a
    secondary sort key so ties are broken the same way on every run, on
    every machine, regardless of hash seed.
    """
    if len(cand) <= max_candidates:
        return cand
    s1_len = len(s1_corpus.names[eid].normalized)
    ranked = sorted(
        cand,
        key=lambda cid: (abs(len(target_corpus.names[cid].normalized) - s1_len), cid),
    )
    return set(ranked[:max_candidates])


def generate_candidates(
    s1_corpus: NormalizedCorpus,
    target_corpus: NormalizedCorpus,
    cfg: BlockingConfig,
) -> Dict[str, Set[str]]:
    """
    Returns: dict mapping each s1 entity_id -> set of candidate entity_ids
    drawn from target_corpus (which should be S2+S3 combined).

    MEMORY-BOUND FIX (2026-09-27): the cap on candidates-per-entity used to
    be applied only once, after every strategy had finished accumulating
    into every entity's set. A single S1 entity can pick up candidates from
    several independent strategies (name tokens, address tokens, postal
    code, n-gram/fuzzy retrieval), each contributing up to hundreds of ids,
    so the *uncapped* per-entity union could run into the thousands --
    held simultaneously in memory for every S1 entity at once. At real
    scale (millions of S1 entities) that is what exhausted this sandbox's
    RAM at only 5% of the real target-corpus size. The cap is now applied
    after every strategy contributes, so no entity's candidate set is ever
    allowed to grow past `max_candidates_per_entity` for more than the
    duration of a single strategy's update -- bounding peak memory to
    O(n_s1_entities * max_candidates_per_entity) throughout, not just at
    the end.
    """
    candidates: Dict[str, Set[str]] = {eid: set() for eid in s1_corpus.ids}

    def _cap_all() -> None:
        for eid in s1_corpus.ids:
            if len(candidates[eid]) > cfg.max_candidates_per_entity:
                candidates[eid] = _trim_to_cap(eid, candidates[eid], s1_corpus, target_corpus, cfg.max_candidates_per_entity)

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
        _cap_all()

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
        _cap_all()

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
        _cap_all()

    # --- C. Character n-gram TF-IDF nearest neighbor retrieval on names ---
    if cfg.enable_ngram_block and target_corpus.ids:
        s1_name_texts = [s1_corpus.names[eid].normalized for eid in s1_corpus.ids]
        target_name_texts = [target_corpus.names[eid].normalized for eid in target_corpus.ids]
        try:
            neighbor_sets = _tfidf_topk_neighbors(
                s1_name_texts, target_name_texts, cfg.ngram_char_range, cfg.ngram_top_k, cfg.ngram_max_postings
            )
            for i, eid in enumerate(s1_corpus.ids):
                for idx in neighbor_sets[i]:
                    candidates[eid].add(target_corpus.ids[idx])
        except ValueError:
            logger.warning("name n-gram blocking skipped: empty vocabulary")
        _cap_all()

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
            neighbor_sets = _tfidf_topk_neighbors(
                s1_combo, target_combo, cfg.fuzzy_char_range, cfg.fuzzy_top_k, cfg.fuzzy_max_postings
            )
            for i, eid in enumerate(s1_corpus.ids):
                for idx in neighbor_sets[i]:
                    candidates[eid].add(target_corpus.ids[idx])
        except ValueError:
            logger.warning("fuzzy blocking skipped: empty vocabulary")
        _cap_all()

    return candidates


def build_corpus(df: pd.DataFrame, normalized_names: Dict[str, NormalizedName], normalized_addresses: Dict[str, NormalizedAddress]) -> NormalizedCorpus:
    return NormalizedCorpus(df, normalized_names, normalized_addresses)
