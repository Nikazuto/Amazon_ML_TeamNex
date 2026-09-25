"""
inference.py
============
Ties blocking + feature extraction + a trained model together into a single
"score every candidate pair for this S1 set" step (Requirements Analysis
§17 / SRS FR-3.3, FR-4.1/4.2, FR-5.1).

This module is deliberately the ONLY place that calls both blocking.py and
features.py back-to-back, so that "the candidate set immediately before
model scoring" (what candidate_pairs.tsv must contain, per the problem
statement) and "the pairs actually scored" are provably the same object --
there is no separate/earlier candidate list that later gets silently
filtered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import pandas as pd

from .blocking import NormalizedCorpus, generate_candidates
from .config import BlockingConfig
from .features import extract_features_for_pairs
from .normalization import NormalizedAddress, NormalizedName
from .training import TrainedModel


@dataclass
class ScoredCandidates:
    candidates: Dict[str, Set[str]]     # final candidate set (== candidate_pairs.tsv content)
    scored_pairs: pd.DataFrame          # source1_entity_id, target_entity_id, score (+ features)


def generate_and_score(
    s1_corpus: NormalizedCorpus,
    target_corpus: NormalizedCorpus,
    s1_names: Dict[str, NormalizedName],
    s1_addrs: Dict[str, NormalizedAddress],
    s1_countries: Dict[str, str],
    t_names: Dict[str, NormalizedName],
    t_addrs: Dict[str, NormalizedAddress],
    t_countries: Dict[str, str],
    blocking_cfg: BlockingConfig,
    model: TrainedModel,
) -> ScoredCandidates:
    """
    Runs blocking to get the final candidate set for every id in
    s1_corpus.ids, extracts pairwise features for every candidate pair, and
    scores them with `model`. Returns both the candidate set and the scored
    pairs so callers can write candidate_pairs.tsv from the exact same
    object that was scored.
    """
    candidates = generate_candidates(s1_corpus, target_corpus, blocking_cfg)

    pairs = []
    for s1_id, cand_set in candidates.items():
        for target_id in cand_set:
            pairs.append((s1_id, target_id))

    feature_df = extract_features_for_pairs(
        pairs, s1_names, s1_addrs, s1_countries, t_names, t_addrs, t_countries
    )

    if len(feature_df) == 0:
        scored = feature_df.copy()
        scored["score"] = pd.Series(dtype=float)
    else:
        scored = feature_df.copy()
        scored["score"] = model.predict_proba(feature_df)

    # Ensure every S1 id (even those with zero candidates) is represented so
    # downstream decision logic never silently drops an entity.
    for s1_id in s1_corpus.ids:
        if s1_id not in candidates:
            candidates[s1_id] = set()

    return ScoredCandidates(candidates=candidates, scored_pairs=scored)
