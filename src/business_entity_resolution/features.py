"""
features.py
============
Pairwise similarity feature extraction for candidate (S1, S2/S3) pairs.

Covers name features, address features, country (soft signal only, per
FR-1.3), missing-data indicators, and interaction features
(Requirements Analysis §10 / §5 table). Every feature is computed from the
supplied dataset alone -- no external lookups.

Missing fields never silently produce a misleadingly high or low
similarity: each comparison has an explicit "missing" indicator feature,
and similarity features default to a neutral (not maximal) value when a
field is absent on either side, so a missing address does not masquerade as
either a strong match or a strong mismatch.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import COL_ENTITY_ID
from .normalization import NormalizedAddress, NormalizedName, char_ngrams, normalize_country
from .similarity import (
    common_prefix_len,
    common_suffix_len,
    jaccard_similarity,
    jaro_winkler_similarity,
    length_ratio,
    levenshtein_similarity,
    token_overlap_count,
)

NEUTRAL_SIM = 0.0  # value used for similarity features when a field is missing on either side

FEATURE_NAMES: List[str] = [
    # name features
    "name_exact_match", "name_token_sorted_exact_match", "name_token_jaccard",
    "name_token_overlap_count", "name_char_ngram_cosine", "name_levenshtein_sim",
    "name_jaro_winkler_sim", "name_token_count_diff", "name_length_ratio",
    "name_common_token_count", "name_prefix_sim", "name_suffix_sim",
    # address features
    "addr_exact_match", "addr_token_jaccard", "addr_token_overlap_count",
    "addr_char_ngram_cosine", "addr_levenshtein_sim", "addr_numeric_token_overlap",
    "addr_postal_equal", "addr_length_ratio", "addr_shared_token_count",
    # country
    "country_exact_match", "country_missing_either",
    # missing-data indicators
    "name_missing_s1", "name_missing_target", "addr_missing_s1", "addr_missing_target",
    "postal_missing_s1", "postal_missing_target",
    # interactions
    "high_name_high_addr", "high_name_low_addr", "low_name_high_addr",
    "country_agrees_and_name_high", "country_agrees_and_addr_high",
]

HIGH_SIM_THRESHOLD = 0.75


def _char_ngram_cosine(a: str, b: str, n_range=(2, 4)) -> float:
    if not a or not b:
        return NEUTRAL_SIM
    try:
        vec = TfidfVectorizer(analyzer="char", ngram_range=n_range, min_df=1)
        mat = vec.fit_transform([a, b])
        num = (mat[0].multiply(mat[1])).sum()
        denom = np.sqrt((mat[0].multiply(mat[0])).sum()) * np.sqrt((mat[1].multiply(mat[1])).sum())
        return float(num / denom) if denom else NEUTRAL_SIM
    except ValueError:
        return NEUTRAL_SIM


def compute_pair_features(
    s1_id: str, target_id: str,
    s1_names: Dict[str, NormalizedName], s1_addrs: Dict[str, NormalizedAddress], s1_countries: Dict[str, str],
    t_names: Dict[str, NormalizedName], t_addrs: Dict[str, NormalizedAddress], t_countries: Dict[str, str],
) -> Dict[str, float]:
    n1, n2 = s1_names[s1_id], t_names[target_id]
    a1, a2 = s1_addrs[s1_id], t_addrs[target_id]
    c1, c2 = s1_countries.get(s1_id, ""), t_countries.get(target_id, "")

    feats: Dict[str, float] = {}

    name_missing = n1.is_missing or n2.is_missing
    feats["name_missing_s1"] = float(n1.is_missing)
    feats["name_missing_target"] = float(n2.is_missing)
    if name_missing:
        feats.update({
            "name_exact_match": 0.0, "name_token_sorted_exact_match": 0.0, "name_token_jaccard": NEUTRAL_SIM,
            "name_token_overlap_count": 0.0, "name_char_ngram_cosine": NEUTRAL_SIM, "name_levenshtein_sim": NEUTRAL_SIM,
            "name_jaro_winkler_sim": NEUTRAL_SIM, "name_token_count_diff": 0.0, "name_length_ratio": NEUTRAL_SIM,
            "name_common_token_count": 0.0, "name_prefix_sim": NEUTRAL_SIM, "name_suffix_sim": NEUTRAL_SIM,
        })
    else:
        feats["name_exact_match"] = float(n1.normalized == n2.normalized)
        feats["name_token_sorted_exact_match"] = float(n1.token_sorted == n2.token_sorted)
        feats["name_token_jaccard"] = jaccard_similarity(n1.tokens, n2.tokens)
        overlap = token_overlap_count(n1.tokens, n2.tokens)
        feats["name_token_overlap_count"] = float(overlap)
        feats["name_common_token_count"] = float(overlap)
        feats["name_char_ngram_cosine"] = _char_ngram_cosine(n1.normalized, n2.normalized)
        feats["name_levenshtein_sim"] = levenshtein_similarity(n1.normalized, n2.normalized)
        feats["name_jaro_winkler_sim"] = jaro_winkler_similarity(n1.normalized, n2.normalized)
        feats["name_token_count_diff"] = float(abs(len(n1.tokens) - len(n2.tokens)))
        feats["name_length_ratio"] = length_ratio(n1.normalized, n2.normalized)
        max_prefix = max(len(n1.normalized), len(n2.normalized)) or 1
        feats["name_prefix_sim"] = common_prefix_len(n1.normalized, n2.normalized) / max_prefix
        feats["name_suffix_sim"] = common_suffix_len(n1.normalized, n2.normalized) / max_prefix

    addr_missing = a1.is_missing or a2.is_missing
    feats["addr_missing_s1"] = float(a1.is_missing)
    feats["addr_missing_target"] = float(a2.is_missing)
    if addr_missing:
        feats.update({
            "addr_exact_match": 0.0, "addr_token_jaccard": NEUTRAL_SIM, "addr_token_overlap_count": 0.0,
            "addr_char_ngram_cosine": NEUTRAL_SIM, "addr_levenshtein_sim": NEUTRAL_SIM,
            "addr_numeric_token_overlap": NEUTRAL_SIM, "addr_length_ratio": NEUTRAL_SIM,
            "addr_shared_token_count": 0.0,
        })
    else:
        feats["addr_exact_match"] = float(a1.normalized == a2.normalized)
        feats["addr_token_jaccard"] = jaccard_similarity(a1.tokens, a2.tokens)
        overlap = token_overlap_count(a1.tokens, a2.tokens)
        feats["addr_token_overlap_count"] = float(overlap)
        feats["addr_shared_token_count"] = float(overlap)
        feats["addr_char_ngram_cosine"] = _char_ngram_cosine(a1.normalized, a2.normalized)
        feats["addr_levenshtein_sim"] = levenshtein_similarity(a1.normalized, a2.normalized)
        feats["addr_numeric_token_overlap"] = jaccard_similarity(a1.numeric_tokens, a2.numeric_tokens)
        feats["addr_length_ratio"] = length_ratio(a1.normalized, a2.normalized)

    feats["postal_missing_s1"] = float(a1.postal_code is None)
    feats["postal_missing_target"] = float(a2.postal_code is None)
    if a1.postal_code and a2.postal_code:
        feats["addr_postal_equal"] = float(a1.postal_code == a2.postal_code)
    else:
        feats["addr_postal_equal"] = NEUTRAL_SIM

    c1n, c2n = normalize_country(c1), normalize_country(c2)
    feats["country_missing_either"] = float(not c1n or not c2n)
    feats["country_exact_match"] = float(bool(c1n) and bool(c2n) and c1n == c2n)

    name_sim = feats["name_char_ngram_cosine"] if not name_missing else NEUTRAL_SIM
    addr_sim = feats["addr_char_ngram_cosine"] if not addr_missing else NEUTRAL_SIM
    feats["high_name_high_addr"] = float(name_sim >= HIGH_SIM_THRESHOLD and addr_sim >= HIGH_SIM_THRESHOLD)
    feats["high_name_low_addr"] = float(name_sim >= HIGH_SIM_THRESHOLD and addr_sim < HIGH_SIM_THRESHOLD)
    feats["low_name_high_addr"] = float(name_sim < HIGH_SIM_THRESHOLD and addr_sim >= HIGH_SIM_THRESHOLD)
    feats["country_agrees_and_name_high"] = float(feats["country_exact_match"] == 1.0 and name_sim >= HIGH_SIM_THRESHOLD)
    feats["country_agrees_and_addr_high"] = float(feats["country_exact_match"] == 1.0 and addr_sim >= HIGH_SIM_THRESHOLD)

    return {k: feats.get(k, 0.0) for k in FEATURE_NAMES}


def extract_features_for_pairs(
    pairs: List[tuple],
    s1_names: Dict[str, NormalizedName], s1_addrs: Dict[str, NormalizedAddress], s1_countries: Dict[str, str],
    t_names: Dict[str, NormalizedName], t_addrs: Dict[str, NormalizedAddress], t_countries: Dict[str, str],
) -> pd.DataFrame:
    """pairs: list of (s1_entity_id, target_entity_id) tuples."""
    rows = []
    for s1_id, target_id in pairs:
        feats = compute_pair_features(s1_id, target_id, s1_names, s1_addrs, s1_countries, t_names, t_addrs, t_countries)
        feats["source1_entity_id"] = s1_id
        feats["target_entity_id"] = target_id
        rows.append(feats)
    cols = ["source1_entity_id", "target_entity_id"] + FEATURE_NAMES
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]
