"""
validation_split.py
====================
Leakage-safe train/validation split at the S1-entity level (FR-5.4, NFR-9).

Splitting at the S1 level (rather than at the pair or S2/S3 level) guarantees
that no information about a given S1 entity -- its ground-truth matches, its
candidate pairs, its features -- crosses between train and validation. This
split must be constructed once, before any blocking-parameter, feature-
parameter, model, or threshold selection happens, and reused unchanged
throughout tuning.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import COL_COUNTRY, COL_ENTITY_ID, SplitConfig


@dataclass
class SplitResult:
    train_s1_ids: List[str]
    val_s1_ids: List[str]
    holdout_country: Optional[str] = None
    holdout_s1_ids: Optional[List[str]] = None  # S1 ids from the held-out proxy country (diagnostic only)


def split_s1_entities(s1_df: pd.DataFrame, cfg: SplitConfig) -> SplitResult:
    """
    Split S1 entity IDs into train/validation sets.

    If cfg.holdout_country_proxy is set, S1 entities of that country are
    removed from BOTH train and validation and returned separately as a
    diagnostic-only "unseen country" proxy set (a stand-in for the real
    unseen France scenario, which cannot be tested during development since
    no France examples exist in training). This proxy set is never used for
    fitting or tuning -- it exists only to sanity-check generalization.
    """
    rng = random.Random(cfg.split_seed)
    all_ids = s1_df[COL_ENTITY_ID].tolist()

    holdout_ids = None
    working_ids = all_ids
    if cfg.holdout_country_proxy:
        target = cfg.holdout_country_proxy.strip().lower()
        country_series = s1_df[COL_COUNTRY].astype(str).str.strip().str.lower()
        holdout_mask = country_series == target
        holdout_ids = s1_df.loc[holdout_mask, COL_ENTITY_ID].tolist()
        working_ids = s1_df.loc[~holdout_mask, COL_ENTITY_ID].tolist()

    ids_shuffled = list(working_ids)
    rng.shuffle(ids_shuffled)
    n_val = max(1, int(round(len(ids_shuffled) * cfg.validation_fraction))) if ids_shuffled else 0
    val_ids = set(ids_shuffled[:n_val])
    train_ids = [i for i in working_ids if i not in val_ids]
    val_ids_ordered = [i for i in working_ids if i in val_ids]

    return SplitResult(
        train_s1_ids=train_ids,
        val_s1_ids=val_ids_ordered,
        holdout_country=cfg.holdout_country_proxy,
        holdout_s1_ids=holdout_ids,
    )
