"""
training.py
============
Positive/negative training-pair construction and supervised pair-classifier
training (Requirements Analysis §9, §7).

Positives come directly from train_ground_truth.tsv (restricted to the
training-split S1 ids). Negatives are sampled from the blocking candidate
set MINUS the true positives -- never from the full Cartesian product --
so the model learns the actual decision boundary it will face at inference
(hard negatives that survived blocking, not trivially-dissimilar random
pairs).

Two model families are trained on the identical feature set and compared
on the same leakage-free validation split (selection happens in
pipeline.py, which also incorporates threshold calibration -- see
threshold.py):
  1. Logistic Regression (scikit-learn, BSD-3-Clause) -- simplest, most
     auditable baseline.
  2. HistGradientBoostingClassifier (scikit-learn, BSD-3-Clause) -- a
     gradient-boosted-tree model used in place of XGBoost, which is not
     available in this offline environment. Both models are trained from
     scratch on the challenge's own data only; neither loads any pretrained
     weights, so no third-party *model* license applies -- only the
     training library's license (scikit-learn, BSD-3-Clause, a permissive
     license) does. Both are far below the 8B-parameter ceiling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import GT_COL_MATCHES, GT_COL_S1, ModelConfig, NegativeSamplingConfig
from .data_loader import parse_id_list
from .features import FEATURE_NAMES


def build_positive_pairs(ground_truth: pd.DataFrame, s1_ids_allowed: Set[str]) -> Set[Tuple[str, str]]:
    positives: Set[Tuple[str, str]] = set()
    for _, row in ground_truth.iterrows():
        s1_id = row[GT_COL_S1]
        if s1_id not in s1_ids_allowed:
            continue
        for match_id in parse_id_list(row[GT_COL_MATCHES]):
            positives.add((s1_id, match_id))
    return positives


def build_training_pairs(
    candidates: Dict[str, Set[str]],
    positives: Set[Tuple[str, str]],
    cfg: NegativeSamplingConfig,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Build the labeled pair list used for model fitting.

    Negatives are drawn from candidates \\ positives, per S1 entity,
    respecting `negative_per_positive` where possible. If an S1 entity has
    no ground-truth positives, a small number of its candidates are still
    sampled as negatives (bounded) so the model also sees "confident
    singleton" examples during training.
    """
    rng = random.Random(cfg.seed)
    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []
    seen: Set[Tuple[str, str]] = set()

    positives_by_s1: Dict[str, Set[str]] = {}
    for s1_id, target_id in positives:
        positives_by_s1.setdefault(s1_id, set()).add(target_id)

    for s1_id, cand_set in candidates.items():
        pos_targets = positives_by_s1.get(s1_id, set())
        for t in pos_targets:
            key = (s1_id, t)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
                labels.append(1)

        negative_pool = [c for c in cand_set if c not in pos_targets]
        rng.shuffle(negative_pool)
        n_neg_needed = max(len(pos_targets) * cfg.negative_per_positive, 3 if not pos_targets else 0)
        chosen_negs = negative_pool[:n_neg_needed]
        for t in chosen_negs:
            key = (s1_id, t)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
                labels.append(0)

    return pairs, labels


@dataclass
class TrainedModel:
    name: str
    estimator: object
    scaler: StandardScaler
    feature_names: List[str]

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        X = feature_df[self.feature_names].values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.estimator.predict_proba(X)[:, 1]


def train_logistic_regression(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> Tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=cfg.logreg_C, max_iter=cfg.logreg_max_iter, random_state=cfg.random_state, class_weight="balanced")
    clf.fit(X_scaled, y)
    return clf, scaler


def train_gradient_boosted_trees(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> Tuple[HistGradientBoostingClassifier, None]:
    clf = HistGradientBoostingClassifier(
        max_depth=cfg.gbt_max_depth,
        max_iter=cfg.gbt_max_iter,
        learning_rate=cfg.gbt_learning_rate,
        l2_regularization=cfg.gbt_l2_regularization,
        random_state=cfg.random_state,
        class_weight="balanced",
    )
    clf.fit(X, y)
    return clf, None


def train_models(feature_df: pd.DataFrame, labels: List[int], cfg: ModelConfig) -> Dict[str, TrainedModel]:
    X = feature_df[FEATURE_NAMES].values
    y = np.array(labels)
    trained: Dict[str, TrainedModel] = {}

    if "logistic_regression" in cfg.models_to_try:
        clf, scaler = train_logistic_regression(X, y, cfg)
        trained["logistic_regression"] = TrainedModel("logistic_regression", clf, scaler, FEATURE_NAMES)

    if "gradient_boosted_trees" in cfg.models_to_try:
        clf, _ = train_gradient_boosted_trees(X, y, cfg)
        trained["gradient_boosted_trees"] = TrainedModel("gradient_boosted_trees", clf, None, FEATURE_NAMES)

    return trained
