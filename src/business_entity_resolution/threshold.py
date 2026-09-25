"""
threshold.py
============
Threshold sweep and match-decision policy (Requirements Analysis §12, §14).

We never use a fixed 0.5 cutoff. Instead we sweep many thresholds over the
validation set's scored candidate pairs, decide predicted match sets under
each, score them with the exact macro-F0.5 metric, and pick the threshold
that maximizes it -- deliberately biasing toward precision, since F0.5
weights precision twice as heavily as recall.

An optional, simple, explainable refinement (section 14) is supported: a
minimum score MARGIN between an S1 entity's best candidate and its
second-best candidate before the best candidate is accepted alongside
others. This is off by default in the sense that it is only ever applied
if it improves validation macro F0.5 over the plain global threshold; the
sweep tries both and keeps whichever is better.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .config import ThresholdConfig
from .evaluation import MacroF05Report, evaluate_macro_f05


def decide_matches_global_threshold(scored_pairs: pd.DataFrame, threshold: float) -> Dict[str, Set[str]]:
    """
    scored_pairs must have columns: source1_entity_id, target_entity_id, score
    Every s1 id that appears (even with zero accepted matches) gets an entry.
    """
    decisions: Dict[str, Set[str]] = {}
    all_s1 = scored_pairs["source1_entity_id"].unique()
    for s1_id in all_s1:
        decisions[s1_id] = set()
    accepted = scored_pairs[scored_pairs["score"] >= threshold]
    for s1_id, group in accepted.groupby("source1_entity_id"):
        decisions[s1_id] = set(group["target_entity_id"].tolist())
    return decisions


def decide_matches_with_margin(scored_pairs: pd.DataFrame, threshold: float, min_gap: float) -> Dict[str, Set[str]]:
    """
    Same as the global threshold rule, except: if an S1 entity has exactly
    one candidate above threshold and it beats the next-best candidate
    (regardless of that candidate's own threshold status) by less than
    min_gap, we withhold the match as too ambiguous to be confident about --
    a conservative refinement aimed specifically at reducing singleton
    false positives on near-tied candidates.
    """
    decisions: Dict[str, Set[str]] = {}
    all_s1 = scored_pairs["source1_entity_id"].unique()
    for s1_id in all_s1:
        decisions[s1_id] = set()

    for s1_id, group in scored_pairs.groupby("source1_entity_id"):
        sorted_group = group.sort_values("score", ascending=False)
        scores = sorted_group["score"].tolist()
        ids = sorted_group["target_entity_id"].tolist()
        accepted = [ids[i] for i in range(len(scores)) if scores[i] >= threshold]
        if len(accepted) == 1 and len(scores) >= 2:
            gap = scores[0] - scores[1]
            if gap < min_gap:
                accepted = []
        decisions[s1_id] = set(accepted)
    return decisions


@dataclass
class ThresholdSweepResult:
    best_threshold: float
    best_use_margin: bool
    best_report: MacroF05Report
    sweep_table: pd.DataFrame


def sweep_thresholds(
    scored_val_pairs: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    cfg: ThresholdConfig,
) -> ThresholdSweepResult:
    thresholds = np.linspace(cfg.min_threshold, cfg.max_threshold, cfg.thresholds_to_sweep)
    rows = []
    best = None

    variants = [("global", False)]
    if cfg.use_margin_rule:
        variants.append(("margin", True))

    for variant_name, use_margin in variants:
        for t in thresholds:
            if use_margin:
                decisions = decide_matches_with_margin(scored_val_pairs, t, cfg.margin_min_gap)
            else:
                decisions = decide_matches_global_threshold(scored_val_pairs, t)
            # ensure every ground-truth s1 id has a decision entry (default empty)
            for s1_id in ground_truth:
                decisions.setdefault(s1_id, set())
            report = evaluate_macro_f05(decisions, ground_truth)
            rows.append({
                "variant": variant_name, "threshold": t, "macro_f05": report.macro_f05,
                "mean_precision": report.mean_precision, "mean_recall": report.mean_recall,
                "n_zero": report.n_zero_matches_predicted, "n_one": report.n_one_match_predicted,
                "n_multi": report.n_multi_match_predicted,
                "singleton_fp": report.n_singleton_false_positives,
            })
            if best is None or report.macro_f05 > best[2].macro_f05:
                best = (t, use_margin, report)

    sweep_table = pd.DataFrame(rows)
    return ThresholdSweepResult(best_threshold=best[0], best_use_margin=best[1], best_report=best[2], sweep_table=sweep_table)


def apply_decision_policy(scored_pairs: pd.DataFrame, threshold: float, use_margin: bool, margin_min_gap: float) -> Dict[str, Set[str]]:
    if use_margin:
        return decide_matches_with_margin(scored_pairs, threshold, margin_min_gap)
    return decide_matches_global_threshold(scored_pairs, threshold)
