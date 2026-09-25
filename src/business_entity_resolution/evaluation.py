"""
evaluation.py
=============
The exact challenge metric (Requirements Analysis §8, SRS §8) plus blocking
quality metrics (§6 / FR-3.5).

Macro F0.5
----------
F0.5 is computed PER S1 ENTITY, then macro-averaged. This is deliberately
NOT a global/micro F0.5 over all pairs -- the challenge statement and the
Requirements Analysis are explicit that per-entity computation is required
because it is what makes singletons first-class, equally-weighted outcomes.

Per-entity convention (documented explicitly, since neither source document
gives the raw 0/0 formula):
  * true empty & predicted empty  -> F0.5 = 1.0  (singleton correctly kept empty)
  * true empty & predicted non-empty -> F0.5 = 0.0  (singleton false positive)
  * true non-empty & predicted empty -> F0.5 = 0.0  (precision undefined / 0,
    recall = 0, so the formula's numerator is 0)
  * true non-empty & predicted non-empty -> standard formula on precision/recall
This matches the two anchor cases the challenge statement states explicitly
(singleton correct -> 1.0, singleton false positive -> 0.0) and extends them
consistently to the non-singleton cases.

Blocking recall / reduction ratio
----------------------------------
Kept in this module (rather than blocking.py) because it is fundamentally an
*evaluation* concern that must be computed against held-out ground truth,
never used to tune blocking parameters (that would be leakage, per FR-5.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

BETA = 0.5
BETA2 = BETA * BETA


def f05_from_counts(true_set: Set[str], pred_set: Set[str]) -> float:
    """Per-entity F0.5 under the convention documented above."""
    if not true_set and not pred_set:
        return 1.0
    if not true_set and pred_set:
        return 0.0
    if true_set and not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    denom = BETA2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + BETA2) * precision * recall / denom


def _precision_recall_for_report(true_set: Set[str], pred_set: Set[str]) -> "tuple[float, float]":
    """Diagnostic precision/recall per entity (same singleton convention as f05)."""
    if not true_set and not pred_set:
        return 1.0, 1.0
    if not true_set and pred_set:
        return 0.0, 0.0
    if true_set and not pred_set:
        return 0.0, 0.0
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    return precision, recall


@dataclass
class MacroF05Report:
    macro_f05: float
    mean_precision: float
    mean_recall: float
    n_entities: int
    n_zero_matches_predicted: int
    n_one_match_predicted: int
    n_multi_match_predicted: int
    n_true_singletons: int
    n_true_singletons_correct: int
    n_singleton_false_positives: int
    per_entity_f05: Dict[str, float] = field(default_factory=dict)


def evaluate_macro_f05(predicted: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> MacroF05Report:
    """
    predicted / ground_truth: s1_entity_id -> set of matched entity ids.
    Every s1 id in ground_truth must have an entry in predicted (missing ids
    are treated as an empty predicted set, i.e. "no match predicted").
    """
    f05_scores: Dict[str, float] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    n_zero = n_one = n_multi = 0
    n_true_singletons = 0
    n_true_singletons_correct = 0
    n_singleton_fp = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predicted.get(s1_id, set())
        f05_scores[s1_id] = f05_from_counts(true_set, pred_set)
        p, r = _precision_recall_for_report(true_set, pred_set)
        precisions.append(p)
        recalls.append(r)

        if len(pred_set) == 0:
            n_zero += 1
        elif len(pred_set) == 1:
            n_one += 1
        else:
            n_multi += 1

        if not true_set:
            n_true_singletons += 1
            if not pred_set:
                n_true_singletons_correct += 1
            else:
                n_singleton_fp += 1

    n = len(ground_truth)
    macro_f05 = sum(f05_scores.values()) / n if n else 0.0
    mean_precision = sum(precisions) / n if n else 0.0
    mean_recall = sum(recalls) / n if n else 0.0

    return MacroF05Report(
        macro_f05=macro_f05,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        n_entities=n,
        n_zero_matches_predicted=n_zero,
        n_one_match_predicted=n_one,
        n_multi_match_predicted=n_multi,
        n_true_singletons=n_true_singletons,
        n_true_singletons_correct=n_true_singletons_correct,
        n_singleton_false_positives=n_singleton_fp,
        per_entity_f05=f05_scores,
    )


# ---------------------------------------------------------------------------
# Blocking quality metrics (FR-3.5)
# ---------------------------------------------------------------------------

@dataclass
class BlockingEvalReport:
    blocking_recall: float                 # fraction of ground-truth matches present in candidates
    n_true_matches: int
    n_true_matches_recovered: int
    n_true_matches_missed: int
    missed_pairs: List["tuple[str, str]"]  # (s1_id, target_id) ground-truth matches NOT in candidates
    reduction_ratio: float                 # 1 - (candidates generated / full cross join size)
    full_cross_join_size: int
    total_candidates_generated: int
    avg_candidates_per_entity: float
    median_candidates_per_entity: float
    max_candidates_per_entity: int
    n_entities_zero_candidates: int
    candidate_count_distribution: Dict[str, int]  # entity_id -> candidate count


def evaluate_blocking(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    n_target_records: int,
) -> BlockingEvalReport:
    """
    candidates: s1_entity_id -> set of candidate target ids (the FINAL
        candidate set, immediately before model scoring -- see FR-3.3).
    ground_truth: s1_entity_id -> set of true matching target ids.
    n_target_records: total number of S2+S3 records the S1 side was blocked
        against (used for the full-cross-join reduction-ratio denominator).
    """
    n_true = 0
    n_recovered = 0
    missed: List["tuple[str, str]"] = []

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        for t in true_set:
            n_true += 1
            if t in cand_set:
                n_recovered += 1
            else:
                missed.append((s1_id, t))

    blocking_recall = n_recovered / n_true if n_true else 1.0

    counts = {eid: len(c) for eid, c in candidates.items()}
    total_candidates = sum(counts.values())
    n_entities = len(candidates)
    avg_candidates = total_candidates / n_entities if n_entities else 0.0
    sorted_counts = sorted(counts.values())
    median_candidates = (
        sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0.0
    )
    max_candidates = max(counts.values()) if counts else 0
    n_zero = sum(1 for c in counts.values() if c == 0)

    full_cross_join_size = n_entities * n_target_records
    reduction_ratio = (
        1.0 - (total_candidates / full_cross_join_size) if full_cross_join_size else 0.0
    )

    return BlockingEvalReport(
        blocking_recall=blocking_recall,
        n_true_matches=n_true,
        n_true_matches_recovered=n_recovered,
        n_true_matches_missed=n_true - n_recovered,
        missed_pairs=missed,
        reduction_ratio=reduction_ratio,
        full_cross_join_size=full_cross_join_size,
        total_candidates_generated=total_candidates,
        avg_candidates_per_entity=avg_candidates,
        median_candidates_per_entity=median_candidates,
        max_candidates_per_entity=max_candidates,
        n_entities_zero_candidates=n_zero,
        candidate_count_distribution=counts,
    )


def diagnose_false_negative_sources(
    predicted: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, List["tuple[str, str]"]]:
    """
    Splits every missed ground-truth pair into exactly one of two buckets:
      - "blocking_false_negatives": the true match never appeared in the
        candidate set at all -- a blocking-stage miss, unrecoverable by any
        downstream threshold choice.
      - "model_threshold_false_negatives": the true match WAS a candidate
        (so features/score existed for it) but was not present in the final
        predicted match set -- a scoring/threshold-stage miss.
    """
    blocking_fn: List["tuple[str, str]"] = []
    threshold_fn: List["tuple[str, str]"] = []

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        pred_set = predicted.get(s1_id, set())
        for t in true_set:
            if t in pred_set:
                continue
            if t in cand_set:
                threshold_fn.append((s1_id, t))
            else:
                blocking_fn.append((s1_id, t))

    return {
        "blocking_false_negatives": blocking_fn,
        "model_threshold_false_negatives": threshold_fn,
    }
