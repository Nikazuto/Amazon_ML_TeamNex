"""
pipeline.py
===========
End-to-end orchestration, matching the required architecture (problem
statement §3 / Requirements Analysis §12 dependency graph):

RAW DATA -> ingestion -> normalization -> leakage-safe split -> blocking
-> blocking eval -> features -> model training -> threshold calibration
-> singleton handling -> test inference -> output validation -> TSVs

Every stage reads its parameters from a single `PipelineConfig`
(config.py) so a run is fully reconstructible from its logged config
(experiment.py).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd

from .blocking import NormalizedCorpus, build_corpus, generate_candidates
from .config import COL_ENTITY_ID, GT_COL_MATCHES, GT_COL_S1, PipelineConfig, seed_everything
from .data_loader import DatasetBundle, load_dataset, parse_id_list
from .evaluation import (
    BlockingEvalReport,
    MacroF05Report,
    diagnose_false_negative_sources,
    evaluate_blocking,
    evaluate_macro_f05,
)
from .experiment import log_run, new_run_id
from .features import extract_features_for_pairs
from .inference import generate_and_score
from .normalization import NormalizedAddress, NormalizedName, normalize_address, normalize_business_name
from .output import write_outputs
from .threshold import ThresholdSweepResult, apply_decision_policy, sweep_thresholds
from .training import TrainedModel, build_positive_pairs, build_training_pairs, train_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_dataframe(df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[
    Dict[str, NormalizedName], Dict[str, NormalizedAddress], Dict[str, str]
]:
    names: Dict[str, NormalizedName] = {}
    addrs: Dict[str, NormalizedAddress] = {}
    countries: Dict[str, str] = {}
    from .config import COL_COUNTRY, COL_NAME, COL_ADDRESS

    for _, row in df.iterrows():
        eid = row[COL_ENTITY_ID]
        names[eid] = normalize_business_name(
            row.get(COL_NAME, ""),
            strip_suffix=cfg.normalization.strip_legal_suffixes,
            expand_abbrev=cfg.normalization.expand_abbreviations,
        )
        addrs[eid] = normalize_address(
            row.get(COL_ADDRESS, ""),
            expand_abbrev=cfg.normalization.expand_abbreviations,
        )
        countries[eid] = row.get(COL_COUNTRY, "") or ""
    return names, addrs, countries


def combine_targets(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    cols = list(s2.columns)
    return pd.concat([s2[cols], s3[cols]], ignore_index=True)


def build_ground_truth_dict(gt_df: pd.DataFrame, s1_ids: List[str]) -> Dict[str, Set[str]]:
    """Every id in s1_ids gets an entry, defaulting to empty set (true singleton)
    if it has no row (or an empty row) in the ground-truth file."""
    gt: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_ids}
    for _, row in gt_df.iterrows():
        s1_id = row[GT_COL_S1]
        if s1_id in gt:
            gt[s1_id] = set(parse_id_list(row[GT_COL_MATCHES]))
    return gt


@dataclass
class PipelineArtifacts:
    run_id: str
    cfg: PipelineConfig
    trained_models: Dict[str, TrainedModel]
    selected_model_name: str
    threshold_result: ThresholdSweepResult
    val_report: MacroF05Report
    blocking_report: BlockingEvalReport
    final_model: TrainedModel


def run_training_and_calibration(cfg: PipelineConfig) -> PipelineArtifacts:
    """Phases 1-3: ingestion -> normalization -> split -> blocking -> features
    -> model training -> threshold calibration, all on TRAIN data only,
    leakage-safe per FR-5.4."""
    seed_everything(cfg.random_seed)
    run_id = new_run_id()
    t0 = time.time()

    logger.info("Loading train dataset from %s", cfg.data_dir)
    bundle: DatasetBundle = load_dataset(cfg.data_dir, "train")
    if bundle.ground_truth is None:
        raise RuntimeError("train_ground_truth.tsv not found; cannot train/calibrate")

    target_df = combine_targets(bundle.s2, bundle.s3)

    logger.info("Normalizing names/addresses (S1=%d, targets=%d)", len(bundle.s1), len(target_df))
    s1_names, s1_addrs, s1_countries = normalize_dataframe(bundle.s1, cfg)
    t_names, t_addrs, t_countries = normalize_dataframe(target_df, cfg)

    s1_corpus_full = build_corpus(bundle.s1, s1_names, s1_addrs)
    target_corpus = build_corpus(target_df, t_names, t_addrs)

    # --- leakage-safe split at the S1 level (FR-5.4) ---
    from .validation_split import split_s1_entities

    split = split_s1_entities(bundle.s1, cfg.split)
    logger.info("Split: %d train S1 ids, %d val S1 ids", len(split.train_s1_ids), len(split.val_s1_ids))

    # --- blocking: run once over ALL train S1 ids (train+val) against the
    # train target corpus. This is NOT leakage: blocking uses only the
    # records' own name/address text, never ground-truth labels, and the
    # TF-IDF vectorizers fit only on target_corpus (the S2/S3 side), never
    # on validation labels (see blocking.py docstring). ---
    logger.info("Generating candidates (blocking)...")
    all_candidates = generate_candidates(s1_corpus_full, target_corpus, cfg.blocking)

    train_id_set = set(split.train_s1_ids)
    val_id_set = set(split.val_s1_ids)
    train_candidates = {k: v for k, v in all_candidates.items() if k in train_id_set}
    val_candidates = {k: v for k, v in all_candidates.items() if k in val_id_set}

    # --- ground truth dicts ---
    full_gt = build_ground_truth_dict(bundle.ground_truth, bundle.s1[COL_ENTITY_ID].tolist())
    val_gt = {k: v for k, v in full_gt.items() if k in val_id_set}

    # --- blocking evaluation on validation split only (FR-3.5) ---
    blocking_report = evaluate_blocking(val_candidates, val_gt, n_target_records=len(target_corpus.ids))
    logger.info(
        "Blocking recall=%.4f reduction_ratio=%.4f avg_cand/entity=%.1f zero_cand=%d",
        blocking_report.blocking_recall, blocking_report.reduction_ratio,
        blocking_report.avg_candidates_per_entity, blocking_report.n_entities_zero_candidates,
    )

    # --- training pair construction (train split only) ---
    train_positives = build_positive_pairs(bundle.ground_truth, train_id_set)
    train_pairs, train_labels = build_training_pairs(train_candidates, train_positives, cfg.negative_sampling)
    logger.info("Training pairs: %d (positives=%d)", len(train_pairs), sum(train_labels))

    train_feature_df = extract_features_for_pairs(
        train_pairs, s1_names, s1_addrs, s1_countries, t_names, t_addrs, t_countries
    )

    trained_models = train_models(train_feature_df, train_labels, cfg.model)

    # --- score validation candidates once per model, sweep thresholds ---
    val_pairs = [(s1_id, t) for s1_id, cset in val_candidates.items() for t in cset]
    val_feature_df = extract_features_for_pairs(
        val_pairs, s1_names, s1_addrs, s1_countries, t_names, t_addrs, t_countries
    )

    best_model_name = None
    best_sweep: ThresholdSweepResult = None
    for name, model in trained_models.items():
        if len(val_feature_df) == 0:
            scored = val_feature_df.copy()
            scored["score"] = pd.Series(dtype=float)
        else:
            scored = val_feature_df.copy()
            scored["score"] = model.predict_proba(val_feature_df)
        # ensure every val s1 id appears even with zero candidates
        if len(scored) == 0:
            scored = pd.DataFrame(columns=["source1_entity_id", "target_entity_id", "score"])
        sweep = sweep_thresholds(scored, val_gt, cfg.threshold)
        logger.info("[%s] best val macro F0.5 = %.4f (threshold=%.3f, margin=%s)",
                    name, sweep.best_report.macro_f05, sweep.best_threshold, sweep.best_use_margin)
        if best_sweep is None or sweep.best_report.macro_f05 > best_sweep.best_report.macro_f05:
            best_sweep = sweep
            best_model_name = name

    logger.info("Selected model: %s", best_model_name)

    # --- optional unseen-country proxy diagnostic (never used for tuning) ---
    if split.holdout_s1_ids:
        logger.info(
            "Holdout-country proxy ('%s'): %d S1 entities held out entirely; diagnostic only, not used for tuning.",
            split.holdout_country, len(split.holdout_s1_ids),
        )

    # --- diagnose false-negative source on validation (blocking vs threshold) ---
    val_decisions = apply_decision_policy(
        scored, best_sweep.best_threshold, best_sweep.best_use_margin, cfg.threshold.margin_min_gap
    )
    for s1_id in val_gt:
        val_decisions.setdefault(s1_id, set())
    diagnosis = diagnose_false_negative_sources(val_decisions, val_candidates, val_gt)
    logger.info(
        "False negatives on val: %d blocking-stage, %d threshold-stage",
        len(diagnosis["blocking_false_negatives"]), len(diagnosis["model_threshold_false_negatives"]),
    )

    # --- retrain the selected model on ALL train data (train+val ids) for
    # the strongest possible final model, now that model/threshold choice
    # is already locked in from the leakage-free split. ---
    logger.info("Retraining %s on full training data for final deployment...", best_model_name)
    full_positives = build_positive_pairs(bundle.ground_truth, set(bundle.s1[COL_ENTITY_ID]))
    full_pairs, full_labels = build_training_pairs(all_candidates, full_positives, cfg.negative_sampling)
    full_feature_df = extract_features_for_pairs(
        full_pairs, s1_names, s1_addrs, s1_countries, t_names, t_addrs, t_countries
    )
    full_cfg = cfg.model
    from dataclasses import replace
    single_model_cfg = replace(full_cfg, models_to_try=[best_model_name])
    final_models = train_models(full_feature_df, full_labels, single_model_cfg)
    final_model = final_models[best_model_name]

    metrics = {
        "selected_model": best_model_name,
        "best_threshold": best_sweep.best_threshold,
        "best_use_margin": best_sweep.best_use_margin,
        "val_macro_f05": best_sweep.best_report.macro_f05,
        "val_mean_precision": best_sweep.best_report.mean_precision,
        "val_mean_recall": best_sweep.best_report.mean_recall,
        "n_zero_matches_predicted": best_sweep.best_report.n_zero_matches_predicted,
        "n_one_match_predicted": best_sweep.best_report.n_one_match_predicted,
        "n_multi_match_predicted": best_sweep.best_report.n_multi_match_predicted,
        "n_singleton_false_positives": best_sweep.best_report.n_singleton_false_positives,
        "n_true_singletons": best_sweep.best_report.n_true_singletons,
        "n_true_singletons_correct": best_sweep.best_report.n_true_singletons_correct,
        "blocking_recall": blocking_report.blocking_recall,
        "reduction_ratio": blocking_report.reduction_ratio,
        "avg_candidates_per_entity": blocking_report.avg_candidates_per_entity,
        "n_entities_zero_candidates": blocking_report.n_entities_zero_candidates,
        "n_blocking_false_negatives": len(diagnosis["blocking_false_negatives"]),
        "n_threshold_false_negatives": len(diagnosis["model_threshold_false_negatives"]),
        "holdout_country_proxy_n": len(split.holdout_s1_ids) if split.holdout_s1_ids else 0,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    log_run(cfg.experiments_dir, run_id, cfg, metrics)

    return PipelineArtifacts(
        run_id=run_id,
        cfg=cfg,
        trained_models=trained_models,
        selected_model_name=best_model_name,
        threshold_result=best_sweep,
        val_report=best_sweep.best_report,
        blocking_report=blocking_report,
        final_model=final_model,
    )


def run_test_inference(cfg: PipelineConfig, artifacts: PipelineArtifacts) -> Dict[str, object]:
    """Phase: generate test candidates, score with the final model, apply
    the calibrated decision policy, write matching_results.tsv and
    candidate_pairs.tsv, and check output invariants (FR-6.1-6.4, FR-17)."""
    logger.info("Loading test dataset from %s", cfg.data_dir)
    bundle = load_dataset(cfg.data_dir, "test")
    target_df = combine_targets(bundle.s2, bundle.s3)

    s1_names, s1_addrs, s1_countries = normalize_dataframe(bundle.s1, cfg)
    t_names, t_addrs, t_countries = normalize_dataframe(target_df, cfg)

    s1_corpus = build_corpus(bundle.s1, s1_names, s1_addrs)
    target_corpus = build_corpus(target_df, t_names, t_addrs)

    result = generate_and_score(
        s1_corpus, target_corpus,
        s1_names, s1_addrs, s1_countries,
        t_names, t_addrs, t_countries,
        cfg.blocking, artifacts.final_model,
    )

    matches = apply_decision_policy(
        result.scored_pairs,
        artifacts.threshold_result.best_threshold,
        artifacts.threshold_result.best_use_margin,
        cfg.threshold.margin_min_gap,
    )
    for s1_id in s1_corpus.ids:
        matches.setdefault(s1_id, set())

    valid_target_ids = set(target_corpus.ids)
    report = write_outputs(
        cfg.output_dir, s1_corpus.ids, matches, result.candidates, valid_target_ids
    )
    if not report.ok:
        logger.warning("Output invariant issues found (%d):", len(report.issues))
        for issue in report.issues[:20]:
            logger.warning("  - %s", issue)
    else:
        logger.info("Output invariants OK.")

    n_zero = sum(1 for v in matches.values() if len(v) == 0)
    n_one = sum(1 for v in matches.values() if len(v) == 1)
    n_multi = sum(1 for v in matches.values() if len(v) > 1)
    logger.info("Test predictions: %d zero-match, %d one-match, %d multi-match (of %d entities)",
                n_zero, n_one, n_multi, len(s1_corpus.ids))

    return {
        "n_test_entities": len(s1_corpus.ids),
        "n_zero_matches": n_zero,
        "n_one_match": n_one,
        "n_multi_match": n_multi,
        "output_report_ok": report.ok,
        "output_issues": report.issues,
    }


def run_full_pipeline(cfg: PipelineConfig) -> Dict[str, object]:
    artifacts = run_training_and_calibration(cfg)
    test_summary = run_test_inference(cfg, artifacts)
    return {
        "run_id": artifacts.run_id,
        "selected_model": artifacts.selected_model_name,
        "val_macro_f05": artifacts.val_report.macro_f05,
        "blocking_recall": artifacts.blocking_report.blocking_recall,
        "reduction_ratio": artifacts.blocking_report.reduction_ratio,
        **test_summary,
    }
