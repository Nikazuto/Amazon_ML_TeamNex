"""
experiment.py
=============
Per-run experiment logging (Requirements Analysis §16 / SRS FR-8.3, AC-12).

Every pipeline run appends one row to experiments/results.csv (flat,
spreadsheet-friendly, easy to diff run-over-run) AND writes a full
experiments/run_<run_id>.json (the exact PipelineConfig plus every metric
computed) so that "which settings produced this submission" is always
reconstructable, not just the headline F0.5 number.
"""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from .config import PipelineConfig


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def log_run(experiments_dir: str, run_id: str, cfg: PipelineConfig, metrics: Dict[str, Any]) -> None:
    os.makedirs(experiments_dir, exist_ok=True)

    # --- full JSON record ---
    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg.to_dict(),
        "metrics": _to_jsonable(metrics),
    }
    json_path = os.path.join(experiments_dir, f"run_{run_id}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    # --- flat CSV summary row ---
    flat_row = {
        "run_id": run_id,
        "timestamp": record["timestamp"],
        "validation_fraction": cfg.split.validation_fraction,
        "split_seed": cfg.split.split_seed,
        "holdout_country_proxy": cfg.split.holdout_country_proxy,
        "strip_legal_suffixes": cfg.normalization.strip_legal_suffixes,
        "expand_abbreviations": cfg.normalization.expand_abbreviations,
        "blocking_methods": "exact+token+ngram+address+fuzzy",
        "max_candidates_per_entity": cfg.blocking.max_candidates_per_entity,
        "negative_per_positive": cfg.negative_sampling.negative_per_positive,
        "models_to_try": ",".join(cfg.model.models_to_try),
        "selected_model": metrics.get("selected_model"),
        "threshold": metrics.get("best_threshold"),
        "use_margin_rule": metrics.get("best_use_margin"),
        "val_macro_f05": metrics.get("val_macro_f05"),
        "val_mean_precision": metrics.get("val_mean_precision"),
        "val_mean_recall": metrics.get("val_mean_recall"),
        "blocking_recall": metrics.get("blocking_recall"),
        "reduction_ratio": metrics.get("reduction_ratio"),
        "avg_candidates_per_entity": metrics.get("avg_candidates_per_entity"),
        "n_entities_zero_candidates": metrics.get("n_entities_zero_candidates"),
        "n_zero_matches_predicted": metrics.get("n_zero_matches_predicted"),
        "n_one_match_predicted": metrics.get("n_one_match_predicted"),
        "n_multi_match_predicted": metrics.get("n_multi_match_predicted"),
        "n_singleton_false_positives": metrics.get("n_singleton_false_positives"),
    }

    csv_path = os.path.join(experiments_dir, "results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_row)

    print(f"[experiment] logged run {run_id} -> {json_path}")
