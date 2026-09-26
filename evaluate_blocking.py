#!/usr/bin/env python3
"""
evaluate_blocking.py
=====================
Standalone blocking-only benchmark and error-analysis tool (Person B
deliverable). Mirrors run_pipeline.py's CLI conventions but runs ONLY the
ingestion -> normalization -> blocking -> blocking-evaluation slice of the
pipeline -- no model training, no feature extraction, no scoring -- so
blocking quality can be measured and iterated on in isolation, quickly,
without needing a trained model or the full ~100k-record dataset.

It reuses the existing, already-tested building blocks rather than
reimplementing any metric:
  * data_loader.load_dataset          -- TSV ingestion
  * pipeline.normalize_dataframe      -- name/address normalization
  * blocking.generate_candidates      -- candidate generation (the module
                                          under evaluation)
  * evaluation.evaluate_blocking      -- blocking recall / reduction ratio /
                                          candidate-count metrics (FR-3.5)
  * evaluation.evaluate_blocking(...).missed_pairs
                                       -- pure blocking-stage error analysis:
                                          any ground-truth match absent from
                                          the candidate set is, by
                                          definition, a blocking miss (no
                                          model/threshold is involved yet at
                                          this stage of the pipeline).

What this script ADDS on top of the existing evaluation.py (rather than
duplicating it): wall-clock runtime measurement for candidate generation,
a synthetic smoke-test mode for developing without real data, and a
human-readable + JSON report.

Usage
-----
Against the real challenge dataset (needs a leakage-safe validation split,
since blocking is scored against held-out ground truth per FR-5.4/NFR-9):

    python evaluate_blocking.py --data-dir dataset --validation-fraction 0.2

Quick smoke test with a small built-in synthetic dataset (no files needed;
this is what "test blocking without the full 100k-record pipeline" means
in practice for a fast iteration loop):

    python evaluate_blocking.py --synthetic

Override any BlockingConfig field and write a JSON report:

    python evaluate_blocking.py --synthetic --config my_blocking_config.json \\
        --output experiments/blocking_baseline_run.json

Sample a subset of S1 entities for a faster dev-loop pass over real data:

    python evaluate_blocking.py --data-dir dataset --sample-size 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

from business_entity_resolution.blocking import build_corpus, generate_candidates  # noqa: E402
from business_entity_resolution.config import (  # noqa: E402
    COL_ENTITY_ID,
    BlockingConfig,
    PipelineConfig,
    seed_everything,
)
from business_entity_resolution.data_loader import load_dataset  # noqa: E402
from business_entity_resolution.evaluation import BlockingEvalReport, evaluate_blocking  # noqa: E402
from business_entity_resolution.normalization import normalize_address, normalize_business_name  # noqa: E402
from business_entity_resolution.pipeline import build_ground_truth_dict, combine_targets  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic smoke-test dataset (no files required)
# ---------------------------------------------------------------------------

def build_synthetic_dataset():
    """A small, hand-crafted dataset exercising every noise pattern called
    out in the challenge statement, so `--synthetic` gives a meaningful
    (if not statistically representative) blocking-recall reading without
    needing the real ~100k-record files."""
    s1_rows = [
        {"entity_id": "S1-001", "business_name": "Acme Corporation", "business_address": "1 Main Street, Springfield", "country": "US"},
        {"entity_id": "S1-002", "business_name": "Sundar Traders Pvt Ltd", "business_address": "12 MG Road, Pune", "country": "India"},
        {"entity_id": "S1-003", "business_name": "Riverside Bakery", "business_address": "42 Elm Street, Springfield", "country": "US"},
        {"entity_id": "S1-004", "business_name": "Totally Singleton LLC", "business_address": "1 Nowhere Ave", "country": "US"},
        {"entity_id": "S1-005", "business_name": "Le Petit Cafe", "business_address": "5 Rue de Paris, Paris", "country": "France"},  # unseen-country case
    ]
    s2_rows = [
        {"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "1 Main St, Springfield", "country": "US"},  # true match for S1-001 (abbrev + punctuation)
        {"entity_id": "S2-002", "business_name": "Sundar Traders Private Limited", "business_address": "12 M.G. Road, Pune", "country": "India"},  # true match for S1-002
        {"entity_id": "S2-003", "business_name": "Unrelated Business Two", "business_address": "77 Other Rd", "country": "US"},
    ]
    s3_rows = [
        {"entity_id": "S3-001", "business_name": "Riverside Bakry", "business_address": "42 Elm St, Springfeld", "country": "US"},  # true match for S1-003 (typo)
        {"entity_id": "S3-002", "business_name": "Cafe Le Petit", "business_address": "5 Rue de Paris", "country": "France"},  # true match for S1-005 (word order)
        {"entity_id": "S3-003", "business_name": "Unrelated Business Three", "business_address": "88 Elsewhere Blvd", "country": "US"},
    ]
    ground_truth = {
        "S1-001": {"S2-001"},
        "S1-002": {"S2-002"},
        "S1-003": {"S3-001"},
        "S1-004": set(),          # true singleton
        "S1-005": {"S3-002"},
    }
    s1_df = pd.DataFrame(s1_rows)
    s2_df = pd.DataFrame(s2_rows)
    s3_df = pd.DataFrame(s3_rows)
    return s1_df, s2_df, s3_df, ground_truth


# ---------------------------------------------------------------------------
# Core benchmark routine
# ---------------------------------------------------------------------------

def normalize_frame(df: pd.DataFrame):
    names, addrs = {}, {}
    for _, row in df.iterrows():
        eid = row[COL_ENTITY_ID]
        names[eid] = normalize_business_name(row.get("business_name", ""))
        addrs[eid] = normalize_address(row.get("business_address", ""))
    return names, addrs


def run_blocking_benchmark(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    blocking_cfg: BlockingConfig,
) -> Dict[str, object]:
    """Runs generate_candidates once, times it, and evaluates the result
    against ground truth using the existing evaluation.evaluate_blocking.
    Returns a plain dict combining timing + the BlockingEvalReport fields +
    a small human-readable error-analysis sample, ready to log or print.
    """
    s1_names, s1_addrs = normalize_frame(s1_df)
    t_names, t_addrs = normalize_frame(target_df)

    s1_corpus = build_corpus(s1_df, s1_names, s1_addrs)
    target_corpus = build_corpus(target_df, t_names, t_addrs)

    t0 = time.perf_counter()
    candidates = generate_candidates(s1_corpus, target_corpus, blocking_cfg)
    elapsed_seconds = time.perf_counter() - t0

    report: BlockingEvalReport = evaluate_blocking(
        candidates, ground_truth, n_target_records=len(target_corpus.ids)
    )

    n_s1 = len(s1_corpus.ids)
    result = {
        "n_s1_entities": n_s1,
        "n_target_records": len(target_corpus.ids),
        "runtime_seconds": round(elapsed_seconds, 4),
        "runtime_seconds_per_1k_s1_entities": round(elapsed_seconds / n_s1 * 1000, 4) if n_s1 else 0.0,
        "blocking_recall": report.blocking_recall,
        "reduction_ratio": report.reduction_ratio,
        "total_candidates_generated": report.total_candidates_generated,
        "avg_candidates_per_entity": report.avg_candidates_per_entity,
        "median_candidates_per_entity": report.median_candidates_per_entity,
        "max_candidates_per_entity": report.max_candidates_per_entity,
        "n_entities_zero_candidates": report.n_entities_zero_candidates,
        "n_true_matches": report.n_true_matches,
        "n_true_matches_recovered": report.n_true_matches_recovered,
        "n_true_matches_missed": report.n_true_matches_missed,
        # Pure blocking-stage error analysis: every entry here is a
        # ground-truth match that never made it into the candidate set,
        # i.e. unrecoverable by any downstream model/threshold choice.
        "missed_pairs_sample": report.missed_pairs[:50],
        "blocking_config": asdict(blocking_cfg),
    }
    return result, candidates


def print_report(result: Dict[str, object]) -> None:
    print("\n=== BLOCKING EVALUATION REPORT ===")
    for key in [
        "n_s1_entities", "n_target_records", "runtime_seconds",
        "runtime_seconds_per_1k_s1_entities", "blocking_recall", "reduction_ratio",
        "total_candidates_generated", "avg_candidates_per_entity",
        "median_candidates_per_entity", "max_candidates_per_entity",
        "n_entities_zero_candidates", "n_true_matches", "n_true_matches_recovered",
        "n_true_matches_missed",
    ]:
        print(f"{key}: {result[key]}")

    if result["missed_pairs_sample"]:
        print(f"\nSample of blocking-stage misses (true match never entered the candidate set), "
              f"up to 50 of {result['n_true_matches_missed']} total:")
        for s1_id, target_id in result["missed_pairs_sample"][:20]:
            print(f"  - {s1_id} -> {target_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Standalone blocking evaluation / error-analysis utility")
    p.add_argument("--synthetic", action="store_true",
                    help="Run against a small built-in synthetic dataset instead of loading files "
                         "(fast smoke test; no dataset/ files required).")
    p.add_argument("--data-dir", type=str, default="dataset",
                    help="Dataset root (expects <data-dir>/train/train_*.tsv). Ignored with --synthetic.")
    p.add_argument("--config", type=str, default=None,
                    help="Path to a JSON/YAML PipelineConfig or bare BlockingConfig-shaped JSON "
                         "(only the 'blocking' section is used, if present).")
    p.add_argument("--sample-size", type=int, default=None,
                    help="Randomly sample at most N Source-1 entities before blocking, for a "
                         "faster dev-loop pass over a large real dataset.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --sample-size sampling.")
    p.add_argument("--output", type=str, default=None, help="Optional path to write the JSON report.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def load_blocking_config(path: str) -> BlockingConfig:
    with open(path) as f:
        d = json.load(f)
    if "blocking" in d:
        # A full PipelineConfig-shaped file: reuse PipelineConfig's own
        # merge logic so tuple fields (ngram_char_range, etc.) are handled
        # consistently with the rest of the pipeline.
        return PipelineConfig.from_dict(d).blocking
    return BlockingConfig(**d)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    seed_everything(args.seed)

    blocking_cfg = load_blocking_config(args.config) if args.config else BlockingConfig()

    if args.synthetic:
        logger.info("Using built-in synthetic dataset (--synthetic).")
        s1_df, s2_df, s3_df, ground_truth = build_synthetic_dataset()
        target_df = combine_targets(s2_df, s3_df)
    else:
        logger.info("Loading train dataset from %s", args.data_dir)
        bundle = load_dataset(args.data_dir, "train")
        if bundle.ground_truth is None:
            print(f"No train_ground_truth.tsv found under {args.data_dir}/train/. "
                  f"Blocking quality can't be scored without labels -- "
                  f"use --synthetic for a smoke test, or point --data-dir at a "
                  f"populated dataset/ directory.")
            sys.exit(1)
        s1_df = bundle.s1
        target_df = combine_targets(bundle.s2, bundle.s3)
        ground_truth = build_ground_truth_dict(bundle.ground_truth, s1_df[COL_ENTITY_ID].tolist())

        if args.sample_size and len(s1_df) > args.sample_size:
            s1_df = s1_df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
            ground_truth = {eid: v for eid, v in ground_truth.items() if eid in set(s1_df[COL_ENTITY_ID])}
            logger.info("Sampled down to %d S1 entities (--sample-size).", len(s1_df))

    result, _candidates = run_blocking_benchmark(s1_df, target_df, ground_truth, blocking_cfg)
    print_report(result)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
