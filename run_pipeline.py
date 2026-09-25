#!/usr/bin/env python3
"""
run_pipeline.py
================
Single entry point. Runs end-to-end with zero arguments using the defaults
in config.py (reproducible: see PipelineConfig.random_seed / split_seed /
model.random_state), or accepts a config file / flag overrides.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config my_config.json
    python run_pipeline.py --data-dir dataset --output-dir output
    python run_pipeline.py --skip-validator      # skip the local validator run

After the pipeline finishes it automatically invokes
utils/validate_submission.py against the files it just wrote, and prints
its PASS/FAIL result.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from business_entity_resolution.config import PipelineConfig  # noqa: E402
from business_entity_resolution.pipeline import run_full_pipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    p.add_argument("--config", type=str, default=None, help="Path to a JSON/YAML PipelineConfig")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--models-dir", type=str, default=None)
    p.add_argument("--experiments-dir", type=str, default=None)
    p.add_argument("--skip-validator", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.models_dir:
        cfg.models_dir = args.models_dir
    if args.experiments_dir:
        cfg.experiments_dir = args.experiments_dir

    summary = run_full_pipeline(cfg)

    print("\n=== PIPELINE SUMMARY ===")
    for k, v in summary.items():
        if k == "output_issues":
            continue
        print(f"{k}: {v}")
    if summary.get("output_issues"):
        print(f"output_issues ({len(summary['output_issues'])}):")
        for issue in summary["output_issues"][:20]:
            print(f"  - {issue}")

    if not args.skip_validator:
        matching_path = os.path.join(cfg.output_dir, "matching_results.tsv")
        candidate_path = os.path.join(cfg.output_dir, "candidate_pairs.tsv")
        test_dir = os.path.join(cfg.data_dir, "test")
        validator_path = os.path.join(os.path.dirname(__file__), "utils", "validate_submission.py")
        print("\n=== RUNNING LOCAL VALIDATOR ===")
        result = subprocess.run(
            [sys.executable, validator_path,
             "--matching", matching_path,
             "--candidate", candidate_path,
             "--test-dir", test_dir],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print("VALIDATOR FAILED. See issues above.")
            sys.exit(result.returncode)
        else:
            print("VALIDATOR PASSED.")


if __name__ == "__main__":
    main()
