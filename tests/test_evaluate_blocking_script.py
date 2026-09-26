"""
test_evaluate_blocking_script.py
=================================
Smoke tests for the standalone evaluate_blocking.py CLI utility (Person B
deliverable: "add a blocking evaluation utility"). These are deliberately
light -- the metric math itself is already covered by test_evaluation.py
and the blocking behavior by test_blocking.py; this file only checks that
the script's own glue code (benchmark runner + synthetic dataset) behaves.
"""
import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from business_entity_resolution.config import COL_ENTITY_ID, BlockingConfig  # noqa: E402

from evaluate_blocking import build_synthetic_dataset, run_blocking_benchmark  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


class TestSyntheticDataset(unittest.TestCase):
    def test_ground_truth_references_only_ids_present_in_sources(self):
        s1_df, s2_df, s3_df, ground_truth = build_synthetic_dataset()
        s1_ids = set(s1_df[COL_ENTITY_ID])
        target_ids = set(s2_df[COL_ENTITY_ID]) | set(s3_df[COL_ENTITY_ID])
        self.assertEqual(set(ground_truth.keys()), s1_ids)
        for matches in ground_truth.values():
            self.assertTrue(matches.issubset(target_ids))

    def test_includes_at_least_one_true_singleton(self):
        _s1, _s2, _s3, ground_truth = build_synthetic_dataset()
        self.assertTrue(any(len(v) == 0 for v in ground_truth.values()))


class TestRunBlockingBenchmark(unittest.TestCase):
    def test_reports_expected_fields_and_runtime_is_nonnegative(self):
        s1_df, s2_df, s3_df, ground_truth = build_synthetic_dataset()
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)
        result, candidates = run_blocking_benchmark(s1_df, target_df, ground_truth, BlockingConfig())

        for key in ["blocking_recall", "reduction_ratio", "avg_candidates_per_entity",
                    "total_candidates_generated", "runtime_seconds", "n_true_matches_missed"]:
            self.assertIn(key, result)
        self.assertGreaterEqual(result["runtime_seconds"], 0.0)
        self.assertEqual(set(candidates.keys()), set(s1_df[COL_ENTITY_ID]))

    def test_perfect_default_config_recalls_every_synthetic_true_match(self):
        # With every strategy enabled and generous defaults, the hand-crafted
        # synthetic true matches (abbreviation, punctuation, typo, word-order
        # cases) should all be recoverable -- this is a regression guard on
        # the synthetic fixture itself, not a claim about real-data recall.
        s1_df, s2_df, s3_df, ground_truth = build_synthetic_dataset()
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)
        result, _candidates = run_blocking_benchmark(s1_df, target_df, ground_truth, BlockingConfig())
        self.assertEqual(result["blocking_recall"], 1.0)
        self.assertEqual(result["n_true_matches_missed"], 0)


class TestCliSmokeTest(unittest.TestCase):
    def test_synthetic_cli_run_exits_zero(self):
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "evaluate_blocking.py"), "--synthetic"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("blocking_recall", result.stdout)

    def test_missing_real_data_exits_nonzero_with_clear_message(self):
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "evaluate_blocking.py"),
             "--data-dir", "/tmp/definitely_does_not_exist_bER"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--synthetic", result.stdout)


if __name__ == "__main__":
    unittest.main()
