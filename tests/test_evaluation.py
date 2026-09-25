import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.evaluation import (
    f05_from_counts,
    evaluate_macro_f05,
    evaluate_blocking,
    diagnose_false_negative_sources,
)


class TestF05Formula(unittest.TestCase):
    def test_true_singleton_correctly_predicted_scores_1(self):
        self.assertEqual(f05_from_counts(set(), set()), 1.0)

    def test_true_singleton_false_positive_scores_0(self):
        self.assertEqual(f05_from_counts(set(), {"S2-001"}), 0.0)

    def test_true_nonempty_predicted_empty_scores_0(self):
        self.assertEqual(f05_from_counts({"S2-001"}, set()), 0.0)

    def test_perfect_match_scores_1(self):
        self.assertEqual(f05_from_counts({"S2-001", "S3-001"}, {"S2-001", "S3-001"}), 1.0)

    def test_worked_example_from_problem_statement(self):
        # predicted [S2-047, S2-193, S3-812], truth [S2-047, S3-812]
        # precision = 2/3, recall = 1.0 -> F0.5 = 0.714 (rounded)
        true_set = {"S2-047", "S3-812"}
        pred_set = {"S2-047", "S2-193", "S3-812"}
        score = f05_from_counts(true_set, pred_set)
        self.assertAlmostEqual(score, 0.7142857142857143, places=6)

    def test_precision_weighted_more_than_recall(self):
        # Same number of errors, but a false positive should hurt more than a false negative
        # under F0.5's precision-heavy weighting.
        true_set = {"S2-001", "S2-002"}
        fp_case = f05_from_counts(true_set, {"S2-001", "S2-002", "S2-003"})  # 1 extra FP
        fn_case = f05_from_counts(true_set, {"S2-001"})  # 1 missing (FN)
        self.assertLess(fn_case, fp_case)


class TestMacroF05(unittest.TestCase):
    def test_macro_average_weights_every_entity_equally(self):
        # One entity with many true matches should NOT dominate a singleton entity.
        gt = {
            "S1-001": {"S2-001", "S2-002", "S2-003", "S2-004"},  # multi-match, perfectly predicted
            "S1-002": set(),  # singleton, false positive
        }
        pred = {
            "S1-001": {"S2-001", "S2-002", "S2-003", "S2-004"},
            "S1-002": {"S2-999"},
        }
        report = evaluate_macro_f05(pred, gt)
        self.assertAlmostEqual(report.macro_f05, 0.5, places=6)  # (1.0 + 0.0) / 2

    def test_missing_prediction_treated_as_empty(self):
        gt = {"S1-001": {"S2-001"}}
        pred = {}  # S1-001 has no entry at all
        report = evaluate_macro_f05(pred, gt)
        self.assertEqual(report.macro_f05, 0.0)

    def test_singleton_stats_tracked_separately(self):
        gt = {"S1-001": set(), "S1-002": {"S2-001"}}
        pred = {"S1-001": set(), "S1-002": {"S2-001"}}
        report = evaluate_macro_f05(pred, gt)
        self.assertEqual(report.n_true_singletons, 1)
        self.assertEqual(report.n_true_singletons_correct, 1)
        self.assertEqual(report.n_singleton_false_positives, 0)


class TestBlockingEvaluation(unittest.TestCase):
    def test_blocking_recall_full_when_all_matches_covered(self):
        candidates = {"S1-001": {"S2-001", "S2-999"}}
        gt = {"S1-001": {"S2-001"}}
        report = evaluate_blocking(candidates, gt, n_target_records=10)
        self.assertEqual(report.blocking_recall, 1.0)
        self.assertEqual(report.n_true_matches_missed, 0)

    def test_blocking_recall_partial_when_match_missed(self):
        candidates = {"S1-001": {"S2-999"}}
        gt = {"S1-001": {"S2-001"}}
        report = evaluate_blocking(candidates, gt, n_target_records=10)
        self.assertEqual(report.blocking_recall, 0.0)
        self.assertEqual(len(report.missed_pairs), 1)

    def test_reduction_ratio_computation(self):
        candidates = {"S1-001": {"S2-001"}, "S1-002": {"S2-001", "S2-002"}}
        report = evaluate_blocking(candidates, {"S1-001": set(), "S1-002": set()}, n_target_records=10)
        # full cross join = 2 entities * 10 targets = 20; total candidates = 3
        self.assertAlmostEqual(report.reduction_ratio, 1 - 3 / 20, places=6)

    def test_false_negative_diagnosis_splits_blocking_vs_threshold(self):
        candidates = {"S1-001": {"S2-001", "S2-002"}}  # S2-002 is a candidate but not predicted
        predicted = {"S1-001": {"S2-001"}}
        gt = {"S1-001": {"S2-001", "S2-002", "S2-003"}}  # S2-003 was never even a candidate
        diagnosis = diagnose_false_negative_sources(predicted, candidates, gt)
        self.assertEqual(diagnosis["blocking_false_negatives"], [("S1-001", "S2-003")])
        self.assertEqual(diagnosis["model_threshold_false_negatives"], [("S1-001", "S2-002")])


if __name__ == "__main__":
    unittest.main()
