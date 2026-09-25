import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from business_entity_resolution.training import build_positive_pairs, build_training_pairs
from business_entity_resolution.config import NegativeSamplingConfig, SplitConfig, COL_ENTITY_ID, COL_COUNTRY
from business_entity_resolution.validation_split import split_s1_entities


class TestPositivePairConstruction(unittest.TestCase):
    def test_positives_extracted_from_ground_truth(self):
        gt = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "matched_entity_ids": ["S2-001,S3-001", ""],
        })
        positives = build_positive_pairs(gt, s1_ids_allowed={"S1-001", "S1-002"})
        self.assertEqual(positives, {("S1-001", "S2-001"), ("S1-001", "S3-001")})

    def test_positives_restricted_to_allowed_s1_ids(self):
        gt = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "matched_entity_ids": ["S2-001", "S3-001"],
        })
        positives = build_positive_pairs(gt, s1_ids_allowed={"S1-001"})
        self.assertEqual(positives, {("S1-001", "S2-001")})


class TestNegativeSampling(unittest.TestCase):
    def test_negatives_drawn_from_candidates_not_full_cartesian(self):
        candidates = {"S1-001": {"S2-001", "S2-002", "S2-003", "S9-999"}}
        positives = {("S1-001", "S2-001")}
        cfg = NegativeSamplingConfig(negative_per_positive=2, seed=1)
        pairs, labels = build_training_pairs(candidates, positives, cfg)
        neg_pairs = [p for p, l in zip(pairs, labels) if l == 0]
        for s1, t in neg_pairs:
            self.assertIn(t, candidates["S1-001"])  # never outside the candidate/blocking set
            self.assertNotEqual((s1, t), ("S1-001", "S2-001"))  # never the true positive

    def test_no_duplicate_pairs(self):
        candidates = {"S1-001": {"S2-001", "S2-002"}}
        positives = {("S1-001", "S2-001")}
        cfg = NegativeSamplingConfig(negative_per_positive=5, seed=1)
        pairs, labels = build_training_pairs(candidates, positives, cfg)
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_true_singleton_still_gets_negative_examples(self):
        # An S1 entity with NO positives should still contribute some negatives
        # (so the model sees "confident singleton" examples).
        candidates = {"S1-001": {"S2-001", "S2-002", "S2-003"}}
        positives = set()
        cfg = NegativeSamplingConfig(negative_per_positive=3, seed=1)
        pairs, labels = build_training_pairs(candidates, positives, cfg)
        self.assertTrue(len(pairs) > 0)
        self.assertTrue(all(l == 0 for l in labels))

    def test_deterministic_given_seed(self):
        candidates = {"S1-001": {f"S2-{i:03d}" for i in range(20)}}
        positives = {("S1-001", "S2-001")}
        cfg = NegativeSamplingConfig(negative_per_positive=3, seed=7)
        pairs1, labels1 = build_training_pairs(candidates, positives, cfg)
        pairs2, labels2 = build_training_pairs(candidates, positives, cfg)
        self.assertEqual(pairs1, pairs2)
        self.assertEqual(labels1, labels2)


class TestLeakageSafeSplit(unittest.TestCase):
    def test_train_val_are_disjoint(self):
        df = pd.DataFrame({
            COL_ENTITY_ID: [f"S1-{i:03d}" for i in range(100)],
            COL_COUNTRY: ["US"] * 50 + ["India"] * 50,
        })
        cfg = SplitConfig(validation_fraction=0.2, split_seed=42)
        result = split_s1_entities(df, cfg)
        self.assertEqual(set(result.train_s1_ids) & set(result.val_s1_ids), set())
        self.assertEqual(len(result.train_s1_ids) + len(result.val_s1_ids), 100)

    def test_split_deterministic_given_seed(self):
        df = pd.DataFrame({
            COL_ENTITY_ID: [f"S1-{i:03d}" for i in range(50)],
            COL_COUNTRY: ["US"] * 50,
        })
        cfg = SplitConfig(validation_fraction=0.3, split_seed=123)
        r1 = split_s1_entities(df, cfg)
        r2 = split_s1_entities(df, cfg)
        self.assertEqual(r1.train_s1_ids, r2.train_s1_ids)
        self.assertEqual(r1.val_s1_ids, r2.val_s1_ids)

    def test_holdout_country_proxy_excluded_from_both_splits(self):
        df = pd.DataFrame({
            COL_ENTITY_ID: [f"S1-{i:03d}" for i in range(30)],
            COL_COUNTRY: ["US"] * 20 + ["India"] * 10,
        })
        cfg = SplitConfig(validation_fraction=0.2, split_seed=1, holdout_country_proxy="India")
        result = split_s1_entities(df, cfg)
        self.assertEqual(len(result.holdout_s1_ids), 10)
        self.assertEqual(set(result.train_s1_ids) & set(result.holdout_s1_ids), set())
        self.assertEqual(set(result.val_s1_ids) & set(result.holdout_s1_ids), set())


if __name__ == "__main__":
    unittest.main()
