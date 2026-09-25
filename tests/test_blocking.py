import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from business_entity_resolution.blocking import build_corpus, generate_candidates
from business_entity_resolution.config import COL_ENTITY_ID, BlockingConfig
from business_entity_resolution.normalization import normalize_address, normalize_business_name


def _corpus(rows):
    df = pd.DataFrame(rows)
    names = {r["entity_id"]: normalize_business_name(r["business_name"]) for r in rows}
    addrs = {r["entity_id"]: normalize_address(r["business_address"]) for r in rows}
    return build_corpus(df, names, addrs)


class TestBlocking(unittest.TestCase):
    def setUp(self):
        self.s1_rows = [
            {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St, Springfield"},
            {"entity_id": "S1-002", "business_name": "Totally Unrelated Biz", "business_address": "99 Nowhere Rd"},
        ]
        self.target_rows = [
            {"entity_id": "S2-001", "business_name": "Acme Corporation", "business_address": "1 Main Street, Springfield"},
            {"entity_id": "S3-001", "business_name": "Blue River Logistics", "business_address": "5 Oak Ave"},
        ]
        self.s1_corpus = _corpus(self.s1_rows)
        self.target_corpus = _corpus(self.target_rows)

    def test_exact_and_fuzzy_blocking_recovers_true_match(self):
        cfg = BlockingConfig()
        cands = generate_candidates(self.s1_corpus, self.target_corpus, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_every_s1_id_present_even_with_no_candidates(self):
        cfg = BlockingConfig()
        cands = generate_candidates(self.s1_corpus, self.target_corpus, cfg)
        self.assertIn("S1-001", cands)
        self.assertIn("S1-002", cands)

    def test_max_candidates_cap_enforced(self):
        # Build a target corpus with many near-identical names to force overflow.
        many_targets = [
            {"entity_id": f"S2-{i:03d}", "business_name": "Acme Corporation", "business_address": "1 Main Street"}
            for i in range(50)
        ]
        target_corpus = _corpus(many_targets)
        cfg = BlockingConfig(max_candidates_per_entity=10)
        cands = generate_candidates(self.s1_corpus, target_corpus, cfg)
        self.assertLessEqual(len(cands["S1-001"]), 10)

    def test_no_hard_country_filtering(self):
        # Blocking must never take a country argument / filter by it structurally.
        import inspect
        sig = inspect.signature(generate_candidates)
        self.assertNotIn("country", sig.parameters)

    def test_disabling_all_strategies_yields_empty_candidates(self):
        cfg = BlockingConfig(
            enable_exact_name_block=False, enable_token_block=False,
            enable_ngram_block=False, enable_address_block=False, enable_fuzzy_block=False,
        )
        cands = generate_candidates(self.s1_corpus, self.target_corpus, cfg)
        self.assertEqual(cands["S1-001"], set())
        self.assertEqual(cands["S1-002"], set())

    def test_candidates_only_reference_target_ids(self):
        cfg = BlockingConfig()
        cands = generate_candidates(self.s1_corpus, self.target_corpus, cfg)
        valid_ids = set(self.target_corpus.ids)
        for s1_id, cand_set in cands.items():
            self.assertTrue(cand_set.issubset(valid_ids))


if __name__ == "__main__":
    unittest.main()
