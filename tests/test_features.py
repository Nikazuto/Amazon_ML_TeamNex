import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.features import compute_pair_features, extract_features_for_pairs, FEATURE_NAMES
from business_entity_resolution.normalization import normalize_address, normalize_business_name


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.s1_names = {"S1-001": normalize_business_name("Acme Corp"), "S1-002": normalize_business_name(None)}
        self.s1_addrs = {"S1-001": normalize_address("1 Main St, Springfield, 62704"), "S1-002": normalize_address(None)}
        self.s1_countries = {"S1-001": "US", "S1-002": "US"}

        self.t_names = {
            "S2-001": normalize_business_name("Acme Corporation"),
            "S2-002": normalize_business_name("Totally Different Business"),
        }
        self.t_addrs = {
            "S2-001": normalize_address("1 Main Street, Springfield, 62704"),
            "S2-002": normalize_address("999 Nowhere Rd"),
        }
        self.t_countries = {"S2-001": "US", "S2-002": "France"}

    def test_similar_pair_scores_higher_than_dissimilar(self):
        feats_sim = compute_pair_features(
            "S1-001", "S2-001", self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        feats_dissim = compute_pair_features(
            "S1-001", "S2-002", self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        self.assertGreater(feats_sim["name_char_ngram_cosine"], feats_dissim["name_char_ngram_cosine"])
        self.assertGreater(feats_sim["addr_char_ngram_cosine"], feats_dissim["addr_char_ngram_cosine"])

    def test_postal_code_equal_when_both_present(self):
        feats = compute_pair_features(
            "S1-001", "S2-001", self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        self.assertEqual(feats["addr_postal_equal"], 1.0)

    def test_missing_name_does_not_produce_false_high_similarity(self):
        feats = compute_pair_features(
            "S1-002", "S2-001", self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        self.assertEqual(feats["name_missing_s1"], 1.0)
        # missing side must not produce a maximal (1.0) similarity score
        self.assertNotEqual(feats["name_char_ngram_cosine"], 1.0)
        self.assertNotEqual(feats["name_exact_match"], 1.0)

    def test_country_is_soft_signal_only_never_a_filter(self):
        # A pair with mismatched country must still get a full feature row (not excluded).
        feats = compute_pair_features(
            "S1-001", "S2-002", self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        self.assertEqual(feats["country_exact_match"], 0.0)
        self.assertEqual(len(feats), len(FEATURE_NAMES))

    def test_all_declared_feature_names_present(self):
        feats = compute_pair_features(
            "S1-001", "S2-001", self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        self.assertEqual(set(feats.keys()), set(FEATURE_NAMES))

    def test_extract_features_for_pairs_shape(self):
        pairs = [("S1-001", "S2-001"), ("S1-001", "S2-002")]
        df = extract_features_for_pairs(
            pairs, self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns[:2]), ["source1_entity_id", "target_entity_id"])
        for f in FEATURE_NAMES:
            self.assertIn(f, df.columns)

    def test_extract_features_empty_pairs(self):
        df = extract_features_for_pairs(
            [], self.s1_names, self.s1_addrs, self.s1_countries,
            self.t_names, self.t_addrs, self.t_countries,
        )
        self.assertEqual(len(df), 0)


if __name__ == "__main__":
    unittest.main()
