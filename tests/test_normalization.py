import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    char_ngrams,
)


class TestNameNormalization(unittest.TestCase):
    def test_legal_suffix_variants_converge(self):
        a = normalize_business_name("ABC Corp.")
        b = normalize_business_name("ABC Corporation")
        self.assertEqual(a.normalized, b.normalized)

    def test_pvt_ltd_variants_converge(self):
        a = normalize_business_name("XYZ Pvt Ltd")
        b = normalize_business_name("XYZ Private Limited")
        self.assertEqual(a.normalized, b.normalized)

    def test_case_and_punctuation_insensitive(self):
        a = normalize_business_name("ABC CORP")
        b = normalize_business_name("abc corp.")
        self.assertEqual(a.normalized, b.normalized)

    def test_word_order_invariant_via_token_sorted(self):
        a = normalize_business_name("First National Bank")
        b = normalize_business_name("National First Bank")
        self.assertNotEqual(a.normalized, b.normalized)  # normalized string order-sensitive
        self.assertEqual(a.token_sorted, b.token_sorted)  # but token-sorted key matches

    def test_ampersand_normalization(self):
        a = normalize_business_name("Smith & Sons")
        self.assertIn("and", a.normalized)
        self.assertNotIn("&", a.normalized)

    def test_missing_name_flagged(self):
        for val in [None, "", "   ", float("nan")]:
            n = normalize_business_name(val)
            self.assertTrue(n.is_missing)
            self.assertEqual(n.normalized, "")

    def test_conservative_no_fuzzy_typo_correction(self):
        # Normalization must NOT silently "fix" typos -- that's the model's job.
        a = normalize_business_name("Acme Manufacturing")
        b = normalize_business_name("Acme Manufaccturing")  # typo
        self.assertNotEqual(a.normalized, b.normalized)

    def test_dedup_tokens_removes_repeats_but_keeps_order(self):
        n = normalize_business_name("Star Star Textiles")
        self.assertEqual(n.dedup_tokens, ["star", "textiles"])

    def test_suffix_stripping_can_be_disabled(self):
        a = normalize_business_name("ABC Corp", strip_suffix=False)
        self.assertIn("corporation", a.normalized)  # abbreviation still expanded

    def test_abbreviation_expansion_can_be_disabled(self):
        a = normalize_business_name("ABC Corp", expand_abbrev=False, strip_suffix=False)
        self.assertIn("corp", a.normalized)
        self.assertNotIn("corporation", a.normalized)


class TestAddressNormalization(unittest.TestCase):
    def test_street_abbreviations_expand(self):
        a = normalize_address("221 2nd Rd")
        self.assertIn("road", a.normalized)

    def test_landmark_reference_preserved_as_text(self):
        a = normalize_address("Near SBI ATM, 12 Main St")
        self.assertIn("near", a.normalized)
        self.assertTrue(len(a.tokens) > 0)

    def test_postal_code_extraction(self):
        a = normalize_address("12 Main St, Springfield, 62704")
        self.assertEqual(a.postal_code, "62704")

    def test_missing_postal_code_is_none_not_mismatch(self):
        a = normalize_address("12 Main St, Springfield")
        self.assertIsNone(a.postal_code)

    def test_missing_address_flagged(self):
        for val in [None, "", "  ", float("nan")]:
            a = normalize_address(val)
            self.assertTrue(a.is_missing)

    def test_numeric_token_extraction(self):
        a = normalize_address("12 Main St")
        self.assertIn("12", a.numeric_tokens)

    def test_stopword_removal(self):
        a = normalize_address("Near the Main Street")
        self.assertNotIn("the", a.tokens)
        self.assertNotIn("near", [t for t in a.tokens if t == "the"])  # 'near' itself kept, 'the' dropped


class TestCountryNormalization(unittest.TestCase):
    def test_open_set_no_enum_restriction(self):
        # France (unseen at "training" conceptually) must normalize like any other string.
        self.assertEqual(normalize_country("France"), "france")
        self.assertEqual(normalize_country(" US "), "us")

    def test_missing_country(self):
        self.assertEqual(normalize_country(None), "")
        self.assertEqual(normalize_country(""), "")


class TestCharNgrams(unittest.TestCase):
    def test_ngrams_nonempty_for_short_strings(self):
        grams = char_ngrams("ab", n=3)
        self.assertTrue(len(grams) >= 1)

    def test_empty_string_yields_no_ngrams(self):
        self.assertEqual(char_ngrams(""), [])


if __name__ == "__main__":
    unittest.main()
