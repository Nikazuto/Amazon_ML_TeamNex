"""
test_blocking.py
=================
Focused unit tests for the blocking / candidate-generation stage
(blocking.py), using small synthetic datasets so the module can be
verified without the full ~100k-record challenge dataset.

Coverage map (Person B deliverables):
  * exact names                      -> TestExactAndTokenSortedBlocking
  * abbreviations / legal suffixes   -> TestAbbreviationHandling
  * token reordering                 -> TestExactAndTokenSortedBlocking
  * punctuation                      -> TestExactAndTokenSortedBlocking
  * typos / transliteration-ish noise-> TestFuzzyAndNgramBlockingTypos
  * address variation                -> TestAddressBlocking
  * missing fields                   -> TestMissingFields
  * duplicate candidates             -> TestDuplicateCandidates
  * common tokens                    -> TestCommonTokenHandling
  * candidate caps                   -> TestCandidateCap
  * config toggles / interface       -> TestConfigToggles, TestPublicInterface
  * evaluation.py integration        -> TestBlockingEvaluationIntegration
"""
import os
import subprocess
import sys
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from src.business_entity_resolution.blocking import build_corpus, generate_candidates
from src.business_entity_resolution.config import COL_ENTITY_ID, BlockingConfig
from src.business_entity_resolution.evaluation import evaluate_blocking
from src.business_entity_resolution.normalization import normalize_address, normalize_business_name


def corpus_from_rows(rows):
    """Build a NormalizedCorpus straight from raw (entity_id, business_name,
    business_address) dicts -- mirrors what pipeline.py does after calling
    normalize_business_name/normalize_address per row."""
    df = pd.DataFrame(rows)
    names = {r["entity_id"]: normalize_business_name(r["business_name"]) for r in rows}
    addrs = {r["entity_id"]: normalize_address(r["business_address"]) for r in rows}
    return build_corpus(df, names, addrs)


class TestExactAndTokenSortedBlocking(unittest.TestCase):
    """Strategy A: exact normalized-name / token-sorted-name blocking."""

    def test_exact_name_match_recovered(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        cands = generate_candidates(s1, target, BlockingConfig())
        self.assertIn("S2-001", cands["S1-001"])

    def test_punctuation_differences_do_not_prevent_match(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Smith & Sons, Inc.", "business_address": "1 Main St"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Smith and Sons Inc", "business_address": "1 Main St"}])
        cands = generate_candidates(s1, target, BlockingConfig())
        self.assertIn("S2-001", cands["S1-001"])

    def test_word_order_transposition_recovered_via_token_sort(self):
        # "token_sorted" is an order-invariant key: reordered words still
        # produce a match even with the ngram/fuzzy strategies disabled.
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Global Widget Manufacturing", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Widget Global Manufacturing", "business_address": ""}])
        cfg = BlockingConfig(enable_ngram_block=False, enable_fuzzy_block=False, enable_token_block=False, enable_address_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_unrelated_names_not_exact_matched(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Totally Different Business", "business_address": "99 Nowhere Rd"}])
        cfg = BlockingConfig(enable_ngram_block=False, enable_fuzzy_block=False, enable_token_block=False, enable_address_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertNotIn("S2-001", cands["S1-001"])


class TestAbbreviationHandling(unittest.TestCase):
    """Legal-suffix / abbreviation convergence (relies on normalization.py,
    but verified here as an integration contract: blocking must actually
    key off the *normalized* representation, not the raw string)."""

    def test_pvt_ltd_vs_private_limited_converge(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Sundar Traders Pvt Ltd", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Sundar Traders Private Limited", "business_address": ""}])
        cfg = BlockingConfig(enable_ngram_block=False, enable_fuzzy_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_corp_vs_corporation_converge(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Blue Sky Corp", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Blue Sky Corporation", "business_address": ""}])
        cfg = BlockingConfig(enable_ngram_block=False, enable_fuzzy_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_ampersand_vs_and_converge(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Johnson & Johnson", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Johnson and Johnson", "business_address": ""}])
        cfg = BlockingConfig(enable_ngram_block=False, enable_fuzzy_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])


class TestFuzzyAndNgramBlockingTypos(unittest.TestCase):
    """Strategies C/F: char n-gram TF-IDF retrieval must recover typos and
    near-miss spellings that exact/token blocking cannot."""

    def test_single_character_typo_recovered_by_ngram_block(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Fenwick Logistics", "business_address": ""}])
        target = corpus_from_rows([
            {"entity_id": "S2-001", "business_name": "Fenwik Logistics", "business_address": ""},  # missing 'c'
            {"entity_id": "S3-001", "business_name": "Totally Unrelated Co", "business_address": ""},
        ])
        # Isolate the n-gram strategy: exact/token/address/fuzzy off. With
        # only 2 target records, ngram_top_k must be pinned below the
        # target-pool size (top_k=1) or NearestNeighbors trivially returns
        # every target as a "neighbor" (k = min(top_k, n_targets)), which
        # would defeat the point of this test rather than exercise it.
        cfg = BlockingConfig(enable_exact_name_block=False, enable_token_block=False,
                              enable_address_block=False, enable_fuzzy_block=False,
                              ngram_top_k=1)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])
        self.assertNotIn("S3-001", cands["S1-001"])

    def test_typo_plus_address_noise_recovered_by_combined_fuzzy_block(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Riverside Bakery", "business_address": "42 Elm Street, Springfield"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Riverside Bakry", "business_address": "42 Elm St, Springfeld"}])
        cfg = BlockingConfig(enable_exact_name_block=False, enable_token_block=False,
                              enable_address_block=False, enable_ngram_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_transliteration_style_variation_recovered(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Al-Fahad Trading", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Al Fahd Trading", "business_address": ""}])
        cfg = BlockingConfig(enable_exact_name_block=False, enable_token_block=False, enable_address_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])


class TestAddressBlocking(unittest.TestCase):
    """Strategy D: shared address tokens + postal-code exact match."""

    def test_street_abbreviation_variants_converge(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "", "business_address": "12 Oak Rd, Denver"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "", "business_address": "12 Oak Road, Denver"}])
        cfg = BlockingConfig(enable_exact_name_block=False, enable_token_block=False,
                              enable_ngram_block=False, enable_fuzzy_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_postal_code_exact_match_recovers_candidate_despite_different_name(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Unrelated Name One", "business_address": "PO Box, 94107"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Unrelated Name Two", "business_address": "Some Suite, 94107"}])
        cfg = BlockingConfig(enable_exact_name_block=False, enable_token_block=False,
                              enable_ngram_block=False, enable_fuzzy_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_landmark_based_address_does_not_crash_and_shared_tokens_still_match(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "", "business_address": "Near SBI ATM, 12 Main Street"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "", "business_address": "12 Main Street, opposite the park"}])
        cfg = BlockingConfig(enable_exact_name_block=False, enable_token_block=False,
                              enable_ngram_block=False, enable_fuzzy_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertIn("S2-001", cands["S1-001"])

    def test_address_min_token_length_is_configurable_not_hardcoded(self):
        # Regression test for the FR-8.3 config-completeness bug: address
        # token blocking used to hard-code a minimum token length of 3
        # instead of reading it from BlockingConfig. A 2-character shared
        # token should only be picked up once the threshold is explicitly
        # lowered via config. "hq" is used (not "nw"/"n"/"s"/"e"/"w", which
        # ADDRESS_ABBREVIATIONS expands to full directional words).
        isolate_cfg = dict(enable_exact_name_block=False, enable_token_block=False,
                            enable_ngram_block=False, enable_fuzzy_block=False)
        default_cfg = BlockingConfig(**isolate_cfg)  # address_min_token_length default = 3
        lowered_cfg = BlockingConfig(address_min_token_length=2, **isolate_cfg)

        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "", "business_address": "hq plaza"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "", "business_address": "hq plaza tower"}])
        cands_default = generate_candidates(s1, target, default_cfg)
        cands_lowered = generate_candidates(s1, target, lowered_cfg)
        self.assertIn("S2-001", cands_lowered["S1-001"])
        self.assertIn("S2-001", cands_default["S1-001"])  # still matches via "plaza"

        # Now isolate strictly on the 2-char token by removing the shared 5-char word.
        s1b = corpus_from_rows([{"entity_id": "S1-001", "business_name": "", "business_address": "hq"}])
        target_b = corpus_from_rows([{"entity_id": "S2-001", "business_name": "", "business_address": "hq"}])
        cands_default_b = generate_candidates(s1b, target_b, default_cfg)
        cands_lowered_b = generate_candidates(s1b, target_b, lowered_cfg)
        self.assertNotIn("S2-001", cands_default_b["S1-001"])
        self.assertIn("S2-001", cands_lowered_b["S1-001"])


class TestMissingFields(unittest.TestCase):
    """Records with a missing name and/or address must not crash blocking
    and must not be spuriously matched to unrelated records via an empty-
    string key."""

    def test_missing_name_does_not_crash_and_yields_no_spurious_match(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "", "business_address": "1 Main St"}])
        target = corpus_from_rows([
            {"entity_id": "S2-001", "business_name": "", "business_address": "99 Other Ave"},  # also empty name
            {"entity_id": "S3-001", "business_name": "Some Business", "business_address": "1 Main St"},
        ])
        # Isolate the exact/token/address name-key strategies -- this test
        # is specifically about the "" (empty-name) key never being treated
        # as a shared blocking key. The ngram/fuzzy ANN strategies are
        # excluded here because with only 2 target records they trivially
        # return every target as a top-k neighbor regardless of similarity
        # (k = min(top_k, n_targets)); that's expected ANN behavior on a
        # toy-sized corpus, not the empty-name edge case being tested.
        cfg = BlockingConfig(enable_ngram_block=False, enable_fuzzy_block=False)
        cands = generate_candidates(s1, target, cfg)
        # two empty-name records must not be matched to each other via a "" key
        self.assertNotIn("S2-001", cands["S1-001"])
        # but the shared address ("1 main street") should still surface S3-001
        self.assertIn("S3-001", cands["S1-001"])

    def test_missing_address_does_not_crash(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": ""}])
        cands = generate_candidates(s1, target, BlockingConfig())
        # name-based strategies should still recover the match
        self.assertIn("S2-001", cands["S1-001"])

    def test_both_fields_missing_yields_empty_but_present_entry(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Something", "business_address": "Somewhere"}])
        cands = generate_candidates(s1, target, BlockingConfig())
        self.assertIn("S1-001", cands)  # entity always present, even with 0 candidates

    def test_empty_target_corpus_does_not_crash(self):
        # Mirrors pipeline.py's combine_targets(): even a zero-row S2/S3
        # frame always retains the required columns (data_loader.py
        # guarantees this), so build an empty-but-columned target corpus
        # rather than a columnless empty DataFrame.
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        empty_df = pd.DataFrame(columns=[COL_ENTITY_ID, "business_name", "business_address", "country"])
        target = build_corpus(empty_df, {}, {})
        cands = generate_candidates(s1, target, BlockingConfig())
        self.assertEqual(cands["S1-001"], set())


class TestDuplicateCandidates(unittest.TestCase):
    """The union of independent blocking strategies must never yield
    duplicate entries for the same target id (candidates is Set[str] by
    construction, but we verify the *count* behaves accordingly, since a
    strategy incorrectly appending to a list instead of a set would be an
    easy regression to introduce)."""

    def test_candidate_found_by_multiple_strategies_appears_once(self):
        # This target matches via exact-name AND address-token AND postal
        # code AND n-gram AND fuzzy blocking simultaneously.
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St, 94107"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "1 Main St, 94107"}])
        cands = generate_candidates(s1, target, BlockingConfig())
        self.assertEqual(cands["S1-001"], {"S2-001"})  # exactly one entry, not a multiset

    def test_no_duplicate_ids_in_returned_sets_by_construction(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        cands = generate_candidates(s1, target, BlockingConfig())
        for cand_set in cands.values():
            self.assertEqual(len(cand_set), len(set(cand_set)))


class TestCommonTokenHandling(unittest.TestCase):
    """Strategy B: tokens that are too common carry no blocking signal and
    must be excluded (both the global stopset AND the configurable
    max_token_block_size boundary)."""

    def test_stopset_token_excluded_even_if_shared(self):
        # "group" is in COMMON_TOKEN_STOPSET; sharing only that token must
        # not produce a candidate via token blocking.
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Group Holdings Zyx", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Group Ventures Qwe", "business_address": ""}])
        cfg = BlockingConfig(enable_exact_name_block=False, enable_ngram_block=False,
                              enable_fuzzy_block=False, enable_address_block=False)
        cands = generate_candidates(s1, target, cfg)
        self.assertNotIn("S2-001", cands["S1-001"])

    def test_token_block_size_boundary_included_at_exactly_max(self):
        # A token shared by exactly max_token_block_size target records is
        # still informative enough to be included (<=, not <).
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Zynthara Holdings", "business_address": ""}])
        target_rows = [
            {"entity_id": f"S2-{i:03d}", "business_name": "Zynthara Enterprises", "business_address": ""}
            for i in range(5)
        ]
        target = corpus_from_rows(target_rows)
        cfg = BlockingConfig(enable_exact_name_block=False, enable_ngram_block=False,
                              enable_fuzzy_block=False, enable_address_block=False,
                              max_token_block_size=5)
        cands = generate_candidates(s1, target, cfg)
        self.assertEqual(len(cands["S1-001"]), 5)

    def test_token_block_size_boundary_excluded_when_over_max(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Zynthara Holdings", "business_address": ""}])
        target_rows = [
            {"entity_id": f"S2-{i:03d}", "business_name": "Zynthara Enterprises", "business_address": ""}
            for i in range(6)  # one more than max_token_block_size below
        ]
        target = corpus_from_rows(target_rows)
        cfg = BlockingConfig(enable_exact_name_block=False, enable_ngram_block=False,
                              enable_fuzzy_block=False, enable_address_block=False,
                              max_token_block_size=5)
        cands = generate_candidates(s1, target, cfg)
        # "zynthara" now appears in 6 records (>5) and is excluded as an
        # uninformative token; nothing else is shared, so no candidates.
        self.assertEqual(len(cands["S1-001"]), 0)

    def test_min_token_length_respected(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Ab Cd Efgh", "business_address": ""}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Ab Cd Ijkl", "business_address": ""}])
        cfg = BlockingConfig(enable_exact_name_block=False, enable_ngram_block=False,
                              enable_fuzzy_block=False, enable_address_block=False,
                              min_token_length=3)
        cands = generate_candidates(s1, target, cfg)
        # only 2-char tokens ("ab", "cd") are shared; below min_token_length=3
        self.assertNotIn("S2-001", cands["S1-001"])


class TestCandidateCap(unittest.TestCase):
    """Safety valve: max_candidates_per_entity must be enforced, and must
    be deterministic (regression test for the hash-seed tie-break bug)."""

    def test_max_candidates_cap_enforced(self):
        many_targets = [
            {"entity_id": f"S2-{i:03d}", "business_name": "Acme Corporation", "business_address": "1 Main Street"}
            for i in range(50)
        ]
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        target = corpus_from_rows(many_targets)
        cfg = BlockingConfig(max_candidates_per_entity=10)
        cands = generate_candidates(s1, target, cfg)
        self.assertLessEqual(len(cands["S1-001"]), 10)

    def test_cap_is_deterministic_across_repeated_calls(self):
        many_targets = [
            {"entity_id": f"S2-{i:03d}", "business_name": "Acme Corporation", "business_address": "1 Main Street"}
            for i in range(50)
        ]
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        target = corpus_from_rows(many_targets)
        cfg = BlockingConfig(max_candidates_per_entity=5)
        result_a = generate_candidates(s1, target, cfg)
        result_b = generate_candidates(s1, target, cfg)
        self.assertEqual(result_a["S1-001"], result_b["S1-001"])

    def test_cap_is_deterministic_across_hash_seeds(self):
        """Regression test for a real bug: the cap's tie-breaker previously
        sorted only by |len diff|, so ties were broken by set-iteration
        order, which depends on PYTHONHASHSEED. Same input/config must now
        yield the identical capped candidate set regardless of hash seed."""
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {os.path.join(os.path.dirname(__file__), "..", "src")!r})
            import pandas as pd
            from business_entity_resolution.blocking import build_corpus, generate_candidates
            from business_entity_resolution.config import BlockingConfig
            from business_entity_resolution.normalization import normalize_address, normalize_business_name

            def corpus(rows):
                df = pd.DataFrame(rows)
                names = {{r["entity_id"]: normalize_business_name(r["business_name"]) for r in rows}}
                addrs = {{r["entity_id"]: normalize_address(r["business_address"]) for r in rows}}
                return build_corpus(df, names, addrs)

            s1 = corpus([{{"entity_id": "S1-001", "business_name": "Acme Corporation", "business_address": "1 Main St"}}])
            target_rows = [
                {{"entity_id": f"S2-{{i:03d}}", "business_name": "Acme Corporation", "business_address": "1 Main St"}}
                for i in range(50)
            ]
            target = corpus(target_rows)
            cfg = BlockingConfig(max_candidates_per_entity=5)
            cands = generate_candidates(s1, target, cfg)
            print(",".join(sorted(cands["S1-001"])))
        """)
        outputs = set()
        for seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            outputs.add(result.stdout.strip())
        self.assertEqual(len(outputs), 1, msg=f"cap result varied across hash seeds: {outputs}")


class TestConfigToggles(unittest.TestCase):
    def test_disabling_all_strategies_yields_empty_candidates(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St, Springfield"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Acme Corporation", "business_address": "1 Main Street, Springfield"}])
        cfg = BlockingConfig(
            enable_exact_name_block=False, enable_token_block=False,
            enable_ngram_block=False, enable_address_block=False, enable_fuzzy_block=False,
        )
        cands = generate_candidates(s1, target, cfg)
        self.assertEqual(cands["S1-001"], set())

    def test_every_s1_id_present_even_with_no_candidates(self):
        s1 = corpus_from_rows([
            {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"},
            {"entity_id": "S1-002", "business_name": "Totally Unrelated Biz", "business_address": "99 Nowhere Rd"},
        ])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Acme Corporation", "business_address": "1 Main Street"}])
        cands = generate_candidates(s1, target, BlockingConfig())
        self.assertIn("S1-001", cands)
        self.assertIn("S1-002", cands)


class TestPublicInterface(unittest.TestCase):
    def test_no_hard_country_filtering(self):
        # Blocking must never take a country argument / filter by it structurally (FR-1.3 / NFR-2).
        import inspect
        sig = inspect.signature(generate_candidates)
        self.assertNotIn("country", sig.parameters)

    def test_candidates_only_reference_target_ids(self):
        s1 = corpus_from_rows([
            {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St, Springfield"},
            {"entity_id": "S1-002", "business_name": "Totally Unrelated Biz", "business_address": "99 Nowhere Rd"},
        ])
        target = corpus_from_rows([
            {"entity_id": "S2-001", "business_name": "Acme Corporation", "business_address": "1 Main Street, Springfield"},
            {"entity_id": "S3-001", "business_name": "Blue River Logistics", "business_address": "5 Oak Ave"},
        ])
        cands = generate_candidates(s1, target, BlockingConfig())
        valid_ids = set(target.ids)
        for s1_id, cand_set in cands.items():
            self.assertTrue(cand_set.issubset(valid_ids))

    def test_return_type_is_dict_of_sets(self):
        s1 = corpus_from_rows([{"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        target = corpus_from_rows([{"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "1 Main St"}])
        cands = generate_candidates(s1, target, BlockingConfig())
        self.assertIsInstance(cands, dict)
        for v in cands.values():
            self.assertIsInstance(v, set)


class TestBlockingEvaluationIntegration(unittest.TestCase):
    """Confirms blocking.py's output plugs directly into evaluation.py's
    evaluate_blocking() (the function pipeline.py actually calls) without
    any adapter -- the real integration point the pipeline's blocking-
    recall diagnostics depend on."""

    def test_generate_candidates_output_is_consumable_by_evaluate_blocking(self):
        s1_rows = [
            {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "1 Main St"},
            {"entity_id": "S1-002", "business_name": "Totally Unrelated", "business_address": "99 Nowhere Rd"},
        ]
        target_rows = [
            {"entity_id": "S2-001", "business_name": "Acme Corporation", "business_address": "1 Main Street"},
            {"entity_id": "S3-001", "business_name": "Blue River Logistics", "business_address": "5 Oak Ave"},
        ]
        s1 = corpus_from_rows(s1_rows)
        target = corpus_from_rows(target_rows)
        cands = generate_candidates(s1, target, BlockingConfig())

        ground_truth = {"S1-001": {"S2-001"}, "S1-002": set()}
        report = evaluate_blocking(cands, ground_truth, n_target_records=len(target.ids))

        self.assertEqual(report.blocking_recall, 1.0)
        self.assertEqual(report.n_true_matches_missed, 0)
        self.assertGreaterEqual(report.avg_candidates_per_entity, 0)


if __name__ == "__main__":
    unittest.main()
