import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.output import check_decisions, write_outputs, write_matching_results, write_candidate_pairs


class TestOutputInvariants(unittest.TestCase):
    def test_matched_subset_of_candidates_violation_detected(self):
        s1_ids = ["S1-001"]
        matches = {"S1-001": {"S2-999"}}       # S2-999 not in candidates!
        candidates = {"S1-001": {"S2-001"}}
        report = check_decisions(s1_ids, matches, candidates, valid_target_ids={"S2-001", "S2-999"})
        self.assertFalse(report.ok)
        self.assertTrue(any("not present in candidate_pairs" in i for i in report.issues))

    def test_self_match_detected(self):
        s1_ids = ["S1-001"]
        matches = {"S1-001": {"S1-001"}}  # self-match
        candidates = {"S1-001": {"S1-001"}}
        report = check_decisions(s1_ids, matches, candidates, valid_target_ids={"S1-001"})
        self.assertFalse(report.ok)
        self.assertTrue(any("self-match" in i for i in report.issues))

    def test_invalid_target_id_detected(self):
        s1_ids = ["S1-001"]
        matches = {"S1-001": {"S9-999"}}  # not a valid target id
        candidates = {"S1-001": {"S9-999"}}
        report = check_decisions(s1_ids, matches, candidates, valid_target_ids={"S2-001"})
        self.assertFalse(report.ok)

    def test_missing_s1_entity_detected(self):
        s1_ids = ["S1-001", "S1-002"]
        matches = {"S1-001": set()}  # S1-002 missing entirely
        candidates = {"S1-001": set()}
        report = check_decisions(s1_ids, matches, candidates, valid_target_ids=set())
        self.assertFalse(report.ok)
        self.assertTrue(any("missing from matches" in i for i in report.issues))

    def test_clean_output_passes(self):
        s1_ids = ["S1-001", "S1-002"]
        matches = {"S1-001": {"S2-001"}, "S1-002": set()}
        candidates = {"S1-001": {"S2-001", "S3-001"}, "S1-002": set()}
        report = check_decisions(s1_ids, matches, candidates, valid_target_ids={"S2-001", "S3-001"})
        self.assertTrue(report.ok)


class TestOutputWriting(unittest.TestCase):
    def test_write_matching_results_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "matching_results.tsv")
            write_matching_results(path, ["S1-001", "S1-002"], {"S1-001": {"S2-001", "S3-001"}, "S1-002": set()})
            with open(path) as f:
                lines = f.read().splitlines()
            self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
            self.assertIn("S1-002\t", lines)

    def test_write_candidate_pairs_sorted_no_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "candidate_pairs.tsv")
            write_candidate_pairs(path, ["S1-001"], {"S1-001": {"S3-002", "S2-001"}})
            with open(path) as f:
                lines = f.read().splitlines()
            # sorted() alphabetically -> S2-001 before S3-002
            self.assertEqual(lines[1], "S1-001\tS2-001,S3-002")

    def test_write_outputs_every_test_entity_present(self):
        with tempfile.TemporaryDirectory() as d:
            s1_ids = ["S1-001", "S1-002", "S1-003"]
            matches = {"S1-001": {"S2-001"}}  # S1-002, S1-003 missing from matches dict
            candidates = {"S1-001": {"S2-001"}}
            report = write_outputs(d, s1_ids, matches, candidates, valid_target_ids={"S2-001"})
            with open(os.path.join(d, "matching_results.tsv")) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 4)  # header + 3 rows, every S1 id present


if __name__ == "__main__":
    unittest.main()
