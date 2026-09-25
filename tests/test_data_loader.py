import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.data_loader import (
    load_source_file,
    load_ground_truth,
    parse_id_list,
    load_dataset,
)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class TestSourceLoading(unittest.TestCase):
    def test_basic_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s1.tsv")
            _write(p, "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                       "S1-001\tAcme Corp\t1 Main St\tUS\n")
            df, report = load_source_file(p, "S1")
            self.assertEqual(len(df), 1)
            self.assertEqual(report.n_rows, 1)
            self.assertTrue(report.is_clean())

    def test_missing_file_returns_empty_with_issue(self):
        df, report = load_source_file("/nonexistent/path.tsv", "S1")
        self.assertEqual(len(df), 0)
        self.assertTrue(len(report.issues) >= 1)

    def test_prefix_mismatch_rows_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s1.tsv")
            _write(p, "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                       "S1-001\tAcme Corp\t1 Main St\tUS\n"
                       "S2-999\tWrong Prefix\t2 Oak Ave\tUS\n")
            df, report = load_source_file(p, "S1")
            self.assertEqual(len(df), 1)
            self.assertEqual(report.n_prefix_mismatch, 1)

    def test_duplicate_ids_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s1.tsv")
            _write(p, "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                       "S1-001\tAcme Corp\t1 Main St\tUS\n"
                       "S1-001\tAcme Corp Dup\t1 Main St\tUS\n")
            df, report = load_source_file(p, "S1")
            self.assertEqual(len(df), 1)
            self.assertEqual(report.n_duplicate_ids, 1)

    def test_null_empty_ids_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s1.tsv")
            _write(p, "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                       "\tAcme Corp\t1 Main St\tUS\n"
                       "S1-002\tOther Corp\t2 Oak Ave\tUS\n")
            df, report = load_source_file(p, "S1")
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["entity_id"], "S1-002")

    def test_missing_required_columns_reported_and_filled(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s1.tsv")
            _write(p, "entity_id\tbusiness_name\n"
                       "S1-001\tAcme Corp\n")
            df, report = load_source_file(p, "S1")
            self.assertIn("business_address", df.columns)
            self.assertTrue(any("missing required columns" in i for i in report.issues))

    def test_commas_in_address_and_name_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s1.tsv")
            _write(p, "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                       "S1-001\tAcme, Inc.\t1 Main St, Suite 4, Springfield\tUS\n")
            df, report = load_source_file(p, "S1")
            self.assertEqual(df.iloc[0]["business_name"], "Acme, Inc.")
            self.assertIn(",", df.iloc[0]["business_address"])


class TestGroundTruthLoading(unittest.TestCase):
    def test_parse_id_list(self):
        self.assertEqual(parse_id_list("S2-001,S3-002"), ["S2-001", "S3-002"])
        self.assertEqual(parse_id_list(""), [])
        self.assertEqual(parse_id_list(None), [])
        self.assertEqual(parse_id_list("nan"), [])
        self.assertEqual(parse_id_list(" S2-001 , S3-002 "), ["S2-001", "S3-002"])

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gt.tsv")
            _write(p, "wrong_col\tmatched_entity_ids\nS1-001\tS2-001\n")
            with self.assertRaises(ValueError):
                load_ground_truth(p)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_ground_truth("/nonexistent/gt.tsv")


class TestDatasetBundle(unittest.TestCase):
    def test_load_dataset_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "train")
            _write(os.path.join(base, "train_source1.tsv"),
                   "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-001\tAcme\t1 Main St\tUS\n")
            _write(os.path.join(base, "train_source2.tsv"),
                   "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-001\tAcme Corp\t1 Main St\tUS\n")
            _write(os.path.join(base, "train_source3.tsv"),
                   "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-001\tOther\t2 Oak Ave\tUS\n")
            _write(os.path.join(base, "train_ground_truth.tsv"),
                   "source1_entity_id\tmatched_entity_ids\nS1-001\tS2-001\n")
            bundle = load_dataset(d, "train")
            self.assertEqual(len(bundle.s1), 1)
            self.assertEqual(len(bundle.s2), 1)
            self.assertEqual(len(bundle.s3), 1)
            self.assertIsNotNone(bundle.ground_truth)
            self.assertEqual(bundle.all_s2_s3_ids(), {"S2-001", "S3-001"})


if __name__ == "__main__":
    unittest.main()
