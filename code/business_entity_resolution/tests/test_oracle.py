import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ber.oracle import run_oracle


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split = self.root / "train"
        self.split.mkdir()
        self.output = self.root / "oracle"
        self.columns = ("entity_id", "business_name", "business_address", "country")
        self.write("source1", self.columns, [
            ("S1-1", "Café Lumière", "12 Rue Victor", "France"),
            ("S1-2", "Alpha Limited", "99 Main Street", "US"),
            ("S1-3", "Singleton", "20 Sample Road", "India"),
        ])
        self.write("source2", self.columns, [
            ("S2-10", "Cafe Lumiere", "12 Rue Victor", "France"),
            ("S2-20", "Alpha Ltd", "98 Main Street", "US"),
            ("S2-90", "Distractor", "4 Nowhere", "France"),
        ])
        self.write("source3", self.columns, [
            ("S3-10", "東京", "12 Rue Victor", "France"),
            ("S3-20", "ALPHA LIMITED", "", "US"),
        ])
        self.write("ground_truth", ("source1_entity_id", "matched_entity_ids"), [
            ("S1-1", "S2-10,S3-10"), ("S1-2", "S2-20,S3-20"), ("S1-3", ""),
        ])

    def write(self, name, columns, rows):
        with (self.split / f"{name}.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)

    def test_all_anchor_oracle_categories_and_raw_preservation(self):
        before = {path.name: path.read_bytes() for path in self.split.iterdir()}
        report = run_oracle(self.split, self.output, anchor_limit=10)
        self.assertEqual(report["sampled_anchors"], 3)
        self.assertEqual(report["sampled_singletons"], 1)
        self.assertEqual(report["positive_pairs"], 4)
        self.assertEqual(report["needed_unique_targets"], 4)
        self.assertEqual(report["target_rows_scanned"], {"source2": 3, "source3": 2})
        self.assertEqual(report["category_counts"]["name_script_disjoint"], 1)
        self.assertEqual(report["category_counts"]["low_name_strong_address"], 1)
        self.assertEqual(report["category_counts"]["any_address_missing"], 1)
        self.assertEqual(report["category_counts"]["address_numbers_conflict"], 1)
        self.assertEqual(report["similarity_quantiles"]["name_edit_similarity"]["count"], 4)
        self.assertEqual(report["match_count_histogram"], {"0": 1, "2": 2})
        with (self.output / "positive_oracle_hard_examples.tsv").open(encoding="utf-8", newline="") as stream:
            examples = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(examples), 4)
        self.assertTrue(any(row["left_name"] == "Café Lumière" for row in examples))
        self.assertTrue(any(row["right_name"] == "東京" for row in examples))
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.split.iterdir()})
        self.assertEqual(json.loads((self.output / "positive_oracle.json").read_text()), report)

    def test_resume_reads_verified_saved_report(self):
        report = run_oracle(self.split, self.output, anchor_limit=2, seed=17)
        with patch("ber.oracle._read_rows", side_effect=AssertionError("must not rescan unchanged inputs")):
            self.assertEqual(run_oracle(self.split, self.output, anchor_limit=2, seed=17), report)

    def test_corrupt_examples_force_regeneration(self):
        report = run_oracle(self.split, self.output)
        (self.output / "positive_oracle_hard_examples.tsv").write_text("corrupted")
        self.assertEqual(run_oracle(self.split, self.output), report)

    def test_uniform_reservoir_reproducible_and_bounded(self):
        first = run_oracle(self.split, self.output, anchor_limit=1, seed=8)
        second = run_oracle(self.split, self.root / "second", anchor_limit=1, seed=8)
        self.assertEqual(first, second)
        self.assertEqual(first["sampled_anchors"], 1)
        self.assertLessEqual(first["positive_pairs"], 2)

    def test_missing_positive_target_fails_without_published_report(self):
        self.write("source3", self.columns, [])
        with self.assertRaisesRegex(ValueError, "Missing positive targets"):
            run_oracle(self.split, self.output)
        self.assertFalse((self.output / "positive_oracle.json").exists())

    def test_missing_ground_truth_fails(self):
        self.write("ground_truth", ("source1_entity_id", "matched_entity_ids"), [("S1-1", "S2-10")])
        with self.assertRaisesRegex(ValueError, "lacks ground truth"):
            run_oracle(self.split, self.output)

    def test_nontraining_split_and_bad_limits_rejected(self):
        for name in ("validation", "calibration", "test"):
            with self.assertRaisesRegex(ValueError, "training split"):
                run_oracle(self.root / name, self.output)
        with self.assertRaises(ValueError):
            run_oracle(self.split, self.output, anchor_limit=0)


if __name__ == "__main__":
    unittest.main()
