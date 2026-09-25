"""Feature artifacts preserve candidate order, raw targets, and split independence."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.pair_dataset import BRANCHES, PAIR_FEATURE_NAMES, TARGET_FACTOR, build_pair_dataset
from ber.text import normalize_text


class PairDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidates = self.root / "candidates"
        self.candidates.mkdir()
        self.queries = [
            {"entity_id": "S1-1", "business_name": "École Alpha", "business_address": "12 Rue", "country": "France"},
            {"entity_id": "S1-2", "business_name": "nan", "business_address": "", "country": "US"},
            {"entity_id": "S1-3", "business_name": "No Candidates", "business_address": "9 Main", "country": "India"},
        ]
        with (self.candidates / "queries.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.queries:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.target_files = {source: self.root / f"source{source}.tsv" for source in (2, 3)}
        self.write_source(2, [("S2-1", "École Alpha", "12 Rue", "France"),
                              ("S2-9", "Not Retrieved", "111 Absent", "US")])
        self.write_source(3, [("S3-1", "Alpha", "13 Rue", "France"), ("S3-2", "nan", "", "US")])
        self.arrays = {
            "anchors": np.array([0, 0, 1], dtype=np.uint32),
            "target_codes": np.array([2 * TARGET_FACTOR + 1, 3 * TARGET_FACTOR + 1, 3 * TARGET_FACTOR + 2], dtype=np.uint64),
            "masks": np.array([8, 1, 8], dtype=np.uint16),
            "scores": np.array([[0, 0, 0], [0.7, 0, 0], [0, 0, 0]], dtype=np.float32),
            "ranks": np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint16),
        }
        for name, values in self.arrays.items():
            np.save(self.candidates / f"{name}.npy", values)
        np.save(self.candidates / "labels.npy", np.array([1, 0, 1], dtype=np.uint8))
        np.save(self.candidates / "truth_counts.npy", np.array([2, 1, 0], dtype=np.uint16))
        corpus = [normalize_text(x) for x in ["École Alpha", "Alpha", "nan", "12 Rue", "13 Rue", "other text"]]
        self.vectorizers = {
            branch: TfidfVectorizer(analyzer="word" if branch == "name_word" else "char", ngram_range=(1, 2),
                                    lowercase=False, dtype=np.float32).fit(corpus)
            for branch in BRANCHES
        }

    def tearDown(self):
        self.temporary.cleanup()

    def write_source(self, source, rows):
        with self.target_files[source].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerows(rows)

    def build(self, output="features", **kwargs):
        return build_pair_dataset(self.candidates, self.target_files, self.vectorizers,
                                  self.root / output, batch_size=2, **kwargs)

    def test_actual_cosines_and_features_follow_exact_candidate_order(self):
        metadata = self.build()
        X = np.load(metadata["X_path"], mmap_mode="r")
        self.assertTrue(metadata["complete"])
        self.assertEqual(X.shape, (3, len(PAIR_FEATURE_NAMES)))
        self.assertEqual(X.dtype, np.float32)
        features = {name: i for i, name in enumerate(metadata["feature_names"])}
        self.assertEqual(metadata["target_lookup"]["selected_unique_targets"], 3)
        self.assertEqual(X[0, features["right_source2"]], 1)
        self.assertEqual(X[1, features["right_source3"]], 1)
        self.assertEqual(X[0, features["retrieval_name_exact"]], 1)
        self.assertEqual(X[0, features["retrieval_name_char"]], 0)
        # Exact-only candidate has genuine cosine, despite its absent retrieval score.
        self.assertEqual(X[0, features["retrieval_name_char_score"]], 0)
        self.assertAlmostEqual(X[0, features["actual_name_char_cosine"]], 1, places=6)
        self.assertGreater(X[1, features["actual_name_word_cosine"]], 0)
        self.assertTrue(np.all((X[:, -3:] >= 0) & (X[:, -3:] <= 1)))
        self.assertEqual(X[2, features["actual_address_char_cosine"]], 0)
        self.assertEqual(X[2, features["name_right_missing"]], 0)
        self.assertAlmostEqual(X[2, features["actual_name_char_cosine"]], 1, places=6)
        self.assertAlmostEqual(X[0, features["retrieval_candidate_count_log1p"]], np.log1p(2), places=6)

    def test_labels_do_not_influence_features_or_resume(self):
        first = self.build()
        before = np.load(first["X_path"]).copy()
        (self.candidates / "labels.npy").write_bytes(b"invalid numpy labels that must never be read")
        (self.candidates / "truth_counts.npy").write_bytes(b"also forbidden")
        resumed = self.build()
        np.testing.assert_array_equal(before, np.load(resumed["X_path"]))
        rebuilt = self.build(output="separate")
        np.testing.assert_array_equal(before, np.load(rebuilt["X_path"]))

    def test_interrupted_batch_resumes_and_matches_uninterrupted_build(self):
        def interrupt(event):
            raise RuntimeError("simulated interruption after committed batch")

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.build(progress=interrupt)
        checkpoint = json.loads((self.root / "features" / "metadata.json").read_text())
        self.assertEqual(checkpoint["rows_completed"], 2)
        self.assertFalse(checkpoint["complete"])
        resumed = self.build()
        reference = self.build(output="reference")
        np.testing.assert_array_equal(np.load(resumed["X_path"]), np.load(reference["X_path"]))

    def test_resume_rejects_changed_inputs_and_corrupted_completed_feature(self):
        metadata = self.build()
        scores = self.arrays["scores"].copy()
        scores[0, 0] = 0.4
        np.save(self.candidates / "scores.npy", scores)
        with self.assertRaisesRegex(ValueError, "inputs/configuration changed"):
            self.build()
        np.save(self.candidates / "scores.npy", self.arrays["scores"])
        X = np.load(metadata["X_path"], mmap_mode="r+")
        X[0, 0] = -100
        X.flush()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.build()

    def test_missing_target_and_duplicate_candidate_rejected(self):
        self.write_source(3, [("S3-1", "Alpha", "13 Rue", "France")])
        with self.assertRaisesRegex(ValueError, "missing"):
            self.build()
        duplicates = self.arrays["target_codes"].copy()
        duplicates[1] = duplicates[0]
        np.save(self.candidates / "target_codes.npy", duplicates)
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            self.build(output="duplicate")

    def test_no_candidates_produces_empty_feature_artifact(self):
        for name, values in self.arrays.items():
            np.save(self.candidates / f"{name}.npy", values[:0])
        metadata = self.build()
        self.assertTrue(metadata["complete"])
        self.assertEqual(np.load(metadata["X_path"]).shape, (0, len(PAIR_FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
