import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ber.candidates import BRANCHES, BRANCH_BITS, build_candidates, decode_target_code, encode_target_id


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queries = [
            {"entity_id": "S1-1", "business_name": "Alpha Ltd", "business_address": "12 Rue", "country": "France"},
            {"entity_id": "S1-2", "business_name": "Singleton", "business_address": "No Number", "country": "US"},
        ]
        self.targets = {
            2: [("S2-88", "Distant", "444 Nowhere", "US"), ("S2-11", "Alpha Ltd", "12 Rue", "France"),
                ("S2-3", "ALPHA LTD", "33 Elsewhere", "France"), ("S2-20", "Different", "12 Rue", "France"),
                ("S2-77", "Hidden Positive", "44 Lost Road", "France")],
            3: [("S3-1", "Alpha Ltd", "12 Rue", "France"), ("S3-15", "Other", "No Number", "US")],
        }
        self.files = {source: self.root / f"source{source}.tsv" for source in (2, 3)}
        for source in (2, 3):
            self.write_targets(source)
        self.results = {source: {} for source in (2, 3)}
        for source in (2, 3):
            for branch in BRANCHES[:3]:
                self.results[source][branch] = {
                    "query_ids": np.array([q["entity_id"] for q in self.queries]),
                    "target_ids": np.full((2, 2), -1, dtype=np.int64),
                    "scores": np.zeros((2, 2), dtype=np.float32), "target_source": f"S{source}",
                }
        self.put(2, "name_char", 0, [88, 11], [0.7, 0.6])
        self.put(2, "name_word", 0, [3], [0.9])
        self.put(2, "address_char", 0, [20], [1.0])
        self.put(3, "name_char", 0, [1], [0.95])
        self.put(2, "name_char", 1, [88], [0.2])
        self.truth = {"S1-1": {"S2-11", "S3-1", "S2-77"}, "S1-2": set()}

    def write_targets(self, source):
        with self.files[source].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(("entity_id", "business_name", "business_address", "country"))
            writer.writerows(self.targets[source])

    def put(self, source, branch, query, ids, scores):
        self.results[source][branch]["target_ids"][query, :len(ids)] = ids
        self.results[source][branch]["scores"][query, :len(ids)] = scores

    def build(self, folder="out", **kwargs):
        return build_candidates(self.queries, self.files, self.results, self.root / folder, **kwargs)

    def test_union_order_provenance_labels_and_ceiling(self):
        result = self.build(truth=self.truth)
        paths = result["paths"]
        anchors, targets, masks, scores, ranks = [np.load(paths[key]) for key in ("anchors", "target_codes", "masks", "scores", "ranks")]
        self.assertEqual(anchors.tolist(), [0, 0, 0, 0, 0, 1])
        self.assertEqual([decode_target_code(x) for x in targets], ["S2-3", "S2-11", "S2-20", "S2-88", "S3-1", "S2-88"])
        self.assertEqual(masks[1], BRANCH_BITS["name_char"] | BRANCH_BITS["name_exact"] | BRANCH_BITS["number_block"])
        self.assertEqual(ranks[1].tolist(), [2, 0, 0])
        self.assertAlmostEqual(scores[1, 0], 0.6)
        self.assertEqual(np.load(paths["labels"]).tolist(), [0, 1, 0, 0, 1, 0])
        self.assertEqual(np.load(paths["truth_counts"]).tolist(), [3, 0])
        d = result["stats"]["label_diagnostics"]
        self.assertAlmostEqual(d["overall"]["link_recall"], 2 / 3)
        self.assertAlmostEqual(d["overall"]["oracle_macro_f0_5"], (2.5 / 2.75 + 1) / 2)
        self.assertEqual(d["by_match_count"]["0"]["oracle_macro_f0_5"], 1)
        self.assertEqual(d["standalone_branch_recall"]["name_char"]["recalled_links"], 2)
        self.assertEqual(d["incremental_branch_recall"]["name_exact"]["new_positive_links"], 0)
        self.assertEqual(anchors.dtype, np.uint32)
        self.assertEqual(targets.dtype, np.uint64)
        self.assertEqual(masks.dtype, np.uint16)
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(ranks.dtype, np.uint16)

    def test_exact_cap_combines_sources_and_is_numeric_deterministic(self):
        # Remove sparse results so only the capped exact union is visible.
        for source in (2, 3):
            for branch in BRANCHES[:3]:
                self.results[source][branch]["target_ids"][:] = -1
                self.results[source][branch]["scores"][:] = 0
        first = self.build(exact_cap=2)
        self.assertEqual(np.load(first["paths"]["target_codes"]).tolist(), [encode_target_id("S2-3"), encode_target_id("S2-11")])
        self.assertEqual(first["stats"]["exact_overflow_queries"], 1)
        self.assertEqual(first["stats"]["exact_discarded_pairs"], 2)
        for source in (2, 3):
            self.targets[source].reverse()
            self.write_targets(source)
        second = self.build("second", exact_cap=2)
        for key in ("anchors", "target_codes", "masks", "scores", "ranks"):
            np.testing.assert_array_equal(np.load(first["paths"][key]), np.load(second["paths"][key]))

    def test_ground_truth_does_not_change_candidates(self):
        unlabeled = self.build("unlabeled")
        labeled = self.build("labeled", truth=self.truth)
        self.assertNotIn("labels", unlabeled["paths"])
        for key in ("anchors", "target_codes", "masks", "scores", "ranks"):
            np.testing.assert_array_equal(np.load(unlabeled["paths"][key]), np.load(labeled["paths"][key]))
        # Address without a number is never a rescue.
        self.assertNotIn(encode_target_id("S3-15"), np.load(unlabeled["paths"]["target_codes"]))

    def test_resume_stale_and_corruption_guards(self):
        first = self.build(truth=self.truth)
        with patch("ber.candidates._rows", side_effect=AssertionError("must reuse complete artifacts")):
            self.assertEqual(self.build(truth=self.truth), first)
        with self.assertRaisesRegex(ValueError, "inputs/config changed"):
            self.build(truth=self.truth, exact_cap=1)
        Path(first["paths"]["masks"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "artifact corruption"):
            self.build(truth=self.truth)

    def test_missing_retrieved_target_rejected(self):
        self.results[2]["name_char"]["target_ids"][0, 0] = 999
        with self.assertRaisesRegex(ValueError, "missing Source 2 targets"):
            self.build()

    def test_misaligned_queries_and_missing_truth_rejected(self):
        self.results[3]["name_char"]["query_ids"] = np.array(["S1-2", "S1-1"])
        with self.assertRaisesRegex(ValueError, "query order"):
            self.build()
        self.results[3]["name_char"]["query_ids"] = np.array(["S1-1", "S1-2"])
        with self.assertRaisesRegex(ValueError, "every query"):
            self.build(truth={"S1-1": set()})

    def test_id_codec_rejects_ambiguous_encodings(self):
        self.assertEqual(decode_target_code(encode_target_id("S3-0")), "S3-0")
        for invalid in ("S1-1", "S2-01", "S3-1000000000000", "S2-X"):
            with self.assertRaises(ValueError):
                encode_target_id(invalid)


if __name__ == "__main__":
    unittest.main()
