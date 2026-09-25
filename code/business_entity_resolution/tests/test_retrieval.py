import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ber.retrieval import BRANCHES, build_vectorizers, retrieve_sparse
from ber.text import normalize_text


def write_tsv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["entity_id", "business_name", "business_address", "country"], delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def record(entity_id, name, address="", country="US"):
    return dict(entity_id=entity_id, business_name=name, business_address=address, country=country)


class RetrievalTests(unittest.TestCase):
    def fixtures(self, root):
        train = [record(f"S1-{i + 1}", name, address) for i, (name, address) in enumerate([
            ("Acme Labs", "12 Pine Street"), ("Acme Clinic", "14 Oak Road"),
            ("Café Modern", "88 Lake Lane"), ("Brahma Market", "8 River Road"),
            ("नमस्ते दुकान", "दिल्ली 22"), ("Alpha Consulting", "23 Other Road"),
        ])]
        path = root / "train" / "source1.tsv"
        write_tsv(path, train)
        vectors = build_vectorizers([path], root / "vectors.joblib", sample_limit=100, min_df=1, max_df=1.0)
        return train, vectors

    def test_chunked_retrieval_matches_dense_all_sources_and_empty_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, vectors = self.fixtures(root)
            queries = [train[0], train[2], train[4], record("S1-99", "", "")]
            for source in (1, 2, 3):
                targets = [record(f"S{source}-{i}", name, address) for i, name, address in [
                    (15, "Acme Labs", "12 Pine Street"), (2, "Acme Clinic", "14 Oak Road"),
                    (30, "Café Modern", "88 Lake Lane"), (4, "नमस्ते दुकान", "दिल्ली 22"),
                    (17, "", ""), (1, "Acme Labs", "12 Pine Street"),
                ]]
                target_path = root / f"source{source}.tsv"
                write_tsv(target_path, targets)
                result = retrieve_sparse(queries, target_path, vectors, root / f"out{source}", k=3, threads=1, query_batch=2, target_batch=2)
                numeric_ids = np.asarray([int(row["entity_id"].split("-")[1]) for row in targets])
                for branch in BRANCHES:
                    field = "business_address" if branch == "address_char" else "business_name"
                    q = vectors[branch].transform([normalize_text(row[field]) for row in queries])
                    t = vectors[branch].transform([normalize_text(row[field]) for row in targets])
                    dense = (q @ t.T).toarray()
                    for index, similarities in enumerate(dense):
                        valid = np.flatnonzero(similarities > 0)
                        order = valid[np.lexsort((numeric_ids[valid], -similarities[valid]))][:3]
                        expected_ids = np.full(3, -1, dtype=np.int64)
                        expected_scores = np.zeros(3, dtype=np.float32)
                        expected_ids[:len(order)] = numeric_ids[order]
                        expected_scores[:len(order)] = similarities[order]
                        np.testing.assert_array_equal(result[branch]["target_ids"][index], expected_ids)
                        np.testing.assert_allclose(result[branch]["scores"][index], expected_scores, rtol=1e-6, atol=1e-7)
                    self.assertEqual(result[branch]["target_source"], f"S{source}")

    def test_equal_score_cutoff_prefers_smallest_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, vectors = self.fixtures(root)
            targets = [record(f"S2-{i}", "Acme Labs", "12 Pine Street") for i in (22, 9, 14, 1, 6, 4)]
            write_tsv(root / "target.tsv", targets)
            result = retrieve_sparse([train[0]], root / "target.tsv", vectors, root / "result", k=2, threads=1, target_batch=100)
            for branch in BRANCHES:
                np.testing.assert_array_equal(result[branch]["target_ids"], [[1, 4]])

    def test_resume_after_interruption_and_reject_changed_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, vectors = self.fixtures(root)
            write_tsv(root / "target.tsv", [dict(row, entity_id=row["entity_id"].replace("S1", "S2")) for row in train])
            def interrupt(event):
                if event["stage"] == "retrieval_shard_complete":
                    raise RuntimeError("simulated interruption")
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                retrieve_sparse(train[:2], root / "target.tsv", vectors, root / "out", k=2, threads=1, target_batch=2, progress=interrupt)
            resumed = retrieve_sparse(train[:2], root / "target.tsv", vectors, root / "out", k=2, threads=1, target_batch=2)
            fresh = retrieve_sparse(train[:2], root / "target.tsv", vectors, root / "fresh", k=2, threads=1, target_batch=2)
            for branch in BRANCHES:
                np.testing.assert_array_equal(resumed[branch]["target_ids"], fresh[branch]["target_ids"])
                np.testing.assert_array_equal(resumed[branch]["scores"], fresh[branch]["scores"])
            with self.assertRaisesRegex(ValueError, "changed"):
                retrieve_sparse(train[:1], root / "target.tsv", vectors, root / "out", k=2, threads=1, target_batch=2)
            cache = next((root / "out" / "target_cache").glob("*/000000.ids.npy"))
            cache.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "corruption"):
                retrieve_sparse(train[:2], root / "target.tsv", vectors, root / "out", k=2, threads=1, target_batch=2)

    def test_training_sample_order_invariant_and_heldout_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, _ = self.fixtures(root)
            write_tsv(root / "reverse.tsv", list(reversed(train)))
            first = build_vectorizers([root / "train/source1.tsv"], root / "a.joblib", sample_limit=4, min_df=1, max_df=1.0)
            second = build_vectorizers([root / "reverse.tsv"], root / "b.joblib", sample_limit=4, min_df=1, max_df=1.0)
            for branch in BRANCHES:
                self.assertEqual(first[branch].vocabulary_, second[branch].vocabulary_)
                np.testing.assert_array_equal(first[branch].idf_, second[branch].idf_)
            metadata = json.loads((root / "a.joblib.manifest.json").read_text())
            self.assertEqual(metadata["sample_rows"], 4)
            write_tsv(root / "validation/source1.tsv", train)
            with self.assertRaisesRegex(ValueError, "held-out"):
                build_vectorizers([root / "validation/source1.tsv"], root / "bad.joblib")

    def test_empty_target_and_empty_query_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, vectors = self.fixtures(root)
            write_tsv(root / "empty.tsv", [])
            result = retrieve_sparse(train[:1], root / "empty.tsv", vectors, root / "empty_result", k=2, threads=1)
            for branch in BRANCHES:
                np.testing.assert_array_equal(result[branch]["target_ids"], [[-1, -1]])
                np.testing.assert_array_equal(result[branch]["scores"], [[0, 0]])
            write_tsv(root / "blank.tsv", [record("S2-1", "", "")])
            result = retrieve_sparse([], root / "blank.tsv", vectors, root / "no_queries", k=2, threads=1)
            for branch in BRANCHES:
                self.assertEqual(result[branch]["target_ids"].shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
