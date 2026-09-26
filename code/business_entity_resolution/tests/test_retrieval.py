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


def test_distinctive_query_keeps_rare_fragments_and_empty_rows():
    from scipy import sparse
    from ber.retrieval import distinctive_query_matrix
    matrix=sparse.csr_matrix(np.array([[.9,.2,.3,.4],[0,0,0,0]],dtype=np.float32))
    result=distinctive_query_matrix(matrix,np.array([1.,3.,3.,2.]),2)
    assert result[0].indices.tolist()==[1,2]
    assert result[1].nnz==0
    assert np.isclose(float(result[0].multiply(result[0]).sum()),1.)
    assert matrix.nnz==4


def test_below_global_cutoff_ties_do_not_change_merged_results():
    from scipy import sparse
    from ber.retrieval import _multiply_topk, _merge_topk
    q=sparse.csr_matrix([[1.,0.]],dtype=np.float32)
    t=sparse.csr_matrix([[.2,0.]]*50,dtype=np.float32)
    ids=np.arange(50,dtype=np.int64)
    full=next(_multiply_topk(q,t,t.T.tocsr(),ids,2,1))
    fast=next(_multiply_topk(q,t,t.T.tocsr(),ids,2,1,np.array([.8])))
    a=_merge_topk(np.array([100,101]),np.array([.9,.8]),*full,2)
    b=_merge_topk(np.array([100,101]),np.array([.9,.8]),*fast,2)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])


def test_parallel_cache_preserves_retrieval(tmp_path):
    helper=RetrievalTests();train,vectors=helper.fixtures(tmp_path)
    path=tmp_path/'targets.tsv';write_tsv(path,[dict(r,entity_id=r['entity_id'].replace('S1','S2')) for r in train])
    a=retrieve_sparse(train,path,vectors,tmp_path/'serial',k=2,threads=1,target_batch=3)
    b=retrieve_sparse(train,path,vectors,tmp_path/'parallel',k=2,threads=1,target_batch=3,cache_workers=2)
    for branch in BRANCHES:
        np.testing.assert_array_equal(a[branch]['target_ids'],b[branch]['target_ids'])
        np.testing.assert_array_equal(a[branch]['scores'],b[branch]['scores'])


def test_approximate_pool_reranks_with_actual_full_cosine(tmp_path):
    helper=RetrievalTests();train,vectors=helper.fixtures(tmp_path)
    targets=[dict(r,entity_id=r['entity_id'].replace('S1','S2')) for r in train]
    path=tmp_path/'targets.tsv';write_tsv(path,targets)
    result=retrieve_sparse(train,path,vectors,tmp_path/'reranked',k=2,threads=1,target_batch=3,query_feature_limit=4,rerank_pool=5)
    for branch in BRANCHES:
        field='business_address' if branch=='address_char' else 'business_name'
        q=vectors[branch].transform([normalize_text(r[field]) for r in train]);t=vectors[branch].transform([normalize_text(r[field]) for r in targets]);expected=(q@t.T).toarray()
        for i in range(len(train)):
            for code,score in zip(result[branch]['target_ids'][i],result[branch]['scores'][i]):
                if code>=0:assert np.isclose(score,expected[i,code-1],atol=1e-6)
