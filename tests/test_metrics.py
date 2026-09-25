"""Mathematical contracts, zero-match semantics, and threshold sweep equivalence."""

import math
import random
import unittest

from ber.metrics import (
    entity_f0_5,
    iter_threshold_array_metrics,
    select_threshold,
    select_threshold_arrays,
    summarize_entities,
    sweep_thresholds,
)


class EntityMetricTests(unittest.TestCase):
    def test_perfect_one_match(self):
        self.assertEqual(entity_f0_5({"S2-1"}, {"S2-1"}), 1.0)

    def test_perfect_multiple_matches(self):
        self.assertEqual(entity_f0_5({"S2-1", "S3-2"}, {"S3-2", "S2-1"}), 1.0)

    def test_one_correct_one_false_positive(self):
        self.assertAlmostEqual(entity_f0_5({"S2-1"}, {"S2-1", "S3-2"}), 5 / 9)

    def test_one_correct_one_false_negative(self):
        self.assertAlmostEqual(entity_f0_5({"S2-1", "S3-2"}, {"S2-1"}), 5 / 6)

    def test_all_matches_missed(self):
        self.assertEqual(entity_f0_5({"S2-1", "S3-2"}, set()), 0.0)

    def test_correct_singleton(self):
        self.assertEqual(entity_f0_5(set(), set()), 1.0)

    def test_false_singleton_merge(self):
        self.assertEqual(entity_f0_5(set(), {"S2-1"}), 0.0)

    def test_duplicate_predictions_and_truth_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            entity_f0_5({"S2-1"}, ["S2-1", "S2-1"])
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            entity_f0_5(["S2-1", "S2-1"], {"S2-1"})
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            summarize_entities({"S1-1": {"S2-1"}}, {"S1-1": ["S2-1", "S2-1"]})

    def test_official_numerical_example(self):
        self.assertAlmostEqual(
            entity_f0_5({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}),
            5 / 7,
        )

    def test_macro_is_per_entity_not_micro(self):
        truth = {"a": {"x"}, "b": set(), "c": {"z", "q"}}
        result = summarize_entities(truth, {"a": {"x", "wrong"}})
        self.assertAlmostEqual(result["macro_f0_5"], (5 / 9 + 1 + 0) / 3)
        self.assertEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertNotEqual(result["macro_f0_5"], result["micro_f0_5"])
        self.assertEqual(result["false_negatives"], 2)
        self.assertEqual(result["singleton_accuracy"], 1.0)
        self.assertEqual(result["n_entities"], 3)

    def test_zero_denominator_conventions(self):
        result = summarize_entities({"a": set()}, {})
        self.assertEqual(result["macro_f0_5"], 1.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["micro_f0_5"], 0.0)
        self.assertIsNone(summarize_entities({"a": {"x"}}, {})["singleton_accuracy"])

    def test_invalid_universe_or_id_input_rejected(self):
        with self.assertRaises(ValueError):
            summarize_entities({}, {})
        with self.assertRaisesRegex(ValueError, "unknown S1"):
            summarize_entities({"a": set()}, {"not-a": set()})
        with self.assertRaises(ValueError):
            entity_f0_5("S2-1", ["S2-1"])


class ThresholdTests(unittest.TestCase):
    def test_thresholds_match_direct_scoring_with_ties_and_missing_candidates(self):
        truth = {"a": {"x", "unretrieved"}, "b": set(), "c": {"y"}, "d": set()}
        pairs = [("a", "x", 0.9), ("a", "wrong", 0.5), ("b", "wrong", 0.5)]
        thresholds = [math.inf, 0.9, 0.7, 0.5, 0.0, -math.inf]
        results = sweep_thresholds(truth, pairs, thresholds)
        for row in results:
            prediction = {s1: set() for s1 in truth}
            for s1, target, score in pairs:
                if score >= row["threshold"]:
                    prediction[s1].add(target)
            expected = summarize_entities(truth, prediction)
            for key, value in expected.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(row[key], value, places=12, msg=key)
                else:
                    self.assertEqual(row[key], value, key)
        self.assertEqual(select_threshold(truth, pairs, thresholds)["threshold"], 0.9)
        self.assertEqual(results[0]["macro_f0_5"], 0.5)
        self.assertEqual(results[-1]["false_negatives"], 2)

    def test_automatic_thresholds_include_predict_nothing(self):
        result = select_threshold({"a": set()}, [("a", "x", 0.99)])
        self.assertTrue(math.isinf(result["threshold"]))
        self.assertEqual(result["macro_f0_5"], 1.0)
        rows = sweep_thresholds({"a": {"x"}}, [], None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["macro_f0_5"], 0.0)

    def test_duplicate_candidates_unknown_anchors_nonfinite_scores_rejected(self):
        for pairs in (
            [("a", "x", 0.8), ("a", "x", 0.9)],
            [("unknown", "x", 0.8)],
            [("a", "x", math.nan)],
            [("a", "x", math.inf)],
        ):
            with self.assertRaises(ValueError):
                sweep_thresholds({"a": {"x"}}, pairs)
        with self.assertRaises(ValueError):
            sweep_thresholds({"a": {"x"}}, [], [math.nan])
        with self.assertRaises(ValueError):
            select_threshold({"a": {"x"}}, [], [])

    def test_seeded_random_sweep_matches_direct_metric(self):
        rng = random.Random(42)
        truth = {f"s{i}": {f"t{j}" for j in range(8) if rng.random() < 0.2} for i in range(50)}
        pairs = [(s1, f"t{j}", rng.choice([0.2, 0.5, 0.8]))
                 for s1 in truth for j in range(10) if rng.random() < 0.4]
        for row in sweep_thresholds(truth, pairs):
            predicted = {s1: set() for s1 in truth}
            for s1, target, score in pairs:
                if score >= row["threshold"]:
                    predicted[s1].add(target)
            self.assertAlmostEqual(row["macro_f0_5"], summarize_entities(truth, predicted)["macro_f0_5"], places=12)


class ArrayThresholdTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Optional NumPy threshold API requires NumPy")
        self.np = np

    def test_array_sweep_matches_id_sweep(self):
        truth = {"a": {"x", "missing"}, "b": set(), "c": {"y"}, "d": set()}
        pairs = [("a", "x", 0.9), ("a", "wrong", 0.5), ("b", "wrong", 0.5)]
        expected = sweep_thresholds(truth, pairs)
        arrays = ([2, 0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0.9, 0.5, 0.5])
        result = list(iter_threshold_array_metrics(*arrays))
        self.assertEqual(len(expected), len(result))
        for left, right in zip(expected, result):
            for key, value in left.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, right[key], places=12, msg=key)
                else:
                    self.assertEqual(value, right[key], key)
        self.assertEqual(select_threshold_arrays(*arrays)["threshold"], 0.9)

    def test_array_empty_candidates_include_all_anchors(self):
        empty_int = self.np.array([], dtype="int64")
        rows = list(iter_threshold_array_metrics([0, 2], empty_int, empty_int, empty_int, []))
        self.assertEqual(rows[0]["macro_f0_5"], 0.5)
        self.assertEqual(rows[0]["false_negatives"], 2)

    def test_array_invalid_contracts_rejected(self):
        bad_inputs = [
            ([1], [0, 0], [5, 5], [1, 1], [0.8, 0.9]),  # duplicate pair
            ([0], [0], [5], [1], [0.8]),  # impossible positive count
            ([1], [1], [5], [1], [0.8]),  # unknown anchor
            ([1], [0], [5], [2], [0.8]),  # label not binary
            ([1], [0], [5], [1], [math.nan]),
            ([1], [0], [5], [1], []),
            ([1.5], [0], [5], [1], [0.8]),
        ]
        for args in bad_inputs:
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(iter_threshold_array_metrics(*args))


if __name__ == "__main__":
    unittest.main()
