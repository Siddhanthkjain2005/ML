"""Tiny deterministic fitting/persistence checks; no competition records used."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ber.model import evaluate_model, fit_models, predict_scores, select_decision


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        rng = np.random.default_rng(42)
        cls.X = rng.normal(size=(240, 3)).astype(np.float32)
        cls.y = (cls.X[:, 0] + 0.25 * cls.X[:, 1] > 0).astype(np.int8)
        cls.paths = fit_models(
            cls.X, cls.y, cls.root / "fit", seed=42, threads=1,
            feature_names=["first", "second", "third"],
            linear_params={"max_iter": 100},
            xgb_params={"n_estimators": 15, "max_depth": 2, "min_child_weight": 1},
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_models_train_save_reload_with_named_importances(self):
        self.assertEqual(set(self.paths), {"linear", "xgboost"})
        for name, path in self.paths.items():
            with self.subTest(model=name):
                scores = predict_scores(path, self.X, batch_size=31)
                self.assertEqual(scores.shape, (240,))
                self.assertGreater(float(((scores >= 0.5) == self.y).mean()), 0.9)
                folder = Path(path).parent
                metadata = json.loads((folder / "metadata.json").read_text())
                self.assertEqual(metadata["status"], "COMPLETE")
                self.assertEqual(metadata["training_candidate_pairs"], 240)
                self.assertEqual(metadata["positive_pairs"], int(self.y.sum()))
                self.assertGreater(metadata["runtime_seconds"], 0)
                self.assertIn("xgboost", metadata["versions"])
                importance = json.loads((folder / "feature_importance.json").read_text())
                self.assertEqual({row["feature"] for row in importance["features"]}, {"first", "second", "third"})

    def test_native_xgboost_reload_preserves_probabilities(self):
        path = Path(self.paths["xgboost"])
        np.testing.assert_array_equal(
            predict_scores(path, self.X), predict_scores(path.parent / "model.json", self.X),
        )

    def test_seeded_refit_is_reproducible(self):
        repeated = fit_models(
            self.X, self.y, self.root / "repeated", seed=42, threads=1,
            feature_names=["first", "second", "third"],
            linear_params={"max_iter": 100},
            xgb_params={"n_estimators": 15, "max_depth": 2, "min_child_weight": 1},
        )
        for name in self.paths:
            np.testing.assert_array_equal(
                predict_scores(self.paths[name], self.X), predict_scores(repeated[name], self.X),
            )

    def test_batched_scores_and_empty_candidate_matrix(self):
        for path in self.paths.values():
            np.testing.assert_allclose(predict_scores(path, self.X, 1), predict_scores(path, self.X, 1000), atol=1e-12)
            self.assertEqual(predict_scores(path, self.X[:0]).shape, (0,))

    def test_refuses_overwrite_and_invalid_training(self):
        with self.assertRaises(FileExistsError):
            fit_models(self.X, self.y, self.root / "fit")
        with self.assertRaises(ValueError):
            fit_models(self.X, np.zeros(len(self.y), dtype=np.int8), self.root / "bad")
        with self.assertRaises(ValueError):
            fit_models(self.X, self.y, self.root / "bad", feature_names=["wrong"])
        with self.assertRaises(ValueError):
            predict_scores(self.paths["linear"], self.X[:, :2])
        with self.assertRaises(ValueError):
            predict_scores(self.paths["linear"], np.full_like(self.X, np.nan))

    def test_fit_failure_has_persisted_failed_metadata(self):
        with mock.patch("ber.model.Pipeline.fit", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                fit_models(self.X, self.y, self.root / "failed")
        metadata = json.loads((self.root / "failed" / "linear" / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "FAILED")
        self.assertEqual(metadata["error_type"], "RuntimeError")
        self.assertGreaterEqual(metadata["runtime_seconds"], 0)

    def test_calibration_selection_keeps_singletons_and_missing_links(self):
        decision = select_decision(
            [2, 0, 1, 0], [0, 0, 1], [1, 2, 2], [1, 0, 0], [0.9, 0.5, 0.5],
        )
        self.assertEqual(decision["threshold"], 0.9)
        self.assertAlmostEqual(decision["macro_f0_5"], (5 / 6 + 1 + 0 + 1) / 4)
        self.assertEqual(decision["false_negatives"], 2)
        self.assertEqual(decision["selection_partition"], "calibration")
        json.dumps(decision, allow_nan=False)

    def test_predict_nothing_decision_is_finite_json(self):
        decision = select_decision([0], [0], [1], [0], [0.9])
        self.assertGreater(decision["threshold"], 1)
        self.assertEqual(decision["macro_f0_5"], 1.0)
        json.dumps(decision, allow_nan=False)

    def test_evaluation_uses_fixed_threshold_without_tuning(self):
        class FixtureModel:
            classes_ = np.array([0, 1])
            n_features_in_ = 1

            def predict_proba(self, X):
                return np.column_stack((1 - X[:, 0], X[:, 0]))

        result = evaluate_model(
            FixtureModel(), np.array([[0.9], [0.5], [0.5]]),
            [2, 0, 1, 0], [0, 0, 1], [1, 2, 2], [1, 0, 0], threshold=0.5,
        )
        self.assertEqual(result["threshold"], 0.5)
        self.assertEqual(result["predicted_links"], 3)
        self.assertEqual(result["correct_singletons"], 1)
        self.assertAlmostEqual(result["macro_f0_5"], (0.5 + 0 + 0 + 1) / 4)


if __name__ == "__main__":
    unittest.main()
