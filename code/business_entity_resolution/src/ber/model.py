"""Reproducible lexical pair classifiers and calibration-only decision selection.

All supplied candidate pairs are used at their natural prevalence. This module
does not generate random negatives, resample labels, or infer split membership.
Callers must supply training, calibration, and validation identity groups that
are disjoint. The linear model is a diagnostic baseline; XGBoost is the final
Apache-2.0 model candidate. No pretrained model or network access is used.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .metrics import iter_threshold_array_metrics, select_threshold_arrays


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl")
    }}


def _peak_rss_bytes() -> int:
    """Process lifetime high-water RSS, not an isolated per-model allocation."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _write_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _check_matrix(X: Any) -> Any:
    if sparse.issparse(X):
        matrix = X.tocsr()
        values = matrix.data
    else:
        matrix = np.asarray(X)
        values = matrix
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("X must be a two-dimensional nonempty-feature matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("X must contain numeric features")
    if not np.isfinite(values).all():
        raise ValueError("X contains nonfinite features")
    return matrix


def _check_training(X: Any, y: Any) -> tuple[Any, np.ndarray]:
    matrix = _check_matrix(X)
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != matrix.shape[0]:
        raise ValueError("y must be one-dimensional with one label per feature row")
    if labels.dtype.kind not in "biu" or not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain binary integer labels")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Training candidates must contain both positive and negative labels")
    return matrix, labels.astype(np.int8, copy=False)


def fit_models(
    X: Any, y: Any, output_dir: str | Path, seed: int = 42, threads: int = 4,
    *, feature_names: list[str] | tuple[str, ...] | None = None,
    linear_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Fit linear and boosted baselines on every supplied training candidate.

    Small parameter overrides support controlled experiments and fast tests.
    Existing model output directories are never overwritten. A failed model
    leaves FAILED metadata and re-raises its exception so the experiment ledger
    can record failure. No validation/test examples enter model fitting.
    """
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    matrix, labels = _check_training(X, y)
    names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(matrix.shape[1])]
    if len(names) != matrix.shape[1] or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("feature_names must be unique nonempty names, one per matrix column")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("linear", "xgboost"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite model directory: {output / name}")
    linear_config = dict(C=1.0, max_iter=500, solver="lbfgs", random_state=seed, class_weight=None)
    linear_config.update(linear_params or {})
    xgb_config = dict(
        n_estimators=250, max_depth=5, learning_rate=0.08, min_child_weight=10,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0, tree_method="hist",
        objective="binary:logistic", eval_metric="logloss", random_state=seed,
        n_jobs=threads, verbosity=0,
    )
    xgb_config.update(xgb_params or {})
    models = {
        "linear": Pipeline([
            ("scaler", StandardScaler(with_mean=not sparse.issparse(matrix))),
            ("classifier", LogisticRegression(**linear_config)),
        ]),
        "xgboost": XGBClassifier(**xgb_config),
    }
    environment = _versions()
    paths = {}
    for name, model in models.items():
        folder = output / name
        folder.mkdir()
        metadata = {
            "status": "RUNNING", "model": name, "seed": seed, "threads": threads,
            "versions": environment, "training_candidate_pairs": len(labels),
            "positive_pairs": int(labels.sum()), "negative_pairs": int((labels == 0).sum()),
            "negative_sampling": "none; all supplied retrieval candidates",
            "feature_names": names,
            "parameters": linear_config if name == "linear" else xgb_config,
            "scaling": {"standard_scaler": True, "with_mean": not sparse.issparse(matrix)} if name == "linear" else None,
            "license_role": "BSD-3-Clause scikit-learn diagnostic baseline" if name == "linear" else "Apache-2.0 XGBoost final-model candidate",
            "process_peak_rss_before_bytes": _peak_rss_bytes(),
            "memory_note": "Process lifetime high-water RSS; not isolated model allocation.",
        }
        _write_json(folder / "metadata.json", metadata)
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=threads):
                warnings.simplefilter("always")
                model.fit(matrix, labels)
            metadata["warnings"] = [f"{warning.category.__name__}: {warning.message}" for warning in captured]
            model_path = folder / "model.joblib"
            joblib.dump(model, model_path, compress=3)
            if name == "xgboost":
                model.save_model(folder / "model.json")
                values = model.feature_importances_
                importance_kind = "XGBoost normalized gain"
                signed = None
            else:
                signed = model.named_steps["classifier"].coef_[0]
                values = np.abs(signed)
                importance_kind = "absolute coefficient after StandardScaler"
            importances = []
            for index in np.argsort(-values, kind="stable"):
                row = {"feature": names[int(index)], "importance": float(values[index])}
                if signed is not None:
                    row["signed_coefficient"] = float(signed[index])
                importances.append(row)
            _write_json(folder / "feature_importance.json", {"kind": importance_kind, "features": importances})
            metadata.update(status="COMPLETE", model_file="model.joblib",
                            xgboost_native_file="model.json" if name == "xgboost" else None)
            paths[name] = str(model_path)
        except Exception as exc:
            metadata.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            metadata["runtime_seconds"] = time.perf_counter() - started
            metadata["process_peak_rss_bytes"] = _peak_rss_bytes()
            _write_json(folder / "metadata.json", metadata)
    return paths


def load_model(model_or_path: Any) -> Any:
    """Load a locally generated trusted model artifact, or accept an estimator.

    Joblib is executable serialization: callers must not supply untrusted files.
    Native XGBoost JSON can also be loaded without Python object deserialization.
    """
    if isinstance(model_or_path, (str, Path)):
        path = Path(model_or_path)
        if path.suffix == ".json":
            model = XGBClassifier()
            model.load_model(path)
            return model
        return joblib.load(path)
    if not hasattr(model_or_path, "predict_proba"):
        raise ValueError("Expected a model path or estimator with predict_proba")
    return model_or_path


def predict_scores(model_or_path: Any, X: Any, batch_size: int = 50_000) -> np.ndarray:
    """Return positive-class probabilities in original candidate-row order."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    matrix = _check_matrix(X)
    model = load_model(model_or_path)
    classes = np.asarray(model.classes_)
    indices = np.flatnonzero(classes == 1)
    if len(indices) != 1:
        raise ValueError("Model must expose exactly one positive class labeled 1")
    if getattr(model, "n_features_in_", matrix.shape[1]) != matrix.shape[1]:
        raise ValueError("Feature count differs from the fitted model")
    result = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        probabilities = np.asarray(model.predict_proba(matrix[start:end]))
        if probabilities.ndim != 2 or probabilities.shape[0] != end - start or probabilities.shape[1] != len(classes):
            raise ValueError("Model returned malformed probabilities")
        result[start:end] = probabilities[:, indices[0]]
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Model returned nonfinite or out-of-range probabilities")
    return result


def _probabilities(scores: Any) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("scores must be one-dimensional finite probabilities in [0, 1]")
    return result


def select_decision(
    true_counts: Any, anchor_indices: Any, target_indices: Any, labels: Any,
    scores: Any, *, thresholds: Any = None,
) -> dict[str, Any]:
    """Choose one global threshold using calibration entities only.

    Default: coarse 0.05..0.99 (step .01), endpoints 0 and 1, and a finite
    predict-nothing threshold just above 1; refine ±.02 around the coarse winner
    at .001 resolution. Explicit thresholds skip refinement. Higher thresholds
    break macro-score ties. All anchors, including candidate misses, must appear
    in true_counts. The result is directly JSON-serializable.
    """
    values = _probabilities(scores)
    no_matches = float(np.nextafter(1.0, math.inf))
    if thresholds is None:
        grid = np.unique(np.concatenate(([0.0, 1.0, no_matches], np.arange(5, 100) / 100)))
    else:
        grid = np.asarray(list(thresholds), dtype=np.float64)
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
            raise ValueError("Explicit thresholds must be a nonempty finite sequence")
    arguments = (true_counts, anchor_indices, target_indices, labels, values)
    best = select_threshold_arrays(*arguments, thresholds=grid)
    if thresholds is None and best["threshold"] <= 1.0:
        lower = max(0, int(math.floor((best["threshold"] - 0.02) * 1000)))
        upper = min(1000, int(math.ceil((best["threshold"] + 0.02) * 1000)))
        refined = np.unique(np.concatenate((grid, np.arange(lower, upper + 1) / 1000)))
        best = select_threshold_arrays(*arguments, thresholds=refined)
        evaluated = len(refined)
    else:
        evaluated = len(grid)
    return {**best, "selection_partition": "calibration", "thresholds_evaluated": evaluated,
            "threshold_strategy": "explicit grid" if thresholds is not None else "coarse grid plus local .001 refinement"}


def evaluate_model(
    model_or_path: Any, X: Any, true_counts: Any, anchor_indices: Any,
    target_indices: Any, labels: Any, *, threshold: float, batch_size: int = 50_000,
) -> dict[str, Any]:
    """Evaluate a previously chosen threshold; never optimize on these labels."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    scores = predict_scores(model_or_path, X, batch_size=batch_size)
    return next(iter_threshold_array_metrics(
        true_counts, anchor_indices, target_indices, labels, scores, [threshold],
    ))
