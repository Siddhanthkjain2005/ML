"""Train a resumable, grouped-OOF graph refinement experiment.

Run from the project root with PYTHONPATH=code/business_entity_resolution/src.
No remote calls are made. Models and score checkpoints use native JSON/numpy
formats; labels are used only for fitting and the designated evaluation stage.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from ber.graph_features import FEATURE_NAMES, build_graph_features
from ber.metrics import iter_threshold_array_metrics
from ber.model import select_decision
from ber.pair_dataset import _array_digest, _digest, _write_json
from ber.pipeline import event, load_arrays


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def checkpoint(path, config):
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    if metadata["config"] != canonical(config):
        raise ValueError(f"Checkpoint inputs changed; use a new output directory: {path}")
    return metadata


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def hashes(directory, names):
    return {name: _digest(directory / name) for name in names}


def matrix(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    X = np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False)
    names = metadata["feature_names"]
    if X.ndim != 2 or X.dtype != np.float32 or X.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError(f"Invalid feature matrix: {directory}")
    checksum = _digest(directory / "X.npy")
    if metadata.get("complete") is False:
        raise ValueError(f"Feature matrix is incomplete: {directory}")
    expected = metadata.get("X_sha256", metadata.get("sha256"))
    if expected is not None and checksum != expected:
        raise ValueError(f"Feature matrix checksum mismatch: {directory}")
    return X, names, checksum


def fit_or_load(X, y, folder, params, config, train_indices=None):
    folder.mkdir(parents=True, exist_ok=True)
    mp, path = folder / "metadata.json", folder / "model.json"
    config = {**config, "parameters": params}
    metadata = checkpoint(mp, config)
    if metadata and metadata["complete"]:
        if _digest(path) != metadata["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path}")
        model = XGBClassifier()
        model.load_model(path)
        model.set_params(n_jobs=params["n_jobs"])
        return model, metadata
    _write_json(mp, {"config": config, "complete": False})
    started = time.monotonic()
    model = XGBClassifier(**params)
    XX = X if train_indices is None else X[train_indices]
    yy = y if train_indices is None else y[train_indices]
    if set(np.unique(yy).tolist()) != {0, 1}:
        raise ValueError("Every OOF training fold must contain both labels")
    with threadpool_limits(limits=params["n_jobs"]):
        model.fit(XX, yy)
    del XX, yy
    gc.collect()
    temporary = path.with_name("model.tmp.json")
    model.save_model(temporary)
    os.replace(temporary, path)
    metadata = {"config": config, "complete": True, "sha256": _digest(path),
                "seconds": time.monotonic() - started, "peak_rss_bytes": peak_bytes()}
    _write_json(mp, metadata)
    event(stage="graph_model_fit", output=str(folder), seconds=metadata["seconds"], peak_rss_bytes=peak_bytes())
    return model, metadata


def score_or_load(model, X, path, config, batch_size, indices=None):
    mp = path.with_suffix(".metadata.json")
    rows = len(X) if indices is None else len(indices)
    config = {**config, "rows": rows, "batch_size": batch_size}
    metadata = checkpoint(mp, config)
    if metadata and metadata["complete"]:
        if _digest(path) != metadata["sha256"]:
            raise ValueError(f"Score checksum mismatch: {path}")
        return np.load(path, mmap_mode="r", allow_pickle=False), metadata
    metadata = metadata or {"config": config, "complete": False, "rows_completed": 0, "batches": []}
    if path.exists():
        values = np.load(path, mmap_mode="r+", allow_pickle=False)
        if values.shape != (rows,) or values.dtype != np.float32:
            raise ValueError("Score checkpoint shape/dtype changed")
    else:
        if metadata["rows_completed"]:
            raise ValueError("Score checkpoint is missing")
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows,))
    position = 0
    for batch in metadata["batches"]:
        if batch["start"] != position or not position < batch["end"] <= rows:
            raise ValueError("Invalid score checkpoint boundaries")
        if _array_digest(values[position:batch["end"]]) != batch["sha256"]:
            raise ValueError("Score checkpoint batch corrupted")
        position = batch["end"]
    if position != metadata["rows_completed"]:
        raise ValueError("Score checkpoint rows do not match completed batches")
    for start in range(position, rows, batch_size):
        end = min(start + batch_size, rows)
        XX = X[start:end] if indices is None else X[indices[start:end]]
        scores = np.asarray(model.predict_proba(XX)[:, 1], dtype=np.float32)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid predicted probabilities")
        values[start:end] = scores
        values.flush()
        metadata["batches"].append({"start": start, "end": end, "sha256": _array_digest(scores)})
        metadata["rows_completed"] = end
        _write_json(mp, metadata)
    metadata.update(complete=True, sha256=_digest(path))
    _write_json(mp, metadata)
    return values, metadata


def concatenate(base, base_sha, graph_dir, output, base_names, batch_size):
    graph_meta = json.loads((graph_dir / "metadata.json").read_text())
    extra = np.load(graph_dir / "extraX.npy", mmap_mode="r", allow_pickle=False)
    config = {"base_sha256": base_sha, "extra_sha256": graph_meta["sha256"], "batch_size": batch_size}
    if _digest(graph_dir / "extraX.npy") != graph_meta["sha256"] or extra.shape[0] != base.shape[0]:
        raise ValueError("Graph feature matrix is inconsistent")
    output.mkdir(parents=True, exist_ok=True)
    path, mp = output / "X.npy", output / "metadata.json"
    metadata = checkpoint(mp, config)
    if metadata and metadata["complete"]:
        if _digest(path) != metadata["sha256"]:
            raise ValueError("Combined graph feature matrix corrupted")
        return np.load(path, mmap_mode="r", allow_pickle=False), metadata
    names = [*base_names, *graph_meta["feature_names"]]
    shape = (len(base), len(names))
    # Concatenation is cheap and deterministic, so an interrupted copy restarts.
    _write_json(mp, {"config": config, "complete": False})
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        X[start:end, :base.shape[1]] = base[start:end]
        X[start:end, base.shape[1]:] = extra[start:end]
    X.flush()
    metadata = {"config": config, "complete": True, "sha256": _digest(path), "shape": list(shape), "feature_names": names}
    _write_json(mp, metadata)
    return X, metadata


def seed_model_path(reference):
    selection_path = reference / "frozen_selection.json"
    selection = json.loads(selection_path.read_text())
    name = selection.get("name", selection.get("selected_model"))
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Frozen reference selection must name a model subdirectory")
    path = reference / name / "model.json"
    return path, {"name": name, "model_sha256": _digest(path), "selection_sha256": _digest(selection_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", default="experiments/pilot_v1")
    parser.add_argument("--feature-dir", default="features_reference_v1")
    parser.add_argument("--raw-feature-dir", default="features")
    parser.add_argument("--reference-model-dir", default="experiments/reference_refinement_v1")
    parser.add_argument("--output", default="experiments/graph_refinement_v1")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--trusted-threshold", type=float, default=.95)
    parser.add_argument("--oof-trees", type=int, default=500)
    parser.add_argument("--depth6-trees", type=int, default=700)
    parser.add_argument("--depth7-trees", type=int, default=1000)
    parser.add_argument("--memory-budget-gib", type=float, default=12.)
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.oof_trees, args.depth6_trees, args.depth7_trees) < 1:
        parser.error("Threads, batch size and tree counts must be positive")
    if not 0 < args.memory_budget_gib <= 12:
        parser.error("The memory budget must be greater than zero and at most 12 GiB")
    root, out = Path(args.pilot_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_path, seed_info = seed_model_path(Path(args.reference_model_dir))
    train_dir, cal_dir = root / "train", root / "calibration"
    train, names, train_sha = matrix(train_dir / args.feature_dir)
    calibration, cal_names, cal_sha = matrix(cal_dir / args.feature_dir)
    if names != cal_names:
        raise ValueError("Train/calibration feature columns differ")
    y = np.load(train_dir / "candidates/labels.npy", mmap_mode="r", allow_pickle=False)
    anchors = np.load(train_dir / "candidates/anchors.npy", mmap_mode="r", allow_pickle=False)
    if y.shape != (len(train),) or anchors.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Training labels/anchors do not match features")
    # Four dense copies cover the memmap pages, training subset, quantized input
    # and an additional conservative workspace allowance. A 2-GiB fixed reserve
    # covers histograms, imported runtimes, indices and bounded graph batches.
    estimated = int(len(train) * (len(names) + len(FEATURE_NAMES)) * 4 * 4 + calibration.nbytes + 2 * 2**30)
    if estimated > args.memory_budget_gib * 2**30:
        raise MemoryError(f"Conservative training estimate {estimated / 2**30:.2f} GiB exceeds configured budget")
    common = dict(tree_method="hist", objective="binary:logistic", eval_metric="logloss", n_jobs=args.threads,
                  random_state=42, subsample=.9, colsample_bytree=.9, verbosity=0)
    oof_params = dict(common, max_depth=7, n_estimators=args.oof_trees, learning_rate=.05, min_child_weight=5, reg_lambda=3.)
    variants = {
        "depth6": dict(common, max_depth=6, n_estimators=args.depth6_trees, learning_rate=.05, min_child_weight=5, reg_lambda=3.),
        "depth7": dict(common, max_depth=7, n_estimators=args.depth7_trees, learning_rate=.05, min_child_weight=5, reg_lambda=3.),
    }
    config = {
        "schema_version": 1, "arguments": vars(args), "seed_model": seed_info,
        "train_features_sha256": train_sha, "calibration_features_sha256": cal_sha,
        "train_candidates": hashes(train_dir / "candidates", ("queries.jsonl", "anchors.npy", "target_codes.npy", "labels.npy")),
        "calibration_candidates": hashes(cal_dir / "candidates", ("queries.jsonl", "anchors.npy", "target_codes.npy", "labels.npy", "truth_counts.npy")),
        "oof_parameters": oof_params, "variants": variants,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "xgboost", "threadpoolctl")},
        "script_sha256": _digest(Path(__file__)), "estimated_peak_bytes": estimated,
    }
    manifest_path = out / "manifest.json"
    manifest = checkpoint(manifest_path, config)
    if manifest is None:
        _write_json(manifest_path, {"config": config, "complete": False})
    event(stage="graph_preflight", train_pairs=len(train), features=len(names), estimated_peak_gib=estimated / 2**30)

    # Stable group assignments depend only on Source1 IDs, never pair labels.
    group_folds = []
    with (train_dir / "candidates/queries.jsonl").open() as handle:
        for line in handle:
            identity = json.loads(line)["entity_id"]
            group_folds.append(int.from_bytes(hashlib.blake2b(("graph-oof-42:" + identity).encode(), digest_size=8).digest(), "big") % 3)
    group_folds = np.asarray(group_folds, dtype=np.uint8)
    row_folds = group_folds[anchors]
    oof_dir = out / "oof"
    oof_dir.mkdir(exist_ok=True)
    folds_path = oof_dir / "anchor_folds.npy"
    if folds_path.exists():
        if not np.array_equal(np.load(folds_path, allow_pickle=False), group_folds):
            raise ValueError("Saved OOF group assignments changed")
    else:
        np.save(folds_path, group_folds)
    fold_info = []
    for fold in range(3):
        train_indices, heldout_indices = np.flatnonzero(row_folds != fold), np.flatnonzero(row_folds == fold)
        if not len(heldout_indices):
            raise ValueError("OOF fold is empty; use a larger training pilot")
        fold_config = {"train_features_sha256": train_sha, "labels_sha256": config["train_candidates"]["labels.npy"],
                       "fold_assignments_sha256": _digest(folds_path), "heldout_fold": fold, "fold_count": 3}
        model, meta = fit_or_load(train, y, oof_dir / f"fold{fold}", oof_params, fold_config, train_indices)
        _, score_meta = score_or_load(model, train, oof_dir / f"fold{fold}/scores.npy",
                                     {**fold_config, "model_sha256": meta["sha256"]}, args.batch_size, heldout_indices)
        fold_info.append({"fold": fold, "training_pairs": len(train_indices), "heldout_pairs": len(heldout_indices),
                          "model_sha256": meta["sha256"], "scores_sha256": score_meta["sha256"]})
        del model, train_indices, heldout_indices
        gc.collect()
    oof_path, oof_meta_path = oof_dir / "scores.npy", oof_dir / "scores.metadata.json"
    oof_config = {"folds": fold_info, "group_assignments_sha256": _digest(folds_path), "train_features_sha256": train_sha}
    oof_meta = checkpoint(oof_meta_path, oof_config)
    if not oof_meta or not oof_meta["complete"]:
        scores = np.lib.format.open_memmap(oof_path, mode="w+", dtype=np.float32, shape=(len(train),))
        for fold in range(3):
            scores[row_folds == fold] = np.load(oof_dir / f"fold{fold}/scores.npy", mmap_mode="r")
        scores.flush()
        oof_meta = {"config": oof_config, "complete": True, "sha256": _digest(oof_path), "score_role": "out_of_fold_train"}
        _write_json(oof_meta_path, oof_meta)
        del scores
    elif _digest(oof_path) != oof_meta["sha256"]:
        raise ValueError("Combined OOF scores corrupted")
    del row_folds, group_folds
    seed_model = XGBClassifier()
    seed_model.load_model(seed_path)
    seed_model.set_params(n_jobs=args.threads)
    if seed_model.n_features_in_ != len(names):
        raise ValueError("Frozen reference model feature count differs from the base matrix")
    _, cal_seed_meta = score_or_load(seed_model, calibration, out / "calibration_seed_scores.npy",
                                    {"seed": seed_info, "features_sha256": cal_sha, "role": "frozen_calibration"}, args.batch_size)
    build_graph_features(train_dir / "candidates", train_dir / args.raw_feature_dir, oof_path, out / "train_graph",
                         score_role="out_of_fold_train", trusted_threshold=args.trusted_threshold, batch_size=args.batch_size,
                         score_provenance=oof_config, progress=lambda p: event(**p))
    build_graph_features(cal_dir / "candidates", cal_dir / args.raw_feature_dir, out / "calibration_seed_scores.npy", out / "calibration_graph",
                         score_role="frozen_calibration", trusted_threshold=args.trusted_threshold, batch_size=args.batch_size,
                         score_provenance=seed_info, progress=lambda p: event(**p))
    X, xm = concatenate(train, train_sha, out / "train_graph", out / "train_features", names, args.batch_size)
    XC, cm = concatenate(calibration, cal_sha, out / "calibration_graph", out / "calibration_features", names, args.batch_size)
    cal = load_arrays(cal_dir / "candidates")
    decisions, models = {}, {}
    for name, params in variants.items():
        model, meta = fit_or_load(X, y, out / name, params, {"features_sha256": xm["sha256"], "labels_sha256": config["train_candidates"]["labels.npy"]})
        scores, sm = score_or_load(model, XC, out / name / "calibration_scores.npy",
                                   {"model_sha256": meta["sha256"], "features_sha256": cm["sha256"]}, args.batch_size)
        decision_config = {"scores_sha256": sm["sha256"], "candidates": config["calibration_candidates"]}
        decision_meta = checkpoint(out / name / "calibration.json", decision_config)
        if decision_meta is None:
            decision = select_decision(cal["truth_counts"], cal["anchors"], cal["target_codes"], cal["labels"], scores)
            decision_meta = {"config": decision_config, "decision": decision}
            _write_json(out / name / "calibration.json", decision_meta)
        decisions[name], models[name] = decision_meta["decision"], meta["sha256"]
        importances = sorted(zip(xm["feature_names"], model.feature_importances_.tolist()), key=lambda pair: -pair[1])
        _write_json(out / name / "feature_importance.json", [{"feature": f, "gain": g} for f, g in importances])
        event(stage="graph_calibration", name=name, **decisions[name])
        del model, scores
        gc.collect()
    best = max(decisions, key=lambda name: (decisions[name]["macro_f0_5"], decisions[name]["threshold"], name))
    selection = {"name": best, "model_sha256": models[best], "threshold": decisions[best]["threshold"],
                 "decisions": decisions, "basis": "calibration only", "seed_model": seed_info,
                 "feature_names": xm["feature_names"], "training_graph_scores": "three-fold grouped OOF"}
    frozen_path = out / "frozen_selection.json"
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != canonical(selection):
            raise ValueError("Frozen graph selection changed")
    else:
        _write_json(frozen_path, selection)
    # Validation is opened only after model and threshold selection are frozen.
    del X, XC, train, calibration, y, anchors, cal
    gc.collect()
    val_dir = root / "validation"
    validation, val_names, val_sha = matrix(val_dir / args.feature_dir)
    if val_names != names:
        raise ValueError("Validation feature columns differ from training")
    val_candidates = hashes(val_dir / "candidates", ("queries.jsonl", "anchors.npy", "target_codes.npy", "labels.npy", "truth_counts.npy"))
    result_config = {"selection_sha256": _digest(frozen_path), "features_sha256": val_sha, "candidates": val_candidates}
    previous = checkpoint(out / "result.json", result_config)
    if previous is not None:
        event(stage="graph_already_complete", selected_model=best, validation=previous["validation"])
        return previous
    score_or_load(seed_model, validation, out / "validation_seed_scores.npy",
                  {"seed": seed_info, "features_sha256": val_sha, "role": "frozen_validation"}, args.batch_size)
    build_graph_features(val_dir / "candidates", val_dir / args.raw_feature_dir, out / "validation_seed_scores.npy", out / "validation_graph",
                         score_role="frozen_validation", trusted_threshold=args.trusted_threshold, batch_size=args.batch_size,
                         score_provenance=seed_info, progress=lambda p: event(**p))
    XV, vm = concatenate(validation, val_sha, out / "validation_graph", out / "validation_features", names, args.batch_size)
    selected_model = XGBClassifier()
    selected_model.load_model(out / best / "model.json")
    selected_model.set_params(n_jobs=args.threads)
    scores, _ = score_or_load(selected_model, XV, out / best / "validation_scores.npy",
                              {"model_sha256": models[best], "features_sha256": vm["sha256"]}, args.batch_size)
    val = load_arrays(val_dir / "candidates")
    result = next(iter_threshold_array_metrics(val["truth_counts"], val["anchors"], val["target_codes"], val["labels"], scores, [selection["threshold"]]))
    output = {"config": result_config, "selected_model": best, "validation": result,
              "calibration": decisions[best], "peak_rss_bytes": peak_bytes(), "validation_evaluations": 1}
    _write_json(out / "result.json", output)
    _write_json(manifest_path, {"config": config, "complete": True, "result_sha256": _digest(out / "result.json")})
    event(stage="graph_refinement_complete", selected_model=best, validation=result, peak_rss_bytes=peak_bytes())
    return output


if __name__ == "__main__":
    main()
