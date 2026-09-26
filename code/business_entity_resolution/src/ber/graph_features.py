"""Bounded, label-free consistency with independently scored target records.

Training scores MUST be grouped out-of-fold predictions: all pairs belonging to
one Source1 anchor must be held out together. Calibration, validation and test
scores must come from an already frozen scorer. This module cannot prove the
provenance of an arbitrary probability file, so the caller declares its role;
that declaration and the exact score checksum are saved with the artifact.

Each candidate is compared with at most three other high-scoring targets for
the same Source1. Its own target is never evidence for itself, even when it is
the only trusted candidate. Same-source targets are permitted. No labels, truth
counts, one-to-one assignment rule, or forced match decision is used here.
"""
from __future__ import annotations

from functools import lru_cache
import heapq
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Mapping

import numpy as np

from .advanced_features import EXTRA_NAMES, extra_pair_features
from .features import _edit_similarity
from .pair_dataset import TARGET_FACTOR, _array_digest, _digest, _lookup_records, _write_json

SCORE_ROLES = ("out_of_fold_train", "frozen_calibration", "frozen_validation", "frozen_inference")
_BASE_NAMES = ("exact", "folded_exact", "token_jaccard", "token_containment",
               "edit_similarity", "numbers_equal", "numbers_overlap", "numbers_jaccard")
_CONTEXT_NAMES = (
    "candidate_score", "support_count", "source2_support_count", "source3_support_count",
    "support_source_diversity", "same_source_support_count", "cross_source_support_count",
    "max_support_score", "mean_support_score", "score_margin",
)
FEATURE_NAMES = (
    *(f"graph_{name}" for name in _CONTEXT_NAMES),
    *(f"graph_support_max_{name}" for name in EXTRA_NAMES),
    *(f"graph_support_max_{field}_{name}" for field in ("name", "address") for name in _BASE_NAMES),
    "graph_support_max_country_equal",
)


def _base_comparison(left, right) -> list[float]:
    values = []
    for field in ("name", "address"):
        a, b = getattr(left, field), getattr(right, field)
        overlap = a.tokens & b.tokens
        union = a.tokens | b.tokens
        number_overlap = a.numbers & b.numbers
        number_union = a.numbers | b.numbers
        values.extend((
            float(bool(a.normalized) and a.normalized == b.normalized),
            float(bool(a.folded) and a.folded == b.folded),
            len(overlap) / len(union) if union else 0.,
            len(overlap) / min(len(a.tokens), len(b.tokens)) if a.tokens and b.tokens else 0.,
            _edit_similarity(a.normalized, b.normalized),
            float(bool(a.numbers and b.numbers) and a.numbers == b.numbers),
            float(bool(number_overlap)),
            len(number_overlap) / len(number_union) if number_union else 0.,
        ))
    values.append(float(bool(left.country) and left.country == right.country))
    return values


def _trusted_indices(anchors, codes, scores, anchor: int, threshold: float, chunk_size: int) -> tuple[int, ...]:
    """Top four let any row select its top three after excluding itself."""
    start, end = (int(np.searchsorted(anchors, anchor, side=side)) for side in ("left", "right"))
    best = []
    for offset in range(start, end, chunk_size):
        stop = min(end, offset + chunk_size)
        for relative in np.flatnonzero(scores[offset:stop] >= threshold):
            index = offset + int(relative)
            item = (float(scores[index]), -int(codes[index]), index)
            if len(best) < 4:
                heapq.heappush(best, item)
            elif item > best[0]:
                heapq.heapreplace(best, item)
    return tuple(item[2] for item in sorted(best, reverse=True))


def _lookup_path(raw_feature_dir: Path) -> tuple[Path, str]:
    metadata_path = raw_feature_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    lookup = raw_feature_dir / "targets.sqlite"
    if not lookup.exists():
        shared = metadata.get("target_lookup", {}).get("shared_source_index")
        if not shared:
            raise ValueError("Raw feature directory has no target lookup")
        lookup = Path(shared)
    lookup = lookup.resolve()
    checksum = _digest(lookup)
    expected = metadata.get("lookup_sha256")
    if expected is not None and checksum != expected:
        raise ValueError("Raw target lookup checksum mismatch")
    # Prepared records are local pickle caches. Only unpickle a cache whose
    # upstream manifest checksum has been verified, as in pair_dataset.
    connection = sqlite3.connect(f"file:{lookup.as_posix()}?mode=ro", uri=True)
    try:
        prepared = any(row[1] == "prepared" for row in connection.execute("PRAGMA table_info(records)"))
    finally:
        connection.close()
    if prepared and expected is None:
        raise ValueError("Prepared target lookup requires its upstream checksum manifest")
    return lookup, checksum


def build_graph_features(
    candidate_dir: str | Path,
    raw_feature_dir: str | Path,
    scores_path: str | Path,
    output_dir: str | Path,
    *,
    score_role: str,
    trusted_threshold: float = .95,
    batch_size: int = 10_000,
    score_provenance: Mapping[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Write extraX.npy only, preserving exact candidate order and row count.

    Supports are chosen before taking batch slices, so arbitrary batch sizes
    produce identical features even when an anchor spans multiple batches.
    ``score_provenance`` can record model/fold identifiers but never affects
    feature computation. Resume verifies content hashes and completed batches.
    """
    if score_role not in SCORE_ROLES:
        raise ValueError(f"score_role must be one of {SCORE_ROLES}")
    if not np.isfinite(trusted_threshold) or not 0 <= trusted_threshold <= 1:
        raise ValueError("trusted_threshold must be finite and between zero and one")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    started = time.perf_counter()
    candidate_dir, raw_feature_dir, output = map(Path, (candidate_dir, raw_feature_dir, output_dir))
    scores_path = Path(scores_path)
    anchors = np.load(candidate_dir / "anchors.npy", mmap_mode="r", allow_pickle=False)
    codes = np.load(candidate_dir / "target_codes.npy", mmap_mode="r", allow_pickle=False)
    scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    if anchors.ndim != 1 or codes.ndim != 1 or anchors.dtype.kind not in "iu" or codes.dtype.kind not in "iu":
        raise ValueError("Candidate anchors and targets must be integer vectors")
    n = len(anchors)
    if codes.shape != (n,) or scores.shape != (n,) or scores.dtype.kind not in "fiu":
        raise ValueError("Scores and target codes must have exactly one value per candidate")
    with (candidate_dir / "queries.jsonl").open(encoding="utf-8") as handle:
        query_count = sum(1 for _ in handle)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a, c, p = anchors[start:end], codes[start:end], scores[start:end]
        if np.any(a < 0) or np.any(a >= query_count):
            raise ValueError("Candidate anchor index is outside queries.jsonl")
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError("Scorer probabilities must be finite and between zero and one")
        prefixes = c // TARGET_FACTOR
        if np.any((prefixes != 2) & (prefixes != 3)):
            raise ValueError("Candidate targets must encode Source2 or Source3")
        previous = max(0, start - 1)
        la, ra = anchors[previous:end - 1], anchors[previous + 1:end]
        lc, rc = codes[previous:end - 1], codes[previous + 1:end]
        if np.any((ra < la) | ((ra == la) & (rc <= lc))):
            raise ValueError("Candidate rows must be unique and sorted by anchor then target code")
    lookup_path, lookup_checksum = _lookup_path(raw_feature_dir)
    config = {
        "schema_version": 1, "score_role": score_role, "score_provenance": dict(score_provenance or {}),
        "scores_sha256": _digest(scores_path), "trusted_threshold": float(trusted_threshold),
        "maximum_supports": 3, "batch_size": batch_size, "lookup_sha256": lookup_checksum,
        "candidate_inputs": {name: _digest(candidate_dir / name) for name in (
            "anchors.npy", "target_codes.npy", "queries.jsonl")},
        "implementation": {name: _digest(Path(__file__).with_name(name)) for name in (
            "graph_features.py", "advanced_features.py", "features.py", "text.py", "romanization.py", "pair_dataset.py")},
    }
    # Canonicalize the supplied provenance to its on-disk JSON form.
    config = json.loads(json.dumps(config, allow_nan=False))
    output.mkdir(parents=True, exist_ok=True)
    metadata_path, matrix_path = output / "metadata.json", output / "extraX.npy"
    shape = (n, len(FEATURE_NAMES))
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["config"] != config:
            raise ValueError("Graph feature inputs/configuration changed; use a new output directory")
        if metadata["complete"]:
            if _digest(matrix_path) != metadata["sha256"]:
                raise ValueError("Completed graph feature artifact checksum mismatch")
            return metadata
    else:
        if any(output.iterdir()):
            raise FileExistsError("Nonempty graph output directory lacks a valid resume manifest")
        metadata = {
            "config": config, "feature_names": list(FEATURE_NAMES), "shape": list(shape),
            "extraX_path": str(matrix_path.resolve()), "rows_completed": 0, "batches": [],
            "complete": False, "runtime_seconds": 0.,
            "training_score_requirement": "Grouped out-of-fold: hold all pairs of each Source1 out together",
        }
        _write_json(metadata_path, metadata)
    previous_runtime = metadata["runtime_seconds"]
    if matrix_path.exists():
        matrix = np.load(matrix_path, mmap_mode="r+", allow_pickle=False)
        if matrix.shape != shape or matrix.dtype != np.float32:
            raise ValueError("Graph checkpoint matrix has unexpected shape/dtype")
    else:
        if metadata["rows_completed"]:
            raise ValueError("Graph checkpoint matrix is missing")
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=shape)
    position = 0
    for batch in metadata["batches"]:
        if batch["start"] != position or not position < batch["end"] <= n:
            raise ValueError("Invalid graph checkpoint batch boundaries")
        if _array_digest(matrix[position:batch["end"]]) != batch["sha256"]:
            raise ValueError("Graph checkpoint batch checksum mismatch")
        position = batch["end"]
    if position != metadata["rows_completed"]:
        raise ValueError("Graph checkpoint row count does not match completed batches")

    @lru_cache(maxsize=4096)
    def supports(anchor):
        return _trusted_indices(anchors, codes, scores, anchor, trusted_threshold, batch_size)

    lookup = sqlite3.connect(f"file:{lookup_path.as_posix()}?mode=ro", uri=True)
    try:
        for start in range(position, n, batch_size):
            end = min(start + batch_size, n)
            support_indices = [index for anchor in np.unique(anchors[start:end]) for index in supports(int(anchor))]
            needed = np.unique(np.concatenate((codes[start:end], codes[support_indices])))
            records = _lookup_records(lookup, needed)
            values = np.zeros((end - start, len(FEATURE_NAMES)), dtype=np.float32)
            for offset, (anchor, code) in enumerate(zip(anchors[start:end], codes[start:end])):
                code = int(code)
                indices = [i for i in supports(int(anchor)) if int(codes[i]) != code][:3]
                probability = float(scores[start + offset])
                values[offset, 0] = probability
                if not indices:
                    continue
                support_codes = [int(codes[i]) for i in indices]
                source2 = sum(c // TARGET_FACTOR == 2 for c in support_codes)
                source3 = len(indices) - source2
                same_source = sum(c // TARGET_FACTOR == code // TARGET_FACTOR for c in support_codes)
                probabilities = [float(scores[i]) for i in indices]
                values[offset, 1:len(_CONTEXT_NAMES)] = (
                    len(indices), source2, source3, bool(source2) + bool(source3), same_source,
                    len(indices) - same_source, max(probabilities), sum(probabilities) / len(probabilities),
                    probability - max(probabilities),
                )
                target = records[code]
                for support_code in support_codes:
                    support = records[support_code]
                    comparisons = np.concatenate((extra_pair_features(target, support), _base_comparison(target, support)))
                    values[offset, len(_CONTEXT_NAMES):] = np.maximum(values[offset, len(_CONTEXT_NAMES):], comparisons)
            if not np.isfinite(values).all():
                raise ValueError("Graph feature generation produced nonfinite values")
            matrix[start:end] = values
            matrix.flush()
            metadata["batches"].append({"start": start, "end": end, "sha256": _array_digest(values)})
            metadata["rows_completed"] = end
            metadata["runtime_seconds"] = previous_runtime + time.perf_counter() - started
            _write_json(metadata_path, metadata)
            if progress:
                progress({"stage": "graph_features", "rows_completed": end, "total_rows": n})
    finally:
        lookup.close()
        matrix.flush()
    metadata.update(complete=True, sha256=_digest(matrix_path), runtime_seconds=previous_runtime + time.perf_counter() - started)
    _write_json(metadata_path, metadata)
    return metadata
