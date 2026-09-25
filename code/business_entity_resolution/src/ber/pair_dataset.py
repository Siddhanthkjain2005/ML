"""Label-independent, resumable features over the exact saved candidate union.

Only bounded batches of target text are loaded from a disk SQLite lookup. The
feature matrix is written directly to a float32 .npy memory map, preserving
candidate row order. Actual TF-IDF cosine is computed for every candidate pair
in all three channels, even when that channel did not retrieve the pair.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable, Mapping

import joblib
import numpy as np

from .features import FEATURE_NAMES, features_for_pairs
from .text import prepare_record

BRANCHES = ("name_char", "name_word", "address_char")
MASK_NAMES = (*BRANCHES, "name_exact", "number_block")
EXTRA_FEATURE_NAMES = tuple(f"actual_{branch}_cosine" for branch in BRANCHES)
PAIR_FEATURE_NAMES = (*FEATURE_NAMES, *EXTRA_FEATURE_NAMES)
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TARGET_FACTOR = 10**12
_ID = re.compile(r"S([123])-([0-9]+)\Z")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _target_code(entity_id: str, expected_source: int) -> int:
    match = _ID.fullmatch(entity_id)
    if match is None or int(match[1]) != expected_source:
        raise ValueError(f"Invalid source{expected_source} ID: {entity_id!r}")
    number = int(match[2])
    if number >= TARGET_FACTOR or entity_id != f"S{expected_source}-{number}":
        raise ValueError(f"Noncanonical or out-of-range target ID: {entity_id!r}")
    return expected_source * TARGET_FACTOR + number


def _inputs(candidate_dir: Path, target_files: Mapping[int, str | Path], vectorizers: Mapping[str, Any], batch_size: int):
    if set(target_files) != {2, 3}:
        raise ValueError("target_files must contain exactly source keys 2 and 3")
    if set(vectorizers) != set(BRANCHES):
        raise ValueError(f"Expected exactly the vectorizer branches {BRANCHES}")
    files = {name: candidate_dir / filename for name, filename in {
        "queries": "queries.jsonl", "anchors": "anchors.npy", "target_codes": "target_codes.npy",
        "masks": "masks.npy", "scores": "scores.npy", "ranks": "ranks.npy",
    }.items()}
    # Labels and truth_counts are deliberately absent: changing them must never
    # change features, trigger a different feature path, or enter fitting here.
    config = {
        "schema_version": 1, "batch_size": batch_size,
        "candidate_inputs": {name: {"path": str(path.resolve()), "sha256": _digest(path)} for name, path in files.items()},
        "target_inputs": {str(source): {"path": str(Path(path).resolve()), "sha256": _digest(Path(path))} for source, path in target_files.items()},
        "vectorizers_sha1": joblib.hash(vectorizers, hash_name="sha1"),
        "source_sha256": {name: _digest(Path(__file__).with_name(name)) for name in ("pair_dataset.py", "features.py", "text.py")},
        "feature_names": list(PAIR_FEATURE_NAMES),
        "mask_names": list(MASK_NAMES), "retrieval_columns": list(BRANCHES),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")},
    }
    queries = []
    seen = set()
    with files["queries"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict) or any(name not in row or not isinstance(row[name], str) for name in SOURCE_COLUMNS):
                raise ValueError("queries.jsonl must contain raw four-field string records")
            match = _ID.fullmatch(row["entity_id"])
            if match is None or match[1] != "1" or row["entity_id"] in seen:
                raise ValueError("Query IDs must be unique Source1 IDs")
            seen.add(row["entity_id"])
            queries.append(row)
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in files.items() if name != "queries"}
    anchors, codes = arrays["anchors"], arrays["target_codes"]
    n = len(anchors)
    if anchors.ndim != 1 or codes.ndim != 1 or anchors.dtype.kind not in "iu" or codes.dtype.kind not in "iu" or len(codes) != n:
        raise ValueError("anchors and target_codes must be equal-length integer vectors")
    if n and (anchors.min() < 0 or anchors.max() >= len(queries)):
        raise ValueError("Candidate anchor index is outside queries.jsonl")
    if arrays["masks"].shape != (n,) or arrays["masks"].dtype.kind not in "iu":
        raise ValueError("masks must be a one-dimensional integer vector")
    if arrays["scores"].shape != (n, 3) or arrays["ranks"].shape != (n, 3):
        raise ValueError("scores and ranks must have shape (candidate pairs, 3)")
    if arrays["ranks"].dtype.kind not in "iu":
        raise ValueError("ranks must be nonnegative integer values")
    # Validate in batches to avoid full-size temporary boolean/sorted copies.
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a, c = anchors[start:end], codes[start:end]
        prefixes = c // TARGET_FACTOR
        if np.any((prefixes != 2) & (prefixes != 3)):
            raise ValueError("Candidate target code must encode S2 or S3 with suffix < 10**12")
        masks = arrays["masks"][start:end]
        if np.any((masks < 1) | (masks > 31)):
            raise ValueError("Candidate masks must contain only the five declared branch bits")
        scores, ranks = arrays["scores"][start:end], arrays["ranks"][start:end]
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(ranks < 0):
            raise ValueError("Candidate retrieval scores/ranks must be finite and nonnegative")
        previous = max(0, start - 1)
        left_a, right_a = anchors[previous:end - 1], anchors[previous + 1:end]
        left_c, right_c = codes[previous:end - 1], codes[previous + 1:end]
        if np.any((right_a < left_a) | ((right_a == left_a) & (right_c <= left_c))):
            raise ValueError("Candidate rows must be unique and sorted by anchor then target code")
    return config, queries, arrays


def _build_lookup(path: Path, target_files: Mapping[int, str | Path], needed: np.ndarray) -> dict[str, int]:
    """Stream each target file once; save only text referenced by candidates."""
    if path.exists():
        path.unlink()  # Incomplete internal lookup; never a user/source file.
    connection = sqlite3.connect(path)
    source_counts = {"source2": 0, "source3": 0}
    selected = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute("CREATE TABLE records (code INTEGER PRIMARY KEY, name TEXT, address TEXT, country TEXT)")
        for source in (2, 3):
            source_path = Path(target_files[source])
            before = source_path.stat()
            with source_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t", strict=True)
                if next(reader, None) != list(SOURCE_COLUMNS):
                    raise ValueError(f"Unexpected target header: {source_path}")
                for row in reader:
                    if len(row) != 4:
                        raise ValueError(f"Malformed target source row: {source_path}:{reader.line_num}")
                    code = _target_code(row[0], source)
                    source_counts[f"source{source}"] += 1
                    position = int(np.searchsorted(needed, code))
                    if position < len(needed) and int(needed[position]) == code:
                        try:
                            connection.execute("INSERT INTO records VALUES (?, ?, ?, ?)", (code, *row[1:]))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"Duplicate needed target ID: {row[0]}") from exc
                        selected += 1
            after = source_path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Target source changed while raw lookup was being built")
            connection.commit()
        if selected != len(needed):
            raise ValueError(f"Candidate targets missing from supplied source files: expected {len(needed)}, found {selected}")
    finally:
        connection.close()
    return {**source_counts, "selected_unique_targets": selected}


def _lookup_records(connection: sqlite3.Connection, codes: np.ndarray) -> dict[int, Any]:
    result = {}
    for start in range(0, len(codes), 500):
        chunk = [int(value) for value in codes[start:start + 500]]
        marks = ",".join("?" for _ in chunk)
        for code, name, address, country in connection.execute(f"SELECT code,name,address,country FROM records WHERE code IN ({marks})", chunk):
            result[code] = prepare_record({
                "entity_id": f"S{code // TARGET_FACTOR}-{code % TARGET_FACTOR}",
                "business_name": name, "business_address": address, "country": country,
            })
    if len(result) != len(codes):
        raise ValueError("Cached target lookup is missing a requested candidate")
    return result


def _compute_batch(queries, arrays, start, end, lookup, counts, vectorizers) -> np.ndarray:
    a = arrays["anchors"][start:end]
    codes = arrays["target_codes"][start:end]
    unique_codes, target_inverse = np.unique(codes, return_inverse=True)
    target_map = _lookup_records(lookup, unique_codes)
    target_records = [target_map[int(code)] for code in unique_codes]
    unique_anchors, query_inverse = np.unique(a, return_inverse=True)
    query_records = [queries[int(anchor)] for anchor in unique_anchors]
    pairs = []
    for offset, (anchor, code) in enumerate(zip(a, codes)):
        absolute = start + offset
        scores = arrays["scores"][absolute]
        ranks = arrays["ranks"][absolute]
        mask = int(arrays["masks"][absolute])
        nonzero_ranks = [int(rank) for rank in ranks if rank > 0]
        name_ranks = [int(rank) for rank in ranks[:2] if rank > 0]
        metadata = {
            "provenance": [name for bit, name in enumerate(MASK_NAMES) if mask & (1 << bit)],
            "candidate_count": int(counts[int(anchor)]),
            "rank": min(nonzero_ranks) if nonzero_ranks else 0,
            "score": float(max(scores)),
            "rank_name": min(name_ranks) if name_ranks else 0,
            "rank_address": int(ranks[2]),
            "score_name": float(max(scores[:2])), "score_address": float(scores[2]),
        }
        for channel, score, rank in zip(BRANCHES, scores, ranks):
            metadata[f"score_{channel}"] = float(score)
            metadata[f"rank_{channel}"] = int(rank)
        pairs.append((queries[int(anchor)], target_map[int(code)], metadata))
    base = features_for_pairs(pairs)
    result = np.empty((end - start, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    result[:, :len(FEATURE_NAMES)] = base
    for index, branch in enumerate(BRANCHES):
        field = "address" if branch == "address_char" else "name"
        left = vectorizers[branch].transform([getattr(row, field).normalized for row in query_records]).tocsr()
        right = vectorizers[branch].transform([getattr(row, field).normalized for row in target_records]).tocsr()
        # Stored vectorizers are L2-normalized, but divide by measured norms so
        # this remains a true cosine for compatible externally supplied models.
        dot = np.asarray(left[query_inverse].multiply(right[target_inverse]).sum(axis=1)).ravel()
        left_norm = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
        right_norm = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
        denominator = left_norm[query_inverse] * right_norm[target_inverse]
        cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
        result[:, len(FEATURE_NAMES) + index] = np.clip(cosine, 0, 1)
    if not np.isfinite(result).all():
        raise ValueError("Feature generation produced nonfinite values")
    return result


def build_pair_dataset(
    candidate_dir: str | Path, target_files: Mapping[int, str | Path],
    vectorizers: Mapping[str, Any], output_dir: str | Path, batch_size: int = 10_000,
    *, progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build or safely resume X.npy, preserving exact saved candidate ordering.

    Returns metadata with X_path, feature_names, shape, rows_completed, complete,
    and input/config fingerprints. A callback can monitor completed batches.
    Resume rejects changed candidate text/arrays, targets, vectorizers, source
    code, or batch size, and verifies checksums of previously completed batches.
    Neither labels.npy nor truth_counts.npy is opened by this function.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    started = time.perf_counter()
    candidate_dir, output = Path(candidate_dir).resolve(), Path(output_dir).resolve()
    config, raw_queries, arrays = _inputs(candidate_dir, target_files, vectorizers, batch_size)
    shape = (len(arrays["anchors"]), len(PAIR_FEATURE_NAMES))
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["config"] != config:
            raise ValueError("Pair-dataset inputs/configuration changed; use a new output directory")
    else:
        if any(output.iterdir()):
            raise FileExistsError("Nonempty pair-dataset directory lacks a valid resume manifest")
        metadata = {"config": config, "X_path": str(output / "X.npy"), "feature_names": list(PAIR_FEATURE_NAMES),
                    "shape": list(shape), "rows_completed": 0, "complete": False,
                    "lookup_complete": False, "batches": [], "runtime_seconds": 0.0}
        _write_json(metadata_path, metadata)
    previous_runtime = metadata["runtime_seconds"]
    lookup_path = output / "targets.sqlite"
    if not metadata["lookup_complete"]:
        needed = np.unique(arrays["target_codes"])
        metadata["target_lookup"] = _build_lookup(lookup_path, target_files, needed)
        metadata["lookup_sha256"] = _digest(lookup_path)
        metadata["lookup_complete"] = True
        _write_json(metadata_path, metadata)
        del needed
    elif _digest(lookup_path) != metadata["lookup_sha256"]:
        raise ValueError("Cached target lookup checksum mismatch")
    X_path = Path(metadata["X_path"])
    if metadata["complete"]:
        if _digest(X_path) != metadata["X_sha256"]:
            raise ValueError("Completed feature artifact checksum mismatch")
        return metadata
    if not X_path.exists():
        if metadata["rows_completed"]:
            raise ValueError("Feature checkpoint exists but X.npy is missing")
        matrix = np.lib.format.open_memmap(X_path, mode="w+", dtype=np.float32, shape=shape)
    else:
        matrix = np.load(X_path, mmap_mode="r+", allow_pickle=False)
        if matrix.shape != shape or matrix.dtype != np.float32:
            raise ValueError("Feature checkpoint matrix has unexpected shape/dtype")
    position = 0
    for batch in metadata["batches"]:
        if batch["start"] != position or batch["end"] > shape[0] or batch["end"] <= position:
            raise ValueError("Invalid feature checkpoint batch boundaries")
        if _array_digest(matrix[batch["start"]:batch["end"]]) != batch["sha256"]:
            raise ValueError("Previously completed feature batch checksum mismatch")
        position = batch["end"]
    if position != metadata["rows_completed"]:
        raise ValueError("Feature checkpoint row count does not match completed batches")
    queries = [prepare_record(row) for row in raw_queries]
    counts = np.bincount(np.asarray(arrays["anchors"], dtype=np.int64), minlength=len(queries))
    lookup = sqlite3.connect(f"file:{lookup_path}?mode=ro", uri=True)
    try:
        for start in range(position, shape[0], batch_size):
            end = min(start + batch_size, shape[0])
            values = _compute_batch(queries, arrays, start, end, lookup, counts, vectorizers)
            matrix[start:end] = values
            matrix.flush()
            metadata["batches"].append({"start": start, "end": end, "sha256": _array_digest(values)})
            metadata["rows_completed"] = end
            metadata["runtime_seconds"] = previous_runtime + time.perf_counter() - started
            _write_json(metadata_path, metadata)
            if progress:
                progress({"stage": "pair_features", "rows_completed": end, "total_rows": shape[0]})
    finally:
        lookup.close()
        matrix.flush()
    metadata["complete"] = True
    metadata["X_sha256"] = _digest(X_path)
    metadata["runtime_seconds"] = previous_runtime + time.perf_counter() - started
    _write_json(metadata_path, metadata)
    return metadata
