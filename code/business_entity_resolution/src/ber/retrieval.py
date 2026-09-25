"""Bounded-memory TF-IDF retrieval with train-only fitting and resumable shards."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalize_text

BRANCHES = ("name_char", "name_word", "address_char")
_FIELDS = {"name_char": "business_name", "name_word": "business_name", "address_char": "business_address"}
_COLUMNS = ("entity_id", "business_name", "business_address", "country")
_ID = re.compile(r"S([123])-([0-9]+)\Z")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if reader.fieldnames != list(_COLUMNS):
            raise ValueError(f"Unexpected source columns in {path}: {reader.fieldnames}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed TSV row in {path}")
            yield row


def _numeric_id(entity_id: str) -> tuple[str, int]:
    match = _ID.fullmatch(entity_id)
    if match is None:
        raise ValueError(f"Invalid entity ID: {entity_id!r}")
    number = int(match[2])
    if number > np.iinfo(np.int64).max or entity_id != f"S{match[1]}-{number}":
        raise ValueError(f"Noncanonical or out-of-range entity ID: {entity_id!r}")
    return f"S{match[1]}", number


def build_vectorizers(
    training_files: Sequence[str | Path],
    output_path: str | Path,
    sample_limit: int = 200_000,
    seed: int = 42,
    *,
    min_df: int | float = 2,
    max_df: int | float = 0.2,
    max_features: int = 300_000,
) -> dict[str, TfidfVectorizer]:
    """Fit three branches on an ID-hash sample of supplied training records.

    The smallest ``sample_limit`` hashes across all files are selected, so the
    sample is reproducible regardless of row or file ordering. The caller must
    pass training-fold files only; obvious held-out/test paths are rejected.
    Small fixtures can explicitly override min_df/max_df. Production defaults
    are never silently relaxed. The returned dictionary is also saved by joblib;
    its provenance is written to ``output_path + '.manifest.json'``.
    """
    if sample_limit <= 0 or max_features <= 0:
        raise ValueError("sample_limit and max_features must be positive")
    paths = sorted({Path(path).resolve() for path in training_files})
    if not paths:
        raise ValueError("At least one training file is required")
    heldout = {"test", "validation", "calibration", "holdout"}
    for path in paths:
        if heldout.intersection(part.casefold() for part in path.parts) or path.name.casefold().startswith(("test_", "validation_", "calibration_")):
            raise ValueError(f"Refusing to fit vectorizers on a held-out path: {path}")
    heap = []
    provenance = []
    total = 0
    for path in paths:
        before = path.stat()
        checksum = _digest(path)
        count = 0
        for row in _rows(path):
            entity_id = row["entity_id"]
            _numeric_id(entity_id)
            priority = int.from_bytes(hashlib.blake2b(f"{seed}\0{entity_id}".encode(), digest_size=16).digest(), "big")
            # Lexical secondary keys make the rare equal-hash case reproducible.
            item = (-priority, entity_id, row["business_name"], row["business_address"])
            if len(heap) < sample_limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            count += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Training file changed during sampling: {path}")
        provenance.append({"path": str(path), "sha256": checksum, "rows": count})
        total += count
    sampled = sorted(heap, key=lambda item: (-item[0], item[1:]))
    if not sampled:
        raise ValueError("No training records available to fit vectorizers")
    names = [normalize_text(item[2]) for item in sampled]
    addresses = [normalize_text(item[3]) for item in sampled]
    common = dict(min_df=min_df, max_df=max_df, max_features=max_features,
                  lowercase=False, dtype=np.float32, norm="l2", sublinear_tf=True)
    vectorizers = {
        "name_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common),
        "name_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b", **common),
        "address_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common),
    }
    for branch, vectorizer in vectorizers.items():
        vectorizer.fit(addresses if branch == "address_char" else names)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    joblib.dump(vectorizers, temporary, compress=3)
    os.replace(temporary, output_path)
    _atomic_json(Path(str(output_path) + ".manifest.json"), {
        "schema_version": 1, "seed": seed, "sample_limit": sample_limit,
        "sample_rows": len(sampled), "seen_rows": total, "training_files": provenance,
        "sample_entity_ids_sha256": _json_digest([item[1] for item in sampled]),
        "normalizer_sha256": _digest(Path(__file__).with_name("text.py")),
        "config": {"min_df": min_df, "max_df": max_df, "max_features": max_features},
        "vocabulary_sizes": {key: len(value.vocabulary_) for key, value in vectorizers.items()},
        "artifact_sha256": _digest(output_path),
    })
    return vectorizers


def _target_cache(target_path, vectorizers, cache_root, target_batch, progress):
    """Build atomically published immutable CSR shards, with raw-input hashes."""
    before = target_path.stat()
    checksum = _digest(target_path)
    config = {"schema_version": 1, "target_sha256": checksum, "target_batch": target_batch,
              "vectorizers_hash": joblib.hash(vectorizers, hash_name="sha1"),
              "normalizer_sha256": _digest(Path(__file__).with_name("text.py"))}
    key = _json_digest(config)
    destination = cache_root / key
    if (destination / "manifest.json").exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["config"] != config:
            raise ValueError("Target cache configuration mismatch")
        for shard in manifest["shards"]:
            for filename, digest in shard["sha256"].items():
                if _digest(destination / filename) != digest:
                    raise ValueError(f"Target cache corruption: {filename}")
        return destination, manifest
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_root) as temp:
        stage = Path(temp) / "cache"
        stage.mkdir()
        shards = []
        source = None
        all_ids = []
        iterator = iter(_rows(target_path))
        while True:
            batch = []
            for _ in range(target_batch):
                row = next(iterator, None)
                if row is None:
                    break
                prefix, number = _numeric_id(row["entity_id"])
                if source is None:
                    source = prefix
                if source != prefix:
                    raise ValueError("Each retrieval target file must contain exactly one source prefix")
                batch.append((number, row))
            if not batch:
                break
            batch.sort(key=lambda item: item[0])
            ids = np.asarray([item[0] for item in batch], dtype=np.int64)
            all_ids.append(ids)
            shard_no = len(shards)
            ids_file = f"{shard_no:06d}.ids.npy"
            np.save(stage / ids_file, ids)
            file_hashes = {ids_file: _digest(stage / ids_file)}
            names = [normalize_text(item[1]["business_name"]) for item in batch]
            addresses = [normalize_text(item[1]["business_address"]) for item in batch]
            for branch in BRANCHES:
                matrix = vectorizers[branch].transform(addresses if branch == "address_char" else names).astype(np.float32).tocsr()
                matrix.sort_indices()
                filename = f"{shard_no:06d}.{branch}.npz"
                sparse.save_npz(stage / filename, matrix, compressed=False)
                file_hashes[filename] = _digest(stage / filename)
            shards.append({"shard": shard_no, "rows": len(batch), "sha256": file_hashes})
            if progress:
                progress({"stage": "target_cache_shard", "shard": shard_no, "rows": len(batch)})
        if all_ids:
            complete_ids = np.concatenate(all_ids)
            complete_ids.sort()
            if np.any(complete_ids[1:] == complete_ids[:-1]):
                raise ValueError("Duplicate entity IDs in retrieval target file")
        after = target_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Target TSV changed while cache was being built")
        manifest = {"config": config, "target_path": str(target_path), "target_source": source,
                    "rows": sum(shard["rows"] for shard in shards), "shards": shards}
        _atomic_json(stage / "manifest.json", manifest)
        os.replace(stage, destination)
    return destination, manifest


def _merge_topk(old_ids, old_scores, new_ids, new_scores, k):
    ids = np.concatenate((old_ids, new_ids))
    scores = np.concatenate((old_scores, new_scores))
    valid = (ids >= 0) & np.isfinite(scores) & (scores > 0)
    ids, scores = ids[valid], scores[valid]
    selected = np.lexsort((ids, -scores))[:k]
    out_ids = np.full(k, -1, dtype=np.int64)
    out_scores = np.zeros(k, dtype=np.float32)
    out_ids[:len(selected)], out_scores[:len(selected)] = ids[selected], scores[selected]
    return out_ids, out_scores


def _multiply_topk(queries, targets, target_transpose, target_ids, k, threads):
    from sparse_dot_topn import sp_matmul_topn

    # An extra result detects boundary ties. Resolve those rows exactly so IDs,
    # rather than library heap traversal order, determine equal-score winners.
    count = min(k + 1, targets.shape[0])
    result = sp_matmul_topn(queries, target_transpose, top_n=count, threshold=0.0,
                           sort=True, n_threads=threads)
    for row in range(queries.shape[0]):
        start, stop = result.indptr[row:row + 2]
        columns, scores = result.indices[start:stop], result.data[start:stop]
        if len(scores) > k and scores[k - 1] == scores[k]:
            exact = (queries[row] @ targets.T).tocsr()
            columns, scores = exact.indices, exact.data
        yield target_ids[columns], scores


def retrieve_sparse(
    query_records: Sequence[Mapping[str, str]],
    target_path: str | Path,
    vectorizers: Mapping[str, TfidfVectorizer],
    output_dir: str | Path,
    k: int = 20,
    threads: int = 4,
    query_batch: int = 256,
    target_batch: int = 100_000,
    *,
    resume: bool = True,
    cache_dir: str | Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """Retrieve positive-cosine top-k per branch for one target source.

    Results contain string ``query_ids``, numeric-suffix ``target_ids`` (int64,
    -1 padding), float32 ``scores`` (0 padding), ``target_source``, and ``path``.
    Target IDs are meaningful with the returned source prefix. Exact ties use
    ascending numeric target ID. Each completed target shard is checkpointed
    atomically. Resuming with changed queries, vectorizers, target content, or
    retrieval settings raises instead of silently mixing incompatible output.
    Cached target matrices can be reused across query sets via ``cache_dir``.
    """
    if any(not isinstance(value, int) or value <= 0 for value in (k, threads, query_batch, target_batch)):
        raise ValueError("k, threads, query_batch, target_batch must be positive integers")
    if set(vectorizers) != set(BRANCHES):
        raise ValueError(f"Expected vectorizer branches {BRANCHES}")
    output_dir, target_path = Path(output_dir).resolve(), Path(target_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = [dict(row) for row in query_records]
    query_ids = np.asarray([row["entity_id"] for row in queries], dtype=str)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate query entity IDs")
    for entity_id in query_ids:
        _numeric_id(str(entity_id))
    cached, cache = _target_cache(target_path, vectorizers,
                                  Path(cache_dir).resolve() if cache_dir is not None else output_dir / "target_cache",
                                  target_batch, progress)
    config = {"schema_version": 1, "query_sha256": _json_digest(queries),
              "target_cache_key": cached.name, "k": k, "query_batch": query_batch,
              "target_batch": target_batch, "threads": threads}
    config_path = output_dir / "retrieval.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous["config"] != config:
            raise ValueError("Retrieval inputs/configuration changed; choose a new output directory")
        if not resume:
            raise FileExistsError("Retrieval output already exists and resume=False")
    else:
        _atomic_json(config_path, {"config": config, "target_source": cache["target_source"], "complete": False})
    ids = {branch: np.full((len(queries), k), -1, dtype=np.int64) for branch in BRANCHES}
    scores = {branch: np.zeros((len(queries), k), dtype=np.float32) for branch in BRANCHES}
    checkpoint = output_dir / "checkpoint.npz"
    completed = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["config_sha256"]) != _json_digest(config):
                raise ValueError("Checkpoint configuration does not match retrieval")
            completed = int(saved["completed_shards"])
            if not 0 <= completed <= len(cache["shards"]):
                raise ValueError("Invalid checkpoint shard count")
            for branch in BRANCHES:
                ids[branch], scores[branch] = saved[f"{branch}_ids"], saved[f"{branch}_scores"]
                if ids[branch].shape != (len(queries), k) or scores[branch].shape != (len(queries), k):
                    raise ValueError("Checkpoint shape mismatch")
    normalized_names = [normalize_text(row.get("business_name", "")) for row in queries]
    normalized_addresses = [normalize_text(row.get("business_address", "")) for row in queries]
    query_matrices = {
        branch: vectorizers[branch].transform(normalized_addresses if branch == "address_char" else normalized_names).astype(np.float32).tocsr()
        for branch in BRANCHES
    } if queries else {}
    for shard in cache["shards"][completed:]:
        number = shard["shard"]
        target_ids = np.load(cached / f"{number:06d}.ids.npy", allow_pickle=False)
        for branch in BRANCHES:
            matrix = sparse.load_npz(cached / f"{number:06d}.{branch}.npz").tocsr()
            target_transpose = matrix.T.tocsr()
            for start in range(0, len(queries), query_batch):
                batch = query_matrices[branch][start:start + query_batch]
                for offset, (local_ids, local_scores) in enumerate(_multiply_topk(batch, matrix, target_transpose, target_ids, k, threads)):
                    row = start + offset
                    ids[branch][row], scores[branch][row] = _merge_topk(
                        ids[branch][row], scores[branch][row], local_ids, local_scores, k)
        completed = number + 1
        _atomic_npz(checkpoint, completed_shards=np.asarray(completed), config_sha256=np.asarray(_json_digest(config)),
                    **{f"{branch}_ids": ids[branch] for branch in BRANCHES},
                    **{f"{branch}_scores": scores[branch] for branch in BRANCHES})
        if progress:
            progress({"stage": "retrieval_shard_complete", "completed_shards": completed,
                      "total_shards": len(cache["shards"]), "target_source": cache["target_source"]})
    results = {}
    for branch in BRANCHES:
        path = output_dir / f"{branch}.npz"
        _atomic_npz(path, query_ids=query_ids, target_ids=ids[branch], scores=scores[branch],
                    target_source=np.asarray(cache["target_source"] or ""))
        results[branch] = {"query_ids": query_ids, "target_ids": ids[branch], "scores": scores[branch],
                           "target_source": cache["target_source"], "path": str(path)}
    _atomic_json(config_path, {"config": config, "target_source": cache["target_source"], "complete": True,
                              "query_count": len(queries), "target_count": cache["rows"],
                              "output_sha256": {branch: _digest(Path(results[branch]["path"])) for branch in BRANCHES}})
    return results
