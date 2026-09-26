"""Deterministic sparse/exact candidate union; labels are diagnostic only."""
from __future__ import annotations

import array
import csv
import hashlib
import heapq
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .text import normalize_text

BRANCHES = ("name_char", "name_word", "address_char", "name_exact", "number_block")
BRANCH_BITS = {name: 1 << index for index, name in enumerate(BRANCHES)}
TARGET_MULTIPLIER = 10**12
_COLUMNS = ("entity_id", "business_name", "business_address", "country")
_ID = re.compile(r"S([123])-([0-9]+)\Z")
_FILES = {"anchors": "anchors.npy", "target_codes": "target_codes.npy", "masks": "masks.npy",
          "scores": "scores.npy", "ranks": "ranks.npy", "queries": "queries.jsonl"}


def encode_target_id(entity_id: str) -> int:
    match = _ID.fullmatch(entity_id)
    if match is None or match[1] not in ("2", "3"):
        raise ValueError(f"Invalid target ID: {entity_id!r}")
    number = int(match[2])
    if number >= TARGET_MULTIPLIER or entity_id != f"S{match[1]}-{number}":
        raise ValueError(f"Noncanonical or too-large target ID: {entity_id!r}")
    return int(match[1]) * TARGET_MULTIPLIER + number


def decode_target_code(code: int) -> str:
    source, number = divmod(int(code), TARGET_MULTIPLIER)
    if source not in (2, 3):
        raise ValueError(f"Invalid target code: {code}")
    return f"S{source}-{number}"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype.str}:{value.shape}".encode())
    if value.size:
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if reader.fieldnames != list(_COLUMNS):
            raise ValueError(f"Unexpected source TSV columns in {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed source TSV record in {path}")
            yield row


def _bucket(size: int) -> str:
    return "0" if size == 0 else "1" if size == 1 else "2-3" if size <= 3 else "4-6" if size <= 6 else "7+"


def _metric_group() -> dict:
    return {"anchors": 0, "true_links": 0, "recalled_links": 0, "positive_anchors": 0,
            "sum_positive_anchor_recall": 0.0, "oracle_macro_f0_5_sum": 0.0}


def _update_group(group: dict, true_count: int, recalled: int):
    group["anchors"] += 1
    group["true_links"] += true_count
    group["recalled_links"] += recalled
    if true_count:
        group["positive_anchors"] += 1
        group["sum_positive_anchor_recall"] += recalled / true_count
        group["oracle_macro_f0_5_sum"] += 1.25 * recalled / (recalled + 0.25 * true_count)
    else:
        group["oracle_macro_f0_5_sum"] += 1.0


def _finish_group(group: dict) -> dict:
    result = dict(group)
    result["link_recall"] = group["recalled_links"] / group["true_links"] if group["true_links"] else None
    result["macro_recall_positive_anchors"] = group["sum_positive_anchor_recall"] / group["positive_anchors"] if group["positive_anchors"] else None
    result["oracle_macro_f0_5"] = group["oracle_macro_f0_5_sum"] / group["anchors"] if group["anchors"] else None
    result.pop("sum_positive_anchor_recall")
    result.pop("oracle_macro_f0_5_sum")
    return result


def _result(output: Path, manifest: dict) -> dict:
    paths = {name: str(output / filename) for name, filename in manifest["artifacts"].items()}
    return {"output_dir": str(output), "manifest": str(output / "manifest.json"),
            "paths": paths, "stats": manifest["stats"], "branches": BRANCHES,
            "runtime_seconds": manifest["runtime_seconds"]}


def build_candidates(query_records: Sequence[Mapping[str, str]], target_files: Mapping[int, str | Path],
                     retrieval_results: Mapping[int, dict], output_dir: str | Path,
                     truth: Mapping[str, set[str]] | None = None, exact_cap: int = 200,
                     *, source_index_path: str | Path | None = None) -> dict:
    """Union three sparse branches and two exact rescues without using labels.

    Sparse results are the dictionaries returned by retrieve_sparse. Target codes
    encode source*10**12 + numeric suffix. Scores/ranks have three sparse columns;
    zero rank means that branch did not retrieve the pair. Exact rescue candidates
    share one per-anchor cap across both target sources and both exact branches;
    the numerically smallest target codes survive overflow. Truth only creates
    optional label/count arrays and recall diagnostics after candidates are fixed.
    """
    started = time.monotonic()
    if not isinstance(exact_cap, int) or exact_cap < 0:
        raise ValueError("exact_cap must be a nonnegative integer")
    if set(target_files) != {2, 3} or set(retrieval_results) != {2, 3}:
        raise ValueError("Both Source 2 and Source 3 target files/results are required")
    queries = [dict(row) for row in query_records]
    if len(queries) > np.iinfo(np.uint32).max:
        raise ValueError("Too many query anchors for uint32")
    query_ids = [row["entity_id"] for row in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate query IDs")
    for entity_id in query_ids:
        match = _ID.fullmatch(entity_id)
        if match is None or match[1] != "1" or entity_id != f"S1-{int(match[2])}":
            raise ValueError(f"Noncanonical Source 1 query ID: {entity_id}")
    files = {source: Path(path).resolve() for source, path in target_files.items()}
    source_stats = {source: (path.stat().st_size, path.stat().st_mtime_ns) for source, path in files.items()}
    retrieval_signature = {}
    required_ids = {}
    normalized_results = {}
    for source in (2, 3):
        normalized_results[source] = {}
        gathered = []
        retrieval_signature[str(source)] = {}
        for branch in BRANCHES[:3]:
            item = retrieval_results[source][branch]
            ids, scores = np.asarray(item["target_ids"]), np.asarray(item["scores"])
            if ids.ndim != 2 or ids.shape != scores.shape or ids.shape[0] != len(queries):
                raise ValueError(f"Invalid {source}/{branch} sparse array shape")
            if ids.shape[1] > np.iinfo(np.uint16).max:
                raise ValueError("Sparse rank exceeds uint16")
            if list(map(str, item["query_ids"])) != query_ids:
                raise ValueError("Sparse query order does not match supplied records")
            if item.get("target_source") not in (f"S{source}", source, str(source)):
                raise ValueError("Sparse target source mismatch")
            if ids.dtype.kind not in "iu" or np.any(ids < -1) or np.any(ids >= TARGET_MULTIPLIER):
                raise ValueError("Invalid sparse target numeric ID")
            valid = ids >= 0
            if np.any(~np.isfinite(scores)) or np.any(scores[valid] <= 0) or np.any(scores[~valid] != 0):
                raise ValueError("Invalid sparse scores/padding")
            normalized_results[source][branch] = (ids, scores)
            gathered.append(ids[valid].astype(np.uint64))
            retrieval_signature[str(source)][branch] = {"ids": _array_digest(ids), "scores": _array_digest(scores)}
        required_ids[source] = np.unique(np.concatenate(gathered)) if gathered else np.empty(0, np.uint64)
    truth_codes = None
    if truth is not None:
        if any(entity_id not in truth for entity_id in query_ids):
            raise ValueError("Truth must contain every query, including singleton rows")
        truth_codes = [{encode_target_id(target) for target in truth[entity_id]} for entity_id in query_ids]
    config = {
        "schema_version": 1, "query_sha256": _json_digest(queries), "exact_cap": exact_cap,
        "target_files": {str(source): {"path": str(path), "sha256": _digest(path)} for source, path in files.items()},
        "retrieval_arrays": retrieval_signature,
        "truth_sha256": _json_digest([sorted(values) for values in truth_codes]) if truth_codes is not None else None,
        "normalizer_sha256": _digest(Path(__file__).with_name("text.py")), "implementation_sha256": _digest(Path(__file__)),
    }
    index_meta = None
    if source_index_path is not None:
        source_index_path = Path(source_index_path).resolve()
        index_meta = json.loads((source_index_path.parent / 'manifest.json').read_text())
        for source in (2, 3):
            if index_meta['config'][str(source)] != config['target_files'][str(source)]:
                raise ValueError('Source index does not match target files')
        checksum = _digest(source_index_path)
        if checksum != index_meta['sha256'] or index_meta['config']['normalizer_sha256'] != config['normalizer_sha256']:
            raise ValueError('Source index corrupted or normalizer changed')
        config['source_index_sha256'] = checksum
    output = Path(output_dir).resolve()
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("complete") is not True or previous.get("config") != config:
            raise ValueError("Candidate inputs/config changed; choose a new output directory")
        for name, expected in previous["sha256"].items():
            if _digest(output / name) != expected:
                raise ValueError(f"Candidate artifact corruption: {name}")
        return _result(output, previous)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete/nonempty candidate output exists; choose a clean directory")
    name_queries = defaultdict(list)
    address_queries = defaultdict(list)
    for index, query in enumerate(queries):
        name = normalize_text(query.get("business_name", ""))
        address = normalize_text(query.get("business_address", ""))
        if name:
            name_queries[name].append(index)
        if address and any(char.isdecimal() for char in address):
            address_queries[address].append(index)
    exact = [{} for _ in queries]
    exact_heaps = [[] for _ in queries]
    exact_eligible = np.zeros(len(queries), dtype=np.uint64)
    exact_branch_eligible = {branch: 0 for branch in BRANCHES[3:]}
    target_rows = {}
    for source in (2, 3):
        required = required_ids[source]
        seen = np.zeros(len(required), dtype=bool)
        buffered_ids = []
        count = 0

        def check_buffer():
            if not buffered_ids or not len(required):
                buffered_ids.clear()
                return
            numbers = np.asarray(buffered_ids, dtype=np.uint64)
            loc = np.searchsorted(required, numbers)
            valid = loc < len(required)
            loc, numbers = loc[valid], numbers[valid]
            hit = loc[required[loc] == numbers]
            if len(hit) != len(np.unique(hit)) or np.any(seen[hit]):
                raise ValueError("Duplicate retrieved target IDs in raw target file")
            seen[hit] = True
            buffered_ids.clear()

        if source_index_path is None:
            raw_rows = _rows(files[source])
        else:
            from .source_index import iter_selected_records
            raw_rows = iter_selected_records(source_index_path, source,
                required + source * TARGET_MULTIPLIER, name_queries, address_queries)
        for row in raw_rows:
            count += 1
            code = encode_target_id(row["entity_id"])
            if code // TARGET_MULTIPLIER != source:
                raise ValueError("Raw target file contains wrong source prefix")
            buffered_ids.append(code % TARGET_MULTIPLIER)
            if len(buffered_ids) == 20000:
                check_buffer()
            name = normalize_text(row["business_name"])
            address = normalize_text(row["business_address"])
            hits = {}
            for index in name_queries.get(name, ()):
                hits[index] = BRANCH_BITS["name_exact"]
                exact_branch_eligible["name_exact"] += 1
            for index in address_queries.get(address, ()):
                hits[index] = hits.get(index, 0) | BRANCH_BITS["number_block"]
                exact_branch_eligible["number_block"] += 1
            for index, mask in hits.items():
                exact_eligible[index] += 1
                selected = exact[index]
                if code in selected:
                    raise ValueError("Duplicate exact target ID in raw target file")
                if len(selected) < exact_cap:
                    selected[code] = mask
                    heapq.heappush(exact_heaps[index], -code)
                elif exact_cap and code < -exact_heaps[index][0]:
                    removed = -heapq.heapreplace(exact_heaps[index], -code)
                    del selected[removed]
                    selected[code] = mask
        check_buffer()
        if not np.all(seen):
            raise ValueError(f"Sparse results reference missing Source {source} targets: {required[~seen][:5].tolist()}")
        target_rows[str(source)] = count if index_meta is None else index_meta['counts'][str(source)]
        if source_stats[source] != (files[source].stat().st_size, files[source].stat().st_mtime_ns):
            raise ValueError("Target file changed during candidate construction")
    anchor_values, target_values, mask_values = array.array("I"), array.array("Q"), array.array("H")
    score_values, rank_values, label_values = array.array("f"), array.array("H"), array.array("B")
    candidate_counts = np.zeros(len(queries), dtype=np.uint32)
    branch_pair_counts = Counter()
    overall, by_country, by_match_count = _metric_group(), defaultdict(_metric_group), defaultdict(_metric_group)
    by_source = {str(source): _metric_group() for source in (2, 3)}
    standalone, incremental = {branch: 0 for branch in BRANCHES}, {branch: 0 for branch in BRANCHES}
    for index, query in enumerate(queries):
        union = {}
        for source in (2, 3):
            for branch_index, branch in enumerate(BRANCHES[:3]):
                ids, scores = normalized_results[source][branch]
                for rank, (target_id, score) in enumerate(zip(ids[index], scores[index]), 1):
                    if target_id < 0:
                        continue
                    code = source * TARGET_MULTIPLIER + int(target_id)
                    item = union.setdefault(code, [0, [0.0] * 3, [0] * 3])
                    if item[0] & BRANCH_BITS[branch]:
                        raise ValueError("Duplicate target within a sparse branch/query")
                    item[0] |= BRANCH_BITS[branch]
                    item[1][branch_index], item[2][branch_index] = float(score), rank
        for code, mask in exact[index].items():
            item = union.setdefault(code, [0, [0.0] * 3, [0] * 3])
            item[0] |= mask
        candidate_counts[index] = len(union)
        true = truth_codes[index] if truth_codes is not None else set()
        recalled = 0
        recalled_source = Counter()
        for code in sorted(union):
            mask, scores, ranks = union[code]
            anchor_values.append(index)
            target_values.append(code)
            mask_values.append(mask)
            score_values.extend(scores)
            rank_values.extend(ranks)
            for branch in BRANCHES:
                branch_pair_counts[branch] += bool(mask & BRANCH_BITS[branch])
            if truth_codes is not None:
                positive = code in true
                label_values.append(positive)
                if positive:
                    recalled += 1
                    recalled_source[str(code // TARGET_MULTIPLIER)] += 1
                    for branch in BRANCHES:
                        standalone[branch] += bool(mask & BRANCH_BITS[branch])
                    first = next(branch for branch in BRANCHES if mask & BRANCH_BITS[branch])
                    incremental[first] += 1
        if truth_codes is not None:
            _update_group(overall, len(true), recalled)
            _update_group(by_country[query.get("country", "")], len(true), recalled)
            _update_group(by_match_count[_bucket(len(true))], len(true), recalled)
            for source in (2, 3):
                ntrue = sum(code // TARGET_MULTIPLIER == source for code in true)
                _update_group(by_source[str(source)], ntrue, recalled_source[str(source)])
    count = len(anchor_values)
    stats = {
        "query_count": len(queries), "candidate_pairs": count, "target_rows": target_rows,
        "candidate_counts": {"mean": float(candidate_counts.mean()) if len(queries) else 0,
                             "max": int(candidate_counts.max()) if len(queries) else 0,
                             **{f"p{q}": float(np.percentile(candidate_counts, q)) if len(queries) else 0 for q in (50, 90, 95, 99)}},
        "candidate_pairs_by_branch": dict(branch_pair_counts), "exact_branch_eligible_pairs": exact_branch_eligible,
        "exact_eligible_pairs": int(exact_eligible.sum()), "exact_retained_pairs": sum(map(len, exact)),
        "exact_overflow_queries": int(np.count_nonzero(exact_eligible > exact_cap)),
        "exact_discarded_pairs": int(sum(max(0, int(value) - exact_cap) for value in exact_eligible)),
        "exact_cap_policy": "Combined exact-name/numeric-address union across both sources; smallest numeric encoded target IDs retained",
    }
    if truth_codes is not None:
        denominator = overall["true_links"]
        cumulative = 0
        incremental_stats = {}
        for branch in BRANCHES:
            cumulative += incremental[branch]
            incremental_stats[branch] = {"new_positive_links": incremental[branch], "cumulative_positive_links": cumulative,
                                         "cumulative_recall": cumulative / denominator if denominator else None}
        stats["label_diagnostics"] = {
            "overall": _finish_group(overall), "by_source": {key: _finish_group(value) for key, value in by_source.items()},
            "by_country": {key: _finish_group(value) for key, value in by_country.items()},
            "by_match_count": {key: _finish_group(value) for key, value in by_match_count.items()},
            "standalone_branch_recall": {branch: {"recalled_links": standalone[branch], "recall": standalone[branch] / denominator if denominator else None} for branch in BRANCHES},
            "incremental_branch_recall": incremental_stats,
            "oracle_note": "Perfect classifier selects only recalled true links; empty true anchors score one; this is a candidate ceiling, not measured model performance.",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.building-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidates"
        stage.mkdir()
        artifacts = dict(_FILES)
        arrays = {"anchors": np.asarray(anchor_values, np.uint32), "target_codes": np.asarray(target_values, np.uint64),
                  "masks": np.asarray(mask_values, np.uint16), "scores": np.asarray(score_values, np.float32).reshape(count, 3),
                  "ranks": np.asarray(rank_values, np.uint16).reshape(count, 3)}
        if truth_codes is not None:
            artifacts.update(labels="labels.npy", truth_counts="truth_counts.npy")
            arrays["labels"] = np.asarray(label_values, np.uint8)
            arrays["truth_counts"] = np.asarray([len(values) for values in truth_codes], np.uint32)
        for name, value in arrays.items():
            np.save(stage / artifacts[name], value, allow_pickle=False)
        with (stage / artifacts["queries"]).open("w", encoding="utf-8") as stream:
            for query in queries:
                stream.write(json.dumps(query, ensure_ascii=False) + "\n")
        manifest = {"complete": True, "config": config, "branches": BRANCHES, "branch_bits": BRANCH_BITS,
                    "artifacts": artifacts, "stats": stats, "runtime_seconds": time.monotonic() - started,
                    "sha256": {filename: _digest(stage / filename) for filename in artifacts.values()}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if output.exists():
            output.rmdir()
        os.replace(stage, output)
    return _result(output, manifest)
