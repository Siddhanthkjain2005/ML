"""Bounded positive-pair inspection of an already materialized training split."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import random
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .features import features_for_pair
from .text import prepare_record

_VERSION = 2
_SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
_GT_COLUMNS = ("source1_entity_id", "matched_entity_ids")
_QUANTILE_FEATURES = (
    "name_edit_similarity", "address_edit_similarity", "name_token_jaccard", "address_token_jaccard",
    "name_token_containment", "address_token_containment", "name_token_sort_similarity", "address_token_sort_similarity",
    "name_length_ratio", "address_length_ratio", "name_numbers_jaccard", "address_numbers_jaccard",
)
_EXAMPLE_COLUMNS = (
    "source1_entity_id", "target_entity_id", "left_name", "right_name", "left_address", "right_address",
    "left_country", "right_country", "name_edit_similarity", "address_edit_similarity", "name_token_jaccard",
    "address_token_jaccard", "hardness", "categories",
)


def _read_rows(path: Path, columns: tuple[str, ...]):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if reader.fieldnames != list(columns):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames!r}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed TSV record {line} in {path}")
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(paths: dict[str, Path]) -> dict:
    return {name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for name, path in paths.items()}


def _quantiles(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    qs = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    return {"count": len(array), "mean": float(array.mean()),
            **{f"p{q}": float(v) for q, v in zip(qs, np.percentile(array, qs))}}


def _scripts(text: str) -> frozenset[str]:
    """Coarse Unicode-name heuristic; not language identification."""
    scripts = set()
    known = ("LATIN", "CYRILLIC", "GREEK", "DEVANAGARI", "ARABIC", "HEBREW", "THAI", "HANGUL", "HIRAGANA", "KATAKANA")
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        script = next((script for script in known if script in name), None)
        if script is None and ("CJK" in name or "IDEOGRAPH" in name):
            script = "HAN"
        if script is not None:
            scripts.add(script)
    return frozenset(scripts)


def _categories(left: dict, right: dict, f: dict) -> list[str]:
    categories = [key for key in (
        "name_exact", "name_folded_exact", "name_legal_exact", "name_core_exact", "address_exact", "address_folded_exact",
        "name_numbers_conflict", "name_numbers_disagree", "address_numbers_conflict", "address_numbers_disagree", "country_conflict",
    ) if f[key] > 0]
    for field, raw in (("name", "business_name"), ("address", "business_address")):
        if f[f"{field}_left_missing"] or f[f"{field}_right_missing"]:
            categories.append(f"any_{field}_missing")
        if left[raw] and left[raw] == right[raw]:
            categories.append(f"raw_{field}_exact")
        if not left[raw].isascii() or not right[raw].isascii():
            categories.append(f"any_{field}_nonascii")
        scripts_left, scripts_right = _scripts(left[raw]), _scripts(right[raw])
        if scripts_left and scripts_right and not scripts_left & scripts_right:
            categories.append(f"{field}_script_disjoint")
        if f[f"{field}_token_overlap"] == 0:
            categories.append(f"{field}_no_shared_tokens")
    if f["name_edit_similarity"] < 0.3:
        categories.append("name_edit_below_0_3")
        if f["address_edit_similarity"] > 0.7:
            categories.append("low_name_strong_address")
    if any(not left[c].isascii() or not right[c].isascii() for c in ("business_name", "business_address")):
        categories.append("any_nonascii")
    return categories


def run_oracle(split_dir: str | Path, output_dir: str | Path, anchor_limit: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Sample S1 uniformly; inspect all its positives without reading any archive.

    split_dir must be the training split, containing source1.tsv/source2.tsv/
    source3.tsv/ground_truth.tsv. Raw input strings are preserved. Existing
    matching outputs are loaded when config, input stat fingerprints and example
    checksum agree. Changed inputs trigger a fresh atomic report publication.
    """
    if not isinstance(anchor_limit, int) or anchor_limit < 1:
        raise ValueError("anchor_limit must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    split_dir, output_dir = Path(split_dir).resolve(), Path(output_dir).resolve()
    if split_dir.name in {"test", "validation", "calibration"}:
        raise ValueError("Positive oracle must be run on the training split")
    paths = {name: split_dir / f"{name}.tsv" for name in ("source1", "source2", "source3", "ground_truth")}
    inputs = _fingerprint(paths)
    config = {"version": _VERSION, "anchor_limit": anchor_limit, "seed": seed}
    report_path, examples_path = output_dir / "positive_oracle.json", output_dir / "positive_oracle_hard_examples.tsv"
    if report_path.exists() and examples_path.exists():
        try:
            old = json.loads(report_path.read_text(encoding="utf-8"))
            if old.get("config") == config and old.get("inputs") == inputs and old.get("hard_examples_sha256") == _sha256(examples_path):
                return old
        except (ValueError, OSError):
            pass
    rng = random.Random(seed)
    reservoir = []
    source1_rows = 0
    for source1_rows, row in enumerate(_read_rows(paths["source1"], _SOURCE_COLUMNS), 1):
        if len(reservoir) < anchor_limit:
            reservoir.append(row)
        else:
            slot = rng.randrange(source1_rows)
            if slot < anchor_limit:
                reservoir[slot] = row
    anchors = {row["entity_id"]: row for row in reservoir}
    if len(anchors) != len(reservoir):
        raise ValueError("Duplicate sampled Source 1 IDs")
    labels = {}
    needed = {"S2": set(), "S3": set()}
    for row in _read_rows(paths["ground_truth"], _GT_COLUMNS):
        anchor = row["source1_entity_id"]
        if anchor not in anchors:
            continue
        if anchor in labels:
            raise ValueError(f"Duplicate sampled ground-truth anchor {anchor}")
        targets = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        if len(set(targets)) != len(targets):
            raise ValueError(f"Repeated target in ground-truth row {anchor}")
        labels[anchor] = targets
        for target in targets:
            source = target.partition("-")[0]
            if source not in needed:
                raise ValueError(f"Unexpected target ID {target}")
            needed[source].add(target)
    missing_anchors = anchors.keys() - labels.keys()
    if missing_anchors:
        raise ValueError(f"Sampled S1 lacks ground truth: {sorted(missing_anchors)[:5]}")
    targets = {}
    scanned_target_rows = {}
    for source in (2, 3):
        wanted = needed[f"S{source}"]
        count = 0
        for row in _read_rows(paths[f"source{source}"], _SOURCE_COLUMNS):
            count += 1
            entity_id = row["entity_id"]
            if entity_id in wanted:
                if entity_id in targets:
                    raise ValueError(f"Repeated needed target ID {entity_id}")
                targets[entity_id] = row
        scanned_target_rows[f"source{source}"] = count
        missing = wanted - targets.keys()
        if missing:
            raise ValueError(f"Missing positive targets in source{source}: {sorted(missing)[:5]}")
    prepared_targets = {key: prepare_record(row) for key, row in targets.items()}
    categories = Counter()
    by_source = Counter()
    countries = Counter()
    distributions = {key: [] for key in _QUANTILE_FEATURES}
    lengths = {f"{side}_{field}": [] for side in ("left", "right") for field in ("name", "address")}
    heap = []
    positive_count = 0
    for anchor_id in sorted(anchors):
        left = anchors[anchor_id]
        prepared_left = prepare_record(left)
        for target_id in sorted(labels[anchor_id]):
            right = targets[target_id]
            f = features_for_pair(prepared_left, prepared_targets[target_id])
            positive_count += 1
            flags = _categories(left, right, f)
            categories.update(flags)
            by_source[target_id.partition("-")[0]] += 1
            countries[f'{left["country"]} -> {right["country"]}'] += 1
            for key in distributions:
                distributions[key].append(f[key])
            for side, row in (("left", left), ("right", right)):
                for field, raw in (("name", "business_name"), ("address", "business_address")):
                    lengths[f"{side}_{field}"].append(len(row[raw]))
            # Large hardness means both strings are difficult under generic edit comparison.
            hardness = 1.0 - (0.6 * f["name_edit_similarity"] + 0.4 * f["address_edit_similarity"])
            example = {
                "source1_entity_id": anchor_id, "target_entity_id": target_id,
                "left_name": left["business_name"], "right_name": right["business_name"],
                "left_address": left["business_address"], "right_address": right["business_address"],
                "left_country": left["country"], "right_country": right["country"],
                **{key: f[key] for key in ("name_edit_similarity", "address_edit_similarity", "name_token_jaccard", "address_token_jaccard")},
                "hardness": hardness, "categories": ",".join(flags),
            }
            entry = (hardness, anchor_id, target_id, example)
            if len(heap) < 50:
                heapq.heappush(heap, entry)
            elif entry[:3] > heap[0][:3]:
                heapq.heapreplace(heap, entry)
    if _fingerprint(paths) != inputs:
        raise RuntimeError("Input split changed during oracle analysis; no report published")
    match_counts = [len(labels[key]) for key in anchors]
    sample_ids = "\n".join(sorted(anchors)).encode("utf-8")
    report = {
        "config": config, "inputs": inputs, "training_only": True,
        "sampling": "Uniform seeded reservoir of Source 1 anchors; all positives of sampled anchors included. Positive-pair fractions weight anchors by their true match count.",
        "source1_population": source1_rows, "sampled_anchors": len(anchors),
        "sampled_anchor_ids_sha256": hashlib.sha256(sample_ids).hexdigest(),
        "sampled_singletons": sum(count == 0 for count in match_counts),
        "positive_pairs": positive_count, "needed_unique_targets": len(targets),
        "target_rows_scanned": scanned_target_rows, "positive_pairs_by_source": dict(by_source),
        "positive_country_pairs": dict(countries),
        "match_count_histogram": {str(k): v for k, v in sorted(Counter(match_counts).items())},
        "match_count_quantiles": _quantiles(match_counts),
        "category_counts": dict(sorted(categories.items())),
        "category_fractions": {key: value / positive_count for key, value in sorted(categories.items())} if positive_count else {},
        "similarity_quantiles": {key: _quantiles(value) for key, value in distributions.items()},
        "raw_character_length_quantiles": {key: _quantiles(value) for key, value in lengths.items()},
        "interpretation": [
            "Oracle label inspection only; no candidate recall, model quality or held-out performance has been measured.",
            "Candidate K must exceed needed positive cardinality and absorb distractors; choose using measured retrieval recall, not this report alone.",
            "Missing indicators recognize blank text only in TSV inputs; literal null/none/nan/n-a strings are retained as text and raw inputs are never rewritten.",
            "Script-disjoint flags use coarse Unicode-name groups and are not transliteration or language detection.",
            "Hard examples rank sampled positives by 1 - (0.6*name edit similarity + 0.4*address edit similarity); they are not a random sample.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".oracle-", dir=output_dir) as temp:
        staged_examples = Path(temp) / examples_path.name
        with staged_examples.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_EXAMPLE_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for *_, example in sorted(heap, key=lambda item: item[:3], reverse=True):
                writer.writerow(example)
        report["hard_examples_count"] = len(heap)
        report["hard_examples_sha256"] = _sha256(staged_examples)
        staged_report = Path(temp) / report_path.name
        staged_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(staged_examples, examples_path)
        os.replace(staged_report, report_path)
    return report
