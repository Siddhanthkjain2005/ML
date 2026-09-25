"""Stream the labeled archive into reproducible, entity-grouped data splits.

Only the four training TSV members are opened. Positive S2/S3 rows inherit
their S1 owner's split; unlinked targets use their own stable ID hash. No
business strings are normalized or interpreted as missing values here.
"""

from __future__ import annotations

import array
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Iterable
import zipfile

import numpy as np


SPLITS = ("train", "calibration", "validation")
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
TRAIN_PREFIX = "student_resource/dataset/train/"
MEMBERS = {
    "ground_truth": TRAIN_PREFIX + "train_ground_truth.tsv",
    **{f"source{i}": TRAIN_PREFIX + f"train_source{i}.tsv" for i in (1, 2, 3)},
}
_ID_RE = re.compile(r"S([123])-([0-9]+)\Z")
_UINT64_MAX = (1 << 64) - 1


def _numeric_id(value: str, source: int) -> int:
    match = _ID_RE.fullmatch(value)
    if match is None or int(match[1]) != source:
        raise ValueError(f"Expected S{source}-<integer> entity ID, got {value!r}")
    number = int(match[2])
    if number > _UINT64_MAX or value != f"S{source}-{number}":
        raise ValueError(f"Noncanonical or out-of-range entity ID: {value!r}")
    return number


def _boundaries(fractions: tuple[float, float, float]) -> tuple[int, int]:
    if len(fractions) != 3 or any(not math.isfinite(x) or x < 0 for x in fractions):
        raise ValueError("fractions must contain three finite, nonnegative values")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("fractions must sum to one")
    return int(fractions[0] * (1 << 64)), int(sum(fractions[:2]) * (1 << 64))


def _fold_for_id(entity_id: str, seed: int, boundaries: tuple[int, int]) -> int:
    payload = f"{seed}\0{entity_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return 0 if value < boundaries[0] else 1 if value < boundaries[1] else 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(archive: zipfile.ZipFile, member: str, columns: tuple[str, ...]):
    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t", strict=True)
            header = next(reader, None)
            if header != list(columns):
                raise ValueError(f"Unexpected columns in {member}: {header!r}")
            for line, row in enumerate(reader, 2):
                if len(row) != len(columns):
                    raise ValueError(f"{member}, record {line}: expected {len(columns)} fields")
                yield row


def _batches(rows: Iterable[list[str]], size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _sorted_registry(ids: array.array, folds: array.array, source: int):
    values = np.asarray(ids, dtype=np.uint64)
    assignments = np.asarray(folds, dtype=np.uint8)
    order = np.argsort(values, kind="stable")
    values, assignments = values[order], assignments[order]
    if len(values) > 1:
        repeated = np.flatnonzero(values[1:] == values[:-1])
        if len(repeated):
            number = int(values[int(repeated[0])])
            reason = "Duplicate S1 ground-truth anchor" if source == 1 else "Target has multiple S1 owners"
            raise ValueError(f"{reason}: S{source}-{number}")
    return values, assignments


def _split_stats() -> dict:
    return {
        "anchor_count": 0,
        "positive_link_count": 0,
        "singleton_count": 0,
        "match_count_histogram": {},
        "positive_target_counts": {"source2": 0, "source3": 0},
        "unlinked_target_counts": {"source2": 0, "source3": 0},
        "source_rows": {f"source{i}": 0 for i in (1, 2, 3)},
        "country_counts": {f"source{i}": {} for i in (1, 2, 3)},
    }


def _increment(counts: dict, key: str, amount: int = 1) -> None:
    counts[key] = counts.get(key, 0) + amount


def prepare_splits(
    zip_path: str | Path,
    output_dir: str | Path,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    *,
    input_sha256: str | None = None,
    batch_size: int = 50_000,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Create train/calibration/validation TSVs and return their manifest.

    Output is published atomically only after reference and uniqueness checks
    pass. Existing nonempty output directories are never overwritten. A known
    archive SHA256 may be provided to avoid hashing its bytes a second time;
    the manifest records whether this checksum was supplied or computed.

    Fractions are hash thresholds, not quotas, so observed proportions need
    not equal requested fractions exactly. Split assignment is independent
    of input row ordering, process hash randomization, and batch size.
    """
    zip_path, output_dir = Path(zip_path).resolve(), Path(output_dir).resolve()
    fractions = tuple(float(value) for value in fractions)
    boundaries = _boundaries(fractions)
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    if input_sha256 is not None and re.fullmatch(r"[0-9a-fA-F]{64}", input_sha256) is None:
        raise ValueError("input_sha256 must be a 64-character hexadecimal SHA256")
    before = zip_path.stat()
    checksum = input_sha256.lower() if input_sha256 is not None else _sha256(zip_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stats = {split: _split_stats() for split in SPLITS}

    def emit(**event):
        if progress is not None:
            progress(event)

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.prepare-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "dataset"
        for split in SPLITS:
            (stage / split).mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            for member in MEMBERS.values():
                if names.count(member) != 1:
                    raise ValueError(f"Expected exactly one training archive member: {member}")
            member_metadata = {
                key: {"member": member, "bytes": archive.getinfo(member).file_size,
                      "crc32": f"{archive.getinfo(member).CRC:08x}"}
                for key, member in MEMBERS.items()
            }
            raw_ids = {i: array.array("Q") for i in (1, 2, 3)}
            raw_folds = {i: array.array("B") for i in (1, 2, 3)}
            files = [(stage / split / "ground_truth.tsv").open("w", encoding="utf-8", newline="") for split in SPLITS]
            try:
                writers = [csv.writer(stream, delimiter="\t", lineterminator="\n") for stream in files]
                for writer in writers:
                    writer.writerow(GROUND_TRUTH_COLUMNS)
                for row in _rows(archive, MEMBERS["ground_truth"], GROUND_TRUTH_COLUMNS):
                    anchor, labels = row
                    anchor_number = _numeric_id(anchor, 1)
                    fold = _fold_for_id(anchor, seed, boundaries)
                    raw_ids[1].append(anchor_number)
                    raw_folds[1].append(fold)
                    targets = labels.split(",") if labels else []
                    if len(set(targets)) != len(targets):
                        raise ValueError(f"Duplicate target in ground-truth row for {anchor}")
                    split_stats = stats[SPLITS[fold]]
                    split_stats["anchor_count"] += 1
                    split_stats["positive_link_count"] += len(targets)
                    split_stats["singleton_count"] += not targets
                    _increment(split_stats["match_count_histogram"], str(len(targets)))
                    for target in targets:
                        match = _ID_RE.fullmatch(target)
                        if match is None or match[1] not in ("2", "3"):
                            raise ValueError(f"Invalid positive target for {anchor}: {target!r}")
                        source = int(match[1])
                        raw_ids[source].append(_numeric_id(target, source))
                        raw_folds[source].append(fold)
                        split_stats["positive_target_counts"][f"source{source}"] += 1
                    writers[fold].writerow(row)
            finally:
                for stream in files:
                    stream.close()

            registries = {}
            for source in (1, 2, 3):
                registries[source] = _sorted_registry(raw_ids.pop(source), raw_folds.pop(source), source)
            emit(stage="ownership_ready", anchors=len(registries[1][0]),
                 positive_links=len(registries[2][0]) + len(registries[3][0]))

            for source in (1, 2, 3):
                key = f"source{source}"
                owner_ids, owner_folds = registries[source]
                seen = np.zeros(len(owner_ids), dtype=np.bool_)
                all_ids = array.array("Q")
                total_rows = 0
                files = [(stage / split / f"{key}.tsv").open("w", encoding="utf-8", newline="") for split in SPLITS]
                try:
                    writers = [csv.writer(stream, delimiter="\t", lineterminator="\n") for stream in files]
                    for writer in writers:
                        writer.writerow(SOURCE_COLUMNS)
                    for batch in _batches(_rows(archive, MEMBERS[key], SOURCE_COLUMNS), batch_size):
                        numbers = np.fromiter((_numeric_id(row[0], source) for row in batch), dtype=np.uint64, count=len(batch))
                        all_ids.frombytes(numbers.tobytes())
                        positions = np.searchsorted(owner_ids, numbers)
                        matched = positions < len(owner_ids)
                        matched[matched] = owner_ids[positions[matched]] == numbers[matched]
                        if source == 1 and not matched.all():
                            bad = batch[int(np.flatnonzero(~matched)[0])][0]
                            raise ValueError(f"Source1 row lacks ground truth: {bad}")
                        folds = np.empty(len(batch), dtype=np.uint8)
                        folds[matched] = owner_folds[positions[matched]]
                        seen[positions[matched]] = True
                        for index in np.flatnonzero(~matched):
                            folds[index] = _fold_for_id(batch[int(index)][0], seed, boundaries)
                        for row, fold, linked in zip(batch, folds, matched):
                            split_stats = stats[SPLITS[int(fold)]]
                            writers[int(fold)].writerow(row)
                            split_stats["source_rows"][key] += 1
                            _increment(split_stats["country_counts"][key], row[3])
                            if source != 1 and not linked:
                                split_stats["unlinked_target_counts"][key] += 1
                        total_rows += len(batch)
                        if total_rows % 1_000_000 < len(batch):
                            emit(stage="source_progress", source=key, rows=total_rows)
                finally:
                    for stream in files:
                        stream.close()
                values = np.asarray(all_ids, dtype=np.uint64)
                values.sort()
                if len(values) > 1 and np.any(values[1:] == values[:-1]):
                    raise ValueError(f"Duplicate entity IDs in {MEMBERS[key]}")
                if not seen.all():
                    missing = int(owner_ids[int(np.flatnonzero(~seen)[0])])
                    raise ValueError(f"Ground-truth reference absent from source{source}: S{source}-{missing}")
                emit(stage="source_complete", source=key, rows=total_rows)
                del all_ids, values, seen

        after = zip_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Input archive changed while splits were being prepared")
        for split in SPLITS:
            stats[split]["files"] = {
                **{f"source{i}": str(output_dir / split / f"source{i}.tsv") for i in (1, 2, 3)},
                "ground_truth": str(output_dir / split / "ground_truth.tsv"),
            }
        manifest = {
            "schema_version": 1,
            "input": {"path": str(zip_path), "bytes": before.st_size, "sha256": checksum,
                      "sha256_source": "provided" if input_sha256 is not None else "computed",
                      "training_members": member_metadata},
            "split_config": {"names": list(SPLITS), "fractions": list(fractions), "seed": seed,
                             "algorithm": "blake2b-64-big-endian(seed + NUL + entity_id)",
                             "positive_targets": "inherit owning S1 split",
                             "unlinked_targets": "stable hash of own entity_id",
                             "normalization": "none; raw UTF-8 field strings preserved"},
            "output_dir": str(output_dir),
            "splits": stats,
            "totals": {
                "anchor_count": sum(s["anchor_count"] for s in stats.values()),
                "positive_link_count": sum(s["positive_link_count"] for s in stats.values()),
                "singleton_count": sum(s["singleton_count"] for s in stats.values()),
                "source_rows": {f"source{i}": sum(s["source_rows"][f"source{i}"] for s in stats.values()) for i in (1, 2, 3)},
            },
            "checks": {"unique_source_ids": True, "unique_ground_truth_anchors": True,
                       "single_owner_per_positive_target": True, "all_references_exist": True,
                       "all_source1_rows_have_ground_truth": True,
                       "test_members_opened": False},
        }
        with (stage / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(stage, output_dir)
    emit(stage="complete", manifest=str(output_dir / "manifest.json"))
    return manifest
