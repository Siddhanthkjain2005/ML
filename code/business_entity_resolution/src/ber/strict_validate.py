"""Bounded-memory validation of the complete official two-file output contract.

This supplements, and never changes, the supplied official validator. Source
membership, seen rows, and candidate lists are indexed in an ephemeral SQLite
database, not retained as millions of Python strings. No global one-to-one
assignment constraint is imposed on targets. Candidate provenance (the exact
inputs actually scored by a model) still requires pipeline reconciliation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids"
ID_PATTERN = re.compile(r"S[123]-[0-9]+\Z")
TARGET_PATTERN = re.compile(r"S[23]-[0-9]+\Z")


class _Issues:
    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0
        self.messages: list[str] = []

    def add(self, message: str) -> None:
        self.count += 1
        if len(self.messages) < self.limit:
            self.messages.append(message)


def _load_source(
    connection: sqlite3.Connection, path: Path, source: int, issues: _Issues,
) -> tuple[int, int]:
    count = france_count = 0
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            if next(reader, None) != SOURCE_HEADER:
                issues.add(f"{path.name}: expected exact four-column source header")
                return 0, 0
            for row in reader:
                count += 1
                if len(row) != 4:
                    issues.add(f"{path.name}:{reader.line_num}: source row must have four columns")
                    continue
                entity_id, _, _, country = row
                if not ID_PATTERN.fullmatch(entity_id) or not entity_id.startswith(f"S{source}-"):
                    issues.add(f"{path.name}:{reader.line_num}: invalid source ID {entity_id!r}")
                    continue
                try:
                    if source == 1:
                        connection.execute("INSERT INTO s1 VALUES (?, ?)", (entity_id, country))
                        france_count += country == "France"
                    else:
                        connection.execute("INSERT INTO targets VALUES (?)", (entity_id,))
                except sqlite3.IntegrityError:
                    issues.add(f"{path.name}:{reader.line_num}: duplicate source ID {entity_id!r}")
    except (OSError, UnicodeError, csv.Error) as exc:
        issues.add(f"Cannot read source {path}: {exc}")
    return count, france_count


def _remove_line_ending(line: str) -> str:
    if line.endswith("\n"):
        line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
    return line


def _validate_output(
    connection: sqlite3.Connection, path: Path, *, candidate: bool,
    required_rows: int, issues: _Issues,
) -> dict[str, int]:
    table = "candidates" if candidate else "matching"
    header = CANDIDATE_HEADER if candidate else MATCHING_HEADER
    rows = pairs = empty_rows = 0
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            if _remove_line_ending(handle.readline()) != header:
                issues.add(f"{path.name}: expected exact header {header!r}")
                return {"rows": 0, "pairs": 0, "empty_rows": 0}
            for line_number, line in enumerate(handle, 2):
                rows += 1
                cells = _remove_line_ending(line).split("\t")
                if len(cells) != 2:
                    issues.add(f"{path.name}:{line_number}: expected exactly two literal-tab-separated columns")
                    continue
                s1, value = cells
                if not ID_PATTERN.fullmatch(s1) or not s1.startswith("S1-"):
                    issues.add(f"{path.name}:{line_number}: invalid Source1 ID {s1!r}")
                    continue
                if connection.execute("SELECT 1 FROM s1 WHERE id=?", (s1,)).fetchone() is None:
                    issues.add(f"{path.name}:{line_number}: unknown Source1 ID {s1!r}")
                try:
                    if candidate:
                        connection.execute("INSERT INTO candidates VALUES (?, ?)", (s1, value))
                    else:
                        connection.execute("INSERT INTO matching VALUES (?)", (s1,))
                except sqlite3.IntegrityError:
                    issues.add(f"{path.name}:{line_number}: duplicate Source1 row {s1!r}")
                if value == "":
                    empty_rows += 1
                    ids: list[str] = []
                else:
                    ids = value.split(",")
                pairs += len(ids)
                distinct = set(ids)
                if len(distinct) != len(ids):
                    issues.add(f"{path.name}:{line_number}: duplicate ID within list for {s1}")
                valid_syntax = []
                for target in sorted(distinct):
                    if not TARGET_PATTERN.fullmatch(target):
                        issues.add(f"{path.name}:{line_number}: invalid target ID {target!r}; empty lists must be literal empty strings")
                    else:
                        valid_syntax.append(target)
                # Chunked IN queries avoid SQLite parameter limits and per-ID calls.
                for offset in range(0, len(valid_syntax), 500):
                    chunk = valid_syntax[offset:offset + 500]
                    marks = ",".join("?" for _ in chunk)
                    found = {row[0] for row in connection.execute(
                        f"SELECT id FROM targets WHERE id IN ({marks})", chunk,
                    )}
                    for target in sorted(set(chunk) - found):
                        issues.add(f"{path.name}:{line_number}: target not present in test sources: {target}")
                if not candidate and distinct:
                    candidate_row = connection.execute("SELECT ids FROM candidates WHERE s1=?", (s1,)).fetchone()
                    candidate_set = set(candidate_row[0].split(",")) if candidate_row and candidate_row[0] else set()
                    missing = distinct - candidate_set
                    if missing:
                        issues.add(f"{path.name}:{line_number}: matched IDs outside final candidates for {s1}: {sorted(missing)[:5]}")
    except (OSError, UnicodeError) as exc:
        issues.add(f"Cannot read output {path}: {exc}")
    if rows != required_rows:
        issues.add(f"{path.name}: row count {rows} differs from Source1 row count {required_rows}")
    missing_count = connection.execute(
        f"SELECT COUNT(*) FROM s1 LEFT JOIN {table} ON s1.id={table}.s1 WHERE {table}.s1 IS NULL"
    ).fetchone()[0]
    if missing_count:
        examples = [row[0] for row in connection.execute(
            f"SELECT s1.id FROM s1 LEFT JOIN {table} ON s1.id={table}.s1 "
            f"WHERE {table}.s1 IS NULL ORDER BY s1.id LIMIT 5"
        )]
        issues.add(f"{path.name}: missing {missing_count} Source1 rows, examples {examples}")
    missing_france = connection.execute(
        f"SELECT COUNT(*) FROM s1 LEFT JOIN {table} ON s1.id={table}.s1 "
        f"WHERE s1.country='France' AND {table}.s1 IS NULL"
    ).fetchone()[0]
    if missing_france:
        issues.add(f"{path.name}: missing {missing_france} France Source1 rows")
    return {"rows": rows, "pairs": pairs, "empty_rows": empty_rows}


def validate_submission(
    matching_path: str | Path, candidate_path: str | Path, test_dir: str | Path,
    *, scratch_dir: str | Path | None = None, max_errors: int = 100,
) -> dict[str, Any]:
    """Validate both outputs against real source files; any issue fails.

    Uses a temporary on-disk SQLite database with a bounded 32 MiB page cache.
    The caller may select a scratch disk; the database is removed on completion.
    At most max_errors details are retained, but error_count counts all detected
    issues. One ID list is held in memory at a time. Row order is unrestricted.
    """
    if max_errors < 1:
        raise ValueError("max_errors must be positive")
    issues = _Issues(max_errors)
    counts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ber_strict_", dir=scratch_dir) as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "validation.sqlite"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("PRAGMA temp_store=FILE")
            connection.executescript(
                "CREATE TABLE s1 (id TEXT PRIMARY KEY, country TEXT NOT NULL) WITHOUT ROWID;"
                "CREATE TABLE targets (id TEXT PRIMARY KEY) WITHOUT ROWID;"
                "CREATE TABLE candidates (s1 TEXT PRIMARY KEY, ids TEXT NOT NULL) WITHOUT ROWID;"
                "CREATE TABLE matching (s1 TEXT PRIMARY KEY) WITHOUT ROWID;"
            )
            for source in (1, 2, 3):
                rows, france = _load_source(
                    connection, Path(test_dir) / f"test_source{source}.tsv", source, issues,
                )
                counts[f"source{source}_rows"] = rows
                if source == 1:
                    counts["france_source1_rows"] = france
                    if rows == 0:
                        issues.add("test_source1.tsv: evaluation universe must not be empty")
                connection.commit()
            if not issues.count:
                counts["candidate"] = _validate_output(
                    connection, Path(candidate_path), candidate=True,
                    required_rows=counts["source1_rows"], issues=issues,
                )
                connection.commit()
                counts["matching"] = _validate_output(
                    connection, Path(matching_path), candidate=False,
                    required_rows=counts["source1_rows"], issues=issues,
                )
                connection.commit()
        except sqlite3.Error as exc:
            issues.add(f"SQLite validation failure: {exc}")
        finally:
            connection.close()
    return {
        "valid": issues.count == 0,
        "error_count": issues.count,
        "errors": issues.messages,
        "errors_truncated": issues.count > len(issues.messages),
        "counts": counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--scratch-dir")
    parser.add_argument("--max-errors", type=int, default=100)
    args = parser.parse_args(argv)
    result = validate_submission(
        args.matching, args.candidate, args.test_dir,
        scratch_dir=args.scratch_dir, max_errors=args.max_errors,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
