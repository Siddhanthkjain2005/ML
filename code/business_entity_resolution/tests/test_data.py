import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from ber.data import GROUND_TRUTH_COLUMNS, MEMBERS, SOURCE_COLUMNS, SPLITS, prepare_splits


def make_archive(path, sources=None, truth=None):
    if sources is None:
        sources = {
            1: [[f"S1-{i}", "NA" if i == 1 else "नमस्ते", "  literal address  ", "India"] for i in range(1, 31)],
            2: [[f"S2-{i}", "NULL", "" if i == 1 else "nan", "India"] for i in range(1, 41)],
            3: [[f"S3-{i}", "Business", "Street\twith quoted tab", "US"] for i in range(1, 41)],
        }
    if truth is None:
        truth = [[f"S1-{i}", f"S2-{i},S3-{i}" if i <= 20 else ""] for i in range(1, 31)]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, rows, columns in [("ground_truth", truth, GROUND_TRUTH_COLUMNS)] + [
            (f"source{i}", sources[i], SOURCE_COLUMNS) for i in (1, 2, 3)
        ]:
            output = io.StringIO(newline="")
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
            archive.writestr(MEMBERS[key], output.getvalue().encode("utf-8"))
        # Decoding or parsing this test member would fail. It must never be opened.
        archive.writestr("student_resource/dataset/test/test_source1.tsv", b"\xff\x00do not read test")
    return sources, truth


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def assignment(manifest, source):
    return {
        row["entity_id"]: split
        for split in SPLITS
        for row in rows(manifest["splits"][split]["files"][f"source{source}"])
    }


class PrepareSplitsTests(unittest.TestCase):
    def test_group_ownership_disjointness_raw_values_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.zip"
            sources, truth = make_archive(archive)
            events = []
            result = prepare_splits(archive, root / "prepared", seed=91, batch_size=3, progress=events.append)
            mapped = {i: assignment(result, i) for i in (1, 2, 3)}
            for anchor, labels in truth:
                for target in labels.split(",") if labels else []:
                    self.assertEqual(mapped[1][anchor], mapped[int(target[1])][target])
            restored_truth = []
            for split in SPLITS:
                split_truth = rows(result["splits"][split]["files"]["ground_truth"])
                for row in split_truth:
                    self.assertEqual(mapped[1][row["source1_entity_id"]], split)
                    restored_truth.append([row[column] for column in GROUND_TRUTH_COLUMNS])
                self.assertEqual(result["splits"][split]["anchor_count"], len(split_truth))
                self.assertEqual(
                    result["splits"][split]["singleton_count"],
                    sum(not row["matched_entity_ids"] for row in split_truth),
                )
            self.assertEqual(sorted(restored_truth), sorted(truth))
            for source in (1, 2, 3):
                restored = [
                    [row[column] for column in SOURCE_COLUMNS]
                    for split in SPLITS
                    for row in rows(result["splits"][split]["files"][f"source{source}"])
                ]
                self.assertEqual(sorted(restored), sorted(sources[source]))
                self.assertEqual(len(mapped[source]), len(sources[source]))
            self.assertEqual(result["totals"]["anchor_count"], 30)
            self.assertEqual(result["totals"]["positive_link_count"], 40)
            self.assertEqual(result["totals"]["singleton_count"], 10)
            self.assertEqual(sum(s["unlinked_target_counts"]["source2"] for s in result["splits"].values()), 20)
            self.assertEqual(result["input"]["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertFalse(result["checks"]["test_members_opened"])
            self.assertEqual(events[-1]["stage"], "complete")
            self.assertEqual(json.loads((root / "prepared" / "manifest.json").read_text()), result)

    def test_reproducible_across_row_order_and_batch_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, truth = make_archive(root / "one.zip")
            make_archive(root / "two.zip", {k: list(reversed(v)) for k, v in sources.items()}, list(reversed(truth)))
            first = prepare_splits(root / "one.zip", root / "one", fractions=(0.5, 0.25, 0.25), seed=12, batch_size=1)
            second = prepare_splits(root / "two.zip", root / "two", fractions=(0.5, 0.25, 0.25), seed=12, batch_size=17)
            for source in (1, 2, 3):
                self.assertEqual(assignment(first, source), assignment(second, source))
            for entity_id in ("S2-40", "S3-40"):
                value = int.from_bytes(hashlib.blake2b(f"12\0{entity_id}".encode(), digest_size=8).digest(), "big")
                expected = SPLITS[0 if value < (1 << 63) else 1 if value < 3 * (1 << 62) else 2]
                self.assertEqual(assignment(first, int(entity_id[1]))[entity_id], expected)

    def test_rejects_multiple_owners_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root / "input.zip", truth=[["S1-1", "S2-1"], ["S1-2", "S2-1"]])
            with self.assertRaisesRegex(ValueError, "multiple S1 owners"):
                prepare_splits(root / "input.zip", root / "prepared")
            self.assertFalse((root / "prepared").exists())

    def test_rejects_dangling_reference_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = [[f"S1-{i}", "S2-999" if i == 1 else ""] for i in range(1, 31)]
            make_archive(root / "input.zip", truth=truth)
            with self.assertRaisesRegex(ValueError, "reference absent"):
                prepare_splits(root / "input.zip", root / "prepared")
            self.assertFalse((root / "prepared").exists())

    def test_rejects_duplicate_source_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, truth = make_archive(root / "base.zip")
            sources[2].append(sources[2][-1])
            make_archive(root / "duplicate.zip", sources, truth)
            with self.assertRaisesRegex(ValueError, "Duplicate entity IDs"):
                prepare_splits(root / "duplicate.zip", root / "prepared")
            self.assertFalse((root / "prepared").exists())

    def test_no_overwrite_and_supplied_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root / "input.zip")
            digest = hashlib.sha256((root / "input.zip").read_bytes()).hexdigest()
            result = prepare_splits(root / "input.zip", root / "prepared", input_sha256=digest)
            self.assertEqual(result["input"]["sha256_source"], "provided")
            with self.assertRaises(FileExistsError):
                prepare_splits(root / "input.zip", root / "prepared")


if __name__ == "__main__":
    unittest.main()
