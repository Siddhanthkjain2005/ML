"""Tiny-fixture tests for requirements missing from the official validator."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ber.strict_validate import validate_submission


class StrictValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.test_dir = self.root / "test"
        self.test_dir.mkdir()
        self.matching = self.root / "matching_results.tsv"
        self.candidate = self.root / "candidate_pairs.tsv"
        source_header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        (self.test_dir / "test_source1.tsv").write_text(
            source_header + "S1-001\tAlpha\tMain\tUS\nS1-002\tBeta\tRue\tFrance\n", encoding="utf-8"
        )
        (self.test_dir / "test_source2.tsv").write_text(
            source_header + "S2-001\tAlpha\t\tUS\nS2-002\tBeta\tRue\tFrance\n", encoding="utf-8"
        )
        (self.test_dir / "test_source3.tsv").write_text(
            source_header + "S3-001\tAlpha\tMain\tUS\n", encoding="utf-8"
        )
        self.matching_text = "source1_entity_id\tmatched_entity_ids\nS1-001\tS2-001,S3-001\nS1-002\t\n"
        self.candidate_text = "source1_entity_id\tcandidate_entity_ids\nS1-002\t\nS1-001\tS2-001,S3-001\n"
        self.matching.write_text(self.matching_text, encoding="utf-8")
        self.candidate.write_text(self.candidate_text, encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, **kwargs):
        return validate_submission(self.matching, self.candidate, self.test_dir, scratch_dir=self.root, **kwargs)

    def test_valid_full_contract_with_arbitrary_row_order(self):
        result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["counts"]["france_source1_rows"], 1)
        self.assertEqual(result["counts"]["matching"]["pairs"], 2)
        self.assertFalse(list(self.root.glob("ber_strict_*")))

    def test_both_output_files_required(self):
        self.candidate.unlink()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("Cannot read output" in e for e in result["errors"]))

    def test_source_files_required_without_optional_fallback(self):
        (self.test_dir / "test_source3.tsv").unlink()
        self.assertFalse(self.validate()["valid"])

    def test_unknown_target_rejected(self):
        self.matching.write_text(self.matching_text.replace("S3-001", "S3-999"))
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("not present in test sources" in e for e in result["errors"]))

    def test_matched_subset_is_mandatory(self):
        self.candidate.write_text(self.candidate_text.replace("S2-001,S3-001", "S2-001"))
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("outside final candidates" in e for e in result["errors"]))

    def test_missing_france_anchor_is_rejected(self):
        self.matching.write_text(self.matching_text.replace("S1-002\t\n", ""))
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("France" in e for e in result["errors"]))

    def test_duplicate_rows_in_either_file_rejected(self):
        for path, original in [(self.matching, self.matching_text), (self.candidate, self.candidate_text)]:
            with self.subTest(path=path.name):
                path.write_text(original + "S1-001\t\n")
                result = self.validate()
                self.assertFalse(result["valid"])
                self.assertTrue(any("duplicate Source1 row" in e for e in result["errors"]))
                path.write_text(original)

    def test_duplicate_lists_in_either_file_rejected(self):
        for path, original in [(self.matching, self.matching_text), (self.candidate, self.candidate_text)]:
            with self.subTest(path=path.name):
                path.write_text(original.replace("S2-001,S3-001", "S2-001,S2-001"))
                result = self.validate()
                self.assertFalse(result["valid"])
                self.assertTrue(any("duplicate ID within list" in e for e in result["errors"]))
                path.write_text(original)

    def test_exact_headers_and_column_count(self):
        malformed = [
            self.matching_text.replace("source1_entity_id", "SOURCE1_ENTITY_ID", 1),
            self.matching_text.replace("source1_entity_id", " source1_entity_id", 1),
            self.matching_text.replace("S2-001,S3-001", "S2-001\tS3-001"),
            self.matching_text.replace("\t", ","),
            "\ufeff" + self.matching_text,
            self.matching_text + "\n",
        ]
        for text in malformed:
            with self.subTest(text=text):
                self.matching.write_text(text, encoding="utf-8")
                self.assertFalse(self.validate()["valid"])

    def test_id_syntax_and_literal_empty(self):
        bad_ids = ["S1-001", "S2-a", "S2-١", "S2-001,", " S2-001", "S2-001 ", "NaN", "nan", "null", '""', " "]
        for bad in bad_ids:
            with self.subTest(bad=bad):
                self.matching.write_text(self.matching_text.replace("S2-001,S3-001", bad))
                self.assertFalse(self.validate()["valid"])

    def test_unknown_and_malformed_s1_rejected(self):
        for bad in ["S1-999", "S1-abc", " S1-001", "S2-001"]:
            with self.subTest(bad=bad):
                self.matching.write_text(self.matching_text.replace("S1-001", bad))
                self.assertFalse(self.validate()["valid"])

    def test_cross_row_target_reuse_not_banned(self):
        self.matching.write_text(self.matching_text.replace("S1-002\t\n", "S1-002\tS2-001\n"))
        self.candidate.write_text(self.candidate_text.replace("S1-002\t\n", "S1-002\tS2-001\n"))
        self.assertTrue(self.validate()["valid"])

    def test_crlf_and_no_final_newline_supported(self):
        self.matching.write_bytes(self.matching_text.rstrip("\n").replace("\n", "\r\n").encode())
        self.assertTrue(self.validate()["valid"])

    def test_duplicate_or_wrong_prefix_source_ids_fail(self):
        path = self.test_dir / "test_source2.tsv"
        original = path.read_text()
        path.write_text(original + "S2-001\tAgain\tAddress\tUS\n")
        self.assertFalse(self.validate()["valid"])
        path.write_text(original.replace("S2-001", "S3-001"))
        self.assertFalse(self.validate()["valid"])

    def test_empty_source1_universe_rejected(self):
        source = self.test_dir / "test_source1.tsv"
        source.write_text(source.read_text().splitlines()[0] + "\n")
        self.matching.write_text(self.matching_text.splitlines()[0] + "\n")
        self.candidate.write_text(self.candidate_text.splitlines()[0] + "\n")
        self.assertFalse(self.validate()["valid"])

    def test_detailed_errors_capped_without_suppressing_failure_count(self):
        self.matching.write_text(self.matching_text.replace("S2-001,S3-001", "NaN,null,S2-bad,S1-001"))
        result = self.validate(max_errors=2)
        self.assertFalse(result["valid"])
        self.assertEqual(len(result["errors"]), 2)
        self.assertGreater(result["error_count"], 2)
        self.assertTrue(result["errors_truncated"])

    def test_cli_exit_status(self):
        command = [sys.executable, "-m", "ber.strict_validate", "--matching", str(self.matching),
                   "--candidate", str(self.candidate), "--test-dir", str(self.test_dir)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.matching.write_text(self.matching_text.replace("S2-001,S3-001", "NaN"))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
