import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

from ber import features
from ber.features import FEATURE_NAMES, features_for_pair, features_for_pairs, summarize_positive_pairs
from ber.text import acronym, fold_diacritics, legal_name_alternates, normalize_text, number_tokens, prepare_record


def record(name="Alpha Ltd", address="12 Main Street", country="France", entity_id="S1-1"):
    return {"entity_id": entity_id, "business_name": name, "business_address": address, "country": country}


class TextTests(unittest.TestCase):
    def test_unicode_and_ampersand_normalization(self):
        self.assertEqual(normalize_text("  ＣＡＦÉ & Fils—Paris\t"), "café and fils paris")
        self.assertEqual(normalize_text("Cafe\u0301"), "café")
        self.assertEqual(fold_diacritics("CAFÉ São"), "cafe sao")
        self.assertIn("राम", normalize_text("राम मार्केटिंग"))

    def test_idempotence(self):
        for text in ("École & Fils", "Alpha, Pvt. Ltd.", "राम कंपनी", "  ００７ / Rue \n"):
            self.assertEqual(normalize_text(normalize_text(text)), normalize_text(text))

    def test_null_markers(self):
        for value in (None, "", " \t ", float("nan"), float("inf")):
            self.assertEqual(normalize_text(value), "")
        for value, expected in (("NULL", "null"), ("None", "none"), ("NaN", "nan"), ("N/A", "n a")):
            self.assertEqual(normalize_text(value), expected)

    def test_legal_alternate_and_acronym(self):
        self.assertEqual(legal_name_alternates("Acme Private Limited"), ("acme pvt ltd", "acme"))
        self.assertEqual(legal_name_alternates("Acme Pvt Ltd"), ("acme pvt ltd", "acme"))
        self.assertEqual(legal_name_alternates("Private Garden"), ("pvt garden", "pvt garden"))
        self.assertEqual(acronym("International Business Machines Corporation"), "ibm")
        self.assertEqual(legal_name_alternates("Limited"), ("ltd", "ltd"))

    def test_unicode_numbers(self):
        self.assertEqual(number_tokens("Unit ００７, ١٢, 003"), frozenset({"7", "12", "3"}))

    def test_raw_record_remains_unchanged(self):
        raw = record(" École & Fils ", "12, Rue de l’Église", None)
        before = copy.deepcopy(raw)
        prepared = prepare_record(raw)
        features_for_pair(raw, raw)
        self.assertEqual(raw, before)
        self.assertEqual(prepared.name.normalized, "école and fils")
        self.assertIs(prepare_record(prepared), prepared)


class FeatureTests(unittest.TestCase):
    def test_france_accent_alternate(self):
        f = features_for_pair(record("Café Lumière"), record("Cafe Lumiere", entity_id="S3-9"))
        self.assertEqual(f["name_exact"], 0)
        self.assertEqual(f["name_folded_exact"], 1)
        self.assertEqual(f["country_equal"], 1)
        self.assertEqual(f["right_source3"], 1)
        self.assertEqual(f["right_source2"], 0)

    def test_empty_pairs_are_not_exact_matches(self):
        f = features_for_pair(record(None, None, None), record("", "", ""))
        for field in ("name", "address"):
            for suffix in ("exact", "folded_exact", "token_jaccard", "edit_similarity", "numbers_equal"):
                self.assertEqual(f[f"{field}_{suffix}"], 0)
            self.assertEqual(f[f"{field}_left_missing"], 1)
        self.assertEqual(f["country_equal"], 0)
        self.assertTrue(all(math.isfinite(v) for v in f.values()))

    def test_house_number_conflict_and_missing_distinguished(self):
        conflict = features_for_pair(record(address="12 Rue Victor"), record(address="99 Rue Victor"))
        self.assertEqual(conflict["address_numbers_conflict"], 1)
        self.assertEqual(conflict["address_numbers_overlap"], 0)
        missing = features_for_pair(record(address="12 Rue Victor"), record(address="Rue Victor"))
        self.assertEqual(missing["address_numbers_conflict"], 0)
        self.assertEqual(missing["address_numbers_both_present"], 0)
        overlap = features_for_pair(record(address="12 Rue Victor 75001"), record(address="99 Rue Victor 75001"))
        self.assertEqual(overlap["address_numbers_equal"], 0)
        self.assertEqual(overlap["address_numbers_overlap"], 1)
        self.assertEqual(overlap["address_numbers_disagree"], 1)
        self.assertAlmostEqual(overlap["address_numbers_jaccard"], 1 / 3)

    def test_token_order_and_legal_alternates(self):
        f = features_for_pair(record("Acme Private Limited", "Rue Victor 12"), record("Acme Pvt Ltd", "12 Victor Rue"))
        self.assertEqual(f["name_legal_exact"], 1)
        self.assertEqual(f["name_core_exact"], 1)
        self.assertEqual(f["address_token_sort_similarity"], 1)
        self.assertEqual(f["address_token_jaccard"], 1)
        self.assertEqual(f["address_exact"], 0)
        f = features_for_pair(record("International Business Machines"), record("IBM"))
        self.assertEqual(f["name_acronym_matches_name"], 1)

    def test_unknown_country_is_generic(self):
        f = features_for_pair(record(country="New Country"), record(country=" new country "))
        self.assertEqual(f["country_equal"], 1)
        f = features_for_pair(record(country="France"), record(country="India"))
        self.assertEqual(f["country_conflict"], 1)

    def test_retrieval_provenance_and_invalid_numbers(self):
        retrieval = {"rank": 2, "score": float("nan"), "candidate_count": -10,
                     "rank_name": 4, "score_address": float("inf"),
                     "provenance": {"name_char": True, "name_exact": False}, "address_exact": True}
        f = features_for_pair(record(), record(), retrieval)
        self.assertEqual(f["retrieval_rank_reciprocal"], 0.5)
        self.assertEqual(f["retrieval_name_rank_reciprocal"], 0.25)
        self.assertEqual(f["retrieval_score"], 0)
        self.assertEqual(f["retrieval_candidate_count_log1p"], 0)
        self.assertEqual(f["retrieval_name_char"], 1)
        self.assertEqual(f["retrieval_address_exact"], 1)
        self.assertEqual(f["retrieval_name_exact"], 0)

    def test_retrieval_channel_aliases(self):
        metadata = {"provenance": ["exact_name", "name_word", "numeric_rescue"],
                    "score_name_char": 0.6, "rank_name_char": 5, "score_name_word": 0.8,
                    "score_address_char": 0.9, "score": 1e300}
        f = features_for_pair(record(), record(), metadata)
        self.assertEqual(f["retrieval_name_exact"], 1)
        self.assertEqual(f["retrieval_name_token"], 1)
        self.assertEqual(f["retrieval_number_block"], 1)
        self.assertEqual(f["retrieval_name_char_score"], 0.6)
        self.assertEqual(f["retrieval_name_char_rank_reciprocal"], 0.2)
        self.assertEqual(f["retrieval_name_word_score"], 0.8)
        self.assertEqual(f["retrieval_address_char_score"], 0.9)
        self.assertTrue(np.isfinite(features_for_pairs([(record(), record(), metadata)])).all())

    def test_feature_order_and_batch_equivalence(self):
        left, right = record(), record(entity_id="S2-8")
        f = features_for_pair(left, right, {"score": 0.8})
        self.assertEqual(tuple(f), FEATURE_NAMES)
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
        batch = features_for_pairs([(left, right, {"score": 0.8}), (prepare_record(left), prepare_record(right))])
        self.assertEqual(batch.shape, (2, len(FEATURE_NAMES)))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_allclose(batch[0], list(f.values()), rtol=1e-6)
        self.assertEqual(features_for_pairs([]).shape, (0, len(FEATURE_NAMES)))
        self.assertEqual(features_for_pairs([(left, right)], as_array=False), [features_for_pair(left, right)])
        with self.assertRaises(ValueError):
            features_for_pairs([(left,)])

    def test_edit_fallback_is_exact_levenshtein(self):
        with patch.object(features, "_Levenshtein", None):
            self.assertAlmostEqual(features._edit_similarity("kitten", "sitting"), 4 / 7)
            self.assertAlmostEqual(features._edit_similarity("café", "cafe"), 0.75)
            self.assertEqual(features._edit_similarity("", ""), 0)
            self.assertEqual(features._edit_similarity("abc", "abc"), 1)

    def test_bounded_positive_oracle_summary(self):
        pairs = [(record(), record()) for _ in range(12)]
        result = summarize_positive_pairs(iter(pairs), max_pairs=3, max_scan=7, seed=42)
        self.assertEqual(result["input_pairs_scanned"], 7)
        self.assertEqual(result["sampled_positive_pairs"], 3)
        self.assertEqual(result["counts"]["name_exact"], 3)
        self.assertEqual(result["fractions"]["address_exact"], 1)
        self.assertEqual(result, summarize_positive_pairs(iter(pairs), max_pairs=3, max_scan=7, seed=42))
        self.assertEqual(summarize_positive_pairs([])["sampled_positive_pairs"], 0)
        with self.assertRaises(ValueError):
            summarize_positive_pairs([], max_pairs=0)


if __name__ == "__main__":
    unittest.main()
