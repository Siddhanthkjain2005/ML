"""Finite, ordered pair features with no fitting or country-specific rules."""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .text import PreparedRecord, PreparedText, prepare_record

try:
    from rapidfuzz.distance import Levenshtein as _Levenshtein
except ImportError:  # The exact fallback has the same numerical definition.
    _Levenshtein = None

PROVENANCE_NAMES = (
    "name_exact", "address_exact", "name_char", "address_char", "name_token",
    "address_token", "number_block", "sorted_neighborhood",
)
_PROVENANCE_ALIASES = {
    "exact_name": "name_exact", "exact_address": "address_exact",
    "name_word": "name_token", "address_word": "address_token", "numeric_rescue": "number_block",
}
_RETRIEVAL_CHANNELS = ("name_char", "name_word", "address_char", "address_word")
_TEXT_FEATURES = (
    "exact", "folded_exact", "token_jaccard", "token_containment", "token_overlap",
    "length_ratio", "token_count_ratio", "edit_similarity", "token_sort_similarity",
    "numbers_both_present", "numbers_equal", "numbers_overlap", "numbers_conflict", "numbers_disagree", "numbers_jaccard",
    "left_missing", "right_missing",
)
FEATURE_NAMES = tuple(f"{field}_{name}" for field in ("name", "address") for name in _TEXT_FEATURES) + (
    "name_legal_exact", "name_core_exact", "name_acronym_exact", "name_acronym_matches_name",
    "country_equal", "country_conflict", "country_left_missing", "country_right_missing",
    "left_source1", "right_source2", "right_source3",
    "retrieval_rank_reciprocal", "retrieval_score", "retrieval_candidate_count_log1p",
    "retrieval_name_rank_reciprocal", "retrieval_address_rank_reciprocal",
    "retrieval_name_score", "retrieval_address_score",
) + tuple(f"retrieval_{name}" for name in PROVENANCE_NAMES) + tuple(
    f"retrieval_{channel}_{suffix}" for channel in _RETRIEVAL_CHANNELS for suffix in ("score", "rank_reciprocal")
)


def _ratio(left: int, right: int) -> float:
    return min(left, right) / max(left, right) if left and right else 0.0


def _equal(left: str, right: str) -> float:
    return float(bool(left) and left == right)


def _edit_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if _Levenshtein is not None:
        return float(_Levenshtein.normalized_similarity(left, right))
    # Standard Levenshtein dynamic program, O(min(lengths)) memory.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _text_features(left: PreparedText, right: PreparedText) -> tuple[float, ...]:
    overlap = len(left.tokens & right.tokens)
    union = len(left.tokens | right.tokens)
    number_overlap = left.numbers & right.numbers
    both_numbers = bool(left.numbers and right.numbers)
    return (
        _equal(left.normalized, right.normalized), _equal(left.folded, right.folded),
        overlap / union if union else 0.0,
        overlap / min(len(left.tokens), len(right.tokens)) if left.tokens and right.tokens else 0.0,
        float(overlap), _ratio(len(left.normalized), len(right.normalized)),
        _ratio(len(left.tokens), len(right.tokens)),
        _edit_similarity(left.normalized, right.normalized),
        _edit_similarity(left.sorted_tokens, right.sorted_tokens),
        float(both_numbers), float(both_numbers and left.numbers == right.numbers),
        float(bool(number_overlap)), float(both_numbers and not number_overlap),
        float(both_numbers and left.numbers != right.numbers),
        len(number_overlap) / len(left.numbers | right.numbers) if both_numbers else 0.0,
        float(not left.normalized), float(not right.normalized),
    )


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(-1e30, min(1e30, result)) if math.isfinite(result) else 0.0


def _reciprocal_rank(value: Any) -> float:
    rank = _finite(value)
    return 1.0 / rank if rank >= 1.0 else 0.0


def _provenance(retrieval: Mapping[str, Any]) -> set[str]:
    value = retrieval.get("provenance", ())
    if isinstance(value, str):
        found = set(value.replace(",", " ").split())
    elif isinstance(value, Mapping):
        found = {str(k) for k, v in value.items() if v}
    else:
        found = set(value or ())
    found |= {name for name in (*PROVENANCE_NAMES, *_PROVENANCE_ALIASES) if retrieval.get(name, False)}
    return {_PROVENANCE_ALIASES.get(name, name) for name in found}


def _feature_values(left: PreparedRecord, right: PreparedRecord, retrieval: Mapping[str, Any] | None) -> tuple[float, ...]:
    retrieval = retrieval or {}
    provenance = _provenance(retrieval)
    compact_left = "".join(left.name.normalized.split())
    compact_right = "".join(right.name.normalized.split())
    acronym_match = ((len(left.name.acronym) > 1 and left.name.acronym == compact_right)
                     or (len(right.name.acronym) > 1 and right.name.acronym == compact_left))
    return _text_features(left.name, right.name) + _text_features(left.address, right.address) + (
        _equal(left.name.legal, right.name.legal), _equal(left.name.core, right.name.core),
        float(len(left.name.acronym) > 1 and left.name.acronym == right.name.acronym), float(acronym_match),
        _equal(left.country, right.country), float(bool(left.country and right.country) and left.country != right.country),
        float(not left.country), float(not right.country),
        float(left.source == "S1"), float(right.source == "S2"), float(right.source == "S3"),
        _reciprocal_rank(retrieval.get("rank")), _finite(retrieval.get("score")),
        math.log1p(max(0.0, _finite(retrieval.get("candidate_count")))),
        _reciprocal_rank(retrieval.get("rank_name")), _reciprocal_rank(retrieval.get("rank_address")),
        _finite(retrieval.get("score_name")), _finite(retrieval.get("score_address")),
    ) + tuple(float(name in provenance) for name in PROVENANCE_NAMES) + tuple(
        value for channel in _RETRIEVAL_CHANNELS for value in (
            _finite(retrieval.get(f"score_{channel}")), _reciprocal_rank(retrieval.get(f"rank_{channel}"))
        )
    )


def features_for_pair(left: Mapping[str, Any] | PreparedRecord, right: Mapping[str, Any] | PreparedRecord,
                      retrieval: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Produce one ordered feature dictionary; empty-empty text is not evidence."""
    return dict(zip(FEATURE_NAMES, _feature_values(prepare_record(left), prepare_record(right), retrieval)))


def features_for_pairs(pairs: Iterable[Sequence[Any]], *, as_array: bool = True):
    """Build a bounded-batch matrix from (left, right[, retrieval]) tuples.

    Prepare records once upstream to avoid repeated normalization. Raw dictionary
    objects reused within this call are cached by object identity. Callers should
    submit bounded batches instead of passing the entire candidate graph.
    """
    cache: dict[int, tuple[Any, PreparedRecord]] = {}

    def prepared(record):
        if isinstance(record, PreparedRecord):
            return record
        key = id(record)
        if key not in cache:
            cache[key] = (record, prepare_record(record))
        return cache[key][1]

    values = []
    for pair in pairs:
        if len(pair) not in (2, 3):
            raise ValueError("Each pair must contain left, right, and optional retrieval metadata")
        values.append(_feature_values(prepared(pair[0]), prepared(pair[1]), pair[2] if len(pair) == 3 else None))
    if not as_array:
        return [dict(zip(FEATURE_NAMES, row)) for row in values]
    import numpy as np
    return np.asarray(values, dtype=np.float32).reshape((-1, len(FEATURE_NAMES)))


def summarize_positive_pairs(pairs: Iterable[Sequence[Any]], *, max_pairs: int = 10000,
                             max_scan: int = 100000, seed: int = 42) -> dict[str, Any]:
    """Quantify actual supplied positives, using a bounded reservoir sample.

    The caller must supply labeled positive pairs only. At most max_scan input
    pairs are inspected; results describe that prefix, not an unbiased sample of
    a larger dataset. No negative examples, labels or evaluation scores are inferred.
    """
    if max_pairs < 1 or max_scan < 1:
        raise ValueError("max_pairs and max_scan must be positive")
    rng = random.Random(seed)
    sample = []
    scanned = 0
    for scanned, pair in enumerate(itertools.islice(pairs, max_scan), 1):
        if len(sample) < max_pairs:
            sample.append(pair)
        else:
            slot = rng.randrange(scanned)
            if slot < max_pairs:
                sample[slot] = pair
    counts: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    for pair in sample:
        f = features_for_pair(pair[0], pair[1], pair[2] if len(pair) == 3 else None)
        for key in ("name_exact", "name_folded_exact", "name_legal_exact", "name_core_exact", "address_exact",
                    "address_folded_exact", "name_numbers_conflict", "address_numbers_conflict",
                    "name_numbers_disagree", "address_numbers_disagree", "country_conflict"):
            counts[key] += int(f[key] > 0)
        counts["name_no_shared_tokens"] += int(f["name_token_overlap"] == 0)
        counts["address_no_shared_tokens"] += int(f["address_token_overlap"] == 0)
        counts["any_name_missing"] += int(f["name_left_missing"] > 0 or f["name_right_missing"] > 0)
        counts["any_address_missing"] += int(f["address_left_missing"] > 0 or f["address_right_missing"] > 0)
        for key in ("name_edit_similarity", "address_edit_similarity", "name_token_jaccard", "address_token_jaccard"):
            sums[key] += f[key]
    n = len(sample)
    return {
        "input_pairs_scanned": scanned, "sampled_positive_pairs": n, "seed": seed,
        "sampling": "reservoir sample over at most max_scan supplied positive pairs; prefix may be unrepresentative",
        "max_scan": max_scan, "max_pairs": max_pairs,
        "counts": dict(counts), "fractions": {k: v / n for k, v in counts.items()} if n else {},
        "mean_similarities": {k: v / n for k, v in sums.items()} if n else {},
    }
