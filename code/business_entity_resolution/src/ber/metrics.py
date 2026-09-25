"""Exact challenge metrics and entity-aware threshold selection.

The evaluation universe is the keys of ``truth``. Missing prediction keys mean
empty predictions; extra keys and duplicate IDs in an input iterable are errors.
Micro precision/recall use 0 for a zero denominator. Singleton accuracy is None
when there are no true singletons; an empty evaluation universe is an error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any


def _id_set(values: Iterable[str], context: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context}: expected an iterable of IDs, not a string")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context}: IDs must be nonempty strings")
        if value in result:
            raise ValueError(f"{context}: duplicate ID {value!r}")
        result.add(value)
    return result


def _truth_sets(truth: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    if not truth:
        raise ValueError("The evaluation universe must contain at least one S1 entity")
    normalized = {}
    for s1, targets in truth.items():
        if not isinstance(s1, str) or not s1:
            raise ValueError("Source1 IDs must be nonempty strings")
        normalized[s1] = _id_set(targets, f"truth[{s1!r}]")
    return normalized


def _f05_counts(tp: int, predicted: int, true: int) -> float:
    denominator = predicted + 0.25 * true
    return 1.25 * tp / denominator if denominator else 1.0


def entity_f0_5(true_ids: Iterable[str], predicted_ids: Iterable[str]) -> float:
    """Return exact per-entity F0.5; reject duplicate IDs before set scoring."""
    true = _id_set(true_ids, "truth")
    predicted = _id_set(predicted_ids, "predictions")
    return _f05_counts(len(true & predicted), len(predicted), len(true))


def _summary(
    *, n_entities: int, macro_sum: float, true_links: int, predicted_links: int,
    true_positives: int, singleton_count: int, correct_singletons: int,
) -> dict[str, Any]:
    fp = predicted_links - true_positives
    fn = true_links - true_positives
    precision = true_positives / predicted_links if predicted_links else 0.0
    recall = true_positives / true_links if true_links else 0.0
    denominator = 1.25 * true_positives + fp + 0.25 * fn
    return {
        "n_entities": n_entities,
        "macro_f0_5": macro_sum / n_entities,
        "precision": precision,
        "recall": recall,
        "micro_f0_5": 1.25 * true_positives / denominator if denominator else 0.0,
        "true_positives": true_positives,
        "false_positives": fp,
        "false_negatives": fn,
        "true_links": true_links,
        "predicted_links": predicted_links,
        "singleton_count": singleton_count,
        "correct_singletons": correct_singletons,
        "false_singleton_merges": singleton_count - correct_singletons,
        "singleton_accuracy": (
            correct_singletons / singleton_count if singleton_count else None
        ),
        "avg_predicted_matches_per_s1": predicted_links / n_entities,
        "false_positives_per_s1": fp / n_entities,
        "false_negatives_per_s1": fn / n_entities,
    }


def summarize_entities(
    truth: Mapping[str, Iterable[str]], predictions: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    """Score every truth anchor, including absent predictions and singletons.

    F0.5 is averaged *per entity*, never calculated from averaged precision and
    recall. Precision/recall in the returned summary are micro link diagnostics.
    """
    true_sets = _truth_sets(truth)
    extra = predictions.keys() - true_sets.keys()
    if extra:
        raise ValueError(f"Predictions contain unknown S1 entities: {sorted(extra)[:5]}")
    true_links = predicted_links = true_positives = singleton_count = correct = 0
    scores = []
    for s1, true in true_sets.items():
        predicted = _id_set(predictions.get(s1, ()), f"predictions[{s1!r}]")
        tp = len(true & predicted)
        scores.append(_f05_counts(tp, len(predicted), len(true)))
        true_links += len(true)
        predicted_links += len(predicted)
        true_positives += tp
        if not true:
            singleton_count += 1
            correct += not predicted
    return _summary(
        n_entities=len(true_sets), macro_sum=math.fsum(scores),
        true_links=true_links, predicted_links=predicted_links,
        true_positives=true_positives, singleton_count=singleton_count,
        correct_singletons=correct,
    )


def iter_threshold_metrics(
    truth: Mapping[str, Iterable[str]],
    scored_pairs: Iterable[tuple[str, str, float]],
    thresholds: Iterable[float] | None = None,
) -> Iterator[dict[str, Any]]:
    """Sweep score >= threshold in descending order with incremental entities.

    Each input is (S1 ID, target ID, finite score). Duplicate pairs are rejected,
    and all truth anchors contribute even if they have no scored candidates.
    If thresholds is None, evaluate +inf (predict nothing) and each distinct
    observed score. Consume this iterator or use select_threshold when there are
    many distinct scores: materializing every summary can otherwise be large.

    Complexity: O(P log P + N + H log H), memory O(P + N + H). Unlike rescoring
    every entity at every threshold, each pair is added once. For large numeric
    candidate tables, use iter_threshold_array_metrics to avoid Python pair
    objects. No assumption of global one-to-one target ownership is imposed.
    """
    true_sets = _truth_sets(truth)
    rows = []
    for s1, target, raw_score in scored_pairs:
        if s1 not in true_sets:
            raise ValueError(f"Scored candidate has unknown S1 entity: {s1!r}")
        if not isinstance(target, str) or not target:
            raise ValueError("Scored candidate target IDs must be nonempty strings")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("Candidate scores must be finite")
        rows.append((s1, target, score, target in true_sets[s1]))
    # Sort instead of retaining a second Python set of millions of pair keys.
    rows.sort(key=lambda row: (row[0], row[1]))
    for i in range(1, len(rows)):
        if rows[i-1][:2] == rows[i][:2]:
            raise ValueError(f"Duplicate scored candidate pair: {rows[i][:2]!r}")
    rows.sort(key=lambda row: row[2], reverse=True)
    if thresholds is None:
        levels = [math.inf]
        levels.extend(row[2] for i, row in enumerate(rows) if i == 0 or row[2] != rows[i-1][2])
    else:
        levels = sorted({float(t) for t in thresholds}, reverse=True)
        if any(math.isnan(t) for t in levels):
            raise ValueError("Thresholds cannot be NaN")
    n_singletons = sum(not value for value in true_sets.values())
    macro_sum = float(n_singletons)
    true_links = sum(map(len, true_sets.values()))
    predicted_links = total_tp = offset = 0
    correct_singletons = n_singletons
    state: dict[str, tuple[int, int]] = {}
    for threshold in levels:
        while offset < len(rows) and rows[offset][2] >= threshold:
            s1, _, _, positive = rows[offset]
            tp, predicted = state.get(s1, (0, 0))
            true_count = len(true_sets[s1])
            before = _f05_counts(tp, predicted, true_count)
            if not true_count and predicted == 0:
                correct_singletons -= 1
            tp += positive
            predicted += 1
            state[s1] = tp, predicted
            macro_sum += _f05_counts(tp, predicted, true_count) - before
            total_tp += positive
            predicted_links += 1
            offset += 1
        result = _summary(
            n_entities=len(true_sets), macro_sum=macro_sum,
            true_links=true_links, predicted_links=predicted_links,
            true_positives=total_tp, singleton_count=n_singletons,
            correct_singletons=correct_singletons,
        )
        result["threshold"] = threshold
        yield result


def sweep_thresholds(
    truth: Mapping[str, Iterable[str]], scored_pairs: Iterable[tuple[str, str, float]],
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """Return descending threshold summaries; prefer iterator for all-score sweeps."""
    return list(iter_threshold_metrics(truth, scored_pairs, thresholds))


def select_threshold(
    truth: Mapping[str, Iterable[str]], scored_pairs: Iterable[tuple[str, str, float]],
    thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Maximize entity macro F0.5, preferring a higher threshold for score ties."""
    return _select_best(iter_threshold_metrics(truth, scored_pairs, thresholds))


def _select_best(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    best = None
    for row in rows:
        if best is None or row["macro_f0_5"] > best["macro_f0_5"] + 1e-12:
            best = row
        elif abs(row["macro_f0_5"] - best["macro_f0_5"]) <= 1e-12:
            if row["threshold"] > best["threshold"]:
                best = row
    if best is None:
        raise ValueError("At least one threshold is required")
    return best


def iter_threshold_array_metrics(
    true_counts: Any, anchor_indices: Any, target_indices: Any,
    positive_labels: Any, scores: Any, thresholds: Iterable[float] | None = None,
) -> Iterator[dict[str, Any]]:
    """NumPy candidate-table sweep without per-pair Python objects.

    true_counts contains the full evaluation universe, including anchors absent
    from candidates. anchor_indices index that array. target_indices are stable
    integer target codes; (anchor, target) duplicates are rejected. positive_labels
    must already have been derived by joining official truth to those exact
    candidate pairs. Counts must include unretrieved positives. These necessary
    join semantics cannot be reconstructed from numeric arrays alone.

    Pass an explicit compact threshold grid for high scale. The default evaluates
    all unique scores; each candidate still enters only once. Requires NumPy only
    when this optional entry point is used.
    """
    import numpy as np

    def integer_vector(value: Any, name: str) -> Any:
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if array.size and (array.min() < 0 or array.max() > np.iinfo(np.int64).max):
            raise ValueError(f"{name} values must fit nonnegative int64")
        return array.astype(np.int64, copy=False)

    truth = integer_vector(true_counts, "true_counts")
    anchors = integer_vector(anchor_indices, "anchor_indices")
    targets = integer_vector(target_indices, "target_indices")
    labels = np.asarray(positive_labels)
    values = np.asarray(scores, dtype=np.float64)
    if not truth.size:
        raise ValueError("The evaluation universe must contain at least one S1 entity")
    if labels.ndim != 1 or labels.dtype.kind not in "biu" or not np.isin(labels, [0, 1]).all():
        raise ValueError("positive_labels must be a one-dimensional binary array")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a one-dimensional finite array")
    if not (len(anchors) == len(targets) == len(labels) == len(values)):
        raise ValueError("Candidate arrays must have equal length")
    if anchors.size and anchors.max() >= truth.size:
        raise ValueError("Candidate anchor index is outside the evaluation universe")
    pair_order = np.lexsort((targets, anchors))
    if len(pair_order) > 1:
        left, right = pair_order[:-1], pair_order[1:]
        if np.any((anchors[left] == anchors[right]) & (targets[left] == targets[right])):
            raise ValueError("Duplicate scored candidate pair")
    del pair_order
    labels = labels.astype(np.int64, copy=False)
    candidate_positives = np.bincount(anchors[labels == 1], minlength=len(truth))
    if np.any(candidate_positives > truth):
        raise ValueError("Candidate positive counts exceed full truth counts")
    order = np.argsort(-values, kind="stable")
    ordered_scores = values[order]
    if thresholds is None:
        levels = np.concatenate(([np.inf], np.unique(values)[::-1]))
    else:
        levels = np.array(sorted({float(t) for t in thresholds}, reverse=True))
        if np.isnan(levels).any():
            raise ValueError("Thresholds cannot be NaN")
    predicted = np.zeros(len(truth), dtype=np.int64)
    tp = np.zeros(len(truth), dtype=np.int64)
    entity_scores = (truth == 0).astype(np.float64)
    n_singletons = int((truth == 0).sum())
    correct_singletons = n_singletons
    macro_sum = float(n_singletons)
    total_tp = offset = 0
    true_links = int(truth.sum())
    # A single negative-score view permits binary-search boundaries per level.
    neg_scores = -ordered_scores
    for threshold in levels:
        end = int(np.searchsorted(neg_scores, -threshold, side="right"))
        entering = order[offset:end]
        if len(entering):
            group_ids, inverse = np.unique(anchors[entering], return_inverse=True)
            added_count = np.bincount(inverse, minlength=len(group_ids))
            added_tp = np.bincount(inverse, weights=labels[entering], minlength=len(group_ids)).astype(np.int64)
            correct_singletons -= int(((truth[group_ids] == 0) & (predicted[group_ids] == 0)).sum())
            old_sum = float(entity_scores[group_ids].sum())
            predicted[group_ids] += added_count
            tp[group_ids] += added_tp
            entity_scores[group_ids] = 1.25 * tp[group_ids] / (predicted[group_ids] + 0.25 * truth[group_ids])
            macro_sum += float(entity_scores[group_ids].sum()) - old_sum
            total_tp += int(added_tp.sum())
        offset = end
        result = _summary(
            n_entities=len(truth), macro_sum=macro_sum,
            true_links=true_links, predicted_links=end, true_positives=total_tp,
            singleton_count=n_singletons, correct_singletons=correct_singletons,
        )
        result["threshold"] = float(threshold)
        yield result


def select_threshold_arrays(
    true_counts: Any, anchor_indices: Any, target_indices: Any,
    positive_labels: Any, scores: Any, thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Array equivalent of select_threshold; see its input join requirements."""
    return _select_best(iter_threshold_array_metrics(
        true_counts, anchor_indices, target_indices, positive_labels, scores, thresholds,
    ))
