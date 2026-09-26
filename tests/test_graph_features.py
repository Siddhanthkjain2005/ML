import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from ber.graph_features import FEATURE_NAMES, build_graph_features
from ber.pair_dataset import TARGET_FACTOR


def fixture_data(tmp_path, *, scores=(.99, .1), sources=(2, 3), anchors=None):
    candidates, raw = tmp_path / "candidates", tmp_path / "raw"
    candidates.mkdir(); raw.mkdir()
    n = len(scores)
    if anchors is None:
        anchors = np.zeros(n, dtype=np.uint32)
    codes = np.array([source * TARGET_FACTOR + index + 1 for index, source in enumerate(sources)], dtype=np.uint64)
    # Fixtures preserve the required anchor/target ordering.
    assert all((int(anchors[i]), int(codes[i])) < (int(anchors[i + 1]), int(codes[i + 1])) for i in range(n - 1))
    np.save(candidates / "anchors.npy", np.asarray(anchors, dtype=np.uint32))
    np.save(candidates / "target_codes.npy", codes)
    np.save(candidates / "labels.npy", np.ones(n, dtype=np.uint8))
    np.save(candidates / "truth_counts.npy", np.ones(max(anchors, default=0) + 1, dtype=np.uint32))
    count = int(max(anchors, default=-1)) + 1
    (candidates / "queries.jsonl").write_text("".join(json.dumps({
        "entity_id": f"S1-{i + 1}", "business_name": "Alpha", "business_address": "12 Road", "country": "US",
    }) + "\n" for i in range(count)))
    connection = sqlite3.connect(raw / "targets.sqlite")
    connection.execute("CREATE TABLE records (code INTEGER PRIMARY KEY, name TEXT, address TEXT, country TEXT)")
    connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", [
        (int(code), "Alpha", "12 Road" if i == 0 else "93 Other Avenue", "US") for i, code in enumerate(codes)
    ])
    connection.commit(); connection.close()
    scores_path = tmp_path / "scores.npy"
    np.save(scores_path, np.asarray(scores, dtype=np.float32))
    return candidates, raw, scores_path


def build(inputs, output, **kwargs):
    metadata = build_graph_features(*inputs, output, score_role="out_of_fold_train", **kwargs)
    return metadata, np.load(output / "extraX.npy")


def column(matrix, name):
    return matrix[:, FEATURE_NAMES.index(name)]


def test_current_candidate_never_its_own_support(tmp_path):
    inputs = fixture_data(tmp_path)
    metadata, matrix = build(inputs, tmp_path / "graph", batch_size=1)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert metadata["complete"] and metadata["config"]["score_role"] == "out_of_fold_train"
    assert column(matrix, "graph_support_count").tolist() == [0., 1.]
    assert column(matrix, "graph_support_max_name_exact").tolist() == [0., 1.]
    assert column(matrix, "graph_support_max_address_numbers_equal").tolist() == [0., 0.]
    assert column(matrix, "graph_score_margin")[1] == pytest.approx(.1 - .99)
    assert np.isfinite(matrix).all()


def test_labels_and_truth_counts_have_no_effect_even_on_resume(tmp_path):
    inputs = fixture_data(tmp_path)
    metadata, before = build(inputs, tmp_path / "graph")
    # Deliberately invalid contents ensure neither is opened or hashed.
    (inputs[0] / "labels.npy").write_bytes(b"this is not a numpy file")
    (inputs[0] / "truth_counts.npy").write_bytes(b"nor is this")
    after_metadata, after = build(inputs, tmp_path / "graph")
    assert after_metadata == metadata
    np.testing.assert_array_equal(before, after)
    _, independent = build(inputs, tmp_path / "independent")
    np.testing.assert_array_equal(before, independent)


def test_no_trusted_candidates_produces_no_support_evidence(tmp_path):
    inputs = fixture_data(tmp_path, scores=(.2, .3))
    _, matrix = build(inputs, tmp_path / "graph")
    np.testing.assert_array_equal(matrix[:, 1:], 0)
    np.testing.assert_allclose(matrix[:, 0], [.2, .3])


def test_same_source_targets_are_valid_supports_and_capped_at_three(tmp_path):
    inputs = fixture_data(tmp_path, scores=(.99, .98, .97, .96, .95), sources=(2, 2, 2, 2, 2))
    _, matrix = build(inputs, tmp_path / "graph", batch_size=2)
    np.testing.assert_array_equal(column(matrix, "graph_support_count"), 3)
    np.testing.assert_array_equal(column(matrix, "graph_same_source_support_count"), 3)
    np.testing.assert_array_equal(column(matrix, "graph_cross_source_support_count"), 0)
    np.testing.assert_array_equal(column(matrix, "graph_source2_support_count"), 3)
    np.testing.assert_array_equal(column(matrix, "graph_source3_support_count"), 0)
    np.testing.assert_array_equal(column(matrix, "graph_support_source_diversity"), 1)
    assert column(matrix, "graph_max_support_score")[0] == pytest.approx(.98)


def test_batch_boundaries_and_multiple_sources_do_not_change_features(tmp_path):
    inputs = fixture_data(tmp_path, scores=(.99, .98, .97, .96, .1, .2), sources=(2, 2, 3, 3, 2, 3), anchors=[0, 0, 0, 0, 1, 1])
    _, small = build(inputs, tmp_path / "small", batch_size=1)
    _, large = build(inputs, tmp_path / "large", batch_size=100)
    np.testing.assert_array_equal(small, large)
    np.testing.assert_array_equal(column(small, "graph_support_source_diversity")[:4], 2)
    np.testing.assert_array_equal(column(small, "graph_support_count")[4:], 0)


def test_equal_score_ties_use_target_code_order(tmp_path):
    inputs = fixture_data(tmp_path, scores=(.99, .99, .99, .99, .99), sources=(2, 2, 2, 3, 3))
    _, matrix = build(inputs, tmp_path / "graph", batch_size=2)
    # The last two candidates use the three Source2 targets, with lowest IDs.
    np.testing.assert_array_equal(column(matrix, "graph_source2_support_count")[3:], 3)
    np.testing.assert_array_equal(column(matrix, "graph_source3_support_count")[3:], 0)


def test_interrupted_batch_can_resume(tmp_path):
    inputs = fixture_data(tmp_path, scores=(.99, .98, .1), sources=(2, 2, 3))
    def interrupt(_):
        raise RuntimeError("simulated interruption")
    with pytest.raises(RuntimeError, match="simulated"):
        build(inputs, tmp_path / "interrupted", batch_size=1, progress=interrupt)
    meta, resumed = build(inputs, tmp_path / "interrupted", batch_size=1)
    _, direct = build(inputs, tmp_path / "direct", batch_size=1)
    assert meta["rows_completed"] == 3
    np.testing.assert_array_equal(resumed, direct)


def test_resume_rejects_changed_scores_and_corrupted_output(tmp_path):
    inputs = fixture_data(tmp_path)
    build(inputs, tmp_path / "graph")
    np.save(inputs[2], np.asarray([.8, .2], dtype=np.float32))
    with pytest.raises(ValueError, match="inputs/configuration changed"):
        build(inputs, tmp_path / "graph")
    build(inputs, tmp_path / "other")
    matrix = np.load(tmp_path / "other" / "extraX.npy", mmap_mode="r+")
    matrix[0, 0] = .555; matrix.flush()
    with pytest.raises(ValueError, match="artifact checksum"):
        build(inputs, tmp_path / "other")


@pytest.mark.parametrize("role", [None, "train", "in_sample_train", "validation"])
def test_score_provenance_role_is_required_and_restrictive(tmp_path, role):
    inputs = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="score_role"):
        build_graph_features(*inputs, tmp_path / "graph", score_role=role)


@pytest.mark.parametrize("bad_scores", [(.9, float("nan")), (.9, 1.01), (-.01, .8)])
def test_invalid_scores_fail(tmp_path, bad_scores):
    inputs = fixture_data(tmp_path, scores=bad_scores)
    with pytest.raises(ValueError, match="probabilities"):
        build(inputs, tmp_path / "graph")


def test_empty_candidate_set(tmp_path):
    inputs = fixture_data(tmp_path, scores=(), sources=())
    meta, matrix = build(inputs, tmp_path / "graph")
    assert meta["complete"] and matrix.shape == (0, len(FEATURE_NAMES))
