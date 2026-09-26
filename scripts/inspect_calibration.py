"""Diagnose one frozen scorer on saved calibration candidates, without tuning.

Example (run from the project root with PYTHONPATH set to the package source):
  python scripts/inspect_calibration.py --experiment-root experiments/pilot_v1 \
    --model experiments/reference_refinement_v1/depth7/model.json \
    --feature-dir features_reference_v1 --threshold .425 \
    --output experiments/reference_errors

No validation or test inputs are read, and no decision rules are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata

import numpy as np

from ber.metrics import iter_threshold_array_metrics
from ber.model import predict_scores


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scripts(text):
    """An explicit script proxy; combining marks/numbers are not scripts."""
    result = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, '')
        if name.startswith(('CJK ', 'IDEOGRAPHIC ')):
            result.add('HAN')
        elif name:
            result.add(name.split()[0])
    return result


def checked_calibration(root, feature_dir):
    root = Path(root).resolve()
    fold = root if root.name == 'calibration' else root / 'calibration'
    if fold.name != 'calibration' or any(x in ('validation', 'test', 'train') for x in fold.parts):
        raise ValueError('Only a calibration partition may be inspected')
    leaf = Path(feature_dir)
    if leaf.is_absolute() or len(leaf.parts) != 1 or leaf.name in ('.', '..'):
        raise ValueError('--feature-dir must name one directory inside calibration')
    features = (fold / leaf).resolve()
    if features.parent != fold.resolve():
        raise ValueError('Feature directory must remain inside calibration')
    return fold, features


def summarize_pairs(mask, positive, chosen):
    tp = int(np.count_nonzero(mask & positive & chosen))
    fp = int(np.count_nonzero(mask & ~positive & chosen))
    fn = int(np.count_nonzero(mask & positive & ~chosen))
    tn = int(np.count_nonzero(mask & ~positive & ~chosen))
    return dict(candidate_pairs=int(mask.sum()), tp=tp, fp=fp, retrieved_fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall_among_retrieved=tp / (tp + fn) if tp + fn else None)


def inspect(args):
    fold, features = checked_calibration(args.experiment_root, args.feature_dir)
    if not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError('Threshold must be finite and between zero and one')
    cand = fold / 'candidates'
    queries = [json.loads(line) for line in (cand / 'queries.jsonl').open()]
    anchors, codes, labels, truth = [np.load(cand / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                                    for name in ('anchors', 'target_codes', 'labels', 'truth_counts')]
    X = np.load(features / 'X.npy', mmap_mode='r', allow_pickle=False)
    meta = json.loads((features / 'metadata.json').read_text())
    names = meta['feature_names']
    if X.shape != (len(anchors), len(names)) or len(names) != len(set(names)):
        raise ValueError('Feature matrix/name shape mismatch')
    if len(truth) != len(queries) or not len(queries) or len(labels) != len(anchors) or len(codes) != len(anchors):
        raise ValueError('Candidate/query/truth shape mismatch')
    if len(anchors) and (anchors.min() < 0 or anchors.max() >= len(queries)):
        raise ValueError('Anchor out of query bounds')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(X).all():
        raise ValueError('Invalid labels or nonfinite features')
    model = Path(args.model).resolve()
    model_meta = model.parent / 'metadata.json'
    if model_meta.exists():
        expected = json.loads(model_meta.read_text()).get('feature_names')
        if expected is not None and expected != names:
            raise ValueError('Model feature ordering differs from calibration matrix')
    scores = predict_scores(model, X)
    chosen, positive = scores >= args.threshold, np.asarray(labels == 1)
    overall = next(iter_threshold_array_metrics(truth, anchors, codes, labels, scores, [args.threshold]))
    n = len(queries)
    as_int = anchors.astype(np.int64)
    retrieved = np.bincount(as_int[positive], minlength=n)
    tp = np.bincount(as_int[positive & chosen], minlength=n)
    predicted = np.bincount(as_int[chosen], minlength=n)
    missing = np.asarray(truth, dtype=np.int64) - retrieved
    if (missing < 0).any():
        raise ValueError('More retrieved positive links than true links')
    denominator = predicted + .25 * truth
    f = np.divide(1.25 * tp, denominator, out=np.ones(n), where=denominator > 0)
    def entity_summary(mask):
        count = int(mask.sum())
        return dict(entities=count, macro_f0_5=float(f[mask].mean()) if count else None,
                    true_links=int(truth[mask].sum()), tp=int(tp[mask].sum()),
                    fp=int((predicted[mask] - tp[mask]).sum()),
                    fn=int((truth[mask] - tp[mask]).sum()),
                    retrieved_fn=int((retrieved[mask] - tp[mask]).sum()),
                    unretrieved_fn=int(missing[mask].sum()),
                    entities_with_unretrieved_links=int(np.count_nonzero(missing[mask])))
    countries = np.asarray([q['country'] for q in queries])
    entity_groups = {f'country:{c}': entity_summary(countries == c) for c in sorted(set(countries))}
    entity_groups.update(singleton=entity_summary(truth == 0), one_true_match=entity_summary(truth == 1),
                         multiple_true_matches=entity_summary(truth > 1))

    raw_meta = json.loads((fold / 'features/metadata.json').read_text())
    lookup = fold / 'features/targets.sqlite'
    if not lookup.exists():
        lookup = Path(raw_meta['target_lookup']['shared_source_index'])
    if not lookup.is_file():
        raise FileNotFoundError(f'Raw calibration target lookup missing: {lookup}')
    db = sqlite3.connect(f'file:{lookup.resolve()}?mode=ro', uri=True)
    unique, inverse = np.unique(codes, return_inverse=True)
    nonascii = np.zeros(len(unique), dtype=bool)
    target_scripts = []
    # Keep script flags only, rather than all raw target records, in memory.
    try:
        for start in range(0, len(unique), 500):
            batch = [int(c) for c in unique[start:start + 500]]
            marks = ','.join('?' for _ in batch)
            rows = {c: (name, address, country) for c, name, address, country in
                    db.execute(f'SELECT code,name,address,country FROM records WHERE code IN ({marks})', batch)}
            if len(rows) != len(batch):
                raise ValueError('Raw lookup is missing candidate targets')
            for i, code in enumerate(batch, start):
                name = rows[code][0]
                nonascii[i] = not name.isascii()
                target_scripts.append(scripts(name))
        query_scripts = [scripts(q['business_name']) for q in queries]
        disjoint = np.asarray([bool(query_scripts[int(a)] and target_scripts[int(t)]) and
                               query_scripts[int(a)].isdisjoint(target_scripts[int(t)])
                               for a, t in zip(anchors, inverse)], dtype=bool)
        columns = {name: i for i, name in enumerate(names)}
        def col(name):
            return np.asarray(X[:, columns[name]]) if name in columns else None
        masks = {f'country:{c}': countries[as_int] == c for c in sorted(set(countries))}
        masks['target_name_non_ascii'] = nonascii[inverse]
        masks['target_name_ascii'] = ~nonascii[inverse]
        masks['name_script_disjoint'] = disjoint
        for field in ('name', 'address'):
            for suffix in ('left_missing', 'right_missing', 'numbers_both_present', 'numbers_equal',
                           'numbers_conflict', 'numbers_disagree', 'without_numbers_exact'):
                key = field + '_' + suffix
                values = col(key)
                if values is not None:
                    masks[key] = values > .5
        for feature, boundary in (('name_roman_ratio', .9), ('address_roman_token_sort', .9),
                                  ('address_first_number_edit', .75)):
            values = col(feature)
            if values is not None:
                masks[f'{feature}>={boundary}'] = values >= boundary
        namecos, addrstrong = col('actual_name_char_cosine'), col('address_roman_token_sort')
        if addrstrong is None:
            addrstrong = col('address_token_sort_similarity')
        if namecos is not None and addrstrong is not None:
            masks['name_char_cosine_zero_address_strong'] = (namecos <= 1e-8) & (addrstrong >= .8)
            masks['cross_script_name_zero_address_strong'] = disjoint & masks['name_char_cosine_zero_address_strong']
        pair_groups = {key: summarize_pairs(mask, positive, chosen) for key, mask in masks.items()}
        bins = {}
        values_to_bin = {'score': scores}
        for key in ('name_roman_ratio', 'name_roman_token_sort', 'address_roman_token_sort',
                    'address_first_number_edit', 'address_best_number_edit', 'address_without_numbers_token_set',
                    'actual_name_char_cosine', 'actual_address_char_cosine',
                    'reference_name_address_margin', 'reference_core_address_margin'):
            values = col(key)
            if values is not None:
                values_to_bin[key] = values
        for key, values in values_to_bin.items():
            edges = [-1.00001, -.5, -.1, 0, .1, .5, .9, 1.00001] if 'margin' in key else [0, .1, .25, .5, .75, .9, .99, 1.00001]
            bins[key] = [dict(lower_inclusive=float(lo), upper_exclusive=float(hi),
                              **summarize_pairs((values >= lo) & (values < hi), positive, chosen))
                         for lo, hi in zip(edges[:-1], edges[1:])]
        examples = {}
        for group, mask, descending in (('false_positives', ~positive & chosen, True),
                                         ('retrieved_false_negatives', positive & ~chosen, False)):
            indices = np.flatnonzero(mask)
            selected = indices[np.argsort(-scores[indices] if descending else scores[indices], kind='stable')[:30]]
            rows = []
            for i in selected:
                a, code = int(anchors[i]), int(codes[i])
                name, address, country = db.execute('SELECT name,address,country FROM records WHERE code=?', (code,)).fetchone()
                rows.append(dict(candidate_row=int(i), score=float(scores[i]), query=queries[a],
                                 target=dict(entity_id=f'S{code // 10**12}-{code % 10**12}', business_name=name,
                                             business_address=address, country=country),
                                 entity_f0_5=float(f[a]), entity_tp=int(tp[a]), entity_fp=int(predicted[a] - tp[a]),
                                 entity_fn=int(truth[a] - tp[a]),
                                 features={key: float(X[i, j]) for j, key in enumerate(names)},
                                 groups=[key for key, group_mask in masks.items() if group_mask[i]]))
            examples[group] = rows
    finally:
        db.close()
    worst = sorted(np.flatnonzero(missing), key=lambda i: (-missing[i], f[i], i))[:30]
    result = dict(partition='calibration', threshold=float(args.threshold), overall=overall,
                  entity_summary=entity_summary(np.ones(n, dtype=bool)), entity_groups=entity_groups,
                  candidate_pair_groups=pair_groups, bins=bins, examples=examples,
                  unretrieved_examples=[dict(query=queries[int(i)], true_links=int(truth[i]),
                                             unretrieved_true_links=int(missing[i]), entity_f0_5=float(f[i])) for i in worst],
                  provenance=dict(model=str(model), model_sha256=digest(model), feature_directory=str(features),
                                  feature_matrix_sha256=digest(features / 'X.npy'),
                                  candidate_sha256={name: digest(cand / name) for name in
                                                    ('anchors.npy', 'target_codes.npy', 'labels.npy', 'truth_counts.npy', 'queries.jsonl')}),
                  notes=['Frozen threshold supplied by caller; this script does not select or fit anything.',
                         'Pair strata overlap and describe retrieved candidates only; group recall excludes retrieval misses.',
                         'Unretrieved positives are reported separately at entity level; their absent raw targets are not invented.',
                         'Feature values are evidence, not model attribution or a causal explanation.',
                         'Name script comparison is a Unicode character-name proxy, not language identification.',
                         'Calibration diagnostics are for model development and are not an unbiased final validation estimate.'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'calibration_errors.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    s = result['entity_summary']
    lines = ['# Frozen scorer: calibration diagnostics', '',
             f"Threshold: {args.threshold:g}. Entities: {n:,}. Candidate pairs: {len(anchors):,}.", '',
             f"Macro F0.5: {s['macro_f0_5']:.6f}. False positives: {s['fp']:,}. "
             f"False negatives: {s['fn']:,} ({s['retrieved_fn']:,} retrieved but rejected; {s['unretrieved_fn']:,} never retrieved).", '',
             '## Entity groups', '', '| Group | Entities | Macro F0.5 | FP | Retrieved FN | Unretrieved FN |',
             '|---|---:|---:|---:|---:|---:|']
    for key, group in entity_groups.items():
        metric = f"{group['macro_f0_5']:.6f}" if group['macro_f0_5'] is not None else 'N/A'
        lines.append(f"| {key} | {group['entities']} | {metric} | {group['fp']} | {group['retrieved_fn']} | {group['unretrieved_fn']} |")
    lines += ['', '## Retrieved candidate strata', '',
              'Groups overlap; these counts cannot be added across rows. Recall here excludes retrieval misses.', '',
              '| Group | Pairs | TP | FP | Retrieved FN | Precision |', '|---|---:|---:|---:|---:|---:|']
    for key, group in pair_groups.items():
        precision = f"{group['precision']:.5f}" if group['precision'] is not None else 'N/A'
        lines.append(f"| {key} | {group['candidate_pairs']} | {group['tp']} | {group['fp']} | {group['retrieved_fn']} | {precision} |")
    lines += ['', '## Inspection priorities', '',
              'The JSON contains score/feature bins, the 30 highest-score false positives, the 30 lowest-score retrieved false negatives, and entities with missing candidates.', '',
              'Use the examples to form hypotheses; this report does not claim that a feature caused an error or propose untested decision rules.', '',
              'Calibration metrics support development only. No validation or test partition was inspected.', '']
    (out / 'calibration_errors.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out), **s), sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', default='experiments/pilot_v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--feature-dir', default='features_reference_v1')
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output', default='experiments/reference_errors')
    inspect(parser.parse_args())


if __name__ == '__main__':
    main()
