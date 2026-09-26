"""Resumable experiment orchestration; identities are split before sampling.

Run ``python -m ber.pipeline --help``. All model/threshold selection uses the
calibration fold; validation is reported only after selection. Retrieval always
searches the entire corresponding target fold, including unlinked distractors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from .retrieval import retrieve_sparse
from .model import fit_models, predict_scores, select_decision
from .metrics import iter_threshold_array_metrics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def event(**kwargs):
    print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs}), flush=True)


def sample_queries(path, limit, seed=42):
    """Uniform deterministic bottom-hash sample, independent of labels/text."""
    heap = []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            identifier = row['entity_id']
            priority = int.from_bytes(hashlib.blake2b(f'{seed}\0{identifier}'.encode(), digest_size=16).digest(), 'big')
            item = (-priority, identifier, row)
            if limit <= 0 or len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: int(item[1].split('-')[1]))]


def read_truth(path, queries):
    wanted = {row['entity_id'] for row in queries}
    truth = {}
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream, delimiter='\t')
        next(reader)
        for identifier, matches in reader:
            if identifier in wanted:
                truth[identifier] = {value for value in matches.split(',') if value}
    if set(truth) != wanted:
        raise ValueError('Ground truth must cover every sampled anchor')
    return truth


def load_arrays(folder):
    folder = Path(folder)
    return {name: np.load(folder / f'{name}.npy', mmap_mode='r')
            for name in ('anchors', 'target_codes', 'labels', 'truth_counts')}


def build_fold(split_root, fold, limit, vectorizers, output, cache_root,
               *, k, seed, threads, target_batch, query_feature_limit=None, cache_workers=1, rerank_pool=None):
    from .candidates import build_candidates
    from .pair_dataset import build_pair_dataset
    source = Path(split_root) / fold
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    queries = sample_queries(source / 'source1.tsv', limit, seed)
    targets = {s: source / f'source{s}.tsv' for s in (2, 3)}
    truth = read_truth(source / 'ground_truth.tsv', queries)
    event(stage='fold_start', fold=fold, anchors=len(queries), true_links=sum(map(len, truth.values())))
    retrieval = {}
    for s, path in targets.items():
        retrieval[s] = retrieve_sparse(queries, path, vectorizers, output / f'retrieval_s{s}',
            k=k, threads=threads, target_batch=target_batch, query_batch=512,
            cache_dir=Path(cache_root) / fold / f'source{s}',
            progress=lambda message: event(fold=fold, **message), query_feature_limit=query_feature_limit,
            cache_workers=cache_workers, rerank_pool=rerank_pool)
    candidate_meta = build_candidates(queries, targets, retrieval, output / 'candidates', truth=truth)
    event(stage='candidates_complete', fold=fold)
    feature_meta = build_pair_dataset(output / 'candidates', targets, vectorizers, output / 'features')
    event(stage='features_complete', fold=fold, shape=feature_meta.get('shape'))
    return candidate_meta, feature_meta


def run_experiment(args):
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key != 'func'}
    if config.get('query_feature_limit') is None:
        config.pop('query_feature_limit', None)
    if config.get('cache_workers', 1) == 1:
        config.pop('cache_workers', None)
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Experiment configuration changed; use a new output directory')
    write_json(config_path, config)
    vectorizers = joblib.load(args.vectorizers)
    feature_names = None
    # Validation remains untouched until the model and threshold are frozen.
    for fold, limit in [('train', args.train_anchors), ('calibration', args.calibration_anchors)]:
        _, meta = build_fold(args.splits, fold, limit, vectorizers, output / fold, args.cache,
            k=args.k, seed=args.seed, threads=args.threads, target_batch=args.target_batch,
            query_feature_limit=getattr(args, 'query_feature_limit', None), cache_workers=getattr(args, 'cache_workers', 1))
        if feature_names is None:
            feature_names = meta['feature_names']
        elif feature_names != meta['feature_names']:
            raise ValueError('Feature schema differs across folds')
    model_manifest = output / 'models.json'
    if model_manifest.exists():
        paths = json.loads(model_manifest.read_text())
    else:
        X = np.load(output / 'train/features/X.npy', mmap_mode='r')
        y = np.load(output / 'train/candidates/labels.npy', mmap_mode='r')
        paths = fit_models(X, y, output / 'models', seed=args.seed, threads=args.threads,
                           feature_names=feature_names)
        write_json(model_manifest, paths)
        event(stage='models_trained', models=list(paths))
    calibration = load_arrays(output / 'calibration/candidates')
    X = np.load(output / 'calibration/features/X.npy', mmap_mode='r')
    decisions = {}
    for name, path in paths.items():
        scores = predict_scores(path, X)
        np.save(output / f'calibration/{name}_scores.npy', scores)
        decisions[name] = select_decision(calibration['truth_counts'], calibration['anchors'],
            calibration['target_codes'], calibration['labels'], scores)
    # The official final-model rule requires MIT/Apache 2.0. The sklearn
    # baseline is diagnostic; its score cannot make it the submitted model.
    selected = 'xgboost'
    write_json(output / 'frozen_decisions.json', {'selected_model': selected, 'decisions': decisions,
        'selection_basis': 'XGBoost is the eligible Apache-2.0 model; threshold chosen on calibration only; linear diagnostic is not submission eligible'})
    event(stage='calibration_complete', selected_model=selected, decisions=decisions)
    _, validation_meta = build_fold(args.splits, 'validation', args.validation_anchors,
        vectorizers, output / 'validation', args.cache, k=args.k, seed=args.seed,
        threads=args.threads, target_batch=args.target_batch,
        query_feature_limit=getattr(args, 'query_feature_limit', None), cache_workers=getattr(args, 'cache_workers', 1))
    if feature_names != validation_meta['feature_names']:
        raise ValueError('Validation feature schema differs')
    validation = load_arrays(output / 'validation/candidates')
    X = np.load(output / 'validation/features/X.npy', mmap_mode='r')
    results = {}
    for name, path in paths.items():
        scores = predict_scores(path, X)
        np.save(output / f'validation/{name}_scores.npy', scores)
        results[name] = next(iter_threshold_array_metrics(validation['truth_counts'], validation['anchors'],
            validation['target_codes'], validation['labels'], scores,
            thresholds=[decisions[name]['threshold']]))
    result = {'status': 'COMPLETE', 'selected_model': selected, 'calibration': decisions,
              'validation': results, 'runtime_seconds': time.monotonic() - started,
              'scope': 'sampled anchors against entire fold target pools',
              'public_leaderboard_comparable': False,
              'caveats': ['Training data has no France examples; target test distribution differs.',
                         'Pilot validation is sampled, not the full holdout.']}
    write_json(output / 'result.json', result)
    event(stage='experiment_complete', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', default='work/splits_v1')
    parser.add_argument('--vectorizers', default='work/vectorizers_v1.joblib')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', default='work/retrieval_cache_v1')
    parser.add_argument('--train-anchors', type=int, default=6000)
    parser.add_argument('--calibration-anchors', type=int, default=3000)
    parser.add_argument('--validation-anchors', type=int, default=3000)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--target-batch', type=int, default=250000)
    parser.add_argument('--query-feature-limit', type=int)
    parser.add_argument('--cache-workers', type=int, default=1)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
