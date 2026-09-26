"""Calibrate accelerated retrieval on the same saved pilot calibration anchors.

Reuses the pilot's complete calibration target pools and cached target matrices.
The scorer is frozen; both its original threshold and a new calibration-only
threshold are reported. This script never accesses validation or fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ber.augment import augment
from ber.metrics import iter_threshold_array_metrics
from ber.model import predict_scores, select_decision
from ber.pipeline import build_fold, event, load_arrays, sample_queries, write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def positive_pairs(arrays):
    indices = np.flatnonzero(arrays['labels'] == 1)
    return {(int(arrays['anchors'][i]), int(arrays['target_codes'][i])) for i in indices}


def run(args):
    started = time.monotonic()
    if args.threads < 1 or not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError('Use positive thread count and threshold between zero and one')
    pilot = Path(args.pilot_root).resolve()
    if any(part in ('validation', 'test', 'train') for part in pilot.parts):
        raise ValueError('Pilot root must not select another data partition')
    original_config = json.loads((pilot / 'config.json').read_text())
    limit, seed = original_config['calibration_anchors'], original_config['seed']
    if limit != 3000:
        raise ValueError(f'Expected the saved 3000-anchor pilot, found {limit}')
    splits = Path(original_config['splits'])
    queries = sample_queries(splits / 'calibration/source1.tsv', limit, seed)
    saved_queries = [json.loads(line) for line in (pilot / 'calibration/candidates/queries.jsonl').open()]
    if len(queries) != 3000 or queries != saved_queries:
        raise ValueError('Reconstructed calibration sample differs from saved pilot queries')
    model = Path(args.model).resolve()
    model_metadata = json.loads((model.parent / 'metadata.json').read_text())
    feature_names = model_metadata['feature_names']
    if len(feature_names) != 112 or model_metadata.get('feature_directory') != 'features_advanced_v1':
        raise ValueError('Expected the frozen 112-feature advanced scorer')
    out = Path(args.output)
    if out.resolve() == pilot:
        raise ValueError('Accelerated experiment needs a distinct output directory')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(partition='calibration', pilot_root=str(pilot), pilot_config=original_config,
                  queries_sha256=digest(pilot / 'calibration/candidates/queries.jsonl'),
                  model=str(model), model_sha256=digest(model), threshold=args.threshold,
                  query_feature_limit=32, rerank_pool=100, k=20, cache_workers=1,
                  threads=args.threads, roman_vectorizers=args.roman_vectorizers,
                  roman_vectorizers_sha256=digest(args.roman_vectorizers),
                  vectorizers_sha256=digest(original_config['vectorizers']))
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Fast-retrieval experiment configuration changed; use a fresh output directory')
    write_json(config_path, config)
    vectorizers = joblib.load(original_config['vectorizers'])
    fold = out / 'calibration'
    event(stage='fast_calibration_start', anchors=len(queries), query_feature_limit=32,
          rerank_pool=100, k=20, threads=args.threads)
    with threadpool_limits(limits=args.threads):
        candidate_meta, _ = build_fold(splits, 'calibration', limit, vectorizers, fold,
            original_config['cache'], k=20, seed=seed, threads=args.threads,
            target_batch=original_config['target_batch'], query_feature_limit=32,
            rerank_pool=100, cache_workers=1)
        generated_queries = [json.loads(line) for line in (fold / 'candidates/queries.jsonl').open()]
        if generated_queries != saved_queries:
            raise ValueError('Accelerated candidate query order differs from original pilot')
        advanced = augment(fold / 'candidates', fold / 'features', args.roman_vectorizers,
                           fold / 'features_advanced_v1')
        if advanced['feature_names'] != feature_names:
            raise ValueError('Accelerated feature schema differs from frozen model')
        arrays = load_arrays(fold / 'candidates')
        X = np.load(fold / 'features_advanced_v1/X.npy', mmap_mode='r')
        scores = predict_scores(model, X)
    np.save(fold / 'frozen_advanced_scores.npy', scores)
    fixed = next(iter_threshold_array_metrics(arrays['truth_counts'], arrays['anchors'],
        arrays['target_codes'], arrays['labels'], scores, [args.threshold]))
    tuned = select_decision(arrays['truth_counts'], arrays['anchors'], arrays['target_codes'],
                            arrays['labels'], scores)
    baseline = load_arrays(pilot / 'calibration/candidates')
    if not np.array_equal(arrays['truth_counts'], baseline['truth_counts']):
        raise ValueError('Truth counts differ despite identical query sample')
    base_pairs, fast_pairs = positive_pairs(baseline), positive_pairs(arrays)
    lost, gained = base_pairs - fast_pairs, fast_pairs - base_pairs
    def pair_example(pair):
        a, code = pair
        return dict(source1=queries[a]['entity_id'], target=f'S{code // 10**12}-{code % 10**12}',
                    country=queries[a]['country'])
    countries = sorted({q['country'] for q in queries})
    comparison = dict(baseline_retrieved_positives=len(base_pairs), fast_retrieved_positives=len(fast_pairs),
                      lost_positive_links=len(lost), gained_positive_links=len(gained),
                      baseline_candidate_pairs=len(baseline['anchors']), fast_candidate_pairs=len(arrays['anchors']),
                      by_country={c: dict(lost=sum(queries[a]['country'] == c for a, _ in lost),
                                          gained=sum(queries[a]['country'] == c for a, _ in gained)) for c in countries},
                      lost_examples=[pair_example(p) for p in sorted(lost)[:30]],
                      gained_examples=[pair_example(p) for p in sorted(gained)[:30]])
    prior_decision = model.parent / 'calibration.json'
    if prior_decision.exists():
        baseline_metric = json.loads(prior_decision.read_text())
        if baseline_metric.get('threshold') == args.threshold:
            comparison['baseline_fixed_threshold_metrics'] = baseline_metric
            comparison['macro_f0_5_delta_at_fixed_threshold'] = fixed['macro_f0_5'] - baseline_metric['macro_f0_5']
    result = dict(status='COMPLETE', partition='calibration', model=str(model), frozen_model=True,
                  fixed_threshold=fixed, calibration_tuned_threshold=tuned,
                  candidate_recall=candidate_meta['stats']['label_diagnostics'],
                  candidate_comparison=comparison, runtime_seconds=time.monotonic() - started,
                  scope='Same 3000 calibration anchors against complete calibration target pools',
                  caveats=['Threshold tuning here uses calibration only; it is not a validation result.',
                           'Scores are not directly comparable with the public leaderboard.',
                           'Scorer was frozen throughout; candidate ranks and retrieval scores can change under acceleration.'])
    write_json(out / 'result.json', result)
    event(stage='fast_calibration_complete', fixed_threshold=fixed, calibration_tuned_threshold=tuned,
          candidate_recall=candidate_meta['stats']['label_diagnostics']['overall'],
          lost_positive_links=len(lost), gained_positive_links=len(gained),
          runtime_seconds=result['runtime_seconds'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', default='experiments/pilot_v1')
    parser.add_argument('--model', default='experiments/advanced_refinement_v1/depth7/model.json')
    parser.add_argument('--roman-vectorizers', default='work/vectorizers_romanized_v1.joblib')
    parser.add_argument('--threshold', type=float, default=.406)
    parser.add_argument('--output', default='experiments/fast_calibration_v1')
    parser.add_argument('--threads', type=int, default=3)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
