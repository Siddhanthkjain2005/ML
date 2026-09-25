"""Entity-level diagnostics for a frozen decision; never selects thresholds."""
import json
from pathlib import Path
import numpy as np
from .pipeline import write_json


def analyze(candidate_dir, scores_path, threshold, output_dir, seed=42):
    folder = Path(candidate_dir)
    queries = [json.loads(line) for line in (folder/'queries.jsonl').open()]
    anchors = np.load(folder/'anchors.npy', mmap_mode='r')
    labels = np.load(folder/'labels.npy', mmap_mode='r')
    truth = np.load(folder/'truth_counts.npy', mmap_mode='r')
    scores = np.load(scores_path, mmap_mode='r')
    if len(scores) != len(anchors) or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Invalid scores/threshold')
    chosen = scores >= threshold
    n = len(queries)
    predicted = np.bincount(anchors[chosen].astype(np.int64), minlength=n)
    tp = np.bincount(anchors[chosen & (labels == 1)].astype(np.int64), minlength=n)
    retrieved = np.bincount(anchors[labels == 1].astype(np.int64), minlength=n)
    denominator = predicted + .25*truth
    f = np.divide(1.25*tp, denominator, out=np.ones(n), where=denominator>0)
    if np.any(retrieved > truth): raise ValueError('Positive labels exceed truth counts')
    def summary(mask):
        count = int(mask.sum())
        return {'entities': count, 'macro_f0_5': float(f[mask].mean()) if count else None,
                'false_positives': int((predicted[mask]-tp[mask]).sum()),
                'false_negatives': int((truth[mask]-tp[mask]).sum()),
                'unretrieved_true_links': int((truth[mask]-retrieved[mask]).sum())}
    countries = np.asarray([q['country'] for q in queries])
    groups = {f'country:{c}': summary(countries==c) for c in sorted(set(countries))}
    groups['singleton'] = summary(truth==0)
    groups['multiple_matches'] = summary(truth>1)
    groups['non_ascii_name'] = summary(np.asarray([not q['business_name'].isascii() for q in queries]))
    rng=np.random.default_rng(seed)
    boot = [float(f[rng.integers(0,n,n)].mean()) for _ in range(400)] if n else []
    worst=[]
    for i in np.argsort(f,kind='stable')[:50]:
        worst.append({**queries[i], 'f0_5':float(f[i]), 'true_matches':int(truth[i]),
                      'tp':int(tp[i]), 'fp':int(predicted[i]-tp[i]), 'fn':int(truth[i]-tp[i]),
                      'unretrieved':int(truth[i]-retrieved[i])})
    result={'threshold':float(threshold),'overall':summary(np.ones(n,dtype=bool)), 'groups':groups,
            'bootstrap_macro_95pct':np.quantile(boot,[.025,.975]).tolist() if boot else None,
            'bootstrap_note':'400 entity bootstrap replicates; sampling uncertainty only, not test distribution shift',
            'worst_entities':worst}
    write_json(Path(output_dir)/'error_analysis.json',result)
    return result
