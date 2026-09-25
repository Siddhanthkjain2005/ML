# Business entity resolution

This pipeline matches each Source1 record to zero or more Source2/Source3 records. It preserves Unicode, empty fields and literal names such as `nan`. It does not use external business data or country-specific filters.

## Environment

Python 3.12; install `requirements.txt` in an isolated environment. Run commands from the project root with `PYTHONPATH=code/business_entity_resolution/src`. The saved input ZIP is not included in source control or submission code.

## Evaluation

`ber.data.prepare_splits` creates identity-disjoint training, calibration and validation folds; all labeled target records inherit the owner's fold, and unlinked targets are deterministically partitioned. Vocabulary fitting uses only training records. Each sampled query searches the entire target pool in its fold. Labels never generate candidates or features.

```
PYTHONPATH=code/business_entity_resolution/src .venv/bin/python -m ber.pipeline \
  --output experiments/pilot_v1 --train-anchors 6000 \
  --calibration-anchors 3000 --validation-anchors 3000
```

The pipeline fits linear and XGBoost pair classifiers, selects model and threshold using entity macro F0.5 on calibration, then evaluates validation with frozen decisions. Missing retrieved positives count as false negatives. Empty truth and prediction sets score one. A sampled validation result is not an estimate guaranteed to match the public leaderboard, especially because France is absent from training.

## Final inference

After selecting and freezing a model, use `python -m ber.inference --help`. Input directory contains `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`. Inference writes `matching_results.tsv` and `candidate_pairs.tsv`, with exactly one row per input Source1 record. Candidates are the exact set scored by the classifier. There is no global one-to-one restriction, forced match, or fixed predicted-match count.

Inference processes bounded query batches, retains committed output partitions and scores, and discards regenerable matrices by default. Interrupted batches resume; changed inputs or configuration are rejected. Final output must pass `ber.strict_validate` and the original official validator with `--check-ids` before submission. The inference command does not submit to any portal.

## Reproducibility and limits

Seeds, input hashes, configuration, feature schemas and library versions are recorded with artifacts. Cache manifests reject mismatched inputs. Candidate recall and the perfect-classifier ceiling are diagnostics, not measured model performance. Running time for full-test inference must be estimated from actual pilot timings. Experiment records and final model selection are stored separately from input data.
