# Phase 0 forensic audit

Completed 2026-09-25 before Phase 1 implementation. Source files were read without modification; audit scratch files were written under /tmp. AWS STS identity was checked, with no cloud resources created. This report records observed facts separately from proposed experiments.

## 1 Repository map

The workspace originally contained only `project.zip` and `ps.docx`. The ZIP has 34 members, including ten substantive files (2,520,603,693 uncompressed bytes); other members are directories and macOS metadata.

```text
project.zip / student_resource/
  README.md
  Documentation_template.md
  utils/validate_submission.py
  dataset/train/train_source1.tsv
  dataset/train/train_source2.tsv
  dataset/train/train_source3.tsv
  dataset/train/train_ground_truth.tsv
  dataset/test/test_source1.tsv
  dataset/test/test_source2.tsv
  dataset/test/test_source3.tsv
```

There was no Git working tree, source implementation, notebook, configuration, model artifact, requirements file, experiment record, prediction, or candidate output. The methodology template contains placeholders. Evidence: complete ZIP inventory and complete reads of both Markdown files and the sole Python file; `git rev-parse` found no repository. The Word problem statement was extracted completely and all seven rendered pages inspected; its substantive rules agree with the README.

## 2 Actual current pipeline

No data-to-prediction pipeline exists. The only executable is the supplied validator:
`main` → `validate` → `read_ids` → optional `load_match_targets` → `validate_id_list_file` → candidate subset warnings → exit status. It checks submitted outputs and never trains or computes the challenge metric.

## 3 Dataset shapes and missing values

Full-file streaming measurements, not samples. Every source file has four columns: `entity_id`, `business_name`, `business_address`, `country`.

| File | Rows | US | India | France | Empty addresses |
|---|---:|---:|---:|---:|---:|
| train_source1.tsv | 2,206,821 | 1,323,633 | 883,188 | 0 | 0 (0.0000%) |
| train_source2.tsv | 5,034,616 | 3,016,817 | 2,017,799 | 0 | 168,967 (3.3561%) |
| train_source3.tsv | 5,285,603 | 3,170,056 | 2,115,547 | 0 | 175,916 (3.3282%) |
| test_source1.tsv | 1,732,544 | 663,106 | 809,986 | 259,452 | 0 (0.0000%) |
| test_source2.tsv | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 | 129,408 (2.6479%) |
| test_source3.tsv | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 | 136,098 (2.6779%) |

Across all six files: no empty IDs, names, or country fields; no whitespace-only fields; no malformed column counts; no unexpected ID prefixes; no duplicate IDs within a source file. Train/test ID intersections are zero for each source. Common null-like name literals are retained as text, not automatically interpreted as missing: train S2 has one `nan` and five `na`; train S3 has eighteen `na`; test S2 has forty-nine `na`; test S3 has sixty `na` and one `null` (counts after strip/casefold).

Train S1 names are all ASCII. Non-ASCII names occur in 764,608 train S2 rows and 606,737 train S3 rows; 40,789 test S1 names also contain non-ASCII characters. Non-ASCII alone does not prove transliteration or establish a true pair. Raw and basic-normalized business-tuple fingerprints are unique within train/test S1. Target sources contain duplicate-looking tuples: 25,060 raw fingerprint groups in train S2, 18,381 in train S3, 21,909 in test S2, and 15,861 in test S3. Different IDs must be preserved.

No business-tuple fingerprint overlap was found in any of the nine train-source/test-source comparisons, either raw or NFKC/casefold/whitespace-normalized. These are BLAKE2b-128 fingerprint checks, not fuzzy identity comparisons. Field separator presence is recorded; a positive fingerprint match is not proof of business identity. Complete length distributions, Unicode counts, and scan timings are in `audit/source_statistics.json`.

## 4 Ground-truth structure and distribution

`train_ground_truth.tsv`: 2,206,821 rows × 2 columns, `source1_entity_id` and `matched_entity_ids`. The second column is a comma-separated ID string; an empty field means a true singleton, not a missing annotation.

There are 7,638,365 positive links: 3,693,619 S2 and 3,944,746 S3. All S1s are covered exactly once; every target exists in its training source. No duplicate target within a list, malformed ID, or target shared across distinct S1 label rows was found. There are 1,340,997 S2 and 1,340,857 S3 records without a labeled S1 link. All observed positive links agree in country.

| True matches per S1 | Number of S1 entities |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

Mean matches per S1: 3.461253. Multiple matches occur for 1,964,417 S1s (89.0157%). Maximum: eleven overall, five S2 and six S3. Target ownership is unique in the observed training labels; this does not justify imposing a new global assignment constraint on test.

## 5 Singleton proportion

123,247 / 2,206,821 = **5.58482088%**. US: 73,896 / 1,323,633; India: 49,351 / 883,188. Predicting empty lists for every training S1 has an arithmetic full-training-label score of 0.0558482088. This is a diagnostic derived from label counts, not a validation/model result.

## 6 Existing blocking strategy

Absent. No existing candidates or retrieval implementation. All-pairs train comparison would require 22,774,876,013,799 pairs; even same-country all-pairs requires 11,839,670,856,657. These are cardinality calculations, not a candidate-generation experiment.

## 7 Existing features

Absent. Raw name, address, country and ID fields are provided, but no normalization or model features are implemented.

## 8 Existing matching model

Absent. No model, checkpoint or pretrained artifact is present. Local Python 3.12.5 successfully imported NumPy 2.1.3, SciPy 1.16.1, pandas 2.2.3, scikit-learn 1.7.2 and XGBoost 3.1.2. XGBoost installed metadata says Apache-2.0; scikit-learn metadata says BSD-3-Clause. This is availability evidence, not training or final-model license approval. RapidFuzz, LightGBM, CatBoost, sentence-transformers and sparse-dot-topn were not installed in that interpreter. Polars metadata reports 1.42.0, but constructing a DataFrame failed with `NameError: PySeries is not defined`; do not treat it as usable.

## 9 Existing validation design

Absent. No split, seed, scorer, fold manifest or calibration partition exists. The positive graph contains disjoint S1-centered stars in the observed labels, so the known labeled identities can be split by S1, assigning all linked targets to the same split. Unlinked distractor records need a deterministic disjoint assignment too.

## 10 Existing threshold and post-processing

Absent. No probability calibration, threshold, singleton rule, top-k decision or assignment rule exists.

## 11–14 Current measured performance

- Candidate recall: **UNKNOWN — must be measured.** Missing blocker and candidate sets.
- Entity macro F0.5: **UNKNOWN — must be measured.** Missing split, scorer and predictions.
- Precision/recall: **UNKNOWN — must be measured.** Missing validation predictions.
- Singleton accuracy: **UNKNOWN — must be measured.** Missing validation predictions.

No training or model evaluation was run during Phase 0. No improvement or regression can be claimed. The official scorer implementation is not supplied; the README gives the mathematical definition and singleton behavior.

The count form to implement next is `1.25*TP / (1.25*TP + FP + 0.25*FN)`, with 1 when truth and prediction are both empty. Average per S1, including singletons. The README's informal “2×” language must not replace its explicit beta=0.5 formula. Its supplied example evaluates to 5/7 ≈ 0.7142857.

## 15 Leakage risks

Random pair splitting would expose the same S1/business across partitions, especially with 89.0157% multi-match S1s. Positive-linked records must inherit their complete group's split. No train/test ID or exact/basic-normalized tuple overlap was found, but near-duplicate identities were not exhaustively investigated. Whether duplicate-looking target tuples span different labeled S1 groups is UNKNOWN — must be measured if relevant to validation stability. Fit learned text statistics on training partitions; separate threshold calibration from final model assessment.

## 16 Rules and submission risks

No solution code exists to demonstrate an external-lookup or model-license violation. External business lookup/enrichment is prohibited; model licensing and size must be recorded before any pretrained use.

The supplied `utils/validate_submission.py` has important gaps, verified using its unchanged source on 21 synthetic CLI fixtures:

- Unknown IDs PASS by default; `--check-ids` makes them FAIL.
- Even with `--check-ids`, a missing target source file disables existence checking with a warning.
- Missing candidate files and matches outside candidates only warn and PASS.
- Headers with case/space changes pass; some extra-tab rows pass in default mode.
- Duplicate S1 rows/list targets, S1 self-targets, missing/extra S1s and invalid prefixes fail.

Evidence functions: `validate_id_list_file` lines 95–205, `load_match_targets` lines 71–91, `validate` lines 208–272, `main` lines 275–349. The README requires real IDs and complete outputs, while validator comments inconsistently claim unknown IDs only lower scores. Follow the stricter contract. Keep the official validator unchanged and add internal strict checks, including reconciliation with actual inference candidates. No real submission was generated or validated during Phase 0.

## 17 France and open-set risks

France contributes 259,452 / 1,732,544 = 14.9752% of test S1; it is entirely absent from training. Test S1 distribution is 38.2735% US, 46.7513% India, 14.9752% France. Training S1 is approximately 60% US / 40% India. Closed-country encoders/filters and ASCII-only processing would be unsafe. French model quality is UNKNOWN — must be measured on future available labels; US/India holdouts cannot certify it. Observed training country agreement alone does not prove hard country filtering safe on test.

## 18 Reproducibility and compute problems

No Git history, pinned environment, runnable entry point, experiment ledger, caches or saved split exists. Ignore binary `__MACOSX/._*.tsv` metadata when discovering source data. The Mac has 8,589,934,592 bytes physical RAM; approximately 19.1 GB free disk was observed before scratch creation. Dense all-pairs comparison is infeasible. Use bounded sparse retrieval, batches and persistent artifacts.

AWS identity verification succeeded with `amazon-ml`, `ap-south-1`, account 206632867498, ARN ending `:root`. Only STS was called: compute quotas, provisioning permissions, resource availability and cost were not established.

Input checksums:
- project.zip SHA256: 82c0d3400d5318f102d649be238313e987f968eb13e5cd0a386a175f8a1ce34a
- ps.docx SHA256: e6e058683518020ee503d32514dcd68f83c21fcfa7bd4f673130dbe37d858328

## 19 Ranked next improvements

Expected benefits below are hypotheses, not measured gains. Cost is relative to this dataset.

| Rank | Improvement | Expected benefit | Implementation risk | Experiment cost |
|---:|---|---|---|---|
| 1 | Exact metric tests and strict output contracts | Prevent invalid model selection/submission | Low | Low |
| 2 | Saved identity-group train/calibration/evaluation splits | Prevent leakage and threshold overfitting | Medium | Low |
| 3 | Complementary name/address sparse retrieval | Raise positive-link recall ceiling | Medium | High |
| 4 | Preserve raw Unicode and parallel normalization | Handle noise without destructive information loss | Low–medium | Medium |
| 5 | Same-retrieval hard negatives; compare linear and boosted models | Improve false-merge discrimination | Medium | Medium–high |
| 6 | Optimize thresholds on exact macro F0.5 | Align precision/singleton decisions with scoring | Low | Low |
| 7 | Address-number contradictions and name/address interactions | Reduce plausible false merges | Medium | Medium |
| 8 | Additional group seeds and country-held-out checks | Detect unstable improvements/domain shift | Medium | High |
| 9 | Versioned configuration, caches, ledger and pinned environment | Make results reproducible and affordable | Low–medium | Medium |
| 10 | Later licensed multilingual component only after lexical error analysis | Potential rescue for cross-script failures | High | High |

## 20 Exact first recommended experiment

First create tested scoring, deterministic identity splits and the ledger. Perform train/validation-only positive-pair oracle analysis to select sensible retrieval bounds. Then isolate the value of address retrieval: hold name retrieval fixed and compare name-only candidates with the union including address candidates, source by source. Start with a seeded uniform validation query sample against the complete held-out target pool, retaining unqueried anchors' targets and unlinked distractors. Measure standalone/incremental positive recall, per-S1 oracle F0.5 ceiling, candidate counts/distribution, runtime and memory; inspect every missed validation link. Empty addresses must yield no address candidates, and zero-similarity filler must be excluded. Confirm at the full selected validation scale before accepting the branch. Do not report retrieval diagnostics as classifier quality.

Phase 1's subsequently supplied requirements add positive-pair oracle analysis before parameter selection and full baseline model benchmarking; they take precedence over preliminary numeric top-K suggestions.

## Commands and evidence

Executed: complete ZIP inventory; full UTF-8 tab-delimited source/label streaming with Python zipfile/csv and compact NumPy arrays; exact ID joins/intersections and BLAKE2b-128 tuple comparisons; complete README/template/validator reads; DOCX extraction/render/inspection; original validator on 21 temporary fixtures; package import checks and Polars runtime smoke check; Git presence check; SHA256 checksums; read-only AWS STS identity check. Source passes total approximately 369.7 seconds; fingerprint/ID comparisons 19.5 seconds; label parsing/summary 21.49 seconds, with subsequent integrity joins. These are stage timings, not an end-to-end training runtime.

Machine-readable evidence: `audit/source_statistics.json`, `audit/ground_truth_statistics.json`. No model was implemented during this audit.
