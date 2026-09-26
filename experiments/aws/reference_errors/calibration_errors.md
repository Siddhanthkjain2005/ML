# Frozen scorer: calibration diagnostics

Threshold: 0.425. Entities: 3,000. Candidate pairs: 306,321.

Macro F0.5: 0.974998. False positives: 84. False negatives: 559 (336 retrieved but rejected; 223 never retrieved).

## Entity groups

| Group | Entities | Macro F0.5 | FP | Retrieved FN | Unretrieved FN |
|---|---:|---:|---:|---:|---:|
| country:India | 1161 | 0.964307 | 38 | 156 | 167 |
| country:US | 1839 | 0.981747 | 46 | 180 | 56 |
| singleton | 160 | 0.968750 | 5 | 0 | 0 |
| one_true_match | 151 | 0.945548 | 5 | 5 | 1 |
| multiple_true_matches | 2689 | 0.977023 | 74 | 331 | 222 |

## Retrieved candidate strata

Groups overlap; these counts cannot be added across rows. Recall here excludes retrieval misses.

| Group | Pairs | TP | FP | Retrieved FN | Precision |
|---|---:|---:|---:|---:|---:|
| country:India | 122334 | 3742 | 38 | 156 | 0.98995 |
| country:US | 183987 | 6152 | 46 | 180 | 0.99258 |
| target_name_non_ascii | 23856 | 1297 | 6 | 44 | 0.99540 |
| target_name_ascii | 282465 | 8597 | 78 | 292 | 0.99101 |
| name_script_disjoint | 11870 | 579 | 5 | 28 | 0.99144 |
| name_left_missing | 0 | 0 | 0 | 0 | N/A |
| name_right_missing | 0 | 0 | 0 | 0 | N/A |
| name_numbers_both_present | 1726 | 146 | 1 | 4 | 0.99320 |
| name_numbers_equal | 437 | 136 | 1 | 3 | 0.99270 |
| name_numbers_conflict | 1204 | 2 | 0 | 0 | 1.00000 |
| name_numbers_disagree | 1289 | 10 | 0 | 1 | 1.00000 |
| name_without_numbers_exact | 6581 | 2897 | 7 | 19 | 0.99759 |
| address_left_missing | 0 | 0 | 0 | 0 | N/A |
| address_right_missing | 7733 | 324 | 43 | 92 | 0.88283 |
| address_numbers_both_present | 265497 | 8764 | 33 | 217 | 0.99625 |
| address_numbers_equal | 7739 | 6504 | 10 | 38 | 0.99846 |
| address_numbers_conflict | 242845 | 386 | 9 | 98 | 0.97722 |
| address_numbers_disagree | 257758 | 2260 | 23 | 179 | 0.98993 |
| address_without_numbers_exact | 2502 | 1526 | 4 | 33 | 0.99739 |
| name_roman_ratio>=0.9 | 14336 | 4605 | 24 | 55 | 0.99482 |
| address_roman_token_sort>=0.9 | 5433 | 4711 | 11 | 79 | 0.99767 |
| address_first_number_edit>=0.75 | 12385 | 7644 | 16 | 94 | 0.99791 |
| name_char_cosine_zero_address_strong | 3386 | 536 | 10 | 47 | 0.98168 |
| cross_script_name_zero_address_strong | 912 | 451 | 1 | 14 | 0.99779 |

## Inspection priorities

The JSON contains score/feature bins, the 30 highest-score false positives, the 30 lowest-score retrieved false negatives, and entities with missing candidates.

Use the examples to form hypotheses; this report does not claim that a feature caused an error or propose untested decision rules.

Calibration metrics support development only. No validation or test partition was inspected.
