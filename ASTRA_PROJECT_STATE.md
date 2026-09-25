# Verified project state — 2026-09-25

Phase 0 complete; evidence in PHASE0_AUDIT.md and audit/*.json. Full baseline experiment now running on AWS.

- Inputs: project.zip; ps.docx. Source schemas and labels internally consistent.
- Train S1/S2/S3: 2,206,821 / 5,034,616 / 5,285,603. Test S1/S2/S3: 1,732,544 / 4,887,273 / 5,082,316.
- Positive links: 7,638,365. Training singletons: 123,247 (5.58482088%). France test S1: 259,452.
- Current best competition score and candidate recall: NOT YET MEASURED. Synthetic smoke test passed but is not a competition performance result.
- Metric: per-S1 macro F0.5, empty/empty=1; multiple matches allowed. No external business enrichment or test-label access.
- Completed: identity-disjoint 80/10/10 full split, train-only 200k vocabulary sample, positive oracle v2, three sparse retrieval branches plus exact rescues, pair features with actual cosine, linear/XGBoost models, calibration-only threshold selection, strict submission validator.
- Foundation/integration checks: 93 unit tests passed before latest inference/analysis tests; synthetic end-to-end training/calibration/validation passed on AWS.
- Active: AWS pilot6000 train /3000 calibration /3000 validation anchors, complete fold target pools (unlinked distractors included). Remote log experiments/pilot_v1.log.
- Local compute: 8 GiB RAM. AWS instance i-09833693cad605fca, 3.110.80.56, Mumbai, r7i.xlarge 4vCPU32GiB,100GBgp3, automatic stop +12h. User explicitly authorized root profile amazon-ml and total US$100 cap. Latest user lifted credit-only restriction. Estimated12h total budget8USD. Terminate after copying results; STOP retains billable disk.
- Azure Students available but unused; no Azure resources created.
- Known weaknesses: test France domain shift, cross-script names, missing addresses, fulltest runtime unmeasured. Public leader0.985884; cannot compare sampled internal score directly.
- Next: measure actual pilot candidate recall and errors, improve on calibration evidence, freeze decision and report independent validation, produce fulltest outputs, strict+official checks, package documentation/code. No portal submission without explicit instruction.
