# Recovery checkpoint — 2026-09-26 01:55 UTC

## Access restored — 2026-09-26 01:57 UTC

User reauthenticated. Instance still running. Workstation IP changed; ingress updated to49.37.249.3/32. Auto-stop renewed to2026-09-26 13:57UTC within100USDcap. Original baseline completed: validation0.9548132363, calibration0.9510222571, threshold0.366. Results/models backed up experiments/aws/. Deeper-scorer experiment selected depth5_long on calibration0.9540594; validation0.9553004 (marginal improvement). Advanced number/romanized comparison features now building on TRAIN and CALIBRATION only, log experiments/augment_pilot_v1.log. Source and scripts in workspace. Deadline question pending.

## Previous access blocker (resolved)
AWS profile amazon-ml session expired. Existing SSH 3.110.80.56 timed out. User asked to reauthenticate; no current machine state verified. Do NOT create a replacement machine before checking i-09833693cad605fca and recovering its disk. Auto-stop was scheduled bootstrap +12h (approximately 2026-09-26 02:18 UTC). A changed client public IP may also explain SSH failure because security group allows only 49.37.251.169/32. After login, inspect status/public address/security group and recover saved results.

## Cloud state
AWS Mumbai r7i.xlarge 4vCPU32GiB,100GBgp3. Instance i-09833693cad605fca; saved IP3.110.80.56. Root profile amazon-ml explicitly authorized; total project cap100USD. Credit-only restriction explicitly lifted. No Azure resources created. Estimated12h budget8USD; compute0.273USD/hour. STOP retains billable disk; copy results before termination. Key .cloud-secrets/amazonml_phase1, SGsg-0dc2b627a847ae0ce; never package keys. CPU quota increase32 requested, ID6592229305524221889d3ae31600b051r0Ili7rH, last status CASE_OPENED.

## Active remote work last observed 2026-09-25 15:11 UTC
Remote root /home/ubuntu/amazonml. Main job: experiments/pilot_v1.log, 6000 training /3000 calibration /3000 validation anchors, entire corresponding target pools. Last observed training S3 retrieval10/17 chunks; training S2 finished. Job may have completed since then, but model results have NOT been recovered or inspected. Read experiments/pilot_v1/result.json and frozen_decisions.json first. No real test inference was launched.
Romanized vocabulary fit completed99.38sec at work/vectorizers_romanized_v1.joblib. PyICU2.16.2 installed with ICU system dependency. Romanized training recall probe running at experiments/roman_probe.log using scripts/approximate_probe.py. Read its report before accepting representation. Synthetic shared-index inference test launched at experiments/inference_smoke_v2.log with strict and official validators; completion not yet read.

## Verified completed work
Phase0 full data audit; identity-disjoint80/10/10 splits work/splits_v1; train-only200k vocabulary work/vectorizers_v1.joblib; oracle v2; complete pipeline and batched inference. Earlier synthetic full training/calibration/validation + strict output tests passed. Latest complete unit suite before parallel-index addition:99 passed. Retrieval-specific suite after parallel-index addition:8 passed. Local source includes tested shared SQLite raw-record lookup avoiding repeated target scans. No external business enrichment.

## Measured search findings (training only; not leaderboard/model scores)
1000 fixed training anchors,1689 known S2 links, entire S2 training target pool; sparse branches only. Full-query k20 baseline recalled1635/1689 (~96.8%). Distinctive12-fragment k20 recalled1566 (92.7176%), lost74/gained5 vs baseline: rejected. Distinctive32-fragment k40 recalled1631 (96.5660%), lost15/gained11, runtime48.29sec with cached target matrices. Not selected yet. Main full-query retrieval sped153sec/shard to28sec by reusing CSR transpose in tie fallback, then ~14sec/shard by skipping ties strictly below existing global kth score. Both preserve exact candidate rankings and passed tests.

## Next actions
1. Restore AWS session and inspect existing instance; update restricted ingress only to current client IP if needed.
2. Download remote experiment reports, trained model artifacts, frozen decisions and scores before shutting down/terminating.
3. Inspect candidate recall + calibration errors, then independent validation without tuning on it. Public target shown0.985884 is not comparable to sampled internal validation.
4. Assess romanized retrieval probe; retain only empirically justified variants. Final model must be MIT/Apache2.0: XGBoost eligible; sklearn linear diagnostic only.
5. Run full test inference only with frozen evaluated model/config, validate both output TSVs with strict and original --check-ids validators, package reproducible code + filled methodology. No portal submission authorized.

Local checkpoint0040cc2 precedes later optimization files; git working tree contains those intentional changes. Never revert them blindly. Current source/test directories are authoritative; old ASTRA_PROJECT_STATE.md was stale before this recovery note.
