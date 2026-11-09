# Complete forecast-gate-bypass receipt — fitting decision

## Mandate and decision requested
User delegates scientific direction to frontier; dispatcher executes/verifies, never chooses research direction. Return explicit next authorization. No fits have run. The prerecorded stop rule is triggered: retained clean all-frame exact collision floor exceeds .010 in ALL3 grouped folds and12-design average. Stop this existing representation experiment; do not reinterpret as raw-waveform impossibility or alter acquisition envelope/denominators. Please assess receipt, batching delta, and select next mechanism checkpoint (or stop) with precise authorization and prerequisites. Do not ask user for routine scientific choices.

## Frozen execution and population
Source support_modal_20261009_v2/dataset.npz:2094 rows ×124 native frames,259656 frames,20 development designs. Source record indices preserved once; no new acquisition. Grouped endpoint population: original12 additional TRAIN designs,3 folds ×4 evaluation designs. Each original primary-noise population has120 rows/14880 frames. Original original4+8 fitting and2 checkpoint-validation memberships unchanged; no reserved evaluation. All13 noise views and original5 profiles retained. Measured waveform, original51-sample calibration, float32 calibrated128 samples,16 sensors and original CUDA estimator/174-coordinate assembly only. Labels never read by extraction; scoring truth used only after combined eligibility frozen. Qualified flags and age never changed.

| 12-design condition | Accepted | Eligible rejected | Full no-mode failures | Prohibited |
| --- | --- | --- | --- | --- |
| clean | 7306 | 140 | 2601 | 4784 |
| iid1 | 1805 | 8293 | 0 | 4782 |
| correlated1 | 1695 | 8403 | 0 | 4782 |

Full20 inventory:35362 cached accepted,136969 genuinely eligible rejected;141271 rejected full extraction trials,4302 no-mode failures,0 full-invalid estimates,0 unexpected numerical/solver exceptions. All eligible rows are finite-skill failures;4226 prefix-no-mode cases exist in clean inventory but NONE yielded eligible full128 rows. Undefined suffix-variance cases0. Combined and finite-only sensitivity are exactly identical; this was not outcome-selected. Forecast-pass originally unobserved cases108 across20 are separately preserved, not relabeled as eligible. Per-record design/profile/noise/replicate counts recorded in all20 replay JSONs and analysis.json.

## Exact floors and comparator leverage
Model inputs are actual float32 normalized174 features plus unchanged observed/age,3 slots at lags(4,2,0), same leading zero padding. Original train-accepted mean/scale/active identical across42/43/44. Within each fold/noise/original stratum group EXACT byte-identical inputs (no binning/rounding or digest-only equality). Bound=sum over classes of minimum sum(w|truth-c|), c=weighted median. Original equal-design weights1/(4×stratum frame count for that design), never renormalized over affected/missing/collisions. Fold-specific predictors: classes never pooled across folds. Average3 four-design bounds equals original12-design equal weighting. Native exact boundary is -.02<truth<.05, not a changed mask.

| Noise | Stratum | Retained exact floor | Zeroing exact floor | Target | Actually changed weight | Old GRU unaffected error |
| --- | --- | --- | --- | --- | --- | --- |
| clean | all | 0.012942401 | 0.013573036 | 0.010 | 2.796% | 0.030607015 |
| clean | boundary | 0.002653529 | 0.002673371 | 0.005 | 0.587% | 0.026068018 |
| iid1 | all | 0.001404834 | 0.040023479 | 0.015 | 65.753% | 0.011590731 |
| iid1 | boundary | 0.001131990 | 0.008739377 | 0.010 | 49.929% | 0.015610944 |
| correlated1 | all | 0.001404834 | 0.042034559 | 0.015 | 66.458% | 0.011567041 |
| correlated1 | boundary | 0.001131990 | 0.009149106 | 0.010 | 51.174% | 0.015564917 |
| iid1_fresh20261020 | all | 0.001405931 | 0.040246420 | 0.015 | 65.840% | 0.011633547 |
| iid1_fresh20261020 | boundary | 0.001131378 | 0.008756083 | 0.010 | 49.975% | 0.015666820 |
| correlated1_fresh20261020 | all | 0.001404062 | 0.042274332 | 0.015 | 66.230% | 0.011639002 |
| correlated1_fresh20261020 | boundary | 0.001131990 | 0.009180964 | 0.010 | 50.832% | 0.015651718 |
| iid1_fresh20261021 | all | 0.001404834 | 0.040541779 | 0.015 | 65.800% | 0.011604828 |
| iid1_fresh20261021 | boundary | 0.001131990 | 0.009212237 | 0.010 | 49.845% | 0.015547112 |
| correlated1_fresh20261021 | all | 0.001405817 | 0.042055611 | 0.015 | 66.324% | 0.011662879 |
| correlated1_fresh20261021 | boundary | 0.001133684 | 0.009257652 | 0.010 | 50.834% | 0.015772586 |

| Clean fold | Exact all-frame floor | Histories still no permitted modal content |
| --- | --- | --- |
| 0 | 0.012881995 | 35.827% |
| 1 | 0.013693978 | 39.698% |
| 2 | 0.012251230 | 36.351% |

All clean collision floor contribution comes from histories still without permitted modal content; weighted-median classes include truth ranges spanning negative to >.21. This is an input-equality statement for this representation, not lack of measured waveform information. Clean newly eligible content affects only2.796% all-frame and0.587% boundary equal-design weight. At1%, actual changes cover65.75–66.46% all-frame and49.93–51.17% boundary; fresh draws similar. Potential noisy leverage exists but cannot overcome joint clean endpoint foreclosure.

Perfect-correction columns leave old unaffected predictions fixed; they are NOT lower bounds on a globally refitted model. Clean old contentGRU all/boundary MAE .031671988/.026250070 would only reach .030607015/.026068018 with perfect affected corrections. Noisy boundary unaffected contribution remains about.01555–.01577 vs.010, but this is leverage, not a global refit theorem.

| Noise | Stratum | Comparator | Old MAE | Unchanged contribution |
| --- | --- | --- | --- | --- |
| clean | all | content_gru | 0.031671988 | 0.030607015 |
| clean | all | ordered_concat | 0.030999530 | 0.030004561 |
| clean | all | unordered_pool | 0.031013632 | 0.030008517 |
| clean | all | framewise_reference | 0.036314122 | 0.034463320 |
| clean | boundary | content_gru | 0.026250070 | 0.026068018 |
| clean | boundary | ordered_concat | 0.026038045 | 0.025857051 |
| clean | boundary | unordered_pool | 0.026161287 | 0.025982394 |
| clean | boundary | framewise_reference | 0.031075068 | 0.030809609 |
| iid1 | all | content_gru | 0.048306818 | 0.011590731 |
| iid1 | all | ordered_concat | 0.048261624 | 0.011526685 |
| iid1 | all | unordered_pool | 0.048082900 | 0.011645739 |
| iid1 | all | framewise_reference | 0.051449739 | 0.013094927 |
| iid1 | boundary | content_gru | 0.034644460 | 0.015610944 |
| iid1 | boundary | ordered_concat | 0.034753201 | 0.015545260 |
| iid1 | boundary | unordered_pool | 0.034539883 | 0.015738964 |
| iid1 | boundary | framewise_reference | 0.039375501 | 0.018264021 |
| correlated1 | all | content_gru | 0.050112394 | 0.011567041 |
| correlated1 | all | ordered_concat | 0.050334151 | 0.011559321 |
| correlated1 | all | unordered_pool | 0.050228004 | 0.011695456 |
| correlated1 | all | framewise_reference | 0.053095350 | 0.013151729 |
| correlated1 | boundary | content_gru | 0.035285454 | 0.015564917 |
| correlated1 | boundary | ordered_concat | 0.035792419 | 0.015578562 |
| correlated1 | boundary | unordered_pool | 0.035681434 | 0.015811403 |
| correlated1 | boundary | framewise_reference | 0.040045881 | 0.018428888 |

Full156 fold/variant/noise/stratum reports include every comparator,seed and per-design contribution and original-denominator masks.5% stress also included, no target-pass claim.

## Reproduction and batching — no equivalence claim
All35362 accepted rows preserved BYTE-FOR-BYTE, all replayed successfully;0 qualification gate mismatches. Full diagnostic eight-mode count/order/availability mismatch0. Cached feature count and availability mismatch0; one cached first3 modal-frequency ordering mismatch. Rebuilt feature values are not identical: full-inventory raw max4.473014311 (D000042 cross replicate1 clean frame71,coordinate10 = second energy-ranked modal alpha). Standardized max.415999/.459203/.439577 across old fold scalings; global fold1 peak coordinate22(first energy-mode damping ratio), same old canonical validation record. Every discrepancy belongs to original_cache:9427 accepted rows; original_added_regime2052 and additional_trajectory23883 accepted rows replay exactly. Cached values NEVER replaced, no tolerance adjustment. This is not certified extraction equivalence.

Your required smoke discrepancy attribution: raw .0087635455 at coords16(max-visible alpha) and27(least-stable visible alpha); active standardized max .0014807480 at coord89(least-stable-mode normalized imaginary cross-Gramian for sensor-pair7). Canonical batching in original fitter_comparison_acquire._process is per-frame source-noise groups; proposed/support batching is per-noise-row windows.

Prespecified six-window paired panel: fixed D000007/cross/rep0 clean/IID1/IID5 first accepted +first eligible rejected, identical calibration/samples. All six mode-count/forecast/frequency-order/spatial-selection states agree; repeat assembly byte identical; maximum canonical/proposed standardized difference9.6758e-13, rejected coefficient relative L2 max7.4429e-14. This panel alone did NOT explain earlier discrepancy.

Separate deterministic reproduction-error-triggered panel frozen before its paired comparison selected the same D000007 clean frame42 for both raw and standardized peaks (ONE window, two triggers, not independent replication). Canonical feature row matches cached row exactly; proposed batch reproduces .0087635455 raw/.0014807480 standardized discrepancy exactly. Both7 modes; forecast,frequency slot order and least-stable visible order unchanged; same assembly repeat byte identical. Within-mode complex coefficient relative L2 difference.00050292264. Coefficient-only swap accounts for spatial standardized maximum.0014807480; remaining poles/energy component maximum.0014373752, not additive contributions. Thus batching sensitivity is demonstrated for that window; not proven absent elsewhere. Larger full-inventory discrepancies remain disclosed, not waved away. No further equivalence diagnosis is needed to enforce the independent clean collision stop rule; frontier may decide relevance to future extraction.

## Independent verification and runtime
Real original GTX1660SUPER population extraction completed; mathematical smoke catches unequal design weights, weighted-median collisions, one-ULP non-collision and383ms warmup/history propagation. Independent dictionary-by-bytes implementation rebuilt primary populations and explicit slot assignments:42 fold×noise×stratum checks; max floor discrepancy1.0408341e-17, max comparator contribution discrepancy6.9388939e-18. Source records once, eligibility finite/genuine/nonempty, accepted cache bytes, original denominator and protected ledger hashes verified. Original old score MAEs reproduced for ALL156 reports and4 comparators ×3 seeds.

Extra rejected full-mode extraction473.912s; combined accepted/rejected feature assembly258.058s, prefix544.021s, forecast158.037s, accepted full120.763s. Total command wall1765.06s includes imports/source checks/serialization. These are diagnostic stage clocks, not full predictor runtime or replay qualification; extra work not hidden. No runtime/availability/warning/uncertainty or independent physics credit.

## Evidence and frozen limits
All artifacts under p4_groups/operational_confirmation_20261003/gate_bypass_receipt_20261010/:protocol.json;replay_<20design>.npz/.json;inventory_summary.json;analysis.json;verification.json;primary_summary.json;grouped_inventory_and_controls.json;reproduction_attribution.json;batching_protocol.json/batching_characterization.json/batching_summary.json;batching_discrepancy_protocol.json/batching_discrepancy_characterization.json/batching_discrepancy_summary.json;smoke/discrepancy_coordinates.json. Executables extract_gate_bypass_receipt.py,analyze_gate_bypass_receipt.py,verify_gate_bypass_receipt.py,characterize_gate_bypass_batching.py. Source-level claims grounded in experiment_support_modal_v2._make_row_result/_row_modes/noise_for, pulse_aware_probe.forecast_skills, fitter_comparison_acquire._process, modal_features.modal_feature_row/_fill_spatial_pattern.

analysis SHA256 781ab030c563b11d67ad2c912b4e207f2d777a282c1615bad278b03d61815f5a. Protected P4_PROGRESS.json remains ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f. Predictor and reserved designs unchanged. Controlled multipoint acquisition remains paused for independent accelerating-flow validity, not revived. No narrowed envelope, no feature/loss/architecture sweep, no fitted-point improvement or endpoint reachability claimed.
