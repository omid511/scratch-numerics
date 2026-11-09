# Raw-response eligibility partition — verified decision receipt

## Authorization and scope
Frontier stopped gate-bypass fitting and authorized rejection metadata/per-sample waveform-mask partition only. All20 original measured shards read on original execution host:2094 records; no new acquisition, modal extraction, predictor calls or fits.12 grouped development evaluation designs,3 folds, original4+8 fit/2 validation memberships, every native frame and equal-design denominator unchanged.7 primary conditions plus6 stress conditions retained.

Source physical gate is per128 window (experiment_support_modal_v2._make_row_result): any command, any nonfinite sample, any calibrated float32 channel>=50 prohibits it. Operational/diagnostic normalization clips but does NOT latch a permanent first-cap cutoff. Consequently native masks exclude those actual samples; a SEPARATE conservative pre-cap sensitivity discards ALL samples from first trajectory cap onward. It is not adopted or presented as existing policy. No clipped value or proposed post-cap predictor input used.

Aperture is384 trailing physical samples, oldest383ms; absent warmup samples are absent, not invented data. Three modal slots4/2/0 tile it; overlapping native windows atlags3/1 stay inside the same aperture. Also report existence of any128-sample contiguous admissible run, not necessarily canonical alignment. Only shorter means longest run1..127; no_samples means zero admissible COMPLETE16-channel samples under the stated policy. Some-channel usable samples are not included; no-channel information theorem follows.

## Original clean all-frame collision floor attribution
Contributions below use each FULL original modal-input collision class weighted median. They sum to the old .012942400855 floor; do NOT add reoptimized subset floors or interpret each attribution as an independent unavoidable error.

| Policy | Raw category | Original weight | Modal floor attribution | Subset modal bound | Subset +mask/time bound | Old GRU error contribution |
| --- | --- | --- | --- | --- | --- | --- |
| native_sample_rule | full128 | 76.0551% | 0.008956323 | 0.007386851 | 0.002136469 | 0.021405673 |
| native_sample_rule | shorter_only | 1.6667% | 0.001315494 | 0.000079026 | 0.000000000 | 0.001562128 |
| native_sample_rule | no_samples | 22.2782% | 0.002670584 | 0.001224267 | 0.000082590 | 0.008704186 |
| precap_sensitivity | full128 | 62.0228% | 0.007090658 | 0.005794659 | 0.001577569 | 0.016338727 |
| precap_sensitivity | shorter_only | 0.8938% | 0.000485752 | 0.000041824 | 0.000000000 | 0.000688720 |
| precap_sensitivity | no_samples | 37.0833% | 0.005365991 | 0.004414894 | 0.000233445 | 0.014644541 |

For native clean full128, raw exists but no modal content on13.3468% original weight, accounting for ALL .008956322815 of that category's floor attribution. Farther-stable profiles (margin .162320..219066) alone account for .007456763442 of the total floor while full raw windows remain available. This prioritizes a representation QUESTION; it does not show these weak/decayed windows support margin estimation.

## Alignment sensitivity and profiles
| Existing windows | History group | Original weight | Full-class contribution | Restricted modal bound | Mask/time-aware bound |
| --- | --- | --- | --- | --- | --- |
| three_slots | without_window | 23.9718% | 0.003986078 | 0.001405817 | 0.000082590 |
| three_slots | window_but_no_modal | 13.3199% | 0.008956323 | 0.007386851 | 0.002136469 |
| five_windows | without_window | 23.9449% | 0.003986078 | 0.001405817 | 0.000082590 |
| five_windows | window_but_no_modal | 13.3468% | 0.008956323 | 0.007386851 | 0.002136469 |

| Native raw category | Profile | Original weight | Margin range | Modal floor contribution |
| --- | --- | --- | --- | --- |
| full128 | cross | 14.563% | -0.005000..0.097613 | 0.000269648 |
| full128 | farther_stable | 20.000% | 0.162320..0.219066 | 0.007456763 |
| full128 | retreat | 20.000% | 0.001505..0.094980 | 0.000532593 |
| full128 | stable | 20.000% | 0.004404..0.099586 | 0.000697318 |
| full128 | unstable_exclusion | 1.492% | -0.029420..-0.016494 | 0.000000000 |
| shorter_only | cross | 1.183% | -0.005000..-0.005000 | 0.000846818 |
| shorter_only | unstable_exclusion | 0.484% | -0.029420..-0.016494 | 0.000468676 |
| no_samples | cross | 4.254% | -0.005000..-0.005000 | 0.000784366 |
| no_samples | unstable_exclusion | 18.024% | -0.029913..-0.015143 | 0.001886219 |

## Primary no-sample bounds and unchanged comparator error
| Noise | Stratum | Target | Original-modal subset bound | +mask/time subset bound | Old GRU error contribution |
| --- | --- | --- | --- | --- | --- |
| clean | all | 0.010 | 0.001224267 | 0.000082590 | 0.008704186 |
| clean | boundary | 0.005 | 0.000996109 | 0.000001596 | 0.010221065 |
| iid1 | all | 0.015 | 0.001224208 | 0.000082590 | 0.008708858 |
| iid1 | boundary | 0.010 | 0.000996428 | 0.000001596 | 0.010232016 |
| correlated1 | all | 0.015 | 0.001224239 | 0.000082590 | 0.008708993 |
| correlated1 | boundary | 0.010 | 0.000998025 | 0.000001596 | 0.010235864 |

No native no-sample restricted-modal bound exceeds a target, in average or individual folds. Pre-cap sensitivity also stays below every primary target. Thus this audit does NOT establish acquisition impossibility. IMPORTANT: simply correcting raw-present decisions and freezing old no-sample predictions leaves native boundary GRU error .0102211(clean)/.0102320(IID1)/.0102359(correlated1), above .005/.010/.010 targets. A raw-only replacement with identical old gap fallback is insufficient by that leverage calculation; refitted mask/time fallback could differ, but no ability to learn it is proven. All4 old comparators/3 seeds and per-profile/per-fold margins/errors are in partition.json.

## Equality contract and uncertainty
Original-modal inputs: fold-specific train-accepted standardized174 coordinates, actual float32,3 lags4/2/0, unchanged boolean qualification and float32 age, original all-zero warmup padding. Auxiliary eligible rows admitted to content only; qualification remainsfalse. No design/profile labels, true margin, simulator poles/state, or approximate equality.

Mask/time-aware conditional bounds append presence,command,finite,calibrated-cap,admissibility masks (384bits each) and exclusive current sample end. All masks and end refer only to observed past; future cap index is never inserted. First-cap sensitivity applied only to past sample eligibility. Leading measured51 calibration offsets/scales remain preserved in mask artifacts but are NOT additional coordinates here. Including those constants or other permitted metadata could split equality classes further. These are EMPIRICAL conditional bounds, not all-causal-predictor information limits. Zero/low exact floor is not transferable estimability, especially with noisy unique features or fine masks.

QuantileGRUModel is source candidate ONLY: existing forward takes(B,channels,T), resets hidden state per forward, outputs ordered(q05,q50,q95). It does not yet implement this16-sensor383ms mask/exclusion/age/timebase experiment or a validated gap fallback. No pretrained model was adapted, scored or fitted.

## Verification
Actual source audit verified EVERY recovered128-window exclusion against saved reasons. Boundary/warmup/cap smoke and unequal weighted-class/mask-distinction smoke passed. Independent dictionary-by-bytes verification:504 primary subset-floor checks;252 category count/weight checks;386880 explicit sample-mask/timebase/run-length checks. Maximum floor discrepancy 4.33680869e-18. New categorical weights sum to original1, class contributions sum to old floor and seeded old errors sum to prior comparator MAEs for all156 reports. Ledger unchanged:ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f. No fits or qualified availability/accuracy/uncertainty/warning/runtime/physical-transfer credit.

## Decision requested
Assess whether evidence warrants ONE bounded raw-versus-modal representation feasibility experiment, not another modal feature correction or acquisition. Its masks/gap fallback and uncertainty must be prospectively specified; correcting only raw-present rows with old gap predictions is insufficient at boundary. Alternatively state the specific missing prerequisite or stop. User delegates scientific choice to frontier. Do not infer usable information from existence of weak raw samples or absence of exact collisions. No fits until explicit authorization.

Evidence: p4_groups/operational_confirmation_20261003/gate_bypass_receipt_20261010/raw_response_partition/{masks_<20design>.npz/.json,partition.json,summary.json,verification.json}. Executables:audit_raw_response_masks.py;partition_raw_response_floor.py;verify_raw_response_partition.py. Explicit authorization:p4_journal/review_20261009/raw_response_partition_authorization.json.
