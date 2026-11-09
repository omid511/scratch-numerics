# Frontier experiment campaign — execution contract

Requested: execute all three experiments in frontier_review.json and return completed evidence to frontier. Original completed experiments, predictor, reserved datasets and P4_PROGRESS.json are read-only. Diagnostic outputs never count as qualification.

## Shared protocol

Existing script directory: p4_groups/operational_confirmation_20261003/. Existing CUDA host: omid5@cachyos.local; key /home/omid5/.ssh/p4_gpu_ed25519; remote repo /home/omid5/p4_o1_execution_20261003, Python /home/omid5/mechanics/.venv/bin/python, PYTHONPATH remote repo. Local .venv/bin/python. Prefix shell commands with rtk proxy. No builds/lint/tests/formatters during implementation; executor runs real smoke and experiments afterward. Pin executable inputs/helpers/implementation before first fit. Use outputs in new *_20261009 directories, never overwrite verified artifacts. Save per-seed predictions, checkpoint/scaling/selection history, row contributions, SHA256 and independent replay/metric verification.

Common model: fixed 32→16 ReLU MLP or GRU32→16 as specified; seeds42/43/44; Adam lr=.003 weight_decay=.0001; max2000 epochs/patience100; full-batch row-mean squared loss. Original clean/IID1 training and checkpoint validation only; fresh views never select. Proper-train-only scaling; identical head initialization and row order within paired arms; no loss/architecture/penalty search. Every original frame remains in scoring. Equal-design MAE averages seed-specific errors, not predictions. Exact boundary: -.02<m<.05. Shared targets: all-frame MAE≤.010 clean/≤.015 each1% family; boundary≤.005 clean/≤.010 each1% family; boundary optimism>.02≤5%; each .01≤|m|≤.05 wrong-sign rate≤10%; farther-stable degradation≤.005. Individual design/seed comparisons, primary clean/IID1/correlated1 and fresh paired1% views,5% stress, profiles, transitions and current/gap populations. Unknown populations stay unknown.

## Experiment1 interface and ownership

Parent owns experiment_current_context.py and verify_current_context.py, current_context_20261009/{protocol,training,predictions,results,verification}.json/npz/pt after the Butler stopped without files. Run exact cached four-arm factorial from additive_spectral_20261009/dataset.npz: full174 or zero last67 normalized columns, crossed all-frame/current-only MSE. Common174 scaling and176 input width; age/observed retained. Use corresponding archived baseline174_seed42/43/44 outputs for all gap decisions during checkpoint validation AND scoring. Independently replay the archived baseline, and verify the full/all training prefix at its original selected epoch within2e-6. Its newly selected epoch need not reproduce the archived checkpoint because the required composite validation criterion changes selection. Save both native and composite predictions; gap outputs must match archived exactly. Report factorial contrasts and interaction per design/seed, original-denominator metrics, accepted/gap transition jumps and guardrails. No current-only model is an availability method.

## Experiment2 interface and ownership

Parent owns the reviewed acquisition and execution after worker handoff: experiment_support_modal_v2.py and support_modal_20261009_v2/* artifacts. The failed initial support_modal_20261009 attempt is retained. Acquire original8 designs and12 additional eligible TRAIN designs, designated deterministically from eligible original TRAIN IDs using seed20261009 BEFORE looking at outcomes. Exclude every validation/held/reserved copy. Additional12 partition into3 grouped evaluation folds of4; expanded fitting = original4 plus additional8 excluding evaluated fold. Original4 versus expanded fitting must score identical fold designs with unchanged original2 checkpoint-validation designs. Old2 held designs secondary. Same original cross/stable/retreat generation and2 replicates,128 samples/1kHz/stride64; add farther_stable (ratio .78–.84) and unstable_exclusion (ratio1.015–1.03) profiles with the same pulse law. A pre-fit physical-domain amendment marks D000053 farther_stable inapplicable: ratio .78 gives Mach1.8074338014281053, below preserved Mach2. Its original profiles and unstable_exclusion remain; that missing secondary regime is unknown, not a pass. All original4 fitting, original2 checkpoint-validation and12 primary evaluation designs retain all5 profiles. Preserve separate original-profile scoring and original fit-population baseline comparison. Clean/IID1 train; all13 frozen noise views evaluate, with clean/1% primary and5% stress. Validate added operating domain, bandwidth, reduced/full physical agreement, ordinary-mixture correspondence, eigenbasis conditioning and causal calibration before using oracle174.

Dataset interface: support_modal_20261009_v2/dataset.npz fields features(float64 R×F×174 measured), oracle_features(same shape), truth(R×F), observed(bool R×F), age(R×F), times(F), records(JSON scalar), replacement diagnostics and protocol SHA. Each record has split(train/validation/held), design, profile, replicate, noise; additional IDs carry split=train and designated evaluation-fold in protocol, not held. No oracle labels/truth in measured features. Oracle replaces only confidently uniquely matched measured modal slots' growth/frequency and sensor complex patterns; recompute dependent modal/physics/spatial features, preserving measured strength fractions, residual/quality/availability and envelope statistics. No replacement by true critical branch. Unresolved slots keep measured entries; all denominators remain. Assignment uses calibrated projected shape plus frequency compatibility; save ambiguity, replacement coverage, matching criteria and conditioning. This is a physical-description diagnostic, not pure numerical-estimation attribution or an upper bound. Row-mean loss maintains normalized regularization; disclose train rows by design/noise/profile. Fit all-frame native outputs with the fixed nonlinear174/176 head. Compare all four combinations, folds and seeds; save models/predictions/scaling. The frozen export accidentally cast measured modal-shape diagnostics to float after features were computed. Mandatory recovered_modal_shapes.npz restores discarded sine coefficients from the measured windows and frozen poles without changing the dataset, feature rows or original real coefficients. Use verify_support_acquisition_recovered.py and verify_support_correspondence.py for independent acquisition checks; retain source-hash-linked diagnostic_recovery.json.

## Experiment3 interface and ownership

Parent owns experiment_bounded_gru_v2.py and verify_bounded_gru_v2.py, bounded_gru_20261009_v2/*. Depends on established experiment1/2 evidence, uses experiment2 measured dataset and its proper/train fold exclusions/checkpoint validation. Three176-entry slots at frame lags4/2/0 (64ms stride;383ms oldest sample); zero missing warmup slots; reset GRU every decision. Native every-frame outputs for all arms. Content GRU32→16, refitted ordered concat MLP32→16, unordered pooling same-width MLP32→16, flags/age-only GRU32→16. Same support, seeds, noise views, normalized loss and selection. No8s recurrent state. Include the already-fitted fold-specific measured framewise reference. Compare strongest same-span comparator on distinct-regime,current-boundary,overall/boundary, retreat, optimism/sign and farther-stable. Dwell-only gains insufficient; gaps are unqualified extrapolation. Full-path latency and broader endpoint validation not passed by these fits.

## Delivery

Workers publish implementation and precise commands first; no training until parent scientific review for support/oracle. Parent runs GPU executions sequentially on shared host; workers MUST NOT launch overlapping GPU workloads. Workers may perform small local import/mechanism smoke. No shared-file ownership overlap. Final evidence includes every result, caught defect, missing prerequisite, no promotion unless full predeclared criteria are met. Frontier receives compact complete actual evidence, precise diagnostic question and all remaining qualification blockers.

## Completed delivery

All three experiments completed:12 cached-factorial fits,24 support/content fits,
36 bounded-GRU/control fits. Independent receipts reside in
current_context_20261009/verification.json,
support_modal_20261009_v2/verification.json and
bounded_gru_20261009_v2/verification.json.
Actual executed support scorer/checker are score_support_modal_corrected.py and
verify_support_modal_corrected.py. Original pinned sources are historical evidence,
not the corrected executable; scoring_correction.json explains the source-label
erratum and exact stored-dtype contribution check.
GRU verification_correction.json records the independent mask repair; models,
scores, selection, policies and tolerances remain unchanged by checker repairs.

Completed evidence and actual frontier reassessment:
p4_journal/review_20261009/{completed_campaign_frontier_packet.md,
completed_campaign_analysis.json,completed_campaign_context_appendix.json,
completed_frontier_reassessment.json}.
The existing reviewer received actual packet/context text after requesting it,
not merely file paths. Additional evidence workers failed service429 before
publishing; no independent Butler audit completion is claimed.
No model promotion, qualification, reserved-design score or progress-ledger change.
The next gate-erasure ablation is a proposal only, not part of this completed run.
