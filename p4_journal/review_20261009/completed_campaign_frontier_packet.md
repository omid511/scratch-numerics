# Completed frontier experiment campaign — evidence for reassessment

## Assignment
All three previously proposed experiments have been executed. Review actual results against the frozen scientific contract. Return only consequential findings: what mechanism is supported, what remains unresolved, whether any arm earns adoption/accuracy progress, and the most discriminating next controlled step if justified. Address the user's concern that experiment1 was a weak/redundant check. No feature/model sweep, no invented endpoint pass, no information-contract relaxation without causal evidence. GRU remains a live research route; reject this implementation if criteria fail, not all recurrence. Read-only review of the supplied packet/artifacts, no independent gathering or model runs.

## Scope and aggregation
Experiment1: four fitting, two original checkpoint-validation, two repeatedly inspected development-held designs; seeds42/43/44; unchanged128-sample causal windows,1kHz,64ms stride,174 content+observed/age inputs,32→16 head. Four arms cross full174/first107 content and all-frame/current-only normalized loss. Common proper-training scaling. Every gap output is the corresponding frozen baseline output, including checkpoint validation. No availability claim.

Experiments2/3: original4 fitting designs plus12 preregistered eligible global-TRAIN designs. Three evaluation folds of4; expanded fitting uses original4+the other8. Fixed original2 checkpoint-validation designs; old2 held designs secondary only. Every copy of evaluation/validation/held designs excluded from proper fitting. Clean/IID1 training only, same optimizer,normalized row-loss regularization and selection. Two realizations per cross/stable/retreat/farther-stable/unstable-exclusion profile; all frames retained. Primary tables below aggregate errors of each seed separately across12 distinct evaluation designs, never averaged predictions.21 fold/noise rows or252 design/seed/noise comparisons are descriptive repeated comparisons, not independent sample sizes. Fresh paired noise draws are not fresh structures.5% stress is separate and included in the machine-readable evidence.

Experiment2 crosses original4/expanded12 fitting with measured174/matched-physical174 descriptions. Physical replacement changes matched branches' growth/frequency/calibrated projected patterns and dependent coordinates; preserves measured energy-order slots, strength fractions, availability, envelope summaries and residuals. No true-critical-branch insertion/selection. Instantaneous physical shapes are not identical to effective finite-window fitted shapes: physical-description counterfactual, not pure numerical-estimation attribution or full oracle.

Experiment3: content-GRU32→16 reset at every decision; three176-entry slots at lags4/2/0.383ms oldest waveform sample; no cross-trajectory or future borrowing, no persistent hidden state. Controls are refitted ordered-concat32→16, unordered-pool32→16 and flags/age-only GRU32→16, plus the frozen matched expanded-measured framewise reference. Native all-frame outputs including unqualified missing-current extrapolations. Same support/seeds/noise/optimization/selection; no width/loss search.

## Frozen targets
Equal-design all-frame MAE≤.010 clean/≤.015 each1% family; exact boundary(-.02<m<.05)≤.005/≤.010. Boundary optimism P(point-truth>.02)≤5%; available-point sign error≤10% on each .01≤|m|≤.05 side. Farther-stable(m≥.10) development noninferiority allowance.005. GRU must beat strongest same-span controls on distinct-regime,current-boundary and original-denominator all/boundary without sign,optimism or retreat regression. Qualified availability,independent-design95% run-risk calibration,delivered warning lead/nuisance burden,full-path p95≤10ms,fresh blind designs and independent physical/model-form validation remain required and unpassed. No endpoint or production cutover from this campaign.

## Experiment1 — narrow elimination check
12 fits. All three interventions worsen mean all-frame,boundary and current-boundary error in all7 primary held noise views; native gaps exactly preserved. Current-boundary design/seed wins are11/42(current-only),18/42(first107/all),1/42(combined); none earns adoption. Missing farther-stable/sign-exclusion populations stay unknown. This rejects these specific cheap corrections under this protocol, not the broader mechanisms or an information bottleneck.

## Experiment2 — primary grouped-evaluation numbers
MAE; each row scores the same12 evaluation designs. O=original4 fitting,E=expanded12 fitting; M=measured,P=matched physical counterfactual.

| Noise | Fit/content | All | Boundary | Current | Current boundary | Gap |
|---|---|---|---|---|---|---|
| clean | original_measured | 0.037109 | 0.032981 | 0.021597 | 0.019842 | 0.052345 |
| clean | original_oracle | 0.040483 | 0.036246 | 0.027638 | 0.025669 | 0.052629 |
| clean | expanded_measured | 0.036314 | 0.031075 | 0.019504 | 0.016320 | 0.053336 |
| clean | expanded_oracle | 0.035868 | 0.030681 | 0.018503 | 0.015304 | 0.053446 |
| iid1 | original_measured | 0.051860 | 0.040235 | 0.021983 | 0.017726 | 0.055926 |
| iid1 | original_oracle | 0.052638 | 0.040943 | 0.028763 | 0.025392 | 0.055810 |
| iid1 | expanded_measured | 0.051450 | 0.039376 | 0.017481 | 0.013709 | 0.056236 |
| iid1 | expanded_oracle | 0.051457 | 0.039608 | 0.017604 | 0.014294 | 0.056202 |
| correlated1 | original_measured | 0.053510 | 0.040618 | 0.023876 | 0.019198 | 0.057203 |
| correlated1 | original_oracle | 0.053960 | 0.040895 | 0.028504 | 0.025003 | 0.057068 |
| correlated1 | expanded_measured | 0.053095 | 0.040046 | 0.019975 | 0.016657 | 0.057419 |
| correlated1 | expanded_oracle | 0.053037 | 0.040166 | 0.019854 | 0.016907 | 0.057387 |
| iid1_fresh20261020 | original_measured | 0.051969 | 0.040260 | 0.022164 | 0.017654 | 0.056014 |
| iid1_fresh20261020 | original_oracle | 0.052762 | 0.040969 | 0.028777 | 0.025681 | 0.055934 |
| iid1_fresh20261020 | expanded_measured | 0.051484 | 0.039450 | 0.017769 | 0.014258 | 0.056248 |
| iid1_fresh20261020 | expanded_oracle | 0.051546 | 0.039719 | 0.018029 | 0.014985 | 0.056243 |
| correlated1_fresh20261020 | original_measured | 0.054212 | 0.040443 | 0.023477 | 0.019196 | 0.058098 |
| correlated1_fresh20261020 | original_oracle | 0.054768 | 0.040837 | 0.029202 | 0.026234 | 0.057954 |
| correlated1_fresh20261020 | expanded_measured | 0.053830 | 0.039932 | 0.019822 | 0.016724 | 0.058290 |
| correlated1_fresh20261020 | expanded_oracle | 0.053812 | 0.040088 | 0.019982 | 0.017207 | 0.058261 |
| iid1_fresh20261021 | original_measured | 0.052324 | 0.039836 | 0.022682 | 0.017582 | 0.056398 |
| iid1_fresh20261021 | original_oracle | 0.053087 | 0.040674 | 0.029000 | 0.025551 | 0.056283 |
| iid1_fresh20261021 | expanded_measured | 0.051973 | 0.039148 | 0.018056 | 0.013799 | 0.056735 |
| iid1_fresh20261021 | expanded_oracle | 0.052022 | 0.039468 | 0.018507 | 0.014852 | 0.056697 |
| correlated1_fresh20261021 | original_measured | 0.053791 | 0.040997 | 0.024617 | 0.020449 | 0.057380 |
| correlated1_fresh20261021 | original_oracle | 0.054306 | 0.041317 | 0.029717 | 0.026866 | 0.057252 |
| correlated1_fresh20261021 | expanded_measured | 0.053337 | 0.040292 | 0.020632 | 0.017328 | 0.057549 |
| correlated1_fresh20261021 | expanded_oracle | 0.053279 | 0.040406 | 0.020254 | 0.017499 | 0.057536 |

Measured expansion improves12-design aggregate all/boundary means across7 primary views but only14/21 fold-noise rows in each; current-boundary20/21. Individual design/seed wins142/252(all),125/252(boundary),168/252(current-boundary); current-boundary increases reach.021451. No consistent improvement across every design/seed. Matched-physical versus measured on original4 improves0/21 means for all/boundary/current-boundary; after expansion improves9/21,4/21,7/21 respectively. The support×content interaction is negative in21/21 all and current-boundary mean rows, but individual interactions have both signs. Negative interaction is not a uniformly beneficial oracle. All/boundary absolute targets remain failed.

Original3-profile versus new2-profile effects expose a population cost, not only the convenient old population:

| Noise | Arm | Original3 MAE | New2 MAE | Retreat MAE |
|---|---|---|---|---|
| clean | original_measured | 0.026115 | 0.053600 | 0.022006 |
| clean | expanded_measured | 0.024212 | 0.054467 | 0.020120 |
| iid1 | original_measured | 0.032185 | 0.081371 | 0.030103 |
| iid1 | expanded_measured | 0.031064 | 0.082028 | 0.028970 |
| correlated1 | original_measured | 0.033113 | 0.084105 | 0.032367 |
| correlated1 | expanded_measured | 0.032105 | 0.084580 | 0.031412 |

Complete replacement accounting:90,902/236,712 slots replaced(38.402%);145,810 retained measured. Reasons:100,538 no frequency correspondence;38,858 spatial ambiguity;6,414 nonunique assignment. Frozen aggregate unresolved_slots=0 did NOT mean full replacement. Read correspondence_accounting.json. Independent clean spot checks cover294 counterfactual windows and417 accepted branch matches, not uniqueness/completeness for every unreplaced slot. Actual protocol frequency acceptance7.874015748031496Hz, overlap≥.8, margin over second overlap≥.1 and eigenbasis conditioning≤1e10; near-degeneracies remain unresolved. Prose previously misstated the frequency and conditioning limits; pinned acquisition/verification used these actual values, with no score or replacement-count change. Frontier received this correction.

Physical-domain amendment: original attempted secondary D000053 farther-stable profile reaches Mach1.807<2. The failed attempt is preserved in support_modal_20261009/. V2 explicitly marks that one secondary profile inapplicable, retains its existing scores and all12 primary designs; no physical gate relaxed. Farther-stable and unstable-exclusion observations were added to BOTH original/expanded fitting populations, not counted as support-only treatment. Support also redistributes fitting contributions; attribute gains to this expanded fitting population, not pure structural diversity. All-frame row-loss weights remain normalized:original9,920 versus expanded29,760 fitting frames; detailed design/noise contribution metadata is in training.json/per-fit artifacts.

## Experiment3 — temporal order versus same-span pooling
MAE; F=frozen expanded measured reference,G=contentGRU,C=orderedconcat,P=unorderedpool,A=flags/ageGRU. All12 primary designs,all5 profiles.

| Noise | Arm | All | Boundary | Current boundary | Distinct regimes |
|---|---|---|---|---|---|
| clean | framewise_reference | 0.036314 | 0.031075 | 0.016320 | 0.026404 |
| clean | content_gru | 0.031672 | 0.026250 | 0.016396 | 0.023702 |
| clean | ordered_concat | 0.031000 | 0.026038 | 0.016870 | 0.024328 |
| clean | unordered_pool | 0.031014 | 0.026161 | 0.016783 | 0.024108 |
| clean | flags_age_gru | 0.050808 | 0.042873 | 0.040691 | 0.035954 |
| iid1 | framewise_reference | 0.051450 | 0.039376 | 0.013709 | 0.037261 |
| iid1 | content_gru | 0.048307 | 0.034644 | 0.011573 | 0.035794 |
| iid1 | ordered_concat | 0.048262 | 0.034753 | 0.013103 | 0.036173 |
| iid1 | unordered_pool | 0.048083 | 0.034540 | 0.012677 | 0.035635 |
| iid1 | flags_age_gru | 0.054757 | 0.043830 | 0.054131 | 0.038003 |
| correlated1 | framewise_reference | 0.053095 | 0.040046 | 0.016657 | 0.038759 |
| correlated1 | content_gru | 0.050112 | 0.035285 | 0.012712 | 0.037401 |
| correlated1 | ordered_concat | 0.050334 | 0.035792 | 0.014838 | 0.038435 |
| correlated1 | unordered_pool | 0.050228 | 0.035681 | 0.014553 | 0.037693 |
| correlated1 | flags_age_gru | 0.055851 | 0.043595 | 0.054472 | 0.038386 |
| iid1_fresh20261020 | framewise_reference | 0.051484 | 0.039450 | 0.014258 | 0.037284 |
| iid1_fresh20261020 | content_gru | 0.048280 | 0.034814 | 0.012020 | 0.035712 |
| iid1_fresh20261020 | ordered_concat | 0.048284 | 0.035093 | 0.013811 | 0.036360 |
| iid1_fresh20261020 | unordered_pool | 0.048141 | 0.034901 | 0.013464 | 0.035640 |
| iid1_fresh20261020 | flags_age_gru | 0.054909 | 0.043847 | 0.054019 | 0.038037 |
| correlated1_fresh20261020 | framewise_reference | 0.053830 | 0.039932 | 0.016724 | 0.039896 |
| correlated1_fresh20261020 | content_gru | 0.050950 | 0.035352 | 0.012719 | 0.038829 |
| correlated1_fresh20261020 | ordered_concat | 0.051241 | 0.036022 | 0.015745 | 0.040077 |
| correlated1_fresh20261020 | unordered_pool | 0.051035 | 0.035837 | 0.015027 | 0.039412 |
| correlated1_fresh20261020 | flags_age_gru | 0.056516 | 0.043523 | 0.054447 | 0.039389 |
| iid1_fresh20261021 | framewise_reference | 0.051973 | 0.039148 | 0.013799 | 0.037221 |
| iid1_fresh20261021 | content_gru | 0.048938 | 0.034352 | 0.011335 | 0.035618 |
| iid1_fresh20261021 | ordered_concat | 0.048905 | 0.034503 | 0.013094 | 0.036011 |
| iid1_fresh20261021 | unordered_pool | 0.048744 | 0.034294 | 0.012445 | 0.035417 |
| iid1_fresh20261021 | flags_age_gru | 0.055175 | 0.043601 | 0.054224 | 0.038348 |
| correlated1_fresh20261021 | framewise_reference | 0.053337 | 0.040292 | 0.017328 | 0.038431 |
| correlated1_fresh20261021 | content_gru | 0.050299 | 0.035350 | 0.012694 | 0.036706 |
| correlated1_fresh20261021 | ordered_concat | 0.050705 | 0.036346 | 0.015947 | 0.037974 |
| correlated1_fresh20261021 | unordered_pool | 0.050559 | 0.036171 | 0.015366 | 0.037060 |
| correlated1_fresh20261021 | flags_age_gru | 0.056115 | 0.043790 | 0.054775 | 0.038005 |

GRU beats framewise-reference mean all/distinct-regime errors in21/21 fold/noise rows; current-boundary20/21. It beats flags/age-only mean all/current-boundary in21/21. Against ordered-concat, GRU mean wins13/21(all),15/21(distinct),14/21(current-boundary); against unordered pooling7/21,12/21,17/21. Corresponding individual design/seed wins versus pooling106/252(all),105/252(distinct),131/252(current-boundary).12-design aggregate wins against both same-span controls:3/7 all,4/7 boundary,7/7 current-boundary,4/7 distinct. This differs from the earlier4-design history experiment: changing-regime gains over current-only now exist, but uniform ordering-specific benefit is not established.

## Risk and observation limits
Rates below average seed-wise event counts over the exact indicated population; no ensemble predictions or safety confidence bound. Sign rates use current/accepted points on each eligible stable/unstable side. Boundary optimism uses the original boundary denominator including gaps.

| Noise | Arm | Boundary optimism | Current stable sign error | Current unstable sign error |
|---|---|---|---|---|
| clean | framewise_reference | 48.64% | 1.68% | 7.05% |
| clean | content_gru | 32.07% | 9.31% | 5.42% |
| clean | ordered_concat | 33.98% | 3.97% | 4.88% |
| clean | unordered_pool | 33.72% | 6.54% | 7.59% |
| iid1 | framewise_reference | 67.14% | 3.15% | 6.50% |
| iid1 | content_gru | 57.16% | 5.95% | 5.96% |
| iid1 | ordered_concat | 56.85% | 4.85% | 5.15% |
| iid1 | unordered_pool | 56.03% | 2.63% | 7.05% |
| correlated1 | framewise_reference | 66.63% | 1.59% | 6.50% |
| correlated1 | content_gru | 56.62% | 4.95% | 4.88% |
| correlated1 | ordered_concat | 57.17% | 2.73% | 5.96% |
| correlated1 | unordered_pool | 56.29% | 2.67% | 7.59% |

GRU increases current stable-sign error versus framewise in19/21 fold/noise rows; versus pooling15/21. Boundary optimism is worse than pooling17/21. Retreat MAE worsens versus pooling14/21; worst increase.002302. Farther-stable(m≥.10) MAE worsens versus pooling19/21; one increase.006108 exceeds the.005 allowance. Those are frozen acceptance failures, not fixed by lower pooled MAE. Exact denominators and all primary/fresh/stress rates are in completed_campaign_analysis.json and original results.json.

Common measured gate accepts7,306/14,880 clean frames(49.10%),1,805/14,880 IID1(12.13%),1,695/14,880 correlated1(11.39%). These are observation acceptance, NOT qualified availability. Original quiet/extrapolated frames remain in all MAE and optimism denominators. Low accepted coverage does not itself prove waveform nonidentifiability, and conditional-current accuracy cannot replace all-frame performance. Added safety slices were observed/scored, not certified. Secondary D000053's inapplicable farther-stable profile cannot pass a guardrail. Uncertainty/clearance,lead/nuisance and full-path timing were not rerun or earned.

## Completed verification and caught defects
| Experiment | Fits | Independent metric rows | Independent network replay | Maximum prediction difference |
|---|---|---|---|---|
| Cached factorial | 12 | 2576 | 624,960 decisions | 2.384e-07 |
| Support/content | 24 | 7176 | 6,231,744 frames | 4.664e-07 |
| Bounded GRU | 36 | 10530 | 3,456 sampled decisions | 1.281e-06 |

Factorial1 also verified92 transition rows and training-prefix/baseline reproduction. Support verified546 factorial contrast rows,24 fit artifacts,exact folds,denominators,proper-scaling/contribution metadata; independent acquisition/correspondence checks preceded fitting. GRU independent NumPy slot/GRU gate replay sampled96 decisions per model, not every neural output; all reported metric/paired rows were recomputed, plus causal future/ordering/reset/warmup smoke. Copied result hashes and45 prediction/reference artifact hashes plus36 GRU checkpoints match receipts. Progress ledger SHA remains ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f.

Caught defects preserved explicitly:
- Original support diagnostics exported complex mode-shapes as float64, losing imaginary coefficients. Recovered from frozen poles/windows using actual least-squares basis; probe showed imaginary amplitude.5884 and reconstruction error1.59e-15. Recovery changed diagnostic storage only, NOT feature rows,predictions,selection or gates.
- Original support scorer indexed a whole stratum map rather than selected stratum; all24 fits completed before KeyError. Corrected scorer recomputed contrasts from unchanged predictions.
- Support verifier compared float32 contribution weights against an incorrectly float64 expectation. Corrected to exact comparison in serialized dtype, not tolerance widening. Successful verification is the corrected checker.
- verification.json score_source_sha256 names archived original support scorer, not the executed corrected scorer. results.json scoring_source_sha256 and scoring_correction.json identify the actual corrected scorer; checker explicitly validates it. Preserve the erratum, do not claim the old scorer produced the corrected result.
- First GRU freeze invocation lacked remote experiment1 receipt; verified local result/receipt transferred before successful freeze/training. After36 fits, GRU checker lacked distinct_regimes_current and used >.005 rather than frozen>=.005. Corrected independent masks; all metrics then passed without changing scores/predictions/tolerances. Exact.005/.004 boundary smoke passed.
- Two requested evidence workers failed with service429 errors before publishing reports. Dispatcher constructed this packet and checked the actual machine evidence; do not imply independent Butler audit completion.

## Supplied artifact map
All under p4_groups/operational_confirmation_20261003/:
- current_context_20261009/{protocol,training,results,verification,decision,execution_erratum}.json; experiment_current_context.py,verify_current_context.py.
- support_modal_20261009_v2/{protocol,training,results,verification,acquisition_verification,correspondence_verification,correspondence_accounting,observation_coverage,diagnostic_recovery,scoring_correction}.json; dataset.npz,predictions.npz,recovered_modal_shapes.npz; experiment_support_modal_v2.py; actual score_support_modal_corrected.py,verify_support_modal_corrected.py plus pinned historical originals.
- bounded_gru_20261009_v2/{protocol,results,verification,verification_correction}.json;36 per-fit NPZs/PTs; experiment_bounded_gru_v2.py,verify_bounded_gru_v2.py. No single aggregate predictions.npz; nine framewise reference NPZs are reused from support.
- Full primary/fresh/stress aggregate metrics,exact denominators,event counts,per-contrast heterogeneity and transferred artifact SHA values: p4_journal/review_20261009/completed_campaign_analysis.json. Original results preserve per-design/seed and all secondary-held metrics; those old2 designs are not additional independent groups.
- Prior full historical feature report and constraint survey: p4_journal/review_20261009/{working_feature_history,constraint_features_and_models,frontier_evidence_packet}.md and frontier_review.json. Feature roster/information contract unchanged.

## Decision question
Reassess the three experiments against your original predictions. Which scientific bottleneck is now evidenced versus merely plausible? Does modest expanded-support benefit plus a partial physical-description null distinguish inadequate conversion/support from extraction/observability? Does GRU improve useful bounded response evolution beyond flags/age while failing recurrence-specific adoption? Give one ranked, falsifiable next mechanism test only if supported, with controls and outcome-dependent decisions; explain why it has greater diagnostic leverage than another marginal feature/loss/model exercise. State if independent physical-validation prerequisite or observed acquisition envelope should now govern the campaign instead. No promotion, endpoint credit or information-contract change is assumed by the dispatcher.
