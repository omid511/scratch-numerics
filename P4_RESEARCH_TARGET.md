# P4 research endpoint — proposal, 2026-10-08

## Decision

Aim for **transferable, uncertainty-qualified flutter headroom and timely advisory warning from short multichannel measurements on unseen plate designs, without a target-specific flutter-calibration campaign**.

This is a proposed endpoint, not an achieved result or authorization to start a new experimental campaign. It retains P4's margin and early-warning objectives. It does not require a GRU, abandon neural models, or introduce automatic re-excitation. Reaching the complete evidence gate below would support a strong submission to a relevant aeroelasticity/structural-dynamics journal; no numerical threshold guarantees Q1 acceptance. “Most Q1 journals” is not a defensible cross-discipline publication promise.

## What the work has established

| Evidence | Established result | Limit |
|---|---|---|
| Pipeline and evaluation audits | Corrected sampler-driven multi-speed gains, missing timebase comparisons, calibration overclaims and interrupted-training provenance | Infrastructure and artifact removal are necessary, not the scientific contribution |
| Spatial modal representation, 77-design TRAIN CV | Baseline42 → spatial107 mean design MAE: clean .016045 → .008587; fresh iid1 .017869 → .010131; fresh correlated1 .019424 → .011031 | Repeated exploratory TRAIN CV, not blind validation; tight-boundary decisions did not improve uniformly |
| Noise-aware fitting and representation controls | Spatial information improves over both the original representation and a width-matched shuffled control; fresh acquisition draws were evaluated | Noise settings are sensitivities, not measured hardware specifications |
| Propagated-trajectory benchmark and diagnosis | Continuous-state evaluation exists; identified onset fitting, disappearing physical response, noise-only poles and harmful threshold calibration | Correct execution did not deliver useful warning; initial-only quiet trajectories lose information before approach |
| Time-varying modal fit, 48 critical-only controls | End-pole recovery 19/48 → 40/48; correct growth signs 34/48 → 48/48 versus archived LS-ESPRIT | Eight quantitative fast-ramp failures remain; ordinary mixtures were correctly skipped |
| Cached modal-transport and carrier-curvature diagnostic | All 24 physical waveforms replay within 2.6e-12 relative RMS; curvature correction improves recovery from 40/48 to 45/48 while preserving 48/48 growth signs; spatial transport separates sensor-envelope growth from physical pole growth | Three quantitative failures remain; noisy-stationary median growth error .13894/s exceeds .05885/s allowance. Oracle mechanism controls, not deployable inputs; no ordinary-mixture or margin/warning qualification |
| Waveform-only noise-aware curvature selection, same 48 controls | Restores noisy-stationary median growth error to .02286/s, below .05885/s allowance; all growth signs remain correct | Recovery falls 45/48 → 40/48, so not promoted. Clean fast-ramp residuals are overwhelmingly spatially coherent model mismatch, not measurement noise; residual-as-noise selection discards useful curvature |
| Same-observation fitter/simple-map versus learned comparison | Framewise+hold has lower common-finite MAE in four of five noise conditions; fitter+hold has higher warning recall at the frozen calibration budgets | Two held designs only; gap policy matters; realized false-warning rates differ; neither method supplies reliable signed margins or timely warning |

The last comparison's decisive failures: fitter6 reports positive margins on all 274 truly unstable held frames in every noise condition; native framewise does so on about 77–80%. In every noisy family, only one of four fitter crossing alarms precedes flutter. These findings are not evidence that the learned approach is obsolete; they are evidence that MAE, warning ranking and signed headroom are different outcomes.

Sources: [journal](p4_journal/JOURNAL.md), [spatial experiment](p4_groups/spatial_modal_20260930/protocol.json), [latest comparison protocol](p4_groups/operational_confirmation_20261003/fitter_margin_comparison_20261008/protocol.json), [latest comparison verification](p4_groups/operational_confirmation_20261003/fitter_margin_comparison_20261008/verification.json), [physical mechanism verification](p4_groups/operational_confirmation_20261003/modal_transport_20261008/mechanism_analysis.json), [carrier correction and accuracy gates](p4_groups/operational_confirmation_20261003/modal_transport_20261008/curvature_analysis.json), and [research review](src/mechanics/p4_margin_estimation/RESEARCH_REVIEW.md), especially Sections 11–14 and 22–24. Historical results remain unchanged.

Noise-aware checkpoint: [protocol](p4_groups/operational_confirmation_20261003/noise_aware_carrier_20261008/protocol.json), [independent verification](p4_groups/operational_confirmation_20261003/noise_aware_carrier_20261008/verification.json), and [coherent-residual diagnosis](p4_groups/operational_confirmation_20261003/noise_aware_carrier_20261008/residual_diagnosis.json).

## Scientific claim to earn

> Joint temporal and spatial modal information carries transferable information about design-dependent flutter headroom beyond growth sign alone. A physically justified observation/inference method can exploit it on unseen structures, quantify when that inference is unreliable, and improve timely warning at a fixed observation and nuisance-alarm budget over the strongest same-information competitors.

This is a hypothesis. The local relation `alpha ≈ -kappa * margin` explains the issue: even accurate growth does not identify normalized speed headroom unless the design-dependent conversion is constrained. Spatial/multimode context is a candidate source of that constraint, not a proven sufficient statistic.

The contribution must include **both** a physical explanation and a comparative engineering benefit. A collection of standard fitting, neural-network and calibration components is insufficient. Demonstrate, through controlled interventions and held-design ablations, what information resolves the conversion, which errors arise from measurement versus hidden structural ambiguity, and where inference fails.

A small calibrated physical model is a legitimate candidate winner. If it wins, do not claim a neural advantage. The paper remains viable only if the physical observation/inference result and its independently validated performance tradeoff are themselves substantively new; otherwise the proposed contribution fails.

### Closest precedent and the remaining burden

- [Crowther–Cooper 2001](https://doi.org/10.1243/0954410011531736) already trains neural flutter predictors on structurally varied model data. The [full-text review](src/mechanics/p4_margin_estimation/RESEARCH_REVIEW.md) records its sequence of modal observations and speed inputs. Simulation-trained cross-design learning alone is not new.
- [Chajjed et al. 2022](https://arxiv.org/html/2210.15667v1) uses random structural models and Bayesian modal priors across subcritical speeds to predict flutter speed and uncertainty. Population priors and probabilistic flutter margins alone are not new.
- [Méndez 2026](https://arxiv.org/html/2609.25440v1) uses an independently established configuration-specific frequency–growth trajectory to reduce finite-record damping uncertainty. Short-record fitting and physical constraints alone are not new; its target is not P4's cross-design normalized speed margin.
- [Wang–Zhou 2026](https://doi.org/10.2514/1.C038983), according to its [publisher-deposited abstract](https://api.crossref.org/works/10.2514/1.C038983), already reports signal-based neural flutter-boundary prediction with wind-tunnel and flight-test validation. Its full observation/structural-transfer protocol remains unread; do not claim superiority or uniqueness from abstract headline errors.

P4 must earn the **joint** advantage of reduced target-specific information, useful signed headroom, calibrated uncertainty and operational benefit. No claim of “first” is currently supported.

## Frozen task and information contract

- Scope: the declared plate family, boundary conditions and aeroelastic regime. Unseen designs from the same population are not arbitrary new airframes. Freeze physical parameter ranges, sensors and operating/excitation conditions before confirmation.
- Target: `m = (ucrit - velocity) / ucrit`; positive is subcritical. Preserve stable and unstable sides and farther-stable guardrails.
- Output: point margin, uncertainty, and explicit current/reliable versus unavailable/stale status. A positive point is not a safe declaration. A conservative lower margin bound above the predeclared clearance level supports only a research advisory, not certified safety.
- Inputs: causal measured sensor response, measured timebase, sensor identities and available causal calibration. No simulator poles, true margin, critical speed, hidden design parameters, future response or target-specific modal trajectory. Excitation timing/load information is allowed only in a separately declared arm if actually available to the monitor.
- Observation budget: maximum .5 s causal aperture and the existing 16-sensor maximum. Freeze sampling, update stride, warm-up and any shorter/adaptive aperture using development designs only. Count physical seconds and mode cycles, not just sample count.
- Primary acquisition conditions: clean, iid1% and correlated1% (`rho=.8`); iid5% and correlated5% are stress conditions. Use one frozen training/calibration policy, not five condition-specific winners. Retain the physical calibration reference; report actual SNR and per-channel noise levels.
- Operational cases: ordinary multimode response, slow/fast approaches, sustained crossings, stable/near-boundary dwells, approach–retreat, pulse onsets and weak/intermittent excitation. Initial-only responses that decay into noise remain explicit observability negative controls, not a promise of reliable passive monitoring through silence.
- The primary warning claim is restricted to a prospectively declared excitation envelope. Every trajectory in that envelope, including resulting quiet gaps, stays in denominators. Define the envelope from physical acquisition conditions, not true critical-mode visibility or post-hoc successful fits. Outside it, report unavailability and misses separately; no universal early-warning claim.
- Give competing methods identical observations and gap/age policies. Unknown positive windows count as misses. Stale holds are not new measurements and must expire under a common, development-selected age rule. Preserve the versioned legacy clip benchmark as a separate track.

## Completion gate

These numbers are **proposed research acceptance thresholds**, not application-derived safety requirements, proven attainable limits or reported performance. Freeze them before a new campaign. All primary-condition gates must pass; pooled clean performance cannot hide a noisy failure.

| Gate | Required evidence |
|---|---|
| Signed headroom | Mean per-design MAE ≤.010 clean and ≤.015 in each 1% family. In `-.02 < m < .05`, ≤.005 clean and ≤.010 in each 1% family. Report accepted/current, stale and unavailable populations separately, with the availability gate below; no selection-only accuracy claim |
| Dangerous point errors | Within the near-boundary slice, `P(prediction - truth > .02) ≤5%` in each primary condition; on available points with `.01 ≤ abs(m) ≤ .05`, wrong-sign fraction ≤10%. Report each side separately and do not interpret the excluded arbitrarily-close-to-zero sign slice as solved |
| Uncertainty/clearance | A predeclared 95% lower margin bound, calibrated on independent designs. One-sided 95% statistical upper bound ≤5% on the probability that a test-design run issues any clearance while actually unstable. Declare the run distribution/horizon. No arbitrary-OOD or per-window conditional coverage guarantee |
| Non-vacuous usefulness | Current qualified estimates during ≥80% of the final 2 s before crossing in the primary excitation envelope. During farther-stable operation (`m≥.10`), a usable positive-headroom advisory on ≥80% of eligible decisions. Unknown/expired outputs count against usefulness; abstaining everywhere fails |
| Actual early warning | Delivered usable lead ≥1 s for ≥90% of crossings, with one-sided 95% bounds supporting ≤10% timely misses. Use existing O1's 1 s research horizon, not an assumed human/aircraft intervention time. Count warm-up, acquisition, computation and declared delivery delays; late alarms fail |
| Nuisance burden | ≤1 nuisance episode/operating hour, supported by an upper confidence bound on independent benign exposure; include stable dwells and approach–retreat. Freeze episode merge, clear/rearm and event-credit rules. A framewise 5% calibration budget is not this criterion |
| Comparative impact | At the same observation contract and nuisance target, ≥25% relative reduction in timely misses against the strongest same-information competitor on a prespecified equally weighted primary-condition score. Require a paired design-level confidence interval excluding no improvement; report conditionwise results and actual held nuisance burden. Baseline already meets the absolute gates? A merely tied frontier does not meet the claimed benefit |
| Stress behavior | At 5% noise, no confident-clearance failure inflation beyond the declared ≤5% design-run criterion; loss of coverage/lead must be explicit. No maintained-accuracy/early-warning claim unless those same gates also pass. Noise stress does not stand in for physical mismatch |
| Online feasibility | Full sensor-to-output computation p95 ≤10 ms on named hardware, matching the earlier provisional target. Report p99/max and deadline misses. Acquisition remains ≤.5 s; fit-only timing or a fast MLP is not full-pipeline timing |
| Independent credibility | Blind design evaluation plus one independently validated plate response/label source with distinct structural discretization and aerodynamic/model-form assumptions. Numerical refinement or a new propagator for the same equations is not independent physical validation. Experimental plate data are stronger if accessible; do not imply unrelated wing data validate this plate estimator |

Point errors, warning performance and reliability remain separate gates: better ranking cannot compensate for biased signed distance, and a conservative bound cannot repair the point estimator. Farther-stable pessimism must not worsen materially versus the matched strongest comparator; predeclare a .005 MAE non-inferiority allowance on development data.

### Statistical and physical prerequisites

Use disjoint complete-design groups for fitting, model/policy selection, uncertainty/alarm calibration and blind evaluation. Existing repeatedly inspected TRAIN/held designs remain development data. Preserve existing reserved designs; their 16-design test group alone cannot support a precise rare-event claim. Do not repeatedly inspect it while developing.

Plan at least 120 fresh blind same-family designs, each with crossing and benign runs in every primary condition, plus a prespecified withheld design-region/model-form track. Freeze sampling and account for unsupported designs and label-generation failures. Noise views, trajectories and overlapping windows are not independent structures. Choose benign exposure and calibration-design counts before data collection, using clustered rate/bound calculations and attainable interval resolution. A zero-event bootstrap interval is not a population-risk bound. Use simultaneous conditionwise inference or a declared multiplicity correction for the primary gates.

Independent validation is a prerequisite for the proposed strong-publication endpoint, not an already available P4 result. Before expensive development, identify the concrete compatible source/solver and validate its flutter labels, coupling, grid/time convergence and stable/unstable behavior. If that route cannot be obtained within the bounded campaign, do not represent same-generator simulation success as meeting the complete endpoint.

The 1 s horizon and 1 episode/h limit are research operating points inherited from O1, not established deployment requirements. If physically feasible warning and benign/retreat trajectories have identical available histories, derive the resulting risk/lead tradeoff before optimizing. Learning cannot distinguish identical causal histories. Failure of the declared envelope is a failed claim, not permission to relabel nuisance cases afterward.

## Progress tracking

User-requested tracking is recorded in [P4_PROGRESS.json](P4_PROGRESS.json). Recompute with `rtk proxy .venv/bin/python p4_groups/operational_confirmation_20261003/update_p4_progress.py`.

- **Headline overall progress:** continuous, weighted attainment across all ten criteria: `sum(weight_i × percentage_i) / sum(weight_i)`. Default weights are equal at **10% each**, stored per criterion in the ledger. This replaces the binary completed-gate fraction; qualification pass/fail remains a separate field and the acceptance thresholds above are unchanged.
- **Secondary estimator diagnostic:** `100 × passed frozen estimator checks / 6`. The six checks are interior fits, physical pole recovery, growth signs, clean stationary error, noisy stationary error, and ramp improvement against both original comparators. This is not overall project progress. Report actual recovery/errors alongside it; a higher diagnostic score cannot hide a regression or justify promotion.
- **Criterion percentages:** equal-weight explicit metric components within each criterion. For lower-is-better measurements, use `100 × min(1, target / observed)`; zero error receives 100%. For higher-is-better measurements, use `100 × clip(observed / target, 0, 1)`. Thus MAE .020 against a .010 target earns **50%**, and 72% timely crossings against a 90% target earns **80%** without fully satisfying either requirement.
- **Missing evidence:** keep the measurement and its attainment null; assign no *documented* credit to its fixed component weight. Do not redistribute that weight onto measured components. Report the unknown-weight percentage alongside the headline: this is a conservative documented-attainment score, not an estimate that missing performance is zero. The possible score upper bound fills only unknown components at 100%; it is not a statistical confidence interval.
- **Consistent reference:** scoring version2 uses the frozen framewise temporal predictor on the two already inspected held TRAIN designs, with clean/IID1/correlated1 primary conditions. MAE uses per-design all-frame and near-boundary errors; gap extrapolation is not silently removed. Wrong-sign measurements are separated by sign side; no eligible unstable points outside the exclusion slice means unknown, not a perfect score. Current comparisons do not supply qualified bounds, final episode/hour risk, or blind validation.
- **Evidence/reporting components:** stress credit includes the fraction of the two 5% conditions with disclosed input-availability and lead outcomes, separately from their still-unknown clearance-risk bounds. Online credit separates acquisition aperture, full-path p95, and p99/max/deadline reporting. Independent-credibility components use fresh blind design count against the planned120 and independently validated response/label source count against1. These evidence components are progress indicators, not performance guarantees.
- **Current documented score:** **20.76%**, with **55.00% of scoring weight unmeasured**. Criterion scores: signed accuracy51.17%, dangerous errors38.71%, uncertainty0% credit (unmeasured), qualified usefulness0% credit (unmeasured), timely warning0%, nuisance burden0% credit (unmeasured), comparative impact0% credit (unmeasured), stress evidence33.33%, online feasibility84.40%, independent credibility0%. Online credit is `(100% aperture + 53.21% latency + 100% timing-report completeness) / 3`; it is **not** a passed latency or replay qualification. The remaining unknown weight limits what this number establishes. Qualification remains0/10 and is not the progress headline.
- Freeze denominators; version changes explicitly. Percentages may decrease. They are not effort estimates, safety assurances, confidence levels or publication probabilities. Different candidates and incompatible evaluation populations must not be pooled to manufacture a passed gate.

| Estimator checkpoint | Passed checks | Stage progress | Physical pole recovery |
|---|---:|---:|---:|
| Linear carrier | 5/6 | 83.3% | 40/48 |
| Free curvature | 4/6 | 66.7% | 45/48 |
| Noise-aware selection | 5/6 | 83.3% | 40/48 |

The latest selector is **not promoted**: it restores a guardrail but discards the earlier physical improvement. Spatially coherent physical/model residual remains distinct from stochastic measurement noise, but the frontier review below requires isolating extraction from headroom conversion before prioritizing a transport correction. Do not lower the selection threshold, add unrestricted channel-amplitude drift, or infer physical pole confidence from residual variance alone. All original accuracy/guardrail checks and the ordinary-mixture qualification gate remain unchanged.

### Full-path timing checkpoint — 2026-10-08

Measured the frozen framewise checkpoint on the existing **GTX1660SUPER / i5-9400F** host, one CPU/BLAS thread, serialized128ms windows. Replayed7440 decisions on the two inspected held TRAIN designs, both replicates, cross/stable/retreat profiles, and all five noise conditions. Included causal calibration/window normalization, command/saturation checks, prefix identification, forecast gate, full-window identification,174 features, scaling/availability/age, CUDA inference, CPU result delivery and warning thresholding. Physical sampling, disk/model/process initialization and sensor transport are separate.

- Worst noise-condition accepted-path p95: **18.7931ms**, exceeding the unchanged10ms target. Pooled accepted-path p99/max:19.2160/21.4216ms. **2309/2309 accepted decisions** miss10ms; across all decisions,2327/7440 miss. Reporting only rejected/cheap paths would hide this failure.
- First decision after model/CUDA initialization:269.0665ms, reported separately from warm percentiles. This is not a complete cold process-start measurement.
- All availability, age and warning decisions reproduce the cache. **8/7440 margin outputs** exceed the predeclared2e-6 replay tolerance; maximum deviation1.2837e-5. Single-window versus original batched CUDA identification is numerically sensitive. Intermediate feature deviations are disclosed, not passed off as identical representations. Replay and online qualification remain **failed**; no accuracy/safety claim follows from timing.
- Independent verification recomputed126 timing statistics and checked saved outputs against the frozen cache. Partial score increases reflect measured performance/report completeness, **not improved predictions**. No retraining, reserved-design evaluation, model promotion or relaxed research target.

Reproduce with `rtk proxy .venv/bin/python p4_groups/operational_confirmation_20261003/measure_framewise_latency.py` on the CUDA host; completed artifacts are immutable. Verify on either host with `rtk proxy .venv/bin/python p4_groups/operational_confirmation_20261003/verify_framewise_latency.py`. Evidence: [results](p4_groups/operational_confirmation_20261003/framewise_latency_20261008/results.json), [verification](p4_groups/operational_confirmation_20261003/framewise_latency_20261008/verification.json), and [protocol](p4_groups/operational_confirmation_20261003/framewise_latency_20261008/protocol.json).

### Frontier signed-headroom checkpoint — 2026-10-08

**Decision:** run a bounded extraction-versus-conversion diagnostic before implementing another carrier/transport correction. The frontier reviewer returned **CONCERNS**: transport is established in critical-only controls, not established as the dominant signed-margin error on ordinary trajectories. A successful oracle diagnostic would support targeted extraction work, not automatically a transport model. GRU remains a downstream candidate; no architecture/loss sweep or scope expansion.

An analysis of the existing frozen framewise cache separates current measured-input predictions from native missing-input extrapolation. Equal-design MAE on the two previously inspected held TRAIN designs:

| Condition | Current-window MAE | Current near-boundary MAE | Gap-output MAE | Current / all decisions |
|---|---:|---:|---:|---:|
| Clean | .008737 | .007959 | .034358 | 1037 / 1488 |
| IID1% | .011431 | .009827 | .029539 | 387 / 1488 |
| Correlated1% | .010696 | .009686 | .029809 | 358 / 1488 |

Near-boundary current counts are662/322/306 of893 for clean/IID1/correlated1. The gate describes input acceptance, **not a qualified reliable advisory**. Gap error is substantial, but current clean-boundary error already exceeds.005: gap handling alone cannot meet the accuracy target. These native-input masks differ from the post-hoc common-fresh fitter masks; do not interchange their denominators.

**Executed diagnostic (frozen protocol retained; results below):**

1. Preserve the four fitting/two validation/two inspected held-design partitions,128-sample aperture, measured-data gate, physical calibration, saturation handling and frozen framewise comparator. Analyze the already cached ordinary-mixture comparison; this does not promote the failed physical fitter or reopen its gated qualification stage.
2. Compare independently fitted copies of the existing six-feature ridge map: **A**, measured fitter pole features; **B**, oracle pole features for the physical branch corresponding to the fitter's selected component; **C**, oracle critical-branch pole features. Keep the same measured residual feature in all arms. Define endpoint growth/frequency, slope summaries over exactly the same128 samples, and growth/frequency ratio before fitting. Use the existing proper-training-only, clean/IID1, leave-one-design-out degree/alpha selection. No oracle routing or inference inputs.
3. Validate critical-branch tracking on the ordinary trajectories first. Freeze which quantity defines the oracle (flutter-onset branch versus instantaneous maximum-growth branch), record dominance switches separately from tracking ambiguity, and verify identity near close branches. The existing nearest-pole/eigenvector-overlap helper validated critical-only controls, not these ordinary-trajectory invariants. Establish fitter-to-branch correspondence offline from spatial/temporal signatures; disclose ambiguous correspondence and branch failures without removing them from original totals or interpreting their contrasts as estimation/selection effects.
4. A→B diagnoses numerical estimation conditional on correspondence; B→C diagnoses branch choice/visibility. These are **not additive causal decompositions**. Offline critical-branch sensor contributions versus other components and injected noise distinguish candidate visibility issues from selection issues; strong contribution is not proof of recoverability. Oracle arms are information counterfactuals, not deployable measurements.
5. On identical eligible fresh windows, report equal-design paired MAE for every primary condition, the exact boundary slice, unstable and farther-stable subsets, dangerous optimism and the prescribed sign-exclusion slices. Retain all-frame/current/stale/gap counts and unknowns. No eligible unstable points in the prescribed sign slice means unknown, not a passed sign criterion.
6. Oracle gains on both development-held designs, conditional accuracy targets and reduced dangerous errors with farther-stable degradation≤.005 would justify targeted extraction work **conditional on measured critical-mode support**. A later correction must demonstrate actual signed-margin improvement on ordinary mixtures under measured-only support selection. Oracle failure under a four-design ridge leaves calibration, training support and map capacity unresolved; it does not prove nonidentifiability. Intermediate evidence remains inconclusive. Fresh-window success cannot satisfy the all-frame accuracy/availability contract.

The review and cache decomposition did not improve predictions. The completed offline diagnostic below also does not warrant a progress-score increase, model promotion or threshold change. Existing runtime/replay failures remain recorded separately.

### Completed oracle and regularization checkpoint — 2026-10-09

Evidence: [frozen protocol](p4_groups/operational_confirmation_20261003/headroom_oracle_20261008/protocol.json), [A/B/C results](p4_groups/operational_confirmation_20261003/headroom_oracle_20261008/results.json), [paired design errors](p4_groups/operational_confirmation_20261003/headroom_oracle_20261008/paired_results.json), [independent verification](p4_groups/operational_confirmation_20261003/headroom_oracle_20261008/verification.json). C is the instantaneous maximum-growth transverse branch at the window endpoint, followed across the same128 samples, not a fixed flutter-onset branch. All original populations, unavailable windows and frozen comparators are preserved.

Equal-design MAE on full critical-valid fresh support, with separately fitted maps on identical training rows:

| Condition | Fresh / all | Measured A | Critical oracle C | C near-boundary |
|---|---:|---:|---:|---:|
| Clean | 1036 / 1488 | .015790 | .012916 | .005364 |
| IID1% | 386 / 1488 | .011862 | .010033 | .006748 |
| Correlated1% | 357 / 1488 | .011023 | .009604 | .006928 |

C improves overall fresh MAE on both held designs, but neither held design meets the clean≤.010 target. Clean-boundary C worsens on D000053 (.005571→.005715) and remains above.005 on both designs. Despite zero boundary overestimation>.02 events, positive point margins on truly unstable windows remain130/130,110/110,110/110. The prescribed unstable sign-exclusion slice and farther-stable noninferiority are **unknown**, because eligible points are absent.

Matched A/B/C support resolves533/209/200 fresh windows;503/177/157 fresh windows remain spatially ambiguous. On that separately trained/selected subset, clean MAE A/B/C=.015868/.013043/.011200; IID1=.008727/.006618/.005567; correlated1=.007771/.005877/.004910. These conditional gains do not explain the full-support failure or demonstrate recoverability in unresolved windows. Nonorthogonal pair energies include interference; strong critical contribution is not proof of measured observability.

The frontier reviewer selected one smaller fixed shrinkage probe before any physical zero-crossing constraint: retain full-support C, degree2, features, fold-local scaling and squared-error objective; replace alpha3162.27766 by zero using rank-aware least squares. No search. [Protocol](p4_groups/operational_confirmation_20261003/headroom_oracle_20261008/regularization_probe_protocol.json), [results](p4_groups/operational_confirmation_20261003/headroom_oracle_20261008/regularization_probe.json), [verification](p4_groups/operational_confirmation_20261003/headroom_oracle_20261008/regularization_probe_verification.json); runner: `probe_headroom_regularization.py` in the operational-confirmation directory.

| Condition | Existing C fresh MAE | Zero-penalty C fresh MAE | Existing C boundary MAE | Zero-penalty C boundary MAE |
|---|---:|---:|---:|---:|
| Clean | .012916 | .027879 | .005364 | .012744 |
| IID1% | .010033 | .019164 | .006748 | .011951 |
| Correlated1% | .009604 | .017992 | .006928 | .011509 |

Removing the penalty fixes most positive-on-unstable points, but introduces stable-side wrong signs110/483,70/180,66/165 and boundary optimism48/662,13/322,12/306, versus zero for the original full-support C. Proper-TRAIN clean in-sample MAE improves.013357→.009664, while fixed-alpha LODO MAE worsens.016840→.045424. This supports a shrinkage/generalization tradeoff in this conversion map, not fundamental nonidentifiability. **Reject alpha0; no physical zero-crossing or transport correction is justified by this result.**

Verification:840 metric checks, five independent map/LODO selections,2351 endpoint pole checks; the fixed probe adds120 independently recomputed metric rows and five NumPy/SciPy least-squares checks. No measured predictor, warning policy, reserved-design evaluation, qualification status or progress-ledger value changed.

**Next bounded campaign, not executed:** expand only proper-TRAIN design support using eligible additional designs with the same label/calibration convention; exclude every copy of validation and development-held designs. Preregister the design list and retain the same ordinary-mixture profiles, causal128-sample windows and clean/IID1 training views. Compare original-four versus expanded support with the existing degree2 A-measured and C-oracle maps, identical support within each contrast, existing row-loss fitting and equal-design scoring. Hold normalized regularization strength constant per arm and paired fit: scale sum-loss alpha with total fitting weight, including within folds. No penalty/model search or simultaneous physical-map constraint. Preserve all gates, gaps and the frozen framewise comparator.

Require overall and boundary improvement on both inspected held designs without increased dangerous optimism or wrong signs across primary noise families. C-only improvement remains diagnostic; measured A must demonstrate the downstream benefit and ultimately beat the deployed comparator and absolute targets. Absent farther-stable/sign-exclusion coverage remains a qualification blocker. Broader support is a hypothesis to test, not an established fix.



Record accepted fitting-row contributions by design and noise family in both campaign arms. Gate-dependent sampling can change effective training weights even with identical trajectory generation; attribute a gain to the expanded training population, not automatically to design diversity alone.

### Completed additive spectral checkpoint — 2026-10-09

The user-requested feature test is separate from the unexecuted proper-TRAIN support-expansion campaign. Retained all174 inputs; tested summed-sensor spectral entropy/fitted-frequency neighborhood mass and broadband spatial concentration separately against the unchanged framewise baseline and size-matched shuffled controls.15 fixed GPU fits, three seeds, original four/two/two design partitions, identical windows/gates/ages, two fresh paired IID/correlated noise draws at1%/5%. No combination or parameter search. Archived seed42 baseline predictions reproduce exactly.

**Neither addition earns adoption.** Spectral-family primary mean overall/boundary errors worsen. Spatial concentration has small average gains, but the shuffled spatial control is better on both all-frame and boundary means in all13 held noise conditions; gains are not consistent across design/seed and positive-on-unstable points increase. Profile/rate/bin-transition regressions and gap populations remain explicit. Missing farther-stable/prescribed unstable-exclusion coverage remains unknown. These are two repeatedly inspected development-held designs, not blind confirmation.

Evidence: `p4_groups/operational_confirmation_20261003/additive_spectral_20261009/{protocol,training,results,verification,decision}.json`, dataset/predictions and15 checkpoints. Independent checks cover1203 waveform-derived descriptor windows,5635 metric rows,9360 saved-network frame replays and468 paired differences. A caught diagnostic float32 narrowing was corrected without changing predictions or tolerances; noise-wording erratum is recorded in verification.json. No operational predictor, qualification or progress-ledger change. The controlled support-expansion campaign above remains the next bounded intervention; this result does not authorize another feature/architecture search.


## Finite experiment sequence and stopping rules

1. **Close the mechanism question — one bounded diagnostic.** On existing controls, separate carrier curvature from spatial transport/coupling; include ordinary multimode waveform evidence before claiming applicability. Test structural-calibration ambiguity using controlled observation likelihoods/physical contrasts, not nearest feature neighbors alone. Reuse completed diagnosis; do not rerun it for reassurance. Define physical accuracy and guardrail checks before fitting. Iterate on evidence-backed corrections within this mechanism, with explicit progress checkpoints; a first failed correction does not abandon the route. The current failed fitter is not qualified by this proposal.
2. **One comparative development campaign.** Corrected observation/inference candidate, strongest calibrated physical/simple map, original-feature ridge, framewise, GRU with identical improved inputs, and direct warning reference. Include a population physics-prior/library inference competitor; target-calibrated multi-speed methods belong in a separately labeled extra-information track. Bound each tunable family to a preregistered small grid (maximum three configurations, three seeds). All use the same age policy, uncertainty/alarm calibration access and observation budget. Freeze the candidate using development results only.
3. **TRAIN-only operational go/no-go checkpoints.** Broader disjoint development designs and propagated ordinary trajectories, all five noise families, with the absolute endpoint gates and uncertainty on comparative improvement. Use diagnosis-driven corrections and repeat affected comparisons while evidence shows progress toward signed accuracy, availability and delivered lead. Reassess the mechanism when progress stalls or a physical limitation is demonstrated, not after an arbitrary attempt count. Do not open unrelated architecture/loss/feature searches.
4. **One blind confirmation and independent validation.** Lock source, preprocessing, models, policies, calibration, physical envelope and statistical analysis before any final designs are scored. Execute the independent-source/model-form track with the same frozen predictor; any permitted acquisition calibration must already be in the contract. Use limited validation cases to establish the scope actually supported, not a fictitious universal transfer claim. No retuning on final outcomes.
5. **Stop and write when all gates pass.** Deliver one frozen method, reproducible benchmark/artifacts, physical mechanism/limit evidence, matched strong comparisons and independent validation. Do not continue improving MAE after the declared endpoint is satisfied.

**Failure decisions:**

- Accurate observed dynamics but ambiguous headroom → pursue evidence-backed population calibration/uncertainty corrections, with measurable progress checkpoints. Reject the current transferable-margin claim only if the ambiguity remains demonstrably unresolved under the observation contract; more epochs are not the default response.
- Good short-record points but poor operational availability/lead → withhold the early-warning claim and address the measured operational bottleneck within the declared observation contract. Reassess only when progress stalls or the limitation is demonstrated. A longer window or commanded re-excitation requires explicit scope approval.
- Simple competitor matches the proposed method → withhold unique neural superiority, not the headroom/warning objective. Continue physically justified improvements and fair comparisons at progress checkpoints; accept a simpler winner if it earns the scientific and engineering gates. Reassess the publication claim if no substantive independently validated benefit emerges.
- Blind or independent-model validation fails → report the failure and narrower demonstrated scope. Do not recycle the final set as a new validation set and claim the original target passed.
- No compatible independent-validation route → the full endpoint remains unmet. A simulation-only submission would require an explicit revised claim and publication assessment, not silently dropping this gate.

## Finished artifact and publication case

A finished P4 result would answer: **how much speed headroom can be inferred, on which unseen plates, from how much measured information, with what optimism risk, availability and delivered warning benefit—and why?**

The paper should center that result and its physical mechanism, not a catalogue of G1/G4/loss/architecture experiments. Negative controls and discarded paths explain limits; only necessary experiments enter the main narrative. Quartile/venue fit and the closest papers' full protocols must be checked before submission. The claimed engineering benefits remain research evidence, not operational certification.

## Consolidated frontier diagnosis and experiment priorities — 2026-10-09

User requested broader diagnosis using the full working-feature history and an unranked
constraint-compatible feature/model survey, not an automatic support-expansion recommendation.
Evidence and exact read-only frontier findings are retained in
`p4_journal/review_20261009/{working_feature_history.md,constraint_features_and_models.md,frontier_evidence_packet.md,frontier_review.json}`.
The historical report reconstructs captured Butler evidence; the survey was Butler-authored.
Additional Butler source checks covered subsets but did not publish full audits; no independent
full-report verification is claimed.

The separately completed384-sample response-history comparison adopts no arm.
Pooled history improves average overall/boundary error across13 held noise conditions, but
ordered and pooled history both worsen distinct-regime and current-boundary error across all
seven primary views. Gains concentrate in dwells, with sign regressions and missing guardrail
populations. This does not justify making temporal modeling the leading correction, and does
not reject independently measured multi-speed information or all bounded recurrent methods.
See `p4_groups/operational_confirmation_20261003/multivel_history_20261009/{protocol,results,verification,decision}.json`.

**Prioritized proposals at the initial review; completed execution recorded below:**

1. **Cached fitting/context factorial first.** Four arms: retain full174 versus zero only
   second-mode67 columns; original all-frame MSE versus accepted-current-only MSE.
   Keep176 input width,32→16 head, common proper-training scaling, four/two/two design
   partitions,128-sample aperture, gates/calibration/saturation, seeds42/43/44, optimizer,
   normalized-loss regularization, budget/patience and checkpoint criterion. For every arm
   use its new prediction only on accepted current windows; use the corresponding seed's
   frozen baseline native gap predictions otherwise. Apply that common composite rule to
   all-frame validation and evaluation. Gaps remain unqualified extrapolation, not available
   measurements. Test current-head objective interference and adverse second-block interaction
   without changing observation acceptance or discarding difficult frames.
2. **Structural support × richer matched-modal information.** Preregister original-four versus
   expanded eligible TRAIN support, crossed with measured174 versus diagnostic matched-modal
   counterfactual174, using one fixed nonlinear head/training recipe. Exclude validation/
   development-held copies and predesignate additional TRAIN design evaluation folds.
   Validate ordinary-mixture correspondence, conditioning and actual channel-calibration
   transforms before oracle replacement; preserve measured slot/routing, strength and validity
   semantics, retain unresolved rows and report replacement coverage. This is a physical-
   description counterfactual, not an oracle upper bound or pure estimation-error attribution.
   Include the missing farther-stable/sign-exclusion regimes. Only measured improvement earns
   accuracy evidence; oracle benefit after adequate support motivates targeted extraction.
3. **Conditional bounded GRU, not another unbounded sequence model.** After the current-head/
   support reference is established, one GRU32→16 over lags4/2/0, reset each decision,
   max383ms oldest waveform sample, versus same-span ordered/pooled inputs and flags/age-only
   GRU. No8s hidden state or architecture/loss sweep. Require changing-regime/current-boundary
   and overall gains beyond pooling and metadata, not dwell-only improvement.

Every proposal retains original denominators and primary absolute targets: all-frame design
MAE≤.010 clean/≤.015 each1% family; exact-boundary≤.005/≤.010; boundary optimism>.02≤5%,
each eligible sign-side wrong fraction≤10%, farther-stable development allowance.005.
Report per-design/seed results, gaps and transition jumps, retreat/stress behavior and unknown
populations. Current-only fitting gains support shared-objective interference;107-content
gains support a second-block interaction; their interaction permits neither single-cause claim.
Reject harmful transition/sign regressions. Measured support gains and persistent oracle
benefit distinguish population/conversion from extraction hypotheses; failure of both does
not establish nonidentifiability. Full outcome rules and prerequisites are in frontier_review.json.

This review identifies falsifiable directions, not proof the endpoint is reachable.
Qualified availability, uncertainty/run-risk calibration, lead/nuisance benefit, runtime/
replay qualification, fresh blind designs and independent physical/model-form validation
remain unpassed. No information-contract relaxation, fitted predictor change, reserved-design
evaluation or progress-ledger increase follows from the review.

## Completed frontier campaign — 2026-10-09

All three requested experiments ran on the SSH GPU with independent verification.
Evidence: `p4_journal/review_20261009/completed_campaign_frontier_packet.md`,
`completed_campaign_analysis.json` and `completed_campaign_context_appendix.json`.
The existing frontier reviewer received the actual evidence and requested
design/seed, population, counterfactual-coverage and stress context for reassessment.

- Cached four-arm factorial:12 fits; all three interventions regress the primary
  mean all/boundary/current-boundary comparisons. Fixed native-gap contribution
  alone exceeds all-frame absolute error gates, even with perfect current outputs.
  It is a narrow elimination check, not a root bottleneck identification.
- Support/content factorial:24 fits on original4 versus original4+additional8
  excluding each of three designated four-design evaluation folds. Measured
  expansion helps12-design aggregate errors but not every fold/design/seed;
  original3-profile improvements coexist with new2-profile deterioration.
  Physical-description replacement is partial and noise-dependent, not a full
  oracle. Absolute point-error targets remain failed.
- Bounded-GRU comparison:36 fits. Changing-regime gains over framewise now exist;
  uniform same-span superiority does not. Stable-sign, optimism, retreat and
  farther-stable regressions prevent adoption; one farther-stable increase
  exceeds the .005 noninferiority allowance. This rejects promotion of this
  implementation, not all recurrence or useful bounded response history.

Actual executed support scorer/checker are `score_support_modal_corrected.py`
and `verify_support_modal_corrected.py`; preserved originals are pinned historical
sources. Diagnostic-complex recovery and scorer/checker corrections are explicitly
recorded; none changes model features, predictions, thresholds or tolerances.
GRU independent verification checks10,530 metric rows and3,456 sampled neural
decisions, not every neural output. All original frames remain in denominators.
New safety-slice observations are not qualified availability. No predictor,
reserved-design evaluation, independent-validation claim or progress credit changes.

### Completed frontier reassessment

Exact returned assessment: `p4_journal/review_20261009/completed_frontier_reassessment.json`.
No arm earns promotion or endpoint credit. Frontier accepts the experiment1
leverage criticism: frozen-gap floors exceed both all-frame and boundary targets
even with perfect current predictions. Do not repeat that narrow elimination test.

Observation erasure/extrapolation is an evidenced major bottleneck, not proof that
rejected waveforms contain no transferable information. Expanded fitting matters
but does not establish conversion adequacy. Noise-dependent partial physical
replacement cannot exonerate extraction. Bounded measured content now improves
changing-regime errors over framewise, without consistent order-specific benefit.

**Earlier proposed diagnostic, now deferred by frontier's subsequent direction:**
Keep expanded-support folds and the same383ms-reset GRU; compare forecast-gate zeroing with genuinely finite
pre-forecast-gate174 estimates in rejected slots, plus a same-width shuffled-content
control. Preserve gates, flags/age, calibration, physical exclusions, partitions,
paired noise sources and every original denominator. Rejected outputs remain
unqualified. First establish that real pre-gate estimates can be distinguished
from manufactured/fallback values; never retain invalid/saturated/commanded or
genuinely unavailable estimates. Require primary gap/boundary gains without
current-boundary, sign, optimism, retreat or farther-stable regressions.
The acquisition envelope must govern this decision, not be narrowed retrospectively.
No new test or information-contract change is authorized merely by recording it.

## Frontier-led operating-point preflight — 2026-10-09

Scientific direction and acceptance decisions now belong to frontier; dispatcher
implements, executes and verifies. The controlled single-versus-multiple operating
point acquisition supersedes the deferred gate-retention proposal above.
Direction: `p4_journal/review_20261009/frontier_research_direction.json`.

Proper-fitting-only S/M/R acquisition and accounting completed:1,632 continuous
447ms physical episodes,8,160 paired noise-view histories. No predictor fits or
evaluation/reserved-design acquisition. Numerical refinement and same-equation
full/reduced comparisons passed36 representative cases; original CUDA gate passed
a1440-window parity sample. Neither is physical-model validation.

**Decision: campaign stopped.** The common900–2240m/s grid creates758/1632
all-missing histories in each primary noise family and exact synthetic-dataset
MAE floor.080928365. Every such history is outside the original near-flutter
margin population; its boundary collision floor is zero. Extreme-grid displacements
also invalidate a small-disturbance physical interpretation. These are documented
acquisition/model-domain failures, not evidence against multiple operating points
or an attainable accuracy limit for the original physical task. No cases removed.

Frontier requires displacement, rotation/slope and small-disturbance assessment
of D000033/D000068 at1240m/s before an independent accelerated-flow comparison.
That comparison needs defensible governing assumptions, rate/timescale bounds,
and uncertainty/model discrepancy below the S-versus-M/R response contrast in
each order. No configured independent treatment or applicable analytical bound is
presently available. No invented rate limit or further predictor fits.
Passing would authorize prospective acquisition redesign only, not full training
or accuracy/availability/warning qualification.

Evidence: `p4_journal/review_20261009/multipoint_preflight_frontier_packet.md`,
`multipoint_preflight_frontier_decision.json`,
`multipoint_frontier_amplitude_amendment.json`, and
`multipoint_independent_aero_prerequisite.json`; complete acquisition receipts under
`p4_groups/operational_confirmation_20261003/multipoint_preflight_20261009/`.
Predictor, protected evaluation splits and progress ledger remain unchanged.

### Geometry prerequisite completed; aerodynamic prerequisite remains

The1240m/s D000033/D000068 geometry screen executed24 S/M/R×replicate0/1×
reduced/full structural cases with physical displacement, analytic slopes and
FSDT rotations assessed across the plate. Conservative spatial bounds at all1789
integration samples show no obvious large-deflection/slope/rotation violation.
Frontier supports that narrow conclusion; no continuous-time extrema, nonlinear
structural error bound or accelerated-flow validity is certified.
Evidence: `multipoint_preflight_20261009/checkpoint_geometry.json` under the
operational campaign, and
`p4_journal/review_20261009/multipoint_geometry_frontier_decision.json`.

Do not repeat the same-equation geometry screen. Multipoint acquisition and related predictor
fitting stay stopped until an independently justified accelerated-freestream
treatment or applicable analytical bound addresses32ms transitions and downstream
memory. Its uncertainty in S-versus-M/R contrasts must be below both orders'
effects, particularly the approximately5.6% D000068 effects. No arbitrary rate
limit, retrospective envelope restriction, converter experiment or endpoint credit.

## Target-directed frontier continuation — 2026-10-09

User requested a next direction with leverage on the unchanged targets.
Frontier selected **gate-erasure ablation, preceded by provenance and leverage
checks**; preserve but pause the controlled multipoint route rather than starting
a CFD-development detour. Exact direction:
`p4_journal/review_20261009/gate_erasure_target_direction.json`.

Scientific question: does forecast-gate zeroing discard genuine measured content
that can improve signed-margin predictions, or do rejected-window estimates add
noise and dangerous bias? The majority gap population and its error mass justify
testing this hypothesis; usefulness of rejected estimates is not established.

**Authorized now, no fitting:** replay the identical174-coordinate extractor on
the existing calibrated128-sample waveforms. Classify command, nonfinite,
saturation, absent/invalid estimates and forecast rejection separately; retain
only genuine finite, command-free, cap-free forecast-rejected content.
Reproduce accepted features and preserve qualification flags, age, per-feature
availability, all original frames, physical calibration and383ms history.
Compute exact equal-design weighted-median bounds for genuinely bit-identical
remaining inputs, overall and at the exact boundary; no approximate collisions.
Report affected/unaffected comparator error contributions and ideal affected-only
correction leverage, not a bound on globally refitted predictors.

**Fits require the returned preflight receipt and frontier's continuation decision.**
Stop if any remaining collision floor forecloses its corresponding primary target,
or eligible content/error mass is absent or insufficient. Conditional comparison:
same expanded folds and reset GRU32→16, original zeroing versus real rejected
content versus a paired, within-partition shuffled-rejected-content control.
All other support, optimizer, selection, loss and seeds stay fixed.

Success requires original-denominator overall/boundary gains over both controls
across primary conditions without current-boundary, optimism, sign-side, retreat
or farther-stable regressions. Partial results return to frontier without repeated
tuning or qualification claims. Failure stops this retention mechanism, not all
raw-waveform information; next fork is extraction versus acquisition-envelope
adequacy, not another feature/loss sweep.

Initial executable provenance inventory completed:2094 rows×124 frames across20
development designs; all20 raw waveform shards are available on the original
execution host. Rejected174 rows were not stored as pre-gate estimates:
additional-row acquisition runs full-window extraction only after forecast
qualification. Replay therefore newly evaluates the same measured-only extractor,
not recovers nonexistent cached rows; extra computation must be disclosed.
Inventory receipts: `gate_erasure_prerequisite_inventory.json` and
`gate_erasure_remote_provenance_inventory.json` in `p4_journal/review_20261009/`.
The complete rejection-provenance/collision/leverage receipt is not yet produced.
No fits, changed predictions, qualification or progress-ledger credit.

### Auxiliary-extraction terminology and execution smoke

Frontier's source-order amendment calls the active intervention **forecast-gate
bypass for auxiliary extraction**, not recovery of previously computed features.
Eligibility is established from the existing command/finite/cap/forecast checks
before full extraction; extraction failures are separate from forecast rejection.
Accepted cached rows remain verbatim. Additional extraction cost must be included
in eventual full-path timing, not hidden by accepted-path-only reporting.
Exact amendment: `p4_journal/review_20261009/gate_bypass_provenance_amendment.json`.

Actual CUDA smoke on D000033/cross/replicate0/IID1:124 windows,13 command-rejected,
28 cap-rejected,83 eligible for prefix forecasting;36 cached accepted and47 finite
forecast rejections. One rejected sample64:192 window (skill.243138) yields eight
estimated modes and174 finite coordinates through the unchanged full extractor.
Its qualification stays false. This establishes an executable example, not
population-wide eligibility, transferable information or error improvement.

Accepted frame0 replay is finite but NOT bit-identical to the cached float64 row:
maximum coordinate difference1.47438e-12. Disclosed without widening any tolerance
or claiming exact reproduction; complete replay differences remain to be accounted
for before fitting. Receipt: `p4_journal/review_20261009/gate_bypass_execution_smoke.json`.
No complete collision/leverage receipt or training authorization yet.

Frontier accepted the smoke's narrow scope and preservation plan:
`p4_journal/review_20261009/gate_bypass_smoke_frontier_review.json`.
The replay discrepancy alone neither demonstrates material error nor certifies
equivalence. The full receipt must identify affected coordinates and assess
standardized-input effects, mode ordering/availability and gate classifications
under existing criteria. Bit identity is not a scientific acceptance requirement;
no tolerances may be widened to obtain a pass. Population-wide eligibility,
reproduction, exact collision floors and leverage remain required before fits.

### Complete gate-bypass receipt and stop decision

The population-wide receipt is complete and independently verified:
`p4_journal/review_20261009/gate_bypass_complete_receipt_frontier.md`.
Frontier explicitly **stopped gate-bypass fitting** because the exact retained-input
clean all-frame floor exceeds the target in every original grouped fold. This
forecloses this modal representation, not raw-waveform information or the endpoint.
Accepted cached rows, qualification, reserved evaluation and progress ledger remain
unchanged; no fits or endpoint credit.
Exact decision:
`p4_journal/review_20261009/gate_bypass_complete_receipt_frontier_decision.json`.

Next authorized checkpoint: partition remaining collisions by raw-response
eligibility within the unchanged383ms span, using existing rejection metadata,
original equal-design denominators, profile/target ranges and comparator errors.
No acquisition or extraction-family comparison. A bounded raw-response-versus-modal
experiment remains unauthorized until frontier assesses this partition.

### Raw-response partition completed

Frontier narrowed the requested absence claims and authorized per-sample mask
inspection. Completed and independently verified; decision packet:
`p4_journal/review_20261009/raw_response_partition_frontier_packet.md`.
The clean collision excess primarily includes raw-present/no-modal histories,
especially farther-stable profiles. No-sample restricted bounds do not foreclose
the targets; this does not demonstrate useful raw-response estimability.
An unchanged old gap fallback is nevertheless insufficient at the boundary,
even with perfect raw-present corrections. Known mask/timebase information must
be distinguished from waveform absence; no immutable information limit inferred.
Global pre-cap truncation is only a conservative sensitivity, not a source-policy
change. All original weights, qualification and protected populations preserved.
No fits or endpoint credit; next scientific decision requested from frontier.

### Frontier-selected bounded raw-content experiment

Frontier assessment and fixed next direction:
`p4_journal/review_20261009/raw_response_partition_frontier_decision.json`.
Stop modal-only repair. Compare a modal+raw GRU32 branch with an exactly matched
zero-waveform branch using the same masks/ages/capacity and fixed16-unit scalar
head; raw content must beat the matched control, not just a frozen reference.
Train both on every original frame, including no-sample histories: the old gap
fallback must not be retained, and new gap estimates remain unqualified.
Same383ms physical aperture, reset encoders, original calibration/exclusions,
populations/denominators and training convention; no new acquisition or search.

**Before fitting:** demonstrate frozen causal/masked input construction, unchanged
cache/calibration, invalid-value exclusion and zero-waveform input equivalence.
Actual-encoding collision bounds must pass primary targets; audit bounds containing
extra mask/time coordinates cannot be imported into a different model encoding.
Only after preflight:18 fixed fits (two arms×three folds×three seeds), with original
absolute targets and conditional sign/optimism/current-boundary/retreat/farther-stable
safeguards. No quantile-calibration, qualification or endpoint credit yet.

### Prior stop — superseded by scoped resumption below

User rejected the experiment chain for lack of target-performance progress.
The raw-versus-mask preflight and18 fits above are **not currently authorized**.
No further experiments or audits. Frontier withdrew the claim of an evidence-backed
route to this endpoint and stopped the transport/spectral/multipoint/raw-model
proposal chain. Existing predictors remain references, not promoted solutions.
The endpoint remains unchanged and unachieved; a substantive scientific reset
must precede any new campaign.

### Scoped execution resumed — 2026-10-10

User resumed work with frontier. Only the fixed raw+modal R versus zero-waveform
M comparison above is restored; no new acquisition, model/loss/source search or
broader experiment chain. Frontier confirmed that R's actual-encoding bounds
gate execution; M's bounds are diagnostic and do not veto it. Full-objective
microbatch2048 accumulation retains one Adam step per epoch.

The integrated input/provenance/causality preflight passed on the original CUDA
host: all2094 source-row rejection masks and calibration parameters matched,
and all42 primary candidate-R bounds passed (maximum .001425733).
Receipts: `p4_groups/operational_confirmation_20261003/raw_modal_20261010/`.
The18 fixed fits are running; no accuracy, qualification or endpoint progress
is claimed yet. No-sample original-denominator contributions and raw-present/
no-modal gains remain required in the comparison. Existing predictor, reserved
evaluation and progress ledger stay unchanged.

### Completed scoped comparison — 2026-10-10

All18 fixed fits and39 development fold/noise scores completed;336 independent
primary per-design error/denominator checks passed and held predictions remain
NaN. `raw_modal_20261010/comparison_summary.json` records all seven primary
conditions, matched-control effects, original-weight no-sample contributions
and safeguards; the journal records the numerical comparison.

Both R and M reduce overall/boundary errors versus every frozen reference in
all primary conditions. R's incremental waveform gain is small, with design/seed
heterogeneity; clean overall/boundary paired intervals include zero. Raw-present/
no-modal subsets do show additional mean gains with negative paired intervals.
The substantial shared improvement includes much better no-sample predictions;
it cannot be attributed to waveform content or to any single bundled M change.

All seven primary conditions still fail absolute overall/boundary accuracy and
boundary optimism targets. Retreat and unstable-sign safeguard regressions
prevent promotion; correlated current-boundary errors also worsen versus the
frozen modal GRU. Farther-stable non-inferiority passes. Measured partial
development-error progress does not complete the endpoint. No predictor,
qualification, reserved evaluation or progress-ledger change; no further
experiment/search is authorized by these results.

Frontier returned CONCERNS and closed this fixed experiment without variants.
R/M remain improved development references, not operational replacements:
`p4_journal/review_20261009/raw_modal_completed_frontier_decision.json`.

### User-authorized stronger-model/training trials — 2026-10-10

User requested actual model/training improvements, not more diagnostic campaigns.
Keep the useful modal, calibrated waveform and sample-mask/age information.
Two fixed arms share a multiscale residual TCN waveform encoder with early/late/
last summaries and the existing modal GRU: native MSE versus boundary-localized
optimism/sign penalties with a farther-stable pessimism counter-penalty.
Soft training penalties do not guarantee the acceptance constraints.

Same data,3 folds,3 seeds and original validation/checkpoint convention;18fits.
Actual512-decision CUDA forward/backward and causal/reset/loss-gradient checks
passed. Identical input provenance/bounds are reused, not re-investigated.
Original protocol: `p4_groups/operational_confirmation_20261003/raw_tcn_20261010/protocol.json`.
Comparisons retain R/M and all frozen references, original denominators and
unchanged target/safeguard gates. User stopped the run for execution optimization
after2h33m with0 completed fits. No target-performance results or qualification/
operational/progress-ledger change.

### Tested execution optimization — 2026-10-10

Use equivalent float32 polyphase causal convolution for training and native
dilated convolution for inference; dense weights, receptive field, all inputs
and the full-frame losses remain unchanged. Original source archived with the
interrupted protocol. New tested protocol:
`p4_groups/operational_confirmation_20261003/raw_tcn_polyphase_20261010/protocol.json`.
Its `execution_verification.json` records actual full-epoch numerical/performance
proof; freezing requires verification against the current source.

Full fold0 train+validation epoch: MSE13.383848→11.096455s;
risk13.492685→11.240913s. Approximately17% shorter.20 convolution boundary/
gradient/causality cases and actual CUDA model/loss checks passed. Largest
full-epoch gradient discrepancy1.86e-8; one-step parameter discrepancy7.89e-6.
Numerically equivalent within checked float32 tolerances, not bit-identical
training trajectories. No accuracy or transfer improvement established.

Depthwise-separable blocks tested before adoption:16% shorter batch steps but
less dense channel-mixing capacity and higher activation memory; not adopted.
Intermediate groups2/4/8 were tested separately before adoption; see below.
Full fitting remains stopped; the optimization does not make18fits a short job.

### Grouped convolution candidate check — 2026-10-10

Same512-window float32 training-step probe, native cuDNN: groups1/2/4/8 took
217.218/425.143/155.345/152.093ms (median5 steps). Finite gradients,
causality/reset/order checks passed. Native groups4/8 beat the tested polyphase
grouped implementation. Groups4 is the conservative runtime candidate;
groups8 is only about2% faster but restricts channel mixing further.
No grouped margin-accuracy result or adoption claimed. The frozen comparison
remains the equivalent dense polyphase optimization, with full fitting stopped.
Evidence lives in `raw_tcn_polyphase_20261010/grouped_native_convolution_probe.json`.
`verify_raw_tcn_execution.py` provides the exercised reusable full-epoch check
and does not overwrite frozen evidence.

### Selected grouped+dilated model running — 2026-10-10

User selected groups4 and authorized integration/running, superseding the
stopped dense comparison as the active model trial. No more convolution search.
Native groups4 causal blocks retain32 latent channels, seven two-convolution
blocks with dilations1/2/4/8/16/32/64 and receptive field509. All raw/modal/mask/
age/time inputs remain.37,025 parameters; no operational adoption.

Integrated checks passed28 grouping-isolation/causality/gradient cases and
actual512-decision CUDA reset/order/backward/loss checks. Actual fold0 full
train+validation epochs: MSE10.120412s, risk9.857944s;29,760 fitting/4,960
checkpoint-validation decisions. Peak allocated2,292,174,336bytes, approximately
the previous dense full-epoch allocation; no extra GPU concurrency demonstrated.
These establish execution behavior/cost, not margin accuracy.

Frozen protocol/proof: `raw_tcn_grouped4_20261010/`. Training started as
`p4-tcn-grouped4-20261010.service` on the original GTX1660SUPER host via Tailscale;
active/running and first fold0/MSE/seed42 epoch observed. Service owns the fixed
18fits plus scoring and guarded comparison; an SSH monitor disconnect does not
stop it. Do not relaunch without checking execution ownership.
Same original splits,3 seeds, MSE/risk objectives, checkpoint convention,
R/M and all frozen references, denominators and target/safeguard gates.
Held designs, qualification, operational predictor and progress ledger unchanged.
No completed-fit accuracy result yet. Stopped dense source/evidence preserved
under `raw_tcn_polyphase_20261010/`, including `frozen_source.tar.gz`.

### Completed fixed modal-only tree comparison — 2026-11-02

One fixed histogram-boosting estimator, three original development folds and
seed42 used the cached causal174-feature/flag/age histories at lags4/2/0.
No waveform input, hyperparameter search, validation selection or protected
held prediction. This is a limited-input competitor, not waveform attribution.

| Primary condition | All-frame MAE | Near-boundary MAE | Boundary optimism |
|---|---:|---:|---:|
| Clean | .028681 | .022863 | 34.23% |
| IID1% | .047410 | .033179 | 55.95% |
| Correlated1% | .049233 | .033837 | 56.05% |

All seven primary views, including fresh noise draws, fail both MAE targets
and the5% optimism limit. Both current sign-side limits and retreat
non-regression pass, but farther-stable non-inferiority fails against the raw
and mask comparators in all six noisy views. Reject this fixed candidate
without variants; this is a negative development result, not model promotion.

Actual fold times60.4344/60.2938/58.6396s; maximum process high-water RSS
1,611,632KiB. Independent audit passed2,485 numerical/population comparisons,
maximum discrepancy4.17e-17; protected held predictions remainNaN and progress
ledger hash is unchanged. Evidence and reproducible verifier:
`modal_tree_20261102/{protocol.json,comparison_summary.json,verification.json,verify.py}`.

Original grouped4 results remain unknown: the accessible mounted remote
checkout predates that experiment and lacks its directory. CUDA runtime has
been restored, but missing waveform/admissibility arrays and provenance shards
must not be replaced silently. A separately frozen one-design restoration
pilot is under `recovery_20261102/protocol.json`; no18-fit restart or new
qualification claim follows. All acceptance thresholds and the ledger remain
unchanged.

### Exact source-replay checkpoint — 2026-11-03

The first restored development source shard, D000007, matches the original
42,258,789-byte artifact and SHA256 exactly. Actual acquisition took18.65s,
including the child's imports/contracts; total helper runtime23.37s.
Evidence: `recovery_20261102/attempt4_receipt.json` and `attempt4.log`.
All frozen inputs, source fingerprints and progress-ledger bytes were retained.

This is numerical replay of an existing generator, not independent physical
validation or a passed research gate. Bulk restoration remains gated on a
representative full-generation design and its unchanged original hash. The
stronger-model comparison and complete endpoint remain unqualified.

### Additional-design replay diagnosis and consumer-input hypothesis

D000152 completed acquisition but failed the frozen intermediate-artifact
criterion:41,270,304bytes/SHA256
`d28379d0fd00d712e817cea7db4b846e4a5cc62f452ae184620064209450f7d2`,
versus41,270,392bytes and original SHA256
`ca8594d4b5d13349a9d7e7edfd0218808884eecdd9687f1e593c607c2fdd3fdf`.
The controlled original-CLI replay reproduced the same mismatch. The prior
bulk-restoration protocol remains stopped; no intermediate restoration is claimed.

Known validation/audit metadata serialize byte-identically under the actual
producer call; original raw/state payload identity is still unavailable.
An exercised local diagnostic reconstructed all130 D000152 training rows'
calibration:2,080 offsets and2,080 scales numerically equal the retained originals,
with no differences. Evidence:
`recovery_20261102/exact_metadata_comparison_v2.json` and
`recovery_20261102/d000152_calibration_comparison_20261103.json`.

A separate consumer-recovery experiment is being prepared: preserve unchanged
native noise, normalization, masks and row order, record real new source
provenance, and require BOTH original complete waveform/admissibility hashes.
Calibration equality alone does not satisfy this criterion. No approximate
acceptance, overwritten evidence, source-check bypass, training authorization,
progress-ledger change or research-gate promotion follows.

