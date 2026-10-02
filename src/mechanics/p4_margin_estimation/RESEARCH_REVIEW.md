# P4 Research and Implementation Review

Date: 2026-09-10

## Verdict

**P4 is a credible research prototype, but not yet a convincing early-warning paper. Overall: 5.5/10.**

The engineering foundation is useful: design-grouped evaluation, realization-specific flutter labels, ordered quantiles, and sensor-corruption handling. The principal weakness is evidence for the proposal's central claim: **earlier, better-calibrated warning than classification or conventional identification methods**.

Real-world validation is an acknowledged limitation, not the central criticism in this review. Most recommendations below can be investigated entirely within simulation.

| Dimension | Score / 10 |
|---|---:|
| Research question and usefulness | 7 |
| Data and physics pipeline | 5.5 |
| Model implementation | 6.5 |
| Evaluation rigor | 4 |
| Computational efficiency | 6 |
| Novelty as currently demonstrated | 4.5 |
| Current journal-submission readiness | 4 |

Scores are reviewer judgments, not publication probabilities. Findings concern the source and artifacts inspected on the review date; older reports and current implementation are not assumed to describe identical experiments.

## 1. Scope and evidence

Reviewed the P4 proposal and roadmap, expanded dataset generator and trainer, transient construction, design sampling, eigenanalysis and flutter search, model and loss implementations, baselines, evaluation, and saved dataset/checkpoint/results artifacts.

Repository-root paths are used below. Line references describe the reviewed snapshot and may move after edits.

Verification performed:

- Inspected saved metadata and label arrays; checked design-level split intersections.
- Computed a constant training-label quantile interval and evaluated it on the held-out test labels.
- Loaded `p4_quantile_s0.pt` into the current implementation and recomputed clean-test metrics in batches.
- Measured local CPU batch-one inference latency after warm-up.
- Ran `src/mechanics/p4_margin_estimation/tests/test_evaluate.py` and `test_p4.py`: **58 passed, 1 skipped, 1 failed**.

The failing test was `TestP4Behavioral.test_domain_randomization_reduces_perturbed_mae`, at `test_p4.py:472`: domain randomization improved its synthetic benchmark in only **1 of 3 trials**, against an expectation of at least two. This does not prove augmentation is generally ineffective; it means this test did not support its claimed benefit in the executed run.

No full retraining or complete dataset regeneration was performed. The solver was inspected, not independently certified across the design space. Local CPU was sufficient for this review; Colab was not used.

## 2. What the results establish

### Dataset

- 160 attempted designs; 106 contributed clips; 54 skipped.
- 6,122 clips, with shape `(8, 512)`.
- 77 training, 13 validation, and 16 test designs.
- 4,454 training, 762 validation, and 906 test clips.
- **No design overlap between partitions**, verified from saved metadata.
- Observed margin range approximately `[-0.14996, 0.31968]`.

The effective evidence for cross-design generalization is 16 test designs, not 906 independent designs. Repeated realizations and excitations are valuable, but do not replace independent design coverage.

### Saved aggregate results

| Model | Test MAE |
|---|---:|
| Constant training-median predictor | 0.09175 |
| Velocity-linear baseline | 0.08960 |
| Quantile TCN, average of three seeds | 0.08708 |
| Huber TCN, average of three seeds | 0.08088 |

Relative improvement over the constant baseline is approximately **5.1% for the quantile model** and **11.9% for Huber**. This is preliminary evidence of predictive information, but not yet a strong argument that uncertainty modeling is the decisive contribution.

### Recomputed checkpoint safety metrics

Loaded `p4_quantile_s0.pt` into the current 16-channel, nine-block model. Evaluated all 906 clean test clips with all-valid mask channels.

| Metric | Recomputed value |
|---|---:|
| MAE | 0.08505 |
| q05–q95 coverage | 91.17% |
| Mean interval width | 0.37323 |
| Median predicts safe among truly unsafe clips | 60.1% |
| Lower bound predicts safe among truly unsafe clips | 0% |
| Fraction of all clips with lower bound above zero | 1.77% |
| Number of truly unsafe clips | 376 |

Unsafe means true margin `<= 0`; a safe declaration means the relevant prediction is `> 0`.

The median is not currently a dependable safety decision. The lower bound avoids false-safe decisions in this sample, but almost never certifies safety. That is conservatism, not automatically useful discrimination. Zero observed errors are not a population guarantee.

A constant interval formed from the training-label 5th and 95th percentiles had:

- Lower endpoint: approximately -0.12273.
- Upper endpoint: approximately 0.26583.
- Width: **0.38857**.
- Test coverage: **88.74%**.

Therefore, roughly 90% model coverage is not compelling by itself. The model must demonstrate useful sharpness and better decisions at comparable coverage or risk.

Recomputed checkpoint metrics differ slightly from saved metrics. Checkpoint shape compatibility was verified, but identical historical preprocessing was not established. Report these as separate measurements rather than silently replacing one with the other.

## 3. Proposal versus implementation

### 3.1 Early warning is not yet demonstrated

Evidence: `quantile_head.py:52–66`, `transient.py:459–462`, and `evaluate.py:35–48`, under `src/mechanics/p4_margin_estimation/`.

The model emits one scalar margin per complete clip. Velocity and the true margin remain constant within each generated clip. The evaluation module correctly acknowledges that broadcasting scalar predictions creates degenerate lead times.

Causal convolutions do not make a complete-clip prediction available at the beginning of that clip. Likewise, improved clip regression does not establish earlier warning of an approaching instability.

**Recommendation:** evaluate physically propagated changing-speed trajectories using rolling windows or prefix predictions. Timestamp every prediction when its observations become available. Define actual instability at margin zero, separately from the positive warning threshold.

A per-timestep head is optional: a rolling-window scalar model can suffice. What is essential is a temporally valid experiment. Do not concatenate independent fixed-speed clips and call that a continuous dynamical trajectory.

The expanded experiment also lacks the promised binary-classifier baseline. Until that comparison exists, the proposal's principal superiority claim remains untested.

### 3.2 The warning metric should distinguish warning from instability

`evaluate.py:56–91` currently uses the same threshold for predicted warning and true crossing. At the default 0.15, this measures anticipation of entering a warning region, not necessarily anticipation of flutter at margin zero.

Both can be legitimate metrics, but they answer different questions. Name the event explicitly, retain physical timestamps, and report lead time to the instability event separately from threshold-region detection.

## 4. Data and physics pipeline findings

### 4.1 Adaptive timebase is computed before final mode selection

Evidence: `transient.py:249–290`.

Adaptive `dt` is computed from all candidate modes before selecting the eight retained modes. High-frequency modes that never enter the generated signal can shorten its duration.

Then `generate_p4_dataset.py:214–222,435–443` discards time/`dt` during serialization. The trainer's `_Clip` also has no time field. Consequently, the physics baselines' physical-time support falls back to sample-index units.

Risks:

- Unnecessarily short windows may conceal slow growth or decay near flutter.
- Different physical observation durations are presented without acquisition metadata.
- Acquisition timing depends on simulator eigeninformation unavailable to an ordinary deployed sensor system.

**Recommendation:** select retained modes before setting a representable timebase; persist `dt` and duration. Compare against a fixed acquisition protocol or a time-aware model. Preserve physical units through augmentation and baseline evaluation.

This is a higher-priority experiment than increasing network complexity.

### 4.2 Growth clamping changes the simulated dynamics

Evidence: `transient.py:358–362,417,434`.

Modal growth exponents are clamped to `[-50,20]`. At the upper limit, exponential growth becomes a constant-amplitude oscillation, which is not the original linear-system response.

**[INFERENCE]** If clipping occurs frequently, the network may learn the numerical saturation rule rather than instability dynamics. The dataset does not record saturation incidence, so its prevalence was not established.

**Recommendation:** record clamping incidence and valid observation duration. Use a physically justified observation cutoff, or a validated nonlinear response model if saturation is part of the intended problem. Do not describe clamped trajectories as exact linear transients.

### 4.3 Rejected designs substantially condition the dataset

54 of 160 attempted designs were skipped: **33.75%**. Reported generalization applies to the successful-generation subset, not automatically to the original sampled distribution.

**Recommendation:** persist rejected design parameters, reasons, and search diagnostics. Separate out-of-envelope designs from numerical failures. Plot acceptance against design parameters and report the actual supported domain. Do not silently replace all rejected cases without documenting how the replacement changes the sampling distribution.

### 4.4 Label-defining instability and retained signal need a consistency audit

Flutter search uses a validated spectral abscissa. Transients retain a frequency-restricted, positive-imaginary spectrum. These are not identical selection contracts.

**Recommendation:** for each realization, record the critical eigenvalue and whether its mode is represented in the generated signal. Explicitly distinguish oscillatory flutter from a nonoscillatory instability if either can determine the boundary. Verify label-to-signal consistency rather than assuming that shared solver infrastructure guarantees it.

No claim is made here that a particular saved clip has a mismatched label; this is an identified contract risk requiring measurement.

### 4.5 Provenance is incomplete

Saved metadata omits full sampled and perturbed parameter records, sampling time, and a source/configuration fingerprint. The inspected results file lacks the current safety summary and robustness metrics.

**Recommendation:** save a compact experiment manifest containing design/realization parameters, seeds, solver/filter settings, preprocessing, acquisition timing, dataset fingerprint, model configuration, and checkpoint-selection/calibration protocol. Include per-clip or per-realization flutter boundaries and relevant failure diagnostics.

This is necessary for reproducibility and for attributing improvements to a specific intervention.

## 5. Modeling and evaluation findings

### 5.1 Conformal calibration reuses model-selection data

Evidence: `train.py:260–267` and `train_p4_expanded.py:528–539`.

Validation labels select the checkpoint, then the same validation set calibrates CQR. Scores are also pooled over correlated clips from only 13 designs.

Empirical coverage remains a measurable statistic, but this protocol does not justify an ordinary split-conformal coverage guarantee for unseen designs.

**Recommendation:** separate training, model selection, calibration, and testing by design. Define the intended unit of coverage: a random clip, a new design, or a complete trajectory. Use a group-aware calibration construction appropriate to that unit, with its assumptions stated explicitly.

Report near-flutter coverage, per-design coverage, interval width, and corruption-conditioned coverage. Marginal coverage is not a guarantee of conditional safety or out-of-distribution detection.

### 5.2 Baselines are not sufficiently controlled

**Growth rate:** `baselines.py:290–310` scales growth by the maximum absolute training rate rather than fitting its relationship to margin labels. Its saved MAE of 0.8607 should not be used as strong evidence that learning beats physical identification.

**GRU:** `train_p4_expanded.py:415–418` passes only the training partition; `train_gru()` removes another grouped validation subset. TCN uses the entire training partition and separate validation designs. The comparison therefore differs in training data and validation conditions.

**Loss ablations:** quantile TCN uses early/late/last summaries, whereas Huber uses global mean pooling. The experiment named `median` trains all three quantiles without safety weights. These are not clean isolated loss comparisons.

**Recommendation:** use identical partitions and temporal summaries for loss ablations. Add a calibrated growth-feature baseline and a suitable modal-identification comparator, such as an AR or stochastic-subspace-identification approach where its assumptions fit. Tune all warning thresholds on validation data and compare at matched false-alarm rates.

### 5.3 Robustness needs direct measurement

The current trainer includes a corrupted-data MAE slice, but the inspected saved results lack it. Moreover, robustness of point predictions does not establish robustness of uncertainty.

**Recommendation:** report clean and corrupted MAE, conditional coverage, interval width, and risk–availability tradeoffs across independently varied corruption severities. Include at least one perturbation family absent from training. Avoid claiming uncertainty automatically grows under mismatch: ordinary quantile regression does not ensure that behavior.

### 5.4 Cross-design interpolation is not extrapolation

The present random design split tests generalization within the sampled distribution and fixed boundary-condition family.

**Recommendation:** add a preregistered held-out parameter region or design family. Keep random-split and extrapolation results separate. With few test designs, use paired design-level resampling for model differences, not only confidence intervals for one model's MAE.

### 5.5 Normalization deserves a focused ablation

Current clip generation defaults to per-channel calibration normalization. It removes inter-channel amplitude ratios that could contain mode-shape information. A global-scale option already exists.

**[INFERENCE]** Global normalization could preserve useful spatial information, but could also increase sensitivity to sensor gain mismatch. Compare the existing options under identical acquisition conditions and explicit gain perturbations. Treat this as a testable tradeoff, not an assumed improvement.

## 6. Efficiency review

Measured current quantile model:

- 16 input channels: signals plus masks.
- Nine residual TCN blocks.
- **57,859 parameters**.
- **2,045-sample receptive field** for 512-sample clips.
- Approximately **11.9 ms** per batch-one CPU forward pass after warm-up, one Torch thread, averaged over 30 calls.

This is a microbenchmark with random weights and inputs, not deployment latency. Acquisition duration, preprocessing, scheduling, and state management are excluded.

The model is reasonably small. More important inefficiencies:

1. **Redundant transient reconstruction:** `transient.py:410–427` constructs `w_all` even when the cached sensor-mode projection subsequently bypasses it. Avoid this work in the cached path.
2. **Unjustified receptive-field excess:** seven blocks provide 509 samples; eight provide 1,021. Nine are not necessary merely to cover a 512-sample input. Benchmark an appropriate smaller alternative.
3. **Memory mapping is not end-to-end:** generation retains all clips before writing, loading lacks `mmap_mode`, and tensor conversion repeatedly copies data.
4. **Unbatched evaluation:** coverage and safety paths process complete partitions at once. Use bounded batches.
5. **Unused test tensors:** training constructs test tensors that do not participate in optimization.
6. **Fixed augmentation:** corruption is generated once. On-the-fly augmentation could increase nuisance diversity, but should be evaluated against its runtime cost and reproducibility requirements.

Priority: remove redundant reconstruction and bound evaluation memory first. Profile eigensolves before attempting modal reduction; any reduced model must preserve near-boundary stability and critical modes.

## 7. Novelty assessment and prior work

Continuous flutter inference, subcritical response identification, probabilistic flutter prediction, temporal neural networks, and conformal quantile calibration already have substantial precedent. Their combination is not automatically a new method.

Relevant sources:

1. [Bayesian analysis of the flutter margin method in aeroelasticity](https://www.sciencedirect.com/science/article/abs/pii/S0022460X1630325X).
2. [Probabilistic Prediction of Coalescence Flutter Using Measurements: Application to the Flutter Margin Method](https://arxiv.org/abs/2210.15667). The abstract describes uncertain modal parameters, cross-airspeed dependence, subcritical observations, and improved probabilistic flutter-speed inference.
3. [Flutter Boundary Prediction Based on Feature Extraction and Bayesian Optimization Transformer–Long Short-Term Memory](https://arc.aiaa.org/doi/full/10.2514/1.C038983). A particularly relevant 2026 comparator: its indexed description mentions deep feature extraction from measured signals. Full text was inaccessible during this review, so architectural equivalence is not asserted.
4. [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222). CQR is an established method; its use alone is not methodological novelty.

This was a targeted prior-art check, not an exhaustive systematic review. P4's normalized velocity-distance label differs from classical flutter-margin constructions; explicitly define that distinction and avoid implying they are interchangeable.

### Strongest defensible contribution

> Cross-design estimation of a physically defined stability margin from short multichannel observations, with useful calibrated uncertainty and warning performance under controlled mismatch.

This could support an application-focused computational mechanics paper if the scientific result is clear. It does not require inventing a neural architecture. Novelty should come from the question answered and the demonstrated advantage, not from attaching more modules.

## 8. Recommended research directions

These are alternatives with different claims, not a checklist requiring every direction. **Choose one central contribution and use the others only when needed to resolve an observed limitation.** The best near-term route is useful uncertainty for sequential warning, supported by an identifiability/timebase experiment.

### Direction A — Risk-controlled warning with useful availability

**Question:** can a response-based margin estimator provide earlier warning while retaining a useful ability to certify safe operation?

This directly addresses the observed result: zero lower-bound false-safe errors, but safe declarations on only 1.77% of clips.

**Experiment:**

- Generate physically propagated trajectories with specified speed histories, plus stable trajectories that never cross instability.
- Use rolling windows with real acquisition timestamps and a separate warning rule.
- Compare quantile regression, point regression, a binary classifier using the same backbone, and calibrated identification baselines.
- Choose thresholds on validation data, calibrate on separate design groups, and evaluate on unseen designs.
- Measure lead time at matched false-alarm rate, missed events, safe-declaration availability, and error among safe declarations.
- Include acquisition delay and the normalization warm-up period in warning timing.

**Potential contribution:** an experimentally demonstrated risk–availability–lead-time tradeoff, rather than a nominal coverage statistic.

**What would make it convincing:** a consistent paired-design advantage over classification and identification at the same operating risk, including a held-out mismatch condition.

**Failure/pivot condition:** if the interval model only avoids errors by abstaining almost everywhere, do not present conservatism as superior monitoring. Report the limit or narrow the operating domain.

**Novelty caution:** standard CQR on rolling windows is not itself new, and temporal/group dependence prevents casually inheriting an iid guarantee. A new theoretical claim would require a correctly specified calibration method and proof; a strong empirical application need not claim a new theorem.

### Direction B — Identify which stability quantity is transferable

**Question:** is normalized velocity distance actually identifiable from a short, normalized response across uncertain designs, or does it depend too strongly on the design prior?

This is scientifically deeper than asking whether a larger network lowers MAE. Short windows mainly expose finite-time modal dynamics, while normalized distance to a design-specific flutter boundary is a different quantity.

**Experiment:**

- Repair and preserve physical sampling time first.
- Compare the current target with a dimensionless intrinsic target such as `-alpha / omega_ref`, where `alpha` is the relevant spectral abscissa and `omega_ref` has an explicit, consistent definition.
- Compare signal-only input with signal plus measured acquisition time; optionally test known operating speed as a separately declared input condition.
- Study error versus observation duration, modal separation, damping, and held-out design regions.
- Search for pairs of designs whose noisy observable responses are difficult to distinguish but whose velocity margins differ materially.

Such pairs would be evidence of finite-window ambiguity, not by themselves a proof of formal non-identifiability.

**Potential contribution:** a demonstrated link between target choice, observability, and cross-design transfer; a positive result could explain when response-based velocity-margin inference is possible and when it is prior-driven.

**What would make it convincing:** the intrinsic target improves held-out-design performance under controlled timing and observation budgets, and the explanation survives nuisance perturbations.

**Failure/pivot condition:** if the current target transfers adequately after timing repairs, keep it. The roadmap already treats intrinsic targets as a gated upgrade; do not turn this into an architecture and target sweep without a specific failure hypothesis.

### Direction C — Cross-simulator mismatch without waiting for real data

**Question:** does uncertainty remain useful when the test response violates the training simulator's assumptions?

Sensor noise and gain perturbations are useful but do not test all forms of structural or dynamical model error.

**Experiment:** train on the current modal generator and test on a deliberately independent response construction where feasible. Candidate differences include a validated higher-resolution model, altered damping structure, different excitation spectra, or direct time integration under a justified forcing model. Clearly distinguish numerical verification from genuine model-form mismatch.

Vary one mismatch family at a time and include a family absent from training. Avoid treating another implementation of exactly the same equations as independent physical truth.

**Potential contribution:** a controlled account of which mismatches break margin inference and whether intervals detect that loss of reliability.

**What would make it convincing:** maintained or transparently degrading decision performance, with intervals that become appropriately cautious without collapsing availability everywhere.

**Failure/pivot condition:** if uncertainty stays narrow while error grows, add an explicit validity-domain or abstention mechanism only after demonstrating the failure; do not claim existing quantiles represent epistemic uncertainty by construction.

### Direction D — Observation-budget and sensor-budget limits

**Question:** how much time and how many sensors are actually needed for reliable near-flutter estimation?

This is a useful secondary direction because P4 is framed as online monitoring, yet current evaluation fixes clip length and sensor count.

**Experiment:** compare a small, motivated set of physical observation durations and sensor subsets under fixed false-safe/false-alarm constraints. Include missing channels and weak excitation of the critical mode. Express duration in seconds and, where useful, modal cycles—not only sample count.

**Potential contribution:** an interpretable operating envelope linking monitoring latency, sensor burden, and stability-estimation reliability.

**Novelty caution:** a generic sensor-count ablation is weak. The stronger result is a physically explained limit or a demonstrated operating tradeoff that existing baselines cannot match.

## 9. Suggested research sequence

1. **Repair the evidence contract:** preserve timebase and provenance; quantify rejection and clipping; verify critical-mode consistency.
2. **Establish a fair static benchmark:** identical splits and heads where appropriate; calibrated growth and identification baselines; paired design-level comparisons.
3. **Run the short-window/target experiment:** determine whether timing and target ambiguity explain the weak quantile advantage before changing architecture.
4. **Choose the main claim:** preferably useful uncertainty for sequential warning, or target transferability if that produces the clearer scientific result.
5. **Test the claim on physical trajectories and held-out mismatch:** reserve independent calibration and test groups; report uncertainty utility as well as accuracy.
6. **Write to the demonstrated result:** a limited but defensible operating envelope is stronger than broad claims about digital twins or graceful degradation unsupported by experiments.

A possible paper framing is **“Cross-design aeroelastic stability-margin estimation from short transient observations: calibration, identifiability, and warning utility.”** This is a framing suggestion, not a claim of established novelty or a recommendation to include all three topics equally.

Do not prioritize a Transformer, an ensemble stack, or a larger architecture search. A methodological ML paper would require a genuinely new estimator, calibration method, or theoretical result beyond standard components. An application-focused mechanics paper can instead succeed through a rigorous new finding, strong physical analysis, and decisive comparisons.

## 10. Publication outlook

**Current state:** not submission-ready for the full early-warning claim. The evidence supports modest clip-level regression; uncertainty utility and temporal superiority remain unproven.

**Plausible route:** an application-focused computational mechanics or structural monitoring paper, provided that a clear cross-design, warning, or mismatch result survives the controlled comparisons above. This assessment does not promise acceptance at any venue.

**Bottom line:** keep P4, but strengthen the experiment before the model. The best novelty opportunity is not a different neural backbone; it is a defensible answer to when short transient observations support transferable, useful, uncertainty-aware stability decisions.

## 11. Contribution and pivot reassessment — 2026-10-01

This addendum updates the contribution recommendation in Sections 7–10 after the spatial-feature and localized-consistency studies. It does not revise their historical experiment results. Research only: no new model fitting, reserved-split scoring, production default, or GRU decision.

### Decision

**No-go for claiming that simulation training, a recurrent network, continuous flutter prediction, noise augmentation, or conformal intervals constitute P4's distinctive contribution. Conditional go for a narrower scientific question: what reliable margin/stability decisions can short observations support across unknown structural designs, without target-specific flutter calibration?**

The candidate practical advantage must be a demonstrated reduction in observation or calibration burden at comparable decision risk—not simply a more restricted input contract. The reviewed sources do not establish that P4's exact combined contract has already been met, but that is not proof of novelty. [INFERENCE] A useful application-focused contribution remains possible; a unique advantage is not yet demonstrated.

### Primary-source comparison

Evidence levels below distinguish full manuscripts from publisher-deposited abstracts. Web-search summaries were used to locate sources, not as evidence for their methods or results.

| Prior work | Verified evidence | Consequence for P4 |
| --- | --- | --- |
| Crowther & Cooper, *Flight test flutter prediction using neural networks* (2001), [local full manuscript](../../../p4_journal/papers/crowther2001.pdf), [DOI](https://doi.org/10.1243/0954410011531736) | MLP predicts damping three speed steps ahead from five damping/frequency observations and current speed. Eight mass-perturbed aircraft models supply training data; a ninth is held out. Noise-matched training recovers approximately noise-free accuracy. Full-text evidence; details in Section 13. | Already includes simulation training, held-out model variation and noise-aware prediction. Uses target operating-point history, not one short waveform. Its damping forecasts and approximate flutter-speed error cannot rank against P4 margin MAE. |
| Zheng et al., *Research on Feature Extracted Method for Flutter Test Based on EMD and CNN* (2021), [publisher-deposited abstract](https://api.crossref.org/works/10.1155/2021/6620368) | Wind-tunnel response signals, normalization, EMD/CNN features and flutter/no-flutter classification; authors report online low-complexity prediction. Abstract-level evidence. | Measured-signal learning and an online classifier are not new. Its reported test accuracy does not establish structural transfer or early-warning lead time. |
| Wang & Ma, *Application of Deep Learning Models to Predict Panel Flutter in Aerospace Structures* (2024), [local full manuscript](../../../p4_journal/papers/aerospace-11-00677.pdf), [DOI](https://doi.org/10.3390/aerospace11080677) | DNN/LSTM/LSTM-NN on simplified simply supported supersonic panel calculations. Reports LSTM flutter-Mach MSE 4.8e-5, R² 0.99982 and average relative error 0.2182%; classification inputs include decay rate used to define the labels. Full-text evidence; details in Section 13. | Parameter-informed simulation surrogate, not demonstrated noisy waveform-only inference. Explicit aspect ratio/load/damping inputs, different target/metrics and no specified design-disjoint split prevent direct ranking against P4. Short physical record duration and calibrated uncertainty are not established. |
| Wang & Zhou, *Flutter Boundary Prediction Based on Feature Extraction and Bayesian Optimization Transformer–Long Short-Term Memory* (2026), [publisher-deposited abstract](https://api.crossref.org/works/10.2514/1.C038983), [DOI](https://doi.org/10.2514/1.C038983) | Acceleration signals, PSD and Transformer–LSTM; wind-tunnel training; reported mean prediction errors 3.73% on unseen wind-tunnel working conditions and 4.14% on one flight-test sortie; error below 10% at about 70% of critical speed. Abstract-level evidence; full manuscript unavailable. | A serious comparator with physical validation already reports response-based subcritical prediction. “Neural early flutter prediction” cannot distinguish P4. The abstract says unseen working conditions, not verified unseen structural designs; observation duration and target-specific calibration are unresolved. |
| Shi et al., *Flutter Boundary Prediction Method Based on Long Short-Term Memory Networks* (2026), [publisher-deposited abstract](https://api.crossref.org/works/10.2514/1.C038429) | Turbulent natural-excitation wind-tunnel signals; CWT/LSTM; condition classification and wind-speed regression within a condition; weighted flutter-state estimation. Abstract-level evidence. | Passive excitation and robustness of signal-based flutter prediction are not sufficient distinctions either. Exact cross-design and warning-risk protocols were not verified. |
| Ueda, Iio & Ikeda, *Flutter Prediction Using Continuous Wavelet Transform* (2009), [local full manuscript](../../../p4_journal/papers/ueda2009_wavelet.pdf), [DOI](https://doi.org/10.2322/tjsass.51.275) | Gabor CWT; reciprocal absolute-coefficient sum at a model-calculated flutter-frequency band; lower-envelope selection and linear extrapolation across subcritical speeds. Three wind-tunnel runs on one wing; approximately 0%, -10%, -4% flutter-speed errors. Full-text evidence; details in Section 15. | Useful lessons: local modal observability, excitation-sensitive amplitude information and failure to visually resolve coalescence. Raw amplitude index is not directly transferable to P4's normalized random-initial-condition transients; temporal/band-local information is a conditional candidate, not a demonstrated gain. |
| Chajjed et al., *Probabilistic Prediction of Coalescence Flutter Using Measurements* (2022 preprint), [full manuscript](https://arxiv.org/html/2210.15667v1) | Bayesian free-decay likelihood; modal priors generated from random structural parameters; dependence across airspeeds; numerical example uses three subcritical speeds and 10% structural parameter COV. | Physics-derived probabilistic priors and random structural models already exist. Its multi-speed target observations differ from one short clip, but the broader claim “learn structural uncertainty from simulation” is occupied. P4 must compete with a physics-prior baseline, not only ridge or a binary model. |
| Méndez, *Physics-constrained identification of near-flutter aeroelastic damping from finite records* (September 2026 preprint), [full manuscript](https://arxiv.org/html/2609.25440v1) | A verified critical-pole trajectory for one modified Isogai configuration constrains frequency/damping; predicted precision gain 3.885, median synthetic Monte Carlo RMSE gain 3.915; at one offset and 30 dB SNR, minimum tested record for >95% sign accuracy drops from 15 to 5 cycles. Trajectory uncertainty and misspecification are examined. | Physics-assisted short-record stability identification and observation-time savings are already directly studied. Its independently established target trajectory is extra information. Transferring useful physical calibration across unknown designs, without that target trajectory, is the remaining hypothesis—not an established P4 achievement. This is a preprint, and the finite-record statistical validation is synthetic, not cross-design flight validation. |
| Zastrow et al., *Data-driven Model Reduction via Operator Inference for Coupled Aeroelastic Flutter* (2022), [author-hosted full manuscript](https://kiwi.oden.utexas.edu/papers/data-driven-model-reduction-aeroelastic-flutter-Zastrow-Willcox.pdf) | Full-state CFD snapshots and structural modal bases; parametric aerodynamic prediction at unseen Mach and coupled flutter temporal extrapolation; reported large speedups over FUN3D. | Fast simulation surrogates and generalization in operating parameters are established. This is not waveform-only inverse inference across unknown structures. Comparing P4 milliseconds with a CFD solve would answer an easier, different problem. |

Additional cautions:

- [Bennett's NASA flutter-margin study](https://ntrs.nasa.gov/citations/19830003800) already warns that reliable classical prediction needs accurate frequencies and damping of both interacting modes; a frequency-only simplification was nonconservative in one case. This does not imply frequency is useless when combined with a validated physical relation.
- [Sodja et al. (2018)](https://research.tudelft.nl/en/publications/experimental-characterisation-of-flutter-and-divergence-of-2d-win/) experimentally identified flutter/divergence of the original wing section while the modified test system remained stable. Safe subcritical identification itself is not a new objective; that method changes the experimental system.
- [Conformalized quantile regression](https://arxiv.org/abs/1905.03222) and [conformal risk control](https://arxiv.org/abs/2208.02814) are established tools. Adding them is not novelty. Grouped/rolling-window dependence and unmodeled distribution shift require an appropriate calibration unit and assumptions; nominal clip coverage is not an operational warning guarantee.
- [Simulation-based inference](https://arxiv.org/abs/1911.01429) is an established research field. An amortized posterior or learned physical prior would be a candidate implementation, not automatically a new method.

The closest 2026 neural paper's percentage boundary errors cannot be ranked against P4's absolute normalized-margin MAE. For a known operating speed, defining relative critical-speed error \(e_U=(\widehat U_{\rm crit}-U_{\rm crit})/U_{\rm crit}\) gives \(e_m=(1-m)e_U/(1+e_U)\). That conversion varies with operating point and signed error; converting a reported mean percentage into mean margin MAE is not justified.

### Which distinctions remain, and whether they matter

| Candidate distinction | Current evidence | Assessment |
| --- | --- | --- |
| Transfer to previously unseen designs without target labels/calibration | Outer design-grouped experiments on 77 original TRAIN designs; no final reserved-design result | A plausible research axis, not verified uniqueness. Random held-out designs from the same population establish interpolation/generalization within that population, not arbitrary structural extrapolation. |
| Less observation time or no speed sweep | Input contract uses measured response/timebase rather than target flutter calibration | A restriction, not yet a performance advantage. Need matched-risk comparisons in physical seconds/cycles, including causal warm-up and pre-clipping usable aperture. |
| Better warning at the same false-warning burden | Existing fold-calibrated clip policies expose a near-boundary tradeoff, not uniform dominance | Potentially valuable if demonstrated. A separate warning head or asymmetric loss alone is not the contribution. |
| Useful uncertainty under mismatch | Current improved-feature experiments have no demonstrated operational risk/availability result under new structural/model-form mismatch | A difficult, relevant candidate. Sensor noise alone does not test hidden structural or dynamical model error. Standard intervals do not guarantee arbitrary OOD safety. |
| Lower computation cost | Warm raw-to-prediction timing measured; timing varies across runs | Engineering evidence, not a comparative scientific advantage yet. Acquisition may dominate computation. Benchmark a same-information estimator, not an online PDE solve that the task never required. |

[INFERENCE] The most useful potential distinction is **reliable cross-design decisions with less target-specific information**, supported by a physical account of when the short observation is insufficient. Industrial impact must be measured as fewer calibration measurements, shorter observation, fewer sensors, or improved decision risk/availability. A restriction without adequate accuracy or availability has no demonstrated benefit.

### Physical reason to change the question before another loss sweep

For a simple destabilizing crossing on a fixed critical branch,

\[
\alpha(U)\approx-\kappa m,\qquad
\kappa=U_{\rm crit}\left.\frac{d\alpha}{dU}\right|_{U_{\rm crit}}.
\]

Estimating current growth is not the same as estimating design-specific normalized speed distance. Other observable modes/shapes may identify the conversion; a universal one-pole conversion cannot be assumed. This local relation is not a global model and need not apply unchanged through branch switches or divergence.

A read-only replay of existing TRAIN metadata matched cached source rows, targets and design IDs exactly. Among 334 clips on 70 designs with \(0<|m|<.01\), finite-distance \(-\alpha_{\rm label}/m\) has p10/median/p90 **272.719/465.365/1033.489 s⁻¹**, a p90/p10 ratio **3.790**. These are finite-distance ratios, not independently measured local derivatives. Label-driving modes are retained in 334/334; retention does not prove excitation or sensor visibility. Median \(|\alpha_{\rm label}|T\) over the full 512-sample aperture is **1.012**, with p10/p90 **0.235/2.882**; median full aperture is **0.499 s**. These values do not describe every usable pre-cap prefix.

Sources: `p4_dataset_saturation/metadata_arrays.npz`, `p4_groups/two_mode_spatial_20260930/features.npz`, and the label-path consistency contract in `transient.py`. No reserved designs or new fitting were used.

The journal already records that spatial features improve overall accuracy while harming boundary behavior, and localized consistency reduces repeated-noise variance without recovering one107/blend50 boundary accuracy. [INFERENCE] Neither result identifies a single root cause. Measurement error, design-dependent calibration, representation limits and finite fitting capacity must be separated.

**Next diagnostic, not a new architecture prescription:** compare the measured features with a TRAIN-only oracle representation consisting of corresponding true modal dynamics/shapes. Never include \(m\), \(U_{\rm crit}\), \(-\alpha/m\), or a target-specific speed derivative as predictive inputs. Match modal content, design folds and downstream capacity; retain both current tabular and later raw-waveform/GRU routes.

- Strong true-modal mapping but weak measured mapping → prioritize identification/measurement uncertainty.
- Persistently poor true-modal mapping → investigate design-dependent calibration, inadequate modal content or fitting capacity.
- Nearby observable responses with materially different margins → test finite-window ambiguity in this actual design family, with noise-aware response likelihoods; nearest feature neighbors alone are not an impossibility proof.
- Reweight the structural prior without changing physical labels → test whether confident margin estimates are driven by population correlations.

An unsuccessful oracle regressor is not a proof of non-identifiability; successful oracle prediction is not an achievable inference result.

### Closely related pivots, ordered by evidence

**A. Recommended central objective: information-limited cross-design stability decisions.**

Ask: *Can a simulation-trained estimator give useful margin/stability decisions on unknown designs without a target flutter-calibration campaign, and determine when the available observation is insufficient?* Preserve the physical margin estimate; assess safe/warn/undetermined decisions, observation budget and structural-prior sensitivity. No promise that abstention solves the problem: measure how much useful availability remains.

The potential mechanics contribution is a validated operating envelope and a physical explanation of cross-design inference limits, plus a comparative advantage where inference works. Selective prediction, modal fitting and risk calibration are existing components. Simply adding an uncertainty gate is not enough.

**B. If margin calibration is the limiting factor: transfer physical calibration, not a universal damping formula.**

Learn the population relationship between joint modal information and margin, optionally using physical auxiliary supervision during training. Propagate ambiguity over the latent structure or pole trajectory instead of imposing one known trajectory on every design. Compare against a training-population physics-prior/library likelihood baseline using the same measurements. This preserves the single-observation contract. It is a mechanism to test under Objective A, not a verified novel architecture or a required literal \(\kappa\) head.

**C. If one observation genuinely lacks enough information: minimal additional acquisition.**

Pivot to the cheapest added observation that enables a reliable decision: a longer causal window or an additional observable channel first; controlled re-excitation at the same operating point if justified. Any force input or target history is a declared contract change. Compare adaptive acquisition with fixed-budget modal identification and learned models at equal risk and observation cost. Repeated noise draws, concatenated unrelated clips and unmeasured future speeds are not additional physical information.

This sacrifices part of the original single-short-observation claim, but retains the nearby engineering objective of reducing monitoring/calibration burden. Generic active sensing is not novel; the contribution would have to be the aeroelastic decision benefit and physically explained acquisition policy. A full target-specific speed sweep is not the preferred first pivot because it abandons the most relevant candidate distinction.

**Model-form mismatch is the credibility test for the chosen objective, not a separate feature checklist.** Use physically justified changes to damping/excitation or an independently validated response model with consistent flutter truth; distinguish numerical refinement from changed physics. No universal guarantee under arbitrary unseen mismatch is proposed.

### Concrete go/no-go experiment sequence

1. **Keep discovery separate from final claims.** Use existing TRAIN designs for the oracle/calibration diagnostic. Freeze the question, inputs, observation budgets and policy before reserved-split evaluation. Do not generate more noise copies and call them new structural evidence.
2. **Build the decisive same-information comparison.** Current one107/full174/blend50; calibrated growth/modal identification; a training-population physics-prior baseline; learned margin plus calibrated policy; and a direct binary decision model. A later GRU can replace the learned representation without changing the comparison contract. All models receive the same causal samples, sensors and timebase. Extra-history/known-structure/known-trajectory methods belong in a separately labeled information track, not a straw-man single-clip implementation of a multi-speed method.
3. **Test the actual edge.** Hold out complete designs and a prespecified design region or physical mismatch family, in addition to random design folds. Compare risk–availability or risk–observation frontiers, not MAE alone. Report stable/unstable boundary slices and farther-stable costs. Choose the risk target and a materially useful availability/time benefit before confirmation; deployment costs/noise specifications are currently undeclared.
4. **Only claim early warning with propagated trajectories.** Include stable never-crossing trajectories, time-varying crossing cases, acquisition/normalization delay, event-level missed warnings and false alarms. Use an explicit definition of the stability event in a nonstationary system. Calibrate at a design/trajectory level appropriate to the risk definition; clip rates are not event guarantees.
5. **Stop or pivot on a discriminating failure.** If true-modal features work but noisy observations do not, pursue identification. If margin depends strongly on hidden calibration/prior even with good modal observations, move to Objective A's uncertainty/validity account and, if needed, Pivot C. If the best same-information baseline matches the learned method's frontier, do not claim a unique learned advantage. A reproducible new physical limit may support a different contribution, but a generic negative ablation does not.

**Recommended next action:** TRAIN-only true-modal versus measured-feature calibration diagnostic, followed by the same-information physics-prior comparison—not another unchanged consistency/local-weighting sweep. This replaces the earlier unqualified “next loss” recommendation at the research-strategy level; no experiment or implementation change is authorized by this note.

### Access limits and claim discipline

Full text was initially read for Chajjed, Méndez and Zastrow; user-supplied Wang–Ma and Crowther–Cooper PDFs have now also been reviewed (Section 13). Current core six-paper set: four full texts, two abstract-only AIAA papers (Wang–Zhou and Shi et al., both 2026). Broader eight-paper table: five full texts, three abstract-only entries including Zheng. These are shortlist counts, not an exhaustive literature census. Prior publisher/full-text retrieval did not supply the two AIAA manuscripts. No exhaustive priority or patent search was performed.

Before a “first,” “unique,” or superior-to-that-paper claim, obtain the closest neural paper's full protocol or a reproducible implementation. That missing information does not block the proposed same-information benchmark or physical diagnostic. No new experiment has established the proposed advantage, no default changed, and GRU remains live.

## 12. Focused comparison of the two accessible inference papers

User scope clarification: extract objectives, results/comparability and useful method components. A complete reproducible baseline specification is **not** requested. No reproduction or implementation is selected by this comparison. Prior work does not preclude a contribution through demonstrably better performance on the same task.

### Chajjed et al.

Source: [full manuscript, Introduction, Sections 3–5 and Table 2](https://arxiv.org/html/2210.15667v1).

- **Objective:** infer a probability distribution over critical flutter speed from noisy subcritical observations. This is related to P4's normalized speed-distance target, but its intermediate Zimmerman–Weissenburger flutter-margin function is not our \(m=(U_{\rm crit}-U)/U_{\rm crit}\).
- **Method:** randomized structural parameters generate a joint prior over modal frequencies and decay rates, including dependence within and across airspeeds. MCMC combines that prior with noisy free-decay observations; uncertainty propagates through flutter-margin fitting to critical speed. The numerical example uses three target airspeeds.
- **Result:** Table 2 reports critical-speed COV 4.5903, 3.7115 and 2.5625 for flat, independent and joint priors: about 44% lower COV for joint versus flat. This is posterior uncertainty spread, not a 44% improvement in prediction error, empirical coverage or safety.
- **Comparability:** no direct ranking against P4 margin MAE or warning rates. Different physical systems, target information and metrics; no matched unseen-design P4 evaluation. Critical-speed predictions could be expressed as normalized margins if operating speed were available, but the reported aggregate COV does not supply margin MAE.
- **Useful component [INFERENCE]:** learn a joint physical distribution from our simulated design population rather than treating each modal feature as independent; propagate uncertain modal observations through margin inference. This may help design-dependent calibration, but does not establish an improvement. Cross-airspeed coupling cannot be imported unchanged without changing the single-observation contract.

### Méndez

Source: [full manuscript, Sections 4–6, Tables 2–3](https://arxiv.org/html/2609.25440v1).

- **Objective:** identify near-flutter growth/damping and its sign from finite records, not estimate normalized distance to critical speed across unknown structures.
- **Method:** compare free pole estimation with estimation constrained to an independently established frequency–growth trajectory. Both use profiled least squares, analytically fitting quadrature amplitudes at each trial pole. Better-resolved phase/frequency supplies information about weak growth. CFD establishes the trajectory for one configuration; reported statistical gains are tested on synthetic noisy records.
- **Results:** median synthetic damping-RMSE ratio free/constrained 3.915, approximately 3.66–4.08 across tested cells. At one operating offset and 30 dB amplitude SNR, minimum tested record for over 95% correct-sign probability drops from 15 to 5 critical-mode cycles. This is not a universal five-cycle guarantee. Trajectory uncertainty weakens closest-to-flutter lower-tail performance.
- **Comparability:** damping RMSE is not margin MAE. Sign accuracy and record length overlap our decision/observation objectives, but require matched distance to flutter, noise definition, modal cycles, excitation and available trajectory information before a numerical ranking. Our 1%/5% acquisition-noise settings are not automatically their 30 dB amplitude SNR.
- **Useful component [INFERENCE]:** investigate joint frequency–growth information and uncertainty-aware physical constraints in identification/calibration. Our representation already includes frequency and damping, so adding frequency columns alone is not the lesson. A transferable relation would need to be learned/conditioned on the design population, not copied from their configuration. Their misspecification study supports soft/probabilistic constraints: a wrong hard trajectory can bias growth and reverse its sign.

**Assessment:** Chajjed is more relevant to uncertainty in growth-to-margin calibration; Méndez to short-record growth identification. Neither published headline result directly ranks against our current P4 scores. Both supply useful mechanisms, not proof that those mechanisms will improve P4. No new training, baseline reproduction, model/default change or approved pivot.

## 13. User-supplied panel and neural flight-test manuscripts — 2026-10-01

Copied, not moved, from `/home/omid5/Downloads/` to `p4_journal/papers/`. Source/copy SHA-256 pairs match. Full-text comparison remains scoped to objective, method, reported results and relevance to P4; no reproduction is requested or selected.

### Wang–Ma 2024: parameter-informed panel surrogate

Sources: [local PDF](../../../p4_journal/papers/aerospace-11-00677.pdf), [DOI](https://doi.org/10.3390/aerospace11080677). Relevant sections: 2, 4.2–4.6 and 5; Tables 3–4.

- **Objective:** classify convergence/exact stability boundary/divergence and regress flutter Mach number, while reducing repeated numerical calculation cost. Same panel-flutter topic, but not our normalized current margin from measured signals.
- **Physical/data model:** simply supported panel, classical simplified aeroelastic equations with first-order piston aerodynamics. Paper reports 707,472,000 sampled points and 197,621 boundary points for flutter-speed prediction. These are reported sample counts, not independent physical structures or experimental observations. Section 4.5 explicitly says high-fidelity CFD–FEM comparison was not conducted.
- **Information:** classification features include aspect ratio, Mach/air density and decay rate; Section 2 defines the class by the sign of the calculated real part/decay rate. Thus this is not evidence for identifying that sign from a noisy waveform. For flutter-Mach regression, Section 4.6 specifically lists aspect ratio, decay rate, longitudinal load, frequency and total damping. Different tasks have different feature sets; do not apply the abstract's generic feature list to every result.
- **Observation/transfer:** describes a 70/25/5 percent data split, not a specified complete-design holdout. LSTM sequence assembly, physical observation seconds/cycles and noisy sensor identification are not established by the described protocol. Do not infer long records merely from the use of LSTM; equally, do not claim demonstrated short-record waveform inference. Section 2 says flutter-regression samples were obtained at the stability boundary, not a demonstrated single subcritical measurement protocol.
- **Classification results (Table 3):** DNN/LSTM/LSTM-NN overall model accuracies 99.35/99.53/99.56%; boundary-class prediction accuracies 98.33/99.17/99.44%. Other state-class accuracies exceed 99%. Separate model/training accuracy from reported class prediction accuracy. Exact-neutral-class accuracy is not near-boundary warning risk, useful availability or probability calibration.
- **Regression results (Table 4/Section 4.6):** DNN/LSTM/LSTM-NN MSE 7.94e-5/4.8e-5/6.12e-5; R² 0.99970/0.99982/0.99978. Paper reports average relative error as low as 0.2182% in the comparison where LSTM performs best. Do not relabel this as P4 normalized-margin MAE or infer a matched-error advantage. The regression discussion reuses the classification dataset/test-count description despite the earlier separate boundary subset; exact evaluation-row provenance is not resolved by that prose.
- **Efficiency:** reports about 5.5 hours of numerical calculation versus under one minute of trained prediction for the stated dataset. This is a simulation-surrogate comparison, not our same-information raw-observation-to-margin inference timing.
- **What to borrow [INFERENCE]:** evaluate stability decisions separately from continuous regression, including boundary-specific outcomes rather than overall accuracy. Feature scaling and class-imbalance handling are ordinary useful practices already represented in P4, not reasons to copy its architecture. Its explicit physical inputs can inform diagnostics of calibration, but importing unmeasured aspect ratio/load/damping would change our contract. No specific new model component is justified by its scores.

### Crowther–Cooper 2001: model-trained operating-point extrapolation

Sources: [local PDF](../../../p4_journal/papers/crowther2001.pdf), [DOI](https://doi.org/10.1243/0954410011531736). Relevant sections: 2–3, 5.2–5.5, 6–7.

- **Objective:** forecast modal damping at a future flight-test speed to support safe envelope expansion, not current normalized margin from an isolated transient.
- **Method/information:** MLP input contains five damping values and five frequency values (current plus four previous target test points), plus current speed. With 15-knot spacing it predicts three steps, or 45 knots, ahead. The underlying model is a four-engined transport aircraft, not a plate. The polynomial control uses damping/speed only; the authors explicitly acknowledge that adding frequency to statistical regression could improve it. Therefore the best neural-versus-polynomial comparison is not equal-information evidence for an architectural advantage.
- **Transfer correction:** ±10% random perturbations of mass-matrix elements produce nine model datasets; eight supply training data and one is an independent held-out case. This already demonstrates some transfer to unseen model variation. Do not claim that holding out structural variation itself is absent from earlier neural work. It does not establish broad cross-family transfer or inference without target speed history.
- **Identification assumptions:** works with model-derived modal quantities, not raw noisy sensor-response identification. Includes mis-sorted eigenvalues in training; chosen validation cases have no sorting errors. Authors report reduced neural accuracy for validation behavior not well represented in training. Modal identity remains a relevant failure mode.
- **Results:** noise-free quadratic extrapolation overpredicts first flutter-mode speed by 25 knots, 6.4% of true flutter speed; adding frequency to the neural predictor improves its result, reported within approximately 5% of true flutter speed. Damping-only neural prediction is only slightly better than the control. No aggregate P4-equivalent MAE or confidence bounds are supplied for these headline comparisons.
- **Noise:** noisy modal features use random perturbations of amplitude ±5% of the clipping ranges (damping −0.04 to 0.10; frequency 0 to 12 Hz). This is not our 5%-of-calibration-RMS waveform noise. A clean-trained network queried with noisy features performs worse than the noisy polynomial control; noisy training substantially recovers accuracy, with mode-12 failures still noted. Do not describe it as unconditional neural robustness.
- **Validation correction:** despite the flight-test title/application, the presented results are numerical model cases. Conclusion says application to real flight-test data is future work.
- **Comparability:** future damping and approximate flutter-speed errors from a target airspeed history cannot directly rank against single-clip cross-design margin MAE. The demonstration is a forecast across operating points, not evidence for event-level early-warning performance on propagated trajectories.
- **What to borrow [INFERENCE]:** noise-aware training with realistic identification errors and richer joint modal information, not merely larger networks. Both principles are already represented in P4. Mode-identity failures deserve diagnostics rather than an assumed cure through recurrent architecture. Genuine same-structure operating-point history could motivate a later forecasting/GRU task, but changes the current single-observation contract and is not approved here.

**Assessment:** neither newly available paper supplies an apples-to-apples numerical ranking against P4. They strengthen the precedent for simulated modal learning, structural variation and noise-aware training, while clarifying differences in target information. A demonstrated useful performance/observation/calibration tradeoff can still be a contribution on the same task. Remaining priority manuscripts are Wang–Zhou (10.2514/1.C038983) and Shi et al. (10.2514/1.C038429). No experiment, reproduction, default change or objective pivot; GRU remains live.

## 14. Research lens correction: mechanisms and applicability first

User clarification: do not center each paper's similarity to P4 or whether headline scores rank directly. The primary question is **what mechanism or physical insight can help P4, under what conditions, and whether it is already covered or inapplicable**. Sections 12–13 retain factual comparison context, but comparability is not a gate for learning from a paper. The assessments below are [INFERENCE], not demonstrated P4 improvements.

### Most useful transferable ideas

1. **Chajjed: preserve joint physical information and propagate ambiguity.** Its structural-model prior preserves dependencies among modal quantities, rather than replacing all uncertainty with independent error bars. For P4, multiple plausible structures/modal interpretations could lead to different margins for a similar observation. A joint physical population model could represent that ambiguity and propagate identification uncertainty into a margin distribution. Our MLP already learns interactions; correlated features alone are not a new ingredient. The potential addition is physically coherent uncertainty propagation. Applicability is conditional on the modeled population and observation likelihood being credible; a wrong structural prior can confidently favor the wrong interpretation. The paper's cross-airspeed dependence is optional information, not the prerequisite for borrowing its broader principle. [Source](https://arxiv.org/html/2210.15667v1), Sections 1 and 3–5.

2. **Méndez: use a better-resolved quantity to inform a weakly resolved one.** Its frequency–growth relation transfers accumulated phase information into damping estimation when envelope change is weak. P4 already has frequency, growth, damping and spacing features; simply adding these columns again misses the lesson. A physically supported joint relation could regularize identification or calibration. Applicability depends on finding a relationship that remains informative across our design population. One fixed trajectory is not assumed transferable. Its wrong-constraint sign reversals favor soft/probabilistic relations when the physical mapping is uncertain. [Source](https://arxiv.org/html/2609.25440v1), Sections 5–6.

3. **Méndez: judge observation sufficiency by information, not sample count.** Critical-mode cycles, growth accumulated over the usable duration and noise affect what a record can resolve. For P4, diagnose modal error against usable pre-cap duration, oscillation cycles, excitation and mode separation. Existing usable-fraction, fit-error, energy and availability information supplies a starting point. True modes may be used for TRAIN diagnostics, not as inference inputs. This is immediately applicable as an interpretation/diagnostic principle; copying its five-cycle threshold is not justified. The paper's profiling of amplitudes also illustrates separating excitation nuisance quantities from structural dynamics, without requiring adoption of its complete estimator. [Source](https://arxiv.org/html/2609.25440v1), Sections 2, 4–5.

4. **Crowther–Cooper: failed prediction can mean missing information, not insufficient capacity.** Its damping-only network does not improve through more training/hidden neurons; adding frequency helps. P4 should distinguish missing/poorly measured physical information from fitting limitations before another capacity change. This supports, rather than replaces, the existing true-modal versus measured-feature diagnostic. It does not establish that our current feature set is sufficient or that larger models/GRU cannot help. [Source](../../../p4_journal/papers/crowther2001.pdf), Section 6.1.

5. **Crowther–Cooper: model the errors the estimator actually encounters.** Noise-matched training helps, but poorly represented modal-sorting behavior still harms prediction. P4's waveform-noise augmentation covers the basic principle already. The incremental useful question is whether held-out failures involve biased growth, missed weak modes, mode substitutions or correlated identification errors, rather than another independent noise draw. Diagnose those errors before deciding on augmentation. Arbitrary independent perturbations of modal feature columns need not be physically or statistically valid. [Source](../../../p4_journal/papers/crowther2001.pdf), Sections 5.5–6.2.

6. **Wang–Ma: overall accuracy can hide failure in the important regime.** Its initially high overall classification accuracy coexists with poor boundary-class accuracy, prompting separate regime analysis and imbalance handling. P4's conditional boundary metrics already implement the main lesson. Class weighting is not an automatic solution: our previous loss experiments exposed tradeoffs, so any intervention must preserve the physically important error balance. Explicit geometry/load/damping inputs can inspire TRAIN-only checks of which hidden physical variation causes calibration errors, not an assumption that those quantities are available at deployment. Scaling and depth/width tuning provide mostly already-covered practice; its results do not identify a compelling new physical mechanism to import. [Source](../../../p4_journal/papers/aerospace-11-00677.pdf), Sections 4.2–4.6.

### Priority for P4

The strongest candidate mechanism is **uncertainty-aware joint modal/physical inference**: use measured dynamics coherently, and express remaining structural-calibration ambiguity. Chajjed informs the population/calibration part; Méndez informs identification and observation sufficiency. Crowther supplies a useful information-versus-capacity diagnostic and realistic-error lesson. Wang–Ma supplies mostly evaluation/training lessons already covered.

This is a mechanism-focused working hypothesis, not a selected probabilistic architecture or approved experiment. If measured identification is the dominant error, prioritize the identification lesson; if even well-measured dynamics leave broad margin ambiguity, prioritize probabilistic physical calibration. Quality indicators must be validated against actual errors; their presence alone is not calibrated confidence. GRU remains a live representation option. No new training, default change, reproduction or approved objective pivot.

## 15. Ueda-Iio-Ikeda 2009: local response intensity and flutter extrapolation

Source: [full manuscript](../../../p4_journal/papers/ueda2009_wavelet.pdf), [publisher DOI](https://doi.org/10.2322/tjsass.51.275). Read all seven pages; rendered pp. 276, 279 and 281 to resolve damaged equation extraction and missing minus signs in Table 1. This is a mechanism-focused assessment, not a reproduction proposal. P4 applicability judgments below are [INFERENCE].

### What the method actually uses

- **Physical observations:** a cantilever wing, aspect ratio 3 and root chord 300 mm, excited continuously by wind-tunnel turbulence. Strain and laser displacement measurements; speeds increased stepwise, typically held about 20 s. Three runs on the same model, not three different structures. Measured flutter speed about 37.6 m/s, flutter frequency 19.5 Hz; repeatability within 0.1 m/s. Original sampling 1 kHz, every fourth sample used for CWT: 250 Hz. Sections 3-5.
- **Transform:** Gaussian-modulated sinusoid (Gabor wavelet), sigma=1 and omega0=2*pi*4. Scale maps to physical frequency as f=4/a. Their computation uses the real transform, with a cosine-symmetry optimization; it is not a learned scalogram model. Equations (2), (10)-(13).
- **Index:** Equation (14) is `F = 1 / sum_j |W_real(b_j, a0)|`, over 2 s. This is a reciprocal absolute-coefficient sum, **not** reciprocal squared wavelet energy or a damping estimate. The fixed scale `1/a0=5.2` corresponds to 20.8 Hz, the **calculated flutter frequency**. Thus the frequency band incorporates target-model analysis, not just information discovered blindly from the subcritical response. Section 6.
- **Selection and extrapolation:** the raw `(mean speed, F)` points scatter strongly because response intensity is intermittent. They retain the lower envelope of F, i.e. the high-intensity response segments, then extrapolate a straight line to F=0 across multiple subcritical speeds. The paper describes selecting intensive parts but does not specify an automatic envelope/selection rule. Figures 5-7.

The useful physical idea is to use increasing response intensity in a flutter-relevant band under continuing excitation, rather than require an accurate damping estimate or visibly resolved frequency coalescence. The authors first attempted visual coalescence tracking and found it unreliable below flutter; this motivated the index. Sections 5-6. Their conclusion attributes the successful amplitude-based methods to structural response under unsteady aerodynamic forcing, but three runs on one model do not establish that amplitude is universally indispensable. Section 8.

### What the experiment establishes

All methods use the restricted 25-32.5 m/s response range. Relative to 37.6 m/s this is approximately 66-86% of flutter speed; the text describes a less-than-85% goal, but its stated upper speed is about 86.4%. Section 6 and Figure 7.

| Method | Test 1 error | Test 2 error | Test 3 error |
| --- | --- | --- | --- |
| Spectral-bandwidth damping | N.A. | N.A. | N.A. |
| Random-decrement damping | N.A. | -15% | -13% |
| Inverse spectral peak (peak hold) | +4.0% | -5.3% | -11% |
| ARMA index | +41% | +24% | +11% |
| Proposed wavelet index | approximately 0% | -10% | -4% |

Source: rendered Table 1, p. 281. Wavelet predicted speeds are 37.5, 33.7 and 36.1 m/s (Section 6); Table 1 prints the first error as `-0.0%`. Do not interpret that rounded entry as exact boundary recovery. Positive error overpredicts critical speed and can be nonconservative; negative error is conservative critical-speed underprediction, not a calibrated warning guarantee.

The wavelet method gives a useful conservative estimate in these tests, while ARMA overpredicts substantially. But wavelets do **not** dominate the simpler inverse spectral peak in every run: peak hold is closer in Test 2. The comparison therefore supports amplitude-sensitive information and local selection, not a uniquely superior wavelet transform. No confidence intervals, calibrated instability probabilities, cross-structure transfer or trajectory-level warning-risk validation are established here. Sections 7-8.

### What can help P4, and what cannot be copied

1. **Local modal observability is the most useful conditional lesson.** The response at one fixed speed changes over time, and whole-record spectra conceal when a band is informative. [INFERENCE] Within P4's usable pre-cap prefix, local band strength/persistence or local fit consistency could distinguish a poorly observed mode from genuinely weak growth. They could be useful quality information for boundary errors. However, P4's current simulator propagates fixed modal poles from random initial conditions, without continuing stochastic forcing; its time variation is not the paper's turbulent intermittency. There must first be evidence of useful local information beyond global modal fitting.
2. **Absolute response amplitude requires an excitation and calibration model.** P4 removes per-channel offset and calibration RMS using the first 51 samples, then caps at +/-50. It generates random modal initial amplitudes independently of stability. A large original transient therefore does not by itself mean small flutter margin. For a fixed linear wavelet, `F(c*x)=F(x)/|c|`; P4 normalization instead makes positive scalar multiples identical when the RMS floor is inactive. The paper's raw amplitude trend cannot be recovered by applying CWT after normalization. Relative within-record band evolution can survive, but it is a different observable and needs separate validation; do not remove physical calibration to manufacture the paper's cue.
3. **A fixed known flutter band and a speed sweep are additional information.** The 20.8 Hz band comes from the target calculation, and F=0 is obtained from multiple target operating points. P4 currently uses a measured waveform and timebase across unknown designs. [INFERENCE] A band-local representation would need a measured-signal-only selection rule or a declared, justified prior; no solver critical-frequency/critical-mode label may enter inference. The extrapolator itself requires a different observation contract. These restrictions do not prevent learning from local response information.
4. **Intensive-segment selection needs honest availability and boundary evaluation.** Lower-envelope selection suppresses weak-response scatter but conditions on sufficiently excited observations. [INFERENCE] A weakly excited critical mode should trigger identification uncertainty or insufficient-observation handling, not an invented stable estimate. If local quality is used, test its relation to actual errors and report how often usable observations exist. Do not equate high amplitude with validated confidence or rely on unspecified hand selection.
5. **The transform is not automatically an online solution.** Equation (13) explicitly includes samples at both `m-k` and `m+k`. On a completed observed clip this does not require data beyond the clip, but a streaming version must account for right-hand support, edge handling and latency. The paper's 2 s aggregation must not be copied as a sample count across P4's variable physical timebases. Bandwidth and temporal aperture remain limited by observed cycles; CWT cannot recover missing information merely by changing representation.

Repository evidence: [`causal_calibration_normalize` and `generate_clip_from_eigendecomposition`](transient.py), [`estimate_modes` and spatial modal rows](modal_features.py), and [the current acquisition replay](../../../p4_groups/spatial_modal_20260930/prepare_features.py). Current identification fits shared poles across channels and already supplies growth, frequency, fitted component-energy fractions, residual error, usable fraction and multi-mode spatial patterns. Those energy fractions rank fitted observability, not mechanical energy. [INFERENCE] The candidate addition is **time-local information/quality**, not another copy of frequency, damping, global energy or spatial features, and not an assumption that the already tested joint/soft integration should be revived.

Mechanism-only numerical check: using the real Gaussian/cosine transform and Eq. (14) on a synthetic 20.8 Hz exponentially decaying tone, multiplying amplitude by 10 gave raw `F(10x)/F(x)=0.100000000000`. After calling the actual P4 51-sample calibration routine, the ratio was `1.000000000000`. Both identities passed numerical assertions. This checks the amplitude/normalization argument only; it is neither a paper reproduction nor a P4 dataset/model result.

**Assessment [INFERENCE]:** useful as evidence to diagnose whether the critical band is observable throughout a short record, and to distinguish excitation-sensitive amplitude from stability information. It does not establish that appending a CWT index or scalogram will improve P4. Existing global modal/spatial information covers much of the physics; local quality is the potentially uncovered part. No new experiment or default selected. The rejected residual/scale integration remains out of the active path; artifacts remain negative evidence, current blend is unchanged, reserved splits remain untouched and GRU remains live.

## 16. Local modal observability experiment: information gain without adoption

User approved testing the Section 15 mechanism. Completed fixed TRAIN-only experiment: [protocol](../../../p4_groups/local_observability_20261001/protocol.json), [results](../../../p4_groups/local_observability_20261001/results.json), [execution](../../../p4_groups/local_observability_20261001/run_experiment.py), [feature extraction](../../../p4_groups/local_observability_20261001/local_features.py), [evaluation](../../../p4_groups/local_observability_20261001/evaluate_results.py). This is not a reproduction of Ueda's amplitude index or multi-speed extrapolator.

### Controlled change and verification

Two measured candidate bands: first/second least-stable modes with >=1% fitted component energy, cached feature columns 28/31. No oracle flutter frequency, pole, velocity or structural parameter enters the added features. Complex Gaussian band coefficients use sigma=2/f, truncated at +/-6 cycles, with complete support strictly before the first selected-channel cap. Two-cycle sigma is a fixed adaptation to short records, not the paper's four-cycle sigma; no width sweep. Band share is coefficient power divided by Gaussian-smoothed total signal power. Local descriptors cover share quantiles/evolution/persistence, effective band-power time fraction and departure of log power from its global trend. Missing support has explicit availability flags; no clips are dropped.

Four arms: unmodified frozen 50/50 107/174-feature spatial blend; both members refitted with 12 global band/aperture controls; both refitted with global12+local12; equal-dimension sham with local12 shuffled within proper/calibration/held partitions, keeping acquisition copies paired. Original MLP32->16, alpha1, four-view independent-noise training and all split conventions retained. No rejected residual/scale integration, joint physical prior or soft constraint is used.

77 original TRAIN designs, 4,454 underlying clips, five outer design folds, three initialization seeds and 37 acquisition conditions. All 90 new member fits converged without warnings. Synthetic behavioral smoke checked steady/burst observability, amplitude/time-unit invariance, cap-tail exclusion and unavailable bands. Actual end-to-end smoke covered 928 held-design clips in all 37 conditions; interval/probability/warning evaluation was also exercised on those outputs. Full run checked finite outputs, proper/calibration/held design isolation, source hashes and frozen point replay at atol/rtol1e-12.

### Accuracy and attribution

Mean design MAE; noise rows average the three fixed seeds and both existing 949/951 draws. These are mean errors across fits/draws, not an ensemble predictor.

| Evaluation | Frozen blend | Global controls | Global + local | Shuffled local |
| --- | --- | --- | --- | --- |
| Clean | 0.007625 | 0.007411 | 0.007359 | 0.007602 |
| Independent 1% | 0.009093 | 0.008892 | 0.008924 | 0.009168 |
| Correlated 1% | 0.009895 | 0.009780 | 0.009751 | 0.009977 |
| Independent 5% | 0.015597 | 0.015478 | 0.015397 | 0.015765 |
| Correlated 5% | 0.024591 | 0.024562 | 0.024054 | 0.024662 |

The local arm improves correlated5% MAE by 2.19% versus frozen and 2.07% versus global controls. Its pooled local-minus-global difference is -0.000509, paired whole-design 95% interval [-0.000943,-0.000223]. Both individual draw intervals also exclude zero: -0.000474 [-0.000806,-0.000176] and -0.000543 [-0.000948,-0.000166]. It also beats the equal-dimension sham in the pooled comparison. This supports useful aligned local information in that stress regime, rather than only extra capacity or global band strength.

The gain is not uniform over initializations: correlated5% local-minus-global mean MAE is approximately +0.000004 for seed42, -0.000966 for seed43 and -0.000564 for seed44. At clean, independent1%, correlated1% and independent5%, the additional local-versus-global pooled MAE intervals include zero. Do not attribute their entire local-versus-frozen improvement to temporal localization.

### Boundary and decision tradeoffs

There are real average boundary improvements. At independent1%, stable/unstable last1% MAE changes from 0.003895/0.003774 to 0.003648/0.003225; the unstable-side local-minus-frozen paired interval excludes zero. At correlated5%, these means change from 0.011200/0.005998 to 0.010496/0.005614, but both paired intervals include zero.

Average boundary error does not protect the tails:

- For unstable last1% clips under independent1%, `prediction-truth > .03` rises from 0.309% to 0.617%; global controls give 0.103%. These are seed/draw-averaged clip rates, not independent population risk estimates.
- Under correlated1%, farther-stable (`margin >= .15`) underestimation by more than .05 rises from 4.756% to 5.335%. The paired difference is +0.005790, 95% interval [+0.000411,+0.011610]: a supported worsening in pessimistic errors.
- At the calibration-selected5% stable false-warning budget, independent1% unstable-last1% miss falls from 3.50% to 1.95%, but the paired interval includes zero. Under correlated5%, it rises from 11.01% to 12.45%, also with an interval including zero. No warning improvement is established at comparable false-alarm behavior.
- The Gaussian residual-RMS probability comparator's pooled Brier score worsens in the independent1%, correlated1% and independent5% families with paired intervals excluding zero. This does not establish failure of every possible conditional uncertainty model; this experiment does not learn a quality-dependent uncertainty or abstention policy.

The local arm passes only 11/48 strict zero-deterioration screens versus frozen, 5/48 versus global and 16/48 versus sham. Crossing zero is uncertainty, not proof of harm; the supported farther-stable deterioration and observed optimistic tails are separate evidence against adoption.

### What the quality diagnostic does and does not explain

Complete-support availability is 92.0%/61.4% for the clean first/second bands across all clips, but both are available on every clean/independent1% stable-last1% clip; correlated5% stable-last1% availability is 100%/99.4% on the first primary draw. Thus overall aperture loss is not a sufficient explanation for the boundary tradeoff.

With per-fold quartile cutoffs fixed on clean proper-fit features, large first-band log-power-fit residual associates descriptively with larger frozen stable-last1% error. At independent1% seed949, the highest residual quartile has design MAE0.007215 versus0.003746/0.003783 in the middle quartiles; at correlated5% seed949, 0.021840 versus0.009581/0.010099. Groups contain different clips/designs; this is not a causal or calibrated reliability result.

A counterexample matters more than treating that association as a confidence rule. For D000093/source3696, seed43, iid1_new949: true margin -0.006458, frozen estimate +0.000029, local estimate +0.023916. Both bands are available; first-band log-power-line RMSE0.001868 is in the lowest clean-training quartile. The optimistic error repeats on both independent/correlated1% draws for that seed. Other new threshold-crossing exposures include D000135/source5198; detailed records in [new_optimistic_tail_cases.json](../../../p4_groups/local_observability_20261001/new_optimistic_tail_cases.json). **Low local band-fit residual is not sufficient evidence of a trustworthy margin or correct critical-mode identity.**

Actual held-waveform timing: added two-band extraction median/p95 3.26/5.33 ms; warm cached-base-feature+waveform-to-distribution median/p95 0.72/0.96 ms frozen versus4.13/8.11 ms local. Both members and local extraction are included for the addition; original normalization/modal/spatial extraction is excluded, so these are not full sensor-to-output latencies. See [timing evidence](../../../p4_groups/local_observability_20261001/inference_timing.json).

**Decision:** retain the existing blend; do not adopt this direct descriptor-augmentation variant. Keep the local-observability result as a live, bounded representation finding, not a failed-wavelets verdict or calibrated-confidence claim. [INFERENCE] The discriminating next diagnosis is why specific apparently consistent measured bands still produce optimistic cross-design margin errors, separating mode identity/calibration/fitting from poor observation; a generic high-residual cutoff would miss the new D000093 case. No further experiment or warning cutoff selected. Existing rejected residual/scale integration remains out of the active path, reserved splits remain untouched and GRU remains live. Previously evaluated noise draws, fixed-prediction2000 design bootstrap, no refit uncertainty or multiplicity correction: this is exploratory evidence, not blind confirmation.

## 17. Optimistic-failure audit: encoded critical dynamics, poor cross-design mapping

Completed the user-approved TRAIN-only diagnosis, not a new training experiment. [Protocol](../../../p4_groups/local_observability_20261001/audit_protocol.json), [consolidated evidence](../../../p4_groups/local_observability_20261001/audit_results.json), [actual replay](../../../p4_groups/local_observability_20261001/audit_failures.py), [fitting/neighborhood execution](../../../p4_groups/local_observability_20261001/audit_mapping.py), [modal replay](../../../p4_groups/local_observability_20261001/audit_modal.py), [joint-pole diagnostic](../../../p4_groups/local_observability_20261001/audit_complex_modes.py), [isolation/provenance checks](../../../p4_groups/local_observability_20261001/audit_evidence.py). Existing 77 TRAIN designs, 4,454 physical clips, 37 acquisition conditions, three arms and three saved seeds. No refitting, simulation, reserved-split scoring or production-policy change.

### Cohort and verification

Strict severe optimism means `prediction-truth > .03`. Across all conditions: 82,306 arm/seed/condition exposures, 15,837 unique condition/clips, 2,030 physical clips. Primary clean+eight949/951 conditions: 25,578 exposures, 4,659 condition/clips, 1,697 physical clips. The unstable-last1% primary subset contains 81 exposures, 30 condition/clips, 12 physical clips and 11 designs. Repeated exposures are not independent observations, and the broad continuous-margin-error cohort is not itself a missed-warning cohort.

Both named failures replay from actual model shards at atol/rtol1e-12. Replayed 51,264 measured acquisitions through original calibration/capping, estimator and full174 encoding: maximum cache discrepancy 8.88e-15. Actual held predictions on the original four acquisition types replay at1e-12; every neighborhood reference was verified to belong to the corresponding proper-fit partition, with five distinct reference designs. No new permanent tests or production source changes.

### Modal identity: frequency-only matching can mislead

The existing broad frequency Hungarian gate reports critical recovery87.85% in primary severe acquisitions versus96.92% in other primary acquisitions. Those descriptive populations have different target distributions; this is not causal evidence or a validated quality rule.

Near flutter, frequency matching is not uniquely identifying: **all30 primary unstable-last1% severe acquisitions have another true pole within twice the frequency gate.** A secondary diagnostic therefore compares complex poles using both growth and angular frequency, restricted to modes already encoded in the energy-top3 and least-stable visible pair. Solver poles are diagnostic references only.

For all30 boundary acquisitions, the nearest encoded complex pole maps to the true retained critical pole. Its error divided by the true critical pole's separation from the other seven has median0.002210, p950.011076 and maximum0.029901. Absolute critical-growth error has median0.04785/s and p950.22010/s. The label-driving mode is represented and matches retained critical index0 throughout the primary severe cohort.

| Named iid1_new949 acquisition | True critical alpha / frequency | Encoded alpha / frequency | Complex error / true pole separation |
| --- | --- | --- | --- |
| D000093 / 3696 | 2.659318/s / 74.447284Hz | 2.672759/s / 74.447870Hz | 0.000294 |
| D000135 / 5198 | 2.702966/s / 92.682171Hz | 2.708228/s / 92.682515Hz | 0.000126 |

In both cases the measured leading mode also maps to critical0. **Missing critical dynamics is not a general explanation for this tight-boundary cohort.** This does not prove the full margin representation is sufficient.

The broader severe cohort still has substantial identification/representation fragility: the best encoded candidate's nearest true pole is critical0 in80.81% of primary severe versus98.08% nonsevere acquisitions; normalized complex error medians0.06062 versus0.000781. Do not generalize the boundary finding to every severe continuous-margin error.

### Actual fitting versus held-design errors

Predict with existing fitted models on their proper-fit original four acquisition views, then held designs on the same acquisition types. Aggregate design MAE pools actual predictions; proper designs repeat across applicable outer models and seeds, held designs across seeds. These are descriptive in-sample comparisons, not deployment estimates or controlled refit effects.

| Arm | Proper-fit design MAE | Held-design MAE | Proper unstable-last1% e>.03 | Held unstable-last1% e>.03 |
| --- | --- | --- | --- | --- |
| Frozen | 0.005330 | 0.010310 | 0/6048 | 8/1944 (0.412%) |
| Global controls | 0.005239 | 0.010169 | 0/6048 | 2/1944 (0.103%) |
| Global + local | 0.005106 | 0.010167 | 0/6048 | 12/1944 (0.617%) |

Zero observed training events is not zero population risk. Local unstable-last1% mean design MAE improves from frozen0.002245/0.003986 to0.001921/0.003555, proper/held, while held optimism tails worsen.

For the same named fresh waveform and initialization seed, other outer models whose proper-fit sets include the design have local errors+0.00195 to+0.00499 for D000093 and-0.00243 to+0.00361 for D000135. Their local point predictions are all negative, whereas the respective held-design models give+0.023916 and+0.027434. These are **seen-design fresh-view predictions**, not exact training-row predictions or controlled leave-one-design experiments: the other fit designs also change between models.

### Neighborhoods: thin support, not conflicting signs in the named cases

Every15,837 severe-union acquisition searches proper-fit designs only in each of the six arm/member spaces. Use original four-view train-only scaling and RMS-standardized Euclidean distance on active columns. Collapse noise copies to each physical reference clip's closest view; report five distinct nearest designs.

Both named cases have **five negative design-neighbor margins in both local member spaces**. Nearest margins are-0.007560 for3696 and-0.006458 for5198; five-design ranges are0.0064-0.0083. Under this metric, contradictory nearby labels do not explain the named positive predictions. Across the30 primary unstable-last1% severe acquisitions, each local member has a negative five-design median in26/30 queries.

The named queries nevertheless have relatively large feature distances: local nearest-distance percentiles against clean proper leave-own-design-out distances are98.36%/99.89% for3696 and96.78%/98.04% for5198,107/174 base-member spaces respectively. This supports investigating coverage, but the reference distribution is global and clean, not noise/margin-matched; it is not calibrated confidence. Distances are nonzero, correlated blocks affect the metric, and neither similar labels nor finite-distance neighbors establish exact feature equivalence, representation sufficiency or an information ceiling.

### Decision and next justified probe

**Retain the existing blend.** [INFERENCE] Named and tight-boundary failures point chiefly to cross-design mapping/coverage rather than universally missing critical modes, bad local-band fits or conflicting neighbor signs. Broader severe failures also involve identification, so a single universal correction is not justified.

The next recommended controlled probe is an **unchanged-feature, design-balanced neighborhood mapping reference versus the existing learned mapping**, scored on complete held-fold populations with boundary tails, farther-stable errors and matched warning budgets—not just the selected failures. It can test local cross-design interpolation before adding another quality feature or sweeping a loss. This is a recommendation, not a started experiment, adopted predictor or warning gate. GRU remains a live learned-mapping candidate; rejected residual/scale integration stays out of the active path.

All evidence is exploratory: previously scored acquisitions, post-hoc severe selection, metric-dependent neighborhoods, no causal attribution, refit uncertainty, multiplicity correction or calibrated-confidence claim. Saved arrays, source/model hashes and executable audit checks make the diagnosis reproducible.

## 18. Unchanged-feature neighborhood reference: named repairs, poor global rule

Completed the Section17 probe after user approval. [Protocol](../../../p4_groups/neighborhood_reference_20261002/protocol.json), [prediction execution](../../../p4_groups/neighborhood_reference_20261002/neighborhood_reference.py), [evaluation](../../../p4_groups/neighborhood_reference_20261002/evaluate_reference.py), [results](../../../p4_groups/neighborhood_reference_20261002/results.json), [isolation verification](../../../p4_groups/neighborhood_reference_20261002/verification.json), [timing](../../../p4_groups/neighborhood_reference_20261002/inference_timing.json).

### Fixed comparison and verification

Keep the existing50/50 107/174-feature MLP blend, three saved initialization seeds, feature tables and design partitions unchanged. Candidate uses the same two feature spaces, original four proper-fit acquisition views and existing train-only active-column scaling. RMS-standardized Euclidean distance collapses each physical clip's four noise copies to its closest view; each proper-fit design contributes its nearest physical clip. Choose five distinct nearest designs, median their five margins separately in each member space, then average member medians50/50. Stable ties, fixed k5; no labels used for neighbor selection, distance weighting, tuning, feature changes or MLP refitting.

Original77 TRAIN designs,4,454 clips, five outer folds and37 acquisition conditions. Calibration designs remain outside the library/scaling. Reuse the existing calibration method on their original four views: clip90% residual and finite90% design-max intervals, residual-RMS Gaussian unsafe probability comparator, stable false-warning budgets1/5/10/20%. **One deterministic reference is not three independent fits.** Paired fixed-prediction2000 whole-design resampling averages learned-reference seed errors/events before resampling; results retain separate seed and fold metrics.

Actual complete-fold smoke covered928 held clips and all37 conditions, including calibration and downstream evaluation. Full run checks every neighbor is proper-fit only, five distinct design votes, complete outer coverage, finite outputs, exact preservation of MLP predictions/thresholds and respected calibration budgets. Original MLP point predictions replay through both actual members at1e-12. Numerical probes checked unequal design sizes, copy collapse, ties, target-blind neighbor choice and deterministic-candidate pairing. On actual noisy features, blocked BLAS/direct distances agree within1.60e-12 and selected neighbors/median labels agree exactly.

### Aggregate and boundary results

Mean design MAE; noise families average both existing949/951 draws and reference seed metrics, not a seed-ensemble predictor.

| Condition | Existing learned blend | Neighborhood reference |
| --- | --- | --- |
| Clean | 0.007625 | 0.018420 |
| Independent1% | 0.009093 | 0.020815 |
| Correlated1% | 0.009895 | 0.021473 |
| Independent5% | 0.015597 | 0.028574 |
| Correlated5% | 0.024591 | 0.031350 |

Every clean/noise-family paired design-MAE interval excludes zero on the worsening side. Independent1% delta+0.011722,95% interval[+0.010881,+0.012495]; correlated5% delta+0.006759[+0.004700,+0.008503].

Both named iid1_new949 observations move onto the correct unstable side:

- Source3696: true-0.006458, neighborhood-0.007159, error-0.000701.
- Source5198: true-0.004086, neighborhood-0.006458, error-0.002371.

Source3696 was a severe **local-augmentation** failure in the prior audit, not a severe failure of every current frozen reference seed. These secondary observations do not establish general improvement.

Average stable-last1% and unstable-last1% errors **both worsen with paired support in all four noise families**. Independent1% design MAEs0.003895/0.003774 become0.007043/0.006067; correlated5%0.011200/0.005998 become0.016440/0.010638.

Observed unstable-last1% optimism `error>.03` changes from0.309% to0 under independent/correlated1%; their paired intervals reach zero, so this is not zero population risk or a calibrated tail guarantee. Independent5%0.720%->0.309% and correlated5%1.337%->2.160% have paired intervals crossing zero.

The farther-stable guardrail fails decisively. For `margin>=.15`, underestimation by more than0.05 changes:

| Noise | Existing blend | Neighborhood reference |
| --- | --- | --- |
| Independent1% | 4.36% | 29.65% |
| Correlated1% | 4.76% | 31.51% |
| Independent5% | 20.72% | 53.35% |
| Correlated5% | 15.34% | 53.47% |

Every paired increase excludes zero. Independent1% increase+0.252895[+0.212060,+0.294976]; correlated5%+0.381307[+0.346278,+0.419171].

### Warning tradeoffs and runtime

Same calibration-selected5% stable false-warning budget, not equalized held false-warning rates:

| Noise | Held stable false warnings: learned -> neighborhood | Unstable-last1% misses: learned -> neighborhood |
| --- | --- | --- |
| Independent1% | 4.99% ->5.87% | 3.50% ->3.40% |
| Correlated1% | 4.57% ->5.51% | 4.12% ->3.70% |
| Independent5% | 4.66% ->3.92% | 5.56% ->7.41% |
| Correlated5% | 2.79% ->3.27% | 11.01% ->9.57% |

All four paired boundary-miss intervals include zero: **no calibrated warning improvement is established**. Correlated1% false warnings worsen with paired support; the other false-warning differences remain uncertain. Residual-RMS Gaussian-comparator Brier worsens with paired support in all families. Interval coverage/availability and all warning budgets are saved in results. Conservative zero-deterioration screen5/56, not a formal noninferiority or deployment certificate.

Warm cached174-feature row through both members and calibrated distribution,150 held queries in the same benchmark: learned median/p950.278/0.424ms, neighborhood4.546/6.735ms. Single-row outputs replay saved batched outputs within9.99e-16. Waveform normalization/modal/spatial extraction is unchanged and excluded; these are not full sensor-to-output latencies or directly interchangeable with earlier timing boundaries.

### Decision and interpretation

**Retain the existing learned blend; reject this fixed neighborhood rule as a replacement.** Named signs and some observed optimism events improve at a substantial average-boundary/far-stable cost, without a supported warning gain.

[INFERENCE] Plain RMS feature proximity and design-balanced median votes are not a useful enough cross-design margin rule under this protocol. The learned mapping contributes useful global structure beyond this interpolation. This experiment does not show that every local method fails, that the representation is sufficient, or that GRU should be abandoned; metric, interpolation, vote balancing and physical-design coverage were not separately isolated.

No post-result tuning or new learner/feature/loss experiment started. Current production default and reserved splits unchanged; GRU remains live and rejected residual/scale integration remains out of the active path. Previously scored acquisitions, repeated exploration, fixed-prediction design bootstrap with no refit uncertainty or multiplicity correction: exploratory evidence, not blind confirmation.

## 19. Independent additions to the frozen blend: retain local late fusion for research only

**Decision:** retain the learned local-observability late-fusion route as an
experimental candidate; exclude neighborhood and both consistency routes
from the retained set. **Do not change the production default or claim better
boundary warnings.** Both previously named near-flutter failures worsen.
Only one new family survives, so the final retained configuration is the
existing spatial blend plus this one predictor, not a multi-addition stack.

This closes the replacement-versus-addition gap in Sections16/18 and the
earlier noise-consistency study. A standalone loss did not establish lack of
complementary information; these are actual additions to the frozen old blend.
Previously completed joint/soft physical integration remains excluded.

### Frozen protocol and fitting isolation

Artifacts: [`blend_additions_20261002`](../../../p4_groups/blend_additions_20261002/).
[`protocol.json`](../../../p4_groups/blend_additions_20261002/protocol.json)
was frozen before full scoring. Original77 TRAIN designs,4454 physical clips,
five outer design folds, seeds42/43/44 and all37 existing acquisition conditions.
No reserved selection/calibration/test designs, simulation, new physical
features, architecture/loss search, post-result tuning or GRU experiment.

Each added point readout is tested independently by:

- Fixed50/50 old-blend/new-predictor late fusion.
- Frozen old blend plus a residual ridge learned from three inner
  **design-disjoint** proper-fit folds, using the original four acquisition
  views and original1047+outer-fold assignments.

Controls: unchanged frozen blend; scalar affine correction; the previous
seven-column blend-only correction; matched custom-supervised predictor for
consistency; matched global-descriptor predictor for local observability.
Each learned addition appends only its own predictor-minus-blend readout to
the seven-column control. Same train-only scaling, unpenalized intercept,
normalized ridge penalty0.1 and original calibration partitions; no learned
uncertainty scale. All points are independently recalibrated for original
raw/design-max90% intervals, Gaussian-RMS probability comparator and
calibration-selected1/5/10/20% false-warning budgets.

Historical standalone consistency fits included the current calibration
designs: **they were not reused.** All new inner/outer consistency fits use
proper-fit designs only. Compatible full-proper observability members and
historical inner base pairs are source/hash/row checked before reuse;
inner base outputs are numerically replayed and unused oracle readouts never
enter a fusion head.

All360 new neural fits converged, iterations455–1659, no recorded native
convergence warnings.465 unique fit/reuse records include45 historical
inner base pairs and60 full-proper observability members.120 independent
point-correction ridge heads; no cross-family head was fitted before selection.

Retention requires supported fresh-family aggregate-MAE improvement beyond
the frozen blend **and** matched controls, plus no statistically supported
deterioration in the predeclared clean/fresh boundary, far-stable, Brier and
calibration5%-warning guardrails. Unresolved intervals are **not** noninferiority
proof. Selection is immutable and bound to independent-results/protocol hashes
in [`retention.json`](../../../p4_groups/blend_additions_20261002/retention.json).

### Full-population results

Mean design MAE; all seeds and both fresh acquisition draws retained, not
predictions averaged into a seed ensemble:

| Condition | Existing blend | Blend-only correction | Correction + global observability | Correction + local observability |
|---|---:|---:|---:|---:|
| Clean |0.007625|0.007446|0.007133|**0.007115**|
| Independent1% |0.009093|0.009012|0.008656|0.008674|
| Correlated1% |0.009895|0.009814|0.009473|**0.009470**|
| Independent5% |0.015597|0.015663|0.015139|**0.015051**|
| Correlated5% |0.024591|0.024100|0.023685|**0.023454**|

Retained local route minus existing blend, paired95% design-bootstrap intervals:

- Independent1%:−0.000419 [−0.000725,−0.000200].
- Correlated1%:−0.000425 [−0.000742,−0.000203].
- Independent5%:−0.000546 [−0.000688,−0.000421].
- Correlated5%:−0.001137 [−0.001411,−0.000882].

Local information **beyond the global-descriptor matched control** is supported
at5%: independent−0.000088 [−0.000165,−0.000009],
correlated−0.000231 [−0.000412,−0.000074]. At1%, the local-minus-global
intervals include zero; do not attribute the entire improvement to local
descriptors. Improvements beyond blend-only correction are supported in all
four fresh families.

### Independent reject decisions

- **Neighborhood:** fixed fusion worsens aggregate errors; learned fusion
  contributes no supported gain beyond blend-only correction. Independent1%
  learned neighborhood-minus-control is+0.0000156
  [+0.0000036,+0.0000287]; correlated5%−0.0000089 has an interval crossing zero.
  Fixed/learned routes have18/14 supported deterioration checks versus frozen
  blend. Exclude these routes, not every possible neighborhood method.
- **Global consistency:** fixed fusion has attributable independent5%-MAE
  gain, but its calibration5%-budget unstable-last1% miss rate worsens by
  +1.75pp [+.31,+3.36] there. Correlated5% stable/unstable last1% MAE and
  boundary misses worsen. Learned fusion has no qualifying matched gain and
  worsens correlated5% stable-last1% MAE and boundary misses. Exclude both.
- **Localized consistency:** neither route establishes the required matched
  incremental gain. Fixed fusion worsens correlated5% stable-last1% MAE by
  +0.000706 [+0.000147,+0.001327]; learned fusion increases independent1%
  held stable false warnings by+0.198pp [+0.012,+0.432] at the5% calibration
  budget. Exclude both. Some aggregate gains do not justify keeping the method.

### Warning limits and known failures

Retained learned local route: zero **supported** deterioration checks but25/40
guardrail intervals still have a positive upper bound. This is exploratory
eligibility, not safety validation. Unstable-last1% MAE improves with paired
support at1%; the5% improvements remain uncertain. Far-stable
underestimation>.05 improves with paired support at both5% settings.
Correlated5% Brier improves by−0.000421
[−0.000804,−0.000040]; other Brier changes are uncertain.

At the calibration-selected5% false-warning budget:

| Noise | Existing unstable-last1% miss rate | Retained route |
|---|---:|---:|
| Independent1% |3.50%|3.09%|
| Correlated1% |4.12%|3.70%|
| Independent5% |5.56%|5.86%|
| Correlated5% |11.01%|12.24%|

Every paired boundary-miss interval includes zero: **no calibrated warning
improvement is established**, and the stress-case point estimates worsen.
Actual waveform replay also shows both named independent1% failures worsening:

- D000093/source3696, seed43: truth−0.006458; existing+0.000029;
  retained+0.008037.
- D000135/source5198, seed42: truth−0.004086; existing+0.024748;
  retained+0.029098. Its optimism now exceeds0.03.

Do not present lower aggregate MAE as repair of these mapping failures.

### Final retained-set run and verification

Only `learned_local_observability` survives. The final run therefore replayed
**existing blend + that readout**, with matched blend-only and frozen controls,
on all37 conditions,4454 clips and three seeds. Saved final predictions and
thresholds are exactly identical to the independently tested retained route.
There is no distinct two-or-more-new-family combination to score and no
rejected method was silently mixed. This replay is not new confirmation data.
See `combination.json`, `combination_predictions.npz`, `combination_results.json`
and `final_verification.json`.

Full verification: all15 arms/3 seeds/5 folds assembled and replayed; frozen
distributions/thresholds, proper/calibration/held separation and all calibration
warning budgets passed. Six actual raw-waveform cases through every arm and
seed: feature max delta0; single-query calibrated-output max delta2.16e-15.
Retained-selection and independent-results hashes remain unchanged.

Cached-query timing,30 repeats on one actual measured feature row during final
evaluation: existing blend median/p950.542/0.602ms; retained local route
1.087/1.587ms. Includes both frozen members, both augmented local members,
residual head and calibrated distribution; excludes waveform normalization,
modal/spatial/local-descriptor extraction. Not sensor-to-output latency or
a deployment timing guarantee.

`results.json` contains every condition, all boundary slices, per-seed/fold
behavior, interval coverage/width/availability, probability reliability,
all warning budgets and paired comparisons.2000 fixed-prediction whole-design
bootstrap draws preserve seeds/acquisition pairs but omit refit uncertainty
and multiplicity correction. Previously scored TRAIN acquisitions and repeated
exploration: **research evidence, not blind confirmation**. Current production
default and reserved splits remain unchanged; GRU remains live.

