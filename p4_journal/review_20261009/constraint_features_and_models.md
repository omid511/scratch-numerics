# P4 constraint-indexed feature and model survey — 2026-10-09

Scope: P4 only. Survey recovered from preceding worker evidence (histories `history://ConstraintFeatureModels`;
native transcript `ConstraintFeatureModels.jsonl`) plus targeted re-reads of current probe maths, the failed
additive-spectral experiment, `P4_RESEARCH_TARGET.md`, and `src/mechanics/p4_margin_estimation/RESEARCH_REVIEW.md`.
No code/model changes, fits, builds, or tests were run for this report.
No literature claim of P4 performance is made: literature results are reported as their own authors' results on
their own tasks, never as expected P4 gains.

## 0. How to read this report (evidence labels)

- **[EVIDENCE]**: directly observed in repo artifacts or full-text sources recorded in the review/journal.
- **[WORKER-EVIDENCE]**: captured by the preceding survey worker (web-search snippets, review excerpts); not
  re-verified to full text by this report. Snippet/abstract-only items are marked so explicitly.
- **[INFERENCE]**: interpretation or mechanism hypothesis, not measured evidence.
- Constraint labels (the ONLY allowed filter):
  - **COMPATIBLE**: usable under the frozen contract of §1 without new information.
  - **RELAXATION**: scientifically relevant but requires an explicitly declared contract change; retained, never
    silently dropped. The relaxation it needs is named.
  - **PROHIBITED**: uses oracle/extra information barred by the contract (simulator poles, true margin/critical
    speed, hidden design parameters, future response, target-specific calibration trajectory). Retained as a
    labelled diagnostic/counterfactual only, never as a deployable candidate.
- Sampler/estimator artifacts are called out where a published or measured effect may come from data generation
  or fitting rather than physics.

## 1. Frozen task and information contract (from `P4_RESEARCH_TARGET.md`)

- **Goal (proposal, not achievement):** transferable, uncertainty-qualified flutter headroom plus timely advisory
  warning from short multichannel measurements on unseen plate designs, **without a target-specific
  flutter-calibration campaign**.
- **Target:** `m = (ucrit − velocity)/ucrit`; positive = subcritical. Preserve stable AND unstable sides plus
  farther-stable guardrails.
- **Outputs:** point margin, uncertainty, explicit current/reliable vs unavailable/stale status. A positive point
  estimate is not a safety declaration.
- **Inputs:** causal measured sensor response, measured timebase, sensor identities, available causal calibration.
  NO simulator poles, true margin, critical speed, hidden design parameters, future response, or target-specific
  modal trajectory. Excitation timing/load allowed ONLY in a separately declared arm if actually available.
- **Observation budget:** max 0.5 s causal aperture, max 16 sensors. Sampling, stride, warm-up frozen on
  development designs only. Count physical seconds and mode cycles, not samples.
- **Noise:** clean / iid1% / correlated1% (`rho=0.8`) primary; iid5% / correlated5% stress. One frozen
  training/calibration policy, not per-condition winners.
- **Cases:** ordinary multimode response, slow/fast approaches, sustained crossings, stable/near-boundary dwells,
  approach–retreat, pulse onsets, weak/intermittent excitation. Initial-only decaying-into-noise responses are
  observability negative controls.
- **Warning:** primary claim restricted to a prospectively declared excitation envelope; every trajectory in the
  envelope stays in denominators. Competing methods get identical observations and gap/age policies.
- **Statistics:** disjoint complete-design groups for fitting / selection / calibration / blind evaluation.
  ≥120 fresh blind same-family designs planned (not yet acquired). Noise views/trajectories/overlapping windows
  are NOT independent structures.
- **Completion gates (proposed thresholds, §"Completion gate"):** signed MAE ≤0.010 clean / ≤0.015 per 1% family
  (≤0.005 / ≤0.010 inside `−0.02<m<0.05`); dangerous-error caps; calibrated 95% lower bound with ≤5% clearance-
  while-unstable; ≥80% qualified availability in final 2 s; ≥1 s lead for ≥90% crossings; ≤1 nuisance
  episode/hour; ≥25% timely-miss reduction vs strongest same-information competitor; 5% stress behavior; full
  pipeline p95 ≤10 ms; blind + independent-source credibility.
- **Closest precedents (remaining burden):** Crowther–Cooper 2001 (simulation-trained neural flutter predictors
  with structural variation — cross-design learning alone is not new); Chajjed et al. 2022 (random structural
  models + Bayesian modal priors — population priors/probabilistic margins alone are not new); Méndez 2026
  (configuration-specific frequency–growth trajectory — short-record fitting/constraints alone are not new, and
  its target is not cross-design normalized speed margin); Wang–Zhou 2026 (abstract-only: signal-based neural
  flutter-boundary prediction with wind-tunnel/flight validation — full transfer protocol unread, no superiority
  claim). P4 must earn the JOINT advantage; no "first" claim is supported.

## 2. Dispatcher regime note

The completed, independently verified experiment uses three nonoverlapping 128-sample windows at
1 kHz, ending at decision lags 4/2/0 on a 64-sample stride: 384 acquired samples, .383 s first-to-last
span, within the .5 s ceiling. No U/ucrit/truth enters the predictor. See
`p4_groups/operational_confirmation_20261003/multivel_history_20261009/{protocol,verification}.json`.
This report classifies candidates against that regime as follows:

- Single 128-sample-window estimators: COMPATIBLE (also with the legacy 0.5 s budget).
- Sequential three-window estimators (tracking, persistence, hysteresis, CUSUM, cross-window averaging with
  shared noise accounting): COMPATIBLE **within the declared 384 ms regime only** — they span 384 ms of
  wall-clock observation and must be labelled as 384 ms-span methods, not 128 ms methods. They must not be
  compared against single-window methods without stating the span difference.
- Multi-operating-point / multi-speed sweep estimators (classical flutter-margin extrapolation across
  airspeeds, Crowther-style speed histories): RELAXATION (needs target operating-point history) or PROHIBITED
  (needs known speeds/trajectory) — see §§7–8. **The 384 ms regime is NOT an independent known-speed sweep**:
  no airspeed is supplied and no cross-speed trend may be assumed.

## 3. Current probe mathematics (exact, read from source)

### 3.1 Acquisition and calibration (COMPATIBLE — this IS the contract)

- `causal_observation` (`operational.py`): one fixed leading calibration (51 samples); per-channel offset =
  leading mean, scale = leading RMS floored at 1e-8; output clipped to ±50 and cast float32. Missing data stay
  NaN; unavailable leading calibration raises. Original calibrated physical amplitudes determine saturation.
- `estimate_modes` (`modal_features.py:26–114`): waveform-only multichannel **LS-ESPRIT**:
  contiguous prefix before first sensor-cap hit; scale-normalized; temporal Hankel (lag 64, `width = min(lag,
  n//2)`) pooled across physical sensors; covariance eigendecomposition; rank ≤ `2*max_modes+1` at 1e-6
  relative singular-value threshold; LS shift-invariance solve; keep oscillatory conjugate representatives
  (`imag > 1e-8`); `alpha = ln|pole|/dt`, `frequency = angle/(2π·dt)`; per-mode energies by least-squares fit
  of `exp(alpha·t)·cos/sin` basis + DC per channel. With `include_shapes=True`, per-mode complex sensor
  coefficients (cos − i·sin), global complex scale arbitrary. Unidentifiable signals return zero modes with
  explicit flags — never an invented estimate. Documented biases: growth clamping, noise, weak excitation,
  close modes can bias/hide poles; fitted energies are NOT an orthogonal decomposition (sums may exceed signal
  energy) and rank observability, not mechanical energy.
- Additive-spectral aperture: `SAMPLE_COUNT = 128`, `SAMPLE_RATE = 1000 Hz`, periodic Hann, bins 1..63
  (`spectral_features.py`).

### 3.2 Feature composition: 42 → 107 → 174 (exact reconstruction)

- **Base42** (`base_feature_row`, `p4_groups/two_mode_spatial_20260930/prepare_features.py`): 7 ridge envelope
  features + 15 modal (`FEATURE_NAMES`: 3 modes × (alpha, frequency, energy_fraction) + max_visible_alpha +
  energy_weighted_alpha + mode_count + relative_fit_error + usable_fraction + available) + 19 analytical
  (`PHYSICS_FEATURE_NAMES`: damping 5 = 3 energy-mode damping ratios + min-visible + energy-weighted, with
  `damping = −alpha/hypot(alpha, omega)`; stability 10 = 3 least-stable visible modes × 3 + visible count;
  spacing 4 = least-stable neighbor relative frequency gap, min visible relative gap, relative pole distance,
  neighbor_present) + 1 cross-correlation (`xcorr` sidecar). 7+15+19+1 = 42.
- **107** = 42 + 65 primary spatial (`modal_spatial_row`: unit-trace complex Gramian of least-stable ≥1%-energy
  mode: 16 channel powers + 24 pair real + 24 pair imag + availability; sensor graph asserts exactly 24
  neighbor pairs).
- **174** = 107 + 67 addition (`modal_spatial_addition_row`: second least-stable visible pattern 65 +
  squared normalized complex inner product gauge-invariant similarity + pair availability).
- The legacy clip working reference is a frozen 50/50 blend of 107/174-feature MLP (32→16) members,
  not a claimed production cutover. The latest trajectory reference is a separate174-feature framewise model.
- Damping/stability/spacing semantics: "least stable" = largest estimated growth among ≥1%-energy modes, NOT
  oracle flutter-mode identity. Frequency gaps are instantaneous, NOT velocity derivatives.

### 3.3 Temporal/relational models already live (COMPATIBLE)

- `TemporalMarginModel` (`temporal.py`): GRU (hidden 64) or TCN (hidden 32) backbone over `(B,16,T)` measured
  prefix + standardized side `(B,175)` = 174 features + log dt; ordered-quantile head (median ± softplus).
  No solver modes/velocity/design/target enter. GRU baseline (`baselines.py:377+`, ~25k params, Huber δ=0.1,
  design-grouped validation) retained as live candidate; review explicitly keeps GRU/TCN live and forbids
  claiming they are obsolete from MAE comparisons.

## 4. Failed spectral additions (exact record — NOT grounds to exclude the broader literature)

Fixed matched experiment `additive_spectral_20261009` (protocol + decision + verification + journal tail all
read; 15 fixed fits, 3 seeds, 2 held development designs, 2 fresh paired noise draws — NOT blind validation):

- Tested on 128-sample windows, dt=0.001, stride 64, all-174 retained:
  1. `summed_power_entropy = −Σ p log p / log(63)`, `p` = normalized summed-sensor power over bins 1..63;
  2. `fitted_frequency_neighborhood_mass` = union mass in nearest-bin ±1 around existing top-3 fitted-strength
     frequencies (lower tie-break, clipped 1..63);
  3. `broadband_spatial_concentration = Σ|Sij|²/total²`, `S = X X^H` summed over bins (code computes the squared
     Frobenius norm; equals `tr(S²)/tr(S)²` for the Hermitian S. Smoke: antiphase pair → 1, two orthogonal
     tones → 0.5, amplitude-invariant).
- Shuffled controls: appended values jointly permuted across physical trajectories within split (same
  permutation across paired views); matched dimension/capacity; NOT deployable models.
- **Decision (verbatim outcome):** neither addition adopted. Spectral family REJECTED (primary mean
  overall/boundary errors worsen; seed/profile/bin-transition and dangerous-error regressions; clean
  accepted-to-accepted signed-jump p95 0.084 vs baseline 0.025 on 214 transitions). Spatial concentration NOT
  adopted (small mean gains inconsistent across design/seed; shuffled size-matched control better on ALL 13
  held noise conditions for both all-frame and boundary; positive-on-unstable exposures increase, e.g. clean
  724/822 vs baseline 658/822 — pooled descriptive counts, not independent trials). Farther-stable and
  sign-exclusion coverage absent → unknown, not passed. Next campaign remains the controlled proper-TRAIN
  support expansion, not further feature search.
- [INFERENCE] failure attribution is unresolved: estimator artifact (bin quantization/fitted-frequency
  dependence), information redundancy with existing spatial features, and genuine information absence are all
  consistent with the record. Frontier must not treat this as a universal impossibility claim for spectral
  features.

## 5. Short-record causal multisensor estimator limitations (128–500 ms)

Applies to every COMPATIBLE estimator below; frontier must separate these estimator facts from model failures:

1. **Time–bandwidth quantization:** 128 samples at 1 kHz → 7.8125 Hz bins. Modes within ~1 bin interact;
   neighborhood-mass (±1 bin) and peak-picking inherit ±bin jitter. Bin-transition regressions in §4 are
   consistent with (not proof of) this artifact.
2. **Few oscillation cycles:** at tens of Hz, 128 ms holds single-digit cycles; weak growth accumulates
   little envelope change within the window. Judge sufficiency by cycles, growth and SNR, not sample count.
   Any quantitative damping-error scaling needs estimator-specific evidence; none is established here.
3. **Coalescence ill-conditioning:** near-flutter pole pairs make exponential bases nearly confluent;
   variable-projection/DMD refinement risks failure exactly where it is most wanted (Askham–Kutz §2.2.3,
   [WORKER-EVIDENCE] snippet-verified; full-text re-check pending).
4. **Cap truncation:** only the contiguous pre-cap prefix is fitted; saturated/late windows lose aperture
   (`usable_fraction` quantifies, does not repair).
5. **Correlated-noise stress:** rho=0.8 correlated noise is a demonstrated failure regime (N1 screen: full-
   covariance ID removes correlated5 stable severe optimism 6.4%→0.9% but raises false flags; §6.6 of review
   record). Unweighted Hankel covariance + LS shift is misspecified under correlated noise.
6. **Overlapping-window dependence:** overlapping windows share noise samples; never multiply window
   probabilities or treat windows as independent evidence (O1 protocol requirement).
7. **Calibration floor:** 51-sample leading calibration + per-channel RMS normalization ERASES absolute
   amplitude (mechanism-checked: `F(10x)/F(x)=0.1` raw → `1.0` after calibration). Any amplitude-as-stability
   cue is irrecoverable downstream — a contract fact, not a model defect.

## 6. Feature candidate inventory (constraint-indexed; NO usefulness ranking)

### 6.1 Multi-mode growth/frequency relations — COMPATIBLE

- **Per-mode (alpha, f, energy_fraction) top-3 + aggregates** (in probe): direct observables; energy-weighted
  alpha, max-visible alpha, mode_count, relative_fit_error, usable_fraction, available.
- **Damping ratios** `ζ = −alpha/hypot(alpha, 2πf)` (in probe): sign convention — negative = growing.
- **Frequency gaps / pole distances** (in probe spacing group): instantaneous separation only.
- **Candidate (not in probe): growth-weighted frequency dispersion** across visible modes; **modal density /
  participation entropy over fitted energies** (distinct from spectral-bin entropy: operates on ≥1%-energy
  fitted components, inherits ID uncertainty rather than bin leakage). COMPATIBLE (pure post-processing of
  estimated poles). [INFERENCE] mechanism: crowding/coalescence precursor; assumption: estimated poles resolve
  the interacting pair.
- **Candidate: envelope-curvature / quadratic log-envelope slope** (ridge `slope_quad`; carrier-curvature
  diagnostic: recovery 40/48→45/48 on 48 critical-only controls, noisy-stationary median growth error
  restored to 0.02286/s vs 0.05885/s allowance by noise-aware selection at cost of recovery 45/48→40/48 —
  [EVIDENCE] journal/target record). COMPATIBLE. Assumption: curvature distinguishes constant-growth decay
  from accelerating approach within one window.

### 6.2 Aerodynamic/structural invariants — COMPATIBLE with stated caveats

- **Routh/Hurwitz two-mode flutter-margin coordinates** (Bennett 1982 NASA-TM-84545, rendered pp. 1–5, 9–10
  per review §21 [WORKER-EVIDENCE/full-text-via-review]): for `p(s) = s⁴+c1·s³+c2·s²+c3·s+c4`, retain
  `Δ2 = c1·c2 − c3`, `Δ3 = Δ2·c3 − c1²·c4`. Where `c1 ≠ 0`, `Δ3/c1²` equals the published two-mode
  flutter-margin function. Tested in-repo as stability-coordinates probe (all three pairs among three
  least-stable visible measured modes + 21-column symmetric-product control; products/Hurwitz ranking heads
  change warning scores only; Hurwitz/correlated5 misses 80/324 near-unstable at nominal 1%, of which only 3
  had severe point error — [EVIDENCE] §22 diagnostic). Rules: NEVER divide by c1 (summed decay can vanish);
  `Δ3=0` alone does not certify neutral stability; pair poles share RMS-natural-frequency scale; no wind-off
  reference/airspeed/solver modes enter. COMPATIBLE (reorganization of measured poles, no new information —
  any gain is inductive bias, per review). Bennett's warning stands: frequency-only simplification was
  nonconservative in one case; both modes' damping AND frequency matter.
- **Jury discrete-time multimode parameters** (Matsuzaki 2011: abstract-only [WORKER-EVIDENCE], full-PDF
  retrieval failed — NO formula claim). RELAXATION-adjacent: motivation to retain >2 modes is COMPATIBLE;
  any specific Jury criterion is UNVERIFIED — do not assert its equation.
- **Classical Zimmerman–Weissenburger extrapolation** (1964; via Chajjed/Bennett descriptions): multi-speed
  flutter-margin trend. **RELAXATION** (needs target speed history) and, with known speeds, a different
  observation contract. Chajjed's intermediate margin function is NOT P4's `m` (review §12).

### 6.3 Mode-shape / phase / transmissibility / FRF distinctions — COMPATIBLE unless noted

- **Spatial Gramian patterns + complex pair entries + inter-mode similarity** (in probe §3.2): gauge-invariant
  (global scale/phase-free, no angle wrap); calibrated-channel collinearity, NOT physical coupling measurement;
  each mode has independent excitation scale/phase. Transport diagnostic [EVIDENCE]: spatial transport separates
  sensor-envelope growth from physical pole growth.
- **MAC** (Modal Assurance Criterion, `|φ1^H φ2|²/(…)`): cross-window / cross-design pattern tracking score.
  **MPC** (Modal Phase Collinearity): real-vs-complex mode indicator. [WORKER-EVIDENCE] snippet-level
  (uncertainty-quantification papers located, full text unread). COMPATIBLE as measured-pattern statistics;
  assumption: calibration preserves relative channel amplitude/phase (drifts/gains break it — gain_bias
  acquisition exists as stress). No P4-performance claim.
- **Transmissibility** `Tij(ω) = Xi(ω)/Xj(ω)` (Devriendt–De Troyer line; TOMA/OMAX: transmissibility-based OMA
  combined with exogenous-input OMAX [WORKER-EVIDENCE] snippets). Properties per literature: load-cancelling
  under common excitation; poles from transmissibilities alone do NOT generally equal system poles. Variants:
  per-bin complex transmissibility vectors; transmissibility coherence across windows; pole estimates from
  transmissibility differences across loading conditions. COMPATIBLE for single-window transmissibility
  features; cross-loading variants are RELAXATION (need second excitation condition). Assumption: reference
  channel observes the mode (weak reference → ratio noise).
- **FRF / input-output methods**: **PROHIBITED** as deployable (need measured excitation); OMAX-style
  exogenous-input arms allowed ONLY as separately declared excitation-envelope arms.
- **Resonance/antiresonance + phase-derivative features**: per-channel spectral peak/valley frequencies,
  peak prominence, −3 dB bandwidth, phase-slope `d∠X/df` at fitted frequencies, group-delay consistency
  across channels. COMPATIBLE (single-window FFT post-processing). [INFERENCE] mechanism: antiresonances
  encode sensor/node geometry complementary to pole locations; assumption: 7.8 Hz bins resolve them (often
  false at 128 ms — estimator limit, §5.1).

### 6.4 Spectral-matrix / subspace estimators (FDD / SSI / ESPRIT / DMD) — COMPATIBLE estimators

- **Current LS-ESPRIT** (§3.1) is the experimentally exercised member of this family.
- **FDD** (Brincker–Zhang–Andersen 2001 [WORKER-EVIDENCE] search-verified venue: Smart Mater. Struct. 10(3);
  full text unread here): SVD of spectral density matrix per bin; singular-value peaks ≈ modes; first singular
  vectors ≈ shapes. COMPATIBLE in principle, with an explicit spectral-averaging estimator and aperture.
  A single-segment per-bin XX^H is rank1; its singular dominance is not a modal-coherence discovery.
  Averaging/tapers trade frequency resolution against variance; verify the chosen estimator rather than
  assuming a reliable128ms FDD variant.
- **SSI / N4SID** (Van Overschee–De Moor [WORKER-EVIDENCE] search-verified book + Automatica N4SID paper;
  full text unread): stochastic subspace from block-Hankel. COMPATIBLE in principle; source-specific
  identifiability assumptions, short-record variance and order-selection instability remain to be verified.
- **Covariance-weighted ESPRIT (N1, [EVIDENCE] review §§22–24)**: quiet-record noise covariance, whiten for
  pole ID, refit coefficients on ORIGINAL channels; never subtract common mean. Single-fold screen:
  correlated5 severe optimism8.333%→0% versus frozen,7.576%→0% versus matched unweighted.
  Five-fold continuation:6.395%→0.872%, replication7.849%→0.872%; false flags increase.
  Executed quiet calibration was independently generated SYNTHETIC noise, not a hardware measurement.
  COMPATIBLE only if an actual quiet record or separately validated noise model is available within the
  declared calibration contract; otherwise needs a declared relaxation. Response-containing leading51
  samples are NOT that prerequisite. Known injected covariance remains a labelled diagnostic oracle.
- **Variable-projection pole refinement (N2 proposal, [EVIDENCE] review §22)**: `min_poles min_coeffs
  ‖X − Φ(poles)·C‖²`, shared conjugate pairs, channel-specific coefficients, exact linear inner solve;
  LS-ESPRIT init; fixed aperture/budget. Primary source: Askham–Kutz optimized-DMD manuscript §§2.2–3
  ([WORKER-EVIDENCE] link + section numbers from review; full-text re-check pending). COMPATIBLE offline
  reference; online claim needs full-pipeline timing (§1 gate: p95 ≤10 ms).
- **Envelope-function / AR / ARMA damping extrapolation** (Naylor-line; flight-test AFFTT survey
  [WORKER-EVIDENCE] snippets; Ueda Table 1: ARMA index overpredicts +41%/+24%/+11% vs wavelet ~0%/−10%/−4%
  on one wing — three runs, same structure, NOT transfer evidence): COMPATIBLE single-window AR damping
  features; multi-speed AR-trend extrapolation is RELAXATION.
- **Flutterometer / jitter on-line margin tracking** (Lind; flight-test evaluation [WORKER-EVIDENCE]
  snippets): RELAXATION (needs model + expanding envelope operation).

### 6.5 Local time–frequency observability — COMPATIBLE

- **Ueda–Iio–Ikeda 2009** ([EVIDENCE] full local manuscript, review §15): Gabor CWT Eqs. (2),(10)–(13); index
  `F = 1/Σ|W_real|` Eq. (14) at model-calculated 20.8 Hz band; lower-envelope selection + linear extrapolation
  to F=0 across speeds. P4-applicability restrictions [EVIDENCE]: fixed known band + speed sweep are EXTRA
  INFORMATION (RELAXATION/PROHIBITED as full method); raw amplitude irrecoverable after calibration (§5.7);
  CWT Eq. (13) uses `m±k` (streaming latency); paper's 2 s aggregation ≠ transferable sample count.
- **Local-observability experiment** ([EVIDENCE] review §§16–17): two-cycle-sigma complex-Gaussian bands at
  first/second least-stable fitted frequencies, complete support pre-cap; share quantiles/evolution/
  persistence + log-power-trend departure + availability. Result: correlated5% MAE −2.2% vs frozen with
  paired intervals excluding zero BUT seed-fragile (seed42 ≈ 0), no established warning gain, supported
  farther-stable worsening, Brier worsening in 3 families, and a decisive counterexample (D000093: lowest-
  quartile band residual yet optimistic error +0.024 on truth −0.0065, repeated on both draws). NOT adopted;
  retained as live bounded finding. Lesson for frontier: low local-fit residual is NOT sufficient evidence of
  trustworthy margin — any confidence rule from local consistency must survive D000093-class
  counterexamples. Audit finding: in 30 primary unstable-last1% severe acquisitions, the encoded nearest pole
  maps to the true critical pole with median normalized error 0.0022 — missing critical dynamics does NOT
  explain the tight-boundary cohort ([EVIDENCE] §17); mapping/coverage is the pointed suspect there, while
  broader severe cohorts still show ID fragility (critical0 match 80.8% vs 98.1%).
- **Neighborhood-reference result** ([EVIDENCE] §18): unchanged-feature k5 design-median reference repairs
  named failures but is a poor global rule. Supports testing interpolation/coverage before adding quality
  features.

### 6.6 Information beyond growth sign; uncertainty/quality — COMPATIBLE unless noted

- **Growth sign itself**: necessary but insufficient — local relation `alpha ≈ −kappa·margin` with
  design-dependent `kappa` unidentified by growth alone (target hypothesis; §1). Fitter6 vs framewise
  comparison [EVIDENCE]: fitter positive on ALL 274 truly-unstable held frames; native framewise on ~77–80% —
  MAE ≠ ranking ≠ signed headroom.
- **Quality/observability descriptors** (usable_fraction, fit error, energy fractions, availability, band
  support flags): COMPATIBLE as *inputs* to learned uncertainty; their presence is NOT calibrated confidence
  (D000093 counterexample). Any uncertainty claim must pass the calibration gates of §1 on held designs.
- **Conformal / risk-control tools** (Romano CQR; Angelopoulos Learn-then-Test; Gibbs–Candès ACI for
  distribution shift [WORKER-EVIDENCE] links+claims from review notes; full texts not re-read here):
  established tools, NOT novelty; grouped/rolling-window dependence + design shift forbid naive pooled-clip
  guarantees — calibration unit must be designs/trajectories with declared estimand. COMPATIBLE as methods;
  any coverage number needs the §1 statistical prerequisites.
- **SBI / neural posterior estimation** (Cranmer–Brehmer–Louppe review arXiv:1911.01429 [WORKER-EVIDENCE]
  snippet): amortized posterior / learned likelihood as implementation candidate, not automatic novelty.
  COMPATIBLE (trained on simulated population, applied to single observation). Assumption: simulator family
  covers deployment physics — the model-form-mismatch track (§7, S2/O1) is the credibility test.

## 7. Hidden-design growth-to-headroom calibration (the scientific bottleneck)

- **[HYPOTHESIS — target's central claim]:** joint temporal+spatial modal information constrains the
  design-dependent conversion `kappa` (growth→normalized-speed margin) beyond growth sign alone.
- **What is established [EVIDENCE]:** spatial features improve overall accuracy while harming boundary
  behavior (§1 table); local consistency reduces repeated-noise variance without recovering boundary
  accuracy; accurately encoded critical poles coexist with severe held errors (§6.5 audit) — i.e. extraction
  and conversion failures are empirically separable, and the 2026-10-08/09 diagnostics (headroom oracle,
  regularization probe, additive spectral) were DESIGNED to isolate extraction vs conversion before
  prioritizing corrections.
- **Candidate conversion mechanisms (all COMPATIBLE unless noted):**
  1. Population joint prior over (modal quantities → margin), TRAIN-fitted, with propagated ID uncertainty
     (Chajjed mechanism; full manuscript §§3–5 per review §12 [WORKER-EVIDENCE]: joint priors cut
     critical-speed COV ~44% vs flat on their task — uncertainty spread, NOT error/coverage; different target
     and 3-speed observation — no P4 ranking).
  2. Transferable frequency–growth / damping–frequency joint relation as soft constraint (Méndez mechanism;
     full manuscript §§4–6 per review §12: median damping-RMSE ratio ~3.9, 15→5 cycles at 30 dB SNR for
     >95% sign accuracy on their configuration; trajectory uncertainty + misspecification weaken it; hard
     wrong constraints flip signs → prefer probabilistic constraints). Needs population-learned relation, not
     their configuration's trajectory.
  3. Training-population physics-prior/library likelihood baseline on SAME measurements (review §11-B):
     the mandatory competitor that decides whether any learned converter beats physics + same data.
  4. Dimensionless intrinsic targets (e.g. `−alpha/omega_ref`) as gated upgrade (review Direction B): needs
     explicit consistent `omega_ref`; RELAXATION-free but changes the estimand — label as target change.
- **Bottleneck-vs-artifact rule for frontier:** oracle-true-modal success with measured failure supports an
  extraction hypothesis only under matched content/support/model controls. Oracle failure leaves converter,
  finite support, representation omissions and model capacity unresolved; it does NOT demonstrate an
  information limit. The completed six-feature/four-fit-design oracle cannot settle a richer multimode
  information question. Reuse its result rather than blindly repeat it. Same-information baseline ties
  weaken a learned-advantage claim, not the margin objective. Offline diagnostic features never include
  m/ucrit/−alpha/m or target speed-derivatives as predictor inputs.

## 8. Models and training approaches (on top of features; no ranking)

All COMPATIBLE unless a contract change is named. Baselines they must beat are same-information (§1).

1. **Temporal encoders (GRU/TCN, live):** `TemporalMarginModel` + raw GRU baseline (§3.3). Open questions for
   falsifiable tests: within-window dynamics vs sidecar-static mapping (ablate backbone vs side-only);
   384 ms-span variants (cross-window state, track-before-detect) LABELLLED as span change. Sampler artifact
   warning: legacy clips draw fresh modal initial conditions per clip — recurrent memory across stitched clips
   is meaningless; only continuous O1 trajectories test tracking.
2. **Relational multi-mode/observation models:** cross-mode attention/set encoders over visible-mode tuples;
   sensor-graph (24-pair) message passing; complex-valued coherence-preserving layers. Assumption: permutation
   symmetries (mode order, sensor permutation) handled; must beat the Ga-invariant handcrafted spatial baseline
   to claim architectural benefit (replaces-notion: same-information control with matched capacity).
3. **Latent system identification:** encoder→(poles,residues)→differentiable reconstruction + margin head
   (Méndez-style soft trajectory prior; variable-projection inner loop as differentiable layer). Assumption:
   gradient through pole parameterization stable near coalescence (known risk, §5.3).
4. **Physical/population priors:** joint TRAIN-fitted prior + MCMC/variational posterior over modal
   interpretation → margin distribution (Chajjed-style; single-observation likelihood, NO cross-speed coupling
   unless RELAXATION declared). Mandatory control: library/physics-prior likelihood with same measurements.
5. **Conditional calibration:** TRAIN-only oracle representation vs measured-feature calibration diagnostic
   (review §11/§7-c); design-conditional (not pooled) calibration maps; calibration designs disjoint from fit
   and blind sets.
6. **Estimator-aware uncertainty / domain randomization:** noise-model-matched augmentation (waveform-level
   iid/correlated per measured PSD, NOT arbitrary modal-column perturbations — Crowther lesson, review §14);
   quiet-calibration covariance (N1); O1 trajectory-level selective prediction with availability accounting
   (abstain-everywhere fails the ≥80% availability gate); design-level conformal/risk control (§6.6).
7. **Robustness / loss / training:** group-DRO/CVaR tail objectives with regularization + matched average-risk
   controls (Sagawa §§2–3 mechanism; S1 screen REJECTED as tested — worst-group CVaR gave fold-sensitive
   8.33%→7.58% with unstable-tail + false-flag + clean regressions [EVIDENCE] §23 — not a universal rejection
   of tail objectives); partial-AUC / low-FPR ranking heads for warning WITHOUT replacing the margin task
   (Zhu §§3–4 mechanism; W1 screen: nonlinear rank head helps misses 8.6%→0% at matched budgets but no
   passing 1% policy and no point-error change [EVIDENCE] §23 — ranking ≠ headroom); constrained training
   subject-to-guardrails (Cotter §§1,3–5); Neyman–Pearson thresholding (Tong–Feng–Li) and Learn-then-Test
   (Angelopoulos §§1–2) for calibration-stage risk control — all need design-level units. Report bias +
   exceedance curves so uniform shifts cannot masquerade as tail repair.

## 9. Constraint-relaxing candidates explicitly retained (do NOT treat as authorized)

The single-operating-point restriction is relaxed for the dispatcher's 384 ms experiment ONLY as stated in §2. None of
the following is automatically authorized: measured velocity input, extra aperture/sensors, target-specific
calibration, excitation actuation, or speed-sweep history.

| # | Candidate | Needs | Why retained |
|---|---|---|---|
| R1 | Multi-speed flutter-margin extrapolation (Zimmerman–Weissenburger; Chajjed 3-speed posterior) | Target operating-point history (± speeds) | Classical baseline; quantifies what single-observation gives up |
| R2 | Crowther-style damping/frequency history + current speed (5-point, 15-knot, +45-knot forecast) | Speed history + measured/derived modal history | Only neural precedent with held-out structural variation; noise-aware-training lesson |
| R3 | Configuration-specific frequency–growth trajectory constraint (Méndez) | Independently established trajectory for the TARGET | Strongest short-record mechanism; misspecification study included |
| R4 | Known-flutter-band energy index + cross-speed extrapolation (Ueda full method) | Target analysis band + multi-speed campaign | Best amplitude-band precedent; envelope-selection lesson |
| R5 | Controlled re-excitation / adaptive extended acquisition (O3) | Actuation + delay accounting | Manages unreliable decisions when free decay dies into noise |
| R6 | Aperture beyond .5 s; >16 sensors | Contract change | Observation-budget pivot only if insufficiency is demonstrated; 384–500 ms remains within the existing ceiling |
| R7 | Explicit geometry/load inputs (aspect ratio, loads — Wang–Ma §4.6 style) | Measured structural data at deployment | Diagnostic of WHICH hidden variation breaks calibration (TRAIN-only allowed; deployment input is the relaxation) |

## 10. Prohibited extra-information / oracle methods (diagnostic counterfactuals only)

VelocityOracle (`(ucrit−v)/ucrit` — perfect-predictor upper bound, in codebase); TRAIN-only true-modal oracle
representation (matched-content diagnostic, never an input); known injected-noise-covariance N1 arm;
hard true-margin sign override; solver-pole/critical-identity routing; future-response or concatenated-clip
"extra history"; `V = ratio·ucrit` sampler shortcut (known exposed leak pattern — must be avoided, §22/S2).
Oracle arms quantify headroom, never qualify predictors.

## 11. What frontier needs: evidence/assumption pairs for falsifiable experiments

Each row gives frontier a controlled intervention, the discriminating prediction, and the target-linked
success/failure reading — WITHOUT pre-selecting winners:

1. **Extraction-vs-conversion diagnostic (§7):** measured/oracle matched-content contrasts discriminate
   extraction only when converter/support adequacy is independently controlled. If both fail, do not
   infer nonidentifiability: test conversion/support/model explanations. Use completed oracle evidence;
   any new counterfactual must identify missing content or a genuinely different hypothesis.
2. **N1 covariance weighting:** predicts correlated-noise tail repair WITHOUT clean-mapping change; failure on
   clean accurate-pole cases is EXPECTED (not a rejection) — those implicate conversion (§7).
3. **Local-observability confidence:** any rule MUST classify D000093-class cases (low residual, optimistic
   error) as unreliable, else it is falsified as a reliability criterion.
4. **Spectral candidates:** must beat matched shuffled/size controls AND preserve dangerous-error guardrails;
   §4 shows mean-gain-without-control is insufficient evidence. A prespecified 128 versus 384–500 ms
   comparison stays within the contract; beyond .5 s requires explicit relaxation R6.
5. **Warning heads:** report actual held false-warning rates + timely-miss tradeoff at BOTH nominal budgets;
   AUC-only or point-MAE-only success is a category error (fitter6/framewise lesson).
6. **384 ms-span methods:** label span; compare against single-window at EQUAL span or state the span
   advantage explicitly; shared-noise accounting mandatory (no independent-window multiplication).
7. **Sampler-artifact guards:** V-shortcut check; fresh-initial-condition vs continuous-state distinction;
   repeated-noise-view non-independence; design-level (not clip-level) resampling everywhere.

## 12. Primary sources and verification status

- **Re-read directly for this report:** `modal_features.py` (all 279 lines), `baselines.py` (all 479),
  `temporal.py`, `sidecar.py` (members/extract path), `operational.py` (acquisition core),
  `p4_groups/…/spectral_features.py` (full), `additive_spectral_20261009/{protocol,decision,verification}.json`,
  `two_mode_spatial_20260930/prepare_features.py` (acquisition loop), `P4_RESEARCH_TARGET.md` (full),
  `RESEARCH_REVIEW.md` §§11–14, 20–24 + surveyed excerpts of §§6–10.
- **Full-text comparisons on record (review §§12–15, 21; worker did not re-open PDFs):** Chajjed
  (arXiv:2210.15667 §§3–5, Table 2), Méndez (arXiv:2609.25440 §§4–6, Tables 2–3), Crowther–Cooper
  (DOI 10.1243/0954410011531736 §§2–3, 5–7; local `p4_journal/papers/crowther2001.pdf`), Wang–Ma
  (DOI 10.3390/aerospace11080677 §§2, 4.2–4.6, 5, Tables 3–4; local PDF), Ueda–Iio–Ikeda
  (DOI 10.2322/tjsass.51.275, all 7 pp.; local PDF), Bennett (NASA-TM-84545, rendered pp. 1–5, 9–10).
- **[WORKER-EVIDENCE] search-located, snippet-only (full text NOT verified):** FDD venue/record, N4SID book +
  Automatica paper, ESPRIT IEEE record, Devriendt/De Troyer transmissibility papers, Naylor AR studies,
  flutterometer reports, Dowell/piston-theory reviews, Cranmer et al. SBI review (arXiv:1911.01429), Gibbs–
  Candès ACI, Sagawa et al. group DRO (arXiv:1911.08731), Askham–Kutz optimized DMD (sections per review),
  MAC/MPC uncertainty papers, Koopman manuscript (arXiv:2505.14697v1), Zastrow operator-inference manuscript.
- **Abstract-only, no method/result claim permitted:** Wang–Zhou 2026 (10.2514/1.C038983), Shi et al.
  (10.2514/1.C038429), Zheng EMD+CNN (per review §11 table), Matsuzaki 2011 (retrieval failed).
- **In-repo executed evidence:** §§16–18 local-observability/audit/neighborhood experiments; §§22–24
  W1/S1/N1 screens + five-fold continuation + O1 trajectory protocol/runner; headroom-oracle, curvature,
  noise-aware-carrier, fitter-comparison, additive-spectral checkpoints (all cited with artifact paths above).

## 13. Coverage limitations of this report

1. Worker histories ended before drafting: no worker-authored synthesis existed to recover — this report was
   composed from worker tool-output evidence + the re-reads of §12. Any worker interpretation beyond captured
   outputs is lost.
2. Web-search literature (§12 snippet-only) was located but not full-text-verified; equations/sections for
   FDD/SSI/ESPRIT/TOMA/conformal/DRO/DMD/MAC/MPC are NOT asserted beyond the cited review records and
   snippets. Frontier must re-verify any equation it uses.
3. `RESEARCH_REVIEW.md` §§1–5, 7–10 were surveyed via excerpts only; reserved-split/O1-implementation details
   post-§24 and any journal entries after the worker snapshot are not covered.
4. No new PDFs were opened and no P4 performance inference was drawn from literature — by design (§1 contract).
5. The completed 384 ms protocol, results and verification were supplied by the dispatcher after this survey
   was drafted. The separate review packet supplies those outcomes; this survey does not claim a history gain.
6. Sibling report (working-history) is separately owned; overlap here is limited to what the constraint survey
   needs. Coordination message sent to `WorkingHistoryReport` to avoid filename collision.
