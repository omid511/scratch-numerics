# Aeroelastic and Structural Modeling of Honeycomb Sandwich Plates: Four ML Frameworks (Updated v3)

This is the fourth revision. Each proposal now includes: the finalized architecture recommendation (informed by independent second-opinion review), a short **project charter** (primary claim / MVP / upgrade path / dependencies / stop condition), and updated cross-cutting notes including an explicit execution order.

---

## Program Overview

**Research theme:** physics-informed probabilistic machine learning for aeroelastic analysis, uncertainty quantification, structural health monitoring, and robust design of honeycomb sandwich structures.

All four proposals build outward from the same fast FSDT solver, each addressing a different limitation or extension of it:

```
                        Fast FSDT solver (0.3s/run)
                                  │
        ┌─────────────────┬──────┴──────┬─────────────────┐
        │                 │             │                 │
   Proposal 1         Proposal 2    Proposal 3        Proposal 4
   Improve fidelity   Inverse       Robust design      Online monitoring
   (measure the       damage        under boundary     (early-warning
   design-dependent   inference     uncertainty        margin estimation
   FSDT/FEM gap)      (localize     (mode-veering-     from transient
                      damage from   aware reliability  response)
                      sparse        surrogate)
                      sensors)
```

Each proposal produces a standalone result on its own (see individual project charters below), but together they form one research narrative: a corrected, uncertainty-aware solver (P1) that can be queried safely under uncertain boundary conditions (P3), whose designs can be monitored for both internal damage (P2) and approaching instability (P4) once deployed. The integration story (P1 feeding P2/P3; P3 and P4 pairing as offline/online safety margins) is the layer to build out if time allows — see Execution Order below for how the individual results sequence toward it.

---

## Proposal 1: Multi-Fidelity Correction Field via Latent-Space Gaussian Process

**Current status:** completed, bounded five-variable COMSOL/FSDT study with
prospectively blinded evaluation. See the [final P1 report](P1_IMPROVEMENT_REPORT.md),
[historical audit](P1_RESULTS_REVIEW.md), and
[source/data availability contract](P1_REPRODUCIBILITY.md).
The broader original proposal is not declared complete by this result.

### 1. Scientific Justification and Contribution

The executed LF model already uses layerwise modified shear-correction
factors from laminate integrals; it is not a constant-global-5/6 model.
The learned discrepancy therefore cannot be attributed solely to replacing
that factor. The source paper's approximately four-percent fixed-case
comparison is a benchmark, not a uniform design-space error bound.

The contribution is an explicitly audited correction experiment:
geometry-grouped learning, guarded mode identities and quarantine records,
train-only representation/regression diagnosis, matched HF budgets,
independent calibration, genuinely prospective evaluation and an
HF-verified frequency-screening task. PCA/GP, co-kriging and neural INR
architectures are established methods; no generally superior new
architecture or uniformly epistemic uncertainty is claimed.

### 2. Executed Physical Scope and Data

- Five geometric variables: `alpha`, `beta`, `theta_c`, `eta1`, `eta2`.
  Fixed 0.3-m plate length, aluminium material and CCCC configuration.
- LF: implemented FSDT/Legendre-Ritz solver, order15, 16 flexural modes,
  80x80 top-surface field grid and penalty-spring CCCC treatment.
- HF: explicit-cell COMSOL shell model, nominal hauto4, 32 eigenvalue
  candidates/16 guarded flexural exports, 80x80 grid and shift1000Hz.
  The original physical builder and solver were not changed for prospective data.
- Historical campaign: 10,000 LF designs and80 HF runs, with48 training,
  16 calibration and16 historical test geometries;721 accepted mode rows.
  Dense pixels do not increase the number of independent design observations.
- After freezing means, scale recipes, sources, reference gauge and threshold:
  32 new calibration,32 new IID test and16 theta>=60 challenge geometries.
  All80 native acquisitions succeeded. Pairing accepts295/320 calibration
  and448/480 evaluation rows; quarantines remain explicit.
- IID calibration/test are independent physical-uniform draws within the
  original bounds. The regional model is trained/tuned only below60 degrees
  on34 historical training geometries and calibrated on23 eligible new
  geometries. Its challenge is distribution-shift diagnosis, not guaranteed
  extrapolative coverage. The global model did not exclude that region.

### 3. Implemented Models and Promotion Decisions

The frozen global mean remains **boundary-enforced linear PCA48 plus
per-mode ARD RBF GP correction**, with a separate per-mode relative-frequency
GP. This is not a neural autoencoder. The perimeter factor enforces zero
transverse correction, not exact normal derivatives, exact CCCC total
fields, a corrected stiffness/mass operator or eigen-consistency.

Direct-HF, original PCA–GP, physical-output PCA–GP, co-kriging and an actual
coordinate-conditioned PyTorch neural INR were fitted and evaluated.
The genuine INR conditions on design, stable mode identity and LF inputs;
the historical Fourier-feature ridge proxy remains separately labelled.

Train-only geometry-grouped validation decomposes decoder representation
and regression errors, including the MSE cross term. No new global mean
passes the predeclared promotion gate, despite better physical-output PCA
oracle reconstruction. Regional co-kriging/RBF passes its in-region gate.
Local residual-based scales fail both UQ promotion gates, so raw scales remain
selected. No final-test result is used to change these decisions.

### 4. Validation, Measured Results and Engineering Task

All90 matched-budget trials complete across HF budgets8/16/24/32/48 and
three nested repeats. Tuned families receive identical train-only splits
and two configurations per family within each counted budget. These curves
use historical test cases; repeat extrema are not confidence intervals,
and no universal HF-efficiency winner is established.

On32 untouched IID geometries, geometry-median field RMS decreases from
**0.01476687 LF to0.005104386 selected PCA–GP**, and frequency error from
**4.585452% to0.9468742%**. Pooled accepted-mode-row medians are different:
selected field RMS0.00559255 and frequency error1.012241% over298 rows.
Complete comparator/tail metrics are in the committed text results.
Candidates that score better on evaluation medians are not post-test promoted.

Independent run-simultaneous selected-mean joint coverage is29/32 at90%
and32/32 at95%; the95% mean joint field/frequency widths are0.835769 and
2833.78Hz. Proper scores, discrimination, referral curves and finite-sample
intervals remain load-bearing: broad intervals and weak ranking do not
establish uniformly useful or purely epistemic uncertainty.

The separately calibrated95% frequency policy screens stable reference
label1 against the frozen961.333328-Hz benchmark. IID:4/32 HF referrals,
zero observed false acceptances/rejections and28 verified correct automatic
decisions, versus24/32 referrals for calibrated LF-only screening.
Challenge:4/16 referrals, zero false acceptances and one false rejection.
The task is not a service-safety certificate or a complete-spectrum
fundamental-frequency guarantee. The validation campaign acquired every HF
reference; saved calls/time are counterfactual operating savings with setup
costs retained, not net campaign-cost savings.

### 5. Claim Boundaries, Reproducibility and Remaining Extensions

Mass-weighted sampled displacement/director two-mode spans support modal
mixing for2209/3408. Thickness, offsets, coupling, rotary inertia and
quadrature sensitivity are included, but individual-field convergence
failures remain. No full-DOF or campaign-wide reference-error certificate,
success on quarantined identities or OOD coverage guarantee is asserted.

The original proposal contemplated Abaqus, broader boundary/aerodynamic
variables, learned autoencoders and active-learning/operator extensions.
Those are not completed by the fixed-material, fixed-CCCC COMSOL result;
damage, aerodynamic and experimental validation require separate evidence.

The GitHub handoff publishes P1 source, meaningful tests, recorded
environment, Markdown and checksum-identified JSON/CSV results only.
Per the publication restriction, **no MPH simulation files, ZIP archives,
raw bulk arrays or fitted binary weights are uploaded**. Numerical arrays,
fitted models and extracted state arrays remain unpacked locally. Authorized
cleanup removed native MPH files and derived/bookkeeping outputs: LF-only
replay remains available, but saved-MPH re-extraction requires recovery or
new solves. See [local reproduction](P1_REPRODUCIBILITY.md#local-unpacked-study--no-zip-required).

---

## Proposal 2: Probabilistic Inverse Damage Identification via Latent-Space Posterior Inference

### 1. Scientific Justification (unchanged)
Multiple distinct damage patterns can produce nearly identical vibration signatures — a genuinely ill-posed inverse problem. A model that returns a full posterior over plausible damage states, rather than one point estimate, is what the problem actually calls for.

### 2. Data Strategy (unchanged from v2)
- Generate generously: 20,000–50,000 samples from the FSDT solver (cheap).
- Enrich the conditioning signal beyond 5 scalar frequencies — use mode shape data or transient features from the full w(x,y,t) output where available.
- Reserve a subset of Proposal 1's Abaqus runs as a genuinely held-out test set to avoid the inverse-crime problem (testing on data drawn from the same simulator used for training).

### 3. Proposed Architecture (finalized — supersedes the "explicit patches" plan from v2)

**The variable-damage-count problem killed the fixed-patch-count design from v2** (an arbitrary patch ordering creates an identifiability issue the model shouldn't have to resolve). The fix is to stop hand-designing the damage representation and let a network learn a compact latent representation directly from realistic damage fields:

```
measurements (frequencies, mode shapes) → measurement encoder → conditioning vector
                                                                        ↓
                                                          posterior network → distribution over latent z
                                                                        ↓
                                                              learned decoder → damage field d(x,y)
```

Whether one damage region or three exist, and where they sit, becomes something the decoder discovers from the training data's structure — not something the architecture has to be told upfront via a fixed patch count.

**Keeping the latent space physically meaningful.** Left unconstrained, the latent code is only guaranteed to decode correctly — nothing forces it to correspond to anything physically sensible, which is a reasonable concern for reviewers. Address this with one (not all) of the following, matched to how damage is actually expected to behave:
- **Sparse priors** on the latent code (e.g. an L1 penalty, or a spike-and-slab-style prior) — appropriate given damage is expected to be localized, encouraging most of the latent capacity to sit near zero except where real damage exists.
- **Smoothness priors** on the decoded field (e.g. a total-variation or Dirichlet-energy penalty) — appropriate if damage boundaries are expected to be gradual rather than sharp.
- **Physics-inspired regularization** — e.g. penalizing decoded fields that violate known constraints (severity bounded in [0, 1], spatial extent bounded by plate dimensions).
A sparse prior is the most natural starting choice given the localized-damage assumption already built into the data generation; the others are worth mentioning as alternatives but not necessary to implement all at once.

**Ranked options, in order of recommendation:**
1. **Latent-space posterior (learned decoder + cINN or flow over the latent)** — first choice. Solves the variable-count problem cleanly, keeps millisecond-scale inference (important given the eventual real-time monitoring goal), and is the most sample-efficient of the three.
2. **Plain image-space conditional diffusion (UNet over the full field)** — second choice if willing to trade inference latency (tens–hundreds of ms rather than ms) for zero representation-engineering — handles any number/shape of damage regions natively, no latent design decision required.
3. **Physics-guided diffusion** (diffusion + a correction term from the differentiable simulator during sampling) — stretch goal once option 1 or 2 works; adds real training complexity, best attempted after a simpler version is validated, not as a starting point.

### 4. Validation (unchanged from v2, extended)
Simulation-based calibration (SBC): draw known damage parameters, simulate resulting measurements, sample from the posterior, check the true value falls in the claimed confidence region as often as claimed. Test on the held-out Abaqus subset, not just FSDT-generated data.

### 5. Project Charter
- **Primary claim:** a latent-space posterior model can localize and quantify damage from sparse vibration measurements with calibrated uncertainty, at inference speeds compatible with real-time use, and does so without requiring the damage-region count to be fixed in advance.
- **MVP:** train the encoder/decoder on synthetic damage fields (using a simple generative process for realistic localized damage), fit a flow or small cINN over the latent, validate via SBC on FSDT-only data.
- **Upgrade path:** if SBC reveals poor calibration or the latent representation misses realistic damage shapes, move to option 2 (image-space diffusion) rather than tuning option 1 indefinitely. Physics-guided diffusion (option 3) only if option 2 succeeds but robustness to simulator mismatch is still lacking on the held-out Abaqus test set.
- **Dependencies:** the held-out Abaqus test set depends on Proposal 1's HF runs existing first — sequence Proposal 1's data generation before Proposal 2's final validation step (not before Proposal 2's development, which can proceed on FSDT data alone).
- **Stop condition:** MVP + one upgrade (if SBC or held-out testing specifically motivates it) is sufficient scope for this direction.

---

## Proposal 3: Robust Design Under Stochastic Boundary Conditions — Mode-Aware Reliability Surrogate

### 1. Scientific Justification (unchanged, sharpened)
Boundary stiffnesses vary in practice; a robust design needs to maintain a flutter safety margin across that uncertainty. The known complication is mode veering — a real physical phenomenon where two competing instability modes interact, producing a critical threshold that can develop sharp transitions or kinks as a function of the inputs, exactly where a plain smooth-kernel surrogate is least trustworthy and exactly where the safety margin is tightest.

### 2. Proposed Architecture (finalized, with an explicit decision gate)

**Primary path, pending a feasibility check:** if the solver can expose which physical mode is dominant (not just the scalar critical threshold), the kink can be avoided entirely rather than modeled around. Learn each mode's response as a separate, individually smooth surrogate, f_A(x) and f_B(x), and combine analytically:
$$\lambda_{cr}(x) = \min\big(f_A(x), f_B(x)\big) \text{ (or max, depending on the physical convention)}$$
Active learning then samples where the two mode predictions cross or where mode identity is uncertain — directly refining the transition surface rather than fighting a global kink. This is a substantially better inductive bias than any single global surrogate, **but it depends entirely on whether mode-level output is actually extractable from the solver** — check this before committing.

**Fallback path, if mode extraction isn't feasible:** gradient-enhanced Gaussian Process (using solver differentiability, as before), combined with:
- **Reliability-oriented active learning** — sample where predicted failure probability is near 50% (the actual decision boundary you care about), not just where variance is highest.
- **Transition detection** — monitor gradient/curvature magnitude during active learning; a spike signals proximity to a veering region and triggers denser local sampling, rather than assuming smoothness everywhere by default.

**Either path — a scope-limiting principle for the final optimization step:** don't require the surrogate to perfectly represent the kink across the entire input space. Use the surrogate as a global guide to propose candidate designs, then validate/refine near the actual candidate optimum using direct solver evaluations (with gradients) rather than trusting the surrogate at the final decision point. This is also the direct mitigation for the surrogate-hacking risk raised earlier (a gradient-based optimizer exploiting the surrogate's blind spots) — the real solver has the final word on any candidate design before it's accepted.

**Explicit design objective (carried over from v2):** e.g. minimize weight subject to P(λ_cr < λ_required) ≤ ε.

### 3. Data Strategy (unchanged from v2)
200 initial FSDT runs (space-filling DOE, sized against the actual input dimensionality), augmented via active learning — genuinely iterative here since the solver is fast, not gated by Abaqus turnaround.

### 4. Validation (unchanged from v2)
Coverage tests on held-out FSDT/FEM values; small-scale brute-force comparison on a reduced-dimension slice where exhaustive Monte Carlo is tractable.

### 5. Project Charter
- **Primary claim:** a gradient-enhanced, mode-aware (or transition-adaptive) surrogate can identify a robust design meeting a stated reliability target using substantially fewer solver evaluations than brute-force Monte Carlo, with honest uncertainty near the mode-veering region specifically.
- **MVP:** the fallback path (gradient-enhanced GP + reliability-oriented active learning + gradient-spike transition detection) on the full parameter set, validated on a reduced-dimension slice against brute-force MC. This is the safer starting point regardless of mode-extractability, since it doesn't depend on a solver capability that hasn't been confirmed yet.
- **Upgrade path — explicit promotion criterion:** attempt the mode-decomposition path only if (a) the solver can expose mode identity, confirmed early via a small feasibility check, and (b) the MVP's coverage tests specifically show degraded calibration near the transition region, indicating the fallback's kink-handling is the actual bottleneck.
- **Dependencies:** shares active-learning infrastructure with Proposal 1 (design-space sensitivity/sampling machinery); doesn't require Proposal 1 or 2 to be complete first.
- **Stop condition:** MVP result, validated against the brute-force slice, is a complete deliverable on its own; the mode-decomposition upgrade is worth attempting only under the explicit gate above, not by default.

---

## Proposal 4: Continuous Stability-Margin Estimation from Transient Response

### 1. Scientific Justification (revised — this is the biggest change in this revision)
The original framing (binary early-warning classifier) throws away most of the available signal. Because the simulator provides an exact, continuous distance-to-instability label at every timestep of every run — not just a pass/fail label at the end — the problem should be framed as **continuous margin estimation**, not classification. This makes fuller use of the one thing this proposal's data uniquely offers (labels are available everywhere, not just at failure), which a binary framing discards.

**Deployment framing, stated explicitly:** this proposal is designed for eventual **online structural health monitoring / digital-twin deployment** — a live model tracking a specific panel's real-time margin during operation, not a one-off offline analysis. Stating this explicitly (rather than leaving it implicit) is what connects Proposal 4 to Proposal 3 as a coherent pair: Proposal 3 establishes a design's safety margin ahead of time (offline, per-design), Proposal 4 tracks a specific in-service panel's margin as conditions and accumulated wear change it (online, per-instance). Framing it this way also motivates the domain-randomization data strategy below, since a deployed monitoring system is exactly the setting where simulation-reality mismatch matters most.

### 2. Data Strategy (unchanged from v2)
Uses only the fast FSDT solver — no Abaqus dependency. Generate transient response clips at a range of airflow speeds approaching (and some exceeding) each design's known critical pressure. For sim-to-real transfer, apply **domain randomization** during data generation: vary sensor noise, damping, calibration error, and sampling frequency across training runs, since real deployment will differ from clean simulation in exactly these ways.

### 3. Proposed Architecture (revised primary task, same backbone)

**Primary task — regress a normalized "distance to instability," not classify safe/unsafe:**
$$\text{margin}(t) = \frac{u_{\text{crit}} - u(t)}{u_{\text{crit}}}$$
This keeps the same lightweight sequence backbone discussed previously (causal 1D-CNN / TCN, chosen for streaming-friendly, low-latency inference) — the change is in the training target and loss, not the network architecture itself.

**Output as a distribution, not a point estimate:** predict a calibrated interval or distribution over the margin (e.g., via quantile regression or a probabilistic head), so a downstream user can distinguish "8% margin ± 1%" from "8% margin ± 8%" — this is reported as the safer choice for sim-to-real transfer specifically, since simulation-reality mismatch tends to show up as appropriately widened uncertainty rather than a silently wrong point estimate.

**Decision layer kept separate from the estimation task:** convert the continuous margin (and its uncertainty) into a warning/alarm using a downstream sequential decision rule, rather than folding the warning threshold into the training objective itself. This separates three questions that a binary classifier conflates: *how close* (inference), *how sure* (uncertainty), and *should we warn now* (decision).

**Worth checking, not required for the MVP:** if the solver can expose an intrinsic stability quantity (e.g. an eigenvalue real part or damping ratio that goes to zero at the instability) rather than just the raw control parameter, that may be a more transferable regression target across different designs and into real deployment — worth a quick check of what the solver already computes internally before committing to the raw-parameter target.

### 4. Validation (revised)
- Coverage tests on the predicted margin intervals (does a claimed 90% interval contain the true margin ~90% of the time on held-out runs).
- Precision/recall at varying lead times, computed *downstream* of the regression (i.e., derived from the continuous margin estimate via the decision layer), not trained directly.
- Compare against a simple baseline (e.g. amplitude-growth-rate threshold).

### 5. Project Charter
- **Primary claim:** a continuous margin-estimation model, trained on simulated transient response with domain randomization, gives earlier and better-calibrated warning of approaching instability than a binary classifier baseline, and degrades gracefully (via its own uncertainty) under simulation-reality mismatch.
- **MVP:** TCN backbone regressing normalized margin with a quantile/probabilistic head, trained on FSDT-generated clips with domain randomization, validated via coverage tests and lead-time precision/recall against the binary-classifier baseline.
- **Upgrade path:** the intrinsic-stability-coordinate target (eigenvalue/damping ratio) is worth attempting only if the MVP shows poor transfer across designs when using the raw control-parameter target — a specific, checkable failure mode, not a default upgrade.
- **Dependencies:** shares design-sampling infrastructure with Proposal 3 (same underlying parameter sweep, different extracted output — full time series vs. single threshold); not blocked by Proposals 1 or 2.
- **Stop condition:** MVP is a complete, standalone deliverable; the intrinsic-coordinate upgrade and any real sensor validation are explicitly out of scope unless the promotion criterion above is met.

---

## Cross-Cutting Notes (updated)

**Shared infrastructure — build only where at least two proposals actually need it:**
- **Proposal 1 and Proposal 2 share the same underlying pattern** — spatial field → compact latent → probabilistic model over the latent, conditioned on something (design parameters for P1, measurements for P2). The encoder/decoder architecture, the boundary-condition-enforcement trick, and a fair amount of training pipeline code can genuinely be shared, not just conceptually similar — worth building once and reusing, since both proposals need it independently regardless of the other's progress.
- **Proposal 1 and Proposal 3 share active-learning/design-sampling machinery** (choosing where to query an expensive or semi-expensive process next).
- **Proposal 3 and Proposal 4 share the underlying design/parameter sweep** — same sampling of the input space, different extracted output (threshold value vs. full time series).
- **Proposal 1's Abaqus runs double as Proposal 2's held-out inverse-crime test set** — sequence P1's HF data generation before P2's final validation, though P2's development doesn't need to wait on this.
- Per the "build duplication first" principle: don't build a generalized shared framework for any of the above before at least two proposals have independently hit the need for it. Data loading, experiment config, evaluation metrics, and logging are the exception — worth centralizing early since they pay for themselves regardless of which proposals proceed.

**Sequencing and scope — the program is a portfolio, not four independent projects:**
- Each proposal above now has an explicit MVP and a gated upgrade path — the sophisticated versions (latent+GP, mode-decomposition, physics-guided diffusion, intrinsic-coordinate regression) should only be pursued if the corresponding MVP specifically exposes the limitation that version addresses, not by default.
- Avoid a linear dependency chain (P1 → P2 → P3 → P4). As scoped, only two real dependencies exist: P2's *final validation* needs P1's Abaqus data, and P3/P4 share sampling infrastructure but neither blocks the other's core development. This keeps the program resilient — a delay or setback in one proposal doesn't stall the others.
- Two decisions still gate data generation and should be settled before that step, as before: where to spend the Abaqus HF budget in P1 (via sensitivity analysis), and confirming whether mode identity is extractable from the solver for P3 (determines which of the two P3 paths is realistic).
- Given four proposals each with real depth now (each MVP is nontrivial on its own), it's worth treating "core MVP for all four" as the realistic primary target, with upgrades and the cross-proposal integration story (P1 feeding P2/P3, P3 pairing with P4) as the layer to pursue only if time remains — that integration layer is what turns four separate results into one coherent program, but it's the right thing to drop first if the schedule gets tight, not something to design around from day one.

**Execution order:**
1. **Proposal 1 first, and start it immediately.** Not because its outputs happen to be reusable (though they are — see below), but because it's the one proposal gated by a hard external constraint: Abaqus turnaround is on the order of days per batch. Every other proposal runs on the fast FSDT solver and can start in parallel without waiting on this, but if P1's HF data generation doesn't start early, it becomes the critical-path bottleneck for the whole program regardless of how the other three proceed.
2. **Proposal 3 in parallel with Proposal 1**, since it depends only on the fast solver and shares no blocking dependency with P1.
3. **Proposal 2 follows**, developed on FSDT data from the start, with its final validation step (the held-out inverse-crime test) slotting in once P1's Abaqus runs exist.
4. **Proposal 4 last of the four to formalize**, since it can reuse P3's parameter-sweep infrastructure directly and extend it to transient-response extraction rather than building sampling machinery from scratch.

This order minimizes duplicated effort while keeping the proposals as independent as the real dependencies allow — only P1's HF generation is a true hard-start constraint; everything else is a scheduling convenience, not a blocking requirement.
