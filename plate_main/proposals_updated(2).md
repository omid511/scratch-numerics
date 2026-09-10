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

### 1. Scientific Justification
The LF implementation does not use a constant global ``kappa = 5/6``.  It
computes layerwise modified shear-correction factors using the
Vlachoutsis-style laminate integrals; the baseline factor is approximately
0.16875 for the current reference laminate.  Proposal 1 therefore learns the
remaining discrepancy between this implemented FSDT model and the explicit
shell reference, rather than attributing the whole gap to replacing 5/6.
The source paper's fixed CCCC comparison cases report an approximately 4%
FSDT-versus-FEM difference; that number is a benchmark, not a design-wide
pilot result.  The correction model must measure, rather than assume, how the
error varies across the sampled design conditions.

**Positioning against existing multi-fidelity surrogate work:** this is not generic multi-fidelity regression (fitting a cheap-to-expensive mapping for its own sake). The contribution is a **physics-consistent correction field with calibrated uncertainty** — boundary behavior is enforced by construction rather than learned, and the uncertainty is specifically epistemic over the design space (grows outside sampled conditions) rather than a generic residual-noise estimate. That distinction should be stated explicitly wherever this proposal is written up, since it's what separates it from a standard multi-fidelity Kriging/co-kriging baseline.

### 2. Data Strategy (unchanged from v2)
- **LF dataset:** 10,000–20,000 samples from the 0.3s FSDT solver.
- **HF dataset:** 50–100 Abaqus runs, spent deliberately using a sensitivity analysis on the LF solver first, Latin Hypercube weighted toward the parameters that actually matter, one well-designed batch (not iterative — Abaqus turnaround is days, not hours).
- Full varied-parameter list (boundary stiffnesses k₁–k₅, cell wall thickness, cell angle θc, independent face/core layer thicknesses, aerodynamic pressure) should be written down explicitly, since this — not an assumed dimensionality — determines HF sample adequacy.
- Split by HF run, not by pixel, when forming train/val/test sets, to avoid leakage.

### 3. Proposed Architecture (finalized)

**Key statistical structure to design around:** each of the 50–100 HF runs gives thousands of correlated spatial points, but there are only 50–100 *independent* observations across the design space. Architectures that implicitly treat every pixel as an independent sample overstate how much design-space data they actually have. The right architecture should exploit the dense spatial supervision while being honest that the design-space coverage is what's actually scarce.

**Recommended: latent-space correction + Gaussian Process.**

```
Stage 1 (spatial):  HF−LF correction fields  →  encoder/decoder (autoencoder)  →  latent z (10–50 dims)
Stage 2 (design):    design parameters θ     →  Gaussian Process  →  posterior over z  →  decoder  →  corrected field
```

- The decoder learns spatial structure from the dense per-run fields (thousands of points per run) — this is the part with plenty of data.
- The GP models the map from design parameters to the compact latent code — this is the part that's actually data-starved (50–100 points), and a GP is the right tool for exactly that regime: closed-form uncertainty that grows outside the sampled design region, rather than an unconstrained neural extrapolation.
- This directly replaces the original "branch-net-based DeepONet" plan — a full neural branch net is the component most at risk of overfitting on only 50–100 unique design examples; separating spatial complexity (handled by the decoder) from design complexity (handled by the GP) is a better match to the data's actual shape.

**Boundary conditions — enforced structurally, not learned:**
$$\Delta(x,y) = b(x,y)\,\hat\Delta(x,y), \quad b(x,y) = x(1-x)y(1-y) \text{ or a signed-distance equivalent}$$
This guarantees the correction vanishes at the domain edge by construction, at no training cost.

**Baseline to build alongside (not optional — reviewers will ask for it):** a coordinate-conditioned implicit residual network (INR-style: MLP decoder queried at arbitrary (x,y), conditioned on design + LF field) is the more direct neural formulation and a reasonable baseline to compare against the latent+GP pipeline. It shares the boundary-enforcement trick but skips the latent compression step — useful to know whether the compression is actually buying you anything.

**Known limitation to check for, not just note:** the decoder is deterministic — if a real correction has spatial structure never seen in the 50–100 training fields, the GP layer can't rescue it. Sanity check: held-out reconstruction error on the decoder alone before trusting the full pipeline.

### 4. Validation (unchanged from v2)
Leave-p-out CV over HF runs, pointwise error maps (not just aggregate norms), coverage checks on the GP's uncertainty against held-out FEM.

### 5. Project Charter
- **Primary claim:** a latent+GP correction model, trained on <=100 HF samples,
  should be tested for improvement against the fixed-case benchmark and
  validated by held-out HF runs; calibrated uncertainty must grow
  appropriately outside the sampled design region.  No universal 4% pilot
  claim is permitted without the corresponding held-out evidence.
- **MVP:** fixed decoder (no active learning yet) trained on one deliberately-designed HF batch; GP fit on top; compared against the plain co-kriging baseline from v1 and the INR baseline above; evaluated via leave-p-out CV.
- **Upgrade path:** if the MVP's GP uncertainty is poorly calibrated or the decoder underfits observed correction diversity, consider a richer decoder (e.g. neural operator backbone) or a second HF batch guided by GP uncertainty. Only pursue if the MVP result specifically points at one of these as the bottleneck.
- **Dependencies:** none required from other proposals; can share encoder/decoder infrastructure with Proposal 2 (see cross-cutting notes) but isn't blocked by it.
- **Stop condition:** MVP result plus one upgrade attempt (if justified) is sufficient; don't chase additional architecture variants without a specific observed limitation motivating each one.

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
