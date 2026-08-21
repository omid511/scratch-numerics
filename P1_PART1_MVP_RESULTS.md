# P1 Part-1 Real-Data Correction Model — First Results

Date: 2026-08-22 · Branch: `proposal1-part1-incorporation` · Data root: canonical Part-1 dataset (see `data/p1_part1/MANIFEST.md`)

Scope: first application of the roadmap Phase 2/3 machinery (`src/mechanics/p1_multifidelity/`) to the **real** COMSOL-vs-FSDT correction dataset (100 designs × 10 modes × 80×80), replacing the synthetic-HF placeholder. All numbers below are leave-run-out or held-out-run results; no test field/pixel entered any fit.

## Setup

- Design θ ∈ R⁵ = (α, β, θ_c, η₁, η₂), sensitivity-warped LHS as recorded in `lhs_samples.csv`.
- Targets:
  - **Scalar:** per-mode relative frequency error `rel_err_pct` from `correction_freq.csv` (COMSOL − FSDT over COMSOL).
  - **Field:** spatial Δw shape-difference fields from `correction_fields/` (both sides max-normalized independently).
- Models: PCA latent encoder (`CorrectionEncoder`, d_z=16) → independent RBF-GP per latent dim (`LatentGP`) → envelope decoder; scalar path is a direct GP on standardized θ.
- Splits: by **run** (charter rule); 80/20 holdout and leave-one-run-out CV.

## Results

### Scalar frequency-error GP — positive result

Leave-one-run-out, per-mode:

| metric | GP on θ | train-mean baseline |
|---|---|---|
| mean LOO RMSE (% rel. err) | **5.35** | 9.45 |
| per-mode LOO RMSE range | 4.29 – 6.43 | 8.96 – 9.97 |

The GP beats the trivial baseline on **all 10 modes**. The 5 design parameters carry genuine information about the FSDT frequency error (~43% RMSE reduction vs predicting the mean). This is the first quantified, cross-validated correction model on real HF data in the repo and already supports a usable scalar calibration of the fast solver.

### Spatial field pipeline — negative-but-informative

Held-out 20 runs, pooled across modes (PCA d_z=16 → GP):

| quantity | value |
|---|---|
| model reconstruction MSE | 0.2028 |
| predict-zero baseline MSE | 0.2035 |
| skill vs zero baseline | **0.32 %** |
| coverage proxy (2σ, uncalibrated) | 0.996 |

Diagnostics (roadmap SP-gate checks):

| probe | result |
|---|---|
| PCA(16) decode-only MSE, held-out fields | **0.00014** (in-sample 0.00015) |
| Per-mode pipelines (modes 1 / 5) skill vs zero | **4.5 % / 3.9 %** |

Interpretation:

1. The bottleneck is **not representation**: correction fields are low-rank and PCA reconstructs them to 7e-4 MSE even on held-out designs.
2. Pooling modes into one GP dilutes signal; conditioning on mode identity (per-mode GPs or a mode input) recovers measurable (~4%) but still modest field-level skill.
3. Δw compares max-normalized shapes, discarding amplitude — part of the residual is plausibly irreducible given degenerate-mode-pair index fragility (see `P1_PART1_REPORT.md` caveats).

## Next levers (ranked, cheapest first)

1. Mode-conditioned field GP (one-hot mode as 6th GP input, or per-mode latent GPs) — probe already shows ~12× skill gain from de-pooling.
2. ARD / per-dimension length-scale kernel for the θ→z GP (current single length-scale RBF).
3. Predict the scalar-error field's *spatial profile* conditional on the (already predictable) scalar magnitude.
4. If field skill stays <10% after 1–2 upgrades, that itself is a finding: report that with ≤100 HF samples, field-level correction learning is data-limited, and ship the scalar calibration as the P1 deliverable — consistent with the charter's stop condition.

Reproduce:

```bash
uv run python run_p1_part1.py summary --data-root data/p1_part1
uv run python -c "from mechanics.p1_multifidelity.real_pipeline import run_field_pipeline, fit_frequency_error_gp; \
print(run_field_pipeline('/tmp/p1_canonical')['metrics']); \
print(fit_frequency_error_gp('/tmp/p1_canonical')['mean_loo_rmse_pct'])"
```

(Bulk corpora expected under the data root; see MANIFEST.)
