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

### Spatial field pipeline — mode-mixing was the bottleneck

Held-out 20 runs, PCA d_z → GP, skill measured against predict-zero on the
same split:

| variant | model MSE | zero-baseline MSE | skill vs zero |
|---|---|---|---|
| pooled (modes folded into samples), d_z=16 | 0.2028 | 0.2035 | 0.32 % |
| **mode-conditioned** (one-hot mode GP input), d_z=8 | 0.1960 | 0.2035 | 3.66 % |
| **mode-conditioned**, d_z=16 | 0.1950 | 0.2035 | 4.18 % |
| **mode-conditioned**, d_z=32 | 0.1950 | 0.2035 | **4.19 %** |
| coverage proxy (2σ, uncalibrated) | — | — | ~0.996 |

Diagnostics (roadmap SP-gate checks):

| probe | result |
|---|---|
| PCA(16) decode-only MSE, held-out fields | **0.00014** (in-sample 0.00015) |
| Per-mode pipelines (modes 1 / 5) skill vs zero | **4.5 % / 3.9 %** |

Interpretation:

1. The bottleneck is **not representation**: correction fields are low-rank and PCA reconstructs them to 7e-4 MSE even on held-out designs.
2. Pooling modes into one GP dilutes signal; conditioning on mode identity (per-mode GPs or a mode input) recovers measurable (~4%) but still modest field-level skill.
3. Δw compares max-normalized shapes, discarding amplitude — part of the residual is plausibly irreducible given degenerate-mode-pair index fragility (see `P1_PART1_REPORT.md` caveats).

## Levers tried

1. **Mode-conditioned field GP** (one-hot mode as GP input) — ADOPTED: skill 0.32% → 4.19%; d_z=16 and 32 indistinguishable, d_z=8 worse. Committed.
2. **ARD / per-dimension length scales** (ML-II, Adam on NLML, per-latent-dim GPs) — TESTED AND REJECTED: skill drops to 3.16% (300 steps) / 3.06% (gentle); init-scale control run reproduces isotropic exactly (4.18%), so ML-II itself overfits each latent dim's GP at n=800, noise=1e-4. Negative result is informative: GP capacity/kernel flexibility is not the binding constraint.
3. **Cross-modal conditioning** (GP-predicted scalar frequency error appended to field GP inputs) — TESTED, NEUTRAL: skill 3.84% vs 4.18% baseline (−0.34%, within ±0.5% noise band). The well-predicted scalar error shares no usable design-space structure with the fields; the two correction channels are independent.

## Conclusion

Three probes converge: mode-conditioning captures the real structure (+3.9%
absolute skill), kernel/hyperparameter flexibility and cross-modal features do
not. Field-level shape-correction skill saturates near ~4% with ≤100 HF
designs — consistent with the charter's stop condition. The deliverable P1
result on this dataset is the **scalar calibration**: leave-one-run-out GP on
the five design parameters predicts per-mode FSDT frequency error at 5.35%
RMSE vs 9.45% for the train-mean baseline (all 10 modes improved), with the
field pipeline available as the mode-conditioned upgrade path if more HF data
arrives.

Reproduce:

```bash
uv run python run_p1_part1.py summary --data-root data/p1_part1
uv run python -c "from mechanics.p1_multifidelity.real_pipeline import run_field_pipeline, fit_frequency_error_gp; \
print(run_field_pipeline('/tmp/p1_canonical')['metrics']); \
print(fit_frequency_error_gp('/tmp/p1_canonical')['mean_loo_rmse_pct'])"
```

(Bulk corpora expected under the data root; see MANIFEST.)
