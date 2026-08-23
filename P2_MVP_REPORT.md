# P2 MVP Report — Spatial Damage-Field Posterior (First Iteration)

Date: 2026-08-23 · Branch: master @ 77d03e4 · Dataset: `data/p2/field_dataset.npz` (400 designs, 8×8 retention grids, 280/60/60 design-level splits)

## What was built this arc

1. **Spatial damage in the solver** — `FSDTSolver.set_damage_field` + Gauss-Legendre
   quadrature stiffness assembly (opt-in; pristine path byte-identical, ones-field
   equivalence 8.7e-15 rel Frobenius on legendre). Batched einsum assembly
   (~35× faster than the naive loop at production size).
2. **Field data infrastructure** — `DamageField`, single/multi-patch samplers,
   NPZ persistence, measurement-noise injection, design-level splits,
   `generate_field_dataset` (0.55 s/sample) + `run_p2.py generate-fields`.
3. **Field-CVAE pipeline** — `field_pipeline.py`: two-phase training adapted to
   frequencies-only conditioning, SP-gate evaluator (per-pixel interval coverage;
   sample-based SBC rank histograms with χ² p-value AND calibration error),
   heteroscedastic head, seed/bootstrap ensembles with BMA combination.
4. **Baselines** — DirectRegression / BayesianRidge (leverage uncertainty) /
   PixelClassifier per the roadmap comparison protocol.
5. **Roadmap amendment** — cINN→CVAE substitution recorded; SP4 redefined as
   sample-based SBC + coverage; zarr→npz deviation recorded.

## Lever ledger (all run on the real dataset)

| Lever | Coverage | Mean-field MSE | Verdict |
|---|---|---|---|
| A point-MSE CVAE (conditioned) | 0.692 | 0.1202 | SP3 fail; collapse at free-bits floor |
| B de-conditioned decoder | 0.0193 | **0.0364** | reconstruction 3.3× better; posterior still deterministic |
| C KL-annealed | 0.6919 | 0.1202 | identical to A |
| D heteroscedastic head (aleatoric σ) | 0.521 | 0.0373 | σ≈0.051 vs residual RMS ≈0.19 — under-dispersed |
| E seed ensemble + BMA | 0.499 | 0.0366 | seed diversity negligible |
| F bootstrap bagging + BMA | 0.502 | 0.0364 | genuine data diversity, same converged map |

Posterior-collapse diagnostics: ELBO pinned exactly at the free-bits floor
(8 nats = 16 dims × 0.5); posterior-head init produced μ≈O(10) against the
N(0,I) prior (KL ≈ 2.7e4 at step zero; fixed by bounded log-variance ±10 +
small-variance head re-init).

SBC gate semantics corrected during this arc: requires χ² p > 0.05 AND
error < 0.10 (the error statistic alone passed a p = 2.6e-18 histogram).

## Honest status vs charter

The MVP claim ("90% credible interval contains true damage at >85% of locations")
is **not met** — SP3 fails on every lever. What exists and is verified: the full
pipeline to measure it (solver → fields → CVAE → SBC/coverage gates), a
posterior-mean predictor 4.6× better than the direct-regression baseline
(0.036 vs 0.166 MSE), and a precise diagnosis of why calibration fails.

## Diagnosis and next levers

Levers D–F isolate the failure: the freq→field residual (~0.19 RMS vs
learned σ ~0.05) is **structural model bias from frequencies-only
conditioning**, not sampling noise — every variance-decomposition lever
converged to the same biased map. Next levers target bias directly:

1. Richer conditioning: mode-shape summaries (P1's HF data has them) or
   learned measurement embeddings instead of 6 log-frequencies.
2. Capacity/inductive bias on the field side: conv decoder with skips
   (roadmap Phase-3 architecture), severity-conditional sigma calibration.
3. If bias remains after 1–2: report that 6 scalar frequencies are
   information-theoretically insufficient for 8×8 field posteriors —
   itself a publishable identifiability result tied to the charter's
   ill-posedness claim.

## Reproduce

```bash
uv run python run_p2.py generate-fields --n 400 --gy 8 --gx 8 \
    -M 6 -N 6 --out data/p2/field_dataset.npz --seed 42
# then: field_pipeline.train_field_cvae / train_field_cvae_heteroscedastic /
# train_heteroscedastic_ensemble + evaluate_sp_gates / evaluate_ensemble_sp_gates
```
