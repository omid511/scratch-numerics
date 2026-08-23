# P2 MVP Report — Spatial Damage-Field Posterior (First Iteration)

Date: 2026-08-23 · Branch: master @ ff566e1 · Dataset: `data/p2/field_dataset.npz` (400 designs, 8×8 retention grids, 280/60/60 design-level splits) + summaries variant `data/p2/fields_ds_summaries.npz` (300 designs)

## What was built this arc

1. **Spatial damage in the solver** — `FSDTSolver.set_damage_field` + Gauss-Legendre
   quadrature stiffness assembly (opt-in; pristine path byte-identical, ones-field
   equivalence 8.7e-15 rel Frobenius on legendre). Batched einsum assembly
   (~35× faster than the naive loop at production size).
2. **Field data infrastructure** — `DamageField`, single/multi-patch samplers,
   NPZ persistence, measurement-noise injection, design-level splits,
   `generate_field_dataset` (0.55 s/sample, optional per-mode shape summaries)
   + `run_p2.py generate-fields`.
3. **Field-CVAE pipeline** — `field_pipeline.py`: two-phase training adapted to
   frequencies-only conditioning, SP-gate evaluator (per-pixel interval coverage;
   sample-based SBC rank histograms with χ² p-value AND calibration error),
   heteroscedastic head, seed/bootstrap ensembles with BMA combination,
   `posterior_init_std` + per-epoch recon/KL component history.
4. **Baselines** — DirectRegression / BayesianRidge (leverage uncertainty) /
   PixelClassifier per the roadmap comparison protocol.
5. **Roadmap amendment** — cINN→CVAE substitution recorded; SP4 redefined as
   sample-based SBC + coverage; zarr→npz deviation recorded.

## Lever ledger (all run on real datasets)

| Lever | Coverage | Mean-field MSE | Verdict |
|---|---|---|---|
| A point-MSE CVAE (conditioned) | 0.692 | 0.1202 | SP3 fail; collapse at free-bits floor |
| B de-conditioned decoder | 0.0193 | **0.0364** | reconstruction 3.3× better; posterior still deterministic |
| C KL-annealed | 0.6919 | 0.1202 | identical to A |
| D heteroscedastic head (aleatoric σ) | 0.521 | 0.0373 | σ≈0.051 vs residual RMS ≈0.19 — under-dispersed |
| E seed ensemble + BMA | 0.499 | 0.0366 | seed diversity negligible |
| F bootstrap bagging + BMA | 0.502 | 0.0364 | genuine data diversity, same converged map |
| G posterior_init_std 0.05–1.0 sweep | 0.727 (std≥0.2) | 0.0543–0.0636 | **bistable**: std≤0.15 re-collapses to the 0.5-nat/dim floor (dead KL gradient); std≥0.2 escapes into O(10⁴)-nat noise encoding — coverage lifts but MSE +42–65% and SBC p=0. No useful middle via init scale |
| H mode-shape summary conditioning | — | +5–10% worse than no-summaries | NEGATIVE: summaries near-redundant with log-freqs under a collapsed posterior |

Posterior-collapse diagnostics: ELBO pinned exactly at the free-bits floor
(8 nats = 16 dims × 0.5); posterior-head init extremes produce either
floor-pinning (dead KL gradient) or O(10⁴)-nat runaway (both characterized
in the lever-G sweep).

SBC gate semantics corrected during this arc: requires χ² p > 0.05 AND
error < 0.10 (the error statistic alone passed a p = 2.6e-18 histogram).

## Honest status vs charter

The MVP claim ("90% credible interval contains true damage at >85% of locations")
is **not met** — SP3 fails on every lever tried. What exists and is verified: the
full pipeline to measure it (solver → fields → CVAE → SBC/coverage gates), a
posterior-mean predictor 4.6× better than the direct-regression baseline
(0.036 vs 0.166 MSE), and a precisely characterized failure geometry.

## Diagnosis

Levers D–G isolate the failure: the freq→field residual (~0.19 RMS vs learned
σ ~0.05) is structural, and lever G proves the ELBO's free-bits+clamp geometry
is bistable — either dead KL gradient or noise-encoded runaway, with no
continuous path between. Conditioning enrichments (H) are downstream of the
collapse, not causes of it.

## Next levers (targeting the collapse mechanism itself)

1. Soft KL-rate penalty or d_z capacity annealing instead of the hard
   free-bits floor + clamp (removes the cliff lever G mapped).
2. mu-scale warmup before enabling the KL term.
3. If the cliff persists after 1–2: report that with an MSE-reconstruction
   objective this architecture family cannot express calibrated spread —
   pivot to diffusion posterior (roadmap's gated upgrade).

## Reproduce

```bash
uv run python run_p2.py generate-fields --n 400 --gy 8 --gx 8 \
    -M 6 -N 6 --out data/p2/field_dataset.npz --seed 42 --summaries
# then: field_pipeline.train_field_cvae / train_field_cvae_heteroscedastic /
# train_heteroscedastic_ensemble + evaluate_sp_gates / evaluate_ensemble_sp_gates
```
