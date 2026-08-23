# P2 MVP Report — Spatial Damage-Field Posterior (First Iteration)

Date: 2026-08-23 · Branch: master @ f1f3729+ · Dataset: `data/p2/field_dataset.npz` (400 designs, 8×8 retention grids, 280/60/60 design-level splits)

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
   sample-based SBC rank histograms with χ² p-value AND calibration error).
4. **Baselines** — DirectRegression / BayesianRidge (leverage uncertainty) /
   PixelClassifier per the roadmap comparison protocol.
5. **Roadmap amendment** — cINN→CVAE substitution recorded; SP4 redefined as
   sample-based SBC + coverage; zarr→npz deviation recorded.

## SP-gate results (validation split, 60 designs)

| Arm | Coverage (gate >0.80) | SBC p-value | SBC error (<0.10) | Mean-field MSE |
|---|---|---|---|---|
| A conditioned decoder | 0.6919 ✗ | 0.000 | 0.0377 ✓(misleading alone) | 0.1202 |
| B de-conditioned decoder | 0.0193 ✗ | 0.000 | 0.0253 ✓(misleading alone) | **0.0364** |
| C KL-annealed (25 ep) | 0.6919 ✗ | 0.000 | 0.0377 ✓(misleading alone) | 0.1202 |
| DirectRegression baseline | — | — | — | 0.1655 |

Posterior-collapse diagnostics that motivated arms B/C: ELBO pinned exactly at the
free-bits floor (8 nats = 16 dims × 0.5); posterior-head init produced μ≈O(10)
against the N(0,I) prior (KL ≈ 2.7e4 at step zero; fixed by bounded log-variance
±10 + small-variance head re-init).

## Findings

1. **SP3 fails for all three arms** — no calibrated spatial posterior yet. With an
   MSE-reconstruction objective the optimal q(z|c) variance collapses regardless of
   de-conditioning or annealing: intervals come out over-confident.
2. **De-conditioning is real signal, not noise**: routing measurements only through
   z improves mean-field MSE 3.3× (0.120→0.036), beating the direct-regression
   baseline by 4.6×. The latent carries measurement information when forced to.
3. **SBC gate semantics fixed**: mean |bin-prop − uniform| passed at χ² p = 0.0;
   the gate now requires BOTH error < 0.10 AND p > 0.05.

## Honest status vs charter

The MVP claim ("90% credible interval contains true damage at >85% of locations")
is **not met**. What exists and is verified: the full pipeline to measure it
(solver → fields → CVAE → SBC/coverage gates), a 4.6×-better-than-baseline
posterior-mean predictor, and a precise diagnosis of why calibration fails
(point-MSE reconstruction gives a CVAE no incentive for honest spread).

## Next levers (ranked)

1. Heteroscedastic Gaussian likelihood head (predict per-pixel μ AND σ; NLL loss)
   — directly targets calibration, small change to decoder/head.
2. Diffusion posterior (roadmap's gated upgrade) if 1 stalls.
3. More HF samples along the severity axis (active learning via the existing
   sampler machinery).

Reproduce:

```bash
uv run python run_p2.py generate-fields --n 400 --gy 8 --gx 8 \
    -M 6 -N 6 --out data/p2/field_dataset.npz --seed 42
# then: field_pipeline.train_field_cvae + evaluate_sp_gates as in tests
```
