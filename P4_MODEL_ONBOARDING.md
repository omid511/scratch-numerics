# P4 Model Onboarding Map

## Purpose

P4 estimates an aeroelastic stability margin from simulated multi-sensor transient plate response. Label: `margin = (u_crit - velocity) / u_crit`; positive is subcritical, zero is flutter, negative is supercritical.

Current pipeline:

```text
Sobol design -> FSDT solver + CFCF BC -> flutter velocity -> filtered state-space modes
-> modal transient clips -> design-level split -> causal TCN quantile regression -> MAE/coverage
```

Start with this document, then open only paths named below.

## Entry Points

| Task | Path | What it does |
| --- | --- | --- |
| Generate current dataset | `generate_p4_dataset.py` | Multiprocess design sampling and serialization. Current main pipeline. |
| Train/evaluate current dataset | `train_p4_expanded.py` | Loads `p4_dataset/`, uses stored design splits, trains 3 TCN variants and 4 baselines. |
| Legacy single-design demo | `run_p4.py` | Generates nominal CFCF clips in memory, grouped by velocity, trains 3 models. Not design-generalization workflow. |
| Plot legacy results | `plot_p4_results.py` | Produces `p4_results.png` and `P4_REPORT.md`. |
| P4 unit/behavior tests | `src/mechanics/p4_margin_estimation/tests/test_p4.py` | Transients, augmentation, causality, quantiles, training behavior. |
| P4 integration tests | `src/mechanics/p4_margin_estimation/tests/test_integration.py` | Solver to clip to train/evaluate smoke test. |

## Source Map

| Path | Owns | Read when |
| --- | --- | --- |
| `src/mechanics/p4_margin_estimation/design_sampler.py` | `DesignSample`, 11-D Sobol sampling, nominal materials, fixed CFCF solver factory, per-design flutter search | Changing design space, materials, geometry, BCs, damping, or flutter labels. |
| `src/mechanics/p4_margin_estimation/transient.py` | `Eigendecomposition`, `TransientClip`, sensor locations, modal propagation, causal RMS normalization | Changing input signals, sensors, clipping, labels, or modal selection. |
| `src/mechanics/p4_margin_estimation/train.py` | Tensor conversion, grouped split for legacy path, TCN training, Huber/median variants, coverage metrics | Changing split protocol, loss, optimizer, metrics, or model invocation. |
| `src/mechanics/p4_margin_estimation/tcn.py` | Causal residual Conv1d backbone | Changing temporal model. Default receptive field is 61 samples. |
| `src/mechanics/p4_margin_estimation/quantile_head.py` | Ordered q05/q50/q95 head and pinball loss | Changing uncertainty outputs or quantile behavior. |
| `src/mechanics/p4_margin_estimation/baselines.py` | Constant, velocity-linear, growth-rate, physics-feature ridge, GRU baselines | Comparing learned model against baselines. |
| `src/mechanics/p4_margin_estimation/domain_randomization.py` | Sensor gain/bias/drift/noise/dropout/skew augmentation | Sim-to-real robustness work. Not invoked by current generator or trainer. |
| `src/mechanics/eigenanalysis.py` | Shared generalized eigenproblem, residual/participation filters | Changing any P4 mode selection or flutter stability calculation. |
| `src/mechanics/solver.py` | FSDT assembly, aeroelastic system, structural damping, flutter scan | Physics/solver changes affecting all P4 labels and signals. |
| `src/mechanics/piston_theory.py` | Mach validation and piston-theory aerodynamics | Changing flow regime or aerodynamic assumptions. |
| `src/mechanics/laminate.py` | `Material`, `Laminate`, laminate stiffness | Changing material/layup mechanics. |

## Physics And Data Contract

- Solver: `FSDTSolver`, five fields; current sampled designs use `M=N=6`, `grid=(32, 32)`, `k_stiffness=1e14`.
- BCs are fixed CFCF: left/top clamped, right/bottom free. They are no longer sampled.
- Design sampler covers `L1`, `L2/L1`, face/core thickness, face stiffness/density, core shear/density, damping `zeta`, air density, and sound speed. See bounds in `sample_designs()`.
- Piston-theory validity is enforced at `Mach >= 2`. Generator accepts designs only when `u_crit / c_air >= 2.0 / 0.68 * 1.001`, ensuring every sampled velocity remains valid.
- `compute_design_u_crit()` delegates to `solver.find_flutter_velocity()` with design air properties and damping. Flutter is first stable-to-unstable crossing from scan plus bisection.
- `solve_eigenproblem()` is sole eigenanalysis path. It uses generalized pencil `[0,I;-K,-C] z = s [I,0;0,M] z`, then filters frequency, positive imaginary branch, transverse participation, and residual.
- `compute_eigendecomposition()` keeps two least-stable modes first, then fills remaining requested modes by frequency. Do not make transient and flutter paths use different filters.
- Clip shape is `(n_sensors, n_timesteps)`, normally `(8, 512)`, over 0.5 s. Sensors are fixed interior locations from `default_sensor_xy()`.
- Modal coefficients use random log-normal amplitudes and phases. Two least-stable modes must receive at least 20% total initial amplitude.
- Clip normalization is causal: one global RMS from first 10% of samples. Do not normalize using future samples.
- Generator applies 3 material/damping realizations per nominal design, 10 continuous velocity ratios per realization sampled from 5 stratified bands (`VELOCITY_RATIO_STRATA`: 2 each in 0.68-0.85, 0.85-0.95, 0.95-1.00, 1.00-1.05, 1.05-1.15), and 2 excitations per velocity. Nominal maximum: 60 clips/design before failures.
- Train/validation/test are assigned once per design: 70%/15%/15%. Never split clips from one design across partitions.

## Model Contract

- Input: `float32 (batch, 8, 512)` sensor signals. Target: scalar margin per clip.
- Default backbone: causal TCN, 32 hidden channels, 4 residual blocks, kernel 3, dilations 1/2/4/8, dropout 0.1; global mean pooling.
- Quantile model predicts `(q05, q50, q95)`. Median-centered `softplus` widths guarantee `q05 <= q50 <= q95` without sorting.
- `train()` uses pinball loss and cosine-decayed Adam. `train_huber()` is point regression; `train_median()` is q50-only pinball regression.
- `evaluate_coverage()` reports median MAE, marginal quantile coverages, q05-q95 interval coverage, and interval width. Coverage alone is not evidence of calibration on small test sets.
- Legacy split groups by exact velocity. Current expanded workflow must pass explicit design-level `train_clips`, `val_clips`, and `test_clips` into training.

## Dataset Files

Expected generated directory contents:

```text
p4_dataset/
  clips.npz            # key: clips; shape (N, 8, 512)
  margins.npy          # shape (N,)
  velocities.npy       # shape (N,)
  design_ids.npy       # shape (N,)
  realization_ids.npy  # shape (N,)
  metadata.json        # design records and design-level split IDs
```

All P4 data, checkpoints, and results are ignored by Git (`.gitignore`). Dataset files (`p4_dataset/clips.npz`, etc.) must be generated via `generate_p4_dataset.py` before training. `train_p4_expanded.py` validates clips on load (finite values, correct shape, no all-zero clips).

`p4_dataset/metadata.json` is written by `generate_p4_dataset.py` and includes a `velocity_sampling` object describing the current 5-band stratified sampling of 10 continuous velocity ratios per realization (see `VELOCITY_RATIO_STRATA`). If your local metadata predates this scheme, regenerate the whole directory; do not mix old metadata/arrays with new clips.

## Commands

Run from repository root. `uv` manages Python environment.

```bash
uv sync
PYTHONPATH=src uv run pytest src/mechanics/p4_margin_estimation/tests -q
PYTHONPATH=src uv run python generate_p4_dataset.py --n-designs 160 --output-dir p4_dataset --seed 42
PYTHONPATH=src uv run python train_p4_expanded.py
PYTHONPATH=src uv run python run_p4.py
```

Dataset generation is expensive: process count defaults to 4 and can be changed with `DESIGN_WORKERS`, for example `DESIGN_WORKERS=2 PYTHONPATH=src uv run python generate_p4_dataset.py -n 8 -o /tmp/p4-smoke`.

## Current State And Guardrails

- Worktree contains uncommitted P4/solver changes. Treat current source, not older reports, as authoritative.
- `P4_REPORT.md` reports legacy nominal-dataset results only. It does not evaluate current design-level generator.
- Historical review notes (`review_round3_notes.md`, `roadmaps/reviews/p4_review_r2.md`) no longer exist; treat current source and tests as the only authoritative record of physics changes.
- Full-order generalized eigensolves dominate runtime. Deferred improvements: modal reduction, adaptive bracketing/mode tracking, physical modal normalization, and constraint elimination.
- `domain_randomization.py`'s dropout distribution field is `channel_dropout_distribution`, exposed as the property `channel_drop_probability`. The legacy non-ASCII name `channel_drop概率` remains as a deprecated read-only alias; prefer the ASCII property in new code.
- On any physics change, rerun P4 tests and regenerate dataset/checkpoints. Labels and clips share solver/eigenanalysis assumptions; changing only one produces invalid experiments.

## Fast Change Routing

| Need | First files |
| --- | --- |
| New design parameter or distribution | `design_sampler.py`, `generate_p4_dataset.py`, dataset metadata consumers |
| Different sensors, duration, normalization, excitation | `transient.py`, generator, tests |
| Change flutter definition/filtering | `eigenanalysis.py`, `solver.py`, `transient.py`, integration tests |
| Change model, loss, metrics | `tcn.py` or `quantile_head.py`, `train.py`, trainer, tests |
| Add robustness augmentation | `domain_randomization.py`, generator/trainer, behavioral tests |
| Reproduce legacy figure | `run_p4.py`, `plot_p4_results.py`, `P4_REPORT.md` |
