# P4 Code Review

## Verdict

**Request changes.** Current end-to-end workflow is blocked before physics or
model quality can be measured. Tests were not run, per review request.

## Critical Findings

- `generate_p4_dataset.py:191-197`: `SeedSequence` receives four positional
  arguments; excitation generation raises `TypeError`. Pass one entropy
  sequence: `SeedSequence([design_seed, r_idx, velocity_key, exc_idx])`.
- `generate_p4_dataset.py:190-220,374-381`: both excitations have identical
  `(design_id, realization_idx, velocity)` keys, so duplicate validation rejects
  every otherwise valid dataset. Persist `excitation_idx` and include it in the
  key.
- `generate_p4_dataset.py:430-437` and `train_p4_expanded.py:115-129`: writer
  creates `metadata_arrays.npz`; loader requires four standalone `.npy` files.
  Generated datasets cannot train. Define one versioned schema and add a
  round-trip test.
- `src/mechanics/p4_margin_estimation/tcn.py:83-89` and
  `train_p4_expanded.py:252-296`: production models use four layers, giving a
  receptive field of 61, but the backbone rejects receptive fields below 512.
  Every TCN variant fails construction. Use sufficient layers or remove the
  full-sequence requirement if temporal summary intentionally supports a
  partial receptive field.

## High-Severity Findings

- `src/mechanics/eigenanalysis.py:240-303` and
  `src/mechanics/p4_margin_estimation/transient.py:173-237`: label boundary uses
  full-spectrum first instability, including static or in-plane divergence;
  clips retain only oscillatory transverse modes. Margin can cross zero without
  observable sensor instability. Use the same mode class for labels and
  features, or explicitly model first instability and include its critical mode.
- `src/mechanics/p4_margin_estimation/train.py:298-387` and
  `train_p4_expanded.py:281-294`: `train_median()` trains three quantiles and
  returns `(B, 3)` while `evaluate_point()` expects `(B, 1)`. The benchmark is
  mislabeled and evaluation broadcasts or fails. Implement a scalar q50 model.
- `src/mechanics/p4_margin_estimation/baselines.py:243-248`: GRU training passes
  `test_split=0.0` into a splitter requiring a positive test split. Add a grouped
  two-way split.
- `train_p4_expanded.py:425-443`: validation designs select checkpoints and
  calibrate CQR. This biases coverage. Reserve independent calibration designs.
- `generate_p4_dataset.py:108-120,213-230,394-416`: output omits realized
  `u_crit`, excitation ID, ratio/Mach, material perturbations, damping, and
  sensor coordinates. Labels cannot be reproduced or audited.
- `src/mechanics/laminate.py:156-159,215-217` and
  `src/mechanics/solver.py:322-336`: `kappa()` orders shear factors as
  `[Q55, Q44]`, while `As` uses `[Q44, Q55]`. Orthotropic shear corrections are
  axis-swapped.

## Medium-Severity Findings

- `src/mechanics/eigenanalysis.py:277-303`: `finite` uses raw-spectrum length
  while `backward_errors` uses finite-spectrum length. Any infinite eigenvalue
  causes mask or index failure.
- `src/mechanics/p4_margin_estimation/transient.py:109-118`: causal normalization
  uses all calibration samples for earlier outputs. This is valid only for
  clip-end prediction, not streaming inference.
- `src/mechanics/p4_margin_estimation/transient.py:90-96,318-351`: random
  coefficients act on arbitrarily normalized right eigenvectors rather than
  physical displacement, velocity, or impulse energy. Excitation distribution
  changes with eigensolver scaling.
- `src/mechanics/solver.py:878-902`: the 40-point scan skips unresolved intervals
  and can miss narrow instability windows. Require scan-density convergence and
  fail rather than silently skipping unresolved brackets.
- `train_p4_expanded.py:311-318`: growth and physics baselines receive 16
  channels and treat eight validity masks as sensor signals. Strip masks or make
  feature extraction mask-aware.
- `pyproject.toml:7-14`: direct `threadpoolctl` import is undeclared. Clean
  installation may fail.

## Test Coverage

- `src/mechanics/p4_margin_estimation/tests/test_integration.py:7-35` is synthetic
  training only. It never exercises the solver, dataset writer, loader,
  production configuration, checkpoint, or CLI.
- `src/mechanics/p4_margin_estimation/tests/test_p4.py:613-621` reimplements
  false-safe arithmetic instead of testing production metric code.
- `src/mechanics/p4_margin_estimation/tests/test_p4.py:568-580,624-646,681-696`
  skips or silently passes failed physics solves, weakening convergence and
  flutter assertions.
- Missing Tier-1 checks: generator-to-loader round trip, production model
  construction, two-excitation identity, transient propagation versus
  `scipy.linalg.expm`, requested damping-ratio recovery, flutter scan
  convergence, critical-mode observability, orthotropic shear ordering, and
  per-clip provenance reconstruction.
- `pyproject.toml:24-27` defines no test paths, slow/physics markers, branch
  coverage, or coverage threshold.

## Efficiency

- `generate_p4_dataset.py:287-338,422-428` holds float64 worker results,
  stacks/copies them to float32, then copies them into a memmap. Stream float32
  shards directly.
- `src/mechanics/p4_margin_estimation/transient.py:326-351` builds full `w_all`
  even when cached `sensor_modes` makes it unused.
- `src/mechanics/p4_margin_estimation/train.py:419-425` and
  `train_p4_expanded.py:159-177` evaluate whole splits in one tensor. Use batched
  loaders to avoid expanded-dataset out-of-memory failures.

## Physics Assessment

Generalized QEP formulation, margin formula, modal damping transform, and the
Mach >= 2 generator guard are reasonable. Current P4 result is not physically
defensible yet because label-driving instability and observable transient modes
are inconsistent, shear correction is wrong for anisotropic laminates, and
convergence and provenance checks are absent.
