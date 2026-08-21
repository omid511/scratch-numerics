# MANIFEST — Proposal-1 Part-1 HF Correction Dataset (`data/p1_part1/`)

Canonical layout of the incorporated Proposal-1 Part-1 high-fidelity (HF)
correction dataset. Programmatic access: `mechanics.p1_multifidelity.hf_dataset`
(CLI entry points: `run_p1_part1.py` at the repo root).

## Layout (the seven `DATA_FILES` entries)

| Key                   | Path                        | Present in git? |
|-----------------------|-----------------------------|-----------------|
| `samples_csv`         | `lhs_samples.csv`           | yes (small)     |
| `fsdt_freqs_csv`      | `lhs_fsdt_results.csv`      | yes (small)     |
| `comsol_freqs_csv`    | `lhs_results_master.csv`    | yes (small)     |
| `correction_freq_csv` | `correction_freq.csv`       | yes (small)     |
| —                     | `mesh_convergence_final.csv`| yes (small)     |
| `fsdt_shapes_dir`     | `fsdt_mode_shapes/`         | **not committed** (bulk, regenerable) |
| `comsol_shapes_dir`   | `Simulation_ModeShapes/`    | **not committed** (bulk, regenerable) |
| `corrections_dir`     | `correction_fields/`        | **not committed** (bulk, regenerable; regenerable from the two shape dirs via `run_p1_part1.py corrections`) |

## Committed CSVs

Five small CSVs are committed:

- `lhs_samples.csv` — 100 designs × 5 parameters
  (`alpha,beta,theta_c,eta1,eta2`), sensitivity-warped LHS.
- `lhs_fsdt_results.csv` — same 5 parameters plus FSDT frequencies
  `f1_fsdt..f10_fsdt`.
- `lhs_results_master.csv` — per-run COMSOL metadata (parameters, geometry,
  `lc`, `tc`, `h_face`, `hc`) plus COMSOL frequencies `f1..f10` and timings.
- `correction_freq.csv` — frequency errors per run/mode:
  `run,f{i}_fsdt,f{i}_comsol,f{i}_abs_err,f{i}_rel_err_pct`
  (41 columns, i = 1..10).
- `mesh_convergence_final.csv` — mesh-convergence study rows
  (`hauto`, element counts, timings, frequencies, mean/max % error vs paper
  FEM values); justifies the production mesh choice.

## Bulk corpora (NOT committed)

The three directories below hold 100 runs × 10 modes = 1000 files each and are
large. They are expected under this root when working with shape data but are
deliberately not committed to git:

- `fsdt_mode_shapes/fsdt_run{N}_mode{M}.csv` — FSDT mode shapes.
- `Simulation_ModeShapes/mode_shape_run{N}_mode{M}.csv` — COMSOL mode shapes.
  (Historical note: some raw copies name this directory `mode_shapes`; the
  canonical name here is `Simulation_ModeShapes`. The `corrections`
  subcommand accepts either.)
- `correction_fields/correction_run{N}_mode{M}.csv` — spatial correction
  fields Δw = COMSOL(norm) − FSDT(norm).

Each mode-shape / correction CSV is an 80×80 grid indexed `[y, x]` with a
3-line text header.

## Provenance

- **HF source:** COMSOL shell model of a CCCC 300×300×10 mm Al honeycomb
  sandwich plate. Production meshes use `hauto = 4` (~94k elements), chosen
  from a mesh-convergence study (`mesh_convergence_final.csv`: ~0.49% mean
  error vs published paper FEM values, with finer meshes changing results by
  <0.02%).
- **Sampling:** 100-design sensitivity-warped LHS over the five parameters
  `alpha, beta, theta_c, eta1, eta2`.
- **Content:** 10 modes per design on 80×80 grids indexed `[y, x]`.

## Regeneration

- Raw corpora live under `~/Downloads/Proposal1_Part1/` (COMSOL export +
  FSDT scripts). The spatial/frequency correction outputs are recomputable
  from a raw root via:

  ```
  python run_p1_part1.py corrections --raw-root <raw Proposal1_Part1 root> \
      --out-dir data/p1_part1
  ```

  which reproduces `correction_freq.csv` and `correction_fields/` byte-for-
  schema-compatible with the original
  `5- compute_correction/compute_corrections.py` logic (minus its plotting).
