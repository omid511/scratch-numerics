# Proposal 1 data

## Frozen design and raw runs

`lhs_samples_v2.csv` is the frozen 100-row, seed-42 design table. It has no
explicit `run_id` column; its one-based row order is the run identity. Do not
sort, rewrite, or append to this file. Newly generated tables from
`lhs_sampling.py` use an explicit `run_id` column and are written to a new
path.

The five sampled variables are:

| variable | meaning |
|---|---|
| `alpha` | core thickness ratio |
| `beta` | top-face / bottom-face thickness ratio |
| `theta_c` | honeycomb cell angle (degrees) |
| `eta1` | honeycomb edge ratio `l2/l1` |
| `eta2` | cell-wall ratio `tc/l1` |

The honeycomb constitutive implementation is retained. The source paper's
Table 1 `G12` entry conflicts with its displayed equation; the code follows
the equation and is protected by a reference-value regression test.

`p1_data_v3/lf/run_XXXX.npz` contains the Legendre-Ritz FSDT result:
common x/y axes, 16 positive flexural frequencies, 80x80 transverse fields,
selected source eigenmode indices, transverse-energy fractions, and
provenance metadata.

`p1_data_v3/hf/run_XXXX.npz` contains the explicit COMSOL shell result on the
same top-face grid. New HF exports solve extra candidate modes, evaluate
`u`, `v`, and `w`, retain only modes with transverse-energy fraction at least
0.5, and store `w_peak_abs`, `transverse_fraction`, and the selected COMSOL
solution numbers. Existing HF bundles without those fields are legacy
bundles and must be regenerated before making a final HF quality claim. The
raw `w_peak_abs`/`w_peak_ratio` values are scale-dependent diagnostics only;
they are not acceptance criteria because COMSOL eigenvector normalization is
arbitrary.

## Current correction revision

`p1_data_v3/corrections_v5/` is the immutable current correction revision. Run
the builder into a new directory for any future rebuild:

```powershell
python compute_corrections.py --output p1_data_v3 --name corrections_v6
```

The manifest records the selected stable non-degenerate LF reference, per-mode
reference identity, deterministic leave-one-out tracking statistics, HF
transverse-mode quality policy, portable relative source paths, and run-level
split policy. Its `training.npz` stores `boundary_mask` and `interior_mask`;
`pairing.csv` stores diagnostic raw HF peak ratios plus interior and edge
correction RMS. Accepted fields remain max-normalized, phase-aligned
differences. Quarantined rows are never zero-filled into training labels.

The current revision has 924 accepted and 76 quarantined pairs. It selects run
90 as a globally coherent, stable non-degenerate LF reference for all ten
labels, removes the invalid raw amplitude filter, and retains only the HF
transverse-energy criterion. Its accepted frequency-error median is 4.3397%
and p95 is 11.8046%; these are pilot measurements, not a universal accuracy
claim.

Generate current-production diagnostics with:

```powershell
python plot_p1_results.py --data p1_data_v3 --name corrections_v5
```

The figures use interior RMS for correction plots and report edge RMS
separately. Apply `boundary_mask` to a future decoder output and exclude the
perimeter from loss/RMSE; do not silently claim that the raw edge residual is
zero.

## Scope and claims

This is a corrected structural mode-shape pilot for Proposal 1. It varies the
five sampled geometry variables, keeps plate material, dimensions, and CCCC
boundary treatment fixed, and compares free-vibration FSDT with a COMSOL
shell. It is not the proposal's full 10,000-20,000 LF / 50-100 Abaqus study,
latent-GP model, calibrated uncertainty result, or boundary-stiffness study.

The proposal's approximately 4 percent comparison is a paper-table result for
the stated fixed CCCC reference cases. It is not a valid blanket claim for this
pilot: the selected correction revision has mode-dependent errors larger than
4 percent. Report the empirical distribution from the selected correction
revision instead.

The LF solver uses layerwise modified shear-correction factors from laminate
integrals; its current reference-laminate baseline is approximately
`kappa = 0.16875`, not a universal `5/6`. The correction field therefore
learns the remaining discrepancy between the implemented FSDT solver and the
explicit shell reference.

## Mesh validation

The corrected reference mesh output in
`mesh_validation_corrected/mesh_convergence_final.csv` is a spot check, not
evidence that every sampled geometry is converged. COMSOL's `hauto` convention
uses lower numbers for finer meshes (`1` is finest, `9` is coarsest); production
uses `hauto=4`. The mesh script compares production `4` with a finer `3`:

```powershell
python simulations/scripts/hc_mesh_convergence.py --extreme-runs 10 --extreme-levels 4,3 --extreme-tolerance 2e-4
```

The corrected one-extreme spot check was run with this ordering. Run 18 changed
by `0.0041485` between `hauto=4` and `hauto=3`, exceeding the `2e-4`
tolerance; therefore production `hauto=4` is not yet a convergence claim for
that sampled extreme. The result is recorded in
`mesh_validation_corrected/mesh_extreme_validation.csv`.

## Handoff and future recomputation

`p1_data_v3/` is the current handoff root. Do not overwrite
`corrections_v5/`. If the mesh decision changes production from `hauto=4`,
generate fresh raw bundles under a new root because the saved COMSOL models
were removed:

```powershell
python fsdt_mode_shapes.py --samples lhs_samples_v2.csv --output p1_data_v4 --modes 16 --order 15 --grid 80
python run_hf_batches.py --samples lhs_samples_v2.csv --output p1_data_v4 --replace
```

The HF command uses the extra-candidate flexural filter, validated
eigenfrequency shift, and one fresh Python/COMSOL worker process per batch.
The wrapper resolves sample and output paths before launching workers. Use
`--reexport` only when compatible saved `.mph` models and sidecars are
available.

Rebuild and inspect any new immutable correction revision:

```powershell
python compute_corrections.py --output p1_data_v4 --name corrections_v6
python plot_p1_results.py --data p1_data_v4 --name corrections_v6
```

The ANN handoff is `p1_data_v3/corrections_v5/training.npz`: use `parameters`
as inputs and `correction` as the target, with `run_ids` and `mode_ids`
retained for grouped validation. The file contains accepted rows only.

Split any ANN evaluation by `run_ids`, never by pixels or individual modes.
Report this pilot as pipeline validation unless a held-out correction model
and uncertainty calibration have actually been run.
