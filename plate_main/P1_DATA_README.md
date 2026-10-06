# Proposal 1 data

## Current P1 result and public availability

Start with the [prospective final report](P1_IMPROVEMENT_REPORT.md), the
[updated proposal](proposals_updated(2).md#proposal-1-multi-fidelity-correction-field-via-latent-space-gaussian-process),
and [reproduction/availability contract](P1_REPRODUCIBILITY.md).
The committed [text evidence manifest](p1_published_results/text_artifact_manifest.json)
indexes unchanged JSON/CSV study outputs. Historical campaign and local-archive
instructions below are provenance, not claims that bulk data are on GitHub.

| Latest IID result, 32 independent geometries | LF | Frozen selected PCA–GP |
| --- | ---: | ---: |
| Geometry-median field RMS | 0.01476687 | 0.005104386 |
| Geometry-median frequency error % | 4.585452 | 0.9468742 |
| Calibrated label1-screening HF referrals | 24/32 | 4/32 |
| Observed false acceptances / rejections | 0 / 0 | 0 / 0 |

All80 new COMSOL references succeeded after method freeze;90 matched-budget
trials include a genuine neural INR. No global mean or residual-scale upgrade
was promoted. Broad intervals, weak discrimination, individual-field reference
failures and one challenge-screening false rejection remain disclosed.
Geometry medians are not pooled mode-row medians; the selected model's pooled
IID field/frequency medians are0.00559255/1.012241% over298 accepted rows.

**Publication restriction:** code, meaningful tests, Markdown and reviewed text
results only. No MPH files, ZIP archives, raw bulk arrays, binary fitted weights
or P1 GitHub Release uploads. Retained numerical data and fitted models use
unpacked `p1_improvement_20261005/` folders under `plate_main`; no ZIP is needed
for replay. Authorized cleanup removed native MPH files and derived outputs;
a public checkout still lacks the ignored raw inputs/weights.

[P1_RESULTS_REVIEW.md](P1_RESULTS_REVIEW.md) retains the historical
10,000-LF/80-HF audit and separates it from the current prospective conclusions.

## Frozen design and raw runs
The original proposal text specifies Abaqus for HF. The selected execution
contract here is the implemented five-variable COMSOL path, using the
`comsol-interp` bundle contract. A full ten-variable extension is a separate
study and is not the command path documented below.

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

The previously generated five-variable pilot LF/HF bundles were removed
before the current P1 rerun. They were COMSOL/FSDT pilot artifacts, not valid
claim evidence. Keep the frozen design table and source solvers as reference
material for regression checks.

## Historical pilot outputs

The generated pilot correction revision and diagnostic figures were removed
before the current P1 rerun. Recreate them only if a pilot comparison is
specifically required; do not mix them with the new claim dataset.

The new P1 target revision must be generated under a new root after the
10,000--20,000-row five-variable LF sweep and the declared 50--100-run COMSOL
HF plan have passed their preflight gates.

## Scope and claims

The retained source code describes a five-variable structural pilot with fixed
materials, dimensions, and CCCC treatment. The workflow below uses that
explicit contract for a fresh P1 COMSOL study.

The removed pilot artifacts do not provide the new P1 design, LF bundles,
COMSOL runs, calibrated uncertainty, or claim report. Those artifacts must be
recreated under the new root and pass all gates.

The retained LF implementation uses layerwise modified shear-correction
factors from laminate integrals. Its five-variable mechanics mapping is the
one used by the runnable workflow below.

## Historical mesh spot check

The generated mesh spot-check CSVs were removed with the pilot outputs. The
source script remains because it documents the legacy COMSOL mesh procedure;
it is not a substitute for a fresh declared COMSOL run.

## Retained source and reference material

The frozen five-variable table, source paper, constitutive code, plate solver,
COMSOL driver, tests, and mesh script remain in the repository for regression
checks and reproducible reruns.

The P1 workflow writes all new artifacts below one versioned root: the design
table, raw LF bundles, HF plan, COMSOL bundles, target revision, holdout
output, train-only surrogate, and claim report.

Split all model evaluation by `run_id`, never by pixels or individual modes.
Report P1 results only after the final preflight and claim guard pass.

## P1 surrogate implementation

`p1_surrogate.py` and `train_p1_surrogate.py` implement the CPU-portable
surrogate stage without a PyTorch dependency. The model accepts the parameter
schema stored in the target revision, including the full ten-parameter
proposal contract. It provides:

- linear boundary-enforced PCA spatial decoding;
- exact ARD Gaussian-process latent and scalar relative-frequency corrections;
- a Fourier-feature ridge coordinate proxy and autoregressive co-kriging baseline;
- deterministic grouped validation by `run_id`;
- diagnostic figures and uncertainty/coverage metrics.

The completed five-variable artifacts are retained in `p1_data_five/`:
48 training, 16 calibration and 16 test geometries, with 721 accepted target
rows. Use `--run-plan` so the deployable model is fitted only on frozen
training IDs. The full ten-variable physics adapter is not implemented.

Do not regenerate this frozen campaign just to inspect it. The generation
commands below describe the pipeline; use a new versioned root for a new
campaign. Use the recheck commands at the end for existing artifacts.

### 1. Freeze the five-variable COMSOL design

The selected P1 contract is:

```text
alpha, beta, theta_c, eta1, eta2
```

These variables represent core thickness fraction, face-sheet asymmetry,
honeycomb cell angle, cell aspect ratio, and dimensionless wall thickness.
The current LF and COMSOL drivers implement this contract.

Generate a new explicit-ID design table. Use at least 10,000 LF candidates
for the final P1 study:

```powershell
python make_p1_design.py `
  --output p1_data_five/lhs_five.csv `
  --manifest p1_data_five/lhs_five.manifest.json `
  --count 10000 `
  --seed 42 `
  --distribution legacy
```

Gate the design before running any simulator:

```powershell
python p1_preflight.py `
  --stage design `
  --design p1_data_five/lhs_five.csv `
  --design-manifest p1_data_five/lhs_five.manifest.json
```

### 2. Run and verify the LF FSDT study

The existing `fsdt_mode_shapes.py` is the five-variable LF driver:

```powershell
python run_lf_batches.py `
  --driver fsdt_mode_shapes.py `
  --samples p1_data_five/lhs_five.csv `
  --output p1_data_five `
  --modes 16 `
  --order 15 `
  --grid 80 `
  --batch-size 32 `
  --workers 1 `
  --attempts 3
```

Verify all LF bundles:

```powershell
python run_lf_batches.py `
  --driver fsdt_mode_shapes.py `
  --samples p1_data_five/lhs_five.csv `
  --output p1_data_five `
  --modes 16 `
  --order 15 `
  --grid 80 `
  --verify-only
```

Then run the formal LF gate:

```powershell
python p1_preflight.py `
  --stage lf `
  --design p1_data_five/lhs_five.csv `
  --design-manifest p1_data_five/lhs_five.manifest.json `
  --lf-root p1_data_five/lf
```

The final target is 10,000--20,000 valid LF bundles with matching run IDs
and five-parameter provenance.

### 3. Select HF COMSOL runs from LF sensitivity

Use LF outputs only to compute sensitivity and select the fixed HF plan:

```powershell
python p1_sensitivity.py `
  --samples p1_data_five/lhs_five.csv `
  --lf p1_data_five/lf `
  --output p1_data_five/hf_plan `
  --modes 16 `
  --hf-count 80 `
  --test-count 16 `
  --calibration-count 16 `
  --seed 42
```

Inspect:

```text
p1_data_five/hf_plan/sensitivity.csv
p1_data_five/hf_plan/hf_plan.csv
p1_data_five/hf_plan/hf_plan.json
```

Do not change the plan after looking at HF results.

### 4. Run the COMSOL HF plan

The existing `run_hf_batches.py` and `hc_HighFidelity_LHS.py` implement this
five-variable COMSOL path.

Dry-run first:

```powershell
python run_p1_hf_plan.py `
  --driver run_hf_batches.py `
  --samples p1_data_five/lhs_five.csv `
  --plan p1_data_five/hf_plan/hf_plan.csv `
  --output p1_data_five `
  --role all `
  --cores 1 `
  --java-heap-gb 8 `
  --mesh-size 4 `
  --attempts 3 `
  --dry-run
```

Execute the plan:

```powershell
python run_p1_hf_plan.py `
  --driver run_hf_batches.py `
  --samples p1_data_five/lhs_five.csv `
  --plan p1_data_five/hf_plan/hf_plan.csv `
  --output p1_data_five `
  --role all `
  --cores 1 `
  --java-heap-gb 8 `
  --mesh-size 4 `
  --attempts 3
```

HF workers now have a **360-minute per-worker deadline**, with no retry
after timeout, and a heartbeat every 60 seconds. Configure these with
`--timeout-minutes` and `--heartbeat-seconds` on either HF entrypoint.
Detailed stage output is durable in `<output>/worker_logs/`; it identifies
geometry/mesh construction, eigensolve/model save and field extraction.
Timeout or interruption stops the worker process tree. Loss of the
launching parent is detected within five seconds at each supervisor level;
the outer plan exits so the batch driver can stop its worker and clean scratch.

Use `--scratch-root D:/` to create a fresh worker-owned TEMP/TMP and Java
temporary directory on that existing volume. The wrapper removes only its
owned scratch directory after the worker exits or is stopped. This execution
setting does not change geometry, material, boundary conditions or extraction.
The deadline/descendant regression, actual parent-exit smoke and licensed
scratch/restart CLI smoke pass; the current P1 suite contains 34 passing tests.

Validate the COMSOL exports:

```powershell
python p1_preflight.py `
  --stage hf `
  --design p1_data_five/lhs_five.csv `
  --design-manifest p1_data_five/lhs_five.manifest.json `
  --lf-root p1_data_five/lf `
  --hf-plan p1_data_five/hf_plan/hf_plan.csv `
  --hf-root p1_data_five/hf `
  --hf-extraction comsol-interp
```

#### Windows heap and interrupted-run recovery

Use `run_hf_batches.py` or `run_p1_hf_plan.py` for production execution.
The batch runner starts each Windows worker with MPh's in-process COMSOL
backend. This avoids the separate server launcher's `comsolmphserver.ini`
heap limit overriding `JAVA_TOOL_OPTIONS`. `--java-heap-gb` replaces an
inherited `-Xmx` for the worker process only; it does not edit the COMSOL
installation or change mesh, eigenmode, or extraction settings.

The worker exits through MPh's `sys.exit()` hook. A failed COMSOL run now
returns failure to the batch runner instead of being reported as successful.
Completed exports retain their original solver-source provenance.

After Ctrl+C, do not delete completed NPZ bundles or alter the frozen plan.
Retry an unfinished run at the same mesh setting:

```powershell
python run_hf_batches.py `
  --samples p1_data_five/lhs_five.csv `
  --output p1_data_five `
  --runs 373 `
  --cores 1 `
  --mesh-size 4 `
  --java-heap-gb 8 `
  --attempts 1
```

Then resume the original plan command. Compatible completed exports are
reused. Keep enough RAM outside the Java heap for COMSOL's native solver and
Windows; increasing heap size is not a mesh-accuracy change.


### 5. Build targets, evaluate holdout, and train

Build the immutable five-variable correction revision:

```powershell
python p1_build_targets.py `
  --data p1_data_five `
  --plan p1_data_five/hf_plan/hf_plan.csv `
  --name corrections_five `
  --scope pilot-five-variable `
  --modes 10
```

Run strict train/calibration/test evaluation:

```powershell
python p1_holdout.py `
  --data p1_data_five `
  --name corrections_five `
  --plan p1_data_five/hf_plan/hf_plan.csv `
  --output p1_data_five/holdout_corrected
```

Optionally run the decoder gate:

```powershell
python p1_decoder_upgrade.py `
  --data p1_data_five `
  --name corrections_five `
  --plan p1_data_five/hf_plan/hf_plan.csv `
  --output p1_data_five/decoder_gate_train_new
```

The gate excludes calibration and test runs and compares the linear and
nonlinear decoders on identical training-run CV folds. The older
`decoder_upgrade/` output used all 80 geometries and is historical, not
a valid promotion gate.

Fit the deployable surrogate only on frozen `train` IDs:

```powershell
python train_p1_surrogate.py `
  --data p1_data_five `
  --name corrections_five `
  --run-plan p1_data_five/hf_plan/hf_plan.csv `
  --output p1_data_five/surrogate_five
```

#### Interpreting target and holdout results

`accepted` and `quarantined` count run-mode pairs, not simulations. With 80
HF runs and 10 requested modes there are 800 candidate pairs. Rejected
near-degenerate or low-MAC labels remain excluded; they are not failed runs.

Target parameter columns are explicitly named by `parameter_names`.
Their order may differ from the design CSV after sorted JSON serialization.
The gates require the same unique names, not identical column order. Keep
the stored names and numeric columns together; never rename or reorder only
the names of an existing target or model.

Field RMS is measured on normalized mode-shape corrections, not in metres
and not as a percentage of the correction itself. `field_relative_rms`
divides that prediction error by the RMS of the true HF-minus-LF correction.
`frequency_error_pct` is signed; use absolute errors for accuracy claims.
Median pointwise coverage of 1.0 is not perfect prediction accuracy or
run-level simultaneous coverage.

The training decoder's `decoder_explained_variance` now uses residual and
target variances in the same normalized interior-field space. Historical
metrics used an invalid mixture of singular-value energy and output
variance; do not reuse that statistic. The corrected frozen-training value
is 0.999255561, not a held-out generalization score.

Conformal scales use the `ceil((n+1)*coverage/100)`-th smallest run score.
Finite 90% intervals require at least 9 calibration runs; finite 95%
intervals require at least 19. The default 16-run calibration split therefore
supports 90%, not finite 95%, intervals. Only the requested level is labelled
as calibrated in the report. The reported score multiplier is divided by the
normal quantile when scaling predictive standard deviations.

#### Refreshing reports produced before the calibration correction

Older reports used the wrong quantile tail and must not support calibrated
coverage claims. Preserve their raw predictions and reuse the original
train-only model to generate a new report without refitting:

```powershell
python p1_holdout.py `
  --data p1_data_five `
  --name corrections_five `
  --plan p1_data_five/hf_plan/hf_plan.csv `
  --model p1_data_five/holdout/latent_gp_train `
  --output p1_data_five/holdout_corrected
```

The saved model must match the target parameter order, grid, and frozen
training rows. For a repaired evidence package, use
`p1_data_five/holdout_corrected/holdout_metrics.json` in both commands below
instead of the original `holdout/holdout_metrics.json`. Simulations, targets,
and the deployable surrogate do not need to be regenerated for these repairs.

#### Export cached errors and figures without rerunning models

The existing plotting entrypoint also exports strict-test error tables and
separately labelled training-run cross-validation comparisons:

```powershell
python plot_p1_results.py `
  --data p1_data_five `
  --name corrections_five `
  --design p1_data_five/lhs_five.csv `
  --holdout p1_data_five/holdout_corrected `
  --surrogate p1_data_five/surrogate_five `
  --out p1_data_five/results_new
```

This command reads cached arrays and CSV metrics. It does not run COMSOL,
FSDT, target building, or model training. Rerunning it replaces only the
derived tables and figures in the chosen output directory. `--design`
explicitly selects the design CSV; no historical design outside the dataset
is selected implicitly.

Outputs:

- `results_summary.json`: absolute frequency errors, normalized field RMS,
  improvement counts, pointwise and simultaneous run-level coverage, and
  mean interval widths.
- `test_metrics.csv`: one row per accepted test pair and model, comparing
  uncorrected FSDT with latent-GP-corrected FSDT.
- `test_by_mode.csv`: median, 95th-percentile, and maximum errors by mode.
- `cv_model_comparison.csv`: co-kriging, INR, and latent-GP field errors from
  grouped cross-validation on the frozen training runs.
- `test_frequency_parity`, `test_error_by_mode`,
  `test_uncertainty_coverage`, and `test_shape_examples`: strict HF test
  figures; the shape examples are the median-error and worst-error cases.
- `cv_model_comparison`: training-run CV, not an HF-test comparison.
- Design coverage, accepted LF/HF pairs, MAC matching quality, quarantine
  reasons, and an accepted target example.

Every figure is saved as PNG and vector PDF. The holdout target hash and
accepted test-pair identities must match the revision. CV model rows must
match the frozen accepted training pairs. Baseline CV results must not be
presented as baseline performance on the separate HF test runs.

### 6. Run final preflight and claim guard

```powershell
python p1_preflight.py `
  --stage final `
  --design p1_data_five/lhs_five.csv `
  --design-manifest p1_data_five/lhs_five.manifest.json `
  --lf-root p1_data_five/lf `
  --hf-plan p1_data_five/hf_plan/hf_plan.csv `
  --hf-root p1_data_five/hf `
  --hf-extraction comsol-interp `
  --targets p1_data_five/corrections_five `
  --holdout p1_data_five/holdout_corrected/holdout_metrics.json
```

```powershell
python p1_claim_report.py `
  --design p1_data_five/lhs_five.csv `
  --lf-root p1_data_five/lf `
  --hf-root p1_data_five/hf `
  --hf-plan p1_data_five/hf_plan/hf_plan.csv `
  --hf-extraction comsol-interp `
  --targets p1_data_five/corrections_five `
  --holdout p1_data_five/holdout_corrected/holdout_metrics.json `
  --output p1_data_five/p1_claim_report.json
```

The final report must say:

```text
ready_for_scientific_review
```

This status checks artifact/count/contract readiness, not completion of
Astra's scientific requirements. Observed coverage on the deterministic
maximin split is not an unconditional conformal guarantee or a universal
four-percent error claim.

### 7. Recheck the frozen P1 evidence without overwriting it

```powershell
python p1_recheck.py `
  --data p1_data_five `
  --output p1_data_five/recheck_new
```

The output directory must not already exist. The script rebuilds all
accepted target arrays, 800 pairing decisions and 721 field CSVs from the
planned raw LF/HF runs in a temporary same-filesystem directory; replays
the saved model and calibration; refits seeded baselines on training rows;
checks held-out decoder reconstruction and geometry/within-mode uncertainty;
performs a genuine retrospective theta_c >=60-degree region exclusion; and
MAC/phase-aligns the existing mesh refinements to accepted target identities.
It does not regenerate the 10,000 LF or 80 COMSOL runs.

The completed recheck is in `p1_data_five/recheck_20261003/`. In
`P1_RESULTS_REVIEW.md`, §11 distinguishes local authentication from public
reproducibility and scientific completion. The primary accuracy numbers and
trained models are unchanged. The new region experiment has only 6/9
simultaneous frequency coverage at nominal 90%; it is not a successful
calibrated-transfer claim. The original Fourier baseline is a ridge proxy,
not a trained neural INR.

Current campaign source is not fully tracked, and `.npz` arrays are ignored
by the repository. A matching source commit and publicly retrievable,
checksum-identified artifact release remain necessary for external review.

The final current-source audit is in
`recheck_20261003/independent_verified/`; the P1/physics regression run passes
34 tests, including decoder-gate held-out-label isolation and the corrected
variance diagnostic's equal-energy analytical case.

### Follow-up execution status

`p1_data_five/hf_followup_20261003/` preserves seven additional full COMSOL
solutions: mesh 3/2 for runs 2209 and 4352, mesh 1 for 2209, and the
compressed-MUMPS mesh-3 control and mesh-2 refinement for 3408. The first
3408 alternate-scratch probe was stopped after an excessive wait, without
a saved NPZ. The user then authorized a fresh bounded resumption, recorded
in `resume3408_protocol.json`; its supervisor/stage logs are durable.
That default-MUMPS resumption was interrupted after its last heartbeat at
299.5 minutes, without a saved reference. No default-MUMPS solver remains
running; its owned scratch was removed. `resume3408_outcome.json` records
the interruption rather than inventing a numerical failure or timeout.

The PARDISO alternative passes a tiny native API smoke, but its full
same-mesh run-3408 control exceeded its 60-minute cap (exit 124), without a
saved reference. Its native trace identifies 5,857,248 DOFs and a switch
to out-of-core factorization; the worker stopped and scratch was removed.

`run3408_solver_check.py` now supports explicit `--solver pardiso` or
`--solver mumps-blr`, using the unchanged physics builder and the same
32 candidates/16 extracted modes/80×80 extraction. The MUMPS block-low-rank
variant uses tolerance 1e-8 with original error checks retained. Its tiny
method smoke passes (maximum field RMS 8.48e-15); this is not a full reference.
The full same-mesh control saved successfully (exit 0; scratch removed):
maximum frequency change 0.000120%, field RMS 0.000003813, zero identity
flags, all ten accepted modes. `mumps_blr_control_results.json` records the
comparison against the original reference under the predeclared
0.01% / 0.000075-RMS limits. The eigensolve took 1255.0 seconds.
The validated mesh-2 refinement also saved successfully (exit 0; scratch
removed). `mumps_blr_refinement_protocol.json` preserves its settings:
four cores, 8-GiB heap, fresh D scratch and 360-minute cap. It solved
10,865,214 DOFs, converged all 32 candidates and exported the original
16-mode/80×80 contract. The eigensolve took 11025.5 seconds.
`finest3408_results.json` includes a same-method compressed mesh 3→2
comparison: maximum frequency change 0.054862%, field RMS 0.004797,
zero identity flags. Frequency passes the 0.1% diagnostic target; the
accepted mode-10 shape fails the 0.00075 field target. Nominal mesh 4→2
field drift reaches 0.027334. The unchanged surrogate's median/max
field error against mesh 2 is 0.006665/0.159342, with median frequency
error 1.1626%; simultaneous field and frequency run-level coverage pass.
No model or calibration was refitted. No owned follow-up solver is running.
Completion evidence: [final verification](p1_data_five/hf_followup_20261003/final_verification.json)
and [execution inventory](p1_data_five/hf_followup_20261003/execution.json).
The frozen design, plan, training archive and holdout metrics hashes match;
the physical-builder and independently audited computational sources match.

Saved-control analysis: `python p1_data_five/hf_followup_20261003/analyze_solver_control.py --protocol p1_data_five/hf_followup_20261003/mumps_blr_control_protocol.json`.
This starts no COMSOL process and fits/recalibrates no model.

Saved-result analyses (raw inputs unchanged; derived outputs regenerated):

```powershell
python p1_data_five/hf_followup_20261003/analyze.py
python p1_data_five/hf_followup_20261003/analyze_finest.py --run 2209
python p1_data_five/hf_followup_20261003/analyze_finest.py --run 3408
python p1_data_five/hf_followup_20261003/analyze_mode7.py
```

These commands do not launch COMSOL or fit a model; they regenerate only
derived analysis outputs. Run 2209's accepted shape tail still changes
by 0.017777 RMS between mesh 2 and 1, and run 3408's by 0.004797 between
compressed meshes 3 and 2. Do not present the numerical-reference
requirement as closed. Frozen nominal-mesh-4 headlines and
model/calibration artifacts have not been overwritten.

The saved modal diagnostic identifies a close HF mode-8/9 pair with a
0.71–0.77% frequency gap, just above the original 0.5% near-repeat cutoff.
The individual mode's mesh4-to1 MAC falls to 0.9513, while the extracted
two-field span has maximum principal angle 0.0524 degrees. This supports
basis sensitivity, not individual-shape convergence. No accepted row,
target or calibration is silently removed or changed.

## P1 improvement study and prospective release

The separately preserved study root is now `plate_main\p1_improvement_20261005`,
moved unpacked from `D:\p1_improvement_20261005`. See
[local layout and verified no-ZIP replay](P1_REPRODUCIBILITY.md#local-unpacked-study--no-zip-required).
The original `p1_data_five` campaign, targets and nominal model predictions
are not overwritten. The study includes mass-weighted shell-state reference
diagnosis, train-only error decomposition and promotion gates, matched
8/16/24/32/48-HF learning curves, and an actual coordinate-conditioned
PyTorch neural INR. The neural correction enforces transverse zero values
at the perimeter; it does not enforce exact CCCC normal derivatives or
full modal/operator consistency.

All 90 matched-budget trials completed. Neither a new global mean nor the
local-residual uncertainty recipe passed its promotion gate. The original
PCA48/RBF mean and raw scales remain selected globally. The regional
co-kriging model is fitted/tuned only on the 34 historical training
geometries below 60 degrees; challenge results are not coverage guarantees.

Six final fitted models, the original stable-mode reference gauge, physical
solver sources, scale recipes, screening threshold and method sources were
frozen at `2026-10-06T09:39:20.827965+00:00`. Only afterward were 32 independent
calibration, 32 IID evaluation and 16 theta>=60 challenge geometries drawn.
All six models produced blinded predictions using real LF inputs and no HF
labels before native acquisition. Nominal COMSOL references use the unchanged
shell builder, hauto4, 32 candidates/16 guarded flexural exports, 80x80 grid,
shift1000, four cores and the original solver. No full prospective MPH is
saved. One native worker runs at a time, with an 8-GiB heap, fresh D scratch,
360-minute deadline, durable progress and no unchanged automatic retry.

The historical run9828 acquisition control reproduces nominal frequencies
within `4.99287e-10%` and phase-aligned fields within `4.69784e-13` RMS.
The complete measured claim disposition is generated from actual final
artifacts as `P1_IMPROVEMENT_REPORT.md` and `improvement_report.json` in the
study root. No successful-stage checkpoint is silently overwritten.

Fresh scientific rerun, from the matching `plate_main` source directory:

```powershell
$env:P1_ORIGINAL_ROOT = (Get-Location).Path
$study = 'D:\p1_improvement_fresh'
python p1_improve.py freeze --output $study
python p1_reference_diagnosis.py top --output $study
python p1_improve.py decompose --output $study
python p1_improve.py select --output $study
python p1_improve.py select-region --output $study
python p1_improvement_study.py learning-curves --output $study
python p1_improvement_study.py uncertainty --output $study
python p1_improvement_study.py uncertainty-region --output $study
python p1_reference_diagnosis.py native --output $study
python p1_reference_diagnosis.py analyze --output $study
python p1_prospective.py native-smoke --run 9828 --output $study
python p1_improve.py prepare-source --execution-stage freeze-models --output $study
$code = "$study\execution_sources\freeze-models"
python "$code\p1_improvement_study.py" freeze-models --output $study
python "$code\p1_prospective.py" plan --output $study
python "$code\p1_prospective.py" lf --output $study
python "$code\p1_prospective.py" blind --output $study
python "$code\p1_prospective.py" hf --group calibration --output $study
python "$code\p1_prospective.py" calibrate --output $study
python "$code\p1_prospective.py" hf --group evaluation --output $study
python "$code\p1_prospective.py" evaluate --output $study
python p1_delivery.py replay --output $study
python p1_delivery.py report --output $study
python p1_delivery.py pack --output $study
python p1_delivery.py verify --output $study
```

Stop on a nonzero exit; inspect the preserved outcome rather than replaying
an unchanged failed attempt. Calibration changes interval multipliers only;
evaluation labels never select a mean, scale recipe or screening decision.
The frequency task screens stable reference label1 against the frozen
961.333328052-Hz benchmark, not an externally certified service constraint
or a guaranteed complete-spectrum fundamental frequency.

The current local handoff is unpacked: original numerical arrays remain in
`p1_data_five/`, and the new study's models, labels, blinded predictions,
calibration and execution snapshots are in `p1_improvement_20261005/`.
Original source revisions remain in `baseline_sources/` and `package_sources/`.
The earlier relocation authenticated every original baseline/captured member.
Subsequent authorized cleanup removed seven native MPH files, caches, generated
figures, duplicate exports and publication bookkeeping; it did not change
scientific inputs/weights. See [cleanup manifest](p1_published_results/cleanup_manifest.json).

Run `python p1_delivery.py replay --output .\p1_improvement_20261005` from
`plate_main` with `P1_ORIGINAL_ROOT` set there. No ZIP extraction, solve or
scientific refit is needed. Post-cleanup six-model/800-row replay again matched
every prediction array exactly. Saved-MPH native re-extraction now requires
restoring those files from a separately retained backup or running new solves.
The fresh native workflow above must not be treated as runnable from a complete
saved-MPH archive in this cleaned folder. Recorded numerical outcomes and
negative results remain unchanged; the original inventories are historical.
See [relocation verification](p1_published_results/unpacked_relocation_verification.json).
No archive, MPH file, raw binary array, fitted weight or COMSOL license is uploaded.

Prospective acquisition completed all **80/80** native references without a
failed run: 32 calibration and 48 evaluation/challenge. Accepted calibration
targets are 295/320 rows; accepted evaluation targets are 448/480 rows.
Quarantined identities remain recorded rather than fabricated or relabeled.
IID geometry-median field RMS decreases from 0.0147669 (LF) to 0.00510439
(frozen selected mean); frequency error decreases from 4.58545% to 0.946874%.
Joint run coverage is29/32 at90% and32/32 at95%, but the95% average joint
field width is0.835769 and frequency width2833.78Hz; discrimination is weak.
The original globally selected mean is not replaced by a family that happens
to score better on these evaluation labels.

The selected95% mode1 screening policy refers4/32 IID cases to HF, with
zero observed false acceptances/rejections and28 verified correct automatic
decisions. Calibrated LF-only screening refers24/32. Challenge screening
refers4/16, with zero false acceptances but one false rejection. Reference
HF time for the IID cases is6839.24s; selected-policy referral HF time would
be1075.65s. The campaign actually paid every reference cost; this is a
counterfactual operational saving, not a net study-cost or zero-risk claim.

All800 LF-only rows replay exactly for all six fitted models when preserving
the original inference batch layout. A smaller float32 neural replay batch
differed by up to1.05e-8; its failure record is retained. The scientific
weights, predictions and selection were not changed and replay tolerances
were not relaxed. Behavioral verification passed43 workflow tests plus the
separately added fitted direct-HF/LF-invariance test.

