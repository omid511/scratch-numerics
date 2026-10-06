# P1 source, text results and reproduction

## Publication boundary

The current P1 result is the [prospective final report](P1_IMPROVEMENT_REPORT.md).
The [updated proposal](proposals_updated(2).md#proposal-1-multi-fidelity-correction-field-via-latent-space-gaussian-process)
distinguishes the original research plan, executed scope and remaining extensions.
The [historical audit](P1_RESULTS_REVIEW.md) retains earlier results and negative findings.

**GitHub publishes source, tests, Markdown and JSON/CSV evidence only.**
No `.mph` simulation file, ZIP archive, raw NPZ/NPY dataset, fitted binary weight,
COMSOL binary/license or unrelated P2 change is uploaded. There are no P1 bulk
GitHub Release assets. The full immutable archives remain local by explicit
publication restriction; do not describe this as a complete public data/model release.

Published evidence is indexed by
[text_artifact_manifest.json](p1_published_results/text_artifact_manifest.json).
Each retained study JSON/CSV has its original byte count and SHA-256; exporting
text results changes no scientific artifact. Original absolute paths in those
records are execution provenance, not paths required by the public checkout.

The complete source/text inventory is in
[publication_manifest.json](p1_published_results/publication_manifest.json);
[source_manifest.json](p1_published_results/source_manifest.json) records all 43
Python files and [historical_text_manifest.json](p1_published_results/historical_text_manifest.json)
retains the referenced earlier audit evidence. Scoped `.gitattributes` disables
newline conversion so a Windows checkout preserves the frozen SHA-256 bytes.

The committed [package manifest](p1_published_results/package_manifest.json),
[package verification](p1_published_results/package_verification.json) and
[full restoration smoke](p1_published_results/archive_restoration_smoke.json)
refer to **local** archives and recorded local checks. They do not authenticate
remote downloads, because the archives are not published.

## Recorded environment

Execution: Windows x64, Python 3.14.5; package versions and two-thread settings
are in [runtime_environment.json](p1_published_results/runtime_environment.json).
The installed PyTorch wheel was `2.14.1+cpu`; neural predictions and checkpoints
are CPU-loaded. The pinned source dependencies are in
[requirements_p1.txt](requirements_p1.txt). SymPy supports the retained optional
basis-derivation helper; it is not an additional scientific model.

From `plate_main`, in an isolated Python environment:

```powershell
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_p1.txt
```

These commands describe environment recreation, not a new validation run.
Exact cross-platform/BLAS reproducibility is not established by the Windows
restoration smoke. Native reruns additionally need an installed licensed
COMSOL Multiphysics 6.4 with Structural Mechanics and the MPh/JPype bridge.
No credential or license file is supplied.

## What a clean public checkout can do

- Inspect the frozen method, independent calibration, accepted/quarantined
  populations, comparator rows and decision verification without bulk arrays.
- Run deterministic synthetic behavioral regressions without COMSOL:

```powershell
python -m unittest test_p1_improvement test_p1_full_workflow test_p1_pairing test_p1_surrogate test_regressions
```

- Recreate the historical matched-budget figures from recorded trial metrics,
  without fitting a model:

```powershell
python -c "import json; from pathlib import Path; from p1_improvement_study import plot_learning_curves; p=Path('p1_published_results/learning_curves'); plot_learning_curves(p,json.loads((p/'results.json').read_text())['trials'])"
```

This writes derived figures locally; generated figures are not commit inputs.
Source tests exercise synthetic training where needed; they do not retrain the
published scientific models. The completed scientific verification is retained
rather than substituted by these synthetic checks.

### Verified source/text checkout — 2026-10-06

[publication_verification.json](p1_published_results/publication_verification.json)
records a fresh index checkout, not an import from this dirty working tree:
all 43 source files parsed, repository-relative Markdown destinations resolved,
and all 150 source/study/historical manifest records matched their exact bytes. The documented
synthetic command passed **44 tests**. Both learning-curve figures were recreated
from saved trial JSON and visually inspected, without fitting.

Using the separately retained local study inputs, this checkout also replayed
**800 rows across all six frozen models**, with zero maximum differences in every
field/frequency mean/scale array. No COMSOL solve or scientific model refit ran.
This proves the source handoff with authorized local inputs, not public bulk-data
availability.

## What requires the retained local archives

The original audited source/raw data/native references are in local
`audited_baseline.zip` (23,527,900,518 bytes):

```text
25cdcb1b746643c86a6ad1d140c34e0d8c868fc7cb390e854c5e90a81c3ccd28
```

The final improvement study/source revisions/fitted weights/blinded predictions
are in local `p1_improvement_artifacts.zip` (3,846,098,878 bytes):

```text
bb1cd1cf53da7cfe73b491063a92501c33d738f97b39339354a7fb5305255a50
```

They are not committed or uploaded. A source-only clone therefore cannot replay
frozen field predictions, independently rebuild targets from raw modal arrays,
or rerun saved-MPH extraction without separately available local inputs.

For an authorized local copy:

1. Extract the baseline to a fresh `plate_main` directory.
2. Extract the improvement archive separately; its `study/` is the study root.
3. Overlay its `source/plate_main` or the matching published P1 source.
4. Set `P1_ORIGINAL_ROOT` to the restored `plate_main` before executing captured
   methods. Never edit frozen execution snapshots or recorded hashes.
5. Use the recorded original inference batch layout:

```powershell
$env:P1_ORIGINAL_ROOT = (Get-Location).Path
python p1_delivery.py replay --output '<restored-release>\study'
```

`replay` loads the six fitted models and real LF-only inputs. It does not solve
COMSOL, fit a model or read HF labels into predictors. Local portable restoration
reproduced all 800 blinded rows with zero differences for field means/scales and
frequencies/scales. Float32 neural inference on a smaller 40-row batch differed
by up to 1.05e-8; the failed subset attempt is retained and replay tolerances were
not relaxed. Replay updates only its derived verification JSON, not frozen data.

`p1_delivery.py verify` checks actual archive hashes when both archives and their
package manifest are colocated. The historical fresh-science commands in
[P1_DATA_README.md](P1_DATA_README.md#p1-improvement-study-and-prospective-release)
require the local data/native references and a **new output root**. They are not
steps run for this source/text publication. All successful scientific stages,
reference failures, quarantines, calibration roles and negative promotion
outcomes remain frozen.

## Scientific limits

- Geometry medians and pooled mode-row medians are distinct populations.
- The selected global mean is PCA48/RBF, not a neural autoencoder. Genuine neural
  INR and the historical Fourier-ridge proxy are separate comparators.
- Broad conformal intervals and finite ranks do not imply sharpness, purely
  epistemic uncertainty, conditional coverage or OOD validity.
- The regional model excludes `theta_c >= 60` during fitting/tuning; the global
  model does not. Regional challenge coverage is diagnostic only.
- Stable reference label1 screening at 961.333328 Hz is a benchmark task, not
  a complete-spectrum fundamental-frequency or service-safety certificate.
- Validation paid all HF references. Operating savings are counterfactual and
  retain upfront training/calibration costs; zero observed false accepts is
  not zero physical risk.
- Sampled mass-weighted two-mode shell states support basis mixing, not full-DOF
  or campaign-wide convergence. Individual-field tolerance failures remain.
- No Abaqus, damage, aerodynamic, ten-variable or experimental validation is
  established by the fixed-material/CCCC five-variable COMSOL study.
