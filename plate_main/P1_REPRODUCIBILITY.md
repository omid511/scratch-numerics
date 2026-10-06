# P1 source, text results and reproduction

## Publication boundary

The current P1 result is the [prospective final report](P1_IMPROVEMENT_REPORT.md).
The [updated proposal](proposals_updated(2).md#proposal-1-multi-fidelity-correction-field-via-latent-space-gaussian-process)
distinguishes the original research plan, executed scope and remaining extensions.
The [historical audit](P1_RESULTS_REVIEW.md) retains earlier results and negative findings.

**GitHub publishes source, tests, Markdown and JSON/CSV evidence only.**
No `.mph` simulation file, ZIP archive, raw NPZ/NPY dataset, fitted binary weight,
COMSOL binary/license or unrelated P2 change is uploaded. There are no P1 bulk
GitHub Release assets. The complete local study now uses unpacked folders under
`plate_main`; its ignored bulk inputs/weights are not a public data/model release.

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

## Local unpacked study — no ZIP required

The working study was moved from `D:\p1_improvement_20261005` to
`plate_main\p1_improvement_20261005`. It is a normal directory, not an archive.
The original audited data stay in `plate_main\p1_data_five`; they were not
overwritten or duplicated. The local layout is:

```text
plate_main/
  p1_data_five/                    original audited data and native references
  p1_improvement_20261005/
    baseline_sources/             original audited source/document revisions
    package_sources/              source revisions captured in the improvement package
    execution_sources/            immutable stage-specific methods
    source_versions/              retained method revisions
    frozen_models/                six fitted models
    prospective/                  LF/HF arrays, blinded predictions and calibration
    reference_native/             extracted shell-state arrays
    unpacked_local_inventory.json local file/dependency hashes
  p1_published_results/            reviewed public text evidence
```

The study directory is deliberately Git-ignored. All current implementation
and behavioral tests are already published at the `plate_main` source root;
historical source snapshots, raw arrays and weights are retained locally.
Existing absolute paths in archived records remain original execution
provenance; they are not rewritten to falsify the historical run.

[unpacked_relocation_verification.json](p1_published_results/unpacked_relocation_verification.json)
records 1,640 moved files independently SHA-256-verified, 1,617 captured
improvement-study members authenticated, and all 11,211 original baseline
members available unpacked with matching hashes. Original baseline/package
source snapshots contain 39/9 files. No ZIP or MPH file was moved into the
new study directory.

From `plate_main`, replay directly:

```powershell
$env:P1_ORIGINAL_ROOT = (Get-Location).Path
python p1_delivery.py replay --output .\p1_improvement_20261005
```

This command passed after the old D-drive working folders were removed:
800 original-batch rows, all six fitted models, zero differences in every
field/frequency mean/scale array. It uses LF-only inputs, not HF labels,
and performs no COMSOL solve, scientific refit or ZIP extraction.
Replay changes only its derived verification JSON. Never edit frozen
execution snapshots or recorded hashes. The retained 40-row neural attempt
differed by up to 1.05e-8; its failure and unchanged tolerances remain recorded.

**The ZIPs are no longer required to preserve or replay this local P1 study.**
The old D-drive directory now contains only the two original ZIP backups,
which were left untouched rather than deleted during relocation. They may
be removed if no additional archive backup is desired; keep a verified
backup of the unpacked folders if deleting them.

### Optional original archive evidence

The original [package manifest](p1_published_results/package_manifest.json)
and [package verification](p1_published_results/package_verification.json)
remain historical evidence. Their original archive hashes are:

| Optional backup | Bytes | SHA-256 |
| --- | ---: | --- |
| `audited_baseline.zip` | 23,527,900,518 | `25cdcb1b746643c86a6ad1d140c34e0d8c868fc7cb390e854c5e90a81c3ccd28` |
| `p1_improvement_artifacts.zip` | 3,846,098,878 | `bb1cd1cf53da7cfe73b491063a92501c33d738f97b39339354a7fb5305255a50` |

The archive-only `p1_delivery.py verify` subcommand requires those ZIPs
and their package manifest colocated; it is not the verifier for the
unpacked working directory. A public clone still cannot reproduce frozen
predictions without the separately retained, ignored inputs/weights.
The historical fresh-science commands in
[P1_DATA_README.md](P1_DATA_README.md#p1-improvement-study-and-prospective-release)
require local data/native references and a **new output root**. They were
not run during relocation. All scientific results and failed promotion
gates remain frozen.

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
