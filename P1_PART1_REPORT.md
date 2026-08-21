# P1 Part-1 Report — Real HF Correction Dataset Incorporation

## Dataset scope

Proposal-1 Part-1 delivers a **real high-fidelity (HF) dataset** for the
multifidelity correction pipeline: 100 honeycomb-sandwich designs sampled by
sensitivity-warped LHS, each solved in COMSOL (shell model) and with the
in-house FSDT solver. Per design: 10 mode frequencies on both sides and
80×80 mode-shape grids indexed `[y, x]`, from which spatial correction fields
Δw = COMSOL(norm) − FSDT(norm) are derived.

- Design parameters: `alpha, beta, theta_c, eta1, eta2` (see
  `data/p1_part1/MANIFEST.md` for ranges).
- HF model: CCCC 300×300×10 mm Al honeycomb sandwich plate, COMSOL shell,
  `hauto = 4` mesh (~94k elements).
- Canonical access: `mechanics.p1_multifidelity.hf_dataset`;
  CLI: `run_p1_part1.py {summary,plots,corrections,crosscheck}`.

## Headline numbers

From the committed `data/p1_part1/correction_freq.csv`
(FSDT vs COMSOL relative frequency error, |rel%|):

| Statistic                     | Value |
|-------------------------------|-------|
| Overall median \|rel%\|       | ~16.8% |
| Overall mean \|rel%\|         | ~17.3% |
| Overall p90 \|rel%\|          | ~29.7% |
| Single worst entry            | ~47.6% |
| Best run (mean over modes)    | run 5, ~2.5% |
| Worst run (mean over modes)   | run 14, ~39.8% |

Typical errors sit around **15–30%**, with per-mode means ranging ~13–22%
(modes 2, 5, 9 worst; modes 3 and 6 mildest). The best individual entries
reach a few hundredths of a percent, but no design is uniformly accurate:
FSDT-vs-COMSOL disagreement is substantial across the whole design space.
This is exactly the signal the correction model must learn.

## Mesh convergence justification

The production mesh uses `hauto = 4` (~94k elements). The study recorded in
`mesh_convergence_final.csv` shows mean frequency error vs published paper
FEM values of **~0.49%** at this setting, while refining to ~179k elements
(`hauto = 1`) shifts frequencies by <0.02% — so discretization error is two
orders of magnitude below the FSDT-vs-COMSOL discrepancy being modeled.
Mesh choice is therefore not a meaningful contributor to the correction
signal; solve cost is ~37% lower than the finest converged mesh (119 s vs
190 s per eigenfrequency solve).

## REQUIRED caveats

These caveats are load-bearing for anyone training on or interpreting this
dataset:

**(a) Symmetric-face simplification in the COMSOL model.**
COMSOL models symmetric faces (`h_face = (h1+h3)/2`) while FSDT honors the
true face asymmetry (`beta`). Part of the learned "correction" is therefore
*model-simplification error*, not pure FSDT method error — the correction
model will absorb whatever bias this asymmetry introduces, and predictions
degrade for designs where asymmetry matters most.

**(b) Geometry always regular hexagons.**
The COMSOL geometry always uses regular hexagons: `eta1` affects only the
equivalent material properties fed to the model, and `theta_c` is declared as
a parameter but has no effect on geometry (the cell angle is fixed). Treat
high-`eta1` / high-`theta_c` samples cautiously: their HF reference does not
exercise the geometric physics those parameters nominally describe.

**(c) Penalty-BC edge artifacts in FSDT.**
The FSDT solver's penalty spring boundary conditions produce edge artifacts
(~8% peak amplitude at clamped edges). Correction fields near the boundary
mix genuine modeling differences with this numerical artifact.

**(d) Degenerate mode pairs make per-mode-index correspondence fragile.**
Several designs exhibit near-degenerate mode pairs (e.g., modes 2/3, 7/8).
Across designs, mode index *i* does not reliably track the same physical
mode; per-mode statistics (and any per-mode-index supervision) inherit this
ambiguity. Prefer shape-based matching or aggregate statistics when strict
mode correspondence matters.

**(e) Normalization: Δw compares max-normalized shapes.**
Correction fields are computed between independently max-abs-normalized mode
shapes. Δw therefore measures **shape difference, not amplitude difference**;
frequency-scale amplitude information is deliberately absent from the spatial
corrections.

## Roadmap alignment

`roadmaps/proposal1_roadmap.md` planned a synthetic-HF placeholder
(physics-informed perturbations added to FSDT solutions) until real Abaqus/FE
data arrived. **This dataset supersedes that placeholder**: the correction
fields here are computed against genuinely independent COMSOL solves, not
perturbed FSDT output, which removes the roadmap's main validity risk
("synthetic HF must be validated before training").

One deliberate deviation: the roadmap specifies zarr stores for packaged
datasets. We keep plain CSV (+NPZ conventions elsewhere in the repo) instead,
to avoid adding a zarr dependency for data that loads fine file-at-a-time.
The pipeline remains storage-agnostic if zarr is introduced later.
