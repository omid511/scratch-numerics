# P1 results review — completed five-variable COMSOL/FSDT campaign

## 1. Conclusion and reporting scope

The completed `p1_data_five` campaign provides a positive **held-out five-variable structural correction result**. On **16 predeclared HF test runs and 150 accepted mode pairs**, latent-GP-corrected FSDT reduces median absolute frequency error from **4.9218% to 1.2558%** and median normalized interior field RMS from **0.013699 to 0.007658**. These are empirical errors against the configured COMSOL reference, not universal physical-error bounds.

Corrected 90% run-level intervals cover all accepted frequencies in **15/16 test runs (93.75%)** and all accepted field pixels/modes in **15/16 runs**. Coverage is purchased with substantial widening: mean normalized-field width **0.977526**, mean frequency width **761.501 Hz**. High coverage alone does not establish sharp uncertainty.

This review consolidates the current campaign's design, simulations, pairing, training CV, strict holdout, calibration, tail failures, figures and provenance. It was assembled from saved artifacts and row-level tables; it does not rerun or alter P1 physics, raw runs or trained models. It supersedes the older pilot document as a report of this campaign, but does not erase that document's historical limitations.

**Scope guard:** the execution is `pilot-five-variable`, with fixed materials and CCCC treatment. The full ten-variable proposal adds boundary stiffnesses and aerodynamic pressure; its simulation adapter is recorded as **not defined in the repository**. This campaign is not evidence for that extension, Abaqus agreement, experimental validation, flutter/aeroelastic performance, or a universal four-percent accuracy claim.

Primary sources: [result summary](p1_data_five/results_five/results_summary.json), [corrected strict holdout](p1_data_five/holdout_corrected/holdout_metrics.json), [claim assessment](p1_data_five/p1_claim_report.json), [final preflight](p1_data_five/final_preflight.json).

## 2. Frozen design and completed simulations

| Item | Completed / declared scope |
| --- | --- |
| Explicit-ID LF design | 10,000 designs, seed 42, legacy five-variable distribution |
| Raw LF bundles | 10,000 |
| Real COMSOL HF bundles | 80 |
| HF role split | 48 training / 16 calibration / 16 test runs |
| Raw HF configuration | 0.3-m plate; CCCC; Al 70 GPa, 2710 kg/m³, Poisson ratio 0.33 |
| HF numerical/extraction configuration | Automatic mesh 4; 32 candidate eigenmodes; 16 extracted flexural modes; 80×80 top-surface grid; `comsol-interp` |
| Correction labels requested | 10 stable LF-reference modes per run, 800 candidate rows |
| Accepted / quarantined correction rows | 721 / 79 |
| Accepted training / calibration / test rows | 431 / 140 / 150 |
| Final artifact gate | `pass` |
| Claim assessment | `ready_for_scientific_review`, not an accuracy guarantee |

The 80 HF runs were selected by **weighted maximin on LF frequency/features**, with roles fixed before HF simulation and no HF labels used for selection. Sampling weights below are the saved sensitivity-guided selection weights; they are not Sobol indices or causal importance estimates.

| Parameter | Meaning | Observed minimum | Observed maximum | HF sampling weight |
| --- | --- | --- | --- | --- |
| alpha | Core thickness ratio | 0.102848 | 0.899955 | 0.068511 |
| beta | Top-face / bottom-face thickness ratio | 0.100000 | 0.999842 | 0.238871 |
| theta_c | Honeycomb cell angle (degrees) | 0.591292 | 74.996389 | 0.272471 |
| eta1 | Cell edge ratio l2/l1 | 0.507442 | 2.999616 | 0.139207 |
| eta2 | Cell-wall ratio tc/l1 | 0.020046 | 0.149983 | 0.280940 |

Sources: [design manifest](p1_data_five/lhs_five.manifest.json), [HF plan](p1_data_five/hf_plan/hf_plan.json), [sensitivity table](p1_data_five/hf_plan/sensitivity.csv), [example raw HF provenance](p1_data_five/hf/run_9828.model.json).

## 3. Pairing, normalization and rejected targets

- Target revision: `p1-corrections-v5`, manifest schema 3. Common reference run: **9828**, a training run.
- Labels follow stable LF-reference identity, **not** each run's sorted-frequency position. The paired LF/HF mode indices can differ from the displayed stable label.
- Both transverse fields are normalized by their per-mode `max(abs(w))` and phase-aligned on a common grid. Field metrics therefore have **dimensionless normalized-modal-amplitude units**, not metres of displacement.
- Euclidean MAC acceptance floor: **0.8**. HF flexural extraction uses transverse energy fraction **≥0.5**. Raw peak displacement ratios are diagnostic only: eigenvector amplitudes are arbitrary and small absolute amplitudes alone do not prove invalid shapes.
- The boundary mask is applied to predictions; perimeter values are excluded from fitting loss and interior RMS.
- Near-repeated eigenvalues without a resolved individual mode identity are quarantined rather than assigned a misleading scalar-mode target.

| Quarantine reason | Rows |
| --- | ---: |
| Near-repeated eigenvalue requiring a subspace target | 68 |
| Low MAC only | 10 |
| Low MAC plus near-repeated eigenvalue | 1 |
| Total | 79 |

The rejection rate is **9.875%**. This is a restriction of the evaluated population: results apply to accepted modes, not the rejected near-degenerate modes.

| Accepted-pair diagnostic | Minimum | P05 | Median |
| --- | --- | --- | --- |
| tracking_mac | 0.840035 | 0.980345 | 0.996491 |
| pair_mac | 0.863959 | 0.974056 | 0.999249 |
| hf_transverse_fraction | 0.924675 | 0.956188 | 0.991497 |

Accepted rows by stable mode 1–10: **80, 74, 74, 79, 76, 70, 67, 70, 69, 62**. Test counts by mode are shown below. The manifest also records an 80-fold leave-one-run-out **LF identity-tracking diagnostic**; that diagnostic is not an 80-fold HF surrogate evaluation.

Sources: [target manifest](p1_data_five/corrections_five/manifest.json), [all pairing decisions](p1_data_five/corrections_five/pairing.csv).

## 4. Models and training-run cross-validation

The fitted primary model uses a **48-dimensional spatial latent representation**, boundary-enforced nonlinear low-rank decoding, ARD Gaussian-process latent corrections and a scalar frequency-correction path. Baselines are autoregressive co-kriging and a Fourier-feature INR.

Validation is **five-fold grouped by training run**. All accepted modes from one geometry stay in the same fold. Only the 48 training runs / 431 accepted rows enter this CV table. This is not the separate strict HF test.

| Model | Field RMS median | Field RMS P95 | Field RMS maximum | Frequency abs. error median (%) | Frequency abs. error P95 (%) |
| --- | --- | --- | --- | --- | --- |
| cokriging_baseline | 0.008460 | 0.046794 | 0.183078 | 1.7782 | 8.0842 |
| inr_baseline | 0.036033 | 0.096070 | 0.178700 | Not reported | Not reported |
| latent_gp | 0.008736 | 0.044242 | 0.182424 | 1.7782 | 8.0842 |

For comparison, the uncorrected FSDT reference on those same accepted training pairs has median/P95 field RMS **0.012181 / 0.077001** and median/P95 absolute frequency error **4.9446% / 14.7875%**.

Interpretation:

- Co-kriging has the slightly smaller median field error; latent GP has the smaller P95 field error. There is no across-the-board field superiority over co-kriging.
- The saved co-kriging and latent-GP frequency summaries are identical. These artifacts do not support a claim that the latent spatial representation independently improves scalar frequency prediction relative to that baseline.
- The INR field baseline is substantially worse than both GP approaches in this training-run CV. No strict-test INR or co-kriging evaluation is reported in the saved strict-holdout artifacts.
- Saved decoder reconstruction diagnostic: median interior RMS **0.000653916**, P95 **0.001512544**, explained variance **0.999999954**. This is a final training-stage reconstruction diagnostic, not an independent strict-test reconstruction result or a full surrogate generalization guarantee.
- `frequency_error_pct` in the original surrogate/holdout summary is **signed**. Its median must not be substituted for median absolute error. The export's absolute-error table is used for headline accuracy here.

Sources: [training metrics](p1_data_five/surrogate_five/metrics.json), [OOF row metrics](p1_data_five/surrogate_five/oof_metrics.csv), [CV absolute-error export](p1_data_five/results_five/cv_model_comparison.csv).

## 5. Strict HF test — primary accuracy result

The frozen test run IDs are **422, 1190, 1826, 2209, 2802, 3377, 3408, 4082, 4352, 4470, 5873, 6148, 7862, 8539, 9311, 9779**. The primary model is fitted on training runs only; calibration controls interval widths rather than refitting the mean.

Frequency error is `100*abs(f_pred-f_COMSOL)/f_COMSOL`. Field RMS is the error in the phase-aligned, max-normalized correction field over interior pixels; adding that correction to normalized FSDT gives the same error against normalized COMSOL. Relative correction RMS can become large when the true correction is nearly zero and is not used as the headline field metric.

| Model | Frequency abs. error median (%) | P95 (%) | Maximum (%) | Interior field RMS median | P95 | Maximum |
| --- | --- | --- | --- | --- | --- | --- |
| fsdt | 4.9218 | 14.9880 | 33.9785 | 0.013699 | 0.084288 | 0.166327 |
| latent_gp | 1.2558 | 6.6505 | 15.1700 | 0.007658 | 0.046186 | 0.154591 |

Relative reductions in the reported distribution statistics: **74.48%** in median frequency error, **55.63%** in P95 frequency error, **44.10%** in median field RMS, and **45.20%** in P95 field RMS. These are ratios of summary statistics, not median pairwise percentage improvements.

Frequency improves on **134/150 pairs (89.33%)**; field RMS improves on **125/150 (83.33%)**. Therefore 16 frequency pairs and 25 field pairs do not improve. Improvement is neither universal nor a worst-case error guarantee.

__omp_shell("[Strict test frequency parity](p1_data_five/results_five/test_frequency_parity.png)")

### 5.1 Results by stable mode

| Stable mode | Test pairs | FSDT freq. median (%) | Corrected median (%) | Corrected P95 (%) | FSDT field RMS median | Corrected median | Corrected P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 3.7539 | 1.0989 | 2.9311 | 0.003251 | 0.001912 | 0.008562 |
| 2 | 16 | 5.4708 | 1.3981 | 5.7061 | 0.007792 | 0.003253 | 0.015344 |
| 3 | 16 | 3.8368 | 0.7693 | 3.4899 | 0.004961 | 0.002744 | 0.015722 |
| 4 | 15 | 4.6252 | 1.3447 | 3.0568 | 0.008315 | 0.003186 | 0.017635 |
| 5 | 15 | 6.0320 | 0.4706 | 9.2406 | 0.017221 | 0.009040 | 0.042507 |
| 6 | 14 | 3.2745 | 1.5477 | 5.1232 | 0.013756 | 0.017193 | 0.046570 |
| 7 | 14 | 6.6033 | 1.5030 | 8.1726 | 0.027739 | 0.006136 | 0.070672 |
| 8 | 16 | 4.6908 | 0.8996 | 5.3725 | 0.030839 | 0.014541 | 0.065793 |
| 9 | 14 | 8.0944 | 2.2132 | 9.7049 | 0.034257 | 0.021996 | 0.085108 |
| 10 | 14 | 6.3366 | 1.6601 | 4.7632 | 0.019346 | 0.016610 | 0.081796 |

All ten modes have lower median absolute frequency error after correction. **Mode 6's median field error worsens** from 0.013756 to 0.017193. Modes 7–10 retain substantial field-error tails; mode 9 retains the largest frequency tail.

__omp_shell("[Strict test error distributions by mode](p1_data_five/results_five/test_error_by_mode.png)")

### 5.2 Worst accepted test cases

| Test run | Stable mode | Corrected absolute frequency error (%) |
| --- | --- | --- |
| 7862 | 9 | 15.1700 |
| 4470 | 5 | 13.9160 |
| 4470 | 7 | 11.9804 |
| 4352 | 8 | 8.3823 |
| 7862 | 5 | 7.2369 |

| Test run | Stable mode | Corrected interior field RMS |
| --- | --- | --- |
| 5873 | 8 | 0.154591 |
| 5873 | 9 | 0.154192 |
| 2209 | 7 | 0.138554 |
| 3408 | 10 | 0.132134 |
| 7862 | 10 | 0.054690 |

The published shape examples deliberately show both the median-error case (**run 4470, mode 8**) and the worst field-error case (**run 5873, mode 8**). Shape panels use normalized amplitudes; the spatial axes are metres. The worst-case residual is visible and must not be hidden by showing only favorable examples.

__omp_shell("[Strict test median and worst shape examples](p1_data_five/results_five/test_shape_examples.png)")

Sources: [all test rows](p1_data_five/results_five/test_metrics.csv), [all modewise medians/P95/maxima](p1_data_five/results_five/test_by_mode.csv).

## 6. Uncertainty and calibration — coverage and sharpness

Corrected intervals use **one normalized maximum-residual score per calibration run**, not per mode or pixel. There are 16 independent calibration runs. The finite-sample rule is `ceil((n+1)*coverage/100)`; at 90%, rank 16 selects the maximum calibration-run score.

Saved 90% multipliers: **5.2621384** for frequencies and **25.9230308** for fields. Run 2504 supplies the maximum frequency score; run 11 supplies the maximum field score. Calibrated mean predictions and point errors are unchanged.

| Metric at nominal 90% | Raw GP intervals | Corrected run-level intervals |
| --- | --- | --- |
| Field pointwise coverage | 93.0671% | 99.9744% |
| All accepted field pixels/modes in a run | 18.7500% | 93.7500% |
| Frequency mode-pair coverage | 85.3333% | 99.3333% |
| All accepted frequencies in a run | 56.2500% | 93.7500% |
| Mean field interval width | 0.062025 | 0.977526 |
| Mean frequency interval width (Hz) | 238.032 | 761.501 |

Raw GP pointwise field coverage appears close to nominal, but **only 3/16 runs** cover all accepted field pixels/modes simultaneously. Raw frequency coverage is **9/16 runs**. Corrected intervals reach **15/16 runs** separately for field and frequency; those two marginal run-level counts are not a reported joint event covering both channels together.

Mean field widths grow by **15.76×** and frequency widths by **3.20×**. The normalized-field intervals are wide relative to typical point errors, so the result supports conservative empirical coverage more strongly than sharp spatial uncertainty. Sixteen test runs are too few to establish precise population coverage from 15 successes.

**Finite calibrated 95% intervals are unsupported:** `ceil(17*0.95)=17`, exceeding 16 calibration runs. At least 19 independent calibration runs are required for a finite rank. Raw 95% GP interval diagnostics remain descriptive; they are not corrected 95% conformal guarantees.

**Use `holdout_corrected/`, not the old interval multipliers in `holdout/calibration.json`.** The earlier file reports smaller 95% than 90% multipliers and an unsupported finite 95% level. Its interval calibration is superseded. The trained model and point predictions were reused for the corrected run-level evaluation; no accuracy gain should be attributed to widening intervals.

__omp_shell("[Strict test coverage: pointwise versus simultaneous](p1_data_five/results_five/test_uncertainty_coverage.png)")

Sources: [corrected calibration](p1_data_five/holdout_corrected/calibration.json), [corrected metrics](p1_data_five/holdout_corrected/holdout_metrics.json), [result summary and widths](p1_data_five/results_five/results_summary.json).

## 7. Historical results — separate and explicitly limited

An older [P1 pilot results review](../P1_PART1_MVP_RESULTS.md) exists for a different 100-design, 10-mode correction corpus. It records a scalar frequency-error GP leave-run-out RMSE of **5.35 percentage points**, versus **9.45** for a train-mean baseline. That result is historical, not recomputed here, and is not pooled with the present 80-HF campaign.

The older document explicitly marks its spatial field scores—including approximately **4.19% field skill** and decoder/ARD/cross-modal probes—as **void pending valid HF re-export**, following an audit that archived correction fields were effectively negated normalized FSDT fields with no HF information. This conclusion relies on the provenance audit, not a universal absolute eigenvector-amplitude cutoff. Those historical spatial scores are not evidence for present P1 accuracy.

The [older R2 code review](../roadmaps/reviews/p1_review_r2.md) concerns the historical pipeline and is not a results review of `p1_data_five`. The current campaign uses real COMSOL interpolation, transverse-energy extraction, phase/MAC pairing and quarantined labels.

## 8. Complete figure and table index

Every listed figure has PNG and vector-PDF versions. Design/target diagnostics use their declared dataset scope; only names beginning `test_` below describe the strict HF test.

| Figure | Scope / interpretation | PNG | Vector PDF |
| --- | --- | --- | --- |
| design_space | Full five-variable design coverage | [PNG](p1_data_five/results_five/design_space.png) | [PDF](p1_data_five/results_five/design_space.pdf) |
| design_pairwise | Pairwise parameter coverage | [PNG](p1_data_five/results_five/design_pairwise.png) | [PDF](p1_data_five/results_five/design_pairwise.pdf) |
| frequency_parity | Accepted LF/COMSOL target frequency pairs; not model test predictions | [PNG](p1_data_five/results_five/frequency_parity.png) | [PDF](p1_data_five/results_five/frequency_parity.pdf) |
| frequency_error_by_mode | Accepted target LF/COMSOL discrepancies by stable mode | [PNG](p1_data_five/results_five/frequency_error_by_mode.png) | [PDF](p1_data_five/results_five/frequency_error_by_mode.pdf) |
| correction_rms_by_mode | Accepted target correction-field RMS by stable mode | [PNG](p1_data_five/results_five/correction_rms_by_mode.png) | [PDF](p1_data_five/results_five/correction_rms_by_mode.pdf) |
| matching_quality | Mode matching and transverse extraction quality | [PNG](p1_data_five/results_five/matching_quality.png) | [PDF](p1_data_five/results_five/matching_quality.pdf) |
| quarantine_reasons | Rejected-mode accounting | [PNG](p1_data_five/results_five/quarantine_reasons.png) | [PDF](p1_data_five/results_five/quarantine_reasons.pdf) |
| example_run0011_mode01 | Accepted target example; run 11 is calibration, not strict test | [PNG](p1_data_five/results_five/example_run0011_mode01.png) | [PDF](p1_data_five/results_five/example_run0011_mode01.pdf) |
| test_frequency_parity | Strict HF test: FSDT and corrected frequency parity | [PNG](p1_data_five/results_five/test_frequency_parity.png) | [PDF](p1_data_five/results_five/test_frequency_parity.pdf) |
| test_error_by_mode | Strict HF test errors by stable mode | [PNG](p1_data_five/results_five/test_error_by_mode.png) | [PDF](p1_data_five/results_five/test_error_by_mode.pdf) |
| test_shape_examples | Strict HF test: median and worst normalized field examples | [PNG](p1_data_five/results_five/test_shape_examples.png) | [PDF](p1_data_five/results_five/test_shape_examples.pdf) |
| test_uncertainty_coverage | Strict HF test: pointwise/mode-pair versus simultaneous run coverage | [PNG](p1_data_five/results_five/test_uncertainty_coverage.png) | [PDF](p1_data_five/results_five/test_uncertainty_coverage.pdf) |
| cv_model_comparison | Grouped training-run CV only | [PNG](p1_data_five/results_five/cv_model_comparison.png) | [PDF](p1_data_five/results_five/cv_model_comparison.pdf) |

Tables and numerical artifacts:

- [Design](p1_data_five/lhs_five.csv), [HF identities and roles](p1_data_five/hf_plan/hf_plan.csv), [LF sensitivity](p1_data_five/hf_plan/sensitivity.csv).
- [Pairing/target rows](p1_data_five/corrections_five/pairing.csv), [training target arrays](p1_data_five/corrections_five/training.npz).
- [OOF CV metrics](p1_data_five/surrogate_five/oof_metrics.csv), [CV predictions](p1_data_five/surrogate_five/oof_predictions.npz), [CV comparison](p1_data_five/results_five/cv_model_comparison.csv).
- [Strict holdout metrics](p1_data_five/holdout_corrected/holdout_metrics.csv), [strict predictions](p1_data_five/holdout_corrected/holdout_predictions.npz), [test comparison](p1_data_five/results_five/test_metrics.csv), [modewise results](p1_data_five/results_five/test_by_mode.csv).
- [Final deployable train-only model](p1_data_five/surrogate_five/latent_gp/), [strict-holdout train-only model](p1_data_five/holdout/latent_gp_train/), [corrected calibration](p1_data_five/holdout_corrected/calibration.json).

## 9. Provenance and reproduction

The four core recorded hashes—design, HF plan, target manifest and corrected holdout report—were checked against `p1_claim_report.json` while assembling this review and matched. The result summary records target-array SHA-256 **`818ec9601afe2a565d0ceac65b373ef947f275329cc66b4ec1566e712b75b151`** and corrected holdout SHA-256 **`714ec7be9da3b69e839b07d78ecb4ab19902401f7cb4781e80124e43f93635fc`**.

| Artifact | SHA-256 |
| --- | --- |
| [p1_data_five/lhs_five.csv](p1_data_five/lhs_five.csv) | `5dd4eb7f5e6bd75d86e7eb55be61f917a1befd6a57f501563909092a3c2fe10c` |
| [p1_data_five/hf_plan/hf_plan.csv](p1_data_five/hf_plan/hf_plan.csv) | `887eed56704e1740f324ded7adc14fff55590b7adad51dfec03f80df82c1b042` |
| [p1_data_five/corrections_five/manifest.json](p1_data_five/corrections_five/manifest.json) | `8d3c3de975a756f458c800fddac0a1a3e06897d5692b17e5f5fd77b22795318e` |
| [p1_data_five/holdout_corrected/holdout_metrics.json](p1_data_five/holdout_corrected/holdout_metrics.json) | `714ec7be9da3b69e839b07d78ecb4ab19902401f7cb4781e80124e43f93635fc` |
| [p1_data_five/results_five/results_summary.json](p1_data_five/results_five/results_summary.json) | `06844e0cb29839bdc9508f47a7db9431dfce4eb8d084125fdb8f7805db76ae79` |

The source honeycomb constitutive implementation is retained; the source paper's Table-1 G12 entry conflicts with its displayed equation, and the retained implementation follows the equation. No material/geometry calibration was altered to obtain these reported improvements.

To regenerate **derived cached-data tables and plots only**, run from `plate_main`:

```powershell
python plot_p1_results.py --data p1_data_five --name corrections_five --design p1_data_five/lhs_five.csv --holdout p1_data_five/holdout_corrected --surrogate p1_data_five/surrogate_five --out p1_data_five/results_five
```

That command does not rerun COMSOL, FSDT, target building or training, but it does replace derived files in the selected output directory. The execution and model-training commands are documented in [P1_DATA_README.md](P1_DATA_README.md). This review used the existing results rather than overwriting them.

## 10. Defensible claims and remaining limitations

**Supported:** the selected five-variable, accepted-mode study has lower held-out frequency and field error after correction; the frequency improvement is large in its median and P95; conservative 90% run-level interval calibration improved the observed coverage on 16 test runs; GP field baselines outperform the implemented INR in grouped training-run CV.

**Not supported:** a universal four-percent or simulator-independent error bound; calibrated finite 95% conformal intervals from 16 runs; correctness on quarantined near-degenerate modes; strict-test superiority over all baselines; the full ten-variable aerodynamic/boundary-stiffness proposal; experimental accuracy; extrapolation outside the design family; or quantified deployment speedup without an actual timing benchmark.

The chosen COMSOL mesh is the numerical reference for this surrogate study. This report does not independently establish mesh-converged physical truth. The later P2 mesh studies use a stricter inverse-observation noise budget and expose additional discretization requirements; they do not erase P1's measured correction against its configured HF reference.

The completed campaign is **ready for scientific review**, with tail failures, selection restrictions and wide uncertainty bands stated alongside the positive average results.
