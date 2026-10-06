# P1 results review — completed five-variable COMSOL/FSDT campaign

## Current final status — prospective P1, 2026-10-06

The completed seven-stage improvement study is the current bounded P1 result.
Read the [final report](P1_IMPROVEMENT_REPORT.md),
[published JSON/CSV manifest](p1_published_results/text_artifact_manifest.json),
[updated P1 proposal](proposals_updated(2).md#proposal-1-multi-fidelity-correction-field-via-latent-space-gaussian-process),
and [source/data availability contract](P1_REPRODUCIBILITY.md).
Sections1–11 below retain the earlier audited campaign; their historical
limitations are not a substitute for the later prospective measurements.

| 32 prospective IID geometries | Uncorrected LF | Frozen selected PCA48/RBF |
| --- | ---: | ---: |
| Geometry-median normalized-field RMS | 0.01476687 | 0.005104386 |
| Geometry-median frequency error % | 4.585452 | 0.9468742 |
| Accepted mode-row median field RMS | 0.01432819 | 0.00559255 |
| Accepted mode-row median frequency error % | 4.140782 | 1.012241 |
| Calibrated stable-label1 HF referrals | 24/32 | 4/32 |
| False acceptances / rejections | 0 / 0 | 0 / 0 |

The mode-row metrics use298 accepted rows; they are not geometry medians.
All80 prospective native runs succeed. No new global mean passes the
training-only promotion gate; the genuine neural INR is a tested comparator,
not the selected mean. Raw scales remain selected after failed residual-scale
promotion. Joint run coverage29/32 at90% and32/32 at95% comes with broad95%
mean field/frequency widths0.835769/2833.78Hz and weak discrimination.
Regional theta>=60 challenge coverage is diagnostic; the selected screening
policy has one challenge false rejection. Mass-weighted sampled spans support
modal mixing without closing individual-field or full-DOF convergence.

**Publication status:** P1 code, tests, Markdown and checksum-identified text
results are the authorized GitHub handoff. MPH/ZIP files, bulk arrays and binary
weights are not uploaded; no P1 bulk GitHub Release is created. The complete
local package/restoration proof remains valid, but public raw-data/model
availability remains intentionally restricted. Some historical figures and
raw-directory references below require that local package.

## 1. Historical audited conclusion and reporting scope

The completed `p1_data_five` campaign provides a positive **held-out five-variable structural correction result**. On **16 predeclared HF test runs and 150 accepted mode pairs**, latent-GP-corrected FSDT reduces median absolute frequency error from **4.9218% to 1.2558%** and median normalized interior field RMS from **0.013699 to 0.007658**. These are empirical errors against the configured COMSOL reference, not universal physical-error bounds.

Corrected 90% run-level intervals cover all accepted frequencies in **15/16 test runs (93.75%)** and all accepted field pixels/modes in **15/16 runs**. Coverage is purchased with substantial widening: mean normalized-field width **0.977526**, mean frequency width **761.501 Hz**. High coverage alone does not establish sharp uncertainty.

This review consolidates the campaign and the independent **2026-10-03 recheck**. The recheck validated all 10,000 LF and 80 HF bundles, rebuilt the target arrays and pairing decisions from raw runs, replayed saved models, and independently recomputed diagnostics. It did not change frozen physics, raw bundles, targets, role plans or primary trained models. New evidence is under [recheck_20261003/](p1_data_five/recheck_20261003/). The older pilot remains separate.

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
| LF numerical configuration | Legendre-Ritz order 15; 16 flexural modes; 80×80 grid; CCCC penalty springs 10¹² |
| Correction labels requested | 10 stable LF-reference modes per run, 800 candidate rows |
| Accepted / quarantined correction rows | 721 / 79 |
| Accepted training / calibration / test rows | 431 / 140 / 150 |
| Final artifact gate | `pass` |
| Claim assessment | `ready_for_scientific_review`, not an accuracy guarantee |

The 80 HF runs were selected by **LF-sensitivity-weighted maximin in parameter space**, with roles fixed before HF simulation and no HF labels used for selection. Sampling weights below are the saved sensitivity-guided selection weights; they are not Sobol indices or causal importance estimates.

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
- The boundary mask is applied to **correction** predictions; perimeter values are excluded from fitting loss and interior RMS. A zero boundary correction preserves the LF perimeter values. It does not make the finite-penalty LF displacement, rotations or full eigenproblem exactly clamped or physically consistent.
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

The fitted primary model uses a **48-dimensional linear, boundary-enforced PCA representation**, ARD Gaussian-process latent corrections and a separate scalar relative-frequency GP. There is no trained nonlinear autoencoder in the deployed model. Baselines are autoregressive pixel-output co-kriging and a coordinate-conditioned Fourier-feature **ridge proxy**, not a trained neural INR MLP.

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
- The Fourier-feature ridge proxy is worse than both GP approaches in training-run CV. This is not evidence against a properly trained neural INR. The independent strict-test comparison is in §5.3.
- Saved decoder training reconstruction RMS: median **0.000653916**, P95 **0.001512544**. The old `decoder_explained_variance` ratio mixed envelope-weighted singular-value energy with output-space residual variance. The code now computes `1 − sum(residual interior-field variances) / sum(target interior-field variances)`, giving **0.999255561** on the frozen training rows. An equal-output-energy rank-one example exposes the old 0.998403 result versus the correct **0.5**. Constant exactly representable targets report 1. Frozen metrics are retained as historical artifacts; this corrected training diagnostic is not a held-out generalization claim. See [variance correction evidence](p1_data_five/recheck_20261003/decoder_variance_correction.json) and §5.4.
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

Local-only figure: `p1_data_five/results_five/test_frequency_parity.png` (strict test frequency parity; not uploaded).

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

Local-only figure: `p1_data_five/results_five/test_error_by_mode.png` (strict test errors by mode; not uploaded).

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

Local-only figure: `p1_data_five/results_five/test_shape_examples.png` (median/worst normalized fields; not uploaded).

## 5.3 Fair strict-test baselines — reproducible settings and paired geometry differences

The recheck refits both implemented baselines on the **431 accepted rows from the 48 frozen training runs**, then evaluates the **same 150 accepted pairs from 16 frozen test runs** as the primary model. Calibration and test labels do not enter fitting. Both GP methods use an 80-iteration optimization cap and seed 42. The Fourier proxy uses 64 random features, ridge 0.0001 and a 120,000-point training cap. Its freshly fitted predictions agree with replay of the saved training-only coordinate-model checkpoint.

The prior throwaway-driver results in `baselines_strict/` did not retain enough configuration to reproduce their field numbers. They are historical, not the authoritative comparison. The following table supersedes them; the primary latent-GP/FSDT results remain unchanged.

| Strict-test model | Frequency abs. error median / P95 / max (%) | Interior field RMS median / P95 / max |
| --- | --- | --- |
| FSDT (uncorrected) | 4.9218 / 14.9880 / 33.9785 | 0.013699 / 0.084288 / 0.166327 |
| Latent GP | 1.2558 / 6.6505 / 15.1700 | 0.007658 / 0.046186 / 0.154591 |
| Co-kriging, matching GP optimization cap | 1.2558 / 6.6505 / 15.1700 | 0.007651 / 0.053231 / 0.152521 |
| Fourier-feature ridge coordinate proxy | no frequency path | 0.039940 / 0.108325 / 0.177459 |

The GP frequency predictions are **bit-identical**: both use the same scalar relative-frequency formulation. Frequency correction is not a benefit attributable to spatial compression.

For fields, define the paired gap as **latent minus co-kriging**; positive values favor co-kriging. The row-paired median is **+0.0001724**, mean **−0.0004863**, with latent winning **64/150 rows**. The median of the 16 geometry-paired medians is **+0.0000906**; its seeded geometry-bootstrap 95% interval is **[−0.0007203, +0.0003145]**. This interval spans zero. Latent wins **7/16 geometry-level field medians** against co-kriging.

The previous statement that a positive +0.000204 paired median favored latent GP had the sign backwards. A lower marginal median and a paired-median comparison are also different statistics; neither establishes overall superiority here.

| Run | Pairs | FSDT field | Latent field | Co-kriging field | Fourier proxy field | FSDT freq. (%) | Corrected freq. (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 422 | 8 | 0.00483 | 0.00662 | 0.00771 | 0.03194 | 7.421 | 1.388 |
| 1190 | 10 | 0.00441 | 0.00304 | 0.00533 | 0.02356 | 4.048 | 0.572 |
| 1826 | 9 | 0.01361 | 0.00450 | 0.00424 | 0.03142 | 2.274 | 1.825 |
| 2209 | 9 | 0.00544 | 0.00988 | 0.00684 | 0.03614 | 3.003 | 1.535 |
| 2802 | 9 | 0.02110 | 0.00577 | 0.00728 | 0.03775 | 4.042 | 1.345 |
| 3377 | 10 | 0.03742 | 0.01101 | 0.00930 | 0.05900 | 4.776 | 0.641 |
| 3408 | 10 | 0.00554 | 0.00674 | 0.00505 | 0.02735 | 7.476 | 1.177 |
| 4082 | 10 | 0.00542 | 0.00561 | 0.00501 | 0.03129 | 5.948 | 0.638 |
| 4352 | 10 | 0.05271 | 0.02234 | 0.01730 | 0.07104 | 9.890 | 3.543 |
| 4470 | 10 | 0.01239 | 0.01163 | 0.01031 | 0.03622 | 2.494 | 3.179 |
| 5873 | 10 | 0.04003 | 0.00922 | 0.01566 | 0.06084 | 6.458 | 1.250 |
| 6148 | 9 | 0.00497 | 0.00290 | 0.00280 | 0.02661 | 6.730 | 1.275 |
| 7862 | 10 | 0.06774 | 0.02986 | 0.01636 | 0.07374 | 12.118 | 2.326 |
| 8539 | 8 | 0.00516 | 0.00801 | 0.00807 | 0.03926 | 4.690 | 1.538 |
| 9311 | 8 | 0.03743 | 0.01069 | 0.01175 | 0.04245 | 10.422 | 0.945 |
| 9779 | 10 | 0.01317 | 0.00502 | 0.00539 | 0.04311 | 1.579 | 3.178 |

Latent GP improves on FSDT for **11/16 field medians** and **14/16 frequency medians**, not almost every geometry in both channels. Five geometry field medians worsen. Its smaller P95 versus co-kriging is a distribution-tail result, not a demonstrated architecture advantage. The defensible contribution remains the honeycomb-specific correction study and careful modal handling.

Sources: [reproducible comparison](p1_data_five/recheck_20261003/independent_verified/strict_comparison.json), [all four-model test rows](p1_data_five/recheck_20261003/independent_verified/strict_rows.csv), [geometry table](p1_data_five/recheck_20261003/independent_verified/strict_by_run.csv), [saved Fourier-model replay](p1_data_five/recheck_20261003/inr_checkpoint_replay.json).

## 5.4 Decoder generalization — train-fit encoder/decoder on held-out correction fields

A fresh `BoundaryLatentDecoder(48)` was fitted on **training rows only**, then encoded/reconstructed calibration and test HF-minus-LF fields. These are representation tests using held-out fields, not GP forecasts: no calibration or test rows enter decoder fitting. The independent recheck reproduces the earlier reconstruction distributions and retains [row-level reconstruction errors](p1_data_five/recheck_20261003/independent_verified/decoder_rows.csv).

| Split | Rows / runs | Reconstruction RMS median | P95 | Maximum |
| --- | --- | --- | --- | --- |
| Train (in-fit) | 431 / 48 | 0.000654 | 0.001513 | 0.005064 |
| Calibration (held out) | 140 / 16 | 0.000725 | 0.002534 | 0.064060 |
| Test (held out) | 150 / 16 | 0.000751 | 0.003363 | 0.078691 |

Held-out medians exceed the in-fit median by only ~15% (0.000751 versus 0.000654 on test), so the 48-dimensional boundary-enforced decoder generalizes to unseen correction fields at the median. The held-out tails are heavier (test P95 0.003363, maximum 0.078691): isolated modes reconstruct poorly, and those tails overlap the worst full-surrogate errors (e.g. runs 5873/2209/3408). Representation error is an order of magnitude below full-surrogate test error at the median (0.000751 versus 0.007658), so GP prediction — not decoder capacity — dominates typical test error, while decoder tails may contribute to the worst cases.

The previous optional nonlinear gate mixed all **80 runs**, including 16 calibration and 16 test runs, into its grouped CV. Its 0.001030 median-of-fold-medians and comparison against a differently split linear decoder were not a valid train-only promotion gate. `p1_decoder_upgrade.py` now requires the frozen `--plan`, selects only training runs, and compares both decoders on **identical train-run CV folds**. The repaired gate gives median-of-fold medians **0.002419 nonlinear versus 0.000755 linear**, with nonlinear winning **14/431 validation rows**. The nonlinear upgrade is not promoted. Its old artifacts are historical; the [training-only gate](p1_data_five/recheck_20261003/decoder_gate_train_only/reconstruction_metrics.json) supersedes them. This repair does not restore prospective blinding to the already examined historical diagnostic or change the frozen primary model.

## 6. Uncertainty and calibration — coverage, sharpness and design-region behavior

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

Corrected intervals cover each channel separately in 15/16 runs, but fail on different geometries: field coverage misses **2209**, frequency coverage misses **4352**. Joint coverage of all accepted field pixels/modes **and** frequencies is therefore **14/16 runs (87.5%)**, not 15/16. Raw joint coverage is 2/16.

### 6.1 Sharpness: predicted uncertainty tracks realized error on fields, weakly on frequencies

The legacy pooled field-standard-deviation quartiles show increasing median field error. They are descriptive summaries of dependent mode rows, not geometry-level validation (`p1_data_five/uncertainty_diagnostics/uncertainty_sharpness.json`). The old frequency quartiles mixed widths in Hz with errors in percent; use the matching-unit diagnostics below instead:

| Predicted field-std quartile | Median predicted std | Median realized field RMS |
| --- | --- | --- |
| Q1 (most confident) | 0.125641 | 0.001740 |
| Q2 | 0.204176 | 0.005034 |
| Q3 | 0.373974 | 0.014386 |
| Q4 (least confident) | 0.465448 | 0.015538 |

The pooled per-mode-row correlation is **0.707 for fields**. At the independent-geometry level it falls to **0.465**. Within-mode field correlations range from **−0.231 to +0.694**, with weak or negative associations in modes 5, 6, 8 and 9. Field uncertainty has partial discrimination, not a blanket ability to identify unreliable spatial predictions. The previous frequency correlation 0.227 compared widths in Hz against percentage errors; matching relative units gives **0.234 per row** and **0.332 per geometry**. Matching absolute Hz units gives 0.501 per row, a different diagnostic. These are descriptive statistics from 16 geometries, not 150 independent samples.

### 6.2 Actual design-region exclusion, not just a within-test slice

The earlier beta/theta quartile slice was post-hoc and excluded no region from fitting. The statement that a true exclusion diagnostic required new HF geometries was too strong: the existing plan supports a **retrospective** exclusion experiment.

The new protocol fixes **theta_c ≥60°** before this refit, removes all 14 training and 6 calibration runs in that region, and fits a separate model on **34 training runs**, with **10 outside-region calibration runs**. The original 16 test runs are still not used for fitting or calibration: 9 runs / 85 accepted pairs lie in the excluded region, 7 runs / 65 pairs are controls. The threshold was not searched against errors.

| Region-excluded model diagnostic | Control, theta_c <60° | Excluded region, theta_c ≥60° |
| --- | --- | --- |
| Test runs / accepted pairs | 7 / 65 | 9 / 85 |
| Field RMS median / P95 | 0.008190 / 0.042402 | 0.013228 / 0.073892 |
| Frequency abs. error median / P95 (%) | 1.7475 / 5.3966 | 1.9377 / 12.9611 |
| Median calibrated field std | 0.293516 | 0.334372 |
| Median calibrated frequency std (% of predicted frequency) | 3.2111 | 4.3745 |
| Mean calibrated field interval width | 1.002437 | 1.122435 |
| Mean calibrated frequency interval width (Hz) | 344.786 | 462.944 |
| Simultaneous field run coverage | 6/7 | 9/9 |
| Simultaneous frequency run coverage | 4/7 | **6/9** |

Widths increase with error in the deliberately excluded region, but the frequency channel covers only **66.7%** of its runs at nominal 90%; field coverage requires very wide bands. This is positive evidence for qualitative uncertainty growth, not successful calibrated extrapolation. Calibration and excluded-region test distributions differ, so ordinary split-conformal guarantees do not transfer automatically. The experiment reuses previously examined HF data: it is a genuine region-exclusion diagnostic, **not** a newly prospectively blinded transfer study.

Sources: [fixed protocol](p1_data_five/recheck_20261003/independent_verified/region_exclusion/protocol.json), [results and coverage](p1_data_five/recheck_20261003/independent_verified/region_exclusion/results.json), [geometry/mode rows](p1_data_five/recheck_20261003/independent_verified/region_exclusion/test_rows.csv).

### 6.3 HF mesh spot check using the actual accepted target identities

The initial recheck used the existing genuine COMSOL mesh-4/3/2 bundles, without retrying the known failed finest solve unchanged. Each refinement is MAC-assigned to campaign HF modes, globally phase-aligned (not merely cast to real and sign-flipped), and evaluated on the **10 stable accepted target modes** and the same interior mask used for surrogate errors. Its 50 available mode/step comparisons were unambiguous under the MAC/margin and near-degeneracy checks. All three mesh-4 recomputations match their campaign frequency/field arrays bit-identically. The table now also includes the successful, independently controlled run-3408 mesh-2 follow-up; its ten additional comparisons have no identity flags.

| Geometry | Mesh 3 vs 4: frequency median / max (%) | Mesh 3 vs 4: interior field RMS median / max | Mesh 2 vs 3: frequency median / max (%) | Mesh 2 vs 3: interior field RMS median / max |
| --- | --- | --- | --- | --- |
| Reference 9828 | 0.08648 / 0.16553 | 0.000424 / 0.000644 | 0.02598 / 0.06284 | 0.000257 / 0.000572 |
| Worst-field test 5873 | 0.06448 / 0.10295 | 0.000159 / 0.000328 | 0.00516 / 0.00803 | 0.000018 / 0.000027 |
| Extreme-angle test 3408 | 0.05223 / 0.08820 | 0.000316 / **0.022539** | 0.04968 / 0.05486 | 0.000097 / **0.004797** |

The new 3408 mesh-2 column compares the original mesh-3 solver with compressed MUMPS, after its tighter same-mesh control passed. The same-method compressed mesh 3→2 comparison below gives essentially the same drift.

These values supersede the old all-16-mode, sign-only/full-grid summary. Frequency changes are substantially below the aggregate corrected median 1.26%; median field changes are below 0.00766. This supports resolution of average correction effects **at these sampled geometries**, not a campaign-wide accuracy bound.

The initial three-geometry exception matters: **stable mode 10 of run 3408 maps to campaign HF mode 12**, with mesh-3/4 field change 0.022539. It cannot be described as a tightly converged individual shape. The original 3408 mesh-2 out-of-core assembly failure is not a physics result. The initial rounded ~0.2% frequency / ~0.023 field maxima were never campaign-wide bounds; the additional execution below exposes larger reference drift.

Sources: [MAC-matched original mesh report](p1_data_five/recheck_20261003/independent_verified/mesh_matched.json), [original matched target-mode comparisons](p1_data_five/recheck_20261003/independent_verified/mesh_matched_rows.csv), local-only original mesh bundles at `p1_data_five/mesh_convergence_check/`, and [successful 3408 fine-reference comparisons](p1_data_five/hf_followup_20261003/finest3408_results.json).

#### Additional HF execution, bounded halt and authorized resumption

Seven new full COMSOL solutions were saved: mesh 3 and 2 for runs **2209** and **4352**, mesh 1 for **2209**, and the controlled compressed-MUMPS mesh 3 and 2 for **3408**. These are six refinements plus one same-mesh numerical-method control. Geometry, material, CCCC treatment, 32 candidate/16 extracted modes and 80×80 extraction were unchanged. The first 2209 mesh-1 attempt failed on native out-of-core I/O; a fresh D-volume TEMP/TMP and Java temporary directory enabled a successful solution. This does not establish the exact cause of the original I/O failure.

| Accepted-mode comparison | Maximum frequency change (%) | Maximum normalized interior field RMS |
| --- | --- | --- |
| 4352, mesh 3→2 | 0.019729 | 0.000370 |
| 2209, mesh 3→2 | 0.056735 | 0.013378 |
| 2209, mesh 2→1 | 0.102036 | 0.017777 |
| 2209, nominal mesh 4→1 reference drift | 0.434329 | 0.088806 |
| 3408, compressed-MUMPS mesh 3→2 | 0.054862 | 0.004797 |
| 3408, nominal mesh 4→2 reference drift | 0.139857 | 0.027334 |

Run 4352's final step meets the predeclared diagnostic targets of 0.1% frequency and 0.00075 field RMS. Run 2209 does not; its shape tail is not converged even at the finest automatic mesh exposed by the current CLI. All nine accepted 2209 identities pass the MAC/margin/near-repeat checks. The unchanged surrogate against its mesh-1 reference has median/max field RMS **0.010033 / 0.227018** and median absolute frequency error **1.7140%**. Its field-run coverage still fails; frequency-run coverage still passes. These are targeted diagnostics, not replacement aggregate test headlines.

The saved run-2209 diagnosis fixes the unstable **accepted mode 7 → HF mode 8**
and chooses HF mode 9 by nearest frequency, not by best-fitting projection.
Their gap is **0.71–0.77%**, above the original 0.5% near-repeat cutoff.
Mesh-4/1 individual MAC is **0.951257**, but their extracted two-field span
has maximum principal angle **0.052352 degrees** and condition number 1.0025.
This supports close-mode numerical basis sensitivity; it does **not** establish
individual-shape convergence or convergence of the full-DOF/mass-weighted
eigenspace. No row, target, model or calibration was removed or changed.
Evidence: [spectral diagnosis](p1_data_five/hf_followup_20261003/mode7_spectral_diagnosis.json),
local-only visually checked fields at `p1_data_five/hf_followup_20261003/mode7_pair_refinement.png`
and the corresponding local-only `.pdf` (neither figure is uploaded).

The first alternate-scratch mesh-2 probe for 3408 was still running when the
user reported a **12-hour wait**. Its owned Python/COMSOL tree was stopped
without a saved NPZ. After explicit user authorization, a fresh probe was
launched with **4 cores, 8-GiB Java heap, fresh D-volume scratch, one attempt,
a 360-minute deadline and 60-second heartbeat**. Its geometry/mesh stage
completed in **194.9 seconds, with 785,895 elements**; the eigensolve/model-save
stage continued until the last heartbeat at **299.5 minutes**, but the wait
was interrupted without a saved reference or terminal supervisor status.
Neither process remains running, and its owned scratch was removed.
This is an interrupted attempt, not a measured timeout or numerical failure.
The original frozen campaign remains untouched.

Retained D-volume files identify **MUMPS out-of-core factorization**. A resource
snapshot showed 19.30-GiB peak working set and 24.53-GiB private commit on a
23.87-GiB-RAM workstation; these observations explain why a disk-backed
execution path matters, but do not prove a specific bottleneck or solver hang.
Both C (NVMe) and D (SATA) are SSDs.

A tiny native API comparison of MUMPS and **PARDISO** passes, with frequency
differences about 2×10⁻¹⁰ Hz. Its 0.012-m geometry and eight candidates/two
extracted modes are only an API/method smoke, **not** a run-3408 reference.
The full run-3408 **mesh-3 PARDISO control** timed out at its declared
60-minute cap (exit 124), without a saved reference; its worker stopped and
owned scratch was removed. Its native trace reports **5,857,248 DOFs** and
PARDISO's switch to out-of-core factorization. It was performing linear
system solves and matrix multiplications, not producing a validated reference.
No finer PARDISO result is accepted from this failed control.

The **MUMPS block-low-rank factorization** control uses tolerance 1e-8 with
the original error checks retained. Its full mesh-3 reference saved successfully
(exit 0; owned scratch removed). All ten original accepted identities pass:
maximum frequency change **0.000120%**, maximum field RMS **0.000003813**,
and zero identity flags, below the predeclared **0.01% / 0.000075-RMS** limits.
The eigensolve took **1255.0 seconds**; the native solver reports peak physical
memory 18.72 GB. This validates the measured solver change on this geometry
and mesh, not a general accuracy or memory-saving bound.
The authorized mesh-2 refinement also saved successfully (exit 0; owned
scratch removed), using the same solver settings, four cores, 8-GiB Java
heap, fresh D scratch and a 360-minute worker cap. It has **785,895 elements
and 10,865,214 DOFs**. All 32 candidate eigenpairs converged; all ten accepted
target identities have zero flags. The eigensolve took **11025.5 seconds**;
the complete worker ran about **188.1 minutes**. Its trace records out-of-core
factorization, 84 linear-system solutions and 252 matrix multiplications.
This accounts for the expensive stage observed in this successful run,
without asserting a specific bottleneck for earlier interrupted attempts.

The same-method compressed mesh 3→2 frequency maximum **0.054862%** meets
the 0.1% diagnostic target, but its field maximum **0.004797** exceeds
0.00075. The field exception is accepted **mode 10 → campaign HF mode 12**;
the other nine accepted shapes have final-step RMS at most 0.000153.
Nominal mesh 4→2 drift reaches **0.027334 RMS**. The unchanged surrogate
against mesh 2 has median/max field RMS **0.006665 / 0.159342** and median
frequency error **1.1626%**; both simultaneous run-level intervals cover.
These targeted results do not replace the frozen aggregate test headlines,
establish useful interval sharpness, or certify individual-shape convergence.
No owned follow-up solver remains running.
The physics builder, geometry, material, CCCC, 1000-Hz shift and 80×80
extraction remain unchanged; the solver-method override is recorded explicitly.
See [COMSOL's linear-solver guidance](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_solver.36.141.html)
and [native progress API](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_general.47.15.html).

Execution now records durable stage logs, kills the owned worker tree on
timeout/interruption, and does not retry timeouts. Parent-exit cleanup is
verified through both the batch and outer plan entrypoints. A real-process
smoke reproduced the old outer-wait orphan and confirmed that the plan and
driver now exit 130 and stop the worker plus descendant. This lifecycle smoke
does not claim a COMSOL solve. The licensed scratch/restart CLI smoke passes,
and the final P1/physics regression run passes **34 tests**.

Evidence: [fixed protocol](p1_data_five/hf_followup_20261003/protocol.json),
[first four refinements](p1_data_five/hf_followup_20261003/results.json),
[finest saved 2209 reference](p1_data_five/hf_followup_20261003/finest2209_results.json),
[finest-reference rows](p1_data_five/hf_followup_20261003/finest2209_rows.csv),
[original halt](p1_data_five/hf_followup_20261003/halt.json),
[authorized resumption](p1_data_five/hf_followup_20261003/resume3408_protocol.json),
[supervisor log](p1_data_five/hf_followup_20261003/resume3408_supervisor.log),
[interrupted resumption outcome](p1_data_five/hf_followup_20261003/resume3408_outcome.json),
[native method smoke](p1_data_five/hf_followup_20261003/solver_probe_results.json),
[fixed solver-control protocol](p1_data_five/hf_followup_20261003/pardiso_control_protocol.json),
[control supervisor log](p1_data_five/hf_followup_20261003/pardiso3408_control_supervisor.log),
[startup-attempt record](p1_data_five/hf_followup_20261003/pardiso_startup_attempts.json),
[PARDISO control timeout](p1_data_five/hf_followup_20261003/pardiso3408_control_mesh3/pardiso_execution.json),
[compressed-method smoke](p1_data_five/hf_followup_20261003/mumps_blr_probe_results.json),
[compressed-control protocol](p1_data_five/hf_followup_20261003/mumps_blr_control_protocol.json),
[compressed-control supervisor](p1_data_five/hf_followup_20261003/mumps_blr3408_control_supervisor.log),
[compressed-control results](p1_data_five/hf_followup_20261003/mumps_blr_control_results.json),
[fine compressed-solver protocol](p1_data_five/hf_followup_20261003/mumps_blr_refinement_protocol.json),
[fine compressed-solver supervisor](p1_data_five/hf_followup_20261003/mumps_blr3408_refinement_supervisor.log),
[fine compressed-solver outcome](p1_data_five/hf_followup_20261003/resume3408_mesh2/solver_execution.json),
[fine 3408 analysis](p1_data_five/hf_followup_20261003/finest3408_results.json),
[fine 3408 accepted-mode rows](p1_data_five/hf_followup_20261003/finest3408_rows.csv),
[final cleanup/provenance verification](p1_data_five/hf_followup_20261003/final_verification.json),
[outer-plan lifecycle smoke](p1_data_five/hf_followup_20261003/plan_parent_watch_smoke.json),
and [34-test output](p1_data_five/hf_followup_20261003/supervisor_regressions.txt).
Saved-result reproduction: `python p1_data_five/hf_followup_20261003/analyze.py`,
`python p1_data_five/hf_followup_20261003/analyze_finest.py --run 2209`,
`python p1_data_five/hf_followup_20261003/analyze_finest.py --run 3408`, and
`python p1_data_five/hf_followup_20261003/analyze_mode7.py`; these regenerate
derived outputs without starting COMSOL or refitting a model.


**Finite calibrated 95% intervals are unsupported:** 16 calibration runs cannot supply the required finite rank; at least 19 are needed. Use `holdout_corrected/`, not the superseded interval calibration in `holdout/`. The model combines GP predictive variance, a fitted noise term and training decoder residual variance; it is not a demonstrated pure design-space epistemic posterior, and omitted cross-latent covariance is an additional modeling approximation. Deterministic sensitivity-weighted maximin role selection does not establish the exchangeability assumed by standard conformal guarantees. Report the observed coverage, not unconditional reliability.

Local-only figure: `p1_data_five/results_five/test_uncertainty_coverage.png` (pointwise versus simultaneous coverage; not uploaded).

Sources: [recomputed calibration and model integrity](p1_data_five/recheck_20261003/independent_final/integrity.json), [geometry and within-mode uncertainty diagnostics](p1_data_five/recheck_20261003/independent_final/uncertainty.json).

## 7. Historical results — separate and explicitly limited

An older [P1 pilot results review](../P1_PART1_MVP_RESULTS.md) exists for a different 100-design, 10-mode correction corpus. It records a scalar frequency-error GP leave-run-out RMSE of **5.35 percentage points**, versus **9.45** for a train-mean baseline. That result is historical, not recomputed here, and is not pooled with the present 80-HF campaign.

The older document explicitly marks its spatial field scores—including approximately **4.19% field skill** and decoder/ARD/cross-modal probes—as **void pending valid HF re-export**, following an audit that archived correction fields were effectively negated normalized FSDT fields with no HF information. This conclusion relies on the provenance audit, not a universal absolute eigenvector-amplitude cutoff. Those historical spatial scores are not evidence for present P1 accuracy.

The [older R2 code review](../roadmaps/reviews/p1_review_r2.md) concerns the historical pipeline and is not a results review of `p1_data_five`. The current campaign uses real COMSOL interpolation, transverse-energy extraction, phase/MAC pairing and quarantined labels.

## 8. Complete figure and table index

Every listed figure has PNG and vector-PDF versions. Design/target diagnostics use their declared dataset scope; only names beginning `test_` below describe the strict HF test.

| Figure | Scope / interpretation | Local PNG (not uploaded) | Local PDF (not uploaded) |
| --- | --- | --- | --- |
| design_space | Full five-variable design coverage | `design_space.png` | `design_space.pdf` |
| design_pairwise | Pairwise parameter coverage | `design_pairwise.png` | `design_pairwise.pdf` |
| frequency_parity | Accepted LF/COMSOL target frequency pairs; not model test predictions | `frequency_parity.png` | `frequency_parity.pdf` |
| frequency_error_by_mode | Accepted target LF/COMSOL discrepancies by stable mode | `frequency_error_by_mode.png` | `frequency_error_by_mode.pdf` |
| correction_rms_by_mode | Accepted target correction-field RMS by stable mode | `correction_rms_by_mode.png` | `correction_rms_by_mode.pdf` |
| matching_quality | Mode matching and transverse extraction quality | `matching_quality.png` | `matching_quality.pdf` |
| quarantine_reasons | Rejected-mode accounting | `quarantine_reasons.png` | `quarantine_reasons.pdf` |
| example_run0011_mode01 | Accepted target example; run11 is calibration, not strict test | `example_run0011_mode01.png` | `example_run0011_mode01.pdf` |
| test_frequency_parity | Strict HF test: FSDT and corrected frequency parity | `test_frequency_parity.png` | `test_frequency_parity.pdf` |
| test_error_by_mode | Strict HF test errors by stable mode | `test_error_by_mode.png` | `test_error_by_mode.pdf` |
| test_shape_examples | Strict HF test: median and worst normalized field examples | `test_shape_examples.png` | `test_shape_examples.pdf` |
| test_uncertainty_coverage | Strict HF test: pointwise/mode-pair versus simultaneous run coverage | `test_uncertainty_coverage.png` | `test_uncertainty_coverage.pdf` |
| cv_model_comparison | Grouped training-run CV only | `cv_model_comparison.png` | `cv_model_comparison.pdf` |

Tables and numerical artifacts:

- [Raw-bundle preflight](p1_data_five/recheck_20261003/preflight.json), [raw-derived target and model integrity](p1_data_five/recheck_20261003/independent_verified/integrity.json), [CSV, saved-decoder and training-CV integrity](p1_data_five/recheck_20261003/auxiliary_integrity.json).
- [Strict baseline comparison](p1_data_five/recheck_20261003/independent_verified/strict_comparison.json), [all four-model test rows](p1_data_five/recheck_20261003/independent_verified/strict_rows.csv), [geometry medians](p1_data_five/recheck_20261003/independent_verified/strict_by_run.csv).
- [Independent decoder generalization](p1_data_five/recheck_20261003/independent_verified/decoder_generalization.json), [training-only nonlinear gate](p1_data_five/recheck_20261003/decoder_gate_train_only/reconstruction_metrics.json), [corrected variance diagnostic](p1_data_five/recheck_20261003/decoder_variance_correction.json).
- [Uncertainty diagnostics](p1_data_five/recheck_20261003/independent_verified/uncertainty.json), [fixed region-exclusion results](p1_data_five/recheck_20261003/independent_verified/region_exclusion/results.json).
- [Accepted-mode mesh comparisons](p1_data_five/recheck_20261003/independent_verified/mesh_matched.json), [reproduction evidence](p1_data_five/recheck_20261003/reproduction.json), [live LF replay](p1_data_five/recheck_20261003/lf_smoke.json), [tiny licensed COMSOL runtime smoke](p1_data_five/recheck_20261003/hf_smoke_output.txt).
- [Current-source audit runtime output](p1_data_five/recheck_20261003/audit_output.txt), [P1/physics regression output](p1_data_five/recheck_20261003/regression_output.txt), [recomputed artifact-ready guard](p1_data_five/recheck_20261003/claim_guard_recomputed.json).

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

The working-tree exporter was re-executed to a fresh output directory. `results_summary.json`, `test_metrics.csv`, `test_by_mode.csv` and `cv_model_comparison.csv` are **byte-identical** to the frozen exports; it writes 13 PNG and 13 PDF figures. The strict shape figure was visually inspected. The 10,000-row design generator, LF-only HF-plan generator and sensitivity CSV reproduce byte-identically. Native LF replay on run 9828 reproduces frequencies and phase-aligned fields exactly; a tiny licensed COMSOL build/solve/interpolation smoke passes. These runtime checks are not a rerun of the full 80-HF campaign. The current-source independent audit exits successfully with the corrected variance code; frozen core hashes remain unchanged.

The final P1/physics regression run passes **34 tests** across `test_p1_full_workflow`, `test_p1_pairing`, `test_p1_surrogate` and `test_regressions`, including held-out-label isolation of the decoder gate, the analytical equal-energy variance case and process-tree deadline enforcement. All local Markdown links resolve; the saved-data audit's computational source hashes match the current implementations.

```powershell
python plot_p1_results.py `
  --data p1_data_five `
  --name corrections_five `
  --design p1_data_five/lhs_five.csv `
  --holdout p1_data_five/holdout_corrected `
  --surrogate p1_data_five/surrogate_five `
  --out p1_data_five/recheck_export_new

python p1_recheck.py `
  --data p1_data_five `
  --output p1_data_five/recheck_new

python p1_decoder_upgrade.py `
  --data p1_data_five `
  --name corrections_five `
  --plan p1_data_five/hf_plan/hf_plan.csv `
  --output p1_data_five/decoder_gate_train_new
```

`p1_recheck.py` retains the previously missing diagnostic driver: raw target rebuilding, model-weight replay, seeded strict baselines, independent decoder reconstruction, run/within-mode uncertainty, actual region exclusion and MAC-matched mesh comparisons. It refuses an existing output directory. It reads archived mesh bundles rather than claiming to generate new converged COMSOL references.

**Historical publication state (2026-10-03):** the campaign implementation and data were not then in the pushed revision, and no commit/push/data upload had been performed for that audit. The current source/text handoff closes the missing-code/report gap but, by explicit restriction, uploads no ZIP/MPH or fitted binary data. Complete public raw-data/model availability therefore remains open. See [P1_REPRODUCIBILITY.md](P1_REPRODUCIBILITY.md); the local immutable package and restoration proof are distinct from public downloads.

## 10. Historical claim boundaries — 2026-10-03

**Supported:** lower median/P95 held-out frequency and field error on the specified accepted-mode population; exact reproduction of raw-derived targets, frozen primary predictions and numerical exports; typical held-out decoder reconstruction far below full-surrogate error; conservative observed 90% marginal run coverage; partial field error/width discrimination; qualitative uncertainty growth in a genuinely excluded but retrospectively chosen design region; sampled HF mesh sensitivities below average correction errors except the stated field tail.

**Not supported:** a universal four-percent or simulator-independent bound; finite calibrated 95% intervals from 16 runs; correctness on 79 quarantined modes; latent superiority over co-kriging; superiority over a trained neural INR; uniformly useful epistemic uncertainty; nominal 90% frequency coverage in the excluded region; prospectively blinded transfer; a campaign-wide mesh-error bound; downstream structural design benefit; HF sample-efficiency superiority; causal attribution of the discrepancy; quantified deployment speedup; or the full aerodynamic/boundary-stiffness proposal.

The results remain suitable for **scientific review of a bounded honeycomb correction experiment**. The automated `ready_for_scientific_review` gate checks artifacts/counts/contracts; it does **not** certify completion of Astra's scientific criteria. My earlier statement that all five Astra items were closed was incorrect.

## 11. Historical Astra disposition — 2026-10-03

This table preserves the earlier audit. The current [claim-by-claim report](P1_IMPROVEMENT_REPORT.md)
adds genuine neural/budget comparisons, prospective evaluation, sampled shell-state
diagnosis and engineering verification without claiming that unsuccessful gates
or restricted bulk-data availability have become successes.

| Astra requirement | Recheck disposition | Remaining limitation / action |
| --- | --- | --- |
| 1. Matching code, IDs, pairing/quarantine records, models, predictions and working reproduction | **Local evidence authenticated**; all 10,000 LF / 80 HF bundles pass, targets/pairing/721 CSVs rebuild identically, saved models/calibration replay exactly, export and plan reproduce | **Public handoff still open**: matching source and data/model release are absent from the tracked/published package |
| 2. Identical frozen-test FSDT / latent GP / co-kriging / INR comparison, tails and geometry-paired differences | **Completed for implemented models**, with retained settings and rows | No latent advantage; coordinate baseline is a Fourier ridge proxy, not the proposal's neural MLP INR |
| 3. Independent train-only decoder generalization | **Completed**, with row-level held-out reconstruction; nonlinear gate leakage fixed | Typical error is small; held-out reconstruction tails reach 0.078691; nonlinear architecture not justified |
| 4. Coverage plus useful width/error behavior, deliberately held-out regions, independent calibration | **Partial**: run/mode diagnostics and genuine retrospective region exclusion now exist | Wide fields, weak/mixed discrimination, excluded-region frequency coverage **6/9**, no newly prospectively blinded transfer or distribution-shift guarantee |
| 5. HF reference accuracy and numerical tolerances on representative/extreme geometries | **Partial numerical evidence over five geometries**, using matched accepted identities; interrupted 3408 computation now completed | 2209 mesh-2/1 field tail 0.017777; 3408 validated same-method mesh-3/2 field tail 0.004797; neither meets the 0.00075 individual-field target, and no campaign-wide convergence certificate exists |

### Additional positioning and improvement recommendations

- **Scope:** the five-variable, fixed-material, fixed-CCCC COMSOL study is a bounded deliverable. Missing optional ten-variable/aerodynamic/Abaqus extensions are not an automatic failure of this narrower experiment.
- **Novelty:** existing [multi-level multi-response GP mode-shape work](https://arxiv.org/abs/2002.09287) and [autoencoder-plus-GP solid-mechanics work](https://arxiv.org/abs/2407.10732) confirm Astra's prior-art objection. The application, explicit-cell reference and modal audit are the contribution; a new architecture claim is unsupported.
- **HF sample efficiency:** 48 training geometries alone do not prove fewer HF calls than alternatives at matched accuracy. No multi-budget comparison against a direct-HF surrogate is present. The 10,000 LF runs were used for selection; the correction GP is not trained on 10,000 independent HF observations.
- **Transfer:** the new exclusion diagnostic is useful but retrospective and inside the existing design bounds. It does not establish new-family or prospectively blinded generalization.
- **Downstream benefit:** no HF-confirmed change in a structural design decision, reliability constraint or deployment accuracy/cost tradeoff has been demonstrated. Average-error improvement is not that demonstration.
- **Physical explanation:** the LF implementation already uses laminate-dependent shear correction, not constant 5/6. Homogenization, finite-cell/boundary effects, layer asymmetry and shear modeling may contribute, but this campaign contains no controlled ablations isolating their causal effects. Do not attribute the full discrepancy to a single missing shear factor.
- **FSDT surrogate / direct-HF alternative:** keep the actual FSDT solver as the reference unless timing identifies it as a bottleneck. No same-budget `FSDT + correction` versus `FSDT surrogate + correction` versus direct-HF comparison or deployment timing exists; approximating FSDT would introduce another error source, not automatic novelty.
- **Physics/PINN constraints:** zero transverse correction at the perimeter is already enforced. Full joint frequency/shape eigen-consistency or admissible stiffness/mass correction would require compatible full modal DOFs and operator data, not just normalized transverse fields. Forcing HF-corrected fields to satisfy the uncorrected FSDT equations could suppress the discrepancy being learned. No unjustified PINN or extra network was added.


## 12. Separately frozen P1 improvement study

The historical claims above describe the audited 10,000-LF/80-HF campaign.
The new study is isolated in unpacked `plate_main\p1_improvement_20261005`
folders, relocated from D without replacing historical models, labels,
nominal predictions or claims. The [data README](P1_DATA_README.md#p1-improvement-study-and-prospective-release)
documents the executable stage order and current unpacked layout.
The final study-root `P1_IMPROVEMENT_REPORT.md` / `improvement_report.json`
derive their claim dispositions from actual prospective results.

Completed numerical diagnosis: five solved MPH references were loaded and
their native displacement/director states sampled on both faces and all
physical core walls. Physical mass weighting includes shell thickness,
offset coupling and director inertia; common-geometry mass agrees with
native integration. Quadrature angle changes pass the predeclared
0.05-degree check. Fine-quadrature maximum principal angles are
0.033625 degrees for 2209 mesh2/1 and 0.043680 degrees for 3408 compressed
mesh3/2. These stable two-mode spans support basis-mixing sensitivity;
the individual-field 0.00075-RMS failures remain. This is not full-DOF
eigenvector convergence or a campaign-wide reference-error bound.

Train-only decomposition confirms a much smaller typical representation
error than full prediction error. A physical-output PCA oracle improves
reconstruction, but none of eight new global configurations passes the
predeclared full-prediction promotion gate. Retain the original PCA48/RBF.
Regional co-kriging/RBF passes the same gate when all fitting and tuning
exclude theta>=60; its training-CV result does not establish OOD transfer.

Matched-budget comparisons completed all **90** trials across five tuned
families plus the fixed original baseline, budgets8/16/24/32/48 and three
nested repeats. This includes a genuine trained coordinate-conditioned
PyTorch INR with actual saved network weights, not the historical Fourier
ridge proxy. Each tuned family receives two configurations and identical
train-only geometry partitions within its counted HF budget. Historical
test curves are not final-model-selection evidence. Repeat extrema are not
confidence intervals, and no universal HF-efficiency advantage is claimed.

Local residual scaling fails both training-only UQ promotion gates.
Global raw/local joint-90 field interval scores are 0.780991/1.698698;
frequency scores are 2678.748/10723.235 Hz, with joint coverage47/48 and
46/48. Regional raw/local field scores are 0.647876/1.168003 and frequency
scores2210.168/5023.409 Hz; coverage is26/34 and30/34. Improved regional
coverage does not offset the width/score degradation under the fixed gate.
Raw scales remain selected. No claim of uniformly useful epistemic
uncertainty follows from these diagnostics.

Final fitted models, recipes, reference gauge, source hashes and engineering
threshold were frozen before the 32 calibration / 32 IID test / 16 challenge
geometries were generated. All 800 real LF-only mode inputs were predicted
and archived before new HF acquisition. Serialized models replay exactly
in the freeze smoke, including the actual INR's arbitrary-coordinate
prediction and transverse zero-boundary correction. The new no-MPH native
acquisition path passes its historical run9828 control. Independent
calibration changes widths only; engineering accept/reject/HF decisions
must be archived before evaluation solves. Prospective accuracy, useful-UQ,
shift and engineering claims require the measured final report, not these
completed setup steps.

### Prospective final results and engineering verification

All **80/80** new native HF acquisitions completed successfully. The fixed
pairing/quality rules accept295/320 calibration and448/480 evaluation rows;
all80 geometries contribute accepted modes. Six blinded model predictions
on all800 real LF-only inputs replay exactly using the original inference
batch layout. No weights, mean configuration or scale recipe was selected
from evaluation labels.

| IID model, 32 new geometries | Geometry-median field RMS | Geometry-median frequency error % |
| --- | ---: | ---: |
| Uncorrected LF | 0.01476687 | 4.585452 |
| Frozen selected PCA48/RBF | 0.005104386 | 0.9468742 |
| Direct-HF/Matern32 | 0.005090374 | 2.129665 |
| Output-metric PCA/Matern32 | 0.004310370 | 0.8275916 |
| Genuine neural INR | 0.01320841 | 1.519260 |
| Regional co-kriging/RBF | 0.005577847 | 1.133173 |

The selected mean's geometry-paired LF-minus-corrected median differences
are0.0102821 field RMS (descriptive bootstrap95% interval
[0.00579709,0.0198042]) and3.74687 percentage points in frequency
([2.05368,4.97993]). These are geometry-resampled diagnostics, not
simultaneous multiple-comparison inference or grounds for post-test
promotion of the output-metric candidate.

IID selected-mean joint coverage is29/32 at90% and32/32 at95%.
The95% joint mean field width is0.835769 and frequency width2833.78Hz.
Raw field/frequency Spearman correlations are0.2364/0.1870. Local residual
scaling increases those correlations to0.3420/0.3618 but worsens95% proper
field/frequency scores from0.835769/2833.78Hz to2.392526/17416.94Hz.
Better referral ranking alone does not establish useful sharp intervals.
The excluded regional model reaches12/16 challenge joint coverage at90%,
versus30/32 IID; no OOD calibration guarantee is asserted.

The fixed961.333328-Hz stable-mode1 task uses independently calibrated95%
frequency intervals. Selected-policy IID outcomes: **4/32 HF referrals,
zero false acceptances, zero false rejections, 28 verified correct automatic
decisions**. Calibrated LF-only screening requires24/32 referrals.
Measured IID reference HF time is6839.24s; selected referrals would require
1075.65s. LF solve/save log times sum to16.0s for these32 cases, rounded
to0.1s per run, excluding process startup. Selected full-field inference
for all80 prospective cases took0.1971s.

The selected policy's16 challenge cases have4 referrals, zero false
acceptances and **one false rejection**;11 automatic decisions are correct.
Zero observed false acceptances is not zero risk: the IID two-sided95%
Clopper–Pearson interval across19 verified automatic accepts is
[0,0.17647]. The campaign paid all80 HF references, and the per-policy
training/calibration setup cost remains explicit. No net campaign-cost
saving, complete-spectrum certificate or external service-safety claim
is made.

Reproduction retains one failed40-row neural replay: float32 batch-layout
changes produced differences up to1.05e-8. Replaying the original800-row
layout gives zero differences for all six models without changing weights
or loosening tolerance. Native API probe failures and earlier solver
interruptions remain in the authenticated baseline/study artifacts.
The43-test workflow run and the additional fitted direct-HF/LF-invariance
test pass. The local release includes actual baseline and improvement
ZIPs, source revisions, raw labels, quarantine audits, fitted weights,
blinded predictions, independent calibration, decisions, runtime settings,
figures, negative results and streaming checksums. The current GitHub handoff
publishes reviewed P1 source and text evidence only; the ZIP/MPH files and fitted
binary inputs remain local, and no P1 bulk release upload is authorized.

The unpacked relocation independently verified 1,640 moved files, all 11,211
original baseline members and the retained original source revisions.
After the D-drive working folders were removed, the six-model 800-row replay
again produced zero differences without ZIP extraction, new solves or refits.
See [relocation verification](p1_published_results/unpacked_relocation_verification.json)
and [local reproduction](P1_REPRODUCIBILITY.md#local-unpacked-study--no-zip-required).
The two unchanged ZIPs left in D are optional backups, not runtime dependencies.

