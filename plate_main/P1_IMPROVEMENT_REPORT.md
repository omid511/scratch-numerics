# P1 improvement study — claim-by-claim results

The audited baseline and its nominal predictions remain unchanged. This GitHub report publishes the measured P1 conclusions; [JSON/CSV evidence](p1_published_results/text_artifact_manifest.json), [reproduction and availability](P1_REPRODUCIBILITY.md), and the [updated proposal](proposals_updated(2).md#proposal-1-multi-fidelity-correction-field-via-latent-space-gaussian-process) are repository-relative. The original study-root report is preserved in the local immutable archive. No MPH or ZIP files are uploaded.

## Claim disposition

| Stage | Claim | Outcome |
| --- | --- | --- |
| 1 | Numerical basis mixing contributes to the individual-field sensitivity | Supported for the measured two-mode shell-state spans |
| 2 | A targeted global model improves the predeclared training-only gate | Not supported; retain original PCA48/RBF |
| 3 | Matched HF-budget comparisons include a genuine neural INR | Completed with actual fitted weights and predictions |
| 4 | Local residual scaling improves useful uncertainty | Rejected by both predeclared training-only promotion gates |
| 5 | Models were frozen before independent prospective geometry and label generation | Authenticated ordering |
| 6 | Frequency screening provides HF-verified operational benefit | Observed criterion met |
| 7 | A matched local source/data/model release is reproducible | LF-only fitted-model/source/data replay authenticated |

## Numerical reference interpretation

All prescribed native state extractions completed. The state includes displacement and dimensionless shell-director displacement on both faces and every clipped core wall, with thickness, offsets, coupling and rotary inertia. Native mass integrals validate the common geometry. Principal angles measure a two-mode span; no individual shape or complete spectrum is declared converged.

| Run / mesh pair | Fine quadrature maximum angle (degrees) | Angle change across quadratures | Minimum individual MAC |
| --- | ---: | ---: | ---: |
| 2209 / mesh2_vs_mesh1 | 0.033625325 | 0.0058951015 | 0.99787638 |
| 2209 / mesh3_vs_mesh1 | 0.036896958 | 0.035000047 | 0.99355242 |
| 2209 / mesh3_vs_mesh2 | 0.047826018 | 0.03630688 | 0.99882423 |
| 3408 / mesh3_compressed_vs_mesh2_compressed | 0.043679573 | 0.0041524831 | 0.99987345 |

## Matched HF budgets

Identical nested geometry subsets; representation fitted within each counted budget. Two train-only validation partitions and two configurations per tuned family. The fixed original RBF is also retained. These are historical-test comparisons, not prospective model-selection evidence.

| Family / HF budget | Completed trials | Geometry-median field RMS, median over repeats | Geometry-median frequency error %, median over repeats |
| --- | ---: | ---: | ---: |
| boundary_kernel_pca/HF8 | 3/3 | 0.01417222756790536 | 2.4524744740320736 |
| boundary_kernel_pca/HF16 | 3/3 | 0.012679544547527635 | 3.0342000867634944 |
| boundary_kernel_pca/HF24 | 3/3 | 0.00821508802404335 | 1.9995239241616214 |
| boundary_kernel_pca/HF32 | 3/3 | 0.007484204243291466 | 1.6073707647055082 |
| boundary_kernel_pca/HF48 | 3/3 | 0.007375264631712135 | 1.4713981326953585 |
| cokriging/HF8 | 3/3 | 0.010937001871774257 | 2.182017038305957 |
| cokriging/HF16 | 3/3 | 0.010654112317779962 | 2.2841939871855574 |
| cokriging/HF24 | 3/3 | 0.008362111262598886 | 1.9995239241616214 |
| cokriging/HF32 | 3/3 | 0.005675811265948315 | 1.58209457256571 |
| cokriging/HF48 | 3/3 | 0.007239842483981009 | 1.4713981326953585 |
| direct_hf/HF8 | 3/3 | 0.018096621467329967 | 8.540961151879362 |
| direct_hf/HF16 | 3/3 | 0.013916097122487066 | 5.7893054209866115 |
| direct_hf/HF24 | 3/3 | 0.0113799498175567 | 5.53671418236323 |
| direct_hf/HF32 | 3/3 | 0.010230264980200632 | 4.408336577441929 |
| direct_hf/HF48 | 3/3 | 0.0077167506874865075 | 4.462952320084602 |
| neural_inr/HF8 | 3/3 | 0.01538751473780755 | 2.679435697617345 |
| neural_inr/HF16 | 3/3 | 0.011012311834927924 | 2.4363567801813506 |
| neural_inr/HF24 | 3/3 | 0.010716977648154604 | 2.6669289058651797 |
| neural_inr/HF32 | 3/3 | 0.010877464581191187 | 2.316386977448953 |
| neural_inr/HF48 | 3/3 | 0.01114731148270419 | 2.4953207809614395 |
| original_fixed_rbf/HF8 | 3/3 | 0.01417222756790536 | 2.3399627255954254 |
| original_fixed_rbf/HF16 | 3/3 | 0.01281490039690136 | 3.0342000867634944 |
| original_fixed_rbf/HF24 | 3/3 | 0.00826885667219708 | 1.9995239241616214 |
| original_fixed_rbf/HF32 | 3/3 | 0.007512889794598781 | 1.6073707647055082 |
| original_fixed_rbf/HF48 | 3/3 | 0.007441344881599365 | 1.3663533159042744 |
| output_pca/HF8 | 3/3 | 0.014157653802253273 | 2.4524744740320736 |
| output_pca/HF16 | 3/3 | 0.014314534809085915 | 3.0342000867634944 |
| output_pca/HF24 | 3/3 | 0.008589655969413634 | 1.9995239241616214 |
| output_pca/HF32 | 3/3 | 0.007596772517760626 | 1.5449364832849113 |
| output_pca/HF48 | 3/3 | 0.00652515365685142 | 1.4713981326953585 |

## Prospective accepted-mode accuracy

Method freeze: `2026-10-06T09:39:20.827965+00:00`. Geometry generation: `2026-10-06T09:39:28.581352+00:00`. LF-only predictions: `2026-10-06T09:40:38.179112+00:00`. No prospective HF input is supplied to predictors.

| Group / model | Accepted geometries / rows | Geometry-median field RMS | Geometry-median frequency error % | Row field P95 |
| --- | ---: | ---: | ---: | ---: |
| challenge / cokriging_matern32 | 16 / 150 | 0.0044637524 | 1.1128465 | 0.089430626 |
| challenge / current_pca48_rbf | 16 / 150 | 0.0049362793 | 1.0136988 | 0.084971845 |
| challenge / direct_hf_matern32 | 16 / 150 | 0.004317646 | 2.3473189 | 0.045716324 |
| challenge / neural_inr_lr0.0003 | 16 / 150 | 0.01632305 | 1.5227544 | 0.096536535 |
| challenge / output_pca_matern32 | 16 / 150 | 0.0048053342 | 1.1128465 | 0.08411177 |
| challenge / region_selected | 16 / 150 | 0.010249976 | 2.5438479 | 0.078832447 |
| challenge / uncorrected_lf | 16 / 150 | 0.019216666 | 6.86091 | 0.099485725 |
| iid / cokriging_matern32 | 32 / 298 | 0.0043457555 | 0.8275916 | 0.063219721 |
| iid / current_pca48_rbf | 32 / 298 | 0.0051043861 | 0.94687425 | 0.055885534 |
| iid / direct_hf_matern32 | 32 / 298 | 0.0050903742 | 2.1296646 | 0.050952596 |
| iid / neural_inr_lr0.0003 | 32 / 298 | 0.013208411 | 1.5192599 | 0.067312114 |
| iid / output_pca_matern32 | 32 / 298 | 0.0043103698 | 0.8275916 | 0.056418239 |
| iid / region_selected | 32 / 298 | 0.0055778466 | 1.1331725 | 0.064425359 |
| iid / uncorrected_lf | 32 / 298 | 0.014766871 | 4.5854519 | 0.071650097 |

## Independent calibration and useful uncertainty

Raw scales remain selected globally and regionally; local residual scales were not promoted. Both recipes are independently calibrated and compared diagnostically without changing the frozen selection. Frequency interval score is in Hz; relative width is descriptive, not a proper score.

| IID selected-mean interval | Joint run coverage | Mean field width | Mean frequency width Hz | Field interval score | Frequency interval score Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
| 90% marginal | 26/32 | 0.48096283 | 457.78683 | 0.48118128 | 469.07946 |
| 90% joint | 29/32 | 0.48096283 | 1630.7673 | 0.48118128 | 1630.7673 |
| 95% marginal | 31/32 | 0.83576866 | 664.61202 | 0.83576866 | 672.43616 |
| 95% joint | 32/32 | 0.83576866 | 2833.7828 | 0.83576866 | 2833.7828 |

Full Clopper–Pearson coverage intervals, Spearman discrimination, retained-risk/referral curves, RMS balls, mode rows, quarantines and geometry-paired bootstrap differences are in the JSON/CSV artifacts. A finite rank is not a sharpness or shift-coverage guarantee.

## HF-verified engineering screening

Threshold: **961.333328 Hz**, fixed from historical training labels. Only stable reference label1 is screened. Decisions were recorded before evaluation HF; uncertain identities or unsupported intervals refer to HF.

| Group / policy | Cases | HF referrals | False acceptances | False rejections | Verified correct saved calls | Unverified automatic decisions | Observed criterion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| challenge/cokriging_matern32 | 16 | 5 | 0 | 1 | 10 | 0 | True |
| challenge/current_pca48_rbf | 16 | 4 | 0 | 1 | 11 | 0 | True |
| challenge/direct_hf_matern32 | 16 | 9 | 0 | 0 | 7 | 0 | True |
| challenge/lf_only | 16 | 11 | 0 | 0 | 5 | 0 | True |
| challenge/neural_inr_lr0.0003 | 16 | 6 | 0 | 1 | 9 | 0 | True |
| challenge/output_pca_matern32 | 16 | 5 | 0 | 1 | 10 | 0 | True |
| challenge/region_selected | 16 | 5 | 0 | 1 | 10 | 0 | True |
| iid/cokriging_matern32 | 32 | 5 | 0 | 0 | 27 | 0 | True |
| iid/current_pca48_rbf | 32 | 4 | 0 | 0 | 28 | 0 | True |
| iid/direct_hf_matern32 | 32 | 16 | 0 | 0 | 16 | 0 | True |
| iid/lf_only | 32 | 24 | 0 | 0 | 8 | 0 | True |
| iid/neural_inr_lr0.0003 | 32 | 6 | 0 | 0 | 26 | 0 | True |
| iid/output_pca_matern32 | 32 | 5 | 0 | 0 | 27 | 0 | True |
| iid/region_selected | 32 | 6 | 0 | 1 | 25 | 0 | True |

The validation campaign acquired every HF reference. Operational savings are counterfactual, not a refund of validation or setup cost. Per-policy training/calibration counts, fit/inference time, measured native reference time and finite-sample false-acceptance uncertainty are preserved. Historical training solve times are unavailable and are not fabricated.

LF solve/save reported-time sums (0.1-second log rounding retained): `{'calibration': {'timed_runs': 32, 'sum_logged_seconds': 16.7, 'maximum_sum_rounding_error_seconds': 1.6}, 'iid': {'timed_runs': 32, 'sum_logged_seconds': 16.0, 'maximum_sum_rounding_error_seconds': 1.6}, 'challenge': {'timed_runs': 16, 'sum_logged_seconds': 8.0, 'maximum_sum_rounding_error_seconds': 0.8}}`. Full JSON identifies rounding limits and omitted process startup.

## Reproduction and release

The two immutable local archives are `audited_baseline.zip` and `p1_improvement_artifacts.zip`; their recorded hashes are in the committed [local package manifest](p1_published_results/package_manifest.json). Both original archives remain local. Per the user publication restriction, no ZIP archives, MPH simulation files, raw NPZ arrays or fitted binary weights are uploaded to Git or GitHub Releases. The repository contains P1 implementation, tests, environment, reports and checksum-identified text results; it is not a complete publicly downloadable raw-data/model release.

The recorded [full portable restoration smoke](p1_published_results/archive_restoration_smoke.json) and [package verification](p1_published_results/package_verification.json) authenticate the local package, not remote downloads. With those local inputs/weights, `python p1_delivery.py replay --output <study-root>` performs a no-refit LF-only replay; without them a clean public checkout cannot reproduce frozen predictions. [P1_REPRODUCIBILITY.md](P1_REPRODUCIBILITY.md) separates source-only checks, saved-data analysis and licensed native reruns. Publication performs no new COMSOL solves or model fitting.

## Retained limitations

- **Stage 1:** Sampled full shell displacement/director state with physical mass quadrature, not assembled full-DOF convergence; individual 2209/3408 field tolerance failures remain.
- **Stage 2:** A physical-output PCA oracle can improve representation without improving full predictions. Regional training-CV promotion is not an OOD guarantee.
- **Stage 3:** Historical evaluation cases; each two-configuration family receives the same tuning splits, not identical optimizer dynamics. Bands are three-repeat extrema, not confidence intervals; do not infer a universal sample-efficiency winner.
- **Stage 4:** Finite conformal ranks and observed coverage do not imply sharpness, conditional coverage, epistemic meaning, or OOD validity. Calibration is over accepted-mode populations.
- **Stage 5:** Independent draws within the same original physical bounds; challenge theta>=60 is excluded only for the regional model, not a new physical family.
- **Stage 6:** Stable reference label1 and a benchmark threshold, not a universal fundamental-frequency or service certificate. Full validation paid all HF costs; savings are counterfactual operation, with training/calibration cost disclosed.
- **Stage 7:** Only source and text evidence are publicly published. Immutable archives and native MPH references remain local; there are no GitHub bulk release downloads. Recorded local checksums/restoration do not establish public raw-data/model availability. Float32 neural replay preserves the original800-row batch layout; a retained subset-layout attempt differed by1.05e-8.
