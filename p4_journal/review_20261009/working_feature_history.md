# P4 working-feature history — evidence reconstruction, 2026-10-09

## Provenance and scope
Two successive working-history butlers collected journal/source/result evidence but did not publish a synthesis before stopping. The dispatcher completed this report from their captured tool outputs (`history://WorkingFeatureHistory`, `history://WorkingHistoryReport`, native WorkingHistoryReport.jsonl), and direct journal/source reads. This is therefore a dispatcher-completed reconstruction of butler-collected evidence, not a falsely attributed completed butler report. The parallel constraint/model survey is separately preserved in `constraint_features_and_models.md`.

Coverage: feature discovery from early sidecar screens through shared modal extraction, sensor placement, analytical coordinates, symmetric reformulation, noise training, nonlinear learners, correlation, conditional diagnosis, first/second spatial patterns, blend and ingredient confirmation. Includes contemporary reversals and scope limits. No new experiment, fit, reserved-design scoring or recommendation ranking was performed for this report. Numbers are mean per-design absolute margin error unless explicitly labelled otherwise. Before/after pairs refer to matched comparisons, not a pooled cross-experiment league table. Source-linked journal entries provide experiment records and their limitations; many outputs were also directly opened by the butler. Earlier verification counts below are archived evidence, not newly rerun checks.

## 1. Benchmark boundaries: do not manufacture a continuous score trajectory
Legacy expanded/saturation dataset: 106 accepted designs from160 attempted;54 rejected (28 already unstable at lower modeled velocity,26 boundary not found). Original TRAIN partition77 designs/4454 physical transient clips, mostly60 clips/design (65 designs), others34–58. Stable2614/unstable1840 clips; stable and unstable last5%918/920. Noise copies do not add structures. These were independent initial-condition clips, not continuous trajectories suitable for recurrent stitching. Original TRAIN grouped CV repeatedly held out designs inside the77, with original reserved selection/calibration/test designs not scored by this feature program.

Most ridge discovery used5 outer/3 inner design folds, train-only scaling, inner alpha selection, original partitions seed42. Later fixed neural studies used32->16 ReLU MLP, alpha1, native LBFGS, seeds42/43/44, same five outer design folds; full-design partitions keep all noise copies together. Report seed-wise metrics averaged, not ensemble predictions. Bootstrap intervals resample fixed OOF whole-design predictions, without model refits, generally without multiplicity correction. Fresh noise is acquisition replication of existing structures, not independent structural confirmation.

Early modal and sensor experiments used causal51-sample per-channel calibration, cap50, measured dt, LS-ESPRIT lag64/max8 oscillatory pairs and fixed relative singular threshold1e-6. Sensor study replayed float64 raw responses on a25-location grid; equal absolute Gaussian noise referenced original-eight aggregate calibration RMS, before per-channel normalization/capping. Selected16 adopted by user: x={8,15,22,30}/31, y={1,8,15,22}/31. Nested selector used only fitting-design physical observability surrogates, not task labels. Noise1/5% are uncalibrated sensitivity settings, not established hardware requirements.

Current temporal development benchmark is a DIFFERENT population:4 fit designs D000007/33/68/87,2 validation D000042/45,2 repeatedly inspected held D000053/126; continuous8s crossing/stable/retreat trajectories,2 replicates,128-sample windows at1kHz/stride64. Command-free finite unsaturated windows plus first64->next64 modal-forecast explained-energy>=.5 gate. Framewise input174+observed+age,32->16 head; rejected feature columns zeroed. Proper clean/IID1 fitting, Adam, validation checkpoints; gap outputs extrapolate. This benchmark has very limited support and missing farther-stable/sign-exclusion populations. Its scores cannot be compared directly with the legacy77-design clip CV as if only architecture changed. The bounded384ms history result is in `frontier_evidence_packet.md` and its pinned artifact directory.

## 2. Early sidecar claims reversed by comparable evaluation
Source: [journal lines173–249](../JOURNAL.md), `p4_groups/ridge_screen_20260928/`, `ridge_cv_20260928/`, `ridge_modal_cv_20260928/`.

Three-epoch sidecar promise screens had54 optimizer updates versus baseline/bundle1400; G6 replaced the readout without a matched no-feature readout-refit control. Thus earlier categorical memorization/interference claims were too strong. Keep screen intent, not final harm claims.

Six-selection-design ridge screen: baseline .047449, bundle .045244, +zcr .046270, +log_dt .046297. Nested77-design CV: baseline .051092, +zcr .051701, +log_dt .050602, cleaned bundle .050429; all paired improvement intervals included zero. Screen did not establish a robust winning feature roster. Envelope slope, dominant frequency, bandwidth duplicated existing ridge features; cal_scale was normalized RMS~1 with std2.17e-9, not recovered physical amplitude. Removing duplicated/constant items preserved the small bundle screen gain.

### Shared waveform-only modal extraction was the first clear representation gain
Baseline7 -> quality-only .050993 -> modal22 .040407 -> modal+cleaned sidecars .039262 (baseline .051092). Modal beats baseline76/77 designs and all5 folds; delta-.010684,95%[-.012799,-.008965]. Modal versus quality-only delta-.010586,95%[-.012946,-.008753]: not just quality/missingness cues. Legacy |m|<.15 slice .038318->.031657; this is broader than the current exact boundary.

Contemporary observation: clean simulated responses recovered8 modes in every clip, median relative residual5.29e-16. Extremely favorable identification, NOT realistic-noise robustness or a deployment guarantee. Opposite-phase channels preserved by pooled CHANNEL Hankel identification, not spatial waveform averaging. No solver eigenvalues, true margins or critical velocities in extraction; max8 budget is simulator-informed prior. Archived runtime exercised4454 clips and baseline replay1.19e-14;10 focused behavioral checks recorded.

## 3. Sensor placement beat simply adding old-layout sensors
Source: `sensor_layout_20260928_v2/results.json`; journal251–362. First aborted indexing run explicitly invalidated: NumPy interpreted tuple layouts as multidimensional indices; corrected integer row arrays, second-mode exposure regression retained. Do not use aborted results.

Nested regularized logdet observability selection, same prediction22/modal extractor and paired candidate waveforms/noise, no GRU fitting:
| Layout | Clean | Matched1% | Matched5% |
|---|---:|---:|---:|
|Original8|.040407|.050655|.051952|
|Greedy4|.042046|.042310|.043079|
|Greedy8|.042533|.043530|.043909|
|Greedy16|.039926|.040076|.044437|
|Extend original8 to16|.041155|.049268|.051017|

Greedy4 versus original8 delta-.008345 at1%,95%[-.009613,-.007022];-.008873 at5%,[-.010112,-.007623]. Greedy16 clean gain uncertain (interval crosses0);1% gain-.010579. Extension16 did not establish noise gains. At1%, critical-mode recovery original8/greedy4/greedy16=77.8/99.3/98.1%; all-mode32.9/97.0/88.9%. At5%, critical73.4/97.1/94.6%. Three original sensors' median signal RMS below the1% injected-noise floor; posthoc diagnostic, not a selection input.

Interpretation at the time: location/weak-channel SNR matters more than sensor count alone;4 economical noise-tolerant,16 lower-noise accuracy alternative. User chose16. Finite-grid physics selector is not globally optimal or task-learned placement.

### Numerical transfer failure was corrected before new feature claims
Clean-fit/noisy-query current8 MAE3147.5/3583.0; selected4 still1.186/28.958. Clean fit-error column std8.42e-13 versus noisy median.722, with1e-8 scale floor, produced huge ridge leverage. Diagnostic holding column at mean was attribution ONLY, not permanent solution. Root correction `fit_ridge_scaling`: unsupported if train std<=1e-8*max(1,|train mean|), force coefficient0; retained columns keep std+1e-8. Train-only, no noisy-query selection or prediction clipping. Selected16 fold0 clean->1% same-alpha clip MAE81.19254->.04205. Mode count/residual/availability were unsupported. Genuine noise-shift failures remained after fixing numerical dust.

## 4. Analytical damping/stability/spacing added information and inductive bias
Source: `physics_features_20260929/{results,features}.json/npz`; journal372–428. First22 columns exactly replay selected16 modal cache; all41 columns finite. Same77 designs/4454 clips/5x3 grouped CV and paired0/1/5% noise. Analytical extraction uses estimated poles only.
| Arm | Clean matched |1% matched|5% matched|
|---|---:|---:|---:|
|Ridge7|.048018|.047517|.054028|
|Modal22|.039926|.040076|.044437|
|+damping|.033391|.033455|.036711|
|+stability|.035771|.037873|.042065|
|+spacing|.030878|.033989|.046573|
|+all physics41|.024913|.028127|.038874|

All-physics versus modal clean delta-.015013,95%[-.016705,-.013363];1%-.011949,[-.013727,-.010125]. Damping helps every matched noise level; spacing alone hurts5%, delta+.002136,[+.001144,+.003280]. |m|<.15 design MAE modal .031908/.032155/.035898 -> all .020602/.022808/.029748.

Clean-trained transfer1/5%: modal .040852/.048435; damping .034674/.044043; stability .039434/.050874; spacing .033201/.078890; all .030562/.092652. Contemporary decision: damping robust core; full physics stronger low-noise precision, NOT universal robustness winner. The original small denominator remained physically/statistically fragile despite finite scaling.

## 5. Noise mixture and symmetric spacing: real improvement with tail tradeoffs
Sources: `robustness_mixed_noise_20260929/`, `robustness_symmetric_spacing_20260929/`, `robustness_symmetric_mixed_20260929/`; journal605–807. Mixed clean/IID1/IID5 views13362 rows still4454 clips/77 designs, all copies grouped; alpha retuned, no fixed-alpha effective-regularization equivalence claim.

Noise-matched -> equal-mixture full physics: clean .024913->.032639,1% .028127->.033763,5% .038874->.036088. Broad-noise training sacrificed low-noise advantage. Damping mix .033209/.034297/.036021 versus matched .033391/.033455/.036711;5% gain uncertain.

Posthoc transfer attribution: selected noisy near-DC lead made denominators tiny (row1245 frequencies66.576/73.306Hz -> .239/66.685Hz); noisy maxima relative distances198.66/278.59 versus clean~2.26. No evidence all failures had same cause or truth identity.

Controlled reformulation only: gap2|f1-f2|/(f1+f2), pole distance2|p1-p2|/(|p1|+|p2|), both<=2. Estimator/visible mode selection/layout/other columns fixed; no clipping, oracle mode routing or discarded cases. Original->symmetric full physics clean .024913->.024384; matched1 .028127->.027410; matched5 .038874->.034381; clean->5 .092652->.047482. Independent noise similarly improves; symmetric model does NOT recover missing modes.

Symmetric mixed versus original mixed clean .032639->.027587;1 .033763->.029585;5 .036088->.034644, all paired delta intervals below0. Fresh independent1/5 .029890/.034741. Still worse low-noise MAE than symmetric specialists. Mean gains did not uniformly repair tails: clean-to5 p99 .20364->.22220; symmetric mixed fresh5 p99 .13527 versus original mixed .12631 and symmetric matched .12475. Subsequent whole-design uncertainty showed p99 difference interval crossed0. Do not convert point tail ranks into proven population risk.

Fixed2:1 overprediction-weighted ridge reduced dangerous tails but introduced pessimistic bias/false-warning proxies, e.g. fresh5 p01-.14721->-.16678; stable predicted-nonpositive7.31%->9.53%. Kept separate conservative candidate; not a99% bound, not replacement. Source `robustness_asymmetric_tail_20260929/results.json`, journal812–867.

## 6. Nonlinear small models helped; correlation survived seeds; size/skip did not earn replacement
Sources: `nonlinear_tabular_20260929/`, `mlp_validation_20260929/`, `mlp_feature_seeds_20260929/`; journal871–1098. Same cached symmetric41 features, equal three-view noise training and grouped nested selection. Boosting: squared loss,200 rounds, selected15 leaves; MLP two hidden layers32->16, selected alpha1. Original MLP search contained max_iter warnings; warnings retained, later fitting cap/convergence diagnosis done rather than suppressed.
|Learner|Clean|1%|5%|Fresh IID1|Fresh IID5|
|---|---:|---:|---:|---:|---:|
|Symmetric mixed ridge|.027587|.029585|.034644|.029890|.034741|
|Boosting|.018886|.021284|.025776|.021745|.026196|
|MLP32->16|.017258|.019408|.025430|.019820|.025809|

Three-seed fixed-alpha MLP confirmation clean .017183,1 .019346,5 .025043,fresh1 .019715,fresh5 .025369. Not an isolated seed. Larger64->32 fresh1 .019595/fresh5 .025455 with intervals crossing0 versus original seed; input/output skip worse than matched custom plain, not evidence against all residual architectures. No GRU abandonment.

Fixed feature revisit with3 seeds: correlation makes42-column roster (symmetric41+mean finite pairwise Pearson). Baseline -> +xcorr clean .017183->.016512;1 .019346->.018624;5 .025043->.023338;fresh1 .019715->.018737;fresh5 .025369->.023507. Every seed/condition improves. Adding zcr/kurt too yields .016407/.018488/.023260/.018600/.023163; marginal gains versus correlation alone small/uncertain. Contemporary choice retained parsimonious correlation, not bundle by default.

Acquisition-only mismatch control tested residual balanced channel gains.9/1.1, shared Gaussian noise rho.8, reported dt+/-2%, without dynamics/source mismatch. MLP42 beat ridge42 overall but residual calibration and common-noise fragility remained. No label/sign retuning or misleading inference of independent physics validation. Source `physical_mismatch_20260929/results.json` and journal1099 onward.

## 7. Augmentation versus paired consistency: control volume and optimizer matter
Sources: `correlated_training_20260929/{augmentation_results,augmentation_seed43,augmentation_seed44,seed_confirmation_analysis}.json`; journal1173–1449. Four-view independent augmentation clean/IID1_928/IID5_928/IID1_929 compared with replacing last view by correlated1. Same42/MLP/alpha/folds.

Seed42 original3 -> independent4 clean .016518->.016037;freshiid1 .018586->.017890;freshcorr1 .020102->.019366. Correlated4 .016612/.018139/.019039: much of original3 improvement is extra training view, not correlated distribution. Three-seed independent4 .016045/.017869/.019424 versus correlated4 .016586/.018149/.019338; correlated gain-.000086 interval[-.000474,+.000292], clean cost+.000541 interval above0. Independent4 became accuracy-led42 baseline.

Paired lambda1 consistency versus SAME custom lambda0 clean .016343->.017253;freshcorr MAE .019095->.019069 essentially unchanged, p99 .097204->.082541; pair prediction gap .008059->.006261. Underestimation/tight-sign tradeoffs remained (corr unstable-positive1.576%->2.120%). Kept correlated augmentation/consistency as bounded alternatives, not uniform wins, reliability guarantees or discarded model classes.

## 8. Conditional diagnosis established clean converter/context failure; weighting only partially repairs it
Sources: `conditional_bias_diagnosis_20260929/`, `local_weighting_20260929/`; journal1451–1617. Original independent4/42/32->16/alpha1/folds/seeds fixed; train-design and OOF errors captured. Held predictions reproduce previous candidate exactly.

Clean train->OOF overall .012279->.016045;fresh1 .014131->.017869. Stable0<m<.05 fresh1 train/OOF bias+.007808/+.009549; unstable last5 bias-.000436/-.000512. Farther-stable bias-.015547/-.020495: regression toward middle already present in training, not solely generalization.

64 severe clean stable clips recover all8 modes; critical alpha median error6.32e-7/s, critical visible61/64. Top3 strength includes critical only56.25%. Fresh noise adds fragility, but failed clean extraction/visibility is not general explanation. Source796 m=.041345, OOF errors+.1272/+.1310/+.1246, in-sample+.0950 with accurately recovered poles; closest training targets .2770/.1846/.2192. Severe and other-stable normalized feature distances similar (.297/.298), but neighbor target dispersion .0289/.0109. Finite-distance neighbors are NOT exact collisions or proof of an information ceiling.

Fixed local weight2 for0<m<.05 normalized mean1 (unchanged effective L2 denominator/scaling), matched explicit-unit native controls. Fresh1 near MAE .014663->.012813, e>.03 11.22%->7.84%; overall .017869->.017686 with interval crossing0. Farther-stable MAE .038442->.039412, underestimation>.05 24.44%->26.18%, both intervals above0. Contemporary decision: fitting-bias component recoverable, but no free correction/sufficient-representation proof; keep original point estimator, local2 ablation. More noise copies cannot fix clean omitted-context cases.

## 9. First spatial pattern was the strongest controlled representation gain
Source `spatial_modal_20260930/`; journal1618–1741. Hypothesis: pooled summaries discard system context. First least-stable>=1%-strength pattern adds65 columns ->107. Same selected16/calibration/estimator/independent4/MLP/alpha/seeds/folds; thirty real/sham fits;42 baseline and local2 predictions reused.107-width sham permutes ONLY spatial columns within outer train/held partitions by physical source row, same mapping across noise/seeds; not deployable. Must beat42 and sham, not merely sham.
|Evaluation|Base42|Spatial107|Shuffled107|Local2/42|
|---|---:|---:|---:|---:|
|Clean|.016045|.008587|.020287|.016219|
|Fresh IID1|.017869|.010131|.022329|.017686|
|Fresh correlated1|.019424|.011031|.023675|.019018|
|Fresh IID5|.023146|.016927|.028603|.022923|

Spatial beats baseline EVERY seed/fold at all11 conditions; fresh1 delta-.007738,95%[-.008819,-.006740]. Stable last5 fresh1 MAE .014663->.007880, optimism>.03 11.22%->3.89%; farther-stable MAE .038442->.021637, under>.05 24.44%->6.58%. Former source796 errors become+.00765/-.00278/+.00885. Not solely wider layer or an isolated lucky seed.

Exception: stable last1 fresh1 wrong signs15.31%->16.28%; unstable last1 essentially unchanged18.31%->18.11%. More accurate headroom does not certify timely warnings. Four new severe clean cases appeared although64 earlier cases removed; outcome-selected repair cohorts descriptive. Contemporary decision:107 strongest working representation, not deployment/information sufficiency; fair GRU comparison remains future/live.

## 10. Second spatial pattern: aggregate information with boundary/missingness costs
Source `two_mode_spatial_20260930/`; journal1743–1886. Append second-LEAST-STABLE visible65 pattern+normalized similarity+pair flag=67, NOT second strongest mode. Same107 controls refit to serialize/expose new queries; prior predictions reproduced exactly. Three arms107/174/shuffled67-addition174,45 fits,3 seeds,5 folds,15 acquisition conditions (four new1% draws).
|Evaluation|One107|Two174|Shuffled174|
|---|---:|---:|---:|
|Clean|.008587|.007049|.009416|
|Prior fresh IID1|.010131|.008883|.011171|
|Prior fresh correlated1|.011031|.009705|.012195|
|Fresh IID5|.016927|.016367|.019514|
|New IID1_935|.010355|.008939|.011542|
|New corr1_935/936|.010816|.009657|.012202|
|New IID1_937|.010075|.008692|.011087|
|New corr1_937/938|.011065|.009708|.012337|

Mean gain EACH seed/condition, not every fold/design. All four new1% global paired intervals below0;5% delta interval crossed0. Farther-stable improves. Stable last1 worsens on all four new1% draws with MAE intervals above0; e.g. IID935 .004335->.005166, wrong signs16.09%->20.35%. Stress near5 MAE .014020->.016583 and unstable-last1 wrong signs19.75%->27.78%. Costs also present in training. Second mode unavailable~38.5%, strongly associated with regime (86.96% missing unstable last5% clean), all cases retained; addition tests patterns AND missingness, not isolated phase/coupling mechanism.

Warm workstation single-thread CPU raw-to-margin timing p95 one6.935/two7.018ms, excluding acquisition/startup/IO. Not comparable with later full GPU temporal replay, not hard-deadline qualification. Contemporary decision:107 boundary reference;174 accuracy comparator, no automatic warning-focused replacement.

## 11. Fixed blend and ingredient studies refined, not eliminated, the tradeoff
Sources `blend_verification_20260930/`, `spatial_comprehensive_20260930/`; journal1888–2140. Fixed50/50 paired107+174, no weight search, no seed ensemble/target routing; frozen native predictors and six new draws. Global newIID1_939/941 one .009964/.009991, two .008876/.008828, blend .008512/.008547; correlated draws one .010860/.010763, two .009602/.009726, blend .009301/.009382. All global comparisons with both members improved with paired intervals below0. Four1% stable-last1 point MAEs improved; strict point guardrail passed3/4, IID939 unstable wrong sign16.87%->17.49% (interval crossed0). Both5% draws failed all-guardrail point rules. Same-run warm blend p95 10.241ms exceeded10ms; not qualified. Kept blend and107 comparators, not universal best-of-both/default.

Factorial ingredients: flags109, powers125, complex pairs157, similarity110, PC173, PS126, CS158, full174; complex entries contain amplitude AND phase.105 reduced-variant fits, lockedP/PC finalists before eight fresh-noise confirmation outcomes, same-width shams. Both finalists beat shams on primary global metrics; flags/similarity alone did not explain gain. Neither passes any of eight full condition-specific no-deterioration guardrail sets. Boundary critical-pole recovery100% clean/99.42% IID1; noise-fit cost persists on correctly frequency-matched critical cohorts and actual training views. Second-mode truth identity switches21.52% IID1,43.67% IID5,28.64% corr1 among resolved matched pairs; unresolved pairs explicit. Boundary repeated-noise prediction variance one->full .000000970->.000005110 atIID1, .000009720->.000043400 atIID5; these are finite two-draw descriptions, not population decomposition. Separate proper-fit/calibration/held-design warning studies used clip proxies, not event/lead qualification. More spatial ingredients did not yield a proven boundary-safe winner.

## 12. Exact current feature dictionary and equations
Source-checked `src/mechanics/p4_margin_estimation/{modal_features,baselines,sidecar}.py` and `two_mode_spatial_20260930/prepare_features.py:31–44`:

*7 envelope/FFT: slope(log spatial RMS) over physical t; coefficient t^2 in quadratic log-RMS fit; first-half minus second-half mean log-RMS; median/IQR channel log-|x| slopes; dominant non-DC frequency of mean-channel FFT; sqrt(sum P(f)(f-fbar)^2/sum P(f)).
*15 modal:3 strongest fitted components×(alpha,f,e), max-visible alpha, energy-weighted alpha, count, relative centered squared fit error, usable fraction, availability. alpha=log|z|/dt;f=arg(z)/(2pi dt). e_i=E_i/sum E_j, E_i=sum_s,t xhat_i,s(t)^2. Nonorthogonal interference means E sums not signal/physical energy.
*19 analytical:5 damping (-alpha/hypot(alpha,2pi f) top3 strength,min-visible,weighted);10 stability (3 visible modes by descending alpha/frequency tie, each alpha/f/e, count);4 spacing. Effective spacing first/third use symmetric ratios2|f1-f2|/(f1+f2) and2|p1-p2|/(|p1|+|p2|), plus min-visible symmetric frequency gap and neighbor flag. Raw `physics_feature_groups` default uses lead-only denominators; CURRENT builder explicitly overrides them.42=7+15+19+1 mean finite channel Pearson correlation.
*65 spatial: b b^H/(b^H b),16 diagonal powers+24 real+24 imaginary fixed-neighbor entries+availability. Select highest alpha among e>=.01.107=42+65.
*67 addition: same65 for SECOND highest-alpha visible pattern, |b1^H b2|^2/(||b1||^2||b2||^2), pair availability.174=107+67. Independent global complex scale/phase invariance; no restored raw channel amplitude ratios, no physical coupling measurement, no stochastic coherence claim. Missing pattern zero with validity flag. At inference no solver identity/design/U/ucrit/margin. Availability/age are appended separately in temporal models.

## 13. Reversals and remaining causal uncertainty
* Earlier0.04 ceiling disproved by new representations; no demonstrated information ceiling.
* Earlier simple sidecar memorization/interference conclusions overstated unequal-update/readout controls.
* Legacy known-speed multi-velocity inversion17–23m/s claims invalidated: U=ratio*ucrit sampling let supplied speeds encode unknown critical speed; velocity-only slope/model reproduced advantage. Those numbers are NOT evidence to reopen known-U inversion. New384ms causal response-history experiment has no supplied speed labels, but is not an independent known-speed sweep.
* Clean-identification success and severe headroom errors coexist; noise-only explanation false, but exact collision/nonidentifiability not demonstrated.
* Symmetry/calibration fixes, nonlinear mapping, correlation and spatial patterns have distinct controlled evidence. Better global MAE and p99 do not imply correct boundary signs, independent calibrated uncertainty or delivered warning benefit.
* GRU feature starvation remains relevant; neither enlarged MLP/skip failures nor current framewise accuracy establish recurrent models are obsolete. A fair GRU comparison needs the SAME improved representation and bounded causal aperture, not stitched independent clips or unbounded8s state.
* No final blind same-family test or independent physical/model-form validation. Progress ledger and fitted operational predictor unchanged by this reporting task.

## Coverage limits for frontier
This reconstructs the feature-development chain, not every later physical warning/model screen; the complete constraint survey and consolidated packet supply those diagnostics. Quantities copied from archived runs are not new verification. Exact historical source controls/results are linked by named artifact directories above; journal maintained original errors and decisions. Literature facts reside in the parallel survey, with snippet/full-text status explicit. Later trajectory128ms/384ms probes have different data, training support, optimization, input ages/gates and missingness than these77-design clips. Frontier must design falsifiable interventions from these differences, not claim a historical MAE delta directly transfers to the operational endpoint.
