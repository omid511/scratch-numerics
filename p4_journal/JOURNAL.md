# P4 lab log

Running notes, newest at bottom. I write what we thought, what we ran,
what surprised us, and what we decided — including the mistakes and the
dead ends, because that's what actually determines what to do next.
Numbers come from results.json files; code and outputs stay where they are.

G0 = saturation baseline. Targets = ucrit/resid/damp arms. Multivel =
inversion + e2e attention. Uncertainty = hybrid intervals. G1/G1b/G3/G4/G5/G6
= preprocessing + sidecar group program.

It started as a review request — check the P4 proposal, implementation,
data and model pipeline, efficiency, novelty, publishability /10. We said
upfront we knew about the sim-only gap, so that couldn't be the finding.
The claim under test was a continuous margin model trained on simulated
transients with domain randomization giving earlier, better-calibrated
warning than binary baselines and degrading gracefully under mismatch.
First decision, and it held the whole time: verify the pipeline before
believing any number, then chase the ridge-vs-GRU tie instead of writing
up early.

The saturation dataset landed 2026-09-25 17:55 — 6122 clips, 4454/762/906,
106 plates from 160 sampled (54 skips), seed 42, splits 77/13/16,
per-channel causal norm off the leading 51 samples, cap +-50,
pre_saturation.npy kept. Most clips sit near the boundary (81.8% |m|<0.15),
which matters later. We thought cap-50 training would transfer to cap
100/200 with recalibration, modestly worse, retrain winning. What we got:
GRU 0.050/0.052/0.052 raw cov ~0.75, ridge 0.0528 raw cov 0.90, CQR
~0.90-0.92 both, and cap transfer basically flat (0.051->0.053) with
matched retrain equal to fixed transfer. That surprised us — retraining
adds nothing, the original just generalizes. A third of clips saturate
(mean frac 0.15) so this isn't a no-op result. We kept fixed-transfer as
primary and banked saturation as a narrow publishable.

Same day the training kept dying — exit 143, orphan .pt files no complete
record referenced. The runner resumed finished records but not interrupted
epochs, so a trained checkpoint between save and record got orphaned and
retrained from scratch. That stung (the user's "everything should be
resumable" was right), so we built the epoch sidecar: atomic, before
on_epoch, carrying optimizer/scheduler/best/history/RNG/shuffle state,
fail-closed validation, completed-training commit before eval. The proof
caught a genuine bug along the way — val_loss wasn't appended before the
sidecar, so resumed history misaligned — and another one where optimizer
cross-loading accepted mismatched widths, so we added key/shape/dtype
checks. Proof passed bit-exact (fresh 4ep == crash+resume) plus every
reject case. After that, no more restarts; the live run stayed untouched.

Next morning the tie became the whole question. GRU 0.05 vs ridge 0.05,
raw coverage 0.75 vs 0.90 — 24k parameters and full waveforms tying seven
pooled envelope stats means the bottleneck is elsewhere, and saying so
felt more honest than defending either model. Two suspects: starved
features or noisy labels. The cheap decisive test was ridge on the
solver's own modal features — stalls at ~0.05 means labels are the
ceiling, drops below means the net is starved. It stalled on damping
(0.0502) and barely moved with +V (0.0494), but V+u_crit halved it
(0.023) with the exact formula residual at 0.000000. So the labels are
velocity in disguise — m = 1-V/u exact to 2.8e-8 — and both models were
estimating velocity ratio through noisy waveform proxies. Error
correlated +0.945 with mean|m| (far-from-boundary worse), +0.63 with
frequency. Decision: stop predicting margin harder; change the target
(ucrit, a real system property) AND blind velocity (resid, damping
content only), additively, same plumbing.

Targets landed 16:02 (`p4_targets_full/`). We expected the GRU to win
ucrit or resid. It lost both: ucrit dead for everyone (GRU ~512 = train
mean, ridge ~438 vs std 628), resid ridge 0.054 beats GRU 0.068-0.074
(paired +0.013, CI excludes 0). The blind arm justification had been
the -0.48 correlation (GRU beats velocity-only hardest where velocity
fails), so symmetric removal hurting the GRU was a real surprise. First
verdict, later revised: margin-from-single-waveform exhausted. Also
logged a minor sin — ridge ucrit physical MAE was inferred (0.698*std),
never stored; only GRU records got the field.

We then killed longer clips with math instead of a run. |alpha| median
19.5/s means |a|T ~= 10 already at 0.5s, 87% readable-unclamped; at 2s
only 18% stay readable, 30% newly clamp, 59% of decaying modes go
silent. Error tracks frequency (+0.63), not weakness. Stayed at 0.5s —
bottleneck is cycles per clip, not growth time. Multi-velocity stayed
live because slope across V is where ucrit actually lives.

Multivel inversion (`p4_multivel_full/`, 8 recs) felt like the unlock:
286/316 units had >=4 clips spanning >=0.25, no regen needed, frozen
heads, u=V/(1-m) pooled by median and LSQ fit. POC went ~100 to ~20.
Full: GRU-median ~20.7 vs ridge ~22.7 (tie), ridge-fit 17.0 vs GRU-fit
~31.5 (ridge wins, paired +14.5). Fit-pooling amplifies GRU's fatter
tails through 1/(1-m); median resists. Verdict then: 25x over
single-clip, net still doesn't beat hand baseline.

End-to-end attention was supposed to beat post-hoc inversion: K=4 clips
through a shared GRU into attention, straight to ucrit. Pilot: 495 vs
512 baseline — the frozen margin-encoder turned out invariant to ucrit
(mean-embedding corr 0.17/-0.05), so attention rearranged noise. Full
from-scratch: 496-501, bands +-1000. Looked structural until the
plumbing audit found per-unit Vn=(V-mean)/std deleting mean(V) — which
IS the signal. A velocity-only MLP went 67 global vs 539 per-unit vs 93
affine. One-line bug, not a finding. Reran fixed: full 143-170 vs
velocity-only 79-92. Waveforms genuinely hurt here. We said stop the
ucrit line unless sub-15 m/s matters; the free inversion stood.

Then the fishy smell, and refusing to conclude from it was the best
decision of the project. Audit: ucrit method valid (316 units, formula
to 3e-8, no leakage), resid clean (blind fit == own-split fit, no
shift), but the 17 m/s V-oracle was circular and had to be struck —
V=ratio*u_crit with fixed strata mean 0.9601 means mean(V)~=0.96u by
construction, slope 1.064 matching 1/0.96, joint R2 gain +0.0004,
absolute-V windows collapsing to 130-180. Every 17-23 number rode the
same sampler artifact. That hurt, but it cleared the table for what
came after.

We pulled in outside adversarial review twice, and it changed the work.
Kept: linearization with its error term and failure modes (kink, shallow
crossing, observability vs validity); "sufficient statistics" demoted to
"7 summaries ~= this GRU under this protocol"; ESPRIT named as the
untracked should-beat candidate; normalization DOFs (per-channel kills
7/7 static ratios, correlations survive); the clean ablation recipe.
Refuted by us: aliasing — dt is adaptive per clip with a hard guard,
audit showed 0.0% above Nyquist; replaced by the missing-timebase
confound (ridge uses physical dt, GRU never sees it). Conformal got
harsher than we had it: calib is 7 designs, not 13, so k=8=n+1 means
standard construction yields infinity and max-score gives 87.5% — not
conservative 90%, need n>=9. Language fixed everywhere. Oracle
"ceilings" demoted to model-class ceilings (ridge missing an exact
formula by 0.024 proves misspecification). The trivial baselines
embarrassed us into honesty: constant-median 0.091 vs 0.05 with GRU
slope 1.12/corr 0.82 kills the shrinkage story. Follow-ups we banked:
dt->omega_max side-channel, nominal-isn't-causal, permutation must be
TESTED (we did, 1e-8), hybrid claims split three ways.

Global-norm regen took two attempts — the first died 0/160 on an `rng`
NameError (40 min gone), then landed identical seeds/splits/labels with
cross-channel spread ~9700x. G3 (global + shared-gain DR) missed margin
(0.048, bar 0.04, corr flat) but damping fell ~30% with GRU beating
ridge 2/3 CIs. The four-way cell read damp-up/margin-flat, but then
plain_clean (0.037, no sidecar, no DR) beat G3 margin outright, killing
H_norm for margin. Static ratios were the bigger erasure and still not
margin. That sequencing mattered: without the clean baseline, G3's nudge
would have looked like progress.

The group program ran under additive-only rules with ridge parity and a
member-split protocol. G1 (clean+log_dt, 0.0395-0.0433) looked like a
timebase breakthrough until G1b's 2x2 showed plain_clean beating G0 by
-0.009..-0.013 with sidecar_dr tying G0 — the gain was DR removal, dt
~0, log_dt junk. Losses converged, so not epochs; DR's geometry (gain
rewriting summaries, skew breaking phase, mask doubling dim) does the
damage, code audited clean. G4 bundle16 (0.0383-0.0388, best-centered,
clears 0.04) looked like heroes until all 7 solos hurt (0.058-0.064):
a dedicated 1->16->16 MLP memorizes faster than it learns, so the
bundle win is joint-pattern-or-capacity. G6 frozen confirmed damage
sits in side-MLP+head; top4 0.0548 best; bundle16 0.070 pure
interference. G5: weights rescue kurt to neutral (0.048), mono adds
nothing and erases the weights gain — junk, harmful in company.
Velocity closed both ways: GRU-solo hurts (0.0629), ridge+V gains
+0.0001. Width controls (paired 0.25 vs shuffled 0.36 vs constant 0.31,
separately recalibrated) proved ordering real. Damping hybrid halved
iscore @0.93; margin hybrid died even calibrated (0.25 vs 0.22-0.24).
Infra along the way: `_signals_2d` mask-strip bug (16-row FFTs), shared
results.json race killing 5/7 singles (relaunched isolated), CPU thrash
7x150% (killed 4, kept top-3), batch/threads bumped for future runs.

Standing: margin plain_clean 0.037-0.042 > G4 bundle 0.0386 > G0-DR
0.050-0.052; damping global GRU 0.190-0.211 > ridge; uncertainty halves
damping iscore, dead on margin; ucrit sampler-ridden, e2e failed,
single-clip dead; falsification <=0.03 NOT met.

What's left, in order: unfrozen G4 LOO on top4 survivors (heroes vs
capacity — G6-LOO was frozen-head, different question); S2 16-sensor
regen only if xcorr/phase frozen probes show signal; oracle rung 1
(ESPRIT) vs rung 3 on |m|>=0.15 only; then real data, because sim
location is closed at clean ~0.04 and only the shape dividend travels.
Publishable as methods/negative: saturation transfer + artifact audit +
null with ceilings. Still 5/10 until real data.

2026-09-28: reopened the feature conclusion. The user clarified that
3-epoch solos were deliberate promise screens, not converged comparisons.
Keep that use, but do not infer final feature harm from comparing their
54 updates (3 epochs, batch 256) against the baseline/bundle's 1400
(20 epochs, batch 64). G6 also replaces the learned readout and has no
matched no-feature readout-refit control; its degradation alone does not
establish interference. The earlier memorization/interference language is
stronger than these controls support.

First cheap check: ridge baseline + each sidecar, tuning regularization
on the six selection designs with train-only scaling. Saved in
`p4_groups/ridge_screen_20260928/results.json`: mean design MAE 0.047449
baseline, 0.045244 bundle (-4.65%), 0.046270 +zcr, 0.046297 +log_dt.
The test/calibration splits were not scored. This was a screen, not
independent validation. env_slope/dom_freq/bandwidth duplicate existing
ridge features (exactly or nearly); cal_scale is normalized RMS ~=1,
with raw std 2.17e-9, not recovered physical calibration amplitude.
Removing all four leaves the bundle gain essentially unchanged.

Then nested grouped CV on the 77 TRAIN designs (4454 clips), five outer
folds and three inner folds, alpha and scaling selected entirely within
each outer training partition. `experiment_p4_ridge_cv.py` reproduces
it; initial results are in `p4_groups/ridge_cv_20260928/results.json`.
Mean design MAE: baseline 0.051092, +zcr 0.051701, +log_dt 0.050602,
clean bundle 0.050429. Paired design-bootstrap delta intervals:
zcr [-0.000731, +0.001726], dt [-0.001932, +0.000555], bundle
[-0.002052, +0.000743]. All include zero. The six-design screen
overstated robustness; no validated shortlist winner yet. These intervals
resample OOF design errors without model refits, and are exploratory,
not multiplicity-adjusted significance tests. CV MAE is not directly
comparable to the old test-set MAE.

The waveform-only modal probe then broke the ridge plateau. Added
`modal_features.py`: multichannel LS-ESPRIT on the unsaturated prefix,
64-sample temporal Hankel lag, at most eight oscillatory pairs plus DC,
fixed 1e-6 relative singular threshold. No solver eigenvalues, critical
velocities, or labels enter extraction. Features describe the three
strongest fitted modes' growth rates/frequencies/relative contributions,
maximum visible and weighted growth, and explicit extraction quality.
The eight-mode budget is a simulator-informed prior, not a learned order.
Sensor signals are not spatially averaged (opposite-phase sensors must
not cancel before identification). Fits use physical dt.

`p4_groups/ridge_modal_cv_20260928/results.json` and `features.npz`
contain the full nested-CV run: baseline 0.051092, +quality-only 0.050993,
+modal 0.040407 (-20.9%), +clean-bundle+modal 0.039262 (-23.2%).
Modal beats baseline on 76/77 designs and all five outer folds; combined
beats baseline on 74/77. Paired modal-minus-baseline delta is -0.010684,
95% design-bootstrap interval [-0.012799, -0.008965]. Modal-minus-quality
is -0.010586 [-0.012946, -0.008753], so the gain is not just the usable
window/quality descriptors. Combined-minus-clean-bundle is -0.011167
[-0.012839, -0.009662]. Same bootstrap caveats as above.

Post-hoc OOF slices (no refits): |m|<0.15 MAE 0.038318 -> 0.031657
with modal, 0.031039 combined; |m|>=0.15 0.110734 -> 0.080988,
0.077283 combined. Gain is not confined to the far-boundary tail.
Do not compare these CV scores directly with historical GRU test scores.

Verification: 10 focused tests passed (known growth/decay poles,
opposite-phase sensor cancellation, channel offsets, dt scaling,
small added noise, clipped-prefix recovery, explicit missing estimates,
invalid dt, and invariance of a fold's predictions/alpha to its held-out
labels). CLI exercised all 4454 training clips; baseline OOF predictions
match existing PhysicsFeatureRidge fit/predict to 1.19e-14.
Reproduce with `rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 .venv/bin/python experiment_p4_ridge_cv.py --modal
--output-dir <new-directory>`; existing result files are not overwritten.

Important limit: all clips yielded eight estimated modes, with median
relative reconstruction error 5.29e-16 on their usable prefixes. These
clean simulated responses are extremely favorable for identification;
synthetic small-noise checks are not a realistic-noise robustness study.
The result supports a representation bottleneck in pooled ridge features,
not an information ceiling or a deployment claim. Selection/calibration/
test designs remain unscored by this CV experiment. Next: freeze this
extractor and compare its robustness under realistic acquisition noise
and mismatch before expanding the neural architecture or sensor layout.

Next decision: user wants sensor placement/count before further GRU work.
Built `generate_p4_sensor_candidates.py` to replay only the 77 training
designs at a 5x5 interior grid (indices 1,8,15,22,30 on the 32x32 solver
grid), including every original sensor. Retains float64 raw physical
responses before normalization/capping, identical initial conditions and
timebase across every layout, and solver poles for diagnostics only.
Each design is an atomic resumable NPZ with a source fingerprint. First
full-design check: all 60 D000000 clips matched original-eight float32
waveforms exactly, with matching labels/dt. Resume skipped that shard.

`experiment_p4_sensor_layout.py` predefines current8, nested greedy4/8/16,
and current8-plus-eight extension. Selection maximizes mean training-design
regularized logdet of a 51-sample, modal-pair-scaled observability Gramian.
This is a physics surrogate, not a task-trained selector. Both outer and
inner CV partitions select their own layouts from their own training
designs. Prediction remains the same 7 ridge + 15 estimated modal features;
no GRU training. A full-training layout is recorded separately from the
fold-specific evaluated layouts.

Noise sensitivity is predeclared at 0/1/5% of original-eight raw calibration
RMS: equal absolute Gaussian noise at candidate locations, before causal
per-channel calibration and cap-50. Layouts share the same candidate noise
draws; weak locations do not receive artificially equalized SNR. These are
controlled noise levels, not calibrated real hardware specifications.
Evaluate both noise-matched ridge fitting and clean-fit/noisy-transfer.
Recovery pairs estimated/true modes by frequency (<=max(2 Hz, 5%)); report
missing modes explicitly and damping errors conditional on matching.

14 focused tests passed, including nested sensor budgets, redundant-sensor
rejection, invariance to eigenvector phase/scale, and outer-held-out-design
isolation. A real D000000 feature smoke exercised all five layouts at all
three noise levels, with finite features. Its current8 mode recovery went
from 100% clean to 21.25%/16.875% at 1%/5% noise. This one-design diagnostic
already warns that clean modal identification is fragile; it is not a
cross-design sensor comparison.

Acquisition completed: all 77 designs / 4454 distinct source rows,
maximum original-eight waveform replay error exactly 0. No held-out
designs regenerated. Candidate raw responses and observability shards
are in `p4_sensor_candidates/`. Full sensor run initially stopped before
predictions because canonical tuple layouts were interpreted as NumPy
multidimensional indices. Fixed by converting each layout to an integer
row-index array once; a regression test now verifies that adding sensor
rows exposes a second known mode. The aborted record is explicitly marked
in `p4_groups/sensor_layout_20260928/results.json`, not silently reused.

Completed results: `p4_groups/sensor_layout_20260928_v2/results.json`.
All outer/inner selections independently chose the same layouts; five
distinct layouts total. Rows below are mean design MAE with NOISE-MATCHED
training and evaluation, not clean-to-noisy transfer:

                         clean       1% noise    5% noise
current8                 0.040407    0.050655    0.051952
greedy4                  0.042046    0.042310    0.043079
greedy8                  0.042533    0.043530    0.043909
greedy16                 0.039926    0.040076    0.044437
current8+8 (extend16)     0.041155    0.049268    0.051017

Greedy4 minus current8: -0.008345 at 1% (95% paired interval
[-0.009613,-0.007022]), -0.008873 at 5%
([-0.010112,-0.007623]). Greedy16 minus current8 at 1%:
-0.010579 [-0.011953,-0.009213]. Its small clean gain has interval
[-0.001280,+0.000333], so no clear clean win over current8. Extension16
has intervals crossing zero at both noise levels: adding to the old
layout is much weaker than moving sensors. At 5%, greedy4 beats greedy16
by -0.001358 [-0.002753,-0.000199]; at 1%, greedy16 wins by 0.002233.
These remain exploratory OOF design-bootstrap intervals, no refit or
multiple-comparison adjustment.

Physical identification supports placement: at 1%, critical-mode recovery
current8/greedy4/greedy16 is 77.8%/99.3%/98.1%; all-mode recovery is
32.9%/97.0%/88.9%. At 5%, critical recovery is 73.4%/97.1%/94.6%.
All layouts identify all modes on clean simulations. Post-hoc raw-signal
diagnostic shows three original sensors have median RMS below the 1%
injected noise floor. See `candidate_signal_strength.json`; it was not
used to choose fold layouts. Per-channel normalization lets weak noisy
locations contribute alongside useful channels, and more channels need
not help this frozen estimator.

Selected four in normalized (x/Lx,y/Ly):
(30/31,1/31), (30/31,8/31), (30/31,15/31), (30/31,22/31).
Selected sixteen are the grid product x={8,15,22,30}/31,
y={1,8,15,22}/31. All current/nested/extension coordinates and fold plans
are recorded in results. Recommendation: four as the economical
noise-tolerant candidate, sixteen as the lower-noise accuracy alternative;
do not double the old layout blindly. A hardware choice still depends on
actual noise/cost; this finite-grid physics selector is not a globally
optimal or Gumbel-learned layout.

Important new failure: CLEAN-trained models are not transferable to this
noise. Current8 design MAE explodes to 3147.5/3583.0 at 1%/5%; even greedy4
is 1.186/28.958. Kept those failures in the primary results. Cached-fold
replay reproduces them. Attribution on outer fold 0 at 1%: the modal
relative-fit-error feature has clean std 8.42e-13 (median ~5e-16), noisy
median 0.722; the ridge standardizer's +1e-8 floor amplifies this shift.
That feature alone contributes mean absolute prediction 4007.07.
Holding it at the training mean diagnostically reduces that fold's clip
MAE from 4007.21 to 0.15208, still poor. This is attribution, not a fix or
replacement result; see `transfer_diagnostic.json`. The layout study
kept the estimator frozen. Before GRU/transfer claims, address numerical
quality-feature scaling and train under the intended acquisition noise.

Verification: 15 focused tests pass; real CLI covered all layouts/noise
conditions and 25 prediction records. Current8 clean OOF predictions are
bit-identical to the previous +modal CV. Completed acquisition and study
resume paths both exercised (no pending shards on completed resume).
Run generation with `rtk proxy env OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python
generate_p4_sensor_candidates.py --workers 4`; evaluate with the same
environment and `experiment_p4_sensor_layout.py --workers 4 --output
p4_groups/sensor_layout_20260928_v2`. No GRU training or held-out
selection/calibration/test scoring was done.

2026-09-29: user keeps selected16 and emphasizes online production use,
not simulator-dependent inference or defending a GRU. Auxiliary learned
modal heads were discussed but deferred: estimated poles and analytical
transforms are already cheap; an auxiliary model needs a demonstrated
robustness/latency advantage. User also pushed back on excessive code.
This round reused cached acquisitions and existing nested-CV functions;
no new experiment runner was added.

Corrected the numerical ridge failure before comparing physics features.
`fit_ridge_scaling` now treats a column as unsupported when training std
<=1e-8*max(1,abs(training mean)); its coefficient is zero, rather than
learning a slope from numerical dust. Retained columns keep std+1e-8
scaling. The decision uses training data only, never noisy/query data;
no prediction clipping. Both PhysicsFeatureRidge and the CV helper use
the same rule. Existing saved results/checkpoints were not rewritten.
A deterministic regression failed before the fix (a numerical-feature
query shift changed prediction by 53718.96) and passed afterward.
Selected16 outer fold 0, clean->1% noise, same alpha: clip MAE
81.19254 -> 0.04205. Unsupported columns were mode count, reconstruction
error, and availability. This is a source correction, unlike the earlier
diagnostic holding one feature at its mean.

Added three waveform-only analytical families in `modal_features.py`:
damping = -alpha/hypot(alpha,2*pi*f), including top-energy, minimum-visible,
and energy-weighted ratios; stability = the three largest growth rates
among >=1%-energy fitted modes, with their associated frequencies/weights;
spacing = nearest-frequency and normalized complex-pole separation around
the least-stable visible mode, minimum relative frequency gap, and a
neighbor-availability flag. "Least stable visible" is not oracle critical
mode identity. Gaps at one operating point are not velocity derivatives.
No solver poles, ucrit, velocities, or labels enter extraction.

`p4_groups/physics_features_20260929/results.json` and `features.npz`
save the full 41-feature matrix, row identities, family columns, source
hashes, alpha curves, inactive columns, OOF predictions, paired intervals,
and timings. The first 22 feature columns exactly reproduce the prior
selected16 feature cache. Same 77 training designs / 4454 clips, 5 outer
and 3 inner folds, same 0/1/5% paired noise draws. The selected16 layout
was identical in every preceding inner/outer fold. All historical holdout
selection/calibration/test designs remain unscored.

Mean design MAE, NOISE-MATCHED training/evaluation:
                         clean       1% noise    5% noise
ridge7                   0.048018    0.047517    0.054028
corrected modal base     0.039926    0.040076    0.044437
+damping                 0.033391    0.033455    0.036711
+stability               0.035771    0.037873    0.042065
+spacing                 0.030878    0.033989    0.046573
+all physics             0.024913    0.028127    0.038874

Near-boundary |m|<0.15 MAE: modal base 0.031908/0.032155/0.035898,
all physics 0.020602/0.022808/0.029748; damping 0.025783/0.025855/
0.028256. Clean all-physics minus modal-base delta -0.015013,
95% paired interval [-0.016705,-0.013363]; 1% delta -0.011949
[-0.013727,-0.010125]. Damping helps at every tested matched noise level;
5% delta -0.007725 [-0.008982,-0.006408]. Spacing alone hurts at 5%:
+0.002136 [+0.001144,+0.003280]. Intervals remain exploratory OOF
design resampling without model refits or multiplicity correction.

CLEAN-trained transfer (1% / 5%): corrected modal base 0.040852/0.048435;
damping 0.034674/0.044043; stability 0.039434/0.050874;
spacing 0.033201/0.078890; all physics 0.030562/0.092652.
The numeric explosion is fixed, but real distribution-shift failures
remain: the full bundle is not the robust winner at higher noise.
Recommendation: damping is the strongest robust core; full physics is
the low-noise accuracy candidate. These transforms add no information
beyond the waveform; they expose nonlinear physical relationships to
ridge. The old ~0.04 information-ceiling language is untenable. The
0.0249 result is training-design CV, NOT a pass of the original held-out
test falsification criterion.

Online computation benchmark: Ryzen 5 5500U, one BLAS thread, 231 windows
(first clip per training design at each noise level), five warmups per
pipeline. Includes calibration, cap, feature extraction, standardization
and fitted ridge prediction; excludes acquisition wait, startup and I/O.
Median/p95 ms: ridge7 1.071/1.815; modal baseline 4.789/6.221;
all physics 4.887/6.042. No simulator or flutter search runs at inference.
Window acquisition still matters (these timed windows span 0.431-0.5 s);
this is not a target-hardware deadline guarantee, a GRU speed comparison,
or proof of superiority to established online estimators.

22 targeted tests passed. Existing PhysicsFeatureRidge fit/predict on
actual selected16 clips agrees with the corrected CV implementation to
1.39e-16. Runtime probes exercised analytical extraction over every
training clip/noise level and all 30 nested-CV conditions.

User asked about sensor positions as features. Not tested as predictor
inputs; they were used for placement/projection. Fixed normalized
selected16 coordinates are constant across clips and add no ridge
information. Absolute coordinates would also encode plate geometry;
known dimensions should be tested explicitly if available in deployment.
Variable-layout, channel-aware models could use coordinates, but adding
them after spatial pooling loses the channel/location association.
Spatial modal patterns and cross-sensor phase remain untested properly;
do not conflate them with constant coordinate columns.

2026-09-29: user asked how to judge current scores and what targets to
pursue, then requested that the targets and additional criteria be logged.
Current analytical results are promising, not production validation.
Best per-condition design MAEs: clean 0.0249, 1% matched noise 0.0281,
5% matched noise 0.0367, clean-to-1% 0.0306, clean-to-5% 0.0440.
These are condition-specific winners, not one configuration achieving
all five. They remain training-design CV; untouched test performance is
unknown. Full physics clean-to-5% MAE is 0.0927, so robustness is still
a central limitation.

Proposed score tiers below are provisional engineering targets, NOT
application-derived safety requirements or established attainable limits.
Use mean per-design MAE, identical evaluation definitions, one frozen
configuration across conditions, and ultimately held-out designs.
Matched-noise conditions permit noise-matched training; clean-to-5%
instead evaluates the clean-trained model without a noisy-data refit.
Noise percentages retain the study's original-eight aggregate calibration
RMS reference; they are not calibrated hardware noise specifications.

| Tier | Clean | 1% matched | 5% matched | Clean-to-5% | Purpose |
|---|---:|---:|---:|---:|---|
| 1: Confirm useful result | <=0.030 | <=0.033 | <=0.040 | <=0.050 | Confirm current gains generalize |
| 2: Strong practical candidate | <=0.020 | <=0.025 | <=0.030 | <=0.035 | Improve precision and robustness |
| 3: High-precision candidate | <=0.010 | <=0.015 | <=0.020 | <=0.025 | Investigate warnings closer to flutter |
| 4: Stretch research target | <=0.005 | <=0.010 | <=0.015 | <=0.020 | Ambitious, not a commitment |

Standing: numerically around Tier 1, but not confirmed with one chosen
configuration on held-out designs. Recommendation: aim for Tier 2 while
confirming Tier 1. Tier 4 may require temporal tracking or additional
operating information; single-window identifiability at that precision
has not been established. A robust 0.02-0.03 under realistic acquisition
is preferable to 0.005 on clean simulations followed by noisy failure.

Interpretation: m = 1 - V/ucrit, so MAE 0.025 is an average error of
2.5 percentage points of critical-velocity ratio, not 2.5% relative error
in margin. That may still be large for distinguishing 2% from 5% below
flutter. Existing "near-boundary" scores use |m|<0.15; they do not prove
precision within the last few percentage points before the boundary.

Additional acceptance criteria, required beyond MAE:
- Unsafe overestimation: quantify the upper tail of predicted minus true
  margin, especially near flutter; this is the dangerous error direction.
- Warning performance: missed-event rate at a specified false-alarm rate
  and warning lead time. Evaluate on continuous trajectories, not
  unrelated clips stitched together.
- Uncertainty: interval coverage AND useful width on held-out designs.
- Availability: frequency of unavailable/unreliable modal estimates;
  do not silently exclude difficult cases from reported performance.
- Online cost: provisional p95 computation target <=10 ms/window on
  target hardware, reporting acquisition duration and update interval
  separately. Current ~6 ms p95 on the benchmark CPU is encouraging,
  not a deployment deadline guarantee.

Numerical safety-tail, missed-event, false-alarm, lead-time, coverage,
width, and availability thresholds remain to be set from the intended
application. No MAE tier alone certifies safe deployment. This entry
records targets and interpretation only; no new experiment was run.

2026-09-29: robustness before further clean-performance optimization.
User approved a bounded robustness study and requested ongoing journal
updates plus a discussion of next robustness options when finished.
The four-hour background job was only a wake-up timer, not an experiment.

Tested one extraction change: temporal Hankel lag 64 -> 128. Selected16,
normalization/cap, maximum mode order, feature definitions, ridge scaling,
alpha grid, 5 outer/3 inner design folds and paired noise draws unchanged.
Used all 4454 clips / 77 training designs. Solver poles are diagnostic
references only, never predictor inputs. No held-out scoring or regeneration.
Initial one-clip-per-design probe suggested better frequency recovery with
longer aperture; full evaluation was necessary before accepting that proxy.
Other hypotheses (fixed-order noise poles and feature-identity instability)
remain unisolated; this experiment does not establish their causes.

Full-data identification, lag64 -> lag128:
- 1% noise: mean clip mode recovery 88.93% -> 93.33%; critical-mode
  recovery 98.07% -> 99.03%; median clip matched-alpha MAE
  0.664 -> 0.502 per second.
- 5% noise: mode recovery 65.36% -> 67.84%; critical recovery
  94.59% -> 95.55%; median clip matched-alpha MAE 3.225 -> 2.949/s.
- Clean: both recover all modes. Nonempty estimates: 100% for both at
  every noise level. Nonempty is NOT a reliability guarantee. Matching
  uses the existing <=max(2 Hz, 5% true frequency) criterion; alpha errors
  are conditional on successful frequency matches.

Mean design margin MAE (lag64 -> lag128):
| Condition | Modal + damping | Full physics |
|---|---:|---:|
| Clean matched | 0.033391 -> 0.033391 | 0.024913 -> 0.024899 |
| 1% matched | 0.033455 -> 0.033648 | 0.028127 -> 0.028174 |
| 5% matched | 0.036711 -> 0.040890 | 0.038874 -> 0.039444 |
| Clean-to-1% | 0.034674 -> 0.034372 | 0.030562 -> 0.029640 |
| Clean-to-5% | 0.044043 -> 0.043998 | 0.092652 -> 0.150327 |

5% matched damping degradation: paired mean delta +0.004178,
95% design-bootstrap interval [+0.003083,+0.005527]. Full-physics
matched delta interval [-0.001742,+0.003243] includes zero. Full-physics
clean-to-5% degradation is tail-sensitive: delta +0.057674, interval
[-0.029303,+0.154342]. These are exploratory OOF design resamples,
not model-refit uncertainty or multiplicity-corrected confirmatory tests.

Dangerous-error tails were not uniformly worse. For 5% matched damping,
clip-weighted p95(predicted-true margin) improves 0.09163 -> 0.08807,
while near-boundary clip MAE worsens 0.02832 -> 0.03251. Full-physics
5% matched p95 worsens 0.08421 -> 0.08808. Full-physics clean-to-5%
p95 improves 0.07942 -> 0.07194 despite much worse overall MAE:
one signed quantile cannot summarize the failure. Saved p95/p99 and
near-boundary p95 use signed errors over all applicable clips, not only
positive errors. Near means |m|<0.15. This study's near MAE is clip-weighted,
unlike the prior design-macro near MAE; do not compare them as identical.

Paired extraction timing: modal estimation + modal/physics transforms
on already preprocessed inputs, one BLAS thread, 231 windows, five warmups.
Median/p95: lag64 3.50/5.18 ms; lag128 7.01/9.19 ms. Excludes
normalization, ridge, startup/I/O and acquisition; not full-pipeline timing.

Decision: DO NOT adopt lag128. Better identification metrics did not
produce better margin robustness, and extraction costs roughly doubled.
Default estimator/source unchanged. No permanent tests added for this
investigation. Baseline modal/physics feature replay checked on every
clip/noise row with allclose atol=rtol=1e-10; actual lag128 extraction and
all ten grouped-CV comparisons exercised. Evidence:
`p4_groups/robustness_lag_20260929/results.json` and
`features_and_diagnostics.npz` retain predictions, folds/alpha curves,
paired intervals, timing, source hashes, row identities, features and
both estimators' identification diagnostics.

Next robustness options, not yet run:
1. Mixed-noise training using existing paired features; keep copies of
   each design together in every fold, weight noise levels explicitly,
   fit scaling/tune alpha only inside training partitions, and evaluate
   separately by noise level. This directly addresses training mismatch
   without increasing inference cost. Evaluation noise realizations must
   remain design-disjoint; a later independent-noise-draw check is needed.
2. Mode-selection/spacing stability: identify which estimated modes supply
   spacing features and whether outliers/identity changes explain the
   transfer tails before adding thresholds or dropping features.
3. Noise-aware order selection or channel weighting if diagnostics justify
   it; guard against losing weak critical modes or physical calibration.
4. Causal mode tracking across windows, only with continuous trajectories.
   Do not create trajectories by concatenating unrelated clips.
Recommendation: mixed-noise training is the next smallest controlled
experiment; retain damping and full physics as separate candidates.
Noise percentages remain sensitivity settings, not measured deployment
specifications. Do not overfit the pipeline to an arbitrary 5% stress case.

2026-09-29: completed equal-mixture noise-training experiment. Reused
lag64 selected16 features; concatenate clean/1%/5% copies, tile targets
and design IDs. 13362 training/evaluation rows but still only 77 designs
and 4454 underlying clips. Existing nested_cv performs all scaling and
alpha selection within training partitions. Augmented outer design sets
were checked identical to the original folds; all copies stay together.
One fitted predictor per outer fold/feature arm evaluates all noise levels.
Equal noise weighting is an experimental choice, not deployment prevalence.
Alpha grid unchanged and retuned; sum-squares ridge means tripling rows
changes effective regularization at a fixed alpha.

Mean design MAE, noise-matched reference -> mixed training:
| Evaluation noise | Modal + damping | Full physics |
|---|---:|---:|
| Clean | 0.033391 -> 0.033209 | 0.024913 -> 0.032639 |
| 1% | 0.033455 -> 0.034297 | 0.028127 -> 0.033763 |
| 5% | 0.036711 -> 0.036021 | 0.038874 -> 0.036088 |

Damping 5% delta -0.000690 has paired 95% design-bootstrap interval
[-0.001913,+0.000212]: no established gain over noise-matched training.
Its 1% degradation +0.000842 has interval [+0.000345,+0.001330].
Full physics 5% improves -0.002787 [-0.004545,-0.001176], but clean
degrades +0.007726 [+0.006316,+0.009177]. Thus the mixed bundle loses
its low-noise advantage. Intervals are exploratory OOF design resampling,
not refit uncertainty or multiplicity-corrected tests.

Independent-noise check: regenerated only measurement noise/features
from retained raw trajectories with SeedSequence([929,source_index]),
versus training seed 928. Same normalization/cap/lag64/feature extraction.
Evaluated original outer-fold training data and selected alphas without
retuning. No solver truth used; no new structural simulations.
Mean design MAE (mixed / matched-noise / clean-trained reference):
- Damping, 1%: 0.034362 / 0.033529 / 0.034917.
- Damping, 5%: 0.036123 / 0.036834 / 0.044423.
- Full physics, 1%: 0.033684 / 0.028619 / 0.030868.
- Full physics, 5%: 0.036308 / 0.039010 / 0.080828.
The tradeoff persists under fresh noise. These are not independent
structures or final test evidence.

Independent 5% damping signed-error p95/p99 improves from matched
0.09129/0.13513 to mixed 0.08910/0.12792. At 1%, damping p95/p99
worsens from 0.08667/0.12248 to 0.09229/0.12700. Full physics 5%
p95/p99 is 0.08595/0.12631 mixed versus 0.08615/0.12818 matched.
Tails are clip-weighted quantiles of predicted minus true margin over
all clips, not conditional on positive errors. Saved near-boundary
metrics use |m|<0.15 and clip weighting. Availability/extractor unchanged;
no clips dropped. No measured new inference-speed claim: feature counts
and prediction path are unchanged, only fitted ridge parameters differ.

Decision: retain mixed-noise damping as a robust candidate across unknown
noise levels, not a universally better replacement. Do not replace the
low-noise full bundle with its mixed-trained version. Main gain is removal
of training mismatch, not a breakthrough below the matched-noise floor.
No source/default changes or permanent tests added for this investigation.
Actual mixed nested CV and frozen-fold independent-noise predictions were
exercised; all extracted features finite. No final held-out scoring.

Evidence: `p4_groups/robustness_mixed_noise_20260929/results.json`
stores protocol, folds, alpha curves, predictions, per-condition errors,
tails, paired intervals and source hashes. `independent_noise_features.npz`
stores fresh-noise features and row identities; original feature matrix
remains in physics_features_20260929. Next proposed investigation:
diagnose spacing-feature outliers and mode/neighbor identity changes
before introducing gating, thresholds or another estimator change.

2026-09-29: user questioned realism of 5% noise. It is an uncalibrated
stress setting, not an established deployment requirement; 1% is also
unvalidated. Sigma is a fraction of original-eight aggregate calibration
RMS, NOT each selected sensor's own RMS. At equal reference/signal RMS,
5% corresponds to about 26 dB SNR; tenfold weaker/stronger signals give
about 6/46 dB. Real sensor bandwidth/noise floor and operating amplitudes
must anchor eventual physical-unit noise tests. Current Gaussian noise
does not cover correlated interference, drift, outliers or sensor failures.
Do not sacrifice low-noise performance solely to optimize this stress case.

Continued robustness: attribution of clean-trained full-physics prediction
shifts on paired 5% data. Reconstructed frozen fold predictions agree with
saved OOF predictions to atol=1e-10. Largest mean absolute contributions:
relative pole distance 0.05773 (maximum 77.21), lead-relative frequency
gap 0.02157 (maximum 44.96). These contributions may cancel; they are
not additive absolute-error attribution. Clean maxima of both ratios
are about 2.26; noisy maxima reach 198.66 and 278.59 respectively.
One concrete case, source row 1245: selected clean lead/neighbor frequencies
66.576/73.306 Hz become 0.239/66.685 Hz. This confirms a radically different
selected pair and a small denominator; it does not prove physical mode
identity or that every failure shares this mechanism.

Tested one representation change, leaving estimator, mode/neighbor
selection, other features, lag64, selected16, normalization and CV fixed:
frequency gap = 2*abs(f1-f2)/(f1+f2);
pole distance = 2*abs(p1-p2)/(abs(p1)+abs(p2)).
Both are symmetric and bounded by 2 for valid estimated oscillatory modes.
No prediction clipping, oracle mode filtering or discarded clips.
This is an explicit feature reformulation, not recovery of missing modes.
Re-extracted original ratios from actual waveforms and checked agreement
with both seed928 and seed929 caches before replacing columns 37/39.
All other columns preserved. Candidate bounds exercised on all rows.

Mean design MAE, original -> symmetric full physics:
| Condition | Original | Symmetric |
|---|---:|---:|
| Clean matched | 0.024913 | 0.024384 |
| 1% matched | 0.028127 | 0.027410 |
| 5% matched | 0.038874 | 0.034381 |
| Clean-to-1% | 0.030562 | 0.030260 |
| Clean-to-5% | 0.092652 | 0.047482 |

Paired design-bootstrap delta intervals: clean [-0.001055,+0.000073],
1% matched [-0.001044,-0.000360], 5% matched [-0.006164,-0.002965],
clean-to-5% [-0.100087,-0.002649]. Exploratory OOF resampling only;
not refit uncertainty or multiplicity-corrected confirmation.
Independent seed929 noise, frozen fold training data/selected alphas:
1% matched 0.028619 -> 0.028010; 5% matched 0.039010 -> 0.034761;
clean-to-1% 0.030868 -> 0.030710; clean-to-5% 0.080828 -> 0.046605.
No independent structural designs or held-out tests scored.

Important safety caveat: better MAE did NOT improve dangerous tails
uniformly. Clean-to-5% signed-error p95/p99 worsens
0.07942/0.20364 -> 0.08874/0.22220, despite near-boundary clip MAE
improving 0.06560 -> 0.03608. 5% matched p99 slightly worsens
0.12388 -> 0.12536; clean p99 0.08404 -> 0.08972.
Tails are clip-weighted predicted-minus-true margin quantiles, not
positive-error-only quantiles; near MAE uses |m|<0.15 and clip weighting.

Decision: symmetric spacing is a promising accuracy-robustness candidate
that preserves low-noise gains, NOT a safety-qualified replacement.
Keep source/default feature definitions unchanged pending tail-focused
comparison; old fitted models must never be fed the new feature semantics
without refitting. No new permanent tests for this investigation.
No new latency benchmark or deployment-noise claim.

Evidence in `p4_groups/robustness_symmetric_spacing_20260929/`:
results.json holds five nested-CV comparisons, four independent-noise
checks, predictions, alphas, paired intervals and tails; features.npz
holds candidate features, independent-noise features, row IDs and selected
lead/neighbor frequencies/growth rates. Names in that NPZ are explicitly
original_feature_names; changed column semantics are recorded in results.
A first in-memory extraction was interrupted by kernel termination;
the standalone temporary probe completed all 77 designs afterward.
Next: tail-focused comparison, potentially mixed-noise training with
symmetric features, before adopting a new default. Do not call lower MAE
alone a solved robustness problem.

2026-09-29: combined symmetric spacing with equal-mixture noise training.
Reused saved candidate matrices; no new extraction, sensor layout or
estimator change. Same nested_cv and alpha grid on clean/1%/5% copies
with tiled design IDs (13362 rows, still 4454 clips / 77 designs).
Checked augmented outer-fold design sets equal original sets and all
three copies present. Inner/outer design isolation enforced by nested_cv.
One fitted model per fold evaluates all noise conditions. Independent
seed929 evaluation uses unchanged training data and selected fold alphas.

Mean design MAE:
| Noise | Original full mixed | Symmetric full mixed | Symmetric noise-matched |
|---|---:|---:|---:|
| Clean | 0.032639 | 0.027587 | 0.024384 |
| 1% | 0.033763 | 0.029585 | 0.027410 |
| 5% | 0.036088 | 0.034644 | 0.034381 |

Symmetric mixed improves on original mixed at every noise level;
paired 95% design-bootstrap delta intervals: clean
[-0.005812,-0.004325], 1% [-0.005002,-0.003342],
5% [-0.002552,-0.000380]. Against symmetric noise-matched models,
clean/1% remain worse by 0.003202/0.002175 with intervals above zero;
5% difference +0.000263 has interval [-0.000561,+0.001103].
Thus combining the changes recovers much of the mixed model's lost
low-noise accuracy but does not remove the specialization tradeoff.

Independent-noise mean design MAE, symmetric mixed:
1% 0.029890; 5% 0.034741. Original mixed: 0.033684/0.036308;
symmetric noise-matched: 0.028010/0.034761. Paired differences versus
original mixed again have intervals below zero. Independent noise is
not independent structures; all intervals remain exploratory OOF design
resampling, without model refits or multiplicity correction.

Tail check, independent seed929, signed predicted-minus-true margin:
| Noise/model | p95 | p99 |
|---|---:|---:|
| 1% symmetric mixed | 0.07389 | 0.11259 |
| 1% original mixed | 0.08575 | 0.11673 |
| 1% symmetric matched | 0.06331 | 0.12796 |
| 5% symmetric mixed | 0.08174 | 0.13527 |
| 5% original mixed | 0.08595 | 0.12631 |
| 5% symmetric matched | 0.08604 | 0.12475 |
| 5% symmetric clean-trained | 0.08107 | 0.22305 |

Combined training reduces p99 versus clean-trained symmetric transfer,
but 5% p99 is worse than original mixed and symmetric matched, despite
better p95. Original seed928 likewise shows 5% p99 0.13886 combined
versus 0.12375 original mixed. Tail comparisons are descriptive
clip-weighted quantiles, not significance claims. No model dominates
all error/safety metrics; no deployment risk threshold is yet specified.

Decision: symmetric mixed is the leading tested broad-noise MAE candidate,
not a safety-qualified universal winner. Retain symmetric matched as
low-noise specialist and original mixed/damping as tail references.
No default/source change, no new permanent tests or runtime benchmark.
Actual nested CV and frozen-fold independent-noise predictions exercised;
all input features finite. No held-out selection/calibration/test scoring.
Evidence: `p4_groups/robustness_symmetric_mixed_20260929/results.json`
contains fold alphas/curves, predictions, condition/reference metrics,
paired intervals, parent data hashes and fitting-source hashes.
Next useful question: are the remaining upper-tail errors concentrated
in specific designs/operating regimes or low-quality mode estimates?
Diagnose that before introducing a safety-biased loss or reliability
gate; MAE-tuned ridge has not been optimized for dangerous-error tails.

2026-09-29: p99 diagnosis and one fixed asymmetric-loss experiment.
Independent 5% symmetric-mixed upper 1% comprises 45 clips across
28 designs (largest design contributions 4 clips each); 44/45 have
|m|<0.15. Sixteen clips recur in the original-noise top1%.
Tail median reconstruction error 0.247 versus 0.107 elsewhere;
median usable fraction 1 and mode count 8 in both groups. This suggests
a quality association, not a validated reliability gate or isolated
extraction cause. Errors are not confined to one or two designs.
The prior p99 ranking itself is uncertain: symmetric mixed minus original
mixed independent-5% delta +0.008960, paired whole-design bootstrap
95% interval [-0.002869,+0.020500], 2000 resamples.

Tested squared loss weighting overprediction 2 versus underprediction 1.
Same symmetric features, equal noise mixture, training-only scaling,
design folds and previously inner-MAE-selected fold alphas. Ratio fixed
before evaluation; no ratio sweep or alpha retuning. Thus this isolates
loss asymmetry, not a fully retuned asymmetric family. Weighted ridge
IRLS with unpenalized intercept converged in 4/4/5/4/4 iterations.
Checks: ratio1 matches existing ridge on actual features to atol1e-10;
intercept-only targets [-1,+1] produce the analytic ratio2 answer -1/3;
weighted normal-equation gradient max<1e-6 at convergence.

Mean design MAE and signed-error p99, symmetric mixed -> asymmetric:
| Condition | MAE | p99 |
|---|---:|---:|
| Clean | 0.027587 -> 0.028173 | 0.093372 -> 0.078736 |
| Original 1% noise | 0.029585 -> 0.029956 | 0.106049 -> 0.090011 |
| Original 5% noise | 0.034644 -> 0.035945 | 0.138860 -> 0.114060 |
| Independent 1% | 0.029890 -> 0.030112 | 0.112588 -> 0.094724 |
| Independent 5% | 0.034741 -> 0.035990 | 0.135265 -> 0.108255 |

Independent-noise paired p99 delta intervals (2000 whole-design
resamples): 1% [-0.026486,-0.015172], 5% [-0.032091,-0.023362].
Independent5% MAE worsens +0.001249, design-bootstrap interval
[+0.000363,+0.002141]. These use fixed OOF predictions, no model refits,
no multiplicity correction, and remain exploratory after repeated studies.

Conservatism cost must accompany the tail gain. Independent1%/5% mean
signed bias is -0.007545/-0.014866. Independent5% p01 worsens
-0.14721 -> -0.16678. Using a zero-margin threshold only as a diagnostic:
stable clips predicted nonpositive increase 4.74% -> 6.69% at 1% and
7.31% -> 9.53% at 5%; unstable clips predicted positive decrease
2.45% -> 1.79% and 1.85% -> 1.58%, respectively. These are clip
classification proxies, NOT event false-alarm rates, missed-event rates
or warning lead times. There is no validated application alarm threshold.

Decision: asymmetric fitting demonstrably reduces dangerous tails with
modest MAE cost, but more conservative/false-warning predictions.
Keep it as a separate conservative candidate, not a replacement for the
point estimator or a calibrated safety bound. A 2:1 asymmetric squared
loss is not a 99% lower quantile or guarantee. No source/default changes,
no permanent tests for this investigation, no held-out scoring, no new
latency claim. Actual fold fits and independent-noise predictions exercised.
Evidence: `p4_groups/robustness_asymmetric_tail_20260929/results.json`
stores diagnosis, protocol/solver checks, predictions, baseline comparisons,
MAE and p99 intervals, signed tails and zero-threshold proxy tradeoffs.
Next decision requires the acceptable false-warning/availability tradeoff;
do not optimize p99 alone or add an unvalidated quality gate.

2026-09-29: user requested both boosting and MLP plus tree importance,
then explicitly specified TWO hidden layers instead of one. Installed
scikit-learn 1.9.1 via `uv add --optional p4-tabular scikit-learn`;
dependency manifest/lock updated. Added experiment_p4_tabular.py to run
the controlled comparison on cached symmetric features. No waveform
regeneration or final held-out scoring.

Predeclared setup: histogram gradient boosting, squared loss, 200 rounds,
learning rate .05, min leaf 100, L2=1, max leaves grid [7,15].
MLP: ReLU 32->16 hidden units, L-BFGS, L2 grid [.01,1], max_iter=2000,
max_fun=50000, tol=1e-7. Initialization seed42; no random early-stopping
split. Same equal clean/1%/5% mixture, original 5 outer/3 inner design
folds; all copies stay together. Train-fold scaling/inactive-feature mask
for both estimators. Select by inner design MAE averaged over noise.
Convergence warnings are retained and printed, not hidden.

Boosting finished first, selecting 15 leaves in every outer fold.
MAE clean/1%/5%: 0.018886/0.021284/0.025776 versus ridge
0.027587/0.029585/0.034644. Independent-noise MAE 0.021745/0.026196.
All paired design-bootstrap improvement intervals exclude zero; still
exploratory fixed-OOF resampling, not independent structural confirmation.
Original-noise p99 0.07233/0.09648/0.11646; independent p99
0.09978/0.11844. MLP evaluation still running at this entry.

Boosting importance: outer-validation permutation increase in design
MAE, three repeats per column or jointly permuted family, separately by
noise. No training importance or causal claims. Correlated features can
mask importance; shuffling can create physically inconsistent combinations.
At 1% noise top individual increases: max_visible_alpha_per_s +0.04428,
energy_mode1_damping_ratio +0.01555, symmetric lead-neighbor frequency
gap +0.01482, energy-weighted damping +0.00746. Family increases:
modal +0.05222, damping +0.02661, spacing +0.01701, stability +0.00983,
original ridge7 +0.00516. Do not interpret these as additive contributions.
Boosting one-window scaling+prediction median/p95 2.64/3.16 ms
(150 calls, five warmups per fold, one BLAS thread); excludes extraction,
acquisition and startup. Partial results saved after each completed model
under p4_groups/nonlinear_tabular_20260929/results.json.

MLP completed: selected L2=1 in every outer fold. Fifteen fits during
the search reached max_iter=2000; warnings are preserved per outer fold
(three each). Do not claim the entire search converged or that this is
a multi-seed neural comparison. No silent warning suppression.
MLP mean design MAE clean/1%/5% = 0.017258/0.019408/0.025430;
independent-noise 1%/5% = 0.019820/0.025809.
All paired improvement intervals versus ridge exclude zero.
MLP minus boosting independent1% delta -0.001926, 95% design-bootstrap
interval [-0.002949,-0.000930]; independent5% delta -0.000388,
interval [-0.001342,+0.000617]. Clean/1% favor MLP, 5% not resolved.
These intervals do not capture seed/optimization or model-refit variation.

MLP p99 clean/1%/5% = 0.06742/0.09209/0.12468; independent
1%/5% = 0.09120/0.11865. Independent5% boosting p99 is 0.11844,
essentially the same observed value; do not infer a safety ranking.
MLP near-boundary clip MAE original clean/1%/5% =
0.01286/0.01448/0.01901; |m|<0.15 remains a broad region.
MLP one-window scaling+prediction median/p95 0.115/0.232 ms,
same 150-call/five-warmup-per-fold protocol; excludes analytical feature
extraction, acquisition and startup. Boosting 2.635/3.163 ms. No summed
full-pipeline p95 claim from separate component measurements.

Decision: nonlinear mapping is materially useful under this protocol.
Two-hidden-layer MLP is the leading low-noise accuracy/compute candidate;
boosting is a strong independent model-family comparator with informative
permutation results. Do not immediately replace deployment defaults:
confirm optimization/seed stability before freezing model selection,
then use reserved selection/calibration/test sets for their intended roles.
Tree importance is evidence of model reliance, not proof of causal physics.
All requested comparisons completed through the actual CLI. No permanent
tests added: no new shared fitting API; existing helpers reused. Optional
dependency and runner are permanent, results/journal document protocol.

Reproduce in a fresh output directory:
`rtk proxy uv sync --extra p4-tabular`, then
`rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiment_p4_tabular.py --output <new-directory>`.
Results retain per-model folds, selected settings, inner score curves,
warnings, original/independent predictions, paired intervals, raw
permutation records and latency. No held-out test scoring was performed.

2026-09-29: user approved MLP validation plus larger architecture, skip
connections and a revisit of previously unhelpful ridge features. Reserved
selection/calibration/test designs remain out of scope. Prespecified:
- Original 32->16 MLP at initialization seeds42/43/44.
- Larger 64->32 MLP at seed42.
- Jointly trained input-to-output linear skip on 32->16, with matched
  no-skip objective/initialization control at seed42. Skip weights start
  at zero and receive the same L2 penalty as other weights.
- zcr, xcorr and kurtosis individually and together on 32->16, seed42.
  zcr previously worsened grouped ridge MAE on average; xcorr/kurtosis
  were weak/unvalidated rather than established harmful features.
  No duplicate modal summaries or invalid normalized cal_scale added.

Architecture/feature comparisons fix alpha=1, selected in every original
outer fold, to isolate changes; this is not per-arm hyperparameter tuning.
Convergence diagnostic revisits both original alphas on outer fold0's
three inner partitions with max_iter10000/max_fun200000. All new fits
retain convergence metadata. Reuse original five grouped design folds,
equal-noise training mixture and independent seed929 evaluation.

Implementation consolidated into existing experiment_p4_tabular.py via
--mlp-validation. Initially created experiment_p4_mlp_validation.py was
removed before running after its code was moved, NOT discarded.
Reusable skip objective lives in
src/mechanics/p4_margin_estimation/skip_mlp.py. User clarified reuse was
the concern and accepted removal once code retention was explained.
Existing MLP fit path now shares configurable fit_network. The skip
objective has a plain sklearn-compatible control and analytical gradient
checks; completion/results will be recorded below.

Run started:
`rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiment_p4_tabular.py --mlp-validation
--output p4_groups/mlp_validation_20260929`.

MLP validation completed: six diagnostic inner fits and all 50 outer
fits across ten arms reported convergence. Convergence diagnosis was
bounded to outer fold0's three inner partitions: alpha .01 needed
4715/2871/4259 iterations and its mean inner MAE worsened from
0.025530 at the old cap to 0.026381 after convergence; alpha1
converged in 794/943/1002 iterations with unchanged MAE 0.023764.
Thus the weakly regularized candidate remained worse; do not claim all
historical warnings were individually traced beyond this diagnostic.
Baseline seed42 outer predictions reproduce the previous result exactly
at displayed precision; all selected baseline fits finished below2000
iterations. Increasing the cap did not create the baseline gain.

Baseline 32->16 seed42/43/44 mean design MAE (mean +/- population SD,
not standard error or confidence interval):
clean 0.017183 +/-0.000210; 1% 0.019346 +/-0.000210;
5% 0.025043 +/-0.000274. Independent1% 0.019715 +/-0.000156;
independent5% 0.025369 +/-0.000311. The original seed was not an
isolated lucky result. Only the baseline received this three-seed check.

Fixed-alpha1 seed42 comparisons, mean design MAE:
| Arm | Clean | 1% | 5% | Independent1% | Independent5% |
|---|---:|---:|---:|---:|---:|
| Baseline 32->16 | 0.017258 | 0.019408 | 0.025430 | 0.019820 | 0.025809 |
| Larger 64->32 | 0.016775 | 0.019158 | 0.024876 | 0.019595 | 0.025455 |
| Matched custom plain | 0.017016 | 0.019340 | 0.025251 | 0.019416 | 0.025443 |
| Input/output skip | 0.017602 | 0.019644 | 0.025600 | 0.020102 | 0.026045 |
| +zcr | 0.017059 | 0.019534 | 0.025377 | 0.019736 | 0.026014 |
| +xcorr | 0.016518 | 0.018635 | 0.023429 | 0.018770 | 0.023567 |
| +kurt | 0.017241 | 0.019505 | 0.025042 | 0.019749 | 0.025351 |
| +all three | 0.016421 | 0.018550 | 0.023441 | 0.018672 | 0.023244 |

Larger network's independent-noise improvement intervals include zero.
Skip must be compared with custom plain, NOT attributed against sklearn:
the same objective/initialization can follow different numerical optimizer
trajectories. Plain matches sklearn predictions on the small objective
check within1e-6 but is not bit-identical on the full dataset.
Skip independent1%/5% MAE deltas versus matched plain are
+0.000686 [+0.000205,+0.001186] and +0.000602
[+0.000041,+0.001170]. This input/output skip did not improve MAE;
it does not rule out all residual architectures.

Feature result: xcorr is the clearest revived candidate. Independent1%
delta -0.001049 [-0.001553,-0.000547], independent5%
-0.002242 [-0.003011,-0.001439] versus same-seed baseline.
Bundle independent5% is 0.023244, but every paired bundle-minus-xcorr
interval includes zero; extra zcr/kurt benefits are not established.
xcorr here is the existing mean pairwise Pearson correlation across
normalized channels, NOT spatial modal shapes or frequency coherence.
Changing input count also changes neural initialization; these single-seed
feature gains need seed validation before declaring a robust winner.

Tails do not improve uniformly: independent5% p99 baseline 0.11865,
larger 0.12239, xcorr 0.12063, bundle 0.11711. Independent1% p99
baseline 0.09120, xcorr 0.08530, bundle 0.08257. Skip p99
0.08715/0.11711 versus matched plain 0.09145/0.12070, but with
worse MAE. These are descriptive clip-weighted signed-error quantiles.
All paired MAE intervals resample fixed OOF design errors, omit refit
uncertainty and multiplicity correction, and follow repeated exploration.

Decision: retain small 32->16 network; larger capacity and this skip path
do not justify replacement. Prioritize +xcorr as the smallest promising
feature addition; retain bundle as comparator, not established improvement
over xcorr. Next useful check is feature-arm seed stability, not reserved
test scoring. No production model default was replaced.

Evidence: p4_groups/mlp_validation_20260929/results.json (fits,
convergence, predictions, conditions, paired baseline comparisons),
analysis.json (seed spread, skip/matched and bundle/xcorr comparisons,
source/input hashes), extra_features.npz (all five noise conditions,
source rows, design IDs and feature names).
24 focused tests passed, including finite-difference gradients with
nonzero skip weights and L2, plus existing physics/scaling/modal/sensor
contracts. The actual consolidated CLI completed all arms; no reserved
selection/calibration/test designs were scored. Runner and reusable skip
objective remain in the repository; no duplicate validation script remains.

2026-09-29: quick feature-seed confirmation completed. Ran only missing
seeds43/44 for +xcorr and +zcr+xcorr+kurt: 20 outer-fold fits, all
converged. Reused saved seed42 feature arms and all three baselines;
cached selected16 features, original design folds, alpha1, 32->16,
noise mixture and independent-noise evaluation unchanged. Used existing
fit_network/score helpers in an evaluation cell, no new script or model
source changes. No reserved designs scored.

Mean per-seed design MAE (NOT predictions averaged into an ensemble):
| Condition | Baseline | +xcorr | +all three |
|---|---:|---:|---:|
| Clean | 0.017183 | 0.016512 | 0.016407 |
| 1% | 0.019346 | 0.018624 | 0.018488 |
| 5% stress | 0.025043 | 0.023338 | 0.023260 |
| Independent1% | 0.019715 | 0.018737 | 0.018600 |
| Independent5% | 0.025369 | 0.023507 | 0.023163 |

Correlation improves MAE versus the same-seed baseline in every condition
for all three seeds. Bundle also improves baseline in every case, but
does not consistently beat correlation alone (original5% seed44 loses).
Every paired bundle-minus-correlation interval on seed-averaged per-design
losses includes zero; independent5% delta -0.000345 with interval
[-0.000837,+0.000071]. These bootstrap fixed OOF design losses, not
refits or seed uncertainty; no multiplicity correction.

Mean of per-seed p99 signed errors, independent1%:
baseline 0.09207, xcorr 0.08656, bundle 0.08267;
independent5%: 0.12007, 0.11809, 0.11458.
These are descriptive averages of quantiles, not an ensemble p99 or
calibrated safety guarantee. Some individual-seed tails still worsen.
Decision: establish +xcorr as the smallest validated-across-seeds working
candidate. Retain bundle as a tail/accuracy challenger, not a proven
necessary addition. No default fitted model replaced.
Evidence: p4_groups/mlp_feature_seeds_20260929/results.json stores new
fits/convergence, predictions, paired same-seed comparisons, reused seed42
records, three-seed summaries and source/parent hashes. No extraction or
baseline reruns, no new permanent tests for this numerical investigation.

2026-09-29: user approved a focused physical mismatch audit before adding
physics constraints. Prespecified acquisition-only probes around existing
independent seed929 1% noise control:
- Residual calibration scale error: balanced fixed channel gains0.9/1.1
  (seed930 permutation), after causal normalization and before clipping.
  This is NOT pre-calibration hardware gain, which normalization can cancel.
- Correlated noise: sqrt(.2)*independent seed929 noise + sqrt(.8)*shared
  seed931 Gaussian noise, same per-channel marginal sigma as control,
  applied before normalization. Cross-channel noise correlation rho=.8.
- Reported dt scaled .98 or1.02 with measured samples/targets unchanged.
All settings are sensitivity probes, not measured hardware tolerances.
No generator/structural dynamics mismatch or operating-order constraint
was tested. Protocol was saved before scoring.

Used same 4454 clips /77 training designs. Compared ridge and MLP on
identical 41 physics features and on42 features including xcorr.
MLP32->16 alpha1 seed42; ridge42 separately nested-tuned on unperturbed
equal-noise-mixture training data. Ridge41 reused its existing selected
alphas. All models fit only unperturbed original mixture and stayed frozen
across conditions; original outer design folds retained.

Mean design MAE (identical42-feature comparison):
| Probe | Ridge42 | MLP42 |
|---|---:|---:|
| Control independent1% | 0.029765 | 0.018770 |
| Residual gain bias | 0.030100 | 0.019044 |
| Correlated noise | 0.031254 | 0.020066 |
| dt -2% | 0.029716 | 0.018753 |
| dt +2% | 0.029888 | 0.018802 |
MLP-minus-ridge paired design-bootstrap intervals exclude zero throughout.
MLP advantage survives these probes; this is not deployment validation.

Strongest tested weakness: MLP42 common-noise MAE worsens +0.001296,
95% interval [+0.000758,+0.001823]. Signed-error p99 grows
0.08530 -> 0.10692; paired whole-design bootstrap delta +0.021620,
95% interval [+0.009721,+0.035069] (2000 resamples).
Calibration bias p99 0.09279; dt-2%/dt+2% p99 0.08503/0.08473.
Common-noise MLP41 without xcorr has MAE0.020680 and p99 0.09935:
xcorr still helps MAE under this shift but no longer wins on p99.
Ridge42 common-noise p99 is0.12645.

Attribution, not a deployable intervention: in frozen MLP42, restore only
xcorr to its paired independent-noise control value while retaining all
other correlated-noise features. MAE0.019992/p990.10604, little better
than actual0.020066/0.10692. Therefore the changed xcorr scalar alone
does not explain the degradation; do not simply drop that feature.
Other feature shifts or interactions remain involved; no claim that one
specific modal quantity has been isolated. Median xcorr changes
0.23438 ->0.23810; median fit error0.004934 ->0.005104. All conditions
return nonempty estimates (not a reliability guarantee).

Recommendation: target measurement-noise robustness, not a new universal
margin-versus-damping monotonicity law. First compare training with
correlated-noise augmentation; only then test whether a paired prediction
consistency penalty adds value beyond that augmentation. Both noise views
share the same physical target, making this a justified nuisance-invariance
candidate. Preserve supervised target loss and report accuracy/tail costs.
No such training change was made in this audit. Timebase/calibration probes
do not currently justify separate constraints. No evidence here about
within-design velocity ordering, so that question remains open.

Verification: control42 feature rows replay cached independent-noise
features at atol=rtol=1e-10; existing ridge41/MLP41/MLP42 control
predictions replay at atol1e-10. All MLP fits converged. Actual perturbations
ran through normalization, clipping, feature extraction and frozen fold
prediction. Saved counterfactual predictions require an unavailable paired
control and are diagnostic only. No source changes or permanent tests.
Evidence in p4_groups/physical_mismatch_20260929/: protocol.json,
features.npz and results.json retain perturbations, gain vector, row IDs,
predictions, folds, ridge42 tuning, diagnostics, source/input hashes and
paired uncertainty. No reserved selection/calibration/test scoring.
Limits: one gain pattern, one shared-noise realization, one MLP seed;
fixed-OOF design resampling, no refit uncertainty or multiplicity correction.

2026-09-29: completed augmentation versus consistency comparison.
User requested parallel experiment execution, then explicitly prohibited
delegating physics/scientific reasoning. Delegation was stopped; no
worker-produced fit results were used. Both experiment processes were
launched directly with one numerical thread each; interpretation performed
directly. Physical assumptions, fixed controls and loss were specified
before dispatch/execution. No reserved-design scoring.

Protocol: same77 training designs/4454 clips, 42 features (symmetric physics
+xcorr), MLP32->16 alpha1 seed42, same5 outer design folds seed42.
Original training uses clean/iid1%/iid5% views. Volume-matched controls add
a fourth1% view: either independent seed929 or correlated sqrt(.2)*seed929
+sqrt(.8)*shared seed931 noise. All four views equally weighted.
This also controls the changed alpha/n_rows regularization relative to the
original3-view baseline. Correlated-noise settings are sensitivity probes,
not hardware-derived specifications.

Consistency uses exactly the correlated4 training inputs, with matched
custom-optimizer lambda0 and lambda1 arms. Fixed loss:
0.5 mean((prediction-target)^2) + alpha/(2*n_rows) sum(weight^2)
+ lambda/2 mean((prediction[iid1]-prediction[correlated1])^2).
Pairs share source index, design ID and target, asserted in every fold.
Scaling uses training rows only. No lambda/architecture search.
New independent seed932 and correlated seed932+shared933 features are used
only for evaluation. All copies of a physical design stay within one fold.
Eleven evaluation conditions retain original clean/1%/5%, independent
seed9291%/5%, gain/timebase probes and original/fresh correlated draws.

Results, mean design MAE; last column is signed-error p99:
| Arm | Clean | Fresh iid1% | Fresh correlated1% | Correlated p99 |
|---|---:|---:|---:|---:|
| Original3 | .016518 | .018586 | .020102 | .105704 |
| Extra independent4 | .016037 | .017890 | .019366 | .105963 |
| Correlated4 | .016612 | .018139 | .019039 | .096549 |
| Correlated4 custom lambda0 | .016343 | .017931 | .019095 | .097204 |
| Correlated4 custom lambda1 | .017253 | .018486 | .019069 | .082541 |

Augmentation versus original3 improves fresh-correlated MAE by .001063
(paired design95% interval[-.001505,-.000623]) and p99 by .009155
([- .017567,-.003524]). Clean MAE change+.000093 has interval including0.
But matched extra-independent4 control matters: correlated4 improves MAE
only .000327 (interval[-.000908,+.000217], uncertain), while p99 improves
.009414 ([-.022074,-.000259]). It worsens clean MAE+.000574
([+.000195,+.000968]). Do not attribute the whole gain over original3 to
the correlated-noise distribution rather than extra training views.

Consistency versus its matched custom lambda0 control leaves correlated
MAE essentially unchanged (-.000025, interval includes0) but reduces p99
.097204 ->.082541 (~15%); delta95%[-.038056,-.002038].
Fresh iid1% p99 .082506 ->.076949; seed929 iid5% p99 .119780 ->.101335.
Both exploratory tail intervals exclude0. Clean MAE worsens+.000910
([+.000350,+.001486], ~5.6%). Fresh iid1% MAE+.000555 interval narrowly
includes0. Paired fresh iid/correlated prediction absolute gap improves
.008059 ->.006261 (~22%), confirming reduced measurement sensitivity.

Safety caveat: p99 improvement does not imply uniformly better decisions.
Fresh correlated noise, custom lambda0 ->lambda1:
- Mean signed error .000735 ->.000337 (not simply a large constant offset).
- Lower-tail p01 -.103631 ->-.113653: more severe underestimation.
- Unstable clips predicted positive1.576% ->2.120%.
- Stable clips predicted nonpositive1.148% ->.956%.
At independent5%, unstable-positive proxy1.957% ->2.935%.
These are descriptive clip-level zero-threshold proxies, not operational
missed-event/false-alarm rates or statistically established rate changes.
No thresholds were tuned.

Decision: retain correlated augmentation as a robustness candidate, not
a universal default replacement; matched independent augmentation remains
the clean-accuracy comparator. Do NOT adopt lambda1 consistency by default:
it trades clean accuracy for measurement consistency/upper-tail reduction,
and does not uniformly improve zero-margin decisions. No broad penalty
sweep or claim of solved safety. Before adoption, establish this tradeoff
across model seeds and connect noise assumptions/decision costs to the
intended acquisition and warning protocol.

Implementation: existing skip_mlp.objective_setup now optionally accepts
consistency_pairs and consistency_weight, preserving zero-weight defaults.
fit_network exposes these only on its custom path and rejects silent use
on the native path. Pair gradients accumulate repeated row endpoints.
All25 outer fits converged. Original3 predictions replay previous physical
audit controls within1e-10. Actual extraction and all11-condition outer-fold
predictions exercised. 31 focused tests passed, including analytic versus
finite-difference paired gradients (plain/skip, repeated endpoints),
pair-loss normalization and invalid constraints, plus existing feature,
scaling and sensor contracts.

Evidence: p4_groups/correlated_training_20260929/ contains protocol.json,
shared.npz, augmentation_results.json, consistency_results.json,
analysis.json, tail_diagnostics.json and replayable experiment scripts.
Results retain predictions, folds, convergence and input/source hashes.
MAE uses10000 paired design resamples; p99 uses2000; fixed OOF, no refits
or multiplicity correction. One MLP seed and one new paired noise draw.
No fitted-model/default adoption, feature re-selection or reserved
selection/calibration/test scoring.

2026-09-29: user asked about dataset balance, then approved final bounded
experimentation. Metadata audit found current training2614 stable/1840
unstable clips (58.7%/41.3%), with918 stable and920 unstable within5% of
flutter. Sampler uses five equal-count velocity strata, not equal-width
margin bins: training margins[-.15,-.05),[-.05,0),[0,.05),[.05,.15),
[.15,.32] contain920/920/918/890/806 clips. Only334 clips lie within1%,
720 within2%; our old abs(m)<.15 slice contains81.9% of training clips.
65/77 designs contribute60 clips; other designs34-58. Noise copies do not
add physical cases. Full source retained106/160 designs;54 exclusions:
28 already unstable at lower modeled velocity,26 boundary not found.
This acceptance filter is a stronger population caveat than the modest
class imbalance. No model scores on reserved data were inspected.

Completed seeds43 and44 for extra_independent4 and correlated4, reusing
seed42. All20 new fits converged. Same42 features,32->16 network, alpha1,
five design folds seed42 and fixed noise draws; no retuning, new extractor
or consistency sweep. Ran directly, no delegation. Existing artifact
runner now takes --seed/--arms/--output and refuses to overwrite evidence.
Actual CLI exercised both new seeds and all11 evaluation conditions.

Three-seed mean design MAE (not predictions averaged into an ensemble):
| Evaluation | Extra independent4 | Correlated4 |
|---|---:|---:|
| Clean | .016045 | .016586 |
| Fresh independent1% | .017869 | .018149 |
| Fresh correlated1% | .019424 | .019338 |
| Independent5%, seed929 | .023146 | .023456 |
Independent augmentation wins clean and fresh independent1% MAE in all
three seeds. Correlated augmentation wins correlated MAE in seeds42/44,
loses in43; averaged difference-.000086,95%[-.000474,+.000292].
Clean cost+.000541,[+.000227,+.000848]; fresh iid1% cost+.000280,
[+.0000046,+.000560]. Tiny lower bound on the latter is not strong
multiple-comparison evidence.

Mean per-seed p99 signed error (again, not an ensemble):
| Evaluation | Extra independent4 | Correlated4 |
|---|---:|---:|
| Clean | .073078 | .069551 |
| Fresh independent1% | .087730 | .082350 |
| Fresh correlated1% | .105133 | .096077 |
| Independent5%, seed929 | .121383 | .117557 |
Correlated-noise tail improvement repeats in all3 seeds; averaged
delta-.009056, paired design95%[-.017076,-.003809], about8.6% lower.
Other listed p99 intervals include0. At iid5%, correlated augmentation
actually loses on p99 in seed44: no uniform stress-noise safety benefit.

New tight boundary report computes each side separately, with design-MAE
averaged only across designs represented in that slice, plus clip MAE,
signed bias, p99 and zero-threshold wrong-side fractions. Same clip
denominators across seeds; repeats are not independent observations.
Slice counts (stable/unstable):
- abs(m)<.01:172/162 clips,58/51 designs.
- abs(m)<.02:358/362 clips,74/73 designs.
- abs(m)<.05:918/920 clips,77/77 designs.
Extra independent4 under fresh iid1%, mean per-seed wrong-side fractions:
| Distance from flutter | Stable predicted nonpositive | Unstable predicted positive |
|---|---:|---:|
| <1% | 15.31% | 18.31% |
| <2% | 7.82% | 8.66% |
| <5% | 3.09% | 3.41% |
Correlated4 at<1%:18.02%/15.43%, a tradeoff rather than uniform improvement.
Under fresh correlated1%, the<1% fractions are14.73%/20.16% for
independent4 and16.67%/17.90% for correlated4.
These are descriptive point-estimate classification proxies, not warning
lead times or operational event rates, and have no paired rate intervals.

Important asymmetry: for independent4 under fresh iid1%, within5% of
flutter stable-side design MAE .014663 versus unstable-side .003744.
Stable-side mean signed error+.009549 versus unstable-side-.000512.
Thus the balanced local counts do not imply symmetric performance;
pre-flutter margin overestimation deserves separate reporting. No causal
explanation established here. Strong aggregate accuracy is not proof of a
reliable sign decision in the final1% before/after flutter.

Candidate decision for the accuracy-led point-estimation method: retain
small32->16 MLP with symmetric physics+xcorr and independent-noise
augmentation; use the four-view independent candidate from this comparison.
Keep correlated augmentation as a robustness ablation/alternative when
correlated-noise upper tails matter. Do not adopt lambda1 consistency as a
default. Do not select a lucky seed or launch another penalty/architecture
search. This concludes the bounded augmentation experiment, not deployment
validation or a final reserved-test evaluation. No fitted default replaced.
Before warning claims: specify warning output/costs, audit calibration/split
provenance, and assess decision-level performance; trajectories are required
for lead time. No calibration/test data were consumed in this confirmation.

Evidence in p4_groups/correlated_training_20260929/:
augmentation_seed43.json, augmentation_seed44.json,
seed_confirmation_analysis.json, analyze_seed_confirmation.py, and updated
run_augmentation_experiment.py (seed42 source hash preserves its earlier
version identity). Analysis verifies row/fold identity, finite predictions
and train/test design separation. Three-seed paired MAE bootstrap averages
losses within each design first; p99 bootstrap resamples whole designs with
shared draws then averages seed quantiles.10000 MAE/2000 p99 resamples;
fixed predictions, no retraining or multiplicity correction. Initialization
repeats do not add designs or new acquisition-noise draws. No new permanent
tests for this numerical investigation; actual execution is the check.

2026-09-29: user requested adding focused conditional overestimation scores
and current results. Boundary MAE/bias already existed; exceedance curves
and safer-case underestimation did not. Added reusable error_exceedance
to decision_metrics.py and integrated it into analyze_seed_confirmation.py.
No retraining, threshold optimization or reserved-design scoring.

Definition: e=prediction-truth. Overestimation curve P(e>delta | 0<m<eps)
for eps .01/.02/.05; delta grid0/.005/.01/.02/.03/.05. Strict inequalities.
Guardrail: P(-e>delta | m>=.15), i.e. underestimation on806 clips/69
designs at least15% below flutter. This is a descriptive safer slice,
not a certified safe operating envelope or operational false-warning rate.
Rates are clip-weighted, averaged over seed42/43/44; not predictions
ensembled before thresholding. Preserve per-seed counts/rates and slice
clip/design counts.2000 bootstrap draws of represented designs retain
their clips and all seeds together. Fixed OOF, no refit/seed uncertainty
or multiplicity correction; zero-event bootstrap bounds do not establish
zero population risk.

Primary candidate: independent4 augmentation, fresh iid1% evaluation.
Thresholds below are absolute margin errors, not fractions of true margin.
| True stable margin | Clips/designs | e>.01 | e>.02 | e>.03 | e>.05 |
|---|---:|---:|---:|---:|---:|
| (0,.01) | 172/58 | 15.31% | 3.29% | .97% | 0.00% |
| (0,.02) | 358/74 | 21.23% | 6.52% | 3.54% | 1.30% |
| (0,.05) | 918/77 | 31.48% | 16.74% | 11.22% | 4.90% |
Selected95% design-bootstrap intervals:
- eps.01,delta.01:9.84-21.69%.
- eps.02,delta.01:16.66-26.03%.
- eps.05,delta.01:26.31-36.77%.
- eps.05,delta.03:8.33-14.23%.
- eps.05,delta.05:3.09-6.95%.

At eps.05, mean signed error+.009549, mean seed-median+.004432,
66.88% of predictions overestimate margin. Thus positive bias is not
restricted to a handful of extreme outliers, though tail failures coexist.
Near eps.01 mean/median+.003651/+.002372; eps.02+.005724/+.003922.
These are conditional descriptive biases, not identified physical causes.

Noise sensitivity, same independent4 candidate at eps.05:
| Evaluation | e>.01 | e>.03 | e>.05 | Mean signed error |
|---|---:|---:|---:|---:|
| Clean | 29.01% | 7.26% | 1.85% | +.006694 |
| Fresh iid1% | 31.48% | 11.22% | 4.90% | +.009549 |
| Fresh correlated1% | 36.64% | 12.85% | 7.55% | +.012719 |
| Independent5% stress | 45.32% | 18.12% | 10.60% | +.017684 |
Correlated4 under fresh correlated1% at eps.05 has e>.01/.03/.05
35.88%/12.64%/6.90%, but e>.02 is19.93% versus19.75% independent4:
lower global p99 does not imply uniform improvement along the local curve.
No paired curve-improvement significance claim was evaluated.

Conservatism guardrail, independent4 fresh iid1%, m>=.15:
Underestimate by>.01:55.87%,95%[48.13,63.64].
Underestimate by>.03:37.68%,95%[30.42,45.01].
Underestimate by>.05:24.44%,95%[18.39,31.08].
Mean signed error-.020495, mean seed-median-.016843. Opposite bias
across margin regions argues against presenting a constant downward
prediction shift as a free correction for near-flutter optimism.
Whole-dataset design MAE remains .017869 on fresh iid1% and .016045 clean.

All11 conditions and both augmentation candidates are saved under
conditional_error_risk in
p4_groups/correlated_training_20260929/seed_confirmation_analysis.json.
Existing boundary scores/MAE remain intact. Metric source hash included.
Verification:25 decision-metric tests passed; new tests cover strict
thresholds, seed rates versus ensembling, underestimation direction,
unequal-size design clusters, repeated-clip interval invariance, and
invalid/nonfinite/empty input. Actual analysis CLI exercised all saved
predictions and generated the extended result artifact.

2026-09-29: discussed whether conditional bias calls for a new loss, more
data or features. Clean-signal failures rule out noise as the sole cause,
not establish the alternative. Near-flutter overestimation plus farther
stable-side underestimation is consistent with predictions pulled toward
the target-range middle; cause remains unproven.
Next: compare training versus held-out-design conditional errors, then
inspect feature neighborhoods/modal diagnostics of severe optimistic
failures. Use evidence to distinguish fitting bias, limited design
diversity and feature/identification ambiguity before choosing an
intervention. More noise copies do not add physical designs; a loss cannot
recover missing information. A locally weighted/asymmetric loss remains
a candidate, with safer-case underestimation and MAE guardrails; the prior
global asymmetric-loss experiment already showed a conservatism tradeoff.
A separate calibrated warning output may manage uncertainty, not repair
the point estimate. No new experiment run or method change decided.

2026-09-29: completed the conditional-bias diagnosis agreed above. User
clarified GRU remains a live path: both earlier approaches were feature-starved;
lighter models are the current experimental vehicle, not a decision to abandon
GRU. Return to it with the improved representation rather than defend or bury it.

Prespecified protocol in p4_groups/conditional_bias_diagnosis_20260929/:
same selected16, symmetric physics+xcorr42, MLP32->16 alpha1, independent4
augmentation, seeds42/43/44 and original five design folds. Refit only to
obtain previously unavailable in-sample predictions. All15 fits converged;
their design-held-out predictions reproduce the saved candidate EXACTLY.
Each physical clip has four in-sample fit exposures per seed versus one OOF
prediction; average errors/rates, NOT predictions. Evaluate clean and fresh
iid1% seed932. Fresh measurement noise is unseen even for in-sample designs.
No reserved selection/calibration/test designs scored or used for fitting.

Whole-dataset mean design MAE, train -> design-held-out:
clean .012279 -> .016045; fresh1% .014131 -> .017869.
Within stable0<m<.05 (918 clips/77 designs):
| Condition | Train design MAE | OOF design MAE | Train bias | OOF bias | Train e>.03 | OOF e>.03 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | .009403 | .011353 | +.005555 | +.006694 | 4.98% | 7.26% |
| Fresh iid1% | .012087 | .014663 | +.007808 | +.009549 | 8.44% | 11.22% |
Fresh1% OOF-minus-train bias+.001740, paired95% [.000559,.003220];
design-MAE gap+.002576 [.001547,.004012]. Generalization adds error, but
near-stable optimism is already present in converged in-sample fitting.
Within unstable-.05<m<0, fresh1% train/OOF design MAE .003260/.003744
and bias-.000436/-.000512. At m>=.15, bias-.015547/-.020495 and
design MAE .029172/.038442: opposite regional bias also exists in training.
These are descriptive in-sample comparisons, not estimates of deployment
accuracy. Intervals use2000 paired whole-design resamples of fixed prediction
metrics, no model refits, seed uncertainty or multiplicity correction.

Defined severe optimism as stable0<m<.05 and e>.03 in at least2/3 saved
seed predictions:64 clean clips/26 designs;103 fresh1% clips/38 designs;
37 cases overlap. Post-hoc waveform audit reuses existing frequency matching
(<=max(2 Hz,5%)) and solver poles only as diagnostic references.
All64 clean severe cases recover ALL8 modes, including the critical mode;
median critical-alpha error6.32e-7/s. Critical mode is >=1%-energy visible
in61/64, and max-visible growth usually reproduces its true growth. Its
energy-top3 inclusion is only56.25%, so summaries may omit other details,
but failed clean identification or the visibility threshold is not a general
explanation. The diagnostic does not prove current features are sufficient.
Fresh1% severe/nonsevere-stable critical recovery88.35%/95.71%;
matched critical-alpha median error2.333/.243/s, with12/35 unmatched
critical modes reported explicitly. Noise adds identification fragility.
These unmatched outcome-selected cohorts also differ in true margin; quality
associations are not causal findings or a validated reliability gate.

Training-only four-view scaling defines feature distance; each OOF query
uses10 nearest distinct physical clips from outer-training designs at the
same acquisition condition, without noise-copy duplication. Clean severe/
other-stable nearest RMS-standardized distance medians .297/.298, but k3
neighbor-target SD medians .0289/.0109 and neighbor-median-minus-query
margin .0280/.0038. Fresh1% offsets .0489/.0046. Removing explicit modal
quality columns leaves the broad pattern intact. Thus errors are associated
with neighborhoods containing larger/conflicting margins, not simply large
global feature distances. Finite-distance neighbors are NOT exact feature
collisions or proof of an information ceiling. Example source796:
true m=.041345; three clean OOF errors+.1272/+.1310/+.1246; in-sample mean
error+.0950, while every pole is accurately recovered. Its three closest
training-design neighbors have margins .2770/.1846/.2192.

Decision: neither more noise copies nor a universal margin-damping constraint
addresses the clean finding. Current evidence leaves loss/regularization
fitting bias versus omitted system/modal/spatial context unresolved; limited
design diversity also contributes to the OOF gap. Next smallest controlled
probe: locally weighted supervised fitting with unchanged features/network,
a matched unweighted control, and both train/OOF conditional scores. Keep
whole-design MAE, unstable-side errors and m>=.15 underestimation as guardrails.
This tests recoverable fitting bias, not a claim that weighting restores
missing information; no loss/model/default change was made in this diagnosis.

Evidence: protocol.json, capture_training_predictions.py,15 atomic resumable
fit shards, predictions.npz, fits.json, analyze_diagnosis.py, diagnostics.npz
and results.json in p4_groups/conditional_bias_diagnosis_20260929/. Results
retain conditional curves/gaps, case identities, neighbor distances/labels,
margin distributions, modal recovery/missing estimates and input/source hashes.
Actual CLIs exercised15 fits, completed-fit resume without retraining, all4454
clips at both acquisition conditions, and every neighborhood/conditional slice.
Cached modal features replay at atol=rtol1e-10. A numerical smoke checked
unequal design-size weighting, prediction-exposure averaging, strict exceedance
and zero paired gaps. No new permanent tests or production source changes.

2026-09-30: completed the fixed local-weighting probe. Protocol recorded
before fitting in p4_groups/local_weighting_20260929/protocol.json. Same
selected16,42 symmetric physics+xcorr features,32->16 MLP, alpha1,
independent4 training views, original five design folds and seeds42/43/44.
Raw training weight2 for0<m<.05, otherwise1, normalized to mean1 within
each outer training partition. All four views of a physical clip receive
the same weight. Native sklearn divides supervised loss and L2 by sum of
weights; preserving that sum preserves alpha's regularization denominator.
Unweighted train-only feature scaling/inactive masks stay identical.
No architecture/weight/alpha sweep, new extraction or reserved-split scoring.

Matched control uses explicit all1 weights on the same native fitting path.
All30 fits converged without warnings. Explicit-unit predictions reproduce
the historical candidate EXACTLY across all11 cached acquisition conditions,
so a changed control optimizer trajectory does not explain this comparison.
Captured four in-sample fit exposures per physical clip per seed and one
design-held-out prediction. Metrics/rates, not predictions, average across
seeds/exposures. Fresh-noise measurements remain unseen on training designs.

OOF results below, unit control -> local2. Near-stable means0<m<.05;
design MAE is averaged over represented physical designs, rates clip-weighted.
| Condition | Whole design MAE | Near-stable design MAE | Near bias | Near e>.03 |
|---|---:|---:|---:|---:|
| Clean | .016045 -> .016219 | .011353 -> .009874 | +.006694 -> +.005034 | 7.26% -> 4.76% |
| Fresh iid1% | .017869 -> .017686 | .014663 -> .012813 | +.009549 -> +.007498 | 11.22% -> 7.84% |
| Fresh correlated1% | .019424 -> .019018 | .017341 -> .015328 | +.012719 -> +.010619 | 12.85% -> 10.82% |
| Independent5% stress | .023146 -> .022923 | .022147 -> .019395 | +.017684 -> +.013811 | 18.12% -> 14.96% |
Near bias and e>.03 improve in every seed in each listed condition.
Fresh1% near design-MAE delta-.001849, paired95%[-.002473,-.001281];
bias delta-.002051[-.002661,-.001461]; e>.03 rate delta-3.38 percentage
points[-4.62,-2.25]. Near e>.05 falls4.90% ->4.14%, delta95%[-1.31,-.25]
percentage points. Whole-design MAE changes are small; all four listed
whole-MAE delta intervals include0. Do not call this an aggregate-MAE win.
In-sample near bias also improves: clean+.005555 ->+.003908;
fresh1%+.007808 ->+.005880. Residual optimism remains substantial.

Guardrail cost at m>=.15, fresh1%: design MAE .038442 ->.039412,
delta+.000970,95%[+.000138,+.001808]. Mean bias-.020495 ->-.022661.
Underestimation by>.05 rises24.44% ->26.18%, delta+1.74 percentage points,
95%[+.24,+3.30]. Clean guardrail MAE .033860 ->.035446 and under>.05
17.91% ->19.64%. Stronger pessimism farther from flutter is a real cost;
it is not an operational false-warning rate or proof of unusability.
Fresh1% unstable-.05<m<0 MAE .003744 ->.003714 has a delta interval
including0. Mean wrong-side proxies within1% improve on fresh1%:
stable15.31% ->12.02%, unstable18.31% ->17.08%; no paired rate intervals
were computed for these zero-threshold proxies. Not uniformly better:
5% stress stable-within1% wrong-side rate12.79% ->14.15%, despite better
local MAE/optimistic tails. No calibrated warning or lead-time claim.

Whole-error mean per-seed p99 also falls: fresh1%.08773 ->.08056;
fresh correlated1%.10513 ->.09571; independent5%.12138 ->.10641.
These are descriptive averages of seed quantiles, not ensemble quantiles
or significance claims. All paired intervals use2000 whole-design resamples
of fixed predictions, omitting refit/seed uncertainty and multiplicity
correction after repeated exploratory work.

Decision: demonstrate a recoverable fitting-bias component using the SAME
representation, but not a free correction or proof that representation is
sufficient. Keep local2 as a bounded risk-focused candidate/ablation;
retain the existing unweighted point-estimation reference. Do not adopt
local2 universally or sweep larger weights: the farther-stable guardrail
already worsens and tight-boundary stress decisions are not uniformly better.
Further representation/system-context work remains justified alongside
explicit warning-cost/calibration work. GRU stays live for a later improved-
representation comparison; this experiment neither tests nor rejects it.
No fitted model/default, reserved selection/calibration/test use or sensor
configuration changed.

Implementation: fit_network accepts optional native sample_weight, default
None; custom fitting rejects supplied weights instead of silently ignoring
them. Existing callers intentionally retain their old default behavior.
Analytic runtime smoke: constant inputs predict weighted mean+.5 versus
unweighted0, all1 agrees, unsupported custom weighting raises.37 existing
focused fitting/scaling/decision tests passed. Actual experiment and analysis
CLIs exercised all30 fits/all11 conditions; completed-fit resume reused every
atomic shard without retraining. Final numerical replay checked normalized
2:1 weights, exact historical control, conditional rates, exposure counts,
physical-row alignment and result/source hashes. No new permanent tests.

Evidence in p4_groups/local_weighting_20260929/: protocol.json,
run_experiment.py,30 atomic fit shards, predictions.npz, fits.json,
analyze_results.py and results.json retain settings, convergence, train/OOF
predictions, conditional curves, paired differences and guardrails.

2026-09-30: completed the modal spatial-pattern ablation approved by the user.
The hypothesis was that existing pooled summaries discard useful system
context, not that this had already been proven or that GRU was dead.
Protocol frozen before extraction/fitting in
p4_groups/spatial_modal_20260930/protocol.json. Same selected16, per-channel
causal51-sample calibration, cap50, lag64, eight-mode budget, original folds,
independent4 equal training views,32->16 MLP, alpha1 and seeds42/43/44.
No structural regeneration, solver-selected predictor mode or reserved
selection/calibration/test scoring.

estimate_modes optionally exposes fitted per-mode complex sensor coefficients
(cosine minus i*sine), in existing energy order. Default callers do not
allocate these arrays. modal_spatial_row selects the largest estimated growth
rate among >=1%-fitted-energy modes, frequency tie-break as in stability
features.65 added columns:16 normalized complex-shape Gramian diagonal powers,
real/imaginary entries for24 fixed nearest-neighbor pairs in the selected4x4
sensor grid, and availability. Entries b_i*conj(b_j)/sum(abs(b)^2) are invariant
to global complex excitation scale/phase, avoid angle wrapping and do not
divide by weak channel amplitudes. Fixed channel/pair identities retain
spatial association. These are patterns of CALIBRATED responses; they do not
restore physical amplitude ratios erased by per-channel normalization.

Replayed all4454 physical training clips at all11 cached acquisition
conditions from retained raw signals, without reading solver poles for feature
selection. Original modal summaries agree within7.60e-13. Reported-dt changes
reuse the same spatial pattern (discrete shape coefficients are timebase-
invariant); scaled pole summaries also replay the corresponding cached rows.
All77 extraction shards are atomic/source-fingerprinted and resumable.
Shuffled-spatial negative control has the same107 inputs/first-layer parameter
count as the real-spatial model: permute the spatial block jointly by physical
source row within outer training/held-out partitions separately, reuse that
mapping across acquisition views, fixed seed934+fold independent of labels
and MLP seeds. No partition crossing or noise-copy duplication. This sham is
not deployable and may create physically inconsistent feature combinations;
its failure alone would not establish useful information. Real spatial must
also beat the original42-feature baseline.

All30 new fits (real/sham x3 seeds x5 folds) converged without warnings.
Baseline and local2 reference predictions are reused, not refit/reselected.
All rows below are mean design MAE, averaging seed metrics, NOT ensembles.
One mixed-trained predictor per fold/arm evaluates every condition; these are
not noise-specialized fits or clean-trained/noisy-transfer results.
| Evaluation | Baseline42 | Spatial107 | Shuffled107 | Local2 reference42 |
|---|---:|---:|---:|---:|
| Clean | .016045 | .008587 | .020287 | .016219 |
| Fresh iid1% | .017869 | .010131 | .022329 | .017686 |
| Fresh correlated1% | .019424 | .011031 | .023675 | .019018 |
| Independent5% stress | .023146 | .016927 | .028603 | .022923 |
Spatial beats baseline in every seed and every outer fold at ALL11
acquisition conditions. Fresh1% per-seed MAE .009984/.010232/.010177;
improves77/77/76 designs. Clean improves77/75/75; independent5%69/68/67.
Fresh1% spatial-minus-baseline design MAE-.007738, paired95%
[-.008819,-.006740]; spatial-minus-sham-.012198[-.013370,-.011013];
spatial-minus-local2-.007555[-.008538,-.006602]. Shape information is useful
under this protocol, not merely a wider input or an isolated lucky seed.
This does not establish a sufficient representation or an information ceiling.

Near-stable0<m<.05 OOF, baseline -> spatial:
| Evaluation | Design MAE | Mean signed bias | e>.03 | e>.05 |
|---|---:|---:|---:|---:|
| Clean | .011353 -> .005599 | +.006694 -> +.001044 | 7.26% -> .73% | 1.85% -> .33% |
| Fresh iid1% | .014663 -> .007880 | +.009549 -> +.003762 | 11.22% -> 3.89% | 4.90% -> 1.78% |
| Fresh correlated1% | .017341 -> .009678 | +.012719 -> +.005520 | 12.85% -> 5.45% | 7.55% -> 2.80% |
| Independent5% stress | .022147 -> .014020 | +.017684 -> +.008049 | 18.12% -> 9.66% | 10.60% -> 5.66% |
Fresh1% near MAE delta-.006783,95%[-.008676,-.005146]; e>.03 delta
-7.33 percentage points[-9.81,-4.91]. Training-design near bias also drops:
clean+.005555 ->+.000484; fresh1%+.007808 ->+.002753.
Of64 previously severe clean cases (e>.03 in>=2/3 seeds), NONE remain
severe, but4 new severe cases appear. Fresh1%103 ->35 severe cases:
23 retained,12 new. These outcome-selected cohort counts are descriptive,
not independent validation or a guarantee that each case was repaired.
Source796 (m=.041345), formerly clean errors+.1272/+.1310/+.1246,
now+.00765/-.00278/+.00885.

Unlike local weighting, the farther-stable guardrail ALSO improves.
At m>=.15, fresh1% design MAE .038442 ->.021637, underestimation by>.05
24.44% ->6.58%, delta-17.87 percentage points,95%[-23.58,-12.75].
Clean under>.05 17.91% ->3.76%; correlated1%26.30% ->6.37%;
5% stress38.96% ->20.93%. All-clip signed-error mean per-seed p99 fresh1%
.08773 ->.05696, correlated1%.10513 ->.06759; quantile comparisons are
descriptive, not significance or calibrated safety claims.

Important exception: tight zero-margin decisions are NOT uniformly better.
Stable0<m<.01 wrong-side fraction clean11.63% ->15.89%, fresh1%
15.31% ->16.28%,5% stress12.79% ->15.50%. Unstable-.01<m<0 fresh1%
18.31% ->18.11% is essentially unchanged, with substantial seed variation.
Stable last1% MAE nevertheless improves (fresh1%.005570 ->.003786).
These are descriptive clip proxies without paired rate intervals, not
operational event/false-warning rates. Lower point-estimation error and less
optimism do not automatically certify a better warning decision.
All paired intervals are2000 whole-design fixed-prediction resamples;
no model-refit/seed uncertainty or multiplicity correction after exploration.
Acquisition noise remains uncalibrated sensitivity settings, and structures
are still the same77 training designs. All clips retained; no spatial estimates
unavailable in the four primary conditions. No new full-pipeline latency claim.

Decision: spatial patterns are the strongest current representation addition
and the unweighted107-feature small MLP is the leading working point-estimation
candidate under this protocol. Do not turn this into a deployment/default or
reserved-test claim. Freeze this feature definition; confirm acquisition-noise
draw robustness and full inference cost before further tuning or final split
use. Last1% decision uncertainty needs its own warning/calibration treatment,
not a claim that another MAE gain solved it. These features can later accompany
waveforms in a fair improved-representation GRU comparison; GRU remains live.
No local2+spatial combination or new architecture was tried.

Verification:27 focused tests pass, including recovered known complex sensor
patterns, least-stable versus highest-energy selection, phase contrast with
equal channel powers, global complex-gauge/timebase invariance, visibility
selection, explicit missing estimates and invalid pairs. Actual D000000
raw-to-spatial smoke passed. Both extraction and fitting completed-resume paths
reuse all77/30 atomic shards without re-extraction/retraining. Final evidence
smoke checked unit trace/pair magnitude identities, finite/availability fields,
all11 conditional rates,12-versus3 training/OOF exposures, identical sham
permutations across MLP seeds, design isolation and current result/source hashes.
The first analysis attempt hit a generic analyze_results import-name collision;
fixed by loading the existing weighting helper from its exact path. No fits
were rerun to address that analysis-only failure.

Evidence: p4_groups/spatial_modal_20260930/ contains protocol.json,
prepare_features.py,77 acquisition-feature shards, spatial_features.npz,
run_experiment.py,30 fit shards, predictions.npz, fits.json,
analyze_results.py and results.json. Shared opt-in extraction and its behavior
tests live in modal_features.py and tests/test_spatial_modal_features.py.

2026-09-30: completed the user-approved two-mode spatial experiment.
Question: add the second least-stable visible pattern and normalized inter-mode
similarity to the current107-feature spatial MLP, rather than densifying a
rank-one single-mode Gramian. Frozen protocol:
p4_groups/two_mode_spatial_20260930/protocol.json.

Same selected16/calibration51/cap50/lag64/max8, four equal independent training
views,32->16 ReLU MLP, alpha1, native LBFGS caps, original5 design folds and
seeds42/43/44.67 additions: second >=1%-fitted-energy visible mode's65-column
pattern/availability, squared normalized complex inner product
abs(b1^H*b2)^2/(norm(b1)^2*norm(b2)^2), and pair availability. Estimated alpha
descending, frequency ascending tie-break; no solver identity or label input.
Each mode has independent global amplitude/phase gauge; similarity is spatial
collinearity of CALIBRATED sensor patterns, not measured physical coupling.
Missing estimates have explicit flags; no clipping/dropout of difficult rows.
Shared modal_spatial_row keeps its original signature/numerical behavior;
modal_spatial_addition_row adds the second pattern/similarity.

Three arms: current spatial107, two_mode174, shuffled_addition174. Sham jointly
permutes ONLY the67 additions within outer training/held-out source-row sets
separately, reusing seed934+fold mapping across acquisition views and MLP seeds.
Original107 inputs stay aligned. Sham can create inconsistent feature
combinations and is not deployable; real gains must beat spatial107 itself.
Refitted15 current controls to obtain untouched fresh-condition predictions
and save native predictors. All45 fits converged without warnings; all prior
training and OOF predictions at11 conditions reproduced EXACTLY (delta0).
All77 extraction and45 fit shards are atomic/fingerprinted/resumable.

Replayed4454 clips at15 conditions: original11 plus iid1_fresh935,
correlated_fresh935_936, iid1_fresh937, correlated_fresh937_938. Fresh draws
use source-row SeedSequence,1% original-eight aggregate calibration RMS,
rho.8 shared Gaussian before normalization, same convention as prior audits.
Neither new draw is trained on. These are new acquisition realizations of the
SAME77 training designs, not new structures or reserved-split validation.
Original modal/spatial features replay within7.60e-13.

OOF mean design MAE; seed METRICS averaged, not predictions/ensembles:
| Evaluation | Current spatial107 | Two-mode174 | Shuffled additions174 |
|---|---:|---:|---:|
| Clean | .008587 | .007049 | .009416 |
| Prior fresh iid1%932 | .010131 | .008883 | .011171 |
| Prior fresh correlated1%932/933 | .011031 | .009705 | .012195 |
| Independent5% stress929 | .016927 | .016367 | .019514 |
| NEW iid1%935 | .010355 | .008939 | .011542 |
| NEW correlated1%935/936 | .010816 | .009657 | .012202 |
| NEW iid1%937 | .010075 | .008692 | .011087 |
| NEW correlated1%937/938 | .011065 | .009708 | .012337 |
Every condition improves mean design MAE in EACH seed. Not every outer fold
improves: all15 fold/seed combinations improve only for clean and the two
correlated conditions932/933 and935/936. NEW iid935 delta-.001416,
paired95%[-.001791,-.001065]; iid937-.001383[-.001739,-.001029];
correlated935/936-.001159[-.001528,-.000801]; correlated937/938-.001358
[-.001743,-.000964]. Both new iid draws improve design MAE in58/60/61 and
58/60/59 of77 designs across seeds; not universal per-design improvement.
Two-mode beats the equal-width sham on all conditions. Independent5% global
delta-.000560,95%[-.001627,+.000014], so that aggregate gain is uncertain.

The gain has a boundary cost. Current -> two-mode at stable0<m<.01:
| Evaluation | Design MAE | Predicted nonpositive fraction |
|---|---:|---:|
| Clean | .003674 ->.004061 | 15.89% ->18.41% |
| Prior fresh iid932 | .003786 ->.004687 | 16.28% ->20.16% |
| NEW iid935 | .004335 ->.005166 | 16.09% ->20.35% |
| NEW iid937 | .003777 ->.004844 | 16.47% ->21.71% |
| NEW correlated935/936 | .004044 ->.005025 | 13.95% ->18.22% |
| NEW correlated937/938 | .003825 ->.004611 | 12.21% ->18.99% |
Stable last1% MAE worsens with paired intervals above zero on ALL four new
draws: iid935 delta+.000831[+.000300,+.001377], iid937+.001067
[+.000542,+.001655], correlated935/936+.000981[+.000501,+.001491],
correlated937/938+.000786[+.000305,+.001312]. Wrong-side rate intervals are
less decisive: iid935+4.26pp[-.52,+8.96], iid937+5.23pp[+.96,+9.88],
correlated935/936+4.26pp[approximately0,+8.71], correlated937/938+6.78pp
[+1.59,+11.96]. Do not treat floating lower bound8.5e-18 as excluding zero.
Unstable last1% predicted-positive fractions also rise on every new draw:
18.31% ->19.96%,18.11% ->19.34%,17.90% ->21.19%,19.55% ->22.43%
(iid935/iid937/correlated935/correlated937); their paired intervals cross zero.
These are clip sign proxies, not operational false-warning/missed-event rates.

Stable0<m<.05 at new draws: design MAE improves only slightly, with all four
paired intervals crossing zero. e>.03 falls3.92% ->3.09% iid935,
3.59% ->2.87% iid937,4.83% ->4.28% correlated935,4.94% ->3.96%
correlated937; intervals also cross zero. At5% stress the boundary guardrail
clearly worsens: near5% design MAE .014020 ->.016583, delta+.002563,
95%[+.001218,+.004215]; e>.03 9.66% ->11.55%. Stable last1% MAE
.005579 ->.008193; unstable last1% wrong-sign19.75% ->27.78%, delta
+8.02pp,95%[+2.48,+14.12]. Broad/global metrics would hide these costs.

Farther-stable m>=.15 improves strongly. Underestimation by>.05:
clean3.76% ->.79%, prior fresh iid9326.58% ->3.80%;
NEW iid9356.91% ->3.43%, iid9376.82% ->3.06%, correlated9356.91% ->4.43%,
correlated9377.32% ->4.80%. Prior iid932 farther-stable design MAE
.021637 ->.016679; NEW iid935 .022213 ->.016507, iid937 .021457 ->.015499.
All-clip signed-error mean per-seed p99 also falls (prior iid932
.05696 ->.04713), descriptive only.

Second mode is unavailable on~38.5% of all clips; availability is strongly
regime-associated. Clean: none missing among stable last1%/last5% or m>=.15;
86.96% missing among unstable last5%. Similar pattern at new1% draws.
No rows excluded. The experiment tests the entire addition INCLUDING
missingness/similarity flags, not proof that phase or similarity alone causes
the gains. Stable last1% noisy MAE worsening is already present in training
predictions (new iid935 .003702 ->.004055; iid937 .003315 ->.003579),
so OOF generalization alone does not explain it. Cause of the boundary cost
is not established; no post-hoc threshold, gating, weight or feature sweep.

Warm single-thread CPU full inference measured on77 fixed clips (one/design),
10 repetitions per arm, interleaved order; seed42/fold0 predictors are timing
vehicles, not new accuracy evaluations. Includes raw25-channel causal
calibration, selected16/cap50, ONE modal estimate, all feature extraction,
train-fold scaling and native prediction. Excludes acquisition/noise generation,
disk IO, startup and cold execution. Current -> two-mode:
median5.560 ->5.603ms, p95 6.935 ->7.018ms, max10.455 ->10.831ms;
median paired overhead.034ms. This is workstation/warm CPU timing, not sensor
acquisition duration, GRU timing, a hard deadline or deployment latency claim.
Every timed raw-to-feature/prediction path agrees with retained fresh935
features and saved predictors.

Decision: two-mode context adds useful aggregate/farther-stable information
with little measured inference overhead, but is NOT a drop-in warning-focused
upgrade. Retain spatial107 as the current boundary-focused reference; keep
two_mode174 as an accuracy-oriented experimental comparator, not a default.
Any next probe should address the observed last1% fitting/representation
tradeoff against BOTH references, not claim that more spatial columns alone
solved warning quality. GRU remains live with improved representations.
No reserved selection/calibration/test scoring or production-default change.
Intervals remain2000 whole-design fixed-prediction resamples, no refit/seed
uncertainty or multiplicity correction after exploratory feature choices.

Verification:34 focused tests pass, including recovered second-mode pattern,
independent complex-gauge/extreme-scale/timebase invariance, visibility/tie
selection, orthogonal/collinear similarities and missing-mode semantics.
Actual raw D000000 extraction and fresh42-feature replay smoke passed.
Unequal-design event-rate bootstrap smoke passed. Completed extraction/fit
resume reused all77/45 shards. Final numerical smoke checked current hashes,
all15 rates/sign changes,12-vs3 exposures, design-isolated consistent sham
permutations, exact historical controls, and ALL45 serialized predictors on
clean/new independent/new correlated held-out queries.

Evidence in p4_groups/two_mode_spatial_20260930/: protocol.json,
prepare_features.py,77 acquisition shards, features.npz, run_experiment.py,
45 fit/predictor shards, predictions.npz, fits.json, analyze_results.py,
results.json, measure_inference.py, inference_measurements.npz and
inference_timing.json. Shared API/tests: modal_features.py and
tests/test_spatial_modal_features.py. Prior experiment artifacts unchanged.

2026-09-30: completed the user-requested fixed-blend verification.
Motivation: after the two-mode boundary cost, user asked about keeping both
benefits. A post-hoc50/50 check on saved935/937 OOF predictions improved
overall and stable last1% MAE, but retained5% stress costs. No blend-weight
search. That was exploratory, not independent verification. User then asked
for verification; froze protocol before new draw generation:
p4_groups/blend_verification_20260930/protocol.json.

Rule remains .5*spatial107 + .5*two_mode174, paired by MLP seed/outer fold.
No seed ensemble, retraining, learned gate, true-margin routing, clipping,
threshold tuning or missing-mode override. Reused30 native predictors from
the completed two-mode experiment. Hash-pinned model manifest/package versions
checked before loading trusted local model bytes; model hashes unchanged after
evaluation. Preserved train-design and OOF predictions at original5 folds
and seeds42/43/44, with12 versus3 exposures per clip.

Six new conditions,4454 physical clips/77 TRAIN designs: iid1_fresh939,
correlated_fresh939_940, iid1_fresh941, correlated_fresh941_942,
iid5_fresh939, iid5_fresh941. Same selected16/causal51/cap50/lag64/max8,
original-eight calibration RMS noise convention, shared-noise rho.8 before
normalization, source-row SeedSequence.1% and5% deliberately reuse each white
realization for paired severity probes. New draws are acquisition robustness
checks of EXISTING structures, not independent structural generalization.
No reserved selection/calibration/test scoring.

OOF mean DESIGN MAE, metrics averaged across paired seeds:
| New draw | One-mode107 | Two-mode174 | Fixed blend50 |
|---|---:|---:|---:|
| Independent1%939 | .009964 | .008876 | .008512 |
| Correlated1%939/940 | .010860 | .009602 | .009301 |
| Independent1%941 | .009991 | .008828 | .008547 |
| Correlated1%941/942 | .010763 | .009726 | .009382 |
| Independent5%939 | .016819 | .015671 | .014643 |
| Independent5%941 | .017389 | .016611 | .015540 |
Blend beats BOTH individual models globally on all six draws; paired95%
intervals exclude zero for each comparison. Against one-mode at new1%:
delta-.001453[-.001636,-.001270] iid939;
-.001559[-.001799,-.001327] correlated939;
-.001444[-.001635,-.001254] iid941;
-.001380[-.001580,-.001185] correlated941.
These are12.82-14.58% relative gains. Global gains versus one-mode repeat in
EVERY seed and EVERY outer fold at all six conditions.

Stable0<m<.01: one-mode -> blend design MAE and nonpositive-prediction rate:
| New1% draw | Design MAE | Wrong-sign fraction |
|---|---:|---:|
| Independent939 | .003782 ->.003387 | 16.86% ->15.31% |
| Correlated939/940 | .004455 ->.003941 | 13.95% ->12.98% |
| Independent941 | .003758 ->.003489 | 15.31% ->14.92% |
| Correlated941/942 | .003865 ->.003788 | 15.31% ->13.76% |
All four point estimates improve. Stable last1% MAE paired intervals exclude
zero for the939 independent/correlated draws but cross zero for both941
draws. All four stable last1% wrong-sign change intervals cross zero.
This is not equivalence/noninferiority proof.

Unstable-.01<m<0 MAE improves at all four1% draws, with paired intervals
excluding zero. Sign decisions are not uniformly better: iid939 wrong-sign
16.87% ->17.49%, delta+.62 percentage points,95%[-1.90,+3.37].
Correlated93919.34% ->18.52%, iid94118.11% ->16.46%,
correlated94118.72% ->18.31%; their sign-change intervals also cross zero.
The predefined strict descriptive criterion required global improvement AND
nonworsening stable last1% MAE/stable and unstable last1% wrong-sign POINT
rates on each1% draw. Three of four pass; iid939 fails the unstable sign
check. Do not average away this failure or treat its uncertain increase as
proven harm. These remain clip proxies, not operational warning/event rates.

Broader stable last5% MAE improves on all six draws, paired intervals below
zero. Farther-stable m>=.15 underestimation by>.05 at new1%:
iid9396.33% ->3.52%, correlated9396.45% ->3.89%,
iid9416.41% ->3.47%, correlated9416.49% ->4.30%;
all four paired intervals below zero. Global benefits are not bought by
worsening this farther-stable guardrail.

5% stress still prevents unrestricted adoption. Stable last1%:
iid939 MAE .005422 ->.005834 (interval crosses zero), wrong-sign
13.76% ->14.53%; iid941 MAE .005348 ->.006066, delta+.000718,
95%[+.000168,+.001326], wrong-sign14.92% ->15.12%.
Unstable last1% wrong-sign19.96% ->23.66% at939,20.58% ->19.14% at941;
both change intervals cross zero. Neither5% draw passes all descriptive
guardrails, even though global and stable last5% MAE improve on both.
No retuning after these results.

Actual shared-extraction inference verified from fresh939 raw signals:
calibration and ONE modal estimate build174 columns; one-mode uses first107,
two-mode uses all174, then average their native predictions. Warm single-thread
CPU,77 clips/10 repetitions each arm, rotating order, no acquisition/noise
generation/disk/startup time. Same-run median/p95/max milliseconds:
one-mode6.145/9.919/12.639; two-mode6.208/9.948/12.756;
blend6.379/10.241/17.165. Median paired blend overhead versus one-mode.235ms,
versus two-mode.175ms. All timed features/predictions match retained raw-to-
feature and predictor results. This run's blend p95 exceeds10ms; no hard
deadline qualification or comparison with older runs' absolute timing.

Decision: fixed50/50 blend is a stronger working accuracy candidate with
encouraging1% boundary MAE, not a verified universal best-of-both replacement.
Keep blend50 and one-mode107 as explicit comparators; no production-default
change. New acquisition draws support the gains, but the strict1% point
criterion fails once and5% boundary costs remain. Noise percentages are
uncalibrated sensitivity settings, so do not route by an assumed deployment
noise level. Intervals are2000 paired whole-design fixed-prediction resamples,
not model-refit/seed uncertainty or corrected multiple comparisons.

Verification: actual extraction/evaluation/analysis/timing CLIs completed.
Fresh extractor replayed the prior174-column convention on a raw D000000
sample. Resume reused77 acquisition and15 evaluation shards without fitting.
Final independent numerical smoke checked30 unchanged model hashes/native
held-out replays, exact50/50 train/OOF arrays, all six conditional/sign rates,
12vs3 exposures, design isolation, current result/timing hashes, timed blend
arithmetic and the reported3of4 primary/0of2 stress point-check outcomes.
No new permanent tests or shared production-code changes.

Evidence: p4_groups/blend_verification_20260930/ contains protocol.json,
prepare_draws.py,77 acquisition shards, features.npz, model_manifest.json,
evaluate_blend.py,15 evaluation shards, predictions.npz,
analyze_verification.py, results.json, measure_blend.py,
inference_measurements.npz and inference_timing.json.

2026-09-30: completed the comprehensive spatial ingredient/diagnostic/
robustness/warning program requested by the user. Frozen protocol and all
artifacts: p4_groups/spatial_comprehensive_20260930/. This is still only
77 TRAIN designs/4454 physical clips; reserved selection/calibration/test
designs remain unscored. New noise draws are acquisition confirmation on
the same structures, not independent structural validation. No shared
production-code change, new architecture, loss sweep, blend-weight search,
true-margin routing, solver-truth feature selection or automatic adoption.

Eight fixed ingredient combinations retain original107 plus explicit
second-mode/pair validity flags: F flags-only109, P channel-powers125,
C complex neighbor relationships157, S similarity110, PC173, PS126,
CS158, PCS full174. C includes amplitude AND phase, not pure phase.
Flags are empirically identical on these draws; preserved their separate
validity semantics rather than changing the frozen definition.
Same selected16, causal51-sample per-channel calibration, cap50, lag64,
max8 estimated modes; same32->16 MLP/alpha1, seeds42/43/44, five outer
design folds and four equally weighted original training views.
Seven reduced variants required105 fits; full174/one107/blend50 references
were frozen. All105 converged without warnings.

Exploration covered all21 cached conditions, including residual gain and
reported timebase +/-2%, with stable/unstable last1/2/5% and farther-stable
m>=.15 slices, train/OOF gaps, per-seed/fold results and paired uncertainty.
Mean global DESIGN MAE over the latest six cached conditions:
one .012631; F .012733; P .012346; C .011940; S .012640;
PC .011680; PS .012258; CS .011822; PCS .011552; blend .010988.
This average is descriptive, not the acceptance test. Powers/complex
relationships improve global accuracy while harming the noisy boundary.
At iid1_fresh939, stable last1% OOF MAE is one .003782, P .004563,
C .004604, PC .004865, PCS .004706, blend .003387.
At iid5_fresh939 it is one .005422, P .006455, C .008430,
PC .008411, PCS .008106, blend .005834.
Flags/similarity alone do not improve the six-condition global average.
No single removed block yields a demonstrated best-of-both replacement.

Prerecorded ranking locked P and PC before new confirmation outcomes,
with all30 finalist model hashes retained in selection.json. These are
exploratory eligibility finalists, NOT formally qualified survivors.
Fit two same-width controls, jointly permuting added columns within outer
training/held-out source rows separately, using seed934+fold and the same
mapping across acquisition views/MLP seeds. Original107 remain aligned.
All30 control fits converged without warnings. Controls are diagnostic,
not deployable; permutations may create physically inconsistent inputs.
control_results.json compares finalists with shams and frozen references
on all21 cached conditions. Both finalists beat their shams globally with
paired intervals below zero on all four new1% confirmation conditions.
Therefore useful signal is not explained merely by input-layer width.

Diagnostic replay: clean, iid1/5 at939/941, correlated1 at939/940 and
941/942. All174 columns replay EXACTLY: maximum delta0 across
7*4454 extractions. Solver poles enter ONLY the diagnostic frequency
matching; recovery means Hungarian frequency match within2Hz or5%,
not guaranteed correct damping or full modal identification.
Stable last1% critical-pole recovery is100% clean,99.42% iid1_939,
95.93% iid5_939 and97.67% correlated1_939/940.
In boundary cohorts whose selected lead frequency matches the critical
pole, one -> PCS train/OOF design MAE:
iid1_939 .002950/.003588 -> .003267/.004329;
iid5_939 .003556/.003911 -> .005703/.006342.
The cost is not solely loss of the critical pole or held-out structures.
On the actual original fitting views, stable last1% training MAE
one -> PCS: clean .003167 -> .003069; iid1_928 .003125 -> .003297;
iid5_928 .004233 -> .004467; iid1_929 .003147 -> .003411.
This supports a fitting/representation tradeoff under noise, without
proving a unique cause or sufficiency of the current representation.

Repeat-draw diagnostics also show instability: second truth identity
switches in21.52% of237 matched boundary pairs at iid1,43.67% of158
at iid5,28.64% of220 at correlated1. Unresolved pairs97/176/114
are reported separately, not counted as successful recovery.
Two-draw boundary prediction variance one -> PCS:
iid1 .000000970 -> .000005110; iid5 .000009720 -> .000043400.
Squared two-draw mean error also rises: iid1 .00002632 -> .00003659;
iid5 .00006358 -> .00010324. These are finite two-draw descriptive
decompositions, not identified population noise/bias components.
Observed recovery/residual/availability cohorts and factorial effects
are associations; no cross-validated observable reliability gate was
established, and solver-truth diagnostics cannot become an inference gate.

Eight new locked conditions use white943/shared944 and white945/shared946,
iid/correlated1% and5%, original-eight RMS convention, rho.8 shared noise
before calibration. Levels share each realization for paired stress
comparison. No fitting or selection on these confirmation draws.
OOF mean DESIGN MAE, averaging metrics across three paired seeds:
| Draw | One107 | Full174 | Blend50 | P125 | PC173 |
|---|---:|---:|---:|---:|---:|
| iid1_943 | .010296 | .008951 | .008763 | .009802 | .009090 |
| corr1_943/944 | .010844 | .009669 | .009392 | .010616 | .009805 |
| iid1_945 | .009960 | .008774 | .008495 | .009484 | .008792 |
| corr1_945/946 | .010758 | .009636 | .009306 | .010424 | .009690 |
| iid5_943 | .016833 | .015978 | .014949 | .016574 | .016099 |
| corr5_943/944 | .027544 | .022586 | .023524 | .025498 | .022719 |
| iid5_945 | .016654 | .015984 | .014782 | .016394 | .015896 |
| corr5_945/946 | .026804 | .022929 | .023445 | .025114 | .023280 |
PC's global gain versus one has paired upper95% bound<0 on all eight
conditions; P's does on only four. Blend beats one globally in every
seed/outer fold on all eight conditions. Full174 has lower global point
MAE than blend on BOTH correlated5% stress draws; blend is not universally
best against both individual models. Do not average away this exception.

Formal acceptance allowed ZERO deterioration: upper paired95% bound<=0
for stable last1% MAE, stable/unstable last1% wrong-sign rates and farther-
stable underestimation>.05 on EVERY new condition, with significant global
gain and primary matched-control gain. Neither P nor PC passes any of the
eight condition-specific full guardrail sets: no qualified survivor.
Neither passes all descriptive point checks on any new condition either.
Both worsen stable last1% MAE on all four new1% conditions:
one .003703/.004410/.003854/.003878;
P .004473/.004950/.004763/.004780;
PC .004871/.004818/.004973/.005194.
Sign-rate noninferiority is not inferred from intervals crossing zero.
All boundary/stress slices, individual acceptance failures, pairwise
comparisons and seed/fold results remain in confirmation_results.json.

Separate warning study:60 additional native MLP fits, all converged without
warnings. For each outer training fold, shuffled design split seed947+fold:
48/49 proper-fit designs,13 calibration designs,16/15 held-out designs.
No design/noise-copy leakage. Proper-fit-only models for one/PCS/P/PC;
blend pairs these proper-fit one/PCS predictors, NOT full-training refits.
Thresholds calibrated on stable clips pooled across four original training
noise views at empirical false-warning budgets1/5/10/20%; nextafter below
the next sorted score handles ties. Thresholds and proper-fit predictors
remain unchanged on all eight new outer-held-out conditions.
Unsafe=m<=0; warning=prediction<=threshold. These are CLIP proxies,
not operational event/lead-time rates, conformal guarantees or deployment
false-alarm specifications. Calibration budgets need not hold under shift.

At the1% calibration budget, iid1_943 all-stable false-warning/all-unstable
miss rates one -> blend: .995%/2.120% -> .905%/1.975%;
paired miss delta95%[-.541,+.217] percentage points.
Correlated1_943/944: .893%/2.373% -> .842%/1.938%;
miss delta95%[-.848,-.018]pp. The independent-draw gain is uncertain;
the second correlated1 draw's miss interval also crosses zero.
At the5% budget, iid1_943:4.807%/.399% ->4.973%/.362%,
but stable LAST1% warning rate rises63.76% ->68.60%, while unstable
LAST1% misses fall4.53% ->3.91%. Small whole-population miss rates hide
a large near-boundary warning burden. At this budget blend misses are
higher than one on BOTH iid5 and BOTH correlated5 stress draws; paired
miss-change intervals cross zero. No uniform warning dominance claim.
All zero/budget policies, near-boundary rates, calibration drift and paired
comparisons are retained in warning_results.json/warning_comparisons.json.

Warm single-thread full inference:77 first-row/design raw25-channel clips,
new iid1_943,10 repetitions/arm, rotating order. ONE modal estimate, shared
extraction for blending; include causal calibration/extraction/fold
transform/native forwards, exclude acquisition/disk/startup.
Reduced pipelines currently construct the full67-column addition before
subsetting; timing is for this implemented path, not an optimized extractor.
Same-run median/p95/max milliseconds:
one9.307/11.077/13.820; PCS9.441/11.231/14.189;
blend9.706/11.518/16.087; P9.468/11.182/15.608;
PC9.464/11.302/18.309. Every arm exceeds the10ms p95 engineering
screen in this run; no hard-real-time qualification or cross-run regression
claim. All timed feature/prediction replays and retained quantiles checked.

Decision: retain blend50 as the working accuracy comparator and one107 as
the boundary-focused comparator; retain full174 for correlated-noise stress
comparison. No reduced replacement, new default, automatic weighted loss
or reliability gate. Ingredient removal does not solve the boundary cost.
Both noise variability and noisy-view in-fit error justify investigating
the fitting/decision objective rather than adding more columns. The next
supported controlled direction is a separate warning-specific objective
on the frozen improved representation, keeping the physically calibrated
margin estimate intact and testing explicit near-boundary warning/miss
costs against this training-calibrated threshold baseline. Do not blindly
repeat the earlier unchanged local-weighting experiment. Application costs,
noise specification and acceptable deterioration remain undeclared; this
study does not select a deployment policy. GRU remains a later improved-
representation comparison, not abandoned or tested in this program.

Verification: all extraction/fitting/analysis/timing CLIs completed;195 new
fits converged without warnings (105 ingredients+30 controls+60 warnings).
Interrupted background stages resumed saved atomic shards without changing
the protocol or finalist lock. Numerical smokes checked batched2000-draw
whole-design bootstrap equivalence to existing helpers, tie/fold threshold
indexing, actual raw-to-native inference, seven ingredient native OOF
replays, sham permutations/design isolation,48 fresh native OOF replays,
56 fresh macro-MAE records, exact train/OOF blend arrays and16 formal
rejections. Warning checks verified60 converged fits, disjoint calibration,
actual calibrated thresholds/native held-out replay,40 zero-policy rates
and timing quantiles. Fingerprints verified60 warning models,30 locked
finalist models,30 unchanged historical models and current source/results.
Intervals use2000 fixed-prediction whole-design paired resamples, averaging
seed/exposure metrics BEFORE resampling; no refit/seed uncertainty or
multiplicity correction after exploratory selection. No new permanent tests.

Evidence in p4_groups/spatial_comprehensive_20260930/: protocol.json,
common.py, run_matrix.py,105 reduced/30 control native shards,
predictions_F_P_C_S.npz, predictions_PC_PS_CS.npz, predictions_controls.npz,
fits_*.json, assess_matrix.py, matrix_results.json, selection.json,
prepare_diagnostics.py,77 diagnostic shards, diagnostics.npz,
analyze_diagnostics.py, diagnostic_results.json, analyze_controls.py,
control_results.json, prepare_confirmation.py,77 acquisition shards,
confirmation_features.npz, confirm.py, confirmation_predictions.npz,
confirmation_results.json, warning_study.py,60 warning model shards,
warning_predictions.npz, warning_results.json, analyze_warning.py,
warning_comparisons.json, measure_inference.py, inference_measurements.npz
and inference_timing.json.

2026-10-01: completed the boundary-localized noise-consistency experiment
started2026-09-30, after resuming interrupted stages from saved model shards.
Frozen protocol and evidence: p4_groups/boundary_noise_consistency_20260930/.
This directly tested the user's proposed intersection of near-flutter
fitting and noise robustness, not another feature expansion.

Scope remains77 original TRAIN designs/4454 physical clips. Reserved
selection/calibration/test untouched; no structural regeneration.
Full174 two-mode spatial features, selected16/causal51/cap50/lag64/max8,
MLP32->16, alpha1, seeds42/43/44 and original5 design folds unchanged.
All THREE arms use the same custom L-BFGS-B objective/optimizer, scaling,
initialization and four supervised training views: clean, iid1_seed928,
iid5_seed928, iid1_seed929. Supervised half-MSE and L2 unchanged.
Arms: supervised lambda0; global consistency lambda1; localized lambda1
on pairs with abs(TRAINING true margin)<.05, on BOTH sides of flutter.
Pair only the two independent1% measurements of the SAME physical clip.
Penalty=lambda/2 * sum(pair_weight*prediction_gap_squared)/sum(pair_weight).
Normalization matches the overall coefficient, not per-included-clip
strength: localization concentrates influence on fewer pairs. No supervised
sample weighting, asymmetry, skip, clipping, band/lambda sweep or inference
true-margin routing. The5% view remains supervised; no paired5% penalty.
45/45 new matched fits converged; native one107/full174 and fixed blend50
remain frozen references. Matched custom control avoids attributing a
different optimizer trajectory to the consistency term.

Fresh verification: white949/shared950 and white951/shared952, each at
independent/correlated1% and5%: eight conditions, features(8,4454,174).
Correlated noise rho.8 before causal calibration; original-eight aggregate
calibration-RMS convention unchanged. No tuning on these draws. The29
previous acquisition conditions are retained as exploratory comparisons.
These are new noise realizations on the SAME designs, not new structures.

Stable LAST1% macro design MAE, primary1% conditions:
condition                 supervised global   localized one107   blend50
iid1_949                  .004624    .004297  .004257   .003973  .003517
correlated1_949/950        .004909    .004495  .004697   .004089  .003866
iid1_951                  .004734    .004352  .004202   .003571  .003429
correlated1_951/952        .004989    .004576  .004728   .003963  .003679
Localized improves this point estimate versus supervised on all four, but
only iid1_951 has a paired95% improvement interval excluding zero:
delta-.0005313 [-.0009936,-.0000646]. First iid draw:
delta-.0003663 [-.0007985,+.0000614]; both correlated intervals cross zero.
Localized versus GLOBAL has mixed point outcomes; all four stable-last1%
paired intervals cross zero. Concentrating the penalty is not a verified
uniform improvement over the global penalty.

Stable LAST1% macro design MAE,5% stress conditions:
condition                 supervised global   localized one107   blend50
iid5_949                  .007848    .006391  .006906   .004983  .005137
correlated5_949/950        .014048    .013140  .012622   .009323  .010602
iid5_951                  .008905    .007617  .007417   .005616  .005863
correlated5_951/952        .014404    .014499  .012387   .008995  .010897
All four localized-minus-supervised stable-last1% intervals exclude zero.
Unstable-last1% MAE also improves versus supervised on all eight point
estimates, but not all paired intervals exclude zero. Relative to blend50,
localized is WORSE on both stable and unstable last1% MAE in ALL eight
conditions, with all16 paired deterioration intervals excluding zero.
One107 also has lower last1% MAE on both sides in all eight point estimates.
Full boundary1/2/5% slices, error exceedance curves and native full174
comparisons are retained in cached_results.json/fresh_results.json.

Global macro MAE improves versus matched supervised on all eight point
estimates; intervals exclude zero on only three. Example iid1_949:
.008868 ->.008718, delta-.0001503 [-.0003009,+.0000044], versus blend.008530.
Global gains versus supervised hold in only7-12 of15 seed/fold cells per
condition, not uniformly. Farther-stable underestimation>.05 is worse on
six of eight point estimates. Example iid1_951:3.350% ->4.012%.
Stable-last1% wrong-sign point rates fall on all eight conditions; unstable
wrong-sign rates worsen on three. Example correlated1_951/952:
19.753% ->20.165%, delta+.412pp, paired95%[-5.428,+6.251]pp.
These are zero-threshold CLIP proxies, not operational warning/miss rates.

Boundary334 clips:172 stable from58 designs,162 unstable from51 designs.
Two-fresh-draw boundary prediction variance, supervised -> localized:
iid1 4.721e-6 ->3.227e-6 (-31.64%);
correlated1 5.579e-6 ->3.438e-6 (-38.37%);
iid5 3.724e-5 ->2.918e-5 (-21.63%);
correlated5 1.056e-4 ->6.067e-5 (-42.56%).
Between-draw sign disagreement respectively:
10.48% ->7.88%,11.68% ->8.58%,25.55% ->20.46%,20.26% ->15.57%.
Useful stability gains, but one107/blend50 retain lower variance and
disagreement in every noise family. Decomposition retains the squared
error of the TWO-DRAW mean, not an estimate of population bias. For
correlated1 this component rises3.887e-5 ->3.913e-5 despite lower variance.
Global consistency also demonstrates why agreement is insufficient:
iid1 variance falls, but squared two-draw mean error rises
3.419e-5 ->3.601e-5. No causal claim that noise is the sole error source.

In-fit stable-last1% MAE improves under localized fitting on the original
noisy training views, but does not recover the blend baseline. iid1_seed928
training/OOF: supervised.003397/.004606; localized.003161/.004388;
blend.002622/.003430. Clean in-fit error slightly worsens
.002988 ->.003029. Thus localization changes fitting as well as noise
variability; neither effect eliminates the remaining boundary cost.

Frozen ZERO-deterioration acceptance: localized must significantly improve
stable-last1% MAE versus supervised on every primary1% condition, and
prove noninferiority for unstable-last1% MAE, both wrong-sign rates, global
MAE and farther-stable underestimation>.05; all stress checks must also
pass. Separately require the one107/blend50 reference guardrails.
NO condition passes its full formal check set against supervised, one107
or blend50:24 condition/reference rejections. Only correlated1_949/950
passes every descriptive point check versus supervised. A zero-crossing
interval is NOT noninferiority proof. All45 fits converged, but neither
matched-control qualification nor reference replacement qualification
passes. No newly trained warning classifier or threshold calibration was
part of this study.

Warm single-thread raw25-channel-to-native prediction,77 fixed first-row/
design iid1_949 clips,10 repetitions/arm, rotating order; one modal estimate,
causal calibration/features/fold transform included, acquisition/disk/
startup excluded. Same-run median/p95/max milliseconds:
one107 6.051/8.083/11.863; full174 6.099/8.053/9.719;
blend50 6.206/8.290/10.562; supervised5.853/7.808/13.468;
global5.721/7.837/10.393; localized5.791/7.640/10.236.
All p95 values pass the10ms engineering screen THIS run, not hard-real-time
qualification or a cross-run speedup claim. Localization adds no inference
gate or additional modal estimate.

Decision: reject this fixed localized penalty as a solution/replacement,
while retaining its bounded fitting/stability gains as evidence. It does
more than suppress disagreement, but cannot recover the existing blend's
boundary accuracy or satisfy the agreed guardrails. Do not adopt it or
automatically sweep its coefficient/band. Keep blend50 as the working
accuracy comparator, one107 for boundary behavior and full174 for
correlated-noise stress; no fitted default changed. This probe does not
rule out every noise-aware objective. A separate warning-specific objective
remains the next supported distinction to test against the established
calibrated-threshold baseline, preserving the physical margin estimate;
application costs/noise specifications remain undeclared. GRU stays live
for a later improved-representation comparison.

Implementation: skip_mlp.objective_setup and fit_network accept optional
consistency_pair_weights; validate finite nonnegative positive-sum weights
and normalize the pair term/gradient. Existing callers intentionally retain
their prior unweighted/native behavior; unsupported native pair weighting
is rejected, not silently ignored. Weighted finite-difference gradients,
repeated endpoints, zero-weight exclusion and weight-scale invariance are
covered by the existing objective test module's extended behavior checks.
Verification:46 focused tests pass. Actual-data weighted fitting/raw174
feature replay passed before the full experiment. All resumed fitting,
fresh evaluation, analysis and timing CLIs completed without changing the
frozen protocol. Numerical smokes checked600 native train/OOF prediction
replays,45 converged fits,30 unchanged reference model hashes, pair/source
and design isolation, exact blend arrays,384 independently recomputed
macro-MAE records, two-draw MSE decomposition,24 formal rejections,
16 significant boundary losses versus blend, current source/result hashes
and timing quantiles. Intervals use2000 paired whole-design fixed-prediction
resamples; no refit/seed uncertainty or multiplicity correction.

Evidence in p4_groups/boundary_noise_consistency_20260930/: protocol.json,
experiment_helpers.py, reference_manifest.json, run_experiment.py,45 model
shards, fits_supervised_global.json, fits_localized.json,
predictions_supervised_global.npz, predictions_localized.npz,
prepare_draws.py,77 acquisition shards, fresh_features.npz, evaluate.py,
evaluation_manifest.json, fresh_predictions.npz, analyze_results.py,
cached_results.json, fresh_results.json, measure_inference.py,
inference_measurements.npz and inference_timing.json.

2026-10-01: user requested a deeper prior-art/impact check and a closely
related harder pivot if the current contribution is not defensible.
Sourced assessment appended to Section11 of
src/mechanics/p4_margin_estimation/RESEARCH_REVIEW.md. Research only;
no new fitting, reserved-split scoring, default change or GRU abandonment.

Important prior-art correction: Crowther/Cooper2001 already trained neural
flutter predictors on model data and examined noise-matched training.
Wang/Ma2024 already combine simulated panel dynamics, explicit structural/
operating inputs and decay information with recurrent prediction.
Wang/Zhou June2026 reports acceleration/PSD/Transformer-LSTM boundary
prediction with wind-tunnel and flight-test validation. Publisher-deposited
abstract: mean errors3.73% on unseen wind-tunnel working conditions and
4.14% on one sortie; below10% error at about70% of critical speed.
Those percentages are NOT directly comparable with our margin MAE.
Unseen working conditions are not verified unseen structural designs.
Full observation/calibration protocols remain unavailable; no uniqueness
or superiority claim follows from that missing information.

Full manuscripts read: Chajjed2022 uses random structural-model modal
priors and multiple target airspeeds for probabilistic flutter prediction;
Mendez September2026 constrains a finite-record pole to an independently
validated trajectory for one configuration, with representative15->5-cycle
sign-identification benefit; Zastrow2022 learns fast coupled ROMs from
full-state CFD/modal information. These already occupy broad claims about
physical priors, short-record identification and fast simulation surrogates.
Primary links, evidence levels and access limitations are in the addendum.

Read-only TRAIN metadata replay matched source rows/targets/design IDs.
334 clips/70 designs with0<abs(m)<.01: finite-distance -label_alpha/m
p10/median/p90=272.719/465.365/1033.489 per second; p90/p10=3.790.
This is NOT a local derivative estimate. Label mode retained334/334 does
not prove visibility/excitation. Full512-sample aperture median.499s;
abs(label_alpha)*aperture median1.012, p10/p90.235/2.882. Full aperture
is not the usable pre-cap prefix. Evidence: metadata_arrays.npz and
two_mode_spatial_20260930/features.npz; no new structures or models.

Assessment: no established distinctive advantage yet; not evidence that
P4 is redundant. Conditional go for cross-design decisions without target
flutter calibration IF a matched-risk observation/calibration/availability
benefit can be demonstrated. Neural architecture, noise augmentation,
continuous output and conformal intervals alone cannot be the contribution.
Restricting inputs without adequate performance is not an advantage.

Recommended central objective: determine what reliable margin/stability
decisions a short observation supports across unknown designs, and when
additional observation is required. The proposed next diagnostic separates
measured modal error from design-dependent growth-to-margin calibration:
TRAIN-only corresponding true-modal versus measured-feature mapping,
matched folds/capacity, no margin/Ucrit/-alpha-per-margin input leakage.
A failed oracle regressor alone is not an identifiability proof.
Then compare a training-population physics-prior baseline on the SAME
samples/sensors/timebase with current comparators and learned decisions.
Known-trajectory/multiple-target-speed methods get a separate information
track. Final structural-region/model-form checks and actual trajectory
warning evidence are still required; clip rates are not event guarantees.

If calibration ambiguity persists, investigate transferable probabilistic
physical calibration; if one observation lacks information, pivot to
minimal extra acquisition (longer window/channel, then justified same-speed
re-excitation), declaring any changed input contract. Generic selective
prediction/active sensing is not novelty either: require a physical finding
and useful comparative benefit. Do not default to a target speed sweep or
another unchanged loss sweep. This research-strategy recommendation refines
the earlier warning-loss next step; it does not select a deployment policy.

2026-10-01 continuation: user clarified that short observation plus unknown
design may be our main edge, but is not the only relevant problem; doing
an established task demonstrably better can still be a contribution.
Correction to the preceding research-strategy framing: prior work rules
out claiming familiar ingredients alone as novel, NOT improving the same
objective. No objective pivot is selected merely because precedent exists.
Immediate research priority is a results-and-protocol comparison before
deciding whether to retain or change the central claim. Existing physical/
oracle diagnostics remain useful, not substitutes for that comparison.

Short observation alone is not an established distinction. Mendez2026
explicitly examines2-60 critical-mode cycles; a representative30dB case
needs5 rather than15 cycles for>95% stability-sign accuracy using an
independently established physical trajectory. This is not cross-design
normalized-speed-margin inference. Chajjed's demonstrated numerical
workflow combines target observations at three subcritical airspeeds.
For the closest2026 neural papers, record duration and whether inference
aggregates multiple windows/airspeeds remain unverified. Do not assume
their observations are long. The candidate combined edge is one short
transient, unseen structural design, no target-specific flutter calibration
or speed-sweep measurements, WITH useful comparative performance.

Other meaningful improvement axes: near-flutter accuracy under noise
without shifting errors conservatively everywhere; calibrated uncertainty
with useful availability; robustness to physical/model-form mismatch;
weak critical-mode excitation and modal ambiguity; actual trajectory-level
warning performance; sensor/observation/calibration/computation efficiency.
These are unresolved P4 problems and comparison axes, not claims that
nobody has studied or solved particular instances of them. We need not
win every axis, or invent a different target/architecture to contribute.

Results to understand rather than dismiss: Wang/Zhou abstract reports
3.73% mean boundary prediction error on unseen wind-tunnel conditions,
4.14% on one flight sortie and below10% at about70% critical speed.
Mendez reports median damping-RMSE gain3.915 across tested synthetic cells.
Chajjed Table2 reports flutter-speed COV4.5903/3.7115/2.5625 for flat/
independent/joint priors: about44% reduction joint versus flat in that
numerical example. Uncertainty spread, stability-sign probability and
boundary percentage error are different metrics from our margin MAE.

User will check access to unavailable full texts, priority order:
1. "Flutter Boundary Prediction Based on Feature Extraction and Bayesian
Optimization Transformer-Long Short-Term Memory" (Wang/Zhou2026),
DOI10.2514/1.C038983.
2. "Flutter Boundary Prediction Method Based on Long Short-Term Memory
Networks" (Shi/Zhou/Guo2026), DOI10.2514/1.C038429.
3. "Application of Deep Learning Models to Predict Panel Flutter in
Aerospace Structures" (Wang/Ma2024), DOI10.3390/aerospace11080677.
4. "Flight test flutter prediction using neural networks"
(Crowther/Cooper2001), DOI10.1243/0954410011531736.
Full methods and supplements are needed, not only abstracts: exact target/
metric, seconds/cycles per prediction, sensor and operating-point inputs,
target-specific calibration/history, structural train/test separation,
noise/physical conditions, baseline results and uncertainty/failure cases.
Then judge whether P4 improves a meaningful result under comparable
conditions; label extra-information comparisons separately.

No new experiment, default, deployment policy or approved pivot. Reserved
splits remain untouched. P4's current objective and the later improved-
representation GRU comparison remain live.

2026-10-01 continuation: user narrowed paper extraction to objectives,
results/direct comparability and useful method components; no complete
reproducible baseline is needed. Focused assessment recorded in Section12
of src/mechanics/p4_margin_estimation/RESEARCH_REVIEW.md, with primary links.
Neither accessible paper's headline numbers directly rank against P4:
Chajjed's posterior critical-speed COV is not margin MAE; Mendez's damping
RMSE/sign probabilities use another system and known physical trajectory.
Chajjed's joint physical priors inform probabilistic margin calibration;
Mendez's frequency-growth constraint informs short-record identification.
Borrow mechanisms only with uncertain, design-dependent relations, not
their fixed calibration or unobserved target airspeed history. Our features
already contain frequency/damping. No reproduction, implementation or new
experiment selected; GRU and current objective remain live.

2026-10-01 continuation: user asked whether probabilistic prediction would
improve real-world usability and academic depth, and requested a sanity
check of unfinished P4 against Chajjed/Mendez without rereading the papers.
Assessment only; no new literature retrieval, experiment or scoring.

Uncertainty correction: P4 already explored quantile predictions,
conformal intervals and calibrated decision thresholds. We do not yet
have demonstrated real-world reliability or validated useful uncertainty
for the current improved representation across unseen designs and physical
mismatch. A predicted distribution expresses uncertainty, not certainty.
Narrow confidently wrong distributions and universally wide intervals
are both failures; lower global MAE does not establish trustworthy
near-flutter decisions. Uncertainty alone does not repair point bias.

[INFERENCE] A physically grounded probabilistic formulation could improve
usability substantially if it identifies unreliable estimates while
retaining useful decisions. Separate measurement/identification uncertainty
from design-dependent growth-to-margin calibration ambiguity and physical
model mismatch. Modeling the first two within the training population
does not automatically detect or provide calibration under the third.
Retain the physical margin estimate; investigate uncertain joint physical
relationships rather than choosing a generic variance-output network.
Judge benefit by dangerous decision rates at comparable useful availability,
not distribution output or interval width alone. No implementation chosen.

[INFERENCE] Academic value could come from explaining which uncertainty
better observation can reduce and which reflects unresolved structural
information. Probabilistic flutter prediction itself is established.
The contribution would be physically explained, useful cross-design
inference/decisions, not merely adding a probabilistic head. GRU remains
compatible with this question and is not abandoned.

[INFERENCE] Comparative sanity check: P4 has substantial experimental
strengths (design-grouped evaluation, controlled noise comparisons, repeated
seeds, physical diagnostics and rejection of failed guardrails), but the
two papers currently have clearer mechanisms and more decisive central
results within their scopes. Chajjed demonstrates uncertainty reduction
with joint physical priors in a simplified two-DOF numerical system;
Mendez demonstrates short-record identification gains using a validated
trajectory for one configuration. Their metrics cannot rank against P4
margin MAE. P4's wider design population and limited target information
give a potentially broader engineering goal, not a demonstrated advantage.
Current aggregate gains coexist with unresolved boundary tradeoffs;
real-world validity, distinctiveness and operational benefit remain unproven.
This is substantial unfinished research, not grounds for abandoning P4.
No approved objective pivot, model/default change, reserved-split scoring
or new experiment.

2026-10-01: user supplied two of the four requested manuscripts in Downloads.
Copied (originals retained) into p4_journal/papers/:
aerospace-11-00677.pdf = Wang/Ma2024; crowther2001.pdf = Crowther/Cooper2001.
Source/copy SHA-256 matched for each:
Wang/Ma: 4b796d2b3d981f52acdc61ae39a39baa13c99fba9dd93ca2ff797dfc381c92bc
Crowther/Cooper: ed9e1004bfba5127617a6cd69777037bebd23633127663fe001d6e2b7084a79b
Reviewed full-text methods/results relevant to objectives, comparability
and useful components; Section13 of RESEARCH_REVIEW.md records the details
and primary/local citations. Updated Section11 evidence/access inventory.
Core six-paper set now4/6 full texts; broader eight-paper table5/8.
Still missing the two2026 AIAA papers: Wang/Zhou and Shi et al.

Wang/Ma: simplified simply supported supersonic panel surrogate with
explicit physical inputs, not demonstrated noisy waveform-only inference.
Flutter-Mach LSTM MSE4.8e-5, R².99982, reported average relative error
.2182%; LSTM-NN boundary-class prediction accuracy99.44%. Classification
labels follow calculated decay sign, which is also a supplied feature.
Flutter-regression inputs specifically include aspect ratio, decay, load,
frequency and total damping; samples described as taken at the stability
boundary. Split70/25/5% is not a specified design-disjoint holdout; physical
record duration/noisy identification/calibrated uncertainty not established.
No CFD-FEM comparison. Scores do not rank against P4 margin MAE or warning
risk. Dataset-count prose mixes the separate flutter-boundary subset and
classification-sized test-count description; do not silently resolve it.

Crowther/Cooper: forecasts damping45knots ahead from five target airspeed
points at15knots spacing, using damping/frequency and current speed.
Important correction: nine±10% mass-perturbed aircraft models, eight for
training and one held out, already supply evidence of unseen model variation.
Do not claim structural holdout is absent from earlier neural work.
Quadratic control overpredicts flutter by25knots/6.4%; frequency-informed
network reported within about5%. Control lacks frequency information and
authors note multivariable regression might improve it. Clean-trained
network fails badly on noisy modal features; noise-matched training recovers
approximately clean accuracy, with remaining mode failures. Their±5%-of-
clipping-range feature noise is not our5%-RMS waveform noise. Presented
results are numerical; real flight-test application is future work.

[INFERENCE] Useful lessons: matched-information comparisons, realistic
identification/noise errors, joint modal information and mode-identity
diagnostics. Scaling, weighting and recurrent architecture alone add no
new scientific mechanism here. No new implementation justified by either
paper's scores. Target operating history/unmeasured physical parameters
cannot be imported without declaring a contract change. Both papers are
relevant precedent, not evidence that P4 is redundant. No new experiment,
reproduction, reserved-split scoring, default or approved objective pivot.
GRU remains live.

2026-10-01 continuation: user corrected the literature-reading emphasis.
Focus on transferable insights/mechanisms and their applicability to P4,
not primarily similarity or direct numerical comparability. A different
objective or information contract does not prevent learning a useful
principle; not every paper must supply an applicable improvement.
Mechanism-focused assessment recorded in RESEARCH_REVIEW.md Section14.

[INFERENCE] Most promising: uncertainty-aware joint modal/physical inference.
Chajjed informs physically coherent population priors and propagation of
identification/calibration ambiguity; Mendez informs use of better-resolved
frequency to constrain weak growth, with uncertain rather than universally
fixed relations. Current features already include frequency/growth/damping/
spacing; another column or generic variance head is not the main insight.
Potential benefit is coherent inference and meaningful uncertainty, not
assumed superiority from adopting a probabilistic architecture.

[INFERENCE] Mendez also motivates observation-sufficiency diagnostics in
usable duration, modal cycles, growth accumulation, excitation and noise,
not sample count alone. Existing fit/energy/usable-fraction/availability
information is a starting point, not validated confidence. True modes
remain TRAIN-only diagnostic quantities; no copied five-cycle threshold.

[INFERENCE] Crowther's frequency-versus-capacity result supports separating
missing information from fitting capacity. Its noise/sorting results motivate
diagnosing missed modes, substitutions and correlated identification errors
before inventing augmentation. Basic noise-aware training is already covered;
more independent noise copies alone do not represent these failure modes.
Wang/Ma's aggregate-versus-boundary result reinforces our existing conditional
metrics; weighting/scaling/depth tuning are mostly already-covered lessons,
not automatic improvements. Physical inputs can inspire hidden-calibration
diagnostics without becoming unmeasured inference inputs.

Existing true-modal/measured-feature diagnosis can distinguish which
mechanism matters: identification failure versus structural margin ambiguity.
No new architecture, experiment, reproduction, default or objective pivot
selected. GRU remains live.

2026-10-01: user approved experiments on the promising paper mechanisms.
Completed joint population uncertainty and soft frequency-growth inference
in p4_groups/physical_joint_inference_20261001/. This is a bounded adaptation
of the insights, not a reproduction of Chajjed or Mendez.

Scope:77 original TRAIN designs/4454 underlying clips; original five outer
design folds, seeds42/43/44, design-disjoint proper-fit/calibration/held-out
partitions matching the frozen warning study. No new simulation, target
airspeed history, reserved selection/calibration/test scoring or default
change. Four existing training acquisition views;37 evaluation conditions.
Fresh confirmation consists of eight independent/correlated1%/5% conditions
with draws949/950 and951/952. These are new noise draws, not new structures.

Mechanism: an eight-component full-covariance Gaussian-mixture prior over
true critical growth, frequency and margin, learned only on proper-fit
designs. A two-component joint measurement-error mixture learns observed
least-stable pole minus solver critical pole from the four training views.
Gaussian conditioning propagates uncertainty to margin, growth and
instability probability; central90% intervals use mixture CDF quantiles,
not a moment-normal approximation. Three controls share the population prior:
growth_only observes growth with the learned error model; joint_naive observes
growth/frequency but assumes negligible measurement error; joint_soft observes
both with the learned joint error model. This is a population-informed density
approximation, not a universally valid hard trajectory or exact physical law.
Inference uses measured quantities only. Solver poles are training/diagnostic
targets, checked against label_alpha/label_omega and design/source alignment.

Frozen proper-fit one107/full174/blend50 predictors were reused, not refit.
Their probability-score comparator is only a Gaussian residual approximation
with calibration RMS; do not interpret it as an established Bayesian baseline.
Additional two-input MLPs fit true or measured critical-pole quantities to
distinguish this representation's limitation from the mixture fitting method.
All45 new fits converged:15 physical-prior/error fits and30 diagnostic MLPs.

Mean design MAE below averages the three initialization seeds and, for noisy
rows, both fresh draws; same4454 physical clips throughout:

condition       growth_only joint_naive joint_soft one107   full174  blend50
clean           .040822     .033468     .035217    .009343 .007514 .007625
independent1%   .041416     .034991     .036270    .010688 .009330 .009093
correlated1%    .041630     .035574     .036686    .011555 .010181 .009895
independent5%   .042240     .041153     .038247    .017527 .016785 .015597
correlated5%    .043035     .044444     .039546    .028201 .023940 .024591

Joint information helps versus the growth-only control. On iid1_new949,
joint_soft minus growth_only design MAE=-.005139, fixed-prediction paired
whole-design95% interval[-.006284,-.004049]. Explicit noise modeling helps
against joint_naive under5% noise, but hurts clean/1% aggregate MAE.
Neither joint variant competes with the existing spatial margin predictors.
The descriptive fresh-draw point-MAE checks pass only10/32 versus one107,
12/32 versus full174 and10/32 versus blend50; no replacement qualifies.

True-critical-pole oracle design MAE=.035122 for the mixture and.034513 for
the two-input MLP. Measured two-input MLP on iid1_new949=.034204, versus
full174=.009219. [INFERENCE] The tested single-pole representation loses
important cross-design margin information; better identification alone does
not close this gap. This is not an irreducible-information proof or a ceiling
for richer joint physical inference. No return to the old global~.04-ceiling
claim: the existing richer representation is substantially better.

Soft growth identification is not a general improvement. On iid1_new949,
measured versus joint_soft design MAE=.382238 versus.469723 (1/s);
soft-minus-measured=.087485,[.064678,.109658]. Clean identification also
worsens from about.001 to.1352. On correlated5_new949_950 it improves
1.660477 ->1.588996, delta=-.071481,[-.095782,-.047455]; independent5%
delta=-.013994,[-.034854,+.004667] is uncertain. Error-mixture behavior is
fold-dependent: folds0/4 allocate one component to rare large frequency
substitutions, while folds1/2/3 separate tight and broad residual populations.
[INFERENCE] One global error mixture can over-shrink already accurate poles
and does not provide acquisition-/quality-conditioned measurement uncertainty.

There is a separate warning signal. At the train-calibrated5% stable false-
warning budget on iid1_new949, joint_soft observed stable false warnings=
5.101%, unstable misses=0%, unstable-last1% misses=0%; blend50=4.884%,
.344%,3.498%, respectively. Joint-minus-blend last1% miss change=-3.498
percentage points, paired95%[-6.748,-1.068]. False-warning change=+.217pp,
[-.929,+1.479]: crossing zero does NOT prove false-alarm noninferiority.
The second fresh iid1 draw repeats0% versus3.498% last1% misses. The simpler
growth_only control also has zero observed misses at this budget, so these
results do not establish that frequency or joint uncertainty caused the gain.
At the1% budget, joint_soft misses3.704% of last1% unstable clips on draw949;
growth_only misses0%, with observed false warnings.944% versus1.148%.
On correlated5 draw949, joint_soft's5%-budget false warnings rise to6.083%
versus blend's2.601%, despite fewer misses. No strict risk-preserving win.
Last1% slices contain172 stable clips/58 designs and162 unstable clips/
51 designs. Zero observed events are not a population-zero-risk guarantee;
fixed-design intervals exclude refit uncertainty and multiplicity correction.
These remain clip decisions, not early-warning results on trajectories.

Uncertainty is also mixed. On iid1_new949, raw joint_soft central90% coverage=
90.525%, mean width=.13841; correlated5 coverage falls to86.267%.
Its Brier score=.006637 versus the reference Gaussian approximation's.018368,
but growth_only=.006121 is better still. The instability score can be useful
even when normalized margin calibration is poor; it does not certify the full
distribution or beat a properly trained/calibrated warning classifier.
Finite-sample90% correction of calibration design-max noncoverage scores
raises iid1 joint_soft coverage to100% and mean width to.49866, but stable
safe-declaration availability becomes0%; blend50 retains24.05% availability
at99.993% coverage and width.37591. No useful calibrated-safe advantage.

Observation diagnostics use usable pre-cap seconds, true critical cycles,
abs(true growth)*usable time, visible energy, fit error and clamp state.
On iid1 draw949, lowest/highest visible-energy quartiles have measured-growth
design MAE=.917505/.006354; lowest/highest fit-error quartiles=.004903/.979124.
More cycles or a longer usable record do not monotonically reduce error in
these pooled clips; unstable strongly visible signals and stable weak modes
confound those comparisons.232 clips are clamped versus4222 unclamped.
[INFERENCE] Quality/visibility matters; no stopping rule, cycle threshold,
causal duration benefit or additional-observation policy was established.

Verification: analytical conditional covariance/mixture quantile checks and
an actual32-held-out-clip posterior smoke passed; full inference, source/
reference fingerprints, split alignment, all37-condition analysis, paired
warning comparisons and actual warm inference timings completed. Seed42/
fold0 cached-pole-to-distribution joint_soft median/p95=6.357/8.252ms over
60 clips; unchanged raw waveform extraction is excluded. Evidence:protocol,
experiment.py, fits.json, saved physical/diagnostic model shards,
predictions.npz, analyze_results.py, results.json, inference_measurements.npz
and inference_timing.json in the experiment directory.

Decision: do not replace current margin models or claim a general soft-
constraint/uncertainty improvement. Preserve these results as a mechanism
diagnosis. [INFERENCE] The next useful uncertainty experiment would retain
the successful multi-mode/spatial context and condition the error model on
measurement quality, rather than reducing inference to one pole or sweeping
mixture size. A separate warning-only comparison is promising, but must keep
the growth-only control and require an acceptable false-alarm tradeoff.
Neither direction is a deployment decision. Reserved splits remain untouched;
no production default or objective pivot; GRU remains live.

2026-10-01 continuation: user identified a missing incremental-value test.
The preceding physical experiments compared standalone models against the
blend; they did NOT combine either readout with the blend. Their poor
standalone margin accuracy cannot reject usefulness as additions. User
approved blend-plus-joint uncertainty, blend-plus-soft growth and blend-plus-
both experiments, with learned rather than arbitrary averaging.

Frozen protocol: p4_groups/blend_physical_integration_20261001/protocol.json.
Keep original proper-fit/calibration/outer-held design splits and seeds;
crossfit both base MLPs and the physical prior in three design folds strictly
inside each proper-fit partition. Fit additive ridge residual corrections on
these OOF inputs, not optimistic in-sample base predictions. Reuse original
frozen full-proper components for calibration/held queries. No outer tuning.
Fixed normalized ridge lambda.1 and the existing train-only scaler.

Joint readout contains posterior margin offset/bounds/probability; soft
readout contains growth correction/standard deviation. These are two
readouts of the shared joint physical/error model, not independent underlying
mechanisms. Eight arms include frozen blend, adaptive-uncertainty-only blend,
affine correction, matched seven-input control, joint, soft, both and jointly
shuffled new-block control. Existing measured growth/frequency, quality and
base disagreement are available to matched controls. All adaptive arms share
the same log-residual scale-head fitting/calibration machinery, so generic
recalibration or variance-head gains cannot be credited to paper mechanisms.
Separate calibration thresholds per arm; primary fresh draws949-952; all37
conditions retained. No new simulation, reserved-split scoring or default
change. GRU remains live.

Actual first-fold/seed42 smoke completed all eight arms and37 conditions on
928 outer-held clips; numerical ridge/interval checks, design isolation and
paired-metric/adaptive-interval smoke passed. Original frozen blend point and
raw interval/probability predictions replayed exactly within numerical
tolerance; adaptive-only arm leaves point predictions exactly unchanged.
Full experiment launched; this entry records protocol/smoke, not a final
result or adoption decision.

2026-10-01: full incremental blend experiment completed. This closes the
missing combination test, not merely another standalone comparison.
Eight arms x three seeds x37 conditions x4454 physical clips; all77 original
TRAIN designs held out by the original five outer folds. All90 inner base
MLP fits and45 physical-prior/error fit pairs converged. Verified disjoint
proper/calibration/held design sets and every inner train/query pair across
all15 outer-seed combinations; inner query partitions cover each proper-fit
design exactly once. Frozen blend replay passed, and adaptive-only blend
points remain bit-for-bit unchanged. No outer-outcome tuning or refitting
the frozen full-proper reference components.

Mean design MAE, averaged over seeds and both fresh draws per noise family.
Control = residual recalibration using existing blend/disagreement/measured
pole/quality inputs; it includes no new physical readout.

condition       frozen    control   +joint    +soft     +both
independent1%   .009093   .009012   .009009   .009048   .009045
correlated1%    .009895   .009814   .009803   .009841   .009830
independent5%   .015597   .015663   .015673   .015679   .015686
correlated5%    .024591   .024100   .024052   .024141   .024097

There is a small attributable joint-readout gain under correlated5% noise.
On draw949/950, joint-minus-control design MAE=-.00004924, paired whole-
design95%[-.00007141,-.00002800]. Draw951/952 repeats-.00004780,
[-.00007268,-.00002465]. Average joint improvement versus the frozen blend
is2.193%, but the generic control already supplies1.996%; joint readout
adds only.201% relative to control. Do not credit the whole2.193% to the
paper mechanism. Joint-minus-control intervals include zero on both1%
families and independent5%; the latter has slight point deterioration.
Soft-only is worse than control in all four averaged noise families;
both is worse than joint-only in all four. No supported complementary
point-estimation benefit from the soft readout in this integration.

The small aggregate gain does not solve the boundary problem. On iid1
draw949, frozen versus joint stable/unstable last1% design MAE=
.004066/.003750 versus.004358/.003983. Joint-minus-frozen changes are
+.0002920,[+.0001063,+.0004740] and+.0002325,
[+.0000059,+.0004420]. Soft and both also worsen these slices.
On correlated5 draw949/950, joint reduces overall MAE but increases
unstable-last1% MAE by.0004092,[.0001777,.0006458]. Zero-tolerance
fixed-prediction upper-bound screens over eight fresh conditions pass only
15/48 (joint),14/48 (soft),14/48 (both) versus frozen blend; versus matched
control16/48,9/48,11/48. These screens cover aggregate/stable-last1%/
unstable-last1%/far-stable MAE and stable over.03/far-stable under.05.
No arm passes all; these are not multiplicity-adjusted deployment guarantees.

Uncertainty-score gains mostly come from generic quality-based scaling.
On iid1 draw949, frozen/adaptive-only/control/joint Brier scores=
.018368/.010967/.011840/.011772. Adaptive-only changes no point predictions
and is better than all three physical-readout combinations here. Joint's
incremental Brier gain versus control at independent5% is supported:
draw949 delta=-.00039236,[-.00052994,-.00024165]; draw951=
-.00032678,[-.00047497,-.00017077]. That is a probability-score gain,
not proof of better safe declarations or warning decisions. Gaussian residual
probabilities are approximations, not newly validated physical posteriors.

On iid1 draw949, raw90% interval coverage/mean width is92.524%/.05033 for
frozen blend,89.964%/.04136 for adaptive-only and89.964%/.04179 for joint.
Design-max calibrated stable safe availability is24.05% for frozen blend,
22.61% for control and18.81% for joint; joint-minus-control availability=
-3.800pp, paired95%[-4.888,-2.674]. On correlated5 draw949, joint group
coverage=99.985%, width=.58343 and stable safe availability=1.93%, versus
frozen98.571%,.37591,31.01% and control99.993%,.54205,3.01%.
Higher coverage can be purchased by losing availability; no utility win.

At the train-calibrated5% false-warning budget on iid1 draw949, frozen/
control/joint false warnings=4.884%/3.940%/3.979%, while last1% unstable
misses=3.498%/5.967%/6.173%. On correlated5 draw949, frozen/joint false
warnings=2.601%/3.724% and last1% misses=9.877%/15.844%. Both comparisons
are tradeoffs or deterioration, not the previous standalone zero-miss result.
The standalone warning signal did not survive this tested residual/scale
integration as a clear warning advantage. Zero observed events or intervals
crossing zero are not risk certification/noninferiority.

Warm cached174-feature-row through both base models and distribution
readouts: frozen median/p95=.929/1.191ms, joint7.210/7.733ms,
soft7.274/7.551ms, both7.346/7.792ms over60 held clips. Unchanged waveform
extraction excluded; sham timing omitted because its cross-clip permutation
is diagnostic, not deployable. Evaluation initially completed all paired
metrics but timing failed loading a frozen physical pickle:
AttributeError: Can't get attribute 'ndtr' on __main__. The old shard resolves
that SciPy ufunc in its entrypoint namespace. Added the required ndtr import
to the evaluator; full evaluation and actual warm timing then completed.
No training rerun, exception suppression or reference-shard change.

Decision: keep current blend/default unchanged. Joint readout has a small
repeatable incremental aggregate gain under correlated5% noise and an
independent5% Brier gain, but neither provides the required boundary/
availability/decision improvement. Reject adoption of this residual/scale
combination; do not generalize that rejection to every way of incorporating
physical information. [INFERENCE] Current evidence favors diagnosing the
near-boundary correction/uncertainty tradeoff rather than another unmodified
combination or hyperparameter sweep. No additional experiment selected.
Reserved splits remain untouched, no objective pivot, GRU remains live.
Evidence in p4_groups/blend_physical_integration_20261001/: frozen protocol,
run_experiment.py,45 inner component shards,120 head shards, fits.json,
predictions.npz, evaluate_results.py, results.json and inference timing arrays/
JSON; numerical/actual928-clip smoke and full isolation/finite-output proof
completed. Fixed-prediction2000 design bootstrap retains seeds/copies together
but excludes refit uncertainty and multiplicity correction.

2026-10-01 continuation: read all seven pages of Ueda-Iio-Ikeda,
"Flutter Prediction Using Continuous Wavelet Transform" (2009),
DOI10.2322/tjsass.51.275; full local PDF in papers/ueda2009_wavelet.pdf.
Rendered pp276/279/281 to verify damaged extraction, Eq14 and table signs.
Mechanism-focused assessment recorded in RESEARCH_REVIEW.md Section15,
and added to its verified paper inventory.

Method: real Gabor CWT; F=1/sum|W_real| over2s, at a fixed20.8Hz band from
target-model flutter calculation. NOT reciprocal squared wavelet energy.
Lower-envelope F selects high-intensity response segments because raw
points scatter under intermittent turbulent forcing; then extrapolate
F=0 across target subcritical speeds25-32.5m/s. Around66-86% of measured
37.6m/s flutter speed, although text states a less-than85% goal. Three
repeated wind-tunnel runs on one wing, not structural generalization.
Table1 wavelet errors approximately0%/-10%/-4%; speeds37.5/33.7/36.1m/s.
Inverse spectral peak errors+4%/-5.3%/-11%: wavelets are not better in every
run. Spectral damping failed; random decrement was N.A./-15%/-13%;
ARMA+41%/+24%/+11%. No calibrated uncertainty or warning-risk guarantee.
Visual frequency-coalescence tracking failed below flutter, motivating
the amplitude index. No automatic lower-envelope rule is specified.

Useful lesson [INFERENCE]: local critical-band observability/quality may
explain failures hidden by a whole-record fit, but CWT is not itself the
physical contribution. Current P4 already has growth/frequency, relative
component energy, residual/usable-prefix diagnostics and multi-mode
spatial information. Local persistence/fit consistency would be the
potentially uncovered information, not duplicated global features.
Need evidence that it carries actual boundary-error information.

Cannot directly import raw amplitude trend: their responses have continuing
turbulent forcing; ours propagate fixed poles from random modal initial
conditions and use per-channel51-sample RMS normalization plus a sensor cap.
Mechanism-only numerical check with Eq11-real/Eq14 and actual P4 calibration:
F(10x)/F(x)=0.100000000000 raw,1.000000000000 after normalization;
assertions passed on a synthetic20.8Hz decaying tone. Not a reproduction,
training experiment or dataset score. Relative within-record band evolution
can survive calibration, but is a different observable needing validation.
Do not remove calibration or assume missing modal information is stable.

Target-calculated band and target speed history are extra information;
P4 waveform-only inference needs signal-derived band selection or a
declared prior, never solver critical-frequency labels. Symmetric CWT
uses m-k and m+k samples: any streaming use needs support/edge/latency
accounting; aggregation must follow physical time, not sample count alone.
No new experiment or default change selected. Rejected residual/scale
integration stays out of the active path, artifacts retained as negative
evidence; existing blend unchanged, reserved splits untouched, GRU live.

2026-10-01 continuation: user approved testing the local-observability
lesson from Ueda-Iio-Ikeda. Protocol frozen before scoring in
p4_groups/local_observability_20261001/protocol.json.
Question: do time-local descriptors add information to the existing blend
beyond global band/aperture controls and an equal-dimension shuffled control?
NOT a reproduction of the raw-amplitude index or target speed extrapolation.

Use measured least-stable first/second >=1%-component-energy band frequencies,
cached columns28/31, never solver critical-mode labels. Complex Gaussian
band coefficient, sigma=2/f, truncate at+/-6cycles; complete support only
before first selected-channel cap, at least8 center samples. Six global
and six local descriptors per band; explicit unavailable flags. Local
share quantiles/evolution/persistence, effective band-power time fraction
and log-power departure from a global line. No padding past observed data.
Two-cycle sigma is a fixed short-record adaptation, not the paper's
four-cycle sigma; no width or threshold sweep.

Arms: frozen107/174-member50/50 blend; refit both members with global12
controls; global12+local12; same24 additions but local descriptors shuffled
within proper/calibration/held partitions, with acquisition copies paired.
MLP32->16/alpha1 and four independent-noise training views unchanged.
Original77 TRAIN designs/4454clips/five folds/seeds42/43/44; proper/calibration
split unchanged. Ninety new member fits planned; no residual/scale integration
head, joint physical prior or soft constraint. Reserved splits untouched.
All37 matched acquisitions retained; primary clean and two existing949/951
draws at independent/correlated1%/5%. Previously scored acquisitions:
exploratory, not blind confirmation or additional structural diversity.
Same design-disjoint raw/group intervals and calibrated warning budgets;
quality descriptors are not automatically calibrated confidence.

Behavioral extractor smoke passed: resolved steady versus intermittent
bands, amplitude/time-unit invariance, cap-tail exclusion and unavailable
short/zero/unidentified bands. Cache-aperture census before scoring suggests
complete support for about92%/61% of clean first/second measured bands.
Exact replay extraction/full fitting/evaluation pending at this entry;
availability losses remain in evaluation rather than dropping clips.
No default change; rejected residual/scale integration remains out of active
path, artifacts retained as negative evidence, GRU remains live.

2026-10-01 continuation: completed local-observability experiment in
p4_groups/local_observability_20261001/. All90 new member fits converged,
77 TRAIN designs/4454clips/five folds/seeds42/43/44/37 acquisition conditions.
Full finite-output/design-isolation/source-fingerprint/frozen-point replay
checks passed. Actual928-held-clip smoke covered all37 conditions and
point/boundary/interval/probability/warning evaluation. No reserved scoring.

Mean design MAE across3 seeds and both existing949/951 noise draws:
                     frozen   global12  global12+local12  shuffled local
clean                .007625  .007411   .007359           .007602
independent1%        .009093  .008892   .008924           .009168
correlated1%         .009895  .009780   .009751           .009977
independent5%        .015597  .015478   .015397           .015765
correlated5%         .024591  .024562   .024054           .024662
These are averaged errors, not an ensemble. Correlated5% local-minus-global
MAE=-.000509, paired95%[-.000943,-.000223]; both individual draw intervals
exclude0. Gain2.19% versus frozen/2.07% versus global; equal-dimension sham
also loses. Thus aligned local information adds value in this stress case.
Not uniform across initializations: local-minus-global approximately
+.000004/-.000966/-.000564 for seeds42/43/44. In clean and other three
primary noise families, local-minus-global MAE intervals include0.

Boundary means often improve, but tails/decisions prevent adoption:
Independent1% stable/unstable last1% MAE .003895/.003774 -> .003648/.003225;
unstable-side paired interval excludes0. Yet unstable-last1% optimism
error>.03 rises .309% -> .617%; global controls=.103%.
Correlated1% far-stable underestimation>.05 rises4.756% ->5.335%;
paired difference+.005790 CI[+.000411,+.011610], supported deterioration.
At calibration-selected5% false-warning budget, independent1% unstable
last1% miss3.50% ->1.95%; correlated5%11.01% ->12.45%. Both paired miss
intervals include0, so neither useful warning gain nor established miss
deterioration is claimed. Probability comparator Brier worsens with paired
support in independent1%, correlated1% and independent5%; not a verdict
on every conditional uncertainty formulation. This experiment uses the
existing constant calibration scheme, not a learned quality-dependent
uncertainty/abstention policy.
Strict zero-deterioration screens local-versus-frozen11/48,
versus-global5/48, versus-sham16/48. CI crossing0 is uncertainty, not proof
of harm; tail observations and far-stable deterioration are separate facts.

Quality lesson: global complete-support availability92.0%/61.4% for clean
bands1/2, but both available on every clean and independent1% stable-last1%
clip. High band1 log-power-fit residual associates with larger frozen
boundary error, descriptively, with clean proper-fit quartile cutoffs.
However, D000093/source3696/seed43/iid1_new949 is newly badly optimistic:
true m=-.006458, frozen+.000029, local+.023916; both bands available and
band1 RMSE.001868 is in the LOWEST residual quartile. Repeats on both
independent/correlated1% draws for this seed. Low local residual is not
calibrated confidence or proof of correct critical-mode identity.
Six newly threshold-crossing seed/condition exposures under1% are recorded
for D000093/3696 and D000135/5198 in new_optimistic_tail_cases.json;
exposures are not six independent physical clips.

Measured added extraction median/p953.26/5.33ms; warm cached174-base-feature
plus calibrated waveform through both members/distribution .72/.96ms
frozen versus4.13/8.11ms local. Original normalization/modal/spatial
extraction excluded, not full sensor-to-output latency.

Decision: keep existing blend; do not adopt this direct local-descriptor
augmentation. Local information has a bounded, attributable stress-case
benefit, not a general warning/reliability improvement or a failed-wavelets
verdict. [INFERENCE] Next useful diagnosis is why apparently consistent
observed bands still give optimistic margin errors: mode identity,
cross-design calibration and fitting versus observation insufficiency.
A generic high-residual cutoff would miss D000093; no new cutoff, feature
sweep or further experiment selected. GRU remains live. Rejected
residual/scale integration stays out of active path, retained as negative
evidence; default/reserved splits unchanged.
Methods/results recorded in RESEARCH_REVIEW.md Section16. Artifacts:
protocol.json, local_features.py, contract.py, prepare_features.py,
per-design descriptor shards/features.npz, run_experiment.py,90 model
shards/calibration shards/fits.json/predictions.npz, evaluate_results.py,
results.json, frozen_observability_diagnostics.json, tail cases and actual
inference timing arrays/JSON. Previously scored acquisitions and repeated
exploration; fixed-prediction2000 whole-design bootstrap retains seeds/draws
together but excludes refit uncertainty and multiplicity correction.

2026-10-02: completed the user-approved optimistic-failure audit.
Read-only use of77 original TRAIN designs,4454 clips,37 existing acquisition
conditions, frozen/global/local arms and seeds42/43/44. No fitting, structural
simulation, production/default/objective change or reserved-split scoring.
Protocol: local_observability_20261001/audit_protocol.json. Actual model-shard
replay reproduces both named failures at atol/rtol1e-12; the --require-accurate
CLI deliberately fails on the observed local error>.03, not on missing setup.

Cohort: strict prediction-truth>.03. Across37 conditions:82306 model/seed/
condition exposures,15837 unique condition/clips,2030 physical clips,77
designs. Primary clean+eight949/951 conditions:25578 exposures,4659 condition/
clips,1697 physical clips. Unstable-last1% primary subset:81 exposures,30
condition/clips,12 physical clips,11 designs. These are not independent events;
the broad severe-margin cohort is not synonymous with missed-warning events.

Modal identity: replayed51264 acquisitions (all severe-union acquisitions
plus all primary-condition acquisitions) through actual calibration/capping,
LS-ESPRIT and174-feature encoding. Maximum cached-feature delta8.88e-15.
Metadata label-driving mode is represented and matches retained critical
index0 throughout the primary severe cohort.
Existing frequency-only Hungarian matching reports critical recovery87.85%
severe versus96.92% nonsevere, descriptively. Its broad frequency gate is
ambiguous near flutter: every primary unstable-last1% severe acquisition has
another true pole within twice that gate. Do not interpret those assignments
as uniquely established modal identity.

Added a SECONDARY joint-complex-pole diagnostic after observing that ambiguity:
use growth AND angular frequency in the already encoded energy-top3 and
least-stable visible pair. Solver quantities remain diagnostic-only.
All30 primary unstable-last1% severe acquisitions already encode a pole whose
nearest true pole is critical index0. Distance to the true critical pole,
divided by its separation from the other true poles: median.002210,
p95.011076,max.029901. Critical-alpha absolute error median.04785/s,
p95.22010/s. Missing critical information is not a general explanation for
THIS boundary cohort; this is not a guarantee of full feature sufficiency.
Named iid1_new949 cases:
- D000093/3696: true alpha2.659318/s,f74.447284Hz; encoded2.672759/s,
  74.447870Hz. Complex error/true separation.000294.
- D000135/5198: true alpha2.702966/s,f92.682171Hz; encoded2.708228/s,
  92.682515Hz. Ratio.000126. Measured leading mode also maps to critical0.
Broader primary severe acquisitions differ: nearest encoded candidate maps
to critical0 in80.81%, versus98.08% nonsevere; complex-error/separation
median.06062 versus.000781. Identification/representation fragility remains
part of that broader population. Outcome-selected associations are not causal.

Training versus held designs: predict with actual existing model shards,
not new refits or training logs. Compare original four acquisition types on
proper-fit and held designs; held predictions replay saved values at1e-12.
Pooled design MAE, proper-fit -> held:
frozen .005330 ->.010310; global .005239 ->.010169;
local .005106 ->.010167. Proper predictions repeat designs across applicable
outer models/seeds; held predictions repeat seeds. These are descriptive
in-sample/generalization comparisons, not independent deployment estimates.
Unstable-last1% optimism>.03 is0/6048 proper-fit prediction rows for EACH arm.
Held same-acquisition rows: frozen8/1944 (.412%), global2/1944 (.103%),
local12/1944 (.617%). Zero observed training events is not zero population risk.
Local unstable-last1% design MAE still improves: .001921 fit,.003555 held,
versus frozen .002245/.003986. Mean gains again conceal held-design tails.

Same named iid1_new949 waveform, same initialization seed, other outer models
whose proper-fit sets INCLUDE its design:
D000093 local errors+.00195 to+.00499, versus held+.03037;
D000135 -.00243 to+.00361, versus held+.03152.
All these seen-design local point predictions have the correct negative sign.
These are fresh acquisition views of training physical clips, NOT exact-row
training predictions or controlled leave-one-design experiments; other fit
designs also differ between outer models.

Neighborhood audit: every15837 severe-union condition/clip, all six arm/member
feature spaces, proper-fit designs ONLY. Original four-view train scaling;
inactive columns removed; RMS-standardized Euclidean metric. Collapse noise
copies to each reference physical clip's closest view, then retain five
DISTINCT nearest designs. Verified all references remain in proper-fit sets.
For BOTH named cases, all five design-neighbor margins are negative in BOTH
local member spaces; nearest margins -.007560 (3696),-.006458 (5198).
Their five-design ranges are only.0064-.0083. Conflicting nearby target signs
do not explain these two optimistic outputs under this diagnostic metric.
Local nearest-distance percentiles against CLEAN proper leave-own-design-out
reference distances: D00009398.36%/99.89%; D00013596.78%/98.04%,107/174
base-member spaces respectively. Thin comparable feature support is plausible.
This global clean reference is not a noise/margin-matched confidence baseline.
Across30 primary unstable-last1% severe acquisitions, five-design median
margin is negative in26/30 queries in each local member space. Distances are
nonzero, correlated feature blocks affect the metric, and matching neighbor
labels neither proves feature sufficiency nor excludes finer structural
ambiguity. No exact feature collision or information ceiling was established.

Decision: retain existing blend/default. [INFERENCE] Named and tight-boundary
failures point primarily to cross-design mapping/coverage, not a generic
low-quality band, missing critical mode or universally conflicting labels.
Broader severe errors also involve identification; do not prescribe one fix
for both populations. Next recommended controlled probe: unchanged measured
features and design splits, a design-balanced neighborhood mapping reference
versus the existing learned mapping, evaluated on ENTIRE held-fold populations
with boundary tails, far-stable errors and matched warning budgets. That
tests whether local cross-design interpolation helps without another quality
feature or loss sweep. It is a recommendation, NOT a newly started experiment,
adopted neighbor predictor or warning rule. GRU remains a live learned-mapping
candidate; no architecture decision or restored residual/scale integration.

Evidence in p4_groups/local_observability_20261001/: audit_failures.py,
audit_mapping.py, audit_modal.py and77 diagnostic shards, audit_complex_modes.py,
audit_evidence.py, audit_protocol.json, audit_cohort.json, fitting/modal/
neighborhood/joint-pole result files and arrays, consolidated audit_results.json
with artifact/model/source hashes. Full CLIs and consolidated isolation checks
passed; small numerical check covered frequency permutation, exact growth/
damping matching, label identity and four-view physical-clip collapse.
No production source or permanent tests changed. Previously scored acquisitions
and post-hoc severe selection; no causal attribution, uncertainty intervals,
refit uncertainty, multiplicity correction or calibrated-confidence claim.
Research review Section17 records the diagnosis.

2026-10-02: completed the user-approved unchanged-feature neighborhood
reference experiment, not another feature/loss/architecture sweep.
Protocol frozen before prediction in
p4_groups/neighborhood_reference_20261002/protocol.json.
Existing frozen50/50 107/174-feature MLP blend is the reference; its three
seed predictors, calibration distributions and thresholds are reused exactly.
Candidate uses the SAME two feature spaces and original four acquisition
views of proper-fit designs for scaling/reference-library construction.
Each physical reference clip uses its closest noise view; each design gets
one closest-clip vote. Fixed five distinct nearest designs, median of five
margin labels per member, then50/50 member average. Stable distance ties;
no target-dependent neighbor choice, distance weights, tuning or new MLP fits.

Original77 TRAIN designs,4454 physical clips, five outer folds and all37
conditions. Proper-fit/calibration/held partitions remain design-disjoint.
Candidate calibration uses the original four views of calibration designs
and the SAME clip90% residual interval, finite90% design-max residual interval,
residual-RMS Gaussian probability comparator and stable false-warning budgets
1/5/10/20%. Calibration labels never enter the neighbor library or scaling.
One deterministic candidate, NOT three independent candidate fits. Comparisons
average reference seed errors/events before whole-design bootstrap resampling.
Named failures are secondary descriptive checks, never the evaluation cohort.

Smoke exercised all928 held clips in fold0 at all37 conditions, including
actual calibration, intervals and warning evaluation. Initial smoke caught a
list-versus-array mismatch in the verification path after saving fold0;
normalized the member-reference container and regenerated that shard before
completed smoke. No numerical method/protocol change. Fixed numerical checks
covered direct distances, physical-noise-copy collapse, unequal design sizes,
five distinct votes, stable ties, target-blind neighbor choice, deterministic
candidate/three-seed pairing and fold-specific warning thresholds.
Actual noisy-clip BLAS distances agree with direct distances within1.60e-12;
selected references and median predictions agree exactly in that check.
Full predictions/calibration and evaluation completed all five folds.
Full isolation check verifies every neighbor belongs to proper-fit designs,
five distinct design votes per query, complete outer coverage, finite outputs,
member-average replay and exact preservation of saved MLP predictions and
thresholds. Every calibration false-warning rate respects its budget.

Mean design MAE; primary noise rows average both existing949/951 draws.
Reference averages metrics across seeds, NOT predictions into an ensemble:
| Condition | Existing learned blend | Neighborhood reference |
|---|---:|---:|
| Clean | .007625 | .018420 |
| Independent1% | .009093 | .020815 |
| Correlated1% | .009895 | .021473 |
| Independent5% | .015597 | .028574 |
| Correlated5% | .024591 | .031350 |
Every clean/noise-family paired design-MAE interval is strictly positive.
Independent1% candidate-minus-reference+.011722 CI[+.010881,+.012495];
correlated5%+.006759[+.004700,+.008503]. This fixed interpolation/metric is
substantially worse overall, not merely an unlucky initialization.

The two named iid1_new949 observations do improve:
source3696 true-.006458 -> neighbor-.007159 (error-.000701);
source5198 true-.004086 -> neighbor-.006458 (error-.002371).
Both predicted on the unstable side. Source3696 was a local-augmentation
failure in the prior audit, not a severe>.03 failure of every current frozen
seed; do not conflate those baselines or call these two repairs global proof.

Average tight-boundary accuracy worsens in BOTH signs, with paired support,
in ALL four primary noise families. Independent1% stable/unstable last1%
design MAE .003895/.003774 -> .007043/.006067; correlated5%
.011200/.005998 -> .016440/.010638.
Unstable-last1% observed optimism>.03, reference -> neighbor:
independent1% .309% ->0%; correlated1% .309% ->0%;
independent5% .720% ->.309%; correlated5%1.337% ->2.160%.
The1% paired optimism interval reaches0; do not claim zero population risk
or independently established tail elimination. The5% comparisons cross0.
Far-stable m>=.15 underestimation>.05 strongly worsens:
independent1%4.36% ->29.65%, correlated1%4.76% ->31.51%,
independent5%20.72% ->53.35%, correlated5%15.34% ->53.47%.
All four paired increases exclude0. Independent1% increase+.252895
CI[+.212060,+.294976], correlated5%+.381307[+.346278,+.419171].

At the SAME calibration-selected5% stable false-warning budget:
| Noise | Held stable false warnings, reference -> neighbor | Unstable last1% misses |
|---|---:|---:|
| iid1% | 4.99% ->5.87% | 3.50% ->3.40% |
| correlated1% | 4.57% ->5.51% | 4.12% ->3.70% |
| iid5% | 4.66% ->3.92% | 5.56% ->7.41% |
| correlated5% | 2.79% ->3.27% | 11.01% ->9.57% |
Every paired last1%-miss interval includes0: no warning improvement is
established. Correlated1% false warnings increase with a paired interval
excluding0; other false-warning differences remain uncertain. Matched
calibration budgets do not equalize held false-warning rates.
The Gaussian residual comparator's Brier worsens with paired support in
all four families; intervals, coverage/availability and all warning budgets
are retained in results.json. Strict zero-deterioration screen5/56, including
unstable-side optimism; CI crossing0 is not automatically proof of harm.

Same-run warm cached174-feature row through both members/calibrated output:
MLP median/p95 .278/.424ms; neighbor4.546/6.735ms,150 actual held queries.
Single-row outputs replay batched saved distributions within9.99e-16.
Waveform normalization/modal/spatial extraction is unchanged and excluded;
not full sensor-to-output latency, and not comparable to earlier timing
boundaries without accounting for what they included.

Decision: retain existing learned blend; do NOT adopt this fixed neighborhood
mapping or tune it after seeing the failures. It repairs named signs and
some observed optimism events at the cost of much worse average boundary
errors and far-stable underestimation, without a supported warning gain.
[INFERENCE] Ordinary RMS feature proximity and design-balanced median votes
are not a sufficiently useful cross-design margin rule here; the learned
global mapping supplies useful information beyond this interpolation.
This is NOT evidence that all local methods fail, that current features
are sufficient, or that GRU should be abandoned. Metric, interpolation,
vote balancing and sample coverage remain distinct unresolved mechanisms.
No next learner/feature/loss experiment started. GRU remains live; rejected
residual/scale integration stays out of the active path. Production default
and all reserved splits unchanged.

Evidence: protocol.json, neighborhood_reference.py, five atomic fingerprinted
fold artifacts with scaling counts, member predictions, neighbor rows/distances,
calibration points and partitions, predictions.npz, manifest.json,
evaluate_reference.py, results.json, verification.json and actual inference
measurement arrays/timing JSON in neighborhood_reference_20261002/.
Previously scored acquisitions and repeated exploration;2000 fixed-prediction
whole-design bootstrap, no refit uncertainty or multiplicity correction.
No production source or permanent tests changed. Research review Section18.

2026-10-02: user requested closing each missing existing-blend integration
separately, then combining only retained additions. Frozen pre-full-score
protocol: p4_groups/blend_additions_20261002/protocol.json.
Families: neighborhood; global/boundary-localized noise consistency with
matched custom-supervised control; local-observability late fusion with
global-descriptor control. Physical joint/soft integration already completed
and remains excluded. No reserved split or production-default change; GRU live.

Each readout is separately tested by fixed50/50 late fusion and a TRAIN-only
design-cross-fitted residual ridge on top of the frozen spatial blend.
Matched frozen/affine/seven-column blend-only correction controls; no learned
uncertainty scale, feature/loss/hyperparameter search or cross-family fit yet.
Historical consistency models included the present calibration designs:
refit them on proper-fit designs instead of reusing leaked predictions.
Original107/174 inner base fits and compatible full-proper observability
fits are source/row/hash checked and numerically replayed before reuse.

Actual first-fold smoke passed:928 held clips,37 acquisitions,15 arms;
all24 new neural fits converged, frozen distributions/thresholds replayed,
inner design separation and all1/5/10/20% calibration warning budgets passed.
Complete point/boundary/interval/probability/warning and paired-family
evaluation smoke passed. Two actual raw-waveform replays through all15 arms
passed: feature delta0, calibrated-output max delta7.22e-16. Smoke inference
timing was under concurrent fitting; final timing will be measured afterward.
Throwaway synthetic combination exercise passed empty/single/joint branches,
unchanged retained predictions, actual ridge/equal-weight outputs and immutable
selection hashes; temporary source removed. This is branch validation, not
P4 utility evidence or a keep decision.

Two implementation mistakes were caught before any full-score conclusion:
initial inner partition seed1031 mismatched historical caches; corrected to
their actual1047+fold before fitting. Imported helpers reorder sys.path,
causing generic evaluate_results to resolve to the old experiment; fixed
combination evaluation import to explicit current-file loading and smoke passed.
The original failed partition attempt created no model cache.

Full five-fold/three-seed independent run is in progress. Keep decisions will
be saved with independent-results hash before cross-family fitting/scoring.
Retention requires supported fresh-family gain beyond matched controls and
no supported clean/fresh safety deterioration; unresolved guardrail intervals
are not noninferiority proof. All evidence remains repeated exploratory
TRAIN evaluation, not blind confirmation or deployment certification.

2026-10-02 continuation: completed all independent blend-addition comparisons
and final retained-set run in p4_groups/blend_additions_20261002/.
Full15 arms x3 seeds x37 conditions x4454 clips; original77 TRAIN designs
and five outer folds. All360 new neural fits converged (455–1659 iterations,
no recorded native convergence warnings);45 historical inner base-pair and
60 full-proper observability-member records reused under source/row/hash checks.
120 independent residual-ridge heads. Historical standalone consistency
predictions were not reused because they included present calibration designs.

Frozen criteria retained ONLY learned_local_observability; fixed local fusion
also passed, but protocol preferred learned when both qualify. Independent
results hash/protocol hash and decisions frozen in retention.json before the
final retained-set run. No jointly fitted cross-family model: exactly one new
family survived, so final configuration is existing blend + local-observability
predictor with matched blend-only correction and frozen controls. Final saved
predictions/thresholds replay that individual route exactly; this is not new
confirmation evidence. No rejected additions silently combined.

Mean design MAE, frozen -> retained learned local route:
clean .007625 -> .007115; fresh independent1% .009093 -> .008674;
correlated1% .009895 -> .009470; independent5% .015597 -> .015051;
correlated5% .024591 -> .023454. All four fresh improvements versus frozen
and blend-only correction have paired intervals excluding zero. Attributable
local-minus-global-control gains supported only at5%:
independent -.000088 [-.000165,-.000009],
correlated -.000231 [-.000412,-.000074]. At1% local-vs-global is uncertain;
do not credit local descriptors for the whole gain.

Reject neighborhood: learned add-on has no supported gain beyond blend-only
correction; fixed fusion worsens aggregate errors. Independent1% learned
minus matched control +.0000156 [.0000036,.0000287].
Reject global consistency: fixed route has matched independent5%-MAE gain,
but unstable-last1% misses at calibration5% budget worsen +1.75pp
[+.31,+3.36], plus correlated5% boundary deterioration. Learned route has no
qualifying matched gain and also worsens correlated5% boundary behavior.
Reject localized consistency: no qualifying matched gain; fixed route worsens
correlated5% stable-last1% MAE, learned route worsens independent1% stable
false warnings by +.198pp [+.012,+.432] at calibration5% budget.

KEEP LOCAL ROUTE FOR RESEARCH ONLY, not production adoption. Zero supported
guardrail deteriorations is not safety proof:25/40 intervals retain positive
upper bounds. No supported calibrated boundary-warning gain; unstable-last1%
miss rates at calibration5% budget frozen -> retained:
iid1 3.50 ->3.09%, correlated1 4.12 ->3.70%, iid5 5.56 ->5.86%,
correlated5 11.01 ->12.24%; every paired miss interval includes zero.
Both known iid1_new949 failures worsen: source3696/seed43 truth-.006458,
frozen+.000029, retained+.008037; source5198/seed42 truth-.004086,
frozen+.024748, retained+.029098 (optimism now>.03). Do not claim case repair.

Full waveform/assembly verification passed: all15 arms,3 seeds,5 folds;
frozen distributions and thresholds, isolated proper/calibration/held designs
and all warning budgets. Six actual waveform cases through every arm/seed:
feature delta0, calibrated-distribution max delta2.16e-15. Final retained
predictions/thresholds exact replay and immutable selection hashes passed.
Cached174-feature plus descriptor inference (30 repeats, during final
evaluation): frozen median/p95 .542/.602ms, retained 1.087/1.587ms;
normalization/modal/spatial/local extraction excluded, not total online latency.

Evidence: protocol/run_experiment/evaluate_results/verify_results,
all inner/outer models and points,15 fusion fold shards, fits.json,
predictions.npz, results.json, retention.json, combine_retained.py,
combination.json/predictions/results, verification.json and
final_verification.json. Research review Section19 records full decisions.
Throwaway synthetic checks passed retention gain/harm/control/family-collapse/
immutability and empty/single/joint combination paths; temporary files removed.
No production source or permanent tests changed. Reserved splits untouched,
GRU live. Repeated previously scored TRAIN acquisitions,2000 fixed-prediction
whole-design bootstrap, no refit uncertainty or multiplicity correction:
exploratory evidence, not blind confirmation.

2026-10-02: started the user-requested GRU/TCN audit and controlled comparison.
Protocol: p4_groups/temporal_comparison_20261002/protocol.json. Both models
receive selected16 unsaturated measured prefixes plus existing174 modal/spatial
features and log(reported dt). GRU hidden64, one unidirectional layer; TCN
width32/eight causal residual blocks with per-timestep channel LayerNorm.
The TCN implementation already covers512 samples: actual default receptive
field1021; corrected its erroneous511-sample docstring. Prefix-indexed
early/late/last summaries exclude saturated/padded tails for both models.

Reproduced and fixed train_gru's undefined grouped-split RNG; seed now controls
initialization as well as design splitting. Shared quantile/Huber/GRU training
uses sample-weighted validation, norm1 gradient clipping and best-state restore.
Explicit TCN splits now reject design overlap and return valid test clips only.
Tensor stacking no longer allocates a Torch tensor for each clip before stacking;
quantile GRU no longer constructs and discards a point-prediction head.
Legacy experiment_p4_phase2 resume trainers were not migrated in this change;
the new comparison uses the shared corrected optimization module.

Checks: P4 package suite190 passed/1 skipped. Consumer regressions cover seeded
grouped fitting, minibatch-invariant validation loss, explicit split leakage,
and saturated-tail independence. CPU batch128/T512 synthetic forward+backward:
packed GRU28.41s versus dense GRU0.84s. Use dense causal recurrence and trim
batches to their longest valid prefix; packing was substantially slower here.
These are batch timings, not end-to-end inference latency.

Replayed all37 existing acquisition conditions for4454 clips/77 TRAIN designs;
input shape(37,4454,16,512). All usable-prefix lengths matched cached measured
features. Raw files, cached tables and30 frozen base models pinned by hashes.
Reserved splits and production defaults untouched. Compare standalone GRU/TCN,
fixed50/50 additions and design-cross-fitted residual additions against frozen
blend and matched blend-only correction. Full five-fold/three-seed comparison
has NOT started; no temporal accuracy or warning result is established.

The first smoke stopped before producing predictions; three fit checkpoints
were saved (GRU inner0/1, TCN inner0). Exit cause was not captured. Resumed from
these checkpoints under persistent service p4-temporal-smoke. Supervision now
writes smoke_status.json (command, PIDs, phase, exit code) and smoke_run.log in
the comparison directory. Smoke completion, waveform replay and risk scoring
remain pending; full comparison starts only after the smoke passes.

2026-10-02: CPU temporal smoke completed with exit0; scoring covered928 held
clips,37 acquisitions and8 arms. Calibration budgets and design isolation
passed. Nine real-waveform replays passed; largest reported fused-output
difference was4.172325134277344e-7. Smoke is not a full comparison result.

GPU setup: published P4 source/protocols as d662243 on p4-review-fixes and
cloned into /home/omid5/mechanics on omid5@cachyos.local. Created a project-local
Python3.11.16 environment using uv. GPU is GTX1660 SUPER,6144MiB; driver615.71.09.
Transferred1,057,762,745 bytes of required private comparison artifacts over
SSH, not GitHub. No existing inspected environment had Torch.

Direct PyPI dependency fetches failed on files.pythonhosted.org timeouts.
Using the Tsinghua PyPI mirror resolved29 packages and started the official
Torch2.12.1+cu130/runtime downloads. Installation and GPU execution remain
unverified; full temporal comparison has not started.

At the user's request, persisted exported Fish universal settings on CachyOS:
UV_HTTP_RETRIES=10, UV_HTTP_CONNECT_TIMEOUT=60, UV_HTTP_TIMEOUT=300,
UV_CONCURRENT_DOWNLOADS=2. Verified their inheritance in a fresh SSH/Python
process and that UV_NO_CACHE is not enabled; cache remains /home/omid5/.cache/uv.
Settings apply to new uv processes, not the existing download. Retries must
reuse the cache; do not use --refresh, --reinstall or --no-cache.

Torch retry completed: the first mirror attempt failed extracting
nvidia-nvshmem after a30-second network timeout. Retried with the persisted
settings and existing cache; only13 packages needed preparation, then29
runtime packages installed. Cancelled the obsolete no-dependencies Torch
attempt so it would not compete with the complete installation.

Actual SSH GPU execution passed: Torch2.12.1+cu130, CUDA13.0, CUDA available=True,
NVIDIA GeForce GTX1660 SUPER; multiplying a2x2 CUDA tensor of ones returned
a2x2 tensor of twos. Torch warned that NumPy is not installed. Per the user's
immediate Torch-only scope, remaining P4 scientific dependencies and the full
GRU/TCN GPU comparison have not been installed/run. No system Python or driver
changes were made.

2026-10-02: user requested completing packages and beginning actual work.
Installed remaining P4 dependencies using uv/Tsinghua with the working CUDA
Torch retained. Verified NumPy2.4.4, SciPy1.17.1, scikit-learn1.9.1,
cloudpickle3.1.2 and Torch2.12.1+cu130. Project installed editable.

Actual original TRAIN waveform source0,512 usable samples: GRU and TCN CUDA
forward/backward both passed finite-output/loss/gradient checks. Loaded feature
contract remained77 designs/4454 clips/37 acquisition conditions.

Added CUDA execution to temporal training/prediction/waveform verification;
device is fingerprinted, CUDA forks rejected, CPU path retained. Added a
supervised pipeline which stops at the first failing preparation/smoke/scoring/
verification stage before full training. Launched detached on CachyOS with
P4_DEVICE=cuda and --pipeline --workers1; supervisor PID83708, child83710.
Observed pipeline_status.json phase=running, stage=prepare and the supervisor
ready message. Free storage before preparation:9.172985GiB.

Pipeline proceeds through GPU smoke, complete smoke scoring and real-waveform
verification, then the unchanged full five-fold/three-seed GRU/TCN comparisons,
full scoring and replay. Logs/status live in the temporal comparison directory.
No full GPU model results are available yet; reserved splits remain untouched.

2026-10-03: user challenged the cost of the 120-fit GPU matrix and approved
finishing seed42/outer-fold0 only, scoring it, then stopping. The matrix count was
3 seeds x5 outer folds x2 architectures x(3 inner cross-fits +1 final fit).
Inner predictions prevent leakage into learned fusion, but five folds/three
seeds were confirmation choices, not prerequisites for exploratory screening.
Launching the full matrix before a fully trained first-fold decision was too
expensive for the immediate question.

Armed a detached inotify watcher for fold_seed42_fold0.npz; it sent SIGTERM to
the validated pipeline supervisor after the completed fold was saved. Stop
receipt:2026-10-02T21:32:38.796669Z. Original pipeline status is failed/exit -15
with termination_signal15: intentional user-approved interruption, not an
observed model/runtime failure. No automatic matrix expansion is queued.

Added score_first_fold.py plus --first-fold scoring/verification support.
Training source, optimizer, batch128, selection/refit rules and checkpoint
fingerprints were unchanged; all eight completed neural fits were reused.
Screening outputs are separate screening_* artifacts. Existing evaluators
scored all928 held clips from16 designs,37 acquisitions,8 arms and seed42.
All predictions were finite, design partitions disjoint and calibration
budgets respected. Nine real-waveform replays/18 temporal inferences passed;
largest replayed fused-prediction discrepancy was1.49e-7.
screening_status.json completed with exit0 at2026-10-02T21:34:58.872887Z.
GPU outputs/checkpoints are locally archived in
p4_groups/temporal_comparison_20261002/gpu_first_fold_screening/.

Fully trained design MAE (frozen blend / learned GRU addition / learned TCN
addition):
clean:0.006929 /0.006364 /0.007052
fresh independent1% family:0.007572 /0.006983 /0.007475
fresh correlated1% family:0.008272 /0.007697 /0.007890
fresh independent5% family:0.014158 /0.013582 /0.012723
fresh correlated5% family:0.019392 /0.018730 /0.016865
Standalone GRU/TCN do not establish a broad replacement advantage.

Primary warning tradeoffs at the same nominal5% calibration false-warning
budget: independent5% near-unstable misses are1.5625% frozen,3.125% learned GRU,
1.5625% learned TCN. Correlated5% misses are7.8125% frozen,9.375% learned GRU,
3.125% learned TCN. Corresponding correlated5% held stable false warnings are
2.3897%,1.8382%,2.3897%; nominal calibration budgets are not held-set guarantees.
Learned TCN minus frozen correlated5% miss difference is -0.046875 with paired
design-bootstrap95% interval[-0.136364,0], so the observed improvement is not
clear confirmation. All near-unstable optimism>0.03 paired differences were
zero in these primary families; zero observed events do not prove zero risk.

Decision: no production adoption and no automatic additional training.
GRU fusion has encouraging MAE but a noisy-warning tradeoff; TCN fusion is a
candidate for a bounded confirmation because its high-noise MAE improves and
correlated-noise misses fall in this fold. One held-design group/one model seed,
repeated TRAIN exploration, fixed-prediction bootstrap and no multiplicity
adjustment remain limitations. Reserved selection/calibration/test designs
remain untouched.

Storage: full uv cache cleanup removed31420 files (5.6GiB reported logical
size); verified /home/omid5/.cache/uv absent and no alternate login-shell cache
path. Free home storage increased from4.0 to4.5GiB. Installed Python/tools and
the project environment were preserved; NumPy/scientific imports and an actual
CUDA matrix computation passed after cleanup.

2026-10-03: follow-up interpretation corrected after user feedback. Primary
question remains near-flutter optimistic errors and warning misses at matched
false-warning budgets, exactly as frozen in protocol.json; aggregate MAE is
secondary. The discussion incorrectly pivoted to MAE/model development and
omitted GRU figures. Neither was a justified change of the research objective.

Recomputed absolute optimistic-error rates from archived screening_predictions.npz
and read warning rates from screening_results.json. Near-unstable slice is
-0.01 < true margin <0:32 physical clips from9 held designs. Fresh noise-family
rates average two acquisition views of these SAME clips; these are not64
independent clips/designs. Still one outer fold and model seed42.

Primary optimism criterion: prediction-minus-truth >0.03. Rates (%):
condition            frozen    standalone_GRU    fixed_GRU    learned_GRU
clean                 0         3.125             0            0
fresh independent1%   0         3.125             0            0
fresh correlated1%    0         1.5625            0            0
fresh independent5%   0         1.5625            0            0
fresh correlated5%    0         0                 0            0
Learned TCN fusion also has zero observed events in these five conditions.
Standalone TCN has1.5625% at fresh independent5%, versus0% frozen.
Thus neither learned fusion demonstrates a reduction of this specified tail:
the baseline already records zero events here. Zero-event bootstrap bounds
cannot certify zero population risk.

Correlated5% near-flutter warning misses / actual held stable false warnings
(both %, same NOMINAL calibration budgets):
budget   frozen          standalone_GRU   fixed_GRU        learned_GRU
1%       40.625/0.9191    34.375/0.5515    25.000/0.8272    37.500/1.1029
5%        7.8125/2.3897   10.9375/2.1140    7.8125/1.9301    9.375/1.8382
Fixed50/50 GRU fusion therefore has a useful observed tight-budget warning
signal: fewer misses at1% without increased held false warnings. At5% it
preserves misses while lowering false warnings. Earlier blanket descriptions
of GRU fusion as only a warning tradeoff were too coarse; fixed and learned
fusion must be assessed separately.

Learned TCN fusion in correlated5%: at1% budget, misses17.1875% versus40.625%
frozen, with equal held false warnings0.9191%; at5%, misses3.125% versus7.8125%,
with equal held false warnings2.3897%. This is the clearest observed learned-
fusion warning signal, not proof of reduced optimism>0.03. The5% miss-difference
paired design-bootstrap interval still touches zero (recorded above).
Independent5% at5% budget: frozen/standalone GRU/fixed GRU/learned GRU misses
1.5625/1.5625/1.5625/3.125%; learned TCN also1.5625%.

No new training or production cutover was launched. Preserve GRU as a live
research path; do not choose a model from MAE alone or collapse fixed/learned
fusion into one result. Any next decision must use the primary conditional
errors, warning-budget tradeoffs and guardrails, with the small-sample and
fixed-prediction uncertainty limitations explicit.

2026-10-03: corrected overly negative interpretation of the warning gains.
Flat optimism>0.03 (zero baseline/fusion events) does not negate improved
warning misses; it means no reduction of that particular primary tail was
demonstrated. Fixed GRU and TCN fusions have genuine observed first-fold
warning wins at the pre-existing1% calibration budget. This does not redefine
the frozen primary5% budget or imply all arms/budgets improve.

Reused two_mode_spatial_20260930/analyze_results.py::paired_rate_difference
for1% near-unstable misses, averaging acquisition views before2000 whole-design
resamples. Reconstructed warning decisions matched stored1% rates exactly.
Correlated5% candidate-minus-frozen miss differences (percentage points,
paired95% intervals):
fixed GRU:-15.625 [-25.000,-5.556]
fixed TCN:-26.5625 [-36.667,-15.000]
learned TCN:-23.4375 [-35.000,-10.000]
Independent5%:
fixed TCN:-14.0625 [-21.880,-6.250]
learned TCN:-12.500 [-18.750,-5.357]
These intervals exclude zero within this held-design group, unlike the
previously discussed5% correlated-noise miss interval. They are conditional
fixed-prediction bootstrap results, not multi-fold/refit confirmation or
multiplicity-adjusted tests. Clean/independent1% improvement intervals for
these fusions touch zero; learned GRU does not show a consistent warning win.

The primary-endpoint chat table omitted material guardrails. Clean design MAE
(frozen/fixed GRU/fixed TCN/learned TCN):
0.006929 /0.006733 /0.008283 /0.007052.
Clean far-stable underestimation>0.05 rates (%), true margin>=0.15:
0.6173 /0.6173 /4.3210 /2.4691.
Fixed TCN clean MAE deterioration is+0.001354 with paired95%
[+0.000440,+0.002216]. Its far-stable underestimation change is+3.7037pp
[0,+8.9286]; learned TCN's is+1.8519pp [0,+4.4872].
Correlated5% stable-last1% design MAE:
0.008535 frozen,0.009488 fixed GRU,0.008691 fixed TCN,0.008105 learned TCN.
Thus the warning improvements should be called first-fold wins with specific
accuracy/pessimism tradeoffs, not dismissed and not described as universal
dominance. No new training or production adoption was launched.

2026-10-03: resumed user-approved physical-literature-to-experiment work.
Retain general ML readings; new experiments explicitly exclude TCN and GRU.
A13200-second background wake-up timer was scheduled at the user's request.

Read Bennett1982 NASA-TM-84545 rendered pp1–5,9–10: use joint modal damping
and frequency, not frequency-only coalescence; preserve both modes and heed
second-mode observability and sensitivity. Its normalized flutter function
and pressure sweep are not our normalized speed-distance target. Read the
Matsuzaki2011 publisher abstract/references; full PDF retrieval failed, so
no full-text result/formula claim. Re-read Chajjed's physical formula and
prior assumptions; do not repeat the already tested joint-prior route.
Koopman parameterized nonlinear prediction requires extra physical inputs
and is not selected. Sources and mechanism assessment in research review21.

Frozen before fitting: stability_coordinates_20261003/protocol.json.
Hypothesis: explicit scale-free quartic/Hurwitz coordinates may improve
the cross-design mapping of existing accurate measured dynamics. No new
information or assured safety improvement is claimed. For each pair among
three least-stable visible modes, append four characteristic coefficients,
Delta2,Delta3 and availability; avoid division by summed damping and never
force margin sign. A same-width generic symmetric modal-product control
separates physical organization from merely adding21 nonlinear columns.

Ten new native32->16 MLP fits: two representations x five existing design
folds, seed42 only. Four unchanged training views; proper/calibration/held
design separation, training-only scaling and alpha1. Replace only the174
member of the existing50/50 spatial blend; reuse107 exactly. All37 existing
measured acquisition conditions. No learned fusion or outcome-driven tuning.
Primary endpoints stay near-unstable optimism>0.03 and misses at nominal5%
calibration false-warning budget;1% secondary. Report actual false warnings,
stable/unstable boundary errors, clean MAE and far-stable underestimation>0.05.
Five-fold TRAIN exploration, one initialization, fixed-prediction bootstrap;
reserved selection/calibration/test and production defaults remain untouched.

2026-10-03: completed stability-coordinate probe, not a primary-endpoint win.
Ten fits converged without warnings, five design folds, seed42, all37
conditions. Numerics matched the published flutter function where defined;
swap/time-scale invariance and singular summed-damping cases passed.
Actual waveform replay:15 acquisitions across five clips including3696
and5198; all174 original features replayed exactly, max new prediction
discrepancy2.17e-15. Frozen distributions/thresholds replayed, all proper/
calibration/held design sets disjoint and all calibration budgets respected.
Products/Hurwitz each had193 training-supported columns in every fold.

Near-unstable cohort now162 physical clips/51 designs; near-stable172/58.
Do not compare these five-fold figures to the earlier32-clip first-fold
temporal screening as though they were the same evaluation population.
Rows below are frozen/products/Hurwitz BLENDS, percentages unless MAE:
condition       unstable_optimism>.03  misses@5%        actual_false_warnings@5%
clean           0/1.2346/1.2346         4.321/4.321/4.321 5.088/4.744/4.591
fresh_iid1      .3086/1.2346/1.2346     3.395/3.395/4.630 5.011/4.858/4.801
fresh_corr1     .3086/1.2346/1.2346     3.704/4.630/5.247 4.610/4.399/4.552
fresh_iid5      .3086/1.2346/1.2346     6.481/5.864/7.099 4.533/4.476/4.438
fresh_corr5     .9259/1.8519/2.1605    11.728/11.420/13.889 2.735/2.544/2.850
Fresh families here average the949/951 paired draws of the same clips.
All37 individual conditions plus separate943/945 replication families
are saved in results.json; every warning budget and both standalone/full
and blend arms are reported, not only these selected display rows.

Positive side effects: Hurwitz clean design MAE0.007414 versus0.007696
frozen, change-0.000282 with paired95%[-0.000452,-0.000094].
Fresh iid5 MAE0.015287 versus0.015632; also beats matched product
blend by-0.000208[-0.000384,-0.000032]. Replication correlated5
STABLE-last1% optimism>.03 falls8.1395%->6.1047%, change-2.0349pp
[-3.6191,-0.5435]; latest correlated5 change7.5581%->6.3953%
is uncertain. These are real secondary/conditional gains, not cancellation
of primary harms. Clean far-stable underestimation>.05 increases
1.3648%->2.2333% (+0.8685pp, interval[0,2.0202]).

Both new blends reintroduce clean severe optimistic predictions at
source3696/D000093 and5198/D000135. Hurwitz predictions0.048521/0.037396
versus true margins-0.006458/-0.004086 and frozen-0.000733/0.025244.
No claim of a better boundary model or cutover. Preserve coordinate results
as negative primary evidence and useful MAE/stable-side signals. Next
bounded probe: small physically parameterized correction of the frozen
blend with explicit accuracy/pessimism constraints, not another member reset.

2026-10-03: froze follow-on stability_correction_20261003/protocol.json
before fitting. New question is constrained calibration of the successful
frozen blend, using the same measured Hurwitz coordinates, not resetting
its174-feature member. Reuse15 archived design-cross-fitted107/174 shards;
ignore the physical-prior outputs. Replay OOF/control provenance and values.

References: unchanged blend and pre-existing seven-column residual-ridge
blend_control. Four new affine readouts per outer fold: products/Hurwitz,
each unconstrained/constrained, same28 columns and objective. Twenty small
head fits, no new base MLP, TCN or GRU. Proper designs shuffled2047+fold and
split in half into objective/constraint sets; all copies stay together,
scaling on objective rows only. Objective combines continuous squared
error, positive error on |margin|<0.05, and coefficient L2 regularization.
No sample is selected by outer errors or named failure identity.

Constraints on disjoint proper-TRAIN constraint designs: each training
view's overall MSE and squared far-stable underestimation beyond0.05 cannot
exceed the reference; clean smooth-MAE cannot exceed the reference.
Zero correction is feasible. These are proxy constraints, not a promise
that held MAE/rates cannot worsen. OOF base learners share fitting designs
across readout partitions: do not claim full independent validation of the
entire algorithm or inherit Cotter's guarantees. All outer calibration
designs remain isolated and primary endpoints/guardrails unchanged.
One seed/five TRAIN folds, all37 conditions, no production cutover.

2026-10-03: completed constrained physical-coordinate correction; reject
this readout as a warning improvement. Evidence:
stability_correction_20261003/{protocol.json,fits.json,predictions.npz,
results.json,verification.json}. Twenty affine heads converged. All nine
TRAIN proxy constraints satisfied for each constrained head (minimum
normalized slack -2.66e-11, within numerical tolerance). Analytic-gradient
errors: objective <=5.71e-10; constraints <=1.08e-7. Fifteen real-waveform
replays, exact feature replay, maximum output error2.23e-15. Reloaded
heads reproduce all20 objective/constraint checks and outer calibration.
Frozen/control predictions and thresholds replay; all calibration budgets
respected. No reserved split, production default, TCN or GRU fitting.

Initial run aborted before fitting on an archived baseline-source mismatch.
Exact byte reconstruction proves the sole change is train_gru: replacing
that function with d662243^ reproduces archived SHA256 ba1dccfc... exactly.
This probe never calls train_gru. Added explicit reconstruction verification,
retaining all other source/version/reference checks; proof in fits.json.
No outcome, input, partition, objective or constraint tuning was performed.

Rates below are percentages, averaged over paired fresh acquisition draws.
Columns: frozen / seven-column control / constrained products / constrained
Hurwitz. Nominal5% stable calibration budget, near-unstable -0.01<m<0:
 clean misses: 4.321 / 5.556 / 6.790 / 6.173
 clean held false warnings: 5.088 / 4.935 / 4.323 / 4.055
 iid5 misses: 6.481 / 8.951 / 8.333 / 8.951
 iid5 held false warnings: 4.533 / 4.897 / 5.891 / 5.375
 correlated5 misses: 11.728 / 13.889 / 13.889 / 16.667
 correlated5 held false warnings: 2.735 / 2.946 / 3.405 / 3.137

Hurwitz-minus-frozen correlated5 miss change +4.938 percentage points,
paired design-bootstrap95% [+2.214,+7.911]; versus seven-column control
+2.778 [+0.309,+5.313]. Replication correlated5 also worsens: miss change
+4.012 [+1.219,+6.962] versus frozen. At1% budget, clean misses rise
25.309%->45.062%, paired change +19.753 [+12.856,+26.516]; fresh correlated5
41.667%->54.012%, +12.346 [+6.323,+18.183]. All four budgets/37 conditions,
per-fold results and unconstrained variants retained, not only these rows.

Primary point-error optimism>0.03: constrained heads retain zero clean
near-unstable events, but fresh iid5 rises0.309%->0.617%; correlated5 rises
0.926%->1.235%, uncertainty includes zero. Replication correlated5 Hurwitz
decreases2.160%->1.543%, uncertainty includes zero. No supported improvement
on this primary tail. TRAIN proxy feasibility is not held-risk protection.

Positive side effect: Hurwitz correlated5 design MAE improves
0.024454->0.023464, paired delta -0.000990 [-0.001265,-0.000709].
Products also improve correlated5 MAE, so no clear physics-specific win.
Cost: iid5 MAE worsens0.015632->0.016189, +0.000557
[+0.000364,+0.000751]. Clean MAE0.007696->0.007576 has interval crossing
zero; far-stable underestimation improves clean1.365%->1.117% but worsens
fresh iid5 21.092%->21.774% and correlated5 16.005%->16.563%, with these
rate-change intervals crossing zero. Generalized safety improvement rejected.

Next bounded mechanism probe: warning-specific low-FPR ranking with the
same measured physical coordinates and matched product/seven-column
controls, preserving the frozen continuous margin/interval outputs exactly.
This can improve warning ranking but cannot reduce point-error optimism.
It is an exploratory warning-only comparison, not adoption or another
attempt to claim a lower MAE as a primary win.

2026-10-03: froze physical_warning_rank_20261003/protocol.json before
fitting. Fifteen small affine score heads: seven-column control/products/
Hurwitz xfive folds. Same fixed four training views and verified OOF base
predictions, scaling on all proper TRAIN rows. Unsafe training cohort
-0.05<m<=0; all m>0 stable rows compete. Objective: squared pairwise hinge
against the top5% stable scores plus coefficient L2; fixed frozen-margin
score is the prior. Low-FPR loss changes ranking, not the measured inputs,
base members, continuous margin or intervals. No post-error selection.
All frozen point/interval fields remain exact; new score is explicitly
uncalibrated, not a probability. Standard held-separated1/5/10/20%
calibration budgets and all37 conditions retained. Optimizer/gradient
checks, waveform replay and matched controls mandatory before decisions.

2026-10-03: completed physical warning-ranking probe. Genuine observed
warning progress; no demonstrated superiority of Hurwitz over generic
products and no point-error repair. All15 affine heads converged without
optimizer changes. Maximum objective-gradient finite-difference error
9.32e-11. Proper/calibration/held designs isolated; archived OOF/control
provenance and numerics replayed. All calibration budgets respected.
Saved heads reproduce all15 objective values and calibration scores.
Fifteen actual raw-waveform acquisitions replayed: exact measured
features, max output error1.07e-14, calibration replay error0. Frozen
point/interval predictions are EXACTLY identical for every row/condition.
New warning_score is uncalibrated, not a probability.

Evidence: p4_groups/physical_warning_rank_20261003/{protocol.json,
run_ranking.py,fits.json,predictions.npz,results.json,verification.json}.
Near-unstable primary cohort162 physical clips from51 designs. Noise
family means average the two paired acquisition draws, not independent
new clips. Rates below are percentages. Each cell is misses/actual held
stable false warnings, using the same nominal calibration budget per arm.
Columns: frozen | rank_control(7) | rank_products(28) | rank_hurwitz(28).

Nominal5% calibration budget (primary):
 clean       4.321/5.088 | 3.704/5.394 | 2.469/5.509 | 3.086/4.897
 iid1        3.395/5.011 | 3.395/5.241 | 2.160/5.203 | 2.469/4.916
 correlated1 3.704/4.610 | 2.778/4.839 | 1.852/4.648 | 1.543/4.380
 iid5        6.481/4.533 | 4.630/3.022 | 4.630/3.386 | 4.938/3.386
 correlated5 11.728/2.735| 8.333/2.219 | 7.407/2.525 | 8.642/2.525

Nominal1% calibration budget (prespecified secondary, not a new primary):
 clean       25.309/0.803|11.111/1.224 | 7.407/1.186 |11.728/1.224
 iid1        21.296/0.918|11.420/1.224 | 6.790/1.224 | 8.951/1.262
 correlated1 22.840/0.765|12.037/0.899 | 7.407/1.033 | 8.951/0.995
 iid5        26.852/1.109|20.679/0.497 |18.827/0.631 |16.975/0.669
 correlated5 41.667/0.593|30.556/0.287 |25.000/0.344 |24.691/0.421

Interpretation of primary5% result: Hurwitz has lower observed misses and
held false warnings in all five displayed families versus frozen. Do not
dismiss these observed wins because intervals include zero. But do not
claim established generalization: fresh correlated5 miss delta
-3.086 percentage points, paired design-bootstrap95% [-7.469,+0.641];
false-warning delta -0.210 [-0.633,+0.212]. Replication correlated5:
misses12.654%->8.951%, delta -3.704 [-8.108,+0.329], false warnings
2.659%->2.257%. Fresh correlated5 per-fold Hurwitz misses improve in3,
worsen in1 and tie in1; not fold-uniform dominance.

The1% improvements have stronger fixed-prediction support. Hurwitz fresh
correlated5 miss delta -16.975 points [-24.097,-9.239]; actual false
warnings0.593%->0.421%, difference interval includes zero. Products:
miss delta -16.667 [-25.000,-8.228], false-warning delta -0.249
[-0.453,-0.058]. Hurwitz fresh iid5 misses26.852%->16.975%,
delta -9.877 [-15.361,-4.321], false warnings1.109%->0.669%,
delta -0.440 [-0.748,-0.150]. Replication correlated5 Hurwitz misses
37.346%->23.148%, delta -14.198 [-20.732,-7.894], false warnings
0.574%->0.459%, difference interval includes zero.

Clean1% tradeoff MUST remain visible: Hurwitz misses25.309%->11.728%
with more held false warnings0.803%->1.224%, increase +0.421 points
[+0.079,+0.768]. Products similarly trade clean false warnings for fewer
misses. Calibration constraints hold empirically on calibration rows,
not on held rows or the population. These are not warning certifications.

Matched controls qualify the physical claim. Seven-column rank_control
already improves several warning endpoints. Fresh correlated5 primary5%
Hurwitz-versus-control misses +0.309 points, interval includes zero;
false warnings +0.306 [+0.039,+0.601]. Hurwitz-versus-products misses
+1.235 [-1.725,+4.668], equal observed false warnings. At1%, Hurwitz-
versus-products correlated5 miss delta -0.309 [-3.289,+3.126], not a
supported physics-specific win. Hurwitz beats control's1% correlated5
misses by -5.864 [-9.667,-2.439], but raises false warnings by +0.134
[+0.019,+0.260]. Retain all three candidates and their tradeoffs.

Point-error optimism>0.03, all MAE/boundary errors, far-stable pessimism
and intervals stay unchanged by construction across all37 conditions.
That preserves the existing predictor rather than fixing its rare
optimistic failures. The warning endpoint improves without claiming
improvement in this other co-primary. All37 individual conditions,8
families,1/5/10/20% budgets and zero-threshold margin proxy remain saved.

Decision: retain the warning-ranking route as a useful research candidate;
reject the preceding constrained margin correction. No automatic default
cutover, no primary-endpoint pivot, no claim that Hurwitz is necessary or
best. Next confirmation should replicate these frozen ranking candidates
across model seeds/design-fold uncertainty, not tune the current errors.
This is still repeated TRAIN exploration with one model seed and2000
fixed-prediction design bootstraps: no refit uncertainty or multiplicity
correction, no blind confirmation. Reserved selection/calibration/test
designs and production outputs remain untouched. GRU remains live, but
none was included in these new experiments.

2026-10-03: user asked what would be acceptable for real-world use, then
how our best scores compare, and requested this journal update.
These are PROPOSED engineering targets for an advisory monitor, not
application-derived requirements, aviation certification criteria, a
production approval or a new frozen experiment protocol. A sole protective
barrier would require a separate system-level safety assessment. No new
experiment, threshold tuning or primary-endpoint change authorized here.

Proposed research-to-advisory targets:
- Missed hazardous approaches <=1%, stretch <=0.1%, with warnings early
  enough for intervention. Real acceptance is event/trajectory-level,
  not detection of already-supercritical clips.
- Severe optimism: probability(prediction-truth>0.03) <=0.1% near flutter.
  Evaluate stable and unstable last1% separately; require a statistical
  upper risk bound below the target, not just zero observed failures.
- Overall mean per-design margin MAE <=0.01-0.02; conditional MAE within
  the last5% of flutter <=0.005-0.01. Accuracy remains secondary to
  dangerous tails and warning decisions. Since m=1-V/ucrit,0.01 margin
  error is one percentage point of critical-speed ratio.
- False alarms: illustrative advisory target <=1 alarm episode per10
  operating hours. Actual acceptable rate remains application-dependent;
  define episode grouping and evaluate continuous operation.
- Available warning lead time sufficient for intervention/recovery and
  a buffer; provisional target >=2x measured response time. Include
  acquisition, preprocessing and communication delays.
- Valid estimates >=99% of intended operating time; unreliable
  observations must be unavailable, never silently declared safe.

Provisional CLIP benchmark gate, distinct from deployment acceptance:
near-unstable-last1% misses <=1% at ACTUAL held stable false-warning
rate <=1%, with severe optimism <=0.1% and no guardrail regression.
The existing primary experiment comparison remains at nominal5%
calibration budget;1% remains prespecified secondary. Do not retroactively
relabel that protocol or confuse nominal calibration and held rates.
At one decision/second,1% positive windows means36 flagged windows/hour,
not necessarily36 separate alarm episodes. Clip rates cannot establish
the proposed operational false-alarm or warning-lead-time targets.

Evidence requirement under independent representative Bernoulli cases:
with zero failures, a one-sided95% exact binomial upper bound below1%
requires at least299 cases; below0.1% requires2995. Counts numerically
checked. Overlapping windows/noise copies do not supply independence.
These counts alone do not establish cross-design or mismatch coverage.
For aircraft applications, aeroelastic substantiation remains separate:
https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1043057

Comparison: best PER-CONDITION scores among the latest four five-fold
warning-ranking arms, at nominal1% calibration budget. Rates are percent,
paired fresh-noise family means. Different rows may choose different heads;
this is not one jointly selected deployable configuration.
 condition    head      near-unstable misses / actual stable false warnings
 clean        products  7.407 /1.186
 iid1         products  6.790 /1.224
 correlated1  products  7.407 /1.033
 iid5         Hurwitz  16.975 /0.669
 correlated5  Hurwitz  24.691 /0.421
None satisfies both proposed1% thresholds. Clean/low-noise winning heads
also slightly exceed the actual false-warning target. If strictly requiring
observed false warnings <=1%, the best evaluated settings among these four
arms instead have misses25.309% clean(frozen),21.296% iid1(frozen),
8.951% correlated1(Hurwitz),16.975% iid5(Hurwitz),24.691% correlated5
(Hurwitz). These remain descriptive observed rates, not upper risk bounds.

At nominal5% calibration budget, latest best noisy-condition misses span
1.543%-7.407%, with the corresponding actual false warnings2.525%-5.203%.
Lower misses at a looser false-warning operating point are useful, but
do not satisfy the stricter proposed gate. Earlier fully trained ONE-FOLD
fixed-TCN fusion scored correlated5 misses1.563%/false warnings2.849%;
at nominal1% it scored14.063%/0.919%. Keep that stronger individual
screening result separate from five-fold evidence: one seed,32 near-
unstable physical clips from9 contributing designs, not generalization
confirmation. TCN/GRU remain live; neither was used in the new probes.

Recent lowest overall MAEs across the stability-coordinate, constrained-
correction and warning-ranking probes: clean0.006861(full_hurwitz);
iid1 0.008849(full_hurwitz); correlated1 0.009642(full_hurwitz);
iid5 0.015287(blend_hurwitz); correlated5 0.023057(hurwitz_unconstrained).
These are per-condition score minima, NOT accepted replacements.
Some lowest-MAE arms worsen optimism or warnings. Never combine one
arm's best accuracy with another arm's best safety score as an achieved
system. Overall accuracy is within the proposed range except correlated5
stress, but that does not establish acceptable boundary reliability.

For the working frozen blend retained by all warning-ranking heads:
stable-last5% conditional MAE clean0.005584, iid5 0.014198,
correlated5 0.032053 versus proposed0.005-0.01. Severe optimism>0.03
on unstable-last1%: clean0%, iid1/correlated1/iid5 0.309%,
correlated5 0.926%. On stable-last1%: iid1/correlated1 0.291%,
iid5 2.616%, correlated5 7.558%. These noisy rates exceed proposed0.1%;
zero clean observed failures are not proof of risk below0.1%.
Warning ranking leaves these point-error rates unchanged.

Sources: physical_warning_rank_20261003/results.json;
stability_coordinates_20261003/results.json;
stability_correction_20261003/results.json;
temporal_comparison_20261002/gpu_first_fold_screening/screening_results.json,
all under p4_groups/. Operational alarm episodes/hour, usable lead time
and availability remain unmeasured. Noise percentages are sensitivity
settings, not calibrated hardware specifications. Current evidence is
simulated, exploratory TRAIN-design CV, not blind real-world validation.
Decision from discussion: useful accuracy and genuine warning progress;
warning misses and particularly stable-side optimism remain the large
gaps. Targets stay provisional until linked to the intended intervention.


2026-10-03: user supplied a second-agent research brief and asked to begin
independent W1/S1/N1 screens, preserving the existing benchmark and defining
O1 alongside them. Re-read Section22 of RESEARCH_REVIEW.md, latest journal,
ranking/training/modal/acquisition contracts. Verified the two reported
diagnostics directly from archived predictions: products/iid1 has64 stable
flag exposures,63 in0<m<.01; Hurwitz/correlated5 has80 near-unstable miss
exposures, only3 with point error>.03. Exposures are two views of clips,
not events.172/2614 stable clips are in the last1%; all-stable1% specificity
permits at most15.198% average precursor-window coverage if every allowed
flag lands there. Keep instantaneous classification labels unchanged.
No retrospective conversion of stable flags into successful advisories.

Frozen protocols and completed independent experiments:
p4_groups/independent_gap_screens_20261003/{protocol.json,
trajectory_protocol.json,warning_screen.py,tail_screen.py,noise_screen.py,
comparison.json,verification.json,mechanism_smoke.json}.
Two prespecified TRAIN screens: heldfold0/seed42 and heldfold1/seed43.
Both run regardless of first-screen results; all37 existing conditions
reported, fresh and replication noise families separate.1808 held physical
clips from32 designs; stable last1%66 clips/24 designs; unstable last1%
58 clips/18 designs. No reserved selection/calibration/test designs, new
TCN/GRU training, production changes or combination of new mechanisms.
Archived reference/ranking and all warm starts remain seed42; seed43
changes W1 initialization and N1 refits, NOT the deterministic S1 warm
start. This is not independent two-seed confirmation of the entire system.

W1: same28 products/control inputs; affine and32/16 nonlinear boundary
heads, with ordinary nonlinear classification control. Point/interval
fields remain bit-identical. Separate prospective1%/5% boundary objectives:
design-balanced near-unsafe softplus loss, per-view smooth stable-rate
constraints, fixed temperature, L2,300-epoch primal-dual Adam screen.
Inner design indicators select epochs; untouched calibration designs set
final thresholds. Every OOF margin member excludes its queried design,
but member fitting overlaps across readout halves: no inherited fully
independent constrained-learning guarantee. Neither fold has ANY boundary
checkpoint satisfying every inner per-view actual indicator budget.
Several final smooth constraints also remain violated. These are failed
constraint-feasibility screens, not solved/certified constrained policies.
Do not hide this behind successful final empirical calibration.
Ordinary classification uses the same score for both calibrated budgets;
its two policy artifacts are not independent learned mechanisms.

Pooled fresh correlated5 results, percent; nominal5% PRIMARY calibration.
Each row is one complete arm, never a mixture of per-condition minima:
 arm                     near-unstable misses / actual stable false flags
 frozen blend             8.621 /2.815
 existing products rank   8.621 /1.622
 W1 boundary nonlinear5   0.000 /3.244
 W1 ordinary nonlinear    0.000 /2.624
 S1 average-tail          8.621 /2.958
 S1 worst-group tail      8.621 /4.008
 N1 full-covariance own   3.448 /3.340

W1 ordinary nonlinear is the useful capacity-control result, not evidence
that the constrained boundary objective is necessary. Versus frozen,
fresh correlated5 miss delta-8.621 percentage points, design-bootstrap95%
[-16.964,-2.026]; false-flag delta-0.191 [-0.937,+0.527].
Replication correlated5 misses9.483->1.724%, false flags2.767->2.433%;
miss delta-7.759 [-13.000,-2.885]. Observed nominal5% misses/false flags
also improve or tie versus frozen in clean/iid1/correlated1/iid5.
Against existing products, correlated5 recall costs extra false flags;
do not claim dominance over the strongest ranking reference.
At nominal1%, boundary nonlinear1 gives fresh correlated5 misses1.724%
and actual false flags1.145% (replication1.724/1.431): fails the proposed
actual<=1% gate. Ordinary nonlinear gives13.793/0.668% fresh and
14.655/0.620% replication. Point optimism remains entirely unchanged.
Decision: retain the ordinary nonlinear28-input warning candidate for
fixed-protocol full-fold confirmation; no approved constrained-head win
or automatic richer174-input stage.

S1: freeze107; fine-tune174 on actual50/50 blend with unchanged features.
Matched average-MSE, average CVaR90 positive-error tail and worst-group
tail objectives; equal design mass and four views crossed with stable/
unstable last1% and1-to5% bins. Fixed regularization/proximal penalty.
Archived inner part0 pair excludes ALL inner validation designs from BOTH
members; identical optimizer/checkpoint candidates, inner-selected epochs,
then proper-fit warm-start training and independent recalibration.
Fresh correlated5 stable-last1% optimism>.03: frozen8.333%, MSE9.848%,
average-tail8.333%, worst-tail7.576%. Worst-tail is fold-sensitive:
fold0 worsens4.688->7.812%, fold1 improves11.765->7.353%.
It introduces unstable optimism0->0.862%, actual5% false flags
2.815->4.008%, unchanged8.621% misses; far-stable pessimism beyond.05
14.474->14.803%. Clean designMAE0.007017->0.007184.
IID5 stable severe optimism0->0.758% and primary misses3.448->6.897%.
Replication correlated5 worst-tail stable optimism7.576->6.061% but
unstable optimism2.586->3.448%;1% misses38.793->47.414%.
Decision: reject this fixed S1 screen; no guardrail-preserving repair,
no broad claim that every nonlinear tail objective must fail.

N1: independent4096-sample synthetic quiet covariance under the existing
iid/rho.8 simulator model,5% diagonal shrinkage. NOT hardware calibration.
Transform covariance through actual leading51-sample channel scales.
Original calibrated signals establish the cap prefix before whitening;
shared poles identified on weighted channels, coefficients/energies/shapes
refitted on ORIGINAL calibrated channels. No across-sensor mean subtraction.
Diagonal/full estimates and known-injected-covariance diagnostic oracle;
fixed-predictor replay separated from own-extractor107/174 MLP refits.
All12 new spatial refits converged; unweighted controls reuse converged
archived models at matched seed42/43.77480 acquisitions, three weighted
extractors, all observed identification availability100%.

Fresh correlated5 stable-last1% optimism>.03: frozen8.333%, matched
unweighted7.576%, diagonal-own3.788%, full-own0%, oracle-own1.515%.
Full-own versus matched unweighted delta-7.576pp,95%[-12.903,-3.448].
Replication full-own0.758% versus unweighted6.818%, delta-6.061pp
[-12.509,-0.714]. Full-own also removes matched unstable severe optimism
0.862->0% fresh and2.586->0% replication. Zero observed is not a risk
bound: only66 stable boundary clips, dependent views and24 designs.
Correlated5 far-stable pessimism14.474->6.086% versus frozen; secondary
designMAE0.021120->0.011620. No accuracy-first acceptance pivot.

N1 warning/guardrail tradeoffs remain: fresh nominal5% misses
8.621->3.448% versus frozen with false flags2.815->3.340%;
versus matched unweighted7.759->3.448%, false flags3.292->3.340%.
Replication versus matched unweighted misses8.621->0.862% but
false flags2.624->3.387%, increase+0.763pp CI[+0.048,+1.524].
Fresh1% misses43.966->18.966%, false flags0.573->1.240% versus frozen.
IID5 full-own introduces stable severe optimism0->0.758%;
versus matched unweighted primary misses1.724->2.586%, false flags
4.389->4.485%. Clean pooledMAE improves slightly, but clean1% misses
12.069->13.793% and false flags1.431->1.813% versus matched control.
Decision: retain full-covariance identification as a strong correlated-
noise tail research candidate; not a general guardrail-preserving cutover.
Oracle covariance does not automatically make the refitted mapping safest.

Verification: actual programs run with .venv/bin/python (system Python
lacks sklearn).18 actual raw-waveform replays across stable boundary,
unstable boundary and far-stable clips, both folds, clean/iid5/correlated5.
Maximum output discrepancy1.42e-14;32 saved-head calibration-threshold
replays exact. All prediction arrays finite, design partitions isolated,
all empirical calibration budgets respected; W1 continuous fields exact.
Warm start reproduces archived nonlinear output; identity covariance
refit matches original modes/shapes; original saturation prefix retained;
all37 measured waveform conditions replay exactly; fractional-mass CVaR
finite-difference gradient error6.99e-11; covariance whitening identity
error1.22e-15.2000 fixed-prediction whole-design bootstraps, no refitting,
multiplicity control, blind validation or population-risk guarantee.
CPU: AMD Ryzen5 5500U, torch2.12.1 CPU. Four concurrent extraction workers;
full-covariance extraction+feature p95~17ms,p99~22ms,max567ms.
These omit normalization/I/O and are NOT full-pipeline deadline evidence.

O1 specified prospectively, NOT implemented or exercised. Continuous
assembled-matrix state propagation; slow/fast crossing, safe/near dwells,
approach-retreat; single continuous noise stream and causal windows;
crossing lead-time horizon sensitivity, clear/rearm alarm episodes,
unavailability/false-safe and delivered latency. No state resets or stitched
clips. Existing clip labels stay unchanged; near dwell/retreat flags are
neither automatically nuisances nor successful anticipation. Fixed1000Hz
sampling requires bandwidth validation; hypothetical0.5/1/2/5s horizons
are not intervention requirements. Physical harmful-response limit,
response/communication delays and hardware noise PSD remain application
prerequisites. Next useful actions: full-fold fixed warning-capacity and
covariance confirmations separately, and implement/validate O1 before any
operational early-warning claim. Do not combine mechanisms yet.


2026-10-03: benchmark-contract clarification agreed with user; record this
and retest after changes. Earlier "keep the benchmark unchanged" was too
broad. Preserve historical comparability, NOT a mismatched acceptance
criterion for the early-warning claim.

Separate three tracks:
1. Versioned legacy clip instability detection, with unchanged labels,
   thresholds, nominal5% primary and1% secondary comparisons.
2. Stable/unstable conditional margin accuracy and positive-error tails,
   retaining clean accuracy and far-stable pessimism guardrails.
3. Continuous-trajectory advisory warning, decisive for the early-warning
   claim once implemented and exercised: timely event misses, delivered/
   usable lead, prospectively defined nuisance episodes and availability.

A flag while m>0 is a false positive for current instability detection,
but may be useful anticipation. Do not infer that usefulness from clips.
Near-boundary dwells and approach-retreat trajectories must challenge the
advisory contract; neither every near-boundary flag nor a uniformly
pessimistic predictor is a success. The proposed1% clip misses/1% all-stable
flags gate is a research detection criterion, NOT advisory deployment
acceptance. Define event/intervention/nuisance rules before scoring or
selecting trajectory policies; do not retroactively relabel saved results.

Implementation-change requirement: retest affected benchmark/model paths
after edits, rerun the legacy comparison track to expose regressions, and
exercise actual propagated trajectories/alarm transitions for O1 changes.
Documentation/specification changes alone do not establish operational
performance. This update records the decision; O1 remains unimplemented.

Post-edit retest completed with .venv/bin/python verify_screens.py and
compare_screens.py:18 actual waveform replays pass, maximum output error
1.42e-14;32 calibration-threshold replays exact; legacy screen comparison
recomputed successfully. Updated decisions.json and trajectory_protocol.json
parse as valid JSON. Changes are documentation/prospective-contract only:
no model, executable benchmark implementation, historical label, threshold
or frozen experiment protocol changed. O1 trajectory behavior was NOT
tested because its implementation does not yet exist.


2026-10-03: implemented independent full-fold continuation and began O1
runtime validation. Scope remains TRAIN only; no combined warning/covariance
model, temporal retraining, reserved-design scoring or production cutover.
Execution protocol: p4_groups/operational_confirmation_20261003/protocol.json.
The original two screens/models are reused exactly. Remaining folds2/3/4
use the frozen objective/settings and prescribed seeds44/42/43; this is
five-fold continuation, NOT three full seed replicates or blind confirmation.

Completed clip evidence: confirmation_results.json, confirmation_fits.json,
confirmed_predictions.npz in that directory. All4454 physical TRAIN clips
and77 designs are represented. Family rates average two paired fresh noise
views, not independent physical events. Nominal5% calibration policy:

                                    IID5 miss/false    correlated5 miss/false
frozen blend                         6.481 /4.533       11.728 /2.735
products ranking                     4.630 /3.386        7.407 /2.525
ordinary nonlinear warning           2.160 /3.864        2.160 /3.022
matched-seed unweighted refit        5.556 /4.457       12.037 /2.774
full-covariance identification       4.321 /4.916        7.099 /3.424

Separate replicated correlated5 draw: frozen12.654 /2.659; ordinary
nonlinear5.247 /2.850; covariance5.556 /3.577; matched unweighted
13.272 /2.429. These rows are complete arms, not mixed per-condition minima.
Warning capacity improves near-unstable recall without changing point/
interval outputs. Versus products it purchases some recall with false flags:
correlated5 miss change -5.247pp, paired95%[-8.544,-2.238]; false flags
+0.497pp,[+0.203,+0.794]. Versus frozen the miss change is -9.568pp,
[-14.826,-5.063]; false flags +0.287pp,[-0.196,+0.803].

Stable-last1 severe optimism P(prediction-truth>0.03): correlated5 frozen
7.558%, matched unweighted6.395%, covariance0.872%. Covariance-minus-matched
change -5.523pp,[-9.122,-2.247]. Replicated correlated5: matched7.849%
versus covariance0.872%, change -6.977pp,[-11.389,-3.205]. IID5 matched
2.326% versus covariance0.581%; its severe-tail interval includes zero.
Correlated5 covariance design MAE0.013941 versus matched0.024723.
Covariance false flags rise against matched unweighted: correlated5
+0.650pp,[+0.114,+1.178]; replication +1.148pp,[+0.511,+1.838].
This is a real acquisition-side improvement, not a free warning improvement.

New folds ONLY, excluding both original screens:2646 clips,45 designs;
new_folds_only_results.json. Correlated5 ordinary warning misses3.365%
versus frozen13.462%; replicated draw7.212% versus14.423%. Covariance
stable severe optimism1.415% versus matched5.660%; replicated0.943%
versus8.491%. Improvements therefore are not solely the reused screens.
The old five-fold scoring helper assumes all five folds are present;
new-fold summaries use its existing slice/rate comparisons without inventing
empty-fold observations.

Tight nominal1% policy remains difficult: correlated5 near-unstable misses
41.667% frozen,16.975% ordinary warning,29.012% covariance. No advisory
deployment gate is inferred from these clip rates. Retain both mechanisms
separately; retain rejection of the tested S1 formulation. GRU remains live.
Uncertainty:2000 fixed-prediction whole-design bootstraps, no refit/
multiplicity/selection correction; dependent views and repeated TRAIN use.

Post-change clip verification completed:60 actual raw-waveform replays,
15 exact calibration replays across all five folds; maximum output error
1.07e-14. Warning point/interval fields remain EXACTLY frozen. Legacy
18-waveform/32-calibration replay and comparison also completed; P4 tests
passed196, skipped1 before adding the final alarm-accounting regressions.

O1 implementation: operational.py plus trajectory_physics.py and
trajectory_runner.py. Continuous64-state structural32-coordinate ROM;
original velocity-dependent assembled matrices, no modal resets or growth
clamp; fixed1kHz observations,512-sample trailing windows/100ms updates,
one leading51-sample calibration, continuous shared noisy samples and
physical transverse-load pulses. Design groups:46 fit,7 policy,8 calibration,
16 trajectory-held; no archived model trained on trajectory-held designs.
Five profiles, three initial states, three excitation families and nine
sensor conditions are retained; acquisition failures stay in denominators.
Numerical identification availability is NOT calibrated correctness confidence.

First O1 execution cancelled without a scorecard: forced response needed
full-order validation, and false-safe duration must use latched alarm state
instead of each negative window flag. cancelled_trajectory_run.json records
this; its partial root-directory artifacts are excluded. Corrected execution
writes o1_validated/. Added fixed/ramp/forced-ramp full-order Radau and
10ms/5ms step comparisons at the actual1kHz output cadence. Actual D000053
forced-response relative RMS discrepancy1.090%, step-halving0.396%;
pole relative gaps below3.7e-6. Its declared audit passes. This is short
numerical validation, not full-mission or hardware fidelity certification.
The earlier D000100 audit exposed retained-mode bandwidth436Hz, outside
the fixed400Hz qualification; it was marked unavailable, not quietly removed.
Eight operational regression cases pass, including alarm latch/clear/missing
data accounting. Full corrected O1 population scorecard is still running;
no operational result or acceptance claim is recorded in this entry.

Continuation guardrails checked from saved complete-arm predictions:
clean design MAE0.007696 frozen,0.007709 matched unweighted,0.007687
covariance. Far-stable underprediction error>0.05: clean1.365% for all
three; IID5 matched21.588% versus covariance16.315%; correlated5
14.950% versus7.878%; replicated correlated5 15.819% versus7.258%.
Correlated5 unstable-last1 severe optimism1.235% matched versus0.617%
covariance. Thus the observed stable-tail repair is not merely a global
downward shift that worsens far-stable pessimism.

Final retest exposed a test-isolation regression: importing the experimental
trajectory runner in the new alarm-accounting tests leaked its shared
torch.float64 default into existing float32 TCN/GRU consumers. Whole P4
suite:13 failed,185 passed,1 skipped. Corrected that fixture to restore
Torch dtype/thread count, thread environment and sys.path after use; no
model casts, suppressed failures or removed validations. Post-fix full suite
and five-fold/legacy waveform replays are running again.

Post-fix retest completed:198 passed,1 skipped. Five-fold waveform/calibration
replay again passes60/15, maximum output error1.07e-14; legacy replay
passes18/32, maximum1.42e-14, and its comparison recomputes successfully.
The fixture isolation fix closes the13 failures without changing any model.
Actual measured-waveform causal smoke: on a continuously propagated D000053
fast crossing with physical pulses and correlated5 noise, changing samples
from800 onward leaves all three prior window predictions/identification
outputs EXACTLY unchanged; injected dropout remains missing/unavailable.
Evidence in causal_waveform_smoke.json. That small smoke uses the original
cancelled-run fits of the unchanged baseline algorithm, not the pending
corrected population scorecard. The full O1 run remains the next dependency.

2026-10-06: continued O1 numerical repair. Corrected forcing propagation uses
an exact first-order hold of linearly interpolated output-step loads, rather
than midpoint-held forcing. Production coefficient updates are2ms, compared
against1ms refinement and full-order Radau references; observation sampling
remains1kHz. No fidelity, bandwidth or Mach qualification was relaxed.
Whole-partition preflight completed for31 policy/calibration/held designs:
no numerical failures; largest production/full-order relative RMS0.20545%,
largest step-halving relative RMS0.16790%. Supported counts5/7 policy,
8/8 calibration,14/16 held. D000092,D000100,D000039,D000059 remain unavailable
because retained-mode bandwidth exceeds400Hz; their trajectories remain
in the evaluation denominators.

Both fit-design baseline members converged:107-input1048 iterations,
174-input1003 iterations; no captured warnings. Repaired D000068 physical-pulse
causal smoke produced1513 finite samples; altering samples from800 onward
left three earlier prediction/identification windows exactly unchanged.
Injected missing samples made five affected windows unavailable. Full P4
regression suite passed199 tests, skipped1; operational subset passed9.

User reports SSH GPU unavailable now. No further SSH execution attempted.
The attempted isolated remote runner produced no trajectory scorecard before
that instruction. Resumed the unchanged corrected benchmark locally as
service p4-o1-local-corrected; observed preflight completion5/8/14.
Outputs remain in operational_confirmation_20261003/o1_corrected/.
Calibration, held operational scoring and completed-population replay are
pending; no operational performance or acceptance claim yet.

User-requested resumable pause: stopped local generation after the second
four supported designs completed. D000087,D000042,D000045,D000126 each
saved405 views/159975 inference windows. Ten whole-design checkpoints
are retained in o1_corrected/, including two unsupported-bandwidth records.
Evidence: requested_pause.json. An event-driven filesystem watcher validated
the four archives and matching fingerprints before terminating the owned
benchmark process group; exit143 is this intentional pause, not a numerical
failure. The full-run replay watcher was disabled before pausing.

All ten saved archives read successfully; the actual trajectory_design
resume path accepted and skipped all ten under the unchanged current
source/input/environment fingerprint. Audits and fitted baselines remain
available. Resume with the same trajectory_runner.py command; unfinished
designs restart at their whole-design boundary. Calibration, held scoring
and completed-population replay remain pending. No operational result claimed.

2026-10-06: user requested continuation and SSH GPU check. SSH to
omid5@cachyos.local works; GTX1660 SUPER6144MiB, Torch2.12.1+cu130,
CUDA available. Actual float64 CUDA2x2 multiplication returned twos.
All ten local design archives validated; current source/input/environment
fingerprint matches, and the actual trajectory_design resume path accepts
and skips all ten.

GPU kernel probe used32 actual D000068 slow-cross measured windows with
correlated5 noise, float64 covariance construction and symmetric eigensolves.
Five timed repetitions after warmup, median remote CPU73.800ms versus
CUDA28.881ms, including transfers:2.555x kernel-stage speedup. Relative
covariance/spectrum discrepancies6.45e-15/1.46e-15. Evidence:
operational_confirmation_20261003/gpu_kernel_probe.json. This is NOT full
modal-estimator, propagation, predictor or causal online-latency qualification.
No CUDA estimator cutover or precision change made.

Transferred complete source, frozen inputs, checkpoint/raw archives and
archived signature/replay dependencies to /home/omid5/p4_o1_execution_20261003
on cachyos. Its actual trajectory_runner --preflight-only passed5/8/14.
Resumed the unchanged CPU numerical baseline for the remaining21 designs,
with automatic calibration, scoring and verify_trajectories replay afterward.
Persistent remote systemd user unit: p4-o1-corrected-resume.service;
observed active/running. remote_execution.py records ten reused checkpoints
and inference host by design in o1_corrected/execution.json. Policy and three
calibration designs retain Fedora timings; remaining calibration/held designs
use cachyos timings. Mixed-host calibration/availability and held-host timing
are not deployment deadline evidence.

Background completion waits on the systemd unit, then retrieves o1_corrected/
to the local experiment directory. No manual polling needed. Generation and
final operational evidence are pending; no completed scorecard claim yet.

2026-10-06: user requested stopping the CPU-only continuation and resuming
with actual CUDA computation. Stopped p4-o1-corrected-resume.service; ten
completed design checkpoints were preserved. Replaced the temporary
remote_execution.py driver with gpu_execution.py and cuda_backend.py.
Propagation/matrix exponentials, modal covariance/eigensolves, LS-ESPRIT
pole/fitting linear algebra and frozen MLP inference use CUDA float64.
CPU work remains for acquisition setup, small feature assembly and I/O;
there is no numerical CPU fallback in the CUDA backend.

Final deployed-source validation: o1_corrected/cuda_validation.json.
49 actual/boundary signals, maximum margin difference 4.13414e-12 and
maximum feature difference 4.53025e-9 versus the CPU numerical reference.
Continuous initial-only, physical-pulse and weak-critical propagation
through a velocity ramp differed by 7.46633e-13, 5.53245e-13 and
5.67862e-13 relative RMS. Three earlier windows remained exactly unchanged
under future perturbations; five missing windows remained unavailable.
Original physical/Mach/bandwidth qualifications were not relaxed.
Existing P4 suite: 199 passed, 1 skipped.

Persistent remote service p4-o1-cuda-resume.service is active/running.
Its MainPID 149876 appeared in nvidia-smi as the Python CUDA process,
using 1726MiB; sampled total GPU utilization was 58%. This proves actual
GPU workload, not an end-to-end speedup or deployment-latency claim.
Startup accepted and skipped all ten saved designs. The retrieved startup
log shows new D000158 slow-cross acquisitions, including physical forcing.
o1_corrected/execution.json records host and CPU/CUDA backend by design:
old CPU checkpoint timings remain unchanged. Nine alternative noise views
are batched at the same trailing-window timestamp, never across future
windows; each view receives the full shared-batch elapsed time.

gpu_execution.py automatically calibrates/scores the completed population
and invokes verify_trajectories.py using the same batched CUDA inference.
A persistent local watcher, p4-o1-cuda-completion, is armed on the remote
PID via pidfd; it requires the successful scoring/replay completion marker
before retrieving o1_corrected/. Completion notification is attached.
Generation, final operational scorecard and population replay remain
pending. No completed operational result claimed.

2026-10-07: status check found the CUDA service inactive/dead after successful
completion, not a failed or paused generation. cuda_execution.log ends with
O1_CUDA_BENCHMARK_AND_REPLAY_COMPLETE: 6480 held trajectory views scored;
42 raw acquisition replays passed with maximum score error 0.0. Both
empirical policy calibration and held scorecard recomputation are exact.
Two unsupported held designs, D000039 and D000059, remain in denominators.
This is trajectory-held evaluation within the original TRAIN population;
the reserved selection/calibration/test populations remain untouched.

User requested resumption. Resumed completed-output retrieval rather than
regenerating successful trajectories. The old completion watcher had not
retrieved outputs when checked; it was stopped after manual retrieval began.
Final trajectory_scorecard.json, trajectory_calibration.json and
trajectory_verification.json were retrieved first; full archive sync follows.

Initial operational findings are poor, despite computational verification.
Under the calibrated 1 episode/hour policy, clean initial-only fast crossings
have available_fraction=0.0137984 and 48/48 events without warning. Clean
physical-pulse fast crossings have available_fraction=0.129297 and 45/48
events without warning; the 1s timely-miss fraction is 0.9375. For sampled
iid1/correlated5/dropout fast-cross conditions, all three forcing types have
48/48 events without warning. These are condition-specific findings, not
an aggregate accuracy or a hardware acceptance claim.

No model/threshold changes made. Replay correctness does not establish
useful warning. Next interpretation must separate modal-identification
availability/quality, predicted scores and empirical threshold calibration.
The scorecard limits string still names four concurrent CPU workers;
execution.json records the actual mixed CPU/CUDA checkpoint provenance
and simultaneous-noise-view batching. Do not use that legacy limits string
as the actual execution description or claim deployment latency.

Full retrieval completed: 754210658-byte output tree synchronized; 125 files
transferred. All 31 whole-design trajectory archives read successfully,
including every compressed member; identities/fingerprints match the
execution manifest, each has 405 descriptors and consistent final offsets.
Ten are retained CPU checkpoints; 21 carry CUDA backend fingerprints.
Stopped the obsolete SSH completion wait after the successful direct sync.
No generation restart was needed.

2026-10-07: user requested closing the run record and moving to failure
diagnosis. Generation, retrieval and numerical replay are complete; the
poor warning outcomes above are the starting evidence, not a reason to
repeat the full run. Diagnose archived trajectories by separating physical
acquisition/identification availability, predicted margin/score behavior,
and empirical warning-threshold calibration. Keep the frozen baseline,
thresholds, archived timings and reserved splits unchanged during diagnosis.
Counterfactual calculations are diagnostic only, not replacement results.

2026-10-07: completed first failure diagnosis without changing the model,
thresholds, archived timings or reserved splits. Evidence:
o1_corrected/failure_diagnosis_20261007.json.

1. Unforced acquisition loses physical information before flutter approach.
A focused archived-event probe on supported D000053 fast_cross/initial_only/
replicate0/clean has 0/39 available approach windows: last available at
1.111s, crossing at 8.846s. No cap hits or physical integration termination.
All 48 clean initial-only fast-cross cases lack available estimates in the
approach interval that could provide a 1s lead; six of these are unsupported.
For D000053 iid1, the measured samples from 5s to crossing are bitwise equal
to injected noise alone (physical/noise RMS=1.65484e-43). Nevertheless all
39 approach windows are identified/available, median fit error 0.963148.
Finite modal estimates are not evidence of observable physical modes.

2. Pulse-onset identification creates nonphysical features, then the MLP
extrapolates. Replayed the worst benign calibration window with CUDA
propagation/inference, keeping the CPU modal reference: D000042 safe_dwell,
physical pulses, replicate2, clean, 3.011s. Predicted margin -3.594823342
matches its archived value within 4.44e-16, versus true margin +0.10.
The first cap hit is sample507/512, leaving a quiet prefix plus a short
forced rise. LS-ESPRIT returns three growing poles, alpha=4123..4158/s,
despite the actual safe critical pole alpha=-19.6210/s. Reconstruction
error is 9.54e-18; near-perfect waveform fit does not certify true poles.
The 174-feature row has 23 active columns outside training ranges and
maximum absolute standardized feature 338.49. All 968 clean-pulse benign
calibration windows with predicted margin below -1 contain a pulse onset.

Controlled window-position comparison on the SAME physical trajectory:
3.111s still includes the onset and predicts -3.58022 with alpha~4000/s.
At 3.611s the onset is outside the window, no cap hits, leading estimated
alpha=-19.6140/s and margin=+0.123753. At 4.111s margin=+0.056366;
by 4.611s no modes remain. This isolates onset/first-cap-prefix fitting
from genuine post-pulse ring-down and its eventual loss of observability.

3. False benign extremes make empirical calibration disable useful warning.
The selected threshold 359.482 requires predicted margin<=-3.594823; the
largest score from PERFECT margins anywhere in the declared profiles is 3.
Each noise/forcing calibration group has 0.3h benign exposure, so a 1/h
budget permits no observed episodes. Simply using legacy_clip1 produces
up to 2380 benign episodes/h; lowering thresholds is not a safe remedy.
A secondary candidate-grid limitation exists: threshold 350 gives 70/h
and would meet the 100/h budget, but is absent from the 22 candidates.
That does not resolve the strict 1/h policy or the identification failures.

Next controlled intervention: small causal pulse-aware/observability
identification experiment, frozen model first, using genuine ring-down
segments rather than interpreting forced rise as autonomous growth.
Preserve cap/physical validation and count abstentions/missed crossings;
do not clamp growing poles or select thresholds to hide false alarms.
Commanded/measured excitation timing may be used only if available to the
intended monitor, not silently supplied as simulator truth. Initial-only
quiet cases remain observability negative controls. No architecture
abandoned; GRU remains open after correcting the input/identification issue.

2026-10-07: completed the user-approved small pulse-aware/observability probe.
Code: p4_groups/operational_confirmation_20261003/pulse_aware_probe.py. Protocol,
results, analysis and raw/gate traces: pulse_aware_probe_20261007/ in that
experiment directory. Two previously diagnosed TRAIN designs, D000042
replicate 2 and D000053 replicate 0; safe_dwell through 5.511s and fast_cross
through 10.511s; initial-only and physical-pulse forcing; clean/iid1/
correlated5. Eight underlying trajectories, 24 paired noise conditions;
post-hoc diagnosis, not blind confirmation or reserved-split scoring.

Three arms: archived baseline; command-aware guard rejecting a full 512-sample
window with any nonzero applied-command samples or sensor-cap hits; guard
plus predictive observability. The latter fits modes on the first 256/384
samples and independently forecasts the next 128 samples, requiring at
least 0.5 explained centered energy in BOTH suffixes. This cutoff was
declared before the probe, not tuned on labels, and is not a calibrated
confidence measure. Gate inputs contain measured samples and actuator
activity only; solver poles, velocity, design identity and margin labels
are evaluation/generation data, not gate inputs. Actuator-log availability
is an explicit conditional acquisition assumption.

No model, sensor cap or warning threshold changed. Original-horizon noise
RNG was reproduced using actual leading 51 samples, without future-response
information. Paired baseline predictions replayed to maximum error 2.88e-12.
The smoke run rejected the 3.011s onset extreme and retained the 3.611s
ring-down. Full run completed 24 conditions on CUDA float64; all 24 retrieved
gate archives passed integrity checks and the result source hash matches
the saved script. Changing future commands leaves earlier command masks
exactly unchanged. Original archived delivery/availability timings are
retained; additional gate cost is NOT included. No latency qualification.

Across the two clean physical-pulse cases:
                         baseline    command guard    guard+predictive
safe accepted windows       75             42                 36
safe margin<-1 windows      18              0                  0
fast accepted windows      156             91                 84
fast margin<-1 windows      30              0                  0
fast 1s-lead approach        45             23                 22
Coverage losses are reported alongside removal of extreme predictions;
conditional error improvements alone are not a model-generalization claim.

The predictive arm preserves NO 1s-lead approach windows in either noisy
physical-pulse condition. It accepts zero noisy safe-dwell windows; only
three iid1 and four correlated5 fast-cross windows survive, all on D000042
AFTER crossing (9.611s onward, crossing 8.846s). All noisy initial-only
controls are rejected. Thus the guard addresses forced-rise artifacts but
the fixed 512-sample ring-down scheme does not provide noisy early warning.
Under the unchanged 1/h policy, baseline misses 11/12 crossing noise cases
at 1s lead, both guarded arms miss 12/12. No new policy was calibrated.

Noise-floor diagnostic ONLY (raw physical response never enters the gate):
in D000042's 3.611s safe ring-down, the final 128-sample physical AC RMS is
0.00854 times the specified iid1 noise RMS. Even a noiseless-response
oracle explains essentially none of the measured suffix variance. This
supports genuine lost observability, not just a poorly fitted predictor.
The 512ms aperture is about 10 safe critical damping time constants here.
These synthetic noise settings remain sensitivities, not deployment specs.

Decision: do not promote this gate or repeat the large benchmark yet.
Next isolate ring-down aperture versus noise/frequency resolution on the
same small physical cases, measuring pole recovery and coverage first.
Shortening the aperture changes the feature distribution: do not silently
feed changed-aperture features to the clip-trained model or lower warning
thresholds. Preserve saturation/physical constraints and count abstentions.

2026-10-07: executed the user-approved diagnostic matrix, without regenerating
the large benchmark. Code: operational_confirmation_20261003/diagnostic_matrix.py
and analyze_diagnostic_matrix.py. Frozen protocol, raw result and analysis:
diagnostic_matrix_20261007/ in that experiment directory. Main GPU run completed
in 204.494s on the GTX1660 SUPER, after a real cached/physical smoke. Source hash
and protocol match retrieved results. Original O1 models, thresholds, delivery
records, physical limits and reserved populations are unchanged.

Cached probes use D000042/rep2 and D000053/rep0, the same eight short underlying
trajectories. Physical controls additionally use preselected original TRAIN
designs D000007 and D000033. Counts:1824 information/oracle rows,840 aperture
rows,72 amplitude rows,264 conditioning rows,204 order rows,120 noise-floor
rows,40 detector records;216 excitation,16 ramp/fixed,24 sampling controls.
These are dependent exploratory observations, not independent sample counts,
blind confirmation, or a population performance estimate.

Recovery definition fixed before execution: closest frequency within5% of the
reference critical frequency AND alpha error<=max(1/s,10% of absolute true
alpha). Report continuous errors and finite-mode false positives separately.
Reference poles are physically transverse modes of the previously audited
32-coordinate plate reduction; simulator truth is evaluation-only.

1. Information versus predictive gate: in noisy safe physical-pulse windows,
the original512ms guard leaves66 iid1 and66 correlated5 identified windows,
but the predictive gate rejects all. None of those rejected suffixes has
diagnostic exact-physical-waveform oracle R2>=.5. This supports genuine
late-window information loss there. Conversely, six clean safe windows have
oracleR2=1 but fail the predictive gate: algorithmic loss also exists.
The oracle fits one scalar to the noiseless physical waveform plus channel DC
offsets solely for diagnosis; it never enters selection or model inputs.

2. Aperture/placement: start-aligned command-free post-pulse windows, lag64,
at least1s before crossing, recover the critical pole as follows:
                       128samples  256samples  512samples
clean                       8/8         8/8         5/8
iid1                        6/8         6/8         5/8
correlated1                 5/8         4/8         4/8
iid5                        5/8         4/8         4/8
correlated5                 5/8         3/8         5/8
End-aligned controls and full errors are retained in analysis.json.
This is pole recovery, NOT recovered margin accuracy or warning performance.
Three of the eight iid1/128ms windows also contain another positive-growth
estimated pole despite subcritical truth: critical-pole recovery does not
make the entire pole set physically trustworthy. No shortened-window
features were sent to the old512ms margin model.

3. Forcing and cap are separate mechanisms: all12 unsaturated onset
conditions still produce false positive-growth poles. Clean onset examples
at3.011s estimate alpha roughly+1509..+3476/s and predict margins<-1 while
stable. Amplitude10 hits the unchanged cap and becomes unavailable, not a
repaired physical estimate. Independent physical amplitude2 replay matches
initial+2*forced_contribution to relative RMS2.24e-15, validating the linear
superposition shortcut on the checked case.

4. Noise: iid/correlated comparisons now use the same marginal realized RMS
per channel at each level and paired white/shared draws (target rho=.8).
Initial calibrated physical response fixes the absolute noise scale, never
the shrinking instantaneous waveform. Correlation effects are mixed across
apertures; neither a universal correlation penalty nor an amplitude-only
explanation is established. Floor sweeps retain absolute meter RMS values.

5. Excitation/visibility: on BOTH additional TRAIN designs at U/Ucrit=.9,
ordinary and critical-only1%-noise128ms responses recover2/2 critical poles;
weak-critical responses recover0/2. Sensor projection gain and initial modal
coefficient are archived. This supports excitation/modal-competition failure
on those controls; critical-only initialization is an oracle positive
control, not an actuator proposal.

6/7. Conditioning/order: across20 selected windows per noise condition,
float32, float64 and direct Hankel SVD each recover4 clean and2 iid1/
correlated5 critical poles. DC removal raises clean recovery to6/20 but
does not improve either noisy condition. Orders4/8/12 do not produce a
consistent recovery improvement. ALL eight pure-noise controls per tested
method/order produce finite mode sets. Lower order can reduce spurious
positive growth without making noise physically observable. No precision,
DC-removal or rank change is promoted.

8. Nonstationarity: matched initial state, load and noise, fixed ratio=.99
versus ramp .97->1.01 over256ms. Across four designs, fixed controls recover
4/4 clean and4/4 iid1; ramp controls recover0/4 in both. Median clean
alpha error against window-end truth is6.711/s on ramps versus8.44e-5/s
fixed. End and midpoint references are retained; a stationary estimate is
not an instantaneous end-of-window pole during a ramp.

9. Sampling: new native1k/native4k physical responses and anti-aliased4k->1k
controls all recover4/4 critical poles for both128 and256ms apertures, with
matched64ms Hankel lag. No supporting evidence for aliasing as the dominant
mechanism in these checked controls. The offline zero-phase polyphase
filter is diagnostic only; no causal filtering or latency claim.

10. Contract: identical physical clip samples yield bitwise-identical
normalized signals through clip/trajectory acquisition paths;174 features
replay to maximum difference1.55e-8 and1.76e-11 on the two cases. Clips use
their stored dt=.0009765625; trajectories use dt=.001, correctly. Later
trajectory segments must retain initial acquisition calibration, not be
silently recalibrated as standalone clips. Real distribution/timebase
differences remain; no same-segment implementation mismatch was found.

11. Preliminary credible-query neighborhoods:24 rows include identical
initial segments shared by safe/fast profiles; analysis deduplicates them.
The max5-training-sigma subset shows modest pessimistic rather than severe
optimistic errors and broad nearby label spans. These early stable examples
cannot settle near-flutter representation ambiguity versus readout bias.
Boundary-query and fit/held clip-residual supplements remain part of the
conditional temporal probe, not evidence already obtained.

12. Policy: a denser calibration-only grid still needs threshold359.482 for
1/h and10/h. For100/h,348 gives70/h; the old coarse grid misses that
looser-budget option, but grid density does not solve the strict policy.
Archived hysteresis counterfactuals quantify counts AND duration. In iid1
initial-only benign calibration, one-negative clearing gives1578 episodes
and836.33 alarm-seconds; original five-negative clearing gives47 episodes
and1056.02 alarm-seconds. Thus fewer episodes can mean more active nuisance.
Unavailable updates still never clear alarms; actual policy remains fixed.
Category-isolated episodes are not additive because hysteresis carries
state through masked updates. Waveform oracle strata are diagnostic only.

Waveform-only detector: fixed causal first-difference-energy rise rule,
20ms recent versus preceding100ms baseline, threshold4 and100ms refractory.
All five checked noise conditions detect16/16 applied-pulse onsets with no
extra detections in the four pulse trajectories or initial-only controls.
Recorded discrete-sample delays are retained. This does not validate real
actuator timing, pulse families, hardware noise or detector compute latency.

Conditional GRU step: short noisy windows do contain recoverable physical
information, so a small randomized experiment is specified and implemented
in temporal_probe.py and temporal_probe_20261007/protocol.json. One fixed
4/2/2 design-grouped train/validation/held split within original TRAIN;
randomized crossing, stable and retreat trajectories, pulse timing and
amplitude; matched176-input framewise and GRU, plus elapsed-time-only GRU.
New models explicitly use128ms inputs; original models stay frozen.
No temporal result yet: GPU host disconnected during transfer before smoke
execution. mDNS transfer failed, rsync retry could not resolve cachyos.local,
recorded Tailscale address timed out, peer status is offline, and the two
cached LAN routes timed out/refused SSH with strict existing host-key checks.
Do not describe the temporal script as tested or a trained model.

Decision: promote no estimator, gate, model or threshold. Short causal
apertures and velocity-ramp identification deserve follow-through; finite
mode sets and reconstruction fits remain insufficient observability tests.
The conditional temporal comparison requires GPU connectivity restored.
Real sensor PSD/RMS/correlation/drift, ADC anti-aliasing, actuator-log
availability, harmful-amplitude/required-lead limits, and end-to-end latency
remain missing deployment prerequisites, not quantities this simulation
can establish.

Verification addendum: analysis retains12 unique credible early waveform
queries from24 raw rows, five in the descriptive max5-training-sigma subset.
Waveform detector maximum discrete delays are0ms clean,2ms iid1/correlated1,
5ms iid5 and6ms correlated5. Local temporal_probe.py py_compile passes;
this is syntax verification only. No temporal data, smoke_results.json,
results.json or trained checkpoints exist locally. GPU execution remains
blocked before smoke, not a failed or completed temporal experiment.

2026-10-07: user restored SSH access; conditional temporal comparison is now
completed, superseding the earlier connectivity blocker. Source:
operational_confirmation_20261003/temporal_probe.py; protocol, data, predictions,
three checkpoints, results and analysis in temporal_probe_20261007/.
Persistent unit p4-temporal-probe-20261007.service exited0/SUCCESS;206.36s
inside the runner,3m30s service runtime. All physical propagation,
identification and experimental training ran on the SSH CUDA GPU.

The first real smoke acquired18 paired trajectory conditions, then failed
with Float inputs versus Double weights: the surrounding frozen environment
uses float64. Fixed new model factories by explicitly matching input device
AND dtype. At the user's direction, new code uses .to(device=...), never
.cuda(); factories now use .to(device=x.device,dtype=x.dtype) and CUDA
seeding only for CUDA inputs. New factories trained successfully in a
throwaway CPU exercise and all three trained successfully on CUDA.
Smoke waveform/feature arrays were reused only after exact byte-hash checks
on features/truth/observed/age/times/records/protocol/fingerprint; original
generator source hash and compatibility migration are retained. Exact tested
smoke source snapshot is smoke_source.py. Logging now explicitly detaches
the training loss before scalar conversion; numerical failure checks remain.

Full run: eight original TRAIN designs, fixed4/2/2 grouped split:
train D000007/D000033/D000068/D000087; validation D000042/D000045;
held D000053/D000126. Two independent randomized schedules per design and
cross/stable/retreat profiles; each waveform paired clean/iid1.48 underlying
8s physical trajectories,96 paired conditions,124 causal frames each.
Actual initial measured51-sample calibration, cap50 and valid physical
domain preserved. Simulator velocity/margin/poles/design identity are not
model inputs. Command activity is an explicitly conditional acquisition input.

128ms trailing inputs,64ms stride; prefix64ms forecasts the already-observed
next64ms, requiring centered explained energy>=.5 plus command-free,
unsaturated input. Accepted windows encode the same174 analytical features
for both framewise and GRU, plus measured-input acceptance and age since last
accepted observation. Missing features are explicitly marked, not replaced
with simulator truth. Time-only GRU receives elapsed time alone. Scaling
uses accepted training inputs only; validation selects checkpoints, held
designs do not select anything. One seed42, Adam, fixed200-epoch budget.

Held results, both noise conditions and three profiles pooled:
                                framewise       GRU       time-only
all2976 frames MAE               .020719      .023607      .030577
accepted1424 frames MAE           .009453      .022159      .029358
gaps>.5s,806 frames MAE           .033860      .028291      .034211
Both sensor-input models outperform the time-only control overall in this
split. GRU improves stale-input gaps but loses overall and accepted-frame
MAE to framewise on BOTH held designs. Stable-last2% MAE is.016260 framewise
versus.025207 GRU; smaller GRU signed bias is not automatically better error.
No calibrated warning policy, confidence bound or safe gap-filling rule is
claimed. All three best checkpoints are epoch200: convergence and robust
architecture superiority are NOT established. Parameter counts differ:
6209 framewise,20705 GRU,3905 time-only. This does not abandon GRU.

Accepted-input fraction on held data:69.69% clean,26.01% iid1. Noisy accepted
windows at least1s before crossing number9/3 on D000053 schedules0/1 and
1/3 on D000126 schedules0/1. These are observation counts, not successful
warning events; inference/filter/delivery cost was not qualified. Outputs
inside missing-input gaps are learned extrapolation, not measured availability.

Completed frozen-baseline supplements: original512ms clip fit MAE.005100
versus trajectory-held-design clip MAE.010949. In0<margin<.02, optimistic
error>.01 is2.70% fit versus17.43% held; mean signed error+.001533 versus
+.003274. This establishes a cross-design conditional generalization gap,
not just an aggregate fitting comparison.29 clean near-boundary trajectory
queries pass the old predictive/command guard;13 satisfy critical-pole
recovery, but NONE of those13 is within the descriptive max5-training-sigma
support subset. Their signed prediction errors are all negative
(-.001305..-.018716). Thus no credible reasonably in-support severe optimistic
trajectory family was found here; representation ambiguity versus readout
bias is still unresolved, not a reason to select another loss blindly.

Integrity: full data/protocol/fingerprint/source hashes match results,
including helper source hashes. Retrieved CUDA-trained framewise/GRU/
time-only checkpoints replay all11904 stored predictions on CPU to maximum
absolute differences2.98e-8/1.10e-6/1.34e-6. CPU portability is verified for
new model training/inference, not claimed for the explicitly CUDA acquisition
backend. The full run emitted one Torch warning when converting a read-only
cached NumPy baseline array; the read-only cache was not relaxed or warning
suppressed. Raw positive-margin fractions only indicate false-stable
predictions when restricted to genuinely unstable truth.

Decision: keep original models/thresholds/reserved populations unchanged;
promote no GRU or new policy. We now have short-window physical information
and a controlled temporal comparison, but sparse noisy inputs and
nonstationarity still limit useful continuous estimates. Next isolate
short-window end-time/ramp bias and near-boundary feature support/readout
errors before expanding architecture or recalibrating warning policies.

2026-10-07: user approved continuing the cached temporal convergence check;
the double-descent question was explicitly withdrawn as another project's
question. No capacity sweep or double-descent conclusion belongs to this run.
Code: operational_confirmation_20261003/temporal_convergence.py.
Frozen protocol, predictions, checkpoints and results:
operational_confirmation_20261003/temporal_convergence_20261007/.

Original200 artifacts remain unchanged. Same96 cached trajectories,
124 frames each,174 analytical features, observed flag, age, design split,
seed42, model shapes, train-only scaling, all-frame MSE and Adam settings.
Only the epoch ceiling and plateau patience changed:200 ->2000 and30 ->100.
All three models reproduced the original first200 training/validation
history entries and selected full-data predictions EXACTLY (maximum errors0).
No new waveforms, reserved populations, model capacity or warning thresholds.

All stopped on100 epochs without a new validation-MAE best, not the ceiling:
framewise selected227/stopped327; GRU328/428; time-only399/499. This establishes
the declared validation-plateau stopping condition, not mathematical optimizer
convergence or architecture superiority.

                         validation all / fresh / missing / gap>0.5s
framewise                 .019783 / .019018 / .020455 / .020744
GRU                       .029540 / .035010 / .024733 / .021535
time-only                 .021285 / .022928 / .019841 / .021587
                         held all / fresh / missing / gap>0.5s
framewise                 .020608 / .009223 / .031055 / .033861
GRU                       .020059 / .018077 / .021877 / .021663
time-only                 .030476 / .029104 / .031736 / .034198

Longer training improves held GRU MAE from.023607 to.020059 and reverses its
small aggregate ranking against framewise. Fresh-input accuracy still favors
framewise; GRU wins held missing-input accuracy. But validation missing-input
MAE ALSO favors framewise (.020455 vs.024733), and validation all-frame
time-only MAE beats GRU. The held gap advantage is not a robust generalization
result. Two validation/two held designs and one initialization remain.

Held stable-last2% overestimation>.01 is53.03% framewise/45.15% GRU; unstable
point>0 is78.65%/54.38%. Validation boundary overestimation>.01 is61.17%/81.55%.
No calibrated warning claim: model rankings and useful warning are different
questions. Next screen the previously proposed fixed fresh-frame/framewise,
missing-frame/GRU rule on validation; do not promote it from held gains.

Verification: GPU command completed successfully; all original-prefix
reproduction checks passed; six retrieved protocol/results/prediction/
checkpoint hashes match remote bytes. CUDA-trained selected checkpoints replay
on CPU at maximum prediction errors2.98e-8/1.39e-6/8.34e-7. Input-device/dtype
factories remain portable. The longer-budget experiment source imports existing
Recurrent/metric/scaling/artifact conventions; original experiment code is
unchanged. Short-window stationary-versus-ramp timing control follows.

2026-10-07: completed the remaining approved hybrid screen and short-window
timing controls. The simple hybrid is REJECTED on validation, not promoted
from its potential held gain. Code: analyze_temporal_convergence.py;
temporal_convergence_20261007/hybrid_validation_screen.json. Fixed rule:
accepted fresh frame -> framewise; missing frame -> GRU extrapolation.
No fitted gate, age threshold search or extra training. Validation MAE
hybrid.022060 versus framewise.019783 and GRU.029540. Boundary overestimation
>.01 improves61.17% ->55.34%, but >.03 worsens33.50% ->35.44%; unstable
point>0 improves69.63% ->59.16%. These tradeoffs do not override the failed
MAE screen or establish calibrated warning. Held hybrid was deliberately NOT
evaluated for selection. Validation already selected checkpoints; not blind.

Physical timing code: short_window_timing.py, analyze_short_window_timing.py.
Frozen protocol, raw controls, results and pole-bank replay analysis:
operational_confirmation_20261003/short_window_timing_20261007/.
Four original TRAIN designs: D000042 replicate2, D000053/7/33 replicate0;
all checked against original77-design training population.96 small physical
free-response controls,192 paired clean/iid1 conditions.128/256 samples span
127/255ms at1kHz. End velocity ratios.99/1.005; ratio rates0/.04/.15625 per
second; rate.15625 retains the earlier stress setting. Ordinary initial state
shared across stationary/ramp counterparts. Critical-only oracle states use
the start-ratio critical eigenvector, matched initial sensor RMS. Zero forcing,
one paired Gaussian draw per design, fixed initial ordinary-response noise
floor, existing leading51-sample calibration, cap50, lag64 and max8 pairs.
No current-model input, cap, acquisition validation or threshold changed.

All192 estimates retain a pole within5% of the end-reference frequency;
none saturates. For128ms stationary ordinary controls, end-pole recovery is
8/8 clean and8/8 iid1; median absolute alpha error.000316/s and.120102/s.
Stationary critical-only controls also recover8/8 in each noise condition;
clean error is approximately2.33e-8/s. This is an informative physical
positive control, not the previous vanishing late-trajectory signal.

Ordinary responses show residual128ms time-location bias. At ratio-rate.04/s,
median absolute alpha error versus start/midpoint/end reference is
1.342/.320/.558 per second clean,1.729/.395/.813 iid1. At256 samples,
midpoint/end errors are.280/1.555 clean and.172/1.560 iid1. Shortening
reduces end-time error here, but does not make the estimate instantaneous.
At128ms stress rate.15625/s, nearest end-pole recovery is0/8 clean,1/8 iid1;
median end alpha error2.464/2.585 per second. Testing the whole pole bank,
not only the closest frequency, also finds0/8 and1/8 matching end poles.

Stronger counterexample to a fixed timestamp correction:128ms clean,
critical-only initialization at rate.04/s returns5-7 apparent mode pairs
(stationary counterpart:1), but NO estimated pole matches current end
alpha/frequency in any of8 windows under the existing5%-frequency plus
max(1/s,10% absolute alpha) criterion. Thus failure is not merely selecting
the wrong nearest pole. Examples: D000042 stable end-ratio.99 has true end
alpha-2.298/s, closest estimate+.635/s; D000007 end-ratio1.005 has true
alpha+2.196/s, estimate-3.152/s. D000053 stable also estimates positive
growth; D000033 supercritical also estimates negative growth. These are pole
sign errors relative to end-time reference, not new margin-model outputs.
Near-perfect reconstruction is not evidence of physically meaningful poles.
The initial critical-mode oracle is an evaluation positive control only.

Cached waveform replay evaluated the current128ms predictive-gate stage:
ALL96 conditions pass its first64 ->next64ms forecast>=.5, finite/uncapped
window and identified-mode checks. In the clean critical-only.04/s family,
scores are.9999998506-.9999999837 despite0/8 end-pole recovery. This stage
measures short-horizon predictability, NOT confidence in instantaneous poles.
It does not include downstream174-feature finite checks or inference/delivery
timing, and is not a full availability or safety qualification.

With ordinary iid1 at the slower ramp, nearest-frequency recovery5/8 becomes
7/8 when checking ANY matching pole with at least1% fitted component energy.
Mode selection ambiguity exists in that family; it does not explain the
clean critical-only or ordinary fast-ramp failures above. Eight256ms fast-ramp
records have reference-frequency span>5%; all128ms reference spans are<=5%.
That descriptive continuity check is not proof of mode identity.

Verification: GPU physical run completed192 conditions; all192 cached
waveforms replayed their selected poles within1e-4/s alpha and1e-6Hz, then
exposed full pole banks and128ms gate scores. Retrieved protocol/results/raw
hashes match remote; raw artifact contains96 physical arrays and4 paired noise
draws. NaN padding beyond each128-sample logical window is storage only and
never enters normalization or estimation. Original temporal data/results/
models and reserved populations remain unchanged.

Decision: no simple hybrid, no GRU replacement, no fixed half-window label
shift, no predictive-gate score treated as physical confidence. GRU remains
live: longer training improved it, but validation does not support its held
gap advantage. Next smallest estimator intervention to TEST: a time-varying
single-mode growth/frequency fit on these cached critical-only controls,
against unchanged stationary LS-ESPRIT; require stationary/noisy controls and
end-time growth-sign recovery before extending to ordinary mixtures. This
is a diagnostic hypothesis, not an implemented or proven fix. No large
dataset, architecture sweep or warning recalibration is justified yet.

2026-10-07: completed the user-approved cached128ms time-varying modal-fit
comparison. User also requested Luna agents for substantial new code, with
main-agent review, targeted small fixes, re-prompting for large issues and an
honest reliability report. Two Luna implementation slices were dispatched:
waveform fitter and experiment runner. Main retained protocol, integration,
all execution/scoring verification and this journal.

Code: operational_confirmation_20261003/time_varying_modal_fit.py and
experiment_time_varying_fit.py. Frozen protocol, results, independent
verification and residual spatial-coefficient audit:
operational_confirmation_20261003/time_varying_fit_20261007/.
Same48 cached critical-only128ms conditions: four original TRAIN designs,
two end ratios, clean/iid1, stationary/.04/.15625 ratio rates per second.
No simulations, new datasets, reserved populations, neural training or
warning changes. Ordinary mixtures were conditionally allowed only after
ALL frozen critical-stage checks passed.

Observation model: shared midpoint growth/frequency and their linear slopes,
per-channel offset/cos/sin coefficients. Centered tau; integrate the linear
growth/frequency trends into quadratic log-envelope/phase. Solve channel
coefficients exactly each nonlinear iterate. Duration is127ms, so endpoint
alpha=alpha_mid+.5*duration*alpha_slope. Measured Hann-windowed aggregate FFT
initialization only; varying fit starts from the stationary waveform fit with
slopes0. Both growth signs allowed. Existing first51 measured calibration,
noise floor, cap50 and logical sample boundaries remain. Float32 calibrated
data promoted to Float64 for fitting; no physical poles in inference arguments.

                    end-pole matches / growth signs correct (of48)
archived LS-ESPRIT          19 /34
stationary single-mode     26 /40
time-varying single-mode   40 /48
LS match count19 is also the ANY-matching-pole bank count, not only closest
frequency. Recovery criterion remains frequency error<=5% and alpha error
<=max(1/s,10% abs(reference alpha)).

All48 varying fits converge successfully inside the prescribed bounds.
The clean stationary max alpha error is7.67e-8/s, below the frozen.001/s guard.
Noisy stationary median alpha error is.022860/s versus archived LS.008847/s;
the frozen allowance is+.05/s, giving.058847/s. This guard passes, but the
extra slopes DO increase stationary-noise estimation error; not a free gain.
Maximum varying frequency relative error is.006254 (0.6254%).

                         ramp alpha MAE, clean / iid1, per second
archived nearest LS            4.241787 /3.670132
stationary single-mode         3.136252 /3.134193
time-varying single-mode        .855921 / .858903
At slower rate.04/s, varying recovers8/8 clean and8/8 iid1, versus archived
0/8 and2/8; growth signs are correct throughout. At stress rate.15625/s,
varying recovers4/8 in each noise arm, versus archived1/8 clean and0/8 iid1.
All growth signs are correct even in these eight quantitative failures.

The frozen all48-end-poles-match check FAILS (40/48), so ordinary mixtures
are explicitly SKIPPED, not silently tested/promoted from partial success.
No warning or margin-model claim follows from the recovered signs in four
repeatedly inspected positive-control designs. Remaining alpha errors are
concentrated at the fast ramp in D000042 stable/supercritical, D000053 stable,
and D000007 stable, in both clean/noisy versions. Worst alpha error3.687/s.
Fits are interior, successful and frequency-close; noise alone cannot explain
the clean counterparts. Example: clean D000042 stable alpha-.157/s versus
true-2.298/s, although its sign is now correct.

Residual observation-model audit, no estimator expansion: hold the inferred
varying carrier fixed, solve channel complex coefficients separately on
first64/last64 samples, compare channel vectors up to common complex scale.
Median normalized direction change clean:0 stationary,.017825 slow,.100717
fast; iid1:.003797/.018221/.100116. All replayed calibrated signal hashes
match the driver. [INFERENCE] Nonconstant spatial coefficients are consistent
with changing mode shapes/coupling or unmodeled carrier curvature, not proof
of a unique cause. This single-mode, constant-channel-coefficient observation
model does not yet supply quantitatively calibrated instantaneous growth.
Adding arbitrary channel amplitude drift could confound growth; next isolate
modal-shape transport/coupling versus carrier curvature on the existing
controls before expanding the estimator. No architecture search justified.

Main verification: four known stationary/chirped synthetic signals, including
both endpoint growth signs, recover alpha/frequency within1e-4/s and1e-5Hz;
quiet, constant, nonfinite and cap50 measurements explicitly fail. Input and
stationary warm-start objects remain unmodified. Main independently executed
all48 physical fits; final driver points/residuals agree EXACTLY and all
frozen scoring/guard booleans were independently recomputed. Source hashes
and cached input provenance checked. CPU experiment completed; varying-fit
median runtime7.59ms, excluding imports and not a full monitor deadline claim.

Luna experience in this batch: fitter supplied sound centered-model math and
waveform-only initialization, but main corrected missing cap rejection,
invalid-measurement failure handling and square-root versus existing SSE
relative-error convention. Runner first wrote687 lines then failed its handoff;
main re-prompted the available Luna worker for a compact implementation.
Revision wrote300 lines, but native message delivery also errored, so main
reviewed the actual files, not an unverified completion claim. Main corrected
stationary-control guards to score the VARYING candidate, added current-code
provenance/explicit error denominators, and fixed a reused reporting variable
that incorrectly displayed the noisy guard threshold. The incorrect initial
report was removed and regenerated; final physical decisions are unchanged.
Unexpected exceptions are not hidden behind catch-all fit fallbacks.

Conclusion on agent reliability: useful bounded implementation assistance,
not hands-off delivery. This is one batch, not a general reliability estimate.
The partial physical improvement/failure is an experiment outcome, distinct
from the implementation/reporting defects caught by main review and execution.

2026-10-08: user authorized the improved-fitter plus simple calibration versus
learned-predictor comparison, asked whether distance to flutter is more useful
than growth/damping, and explicitly required noisy conditions. This tests the
existing128ms framewise/GRU experiment, not a final77-design production-model
replacement or a new claim of blind superiority.

Code: fitter_comparison_acquire.py, fitter_comparison_score.py and
fitter_comparison_gap_control.py under operational_confirmation_20261003.
Frozen protocol, raw replay cache, dataset, predictions, results,
post-hoc gap-policy control and independent verification:
operational_confirmation_20261003/fitter_margin_comparison_20261008/.
Eight original TRAIN designs:4 proper/2 validation/2 previously inspected
held. Replayed the same48 physical schedules/initial states; no new structural
designs, speed sweeps, neural training or reserved-population evaluation.
Five views per schedule: clean, iid1, iid5, correlated1, correlated5;240
trajectories x124 frames. Correlation rho.8; realized per-channel noise RMS
matched to independent draw, fixed absolute floor from initial51 physical
samples. Noise levels are sensitivity settings, not hardware specifications.

All methods use the same128-sample causal aperture/64-sample stride, measured
initial51 calibration/cap50, command-free/finite/unsaturated checks and the
unchanged first64->next64 forecast-skill>=.5 shared gate. This intentionally
does not qualify the candidate fitter or replace its failed all48 physical
accuracy gate. A future fitter-specific observation gate could differ.
Stationary warm-start and varying fits are unchanged. Both must succeed
inside bounds; failed/bound fits remain in explicit denominators. No true
pole/velocity/design/target/future waveform in predictor features.

Baseline6: alpha_end, frequency_end, growth/frequency slopes, signed
alpha/(2*pi*f), SSE relative-fit-error; degree1/2 polynomial ridge with
existing scaling/sum-squares regularization convention. Degree and alpha
selected by4-design LODO within proper clean/iid1 only; degree2/alpha100
wins. A pre-run linear-ridge control on the original174 LS analytical
features selects alpha10000. Both maps fit proper clean/iid1 observations
only. Missing outputs hold last estimate causally; before first fit they
remain unavailable, not a population-mean fallback or a safe declaration.
Frozen converged framewise/GRU/time-only checkpoints are not retrained.

Matched fresh held-design MAE and fitter availability:
                    fitter6    ridge174   framewise    GRU       availability
clean               .015550    .013246    .008699     .016165       69.62%
iid1                .011209    .011286    .010564     .022930       25.94%
iid5                .010998    .010350    .008524     .025209       18.15%
correlated1         .010401    .010806    .010163     .027588       23.99%
correlated5         .010138    .010724    .011011     .027391       17.14%
Same fresh rows within each noise condition; different accepted subsets
between conditions. Smaller5% conditional MAE does NOT show noise improves
the estimator. Framewise is better on pooled fresh MAE in4/5 conditions;
individual-design directions are mixed in some noisy cases. No significance
claim from two designs. Original-gate availability is within one held frame
of fitter availability in each condition.

Primary all-common-finite-frame MAE first favored fitter+hold over native
neural gap outputs under every noise condition. Main then added an explicit
POST-HOC gap-policy control, without new fitting or selecting a winner from
held results: hold last accepted FRAMEWISE/GRU prediction; also hold raw
alpha. Completed primary protocol/results remain immutable.
                    fitter6+hold   framewise+hold   native framewise
clean                 .018499        .009566           .016485
iid1                  .019535        .017085           .024730
iid5                  .020155        .019606           .026450
correlated1           .018358        .017144           .024558
correlated5           .019049        .023079           .027194
Thus much of the apparent initial all-frame advantage was GAP POLICY,
not proof the new physical readout beats a neural readout. Framewise+hold
wins this MAE comparison in4/5 conditions; fitter+hold wins correlated5.
GRU/native or hold does not establish an advantage in this small experiment.
All hold outputs are stale extrapolation, not fresh measured availability.

Warning positive region is true m<=.02 (near boundary INCLUDING unstable);
negative region m>.02. Calibrate strict score>threshold at1/5/10% empirical
false-warning budgets on validation clean/iid1 pooled only, then freeze
thresholds for all noises/held. Validation already participated in neural
checkpoint selection; these are exploratory thresholds, not independent
risk guarantees. Unknown positive windows count as misses; unknown negatives
remain in population denominators. No stress-noise threshold retuning.

At the5% CALIBRATION budget, held recall (actual held false-warning rate):
                    fitter6+hold          framewise+hold
clean               94.99% (1.43%)        53.76% (0.38%)
iid1                71.98% (0.48%)        51.25% (1.33%)
iid5                66.97% (0.10%)        55.58% (9.44%)
correlated1         72.44% (0.48%)        46.92% (0.10%)
correlated5         67.20% (0.10%)        38.95% (1.81%)
These are the same validation budget, NOT equal realized held risk. Fitter
has higher recall in all5; clean/correlated1 trade actual held false warnings,
while iid1/iid5/correlated5 improve both relative to framewise+hold. The
framewise+hold independent5% condition exceeds its nominal5% budget after
transfer. Native framewise recall23.23-28.02%; native GRU11.39-21.87%.
Fresh literal growth-sign alpha>0 recall24.60-28.47%, but availability
differs. Calibrated hold-alpha emits no held near-boundary true positives
at5% budget; raw growth scale/its hold policy is not automatically a useful
transferable margin warning.

CRITICAL LIMIT: fitter6 margin predictions are POSITIVE on274/274 truly
unstable held frames in EVERY noise condition:100% false-stable point sign.
Native framewise falsely reports positive margins in77.01-80.29% of those
frames; native GRU48.54-68.25%. The successful conservative2% warning threshold
does not repair point calibration at m=0. None of these point heads is a
qualified safe-state estimator. Do not equate a useful warning ranking with
an accurate signed distance to flutter.

EARLY-WARNING LIMIT: fitter6 triggers after near-boundary entry in all four
held cross schedules, but in every noisy family only1/4 alerts precedes
flutter. Lead relative to flutter:[+.576,-.768,-.768,-.256]s in schedule
order. High positive-frame recall includes many post-crossing warnings;
it is NOT proof of reliable early warning. Pre-near-boundary false alerts
are separately counted, not credited as useful lead. Gate sparsity and
stale estimates remain major limitations.

Verification: actual GPU replay/acquisition completed;6880 fit attempts kept.
All96 original clean/iid1 inputs/gates/labels/ages match archived data;
maximum174-feature difference0. Frozen framewise/GRU/time-only predictions
also reproduce EXACTLY. Main independent calculations verify420 primary
metric entries,160 post-hoc control entries, all warning budget/missingness
rules and causal ages/holds. Six raw-waveform refits cover all5 noise families
and a bound rejection; all fitted parameters/residuals reproduce exactly.
Four tie/missing-budget smoke cases plus unknown-positive, causal-hold and
correct flutter-lead/pre-entry-alarm checks passed. No permanent tests.
Both Luna tasks timed out without handoffs after writing files; main reviewed
and corrected schema/subset validation, import/provenance, bound-failure
denominators and the initially post-flutter-only lead computation before
executing the full comparison. No agent completion claim substituted for proof.

Decision: distance to flutter remains an actionable engineering target
(speed headroom/operating envelope), but its value is not uniquely neural.
The simple competitor ALSO learns a distance calibration from simulation
labels; this compares readout complexity, not learning versus no learning.
No unique neural winner, no fitter promotion, no safe-declaration or early-
warning claim, no threshold/model change. Strong simple competitor established
in this diagnostic; framewise retains fresh-point accuracy and matched-hold
MAE benefits. Next comparative decision must focus on signed boundary
calibration and warning timing/availability on broader disjoint TRAIN
designs, with fixed gap policies and explicit failure accounting, before
using reserved designs or claiming a distinctive P4 contribution.

2026-10-08: user requested a concrete P4 endpoint and finite research target
with a strong novelty/impact case for a relevant Q1 submission, starting from
the completed work rather than continuing indefinitely. Research assessment
and proposed contract: ../P4_RESEARCH_TARGET.md. This is a recommendation,
not an already agreed safety requirement, completed result or authorization
for another campaign. No model, gate, threshold, split or historical result
changed; no experiments were restarted.

Assessment: spatial/multimode representation gains on77-design TRAIN CV are
substantial, and the pipeline/causal-trajectory audits are valuable evidence.
Time-varying fitting repairs all demonstrated growth-sign errors but still
fails8 quantitative fast-ramp controls. Latest two-held-design comparison
does not establish a neural winner: framewise+hold wins MAE in4/5 conditions;
simple fitter+calibration warns more often, but its point margins remain
positive on every unstable held frame and3/4 noisy crossing alerts are late.
These are distinct limits, not an aggregate score or reason to defend GRU.

Recommended central claim to EARN: transferable, uncertainty-qualified
normalized flutter headroom and timely advisory warning from short measured
multichannel transients on unseen plates, without target-specific flutter
calibration. Explain physically what joint temporal/spatial information
constrains the design-dependent growth-to-margin conversion, and establish
useful risk/availability benefit against the strongest same-information
physical/simple competitor. A small physical readout may win; a generic
fitter/MLP/interval combination or neural architecture is not itself novelty.

Proposed primary clean/iid1/correlated1 gates: mean design MAE<=.010/.015;
near-boundary MAE<=.005/.010; dangerous point optimism>.02<=5%;
non-vacuous>=80% current approach availability and farther-stable useful
headroom; >=90% crossings warned with>=1s delivered usable lead at<=1
nuisance episode/hour; >=25% relative timely-miss reduction versus strongest
matched competitor, with paired design-level evidence. Independent lower-
bound calibration and a<=5% upper-confidence-bound design-run false-clear
criterion are separate from point accuracy. Proposed research thresholds,
not aircraft safety requirements or demonstrated attainable performance.
Both5% noise families remain explicit stress tracks; no silent exclusion of
unknowns, stale holds, onset failures, quiet periods or unsupported designs.
Keep16 sensors/.5s maximum causal aperture and provisional10ms p95 full-
pipeline compute. No automatic re-excitation added to the input contract.

Scope warning claim to a prospectively declared physical excitation envelope,
including its quiet gaps in denominators. Initial-only silence remains a
negative control: no promise of detecting future crossings after all physical
response disappears. Challenge causal-history ambiguity with benign dwell/
retreat cases before optimization. Preserve legacy clip and signed-margin
tracks; a useful ranking does not establish signed distance or early warning.

Finite campaign proposed: one bounded physical mechanism diagnostic/estimator
revision; one small-grid same-information comparison including physical-prior
inference, simple maps, framewise, improved-input GRU and direct warning;
one broader disjoint TRAIN operational screen with at most one targeted
correction/repeat; then one locked blind confirmation plus independent
response/label-model validation. Stop and write on all-gate success. Repeated
development failure or failed blind validation ends this claim; changed
observation/actuation scope needs explicit approval, not another open sweep.
Compatible independent physical/model-form validation must be identified
before expensive development; a new propagator of the same equations does
not qualify. Existing sim-only status is not presented as a newly discovered
defect. The proposed strong-publication endpoint requires stronger evidence.

Plan>=120 fresh blind designs; original16 test designs alone cannot support
the proposed precise risk claim. Planning check with scipy.stats.beta:
zero failures/16 independent design-runs gives one-sided95% upper bound
.170750; zero/120 gives.024655, or.033544 with three-condition Bonferroni.
These are sample-resolution examples, not risk estimates for P4, and more
noise copies/windows do not add independent structures. Exposure, clustering,
multiplicity and nonzero failures must be addressed before final collection.

Primary-source refresh read Chajjed2022 and Mendez2026 full manuscripts and
Wang-Zhou2026 publisher-deposited abstract; prior full-text Crowther/Wang-Ma
review retained. Short-record identification, neural boundary prediction and
probabilistic structural priors already exist. Their headline scores are not
ranked against our MAE. Closest neural full protocols remain needed before
uniqueness/superiority claims. No numeric gate guarantees acceptance by
"most Q1 journals"; aim for a defensible strong relevant-journal submission.
Verification this turn: executable statistical-resolution check and all six
local references in the proposed target resolve. No new performance result,
runtime behavior change, training or reserved-design evaluation.

2026-10-08: completed the goal-directed cached modal mechanism diagnostic
and an evidence-backed shared-carrier curvature correction. User clarified
that failure of one correction must inform iteration, not end the route.
Updated P4_RESEARCH_TARGET.md progress/stopping language accordingly:
preserve the physical/operational goals and reassess on stalled progress or
a demonstrated limitation, not an arbitrary first-attempt exit. Earlier
single-correction/repeat wording is superseded by these progress checkpoints.

Code: operational_confirmation_20261003/analyze_modal_transport.py and
evaluate_carrier_curvature.py. Frozen contracts/results and independent
analysis under operational_confirmation_20261003/modal_transport_20261008/:
protocol.json, results.json, traces.npz, mechanism_analysis.json,
curvature_protocol.json, curvature_results.json, curvature_analysis.json.
Same four previously inspected original TRAIN designs, 24 unique physical
critical-only 128-sample waveforms, 48 clean/iid1 views. Actual span127ms.
Existing inputs, physical sources and identities hash-verified. No new
physical case draws, ordinary-mixture fitting, neural training, warning
policy changes or reserved-design scoring.

Physical comparison: reconstruct the original fixed physical state basis;
track the critical eigenpair at all128 timestamps, with parallel phase and
unit state-vector norm. Its dual left vector obeys w^T v=1. Independently
replay the exact cached zero-input dynamics using the original2ms held
midpoint matrices and1ms propagation steps. Fit only the original two
critical-pair initial coefficients, not new trajectories. All24 cached
waveforms replay with relative RMS<=2.5622e-12. Critical branch remains the
least-stable physically transverse branch at every sample. Maximum reference
alpha difference5.10e-11/s, frequency difference3.90e-12Hz; maximum relative
eigen residual2.36e-16.

Oracle mechanism controls: exact integrated lambda(t) with frozen spatial
shape; moving shape without transport; moving shape with the diagonal
connection coefficient equation c_dot=(lambda-w^T v_dot)c. Full replay
contains off-diagonal coupling as well. These are diagnostic counterfactuals,
NEVER deployed inputs. Preserve ORIGINAL measured leading51 offset/scale,
paired absolute noise and float32/cap50 conventions, not per-counterfactual
recalibration. Archived baseline fits were reused, not refitted for reassurance.

Findings: physical pole curvature AND transport/coupling contribute.
With clean fast ramps, freezing the spatial shape improves linear-fit
recovery4/8->7/8 and alpha MAE1.37606->.51755/s, but does not remove all
error. The maximum departure of physical alpha from a linear trajectory is
1.12675/s. Instantaneous sensor-visible modal-envelope growth is not the
frozen-system pole real part: for D000007/end_ratio.99/fast ramp,
physical alpha=-5.32440/s versus envelope growth=-7.67163/s. Continuous-field
components: diagonal transport=-2.43241/s, off-diagonal residual=+.14133/s,
projection log-gain=-.05615/s. The noncritical sensor component is25.7% of
replay RMS in this case. Do not equate modal-envelope growth with the norm
derivative of the raw oscillating signal, or attribute every fit error to
one of these components.

Independent verification used scipy's separately computed left eigenvectors
and a five-point backward derivative at.25ms spacing on all24 endpoints.
Maximum envelope-growth discrepancy against the recorded finite-difference
analysis=.0002911/s. Continuous-field derivatives evaluate the physical
matrix at the endpoint, not the exact last midpoint-held replay matrix;
checking that distinction gave maximum discrepancy.09848/s. This numerical
discretization does not account for the multi-unit transport contrasts.
A separate gauge smoke preserves physical growth=-2/s with the connection;
ignoring it produces apparent+1/s on the same reconstructed trajectory.
This is mechanism evidence, not a four-design identifiability theorem.

Correction: waveform-only variable projection with quadratic alpha(t) and
frequency(t), integrated into cubic log-envelope/phase. Channel offsets
and cos/sin coefficients remain CONSTANT; no free channel-amplitude drift.
Initialize from the saved waveform-only linear fit; curvature starts0.
Dimensionless time-conditioned parameters, original mid/slope solver bounds,
bounded curvature and measured FFT physical-band slack. Reference poles,
design and velocity are never fitter inputs. Run all48 native measurements
and all48 frozen-shape controls; retain original physical accuracy criteria.

Native result:
                      Linear carrier    Curved carrier
End-pole recovery         40/48             45/48
Correct growth signs      48/48             48/48
Clean ramp alpha MAE       .85592/s          .45297/s
IID1 ramp alpha MAE        .85890/s          .45461/s

All48 curved native fits succeed and are interior; clean stationary maximum
alpha error6.685e-7/s passes. Remaining quantitative pole failures:
D000042/.99/clean, D000007/.99/clean, D000007/.99/iid1, all fast ramps.
Frozen-shape clean fast-ramp recovery reaches8/8; noisy reaches7/8. Thus
curvature addresses a demonstrated carrier limitation, not transport in
the native waveform. Native noisy-stationary median alpha error=.138944/s,
exceeding the frozen LS median+.05/s limit=.058847/s; the linear candidate
median was.022860/s. This is a failed noise guardrail, not a new tolerance.

Independent scoring recomputed96 physical/counterfactual recovery/sign/error
records and checked the full48-condition set and implementation/input hashes.
Both clean/iid1 native ramp MAEs remain lower than the original stationary
and archived LS comparators. Waveform-only smoke recovered two known curved
endpoints of opposite growth sign within1e-4/s and1e-5Hz; zero-energy,
nonfinite and cap50 measurements returned explicit failures. No permanent
tests added: this remains an unqualified experimental estimator comparison,
not a production fix or cutover.

Decision: no promotion of the free-curvature variant, no abandonment of P4.
Measured progress40->45/48 justifies continuing this mechanism. Next bounded
estimator checkpoint: waveform-only noise-aware curvature selection/shrinkage
to preserve stationary controls, with the transport distinction explicit.
Accept it only against the SAME48 physical controls and noise guardrails;
lower residual or correct signs alone is insufficient. Remaining physical
bias then requires a constrained structural-dynamics/transport model, not
arbitrary channel gain drift. Keep ordinary mixtures gated on all critical
checks; structural headroom ambiguity and downstream margin/warning benefit
are still untested here. No architecture/loss sweep, longer observation or
scope relaxation is justified by this diagnostic.

2026-10-08: user approved the physics-constrained/non-neural correction route
and requested progress percentages based on the agreed metrics/criteria.
Initialized P4_PROGRESS.json and update_p4_progress.py. Progress is criteria
attainment, not task completion, effort, safety or publication probability.
Current estimator stage: equal-weight passed checks/6. Overall endpoint:
fully demonstrated completion gates/10 from P4_RESEARCH_TARGET.md. Missing
or unqualified evidence scores0; scores can decrease. Freeze denominators
and report actual metrics beside percentages so a guardrail improvement
cannot conceal loss of recovery. Existing development evidence must not
be pooled across different candidates/populations to pass an endpoint gate.

Ran the first noise-aware curvature selection checkpoint. Code:
operational_confirmation_20261003/noise_aware_carrier.py,
experiment_noise_aware_carrier.py, verify_noise_aware_carrier.py.
Frozen protocol, results, independent verification and residual diagnosis:
operational_confirmation_20261003/noise_aware_carrier_20261008/.
Same48 native critical-only controls and48 oracle frozen-carrier views.
No new physical cases, supervised residual correction, neural training,
ordinary-mixture fitting, warning-policy changes or reserved-design scoring.
Pinned waveform/calibration/source hashes checked; prior fits reused.

Method: dimensionless six-parameter profiled-residual Jacobian from central
differences, per-channel residual variance and heteroscedastic sandwich
covariance. Choose curvature only when joint curvature Wald statistic
exceeds the fixed two-parameter BIC-scale penalty2*log(16*128)=15.24924.
Otherwise use the measured linear fit. No true pole, design, velocity,
noise label or margin enters selection. Failed/unsupported curvature
diagnostics remain explicit. This is an approximate model-selection rule,
not true heteroscedastic maximum-likelihood BIC or calibrated confidence.
No strength/threshold sweep on these repeatedly inspected designs.

Result: native curved fit selected in only1/48 views, a clean stationary
case. No native ramp selects curvature. Recovery returns45/48->40/48;
all48 signs and interior fits preserved. Clean/iid1 ramp alpha MAE returns
to.85592/.85890 per second. Noisy-stationary median error returns to
.022860/s, below the UNCHANGED .058847/s limit. It passes5/6 critical checks,
but still fails all48 physical-pole recovery and loses the curvature gains.
The rise in stage percentage is NOT a reason to promote this method.

Matched mechanism contrast: all16 clean oracle frozen-carrier ramp views
select curvature, while none of the32 native clean/iid1 ramp views do.
Frozen clean fast-ramp statistics range15766.7 to888583.6; native clean
fast-ramp statistics only2.36 to10.25. This contrast implicates the native
observation/model residual in the variance-based selection failure; it
does not prove curvature is physically unobservable from native waveforms.

Additional direct check on all8 clean native fast-ramp residuals:
residual RMS is at least220761 times the conservative float32 rounding
scale. Top3 spatial residual components account for at least99.99947%
of residual energy. These controls contain NO injected sensor noise.
Treating this highly coherent model mismatch as independent measurement
noise makes the curvature statistic overly conservative. This is a
measured residual-model problem, not justification for lowering the
threshold against pole labels or claiming a calibrated confidence bound.

Independent verification: analytic variable-projection derivatives and
QR-based covariance, separate from the finite-difference/SVD implementation,
checked all96 selection statistics. Maximum relative statistic discrepancy
8.12e-9; all model choices, physical scoring and six pass flags agree.
Complete live fitter replays on native noisy-stationary D000042 and clean
fast-ramp D000007 reproduce selected models and alpha endpoints within1e-8/s.
Synthetic smoke retains the linear model on a heteroscedastic noisy
stationary waveform, selects curvature on two known opposite-sign growth
cases, and recovers alpha/frequency endpoints within1e-4/s and1e-5Hz.
Quiet, nonfinite and cap50 measurements return explicit failures.
The verification writer initially passed a string __file__ to a Path-only
hash helper; corrected to Path(__file__) before saving verified evidence.
No scientific threshold or score changed for that tooling correction.

Progress ledger:
Linear carrier:         5/6 =83.3%; physical pole recovery40/48.
Free curvature:         4/6 =66.7%; physical pole recovery45/48.
Noise-aware selection:  5/6 =83.3%; physical pole recovery40/48.
Overall endpoint:       0/10=0%; no complete locked end-to-end gate established.
This is not0% scientific/engineering effort. Endpoint status stays separate
from the small critical-mode diagnostic and requires its declared evidence.
The updater checks saved independent verification and recomputes percentages;
this estimator checkpoint does not change endpoint qualification statuses.

Decision: do NOT promote the residual-variance selector, and do NOT discard
the demonstrated carrier-curvature improvement or abandon the goal.
Next bounded correction must separate coherent physical/model residual
from stochastic sensor noise and represent transport through constrained
shared-state physical dynamics/observation, not arbitrary channel-amplitude
drift. Keep population priors and physical gauge constraints explicit;
target true geometry/poles/speed cannot become inference inputs.
Re-evaluate against the SAME physical accuracy and noise guardrails before
ordinary mixtures, then matched framewise/GRU comparisons. No threshold
relaxation, architecture/loss sweep or longer-window scope change.

2026-10-08: user clarified that requested progress is OVERALL attainment
against ALL specified research criteria, not the current pole/estimator
stage. Made endpoint.percentage the primary reporting metric in
P4_PROGRESS.json and updater output; the6-check estimator percentage is
explicitly secondary. No criterion, threshold, weight or denominator changed.

Reviewed each of the10 gates against current saved comparison and historical
timing evidence. Added criterion-specific status, evidence sources and gaps
instead of a generic repeated "not demonstrated" note. Overall completed
criteria remain0/10=0%; this is strict endpoint attainment, not effort or
a claim that partial research progress is zero. Missing measurements do
not imply zero performance. Preserve partial evidence without crediting
a complete gate or combining incompatible candidates/populations.

Evidence examples: fresh-window prediction accuracy is promising, but
noisy gaps and signed boundary errors remain; independent uncertainty/
false-clearance evidence and qualified usefulness are absent; fitter6
has0/4 held>=1s crossing leads in every noise family at the frozen5%
frame calibration budget (best clean.896s, noisy.576s); episode/hour burden
and>=25% timely-miss benefit are not established. Historical warm full-path
timing includes p95<=10ms for spatial107/two_mode174, but blend50 exceeds
10ms and final-method/p99/deadline evidence remains incomplete. Blind and
independent-model confirmation remain unfinished. Sign-risk fractions from
all unstable frames are NOT substituted for the prescribed exclusion-slice
wrong-sign metric, and positive point margins are not called clearance.
No model rerun, new experiment, scoring of reserved designs or promotion.

2026-10-08: user corrected the progress definition: each condition has its
own partial attainment percentage; overall progress is a weighted average,
NOT the binary fraction of fully passed conditions. Replaced the headline
scoring formula in update_p4_progress.py, P4_PROGRESS.json and the research
target. Acceptance thresholds and separate qualification booleans are unchanged.

Scoring version2: ten equal default criterion weights (10% each); equal
metric-component weights inside each criterion. Lower-is-better uses
min(1,target/value), with zero error receiving full numerical credit;
higher-is-better uses clipped value/target. Measurements remain null when
unknown and earn no evidenced credit; their weight is NOT redistributed.
Report missing weight, so no-evidence credit is not mistaken for measured
failure. The score's missing-component upper bound is not a confidence interval.

Used one declared development reference: frozen framewise temporal model,
native predictions from the verified matched-comparison cache, two inspected
held TRAIN designs. No mixing with fitter or spatial107 timing to improve
the score. Recomputed per-design all-frame and exact -.02<m<.05 boundary
MAEs. Computed optimism>.02 and sign errors on current points with
.01<=abs(m)<=.05, separating sign sides. This cache has no eligible unstable
points on that excluded-near-zero sign slice; that component is unknown.

Documented overall attainment15.6544%; measured scoring weight38.3333%,
unknown61.6667%. Accuracy51.1682%; dangerous errors38.7091%; stress evidence
and online feasibility33.3333% each. Native framewise has no >=1s held
crossing leads in the primary conditions, so measured timely fraction earns
zero; independent timely-miss bounds remain unknown. Clearance, qualified
usefulness, episode/hour burden and matched-nuisance comparative impact
remain unmeasured, not numerically observed failures. Fresh blind design
count and independent validated response/label source count are both0.
Stress credit is for disclosed coverage/lead results, not proven robustness;
online credit is for the128ms acquisition aperture, not a proven full-path
deadline. Qualification remains0/10, separately from partial progress.

Verification: ran the actual updater against existing artifacts; prediction
and result hashes match prior independent verification. Independent
scalar-loop recomputation agrees with accuracy and dangerous-error scores.
Exercised50% partial MAE attainment,80% partial warning-rate attainment,
zero-error saturation, negative/over-target higher-is-better clipping,
null evidence, and a changed criterion weight; increasing accuracy weight
from.1 to.2 changes the overall score to18.8829%, matching the independent
weighted calculation. No new model run, reserved-design scoring, promotion,
or change to the next physical transport-correction objective.

2026-10-08: user requested the easiest available next step. Chose a bounded
full sensor-to-output timing measurement of the frozen framewise predictor,
not another training/model search. Used existing cached development waveforms
on cachyos: GTX1660SUPER, Intel i5-9400F, torch2.12.1+cu130, one CPU/BLAS
thread. Frozen protocol and source/input hashes before the runs. All7440
decisions cover both held TRAIN designs, two replicates, three profiles and
all five noise families. Reserved designs untouched.

Runner: measure_framewise_latency.py. First attempt exposed shared-import
float64 defaults; fixed the new runner to the checkpoint's explicitfloat32.
Second attempt exposed single-window versus original batched CUDA modal
numerics. Controlled frame180/21: measured samples identical; original
two-view batch reproduced raw features exactly; either strided or contiguous
single-window input differed by6.6856e-5 at feature13, with frozen-scaled
difference3.9522e-6 and margin difference1.49e-8. A later frame also exceeded
the scaled-feature guard. Preserved failed_protocol_v1/v2 rather than hiding
the aborted comparisons. Final protocol collects ALL decisions and compares
consumer margin/availability/age/warning outputs; raw/scaled feature errors
remain disclosed diagnostics. Original margin tolerance2e-6, endpoint
thresholds, physical calibration and model weights unchanged.

Completed warm serialized computation: worst per-noise ACCEPTED-path p95
18.7930545ms; pooled accepted median17.061411ms, p9919.21600648ms,
max21.421561ms. All2309 accepted decisions miss10ms. Across7440 decisions,
2327 miss; rejected paths are much cheaper and cannot justify a passing
latency claim. First decision after initialization269.066478ms, separately
reported; process/model/CUDA initialization, acquisition and sensor transport
are not included in warm computation. Aperture remains128ms.

Parity: availability, age and warnings match;8/7440 margin outputs exceed
the frozen2e-6 tolerance, maximum1.283735e-5. Maximum raw feature error.021809
and scaled-input error.005583. Final replay_valid=false and criterion_met=false.
No standalone deployment or exact feature-equivalence claim.

Independent verifier: verify_framewise_latency.py. Recomputed126 statistics
using sorted scalar interpolation and independently checked all saved
predictions, ages, availability and warning bits against the original cache.
Verification preserves the8 prediction mismatches rather than asserting
success. Hash checks cover measurement arrays, protocol, source and inputs.

Weighted documented attainment15.6544->20.7614%; unknown scoring weight
61.6667->55%. Online criterion33.3333->84.4037%:
(100% acquisition +53.2111% latency +100% reporting)/3. This is partial
metric/evidence credit, not speed improvement or passed qualification;
qualification remains0/10. Smoke-ran actual updater and independently checked
the weighted change, preserved failed replay/target status and all ten
criterion weights. Research target and ledger updated with the new evidence.

Evidence: framewise_latency_20261008/{protocol.json,results.json,
measurements.npz,verification.json,replay_diagnosis.json}. Measurement closes
an unknown, reveals a missed computation target and exposes numerical
batch sensitivity; it does not resolve the physical transport/headroom
problem. Next small engineering action: stage-profile identification/
forecast/features/inference without dropping gates or weakening numerical
checks. Core physical correction objective remains unchanged; no new
architecture/loss sweep, neural training or model promotion.

2026-10-08: user requested a frontier reviewer to unblock progress on one
criterion, after correctly rejecting timing/reporting credit as a prediction
improvement. Focused review on signed-headroom accuracy, not latency or the
overall score. Provided the acceptance/information contract, measured
framewise/ridge/GRU comparisons, gap-policy controls,77-design spatial
representation evidence, physical transport/curvature diagnostics and rejected
noise-aware correction. Prepared evidence packet at
local://p4-headroom-frontier-packet.md; reviewer P4HeadroomFrontier.

Initial verdict CONCERNS: do not prioritize constrained transport yet.
Transport is established in critical-only controls, but its dominance in
ordinary-trajectory signed-margin errors is not established. Gap handling
materially changes MAE; a fitter map positive on all274 unstable held clean
frames does not alone distinguish extraction from conversion/calibration.

Computed a conditional decomposition directly from the completed comparison's
predictions.npz, without fitting or acquiring data. Native framewise current
equal-design MAE clean/IID1/correlated1=.00873673/.01143148/.01069597;
current boundary=.00795916/.00982669/.00968566. Gap-output all-frame
MAE=.03435844/.02953873/.02980912. Current counts1037/387/358 of1488;
boundary current662/322/306 of893. Current here means measured-input gate
acceptance, not a qualified reliable advisory. Native populations differ from
the post-hoc common-fresh fitter population. Gap correction alone cannot reach
the.005 clean boundary target.

Reviewer recommended a bounded extraction-versus-conversion diagnostic.
Follow-up refined it to three separately calibrated maps: measured fitter
features; oracle features for the branch matched to the fitter component;
oracle critical-branch features. Same measured residual feature,128 samples,
proper-training-only map selection and original data/gate/calibration.
Measured-to-matched oracle tests parameter estimation conditional on physical
correspondence; matched-to-critical oracle tests branch choice/visibility.
These contrasts are not additive causal decompositions. Oracle information is
offline diagnosis, never operational input or proof of measured observability.

Further review SUPPORTED retaining the branch prerequisite: existing spectrum
helper initializes at max real part then follows nearest previous root. It
does not guarantee instantaneous max-growth identity or distinguish dominance
switches from ambiguity. Freeze oracle branch definition and validate ordinary
trajectory identity/correspondence before interpreting contrasts. Keep all
ambiguous/unresolved cases in population accounting. Critical-only validation
is insufficient for this prerequisite.

Decision rules: paired equal-design primary-condition/boundary/unstable/
farther-stable errors and dangerous point errors; preserve original all-frame
and gap counts. Oracle gains on both held development designs with conditional
accuracy targets, reduced dangerous errors and farther-stable degradation
<=.005 would justify targeted extraction work conditional on measured mode
support, not transport automatically. Oracle failure with four fitting designs
leaves calibration/support/capacity unresolved, not proven nonidentifiability.
No eligible unstable sign-exclusion-slice points means unknown, not success.

Recorded the next checkpoint in P4_RESEARCH_TARGET.md. It is NOT yet executed.
No physical correction, neural training, model promotion, reserved-design
evaluation, threshold relaxation or performance gain. P4_PROGRESS.json and
its headline score remain unchanged. This review selects the next
discriminating experiment; it does not clear an accuracy or availability gate.

Verification of the new decomposition: prediction-cache SHA256 matches the
archived comparison verifier; all six native accuracy metrics/counts match the
ledger; current+gap counts and weighted per-design errors reconstruct the
original all-frame/boundary populations within1e-12. Recorded derived rows at
local://p4-headroom-current-gap-evidence.json. No runtime failure was rerun.


2026-10-09: completed the ordinary-mixture extraction/branch/conversion
diagnostic and a frontier-directed fixed regularization probe.
Evidence: p4_groups/operational_confirmation_20261003/
headroom_oracle_20261008/{protocol,results,paired_results,verification}.json.
No new physical cases, neural training, operating predictor or warning change.
Recorded original CUDA states replay the archived sensor trajectories.
C is instantaneous maximal-growth transverse endpoint mode, tracked across
the same128 samples; it is not a fixed flutter-onset branch. B correspondence
uses temporal and phase-invariant spatial signatures, never margin/error.
Unresolved observations and all original gaps remain in denominators.

Two separately fitted/selected paired-support contrasts, not one additive
causal decomposition. Full critical-valid fresh A/C supports:
clean1036, IID1 386, correlated1 357 of1488 decisions each.

| Condition | A measured fresh MAE | C critical-oracle fresh MAE | A boundary MAE | C boundary MAE |
|---|---:|---:|---:|---:|
| Clean | .015790416 | .012915960 | .005764137 | .005364184 |
| IID1% | .011862499 | .010033138 | .008173790 | .006747937 |
| Correlated1% | .011023431 | .009603673 | .008010217 | .006928069 |

Overall C improves both held designs, but clean MAE D000053=.010096999,
D000126=.015734921, both above.010. Clean D000053 boundary worsens
.005570947→.005715265; D000126 improves.005957327→.005013104,
still above.005. C optimism>.02 is0/662,0/322,0/306, but positive
point margins on actual unstable windows remain130/130,110/110,110/110.
No eligible unstable points in.01<=abs(m)<=.05; no fresh m>=.10 points:
the prescribed unstable-sign and farther-stable checks are unknown.
Zero large optimism events is not correct sign or clearance.

Matched-common A/B/C supports533/209/200; unresolved spatial correspondence
503/177/157 of original fresh windows. Clean A/B/C MAE
.015868428/.013042892/.011199751; IID1
.008727137/.006618335/.005567115; correlated1
.007771465/.005876556/.004909642. Matched C boundary
.001709878/.001570867/.001560278 and positive-on-unstable
1/98,0/88,0/88. Different selected training rows/maps: these results
cannot be attributed solely to mode matching or extrapolated to unresolved
windows. The conditional opportunity is real; full-population success is not.
Nonorthogonal pair energies include interference; condition/projector
diagnostics and large critical energy do not guarantee recoverability.
The reduced-model critical growth at stored full-model ucrit is not exactly
zero (range-.144684 to+.053962/s across the eight systems).

Independent verifier completed840 metric checks, five SVD map/LODO
selection checks (max prediction disagreement4.82e-15),2351 endpoint
pole checks (max2.10e-12). Original populations and frozen comparators
are unchanged. No accuracy/availability gate cleared.

Frontier review: first test shrinkage, not a forced physical zero-crossing
conversion. Full-support C already predicts positive on191/194 clean and
163/165 IID1 proper-TRAIN unstable windows, so its failure is not solely
held-design extrapolation. Fixed probe keeps degree2 and the six features,
fold-local preprocessing, objective and support; only alpha3162.27766→0.
Rank-aware minimum-norm least squares, no inverse normal equations, clipping,
or penalty search. Evidence: regularization_probe{,_protocol,_verification}.json
and regularization_probe_predictions.npz in the same diagnostic directory;
runner: operational_confirmation_20261003/probe_headroom_regularization.py.

Zero penalty improves clean TRAIN in-sample MAE.013356870→.009664166
and positive-on-unstable191/194→4/194. But proper fixed-alpha LODO clean
MAE worsens.016840325→.045424233, IID1.012504899→.033682152.
Held clean/IID1/correlated1 MAE worsens to
.027878921/.019163830/.017991535; boundary to
.012744431/.011950707/.011508713. D000053 worsens overall,
D000126 improves overall; clean boundary worsens on both designs.
Stable-side exclusion-slice wrong signs rise from zero to
110/483,70/180,66/165; boundary optimism from zero to
48/662,13/322,12/306. Most unstable positive signs disappear, but this
trades one error for other unacceptable errors. Reject the zero-penalty map.
LODO uses already selected fixed baseline alpha, not nested blind assessment.
Verified120 metric rows independently, five NumPy/SciPy solves, unchanged
full-support/gap masks and source artifacts. Rank26/27 is expected from
derived-feature polynomial redundancy, not proof of missing physical information.

Final frontier review SUPPORTED: next is ONE controlled expansion of proper
TRAIN design support, retaining the existing conversion model. Preregister
eligible additional designs with the same labels/calibration; exclude every
copy of validation/development-held designs. Same ordinary-mixture profiles,
128-sample windows and clean/IID1 fitting views. Paired A-measured/C-oracle
original-four versus expanded support; preserve row-loss fitting and equal-
design scoring. Scale sum-loss alpha with fitting weight to hold normalized
regularization strength constant per arm/paired fit, including folds.
No penalty grid, physical-map constraint, transport correction or neural sweep.
Require overall/boundary gains on both held designs, no increased dangerous
errors, unchanged population accounting and frozen framewise comparison.
C-only gain is diagnostic; measured A must earn actual prediction improvement.
Broader support is the next hypothesis, not proven treatment. Campaign not run.

Actual progress: better discrimination of shrinkage versus extraction/support,
but no deployable performance gain. P4_PROGRESS.json unchanged; its legacy
20.76144% documented-attainment value is NOT increased or reinterpreted as
prediction progress. Missing physical/sign/availability/lead evidence remains
missing. Research target updated; existing runtime/replay failures retained.

Reviewer clarification: preserve row-weighted fitting, equal-design scoring
and alpha/total fitting-weight normalization. Record accepted training rows
by design/noise: gate-dependent sampling may change their effective weights.
Any gain is attributable to the expanded training population, not automatically
design diversity alone.

2026-10-09: user requested identification-quality explanation and consultation
on transmissibility, group delay and ADDITIONS to the proven feature roster.
Keep all174 current features. Existing assembly includes42 base summaries,
65 first-mode spatial entries and67 second-mode/spatial-similarity entries.
Quality entries are reconstruction residual SSE/centered-signal SSE, detected
mode count, pre-saturation usable fraction and availability, not reliable-
margin uncertainty. Existing sidecar phase_slope is median absolute discrete
cross-correlation lag, not spectral phase slope/group delay; it is not part
of the current42-base assembly.

Verified prior spatial evidence at spatial_modal_20260930/results.json:
77 TRAIN designs/4454 clips; baseline42 vs spatial107 mean design MAE
clean.0160453→.00858679, fresh IID1.0178688→.0101308,
fresh correlated1.0194239→.0110308. Shuffled-spatial controls are worse.
These exploratory legacy clip results justify retaining the useful roster,
not conflating it with current128ms ordinary-trajectory performance.

Researcher produced complete primary-source brief at
local://p4-frequency-feature-primary-sources.md. Forwarded ALL206 original
lines, including counterevidence, to the existing frontier reviewer at user's
explicit request. Source-informed reassessment SUPPORTED two additions only:

1. Non-cancelling spectral distribution: summed sensor powers, normalized
spectral entropy and measured power in neighborhoods of the existing up-to-
three fitted frequencies. Unlike FFT of the mean signal, summing powers
does not cancel antiphase sensor components. These are measured spectral
descriptors, not additional exact modal energy or close-mode resolution.
2. Broadband spatial concentration: S=sum_k X[k]X[k]^H and
tr(S^2)/tr(S)^2. Captures spatial alignment across spectral content beyond
the two retained fitted shapes. Second priority; not per-frequency coherence,
FDD modal identification, uncertainty, or excitation-invariant transmissibility.

Freeze definitions for any future ablation: existing calibrated channels,
channel demeaning, periodic Hann, unpadded128-point FFT at1kHz, positive
bins1–63. Normalize entropy by log63. Frequency neighborhoods use each
existing valid fitted frequency's nearest bin ±1, clipped to1–63; ties
choose the lower bin, overlapping neighborhoods form a union. No oracle
frequency routing or mode-selection search. Entropy/concentration require
positive power/trace; neighborhood mass also requires a valid fitted frequency.
Explicit feature-validity masks, no fake finite estimate or discarded rows.
No longer aperture, additional sensor, force input or gate/age-policy change.

Group delay deferred: derivative stability for short nonstationary response
ratios is unvalidated. Deferral is NOT a claim that phase derivatives require
analytically known filters. Naive single-periodogram magnitude-squared
coherence and per-frequency rank rejected: coherence is identically1
for nonzero single-segment spectra and a spectral outer product has rank1.
True input-output FRFs are not available just from a generic output ratio;
unknown loading and reference zeros complicate transmissibility interpretation.

Reviewer corrected research overstatements, also corrected in the internal
brief: matched marginal noise PSD gives the same EXPECTED summed noise power
whether sensors are independent or correlated; cross-channel correlation
changes spatial structure and sampling behavior, not automatically that mean.
Sensor/band aggregation can reduce variability, so descriptors need not
inherit a single periodogram bin's full variance. Neither guarantees robustness.
One cited transmissibility method's multiple-loading requirement is not a
universal identification theorem. Narrowband/linear-phase assumptions govern
envelope-delay interpretation, not the definition of a phase derivative.
Long source records do not establish performance on our128ms transients.

Smallest fair feature test: on ONE frozen proper-TRAIN population, compare
baseline174 against EACH family separately, same partitions/settings,
paired fresh IID1/correlated1 evaluation and clean/IID1-only fitting.
Keep feature-size/mask effects explicit and reuse matched-size negative
controls if needed; do not increase hidden-model capacity or open a grid.
Score equal-design overall/exact-boundary errors, optimism and both sign
sides, farther-stable guardrails and5% stress outcomes. Preserve original
current/stale/gap counts; missing guardrail coverage remains unresolved.
Stratify cached stationary/slow/fast and cross/stable/retreat windows by design
and overlapping true-margin ranges, using truth ONLY for scoring. Aggregate
profile-mix gains cannot hide matched-margin ramp/retreat regressions.
Check fitted-frequency bin transitions for neighborhood-mass prediction jumps;
reject material signed-error regressions, not silently smooth or retune.
Noise-only spatial concentration is a confounding control, not an automatic
reason to discard family2. Combine families only after individual support.

The already planned TRAIN-support expansion remains first priority.
Feature additions must be a separate ablation, not bundled with that expansion
so that treatment attribution remains clear. No feature implemented, model
trained, physical cases generated, reserved designs scored or P4 predictive
gain demonstrated in this consultation. Progress ledger unchanged.

Definition-only executable smokes passed: antiphase cancellation in pooled
FFT versus retained summed power;128-point bin spacing7.8125Hz;
single-segment coherence deviation from1 at most4.44e-16 and snapshot
spectral rank1. Candidate formulas pass global-amplitude invariance,
zero-signal/missing-frequency validity cases and bounded outputs; IID and
correlated noise change descriptors. These checks are NOT P4 robustness
or accuracy tests. Throwaway code ran in subprocesses; no test/code files added.

Primary sources:
- Devriendt et al., operational modal analysis with changing distributed loads,
ICSV14 2007: https://www.acoustics.asn.au/conference_proceedings/ICSV14/papers/p470.pdf
- SciPy coherence/CSD:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
and https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html
- MathWorks averaged-coherence warning:
https://www.mathworks.com/help/signal/ug/cross-spectrum-and-magnitude-squared-coherence.html
- Brincker et al., output-only frequency-domain decomposition:
https://vbn.aau.dk/ws/files/203990023/Modal_Identification_of_Output_Only_Systems_using_Frequency_Domain_Decomposition.pdf
- Group delay definition/interpretation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html
and https://www.dsprelated.com/freebooks/filters/Phase_Group_Delay.html

2026-10-09: completed the user-requested additive spectral-feature test on
the SSH GTX1660 SUPER. Evidence:
`p4_groups/operational_confirmation_20261003/additive_spectral_20261009/`
contains protocol, acquisition, dataset,15 checkpoints, predictions,
training histories, conditional results, independent verification and decision.
Runners: `experiment_additive_spectral.py`, `score_additive_spectral.py`
and `verify_additive_spectral.py` in the parent directory; descriptor
definitions and mathematical smoke checks in `spectral_features.py`.

Keep all174 existing features. Compare baseline, summed-sensor spectral
entropy + mass around existing top3 fitted frequencies, broadband spatial
concentration, and separate size-matched shuffled controls. No combination,
TRAIN-support expansion, physical correction, GRU comparison or parameter
search in this experiment. Three initialization seeds42/43/44; unchanged
32->16 framewise learner, Adam, loss, selection and stopping settings.
Four fitting/two validation/two previously inspected held development designs;
same48 cached physical trajectories,128-sample windows,64-sample stride.
Fit/select on original clean/IID1 only. Evaluate original five noise views
and two fresh paired IID/correlated1%/5% draws on the12 held trajectories.
Every family has identical observations, targets, ages and gap population.
Canonical96 clean/IID1 records are reused exactly; seed42 baseline reproduces
the archived selected predictions bit-for-bit. Other original174-feature rows
retain the prior acquisition tolerance, independently checked.

Descriptors use demeaned calibrated channels, periodic Hann, positive FFT
bins1..63; no pooled-signal cancellation. Neighborhoods are the union of
nearest-bin +/-1 around existing fitted-strength frequencies, lower tie.
Spatial concentration is tr(S^2)/tr(S)^2 over the integrated spectral matrix,
not per-frequency coherence/rank, pole confidence or physical modal energy.
Undefined values remain NaN with explicit validity; model imputation uses
proper-TRAIN means and preserves the original observation gate. Controls
shuffle standardized appended values across physical trajectories within
split, consistently across noise views, keeping target validity flags;
missing donors contribute standardized0. Controls are not deployable models.

Equal-design all-frame MAE, averaging seed-wise scores, not predictions.
Noisy rows average the TWO FRESH draws only; clean has no noise draw.

| Condition | Baseline174 | Spectral distribution | Spatial concentration | Shuffled spectral | Shuffled spatial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | .017820 | .018726 | .016990 | .017994 | .016460 |
| Fresh IID1% | .024886 | .025052 | .024608 | .024643 | .024440 |
| Fresh correlated1% | .025161 | .025303 | .025010 | .025037 | .024866 |
| Fresh IID5% | .027049 | .027438 | .026921 | .026876 | .026741 |
| Fresh correlated5% | .027545 | .027736 | .027506 | .027437 | .027429 |

Near-boundary all-decision MAE also worsens for the spectral family across
these conditions. Spatial concentration gives small average gains, but its
shuffled control is better on ALL13 held noise conditions for BOTH all-frame
and boundary means. Neither family gives improvement on every held
design/seed across primary fresh-noise conditions. Clean cross-profile
boundary error worsens with both additions; spatial noisy stationary
strata also regress. Profile, margin-matched profile, stationary/slow/fast
ramp, current/gap and bin-transition rows are retained in results.json.
Bin changes include strength-order reordering; report those separately.
On214 clean accepted-to-accepted transitions, spectral seed42 signed-error
jump p95 is.084212 versus baseline.025262; this alone does not identify
which descriptor or identification change caused the jump.

Dangerous signs do not improve: spatial concentration positive-on-unstable
events are724/822 seed-frame exposures clean versus658/822 baseline;
fresh IID1%1480/1644 versus1357/1644 and correlated1%1480/1644 versus1362/1644.
These pooled descriptive counts retain repeated seeds/noise views, NOT
independent trials or calibrated warning risk. Farther-stable and prescribed
negative sign-exclusion populations are absent; remain unknown, not passed.

Verification:1344 original-population checks,1203 independently reconstructed
measured spectral windows (SVD-based spatial check),5635 metric rows,
9360 saved-network frame replays using independent NumPy algebra, and468
paired-difference rows. Max descriptor discrepancy1.67e-15; saved-network
prediction discrepancy1.66e-7, below unchanged2e-6 tolerance. The verifier
caught float32 narrowing in the diagnostic jump buffer; fixed to float64
and recomputed unverified scores, WITHOUT changing predictions or tolerance.
Recorded in precision_failure.json. Protocol wording erratum: "original-eight"
RMS is stale; pinned M.noises and the original temporal source actually use
all16 current sensor channels. Independent waveform reconstruction confirms
the executed noise magnitudes; no generation or result was retuned.

Decision: neither addition earns adoption in this fixed experiment. Reject
the spectral family as tested; spatial mean gains do not demonstrate added
information beyond the matched shuffled control and worsen dangerous signs.
This is not a universal impossibility claim for spectral features. Keep all174
existing features and the operational predictor unchanged. No qualification,
reserved-design scoring or progress increase. P4_PROGRESS.json hash matches
the pre-experiment regularization-probe record exactly. Previously specified
controlled proper-TRAIN support expansion remains the next bounded campaign,
not executed here. Three seeds/two fresh draws do not turn two repeatedly
inspected held designs into blind validation.


2026-10-09: completed the user-requested bounded multi-regime response-history
experiment while two Butler evidence reports were being assembled. Evidence:
p4_groups/operational_confirmation_20261003/multivel_history_20261009/.
Same four fit/two validation/two inspected held designs, cached trajectories,
174 features, original observation gates/gaps, three seeds and fresh paired
noise as the additive-spectral comparison. Three nonoverlapping128-sample
windows, decision lags4/2/0 at64-sample stride,384 acquired samples/.383s span.
No supplied velocity, critical speed, margin, solver identity or future data.
Controls: archived single128, capacity-matched repeated-current, ordered
history, pooled/unordered history and shuffled past with current preserved.

Pooled history lowers mean overall AND boundary error versus single128 in
all13 held noise conditions, but both real-history arms worsen distinct-regime
and current-boundary MAE in ALL seven primary original/fresh conditions.
The improvement concentrates in dwells; pooled history improves only60/84
primary design/seed overall/boundary comparisons. Stable wrong signs increase;
neither arm reaches the primary overall/boundary targets or earns adoption.
Complete numbers and per-seed events are retained in results.json/decision.json.
This tests bounded response history, NOT an independent known-speed sweep:
the original128ms source is already time-varying. No claim that arbitrary
multi-speed information is useless or that a temporal information ceiling exists.

Independent verification:120 raw-support checks,16380 saved-network frames
(max deviation1.3051e-7),3220 metric rows; original baseline unchanged, no
future/cross-split history or velocity labels in inputs. Ledger hash unchanged.
Farther-stable and prescribed unstable-exclusion coverage still absent/unknown;
no calibrated warning, blind validation, predictor change or progress increase.
User requested broader scientific diagnosis instead of automatically treating
TRAIN-support expansion as the next answer. Consolidated reports/equations,
failed spectral additions and this completed history evidence go to frontier
for prioritized falsifiable experiments with target-linked outcome rules.

2026-10-09: completed the consolidated frontier review requested by the user.
Both complete reports, source-checked174/107/42 equations, failed additive
spectral evidence and completed384ms history results were delivered inline.
Full findings: p4_journal/review_20261009/frontier_review.json. The working
history report was reconstructed by the dispatcher from collected Butler
evidence after workers stopped before publishing; the broad constraint/model
survey was Butler-authored. At the user's renewed delegation request, two
further Butler source checks covered subsets but stopped before complete
audits. Do not claim independently verified full-report coverage. Material
survey corrections distinguish synthetic quiet calibration from an available
hardware noise record, screen/five-fold N1 rates, candidate/deployed scope,
single-segment spectral rank and oracle-model failure from information limits.

Frontier diagnosis: useful system context and conversion explain the strongest
historical gains; second-mode context has boundary/missingness/noise costs.
No new spectral family or temporal-method promotion is justified. Accurate
poles plus poor margins implicate converter/support/representation interactions,
not proof that more designs alone is the answer or that information is absent.

Ranked proposed experiments: (1) cached four-arm factorial, second-block
content174/107 crossed with all-frame/current-only fitting, identical-width
head and fixed original native gap outputs across arms; (2) structural TRAIN
support expansion crossed with richer matched-modal diagnostic content, with
correspondence/calibration prerequisites and missing regimes; (3) conditional
bounded GRU versus same-span pooling/concatenation and flags/age controls,
reset each decision inside383ms span. Exact controls/target-linked success,
regression and outcome-dependent decisions recorded in P4_RESEARCH_TARGET.md
and the untruncated frontier result. None of these new proposals was run.
Only measured signed-margin improvements can earn accuracy progress; oracle
or dwell-only gains cannot qualify the endpoint. Original predictor, ledger,
reserved splits and all physical/warning/runtime acceptance gates unchanged.

2026-10-09: executing the three frontier-proposed experiments at the user's
request. The cached fitting/context factorial completed in
`p4_groups/operational_confirmation_20261003/current_context_20261009/`.
Current-only fitting, removing the last67 second-mode coordinates, and their
combination each worsened mean overall and current-boundary error in all7
primary noise conditions. This rejects those specific matched interventions,
not a general objective or representation class. Independent verification:
624960 saved-network decisions,2576 metric rows,92 transition rows; original
native gap predictions exactly preserved. No arm adopted. The user challenged
its utility: limited elimination value, not a diagnosis of estimation,
training-support or physical-information limitations.

Support acquisition required an explicit physical-domain amendment. The
initial attempt failed at D000053 farther-stable ratio.78: Mach
1.8074338014281053 is below the preserved Mach2 gate. Retain that failed
protocol/source and completed shards in `support_modal_20261009/`. Active
campaign is `support_modal_20261009_v2/`, with the same preregistered12 added
TRAIN designs and3 folds. D000053 keeps its original profiles and new
unstable-exclusion condition; its farther-stable condition remains physically
inapplicable/unknown. All original fitting/checkpoint-validation designs and
all12 primary evaluation designs retain all5 profiles.

Acquisition/assembly completed:2094 records x124 frames x174 coordinates,
35362 accepted measured windows. Matched physical replacement covers
90902/236712 modal slots (38.40194%); this is not a full oracle or information
upper bound. A diagnostic serialization bug discarded imaginary components
of saved measured mode shapes AFTER measured/oracle features were computed.
`recover_modal_shapes.py` recovered sine coefficients against measured windows
and frozen ordered poles for all35362 windows. Original real coefficients
are bit-identical; maximum refit real-relative discrepancy4.6576e-10.
Mandatory `recovered_modal_shapes.npz` is hash-linked to the unchanged
dataset; no pole re-estimation, feature regeneration or prediction change.
The independent recovered-diagnostic acquisition checker passed294 matched
windows across all20 designs, including calibrated physical coordinates and
state-to-sensor reconstruction. SSH disconnected during the separate
correspondence checker; the remote process remained running on reconnection.
Comparative support fits and conditional bounded GRU are not yet verified.
No predictor, qualification, reserved-design score or progress-ledger change.

2026-10-09: completed all three requested frontier experiments. Evidence:
`p4_journal/review_20261009/completed_campaign_frontier_packet.md`,
`completed_campaign_analysis.json`, and `completed_campaign_context_appendix.json`;
actual packet and requested numeric context were delivered to the existing
frontier reviewer for reassessment. Two additional evidence workers failed with
service429 errors before publishing; dispatcher compiled this evidence.

Cached factorial:12 fits,624960 independent network decisions,2576 metric rows
and92 transition rows. No intervention improves the seven primary held means;
native gaps remain exactly frozen. A newly quantified leverage limit: fixed-gap
contribution alone sets equal-design all-frame MAE floors .010411 clean,
.021958 IID1 and .022764 correlated1, already above .010/.015 gates even with
perfect current estimates. This was a narrow elimination check, not an
experiment capable of passing those all-frame absolute targets.

Support/content factorial:24 fits,6231744 independent network frames,7176
metric rows and546 contrast rows. Expanded measured support improves aggregate
all/boundary means on12 grouped evaluation designs, but only14/21 fold-noise
rows and142/252(all),125/252(boundary) design/seed comparisons. Current-boundary
improves20/21 means,168/252 comparisons. New farther-stable/unstable-exclusion
profiles worsen in aggregate while original3-profile errors improve. Partial
physical replacement helps no original-four mean contrasts; after expansion
all/boundary/current-boundary wins are9/21,4/21,7/21. Replacement is38.40194%
overall; on primary12 designs it is84.45% clean,27.17% IID1,26.08% correlated1.
No full-oracle or absent-information conclusion. All absolute error gates fail.

Bounded GRU:36 fits,10530 independently recomputed metric rows and3456 sampled
NumPy network/GRU decisions; max difference1.2815e-6 within unchanged tolerance.
Same383ms aperture and grouped support. GRU improves all/distinct-regime means
versus framewise21/21, unlike the earlier four-design history result; this is
not just dwell gain. Against both same-span controls,12-design aggregate wins
are3/7 all,4/7 boundary,7/7 current-boundary,4/7 distinct regimes. Stable-sign
error worsens versus framewise19/21 means; optimism worsens versus pooling17/21.
Retreat and farther-stable regressions remain; one farther-stable increase
.006108 exceeds the .005 allowance. No ordering-specific or adoption claim.

Support scoring caught a stratum-map indexing bug after all24 fits; corrected
scorer uses unchanged predictions. Contribution-weight verification now uses
exact serialized-dtype arithmetic, not widened tolerance. The archived
score-source label in verification.json is clarified in scoring_correction.json;
the actual corrected scorer hash is in results.json. GRU verification caught
a missing distinct_regimes_current mask and wrong equality boundary; corrected
checker passed all metrics. Exact .005 versus .004 regime-boundary smoke passed.
Original feature/model/policy values remain unchanged by these checker repairs.
No model promotion, calibrated availability, warning/runtime qualification,
reserved-design scoring or progress-ledger change.

2026-10-09: frontier reassessment completed and preserved verbatim as structured
data in `p4_journal/review_20261009/completed_frontier_reassessment.json`.
Verdict:CONCERNS; no arm earns promotion or endpoint credit.
Frontier agrees the user's experiment1 leverage criticism is substantially right:
fixed gaps force both all-frame and boundary errors above targets even with
perfect current predictions. Boundary floors are .010169 clean,.017627 IID1,
.017965 correlated1 versus .005/.010 targets. Require such feasibility checks
before prioritizing future corrections; do not repeat this elimination test.

Evidence-supported bottleneck: observation erasure/extrapolation, not proven
raw-waveform nonidentifiability. Expansion matters but does not establish
conversion adequacy; noisy physical replacement is especially incomplete.
Bounded measured content helps, including distinct regimes, without showing
uniform ordering-specific benefit over same-span controls. The earlier
four-design history conclusion is revised at this scope, not universally reversed.

Frontier's next proposed, NOT executed, mechanism test: diagnostic gate-erasure
ablation on the same expanded folds and383ms-reset GRU. Retain genuinely finite
pre-forecast-gate174 estimates in forecast-rejected slots versus current zeroing,
with a same-width shuffled-rejected-content control; preserve physical-source
pairing/partitions, gates, flags/age, calibration, exclusions and denominators.
Rejected predictions stay unqualified. First prerequisite: distinguish genuine
pre-gate estimates from fallback/manufactured values; saturated, commanded,
invalid and genuinely unavailable estimates remain masked. Require gap/boundary
improvements across primary families without current-boundary/sign/optimism/
retreat/farther-stable regressions. Failure rejects this retention mechanism,
not all measured information. Do not retrospectively narrow the acquisition
envelope. Independent physical/model-form validation remains a separate prerequisite.

2026-10-09: frontier now owns scientific prioritization/acceptance; dispatcher
owns implementation/execution/verification and returns decision checkpoints.
The previously completed ramp-history comparison did not test the requested
single-versus-distinct-operating-point acquisition intervention. Frontier selected
that controlled contrast and deferred gate-retention training and physical inversion.
Exact direction: `p4_journal/review_20261009/frontier_research_direction.json`.

Completed proper-fitting-only acquisition preflight: S holds the endpoint;
M uses endpoint-20,endpoint+20,endpoint m/s; R reverses the first pair.
Three128-sample windows with two32ms cubic transitions; continuous mechanical
state,16 sensors, unchanged measured calibration/gates and paired physical noise.
448 samples at1kHz span447ms. Same68 absolute endpoints900–2240m/s applied to
all four original fitting designs, two replicates;1,632 physical episodes and
8,160 noise-view histories. No evaluation/validation/reserved acquisition or fits.
Initial preparation still uses eight simulator modes at.9*ucrit, normalized to
1µm sensor RMS; matching arms does not establish realistic blind preparation.

Numerical evidence:36 representative design/endpoint/arm cases passed the
unchanged refinement and same-equation full/reduced checks; max relative response
RMS9.224e-6 and4.479e-6, respectively; transition-boundary projection gaps zero.
Original CUDA gate passed1440 fixture windows,463 accepted on each backend,
zero classification disagreements. This is a parity sample, not full extraction
or inference replay; numerical agreement is not independent physical validation.

Each primary noise family has758/1632 all-missing histories with exactly identical
zeroed features, flags and ages. Exact equal-design synthetic-dataset MAE floor
.08092836517617448; independent weighted-median enumeration agreed. All these
targets are outside the original margin[-.03,.22] population and exact boundary;
the boundary missing-group floor is zero. Clean boundary has40/40 three-slot
histories per arm. Nonzero waveform contrast does not establish signed-margin gain.

Frontier decision:STOP before full fitting. Its common-grid proposal cannot keep
all training designs in the original margin range: required common speed would be
at least1615.50m/s but at most1234.70m/s. This is an acquisition/population conflict,
not a general impossibility result. Subsequent representative amplitude evidence
also invalidates the extreme-grid small-disturbance physical interpretation:
at2240m/s peak sensor displacement ranges from39.3m to1.39e20m across designs;
worst displacement/panel-length ratio1.525e20. Preserve all cases as documented
acquisition/model-domain failures. Their collision floor is correct for the
synthetic dataset, not evidence of a physical accuracy limit.

Exact decisions: `multipoint_preflight_frontier_decision.json` and
`multipoint_frontier_amplitude_amendment.json` in `p4_journal/review_20261009/`;
complete accounting/numerics/parity evidence is under
`p4_groups/operational_confirmation_20261003/multipoint_preflight_20261009/`.
No predictor, reserved-design score, endpoint claim or progress-ledger change.

Required next checkpoint, no predictor fitting: first establish displacement,
rotation/slope and structural/aerodynamic small-disturbance compatibility for
D000033/D000068 at1240m/s, not merely at the sensors. Then compare the existing
S/M/R trajectories with an independently justified accelerated-flow/memory
treatment, with assumptions, rate/timescale limits and uncertainty smaller than
the measured S-versus-M/R contrast in each order. No configured independent
evaluator or applicable analytical error bound is presently available locally.
Verified NASA abstracts do not supply such a bound; search-summary formulas were
not adopted. See `multipoint_independent_aero_prerequisite.json`. Keep campaign
stopped; no invented rate limit, retrospective filtering or substitute model sweep.

2026-10-09: completed frontier's1240m/s geometry prerequisite on D000033/D000068:
24 S/M/R×replicate0/1×reduced/full structural cases; all1789 quarter-millisecond
samples finite/no termination. Reconstructed physical displacement, analytic
spatial derivatives and FSDT director rotations, with continuous-spatial-domain
triangle bounds at each integration time. Sensor reconstruction agreed within
rtol1e-11/atol1e-14;65x65 bound-peak-time spatial checks stayed within the bounds.
Worst displacement bounds18.5561/15.6388µm, slope norms1.17177e-4/1.73075e-4,
director-rotation norms9.19246e-5/1.16619e-4rad, w/thickness.00226651/.00184794.
No certified continuous-time extrema or nonlinear/aerodynamic error bound.
Initial dense-all-time probe timed out and its owned child was terminated;
the completed optimized probe retains every case/time for analytic bounds.
Exact data, source hashes and executable diagnostic source are saved in
`multipoint_preflight_20261009/checkpoint_geometry.json`.

Returned frontier assessment:SUPPORTED for this narrow geometry screen only;
`p4_journal/review_20261009/multipoint_geometry_frontier_decision.json`.
The obvious amplitude/slope/rotation objection is screened out at the checkpoint,
not for the extreme grid. Do not repeat the same-equation screen. Full acquisition
and fitting stay stopped pending a justified independent accelerated-freestream
treatment or applicable analytical bound addressing32ms transitions and downstream
memory. Its arm-contrast uncertainty must be below both orders' effects,
particularly D000068's approximately5.6% contrasts. No model sweep, invented
rate limit, predictor/reserved-design change or progress credit.

2026-10-09: user asked frontier for next directions that can advance the targets,
not just further feasibility validation. Frontier ranked gate-erasure first:
the majority gap population/error mass supplies leverage, but useful rejected
content remains a hypothesis. Controlled multipoint acquisition is preserved
and paused; its dependencies do not justify a CFD-development detour now.
Direction: `p4_journal/review_20261009/gate_erasure_target_direction.json`.

Authorized next checkpoint is provenance/leverage only, no fitting: replay the
identical174 extractor on existing calibrated128-sample waveforms, classify
rejection reasons, reproduce accepted features, identify genuine finite unsaturated
command-free estimates rejected by forecast qualification, and preserve flags,
age, per-feature availability,383ms histories and every original denominator.
Compute exact remaining-input weighted-median MAE floors and affected/unaffected
comparator error contributions before training. No approximate collisions; ideal
affected-only correction is leverage, not a lower bound for globally refitted maps.
Stop if floors foreclose primary targets or eligible content/error mass is inadequate.
Return the complete receipt to frontier before any fits.

If authorized, same expanded folds/reset GRU32→16 compare zeroing, real rejected
content and one physical-source-paired within-partition shuffled control. No
width/loss/penalty/support sweep. Require primary overall/boundary gains over both
controls without current-boundary, optimism, sign-side, retreat or farther-stable
regression. Failed-gate predictions remain unqualified. Success supports a point
correction, partial gains return residual budgets, failure stops retention and
forks to raw extraction versus acquisition-envelope adequacy.

Began the executable source inventory:2094 cached records×124 frames across20
development designs. Across all development partitions, each primary family has
24552 frames; accepted clean/IID1/correlated1 counts12171/2991/2804, rejected
12381/21561/21748. These are not12-design evaluation percentages or forecast-only
eligibility counts. Local original raw cache contains48 trajectories over8 designs;
all20 waveform shards were checked readable on the original execution host.
SampleD000033 has10×16×8001 measured samples and10×8001 command flags.
No new acquisition or model fitting is required to begin this provenance work.

Source-order correction: additional-row acquisition applies command/finite/cap
checks and prefix forecast qualification BEFORE full-window174 extraction;
original rows reuse canonical accepted features. Rejected full rows were never
stored. Replay must compute the identical extractor on their own measured windows,
not invent or claim recovery of nonexistent estimates; runtime implications remain
explicit. Returned this fact to frontier. Inventory receipts:
`gate_erasure_prerequisite_inventory.json` and
`gate_erasure_remote_provenance_inventory.json` in the review directory.
Complete eligibility/collision/leverage receipt remains to be produced; no fitting
or target improvement claimed and the progress ledger is unchanged.

2026-10-09: frontier accepted the source-order correction and renamed the active
intervention forecast-gate bypass for auxiliary extraction, not recovery of
previously computed features. Eligibility precedes full extraction, full-extractor
failures are counted separately, cached accepted rows stay verbatim, and eventual
runtime must include the additional work. Exact amendment:
`p4_journal/review_20261009/gate_bypass_provenance_amendment.json`.

Executed a CUDA smoke on the original GTX1660SUPER using D000033/cross/replicate0/
IID1:124 windows,13 command rejections,28 cap rejections and83 prefix candidates;
36 accepted,47 finite forecast rejections. The rejected frame1 samples64:192
window has skill.243137945 and produces8 estimated modes/174 finite nonzero
coordinates through the original measured-only extractor. Qualification remains
false; one example does not establish population eligibility or predictive utility.

Accepted frame0 replay(skill.922509334) is finite but differs from its cached
float64 row by at most1.4743761767e-12: NOT bit/numerically identical. No tolerance
widened or exact-reproduction pass claimed; preserve cached accepted rows verbatim
and disclose complete replay differences before training. Receipt:
`p4_journal/review_20261009/gate_bypass_execution_smoke.json`. Returned these facts
to frontier. Full eligibility/reproduction/collision/leverage checkpoint remains
next; no fits, operational changes or progress credit.

Frontier accepted this smoke's narrow scope, not feature accuracy or usefulness.
Exact response: `p4_journal/review_20261009/gate_bypass_smoke_frontier_review.json`.
The1.47e-12 replay difference alone neither establishes material error nor
certifies equivalence. Identify coordinates, standardized-input effects, mode
ordering/availability and gate classifications under existing criteria; do not
require bit identity as a scientific gate or widen tolerances for a pass.
Next authorized work remains the population-wide receipt; fits are unauthorized.

2026-10-10: completed frontier's authorized population-wide forecast-gate-bypass
receipt; no fits. Evidence packet:
`p4_journal/review_20261009/gate_bypass_complete_receipt_frontier.md`.
Outputs: `p4_groups/operational_confirmation_20261003/gate_bypass_receipt_20261010/`.
All2094 rows ×124 frames/20 development designs replayed on the original
GTX1660SUPER. Original measured calibration, sensors, extractor, flags/age,
native frames and source record indices preserved. Labels entered scoring only.
35362 accepted cached rows preserved byte-for-byte;0 refreshed gate mismatches.
141271 rejected full128 extraction trials yielded136969 genuine finite174 rows
and4302 no-mode failures;0 invalid rows or unexpected solver/numerical failures.
All eligible rows had finite failed forecasts. Prefix-no-mode cases produced no
eligible full128 estimates; combined/finite-only sensitivity therefore agrees.

The endpoint population stays the original12 grouped evaluation designs, not
the20-design inventory. Each primary view has14880 native frames. Clean:
7306 accepted,140 newly eligible,2601 full-no-mode and4784 prohibited windows.
IID1:1805 accepted/8293 eligible; correlated1:1695 accepted/8403 eligible.
Qualification remains unchanged; this is auxiliary input, not new availability.

Exact byte-identical float32 three-slot inputs, including unchanged flags/age
and warmup padding, grouped within each fold. Weighted-median L1 floors use
ORIGINAL equal-design all/boundary weights, never missing/affected renormalization.
Clean all-frame floor=.012942400855>.010 target; ALL3 folds fail individually:
.012881994839/.013693977896/.012251229831. Remaining no-content histories account
for the whole floor. Clean boundary floor=.002653528564<.005; IID1/correlated1
all/boundary floors approximately.001404834459/.001131989974, below their targets.
Thus the predeclared stop-before-fitting rule fires on clean all-frame accuracy.
No representation fits launched; no raw-waveform impossibility claim follows.

Clean retained content changes only2.796% all-frame/0.587% boundary original weight.
Even perfect correction there leaves old GRU error .030607015/.026068018.
This is correction leverage with unaffected predictions held fixed, NOT a bound
on globally refitted predictors. Noisy changed populations are much larger, but
cannot remove the independent joint-target clean collision foreclosure.
All4 comparators/3 seeds, fresh noise,5% stress and per-design contributions saved.

Reproduction is not certified equivalent. Old canonical cached feature rows show
raw maximum4.473014311 and active standardized maximum.459203; one cached first3
frequency-order mismatch, no mode-count/availability change. All25935 accepted
original-added-regime/additional-trajectory rows reproduce exactly. Cache never
replaced and no tolerance widened. Prespecified6-window paired batching panel
agrees to9.676e-13 standardized. A separate, frozen reproduction-error-triggered
ONE D000007 clean/frame42 comparison explains the earlier .0087635455 raw and
.0014807480 standardized discrepancy: canonical per-frame batching matches cache
exactly; proposed per-row batching reproduces the discrepancy, with unchanged
7-mode/gate/spatial-order states but changed within-mode coefficients. This proves
batch sensitivity there, not universal equivalence elsewhere.

Independent dictionary-by-bytes/explicit-slot verification:42 primary fold/noise/
stratum checks; maximum floor discrepancy1.041e-17 and comparator contribution
discrepancy6.939e-18. Source rows once, finite/nonempty eligible estimates and
accepted bytes verified. Extra rejected full-mode extraction473.912s; combined
accepted/rejected feature assembly258.058s also reported. Not deployment latency
or runtime qualification. Complete receipt returned to frontier for explicit
stop/next-mechanism direction. Predictor, reserved evaluation and progress ledger
remain unchanged; no headroom, warning, calibration or endpoint improvement claim.

Frontier returned CONCERNS and explicitly stopped gate-bypass fitting:
`p4_journal/review_20261009/gate_bypass_complete_receipt_frontier_decision.json`.
No campaign to repair this foreclosed arm or global equivalence claim.
Next authorization is metadata-only: partition remaining collisions into histories
with no permitted command-free/finite/cap-free raw response within383ms versus
histories with permitted raw response but no modal content; supply ORIGINAL weights,
profile/target ranges and comparator error contributions. No acquisition, extractor
comparison or fits. A raw-response experiment needs this partition and authorization.
Requested clarification: existing exclusions are per128-sample window, not proof
that no shorter permissible waveform snippet exists; also distinguish modal slots
4/2/0 from all5 existing overlapping windows within the same383ms physical span.

Frontier narrowed the wording: full128-window bounds are representation bounds,
not immutable information limits. It authorized per-sample command/finite/cap-mask
inspection, including shorter snippets, with no acquisition/extraction/fits:
`p4_journal/review_20261009/raw_response_partition_authorization.json`.

Completed that audit on all20 measured shards, then partitioned the original12
grouped populations at every native frame. Original383ms aperture preserved.
Clean all-frame raw categories (native sample rule):
full128 available76.0551%, shorter-only1.6667%, no complete16-channel sample22.2782%.
Contributions at FULL original modal-class medians:
.008956323/.001315494/.002670584, summing to .012942401.
The full128 category contains13.3468% raw-present/no-modal histories; farther-stable
profiles (margin.162320..219066) alone contribute.007456763 while raw remains present.
This identifies a representation question, not useful estimability.

Restricted no-sample modal floor=.001224267 all/.000996109 boundary, below targets.
Known causal sample masks/timebase split those classes further: .000082590/
.000001596. Explicit keys include original float32 modal/flag/age/padding inputs,
five384-bit presence/command/finite/cap/admissibility masks and current sample end.
No design/target/state/poles inserted. Leading calibration constants remain preserved
but are not extra equality coordinates; no claim about every causal predictor.
Three-slot and five-overlapping-window partitions separately reported.

Source gate is per-window, not permanently latched after first cap. A SEPARATE
conservative global pre-cap sensitivity was computed, not adopted: its no-sample
clean floors .004414894/.001415768 also stay below targets in every fold.
No capped value or post-cap predictor input was used; no predictor was called.

Important remaining leverage obstacle: preserving old no-sample GRU predictions
alone contributes boundary MAE .0102211 clean/.0102320 IID1/.0102359 correlated1,
above .005/.010/.010 targets even with perfect corrections elsewhere. This is NOT
a globally refitted predictor bound; a raw replacement with the identical old gap
fallback is insufficient. A mask-aware fallback would need prospective justification.

Independent verification:504 restricted-floor dictionary checks,252 categorical
count/weight checks and386880 explicit sample-mask/timebase/run-length checks;
maximum floor discrepancy4.337e-18. All156 report partitions preserve original
weights and sum back to prior class floors and comparator errors. Receipt:
`p4_groups/operational_confirmation_20261003/gate_bypass_receipt_20261010/raw_response_partition/`.
Complete packet returned to frontier:
`p4_journal/review_20261009/raw_response_partition_frontier_packet.md`.
No fits, qualification change, reserved evaluation or progress-ledger change.

Frontier returned SUPPORTED and chose ONE bounded raw-content experiment after
implementation preflight, not more modal repair. Exact response:
`p4_journal/review_20261009/raw_response_partition_frontier_decision.json`.
R: existing three-slot174 modal branch plus raw GRU32 over384 physical samples,
joined through a fixed16-unit scalar margin head. M: SAME architecture/masks/ages/
training, but all waveform values zero; this isolates raw content from capacity,
richer missingness and refitted gap predictions. R must beat M, not merely frozen
references. Both encoders reset at every decision; oldest waveform sample383ms.
Preserve calibrated16-channel amplitudes, actual snippet positions, source
per-window exclusions and qualification; no interpolation, clipped inputs,
trajectory-wide cap latch, absolute schedule labels or simulator quantities.

Frontier explicitly rejects preserving the old gap fallback. Both arms train on
every original frame, including no-sample histories, with masks/ages available;
those estimates remain unqualified. Same original3 folds,4+8 fitting designs/
checkpoint-validation designs,5 profiles,3 seeds42/43/44 and training/loss/
regularization/checkpoint convention.18 fixed fits, no architecture/loss search.
Before ANY fits: frozen input/architecture specification, actual causal/reset/
invalid-value/calibration/cache preservation proof, exact zero-waveform R→M
input identity, and collision bounds under the ACTUAL proposed encoding.
Do not import this audit's low bounds if the candidate omits mask coordinates
or exclusive endpoint time; actual candidate uses relative sample times.
Fits require passing that preflight and actual-encoding primary floor gates.
Scalar point estimator, not a calibrated quantile claim. Raw-specific primary
overall/boundary gains and original design/seed effects required; unchanged
sign/current-boundary/optimism/retreat/farther-stable safeguards and absolute
targets remain. No results or progress credit for this unexecuted experiment.

User rejected the experiment chain for failing to deliver target-performance
progress. Current authorizations are withdrawn, including the raw-versus-mask
preflight/fits above. No new experiments or audits are authorized.
Frontier subsequently withdrew its claim of an evidence-backed route to the
endpoint and stopped the transport, spectral, multipoint-physics and raw-model
proposal chain. Preserve existing results/predictors as references; none promoted.
Extraction error, cross-design margin conversion and unobserved-gap prediction
remain distinct unresolved problems. A substantive research-strategy reset is
required, not automatic continuation of the recorded experiment proposals.

2026-10-10: user resumed work with the existing frontier lead. Authorization
restored only for the fixed raw+modal R versus zero-waveform M contrast above;
no acquisition, source/model/loss search or broader experiment chain.
Frontier clarified that R's actual-encoding bounds gate execution; M's bounds
are a diagnostic control, not a veto. Fixed microbatch2048 gradient accumulation
preserves full native-frame mean MSE and one Adam step per epoch.

Integrated preflight passed on the original GTX1660SUPER execution host:
2094 rows matched the original rejection masks and calibration offsets/scales;
causal384-sample slicing, warmup, invalid-value exclusion, reset/order invariance,
unchanged cached modal content and exact waveform-erasure identity passed.
All42 primary fold/noise/stratum candidate-R bounds passed; maximum .001425733.
Protocol/preflight/acquisition receipts:
`p4_groups/operational_confirmation_20261003/raw_modal_20261010/`.
The18 fixed fits are running. This is execution authorization, not measured
target-performance progress. Predictor, qualification, protected evaluation and
progress ledger remain unchanged; outcomes must include original-denominator
no-sample contributions and raw-present/no-modal effects.

2026-10-10: completed all18 fixed raw+modal/zero-waveform fits and39 development
fold/noise scoring conditions. Independently recomputed336 primary per-design
overall/boundary errors and original frame denominators from saved predictions;
all matched. Protected held predictions remain NaN. Sources/results:
`p4_groups/operational_confirmation_20261003/raw_modal_20261010/`;
`analyze_raw_modal.py` generates `comparison_summary.json`.

Mean equal-design MAE (seedwise errors averaged, not ensemble predictions):

| Condition | R overall | M overall | Best frozen overall | R boundary | M boundary |
|---|---:|---:|---:|---:|---:|
| clean | .023071 | .023697 | .031000 | .014334 | .014870 |
| IID1 | .037740 | .038139 | .048083 | .030983 | .031856 |
| correlated1 | .038794 | .039178 | .050112 | .032820 | .033494 |
| IID1 fresh20261020 | .037914 | .038366 | .048141 | .031298 | .032123 |
| correlated1 fresh20261020 | .038994 | .039293 | .050950 | .032975 | .033493 |
| IID1 fresh20261021 | .038094 | .038469 | .048744 | .031226 | .032094 |
| correlated1 fresh20261021 | .038861 | .039228 | .050299 | .032593 | .033354 |

Best frozen overall is the lowest measured error among the unchanged modal GRU,
ordered concatenation, unordered pooling and framewise references in that
condition; no pointwise routing or new baseline selection model.
Both new arms substantially improve overall/boundary errors versus the frozen
references. The additional R-over-M mean gain is small: .000299–.000626 overall,
.000518–.000872 boundary. Clean paired design intervals include zero; one fresh
correlated overall interval also includes zero. All three mean overall seed
effects favor R, but individual design/seed effects differ.
Bootstrap:2000 seed42 design resamples retaining all3 paired seeds; no refit/seed
uncertainty or multiplicity correction. Twelve designs, not independent windows.

Raw-present/no-modal subsets favor R: clean overall delta−.0015293,
boundary−.0036541; corresponding paired intervals exclude zero. All six noisy
subset mean gains/intervals also favor R; correlated boundary seed43 can reverse.
The large shared improvement must not be attributed to raw waveform content:
clean original-weight no-sample overall contribution falls from .0087042 for
the frozen modal GRU to .0015248(R)/.0015193(M). Boundary contribution falls
from .0102211 to .0017413(R)/.0017194(M). Original no-sample weights
.222782(overall)/.263908(boundary); no renormalization. Capacity, masks and
refitting remain bundled in M, not individually identified.

No adoption: all7 primary conditions fail both absolute MAE targets and the5%
boundary optimism gate. R optimism is12.8795% clean and41.8249–44.4641% noisy.
Available wrong-sign fractions meet the absolute10% limits, but unstable-side
errors worsen relative to the frozen modal GRU. Retreat worsens against all
four frozen references in every primary condition. Correlated current-boundary
MAE also worsens against the frozen modal GRU; farther-stable non-inferiority
passes. This is measured partial development-error progress, not endpoint,
qualified-warning or independent-transfer success. Predictor, qualification,
reserved evaluation and P4_PROGRESS remain unchanged. Returned measured evidence
to frontier for a concise decision on this completed comparison; no new campaign.

Frontier verdict CONCERNS: real partial accuracy progress, smaller/heterogeneous
raw-specific benefit, no promotion. Close this fixed experiment without variants;
retain R/M as improved development references only. Exact assessment:
`p4_journal/review_20261009/raw_modal_completed_frontier_decision.json`.
No new experiment or audit authorized; no operational, qualification or ledger
cutover.

2026-10-10: user authorized trying stronger models/training constraints instead
of further diagnostic campaigns, retaining useful information from R/M.
Retained all174 cached modal features, observation flags/ages, per-sample
admissibility, relative time and16 calibrated raw channels. No gate-bypass
features, source/qualification change or new acquisition.

Two fixed matched-seed trials, same3 folds and42/43/44 seeds (18fits):
`tcn_mse` and `tcn_risk`. Both replace the raw GRU with the existing residual
TCN32,7 two-convolution blocks/kernel3/dilations1–64, no dropout. Receptive
field509 covers the384-sample input; extra context is zero padding, not more
observations. Early64/late64/last latent summaries project96→32; retain modal
GRU176→32 and scalar64→16→1 head.

MSE arm retains native-frame MSE. Risk arm uses .5 all-frame MSE + .5 boundary
MSE +2 squared overestimation hinge above.02 in the boundary slice + a squared
signed-margin hinge at.005 on current eligible sign points + a squared farther-
stable underestimation hinge above.005. Eligibility uses original float64 labels,
only in training loss. Each term uses its full fitting-stratum denominator.
These are soft penalties, not guaranteed hard constraints. They localize
optimism pressure and counter farther-stable pessimism rather than globally
shifting estimates down. Same optimizer/regularization, all-native-frame
validation MAE checkpoint and2000epochs/patience100; no loss-weight search.
Fixed microbatch512 accumulation still takes one Adam step per epoch.

Actual CUDA forward/backward on512 measured decisions passed; causal-prefix,
reset/order invariance and analytic loss-gradient/global-normalization checks
passed.69,281 parameters; smoke peak allocation1,276,727,296bytes. Reuse the
completed identical-input calibration/exclusion proof and input bounds; no new
bound/feature diagnostic campaign. Frozen protocol:
`p4_groups/operational_confirmation_20261003/raw_tcn_20261010/protocol.json`.
Fits running; results must compare R/M and all prior references using unchanged
original denominators, current/sign/retreat/optimism/farther-stable safeguards.
No measured improvement or operational/ledger cutover claimed yet.

2026-10-10: interrupted this TCN run at user's selected "Stop and optimize".
Snapshot9209.85seconds after protocol freeze showed0/18 completed fit artifacts.
That is not a measured per-fit average; the earlier two-day estimate was only
a conditional extrapolation. Actual epoch throughput was not measured before
launch, an execution error. Tailscale-only access is now verified; training
PID38364 received SIGTERM and its exit was confirmed. No new accuracy results.
Do not automatically resume this expensive protocol.

Workload counted from actual frozen records: each fold has12 fitting designs,
240 clean/IID1 trajectories ×124 native decisions =29,760 training decisions;
4,960 validation decisions. Microbatch512 means59 training and10 validation
batches per epoch, up to2000epochs per fit. Each raw window passes through
14 width32 temporal convolutions plus the retained modal GRU. The maximum
59.52million training-decision presentations per fit is a workload bound,
not evidence identifying the slow kernel. A bounded actual-data timing/
profiling probe is running now; no full fitting restart or model/input/loss
change has yet been made.

2026-10-10: execution optimization tested before any training restart.
Actual512-window dense step median217ms. Profiler: convolution backward49.77%
of device time, LayerNorm forward/backward about26%. Channels-last Conv2d was
much slower; autotuning and compilation did not produce a useful gain.
Smaller microbatches were slower with both original global-normalized losses.
Do not adopt those changes.

Adopted equivalent polyphase training convolution in `experiment_raw_tcn.py`:
split temporal residue classes modulo dilation, run nondilated causal Conv1d,
restore temporal order. Original dense weights/state names, float32, all
inputs, receptive field509, losses, folds and optimizer steps are unchanged.
Inference keeps the native dilated path: polyphase forward alone was slower.
Interrupted source preserved in
`raw_tcn_20261010/interrupted_source.tar.gz`; that protocol is not resumed.

Full actual fold0 epoch proof (29,760 fitting +4,960 validation decisions):
MSE13.383848→11.096455seconds; risk13.492685→11.240913seconds. About17% shorter,
not an order-of-magnitude speedup. Full-frame objective/MSE matched numerically;
maximum gradient discrepancy1.86e-8, maximum parameter difference after one
Adam step7.89e-6. Float32 reduction-order differences remain; this is not a
claim of bit-identical training trajectories.20 CPU forward/input-gradient/
weight-gradient/causality edge cases passed across dilations1/2/4/8/64 and
lengths7/65/129/384, including lengths not divisible by dilation. Actual CUDA
512-decision causal/reset/order/finite-backward and analytic-loss checks passed.
`raw_tcn_polyphase_20261010/execution_verification.json` and `protocol.json`
preserve proof and frozen sources. Freeze requires proof on the current source.

User also requested a modern convolution candidate. Depthwise-separable causal
blocks passed actual finite-gradient/causal/reset/order checks but only improved
512-window steps217→182ms, increased activation memory, and reduced parameters
69,281→42,401. Not adopted over the comparable-speed equivalent dense rewrite.
Intermediate grouped variants are being tested separately; they change
channel-mixing capacity, so runtime checks do not establish margin accuracy.
No18-fit restart, target-performance improvement, qualification, operational
predictor or progress-ledger change. At11.2seconds/epoch,2000epochs still project
to about6.2hours per fit before setup/scoring; the full plan remains expensive.

Grouped convolution requested and tested before any adoption. Native cuDNN,
same512-window batch, median of5 forward/backward/Adam steps:
groups1 217.218ms/69,281params; groups2 425.143ms/47,777;
groups4 155.345ms/37,025; groups8 152.093ms/31,649. Polyphase grouped kernels
were also measured: groups2/4/8 were380.291/168.939/196.586ms, so native
grouped execution is preferable for4/8. All variants passed finite gradients,
causality, decision-reset and batch-order checks. This is not an accuracy test.
Groups4 is the conservative grouped candidate: approximately28% shorter than
original dense steps; groups8 buys only about2% more speed for less channel
mixing/capacity. Neither is adopted into the frozen dense comparison without
margin-error evidence. All sensor/modal inputs remain; a grouping change is
an architectural restriction, not an equivalent dense-convolution optimization.

Evidence: `raw_tcn_polyphase_20261010/grouped_convolution_probe.json`,
`grouped_native_convolution_probe.json`, and `execution_probes_source.tar.gz`.
Reusable full-epoch verifier added: `verify_raw_tcn_execution.py`; exercised on
the frozen source,20 edge cases and both full-loss epochs passed again. Frozen
verification was not overwritten. Full fitting stays stopped.

2026-10-10: user selected groups=4 with the existing dilation schedule and
authorized integration and running, ending convolution-variant search.
`experiment_raw_tcn.py` now constructs native groups4 causal residual blocks
directly, reusing the existing padding, normalization, residual and summary
behavior; no discarded dense-block initialization. All16 measured sensors,
174 modal features, flags/ages, sample admissibility and time remain.
Dilations1/2/4/8/16/32/64, two convolutions per block, receptive field509.
The stopped dense protocol/evidence remain archived under
`raw_tcn_polyphase_20261010/`, including `frozen_source.tar.gz`.

Integrated verification passed28 grouping-isolation/causality/gradient cases,
actual512-decision reset/order/loss-backward checks, and full fold0 epochs for
both objectives. Same29,760 fitting and4,960 checkpoint-validation decisions:
MSE train9.535030s + validation0.585383s =10.120412s;
risk train9.272755s + validation0.585189s =9.857944s.
37,025 parameters; full-epoch peak allocated2,292,174,336bytes.
Finite gradients and post-Adam parameters verified. Different architecture,
not an equivalence claim or an accuracy result.

New frozen protocol/proof: `raw_tcn_grouped4_20261010/`.
The fixed18fits (MSE/risk,3 folds,3 seeds) launched as detached user service
`p4-tcn-grouped4-20261010.service` on the original CUDA host via Tailscale.
Startup observed active/running, MainPID61820; fold0/MSE/seed42 epoch1 completed
in9.953866s. The service runs training, original-denominator scoring and guarded
comparison reporting in sequence. The waiting SSH connection subsequently
timed out; the independently observed service remains the execution owner.
Do not launch duplicate fits. Read service state before any recovery.
Existing input bounds, splits, optimizer/checkpoint rules, all frozen references,
target/safeguard gates, held-design protection and progress ledger unchanged.
Training is running; no completed-fit accuracy or promotion claimed.

2026-11-02: research continuation and orchestration review.
User authorized supervised delegation, resource accounting and small,
evidence-backed global orchestration improvements; all ten acceptance gates
in P4_RESEARCH_TARGET.md remain unchanged.

Dispatcher mistakes: continued serial log/setup exploration after enough
context existed to dispatch independent lookups; let orchestration bootstrap
displace research; omitted the known invalid COMSOL shape corpus from the
source-lookup brief. Evidence is the current parent session
2026-11-02T15-15-44-598Z_01a19cd4-a74f-77b9-9b1f-a01ca8e4b0c2,
including user corrections and IndependentSourceRoute's corrected handoff.
These are dispatcher failures, not grounds to blame worker capability.
Global AGENTS.md now requires a post-scope dispatch checkpoint, relevant
invalidated evidence in briefs, and self-monitoring/correction without
waiting for user reminders. Added research clauses distinguish exploratory,
oracle, numerical and independent physical evidence; require actual-data
cost probes and preserve denominators. These are prompt mitigations,
not measured reliability improvements.

RecoverTCNOutcome established SSH/Tailscale availability and no currently
observed Python training owner. The recorded grouped4 service is not-found,
with no journal entries; ExecMainStatus0 is not completion evidence.
Remote /home/omid5/mechanics is a dangling link into unmounted Btrfs UUID
a6a76b35-99a8-4e07-a4e0-18c05f0be50e, currently /dev/sdb5.
Parent verified UUID with lsblk. Read-only UDisks mount was denied by polkit;
sudo -n also requires a password. Original remote fit/results counts remain
unknown. Local copy has zero completed checkpoint/prediction pairs and a
temporary first-fit artifact, which does not establish the remote count.
No duplicate training, destructive recovery or reserved evaluation launched.

IndependentSourceRoute found no qualifying independent aeroelastic source
in its bounded inspection. COMSOL shell eigenfrequency studies are a possible
structural component, not coupled flutter validation; saved shape exports
remain VOID per RESEARCH_LOG.md:139-141. Scalar frequency calibration is
unaffected. plate_main's alternate plate implementation uses the same
first-order piston aerodynamic form and does not satisfy the independent
model-form gate. The scout initially omitted the void-corpus caveat; the
dispatcher supplied it and the worker corrected the handoff. Its two failed
semantic searches were already in flight together, not retries after402.
No gate, predictor, qualification or progress-ledger promotion follows.

Resource snapshot at2026-11-02T15:40:48Z: model-reported parent session
USD0.6346364 (includes preceding orchestration conversation), reviewed workers
USD0.1374432, totalUSD0.7720796. Worker session message-usage sums are scout
0.0828144 including its correction and recovery worker0.0546288; checked that
the two scout lifecycle entries describe separate usage increments rather
than summing unverified cumulative snapshots. Billed credits are unknown.
Both outcomes entered ~/.omp/agent/telemetry/reviews.jsonl. Scout definition
now distinguishes intended generator output from validated artifacts and
requires invalidations in evidence handoffs; fresh-spawn loading/effect has
not yet been verified on a new real assignment. Do not claim improvement.

Executed statistical feasibility check with local Python/NumPy/SciPy:
for a candidate simultaneous family of11 rare-event statements
(3primary+2stress unstable-clearance risks,3primary timely-miss risks,
3primary nuisance rates), Bonferroni per-statement alpha=.05/11.
Exact one-sided binomial upper bound at0/120 independent design-run events
is0.04395174974. At1/120 it is0.06113436186, failing the5% risk target.
Minimum zero-event samples are106 for5% and52 for10%; with120 samples,
timely misses may be at most3 (upper0.08945905245);4 yields0.10227135270.
Checks asserted the zero-event sample boundary and0-versus1 event consequence.
Thus120 designs support these particular bounds only at very low event counts;
they do not guarantee an attainable gate, independent physics or adequate power.
This11-statement family is a planning calculation, not a frozen multiplicity
policy for all10 gates. It assumes one prospectively defined independent run
outcome per design/condition; repeated windows/trajectories are not extra designs.

The zero-count Poisson calculation gives5.393627546 hours per nuisance
condition at the same adjusted alpha and1episode/hour target. This is only a
conditional bound under a justified count/exposure process, not an exposure
recommendation for clustered design-hour runs. A prospective clustered design
and interval method remains required. No risk or nuisance observation was made.

Continuation decision: restore original volume access, inspect completed
grouped4 pairs and frozen source/input provenance, then score completed fits
before authorizing further training. Preserve all original denominators and
guardrails; no architecture search or blind evaluation while outcome is unknown.
Independent physical validation remains a separate unresolved prerequisite.
Local scientific runtime has NumPy2.5.0/SciPy1.18.0 but no Torch; do not substitute
a CPU rerun for recovery of the existing CUDA experiment. All10 gates remain
unqualified; acceptance status and P4_PROGRESS.json are unchanged.

One bounded external-source lookup identified an independently formulated
FSI route, not an available matching validation dataset. Official SU2 dynamic
FSI documentation uses compressible RANS coupled to mass-normalized Nastran
modal structures, with separate fluid/solid/coupling convergence. Its supplied
case is a2DOF NACA0012 atMach.1-.364, not this supersonic honeycomb plate:
https://su2code.github.io/tutorials/Dynamic_FSI_Python/
The NASA-hosted January2025 Ganti/Bravo/Ghosal/Khare slides describe coupled
compressible-flow/nonlinear-elasticity flexible panels, includingMach2.11
thin-aluminum work explicitly marked unpublished andMach5.86 panel examples:
https://nescacademy.nasa.gov/workshop/AePW4/Slides/high_speed/UC_HSWG_January_2025.pdf
Neither source establishes a supplied runnable compatible CFCF sandwich-plate
case, valid flutter labels, or a benchmark at the proposed32ms speed changes.
The slides' double-cone pressure validation is not validation of plate flutter.
No solver installation or coupled-case construction was attempted.

Parent current-state ledger check verified all10 gate entries remain false,
documented attainment20.7614398398%, and exact protected SHA256
ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f.
This is a provenance/status check, not new performance evidence. Remote result
recovery remains blocked on an authorized mount; independent physical source
qualification remains unresolved. Goal remains active, not completed.

Agent-led storage recovery follow-up: RecoverTCNOutcome inspected fstab,
existing mounts and narrowly scoped alternate project locations. No original
UUID mount was present; the named alternate mechanics directory and Projects
were empty. The remaining prerequisite is interactive authorization of a
read-only mount, not SSH connectivity or a proven lost experiment. Exact action
on cachyos: sudo udisksctl mount --block-device
/dev/disk/by-uuid/a6a76b35-99a8-4e07-a4e0-18c05f0be50e --options ro
--no-user-interaction. Use its reported mountpoint, which may differ by caller.
No repair, policy changes, package installation or training was attempted.
The worker initially used a loop incompatible with the remote fish shell and
self-corrected. Dispatcher had not supplied remote-shell identity. Evidence:
history://RecoverTCNOutcome, follow-up recovery handoff. Worker preflight now
requires establishing remote shell or explicitly choosing an available shell;
this prompt mitigation has not yet been verified in a fresh real assignment.

Recovery checkpoints after user-authorized mount: the GUI/PTY terminal attempt
was a dispatcher failure, not a user error. An agent-owned PTY was not proven
user-accessible; a launched Kitty process did not establish a visible window.
Its exit0 did not prove mount success. RecoverTCNOutcome's subsequent SSH
check correctly found no mount and did not retry privileged operations.
Preventive choice: use a user-run one-line SSH command for interactive secrets
unless the user-facing input surface has actually been established.

User then supplied successful mount output for /dev/sdb5 at
/run/media/root/a6a76b35-99a8-4e07-a4e0-18c05f0be50e. A newline executed
--no-user-interaction separately in fish; this did not undo the mount.
The recovery worker confirmed Btrfs ro, top-level subvolume, but namei showed
/run/media/root owned root:root0750 blocking omid5 traversal. Files inside
remain uninspected, not missing. User was supplied a one-line SSH sudo setfacl
command granting only omid5 execute/traversal on that parent. No training owner
was observed; no artifact, service, symlink or permission change by an agent.
The worker explicitly verified its updated remote-shell preflight clause
loaded on this real resumed assignment. A fresh-spawn load and comparative
reliability effect remain unverified.

Five-assignment research delegation retrospective,2026-11-02T19:08:52Z:
source scout corrected after dispatcher supplied known invalidation; recovery
worker delivered four accepted bounded investigations (original inaccessible
storage, alternate paths, mount-exit verification, mounted-path permissions).
Runtime completion was not confused with experiment completion. Reusing the
worker preserved remote context; the dispatcher repeatedly caused avoidable
friction through serial exploration and unverified password-input surfaces.
Keep bounded worker routing; no evidence supports expanding scientific
interpretation authority. One next real-work observation: check whether the
same concise recovery brief yields directly usable original fit provenance
after traversal authorization; no synthetic test or additional benchmark.
Reported worker-session costs checked from message usage, not by summing
possibly cumulative lifecycle entries: scoutUSD0.0828144, recoveryUSD0.1846672,
combinedUSD0.2674816. These are reported model costs, billed credits unknown.
Parent/other tool/compute costs are not included in that worker-only subtotal.

Continuation evidence after traversal authorization: mounted checkout commit
d6622432b03846c26b1ae570000b2e842cb0acb9 predates the October10 experiment;
the operational_confirmation directory is absent there. Original run outcomes
remain unknown. Mounted cp311 package metadata exists, but its absolute
interpreter link is broken. Remote standalone-Python download failed DNS twice;
whole-tree rsync hit its180s deadline. A measured1MiB single-file scp succeeded
in24.02s including connection startup. Authorized existing33,496,830byte local
standalone archive transfer with1200s bounded deadline; no repeated probe or
download. Real overlay/CUDA proof is still pending.

Local frozen-input inventory:20/22 grouped4 pins exist (~129MB); missing
waveforms.npy and admissible.npy, and runtime-required20 physical shards.
All21 support acquisition protocol inputs and12 implementation paths exist;
the8 original physical-state archives and source raw_signals.npz also exist.
This establishes a possible reconstruction route, not deterministic hash
reproduction. No missing-input/provenance check was bypassed.

Dispatcher correction: the earlier claim that the local scientific runtime
lacked Torch was based on /usr/bin/python, not the project interpreter.
Actual .venv/bin/python imports Torch2.12.1+cpu, NumPy2.4.4, SciPy1.17.1 and
sklearn1.9.1; CUDA false. Check the project launcher before declaring runtime
absence. This error delayed reachable CPU research and is not a worker failure.

Authorized one fixed CPU modal-history histogram-boosting comparison while
raw/CUDA recovery proceeds independently. Same cached174 features plus flags/
age in three causal slots, original3 evaluation folds, proper clean/IID1 only,
200 rounds/.05 learning rate/15leaves/minleaf100/L2=1/no early stopping, seed42.
No search, held predictions or raw-source loading. A representative fullfold
fit must measure actual runtime/memory before the remaining2fits; reuse its
artifact rather than refitting. This limited-input competitor tests nonlinear
conversion, not waveform attribution or independent physical validation.
Worker owns a new runner only; parent owns execution and independent scoring.

Completed the fixed modal-tree comparison. Parent ran actual `preflight`
(75.55s wall) then `run` (141.53s wall), retaining completed fold0. Fold fitting/
prediction receipts60.4344/60.2938/58.6396s; process high-water RSS peaks
1,526,688/1,581,264/1,611,632KiB, not isolated per-fit allocations.
Primary clean/IID1/correlated1 all-frame MAE=.028681/.047410/.049233;
boundary=.022863/.033179/.033837. All7primary views fail both point targets
and boundary optimism, which ranges34.23–57.34%. Current sign-side fractions
stay below10% and retreat improves versus all6comparators; farther-stable
non-inferiority fails versus raw/mask in every noisy primary view. Reject
the fixed candidate without tuning. Nonlinear modal conversion alone did not
improve this comparison; this is not a proof of absent waveform information.

VerifyModalTree independently recomputed2,485 numerical/population comparisons,
maxdifference4.163336342344337e-17. All input/output hashes and protected
progresshash match; every fold excludes validation/evaluation/held designs
from fitting, and29,016 held predictions per fold stayNaN. Fold0 byte identity
matches its original receipt. Exact exercised auditor preserved as
modal_tree_20261102/verify.py, SHA256
f13bc4ee7d17eb200f99a3daf61375ef4c08eedd922aa5dd7de471ed7baa37ab;
verification.json records the original/tmp command and observed results.
No new run under the preserved path was claimed; owned temporary copy removed.
One seed versus three-seed comparators and limited modal inputs remain limits.
No gate, operational predictor, qualification or progress-ledger cutover.

CompleteCUDARecovery verified the33,496,830byte interpreter archive on both
hosts, matchingSHA25601b71b5fb231bc932d4e0f5d93417860cefa2fc0c4212ec7dad28c08fa66d914.
RecoveredPython3.11.15 with existing mounted NumPy2.4.4/SciPy1.17.1/
Torch2.12.1+cu130 imports worked; actual synchronizedCUDA2x2onesmatmul
returnedall2. Runtime prerequisite passed, not research-gate validation.
One-design immutable restoration protocol frozen at recovery_20261102/
protocol.json: recreate D000007 only, exact original shard hash/size required,
600sacquisition bound, oneGPUowner; stop on mismatch before scaling.
Staging manifest verified21input+12implementationpins,271files/99,686,089bytes;
compressed95,507,253byte archive, SHA256
717bcf5e241c8a40c0b658a277812a679922d0c255bd9b2a45083882cc7aa23a.
One bounded1800s transfer is owned by CompleteCUDARecovery; no duplicate.

PrimaryPanelAssets delivered partial public-source evidence only after its
600sruntime expired and parent requested existing findings without new fetches.
It verified NASA RC19/HyMAX public ZIP endpoints, small archive names and RC19
MAT top-level schemas; no compatible signed-headroom label source established.
RC19Mach1.92 and metallic clamped geometry differ from the currentMach>=2
CFCFhoneycomb family. HyMAXMach5.8 cantilever/shock data also differ; its
25,746,118byte data package was not downloaded under the10MiBlimit.
RC19July2023 slides use modal plate/piston theory, not independently coupled
CFDflutterlabels. Evidence links are in history://PrimaryPanelAssets and the
official NASA AePW3 HighSpeedWorkingGroup page. No independent-credibility
credit. Researcher-role attempt failed before content with NoModelSelected;
no model/account routing changes were made.

Observed asset-worker failure: network batches timed out180s and kernel exited,
with no substantive handoff before the overall600slimit. Dispatcher contribution:
bounded sources/bytes but no explicit per-fetch deadlines or early checkpoint.
One recovery request returned useful partial facts; do not restart that search.
Worker preflight now requires first-evidence checkpoints and per-request
timeouts for long external operations. This is a prompt mitigation, not a
measured reliability improvement; fresh-spawn effect remains unverified.
Next scientific action remains the one-design restoration pilot, not another
tree/loss/model sweep or unsupported public-data transfer claim.

Restoration staging stopped before execution. The worker's600slifecycle
cancelled its still-active1800sSCP child:95,507,253byte source archive left
66,063,360bytes remotely, no extraction, acquisition or receipt. Dispatcher
contribution: assigned an operation longer than worker lifetime without a
durable independent transport owner. CompleteCUDARecovery correctly preserved
owner/partial-byte evidence and did not relaunch. Global dispatcher rules now
require durable ownership and checkpoint/resume details before waiting on
operations that may outlive workers. This is mitigation, not measured effect.

Parent assumed explicit transfer ownership and used rsync--append-verify to
resume the same archive, not retransmit fromzero. First attemptSSHConnectTimeout10
failed after10.08s; one retry withConnectTimeout30 failed after30.08s, both
rsync255/0bytesreceived. Worker separately observed peeronline/active but its
single boundedTailscaleping returnednoreply. Earlier helpercopies succeeded;
that earlier connectivity is not evidence of present reachability. No further
network retries or privileged changes authorized. Current archive size is
unverified since connection loss; last observed partial remains66,063,360bytes.

Small remote helper preparation had succeeded with byte hashes:
run_pilot.py728860e2abb6dd2f4aa536512151e4648cd92e1469cec6a36e76339e6f8da5b3,
pilot_manifest.json652b6cd234b4a54873d92a4cd3bb141eda3c8d4695fb1a601eb07b21318e6ecb.
User systemd manager wasrunning; no pilotunit was launched.
Parent review found the proposedSIGALRM600 could defer inside nativecode.
Local control now uses subprocess.run(timeout=600), with its scope explicitly
including childimports/contracts/acquisition, inside the proposed900stotalunit.
A native5ssleep was interrupted at.216912s by a.2s subprocess deadline and
childreaping was checked; syntaxpassed. This exercises the watchdog primitive,
not the realGPU acquisition. Durable local updated controlSHA256
274eda1b5b769f2054e0503a574354496bed37a0e27bf51dc0bce347b4952ea3
is preserved in recovery_20261102/run_pilot.py; remote still has the old helper
and must be synchronized/hash-verified before any launch. The manifest and
explicit outstanding steps are preserved as input_manifest.json/checkpoint.json.
Transfer timing is labeled successfulsuffix-resume only; cancelled-attempt
duration is unknown, not silently omitted from a claimed end-to-end total.

Resource snapshot2026-11-02T21:25:21Z, model-reportedUSD:
parent4.0810264; allworker-session messageusage1.5537486; combined5.634775.
Workers: source scout.8471714, oldrecovery.3091922, modalimplementation.0958336,
failedresearcher0reported, independentverification.1163422, assetinventory
.0419884, newCUDArecovery.1432208. These are unique session message-cost sums,
not summed resumed lifecycle snapshots. Billed credits and externalsearch/
machine costs remainunknown. No additional gate is passed; goal not complete.
Next dependent action requires SSHreachability restored, then ownedpartial
resume, fullhash, updatedhelper sync and the frozen one-designpilot.

2026-11-03: original-design restoration pilot passed byte-for-byte.
After the user-confirmed network interruption, parent-owned resumable transfers
completed without restarting partial files. The 95,507,253-byte source archive
matched its original hash; its successful suffix resume sent 29,443,893 literal
bytes in 152.57s. Five exact fingerprint dependencies then transferred
134,956,446 bytes in 121.33s. Exact old-mount reuse avoided another187,989,388
WAN bytes. These measured transfers are separate from prior cancelled-attempt
durations, which remain unknown; bandwidth is not assumed constant.

Attempts1–3 preserved real prerequisite failures rather than bypassing them:
module discovery after extraction, upstream fingerprint metadata, and the
missing original D000007 context trajectory. Source bytes were not patched.
The source fingerprint required old study metadata: all77 referenced artifacts
matched after staging;15 additional exact protocol.json files added58,221bytes.
Actual inputs() and its independent fingerprint equality then passed. Existing
trajectory_D000007 data was restored exactly (4,868,858bytes), followed by the
4,912-byte sensor manifest from the old mount. All frozen within-simulator
context/eigen checks passed for the six declared ratios; the solver warning was
retained. These are numerical/admissibility checks, not independent validation.

Attempt4 used durable unit p4-recovery-d000007-attempt4-20261102 with a600s
independent acquisition-subprocess deadline and900s total cap. Output:
support_modal_20261009_v2/shards/D000007.npz, exactly42,258,789bytes and SHA256
cd203cee952cba0b64365780b0da3e47e632664b38fda8628ae1f4a8539921d7,
matching the original acquisition manifest. Acquisition including child imports/
contracts took18.651346815s; helper total23.371109516s; systemd23.850s;
CPU23.053s; peak1.7G. Unit MainPID0 after success, no continuing acquisition.
Progress hash is unchanged. Local attempt4_receipt.json/log are byte-identical
to remote evidence, hashes90b62fb37bdba7b8e01c8b68a9e77fbd980f34d124216ebea3f66af5a2d5acdd
and1127a0e7516201608d4a59093e1d43634252eb704f5dbcb7fa1110f779ef646f.
This proves one original-design source replay, not point accuracy, blind transfer,
independent physics or any endpoint gate.

Configured minion-worker route exhausted its provider usage quota; task-local
model overrides were rejected by the runtime schema. No saved routing,
permissions or account configuration changed. An explicitly disclosed general
task fallback used the available gpt-5.6-luna route. Its first phase delivered
no checkpoint before600s; one recovery returned the completed metadata/readiness
work. Later isolated briefs requested primary evidence within five tool calls.
The fallback initially returned the secondary19-file inventory instead of
attempt4 acceptance; a single correction isolated the check and obtained the
actual numeric proof. Dispatcher contribution: combined gate acceptance with
later planning. Global dispatch guidance now separates those phases. These are
prompt mitigations and real-work observations, not controlled improvement claims.

The20-design roster is protocol-derived. Remaining19 context trajectories all
exist locally (76,453,581bytes), with no matching old-mount copies. An original
design reuses six archived trajectories and generates added regimes; its cost
does not by itself represent the12 full-generation additional designs. Before
bulk recovery, a separately frozen D000152 pilot is being prepared: five
profiles/two replicates, original source-shard target41,270,392bytes and SHA256
ca8594d4b5d13349a9d7e7edfd0218808884eecdd9687f1e593c607c2fdd3fdf.
Its exact3,468,164-byte context source transferred in6.59s. No D000152
acquisition or bulk restoration is claimed yet; all endpoint gates unchanged.

2026-11-03: resumed recovery diagnosis; intermediate replay remains unsuccessful.
The additional-design D000152 acquisition completed, but produced41,270,304bytes,
SHA256 d28379d0fd00d712e817cea7db4b846e4a5cc62f452ae184620064209450f7d2,
not the frozen41,270,392-byte ca8594d4b5d13349a9d7e7edfd0218808884eecdd9687f1e593c607c2fdd3fdf.
The separately controlled original-CLI replay reproduced that same mismatch.
No additional runtime variants or bulk restoration are authorized under that
failed protocol. Evidence: recovery_20261102/additional_attempt_receipt.json,
canonical_attempt_receipt.json and preserved_mismatch_D000152_attempt1.npz.

Corrected metadata comparison v2, SHA256
559a62a85f93558157bea8aa8595f56c6bc5bbf4af6a08e7a627cc99151b931a,
uses the actual NPZ producer's json.dumps(value) with no keyword arguments.
Retained validation and audit objects serialize byte-identically to the current
shard scalars:12,456bytes/c06d455ac75aced9926bbd3044be592dae5cd57464dea142a84b249bd244275a
and10,650bytes/323c2ddbb862918ab9525c23625eb228e8d2ff605014dd632fb72184bf9a4b5b.
The earlier pretty-file comparison used the wrong serialization boundary;
the worker corrected it once the dispatcher isolated the exact calls.
Original waveform/state payload identity remains unknown, not metadata-only.

A new local diagnostic exercised unchanged noise_for/M.normalize on all130
D000152 training rows. All2,080 calibration offsets and2,080 scales were
numerically equal to retained raw_modal_20261010/acquisition.json values:
zero differing values and zero maximum absolute difference, elapsed1.060s.
This is calibration evidence only, not waveform bit identity. The manifest had
2,094rows and its retained row-order hash matched; protected progress SHA256
ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f remained unchanged.
Next hypothesis: differing intermediate NPZs may still reproduce both exact
frozen consumer-array hashes. A separate preparation phase preserves truthful
new provenance and unchanged normalization/masks; no model training is authorized.

Dispatcher process review: initial historical consult preflights used guessed,
incorrect encoded project/timestamp paths. A scoped glob resolved the actual
paths and both consults completed. The semantic search backend also returned
HTTP402; literal search was used instead. The codex-role bootstrap skill was
unavailable; setup did not block research. Saved terra worker routing and prior
luna fallback had reached provider limits; explicit task-local fallback is used
without changing saved routing, permissions or account settings. Global clauses
now require observed archive paths and actual producer serialization arguments.
These are scoped prompt mitigations, not measured reliability improvements.
All research gates remain unpromoted; no change to the progress ledger.

2026-11-04T22:09:20.941630818+00:00 (observed host UTC clock): consumer-restoration orchestration review. PrepareConsumerRecovery's preparation deliverable is rejected: protocol.json required all 20 shards while forbidding generation of the missing 18; its inventory remains useful partial evidence, and one recovery is not a new review/count. The dispatcher bundled inventory, protocol, and controller in one overlarge preparation phase and then wrote unexecuted protocol_v2/controller artifacts; neither launched. The user preference is now dispatcher mostly read/instruct with agents owning ordinary writes; existing policy already required bounded phases and delegated ordinary writing, so no new generic rule or prompt-improvement claim is made.
The 99-MB dataset rsync failed with code 30 after 199.49 s and its exact target/partial was absent on later inventory; the 3,486,411-byte D146 context transfer timed out at 180 s. The consumed-field lossless diagnostic received conditional design approval only, was not executed or accepted, and must retain native source/physics checks, propagation/projection/commands/raw/RNG behavior, truthful carrier-versus-original provenance, byte-exact D7/current-D152 differential checks, and both complete consumer hashes. Do not review live BuildLosslessCarriers or ImplementRawOnlyAcquisition before parent acceptance.
The native signature probe exited 0 with before==during==after under scoped G.O.OUT redirection, restored the path, and observed signature SHA256 d874db664d0b9d06b2e294bd3f375bdb71e9ebf0f56680674d8efb485faf7f88; this is an invariant observation, not physical or gate validation. D7's intermediate is exact; D152's differs from original but canonical replay reproduces the same current intermediate. Numerical calibration equality covers all 130 D152 training rows (2,080 offsets and 2,080 scales) only; it does not establish intermediate identity, waveform/consumer identity, or physical validation. Complete consumer identity and D7/D152 differential remain unverified; P4_PROGRESS SHA256 ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f is unchanged and 0/10 research gates qualify.
The apparent 19-context-size discrepancy was a denominator mismatch: all 20 contexts are 81,322,439 bytes; excluding D7 is 76,453,581 bytes; excluding D152 is 77,854,275 bytes. No five-assignment retrospective is appended because the newly reviewed count is unknown/not yet due after deduplication and exclusion of unreviewed live work; no count is guessed.

2026-11-04T22:40:11.623141+00:00 (review recording; observed host UTC clock): five distinct parent-reviewed delegations are recorded in the review ledger. BuildLosslessCarriers is rejected: no artifacts, two local runtime errors and last package count 2; the exact cause and provider/timeout attribution are unknown. ImplementRawOnlyAcquisition is accepted-unverified: parent source review covered 109 lines and an import smoke was reported, but GPU differential remains pending. MinionCarrierRecovery is accepted on the parent's independent 22 NPY member and 20 audit byte equalities, 42 tar files, and native manifest 2,094 rows/124 times with record-order SHA256 c52f15e434d44013ebdb759a847c3ab7d4c65463d25f7e258fecd67bf670b260; bundle SHA256 7864949dd5da62e1d0ebe445100fdb42904aaf1f1bbae37a45c468000fb96618. MinionNativeContractCheck is accepted for bounded source findings, not remote readiness or a full fingerprint-chain audit. RecordRecoveryReview is accepted on the parent's reading of journal lines 7094-7097 and ledger lines 108-109 and reported parse of 109 rows. Resumes and corrections are not new delegations.
Dispatcher mistake: continuation was treated as authorization to use generic task/Luna instead of the configured minion; the user now explicitly forbids that substitution without fresh authorization. The ordinary-delegation unavailability clause now requires fresh explicit user authorization to switch to generic task or another model, otherwise report the blocker. This is a prompt mitigation, not a measured improvement; a later real-work child spawned after the edit must establish policy loading. The configured minion terminal telemetry identifies openai-codex/gpt-6.1-terra; usage snapshots are recorded individually, not summed across resumes, and billed credits are unknown. Briefs and phase scopes differed, so minion artifact success and generic failure are not a controlled model-performance comparison. No alternate-model experiment was authorized or run.
The ledger retrospective covers 11 distinct reviews since the prior covered entry 102, including these five new reviews; current workers are not reviewed prematurely. No scientific gate advanced: D7/current D152 raw, command and native-normalized differentials and both complete consumer hashes remain mandatory before their respective scaling/identity claims. No original whole-shard identity, model, or gate claim follows from the carrier evidence.

2026-11-04T23:12:12.775003+00:00 (observed host UTC, evidence-preservation task): the parent-launched p4-raw-only-differential-v1.service ended failed, exit 1, runtime 43.784 s, CPU 43.650 s, peak 1.1 G, parent SSH 46.64 s. Its receipt reports error at acquire_D000152 after helper elapsed 39.614476222 s: RuntimeError, D000152 raw byte mismatch. D7 raw <f8 [10,16,8001] matched SHA256 505491909622b839bc14ddd2b738c8d62f5f00faacbd257bda8fcc66ac02c4ca; D7 command |b1 [10,8001] matched SHA256 21cad463dcf3b4e8ad346c2477bad5e39b891381e75a2c72b8dea733ae71e67e. D7 candidate source was 10,046,941 bytes, SHA256 8833223a0a138048f1b4dbd4fb518fd74304e37f87afea90772ac3b766081f91. D152 candidate raw SHA256 6d7c0108c91ab835d10f9bbe31781c3bfec93269e6ef7858a82a8fdfbe00df6b differs from reference a05a209ffe6d1d4c5df92cb8fc83cf336a45de6845b888ca17564d738fd516fd, with the same dtype and shape. D152 command was not compared, normalization did not run, and the full consumer remains unverified; no gate credit or tolerance relaxation.
The failed D152 candidate was lost because the comparison raised before persistence: the dispatcher brief itself specified comparison/stop before persistence, and the implementation followed that ordering. Source implementation was delivered by MinionDifferentialRunner, but runtime acceptance failed; its ledger amendment is a phase correction, not a new delegation. A single bounded SSH read copied receipt, checkpoint, unit log and the three executed control files byte-identically into consumer_recovery_20261103/raw_differential_v1_evidence; local SHA256 and byte counts matched remote metadata. No arrays were transferred. The observed terminal unit properties were ActiveState=failed, SubState=failed, Result=exit-code, ExecMainStatus=1. Frozen progress SHA256 ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f is the previously reported value, not newly observed in this copy.
The controlled V2 contrast is not executed or promoted: fresh subprocess for D152 only, reuse pinned D7 candidate, persist candidates before strict checks, retain original criteria. Process history remains an unproven hypothesis; TP.initial uses a design-seeded local RNG and G.sensor uses ROOT/p4_sensor_candidates manifest, not context OUT. The global scientific-research instruction now requires candidate and provenance persistence before pass/fail gates; this is a mitigation, not measured improvement, and loading in a later fresh real task remains pending. Current global protocol requires fresh explicit user authorization before generic or other-model fallback if the configured minion is unavailable; no synthetic test or fallback was performed.

2026-11-04T23:44:11.961754+00:00 (observed host UTC; V2 evidence preservation): the parent accepted the actual p4-raw-only-differential-v2 runtime for the D7/D152 subset. Unit runtime 38.864 s, CPU 37.714 s, peak 1.5 G; helper 34.937130806 s and D152 child 27.294819357 s. Parent parsed the receipt and independently rehashed both actual paired normalized files. D7 and D152 raw and command dtype, shape and payload matched exactly with zero bitwise differing values, and ordered keys agreed. All 180 training rows (D7 50, D152 130) yielded native-normalized waveform files of 92,171,648 bytes with both SHA256 10c47b209f2dc54aac08254a2552872a8249d5358c9bd6a527b5711802b738c1, and admissible files of 1,440,308 bytes with both SHA256 62b0414e9d93fb9e82558332e05b37f9c0c2cdaedb95123fc4ce3ab492d4c223. Parent observed protected progress unchanged at SHA256 ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f. V2 D152 candidate was 10,035,169 bytes, SHA256 75262bc97461e65d28d1e1308ab302d89fd284cc15417582fb906e9bfcac9c03.
The contrast supports process isolation in this case; the mechanism and magnitude of V1's raw difference remain unknown. It is not evidence of a complete 2,094-row consumer identity or scientific gate credit. The parent has separately authorized full reconstruction implementation with fresh per-design processes; no full job has yet executed. MinionDifferentialRunner's V2 runtime acceptance is a subset-only phase amendment, not a new delegation. The receipt, both checkpoints, unit log, executed V2 script and protocol, and distinct reference/candidate acquisition JSON metadata were copied without reserialization into consumer_recovery_20261103/raw_differential_v2_evidence; remote and local SHA256 and byte counts matched. No NPY or raw arrays were transferred.

2026-11-04T23:46:43.087539+00:00 (observed host UTC; instruction review): MinionFullConsumer returned ESCALATE after 84.2 s without files when the frozen callback-orchestration assignment encountered the worker body clause requiring escalation for scientific or performance-sensitive implementation. This was an expected boundary escalation under the literal body wording, not a worker error: the dispatcher-facing description already restricted escalation to scientific or performance-sensitive judgment, while the body was broader. The dispatcher contribution was assigning the frozen implementation without resolving that ambiguity. The active minion-worker body now changes only that clause to require escalation for implementation requiring scientific or performance-sensitive judgment; architectural decisions, unresolved research judgment and new strategy remain escalated. Choosing or changing methods, assumptions, parameters or criteria remains dispatcher-owned. This is a scoped prompt mitigation, not a measured improvement; a fresh real implementation spawn must establish loading. Model routing, permissions and approval gates are unchanged.

2026-11-05T00:06:41.918312+00:00 (observed host UTC; launch record): parent reports fresh MinionFullConsumerFixed implemented the frozen algorithm under the clarified role; runner SHA256 c13906eb30b06264dd3f5f06fec576b4084ce2f347d2dc8eca6807643fd5526b and protocol SHA256 f74bc02315e8407b7148a9785fa7312fe27ce6509635e765ff19dd6c8f8b9440 were staged and verified remotely. Local preflight had the expected two missing references; corrected remote preflight passed without errors on Torch 2.12.1+cu130, CUDA GTX 1660 SUPER. Parent observed p4-full-consumer-v1.service loaded/active/running, PID 46534; internal timeout 900 s, external 960 s. The 20 s eval launch ACK timed out, so parent recovered ownership once before any retry, confirmed active and did not duplicate launch. Checkpoint and launch details are in consumer_recovery_20261103/full_consumer_launch_record.json; final 2,094-row hashes remain unverified. The hypothesis is exact native consumer reconstruction using 18 fresh children and two V2 reuses; 18 times the measured 27.294819357 s child is approximately 491.31 s plus 60 s margin, with a 900 s stop.
Parent accepted the dataset first-1 MiB smoke only: SHA256 47c67d989333845d1af4113f8b03e2967eed6b9aae9769a215bbef337cd9de38, chunk 15.440410549 s, total 18.781303921 s, status 2.987499799 s; sender final SHA256 6903d1017981356b5f2695f7d703b638c250dfee84ec3a73cab0bc5db9c854a2. Receipt prep/dataset_transfer_receipt_3f4a316476d24c5cad2a527583e87221.json. Parent owns bg_9, one complete attempt bounded at 1200 s with 1300 s external deadline; projected remaining ~24 min exceeds one attempt. At most two bounded attempts are authorized, with prefix verification between them; final 99,114,670-byte dataset SHA256 170f5e783e2fbab6abefce7c4b0a18b54af4d36b345e546ab7c01dcb23154499 remains mandatory and unverified.
Dispatcher failure: first remote preflight failed with No module named numpy because the parent reconstructed the old mount overlay using /home/ rather than /@home/ and omitted ROOT. The exact working overlay was recovered from retained v2_preflight_argv and the corrected preflight passed. This is not a minion error. Global instruction now requires recovery of runtime paths and environment overlays from verified launch records rather than condensed history; it is a mitigation, not a measured improvement, and fresh real-workload loading remains pending. The prior worker-role clarification did load in a fresh real mechanical implementation, but that observation is not a controlled reliability comparison. Reviewed outcomes are limited to FullConsumerFixed implementation and preflights, CarrierRecovery first-chunk smoke, and NTRSPlateEvidence first bounded primary lookup; no bulk commit, complete benchmark or qualification is implied. Retrospective count is not provably five new distinct reviewed delegations; defer rather than double-count amendments or resumed snapshots.

2026-11-05T06:54:09.574747+00:00 (parent verification): cached dataset restoration ACCEPTED: 99,114,670 bytes SHA256 170f5e783e2fbab6abefce7c4b0a18b54af4d36b345e546ab7c01dcb23154499. Complete transfer attempt took 1141.770678 seconds and committed within its first 1200-second attempt; no second attempt was needed.
Full consumer experiment completed all20 raw sources and2094 rows in483.114732 seconds, but FAILED its unchanged exact-waveform criterion. Parent independently hashed actual remote files: waveforms1072262144 bytes SHA256 6faa37e5c8a59fdf5bed0afe44135ab61658eabbc8e7c7f5f972a36c4666799d versus required5e90e4652ead98d06067865d5cec710b75ac86a8d322c37b1604e302884acf90; admissible16754222 bytes matches788df27bfc37d292884d72dda633accf67435841fe4c72a77b250a5acf1e8b4b. Preserve full_consumer_v1; do not retag or fit old frozen protocol. Local recovery_acceptance_20261105.json records primary hashes. No tolerance or gate changed.
Causal-twin diagnostic: configured minion hit provider usage_limit_reached, leaving complete script/protocol. Parent reviewed and ran causal_twins_20261105/check.py: all exact rational, binary64-prefix, first-crossing, derivative, rate and conditional-bound checks passed; results SHA256 c02e3d3cbe5f18d30897737719c7e1530ab3e2e8022f69f6e966d54f3c0d56c7. For ambiguous crossing fraction alpha, q_min=max(0,1-.1/alpha); independently rearmed benign opportunity rate lambda_b implies nuisance>=lambda_b*q_min. This requires physical feasibility, shared-prefix distributions, clear/rearm and event-credit assumptions; hypothetical grid is not a measured operational distribution or unconditional impossibility proof.
Independent source lookup: NASA TN D-5508 tableIII thermally stressed aluminum case is not a qualified honeycomb-CFCC source. Parent official-text inspection confirms q39psi/Pt159psia transcription, but ideal-air isentropic Mach3 predicts q/Pt=.171509 versus reported.245283. This unresolved source/OCR/convention discrepancy is not established as a minion transcription error. PDF fetch incomplete; official metadata and text derivative worked. primary_ntrs_19690030702_evidence.json preserves limits; no further bulk download authorized.
Scout correction: raw_modal_data.E and S=experiment_support_modal_v2 are verified identical by the executed full-run preflight, so changing S.SHARDS changes E.SHARDS. The handoff incorrectly implied distinct settings from alias spelling; its other R.OUT and metadata constraints remain. Dispatcher brief omitted explicit identity. Small generic alias-resolution instruction added to minion-scout; routing/permissions unchanged. Fresh real-work loading pending; mitigation, not measured improvement.
Next research direction is a separately frozen raw-versus-mask controlled development probe using only the new waveform lineage, not stale modal-feature values. No fit is launched yet; avoid mixing response realizations. Existing raw384-sample and measured-modal128-sample lags4/2/0 span384ms; canonical upstream modal provenance remains separate. Old TCN stopping limits2000epochs/patience100 require measured real-unit budgeting before any campaign.
Usage snapshot: parent session reportedUSD18.1826416 including archived work; selected six minion sessions separately recorded in usage_snapshot_20261105.json, with resumed lifecycle snapshots not summed. Billed credits, external search charges and machine cost unknown. Progress SHA256 ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f unchanged; zero qualified gates, goal incomplete.
