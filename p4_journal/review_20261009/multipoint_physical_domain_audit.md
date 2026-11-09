# Controlled multi-point acquisition — physical-domain checkpoint

**Decision:** numerical simulator-only feasibility can be checked; the present source does not independently validate32ms freestream transitions or establish actuator feasibility. No full fit is authorized. This is not a deployment envelope or a physical-validation pass.

## Established source facts

- The pressure model is first-order, instantaneous in supplied freestream velocity and plate slope/velocity. Its coefficient contains `rho*U²/sqrt(M²−1)` and a plate-velocity damping term, not a freestream-acceleration term. The configured pressure-model guard is Mach≥2; aerodynamic assembly itself checks only Mach>1. This new acquisition applies the stricter Mach≥2 guard. [pressure model](../../src/mechanics/piston_theory.py), [assembly](../../src/mechanics/solver.py).
- `PlateDynamics` projects these stiffness/damping terms into a velocity-independent structural basis. Substituting time-varying velocity changes these quasi-steady coefficients; it does not introduce flow-memory dynamics. The existing review explicitly describes this as quasi-steady piston-theory simulation, not full unsteady-aerodynamic validation. [physical dynamics](../../p4_groups/operational_confirmation_20261003/trajectory_physics.py), [research review](../../src/mechanics/p4_margin_estimation/RESEARCH_REVIEW.md).
- Existing acquisition audit checks full/reduced response RMS≤.02, step-halving RMS≤.005, critical relative pole gap≤.005, retained-eight frequency<400Hz and minimum Mach≥2. Its original velocity cases are .9/1/1.03 times each design's flutter speed and its ramp is .97→1.03 over120ms. Those checks are not validation of the new absolute900–2240m/s endpoint grid or32ms transitions. [audit implementation](../../p4_groups/operational_confirmation_20261003/trajectory_physics.py).
- Existing initial physical displacement/velocity is formed from eight eigensystem modes at .9 times each design's known simulator flutter speed and normalized to1µm sensor RMS. We preserve this initial-condition law, including its truth-dependent simulator preparation, and pair exactly the same initial mechanical state across S/M/R. It is **not** evidence of independent physical excitation realism or a waveform-only experimental preparation. No separate state reset occurs between plateaus. [initial state](../../p4_groups/operational_confirmation_20261003/trajectory_physics.py).
- The physical pulse helper starts at1s. The later temporal-probe pulse law starts at≥.6s. Neither produces a pulse within the new0–447ms episode. These checks are initial-response experiments; they do not establish usefulness in long, weakly excited monitoring gaps. [physical pulse](../../p4_groups/operational_confirmation_20261003/trajectory_physics.py), [temporal pulse schedule](../../p4_groups/operational_confirmation_20261003/temporal_probe.py).
- Causal calibration uses measured samples0:51, inside the first plateau. Each arm keeps its own measured offset/scale and cap50. For the contrast, actual physical noise arrays are paired across arms using S's calibration-RMS reference; this preserves S's existing sensitivity convention but is deliberately not separate arm-relative noise rescaling. Thus physical operating-point changes may change effective M/R relative SNR. [normalization and noise convention](../../p4_groups/operational_confirmation_20261003/diagnostic_matrix.py), [frozen new protocol](../../p4_groups/operational_confirmation_20261003/multipoint_preflight_20261009/protocol.json).

## Concrete proposal and limits

The four original fitting designs have flutter speeds1198.7415–2071.1506m/s and sound speeds335.8931–344.8925m/s. Training-only rules freeze68 endpoints900–2240m/s in20m/s increments and delta20m/s. S/M/R apply the same commands to every design; true margins are computed afterward. All plateau speeds are880–2260m/s, so minimum Mach over these four designs is approximately2.552. No held/checkpoint-validation/reserved design supplied endpoint information.

Cubic40m/s change over32ms has peak1875m/s²; the20m/s return has peak937.5m/s². These are proposed simulator rates, **not validated permitted rates**. The original sampled structural length/speed gives a convective scale approximately0.36–1.18ms across the proposed speed range. Comparing that scale with32ms is at most a plausibility screen; it does not prove negligible unsteady-flow error, small-disturbance validity, actuator feasibility or acceptable physical model discrepancy.

**Missing prerequisites:** evidence for the time-varying freestream model at these rates, an actuator/acquisition interpretation that can realize the commands, and independent physical/model-form validation. Large-supercritical cases created by a common absolute grid must remain in accounting; saturation or invalid physical behavior cannot be silently removed to improve scores. Numerical convergence can eliminate integration error as an explanation only within the adopted simulator.

The two requested background workers returned provider429 failures without a usable audit delivery. Dispatcher completed this source audit directly; no external literature claim is made or independently verified physical rate inferred.

## Returned preflight decision and independent-treatment prerequisite

Frontier stopped the full fitting campaign after the proper-fitting acquisition
and accounting checkpoint. Exact decisions:
[initial assessment](multipoint_preflight_frontier_decision.json) and
[physical-domain amendment](multipoint_frontier_amplitude_amendment.json).
The representative extreme-grid trajectories produce displacements from39.3m
to1.39e20m at2240m/s; largest displacement/panel-length ratio1.525e20.
They cannot be interpreted as small-disturbance physical examples. Preserve them
as acquisition/model-domain failures, with all accounting intact. The.080928365
collision floor remains an exact bound for the synthetic representation, not a
physical accuracy limitation; no all-missing histories are inside the prior
near-flutter margin population. [raw amplitude evidence](../../p4_groups/operational_confirmation_20261003/multipoint_preflight_20261009/amplitude_compatibility.json),
[independent accounting receipt](../../p4_groups/operational_confirmation_20261003/multipoint_preflight_20261009/verification.json).

Required checkpoint is D000033/D000068 at1240m/s, with the unchanged S/M/R
trajectories, initial states, calibration and aperture. First assess displacement,
rotation/slope and applicable structural/aerodynamic small-disturbance conditions
over the plate, not just the sensors. Then supply an independently justified
accelerated-flow or flow-memory evaluator, its rate/timescale applicability and
numerical uncertainty. Its discrepancy/uncertainty must be smaller than the
S-versus-M/R response contrast in each order. Passing authorizes prospective
acquisition redesign only, not full predictor fits or endpoint claims.

No matching independent implementation was found in the inspected mechanics
source/root experiments; the recorded local command census found no `SU2_CFD`,
`rhoPimpleFoam`, `rhoCentralFoam` or `simpleFoam`. That is local unavailability,
not a claim that no external solver exists. No applicable analytical error bound
has been established. [prerequisite record](multipoint_independent_aero_prerequisite.json).

Primary-source metadata verification corrected misleading search summaries:
[NASA19710015225](https://ntrs.nasa.gov/api/citations/19710015225) is the collected
*Performance and Dynamics of Aerospace Vehicles*; no acceleration bound was verified.
[NASA19720010339](https://ntrs.nasa.gov/api/citations/19720010339) is Bennett's
*Application of a one-strip integral method to the unsteady supersonic aerodynamics
of an inclined flat surface*. Its abstract concerns small perturbations about
mean steady supersonic flow and explicitly describes limited accuracy where
forces vary rapidly with frequency. It does not supply a validated bound for this
accelerating-freestream acquisition. Full PDF retrieval/scan extraction was
incomplete; search-summary formulas were not adopted. No rate limit invented.

## Completed1240m/s plate-field geometry screen

Executed24 cases: two proper-fitting designs × S/M/R × initial-condition
replicate0/1 × reduced/full structural systems. Unchanged initial states,
absolute velocity commands and continuous dynamics. Each case completed all1789
quarter-millisecond integration samples without termination. Physical generalized
coefficients were reconstructed from the energy-scaled modal state. The original
sensor projection agreed with reconstructed displacement at rtol1e-11/atol1e-14.

Displacement, analytic physical-coordinate slope and FSDT director rotations
were assessed over the plate, not inferred from sensor amplitudes. Triangle
bounds use |P_n|≤1 and |dP_n/dx|≤n(n+1)/L (or corresponding trigonometric bounds).
They cover the continuous spatial domain at each integration sample. A65x65
spatial screen at times maximizing those bounds remained within them; it is
not a continuous-time-extrema or nonlinear-model error certificate.

Worst conservative bounds over arms, replicates and structural systems:

| Design | Displacement (µm) | Displacement/thickness | Slope norm | Director-rotation norm (rad) | Normal-velocity/sound-speed |
|---|---:|---:|---:|---:|---:|
| D000033 | 18.5561 | 0.00226651 | 0.000117177 | 0.0000919246 | 0.000346902 |
| D000068 | 15.6388 | 0.00184794 | 0.000173075 | 0.000116619 | 0.000488984 |

The normal-velocity bound uses |w_t|+U|w_x|; it does not bound accelerating-flow
or downstream-memory error. The leading omitted geometric-strain screen
.5*||grad(w)||² is bounded by6.86523e-9 and1.49774e-8, respectively; no calibrated
nonlinear structural/aerodynamic error limit is inferred.

[Complete24-case receipt and executable diagnostic source](../../p4_groups/operational_confirmation_20261003/multipoint_preflight_20261009/checkpoint_geometry.json).
The initial all-time dense-grid diagnostic timed out; its owned orphan process
was terminated and no result credited. The completed diagnostic retained all
cases and all integration-time analytic bounds; only the dense spatial
cross-check was restricted to bound-peak times.

[Frontier's returned decision](multipoint_geometry_frontier_decision.json):
the obvious large-deflection/slope/rotation objection is screened out for these
checkpoint cases at sampled integration times. Do not repeat this same-equation
screen. Extreme-grid cases remain invalid as small-disturbance physical examples.
Full acquisition and predictor fitting remain stopped. The missing independent
accelerated-freestream treatment/error bound must address32ms transitions and
downstream memory, with contrast uncertainty below each S-versus-M/R effect;
the weak D000068 contrast is approximately5.6%. No invented rate limit,
substitute converter experiment or accuracy/availability qualification.
