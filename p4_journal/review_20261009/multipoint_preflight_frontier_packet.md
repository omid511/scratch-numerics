# Frontier-led controlled multi-point acquisition — pretraining checkpoint

## Mandate and decision needed
User asks you to own P4 scientific prioritization and continuation decisions; dispatcher implements/runs/verifies, user monitors. The previously reported384ms ramp-history experiment did not manipulate operating-point diversity. Your selected S/M/R acquisition was authorized for feasibility/accounting only, not full fits. Gate-retention and physical inversion were deferred.

Decide whether this concrete preflight supports any fitting campaign, requires physical/acquisition correction, or should stop. In particular: do NOT turn the expanded-grid overall floor into evidence against the original near-flutter task. It occurs entirely outside the previous acquisition margin population. Do not ask the user to choose routine next experiments. Give one explicit continuation/stop decision with exact prerequisites and scope. No unearned endpoint credit.

## Prior outcome context
The last completed campaign ran12 cached fitting/context fits (all worsened primary means),24 support/partial-physical-description fits (measured expansion modestly helped, safety profiles worsened; only38.402% replacement overall, especially incomplete under noise), and36 bounded-GRU/control fits (measured history helped changing regimes but no uniform recurrence-specific win; risk guardrails failed). No promotion. The cached frozen-gap floor made its endpoint impossible and should have been calculated before training. Current evidence indicates observation erasure/extrapolation and converter/context interactions, not proven absence of raw information. The GRU route remains live. Reserved designs and progress ledger are protected.

## Exact acquisition actually executed
Only original proper-fitting designs D000007,D000033,D000068,D000087. No acquisition on checkpoint-validation, original held,12 additional TRAIN evaluation-fold or reserved designs. Two original replicate initial-condition seeds.448 sensor samples at1kHz; observed windows0:128,160:288,320:448. Continuous physical state, five separately aligned integration pieces; smooth cubic transitions128–160ms and288–320ms consume aperture. No predictor memory before0. First-to-last447ms.

S: Uend throughout. M: Uend−20,Uend+20,Uend. R: reverse initial pair with identical endpoint. Proper-training-only rules freeze68 absolute endpoints900–2240m/s in20m/s increments, delta20m/s. Lower grid=20*ceil(max(.75*min(training_ucrit),2*max(training_cair)+delta)/20); upper=20*ceil(1.08*max(training_ucrit)/20); delta=5*floor(.02*min(training_ucrit)/5). Same complete grid, amplitudes and timing applied to all designs; labels computed afterward only. True training flutter speed range1198.7415–2071.1506m/s explains the broad common grid. Endpoints are NOT evaluated-design-dependent.

Preserved16 sensor coordinates, measured leading51-sample causal channel calibration and cap50. Exact same original physical initial displacement/velocity per design/replicate across arms; no extra pulse. Both existing physical/temporal pulse laws start after this0–447ms episode. Initial conditions still use source8-mode eigensystem preparation at.9*ucrit, normalized to1micrometre sensor RMS; this simulator-truth-dependent preparation is disclosed and is not independently realistic excitation evidence. This tests early initial responses, not long-gap warning.

Noise: clean/iid1/correlated1 primary, iid5/correlated5 stress. Actual physical noise arrays are paired unchanged across S/M/R using S's leading-calibration RMS; no arm-relative noise rescaling. Each arm retains its own causal measured calibration. This preserves S's old convention but effective M/R relative SNR may differ. Forecast gate is unchanged: finite/cap-free plateau, prefix64 estimate predicts observed next64 with skill≥.5, nonempty full128 estimate and finite174-coordinate row. Rejected slots zeroed with original flags/age; no SNR threshold. CPU transcription used for accounting; original CUDA gate replay on GTX1660SUPER passed all1440 fixture windows with463 accepted on each backend and zero classification disagreements. This is a parity sample, not full-dataset feature/inference replay.

## Numerical/physical evidence
{
  "numerical_arm_cases": 36,
  "all_numerical_pass": true,
  "all_rom_pass": true,
  "maximum_default_vs_fine_rms": 9.223976689449643e-06,
  "maximum_rom_rms": 4.479254297422776e-06,
  "maximum_adaptive_ode_response_rms": 6.708191547634061e-07,
  "maximum_adaptive_ode_final_state_relative_error": 0.000572428171552265,
  "maximum_boundary_gap": 0.0
}

These36 arm cases are four designs×three representative endpoints900/1240/2240×S/M/R. Default1ms coefficients compared with.25ms reference; entire6-step refinement10/4/2/1/.5/.25ms retained. Full/reduced comparison uses same.25ms step. Independent DOP853 at rtol1e-9/atol1e-12/maxstep.5ms replays the reduced system. No piece spans transition boundaries, no state reset, all boundary projection gaps0. Numerical tolerance.005 and ROM tolerance.02 remain unchanged. This is convergence/ROM agreement for the same physical equations, NOT independent physical/model-form validation, nor an all-grid ROM/bandwidth audit.

All plateau speeds880–2260m/s; minimum Mach on these four designs approximately2.552. Stricter configured Mach≥2 enforced. Cubic peak rates1875 and937.5m/s² are proposed simulator rates, not validated actuator or unsteady-flow bounds. Source has instantaneous piston-theory coefficients, not freestream acceleration/flow-memory dynamics. Independent validity of these32ms transitions remains unresolved; no rate limit invented. Convective timescale check is only plausibility, not proof.

Measured raw waveform contrasts and all-state pole changes at1240m/s:
[
  {
    "design": "D000007",
    "endpoint_mps": 1240.0,
    "maximum_state_pole_growth_per_s": [
      5.5303632284755935,
      11.32005286765029,
      16.299766803976794
    ],
    "waveform_M_vs_S_rms": 0.2377686024693998,
    "waveform_R_vs_S_rms": 0.2558082631621428
  },
  {
    "design": "D000033",
    "endpoint_mps": 1240.0,
    "maximum_state_pole_growth_per_s": [
      -15.737812822536853,
      -11.574710534398118,
      -2.9130816582310928
    ],
    "waveform_M_vs_S_rms": 0.32080927972420104,
    "waveform_R_vs_S_rms": 0.5595093651025037
  },
  {
    "design": "D000068",
    "endpoint_mps": 1240.0,
    "maximum_state_pole_growth_per_s": [
      -23.133296554584717,
      -23.166548798999834,
      -23.19706822752913
    ],
    "waveform_M_vs_S_rms": 0.056370528589034324,
    "waveform_R_vs_S_rms": 0.055957869605879534
  },
  {
    "design": "D000087",
    "endpoint_mps": 1240.0,
    "maximum_state_pole_growth_per_s": [
      7.625271975259466,
      13.55918218722099,
      18.775168519336887
    ],
    "waveform_M_vs_S_rms": 0.06853174008181957,
    "waveform_R_vs_S_rms": 0.18001463548040347
  }
]
All-state maximum growth is explicitly NOT a tracked true critical-branch identity. Nonzero waveform contrast is not estimator benefit.

## Complete accounting and exact input-collision bound
1,632 physical episodes×five noise views=8,160 histories. All originals retained. Proper-training accounting took315.666s CPU; numerical36-case verification took861.83s. Rough linear acquisition/extraction projection for20 designs≈26.3min on the same CPU, excluding numerical audit/fitting; not a GPU or end-to-end latency benchmark.

| Noise | Arm | Histories | Accepted slots0/1/2 | All missing | Overall missing-group floor | Boundary histories | Boundary floor |
|---|---|---|---|---|---|---|---|
| clean | S | 544 | [292, 240, 230] | 252 | 0.080362 | 40 | 0.0 |
| clean | M | 544 | [298, 240, 230] | 246 | 0.076618 | 40 | 0.0 |
| clean | R | 544 | [284, 240, 232] | 260 | 0.085681 | 40 | 0.0 |
| iid1 | S | 544 | [292, 33, 18] | 252 | 0.080362 | 40 | 0.0 |
| iid1 | M | 544 | [298, 30, 19] | 246 | 0.076618 | 40 | 0.0 |
| iid1 | R | 544 | [284, 34, 20] | 260 | 0.085681 | 40 | 0.0 |
| correlated1 | S | 544 | [292, 32, 17] | 252 | 0.080362 | 40 | 0.0 |
| correlated1 | M | 544 | [298, 29, 17] | 246 | 0.076618 | 40 | 0.0 |
| correlated1 | R | 544 | [284, 33, 18] | 260 | 0.085681 | 40 | 0.0 |

Shared all-arm predictor:758/1632 all-missing histories in EACH primary family, exact equal-design overall lower bound.08092836517617448. Independent enumeration over all candidate missing outputs reproduced weighted-median accounting within1e-12. All zero174-feature slots, flagsfalse and ages[.127,.287,.447] are bit-identical; every other output was optimistically assumed perfect.

CRITICAL SCOPE: every primary all-missing target lies in margin[−.8686264,−.08152446]. NONE lies in original acquisition margin[−.03,+.22], exact boundary(−.02,.05), either(.01–.05) sign-side slice, or farther-stable m≥.15. These misses are first-window cap rejections in severely supercritical cases produced by our broad common grid. They cannot establish an accuracy impossibility inside the original population. They do make all-frame accuracy targets impossible for this expanded-grid/zeroing method as literally scored. No posthoc rows were removed. Do not retrospectively scope them away; any new envelope decision must be prospective and physically justified.

Clean boundary:40 histories/arm, all40 have all three accepted windows. IID1 boundary accepted counts S[40,23,18],M[40,20,19],R[40,24,18]; all-three18/19/18 respectively. Correlated1 boundary S[40,22,17],M[40,19,17],R[40,23,16]; all-three16/16/15. Thus clean acquired multi-point content is present; noisy boundary observations often lose later slots without an exact all-missing bound. Primary shared-all-arm exact boundary120 histories; each eligible sign slice66; farther-stable≥.15 has396. No all-missing rows in these slices. Zero floor is not proof of accuracy or identifiability.

## Frozen acceptance / remaining limits
No full fit run or authorized. If authorized later: original expanded TRAIN folds, grouped acquisition/noise copies, original two checkpoint-validation designs, one shared balanced S/M/R reset three-slot GRU32→16, measured174 content+flags/age, no arm/speed/future/truth inputs, seeds42/43/44, original optimizer/regularization/checkpoint rule. Retain all denominators and primary/stress view pairing. Overall MAE≤.010 clean/≤.015 each1%; boundary≤.005/.010; boundary optimism≤5%; each eligible sign side≤10%; farther-stable MAE noninferiority≤.005. A single-order gain is path/state effect, not general diversity advantage.

Qualified availability, calibrated95% run risk, warning lead/nuisance benefit, full-path latency/replay, blind transfer and independent physical validation remain unearned. No held/reserved evaluation, prediction replacement or ledger change. Ledger checksum independently equals the previous completed-campaign checksum.

Evidence: p4_groups/operational_confirmation_20261003/multipoint_preflight_20261009/{protocol,accounting,numerics,numerical_summary,contrast,verification}.json and dataset.npz; numerical/driver/verifier scripts alongside it. Physical source audit: p4_journal/review_20261009/multipoint_physical_domain_audit.md.

Return critical scientific assessment, explicit authorize/stop decision and one falsifiable next checkpoint. Account for the broad-grid population confound and truth-dependent initialization; do not automatically propose another marginal model/feature/loss exercise.
