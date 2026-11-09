# Experiment 2 support-contrast evidence — original4 vs expanded (original4+additional8) × measured vs matched-physical-description maps

Role: researcher. Report only; no source/artifact edits, no model/loss/scope recommendations.

## 1. Scope, paths, provenance

- Primary: additional12 TRAIN designs in 3 grouped evaluation folds of 4, scored identically by original4 fits vs expanded (original4+additional8-minus-evaluated-fold) fits, with unchanged original2 checkpoint-validation designs. Secondary: old held designs D000053/D000126.
- Folds: fold0 = D000062/D000152/D000008/D000146; fold1 = D000144/D000029/D000020/D000120; fold2 = D000129/D000114/D000158/D000115.
- Original TRAIN: D000007/D000033/D000068/D000087. Validation: D000042/D000045. Held: D000053/D000126.
- Read-only sources (all under `p4_groups/operational_confirmation_20261003/support_modal_20261009_v2/`):
  `results.json`, `protocol.json`, `verification.json`, `observation_coverage.json`, `scoring_correction.json`,
  `correspondence_accounting.json`, `diagnostic_recovery.json`, `correspondence_verification.json`,
  `acquisition.json`, `acquisition_verification.json`, `training.json`. Campaign targets: `p4_journal/review_20261009/campaign_contract.md`.
- Hashes: protocol `786f301f…4a4129`; dataset `170f5e78…3154499`; predictions `86b817b9…fa12815`;
  results `13ba3408…f21dda1`. Actual corrected scorer `711a1a2d…8592e7d3` (= `results.json:scoring_source_sha256`).
  `verification.json:score_source_sha256` (`5d2aea3d…e5103ab`) labels the archived **original** scorer, not the executed
  corrected scorer — see `scoring_correction.json` (original scorer raised `KeyError per_design`; fix indexes each
  selected stratum before the four paired contrasts; no change to predictions/selection/thresholds/formulas/tolerances).
  Corrected verifier `e4e08618…6d121b`; training source `c66681cf…670403`.
- Verification (numerical/development only; grouped TRAIN folds + old held are **not** blind qualification):
  24 fit artifacts; contribution metadata verified; 6231744 independent saved-network frames, max prediction error
  4.663876663757449e-07; 7176 metric rows; 546 factorial-contrast rows; 78 conditions; seeds 42/43/44;
  fold exclusions + validation/held-excluded-from-proper + hashes/denominators verified.
- Model/selection (shared, no search): fixed 32→16 ReLU MLP (6209 params, input width 176); Adam lr=.003 wd=.0001;
  max 2000 epochs/patience 100; full-batch row-mean squared loss; proper-train-only scaling; identical head
  initialization and row order within paired arms; checkpoint validation on original clean/IID1 training views only —
  fresh views never select. Boundary definition exact: `-0.02<m<0.05`.
- Content counterfactual (partial, **not** a full-oracle or information upper bound): oracle replaces only confidently
  uniquely matched measured modal slots' growth/frequency + sensor complex patterns, recomputes dependent
  modal/physics/spatial features, preserves measured strength fractions, residual/quality/availability, envelope
  statistics; no replacement by true critical branch. Full accounting is `correspondence_accounting.json`:
  236712 modal slots; 90902 replaced (38.40193991010173%); 145810 unreplaced, all retaining measured entries —
  reasons: no_frequency_correspondence 100538, spatial_correspondence_ambiguous 38858, nonunique_assignment 6414.
  The frozen aggregate `unresolved_slots=0` counts only literal unresolved/unidentified reason strings — do not cite it
  as completeness. Correspondence spot-checks: 1417 branches, min accepted overlap 0.8260510628196013,
  max accepted freq distance 6.041234580665498 Hz. Acquisition spot-checks: 294 windows, max counterfactual feature
  error 6.178599298856113e-13, max eigen residual 2.9729071648410545e-15.
- `recovered_modal_shapes.npz` (mandatory): restores lost imaginary diagnostics only — 35362 windows,
  max real relative error 4.657600823879598e-10, max recovered imaginary 71.20985818732326;
  features, oracle rows, predictions unchanged.
- Applicability: primary additional12 + original4 fitting + 2 validation designs retain all 5 profiles.
  Only secondary D000053 farther-stable is physically inapplicable (amendment: v1 acquisition failed at .78 ratio,
  Mach 1.8074338014281053 < preserved Mach 2; cannot generate within validated domain) — retained as **unknown**,
  not a safety pass. `protocol.json:applicable_profiles` confirms D000053 = cross/stable/retreat/unstable_exclusion
  (no farther_stable); every other listed design has all 5.
- No qualification/progress credit is taken from oracle or development fits (preregistered interpretation).

## 2. Aggregation definitions (explicit)

- `equal_design_mae`: mean over designs of per-design MAE, where each per-design MAE is itself averaged over that
  design's frames for one seed; then averaged over seeds 42/43/44. Seed-specific **errors** are averaged —
  predictions are never averaged (`no_seed_prediction_averaging:true`).
- Paired contrasts are per-design-per-seed MAE differences (12 cells = 4 designs × 3 seeds per fold/stratum/noise):
  `expanded_minus_original_measured`, `expanded_minus_original_oracle`, `oracle_minus_measured_original`,
  `oracle_minus_measured_expanded` (each with `delta_by_design_seed`, `signed_delta_by_design_seed`, `mean_delta` =
  mean of the 12, `improved_every_design_seed` flag). Negative delta = second-named arm worse / first-named arm better
  for `expanded_minus_original` (expanded wins); for `oracle_minus_measured`, negative = oracle wins.
- `support_x_content_interaction` = (expanded−original)₍oracle₎ − (expanded−original)₍measured₎ per design-seed;
  `mean_interaction` over 12. Negative = expanded-support benefit is larger (or oracle penalty smaller) under oracle
  content than under measured content. Direction retained everywhere below; it is a description of this factorial, not
  an attribution of information value.
- "Mean wins by fold/noise": count of the 7 paired strata (`all/current/gap/boundary/boundary_current/farther_stable/
  unstable_exclusion`) with mean delta (or mean interaction) < 0. "Design/seed wins": count of the 12 (per stratum) or
  84 (pooled over 7 strata) design-seed cells with delta < 0. Denominators always stated; no silent passing.
- Score strata reported: all, current, gap, boundary, boundary_current (+ farther_stable / unstable_exclusion slices).
  Profile aggregates: `profile_original3` (cross+stable+retreat), `profile_new2` (farther_stable+unstable_exclusion),
  `profile_retreat`, plus boundary/current/gap splits. Boundary farther-stable is structurally empty (0 frames,
  §3) — reported as empty, not zero error.
- Noises reported as primary: `clean`, original `iid1`, original `correlated1`, and **both** fresh paired 1% draws
  (`iid1_fresh20261020`, `correlated1_fresh20261020`, `iid1_fresh20261021`, `correlated1_fresh20261021`).
  5% stress (`iid5`/`correlated5` × original + 2 fresh draws each) is kept separate in §11.
- Shared targets (contract): all-frame MAE ≤ .010 clean / ≤ .015 each 1% family; boundary MAE ≤ .005 clean /
  ≤ .010 each 1% family; boundary optimism>.02 ≤ 5%; each .01≤|m|≤.05 wrong-sign ≤ 10%; farther-stable degradation ≤ .005.

## 3. Denominators (frames; identical across the 7 primary noises except stated)

Per-fold evaluation strata, clean (= all primary noises; verified identical):
- fold0: all 4960; current 2556; gap 2404; boundary 2047; boundary_current 1097; farther_stable 992; unstable_exclusion 992.
- fold1: all 4960; current 2273; gap 2687; boundary 2161; boundary_current 1160; farther_stable 992; unstable_exclusion 992.
- fold2: all 4960; current 2477; gap 2483; boundary 2006; boundary_current 1168; farther_stable 992; unstable_exclusion 992.
- Wrong-sign slice denominators (clean; same all primary noises): fold0: stable slice 1056 (current 846 / gap 210 /
  boundary 1056 / bndcur 846), exclusion slice 992 (current 29 / gap 963 / boundary 380 / bndcur 12), unstable-positive
  1558 (current 229 / gap 1329 / boundary 946 / bndcur 212). fold1: stable 1208 (900/308/1208/900), exclusion 992
  (41/951/379/15), unstable-positive 1488 (221/1267/875/195). fold2: stable 989 (796/193/989/796), exclusion 992
  (53/939/380/34), unstable-positive 1486 (266/1220/874/247). farther_stable stratum carries no wrong-sign
  denominators (null) in any fold.
- Profile aggregates (designs=4 each; boundary_profile_farther_stable = 0 frames / 0 designs in **all** folds):
  profile_original3 2976; profile_new2 1984; profile_retreat 992; profile_farther_stable 992;
  profile_unstable_exclusion 992; boundary_profile_original3: fold0 1667 / fold1 1782 / fold2 1626;
  boundary_profile_new2: fold0 380 / fold1 379 / fold2 380; boundary_profile_unstable_exclusion: 380/379/380
  (fold0: 3 designs); current_profile_original3: 2053/1774/1930; current_profile_new2: 503/499/547;
  gap_profile_original3: 923/1202/1046; gap_profile_new2: 1481/1485/1437.
- Margin strata (stable wrong-sign denominators live here): fold0: −0.02–0.0: 946 frames (stableWS 0 / exclWS 380);
  0.0–0.01: 45 (0/0); 0.01–0.02: 111 (111/0); 0.02–0.05: 945 (945/0); 0.05–0.1: 1309 (0/0). fold1: −0.02–0.0: 875
  (0/379); 0–0.01: 78; 0.01–0.02: 308 (308/0); 0.02–0.05: 900 (900/0); 0.05–0.1: 1194. fold2: −0.02–0.0: 874 (0/380);
  0–0.01: 143; 0.01–0.02: 223 (223/0); 0.02–0.05: 766 (766/0); 0.05–0.1: 1350.
- Native observability (`observation_coverage.json`, descriptive not a safety pass): primary_additional/clean:
  all 14880 frames (7306 observed); boundary 6214 (3425); cross/stable/retreat/farther_stable 2976 each
  (1620/2088/2049/1426 observed); unstable_exclusion 2976 (**123 observed**); unstable_sign_slice 2976 (123);
  farther_stable15 2976 (1426). secondary_held/clean: all 2232 (1209); boundary 1074 (679); cross 496 (310);
  stable 496 (363); retreat 496 (364); farther_stable 248 (140); unstable_exclusion 496 (**32**).

## 4. Absolute MAE (equal-design; expanded measured arm + original measured reference)

Targets recalled per cell: FAIL marks absolute-target failure (not a relative judgment).

fold0 — original measured: clean all .03661 / cur .02064 / gap .05388 / bnd .03293 / bndcur .01884;
iid1 all .05265 / cur .02092 / bnd .03748 / bndcur .01620; corr1 all .05433 / cur .02335 / bnd .03742 / bndcur .01791.
fold0 — **expanded** measured: clean all **.03387** / cur **.01513** / gap **.05381** / bnd **.02955** / bndcur **.01296**;
iid1 all .05204 / cur .01384 / bnd .03706 / bndcur .01116; corr1 all .05374 / cur .01675 / bnd .03720 / bndcur .01390;
iid1_f20 all .05126 / cur .01454 / bnd .03739 / bndcur .01229; corr1_f20 all .05484 / cur .01526 / bnd .03700 / bndcur .01265;
iid1_f21 all .05257 / cur .01402 / bnd .03650 / bndcur .01141; corr1_f21 all .05371 / cur .01576 / bnd .03749 / bndcur .01346.
fold1 — original: clean all .03505 / cur .01620 / gap .05137 / bnd .02804 / bndcur .01359.
fold1 — **expanded**: clean all **.03680** / cur **.01816** / gap **.05296** / bnd **.02825** / bndcur **.01266**;
iid1 all .05143 / cur .01542 / bnd .04016 / bndcur .01107; corr1 all .05205 / cur .01624 / bnd .04036 / bndcur .01263;
iid1_f20 all .05127 / cur .01568 / bnd .04029 / bndcur .01179; corr1_f20 all .05362 / cur .01767 / bnd .04100 / bndcur .01395;
iid1_f21 all .05170 / cur .01679 / bnd .04002 / bndcur .01156; corr1_f21 all .05273 / cur .01737 / bnd .04087 / bndcur .01394.
fold2 — original: clean all .03967 / cur .02796 / gap .05179 / bnd .03797 / bndcur .02710.
fold2 — **expanded**: clean all **.03827** / cur **.02523** / gap **.05324** / bnd **.03543** / bndcur **.02334**;
iid1 all .05088 / cur .02318 / bnd .04091 / bndcur .01890; corr1 all .05350 / cur .02693 / bnd .04258 / bndcur .02344;
iid1_f20 all .05192 / cur .02310 / bnd .04067 / bndcur .01869; corr1_f20 all .05303 / cur .02654 / bnd .04180 / bndcur .02357;
iid1_f21 all .05164 / cur .02336 / bnd .04092 / bndcur .01842; corr1_f21 all .05357 / cur .02877 / bnd .04252 / bndcur .02459.
- Absolute verdict: **every** all-frame cell FAILs (clean .0339–.0383 vs ≤.010; 1% family .0509–.0548 vs ≤.015).
  **Every** boundary cell FAILs (clean .0282–.0354 vs ≤.005; 1% .0365–.0426 vs ≤.010) and every boundary_current cell
  FAILs (clean .0130/.0127/.0233; 1% .0111–.0246). Expanded support narrows but never closes these gaps (§5 deltas are
  ≤ .009 in magnitude on current/bndcur and ~0 on gap). Gap MAE (measured-arm primary gap .0514–.0590; oracle/stress
  cells in §§5–6/11) dominates all-frame error; current-only MAE is 2–4× smaller than all-frame in every cell.

## 5. Support contrast: expanded − original, measured arm (mean_delta; negative = expanded wins)

| fold/stratum | clean | iid1 | corr1 | iid1_f20 | corr1_f20 | iid1_f21 | corr1_f21 |
|---|---|---|---|---|---|---|---|
| fold0 all | −.00273 | −.00061 | −.00059 | −.00067 | −.00050 | −.00058 | −.00062 |
| fold0 current | −.00551 | −.00709 | −.00660 | −.00725 | −.00721 | −.00898 | −.00749 |
| fold0 gap | −.00007 | +.00000 | −.00003 | −.00002 | +.00010 | +.00011 | −.00001 |
| fold0 boundary | −.00338 | −.00042 | −.00022 | −.00048 | −.00031 | −.00028 | −.00046 |
| fold0 bndcur | −.00588 | −.00504 | −.00401 | −.00499 | −.00531 | −.00538 | −.00512 |
| fold0 fstable | −.00238 | −.00109 | −.00113 | −.00151 | −.00071 | −.00077 | −.00134 |
| fold0 excl | −.00054 | −.00054 | −.00054 | −.00054 | −.00055 | −.00052 | −.00052 |
| fold1 all | +.00175 | +.00021 | +.00027 | +.00030 | +.00015 | +.00041 | +.00024 |
| fold1 current | +.00196 | −.00148 | −.00052 | −.00055 | −.00056 | +.00020 | +.00044 |
| fold1 gap | +.00159 | +.00039 | +.00035 | +.00040 | +.00021 | +.00044 | +.00021 |
| fold1 boundary | +.00020 | +.00014 | +.00054 | +.00045 | +.00061 | +.00026 | +.00047 |
| fold1 bndcur | −.00093 | −.00216 | −.00013 | −.00010 | +.00049 | −.00195 | −.00008 |
| fold1 fstable | +.00374 | −.00070 | −.00114 | −.00081 | −.00178 | −.00042 | −.00111 |
| fold1 excl | +.00379 | +.00379 | +.00381 | +.00376 | +.00380 | +.00383 | +.00379 |
| fold2 all | −.00140 | −.00082 | −.00092 | −.00108 | −.00079 | −.00088 | −.00097 |
| fold2 current | −.00273 | −.00494 | −.00459 | −.00539 | −.00320 | −.00510 | −.00491 |
| fold2 gap | +.00145 | +.00054 | +.00032 | +.00033 | +.00027 | +.00046 | +.00031 |
| fold2 boundary | −.00254 | −.00231 | −.00203 | −.00241 | −.00183 | −.00204 | −.00213 |
| fold2 bndcur | −.00375 | −.00485 | −.00349 | −.00510 | −.00260 | −.00402 | −.00415 |
| fold2 fstable | −.00210 | −.00029 | −.00092 | −.00081 | −.00075 | −.00096 | −.00147 |
| fold2 excl | +.00271 | +.00277 | +.00278 | +.00274 | +.00277 | +.00276 | +.00277 |

Support under the oracle arm (expanded − original, oracle; same layout): fold0 all −.00895/−.00143/−.00102/−.00146/
−.00095/−.00127/−.00106; current −.01693/−.01540/−.01106/−.01540/−.01307/−.01489/−.01268; gap −.00037/+.00003/+.00000/
−.00002/+.00017/+.00012/+.00001; boundary −.00995/−.00109/−.00042/−.00109/−.00055/−.00103/−.00063; bndcur −.01818/−.01317/
−.00908/−.01402/−.01248/−.01262/−.01134; fstable −.00418/−.00199/−.00171/−.00224/−.00130/−.00148/−.00197; excl −.00074/
−.00077/−.00075/−.00076/−.00075/−.00077/−.00074. fold1: all +.00051/−.00033/+.00001/−.00009/−.00018/−.00000/−.00013;
current −.00094/−.00891/−.00576/−.00649/−.00656/−.00603/−.00620; gap +0.00143–+.00071 all-positive; boundary −.00166/
−.00020/+.00070/+.00036/+.00062/+.00032/+.00042; bndcur −.00500/−.01066/−.00615/−.00737/−.00627/−.00808/−.00723;
fstable +.00274/−.00246/−.00251/−.00223/−.00304/−.00200/−.00250; excl +.00325/+.00319/+.00320/+.00320/+.00322/+.00323/
+.00322. fold2: all −.00540/−.00179/−.00176/−.00210/−.00174/−.00192/−.00189; current −.00954/−.00917/−.00914/−.01035/
−.00803/−.01056/−.00951; gap +0.00139–+.00022 all-positive; boundary −.00509/−.00272/−.00246/−.00302/−.00231/−.00291/
−.00253; bndcur −.00792/−.00947/−.00907/−.01069/−.00834/−.01140/−.00953; fstable −.00730/−.00232/−.00263/−.00287/
−.00272/−.00280/−.00337; excl +.00273/+.00271/+.00270/+.00281/+.00270/+.00273/+.00268.
- Reading: support helps current/bndcur in fold0 (all 7 noises) and fold2 (all 7), is ~null on gap everywhere
  (|mean| ≤ .0016), helps boundary in fold0/fold2 but is ~null-positive in fold1 boundary (clean +.00020, 1% +.
  00014–+.00061). fold1 all-frame degrades on clean (+.00175); fold1/fold2 unstable_exclusion degrades uniformly
  (+.0038/+.0027, noise-invariant because the slice is ~entirely gap frames). `improved_every_design_seed` is true
  in 7 of 147 support-measured cells — all fold0: current under corr1/iid1_f20/corr1_f20/corr1_f21 and farther_stable
  under iid1/corr1/iid1_f20 (12/12 design-seeds improve in each); the other 140 cells have at least one regressing
  design-seed.

## 6. Content contrast (oracle − measured) + support×content interaction

oracle − measured under **original** support (negative = oracle wins): fold0: all +.00594/+.00098/+.00061/+.00098/
+.00062/+.00095/+.00059; current +.01098/+.00953/+.00623/+.00959/+.00741/+.00825/+.00650; gap +.00038/+.00003/+.00005/
+.00005/−.00002/+.00005/+.00005; boundary +.00679/+.00100/+.00057/+.00091/+.00063/+.00124/+.00048; bndcur +.01255/+.00978/
+.00693/+.01088/+.00928/+.00993/+.00797; fstable +.00237/+.00123/+.00095/+.00099/+.00091/+.00099/+.00103; excl +.00036/
+.00039/+.00038/+.00040/+.00037/+.00042/+.00038. fold1: all +.00114/+.00053/+.00028/+.00038/+.00040/+.00042/+.00038;
current +.00266/+.00745/+.00533/+.00579/+.00658/+.00612/+.00674; gap +.00017/−.00029/−.00029/−.00026/−.00032/−.00026/
−.00032; boundary +.00220/+.00052/+.00005/+.00026/+.00025/+.00016/+.00026; bndcur +.00466/+.00944/+.00723/+.00815/
+.00817/+.00730/+.00834; fstable +.00126–+.00165; excl +.00060–+.00065. fold2: all +.00304/+.00082/+.00047/+.00102/
+.00065/+.00093/+.00057; current +.00449/+.00336/+.00233/+.00445/+.00319/+.00458/+.00205; gap +.00029/−.00008/−.00017/
−.00003/−.00010/−.00014/−.00011; boundary +.00080/+.00060/+.00022/+.00095/+.00031/+.00111/+.00022; bndcur +.00027/
+.00378/+.00326/+.00505/+.00366/+.00667/+.00294; fstable +.00650/+.00168/+.00129/+.00195/+.00172/+.00148/+.00155;
excl +.00029–+.00037.
- Content direction under original support is almost entirely **harmful-or-null** (134 of 147 fold/noise/stratum
  means positive; the 13 negatives are all gap-stratum 1% cells ≈ −.00002–−.00033: fold0 corr1_f20; fold1 all six 1%
  noises; fold2 all six 1% noises). On current/boundary/bndcur the oracle never wins (105/105 means positive).
  Richer matched-physical-description maps do not buy accuracy under original4 support — but (§Limitation) this is
  a 38%-replacement counterfactual, not evidence that the information is absent.
oracle − measured under **expanded** support: fold0: all −.00027/+.00016/+.00018/+.00019/+.00018/+.00026/+.00015;
current −.00044/+.00123/+.00177/+.00144/+.00154/+.00234/+.00131; gap +.00009/+.00005/+.00008/+.00005/+.00005/+.00006/
+.00007; boundary +.00023/+.00033/+.00037/+.00030/+.00039/+.00050/+.00031; bndcur +.00025/+.00165/+.00186/+.00184/
+.00211/+.00270/+.00176; fstable +.00057/+.00033/+.00037/+.00025/+.00032/+.00029/+.00039; excl +.00016/+.00017/+.00017/
+.00017/+.00017/+.00016. fold1: all −.00010/+.00000/+.00002/−.00001/+.00007/+.00000/+.00001; current −.00024/+.00001/
+.00009/−.00015/+.00058/−.00010/+.00010; gap ≈ +.00001; boundary +.00034/+.00017/+.00021/+.00017/+.00026/+.00021/+.00021;
bndcur +.00059/+.00094/+.00121/+.00088/+.00141/+.00117/+.00120; fstable +.00026/−.00012/−.00011/−.00013/+.00003/−.00012/
−.00006; excl +.00002–+.00006. fold2: all −.00097/−.00014/−.00038/+.00001/−.00030/−.00011/−.00034; current −.00232/
−.00087/−.00222/−.00051/−.00165/−.00088/−.00255; gap +.00023/−.00016/−.00018/−.00007/−.00015/−.00018/−.00012; boundary
−.00175/+.00019/−.00022/+.00033/−.00018/+.00025/−.00018; bndcur −.00389/−.00084/−.00232/−.00054/−.00207/−.00070/−.00244;
fstable +.00131/−.00034/−.00042/−.00010/−.00024/−.00036/−.00034; excl +.00026–+.00032.
- Interaction (support×content mean; negative = support benefit larger under oracle, i.e. expanded support removes
  the oracle penalty): fold0: all −.00621/−.00082/−.00042/−.00079/−.00044/−.00069/−.00044; current −.01142/−.00831/
  −.00446/−.00815/−.00586/−.00592/−.00519; gap −.00030/+.00003/+.00003/+.00000/+.00007/+.00001/+.00002; boundary −.00657/
  −.00067/−.00020/−.00061/−.00024/−.00074/−.00017; bndcur −.01230/−.00813/−.00507/−.00903/−.00717/−.00724/−.00621;
  fstable −.00180/−.00090/−.00058/−.00074/−.00059/−.00070/−.00063; excl −.00020–−.00025. fold1: all −.00124/−.00053/
  −.00026/−.00039/−.00033/−.00042/−.00037; current −.00290/−.00744/−.00524/−.00594/−.00600/−.00623/−.00664; gap −.00016/
  +.00029/+.00029/+.00026/+.00032/+.00026/+.00033; boundary −.00186/−.00034/+.00016/−.00009/+.00001/+.00006/−.00005;
  bndcur −.00407/−.00850/−.00602/−.00727/−.00676/−.00614/−.00715; fstable −.00100/−.00176/−.00136/−.00142/−.00127/
  −.00158/−.00139; excl −.00054–−.00060. fold2: all −.00401/−.00096/−.00084/−.00102/−.00095/−.00104/−.00091; current
  −.00681/−.00423/−.00455/−.00496/−.00484/−.00545/−.00460; gap −.00006/−.00008/−.00001/−.00004/−.00005/−.00005/−.00001;
  boundary −.00255/−.00041/−.00043/−.00061/−.00049/−.00087/−.00040; bndcur −.00416/−.00462/−.00558/−.00559/−.00573/
  −.00738/−.00538; fstable −.00519/−.00203/−.00171/−.00205/−.00197/−.00184/−.00190; excl +.00002/−.00006/−.00008/
  +.00006/−.00007/−.00003/−.00010.
- Interaction direction: negative in 131 of 147 fold/noise/stratum cells (16 non-negative: fold0 gap 1% six cells
  +.00000–+.00007; fold1 gap six 1% cells +.00026–+.00033 plus boundary corr1 +.00016 / corr1_f20 +.00001 /
  iid1_f21 +.00006; fold2 excl clean +.00002 / iid1_f20 +.00006). Support and content are
  sub-additive in this factorial: the oracle arm benefits more from expanded support because it starts worse.

## 7. Win counts

Mean wins (strata with mean<0 of 7): support-measured — fold0: clean 7/7, iid1 6/7, corr1 7/7, iid1_f20 7/7,
corr1_f20 6/7, iid1_f21 6/7, corr1_f21 7/7. fold1: clean 1/7, iid1 3/7, corr1 3/7, iid1_f20 3/7, corr1_f20 2/7,
iid1_f21 2/7, corr1_f21 2/7. fold2: 5/7 in all 7 noises. Content-original (oracle wins): fold0 0/7 clean/iid1/corr1/
both fresh iid1, 1/7 corr1_f20 (gap −.00002 only); fold1 0/7 clean, 1/7 in each 1% noise; fold2 0/7 clean, 1/7 in each
1% noise. Content-expanded: fold0 clean 2/7, else 0/7; fold1 clean 2/7, iid1 1/7, corr1 1/7, iid1_f20 3/7,
corr1_f20 0/7, iid1_f21 2/7, corr1_f21 1/7; fold2 clean 4/7, iid1 5/7, corr1 6/7, iid1_f20 4/7, corr1_f20 6/7,
iid1_f21 5/7, corr1_f21 6/7.
- Design/seed wins pooled over 7 strata (of 84): support-measured — fold0: 62/64/67/65/63/61/66
  (clean/iid1/corr1/iid1_f20/corr1_f20/iid1_f21/corr1_f21); fold1: 30/34/35/34/36/32/37; fold2: 41/38/38/43/37/40/45.
  Content-original negatives of 84: fold0 19/26/27/28/33/25/28; fold1 30/21/31/30/27/33/27; fold2 34/42/41/39/40/37/43.
  Content-expanded: fold0 38/30/28/33/28/31/27; fold1 33/37/29/33/20/33/34; fold2 54/48/52/42/49/49/50.
  Interaction negatives of 84: fold0 46/44/43/38/40/37/37; fold1 53/61/54/57/55/56/59; fold2 51/49/50/50/54/52/52.
- Clean per-stratum design/seed wins (order supMeas|supOracle|oracleOrig|oracleExp|interact, /12): fold0:
  all 9|11|1|5|9; current 9|11|1|5|9; gap 8|8|4|7|4; boundary 9|10|2|5|7; bndcur 9|10|2|5|7; fstable 9|8|5|5|6;
  excl 9|8|4|6|4. fold1: all 5|3|5|5|7; current 6|6|7|5|6; gap 0|0|4|4|8; boundary 7|7|3|5|8; bndcur 8|9|3|5|8;
  fstable 4|5|4|6|8; excl 0|0|4|3|8. fold2: all 7|6|5|8|7; current 8|7|4|8|7; gap 0|2|4|7|8; boundary 8|9|7|9|7;
  bndcur 8|9|6|9|7; fstable 10|9|3|5|8; excl 0|1|5|8|7. Note fold1/fold2 gap+excl support cells: 0/12 clean wins —
  expanded support never wins a gap-dominated design-seed there.

## 8. Profile aggregates (equal-design MAE; Δ = expanded − original measured)

fold0 (frames: orig3 2976 / new2 1984 / retreat 992 / bnd-orig3 1667 / bnd-new2 380):
clean: orig3 .02525→.02167 (Δ−.00358); new2 .05364→.05218 (Δ−.00146); retreat .01988→.01596 (Δ−.00392);
bnd-orig3 .03207→.02836 (Δ−.00371); bnd-new2 .03754→.03716 (Δ−.00038).
1% family Δ range: orig3 −.00042–−.00054; new2 −.00063–−.00102; retreat −.00017–−.00068; bnd-orig3 −.00013–−.00037;
bnd-new2 −.00039–−.00041 (all six 1% noises; per-noise: iid1 −.00047/−.00082/−.00017/−.00031/−.00040;
corr1 −.00043/−.00084/−.00032/−.00013/−.00040; iid1_f20 −.00044/−.00102/−.00061/−.00037/−.00039;
corr1_f20 −.00042/−.00063/−.00057/−.00019/−.00041; iid1_f21 −.00054/−.00065/−.00068/−.00017/−.00041;
corr1_f21 −.00042/−.00093/−.00042/−.00033/−.00039).
fold1 (bnd-orig3 1782 / bnd-new2 379): clean: orig3 .02350→.02390 (Δ+.00040); new2 .05240→.05610 (Δ+.00376);
retreat .01920→.01949 (Δ+.00029); bnd-orig3 .02780→.02720 (Δ−.00062); bnd-new2 .02500→.02910 (Δ+.00408).
1% Δ: orig3 −.00043–−.00069 (all negative); new2 +.00101–+.00171 (all positive); retreat −.00074–−.00118 (all
negative); bnd-orig3 −.00007–−.00062 (all negative); bnd-new2 +.00407–+.00411 (all positive, all 7 noises).
fold2 (bnd-orig3 1626 / bnd-new2 380): clean: orig3 .02960→.02710 (Δ−.00253); new2 .05480→.05510 (Δ+.00030);
retreat .02694→.02490 (Δ−.00203); bnd-orig3 .03630→.03290 (Δ−.00334); bnd-new2 .04290→.04300 (Δ+.00016).
1% Δ: orig3 −.00200–−.00244; new2 +.00065–+.00124; retreat −.00152–−.00232; bnd-orig3 −.00246–−.00319;
bnd-new2 +.00028–+.00033 (all six 1% noises positive).
- Retreat absolute (orig→exp): fold0 clean .01988→.01596, 1% .02694–.03260→.02626–.03218; fold1 clean .01920→.01949,
  1% .03345–.03536→.03227–.03461; fold2 clean .02694→.02490, 1% .02732–.03068→.02567–.02901. Retreat improves with
  support in 20 of 21 fold/noise cells (exception fold1 clean +.00029).
- Added-2-profile (new2) regression concentrates in fold1 (all 7 noises +.00101–+.00408) and fold2 bnd-new2
  (+.00016–+.00033); fold0 new2 improves in all 7 (−.00063–−.00146). Original-3-profile improves in 20 of 21 cells
  (exception fold1 clean +.00040).

## 9. Guards: optimism>.02, wrong-sign slices, farther-stable

- Boundary optimism>.02 (target ≤5%; by-seed rates, expanded arm): fold0 boundary clean .4353/.4485/.5374;
  iid1 .6141/.6048/.6185; corr1 .5755/.5696/.5823; iid1_f20 .6224/.6160/.6287; corr1_f20 .6150/.6111/.6243;
  iid1_f21 .5721/.5647/.5774; corr1_f21 .5965/.5921/.6048 — FAIL all 21 cells (nearest: fold1 seed42 clean .3915 still
  8× target). fold1 boundary: clean .3915/.3938/.5752; 1% .7099–.7436. fold2 boundary: clean .6152/.5160/.4736;
- Boundary_current optimism (expanded): fold0 clean .1331/.1705/.3163, 1% .0609–.1083;
  fold1 clean .0259/.0328/.3793, 1% .0337–.2796; fold2 clean .4101/.2654/.1918, 1% .0855–.4276 — FAIL on the
  seed-averaged rate in all 21 fold/noise cells. At seed level, 7 of 63 cells pass, all fold1 expanded bndcur:
  clean seed42 .0259 / seed43 .0328; iid1 seed42 .0358 / seed43 .0409; iid1_f20 seed43 .0337;
  iid1_f21 seed42 .0494 / seed43 .0442; the other 56 fail (seed44 always .1918–.3793; all fold0/fold2 seeds fail).
  No fold/noise cell passes on all 3 seeds. Original arm is no better (fold0 boundary orig clean .4568/.4211/.5164).
  Gap optimism ≈ .36–.56 both arms in every cell (fold0 .357–.509; fold1 .464–.556; fold2 .421–.525).
- Stable wrong-sign (target ≤10% on each .01≤|m|≤.05 slice; boundary-slice by-seed, expanded): fold0 clean
  .0019/.0057/.0047 PASS; iid1 .0568/.0616/.0540 PASS; corr1 .1004/.1013/.0956 (2 of 3 seeds marginally FAIL);
  iid1_f20 .0597/.0616/.0568 PASS; corr1_f20 .0568/.0606/.0521 PASS; iid1_f21 .1032/.1061/.0994 FAIL (2 of 3);
  corr1_f21 .1004/.1023/.0966 (2 of 3 marginally FAIL). fold1: clean .0033/.0281/.0381 PASS; all six 1% ≤.0083 PASS.
  fold2: clean .0313/.0000/.0000 PASS; all six 1% ≤.0142 PASS. Exclusion wrong-sign (expanded): fold0 all-slice
  .5323/.5081/.5403, boundary .5868/.5553/.5974; fold1 all .6623/.6704/.6008, boundary .5963/.6095/.5119; fold2 all
  .6804/.6673/.6331, boundary .7368/.7211/.7053 — FAIL every cell by ~5–7×; identical across the 7 primary noises
  within each fold (noise-invariant: the slice is gap-dominated).
- Margin slices .01–.02 / .02–.05 stable wrong-sign (by-seed; orig|exp): fold0 clean: .01–.02 (111 frames)
  orig .1441/.3694/.2072 FAIL → exp .0180/.0450/.0450 PASS; .02–.05 (945) orig .0275/.1323/.0339 (seed43 FAIL) →
  exp .0000/.0011/.0000 PASS. fold0 1%: .01–.02 orig iid1 .0090/.0631/.0000, exp .0270/.0270/.0000; corr1 orig
  .1081/.1351/.0991, exp .1261/.1081/.0991 (no support separation; all ≤.1352). .02–.05 1%: orig/exp both .0593–.1037
  (no separation). fold1 clean: .01–.02 (308) orig .0455/.1851/.2110 → exp .0130/.0065/.1006 (seed44 borderline);
  .02–.05 (900) orig .0000/.0478/.0222 → exp .0000/.0356/.0167 PASS. fold1 1%: all six noises both margins ≤.0715;
  per-seed exp≤orig in 26 of 36 seed comparisons (6 noises × 2 margins × 3 seeds). fold2 clean: .01–.02 (223)
  orig .0448/.0135/.0090 → exp .1211/.0000/.0000 (seed42 regresses); .02–.05 (766) orig .0170/.0039/.0117 →
  exp .0052/.0000/.0000. fold2 1%: all ≤.0359, per-seed exp≤orig in 29 of 36. Denominators: §3; no margin cell is
  silently omitted (0.0–0.01 and 0.05–0.1 carry no stable denominators by construction).
- Farther-stable guard (target degradation ≤ .005 on farther_stable MAE, expanded−original): fold0 clean −.00238;
  1% −.00071–−.00151. fold1 clean +.00374 (PASS, degradation within .005); 1% −.00042–−.00178. fold2 clean −.00210;
  1% −.00029–−.00147. Absolute farther-stable MAE stays large: clean orig→exp fold0 .06901→.06662,
  fold1 .06627→.07001, fold2 .07046→.06836; 1% family .11608–.14157→.11527–.14086. Guard passes in all 21 cells
  (|Δ| ≤ .00374), but absolute error is the worst of any stratum and boundary-overlap is nil (§3: 0 frames), so the
  guard says nothing about boundary behavior there.

## 10. Heterogeneous failures and regressions (design/seed level, measured support contrast, clean)

- fold0 (winning fold, still heterogeneous): D000146-seed44 regresses on all (+.00118), current (+.00236),
  boundary (+.00591), bndcur (+.01237); D000152-seed42 regresses on all (+.00161), current (+.00262);
  D000008-seed42 ~null (+.00034 all). Largest wins: D000008-seed44 current −.01513 / bndcur −.01534; D000062 all
  seeds −.00417–−.00521 all-frame. Gap per-design deltas ∈ [−.00056, +.00051] — null for every design-seed.
- fold1: seed44 dominates regressions — D000020-seed44 all +.00695 / current +.01653 / boundary +.00867 / bndcur
  +.01582; D000144-seed44 +.00616/+.01066/+.00486/+.00857; D000120-seed44 current +.01171 / fstable +.00982;
  D000029-seed44 fstable +.01303. Gap degrades in **all 12** cells (+.00044–+.00234); unstable_exclusion degrades in
  all 12 (+.00187–+.00571). Current wins exist (D000020-seed43 −.00954, D000120-seed42 −.00723) but do not survive
  aggregation (fold mean current +.00196).
- fold2: D000115-seed44 all +.00626 / current +.01565 / boundary +.00838 / bndcur +.02145 / fstable −.00909 (sign
  flips by stratum within one design-seed); D000114-seed42 fstable +.01076, D000115-seed42 fstable +.02043;
  D000158-seed42 wins all −.01487 / current −.02376 / boundary −.01820 / bndcur −.02727 (largest single-cell win in
  the campaign) while D000115-seed42 loses bndcur −.02607 on the same fold. Gap degrades in all 12 (+.00008–+.00333);
  excl degrades in all 12 (+.00060–+.00425).
- Systematic: gap predictions are support-invariant to ~1e-4 (all folds/noises); unstable_exclusion inherits that
  null (fold0) or degrades (folds1–2); current/bndcur carry all support signal with seed44 as the recurring
  outlier in both directions.

## 11. 5% stress (separate; native iid5/corr1-5 views + 2 fresh draws each; mean Δ = expanded − original measured)

fold0: iid5 all +.000119 (orig .06411/exp .06423) / cur −.005903 (.01647/.01057) / bnd +.000339 (.03604/.03638) /
bndcur −.004438 (.01325/.00881); corr5 all +.000389 (.06744/.06783) / cur −.003891 (.01737/.01348) / bnd +.000531
(.03766/.03819) / bndcur −.003558 (.01638/.01282). Fresh draws replicate the pattern (all/bnd ≈ +.00016–+.00050;
cur/bndcur ≈ −.00291–−.00557).
fold1: iid5 all −.000699 / cur −.000411 / bnd +.000472 / bndcur −.000422; corr5 all −.000874 / cur +.001233 /
bnd +.000594 / bndcur +.002066 (corr5_f21 bndcur +.003214 worst stress cell); fresh draws: iid5_f20/f21 bndcur
−.001888/−.002493, corr5_f20 bndcur −.000695.
fold2: all six 5% noises improve on all four strata: all −.001457–−.001775; cur −.003075–−.005563;
bnd −.001596–−.002103; bndcur −.002782–−.005047 (absolutes e.g. iid5 all .06477→.06307, bndcur .02166→.01740).
Stress does not change the absolute verdict (all/boundary far above targets) and does not reverse the gap null.

## 12. Secondary held (reference only; 2232 frames = D000053 992 + D000126 1240)

Original-measured clean: all .04389; current .04201 (1209 fr); gap .04497 (1023); boundary .04578 (1074);
bndcur .03693 (679); farther_stable .06234 (248 fr, D000126 only — D000053 inapplicable/unknown per §1);
unstable_exclusion .04038 (496). Per-design all-MAE by seed: D000053 .05738/.05503/.03666 (992 fr);
D000126 .04054/.03961/.03409 (1240 fr). Farther-stable per-design: D000126 only .06886/.06399/.05418.
Optimism clean (orig): all .6456/.6331/.5565; boundary .8324/.8138/.6806; bndcur .7850/.7526/.5346.
StableWS clean ≈ 0–.0021; exclWS clean all .5968/.6048/.6331, boundary .7790/.7901/.8122. Held corroborates the
primary pattern (large absolute error, high optimism/exclusion wrong-sign) with no qualification claim.

## 13. Limitations (must-read alongside any use of §§4–12)

2. Partial oracle: 38.4% slot replacement; 145810 slots keep measured entries (100538 no-frequency, 38858
   spatially ambiguous, 6414 nonunique). The almost-entirely-non-positive content contrast (§6: 134/147 means
   positive, negatives only in gap 1% cells) is conditional on this
   counterfactual — **do not** read it as proof that measured modal features carry no information about the target.
3. Native observability is thin where errors are worst: exclusion slice 123/2976 primary frames observed
   (32/496 secondary); gap dominates error while being support-invariant.
4. Boundary farther-stable is empty (0 frames/0 designs, all folds); the passing farther-stable guard (§9) therefore
   constrains only non-boundary error at the worst absolute level in the study.
5. Heterogeneity: seed44 drives both the largest wins and largest regressions; `improved_every_design_seed`=true
   in only 7 of 147 support-measured cells (all fold0 current/farther-stable); fold1 all-frame degrades on clean
   while its bndcur improves — any summary that averages over folds without showing §§5/10 hides sign flips.
6. Absolute targets fail everywhere except isolated stable wrong-sign margin cells and 7 seed-level bndcur optimism
   cells (§9); relative gains in §5
   (largest: fold0 oracle-arm bndcur clean −.01818; measured-arm current clean −.00551–−.00898 across 1% draws)
   never approach the absolute thresholds.
7. Secondary D000053 farther-stable remains unknown (Mach 1.8074 < 2); excluded from denominators, not counted as
   pass or fail.
