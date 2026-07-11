# P1 Multi-Fidelity Correction Field — Code Review R2

## Critical Issues from R1 — Fix Verification

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | INR doesn't train | **FIXED** | `_train_step` now has full manual backprop (forward + backward + SGD update) |
| 2 | Sign info destroyed | **FIXED** | `extract_correction_fields` divides by signed `lf_shapes`, not `lf_abs` |
| 3 | Scope leak on `result` | **FIXED** | `grid_x`/`grid_y` initialized to `None`, extracted inside loop, fallback for n=0 |
| 4 | Coverage metric units mismatch | **FIXED** | Samples GP posterior → decodes to field space → computes field-space std before comparison |

All 4 critical issues are correctly resolved.

## New Issues Introduced by Fixes

### N1. INR training/inference boundary mismatch

`_train_step` (`inr_baseline.py:104`) computes loss on raw MLP output:
```python
output = h[..., 0]
loss = float(np.mean((output - all_tgt) ** 2))
```

But `predict_grid` (`inr_baseline.py:150`) applies boundary envelope at inference:
```python
delta = delta_raw.reshape(...) * self._b
```

Training targets from `extract_correction_fields` are raw (non-zero at boundaries). The INR is penalized during training for values it doesn't need to produce at inference, and the boundary enforcement zeros out whatever it learned near edges. Not a correctness bug — the boundary is a safety guarantee — but the training loss doesn't reflect inference behavior. Consider applying boundary inside `_train_step` too for consistency.

Severity: **Improvement**

### N2. `create_synthetic_hf` can produce negative frequencies

`data.py:165`:
```python
hf_freqs = lf_dataset["frequencies"] * (1.0 + rng.normal(0, 0.01, ...))
```

`rng.normal(0, 0.01, ...)` can produce values < -1, making frequencies negative. Physical frequencies must be positive. Should clamp or use log-normal perturbation.

Severity: **Improvement** (pre-existing, not introduced by fixes)

### N3. `full_pipeline` never trains INR

The pipeline trains autoencoder + GP but never trains the INR baseline. No end-to-end comparison between the two methods is possible without manually calling `train_inr`. The R1 improvement (missing `apply_correction` API) is also still absent — the trained model has no documented way to correct actual FSDT predictions.

Severity: **Improvement** (pre-existing)

## R1 Improvements Still Unresolved

| # | Issue | Status |
|---|-------|--------|
| 5 | GP Cholesky jitter fragile (single retry) | **Unfixed** |
| 6 | No `apply_correction` API | **Unfixed** |
| 7 | No Sobol index computation | **Unfixed** |
| 8 | No minimum n_train guard in `full_pipeline` | **Unfixed** — GP with 2 training points is meaningless |

## Test Quality Assessment

### Behavioral tests (good)

- `TestBoundaryEnvelope`: Verifies physical properties — vanishes at edges, positive inside, peak at center. Real behavioral tests.
- `TestGP.test_uncertainty_increases_away`: Checks that variance grows away from training data. Correct GP behavior.
- `TestEndToEnd.test_synthetic_hf_correction_fields`: Verifies correction values are ~0.04 for known perturbation. Behavioral.
- `TestDecoder.test_boundary_enforcement`: Verifies edges are exactly zero. Behavioral.
- `TestINR.test_boundary_enforcement`: Same. Behavioral.

### Shape checks (weak)

- `TestEncoder.test_encode_decode_shape`: Only checks output dimensions.
- `TestGP.test_fit_predict_shape`, `test_sample_shape`: Only dimensions.
- `TestINR.test_predict_grid_shape`, `test_predict_batch`: Only dimensions.
- `TestTrain.test_train_autoencoder`, `test_train_gp`: Only checks key presence and shape.

### Missing behavioral tests

1. **INR actually learns**: No test verifies loss decreases over training steps. The INR could return constant output and no test would catch it.
2. **GP interpolation accuracy**: With known `z = f(θ)`, verify GP prediction at training points ≈ true values (near-zero error).
3. **End-to-end improvement**: Generate LF/HF → train pipeline → verify corrected frequencies are closer to HF than LF alone.
4. **PCA variance explained**: No test checks that d_z components capture sufficient variance.
5. **Edge cases**: n_samples=1, d_z > n_samples, NaN in mode shapes, collinear θ values.
6. **GP posterior calibration**: 95% credible interval should contain ~95% of true values over many test points.

## Summary

The 4 critical fixes are correctly applied. One new improvement-level issue introduced (INR boundary mismatch). Four R1 improvements remain unfixed. Test suite is ~40% behavioral, ~60% shape-checking. The INR baseline trains but has no test verifying it actually learns. The pipeline still lacks an `apply_correction` API and end-to-end comparison between methods.
