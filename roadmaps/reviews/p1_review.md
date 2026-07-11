# P1 Multi-Fidelity Correction Field — Code Review

## Summary

The PCA-based autoencoder + latent GP architecture is sound for low-data regimes. The boundary envelope `b(x,y)=x(1-x)y(1-y)` is correctly applied. However, the INR baseline is non-functional (no gradient computation), the coverage metric mixes latent/field-space units, and there is no function to apply the trained correction to actual FSDT predictions. Test suite is predominantly shape-checking with few behavioral assertions.

## Critical Issues

### 1. INR baseline doesn't train (`train.py:100-103`)

```python
# Numerical gradient (very simple — just track loss)
# For real training you'd want autograd; this is a placeholder
```

The training loop computes forward passes but never updates weights. The returned "trained" INR is identical to initialization. Every test on INR output is testing random weights. Either implement numerical gradient (finite differences), use JAX/torch, or remove the INR and label it as a design sketch, not a trainable baseline.

### 2. Coverage metric is dimensionally wrong (`train.py:149-152`)

```python
coverage = float(np.mean(
    np.abs(test_true - decoder.decode(z_pred_mean, apply_boundary=True))
    < 2 * np.sqrt(z_pred_var.mean(axis=1))[:, np.newaxis, np.newaxis]
))
```

`z_pred_var` is `(n_test, d_z)` — variance in **latent space**. `test_true - test_recon` is a **field-space** residual `(n_test, ny, nx)`. Comparing them directly is comparing apples to oranges. To compute field-space uncertainty, you need to decode through the decoder's Jacobian: `field_var ≈ J @ diag(z_var) @ J^T`, or sample many z from the GP posterior, decode each, and compute field-space statistics. The current metric is undefined.

### 3. `extract_correction_fields` destroys sign information (`data.py:180-183`)

```python
lf_abs = np.abs(lf_shapes)
hf_abs = np.abs(hf_shapes)
lf_abs[lf_abs < 1e-12] = 1e-12
return hf_abs / lf_abs - 1.0
```

Mode shapes have physical sign conventions (e.g., upward vs downward displacement). Taking `abs()` means a correction field of `+0.04` could mean either "HF is 4% larger in magnitude" or "HF has opposite sign and is 4% larger" — the model can't distinguish these. The correct approach: normalize by the sign of the LF mode shape, or use `np.sign(lf_shapes) * hf_shapes / np.abs(lf_shapes)`. The comment "signs are arbitrary" is only true globally per mode, not element-wise.

### 4. `generate_lf_dataset` uses `result` outside loop scope (`data.py:109-110`)

```python
all_shapes[i] = result.mode_shapes
...
return {
    ...
    "grid_x": result.grid_x,  # relies on last loop iteration
    "grid_y": result.grid_y,
}
```

If `n_samples=0`, `result` is undefined → `NameError`. Even with `n_samples > 0`, this relies on the implicit variable leaking from the loop body. Extract `grid_x`/`grid_y` outside the loop.

## Improvements

### 5. GP Cholesky jitter is fragile (`gp_model.py:56-62`)

Single retry with `1e-6` jitter. If the kernel matrix is ill-conditioned, this may still fail. Use an exponential jitter search: `for jitter in [1e-8, 1e-6, 1e-4, 1e-2]: try Cholesky`. Or use `np.linalg.solve` with regularization as a fallback.

### 6. No `apply_correction` function

The pipeline trains a model that predicts correction fields, but there is no function to apply the correction to FSDT predictions:

```python
# This is missing:
def apply_correction(lf_result, gp, encoder, decoder, theta):
    """Correct FSDT mode shapes using trained multi-fidelity model."""
    ...
```

Without this, the trained model is a dead end. The user must manually wire encoder → GP → decoder → multiply with LF mode shapes.

### 7. No Sobol index computation

The proposal mentions sensitivity analysis via Sobol indices, but the code has no implementation. The `SampleConfig` samples parameters but never computes their importance. Add `compute_sobol_indices(theta, correction_fields)` or remove the claim.

### 8. `full_pipeline` split logic (`train.py:120-123`)

```python
n_test = max(1, int(n * test_fraction))
perm = rng.permutation(n)
train_idx = perm[n_test:]
test_idx = perm[:n_test]
```

With `n=3` and `test_fraction=0.2`: `n_test=1`, `n_train=2`. A GP with 2 training points is meaningless. Add a minimum `n_train` guard (e.g., `n_train = max(d_z + 1, n - n_test)`).

### 9. `CorrectionDecoder` grid convention (`decoder.py:24-25`)

```python
gx = np.linspace(0, 1, self._grid_size[0])
gy = np.linspace(0, 1, self._grid_size[1])
```

`_grid_size` comes from `encoder.grid_size` which is `(grid_nx, grid_ny)`. So `gx` gets `grid_nx` points and `gy` gets `grid_ny` points. This is correct, but the naming `_grid_size` is ambiguous — should be `_grid_shape` or `(nx, ny)` in the docstring.

## Nice-to-Haves

- **`train_inr` reuses same batch every step** (`train.py:91`): `rng.choice` with `replace=False` is fine, but the batch size is fixed at 8 regardless of dataset size. For n=1000, batch=8 is wastefully small.
- **INR positional encoding hardcodes `2*pi*k`** (`inr_baseline.py:54-57`): Consider making frequency scaling configurable or learning it.
- **`svds` vs `svd` fallback logic** (`encoder.py:34`): The `n > 2` check is conservative. `scipy.sparse.linalg.svds` works fine for `n >= k+1`. The fallback to full SVD for `n <= 2` is correct but could be cleaner.
- **GP stores per-dimension Cholesky inverses** (`gp_model.py:48-62`): With d_z=16, storing 16 `K_inv` matrices of size `n×n`. For n=500, that's 16 × 2MB = 32MB. Fine for this use case but worth noting.

## Test Coverage Gaps

### Behavioral tests needed (not shape checks):

1. **End-to-end correction accuracy**: Generate LF/HF pairs → train pipeline → verify corrected frequencies are closer to HF than LF alone.
2. **Boundary enforcement correctness**: Verify that `decoder.decode(z, apply_boundary=True)` produces fields that are *exactly* zero at edges (current test does this, good).
3. **GP interpolation accuracy**: With known function `z = f(θ)`, verify GP prediction error at training points is near zero.
4. **Encoder reconstruction invariance**: Verify `encode` is linear (PCA property) — `encode(a*f1 + b*f2) ≈ a*encode(f1) + b*encode(f2)`.
5. **Edge case: `n_samples=1`**: `generate_lf_dataset` with n=1, encoder with 1 sample.
6. **Edge case: `d_z > n_samples`**: Encoder should handle gracefully (it caps k, but no test).
7. **Edge case: NaN/Inf in mode shapes**: What happens when the solver returns NaN modes?
8. **Edge case: collinear design parameters**: GP with duplicate θ values.
9. **`create_synthetic_hf` frequency perturbation**: No test verifies the frequency shift is applied correctly (only mode shapes tested in `test_synthetic_hf_correction_fields`).
10. **GP posterior calibration**: Over many test points, the 95% credible interval should contain ~95% of true values. Currently untested.
11. **`full_pipeline` integration test**: Missing entirely.
12. **PCA variance explained**: No test checks that d_z components capture sufficient variance (e.g., >90%).

## Specific Code References

| Issue | File:Line | Severity |
|-------|-----------|----------|
| INR doesn't train | `train.py:100-103` | Critical |
| Coverage metric units mismatch | `train.py:149-152` | Critical |
| Sign info destroyed | `data.py:180-183` | Critical |
| `result` scope leak | `data.py:109-110` | Critical |
| Cholesky jitter fragility | `gp_model.py:56-62` | Improvement |
| No `apply_correction` API | (missing) | Improvement |
| No Sobol indices | (missing) | Improvement |
| Small-n train/test split | `train.py:120` | Improvement |
| INR batch too small for large n | `train.py:91` | Nice-to-have |
| `_grid_size` naming ambiguity | `decoder.py:21` | Nice-to-have |
