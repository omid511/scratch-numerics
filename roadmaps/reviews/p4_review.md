# P4: Continuous Stability-Margin Estimation — Code Review

## Summary

Modal superposition transient generation is structurally sound and the TCN+quantile architecture is clean. However, the margin computation has a **unit mismatch bug** that produces nonsensical training labels, and `torch.sort` for quantile monotonicity kills gradient flow. Both must be fixed before any training run.

---

## Critical Issues (must fix)

### 1. `find_flutter_boundary` returns λ (non-dimensional), not velocity

`transient.py:90-92` stores the result as `u_crit` and computes:
```python
margin = (u_crit - velocity) / u_crit
```
But `solver.find_flutter_boundary()` returns a **non-dimensional lambda** (CFAP), not a velocity in m/s (`solver.py:495-498`). Mixing λ with velocity in the margin formula produces a physically meaningless label. Every training sample gets a wrong target.

**Fix:** Convert lambda to velocity before margin computation:
```python
from mechanics.piston_theory import velocity_from_lambda
D11 = solver._compute_D11()
u_crit_vel = velocity_from_lambda(u_crit, rho, solver.L1, D11, c_sound)
margin = (u_crit_vel - velocity) / u_crit_vel
```

### 2. `torch.sort` breaks quantile gradient signal

`quantile_head.py:44`:
```python
out, _ = torch.sort(out, dim=1)
```
`torch.sort` is non-differentiable (zero gradient almost everywhere). During backprop, the pinball loss gradient cannot propagate through the sort to adjust the raw head outputs. The model learns nothing from quantile ordering violations — the sort silently patches them at forward time but the loss never penalizes the raw predictions.

**Fix:** Replace sort with a differentiable monotonicity enforcement, e.g. cumulative sum of softplus on differences, or a learned monotonic transform (the `monotonic-linear` approach). Quick fix: remove sort, rely on pinball loss to implicitly enforce ordering (q0.05 naturally learns to predict below q0.50). That's not perfect but trains.

### 3. `evaluate_coverage` crashes on `NameError` for `hi`/`lo`

`train.py:119-133`: The interval variables `lo` and `hi` are only assigned inside `if tau == quantiles[0]` / `elif tau == quantiles[-1]` branches. Then `train.py:133` unconditionally references `hi` and `lo`:
```python
width = (hi - lo).mean().item()
```
If `len(quantiles) < 3`, or if `quantiles[0] == quantiles[-1]` (degenerate), this crashes with `NameError`.

**Fix:** Compute interval width conditionally only when `len(quantiles) >= 2`.

---

## Improvements (should fix)

### 4. Per-sensor normalization destroys inter-sensor amplitude relationships

`transient.py:81-83` normalizes each sensor independently to [-1, 1]. For a TCN learning flutter onset, relative amplitude between spatial sensors carries physical meaning (e.g., localized vs distributed deformation). Normalizing per-sensor makes all signals look uniformly scaled.

**Suggestion:** Normalize globally (one peak across all sensors) or don't normalize at all — the TCN can learn its own input scaling. If overflow from unstable modes is the concern, clip instead of normalize.

### 5. Broad exception swallowing in `generate_dataset` and margin computation

- `transient.py:93`: `except Exception: pass` silently discards `find_flutter_boundary` failures.
- `transient.py:131`: `except Exception: continue` silently skips broken solver calls.

Both produce silent data loss. A solver that throws on 80% of velocities yields a dataset of 20 clips with no warning.

**Fix:** At minimum, log a warning. Better: count and report how many samples were skipped.

### 6. `pinball_loss` returns per-quantile losses stacked, then `.mean()` — loses per-quantile signal

`quantile_head.py:62-63`: `torch.stack(losses).mean()` averages across quantiles. Low-quantile errors get drowned by the median. This is fine for a first pass but makes the model insensitive to tail calibration.

**Suggestion:** Consider weighted average or separate training for extreme quantiles.

### 7. TCN `BatchNorm1d` with small batches

`tcn.py:32-33`: With `batch_size=32` (train.py:19) and a small dataset (potentially <50 clips after NaN filtering), BN statistics are noisy. This adds training instability.

**Suggestion:** Use `GroupNorm` or `LayerNorm` for small-batch regimes.

---

## Nice-to-Haves

### 8. No lead-time computation anywhere

The proposal mentions "lead-time" (how far in advance the margin warning fires). The code has no concept of lead-time. `evaluate_coverage` only uses the last timestep. A sliding-window coverage metric would be more useful.

### 9. `generate_dataset` doesn't parallelize

`solver.solve_complex_modal` is pure numpy. `generate_dataset` runs sequentially over `n_samples`. For production dataset generation, `joblib.Parallel` or vectorized batch solves would help.

### 10. No velocity-dependent signal statistics test

Domain randomization tests verify shape preservation and determinism, but don't verify that perturbations change signal statistics in a measurable way (e.g., variance, SNR).

### 11. Quantile output selects last timestep only for evaluation

`train.py:112`: `pred_last = pred[:, :, -1]` — only the last timestep is evaluated. For a TCN producing seq-to-seq output, earlier timesteps might have better calibration. Evaluating across all timesteps (or a learned aggregation) would be more robust.

---

## Test Coverage Gaps

| Missing behavioral test | What to verify |
|---|---|
| **Quantile monotonicity under gradient** | After `loss.backward()`, verify raw head outputs (before sort) still produce a valid pinball gradient — currently untestable because sort is non-differentiable |
| **Margin is physically bounded** | `margin ∈ (-∞, 1)` for subcritical, `margin > 0` for supercritical — not checked anywhere |
| **`train()` end-to-end** | The actual `train()` function is never tested; only `evaluate_coverage` is |
| **Domain randomization changes statistics** | Noise increases variance, gain shifts mean, resampling changes frequency content — no tests verify these |
| **Coverage metric correctness** | Given a known quantile prediction vs known target, verify coverage fraction matches expected τ |
| **Subsonic velocity handling** | Solver raises `ValueError` for subsonic; transient.py catches it silently — no test verifies graceful degradation |
| **Eigenvalue sign convention** | Verify that stable eigenvalues (Re < 0) produce decaying signals and unstable ones produce growing signals — the modal superposition math depends on this |
| **All-eigenvalues-stable edge case** | If all eigenvalues are stable, margin never reaches 0 — verify the dataset handles this without NaN |

---

## Specific Code References

| File:Line | Issue |
|---|---|
| `transient.py:90-92` | **CRITICAL**: λ vs velocity mismatch in margin formula |
| `quantile_head.py:44` | **CRITICAL**: `torch.sort` kills gradient flow |
| `train.py:119,133` | **CRITICAL**: `NameError` if `len(quantiles) < 3` |
| `transient.py:81-83` | Per-sensor normalization destroys spatial amplitude info |
| `transient.py:93` | Silent exception swallowing on margin computation |
| `transient.py:131` | Silent exception swallowing on dataset generation |
| `tcn.py:32-33` | BN instability with small batches |
| `train.py:112` | Last-timestep-only evaluation discards seq2seq information |
