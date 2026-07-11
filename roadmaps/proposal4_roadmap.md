# Proposal 4: Continuous Stability-Margin Estimation from Transient Response

## Implementation Roadmap

---

## 1. Project Charter

**Primary claim:** A continuous margin-estimation model, trained on simulated transient response with domain randomization, gives earlier and better-calibrated warning of approaching instability than binary-classifier baselines, and degrades gracefully under simulation-reality mismatch.

**MVP:** TCN backbone regressing normalized margin with quantile head, trained on FSDT-generated clips with domain randomization, validated via coverage tests and lead-time precision/recall.

**Upgrade path:** Intrinsic-stability-coordinate target (eigenvalue real part / damping ratio) — only if raw control-parameter target shows poor cross-design transfer.

**Dependencies:**
- Shares design-sampling infrastructure with Proposal 3 (same parameter sweep, different output extraction)
- Not blocked by Proposals 1 or 2

**Stop condition:** MVP result (coverage tests + lead-time metrics vs baselines) is a complete deliverable. No architecture variant sweeps without a specific observed limitation.

---

## 2. Key Solver Insight

`solve_complex_modal` returns complex eigenvalues $\lambda_k = \sigma_k + i\omega_k$ where:
- $\sigma_k$ (real part) = exponential growth/decay rate per mode
- $\omega_k$ (imaginary part) = damped natural frequency
- Flutter occurs when any $\sigma_k \geq 0$

This gives us **two viable regression targets:**

| Target | Formula | Pros | Cons |
|--------|---------|------|------|
| Normalized margin (raw) | $(u_{crit} - u(t)) / u_{crit}$ | Simple, interpretable | Design-specific, poor cross-design transfer |
| Eigenvalue real-part margin | $-\sigma_{max}(t) / \omega_{ref}$ | Intrinsic, transferable across designs | Requires eigenvalue tracking per timestep |

The eigenvalue target is the better long-term choice. The raw-parameter target is the MVP path since it requires no eigenvalue tracking infrastructure.

---

## 3. File Structure

All new files under `src/mechanics/`:

```
src/mechanics/
├── transient.py          # Phase 1: Modal superposition transient response generator
├── domain_random.py      # Phase 2: Domain randomization for sim-to-real transfer
├── tcn.py                # Phase 3: Temporal Convolutional Network backbone
├── quantile_head.py      # Phase 4: Quantile/probabilistic regression head
├── data_gen.py           # Phase 5: Synthetic dataset generation pipeline
├── train.py              # Phase 5: Training loop
├── baselines.py          # Phase 6: Baseline models (growth-rate, binary, LSTM)
├── evaluate.py           # Phase 6: Validation framework (coverage, lead-time metrics)
└── margin_dataset.py     # Phase 5: Dataset class for loading/training
```

Test files under `tests/`:
```
tests/
├── test_transient.py
├── test_domain_random.py
├── test_tcn.py
├── test_quantile_head.py
├── test_data_gen.py
├── test_train.py
└── test_evaluate.py
```

---

## 4. Implementation Phases

### Phase 1: Transient Response Generator

**Goal:** Generate time-domain transients from frequency-domain solver via modal superposition.

**Approach — Modal Superposition:**
1. Call `solve_complex_modal(velocity)` to get eigenvalues $\{\lambda_k\}$ and eigenvectors $\{\phi_k\}$
2. Project initial conditions onto modes: $q_k(0) = \phi_k^T M q(0)$
3. Each modal coordinate evolves as: $q_k(t) = q_k(0) e^{\lambda_k t}$
4. Reconstruct physical DOF: $w(x,y,t) = \sum_k \text{Re}[q_k(t) \phi_k(x,y)]$

**Files:**
- `transient.py`: `TransientGenerator` class
  - `generate(solver, velocity, t_span, n_points, initial_condition) -> TransientResult`
  - `compute_margin(transient_result, u_crit) -> np.ndarray` (margin over time)
  - Support arbitrary initial conditions (random perturbation, impulse, etc.)

**Tests:**
- `test_transient.py`: Eigenvalue reconstruction, superposition linearity, stability detection (growing vs decaying modes), margin computation at $t=0$

**Milestone:** Can generate margin time-series for a single velocity near flutter boundary.

---

### Phase 2: Domain Randomization Infrastructure

**Goal:** Wrap the transient generator to inject realistic sim-to-real perturbations during data generation.

**Randomizations (per sample):**
| Parameter | Range | Rationale |
|-----------|-------|-----------|
| Sensor noise $\sigma_n$ | [0, 5%] of peak amplitude | Real sensor noise |
| Structural damping ratio $\zeta$ | [0, 2%] | Manufacturing/material variability |
| Calibration error (gain) | [0.9, 1.1] | Sensor calibration drift |
| Sampling frequency | [500, 2000] Hz | Variable data acquisition |
| Initial condition | Random impulse location/amplitude | Unknown operational disturbances |
| Air density $\rho$ | [0.9, 1.1] × nominal | Atmospheric variation |

**Files:**
- `domain_random.py`: `DomainRandomizer` class
  - `randomize_params(rng) -> PerturbedParams`
  - `apply_perturbations(signal, params) -> np.ndarray`

**Tests:**
- `test_domain_random.py`: Range validation, parameter independence, deterministic reproducibility with seed

**Milestone:** Can generate a batch of 100 randomized transient clips with known perturbation metadata.

---

### Phase 3: TCN Backbone

**Goal:** Implement a Temporal Convolutional Network for streaming-friendly sequence-to-sequence regression.

**Architecture:**
```
Input: (batch, n_channels, seq_len)
  n_channels: [w(x_i, y_j, t)] — selected sensor locations (4-16 spatial points)

TCN Block × 4:
  - CausalConv1d(in, hidden, kernel=3, dilation=2^i)
  - BatchNorm
  - ReLU
  - Dropout(0.1)

Output: (batch, n_quantiles, seq_len)  — or (batch, 1, seq_len) for point estimate
```

**Design decisions:**
- Causal convolutions only (no future leakage)
- Dilation exponential growth: effective receptive field = $2^{L}-1$ with $L$ layers
- Lightweight: <50K parameters (streaming-friendly)

**Files:**
- `tcn.py`: `TCNBackbone` class
  - `forward(x) -> features`
  - `predict(x) -> (mean, quantiles)` when combined with quantile head

**Tests:**
- `test_tcn.py`: Output shape, causal constraint (output at $t$ doesn't depend on input at $t+1$), gradient flow, parameter count

**Milestone:** TCN can fit a simple synthetic sine+exponential signal end-to-end.

---

### Phase 4: Quantile/Probabilistic Head

**Goal:** Predict calibrated uncertainty intervals, not just point estimates.

**Approach:** Quantile regression head — predict $\tau$-quantiles of the margin distribution directly.

**Quantiles:** [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

**Loss:** Pinball loss: $L_\tau(y, \hat{q}) = \max(\tau(y - \hat{q}), (\tau-1)(y - \hat{q}))$

**Files:**
- `quantile_head.py`: `QuantileHead` class
  - `forward(features) -> (median, quantiles)` where quantiles shape = (batch, n_quantiles, seq_len)

**Tests:**
- `test_quantile_head.py`: Output shape, quantile ordering constraint ($q_{0.1} \leq q_{0.5} \leq q_{0.9}$), pinball loss gradient

**Milestone:** Combined TCN+quantile head can fit synthetic data with meaningful uncertainty bands.

---

### Phase 5: Training Pipeline with Synthetic Data

**Goal:** End-to-end data generation, training, and checkpointing.

**Data generation strategy:**
1. Sweep design parameter space (reuse Proposal 3's sampling infrastructure)
2. For each design, run transient generator at 5-10 velocities spanning sub-critical to super-critical
3. Apply domain randomization to each clip
4. Label: margin$(t) = (u_{crit} - u(t))/u_{crit}$ where $u(t)$ is the characteristic amplitude (e.g., max $|w(x,y,t)|$ over the plate)
5. Store: `(sensor_signals, margin_timeseries, metadata)` as a single sample

**Files:**
- `data_gen.py`: `MarginDataGenerator` class
  - `generate_dataset(config, n_samples) -> Dataset`
  - Integrates `TransientGenerator` + `DomainRandomizer`
- `margin_dataset.py`: `MarginDataset(torch.utils.data.Dataset)`
  - `__getitem__` returns `(sensor_signals, margin_target, metadata)`
- `train.py`: Training loop
  - `train(config) -> trained_model`
  - Quantile loss, early stopping, LR scheduling

**Tests:**
- `test_data_gen.py`: Dataset size, margin range [0, 1], NaN/Inf absence, metadata integrity
- `test_train.py`: One-epoch smoke test (loss decreases), checkpoint save/load

**Milestone:** Trained model checkpoint on 50K+ samples, training loss converged.

---

### Phase 6: Validation Framework

**Goal:** Evaluate against baselines with interpretable metrics.

**Baselines:**
1. **Amplitude-growth-rate threshold:** $\dot{A}/A > \theta$ triggers warning. Simple, but binary and requires manual threshold tuning.
2. **Binary classifier:** LSTM or TCN trained on binary safe/unsafe labels. The comparison point from the original proposal.
3. **Simple LSTM:** Point-estimate regressor without quantile head. Tests whether quantile head adds value.
4. **No-ML heuristic:** Linear extrapolation from first few timesteps of observed amplitude trend.

**Metrics:**
| Metric | Description |
|--------|-------------|
| Coverage | Fraction of true margins within predicted $\tau$-quantile intervals |
| Interval width | Mean width of 90% prediction interval |
| MAE | Mean absolute error of median prediction |
| Lead time precision | At lead time $L$, fraction of warnings that precede actual instability by $\geq L$ |
| Lead time recall | At lead time $L$, fraction of actual instabilities preceded by warning by $\geq L$ |
| False alarm rate | Fraction of stable runs with warning triggered |
| AUC (binary) | For threshold-based decision layer applied to continuous margin |

**Files:**
- `baselines.py`: Baseline model implementations
  - `GrowthRateBaseline`, `BinaryClassifierBaseline`, `LSTMBaseline`
- `evaluate.py`: Evaluation pipeline
  - `evaluate(model, test_set) -> EvaluationReport`
  - `compute_coverage(pred, target, quantiles)`
  - `compute_lead_time(pred, target, velocities, u_crit, lead_times)`
  - `generate_report(results) -> markdown`

**Tests:**
- `test_evaluate.py`: Coverage at nominal level (0.05 quantile should contain ~5% of points), lead-time computation correctness, report generation

**Milestone:** Full comparison table: TCN+quantile vs baselines on coverage, lead-time, and false-alarm metrics.

---

## 5. Testing Strategy

| Phase | Test level | What |
|-------|-----------|------|
| 1 | Unit | Eigenvalue reconstruction, superposition linearity |
| 2 | Unit | Randomization range checks, reproducibility |
| 3 | Unit + Integration | TCN shape/causality, end-to-end gradient flow |
| 4 | Unit | Quantile ordering, pinball loss |
| 5 | Integration | Dataset integrity, training convergence |
| 6 | System | Full evaluation pipeline, baseline comparison |

**Always run:** `rtk pytest tests/` after each phase.

---

## 6. Stopping Points

| Condition | Action |
|-----------|--------|
| Phase 1 fails: modal superposition doesn't reproduce stable/decaying transients | Debug eigenvector projection; may need Newmark integrator instead |
| Phase 3 TCN can't fit simple synthetic signal | Check causality constraint, increase receptive field, try DilatedCausalConv |
| Coverage tests fail badly (claimed 90% interval covers <70%) | Quantile head may need architecture change, or training data is biased |
| Baselines beat TCN+quantile on lead-time metrics | Stop, investigate whether margin regression is actually harder than classification for this problem |
| No improvement over amplitude-growth-rate baseline | Margin regression target may be the wrong framing; consider eigenvalue-based target |

**Hard stop:** If Phase 1-3 complete and baseline comparison shows no clear advantage over growth-rate threshold, the continuous-margin framing may not be the right approach for this physical system. Report honestly and pivot to eigenvalue-based target.

---

## 7. Dependencies and Integrations

**New dependencies to add to `pyproject.toml`:**
```
"torch>=2.0",
```

**Shared infrastructure with Proposal 3:**
- Parameter sweep sampling (same `ExperimentConfig` + `SweepConfig` structure)
- Flutter boundary finding (reuse `FSDTSolver.find_flutter_boundary`)
- Design parameter catalog (same material/boundary configurations)

**Integration point (future):**
- Proposal 4's trained model becomes the online monitoring component paired with Proposal 3's offline design-time reliability surrogate

---

## 8. Timeline Estimate

| Phase | Effort | Depends on |
|-------|--------|------------|
| Phase 1 | 2-3 days | Nothing (uses existing solver) |
| Phase 2 | 1 day | Phase 1 |
| Phase 3 | 2 days | Nothing (can develop in parallel with Phase 2) |
| Phase 4 | 1 day | Phase 3 |
| Phase 5 | 2-3 days | Phases 1-4 |
| Phase 6 | 2-3 days | Phase 5 |
| **Total** | **10-13 days** | |

---

## 9. File Creation Order

1. `src/mechanics/transient.py` + `tests/test_transient.py`
2. `src/mechanics/domain_random.py` + `tests/test_domain_random.py`
3. `src/mechanics/tcn.py` + `tests/test_tcn.py`
4. `src/mechanics/quantile_head.py` + `tests/test_quantile_head.py`
5. `src/mechanics/margin_dataset.py` + `src/mechanics/data_gen.py` + tests
6. `src/mechanics/train.py` + `tests/test_train.py`
7. `src/mechanics/baselines.py` + `src/mechanics/evaluate.py` + tests
8. Update `pyproject.toml` with `torch` dependency
