# Proposal 1: Multi-Fidelity Correction Field via Latent-Space Gaussian Process

## Project Charter

**Primary Claim:** A latent-space Gaussian Process correction model, trained on ≤100 HF samples, reduces FSDT-vs-FEM error meaningfully below the baseline ~4%, with calibrated uncertainty that grows appropriately outside the sampled design region.

**MVP:** Fixed decoder (no active learning yet) trained on one deliberately-designed HF batch; GP fit on top; compared against plain co-kriging baseline and INR baseline; evaluated via leave-p-out CV.

**Upgrade Path (explicit promotion criteria):**
1. **Decoder upgrade** (neural operator backbone): Only if held-out reconstruction error on decoder alone > 0.01, indicating spatial underfitting.
2. **Second HF batch** (active learning): Only if GP uncertainty is poorly calibrated (coverage < 80% or calibration error > 15%), indicating design-space coverage is insufficient.
3. **Deep GP** (hierarchical GP): Only if GP uncertainty is miscalibrated in specific regions (e.g., near mode veering), indicating non-stationary length scales.

**Do NOT pursue:** Architecture variants without a specific observed limitation motivating each one.

**Dependencies:** None required from other proposals; can share encoder/decoder infrastructure with Proposal 2 but isn't blocked by it.

**Stop Condition:** MVP result plus one upgrade attempt (if justified) is sufficient; don't chase additional architecture variants without a specific observed limitation motivating each one.

**Key Statistical Insight:** Each of the 50–100 HF runs gives thousands of correlated spatial points, but there are only 50–100 *independent* observations across the design space. The architecture must exploit dense spatial supervision while being honest that design-space coverage is what's actually scarce. This is why the decoder handles spatial complexity (thousands of points per run) and the GP handles design complexity (50–100 points across runs).

---

## Critical Solver Constraint

The FSDT solver uses **global basis functions** (Legendre/trigonometric) with **precomputed basis integrals**. The shear correction factor κ=5/6 is applied globally in `solver.py:assemble_stiffness()` via the ABDAs matrix. The ~4% error vs 3D FEM arises because this global factor doesn't capture spatial variations in shear deformation, especially near boundaries and in honeycomb cores with soft cores.

**Correction field approach:** Learn Δ(x,y) = κ_true(x,y)/κ_FSDT - 1, a spatially-varying correction that, when applied to the FSDT solution, recovers FEM accuracy. The correction field is defined on the same grid as the solver output (default 64×64).

**Boundary enforcement:** Δ(x,y) = b(x,y)·Δ̂(x,y), where b(x,y) = x(1-x)y(1-y) (on normalized domain [0,1]²). This guarantees correction vanishes at domain edges by construction.

## Existing Infrastructure

**Already implemented:**
- `data_generation.py` — LHS sampling, FSDT sweeps, NPZ output with provenance
- `sweep_demo.py` — Simple parameter sweep example
- `validate.py` — Physics validation against analytical solutions
- Full test suite (166 tests, 99% coverage)

**What we extend (not rewrite):**
- `data_generation.py` → add mode shape storage, correction field computation
- Existing `Laminate`, `Material`, `FSDTSolver` classes → no changes needed
- Existing test patterns → extend for multifidelity module

---

## Module Breakdown

### New Package: `src/multifidelity/`

| File | Contents |
|------|----------|
| `__init__.py` | Package exports |
| `correction_field.py` | `CorrectionField` dataclass, boundary envelope `b(x,y)`, correction extraction |
| `data_gen.py` | LF sweep generation, correction field computation (FSDT vs "truth") |
| `encoder.py` | `CorrectionEncoder`: correction fields → latent codes `z` |
| `decoder.py` | `CorrectionDecoder`: latent codes `z` → correction fields `Δ(x,y)` |
| `gp.py` | Gaussian Process over latent space: design params θ → posterior over z |
| `inr.py` | Implicit Neural Representation baseline: MLP decoder at (x,y) |
| `baselines.py` | Co-kriging, plain neural correction baselines |
| `losses.py` | Reconstruction loss, boundary loss, GP marginal likelihood |
| `train.py` | Training loops, checkpointing, data loaders |
| `eval.py` | Leave-p-out CV, pointwise error maps, coverage checks |
| `config.py` | Hyperparameter configs (dataclasses) |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `torch`, `gpytorch`/`botorch` to dependencies |

### Test Files

| File | Contents |
|------|----------|
| `tests/multifidelity/test_correction_field.py` | Correction field extraction, boundary envelope |
| `tests/multifidelity/test_data_gen.py` | LF sweep generation, dataset creation |
| `tests/multifidelity/test_encoder.py` | Encoder forward pass, output shape |
| `tests/multifidelity/test_decoder.py` | Decoder forward pass, output shape, reconstruction |
| `tests/multifidelity/test_gp.py` | GP fit, posterior prediction, uncertainty quantification |
| `tests/multifidelity/test_inr.py` | INR baseline forward pass, coordinate conditioning |
| `tests/multifidelity/test_baselines.py` | Baseline model training, evaluation |
| `tests/multifidelity/test_eval.py` | Leave-p-out CV, coverage metrics |

---

## Implementation Phases

### Phase 1: Data Generation Infrastructure

**Goal:** Generate LF sweeps from FSDT solver, compute correction fields for training.

**Existing Infrastructure:** `data_generation.py` already implements LHS sampling and FSDT sweeps. We extend this to:
1. Store full mode shapes (not just frequencies)
2. Compute correction fields between FSDT and synthetic HF
3. Package as zarr stores for ML training

**Key insight:** Without Abaqus data, we can't compute true HF-LF correction fields yet. Instead, we:
1. Generate LF data from FSDT solver across design parameter space (extend existing `data_generation.py`)
2. Create synthetic "HF" data by adding physics-informed perturbations to FSDT solutions (for development/testing)
3. Once Abaqus data arrives, replace synthetic HF with real HF data

**Implementation:**
1. Create `src/multifidelity/correction_field.py`:
   - `CorrectionField` dataclass: `grid_x`, `grid_y`, `values` (Δ(x,y)), `metadata`
   - `boundary_envelope(x, y, L1, L2)` → b(x,y) = x(1-x)y(1-y)
   - `apply_boundary_correction(delta_hat, x, y)` → delta = b(x,y) * delta_hat
   - `extract_correction_field(fsdt_field, fem_field)` → Δ = fem/fsdt - 1

2. Extend `src/multifidelity/data_gen.py` (build on existing `data_generation.py`):
   - `generate_lf_sweep(config, n_samples)` → zarr store with FSDT solutions (frequencies + mode shapes)
   - `compute_correction_fields(lf_data, hf_data)` → correction fields
   - `create_synthetic_hf(lf_data, perturbation_model)` → synthetic HF for development
   - Parameter ranges: face thickness, core thickness, cell angle, boundary stiffnesses
   - **Reuse existing LHS sampling from `data_generation.py:generate_lhs_samples()`**

3. Create `tests/multifidelity/test_correction_field.py` and `tests/multifidelity/test_data_gen.py`

**Design decisions:**
- Extend existing `data_generation.py` pattern (not rewrite)
- Store full mode shapes as `(n_modes, grid_ny, grid_nx)` per design point
- Correction fields computed on solver grid (64×64)
- Boundary envelope applied during training, not during data generation
- Use zarr for chunked storage (better than NPZ for large datasets)

**Milestone:** Zarr store with 1000 LF samples (frequencies + mode shapes), synthetic HF data with known correction structure, correction fields extractable and visualizable.

**Tests:**
- `test_correction_field.py`: boundary envelope vanishes at edges, correction field shape matches grid, extraction works for known perturbations
- `test_data_gen.py`: 10-sample generation completes, frequencies change with parameters, zarr readable

---

### Phase 2: Encoder/Decoder Architecture + Training Pipeline

**Goal:** Learn compact latent representation of correction fields, train autoencoder for reconstruction.

**Architecture:**
- Encoder: correction field Δ(x,y) ∈ R^{Gx×Gy} → latent z ∈ R^{d_z} (10-50 dims)
- Decoder: latent z → reconstructed correction field Δ̂(x,y) ∈ R^{Gx×Gy}
- Boundary condition enforced in decoder output: Δ̂ = b(x,y) · Δ̂_raw

**Implementation:**
1. Create `src/multifidelity/encoder.py`:
   - `CorrectionEncoder(grid_size, d_z, hidden_dims)` — Conv2d stack → FC → latent
   - Input: `(batch, 1, Gx, Gy)` correction field
   - Output: `(batch, d_z)` latent code
   - Optional: condition on design parameters θ (concatenate after convolution)

2. Create `src/multifidelity/decoder.py`:
   - `CorrectionDecoder(d_z, grid_size, hidden_dims)` — FC → reshape → ConvTranspose2d
   - Input: `(batch, d_z)` latent code
   - Output: `(batch, 1, Gx, Gy)` raw correction field
   - Apply boundary envelope: `Δ = b(x,y) · Δ_raw`

3. Create `src/multifidelity/losses.py`:
   - `reconstruction_loss(pred, target)` — MSE or L1
   - `boundary_loss(pred, x, y)` — penalty for non-zero at boundaries
   - `total_loss = recon + λ_boundary * boundary`

4. Create `src/multifidelity/train.py`:
   - `train_autoencoder(encoder, decoder, train_loader, config)` — training loop
   - `validate_autoencoder(encoder, decoder, val_loader)` — validation metrics
   - Checkpointing, early stopping, learning rate scheduling

5. Create `tests/multifidelity/test_encoder.py`, `tests/multifidelity/test_decoder.py`

**Design decisions:**
- `d_z = 32` (start conservative, increase if underfitting)
- Decoder applies boundary envelope after final conv, before loss
- Autoencoder pre-training: 100 epochs, batch size 32, Adam optimizer
- Reconstruction target: correction fields from Phase 1

**Milestone:** Autoencoder reconstructs training correction fields with MSE < 0.01, boundary violation < 1e-6.

**Tests:**
- `test_encoder.py`: forward pass shape, gradient flow, latent dimension matches config
- `test_decoder.py`: output shape, boundary envelope applied, values in reasonable range
- Autoencoder end-to-end: reconstruction loss decreases, validation loss tracks training

---

### Phase 3: GP Integration Over Latent Space

**Goal:** Fit GP mapping design parameters θ → latent codes z, enabling prediction at new design points with uncertainty.

**Architecture:**
- Input: design parameters θ ∈ R^{d_θ} (face thickness, core thickness, cell angle, etc.)
- Output: distribution over latent codes z ∈ R^{d_z}
- GP with separate kernel per latent dimension (or shared kernel)

**Implementation:**
1. Create `src/multifidelity/gp.py`:
   - `LatentGP(d_z, kernel_type)` — GPyTorch/BoTorch GP wrapper
   - `fit(theta_train, z_train)` — fit GP to training data
   - `predict(theta_test)` → `(z_mean, z_var)` — posterior mean and variance
   - `sample(theta_test, n_samples)` → `z_samples` — draw from posterior

2. Create `src/multifidelity/config.py`:
   - `GPConfig`: kernel type (RBF, Matern), lengthscale prior, noise variance
   - `TrainingConfig`: epochs, batch size, learning rate, λ_boundary
   - `DataConfig`: parameter ranges, n_samples, grid size

3. Create `tests/multifidelity/test_gp.py`

**Design decisions:**
- Use GPyTorch for GP fitting (native PyTorch integration)
- Separate GP per latent dimension initially (simpler, interpretable)
- Matern 5/2 kernel (smooth but not infinitely differentiable)
- Optimize GP hyperparameters via marginal likelihood
- Uncertainty grows outside training region (key property)

**Milestone:** GP predicts latent codes at held-out design points, uncertainty increases with distance from training data.

**Critical check (from proposals.md):** "Known limitation to check for, not just note: the decoder is deterministic — if a real correction has spatial structure never seen in the 50–100 training fields, the GP layer can't rescue it. Sanity check: held-out reconstruction error on the decoder alone before trusting the full pipeline."

**Tests:**
- `test_gp.py`: fit on 10 points, predict on 5, uncertainty reasonable, samples from prior/posterior match expectations
- End-to-end: encoder-decoder-GP pipeline reconstructs correction fields at new design points
- **Decoder alone:** held-out reconstruction error < 0.01 (if higher, decoder is bottleneck, not GP)

---

### Phase 4: INR Baseline Implementation

**Goal:** Implement coordinate-conditioned Implicit Neural Representation as baseline comparison.

**Architecture:**
- Input: coordinates (x,y) + design parameters θ + LF solution
- Output: correction value Δ(x,y) at query point
- MLP with positional encoding

**Implementation:**
1. Create `src/multifidelity/inr.py`:
   - `CorrectionINR(d_θ, hidden_dims, n_frequencies)` — MLP with positional encoding
   - `forward(x, y, theta)` → `delta` — query at specific point
   - `forward_grid(theta, grid)` → `delta_grid` — query entire grid
   - Positional encoding: `sin(2πf x), cos(2πf x)` for frequencies f = 1,...,n_freq

2. Create `tests/multifidelity/test_inr.py`

**Design decisions:**
- 4-6 layer MLP, 256 hidden units
- 10 frequencies for positional encoding
- Condition on design parameters via concatenation after positional encoding
- Same boundary enforcement as decoder: Δ = b(x,y) · Δ_raw

**Milestone:** INR baseline trained, achieves reasonable reconstruction, serves as comparison for latent+GP approach.

**Tests:**
- `test_inr.py`: forward pass shape, gradient flow, boundary enforcement
- Training: reconstruction loss decreases, overfits small dataset (sanity check)

---

### Phase 5: Validation Framework

**Goal:** Rigorous evaluation with leave-p-out CV, pointwise error maps, coverage checks.

**Key insight from proposals.md:** Evaluate via pointwise error maps (not just aggregate norms). A model that gets 4% RMSE but 20% error at boundaries is worse than a model with 5% RMSE but uniform error distribution.

**Metrics:**
- **Pointwise error:** |Δ_pred - Δ_true| at each spatial location → error maps
- **Aggregate error:** RMSE, MAE over all spatial points
- **Boundary error:** Error specifically at domain edges (should be ~0 due to enforcement)
- **Coverage:** % of held-out correction fields where true value falls within GP's 90% credible interval
- **Calibration:** GP uncertainty matches actual error distribution (coverage should be ~90%)
- **Spatial correlation:** Error should be smooth, not random (indicates model captures spatial structure)

**Implementation:**
1. Create `src/multifidelity/eval.py`:
   - `leave_p_out_cv(model, data, p, n_splits)` — cross-validation
   - `compute_error_maps(pred, target)` — spatial error distribution
   - `compute_coverage(pred_mean, pred_var, target, alpha=0.9)` — interval coverage
   - `plot_error_maps(errors, grid_x, grid_y)` — visualization
   - `plot_calibration(pred_var, errors)` — uncertainty calibration

2. Create `src/multifidelity/baselines.py`:
   - `CoKrigingBaseline`: standard co-kriging (no latent space)
   - `NeuralCorrectionBaseline`: direct MLP correction (no compression)
   - `INRBaseline`: wrapper around `CorrectionINR`

3. Create `tests/multifidelity/test_eval.py`, `tests/multifidelity/test_baselines.py`

**Design decisions:**
- Leave-2-out CV (sensible for 50-100 HF samples)
- 90% credible interval for coverage
- Error maps on 64×64 grid for visualization
- Compare against three baselines: co-kriging, neural correction, INR

**Milestone:** Full evaluation pipeline, error maps generated, coverage > 85%, comparison against baselines.

**Tests:**
- `test_eval.py`: CV splits non-overlapping, coverage computation correct, error maps have correct shape
- `test_baselines.py`: baselines train and predict, metrics computed correctly

---

## File Structure

```
src/multifidelity/
├── __init__.py
├── correction_field.py      # CorrectionField, boundary envelope, extraction
├── data_gen.py              # LF sweep generation, synthetic HF, dataset creation
├── encoder.py               # Correction fields → latent codes
├── decoder.py               # Latent codes → correction fields
├── gp.py                    # GP over latent space
├── inr.py                   # INR baseline
├── baselines.py             # Co-kriging, neural correction baselines
├── losses.py                # Reconstruction, boundary, total loss
├── train.py                 # Training loops, checkpointing
├── eval.py                  # Leave-p-out CV, error maps, coverage
└── config.py                # Hyperparameter configs (dataclasses)

tests/multifidelity/
├── test_correction_field.py
├── test_data_gen.py
├── test_encoder.py
├── test_decoder.py
├── test_gp.py
├── test_inr.py
├── test_baselines.py
└── test_eval.py

# Existing infrastructure (extended, not rewritten):
data_generation.py          # LHS sampling, FSDT sweeps (extend for mode shapes)
sweep_demo.py               # Simple parameter sweep example
validate.py                 # Physics validation
src/mechanics/              # Core solver (no changes needed)
roadmaps/proposal1_roadmap.md  # This file
```

---

## Baselines

### Baseline 1: Co-Kriging
- Standard multi-fidelity co-kriging without latent compression
- Input: design parameters θ
- Output: correction field Δ(x,y) directly (flattened)
- **Expected:** Good with enough data, but scales poorly with spatial dimension

### Baseline 2: Plain Neural Correction
- MLP mapping θ → flattened Δ(x,y)
- No spatial structure exploitation
- **Expected:** Fast inference, poor spatial coherence, no uncertainty

### Baseline 3: INR Baseline (implemented in Phase 4)
- Coordinate-conditioned MLP
- **Expected:** Good spatial reconstruction, no uncertainty, slower inference

### Comparison Protocol:
- All baselines use same train/val/test split
- Evaluate: RMSE, MAE, coverage, calibration
- Latent+GP should beat baselines on coverage and calibration

---

## Stopping Points

| Check | Criterion | Action if met |
|-------|-----------|---------------|
| **SP1** | Phase 1 data generation works, correction fields show spatial structure | Proceed to Phase 2 |
| **SP2** | Autoencoder reconstruction MSE < 0.01, boundary violation < 1e-6 | Proceed to Phase 3 |
| **SP3** | GP uncertainty increases outside training region, coverage > 80% | Proceed to Phase 5 |
| **SP4** | Leave-p-out CV coverage > 85%, calibration error < 10% | MVP complete, publish results |
| **SP5** | Latent+GP outperforms all three baselines on coverage | Strong result, consider decoder upgrade |
| **Fail: solver too slow** | >5s per sample at M=N=15 | Reduce M/N, use smaller grid, cache aggressively |
| **Fail: latent space collapses** | Decoder cannot reconstruct diverse correction fields | Increase d_z, add reconstruction regularization |
| **Fail: GP uncertainty poorly calibrated** | Coverage < 70% or calibration error > 20% | Check kernel choice, add more HF samples, consider deep GP |

---

## Dependencies to Add

```toml
# pyproject.toml additions
[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "gpytorch",
    "botorch",
]
```

**If using BoTorch (built on GPyTorch):**
- Automatic kernel selection
- Built-in acquisition functions for active learning
- PyTorch integration

**Alternative:** Use `scikit-learn` GaussianProcessRegressor for simpler implementation (no GPU, but fewer dependencies).

## Cross-Cutting Notes (from proposals.md)

**Shared infrastructure with Proposal 2:**
- Encoder/decoder architecture can be shared (both learn spatial field → latent → probabilistic model)
- Boundary-condition enforcement trick is identical
- Build once, reuse for both proposals

**Shared infrastructure with Proposal 3:**
- Active-learning/design-sampling machinery (choosing where to query expensive process next)
- Sensitivity analysis on LF solver to identify which parameters matter most for HF budget

**Proposal 1's Abaqus runs double as Proposal 2's held-out test set:**
- Sequence P1's HF data generation before P2's final validation
- P2's development doesn't need to wait on this

**Execution order:**
1. Proposal 1 first (Abaqus turnaround is critical path)
2. Proposal 3 in parallel (fast solver, no blocking dependency)
3. Proposal 2 follows (development on FSDT data)
4. Proposal 4 last (reuses P3's sampling infrastructure)

---

## Implementation Priority

**Critical path:** Phase 1 → Phase 2+3 (autoencoder + GP) → Phase 5 (validation)
**Parallelizable:** Phase 4 (INR baseline) can be implemented alongside Phase 2+3

**Estimated effort:**
- Phase 1: 0.5 day (extend existing `data_generation.py`, not rewrite)
- Phase 2: 2 days (standard autoencoder architecture)
- Phase 3: 2 days (GP fitting, hyperparameter tuning)
- Phase 4: 1 day (INR baseline is straightforward)
- Phase 5: 2 days (evaluation pipeline, baseline comparisons)
- **Total: ~7.5 days for MVP**

**Note:** Phase 1 is faster because `data_generation.py` already implements LHS sampling and FSDT sweeps. We just need to extend it to store full mode shapes and compute correction fields.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Solver too slow for large sweeps | Medium | High | Precompute basis integrals, cache LU factors, reduce M/N |
| GP training instability | Low | Medium | Start with RBF kernel, use standard GPyTorch, monitor length scales |
| Latent space not expressive enough | Medium | High | Increase d_z, add skip connections, try VAE instead of autoencoder |
| Boundary enforcement breaks gradients | Low | Medium | Apply envelope in forward pass with gradient-friendly implementation |
| No Abaqus data for validation | Certain | High | Use synthetic HF for development, real HF arrives later |
| Co-kriging baseline outperforms latent+GP | Low | Medium | Check if latent compression is actually helping, consider removing it |
| Decoder underfits correction diversity | Medium | High | Check held-out reconstruction error on decoder alone before trusting full pipeline |
| GP uncertainty poorly calibrated | Medium | Medium | Check kernel choice, add more HF samples, consider deep GP |

## Key Design Decisions (Make Early)

1. **Latent dimension d_z:** Start with 32, increase if underfitting. Too small → poor reconstruction, too large → GP overfits.
2. **GP kernel:** Matern 5/2 (smooth but not infinitely differentiable). RBF if correction fields are very smooth.
3. **Boundary envelope:** x(1-x)y(1-y) on normalized domain. Simple, works, easy to visualize.
4. **Synthetic HF model:** Physics-informed perturbation (boundary error + core softness + face thinness). Must be validated before training.
5. **Training split:** By HF run (not by pixel) to avoid leakage. 80/10/10 train/val/test.
6. **Evaluation metric:** Pointwise error maps + coverage. Not just RMSE.

---

## Synthetic HF Strategy (Until Abaqus Data Arrives)

Since we don't have Abaqus data yet, we need synthetic "HF" data for development:

**Perturbation model (physics-informed):**
```python
def synthetic_hf_correction(fsdt_solution, params):
    """Generate synthetic HF correction field mimicking FSDT error patterns."""
    # Error increases near boundaries (penalty spring approximation)
    boundary_error = boundary_envelope(x, y) * 0.02  # 2% near edges
    
    # Error increases with core softness (shear deformation underestimation)
    core_softness = 1.0 / (params['core_thickness'] * params['G_core'])
    core_error = core_softness * 0.01  # 1% for soft cores
    
    # Error increases with face thinness (bending-shear coupling)
    face_thinness = 1.0 / params['face_thickness']
    face_error = face_thinness * 0.005  # 0.5% for thin faces
    
    # Total correction: multiplicative on mode shapes
    correction = 1.0 + boundary_error + core_error + face_error
    return correction  # Applied to mode shapes: fem_field = fsdt_field * correction
```

**Validation of synthetic HF:**
- Correction field should be smooth (no artificial discontinuities)
- Magnitude should be ~2-6% (matching known FSDT error range)
- Spatial pattern should be physically plausible (larger near boundaries, larger for soft cores)

**Replace with real HF:** Once Abaqus runs are complete, replace synthetic HF with real data. The pipeline doesn't change — just the data source.

---

## What Can Be Tested RIGHT NOW (No Abaqus Dependency)

1. **LF sweep generation:** Run FSDT solver across design parameter space (use existing `data_generation.py`)
2. **Correction field extraction:** Compute Δ(x,y) between FSDT and synthetic HF
3. **Autoencoder training:** Learn latent representation of correction fields
4. **GP fitting:** Map design parameters to latent codes
5. **INR baseline:** Train coordinate-conditioned MLP
6. **Evaluation pipeline:** Leave-p-out CV, error maps, coverage checks
7. **Boundary enforcement:** Verify Δ(x,y) = b(x,y)·Δ̂(x,y) vanishes at edges
8. **Synthetic HF validation:** Test correction field extraction with known perturbations

**Everything except final validation against real Abaqus data can be developed and tested now.**

**Quick start:**
```bash
# Generate 100-sample LF sweep (existing infrastructure)
python data_generation.py

# Run existing validation to confirm solver works
python validate.py

# Run existing tests to confirm infrastructure
pytest tests/ -v
```

---

## Testing Strategy

**What tests at each phase:**
- Phase 1: Data generation, correction field extraction, boundary envelope
- Phase 2: Encoder/decoder forward pass, reconstruction loss, gradient flow
- Phase 3: GP fit, posterior prediction, uncertainty quantification
- Phase 4: INR forward pass, coordinate conditioning, boundary enforcement
- Phase 5: Leave-p-out CV, error maps, coverage, baseline comparisons

**How to validate without Abaqus data:**
1. Use synthetic HF with known correction structure
2. Verify correction fields are smooth and physically plausible
3. Check that boundary enforcement works (Δ=0 at edges)
4. Validate autoencoder reconstruction on training set
5. Check GP uncertainty increases outside training region
6. Compare against baselines (co-kriging, neural correction, INR)
7. **Decoder alone:** held-out reconstruction error < 0.01 (critical check before trusting full pipeline)

**Key insight:** Everything except final validation against real Abaqus data can be developed and tested now. The pipeline is designed to be data-source agnostic — just swap synthetic HF for real HF when it arrives.

**Test commands:**
```bash
# Run all multifidelity tests
pytest tests/multifidelity/ -v

# Run with coverage
pytest tests/multifidelity/ --cov=src/multifidelity -v

# Run specific test file
pytest tests/multifidelity/test_decoder.py -v
```