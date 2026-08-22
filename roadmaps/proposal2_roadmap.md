# Proposal 2: Probabilistic Inverse Damage Identification via Latent-Space Posterior Inference

## Project Charter

**Primary Claim:** A conditional invertible neural network (cINN) can map vibration measurements (natural frequencies + mode shapes) to a full posterior over spatially-varying damage fields in honeycomb sandwich plates, capturing the inherent ill-posedness of the inverse problem.

**MVP:** Posterior network that, given a measurement vector, produces a distribution over damage fields whose 90% credible interval contains the true damage at >85% of spatial locations.

**Upgrade Path:** Physics-guided diffusion model over the full damage image space, conditioned on measurements.

**Dependencies:** `torch`, `frEi` (or custom cINN), `zarr` (data), existing `mechanics` package.

**Stop Condition:** Posterior mean MSE < 2× the oracle lower bound (computed from held-out measurement noise), OR posterior calibration error < 5% on SBC.

---

## Critical Solver Constraint

The FSDT solver uses **global basis functions** (Legendre/trigonometric), not finite elements. Material properties enter as **global ABD matrices** via `laminate.ABD()`. This means:

- **Uniform damage** (whole-plate stiffness reduction): trivial, no solver changes needed. Create a new `Laminate` with degraded material.
- **Spatially-varying damage**: requires modifying the stiffness assembly to apply element-wise scaling to the precomputed basis integrals. The solver's `_expanded[(ii,jj)]` matrices (shape `MN_eff × MN_eff`) are the target — they must be multiplied by a spatial damage mask before entering the stiffness sum.

**Strategy:** Phase 1 uses uniform damage only (fast, no solver changes). Phase 2+ modifies `solver.py:assemble_stiffness()` to accept a spatial damage field `d(x,y) ∈ [0,1]` that scales local stiffness contributions.

---

## Module Breakdown

### New Package: `src/inverse_damage/`

| File | Contents |
|------|----------|
| `__init__.py` | Package exports |
| `damage_field.py` | `DamageField` dataclass, parametric damage generators (single patch, multi-patch, random), spatial discretization |
| `data_gen.py` | Synthetic dataset generation: damage field → modified laminate → solver → (measurements, damage_field) pairs |
| `encoder.py` | `MeasurementEncoder`: frequencies + flattened mode shapes → conditioning vector `c` |
| `decoder.py` | `DamageDecoder`: latent `z` (+ optional `c`) → damage field `d(x,y)` on grid |
| `posterior.py` | `PosteriorNetwork`: cINN wrapper mapping `z ↔ x` conditioned on `c`, with prior `p(z)` |
| `losses.py` | NLL, calibration loss, SBC metrics, posterior coverage |
| `train.py` | Training loop, data loaders, checkpointing |
| `eval.py` | Evaluation: posterior predictive, calibration plots, SBC |
| `baselines.py` | Direct regression baseline, Bayesian regression baseline |

### Modified Files

| File | Change |
|------|--------|
| `src/mechanics/solver.py` | Add `_assemble_stiffness_damage(damage_mask)` method that scales `_expanded[(ii,jj)]` by local damage values before R-matrix multiplication. Does not change existing `assemble_stiffness()` behavior. |
| `pyproject.toml` | Add `torch`, `zarr` to dependencies |

### Test Files

| File | Contents |
|------|----------|
| `tests/inverse_damage/test_damage_field.py` | Damage field generation, discretization, masking |
| `tests/inverse_damage/test_data_gen.py` | Synthetic data pipeline, solver integration |
| `tests/inverse_damage/test_encoder.py` | Encoder forward pass, output shape |
| `tests/inverse_damage/test_decoder.py` | Decoder forward pass, output shape, reconstruction |
| `tests/inverse_damage/test_posterior.py` | cINN forward/inverse, sampling, prior match |
| `tests/inverse_damage/test_baselines.py` | Baseline model training, evaluation |
| `tests/inverse_damage/test_sbc.py` | Simulation-based calibration |

---

## Implementation Phases

### Phase 1: Synthetic Damage Data Generation

**Goal:** Generate `(measurement, damage_field)` pairs using only the existing FSDT solver.

**Damage model (uniform):** Whole-plate stiffness reduction. `d ∈ [0,1]` where `d=1` is pristine and `d=0` is fully damaged.

**Implementation:**
1. Create `src/inverse_damage/damage_field.py`:
   - `DamageField` dataclass with `grid`, `values` array, metadata
   - `sample_uniform_damage(rng)` → scalar `d ∈ [d_min, 1.0]`
   - `sample_single_patch(rng, plate_dims)` → `(d, patch_center, patch_size)`
   - `sample_multi_patch(rng, plate_dims, max_patches)` → list of patches
   - `damage_to_laminate(base_laminate, damage_field)` → new `Laminate` with scaled material properties

2. Create `src/inverse_damage/data_gen.py`:
   - `generate_sample(base_laminate, solver_factory, damage_sampler, rng)` → `(frequencies, mode_shapes, damage_field)`
   - `generate_dataset(config, n_samples, output_path)` → writes zarr store
   - Config: solver params (M, N, grid), damage range, n_modes, noise level

3. Create `tests/inverse_damage/test_damage_field.py` and `tests/inverse_damage/test_data_gen.py`

**Key design:** `damage_to_laminate` creates a new `Laminate` where each material's `E1, E2, G12, G13, G23` are multiplied by `d`. This is physically approximate (real damage is more complex) but sufficient for MVP validation.

**Solver cost at M=N=15:** ~100ms per solve. 10k samples ≈ 17 min. Acceptable for offline generation.

**Milestone:** Zarr store with 1000 samples, verified frequency shifts correlate with damage level.

**Tests:**
- `test_damage_field.py`: damage values in [0,1], laminate scaling preserves symmetry, ABD matrix scales correctly
- `test_data_gen.py`: 10-sample generation completes, frequencies decrease with damage, mode shapes change, output zarr readable

---

### Phase 2: Measurement Encoder

**Goal:** Map raw measurements (frequencies + mode shapes) to a fixed-length conditioning vector `c`.

**Architecture:**
- Input: `freq ∈ R^n_modes` + `mode_shapes ∈ R^{n_modes × Gx × Gy}` (flattened or pooled)
- Frequencies: linear layer → embedding
- Mode shapes: 2D CNN or global average pooling → embedding
- Concatenate → MLP → conditioning vector `c ∈ R^{d_c}` (default `d_c = 128`)

**Implementation:**
1. Create `src/inverse_damage/encoder.py`:
   - `MeasurementEncoder(n_modes, grid_size, d_c, mode)` — mode ∈ {"mlp", "cnn"}
   - CNN mode: Conv2d stack on each mode shape, then concat all mode embeddings
   - MLP mode: flatten all mode shapes, concat with frequencies, feed through MLP

2. Create `tests/inverse_damage/test_encoder.py`

**Design decisions:**
- Mode shapes are normalized per-mode (L2 norm = 1) to remove amplitude ambiguity
- Frequencies are logged (log(freq)) to compress dynamic range
- Encoder must handle variable numbers of modes (pad to max, mask)

**Milestone:** Encoder produces `c ∈ R^{128}` from `(10, 64, 64)` input, gradient flows through.

**Tests:**
- Forward pass shape: `(batch, n_modes, Gx, Gy)` → `(batch, d_c)`
- Gradient flow confirmed via `torch.autograd.gradcheck`
- Output is normalized (unit norm per sample)

---

### Phase 3: Decoder for Damage Field Reconstruction

**Goal:** Map latent code `z` to a damage field `d(x,y)` on the output grid.

**Architecture:**
- Input: `z ∈ R^{d_z}` (default `d_z = 64`)
- Optional: conditioning vector `c ∈ R^{d_c}` (concatenated or FiLM-conditioned)
- Transposed convolution network (ConvTranspose2d) → damage field `∈ R^{Gx × Gy}`
- Output sigmoid activation → values in [0,1]

**Implementation:**
1. Create `src/inverse_damage/decoder.py`:
   - `DamageDecoder(d_z, d_c, grid_size, hidden_dims)` — FC → reshape → ConvTranspose2d stack
   - Input: `z` (and optionally `c`)
   - Output: damage field on grid, clamped to [0,1]

2. Create `tests/inverse_damage/test_decoder.py`

**Design decisions:**
- Decoder is deterministic: `z → d`. Stochasticity comes from the posterior.
- Output grid matches solver grid (default 64×64)
- Skip connections from encoder to decoder (U-Net style) for conditional reconstruction

**Milestone:** Decoder reconstructs a single damage field from a known latent code (autoencoder sanity check).

**Tests:**
- Forward pass shape: `(batch, d_z)` → `(batch, 1, Gx, Gy)`
- Output values in [0,1]
- Reconstruction loss < 0.01 on training set after 50 epochs (autoencoder test)

---

### Phase 4: Posterior Network (cINN)

**Goal:** Learn the conditional distribution `p(z | measurement)` over latent codes, which the decoder maps to damage fields.

**Architecture (conditional INN):**
- Forward: `(z, measurement) → (z', h)` where `h` is the latent
- Inverse: `(z', h) → (z, measurement)` — used for sampling
- Conditioned on `c = encoder(measurement)`
- Prior: `p(z) = N(0, I)`
- Invertible coupling layers with ActNorm + permutations

**Implementation:**
1. Create `src/inverse_damage/posterior.py`:
   - `PosteriorNetwork(d_z, d_c, n_coupling_layers, hidden_dims)`
   - `forward(z, c)` → `(z_out, log_det)` — for training (negative log-likelihood)
   - `sample(c, n_samples)` → `z_samples` — for inference (draw from posterior)
   - `log_prob(z, c)` → `log_p` — for evaluation

2. Training loss: `-log p(z_sample)` where `z_sample` maps back to training damage via encoder
   - Or: `NLL = -log_prior(z) + log_det_jacobian`
   - Conditioned on `c = encoder(measurement)`

3. Create `tests/inverse_damage/test_posterior.py`

**Training procedure:**
1. Pre-train encoder-decoder as autoencoder (Phase 2+3 jointly)
2. Freeze encoder, train cINN on `(encoder(damage_measurement), latent_code)` pairs
3. Or: joint training with NLL loss

**Design decisions:**
- Use `frEi` library if available, otherwise implement minimal coupling layers
- `d_z = 64` (latent dimension for damage field)
- `n_coupling_layers = 8` (start with 4, increase if underfitting)
- ActNorm for activation normalization (handles non-stationary distributions)

**Milestone:** Posterior samples cover training damage fields; 90% credible interval has >80% coverage on validation set.

**Tests:**
- `sample(c)` returns `(n_samples, d_z)` tensor
- `log_prob(z, c)` returns `(batch,)` tensor
- Prior samples `sample(c_fixed)` have approximately `N(0,I)` marginal distributions
- Forward-inverse consistency: `z_inv, _ = posterior.inverse(*posterior.forward(z, c))` → `z_inv ≈ z`

---

### Phase 5: Validation Framework

**Goal:** Rigorous evaluation of posterior quality.

**Simulation-Based Calibration (SBC):**
1. For each held-out sample: draw `z_true ~ p(z)`, generate `d_true = decoder(z_true)`, compute `meas = forward_solve(d_true)`
2. Draw `n_samples` from `p(z | meas)`
3. Check: rank of `z_true` among samples should be uniform on `[0, n_samples]`
4. Calibration error: `|rankCDF - uniformCDF|`

**Held-out testing:**
- 80/10/10 train/val/test split
- Metrics: MSE of posterior mean, CRPS, interval coverage, calibration error
- Oracle lower bound: noise-free measurement → reconstruction error

**Implementation:**
1. Create `src/inverse_damage/eval.py`:
   - `compute_sbc(posterior, decoder, solver, n_samples_per_test=100)`
   - `compute_metrics(predictions, targets)` — MSE, CRPS, coverage
   - `plot_calibration(sbc_ranks)` — rank histogram, CDF comparison

2. Create `src/inverse_damage/losses.py`:
   - `nll_loss(posterior, encoder, decoder, measurements, damage_fields)`
   - `calibration_loss(sbc_ranks)` — penalty for non-uniform ranks

3. Create `tests/inverse_damage/test_sbc.py`

**Milestone:** SBC calibration error < 5%, 90% coverage > 85%, posterior mean MSE < 2× oracle bound.

**Tests:**
- SBC on 100 samples produces rank histogram
- Coverage computation matches expected values for uniform posterior
- CRPS is non-negative and decreases with better models

---

## Baselines

### Baseline 1: Direct Regression
- `encoder → MLP → flattened damage field → reshape`
- Deterministic point estimate
- Loss: MSE between predicted and true damage field
- **Expected:** Good on average, poor on rare damage patterns, no uncertainty

### Baseline 2: Bayesian Regression
- Same architecture as Baseline 1, but output is `(mean, log_variance)` per pixel
- Loss: Gaussian NLL
- **Expected:** Better uncertainty, but assumes independent pixel noise (wrong)

### Baseline 3: Simple Classifier
- Discretize damage into `k` levels (e.g., 5: none/mild/moderate/severe/critical)
- Per-pixel classification
- **Expected:** Coarse but robust, no continuous damage estimation

### Comparison protocol:
- All baselines use same train/val/test split
- Evaluate: MSE, CRPS (where applicable), 90% interval coverage
- cINN posterior should beat baselines on coverage and CRPS

---

## File Structure

```
src/inverse_damage/
├── __init__.py
├── damage_field.py          # DamageField, generators, laminate modification
├── data_gen.py              # Synthetic data pipeline
├── encoder.py               # Measurement → conditioning vector
├── decoder.py               # Latent → damage field
├── posterior.py             # cINN posterior network
├── losses.py                # NLL, calibration, SBC metrics
├── train.py                 # Training loops, checkpointing
├── eval.py                  # Evaluation, plots, SBC
├── baselines.py             # Baseline models
└── config.py                # Hyperparameter configs (dataclasses)

tests/inverse_damage/
├── test_damage_field.py
├── test_data_gen.py
├── test_encoder.py
├── test_decoder.py
├── test_posterior.py
├── test_baselines.py
└── test_sbc.py

roadmaps/
└── proposal2_roadmap.md     # This file
```

---

## Stopping Points

| Check | Criterion | Action if met |
|-------|-----------|---------------|
| **SP1** | Phase 1 data generation works, frequency shifts are monotonic in damage | Proceed to Phase 2 |
| **SP2** | Autoencoder (encoder+decoder) reconstruction MSE < 0.01 on validation | Proceed to Phase 4 |
| **SP3** | Posterior 90% coverage > 80% on validation | Proceed to Phase 5 |
| **SP4** | SBC calibration error < 10% | MVP complete, publish results |
| **SP5** | Posterior mean MSE < 2× oracle bound | Strong result, consider diffusion upgrade |
| **Fail: solver too slow** | >5s per sample at M=N=15 | Reduce M,N, use uniform damage only, or cache aggressively |
| **Fail: posterior collapses** | Coverage < 50% after 100 epochs | Increase coupling layers, check encoder capacity, add data augmentation |
| **Fail: ill-posedness too severe** | MSE plateau > 0.1 | Add more measurement modes (aeroelastic), increase n_modes, accept wider posteriors |

---

## Dependencies to Add

```toml
# pyproject.toml additions
[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "zarr",
]
```

**If using frEi (Facebook's cINN library):**
```toml
ml = [
    "torch>=2.0",
    "frEi",
    "zarr",
]
```

**Otherwise:** Implement minimal coupling layers (~100 lines) in `posterior.py`. Avoids extra dependency.

---

## Implementation Priority

**Critical path:** Phase 1 → Phase 2+3 (autoencoder) → Phase 4 (cINN) → Phase 5 (validation)

**Parallelizable:** Baselines can be implemented alongside Phase 2+3. Evaluation framework alongside Phase 4.

**Estimated effort:**
- Phase 1: 1 day (data gen is mechanical)
- Phase 2+3: 2 days (standard architectures)
- Phase 4: 3 days (cINN is the novel part, debugging training)
- Phase 5: 1 day (evaluation is mechanical)
- **Total: ~7 days for MVP**

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Solver too slow for 10k samples | Medium | High | Precompute basis integrals, cache LU factors, reduce M/N |
| cINN training instability | High | High | Start with 4 coupling layers, use AdamW, gradient clipping, monitor log_det |
| Posterior collapses to point estimate | Medium | High | Add noise to training data, use multiple latent dimensions, check decoder capacity |
| Mode shapes insufficient to distinguish damage patterns | Low | Medium | Add aeroelastic measurements (flutter boundary), increase n_modes |
| FrEi library incompatible with PyTorch 2.x | Low | Medium | Implement minimal cINN from scratch (~150 LOC) |

## Amendment 2026-08-22: CVAE substitution

As implemented in `src/mechanics/p2_inverse_damage/`, three roadmap decisions were revised:

1. **Phase-4 cINN → diagonal-Gaussian CVAE head.** The conditional invertible neural network
   (and the frEi library option) was never adopted. Phase 4 instead uses a conditional VAE
   posterior (`ConditionalPosterior`): a diagonal-Gaussian head mapping the conditioning vector
   `c` to `(mu, log_var)` with reparameterized sampling against an N(0, I) prior, trained by ELBO
   with a free-bits KL floor. Invertibility is no longer required anywhere in the pipeline.
2. **SP4 redefined in CVAE terms.** The original SP4 ("SBC calibration error < 10%") assumed
   log_prob evaluation over an invertible posterior. SP4 is now measured with **sample-based SBC**:
   rank histograms computed over posterior *samples* (not analytic log-prob/rank machinery),
   together with the existing **90% interval coverage > 80%** gate (SP3 criterion retained as part
   of calibration sign-off).
3. **zarr → .npz persistence.** Datasets are stored as NumPy `.npz` archives with an embedded
   provenance JSON record, matching repo norms; the `zarr` dependency is dropped from the proposed
   `ml` extra.
