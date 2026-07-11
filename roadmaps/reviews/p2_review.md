# P2 Review: Probabilistic Inverse Damage Identification

## Summary

**Nothing is implemented.** The `src/inverse_damage/` package contains only a one-line `__init__.py` docstring and a `config.py` with 6 dataclasses. No encoder, decoder, posterior network, data generator, training loop, tests, or evaluation code exists. The roadmap (`proposal2_roadmap.md`) is well-structured but has zero executable code behind it. This review covers the existing config and the roadmap's design decisions that will affect implementation.

## Critical Issues

### 1. No implementation exists — only config and docstring

- `src/inverse_damage/__init__.py:1` — one docstring, no exports
- `src/inverse_damage/config.py` — 6 dataclasses, no logic
- No `damage_field.py`, `data_gen.py`, `encoder.py`, `decoder.py`, `posterior.py`, `train.py`, `eval.py`, `losses.py`, `baselines.py`
- No tests in `tests/inverse_damage/`

**Impact:** Nothing to review for correctness, API integration, or test quality. The review below evaluates the roadmap's design choices and config against the codebase.

### 2. Config `DecoderConfig` doesn't match roadmap design

`config.py:37-41`: `DecoderConfig` has `hidden_dims: [256, 512, 1024]` with no `d_c` (conditioning dimension). The roadmap (`proposal2_roadmap.md:139`) says decoder takes `z ∈ R^{d_z}` plus optional `c ∈ R^{d_c}`. The config is missing the conditioning path.

### 3. Config `EncoderConfig` uses `mode: str = "mlp"` — but roadmap expects CNN

`config.py:33`: `mode: str = "mlp"` — the roadmap (`proposal2_roadmap.md:106-108`) describes both MLP and CNN modes for mode shapes. CNN mode (Conv2d stack) is more appropriate for spatial mode shape data. Config defaults to the weaker option.

### 4. Solver integration not stubbed

The roadmap (`proposal2_roadmap.md:49`) says `solver.py` needs `_assemble_stiffness_damage(damage_mask)`. `solver.py` has no such method. No stub or interface contract exists. This is the critical physics integration point — without it, spatially-varying damage is impossible.

### 5. Config `SolverConfig` grid mismatch

`config.py:23`: `grid: tuple[int, int] = (64, 64)` — but `DamageGenConfig.grid: tuple[int, int] = (16, 16)`. The damage field grid (16×16) is coarser than the solver grid (64×64). This is intentional (damage is lower resolution) but the config doesn't document the upsampling strategy. The decoder output must be upsampled to match solver resolution.

## Improvements

### 1. Config `PosteriorConfig` missing ELBO weighting

`config.py:44-51`: ELBO has two terms (reconstruction + KL). No `kl_weight` or `beta` parameter for KL annealing. Without this, posterior collapse is undetectable and unfixable at config level.

### 2. Config `DataConfig` missing mode shape output path

`config.py:56-62`: `output_path` is for zarr store but doesn't specify whether mode shapes are stored flattened or per-grid. Mode shapes at (n_modes, 64, 64) = 4096 floats per sample × 10k samples = ~300MB. Zarr chunking matters.

### 3. Config `DamageGenConfig.damage_min = 0.1` is too aggressive

`config.py:9`: Damage range [0.1, 1.0] means the most damaged material still has 10% stiffness. Real delamination can approach 0% locally. The roadmap (`proposal2_roadmap.md:72`) says `d ∈ [0,1]` where `d=0` is fully damaged. Config clips away the most informative damage regime.

### 4. No `Laminate` helper to apply damage

The roadmap (`proposal2_roadmap.md:89`) says `damage_to_laminate(base_laminate, damage_field)` creates a new `Laminate` with scaled material properties. No such function exists in `laminate.py` or `inverse_damage/`. The `Material` dataclass (`laminate.py:9-32`) is mutable (dataclass), so damage could be applied by modifying `E1, E2, G12, G13, G23` in place, but the roadmap wants immutable copies.

### 5. Missing `__init__.py` exports

`src/inverse_damage/__init__.py:1` — no exports. P1's `__init__.py` exports all public classes. P2 should match.

## Nice-to-Haves

### 1. Config could use `from_yaml` like main config

`config.py` defines dataclasses but has no serialization. P1's data pipeline uses inline config. For reproducibility, a YAML loader matching `config.py:ExperimentConfig` would help.

### 2. `EncoderConfig.d_conditioning` matches `DecoderConfig.d_conditioning`

Both use 128. This is good — the conditioning vector size is consistent. But `PosteriorConfig.d_conditioning` is also 128. If the posterior network takes `c` from encoder and passes to decoder, all three must agree. Currently they do, but there's no assertion enforcing this.

### 3. `PosteriorConfig.n_coupling_layers = 8` is aggressive start

Roadmap (`proposal2_roadmap.md:197`) says "start with 4, increase if underfitting". Config starts at 8. Coupling layers are expensive and prone to training instability.

## Test Coverage Gaps

**No tests exist.** All gaps below are from the roadmap's test plan (`proposal2_roadmap.md:56-63`):

| Planned Test | What It Verifies | Status |
|---|---|---|
| `test_damage_field.py` | Damage values in [0,1], laminate scaling, ABD correctness | Not written |
| `test_data_gen.py` | 10-sample generation, frequency monotonicity, zarr output | Not written |
| `test_encoder.py` | Forward shape, gradient flow, output normalization | Not written |
| `test_decoder.py` | Forward shape, output in [0,1], autoencoder reconstruction | Not written |
| `test_posterior.py` | cINN forward/inverse, sampling, prior match, forward-inverse consistency | Not written |
| `test_baselines.py` | Baseline training, evaluation | Not written |
| `test_sbc.py` | SBC rank histogram, calibration error | Not written |

### Missing behavioral tests (from roadmap, never planned):

1. **Damage scaling physics**: Does Material E×d produce correct frequency shift? (need numerical check: f(d=0.5) ≈ f(d=1.0) × sqrt(0.5) for thin plate bending)
2. **Encoder-decoder roundtrip**: After autoencoder training, does `decoder(encoder(damage)) ≈ damage` within tolerance?
3. **Posterior collapse detection**: Does KL→0 during training? Is posterior variance non-degenerate?
4. **Extreme damage edge cases**: d=0.0 (singular), d=1.0 (identity), d<0 (invalid)
5. **Solver stability under material scaling**: Does FSDT solver converge with E scaled to 0.3× or 0.01×?

## Specific Code References

| File:Line | Issue |
|---|---|
| `src/inverse_damage/__init__.py:1` | No exports, just docstring |
| `src/inverse_damage/config.py:9` | `damage_min=0.1` clips informative damage regime |
| `src/inverse_damage/config.py:23` | `grid=(64,64)` doesn't document upsample from damage grid (16,16) |
| `src/inverse_damage/config.py:33` | `mode="mlp"` defaults to weaker encoder mode |
| `src/inverse_damage/config.py:41` | `DecoderConfig` missing `d_c` conditioning dimension |
| `src/inverse_damage/config.py:48` | `PosteriorConfig` missing `kl_weight` for KL annealing |
| `src/mechanics/solver.py:197-235` | `assemble_stiffness()` has no damage mask parameter |
| `src/mechanics/laminate.py:9-32` | `Material` is mutable dataclass — damage scaling could mutate in place |
| `roadmaps/proposal2_roadmap.md:89` | Proposes `damage_to_laminate()` but no interface defined |

## Verdict

**Cannot proceed with implementation review — nothing exists.** The roadmap is sound but the config needs fixes before Phase 1 starts. Priorities:

1. Fix `DamageGenConfig.damage_min` → 0.0 (allow full damage)
2. Add `d_c` to `DecoderConfig`
3. Add `kl_weight` to `PosteriorConfig`
4. Document damage→solver grid upsample strategy
5. Create at least `damage_field.py` with `DamageField` + `damage_to_laminate()` to validate the physics before building ML components
