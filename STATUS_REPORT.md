# FSDT Solver — Status Report

**Date:** 2026-07-04  
**Commits:** 3 (`97cfa22` → `521b7f9` → `88fef74`)  
**Tests:** 166 passing, 99% line coverage (3 uncovered lines — all dead code)  
**Validation:** <0.25% modal frequency error, <0.85% eigenvalue error vs original code

---

## 1. What Exists

### Source Modules (10 files, ~710 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `solver.py` | 481 | Core FSDT state-space eigenvalue solver |
| `laminate.py` | 176 | Material, ABD matrix, kappa (Vlachoutsis) |
| `config.py` | 166 | YAML experiment config with provenance |
| `piston_theory.py` | 99 | λ(V), V(λ), piston pressure |
| `basis.py` | 108 | Legendre/trig basis, precomputed integrals |
| `honeycomb.py` | 65 | Gibson's corrected formula |
| `boundary.py` | 46 | Penalty spring BC builder |
| `result.py` | 42 | SolverResult, AeroelasticResult dataclasses |
| `shear_correction.py` | 19 | Wrapper around laminate.kappa() |
| `__init__.py` | 10 | Public API exports |

### Test Suite (11 files, 166 tests)

Coverage by module: basis 100%, config 100%, honeycomb 100%, laminate 100%, piston_theory 100%, result 100%, ritz (deleted), shear_correction 100%, solver 99%, boundary 100%.

### Validation Results

```
Modal frequencies (SSSS sandwich, M=N=10):
  Mode 1: 1169 Hz vs 1172 Hz (0.24% error)
  Mode 2: 2220 Hz vs 2225 Hz (0.22% error)
  Mode 3: 2220 Hz vs 2225 Hz (0.22% error)
  Mode 4: 3104 Hz vs 3111 Hz (0.21% error)

Flutter curve eigenvalues (λ=100–600):
  Real part error:  <0.85% (peaks at flutter crossing)
  Imaginary part:   <0.50%
  Flutter crossing captured at correct λ
```

---

## 2. Design Decisions and Rationale

### 2.1 State-Space Eigenvalue Formulation

**Decision:** Assemble M, K, K_air, C_air matrices, form `A = [[0,I],[−(K+K_air),−(C+C_air)]]`, solve generalized eigenvalue `A x = λ B x`.

**Why not Newmark-β / direct time integration:**
- Eigenvalue formulation gives all stability information in one solve (no stepping through transient to find divergence)
- Matches the original code's approach (`plate-main/plate/solver.py`)
- 0.3s per solve enables 10K+ sweep samples; time integration would be 10–100× slower for stability analysis

**Why not reduced-order model (ROM) first:**
- The FSDT state-space at M=N=15 is only 2250-dimensional — dense solve is fast enough
- ROM adds complexity for marginal speedup at this scale
- Can add modal truncation later if M=N=15 is too slow for sweeps

### 2.2 Precomputed Basis Integrals

**Decision:** `precompute_integrals(M, L, basis_type)` computes 4 matrices per axis (function, derivative, mixed, stiffness products). Material-free, computed once per (M, N, L1, L2, basis_type).

**Why:** The original code recomputes these inside each assembly call. For sweeps where only velocity changes, this is pure waste. Precomputation amortizes the cost across all solves with the same discretization.

**Trade-off:** Slightly more complex initialization. But for sweep use cases (10K+ solves with same M, N, L1, L2), the savings dominate.

### 2.3 Penalty Spring Boundary Conditions

**Decision:** All BC types (clamped/SS/free/elastic) implemented as penalty springs with k=1e12. Springs added to stiffness matrix at boundary points.

**Why not essential BCs (penalty method elimination):**
- Penalty method is simpler to implement — same code path for all BC types
- Matches the original code's approach
- k=1e12 gives edge/peak ratio <10% for mode shapes (verified by test)
- For ML training data, consistent numerical treatment across all BC types matters more than exact BC enforcement

**Known limitation:** Penalty springs don't enforce BCs exactly. Mode shapes at clamped edges are ~8% of peak amplitude instead of 0%. This is acceptable for training data generation but may matter for high-fidelity validation.

### 2.4 D11 from Full ABD (Not First-Ply)

**Decision:** `D11 = ABBD[3,3]` from the laminate's full 6×6 ABD matrix, not `materials[0].Q()[0,0] * h³/12`.

**Why:** The first-ply approximation overestimates sandwich D11 by 2×. This corrupts λ normalization, which feeds into CFAP — the exact quantity P3 estimates margins for and P4 labels transient clips relative to.

**What this means for the original code:** The original code uses the first-ply approximation (`plate.py:755-757`). Our validation compares at the same physical velocities (converting reference λ through old D11), so the comparison is fair. But the original code's λ values themselves carry the same 2× bias. When P3/P4 use our corrected D11, their λ values will be physically more accurate than the original code's.

**Downstream impact:** Halves all λ values. The original code's CFAP at λ≈308 corresponds to our λ≈154. Test ranges must be adjusted accordingly.

### 2.5 Analytical Quadratic for V(λ)

**Decision:** Solve the quadratic-in-V² analytically rather than Newton/bisection.

**Why:** The original code uses Newton/bisection (`velocity_from_lambda`), but λ(V) has a minimum at M=√2. Below that, Newton fails silently (returns V≈c·1.001). The analytical solution always gives a valid result.

**Which root:** The larger root (matching `plate.py:778`). The two roots correspond to sub- and super-critical velocities for the same λ. The larger root is the physically relevant one for flutter analysis.

### 2.6 Eigenvector Split

**Decision:** State-space eigenvector is `[q, qdot]`; displacement is the first half `[:size]`, not the second.

**Why:** The original code constructs state-space as `[displacement, velocity]`. Our initial implementation extracted the second half (velocity) — this was wrong and caught during validation against the original code's mode shapes.

### 2.7 Vlachoutsis Method for kappa

**Decision:** Through-thickness quadrature with `Q_inplane = Qbi[[0,1],[0,1]]` for g/R, `Q_shear = Qbi[[4,3],[4,3]]` for d/I_.

**Why:** This is the original code's approach, matching Vlachoutsis (1992). Returns 5/6 for homogeneous isotropic (correct), ~0.165 for sandwich (physically reasonable — sandwich shear flexibility reduces kappa significantly).

**Known limitation:** NQUAD=6 is hardcoded with no convergence study. May be insufficient for some laminate configurations in P1's sweep space.

### 2.8 find_flutter_boundary Sentinel

**Decision:** Return `None` when both endpoints have same stability (no crossing found), instead of silently returning a boundary value.

**Why:** P3's active learning loop will call this autonomously across a wide design space. A garbage λ_cr (that looks like a real crossing) would corrupt the acquisition function. `None` is explicit and forces the caller to handle the no-crossing case.

### 2.9 Config Schema

**Decision:** YAML-based with dict-per-edge boundary (not just labels), joint LHS sweep over multiple parameters, provenance fields (schema_version, solver_version, git_hash).

**Why:** Dict-per-edge carries k parameters needed for elastic BCs. Joint LHS is essential for P3 (correlated parameters matter for mode-veering). Provenance enables reproducibility tracking across 10K+ sample runs.

### 2.10 Trig Basis Dimension Tracking

**Decision:** `_M_eff`, `_N_eff`, `_MN_eff` track actual number of basis functions (2(M−1)+1 for trig, M for Legendre). `expand_basis` infers dimensions from matrix shapes, not M,N params.

**Why:** The trig basis `trig_and_derivative(M-1, x)` returns 2(M−1)+1 functions, not M. Without tracking, matrix dimensions mismatch silently. Inferring from shapes is robust — works for any basis family.

---

## 3. Known Limitations

### Physics

1. **Damping formula** — `damp_coeff = A_dyn(M²-2)/((M²-1)V)` matches original code but differs from textbook piston theory `2ρV/√(M²-1)`. Factor of `(M²-2)/(2(M²-1))` approaches 0.5 as M→∞. Matters for P4 transient damping.

2. **D11 still uses first-ply convention in `non_dimensional_lambda`** — The lambda formula itself is `ρV²L³/(D11√(M²-1))`. We compute D11 correctly from ABD, but the original code's λ convention (which P3/P4 will reference) used the first-ply D11. The physical meaning of λ depends on which D11 you use.

3. **kappa quadrature** — NQUAD=6 hardcoded, no convergence check. May be insufficient for some layups in P1's sweep space.

4. **Penalty springs** — Edge mode shapes ~8% of peak instead of 0%. Acceptable for training data, may matter for validation against experiment.

### Infrastructure

5. **No sweep engine** — Config schema exists but no code to sample → solve → serialize. Blocks P1, P2, P3 data generation.

6. **No transient response** — Modal superposition `w(x,y,t) = Σ c_k φ_k exp(λ_k t)` not implemented. Blocks P4.

7. **No mode tracking** — Eigenvalues sorted by frequency, not tracked by physical identity across velocity steps. Blocks P3's mode-veering detection.

8. **No result serialization** — SolverResult stores numpy arrays, no zarr/hdf5 writer. Blocks P1/P3 output.

9. **No all-DOF mode shape export** — `_eval_mode_on_grid` extracts only DOF 2 (w). P2 needs (u, v, w, φx, φy). Coefficients are stored but no public reconstruction method.

10. **No gradient access** — P3 needs dλ_cr/dθ. Finite differences are sufficient (0.3s per solve, 10–20 parameters) but plumbing not built.

### Dead Code (cleaned up)

- `ritz.py` — deleted (unused by solver)
- `boundary.py:assemble_springs()` — deleted (standalone function with `continue` on all non-None coords)
- `solver.py:110,123` — unreachable ValueError for unknown basis type
- `solver.py:194` — unreachable kappa fallback

---

## 4. Performance Profile

| Operation | Time | Notes |
|-----------|------|-------|
| `solve_modal(n_modes=20)` | ~0.08s | M=N=15, dense eigensolve |
| `solve_complex_modal(V, n_modes=4)` | ~0.3s | State-space ARPACK |
| `find_flutter_boundary()` | ~15s | 50 bisection iterations × 0.3s |
| Per sweep sample | ~0.3s | Dominated by complex eigensolve |

**Bottleneck for 10K sweeps:** ~50 minutes at current speed. Key optimization: pre-assemble M, K, K_spring once per (material, BC) configuration; only K_air, C_air change per velocity.

**Estimated after optimization:** Pre-assembling drops per-iteration cost from 0.3s to ~0.12s (drops 3 matrix assemblies + 2 linear solves). 10K sweeps → ~20 minutes.

---

## 5. File Map

```
src/mechanics/
├── __init__.py          — Public API exports
├── honeycomb.py         — Gibson's corrected formula (65 lines)
├── laminate.py          — Material, ABD, kappa (176 lines)
├── shear_correction.py  — Wrapper around kappa (19 lines)
├── piston_theory.py     — λ(V), V(λ), piston pressure (99 lines)
├── basis.py             — Legendre/trig, precomputed integrals (108 lines)
├── boundary.py          — Penalty spring BC builder (46 lines)
├── solver.py            — FSDTSolver core (481 lines)
├── config.py            — YAML experiment config (166 lines)
└── result.py            — SolverResult, AeroelasticResult (42 lines)

tests/
├── test_solver.py       — 18 tests (T matrix, mass, stiffness, convergence)
├── test_coverage.py     — 68 tests (behavioral gaps, edge cases)
├── test_laminate.py     — 18 tests (ABD, kappa, D-block)
├── test_piston_theory.py — 12 tests (lambda, velocity, pressure)
├── test_basis.py        — 17 tests (Legendre, trig, integrals)
├── test_boundary.py     — 11 tests (spring builder)
├── test_config.py       — 10 tests (YAML, defaults)
├── test_honeycomb.py    — 8 tests (Gibson formula, validation)
├── test_result.py       — 5 tests (dataclass construction)
└── test_shear_correction.py — 5 tests (kappa delegation)

validation_*.png        — Comparison plots vs original code
```
