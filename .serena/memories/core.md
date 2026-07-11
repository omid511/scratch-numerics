# Core Project Information

## Project Overview
Aeroelastic analysis of laminated honeycomb sandwich plates using fast FSDT solver with precomputed basis integrals. Four ML proposals for physics-informed probabilistic machine learning.

## Source Map
- `src/mechanics/solver.py` — FSDTSolver with modal/complex-modal flutter analysis
- `src/mechanics/laminate.py` — Material, Laminate classes with ABD matrices
- `src/mechanics/basis.py` — Precomputed basis integrals (Legendre/trigonometric)
- `src/mechanics/boundary.py` — Boundary spring assembly
- `src/mechanics/piston_theory.py` — Piston theory for aerodynamic loads
- `src/mechanics/config.py` — ExperimentConfig with provenance
- `src/mechanics/result.py` — SolverResult, AeroelasticResult dataclasses
- `src/mechanics/honeycomb.py` — Honeycomb core properties
- `src/mechanics/shear_correction.py` — Vlachoutsis shear correction

## Key Invariants
- Solver speed: ~0.3s per run
- FSDT systematic ~4% error vs 3D FEM due to global shear correction factor (κ=5/6)
- Precomputed basis integrals are material-free, done once per (M, N, a, b)
- Boundary conditions enforced via spring assembly with high stiffness (1e12-1e14)

## References
- `mem:tech_stack` — language, frameworks, dependencies
- `mem:suggested_commands` — dev, test, lint commands
- `mem:conventions` — code style, naming patterns
- `mem:task_completion` — completion commands
- `mem:proposals` — research proposals overview