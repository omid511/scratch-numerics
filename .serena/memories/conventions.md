# Code Conventions

## Style
- Type hints used throughout
- Docstrings for public functions and classes
- Dataclasses for data structures (Material, Laminate, SolverResult)
- NumPy arrays for all numerical operations

## Naming
- Snake_case for functions and variables
- PascalCase for classes
- UPPER_CASE for constants (AIR_DENSITY, SOUND_SPEED)
- Private methods prefixed with underscore (_base_matrices, _eval_mode_on_grid)

## Architecture
- Solver uses precomputed basis integrals for speed
- Boundary conditions via spring assembly (high stiffness approximation)
- Material properties as dataclasses with computed properties (Q(), ABD())
- Configuration via ExperimentConfig dataclass with YAML serialization

## Testing
- pytest with coverage
- Physics-based validation against analytical solutions
- Convergence checks for numerical parameters