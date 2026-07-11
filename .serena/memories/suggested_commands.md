# Suggested Commands

## Development
```bash
# Run validation script
python validate.py

# Run sweep demo
python sweep_demo.py

# Run tests with coverage
pytest tests/ --cov=src/mechanics

# Run specific test
pytest tests/test_solver.py -v
```

## Code Quality
```bash
# Type checking (if mypy configured)
mypy src/mechanics/

# Linting (if ruff configured)
ruff check src/mechanics/

# Formatting
black src/mechanics/
```

## Build & Install
```bash
# Install in development mode
pip install -e .

# Or with uv
uv pip install -e .
```