# Task Completion Commands

## When a coding task is done

1. Run tests:
```bash
pytest tests/ --cov=src/mechanics -v
```

2. Run validation:
```bash
python validate.py
```

3. Check code quality:
```bash
# If configured
ruff check src/mechanics/
mypy src/mechanics/
```

4. Verify no regressions:
```bash
pytest tests/test_solver.py -v
pytest tests/test_laminate.py -v
```