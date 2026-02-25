# Contributing to ATLAS

Thank you for your interest in contributing to ATLAS!

## Development Setup

```bash
# Clone and create virtual environment
git clone https://github.com/<your-org>/atlas-materials-research.git
cd atlas-materials-research
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install in editable mode with dev dependencies
pip install -r requirements.txt
pip install -e ".[test]"
pip install ruff
```

## Code Quality

We use **ruff** for linting. Run before committing:

```bash
ruff check atlas/ tests/ scripts/
```

Auto-fix safe issues:

```bash
ruff check atlas/ tests/ --fix
```

Configuration lives in [`ruff.toml`](ruff.toml).

## Testing

```bash
# Run unit tests (fast, no GPU needed)
pytest tests/unit/ -v

# Run integration tests (may need optional dependencies)
pytest -m integration -v

# Coverage report
pytest tests/unit/ --cov=atlas --cov-report=html
```

### Writing Tests

- Place unit tests in `tests/unit/<module>/`
- Use `pytest` fixtures and `unittest.mock` for heavy dependencies
- Test file naming: `test_<module_name>.py`
- All tests must pass on CPU without external data downloads

## Project Structure

```
atlas/              # Core package
├── models/         # GNN architectures (CGCNN, EquivariantGNN, M3GNet, MultiTask)
├── data/           # Data loading and preprocessing
├── training/       # Training loop, losses, metrics
├── active_learning/# Discovery controller
├── topology/       # Topological classification
├── potentials/     # MACE relaxation
├── benchmark/      # Matbench evaluation
└── research/       # Method registry and reproducibility workflows

scripts/            # Phase-specific execution scripts
tests/              # Unit and integration tests
docs/               # Project documentation
```

## Pull Request Process

1. Create a feature branch from `develop`
2. Write tests for new functionality
3. Ensure `ruff check` passes with zero errors
4. Ensure `pytest tests/unit/` all passes
5. Update `CHANGELOG.md` under `[Unreleased]`
6. Submit PR with a clear description

## Commit Messages

Use clear, descriptive commit messages:

```
fix(topology): remove duplicate CrystalGraphBuilder from classifier
feat(training): add evidential loss for uncertainty-aware training
test(models): add unit tests for EnsembleUQ and MCDropoutUQ
docs: add CONTRIBUTING.md
```
