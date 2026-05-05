# Contributing to Hatchery Core

Thanks for helping improve Hatchery Core. This repository is the open-source
core package for local and self-hosted training workflows.

## Development Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,test]'
```

For GPU-backed local testing, install the PyTorch build appropriate for your
machine, then use the same editable install:

```bash
uv pip install -e '.[gpu,dev,test]'
```

The `gpu` extra is currently a compatibility placeholder. PyTorch wheel
selection is controlled by your package manager and index configuration.

## Checks

Run the focused checks before opening a pull request:

```bash
ruff check hatchery/ tests/
ruff format --check hatchery/ tests/
python -m pytest --ignore=tests/torch_tests -q
```

For packaging changes, also build and inspect a wheel:

```bash
python -m pip install build
python -m build
python -m zipfile -l dist/*.whl
```

GPU tests live under `tests/torch_tests/` and require a CUDA-capable machine:

```bash
python -m pytest tests/torch_tests/ -q
```

## Pull Requests

- Keep changes scoped to the behavior being changed.
- Add or update tests when changing public APIs, protocol contracts, storage
  behavior, auth, scheduler behavior, or Tinker compatibility.
- Update README/docs when changing install commands, user-facing defaults, or
  supported deployment shapes.
- Do not commit generated artifacts such as `*.egg-info`, `dist/`, caches, or
  local model outputs.

## Open-Core Boundary

Core should remain usable without hosted services. Keep production-specific
infrastructure behind protocols, plugins, or extension packages. See
`content/docs/learn/open-core.mdx` for the intended boundary.
