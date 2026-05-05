# Release Checklist

This checklist is for maintainers preparing a public Hatchery Core release.

## Before Tagging

- Confirm `VERSION` contains the intended version.
- Confirm `CHANGELOG.md` has a release section for the version.
- Run lint and tests:

```bash
ruff check hatchery/ tests/
ruff format --check hatchery/ tests/
python -m pytest --ignore=tests/torch_tests -q
```

- Run GPU tests on an appropriate runner when the release touches worker,
  trainer, loss, precision, model-pool, or sampling behavior:

```bash
python -m pytest tests/torch_tests/ -q --tb=short
```

- Build and inspect artifacts:

```bash
python -m build
python -m zipfile -l dist/*.whl
```

- Install the wheel in a clean environment and smoke-test imports:

```bash
python -m pip install dist/*.whl
python -c "import hatchery.core.gateway; import hatchery.core.local_dev"
```

## Publishing

Publishing is tag-triggered through `.github/workflows/publish.yml`.

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Do not reuse tags for changed artifacts. If a release is bad, publish a new
patch version.
