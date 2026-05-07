# Venvstacks Packaging

This directory contains a `venvstacks` stack definition for packaging Hatchery
Core as layered Python environments.

`venvstacks` installs from wheels for reproducible builds, so build a local
wheel for the current checkout before locking or building the stack:

```bash
python -m build --wheel --outdir packaging/venvstacks/wheelhouse
uvx --from venvstacks venvstacks show packaging/venvstacks/venvstacks.toml
uvx --from venvstacks venvstacks lock \
  --local-wheels wheelhouse \
  packaging/venvstacks/venvstacks.toml
uvx --from venvstacks venvstacks build \
  --local-wheels wheelhouse \
  --include-dependencies \
  packaging/venvstacks/venvstacks.toml
```

Published Hatchery releases can omit `--local-wheels` and resolve
`hatchery-core` from the configured package indexes.

The stack provides these application layers:

- `hatchery-local-dev`: starts the combined local gateway/worker launcher.
- `hatchery-gateway`: starts the FastAPI gateway with Uvicorn.
- `hatchery-worker`: starts a standalone worker.

Runtime configuration remains environment-variable driven, matching the normal
`python -m hatchery.core.local_dev`, `uvicorn hatchery.core.gateway:create_app`,
and `python -m hatchery.core.worker` workflows.
