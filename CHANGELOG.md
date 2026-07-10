# Changelog

All notable changes to Hatchery Core will be documented here.

This project follows semantic versioning once public releases begin.

## Unreleased

- Prepared repository metadata, licensing, packaging, CI, and contributor docs
  for an open-source release.

## 0.3.0 - 2026-07-10

### Distributed training

- Added a `DistributedHelpers` registry to `hatchery.core.parallel_hooks`
  (`register_distributed_helpers` / `get_distributed_helpers`), letting an
  extension package register torch-distributed helpers (process-group init,
  device-mesh construction, optional context-parallel region) that core can
  reach without importing or naming the extension. Purely additive; core's
  existing DP-only FSDP2 path is unchanged and there is no in-tree consumer
  yet.

### Documentation

- Documented the operator loss-plugin registry (`register_loss` /
  `hatchery.losses` entry-point group) in the extending guide.
- Added a Fumadocs-based public docs site under `docs-site/`, rendering the
  existing `content/docs` MDX in place, including a new Tinker-migration
  guide.

### Breaking changes

- None. `hatchery-core` 0.2.0 (tag `v0.2.0`) already shipped the FP8/TorchAO
  fixes, the `/api/v1/create_sampling_session` tinker-parity route, and the
  loss-plugin registry — those predate this release and are not repeated
  here. The only source change since `v0.2.0` is the additive
  `DistributedHelpers` registry above; no public symbol was renamed or
  removed.
