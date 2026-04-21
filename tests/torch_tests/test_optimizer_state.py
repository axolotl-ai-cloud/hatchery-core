# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for :mod:`hatchery.core.optimizer_state` — the FullOptimizerPersister.

Covers the baseline (no-compression) persister that ships with core.
The delta-compressed variant is covered in
the hatchery-hosted test suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hatchery.core.backends.object_store.memory import (  # noqa: E402
    InMemoryObjectStore as MemoryObjectStore,
)
from hatchery.core.optimizer_state import (  # noqa: E402
    OPTIMIZER_SNAPSHOT_FILE,
    FullOptimizerPersister,
    OptimizerStateConfig,
)


def _opt_state() -> dict:
    torch.manual_seed(0)
    return {
        "state": {
            0: {
                "step": torch.tensor(4),
                "exp_avg": torch.randn(8, 16),
                "exp_avg_sq": torch.randn(8, 16).abs(),
            }
        },
        "param_groups": [{"lr": 1e-4, "betas": (0.9, 0.999), "params": [0]}],
    }


async def test_save_none_state_writes_nothing():
    objects = MemoryObjectStore()
    p = FullOptimizerPersister()
    result, cache = await p.save(
        objects,
        "sess/live",
        None,
        snapshot_cache=None,
        snapshot_version=0,
        delta_count=0,
        cfg=OptimizerStateConfig(),
    )
    assert result.wrote_snapshot is False
    assert result.wrote_delta is False
    assert cache is None
    assert not await objects.exists(f"sess/live/{OPTIMIZER_SNAPSHOT_FILE}")


async def test_save_writes_full_snapshot():
    objects = MemoryObjectStore()
    p = FullOptimizerPersister()
    state = _opt_state()
    result, _ = await p.save(
        objects,
        "sess/live",
        state,
        snapshot_cache=None,
        snapshot_version=0,
        delta_count=0,
        cfg=OptimizerStateConfig(),
    )
    assert result.wrote_snapshot is True
    assert result.snapshot_version == 1
    assert result.snapshot_bytes > 0
    assert await objects.exists(f"sess/live/{OPTIMIZER_SNAPSHOT_FILE}")


async def test_load_round_trip():
    objects = MemoryObjectStore()
    p = FullOptimizerPersister()
    state = _opt_state()
    await p.save(
        objects,
        "sess/live",
        state,
        snapshot_cache=None,
        snapshot_version=0,
        delta_count=0,
        cfg=OptimizerStateConfig(),
    )
    loaded, _cache, _ver, _dc = await p.load(objects, "sess/live")
    assert loaded is not None
    assert torch.equal(loaded["state"][0]["exp_avg"], state["state"][0]["exp_avg"])
    assert torch.equal(loaded["state"][0]["exp_avg_sq"], state["state"][0]["exp_avg_sq"])
    assert loaded["param_groups"] == state["param_groups"]


async def test_load_missing_returns_none():
    objects = MemoryObjectStore()
    p = FullOptimizerPersister()
    state, cache, ver, dc = await p.load(objects, "sess/live")
    assert state is None
    assert cache is None
    assert (ver, dc) == (0, 0)
