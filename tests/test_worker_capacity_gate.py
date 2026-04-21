# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for GPUWorker._enforce_fft_capacity zombie eviction.

Regression: a successful FP init followed by a gateway-side failure
between worker-ack and client response leaves ``_fp_base_state[sid]``
populated forever, wedging every subsequent FP create_model on this
worker. The gate evicts incumbent FP sessions that have never received
a training op, but still protects sessions actively training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

pytest.importorskip("torch")


@dataclass
class _Runtime:
    session_id: str
    meta: dict = field(default_factory=lambda: {"total_steps": 0})


class _StubCache:
    """Minimal SmartLoRACache stand-in: get + evict, no scoring."""

    def __init__(self) -> None:
        self._cache: dict[str, _Runtime] = {}
        self.evicted: list[str] = []

    def put(self, sid: str, rt: _Runtime) -> None:
        self._cache[sid] = rt

    def get(self, sid: str) -> Optional[_Runtime]:
        return self._cache.get(sid)

    def evict(self, sid: str) -> None:
        if sid in self._cache:
            self._cache.pop(sid)
            self.evicted.append(sid)


class _StubWorker:
    """Bind only ``_enforce_fft_capacity`` from GPUWorker."""

    from hatchery.core.worker import GPUWorker

    _enforce_fft_capacity = GPUWorker._enforce_fft_capacity

    def __init__(self) -> None:
        self._fp_base_state: dict[str, dict] = {}
        self._cache = _StubCache()


def _attach(w: _StubWorker, sid: str, *, total_steps: int = 0) -> None:
    """Simulate the post-init worker state for an FP session."""
    w._fp_base_state[sid] = {"weight": object()}
    w._cache.put(sid, _Runtime(session_id=sid, meta={"total_steps": total_steps}))


def test_capacity_gate_evicts_zombie_session():
    """First FP init succeeded but gateway crashed → zombie. Second
    init for a different session_id must succeed by evicting it."""
    w = _StubWorker()
    _attach(w, "sid_zombie", total_steps=0)

    w._enforce_fft_capacity("sid_live")

    assert "sid_zombie" not in w._fp_base_state
    assert "sid_zombie" in w._cache.evicted
    assert w._cache.get("sid_zombie") is None


def test_capacity_gate_rejects_when_incumbent_is_training():
    """Incumbent has done real work → reject the new FP init so the
    active workload isn't trampled."""
    w = _StubWorker()
    _attach(w, "sid_a", total_steps=1)

    with pytest.raises(RuntimeError, match="single-FFT-per-worker"):
        w._enforce_fft_capacity("sid_b")

    assert "sid_a" in w._fp_base_state
    assert w._cache.evicted == []


def test_capacity_gate_evicts_when_runtime_missing():
    """``_fp_base_state`` populated but ``_cache`` already lost the
    runtime (e.g. cache eviction race) → also treat as zombie."""
    w = _StubWorker()
    w._fp_base_state["sid_orphan"] = {"weight": object()}
    # Note: _cache has no entry for sid_orphan.

    w._enforce_fft_capacity("sid_new")

    assert "sid_orphan" not in w._fp_base_state


def test_capacity_gate_noop_when_same_session():
    """Re-attach for the same session_id is allowed (idempotent init)."""
    w = _StubWorker()
    _attach(w, "sid_x", total_steps=5)

    w._enforce_fft_capacity("sid_x")  # must not raise, must not evict

    assert "sid_x" in w._fp_base_state
    assert w._cache.evicted == []


def test_capacity_gate_admits_first_session():
    """Empty state → no rejection, no eviction."""
    w = _StubWorker()
    w._enforce_fft_capacity("sid_first")
    assert w._fp_base_state == {}
    assert w._cache.evicted == []
