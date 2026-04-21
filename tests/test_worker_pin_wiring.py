# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for GPUWorker's grad_accum hard-pin wiring.

Exercises ``GPUWorker._update_accum_pin`` with a stub queue — we're
verifying the per-op dispatch logic (which ops pin vs clear), not
the full worker pipeline. Full-pipeline coverage lives in the GPU
test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

pytest.importorskip("torch")

from hatchery.core.protocols import QueuedJob  # noqa: E402


@dataclass
class _PinCall:
    kind: str  # "set" or "clear"
    session_id: str
    worker_id: Optional[str] = None


class _StubQueue:
    """Records set_accum_pin / clear_accum_pin calls."""

    def __init__(self) -> None:
        self.calls: list[_PinCall] = []

    async def set_accum_pin(self, session_id: str, worker_id: str, *, ttl_s: float = 600.0) -> None:
        self.calls.append(_PinCall(kind="set", session_id=session_id, worker_id=worker_id))

    async def clear_accum_pin(self, session_id: str) -> None:
        self.calls.append(_PinCall(kind="clear", session_id=session_id))


class _StubConfig:
    def __init__(self, queue: _StubQueue) -> None:
        self.queue = queue


class _StubWorker:
    """Minimal stand-in that binds ``_update_accum_pin`` from GPUWorker."""

    # Bind only what ``_update_accum_pin`` actually reads.
    from hatchery.core.worker import GPUWorker

    _ACCUM_PINNING_OPS = GPUWorker._ACCUM_PINNING_OPS
    _update_accum_pin = GPUWorker._update_accum_pin

    def __init__(self) -> None:
        self.queue = _StubQueue()
        self.config = _StubConfig(self.queue)
        self.worker_id = "w-test"


def _job(op: str, session_id: str = "s1") -> QueuedJob:
    return QueuedJob(
        job_id=f"j-{op}",
        session_id=session_id,
        operation=op,
        payload=b"",
    )


@pytest.mark.parametrize(
    "operation",
    ["forward_backward", "forward_custom_step2"],
)
async def test_pin_set_for_accumulation_ops(operation):
    w = _StubWorker()
    await w._update_accum_pin(_job(operation))
    assert w.queue.calls == [_PinCall(kind="set", session_id="s1", worker_id="w-test")]


async def test_pin_cleared_for_optim_step():
    w = _StubWorker()
    await w._update_accum_pin(_job("optim_step"))
    assert w.queue.calls == [_PinCall(kind="clear", session_id="s1")]


@pytest.mark.parametrize(
    "operation",
    ["init_session", "sample", "compute_logprobs", "load_weights"],
)
async def test_pin_noop_for_unrelated_ops(operation):
    w = _StubWorker()
    await w._update_accum_pin(_job(operation))
    assert w.queue.calls == []


async def test_pin_silently_skipped_when_queue_lacks_methods():
    """Backward-compat: queues without pin methods don't break the worker."""

    class _OldQueue:
        pass

    class _OldWorker(_StubWorker):
        def __init__(self) -> None:  # noqa: D401
            self.config = _StubConfig(_OldQueue())  # type: ignore[arg-type]
            self.worker_id = "w-test"

    w = _OldWorker()
    await w._update_accum_pin(_job("forward_backward"))
    await w._update_accum_pin(_job("optim_step"))
    # No AttributeError, no crash.
