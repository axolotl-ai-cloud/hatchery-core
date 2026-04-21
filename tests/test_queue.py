# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the in-memory job queue."""

from __future__ import annotations

import asyncio

import pytest

from hatchery.core.backends.queue.memory import InMemoryJobQueue
from hatchery.core.protocols import JobResult, JobStatus, QueuedJob


def make_job(job_id: str, **kwargs) -> QueuedJob:
    return QueuedJob(
        job_id=job_id,
        session_id=kwargs.pop("session_id", "s1"),
        operation=kwargs.pop("operation", "forward_backward"),
        payload=kwargs.pop("payload", b"p"),
        **kwargs,
    )


@pytest.fixture
async def queue():
    q = InMemoryJobQueue()
    await q.initialize()
    yield q
    await q.close()


async def test_enqueue_dequeue(queue):
    await queue.enqueue(make_job("j1"))
    job = await queue.dequeue("worker-a")
    assert job is not None
    assert job.job_id == "j1"


async def test_dequeue_empty_returns_none(queue):
    job = await queue.dequeue("worker-a")
    assert job is None


async def test_priority_ordering(queue):
    # Priority orders jobs ACROSS sessions; within a session FIFO always
    # wins (see tests/test_pipelining.py). So we use distinct sessions
    # here to exercise the cross-session priority path.
    await queue.enqueue(make_job("low", session_id="s-low", priority=0))
    await queue.enqueue(make_job("high", session_id="s-high", priority=10))
    await queue.enqueue(make_job("mid", session_id="s-mid", priority=5))
    first = await queue.dequeue("w")
    await queue.ack(first.job_id, JobResult(job_id=first.job_id, status=JobStatus.COMPLETED))
    second = await queue.dequeue("w")
    await queue.ack(second.job_id, JobResult(job_id=second.job_id, status=JobStatus.COMPLETED))
    third = await queue.dequeue("w")
    assert [first.job_id, second.job_id, third.job_id] == ["high", "mid", "low"]


async def test_model_filter(queue):
    # Different sessions so both heads are eligible.
    await queue.enqueue(make_job("a", session_id="s-a", required_model="modelA"))
    await queue.enqueue(make_job("b", session_id="s-b", required_model="modelB"))
    job = await queue.dequeue("w", model_filter="modelB")
    assert job.job_id == "b"
    job = await queue.dequeue("w", model_filter="modelA")
    assert job.job_id == "a"


async def test_model_filter_list_matches_any(queue):
    """A list filter matches any worker-loaded model; unknown models skipped."""
    await queue.enqueue(make_job("a", session_id="s-a", required_model="modelA"))
    await queue.enqueue(make_job("b", session_id="s-b", required_model="modelB"))
    await queue.enqueue(make_job("c", session_id="s-c", required_model="modelC"))
    # Worker holds A and B. Should be able to dequeue a and b, but NOT c.
    got = set()
    for _ in range(2):
        job = await queue.dequeue("w", model_filter=["modelA", "modelB"])
        assert job is not None
        got.add(job.job_id)
        await queue.ack(job.job_id, JobResult(job_id=job.job_id, status=JobStatus.COMPLETED))
    assert got == {"a", "b"}
    # c is for modelC, which the worker doesn't hold.
    assert await queue.dequeue("w", model_filter=["modelA", "modelB"]) is None


async def test_model_filter_empty_list_is_no_filter(queue):
    """Empty list / None both mean no filter."""
    await queue.enqueue(make_job("a", session_id="s-a", required_model="modelA"))
    job_none = await queue.dequeue("w", model_filter=None)
    assert job_none is not None
    await queue.ack(job_none.job_id, JobResult(job_id=job_none.job_id, status=JobStatus.COMPLETED))
    await queue.enqueue(make_job("b", session_id="s-b", required_model="modelA"))
    job_empty = await queue.dequeue("w", model_filter=[])
    assert job_empty is not None


async def test_get_queue_depth_list_filter(queue):
    await queue.enqueue(make_job("a", session_id="s-a", required_model="modelA"))
    await queue.enqueue(make_job("b", session_id="s-b", required_model="modelB"))
    await queue.enqueue(make_job("c", session_id="s-c", required_model="modelC"))
    assert await queue.get_queue_depth(model_filter=["modelA", "modelB"]) == 2
    assert await queue.get_queue_depth(model_filter=["modelA"]) == 1
    assert await queue.get_queue_depth(model_filter=[]) == 3
    assert await queue.get_queue_depth() == 3


async def test_worker_affinity_preferred(queue):
    # Affinity only matters across distinct sessions (within a session
    # FIFO is strict). Put the two jobs on different sessions so both
    # heads are eligible at dequeue time.
    await queue.enqueue(make_job("a", session_id="s-a", preferred_worker="w-1"))
    await queue.enqueue(make_job("b", session_id="s-b", preferred_worker="w-2"))
    # w-2 should get its preferred job first.
    job = await queue.dequeue("w-2")
    assert job.job_id == "b"
    # w-1 should get its preferred job.
    await queue.ack(job.job_id, JobResult(job_id=job.job_id, status=JobStatus.COMPLETED))
    job = await queue.dequeue("w-1")
    assert job.job_id == "a"


async def test_sticky_affinity_window(queue):
    """During the affinity window, only the preferred worker can dequeue."""
    await queue.enqueue(make_job("a", session_id="s-a", preferred_worker="w-1"))
    # w-2 should NOT see job "a" within the affinity window.
    job = await queue.dequeue("w-2")
    assert job is None
    # w-1 SHOULD see it.
    job = await queue.dequeue("w-1")
    assert job is not None
    assert job.job_id == "a"


async def test_sticky_affinity_expires():
    """After the affinity window, any worker can dequeue the job."""
    from hatchery.core.backends.queue.memory import _AFFINITY_WINDOW_S

    # Use a controllable clock: start at T=0, advance past the window.
    current_time = [0.0]
    q = InMemoryJobQueue(clock=lambda: current_time[0])
    await q.initialize()

    await q.enqueue(make_job("a", session_id="s-a", preferred_worker="w-1"))
    # At T=0: w-2 blocked by sticky window.
    job = await q.dequeue("w-2")
    assert job is None

    # Advance past the window.
    current_time[0] = _AFFINITY_WINDOW_S + 1
    job = await q.dequeue("w-2")
    assert job is not None
    assert job.job_id == "a"
    await q.close()


async def test_no_preferred_worker_always_visible(queue):
    """Jobs without preferred_worker are always visible to any worker."""
    await queue.enqueue(make_job("a", session_id="s-a"))
    job = await queue.dequeue("any-worker")
    assert job is not None
    assert job.job_id == "a"


async def test_ack_delivers_result(queue):
    await queue.enqueue(make_job("j1"))
    await queue.dequeue("w")
    result = JobResult(job_id="j1", status=JobStatus.COMPLETED, result=b"done")
    await queue.ack("j1", result)
    got = await queue.wait_for_result("j1", timeout=1.0)
    assert got.status == JobStatus.COMPLETED
    assert got.result == b"done"


async def test_wait_for_result_blocks_until_ack(queue):
    await queue.enqueue(make_job("j1"))
    await queue.dequeue("w")

    async def slow_ack():
        await asyncio.sleep(0.05)
        await queue.ack("j1", JobResult(job_id="j1", status=JobStatus.COMPLETED))

    await asyncio.gather(slow_ack(), queue.wait_for_result("j1", timeout=2.0))


async def test_wait_for_result_timeout(queue):
    await queue.enqueue(make_job("j1"))
    result = await queue.wait_for_result("j1", timeout=0.1)
    assert result.status == JobStatus.TIMED_OUT


async def test_nack_requeues_job(queue):
    await queue.enqueue(make_job("j1"))
    await queue.dequeue("w-1")
    await queue.nack("j1", "transient error")
    # Job should be re-queued.
    job2 = await queue.dequeue("w-2")
    assert job2 is not None
    assert job2.job_id == "j1"


async def test_nack_dead_letters_after_max_attempts(queue):
    await queue.enqueue(make_job("j1"))
    for _ in range(3):
        job = await queue.dequeue("w")
        assert job is not None
        await queue.nack("j1", "fail")
    # After 3 failures the waiting caller gets a FAILED result.
    result = await queue.wait_for_result("j1", timeout=0.1)
    assert result.status == JobStatus.FAILED


async def test_visibility_timeout_requeue():
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        await q.enqueue(make_job("j1"))
        job = await q.dequeue("w", visibility_timeout=0)
        assert job is not None
        # Immediately dequeue again — visibility timeout has already expired.
        await asyncio.sleep(0.01)
        job2 = await q.dequeue("w2")
        assert job2 is not None
        assert job2.job_id == "j1"
    finally:
        await q.close()


async def test_queue_depth(queue):
    await queue.enqueue(make_job("a", required_model="m1"))
    await queue.enqueue(make_job("b", required_model="m1"))
    await queue.enqueue(make_job("c", required_model="m2"))
    assert await queue.get_queue_depth() == 3
    assert await queue.get_queue_depth(model_filter="m1") == 2
    assert await queue.get_queue_depth(model_filter="m2") == 1


# ── Accumulation hard-pin ──────────────────────────────────────────


async def test_accum_pin_blocks_non_owner(queue):
    await queue.set_accum_pin("s1", "w-owner")
    await queue.enqueue(make_job("j1", session_id="s1"))
    # A peer must not see the pinned session.
    assert await queue.dequeue("w-peer") is None
    # The owner still can.
    job = await queue.dequeue("w-owner")
    assert job is not None
    assert job.job_id == "j1"


async def test_accum_pin_cleared(queue):
    await queue.set_accum_pin("s1", "w-owner")
    await queue.clear_accum_pin("s1")
    await queue.enqueue(make_job("j1", session_id="s1"))
    # Anyone can take it after the pin is cleared.
    job = await queue.dequeue("w-peer")
    assert job is not None


async def test_accum_pin_ttl_expires():
    # Drive the clock manually so the test isn't timing-sensitive.
    now = {"t": 1000.0}
    q = InMemoryJobQueue(clock=lambda: now["t"])
    await q.initialize()
    try:
        await q.set_accum_pin("s1", "w-owner", ttl_s=30.0)
        await q.enqueue(make_job("j1", session_id="s1"))
        assert await q.dequeue("w-peer") is None
        # Advance past the TTL.
        now["t"] += 31.0
        job = await q.dequeue("w-peer")
        assert job is not None
    finally:
        await q.close()


async def test_accum_pin_refresh_is_idempotent(queue):
    await queue.set_accum_pin("s1", "w-owner", ttl_s=10.0)
    await queue.set_accum_pin("s1", "w-owner", ttl_s=600.0)
    await queue.enqueue(make_job("j1", session_id="s1"))
    assert await queue.dequeue("w-peer") is None


async def test_clear_absent_pin_noop(queue):
    # Should not raise even if nothing is pinned.
    await queue.clear_accum_pin("nonexistent")


async def test_accum_pin_owner_survives_visibility_timeout():
    """Dead owner retry scenario: pin still blocks peers until TTL."""
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        await q.set_accum_pin("s1", "w-owner", ttl_s=600.0)
        await q.enqueue(make_job("j1", session_id="s1"))
        # Owner dequeues, then "dies" — visibility timeout expires.
        job = await q.dequeue("w-owner", visibility_timeout=0)
        assert job is not None
        await asyncio.sleep(0.01)
        # Peer may not pick it up — pin still held by w-owner.
        assert await q.dequeue("w-peer") is None
        # Recovery: clear pin (e.g. operator intervention), peer can retry.
        await q.clear_accum_pin("s1")
        recovered = await q.dequeue("w-peer")
        assert recovered is not None
        assert recovered.job_id == "j1"
    finally:
        await q.close()
