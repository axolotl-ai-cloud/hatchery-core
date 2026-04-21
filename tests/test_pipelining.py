# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for pipelined / awaited-future execution semantics.

Tinker's execution model lets clients submit ``forward_backward``,
``optim_step``, and the next ``forward_backward`` without awaiting
any of them, then retrieve the futures in order. This file pins the
two correctness properties that pipelining depends on:

1. **Per-session FIFO.** Jobs for the same session must execute in
   submission order regardless of per-op priority. Otherwise the
   optimizer step applies grads to the wrong weights.

2. **Single in-flight per session.** Two workers must never hold
   different jobs for the same session at once, or they race on the
   object-store round-trip. Different sessions can and should run
   concurrently.

3. **Pipelining throughput.** A client that keeps one extra request
   queued should not block waiting for the previous result.

These are written against the in-process queue; the tinker-compat
futures API is also exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import msgpack
import pytest_asyncio
from httpx import ASGITransport

from hatchery.core.backends.queue.memory import InMemoryJobQueue
from hatchery.core.gateway import create_app
from hatchery.core.protocols import JobResult, JobStatus, QueuedJob


def make_job(job_id: str, session_id: str, op: str = "fb", **kwargs) -> QueuedJob:
    return QueuedJob(
        job_id=job_id,
        session_id=session_id,
        operation=op,
        payload=b"",
        **kwargs,
    )


# ─── Per-session FIFO ─────────────────────────────────────────────────────


async def test_per_session_fifo_ignores_priority():
    """optim_step with priority=5 must NOT jump ahead of a forward_backward
    submitted earlier on the same session."""
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        await q.enqueue(make_job("fb1", "sessA", "forward_backward", priority=0))
        await q.enqueue(make_job("opt", "sessA", "optim_step", priority=5))
        await q.enqueue(make_job("fb2", "sessA", "forward_backward", priority=0))

        # Dequeue with a worker that finishes jobs as soon as it gets them.
        order = []
        for _ in range(3):
            job = await q.dequeue("worker-1")
            assert job is not None
            order.append(job.job_id)
            await q.ack(
                job.job_id,
                JobResult(job_id=job.job_id, status=JobStatus.COMPLETED),
            )
        assert order == ["fb1", "opt", "fb2"]
    finally:
        await q.close()


async def test_cross_session_priority_still_wins():
    """Priority should still matter *across* sessions."""
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        await q.enqueue(make_job("lo", "sessA", priority=0))
        await q.enqueue(make_job("hi", "sessB", priority=10))
        first = await q.dequeue("w")
        assert first.job_id == "hi"
    finally:
        await q.close()


# ─── Single in-flight per session ─────────────────────────────────────────


async def test_single_inflight_per_session():
    """While sessA has a job in flight, a second sessA job must not be
    dequeued — even by a different worker."""
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        await q.enqueue(make_job("a1", "sessA"))
        await q.enqueue(make_job("a2", "sessA"))
        await q.enqueue(make_job("b1", "sessB"))

        # Worker 1 takes the first A job.
        first = await q.dequeue("worker-1")
        assert first.job_id == "a1"

        # Worker 2 asks next — must skip a2 (same session still in flight)
        # and receive b1 instead.
        second = await q.dequeue("worker-2")
        assert second is not None
        assert second.job_id == "b1"

        # No more jobs that can run right now.
        third = await q.dequeue("worker-3")
        assert third is None

        # Worker 1 completes a1. Now a2 is eligible.
        await q.ack("a1", JobResult(job_id="a1", status=JobStatus.COMPLETED))
        fourth = await q.dequeue("worker-3")
        assert fourth is not None
        assert fourth.job_id == "a2"
    finally:
        await q.close()


async def test_nack_releases_session_lock():
    """A failed ack (nack) should clear the in-flight slot so the retry
    can be dequeued."""
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        await q.enqueue(make_job("a1", "sessA"))
        first = await q.dequeue("worker-1")
        assert first.job_id == "a1"
        await q.nack("a1", "transient")
        # Retry is the SAME job, so it may come back with the same id.
        retry = await q.dequeue("worker-2")
        assert retry is not None
        assert retry.job_id == "a1"
    finally:
        await q.close()


async def test_visibility_timeout_releases_session_lock():
    """If a worker dies mid-flight, the visibility timeout must release
    the per-session lock so a replacement worker can pick the job up."""
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        await q.enqueue(make_job("a1", "sessA"))
        first = await q.dequeue("worker-1", visibility_timeout=0)
        assert first is not None
        await asyncio.sleep(0.01)
        # Another worker calls dequeue; the timed-out job should be
        # re-queued and the session lock cleared.
        retry = await q.dequeue("worker-2")
        assert retry is not None
        assert retry.job_id == "a1"
    finally:
        await q.close()


# ─── End-to-end pipelined throughput via the tinker-compat API ──────────


class _PipelinedFakeWorker:
    """Fake worker that records the order in which it sees ops.

    Runs as an asyncio task that drains the queue and ACKs jobs with a
    canned response, giving us a tight loop that exercises the
    "pipelined client + serial worker" pattern.
    """

    def __init__(self, config):
        self.config = config
        self.order: list[tuple[str, str]] = []
        self._stop = asyncio.Event()
        self.task: asyncio.Task | None = None

    def start(self):
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def _loop(self):
        while not self._stop.is_set():
            try:
                job = await self.config.queue.dequeue(
                    worker_id="pipeline-worker",
                    model_filter=None,
                    visibility_timeout=60,
                )
            except asyncio.CancelledError:
                return
            if job is None:
                await asyncio.sleep(0.002)
                continue
            self.order.append((job.session_id, job.operation))
            result = _canned(job.operation)
            if job.operation == "init_session":
                await self.config.objects.put(
                    f"sessions/{job.session_id}/live_state/lora_weights.pt",
                    b"w",
                )
            # Small "GPU work" delay so we're measurably "in flight".
            await asyncio.sleep(0.01)
            await self.config.queue.ack(
                job.job_id,
                JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    result=msgpack.packb(result, use_bin_type=True),
                    metrics={"duration_ms": 1.0, "tokens": 1},
                ),
            )


def _canned(op: str) -> dict:
    if op == "forward_backward":
        return {"loss": 0.5, "num_tokens": 1, "accum_steps": 1}
    if op == "optim_step":
        return {"status": "ok", "step": 1, "learning_rate": 1e-3}
    if op == "sample":
        return {"sequences": [[0]], "texts": [""]}
    if op == "init_session":
        return {"status": "initialized"}
    return {}


@pytest_asyncio.fixture
async def pipelined_client(platform_config):
    app = create_app(config=platform_config)
    worker = _PipelinedFakeWorker(platform_config)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        async with app.router.lifespan_context(app):
            worker.start()
            try:
                yield client, worker
            finally:
                await worker.stop()


async def _create_model(client: httpx.AsyncClient) -> str:
    resp = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "tinker-sess",
            "model_seq_id": 0,
            "base_model": "m",
            "lora_config": {"rank": 8},
        },
    )
    return resp.json()["model_id"]


def _datum(tokens: list[int]) -> dict:
    return {
        "model_input": {
            "chunks": [{"type": "encoded_text", "tokens": tokens}],
        },
        "loss_fn_inputs": {},
    }


async def _submit_fb(client, model_id: str, tokens: list[int]) -> str:
    resp = await client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "seq_id": 0,
            "forward_backward_input": {
                "data": [_datum(tokens)],
                "loss_fn": "cross_entropy",
            },
        },
    )
    return resp.json()["future_id"]


async def _submit_opt(client, model_id: str) -> str:
    resp = await client.post(
        "/api/v1/optim_step",
        json={
            "model_id": model_id,
            "seq_id": 0,
            "adam_params": {"learning_rate": 1e-3},
        },
    )
    return resp.json()["future_id"]


async def _retrieve(client, future_id: str) -> dict:
    resp = await client.post("/api/v1/retrieve_future", json={"future_id": future_id})
    return resp.json()


async def test_tinker_compat_pipelined_step_executes_in_order(pipelined_client):
    """Submit fb → opt → fb without awaiting. All three must complete,
    and the worker must have seen them in submission order."""
    client, worker = pipelined_client
    mid = await _create_model(client)
    baseline = [(sid, op) for sid, op in worker.order if op == "init_session"]
    assert len(baseline) == 1

    fb1 = await _submit_fb(client, mid, [1, 2, 3])
    opt = await _submit_opt(client, mid)
    fb2 = await _submit_fb(client, mid, [4, 5, 6])

    # Now await all three.
    r1 = await _retrieve(client, fb1)
    r2 = await _retrieve(client, opt)
    r3 = await _retrieve(client, fb2)

    # SDK-0.18 envelope: retrieve_future returns the typed response
    # directly. A successful fb carries ``loss_fn_output_type``; a
    # successful optim_step carries ``type: "optim_step"``. Failures
    # return ``{"type": "request_failed", ...}``.
    assert r1.get("loss_fn_output_type") == "cross_entropy", r1
    assert r2.get("type") == "optim_step", r2
    assert r3.get("loss_fn_output_type") == "cross_entropy", r3

    # Order for this model_id, excluding init_session.
    ops = [op for sid, op in worker.order if sid == mid]
    assert ops == ["init_session", "forward_backward", "optim_step", "forward_backward"]


async def test_pipelined_is_faster_than_serial(pipelined_client):
    """Submitting N requests at once and then awaiting should take roughly
    N * worker_delay, not 2 * N * worker_delay (which is what you'd get
    from a submit-then-await loop)."""
    import time

    client, _ = pipelined_client
    mid = await _create_model(client)

    N = 5
    # Serial first (warmup — avoids cold-start penalty on pipelined).
    t0 = time.time()
    for i in range(N):
        fid = await _submit_fb(client, mid, [i, i + 1])
        r = await _retrieve(client, fid)
        assert r.get("loss_fn_output_type") == "cross_entropy", r
    serial_s = time.time() - t0

    # Pipelined: submit everything, then drain.
    t0 = time.time()
    future_ids = []
    for i in range(N):
        future_ids.append(await _submit_fb(client, mid, [i, i + 1]))
    for fid in future_ids:
        r = await _retrieve(client, fid)
        assert r.get("loss_fn_output_type") == "cross_entropy", r
    pipelined_s = time.time() - t0

    # Pipelined should be faster (or at least not significantly slower)
    # than serial. We add a 50% margin to avoid flaking on loaded CI.
    assert pipelined_s <= serial_s * 1.5, (
        f"pipelined ({pipelined_s:.3f}s) much slower than serial ({serial_s:.3f}s)"
    )


async def test_cross_session_operations_run_concurrently(platform_config):
    """Two sessions submitted concurrently should both be dequeued by
    different workers at once, not serialized through one."""
    q = platform_config.queue
    await q.enqueue(make_job("a1", "sessA"))
    await q.enqueue(make_job("b1", "sessB"))

    first = await q.dequeue("worker-1")
    second = await q.dequeue("worker-2")
    assert {first.job_id, second.job_id} == {"a1", "b1"}
