# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import msgpack
import pytest

from hatchery.core.distributed import DistributedRuntime
from hatchery.core.parallel import ParallelConfig
from hatchery.core.protocols import JobResult, JobStatus, QueuedJob
from hatchery.core.worker import GPUWorker, _SessionRuntime


class _FakeBus:
    def __init__(self, incoming: list[dict | None] | None = None) -> None:
        self.incoming = list(incoming or [])
        self.broadcasts: list[dict | None] = []
        self.errors: list[str | None] = []

    def broadcast(self, value):
        if value is None:
            return self.incoming.pop(0)
        self.broadcasts.append(value)
        return value

    def gather_errors(self, error):
        self.errors.append(error)
        return [error, None]


class _RecordingState:
    def __init__(self) -> None:
        self.local = self
        self.puts: list[str] = []
        self.dirty: list[str] = []
        self.flushes: list[str] = []

    async def put(self, key: str, data: bytes) -> None:
        self.puts.append(key)

    def mark_dirty(self, session_id: str) -> None:
        self.dirty.append(session_id)

    async def flush(self, session_id: str, *, timeout=None) -> None:
        self.flushes.append(session_id)


class _RecordingCompute:
    def __init__(self) -> None:
        self.registered = []

    async def register_worker(self, info) -> None:
        self.registered.append(info)

    async def list_workers(self):
        return list(self.registered)

    async def get_worker(self, worker_id: str):
        return next((info for info in self.registered if info.worker_id == worker_id), None)

    async def health_check(self, worker_id: str) -> bool:
        return await self.get_worker(worker_id) is not None


class _RecordingQueue:
    def __init__(self) -> None:
        self.dequeues: list[str] = []
        self.acks: list[str] = []
        self.nacks: list[str] = []

    async def dequeue(self, worker_id: str, model_filter=None, visibility_timeout: int = 300):
        self.dequeues.append(worker_id)
        return None

    async def ack(self, job_id: str, result: JobResult) -> None:
        self.acks.append(job_id)

    async def nack(self, job_id: str, error: str) -> None:
        self.nacks.append(job_id)


class _RecordingMetadata:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict]] = []

    async def update_session(self, session_id: str, **kwargs) -> None:
        self.updates.append((session_id, kwargs))


class _RecordingSessionRegistry:
    def __init__(self) -> None:
        self.sets: list[tuple[str, str]] = []

    async def set(self, session_id: str, worker_id: str) -> None:
        self.sets.append((session_id, worker_id))


class _FakeDistributedWorker(GPUWorker):
    def __init__(
        self,
        *args,
        dp_rank: int,
        bus: _FakeBus,
        global_rank: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, load_model=False, parallel=ParallelConfig(dp_degree=2), **kwargs)
        global_rank = dp_rank if global_rank is None else global_rank
        self._distributed_runtime = DistributedRuntime(
            global_rank=global_rank,
            local_rank=global_rank,
            dp_rank=dp_rank,
            world_size=2,
            dp_world_size=2,
            device=None,
        )
        self._command_bus = bus
        self.executed: list[str] = []

    async def _execute_job(self, job: QueuedJob) -> JobResult:
        self.executed.append(job.job_id)
        return JobResult(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            result=msgpack.packb({"ok": True}, use_bin_type=True),
            metrics={"tokens": 3},
        )


def _job(job_id: str = "job-1") -> QueuedJob:
    return QueuedJob(
        job_id=job_id,
        session_id="session-1",
        operation="forward_backward",
        payload=msgpack.packb({"data": [{"input_ids": [1], "labels": [1]}]}, use_bin_type=True),
        required_model="base",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_distributed_coordinator_dequeues_broadcasts_and_acks(platform_config):
    await platform_config.queue.enqueue(_job())
    bus = _FakeBus()
    worker = _FakeDistributedWorker(
        "worker-0",
        "base",
        platform_config,
        device="cpu",
        dp_rank=0,
        bus=bus,
    )

    await worker._run_coordinator(max_jobs=1)

    assert worker.executed == ["job-1"]
    assert [msg["type"] for msg in bus.broadcasts] == ["job", "shutdown"]
    result = await platform_config.queue.wait_for_result("job-1", timeout=0.1)
    assert result.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_distributed_follower_executes_broadcast_job_without_dequeue_or_ack(platform_config):
    queued = _job("queued-job")
    await platform_config.queue.enqueue(queued)
    incoming = [
        {"type": "idle"},
        {
            "type": "job",
            "job_id": "broadcast-job",
            "session_id": "session-1",
            "operation": "forward_backward",
            "payload": queued.payload,
            "required_model": "base",
            "required_cp_degree": 1,
        },
        {"type": "shutdown"},
    ]
    worker = _FakeDistributedWorker(
        "worker-1",
        "base",
        platform_config,
        device="cpu",
        dp_rank=1,
        bus=_FakeBus(incoming),
    )

    assert await worker.process_next(timeout=0.01) is False
    await worker._run_follower(max_jobs=None)

    assert worker.executed == ["broadcast-job"]
    still_queued = await platform_config.queue.dequeue("rank-0", model_filter="base")
    assert still_queued is not None
    assert still_queued.job_id == "queued-job"


def test_worker_dp_batching_uses_runtime_rank(platform_config):
    worker = _FakeDistributedWorker(
        "worker-1",
        "base",
        platform_config,
        device="cpu",
        dp_rank=1,
        bus=_FakeBus(),
    )
    worker.parallel.batch_strategy = "split"

    data = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]

    assert worker._allocate_batch(data) == [{"id": 2}, {"id": 3}]


@pytest.mark.asyncio
async def test_follower_session_init_marker_stays_local(platform_config):
    worker = _FakeDistributedWorker(
        "worker-1",
        "base",
        platform_config,
        device="cpu",
        dp_rank=1,
        bus=_FakeBus(),
    )
    state = _RecordingState()
    worker._state = state
    runtime = _SessionRuntime(session_id="session-1", training_mode="full_param")

    await worker._save_fp_init_marker("session-1", runtime)

    assert state.puts == ["sessions/session-1/live_state/session_meta.json"]
    assert state.dirty == []
    assert state.flushes == []


@pytest.mark.asyncio
async def test_global_nonzero_dp_zero_rank_does_not_touch_external_state(platform_config):
    platform_config.compute = _RecordingCompute()
    platform_config.queue = _RecordingQueue()
    platform_config.metadata = _RecordingMetadata()
    platform_config.session_registry = _RecordingSessionRegistry()
    worker = _FakeDistributedWorker(
        "worker-1",
        "base",
        platform_config,
        device="cpu",
        dp_rank=0,
        global_rank=1,
        bus=_FakeBus(),
    )
    state = _RecordingState()
    worker._state = state
    runtime = _SessionRuntime(session_id="session-1", training_mode="full_param")
    runtime.meta["accum_steps"] = 1

    await worker.register()
    assert await worker.process_next(timeout=0.01) is False
    result = JobResult(
        job_id="job-1",
        status=JobStatus.COMPLETED,
        result=msgpack.packb({"ok": True}, use_bin_type=True),
        metrics={"tokens": 1},
    )
    await worker._ack_completed_job(_job(), result, t0=0.0)
    await worker._nack_failed_job(_job(), "RuntimeError: boom")
    await worker._save_session_to_store("session-1", runtime, sync_remote=True)
    await worker._run_coordinator(max_jobs=1)

    assert platform_config.compute.registered == []
    assert platform_config.queue.dequeues == []
    assert platform_config.queue.acks == []
    assert platform_config.queue.nacks == []
    assert platform_config.metadata.updates == []
    assert platform_config.session_registry.sets == []
    assert state.dirty == []
    assert state.flushes == []
