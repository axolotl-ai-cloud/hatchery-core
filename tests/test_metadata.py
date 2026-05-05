# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for metadata store backends."""

from __future__ import annotations

import time

import pytest
import pytest_asyncio

from hatchery.core.backends.metadata.memory import InMemoryMetadataStore
from hatchery.core.protocols import (
    JobRecord,
    JobStatus,
    SessionRecord,
    SessionStatus,
)

# SQLite metadata lives in hosted — only exercise it when installed.
try:
    from hatchery.core.backends.metadata.sqlite import SQLiteMetadataStore

    _STORE_PARAMS = ["memory", "sqlite"]
except ImportError:
    SQLiteMetadataStore = None  # type: ignore[assignment]
    _STORE_PARAMS = ["memory"]


@pytest_asyncio.fixture(params=_STORE_PARAMS)
async def store(request, tmp_path):
    if request.param == "memory":
        s = InMemoryMetadataStore()
    else:
        s = SQLiteMetadataStore(path=str(tmp_path / "m.db"))
    await s.initialize()
    yield s
    await s.close()


def _sample_session(session_id="s1", user_id="u1"):
    return SessionRecord(
        session_id=session_id,
        user_id=user_id,
        base_model="Qwen/Qwen2-0.5B",
        lora_rank=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        total_steps=0,
        accum_steps=0,
        created_at=time.time(),
        last_accessed=time.time(),
        status=SessionStatus.ACTIVE,
        state_prefix=f"sessions/{session_id}/live_state",
    )


def _sample_job(job_id="j1", session_id="s1", user_id="u1"):
    return JobRecord(
        job_id=job_id,
        session_id=session_id,
        user_id=user_id,
        operation="forward_backward",
        status=JobStatus.QUEUED,
        created_at=time.time(),
    )


async def test_session_crud(store):
    rec = _sample_session()
    await store.create_session(rec)

    fetched = await store.get_session("s1")
    assert fetched.user_id == "u1"
    assert fetched.target_modules == ["q_proj", "v_proj"]

    await store.update_session("s1", total_steps=5, status=SessionStatus.SUSPENDED)
    updated = await store.get_session("s1")
    assert updated.total_steps == 5
    assert updated.status == SessionStatus.SUSPENDED


async def test_list_sessions_filter(store):
    await store.create_session(_sample_session("s1", user_id="a"))
    await store.create_session(_sample_session("s2", user_id="a"))
    await store.create_session(_sample_session("s3", user_id="b"))

    rows = await store.list_sessions(user_id="a")
    assert {r.session_id for r in rows} == {"s1", "s2"}

    await store.update_session("s2", status=SessionStatus.SUSPENDED)
    rows = await store.list_sessions(user_id="a", status=SessionStatus.ACTIVE)
    assert {r.session_id for r in rows} == {"s1"}


async def test_list_sessions_by_worker(store):
    await store.create_session(_sample_session("s1"))
    await store.update_session("s1", last_worker_id="w-42")
    rows = await store.list_sessions_by_worker("w-42")
    assert len(rows) == 1
    assert rows[0].session_id == "s1"


async def test_job_crud(store):
    await store.create_session(_sample_session())
    await store.create_job(_sample_job())

    job = await store.get_job("j1")
    assert job.operation == "forward_backward"

    await store.update_job(
        "j1",
        status=JobStatus.COMPLETED,
        completed_at=time.time(),
        gpu_time_ms=123.4,
        tokens_processed=256,
    )
    job = await store.get_job("j1")
    assert job.status == JobStatus.COMPLETED
    assert job.gpu_time_ms == pytest.approx(123.4)


async def test_pending_jobs(store):
    await store.create_session(_sample_session())
    await store.create_job(_sample_job("j1"))
    await store.create_job(_sample_job("j2"))
    await store.update_job("j2", status=JobStatus.COMPLETED)

    pending = await store.get_pending_jobs("s1")
    assert {j.job_id for j in pending} == {"j1"}


async def test_step_history(store):
    await store.create_session(_sample_session())
    for i in range(3):
        await store.create_job(_sample_job(f"j{i}"))
        await store.update_job(
            f"j{i}",
            status=JobStatus.COMPLETED,
            gpu_time_ms=100 + i,
            tokens_processed=10,
            completed_at=time.time() + i,
        )
    history = await store.get_session_step_history("s1", last_n=2)
    assert len(history) == 2
    # Most recent first.
    assert history[0]["duration_ms"] >= history[1]["duration_ms"]


async def test_active_session_count_by_worker(store):
    await store.create_session(_sample_session("s1"))
    await store.create_session(_sample_session("s2"))
    await store.update_session("s1", last_worker_id="w-1")
    await store.update_session("s2", last_worker_id="w-1")
    counts = await store.get_active_session_count_by_worker()
    assert counts.get("w-1") == 2


async def test_update_session_rejects_unknown_field(store):
    await store.create_session(_sample_session())
    with pytest.raises(AttributeError):
        await store.update_session("s1", bogus_field=42)


async def test_create_duplicate_session(store):
    await store.create_session(_sample_session())
    with pytest.raises((ValueError, Exception)):
        await store.create_session(_sample_session())
