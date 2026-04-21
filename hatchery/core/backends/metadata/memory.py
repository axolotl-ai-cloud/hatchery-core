# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""In-memory metadata store — for tests and solo-dev unified mode."""

from __future__ import annotations

import asyncio
import copy
from typing import Optional

from hatchery.core.protocols import (
    CheckpointRecord,
    JobRecord,
    JobStatus,
    SessionRecord,
    SessionStatus,
)


class InMemoryMetadataStore:
    """Single-process metadata store. Snapshots on read to prevent mutation."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._checkpoints: dict[str, CheckpointRecord] = {}  # checkpoint_id -> record
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    # ─── Sessions ───────────────────────────────────────────

    async def create_session(self, record: SessionRecord) -> None:
        async with self._lock:
            if record.session_id in self._sessions:
                raise ValueError(f"Session {record.session_id} already exists")
            self._sessions[record.session_id] = copy.deepcopy(record)

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        async with self._lock:
            rec = self._sessions.get(session_id)
            return copy.deepcopy(rec) if rec else None

    async def update_session(self, session_id: str, **kwargs) -> None:
        async with self._lock:
            rec = self._sessions.get(session_id)
            if rec is None:
                raise KeyError(session_id)
            for k, v in kwargs.items():
                if not hasattr(rec, k):
                    raise AttributeError(f"SessionRecord has no field '{k}'")
                setattr(rec, k, v)

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionRecord]:
        async with self._lock:
            rows = list(self._sessions.values())
        if user_id is not None:
            rows = [r for r in rows if r.user_id == user_id]
        if status is not None:
            rows = [r for r in rows if r.status == status]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return [copy.deepcopy(r) for r in rows[offset : offset + limit]]

    async def list_sessions_by_worker(self, worker_id: str) -> list[SessionRecord]:
        async with self._lock:
            return [
                copy.deepcopy(r) for r in self._sessions.values() if r.last_worker_id == worker_id
            ]

    # ─── Jobs ───────────────────────────────────────────────

    async def create_job(self, record: JobRecord) -> None:
        async with self._lock:
            if record.job_id in self._jobs:
                raise ValueError(f"Job {record.job_id} already exists")
            self._jobs[record.job_id] = copy.deepcopy(record)

    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        async with self._lock:
            rec = self._jobs.get(job_id)
            return copy.deepcopy(rec) if rec else None

    async def update_job(self, job_id: str, **kwargs) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            for k, v in kwargs.items():
                if not hasattr(rec, k):
                    raise AttributeError(f"JobRecord has no field '{k}'")
                setattr(rec, k, v)

    async def get_pending_jobs(self, session_id: str) -> list[JobRecord]:
        pending_states = {JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING}
        async with self._lock:
            rows = [
                copy.deepcopy(j)
                for j in self._jobs.values()
                if j.session_id == session_id and j.status in pending_states
            ]
        rows.sort(key=lambda j: j.created_at)
        return rows

    # ─── Checkpoints ───────────────────────────────────────

    async def create_checkpoint(self, record: CheckpointRecord) -> None:
        async with self._lock:
            if record.checkpoint_id in self._checkpoints:
                raise ValueError(f"Checkpoint {record.checkpoint_id} already exists")
            self._checkpoints[record.checkpoint_id] = copy.deepcopy(record)

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Optional[CheckpointRecord]:
        async with self._lock:
            rec = self._checkpoints.get(checkpoint_id)
            if rec and rec.session_id == session_id:
                return copy.deepcopy(rec)
            return None

    async def list_checkpoints(
        self, session_id: str, checkpoint_type: Optional[str] = None
    ) -> list[CheckpointRecord]:
        async with self._lock:
            rows = [r for r in self._checkpoints.values() if r.session_id == session_id]
        if checkpoint_type is not None:
            rows = [r for r in rows if r.checkpoint_type == checkpoint_type]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return [copy.deepcopy(r) for r in rows]

    async def update_checkpoint(self, session_id: str, checkpoint_id: str, **kwargs) -> None:
        async with self._lock:
            rec = self._checkpoints.get(checkpoint_id)
            if rec is None or rec.session_id != session_id:
                raise KeyError(checkpoint_id)
            for k, v in kwargs.items():
                if not hasattr(rec, k):
                    raise AttributeError(f"CheckpointRecord has no field '{k}'")
                setattr(rec, k, v)

    async def delete_checkpoint(self, session_id: str, checkpoint_id: str) -> None:
        async with self._lock:
            rec = self._checkpoints.get(checkpoint_id)
            if rec and rec.session_id == session_id:
                del self._checkpoints[checkpoint_id]

    async def get_expired_checkpoints(self, now: Optional[float] = None) -> list[CheckpointRecord]:
        import time as _time

        ts = now if now is not None else _time.time()
        async with self._lock:
            return [
                copy.deepcopy(r)
                for r in self._checkpoints.values()
                if r.expires_at is not None and r.expires_at <= ts
            ]

    # ─── Metrics Queries ────────────────────────────────────

    async def get_session_step_history(self, session_id: str, last_n: int = 50) -> list[dict]:
        async with self._lock:
            rows = [
                j
                for j in self._jobs.values()
                if j.session_id == session_id
                and j.status == JobStatus.COMPLETED
                and j.gpu_time_ms is not None
            ]
        rows.sort(key=lambda j: j.completed_at or 0, reverse=True)
        return [
            {
                "job_id": j.job_id,
                "operation": j.operation,
                "duration_ms": j.gpu_time_ms,
                "tokens": j.tokens_processed,
                "completed_at": j.completed_at,
            }
            for j in rows[:last_n]
        ]

    async def get_active_session_count_by_worker(self) -> dict[str, int]:
        async with self._lock:
            counts: dict[str, int] = {}
            for r in self._sessions.values():
                if r.status != SessionStatus.ACTIVE or r.last_worker_id is None:
                    continue
                counts[r.last_worker_id] = counts.get(r.last_worker_id, 0) + 1
            return counts
