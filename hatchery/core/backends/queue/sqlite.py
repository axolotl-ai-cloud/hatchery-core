# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SQLite-backed job queue.

Same invariants as ``InMemoryJobQueue`` (per-session FIFO, single
in-flight per session, priority across sessions, visibility timeout)
but durable and cross-process. Two processes pointing at the same
``.db`` file can enqueue and dequeue concurrently — the gateway runs
in one process and the worker(s) run in another.

Why SQLite: it's the lowest-friction cross-process store available on
every dev box. Postgres is the right answer for production (and we
ship ``PostgresMetadataStore`` for metadata already), but adding a
Postgres queue means standing up Postgres just to run the local
dev loop, and that's a lot of yaks to shave when a single ``.db``
file works fine.

Concurrency model:
* WAL mode + NORMAL sync → multi-reader / single-writer, crash-safe.
* IMMEDIATE transaction on every mutating op so two writers can't
  select the same head simultaneously.
* ``wait_for_result`` polls every 25ms. SQLite has no pub/sub, so
  polling is the only option — in practice workers finish jobs in
  hundreds of milliseconds, so the overhead is negligible.

Schema is created lazily on ``initialize``. Result BLOBs are stored
inline in the row; if you need to send multi-megabyte results, push
them through the object store and store only the key here.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from hatchery.core.protocols import JobResult, JobStatus, QueuedJob

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue_jobs (
    job_id              TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    operation           TEXT NOT NULL,
    payload             BLOB NOT NULL,
    priority            INTEGER NOT NULL DEFAULT 0,
    preferred_worker    TEXT,
    required_model      TEXT,
    estimated_duration_ms REAL,
    user_id             TEXT,
    enqueued_at         REAL NOT NULL,
    seq                 INTEGER NOT NULL,
    state               TEXT NOT NULL,
    worker_id           TEXT,
    visibility_deadline REAL,
    attempts            INTEGER NOT NULL DEFAULT 0,
    result              BLOB,
    error               TEXT,
    metrics             TEXT,
    completed_at        REAL
);
CREATE INDEX IF NOT EXISTS idx_queue_state_priority ON queue_jobs(state, priority DESC, seq);
CREATE INDEX IF NOT EXISTS idx_queue_session_state ON queue_jobs(session_id, state, seq);
CREATE TABLE IF NOT EXISTS queue_seq (
    id INTEGER PRIMARY KEY CHECK (id = 0),
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO queue_seq (id, value) VALUES (0, 0);
CREATE TABLE IF NOT EXISTS queue_accum_pins (
    session_id  TEXT PRIMARY KEY,
    worker_id   TEXT NOT NULL,
    expires_at  REAL NOT NULL
);
"""


class SQLiteJobQueue:
    def __init__(
        self,
        path: str,
        *,
        poll_interval: float = 0.025,
        max_attempts: int = 3,
    ) -> None:
        self.path = path
        self.poll_interval = poll_interval
        self.max_attempts = max_attempts
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────

    async def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path, isolation_level=None)
        # WAL + NORMAL gives us multi-reader / single-writer across
        # processes with crash-safety that's good enough for a job queue.
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteJobQueue not initialized")
        return self._db

    async def _next_seq(self) -> int:
        # Atomic counter bump. Done inside its own immediate txn.
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            async with self.db.execute("SELECT value FROM queue_seq WHERE id = 0") as c:
                row = await c.fetchone()
            value = (row[0] if row else 0) + 1
            await self.db.execute("UPDATE queue_seq SET value = ? WHERE id = 0", (value,))
            await self.db.execute("COMMIT")
            return value
        except BaseException:
            await self.db.execute("ROLLBACK")
            raise

    # ── Core ops ─────────────────────────────────────────────

    async def enqueue(self, job: QueuedJob) -> None:
        if job.enqueued_at is None:
            job.enqueued_at = time.time()
        async with self._lock:
            seq = await self._next_seq()
            await self.db.execute(
                """
                INSERT INTO queue_jobs (
                    job_id, session_id, operation, payload, priority,
                    preferred_worker, required_model, estimated_duration_ms,
                    user_id, enqueued_at, seq, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')
                """,
                (
                    job.job_id,
                    job.session_id,
                    job.operation,
                    job.payload,
                    job.priority,
                    job.preferred_worker,
                    job.required_model,
                    job.estimated_duration_ms,
                    job.user_id,
                    job.enqueued_at,
                    seq,
                ),
            )

    async def _release_expired_pins(self) -> None:
        """Drop accum pins whose TTL has lapsed."""
        await self.db.execute(
            "DELETE FROM queue_accum_pins WHERE expires_at <= ?",
            (time.time(),),
        )

    async def set_accum_pin(self, session_id: str, worker_id: str, *, ttl_s: float = 600.0) -> None:
        async with self._lock:
            await self.db.execute(
                """
                INSERT INTO queue_accum_pins (session_id, worker_id, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    worker_id = excluded.worker_id,
                    expires_at = excluded.expires_at
                """,
                (session_id, worker_id, time.time() + ttl_s),
            )

    async def clear_accum_pin(self, session_id: str) -> None:
        async with self._lock:
            await self.db.execute(
                "DELETE FROM queue_accum_pins WHERE session_id = ?",
                (session_id,),
            )

    async def _release_expired(self) -> None:
        """Release visibility-timeout expired inflight jobs."""
        now = time.time()
        await self.db.execute(
            """
            UPDATE queue_jobs
            SET state = 'queued', worker_id = NULL, visibility_deadline = NULL
            WHERE state = 'inflight' AND visibility_deadline IS NOT NULL
              AND visibility_deadline <= ?
            """,
            (now,),
        )

    async def dequeue(
        self,
        worker_id: str,
        model_filter: Optional[str] = None,
        visibility_timeout: int = 300,
    ) -> Optional[QueuedJob]:
        async with self._lock:
            # Must be its own immediate txn so two workers don't pick
            # the same head concurrently.
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                await self._release_expired()
                await self._release_expired_pins()

                params: list = []
                extra_where = ""
                if model_filter is not None:
                    extra_where = " AND (j.required_model IS NULL OR j.required_model = ?)"
                    params.append(model_filter)

                # Pick the head of each session that has no in-flight job
                # AND no hard-pin owned by a different worker, ordered by
                # priority desc then seq asc, pick one.
                query = f"""
                    WITH session_heads AS (
                        SELECT session_id, MIN(seq) AS head_seq
                        FROM queue_jobs
                        WHERE state = 'queued'
                        GROUP BY session_id
                    )
                    SELECT j.*
                    FROM queue_jobs j
                    INNER JOIN session_heads sh
                        ON sh.session_id = j.session_id AND sh.head_seq = j.seq
                    WHERE j.state = 'queued'
                      AND NOT EXISTS (
                          SELECT 1 FROM queue_jobs j2
                          WHERE j2.session_id = j.session_id
                            AND j2.state = 'inflight'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM queue_accum_pins p
                          WHERE p.session_id = j.session_id
                            AND p.worker_id <> ?
                      )
                      {extra_where}
                    ORDER BY
                        j.priority DESC,
                        (CASE WHEN j.preferred_worker = ? THEN 0 ELSE 1 END),
                        j.seq ASC
                    LIMIT 1
                """
                params.insert(0, worker_id)
                params.append(worker_id)
                async with self.db.execute(query, params) as cur:
                    row = await cur.fetchone()

                if row is None:
                    await self.db.execute("COMMIT")
                    return None

                job = _row_to_job(row)
                deadline = time.time() + visibility_timeout
                await self.db.execute(
                    """
                    UPDATE queue_jobs
                    SET state = 'inflight',
                        worker_id = ?,
                        visibility_deadline = ?
                    WHERE job_id = ?
                    """,
                    (worker_id, deadline, job.job_id),
                )
                await self.db.execute("COMMIT")
                return job
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise

    async def ack(self, job_id: str, result: JobResult) -> None:
        async with self._lock:
            metrics = json.dumps(result.metrics) if result.metrics else None
            await self.db.execute(
                """
                UPDATE queue_jobs
                SET state = 'completed',
                    result = ?,
                    error = NULL,
                    metrics = ?,
                    completed_at = ?,
                    worker_id = NULL,
                    visibility_deadline = NULL
                WHERE job_id = ?
                """,
                (result.result, metrics, time.time(), job_id),
            )

    async def nack(self, job_id: str, error: str) -> None:
        async with self._lock:
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                async with self.db.execute(
                    "SELECT attempts FROM queue_jobs WHERE job_id = ? AND state = 'inflight'",
                    (job_id,),
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    await self.db.execute("COMMIT")
                    return
                attempts = (row[0] or 0) + 1
                if attempts >= self.max_attempts:
                    await self.db.execute(
                        """
                        UPDATE queue_jobs
                        SET state = 'failed',
                            attempts = ?,
                            error = ?,
                            completed_at = ?,
                            worker_id = NULL,
                            visibility_deadline = NULL
                        WHERE job_id = ?
                        """,
                        (attempts, error, time.time(), job_id),
                    )
                else:
                    await self.db.execute(
                        """
                        UPDATE queue_jobs
                        SET state = 'queued',
                            attempts = ?,
                            error = ?,
                            worker_id = NULL,
                            visibility_deadline = NULL
                        WHERE job_id = ?
                        """,
                        (attempts, error, job_id),
                    )
                await self.db.execute("COMMIT")
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise

    async def wait_for_result(self, job_id: str, timeout: float = 120.0) -> JobResult:
        deadline = time.time() + timeout
        while True:
            async with self.db.execute(
                "SELECT state, result, error, metrics FROM queue_jobs WHERE job_id = ?",
                (job_id,),
            ) as cur:
                row = await cur.fetchone()
            if row is not None:
                state, result_blob, error, metrics_json = row
                if state == "completed":
                    metrics = json.loads(metrics_json) if metrics_json else None
                    return JobResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        result=result_blob,
                        metrics=metrics,
                    )
                if state == "failed":
                    return JobResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        error=error or "job failed",
                    )
            if time.time() >= deadline:
                return JobResult(
                    job_id=job_id,
                    status=JobStatus.TIMED_OUT,
                    error=f"Timed out after {timeout}s waiting for result",
                )
            await asyncio.sleep(self.poll_interval)

    async def get_queue_depth(self, model_filter: Optional[str] = None) -> int:
        clauses = ["state IN ('queued', 'inflight')"]
        params: list = []
        if model_filter is not None:
            clauses.append("(required_model IS NULL OR required_model = ?)")
            params.append(model_filter)
        where = " AND ".join(clauses)
        async with self.db.execute(f"SELECT COUNT(*) FROM queue_jobs WHERE {where}", params) as cur:
            row = await cur.fetchone()
        return int(row[0] if row else 0)


def _row_to_job(row: aiosqlite.Row) -> QueuedJob:
    # Column order matches queue_jobs — robust to the schema growing by
    # indexing by name if the underlying connection has Row factory.
    def col(idx: int, name: str):
        try:
            return row[name]  # Row factory (if the caller sets it)
        except (IndexError, KeyError, TypeError):
            return row[idx]

    return QueuedJob(
        job_id=col(0, "job_id"),
        session_id=col(1, "session_id"),
        operation=col(2, "operation"),
        payload=col(3, "payload"),
        priority=col(4, "priority") or 0,
        preferred_worker=col(5, "preferred_worker"),
        required_model=col(6, "required_model"),
        estimated_duration_ms=col(7, "estimated_duration_ms"),
        user_id=col(8, "user_id"),
        enqueued_at=col(9, "enqueued_at"),
    )
