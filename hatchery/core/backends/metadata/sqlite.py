# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SQLite metadata store using aiosqlite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import aiosqlite

from hatchery.core.protocols import (
    CheckpointRecord,
    JobRecord,
    JobStatus,
    SessionRecord,
    SessionStatus,
)

_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id           TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL,
    base_model           TEXT NOT NULL,
    lora_rank            INTEGER,
    lora_alpha           INTEGER NOT NULL,
    target_modules       TEXT NOT NULL,
    total_steps          INTEGER NOT NULL,
    accum_steps          INTEGER NOT NULL,
    created_at           REAL NOT NULL,
    last_accessed        REAL NOT NULL,
    status               TEXT NOT NULL,
    state_prefix         TEXT NOT NULL,
    last_worker_id       TEXT,
    avg_step_duration_ms REAL,
    total_tokens_processed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_status
    ON sessions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_sessions_worker
    ON sessions(last_worker_id);
"""

_JOBS_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id             TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    user_id            TEXT NOT NULL,
    operation          TEXT NOT NULL,
    status             TEXT NOT NULL,
    created_at         REAL NOT NULL,
    assigned_at        REAL,
    completed_at       REAL,
    worker_id          TEXT,
    payload_key        TEXT,
    payload_inline     BLOB,
    result_key         TEXT,
    result_inline      BLOB,
    error_message      TEXT,
    gpu_time_ms        REAL,
    tokens_processed   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_jobs_session_status
    ON jobs(session_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created
    ON jobs(status, created_at);
"""

_CHECKPOINTS_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id    TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    name             TEXT NOT NULL,
    checkpoint_type  TEXT NOT NULL DEFAULT 'training',
    created_at       REAL NOT NULL,
    expires_at       REAL,
    size_bytes       INTEGER NOT NULL DEFAULT 0,
    public           INTEGER NOT NULL DEFAULT 0,
    object_key       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_session
    ON checkpoints(session_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_expires
    ON checkpoints(expires_at);
"""

_CHECKPOINT_COLUMNS = [
    "checkpoint_id",
    "session_id",
    "user_id",
    "name",
    "checkpoint_type",
    "created_at",
    "expires_at",
    "size_bytes",
    "public",
    "object_key",
]

_SESSION_COLUMNS = [
    "session_id",
    "user_id",
    "base_model",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "total_steps",
    "accum_steps",
    "created_at",
    "last_accessed",
    "status",
    "state_prefix",
    "last_worker_id",
    "avg_step_duration_ms",
    "total_tokens_processed",
]

_JOB_COLUMNS = [
    "job_id",
    "session_id",
    "user_id",
    "operation",
    "status",
    "created_at",
    "assigned_at",
    "completed_at",
    "worker_id",
    "payload_key",
    "payload_inline",
    "result_key",
    "result_inline",
    "error_message",
    "gpu_time_ms",
    "tokens_processed",
]


class SQLiteMetadataStore:
    """aiosqlite-backed metadata store with schema auto-init."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SESSIONS_DDL)
        await self._db.executescript(_JOBS_DDL)
        await self._db.executescript(_CHECKPOINTS_DDL)
        await self._migrate_lora_rank_nullable()
        await self._db.commit()

    async def _migrate_lora_rank_nullable(self) -> None:
        # Pre-FFT schemas declared ``lora_rank INTEGER NOT NULL``. SQLite
        # can't ALTER a column's constraint, so we detect the old shape
        # via ``PRAGMA table_info`` and do a table-copy migration.
        # (``notnull`` is a reserved word in a SELECT context, so we use
        # the PRAGMA form and index into the row tuple instead.)
        cur = await self.db.execute("PRAGMA table_info('sessions')")
        rows = await cur.fetchall()
        await cur.close()
        notnull = False
        for r in rows:
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            if r["name"] == "lora_rank" and r["notnull"]:
                notnull = True
                break
        if not notnull:
            return
        await self.db.executescript(
            """
            CREATE TABLE sessions_new (
                session_id           TEXT PRIMARY KEY,
                user_id              TEXT NOT NULL,
                base_model           TEXT NOT NULL,
                lora_rank            INTEGER,
                lora_alpha           INTEGER NOT NULL,
                target_modules       TEXT NOT NULL,
                total_steps          INTEGER NOT NULL,
                accum_steps          INTEGER NOT NULL,
                created_at           REAL NOT NULL,
                last_accessed        REAL NOT NULL,
                status               TEXT NOT NULL,
                state_prefix         TEXT NOT NULL,
                last_worker_id       TEXT,
                avg_step_duration_ms REAL,
                total_tokens_processed INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions_new SELECT * FROM sessions;
            DROP TABLE sessions;
            ALTER TABLE sessions_new RENAME TO sessions;
            CREATE INDEX IF NOT EXISTS idx_sessions_user_status
                ON sessions(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_sessions_worker
                ON sessions(last_worker_id);
            """
        )

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteMetadataStore not initialized")
        return self._db

    # ─── Sessions ───────────────────────────────────────────

    async def create_session(self, record: SessionRecord) -> None:
        await self.db.execute(
            f"INSERT INTO sessions ({','.join(_SESSION_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(_SESSION_COLUMNS))})",
            (
                record.session_id,
                record.user_id,
                record.base_model,
                record.lora_rank,
                record.lora_alpha,
                json.dumps(record.target_modules),
                record.total_steps,
                record.accum_steps,
                record.created_at,
                record.last_accessed,
                record.status.value,
                record.state_prefix,
                record.last_worker_id,
                record.avg_step_duration_ms,
                record.total_tokens_processed,
            ),
        )
        await self.db.commit()

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        async with self.db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_session(row) if row else None

    async def update_session(self, session_id: str, **kwargs) -> None:
        if not kwargs:
            return
        allowed = set(_SESSION_COLUMNS) - {"session_id"}
        sets = []
        values: list = []
        for k, v in kwargs.items():
            if k not in allowed:
                raise AttributeError(f"SessionRecord has no updatable field '{k}'")
            if k == "target_modules":
                v = json.dumps(v)
            if k == "status" and isinstance(v, SessionStatus):
                v = v.value
            sets.append(f"{k} = ?")
            values.append(v)
        values.append(session_id)
        await self.db.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE session_id = ?", values)
        await self.db.commit()

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionRecord]:
        clauses, values = [], []
        if user_id is not None:
            clauses.append("user_id = ?")
            values.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.extend([limit, offset])
        async with self.db.execute(
            f"SELECT * FROM sessions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            values,
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_session(r) for r in rows]

    async def list_sessions_by_worker(self, worker_id: str) -> list[SessionRecord]:
        async with self.db.execute(
            "SELECT * FROM sessions WHERE last_worker_id = ?", (worker_id,)
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_session(r) for r in rows]

    # ─── Jobs ───────────────────────────────────────────────

    async def create_job(self, record: JobRecord) -> None:
        await self.db.execute(
            f"INSERT INTO jobs ({','.join(_JOB_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(_JOB_COLUMNS))})",
            (
                record.job_id,
                record.session_id,
                record.user_id,
                record.operation,
                record.status.value,
                record.created_at,
                record.assigned_at,
                record.completed_at,
                record.worker_id,
                record.payload_key,
                record.payload_inline,
                record.result_key,
                record.result_inline,
                record.error_message,
                record.gpu_time_ms,
                record.tokens_processed,
            ),
        )
        await self.db.commit()

    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        async with self.db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        return self._row_to_job(row) if row else None

    async def update_job(self, job_id: str, **kwargs) -> None:
        if not kwargs:
            return
        allowed = set(_JOB_COLUMNS) - {"job_id"}
        sets = []
        values: list = []
        for k, v in kwargs.items():
            if k not in allowed:
                raise AttributeError(f"JobRecord has no updatable field '{k}'")
            if k == "status" and isinstance(v, JobStatus):
                v = v.value
            sets.append(f"{k} = ?")
            values.append(v)
        values.append(job_id)
        await self.db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", values)
        await self.db.commit()

    async def get_pending_jobs(self, session_id: str) -> list[JobRecord]:
        pending = [
            JobStatus.QUEUED.value,
            JobStatus.ASSIGNED.value,
            JobStatus.RUNNING.value,
        ]
        placeholders = ",".join("?" * len(pending))
        async with self.db.execute(
            f"SELECT * FROM jobs WHERE session_id = ? "
            f"AND status IN ({placeholders}) ORDER BY created_at",
            (session_id, *pending),
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_job(r) for r in rows]

    # ─── Checkpoints ───────────────────────────────────────

    async def create_checkpoint(self, record: CheckpointRecord) -> None:
        await self.db.execute(
            f"INSERT INTO checkpoints ({','.join(_CHECKPOINT_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(_CHECKPOINT_COLUMNS))})",
            (
                record.checkpoint_id,
                record.session_id,
                record.user_id,
                record.name,
                record.checkpoint_type,
                record.created_at,
                record.expires_at,
                record.size_bytes,
                int(record.public),
                record.object_key,
            ),
        )
        await self.db.commit()

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Optional[CheckpointRecord]:
        async with self.db.execute(
            "SELECT * FROM checkpoints WHERE checkpoint_id = ? AND session_id = ?",
            (checkpoint_id, session_id),
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_checkpoint(row) if row else None

    async def list_checkpoints(
        self, session_id: str, checkpoint_type: Optional[str] = None
    ) -> list[CheckpointRecord]:
        if checkpoint_type is not None:
            async with self.db.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? AND checkpoint_type = ? "
                "ORDER BY created_at DESC",
                (session_id, checkpoint_type),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self.db.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_checkpoint(r) for r in rows]

    async def update_checkpoint(self, session_id: str, checkpoint_id: str, **kwargs) -> None:
        if not kwargs:
            return
        allowed = set(_CHECKPOINT_COLUMNS) - {"checkpoint_id", "session_id"}
        sets, values = [], []
        for k, v in kwargs.items():
            if k not in allowed:
                raise AttributeError(f"CheckpointRecord has no updatable field '{k}'")
            if k == "public":
                v = int(v)
            sets.append(f"{k} = ?")
            values.append(v)
        values.extend([checkpoint_id, session_id])
        await self.db.execute(
            f"UPDATE checkpoints SET {', '.join(sets)} WHERE checkpoint_id = ? AND session_id = ?",
            values,
        )
        await self.db.commit()

    async def delete_checkpoint(self, session_id: str, checkpoint_id: str) -> None:
        await self.db.execute(
            "DELETE FROM checkpoints WHERE checkpoint_id = ? AND session_id = ?",
            (checkpoint_id, session_id),
        )
        await self.db.commit()

    async def get_expired_checkpoints(self, now: Optional[float] = None) -> list[CheckpointRecord]:
        import time as _time

        ts = now if now is not None else _time.time()
        async with self.db.execute(
            "SELECT * FROM checkpoints WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (ts,),
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_checkpoint(r) for r in rows]

    # ─── Metrics queries ────────────────────────────────────

    async def get_session_step_history(self, session_id: str, last_n: int = 50) -> list[dict]:
        async with self.db.execute(
            "SELECT job_id, operation, gpu_time_ms, tokens_processed, completed_at "
            "FROM jobs WHERE session_id = ? AND status = ? "
            "AND gpu_time_ms IS NOT NULL "
            "ORDER BY completed_at DESC LIMIT ?",
            (session_id, JobStatus.COMPLETED.value, last_n),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "job_id": r["job_id"],
                "operation": r["operation"],
                "duration_ms": r["gpu_time_ms"],
                "tokens": r["tokens_processed"],
                "completed_at": r["completed_at"],
            }
            for r in rows
        ]

    async def get_active_session_count_by_worker(self) -> dict[str, int]:
        async with self.db.execute(
            "SELECT last_worker_id, COUNT(*) as c FROM sessions "
            "WHERE status = ? AND last_worker_id IS NOT NULL "
            "GROUP BY last_worker_id",
            (SessionStatus.ACTIVE.value,),
        ) as cur:
            rows = await cur.fetchall()
        return {r["last_worker_id"]: r["c"] for r in rows}

    # ─── Helpers ────────────────────────────────────────────

    @staticmethod
    def _row_to_session(row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            base_model=row["base_model"],
            lora_rank=row["lora_rank"],
            lora_alpha=row["lora_alpha"],
            target_modules=json.loads(row["target_modules"]),
            total_steps=row["total_steps"],
            accum_steps=row["accum_steps"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            status=SessionStatus(row["status"]),
            state_prefix=row["state_prefix"],
            last_worker_id=row["last_worker_id"],
            avg_step_duration_ms=row["avg_step_duration_ms"],
            total_tokens_processed=row["total_tokens_processed"] or 0,
        )

    @staticmethod
    def _row_to_checkpoint(row) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=row["checkpoint_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            name=row["name"],
            checkpoint_type=row["checkpoint_type"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            size_bytes=row["size_bytes"],
            public=bool(row["public"]),
            object_key=row["object_key"],
        )

    @staticmethod
    def _row_to_job(row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            operation=row["operation"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            assigned_at=row["assigned_at"],
            completed_at=row["completed_at"],
            worker_id=row["worker_id"],
            payload_key=row["payload_key"],
            payload_inline=row["payload_inline"],
            result_key=row["result_key"],
            result_inline=row["result_inline"],
            error_message=row["error_message"],
            gpu_time_ms=row["gpu_time_ms"],
            tokens_processed=row["tokens_processed"],
        )
