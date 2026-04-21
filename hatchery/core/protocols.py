# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Protocol definitions and data classes used across the platform.

Every pluggable component in the platform is defined as a Python `Protocol`
so that backends can be swapped at startup via config. No concrete backend
is imported at the interface level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, BinaryIO, Optional, Protocol, Union

# ─── Enums ────────────────────────────────────────────────────────────────


class SessionStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


# ─── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class SessionRecord:
    session_id: str
    user_id: str
    base_model: str
    # ``lora_rank`` is None for full-parameter sessions (Fireworks-style
    # signal: rank omitted ⇒ FFT). Existing LoRA sessions stay int.
    lora_rank: Optional[int]
    lora_alpha: int
    target_modules: list[str]
    total_steps: int
    accum_steps: int
    created_at: float
    last_accessed: float
    status: SessionStatus
    state_prefix: str
    last_worker_id: Optional[str] = None
    avg_step_duration_ms: Optional[float] = None
    total_tokens_processed: int = 0


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    user_id: str
    operation: str
    status: JobStatus
    created_at: float
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    payload_key: Optional[str] = None
    payload_inline: Optional[bytes] = None
    result_key: Optional[str] = None
    result_inline: Optional[bytes] = None
    error_message: Optional[str] = None
    gpu_time_ms: Optional[float] = None
    tokens_processed: Optional[int] = None


@dataclass
class QueuedJob:
    job_id: str
    session_id: str
    operation: str
    payload: bytes
    priority: int = 0
    preferred_worker: Optional[str] = None
    required_model: Optional[str] = None
    estimated_duration_ms: Optional[float] = None
    user_id: Optional[str] = None
    enqueued_at: Optional[float] = None
    required_cp_degree: int = 1  # context parallelism needed (1 = single GPU)
    # Note: per-job scoped credentials live inside the serialized
    # ``payload`` under the ``_scoped_token`` key — see the gateway's
    # ``_enqueue_job`` and the worker's ``_execute_job`` for how the
    # token is minted and verified. Keeping it in the payload means
    # no queue-schema change is required to enable job-scoping.


@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    result: Optional[bytes] = None
    error: Optional[str] = None
    metrics: Optional[dict] = None


@dataclass
class CheckpointRecord:
    """Metadata for a saved checkpoint."""

    checkpoint_id: str
    session_id: str
    user_id: str
    name: str
    checkpoint_type: str = "training"  # "training" | "sampler"
    created_at: float = 0.0
    expires_at: Optional[float] = None  # None = never expires
    size_bytes: int = 0
    public: bool = False
    object_key: str = ""  # key prefix in object store


@dataclass
class WorkerInfo:
    worker_id: str
    provider: str
    gpu_type: str
    gpu_count: int
    loaded_models: list[str]
    status: str
    max_concurrent_loras: int
    vram_free_mb: int
    region: Optional[str] = None
    spot: bool = False
    last_heartbeat: Optional[float] = None
    cp_degree: int = 1  # context parallelism degree this worker supports


@dataclass
class AuthenticatedUser:
    user_id: str
    email: Optional[str] = None
    org_id: Optional[str] = None
    roles: list[str] = field(default_factory=list)
    tier: str = "free"
    max_concurrent_sessions: int = 5
    max_rank: int = 64
    allowed_models: Optional[list[str]] = None


# ─── Protocols ────────────────────────────────────────────────────────────


class ObjectStore(Protocol):
    """Abstract blob storage for session state.

    Key schema:
      sessions/{session_id}/live_state/lora_weights.pt
      sessions/{session_id}/live_state/optimizer_state.pt
      sessions/{session_id}/live_state/grad_accum.pt
      sessions/{session_id}/live_state/session_meta.json
      sessions/{session_id}/checkpoints/{name}/lora_weights.pt
    """

    async def put(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def get_stream(self, key: str) -> BinaryIO: ...

    async def delete(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool: ...

    async def list_keys(self, prefix: str) -> list[str]: ...

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str: ...

    async def copy(self, src_key: str, dst_key: str) -> None: ...


class MetadataStore(Protocol):
    """Structured metadata for sessions and jobs."""

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...

    # Sessions
    async def create_session(self, record: SessionRecord) -> None: ...
    async def get_session(self, session_id: str) -> Optional[SessionRecord]: ...
    async def update_session(self, session_id: str, **kwargs) -> None: ...
    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionRecord]: ...
    async def list_sessions_by_worker(self, worker_id: str) -> list[SessionRecord]: ...

    # Jobs
    async def create_job(self, record: JobRecord) -> None: ...
    async def get_job(self, job_id: str) -> Optional[JobRecord]: ...
    async def update_job(self, job_id: str, **kwargs) -> None: ...
    async def get_pending_jobs(self, session_id: str) -> list[JobRecord]: ...

    # Checkpoints
    async def create_checkpoint(self, record: CheckpointRecord) -> None: ...
    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Optional[CheckpointRecord]: ...
    async def list_checkpoints(
        self, session_id: str, checkpoint_type: Optional[str] = None
    ) -> list[CheckpointRecord]: ...
    async def update_checkpoint(self, session_id: str, checkpoint_id: str, **kwargs) -> None: ...
    async def delete_checkpoint(self, session_id: str, checkpoint_id: str) -> None: ...
    async def get_expired_checkpoints(
        self, now: Optional[float] = None
    ) -> list[CheckpointRecord]: ...

    # Metrics queries
    async def get_session_step_history(self, session_id: str, last_n: int = 50) -> list[dict]: ...
    async def get_active_session_count_by_worker(self) -> dict[str, int]: ...


class JobQueue(Protocol):
    """Work queue between gateway and GPU workers."""

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...

    async def enqueue(self, job: QueuedJob) -> None: ...
    async def dequeue(
        self,
        worker_id: str,
        model_filter: Optional[Union[str, list[str]]] = None,
        visibility_timeout: int = 300,
    ) -> Optional[QueuedJob]: ...
    async def ack(self, job_id: str, result: JobResult) -> None: ...
    async def nack(self, job_id: str, error: str) -> None: ...
    async def wait_for_result(self, job_id: str, timeout: float = 120.0) -> JobResult: ...
    async def get_queue_depth(
        self, model_filter: Optional[Union[str, list[str]]] = None
    ) -> int: ...

    # ── Accumulation hard-pin ───────────────────────────────────────
    # While a pin is active, ``dequeue`` must only return jobs for the
    # pinned session to the pinned worker — regardless of the sticky
    # affinity window. Intended for sessions whose grad_accum only
    # exists on the owner worker's local disk (between forward_backward
    # and the next optim_step). Without this guard, a visibility-timeout
    # retry could silently hand the session to a peer that reads stale
    # pre-accumulation state from the object store.
    #
    # Semantics:
    #   - ``set_accum_pin`` is idempotent; a fresh call refreshes the TTL.
    #   - ``clear_accum_pin`` on an absent pin is a no-op.
    #   - Pins auto-expire after ``ttl_s`` so a dead owner doesn't
    #     permanently block the session.
    async def set_accum_pin(
        self, session_id: str, worker_id: str, *, ttl_s: float = 600.0
    ) -> None: ...
    async def clear_accum_pin(self, session_id: str) -> None: ...


class AuthProvider(Protocol):
    """Authentication and authorization."""

    async def authenticate(self, token: str) -> Optional[AuthenticatedUser]: ...
    async def authorize(self, user: AuthenticatedUser, action: str, resource: str) -> bool: ...


class ComputeBackend(Protocol):
    """Worker directory for core routing and health checks."""

    async def list_workers(self) -> list[WorkerInfo]: ...
    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]: ...
    async def health_check(self, worker_id: str) -> bool: ...


class ManagedComputeBackend(ComputeBackend, Protocol):
    """Extends ComputeBackend with cloud lifecycle operations.

    Used by orchestration extensions to provision, terminate, and drain
    workers. Core code never calls these methods — they exist for
    autoscalers and admin tooling.
    """

    async def provision_worker(
        self,
        model: str,
        gpu_type: str = "A100-80GB",
        spot: bool = True,
        region: Optional[str] = None,
    ) -> WorkerInfo: ...
    async def terminate_worker(self, worker_id: str) -> None: ...
    async def drain_worker(self, worker_id: str) -> None: ...


class MetricsCollector(Protocol):
    """Structured metrics for monitoring and scheduling."""

    def record_job_duration(
        self,
        session_id: str,
        user_id: str,
        operation: str,
        duration_ms: float,
        tokens: int,
        worker_id: str,
        gpu_type: str,
        cost_dimensions: Optional[dict[str, Any]] = None,
    ) -> None: ...

    # ``cost_dimensions`` carries the internal cost-analysis fields:
    #   model_name: str           — HF hub model name
    #   model_params_b: float     — approx parameter count in billions
    #   max_seq_len: int          — padded batch sequence length
    #   batch_size: int           — number of items in the batch
    #   lora_rank: int            — adapter rank for this session
    #   loss_fn: str              — which loss was used
    #   fused_path: bool          — did we take the fused CE shortcut?
    #   dp_degree: int            — data parallel degree
    #   tp_degree: int            — tensor parallel degree
    #   cp_degree: int            — context parallel degree
    #   is_context_parallel: bool — seq_len triggered CP routing?
    # All keys are optional — callers include what they know.

    def record_queue_depth(self, model: str, depth: int) -> None: ...

    def record_worker_utilization(
        self, worker_id: str, gpu_util_pct: float, vram_used_mb: int
    ) -> None: ...

    def record_lora_swap_time(
        self,
        session_id: str,
        swap_direction: str,
        duration_ms: float,
        state_size_bytes: int,
    ) -> None: ...

    def record_object_store_io(
        self,
        operation: str,
        key: str,
        size_bytes: int,
        duration_ms: float,
    ) -> None: ...

    def record_session_event(self, session_id: str, event: str) -> None: ...

    def increment_counter(self, name: str, tags: dict[str, str]) -> None: ...

    def set_gauge(self, name: str, value: float, tags: dict[str, str]) -> None: ...


class SessionRegistry(Protocol):
    """Maps session IDs to the worker that currently holds them.

    Used by the gateway to set ``preferred_worker`` on enqueued jobs.
    Workers write their active sessions after each job; gateways read
    before enqueue. The mapping has a TTL so dead workers' entries
    expire without explicit cleanup.

    The in-memory implementation in core is a simple dict (sufficient
    for single-gateway deployments). Extension packages can provide
    distributed implementations for cross-replica consistency.
    """

    async def set(self, session_id: str, worker_id: str) -> None:
        """Record that ``worker_id`` currently holds ``session_id``."""
        ...

    async def get(self, session_id: str) -> Optional[str]:
        """Return the worker_id holding ``session_id``, or None."""
        ...

    async def remove(self, session_id: str) -> None:
        """Remove a session mapping (e.g., on session termination)."""
        ...

    async def get_sessions_for_worker(self, worker_id: str) -> list[str]:
        """Return all session IDs currently mapped to ``worker_id``."""
        ...


class LoraStatePersister(Protocol):
    """Persists a LoRA adapter's state dict to an ObjectStore.

    The default implementation writes a full bf16 snapshot on every
    save (see :class:`~hatchery.core.lora_state.Bf16SnapshotPersister`).
    Alternative implementations may layer compression schemes (e.g.,
    quantized deltas against a periodic snapshot) on top of the same
    on-disk layout — ``lora_weights.pt`` is always the snapshot file,
    and any extra per-save artifacts live next to it.

    The ``snapshot_cache`` / ``snapshot_version`` / ``delta_count``
    arguments threaded through ``save`` and ``load`` are the worker's
    in-RAM bookkeeping that a delta-style implementation may rely on.
    Snapshot-only implementations can ignore them (returning trivial
    defaults); the API is kept uniform so the worker doesn't branch
    on which persister is in use.
    """

    async def save(
        self,
        objects: ObjectStore,
        prefix: str,
        current_state: dict,
        *,
        snapshot_cache: Optional[dict],
        snapshot_version: int,
        delta_count: int,
        cfg: Any,
    ) -> tuple[Any, dict]:
        """Persist ``current_state`` under ``prefix``.

        Returns ``(SaveResult, new_snapshot_cache)``. The worker should
        replace its ``snapshot_cache`` with the returned value and
        update ``snapshot_version`` / ``delta_count`` from the
        ``SaveResult``.
        """
        ...

    async def load(
        self,
        objects: ObjectStore,
        prefix: str,
    ) -> tuple[dict, dict, int, int]:
        """Reconstruct the live LoRA state from ``prefix``.

        Returns ``(fp32_state, snapshot_cache_bf16, snapshot_version,
        delta_count)``.
        """
        ...

    async def materialize(
        self,
        objects: ObjectStore,
        src_prefix: str,
        dst_prefix: str,
    ) -> int:
        """Write a flattened, self-contained ``lora_weights.pt`` at
        ``dst_prefix``, resolving any pending per-save artifacts
        against the snapshot at ``src_prefix``.

        For a snapshot-only persister this is a direct object-store
        copy; implementations that keep auxiliary per-save state
        reconstruct the live state and write a fresh full snapshot.
        """
        ...


class OptimizerStatePersister(Protocol):
    """Persists an optimizer ``state_dict`` to an ObjectStore.

    The default implementation (:class:`hatchery.core.optimizer_state.FullOptimizerPersister`)
    writes the full dict on every save. Extension packages can provide
    compression-style implementations that layer deltas against periodic
    snapshots on top of the same ``optimizer_state.pt`` filename so both
    paths share the same on-disk layout and the snapshot file remains
    readable by
    the baseline persister.

    The ``snapshot_cache`` / ``snapshot_version`` / ``delta_count``
    arguments mirror the ``LoraStatePersister`` contract. ``None`` is
    accepted for ``current_state`` in ``save`` (session has no optim
    state yet — e.g., before the first ``optim_step``) and ``load``
    may return ``None`` when no snapshot exists.
    """

    async def save(
        self,
        objects: ObjectStore,
        prefix: str,
        current_state: Optional[dict],
        *,
        snapshot_cache: Optional[dict],
        snapshot_version: int,
        delta_count: int,
        cfg: Any,
    ) -> tuple[Any, Optional[dict]]:
        """Persist ``current_state`` under ``prefix``.

        Returns ``(OptimizerSaveResult, new_snapshot_cache)``.
        """
        ...

    async def load(
        self,
        objects: ObjectStore,
        prefix: str,
    ) -> tuple[Optional[dict], Optional[dict], int, int]:
        """Reconstruct the live optimizer state from ``prefix``.

        Returns ``(state, snapshot_cache, snapshot_version, delta_count)``.
        ``state`` is ``None`` when no snapshot is present yet.
        """
        ...
