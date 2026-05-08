# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Core platform configuration and factory.

``Config`` is the core dataclass — protocols only. It drives the
open-source gateway/worker/trainer with the in-memory and local-
filesystem backends that ship in this package.

Extension packages can subclass ``Config`` to add their own fields
and provide an alternative factory that wires production backends
(object stores, metadata stores, queues, metrics, billing, …) based
on environment variables.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Optional

from hatchery.core.protocols import (
    AuthProvider,
    ComputeBackend,
    JobQueue,
    LoraStatePersister,
    MetadataStore,
    MetricsCollector,
    ObjectStore,
    OptimizerStatePersister,
)


def _default_lora_state_persister() -> LoraStatePersister:
    """Default LoRA state persister: bf16 snapshot on every save.

    Lazy-imported so constructing a bare ``Config()`` in a non-torch
    environment doesn't trip on the import.
    """
    from hatchery.core.lora_state import Bf16SnapshotPersister

    return Bf16SnapshotPersister()


def _default_optimizer_state_persister() -> OptimizerStatePersister:
    """Default optimizer state persister: full dict on every save."""
    from hatchery.core.optimizer_state import FullOptimizerPersister

    return FullOptimizerPersister()


@dataclass
class Config:
    """Injected at startup. Every backend is a protocol implementation.

    This is the core config: only core protocols. Extension packages
    can subclass to add fields for additional features (e.g., balance,
    billing, distributed session registry).
    """

    auth: AuthProvider
    metadata: MetadataStore
    objects: ObjectStore
    queue: JobQueue
    compute: ComputeBackend
    metrics: MetricsCollector

    # Sampling backend for GRPO rollouts / customer inference.
    # ``None`` means use LocalPEFTSamplingBackend (worker-local).
    sampling_backend: Optional[object] = None

    # DFlash speculative decoding config. ``None`` disables DFlash entirely.
    # Set to a ``DFlashConfig`` instance to enable speculative decoding for
    # eligible ``sample`` requests handled by ``GPUWorker._handle_sample``.
    # Import lazily to avoid pulling in dflash at core import time.
    dflash: Optional[object] = None

    # LoRA state persister. Default is the bf16-snapshot-on-every-save
    # implementation in ``hatchery.core.lora_state``. Alternative persisters
    # that layer compression on top of the same on-disk layout can be
    # substituted here.
    lora_state: LoraStatePersister = field(default_factory=_default_lora_state_persister)

    # Optimizer state persister. Default is the full-dict-on-every-save
    # implementation in ``hatchery.core.optimizer_state``. Extension packages
    # may substitute a blockwise-int8-delta persister via OPTIMIZER_STATE_BACKEND.
    optimizer_state: OptimizerStatePersister = field(
        default_factory=_default_optimizer_state_persister
    )

    # Scheduling config
    idle_suspend_seconds: int = 300
    idle_terminate_seconds: int = 2_592_000
    max_job_timeout_seconds: int = 600

    # Object store key prefixes
    sessions_prefix: str = "sessions"
    jobs_prefix: str = "jobs"

    # Payload inlining threshold (bytes)
    inline_payload_threshold: int = 256 * 1024

    def build_session_store(self, *, local, worker=None):
        """Construct the session-state store for a worker.

        Core returns a local-only store. Extension packages can
        override this to return a mirrored store that asynchronously
        syncs to ``self.objects``.

        Parameters
        ----------
        local:
            The worker's local object store (disk-backed).
        worker:
            The worker being built — passed through so extensions can
            introspect ``worker.base_model_name`` / register a peer
            watcher without needing a separate wiring channel.
        """
        from hatchery.core.session_store import LocalSessionStateStore

        return LocalSessionStateStore(local=local, remote=self.objects)

    def apply_runtime_model_optimizations(
        self,
        model: Any,
        *,
        base_model_name: str,
        lora_config: Any = None,
    ) -> dict[str, Any]:
        """Best-effort hook for extension packages to patch model runtime behavior.

        Core leaves the model unchanged. Hosted installs can override this to
        apply optional, lazy-loaded kernel paths such as ScatterMoE-LoRA.
        The return value is surfaced in worker registration metadata so
        operators can see which optional paths were selected.
        """
        return {}


def build_core_config() -> Config:
    """Build a ``Config`` using only core (in-memory / local) backends.

    Suitable for development, tests, and self-managed deployments
    that don't need production-grade external services.
    """
    # ── Object Store ──
    obj_backend = os.environ.get("HATCHERY_OBJECT_STORE", "local")
    if obj_backend == "memory":
        from hatchery.core.backends.object_store.memory import InMemoryObjectStore

        objects: ObjectStore = InMemoryObjectStore()
    else:
        from hatchery.core.backends.object_store.local import LocalObjectStore

        objects = LocalObjectStore(
            root=os.environ.get("HATCHERY_LOCAL_STORE_PATH", "/tmp/hatchery_data"),
        )

    # ── Metadata Store ──
    meta_backend = os.environ.get("HATCHERY_METADATA_STORE", "memory")
    if meta_backend == "sqlite":
        from hatchery.core.backends.metadata.sqlite import SQLiteMetadataStore

        metadata: MetadataStore = SQLiteMetadataStore(
            path=os.environ.get("HATCHERY_SQLITE_PATH", "/tmp/hatchery_data/metadata.db"),
        )
    else:
        from hatchery.core.backends.metadata.memory import InMemoryMetadataStore

        metadata = InMemoryMetadataStore()

    # ── Job Queue ──
    queue_backend = os.environ.get("HATCHERY_JOB_QUEUE", "memory")
    if queue_backend == "sqlite":
        from hatchery.core.backends.queue.sqlite import SQLiteJobQueue

        queue: JobQueue = SQLiteJobQueue(
            path=os.environ.get("HATCHERY_SQLITE_QUEUE_PATH", "/tmp/hatchery_data/queue.db"),
        )
    else:
        from hatchery.core.backends.queue.memory import InMemoryJobQueue

        queue = InMemoryJobQueue()

    # ── Auth ──
    from hatchery.core.backends.auth.api_key import APIKeyAuthProvider

    auth: AuthProvider = APIKeyAuthProvider()
    admin_key = os.environ.get("HATCHERY_ADMIN_API_KEY")
    if not admin_key:
        # Nothing supplied — mint a key so the gateway isn't unusable on
        # first launch. Prefixed so it's recognizable in logs/config
        # files, and printed loud enough that a user deploying to a
        # public host (Hugging Face Space, Railway, Fly) can't miss it.
        admin_key = f"apta_{secrets.token_urlsafe(32)}"
        _banner = (
            "\n"
            + "=" * 72
            + "\n"
            + "  hatchery: no HATCHERY_ADMIN_API_KEY set — generated a one-time admin key\n"
            + f"    {admin_key}\n"
            + "  Save this now. It is the only credential that can reach the gateway.\n"
            + "  Set HATCHERY_ADMIN_API_KEY in the environment to reuse a stable key across\n"
            + "  restarts; otherwise a fresh key is minted each launch.\n"
            + "=" * 72
            + "\n"
        )
        # Print to stderr so it shows up even when logging is silenced,
        # and also emit via the standard logger for structured-log setups.
        print(_banner, flush=True)
        logging.getLogger("hatchery.core.config").warning(
            "generated_admin_api_key", extra={"admin_api_key": admin_key}
        )
    auth.add_key(  # type: ignore[attr-defined]
        token=admin_key,
        user_id=os.environ.get("HATCHERY_ADMIN_USER_ID", "admin"),
        max_concurrent_sessions=100,
        max_rank=256,
    )

    # ── Compute ──
    from hatchery.core.backends.compute.local import LocalComputeBackend

    compute: ComputeBackend = LocalComputeBackend()

    # ── Metrics ──
    from hatchery.core.backends.metrics.log import LogMetrics

    metrics: MetricsCollector = LogMetrics()

    # Sampling backend is None in core config. Extension packages may wire
    # a VLLMSamplingBackend or similar via their config factory.
    sampling_backend = None

    return Config(
        auth=auth,
        metadata=metadata,
        objects=objects,
        queue=queue,
        compute=compute,
        metrics=metrics,
        sampling_backend=sampling_backend,
    )
