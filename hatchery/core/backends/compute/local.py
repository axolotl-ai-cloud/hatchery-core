# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Local compute backend — tracks workers as in-process registrations.

Cloud compute backends provision GPUs via provider APIs. The local
backend is used when workers are managed externally (i.e., you start
them with ``python -m hatchery.core.worker``) and just registers themselves
with the backend for observability.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from hatchery.core.protocols import WorkerInfo


class LocalComputeBackend:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = asyncio.Lock()

    async def register_worker(self, info: WorkerInfo) -> None:
        """Worker calls this on startup. Not part of the protocol."""
        async with self._lock:
            info.last_heartbeat = time.time()
            self._workers[info.worker_id] = info

    async def heartbeat(
        self,
        worker_id: str,
        *,
        status: Optional[str] = None,
        vram_free_mb: Optional[int] = None,
    ) -> None:
        async with self._lock:
            info = self._workers.get(worker_id)
            if info is None:
                return
            info.last_heartbeat = time.time()
            if status is not None:
                info.status = status
            if vram_free_mb is not None:
                info.vram_free_mb = vram_free_mb

    async def unregister_worker(self, worker_id: str) -> None:
        async with self._lock:
            self._workers.pop(worker_id, None)

    async def list_workers(self) -> list[WorkerInfo]:
        async with self._lock:
            return list(self._workers.values())

    async def provision_worker(
        self,
        model: str,
        gpu_type: str = "A100-80GB",
        spot: bool = True,
        region: Optional[str] = None,
    ) -> WorkerInfo:
        # Local backend can't provision. It simulates by returning a
        # placeholder that must be started externally.
        raise NotImplementedError(
            "LocalComputeBackend does not provision workers. "
            "Start workers externally via python -m hatchery.core.worker."
        )

    async def terminate_worker(self, worker_id: str) -> None:
        async with self._lock:
            w = self._workers.get(worker_id)
            if w:
                w.status = "offline"

    async def drain_worker(self, worker_id: str) -> None:
        async with self._lock:
            w = self._workers.get(worker_id)
            if w:
                w.status = "draining"

    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        async with self._lock:
            return self._workers.get(worker_id)

    async def health_check(self, worker_id: str) -> bool:
        async with self._lock:
            w = self._workers.get(worker_id)
            if w is None:
                return False
            # Consider alive if heartbeat within 30s.
            return (
                w.last_heartbeat is not None
                and time.time() - w.last_heartbeat < 30
                and w.status != "offline"
            )
