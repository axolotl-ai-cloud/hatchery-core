# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unified solo-dev launcher: gateway + worker in one process.

Run this on a workstation to get a fully functional platform without
needing Redis, Postgres, S3, or any external dependencies.  The gateway
and the worker share an in-memory job queue; the object store is either
``memory`` or ``local`` (filesystem).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Optional

import structlog

from hatchery.core.backends.auth.api_key import APIKeyAuthProvider
from hatchery.core.backends.compute.local import LocalComputeBackend
from hatchery.core.backends.metadata.memory import InMemoryMetadataStore
from hatchery.core.backends.metrics.log import LogMetrics
from hatchery.core.backends.object_store.local import LocalObjectStore
from hatchery.core.backends.object_store.memory import InMemoryObjectStore
from hatchery.core.backends.queue.memory import InMemoryJobQueue
from hatchery.core.config import Config
from hatchery.core.gateway import create_app, set_config

logger = structlog.get_logger("hatchery.core.unified")


def build_unified_config(
    *,
    persistent: bool = True,
    root: str = "/tmp/hatchery_data",
) -> Config:
    """Build a self-contained config suitable for single-process use.

    ``persistent=True`` uses a local-filesystem object store so blob
    state survives across process restarts; metadata always uses the
    in-memory store in core. Extension packages that ship a durable
    metadata backend can build their own equivalent of this function.
    """
    if persistent:
        objects = LocalObjectStore(root=root)
        metadata = InMemoryMetadataStore()
    else:
        objects = InMemoryObjectStore()
        metadata = InMemoryMetadataStore()
    return Config(
        auth=APIKeyAuthProvider(),
        metadata=metadata,
        objects=objects,
        queue=InMemoryJobQueue(),
        compute=LocalComputeBackend(),
        metrics=LogMetrics(),
    )


@asynccontextmanager
async def unified_runtime(
    *,
    base_model: str,
    device: str = "cuda:0",
    persistent: bool = True,
    root: str = "/tmp/hatchery_data",
    load_model: bool = True,
):
    """Async context manager that wires up backends + a GPU worker."""
    from hatchery.core.worker import GPUWorker

    config = build_unified_config(persistent=persistent, root=root)
    await config.metadata.initialize()
    await config.queue.initialize()
    set_config(config)

    worker = GPUWorker(
        worker_id=f"worker-{uuid.uuid4().hex[:8]}",
        base_model_name=base_model,
        config=config,
        device=device,
        load_model=load_model,
    )
    await worker.register()

    stop_event = asyncio.Event()

    async def _worker_loop() -> None:
        while not stop_event.is_set():
            job = await config.queue.dequeue(
                worker_id=worker.worker_id,
                model_filter=base_model,
                visibility_timeout=300,
            )
            if job is None:
                await asyncio.sleep(0.02)
                continue
            await worker._process_one(job)

    task = asyncio.create_task(_worker_loop())
    try:
        yield config, worker
    finally:
        stop_event.set()
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
        await config.queue.close()
        await config.metadata.close()


def create_unified_app(
    *,
    base_model: Optional[str] = None,
    device: str = "cuda:0",
    persistent: bool = True,
    root: str = "/tmp/hatchery_data",
):
    """Create a FastAPI app that also runs a worker on startup.

    Designed for ``uvicorn hatchery.core.unified:create_unified_app --factory``.
    """
    from fastapi import FastAPI

    from hatchery.core.worker import GPUWorker

    model = base_model or os.environ.get("HATCHERY_BASE_MODEL", "Qwen/Qwen2-0.5B")
    config = build_unified_config(persistent=persistent, root=root)
    set_config(config)

    worker_holder: dict = {}

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        await config.metadata.initialize()
        await config.queue.initialize()
        worker = GPUWorker(
            worker_id=f"worker-{uuid.uuid4().hex[:8]}",
            base_model_name=model,
            config=config,
            device=device,
        )
        await worker.register()
        worker_holder["worker"] = worker
        stop = asyncio.Event()
        worker_holder["stop"] = stop

        async def _loop() -> None:
            while not stop.is_set():
                job = await config.queue.dequeue(
                    worker_id=worker.worker_id,
                    model_filter=model,
                    visibility_timeout=300,
                )
                if job is None:
                    await asyncio.sleep(0.02)
                    continue
                await worker._process_one(job)

        worker_holder["task"] = asyncio.create_task(_loop())
        try:
            yield
        finally:
            stop.set()
            worker_holder["task"].cancel()
            with suppress(asyncio.CancelledError, Exception):
                await worker_holder["task"]
            await config.queue.close()
            await config.metadata.close()

    app = create_app(config=config)
    app.router.lifespan_context = _lifespan
    return app
