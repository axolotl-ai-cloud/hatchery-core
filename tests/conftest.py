# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Pytest fixtures and early environment setup for hatchery-core tests.

The env vars must be set before any torch import — torchao's module-level
``has_triton()`` probe indexes into the cached device-properties list and
raises IndexError when ``is_available()`` and ``device_count()`` disagree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from hatchery.core.backends.auth.api_key import APIKeyAuthProvider  # noqa: E402
from hatchery.core.backends.compute.local import LocalComputeBackend  # noqa: E402
from hatchery.core.backends.metadata.memory import InMemoryMetadataStore  # noqa: E402
from hatchery.core.backends.metrics.log import LogMetrics  # noqa: E402
from hatchery.core.backends.object_store.local import LocalObjectStore  # noqa: E402
from hatchery.core.backends.object_store.memory import InMemoryObjectStore  # noqa: E402
from hatchery.core.backends.queue.memory import InMemoryJobQueue  # noqa: E402
from hatchery.core.config import Config  # noqa: E402


@pytest_asyncio.fixture
async def memory_store():
    store = InMemoryObjectStore()
    yield store


@pytest_asyncio.fixture
async def local_store(tmp_path):
    store = LocalObjectStore(root=str(tmp_path / "objects"))
    yield store


@pytest_asyncio.fixture
async def memory_metadata():
    store = InMemoryMetadataStore()
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def memory_queue():
    q = InMemoryJobQueue()
    await q.initialize()
    try:
        yield q
    finally:
        await q.close()


@pytest.fixture
def api_key_auth():
    auth = APIKeyAuthProvider()
    auth.add_key(
        "test-token",
        user_id="user-1",
        tier="pro",
        max_concurrent_sessions=10,
        max_rank=128,
    )
    return auth


@pytest.fixture
def metrics():
    return LogMetrics()


@pytest.fixture
def compute():
    return LocalComputeBackend()


@pytest_asyncio.fixture
async def platform_config(
    api_key_auth,
    memory_metadata,
    memory_store,
    memory_queue,
    compute,
    metrics,
):
    return Config(
        auth=api_key_auth,
        metadata=memory_metadata,
        objects=memory_store,
        queue=memory_queue,
        compute=compute,
        metrics=metrics,
    )
