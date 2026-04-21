# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""In-memory object store — for tests and solo-dev unified mode."""

from __future__ import annotations

import asyncio
import io
import time
from typing import BinaryIO, Optional


class InMemoryObjectStore:
    """Thread-safe in-memory object store.

    Uses a single :class:`asyncio.Lock` to protect the backing dict so that
    concurrent coroutines can safely read/write. Values are immutable
    ``bytes`` so reads do not need to copy.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._metadata: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    async def put(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"put() expects bytes, got {type(data).__name__}")
        async with self._lock:
            self._store[key] = bytes(data)
            self._metadata[key] = {
                "content_type": content_type,
                "size": str(len(data)),
                "ts": str(time.time()),
                **(metadata or {}),
            }

    async def get(self, key: str) -> bytes:
        async with self._lock:
            if key not in self._store:
                raise KeyError(key)
            return self._store[key]

    async def get_stream(self, key: str) -> BinaryIO:
        return io.BytesIO(await self.get(key))

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)
            self._metadata.pop(key, None)

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._store

    async def list_keys(self, prefix: str) -> list[str]:
        async with self._lock:
            return sorted(k for k in self._store if k.startswith(prefix))

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        # In-memory store has no real URL; return a pseudo-URI so callers
        # that just want to embed the reference can still work.
        return f"memory://{key}?expires_in={expires_in}"

    async def copy(self, src_key: str, dst_key: str) -> None:
        async with self._lock:
            if src_key not in self._store:
                raise KeyError(src_key)
            self._store[dst_key] = self._store[src_key]
            self._metadata[dst_key] = dict(self._metadata.get(src_key, {}))
