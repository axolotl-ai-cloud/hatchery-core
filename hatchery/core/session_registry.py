# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""In-memory session registry for single-gateway deployments.

Maps session_id → worker_id so the gateway knows which worker to
prefer when enqueuing jobs. Sufficient when there's only one gateway
replica; for multi-replica deployments, use an external shared
registry implementation provided by an extension package.
"""

from __future__ import annotations

import time
from typing import Optional


class InMemorySessionRegistry:
    """Dict-backed session registry with TTL expiry.

    Entries expire after ``ttl_seconds`` if not refreshed. This handles
    the case where a worker dies without explicitly removing its sessions.
    """

    def __init__(self, ttl_seconds: float = 120.0) -> None:
        self._map: dict[str, tuple[str, float]] = {}  # session_id → (worker_id, expires_at)
        self._ttl = ttl_seconds

    async def set(self, session_id: str, worker_id: str) -> None:
        self._map[session_id] = (worker_id, time.time() + self._ttl)

    async def get(self, session_id: str) -> Optional[str]:
        entry = self._map.get(session_id)
        if entry is None:
            return None
        worker_id, expires_at = entry
        if time.time() > expires_at:
            del self._map[session_id]
            return None
        return worker_id

    async def remove(self, session_id: str) -> None:
        self._map.pop(session_id, None)

    async def get_sessions_for_worker(self, worker_id: str) -> list[str]:
        now = time.time()
        result = []
        expired = []
        for sid, (wid, expires_at) in self._map.items():
            if now > expires_at:
                expired.append(sid)
            elif wid == worker_id:
                result.append(sid)
        for sid in expired:
            del self._map[sid]
        return result
