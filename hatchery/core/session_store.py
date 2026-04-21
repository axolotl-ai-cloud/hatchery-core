# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Local-only session state store.

Core ships a minimal local-disk store. Extension packages can replace
it with a mirrored variant that adds background sync to a shared
remote store, ``mark_dirty`` / ``flush`` semantics, and single-peer
optimizations.

Why this exists
---------------
On a single-node / dev deploy, session state lives on local disk and
the worker never hands off to a peer. This store writes to local disk
and nothing else. ``load_remote`` is wired to ``config.objects`` (which
may itself be a local directory) so the worker's fallback-read path
still works when the cache is cold.

Extension packages can override ``Config.build_session_store()`` to
return a mirrored store that syncs to a remote object store.

Invariant preserved
-------------------
The worker cache (``SmartLoRACache``) holds the *authoritative*
in-memory state. Local disk is the durable-on-same-pod tier. The
remote store (when present, extension-provided) is the
durable-across-pod tier. A new worker picking up a session without a
local copy reads from the remote store — which may lag the owner's
local copy by up
to one sync interval. That's fine as long as sticky affinity keeps
the session on its current worker (enforced by the queue). When
sticky expires, the extension store hard-flushes before releasing the job so the
next claimer sees up-to-date remote state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger("hatchery.core.session_store")


class LocalSessionStateStore:
    """Local-disk-only session store.

    Exposes the full session-store surface but treats the mirror
    methods (``mark_dirty`` / ``flush`` / ``drain``) as no-ops. The
    worker calls these unconditionally on the handoff boundaries
    (optim_step ack, save_weights, eviction, shutdown drain); extension packages
    swaps in a real mirrored store to make those calls meaningful.

    Parameters
    ----------
    local:
        Fast, worker-local object store (typically
        :class:`~hatchery.core.backends.object_store.local.LocalObjectStore`
        rooted at a tmp dir). All writes and reads target this.
    remote:
        Optional object store for cross-worker / cross-restart reads
        via :meth:`load_remote`. When ``None``, ``load_remote``
        always raises :class:`KeyError`.
    """

    def __init__(
        self,
        local: Any,
        *,
        remote: Optional[Any] = None,
    ) -> None:
        self.local = local
        self.remote = remote

    # ── Local-disk primitives ────────────────────────────────────────

    async def save_local(self, prefix: str, blobs: dict[str, bytes]) -> None:
        """Atomically write a set of blobs to local disk under ``prefix``.

        The underlying ``LocalObjectStore.put`` already does tmp+rename,
        so each blob appears atomically. We don't guarantee atomicity
        across the *set* of blobs — a crash between puts leaves a
        partially-consistent fileset. Callers that care (``load_local``)
        read ``session_meta.json`` last, so a missing/old meta simply
        causes a remote-fallback load instead of silent corruption.
        """
        await asyncio.gather(*(self.local.put(f"{prefix}/{k}", v) for k, v in blobs.items()))

    async def load_local(self, key: str) -> Optional[bytes]:
        try:
            return await self.local.get(key)
        except KeyError:
            return None

    async def load_remote(self, key: str) -> bytes:
        """Read from the remote store (remote object store).

        Raises :class:`KeyError` when no remote is configured — pure
        local deploys have nothing to fall back to.
        """
        if self.remote is None:
            raise KeyError(f"no remote configured for {key!r}")
        return await self.remote.get(key)

    async def clear_local(self, session_id: str) -> None:
        """Remove the local copy (does not touch remote).

        Called after eviction + successful flush, or on explicit
        session deletion. Safe to call on a missing session.
        """
        prefix = f"sessions/{session_id}/live_state/"
        keys = await self.local.list_keys(prefix)
        await asyncio.gather(
            *(self.local.delete(k) for k in keys),
            return_exceptions=True,
        )

    # ── No-op mirror surface ─────────────────────────────────────────
    # Extension packages may override these with mirrored stores. The core
    # variant provides no-ops so the worker's calls (``mark_dirty``
    # after each write, ``flush`` on handoff, etc.) are safe on pure
    # local-only deploys.

    def mark_dirty(self, session_id: str) -> None:
        return None

    async def flush(self, session_id: str, *, timeout: Optional[float] = None) -> None:
        return None

    async def drain(self, *, timeout: Optional[float] = None) -> None:
        return None

    def has_pending(self, session_id: str) -> bool:
        return False


# Backwards-compatible alias so existing imports keep working while
# the mirrored variant is wired in. Deprecated — prefer
# ``LocalSessionStateStore`` in core, or the mirrored variant in
# extension packages.
SessionStateStore = LocalSessionStateStore


def default_local_root(worker_id: str) -> str:
    """Return the default local-disk path for a given worker.

    Overridable via ``HATCHERY_WORKER_LOCAL_DIR`` env var.
    """
    import os

    override = os.environ.get("HATCHERY_WORKER_LOCAL_DIR")
    if override:
        return override
    # /tmp is tmpfs on most Linux distros — fast but size-limited.
    # For bigger worker state (>1GB) override to a disk path.
    return f"/tmp/hatchery-worker-{worker_id}"


def ensure_local_root(path: str) -> str:
    """Ensure the local root exists; return it."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
