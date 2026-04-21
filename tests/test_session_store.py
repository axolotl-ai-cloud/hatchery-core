# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for :class:`hatchery.core.session_store.LocalSessionStateStore`.

Core ships a local-only session store: writes go to local disk, reads
fall back to the remote object store when a local miss occurs, and
the mirror surface (``mark_dirty`` / ``flush`` / ``drain``) is a
no-op. Hosted's ``MirroredSessionStateStore`` exercises the mirror
behavior in the hatchery-hosted test suite.
"""

from __future__ import annotations

import os

import pytest

from hatchery.core.backends.object_store.local import LocalObjectStore
from hatchery.core.backends.object_store.memory import InMemoryObjectStore
from hatchery.core.session_store import (
    LocalSessionStateStore,
    default_local_root,
    ensure_local_root,
)


@pytest.fixture
def local(tmp_path):
    return LocalObjectStore(root=str(tmp_path / "local"))


@pytest.fixture
def remote():
    return InMemoryObjectStore()


@pytest.fixture
def store(local, remote):
    return LocalSessionStateStore(local=local, remote=remote)


def _session_prefix(sid: str) -> str:
    return f"sessions/{sid}/live_state"


# ── save_local / load_local ─────────────────────────────────────────


async def test_save_local_writes_all_blobs(store, local):
    sid = "sess-A"
    prefix = _session_prefix(sid)
    await store.save_local(
        prefix,
        {
            "lora_weights.pt": b"weights",
            "grad_accum.pt": b"grads",
            "session_meta.json": b"{}",
        },
    )

    assert await local.get(f"{prefix}/lora_weights.pt") == b"weights"
    assert await local.get(f"{prefix}/grad_accum.pt") == b"grads"
    assert await local.get(f"{prefix}/session_meta.json") == b"{}"


async def test_load_local_missing_returns_none(store):
    assert await store.load_local("sessions/nope/live_state/lora_weights.pt") is None


async def test_load_local_returns_bytes_when_present(store, local):
    await local.put("k", b"payload")
    assert await store.load_local("k") == b"payload"


async def test_load_remote_raises_keyerror_for_missing(store):
    with pytest.raises(KeyError):
        await store.load_remote("does-not-exist")


async def test_load_remote_reads_from_remote_store(store, remote):
    await remote.put("some/key", b"payload")
    assert await store.load_remote("some/key") == b"payload"


async def test_load_remote_without_remote_raises(local):
    """A store with no ``remote`` still exposes load_remote but raises."""
    store = LocalSessionStateStore(local=local, remote=None)
    with pytest.raises(KeyError):
        await store.load_remote("any/key")


# ── no-op mirror surface ───────────────────────────────────────────


async def test_mark_dirty_is_a_noop(store):
    # Simply must not raise and must not schedule any background task.
    store.mark_dirty("sess-any")
    # No ``_sync_tasks`` attribute at all on the core variant.
    assert not hasattr(store, "_sync_tasks")


async def test_flush_is_a_noop(store):
    # Returns quickly without error on any input.
    await store.flush("sess-any", timeout=0.1)


async def test_drain_is_a_noop(store):
    await store.drain(timeout=0.1)


async def test_has_pending_always_false(store):
    assert store.has_pending("sess-any") is False


# ── clear_local ────────────────────────────────────────────────────


async def test_clear_local_removes_all_session_keys(store, local):
    sid = "sess-clear"
    prefix = _session_prefix(sid)
    await store.save_local(
        prefix,
        {"lora_weights.pt": b"W", "grad_accum.pt": b"G", "session_meta.json": b"{}"},
    )
    await store.clear_local(sid)
    assert await local.list_keys(prefix) == []


async def test_clear_local_missing_session_is_safe(store):
    # Should not raise.
    await store.clear_local("never-existed")


# ── default_local_root / ensure_local_root ─────────────────────────


def test_default_local_root_default_shape():
    prev = os.environ.pop("HATCHERY_WORKER_LOCAL_DIR", None)
    try:
        assert default_local_root("w-42") == "/tmp/hatchery-worker-w-42"
    finally:
        if prev is not None:
            os.environ["HATCHERY_WORKER_LOCAL_DIR"] = prev


def test_default_local_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HATCHERY_WORKER_LOCAL_DIR", str(tmp_path / "override"))
    assert default_local_root("ignored") == str(tmp_path / "override")


def test_ensure_local_root_creates_dir(tmp_path):
    target = tmp_path / "nested" / "state"
    assert not target.exists()
    returned = ensure_local_root(str(target))
    assert returned == str(target)
    assert target.is_dir()


def test_ensure_local_root_is_idempotent(tmp_path):
    target = tmp_path / "already" / "there"
    ensure_local_root(str(target))
    # Second call on an existing dir must not raise.
    ensure_local_root(str(target))
    assert target.is_dir()
