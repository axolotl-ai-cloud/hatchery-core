# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for object store backends."""

from __future__ import annotations

import pytest

from hatchery.core.backends.object_store.local import LocalObjectStore
from hatchery.core.backends.object_store.memory import InMemoryObjectStore


@pytest.fixture(params=["memory", "local"])
def store_factory(request, tmp_path):
    if request.param == "memory":
        return lambda: InMemoryObjectStore()
    return lambda: LocalObjectStore(root=str(tmp_path / "obj"))


async def test_put_get_roundtrip(store_factory):
    store = store_factory()
    await store.put("foo/bar.bin", b"hello")
    assert await store.get("foo/bar.bin") == b"hello"


async def test_get_missing_raises(store_factory):
    store = store_factory()
    with pytest.raises(KeyError):
        await store.get("nope")


async def test_exists(store_factory):
    store = store_factory()
    assert not await store.exists("a")
    await store.put("a", b"x")
    assert await store.exists("a")


async def test_delete(store_factory):
    store = store_factory()
    await store.put("a", b"x")
    await store.delete("a")
    assert not await store.exists("a")
    # Delete is idempotent.
    await store.delete("a")


async def test_list_keys_prefix(store_factory):
    store = store_factory()
    for key in ["a/1", "a/2", "a/sub/3", "b/4"]:
        await store.put(key, b"x")
    keys = await store.list_keys("a/")
    assert set(keys) == {"a/1", "a/2", "a/sub/3"}
    keys = await store.list_keys("b/")
    assert set(keys) == {"b/4"}


async def test_copy(store_factory):
    store = store_factory()
    await store.put("src", b"payload")
    await store.copy("src", "dst")
    assert await store.get("dst") == b"payload"


async def test_copy_missing_raises(store_factory):
    store = store_factory()
    with pytest.raises(KeyError):
        await store.copy("missing", "dst")


async def test_get_stream(store_factory):
    store = store_factory()
    await store.put("a", b"stream-me")
    stream = await store.get_stream("a")
    assert stream.read() == b"stream-me"


async def test_put_overwrites(store_factory):
    store = store_factory()
    await store.put("a", b"1")
    await store.put("a", b"2")
    assert await store.get("a") == b"2"


async def test_local_store_rejects_traversal(tmp_path):
    store = LocalObjectStore(root=str(tmp_path))
    with pytest.raises(ValueError):
        await store.put("../escape", b"x")
    with pytest.raises(ValueError):
        await store.put("/abs/path", b"x")


async def test_local_store_atomic_write(tmp_path):
    store = LocalObjectStore(root=str(tmp_path))
    await store.put("subdir/file.bin", b"contents")
    # Parent directories are auto-created.
    assert (tmp_path / "subdir" / "file.bin").exists()
