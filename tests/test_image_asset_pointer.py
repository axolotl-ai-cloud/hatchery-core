# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for ``ImageAssetPointerChunk`` parsing + resolution.

Covers:

* Pydantic parsing of the Tinker SDK's unified ``chunks`` discriminated
  union (``encoded_text`` / ``image`` / ``image_asset_pointer``) into
  Hatchery's internal ``chunks`` / ``image_chunks`` buckets.
* ``_fetch_image_asset`` via a monkeypatched ``httpx.AsyncClient``
  (respx is not a declared dep — we stub the client directly).
* ``_resolve_image_asset_pointers`` mutation of a ``ModelInput``.
* End-to-end: posting a ``forward_backward`` with an
  ``image_asset_pointer`` chunk results in the fetched bytes arriving
  in the worker-facing payload as if the client had sent an
  ``EncodedImageChunk``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from typing import Any

import httpx
import msgpack
import pytest
import pytest_asyncio
from httpx import ASGITransport

from hatchery.core.gateway import create_app
from hatchery.core.protocols import JobResult, JobStatus
from hatchery.core.tinker_compat import (
    Datum,
    EncodedImageChunk,
    ImageAssetPointerChunk,
    ModelInput,
    _fetch_image_asset,
    _resolve_image_asset_pointers,
)


def _tiny_png_bytes() -> bytes:
    """Return a 1x1 transparent PNG. Small, deterministic, no Pillow dep."""
    # Minimal valid PNG (1x1, 8-bit RGBA, fully transparent).
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
        "ae426082"
    )


# ─── Pydantic parsing of SDK-unified chunks ───────────────────────────────


def test_parse_sdk_unified_chunks_with_pointer():
    """SDK sends one ``chunks`` list containing text + asset-pointer entries.

    Hatchery's validator must split them into the internal buckets so every
    downstream helper keeps working.
    """
    mi = ModelInput(
        chunks=[
            {"type": "encoded_text", "tokens": [1, 2, 3]},
            {
                "type": "image_asset_pointer",
                "format": "png",
                "location": "https://example.com/foo.png",
                "expected_tokens": 42,
            },
            {"type": "encoded_text", "tokens": [4, 5]},
        ]
    )
    assert [c.tokens for c in mi.chunks] == [[1, 2, 3], [4, 5]]
    assert len(mi.image_chunks) == 1
    ic = mi.image_chunks[0]
    assert isinstance(ic, ImageAssetPointerChunk)
    assert ic.location == "https://example.com/foo.png"
    assert ic.format == "png"
    assert ic.expected_tokens == 42


def test_parse_sdk_unified_chunks_with_inline_image():
    """SDK ``image`` (inline bytes) chunk also routes to image_chunks."""
    b64 = base64.b64encode(_tiny_png_bytes()).decode("ascii")
    mi = ModelInput(
        chunks=[
            {"type": "encoded_text", "tokens": [7]},
            {"type": "image", "data": b64, "format": "png"},
        ]
    )
    assert len(mi.chunks) == 1
    assert len(mi.image_chunks) == 1
    assert isinstance(mi.image_chunks[0], EncodedImageChunk)


def test_split_chunks_still_accepted():
    """Existing Hatchery clients send chunks/image_chunks separately — still works."""
    b64 = base64.b64encode(_tiny_png_bytes()).decode("ascii")
    mi = ModelInput(
        chunks=[{"type": "encoded_text", "tokens": [1, 2]}],
        image_chunks=[{"type": "image", "data": b64}],
    )
    assert len(mi.chunks) == 1 and mi.chunks[0].tokens == [1, 2]
    assert len(mi.image_chunks) == 1


# ─── HTTP fetch helper ────────────────────────────────────────────────────


class _StubResponse:
    def __init__(self, *, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


class _StubAsyncClient:
    """Drop-in replacement for ``httpx.AsyncClient`` for tests.

    ``handler`` is called with the GET URL and must return a ``_StubResponse``
    (or raise an ``httpx.HTTPError`` subclass).
    """

    def __init__(self, handler, **_kwargs: Any) -> None:
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url: str):
        return self._handler(url)


@pytest.mark.asyncio
async def test_fetch_image_asset_http_success(monkeypatch):
    body = _tiny_png_bytes()

    def handler(url: str) -> _StubResponse:
        assert url == "https://example.com/img.png"
        return _StubResponse(status_code=200, content=body)

    # ``_fetch_image_asset`` does ``import httpx`` inside the function,
    # so patching ``httpx.AsyncClient`` on the module object reaches it.
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _StubAsyncClient(handler, **kw))

    out = await _fetch_image_asset("https://example.com/img.png")
    assert out == body


@pytest.mark.asyncio
async def test_fetch_image_asset_rejects_non_http(monkeypatch):
    # No monkeypatching: the scheme guard must fire before any fetch.
    with pytest.raises(Exception) as exc:
        await _fetch_image_asset("s3://bucket/key.png")
    # HTTPException subclasses Exception; check the detail.
    assert "scheme not supported" in str(exc.value) or "http/https only" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_fetch_image_asset_size_limit(monkeypatch):
    big = b"\x00" * (60 * 1024 * 1024)  # 60 MB > 50 MB cap

    def handler(_url: str) -> _StubResponse:
        return _StubResponse(status_code=200, content=big)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _StubAsyncClient(handler, **kw))
    with pytest.raises(Exception) as exc:
        await _fetch_image_asset("https://example.com/huge.png")
    assert "byte limit" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_fetch_image_asset_bad_status(monkeypatch):
    def handler(_url: str) -> _StubResponse:
        return _StubResponse(status_code=404, content=b"not found")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _StubAsyncClient(handler, **kw))
    with pytest.raises(Exception) as exc:
        await _fetch_image_asset("https://example.com/missing.png")
    assert "404" in str(exc.value.detail)


# ─── Resolver (pointer → encoded chunk) ───────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_image_asset_pointers_swaps_chunk(monkeypatch):
    body = _tiny_png_bytes()

    def handler(_url: str) -> _StubResponse:
        return _StubResponse(status_code=200, content=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _StubAsyncClient(handler, **kw))

    mi = ModelInput(
        chunks=[
            {"type": "encoded_text", "tokens": [1]},
            {
                "type": "image_asset_pointer",
                "format": "png",
                "location": "https://example.com/a.png",
            },
        ]
    )
    assert isinstance(mi.image_chunks[0], ImageAssetPointerChunk)
    await _resolve_image_asset_pointers(mi)
    assert len(mi.image_chunks) == 1
    resolved = mi.image_chunks[0]
    assert isinstance(resolved, EncodedImageChunk)
    assert resolved.data == body
    assert resolved.format == "png"
    assert resolved.mime_type == "image/png"


@pytest.mark.asyncio
async def test_resolve_image_asset_pointers_noop_when_empty():
    mi = ModelInput(chunks=[{"type": "encoded_text", "tokens": [1]}])
    await _resolve_image_asset_pointers(mi)  # must not raise, must not fetch
    assert mi.image_chunks == []


@pytest.mark.asyncio
async def test_datum_with_resolved_pointer_carries_bytes(monkeypatch):
    """After resolving, ``_datum_to_training_item`` surfaces raw bytes under ``images``."""
    from hatchery.core.tinker_compat import _datum_to_training_item

    body = _tiny_png_bytes()

    def handler(_url: str) -> _StubResponse:
        return _StubResponse(status_code=200, content=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _StubAsyncClient(handler, **kw))

    datum = Datum(
        model_input={
            "chunks": [
                {"type": "encoded_text", "tokens": [1, 2, 3]},
                {
                    "type": "image_asset_pointer",
                    "format": "jpeg",
                    "location": "https://example.com/b.jpg",
                },
            ]
        },
        loss_fn_inputs={},
    )
    await _resolve_image_asset_pointers(datum.model_input)
    item = _datum_to_training_item(datum)
    assert item["input_ids"] == [1, 2, 3]
    assert item["images"] == [body]


# ─── End-to-end via gateway ASGI app ─────────────────────────────────────


class _CapturingWorker:
    """FakeWorker that records every job payload it dequeues, then acks."""

    def __init__(self, config):
        self.config = config
        self.task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.seen_payloads: list[dict] = []

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = await self.config.queue.dequeue(
                    worker_id="fake",
                    model_filter=None,
                    visibility_timeout=60,
                )
            except asyncio.CancelledError:
                return
            if job is None:
                await asyncio.sleep(0.005)
                continue
            payload = msgpack.unpackb(job.payload, raw=False) if job.payload else {}
            self.seen_payloads.append({"op": job.operation, "payload": payload})
            response: dict[str, Any] = {}
            if job.operation == "init_session":
                response = {"status": "initialized"}
                await self.config.objects.put(
                    f"sessions/{job.session_id}/live_state/lora_weights.pt",
                    b"w",
                )
            elif job.operation == "forward_backward":
                response = {"loss": 0.5, "num_tokens": 3, "accum_steps": 1}
            await self.config.queue.ack(
                job.job_id,
                JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    result=msgpack.packb(response, use_bin_type=True),
                    metrics={"duration_ms": 1.0, "tokens": 3},
                ),
            )


@pytest_asyncio.fixture
async def capturing_client(platform_config):
    app = create_app(config=platform_config)
    worker = _CapturingWorker(platform_config)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        async with app.router.lifespan_context(app):
            worker.start()
            try:
                yield client, worker
            finally:
                await worker.stop()


@pytest.mark.asyncio
async def test_forward_backward_with_image_asset_pointer(capturing_client, monkeypatch):
    client, worker = capturing_client
    body = _tiny_png_bytes()

    def handler(url: str) -> _StubResponse:
        assert url == "https://cdn.example.com/pic.png"
        return _StubResponse(status_code=200, content=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _StubAsyncClient(handler, **kw))
    import hatchery.core.tinker_compat as _tc

    monkeypatch.setattr(_tc, "_is_private_ip", lambda _host: False)

    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "tinker-sess",
                "model_seq_id": 0,
                "base_model": "Qwen/Qwen2-0.5B",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    datum = {
        "model_input": {
            # SDK-unified wire format: one ``chunks`` list with a mix
            # of text + image_asset_pointer entries.
            "chunks": [
                {"type": "encoded_text", "tokens": [10, 11, 12]},
                {
                    "type": "image_asset_pointer",
                    "format": "png",
                    "location": "https://cdn.example.com/pic.png",
                },
            ],
        },
        "loss_fn_inputs": {
            "target_tokens": {"dtype": "int64", "shape": [3], "data": [10, 11, 12]},
            "weights": {"dtype": "float32", "shape": [3], "data": [1, 1, 1]},
        },
    }

    resp = await client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": mid,
            "seq_id": 0,
            "forward_backward_input": {
                "data": [datum],
                "loss_fn": "cross_entropy",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    fid = resp.json()["future_id"]
    resp = await client.post("/api/v1/retrieve_future", json={"future_id": fid})
    assert resp.status_code == 200

    # The worker must have observed the fetched bytes in place of the pointer.
    fb_payloads = [p for p in worker.seen_payloads if p["op"] == "forward_backward"]
    assert fb_payloads, "forward_backward never reached the worker"
    data = fb_payloads[0]["payload"]["data"]
    assert len(data) == 1
    assert data[0]["input_ids"] == [10, 11, 12]
    assert "images" in data[0], "resolved image bytes missing from worker payload"
    assert data[0]["images"] == [body]
