# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Msgpack transport middleware tests.

The middleware is strictly opt-in — clients that don't send
``Content-Type: application/msgpack`` or ``Accept: application/msgpack``
should see identical JSON behavior. Clients that do opt in should get
transparent unpack/repack.

Stacks with response compression: a client that sends
``Accept: application/msgpack`` + ``Accept-Encoding: zstd`` should get a
zstd-compressed msgpack body.
"""

from __future__ import annotations

import gzip
import json

import msgpack
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from hatchery.core.gateway import (
    _install_compression_middleware,
    _install_msgpack_middleware,
)


def _build_app(with_compression: bool = False) -> FastAPI:
    app = FastAPI()
    # Middleware order must match ``create_app``: msgpack first (inner)
    # so compression wraps it (outer).
    _install_msgpack_middleware(app)
    if with_compression:
        _install_compression_middleware(app)

    @app.post("/echo")
    async def echo(req: dict):
        # Response padded well above the compression min_size (1024)
        # so we can see content-encoding when it kicks in.
        return {"got": req, "pad": "x" * 4000}

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    return app


# ─── Pass-through behaviour (no opt-in) ───────────────────────────────────


def test_json_client_untouched():
    """No msgpack headers → plain JSON in, plain JSON out."""
    client = TestClient(_build_app())
    r = client.post("/echo", json={"x": 1})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["got"] == {"x": 1}


def test_get_request_untouched():
    """GET without Accept: msgpack returns JSON."""
    client = TestClient(_build_app())
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"pong": True}


# ─── Request side ─────────────────────────────────────────────────────────


def test_msgpack_request_body_unpacked():
    """Content-Type: application/msgpack → server sees the dict."""
    client = TestClient(_build_app())
    payload = msgpack.packb({"x": 42, "s": "hello"}, use_bin_type=True)
    r = client.post(
        "/echo",
        content=payload,
        headers={"content-type": "application/msgpack"},
    )
    assert r.status_code == 200
    assert r.json()["got"] == {"x": 42, "s": "hello"}


def test_invalid_msgpack_body_rejected():
    """Malformed msgpack body returns 400."""
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        content=b"\xff\xff\xff\xff not msgpack",
        headers={"content-type": "application/msgpack"},
    )
    assert r.status_code == 400


# ─── Response side ────────────────────────────────────────────────────────


def test_accept_msgpack_response_is_packed():
    """Accept: application/msgpack → msgpack bytes come back."""
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={"accept": "application/msgpack"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/msgpack")
    decoded = msgpack.unpackb(r.content, raw=False)
    assert decoded["got"] == {"x": 1}
    assert decoded["pad"] == "x" * 4000


def test_bidirectional_msgpack():
    """Both request and response in msgpack."""
    client = TestClient(_build_app())
    payload = msgpack.packb({"x": [1, 2, 3]}, use_bin_type=True)
    r = client.post(
        "/echo",
        content=payload,
        headers={
            "content-type": "application/msgpack",
            "accept": "application/msgpack",
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/msgpack")
    decoded = msgpack.unpackb(r.content, raw=False)
    assert decoded["got"] == {"x": [1, 2, 3]}


def test_accept_json_preferred_over_anything_is_still_json():
    """A client that asks for JSON gets JSON even though msgpack mw is on."""
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={"accept": "application/json"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["got"] == {"x": 1}


# ─── Stacked with compression ─────────────────────────────────────────────


def test_msgpack_plus_gzip_compresses_msgpack_bytes():
    """Accept: msgpack + Accept-Encoding: gzip → gzipped msgpack.

    TestClient auto-decompresses gzip, so ``r.content`` is the raw
    msgpack payload. We only assert the negotiation headers to confirm
    the stack chose gzip + msgpack.
    """
    client = TestClient(_build_app(with_compression=True))
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={
            "accept": "application/msgpack",
            "accept-encoding": "gzip",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"
    assert r.headers["content-type"].startswith("application/msgpack")
    decoded = msgpack.unpackb(r.content, raw=False)
    assert decoded["got"] == {"x": 1}


def test_msgpack_request_plus_gzip_request_body():
    """Content-Encoding: gzip + Content-Type: msgpack → decompress, unpack."""
    client = TestClient(_build_app(with_compression=True))
    payload = msgpack.packb({"x": 7}, use_bin_type=True)
    gzipped = gzip.compress(payload)
    r = client.post(
        "/echo",
        content=gzipped,
        headers={
            "content-type": "application/msgpack",
            "content-encoding": "gzip",
        },
    )
    assert r.status_code == 200
    assert r.json()["got"] == {"x": 7}


# ─── Payload size wins ────────────────────────────────────────────────────


def test_msgpack_is_smaller_than_json_for_binary_payload():
    """The real win for msgpack over JSON is binary data: where JSON
    forces a base64 round-trip (4 chars per 3 bytes), msgpack ships
    the raw bytes. This is why ``TensorData`` payloads benefit."""
    import base64

    raw = b"\x00\x01\x02\x03" * 2048  # 8 KB of binary
    json_payload = {"tensor_b64": base64.b64encode(raw).decode("ascii")}
    mp_payload = {"tensor_bytes": raw}
    json_bytes = json.dumps(json_payload).encode("utf-8")
    mp_bytes = msgpack.packb(mp_payload, use_bin_type=True)
    # Base64 adds ~33% overhead; msgpack bin has a tiny header.
    assert len(mp_bytes) < len(json_bytes), (
        f"msgpack ({len(mp_bytes)}) should beat JSON+b64 ({len(json_bytes)})"
    )
    # Raw size is 8192; msgpack should be within a few bytes of that.
    assert len(mp_bytes) < len(raw) + 64


# ─── Disable knob ─────────────────────────────────────────────────────────


def test_msgpack_disabled_via_env(monkeypatch):
    """HATCHERY_MSGPACK_ENABLED=0 skips middleware install.

    With the middleware off, sending msgpack to a JSON endpoint should
    fail: FastAPI's JSON body parser rejects msgpack bytes. Any 4xx/5xx
    counts as proof that the middleware didn't silently unpack it.
    """
    monkeypatch.setenv("HATCHERY_MSGPACK_ENABLED", "0")
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post(
        "/echo",
        content=msgpack.packb({"x": 1}, use_bin_type=True),
        headers={"content-type": "application/msgpack"},
    )
    assert r.status_code >= 400, r.status_code


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
