# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Multi-encoding compression middleware tests.

Covers the negotiation matrix and decode paths for brotli and zstd in
addition to gzip. The gzip-only tests live in
``test_gzip_middleware.py`` and still pass against the new middleware.

Note: httpx's ``TestClient`` transparently decompresses gzip, brotli, and
zstd response bodies when those packages are installed. We therefore
assert on ``r.headers["content-encoding"]`` to verify the negotiation
outcome rather than trying to re-decode ``r.content`` ourselves.
"""

from __future__ import annotations

import json

import brotli
import pytest
import zstandard
from fastapi import FastAPI
from starlette.testclient import TestClient

from hatchery.core.gateway import _install_compression_middleware


def _build_app() -> FastAPI:
    app = FastAPI()
    _install_compression_middleware(app)

    @app.post("/echo")
    async def echo(req: dict):
        # Return a body above HATCHERY_COMPRESS_MIN_SIZE (1024 bytes).
        return {"got": req, "pad": "x" * 4000}

    return app


def test_negotiation_prefers_zstd_over_br_and_gzip():
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={"accept-encoding": "gzip, br, zstd"},
    )
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "zstd"
    assert r.json()["got"] == {"x": 1}


def test_negotiation_prefers_br_when_zstd_not_offered():
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={"accept-encoding": "gzip, br"},
    )
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "br"
    assert r.json()["got"] == {"x": 1}


def test_gzip_only_still_served_as_gzip():
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={"accept-encoding": "gzip"},
    )
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"


def test_wildcard_accept_encoding_picks_best():
    client = TestClient(_build_app())
    r = client.post("/echo", json={"x": 1}, headers={"accept-encoding": "*"})
    assert r.status_code == 200
    # Wildcard → server picks its top preference (zstd when available).
    assert r.headers.get("content-encoding") == "zstd"


def test_q_zero_excludes_encoding():
    client = TestClient(_build_app())
    # Client explicitly refuses zstd. Server falls back to br.
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={"accept-encoding": "zstd;q=0, br, gzip"},
    )
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "br"


def test_below_min_size_not_compressed():
    app = FastAPI()
    _install_compression_middleware(app)

    @app.get("/small")
    async def small():
        return {"ok": True}

    r = TestClient(app).get("/small", headers={"accept-encoding": "zstd"})
    assert r.status_code == 200
    # Body is ~13 bytes — well below the 1024 floor.
    assert "content-encoding" not in r.headers


def test_vary_header_set():
    client = TestClient(_build_app())
    r = client.post("/echo", json={"x": 1}, headers={"accept-encoding": "zstd"})
    assert r.headers.get("vary") is not None
    assert "accept-encoding" in r.headers["vary"].lower()


def test_brotli_request_body_is_decoded():
    client = TestClient(_build_app())
    body = json.dumps({"x": list(range(50))}).encode()
    compressed = brotli.compress(body)
    r = client.post(
        "/echo",
        content=compressed,
        headers={
            "content-encoding": "br",
            "content-type": "application/json",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["got"]["x"]) == 50


def test_zstd_request_body_is_decoded():
    client = TestClient(_build_app())
    body = json.dumps({"x": list(range(50))}).encode()
    compressed = zstandard.ZstdCompressor(level=3).compress(body)
    r = client.post(
        "/echo",
        content=compressed,
        headers={
            "content-encoding": "zstd",
            "content-type": "application/json",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["got"]["x"]) == 50


def test_malformed_brotli_body_returns_400():
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        content=b"notbrotli",
        headers={
            "content-encoding": "br",
            "content-type": "application/json",
        },
    )
    assert r.status_code == 400
    assert "invalid br" in r.text


def test_malformed_zstd_body_returns_400():
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        content=b"notzstd",
        headers={
            "content-encoding": "zstd",
            "content-type": "application/json",
        },
    )
    assert r.status_code == 400
    assert "invalid zstd" in r.text


def test_identity_encoding_passes_through():
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        json={"x": 1},
        headers={"accept-encoding": "identity"},
    )
    assert r.status_code == 200
    assert "content-encoding" not in r.headers


def test_unknown_content_encoding_untouched():
    """An unknown request ``Content-Encoding`` flows through to the app
    (which typically returns 4xx from JSON parsing). The middleware must
    not turn this into a 500."""
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        content=b"not-a-known-encoding",
        headers={
            "content-encoding": "snappy",
            "content-type": "application/json",
        },
    )
    assert r.status_code < 500


@pytest.mark.parametrize("level", [1, 3, 9])
def test_zstd_compression_roundtrips_regardless_of_level(monkeypatch, level):
    monkeypatch.setenv("HATCHERY_COMPRESS_ZSTD_LEVEL", str(level))
    app = FastAPI()
    _install_compression_middleware(app)

    @app.post("/echo")
    async def echo(req: dict):
        return {"got": req, "pad": "x" * 4000}

    r = TestClient(app).post(
        "/echo",
        json={"x": 1},
        headers={"accept-encoding": "zstd"},
    )
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "zstd"
    assert r.json()["got"] == {"x": 1}


def test_existing_response_encoding_not_double_compressed():
    """If a handler sets its own ``Content-Encoding`` header, the
    middleware must leave the body alone."""
    from fastapi import Response

    app = FastAPI()
    _install_compression_middleware(app)

    @app.get("/preencoded")
    async def preencoded():
        # Body is big enough to trigger the min-size threshold.
        payload = b"x" * 4000
        return Response(
            content=payload,
            headers={"content-encoding": "identity"},
            media_type="application/octet-stream",
        )

    r = TestClient(app).get("/preencoded", headers={"accept-encoding": "zstd"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "identity"
