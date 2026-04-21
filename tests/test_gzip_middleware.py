# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Gzip request/response middleware for the gateway.

Covers the four behaviors callers rely on:

* Uncompressed JSON requests still work (default path).
* Requests with ``Content-Encoding: gzip`` are transparently decoded.
* Responses are gzipped when the client sends ``Accept-Encoding: gzip``
  and the body exceeds ``HATCHERY_GZIP_MIN_SIZE``.
* A malformed gzip body → 400 (not a silent 500).
"""

from __future__ import annotations

import gzip
import json

from fastapi import FastAPI
from starlette.testclient import TestClient

from hatchery.core.gateway import _install_compression_middleware


def _build_app() -> FastAPI:
    app = FastAPI()
    _install_compression_middleware(app)

    @app.post("/echo")
    async def echo(req: dict):
        return {"got": req}

    return app


def test_plain_json_request_still_works():
    client = TestClient(_build_app())
    r = client.post("/echo", json={"x": 1})
    assert r.status_code == 200
    assert r.json() == {"got": {"x": 1}}


def test_gzip_request_body_is_decoded():
    client = TestClient(_build_app())
    body = json.dumps({"x": list(range(100))}).encode()
    gz = gzip.compress(body)
    r = client.post(
        "/echo",
        content=gz,
        headers={"content-encoding": "gzip", "content-type": "application/json"},
    )
    assert r.status_code == 200
    assert len(r.json()["got"]["x"]) == 100


def test_response_is_gzipped_when_accepted():
    client = TestClient(_build_app())
    # TestClient will transparently decompress gzipped responses, so we
    # just verify the Content-Encoding header the server set.
    r = client.post(
        "/echo",
        json={"big": "x" * 5000},
        headers={"accept-encoding": "gzip"},
    )
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"


def test_malformed_gzip_body_returns_400():
    client = TestClient(_build_app())
    r = client.post(
        "/echo",
        content=b"notgzip",
        headers={"content-encoding": "gzip", "content-type": "application/json"},
    )
    assert r.status_code == 400
    assert "invalid gzip" in r.text
