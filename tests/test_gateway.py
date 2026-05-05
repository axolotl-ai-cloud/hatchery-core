# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""End-to-end tests for the FastAPI gateway using a fake worker.

The gateway is stateless and doesn't touch the GPU, so we can exercise
every route with a synthetic worker that ACKs jobs deterministically.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import msgpack
import pytest_asyncio
from httpx import ASGITransport

from hatchery.core.gateway import create_app
from hatchery.core.protocols import JobResult, JobStatus

# ─── Fake worker ──────────────────────────────────────────────────────────


class FakeWorker:
    """Drains the queue and ACKs every job with a canned response."""

    def __init__(self, config):
        self.config = config
        self.processed: list[dict] = []
        self.task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self):
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def _loop(self):
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
            self.processed.append(
                {"op": job.operation, "session_id": job.session_id, "payload": payload}
            )
            response = _canned_response(job.operation, payload)
            await self.config.queue.ack(
                job.job_id,
                JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    result=msgpack.packb(response, use_bin_type=True),
                    metrics={"duration_ms": 1.0, "tokens": 42},
                ),
            )
            # Simulate worker writing live state for save_weights to succeed.
            if job.operation == "init_session":
                await self.config.objects.put(
                    f"sessions/{job.session_id}/live_state/lora_weights.pt",
                    b"fake-weights",
                )


def _canned_response(op: str, payload: dict) -> dict:
    if op == "init_session":
        return {"status": "initialized"}
    if op == "forward_backward":
        return {"loss": 0.5, "num_tokens": 10, "accum_steps": 1}
    if op == "forward_only":
        return {"loss": 0.25, "num_tokens": 7}
    if op == "optim_step":
        return {"status": "ok", "step": 1, "learning_rate": payload.get("learning_rate")}
    if op == "sample":
        return {"sequences": [[1, 2, 3]], "texts": ["hello"]}
    if op == "compute_logprobs":
        return {"logprobs": [[-0.1, -0.2]]}
    return {}


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def gateway_client(platform_config):
    app = create_app(config=platform_config)
    transport = ASGITransport(app=app)
    worker = FakeWorker(platform_config)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        # Manually drive lifespan so in-memory queue/metadata get initialized.
        async with app.router.lifespan_context(app):
            worker.start()
            try:
                yield client, worker, platform_config
            finally:
                await worker.stop()


# ─── Tests ────────────────────────────────────────────────────────────────


async def test_health(gateway_client):
    client, _, _ = gateway_client
    resp = await client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_auth_required(gateway_client):
    client, _, _ = gateway_client
    resp = await client.post(
        "/v1/sessions",
        json={"base_model": "m", "rank": 8},
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401


async def test_create_session_flow(gateway_client):
    client, worker, _ = gateway_client
    resp = await client.post(
        "/v1/sessions",
        json={"base_model": "Qwen/Qwen2-0.5B", "rank": 16},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "active"
    sid = body["session_id"]

    # Subsequent get/list should reflect the session.
    resp = await client.get(f"/v1/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json()["base_model"] == "Qwen/Qwen2-0.5B"

    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200
    assert any(s["session_id"] == sid for s in resp.json()["sessions"])


async def test_quota_enforcement(gateway_client, platform_config):
    # Lower the user's quota to 1 concurrent session.
    platform_config.auth.add_key(
        "cheap",
        user_id="user-cheap",
        max_concurrent_sessions=1,
        max_rank=8,
    )
    client, _, _ = gateway_client
    headers = {"Authorization": "Bearer cheap"}
    resp = await client.post(
        "/v1/sessions",
        json={"base_model": "m", "rank": 4},
        headers=headers,
    )
    assert resp.status_code == 200

    resp = await client.post(
        "/v1/sessions",
        json={"base_model": "m", "rank": 4},
        headers=headers,
    )
    assert resp.status_code == 429


async def test_rank_quota(gateway_client, platform_config):
    platform_config.auth.add_key(
        "smallrank",
        user_id="user-smallrank",
        max_concurrent_sessions=5,
        max_rank=8,
    )
    client, _, _ = gateway_client
    resp = await client.post(
        "/v1/sessions",
        json={"base_model": "m", "rank": 64},
        headers={"Authorization": "Bearer smallrank"},
    )
    assert resp.status_code == 403


async def test_save_and_list_checkpoints(gateway_client):
    client, worker, _ = gateway_client
    resp = await client.post("/v1/sessions", json={"base_model": "m", "rank": 8})
    sid = resp.json()["session_id"]

    resp = await client.post("/v1/save_weights", json={"session_id": sid, "name": "ckpt1"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["path"] == f"tinker://{sid}/checkpoints/ckpt1"

    resp = await client.get(f"/v1/sessions/{sid}/checkpoints")
    assert resp.status_code == 200
    assert "ckpt1" in resp.json()["checkpoints"]


async def test_delete_session(gateway_client):
    client, _, _ = gateway_client
    sid = (await client.post("/v1/sessions", json={"base_model": "m", "rank": 8})).json()[
        "session_id"
    ]
    resp = await client.delete(f"/v1/sessions/{sid}")
    assert resp.status_code == 200
    # Subsequent access should fail.
    resp = await client.get(f"/v1/sessions/{sid}")
    assert resp.status_code == 410


async def test_cross_user_access_denied(gateway_client, platform_config):
    platform_config.auth.add_key("bob", user_id="bob")
    client, _, _ = gateway_client
    resp = await client.post("/v1/sessions", json={"base_model": "m", "rank": 8})
    sid = resp.json()["session_id"]
    resp = await client.get(f"/v1/sessions/{sid}", headers={"Authorization": "Bearer bob"})
    assert resp.status_code == 403


async def test_resume_session(gateway_client, platform_config):
    client, _, _ = gateway_client
    sid = (await client.post("/v1/sessions", json={"base_model": "m", "rank": 8})).json()[
        "session_id"
    ]
    # Suspend manually.
    from hatchery.core.protocols import SessionStatus

    await platform_config.metadata.update_session(sid, status=SessionStatus.SUSPENDED)
    resp = await client.post(f"/v1/sessions/{sid}/resume")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == sid


# ── Request body size limits ────────────────────────────────────────────


async def test_oversized_msgpack_body_rejected(gateway_client):
    """Msgpack bodies exceeding the size limit get 413."""
    import msgpack as mp

    client, _, _ = gateway_client
    # Default limit is 50MB; send a ~1MB payload with msgpack content type
    # after temporarily lowering the limit.
    import hatchery.core.gateway as gw

    original = gw._MAX_REQUEST_BODY
    gw._MAX_REQUEST_BODY = 100  # 100 bytes
    try:
        body = mp.packb({"data": "x" * 200}, use_bin_type=True)
        resp = await client.post(
            "/v1/health",
            content=body,
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/msgpack",
            },
        )
        assert resp.status_code == 413
    finally:
        gw._MAX_REQUEST_BODY = original
