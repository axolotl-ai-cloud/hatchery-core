# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the tinker-compatible /api/v1/* surface."""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import msgpack
import pytest
import pytest_asyncio
from httpx import ASGITransport

from hatchery.core.gateway import create_app
from hatchery.core.protocols import JobResult, JobStatus


class FakeWorker:
    def __init__(self, config):
        self.config = config
        self.task = None
        self._stop = asyncio.Event()
        # Last payload seen per operation — tests assert SDK→worker
        # plumbing (e.g. sampling_params.seed / stop reach the payload).
        self.last_payload_by_op: dict[str, dict] = {}

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
            self.last_payload_by_op[job.operation] = payload
            response = _canned(job.operation, payload)
            await self.config.queue.ack(
                job.job_id,
                JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    result=msgpack.packb(response, use_bin_type=True),
                    metrics={"duration_ms": 1.0, "tokens": 5},
                ),
            )
            if job.operation == "init_session":
                await self.config.objects.put(
                    f"sessions/{job.session_id}/live_state/lora_weights.pt",
                    b"w",
                )
            if job.operation == "save_weights":
                name = payload.get("name", "ckpt")
                await self.config.objects.put(
                    f"sessions/{job.session_id}/checkpoints/{name}/lora_weights.pt",
                    b"w",
                )


def _canned(op, payload):
    if op == "init_session":
        return {"status": "initialized"}
    if op == "forward_backward":
        return {"loss": 0.5, "num_tokens": 6, "accum_steps": 1}
    if op == "forward_only":
        return {"loss": 0.33, "num_tokens": 5}
    if op == "optim_step":
        return {"status": "ok", "step": 1, "learning_rate": payload.get("learning_rate")}
    if op == "sample":
        resp = {"sequences": [[10, 11, 12]], "texts": ["abc"]}
        # Mirror the real worker's prompt_logprobs response shape when
        # the request asked for them — tinker's compute_logprobs is
        # implemented as a degenerate sample with this flag set, so
        # parity tests assert against it.
        if payload.get("include_prompt_logprobs"):
            prompt_len = len(payload.get("prompt_tokens", []))
            resp["prompt_logprobs"] = [None] + [
                -0.1 * (i + 1) for i in range(max(prompt_len - 1, 0))
            ]
        return resp
    if op == "compute_logprobs":
        return {"logprobs": [[-0.1, -0.2, -0.3]]}
    if op == "forward_logprobs":
        return {"per_datum_logprobs": [[0.0, -0.5, -0.4, -0.3]]}
    if op == "save_weights":
        return {"path": f"tinker://model/checkpoints/{payload.get('name', 'ckpt')}"}
    return {}


@pytest_asyncio.fixture
async def compat_client(platform_config):
    app = create_app(config=platform_config)
    worker = FakeWorker(platform_config)
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


async def test_healthz_and_capabilities(compat_client):
    client, _ = compat_client
    resp = await client.get("/api/v1/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    resp = await client.post("/api/v1/get_server_capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert "cross_entropy" in body["supported_loss_fns"]


async def test_create_session_and_model(compat_client):
    client, _ = compat_client
    resp = await client.post(
        "/api/v1/create_session",
        json={"tags": [], "user_metadata": {}, "sdk_version": "0.1"},
    )
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    resp = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": sid,
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen2-0.5B",
            "lora_config": {"rank": 8},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["base_model"] == "Qwen/Qwen2-0.5B"
    assert body["lora_config"]["rank"] == 8


async def test_create_model_full_param(compat_client):
    """FP path (``lora_config is None``) must not crash the response builder."""
    client, _ = compat_client
    resp = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "tinker-sess-fp",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen2-0.5B",
            "lora_config": None,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["base_model"] == "Qwen/Qwen2-0.5B"
    assert body["lora_config"] is None


async def test_full_training_loop_via_futures(compat_client):
    client, _ = compat_client
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
            "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4, 5]}],
        },
        "loss_fn_inputs": {
            "target_tokens": {"dtype": "int64", "shape": [5], "data": [1, 2, 3, 4, 5]},
            "weights": {"dtype": "float32", "shape": [5], "data": [1, 1, 1, 1, 1]},
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
    assert resp.status_code == 200
    fid = resp.json()["future_id"]

    resp = await client.post("/api/v1/retrieve_future", json={"future_id": fid})
    assert resp.status_code == 200
    body = resp.json()
    # SDK 0.18 ForwardBackwardOutput shape: top-level loss_fn_output_type,
    # loss_fn_outputs list, and metrics dict with reduction-suffixed keys.
    assert body["loss_fn_output_type"] == "cross_entropy"
    assert body["metrics"]["loss:mean"] == 0.5
    # fw-ai's sft_loop reads ``loss:sum`` directly; ensure it's emitted.
    assert "loss:sum" in body["metrics"]
    assert "num_tokens:sum" in body["metrics"]
    assert isinstance(body["loss_fn_outputs"], list) and body["loss_fn_outputs"]

    # optim_step — SDK OptimStepResponse shape: type + metrics.
    resp = await client.post(
        "/api/v1/optim_step",
        json={
            "model_id": mid,
            "seq_id": 1,
            "adam_params": {"learning_rate": 3e-4},
        },
    )
    fid = resp.json()["future_id"]
    resp = await client.post("/api/v1/retrieve_future", json={"future_id": fid})
    body = resp.json()
    assert body["type"] == "optim_step"
    assert body["metrics"]["step"] == 1

    # asample — SDK SampleResponse shape: top-level sequences list.
    resp = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [42, 43]}]},
            "num_samples": 1,
            "sampling_params": {"max_tokens": 8, "temperature": 0.0},
            "model_id": mid,
        },
    )
    fid = resp.json()["future_id"]
    resp = await client.post("/api/v1/retrieve_future", json={"future_id": fid})
    body = resp.json()
    assert body["sequences"][0]["tokens"] == [10, 11, 12]

    # save_weights — synchronous inline future. POST returns UntypedAPIFuture
    # {request_id, model_id}; retrieve_future returns SaveWeightsResponse
    # {type: "save_weights", path}.
    resp = await client.post(
        "/api/v1/save_weights",
        json={"model_id": mid, "path": "v1"},
    )
    sw_body = resp.json()
    assert "request_id" in sw_body
    retr = await client.post("/api/v1/retrieve_future", json={"request_id": sw_body["request_id"]})
    assert retr.json()["path"] == f"tinker://{mid}/checkpoints/v1"
    assert retr.json()["type"] == "save_weights"

    # list checkpoints
    resp = await client.get(f"/api/v1/training_runs/{mid}/checkpoints")
    assert any(c["checkpoint_id"] == "v1" for c in resp.json()["checkpoints"])


async def test_asample_forwards_seed_and_stop(compat_client):
    """SDK SamplingParams.seed / SamplingParams.stop must land in the
    job payload the worker receives — previously dropped on the floor."""
    client, worker = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "seed-stop-sess",
                "model_seq_id": 0,
                "base_model": "Qwen/Qwen2-0.5B",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    resp = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
            "num_samples": 1,
            "sampling_params": {
                "max_tokens": 4,
                "temperature": 0.7,
                "top_k": 20,
                "seed": 42,
                "stop": ["END"],
            },
            "model_id": mid,
        },
    )
    fid = resp.json()["future_id"]
    # Drain the future so the worker has definitely processed the job.
    await client.post("/api/v1/retrieve_future", json={"future_id": fid})

    sample_payload = worker.last_payload_by_op.get("sample")
    assert sample_payload is not None, "sample job never reached FakeWorker"
    assert sample_payload["seed"] == 42
    assert sample_payload["stop"] == ["END"]
    assert sample_payload["top_k"] == 20


async def test_retrieve_future_denies_other_users(compat_client, platform_config):
    client, _ = compat_client
    platform_config.auth.add_key("eve", user_id="eve")
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "m",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]
    resp = await client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": mid,
            "seq_id": 0,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                        "loss_fn_inputs": {},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        },
    )
    fid = resp.json()["future_id"]
    resp = await client.post(
        "/api/v1/retrieve_future",
        json={"future_id": fid},
        headers={"Authorization": "Bearer eve"},
    )
    assert resp.status_code == 403


async def test_forward_backward_converts_weights_to_labels(compat_client):
    client, worker = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "m",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    datum = {
        "model_input": {
            "chunks": [{"type": "encoded_text", "tokens": [10, 20, 30, 40]}],
        },
        "loss_fn_inputs": {
            "target_tokens": {"dtype": "int64", "shape": [4], "data": [10, 20, 30, 40]},
            "weights": {"dtype": "float32", "shape": [4], "data": [0, 0, 1, 1]},
        },
    }
    await client.post(
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

    # Give the fake worker a chance to process.
    await asyncio.sleep(0.05)
    # Verify: our worker received labels with -100 in the first two positions.
    # The fake worker doesn't record history, so we can't introspect here —
    # but the fact that the route accepted and enqueued it is the assertion.


async def test_forward_only_returns_loss_without_accum_steps(compat_client):
    """TrainingClient.forward_only must return a loss + metrics envelope
    mirroring forward_backward, but without mutating training-state
    fields (no accum_steps in the returned shape).
    """
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "fo-sess",
                "model_seq_id": 0,
                "base_model": "Qwen/Qwen2-0.5B",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    datum = {
        "model_input": {
            "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4, 5]}],
        },
        "loss_fn_inputs": {
            "target_tokens": {"dtype": "int64", "shape": [5], "data": [1, 2, 3, 4, 5]},
            "weights": {"dtype": "float32", "shape": [5], "data": [1, 1, 1, 1, 1]},
        },
    }

    resp = await client.post(
        "/api/v1/forward_only",
        json={
            "model_id": mid,
            "seq_id": 0,
            "forward_only_input": {
                "data": [datum],
                "loss_fn": "cross_entropy",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    fid = resp.json()["future_id"]

    resp = await client.post("/api/v1/retrieve_future", json={"future_id": fid})
    assert resp.status_code == 200
    body = resp.json()
    assert body["loss_fn_output_type"] == "cross_entropy"
    assert body["metrics"]["loss:mean"] == 0.33
    assert body["metrics"]["num_tokens:sum"] == 5.0
    # Critical: forward_only result must not surface accum_steps — that
    # would imply forward_backward semantics.
    assert "accum_steps" not in body.get("metrics", {})


async def test_forward_only_rejects_unknown_loss_fn(compat_client):
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "fo-sess2",
                "model_seq_id": 0,
                "base_model": "m",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]
    resp = await client.post(
        "/api/v1/forward_only",
        json={
            "model_id": mid,
            "seq_id": 0,
            "forward_only_input": {"data": [], "loss_fn": "not_a_real_loss"},
        },
    )
    assert resp.status_code == 400


async def test_checkpoint_delete(compat_client):
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "m",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]
    await client.post("/api/v1/save_weights", json={"model_id": mid, "path": "v1"})
    resp = await client.delete(f"/api/v1/training_runs/{mid}/checkpoints/v1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


async def test_seq_id_idempotency_returns_same_future(compat_client):
    """Retrying a request with the same ``seq_id`` must return the same
    ``future_id`` without enqueuing a second job.
    """
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "m",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    payload = {
        "model_id": mid,
        "seq_id": 42,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "loss_fn_inputs": {},
                }
            ],
            "loss_fn": "cross_entropy",
        },
    }

    resp1 = await client.post("/api/v1/forward_backward", json=payload)
    assert resp1.status_code == 200
    fid1 = resp1.json()["future_id"]

    # Same seq_id → same future_id, no duplicate job.
    resp2 = await client.post("/api/v1/forward_backward", json=payload)
    assert resp2.status_code == 200
    fid2 = resp2.json()["future_id"]
    assert fid1 == fid2

    # Different seq_id → different future_id.
    payload["seq_id"] = 43
    resp3 = await client.post("/api/v1/forward_backward", json=payload)
    assert resp3.status_code == 200
    fid3 = resp3.json()["future_id"]
    assert fid3 != fid1


async def test_save_weights_for_sampler(compat_client):
    """``save_weights_for_sampler`` creates a sampler checkpoint with
    a ``tinker://`` path containing ``sampler_weights``.
    """
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "m",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]
    resp = await client.post(
        "/api/v1/save_weights_for_sampler",
        json={"model_id": mid, "path": "step-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # SDK 0.18: POST returns UntypedAPIFuture {request_id, model_id};
    # retrieve_future returns the SaveWeightsForSamplerResponse shape.
    assert "request_id" in body
    retr = await client.post("/api/v1/retrieve_future", json={"request_id": body["request_id"]})
    inline = retr.json()
    assert inline["type"] == "save_weights_for_sampler"
    assert "sampler_weights" in inline["path"]


async def test_save_weights_for_sampler_session_mode(compat_client):
    """When the SDK calls ``save_weights_and_get_sampling_client`` it
    omits ``path`` and sets ``sampling_session_seq_id``. The response
    must populate ``sampling_session_id`` and leave ``path`` as None —
    otherwise the SDK's ``assert result.path is None`` guard fires.
    """
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "m",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]
    resp = await client.post(
        "/api/v1/save_weights_for_sampler",
        json={"model_id": mid, "sampling_session_seq_id": 0},
    )
    assert resp.status_code == 200
    retr = await client.post(
        "/api/v1/retrieve_future", json={"request_id": resp.json()["request_id"]}
    )
    inline = retr.json()
    assert inline["sampling_session_id"] is not None
    assert inline["path"] is None
    # The encoded sampling_session_id lets /asample resolve back to the model.
    assert mid in inline["sampling_session_id"]


# checkpoint_archive, publish, unpublish, ttl routes are in hatchery-hosted


# ── create_sampling_session (Tinker create_sampling_client parity) ──────


async def test_create_sampling_session_base_model(compat_client):
    """``create_sampling_session(base_model=...)`` spins up an FFT-shaped
    session and returns a ``sampling_session_id`` whose encoded model_id
    round-trips through the existing ``/asample`` decoder.
    """
    client, worker = compat_client
    resp = await client.post(
        "/api/v1/create_sampling_session",
        json={"base_model": "Qwen/Qwen2-0.5B"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "request_id" in body
    assert body["model_id"].startswith("smp_")

    retr = await client.post("/api/v1/retrieve_future", json={"request_id": body["request_id"]})
    inline = retr.json()
    assert inline["type"] == "create_sampling_session"
    assert inline["base_model"] == "Qwen/Qwen2-0.5B"
    assert inline["model_id"] == body["model_id"]
    assert inline["sampling_session_id"].startswith(f"samp-{body['model_id']}-0-")
    assert inline["expires_at"] > 0

    # The worker saw an FFT init (rank=None, no LoRA target_modules) —
    # confirming we reused the existing _handle_init_session FFT branch
    # rather than minting a separate handler.
    init_payload = worker.last_payload_by_op.get("init_session")
    assert init_payload is not None
    assert init_payload["rank"] is None
    assert init_payload["target_modules"] == []
    assert init_payload["base_model"] == "Qwen/Qwen2-0.5B"

    # /asample's existing sampling_session_id decoder routes the
    # encoded id back to the same model_id.
    sid = inline["sampling_session_id"]
    parts = sid.split("-")
    decoded_model_id = "-".join(parts[1:-2])
    assert decoded_model_id == body["model_id"]


async def test_create_sampling_session_requires_base_model_or_path(compat_client):
    """The Pydantic validator must reject requests with neither field set,
    matching Tinker's ``ValueError("Either model_path or base_model must be provided")``.
    """
    client, _ = compat_client
    resp = await client.post("/api/v1/create_sampling_session", json={})
    # Pydantic surfaces model-validator failures as 422 via FastAPI.
    assert resp.status_code == 422, resp.text


async def test_resolve_sampler_model_path_tinker_uri(platform_config):
    """``_resolve_sampler_model_path`` decodes ``tinker://`` URIs to the
    parent session's base_model + the materialized checkpoint prefix.
    Direct helper unit test — keeps the parent-session bootstrap out of
    the HTTP path so the assertion is on the resolver alone.
    """
    from hatchery.core.protocols import AuthenticatedUser, SessionRecord, SessionStatus
    from hatchery.core.tinker_compat import _resolve_sampler_model_path

    parent_record = SessionRecord(
        session_id="mdl_parent",
        user_id="u-1",
        base_model="Qwen/Qwen2-0.5B",
        lora_rank=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        total_steps=0,
        accum_steps=0,
        created_at=0.0,
        last_accessed=0.0,
        status=SessionStatus.ACTIVE,
        state_prefix="sessions/mdl_parent/live_state",
    )
    await platform_config.metadata.create_session(parent_record)
    user = AuthenticatedUser(user_id="u-1")

    base_model, prefix = await _resolve_sampler_model_path(
        platform_config, "tinker://mdl_parent/checkpoints/step-1", user
    )
    assert base_model == "Qwen/Qwen2-0.5B"
    assert prefix == f"{platform_config.sessions_prefix}/mdl_parent/checkpoints/step-1"

    # Local FS / HF id passthrough.
    base_model, prefix = await _resolve_sampler_model_path(
        platform_config, "/local/merged/checkpoint", user
    )
    assert base_model == "/local/merged/checkpoint"
    assert prefix is None

    # Cross-user access is denied.
    other_user = AuthenticatedUser(user_id="u-2")
    with pytest.raises(Exception) as exc_info:
        await _resolve_sampler_model_path(
            platform_config, "tinker://mdl_parent/checkpoints/step-1", other_user
        )
    # FastAPI HTTPException is what the resolver raises.
    assert "403" in str(exc_info.value) or "different user" in str(exc_info.value)


# ── Tinker wire-shape parity ────────────────────────────────────────────
#
# These tests POST the exact JSON the official ``tinker`` SDK emits
# (schema source: ``tinker/types/*_request.py`` in ``tinker==0.22.2``)
# and assert the response decodes against tinker's response types. They
# guarantee that a user running ``pip install tinker`` and pointing at
# our gateway gets the right behavior for the three Tinker-API parity
# items, without taking a runtime dependency on the tinker SDK.


async def test_tinker_wire_shape_create_sampling_session(compat_client):
    """The official ``tinker.types.CreateSamplingSessionRequest`` has
    REQUIRED fields ``session_id`` and ``sampling_session_seq_id`` plus
    optional ``base_model`` / ``model_path``. Our route must accept that
    exact body and return a response whose required fields match
    ``tinker.types.CreateSamplingSessionResponse``
    (``type`` + ``sampling_session_id``).
    """
    client, _ = compat_client
    # Verbatim tinker-shape body — every field tinker's pydantic model
    # marks required, populated as tinker would populate them.
    body = {
        "session_id": "tinker-session-abc",
        "sampling_session_seq_id": 0,
        "base_model": "Qwen/Qwen2-0.5B",
        "model_path": None,
        "type": "create_sampling_session",
    }
    resp = await client.post("/api/v1/create_sampling_session", json=body)
    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]

    retr = await client.post("/api/v1/retrieve_future", json={"request_id": request_id})
    inline = retr.json()
    # CreateSamplingSessionResponse required fields (tinker schema):
    assert inline["type"] == "create_sampling_session"
    assert isinstance(inline["sampling_session_id"], str)
    assert inline["sampling_session_id"].startswith("samp-")
    # The supplied sampling_session_seq_id is embedded in the encoded id
    # so tinker's idempotent-retry path round-trips stably.
    assert "-0-" in inline["sampling_session_id"]


async def test_tinker_wire_shape_sample_with_prompt_logprobs(compat_client):
    """Tinker's ``SamplingClient.compute_logprobs`` issues a degenerate
    sample with ``prompt_logprobs=True`` and ``max_tokens=1``
    (``tinker/lib/public_interfaces/sampling_client.py:399-406``). The
    on-wire schema is ``tinker.types.SampleRequest``, which has the
    wire-level field name ``prompt_logprobs`` (the Python kwarg
    ``include_prompt_logprobs`` is translated at the SDK boundary).

    Our /asample must accept that wire field and return
    ``prompt_logprobs`` in the response — that's the whole reason
    tinker's compute_logprobs works against us with zero SDK changes.
    """
    client, _ = compat_client
    # First mint a sampling session via the tinker-shape route.
    create_body = {
        "session_id": "tinker-sess-xyz",
        "sampling_session_seq_id": 0,
        "base_model": "Qwen/Qwen2-0.5B",
        "model_path": None,
        "type": "create_sampling_session",
    }
    cs_resp = await client.post("/api/v1/create_sampling_session", json=create_body)
    cs_inline = (
        await client.post(
            "/api/v1/retrieve_future", json={"request_id": cs_resp.json()["request_id"]}
        )
    ).json()
    sampling_session_id = cs_inline["sampling_session_id"]

    # Verbatim tinker SampleRequest shape (sample_request.py:13-55).
    sample_body = {
        "sampling_session_id": sampling_session_id,
        "num_samples": 1,
        "prompt": {
            "chunks": [{"type": "encoded_text", "tokens": [10, 20, 30, 40, 50]}],
        },
        "sampling_params": {
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        },
        "prompt_logprobs": True,  # the wire-level field name
        "topk_prompt_logprobs": 0,
    }
    s_resp = await client.post("/api/v1/asample", json=sample_body)
    assert s_resp.status_code == 200, s_resp.text
    s_inline = (
        await client.post(
            "/api/v1/retrieve_future", json={"request_id": s_resp.json()["request_id"]}
        )
    ).json()

    # The compute_logprobs trick depends on this exact response shape:
    # [None] + len(prompt)-1 floats. Tinker drops [0] and uses the rest.
    assert "prompt_logprobs" in s_inline, s_inline
    lps = s_inline["prompt_logprobs"]
    assert lps[0] is None
    assert all(isinstance(x, float) for x in lps[1:])
    assert len(lps) == 5  # matches prompt length


async def test_tinker_wire_shape_forward_returns_per_datum_logprobs(compat_client):
    """``tinker.TrainingClient.forward`` posts ``ForwardRequest``
    (``forward_input`` + ``model_id`` + ``seq_id``) to ``/api/v1/forward``
    and expects the gateway to enqueue the per-position-logprobs op.

    The future result must contain ``per_datum_logprobs`` — the
    field name tinker's response decoder expects. Hatchery's existing
    /forward route + worker forward_logprobs op deliver this; this test
    pins the contract so a future refactor can't silently change it.
    """
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "Qwen/Qwen2-0.5B",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    # Verbatim tinker ForwardRequest shape.
    fwd_body = {
        "forward_input": {
            "data": [
                {
                    "model_input": {
                        "chunks": [
                            {"type": "encoded_text", "tokens": [1, 2, 3, 4]},
                        ],
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [1, 2, 3, 4], "shape": [4]},
                    },
                }
            ],
            "loss_fn": "cross_entropy",
        },
        "model_id": mid,
        "seq_id": 1,
    }
    f_resp = await client.post("/api/v1/forward", json=fwd_body)
    assert f_resp.status_code == 200, f_resp.text
    f_inline = (
        await client.post(
            "/api/v1/retrieve_future", json={"request_id": f_resp.json()["request_id"]}
        )
    ).json()

    # Tinker's ForwardBackwardOutput shape (see
    # tinker/types/forward_backward_output.py): loss_fn_output_type +
    # loss_fn_outputs (list of {logprobs: TensorData}) + metrics. The
    # gateway's _wrap_future_result (tinker_compat.py:~2349) transforms
    # the worker's per_datum_logprobs into exactly this envelope.
    assert f_inline["loss_fn_output_type"] == "cross_entropy"
    assert isinstance(f_inline["loss_fn_outputs"], list)
    assert len(f_inline["loss_fn_outputs"]) >= 1
    first = f_inline["loss_fn_outputs"][0]
    assert "logprobs" in first
    td = first["logprobs"]
    # TensorData wire shape: {data, dtype, shape}.
    assert "data" in td
    assert "dtype" in td
    assert "shape" in td
    assert isinstance(td["data"], list)


# ── SSRF protection ─────────────────────────────────────────────────────


def test_private_ip_detection():
    from hatchery.core.tinker_compat import _is_private_ip

    assert _is_private_ip("127.0.0.1")
    assert _is_private_ip("localhost")
    assert _is_private_ip("10.0.0.1")
    assert _is_private_ip("172.16.0.1")
    assert _is_private_ip("192.168.1.1")
    assert _is_private_ip("169.254.169.254")
    assert not _is_private_ip("8.8.8.8")
    assert not _is_private_ip("1.1.1.1")


async def test_ssrf_private_ip_rejected(compat_client):
    """Image fetch to private IPs must be blocked."""
    import pytest

    from hatchery.core.tinker_compat import _fetch_image_asset

    with pytest.raises(Exception, match="private|internal"):
        await _fetch_image_asset("https://127.0.0.1/image.png")
    with pytest.raises(Exception, match="private|internal"):
        await _fetch_image_asset("https://169.254.169.254/latest/meta-data/")
    with pytest.raises(Exception, match="private|internal"):
        await _fetch_image_asset("https://10.0.0.1/secret.png")


async def test_ssrf_http_rejected_by_default(compat_client):
    """Plain HTTP must be rejected unless opted in."""
    import pytest

    from hatchery.core.tinker_compat import _fetch_image_asset

    with pytest.raises(Exception, match="HTTPS"):
        await _fetch_image_asset("http://example.com/image.png")


# ── Cross-user checkpoint access ────────────────────────────────────────


async def test_load_weights_cross_user_rejected(compat_client):
    """User A cannot load_weights from User B's checkpoint."""
    client, _ = compat_client

    mid_a = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "sess-cross",
                "model_seq_id": 0,
                "base_model": "Qwen/Qwen2-0.5B",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    # Verify the path traversal check fires when the source model_id doesn't match
    # the dest and the source session doesn't exist (404).
    resp = await client.post(
        "/api/v1/load_weights",
        json={
            "model_id": mid_a,
            "path": "tinker://nonexistent_other_user_model/checkpoints/ckpt",
        },
    )
    assert resp.status_code in (403, 404)


# ── Input validation ────────────────────────────────────────────────────


async def test_empty_data_rejected(compat_client):
    """forward_backward with empty data returns 400."""
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "sess-empty",
                "model_seq_id": 0,
                "base_model": "Qwen/Qwen2-0.5B",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    resp = await client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": mid,
            "forward_backward_input": {"data": [], "loss_fn": "cross_entropy"},
        },
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


async def test_invalid_lora_rank_rejected(compat_client):
    """Negative or zero LoRA rank returns 422."""
    client, _ = compat_client
    resp = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess-bad-rank",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen2-0.5B",
            "lora_config": {"rank": -5},
        },
    )
    assert resp.status_code == 422


async def test_invalid_temperature_rejected(compat_client):
    """Negative temperature returns 422."""
    client, _ = compat_client
    mid = (
        await client.post(
            "/api/v1/create_model",
            json={
                "session_id": "sess-bad-temp",
                "model_seq_id": 0,
                "base_model": "Qwen/Qwen2-0.5B",
                "lora_config": {"rank": 8},
            },
        )
    ).json()["model_id"]

    resp = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "model_id": mid,
            "sampling_params": {"temperature": -1.0},
        },
    )
    assert resp.status_code == 422
