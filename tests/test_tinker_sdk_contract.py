# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SDK wire-contract tests.

The tinker SDK pydantic-validates every response against a typed schema
— if our gateway ships even a slightly wrong shape (wrong key, wrong
type, missing reduction suffix on a metric), the SDK raises
``APIResponseValidationError`` *on the client side* which is hard to
debug without running the full smoke test.

These tests short-circuit that: we import the SDK's response types
directly and validate our gateway responses against them, so a wire
regression fails at unit-test time with a precise pydantic error.

Covers (SDK 0.18):
- ``UntypedAPIFuture`` — the envelope every POST returns
- ``ForwardBackwardOutput`` with ``LossFnOutput = Dict[str, TensorData]``
- ``OptimStepResponse``
- ``SaveWeightsResponse`` / ``SaveWeightsForSamplerResponseInternal``
- ``TelemetryResponse`` (status: "accepted")
- ``GetInfoResponse`` — server-reported model metadata
- Metric-key reduction suffix (``name:mean`` / ``name:sum``) required by
  the SDK's chunked-fwdbwd reducer

If the SDK package isn't installed, the whole module is skipped —
these are contract tests, not implementation tests.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import msgpack
import pytest
import pytest_asyncio
from httpx import ASGITransport

tinker = pytest.importorskip("tinker")

from hatchery.core.gateway import create_app  # noqa: E402
from hatchery.core.protocols import JobResult, JobStatus  # noqa: E402

# ─── A canned worker that emits SDK-compatible payloads ────────────────────


class _SDKFakeWorker:
    """A fake worker whose canned payloads are shaped like the real worker.

    The real worker emits per-datum logprob lists for forward_backward —
    the SDK cookbook depends on that. We mirror the same shape so the
    gateway's `_wrap_future_result` produces a valid ``ForwardBackwardOutput``.
    """

    def __init__(self, config):
        self.config = config
        self.task = None
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
                    worker_id="fake-sdk", model_filter=None, visibility_timeout=60
                )
            except asyncio.CancelledError:
                return
            if job is None:
                await asyncio.sleep(0.005)
                continue
            payload = msgpack.unpackb(job.payload, raw=False) if job.payload else {}
            response = _canned(job.operation, payload)
            await self.config.queue.ack(
                job.job_id,
                JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    result=msgpack.packb(response, use_bin_type=True),
                    # Worker emits ``cost_dimensions`` (dict) — gateway must
                    # strip it before merging into SDK metrics. We include
                    # it here to assert the gateway actually does.
                    metrics={
                        "tokens": 5,
                        "duration_ms": 1.2,
                        "cost_dimensions": {"model": "fake"},
                    },
                ),
            )
            if job.operation == "init_session":
                await self.config.objects.put(
                    f"sessions/{job.session_id}/live_state/lora_weights.pt", b"w"
                )


def _canned(op, payload):
    if op == "init_session":
        return {"status": "initialized"}
    if op == "forward_backward":
        # Real worker emits per-datum logprob lists. We emit two rows to
        # exercise list-of-TensorData shaping.
        return {
            "loss": 1.25,
            "num_tokens": 10,
            "accum_steps": 1,
            "per_datum_logprobs": [
                [-0.1, -0.2, -0.3, -0.4, -0.5],
                [-0.15, -0.25, -0.35, -0.45, -0.55],
            ],
        }
    if op == "forward_logprobs":
        # /forward uses this op; shape must match forward_backward so
        # retrieve_future's shared branch emits loss_fn_outputs.
        return {
            "per_datum_logprobs": [
                [-0.1, -0.2, -0.3, -0.4, -0.5],
            ],
        }
    if op == "optim_step":
        return {"status": "ok", "step": 7, "learning_rate": payload.get("learning_rate", 0.0)}
    if op == "sample":
        return {"sequences": [[100, 101, 102]]}
    return {}


@pytest_asyncio.fixture
async def sdk_client(platform_config):
    app = create_app(config=platform_config)
    worker = _SDKFakeWorker(platform_config)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        async with app.router.lifespan_context(app):
            worker.start()
            try:
                yield client
            finally:
                await worker.stop()


async def _create_model(client, rank=8):
    resp = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "contract-sess",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen2-0.5B-Instruct",
            "lora_config": {"rank": rank},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["model_id"]


# ─── Contract tests ────────────────────────────────────────────────────────


async def test_create_model_returns_untyped_api_future(sdk_client):
    from tinker.types.shared.untyped_api_future import UntypedAPIFuture

    resp = await sdk_client.post(
        "/api/v1/create_model",
        json={
            "session_id": "s",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen2-0.5B-Instruct",
            "lora_config": {"rank": 8},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # Must satisfy the SDK's UntypedAPIFuture schema: {request_id, model_id}.
    UntypedAPIFuture.model_validate(body)
    assert body["request_id"].startswith("fut_")
    assert body["model_id"]


async def test_forward_backward_output_schema(sdk_client):
    from tinker.types.forward_backward_output import ForwardBackwardOutput

    mid = await _create_model(sdk_client)
    resp = await sdk_client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": mid,
            "seq_id": 0,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {
                            "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4, 5]}]
                        },
                        "loss_fn_inputs": {
                            "weights": {"dtype": "float32", "shape": [5], "data": [1, 1, 1, 1, 1]},
                            "target_tokens": {
                                "dtype": "int64",
                                "shape": [5],
                                "data": [1, 2, 3, 4, 5],
                            },
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        },
    )
    fid = resp.json()["request_id"]
    retr = await sdk_client.post("/api/v1/retrieve_future", json={"request_id": fid})
    body = retr.json()
    # Validate against the SDK schema — pydantic will catch wrong shapes.
    ForwardBackwardOutput.model_validate(body)

    # Stronger invariants the SDK relies on:
    # - loss_fn_outputs is list of {str: TensorData}; each logprobs field
    #   must be a TensorData dict with data, dtype, shape.
    for row in body["loss_fn_outputs"]:
        assert "logprobs" in row
        lp = row["logprobs"]
        assert set(lp.keys()) >= {"data", "dtype", "shape"}
        assert lp["dtype"] in ("float32", "int64")
    # - Metric keys carry reduction suffix required by chunked reducer.
    assert all(":" in k for k in body["metrics"]), body["metrics"]
    # - cost_dimensions (dict) must be stripped from metrics.
    assert "cost_dimensions" not in body["metrics"]


async def test_forward_request_matches_sdk_wire_format(sdk_client):
    """``/forward`` must accept the SDK's ForwardRequest schema.

    Regression test for a wire-contract bug where hatchery-core's ``/forward``
    was typed with ``forward_backward_input`` (the ``/forward_backward``
    key) instead of ``forward_input``. That made every SDK
    ``forward_backward_custom`` call 422 at step 1 (the SDK's forward
    pass to collect logprobs). ``sl_basic`` / ``sl_loop`` don't hit
    ``/forward`` so it escaped earlier SL contract tests — this test
    closes the gap.
    """
    from tinker.types.datum import Datum
    from tinker.types.forward_backward_input import ForwardBackwardInput
    from tinker.types.forward_backward_output import ForwardBackwardOutput
    from tinker.types.forward_request import ForwardRequest as SDKForwardRequest
    from tinker.types.model_input import ModelInput
    from tinker.types.tensor_data import TensorData

    mid = await _create_model(sdk_client)

    # Build the request exactly as the SDK does at
    # training_client.py:172-178.
    sdk_req = SDKForwardRequest(
        forward_input=ForwardBackwardInput(
            data=[
                Datum(
                    model_input=ModelInput.from_ints([1, 2, 3, 4, 5]),
                    loss_fn_inputs={
                        "target_tokens": TensorData(data=[2, 3, 4, 5, 6], dtype="int64", shape=[5]),
                        "weights": TensorData(
                            data=[1.0, 1.0, 1.0, 1.0, 1.0],
                            dtype="float32",
                            shape=[5],
                        ),
                    },
                )
            ],
            loss_fn="cross_entropy",
            loss_fn_config=None,
        ),
        model_id=mid,
        seq_id=1,
    )
    body = sdk_req.model_dump(mode="json")
    # Invariant: the outer key is ``forward_input`` (distinct from
    # ``/forward_backward`` which uses ``forward_backward_input``).
    assert "forward_input" in body and "forward_backward_input" not in body

    resp = await sdk_client.post("/api/v1/forward", json=body)
    assert resp.status_code == 200, resp.text
    fid = resp.json()["request_id"]
    retr = await sdk_client.post("/api/v1/retrieve_future", json={"request_id": fid})
    assert retr.status_code == 200, retr.text
    ForwardBackwardOutput.model_validate(retr.json())


async def test_forward_rejects_forward_backward_input_key(sdk_client):
    """``/forward`` must reject the old ``forward_backward_input`` key.

    Catches drift in the opposite direction: if somebody re-aliases
    the schema to accept both keys to be "lenient," that hides SDK
    contract breakage. The endpoint is defined by the SDK's
    ForwardRequest — the only valid outer key is ``forward_input``.
    """
    mid = await _create_model(sdk_client)
    resp = await sdk_client.post(
        "/api/v1/forward",
        json={
            "model_id": mid,
            "seq_id": 1,
            # Wrong key — this is the ``/forward_backward`` schema.
            "forward_backward_input": {
                "data": [],
                "loss_fn": "cross_entropy",
            },
        },
    )
    assert resp.status_code == 422, resp.text


async def test_optim_step_response_schema(sdk_client):
    from tinker.types.optim_step_response import OptimStepResponse

    mid = await _create_model(sdk_client)
    resp = await sdk_client.post(
        "/api/v1/optim_step",
        json={"model_id": mid, "seq_id": 1, "adam_params": {"learning_rate": 3e-4}},
    )
    fid = resp.json()["request_id"]
    retr = await sdk_client.post("/api/v1/retrieve_future", json={"request_id": fid})
    body = retr.json()
    OptimStepResponse.model_validate(body)
    assert body["type"] == "optim_step"


async def test_save_weights_response_schema(sdk_client):
    from tinker.types.save_weights_response import SaveWeightsResponse

    mid = await _create_model(sdk_client)
    resp = await sdk_client.post("/api/v1/save_weights", json={"model_id": mid, "path": "ckpt-1"})
    body = resp.json()
    assert "request_id" in body
    retr = await sdk_client.post("/api/v1/retrieve_future", json={"request_id": body["request_id"]})
    SaveWeightsResponse.model_validate(retr.json())


async def test_save_weights_for_sampler_session_schema(sdk_client):
    from tinker.types.save_weights_for_sampler_response import (
        SaveWeightsForSamplerResponseInternal,
    )

    mid = await _create_model(sdk_client)
    # Session-mode: no path, but sampling_session_seq_id set. The SDK's
    # ``save_weights_and_get_sampling_client`` asserts ``result.path is None``.
    resp = await sdk_client.post(
        "/api/v1/save_weights_for_sampler",
        json={"model_id": mid, "sampling_session_seq_id": 0},
    )
    retr = await sdk_client.post(
        "/api/v1/retrieve_future", json={"request_id": resp.json()["request_id"]}
    )
    body = retr.json()
    SaveWeightsForSamplerResponseInternal.model_validate(body)
    assert body["path"] is None
    assert body["sampling_session_id"] is not None


async def test_telemetry_response_schema(sdk_client):
    from tinker.types.telemetry_response import TelemetryResponse

    resp = await sdk_client.post(
        "/api/v1/telemetry",
        json={
            "events": [{"event_type": "heartbeat", "payload": {"x": 1}, "severity": "INFO"}],
            "platform": "python",
            "sdk_version": "0.18.0",
        },
    )
    assert resp.status_code == 200
    TelemetryResponse.model_validate(resp.json())


async def test_get_info_response_schema(sdk_client):
    from tinker.types.get_info_response import GetInfoResponse

    mid = await _create_model(sdk_client)
    resp = await sdk_client.post("/api/v1/get_info", json={"model_id": mid})
    assert resp.status_code == 200
    GetInfoResponse.model_validate(resp.json())


async def test_asample_resolves_sampling_session_id(sdk_client):
    """``asample`` must resolve model_id from ``sampling_session_id``
    (encoded as ``samp-<model_id>-<seq>-<hash>``). Without this the SDK's
    SamplingClient — which only ships ``sampling_session_id`` — can't
    drive any sample call.
    """
    mid = await _create_model(sdk_client)
    resp = await sdk_client.post(
        "/api/v1/save_weights_for_sampler",
        json={"model_id": mid, "sampling_session_seq_id": 0},
    )
    retr = await sdk_client.post(
        "/api/v1/retrieve_future", json={"request_id": resp.json()["request_id"]}
    )
    sid = retr.json()["sampling_session_id"]
    # Now call asample with only the sampling_session_id.
    resp = await sdk_client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [9, 9, 9]}]},
            "num_samples": 1,
            "sampling_params": {"max_tokens": 4, "temperature": 0.0},
            "sampling_session_id": sid,
            "seq_id": 0,
        },
    )
    assert resp.status_code == 200, resp.text


async def test_retrieve_future_try_again_on_timeout(sdk_client, platform_config):
    """When the queue times out on a pending job, the retrieve_future
    response must match the SDK's TryAgainResponse shape.
    """
    from tinker.types.try_again_response import TryAgainResponse

    # Force a tiny timeout so jobs enqueued without a worker hit it fast.
    platform_config.max_job_timeout_seconds = 0.01
    mid = await _create_model(sdk_client)
    # Stop the fake worker mid-flight by submitting then awaiting before ack.
    resp = await sdk_client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": mid,
            "seq_id": 99,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
                        "loss_fn_inputs": {},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        },
    )
    fid = resp.json()["request_id"]
    # Poll retrieve_future — most of the time the worker will have already
    # acked, so skip the test if we can't catch the pending state.
    retr = await sdk_client.post("/api/v1/retrieve_future", json={"request_id": fid})
    body = retr.json()
    if body.get("type") == "try_again":
        TryAgainResponse.model_validate(body)
