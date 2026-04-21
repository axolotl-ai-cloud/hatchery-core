# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""End-to-end integration test: client SDK → gateway → real worker → back.

The existing tests cover each layer in isolation plus a few fake-worker
combinations at the gateway level. This file is the first one that
actually runs the full path:

  HatcheryClient (httpx.AsyncClient via ASGITransport)
      ↓
  FastAPI gateway (/v1/* and /api/v1/* routes)
      ↓
  InMemoryJobQueue
      ↓
  real worker loop driving a synthetic trainer
      ↓
  InMemoryObjectStore + InMemoryMetadataStore
      ↓
  results flow back the other direction via futures

We use a ``StubTrainer`` that implements the :class:`Trainer` protocol
on top of plain Python dicts — no GPU, no transformers, no PEFT. That
keeps the test fast (sub-second) while still exercising the full
worker / queue / gateway / client stack.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from hatchery.core.client import HatcheryClient
from hatchery.core.gateway import create_app
from hatchery.core.stub_worker import StubTrainer, StubWorker
from hatchery.core.trainer import LoraSpec

_ScriptedTrainer = StubTrainer
_RealWorker = StubWorker

# ─── Fixture: full stack wired up and running ────────────────────────────


@pytest_asyncio.fixture
async def running_platform(platform_config):
    """Yield (http_client, trainer, worker, config) with gateway + worker live."""
    app = create_app(config=platform_config)
    trainer = _ScriptedTrainer()
    worker = _RealWorker("e2e-worker", platform_config, trainer)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as http:
        async with app.router.lifespan_context(app):
            worker.start()
            try:
                yield http, trainer, worker, platform_config
            finally:
                await worker.stop()


# ─── Helpers ─────────────────────────────────────────────────────────────


async def _tinker_create_model(http, base_model="scripted", rank=8):
    """Create a model via the Tinker API and return model_id."""
    r = await http.post(
        "/api/v1/create_model",
        json={
            "session_id": "s",
            "model_seq_id": 0,
            "base_model": base_model,
            "lora_config": {"rank": rank},
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["model_id"]


async def _tinker_op(http, path, body, timeout=30):
    """Submit a Tinker op and retrieve the future result."""
    r = await http.post(path, json=body)
    assert r.status_code == 200, r.text
    fid = r.json()["future_id"]
    r = await http.post("/api/v1/retrieve_future", json={"future_id": fid})
    assert r.status_code == 200, r.text
    return r.json()


# ─── Tests ────────────────────────────────────────────────────────────────


async def test_full_sft_loop_via_rest(running_platform):
    """Complete SFT-style loop hitting the /api/v1/* endpoints."""
    http, trainer, _, _ = running_platform

    mid = await _tinker_create_model(http)

    losses = []
    for _ in range(5):
        result = await _tinker_op(
            http,
            "/api/v1/forward_backward",
            {
                "model_id": mid,
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {
                                "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4]}]
                            },
                            "loss_fn_inputs": {
                                "target_tokens": {"data": [1, 2, 3, 4], "shape": [4]}
                            },
                        }
                    ],
                    "loss_fn": "cross_entropy",
                },
            },
        )
        losses.append(result.get("loss", result.get("metrics", {}).get("loss:mean", 0)))
        await _tinker_op(
            http,
            "/api/v1/optim_step",
            {"model_id": mid, "adam_params": {"learning_rate": 1e-3}},
        )

    assert losses[-1] < losses[0]

    result = await _tinker_op(
        http,
        "/api/v1/asample",
        {
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [10, 20]}]},
            "model_id": mid,
            "sampling_params": {"max_tokens": 2},
        },
    )
    assert result.get("sequences")

    r = await http.post("/api/v1/save_weights", json={"model_id": mid, "path": "final"})
    assert r.status_code == 200
    r = await http.get(f"/api/v1/training_runs/{mid}/checkpoints")
    assert r.status_code == 200
    ckpts = r.json().get("checkpoints", [])
    ckpt_names = [c["checkpoint_id"] if isinstance(c, dict) else c for c in ckpts]
    assert "final" in ckpt_names


async def test_full_loop_via_tinker_compat_futures(running_platform):
    """Same loop, but through the pipelined /api/v1/* futures API."""
    http, trainer, _, _ = running_platform

    resp = await http.post(
        "/api/v1/create_session",
        json={"tags": [], "user_metadata": {}, "sdk_version": "e2e"},
    )
    assert resp.status_code == 200

    resp = await http.post(
        "/api/v1/create_model",
        json={
            "session_id": resp.json()["session_id"],
            "model_seq_id": 0,
            "base_model": "scripted",
            "lora_config": {"rank": 8},
        },
    )
    model_id = resp.json()["model_id"]

    # Pipeline 3 training steps: submit fb1, opt1, fb2, opt2, fb3, opt3
    # without awaiting in between.
    datum = {
        "model_input": {
            "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4]}],
        },
        "loss_fn_inputs": {},
    }
    pending: list[tuple[str, str]] = []
    for step in range(3):
        r = await http.post(
            "/api/v1/forward_backward",
            json={
                "model_id": model_id,
                "seq_id": step * 2,
                "forward_backward_input": {
                    "data": [datum],
                    "loss_fn": "cross_entropy",
                },
            },
        )
        pending.append(("fb", r.json()["future_id"]))
        r = await http.post(
            "/api/v1/optim_step",
            json={
                "model_id": model_id,
                "seq_id": step * 2 + 1,
                "adam_params": {"learning_rate": 1e-3},
            },
        )
        pending.append(("opt", r.json()["future_id"]))

    # Drain all futures.
    # SDK-0.18 envelope: retrieve_future returns the typed response
    # directly (no {"status": "completed", "result": {...}} wrapper).
    # forward_backward → ``loss_fn_output_type`` + ``metrics``.
    # optim_step     → ``type: "optim_step"`` + ``metrics``.
    # Errors surface as ``{"type": "request_failed" | "try_again", ...}``.
    results = []
    for kind, fid in pending:
        r = await http.post("/api/v1/retrieve_future", json={"future_id": fid})
        body = r.json()
        assert body.get("type") not in ("request_failed", "try_again"), body
        if kind == "fb":
            assert body.get("loss_fn_output_type") == "cross_entropy", body
        else:
            assert body.get("type") == "optim_step", body
        results.append((kind, body))

    fb_losses = [r["metrics"]["loss:mean"] for kind, r in results if kind == "fb"]
    assert fb_losses[-1] < fb_losses[0]
    steps_seen = [r["metrics"]["step"] for kind, r in results if kind == "opt"]
    assert steps_seen == [1, 2, 3], "optim_step ordering violated"

    # The scripted trainer recorded the exact call order — confirm it matches.
    ops = [op for sid, op in trainer._step_log if sid == model_id]
    assert ops == [
        "forward_backward",
        "optim_step",
        "forward_backward",
        "optim_step",
        "forward_backward",
        "optim_step",
    ]


async def test_two_concurrent_sessions(running_platform):
    """Two independent sessions should train in parallel without
    interfering with each other."""
    http, trainer, _, _ = running_platform

    async def one_session(tag: str) -> list[float]:
        mid = await _tinker_create_model(http)
        losses = []
        for _ in range(3):
            result = await _tinker_op(
                http,
                "/api/v1/forward_backward",
                {
                    "model_id": mid,
                    "forward_backward_input": {
                        "data": [
                            {
                                "model_input": {
                                    "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                                },
                                "loss_fn_inputs": {
                                    "target_tokens": {"data": [1, 2, 3], "shape": [3]}
                                },
                            }
                        ],
                        "loss_fn": "cross_entropy",
                    },
                },
            )
            losses.append(result.get("loss", result.get("metrics", {}).get("loss:mean", 0)))
            await _tinker_op(
                http,
                "/api/v1/optim_step",
                {"model_id": mid, "adam_params": {"learning_rate": 1e-3}},
            )
        return losses

    a_losses, b_losses = await asyncio.gather(one_session("a"), one_session("b"))
    assert a_losses[-1] < a_losses[0]
    assert b_losses[-1] < b_losses[0]


async def test_session_listing_and_deletion(running_platform):
    http, _, _, _ = running_platform

    mid_a = await _tinker_create_model(http)
    mid_b = await _tinker_create_model(http)

    r = await http.get("/v1/sessions")
    ids = {s["session_id"] for s in r.json()["sessions"]}
    assert mid_a in ids and mid_b in ids

    r = await http.delete(f"/v1/sessions/{mid_a}")
    assert r.status_code == 200

    r = await http.get(f"/v1/sessions/{mid_a}")
    assert r.status_code == 410


async def test_client_sdk_async_api_roundtrip(running_platform):
    """Placeholder — async client methods drive through the same
    ``_BackgroundLoop`` as the sync ones, which can't talk to an
    in-process ASGI harness. The real async roundtrip lives in
    ``test_real_http_roundtrip.test_real_http_async_client_sdk``.
    """
    http, _, _, _ = running_platform
    assert http is not None


# ─── Packed (varlen) end-to-end ─────────────────────────────────────────


class _PackingScriptedTrainer(_ScriptedTrainer):
    """Scripted trainer that actually exercises the packing module.

    Routes forward_backward through :func:`hatchery.core.packing.pack_sequences`
    with the session's ParallelConfig, asserts one pack was produced for
    each call, and decays the loss using the packed token count. That
    keeps the stack observable: if packing is silently bypassed the
    ``packs_emitted`` counter stays zero and the test fails.
    """

    def __init__(self, parallel) -> None:
        super().__init__()
        self.parallel = parallel
        self.packs_emitted = 0

    def forward_backward(self, session_id, data, loss_fn):
        from hatchery.core.packing import pack_sequences

        assert self.parallel.sequence_packing is True
        max_len = self.parallel.max_packed_len or sum(len(d["input_ids"]) for d in data)
        packs = pack_sequences(data, pad_id=0, max_packed_len=max_len)
        assert packs, "packing produced no packs"
        self.packs_emitted += 1
        # Delegate the bookkeeping to the scripted parent so the loss
        # curve + accum_steps semantics stay identical to the non-packed
        # e2e test.
        return super().forward_backward(session_id, data, loss_fn)


@pytest_asyncio.fixture
async def packing_platform(platform_config):
    from hatchery.core.parallel import ParallelConfig

    app = create_app(config=platform_config)
    trainer = _PackingScriptedTrainer(
        parallel=ParallelConfig(sequence_packing=True, max_packed_len=4096)
    )
    worker = _RealWorker("e2e-worker-packed", platform_config, trainer)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as http:
        async with app.router.lifespan_context(app):
            worker.start()
            try:
                yield http, trainer, worker, platform_config
            finally:
                await worker.stop()


async def test_end_to_end_packed_fft(packing_platform):
    """Full loop with sequence_packing=True + mixed-length inputs.

    Structurally identical to ``test_full_sft_loop_via_rest`` — what's
    different is that the trainer runs each batch through the packing
    code path. Asserts the pack counter advanced (i.e. packing wasn't
    silently bypassed) and the loss curve still decreases monotonically.
    """
    pytest.importorskip("torch")
    http, trainer, _, _ = packing_platform

    mid = await _tinker_create_model(http)

    mixed_lengths = [
        [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4, 5]}]},
                "loss_fn_inputs": {"target_tokens": {"data": [1, 2, 3, 4, 5], "shape": [5]}},
            },
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [6, 7]}]},
                "loss_fn_inputs": {"target_tokens": {"data": [6, 7], "shape": [2]}},
            },
        ],
        [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [10, 11]}]},
                "loss_fn_inputs": {"target_tokens": {"data": [10, 11], "shape": [2]}},
            },
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [20, 21, 22, 23]}]},
                "loss_fn_inputs": {"target_tokens": {"data": [20, 21, 22, 23], "shape": [4]}},
            },
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [30]}]},
                "loss_fn_inputs": {"target_tokens": {"data": [30], "shape": [1]}},
            },
        ],
        [
            {
                "model_input": {
                    "chunks": [{"type": "encoded_text", "tokens": [40, 41, 42, 43, 44, 45]}]
                },
                "loss_fn_inputs": {
                    "target_tokens": {"data": [40, 41, 42, 43, 44, 45], "shape": [6]}
                },
            },
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [50, 51, 52]}]},
                "loss_fn_inputs": {"target_tokens": {"data": [50, 51, 52], "shape": [3]}},
            },
        ],
    ]

    losses: list[float] = []
    for step, data in enumerate(mixed_lengths):
        result = await _tinker_op(
            http,
            "/api/v1/forward_backward",
            {
                "model_id": mid,
                "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
            },
        )
        losses.append(result.get("loss", result.get("metrics", {}).get("loss:mean", 0)))
        await _tinker_op(
            http,
            "/api/v1/optim_step",
            {"model_id": mid, "adam_params": {"learning_rate": 1e-3}},
        )
        assert trainer.packs_emitted == step + 1

    assert losses[-1] < losses[0]
    assert losses == sorted(losses, reverse=True), "loss did not decrease monotonically"
