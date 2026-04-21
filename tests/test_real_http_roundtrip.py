# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""End-to-end over a real HTTP socket.

``test_end_to_end.py`` uses httpx's ASGI transport so the gateway and
worker share an asyncio loop. That catches most integration bugs, but
it doesn't prove that the process-level story works: the sync client
in a background thread, a uvicorn server on a real port, requests
going over localhost, the worker as a separate asyncio task.

This file boots a real uvicorn server in a daemon thread, points a
vanilla ``httpx.Client`` at it, and runs a small SFT-style loop.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import closing

import httpx
import pytest
import uvicorn

from hatchery.core.gateway import create_app
from hatchery.core.stub_worker import StubTrainer as _ScriptedTrainer
from hatchery.core.stub_worker import StubWorker as _RealWorker


def _pick_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread(threading.Thread):
    """Run uvicorn in a daemon thread on its own asyncio loop."""

    def __init__(self, app, host: str, port: int, trainer, config):
        super().__init__(daemon=True, name="uvicorn-e2e")
        self.app = app
        self.host = host
        self.port = port
        self.trainer = trainer
        self.config = config
        self.server: uvicorn.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _boot() -> None:
            # Run the gateway's lifespan manually so the worker launches
            # in the SAME event loop as the HTTP handlers. If we let
            # uvicorn handle lifespan, the worker task would live on a
            # different loop and share asyncio primitives across threads.
            await self.config.metadata.initialize()
            await self.config.queue.initialize()
            worker = _RealWorker("http-e2e-worker", self.config, self.trainer)
            worker.start()
            cfg = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                lifespan="off",
            )
            self.server = uvicorn.Server(cfg)
            self._ready.set()
            try:
                await self.server.serve()
            finally:
                await worker.stop()
                await self.config.queue.close()
                await self.config.metadata.close()

        loop.run_until_complete(_boot())

    def wait_ready(self, timeout: float = 10.0) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError("uvicorn did not come up in time")
        # Also poll /v1/health until it answers.
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(f"http://{self.host}:{self.port}/v1/health", timeout=1)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.05)
        raise TimeoutError("health check never returned 200")

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        self.join(timeout=5)


@pytest.fixture
def http_server(platform_config):
    trainer = _ScriptedTrainer()
    app = create_app(config=platform_config)
    port = _pick_free_port()
    thread = _ServerThread(app, "127.0.0.1", port, trainer, platform_config)
    thread.start()
    try:
        thread.wait_ready()
        yield f"http://127.0.0.1:{port}", trainer
    finally:
        thread.stop()


def test_real_http_training_loop(http_server):
    base_url, trainer = http_server
    with httpx.Client(
        base_url=base_url,
        headers={"Authorization": "Bearer test-token"},
        timeout=30,
    ) as client:

        def _do_op(path, body):
            r = client.post(path, json=body)
            assert r.status_code == 200, r.text
            fid = r.json()["future_id"]
            r = client.post("/api/v1/retrieve_future", json={"future_id": fid})
            assert r.status_code == 200, r.text
            return r.json()

        r = client.post(
            "/api/v1/create_model",
            json={
                "session_id": "s",
                "model_seq_id": 0,
                "base_model": "scripted",
                "lora_config": {"rank": 8},
            },
        )
        assert r.status_code == 200, r.text
        mid = r.json()["model_id"]

        datum = {
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4]}]},
            "loss_fn_inputs": {"target_tokens": {"data": [1, 2, 3, 4], "shape": [4]}},
        }
        losses = []
        for _ in range(5):
            result = _do_op(
                "/api/v1/forward_backward",
                {
                    "model_id": mid,
                    "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"},
                },
            )
            losses.append(result.get("loss", result.get("metrics", {}).get("loss:mean", 0)))
            _do_op(
                "/api/v1/optim_step",
                {"model_id": mid, "adam_params": {"learning_rate": 1e-3}},
            )

        assert losses[-1] < losses[0]

        r = client.post("/api/v1/save_weights", json={"model_id": mid, "path": "final"})
        assert r.status_code == 200


_SDK_DATUM = {
    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
    "loss_fn_inputs": {},
}


def test_sync_client_sdk_preserves_submission_order(http_server):
    """Regression test for a pipelining bug in the sync client facade.

    Submits 3 fb+opt pairs without awaiting and verifies ordering via
    the SDK-shaped response: every fb's ``metrics["accum_steps:mean"]``
    should be ``1.0`` (the optim_step between consecutive fb's reset
    the counter), and optim_step numbers are strictly increasing.
    """
    from hatchery.core.client import HatcheryClient

    base_url, _ = http_server
    client = HatcheryClient(base_url=base_url, token="test-token", timeout=30)
    try:
        tc = client.create_lora_training_client("scripted", rank=8)
        pending = []
        for _ in range(3):
            pending.append(("fb", tc.forward_backward([_SDK_DATUM])))
            pending.append(("opt", tc.optim_step(learning_rate=1e-3)))

        results: list[tuple[str, dict]] = []
        for kind, fut in pending:
            results.append((kind, fut.result(timeout=30)))

        fb_results = [r for kind, r in results if kind == "fb"]
        opt_results = [r for kind, r in results if kind == "opt"]

        for r in fb_results:
            assert r["metrics"]["accum_steps:mean"] == 1.0, (
                f"ordering violated: expected accum_steps=1, got {r}"
            )
        assert [r["metrics"]["step"] for r in opt_results] == [1.0, 2.0, 3.0]
    finally:
        client.close()


def test_real_http_sync_client_sdk(http_server):
    """Drive the platform through the sync ``HatcheryClient`` facade —
    background-thread event loop included."""
    from hatchery.core.client import HatcheryClient

    base_url, _ = http_server
    client = HatcheryClient(base_url=base_url, token="test-token", timeout=30)
    try:
        tc = client.create_lora_training_client("scripted", rank=8)
        assert tc.session_id

        for _ in range(3):
            fb = tc.forward_backward([_SDK_DATUM])
            assert "loss:mean" in fb.result(timeout=30)["metrics"]
            opt = tc.optim_step(learning_rate=1e-3)
            assert opt.result(timeout=30)["metrics"]["step"] >= 1

        ckpt = tc.save_weights("v1").result(timeout=10)
        assert "path" in ckpt
        assert "v1" in tc.list_checkpoints()
    finally:
        client.close()


async def test_real_http_async_client_sdk(http_server):
    """Drive the client's ``_async`` methods: submit via the background
    loop's HTTP client, await the returned ``_HatcheryFuture`` from the
    caller's event loop via ``asyncio.wrap_future``."""
    from hatchery.core.client import HatcheryClient

    base_url, _ = http_server
    client = HatcheryClient(base_url=base_url, token="test-token", timeout=30)
    try:
        tc = await client.create_lora_training_client_async(base_model="scripted", rank=8)
        assert tc.session_id

        fb = await tc.forward_backward_async([_SDK_DATUM])
        fb_result = await fb.result_async(timeout=30)
        assert "loss:mean" in fb_result["metrics"]

        opt = await tc.optim_step_async(learning_rate=1e-3)
        opt_result = await opt.result_async(timeout=30)
        assert opt_result["metrics"]["step"] == 1.0

        samp = await tc.sample_async([10, 20], max_tokens=2)
        samp_result = await samp.result_async(timeout=30)
        assert "sequences" in samp_result

        sw = await tc.save_weights_async("v1")
        await sw.result_async(timeout=10)
        cps = await tc.list_checkpoints_async()
        assert "v1" in cps
    finally:
        await client.aclose()
