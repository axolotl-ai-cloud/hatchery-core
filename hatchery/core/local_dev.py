# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Single-process local dev launcher.

Boots the gateway + worker + in-memory backends in one Python process.
Auto-detects GPU availability — if torch + CUDA are present, loads a
real model; otherwise falls back to a scripted (no-GPU) worker.

    python -m hatchery.core.local_dev

That's it. No env vars required for the default experience. Override
with env vars if needed:

  HATCHERY_DEV_PORT        HTTP port (default 8420)
  HATCHERY_DEV_API_KEY     Bearer token (default "dev")
  HATCHERY_DEV_BASE_MODEL  HF model id (default Qwen/Qwen2-0.5B-Instruct)
  HATCHERY_DEV_DEVICE      Torch device (default: auto-detected)
  HATCHERY_DEV_RANK        Default LoRA rank (default 32)
  HATCHERY_DEV_NO_GPU      Set to "1" to force scripted worker even with GPU
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import uuid

import uvicorn

from hatchery.core.backends.auth.api_key import APIKeyAuthProvider
from hatchery.core.backends.compute.local import LocalComputeBackend
from hatchery.core.backends.metadata.memory import InMemoryMetadataStore
from hatchery.core.backends.metrics.log import LogMetrics
from hatchery.core.backends.object_store.memory import InMemoryObjectStore
from hatchery.core.backends.queue.memory import InMemoryJobQueue
from hatchery.core.config import Config
from hatchery.core.gateway import create_app, set_config


def _has_gpu() -> bool:
    if os.environ.get("HATCHERY_DEV_NO_GPU", "0") == "1":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def _build_config(api_key: str) -> Config:
    auth = APIKeyAuthProvider()
    auth.add_key(
        api_key,
        user_id="dev",
        tier="enterprise",
        max_concurrent_sessions=100,
        max_rank=256,
    )
    return Config(
        auth=auth,
        metadata=InMemoryMetadataStore(),
        objects=InMemoryObjectStore(),
        queue=InMemoryJobQueue(),
        compute=LocalComputeBackend(),
        metrics=LogMetrics(),
    )


async def _serve() -> None:
    port = int(os.environ.get("HATCHERY_DEV_PORT", "8420"))
    api_key = os.environ.get("HATCHERY_DEV_API_KEY", "dev")
    base_model = os.environ.get("HATCHERY_DEV_BASE_MODEL", "Qwen/Qwen2-0.5B-Instruct")
    device = os.environ.get("HATCHERY_DEV_DEVICE", _auto_device())
    use_gpu = _has_gpu() and device != "cpu"

    config = _build_config(api_key)
    set_config(config)
    app = create_app(config=config)

    await config.metadata.initialize()
    await config.queue.initialize()

    worker_id = f"local-dev-{uuid.uuid4().hex[:6]}"
    worker_task: asyncio.Task | None = None

    if use_gpu:
        from hatchery.core.worker import GPUWorker

        worker = GPUWorker(
            worker_id=worker_id,
            base_model_name=base_model,
            config=config,
            device=device,
        )
        await worker.register()

        async def _worker_loop() -> None:
            while True:
                job = await config.queue.dequeue(
                    worker_id=worker_id,
                    model_filter=base_model,
                    visibility_timeout=300,
                )
                if job is None:
                    await asyncio.sleep(0.02)
                    continue
                await worker._process_one(job)

        worker_task = asyncio.create_task(_worker_loop())
        mode = f"GPU worker on {device} ({base_model})"
    else:
        from hatchery.core.stub_worker import StubTrainer, StubWorker

        trainer = StubTrainer()
        stub = StubWorker(worker_id, config, trainer)
        stub.start()
        worker_task = stub._task
        mode = "scripted worker (no GPU)"

    print()
    print("  Hatchery local dev server")
    print("  ─────────────────────")
    print(f"  URL:    http://127.0.0.1:{port}")
    print(f"  Token:  {api_key}")
    print(f"  Worker: {mode}")
    print()
    print("  Example:")
    print("    python hatchery/core/examples/train_sft.py \\")
    print(
        f"      --base-url http://127.0.0.1:{port} --token {api_key}"
        + (f" --base-model {base_model}" if use_gpu else " --base-model scripted")
    )
    print(flush=True)

    cfg = uvicorn.Config(
        app,
        host="0.0.0.0",  # nosec B104
        port=port,
        log_level="info",
        lifespan="off",
    )
    server = uvicorn.Server(cfg)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _on_signal() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _on_signal)

    serve_task = asyncio.create_task(server.serve())
    await stop.wait()
    server.should_exit = True
    await serve_task
    if worker_task is not None:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker_task
    await config.queue.close()
    await config.metadata.close()


def main() -> int:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_serve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
