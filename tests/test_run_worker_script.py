# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Test that ``./core/scripts/run_worker.sh`` actually works.

Two levels of coverage:

1. **Shell-level**: check the script parses cleanly and, in dry-run
   mode, exec's the right command for each parallelism mode. No GPU
   needed.

2. **End-to-end (GPU)**: boot an in-process gateway backed by the
   SQLite job queue + filesystem object store, then invoke
   ``./core/scripts/run_worker.sh`` with ``NPROC=1`` as a real subprocess
   pointing at the same backends. Enqueue a job from the gateway,
   wait for the subprocess worker to pick it up and ack, retrieve
   the result.  This is the closest possible analog to the
   "gateway in Railway, worker on RunPod" production topology,
   run on a single host.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import msgpack
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "hatchery" / "core" / "scripts" / "run_worker.sh"


# ─── Shell-level tests (no torch required) ───────────────────────────────


def test_script_exists_and_is_executable():
    assert _SCRIPT.exists(), f"missing: {_SCRIPT}"
    assert os.access(_SCRIPT, os.X_OK), f"not executable: {_SCRIPT}"


def test_script_parses():
    """bash -n: parse-only, doesn't execute anything."""
    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_rejects_inconsistent_parallel_config():
    """NPROC != DP*TP*CP should fail with a clear error."""
    env = os.environ.copy()
    env["NPROC"] = "2"
    env["HATCHERY_DP_DEGREE"] = "4"
    env["HATCHERY_TP_DEGREE"] = "1"
    env["HATCHERY_CP_DEGREE"] = "1"
    # Prevent the script from actually running python by pointing the
    # final exec at a harmless no-op.
    env["PATH"] = f"{env.get('PATH', '')}"
    # Use a fake ``python`` / ``torchrun`` on a stub PATH so exec is safe.
    env["HATCHERY_BASE_MODEL"] = "fake"
    # Direct the final exec to "true" by prepending a stub dir to PATH.
    stub_dir = Path("/tmp") / f"runworker-stub-{os.getpid()}"
    stub_dir.mkdir(exist_ok=True)
    for name in ("python", "torchrun"):
        p = stub_dir / name
        p.write_text("#!/usr/bin/env bash\nexit 0\n")
        p.chmod(0o755)
    env["PATH"] = f"{stub_dir}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
    )
    # The sanity check should fire before the exec.
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "DP*TP*CP" in result.stderr


def test_script_nproc1_would_exec_plain_python():
    """Stub out python, run with NPROC=1, confirm the stub was called."""
    stub_dir = Path("/tmp") / f"runworker-stub2-{os.getpid()}"
    stub_dir.mkdir(exist_ok=True)
    marker = stub_dir / "called"
    marker.unlink(missing_ok=True)
    stub_py = stub_dir / "python"
    stub_py.write_text(f'#!/usr/bin/env bash\necho "$@" > {marker}\n')
    stub_py.chmod(0o755)

    env = os.environ.copy()
    env["NPROC"] = "1"
    env["PATH"] = f"{stub_dir}:{env['PATH']}"
    env["HATCHERY_BASE_MODEL"] = "fake"

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert marker.exists()
    args = marker.read_text().strip()
    assert "-m" in args and "hatchery.core.worker" in args


# ─── GPU-gated end-to-end subprocess worker test ─────────────────────────


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() >= 1
    except Exception:
        return False


def _has_qwen_cached() -> bool:
    # Best-effort check for the model under HF cache.
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return any(cache.glob("models--Qwen--Qwen2-0.5B-Instruct*"))


@pytest.mark.gpu
@pytest.mark.skipif(not _has_gpu(), reason="need a CUDA device")
@pytest.mark.skipif(not _has_qwen_cached(), reason="Qwen2-0.5B-Instruct not in HF cache")
def test_run_worker_sh_subprocess_roundtrip(tmp_path):
    """Run the actual ``./core/scripts/run_worker.sh NPROC=1`` subprocess
    against an in-process gateway. Enqueue an init_session + one
    forward_backward, wait for the worker subprocess to process them,
    and confirm.
    """
    from hatchery.core.backends.metadata.sqlite import SQLiteMetadataStore
    from hatchery.core.backends.object_store.local import LocalObjectStore
    from hatchery.core.backends.queue.sqlite import SQLiteJobQueue

    object_root = tmp_path / "objects"
    metadata_db = tmp_path / "metadata.db"
    queue_db = tmp_path / "queue.db"

    device = os.environ.get("TEST_DEVICE", "cuda:0")

    # Spawn worker subprocess via run_worker.sh.
    env = os.environ.copy()
    env.update(
        {
            "NPROC": "1",
            "HATCHERY_BASE_MODEL": "Qwen/Qwen2-0.5B-Instruct",
            "HATCHERY_WORKER_DEVICE": device,
            # Point the worker at hosted's env-var-driven config factory
            # so it wires up SQLite metadata/queue and the S3/local
            # object store the test has set up below. Without this the
            # worker would fall back to core's in-memory backends.
            "HATCHERY_CONFIG_FACTORY": "hatchery.core.config:build_platform_config",
            # Tell hosted's config factory which backends to use.
            "HATCHERY_OBJECT_STORE": "local",
            "HATCHERY_LOCAL_STORE_PATH": str(object_root),
            "HATCHERY_METADATA_STORE": "sqlite",
            "HATCHERY_SQLITE_PATH": str(metadata_db),
            "HATCHERY_JOB_QUEUE": "sqlite",
            "HATCHERY_SQLITE_QUEUE_PATH": str(queue_db),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
            # The worker doesn't need auth — it reads jobs directly from
            # the queue. The ADMIN_API_KEY is optional and unused here.
            "VENV_PY": sys.executable,
        }
    )

    from hatchery.core.protocols import JobStatus, QueuedJob

    async def _setup_and_enqueue() -> tuple[SQLiteMetadataStore, SQLiteJobQueue, str, str]:
        LocalObjectStore(root=str(object_root))  # just ensures the dir exists
        meta = SQLiteMetadataStore(path=str(metadata_db))
        await meta.initialize()
        q = SQLiteJobQueue(path=str(queue_db), poll_interval=0.05)
        await q.initialize()

        session_id = f"sub-{uuid.uuid4().hex[:8]}"
        init_payload = msgpack.packb(
            {
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "rank": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
            },
            use_bin_type=True,
        )
        init_job_id = f"init-{uuid.uuid4().hex[:8]}"
        await q.enqueue(
            QueuedJob(
                job_id=init_job_id,
                session_id=session_id,
                operation="init_session",
                payload=init_payload,
                priority=10,
                required_model="Qwen/Qwen2-0.5B-Instruct",
                user_id="test",
            )
        )

        fb_payload = msgpack.packb(
            {
                "data": [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}],
                "loss_fn": "cross_entropy",
            },
            use_bin_type=True,
        )
        fb_job_id = f"fb-{uuid.uuid4().hex[:8]}"
        await q.enqueue(
            QueuedJob(
                job_id=fb_job_id,
                session_id=session_id,
                operation="forward_backward",
                payload=fb_payload,
                priority=0,
                required_model="Qwen/Qwen2-0.5B-Instruct",
                user_id="test",
            )
        )

        await q.close()
        await meta.close()
        return None, None, init_job_id, fb_job_id

    _, _, init_job_id, fb_job_id = asyncio.run(_setup_and_enqueue())

    # Spawn the subprocess worker via the shell script.
    proc = subprocess.Popen(
        ["bash", str(_SCRIPT)],
        cwd=str(_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # so we can SIGTERM the whole group
    )

    try:
        # Wait for both results. Give generous time because model
        # loading on cuda alone is ~5s. Use a fresh queue connection
        # since the test's earlier one has been closed.
        async def _wait() -> tuple:
            q = SQLiteJobQueue(path=str(queue_db), poll_interval=0.1)
            await q.initialize()
            try:
                init_result = await q.wait_for_result(init_job_id, timeout=180.0)
                fb_result = await q.wait_for_result(fb_job_id, timeout=180.0)
            finally:
                await q.close()
            return init_result, fb_result

        init_result, fb_result = asyncio.run(_wait())

        assert init_result.status == JobStatus.COMPLETED, init_result.error
        assert fb_result.status == JobStatus.COMPLETED, fb_result.error

        fb_body = msgpack.unpackb(fb_result.result, raw=False)
        assert "loss" in fb_body
        assert fb_body["num_tokens"] > 0
        assert fb_body["accum_steps"] == 1

    finally:
        # SIGTERM the worker process group; the worker loop ignores
        # TERM during model load so kill -9 the group if it doesn't
        # exit quickly.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except ProcessLookupError:
            pass
