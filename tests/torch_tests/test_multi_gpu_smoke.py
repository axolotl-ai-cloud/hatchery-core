# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Pytest wrapper that launches the multi-GPU smoke tests via torchrun.

These tests only run when at least 2 CUDA devices are visible and
torchrun is on ``$PATH`` (which it is whenever torch is installed).
Each test subprocess-invokes a standalone script in ``tests/dist/`` and
asserts on the JSON result the script writes.

Rationale for subprocess: ``torch.distributed.init_process_group`` must
run in a fresh process per rank, and doing that cleanly inside pytest's
single event loop is painful. Shelling out to torchrun matches how real
users would launch workers in production.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DIST_DIR = _REPO_ROOT / "tests" / "dist"


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _has_torchrun() -> bool:
    return shutil.which("torchrun") is not None


def _has_bf16() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_gpu_count() < 2, reason="need >= 2 CUDA devices"),
    pytest.mark.skipif(not _has_torchrun(), reason="torchrun not on PATH"),
    pytest.mark.skipif(not _has_bf16(), reason="bf16 not supported (dist scripts require it)"),
]


def _run_torchrun(script: Path, out_path: Path, extra: list[str]) -> dict:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    cmd = [
        "torchrun",
        "--nproc-per-node=2",
        "--standalone",
        str(script),
        "--out",
        str(out_path),
        *extra,
    ]
    subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=True, timeout=600)
    assert out_path.exists(), f"script {script} did not write {out_path}"
    with open(out_path) as f:
        return json.load(f)


def test_fsdp2_smoke_pure_torch(tmp_path):
    """FSDP2 on a toy model — validates parallel.py plumbing end-to-end."""
    out = tmp_path / "fsdp2.json"
    result = _run_torchrun(_DIST_DIR / "fsdp2_smoke.py", out, [])
    assert result["status"] == "ok"
    assert result["world_size"] == 2
    assert result["mesh_dims"] == ["dp"]
    assert result["last_loss"] < result["first_loss"]


def test_fsdp2_smoke_vanilla_trainer(tmp_path):
    """VanillaTrainer + PEFT + FSDP2 on real Qwen2-0.5B."""
    out = tmp_path / "trainer.json"
    result = _run_torchrun(
        _DIST_DIR / "trainer_fsdp2_smoke.py",
        out,
        ["--steps", "3"],
    )
    assert result["status"] == "ok", result.get("error")
    assert result["world_size"] == 2
    assert result["losses_decreased"], result


def test_context_parallel_smoke_pure_torch(tmp_path):
    """Context parallel on a hand-written attention block.

    Validates the ``context_parallel_region`` helper in
    :mod:`distributed parallel` and the fact that torch's ring
    attention runs end-to-end with our mesh layout.
    """
    out = tmp_path / "cp.json"
    result = _run_torchrun(_DIST_DIR / "cp_smoke.py", out, [])
    assert result["status"] == "ok", result.get("error")
    assert result["mesh_dims"] == ["cp"]
    assert result["world_size"] == 2
    assert result["loss_decreased"], result


def test_multi_rank_worker_coordinator_follower(tmp_path):
    """End-to-end multi-rank GPUWorker: rank 0 coordinates the queue,
    rank 1 follows along via ``broadcast_object_list``, both run FSDP
    forward/backward on Qwen2-0.5B. Validates the rank-0 queue-broker
    pattern that makes FSDP workers look like a single logical unit
    from the gateway's perspective.
    """
    out = tmp_path / "mr.json"
    result = _run_torchrun(
        _DIST_DIR / "multi_rank_worker_smoke.py",
        out,
        ["--steps", "2"],
    )
    assert result["status"] == "ok", result.get("error")
    assert result["world_size"] == 2
    assert result["acked"] == 2, result


def test_multi_rank_dp_split_batch(tmp_path):
    """BatchStrategy.SPLIT end-to-end on real Qwen2-0.5B with 2 ranks.

    Each rank gets exactly one item from a batch of 2. Both ranks run
    the forward/backward together under FSDP; gradients are averaged
    across ranks via the default reduce-mean. The loss must decrease
    and each rank's local num_tokens should reflect only one item.
    """
    out = tmp_path / "dp_split.json"
    result = _run_torchrun(
        _DIST_DIR / "multi_rank_dp_split_smoke.py",
        out,
        ["--steps", "3"],
    )
    assert result["status"] == "ok", result.get("error")
    assert result["world_size"] == 2
    assert result["loss_decreased"], result
    assert result["split_worked"], result


def test_multi_rank_worker_simple_direct_execute(tmp_path):
    """Simpler companion to the coordinator test: each rank calls
    ``_execute_job`` directly with identical payloads. If this ever
    fails but the full coordinator test still passes, the bug is in
    the queue / broadcast layer, not in FSDP+PEFT integration.
    """
    out = tmp_path / "simple.json"
    result = _run_torchrun(
        _DIST_DIR / "multi_rank_worker_simple.py",
        out,
        ["--steps", "2"],
    )
    assert result["status"] == "ok", result.get("error")
    assert result["decreased"], result


def test_context_parallel_smoke_vanilla_trainer(tmp_path):
    """VanillaTrainer + PEFT + CP on real Qwen2-0.5B with a 512-token input.

    This is the realistic agentic-trace case: long sequences get
    sharded across the CP group so each GPU only holds half the tokens'
    activations.
    """
    out = tmp_path / "trainer_cp.json"
    result = _run_torchrun(
        _DIST_DIR / "trainer_cp_smoke.py",
        out,
        ["--steps", "3", "--seq-len", "512"],
    )
    assert result["status"] == "ok", result.get("error")
    assert result["world_size"] == 2
    assert result["seq_len"] >= 256
    assert result["loss_decreased"], result
