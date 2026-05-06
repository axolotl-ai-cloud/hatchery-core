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

_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    """VanillaTrainer + PEFT + FSDP2 on a tiny local causal LM."""
    out = tmp_path / "trainer.json"
    result = _run_torchrun(
        _DIST_DIR / "trainer_fsdp2_smoke.py",
        out,
        ["--steps", "4"],
    )
    assert result["status"] == "ok", result.get("error")
    assert result["world_size"] == 2
    assert result["losses_decreased"], result
