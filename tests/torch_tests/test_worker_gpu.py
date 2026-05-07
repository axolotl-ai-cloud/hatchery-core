# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""GPU integration test with a real Qwen2-0.5B base model.

Runs end-to-end through the worker with a real LoRA, doing a handful
of forward/backward + optim_step cycles on a synthetic overfit batch
and confirming that loss decreases.

Skipped automatically if CUDA isn't available, and also skipped if the
user has not pre-loaded (or cannot download) Qwen2-0.5B.

Parametrized on dtype: fp16 runs everywhere, bf16 only on Ampere+.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gpu]

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

try:
    torch.cuda.current_device()
except RuntimeError as _e:
    pytest.skip(f"CUDA init failed (driver too old?): {_e}", allow_module_level=True)

pytest.importorskip("peft")
pytest.importorskip("transformers")


MODEL_NAME = os.environ.get("TEST_BASE_MODEL", "Qwen/Qwen2-0.5B")
TEST_DEVICE = os.environ.get("TEST_DEVICE", "cuda:0")

_DTYPES = [torch.float16]
if torch.cuda.is_bf16_supported():
    _DTYPES.append(torch.bfloat16)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: d.__repr__().split(".")[-1])
async def test_full_lifecycle_with_real_model(platform_config, tmp_path, dtype):
    from hatchery.core.worker import GPUWorker

    worker = GPUWorker(
        worker_id="gpu-test",
        base_model_name=MODEL_NAME,
        config=platform_config,
        device=TEST_DEVICE,
        dtype=dtype,
        attn_implementation="sdpa",
    )

    sid = "gpu-sess-1"
    await worker._handle_init_session(
        sid,
        {
            "rank": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
    )

    tokenizer = worker.tokenizer
    text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    data = [{"input_ids": ids}]

    losses = []
    for _step in range(5):
        result, _ = await worker._handle_forward_backward(
            sid, {"data": data, "loss_fn": "cross_entropy"}
        )
        losses.append(result["loss"])
        await worker._handle_optim_step(sid, {"learning_rate": 1e-3})

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    # Sampling
    prompt_ids = tokenizer("Hello", return_tensors=None)["input_ids"]
    sample_result, _ = await worker._handle_sample(
        sid,
        {
            "prompt_tokens": prompt_ids,
            "max_tokens": 8,
            "temperature": 0.0,
            "n": 1,
        },
    )
    assert len(sample_result["sequences"]) == 1
    assert len(sample_result["sequences"][0]) > 0

    # Read live weights from worker's local session store.
    key = f"sessions/{sid}/live_state/lora_weights.pt"
    lora_bytes = await worker._state.load_local(key)
    assert lora_bytes is not None, "live_state not on local disk"
    dst = f"sessions/{sid}/checkpoints/step5/lora_weights.pt"
    await platform_config.objects.put(dst, lora_bytes)
    assert await platform_config.objects.exists(dst)
