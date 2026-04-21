# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Regression test: client-controlled LR must be honored per optim_step.

torch's optimizer.load_state_dict() overwrites the LR with the saved
value. Without the post-load fix, every step after the first would
use the LR from step 1, breaking client-side LR scheduling
(warmup, cosine decay, RL-adaptive LR, etc.).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_load_state_dict_does_not_override_lr():
    """Prove the raw PyTorch bug exists — this is what we're fixing."""
    p = torch.nn.Parameter(torch.randn(4))
    opt1 = torch.optim.AdamW([p], lr=1e-3, fused=False)
    p.grad = torch.randn(4)
    opt1.step()
    state = opt1.state_dict()

    opt2 = torch.optim.AdamW([p], lr=5e-4, fused=False)
    opt2.load_state_dict(state)
    # Without fix: LR is 1e-3 (from state), not 5e-4 (from constructor).
    assert opt2.param_groups[0]["lr"] == pytest.approx(1e-3)

    # After manually setting it back:
    opt2.param_groups[0]["lr"] = 5e-4
    assert opt2.param_groups[0]["lr"] == pytest.approx(5e-4)


async def test_trainer_honors_per_step_lr(platform_config):
    """VanillaTrainer.optim_step must use the requested LR, not the
    saved LR from a previous step."""
    from hatchery.core.trainer import LoraSpec, VanillaTrainer

    trainer = VanillaTrainer(
        base_model_name="hf-internal-testing/tiny-random-GPT2LMHeadModel",
        device="cpu",
        dtype=torch.float32,
    )
    spec = LoraSpec(rank=4, lora_alpha=4, target_modules=["c_attn"])
    trainer.attach_session("s1", spec)
    state = trainer.init_session_state("s1", spec)
    trainer.load_state("s1", state)

    # Step 1: forward_backward + optim_step at lr=1e-3
    data = [{"input_ids": [1, 2, 3, 4, 5]}]
    trainer.forward_backward("s1", data, "cross_entropy")
    result1 = trainer.optim_step("s1", {"learning_rate": 1e-3})
    assert result1.learning_rate == pytest.approx(1e-3)

    # Step 2: forward_backward + optim_step at lr=5e-5 (10x lower)
    trainer.forward_backward("s1", data, "cross_entropy")
    result2 = trainer.optim_step("s1", {"learning_rate": 5e-5})
    assert result2.learning_rate == pytest.approx(5e-5)

    # The key assertion: step 2's LR was NOT overridden by step 1's saved state.
    # If the fix is missing, result2.learning_rate would be 1e-3.
    assert result2.learning_rate != pytest.approx(1e-3)
