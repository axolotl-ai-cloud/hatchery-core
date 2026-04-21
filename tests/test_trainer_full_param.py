# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for full-parameter training in :class:`VanillaTrainer`.

Uses a tiny CPU GPT-2 (vocab 64, 2 layers, n_embd 32) so the suite
runs in the torch-CPU lane without a GPU. Covers:

* Full-param session learns — loss decreases over a few fb+optim cycles.
* Mixed mode: LoRA + full-param sessions coexist on the same trainer
  without trampling each other's base weights.
* State round-trip: extract → detach → re-attach + load_state restores
  trained weights.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
pytest.importorskip("peft")
pytest.importorskip("transformers")


def _tiny_trainer():
    """Build a VanillaTrainer wrapped around a tiny CPU GPT-2.

    Bypasses HF download — installs a hand-built GPT2 + tokenizer
    directly on the trainer instance so the test runs offline.
    """
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    from hatchery.core.trainer import VanillaTrainer

    cfg = GPT2Config(
        vocab_size=64,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    trainer = VanillaTrainer(
        base_model_name="tiny-gpt2",
        device="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
        load_model=False,
    )
    trainer._raw_base = model
    # Snapshot pristine base — normally _load_base does this, but
    # we bypassed it to avoid the HF download.
    trainer._pristine_base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # Minimal tokenizer: GPT2TokenizerFast needs files; for the test
    # we never call decode, so a stub is enough.
    trainer.tokenizer = type(
        "Stub",
        (),
        {"pad_token_id": 0, "eos_token_id": 1, "pad_token": "<pad>"},
    )()
    return trainer


def _example(num_tokens: int = 8) -> dict:
    ids = list(range(2, 2 + num_tokens))
    return {"input_ids": ids, "labels": ids}


def test_full_param_attach_and_train():
    """attach + fb + optim cycle on full-param session moves loss down."""
    from hatchery.core.trainer import LoraSpec

    trainer = _tiny_trainer()
    spec = LoraSpec.full_param()
    state = trainer.init_session_state("fp1", spec)
    assert state.meta["training_mode"] == "full_param"
    assert state.lora_weights == {}  # init snapshot is empty

    losses = []
    for _ in range(8):
        result = trainer.forward_backward("fp1", [_example()], "cross_entropy")
        losses.append(result.loss)
        trainer.optim_step("fp1", {"learning_rate": 1e-2})

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_full_param_extract_load_roundtrip():
    """extract_state on a trained FP session, detach, re-attach,
    load_state — the trained weights survive the round trip."""
    from hatchery.core.trainer import LoraSpec

    trainer = _tiny_trainer()
    trainer.attach_session("fp1", LoraSpec.full_param())
    for _ in range(4):
        trainer.forward_backward("fp1", [_example()], "cross_entropy")
        trainer.optim_step("fp1", {"learning_rate": 5e-2})

    # Snapshot the live base after training, then round-trip.
    snapshot = trainer.extract_state("fp1")
    assert snapshot.meta["training_mode"] == "full_param"
    assert snapshot.lora_weights, "extract_state should populate weights"
    trained_sample = next(iter(snapshot.lora_weights.values())).clone()

    trainer.detach_session("fp1")
    assert not trainer.has_session("fp1")
    # After detach the base should be pristine again.
    pristine_sample = trainer._pristine_base_sd[next(iter(snapshot.lora_weights))]
    # Trained ≠ pristine (sanity — training did move weights).
    assert not torch.allclose(trained_sample, pristine_sample)

    trainer.load_state("fp1", snapshot)
    # _activate_session is lazy — trigger it via a forward op.
    trainer.forward_backward("fp1", [_example()], "cross_entropy")
    restored = trainer.extract_state("fp1")
    restored_sample = restored.lora_weights[next(iter(snapshot.lora_weights))]
    # Allow tiny drift from one fb step.
    assert torch.allclose(restored_sample, trained_sample, atol=1e-2)


def test_mixed_lora_and_full_param_coexist():
    """LoRA session and full-param session on the same trainer.

    Activating the LoRA session must restore the pristine base;
    activating the FP session must restore its trained base. Each
    session sees its own state across many switches.
    """
    from hatchery.core.trainer import LoraSpec

    trainer = _tiny_trainer()

    # Attach LoRA first so PEFT wraps the base — this exercises the
    # tricky case where _raw_base has .base_layer keys.
    lora_spec = LoraSpec(rank=4, lora_alpha=8, target_modules=["c_attn"])
    trainer.attach_session("lora1", lora_spec)
    fp_spec = LoraSpec.full_param()
    trainer.attach_session("fp1", fp_spec)

    # Train FP session for several steps.
    fp_losses = []
    for _ in range(5):
        r = trainer.forward_backward("fp1", [_example()], "cross_entropy")
        fp_losses.append(r.loss)
        trainer.optim_step("fp1", {"learning_rate": 1e-2})
    assert fp_losses[-1] < fp_losses[0], f"FP loss didn't decrease: {fp_losses}"

    # Switch to LoRA session: the base loaded into _raw_base must be
    # pristine, NOT the trained FP base. We verify by computing the
    # forward-only loss with adapters disabled — should match what a
    # fresh-from-load model produces.
    fresh_trainer = _tiny_trainer()
    fresh_trainer.attach_session("probe", lora_spec)
    fresh_loss = fresh_trainer.forward_only("probe", [_example()], "cross_entropy").loss

    # In mixed-mode trainer, switch to LoRA and run forward-only with
    # the lora-zero (initial init means lora_B=0 so adapter is identity).
    trainer.forward_backward("lora1", [_example()], "cross_entropy")  # activates
    lora_loss = trainer.forward_only("lora1", [_example()], "cross_entropy").loss
    # PEFT initializes lora_B to zero, so the adapter starts as identity;
    # forward should match the pristine base. After one fb step the
    # adapter has moved very slightly. Allow loose tolerance.
    assert abs(lora_loss - fresh_loss) < 0.5, (
        f"lora1 base looks contaminated: {lora_loss=} {fresh_loss=}"
    )

    # Switch back to FP session — its trained weights must be restored,
    # not pristine. Loss should be lower than pristine (we've trained).
    fp_loss_after = trainer.forward_only("fp1", [_example()], "cross_entropy").loss
    assert fp_loss_after < fp_losses[0], (
        f"fp1 lost its trained weights after the lora1 detour: {fp_loss_after=} {fp_losses[0]=}"
    )


def test_attach_lora_after_fp_does_not_corrupt_lora_base():
    """Regression: attaching LoRA after an FP session has trained must
    rewrap on the pristine base, not the trained one."""
    from hatchery.core.trainer import LoraSpec

    trainer = _tiny_trainer()
    trainer.attach_session("fp1", LoraSpec.full_param())
    for _ in range(3):
        trainer.forward_backward("fp1", [_example()], "cross_entropy")
        trainer.optim_step("fp1", {"learning_rate": 5e-2})

    # Now wrap with LoRA. The PEFT base_layer must be pristine.
    lora_spec = LoraSpec(rank=4, lora_alpha=8, target_modules=["c_attn"])
    trainer.attach_session("lora1", lora_spec)

    # Confirm a base_layer weight matches pristine.
    live = trainer._raw_base.state_dict()
    sample_pristine_key = next(iter(trainer._pristine_base_sd))
    parts = sample_pristine_key.rpartition(".")
    wrapped_key = f"{parts[0]}.base_layer.{parts[2]}" if parts[0] else sample_pristine_key
    if wrapped_key in live:
        live_w = live[wrapped_key]
    else:
        live_w = live[sample_pristine_key]
    assert torch.allclose(live_w.cpu(), trainer._pristine_base_sd[sample_pristine_key])
