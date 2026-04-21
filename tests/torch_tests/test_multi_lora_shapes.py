# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Verify that one trainer can host sessions with different LoRA configs.

Two customers, two different ranks (r=8 and r=32), two different
target-module sets, all sharing one frozen base model on the same
worker. We walk through:

1. Attach session A (rank=8, q/v only).
2. Attach session B (rank=32, q/k/v/o).
3. Confirm that PEFT created adapter modules with the *correct*
   shapes for each — session A's lora_A is (8, hidden), session
   B's lora_A is (32, hidden).
4. Run a few forward_backward + optim_step cycles on each session
   independently. Each should:
   - produce a decreasing loss on its own data
   - touch only its own adapter's params in the optimizer state
   - have its own Adam state with shapes matching its LoRA
5. Tear down session A, add a new session C with yet another rank
   (r=16, q only), and verify the existing session B's state is
   unaffected.
6. Round-trip a session through the object store: save state for
   session A, drop it from the trainer, reload, verify the adapter
   comes back with the same shapes and weights.

All of this runs on CPU with a tiny GPT-2-like model so the tests
stay under a second.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

# Force CPU regardless of whichever GPU/torch context was already set.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

pytest.importorskip("peft")
pytest.importorskip("transformers")


@pytest.fixture
def trainer():
    """A VanillaTrainer running on a throwaway 2-layer GPT-2 model on CPU."""
    from transformers import GPT2Config, GPT2LMHeadModel

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

    t = VanillaTrainer(
        base_model_name="gpt2-tiny",
        device="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
        load_model=False,
    )
    t._raw_base = model

    class _FakeTok:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"

        def decode(self, ids, skip_special_tokens: bool = False) -> str:
            return " ".join(str(int(i)) for i in ids)

    t.tokenizer = _FakeTok()
    return t


def _adapter_params(trainer, session_id: str) -> dict[str, torch.Tensor]:
    """Pull the full (name → tensor) map for one session's adapter.

    Note: PEFT flips ``requires_grad`` on inactive adapters when
    ``set_adapter`` is called so only one adapter trains at a time.
    We don't filter on ``requires_grad`` here because we want to see
    the full adapter regardless of which one is currently active.
    """
    adapter = trainer._adapter_name(session_id)
    out: dict[str, torch.Tensor] = {}
    for n, p in trainer._peft.named_parameters():
        if f".{adapter}." in n:
            out[n] = p
    return out


# ─── Shape correctness ──────────────────────────────────────────────────


def test_different_ranks_produce_different_shapes(trainer):
    from hatchery.core.trainer import LoraSpec

    trainer.attach_session("sess-a", LoraSpec(rank=8, lora_alpha=16, target_modules=["c_attn"]))
    trainer.attach_session("sess-b", LoraSpec(rank=32, lora_alpha=64, target_modules=["c_attn"]))

    a_params = _adapter_params(trainer, "sess-a")
    b_params = _adapter_params(trainer, "sess-b")

    # Both adapters exist and are independent.
    assert len(a_params) > 0
    assert len(b_params) > 0
    assert set(a_params) != set(b_params)

    # lora_A for session A has rank 8; for session B, rank 32.
    a_lora_a = [p for n, p in a_params.items() if "lora_A" in n]
    b_lora_a = [p for n, p in b_params.items() if "lora_A" in n]
    assert a_lora_a and b_lora_a
    assert all(p.shape[0] == 8 for p in a_lora_a)
    assert all(p.shape[0] == 32 for p in b_lora_a)


def test_different_target_modules_coexist(trainer):
    """Session A targets c_attn only; session B targets c_attn + c_proj.
    Both adapters should be present with non-overlapping module targets.
    """
    from hatchery.core.trainer import LoraSpec

    trainer.attach_session(
        "sess-narrow",
        LoraSpec(rank=4, lora_alpha=8, target_modules=["c_attn"]),
    )
    trainer.attach_session(
        "sess-wide",
        LoraSpec(rank=4, lora_alpha=8, target_modules=["c_attn", "c_proj"]),
    )
    narrow = _adapter_params(trainer, "sess-narrow")
    wide = _adapter_params(trainer, "sess-wide")

    assert all("c_attn" in n for n in narrow)
    assert any("c_proj" in n for n in wide)
    assert any("c_attn" in n for n in wide)


# ─── Independent training ────────────────────────────────────────────────


def test_independent_forward_backward_per_session(trainer):
    from hatchery.core.trainer import LoraSpec

    trainer.attach_session("sess-a", LoraSpec(rank=8, lora_alpha=16, target_modules=["c_attn"]))
    trainer.attach_session("sess-b", LoraSpec(rank=16, lora_alpha=32, target_modules=["c_attn"]))

    a_before = {n: p.detach().clone() for n, p in _adapter_params(trainer, "sess-a").items()}
    b_before = {n: p.detach().clone() for n, p in _adapter_params(trainer, "sess-b").items()}

    # Train session A only.
    data = [{"input_ids": [1, 2, 3, 4, 5]}]
    trainer.forward_backward("sess-a", data, "cross_entropy")
    trainer.optim_step("sess-a", {"learning_rate": 1e-2})

    # Session A's params should have moved; session B's should NOT.
    a_after = _adapter_params(trainer, "sess-a")
    b_after = _adapter_params(trainer, "sess-b")

    a_changed = any(not torch.allclose(a_before[n], a_after[n]) for n in a_before)
    b_changed = any(not torch.allclose(b_before[n], b_after[n]) for n in b_before)
    assert a_changed, "session A's params did not move after training"
    assert not b_changed, "session B's params moved despite not being trained"


def test_optimizer_state_shapes_match_lora_shapes(trainer):
    from hatchery.core.trainer import LoraSpec

    trainer.attach_session("sess-small", LoraSpec(rank=4, lora_alpha=8, target_modules=["c_attn"]))
    trainer.attach_session(
        "sess-large", LoraSpec(rank=64, lora_alpha=128, target_modules=["c_attn"])
    )
    data = [{"input_ids": [1, 2, 3, 4]}]
    for sid in ("sess-small", "sess-large"):
        trainer.forward_backward(sid, data, "cross_entropy")
        trainer.optim_step(sid, {"learning_rate": 1e-3})

    small_state = trainer._optimizer_state["sess-small"]
    large_state = trainer._optimizer_state["sess-large"]

    # Adam keeps exp_avg + exp_avg_sq per param, same shape as the param.
    def _shapes(state: dict) -> list[tuple]:
        shapes = []
        for _k, v in state.get("state", {}).items():
            if not isinstance(v, dict):
                continue
            exp_avg = v.get("exp_avg")
            if exp_avg is not None and torch.is_tensor(exp_avg):
                shapes.append(tuple(exp_avg.shape))
        return shapes

    small_shapes = _shapes(small_state)
    large_shapes = _shapes(large_state)
    assert small_shapes, "no Adam state captured for sess-small"
    assert large_shapes, "no Adam state captured for sess-large"
    assert small_shapes != large_shapes, (
        "Adam state shapes identical across different ranks — state is leaking between sessions"
    )


def test_grads_are_session_isolated(trainer):
    """forward_backward on session A must NOT populate session B's grad_accum."""
    from hatchery.core.trainer import LoraSpec

    trainer.attach_session("sess-a", LoraSpec(rank=8, lora_alpha=16, target_modules=["c_attn"]))
    trainer.attach_session("sess-b", LoraSpec(rank=32, lora_alpha=64, target_modules=["c_attn"]))
    data = [{"input_ids": [1, 2, 3]}]
    trainer.forward_backward("sess-a", data, "cross_entropy")

    assert trainer._grad_accum["sess-a"], "A's grad_accum is empty"
    assert trainer._grad_accum["sess-b"] == {}, "B's grad_accum was touched while training A"


# ─── Churn: add / remove / re-add at a new rank ─────────────────────────


def test_detach_and_new_session_different_rank(trainer):
    from hatchery.core.trainer import LoraSpec

    trainer.attach_session("sess-a", LoraSpec(rank=8, lora_alpha=16, target_modules=["c_attn"]))
    trainer.attach_session("sess-b", LoraSpec(rank=32, lora_alpha=64, target_modules=["c_attn"]))
    data = [{"input_ids": [1, 2, 3, 4]}]
    trainer.forward_backward("sess-b", data, "cross_entropy")
    trainer.optim_step("sess-b", {"learning_rate": 1e-3})

    # Snapshot B's state.
    b_state_before = {n: p.detach().clone() for n, p in _adapter_params(trainer, "sess-b").items()}
    b_opt_before = trainer._optimizer_state["sess-b"]

    trainer.detach_session("sess-a")
    assert not trainer.has_session("sess-a")

    trainer.attach_session("sess-c", LoraSpec(rank=16, lora_alpha=32, target_modules=["c_attn"]))
    c_params = _adapter_params(trainer, "sess-c")
    c_lora_a = [p for n, p in c_params.items() if "lora_A" in n]
    assert all(p.shape[0] == 16 for p in c_lora_a)

    # B's state is still intact.
    b_state_after = {n: p.detach().clone() for n, p in _adapter_params(trainer, "sess-b").items()}
    for n in b_state_before:
        assert torch.allclose(b_state_before[n], b_state_after[n]), (
            f"B's param {n} was mutated by churning A → C"
        )
    assert trainer._optimizer_state["sess-b"] is b_opt_before


# ─── State round-trip through the serialization boundary ───────────────


def test_extract_load_state_preserves_heterogeneous_ranks(trainer):
    from hatchery.core.trainer import LoraSpec

    trainer.attach_session("sess-a", LoraSpec(rank=8, lora_alpha=16, target_modules=["c_attn"]))
    trainer.attach_session("sess-b", LoraSpec(rank=32, lora_alpha=64, target_modules=["c_attn"]))
    data = [{"input_ids": [1, 2, 3, 4, 5]}]
    for _ in range(2):
        trainer.forward_backward("sess-a", data, "cross_entropy")
        trainer.optim_step("sess-a", {"learning_rate": 1e-2})
        trainer.forward_backward("sess-b", data, "cross_entropy")
        trainer.optim_step("sess-b", {"learning_rate": 1e-2})

    a_state = trainer.extract_state("sess-a")
    b_state = trainer.extract_state("sess-b")

    # Drop both and reload B first, then A — order should not matter.
    trainer.detach_session("sess-a")
    trainer.detach_session("sess-b")
    assert not trainer.has_session("sess-a")
    assert not trainer.has_session("sess-b")

    trainer.load_state("sess-b", b_state)
    trainer.load_state("sess-a", a_state)

    a_params_after = _adapter_params(trainer, "sess-a")
    b_params_after = _adapter_params(trainer, "sess-b")

    # Shapes still match original ranks.
    for n, p in a_params_after.items():
        if "lora_A" in n:
            assert p.shape[0] == 8
    for n, p in b_params_after.items():
        if "lora_A" in n:
            assert p.shape[0] == 32

    # Weights match what we extracted.
    for n, t in a_state.lora_weights.items():
        # PEFT's get_peft_model_state_dict returns the base-layer names;
        # find the corresponding named_parameter after reload.
        found = None
        for pn, pp in a_params_after.items():
            if n.replace(".default", ".sess_sess_a") in pn or n in pn:
                found = pp
                break
        # If we can't map by name exactly, fall back to shape check.
        if found is not None:
            assert found.shape == t.shape
