# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for fused loss paths (CE + GRPO).

We exercise the torch-only chunked implementation on CPU (no CUDA,
no Triton). The fused kernel path (CCE / Liger) takes the same
entry point, so if the chunked path is correct the worker's
gradient parity test in ``test_worker_cpu.py`` covers the end-to-end
code path; these unit tests pin the per-function math.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hatchery.core.fused_losses import (  # noqa: E402
    _chunked_cross_entropy,
    _liger_loss_tensor,
    _lm_head_is_clean,
    is_fused_eligible,
    is_fused_grpo_eligible,
)


def test_chunked_ce_matches_reference_no_ignore():
    """With no ignored tokens, chunked CE should equal torch's CE on
    the fully materialized logits.
    """
    torch.manual_seed(0)
    B, T, H, V = 2, 8, 16, 64
    flat_hidden = torch.randn(B * T, H, requires_grad=True)
    lm_weight = torch.randn(V, H)
    lm_bias = torch.randn(V)
    targets = torch.randint(0, V, (B * T,))

    ours = _chunked_cross_entropy(flat_hidden, lm_weight, lm_bias, targets, chunk_size=9)

    # Reference: materialize full logits.
    ref_logits = torch.nn.functional.linear(flat_hidden.float(), lm_weight.float(), lm_bias.float())
    ref = torch.nn.functional.cross_entropy(ref_logits, targets, ignore_index=-100)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-5)


def test_chunked_ce_respects_ignore_index():
    torch.manual_seed(1)
    B, T, H, V = 1, 6, 8, 16
    flat_hidden = torch.randn(B * T, H, requires_grad=True)
    lm_weight = torch.randn(V, H)
    targets = torch.tensor([3, -100, 5, 0, -100, 2])

    ours = _chunked_cross_entropy(flat_hidden, lm_weight, None, targets, chunk_size=4)

    ref_logits = torch.nn.functional.linear(flat_hidden.float(), lm_weight.float())
    ref = torch.nn.functional.cross_entropy(ref_logits, targets, ignore_index=-100)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_chunked_ce_gradient_matches_reference():
    """The gradient w.r.t. ``flat_hidden`` must match the reference
    path — that's the tensor the upstream LoRA backward lands on.
    """
    torch.manual_seed(2)
    B, T, H, V = 2, 4, 12, 20
    flat_hidden_a = torch.randn(B * T, H, requires_grad=True)
    flat_hidden_b = flat_hidden_a.detach().clone().requires_grad_(True)
    lm_weight = torch.randn(V, H)
    targets = torch.randint(0, V, (B * T,))
    targets[3] = -100
    targets[5] = -100

    ours = _chunked_cross_entropy(flat_hidden_a, lm_weight, None, targets, chunk_size=3)
    ours.backward()

    ref_logits = torch.nn.functional.linear(flat_hidden_b.float(), lm_weight.float())
    ref = torch.nn.functional.cross_entropy(ref_logits, targets, ignore_index=-100)
    ref.backward()

    assert torch.allclose(flat_hidden_a.grad, flat_hidden_b.grad, atol=1e-5, rtol=1e-5)


def test_liger_loss_tensor_accepts_scalar_and_tuple_outputs():
    loss = torch.tensor(1.25, requires_grad=True)
    z_loss = torch.tensor(0.5)
    token_accuracy = torch.tensor(0.75)

    assert _liger_loss_tensor(loss) is loss
    assert _liger_loss_tensor((loss, z_loss, token_accuracy)) is loss


def test_is_fused_eligible_rejects_2d_labels():
    """SDFT's 2-D labels path can't use the fused kernel."""
    labels = torch.zeros(2, 4, 3, dtype=torch.long)
    assert (
        is_fused_eligible(
            loss_fn="cross_entropy",
            labels=labels,
            weights=None,
            peft_model=_fake_peft_clean_lm_head(),
        )
        is False
    )


def test_is_fused_eligible_rejects_custom_weights():
    labels = torch.zeros(2, 4, dtype=torch.long)
    weights = torch.ones(2, 4)
    assert (
        is_fused_eligible(
            loss_fn="cross_entropy",
            labels=labels,
            weights=weights,
            peft_model=_fake_peft_clean_lm_head(),
        )
        is False
    )


def test_is_fused_eligible_rejects_non_ce_losses():
    labels = torch.zeros(2, 4, dtype=torch.long)
    for loss_fn in ["ppo", "importance_sampling", "cispo", "dro"]:
        assert (
            is_fused_eligible(
                loss_fn=loss_fn,
                labels=labels,
                weights=None,
                peft_model=_fake_peft_clean_lm_head(),
            )
            is False
        )


def test_is_fused_eligible_accepts_sft_default():
    labels = torch.zeros(2, 4, dtype=torch.long)
    assert (
        is_fused_eligible(
            loss_fn="cross_entropy",
            labels=labels,
            weights=None,
            peft_model=_fake_peft_clean_lm_head(),
        )
        is True
    )


def test_capability_lookup_known_archs():
    from hatchery.core.fused_losses import (
        get_model_capability,
        list_known_architectures,
    )

    class FakeLlama(torch.nn.Module):
        pass

    class FakeMysteryModel(torch.nn.Module):
        pass

    FakeLlama.__name__ = "LlamaForCausalLM"
    FakeMysteryModel.__name__ = "NovelArchitecture2026"

    class FakePeftLlama:
        class base_model:  # noqa: N801
            model = FakeLlama()

    class FakePeftMystery:
        class base_model:  # noqa: N801
            model = FakeMysteryModel()

    cap_llama = get_model_capability(FakePeftLlama)
    assert cap_llama.supports_cce is True
    assert cap_llama.supports_liger is True
    assert "llama" in cap_llama.liger_patch_module

    # Unknown arch falls through to the permissive default.
    cap_mystery = get_model_capability(FakePeftMystery)
    assert cap_mystery.supports_cce is True
    assert cap_mystery.supports_chunked is True

    known = list_known_architectures()
    assert "LlamaForCausalLM" in known
    assert "Qwen2ForCausalLM" in known


def test_gpt2_capability_disables_liger():
    """GPT-2's tied embedding races with the Identity swap in the
    Liger path. The capability table should mark it Liger-disabled
    so the worker falls through to CCE or chunked.
    """
    from hatchery.core.fused_losses import get_model_capability

    class FakeGPT2(torch.nn.Module):
        pass

    FakeGPT2.__name__ = "GPT2LMHeadModel"

    class FakePeft:
        class base_model:  # noqa: N801
            model = FakeGPT2()

    cap = get_model_capability(FakePeft)
    assert cap.supports_liger is False
    assert cap.supports_chunked is True


def test_lm_head_is_clean_rejects_lora_wrapper():
    """If the LoRA adapter targets lm_head, we must fall back to the
    direct path. A peft LoraLayer always has a ``base_layer`` attribute.
    """
    fake_lora_lm_head = torch.nn.Linear(16, 64)
    fake_lora_lm_head.base_layer = torch.nn.Linear(16, 64)  # type: ignore[attr-defined]

    class FakePeft:
        class base_model:  # noqa: N801
            class model:  # noqa: N801
                pass

    FakePeft.base_model.model.lm_head = fake_lora_lm_head
    assert _lm_head_is_clean(FakePeft) is False


# ─── helpers ────────────────────────────────────────────────────────


class _FakePeft:
    class base_model:  # noqa: N801
        class model:  # noqa: N801
            lm_head = torch.nn.Linear(16, 64)


def _fake_peft_clean_lm_head():
    return _FakePeft


# ─── Fused GRPO eligibility ──────────────────────────────────────────


def test_fused_grpo_eligible_for_grpo():
    labels = torch.tensor([[1, 2, 3]])
    weights = torch.ones_like(labels, dtype=torch.float32)
    old_lp = torch.zeros_like(labels, dtype=torch.float32)
    adv = torch.ones_like(labels, dtype=torch.float32)
    # Only eligible if Liger is importable.
    result = is_fused_grpo_eligible("grpo", labels, weights, old_lp, adv)
    # We don't require Liger to be installed, but the function shouldn't crash.
    assert isinstance(result, bool)


def test_fused_grpo_eligible_for_dapo():
    labels = torch.tensor([[1, 2, 3]])
    old_lp = torch.zeros_like(labels, dtype=torch.float32)
    adv = torch.ones_like(labels, dtype=torch.float32)
    result = is_fused_grpo_eligible("dapo", labels, None, old_lp, adv)
    assert isinstance(result, bool)


def test_fused_grpo_rejects_cross_entropy():
    labels = torch.tensor([[1, 2, 3]])
    assert is_fused_grpo_eligible("cross_entropy", labels, None) is False


def test_fused_grpo_rejects_missing_advantages():
    labels = torch.tensor([[1, 2, 3]])
    old_lp = torch.zeros_like(labels, dtype=torch.float32)
    assert is_fused_grpo_eligible("grpo", labels, None, old_lp, None) is False


def test_fused_grpo_rejects_3d_labels():
    labels = torch.tensor([[[1, 2], [3, 4]]])  # 3-D
    old_lp = torch.zeros(1, 2, dtype=torch.float32)
    adv = torch.ones(1, 2, dtype=torch.float32)
    assert is_fused_grpo_eligible("grpo", labels, None, old_lp, adv) is False


def test_per_job_dispatch_independence():
    """Different jobs can use different loss paths without state leakage.
    This validates that the eligibility checks are stateless."""
    labels_1d = torch.tensor([[1, 2, 3]])
    labels_3d = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
    weights = torch.ones(1, 3, dtype=torch.float32)
    old_lp = torch.zeros(1, 3, dtype=torch.float32)
    adv = torch.ones(1, 3, dtype=torch.float32)

    fake_peft = _fake_peft_clean_lm_head()

    # CE job → fused CE eligible (with clean lm_head).
    ce_fused = is_fused_eligible(
        loss_fn="cross_entropy", labels=labels_1d, weights=None, peft_model=fake_peft
    )
    # GRPO job with 1-D labels → fused GRPO eligible (if Liger available).
    grpo_fused = is_fused_grpo_eligible("grpo", labels_1d, weights, old_lp, adv)
    # CE job with 2-D labels → not fused eligible.
    ce_2d = is_fused_eligible(
        loss_fn="cross_entropy", labels=labels_3d, weights=None, peft_model=fake_peft
    )
    # PPO job → not fused GRPO eligible (not in Liger GRPO map).
    ppo_fused = is_fused_grpo_eligible("ppo", labels_1d, weights, old_lp, adv)

    # These are independent — no shared state between checks.
    assert ce_fused is True  # standard SFT case
    assert ce_2d is False  # 2-D labels not eligible
    assert ppo_fused is False  # PPO not in Liger GRPO map
    # grpo_fused depends on Liger availability — just check it's bool.
    assert isinstance(grpo_fused, bool)
