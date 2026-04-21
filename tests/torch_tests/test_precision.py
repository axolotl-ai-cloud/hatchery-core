# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for selective fp32 upcasting of precision-sensitive modules.

These build tiny fake models whose qualified module names mirror the
real MoE architectures (``block_sparse_moe.gate``, ``mlp.gate``,
etc.) so the suffix-matching logic in
:func:`hatchery.core.precision.apply_precision_policy` can be
exercised without downloading a real model.

The key properties we pin:

* Upcast modules end up in fp32 and survive forward calls on a bf16
  model (i.e., the dtype-safety pre-hook works).
* Unknown architectures are a no-op.
* Embeddings are only upcast when explicitly requested.
* Double-application is idempotent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from hatchery.core.fused_losses import ModelCapability  # noqa: E402
from hatchery.core.precision import (  # noqa: E402
    _is_floating_tensor,
    apply_precision_policy,
)


class _FakeMixtralBlock(nn.Module):
    """Mimics the relevant parts of ``MixtralDecoderLayer.block_sparse_moe``.

    Has a router (``gate``) that would normally live at
    ``layers.N.block_sparse_moe.gate`` in the full model. We don't
    wire up actual expert MLPs — the test only cares about the
    router promotion.
    """

    def __init__(self, hidden: int, num_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)


class _FakeMixtralLayer(nn.Module):
    def __init__(self, hidden: int, num_experts: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.block_sparse_moe = _FakeMixtralBlock(hidden, num_experts)


class _FakeMixtral(nn.Module):
    """Whole fake model. Has an embedding, a stack of layers, and an
    lm_head — matching the shape HF models use at the class level.
    """

    def __init__(
        self, *, vocab: int = 64, hidden: int = 16, num_layers: int = 2, num_experts: int = 4
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_FakeMixtralLayer(hidden, num_experts) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


def _mixtral_capability() -> ModelCapability:
    return ModelCapability(fp32_module_suffixes=("block_sparse_moe.gate",))


def test_router_is_upcast_to_fp32():
    model = _FakeMixtral().to(torch.bfloat16)
    cap = _mixtral_capability()

    report = apply_precision_policy(model, capability=cap, main_dtype=torch.bfloat16)

    # Both layers' routers should be upcast.
    assert len(report.upcast_modules) == 2
    assert all(name.endswith("block_sparse_moe.gate") for name in report.upcast_modules)

    for layer in model.layers:
        assert layer.block_sparse_moe.gate.weight.dtype == torch.float32
        # Unrelated modules stay in bf16.
        assert layer.q_proj.weight.dtype == torch.bfloat16


def test_router_accepts_bf16_input_after_promotion():
    """The whole point of the pre-hook: feeding a bf16 tensor into
    the fp32 router must work without a dtype error, and the output
    must land in fp32 for the containing MoE block to downcast.
    """
    model = _FakeMixtral().to(torch.bfloat16)
    apply_precision_policy(model, capability=_mixtral_capability(), main_dtype=torch.bfloat16)

    gate = model.layers[0].block_sparse_moe.gate
    hidden = torch.randn(2, 16, dtype=torch.bfloat16)
    out = gate(hidden)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_no_upcast_for_empty_capability():
    """A capability with no fp32 targets is a no-op — nothing gets
    promoted, report is empty with a reason.
    """
    model = _FakeMixtral().to(torch.bfloat16)
    cap = ModelCapability()  # default: no fp32 config
    report = apply_precision_policy(model, capability=cap, main_dtype=torch.bfloat16)
    assert report.upcast_modules == []
    assert report.upcast_embeddings == []
    assert "no fp32" in report.skipped_reason
    # Confirm nothing actually moved.
    for layer in model.layers:
        assert layer.block_sparse_moe.gate.weight.dtype == torch.bfloat16


def test_embedding_upcast_opt_in():
    model = _FakeMixtral().to(torch.bfloat16)
    cap = ModelCapability(fp32_embeddings=True)
    report = apply_precision_policy(model, capability=cap, main_dtype=torch.bfloat16)

    assert report.upcast_embeddings == ["embed_tokens"]
    assert model.embed_tokens.weight.dtype == torch.float32
    # Routers NOT upcast since this capability doesn't declare them.
    for layer in model.layers:
        assert layer.block_sparse_moe.gate.weight.dtype == torch.bfloat16


def test_both_router_and_embedding_upcast():
    model = _FakeMixtral().to(torch.bfloat16)
    cap = ModelCapability(
        fp32_module_suffixes=("block_sparse_moe.gate",),
        fp32_embeddings=True,
    )
    report = apply_precision_policy(model, capability=cap, main_dtype=torch.bfloat16)

    assert len(report.upcast_modules) == 2
    assert len(report.upcast_embeddings) == 1
    assert report.total == 3


def test_apply_is_idempotent():
    """Running the policy twice on the same model must NOT register
    two hooks — double-casting would work but would slow the forward
    and make the report misleading.
    """
    model = _FakeMixtral().to(torch.bfloat16)
    cap = _mixtral_capability()

    apply_precision_policy(model, capability=cap, main_dtype=torch.bfloat16)
    gate = model.layers[0].block_sparse_moe.gate

    # The internal tag should be present after the first run.
    assert getattr(gate, "_tinker_fp32_hook_registered", False) is True
    pre_hooks_after_first = len(gate._forward_pre_hooks)

    apply_precision_policy(model, capability=cap, main_dtype=torch.bfloat16)
    pre_hooks_after_second = len(gate._forward_pre_hooks)

    assert pre_hooks_after_first == pre_hooks_after_second, (
        f"pre-hook registered twice: {pre_hooks_after_first} -> {pre_hooks_after_second}"
    )


def test_suffix_matching_is_strict_not_substring():
    """The policy must match module-name *suffixes*, not substrings.
    A module named ``something.gate_proj`` should NOT be upcast
    when the pattern is ``mlp.gate`` — that would clobber Llama's
    SwiGLU gate projection (which needs to stay bf16).
    """

    class _LlamaMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 8)  # SwiGLU gate — NOT an MoE router
            self.up_proj = nn.Linear(8, 8)
            self.down_proj = nn.Linear(8, 8)

    class _FakeQwenMoeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # real MoE router — has attribute name ``gate``
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(8, 4)
            # Also has a Llama-style MLP with ``gate_proj`` which
            # must NOT be promoted by the suffix ``mlp.gate``.
            self.mlp_dense = _LlamaMLP()

    class _FakeRoot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_FakeQwenMoeLayer()])

    model = _FakeRoot().to(torch.bfloat16)
    cap = ModelCapability(fp32_module_suffixes=("mlp.gate",))
    report = apply_precision_policy(model, capability=cap, main_dtype=torch.bfloat16)

    # Only the Qwen-MoE router matches — gate_proj is left alone.
    assert len(report.upcast_modules) == 1
    assert report.upcast_modules[0].endswith("mlp.gate")
    assert model.layers[0].mlp_dense.gate_proj.weight.dtype == torch.bfloat16


def test_kwarg_inputs_are_upcast():
    """Some HF forwards pass the hidden state as a keyword arg. The
    pre-hook must upcast kwargs as well as positional args.
    """

    class _CheckDtype(nn.Linear):
        def forward(self, *args, **kwargs):  # type: ignore[override]
            # Record what dtype we actually saw on the input.
            seen = None
            if args and isinstance(args[0], torch.Tensor):
                seen = args[0].dtype
            elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
                seen = kwargs["input"].dtype
            self._tinker_test_observed_dtype = seen  # type: ignore[attr-defined]
            return super().forward(*args, **kwargs)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = _CheckDtype(8, 4)

    class _Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = _Block()

    class _Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Wrap()])

    model = _Root().to(torch.bfloat16)
    apply_precision_policy(
        model,
        capability=ModelCapability(fp32_module_suffixes=("block_sparse_moe.gate",)),
        main_dtype=torch.bfloat16,
    )

    gate = model.layers[0].block_sparse_moe.gate
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    _ = gate(input=x)
    assert gate._tinker_test_observed_dtype == torch.float32


def test_known_moe_architectures_declare_routers():
    """Pin the capability table entries for MoE archs. If someone
    adds a new MoE model they should also add a precision rule, and
    if someone removes the rule by accident this test catches it.
    """
    from hatchery.core.fused_losses import _KNOWN_CAPABILITIES

    moe_archs_with_routers = {
        "MixtralForCausalLM": "block_sparse_moe.gate",
        "Qwen2MoeForCausalLM": "mlp.gate",
        "Qwen3MoeForCausalLM": "mlp.gate",
        "DeepseekV3ForCausalLM": "mlp.gate",
    }
    for arch, expected_suffix in moe_archs_with_routers.items():
        cap = _KNOWN_CAPABILITIES[arch]
        assert expected_suffix in cap.fp32_module_suffixes, (
            f"{arch} missing fp32 router config for {expected_suffix}"
        )


def test_is_floating_tensor_helper():
    assert _is_floating_tensor(torch.tensor([1.0])) is True
    assert _is_floating_tensor(torch.tensor([1], dtype=torch.long)) is False
    assert _is_floating_tensor(None) is False
    assert _is_floating_tensor("not a tensor") is False
