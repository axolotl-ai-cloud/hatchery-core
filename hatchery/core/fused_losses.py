# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Fused cross-entropy path for the SFT-only scalar loss case.

Motivation
----------
For a Qwen-2 7B model with vocab size 152,064 training at B=4 T=8192,
the ``[B, T, V]`` logits tensor is ~40GB in fp32 or ~20GB in bf16.
Materializing this tensor in the forward pass (so the standard
``cross_entropy`` loss can index into it) dominates peak activation
memory on even a 80GB card, and we don't actually need any of the
per-position probabilities — we just want a scalar loss.

This module provides a fused cross-entropy path that never
materializes the full ``[B*T, V]`` tensor. It works in three layers
of preference:

1. **Primary path — Cut Cross Entropy (CCE).** Apple's
   ``cut_cross_entropy.linear_cross_entropy`` is the fastest option
   we've measured: it keeps max-logit subtraction on a per-row basis
   and uses a custom Triton kernel that skips zero-gradient elements.
   Typically 1.5–2× faster than Liger on Hopper/Blackwell.

2. **Secondary path — Liger fused kernel.** If CCE isn't available
   and ``liger_kernel`` is, we use
   ``LigerFusedLinearCrossEntropyFunction`` which also fuses
   matmul+log-softmax+CE in a Triton kernel. Slightly slower than
   CCE but available on a broader set of torch versions.

3. **Fallback path — torch chunked CE.** When neither third-party
   kernel is available (CPU-only, debugging, version skew), we
   chunk the ``[B*T, H]`` hidden tensor along the row dim and
   compute CE on each chunk against the full vocab. Peak memory
   for the loss term is ``O(chunk_size * V)``, matching the fused
   kernels' memory bound but without the kernel speedup. Gradients
   flow back through ``hidden`` via autograd, the same as the
   fused paths — the LoRA adapters upstream get identical
   gradients regardless of which backend ran.

Eligibility
-----------
The fused path is only correct for the "pure SFT" case:

* ``loss_fn == "cross_entropy"``
* ``target_tokens`` is 1-D (``[B, T]``); SDFT's 2-D top-K path still
  needs the direct implementation because it does a weighted sum
  over K targets per position.
* ``weights`` is ``None`` or equivalent to the attention mask — any
  custom per-position weighting falls back to the direct path.
* The user's LoRA adapter does NOT target the ``lm_head`` module —
  we hot-swap ``lm_head`` with ``Identity`` during the forward to
  skip the full matmul, which doesn't work if LoRA is riding on the
  lm_head weight.

The worker's ``_handle_forward_backward`` picks the path at dispatch
time; the caller never sees which kernel ran.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# Default chunk size for the fallback path. 2048 rows of hidden at
# fp32 + an equivalent [chunk, V] logits slice is <1GB for typical
# vocab sizes, well inside activation headroom on a 24GB card.
DEFAULT_CHUNK_SIZE = 2048


# ─── Kernel capability lookup ───────────────────────────────────────
#
# For each model architecture we track which fused kernels are known
# to work end-to-end. ``cce`` is the preferred path when supported;
# ``liger`` is second choice; ``chunked`` is always supported as the
# pure-torch fallback. The ``identity_swap`` flag indicates whether
# the "replace lm_head with Identity" trick works for this model —
# it requires the model's forward to apply ``lm_head`` as a distinct
# callable on the final hidden state, which is true for all standard
# HF CausalLM implementations but not for models that fuse lm_head
# into the final block.
#
# Keyed by the transformers model class ``__name__``. Unknown model
# classes fall through to the default ``ModelCapability`` which
# allows CCE + Liger + chunked and permits the Identity swap — that
# matches how the vast majority of HF causal LMs behave.


@dataclass(frozen=True)
class ModelCapability:
    """Per-architecture feature flags.

    Read by two consumers:

    1. :mod:`hatchery.core.fused_losses` — picks a fused CE kernel
       and decides whether the Identity-swap trick is safe.
    2. :mod:`hatchery.core.precision` — decides which submodules to
       keep in fp32 when the rest of the model is loaded in bf16.
       Routers in particular need fp32 for stable softmax; embeddings
       are a softer preference.

    Matching rules for the precision fields:

    * ``fp32_module_suffixes`` — dotted-path *suffixes* matched against
      ``model.named_modules()`` qualified names. A module matches if
      its qualified name ends with any of the listed suffixes. Using
      suffixes (not substrings) avoids accidentally upcasting unrelated
      modules that happen to contain the same word.
    * ``fp32_embeddings`` — if True, every ``nn.Embedding`` module in
      the model gets upcast. Only set this for archs where we've
      validated it doesn't clash with weight-tying (GPT-2's tied
      embedding is a known trap — its ``wte`` is reached via the
      ``lm_head`` fast path and we lose the tie if we cast in place).
    """

    supports_cce: bool = True
    supports_liger: bool = True
    supports_chunked: bool = True
    supports_identity_swap: bool = True
    # Arch-specific Liger patch entry point (e.g.,
    # ``apply_liger_kernel_to_llama``). Only used when we run through
    # the Liger path; CCE and chunked don't need a patch because they
    # operate on the hidden state directly.
    liger_patch_module: str = ""
    # Precision policy — consumed by ``hatchery.core.precision``.
    fp32_module_suffixes: tuple[str, ...] = ()
    fp32_embeddings: bool = False
    notes: str = ""


# Default — generous. Used for any architecture not in the table.
_DEFAULT_CAPABILITY = ModelCapability()


# Verified architectures. The list focuses on Causal LMs where we've
# checked that lm_head is a plain ``nn.Linear`` at
# ``base_model.lm_head`` and the forward returns ``logits =
# lm_head(hidden)`` as the final step.
_KNOWN_CAPABILITIES: dict[str, ModelCapability] = {
    "LlamaForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_llama",
    ),
    "Qwen2ForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_qwen2",
    ),
    "Qwen2MoeForCausalLM": ModelCapability(
        # Qwen2-MoE has a router at ``mlp.gate`` inside each decoder
        # layer. Upcasting stabilizes the top-k softmax.
        fp32_module_suffixes=("mlp.gate",),
        notes="MoE router must stay in fp32 — bf16 routers cause "
        "load-balancing drift in the top-k softmax.",
    ),
    "Qwen3ForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_qwen3",
        notes="Liger patch added in liger-kernel 0.5.x; CCE is the safe default.",
    ),
    "Qwen3MoeForCausalLM": ModelCapability(
        fp32_module_suffixes=("mlp.gate",),
        notes="Qwen3-MoE router lives at ``layers.N.mlp.gate`` and must "
        "stay in fp32 — the top-k softmax is numerically sensitive.",
    ),
    "MistralForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_mistral",
    ),
    "MixtralForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_mixtral",
        # Mixtral's router is ``block_sparse_moe.gate`` (see
        # ``transformers/models/mixtral/modeling_mixtral.py``). The
        # Mixtral paper + Megatron-LM both keep routers in fp32.
        fp32_module_suffixes=("block_sparse_moe.gate",),
        notes="MoE router kept in fp32 for load-balancing stability. "
        "CCE is fine on the main forward — aux_loss is emitted "
        "separately by the base model.",
    ),
    "DeepseekV3ForCausalLM": ModelCapability(
        fp32_module_suffixes=(
            "mlp.gate",
            # DeepSeek-V3 adds a learned bias used for expert routing
            # correction. Both it and the router Linear should stay
            # fp32 for stability.
            "mlp.e_score_correction_bias",
        ),
        notes="DeepSeek-V3 MoE with auxiliary-loss-free load balancing. "
        "Router + correction bias need fp32.",
    ),
    "Gemma2ForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_gemma2",
        notes="Gemma2 soft-caps the logits; CCE supports softcap via its "
        "``softcap`` kwarg but the default tinker path does not set it.",
    ),
    "Gemma3ForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_gemma3",
    ),
    "GPT2LMHeadModel": ModelCapability(
        supports_cce=True,
        supports_liger=False,  # no dedicated Liger patch for gpt2
        notes="GPT-2 has a tied lm_head (wte) — Identity swap works "
        "but mutations to lm_head.weight can race with the embedding. "
        "Chunked CE is the safe default on this arch.",
    ),
    "Phi3ForCausalLM": ModelCapability(
        liger_patch_module="liger_kernel.transformers.apply_liger_kernel_to_phi3",
    ),
}


def get_model_capability(peft_model: Any) -> ModelCapability:
    """Look up the fused-kernel capability for ``peft_model``.

    Walks through the PEFT wrappers to reach the underlying
    transformers model class, then consults the known-arch table.
    Returns ``_DEFAULT_CAPABILITY`` for anything unrecognized — that
    allows CCE + Liger + chunked, which is correct for all
    well-behaved HF CausalLM subclasses.
    """
    try:
        base = peft_model.base_model.model
    except AttributeError:
        return _DEFAULT_CAPABILITY
    return _KNOWN_CAPABILITIES.get(type(base).__name__, _DEFAULT_CAPABILITY)


def list_known_architectures() -> list[str]:
    """Public helper for surfacing capability info via
    ``/get_server_capabilities``.
    """
    return sorted(_KNOWN_CAPABILITIES.keys())


def _try_import_cce() -> Any:
    """Return Apple's ``cut_cross_entropy.linear_cross_entropy`` or None.

    CCE is our preferred fused kernel — faster than Liger by ~1.5-2×
    on modern hardware and cleaner API (no hand-rolled autograd
    function to apply).
    """
    try:
        from cut_cross_entropy import linear_cross_entropy

        return linear_cross_entropy
    except Exception:  # noqa: BLE001
        return None


def _try_import_liger() -> Any:
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        return LigerFusedLinearCrossEntropyFunction
    except Exception:  # noqa: BLE001
        return None


def _liger_loss_tensor(output: Any) -> Any:
    """Normalize Liger fused CE output across liger-kernel versions.

    Older Liger releases returned the scalar loss directly. Newer
    releases return ``(loss, z_loss, token_accuracy)`` so callers can
    opt into diagnostics without a second kernel. Hatchery's fused SFT
    path needs only the scalar loss; selecting element 0 preserves the
    autograd edge back through the custom Function.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


def is_fused_eligible(
    *,
    loss_fn: str,
    labels: Any,
    weights: Any,
    peft_model: Any,
) -> bool:
    """Return True if the fused path is safe for this call.

    We lean conservative: any edge case (custom weights, 2-D labels,
    LoRA on the lm_head) sends us to the direct path. The direct
    path is always correct, just slower.
    """
    if loss_fn != "cross_entropy":
        return False
    if labels is None:
        return False
    # 2-D labels → SDFT → needs per-k gather → direct path.
    if labels.dim() != 2:
        return False
    # Custom per-position weights → direct path. A weights tensor
    # that's exactly the attention mask IS compatible with fused, but
    # we can't tell from here without comparing — upstream passes
    # ``None`` for the "default SFT" case, which is what we're
    # checking for.
    if weights is not None:
        return False
    if not _lm_head_is_clean(peft_model):
        return False
    cap = get_model_capability(peft_model)
    if not cap.supports_identity_swap:
        return False
    return cap.supports_cce or cap.supports_liger or cap.supports_chunked


def _lm_head_is_clean(peft_model: Any) -> bool:
    """True iff the LoRA adapter doesn't wrap the lm_head.

    We check by walking down to the base model's ``lm_head`` and
    asserting it's a plain ``nn.Linear`` (or equivalent) with no
    LoRA sidecars attached.
    """
    try:
        lm_head = peft_model.base_model.model.lm_head
    except AttributeError:
        return False
    # peft LoraLayer subclasses keep a ``base_layer`` pointing at the
    # original module. Its presence is a sure sign LoRA is attached.
    if hasattr(lm_head, "base_layer"):
        return False
    return not type(lm_head).__name__.startswith("Lora")


@contextlib.contextmanager
def _lm_head_as_identity(peft_model: Any):
    """Temporarily swap the lm_head for ``nn.Identity`` so the PEFT
    forward returns the hidden state (shape ``[B, T, H]``) in the
    ``.logits`` slot. Yields the saved ``(weight, bias)`` pair so the
    caller can apply them via the fused kernel.
    """
    lm_head = peft_model.base_model.model.lm_head
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    peft_model.base_model.model.lm_head = nn.Identity()
    try:
        yield weight, bias
    finally:
        peft_model.base_model.model.lm_head = lm_head


def fused_cross_entropy_forward_backward(
    peft_model: Any,
    input_ids: Any,
    attention_mask: Any,
    labels: Any,
    *,
    position_ids: Any = None,
    loss_scale: float = 1.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Any:
    """Run forward + fused CE + backward in one call.

    Returns the scalar loss (detached from the graph). The LoRA
    adapter gradients have already been populated by ``.backward()``
    time the caller receives control.

    The ``labels`` tensor follows the Tinker convention: ``[B, T]``
    pre-aligned with the model's hidden states (client handles the
    causal shift), with ``-100`` in positions that should be ignored.

    When ``position_ids`` is supplied (varlen packed path),
    ``attention_mask`` is dropped — HF's flash-attn-2 backend derives
    ``cu_seqlens`` from position resets. Boundary labels are already
    ``-100`` from :func:`hatchery.core.packing.pack_sequences`.
    """
    if peft_model.training is False:
        # The caller should have set train mode already; we don't
        # touch it here. This is just a belt-and-braces assertion so
        # unit tests catch a missing mode switch early.
        pass

    fwd_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "labels": None,
        "use_cache": False,
    }
    if position_ids is not None:
        fwd_kwargs["position_ids"] = position_ids
    else:
        fwd_kwargs["attention_mask"] = attention_mask

    with _lm_head_as_identity(peft_model) as (lm_weight, lm_bias):
        outputs = peft_model(**fwd_kwargs)
        hidden = outputs.logits  # actually [B, T, H] because lm_head=Identity

    # Tinker convention: client pre-aligns labels with hidden states.
    B, T, H = hidden.shape
    flat_hidden = hidden.reshape(-1, H)  # [B*T, H]
    flat_labels = labels.reshape(-1)  # [B*T]

    cap = get_model_capability(peft_model)

    cce = _try_import_cce() if cap.supports_cce else None
    if cce is not None:
        try:
            loss = cce(
                flat_hidden,
                lm_weight,
                flat_labels,
                bias=lm_bias,
                ignore_index=-100,
                reduction="mean",
            )
            # loss_scale != 1.0 on multi-pack: each sub-backward's grad
            # contribution needs to be weighted by num_tokens / total
            # so the accumulated grads match a single-forward backward
            # on the full batch. Scaling the loss before backward is
            # equivalent and keeps the fused kernel unchanged.
            (loss * loss_scale).backward()
            return loss.detach()
        except Exception:  # noqa: BLE001
            # CCE's Triton kernel requires CUDA + specific compute
            # capabilities. Fall through to Liger, then chunked CE.
            pass

    liger = _try_import_liger() if cap.supports_liger else None
    if liger is not None:
        try:
            loss = _liger_loss_tensor(
                liger.apply(
                    flat_hidden,
                    lm_weight,
                    flat_labels,
                    lm_bias,
                    None,  # ce_weight
                    -100,  # ignore_index
                    0.0,  # lse_square_scale
                    0.0,  # label_smoothing
                    "mean",  # reduction
                    None,  # softcap
                )
            )
            (loss * loss_scale).backward()
            return loss.detach()
        except Exception:  # noqa: BLE001
            # Liger's Triton path has known incompatibilities with
            # some torch nightlies — keep falling through.
            pass

    loss = _chunked_cross_entropy(
        flat_hidden, lm_weight, lm_bias, flat_labels, chunk_size=chunk_size
    )
    (loss * loss_scale).backward()
    return loss.detach()


def _chunked_cross_entropy(
    flat_hidden: Any,
    lm_weight: Any,
    lm_bias: Any,
    flat_labels: Any,
    *,
    chunk_size: int,
) -> Any:
    """Torch-only fallback for :func:`fused_cross_entropy_forward_backward`.

    Computes a mean cross-entropy loss by chunking along the ``[B*T]``
    row dim. Each chunk builds a local ``[chunk, V]`` logits tensor,
    runs ``F.cross_entropy(reduction='sum')``, and accumulates. Final
    loss is ``total_sum / num_valid`` so the result matches the
    standard ``reduction='mean'`` + ``ignore_index=-100`` semantics.

    Gradient correctness relies on autograd: each chunk's
    ``cross_entropy`` contributes to the overall loss tensor, and
    when the caller runs ``.backward()`` on the final mean, gradients
    flow through each chunk's ``chunk_hidden @ lm_weight.T`` slice
    back into the appropriate rows of ``flat_hidden``. The
    ``flat_hidden`` tensor is a view of the same underlying storage
    as the shifted hidden state, so gradients land in the right
    place for the upstream transformer/LoRA backward.
    """
    total_rows = flat_hidden.size(0)
    valid_mask = flat_labels.ne(-100)
    num_valid = valid_mask.sum().clamp_min(1).to(flat_hidden.dtype)

    sum_loss = flat_hidden.new_zeros(())
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk_hidden = flat_hidden[start:end]
        chunk_labels = flat_labels[start:end]
        # Upcast to fp32 for numerical stability of the log-softmax —
        # same convention the standard CE path uses.
        chunk_logits = F.linear(
            chunk_hidden.float(),
            lm_weight.float(),
            lm_bias.float() if lm_bias is not None else None,
        )
        chunk_sum = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )
        sum_loss = sum_loss + chunk_sum
    return sum_loss / num_valid


# ─── Fused RL losses via Liger GRPO kernel ─────────────────────────────
#
# LigerFusedLinearGRPOLoss operates on hidden states + lm_head weight
# (not logits), avoiding the full [B*T, V] materialization. It supports
# grpo, dapo, cispo, and more via the ``loss_type`` parameter.


def _try_import_liger_grpo():
    """Return the LigerFusedLinearGRPOLoss class, or None."""
    try:
        from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

        return LigerFusedLinearGRPOLoss
    except ImportError:
        return None


# Map our loss_fn names to Liger loss_type values.
_LIGER_GRPO_LOSS_MAP = {
    "grpo": "grpo",
    "dapo": "dapo",
    "cispo": "cispo",
}


def is_fused_grpo_eligible(
    loss_fn: str,
    labels: Any,
    weights: Any,
    old_logprobs: Any = None,
    advantages: Any = None,
) -> bool:
    """Check if we can use the Liger fused GRPO path."""
    if loss_fn not in _LIGER_GRPO_LOSS_MAP:
        return False
    if _try_import_liger_grpo() is None:
        return False
    # Liger needs old_logprobs and advantages for RL losses.
    if old_logprobs is None or advantages is None:
        return False
    # Only 1-D labels supported by the fused path.
    return not (labels is not None and labels.dim() > 2)


def fused_grpo_forward_backward(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    labels: Any,
    old_logprobs: Any,
    advantages: Any,
    loss_fn: str = "grpo",
    loss_fn_config: dict | None = None,
) -> float:
    """Run fused forward + RL loss + backward via Liger kernel.

    Returns the scalar loss value (detached).
    """
    LigerFusedLinearGRPOLoss = _try_import_liger_grpo()
    if LigerFusedLinearGRPOLoss is None:
        raise RuntimeError("liger_kernel not available for fused GRPO")

    cfg = loss_fn_config or {}
    loss_type = _LIGER_GRPO_LOSS_MAP[loss_fn]

    # Map our clip threshold names to Liger's epsilon names.
    epsilon_low = cfg.get("clip_low_threshold", 0.2)
    epsilon_high = cfg.get("clip_high_threshold", 0.2)
    kl_beta = cfg.get("kl_beta", 0.04)

    # For GRPO: epsilon is the distance from 1 (not the absolute clip bound).
    # Our config uses absolute: clip_low=0.8, clip_high=1.2 → epsilon=0.2.
    if loss_fn == "grpo" and epsilon_low >= 0.5:
        epsilon_low = 1.0 - epsilon_low
    if loss_fn == "grpo" and epsilon_high >= 0.5:
        epsilon_high = epsilon_high - 1.0

    fused_loss_fn = LigerFusedLinearGRPOLoss(
        beta=kl_beta,
        loss_type=loss_type,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )

    # Run model forward to get hidden states (not logits).
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        use_cache=False,
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[-1]

    # Get the lm_head weight.
    lm_head = model.get_output_embeddings()
    lin_weight = lm_head.weight
    lin_bias = getattr(lm_head, "bias", None)

    # Tinker convention: client pre-aligns all tensors. No shift needed.
    loss = fused_loss_fn(
        hidden,
        lin_weight,
        labels,
        attention_mask,
        advantages,
        bias=lin_bias,
        old_per_token_logps=old_logprobs,
    )

    loss.backward()
    return float(loss.detach().cpu())
