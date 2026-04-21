# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Selective fp32 upcasting for precision-sensitive submodules.

Problem
-------
Loading a modern LLM in bf16 is cheap on memory but breaks a few
specific submodules that need higher precision to stay numerically
stable:

* **MoE routers / gate networks.** The top-k softmax over expert
  logits is scale-sensitive: in bf16, the ~1e-3 logit spread between
  experts can collapse under rounding, producing load-balancing drift
  and occasional NaNs in the softmax normalizer. Megatron-LM and the
  Mixtral paper both prescribe fp32 routers.
* **Embeddings** (softer preference). Some training recipes keep the
  embedding table in fp32 — bf16 rounding at embedding lookup is a
  source of gradient bias. For LoRA we don't train the base embedding
  so the forward-pass rounding is the only concern, usually tolerable.

Policy source of truth
----------------------
Per-architecture rules live in
``hatchery.core.fused_losses.ModelCapability`` — ``fp32_module_suffixes``
for precise name-suffix matches (routers, correction biases, etc.)
and ``fp32_embeddings`` as a blanket toggle for ``nn.Embedding``
modules. This keeps the capability table as the single source of
truth; this module just *applies* what the table describes.

Cast + hook
-----------
Upcasting the weight alone isn't enough because ``nn.Linear`` expects
matching dtypes on input and weight. For each promoted module we:

1. ``module.to(torch.float32)`` — move its own parameters + buffers.
2. Register a forward pre-hook that upcasts every floating-point
   positional + keyword arg to fp32 before the module sees it.
3. Optionally register a forward post-hook that downcasts the result
   back to the caller's original dtype. Most HF MoE forwards do this
   themselves (Mixtral casts ``routing_weights`` back to
   ``hidden_states.dtype`` after the softmax), so by default we keep
   the post-hook OFF and let the containing code decide when to
   downcast. Turn it on only for architectures where the containing
   MoE block assumes the router output is already in the main dtype.

What we do NOT do
-----------------
* We don't touch layer norms / RMS norms. HF's
  ``LlamaRMSNorm.forward`` already upcasts internally (it does
  ``x.to(torch.float32)`` before the normalization), and the
  *weight* being in bf16 is fine because it's applied as a final
  multiply where rounding is benign.
* We don't touch the LM head — it's handled by the fused CE path
  and doesn't need per-weight upcasting.
* We don't touch the attention softmax — that already runs in fp32
  via ``F.scaled_dot_product_attention`` or the model-specific
  attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

from hatchery.core.fused_losses import ModelCapability, get_model_capability


@dataclass
class PrecisionApplyReport:
    """What :func:`apply_precision_policy` actually did. Surfaced so
    the worker can log it once per model load and so tests can assert
    on specific submodules.
    """

    upcast_modules: list[str]
    upcast_embeddings: list[str]
    skipped_reason: str = ""

    @property
    def total(self) -> int:
        return len(self.upcast_modules) + len(self.upcast_embeddings)


def apply_precision_policy(
    model: Any,
    *,
    capability: ModelCapability | None = None,
    main_dtype: Any = None,
) -> PrecisionApplyReport:
    """Walk ``model`` and upcast precision-sensitive submodules to fp32.

    Parameters
    ----------
    model:
        The raw transformers model (NOT the PEFT wrapper — call this
        before ``get_peft_model`` so the adapter sees the correct
        dtypes). Passing a PEFT-wrapped model also works; we look up
        the capability via the same fallback path.
    capability:
        Explicit capability override. If omitted, we look it up via
        ``get_model_capability(model)`` using the underlying model
        class name.
    main_dtype:
        The model's dominant dtype (typically bf16). Used by the
        forward pre-hook to know what to downcast back to on the way
        out of promoted modules; pass ``None`` to auto-detect from
        the first floating-point parameter we find outside the
        promoted set.

    Returns
    -------
    A :class:`PrecisionApplyReport` listing the qualified names of
    every module we touched. An empty report means the policy is
    a no-op for this architecture — safe, just not helpful.
    """
    if capability is None:
        capability = get_model_capability(model)

    suffixes = tuple(capability.fp32_module_suffixes)
    want_embeddings = bool(capability.fp32_embeddings)

    if not suffixes and not want_embeddings:
        return PrecisionApplyReport(
            upcast_modules=[],
            upcast_embeddings=[],
            skipped_reason="capability declares no fp32 modules",
        )

    # Auto-detect main dtype from a non-matching floating parameter.
    if main_dtype is None:
        main_dtype = _detect_main_dtype(model)

    upcast_modules: list[str] = []
    upcast_embeddings: list[str] = []

    for qname, module in model.named_modules():
        if not qname:
            # Skip the root module itself.
            continue
        if suffixes and any(qname.endswith(s) for s in suffixes):
            _promote_module(module, main_dtype=main_dtype)
            upcast_modules.append(qname)
            continue
        if want_embeddings and isinstance(module, nn.Embedding):
            _promote_module(module, main_dtype=main_dtype)
            upcast_embeddings.append(qname)

    return PrecisionApplyReport(
        upcast_modules=upcast_modules,
        upcast_embeddings=upcast_embeddings,
    )


def _detect_main_dtype(model: Any) -> Any:
    """Best-effort scan of the model for its dominant floating dtype.

    We look for the first parameter that's a floating tensor and use
    its dtype as ``main_dtype``. For a typical bf16 model this returns
    ``torch.bfloat16`` on the first layer's query projection.
    """
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.bfloat16


def _promote_module(module: Any, *, main_dtype: Any) -> None:
    """Cast ``module`` to fp32 and install the dtype-safety pre-hook.

    The pre-hook upcasts every floating-point input to fp32 so the
    module's ``F.linear`` / ``F.embedding`` calls don't blow up on a
    dtype mismatch. We don't install a post-hook because the common
    MoE code paths (Mixtral, Qwen-MoE) already downcast the router
    output inside the containing block — adding our own post-hook
    would double-cast.

    The hook is registered with ``with_kwargs=True`` so it sees both
    positional and keyword arguments; some HF forwards (e.g., Gemma's
    router) pass the hidden state as a kwarg.
    """
    module.to(torch.float32)

    def _upcast_pre_hook(_mod: Any, args: tuple, kwargs: dict):
        new_args = tuple(a.to(torch.float32) if _is_floating_tensor(a) else a for a in args)
        new_kwargs = {
            k: (v.to(torch.float32) if _is_floating_tensor(v) else v) for k, v in kwargs.items()
        }
        return new_args, new_kwargs

    # Tag the hook so tests (and subsequent idempotent calls) can
    # detect whether a module has already been promoted. Prevents
    # accidental double-registration if apply_precision_policy is
    # called twice on the same model.
    if getattr(module, "_tinker_fp32_hook_registered", False):
        return
    module.register_forward_pre_hook(_upcast_pre_hook, with_kwargs=True)
    module._tinker_fp32_hook_registered = True  # type: ignore[attr-defined]
    # Store the "outside" dtype so downstream code (e.g., tests) can
    # introspect what the boundary looked like at promotion time.
    module._tinker_main_dtype = main_dtype  # type: ignore[attr-defined]


def _is_floating_tensor(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor) and x.is_floating_point()
