# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""1-bit / 1.58-bit LLM detection and loader wiring.

Ternary-weight ("onebit") language models like the BitNet b1.58 family
store *master weights* in BF16 on disk and apply weight / activation
quantization inside the forward pass (straight-through estimator style),
so regular :class:`AutoModelForCausalLM.from_pretrained` can load them
today. What the training stack still needs is:

1. A way to *detect* that a base model belongs to a 1-bit family
   before we commit to dtype, attention backend, or LoRA plan.
2. A config surface where callers can opt in (or force in) a 1-bit
   loader path — useful when we one day want to bolt an explicit
   quantized-inference kernel on top of the same checkpoint.
3. A routing hook inside the model pool / trainer so the above stay
   out of the hot path for ordinary full-precision models.

Trainability note
-----------------
BitNet-family models are trained with quantization-aware training:
the published BF16 checkpoints are the *master weights* that flow
through a straight-through estimator at forward time. The faithful
fine-tuning path is full-parameter (FFT) on those master weights —
that's how the original training ran, and it's what the BitNet
authors recommend.

LoRA on top of a BitNet base is *possible* (base linears stay
frozen, LoRA deltas train normally) but the low-rank updates sit
*before* the forward-time quantizer, so the dynamics are muddled —
gradients from the STE-quantized forward flow into the low-rank
A/B matrices and the training signal is weaker than on a vanilla
base. Some community fine-tunes do work this way, but it's an
informed trade-off, not the default.

We therefore default to ``require_full_param=True`` for the
``"onebit"`` scheme: a LoRA session attaching to a detected
BitNet base is refused, and callers who know they want LoRA must
flip the flag explicitly.

Detection strategy
------------------
We look at the HF ``AutoConfig`` for the model:

* ``config.model_type == "bitnet"`` — the canonical signal used by
  ``transformers>=4.51``.
* ``architectures`` containing a class name starting with ``BitNet``
  (defensive: some checkpoints set this, some don't).
* ``quantization_config.quant_method == "bitnet"`` — for hypothetical
  post-training-quantized checkpoints.
* ``config._name_or_path`` containing a known BitNet slug, as a last
  resort for lovingly-hand-edited configs.

All checks are best-effort and pure; they don't touch CUDA or download
weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "QuantConfig",
    "QuantScheme",
    "detect_quant_scheme",
    "is_onebit_by_name",
    "is_onebit_model",
    "prepare_onebit_loader_kwargs",
    "resolve_quant_scheme",
]


# Known scheme tags. Kept as strings rather than an Enum so the
# dataclass stays trivially serializable alongside other configs.
QuantScheme = str  # "none" | "onebit" (1.58-bit ternary master-weight)
SCHEME_NONE: QuantScheme = "none"
SCHEME_ONEBIT: QuantScheme = "onebit"

_BITNET_ARCH_PREFIXES = ("BitNet",)
_BITNET_MODEL_TYPES = ("bitnet",)
# Slug substrings used as a last-resort heuristic when config is
# missing or mis-filled. Lowercase-compared.
_BITNET_NAME_HINTS = ("bitnet-b1.58", "bitnet_b1_58", "1bitllm/bitnet")


@dataclass
class QuantConfig:
    """Quantization knobs for base-model loading.

    Attributes
    ----------
    scheme:
        One of ``"none"`` (default — ordinary BF16/FP16 load) or
        ``"onebit"`` (1.58-bit BitNet master-weight load).
    force:
        If ``True``, callers assert the model is ``scheme`` regardless
        of auto-detection. Used for tests, or to override a
        misconfigured checkpoint. If ``False`` (default), the loader
        auto-detects and may upgrade ``"none"`` → ``"onebit"`` when
        the checkpoint is recognised as BitNet.
    require_full_param:
        If ``True`` and ``scheme == "onebit"``, LoRA attach is refused
        — BitNet's training recipe is full-parameter on the BF16
        master weights (see the module docstring). Defaults to
        ``True``; set to ``False`` explicitly if you've decided to
        train a LoRA adapter on top of a BitNet base despite the
        caveats.
    """

    scheme: QuantScheme = SCHEME_NONE
    force: bool = False
    require_full_param: bool = True

    def __post_init__(self) -> None:
        if self.scheme not in (SCHEME_NONE, SCHEME_ONEBIT):
            raise ValueError(
                f"QuantConfig.scheme must be one of 'none' / 'onebit', got {self.scheme!r}"
            )

    @property
    def is_onebit(self) -> bool:
        return self.scheme == SCHEME_ONEBIT


def _architectures_look_bitnet(architectures: Any) -> bool:
    if not architectures:
        return False
    try:
        for arch in architectures:
            if isinstance(arch, str) and arch.startswith(_BITNET_ARCH_PREFIXES):
                return True
    except TypeError:
        return False
    return False


def _quantization_config_is_bitnet(quantization_config: Any) -> bool:
    if quantization_config is None:
        return False
    # ``quantization_config`` on HF configs is usually a plain dict or
    # a ``QuantizationConfigMixin`` subclass. Accept both.
    quant_method: Any
    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method")
    else:
        quant_method = getattr(quantization_config, "quant_method", None)
    return isinstance(quant_method, str) and quant_method.lower() == "bitnet"


def _name_hint_is_bitnet(name: Optional[str]) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in _BITNET_NAME_HINTS)


def is_onebit_by_name(model_name: Optional[str]) -> bool:
    """Cheap pre-load check: does the slug *look like* a 1-bit model?

    This is the "free" half of :func:`is_onebit_model` — callable
    before we've paid for an HF ``AutoConfig`` round-trip. Use it to
    decide whether to do the more expensive config-based detection.
    """
    return _name_hint_is_bitnet(model_name)


def is_onebit_model(hf_config: Any, *, model_name: Optional[str] = None) -> bool:
    """Return ``True`` if the given HF config describes a 1-bit LLM.

    Parameters
    ----------
    hf_config:
        The object returned by ``transformers.AutoConfig.from_pretrained``.
        Any duck-typed object with ``model_type`` / ``architectures``
        attributes also works (unit tests pass a ``SimpleNamespace``).
    model_name:
        Optional repo name used as a last-resort hint — useful when
        the caller has the slug in hand but the config has been
        round-tripped through a tool that strips ``_name_or_path``.
    """
    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str) and model_type.lower() in _BITNET_MODEL_TYPES:
        return True
    if _architectures_look_bitnet(getattr(hf_config, "architectures", None)):
        return True
    if _quantization_config_is_bitnet(getattr(hf_config, "quantization_config", None)):
        return True
    if _name_hint_is_bitnet(model_name):
        return True
    return _name_hint_is_bitnet(getattr(hf_config, "_name_or_path", None))


def detect_quant_scheme(hf_config: Any, *, model_name: Optional[str] = None) -> QuantScheme:
    """Classify a model into one of the known schemes.

    Pure — no torch, no network, no filesystem work beyond whatever
    the caller already did to obtain ``hf_config``.
    """
    if is_onebit_model(hf_config, model_name=model_name):
        return SCHEME_ONEBIT
    return SCHEME_NONE


def resolve_quant_scheme(
    hf_config: Any,
    *,
    model_name: Optional[str] = None,
    requested: Optional[QuantConfig] = None,
) -> QuantScheme:
    """Combine a caller's :class:`QuantConfig` with auto-detection.

    Rules
    -----
    * ``requested.force`` short-circuits the whole resolver and returns
      ``requested.scheme`` verbatim — the caller is telling us they
      know better.
    * Otherwise, autodetect; if the caller asked for a specific scheme
      and autodetection agrees (or the caller asked for ``"none"`` and
      autodetect returns ``"onebit"``), the autodetect result wins —
      we'd rather silently upgrade than mis-load.
    """
    if requested is not None and requested.force:
        return requested.scheme
    auto = detect_quant_scheme(hf_config, model_name=model_name)
    if requested is None:
        return auto
    # Silent upgrade when we find a 1-bit model the caller didn't flag.
    if auto == SCHEME_ONEBIT:
        return SCHEME_ONEBIT
    return requested.scheme


def prepare_onebit_loader_kwargs(
    base_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Adjust ``from_pretrained`` kwargs for a 1-bit checkpoint.

    BitNet's master-weight checkpoints are BF16; they tolerate an
    explicit ``torch_dtype`` of bfloat16 but NOT a forced downcast to
    fp16 (the quantizer gets quirky near fp16's limited dynamic
    range). If the caller passed no dtype, leave it unset and let HF
    pick the checkpoint's native dtype. If the caller forced fp16,
    upgrade to bfloat16 with a silent tweak — documented here rather
    than warned, since this function is called from hot-path loaders.

    ``attn_implementation`` is left unchanged: HF's BitNet integration
    supports SDPA (the pool's default), so there's nothing to do.
    """
    kwargs = dict(base_kwargs)
    try:
        import torch
    except ImportError:
        return kwargs
    dtype = kwargs.get("torch_dtype")
    if dtype is torch.float16:
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs
