# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Base-model quantization detection and loader wiring.

Supported schemes
-----------------
* ``"none"`` — ordinary BF16/FP16 load (default).
* ``"onebit"`` — 1.58-bit ternary BitNet master-weight load.
* ``"fp8_torchao"`` — FP8 weight quantization via TorchAO; autograd-
  compatible for training and PEFT LoRA.

1-bit / BitNet (``"onebit"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ternary-weight models like BitNet b1.58 store *master weights* in BF16
on disk and apply weight/activation quantization inside the forward pass
(straight-through estimator style).  The faithful fine-tuning path is
full-parameter (FFT) on those master weights.  LoRA is possible but the
deltas sit before the STE quantizer, weakening the training signal.  We
therefore default to ``require_full_param=True`` for ``"onebit"``.

Detection: ``config.model_type == "bitnet"``, ``architectures`` starting
with ``"BitNet"``, ``quantization_config.quant_method == "bitnet"``, or
a model-name slug hint.

FP8 via TorchAO (``"fp8_torchao"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TorchAO float8 kernels are autograd-compatible — gradients flow through
FP8 linear layers during forward *and* backward.  This is the correct
path for FP8 training; do **not** use the Transformers
``finegrained_fp8`` / ``FP8Linear`` kernels, which are inference-only.

PEFT LoRA is fully supported on TorchAO-quantized models (``require_full_param``
does **not** apply to ``"fp8_torchao"``).

Two sub-modes controlled by :attr:`QuantConfig.fp8_mode`:

* ``"weight_only"`` (default) — stores weights in FP8; activations stay
  in BF16.  Works on all FP8-capable hardware (Ada Lovelace, Hopper,
  Blackwell).  Requires ``torchao >= 0.4`` and ``transformers >= 4.46``.
* ``"dynamic"`` — quantizes weights **and** activations dynamically per
  token.  Faster on large batches but requires CUDA compute capability
  >= 9.0 (NVIDIA H100 / H200 / Blackwell).

Detection from an existing FP8 TorchAO checkpoint: ``quantization_config``
dict/object with ``quant_type == "torchao"`` and a ``torchao_config``
string that contains ``"float8"`` or ``"Float8"``.

BF16 fallback
~~~~~~~~~~~~~
BF16 is **not** the default FP8 fallback.  Set
``HATCHERY_QUANT_SCHEME=none`` explicitly if you want a plain BF16 load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "QuantConfig",
    "QuantScheme",
    "detect_quant_scheme",
    "is_fp8_torchao_model",
    "is_onebit_by_name",
    "is_onebit_model",
    "prepare_fp8_torchao_loader_kwargs",
    "prepare_onebit_loader_kwargs",
    "resolve_quant_scheme",
]


# Known scheme tags. Kept as strings rather than an Enum so the
# dataclass stays trivially serializable alongside other configs.
QuantScheme = str  # "none" | "onebit" | "fp8_torchao"
SCHEME_NONE: QuantScheme = "none"
SCHEME_ONEBIT: QuantScheme = "onebit"
SCHEME_FP8_TORCHAO: QuantScheme = "fp8_torchao"

_VALID_SCHEMES = (SCHEME_NONE, SCHEME_ONEBIT, SCHEME_FP8_TORCHAO)

_BITNET_ARCH_PREFIXES = ("BitNet",)
_BITNET_MODEL_TYPES = ("bitnet",)
# Slug substrings used as a last-resort heuristic when config is
# missing or mis-filled. Lowercase-compared.
_BITNET_NAME_HINTS = ("bitnet-b1.58", "bitnet_b1_58", "1bitllm/bitnet")

# Substrings that identify a TorchAO FP8 config within the serialised
# torchao_config field (either the class repr or the short string alias).
_TORCHAO_FP8_HINTS = ("float8", "Float8")


@dataclass
class QuantConfig:
    """Quantization knobs for base-model loading.

    Attributes
    ----------
    scheme:
        One of ``"none"`` (default — ordinary BF16/FP16 load),
        ``"onebit"`` (1.58-bit BitNet master-weight load), or
        ``"fp8_torchao"`` (FP8 via TorchAO — training-compatible).
    force:
        If ``True``, callers assert the model is ``scheme`` regardless
        of auto-detection. If ``False`` (default), the loader
        auto-detects and may upgrade when the checkpoint is recognised.
    require_full_param:
        If ``True`` and ``scheme == "onebit"``, LoRA attach is refused.
        Irrelevant for ``"fp8_torchao"`` — TorchAO FP8 fully supports
        PEFT LoRA via autograd-compatible forward/backward.
    fp8_mode:
        Sub-mode for ``scheme == "fp8_torchao"``.  ``"weight_only"``
        (default) stores weights in FP8, activations in BF16.
        ``"dynamic"`` quantizes both weights and activations per-token;
        requires CUDA compute capability >= 9.0 (H100/H200/Blackwell).
        Ignored for other schemes.
    """

    scheme: QuantScheme = SCHEME_NONE
    force: bool = False
    require_full_param: bool = True
    fp8_mode: str = "weight_only"

    def __post_init__(self) -> None:
        if self.scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"QuantConfig.scheme must be one of {_VALID_SCHEMES!r}, got {self.scheme!r}"
            )
        if self.fp8_mode not in ("weight_only", "dynamic"):
            raise ValueError(
                f"QuantConfig.fp8_mode must be 'weight_only' or 'dynamic', got {self.fp8_mode!r}"
            )

    @property
    def is_onebit(self) -> bool:
        return self.scheme == SCHEME_ONEBIT

    @property
    def is_fp8_torchao(self) -> bool:
        return self.scheme == SCHEME_FP8_TORCHAO


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


def _quantization_config_is_fp8_torchao(quantization_config: Any) -> bool:
    """Return True if quantization_config describes a TorchAO FP8 setup.

    Accepts both the dict form (serialised config.json) and a live
    HF ``QuantizationConfigMixin`` object.
    """
    if quantization_config is None:
        return False
    if isinstance(quantization_config, dict):
        quant_type = quantization_config.get("quant_type", "")
        torchao_str = str(quantization_config.get("torchao_config", ""))
    else:
        quant_type = getattr(quantization_config, "quant_type", "")
        torchao_str = str(getattr(quantization_config, "torchao_config", ""))
    if not isinstance(quant_type, str) or quant_type.lower() != "torchao":
        return False
    return any(hint in torchao_str for hint in _TORCHAO_FP8_HINTS)


def is_fp8_torchao_model(hf_config: Any) -> bool:
    """Return ``True`` if the HF config describes a TorchAO FP8 model.

    Detects checkpoints that were already saved with a TorchAO FP8
    ``quantization_config``.  When the caller is *applying* FP8 at
    load time via :func:`prepare_fp8_torchao_loader_kwargs`, this will
    be ``False`` on the raw checkpoint config (pre-load) and ``True``
    on the loaded model config (post-quantization).
    """
    return _quantization_config_is_fp8_torchao(getattr(hf_config, "quantization_config", None))


def detect_quant_scheme(hf_config: Any, *, model_name: Optional[str] = None) -> QuantScheme:
    """Classify a model into one of the known schemes.

    Pure — no torch, no network, no filesystem work beyond whatever
    the caller already did to obtain ``hf_config``.

    Ordering: FP8 TorchAO is checked before 1-bit because an explicit
    ``quantization_config.quant_type == "torchao"`` in the checkpoint is
    a more specific signal than the ``model_type``/slug heuristics used
    for BitNet detection.  A BitNet model saved with a TorchAO FP8
    ``quantization_config`` is classified as ``"fp8_torchao"``.
    """
    if is_fp8_torchao_model(hf_config):
        return SCHEME_FP8_TORCHAO
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
    * Otherwise, autodetect; if the checkpoint already has a recognised
      quantization scheme in its config, autodetection wins — we'd
      rather silently upgrade than mis-load.
    """
    if requested is not None and requested.force:
        return requested.scheme
    auto = detect_quant_scheme(hf_config, model_name=model_name)
    if requested is None:
        return auto
    # Silent upgrade when we detect a known scheme the caller didn't flag.
    if auto in (SCHEME_ONEBIT, SCHEME_FP8_TORCHAO):
        return auto
    return requested.scheme


def prepare_fp8_torchao_loader_kwargs(
    base_kwargs: dict[str, Any],
    *,
    fp8_mode: str = "weight_only",
) -> dict[str, Any]:
    """Inject a TorchAO FP8 quantization config into ``from_pretrained`` kwargs.

    This is the training-compatible FP8 path.  It sets
    ``quantization_config`` to a :class:`transformers.TorchAoConfig`
    wrapping either :class:`~torchao.quantization.Float8WeightOnlyConfig`
    or :class:`~torchao.quantization.Float8DynamicActivationFloat8WeightConfig`.

    **Do NOT use** the Transformers ``finegrained_fp8`` / ``FP8Linear``
    path for training — those kernels are inference-only and lack
    autograd support.

    Hardware / dependency caveats
    ------------------------------
    * Requires ``torchao >= 0.4`` and ``transformers >= 4.46``.
    * ``fp8_mode="weight_only"`` — stores weights in FP8; activations
      remain in BF16.  Compatible with all FP8-capable hardware
      (NVIDIA Ada Lovelace L40, Hopper H100/H200, Blackwell).
    * ``fp8_mode="dynamic"`` — dynamically quantizes weights *and*
      activations to FP8 per-token.  Faster on large batches but
      requires CUDA compute capability >= 9.0 (H100 / H200 / Blackwell
      only).  Will raise ``RuntimeError`` on older hardware at runtime.

    LoRA note
    ---------
    TorchAO FP8 models support PEFT LoRA — gradients flow through FP8
    linear layers via the autograd-compatible torchao float8 kernels.
    The ``require_full_param`` guard in the trainer does **not** apply
    to ``"fp8_torchao"`` sessions.

    BF16 fallback
    -------------
    BF16 is **not** the default fallback.  If you want a plain BF16
    load, set ``HATCHERY_QUANT_SCHEME=none`` explicitly.

    Parameters
    ----------
    base_kwargs:
        Starting keyword-argument dict for ``AutoModelForCausalLM.from_pretrained``.
        Returned dict is a copy with ``quantization_config`` added.
    fp8_mode:
        ``"weight_only"`` or ``"dynamic"`` (see above).
    """
    try:
        from transformers import TorchAoConfig
    except ImportError:
        raise RuntimeError(
            "transformers.TorchAoConfig not found. "
            "Upgrade to transformers >= 4.46 to enable TorchAO quantization."
        ) from None

    if fp8_mode == "dynamic":
        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

            torchao_cfg: Any = Float8DynamicActivationFloat8WeightConfig()
        except ImportError:
            raise RuntimeError(
                "torchao.quantization.Float8DynamicActivationFloat8WeightConfig not found. "
                "Install torchao >= 0.4. "
                "fp8_mode='dynamic' also requires CUDA compute capability >= 9.0 "
                "(NVIDIA H100 / H200 / Blackwell)."
            ) from None
    else:
        try:
            from torchao.quantization import Float8WeightOnlyConfig

            torchao_cfg = Float8WeightOnlyConfig()
        except ImportError:
            raise RuntimeError(
                "torchao.quantization.Float8WeightOnlyConfig not found. "
                "Install torchao >= 0.4 for FP8 weight-only quantization. "
                "Runtime also requires FP8-capable hardware (NVIDIA Ada Lovelace "
                "L40, Hopper H100/H200, or Blackwell)."
            ) from None

    kwargs = dict(base_kwargs)
    kwargs["quantization_config"] = TorchAoConfig(torchao_cfg)
    return kwargs


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
