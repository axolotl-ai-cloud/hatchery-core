# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Runtime compatibility shims for PEFT.

These patch around version-skew bugs in third-party libraries that we can't
fix from the outside any other way. Each shim is idempotent and self-disabling
— it only patches when the bug is actually present, so it becomes a no-op once
the upstream fix is released and pulled in.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect

_TORCHAO_LORA_PATCHED = False


def ensure_peft_torchao_lora_compat() -> None:
    """Make peft's ``TorchaoLoraLinear`` tolerate a missing ``get_apply_tensor_subclass``.

    peft's ``TorchaoLoraLinear.__init__`` hard-requires the keyword-only argument
    ``get_apply_tensor_subclass``, which peft can only source from an HF quantizer's
    config (``model.hf_quantizer.quantization_config``). A base quantized *directly*
    via ``torchao.quantize_()`` — e.g. the optional FP8 requant path in
    :func:`hatchery.core.model_pool._maybe_requant_finegrained_fp8_to_torchao` — has no
    HF quantizer, so peft drops the kwarg and adapter injection dies with::

        TypeError: TorchaoLoraLinear.__init__() missing 1 required keyword-only
        argument: 'get_apply_tensor_subclass'

    That kwarg is consumed only by ``merge()`` / ``unmerge()`` (to re-quantize the
    merged weight) — never during training — so we default it to ``None``. Merging
    adapters into a manually-quantized torchao base still needs it and will raise
    there; training is unaffected.

    Idempotent and self-disabling: only patches when torchao is importable and the
    installed peft still hard-requires the kwarg. Becomes a no-op once the upstream
    fix (make ``get_apply_tensor_subclass`` optional) lands in peft.
    """
    global _TORCHAO_LORA_PATCHED
    if _TORCHAO_LORA_PATCHED:
        return
    # No torchao => no torchao-quantized base => nothing to patch.
    if importlib.util.find_spec("torchao") is None:
        return
    try:
        from peft.tuners.lora import torchao as _ptao
    except Exception:
        return

    cls = getattr(_ptao, "TorchaoLoraLinear", None)
    if cls is None:
        _TORCHAO_LORA_PATCHED = True
        return

    # Self-disable once upstream makes the kwarg optional (default present).
    gats = inspect.signature(cls.__init__).parameters.get("get_apply_tensor_subclass")
    if gats is None or gats.default is not inspect.Parameter.empty:
        _TORCHAO_LORA_PATCHED = True
        return

    _orig_init = cls.__init__

    @functools.wraps(_orig_init)
    def _init(self, *args, **kwargs):
        kwargs.setdefault("get_apply_tensor_subclass", None)
        _orig_init(self, *args, **kwargs)

    cls.__init__ = _init
    _TORCHAO_LORA_PATCHED = True
