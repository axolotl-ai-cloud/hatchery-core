# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Runtime helpers for TorchAO tensor subclasses.

TorchAO Float8Tensor wrappers are Python tensor subclasses. Some runtime
environments call tensor ops during model movement or tensor packing that
TorchAO does not currently implement for every PyTorch version. This module
keeps Hatchery's small compatibility surface in one place:

* register missing Float8Tensor ops that preserve qdata/scale invariants;
* move module parameters and buffers without ``nn.Module._apply`` when that
  path would swap tensor subclass storage via unsupported ``aten.set_``.

The module is safe to import without TorchAO installed. Optional imports happen
inside the functions that need them.
"""

from __future__ import annotations

from typing import Any

import torch


def is_torchao_float8_tensor(tensor: Any) -> bool:
    """Return True for TorchAO Float8Tensor-like objects.

    We intentionally duck-type here so this helper works across TorchAO versions
    without importing TorchAO at module import time.
    """

    return (
        type(tensor).__name__ == "Float8Tensor"
        and hasattr(tensor, "qdata")
        and hasattr(tensor, "scale")
    )


def rebuild_float8_tensor(
    tensor: Any,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: Any,
) -> Any:
    """Reconstruct a TorchAO Float8Tensor while preserving wrapper metadata."""

    cls = tensor.__class__
    try:
        return cls(
            qdata,
            scale,
            tensor.block_size,
            tensor.mm_config,
            tensor.hp_value_lb,
            tensor.hp_value_ub,
            tensor.act_quant_kwargs,
            tensor.kernel_preference,
            dtype,
        )
    except AttributeError:
        return cls(
            qdata,
            scale,
            tensor.block_size,
            tensor.mm_config,
            tensor.act_quant_kwargs,
            tensor.kernel_preference,
            dtype,
        )


def move_tensor_without_apply(tensor: Any, device: str | torch.device) -> Any:
    """Move a tensor, rebuilding TorchAO Float8 wrappers instead of using _apply."""

    if is_torchao_float8_tensor(tensor):
        return rebuild_float8_tensor(
            tensor,
            tensor.qdata.to(device),
            tensor.scale.to(device),
            tensor.dtype,
        )
    return tensor.to(device)


def move_module_tensors_without_apply(model: Any, device: str | torch.device) -> None:
    """Move module parameters and buffers without calling ``nn.Module._apply``.

    This is useful in runtimes where ``Module.to(device)`` tries to swap tensor
    subclass storage with operators that TorchAO Float8Tensor does not
    implement. The function mutates ``model`` in place.
    """

    import torch.nn as nn

    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            moved = move_tensor_without_apply(parameter.detach(), device)
            module._parameters[name] = nn.Parameter(moved, requires_grad=parameter.requires_grad)
        for name, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            module._buffers[name] = move_tensor_without_apply(buffer, device)


def patch_torchao_float8_tensor_ops() -> bool:
    """Register Hatchery compatibility ops for TorchAO Float8Tensor.

    Returns ``True`` when TorchAO is available and the patch is installed or was
    already installed. Returns ``False`` when TorchAO or the required PyTorch
    dispatch helper is unavailable.
    """

    try:
        import inspect

        from torch.utils._python_dispatch import return_and_correct_aliasing
        from torchao.quantization import Float8Tensor
    except Exception:  # noqa: BLE001 - optional dependency / version surface
        return False

    if getattr(Float8Tensor, "_hatchery_float8_ops_patch", False):
        return True

    aten = torch.ops.aten
    implements = Float8Tensor.implements
    init_params = inspect.signature(Float8Tensor).parameters

    def _new_float8_tensor(self: Any, qdata: torch.Tensor, scale: torch.Tensor, out_dtype: Any):
        if "hp_value_lb" in init_params:
            return self.__class__(
                qdata,
                scale,
                self.block_size,
                self.mm_config,
                self.hp_value_lb,
                self.hp_value_ub,
                self.act_quant_kwargs,
                self.kernel_preference,
                out_dtype,
            )
        return self.__class__(
            qdata,
            scale,
            self.block_size,
            self.mm_config,
            self.act_quant_kwargs,
            self.kernel_preference,
            out_dtype,
        )

    @implements(aten.empty_like.default)
    def _(func, types, args, kwargs):
        original_kwargs = dict(kwargs or {})
        self = args[0]
        out_dtype = original_kwargs.pop("dtype", None) or self.dtype
        qdata = aten.empty_like.default(self.qdata, **original_kwargs)
        scale_kwargs = {k: v for k, v in original_kwargs.items() if k != "dtype"}
        scale = aten.empty_like.default(self.scale, **scale_kwargs)
        new = _new_float8_tensor(self, qdata, scale, out_dtype)
        return return_and_correct_aliasing(func, args, kwargs, new)

    @implements(aten._to_copy.default)
    def _(func, types, args, kwargs):
        kwargs = dict(kwargs or {})
        self = args[0]
        out_dtype = kwargs.get("dtype") or self.dtype
        inner_kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        qdata = aten._to_copy.default(self.qdata, **inner_kwargs)
        scale = aten._to_copy.default(self.scale, **inner_kwargs)
        new = _new_float8_tensor(self, qdata, scale, out_dtype)
        return return_and_correct_aliasing(func, args, kwargs, new)

    @implements([aten.detach.default, aten.clone.default])
    def _(func, types, args, kwargs):
        kwargs = dict(kwargs or {})
        self = args[0]
        new = _new_float8_tensor(
            self,
            func(self.qdata, **kwargs),
            func(self.scale, **kwargs),
            self.dtype,
        )
        return return_and_correct_aliasing(func, args, kwargs, new)

    Float8Tensor._hatchery_float8_ops_patch = True
    return True
