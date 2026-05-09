# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import pytest
import torch
from torch import nn

from hatchery.core.torchao_runtime import (
    is_torchao_float8_tensor,
    move_module_tensors_without_apply,
    patch_torchao_float8_tensor_ops,
)


def test_move_module_tensors_without_apply_moves_plain_parameters_and_buffers():
    module = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    move_module_tensors_without_apply(module, "cpu")

    assert {str(parameter.device) for parameter in module.parameters()} == {"cpu"}
    assert {str(buffer.device) for buffer in module.buffers()} == {"cpu"}
    assert module[0].weight.requires_grad is True


def test_patch_torchao_float8_tensor_ops_is_optional():
    try:
        import torchao  # noqa: F401
    except ImportError:
        assert patch_torchao_float8_tensor_ops() is False
    else:
        assert patch_torchao_float8_tensor_ops() is True
        assert patch_torchao_float8_tensor_ops() is True


def test_torchao_float8_patch_and_manual_module_move():
    torchao = pytest.importorskip("torchao")
    assert torchao is not None
    from torchao.quantization import quantize_
    from torchao.quantization.quant_api import Float8WeightOnlyConfig

    class TinyLoraLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = nn.Linear(16, 16, bias=False)
            self.adapter = nn.Linear(16, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.adapter(x)

    assert patch_torchao_float8_tensor_ops() is True

    module = TinyLoraLike().bfloat16()
    quantize_(module.base, Float8WeightOnlyConfig())
    for parameter in module.base.parameters():
        parameter.requires_grad_(False)

    base_weight = next(module.base.parameters()).detach()
    assert is_torchao_float8_tensor(base_weight)
    assert is_torchao_float8_tensor(torch.empty_like(base_weight, pin_memory=False))
    assert is_torchao_float8_tensor(base_weight.clone())
    assert is_torchao_float8_tensor(base_weight.to("cpu"))

    target = "cuda" if torch.cuda.is_available() else "cpu"
    move_module_tensors_without_apply(module, target)
    assert {str(tensor.device) for tensor in list(module.parameters()) + list(module.buffers())} == {
        str(torch.device(target))
    }

    if torch.cuda.is_available():
        y = module(torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)).sum()
        y.backward()
        assert all(parameter.grad is not None for parameter in module.adapter.parameters())
