# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Optimizer dispatcher — picks AdamW vs 8-bit AdamW based on size budget.

LoRA sessions always use plain ``torch.optim.AdamW`` because the
trainable param count is tiny (a few MB of state). Full-parameter
sessions on larger models (e.g. 1.5B+) would otherwise blow the VRAM
budget on fp32 optimizer state alone — 8 bytes per param ≈ 12 GB
for a 1.5B model. Switching to **torchao**'s ``AdamW8bit`` shrinks
that ~4x with negligible quality impact for short fine-tunes.

Why torchao over bitsandbytes? bnb's 8-bit optimizer kernels expect
plain CUDA tensors and don't handle DTensor — they crash with
illegal-memory-access under FSDP2 (bnb #1633, #89). torchao is
explicitly DTensor-aware in ``_new_buffer`` and is the path the
PyTorch + HF stacks recommend for FSDP2.

The selector intentionally errs on the side of plain AdamW when the
budget is comfortable; 8-bit kicks in only when fp32 state would
exceed ``vram_budget_frac`` of the device's free memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

import torch

logger = logging.getLogger("hatchery.core.optim_dispatch")

OptimizerKind = Literal["adamw", "adamw_8bit"]

# fp32 AdamW carries m + v at 4 bytes each = 8 bytes per trainable param.
_FP32_ADAMW_BYTES_PER_PARAM = 8


def select_optimizer_kind(
    *,
    training_mode: str,
    trainable_param_count: int,
    vram_free_bytes: int,
    vram_budget_frac: float = 0.40,
) -> OptimizerKind:
    """Pick an optimizer kind for this session.

    LoRA always returns ``"adamw"``. FP returns ``"adamw_8bit"`` when
    fp32 optimizer state would exceed ``vram_budget_frac`` of free VRAM,
    otherwise ``"adamw"``.
    """
    if training_mode != "full_param":
        return "adamw"
    if trainable_param_count <= 0 or vram_free_bytes <= 0:
        return "adamw"
    fp32_state_bytes = trainable_param_count * _FP32_ADAMW_BYTES_PER_PARAM
    budget_bytes = int(vram_free_bytes * vram_budget_frac)
    if fp32_state_bytes > budget_bytes:
        return "adamw_8bit"
    return "adamw"


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    kind: OptimizerKind,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    fused: bool = False,
) -> torch.optim.Optimizer:
    """Construct an optimizer of the given kind.

    Falls back to ``torch.optim.AdamW`` with a warning if torchao is
    requested but unavailable — keeps the worker functional on
    devices/images without torchao installed.
    """
    params_list = list(params)
    if kind == "adamw_8bit":
        try:
            from torchao.optim import AdamW8bit  # noqa: PLC0415
        except ImportError:
            logger.warning("torchao not installed; falling back to fp32 AdamW (kind=adamw_8bit)")
        else:
            return AdamW8bit(
                params_list,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )

    return torch.optim.AdamW(
        params_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=fused,
    )


def vram_free_bytes(device: str) -> int:
    """Best-effort free-VRAM probe for the active CUDA device."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:  # noqa: BLE001
        return 0
