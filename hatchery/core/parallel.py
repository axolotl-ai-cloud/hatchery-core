# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Parallelism configuration dataclasses.

These are the config types that flow through the core package (worker,
trainer, batching). They don't import torch — they're pure dataclasses
with env-var parsing.

The actual distributed setup (FSDP2 wrapping, tensor parallelism,
context parallelism, device mesh construction) lives in an extension
package that layers the torch machinery on top of these configs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from hatchery.core.quantization import QuantConfig


@dataclass
class OffloadConfig:
    """Memory-vs-speed knobs for CPU/NVMe offloading."""

    cpu_offload_params: bool = False
    cpu_offload_optimizer: bool = False
    activation_checkpointing: bool = False
    nvme_path: Optional[str] = None


@dataclass
class ParallelConfig:
    """Parallelism + offloading knobs.

    A single-GPU worker passes ``ParallelConfig()`` — all degrees are 1
    and the non-distributed code path is used.

    ``quant`` selects a base-model quantization scheme. The default
    (``QuantConfig()`` with ``scheme="none"``) is a no-op; setting
    ``scheme="onebit"`` routes the loader through the BitNet /
    1.58-bit path (see :mod:`hatchery.core.quantization`).
    """

    dp_degree: int = 1
    tp_degree: int = 1
    cp_degree: int = 1
    sequence_parallel: bool = False
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    batch_strategy: str = "auto"
    # Sequence packing (varlen v1): concatenate short examples into one
    # long sequence with position_ids resets at doc boundaries. HF's
    # flash-attn-2 path derives cu_seqlens from the resets.
    sequence_packing: bool = False
    max_packed_len: Optional[int] = None
    quant: QuantConfig = field(default_factory=QuantConfig)

    def world_size(self) -> int:
        return self.dp_degree * self.tp_degree * self.cp_degree

    def is_distributed(self) -> bool:
        return self.world_size() > 1

    def __post_init__(self) -> None:
        for field_name, value in (
            ("dp_degree", self.dp_degree),
            ("tp_degree", self.tp_degree),
            ("cp_degree", self.cp_degree),
        ):
            if value < 1:
                raise ValueError(f"{field_name} must be >= 1, got {value}")
        if self.sequence_parallel and self.tp_degree == 1:
            raise ValueError("sequence_parallel only makes sense with tp_degree > 1")
        if self.sequence_packing and self.cp_degree > 1:
            # Packed sequences and context parallel both chop up the
            # sequence axis; reconciling them is out of scope for v1.
            raise ValueError("sequence_packing with cp_degree > 1 is not supported in v1")
        if self.max_packed_len is not None and self.max_packed_len <= 0:
            raise ValueError(f"max_packed_len must be > 0, got {self.max_packed_len}")

    @classmethod
    def from_env(cls) -> ParallelConfig:
        """Build a config from environment variables set by torchrun.

        Quantization env vars
        ---------------------
        ``HATCHERY_QUANT_SCHEME`` — ``"none"`` (default), ``"onebit"``,
        or ``"fp8_torchao"``.  Unknown values degrade silently to
        ``"none"`` so a typo doesn't prevent the worker from booting.

        ``HATCHERY_FP8_MODE`` — ``"weight_only"`` (default) or
        ``"dynamic"``.  Only meaningful when
        ``HATCHERY_QUANT_SCHEME=fp8_torchao``.  ``"dynamic"`` requires
        CUDA compute capability >= 9.0 (H100 / H200 / Blackwell).
        Unknown values degrade to ``"weight_only"``.
        """
        raw_max_packed = os.environ.get("HATCHERY_MAX_PACKED_LEN")
        max_packed_len = int(raw_max_packed) if raw_max_packed else None
        _valid_schemes = ("none", "onebit", "fp8_torchao")
        quant_scheme = os.environ.get("HATCHERY_QUANT_SCHEME", "none").lower()
        raw_fp8_mode = os.environ.get("HATCHERY_FP8_MODE", "weight_only").lower()
        quant = QuantConfig(
            scheme=quant_scheme if quant_scheme in _valid_schemes else "none",
            force=os.environ.get("HATCHERY_QUANT_FORCE", "0") == "1",
            require_full_param=os.environ.get("HATCHERY_QUANT_REQUIRE_FULL_PARAM", "1") == "1",
            fp8_mode=raw_fp8_mode if raw_fp8_mode in ("weight_only", "dynamic") else "weight_only",
        )
        return cls(
            dp_degree=int(os.environ.get("HATCHERY_DP_DEGREE", "1")),
            tp_degree=int(os.environ.get("HATCHERY_TP_DEGREE", "1")),
            cp_degree=int(os.environ.get("HATCHERY_CP_DEGREE", "1")),
            sequence_parallel=os.environ.get("HATCHERY_SP", "0") == "1",
            offload=OffloadConfig(
                cpu_offload_params=os.environ.get("HATCHERY_CPU_OFFLOAD_PARAMS", "0") == "1",
                cpu_offload_optimizer=os.environ.get("HATCHERY_CPU_OFFLOAD_OPTIMIZER", "0") == "1",
                activation_checkpointing=os.environ.get("HATCHERY_ACTIVATION_CKPT", "0") == "1",
                nvme_path=os.environ.get("HATCHERY_NVME_OFFLOAD_PATH") or None,
            ),
            batch_strategy=os.environ.get("HATCHERY_BATCH_STRATEGY", "auto"),
            sequence_packing=os.environ.get("HATCHERY_SEQUENCE_PACKING", "0") == "1",
            max_packed_len=max_packed_len,
            quant=quant,
        )
