# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Lazy ScatterMoE-LoRA kernel integration for Hatchery workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

SUPPORTED_MODEL_TYPES = {
    "qwen2_moe",
    "qwen3_moe",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_next",
    "qwen3_vl_moe",
    "qwen3_omni_moe",
    "olmoe",
    "mixtral",
    "minimax",
    "glm_moe_dsa",
    "deepseek_v3",
    "glm4_moe",
    "glm4v_moe",
    "minimax_m2",
    "gemma4_text",
}

SUPPORTED_BASE_MODEL_PREFIXES = ("Qwen/Qwen3.6-35B-A3B",)


@dataclass(frozen=True)
class ScatterMoEKernelConfig:
    """ScatterMoE-LoRA kernel configuration."""

    enabled: bool = False
    kernel_ref: str = "axolotl-ai-co/scattermoe-lora"
    strict: bool = False


@dataclass(frozen=True)
class ScatterMoEKernelReport:
    """Structured result for a ScatterMoE kernel application attempt."""

    status: str
    kernel_ref: str
    applied: bool
    compatible: bool
    base_model_name: str
    model_type: Optional[str] = None
    reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kernel_ref": self.kernel_ref,
            "applied": self.applied,
            "compatible": self.compatible,
            "base_model_name": self.base_model_name,
            "model_type": self.model_type,
            "reason": self.reason,
        }


def _try_import_kernels() -> Any:
    try:
        from kernels import Mode, get_kernel, kernelize

        return {
            "get_kernel": get_kernel,
            "kernelize": kernelize,
            "Mode": Mode,
        }
    except ImportError:
        return None


def _extract_model_type(model: Any) -> Optional[str]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str) and model_type:
        return model_type
    return None


def _is_scattermoe_compatible(model: Any, *, base_model_name: str) -> bool:
    if base_model_name in SUPPORTED_BASE_MODEL_PREFIXES:
        return True
    if any(base_model_name.startswith(prefix) for prefix in SUPPORTED_BASE_MODEL_PREFIXES):
        return True

    model_type = _extract_model_type(model)
    if model_type is None:
        return False
    return model_type.lower() in SUPPORTED_MODEL_TYPES


def apply_scattermoe_kernel(
    model: Any,
    *,
    base_model_name: str,
    config: ScatterMoEKernelConfig,
    lora_config: Any = None,  # noqa: ARG001 - signature matches Config hook shape.
) -> ScatterMoEKernelReport:
    """Try to apply the ScatterMoE-LoRA HF kernel to ``model``."""

    model_type = _extract_model_type(model)
    compatible = _is_scattermoe_compatible(model, base_model_name=base_model_name)
    if not config.enabled:
        return ScatterMoEKernelReport(
            status="disabled",
            kernel_ref=config.kernel_ref,
            applied=False,
            compatible=compatible,
            base_model_name=base_model_name,
            model_type=model_type,
            reason="scattermoe_disabled",
        )
    if not compatible:
        return ScatterMoEKernelReport(
            status="incompatible",
            kernel_ref=config.kernel_ref,
            applied=False,
            compatible=False,
            base_model_name=base_model_name,
            model_type=model_type,
            reason="model_incompatible",
        )

    kernels_mod = _try_import_kernels()
    if kernels_mod is None:
        if config.strict:
            raise ImportError(
                "ScatterMoE-LoRA kernel was enabled but the `kernels` package is not installed"
            )
        return ScatterMoEKernelReport(
            status="unavailable",
            kernel_ref=config.kernel_ref,
            applied=False,
            compatible=True,
            base_model_name=base_model_name,
            model_type=model_type,
            reason="kernels_not_installed",
        )

    get_kernel = kernels_mod["get_kernel"]
    kernelize = kernels_mod["kernelize"]
    mode = kernels_mod["Mode"]

    try:
        get_kernel(config.kernel_ref)
        kernelize(model, mode=mode.TRAINING)
    except Exception as exc:  # noqa: BLE001
        if config.strict:
            raise
        return ScatterMoEKernelReport(
            status="fallback",
            kernel_ref=config.kernel_ref,
            applied=False,
            compatible=True,
            base_model_name=base_model_name,
            model_type=model_type,
            reason=f"{type(exc).__name__}:{exc}",
        )

    return ScatterMoEKernelReport(
        status="applied",
        kernel_ref=config.kernel_ref,
        applied=True,
        compatible=True,
        base_model_name=base_model_name,
        model_type=model_type,
    )
