# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Lazy ScatterMoE-LoRA kernel integration for Hatchery workers.

Mirrors the kernel-registration logic in axolotl's
``src/axolotl/integrations/kernels/plugin.py``: register a kernel mapping
keyed on ``HFScatterMoEParallelExperts`` and call
``replace_kernel_forward_from_hub`` on each transformers MoE block class
for the model type, then ``kernelize(model, mode=TRAINING)``.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

LOG = logging.getLogger(__name__)

# Maps model_type to the SparseMoeBlock class name(s) in transformers.
# Mirrors axolotl/integrations/kernels/constants.py:SPARSE_MOE_BLOCK so the
# same model coverage applies here. Values may be a single class name or a
# list (e.g. qwen3_omni_moe has Thinker + Talker MoE blocks).
SPARSE_MOE_BLOCK: dict[str, Any] = {
    "qwen2_moe": "Qwen2MoeSparseMoeBlock",
    "qwen3_moe": "Qwen3MoeSparseMoeBlock",
    "qwen3_5_moe": "Qwen3_5MoeSparseMoeBlock",
    "qwen3_5_moe_text": "Qwen3_5MoeSparseMoeBlock",
    "qwen3_next": "Qwen3NextSparseMoeBlock",
    "qwen3_vl_moe": "Qwen3VLMoeTextSparseMoeBlock",
    "qwen3_omni_moe": [
        "Qwen3OmniMoeThinkerTextSparseMoeBlock",
        "Qwen3OmniMoeTalkerTextSparseMoeBlock",
    ],
    "olmoe": "OlmoeSparseMoeBlock",
    "mixtral": "MixtralSparseMoeBlock",
    "minimax": "MiniMaxSparseMoeBlock",
    "glm_moe_dsa": "GlmMoeDsaMoE",
    "deepseek_v3": "DeepseekV3MoE",
    "glm4_moe": "Glm4MoeMoE",
    "glm4v_moe": "Glm4vMoeTextMoE",
    "minimax_m2": "MiniMaxM2SparseMoeBlock",
}

SUPPORTED_MODEL_TYPES = set(SPARSE_MOE_BLOCK.keys()) | {"gemma4_text"}

SUPPORTED_BASE_MODEL_PREFIXES = ("Qwen/Qwen3.6-35B-A3B",)


def _resolve_moe_block_classes(model_type: str) -> list[type]:
    """Resolve all MoE block classes from transformers for ``model_type``.

    Mirrors axolotl's ``resolve_moe_block_classes``.
    """
    entry = SPARSE_MOE_BLOCK.get(model_type)
    if entry is None:
        raise ValueError(
            f"Unsupported MoE model type '{model_type}'. "
            f"Supported types: {sorted(SPARSE_MOE_BLOCK.keys())}"
        )

    cls_names = entry if isinstance(entry, list) else [entry]
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        if model_type.endswith("_text"):
            parent = model_type.removesuffix("_text")
            module_path = f"transformers.models.{parent}.modeling_{parent}"
            module = importlib.import_module(module_path)
        else:
            raise

    classes: list[type] = []
    for cls_name in cls_names:
        moe_cls = getattr(module, cls_name, None)
        if moe_cls is None:
            raise ValueError(f"Could not find class '{cls_name}' in '{module_path}'")
        classes.append(moe_cls)
    return classes


@dataclass(frozen=True)
class ScatterMoEKernelConfig:
    """ScatterMoE-LoRA kernel configuration."""

    enabled: bool = False
    kernel_ref: str = "axolotl-ai-co/scattermoe-lora"
    strict: bool = False
    trust_remote_code: bool = True


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
        from kernels import (
            LayerRepository,
            Mode,
            get_kernel,
            kernelize,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
        )

        return {
            "LayerRepository": LayerRepository,
            "Mode": Mode,
            "get_kernel": get_kernel,
            "kernelize": kernelize,
            "register_kernel_mapping": register_kernel_mapping,
            "replace_kernel_forward_from_hub": replace_kernel_forward_from_hub,
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

    LayerRepository = kernels_mod["LayerRepository"]
    Mode = kernels_mod["Mode"]
    get_kernel = kernels_mod["get_kernel"]
    kernelize = kernels_mod["kernelize"]
    register_kernel_mapping = kernels_mod["register_kernel_mapping"]
    replace_kernel_forward_from_hub = kernels_mod["replace_kernel_forward_from_hub"]

    try:
        # Warm the local snapshot first so the layer is downloadable.
        try:
            get_kernel(config.kernel_ref, trust_remote_code=config.trust_remote_code)
        except TypeError:
            # Older ``kernels`` versions do not accept ``trust_remote_code``.
            get_kernel(config.kernel_ref)

        # Mirror axolotl's _register_kernels: register a Hub-backed layer
        # under the well-known mapping key ``HFScatterMoEParallelExperts``.
        # The kernel repo's exported layer is ``HFScatterMoEGatedMLP``.
        try:
            training_layer = LayerRepository(
                repo_id=config.kernel_ref,
                layer_name="HFScatterMoEGatedMLP",
                trust_remote_code=config.trust_remote_code,
            )
            inference_layer = LayerRepository(
                repo_id=config.kernel_ref,
                layer_name="HFScatterMoEGatedMLP",
                trust_remote_code=config.trust_remote_code,
            )
        except TypeError:
            # Older ``kernels`` LayerRepository may not accept trust_remote_code.
            training_layer = LayerRepository(
                repo_id=config.kernel_ref,
                layer_name="HFScatterMoEGatedMLP",
            )
            inference_layer = LayerRepository(
                repo_id=config.kernel_ref,
                layer_name="HFScatterMoEGatedMLP",
            )

        register_kernel_mapping(
            {
                "HFScatterMoEParallelExperts": {
                    "cuda": {
                        Mode.TRAINING: training_layer,
                        Mode.INFERENCE: inference_layer,
                    },
                }
            }
        )
        LOG.info(
            "scattermoe_kernel.register_kernel_mapping repo=%s layer=%s",
            config.kernel_ref,
            "HFScatterMoEGatedMLP",
        )

        # Mirror axolotl's _kernelize_model: replace the forward on every
        # MoE block class for this model type so kernelize() can route it.
        if model_type is None:
            raise RuntimeError(
                "scattermoe_kernel: model_type is unset on model.config; "
                "cannot resolve MoE block classes for forward replacement"
            )
        moe_classes = _resolve_moe_block_classes(model_type)
        for moe_cls in moe_classes:
            replace_kernel_forward_from_hub(moe_cls, "HFScatterMoEParallelExperts")
            LOG.info(
                "scattermoe_kernel.replace_kernel_forward_from_hub class=%s model_type=%s",
                moe_cls.__name__,
                model_type,
            )

        kernelize(model, mode=Mode.TRAINING)
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
