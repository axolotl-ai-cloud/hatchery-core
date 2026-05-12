# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Model ID resolver with context-length routing.

Tinker/Fireworks convention: long-context variants append a
``:peft:<max_tokens>`` suffix to the base model ID::

    Qwen/Qwen3.5-397B-A17B              → 64K standard
    Qwen/Qwen3.5-397B-A17B:peft:262144  → 256K long-context

The resolver parses the model ID, determines:

1. **Base model** — the HF hub name (stripped of suffixes).
2. **Max context length** — from the suffix or model defaults.
3. **Required CP degree** — automatically computed when the
   requested context exceeds what a single GPU can handle.
4. **Pricing tier** — standard vs. long-context multiplier.

Workers advertise their ``cp_degree`` at registration time.
Jobs with ``required_cp_degree > 1`` are nacked by standard
workers and picked up by CP-enabled ones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedModel:
    """Result of parsing a model ID."""

    base_model: str
    max_context_length: int
    is_long_context: bool
    required_cp_degree: int
    raw_model_id: str


# Default context lengths by model family (when no suffix is given).
# These match the TM docs. Models not in this table get 32K.
_DEFAULT_CONTEXT: dict[str, int] = {
    "Qwen/Qwen3.5": 65536,
    "Qwen/Qwen3.6": 262144,
    "nvidia/NVIDIA-Nemotron-3": 65536,
}

# Standard single-GPU context limit. Beyond this, CP is required.
# Conservative — most 80GB GPUs handle 32K with LoRA; 64K is the edge.
_SINGLE_GPU_CONTEXT_LIMIT = 65536

# CP degree steps: how many GPUs needed for each context tier.
_CP_TIERS = [
    (65536, 1),  # ≤64K: single GPU
    (131072, 2),  # ≤128K: 2-way CP
    (262144, 4),  # ≤256K: 4-way CP
    (524288, 8),  # ≤512K: 8-way CP
]


def resolve_model_id(model_id: str) -> ResolvedModel:
    """Parse a model ID and return routing configuration.

    Examples::

        >>> resolve_model_id("meta-llama/Llama-3.1-8B")
        ResolvedModel(base_model='meta-llama/Llama-3.1-8B', max_context_length=32768, ...)

        >>> resolve_model_id("Qwen/Qwen3.5-397B-A17B:peft:262144")
        ResolvedModel(base_model='Qwen/Qwen3.5-397B-A17B', max_context_length=262144, ...)
    """
    base_model, max_ctx, is_long = _parse_model_id(model_id)
    cp_degree = _compute_cp_degree(max_ctx)

    return ResolvedModel(
        base_model=base_model,
        max_context_length=max_ctx,
        is_long_context=is_long,
        required_cp_degree=cp_degree,
        raw_model_id=model_id,
    )


def _parse_model_id(model_id: str) -> tuple[str, int, bool]:
    """Parse `base_model:peft:context_length` format.

    Returns (base_model, max_context_length, is_long_context).
    """
    parts = model_id.split(":")
    if len(parts) >= 3 and parts[-2] == "peft":
        base = ":".join(parts[:-2])
        try:
            ctx = int(parts[-1])
            return base, ctx, True
        except ValueError:
            pass
    # No suffix — use default context for this model family.
    base = model_id
    default_ctx = _lookup_default_context(base)
    return base, default_ctx, False


def _lookup_default_context(base_model: str) -> int:
    """Look up the default context length for a model."""
    for prefix, ctx in _DEFAULT_CONTEXT.items():
        if base_model.startswith(prefix):
            return ctx
    return 32768  # Conservative default.


def _compute_cp_degree(max_ctx: int) -> int:
    """Determine the minimum CP degree needed for the context length."""
    for threshold, degree in _CP_TIERS:
        if max_ctx <= threshold:
            return degree
    # Beyond all known tiers — extrapolate.
    return max(8, max_ctx // _SINGLE_GPU_CONTEXT_LIMIT)
