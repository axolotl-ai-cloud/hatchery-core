# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Registry for parallel-training extensions.

Core owns single-process execution and the FSDP2 data-parallel-only
runtime. Tensor parallelism, context parallelism, and mixed meshes are
selected through capability-bearing extensions registered here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ParallelExtension:
    """Capability-bearing extension for non-core parallel configurations."""

    name: str
    init_runtime: Callable[[Any], Any]
    apply_parallel_plan: Callable[[Any, Any, Any], None]
    supports_tp: bool = False
    supports_cp: bool = False
    supports_mixed_dp: bool = False
    context_parallel_region: Optional[Callable[..., Any]] = None
    cleanup_runtime: Optional[Callable[[Any], None]] = None


_EXTENSIONS: list[ParallelExtension] = []


def register_parallel_extension(extension: ParallelExtension) -> None:
    """Register a capability-bearing parallel extension. Last match wins."""
    _EXTENSIONS.append(extension)


def registered_parallel_extensions() -> tuple[ParallelExtension, ...]:
    """Return registered parallel extensions in registration order."""
    return tuple(_EXTENSIONS)


def select_parallel_extension(config: Any) -> Optional[ParallelExtension]:
    """Return the extension that should own a TP/CP/mixed config, if any.

    Core DP-only intentionally never selects an extension; that runtime
    is owned by :mod:`hatchery.core.distributed`.
    """
    needs_tp = config.tp_degree > 1
    needs_cp = config.cp_degree > 1
    mixed_dp = config.dp_degree > 1 and (needs_tp or needs_cp)
    if not (needs_tp or needs_cp):
        return None

    for extension in reversed(_EXTENSIONS):
        if needs_tp and not extension.supports_tp:
            continue
        if needs_cp and not extension.supports_cp:
            continue
        if mixed_dp and not extension.supports_mixed_dp:
            continue
        return extension
    return None


def _reset_parallel_hooks_for_tests() -> None:
    """Clear module-level registries for isolated unit tests."""
    _EXTENSIONS.clear()
