# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Registry for distributed-training helper functions.

Core ships parallelism config dataclasses (:mod:`hatchery.core.parallel`)
but not the torch-level distributed setup — FSDP2 wrapping, tensor
parallelism, context parallelism, and device-mesh construction all
require torch.distributed and are provided by an extension package.

Extensions register their helpers via :func:`register_distributed_helpers`
on import. Core looks them up at distributed-run time and raises a
helpful error if a distributed mode was requested without any
extension registering the helpers.

The helpers are plain callables rather than a Protocol class so core
does not need to import torch just to declare the types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DistributedHelpers:
    """Callables a distributed-training extension must provide."""

    init_distributed_if_needed: Callable[[Any], None]
    build_device_mesh: Callable[[Any], Any]
    get_cp_mesh: Callable[[Any, Any], Any]
    context_parallel_region: Callable[..., Any]


_HELPERS: Optional[DistributedHelpers] = None


def register_distributed_helpers(helpers: DistributedHelpers) -> None:
    """Install an extension's distributed helpers. Last caller wins."""
    global _HELPERS
    _HELPERS = helpers


def get_distributed_helpers() -> DistributedHelpers:
    """Return the installed helpers or raise a helpful error."""
    if _HELPERS is None:
        raise RuntimeError(
            "Distributed training (dp/tp/cp > 1) requires an extension "
            "package that registers distributed helpers via "
            "hatchery.core.parallel_hooks.register_distributed_helpers(). "
            "No helpers are registered — run single-GPU only, or install "
            "an extension package that provides the torch.distributed "
            "integration."
        )
    return _HELPERS
