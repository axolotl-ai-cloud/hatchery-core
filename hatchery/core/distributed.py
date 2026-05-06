# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Distributed runtime boundary for core-owned data parallelism."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from hatchery.core.parallel import ParallelConfig
from hatchery.core.parallel_hooks import (
    ParallelExtension,
    _legacy_helpers_for_config,
    select_parallel_extension,
)

CORE_DP_EXTENSION_NAME = "hatchery-core-fsdp2-dp"
LEGACY_HELPERS_EXTENSION_NAME = "legacy-distributed-helpers"


@dataclass
class DistributedRuntime:
    """Runtime metadata returned by core or an extension."""

    global_rank: int
    local_rank: int
    dp_rank: int
    world_size: int
    dp_world_size: int
    device: Any | None
    mesh: Any | None = None
    dp_mesh: Any | None = None
    owns_process_group: bool = False
    owns_runtime: bool = False
    extension_name: str | None = None
    extension_handle: Any | None = None
    is_core_dp_only: bool = False

    @property
    def rank(self) -> int:
        """Backward-compatible alias for ``global_rank``."""
        return self.global_rank

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def init_distributed_runtime(config: ParallelConfig) -> DistributedRuntime:
    """Initialize and return the runtime for ``config``.

    Core handles no-op single process and DP-only FSDP2. TP, CP, and
    mixed meshes are extension-owned.
    """
    if not config.is_distributed():
        return DistributedRuntime(
            global_rank=0,
            local_rank=0,
            dp_rank=0,
            world_size=1,
            dp_world_size=1,
            device=None,
        )

    if _is_core_dp_only(config):
        return _init_core_dp_runtime(config)

    extension = select_parallel_extension(config)
    if extension is not None:
        return _init_extension_runtime(extension, config)

    legacy_helpers = _legacy_helpers_for_config(config)
    if legacy_helpers is not None:
        legacy_helpers.init_distributed_if_needed(config)
        mesh = legacy_helpers.build_device_mesh(config)
        return DistributedRuntime(
            global_rank=_int_env("RANK", 0),
            local_rank=_int_env("LOCAL_RANK", 0),
            dp_rank=0,
            world_size=_int_env("WORLD_SIZE", config.world_size()),
            dp_world_size=config.dp_degree,
            device=_legacy_device(),
            mesh=mesh,
            dp_mesh=None,
            owns_process_group=False,
            owns_runtime=False,
            extension_name=LEGACY_HELPERS_EXTENSION_NAME,
            extension_handle=legacy_helpers,
        )

    raise RuntimeError(_unsupported_parallel_config_message(config))


def destroy_distributed_runtime(runtime: Optional[DistributedRuntime] = None) -> None:
    """Clean up runtime state if the owner marked it as cleanup-owned."""
    if runtime is None:
        return
    extension = runtime.extension_handle
    cleanup = getattr(extension, "cleanup_runtime", None)
    if callable(cleanup) and runtime.owns_runtime:
        cleanup(runtime)
        return
    if not runtime.owns_process_group:
        return

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_core_dp_only(config: ParallelConfig) -> bool:
    return config.dp_degree > 1 and config.tp_degree == 1 and config.cp_degree == 1


def _init_core_dp_runtime(config: ParallelConfig) -> DistributedRuntime:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    global_rank = _required_int_env("RANK")
    local_rank = _required_int_env("LOCAL_RANK")
    world_size = _required_int_env("WORLD_SIZE")
    if world_size != config.dp_degree:
        raise RuntimeError(
            "Core DP-only runtime requires WORLD_SIZE to equal "
            f"dp_degree; got WORLD_SIZE={world_size}, dp_degree={config.dp_degree}."
        )

    device = None
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group()
        initialized_here = True

    mesh_device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(mesh_device, (config.dp_degree,), mesh_dim_names=("dp",))
    return DistributedRuntime(
        global_rank=global_rank,
        local_rank=local_rank,
        dp_rank=global_rank,
        world_size=world_size,
        dp_world_size=config.dp_degree,
        device=device,
        mesh=mesh,
        dp_mesh=mesh,
        owns_process_group=initialized_here,
        owns_runtime=initialized_here,
        extension_name=CORE_DP_EXTENSION_NAME,
        extension_handle=None,
        is_core_dp_only=True,
    )


def _init_extension_runtime(
    extension: ParallelExtension, config: ParallelConfig
) -> DistributedRuntime:
    runtime = extension.init_runtime(config)
    if not isinstance(runtime, DistributedRuntime):
        raise TypeError(
            f"Parallel extension {extension.name!r} returned "
            f"{type(runtime).__name__}, expected DistributedRuntime."
        )
    runtime.extension_name = runtime.extension_name or extension.name
    runtime.extension_handle = runtime.extension_handle or extension
    return runtime


def _unsupported_parallel_config_message(config: ParallelConfig) -> str:
    return (
        "hatchery-core supports FSDP2 data parallel only for "
        "dp_degree>1,tp_degree=1,cp_degree=1. "
        f"Requested dp={config.dp_degree},tp={config.tp_degree},cp={config.cp_degree}. "
        "Install/register a parallel extension for TP/CP support."
    )


def _required_int_env(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"Core DP-only runtime requires torchrun env var {name}.")
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Core DP-only runtime env var {name} must be an integer.") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _legacy_device() -> str | None:
    local_rank = os.environ.get("LOCAL_RANK")
    return f"cuda:{local_rank}" if local_rank is not None else None
