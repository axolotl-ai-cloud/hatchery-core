# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from types import SimpleNamespace

from hatchery.core.distributed import DistributedRuntime, apply_core_fsdp2_dp
from hatchery.core.parallel import ParallelConfig


class _Block:
    pass


class _ModelWithGenerate:
    def __init__(self) -> None:
        self.layers = [_Block(), _Block()]
        self.base_model = SimpleNamespace(
            model=SimpleNamespace(transformer=SimpleNamespace(h=self.layers))
        )

    def generate(self) -> None:
        pass


class _ModelWithoutGenerate:
    def __init__(self) -> None:
        self.layers = [_Block()]
        self.base_model = SimpleNamespace(
            model=SimpleNamespace(transformer=SimpleNamespace(h=self.layers))
        )


def _runtime() -> DistributedRuntime:
    return DistributedRuntime(
        global_rank=0,
        local_rank=0,
        dp_rank=0,
        world_size=2,
        dp_world_size=2,
        device=None,
        mesh=object(),
        dp_mesh=object(),
        is_core_dp_only=True,
    )


def test_core_fsdp2_registers_generate_as_forward_method(monkeypatch):
    import torch.distributed.fsdp as fsdp

    sharded: list[object] = []
    registered: list[tuple[object, str]] = []
    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **_: sharded.append(module))
    monkeypatch.setattr(
        fsdp,
        "register_fsdp_forward_method",
        lambda module, method_name: registered.append((module, method_name)),
    )

    model = _ModelWithGenerate()
    apply_core_fsdp2_dp(model, _runtime(), ParallelConfig(dp_degree=2))

    assert sharded == [*model.layers, model]
    assert registered == [(model, "generate")]


def test_core_fsdp2_skips_generate_registration_for_plain_modules(monkeypatch):
    import torch.distributed.fsdp as fsdp

    registered: list[tuple[object, str]] = []
    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **_: None)
    monkeypatch.setattr(
        fsdp,
        "register_fsdp_forward_method",
        lambda module, method_name: registered.append((module, method_name)),
    )

    apply_core_fsdp2_dp(_ModelWithoutGenerate(), _runtime(), ParallelConfig(dp_degree=2))

    assert registered == []
