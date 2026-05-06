# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import builtins
import sys
import types

import pytest

from hatchery.core.distributed import (
    CORE_DP_EXTENSION_NAME,
    DistributedRuntime,
    apply_core_fsdp2_dp,
    destroy_distributed_runtime,
    init_distributed_runtime,
    iter_decoder_layers,
)
from hatchery.core.parallel import ParallelConfig
from hatchery.core.parallel_hooks import (
    ParallelExtension,
    _reset_parallel_hooks_for_tests,
    register_parallel_extension,
)


@pytest.fixture(autouse=True)
def reset_parallel_hooks():
    _reset_parallel_hooks_for_tests()
    yield
    _reset_parallel_hooks_for_tests()


def test_noop_runtime_is_lazy_and_torch_free(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("single-GPU runtime should not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    runtime = init_distributed_runtime(ParallelConfig())

    assert runtime.global_rank == 0
    assert runtime.local_rank == 0
    assert runtime.dp_rank == 0
    assert runtime.world_size == 1
    assert runtime.dp_world_size == 1
    assert runtime.device is None
    assert runtime.mesh is None
    assert runtime.dp_mesh is None
    assert runtime.extension_name is None
    assert not runtime.owns_process_group
    assert not runtime.owns_runtime
    assert not runtime.is_core_dp_only


def test_core_dp_selection_wins_over_registered_extensions(monkeypatch):
    calls: list[str] = []
    register_parallel_extension(
        ParallelExtension(
            name="mixed-extension",
            init_runtime=lambda config: calls.append("init"),
            apply_parallel_plan=lambda model, runtime, config: None,
            supports_tp=True,
            supports_cp=True,
            supports_mixed_dp=True,
        )
    )
    _install_fake_torch(monkeypatch, cuda_available=True, dist_initialized=False)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    runtime = init_distributed_runtime(ParallelConfig(dp_degree=2))

    assert calls == []
    assert runtime.global_rank == 1
    assert runtime.local_rank == 1
    assert runtime.dp_rank == 1
    assert runtime.world_size == 2
    assert runtime.dp_world_size == 2
    assert runtime.mesh == {"device": "cuda", "shape": (2,), "names": ("dp",)}
    assert runtime.dp_mesh == runtime.mesh
    assert runtime.extension_name == CORE_DP_EXTENSION_NAME
    assert runtime.is_core_dp_only
    assert runtime.owns_process_group
    assert runtime.owns_runtime


@pytest.mark.parametrize(
    "config",
    [
        ParallelConfig(tp_degree=2),
        ParallelConfig(cp_degree=2),
        ParallelConfig(dp_degree=2, tp_degree=2),
        ParallelConfig(dp_degree=2, cp_degree=2),
    ],
)
def test_unsupported_tp_cp_without_extension_raises(config):
    with pytest.raises(RuntimeError, match="Install/register a parallel extension"):
        init_distributed_runtime(config)


def test_capability_extension_selection_for_tp_cp():
    calls: list[tuple[str, tuple[int, int, int]]] = []

    def init_runtime(config: ParallelConfig) -> DistributedRuntime:
        calls.append(("init", (config.dp_degree, config.tp_degree, config.cp_degree)))
        return DistributedRuntime(
            global_rank=0,
            local_rank=0,
            dp_rank=0,
            world_size=config.world_size(),
            dp_world_size=config.dp_degree,
            device="extension-device",
            mesh="extension-mesh",
            dp_mesh="extension-dp-mesh",
        )

    extension = ParallelExtension(
        name="mixed-extension",
        init_runtime=init_runtime,
        apply_parallel_plan=lambda model, runtime, config: None,
        supports_tp=True,
        supports_cp=True,
        supports_mixed_dp=True,
    )
    register_parallel_extension(extension)

    runtime = init_distributed_runtime(ParallelConfig(dp_degree=2, tp_degree=2, cp_degree=2))

    assert calls == [("init", (2, 2, 2))]
    assert runtime.extension_name == "mixed-extension"
    assert runtime.extension_handle is extension
    assert runtime.mesh == "extension-mesh"


def test_registered_extension_handles_tp():
    register_parallel_extension(
        ParallelExtension(
            name="tp-extension",
            init_runtime=lambda config: DistributedRuntime(
                global_rank=0,
                local_rank=0,
                dp_rank=0,
                world_size=config.world_size(),
                dp_world_size=config.dp_degree,
                device=None,
                mesh="extension-mesh",
            ),
            apply_parallel_plan=lambda model, runtime, config: None,
            supports_tp=True,
        )
    )

    runtime = init_distributed_runtime(ParallelConfig(tp_degree=2))

    assert runtime.extension_name == "tp-extension"
    assert runtime.mesh == "extension-mesh"


def test_cleanup_ownership_destroys_only_owned_process_group(monkeypatch):
    fake = _install_fake_torch(monkeypatch, cuda_available=False, dist_initialized=True)
    destroy_distributed_runtime(
        DistributedRuntime(
            global_rank=0,
            local_rank=0,
            dp_rank=0,
            world_size=2,
            dp_world_size=2,
            device=None,
            owns_process_group=False,
        )
    )
    assert fake.dist.destroy_calls == 0

    destroy_distributed_runtime(
        DistributedRuntime(
            global_rank=0,
            local_rank=0,
            dp_rank=0,
            world_size=2,
            dp_world_size=2,
            device=None,
            owns_process_group=True,
        )
    )
    assert fake.dist.destroy_calls == 1


def test_cleanup_delegates_to_owning_extension():
    cleaned: list[DistributedRuntime] = []
    extension = ParallelExtension(
        name="cleanup-extension",
        init_runtime=lambda config: None,
        apply_parallel_plan=lambda model, runtime, config: None,
        cleanup_runtime=cleaned.append,
    )
    runtime = DistributedRuntime(
        global_rank=0,
        local_rank=0,
        dp_rank=0,
        world_size=2,
        dp_world_size=1,
        device=None,
        owns_runtime=True,
        extension_handle=extension,
    )

    destroy_distributed_runtime(runtime)

    assert cleaned == [runtime]


def test_iter_decoder_layers_supports_peft_and_gpt2_layouts():
    peft_layers = [object(), object()]
    peft_gpt2_layers = [object()]
    gpt2_layers = [object()]
    peft = types.SimpleNamespace(
        base_model=types.SimpleNamespace(
            model=types.SimpleNamespace(model=types.SimpleNamespace(layers=peft_layers))
        )
    )
    peft_gpt2 = types.SimpleNamespace(
        base_model=types.SimpleNamespace(
            model=types.SimpleNamespace(transformer=types.SimpleNamespace(h=peft_gpt2_layers))
        )
    )
    gpt2 = types.SimpleNamespace(transformer=types.SimpleNamespace(h=gpt2_layers))

    assert list(iter_decoder_layers(peft)) == peft_layers
    assert list(iter_decoder_layers(peft_gpt2)) == peft_gpt2_layers
    assert list(iter_decoder_layers(gpt2)) == gpt2_layers


def test_apply_core_fsdp2_dp_wraps_discovered_layers(monkeypatch):
    calls: list[tuple[object, dict]] = []
    fake_fsdp = types.ModuleType("torch.distributed.fsdp")

    class CPUOffloadPolicy:  # noqa: D401 - test stub
        pass

    def fully_shard(block, **kwargs):
        calls.append((block, kwargs))

    fake_fsdp.CPUOffloadPolicy = CPUOffloadPolicy
    fake_fsdp.fully_shard = fully_shard
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp)

    layers = [object(), object()]
    model = types.SimpleNamespace(transformer=types.SimpleNamespace(h=layers))
    runtime = DistributedRuntime(
        global_rank=0,
        local_rank=0,
        dp_rank=0,
        world_size=2,
        dp_world_size=2,
        device=None,
        dp_mesh="dp-mesh",
        is_core_dp_only=True,
    )

    apply_core_fsdp2_dp(model, runtime, ParallelConfig(dp_degree=2))

    assert calls == [
        (layers[0], {"mesh": "dp-mesh"}),
        (layers[1], {"mesh": "dp-mesh"}),
        (model, {"mesh": "dp-mesh"}),
    ]


def test_apply_core_fsdp2_dp_raises_on_unknown_layout(monkeypatch):
    fake_fsdp = types.ModuleType("torch.distributed.fsdp")
    fake_fsdp.CPUOffloadPolicy = object
    fake_fsdp.fully_shard = lambda block, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp)
    runtime = DistributedRuntime(
        global_rank=0,
        local_rank=0,
        dp_rank=0,
        world_size=2,
        dp_world_size=2,
        device=None,
        dp_mesh="dp-mesh",
        is_core_dp_only=True,
    )

    with pytest.raises(RuntimeError, match="could not discover decoder layers"):
        apply_core_fsdp2_dp(types.SimpleNamespace(), runtime, ParallelConfig(dp_degree=2))


def _install_fake_torch(monkeypatch, *, cuda_available: bool, dist_initialized: bool):
    fake_torch = types.ModuleType("torch")
    fake_cuda = _FakeCuda(cuda_available)
    fake_torch.cuda = fake_cuda
    fake_torch.device = lambda kind, index=None: f"{kind}:{index}" if index is not None else kind

    fake_dist = _FakeDist(dist_initialized)
    fake_device_mesh = types.ModuleType("torch.distributed.device_mesh")
    fake_device_mesh.init_device_mesh = lambda device, shape, mesh_dim_names: {
        "device": device,
        "shape": shape,
        "names": mesh_dim_names,
    }

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "torch.distributed", fake_dist)
    monkeypatch.setitem(
        __import__("sys").modules, "torch.distributed.device_mesh", fake_device_mesh
    )
    return types.SimpleNamespace(torch=fake_torch, cuda=fake_cuda, dist=fake_dist)


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.devices: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def set_device(self, local_rank: int) -> None:
        self.devices.append(local_rank)


class _FakeDist(types.ModuleType):
    def __init__(self, initialized: bool) -> None:
        super().__init__("torch.distributed")
        self.initialized = initialized
        self.init_calls = 0
        self.destroy_calls = 0

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def init_process_group(self) -> None:
        self.initialized = True
        self.init_calls += 1

    def destroy_process_group(self) -> None:
        self.initialized = False
        self.destroy_calls += 1
