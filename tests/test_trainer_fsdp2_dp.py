# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _runtime(*, dp_rank: int = 0, dp_world_size: int = 2):
    from hatchery.core.distributed import DistributedRuntime

    return DistributedRuntime(
        global_rank=dp_rank,
        local_rank=dp_rank,
        dp_rank=dp_rank,
        world_size=dp_world_size,
        dp_world_size=dp_world_size,
        device=None,
        mesh="mesh",
        dp_mesh="dp-mesh",
        extension_name="hatchery-core-fsdp2-dp",
        is_core_dp_only=True,
    )


def _trainer():
    from hatchery.core.trainer import VanillaTrainer

    return VanillaTrainer(
        base_model_name="mock",
        device="cpu",
        dtype=torch.float32,
        load_model=False,
    )


def test_allocate_batch_uses_runtime_dp_rank_not_dist_get_rank():
    from hatchery.core.parallel import ParallelConfig

    trainer = _trainer()
    trainer.parallel = ParallelConfig(dp_degree=2, batch_strategy="split")
    trainer._distributed_runtime = _runtime(dp_rank=1, dp_world_size=2)

    data = [{"idx": i, "input_ids": [i + 1]} for i in range(4)]

    assert trainer._allocate_batch(data) == data[2:]


def test_core_fsdp2_full_param_sessions_are_rejected():
    from hatchery.core.parallel import ParallelConfig
    from hatchery.core.trainer import LoraSpec

    trainer = _trainer()
    trainer.parallel = ParallelConfig(dp_degree=2)
    trainer._distributed_runtime = _runtime()

    with pytest.raises(RuntimeError, match="Full-parameter sessions are unsupported"):
        trainer.attach_session("fp", LoraSpec.full_param())


def test_core_fsdp2_rejects_later_dynamic_adapter_attach():
    from hatchery.core.parallel import ParallelConfig
    from hatchery.core.trainer import LoraSpec

    trainer = _trainer()
    trainer.parallel = ParallelConfig(dp_degree=2)
    trainer._distributed_runtime = _runtime()
    trainer._parallel_applied = True
    trainer._peft = type("FakePeft", (), {"peft_config": {"sess_first": object()}})()

    with pytest.raises(RuntimeError, match="Dynamic adapter attach"):
        trainer.attach_session("second", LoraSpec(rank=4, lora_alpha=8, target_modules=["q_proj"]))


def test_core_fsdp2_rejects_detach_eviction():
    from hatchery.core.parallel import ParallelConfig
    from hatchery.core.trainer import LoraSpec

    trainer = _trainer()
    trainer.parallel = ParallelConfig(dp_degree=2)
    trainer._distributed_runtime = _runtime()
    trainer._specs["first"] = LoraSpec(rank=4, lora_alpha=8, target_modules=["q_proj"])

    with pytest.raises(RuntimeError, match="detach/eviction"):
        trainer.detach_session("first")


def test_apply_parallel_plan_delegates_to_core_dp_helper(monkeypatch):
    from hatchery.core.parallel import ParallelConfig

    calls = []
    trainer = _trainer()
    trainer.parallel = ParallelConfig(dp_degree=2)
    trainer._distributed_runtime = _runtime()
    trainer._peft = object()

    monkeypatch.setattr(
        "hatchery.core.trainer.apply_core_fsdp2_dp",
        lambda model, runtime, config: calls.append((model, runtime, config)),
    )

    trainer._apply_parallel_plan()

    assert calls == [(trainer._peft, trainer._distributed_runtime, trainer.parallel)]
