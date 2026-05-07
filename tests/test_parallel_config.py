# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU unit tests for FSDP2 plan resolution via ParallelConfig.

Covers: world-size calculation, is_distributed routing, __post_init__
validation guards, and from_env parsing.  No torch import required.
"""

from __future__ import annotations

import pytest

from hatchery.core.parallel import OffloadConfig, ParallelConfig

pytestmark = [pytest.mark.fsdp2]


# ---------------------------------------------------------------------------
# world_size / is_distributed
# ---------------------------------------------------------------------------


def test_default_parallel_config_is_single_process():
    cfg = ParallelConfig()
    assert cfg.world_size() == 1
    assert not cfg.is_distributed()


def test_parallel_config_world_size_is_product_of_degrees():
    cfg = ParallelConfig(dp_degree=2, tp_degree=3, cp_degree=4)
    assert cfg.world_size() == 24


def test_dp_only_config_is_distributed():
    assert ParallelConfig(dp_degree=2).is_distributed()


def test_tp_only_config_is_distributed():
    assert ParallelConfig(tp_degree=2).is_distributed()


def test_cp_only_config_is_distributed():
    assert ParallelConfig(cp_degree=2).is_distributed()


# ---------------------------------------------------------------------------
# __post_init__ validation — these guard the plan-resolution dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dp_degree": 0},
        {"dp_degree": -1},
        {"tp_degree": 0},
        {"cp_degree": -1},
    ],
)
def test_parallel_config_rejects_non_positive_degrees(kwargs):
    with pytest.raises(ValueError, match="must be >= 1"):
        ParallelConfig(**kwargs)


def test_parallel_config_rejects_sequence_parallel_without_tp():
    with pytest.raises(ValueError, match="sequence_parallel only makes sense with tp_degree > 1"):
        ParallelConfig(sequence_parallel=True, tp_degree=1)


def test_parallel_config_allows_sequence_parallel_with_tp():
    cfg = ParallelConfig(sequence_parallel=True, tp_degree=2)
    assert cfg.sequence_parallel


def test_parallel_config_rejects_sequence_packing_with_cp():
    with pytest.raises(ValueError, match="sequence_packing with cp_degree > 1 is not supported"):
        ParallelConfig(sequence_packing=True, cp_degree=2)


def test_parallel_config_allows_sequence_packing_without_cp():
    cfg = ParallelConfig(sequence_packing=True, cp_degree=1)
    assert cfg.sequence_packing


def test_parallel_config_rejects_non_positive_max_packed_len():
    with pytest.raises(ValueError, match="max_packed_len must be > 0"):
        ParallelConfig(max_packed_len=0)


# ---------------------------------------------------------------------------
# from_env — plan resolution reads these vars at worker startup
# ---------------------------------------------------------------------------


def test_from_env_defaults_to_single_process(monkeypatch):
    for var in (
        "HATCHERY_DP_DEGREE",
        "HATCHERY_TP_DEGREE",
        "HATCHERY_CP_DEGREE",
        "HATCHERY_SP",
        "HATCHERY_SEQUENCE_PACKING",
        "HATCHERY_MAX_PACKED_LEN",
        "HATCHERY_QUANT_SCHEME",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = ParallelConfig.from_env()
    assert cfg == ParallelConfig()


def test_from_env_parses_dp_degree(monkeypatch):
    monkeypatch.setenv("HATCHERY_DP_DEGREE", "4")
    cfg = ParallelConfig.from_env()
    assert cfg.dp_degree == 4
    assert cfg.world_size() == 4


def test_from_env_parses_offload_flags(monkeypatch):
    monkeypatch.setenv("HATCHERY_CPU_OFFLOAD_PARAMS", "1")
    monkeypatch.setenv("HATCHERY_CPU_OFFLOAD_OPTIMIZER", "1")
    monkeypatch.setenv("HATCHERY_ACTIVATION_CKPT", "1")

    cfg = ParallelConfig.from_env()
    assert cfg.offload.cpu_offload_params
    assert cfg.offload.cpu_offload_optimizer
    assert cfg.offload.activation_checkpointing


def test_from_env_ignores_unknown_quant_scheme(monkeypatch):
    monkeypatch.setenv("HATCHERY_QUANT_SCHEME", "bogus")
    cfg = ParallelConfig.from_env()
    assert cfg.quant.scheme == "none"


# ---------------------------------------------------------------------------
# OffloadConfig defaults
# ---------------------------------------------------------------------------


def test_offload_config_defaults_are_all_false():
    off = OffloadConfig()
    assert not off.cpu_offload_params
    assert not off.cpu_offload_optimizer
    assert not off.activation_checkpointing
    assert off.nvme_path is None
