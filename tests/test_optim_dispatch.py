# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Selector + builder for AdamW vs 8-bit AdamW dispatch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hatchery.core.optim_dispatch import build_optimizer, select_optimizer_kind  # noqa: E402


def test_lora_always_uses_fp32_adamw():
    # LoRA ignores the budget — its state is tiny.
    assert (
        select_optimizer_kind(
            training_mode="lora",
            trainable_param_count=10**12,
            vram_free_bytes=1,
            vram_budget_frac=0.40,
        )
        == "adamw"
    )


def test_fp_small_model_stays_on_fp32():
    # 100M params * 8 = 800 MB; well under 0.4 * 80 GB = 32 GB.
    kind = select_optimizer_kind(
        training_mode="full_param",
        trainable_param_count=100_000_000,
        vram_free_bytes=80 * 1024**3,
        vram_budget_frac=0.40,
    )
    assert kind == "adamw"


def test_fp_large_model_promotes_to_8bit():
    # 1.5B params * 8 = 12 GB; > 0.4 * 24 GB (= 9.6 GB) → 8-bit.
    kind = select_optimizer_kind(
        training_mode="full_param",
        trainable_param_count=1_500_000_000,
        vram_free_bytes=24 * 1024**3,
        vram_budget_frac=0.40,
    )
    assert kind == "adamw_8bit"


def test_fp_unknown_vram_falls_back_to_fp32():
    # Selector can't reason without a VRAM probe — be conservative.
    kind = select_optimizer_kind(
        training_mode="full_param",
        trainable_param_count=10**10,
        vram_free_bytes=0,
        vram_budget_frac=0.40,
    )
    assert kind == "adamw"


def test_build_adamw_returns_torch_adamw():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer(
        [p],
        kind="adamw",
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=False,
    )
    assert isinstance(opt, torch.optim.AdamW)


def test_build_8bit_returns_torchao_when_available():
    torchao_optim = pytest.importorskip("torchao.optim")
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer(
        [p],
        kind="adamw_8bit",
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    assert isinstance(opt, torchao_optim.AdamW8bit)


def test_build_8bit_falls_back_when_torchao_missing(monkeypatch):
    # Force the import path used inside build_optimizer to fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchao.optim" or name.startswith("torchao.optim."):
            raise ImportError("simulated missing torchao")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer(
        [p],
        kind="adamw_8bit",
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    assert isinstance(opt, torch.optim.AdamW)
