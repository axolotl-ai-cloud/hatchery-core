# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Integration tests for :class:`InProcessVLLMSampler`.

These tests actually spin up a small vLLM engine (``Qwen3-0.6B``) on a
real GPU and exercise the wake/sleep lifecycle. They're skipped if
``vllm`` is not importable, if CUDA is not available, or if the
``HATCHERY_SKIP_GPU_TESTS`` env var is set.

The non-GPU unit coverage (protocol shape, sleep-mode=False no-op path,
adapter bookkeeping) lives in ``test_sampling.py`` alongside the other
backends; this file is only for the cases where we need a live engine
to prove the abstraction holds.
"""

from __future__ import annotations

import os

import pytest

# Skip the whole module cleanly when the environment can't support it.
pytest.importorskip("vllm")

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    _CUDA_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="needs CUDA"),
    pytest.mark.skipif(
        os.environ.get("HATCHERY_SKIP_GPU_TESTS") == "1",
        reason="HATCHERY_SKIP_GPU_TESTS=1",
    ),
]

from hatchery.core.sampling import (  # noqa: E402
    InProcessVLLMSampler,
    SampledSequence,
    awake,
)

# A small instruct-tuned model that's quick to load. Override via env for
# environments where this isn't the cheapest option available.
_MODEL = os.environ.get("HATCHERY_TEST_SMALL_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
async def sampler() -> InProcessVLLMSampler:
    """Shared sleep-enabled sampler for the whole module. Loading vLLM
    is expensive (~15s) so we pay it once and reuse across tests."""
    s = InProcessVLLMSampler(
        model=_MODEL,
        enable_sleep_mode=True,
        gpu_memory_utilization=0.35,
        max_model_len=1024,
    )
    await s.initialize()
    yield s
    await s.close()


async def test_initialize_leaves_sampler_asleep(sampler: InProcessVLLMSampler) -> None:
    """With ``enable_sleep_mode=True`` initialize() should finish in the
    asleep state so a trainer can claim VRAM for its own setup."""
    assert sampler.is_awake is False


async def test_sample_while_asleep_raises(sampler: InProcessVLLMSampler) -> None:
    """Sampling without waking first is a caller error — we surface it
    loudly rather than silently auto-waking (which would hide the
    intended coordination with a trainer)."""
    assert sampler.is_awake is False
    with pytest.raises(RuntimeError, match="asleep"):
        await sampler.sample(
            adapter_name="base",
            prompt_tokens=[1, 2, 3],
            max_tokens=4,
        )


async def test_awake_context_manager_wakes_and_sleeps(
    sampler: InProcessVLLMSampler,
) -> None:
    """awake() is the ergonomic entry point: the sampler is awake inside
    the block and asleep again after exit."""
    assert sampler.is_awake is False
    async with awake(sampler):
        assert sampler.is_awake is True
        out = await sampler.sample(
            adapter_name="base",
            prompt_tokens=[1],
            max_tokens=4,
        )
        assert out and isinstance(out[0], SampledSequence)
        assert len(out[0].tokens) > 0
    assert sampler.is_awake is False


async def test_explicit_wake_then_multiple_samples(
    sampler: InProcessVLLMSampler,
) -> None:
    """For rollout bursts where many sample() calls back-to-back, the
    caller can wake once and sleep at the end — avoids paying wake latency
    per request."""
    await sampler.wake()
    try:
        out1 = await sampler.sample(adapter_name="base", prompt_tokens=[1, 2], max_tokens=4)
        out2 = await sampler.sample(adapter_name="base", prompt_tokens=[3, 4, 5], max_tokens=4)
        assert out1 and out2
    finally:
        await sampler.sleep()
    assert sampler.is_awake is False


async def test_wake_is_idempotent(sampler: InProcessVLLMSampler) -> None:
    """Calling wake() twice in a row is fine — useful when the caller
    doesn't want to track state carefully."""
    await sampler.wake()
    await sampler.wake()
    assert sampler.is_awake is True
    await sampler.sleep()
    await sampler.sleep()
    assert sampler.is_awake is False


async def test_post_wake_sampling_works_across_cycles(
    sampler: InProcessVLLMSampler,
) -> None:
    """The real invariant we care about: sampling after wake() returns
    valid output on every cycle, not just the first. A broken sleep
    implementation (e.g., graphs invalidated by wake) would show up as
    garbage tokens or an engine error here."""
    for _ in range(3):
        async with awake(sampler):
            out = await sampler.sample(
                adapter_name="base",
                prompt_tokens=[1, 2, 3, 4],
                max_tokens=8,
                temperature=0.0,  # greedy — deterministic output
            )
        assert out and out[0].tokens
        assert out[0].stop_reason in ("stop", "length")


async def test_health_check_reflects_initialization() -> None:
    """health_check is False before initialize(), True after, False after
    close()."""
    s = InProcessVLLMSampler(
        model=_MODEL,
        enable_sleep_mode=True,
        gpu_memory_utilization=0.35,
        max_model_len=512,
    )
    assert await s.health_check() is False
    await s.initialize()
    try:
        assert await s.health_check() is True
    finally:
        await s.close()
    assert await s.health_check() is False


# ── Non-GPU unit tests for the no-sleep-mode path ────────────────────────


async def test_sleep_mode_disabled_wake_and_sleep_are_noops() -> None:
    """When ``enable_sleep_mode=False`` wake/sleep do not touch the engine
    — they're protocol-conformance no-ops. This branch is exercised
    without a live engine because it doesn't need one.

    We don't call initialize() here — the point is that wake()/sleep()
    short-circuit on ``enable_sleep_mode`` before touching ``self._llm``.
    """
    s = InProcessVLLMSampler(model=_MODEL, enable_sleep_mode=False)
    # No initialize — these should still succeed as no-ops.
    await s.wake()
    await s.sleep()
    # is_awake stays at its default (False) because we never called
    # initialize(); that's fine — the invariant is just "no exception".
