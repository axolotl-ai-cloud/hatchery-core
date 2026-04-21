# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Data-parallel batch allocation.

When a worker runs with ``dp_degree > 1`` the caller (typically
``GPUWorker._handle_forward_backward``) has to decide what each rank
should see. Three strategies are supported:

``REPLICATE``
    Every rank processes the full batch. FSDP's default
    reduce-mean on the backward pass gives the same gradient on
    every rank, and the all-reduce is effectively a no-op. This
    wastes compute (work scales as ``batch_size * dp_degree``)
    but is always correct regardless of batch size. Use this
    when the user sent fewer items than you have ranks.

``SPLIT``
    The batch is divided evenly across ranks. Requires
    ``batch_size % dp_degree == 0`` — raises otherwise. This is
    true data parallelism: each rank does ``batch_size / dp_degree``
    items of work and the all-reduce combines their gradients
    into the full-batch gradient (mathematically identical to
    running the full batch on one rank).

``AUTO``
    SPLIT when ``batch_size % dp_degree == 0``, REPLICATE otherwise.
    This is the default because it handles both "small batch,
    more GPUs than data" and "aligned batch, use full compute"
    without the user having to think about it.

Why no PAD_AND_SPLIT
--------------------
Padding a short batch with dummy items (``labels=-100``) so it
divides evenly looks tempting, but the gradient ends up scaled by
``real_items / total_items_after_pad`` — incorrect unless you also
rescale the loss. Fixing that introduces a dp-aware loss scaler
that the user-visible ``loss_fn`` values would then disagree with,
which is a much bigger interface change than it's worth. REPLICATE
handles the same case correctly and the wasted compute is already
the operating cost of running FSDP on a too-small batch.

Gradient correctness sketch
---------------------------
Let ``B = batch_size``, ``N = dp_degree``, items ``x_0…x_{B-1}``.
The full-batch gradient is ``g = (1/B) * Σ d/dW loss(x_i)``.

For SPLIT with ``B = N * k``:
  each rank computes ``g_r = (1/k) * Σ d/dW loss(x_{rk+j})``.
  FSDP all-reduce mean gives ``(1/N) * Σ g_r = (1/B) * Σ d/dW loss(x_i) = g``. ✓

For REPLICATE:
  every rank computes ``g_r = (1/B) * Σ d/dW loss(x_i) = g``.
  FSDP all-reduce mean gives ``(1/N) * N * g = g``. ✓

Both modes produce the same gradient — the only difference is wall-clock
cost: SPLIT takes ``k`` items per rank of compute, REPLICATE takes ``B``
items per rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BatchStrategy(StrEnum):
    AUTO = "auto"
    REPLICATE = "replicate"
    SPLIT = "split"


@dataclass
class BatchAllocation:
    """What one rank should run for this step.

    Attributes
    ----------
    data:
        The local rank's slice of the data list. For REPLICATE this
        is the full input (a shallow copy); for SPLIT it is exactly
        ``batch_size / dp_degree`` items.
    total_items:
        The full batch size the caller sent in, before any splitting.
        Emitted as a metric for utilization tracking.
    local_items:
        ``len(data)``. Equal to ``total_items`` for REPLICATE,
        ``total_items // dp_degree`` for SPLIT.
    strategy:
        The concrete strategy that was actually applied. If the
        caller passed AUTO this tells you whether the allocator
        chose REPLICATE or SPLIT for this particular batch.
    replicated:
        ``True`` when every rank sees the same data, ``False`` when
        the data was split. Mostly used for assertions in tests.
    wasted_compute_pct:
        Fraction of FSDP compute that's redundant. Zero when the
        batch splits cleanly (SPLIT), ``(dp - 1) / dp`` when the
        allocator fell back to REPLICATE. Useful as a Prometheus
        metric or log line when operators are diagnosing throughput.
    """

    data: list[dict]
    total_items: int
    local_items: int
    strategy: BatchStrategy
    replicated: bool
    wasted_compute_pct: float


class BatchAllocationError(ValueError):
    """Raised when the requested strategy is incompatible with the batch."""


def prepare_batch_for_dp(
    data: list[dict],
    *,
    dp_degree: int,
    rank: int,
    strategy: BatchStrategy = BatchStrategy.AUTO,
) -> BatchAllocation:
    """Allocate the local rank's slice of a data-parallel batch.

    Parameters
    ----------
    data:
        The full batch, as a list of dicts matching the worker's
        collate contract (``input_ids``, optional ``labels``).
    dp_degree:
        Number of DP ranks. Pass 1 on single-GPU workers — the
        function just echoes ``data`` back unchanged.
    rank:
        This caller's DP rank, 0-indexed. Must be in
        ``[0, dp_degree)``.
    strategy:
        AUTO, REPLICATE, or SPLIT. See the module docstring.

    Raises
    ------
    BatchAllocationError
        If the inputs are inconsistent (``rank >= dp_degree``,
        ``dp_degree < 1``, empty batch, SPLIT requested on a batch
        that doesn't divide evenly).
    """
    if dp_degree < 1:
        raise BatchAllocationError(f"dp_degree must be >= 1, got {dp_degree}")
    if not 0 <= rank < dp_degree:
        raise BatchAllocationError(f"rank {rank} out of range for dp_degree {dp_degree}")
    if not data:
        raise BatchAllocationError("cannot allocate an empty batch")

    total = len(data)

    # dp_degree == 1 fast path — strategy is irrelevant.
    if dp_degree == 1:
        return BatchAllocation(
            data=list(data),
            total_items=total,
            local_items=total,
            strategy=BatchStrategy.REPLICATE,
            replicated=True,
            wasted_compute_pct=0.0,
        )

    resolved = strategy
    if resolved == BatchStrategy.AUTO:
        resolved = BatchStrategy.SPLIT if total % dp_degree == 0 else BatchStrategy.REPLICATE

    if resolved == BatchStrategy.REPLICATE:
        return BatchAllocation(
            data=list(data),
            total_items=total,
            local_items=total,
            strategy=BatchStrategy.REPLICATE,
            replicated=True,
            wasted_compute_pct=(dp_degree - 1) / dp_degree,
        )

    if resolved == BatchStrategy.SPLIT:
        if total % dp_degree != 0:
            raise BatchAllocationError(
                f"SPLIT requires batch_size ({total}) to be divisible "
                f"by dp_degree ({dp_degree}). Use AUTO or REPLICATE "
                f"if your batch is too small, or pad your dataloader."
            )
        per_rank = total // dp_degree
        start = rank * per_rank
        end = start + per_rank
        return BatchAllocation(
            data=list(data[start:end]),
            total_items=total,
            local_items=per_rank,
            strategy=BatchStrategy.SPLIT,
            replicated=False,
            wasted_compute_pct=0.0,
        )

    raise BatchAllocationError(f"unknown strategy: {strategy!r}")


def replicate_batch_across_dp(data: list[dict], *, dp_degree: int, rank: int) -> BatchAllocation:
    """Convenience shortcut: force REPLICATE on a batch."""
    return prepare_batch_for_dp(
        data, dp_degree=dp_degree, rank=rank, strategy=BatchStrategy.REPLICATE
    )


def split_batch_across_dp(data: list[dict], *, dp_degree: int, rank: int) -> BatchAllocation:
    """Convenience shortcut: force SPLIT on a batch. Raises if not aligned."""
    return prepare_batch_for_dp(data, dp_degree=dp_degree, rank=rank, strategy=BatchStrategy.SPLIT)
