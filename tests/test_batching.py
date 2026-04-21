# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for hatchery.core.batching.

Covers every strategy × every edge case. No torch, no dist, no GPU —
just arithmetic on Python lists, so the suite finishes in ms.
"""

from __future__ import annotations

import pytest

from hatchery.core.batching import (
    BatchAllocationError,
    BatchStrategy,
    prepare_batch_for_dp,
    replicate_batch_across_dp,
    split_batch_across_dp,
)


def _data(n: int) -> list[dict]:
    return [{"input_ids": [i, i + 1, i + 2]} for i in range(n)]


# ─── dp_degree == 1 fast path ───────────────────────────────────────────


def test_single_rank_is_passthrough():
    data = _data(3)
    alloc = prepare_batch_for_dp(data, dp_degree=1, rank=0)
    assert alloc.data == data
    assert alloc.strategy == BatchStrategy.REPLICATE
    assert alloc.replicated
    assert alloc.wasted_compute_pct == 0.0
    assert alloc.total_items == 3
    assert alloc.local_items == 3


def test_single_rank_ignores_strategy():
    data = _data(5)
    for strategy in (BatchStrategy.AUTO, BatchStrategy.REPLICATE, BatchStrategy.SPLIT):
        alloc = prepare_batch_for_dp(data, dp_degree=1, rank=0, strategy=strategy)
        assert alloc.local_items == 5


# ─── REPLICATE ───────────────────────────────────────────────────────────


def test_replicate_gives_every_rank_full_batch():
    data = _data(2)
    for rank in range(4):
        alloc = prepare_batch_for_dp(data, dp_degree=4, rank=rank, strategy=BatchStrategy.REPLICATE)
        assert alloc.data == data
        assert alloc.local_items == 2
        assert alloc.total_items == 2
        assert alloc.replicated is True
        assert alloc.strategy == BatchStrategy.REPLICATE
        assert alloc.wasted_compute_pct == pytest.approx(0.75)


def test_replicate_data_is_copy_not_reference():
    data = _data(3)
    alloc = prepare_batch_for_dp(data, dp_degree=2, rank=0, strategy=BatchStrategy.REPLICATE)
    # Mutating the allocation's data must not poison the caller's list.
    alloc.data.append({"input_ids": [999]})
    assert len(data) == 3


def test_replicate_even_when_batch_is_divisible():
    # REPLICATE always replicates, even if SPLIT would work.
    data = _data(4)
    alloc = prepare_batch_for_dp(data, dp_degree=4, rank=2, strategy=BatchStrategy.REPLICATE)
    assert len(alloc.data) == 4
    assert alloc.replicated is True


def test_replicate_convenience_function():
    data = _data(2)
    alloc = replicate_batch_across_dp(data, dp_degree=4, rank=0)
    assert alloc.strategy == BatchStrategy.REPLICATE
    assert alloc.replicated
    assert alloc.data == data


# ─── SPLIT ───────────────────────────────────────────────────────────────


def test_split_divides_evenly():
    data = _data(8)
    rank_allocations = [
        prepare_batch_for_dp(data, dp_degree=4, rank=r, strategy=BatchStrategy.SPLIT)
        for r in range(4)
    ]
    # Every rank got exactly 2 items.
    for alloc in rank_allocations:
        assert alloc.local_items == 2
        assert alloc.total_items == 8
        assert not alloc.replicated
        assert alloc.wasted_compute_pct == 0.0
    # Concatenating the slices rebuilds the original batch exactly.
    combined = sum((a.data for a in rank_allocations), [])
    assert combined == data


def test_split_with_indivisible_raises():
    with pytest.raises(BatchAllocationError, match="divisible by dp_degree"):
        prepare_batch_for_dp(_data(7), dp_degree=4, rank=0, strategy=BatchStrategy.SPLIT)


def test_split_on_batch_of_one_rank_count_raises():
    # Batch=1, dp=4 → not divisible → error.
    with pytest.raises(BatchAllocationError):
        prepare_batch_for_dp(_data(1), dp_degree=4, rank=0, strategy=BatchStrategy.SPLIT)


def test_split_convenience_function():
    data = _data(4)
    alloc = split_batch_across_dp(data, dp_degree=2, rank=1)
    assert alloc.data == data[2:4]
    assert alloc.strategy == BatchStrategy.SPLIT


def test_split_into_one_item_per_rank():
    data = _data(4)
    for r in range(4):
        alloc = prepare_batch_for_dp(data, dp_degree=4, rank=r, strategy=BatchStrategy.SPLIT)
        assert alloc.local_items == 1
        assert alloc.data == [data[r]]


# ─── AUTO ────────────────────────────────────────────────────────────────


def test_auto_chooses_split_when_divisible():
    data = _data(8)
    alloc = prepare_batch_for_dp(data, dp_degree=4, rank=0)  # AUTO default
    assert alloc.strategy == BatchStrategy.SPLIT
    assert alloc.local_items == 2
    assert not alloc.replicated


def test_auto_falls_back_to_replicate_when_indivisible():
    data = _data(3)
    alloc = prepare_batch_for_dp(data, dp_degree=4, rank=0)  # AUTO default
    assert alloc.strategy == BatchStrategy.REPLICATE
    assert alloc.local_items == 3
    assert alloc.replicated


def test_auto_small_batch_replicates_across_every_rank():
    data = _data(1)
    for r in range(8):
        alloc = prepare_batch_for_dp(data, dp_degree=8, rank=r)
        assert alloc.strategy == BatchStrategy.REPLICATE
        assert alloc.data == data


def test_auto_preserves_item_order_across_ranks():
    """Under AUTO+SPLIT with batch=N*k, the concatenation of rank slices
    is the original list in order. This is what lets grad averaging
    give the same answer as a single-rank run."""
    data = _data(12)
    slices = [prepare_batch_for_dp(data, dp_degree=3, rank=r).data for r in range(3)]
    assert slices[0] == data[0:4]
    assert slices[1] == data[4:8]
    assert slices[2] == data[8:12]


# ─── Input validation ───────────────────────────────────────────────────


def test_rank_out_of_range_raises():
    with pytest.raises(BatchAllocationError, match="out of range"):
        prepare_batch_for_dp(_data(4), dp_degree=2, rank=2)
    with pytest.raises(BatchAllocationError, match="out of range"):
        prepare_batch_for_dp(_data(4), dp_degree=2, rank=-1)


def test_dp_degree_zero_raises():
    with pytest.raises(BatchAllocationError, match=">= 1"):
        prepare_batch_for_dp(_data(4), dp_degree=0, rank=0)


def test_negative_dp_degree_raises():
    with pytest.raises(BatchAllocationError, match=">= 1"):
        prepare_batch_for_dp(_data(4), dp_degree=-1, rank=0)


def test_empty_batch_raises():
    with pytest.raises(BatchAllocationError, match="empty batch"):
        prepare_batch_for_dp([], dp_degree=4, rank=0)


def test_unknown_strategy_string_raises():
    """If a user passes a raw string that isn't a valid enum member,
    the enum constructor fails. This guards against typos like
    ``strategy='replicated'`` (note the trailing d).
    """
    with pytest.raises(ValueError):
        BatchStrategy("replicated")


# ─── Wasted compute accounting ──────────────────────────────────────────


def test_wasted_compute_pct_scales_with_dp_degree():
    data = _data(2)
    for dp in (2, 4, 8, 16):
        alloc = prepare_batch_for_dp(data, dp_degree=dp, rank=0, strategy=BatchStrategy.REPLICATE)
        expected = (dp - 1) / dp
        assert alloc.wasted_compute_pct == pytest.approx(expected)


def test_split_has_zero_wasted_compute():
    data = _data(16)
    alloc = prepare_batch_for_dp(data, dp_degree=4, rank=2, strategy=BatchStrategy.SPLIT)
    assert alloc.wasted_compute_pct == 0.0


# ─── Gradient correctness property ──────────────────────────────────────


def test_split_preserves_item_identity():
    """Mathematical property: if we sum item-level gradients via the
    split-then-average scheme, we get the same result as operating on
    the full batch. This is the precondition that makes SPLIT correct.

    We simulate gradients as integer indices so we can exactly verify
    the arithmetic holds without any floating-point noise.
    """
    data = _data(8)
    dp = 4
    # Simulate: each rank's "local gradient" = sum of its items' indices.
    # Sum over ranks divided by dp must equal sum(all items) / batch.
    per_rank_sums = []
    for r in range(dp):
        alloc = prepare_batch_for_dp(data, dp_degree=dp, rank=r, strategy=BatchStrategy.SPLIT)
        per_rank_grad = sum(d["input_ids"][0] for d in alloc.data) / alloc.local_items
        per_rank_sums.append(per_rank_grad)
    split_result = sum(per_rank_sums) / dp

    # Reference: compute the "full batch gradient" directly.
    full_result = sum(d["input_ids"][0] for d in data) / len(data)
    assert split_result == pytest.approx(full_result)


def test_replicate_preserves_item_identity():
    """Same math check for REPLICATE: every rank computes the full-batch
    gradient; averaging across ranks gives the same full-batch value.
    """
    data = _data(8)
    dp = 4
    per_rank_sums = []
    for r in range(dp):
        alloc = prepare_batch_for_dp(data, dp_degree=dp, rank=r, strategy=BatchStrategy.REPLICATE)
        per_rank_grad = sum(d["input_ids"][0] for d in alloc.data) / alloc.local_items
        per_rank_sums.append(per_rank_grad)
    replicate_result = sum(per_rank_sums) / dp

    full_result = sum(d["input_ids"][0] for d in data) / len(data)
    assert replicate_result == pytest.approx(full_result)


def test_split_and_replicate_match_when_batch_is_aligned():
    """The strong property: REPLICATE and SPLIT produce the same
    gradient when the batch divides cleanly. If this ever regresses,
    one of them has a bug."""
    data = _data(12)
    dp = 3

    split_grad = 0.0
    for r in range(dp):
        a = prepare_batch_for_dp(data, dp_degree=dp, rank=r, strategy=BatchStrategy.SPLIT)
        split_grad += sum(d["input_ids"][0] for d in a.data) / a.local_items
    split_grad /= dp

    rep_grad = 0.0
    for r in range(dp):
        a = prepare_batch_for_dp(data, dp_degree=dp, rank=r, strategy=BatchStrategy.REPLICATE)
        rep_grad += sum(d["input_ids"][0] for d in a.data) / a.local_items
    rep_grad /= dp

    assert split_grad == pytest.approx(rep_grad)
