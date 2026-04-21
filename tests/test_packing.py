# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit + equivalence tests for hatchery.core.packing.

No CUDA — everything runs on CPU. The equivalence smoke uses a
tiny ``nn.Module`` stand-in so we don't need HF / PEFT weights on
disk. What we're asserting is that the packed layout preserves
per-datum outputs after unpacking, which is exactly what the
trainer's post-forward path relies on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hatchery.core.packing import (  # noqa: E402, I001
    PackedBatch,
    pack_sequences,
    should_pack,
    unpack_outputs,
)


# ─── cu_seqlens / position_ids ───────────────────────────────────────────


def test_cu_seqlens_three_docs():
    items = [
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8]},
        {"input_ids": [9, 10, 11, 12]},
    ]
    packs = pack_sequences(items, pad_id=0, max_packed_len=32)
    assert len(packs) == 1
    pack = packs[0]
    assert pack.cu_seqlens == [0, 5, 8, 12]
    assert pack.total_tokens == 12
    assert pack.num_docs == 3


def test_position_ids_reset_at_each_boundary():
    items = [
        {"input_ids": [10, 11, 12, 13]},
        {"input_ids": [20, 21]},
        {"input_ids": [30, 31, 32]},
    ]
    packs = pack_sequences(items, pad_id=0, max_packed_len=32)
    pack = packs[0]
    pos = pack.position_ids[0].tolist()
    assert pos == [0, 1, 2, 3, 0, 1, 0, 1, 2]


def test_input_ids_concatenated_in_caller_order():
    items = [
        {"input_ids": [1, 1, 1]},
        {"input_ids": [2, 2]},
        {"input_ids": [3, 3, 3, 3]},
    ]
    pack = pack_sequences(items, pad_id=0, max_packed_len=32)[0]
    assert pack.input_ids[0].tolist() == [1, 1, 1, 2, 2, 3, 3, 3, 3]


# ─── Label masking at boundaries ─────────────────────────────────────────


def test_first_label_of_each_doc_is_masked():
    items = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ]
    pack = pack_sequences(items, pad_id=0, max_packed_len=16)[0]
    labels = pack.labels[0].tolist()
    # Doc 0 starts at index 0, doc 1 starts at index 3.
    assert labels[0] == -100
    assert labels[3] == -100
    # Everything else preserved.
    assert labels[1:3] == [2, 3]
    assert labels[4:] == [5]


def test_labels_default_to_input_ids_when_absent():
    items = [{"input_ids": [7, 8, 9]}]
    pack = pack_sequences(items, pad_id=0, max_packed_len=16)[0]
    assert pack.labels[0].tolist() == [-100, 8, 9]


def test_label_boundary_mask_can_be_disabled():
    items = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ]
    pack = pack_sequences(items, pad_id=0, max_packed_len=16, label_boundary_mask=False)[0]
    assert pack.labels[0].tolist() == [1, 2, 3, 4, 5]


def test_mismatched_labels_length_raises():
    items = [{"input_ids": [1, 2, 3], "labels": [1, 2]}]
    with pytest.raises(ValueError, match="labels length"):
        pack_sequences(items, pad_id=0, max_packed_len=16)


# ─── unpack_outputs round-trip ───────────────────────────────────────────


def test_unpack_outputs_roundtrip_1d():
    items = [
        {"input_ids": [10, 11, 12, 13]},
        {"input_ids": [20, 21]},
        {"input_ids": [30, 31, 32]},
    ]
    pack = pack_sequences(items, pad_id=0, max_packed_len=32)[0]
    # Treat input_ids as "per-token output" for the round-trip check.
    per_datum = unpack_outputs(pack.input_ids, pack.slices)
    recovered = [p.tolist() for p in per_datum]
    assert recovered == [[10, 11, 12, 13], [20, 21], [30, 31, 32]]


def test_unpack_outputs_roundtrip_2d_logits():
    items = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5]},
    ]
    pack = pack_sequences(items, pad_id=0, max_packed_len=16)[0]
    T = pack.total_tokens
    V = 7
    fake_logits = torch.arange(T * V, dtype=torch.float32).view(1, T, V)
    pieces = unpack_outputs(fake_logits, pack.slices)
    assert len(pieces) == 2
    assert pieces[0].shape == (3, V)
    assert pieces[1].shape == (2, V)
    # First piece equals the first 3 rows of the unsqueezed logits.
    assert torch.equal(pieces[0], fake_logits[0, 0:3, :])
    assert torch.equal(pieces[1], fake_logits[0, 3:5, :])


# ─── Mixed-length bin packing ────────────────────────────────────────────


def test_mixed_length_bin_packing():
    items = [
        {"input_ids": [0] * 10},
        {"input_ids": [1] * 2},
        {"input_ids": [2] * 3},
        {"input_ids": [3] * 4},
    ]
    packs = pack_sequences(items, pad_id=0, max_packed_len=12)
    # FFD: sort desc [10, 4, 3, 2]. First bin takes the 10 (can fit a
    # 2 later → 12). Next, 4 opens a new bin. 3 goes with the 4 (7).
    # 2 joins the first bin (10+2=12). Final: two bins, sizes 12 and 7.
    assert len(packs) == 2
    sizes = sorted(p.total_tokens for p in packs)
    assert sizes == [7, 12]


def test_item_longer_than_max_becomes_solo_pack():
    items = [
        {"input_ids": [0] * 20},
        {"input_ids": [1] * 3},
    ]
    packs = pack_sequences(items, pad_id=0, max_packed_len=8)
    assert len(packs) == 2
    sizes = sorted(p.total_tokens for p in packs)
    assert sizes == [3, 20]


def test_empty_items_returns_empty():
    assert pack_sequences([], pad_id=0, max_packed_len=16) == []


def test_invalid_max_packed_len_raises():
    with pytest.raises(ValueError, match="max_packed_len"):
        pack_sequences([{"input_ids": [1]}], pad_id=0, max_packed_len=0)


# ─── Equivalence smoke: pad-batch vs packed-batch ─────────────────────────


class _ScriptedLM(torch.nn.Module):
    """Deterministic stand-in for an HF CausalLM.

    Emits per-token "logits" that are just a learned linear projection
    of the one-hot input_ids. No attention — which is exactly the
    point: we're asserting that the packed layout + unpack doesn't
    mangle per-datum outputs, not that attention is correctly masked
    (that's the flash-attn path's responsibility and requires a GPU
    to exercise).
    """

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.proj = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        return self.proj(h)


def test_pack_unpack_matches_padded_forward():
    """Per-datum logits after pack→forward→unpack equal the
    pad-batch forward's per-datum slice (modulo bfloat16 noise —
    this runs in float32).
    """
    model = _ScriptedLM()
    items = [
        {"input_ids": [1, 2, 3, 4]},
        {"input_ids": [5, 6]},
        {"input_ids": [7, 8, 9]},
    ]

    # Pad-batch forward (the current trainer path).
    max_len = max(len(it["input_ids"]) for it in items)
    pad_id = 0
    padded = torch.tensor(
        [it["input_ids"] + [pad_id] * (max_len - len(it["input_ids"])) for it in items],
        dtype=torch.long,
    )
    with torch.no_grad():
        pad_logits = model(padded)  # [B, T, V]

    # Packed forward.
    pack = pack_sequences(items, pad_id=pad_id, max_packed_len=32)[0]
    with torch.no_grad():
        packed_logits = model(pack.input_ids)  # [1, T, V]
    pieces = unpack_outputs(packed_logits, pack.slices)

    for i, item in enumerate(items):
        n = len(item["input_ids"])
        assert torch.allclose(pieces[i], pad_logits[i, :n, :], atol=1e-6)


# ─── PackedBatch dataclass ───────────────────────────────────────────────


def test_packed_batch_fields():
    pack = PackedBatch(
        input_ids=torch.tensor([[1, 2, 3]]),
        labels=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        cu_seqlens=[0, 3],
        slices=[(0, 3)],
    )
    assert pack.total_tokens == 3
    assert pack.num_docs == 1


# ─── should_pack heuristic ───────────────────────────────────────────────


def test_should_pack_rejects_trivial_cases():
    assert should_pack([{"input_ids": [1]}], max_packed_len=16) is False
    assert should_pack([], max_packed_len=16) is False
    assert should_pack([{"input_ids": [1]}] * 3, max_packed_len=None) is False


def test_should_pack_accepts_worthwhile_batch():
    items = [{"input_ids": [0] * n} for n in (4, 3, 5)]
    assert should_pack(items, max_packed_len=32) is True


def test_should_pack_rejects_single_oversize():
    items = [{"input_ids": [0] * 100}, {"input_ids": [0] * 2}]
    assert should_pack(items, max_packed_len=64) is False


# ─── ParallelConfig integration ──────────────────────────────────────────


def test_parallel_config_rejects_packing_with_cp():
    from hatchery.core.parallel import ParallelConfig

    with pytest.raises(ValueError, match="sequence_packing with cp_degree"):
        ParallelConfig(sequence_packing=True, cp_degree=2)


def test_parallel_config_accepts_packing_without_cp():
    from hatchery.core.parallel import ParallelConfig

    cfg = ParallelConfig(sequence_packing=True, max_packed_len=4096)
    assert cfg.sequence_packing is True
    assert cfg.max_packed_len == 4096


def test_parallel_config_rejects_nonpositive_max_packed_len():
    from hatchery.core.parallel import ParallelConfig

    with pytest.raises(ValueError, match="max_packed_len"):
        ParallelConfig(sequence_packing=True, max_packed_len=0)


def test_parallel_config_from_env_reads_packing_vars(monkeypatch):
    from hatchery.core.parallel import ParallelConfig

    monkeypatch.setenv("HATCHERY_SEQUENCE_PACKING", "1")
    monkeypatch.setenv("HATCHERY_MAX_PACKED_LEN", "8192")
    cfg = ParallelConfig.from_env()
    assert cfg.sequence_packing is True
    assert cfg.max_packed_len == 8192
