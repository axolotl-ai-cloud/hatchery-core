# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Sequence packing (varlen) v1.

Concatenates multiple short training examples into one long sequence
so the attention kernel can skip the padding that a rectangular
``[B, T]`` batch would otherwise carry. Document boundaries are
communicated to the model via ``position_ids`` that reset at each
boundary — HF's flash-attention-2 path derives ``cu_seqlens`` from
those resets and masks cross-document attention accordingly.

This module is pure Python + torch — no transformers / PEFT / CUDA
dependencies. The trainer calls :func:`pack_sequences` on its
collate path; after the forward pass it calls :func:`unpack_outputs`
to split per-token outputs back into per-datum pieces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:  # pragma: no cover
    import torch
except ImportError:
    torch = None  # type: ignore


@dataclass
class PackedBatch:
    """One packed row ready to feed into a causal LM.

    Every field is 1-D over the flattened token axis except
    ``cu_seqlens`` (length N+1 for N docs) and ``slices``
    (one ``(start, end)`` per input datum).
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: list[int]
    slices: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return int(self.input_ids.shape[-1])

    @property
    def num_docs(self) -> int:
        return len(self.slices)


def pack_sequences(
    items: list[dict],
    pad_id: int,
    max_packed_len: int,
    *,
    label_boundary_mask: bool = True,
) -> list[PackedBatch]:
    """First-fit-decreasing bin-pack ``items`` into one or more packs.

    Each item is a dict with ``input_ids`` (required) and optional
    ``labels``. Items longer than ``max_packed_len`` occupy their own
    pack of one (no truncation — the caller decided their input was
    valid, we don't get to drop tokens silently).

    ``label_boundary_mask`` — when True, the first label of each
    document is set to -100 so the causal-shift loss never asks the
    model to predict token 0 of document k+1 from token T-1 of
    document k. This matches the masking HF's trainer applies when
    packing via ``DataCollatorForLanguageModeling``.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("pack_sequences requires torch")
    if max_packed_len <= 0:
        raise ValueError(f"max_packed_len must be > 0, got {max_packed_len}")
    if not items:
        return []

    # Indexed + sorted descending by length for FFD.
    indexed = list(enumerate(items))
    indexed.sort(key=lambda p: len(p[1]["input_ids"]), reverse=True)

    # Each bin is a list of (orig_index, item); we lay them out in
    # original-item order inside a pack but the FFD decision uses
    # sorted order.
    bins: list[list[tuple[int, dict]]] = []
    bin_lens: list[int] = []
    for orig_idx, item in indexed:
        ln = len(item["input_ids"])
        if ln > max_packed_len:
            bins.append([(orig_idx, item)])
            bin_lens.append(ln)
            continue
        placed = False
        for b, cur_len in enumerate(bin_lens):
            if cur_len + ln <= max_packed_len:
                bins[b].append((orig_idx, item))
                bin_lens[b] = cur_len + ln
                placed = True
                break
        if not placed:
            bins.append([(orig_idx, item)])
            bin_lens.append(ln)

    packs: list[PackedBatch] = []
    for members in bins:
        # Restore original order inside the pack so the caller's
        # output is predictable (slices[i] matches items[i] of the
        # subset — but since we stored original indices, we preserve
        # the caller's item ordering where possible).
        members.sort(key=lambda p: p[0])
        packs.append(_build_pack(members, pad_id, label_boundary_mask))
    return packs


def _build_pack(
    members: list[tuple[int, dict]],
    pad_id: int,
    label_boundary_mask: bool,
) -> PackedBatch:
    flat_ids: list[int] = []
    flat_labels: list[int] = []
    flat_pos: list[int] = []
    cu: list[int] = [0]
    slices: list[tuple[int, int]] = []

    offset = 0
    for _orig_idx, item in members:
        ids = list(item["input_ids"])
        lbls = list(item.get("labels", ids))
        if len(lbls) != len(ids):
            raise ValueError(f"labels length {len(lbls)} != input_ids length {len(ids)}")
        if label_boundary_mask and lbls:
            # First label of each document is unreachable by the
            # causal shift from the *previous* document, so masking
            # it is belt-and-suspenders relative to position_ids
            # resets. Leave it on by default — cheap and eliminates
            # a class of off-by-one bugs at doc boundaries.
            lbls = [-100, *lbls[1:]]
        flat_ids.extend(ids)
        flat_labels.extend(lbls)
        flat_pos.extend(range(len(ids)))
        end = offset + len(ids)
        slices.append((offset, end))
        cu.append(end)
        offset = end

    _ = pad_id  # reserved — v1 doesn't pad within a pack, but future
    # variants may left-pad to a multiple of 8/16 for tensor-core
    # alignment. Keeping the parameter in the signature so callers
    # don't have to change when that lands.

    return PackedBatch(
        input_ids=torch.tensor([flat_ids], dtype=torch.long),
        labels=torch.tensor([flat_labels], dtype=torch.long),
        position_ids=torch.tensor([flat_pos], dtype=torch.long),
        cu_seqlens=cu,
        slices=slices,
    )


def unpack_outputs(packed: torch.Tensor, slices: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Split a packed token-axis tensor back into per-datum pieces.

    Accepts either a 1-D per-token tensor ``[T]`` (e.g. per-token
    logprobs) or a 2-D ``[T, V]`` logits tensor. A leading batch
    dim of size 1 is squeezed first — trainers that held onto the
    ``[1, T, ...]`` shape can still pass it through unchanged.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("unpack_outputs requires torch")
    if packed.dim() >= 2 and packed.shape[0] == 1:
        packed = packed.squeeze(0)
    out: list[torch.Tensor] = []
    for start, end in slices:
        out.append(packed[start:end])
    return out


def should_pack(
    items: list[dict],
    *,
    max_packed_len: Optional[int],
    min_items: int = 2,
) -> bool:
    """Cheap heuristic: is packing worth it for this batch?

    Returns False for trivially-small batches (a single item, or
    below ``min_items``) and when the total length would overflow
    ``max_packed_len`` by so much that we'd end up with one pack per
    item anyway. Trainer-side quick-reject so we don't pay the
    bookkeeping cost when there's no pad to recover.
    """
    if max_packed_len is None or max_packed_len <= 0:
        return False
    if len(items) < min_items:
        return False
    lens = [len(it["input_ids"]) for it in items]
    return max(lens) < max_packed_len and sum(lens) >= max(lens) * 2
