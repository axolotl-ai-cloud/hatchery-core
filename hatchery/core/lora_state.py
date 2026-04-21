# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""LoRA state persistence (bf16 snapshot implementation).

LoRA state dicts are the dominant per-step bandwidth to the object
store on this platform: for a 7B LoRA at rank 32 we push tens of
megabytes every ``forward_backward``. This module ships a simple,
correct persister that writes a full bf16 snapshot on every save.

Alternative persisters that layer compression on top of the same
on-disk layout (e.g., a quantized-delta scheme) can be dropped in via
``Config.lora_state``; the shared types below define the contract.

Layout
------
Per-session live state lives under ``sessions/{sid}/live_state``:

* ``lora_weights.pt`` — full bf16 snapshot, always the current live
  state when this persister is in use.

* ``session_meta.json`` — carries ``snapshot_version`` and
  ``delta_count`` bookkeeping fields that the worker threads back
  through ``save`` / ``load`` calls. In the snapshot-only path here
  ``snapshot_version`` increments on every save and ``delta_count``
  stays at 0; implementations that write per-save auxiliary artifacts
  can use these fields to decide when to roll a new snapshot.

Shared types
------------
``LoraStateConfig``, ``SaveResult``, ``SNAPSHOT_FILE``, and the meta
helpers (``meta_with_compression`` / ``read_compression_meta`` /
``serializable_meta_for_json``) live here and are the contract any
alternative persister must agree on.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Optional

try:  # pragma: no cover
    import torch
except ImportError:
    torch = None  # type: ignore


# File name for the full snapshot. Any alternative persister that
# layers compression on top of this module should write its snapshot
# to the same filename so the external on-disk layout stays stable.
SNAPSHOT_FILE = "lora_weights.pt"

# Cadence knob used by compression-style persisters that roll a fresh
# snapshot every N saves. Kept here so ``LoraStateConfig`` is a single
# shared type regardless of which persister is active; the bf16
# persister ignores it.
DEFAULT_SNAPSHOT_EVERY = 16


@dataclass
class LoraStateConfig:
    """Policy knobs shared by all persisters.

    ``snapshot_every`` only applies to compression-style persisters
    that roll a new snapshot every N saves; the bf16 persister always
    writes a snapshot and ignores this field. ``enable_delta`` is an
    escape hatch for forcing the snapshot-only path even when a
    compression persister is configured — useful for tests that want
    an exact round trip.
    """

    snapshot_every: int = DEFAULT_SNAPSHOT_EVERY
    enable_delta: bool = True


@dataclass
class SaveResult:
    """What happened during a save — used by the worker to update its
    in-RAM bookkeeping and surface metrics.
    """

    wrote_snapshot: bool
    wrote_delta: bool
    snapshot_version: int
    delta_count: int
    snapshot_bytes: int = 0
    delta_bytes: int = 0


# ─── internal helpers ──────────────────────────────────────────────


def _state_to_bf16(state: dict) -> dict:
    return {k: v.detach().to(torch.bfloat16).cpu() for k, v in state.items()}


def _torch_save(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _torch_load(b: bytes) -> Any:
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


# ─── Bf16SnapshotPersister ─────────────────────────────────────────


@dataclass
class Bf16SnapshotPersister:
    """Write the full bf16 state dict on every save.

    This is the default persister. It has no delta logic — every save
    overwrites ``lora_weights.pt`` with a fresh bf16 dump of the
    current adapter state, and every load reads that file straight
    back. The ``snapshot_version`` counter is incremented on every
    save so external tools that track the counter keep working across
    alternative persister implementations.
    """

    async def save(
        self,
        objects: Any,
        prefix: str,
        current_state: dict,
        *,
        snapshot_cache: Optional[dict],
        snapshot_version: int,
        delta_count: int,
        cfg: LoraStateConfig,
    ) -> tuple[SaveResult, dict]:
        """Write a full bf16 snapshot to ``{prefix}/lora_weights.pt``."""
        new_version = snapshot_version + 1
        new_snapshot = _state_to_bf16(current_state)
        blob = _torch_save(
            {
                "version": new_version,
                "dtype": "bfloat16",
                "state": new_snapshot,
            }
        )
        await objects.put(f"{prefix}/{SNAPSHOT_FILE}", blob)
        return (
            SaveResult(
                wrote_snapshot=True,
                wrote_delta=False,
                snapshot_version=new_version,
                delta_count=0,
                snapshot_bytes=len(blob),
            ),
            new_snapshot,
        )

    async def load(
        self,
        objects: Any,
        prefix: str,
    ) -> tuple[dict, dict, int, int]:
        """Read ``{prefix}/lora_weights.pt`` and return the state + cache."""
        blob = await objects.get(f"{prefix}/{SNAPSHOT_FILE}")
        obj = _torch_load(blob)

        if isinstance(obj, dict) and "state" in obj and "version" in obj:
            snapshot_state = obj["state"]
            snapshot_version = int(obj["version"])
        else:
            # Bare state dict format (no envelope) without envelope. Treat
            # it as version 0 — it'll be re-envelope on the next save.
            snapshot_state = obj
            snapshot_version = 0

        snapshot_bf16 = {k: v.to(torch.bfloat16) for k, v in snapshot_state.items()}
        reconstructed = {k: v.to(torch.float32) for k, v in snapshot_bf16.items()}
        return reconstructed, snapshot_bf16, snapshot_version, 0

    async def materialize(
        self,
        objects: Any,
        src_prefix: str,
        dst_prefix: str,
    ) -> int:
        """Copy the snapshot file from ``src_prefix`` to ``dst_prefix``.

        With no delta layer there is nothing to flatten — a direct
        object-store copy is sufficient and avoids the encode/decode
        round trip. Returns the size of the copied blob.
        """
        src = f"{src_prefix}/{SNAPSHOT_FILE}"
        dst = f"{dst_prefix}/{SNAPSHOT_FILE}"
        # Some object stores expose ``copy``; fall back to a read +
        # write for any that don't.
        copy = getattr(objects, "copy", None)
        if copy is not None:
            await copy(src, dst)
            try:
                return await objects.size(src)  # type: ignore[attr-defined]
            except (AttributeError, KeyError):
                pass
        blob = await objects.get(src)
        await objects.put(dst, blob)
        return len(blob)


# ─── Shared meta helpers ───────────────────────────────────────────


def meta_with_compression(meta: dict, snapshot_version: int, delta_count: int) -> dict:
    """Merge persister bookkeeping into a ``session_meta.json`` payload.

    The same key names are used by both persisters so workers don't
    need to branch on which backend is configured.
    """
    out = dict(meta)
    out["snapshot_version"] = snapshot_version
    out["delta_count"] = delta_count
    return out


def read_compression_meta(meta: dict) -> tuple[int, int]:
    """Inverse of :func:`meta_with_compression`. Returns ``(version,
    delta_count)`` with sane defaults for meta files that pre-date the
    bookkeeping fields.
    """
    return int(meta.get("snapshot_version", 0)), int(meta.get("delta_count", 0))


def serializable_meta_for_json(meta: dict) -> dict:
    """Filter a meta dict down to JSON-serializable entries. The worker
    stores ``lora_config`` (a dataclass) in ``meta`` which is handled
    separately — this helper just ensures numeric/string fields land
    in the JSON output cleanly.
    """
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            out[k] = v
    return out


# ─── bytes-level helpers (for tests and size accounting) ──────────


def dumps_state_fp32(state: dict) -> bytes:
    return _torch_save({k: v.detach().to(torch.float32).cpu() for k, v in state.items()})


def dumps_state_bf16(state: dict) -> bytes:
    return _torch_save(_state_to_bf16(state))
