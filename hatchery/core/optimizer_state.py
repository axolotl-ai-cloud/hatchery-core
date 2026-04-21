# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Optimizer state persistence (full-snapshot implementation).

AdamW optimizer state (``exp_avg`` + ``exp_avg_sq`` tensors, roughly 2×
the LoRA weight footprint) is the second-largest per-step object-store
write after LoRA weights. This module ships a correct baseline that
torch-saves the full dict on every save.

Alternative persisters that layer compression on top of the same
on-disk layout (e.g., blockwise int8 delta against a periodic bf16
snapshot) can be dropped in via ``Config.optimizer_state``.

Layout
------
Per-session live state lives under ``sessions/{sid}/live_state``:

* ``optimizer_state.pt`` — the full dict written by PyTorch's
  ``optimizer.state_dict()``. When a compression persister is in use
  this filename still holds the canonical snapshot.
* ``optimizer_delta.pt`` *(optional, compression persisters only)* —
  per-save quantized delta against the snapshot.

Meta bookkeeping
----------------
Workers thread ``optim_snapshot_version`` and ``optim_delta_count``
through ``save``/``load`` via ``session_meta.json``. Snapshot-only
implementations can ignore them; the counters still increment on
every save so external tools observing the meta see a monotonic
version even if no delta file is ever written.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Optional

try:  # pragma: no cover
    import torch
except ImportError:
    torch = None  # type: ignore


OPTIMIZER_SNAPSHOT_FILE = "optimizer_state.pt"
OPTIMIZER_DELTA_FILE = "optimizer_delta.pt"

DEFAULT_OPTIM_SNAPSHOT_EVERY = 16


@dataclass
class OptimizerStateConfig:
    """Policy knobs shared by all optimizer persisters."""

    snapshot_every: int = DEFAULT_OPTIM_SNAPSHOT_EVERY
    enable_delta: bool = True


@dataclass
class OptimizerSaveResult:
    """What happened during an optimizer save."""

    wrote_snapshot: bool
    wrote_delta: bool
    snapshot_version: int
    delta_count: int
    snapshot_bytes: int = 0
    delta_bytes: int = 0


# ─── internal helpers ──────────────────────────────────────────────


def _torch_save(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _torch_load(b: bytes) -> Any:
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


# ─── FullOptimizerPersister ────────────────────────────────────────


@dataclass
class FullOptimizerPersister:
    """Write the full optimizer ``state_dict`` on every save.

    This is the default persister — equivalent to the pre-existing
    ``torch.save(optimizer.state_dict(), path)`` hot path. It handles
    empty/None state by writing nothing.
    """

    async def save(
        self,
        objects: Any,
        prefix: str,
        current_state: Optional[dict],
        *,
        snapshot_cache: Optional[dict],
        snapshot_version: int,
        delta_count: int,
        cfg: OptimizerStateConfig,
    ) -> tuple[OptimizerSaveResult, Optional[dict]]:
        if current_state is None:
            return (
                OptimizerSaveResult(
                    wrote_snapshot=False,
                    wrote_delta=False,
                    snapshot_version=snapshot_version,
                    delta_count=delta_count,
                ),
                snapshot_cache,
            )
        blob = _torch_save(current_state)
        await objects.put(f"{prefix}/{OPTIMIZER_SNAPSHOT_FILE}", blob)
        return (
            OptimizerSaveResult(
                wrote_snapshot=True,
                wrote_delta=False,
                snapshot_version=snapshot_version + 1,
                delta_count=0,
                snapshot_bytes=len(blob),
            ),
            None,
        )

    async def load(
        self,
        objects: Any,
        prefix: str,
    ) -> tuple[Optional[dict], Optional[dict], int, int]:
        """Read ``optimizer_state.pt``.

        Returns ``(state, snapshot_cache, snapshot_version, delta_count)``.
        If the file is missing, returns ``(None, None, 0, 0)`` — the
        session has not yet run an optim_step.
        """
        try:
            blob = await objects.get(f"{prefix}/{OPTIMIZER_SNAPSHOT_FILE}")
        except KeyError:
            return None, None, 0, 0
        state = _torch_load(blob)
        return state, None, 0, 0


__all__ = [
    "OPTIMIZER_SNAPSHOT_FILE",
    "OPTIMIZER_DELTA_FILE",
    "DEFAULT_OPTIM_SNAPSHOT_EVERY",
    "OptimizerStateConfig",
    "OptimizerSaveResult",
    "FullOptimizerPersister",
]
