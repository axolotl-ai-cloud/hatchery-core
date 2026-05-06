# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Standalone torchrun smoke test for core FSDP2 data parallelism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from hatchery.core.distributed import destroy_distributed_runtime, init_distributed_runtime
from hatchery.core.parallel import ParallelConfig


class _Transformer(torch.nn.Module):
    def __init__(self, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([_Block(hidden_size) for _ in range(layers)])


class _Block(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.fc2(torch.nn.functional.gelu(self.fc1(x))))


class _TinyLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 32, hidden_size: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = _Transformer(hidden_size, layers)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(x)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, sort_keys=True)


def _mean_loss(loss: torch.Tensor) -> float:
    reduced = loss.detach().float()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced.cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    runtime = None
    try:
        torch.manual_seed(7)
        runtime = init_distributed_runtime(ParallelConfig(dp_degree=2))
        if runtime.device is None:
            raise RuntimeError("fsdp2_smoke.py requires CUDA under the pytest wrapper")

        from hatchery.core.distributed import apply_core_fsdp2_dp

        device = runtime.device
        torch.cuda.manual_seed_all(7)
        model = _TinyLM().to(device=device, dtype=torch.bfloat16)
        apply_core_fsdp2_dp(model, runtime, ParallelConfig(dp_degree=2))

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2, fused=True)
        input_ids = torch.tensor(
            [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]],
            device=device,
            dtype=torch.long,
        )
        labels = input_ids.clone()

        losses: list[float] = []
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            losses.append(_mean_loss(loss))

        if runtime.global_rank == 0:
            _write_json(
                args.out,
                {
                    "status": "ok",
                    "world_size": runtime.world_size,
                    "mesh_dims": list(runtime.mesh.mesh_dim_names),
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                    "losses": losses,
                },
            )
    except Exception as exc:  # noqa: BLE001
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            _write_json(args.out, {"status": "error", "error": repr(exc)})
    finally:
        destroy_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
