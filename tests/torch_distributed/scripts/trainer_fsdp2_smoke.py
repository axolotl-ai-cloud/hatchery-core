# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Standalone torchrun smoke test for VanillaTrainer with core FSDP2 DP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from hatchery.core.distributed import destroy_distributed_runtime
from hatchery.core.parallel import ParallelConfig
from hatchery.core.trainer import LoraSpec, VanillaTrainer


class _TokenizerStub:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, sort_keys=True)


def _build_trainer() -> VanillaTrainer:
    torch.manual_seed(11)
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    parallel = ParallelConfig(dp_degree=2, batch_strategy="replicate")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    model = GPT2LMHeadModel(cfg).to(device=device, dtype=torch.bfloat16)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    trainer = VanillaTrainer(
        base_model_name="tiny-local-gpt2",
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        parallel=parallel,
        load_model=False,
    )
    trainer._raw_base = model
    trainer._pristine_base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trainer.tokenizer = _TokenizerStub()
    return trainer


def _build_hf_trainer(base_model: str) -> VanillaTrainer:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    return VanillaTrainer(
        base_model_name=base_model,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        parallel=ParallelConfig(dp_degree=2, batch_strategy="replicate"),
    )


def _build_data(base_model: str | None) -> list[dict]:
    if base_model is None:
        return [
            {"input_ids": [2, 3, 4, 5, 6, 7, 8, 9], "labels": [2, 3, 4, 5, 6, 7, 8, 9]},
            {"input_ids": [9, 8, 7, 6, 5, 4, 3, 2], "labels": [9, 8, 7, 6, 5, 4, 3, 2]},
        ]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prompts = [
        "Hatchery trains adapters with data parallel FSDP.",
        "A small validation batch should reduce loss after optimizer steps.",
    ]
    data = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ids = ids[:32]
        data.append({"input_ids": ids, "labels": ids})
    return data


def _mean_loss(loss: float) -> float:
    local_rank = int(os.environ["LOCAL_RANK"])
    tensor = torch.tensor(float(loss), device=f"cuda:{local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()

    trainer = None
    try:
        torch.cuda.manual_seed_all(11)
        trainer = _build_hf_trainer(args.base_model) if args.base_model else _build_trainer()
        target_modules = (
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if args.base_model
            else ["c_attn"]
        )
        spec = LoraSpec(rank=4, lora_alpha=8, target_modules=target_modules)
        state = trainer.init_session_state("fsdp2-smoke", spec)
        trainer.load_state("fsdp2-smoke", state)

        data = _build_data(args.base_model)
        pre_sample = trainer.sample(
            "fsdp2-smoke",
            data[0]["input_ids"][:4],
            {"max_tokens": 2, "temperature": 0.0},
        )
        losses: list[float] = []
        for _ in range(args.steps):
            result = trainer.forward_backward("fsdp2-smoke", data, "cross_entropy")
            losses.append(_mean_loss(result.loss))
            trainer.optim_step(
                "fsdp2-smoke",
                {
                    "learning_rate": 1e-2,
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            )
        post_sample = trainer.sample(
            "fsdp2-smoke",
            data[0]["input_ids"][:4],
            {"max_tokens": 2, "temperature": 0.0},
        )

        runtime = trainer._distributed_runtime
        if runtime.global_rank == 0:
            _write_json(
                args.out,
                {
                    "status": "ok",
                    "world_size": runtime.world_size,
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                    "losses": losses,
                    "losses_decreased": losses[-1] < losses[0],
                    "pre_sample_tokens": pre_sample.total_tokens,
                    "post_sample_tokens": post_sample.total_tokens,
                },
            )
    except Exception as exc:  # noqa: BLE001
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            _write_json(args.out, {"status": "error", "error": repr(exc)})
    finally:
        runtime = trainer._distributed_runtime if trainer is not None else None
        destroy_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
