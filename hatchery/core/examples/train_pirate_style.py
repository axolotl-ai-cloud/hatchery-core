# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SFT smoke test for chat-style transfer on pirate Ultrachat."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

from hatchery.core.client import HatcheryClient
from hatchery.core.examples.train_sft import (
    default_tokenizer_model,
    shifted_completion_labels,
)


def load_messages(max_examples: int) -> list[list[dict[str, str]]]:
    from datasets import load_dataset

    ds = load_dataset("winglian/pirate-ultrachat-10k", split="train")
    examples = []
    for row in ds:
        messages = row["messages"]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            continue
        if not messages[-1].get("content", "").strip():
            continue
        examples.append(messages)
        if len(examples) >= max_examples:
            break
    return examples


def tokenize_chat(tok, messages: list[dict[str, str]], *, enable_reasoning: bool = False) -> dict:
    prompt_messages = messages[:-1]
    full_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_reasoning,
    )
    prompt_text = tok.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_reasoning,
    )
    full_ids = tok.encode(full_text, add_special_tokens=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    input_ids, labels = shifted_completion_labels(full_ids, len(prompt_ids))
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_ids}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": labels, "shape": [len(labels)]},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT pirate style transfer smoke test")
    parser.add_argument(
        "--base-url", default=os.environ.get("HATCHERY_BASE_URL", "http://127.0.0.1:8420")
    )
    parser.add_argument("--token", default=os.environ.get("HATCHERY_API_KEY", "dev"))
    parser.add_argument("--base-model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--enable-reasoning", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer_model = args.tokenizer_model or default_tokenizer_model(args.base_model)
    tok = AutoTokenizer.from_pretrained(tokenizer_model)

    messages = load_messages(args.steps)
    data = [tokenize_chat(tok, m, enable_reasoning=args.enable_reasoning) for m in messages]
    n_batches = (len(data) + args.batch_size - 1) // args.batch_size
    print(f"dataset: {len(data)} examples")
    print(f"base_model: {args.base_model}")
    print(f"tokenizer_model: {tokenizer_model}")
    print(f"steps: {n_batches}")

    client = HatcheryClient(base_url=args.base_url, token=args.token, timeout=300)
    try:
        tc = client.create_lora_training_client(base_model=args.base_model, rank=args.rank)
        print(f"session {tc.session_id}")
        for step in range(n_batches):
            batch = data[step * args.batch_size : (step + 1) * args.batch_size]
            fb = tc.forward_backward(batch).result(timeout=120)
            tc.optim_step(learning_rate=args.lr).result(timeout=60)
            loss = fb.get("loss", fb.get("metrics", {}).get("loss:mean", 0))
            if step % 5 == 0 or step == n_batches - 1:
                print(f"  step {step + 1:>3}/{n_batches}  loss={loss:.4f}")

        ckpt = tc.save_weights("pirate-style-checkpoint").result(timeout=60)
        print(f"checkpoint: {ckpt.get('path', ckpt)}")

        prompts = [
            "Explain why regular backups are important.",
            "Write a short welcome message for a new teammate.",
            "Summarize why exercise is healthy.",
        ]
        print("\nInference:")
        for prompt in prompts:
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_text = tok.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_reasoning,
            )
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            result = tc.sample(
                prompt_ids,
                max_tokens=96,
                temperature=0.0,
                stop=[tok.eos_token] if tok.eos_token else [],
            ).result(timeout=30)
            seqs = result.get("sequences", [[]])
            gen_ids = seqs[0] if seqs else []
            if isinstance(gen_ids, dict):
                gen_ids = gen_ids.get("tokens", gen_ids.get("input_ids", []))
            text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            print(f"  prompt: {prompt}")
            print(f"  output: {text!r}")
        return 0
    finally:
        with contextlib.suppress(Exception):
            client.close()


if __name__ == "__main__":
    sys.exit(main())
