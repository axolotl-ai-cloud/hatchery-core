# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SFT + GRPO on WikiText pig-latin: end-to-end fine-tuning and RL.

Phase 1 (SFT): Fine-tunes a model to translate English → pig-latin
using sentences from WikiText-2 with the tokenizer's chat template.

Phase 2 (GRPO): Samples 4 completions per prompt, scores them with
a character-level reward (edit-distance to the correct pig-latin),
computes advantages, and runs GRPO optimization.

Demonstrates:
- Chat-template tokenization with prompt masking
- Batched SFT training (batch_size=8)
- Multi-sample generation (n=4) for GRPO rollouts
- Server-side GRPO loss with old_logprobs + advantages
- Weights & Biases logging

Usage:

    python -m hatchery.core.local_dev
    python hatchery/core/examples/train_sft.py --steps 100 --rl-steps 25 --lr 1e-3
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys

from hatchery.core.client import HatcheryClient

# ── Pig-latin translation ────────────────────────────────────────────────

VOWELS = set("aeiouAEIOU")


def piglatinize(text: str) -> str:
    words = []
    for w in text.split():
        lo = w.lower()
        if not lo or not lo[0].isalpha():
            words.append(lo)
            continue
        if lo[0] in VOWELS:
            words.append(lo + "way")
        else:
            i = next((i for i, c in enumerate(lo) if c in VOWELS), len(lo))
            words.append(lo[i:] + lo[:i] + "ay")
    return " ".join(words)


# ── Dataset ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "Translate the following English text into pig-latin."


def load_wikitext_sentences(max_examples: int = 200) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sentences: list[str] = []
    for row in ds:
        text = row["text"].strip()
        if not text or text.startswith("="):
            continue
        words = text.split()
        if len(words) < 6 or len(words) > 30:
            continue
        if "@" in text or ";" in text or "(" in text:
            continue
        if not text[0].isupper() or not text.endswith("."):
            continue
        alpha = sum(1 for w in words if re.match(r"^[a-zA-Z]+$", w))
        if alpha / len(words) < 0.7:
            continue
        sentences.append(text)
        if len(sentences) >= max_examples:
            break
    return sentences


def tokenize_example(tok, phrase: str) -> dict:
    target = piglatinize(phrase)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": phrase},
        {"role": "assistant", "content": target},
    ]
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": phrase},
    ]
    prompt_text = tok.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    assert len(labels) == len(full_ids)
    return {
        "model_input": {
            "chunks": [{"type": "encoded_text", "tokens": list(full_ids)}],
        },
        "loss_fn_inputs": {
            "target_tokens": {"data": list(labels), "shape": [len(labels)]},
        },
    }


# ── Reward function ──────────────────────────────────────────────────────


def _word_score(gen_word: str, ref_word: str) -> float:
    """Score a single generated word against reference.

    1.0 for exact match; partial credit for matching suffix (the
    pig-latin transformation changes the beginning of each word but
    preserves the ending pattern, so suffix overlap is a useful signal).
    """
    if gen_word == ref_word:
        return 1.0
    if not gen_word or not ref_word:
        return 0.0
    # Suffix overlap: count matching characters from the end.
    suffix = 0
    for a, b in zip(reversed(gen_word), reversed(ref_word), strict=False):
        if a == b:
            suffix += 1
        else:
            break
    return min(suffix / len(ref_word), 0.8)


def pig_latin_reward(generated: str, reference: str) -> float:
    """Word-level reward with partial credit for suffix matches."""
    gen_words = generated.strip().lower().split()
    ref_words = reference.strip().lower().split()
    if not ref_words:
        return 1.0 if not gen_words else 0.0
    total = 0.0
    for i, ref_w in enumerate(ref_words):
        if i < len(gen_words):
            total += _word_score(gen_words[i], ref_w)
    return total / len(ref_words)


# ── Training loop ────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT + GRPO on WikiText pig-latin")
    parser.add_argument(
        "--base-url", default=os.environ.get("HATCHERY_BASE_URL", "http://127.0.0.1:8420")
    )
    parser.add_argument("--token", default=os.environ.get("HATCHERY_API_KEY", "dev"))
    parser.add_argument("--base-model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--rl-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rl-lr", type=float, default=5e-5)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--sample-temp", type=float, default=0.4)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", default="hatchery-piglatin")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model)

    # W&B setup.
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config={
                "base_model": args.base_model,
                "rank": args.rank,
                "sft_steps": args.steps,
                "sft_epochs": args.sft_epochs,
                "rl_steps": args.rl_steps,
                "batch_size": args.batch_size,
                "sft_lr": args.lr,
                "rl_lr": args.rl_lr,
                "kl_beta": args.kl_beta,
                "n_samples": args.n_samples,
                "sample_temp": args.sample_temp,
            },
        )

    # Load dataset.
    print("loading WikiText-2 sentences...")
    all_sentences = load_wikitext_sentences(
        max_examples=args.steps + args.rl_steps * args.batch_size + 10
    )
    print(f"  {len(all_sentences)} sentences extracted")

    train_data = [tokenize_example(tok, s) for s in all_sentences]
    print(f"\ndataset: {len(train_data)} examples (chat-template formatted)")
    print(f"  eos_token: {tok.eos_token!r}")

    client = HatcheryClient(base_url=args.base_url, token=args.token, timeout=300)

    try:
        tc = client.create_lora_training_client(base_model=args.base_model, rank=args.rank)
        print(f"\nsession {tc.session_id} (rank={args.rank})")
        global_step = 0

        # ────────────────────────────────────────────────────────────
        # Phase 1: SFT
        # ────────────────────────────────────────────────────────────
        sft_data = train_data[: min(args.steps, len(train_data))]
        n_batches_per_epoch = (len(sft_data) + args.batch_size - 1) // args.batch_size
        total_sft_steps = n_batches_per_epoch * args.sft_epochs
        print(
            f"\n═══ Phase 1: SFT ({len(sft_data)} examples, {args.sft_epochs} epochs, "
            f"{total_sft_steps} steps, bs={args.batch_size}) ═══"
        )

        for epoch in range(args.sft_epochs):
            for batch_idx in range(n_batches_per_epoch):
                start = batch_idx * args.batch_size
                batch = sft_data[start : start + args.batch_size]
                fb = tc.forward_backward(batch).result(timeout=120)
                tc.optim_step(learning_rate=args.lr).result(timeout=60)
                global_step += 1
                loss = fb.get("loss", fb.get("metrics", {}).get("loss:mean", 0))
                tokens = fb.get("num_tokens", fb.get("metrics", {}).get("num_tokens:sum", 0))

                if run:
                    run.log({"sft/loss": loss, "sft/tokens": tokens, "step": global_step})
                if batch_idx % 5 == 0 or batch_idx == n_batches_per_epoch - 1:
                    print(
                        f"  epoch {epoch + 1}/{args.sft_epochs}  "
                        f"step {global_step:>3}/{total_sft_steps}  loss={loss:.4f}"
                    )

        ckpt = tc.save_weights("sft-checkpoint").result(timeout=60)
        print(f"  checkpoint: {ckpt.get('path', ckpt)}")

        # ────────────────────────────────────────────────────────────
        # Phase 2: GRPO
        # ────────────────────────────────────────────────────────────
        rl_sentences = all_sentences[len(sft_data) :]
        if not rl_sentences:
            rl_sentences = all_sentences
        n_rl_batches = args.rl_steps
        print(
            f"\n═══ Phase 2: GRPO ({n_rl_batches} steps, bs={args.batch_size}, "
            f"n={args.n_samples} samples/prompt) ═══"
        )

        for rl_step in range(n_rl_batches):
            start = (rl_step * args.batch_size) % len(rl_sentences)
            batch_phrases = []
            for j in range(args.batch_size):
                batch_phrases.append(rl_sentences[(start + j) % len(rl_sentences)])

            # 2a. Sample n completions per prompt.
            all_grpo_data = []
            batch_rewards = []

            for phrase in batch_phrases:
                reference = piglatinize(phrase)
                prompt_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": phrase},
                ]
                prompt_text = tok.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

                result = tc.sample(
                    prompt_ids,
                    max_tokens=64,
                    temperature=args.sample_temp,
                    n=args.n_samples,
                ).result(timeout=60)

                raw_seqs = result.get("sequences", [])
                token_seqs = []
                logprob_seqs = []
                for s in raw_seqs:
                    if isinstance(s, dict):
                        token_seqs.append(s.get("tokens", []))
                        logprob_seqs.append(s.get("logprobs") or [])
                    else:
                        token_seqs.append(s)
                        logprob_seqs.append([])

                # 2b. Score each completion.
                rewards = []
                for seq in token_seqs:
                    gen_text = tok.decode(seq, skip_special_tokens=True)
                    r = pig_latin_reward(gen_text, reference)
                    rewards.append(r)

                # 2c. Compute advantages (group-relative: normalize within group).
                mean_r = sum(rewards) / max(len(rewards), 1)
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)) ** 0.5
                std_r = max(std_r, 1e-6)
                advantages = [(r - mean_r) / std_r for r in rewards]

                batch_rewards.extend(rewards)

                # 2d. Build GRPO training data for each completion.
                for seq, old_lp, adv in zip(token_seqs, logprob_seqs, advantages, strict=False):
                    full_ids = list(prompt_ids) + list(seq)
                    labels = [-100] * len(prompt_ids) + list(seq)

                    comp_len = len(seq)
                    if len(old_lp) < comp_len:
                        old_lp = list(old_lp) + [0.0] * (comp_len - len(old_lp))
                    old_lp = old_lp[:comp_len]
                    padded_old_lp = [0.0] * len(prompt_ids) + list(old_lp)
                    adv_per_token = [0.0] * len(prompt_ids) + [adv] * comp_len

                    all_grpo_data.append(
                        {
                            "model_input": {
                                "chunks": [{"type": "encoded_text", "tokens": full_ids}],
                            },
                            "loss_fn_inputs": {
                                "target_tokens": {"data": labels, "shape": [len(labels)]},
                                "logprobs": {"data": padded_old_lp, "shape": [len(padded_old_lp)]},
                                "advantages": {
                                    "data": adv_per_token,
                                    "shape": [len(adv_per_token)],
                                },
                            },
                        }
                    )

            # 2e. Run GRPO forward_backward in mini-batches to fit in VRAM,
            # then one optim_step to apply accumulated gradients.
            grpo_mini_bs = 8
            grpo_losses = []
            for mb_start in range(0, len(all_grpo_data), grpo_mini_bs):
                mb = all_grpo_data[mb_start : mb_start + grpo_mini_bs]
                fb = tc.forward_backward(
                    mb,
                    loss_fn="grpo",
                    loss_fn_config={
                        "kl_beta": args.kl_beta,
                        "clip_low_threshold": 0.8,
                        "clip_high_threshold": 1.2,
                    },
                ).result(timeout=120)
                grpo_losses.append(fb.get("loss", fb.get("metrics", {}).get("loss:mean", 0)))
            tc.optim_step(learning_rate=args.rl_lr).result(timeout=60)
            global_step += 1

            grpo_loss = sum(grpo_losses) / max(len(grpo_losses), 1)
            mean_reward = sum(batch_rewards) / max(len(batch_rewards), 1)

            if run:
                run.log(
                    {
                        "grpo/loss": grpo_loss,
                        "grpo/mean_reward": mean_reward,
                        "grpo/batch_size": len(all_grpo_data),
                        "step": global_step,
                    }
                )
            if rl_step % 5 == 0 or rl_step == n_rl_batches - 1:
                print(
                    f"  grpo step {rl_step + 1:>3}/{n_rl_batches}  "
                    f"loss={grpo_loss:.4f}  reward={mean_reward:.3f}"
                )

        ckpt = tc.save_weights("grpo-checkpoint").result(timeout=60)
        print(f"  checkpoint: {ckpt.get('path', ckpt)}")

        # ────────────────────────────────────────────────────────────
        # Inference
        # ────────────────────────────────────────────────────────────
        print("\n═══ Inference ═══")
        eval_phrases = all_sentences[:3] + all_sentences[-3:]
        for phrase in eval_phrases:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": phrase},
            ]
            prompt_text = tok.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            result = tc.sample(
                prompt_ids,
                max_tokens=64,
                temperature=0.0,
                stop=[tok.eos_token] if tok.eos_token else [],
            ).result(timeout=30)
            seqs = result.get("sequences", [[]])
            gen_ids = seqs[0] if seqs else []
            if isinstance(gen_ids, dict):
                gen_ids = gen_ids.get("tokens", gen_ids.get("input_ids", []))
            gen_text = (
                tok.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else "<empty>"
            )
            expected = piglatinize(phrase)
            reward = pig_latin_reward(gen_text, expected)
            print(f"  '{phrase}'")
            print(f"    → '{gen_text}'")
            print(f"    want: '{expected}'  reward={reward:.2f}")
            print()

        if run:
            run.finish()
        return 0
    finally:
        from hatchery.core.client import _BackgroundLoop

        fut = _BackgroundLoop.get().submit(client.aclose())
        with contextlib.suppress(Exception):
            fut.result(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
