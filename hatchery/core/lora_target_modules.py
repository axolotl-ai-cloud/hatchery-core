# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Architecture-aware LoRA target module selection.

The tinker SDK expresses LoRA coverage as three booleans — ``train_attn``,
``train_mlp``, ``train_unembed`` — but PEFT wants concrete linear-layer
names (``q_proj``, ``gate_proj``, ``lm_head`` …). Those names differ
across architectures:

* Llama / Qwen / Mistral / Gemma share the same projection names.
* DeepSeek V3 and Kimi K2/K2.5 use MLA (``q_a_proj``, ``q_b_proj``,
  ``kv_a_proj_with_mqa``, ``kv_b_proj``).
* GPT-2 / GPT-NeoX style uses ``c_attn``, ``c_proj``, ``c_fc``.

We resolve by substring match on the base model name (we're already
allowlisting models upstream, so the name space is narrow). Anything
unrecognised falls back to the Llama/Qwen projection set — the
dominant convention for modern open models.

Public entry point: :func:`target_modules_for`.
"""

from __future__ import annotations

LLAMA_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
LLAMA_MLP = ["gate_proj", "up_proj", "down_proj"]

MLA_ATTN = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
MLA_MLP = ["gate_proj", "up_proj", "down_proj"]

GPT2_ATTN = ["c_attn"]
GPT2_MLP = ["c_fc", "c_proj"]  # ``c_proj`` doubles as attn.out — PEFT substring match catches both.

# Ordered so the first match wins. Keys are lowercase substrings of
# HuggingFace repo names (org + slash tolerated because we always
# lowercase the full path first).
_RULES: list[tuple[tuple[str, ...], tuple[list[str], list[str]]]] = [
    # MLA families — check BEFORE the generic "deepseek" / "kimi" rules
    # so V3 / K2 get the right attn names.
    (("deepseek-v3", "deepseek_v3", "deepseekv3"), (MLA_ATTN, MLA_MLP)),
    (("kimi-k2", "kimi_k2", "kimi-k25", "kimi_k25"), (MLA_ATTN, MLA_MLP)),
    # Llama-family with standard MHA / GQA
    (("llama",), (LLAMA_ATTN, LLAMA_MLP)),
    (("qwen",), (LLAMA_ATTN, LLAMA_MLP)),
    (("mistral",), (LLAMA_ATTN, LLAMA_MLP)),
    (("mixtral",), (LLAMA_ATTN, LLAMA_MLP)),
    (("gemma",), (LLAMA_ATTN, LLAMA_MLP)),
    (("phi-3", "phi3", "phi-2", "phi2"), (LLAMA_ATTN, LLAMA_MLP)),
    # GPT-2 family (rare for post-training but cheap to support)
    (("gpt2", "gpt-2", "gpt_neox", "gpt-neox", "pythia"), (GPT2_ATTN, GPT2_MLP)),
]

_FALLBACK: tuple[list[str], list[str]] = (LLAMA_ATTN, LLAMA_MLP)


def _resolve(base_model: str) -> tuple[list[str], list[str]]:
    name = base_model.lower().replace("/", "_")
    for needles, mods in _RULES:
        if any(n in name for n in needles):
            return mods
    return _FALLBACK


def target_modules_for(
    base_model: str,
    *,
    train_attn: bool = True,
    train_mlp: bool = True,
    train_unembed: bool = False,
) -> list[str]:
    """Return ordered ``target_modules`` for PEFT based on the model.

    ``train_unembed=True`` appends ``lm_head`` — callers should apply the
    vLLM lm_head monkeypatch (or disable vLLM sampling) if they plan to
    serve the resulting adapter through vLLM, which does not natively
    accept lm_head LoRA.
    """
    attn_mods, mlp_mods = _resolve(base_model)

    out: list[str] = []
    seen: set[str] = set()
    if train_attn:
        for m in attn_mods:
            if m not in seen:
                seen.add(m)
                out.append(m)
    if train_mlp:
        for m in mlp_mods:
            if m not in seen:
                seen.add(m)
                out.append(m)
    if train_unembed and "lm_head" not in seen:
        out.append("lm_head")
    return out


__all__ = ["target_modules_for"]
