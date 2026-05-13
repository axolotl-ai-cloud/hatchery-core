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

Resolution order:

1. Substring match on the lowercased ``base_model`` repo string. Cheap
   and covers most well-known model names.
2. Fall back to inspecting ``AutoConfig.from_pretrained(base_model).model_type``
   if the repo string didn't match. This catches custom-named DeepseekV3
   or Kimi derivatives whose HF repo doesn't contain the architecture
   substring — e.g. ``Foremost04/will_king_v2`` (a fine-tuned Moonlight)
   would otherwise hit the Llama fallback and silently miss MLA modules.
3. Final fallback: Llama/Qwen projection set.

The model_type lookup is best-effort — if HF is unreachable or the
config is gated, we fall through to the Llama default rather than raise.

Public entry point: :func:`target_modules_for`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LLAMA_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
LLAMA_MLP = ["gate_proj", "up_proj", "down_proj"]

MLA_ATTN = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
# DeepseekV3 with ``q_lora_rank=None`` (e.g. some Moonlight derivatives
# like will_king_v2) keeps ``q_proj`` unfactored, so we list both forms.
# PEFT silently ignores names that don't exist on the model.
MLA_ATTN_ALL = ["q_proj", *MLA_ATTN]
MLA_MLP = ["gate_proj", "up_proj", "down_proj"]

GPT2_ATTN = ["c_attn"]
GPT2_MLP = ["c_fc", "c_proj"]  # ``c_proj`` doubles as attn.out — PEFT substring match catches both.

# Ordered so the first match wins. Keys are lowercase substrings of
# HuggingFace repo names (org + slash tolerated because we always
# lowercase the full path first).
_RULES: list[tuple[tuple[str, ...], tuple[list[str], list[str]]]] = [
    # MLA families — check BEFORE the generic "deepseek" / "kimi" rules
    # so V3 / K2 get the right attn names.
    (("deepseek-v3", "deepseek_v3", "deepseekv3"), (MLA_ATTN_ALL, MLA_MLP)),
    (("kimi-k2", "kimi_k2", "kimi-k25", "kimi_k25"), (MLA_ATTN_ALL, MLA_MLP)),
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

# When the repo string doesn't match, fall back to ``model_type`` from
# the HF config. Keys here are exact ``model_type`` values.
_MODEL_TYPE_RULES: dict[str, tuple[list[str], list[str]]] = {
    "deepseek_v3": (MLA_ATTN_ALL, MLA_MLP),
    "kimi_k2": (MLA_ATTN_ALL, MLA_MLP),
    "kimi_k25": (MLA_ATTN_ALL, MLA_MLP),
    "llama": (LLAMA_ATTN, LLAMA_MLP),
    "qwen2": (LLAMA_ATTN, LLAMA_MLP),
    "qwen2_moe": (LLAMA_ATTN, LLAMA_MLP),
    "qwen3": (LLAMA_ATTN, LLAMA_MLP),
    "qwen3_moe": (LLAMA_ATTN, LLAMA_MLP),
    "mistral": (LLAMA_ATTN, LLAMA_MLP),
    "mixtral": (LLAMA_ATTN, LLAMA_MLP),
    "gemma": (LLAMA_ATTN, LLAMA_MLP),
    "gemma2": (LLAMA_ATTN, LLAMA_MLP),
    "gemma3_text": (LLAMA_ATTN, LLAMA_MLP),
    "phi3": (LLAMA_ATTN, LLAMA_MLP),
    "gpt2": (GPT2_ATTN, GPT2_MLP),
    "gpt_neox": (GPT2_ATTN, GPT2_MLP),
}

_FALLBACK: tuple[list[str], list[str]] = (LLAMA_ATTN, LLAMA_MLP)


def _resolve_by_name(base_model: str) -> tuple[list[str], list[str]] | None:
    name = base_model.lower().replace("/", "_")
    for needles, mods in _RULES:
        if any(n in name for n in needles):
            return mods
    return None


def _resolve_by_model_type(base_model: str) -> tuple[list[str], list[str]] | None:
    """Best-effort: inspect ``AutoConfig.from_pretrained(...).model_type``.

    Returns ``None`` if the config is unreachable or the model_type isn't
    in our table. Never raises — the caller falls through to the Llama
    default in that case.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        return None
    try:
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=False)
    except Exception as e:  # noqa: BLE001
        logger.debug("AutoConfig.from_pretrained(%s) failed: %s", base_model, e)
        return None
    mt = (getattr(cfg, "model_type", "") or "").lower()
    return _MODEL_TYPE_RULES.get(mt)


def _resolve(base_model: str) -> tuple[list[str], list[str]]:
    """Pick (attn_modules, mlp_modules) for the given base model.

    See module docstring for the resolution order (name → model_type →
    Llama fallback).
    """
    hit = _resolve_by_name(base_model)
    if hit is not None:
        return hit
    hit = _resolve_by_model_type(base_model)
    if hit is not None:
        logger.info(
            "LoRA targets for %r selected via model_type fallback "
            "(repo string didn't match any architecture rule)",
            base_model,
        )
        return hit
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
