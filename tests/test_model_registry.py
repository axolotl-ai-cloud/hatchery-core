# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for model ID resolver and context-length routing."""

from __future__ import annotations

from hatchery.core.model_registry import resolve_model_id


def test_standard_model_id():
    r = resolve_model_id("meta-llama/Llama-3.1-8B")
    assert r.base_model == "meta-llama/Llama-3.1-8B"
    assert r.max_context_length == 32768
    assert r.is_long_context is False
    assert r.required_cp_degree == 1


def test_long_context_peft_suffix():
    r = resolve_model_id("Qwen/Qwen3.5-397B-A17B:peft:262144")
    assert r.base_model == "Qwen/Qwen3.5-397B-A17B"
    assert r.max_context_length == 262144
    assert r.is_long_context is True
    assert r.required_cp_degree == 4


def test_128k_context():
    r = resolve_model_id("openai/gpt-oss-120b:peft:131072")
    assert r.base_model == "openai/gpt-oss-120b"
    assert r.max_context_length == 131072
    assert r.is_long_context is True
    assert r.required_cp_degree == 2  # 128K needs 2-way CP


def test_qwen35_default_context():
    """Qwen 3.5 family defaults to 64K."""
    r = resolve_model_id("Qwen/Qwen3.5-27B")
    assert r.max_context_length == 65536
    assert r.required_cp_degree == 1  # 64K fits on single GPU


def test_qwen36_default_context():
    """Qwen 3.6 family defaults to 262K."""
    r = resolve_model_id("Qwen/Qwen3.6-35B-A3B")
    assert r.max_context_length == 262144
    assert r.required_cp_degree == 4


def test_unknown_model_defaults_to_32k():
    r = resolve_model_id("some-org/unknown-model-7B")
    assert r.max_context_length == 32768
    assert r.required_cp_degree == 1


def test_invalid_peft_suffix_ignored():
    """If the suffix isn't a valid int, treat as plain model ID."""
    r = resolve_model_id("model:peft:not-a-number")
    assert r.base_model == "model:peft:not-a-number"
    assert r.is_long_context is False


def test_raw_model_id_preserved():
    r = resolve_model_id("Qwen/Qwen3.5-397B-A17B:peft:262144")
    assert r.raw_model_id == "Qwen/Qwen3.5-397B-A17B:peft:262144"


def test_512k_context_needs_8way_cp():
    r = resolve_model_id("big-model:peft:524288")
    assert r.required_cp_degree == 8


def test_very_long_context_extrapolates():
    r = resolve_model_id("big-model:peft:1048576")  # 1M tokens
    assert r.required_cp_degree >= 8
