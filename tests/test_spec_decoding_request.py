# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for speculative decoding request parsing.

Covers ``SamplingParams.speculative_decoding`` and ``enable_thinking``
fields — ensuring existing behavior is preserved when the new fields are
absent, and that the new fields round-trip correctly from JSON payloads.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hatchery.core.spec_decoding import (
    SpeculativeDecodingMetadata,
    SpeculativeDecodingRequest,
)
from hatchery.core.tinker_compat import SamplingParams

# ── SamplingParams backward-compatibility ─────────────────────────────────


def test_sampling_params_existing_fields_unchanged():
    """All pre-existing SamplingParams fields work without spec_decoding."""
    params = SamplingParams(temperature=0.7, max_tokens=100, top_p=0.9, seed=42)
    assert params.temperature == 0.7
    assert params.max_tokens == 100
    assert params.top_p == 0.9
    assert params.seed == 42
    assert params.speculative_decoding is None
    assert params.enable_thinking is None


def test_sampling_params_spec_decoding_defaults_to_none():
    params = SamplingParams()
    assert params.speculative_decoding is None


def test_sampling_params_enable_thinking_defaults_to_none():
    params = SamplingParams()
    assert params.enable_thinking is None


def test_sampling_params_spec_decoding_enable_true():
    params = SamplingParams(speculative_decoding={"enable": True})
    assert params.speculative_decoding is not None
    assert params.speculative_decoding.enable is True
    assert params.speculative_decoding.backend is None
    assert params.speculative_decoding.max_draft_tokens is None
    assert params.speculative_decoding.strict is False


def test_sampling_params_spec_decoding_enable_false():
    params = SamplingParams(speculative_decoding={"enable": False})
    assert params.speculative_decoding.enable is False


def test_sampling_params_spec_decoding_backend():
    params = SamplingParams(speculative_decoding={"backend": "dflash"})
    assert params.speculative_decoding.backend == "dflash"


def test_sampling_params_spec_decoding_strict():
    params = SamplingParams(speculative_decoding={"enable": True, "strict": True})
    assert params.speculative_decoding.strict is True


def test_sampling_params_spec_decoding_max_draft_tokens():
    params = SamplingParams(speculative_decoding={"max_draft_tokens": 8})
    assert params.speculative_decoding.max_draft_tokens == 8


def test_sampling_params_spec_decoding_max_draft_tokens_bounds():
    with pytest.raises(ValidationError):
        SamplingParams(speculative_decoding={"max_draft_tokens": 0})
    with pytest.raises(ValidationError):
        SamplingParams(speculative_decoding={"max_draft_tokens": 65})


def test_sampling_params_enable_thinking_false():
    params = SamplingParams(enable_thinking=False)
    assert params.enable_thinking is False


def test_sampling_params_enable_thinking_true():
    params = SamplingParams(enable_thinking=True)
    assert params.enable_thinking is True


# ── SpeculativeDecodingRequest standalone ─────────────────────────────────


def test_spec_decoding_request_all_defaults():
    req = SpeculativeDecodingRequest()
    assert req.enable is None
    assert req.backend is None
    assert req.max_draft_tokens is None
    assert req.strict is False


def test_spec_decoding_request_full():
    req = SpeculativeDecodingRequest(enable=True, backend="dflash", max_draft_tokens=4, strict=True)
    assert req.enable is True
    assert req.backend == "dflash"
    assert req.max_draft_tokens == 4
    assert req.strict is True


def test_spec_decoding_request_model_dump_excludes_none():
    req = SpeculativeDecodingRequest(enable=True)
    d = req.model_dump(exclude_none=True)
    assert "enable" in d
    assert "backend" not in d
    assert "max_draft_tokens" not in d
    assert d["strict"] is False


# ── SpeculativeDecodingMetadata ───────────────────────────────────────────


def test_spec_decoding_metadata_all_none():
    meta = SpeculativeDecodingMetadata()
    assert meta.requested_backend is None
    assert meta.used_backend is None
    assert meta.draft_model is None
    assert meta.max_draft_tokens is None
    assert meta.fallback_reason is None


def test_spec_decoding_metadata_populated():
    meta = SpeculativeDecodingMetadata(
        requested_backend="dflash",
        used_backend="dflash",
        draft_model="org/draft-model",
        max_draft_tokens=4,
        fallback_reason=None,
    )
    assert meta.used_backend == "dflash"
    assert meta.draft_model == "org/draft-model"
    assert meta.max_draft_tokens == 4
    assert meta.fallback_reason is None


def test_spec_decoding_metadata_fallback():
    meta = SpeculativeDecodingMetadata(
        requested_backend="dflash",
        used_backend="none",
        fallback_reason="model_not_supported",
    )
    assert meta.used_backend == "none"
    assert meta.fallback_reason == "model_not_supported"
