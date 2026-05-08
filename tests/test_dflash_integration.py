# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU-only tests for the DFlash speculative decoding integration.

All tests run without a GPU and without the real dflash library installed.
The dflash module is injected via monkeypatch so the import path is exercised
without an actual dflash dependency.

Test coverage:
  - resolve_dflash_policy: policy decisions for various request/config combos
  - parse_spec_request: payload extraction
  - run_dflash_sample: success, fallback (not installed, runtime error, strict)
  - vanilla sampling path unchanged when speculative_decoding absent
  - _handle_sample routing: DFlash success path, fallback path, vanilla path
  - PEFT/wrapped verifier: model is passed through to dflash.generate() as-is
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from hatchery.core.dflash_integration import (
    DFlashConfig,
    _normalize_dflash_output,
    parse_spec_request,
    resolve_dflash_policy,
    run_dflash_sample,
)
from hatchery.core.spec_decoding import (
    SPEC_BACKEND_DFLASH,
    SpeculativeDecodingMetadata,
    SpeculativeDecodingRequest,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_dflash_module(generate_return: Any = None) -> ModuleType:
    """Return a minimal fake dflash module whose generate() returns a preset value."""
    mod = ModuleType("dflash")
    if generate_return is None:
        generate_return = {
            "sequences": [[10, 11, 12]],
            "texts": ["hello"],
            "stop_reasons": ["length"],
            "sequence_logprobs": [[-0.1, -0.2, -0.3]],
            "acceptance_rate": 0.85,
        }
    mod.generate = MagicMock(return_value=generate_return)
    return mod


def _make_spec_request(**kwargs: Any) -> SpeculativeDecodingRequest:
    defaults = {"enable": True}
    defaults.update(kwargs)
    return SpeculativeDecodingRequest(**defaults)


def _make_dflash_config(**kwargs: Any) -> DFlashConfig:
    defaults = {"draft_model": "org/draft-1b", "max_draft_tokens": 4}
    defaults.update(kwargs)
    return DFlashConfig(**defaults)


# ── parse_spec_request ────────────────────────────────────────────────────────


def test_parse_spec_request_absent():
    assert parse_spec_request({}) is None


def test_parse_spec_request_none_value():
    assert parse_spec_request({"speculative_decoding": None}) is None


def test_parse_spec_request_from_dict():
    req = parse_spec_request({"speculative_decoding": {"enable": True, "backend": "dflash"}})
    assert req is not None
    assert req.enable is True
    assert req.backend == "dflash"


def test_parse_spec_request_passthrough():
    orig = SpeculativeDecodingRequest(enable=True)
    result = parse_spec_request({"speculative_decoding": orig})
    assert result is orig


# ── resolve_dflash_policy ─────────────────────────────────────────────────────


def test_policy_no_request_returns_false():
    should_use, meta = resolve_dflash_policy(None, _make_dflash_config())
    assert should_use is False
    assert meta.requested_backend is None
    assert meta.fallback_reason is None


def test_policy_request_enable_false_returns_false():
    req = SpeculativeDecodingRequest(enable=False)
    should_use, meta = resolve_dflash_policy(req, _make_dflash_config())
    assert should_use is False


def test_policy_wrong_backend_returns_false():
    req = SpeculativeDecodingRequest(enable=True, backend="ngram")
    should_use, meta = resolve_dflash_policy(req, _make_dflash_config())
    assert should_use is False
    assert meta.fallback_reason is None


def test_policy_no_config_returns_false_with_fallback_reason():
    req = _make_spec_request()
    should_use, meta = resolve_dflash_policy(req, None)
    assert should_use is False
    assert meta.fallback_reason == "dflash_disabled"


def test_policy_disabled_config_returns_false_with_fallback_reason():
    req = _make_spec_request()
    cfg = _make_dflash_config(enabled=False)
    should_use, meta = resolve_dflash_policy(req, cfg)
    assert should_use is False
    assert meta.fallback_reason == "dflash_disabled"


def test_policy_no_draft_model_returns_false_with_fallback_reason():
    req = _make_spec_request()
    cfg = DFlashConfig(draft_model=None)
    should_use, meta = resolve_dflash_policy(req, cfg)
    assert should_use is False
    assert meta.fallback_reason == "no_draft_model_configured"


def test_policy_qwen36_base_model_infers_known_draft():
    req = _make_spec_request()
    cfg = DFlashConfig(draft_model=None)
    should_use, meta = resolve_dflash_policy(
        req,
        cfg,
        base_model_name="Qwen/Qwen3.6-35B-A3B",
    )
    assert should_use is True
    assert meta.draft_model == "z-lab/Qwen3.6-35B-A3B-DFlash"


def test_policy_strict_no_config_raises():
    req = SpeculativeDecodingRequest(enable=True, strict=True)
    with pytest.raises(ValueError, match="disabled"):
        resolve_dflash_policy(req, None)


def test_policy_strict_no_draft_model_raises():
    req = SpeculativeDecodingRequest(enable=True, strict=True)
    cfg = DFlashConfig(draft_model=None)
    with pytest.raises(ValueError, match="draft_model"):
        resolve_dflash_policy(req, cfg)


def test_policy_eligible_request_returns_true():
    req = _make_spec_request()
    cfg = _make_dflash_config()
    should_use, meta = resolve_dflash_policy(req, cfg)
    assert should_use is True
    assert meta.draft_model == "org/draft-1b"
    assert meta.max_draft_tokens == 4
    assert meta.requested_backend == SPEC_BACKEND_DFLASH
    assert meta.fallback_reason is None


def test_policy_per_request_max_draft_tokens_overrides_config():
    req = SpeculativeDecodingRequest(enable=True, max_draft_tokens=8)
    cfg = _make_dflash_config(max_draft_tokens=3)
    should_use, meta = resolve_dflash_policy(req, cfg)
    assert should_use is True
    assert meta.max_draft_tokens == 8


def test_policy_enable_none_treated_as_enabled():
    """enable=None defers to server config — should be treated as opt-in."""
    req = SpeculativeDecodingRequest(enable=None)
    cfg = _make_dflash_config()
    should_use, meta = resolve_dflash_policy(req, cfg)
    assert should_use is True


# ── run_dflash_sample — dflash not installed ──────────────────────────────────


def _run_sample_no_dflash(**extra_kwargs: Any):
    """Helper: call run_dflash_sample with dflash forcibly absent."""
    return run_dflash_sample(
        verifier_model=MagicMock(),
        tokenizer=MagicMock(),
        prompt_tokens=[1, 2, 3],
        max_new_tokens=32,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        n=1,
        seed=None,
        stop=None,
        spec_request=_make_spec_request(),
        dflash_config=_make_dflash_config(),
        device="cpu",
        base_model_name="Qwen/Qwen3.6-35B-A3B",
        **extra_kwargs,
    )


def test_run_dflash_sample_not_installed_returns_none(monkeypatch):
    """When dflash is not installed, run_dflash_sample returns (None, meta)."""
    monkeypatch.setitem(sys.modules, "dflash", None)  # type: ignore[arg-type]
    result, meta = _run_sample_no_dflash()
    assert result is None
    assert meta.fallback_reason == "dflash_not_installed"
    assert meta.used_backend is None


def test_run_dflash_sample_not_installed_strict_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "dflash", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="not installed"):
        run_dflash_sample(
            verifier_model=MagicMock(),
            tokenizer=MagicMock(),
            prompt_tokens=[1, 2, 3],
            max_new_tokens=32,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            n=1,
            seed=None,
            stop=None,
            spec_request=SpeculativeDecodingRequest(enable=True, strict=True),
            dflash_config=_make_dflash_config(),
            device="cpu",
            base_model_name="Qwen/Qwen3.6-35B-A3B",
        )


# ── run_dflash_sample — dflash installed ─────────────────────────────────────


def _run_sample_with_mock_dflash(
    dflash_mod: ModuleType,
    *,
    spec_request: Optional[SpeculativeDecodingRequest] = None,
    dflash_config: Optional[DFlashConfig] = None,
    verifier_model: Any = None,
    **extra: Any,
):
    """Run run_dflash_sample with a fake dflash module injected."""
    if spec_request is None:
        spec_request = _make_spec_request()
    if dflash_config is None:
        dflash_config = _make_dflash_config()
    if verifier_model is None:
        verifier_model = MagicMock(name="verifier")

    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        return run_dflash_sample(
            verifier_model=verifier_model,
            tokenizer=MagicMock(),
            prompt_tokens=[1, 2, 3],
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
            top_k=-1,
            n=1,
            seed=42,
            stop=None,
            spec_request=spec_request,
            dflash_config=dflash_config,
            device="cpu",
            **extra,
        )


def test_run_dflash_sample_success():
    dflash_mod = _make_dflash_module()
    result, meta = _run_sample_with_mock_dflash(dflash_mod)

    assert result is not None
    assert result["sequences"] == [[10, 11, 12]]
    assert result["texts"] == ["hello"]
    assert meta.used_backend == SPEC_BACKEND_DFLASH
    assert meta.fallback_reason is None


def test_run_dflash_sample_scores_missing_logprobs():
    dflash_mod = _make_dflash_module(
        generate_return={
            "sequences": [[4, 5]],
            "texts": ["4 5"],
            "stop_reasons": ["length"],
        }
    )
    verifier = _FakePEFTModel()

    result, meta = _run_sample_with_mock_dflash(dflash_mod, verifier_model=verifier)

    assert result is not None
    assert result["sequences"] == [[4, 5]]
    assert len(result["sequence_logprobs"]) == 1
    assert len(result["sequence_logprobs"][0]) == 2
    assert meta.used_backend == SPEC_BACKEND_DFLASH


def test_run_dflash_sample_calls_generate_with_verifier(monkeypatch):
    """The PEFT-wrapped verifier is forwarded to dflash.generate() as-is."""
    verifier = MagicMock(name="peft_wrapped_model")
    dflash_mod = _make_dflash_module()

    _run_sample_with_mock_dflash(dflash_mod, verifier_model=verifier)

    assert dflash_mod.generate.called
    call_kwargs = dflash_mod.generate.call_args[1]
    assert call_kwargs["verifier"] is verifier


def test_run_dflash_sample_passes_draft_model():
    cfg = _make_dflash_config(draft_model="org/small-draft")
    dflash_mod = _make_dflash_module()
    _run_sample_with_mock_dflash(dflash_mod, dflash_config=cfg)
    call_kwargs = dflash_mod.generate.call_args[1]
    assert call_kwargs["draft"] == "org/small-draft"


def test_run_dflash_sample_infers_qwen36_draft_model():
    dflash_mod = _make_dflash_module()
    _run_sample_with_mock_dflash(
        dflash_mod,
        dflash_config=DFlashConfig(draft_model=None),
        base_model_name="Qwen/Qwen3.6-35B-A3B",
    )
    call_kwargs = dflash_mod.generate.call_args[1]
    assert call_kwargs["draft"] == "z-lab/Qwen3.6-35B-A3B-DFlash"


def test_run_dflash_sample_passes_max_draft_tokens():
    req = SpeculativeDecodingRequest(enable=True, max_draft_tokens=7)
    dflash_mod = _make_dflash_module()
    _run_sample_with_mock_dflash(dflash_mod, spec_request=req)
    call_kwargs = dflash_mod.generate.call_args[1]
    assert call_kwargs["max_draft_tokens"] == 7


def test_run_dflash_sample_runtime_error_fallback():
    """Runtime error in dflash.generate() falls back gracefully."""
    dflash_mod = _make_dflash_module()
    dflash_mod.generate.side_effect = RuntimeError("cuda oom")

    result, meta = _run_sample_with_mock_dflash(dflash_mod)
    assert result is None
    assert "RuntimeError" in meta.fallback_reason
    assert meta.used_backend is None


def test_run_dflash_sample_runtime_error_strict_reraises():
    dflash_mod = _make_dflash_module()
    dflash_mod.generate.side_effect = RuntimeError("kaboom")
    req = SpeculativeDecodingRequest(enable=True, strict=True)

    with pytest.raises(RuntimeError, match="kaboom"):
        _run_sample_with_mock_dflash(dflash_mod, spec_request=req)


def test_run_dflash_sample_policy_mismatch_returns_none():
    """When policy says don't use DFlash, run_dflash_sample returns (None, meta)."""
    dflash_mod = _make_dflash_module()
    # enable=False → policy returns False
    req = SpeculativeDecodingRequest(enable=False)
    result, meta = _run_sample_with_mock_dflash(dflash_mod, spec_request=req)
    assert result is None
    assert dflash_mod.generate.call_count == 0


@pytest.mark.asyncio
async def test_handle_sample_explicit_disable_skips_qwen36_dflash():
    """Explicit enable=False must not force the Qwen3.6 DFlash draft mapping."""
    dflash_mod = _make_dflash_module()
    cfg = _make_dflash_config(draft_model=None)
    w = _make_minimal_worker_for_sample(dflash_cfg=cfg)

    payload = {
        "prompt_tokens": [1, 2],
        "max_tokens": 8,
        "temperature": 1.0,
        "speculative_decoding": {"enable": False},
    }
    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        result, metrics = await w._handle_sample("sess-1", payload)

    assert "spec_decoding_metadata" not in result
    assert "spec_backend" not in metrics
    assert dflash_mod.generate.call_count == 0


# ── _normalize_dflash_output ─────────────────────────────────────────────────


def test_normalize_full_dict():
    raw = {
        "sequences": [[1, 2, 3]],
        "texts": ["hello"],
        "stop_reasons": ["stop"],
        "sequence_logprobs": [[-0.1]],
        "acceptance_rate": 0.9,
    }
    out = _normalize_dflash_output(raw, tokenizer=None)
    assert out["sequences"] == [[1, 2, 3]]
    assert out["texts"] == ["hello"]
    assert out["stop_reasons"] == ["stop"]
    assert out["sequence_logprobs"] == [[-0.1]]


def test_normalize_missing_optional_fields():
    raw = {"sequences": [[5, 6]]}
    mock_tok = MagicMock()
    mock_tok.decode.return_value = "decoded"
    out = _normalize_dflash_output(raw, tokenizer=mock_tok)
    assert out["stop_reasons"] == ["length"]
    assert out["sequence_logprobs"] == [[]]
    assert out["texts"] == ["decoded"]


# ── Vanilla sampling path (no speculative_decoding) ───────────────────────────


def test_vanilla_path_does_not_invoke_dflash_policy():
    """When speculative_decoding is absent, DFlash integration is never reached."""
    # parse_spec_request should return None for a payload with no spec field.
    result = parse_spec_request({"prompt_tokens": [1, 2], "max_tokens": 32})
    assert result is None


def test_vanilla_path_enable_false_does_not_use_dflash():
    """Explicit enable=False from a previously configured policy stays off."""
    req = SpeculativeDecodingRequest(enable=False)
    should_use, meta = resolve_dflash_policy(req, _make_dflash_config())
    assert should_use is False
    assert meta.fallback_reason is None


# ── _handle_sample routing via mocked worker ─────────────────────────────────


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(t) for t in tokens)


class _FakePEFTModel:
    """Minimal PEFT-wrapped model stub that implements .eval() and .generate()."""

    def __init__(self, gen_output=None):
        self._gen = gen_output or _FakeGenOutput([[2, 3, 4]])

    def eval(self):
        return self

    def generate(self, **kwargs):
        return self._gen

    def __call__(self, **kwargs):
        vocab = 10
        T = kwargs["input_ids"].shape[1]
        logits = torch.zeros(1, T, vocab)
        return SimpleNamespace(logits=logits)


class _FakeGenOutput:
    def __init__(self, seqs):
        # sequences: [n, prompt_len + gen_len] — prefix with 2 dummy prompt tokens
        self.sequences = torch.tensor([[0, 0] + s for s in seqs])
        # scores: tuple of [n, V] per generated position
        T = max(len(s) for s in seqs)
        V = 10
        self.scores = tuple(torch.zeros(len(seqs), V) for _ in range(T))


from hatchery.core.worker import GPUWorker  # noqa: E402 — after torch import


def _make_minimal_worker_for_sample(*, dflash_cfg=None):
    """Build a minimal GPUWorker-like object wired for _handle_sample testing.

    Uses object.__new__ to avoid invoking real __init__ (which requires
    CUDA, model weights, etc.).
    """
    w = object.__new__(GPUWorker)
    w.tokenizer = _FakeTokenizer()
    w.device = "cpu"

    # Minimal config with optional dflash field
    cfg = SimpleNamespace(dflash=dflash_cfg)
    w.config = cfg

    # Minimal session runtime machinery
    runtime_stub = SimpleNamespace(
        model=None,
        lora_config=None,
        active_adapter=None,
    )
    peft_model = _FakePEFTModel()

    async def _ensure_session_loaded(sid):
        return runtime_stub

    def _activate_session(sid, runtime):
        return peft_model

    import contextlib

    @contextlib.contextmanager
    def _exec_context(runtime):
        yield

    # Bind as instance attributes (GPUWorker uses self.x naming)
    w._ensure_session_loaded = _ensure_session_loaded
    w._activate_session = _activate_session
    w._exec_context = _exec_context
    return w


@pytest.mark.asyncio
async def test_handle_sample_vanilla_no_spec_decoding():
    """_handle_sample with no speculative_decoding key runs vanilla HF path."""
    w = _make_minimal_worker_for_sample()
    payload = {"prompt_tokens": [1, 2], "max_tokens": 4, "temperature": 0.0}
    result, metrics = await w._handle_sample("sess-1", payload)
    assert "sequences" in result
    assert "spec_decoding_metadata" not in result
    assert "spec_backend" not in metrics


@pytest.mark.asyncio
async def test_handle_sample_dflash_success_path():
    """When DFlash is available and config is set, _handle_sample takes the DFlash path."""
    dflash_mod = _make_dflash_module(
        generate_return={
            "sequences": [[10, 11]],
            "texts": ["hi"],
            "stop_reasons": ["length"],
            "sequence_logprobs": [[-0.1, -0.2]],
            "acceptance_rate": 0.8,
        }
    )
    cfg = _make_dflash_config()
    w = _make_minimal_worker_for_sample(dflash_cfg=cfg)

    payload = {
        "prompt_tokens": [1, 2],
        "max_tokens": 8,
        "temperature": 1.0,
        "speculative_decoding": {"enable": True},
    }
    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        result, metrics = await w._handle_sample("sess-1", payload)

    assert result["sequences"] == [[10, 11]]
    assert metrics["spec_backend"] == "dflash"
    assert result["spec_decoding_metadata"]["used_backend"] == "dflash"
    assert dflash_mod.generate.call_count == 1


@pytest.mark.asyncio
async def test_handle_sample_dflash_fallback_on_runtime_error():
    """When DFlash raises at runtime, _handle_sample falls back to HF generate."""
    dflash_mod = _make_dflash_module()
    dflash_mod.generate.side_effect = RuntimeError("GPU OOM")
    cfg = _make_dflash_config()
    w = _make_minimal_worker_for_sample(dflash_cfg=cfg)

    payload = {
        "prompt_tokens": [1, 2],
        "max_tokens": 4,
        "temperature": 0.0,
        "speculative_decoding": {"enable": True},
    }
    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        result, metrics = await w._handle_sample("sess-1", payload)

    # Fell back to HF generate path
    assert "sequences" in result
    assert "spec_backend" not in metrics
    assert result["spec_decoding_metadata"]["fallback_reason"].startswith("dflash_runtime_error")


@pytest.mark.asyncio
async def test_handle_sample_dflash_fallback_no_config():
    """Without DFlashConfig, client gets spec_decoding_metadata with fallback_reason."""
    dflash_mod = _make_dflash_module()
    w = _make_minimal_worker_for_sample(dflash_cfg=None)

    payload = {
        "prompt_tokens": [1, 2],
        "max_tokens": 4,
        "temperature": 0.0,
        "speculative_decoding": {"enable": True},
    }
    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        result, metrics = await w._handle_sample("sess-1", payload)

    # DFlash module not consulted — config is None
    assert dflash_mod.generate.call_count == 0
    assert "sequences" in result
    # B2 fix: metadata is present so the client knows DFlash was disabled
    assert result["spec_decoding_metadata"]["fallback_reason"] == "dflash_disabled"


@pytest.mark.asyncio
async def test_handle_sample_dflash_skipped_when_prompt_logprobs_requested():
    """DFlash is skipped when prompt_logprobs are requested (M3: bypass prevention)."""
    dflash_mod = _make_dflash_module()
    cfg = _make_dflash_config()
    w = _make_minimal_worker_for_sample(dflash_cfg=cfg)

    payload = {
        "prompt_tokens": [1, 2],
        "max_tokens": 4,
        "temperature": 0.0,
        "include_prompt_logprobs": True,
        "speculative_decoding": {"enable": True},
    }
    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        result, metrics = await w._handle_sample("sess-1", payload)

    # DFlash skipped — falls through to HF generate
    assert dflash_mod.generate.call_count == 0
    assert "sequences" in result
    assert result["spec_decoding_metadata"]["fallback_reason"] == "prompt_logprobs_requested"


@pytest.mark.asyncio
async def test_handle_sample_malformed_spec_decoding_falls_back():
    """Malformed speculative_decoding payload falls back to HF generate gracefully (B1)."""
    w = _make_minimal_worker_for_sample(dflash_cfg=_make_dflash_config())
    payload = {
        "prompt_tokens": [1, 2],
        "max_tokens": 4,
        "temperature": 0.0,
        # max_draft_tokens=0 is below the ge=1 bound — will fail Pydantic validation
        "speculative_decoding": {"enable": True, "max_draft_tokens": 0},
    }
    result, metrics = await w._handle_sample("sess-1", payload)
    # Parse error is swallowed; vanilla HF path runs
    assert "sequences" in result
    assert "spec_backend" not in metrics


@pytest.mark.asyncio
async def test_handle_sample_peft_verifier_passed_to_dflash():
    """The PEFT-wrapped model (from _activate_session) is forwarded to dflash.generate()."""
    dflash_mod = _make_dflash_module()
    cfg = _make_dflash_config()

    # Capture which verifier was passed
    captured_verifier: list = []
    original_generate = dflash_mod.generate

    def capturing_generate(**kwargs):
        captured_verifier.append(kwargs.get("verifier"))
        return original_generate.return_value

    dflash_mod.generate = MagicMock(side_effect=capturing_generate)

    w = _make_minimal_worker_for_sample(dflash_cfg=cfg)
    # The PEFT model is what _activate_session returns
    expected_model = w._activate_session("sess-1", None)

    payload = {
        "prompt_tokens": [1, 2],
        "max_tokens": 8,
        "temperature": 1.0,
        "speculative_decoding": {"enable": True},
    }
    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        await w._handle_sample("sess-1", payload)

    assert len(captured_verifier) == 1
    assert captured_verifier[0] is expected_model


# ── New canonical-DFlash adapter behaviors ───────────────────────────────────


def _make_real_shaped_dflash_module(generate_fn=None) -> ModuleType:
    """Fake dflash module with the canonical attribute surface.

    Exposes ``model.dflash_generate`` and ``DFlashDraftModel`` so the
    integration takes the real-draft path. ``generate_fn`` defaults to
    returning a 1×(prompt+gen) tensor — the shape ``return_stats=False``
    would produce — so we exercise the tensor normalization branch.
    """
    mod = ModuleType("dflash")
    inner = ModuleType("dflash.model")
    if generate_fn is None:
        # Default: return a tensor 1×(prompt+gen) of 5 generated ids.
        def _default_gen(**kwargs):
            input_ids = kwargs["input_ids"]
            extra = torch.full((1, 5), 99, dtype=torch.long)
            return torch.cat([input_ids, extra], dim=1)

        generate_fn = _default_gen
    inner.dflash_generate = MagicMock(side_effect=generate_fn)
    mod.model = inner
    mod.dflash_generate = inner.dflash_generate
    mod.DFlashDraftModel = MagicMock(name="DFlashDraftModel")
    return mod


class _FakePEFTWrappedVerifier:
    """Stand-in for a PeftModelForCausalLM with a LoRA-augmented inner causal LM."""

    class _InnerCausalLM:
        # Has the attributes DFlash reaches into.
        def __init__(self):
            self.model = SimpleNamespace()  # transformer body
            self.lm_head = SimpleNamespace()

    class _LoraWrapper:
        def __init__(self, inner):
            self.model = inner

    def __init__(self):
        self._inner = _FakePEFTWrappedVerifier._InnerCausalLM()
        self.base_model = _FakePEFTWrappedVerifier._LoraWrapper(self._inner)
        self.device = torch.device("cpu")
        self.dtype = torch.float32


def test_resolve_verifier_for_dflash_unwraps_peft():
    """PEFT-wrapped verifier resolves to the inner LoRA-augmented causal LM."""
    from hatchery.core.dflash_integration import _resolve_verifier_for_dflash

    verifier = _FakePEFTWrappedVerifier()
    resolved = _resolve_verifier_for_dflash(verifier)
    assert resolved is verifier._inner  # inner causal LM with .model + .lm_head


def test_resolve_verifier_for_dflash_passthrough_for_plain_model():
    from hatchery.core.dflash_integration import _resolve_verifier_for_dflash

    plain = SimpleNamespace(model=SimpleNamespace(), lm_head=SimpleNamespace())
    assert _resolve_verifier_for_dflash(plain) is plain


def test_resolve_stop_token_ids_includes_eos():
    from hatchery.core.dflash_integration import _resolve_stop_token_ids

    tok = MagicMock()
    tok.eos_token_id = 7
    assert _resolve_stop_token_ids(None, tok) == [7]


def test_resolve_stop_token_ids_string_to_single_token():
    from hatchery.core.dflash_integration import _resolve_stop_token_ids

    tok = MagicMock()
    tok.eos_token_id = 2
    tok.encode.return_value = [42]
    out = _resolve_stop_token_ids(["</s>"], tok)
    assert out == [2, 42]


def test_resolve_stop_token_ids_drops_multi_token_string():
    from hatchery.core.dflash_integration import _resolve_stop_token_ids

    tok = MagicMock()
    tok.eos_token_id = 2
    tok.encode.return_value = [3, 4, 5]
    out = _resolve_stop_token_ids(["multi token stop"], tok)
    assert out == [2]  # eos kept; multi-token stop dropped


def test_resolve_stop_token_ids_int_passthrough_dedup():
    from hatchery.core.dflash_integration import _resolve_stop_token_ids

    tok = MagicMock()
    tok.eos_token_id = 5
    out = _resolve_stop_token_ids([5, 99], tok)
    assert out == [5, 99]


def test_real_dflash_n_gt_1_falls_back_with_metadata():
    """Real-shaped dflash module + n>1 produces n_gt_1_unsupported fallback."""
    dflash_mod = _make_real_shaped_dflash_module()
    verifier = _FakePEFTWrappedVerifier()
    tok = MagicMock()
    tok.eos_token_id = 0

    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        result, meta = run_dflash_sample(
            verifier_model=verifier,
            tokenizer=tok,
            prompt_tokens=[1, 2, 3],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            n=2,  # > 1 → fallback
            seed=None,
            stop=None,
            spec_request=_make_spec_request(),
            dflash_config=_make_dflash_config(),
            device="cpu",
        )

    assert result is None
    assert meta.fallback_reason == "n_gt_1_unsupported"


def test_real_dflash_top_p_falls_back_with_metadata():
    dflash_mod = _make_real_shaped_dflash_module()
    verifier = _FakePEFTWrappedVerifier()
    tok = MagicMock()
    tok.eos_token_id = 0

    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        result, meta = run_dflash_sample(
            verifier_model=verifier,
            tokenizer=tok,
            prompt_tokens=[1, 2, 3],
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.9,  # constrained → fallback
            top_k=-1,
            n=1,
            seed=None,
            stop=None,
            spec_request=_make_spec_request(),
            dflash_config=_make_dflash_config(),
            device="cpu",
        )

    assert result is None
    assert meta.fallback_reason == "top_p_top_k_unsupported"


def test_real_dflash_strict_top_p_raises():
    dflash_mod = _make_real_shaped_dflash_module()
    verifier = _FakePEFTWrappedVerifier()
    tok = MagicMock()
    tok.eos_token_id = 0
    req = SpeculativeDecodingRequest(enable=True, strict=True)

    with patch.dict(sys.modules, {"dflash": dflash_mod}):
        with pytest.raises(ValueError, match="top_p"):
            run_dflash_sample(
                verifier_model=verifier,
                tokenizer=tok,
                prompt_tokens=[1, 2, 3],
                max_new_tokens=8,
                temperature=0.0,
                top_p=0.5,
                top_k=-1,
                n=1,
                seed=None,
                stop=None,
                spec_request=req,
                dflash_config=_make_dflash_config(),
                device="cpu",
            )


def test_real_dflash_tensor_output_normalization():
    """Real-shaped module returning a tensor produces a strip-prompt result with text."""
    captured_kwargs = {}

    def fake_generate(**kwargs):
        captured_kwargs.update(kwargs)
        # Build output_ids = prompt + 4 new tokens; second-to-last is EOS=2.
        in_ids = kwargs["input_ids"]
        new = torch.tensor([[10, 11, 2, 13]], dtype=torch.long)
        return torch.cat([in_ids, new], dim=1)

    dflash_mod = _make_real_shaped_dflash_module(generate_fn=fake_generate)
    verifier = _FakePEFTWrappedVerifier()
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.decode.return_value = "decoded"

    # Patch the loader to skip AutoModel.from_pretrained.
    with (
        patch.dict(sys.modules, {"dflash": dflash_mod}),
        patch(
            "hatchery.core.dflash_integration._load_draft_model",
            return_value=MagicMock(name="loaded_draft"),
        ),
    ):
        result, meta = run_dflash_sample(
            verifier_model=verifier,
            tokenizer=tok,
            prompt_tokens=[1, 2, 3],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            n=1,
            seed=None,
            stop=None,
            spec_request=_make_spec_request(),
            dflash_config=_make_dflash_config(),
            device="cpu",
        )

    assert result is not None
    assert result["sequences"] == [[10, 11, 2, 13]]
    assert result["texts"] == ["decoded"]
    # The integration unwrapped the PEFT verifier to the inner LoRA-augmented LM
    # before forwarding to dflash.
    assert captured_kwargs["target"] is verifier._inner
    # Stop ids include the tokenizer's EOS.
    assert 2 in captured_kwargs["stop_token_ids"]
    assert meta.used_backend == "dflash"
    # Real call uses canonical kwargs (no legacy verifier=/draft= keys).
    assert "verifier" not in captured_kwargs
    assert "draft" not in captured_kwargs
    assert captured_kwargs["return_stats"] is True


def test_real_dflash_draft_load_error_falls_back():
    dflash_mod = _make_real_shaped_dflash_module()
    verifier = _FakePEFTWrappedVerifier()
    tok = MagicMock()
    tok.eos_token_id = 0

    def boom(*a, **kw):
        raise OSError("could not download draft weights")

    with (
        patch.dict(sys.modules, {"dflash": dflash_mod}),
        patch("hatchery.core.dflash_integration._load_draft_model", side_effect=boom),
    ):
        result, meta = run_dflash_sample(
            verifier_model=verifier,
            tokenizer=tok,
            prompt_tokens=[1, 2, 3],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            n=1,
            seed=None,
            stop=None,
            spec_request=_make_spec_request(),
            dflash_config=_make_dflash_config(),
            device="cpu",
        )

    assert result is None
    assert meta.fallback_reason == "draft_load_error:OSError"


def test_real_dflash_module_without_generate_falls_back():
    """If the installed dflash exposes neither dflash_generate nor generate, fall back."""
    mod = ModuleType("dflash")
    # Has DFlashDraftModel + model namespace but no callable generate.
    mod.DFlashDraftModel = MagicMock()
    mod.model = ModuleType("dflash.model")  # no dflash_generate

    verifier = _FakePEFTWrappedVerifier()
    tok = MagicMock()
    tok.eos_token_id = 0

    with patch.dict(sys.modules, {"dflash": mod}):
        result, meta = run_dflash_sample(
            verifier_model=verifier,
            tokenizer=tok,
            prompt_tokens=[1, 2, 3],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            n=1,
            seed=None,
            stop=None,
            spec_request=_make_spec_request(),
            dflash_config=_make_dflash_config(),
            device="cpu",
        )

    assert result is None
    assert meta.fallback_reason == "dflash_api_unavailable"
