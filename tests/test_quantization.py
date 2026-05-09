# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for 1-bit / BitNet loader routing.

These are pure-python: no GPU, no transformers download, no model
weights. We exercise:

* :class:`QuantConfig` construction and validation.
* :func:`is_onebit_model` — detection from a duck-typed HF config.
* :func:`is_onebit_by_name` — the cheap pre-load check.
* :func:`resolve_quant_scheme` — combining caller intent with auto-
  detection.
* :func:`prepare_onebit_loader_kwargs` — the fp16→bf16 upgrade.
* The pool hooks: ``_detect_scheme_on_model`` records the scheme on a
  :class:`PoolSlot`, and ``RewrapModelPool`` forwards the scheme.

A real-model smoke test that loads ``HATCHERY_ONEBITLLM_TEST_MODEL`` is
gated behind that env var so CI and the default test run stay
CPU/network-free.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from hatchery.core.model_pool import PoolSlot, RewrapModelPool, TieredModelPool
from hatchery.core.parallel import ParallelConfig
from hatchery.core.quantization import (
    QuantConfig,
    detect_quant_scheme,
    is_fp8_torchao_model,
    is_onebit_by_name,
    is_onebit_model,
    prepare_fp8_torchao_loader_kwargs,
    prepare_onebit_loader_kwargs,
    resolve_quant_scheme,
)

# ── QuantConfig ────────────────────────────────────────────────────────


def test_quant_config_defaults_are_noop():
    q = QuantConfig()
    assert q.scheme == "none"
    assert q.force is False
    assert q.require_full_param is True
    assert q.is_onebit is False


def test_quant_config_fp8_torchao():
    q = QuantConfig(scheme="fp8_torchao")
    assert q.is_fp8_torchao is True
    assert q.is_onebit is False
    assert q.scheme == "fp8_torchao"


def test_quant_config_fp8_torchao_modes():
    assert QuantConfig(scheme="fp8_torchao", fp8_mode="weight_only").fp8_mode == "weight_only"
    assert QuantConfig(scheme="fp8_torchao", fp8_mode="dynamic").fp8_mode == "dynamic"
    with pytest.raises(ValueError):
        QuantConfig(scheme="fp8_torchao", fp8_mode="int8")


def test_quant_config_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        QuantConfig(scheme="int4")


def test_quant_config_onebit_flags():
    q = QuantConfig(scheme="onebit")
    assert q.is_onebit is True


def test_parallel_config_carries_quant():
    p = ParallelConfig()
    assert isinstance(p.quant, QuantConfig)
    assert p.quant.scheme == "none"


def test_parallel_config_from_env(monkeypatch):
    monkeypatch.setenv("HATCHERY_QUANT_SCHEME", "onebit")
    monkeypatch.setenv("HATCHERY_QUANT_FORCE", "1")
    monkeypatch.setenv("HATCHERY_QUANT_REQUIRE_FULL_PARAM", "0")
    p = ParallelConfig.from_env()
    assert p.quant.scheme == "onebit"
    assert p.quant.force is True
    assert p.quant.require_full_param is False


def test_parallel_config_fp8_torchao_from_env(monkeypatch):
    monkeypatch.setenv("HATCHERY_QUANT_SCHEME", "fp8_torchao")
    monkeypatch.setenv("HATCHERY_FP8_MODE", "dynamic")
    p = ParallelConfig.from_env()
    assert p.quant.scheme == "fp8_torchao"
    assert p.quant.fp8_mode == "dynamic"
    assert p.quant.is_fp8_torchao is True


def test_parallel_config_fp8_mode_defaults_to_weight_only(monkeypatch):
    monkeypatch.setenv("HATCHERY_QUANT_SCHEME", "fp8_torchao")
    p = ParallelConfig.from_env()
    assert p.quant.fp8_mode == "weight_only"


def test_parallel_config_fp8_mode_unknown_degrades_silently(monkeypatch):
    monkeypatch.setenv("HATCHERY_QUANT_SCHEME", "fp8_torchao")
    monkeypatch.setenv("HATCHERY_FP8_MODE", "int8_bogus")
    p = ParallelConfig.from_env()
    assert p.quant.fp8_mode == "weight_only"


def test_parallel_config_from_env_rejects_unknown_scheme_silently(monkeypatch):
    # An unknown scheme must degrade to "none" rather than crash the
    # config loader — ParallelConfig is built at worker startup and we
    # don't want a typo to prevent the worker from booting.
    monkeypatch.setenv("HATCHERY_QUANT_SCHEME", "int4")
    p = ParallelConfig.from_env()
    assert p.quant.scheme == "none"


# ── Detection ──────────────────────────────────────────────────────────


def _cfg(**kwargs: Any) -> SimpleNamespace:
    """Build a HF-config-shaped namespace for detection tests."""
    base = dict(
        model_type=None,
        architectures=None,
        quantization_config=None,
        _name_or_path=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_detect_by_model_type():
    assert is_onebit_model(_cfg(model_type="bitnet"))
    assert not is_onebit_model(_cfg(model_type="llama"))


def test_detect_by_architectures():
    assert is_onebit_model(_cfg(architectures=["BitNetForCausalLM"]))
    assert not is_onebit_model(_cfg(architectures=["LlamaForCausalLM"]))


def test_detect_handles_non_iterable_architectures():
    # Some exotic configs set architectures to a string (not a list).
    # Our guard should just report no-match rather than crash.
    assert not is_onebit_model(_cfg(architectures=42))


def test_detect_by_quantization_config_dict():
    q = {"quant_method": "bitnet", "bits": 1.58}
    assert is_onebit_model(_cfg(quantization_config=q))


def test_detect_by_quantization_config_object():
    class QC:
        quant_method = "bitnet"

    assert is_onebit_model(_cfg(quantization_config=QC()))


def test_detect_by_model_name_hint():
    assert is_onebit_by_name("microsoft/bitnet-b1.58-2B-4T")
    assert is_onebit_by_name("1bitllm/bitnet_b1_58-large")
    assert not is_onebit_by_name("meta-llama/Llama-3.1-8B")
    assert not is_onebit_by_name(None)


def test_detect_quant_scheme_returns_string():
    assert detect_quant_scheme(_cfg(model_type="bitnet")) == "onebit"
    assert detect_quant_scheme(_cfg(model_type="llama")) == "none"


def test_resolve_respects_force():
    # force=True bypasses autodetect — caller is right.
    req = QuantConfig(scheme="none", force=True)
    assert resolve_quant_scheme(_cfg(model_type="bitnet"), requested=req) == "none"
    req = QuantConfig(scheme="onebit", force=True)
    assert resolve_quant_scheme(_cfg(model_type="llama"), requested=req) == "onebit"


def test_resolve_silently_upgrades_to_onebit():
    # Caller left it at "none", but the config says bitnet — we should
    # upgrade rather than mis-load.
    req = QuantConfig()  # scheme="none"
    assert resolve_quant_scheme(_cfg(model_type="bitnet"), requested=req) == "onebit"


def test_resolve_defaults_to_none_without_signals():
    assert resolve_quant_scheme(_cfg(model_type="llama")) == "none"


# ── FP8 TorchAO detection ─────────────────────────────────────────────


def test_detect_fp8_torchao_from_quantization_config_dict():
    """A serialised config.json with quant_type=torchao + Float8 is detected."""
    q_dict = {"quant_type": "torchao", "torchao_config": "Float8WeightOnlyConfig()"}
    assert is_fp8_torchao_model(_cfg(quantization_config=q_dict))


def test_detect_fp8_torchao_from_quantization_config_object():
    """A live TorchAoConfig object (duck-typed) is also detected."""

    class _TorchAoCfg:
        quant_type = "torchao"
        torchao_config = "Float8DynamicActivationFloat8WeightConfig()"

    assert is_fp8_torchao_model(_cfg(quantization_config=_TorchAoCfg()))


def test_non_float8_torchao_not_detected_as_fp8():
    """INT4/INT8 TorchAO configs must not be misidentified as FP8."""
    q_int4 = {"quant_type": "torchao", "torchao_config": "Int4WeightOnlyConfig()"}
    assert not is_fp8_torchao_model(_cfg(quantization_config=q_int4))


def test_detect_quant_scheme_returns_fp8_torchao():
    q = {"quant_type": "torchao", "torchao_config": "Float8WeightOnlyConfig()"}
    assert detect_quant_scheme(_cfg(quantization_config=q)) == "fp8_torchao"


def test_detect_quant_scheme_none_when_no_quant_config():
    assert detect_quant_scheme(_cfg()) == "none"


def test_resolve_silently_upgrades_to_fp8_torchao():
    """A checkpoint already tagged fp8_torchao wins over a 'none' request."""
    req = QuantConfig()  # scheme="none"
    q = {"quant_type": "torchao", "torchao_config": "Float8WeightOnlyConfig()"}
    assert resolve_quant_scheme(_cfg(quantization_config=q), requested=req) == "fp8_torchao"


def test_resolve_force_overrides_fp8_detection():
    req = QuantConfig(scheme="none", force=True)
    q = {"quant_type": "torchao", "torchao_config": "Float8WeightOnlyConfig()"}
    assert resolve_quant_scheme(_cfg(quantization_config=q), requested=req) == "none"


# ── Loader kwargs ──────────────────────────────────────────────────────


def test_prepare_onebit_kwargs_upgrades_fp16_to_bf16():
    torch = pytest.importorskip("torch")
    out = prepare_onebit_loader_kwargs({"torch_dtype": torch.float16})
    assert out["torch_dtype"] is torch.bfloat16


def test_prepare_onebit_kwargs_leaves_bfloat16_alone():
    torch = pytest.importorskip("torch")
    out = prepare_onebit_loader_kwargs({"torch_dtype": torch.bfloat16})
    assert out["torch_dtype"] is torch.bfloat16


def test_prepare_onebit_kwargs_leaves_none_alone():
    out = prepare_onebit_loader_kwargs({"torch_dtype": None})
    assert out["torch_dtype"] is None


def test_prepare_onebit_kwargs_preserves_other_kwargs():
    out = prepare_onebit_loader_kwargs({"attn_implementation": "sdpa"})
    assert out["attn_implementation"] == "sdpa"


# ── FP8 TorchAO loader kwargs ──────────────────────────────────────────
#
# Key requirement: prepare_fp8_torchao_loader_kwargs must INJECT a
# TorchAoConfig into the kwargs, NOT clear quantization_config like the
# prior BF16 dequantize approach (task_01KR4CZCG3R4MEY62HFZYV1JKD).
#
# Tests with real imports are gated on torchao+transformers being present.
# The mock-based test always runs and verifies the routing invariant.


def test_fp8_torchao_loader_kwargs_inject_not_clear(monkeypatch):
    """Routing invariant: FP8 loader injects TorchAoConfig, not clears quant config.

    Uses mocks so this test always runs regardless of whether torchao /
    transformers are installed.  The assertion is structural: the returned
    kwargs must contain a non-None 'quantization_config' whose value was
    produced by TorchAoConfig, not by clearing the field.
    """
    import sys
    from unittest.mock import MagicMock, patch

    sentinel = object()  # unique — proves TorchAoConfig was called

    mock_float8_cfg = object()
    MockFloat8WeightOnlyConfig = MagicMock(return_value=mock_float8_cfg)
    MockTorchAoConfig = MagicMock(return_value=sentinel)

    mock_torchao_quant = MagicMock()
    mock_torchao_quant.Float8WeightOnlyConfig = MockFloat8WeightOnlyConfig

    mock_transformers = MagicMock()
    mock_transformers.TorchAoConfig = MockTorchAoConfig

    # prepare_fp8_torchao_loader_kwargs uses local `from X import Y` at
    # call time, so patching sys.modules before the call is sufficient —
    # no importlib.reload needed (reload mutates module state globally).
    with patch.dict(
        sys.modules,
        {
            "torchao": MagicMock(),
            "torchao.quantization": mock_torchao_quant,
            "transformers": mock_transformers,
        },
    ):
        out = prepare_fp8_torchao_loader_kwargs(
            {"torch_dtype": "auto", "attn_implementation": "sdpa"}, fp8_mode="weight_only"
        )

    # Must set quantization_config to the sentinel produced by TorchAoConfig.
    assert out["quantization_config"] is sentinel
    # TorchAoConfig must have been called with the Float8WeightOnlyConfig instance.
    MockTorchAoConfig.assert_called_once_with(mock_float8_cfg)
    # Other kwargs survive unchanged.
    assert out["attn_implementation"] == "sdpa"


def test_fp8_torchao_loader_kwargs_dynamic_mode(monkeypatch):
    """dynamic fp8_mode routes to Float8DynamicActivationFloat8WeightConfig."""
    import sys
    from unittest.mock import MagicMock, patch

    sentinel_dynamic = object()
    mock_dyn_cfg = object()
    MockDynConfig = MagicMock(return_value=mock_dyn_cfg)
    MockTorchAoConfig = MagicMock(return_value=sentinel_dynamic)

    mock_torchao_quant = MagicMock()
    mock_torchao_quant.Float8DynamicActivationFloat8WeightConfig = MockDynConfig
    mock_transformers = MagicMock()
    mock_transformers.TorchAoConfig = MockTorchAoConfig

    with patch.dict(
        sys.modules,
        {
            "torchao": MagicMock(),
            "torchao.quantization": mock_torchao_quant,
            "transformers": mock_transformers,
        },
    ):
        out = prepare_fp8_torchao_loader_kwargs({"torch_dtype": "auto"}, fp8_mode="dynamic")

    assert out["quantization_config"] is sentinel_dynamic
    MockTorchAoConfig.assert_called_once_with(mock_dyn_cfg)


@pytest.mark.skipif(
    not os.environ.get("HATCHERY_FP8_INTEGRATION_TEST"),
    reason="set HATCHERY_FP8_INTEGRATION_TEST=1 to run with real torchao+transformers",
)
def test_prepare_fp8_torchao_loader_kwargs_real_torchao_config():
    """Integration test: real TorchAoConfig injected into kwargs."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "TorchAoConfig"):
        pytest.skip("transformers.TorchAoConfig not available (upgrade to >= 4.46)")
    pytest.importorskip("torchao")

    out = prepare_fp8_torchao_loader_kwargs({"torch_dtype": "auto"}, fp8_mode="weight_only")
    assert "quantization_config" in out
    assert isinstance(out["quantization_config"], transformers.TorchAoConfig)
    assert out["quantization_config"] is not None


# ── Pool routing ───────────────────────────────────────────────────────


class _FakeBaseWithConfig:
    """Minimal stand-in for an HF model — has a ``.config`` attribute."""

    def __init__(self, name: str, model_type: str = "llama") -> None:
        self.name = name
        self.config = SimpleNamespace(
            model_type=model_type,
            architectures=None,
            quantization_config=None,
            _name_or_path=name,
        )


def test_rewrap_pool_records_scheme_none_by_default():
    def loader(name: str) -> Any:
        return _FakeBaseWithConfig(name, model_type="llama")

    pool = RewrapModelPool(max_slots=2, device="cpu", loader=loader)
    slot = pool.get_or_load("meta-llama/Llama-3.1-8B")
    assert isinstance(slot, PoolSlot)
    assert slot.quant_scheme == "none"


def test_rewrap_pool_records_onebit_from_config():
    def loader(name: str) -> Any:
        return _FakeBaseWithConfig(name, model_type="bitnet")

    pool = RewrapModelPool(max_slots=2, device="cpu", loader=loader)
    slot = pool.get_or_load("microsoft/bitnet-b1.58-2B-4T")
    assert slot.quant_scheme == "onebit"


def test_rewrap_pool_respects_forced_scheme_without_config():
    """Test fakes that return plain objects (no .config) still honour
    a caller-supplied scheme so tests don't need to build SimpleNamespaces."""

    class _Plain:
        def __init__(self, name: str) -> None:
            self.name = name

    def loader(name: str) -> Any:
        return _Plain(name)

    q = QuantConfig(scheme="onebit", force=True)
    pool = RewrapModelPool(max_slots=1, device="cpu", loader=loader, quant_config=q)
    slot = pool.get_or_load("anything")
    assert slot.quant_scheme == "onebit"


def test_tiered_pool_records_scheme():
    def loader(name: str) -> Any:
        return _FakeBaseWithConfig(name, model_type="bitnet")

    pool = TieredModelPool(max_vram_slots=1, max_host_slots=0, device="cpu", loader=loader)
    slot = pool.get_or_load("microsoft/bitnet-b1.58-2B-4T")
    assert slot.quant_scheme == "onebit"


def test_pool_does_not_consult_autoconfig_for_llama_slugs():
    """Fast-path: if the slug doesn't match a BitNet hint AND no
    QuantConfig is supplied, we must not try to contact HF (the
    test runs offline — a network hit would show up as a failure)."""

    def loader(name: str) -> Any:
        return _FakeBaseWithConfig(name, model_type="llama")

    pool = RewrapModelPool(max_slots=1, device="cpu", loader=loader)
    # If this triggers an AutoConfig lookup, it would fail in an
    # offline / unauthenticated CI. The test passes because the fast
    # path skips the round-trip.
    slot = pool.get_or_load("meta-llama/Llama-3.1-8B")
    assert slot.quant_scheme == "none"


def test_rewrap_pool_records_fp8_torchao_from_config():
    """Pool detects fp8_torchao from a quantization_config on the loaded model."""

    class _FakeBaseWithFP8Config:
        def __init__(self, name: str) -> None:
            self.name = name
            self.config = SimpleNamespace(
                model_type="llama",
                architectures=None,
                quantization_config={
                    "quant_type": "torchao",
                    "torchao_config": "Float8WeightOnlyConfig()",
                },
                _name_or_path=name,
            )

    def loader(name: str) -> Any:
        return _FakeBaseWithFP8Config(name)

    pool = RewrapModelPool(max_slots=1, device="cpu", loader=loader)
    slot = pool.get_or_load("meta-llama/Llama-3.1-8B-fp8")
    assert slot.quant_scheme == "fp8_torchao"


# ── Trainer policy ─────────────────────────────────────────────────────


def test_trainer_lora_rejected_on_onebit_by_default():
    """BitNet is trained QAT; the default recipe is FFT on the master
    weights. A LoRA attach on a detected onebit base must be refused
    unless the caller opts in explicitly."""
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("transformers")
    from hatchery.core.trainer import LoraSpec, VanillaTrainer

    # Build a trainer but skip the HF load — we'll stub out the state
    # the guard reads.
    trainer = VanillaTrainer(
        base_model_name="microsoft/bitnet-b1.58-2B-4T",
        device="cpu",
        parallel=ParallelConfig(quant=QuantConfig(scheme="onebit", force=True)),
        load_model=False,
    )
    trainer._quant_scheme = "onebit"

    with pytest.raises(RuntimeError, match="1-bit"):
        trainer.attach_session("sess-1", LoraSpec(rank=8, lora_alpha=16, target_modules=["q_proj"]))


def test_trainer_full_param_unaffected_on_onebit():
    """Full-param attach on a onebit base is the happy path — no guard
    fires. We only need to confirm the guard lets FFT through."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from hatchery.core.trainer import LoraSpec, VanillaTrainer

    trainer = VanillaTrainer(
        base_model_name="microsoft/bitnet-b1.58-2B-4T",
        device="cpu",
        parallel=ParallelConfig(quant=QuantConfig(scheme="onebit", force=True)),
        load_model=False,
    )
    trainer._quant_scheme = "onebit"
    trainer._pristine_base_sd = {"fake": torch.zeros(1)}

    trainer.attach_session("sess-1", LoraSpec.full_param())
    assert "sess-1" in trainer._specs


def test_trainer_lora_allowed_on_fp8_torchao():
    """FP8 TorchAO models must NOT trigger the LoRA-refusal guard.

    Unlike BitNet ('onebit'), TorchAO FP8 uses autograd-compatible
    forward/backward and PEFT LoRA is fully supported.  The guard must
    never fire for scheme='fp8_torchao', regardless of require_full_param.
    """
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("transformers")
    from hatchery.core.trainer import LoraSpec, VanillaTrainer

    trainer = VanillaTrainer(
        base_model_name="meta-llama/Llama-3.1-8B",
        device="cpu",
        parallel=ParallelConfig(quant=QuantConfig(scheme="fp8_torchao", force=True)),
        load_model=False,
    )
    trainer._quant_scheme = "fp8_torchao"

    # Guard must not raise RuntimeError("1-bit").
    # A later PEFT / raw-base failure is fine — we're only testing the guard.
    try:
        trainer.attach_session(
            "sess-fp8",
            LoraSpec(rank=8, lora_alpha=16, target_modules=["q_proj"]),
        )
    except RuntimeError as exc:
        assert "1-bit" not in str(exc), f"Guard incorrectly fired for fp8_torchao: {exc}"
    except Exception:
        pass  # any non-guard failure is acceptable here


def test_trainer_lora_allowed_on_onebit_when_opted_in():
    """When ``require_full_param=False``, LoRA on a onebit base is
    allowed — the caller has explicitly chosen the non-default path.
    We stop short of actually building a PEFT wrapper (requires a real
    base); just confirm the guard doesn't fire.
    """
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("transformers")
    from hatchery.core.trainer import LoraSpec, VanillaTrainer

    trainer = VanillaTrainer(
        base_model_name="microsoft/bitnet-b1.58-2B-4T",
        device="cpu",
        parallel=ParallelConfig(
            quant=QuantConfig(scheme="onebit", force=True, require_full_param=False),
        ),
        load_model=False,
    )
    trainer._quant_scheme = "onebit"

    # The guard must not fire. A later PEFT import will fail because
    # ``_raw_base`` is None — anything that isn't our "1-bit" message
    # counts as "guard didn't block".
    try:
        trainer.attach_session("sess-1", LoraSpec(rank=8, lora_alpha=16, target_modules=["q_proj"]))
    except RuntimeError as e:
        assert "1-bit" not in str(e)
    except Exception:
        pass  # non-guard failure is fine for this test


# ── Real-model smoke (gated) ───────────────────────────────────────────


_SMOKE_ENV = "HATCHERY_ONEBITLLM_TEST_MODEL"


@pytest.mark.skipif(
    not os.environ.get(_SMOKE_ENV),
    reason=f"set {_SMOKE_ENV} to a gated BitNet repo slug to run",
)
def test_real_onebit_model_loads_and_forward_passes():
    """Opt-in smoke test.

    Requires ``HATCHERY_ONEBITLLM_TEST_MODEL`` to be set to a BitNet-family
    HuggingFace repo slug the current environment is authenticated
    for. We load the model on CPU (cheap, no CUDA dependency on the
    test harness) and run a one-token forward pass. If the repo is
    gated and the token is missing, HF surfaces a clear error — we
    let it propagate rather than swallow it.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model_slug = os.environ[_SMOKE_ENV]

    pool = RewrapModelPool(
        max_slots=1,
        device="cpu",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quant_config=QuantConfig(scheme="onebit"),
    )
    slot = pool.get_or_load(model_slug)
    assert slot.quant_scheme == "onebit"
    assert slot.raw_base is not None

    # Run a forward pass on a 1-token input.
    with torch.no_grad():
        out = slot.raw_base(input_ids=torch.tensor([[1]], dtype=torch.long))
    assert out.logits.shape[0] == 1
