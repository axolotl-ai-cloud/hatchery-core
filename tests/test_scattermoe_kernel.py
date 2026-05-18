from __future__ import annotations

from types import SimpleNamespace

import pytest

from hatchery.core.config import Config, build_core_config
from hatchery.core.scattermoe_kernel import ScatterMoEKernelConfig


class _FakeModel:
    def __init__(self, model_type: str = "qwen3_5_moe") -> None:
        self.config = SimpleNamespace(model_type=model_type)
        self.kernelized = False


def _config(scattermoe_kernel=None) -> Config:
    return Config(
        auth=SimpleNamespace(),
        metadata=SimpleNamespace(),
        objects=SimpleNamespace(),
        queue=SimpleNamespace(),
        compute=SimpleNamespace(),
        metrics=SimpleNamespace(),
        scattermoe_kernel=scattermoe_kernel,
    )


def test_core_config_default_keeps_runtime_optimizations_empty():
    report = _config().apply_runtime_model_optimizations(
        _FakeModel(),
        base_model_name="Qwen/Qwen3.6-35B-A3B",
    )
    assert report == {}


def test_scattermoe_kernel_applies_for_supported_model_when_available(monkeypatch):
    calls: dict[str, object] = {}
    registered: dict[str, object] = {}
    replaced: list[tuple[type, str]] = []

    def fake_get_kernel(kernel_ref: str, **_kw):
        calls["get_kernel"] = kernel_ref
        return object()

    def fake_kernelize(model, *, mode):
        calls["kernelize_model"] = model
        calls["mode"] = mode
        model.kernelized = True

    class FakeLayerRepository:
        def __init__(self, *, repo_id: str, layer_name: str, **_kw):
            self.repo_id = repo_id
            self.layer_name = layer_name

    def fake_register_kernel_mapping(mapping):
        registered.update(mapping)

    def fake_replace_kernel_forward_from_hub(cls, key):
        replaced.append((cls, key))

    # Patch the transformers MoE class lookup so the test doesn't need
    # transformers' real qwen3_5_moe internals.
    class FakeQwenMoeBlock:
        pass

    monkeypatch.setattr(
        "hatchery.core.scattermoe_kernel.SPARSE_MOE_BLOCK",
        {"qwen3_5_moe": [(FakeQwenMoeBlock.__module__, FakeQwenMoeBlock.__name__)]},
    )
    monkeypatch.setattr(
        "hatchery.core.scattermoe_kernel._resolve_moe_block_classes",
        lambda model_type: [FakeQwenMoeBlock],
    )
    monkeypatch.setattr(
        "hatchery.core.scattermoe_kernel._try_import_kernels",
        lambda: {
            "LayerRepository": FakeLayerRepository,
            "Mode": SimpleNamespace(TRAINING="training", INFERENCE="inference"),
            "get_kernel": fake_get_kernel,
            "kernelize": fake_kernelize,
            "register_kernel_mapping": fake_register_kernel_mapping,
            "replace_kernel_forward_from_hub": fake_replace_kernel_forward_from_hub,
        },
    )

    config = _config(
        ScatterMoEKernelConfig(
            enabled=True,
            kernel_ref="kernels-test/scattermoe",
            strict=True,
        )
    )
    model = _FakeModel()

    report = config.apply_runtime_model_optimizations(
        model,
        base_model_name="Qwen/Qwen3.6-35B-A3B",
    )

    assert report == {
        "scattermoe_kernel": {
            "status": "applied",
            "kernel_ref": "kernels-test/scattermoe",
            "applied": True,
            "compatible": True,
            "base_model_name": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "reason": None,
        }
    }
    assert calls["get_kernel"] == "kernels-test/scattermoe"
    assert calls["mode"] == "training"
    assert model.kernelized is True
    # New: registration + replacement actually happened.
    assert "HFScatterMoEParallelExperts" in registered
    assert (FakeQwenMoeBlock, "HFScatterMoEParallelExperts") in replaced


def test_scattermoe_kernel_falls_back_when_missing_and_not_strict(monkeypatch):
    monkeypatch.setattr("hatchery.core.scattermoe_kernel._try_import_kernels", lambda: None)

    report = _config(
        ScatterMoEKernelConfig(enabled=True, strict=False)
    ).apply_runtime_model_optimizations(
        _FakeModel(),
        base_model_name="Qwen/Qwen3.6-35B-A3B",
    )

    assert report["scattermoe_kernel"]["status"] == "unavailable"
    assert report["scattermoe_kernel"]["reason"] == "kernels_not_installed"


def test_scattermoe_kernel_errors_when_missing_and_strict(monkeypatch):
    monkeypatch.setattr("hatchery.core.scattermoe_kernel._try_import_kernels", lambda: None)

    with pytest.raises(ImportError, match="kernels"):
        _config(
            ScatterMoEKernelConfig(enabled=True, strict=True)
        ).apply_runtime_model_optimizations(
            _FakeModel(),
            base_model_name="Qwen/Qwen3.6-35B-A3B",
        )


def test_build_core_config_populates_scattermoe_kernel(monkeypatch):
    monkeypatch.setenv("HATCHERY_SCATTERMOE_KERNEL_ENABLED", "1")
    monkeypatch.setenv("HATCHERY_SCATTERMOE_KERNEL_REF", "kernels-test/scattermoe")
    monkeypatch.setenv("HATCHERY_SCATTERMOE_KERNEL_STRICT", "true")

    config = build_core_config()

    assert config.scattermoe_kernel is not None
    assert config.scattermoe_kernel.enabled is True
    assert config.scattermoe_kernel.kernel_ref == "kernels-test/scattermoe"
    assert config.scattermoe_kernel.strict is True


def test_build_core_config_uses_scattermoe_lora_default_ref(monkeypatch):
    monkeypatch.setenv("HATCHERY_SCATTERMOE_KERNEL_ENABLED", "1")

    config = build_core_config()

    assert config.scattermoe_kernel is not None
    assert config.scattermoe_kernel.kernel_ref == "axolotl-ai-co/scattermoe-lora"


def test_local_dev_config_honors_scattermoe_kernel_env(monkeypatch):
    from hatchery.core.local_dev import _build_config

    monkeypatch.setenv("SCATTERMOE_KERNEL_ENABLED", "1")
    monkeypatch.setenv("SCATTERMOE_KERNEL_STRICT", "true")

    config = _build_config("dev")

    assert config.scattermoe_kernel is not None
    assert config.scattermoe_kernel.enabled is True
    assert config.scattermoe_kernel.kernel_ref == "axolotl-ai-co/scattermoe-lora"
    assert config.scattermoe_kernel.strict is True
    assert config.scattermoe_kernel.trust_remote_code is True
