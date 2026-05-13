# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for the loss-plugin registry in ``hatchery.core.losses``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hatchery.core import losses as losses_mod  # noqa: E402
from hatchery.core.losses import (  # noqa: E402
    LOSS_ENTRY_POINT_GROUP,
    LossInputs,
    LossNotImplementedError,
    compute,
    declared_loss_fns,
    is_registered,
    register_loss,
    registered_loss_fns,
    supported_loss_fns,
    unregister_loss,
)


def _simple_inputs() -> LossInputs:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 0, -100]], dtype=torch.long)
    weights = torch.ones_like(targets, dtype=torch.float32)
    return LossInputs(logits=logits, target_tokens=targets, weights=weights)


# ─── Test isolation helpers ─────────────────────────────────────────────


@pytest.fixture
def clean_registry():
    """Snapshot the registry and entry-point flag, restore on teardown."""
    saved_registry = dict(losses_mod._REGISTRY)
    saved_loaded = losses_mod._ENTRY_POINTS_LOADED
    try:
        yield
    finally:
        losses_mod._REGISTRY.clear()
        losses_mod._REGISTRY.update(saved_registry)
        losses_mod._ENTRY_POINTS_LOADED = saved_loaded


# ─── register_loss / unregister_loss / is_registered ────────────────────


def test_register_loss_happy_path(clean_registry):
    sentinel = object()

    def fake_loss(inputs: LossInputs):
        return sentinel

    register_loss("fake_name", fake_loss)
    assert is_registered("fake_name") is True
    assert "fake_name" in registered_loss_fns()
    assert "fake_name" in supported_loss_fns()
    assert compute("fake_name", _simple_inputs()) is sentinel


def test_register_loss_collides_with_builtin(clean_registry):
    def fake_loss(inputs: LossInputs):
        return torch.tensor(0.0)

    with pytest.raises(ValueError, match="collides with a built-in"):
        register_loss("cross_entropy", fake_loss)

    # override=True succeeds.
    register_loss("cross_entropy", fake_loss, override=True)
    out = compute("cross_entropy", _simple_inputs())
    assert torch.is_tensor(out) and float(out) == 0.0


def test_unregister_builtin_refused(clean_registry):
    with pytest.raises(ValueError, match="built-in"):
        unregister_loss("cross_entropy")
    assert is_registered("cross_entropy")


def test_unregister_plugin_removes_it(clean_registry):
    def fake_loss(inputs: LossInputs):
        return torch.tensor(0.0)

    register_loss("plugin_to_remove", fake_loss)
    assert is_registered("plugin_to_remove")
    unregister_loss("plugin_to_remove")
    assert not is_registered("plugin_to_remove")
    with pytest.raises(ValueError, match="unknown loss_fn"):
        compute("plugin_to_remove", _simple_inputs())


def test_unregister_unknown_raises_keyerror(clean_registry):
    with pytest.raises(KeyError):
        unregister_loss("never_registered_and_not_a_builtin")


def test_dro_can_be_overridden_by_registered_implementation(clean_registry):
    sentinel = object()

    def my_dro(inputs: LossInputs):
        return sentinel

    # ``dro`` is not a built-in, so no override flag needed.
    register_loss("dro", my_dro)
    assert compute("dro", _simple_inputs()) is sentinel


@pytest.mark.parametrize(
    "bad_name", ["", "has spaces", "1starts-with-digit", "has-dash", "with.dot"]
)
def test_register_loss_invalid_name(clean_registry, bad_name):
    def fake_loss(inputs: LossInputs):
        return torch.tensor(0.0)

    with pytest.raises(ValueError, match="identifier"):
        register_loss(bad_name, fake_loss)


def test_register_loss_non_callable_rejected(clean_registry):
    with pytest.raises(ValueError, match="must be callable"):
        register_loss("not_callable", "definitely_not_a_function")  # type: ignore[arg-type]


# ─── declared_loss_fns / supported_loss_fns ─────────────────────────────


def test_declared_includes_dro_and_registered(clean_registry):
    def fake_loss(inputs: LossInputs):
        return torch.tensor(0.0)

    register_loss("declared_plugin", fake_loss)
    declared = declared_loss_fns()
    assert "dro" in declared
    assert "declared_plugin" in declared
    assert "cross_entropy" in declared
    assert declared == tuple(sorted(declared))


def test_registered_loss_fns_sorted(clean_registry):
    names = registered_loss_fns()
    assert names == tuple(sorted(names))
    assert "cross_entropy" in names
    assert "orpo" in names


# ─── compute() dispatch ─────────────────────────────────────────────────


def test_dro_still_raises_with_message(clean_registry):
    with pytest.raises(LossNotImplementedError, match="not in the public docs"):
        compute("dro", _simple_inputs())


def test_unknown_loss_fn_raises(clean_registry):
    with pytest.raises(ValueError, match="unknown loss_fn"):
        compute("definitely_not_a_loss", _simple_inputs())


# ─── Entry-point discovery ──────────────────────────────────────────────


class _FakeEntryPoint:
    def __init__(self, name: str, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


class _FakeEntryPoints:
    def __init__(self, by_group: dict):
        self._by_group = by_group

    def select(self, *, group: str):
        return list(self._by_group.get(group, []))


def _install_fake_eps(monkeypatch, eps_for_group):
    fake = _FakeEntryPoints({LOSS_ENTRY_POINT_GROUP: eps_for_group})

    def fake_entry_points():
        return fake

    # The function imports `importlib.metadata` lazily inside
    # `_load_entry_points`, so patch the underlying module.
    import importlib.metadata as importlib_metadata

    monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)
    # Force re-discovery on next call.
    losses_mod._ENTRY_POINTS_LOADED = False


def test_entry_point_discovery_registers_plugin(clean_registry, monkeypatch):
    sentinel = object()

    def loader():
        def plugin_fn(inputs: LossInputs):
            return sentinel

        return plugin_fn

    _install_fake_eps(monkeypatch, [_FakeEntryPoint("plugin_name", loader)])

    # First call to compute triggers entry-point discovery.
    result = compute("plugin_name", _simple_inputs())
    assert result is sentinel
    assert is_registered("plugin_name")
    assert "plugin_name" in supported_loss_fns()


def test_broken_plugin_is_logged_and_skipped(clean_registry, monkeypatch, caplog):
    def good_loader():
        def plugin_fn(inputs: LossInputs):
            return torch.tensor(1.5)

        return plugin_fn

    def broken_loader():
        raise RuntimeError("boom from broken plugin")

    _install_fake_eps(
        monkeypatch,
        [
            _FakeEntryPoint("broken_plugin", broken_loader),
            _FakeEntryPoint("good_plugin", good_loader),
        ],
    )

    with caplog.at_level("ERROR", logger="hatchery.core.losses"):
        # Built-ins still work even with a broken plugin in the group.
        out = compute("cross_entropy", _simple_inputs())
        assert torch.is_tensor(out)

    assert any("broken_plugin" in record.getMessage() for record in caplog.records), (
        "expected an ERROR log mentioning the broken plugin name"
    )
    # The broken plugin must not be registered.
    assert not is_registered("broken_plugin")
    # The other plugin in the same group still loads.
    assert is_registered("good_plugin")


def test_entry_point_discovery_runs_once(clean_registry, monkeypatch):
    call_count = {"n": 0}

    def loader():
        call_count["n"] += 1

        def plugin_fn(inputs: LossInputs):
            return torch.tensor(0.0)

        return plugin_fn

    _install_fake_eps(monkeypatch, [_FakeEntryPoint("once_plugin", loader)])

    # Multiple lookups should still only run discovery once.
    is_registered("anything")
    registered_loss_fns()
    supported_loss_fns()
    compute("cross_entropy", _simple_inputs())

    assert call_count["n"] == 1
