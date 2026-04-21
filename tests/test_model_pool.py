# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for RewrapModelPool.

We inject a fake loader so the tests don't touch HF or transformers.
The pool contract cares about LRU eviction, load-once-per-name,
capacity enforcement, and eviction hooks — all of which we can
validate with dict-shaped fakes.
"""

from __future__ import annotations

from typing import Any

import pytest

from hatchery.core.model_pool import (
    RewrapModelPool,
    TieredModelPool,
    build_default_model_pool,
)


class _FakeBase:
    def __init__(self, name: str) -> None:
        self.name = name


def _loader(calls: list[str]):
    def _load(name: str) -> Any:
        calls.append(name)
        return _FakeBase(name)

    return _load


def test_load_once_and_cache_hit():
    calls: list[str] = []
    pool = RewrapModelPool(max_slots=2, device="cpu", loader=_loader(calls))

    slot1 = pool.get_or_load("modelA")
    slot2 = pool.get_or_load("modelA")
    assert slot1 is slot2
    assert calls == ["modelA"]
    assert pool.size() == 1


def test_lru_eviction_when_at_capacity():
    calls: list[str] = []
    pool = RewrapModelPool(max_slots=2, device="cpu", loader=_loader(calls))

    pool.get_or_load("modelA")
    pool.get_or_load("modelB")
    pool.get_or_load("modelC")  # evicts modelA

    assert set(pool.loaded_models()) == {"modelB", "modelC"}
    # modelA re-load is a fresh call.
    pool.get_or_load("modelA")
    assert calls == ["modelA", "modelB", "modelC", "modelA"]


def test_get_moves_to_mru_and_protects_from_eviction():
    calls: list[str] = []
    pool = RewrapModelPool(max_slots=2, device="cpu", loader=_loader(calls))

    pool.get_or_load("modelA")
    pool.get_or_load("modelB")
    # Touching A should make B the LRU victim.
    pool.get("modelA")
    pool.get_or_load("modelC")

    assert set(pool.loaded_models()) == {"modelA", "modelC"}


def test_evict_by_name():
    pool = RewrapModelPool(max_slots=3, device="cpu", loader=_loader([]))
    pool.get_or_load("modelA")
    pool.get_or_load("modelB")
    pool.evict("modelA")
    assert set(pool.loaded_models()) == {"modelB"}
    # Evicting a name that isn't loaded is a no-op.
    pool.evict("modelA")
    assert set(pool.loaded_models()) == {"modelB"}


def test_evict_lru_on_empty_pool_returns_none():
    pool = RewrapModelPool(max_slots=1, device="cpu", loader=_loader([]))
    assert pool.evict_lru() is None


def test_max_slots_one_always_has_at_most_one():
    calls: list[str] = []
    pool = RewrapModelPool(max_slots=1, device="cpu", loader=_loader(calls))
    for name in ("A", "B", "C", "D"):
        pool.get_or_load(name)
        assert pool.size() == 1
    assert pool.loaded_models() == ["D"]
    assert calls == ["A", "B", "C", "D"]


def test_rejects_zero_slots():
    with pytest.raises(ValueError):
        RewrapModelPool(max_slots=0, device="cpu", loader=_loader([]))


def test_load_time_recorded():
    pool = RewrapModelPool(max_slots=1, device="cpu", loader=_loader([]))
    slot = pool.get_or_load("modelA")
    assert slot.load_time_s is not None
    assert slot.load_time_s >= 0


def test_build_default_factory_honors_env(monkeypatch):
    monkeypatch.setenv("HATCHERY_MODEL_POOL_MAX_SLOTS", "3")
    pool = build_default_model_pool(device="cpu", loader=_loader([]))
    assert isinstance(pool, RewrapModelPool)
    assert pool.max_slots == 3


def test_eviction_clears_raw_base_reference():
    """After evict, the slot's raw_base should be dropped so Python
    GC can reclaim whatever the fake loader returned."""
    pool = RewrapModelPool(max_slots=2, device="cpu", loader=_loader([]))
    pool.get_or_load("A")
    pool.get_or_load("B")
    # Evicting A should remove it from the ordered dict entirely.
    pool.get_or_load("C")  # evicts A
    assert "A" not in pool.loaded_models()
    # And a subsequent A load reconstructs a fresh slot.
    new = pool.get_or_load("A")
    assert new.raw_base is not None
    assert new.raw_base.name == "A"


def test_peft_model_and_adapters_start_unset():
    """The pool holds only the frozen base. Adapters are managed
    by the trainer; the pool never touches PEFT state directly."""
    pool = RewrapModelPool(max_slots=1, device="cpu", loader=_loader([]))
    slot = pool.get_or_load("A")
    assert slot.peft_model is None
    assert slot.adapters == set()


# ─── Tokenizer / VLM ownership ────────────────────────────────────────


def test_test_loader_without_tokenizer_loader_leaves_fields_default():
    """With a model ``loader`` but no ``tokenizer_loader`` (the common
    test path), the pool must not try to hit HF for a tokenizer."""
    pool = RewrapModelPool(max_slots=1, device="cpu", loader=_loader([]))
    slot = pool.get_or_load("A")
    assert slot.tokenizer is None
    assert slot.processor is None
    assert slot.is_vlm is False
    assert slot.vision_token_ids == set()
    assert slot.host_state_dict is None


def test_tokenizer_loader_hook_populates_slot():
    """Tests can inject a tokenizer_loader to simulate a real load
    without touching transformers."""

    def _tok_load(name: str, raw: Any):
        class _FakeTok:
            pad_token = "<pad>"
            eos_token = "</s>"

        return _FakeTok(), None, False, set()

    pool = RewrapModelPool(
        max_slots=1,
        device="cpu",
        loader=_loader([]),
        tokenizer_loader=_tok_load,
    )
    slot = pool.get_or_load("A")
    assert slot.tokenizer is not None
    assert slot.tokenizer.pad_token == "<pad>"
    assert slot.is_vlm is False


def test_tokenizer_loader_hook_can_mark_vlm():
    def _tok_load(name: str, raw: Any):
        return object(), object(), True, {42, 43, 44}

    pool = RewrapModelPool(
        max_slots=1,
        device="cpu",
        loader=_loader([]),
        tokenizer_loader=_tok_load,
    )
    slot = pool.get_or_load("vlm-model")
    assert slot.is_vlm is True
    assert slot.processor is not None
    assert slot.vision_token_ids == {42, 43, 44}


def test_vlm_helpers_detect_class_name():
    """_is_vlm_model identifies models by class name; exposed on the
    pool module and re-exported by worker."""
    from hatchery.core.model_pool import _is_vlm_model

    class FakeQwen2VL:
        pass

    FakeQwen2VL.__name__ = "Qwen2VLForConditionalGeneration"

    class FakeLlama:
        pass

    FakeLlama.__name__ = "LlamaForCausalLM"

    assert _is_vlm_model(FakeQwen2VL()) is True
    assert _is_vlm_model(FakeLlama()) is False


def test_vision_token_ids_collects_image_markers():
    from hatchery.core.model_pool import _get_vision_token_ids

    class _Tok:
        def get_vocab(self):
            return {
                "hello": 1,
                "<image>": 2,
                "<|vision_start|>": 3,
                "<|image_pad|>": 4,
                "world": 5,
            }

    ids = _get_vision_token_ids(_Tok())
    assert ids == {2, 3, 4}


# ─── TieredModelPool ──────────────────────────────────────────────────


def _noop_demote(slot):
    # Record that demotion ran without touching torch.
    slot.host_state_dict = {"_demoted": True}


def _noop_promote(slot):
    slot.host_state_dict = None


def _tiered(max_vram=1, max_host=0, calls=None):
    return TieredModelPool(
        max_vram_slots=max_vram,
        max_host_slots=max_host,
        device="cpu",
        loader=_loader(calls if calls is not None else []),
        demote_hook=_noop_demote,
        promote_hook=_noop_promote,
    )


def test_tiered_vram_only_parity_with_rewrap():
    """With max_host_slots=0, TieredModelPool matches RewrapModelPool."""
    calls: list[str] = []
    pool = _tiered(max_vram=2, max_host=0, calls=calls)
    pool.get_or_load("A")
    pool.get_or_load("B")
    pool.get_or_load("C")  # evicts A (destroyed, since host=0)
    assert set(pool.loaded_models()) == {"B", "C"}
    assert pool.host_resident_models() == []
    pool.get_or_load("A")  # cold reload, no host promotion
    assert calls == ["A", "B", "C", "A"]


def test_tiered_host_only_mode_passes_demotion_hook():
    """With host slots >0, evicted VRAM goes to host instead of teardown."""
    calls: list[str] = []
    pool = _tiered(max_vram=1, max_host=2, calls=calls)
    pool.get_or_load("A")
    pool.get_or_load("B")  # A demoted to host
    assert pool.loaded_models() == ["B"]
    assert pool.host_resident_models() == ["A"]
    a_slot = pool._host["A"]  # noqa: SLF001 — test inspects internals
    assert a_slot.host_state_dict == {"_demoted": True}


def test_tiered_host_hit_promotes_back_to_vram():
    calls: list[str] = []
    pool = _tiered(max_vram=1, max_host=1, calls=calls)
    pool.get_or_load("A")
    pool.get_or_load("B")  # A -> host
    pool.get_or_load("A")  # A promoted back; no new load
    assert calls == ["A", "B"]
    assert pool.loaded_models() == ["A"]
    assert pool.host_resident_models() == ["B"]


def test_tiered_host_full_evicts_lru_host_on_demotion():
    calls: list[str] = []
    pool = _tiered(max_vram=1, max_host=1, calls=calls)
    pool.get_or_load("A")
    pool.get_or_load("B")  # A -> host
    pool.get_or_load("C")  # B -> host, A torn down (host LRU)
    assert pool.loaded_models() == ["C"]
    assert pool.host_resident_models() == ["B"]
    # A is gone entirely — next access is a cold load.
    pool.get_or_load("A")  # C -> host (evicting B), A loaded fresh
    assert calls == ["A", "B", "C", "A"]


def test_tiered_evict_removes_from_both_tiers():
    pool = _tiered(max_vram=1, max_host=2)
    pool.get_or_load("A")
    pool.get_or_load("B")  # A -> host
    pool.evict("A")
    assert pool.host_resident_models() == []
    pool.evict("B")
    assert pool.loaded_models() == []


def test_tiered_get_only_surfaces_vram_residents():
    pool = _tiered(max_vram=1, max_host=1)
    pool.get_or_load("A")
    pool.get_or_load("B")  # A demoted
    assert pool.get("A") is None
    assert pool.get("B") is not None


def test_tiered_rejects_invalid_params():
    with pytest.raises(ValueError):
        TieredModelPool(max_vram_slots=0, max_host_slots=0, device="cpu")
    with pytest.raises(ValueError):
        TieredModelPool(max_vram_slots=1, max_host_slots=-1, device="cpu")


def test_build_default_picks_tiered_from_env(monkeypatch):
    monkeypatch.setenv("HATCHERY_MODEL_POOL", "tiered")
    monkeypatch.setenv("HATCHERY_MODEL_POOL_MAX_SLOTS", "2")
    monkeypatch.setenv("HATCHERY_MODEL_POOL_MAX_HOST_SLOTS", "3")
    pool = build_default_model_pool(device="cpu", loader=_loader([]))
    assert isinstance(pool, TieredModelPool)
    assert pool.max_vram_slots == 2
    assert pool.max_host_slots == 3


def test_build_default_tiered_env_host_slots_default_zero(monkeypatch):
    """Without MAX_HOST_SLOTS, tiered pool is VRAM-only parity."""
    monkeypatch.setenv("HATCHERY_MODEL_POOL", "tiered")
    monkeypatch.delenv("HATCHERY_MODEL_POOL_MAX_HOST_SLOTS", raising=False)
    pool = build_default_model_pool(device="cpu", loader=_loader([]))
    assert isinstance(pool, TieredModelPool)
    assert pool.max_host_slots == 0


def test_build_default_rewrap_is_default(monkeypatch):
    monkeypatch.delenv("HATCHERY_MODEL_POOL", raising=False)
    pool = build_default_model_pool(device="cpu", loader=_loader([]))
    assert isinstance(pool, RewrapModelPool)
