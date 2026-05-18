"""Tests for the model_type fallback in lora_target_modules."""

from __future__ import annotations

from hatchery.core.lora_target_modules import (
    LLAMA_ATTN,
    LLAMA_MLP,
    MLA_ATTN_ALL,
    MLA_MLP,
    _resolve,
    target_modules_for,
)

# ── name-based resolution (existing behavior) ───────────────────────────


def test_name_resolves_deepseek_v3():
    attn, mlp = _resolve("deepseek-ai/DeepSeek-V3")
    assert attn == MLA_ATTN_ALL
    assert mlp == MLA_MLP


def test_name_resolves_kimi_k2():
    attn, mlp = _resolve("moonshotai/Kimi-K2.5-Instruct")
    assert attn == MLA_ATTN_ALL
    assert mlp == MLA_MLP


def test_name_resolves_llama():
    attn, mlp = _resolve("meta-llama/Llama-3.1-8B-Instruct")
    assert attn == LLAMA_ATTN
    assert mlp == LLAMA_MLP


# ── model_type fallback (new behavior) ──────────────────────────────────


def test_unknown_repo_falls_back_to_model_type(monkeypatch):
    """Custom-named DeepseekV3 derivative resolves to MLA via model_type."""

    class FakeCfg:
        model_type = "deepseek_v3"

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, name, **kw):
            return FakeCfg()

    import transformers

    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)

    # Repo string contains no architecture hint.
    attn, mlp = _resolve("custom-org/will_king_v2")
    assert attn == MLA_ATTN_ALL
    assert mlp == MLA_MLP


def test_unknown_repo_unknown_model_type_falls_back_to_llama(monkeypatch):
    class FakeCfg:
        model_type = "some_brand_new_arch"

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, name, **kw):
            return FakeCfg()

    import transformers

    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)

    attn, mlp = _resolve("custom-org/something_unknown")
    assert attn == LLAMA_ATTN
    assert mlp == LLAMA_MLP


def test_autoconfig_failure_falls_back_to_llama(monkeypatch):
    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, name, **kw):
            raise OSError("network unreachable")

    import transformers

    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)

    attn, mlp = _resolve("custom-org/no-internet-here")
    assert attn == LLAMA_ATTN
    assert mlp == LLAMA_MLP


# ── target_modules_for end-to-end ───────────────────────────────────────


def test_target_modules_for_includes_unembed():
    out = target_modules_for("meta-llama/Llama-3.1-8B-Instruct", train_unembed=True)
    assert "lm_head" in out


def test_target_modules_for_dedupes():
    out = target_modules_for("moonshotai/Kimi-K2.5-Instruct", train_attn=True, train_mlp=True)
    assert len(out) == len(set(out))
