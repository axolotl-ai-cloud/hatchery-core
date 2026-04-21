# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the SamplingBackend implementations.

These test the protocol-level behavior without needing a live vLLM
server. For ``VLLMSamplingBackend`` we mock the HTTP calls; for
``LocalPEFTSamplingBackend`` we use a fake worker stub.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hatchery.core.sampling import (
    LocalPEFTSamplingBackend,
    SampledSequence,
    VLLMSamplingBackend,
)


def _mock_vllm_completions_response(tokens=None, text="hello", finish="length"):
    """Build a mock vLLM /v1/completions JSON response."""
    return {
        "choices": [
            {
                "tokens": tokens or [10, 11, 12],
                "text": text,
                "finish_reason": finish,
                "logprobs": None,
            }
        ]
    }


class _FakeHTTPResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


async def test_vllm_backend_publish_and_sample(monkeypatch):
    """Round-trip: publish an adapter then sample from it."""
    backend = VLLMSamplingBackend.from_urls(["http://vllm:8000"])

    calls = []

    async def _mock_post(self, url, **kwargs):
        calls.append(("POST", url, kwargs.get("json")))
        if "load_lora" in url:
            return _FakeHTTPResponse(200, {"result": "ok"})
        return _FakeHTTPResponse(200, _mock_vllm_completions_response([42, 43, 44]))

    async def _mock_aenter(self):
        return self

    async def _mock_aexit(self, *args):
        pass

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "post", _mock_post)
    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", _mock_aenter)
    monkeypatch.setattr(httpx.AsyncClient, "__aexit__", _mock_aexit)

    await backend.publish_adapter(adapter_name="sess-123", adapter_path="/mnt/adapters/sess-123")
    assert "sess-123" in backend._loaded_adapters

    results = await backend.sample(
        adapter_name="sess-123",
        prompt_tokens=[1, 2, 3],
        max_tokens=10,
    )
    assert len(results) == 1
    assert results[0].tokens == [42, 43, 44]

    # Verify the completions call used the adapter as the model name.
    completions_call = [c for c in calls if "completions" in c[1]]
    assert completions_call[0][2]["model"] == "sess-123"


async def test_vllm_backend_skips_already_loaded(monkeypatch):
    """publish_adapter should be a no-op if the adapter is already loaded."""
    backend = VLLMSamplingBackend.from_urls(["http://vllm:8000"])
    backend._loaded_adapters.add("sess-already")

    calls = []

    async def _mock_post(self, url, **kwargs):
        calls.append(url)
        return _FakeHTTPResponse(200)

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "post", _mock_post)
    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", lambda self: self)
    monkeypatch.setattr(httpx.AsyncClient, "__aexit__", AsyncMock())

    await backend.publish_adapter(adapter_name="sess-already", adapter_path="/whatever")
    assert len(calls) == 0  # no HTTP call made


async def test_vllm_backend_round_robin():
    """Endpoint selection should rotate across available endpoints."""
    backend = VLLMSamplingBackend.from_urls(["http://a:8000", "http://b:8000", "http://c:8000"])
    seen = [backend._next_endpoint().url for _ in range(6)]
    assert seen == [
        "http://a:8000",
        "http://b:8000",
        "http://c:8000",
        "http://a:8000",
        "http://b:8000",
        "http://c:8000",
    ]


async def test_vllm_backend_skips_unhealthy():
    backend = VLLMSamplingBackend.from_urls(["http://a:8000", "http://b:8000"])
    backend.endpoints[0].healthy = False
    ep = backend._next_endpoint()
    assert ep.url == "http://b:8000"


async def test_local_peft_backend_delegates_to_worker():
    """LocalPEFT backend should call the worker's _handle_sample."""
    fake_worker = AsyncMock()
    fake_worker._handle_sample.return_value = (
        {"sequences": [[99, 100]]},
        {"tokens": 2},
    )

    backend = LocalPEFTSamplingBackend(_worker=fake_worker)
    results = await backend.sample(
        adapter_name="sess_test_session",
        prompt_tokens=[1, 2, 3],
        max_tokens=5,
    )
    assert len(results) == 1
    assert results[0].tokens == [99, 100]
    fake_worker._handle_sample.assert_called_once()


async def test_local_peft_backend_forwards_seed_and_stop():
    """``seed`` and ``stop`` from SDK SamplingParams must reach the worker
    payload so ``torch.manual_seed`` / ``StopStringCriteria`` can apply."""
    fake_worker = AsyncMock()
    fake_worker._handle_sample.return_value = (
        {"sequences": [[1, 2]]},
        {"tokens": 2},
    )

    backend = LocalPEFTSamplingBackend(_worker=fake_worker)
    await backend.sample(
        adapter_name="sess_x",
        prompt_tokens=[1, 2, 3],
        max_tokens=5,
        top_k=40,
        stop=["END", "STOP"],
        seed=42,
    )
    _, kwargs_or_payload = fake_worker._handle_sample.call_args.args
    assert kwargs_or_payload["seed"] == 42
    assert kwargs_or_payload["stop"] == ["END", "STOP"]
    assert kwargs_or_payload["top_k"] == 40


async def test_local_peft_backend_raises_without_worker():
    backend = LocalPEFTSamplingBackend()
    with pytest.raises(RuntimeError, match="not connected"):
        await backend.sample(
            adapter_name="x",
            prompt_tokens=[1],
            max_tokens=1,
        )


async def test_local_peft_publish_is_noop():
    backend = LocalPEFTSamplingBackend()
    await backend.publish_adapter(adapter_name="x", adapter_path="/y")


async def test_sampled_sequence_defaults():
    seq = SampledSequence(tokens=[1, 2, 3])
    assert seq.stop_reason == "length"
    assert seq.logprobs is None
    assert seq.text is None
