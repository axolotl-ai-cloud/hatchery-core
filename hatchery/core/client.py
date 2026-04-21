# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Python client SDK for the Hatchery gateway.

This module implements Hatchery's own async/pipelined client against the
``/api/v1/*`` futures-based endpoints — the same surface the Tinker SDK
speaks, but without importing the ``tinker`` package. Use this when you
want pipelined training on Hatchery without picking up the upstream SDK as
a runtime dependency.

Shape
-----
Every training op (``forward_backward``, ``optim_step``, ``sample``,
``save_weights``) returns an :class:`_HatcheryFuture` immediately after the
submit POST. The future's background poll runs against
``/api/v1/retrieve_future`` (long-poll, 45s per HTTP request, retried on
408/5xx) and the completed payload is cached on the future. Pipelining
works the same way as the Tinker SDK:

.. code-block:: python

    client = HatcheryClient(base_url, token)
    tc = client.create_lora_training_client("Qwen/Qwen2-0.5B-Instruct", rank=32)

    pending = []
    for batch in batches:
        pending.append(tc.forward_backward(batch, "cross_entropy"))
        pending.append(tc.optim_step(learning_rate=1e-4))
    results = [f.result(timeout=300) for f in pending]

The ``.result()`` payload is whatever ``/api/v1/retrieve_future``
returned — see :func:`hatchery.core.tinker_compat._wrap_future_result` for
the per-op schema. That matches the Tinker SDK response shapes so code
written against the SDK can lift straight over by swapping the client
import.

Sync vs. async
--------------
The sync facade (``tc.forward_backward(...)``) pins all network and
poll activity onto a shared background event loop
(:class:`_BackgroundLoop`) so callers in sync Python code can pipeline
without needing their own asyncio harness. Async-native callers use the
``*_async`` variants: ``await tc.forward_backward_async(...)`` returns
an :class:`_HatcheryFuture` whose poll also runs in the background loop,
but whose result can be awaited directly via ``await future`` or
``await future.result_async()``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Optional

import httpx

logger = logging.getLogger("hatchery.core.client")


class HatcheryClientError(RuntimeError):
    """Raised when the gateway returns a non-2xx response."""


class RequestFailedError(RuntimeError):
    """Raised when ``/retrieve_future`` returns ``type: request_failed``."""


# Per-retrieve_future HTTP timeout. Matches the Tinker SDK (45s) so the
# long-poll loop exits often enough to update queue-state observers and
# short enough that a stuck server doesn't pin a connection.
_RETRIEVE_HTTP_TIMEOUT_S = 45.0


class _BackgroundLoop:
    """Runs an asyncio event loop on a dedicated thread.

    Gives the sync facade somewhere to schedule coroutines without
    blocking the caller's thread. A single loop is shared by every
    client instance so HTTP connection pools can be reused.
    """

    _instance: Optional[_BackgroundLoop] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="hatchery-client-loop", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @classmethod
    def get(cls) -> _BackgroundLoop:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class _HatcheryFuture:
    """Future for an ``/api/v1/*`` op submitted to the Hatchery gateway.

    The poll coroutine always runs on :class:`_BackgroundLoop`, so both
    sync (``.result()``) and async (``await future``) access are safe
    from any thread or event loop. Construction happens after the
    submit POST succeeds — the ``request_id`` comes from the gateway's
    ``_future_response`` envelope.
    """

    def __init__(self, client: HatcheryClient, request_id: str, operation: str) -> None:
        self._client = client
        self.request_id = request_id
        self.operation = operation
        self._cf: Future = _BackgroundLoop.get().submit(self._poll())

    async def _poll(self) -> dict:
        """Long-poll ``/api/v1/retrieve_future`` until the job resolves.

        Retries 408 (server-side long-poll timeout) and 5xx transparently;
        raises on 4xx != 408 so user errors surface promptly. Treats
        ``{"type": "try_again"}`` and ``{"status": "complete_metadata"}``
        as "keep polling" per the gateway contract.
        """
        allow_metadata_only = True
        iteration = 0
        while True:
            body = {
                "request_id": self.request_id,
                "allow_metadata_only": allow_metadata_only,
            }
            try:
                result = await self._client._post(
                    "/api/v1/retrieve_future",
                    body,
                    timeout=_RETRIEVE_HTTP_TIMEOUT_S,
                    _retrieve=True,
                )
            except httpx.TimeoutException:
                iteration += 1
                continue
            except HatcheryClientError as exc:
                # 408 and 5xx are retryable; 410 means the promise expired.
                status = getattr(exc, "status_code", None)
                if status == 408 or (status is not None and 500 <= status < 600):
                    iteration += 1
                    continue
                raise

            # Gateway says "not ready yet, try again". Same semantics as
            # a 408 — just keep polling.
            if isinstance(result, dict) and result.get("type") == "try_again":
                iteration += 1
                continue

            # Metadata-only responses come back once; the second poll
            # fetches the full payload. See tinker_compat retrieve_future.
            if isinstance(result, dict) and result.get("status") == "complete_metadata":
                allow_metadata_only = False
                continue

            if isinstance(result, dict) and result.get("type") == "request_failed":
                raise RequestFailedError(
                    f"{self.operation} request failed for {self.request_id}: "
                    f"{result.get('error', 'unknown error')}"
                )

            return result

    # ── Sync + async result access ─────────────────────────────────────

    def result(self, timeout: Optional[float] = None) -> dict:
        """Block until the future resolves. Safe to call from sync code."""
        return self._cf.result(timeout=timeout)

    async def result_async(self, timeout: Optional[float] = None) -> dict:
        """Await the future's result from an event loop."""
        aw = asyncio.wrap_future(self._cf)
        if timeout is None:
            return await aw
        return await asyncio.wait_for(aw, timeout)

    def __await__(self):
        return self.result_async().__await__()

    def done(self) -> bool:
        return self._cf.done()

    def cancel(self) -> bool:
        return self._cf.cancel()


# ─── Client ─────────────────────────────────────────────────────────────


def _bg_sync(coro) -> Any:
    """Run a coroutine on the shared background loop, block for result."""
    return _BackgroundLoop.get().submit(coro).result()


async def _bg_aio(coro) -> Any:
    """Run a coroutine on the shared background loop, await from any loop.

    If the caller is *already* on the bg loop (e.g. from inside an
    ``_HatcheryFuture._poll``), skip the round-trip and just ``await`` the
    coroutine directly. That avoids double-scheduling and the subtle
    ``run_coroutine_threadsafe``-from-the-loop-thread corner case.
    """
    bg = _BackgroundLoop.get()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is bg.loop:
        return await coro
    return await asyncio.wrap_future(bg.submit(coro))


class HatcheryClient:
    """Hatchery gateway client with Tinker-compatible futures semantics.

    All HTTP traffic (submits, polls, and long-poll retrieves) is driven
    from a shared background event loop (:class:`_BackgroundLoop`), so
    both sync callers and callers already inside an event loop can use
    the same client instance without loop-affinity conflicts in the
    underlying ``httpx.AsyncClient`` connection pool.

    Parameters
    ----------
    base_url:
        Gateway root (e.g. ``https://hatchery-gateway.example.com``).
    token:
        Bearer token for authentication.
    timeout:
        Default HTTP timeout for non-poll requests.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # Lazily constructed inside the background loop so the client
        # binds to that loop's httpx connection pool.
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client.

        httpx.AsyncClient lives on the bg loop, so closing must happen
        there too — ``_bg_aio`` handles the dispatch from any caller.
        """
        await _bg_aio(self._aclose_on_bg())

    async def _aclose_on_bg(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def close(self) -> None:
        _bg_sync(self._aclose_on_bg())

    async def _post(
        self,
        path: str,
        body: dict,
        *,
        timeout: Optional[float] = None,
        _retrieve: bool = False,
    ) -> dict:
        return await _bg_aio(self._post_on_bg(path, body, timeout=timeout))

    async def _post_on_bg(
        self,
        path: str,
        body: dict,
        *,
        timeout: Optional[float] = None,
    ) -> dict:
        cl = await self._ensure_client()
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await cl.post(path, json=body, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            err = HatcheryClientError(f"{resp.status_code} POST {path}: {detail}")
            err.status_code = resp.status_code  # type: ignore[attr-defined]
            raise err
        return resp.json() if resp.content else {}

    async def _get(self, path: str, **kwargs: Any) -> dict:
        return await _bg_aio(self._get_on_bg(path, **kwargs))

    async def _get_on_bg(self, path: str, **kwargs: Any) -> dict:
        cl = await self._ensure_client()
        resp = await cl.get(path, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            err = HatcheryClientError(f"{resp.status_code} GET {path}: {detail}")
            err.status_code = resp.status_code  # type: ignore[attr-defined]
            raise err
        return resp.json() if resp.content else {}

    # ── Session creation ───────────────────────────────────────────────

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: Optional[int] = 32,
        *,
        lora_alpha: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = False,
    ) -> TrainingClient:
        """Create a LoRA training session.

        ``rank=None`` flips to full-parameter fine-tuning (``lora_config``
        sent as ``null`` over the wire).
        """
        sess_resp = await self._post(
            "/api/v1/create_session",
            {"tags": [], "user_metadata": {}},
        )
        session_id = sess_resp["session_id"]

        lora_config: Optional[dict[str, Any]]
        if rank is None:
            lora_config = None
        else:
            lora_config = {
                "rank": rank,
                "train_attn": train_attn,
                "train_mlp": train_mlp,
                "train_unembed": train_unembed,
            }
            if lora_alpha is not None:
                lora_config["lora_alpha"] = lora_alpha

        model_resp = await self._post(
            "/api/v1/create_model",
            {
                "session_id": session_id,
                "base_model": base_model,
                "lora_config": lora_config,
            },
        )
        # ``/create_model`` returns the resolved ``model_id`` inline — we
        # don't need to retrieve the future for this op.
        model_id = model_resp.get("model_id")
        if not model_id:
            raise HatcheryClientError(f"create_model: missing model_id in response: {model_resp}")
        return TrainingClient(self, model_id, base_model, rank)

    def create_lora_training_client(
        self,
        base_model: str,
        rank: Optional[int] = 32,
        *,
        lora_alpha: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = False,
    ) -> TrainingClient:
        return _bg_sync(
            self.create_lora_training_client_async(
                base_model,
                rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
        )

    async def create_full_param_training_client_async(self, base_model: str) -> TrainingClient:
        return await self.create_lora_training_client_async(base_model, rank=None)

    def create_full_param_training_client(self, base_model: str) -> TrainingClient:
        return _bg_sync(self.create_full_param_training_client_async(base_model))

    async def list_sessions_async(self) -> list[dict]:
        """List active sessions.

        No ``/api/v1/*`` equivalent exists today, so this falls back to
        the ``/v1/sessions`` endpoint.
        """
        r = await self._get("/v1/sessions")
        return r.get("sessions", [])

    def list_sessions(self) -> list[dict]:
        return _bg_sync(self.list_sessions_async())


class TrainingClient:
    """Per-model handle. Every training op returns an :class:`_HatcheryFuture`."""

    def __init__(
        self,
        client: HatcheryClient,
        model_id: str,
        base_model: str,
        rank: Optional[int],
    ) -> None:
        self._client = client
        # ``session_id`` is an alias for ``model_id`` preserved
        # for back-compat — the gateway treats them as the same handle.
        self.session_id = model_id
        self.model_id = model_id
        self.base_model = base_model
        self.rank = rank
        self._seq_lock = threading.Lock()
        self._seq_id = 0

    def _next_seq(self) -> int:
        """Monotonic per-client seq_id, used for gateway idempotency."""
        with self._seq_lock:
            self._seq_id += 1
            return self._seq_id

    async def _submit(self, path: str, body: dict, operation: str) -> _HatcheryFuture:
        resp = await self._client._post(path, body)
        request_id = resp.get("request_id") or resp.get("future_id")
        if not request_id:
            raise HatcheryClientError(f"{operation}: no request_id in response: {resp}")
        return _HatcheryFuture(self._client, request_id, operation)

    # ── Training ops ───────────────────────────────────────────────────

    async def forward_backward_async(
        self,
        data: list,
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict] = None,
    ) -> _HatcheryFuture:
        normalized = [d.model_dump() if hasattr(d, "model_dump") else d for d in data]
        body: dict[str, Any] = {
            "model_id": self.model_id,
            "seq_id": self._next_seq(),
            "forward_backward_input": {
                "data": normalized,
                "loss_fn": loss_fn,
            },
        }
        if loss_fn_config is not None:
            body["forward_backward_input"]["loss_fn_config"] = loss_fn_config
        return await self._submit("/api/v1/forward_backward", body, "forward_backward")

    def forward_backward(
        self,
        data: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict] = None,
    ) -> _HatcheryFuture:
        return _bg_sync(self.forward_backward_async(data, loss_fn, loss_fn_config))

    async def forward_only_async(
        self,
        data: list,
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict] = None,
    ) -> _HatcheryFuture:
        normalized = [d.model_dump() if hasattr(d, "model_dump") else d for d in data]
        body: dict[str, Any] = {
            "model_id": self.model_id,
            "seq_id": self._next_seq(),
            "forward_only_input": {
                "data": normalized,
                "loss_fn": loss_fn,
            },
        }
        if loss_fn_config is not None:
            body["forward_only_input"]["loss_fn_config"] = loss_fn_config
        return await self._submit("/api/v1/forward_only", body, "forward_only")

    def forward_only(
        self,
        data: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict] = None,
    ) -> _HatcheryFuture:
        return _bg_sync(self.forward_only_async(data, loss_fn, loss_fn_config))

    async def optim_step_async(
        self,
        adam_params: Any = None,
        *,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 0.0,
        grad_accumulation_normalization: Optional[str] = None,
    ) -> _HatcheryFuture:
        if adam_params is not None:
            if hasattr(adam_params, "model_dump"):
                params = adam_params.model_dump()
            elif isinstance(adam_params, dict):
                params = adam_params
            else:
                params = {
                    k: getattr(adam_params, k)
                    for k in (
                        "learning_rate",
                        "beta1",
                        "beta2",
                        "eps",
                        "weight_decay",
                        "grad_clip_norm",
                    )
                    if hasattr(adam_params, k)
                }
        else:
            params = {
                "learning_rate": learning_rate,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "weight_decay": weight_decay,
                "grad_clip_norm": grad_clip_norm,
            }
        body: dict[str, Any] = {
            "model_id": self.model_id,
            "seq_id": self._next_seq(),
            "adam_params": params,
        }
        if grad_accumulation_normalization is not None:
            body["grad_accumulation_normalization"] = grad_accumulation_normalization
        return await self._submit("/api/v1/optim_step", body, "optim_step")

    def optim_step(self, adam_params: Any = None, **kwargs: Any) -> _HatcheryFuture:
        return _bg_sync(self.optim_step_async(adam_params, **kwargs))

    async def sample_async(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[list[Any]] = None,
    ) -> _HatcheryFuture:
        body: dict[str, Any] = {
            "model_id": self.model_id,
            "seq_id": self._next_seq(),
            "prompt": {
                "chunks": [{"type": "encoded_text", "tokens": list(prompt_tokens)}],
            },
            "num_samples": n,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "stop": stop,
            },
        }
        return await self._submit("/api/v1/asample", body, "sample")

    def sample(self, prompt_tokens: list[int], **kwargs: Any) -> _HatcheryFuture:
        return _bg_sync(self.sample_async(prompt_tokens, **kwargs))

    async def save_weights_async(self, name: str) -> _HatcheryFuture:
        body = {
            "model_id": self.model_id,
            "path": name,
            "seq_id": self._next_seq(),
        }
        return await self._submit("/api/v1/save_weights", body, "save_weights")

    def save_weights(self, name: str) -> _HatcheryFuture:
        return _bg_sync(self.save_weights_async(name))

    async def save_weights_for_sampler_async(self, name: str) -> _HatcheryFuture:
        body = {
            "model_id": self.model_id,
            "path": name,
            "seq_id": self._next_seq(),
        }
        return await self._submit(
            "/api/v1/save_weights_for_sampler", body, "save_weights_for_sampler"
        )

    def save_weights_for_sampler(self, name: str) -> _HatcheryFuture:
        return _bg_sync(self.save_weights_for_sampler_async(name))

    async def save_state_async(self, name: str) -> _HatcheryFuture:
        body = {
            "model_id": self.model_id,
            "path": name,
            "seq_id": self._next_seq(),
        }
        return await self._submit("/api/v1/save_state", body, "save_state")

    def save_state(self, name: str) -> _HatcheryFuture:
        return _bg_sync(self.save_state_async(name))

    async def load_weights_async(self, path: str, *, optimizer: bool = False) -> _HatcheryFuture:
        body = {
            "model_id": self.model_id,
            "path": path,
            "optimizer": optimizer,
            "seq_id": self._next_seq(),
        }
        return await self._submit("/api/v1/load_weights", body, "load_weights")

    def load_weights(self, path: str, *, optimizer: bool = False) -> _HatcheryFuture:
        return _bg_sync(self.load_weights_async(path, optimizer=optimizer))

    async def list_checkpoints_async(self) -> list[str]:
        r = await self._client._get(f"/api/v1/training_runs/{self.model_id}/checkpoints")
        return [c["checkpoint_id"] for c in r.get("checkpoints", [])]

    def list_checkpoints(self) -> list[str]:
        return _bg_sync(self.list_checkpoints_async())

    def save_weights_and_get_sampling_client(self, name: Optional[str] = None) -> SamplingClient:
        """Save current weights and return a SamplingClient for inference.

        Matches the official Tinker SDK's
        ``TrainingClient.save_weights_and_get_sampling_client()``.
        """
        fut = self.save_weights_for_sampler(name or f"sampler-{self._next_seq()}")
        result = fut.result(timeout=60)
        sampling_session_id = result.get("sampling_session_id")
        if not sampling_session_id:
            sampling_session_id = result.get("path", "").replace("tinker://", "").split("/")[0]
        return SamplingClient(
            client=self._client,
            model_id=self.model_id,
            sampling_session_id=sampling_session_id,
        )


class SamplingClient:
    """Inference client for a saved model checkpoint.

    Returned by ``TrainingClient.save_weights_and_get_sampling_client()``.
    Provides ``sample()`` and ``compute_logprobs()`` against the saved
    weights, matching the official Tinker SDK's ``SamplingClient``.
    """

    def __init__(
        self,
        client: HatcheryClient,
        model_id: str,
        sampling_session_id: str,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.sampling_session_id = sampling_session_id
        self._seq_lock = threading.Lock()
        self._seq_id = 0

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq_id += 1
            return self._seq_id

    async def sample_async(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[list[Any]] = None,
    ) -> _HatcheryFuture:
        body: dict[str, Any] = {
            "sampling_session_id": self.sampling_session_id,
            "seq_id": self._next_seq(),
            "prompt": {
                "chunks": [{"type": "encoded_text", "tokens": list(prompt_tokens)}],
            },
            "num_samples": n,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "stop": stop or [],
            },
        }
        resp = await self._client._post("/api/v1/asample", body)
        future_id = resp.get("future_id") or resp.get("request_id")
        return _HatcheryFuture(self._client, future_id, "sample")

    def sample(self, prompt_tokens: list[int], **kwargs: Any) -> _HatcheryFuture:
        return _bg_sync(self.sample_async(prompt_tokens, **kwargs))
