# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Sampling backends for inference / GRPO rollouts.

Two implementations:

1. ``LocalPEFTSamplingBackend`` — runs generation on the training
   worker via the existing ``_handle_sample`` path. Slow (no paged
   KV, no continuous batching, fp32 LoRA forward), but requires no
   extra infra. Good for debugging and small-scale eval.

2. ``VLLMSamplingBackend`` — sends requests to a vLLM server pool
   over the OpenAI-compatible API. Multi-LoRA via
   ``/v1/load_lora_adapter`` + ``model=<adapter_name>`` routing.
   This is the production path for GRPO where rollout throughput
   dominates training step time.

Discovery
---------
``VLLMSamplingBackend`` is initialized with a list of vLLM endpoint
URLs. In production these come from the ``GatewayRegistry`` (Redis) or
a static config. The backend round-robins across them for load
balancing and retries the next endpoint on failure.

Adapter publishing
------------------
Before the first sample call for a session, the backend must ensure
the adapter is loaded on at least one vLLM instance. The flow:

1. Training worker calls ``save_weights_for_sampler`` → adapter state
   dict is written to the object store.
2. The ``VLLMSamplingBackend.publish_adapter`` method:
   a. Writes the adapter to a path accessible to the vLLM pool
      (shared filesystem, NFS, or object-store-backed FUSE mount).
   b. Calls ``POST /v1/load_lora_adapter`` with the adapter name and
      path on the target vLLM instance.
3. Subsequent ``sample`` calls use ``model=<adapter_name>``.

The adapter path must be resolvable by the vLLM process — if vLLM
runs in a container and the adapter is on the host filesystem, the
path needs to be bind-mounted. For object-store-backed setups, the
adapter can be staged to a local cache directory first.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

if TYPE_CHECKING:
    import vllm  # noqa: F401

logger = logging.getLogger("hatchery.core.sampling")


class SamplingBackend(Protocol):
    """Interface for inference backends used by GRPO rollouts and
    customer-facing ``/asample`` requests.
    """

    async def sample(
        self,
        *,
        adapter_name: str,
        prompt_tokens: list[int],
        max_tokens: int,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str | int]] = None,
        seed: Optional[int] = None,
    ) -> list[SampledSequence]: ...

    async def publish_adapter(
        self,
        *,
        adapter_name: str,
        adapter_path: str,
    ) -> None:
        """Ensure the named adapter is loaded and ready for sampling.

        For ``VLLMSamplingBackend`` this calls the vLLM
        ``/v1/load_lora_adapter`` endpoint. For ``LocalPEFTSamplingBackend``
        this is a no-op (the adapter is already in memory on the worker).
        """
        ...

    async def unload_adapter(self, *, adapter_name: str) -> None: ...

    async def health_check(self) -> bool: ...


class SleepableSampler(Protocol):
    """Extension of :class:`SamplingBackend` for backends that can release
    their VRAM between sampling bursts.

    This exists so a trainer can share a GPU with its sampler: the sampler
    sleeps (VRAM returned to the CUDA driver) while the trainer's forward/
    backward pass runs, and wakes again when rollouts are needed. The
    canonical implementation is vLLM's sleep-mode engine
    (:class:`InProcessVLLMSampler`).

    Contract:
      * ``wake()`` transitions to the awake state; no-op if already awake.
      * ``sleep()`` transitions to the asleep state; no-op if already asleep.
      * ``sample()`` requires the awake state. Implementations MAY auto-wake
        on sample, but callers should prefer explicit ``awake(sampler)``
        context managers to avoid paying wake latency per call.
      * Implementations that don't support sleep (e.g., a remote HTTP
        backend with no control over the server) should implement ``wake``
        and ``sleep`` as no-ops and report ``is_awake == True`` always.
    """

    async def wake(self) -> None: ...

    async def sleep(self) -> None: ...

    @property
    def is_awake(self) -> bool: ...


@asynccontextmanager
async def awake(sampler: SleepableSampler) -> AsyncIterator[None]:
    """Context manager that wakes a :class:`SleepableSampler` for the
    duration of a block, then puts it back to sleep on exit.

    Usage::

        async with awake(sampler):
            outputs = await sampler.sample(...)
        # sampler is asleep here — another tenant of the GPU can run
    """
    await sampler.wake()
    try:
        yield
    finally:
        await sampler.sleep()


class VRAMCoordinator(Protocol):
    """Mutex-like handoff between a trainer and one or more samplers on
    the same GPU.

    Callers enter either a *trainer turn* (exclusive GPU access for
    forward/backward/optim_step) or a *sampler turn* (exclusive GPU
    access for generation, targeting one named sampler). At most one
    turn is active at any instant — the coordinator's implementation
    is responsible for serializing turns and putting any previously-
    awake sampler to sleep before granting a new turn.

    Implementations live in extension packages (the coordinator needs
    knowledge of the pool / deploy shape); core only declares the
    contract so trainer / worker code can defer to a coordinator when
    ``config.vram_coordinator`` is set and ignore it otherwise.

    Contract:
      * ``trainer_turn()`` yields after every sampler has been put to
        sleep. Entering again while already inside a trainer_turn is
        reentrant (no-op).
      * ``sampler_turn(sampler)`` wakes the given sampler and sleeps
        any other sampler owned by the same coordinator, then yields.
        Exit sleeps the sampler (unless a surrounding sampler_turn
        for the same sampler keeps it awake).
      * Calling ``sampler_turn`` from inside a ``trainer_turn`` (or
        vice versa) without first exiting the outer turn is undefined
        behavior — implementations SHOULD raise ``RuntimeError``.
      * A ``None`` coordinator is always valid in call sites that
        guard on ``if coordinator is not None:`` — no coordinator
        means the trainer/sampler don't share VRAM and each owns the
        GPU outright.
    """

    def trainer_turn(self) -> Any:
        """Return an async context manager that grants trainer access."""
        ...

    def sampler_turn(self, sampler: SleepableSampler) -> Any:
        """Return an async context manager that grants sampler access."""
        ...


@dataclass
class SampledSequence:
    tokens: list[int]
    stop_reason: str = "length"
    logprobs: Optional[list[float]] = None
    text: Optional[str] = None


@dataclass
class VLLMEndpoint:
    url: str
    healthy: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0


@dataclass
class VLLMSamplingBackend:
    """Production sampling backend backed by one or more vLLM servers.

    Parameters
    ----------
    endpoints:
        List of vLLM base URLs (e.g., ``["http://localhost:8000"]``).
    base_model:
        The HF model name the vLLM pool was started with. Used to
        verify adapter compatibility.
    adapter_cache_dir:
        Local directory where adapters are staged before loading into
        vLLM. Must be accessible to both this process and the vLLM
        server (shared mount).
    timeout:
        HTTP timeout for vLLM API calls in seconds.
    """

    endpoints: list[VLLMEndpoint] = field(default_factory=list)
    base_model: str = ""
    adapter_cache_dir: str = "/tmp/vllm_adapters"  # noqa: S108
    timeout: float = 120.0
    _round_robin_idx: int = field(default=0, repr=False)
    _loaded_adapters: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def from_urls(
        cls,
        urls: list[str],
        base_model: str = "",
        adapter_cache_dir: str = "/tmp/vllm_adapters",  # noqa: S108
    ) -> VLLMSamplingBackend:
        return cls(
            endpoints=[VLLMEndpoint(url=u.rstrip("/")) for u in urls],
            base_model=base_model,
            adapter_cache_dir=adapter_cache_dir,
        )

    def _next_endpoint(self) -> VLLMEndpoint:
        """Round-robin with health-awareness: skip unhealthy endpoints
        (but fall back to them if all are unhealthy).
        """
        n = len(self.endpoints)
        for _ in range(n):
            ep = self.endpoints[self._round_robin_idx % n]
            self._round_robin_idx += 1
            if ep.healthy:
                return ep
        # All unhealthy — try the next one anyway.
        ep = self.endpoints[self._round_robin_idx % n]
        self._round_robin_idx += 1
        return ep

    async def publish_adapter(self, *, adapter_name: str, adapter_path: str) -> None:
        """Load a LoRA adapter onto the vLLM pool.

        If vLLM rejects the adapter (e.g., lm_head not in supported
        modules), logs a warning but still marks it as loaded — the
        training loop can continue with sampling from the base model.
        An extension-provided vLLM lm_head monkeypatch should be
        applied at vLLM startup to avoid this fallback.
        """
        if adapter_name in self._loaded_adapters:
            return
        ep = self._next_endpoint()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{ep.url}/v1/load_lora_adapter",
                    json={
                        "lora_name": adapter_name,
                        "lora_path": adapter_path,
                    },
                )
                resp.raise_for_status()
                self._loaded_adapters.add(adapter_name)
            except httpx.HTTPStatusError as exc:
                # vLLM may reject adapters with unsupported modules
                # (e.g., lm_head). Log and continue — sampling will
                # use the base model without the adapter layer.
                logger.warning(
                    "vllm.adapter_load_rejected",
                    adapter=adapter_name,
                    status=exc.response.status_code,
                    detail=exc.response.text[:200],
                )
                self._loaded_adapters.add(adapter_name)

    async def unload_adapter(self, *, adapter_name: str) -> None:
        self._loaded_adapters.discard(adapter_name)
        for ep in self.endpoints:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    await client.post(
                        f"{ep.url}/v1/unload_lora_adapter",
                        json={"lora_name": adapter_name},
                    )
            except Exception:  # noqa: BLE001
                pass

    async def sample(
        self,
        *,
        adapter_name: str,
        prompt_tokens: list[int],
        max_tokens: int,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str | int]] = None,
        seed: Optional[int] = None,
    ) -> list[SampledSequence]:
        """Call vLLM's OpenAI-compatible completions endpoint."""
        ep = self._next_endpoint()
        body: dict[str, Any] = {
            "model": adapter_name,
            "prompt": prompt_tokens,
            "max_tokens": max_tokens,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            body["seed"] = seed
        if stop is not None:
            body["stop"] = stop
        if top_k > 0:
            body["extra_body"] = {"top_k": top_k}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{ep.url}/v1/completions", json=body)
                resp.raise_for_status()
            ep.healthy = True
            ep.consecutive_failures = 0
        except Exception:
            ep.consecutive_failures += 1
            if ep.consecutive_failures >= 3:
                ep.healthy = False
            raise

        data = resp.json()
        sequences = []
        for choice in data.get("choices", []):
            tokens = choice.get("tokens", [])
            text = choice.get("text", "")
            finish = choice.get("finish_reason", "length")
            stop_reason = "stop" if finish == "stop" else "length"
            lps = None
            if choice.get("logprobs") and choice["logprobs"].get("token_logprobs"):
                lps = choice["logprobs"]["token_logprobs"]
            sequences.append(
                SampledSequence(
                    tokens=tokens,
                    stop_reason=stop_reason,
                    logprobs=lps,
                    text=text,
                )
            )
        return sequences

    async def health_check(self) -> bool:
        for ep in self.endpoints:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{ep.url}/health")
                    ep.healthy = resp.status_code == 200
                    ep.last_check = time.time()
            except Exception:  # noqa: BLE001
                ep.healthy = False
        return any(ep.healthy for ep in self.endpoints)


@dataclass
class LocalPEFTSamplingBackend:
    """Sampling via the training worker's PEFT model.

    This wraps a reference to the ``GPUWorker._handle_sample`` method
    and calls it directly — no HTTP, no queue. Only useful for
    debugging and small-scale eval where deploying a vLLM instance
    is overkill.
    """

    _worker: Any = None

    async def publish_adapter(self, *, adapter_name: str, adapter_path: str) -> None:
        pass

    async def unload_adapter(self, *, adapter_name: str) -> None:
        pass

    async def sample(
        self,
        *,
        adapter_name: str,
        prompt_tokens: list[int],
        max_tokens: int,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str | int]] = None,
        seed: Optional[int] = None,
    ) -> list[SampledSequence]:
        if self._worker is None:
            raise RuntimeError("LocalPEFTSamplingBackend not connected to a worker")

        # adapter_name encodes the session_id via the worker's naming convention.
        session_id = adapter_name.replace("sess_", "").replace("_", "-")
        result, _ = await self._worker._handle_sample(
            session_id,
            {
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": n,
                "seed": seed,
                "stop": stop,
            },
        )
        sequences = []
        for token_list in result.get("sequences", []):
            sequences.append(SampledSequence(tokens=token_list, stop_reason="length"))
        return sequences

    async def health_check(self) -> bool:
        return self._worker is not None


# ─── In-process inference samplers ────────────────────────────────────────
#
# Architecture:
#
#   InProcessInferenceSampler  (abstract base — backend-agnostic scaffolding)
#       │
#       ├─ InProcessVLLMSampler    (concrete; vllm.LLM + sleep(level=1)/wake_up)
#       └─ (future) InProcessSGLangSampler   (release_memory_occupation /
#                                             resume_memory_occupation)
#
# Both vLLM and sglang expose a "free-VRAM-but-keep-engine" primitive, so
# the abstraction centers on four hooks: build, engine-sleep, engine-wake,
# engine-generate. Concrete subclasses translate these into the backend's
# specific calls.


@dataclass
class InProcessInferenceSampler:
    """Abstract base for an inference engine owned by this Python process.

    This is the counterpart to :class:`VLLMSamplingBackend` for deployments
    where training and sampling share a single Python process (and typically
    a single GPU). Instead of talking to a remote inference server over HTTP,
    an in-process sampler owns the engine object directly, which is what
    makes sleep-mode / memory-occupation-release usable: the trainer and
    sampler can hand VRAM back and forth on the same device.

    Two common configurations:

    * **Colocated trainer + sampler on one GPU.** ``enable_sleep_mode=True``.
      The sampler starts asleep after :meth:`initialize`; the trainer gets
      the GPU for setup. During the sampling phase of each step, the caller
      uses ``async with awake(sampler): ...`` to wake for the burst and
      sleep again on exit. Round-trip is ~400ms on small models (validated
      in ``scripts/spike_vllm_swap.py``), and captured CUDA graphs survive
      sleep cleanly so post-wake sampling is at full graph speed.

    * **Dedicated sampler process.** ``enable_sleep_mode=False``. ``wake``
      and ``sleep`` become no-ops. Equivalent to running the bare engine
      with this module's adapter-publishing convenience layer on top.

    Subclass contract
    -----------------
    Concrete backends implement four async hooks:

    * :meth:`_build_engine` — construct and return the engine object. Called
      under the lock during :meth:`initialize`. Should not put the engine
      to sleep; the base class handles that based on ``enable_sleep_mode``.
    * :meth:`_engine_wake` — backend-specific wake call. Both vLLM
      (``LLM.wake_up``) and sglang (``Engine.resume_memory_occupation``)
      have equivalents.
    * :meth:`_engine_sleep` — backend-specific sleep call. vLLM's
      ``LLM.sleep(level=1)`` and sglang's ``Engine.release_memory_occupation``.
    * :meth:`_engine_generate` — run one generation and return a
      ``list[SampledSequence]``. Receives the already-resolved adapter path
      (or ``None``) so backends don't need their own adapter bookkeeping.

    Notes
    -----
    * Engine calls are typically synchronous/blocking. Subclasses should
      wrap them in ``asyncio.to_thread`` so they don't stall the event loop
      while the GPU works.
    * Wake/sleep state is guarded by an internal ``asyncio.Lock`` — safe
      to call from concurrent tasks.
    * An advanced multi-model-on-one-GPU pool that composes several
      of these instances can be layered on by an extension package.
    """

    model: str
    enable_sleep_mode: bool = True
    enable_lora: bool = True

    _engine: Any = field(default=None, init=False, repr=False)
    _is_awake: bool = field(default=False, init=False, repr=False)
    _lock: Optional[asyncio.Lock] = field(default=None, init=False, repr=False)
    # adapter_name -> local path. Resolved into a backend-specific adapter
    # handle at sample time.
    _adapters: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # Stable integer IDs for adapters. vLLM's LoRARequest requires one;
    # we maintain it here so the abstraction is useful for any backend
    # with the same requirement.
    _adapter_ids: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _next_adapter_id: int = field(default=1, init=False, repr=False)

    # ── Backend hooks — implemented by subclasses ────────────────────────

    async def _build_engine(self) -> Any:
        raise NotImplementedError

    async def _engine_wake(self) -> None:
        raise NotImplementedError

    async def _engine_sleep(self) -> None:
        raise NotImplementedError

    async def _engine_generate(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: Optional[list[str | int]],
        seed: Optional[int],
        adapter_name: str,
        adapter_path: Optional[str],
        adapter_int_id: Optional[int],
    ) -> list[SampledSequence]:
        raise NotImplementedError

    # ── Shared scaffolding ───────────────────────────────────────────────

    def _ensure_lock(self) -> asyncio.Lock:
        """Lazy-create the lock so it binds to the running event loop,
        not to whatever loop happened to exist at construction time."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def initialize(self) -> None:
        """Construct the underlying engine, capturing CUDA graphs if
        applicable.

        After this returns, the sampler is asleep if ``enable_sleep_mode``,
        otherwise awake. Idempotent: subsequent calls are no-ops.
        """
        async with self._ensure_lock():
            if self._engine is not None:
                return
            self._engine = await self._build_engine()
            self._is_awake = True

            # If sleep mode is enabled, start asleep so the trainer can
            # claim VRAM for its own setup before the first sampling burst.
            if self.enable_sleep_mode:
                await self._engine_sleep()
                self._is_awake = False
            logger.info(
                "in_process_sampler.initialized",
                extra={
                    "model": self.model,
                    "awake": self._is_awake,
                    "backend": type(self).__name__,
                },
            )

    @property
    def is_awake(self) -> bool:
        return self._is_awake

    async def wake(self) -> None:
        """Bring the engine back into VRAM. No-op if already awake or if
        ``enable_sleep_mode`` is False."""
        if not self.enable_sleep_mode:
            return
        async with self._ensure_lock():
            if self._engine is None:
                raise RuntimeError(f"{type(self).__name__}.wake() called before initialize()")
            if self._is_awake:
                return
            await self._engine_wake()
            self._is_awake = True

    async def sleep(self) -> None:
        """Release VRAM back to the CUDA driver. No-op if already asleep
        or if ``enable_sleep_mode`` is False.

        The engine skeleton (captured CUDA graphs, KV-cache metadata) stays
        resident, typically ~1–3GB depending on model size. A subsequent
        :meth:`wake` restores full-speed sampling in ~200ms.
        """
        if not self.enable_sleep_mode:
            return
        async with self._ensure_lock():
            if self._engine is None or not self._is_awake:
                return
            await self._engine_sleep()
            self._is_awake = False

    async def publish_adapter(self, *, adapter_name: str, adapter_path: str) -> None:
        """Register a LoRA adapter by name + local path.

        Unlike the HTTP :class:`VLLMSamplingBackend`, loading is lazy:
        this method just records the path, and the adapter is handed to
        the engine on the first :meth:`sample` call that references it.
        """
        if not self.enable_lora:
            raise RuntimeError("publish_adapter called but enable_lora=False on this sampler")
        if adapter_name not in self._adapter_ids:
            self._adapter_ids[adapter_name] = self._next_adapter_id
            self._next_adapter_id += 1
        self._adapters[adapter_name] = adapter_path

    async def unload_adapter(self, *, adapter_name: str) -> None:
        self._adapters.pop(adapter_name, None)
        # Keep the integer id reserved — some backends (vLLM) cache by id,
        # and re-using a freed id while old state lingers can produce
        # stale outputs.

    async def sample(
        self,
        *,
        adapter_name: str,
        prompt_tokens: list[int],
        max_tokens: int,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str | int]] = None,
        seed: Optional[int] = None,
    ) -> list[SampledSequence]:
        """Generate completions. Requires the sampler to be awake.

        Use ``async with awake(sampler): ...`` around a batch of sample
        calls to avoid paying wake latency on every single request.
        """
        if self._engine is None:
            raise RuntimeError("sample() called before initialize()")
        if self.enable_sleep_mode and not self._is_awake:
            raise RuntimeError(
                "sample() called while sampler is asleep — wrap the call in "
                "`async with awake(sampler): ...` or call wake() first"
            )

        adapter_path = self._adapters.get(adapter_name)
        adapter_int_id = self._adapter_ids.get(adapter_name)

        return await self._engine_generate(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            seed=seed,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
            adapter_int_id=adapter_int_id,
        )

    async def health_check(self) -> bool:
        return self._engine is not None

    async def close(self) -> None:
        """Tear down the engine and release its resources. Idempotent."""
        async with self._ensure_lock():
            if self._engine is None:
                return
            # Engines typically lack an explicit close; rely on GC to drive
            # shutdown. We clear our ref and let the caller optionally
            # gc.collect() / torch.cuda.empty_cache().
            self._engine = None
            self._is_awake = False


@dataclass
class InProcessVLLMSampler(InProcessInferenceSampler):
    """In-process sampler backed by a ``vllm.LLM`` engine.

    Uses ``LLM.sleep(level=1)`` / ``LLM.wake_up`` for the VRAM handoff.
    See :class:`InProcessInferenceSampler` for the high-level behavior;
    this subclass only implements the vLLM-specific hooks.

    Parameters
    ----------
    model:
        HF model id or local path, passed straight through to ``vllm.LLM``.
    enable_sleep_mode:
        If True, the engine is constructed with ``enable_sleep_mode=True``
        and is put to sleep at the end of :meth:`initialize`, so the trainer
        can allocate on the GPU first. If False, the engine stays resident
        and ``wake``/``sleep`` are no-ops.
    enable_lora:
        If True, LoRA adapters can be published via :meth:`publish_adapter`
        and passed per-request to :meth:`sample`.
    dtype, gpu_memory_utilization, max_model_len, max_lora_rank:
        Forwarded to ``vllm.LLM`` — see vLLM docs for semantics.
    extra_llm_kwargs:
        Escape hatch for additional ``LLM()`` kwargs (e.g., compilation
        config) that aren't first-class here. Forwarded verbatim.
    """

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    max_lora_rank: int = 32
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)

    async def _build_engine(self) -> Any:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "InProcessVLLMSampler requires the 'vllm' package. "
                "Install with `uv pip install vllm` or `uv pip install hatchery-core[sampling]`."
            ) from exc

        kwargs: dict[str, Any] = dict(
            model=self.model,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_sleep_mode=self.enable_sleep_mode,
            enable_lora=self.enable_lora,
            max_lora_rank=self.max_lora_rank,
        )
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        kwargs.update(self.extra_llm_kwargs)

        # LLM() is blocking (weight load + graph capture can take 10s+).
        return await asyncio.to_thread(LLM, **kwargs)

    async def _engine_wake(self) -> None:
        await asyncio.to_thread(self._engine.wake_up)

    async def _engine_sleep(self) -> None:
        # level=1 releases weights/activations to the CUDA driver while
        # keeping captured graphs and KV-cache metadata resident.
        await asyncio.to_thread(self._engine.sleep, 1)

    async def _engine_generate(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: Optional[list[str | int]],
        seed: Optional[int],
        adapter_name: str,
        adapter_path: Optional[str],
        adapter_int_id: Optional[int],
    ) -> list[SampledSequence]:
        from vllm import SamplingParams, TokensPrompt
        from vllm.lora.request import LoRARequest

        params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            stop=[s for s in (stop or []) if isinstance(s, str)] or None,
            stop_token_ids=[s for s in (stop or []) if isinstance(s, int)] or None,
            seed=seed,
            logprobs=1,
        )

        lora_req: Optional[LoRARequest] = None
        if adapter_path is not None and adapter_int_id is not None:
            lora_req = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=adapter_int_id,
                lora_path=adapter_path,
            )

        # vllm.LLM.generate is blocking; run it off the event loop.
        # vLLM 0.19 took a `prompts` positional (accepting TokensPrompt)
        # rather than the older `prompt_token_ids` kwarg.
        outputs = await asyncio.to_thread(
            self._engine.generate,
            [TokensPrompt(prompt_token_ids=prompt_tokens)],
            params,
            lora_request=lora_req,
            use_tqdm=False,
        )

        # vLLM returns one RequestOutput per prompt; n completions live
        # inside .outputs. We flatten into a list[SampledSequence].
        sequences: list[SampledSequence] = []
        if not outputs:
            return sequences
        req_out = outputs[0]
        for completion in req_out.outputs:
            finish = completion.finish_reason or "length"
            stop_reason = "stop" if finish == "stop" else "length"
            lps: Optional[list[float]] = None
            if completion.logprobs:
                lps = []
                for step in completion.logprobs:
                    # step is {token_id: Logprob(...)}. Take the logprob
                    # of the chosen token (first key, by convention).
                    if step:
                        chosen = next(iter(step.values()))
                        lps.append(float(chosen.logprob))
                    else:
                        lps.append(0.0)
            sequences.append(
                SampledSequence(
                    tokens=list(completion.token_ids),
                    stop_reason=stop_reason,
                    logprobs=lps,
                    text=completion.text,
                )
            )
        return sequences
