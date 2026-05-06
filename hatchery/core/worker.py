# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""GPU worker — loads a base model, pulls jobs, runs LoRA training.

Workers are the only component that touches GPU memory. They are
ephemeral by design: session state lives in the object store, so the
worker can die (spot preemption, BIOS update, OOM) and another worker
resumes by downloading state and re-running the failed job.

Architecture:
* Exactly one base model is loaded per worker.
* LoRA adapters for individual sessions are added on demand via
  ``peft_model.add_adapter(session_id, lora_config)``.
* A cache (e.g. :class:`~hatchery.core.scheduler.SmartLoRACache`) tracks
  which session adapters are live on the GPU. On eviction, the adapter
  is deleted from the PEFT model to free VRAM.
* ``forward_backward`` accumulates gradients in a separate CPU dict and
  persists them so ``optim_step`` can resume regardless of which worker
  picks it up next.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import msgpack
import structlog
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from hatchery.core.batching import (
    BatchStrategy,
    prepare_batch_for_dp,
)
from hatchery.core.config import Config
from hatchery.core.distributed import DistributedRuntime, init_distributed_runtime
from hatchery.core.optim_dispatch import (
    build_optimizer,
    select_optimizer_kind,
    vram_free_bytes,
)
from hatchery.core.parallel import ParallelConfig
from hatchery.core.protocols import (
    JobResult,
    JobStatus,
    QueuedJob,
    WorkerInfo,
)
from hatchery.core.scheduler import SmartLoRACache, StepCadenceTracker

logger = structlog.get_logger("hatchery.core.worker")

# VLM detection helpers live in model_pool — they describe a freshly
# loaded base model, which is the pool's concern. Re-exported here so
# existing ``from hatchery.core.worker import _is_vlm_model`` imports
# (tests, downstream code) keep working.
from hatchery.core.model_pool import (  # noqa: E402, F401
    _VLM_CLASS_NAMES,
    _get_vision_token_ids,
    _is_vlm_model,
)


class _DistributedCommandBus:
    """Thin wrapper around torch.distributed collectives used by GPUWorker."""

    def __init__(self, runtime: DistributedRuntime) -> None:
        self.runtime = runtime

    def broadcast(self, value: Any) -> Any:
        import torch.distributed as dist

        box = [value]
        dist.broadcast_object_list(box, 0)
        return box[0]

    def gather_errors(self, error: Optional[str]) -> list[Optional[str]]:
        import torch.distributed as dist

        gathered: list[Optional[str]] = [None for _ in range(self.runtime.world_size)]
        dist.all_gather_object(gathered, error)
        return gathered


def _strip_vision_tokens(input_ids: list[int], vision_ids: set[int]) -> list[int]:
    """Remove vision placeholder tokens from a token list."""
    if not vision_ids:
        return input_ids
    return [t for t in input_ids if t not in vision_ids]


@dataclass
class _SessionRuntime:
    """In-RAM state for an active session on this worker."""

    session_id: str
    # ``lora_config`` is None for full-parameter sessions — they don't
    # carry adapter metadata. ``training_mode`` is the authoritative
    # branch flag; ``lora_config`` presence is just the consequence.
    lora_config: Optional[LoraConfig] = None
    training_mode: str = "lora"
    grad_accum: dict[str, torch.Tensor] = field(default_factory=dict)
    optimizer_state: Optional[dict] = None
    meta: dict = field(default_factory=lambda: {"accum_steps": 0, "total_steps": 0})
    # Intermediate state for forward_backward_custom — keyed by client
    # ``custom_id`` so step2 can replay the same collated batch as step1.
    # Not persisted; lives only on the worker that ran step1.
    custom_cache: dict = field(default_factory=dict)
    # Delta-compression bookkeeping (see hatchery.core.lora_state).
    # ``snapshot_cache`` is the bf16 copy of the last full snapshot on
    # disk; we diff against it on each save to produce a delta file.
    snapshot_cache: Optional[dict] = None
    snapshot_version: int = 0
    delta_count: int = 0
    # Optimizer-state compression bookkeeping (see
    # hatchery.core.optimizer_state). Independent from the LoRA snapshot
    # counters — rollover cadence and disk files are separate.
    optim_snapshot_cache: Optional[dict] = None
    optim_snapshot_version: int = 0
    optim_delta_count: int = 0
    # Cached tokenized input_ids / labels are too expensive to round-trip
    # through msgpack — but that's the gateway's problem, not the worker's.


class GPUWorker:
    """Ephemeral GPU worker.

    Parameters
    ----------
    worker_id:
        Unique identifier for this worker instance.
    base_model_name:
        HF hub name of the base model to load.
    config:
        :class:`Config` providing queue/object store/metadata backends.
    device:
        Torch device string, e.g. ``"cuda:0"``.
    dtype:
        Torch dtype for base model weights.
    max_lora_slots:
        Maximum number of session adapters to keep resident.
    attn_implementation:
        Transformers attention backend (``"eager"`` for max portability,
        ``"sdpa"`` for speed on recent GPUs, ``"flash_attention_2"`` when
        available).
    """

    def __init__(
        self,
        worker_id: str,
        base_model_name: str,
        config: Config,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_lora_slots: int = 20,
        attn_implementation: str = "sdpa",
        cadence_tracker: Optional[StepCadenceTracker] = None,
        load_model: bool = True,
        parallel: Optional[ParallelConfig] = None,
        pool: Any = None,
    ) -> None:
        self.worker_id = worker_id
        self.base_model_name = base_model_name
        self.config = config
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.parallel = parallel or ParallelConfig()
        self.running = True
        self._distributed_runtime = DistributedRuntime(
            global_rank=0,
            local_rank=0,
            dp_rank=0,
            world_size=1,
            dp_world_size=1,
            device=None,
        )
        self._command_bus: Optional[_DistributedCommandBus] = None
        self._mesh = None
        self._distributed_runtime_initialized = False

        self._raw_base: Any = None
        self._peft: Optional[PeftModel] = None
        self.tokenizer = None
        self.processor = None
        self.is_vlm = False
        self._vision_token_ids: set[int] = set()
        # Model pool: owns base-model lifecycle (load, cache, evict).
        # Injected for tests; in production we build the default pool
        # driven by HATCHERY_MODEL_POOL / HATCHERY_MODEL_POOL_MAX_SLOTS env
        # vars. For the current one-model-per-worker deploy we run with
        # max_vram_slots=1 so behavior is unchanged.
        self._pool = pool
        self._slot: Any = None

        self.cadence = cadence_tracker or StepCadenceTracker()
        self._cache = SmartLoRACache(max_slots=max_lora_slots, cadence_tracker=self.cadence)
        self._cache.on_evict = self._on_evict
        # Per-session (total_steps, accum_steps) tuple recorded at the
        # end of every _save_session_to_store on this worker. Used by
        # _is_cache_stale to skip a round-trip GET when we know we
        # wrote the most recent state ourselves — no other worker can
        # have advanced past us within the sticky affinity window.
        self._self_saved_versions: dict[str, tuple[int, int]] = {}

        # Per-full-param-session base weights, kept on CPU. Keyed by
        # session_id so the same set of snapshots survives multi-model
        # slot swaps. Sized like the base model (~1 GB / 0.5B in bf16),
        # so we hold at most a handful of FP sessions at once. The
        # active session id lives on ``self._slot`` so it tracks
        # whichever slot is currently routed.
        self._fp_base_state: dict[str, dict[str, torch.Tensor]] = {}

        if load_model:
            self._init_distributed_runtime()

        # Two-tier session state store: fast local disk for hot-path
        # mutations, asynchronous mirror to the configured remote store.
        # See hatchery/core/session_store.py for the invariants.
        from hatchery.core.backends.object_store.local import LocalObjectStore
        from hatchery.core.session_store import (
            default_local_root,
            ensure_local_root,
        )

        local_root_path = default_local_root(worker_id)
        if self._is_distributed and not self._is_rank0:
            local_root_path = os.path.join(
                local_root_path, f"rank-{self._distributed_runtime.global_rank}"
            )
        local_root = ensure_local_root(local_root_path)
        # The session store binds both a local cache (hot path) and an
        # optional remote mirror. Core ships a local-only variant; the
        # Extension packages can override ``Config.build_session_store``
        # to substitute a mirrored variant that writes to
        # ``config.objects`` asynchronously.
        self._state = config.build_session_store(
            local=LocalObjectStore(root=local_root),
            worker=self,
        )

        if load_model:
            self._load_base_model()

        self._model_lock = asyncio.Lock()

        # Generic lifecycle extension points. Extension packages
        # attach behavior here (e.g. flush coordinators, peer
        # watchers) without core needing to know what they do.
        # ``install_worker_hooks`` is called at the start
        # of ``run()`` and gives the platform config a chance to
        # register its handlers.
        self._on_start_hooks: list[Any] = []
        self._pre_load_session_hooks: list[Any] = []

    @property
    def _is_distributed(self) -> bool:
        return self._distributed_runtime.is_distributed

    @property
    def _is_rank0(self) -> bool:
        return self._distributed_runtime.global_rank == 0

    def _persists_external_state(self) -> bool:
        return not self._is_distributed or self._is_rank0

    def _init_distributed_runtime(self) -> None:
        if self._distributed_runtime_initialized:
            return
        self._distributed_runtime = init_distributed_runtime(self.parallel)
        self._mesh = self._distributed_runtime.mesh
        if self._distributed_runtime.device is not None:
            self.device = str(self._distributed_runtime.device)
        if self._distributed_runtime.is_distributed:
            self._command_bus = _DistributedCommandBus(self._distributed_runtime)
        self._distributed_runtime_initialized = True

    # ── Model loading ─────────────────────────────────────────

    def _load_base_model(self) -> None:
        logger.info("worker.loading_model", model=self.base_model_name, device=self.device)

        self._init_distributed_runtime()

        # Delegate base-model / tokenizer / VLM processor loading to
        # the pool. With max_vram_slots=1 (the default) this is
        # functionally identical to the old inline load; with a
        # tiered pool + max_host_slots>0, swaps between base models
        # skip the HF loader round-trip.
        if self._pool is None:
            from hatchery.core.model_pool import build_default_model_pool

            self._pool = build_default_model_pool(
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )

        slot = self._pool.get_or_load(self.base_model_name)
        self._slot = slot
        self._raw_base = slot.raw_base
        self.tokenizer = slot.tokenizer
        self.processor = slot.processor
        self.is_vlm = slot.is_vlm
        self._vision_token_ids = slot.vision_token_ids
        if self.is_vlm and slot.processor is not None:
            logger.info(
                "worker.vlm_processor_loaded",
                model=self.base_model_name,
                vision_tokens=len(self._vision_token_ids),
            )

        # Selective fp32 upcast for precision-sensitive submodules
        # (MoE routers, optionally embeddings). Idempotent per slot —
        # once a slot has been upcast, promoting it back from the
        # host tier must not re-apply (the param dtypes are already
        # correct and re-running would be a no-op, but the flag
        # makes that explicit and the log single-shot).
        if not slot.precision_applied:
            from hatchery.core.precision import apply_precision_policy

            report = apply_precision_policy(self._raw_base, main_dtype=self.dtype)
            if report.total:
                logger.info(
                    "worker.precision_policy_applied",
                    upcast_modules=len(report.upcast_modules),
                    upcast_embeddings=len(report.upcast_embeddings),
                )
            slot.precision_applied = True

        # NOTE: FSDP2 wrapping is deferred until after the first PEFT
        # adapter is attached in _attach_adapter. Wrapping here (before
        # PEFT) converts q_proj.weight into a DTensor whose ``out_features``
        # reads as the local shard size; PEFT then sizes lora_B against
        # that shard instead of the full dimension, producing the
        # "tensor a (896) must match tensor b (448)" error. The same
        # pattern is used in VanillaTrainer._apply_parallel_plan.
        self._parallel_applied = slot.parallel_applied

        # Capture pristine base weights at boot, before any session can
        # mutate them. This is the "insurance policy" for FP reload after
        # eviction: if the adapter file hasn't flushed to object store
        # when the SmartLoRACache evicts the runtime, the reload path
        # falls back to pristine_sd instead of hitting a KeyError. The
        # capture is idempotent (``_ensure_pristine_snapshot`` is a
        # no-op when the slot already has a snapshot), so host-tier
        # promotions that already populated pristine_sd don't re-run.
        self._ensure_pristine_snapshot()

    # ── Registration / lifecycle ──────────────────────────────

    async def register(self) -> None:
        if not self._persists_external_state():
            return
        info = WorkerInfo(
            worker_id=self.worker_id,
            provider="local",
            gpu_type=torch.cuda.get_device_name(self._torch_device_index())
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu",
            gpu_count=1,
            loaded_models=[self.base_model_name],
            status="idle",
            max_concurrent_loras=self._cache.max_slots,
            vram_free_mb=self._vram_free_mb(),
            cp_degree=self.parallel.cp_degree,
        )
        register = getattr(self.config.compute, "register_worker", None)
        if register is not None:
            await register(info)

    async def heartbeat(self, status: str = "idle") -> None:
        if not self._persists_external_state():
            return
        heartbeat = getattr(self.config.compute, "heartbeat", None)
        if heartbeat is not None:
            await heartbeat(
                self.worker_id,
                status=status,
                vram_free_mb=self._vram_free_mb(),
            )

    def _torch_device_index(self) -> int:
        if not self.device.startswith("cuda"):
            return 0
        _, _, idx = self.device.partition(":")
        return int(idx or "0")

    def _vram_free_mb(self) -> int:
        if not (self.device.startswith("cuda") and torch.cuda.is_available()):
            return 0
        free, _total = torch.cuda.mem_get_info(self._torch_device_index())
        return int(free / (1024 * 1024))

    # ── Main loop ─────────────────────────────────────────────

    async def run(self, *, max_jobs: Optional[int] = None) -> None:
        """Main worker loop.

        On a single-rank worker this just pulls jobs from the queue in
        a tight loop. On a multi-rank (FSDP/TP) worker group launched
        via torchrun, rank 0 is the *coordinator* and the other ranks
        are *followers* — see :meth:`_run_coordinator` /
        :meth:`_run_follower` for the details of how job data flows
        across the collective.
        """
        await self.register()
        logger.info("worker.registered", worker_id=self.worker_id, model=self.base_model_name)
        # Give the platform config a chance to attach extension
        # behavior (e.g. flush coordinators, peer watchers). Core
        # has no opinion about what the hooks do.
        installer = getattr(self.config, "install_worker_hooks", None)
        if installer is not None:
            maybe_coro = installer(self)
            if hasattr(maybe_coro, "__await__"):
                await maybe_coro
        for hook in list(self._on_start_hooks):
            try:
                await hook()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "worker.on_start_hook_failed",
                    worker_id=self.worker_id,
                )
        if self._is_distributed:
            self._command_bus = self._command_bus or _DistributedCommandBus(
                self._distributed_runtime
            )
            if self._is_rank0:
                await self._run_coordinator(max_jobs=max_jobs)
            else:
                await self._run_follower(max_jobs=max_jobs)
            return

        processed = 0
        idle_polls = 0
        logger.info("worker.dequeue_loop.start", worker_id=self.worker_id)
        while self.running:
            if max_jobs is not None and processed >= max_jobs:
                break
            try:
                job = await self.config.queue.dequeue(
                    worker_id=self.worker_id,
                    model_filter=self.base_model_name,
                    visibility_timeout=300,
                )
            except Exception as exc:
                logger.exception(
                    "worker.dequeue.error",
                    worker_id=self.worker_id,
                    err=f"{type(exc).__name__}: {exc}",
                )
                # Back off a bit so we don't spin on a persistent error
                # (e.g. Redis momentarily unavailable) but keep running.
                await asyncio.sleep(2.0)
                continue
            if job is None:
                idle_polls += 1
                # Heartbeat log every ~30s (600 * 50ms) so container log
                # viewers have proof-of-life while the queue is empty.
                if idle_polls % 600 == 0:
                    logger.info(
                        "worker.dequeue.idle",
                        worker_id=self.worker_id,
                        polls=idle_polls,
                    )
                await asyncio.sleep(0.05)
                continue
            logger.info(
                "worker.dequeue.got_job",
                worker_id=self.worker_id,
                job_id=job.job_id,
                operation=getattr(job, "operation", None),
            )
            # Reject jobs that require more CP parallelism than we support.
            if job.required_cp_degree > self.parallel.cp_degree:
                await self.config.queue.nack(
                    job.job_id,
                    f"Worker cp_degree={self.parallel.cp_degree}, "
                    f"job requires {job.required_cp_degree}",
                )
                continue
            await self._process_one(job)
            processed += 1

    # ── Multi-rank coordination ──────────────────────────────

    async def _run_coordinator(self, *, max_jobs: Optional[int]) -> None:
        """Rank-0 loop.

        Pulls jobs from the queue, broadcasts them to every follower
        rank via ``dist.broadcast_object_list``, runs the op on rank 0
        so it participates in FSDP collectives alongside the followers,
        and acks/nacks the result. If ``dequeue`` returns ``None`` we
        still have to broadcast a ``None`` sentinel so followers don't
        hang forever on their matching broadcast call.

        Note: we call ``broadcast_object_list`` synchronously from the
        event-loop thread rather than via ``asyncio.to_thread``.
        ``torch.cuda.set_device`` is thread-local, so running the NCCL
        op on a worker thread means the thread has no active CUDA
        device and NCCL binds rank 0 and rank 1 to device 0 — which
        shows up as "Duplicate GPU detected". The broadcasts are tiny
        (a few KB), so briefly blocking the loop is fine.
        """
        if not self._persists_external_state():
            return
        bus = self._command_bus or _DistributedCommandBus(self._distributed_runtime)
        processed = 0
        while self.running:
            if max_jobs is not None and processed >= max_jobs:
                bus.broadcast({"type": "shutdown"})
                break

            try:
                job = await self.config.queue.dequeue(
                    worker_id=self.worker_id,
                    model_filter=self.base_model_name,
                    visibility_timeout=300,
                )
            except Exception as exc:
                logger.exception(
                    "worker.dequeue.error",
                    worker_id=self.worker_id,
                    err=f"{type(exc).__name__}: {exc}",
                )
                bus.broadcast({"type": "idle"})
                await asyncio.sleep(2.0)
                continue
            if job is None:
                bus.broadcast({"type": "idle"})
                await asyncio.sleep(0.05)
                continue
            if job.required_cp_degree > self.parallel.cp_degree:
                bus.broadcast({"type": "idle"})
                await self.config.queue.nack(
                    job.job_id,
                    f"Worker cp_degree={self.parallel.cp_degree}, "
                    f"job requires {job.required_cp_degree}",
                )
                continue

            signal = {
                "type": "job",
                "job_id": job.job_id,
                "session_id": job.session_id,
                "operation": job.operation,
                "payload": job.payload,
                "user_id": job.user_id,
                "preferred_worker": job.preferred_worker,
                "required_model": job.required_model,
                "required_cp_degree": job.required_cp_degree,
            }
            bus.broadcast(signal)
            t0 = time.time()
            result: Optional[JobResult] = None
            local_error: Optional[str] = None
            try:
                result = await self._execute_job(job)
            except Exception as exc:  # noqa: BLE001
                local_error = self._format_job_error(exc)
            gathered = bus.gather_errors(local_error)
            collective_error = next((err for err in gathered if err), None)
            if collective_error is not None:
                await self._nack_failed_job(job, collective_error)
            elif result is not None and result.status == JobStatus.COMPLETED:
                await self._ack_completed_job(job, result, t0)
            elif result is not None:
                await self.config.queue.ack(job.job_id, result)
            processed += 1

    async def _run_follower(self, *, max_jobs: Optional[int]) -> None:
        """Rank > 0 loop.

        Blocks on ``dist.broadcast_object_list`` for each iteration
        waiting for rank 0 to hand us a job. Runs the op on the local
        rank so FSDP's forward/backward collectives stay balanced. We
        never touch the queue and never ack — rank 0 owns that.
        """
        bus = self._command_bus or _DistributedCommandBus(self._distributed_runtime)
        processed = 0
        while self.running:
            data = bus.broadcast(None)
            if data is None or data.get("type") == "idle":
                continue
            if data.get("type") == "shutdown":
                break
            fake_job = QueuedJob(
                job_id=data["job_id"],
                session_id=data["session_id"],
                operation=data["operation"],
                payload=data["payload"],
                user_id=data.get("user_id"),
                preferred_worker=data.get("preferred_worker"),
                required_model=data.get("required_model"),
                required_cp_degree=data.get("required_cp_degree", 1),
            )
            local_error: Optional[str] = None
            try:
                # Don't ack/nack — rank 0 handles that. We only need to
                # run the forward/backward so our FSDP collectives
                # complete in sync with rank 0's call. Note: no outer
                # lock acquire here — ``_execute_job`` already holds
                # ``_model_lock``, and asyncio.Lock is not reentrant.
                await self._execute_job(fake_job)
            except Exception as exc:  # noqa: BLE001
                local_error = self._format_job_error(exc)
                logger.exception("follower.execute_failed", job_id=fake_job.job_id)
            bus.gather_errors(local_error)
            processed += 1

    async def process_next(self, timeout: float = 1.0) -> bool:
        """Process a single queued job. Returns ``True`` if one was handled."""
        if self._is_distributed and not self._is_rank0:
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = await self.config.queue.dequeue(
                worker_id=self.worker_id,
                model_filter=self.base_model_name,
                visibility_timeout=300,
            )
            if job is not None:
                await self._process_one(job)
                return True
            await asyncio.sleep(0.02)
        return False

    async def _process_one(self, job: QueuedJob) -> None:
        t0 = time.time()
        try:
            result = await self._execute_job(job)
            if result.status == JobStatus.COMPLETED:
                await self._ack_completed_job(job, result, t0)
            else:
                await self.config.queue.ack(job.job_id, result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker.job_failed", job_id=job.job_id)
            await self._nack_failed_job(job, self._format_job_error(exc))

    def _format_job_error(self, exc: BaseException) -> str:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=30))
        return f"{type(exc).__name__}: {exc}\n{tb}"

    async def _ack_completed_job(self, job: QueuedJob, result: JobResult, t0: float) -> None:
        if not self._persists_external_state():
            return
        # Grad-accum hard-pin management must happen *before* ack so
        # no peer can race in and dequeue the next job for this
        # session between our ack and our pin set. Ops that leave
        # grad_accum live on local disk pin the session; optim_step
        # (which drains it) clears.
        await self._update_accum_pin(job)
        await self.config.queue.ack(job.job_id, result)
        duration_ms = (time.time() - t0) * 1000
        self.cadence.record(job.session_id, job.operation, duration_ms)
        with contextlib.suppress(KeyError):
            await self.config.metadata.update_session(
                job.session_id,
                last_worker_id=self.worker_id,
                avg_step_duration_ms=self.cadence.avg_duration_ms(job.session_id),
            )
        # Update session registry for fast preferred_worker lookups.
        # ``session_registry`` is an extension-provided field (e.g.
        # a Redis-backed store); the core Config doesn't carry it,
        # so use getattr with a None default.
        session_registry = getattr(self.config, "session_registry", None)
        if session_registry is not None:
            with contextlib.suppress(Exception):
                await session_registry.set(job.session_id, self.worker_id)
        metrics = result.metrics or {}
        self.config.metrics.record_job_duration(
            session_id=job.session_id,
            user_id=job.user_id or "",
            operation=job.operation,
            duration_ms=duration_ms,
            tokens=metrics.get("tokens", 0),
            worker_id=self.worker_id,
            gpu_type=self._gpu_label(),
            cost_dimensions=metrics.get("cost_dimensions"),
        )

    async def _nack_failed_job(self, job: QueuedJob, error: str) -> None:
        if not self._persists_external_state():
            return
        await self.config.queue.nack(job.job_id, error)
        first_line = error.splitlines()[0] if error else "unknown"
        error_type = first_line.split(":", 1)[0] if ":" in first_line else first_line
        self.config.metrics.increment_counter(
            "job_failures",
            {"operation": job.operation, "error_type": error_type},
        )

    _ACCUM_PINNING_OPS = frozenset({"forward_backward", "forward_custom_step2"})

    async def _update_accum_pin(self, job: QueuedJob) -> None:
        """Set or clear the queue's grad_accum pin based on the op.

        ``forward_backward`` / ``forward_custom_step2`` leave live
        gradient state on this worker's local disk; a peer that
        inherits the session via visibility-timeout retry would read
        stale pre-accumulation state from the object store. Pinning
        the session to this worker blocks that handoff. ``optim_step``
        drains grad_accum and clears the pin.
        """
        queue = self.config.queue
        set_pin = getattr(queue, "set_accum_pin", None)
        clear_pin = getattr(queue, "clear_accum_pin", None)
        if job.operation in self._ACCUM_PINNING_OPS and set_pin is not None:
            with contextlib.suppress(Exception):
                await set_pin(job.session_id, self.worker_id)
        elif job.operation == "optim_step" and clear_pin is not None:
            with contextlib.suppress(Exception):
                await clear_pin(job.session_id)

    def _gpu_label(self) -> str:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.get_device_name(self._torch_device_index())
        return "cpu"

    def _build_cost_dimensions(
        self,
        runtime: _SessionRuntime,
        batch_size: int,
        max_seq_len: int,
        loss_fn: str = "",
        fused_path: bool = False,
    ) -> dict:
        """Assemble the internal cost-analysis dimensions for one job.

        This dict flows into ``record_job_duration(cost_dimensions=...)``
        and from there into the structured log stream / metrics store.
        The downstream cost-analysis pipeline uses these fields to build
        per-(model, seq_len_bucket, parallel_config) cost curves and
        detect pricing mismatches.
        """
        return {
            "model_name": self.base_model_name,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "lora_rank": runtime.lora_config.r if runtime.lora_config is not None else 0,
            "loss_fn": loss_fn,
            "fused_path": fused_path,
            "dp_degree": self.parallel.dp_degree,
            "tp_degree": self.parallel.tp_degree,
            "cp_degree": self.parallel.cp_degree,
            "is_context_parallel": self.parallel.cp_degree > 1,
        }

    # ── Job dispatch ──────────────────────────────────────────

    async def _execute_job(self, job: QueuedJob) -> JobResult:
        payload = await self._read_payload(job.payload)

        # Verify the scoped token (if present). Only enforced when
        # HATCHERY_INTERNAL_SECRET is set — unified mode and solo dev
        # Plugin payload verifiers (e.g. an extension-registered
        # scoped-token check). The verifier pops the token from the
        # payload before handlers run.
        from hatchery.core.plugins import verify_payload

        if isinstance(payload, dict):
            try:
                verify_payload(job, payload)
            except Exception as exc:  # noqa: BLE001
                return JobResult(
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    error=str(exc),
                )

        handlers = {
            "init_session": self._handle_init_session,
            "forward_backward": self._handle_forward_backward,
            "forward_only": self._handle_forward_only,
            "forward_custom_step1": self._handle_forward_custom_step1,
            "forward_custom_step2": self._handle_forward_custom_step2,
            "optim_step": self._handle_optim_step,
            "save_weights": self._handle_save_weights,
            "load_weights": self._handle_load_weights,
            "sample": self._handle_sample,
            "compute_logprobs": self._handle_compute_logprobs,
            "forward_logprobs": self._handle_forward_logprobs,
        }
        handler = handlers.get(job.operation)
        if handler is None:
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error=f"Unknown operation: {job.operation}",
            )
        t_gpu = time.time()
        async with self._model_lock:
            result_data, extra_metrics = await handler(job.session_id, payload)
        duration_ms = (time.time() - t_gpu) * 1000
        metrics = {"duration_ms": duration_ms, **extra_metrics}
        return JobResult(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            result=msgpack.packb(result_data, use_bin_type=True),
            metrics=metrics,
        )

    async def _read_payload(self, transport: bytes) -> dict:
        """Decode inline payload or fetch from object store.

        Heuristic: msgpack bytes from the gateway always begin with a
        dict/map marker (0x80-0x8f for fixmap, 0xde, 0xdf). If the first
        byte isn't one of those, the bytes represent an object-store key.
        """
        if not transport:
            return {}
        first = transport[0]
        is_msgpack = 0x80 <= first <= 0x8F or first in (0xDE, 0xDF)
        if is_msgpack:
            return msgpack.unpackb(transport, raw=False)
        payload_bytes = await self.config.objects.get(transport.decode())
        return msgpack.unpackb(payload_bytes, raw=False)

    # ── Cache / session lifecycle ─────────────────────────────

    def _adapter_name(self, session_id: str) -> str:
        # PEFT adapter names can't contain "-" freely; normalize.
        return "sess_" + session_id.replace("-", "_")

    # ── Mixed-mode (LoRA + full-param) base-weight management ─────
    #
    # Mirrors the design in :class:`hatchery.core.trainer.VanillaTrainer`.
    # See that file for the design rationale; the worker version keeps
    # the pristine snapshot on the pool slot (so multi-model swaps
    # don't lose it) and holds per-session full-param weights on the
    # worker keyed by session_id.

    @staticmethod
    def _is_lora_param_name(name: str) -> bool:
        return (
            ".lora_A." in name
            or ".lora_B." in name
            or ".lora_embedding_A." in name
            or ".lora_embedding_B." in name
        )

    def _ensure_pristine_snapshot(self) -> None:
        """Capture the current slot's pre-mutation base weights.

        Idempotent: subsequent calls are no-ops. Must run before any
        full-param session has trained on this slot — we call it from
        ``_attach_full_param_session`` and from the FP load path.
        """
        if self._slot is None or self._slot.pristine_sd is not None:
            return
        self._slot.pristine_sd = self._capture_live_base_weights()

    def _capture_live_base_weights(self) -> dict[str, torch.Tensor]:
        """Snapshot the base-portion weights of ``_raw_base`` to CPU.

        Skips LoRA adapter params (``lora_A``/``lora_B``) and strips
        ``.base_layer.`` from PEFT-wrapped keys so the dict is
        portable to a non-PEFT load path.
        """
        out: dict[str, torch.Tensor] = {}
        for k, v in self._raw_base.state_dict().items():
            if ".lora_A." in k or ".lora_B." in k:
                continue
            pristine_k = k.replace(".base_layer.", ".")
            out[pristine_k] = v.detach().cpu().clone()
        return out

    def _load_base_sd_into_live(self, base_sd: dict[str, torch.Tensor]) -> None:
        """Load a pre-PEFT-keyed base state dict into the live model.

        Inserts ``.base_layer.`` before the trailing param name when the
        live model has been PEFT-wrapped. ``strict=False`` so any LoRA
        adapter keys present in the live model are left alone.
        """
        live_keys = set(self._raw_base.state_dict().keys())
        payload: dict[str, torch.Tensor] = {}
        for k, v in base_sd.items():
            if k in live_keys:
                live_k = k
            else:
                head, _, tail = k.rpartition(".")
                wrapped = f"{head}.base_layer.{tail}" if head else k
                if wrapped not in live_keys:
                    continue
                live_k = wrapped
            payload[live_k] = v.to(self.device, dtype=v.dtype)
        self._raw_base.load_state_dict(payload, strict=False)

    def _restore_pristine_base(self) -> None:
        if self._slot is None or self._slot.pristine_sd is None:
            return
        self._load_base_sd_into_live(self._slot.pristine_sd)

    def _restore_full_param_base(self, session_id: str) -> None:
        sd = self._fp_base_state.get(session_id)
        if sd is None:
            self._restore_pristine_base()
            return
        self._load_base_sd_into_live(sd)

    def _stash_active_full_param_base(self) -> None:
        """If the slot's currently-active session is full-param, snapshot
        its live base weights back into ``_fp_base_state``."""
        if self._slot is None:
            return
        active = self._slot.active_session_id
        if active is None:
            return
        runtime = self._cache.get(active)
        if runtime is None or runtime.training_mode != "full_param":
            return
        self._fp_base_state[active] = self._capture_live_base_weights()

    def _activate_session(self, session_id: str, runtime: _SessionRuntime) -> Any:
        """Materialize the right base weights + adapter for this session.

        Returns the ``nn.Module`` to call for forward passes. For LoRA
        sessions returns ``self._peft`` with the right adapter set;
        for full-param sessions returns ``self._raw_base`` after
        loading the session's stashed weights (or pristine on first
        activation).
        """
        slot = self._slot
        is_fp = runtime.training_mode == "full_param"

        if slot is not None and slot.active_session_id != session_id:
            # Check if the outgoing session was full-param (mutated base).
            outgoing = self._cache.get(slot.active_session_id) if slot.active_session_id else None
            outgoing_is_fp = outgoing is not None and outgoing.training_mode == "full_param"
            # Outgoing: stash any in-flight FP weights from this slot.
            self._stash_active_full_param_base()
            # Incoming: load the right base weights for this session.
            if is_fp:
                self._restore_full_param_base(session_id)
            elif outgoing_is_fp:
                self._restore_pristine_base()
            slot.active_session_id = session_id
            self._set_grad_for_session(runtime)
        elif not is_fp and self._peft is not None:
            # Same LoRA session — re-set the adapter to be defensive
            # against any code path that may have flipped it.
            self._peft.set_adapter(self._adapter_name(session_id))

        if is_fp:
            return self._raw_base
        return self._peft

    def _set_grad_for_session(self, runtime: _SessionRuntime) -> None:
        if runtime.training_mode == "full_param":
            for name, p in self._raw_base.named_parameters():
                p.requires_grad_(not self._is_lora_param_name(name))
        else:
            adapter = self._adapter_name(runtime.session_id)
            for p in self._raw_base.parameters():
                p.requires_grad_(False)
            if self._peft is not None:
                self._peft.set_adapter(adapter)

    def _exec_context(self, runtime: _SessionRuntime):
        """Wrap a forward pass for this session.

        For full-param sessions on a slot that has been LoRA-wrapped,
        ``peft.disable_adapter`` zeros the adapter contribution so the
        forward uses base weights only.
        """
        from contextlib import nullcontext

        if runtime.training_mode == "full_param" and self._peft is not None:
            return self._peft.disable_adapter()
        return nullcontext()

    def _on_evict(self, session_id: str, runtime: _SessionRuntime) -> None:
        if runtime.training_mode == "full_param":
            # Drop the per-session CPU snapshot. If this session is the
            # one currently materialized into the slot, restore pristine
            # so the next session sees a clean base.
            self._fp_base_state.pop(session_id, None)
            if self._slot is not None and self._slot.active_session_id == session_id:
                self._restore_pristine_base()
                self._slot.active_session_id = None
        else:
            adapter = self._adapter_name(session_id)
            if self._peft is not None:
                try:
                    self._peft.delete_adapter(adapter)
                except Exception:  # noqa: BLE001
                    pass
            if self._slot is not None and self._slot.active_session_id == session_id:
                self._slot.active_session_id = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Eviction is a handoff boundary — any deferred remote mirror
        # writes must complete before another worker can reasonably
        # claim this session. We can't await from a sync callback, so
        # schedule a flush task; worker shutdown drain is the safety
        # net if the pod dies before this completes.
        if self._persists_external_state() and self._state.has_pending(session_id):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                loop.create_task(self._state.flush(session_id))

    async def _ensure_session_loaded(self, session_id: str) -> _SessionRuntime:
        """Ensure the session's adapter is live on the GPU with current weights.

        If the session is in the local SmartLoRACache, check whether
        another worker has advanced the training state since we last
        touched it (by comparing ``total_steps`` + ``accum_steps`` in
        object store meta against our cached copy). If stale, evict
        and reload from the object store.
        """
        runtime = self._cache.get(session_id)
        if runtime is not None:
            # Check freshness — another worker may have run steps.
            if await self._is_cache_stale(session_id, runtime):
                logger.info(
                    "worker.cache_stale_reload",
                    session_id=session_id,
                    worker_id=self.worker_id,
                )
                self._cache.evict(session_id)
                return await self._load_session_from_store(session_id)
            # Activation (LoRA adapter swap, FP base-weight swap) is
            # deferred to ``_active_module`` so the caller can wrap a
            # forward pass with the correct exec context.
            return runtime
        return await self._load_session_from_store(session_id)

    async def _is_cache_stale(self, session_id: str, runtime: _SessionRuntime) -> bool:
        """Check if the object store has a newer version than our cache.

        This can happen when:
        - Worker A caches session S, then works on other sessions
        - Worker B picks up session S jobs meanwhile (sticky affinity
          expired or A was busy), runs steps, saves to object store
        - Worker A gets session S again — local cache is stale

        Note: the single-in-flight-per-session queue invariant guarantees
        that by the time we dequeue a job, the previous worker has fully
        written its state to the object store (ack happens after save).
        So we never read a partially-written state here.
        """
        prefix = f"sessions/{session_id}/live_state"
        try:
            meta_bytes = await self.config.objects.get(f"{prefix}/session_meta.json")
            meta = json.loads(meta_bytes)
            store_steps = meta.get("total_steps", 0)
            store_accum = meta.get("accum_steps", 0)
            local_steps = runtime.meta.get("total_steps", 0)
            local_accum = runtime.meta.get("accum_steps", 0)
            # Stale if the store is ahead in either dimension.
            return store_steps > local_steps or (
                store_steps == local_steps and store_accum > local_accum
            )
        except (KeyError, json.JSONDecodeError):
            # Can't read meta — assume fresh (will fail downstream if
            # truly broken, but don't evict valid cache on transient errors).
            return False

    async def _load_session_from_store(self, session_id: str) -> _SessionRuntime:
        """Load a session's state from local disk, falling back to remote.

        Probes local ``session_meta.json`` first; if present, loads the
        whole fileset from local (sub-ms after kernel cache warms).
        If local doesn't have this session (fresh worker, freshly-
        evicted, or cross-worker handoff), falls back to the remote
        store. On remote fallback, we also write the fetched blobs
        through to local disk so the next cache miss is a hit.
        """
        from hatchery.core.lora_state import read_compression_meta

        prefix = f"sessions/{session_id}/live_state"
        t0 = time.time()

        # Pick the authoritative source: local if it has this session,
        # else remote. Before the remote read we give any registered
        # pre-load hooks a chance to run (e.g. a flush coordinator that
        # asks the current owner to push a fresh copy). All blobs for
        # this load come from the same source so we never mix a local
        # snapshot with a remote delta.
        try:
            meta_bytes = await self._state.local.get(f"{prefix}/session_meta.json")
            source = self._state.local
            source_label = "local"
        except KeyError:
            for hook in list(self._pre_load_session_hooks):
                try:
                    await hook(session_id)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "worker.pre_load_hook_failed",
                        session_id=session_id,
                    )
            meta_bytes = await self.config.objects.get(f"{prefix}/session_meta.json")
            source = self.config.objects
            source_label = "remote"

        meta = json.loads(meta_bytes)
        # Default mode for state files without ``training_mode`` (no ``training_mode`` field):
        # if a ``lora_config`` is present, it's a LoRA session.
        training_mode = meta.get("training_mode") or ("lora" if meta.get("lora_config") else "lora")

        lora_config: Optional[LoraConfig] = None
        if training_mode == "lora":
            lora_config_dict = meta["lora_config"]
            lora_kwargs = {
                "r": lora_config_dict["r"],
                "lora_alpha": lora_config_dict["lora_alpha"],
                "target_modules": lora_config_dict["target_modules"],
                "lora_dropout": lora_config_dict.get("lora_dropout", 0.0),
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            if lora_config_dict.get("use_rslora"):
                lora_kwargs["use_rslora"] = True
            init_weights = lora_config_dict.get("init_lora_weights", "default")
            if init_weights != "default":
                lora_kwargs["init_lora_weights"] = init_weights
            lora_config = LoraConfig(**lora_kwargs)

        # FP sessions that never trained skip the snapshot upload on
        # init (see ``_save_fp_init_marker``). On load, re-materialize
        # the pristine base from this slot rather than reading a
        # non-existent ``lora_weights.pt``.
        #
        # The ``fp_pristine_init`` marker is a fast-path hint: if set,
        # we know the weights blob was deliberately never written and
        # can go straight to pristine. If it's absent, the blob is
        # *supposed* to exist, but a cross-worker mirror race can land
        # the meta on remote before the weights blob — the outgoing
        # worker's ``_mirror_session`` uploads all keys concurrently,
        # so the order is non-deterministic. In that case we still
        # fall back to pristine rather than raising; the session has
        # at most one optim step's worth of divergence from pristine,
        # and the alternative is a hard failure of every FP reload
        # that races the mirror.
        fp_pristine_init = bool(meta.get("fp_pristine_init"))
        use_pristine_fallback = training_mode == "full_param" and fp_pristine_init
        logger.info(
            "worker.load_session.entering",
            session_id=session_id,
            source=source_label,
            training_mode=training_mode,
            fp_pristine_init=fp_pristine_init,
            use_pristine_fallback=use_pristine_fallback,
            total_steps=meta.get("total_steps", 0),
        )
        lora_state: Optional[dict] = None
        snapshot_cache: Optional[dict] = None
        snapshot_version = 0
        delta_count = 0
        if not use_pristine_fallback:
            try:
                (
                    lora_state,
                    snapshot_cache,
                    snapshot_version,
                    delta_count,
                ) = await self.config.lora_state.load(source, prefix)
                meta_ver, meta_dc = read_compression_meta(meta)
                if meta_ver:
                    snapshot_version = max(snapshot_version, meta_ver)
                    delta_count = meta_dc
            except KeyError as exc:
                logger.warning(
                    "worker.load_session.key_error",
                    session_id=session_id,
                    source=source_label,
                    training_mode=training_mode,
                    key=str(exc),
                )
                if training_mode != "full_param":
                    raise
                logger.warning(
                    "worker.fp_reload_pristine_fallback",
                    session_id=session_id,
                    source=source_label,
                    reason="lora_weights.pt missing — mirror race",
                )
                use_pristine_fallback = True
        if use_pristine_fallback:
            self._ensure_pristine_snapshot()
            assert self._slot is not None and self._slot.pristine_sd is not None
            lora_state = {k: v.clone() for k, v in self._slot.pristine_sd.items()}
            snapshot_cache = {k: v.to(torch.bfloat16) for k, v in lora_state.items()}
            snapshot_version = 0
            delta_count = 0
        assert lora_state is not None and snapshot_cache is not None

        try:
            grad_bytes = await source.get(f"{prefix}/grad_accum.pt")
            grad_accum = torch.load(io.BytesIO(grad_bytes), map_location="cpu", weights_only=True)
        except KeyError:
            grad_accum = {}

        (
            optimizer_state,
            optim_snapshot_cache,
            optim_snapshot_version,
            optim_delta_count,
        ) = await self.config.optimizer_state.load(source, prefix)
        meta_optim_ver = int(meta.get("optim_snapshot_version", 0) or 0)
        meta_optim_dc = int(meta.get("optim_delta_count", 0) or 0)
        if meta_optim_ver:
            optim_snapshot_version = max(optim_snapshot_version, meta_optim_ver)
            optim_delta_count = meta_optim_dc

        if training_mode == "full_param":
            self._attach_full_param_session(session_id, weights=lora_state)
        else:
            assert lora_config is not None
            self._attach_adapter(session_id, lora_config, lora_state)

        runtime = _SessionRuntime(
            session_id=session_id,
            lora_config=lora_config,
            training_mode=training_mode,
            grad_accum=grad_accum,
            optimizer_state=optimizer_state,
            meta=meta,
            snapshot_cache=snapshot_cache,
            snapshot_version=snapshot_version,
            delta_count=delta_count,
            optim_snapshot_cache=optim_snapshot_cache,
            optim_snapshot_version=optim_snapshot_version,
            optim_delta_count=optim_delta_count,
        )
        self._cache.put(session_id, runtime)

        duration_ms = (time.time() - t0) * 1000
        # Best-effort size metric — the snapshot bytes aren't separately
        # tracked on load, so fall back to the fp32 footprint estimate.
        approx_bytes = sum(v.numel() * 4 for v in lora_state.values())
        self.config.metrics.record_lora_swap_time(session_id, "load", duration_ms, approx_bytes)
        logger.info(
            "worker.load_session",
            session_id=session_id,
            source=source_label,
            duration_ms=round(duration_ms, 2),
        )
        return runtime

    def _select_fp_optimizer_kind(self) -> str:
        """Pick fp32 vs 8-bit AdamW for a new FP session on this worker.

        Reads the VRAM budget fraction from ``HATCHERY_FFT_OPTIMIZER_VRAM_BUDGET_FRAC``
        (default 0.40). Falls back to ``"adamw"`` if param count is unknown.
        torchao's ``AdamW8bit`` is DTensor-aware so this is safe under FSDP2.
        """
        if self._raw_base is None:
            return "adamw"
        try:
            param_count = sum(p.numel() for p in self._raw_base.parameters())
        except Exception:  # noqa: BLE001
            return "adamw"
        try:
            budget_frac = float(os.environ.get("HATCHERY_FFT_OPTIMIZER_VRAM_BUDGET_FRAC", "0.40"))
        except ValueError:
            budget_frac = 0.40
        return select_optimizer_kind(
            training_mode="full_param",
            trainable_param_count=param_count,
            vram_free_bytes=vram_free_bytes(self.device),
            vram_budget_frac=budget_frac,
        )

    def _enforce_fft_capacity(self, new_session_id: str) -> None:
        """Reject a new FP session if another FP session is already live.

        FFT carries a multi-GB optimizer/grad/base footprint per session;
        running two on the same worker risks OOM. The gate fires before
        any state is materialized so callers see a clean rejection.
        ``_fp_base_state`` is the authoritative live-FP tracker — entries
        are cleared on session unload.

        Zombie eviction: if the incumbent FP session has never received
        a training op (``meta.total_steps == 0``), the gateway either
        crashed or the client never got the session_id back. In either
        case it cannot be in active use, so evict it and admit the new
        session. Without this, a single failed create_model permanently
        wedges the worker until restart.
        """
        for sid in list(self._fp_base_state):
            if sid == new_session_id:
                continue
            rt = self._cache.get(sid)
            if rt is None or int(rt.meta.get("total_steps", 0) or 0) == 0:
                self._cache.evict(sid)
                self._fp_base_state.pop(sid, None)
                continue
            raise RuntimeError(
                f"worker already hosts FP session {sid}; single-FFT-per-worker capacity gate"
            )

    def _attach_full_param_session(
        self,
        session_id: str,
        weights: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """Initialize a full-parameter session on this worker.

        Captures the pristine snapshot if not yet taken (so future LoRA
        sessions on the same slot still see a clean base) and stages
        the session's CPU-resident base weights — either freshly cloned
        from pristine, or loaded from the supplied checkpoint.
        """
        # If a different FP session is currently materialized in the
        # slot, snapshot its weights before we (eventually) swap in
        # this session's. Doing it here, before pristine capture, is
        # safe because pristine_sd is independent of which FP session
        # is currently active — it captures the canonical base.
        self._stash_active_full_param_base()
        self._ensure_pristine_snapshot()
        if weights:
            self._fp_base_state[session_id] = {
                k: v.detach().cpu().clone() for k, v in weights.items()
            }
        elif session_id not in self._fp_base_state:
            assert self._slot is not None and self._slot.pristine_sd is not None
            self._fp_base_state[session_id] = {
                k: v.clone() for k, v in self._slot.pristine_sd.items()
            }

    def _attach_adapter(
        self,
        session_id: str,
        lora_config: LoraConfig,
        state_dict: dict,
    ) -> None:
        adapter = self._adapter_name(session_id)
        first_adapter = self._peft is None
        if first_adapter:
            self._peft = get_peft_model(self._raw_base, lora_config, adapter_name=adapter)
        elif adapter not in self._peft.peft_config:
            self._peft.add_adapter(adapter, lora_config)
        self._peft.set_adapter(adapter)

        if state_dict:
            set_peft_model_state_dict(self._peft, state_dict, adapter_name=adapter)

        # Persist the PEFT instance back to the pool slot so multi-model
        # workers don't lose adapters across base-model swaps. The slot
        # is the durable carrier — ``self._peft`` mirrors whatever slot
        # is active right now.
        if self._slot is not None:
            self._slot.peft_model = self._peft

        # Apply FSDP2 / TP / CP plan on the first adapter. Must happen
        # AFTER PEFT wraps the base so lora_A/lora_B are sized against
        # the full hidden dim, not a sharded DTensor view. Same shape
        # trap as VanillaTrainer._apply_parallel_plan.
        if first_adapter and self.parallel.is_distributed() and not self._parallel_applied:
            self._apply_parallel_plan()
            self._parallel_applied = True
            if self._slot is not None:
                self._slot.parallel_applied = True

    def _apply_parallel_plan(self) -> None:
        if self._peft is None:
            return
        extension = self._distributed_runtime.extension_handle
        apply_plan = getattr(extension, "apply_parallel_plan", None)
        if callable(apply_plan):
            apply_plan(self._peft, self._distributed_runtime, self.parallel)
            return
        if self._mesh is None:
            return
        from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

        try:
            inner = self._peft.base_model.model.model.layers
        except AttributeError:
            return

        dp_mesh = None
        if self.parallel.dp_degree > 1 and "dp" in self._mesh.mesh_dim_names:
            dp_mesh = self._mesh["dp"]
        if dp_mesh is None:
            return

        kwargs: dict[str, Any] = {"mesh": dp_mesh}
        if self.parallel.offload.cpu_offload_params:
            kwargs["offload_policy"] = CPUOffloadPolicy()

        for block in inner:
            fully_shard(block, **kwargs)

    async def _save_session_to_store(
        self,
        session_id: str,
        runtime: _SessionRuntime,
        *,
        sync_remote: bool = False,
    ) -> None:
        """Persist session state to worker-local disk, then mirror to remote.

        Writes all blobs (LoRA snapshot, ``grad_accum.pt``,
        ``optimizer_state.pt``, ``session_meta.json``) to the local
        filesystem in parallel. By default, the remote mirror runs
        asynchronously — the caller's ack returns as soon as local
        disk is consistent, without waiting for the remote round trip.

        When ``sync_remote=True`` (``init_session``, ``save_weights``,
        eviction, shutdown), this waits for the remote mirror to
        complete before returning. That's the durability boundary
        where another worker could pick up the session — the remote
        store must be authoritative before we release.

        ``session_meta.json`` is written last so that on partial-write
        recovery, a missing/old meta triggers a remote-fallback load
        instead of returning an inconsistent snapshot+delta pairing.
        """
        from hatchery.core.lora_state import LoraStateConfig

        prefix = f"sessions/{session_id}/live_state"

        t_prep = time.time()
        if runtime.training_mode == "full_param":
            # Capture from the live model if this session is the one
            # currently materialized on the slot, else fall back to the
            # CPU-resident snapshot we stashed at session-deactivation.
            slot_active = self._slot is not None and (self._slot.active_session_id == session_id)
            if slot_active:
                lora_state = self._capture_live_base_weights()
                # Refresh the stash so the in-memory copy stays in sync
                # with what we're about to write to disk.
                self._fp_base_state[session_id] = {k: v.clone() for k, v in lora_state.items()}
            else:
                stashed = self._fp_base_state.get(session_id)
                lora_state = {k: v.clone() for k, v in stashed.items()} if stashed else {}
        else:
            adapter = self._adapter_name(session_id)
            lora_state = get_peft_model_state_dict(self._peft, adapter_name=adapter)
            lora_state = {k: v.detach().cpu() for k, v in lora_state.items()}
        prep_ms = (time.time() - t_prep) * 1000

        # ── Phase 1: write to local disk (hot path; μs–ms) ──────────
        # ``lora_state.save`` encapsulates bf16 compression / delta
        # logic — we pass it the local store so the exact same blob
        # layout goes to local and later gets byte-copied to remote.
        async def _save_lora_local() -> tuple[Any, dict]:
            return await self.config.lora_state.save(
                self._state.local,
                prefix,
                lora_state,
                snapshot_cache=runtime.snapshot_cache,
                snapshot_version=runtime.snapshot_version,
                delta_count=runtime.delta_count,
                cfg=LoraStateConfig(),
            )

        async def _save_grad_local() -> int:
            if not runtime.grad_accum:
                return 0
            # Post-optim_step, ``_handle_optim_step`` resets grad_accum
            # to a dict of fresh ``zeros_like`` tensors that carry no
            # information but would otherwise cost MBs on the wire for
            # every step. ``accum_steps == 0`` is the boundary — before
            # any fwd_bwd of a new accumulation cycle we serialize an
            # empty dict, which the load path handles identically to a
            # missing file. The empty-dict write (rather than a delete)
            # is deliberate: the remote mirror only propagates ``put``s,
            # so writing ensures a stale non-zero blob from an earlier
            # mid-accumulation save gets overwritten remotely.
            to_save: dict[str, torch.Tensor] = (
                {} if runtime.meta.get("accum_steps", 0) == 0 else runtime.grad_accum
            )
            buf = io.BytesIO()
            torch.save(to_save, buf)
            data = buf.getvalue()
            await self._state.local.put(f"{prefix}/grad_accum.pt", data)
            return len(data)

        async def _save_opt_local() -> tuple[Any, Optional[dict], int]:
            from hatchery.core.optimizer_state import OptimizerStateConfig

            result, new_cache = await self.config.optimizer_state.save(
                self._state.local,
                prefix,
                runtime.optimizer_state,
                snapshot_cache=runtime.optim_snapshot_cache,
                snapshot_version=runtime.optim_snapshot_version,
                delta_count=runtime.optim_delta_count,
                cfg=OptimizerStateConfig(),
            )
            return result, new_cache, result.snapshot_bytes + result.delta_bytes

        t_bulk = time.time()
        lora_result, grad_bytes, opt_triple = await asyncio.gather(
            _save_lora_local(), _save_grad_local(), _save_opt_local()
        )
        bulk_ms = (time.time() - t_bulk) * 1000

        save_result, new_snapshot_cache = lora_result
        runtime.snapshot_cache = new_snapshot_cache
        runtime.snapshot_version = save_result.snapshot_version
        runtime.delta_count = save_result.delta_count

        opt_result, new_optim_cache, opt_bytes = opt_triple
        runtime.optim_snapshot_cache = new_optim_cache
        runtime.optim_snapshot_version = opt_result.snapshot_version
        runtime.optim_delta_count = opt_result.delta_count

        meta = dict(runtime.meta)
        meta["training_mode"] = runtime.training_mode
        if runtime.lora_config is not None:
            meta["lora_config"] = {
                "r": runtime.lora_config.r,
                "lora_alpha": runtime.lora_config.lora_alpha,
                "target_modules": list(runtime.lora_config.target_modules),
                "lora_dropout": runtime.lora_config.lora_dropout,
                "use_rslora": getattr(runtime.lora_config, "use_rslora", False),
                "init_lora_weights": getattr(runtime.lora_config, "init_lora_weights", True),
            }
        meta["snapshot_version"] = runtime.snapshot_version
        meta["delta_count"] = runtime.delta_count
        meta["optim_snapshot_version"] = runtime.optim_snapshot_version
        meta["optim_delta_count"] = runtime.optim_delta_count
        t_meta = time.time()
        await self._state.local.put(
            f"{prefix}/session_meta.json",
            json.dumps(meta).encode("utf-8"),
        )
        meta_ms = (time.time() - t_meta) * 1000

        # Remember that we just wrote this session so the next op on
        # the same worker can skip the _is_cache_stale GET — the
        # object store is authoritative-for-others, but *we* just
        # wrote the latest meta, so there's nothing newer out there.
        if self._persists_external_state():
            self._self_saved_versions[session_id] = (
                runtime.meta.get("total_steps", 0),
                runtime.meta.get("accum_steps", 0),
            )

        # ── Phase 2: schedule (or await) remote mirror ──────────────
        if self._persists_external_state():
            self._state.mark_dirty(session_id)
        remote_ms = 0.0
        if sync_remote and self._persists_external_state():
            t_remote = time.time()
            await self._state.flush(session_id)
            remote_ms = (time.time() - t_remote) * 1000

        logger.info(
            "worker.save_session.phases",
            session_id=session_id,
            sync_remote=sync_remote,
            prep_ms=round(prep_ms, 2),
            bulk_local_ms=round(bulk_ms, 2),
            meta_ms=round(meta_ms, 2),
            remote_ms=round(remote_ms, 2),
            lora_snapshot_bytes=save_result.snapshot_bytes,
            grad_bytes=grad_bytes,
            opt_bytes=opt_bytes,
        )

    async def _save_fp_init_marker(
        self,
        session_id: str,
        runtime: _SessionRuntime,
    ) -> None:
        """Write just ``session_meta.json`` for a freshly-init'd FP session.

        Skips the multi-GB ``lora_weights.pt`` upload that
        ``_save_session_to_store`` would perform. The pristine base is
        identical to the HF source and can be re-materialized on load
        when the snapshot file is absent + ``total_steps == 0``.
        """
        prefix = f"sessions/{session_id}/live_state"
        meta = dict(runtime.meta)
        meta["training_mode"] = runtime.training_mode
        meta["snapshot_version"] = 0
        meta["delta_count"] = 0
        meta["optim_snapshot_version"] = 0
        meta["optim_delta_count"] = 0
        meta["fp_pristine_init"] = True  # load path uses this to reconstruct from HF
        payload = json.dumps(meta).encode("utf-8")
        await self._state.local.put(f"{prefix}/session_meta.json", payload)
        if self._persists_external_state():
            self._state.mark_dirty(session_id)
            await self._state.flush(session_id)

    # ── Operation handlers ────────────────────────────────────

    async def _handle_init_session(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        if payload.get("rank") is None:
            # Full-parameter session: no adapter, no lora_config. The
            # base weights we'll train against are this slot's pristine
            # snapshot, captured lazily by ``_attach_full_param_session``.
            self._enforce_fft_capacity(session_id)
            self._attach_full_param_session(session_id)
            optimizer_kind = self._select_fp_optimizer_kind()
            runtime = _SessionRuntime(
                session_id=session_id,
                lora_config=None,
                training_mode="full_param",
                grad_accum={},
                optimizer_state=None,
                meta={
                    "accum_steps": 0,
                    "total_steps": 0,
                    "training_mode": "full_param",
                    "optimizer_kind": optimizer_kind,
                },
            )
            self._cache.put(session_id, runtime)
            # FP init snapshot is identical to the HF base — uploading
            # GBs of pristine weights here would block the dequeue loop
            # past the gateway's create_model timeout. The first
            # optim_step will mirror the real (diverged) state.
            await self._save_fp_init_marker(session_id, runtime)
            return {"status": "initialized"}, {}

        lora_kwargs = {
            "r": payload["rank"],
            "lora_alpha": payload["lora_alpha"],
            "target_modules": payload["target_modules"],
            "lora_dropout": payload.get("lora_dropout", 0.0),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if payload.get("use_rslora"):
            lora_kwargs["use_rslora"] = True
        init_weights = payload.get("init_lora_weights", "default")
        if init_weights != "default":
            lora_kwargs["init_lora_weights"] = init_weights
        lora_config = LoraConfig(**lora_kwargs)
        self._attach_adapter(session_id, lora_config, state_dict={})
        runtime = _SessionRuntime(
            session_id=session_id,
            lora_config=lora_config,
            training_mode="lora",
            grad_accum={},
            optimizer_state=None,
            meta={"accum_steps": 0, "total_steps": 0},
        )
        self._cache.put(session_id, runtime)
        # init_session is the handoff boundary for first-time reads —
        # the remote store must hold this session's state before we
        # ack so the gateway's eligibility checks and any future
        # cross-worker load can find it.
        await self._save_session_to_store(session_id, runtime, sync_remote=True)
        return {"status": "initialized"}, {}

    def _allocate_batch(self, data_items: list[dict]) -> list[dict]:
        """Return the local rank's slice of a DP batch.

        Single-rank workers passthrough. Multi-rank workers dispatch
        to :func:`prepare_batch_for_dp` with the strategy configured
        on ``self.parallel``.
        """
        if not self._is_distributed or self._distributed_runtime.dp_world_size <= 1:
            return list(data_items)

        try:
            strategy = BatchStrategy(self.parallel.batch_strategy)
        except ValueError:
            strategy = BatchStrategy.AUTO

        allocation = prepare_batch_for_dp(
            data_items,
            dp_degree=self._distributed_runtime.dp_world_size,
            rank=self._distributed_runtime.dp_rank,
            strategy=strategy,
        )
        # Emit a metric so operators can see how much compute is being
        # replicated vs split. This is cheap and catches misconfig
        # like dp_degree=8 always replicating because the user's
        # batches are always 1.
        self.config.metrics.set_gauge(
            "batch_wasted_compute_pct",
            allocation.wasted_compute_pct * 100.0,
            {"strategy": allocation.strategy.value},
        )
        return allocation.data

    async def _handle_forward_backward(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        from hatchery.core.fused_losses import (
            fused_cross_entropy_forward_backward,
            is_fused_eligible,
        )

        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)
        model.train()

        loss_fn = payload.get("loss_fn", "cross_entropy")
        loss_fn_config = payload.get("loss_fn_config")
        data_items = payload["data"]
        # When the caller needs per-datum logprobs back (e.g. tinker SDK's
        # ForwardBackwardOutput schema), skip the fused CE path — it never
        # materializes logits, and the scalar-loss-only shape it returns
        # is incompatible with the SDK's LossFnOutput contract.
        return_per_datum_logprobs = bool(payload.get("return_per_datum_logprobs", False))
        if not data_items:
            raise ValueError("forward_backward requires non-empty 'data'")

        # Data-parallel batch allocation (see hatchery.core.batching).
        data_items = self._allocate_batch(data_items)

        # Packing produces a list of sub-batches. When packing is off
        # or ineligible, the helper returns a single padded batch so
        # this path remains identical to the pre-packing behavior.
        # Disallow packing when the caller wants per-datum logprobs —
        # those are a 1:1 mapping onto input rows and don't cleanly
        # survive a re-pack into sub-batches.
        sub_batches = self._collate_batches(data_items, allow_packing=not return_per_datum_logprobs)

        # Zero trainable grads once, before any sub-batch backward.
        for p in model.parameters():
            if p.requires_grad:
                p.grad = None

        per_item_logprobs: Optional[list[list[float]]] = None
        # Per-call diagnostic metrics from richer losses (e.g. orpo).
        # Each entry is the metrics dict from one sub-batch; we reduce
        # by taking the mean across sub-batches when surfacing them.
        sub_loss_metrics: list[dict] = []

        # Fused CE only fits the LoRA path today — the kernel takes a
        # PEFT model and reads adapter-aware metadata. Force the
        # generic path for full-param so the same forward shape works.
        # Packing is orthogonal: ``fused_cross_entropy_forward_backward``
        # accepts ``position_ids`` and plumbs it through the PEFT
        # forward; the causal-shift CE is unchanged because boundary
        # labels are already -100 from pack_sequences.
        is_fp = runtime.training_mode == "full_param"
        probe = sub_batches[0]
        probe_labels = probe["labels"].to(self.device)
        probe_weights = (
            probe["weights"].to(self.device) if probe.get("weights") is not None else None
        )
        fused_eligible = (
            not is_fp
            and not return_per_datum_logprobs
            and is_fused_eligible(
                loss_fn=loss_fn,
                labels=probe_labels,
                weights=probe_weights,
                peft_model=self._peft,
            )
        )

        # Sum non-ignored target tokens across sub-batches so each
        # sub-backward scales its mean-loss by (sub_tokens / total) —
        # reconstructs the single-forward gradient from N sub-calls.
        sub_token_counts = [int((b["labels"] != -100).sum().cpu()) for b in sub_batches]
        total_tokens = max(sum(sub_token_counts), 1)

        if fused_eligible:
            weighted_loss_sum = 0.0
            for batch, num in zip(sub_batches, sub_token_counts, strict=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = (
                    batch["attention_mask"].to(self.device)
                    if batch["attention_mask"] is not None
                    else None
                )
                labels = batch["labels"].to(self.device)
                position_ids = (
                    batch["position_ids"].to(self.device)
                    if batch.get("position_ids") is not None
                    else None
                )
                # ``loss_scale`` = num / total_tokens preserves the
                # single-forward gradient magnitude across N sub-backward
                # calls. For the common single-sub-batch path this is
                # 1.0 and the behavior matches the pre-packing code.
                loss_detached = fused_cross_entropy_forward_backward(
                    self._peft,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                    loss_scale=num / total_tokens,
                )
                weighted_loss_sum += float(loss_detached.cpu()) * num

            loss_val = weighted_loss_sum / total_tokens
        else:
            weighted_loss_sum = 0.0
            for batch, num in zip(sub_batches, sub_token_counts, strict=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = (
                    batch["attention_mask"].to(self.device)
                    if batch["attention_mask"] is not None
                    else None
                )
                labels = batch["labels"].to(self.device)
                position_ids = (
                    batch["position_ids"].to(self.device)
                    if batch.get("position_ids") is not None
                    else None
                )
                weights = (
                    batch["weights"].to(self.device) if batch.get("weights") is not None else None
                )
                old_logprobs = (
                    batch["old_logprobs"].to(self.device)
                    if batch.get("old_logprobs") is not None
                    else None
                )
                advantages = (
                    batch["advantages"].to(self.device)
                    if batch.get("advantages") is not None
                    else None
                )

                model_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "labels": None,
                    "use_cache": False,
                }
                if position_ids is not None:
                    # Packed path: HF flash-attn-2 derives cu_seqlens
                    # from position_ids resets; a rectangular mask
                    # would contradict it.
                    model_kwargs["position_ids"] = position_ids
                else:
                    model_kwargs["attention_mask"] = attention_mask
                if batch.get("pixel_values") is not None:
                    model_kwargs["pixel_values"] = batch["pixel_values"].to(self.device)
                if batch.get("image_grid_thw") is not None:
                    model_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(self.device)

                with self._exec_context(runtime):
                    outputs = model(**model_kwargs)
                logits = outputs.logits
                # For loss functions that consume attention_mask for
                # padding weights, synthesize an all-ones mask on the
                # packed path — every packed token is a real token.
                loss_mask = (
                    attention_mask
                    if attention_mask is not None
                    else torch.ones_like(input_ids, dtype=torch.long)
                )
                loss, loss_metrics = self._compute_loss(
                    logits,
                    labels,
                    loss_mask,
                    loss_fn,
                    weights=weights,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    loss_fn_config=loss_fn_config,
                )
                if loss_metrics is not None:
                    sub_loss_metrics.append(loss_metrics)
                # Per-datum logprobs path is only reachable on the
                # single-sub-batch padded route (guard above disables
                # packing when return_per_datum_logprobs is set).
                if return_per_datum_logprobs:
                    from hatchery.core.losses import compute_target_logprobs

                    # Per-datum logprobs only run on the padded route,
                    # where attention_mask is always supplied.
                    assert attention_mask is not None
                    with torch.no_grad():
                        if labels.dim() == 2:
                            safe_labels = labels.clone()
                            safe_labels[safe_labels == -100] = 0
                            lp = compute_target_logprobs(logits, safe_labels)
                            lp = lp.masked_fill(labels == -100, 0.0)
                        else:
                            lp = torch.zeros(
                                input_ids.size(0), input_ids.size(1), device=self.device
                            )
                        lp_cpu = lp.detach().float().cpu()
                        per_item_logprobs = []
                        assert attention_mask is not None
                        for i in range(lp_cpu.size(0)):
                            orig_len = int(attention_mask[i].sum().cpu())
                            per_item_logprobs.append(lp_cpu[i, :orig_len].tolist())

                scale = num / total_tokens
                (loss * scale).backward()
                weighted_loss_sum += float(loss.detach().cpu()) * num

            loss_val = weighted_loss_sum / total_tokens

        # num_tokens counts real target positions across all sub-batches.
        num_tokens = total_tokens

        # Accumulate grads on CPU.
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            g = param.grad.detach().float().cpu()
            if name in runtime.grad_accum:
                runtime.grad_accum[name] = runtime.grad_accum[name] + g
            else:
                runtime.grad_accum[name] = g

        runtime.meta["accum_steps"] = runtime.meta.get("accum_steps", 0) + 1
        # Track totals for grad_accumulation_normalization modes.
        runtime.meta["accum_loss_tokens"] = runtime.meta.get("accum_loss_tokens", 0) + num_tokens
        runtime.meta["accum_sequences"] = runtime.meta.get("accum_sequences", 0) + int(
            input_ids.size(0)
        )
        await self._save_session_to_store(session_id, runtime)

        if self._persists_external_state():
            with contextlib.suppress(KeyError):
                await self.config.metadata.update_session(
                    session_id, accum_steps=runtime.meta["accum_steps"]
                )

        cost_dims = self._build_cost_dimensions(
            runtime=runtime,
            batch_size=input_ids.size(0),
            max_seq_len=input_ids.size(1),
            loss_fn=loss_fn,
            fused_path=fused_eligible,
        )

        result: dict[str, Any] = {
            "loss": loss_val,
            "num_tokens": num_tokens,
            "accum_steps": runtime.meta["accum_steps"],
        }
        if per_item_logprobs is not None:
            result["per_datum_logprobs"] = per_item_logprobs

        extra_metrics: dict[str, Any] = {
            "tokens": num_tokens,
            "cost_dimensions": cost_dims,
        }
        # Mean-reduce loss-fn diagnostics across sub-batches so callers
        # see a single scalar per metric name (matches the SDK's
        # ``name:reduction`` envelope conventions; see tinker_compat
        # _wrap_future_result). Only ever populated by losses that opt
        # in (currently orpo); the bare-scalar return path leaves
        # ``sub_loss_metrics`` empty.
        if sub_loss_metrics:
            agg = _mean_reduce_loss_metrics(sub_loss_metrics)
            extra_metrics.update(agg)
        return (result, extra_metrics)

    async def _handle_forward_only(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        """No-grad forward pass with a caller-supplied loss function.

        Mirrors :meth:`_handle_forward_backward` minus anything that
        would mutate training state: no ``.backward()``, no CPU grad
        accumulation, no ``accum_steps`` bump, no session state
        persisted. Token counts are still reported so billing /
        usage tracking charges a forward pass uniformly with
        forward_backward (matches Tinker's wire semantics).
        """
        from hatchery.core.losses import DECLARED_LOSS_FNS, SUPPORTED_LOSS_FNS

        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)
        # eval() so LoRA/attention dropout is off — forward_only is
        # typically used for eval/held-out loss where determinism matters.
        model.eval()

        loss_fn = payload.get("loss_fn", "cross_entropy")
        loss_fn_config = payload.get("loss_fn_config")
        data_items = payload["data"]
        if not data_items:
            raise ValueError("forward_only requires non-empty 'data'")
        if loss_fn not in SUPPORTED_LOSS_FNS:
            if loss_fn in DECLARED_LOSS_FNS:
                raise ValueError(f"loss_fn {loss_fn!r} is declared but not implemented server-side")
            raise ValueError(f"unknown loss_fn {loss_fn!r}")

        data_items = self._allocate_batch(data_items)
        sub_batches = self._collate_batches(data_items)
        sub_token_counts = [int((b["labels"] != -100).sum().cpu()) for b in sub_batches]
        total_tokens = max(sum(sub_token_counts), 1)

        weighted_loss_sum = 0.0
        last_input_ids: Optional[torch.Tensor] = None
        sub_loss_metrics: list[dict] = []
        for batch, num in zip(sub_batches, sub_token_counts, strict=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = (
                batch["attention_mask"].to(self.device)
                if batch["attention_mask"] is not None
                else None
            )
            labels = batch["labels"].to(self.device)
            position_ids = (
                batch["position_ids"].to(self.device)
                if batch.get("position_ids") is not None
                else None
            )
            weights = batch["weights"].to(self.device) if batch.get("weights") is not None else None
            old_logprobs = (
                batch["old_logprobs"].to(self.device)
                if batch.get("old_logprobs") is not None
                else None
            )
            advantages = (
                batch["advantages"].to(self.device) if batch.get("advantages") is not None else None
            )

            model_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "labels": None,
                "use_cache": False,
            }
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask
            if batch.get("pixel_values") is not None:
                model_kwargs["pixel_values"] = batch["pixel_values"].to(self.device)
            if batch.get("image_grid_thw") is not None:
                model_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(self.device)

            with torch.no_grad(), self._exec_context(runtime):
                outputs = model(**model_kwargs)
                logits = outputs.logits
                loss_mask = (
                    attention_mask
                    if attention_mask is not None
                    else torch.ones_like(input_ids, dtype=torch.long)
                )
                loss, loss_metrics = self._compute_loss(
                    logits,
                    labels,
                    loss_mask,
                    loss_fn,
                    weights=weights,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    loss_fn_config=loss_fn_config,
                )
                if loss_metrics is not None:
                    sub_loss_metrics.append(loss_metrics)

            weighted_loss_sum += float(loss.detach().cpu()) * num
            last_input_ids = input_ids

        loss_val = weighted_loss_sum / total_tokens
        num_tokens = total_tokens

        cost_dims = self._build_cost_dimensions(
            runtime=runtime,
            batch_size=last_input_ids.size(0) if last_input_ids is not None else 0,
            max_seq_len=last_input_ids.size(1) if last_input_ids is not None else 0,
            loss_fn=loss_fn,
            fused_path=False,
        )

        extra_metrics: dict[str, Any] = {
            "tokens": num_tokens,
            "cost_dimensions": cost_dims,
        }
        if sub_loss_metrics:
            extra_metrics.update(_mean_reduce_loss_metrics(sub_loss_metrics))
        return (
            {"loss": loss_val, "num_tokens": num_tokens},
            extra_metrics,
        )

    async def _handle_forward_custom_step1(
        self, session_id: str, payload: dict
    ) -> tuple[dict, dict]:
        """First leg of ``forward_backward_custom``.

        Runs a forward pass and returns per-position log π(target_t)
        so the client can compute a custom loss and send back
        ``grad_logprobs`` on step 2. No backward happens here.

        We stash a cache keyed by a client-generated ``custom_id`` so
        step 2 can re-run the forward with the same activations. To
        keep worker memory bounded we stash only the collated inputs
        (not the activations); step 2 re-runs forward under autograd.
        """
        from hatchery.core.losses import compute_target_logprobs

        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)
        model.train()

        data_items = payload["data"]
        custom_id = payload.get("custom_id")
        if not data_items:
            raise ValueError("forward_custom_step1 requires non-empty 'data'")
        if not custom_id:
            raise ValueError("forward_custom_step1 requires 'custom_id'")

        data_items = self._allocate_batch(data_items)
        batch = self._collate(data_items)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.no_grad(), self._exec_context(runtime):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
            )
            logprobs = compute_target_logprobs(outputs.logits, labels)

        # Cache the collated batch for step2, along with per-item
        # original lengths so step2 can reconstruct the same padding.
        item_lengths = [len(item["input_ids"]) for item in data_items]
        runtime.custom_cache[custom_id] = {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "labels": labels.detach().cpu(),
            "item_lengths": item_lengths,
        }

        # Return per-item ragged logprob lists, trimmed to the
        # original (unpadded) length of each Datum.  ``logprobs`` is
        # now [B, T] (Tinker convention: position 0 = 0.0, positions
        # 1..T-1 carry the causal logprobs). Each item gets exactly
        # ``orig_len`` entries — a 1:1 mapping with input token
        # positions.
        per_item_logprobs = []
        per_item_shapes = []
        logprobs_cpu = logprobs.detach().cpu()
        for i, orig_len in enumerate(item_lengths):
            row = logprobs_cpu[i, :orig_len]
            per_item_logprobs.append(row.tolist())
            per_item_shapes.append(list(row.shape))

        num_tokens = int((labels != -100).sum().cpu())
        return (
            {
                "logprobs": per_item_logprobs,
                "shapes": per_item_shapes,
                "num_tokens": num_tokens,
            },
            {"tokens": num_tokens},
        )

    async def _handle_forward_custom_step2(
        self, session_id: str, payload: dict
    ) -> tuple[dict, dict]:
        """Second leg of ``forward_backward_custom``.

        Re-runs the forward under autograd, builds the surrogate loss
        ``Σ grad_logprobs.detach() * logprobs`` so the chain rule
        delivers the same parameter gradient the user's custom loss
        would have produced, and accumulates the LoRA grads.
        """
        from hatchery.core.losses import (
            compute_target_logprobs,
            surrogate_loss_from_grad,
        )

        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)
        model.train()

        custom_id = payload.get("custom_id")
        grad_list = payload.get("grad_logprobs")
        if not custom_id:
            raise ValueError("forward_custom_step2 requires 'custom_id'")
        if grad_list is None:
            raise ValueError("forward_custom_step2 requires 'grad_logprobs'")

        cache = runtime.custom_cache
        cached = cache.get(custom_id)
        if cached is None:
            raise ValueError(
                f"no cached forward for custom_id={custom_id!r}; "
                "did step1 run on this worker and session?"
            )

        input_ids = cached["input_ids"].to(self.device)
        attention_mask = cached["attention_mask"].to(self.device)
        labels = cached["labels"].to(self.device)

        # ``grad_list`` is ragged (per-item T-length lists) if step1
        # returned T-length logprobs, or a flat [B, T] rectangle for
        # callers that already padded. We pad back to the collated
        # [B, T] shape (matching compute_target_logprobs output).
        B = input_ids.size(0)
        T = input_ids.size(1)
        if isinstance(grad_list[0], list):
            grad_logprobs = torch.zeros(B, T, dtype=torch.float32, device=self.device)
            for i, row in enumerate(grad_list):
                row_len = len(row)
                grad_logprobs[i, :row_len] = torch.tensor(row, dtype=torch.float32)
        else:
            grad_logprobs = torch.tensor(grad_list, dtype=torch.float32, device=self.device)

        for p in model.parameters():
            if p.requires_grad:
                p.grad = None

        with self._exec_context(runtime):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
            )
        logprobs = compute_target_logprobs(outputs.logits, labels)
        surrogate = surrogate_loss_from_grad(logprobs, grad_logprobs)
        surrogate.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            g = param.grad.detach().float().cpu()
            if name in runtime.grad_accum:
                runtime.grad_accum[name] = runtime.grad_accum[name] + g
            else:
                runtime.grad_accum[name] = g

        runtime.meta["accum_steps"] = runtime.meta.get("accum_steps", 0) + 1
        # Evict the cache — step2 is single-use.
        with contextlib.suppress(KeyError):
            del cache[custom_id]
        await self._save_session_to_store(session_id, runtime)

        if self._persists_external_state():
            with contextlib.suppress(KeyError):
                await self.config.metadata.update_session(
                    session_id, accum_steps=runtime.meta["accum_steps"]
                )

        num_tokens = int((labels != -100).sum().cpu())
        return (
            {
                "surrogate": float(surrogate.detach().cpu()),
                "num_tokens": num_tokens,
                "accum_steps": runtime.meta["accum_steps"],
            },
            {"tokens": num_tokens},
        )

    @staticmethod
    def _grad_norm_divisor(mode: str, runtime: _SessionRuntime) -> float:
        """Compute the gradient normalization divisor.

        Modes:
        - ``num_loss_tokens`` — total non-zero-weight tokens across
          accumulated forward_backward calls.
        - ``num_sequences`` — total sequence (example) count across
          accumulated forward_backward calls.
        """
        if mode == "num_loss_tokens":
            return float(max(runtime.meta.get("accum_loss_tokens", 1), 1))
        if mode == "num_sequences":
            return float(max(runtime.meta.get("accum_sequences", 1), 1))
        raise ValueError(f"Unknown grad_accumulation_normalization mode: {mode!r}")

    async def _handle_optim_step(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)

        lr = float(payload.get("learning_rate", 1e-4))
        beta1 = float(payload.get("beta1", 0.9))
        beta2 = float(payload.get("beta2", 0.999))
        eps = float(payload.get("eps", 1e-8))
        weight_decay = float(payload.get("weight_decay", 0.01))
        grad_clip_norm = float(payload.get("grad_clip_norm", 0.0))

        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        if not runtime.grad_accum:
            raise RuntimeError("optim_step called with no accumulated grads")

        # LoRA → fp32 fused AdamW (state is tiny). FP → 8-bit AdamW
        # when fp32 state would exceed the VRAM budget. The kind was
        # decided at init_session and stored in runtime.meta.
        kind = runtime.meta.get("optimizer_kind", "adamw")
        use_fused = kind == "adamw" and self.device.startswith("cuda") and torch.cuda.is_available()
        optimizer = build_optimizer(
            [p for _, p in named_params],
            kind=kind,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            fused=use_fused,
        )
        if runtime.optimizer_state is not None:
            try:
                optimizer.load_state_dict(runtime.optimizer_state)
            except Exception:  # noqa: BLE001
                # Schema shift from a prior optimizer config — start fresh.
                logger.warning("optim.reset", session_id=session_id)

        # load_state_dict overrides the LR with the saved value.
        # Re-apply the client's requested LR so per-step scheduling works.
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["betas"] = (beta1, beta2)
            pg["eps"] = eps
            pg["weight_decay"] = weight_decay

        # Normalize accumulated gradients before applying.
        norm_mode = payload.get("grad_accumulation_normalization")
        if norm_mode is not None:
            divisor = self._grad_norm_divisor(norm_mode, runtime)
            if divisor > 1.0:
                for g in runtime.grad_accum.values():
                    g.div_(divisor)

        # Copy accumulated grads back onto params.
        for name, param in named_params:
            g = runtime.grad_accum.get(name)
            if g is None:
                param.grad = None
                continue
            param.grad = g.to(param.device, dtype=param.dtype)

        # Gradient clipping (if requested).
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in named_params], max_norm=grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Persist optimizer state (on CPU) and reset accumulators.
        runtime.optimizer_state = _move_optimizer_state_to_cpu(optimizer.state_dict())
        runtime.grad_accum = {
            name: torch.zeros_like(param, device="cpu", dtype=torch.float32)
            for name, param in named_params
        }
        runtime.meta["total_steps"] = runtime.meta.get("total_steps", 0) + 1
        runtime.meta["accum_steps"] = 0
        runtime.meta["accum_loss_tokens"] = 0
        runtime.meta["accum_sequences"] = 0

        await self._save_session_to_store(session_id, runtime)

        if self._persists_external_state():
            with contextlib.suppress(KeyError):
                await self.config.metadata.update_session(
                    session_id,
                    total_steps=runtime.meta["total_steps"],
                    accum_steps=0,
                )

        return (
            {
                "status": "ok",
                "step": runtime.meta["total_steps"],
                "learning_rate": lr,
            },
            {},
        )

    async def _handle_save_weights(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        """Checkpoint the current session state to the object store."""
        name = payload["name"]
        runtime = self._cache.get(session_id)
        if runtime is None:
            raise RuntimeError(f"session {session_id} not loaded on this worker")

        await self._save_session_to_store(session_id, runtime, sync_remote=True)
        if not self._persists_external_state():
            return {"path": f"tinker://{session_id}/checkpoints/{name}"}, {}

        src_prefix = f"{self.config.sessions_prefix}/{session_id}/live_state"
        dst_prefix = f"{self.config.sessions_prefix}/{session_id}/checkpoints/{name}"

        # Ensure live state is in the remote object store before
        # materializing the checkpoint. When a mirrored
        # store's flush handles this; in core mode (local-only store)
        # we copy each blob from local disk to config.objects explicitly.
        for blob_name in (
            "lora_weights.pt",
            "optimizer_state.pt",
            "grad_accum.pt",
            "session_meta.json",
        ):
            key = f"{src_prefix}/{blob_name}"
            if await self.config.objects.exists(key):
                continue
            local_bytes = await self._state.load_local(key)
            if local_bytes is not None:
                await self.config.objects.put(key, local_bytes)

        await self.config.lora_state.materialize(self.config.objects, src_prefix, dst_prefix)

        for extra in ("optimizer_state.pt", "grad_accum.pt", "session_meta.json"):
            try:
                blob = await self.config.objects.get(f"{src_prefix}/{extra}")
                await self.config.objects.put(f"{dst_prefix}/{extra}", blob)
            except (KeyError, Exception):  # noqa: BLE001
                pass

        return {"path": f"tinker://{session_id}/checkpoints/{name}"}, {}

    async def _handle_load_weights(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        """Resume a session from a previously saved checkpoint.

        Reads the LoRA state from ``checkpoint_prefix/lora_weights.pt``
        and optionally restores the optimizer state from
        ``checkpoint_prefix/optimizer_state.pt``. The session's
        grad_accum is reset (the saved grads belong to a different
        accumulation cycle).
        """
        ckpt_prefix = payload["checkpoint_prefix"]
        restore_optimizer = payload.get("restore_optimizer", False)

        lora_state, snap_cache, snap_ver, delta_count = await self.config.lora_state.load(
            self.config.objects, ckpt_prefix
        )

        runtime = await self._ensure_session_loaded(session_id)
        if runtime.training_mode == "full_param":
            # FP checkpoint = full base state. Update the per-session
            # CPU snapshot, then push it into the live model if this
            # session is currently materialized on the slot.
            self._fp_base_state[session_id] = {
                k: v.detach().cpu().clone() for k, v in lora_state.items()
            }
            if self._slot is not None and self._slot.active_session_id == session_id:
                self._load_base_sd_into_live(self._fp_base_state[session_id])
        else:
            adapter = self._adapter_name(session_id)
            assert self._peft is not None
            set_peft_model_state_dict(self._peft, lora_state, adapter_name=adapter)

        runtime.snapshot_cache = snap_cache
        runtime.snapshot_version = snap_ver
        runtime.delta_count = delta_count

        if restore_optimizer:
            # Restore optimizer state (goes through the configured
            # persister so delta-compressed checkpoints load cleanly).
            (
                opt_state,
                opt_cache,
                opt_ver,
                opt_dc,
            ) = await self.config.optimizer_state.load(self.config.objects, ckpt_prefix)
            runtime.optimizer_state = opt_state
            runtime.optim_snapshot_cache = opt_cache
            runtime.optim_snapshot_version = opt_ver
            runtime.optim_delta_count = opt_dc

            # Restore accumulated gradients (for mid-accumulation resume).
            try:
                grad_bytes = await self.config.objects.get(f"{ckpt_prefix}/grad_accum.pt")
                runtime.grad_accum = torch.load(
                    io.BytesIO(grad_bytes), map_location="cpu", weights_only=True
                )
            except KeyError:
                runtime.grad_accum.clear()
        else:
            runtime.grad_accum.clear()

        await self._save_session_to_store(session_id, runtime)

        return (
            {"path": ckpt_prefix, "type": "load_weights"},
            {},
        )

    async def _handle_sample(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)
        assert self.tokenizer is not None
        model.eval()

        prompt_tokens = payload["prompt_tokens"]
        input_ids = torch.tensor([prompt_tokens], device=self.device, dtype=torch.long)
        max_new_tokens = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 1.0))
        top_p = float(payload.get("top_p", 1.0))
        top_k_raw = payload.get("top_k", -1)
        top_k = int(top_k_raw) if top_k_raw is not None else -1
        n = int(payload.get("n", 1))
        seed = payload.get("seed")
        stop = payload.get("stop")
        include_prompt_logprobs = bool(payload.get("include_prompt_logprobs", False))
        topk_prompt_logprobs = int(payload.get("topk_prompt_logprobs", 0) or 0)

        do_sample = temperature > 0 and (temperature != 1.0 or top_p != 1.0 or n > 1)
        if temperature == 0.0:
            do_sample = False

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.pad_token_id,
            # Emit per-token scores so we can hand the rollout log-prob
            # back to the client — cookbook RL recipes read these as
            # ``old_logprobs`` for importance_sampling / PPO.
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k

        # ``stop`` may be a single string or list of strings (SDK SamplingParams).
        # transformers >=4.34 accepts ``stop_strings`` on generate when
        # ``tokenizer`` is also supplied.
        if stop is not None:
            if isinstance(stop, str):
                stop_strings: list[str] = [stop]
            else:
                stop_strings = [str(s) for s in stop if isinstance(s, (str, bytes))]
            if stop_strings:
                gen_kwargs["stop_strings"] = stop_strings
                gen_kwargs["tokenizer"] = self.tokenizer

        # Per-request deterministic seed. Applies to both CPU and CUDA RNGs
        # — cookbook callers use this for reproducible eval rollouts.
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad(), self._exec_context(runtime):
            gen_out = model.generate(input_ids=input_ids, **gen_kwargs)

        out = gen_out.sequences  # [n, prompt+gen]
        # ``scores`` is a tuple of length ``gen_len`` of [n, V] pre-softmax
        # logits (or processed logits, depending on config). We log_softmax
        # per-step and gather the chosen token's logprob.
        scores = gen_out.scores  # tuple of [n, V]

        # Strip prompt; return only generated ids.
        prompt_len = input_ids.shape[1]
        completions: list[list[int]] = []
        texts: list[str] = []
        stop_reasons: list[str] = []
        per_seq_logprobs: list[list[float]] = []
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        for i, seq in enumerate(out):
            gen_ids = seq[prompt_len:].tolist()
            # Truncate trailing pad tokens (generate pads shorter seqs
            # to batch max when num_return_sequences > 1).
            real_len = len(gen_ids)
            if pad_id is not None:
                while real_len > 0 and gen_ids[real_len - 1] == pad_id:
                    real_len -= 1
            gen_ids = gen_ids[:real_len]
            completions.append(gen_ids)
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
            # Stop reason: hit EOS before max_new_tokens → "stop", otherwise "length".
            hit_eos = (eos_id is not None and len(gen_ids) > 0 and gen_ids[-1] == eos_id) or len(
                gen_ids
            ) < max_new_tokens
            stop_reasons.append("stop" if hit_eos else "length")

            # Gather per-token logprobs for this sequence.
            lp_row: list[float] = []
            for t, token_id in enumerate(gen_ids):
                if t >= len(scores):
                    break
                step_logits = scores[t][i].float()
                step_logp = F.log_softmax(step_logits, dim=-1)
                lp_row.append(float(step_logp[int(token_id)].item()))
            per_seq_logprobs.append(lp_row)

        response: dict = {
            "sequences": completions,
            "texts": texts,
            "stop_reasons": stop_reasons,
            "sequence_logprobs": per_seq_logprobs,
        }

        # Per-prompt-token logprobs and top-K. These are expensive
        # (extra forward pass over the prompt, full log-softmax at
        # every position, topk over the vocab), so we only compute
        # them when the caller explicitly asked.
        if include_prompt_logprobs or topk_prompt_logprobs > 0:
            with torch.no_grad(), self._exec_context(runtime):
                prompt_out = model(input_ids=input_ids, use_cache=False)
            logits = prompt_out.logits.float()  # [1, T, V]
            # Shift: logits[:, :-1, :] predicts tokens[:, 1:]. Position
            # 0 has no "previous context" so we pad a None at the start
            # — matches Tinker's `list[Optional[float]]` shape.
            shifted_logits = logits[:, :-1, :]
            shifted_targets = input_ids[:, 1:]
            log_probs = F.log_softmax(shifted_logits, dim=-1)

            if include_prompt_logprobs:
                # One float per prompt token. First position is None.
                per_token = (
                    log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)[0].tolist()
                )
                response["prompt_logprobs"] = [None] + per_token

            if topk_prompt_logprobs > 0:
                k = min(topk_prompt_logprobs, log_probs.shape[-1])
                topk_vals, topk_idx = log_probs[0].topk(k, dim=-1)  # [T-1, k]
                topk_per_pos: list = [None]  # position 0 has no context
                for i in range(topk_vals.shape[0]):
                    pos_entries = [(int(topk_idx[i, j]), float(topk_vals[i, j])) for j in range(k)]
                    topk_per_pos.append(pos_entries)
                response["topk_prompt_logprobs"] = topk_per_pos

        return response, {"tokens": sum(len(c) for c in completions)}

    async def _handle_forward_logprobs(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        """Per-position logprobs at ``target_tokens`` — SDK ``/forward``.

        Used by the tinker SDK's ``forward_backward_custom`` flow (and
        any ``forward_async`` caller that passes ``target_tokens`` via
        ``loss_fn_inputs``). Mirrors ``forward_custom_step1`` but does
        not cache state — the SDK's flow re-sends the batch on
        ``/forward_backward`` with weights = -dC/dlogprobs so there's
        no reason to stash activations.

        Follows the Tinker convention: per-item logprobs have the same
        length as the item's input_ids; position 0 is 0.0.
        """
        from hatchery.core.losses import compute_target_logprobs

        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)
        model.eval()

        data_items = payload["data"]
        if not data_items:
            raise ValueError("forward_logprobs requires non-empty 'data'")

        data_items = self._allocate_batch(data_items)
        batch = self._collate(data_items)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.no_grad(), self._exec_context(runtime):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
            )
            logprobs = compute_target_logprobs(outputs.logits, labels)

        item_lengths = [len(item["input_ids"]) for item in data_items]
        per_item_logprobs: list[list[float]] = []
        logprobs_cpu = logprobs.detach().cpu()
        for i, orig_len in enumerate(item_lengths):
            row = logprobs_cpu[i, :orig_len]
            per_item_logprobs.append(row.tolist())

        num_tokens = int((labels != -100).sum().cpu())
        return (
            {"per_datum_logprobs": per_item_logprobs},
            {"tokens": num_tokens},
        )

    async def _handle_compute_logprobs(self, session_id: str, payload: dict) -> tuple[dict, dict]:
        runtime = await self._ensure_session_loaded(session_id)
        model = self._activate_session(session_id, runtime)
        model.eval()

        token_lists = payload["input_tokens"]
        results: list[list[float]] = []
        total_tokens = 0
        for tokens in token_lists:
            input_ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
            with torch.no_grad(), self._exec_context(runtime):
                outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits.float()
            shifted_logits = logits[:, :-1, :]
            shifted_targets = input_ids[:, 1:]
            logprobs = F.log_softmax(shifted_logits, dim=-1)
            per_token = logprobs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
            results.append(per_token[0].cpu().tolist())
            total_tokens += per_token.numel()

        return {"logprobs": results}, {"tokens": total_tokens}

    # ── Helpers ───────────────────────────────────────────────

    def _collate(self, data_items: list[dict]) -> dict[str, Optional[torch.Tensor]]:
        """Pad a batch of ``{input_ids, labels?, weights?, logprobs?, advantages?}``
        dicts into a dict of tensors.

        The returned dict always has ``input_ids``, ``attention_mask``,
        and ``labels``. It optionally has ``weights``, ``old_logprobs``,
        and ``advantages`` when at least one datum carries them — all
        padded to the same max sequence length with semantically-correct
        defaults (0.0 for weights / advantages, 0.0 for old logprobs).

        Supports two shapes of ``labels`` + ``weights``:

        - 1-D (standard SFT / RL): each is a list of length T_i. All
          standard losses use this shape.
        - 2-D (SDFT / top-K distillation): each is a list of lists with
          outer length T_i and inner length K. The caller must ensure K
          is constant across all items in the batch. The returned
          ``labels``/``weights`` tensor has shape ``[B, T, K]``.
        """
        assert self.tokenizer is not None, "tokenizer must be loaded before _collate"
        pad_id = self.tokenizer.pad_token_id

        # For VLM models: strip vision placeholder tokens from text-only
        # items BEFORE computing max_len, so padding is consistent.
        if self.is_vlm and self._vision_token_ids:
            sanitized: list[dict] = []
            for item in data_items:
                if not item.get("images"):
                    clean_ids = _strip_vision_tokens(
                        list(item["input_ids"]), self._vision_token_ids
                    )
                    new_item = dict(item, input_ids=clean_ids)
                    # If labels mirror input_ids (self-prediction), strip those too.
                    lbls = item.get("labels")
                    if (
                        lbls is not None
                        and isinstance(lbls, list)
                        and not isinstance(lbls[0], list)
                    ):
                        new_item["labels"] = _strip_vision_tokens(
                            [int(x) for x in lbls], self._vision_token_ids
                        )
                    sanitized.append(new_item)
                else:
                    sanitized.append(item)
            data_items = sanitized

        max_len = max(len(item["input_ids"]) for item in data_items)

        # Detect 2-D labels from the first item that has any.
        labels_2d_k: Optional[int] = None
        for item in data_items:
            lbls = item.get("labels")
            if isinstance(lbls, list) and lbls and isinstance(lbls[0], list):
                labels_2d_k = len(lbls[0])
                break

        has_weights = any("weights" in item for item in data_items)
        has_logprobs = any("logprobs" in item for item in data_items)
        has_advantages = any("advantages" in item for item in data_items)

        input_ids_rows: list[list[int]] = []
        attn_rows: list[list[int]] = []
        labels_rows: list = []
        weights_rows: list = []
        logprobs_rows: list = []
        advantages_rows: list = []

        for item in data_items:
            ids = list(item["input_ids"])
            pad_len = max_len - len(ids)
            input_ids_rows.append(ids + [pad_id] * pad_len)
            attn_rows.append([1] * len(ids) + [0] * pad_len)

            # ── labels ──
            lbls = item.get("labels", ids)
            if labels_2d_k is not None:
                # Normalize to 2-D: broadcast 1-D labels into [T, K].
                if isinstance(lbls[0], int):
                    lbls_2d = [[x] + [-100] * (labels_2d_k - 1) for x in lbls]
                else:
                    lbls_2d = [list(row) for row in lbls]
                    for row in lbls_2d:
                        if len(row) != labels_2d_k:
                            raise ValueError(
                                f"inconsistent inner label width: expected {labels_2d_k}, got {len(row)}"
                            )
                pad_block = [[-100] * labels_2d_k] * pad_len
                labels_rows.append(lbls_2d + pad_block)
            else:
                if len(lbls) != len(ids):
                    raise ValueError(f"labels length {len(lbls)} != input_ids length {len(ids)}")
                labels_rows.append(list(lbls) + [-100] * pad_len)

            # ── weights ──
            if has_weights:
                w = item.get("weights")
                if w is None:
                    if labels_2d_k is not None:
                        w = [[0.0] * labels_2d_k] * len(ids)
                    else:
                        w = [1.0] * len(ids)
                if labels_2d_k is not None:
                    if isinstance(w[0], (int, float)):
                        w_2d = [[float(x)] + [0.0] * (labels_2d_k - 1) for x in w]
                    else:
                        w_2d = [[float(v) for v in row] for row in w]
                    pad_block = [[0.0] * labels_2d_k] * pad_len
                    weights_rows.append(w_2d + pad_block)
                else:
                    weights_rows.append([float(x) for x in w] + [0.0] * pad_len)

            # ── old logprobs (RL) ──
            if has_logprobs:
                lp = item.get("logprobs")
                if lp is None:
                    lp = [0.0] * len(ids)
                logprobs_rows.append([float(x) for x in lp] + [0.0] * pad_len)

            # ── advantages (RL) ──
            if has_advantages:
                adv = item.get("advantages")
                if adv is None:
                    adv = [0.0] * len(ids)
                advantages_rows.append([float(x) for x in adv] + [0.0] * pad_len)

        out: dict[str, Optional[torch.Tensor]] = {
            "input_ids": torch.tensor(input_ids_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attn_rows, dtype=torch.long),
            "labels": torch.tensor(labels_rows, dtype=torch.long),
            "weights": None,
            "old_logprobs": None,
            "advantages": None,
            "pixel_values": None,
            "image_grid_thw": None,
        }
        if has_weights:
            out["weights"] = torch.tensor(weights_rows, dtype=torch.float32)
        if has_logprobs:
            out["old_logprobs"] = torch.tensor(logprobs_rows, dtype=torch.float32)
        if has_advantages:
            out["advantages"] = torch.tensor(advantages_rows, dtype=torch.float32)

        # Process images for VLM models.
        if self.is_vlm and self.processor is not None:
            all_images = []
            for item in data_items:
                images = item.get("images", [])
                if images:
                    all_images.extend(images)
            if all_images:
                out = self._process_vlm_images(out, all_images, data_items)

        return out

    def _collate_batches(
        self,
        data_items: list[dict],
        *,
        allow_packing: bool = True,
    ) -> list[dict[str, Optional[torch.Tensor]]]:
        """Return a list of sub-batches (one forward call each).

        Returns ``[self._collate(data_items)]`` (length 1) by default —
        the pre-packing behavior. When ``self.parallel.sequence_packing``
        is enabled AND the batch is SFT-shaped (no RL inputs, no VLM
        images, no 2-D labels) AND ``allow_packing`` is True, returns
        one dict per pack from first-fit-decreasing. Packed dicts have
        ``attention_mask=None`` + ``position_ids`` set so the flash-attn-2
        backend infers cu_seqlens from position resets.

        Callers that need a single padded [B, T] layout (fused-CE,
        per-datum logprob emission, custom-step caching) should pass
        ``allow_packing=False``.
        """
        if not (allow_packing and self.parallel.sequence_packing):
            return [self._collate(data_items)]

        # Packing is incompatible with RL inputs, VLM, and 2-D labels.
        has_rl_inputs = any(
            "weights" in it or "logprobs" in it or "advantages" in it for it in data_items
        )
        has_images = any(it.get("images") for it in data_items)
        has_2d_labels = any(
            isinstance(it.get("labels"), list)
            and it.get("labels")
            and isinstance(it["labels"][0], list)
            for it in data_items
        )
        if has_rl_inputs or has_images or has_2d_labels:
            return [self._collate(data_items)]

        from hatchery.core.packing import pack_sequences

        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        max_len = self.parallel.max_packed_len or max(
            sum(len(it["input_ids"]) for it in data_items), 1
        )
        packs = pack_sequences(data_items, pad_id=pad_id, max_packed_len=max_len)
        out: list[dict[str, Optional[torch.Tensor]]] = []
        for pack in packs:
            out.append(
                {
                    "input_ids": pack.input_ids,
                    "attention_mask": None,
                    "labels": pack.labels,
                    "position_ids": pack.position_ids,
                    "weights": None,
                    "old_logprobs": None,
                    "advantages": None,
                    "pixel_values": None,
                    "image_grid_thw": None,
                }
            )
        return out

    def _process_vlm_images(
        self,
        batch: dict[str, Optional[torch.Tensor]],
        image_bytes_list: list[bytes],
        data_items: list[dict],
    ) -> dict[str, Optional[torch.Tensor]]:
        """Process raw image bytes through the VLM processor.

        Converts base64-decoded image bytes to PIL Images, runs them
        through the processor to get pixel_values and any model-specific
        image metadata (e.g., Qwen-VL's image_grid_thw).
        """
        from PIL import Image

        pil_images = []
        for img_bytes in image_bytes_list:
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        # Use the processor to get pixel values.
        try:
            proc_out = self.processor(images=pil_images, return_tensors="pt")
            if "pixel_values" in proc_out:
                batch["pixel_values"] = proc_out["pixel_values"]
            if "image_grid_thw" in proc_out:
                batch["image_grid_thw"] = proc_out["image_grid_thw"]
        except Exception:  # noqa: BLE001
            logger.warning("vlm.image_processing_failed", num_images=len(pil_images))

        return batch

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_fn: str,
        *,
        weights: Optional[torch.Tensor] = None,
        old_logprobs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        loss_fn_config: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """Dispatch to :mod:`hatchery.core.losses`.

        Falls back to an attention-mask-derived weight when the caller
        didn't provide one; that preserves the pre-refactor behavior of
        ``F.cross_entropy(ignore_index=-100)`` for simple SFT paths.

        Returns ``(loss_tensor, extra_metrics)``. Most losses return a
        bare scalar from :func:`losses.compute`; richer losses (orpo)
        return ``(scalar, dict)`` so they can surface diagnostic
        metrics. ``extra_metrics`` is ``None`` for the bare-scalar path.
        """
        from hatchery.core.losses import LossInputs, compute

        if weights is None:
            # Derive a per-token weight from attention_mask so padding
            # never contributes to the loss.
            weights = attention_mask.to(torch.float32)
        inputs = LossInputs(
            logits=logits,
            target_tokens=labels,
            weights=weights,
            old_logprobs=old_logprobs,
            advantages=advantages,
            loss_fn_config=loss_fn_config,
        )
        result = compute(loss_fn, inputs)
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None


def _mean_reduce_loss_metrics(metrics_list: list[dict]) -> dict:
    """Mean-reduce per-sub-batch loss metrics into a single flat dict.

    Used by the worker to fold the per-sub-batch diagnostic dicts that
    a richer loss (e.g. orpo) returns into a single per-call set
    surfaced via :class:`JobResult.metrics`. Only numeric scalars are
    averaged; non-numeric entries (if any) are dropped.
    """
    if not metrics_list:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for m in metrics_list:
        for k, v in m.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def _move_optimizer_state_to_cpu(state: dict) -> dict:
    """Deep-copy optimizer state to CPU for safe serialization."""
    out: dict = {"state": {}, "param_groups": state.get("param_groups", [])}
    for k, v in state.get("state", {}).items():
        if isinstance(v, dict):
            out["state"][k] = {
                kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()
            }
        else:
            out["state"][k] = v
    return out


async def run_worker_from_env() -> None:
    """Launcher used by ``python -m hatchery.core.worker``.

    Honors the ``HATCHERY_CONFIG_FACTORY`` env var (``module:callable``) so
    extension packages can inject their own env-var-driven config
    builder. Falls back to core's in-memory / local backends.
    """
    factory_ref = os.environ.get("HATCHERY_CONFIG_FACTORY")
    if factory_ref:
        import importlib

        mod_path, _, attr = factory_ref.partition(":")
        try:
            factory = getattr(importlib.import_module(mod_path), attr)
            config = factory()
        except (ImportError, AttributeError) as exc:
            # Don't let a missing extra silently demote a production
            # worker to in-memory backends — that turns a config typo
            # or a forgotten pip dep into an
            # invisible "worker polls but never dequeues" failure.
            # Log the exception loudly, then fall back so dev boxes
            # without extension extras still run.
            logger.exception(
                "worker.boot.config_factory_failed",
                factory=factory_ref,
                error=str(exc),
            )
            from hatchery.core.config import build_core_config

            config = build_core_config()
    else:
        from hatchery.core.config import build_core_config

        config = build_core_config()
    logger.info(
        "worker.boot.config_ready",
        queue=type(config.queue).__name__,
        metadata=type(config.metadata).__name__,
        objects=type(config.objects).__name__,
        compute=type(config.compute).__name__,
    )
    await config.metadata.initialize()
    logger.info("worker.boot.metadata_initialized")
    await config.queue.initialize()
    logger.info("worker.boot.queue_initialized")

    base = os.environ.get("HATCHERY_BASE_MODEL", "Qwen/Qwen2-0.5B")
    device = os.environ.get("HATCHERY_WORKER_DEVICE", "cuda:0")
    worker = GPUWorker(
        worker_id=f"worker-{uuid.uuid4().hex[:8]}",
        base_model_name=base,
        config=config,
        device=device,
    )
    logger.info("worker.boot.gpuworker_ready", worker_id=worker.worker_id)
    try:
        await worker.run()
    finally:
        logger.info("worker.boot.loop_exited")
        await config.queue.close()
        await config.metadata.close()


def _main() -> None:
    """Entrypoint wrapper for ``python -m hatchery.core.worker``.

    Ensures two production-critical things that ``asyncio.run`` alone
    doesn't give us:

    1. A surviving traceback on stdout when the worker loop dies —
       container runtimes only capture stdout/stderr, and uncaught
       exceptions in async code can end up silently swallowed if
       nothing is configured to surface them. We log the exception
       via ``logger.exception`` *and* fall back to ``traceback.print_exc``
       so the failure is visible however logging is (or isn't) set up.

    2. A crash-loop brake. If ``asyncio.run(run_worker_from_env())``
       fails immediately (bad env var, unreachable queue, missing
       credential, etc.) the container runtime restarts it within
       seconds. That eats logs and burns GPU-hours. Sleeping
       briefly on failure lets you read the last traceback, and in
       a healthy deploy this path simply isn't taken.
    """
    import logging
    import sys
    import traceback

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(
        level=os.environ.get("HATCHERY_LOG_LEVEL") or os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    try:
        asyncio.run(run_worker_from_env())
    except Exception:
        logger.exception("worker.fatal")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Hold so the crash is visible in log viewers before the
        # container restarts. Tuned via HATCHERY_CRASH_HOLD_SECONDS (default
        # 60s); set to 0 to exit immediately.
        hold = int(os.environ.get("HATCHERY_CRASH_HOLD_SECONDS", "60"))
        if hold > 0:
            time.sleep(hold)
        raise


if __name__ == "__main__":
    _main()
