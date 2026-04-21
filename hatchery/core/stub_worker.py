# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Stub trainer and worker for local dev and testing.

Implements the trainer and worker protocols using plain Python dicts —
no torch, no transformers, no PEFT. The loss decays monotonically so
training loops can verify convergence without a GPU.

Used by ``python -m hatchery.core.local_dev`` and by the e2e test suite.
"""

from __future__ import annotations

import asyncio
import contextlib

from hatchery.core.protocols import QueuedJob
from hatchery.core.trainer import (
    ForwardBackwardResult,
    LogprobsResult,
    LoraSpec,
    OptimStepResult,
    SampleResult,
    TrainerState,
)


class StubTrainer:
    """Trainer-protocol-compatible stub with no GPU work.

    Decrements loss monotonically on each forward_backward so callers
    can assert "the loop actually did work."
    """

    base_model_name = "scripted"
    tokenizer = None

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}
        self._step_log: list[tuple[str, str]] = []

    def attach_session(self, session_id: str, spec: LoraSpec) -> None:
        self._state.setdefault(
            session_id,
            {"accum_steps": 0, "total_steps": 0, "loss": 2.0, "spec": spec},
        )

    def detach_session(self, session_id: str) -> None:
        self._state.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        return session_id in self._state

    def init_session_state(self, session_id: str, spec: LoraSpec) -> TrainerState:
        self.attach_session(session_id, spec)
        return TrainerState(
            lora_weights={},
            grad_accum={},
            meta={
                "accum_steps": 0,
                "total_steps": 0,
                "lora_config": {
                    "r": spec.rank,
                    "lora_alpha": spec.lora_alpha,
                    "target_modules": list(spec.target_modules),
                },
            },
        )

    def load_state(self, session_id: str, state: TrainerState) -> None:
        self._state[session_id] = dict(state.meta)
        self._state[session_id].setdefault("loss", 2.0)

    def extract_state(self, session_id: str) -> TrainerState:
        s = self._state[session_id]
        return TrainerState(
            lora_weights={},
            grad_accum={},
            meta={
                "accum_steps": s.get("accum_steps", 0),
                "total_steps": s.get("total_steps", 0),
                "loss": s.get("loss", 0.0),
            },
        )

    def forward_backward(
        self, session_id: str, data: list[dict], loss_fn: str
    ) -> ForwardBackwardResult:
        s = self._state[session_id]
        s["accum_steps"] = s.get("accum_steps", 0) + 1
        s["loss"] = s.get("loss", 2.0) * 0.9
        self._step_log.append((session_id, "forward_backward"))
        total_tokens = sum(len(d["input_ids"]) for d in data)
        return ForwardBackwardResult(
            loss=s["loss"], num_tokens=total_tokens, accum_steps=s["accum_steps"]
        )

    def optim_step(self, session_id: str, adam_params: dict) -> OptimStepResult:
        s = self._state[session_id]
        s["total_steps"] = s.get("total_steps", 0) + 1
        s["accum_steps"] = 0
        self._step_log.append((session_id, "optim_step"))
        return OptimStepResult(
            step=s["total_steps"],
            learning_rate=float(adam_params.get("learning_rate", 1e-4)),
        )

    def sample(self, session_id: str, prompt_tokens: list[int], params: dict) -> SampleResult:
        self._step_log.append((session_id, "sample"))
        fake = [prompt_tokens[-1] + 1] if prompt_tokens else [0]
        return SampleResult(
            sequences=[fake],
            texts=["synthetic"],
            total_tokens=len(fake),
        )

    def compute_logprobs(self, session_id: str, input_tokens: list[list[int]]) -> LogprobsResult:
        self._step_log.append((session_id, "compute_logprobs"))
        return LogprobsResult(
            logprobs=[[-0.1] * (len(t) - 1) for t in input_tokens],
            total_tokens=sum(len(t) - 1 for t in input_tokens),
        )


class StubWorker:
    """Queue-draining worker backed by StubTrainer."""

    def __init__(self, worker_id: str, config, trainer: StubTrainer) -> None:
        self.worker_id = worker_id
        self.config = config
        self.trainer = trainer
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        import msgpack

        from hatchery.core.protocols import JobResult, JobStatus

        while not self._stop.is_set():
            try:
                job = await self.config.queue.dequeue(
                    worker_id=self.worker_id,
                    model_filter=None,
                    visibility_timeout=60,
                )
            except asyncio.CancelledError:
                return
            if job is None:
                await asyncio.sleep(0.002)
                continue
            try:
                result = await self._execute(job)
                await self.config.queue.ack(
                    job.job_id,
                    JobResult(
                        job_id=job.job_id,
                        status=JobStatus.COMPLETED,
                        result=msgpack.packb(result, use_bin_type=True),
                        metrics={"duration_ms": 1.0, "tokens": 1},
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                await self.config.queue.nack(job.job_id, f"{type(exc).__name__}: {exc}")

    async def _execute(self, job: QueuedJob) -> dict:
        import msgpack

        payload = msgpack.unpackb(job.payload, raw=False) if job.payload else {}
        if job.operation == "init_session":
            spec = LoraSpec(
                rank=payload["rank"],
                lora_alpha=payload["lora_alpha"],
                target_modules=payload["target_modules"],
            )
            state = self.trainer.init_session_state(job.session_id, spec)
            await self._persist_state(job.session_id, state)
            return {"status": "initialized"}
        if job.operation == "forward_backward":
            await self._load_or_attach(job.session_id)
            r = self.trainer.forward_backward(
                job.session_id,
                payload["data"],
                payload.get("loss_fn", "cross_entropy"),
            )
            await self._persist_state(job.session_id, self.trainer.extract_state(job.session_id))
            return {"loss": r.loss, "num_tokens": r.num_tokens, "accum_steps": r.accum_steps}
        if job.operation == "optim_step":
            await self._load_or_attach(job.session_id)
            r = self.trainer.optim_step(job.session_id, payload)
            await self._persist_state(job.session_id, self.trainer.extract_state(job.session_id))
            return {"status": "ok", "step": r.step, "learning_rate": r.learning_rate}
        if job.operation == "save_weights":
            name = payload["name"]
            src = f"sessions/{job.session_id}/live_state/lora_weights.pt"
            dst = f"sessions/{job.session_id}/checkpoints/{name}/lora_weights.pt"
            try:
                blob = await self.config.objects.get(src)
                await self.config.objects.put(dst, blob)
            except KeyError:
                pass
            return {"path": f"tinker://{job.session_id}/checkpoints/{name}"}
        if job.operation == "sample":
            await self._load_or_attach(job.session_id)
            r = self.trainer.sample(job.session_id, payload["prompt_tokens"], payload)
            return {"sequences": r.sequences, "texts": r.texts}
        if job.operation == "compute_logprobs":
            await self._load_or_attach(job.session_id)
            r = self.trainer.compute_logprobs(job.session_id, payload["input_tokens"])
            return {"logprobs": r.logprobs}
        raise ValueError(f"unknown op {job.operation!r}")

    async def _persist_state(self, session_id: str, state: TrainerState) -> None:
        import json

        prefix = f"sessions/{session_id}/live_state"
        await self.config.objects.put(f"{prefix}/lora_weights.pt", b"scripted-weights")
        await self.config.objects.put(
            f"{prefix}/session_meta.json",
            json.dumps(state.meta).encode("utf-8"),
        )

    async def _load_or_attach(self, session_id: str) -> None:
        import json

        if self.trainer.has_session(session_id):
            return
        meta_key = f"sessions/{session_id}/live_state/session_meta.json"
        try:
            meta_bytes = await self.config.objects.get(meta_key)
            meta = json.loads(meta_bytes)
            cfg = meta.get("lora_config", {"r": 8, "lora_alpha": 16, "target_modules": []})
            self.trainer.attach_session(
                session_id,
                LoraSpec(
                    rank=cfg["r"],
                    lora_alpha=cfg["lora_alpha"],
                    target_modules=list(cfg["target_modules"]),
                ),
            )
        except Exception:  # noqa: BLE001
            self.trainer.attach_session(
                session_id,
                LoraSpec(rank=8, lora_alpha=16, target_modules=[]),
            )
