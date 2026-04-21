# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the Trainer protocol using a fake in-memory implementation.

These pin the contract so a future AxolotlTrainer or torchtitan trainer
implementation can be plugged in confidently — the worker only calls
into the protocol, never VanillaTrainer directly.
"""

from __future__ import annotations

from hatchery.core.trainer import (
    ForwardBackwardResult,
    ForwardOnlyResult,
    LogprobsResult,
    LoraSpec,
    OptimStepResult,
    SampleResult,
    TrainerState,
)


class _FakeTrainer:
    """Minimal Trainer-protocol-compatible stub used only for structural tests."""

    base_model_name = "fake-model"
    tokenizer = None

    def __init__(self) -> None:
        self._sessions: dict[str, TrainerState] = {}
        self._calls: list[tuple[str, str]] = []

    def attach_session(self, session_id: str, spec: LoraSpec) -> None:
        self._calls.append(("attach", session_id))
        self._sessions.setdefault(
            session_id,
            TrainerState(
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
            ),
        )

    def detach_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def load_state(self, session_id: str, state: TrainerState) -> None:
        self._sessions[session_id] = state

    def extract_state(self, session_id: str) -> TrainerState:
        return self._sessions[session_id]

    def init_session_state(self, session_id: str, spec: LoraSpec) -> TrainerState:
        self.attach_session(session_id, spec)
        return self._sessions[session_id]

    def forward_backward(self, session_id, data, loss_fn):
        state = self._sessions[session_id]
        state.meta["accum_steps"] = state.meta.get("accum_steps", 0) + 1
        return ForwardBackwardResult(
            loss=0.123, num_tokens=42, accum_steps=state.meta["accum_steps"]
        )

    def forward_only(self, session_id, data, loss_fn, loss_fn_config=None):
        # Key invariant: forward_only does NOT mutate session training
        # state (no accum_steps bump, no grad accumulation).
        _ = self._sessions[session_id]
        return ForwardOnlyResult(loss=0.456, num_tokens=17)

    def optim_step(self, session_id, adam_params):
        state = self._sessions[session_id]
        state.meta["total_steps"] = state.meta.get("total_steps", 0) + 1
        state.meta["accum_steps"] = 0
        return OptimStepResult(
            step=state.meta["total_steps"],
            learning_rate=adam_params.get("learning_rate", 1e-4),
        )

    def sample(self, session_id, prompt_tokens, params):
        return SampleResult(sequences=[[1, 2, 3]], texts=["hello"], total_tokens=3)

    def compute_logprobs(self, session_id, input_tokens):
        return LogprobsResult(logprobs=[[-0.1, -0.2]], total_tokens=2)


def test_fake_trainer_matches_protocol_shape():
    trainer = _FakeTrainer()
    spec = LoraSpec(rank=8, lora_alpha=16, target_modules=["q_proj"])
    state = trainer.init_session_state("s1", spec)
    assert trainer.has_session("s1")
    assert state.meta["lora_config"]["r"] == 8

    fb = trainer.forward_backward("s1", [{"input_ids": [1, 2, 3]}], "cross_entropy")
    assert fb.accum_steps == 1

    # forward_only must not touch accum_steps — eval-only op.
    fo = trainer.forward_only("s1", [{"input_ids": [1, 2, 3]}], "cross_entropy")
    assert isinstance(fo, ForwardOnlyResult)
    assert fo.loss == 0.456
    assert fo.num_tokens == 17
    assert state.meta["accum_steps"] == 1  # unchanged by forward_only

    step = trainer.optim_step("s1", {"learning_rate": 3e-4})
    assert step.step == 1
    assert step.learning_rate == 3e-4

    sample = trainer.sample("s1", [1, 2], {"max_tokens": 8})
    assert sample.sequences == [[1, 2, 3]]

    lp = trainer.compute_logprobs("s1", [[1, 2, 3]])
    assert len(lp.logprobs) == 1


def test_extract_state_roundtrip():
    trainer = _FakeTrainer()
    spec = LoraSpec(rank=8, lora_alpha=16, target_modules=["q_proj"])
    trainer.init_session_state("s1", spec)
    trainer.forward_backward("s1", [{"input_ids": [1, 2]}], "cross_entropy")
    state = trainer.extract_state("s1")
    assert state.meta["accum_steps"] == 1

    # Simulate worker evict/reload.
    trainer.detach_session("s1")
    assert not trainer.has_session("s1")
    trainer.attach_session("s1", spec)
    trainer.load_state("s1", state)
    assert trainer.extract_state("s1").meta["accum_steps"] == 1


def test_vanilla_trainer_matches_protocol_surface():
    """Structural check: VanillaTrainer declares every method on Trainer."""
    from hatchery.core.trainer import VanillaTrainer

    for method in (
        "attach_session",
        "detach_session",
        "has_session",
        "load_state",
        "extract_state",
        "init_session_state",
        "forward_backward",
        "forward_only",
        "optim_step",
        "sample",
        "compute_logprobs",
    ):
        assert callable(getattr(VanillaTrainer, method)), method
