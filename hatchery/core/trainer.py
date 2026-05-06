# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Trainer abstraction.

The GPU worker is intentionally thin — it pulls jobs from the queue,
reads session state from the object store, delegates the actual
training to a :class:`Trainer`, and writes state back out. This file
defines the protocol and ships the current in-process implementation
as :class:`VanillaTrainer`.

Why an abstraction
------------------
Three concrete trainers are on the table:

1. :class:`VanillaTrainer` — what we already have. Minimal deps,
   single-process, good enough for tests and small models.
2. :class:`AxolotlTrainer` — a thin wrapper around
   https://github.com/axolotl-ai-cloud/axolotl that inherits its
   FSDP2/TP/CP/offload wiring, sample packing, multipack, RL objectives
   (GRPO/DPO/KTO), and optimizer zoo. The right production choice.
3. A torchtitan-backed trainer for the pretraining-scale case.

The worker should not need to care which one it's talking to, so this
protocol is the contract. ``load_state``/``extract_state`` are the
serialization boundary (tensors on CPU, keyed by name), so the worker
can keep its object-store round-trip logic unchanged.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol

from hatchery.core.distributed import (
    apply_core_fsdp2_dp,
    init_distributed_runtime,
)
from hatchery.core.parallel import ParallelConfig

# Torch is an optional dependency. The gateway, scripted test trainer,
# and the trainer protocol types don't need it — only :class:`VanillaTrainer`
# does. We try the import at module load time; if it fails we leave
# ``torch`` / ``F`` as ``None`` and raise a clear error the first time
# someone tries to instantiate VanillaTrainer. Type annotations are
# string form (via ``from __future__ import annotations``) so dataclass
# fields like ``dict[str, torch.Tensor]`` don't force an eager import.
try:  # pragma: no cover — the happy path is "torch installed".
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

if TYPE_CHECKING:
    import torch  # noqa: F401


@dataclass
class TrainerState:
    """Serialization wrapper for a session's in-memory state.

    The ``lora_weights`` field carries the session's trainable
    parameters keyed by module name. Despite the name, in
    ``full_param`` mode it holds the full base-model state dict
    rather than a LoRA delta — kept for backward compatibility
    with on-disk artefacts written before full-param mode existed.
    """

    lora_weights: dict[str, torch.Tensor]
    grad_accum: dict[str, torch.Tensor] = field(default_factory=dict)
    optimizer_state: Optional[dict] = None
    meta: dict = field(default_factory=dict)


@dataclass
class LoraSpec:
    rank: int
    lora_alpha: int
    target_modules: list[str]
    use_rslora: bool = False
    init_lora_weights: str = "default"
    lora_dropout: float = 0.0
    # "lora" wraps the base in PEFT and trains a per-session adapter.
    # "full_param" trains the base model directly — the LoRA-specific
    # fields above are ignored. A trainer holding a full-param session
    # cannot also hold LoRA sessions (they fight over base weights).
    mode: str = "lora"

    @property
    def is_full_param(self) -> bool:
        return self.mode == "full_param"

    @staticmethod
    def full_param() -> LoraSpec:
        """Build a spec describing a full-parameter (non-LoRA) session.

        The LoRA-specific fields are zeroed out — they're ignored when
        ``mode == "full_param"`` but we still need *some* value because
        the dataclass fields are required. Use this constructor instead
        of building one by hand to make intent obvious at call sites.
        """
        return LoraSpec(
            rank=0,
            lora_alpha=0,
            target_modules=[],
            mode="full_param",
        )


@dataclass
class ForwardBackwardResult:
    loss: float
    num_tokens: int
    accum_steps: int


@dataclass
class ForwardOnlyResult:
    """Result of a no-grad forward pass with a caller-supplied loss.

    Semantically mirrors :class:`ForwardBackwardResult` minus anything
    that would imply gradient accumulation — no ``accum_steps`` field,
    since forward-only never mutates session training state.
    """

    loss: float
    num_tokens: int


@dataclass
class OptimStepResult:
    step: int
    learning_rate: float


@dataclass
class SampleResult:
    sequences: list[list[int]]
    texts: list[str]
    total_tokens: int


@dataclass
class LogprobsResult:
    logprobs: list[list[float]]
    total_tokens: int


class Trainer(Protocol):
    """Trainer contract — the worker holds one instance per process."""

    base_model_name: str
    tokenizer: Any  # HF tokenizer-compatible

    def attach_session(self, session_id: str, spec: LoraSpec) -> None:
        """Create an adapter for ``session_id`` with the given LoRA spec."""
        ...

    def detach_session(self, session_id: str) -> None:
        """Tear down adapter resources for ``session_id``."""
        ...

    def has_session(self, session_id: str) -> bool: ...

    def load_state(self, session_id: str, state: TrainerState) -> None: ...

    def extract_state(self, session_id: str) -> TrainerState: ...

    def init_session_state(self, session_id: str, spec: LoraSpec) -> TrainerState:
        """Return freshly initialized state for a newly created session."""
        ...

    # Operations
    def forward_backward(
        self, session_id: str, data: list[dict], loss_fn: str
    ) -> ForwardBackwardResult: ...
    def forward_only(
        self,
        session_id: str,
        data: list[dict],
        loss_fn: str,
        loss_fn_config: Optional[dict] = None,
    ) -> ForwardOnlyResult:
        """Run a no-grad forward pass computing the given loss.

        Mirrors :meth:`forward_backward` but without ``.backward()``,
        without CPU grad accumulation, and without bumping
        ``accum_steps``. Use this for eval / held-out loss computation
        against a session without disturbing its training trajectory.
        """
        ...

    def optim_step(self, session_id: str, adam_params: dict) -> OptimStepResult: ...
    def sample(self, session_id: str, prompt_tokens: list[int], params: dict) -> SampleResult: ...
    def compute_logprobs(
        self, session_id: str, input_tokens: list[list[int]]
    ) -> LogprobsResult: ...


# ─── Vanilla (in-process) trainer ────────────────────────────────────────


class VanillaTrainer:
    """Single-process PEFT-on-transformers trainer.

    This is a straight extraction of the logic that lives in
    :class:`hatchery.core.worker.GPUWorker`. It now owns the model,
    the PEFT wrapper, and the per-session optimizer state; the worker
    simply asks it to run operations.

    Multi-GPU support is wired through :class:`ParallelConfig` and
    :class:`hatchery.core.distributed.DistributedRuntime`. Core owns
    DP-only FSDP2; TP/CP remain extension-owned.
    """

    def __init__(
        self,
        base_model_name: str,
        *,
        device: str = "cuda:0",
        dtype: Any = None,
        attn_implementation: str = "sdpa",
        parallel: Optional[ParallelConfig] = None,
        load_model: bool = True,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "VanillaTrainer requires torch + transformers + peft. "
                'Install with `uv pip install -e ".[gpu]"`.'
            )
        if dtype is None:
            dtype = torch.bfloat16

        self.base_model_name = base_model_name
        self.device = device
        self.dtype = dtype
        self.parallel = parallel or ParallelConfig()
        # Sequence packing (varlen) v1 requires HF's flash-attn-2 path:
        # it's the only backend that derives ``cu_seqlens`` from the
        # ``position_ids`` resets. Override any caller-supplied value
        # silently — explicit sequence_packing=True is a strong enough
        # signal, and SDPA would silently ignore the resets and attend
        # across doc boundaries.
        if self.parallel.sequence_packing:
            attn_implementation = "flash_attention_2"
        self.attn_implementation = attn_implementation

        self._raw_base: Any = None
        self._peft: Any = None  # PeftModel after first attach
        self.tokenizer: Any = None

        # Per-session bookkeeping kept out of the model graph so swapping
        # is cheap.
        self._grad_accum: dict[str, dict[str, torch.Tensor]] = {}
        self._optimizer_state: dict[str, Optional[dict]] = {}
        self._specs: dict[str, LoraSpec] = {}
        self._meta: dict[str, dict] = {}

        # Base-weight management for mixed-mode (LoRA + full-param)
        # operation. ``_pristine_base_sd`` is captured once at load
        # time, before PEFT touches anything, and is what every LoRA
        # session sees as its "base". ``_fp_base_state`` holds the
        # trained base weights of each full-param session so we can
        # swap them in when that session is reactivated. The active
        # session id tracks which slot is currently materialized into
        # ``_raw_base`` so we don't pay for redundant swaps.
        self._pristine_base_sd: Optional[dict[str, torch.Tensor]] = None
        self._fp_base_state: dict[str, dict[str, torch.Tensor]] = {}
        self._active_session_id: Optional[str] = None

        # Tracks whether we've applied the parallel plan to ``self._peft``
        # yet. FSDP2/TP are deferred until after the first PEFT adapter
        # is attached — see attach_session for the "why".
        self._parallel_applied = False

        # Quantization scheme of the loaded base ("none" | "onebit").
        # Set by :meth:`_load_base`; consulted by
        # :meth:`_attach_full_param_session` to enforce the
        # LoRA-required policy for BitNet bases.
        self._quant_scheme: str = "none"

        self._distributed_runtime = init_distributed_runtime(self.parallel)
        self._mesh = self._distributed_runtime.mesh
        self._dp_mesh = self._distributed_runtime.dp_mesh

        if load_model:
            self._load_base()

    # ── Model bootstrap ─────────────────────────────────────

    def _load_base(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from hatchery.core.quantization import (
            is_onebit_by_name,
            is_onebit_model,
            prepare_onebit_loader_kwargs,
        )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
        }
        # 1-bit / BitNet routing. The trainer has a ``ParallelConfig``
        # with an optional :class:`QuantConfig`; combined with the
        # model name, this tells us whether to run the onebit loader
        # path. Detection is cheap (no network) — only touches the
        # slug — which keeps the full-precision common case fast.
        quant_cfg = self.parallel.quant
        if quant_cfg.is_onebit or is_onebit_by_name(self.base_model_name):
            load_kwargs = prepare_onebit_loader_kwargs(load_kwargs)
            self._quant_scheme = "onebit"
        else:
            self._quant_scheme = "none"

        self._raw_base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **load_kwargs,
        ).to(self.device)
        self._raw_base.eval()
        for p in self._raw_base.parameters():
            p.requires_grad = False

        # Re-detect after load — the HF config on the real model is
        # authoritative and may flip "none" → "onebit" if a hand-
        # edited config didn't match the slug.
        real_cfg = getattr(self._raw_base, "config", None)
        if real_cfg is not None and is_onebit_model(real_cfg, model_name=self.base_model_name):
            self._quant_scheme = "onebit"

        # Snapshot the pristine base now, before any session can attach
        # a LoRA adapter or mutate weights. Stored CPU-side so it can
        # outlive any number of full-param training runs without
        # competing for VRAM. Keys are pre-PEFT — see ``_resolve_live_key``
        # for how they're remapped at restore time when PEFT is wrapped.
        self._pristine_base_sd = {
            k: v.detach().cpu().clone() for k, v in self._raw_base.state_dict().items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── Session management ─────────────────────────────────

    @staticmethod
    def _adapter_name(session_id: str) -> str:
        return "sess_" + session_id.replace("-", "_")

    def has_session(self, session_id: str) -> bool:
        spec = self._specs.get(session_id)
        if spec is not None and spec.is_full_param:
            return True
        if self._peft is None:
            return False
        return self._adapter_name(session_id) in getattr(self._peft, "peft_config", {})

    def _active_module(self, session_id: str) -> Any:
        """Return the nn.Module to run for this session.

        Triggers the base-weight swap if the active session is changing,
        flips the active adapter on the PEFT wrapper for LoRA sessions,
        and adjusts ``requires_grad`` so only the right parameters get
        grads on the next backward pass.
        """
        spec = self._specs.get(session_id)
        if spec is None:
            raise RuntimeError(f"unknown session {session_id}")

        if self._active_session_id != session_id:
            # Outgoing: if the previous session was full-param, snapshot
            # its current base weights so we can restore them next time.
            self._stash_active_full_param_base()
            # Incoming: load the right base weights for this session.
            if spec.is_full_param:
                self._restore_full_param_base(session_id)
            else:
                self._restore_pristine_base()
            self._active_session_id = session_id
            self._set_grad_for_session(session_id)
        elif not spec.is_full_param:
            # Same LoRA session, but session-id equality alone doesn't
            # guarantee the right adapter is the one PEFT thinks is
            # active (other code paths may have called set_adapter).
            # Cheap to re-set; cheaper than debugging a silent swap.
            adapter = self._adapter_name(session_id)
            self._peft.set_adapter(adapter)

        if spec.is_full_param:
            return self._raw_base
        return self._peft

    @staticmethod
    def _is_lora_param_name(name: str) -> bool:
        """Whether ``name`` (from ``named_parameters``) is a LoRA adapter param.

        Matches PEFT's ``lora_A``/``lora_B`` modules across attention,
        MLP, and embedding wrappers (``lora_embedding_A``/``B``). The
        check is path-aware (``.lora_A.``, etc.) so a base parameter
        whose own name happens to contain the substring elsewhere is
        not misclassified.
        """
        return (
            ".lora_A." in name
            or ".lora_B." in name
            or ".lora_embedding_A." in name
            or ".lora_embedding_B." in name
        )

    def _set_grad_for_session(self, session_id: str) -> None:
        """Set ``requires_grad`` so only the right params train this step.

        For LoRA sessions: freeze the base, let PEFT manage which
        adapter's lora_A/B are trainable (``set_adapter`` does this).
        For full-param: enable base params, freeze any LoRA adapter
        params that happen to be wrapping the model from a sibling
        LoRA session — those must not accumulate grads while the
        full-param session runs.
        """
        spec = self._specs.get(session_id)
        if spec is None:
            return
        if spec.is_full_param:
            for name, p in self._raw_base.named_parameters():
                p.requires_grad_(not self._is_lora_param_name(name))
        else:
            adapter = self._adapter_name(session_id)
            for p in self._raw_base.parameters():
                p.requires_grad_(False)
            if self._peft is not None:
                self._peft.set_adapter(adapter)  # re-enables this adapter's lora_A/B

    def _stash_active_full_param_base(self) -> None:
        """If the currently-active session is full-param, copy its
        live base weights back into ``_fp_base_state`` so a future
        activation can restore them."""
        if self._active_session_id is None:
            return
        spec = self._specs.get(self._active_session_id)
        if spec is None or not spec.is_full_param:
            return
        self._fp_base_state[self._active_session_id] = self._capture_live_base_weights()

    def _capture_live_base_weights(self) -> dict[str, torch.Tensor]:
        """Snapshot the current base-portion weights of ``_raw_base``.

        Skips LoRA adapter params (``lora_A``/``lora_B``) and strips
        the ``.base_layer.`` segment from PEFT-wrapped keys so the
        returned state dict has pre-PEFT keys — portable to a trainer
        that may or may not be wrapped in PEFT later.
        """
        out: dict[str, torch.Tensor] = {}
        for k, v in self._raw_base.state_dict().items():
            if ".lora_A." in k or ".lora_B." in k:
                continue
            pristine_k = k.replace(".base_layer.", ".")
            out[pristine_k] = v.detach().cpu().clone()
        return out

    def _restore_pristine_base(self) -> None:
        """Load the pristine snapshot back into ``_raw_base``.

        Used when activating a LoRA session — every LoRA adapter is
        defined relative to the pristine base, so any drift from
        prior full-param training must be undone first.
        """
        if self._pristine_base_sd is None:
            return
        self._load_base_sd_into_live(self._pristine_base_sd)

    def _restore_full_param_base(self, session_id: str) -> None:
        """Load this session's saved base weights into ``_raw_base``."""
        sd = self._fp_base_state.get(session_id)
        if sd is None:
            # No prior snapshot (e.g. first activation after attach
            # filled it from pristine). Fall back to pristine to keep
            # behavior defined.
            self._restore_pristine_base()
            return
        self._load_base_sd_into_live(sd)

    def _exec_context(self, spec: LoraSpec):
        """Return a CM to wrap a forward pass for this session.

        For full-param sessions on a trainer that has LoRA-wrapped the
        base for some sibling session, ``peft.disable_adapter`` zeros
        out the lora_A/lora_B contribution so the forward pass uses
        only base weights — what full-param training expects.
        Otherwise this is a no-op.
        """
        from contextlib import nullcontext

        if spec.is_full_param and self._peft is not None:
            return self._peft.disable_adapter()
        return nullcontext()

    def _load_base_sd_into_live(self, base_sd: dict[str, torch.Tensor]) -> None:
        """Load a pre-PEFT-keyed base state dict into the (possibly
        PEFT-wrapped) ``_raw_base``.

        Each pristine key like ``model.layers.0.self_attn.q_proj.weight``
        is matched against the live state dict, with ``.base_layer.``
        inserted before the trailing param name when PEFT has wrapped
        that module. Loaded with ``strict=False`` so any LoRA adapter
        keys present in the live model stay untouched.
        """
        live_keys = set(self._raw_base.state_dict().keys())
        payload: dict[str, torch.Tensor] = {}
        for k, v in base_sd.items():
            if k in live_keys:
                live_k = k
            else:
                # Try the PEFT-wrapped form: insert .base_layer before
                # the final ``.weight``/``.bias`` segment.
                head, _, tail = k.rpartition(".")
                wrapped = f"{head}.base_layer.{tail}" if head else k
                if wrapped in live_keys:
                    live_k = wrapped
                else:
                    # Key vanished from the live model (rare — e.g.
                    # tied-weights or a buffer that's been re-shaped).
                    # Skip rather than crash; ``strict=False`` would
                    # have done the same.
                    continue
            payload[live_k] = v.to(self.device, dtype=v.dtype)
        self._raw_base.load_state_dict(payload, strict=False)

    def attach_session(self, session_id: str, spec: LoraSpec) -> None:
        if spec.is_full_param:
            self._attach_full_param_session(session_id, spec)
            return

        # BitNet bases are trained quantization-aware — the published
        # BF16 checkpoints are master weights meant for full-parameter
        # fine-tuning. A LoRA adapter still *works* (base is frozen,
        # deltas train), but the low-rank path sits before the STE
        # quantizer and the training signal is weak. Refuse by default;
        # callers who know what they want must flip
        # ``parallel.quant.require_full_param = False``.
        if self._quant_scheme == "onebit" and getattr(
            self.parallel.quant, "require_full_param", True
        ):
            raise RuntimeError(
                "LoRA fine-tuning on a 1-bit / BitNet base is refused by "
                "default: BitNet's recipe is full-parameter training on "
                "the BF16 master weights, and LoRA deltas before the STE "
                "quantizer produce muddled gradients. Use a full-param "
                "session (LoraSpec(..., is_full_param=True)), or opt in "
                "to LoRA explicitly by setting "
                "ParallelConfig.quant.require_full_param = False."
            )

        adapter = self._adapter_name(session_id)
        first_adapter = self._peft is None
        adapter_exists = (
            self._peft is not None and adapter in getattr(self._peft, "peft_config", {})
        )
        if (
            self._distributed_runtime.is_core_dp_only
            and self._parallel_applied
            and not adapter_exists
        ):
            raise RuntimeError(
                "Core FSDP2 DP currently supports only the first LoRA adapter "
                "attached before FSDP wrapping. Dynamic adapter attach, reload, "
                "eviction, and cross-rank portability for later adapters are "
                "unsupported in v1."
            )

        from peft import LoraConfig, get_peft_model

        # If a full-param session was previously active, the base in
        # ``_raw_base`` is its trained weights, not pristine. Stash
        # those before letting LoRA wrap a (potentially) different
        # base — the LoRA session expects the pristine base it was
        # set up against.
        self._stash_active_full_param_base()
        self._restore_pristine_base()
        lora_kwargs = {
            "r": spec.rank,
            "lora_alpha": spec.lora_alpha,
            "target_modules": list(spec.target_modules),
            "lora_dropout": spec.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if spec.use_rslora:
            lora_kwargs["use_rslora"] = True
        if spec.init_lora_weights != "default":
            lora_kwargs["init_lora_weights"] = spec.init_lora_weights
        lora_config = LoraConfig(**lora_kwargs)
        if first_adapter:
            self._peft = get_peft_model(self._raw_base, lora_config, adapter_name=adapter)
        elif not adapter_exists:
            self._peft.add_adapter(adapter, lora_config)
        self._peft.set_adapter(adapter)
        self._specs[session_id] = spec
        self._meta.setdefault(session_id, {"accum_steps": 0, "total_steps": 0})
        self._grad_accum.setdefault(session_id, {})
        self._optimizer_state.setdefault(session_id, None)

        # Apply the parallel plan only after the first adapter is in
        # place. FSDP2 must wrap the *peft-wrapped* decoder layer so that
        # the LoRA modules live inside the sharded group; otherwise the
        # lora_A/lora_B out_features are sized from DTensor views and
        # you get the "tensor a (896) must match tensor b (448)" error.
        if first_adapter and self.parallel.is_distributed() and not self._parallel_applied:
            self._apply_parallel_plan()
            self._parallel_applied = True

    def _attach_full_param_session(self, session_id: str, spec: LoraSpec) -> None:
        """Attach a session that trains the raw base model directly.

        Full-param sessions can coexist with LoRA sessions on the same
        trainer; they each carry a private base-weight snapshot in
        ``_fp_base_state`` that's swapped into ``_raw_base`` whenever
        the session becomes active. New full-param sessions start from
        the pristine base captured at load time.
        """
        if self._distributed_runtime.is_core_dp_only:
            raise RuntimeError(
                "Full-parameter sessions are unsupported under core FSDP2 DP "
                "in v1. Use LoRA with a single adapter, or run full-parameter "
                "training with dp_degree=1."
            )
        if session_id in self._specs:
            existing = self._specs[session_id]
            if not existing.is_full_param:
                raise RuntimeError(
                    f"session {session_id!r} is already attached as LoRA; "
                    "cannot re-attach as full-param."
                )
            return  # idempotent re-attach
        # Initialize this session's base snapshot to pristine. The
        # actual swap into ``_raw_base`` and grad-bit management
        # happen at activation time, not here — attach is just
        # bookkeeping.
        if self._pristine_base_sd is None:
            raise RuntimeError(
                "_attach_full_param_session called before _load_base — pristine snapshot missing."
            )
        self._fp_base_state[session_id] = {k: v.clone() for k, v in self._pristine_base_sd.items()}
        self._specs[session_id] = spec
        self._meta.setdefault(session_id, {"accum_steps": 0, "total_steps": 0})
        self._grad_accum.setdefault(session_id, {})
        self._optimizer_state.setdefault(session_id, None)

    def _apply_parallel_plan(self) -> None:
        """Apply the runtime-selected parallel plan to the PEFT-wrapped base.

        Called exactly once, the first time a session is attached. After
        this runs the base-model decoder layers are sharded across the
        DP mesh and every subsequent session just adds an adapter on top
        of the already-parallelized modules.
        """
        if self._peft is None or not self._distributed_runtime.is_distributed:
            return
        if self._distributed_runtime.is_core_dp_only:
            apply_core_fsdp2_dp(self._peft, self._distributed_runtime, self.parallel)
            return

        extension = self._distributed_runtime.extension_handle
        apply_plan = getattr(extension, "apply_parallel_plan", None)
        if callable(apply_plan):
            apply_plan(self._peft, self._distributed_runtime, self.parallel)
            return
        raise RuntimeError(
            "Distributed runtime did not provide a parallel plan for "
            f"dp={self.parallel.dp_degree},tp={self.parallel.tp_degree},"
            f"cp={self.parallel.cp_degree}."
        )

    def _allocate_batch(self, data: list[dict]) -> list[dict]:
        """Route DP batch allocation through the shared helper.

        Passthrough on single-rank trainers. On multi-rank FSDP
        trainers, either replicates or splits based on the strategy
        configured on ``self.parallel``. See
        :func:`hatchery.core.batching.prepare_batch_for_dp`.
        """
        if not self.parallel.is_distributed() or self._distributed_runtime.dp_world_size <= 1:
            return list(data)

        from hatchery.core.batching import BatchStrategy, prepare_batch_for_dp

        try:
            strategy = BatchStrategy(self.parallel.batch_strategy)
        except ValueError:
            strategy = BatchStrategy.AUTO
        allocation = prepare_batch_for_dp(
            data,
            dp_degree=self._distributed_runtime.dp_world_size,
            rank=self._distributed_runtime.dp_rank,
            strategy=strategy,
        )
        return allocation.data

    def detach_session(self, session_id: str) -> None:
        spec = self._specs.get(session_id)
        if self._distributed_runtime.is_core_dp_only and spec is not None:
            raise RuntimeError(
                "Core FSDP2 DP does not support session detach/eviction in v1; "
                "adapter deletion and later reload after FSDP wrapping are not "
                "portable yet."
            )
        if spec is not None and not spec.is_full_param:
            adapter = self._adapter_name(session_id)
            if self._peft is not None and adapter in getattr(self._peft, "peft_config", {}):
                try:
                    self._peft.delete_adapter(adapter)
                except Exception:  # noqa: BLE001
                    pass
        # Drop full-param base snapshot if any; if this was the active
        # session, restore pristine base so the next attach starts clean.
        self._fp_base_state.pop(session_id, None)
        if self._active_session_id == session_id:
            self._restore_pristine_base()
            self._active_session_id = None
        self._specs.pop(session_id, None)
        self._meta.pop(session_id, None)
        self._grad_accum.pop(session_id, None)
        self._optimizer_state.pop(session_id, None)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def init_session_state(self, session_id: str, spec: LoraSpec) -> TrainerState:
        self.attach_session(session_id, spec)
        if spec.is_full_param:
            # Full-param: don't snapshot the (potentially huge) base
            # weights at init time — the base hasn't diverged from
            # ``base_model_name`` yet, so a downstream loader can
            # reconstruct it from HF. ``extract_state`` will populate
            # the dict once the session has trained.
            return TrainerState(
                lora_weights={},
                grad_accum={},
                optimizer_state=None,
                meta={
                    "accum_steps": 0,
                    "total_steps": 0,
                    "training_mode": "full_param",
                },
            )

        from peft.utils import get_peft_model_state_dict

        adapter = self._adapter_name(session_id)
        weights = get_peft_model_state_dict(self._peft, adapter_name=adapter)
        weights_cpu = {k: v.detach().cpu() for k, v in weights.items()}
        return TrainerState(
            lora_weights=weights_cpu,
            grad_accum={},
            optimizer_state=None,
            meta={
                "accum_steps": 0,
                "total_steps": 0,
                "training_mode": "lora",
                "lora_config": {
                    "r": spec.rank,
                    "lora_alpha": spec.lora_alpha,
                    "target_modules": list(spec.target_modules),
                },
            },
        )

    # ── State (de)serialization ─────────────────────────────

    def load_state(self, session_id: str, state: TrainerState) -> None:
        # Mode dispatch: prefer the explicit ``training_mode`` tag,
        # fall back to "lora" if a ``lora_config`` is present (default
        # state files predate the tag), then to whatever spec is
        # already attached.
        mode = state.meta.get("training_mode")
        if mode is None:
            mode = "lora" if state.meta.get("lora_config") else None

        if mode == "full_param":
            spec = LoraSpec.full_param()
            self.attach_session(session_id, spec)
            if state.lora_weights:
                # Update both the stashed snapshot (so a future
                # activation restores these weights) and, if this
                # session is currently active, the live model.
                self._fp_base_state[session_id] = {
                    k: v.detach().cpu().clone() for k, v in state.lora_weights.items()
                }
                if self._active_session_id == session_id:
                    self._load_base_sd_into_live(self._fp_base_state[session_id])
            self._grad_accum[session_id] = dict(state.grad_accum)
            self._optimizer_state[session_id] = state.optimizer_state
            self._meta[session_id] = dict(state.meta) or {
                "accum_steps": 0,
                "total_steps": 0,
            }
            return

        from peft.utils import set_peft_model_state_dict

        spec_dict = state.meta.get("lora_config")
        if spec_dict is None:
            spec = self._specs.get(session_id)
            if spec is None:
                raise RuntimeError(
                    f"Cannot load state for unknown session {session_id} "
                    "without a lora_config in the meta dict"
                )
        else:
            spec = LoraSpec(
                rank=spec_dict["r"],
                lora_alpha=spec_dict["lora_alpha"],
                target_modules=list(spec_dict["target_modules"]),
            )

        self.attach_session(session_id, spec)
        adapter = self._adapter_name(session_id)
        if state.lora_weights:
            set_peft_model_state_dict(self._peft, state.lora_weights, adapter_name=adapter)
        self._grad_accum[session_id] = dict(state.grad_accum)
        self._optimizer_state[session_id] = state.optimizer_state
        self._meta[session_id] = dict(state.meta) or {
            "accum_steps": 0,
            "total_steps": 0,
        }

    def extract_state(self, session_id: str) -> TrainerState:
        spec = self._specs.get(session_id)
        meta = dict(self._meta.get(session_id, {}))

        if spec is not None and spec.is_full_param:
            # If this session is currently materialized in ``_raw_base``,
            # snapshot from there; otherwise return the stashed copy.
            # Either way the result is pre-PEFT-keyed and skips LoRA
            # adapter params from any sibling session.
            if self._active_session_id == session_id:
                weights_cpu = self._capture_live_base_weights()
            else:
                weights_cpu = {
                    k: v.clone() for k, v in self._fp_base_state.get(session_id, {}).items()
                }
            meta["training_mode"] = "full_param"
            return TrainerState(
                lora_weights=weights_cpu,
                grad_accum=dict(self._grad_accum.get(session_id, {})),
                optimizer_state=self._optimizer_state.get(session_id),
                meta=meta,
            )

        from peft.utils import get_peft_model_state_dict

        adapter = self._adapter_name(session_id)
        weights = get_peft_model_state_dict(self._peft, adapter_name=adapter)
        weights_cpu = {k: v.detach().cpu() for k, v in weights.items()}
        # Always roll lora_config into meta so a future reload can
        # reconstruct the adapter shape from the saved state alone.
        # Without this, a session whose trainer state was only touched
        # via attach_session (not init_session_state) would forget its
        # rank / target_modules on serialization.
        meta["training_mode"] = "lora"
        if spec is not None:
            meta["lora_config"] = {
                "r": spec.rank,
                "lora_alpha": spec.lora_alpha,
                "target_modules": list(spec.target_modules),
            }
        return TrainerState(
            lora_weights=weights_cpu,
            grad_accum=dict(self._grad_accum.get(session_id, {})),
            optimizer_state=self._optimizer_state.get(session_id),
            meta=meta,
        )

    # ── Operations ──────────────────────────────────────────

    def forward_backward(
        self, session_id: str, data: list[dict], loss_fn: str
    ) -> ForwardBackwardResult:
        model = self._active_module(session_id)
        spec = self._specs[session_id]
        model.train()

        data = self._allocate_batch(data)
        sub_batches = self._collate(data)

        # Sum non-ignored label positions across all sub-batches so each
        # sub-backward can scale its mean-loss by (sub_tokens / total_tokens).
        # That reconstructs the true whole-batch gradient from N separate
        # .backward() calls — without this the grads from N packs would be
        # N× larger than the equivalent single-forward gradient.
        sub_tokens = [int((b["labels"] != -100).sum().cpu()) for b in sub_batches]
        total_tokens = max(sum(sub_tokens), 1)

        for p in model.parameters():
            if p.requires_grad:
                p.grad = None

        weighted_loss_sum = 0.0
        for batch, num in zip(sub_batches, sub_tokens, strict=False):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attn_mask = batch["attention_mask"]
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device)
            position_ids = batch["position_ids"]
            if position_ids is not None:
                position_ids = position_ids.to(self.device)

            cp_ctx = self._context_parallel_region(input_ids, attn_mask, labels)

            with self._exec_context(spec), cp_ctx:
                outputs = model(**self._model_kwargs(input_ids, attn_mask, position_ids))
                logits = outputs.logits
                loss = self._loss(logits, labels, loss_fn)
                scale = num / total_tokens
                (loss * scale).backward()

            weighted_loss_sum += float(loss.detach().cpu()) * num

        loss_val = weighted_loss_sum / total_tokens

        accum = self._grad_accum.setdefault(session_id, {})
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            g = param.grad.detach().float().cpu()
            accum[name] = accum.get(name, torch.zeros_like(g)) + g

        meta = self._meta.setdefault(session_id, {"accum_steps": 0, "total_steps": 0})
        meta["accum_steps"] = meta.get("accum_steps", 0) + 1

        return ForwardBackwardResult(
            loss=loss_val,
            num_tokens=total_tokens,
            accum_steps=meta["accum_steps"],
        )

    def forward_only(
        self,
        session_id: str,
        data: list[dict],
        loss_fn: str,
        loss_fn_config: Optional[dict] = None,
    ) -> ForwardOnlyResult:
        model = self._active_module(session_id)
        spec = self._specs[session_id]
        # eval() — matches ``compute_logprobs`` / ``sample``. Keeps
        # LoRA dropout off for deterministic eval loss.
        model.eval()

        data = self._allocate_batch(data)
        sub_batches = self._collate(data)
        sub_tokens = [int((b["labels"] != -100).sum().cpu()) for b in sub_batches]
        total_tokens = max(sum(sub_tokens), 1)

        weighted_loss_sum = 0.0
        for batch, num in zip(sub_batches, sub_tokens, strict=False):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attn_mask = batch["attention_mask"]
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device)
            position_ids = batch["position_ids"]
            if position_ids is not None:
                position_ids = position_ids.to(self.device)

            with self._exec_context(spec), torch.no_grad():
                outputs = model(**self._model_kwargs(input_ids, attn_mask, position_ids))
                logits = outputs.logits
                loss = self._loss(logits, labels, loss_fn, loss_fn_config=loss_fn_config)

            weighted_loss_sum += float(loss.detach().cpu()) * num

        loss_val = weighted_loss_sum / total_tokens
        return ForwardOnlyResult(loss=loss_val, num_tokens=total_tokens)

    def optim_step(self, session_id: str, adam_params: dict) -> OptimStepResult:
        model = self._active_module(session_id)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        accum = self._grad_accum.get(session_id, {})
        if not accum:
            raise RuntimeError("optim_step called with no accumulated grads")

        lr = float(adam_params.get("learning_rate", 1e-4))
        grad_clip_norm = float(adam_params.get("grad_clip_norm", 0.0))
        use_fused = str(self.device).startswith("cuda") and torch.cuda.is_available()
        optimizer = torch.optim.AdamW(
            [p for _, p in named],
            lr=lr,
            betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
            eps=adam_params.get("eps", 1e-8),
            weight_decay=adam_params.get("weight_decay", 0.0),
            fused=use_fused,
        )
        opt_state = self._optimizer_state.get(session_id)
        if opt_state is not None:
            with contextlib.suppress(Exception):
                optimizer.load_state_dict(opt_state)
            # load_state_dict overrides LR with saved value — re-apply client's request.
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        for name, param in named:
            g = accum.get(name)
            if g is None:
                param.grad = None
                continue
            param.grad = g.to(param.device, dtype=param.dtype)

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in named], max_norm=grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        self._optimizer_state[session_id] = _move_optimizer_state_to_cpu(optimizer.state_dict())
        self._grad_accum[session_id] = {
            name: torch.zeros_like(accum[name])
            if name in accum
            else torch.zeros_like(param.detach().cpu(), dtype=torch.float32)
            for name, param in named
        }
        meta = self._meta.setdefault(session_id, {"accum_steps": 0, "total_steps": 0})
        meta["total_steps"] = meta.get("total_steps", 0) + 1
        meta["accum_steps"] = 0
        return OptimStepResult(step=meta["total_steps"], learning_rate=lr)

    def sample(
        self,
        session_id: str,
        prompt_tokens: list[int],
        params: dict,
    ) -> SampleResult:
        model = self._active_module(session_id)
        spec = self._specs[session_id]
        model.eval()

        input_ids = torch.tensor([prompt_tokens], device=self.device, dtype=torch.long)
        max_new = int(params.get("max_tokens", 256))
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        n = int(params.get("n", 1))
        do_sample = temperature > 0 and (temperature != 1.0 or top_p != 1.0 or n > 1)
        if temperature == 0.0:
            do_sample = False

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p

        with self._exec_context(spec), torch.no_grad():
            out = model.generate(input_ids=input_ids, **gen_kwargs)

        prompt_len = input_ids.shape[1]
        sequences: list[list[int]] = []
        texts: list[str] = []
        for seq in out:
            gen_ids = seq[prompt_len:].tolist()
            sequences.append(gen_ids)
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        total = sum(len(s) for s in sequences)
        return SampleResult(sequences=sequences, texts=texts, total_tokens=total)

    def compute_logprobs(self, session_id: str, input_tokens: list[list[int]]) -> LogprobsResult:
        model = self._active_module(session_id)
        spec = self._specs[session_id]
        model.eval()

        results: list[list[float]] = []
        total = 0
        for tokens in input_tokens:
            input_ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
            with self._exec_context(spec), torch.no_grad():
                out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits.float()
            shifted_logits = logits[:, :-1, :]
            shifted_targets = input_ids[:, 1:]
            logprobs = F.log_softmax(shifted_logits, dim=-1)
            per_token = logprobs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
            results.append(per_token[0].cpu().tolist())
            total += per_token.numel()
        return LogprobsResult(logprobs=results, total_tokens=total)

    # ── Helpers ─────────────────────────────────────────────

    def _context_parallel_region(self, input_ids: Any, attn_mask: Any, labels: Any):
        """Return the extension-owned CP context, or a no-op for core DP."""
        from contextlib import nullcontext

        if self.parallel.cp_degree <= 1:
            return nullcontext()

        extension = self._distributed_runtime.extension_handle
        context_region = getattr(extension, "context_parallel_region", None)
        if callable(context_region):
            cp_mesh = getattr(self._distributed_runtime, "cp_mesh", None)
            if cp_mesh is None:
                from hatchery.core.parallel_hooks import get_distributed_helpers

                helpers = get_distributed_helpers()
                cp_mesh = helpers.get_cp_mesh(self._mesh, self.parallel)
            return context_region(
                cp_mesh,
                buffers=[input_ids, attn_mask, labels],
                seq_dims=[1, 1, 1],
                no_restore={input_ids, attn_mask, labels},
            )

        from hatchery.core.parallel_hooks import get_distributed_helpers

        helpers = get_distributed_helpers()
        cp_mesh = helpers.get_cp_mesh(self._mesh, self.parallel)
        return helpers.context_parallel_region(
            cp_mesh,
            buffers=[input_ids, attn_mask, labels],
            seq_dims=[1, 1, 1],
            no_restore={input_ids, attn_mask, labels},
        )

    def _collate(self, data: list[dict]) -> list[dict[str, Any]]:
        """Return a list of sub-batches, each a forward-call's worth of tensors.

        Per sub-batch dict keys: ``input_ids``, ``labels``,
        ``attention_mask``, ``position_ids``, ``slices``. In the default
        (pad-batch) mode this list is always length 1 with
        ``position_ids``/``slices = None``. In packing mode
        ``attention_mask`` is ``None`` — HF's flash-attn-2 path reads
        the ``position_ids`` resets to derive ``cu_seqlens`` — and the
        list has one entry per pack produced by first-fit-decreasing.
        """
        if self.parallel.sequence_packing:
            return self._collate_packed(data)
        return [self._collate_padded(data)]

    def _collate_padded(self, data: list[dict]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in data)
        input_ids_list = []
        labels_list = []
        attn_list = []
        for item in data:
            ids = list(item["input_ids"])
            lbls = list(item.get("labels", ids))
            if len(lbls) != len(ids):
                raise ValueError(f"labels length {len(lbls)} != input_ids length {len(ids)}")
            pad = max_len - len(ids)
            input_ids_list.append(ids + [pad_id] * pad)
            labels_list.append(lbls + [-100] * pad)
            attn_list.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "position_ids": None,
            "slices": None,
        }

    def _collate_packed(self, data: list[dict]) -> list[dict[str, Any]]:
        from hatchery.core.packing import pack_sequences

        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        max_len = self.parallel.max_packed_len or max(sum(len(it["input_ids"]) for it in data), 1)
        packs = pack_sequences(data, pad_id=pad_id, max_packed_len=max_len)
        return [
            {
                "input_ids": pack.input_ids,
                "attention_mask": None,
                "labels": pack.labels,
                "position_ids": pack.position_ids,
                "slices": pack.slices,
            }
            for pack in packs
        ]

    def _model_kwargs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        """Build the forward-call kwargs.

        When ``position_ids`` is provided (packed path) the attention
        mask is dropped — HF's flash-attn-2 backend infers cu_seqlens
        from the position_ids resets, and passing a rectangular mask
        would contradict that.
        """
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": None,
            "use_cache": False,
        }
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        else:
            kwargs["attention_mask"] = attention_mask
        return kwargs

    def _loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: str,
        *,
        weights: Optional[torch.Tensor] = None,
        old_logprobs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        loss_fn_config: Optional[dict] = None,
    ) -> torch.Tensor:
        """Dispatch to the shared loss registry in
        :mod:`hatchery.core.losses`. See ``worker._compute_loss`` for
        the same routing pattern — we keep the trainer's interface
        minimal so the Trainer protocol stays unchanged even as new
        losses land.
        """
        from hatchery.core.losses import LossInputs, compute

        inputs = LossInputs(
            logits=logits,
            target_tokens=labels,
            weights=weights,
            old_logprobs=old_logprobs,
            advantages=advantages,
            loss_fn_config=loss_fn_config,
        )
        return compute(loss_fn, inputs)


def _move_optimizer_state_to_cpu(state: dict) -> dict:
    out: dict = {"state": {}, "param_groups": state.get("param_groups", [])}
    for k, v in state.get("state", {}).items():
        if isinstance(v, dict):
            out["state"][k] = {
                kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()
            }
        else:
            out["state"][k] = v
    return out


# ─── Axolotl-backed trainer (stub) ────────────────────────────────────────


class AxolotlTrainer:
    """Stub Trainer that delegates to the Axolotl training framework.

    Not implemented yet — but the interface is fixed so the worker
    doesn't need to change when we swap it in.  Plan:

    * Construct an Axolotl ``AxolotlTrainingArguments`` from our
      ``ParallelConfig``. Axolotl already maps FSDP2/TP/CP/sequence
      parallel / CPU offloading / sample packing from its config.yml,
      so we just translate our knobs into its schema.
    * Use Axolotl's model loader to get a PEFT-wrapped model that's
      already sharded, optionally with sample packing and multipack.
    * Route ``forward_backward`` / ``optim_step`` / ``sample`` through
      Axolotl's trainer loop (or a trimmed-down step function that
      accepts a single microbatch at a time).
    * Reuse VanillaTrainer's state serialization — LoRA weights, grad
      accumulators, and optimizer state are still plain tensor dicts.

    See https://github.com/axolotl-ai-cloud/axolotl for upstream.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "AxolotlTrainer is not implemented yet — use VanillaTrainer "
            "for now. Track https://github.com/axolotl-ai-cloud/axolotl for "
            "the Trainer-protocol integration."
        )
