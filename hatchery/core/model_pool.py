# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Per-worker pool of base models.

Two implementations:

:class:`RewrapModelPool`
    Single-tier VRAM pool. Cold miss pays the full HF loader cost.
    Eviction destroys the wrapper and lets Python GC reclaim VRAM;
    reload goes back through the HF cache via ``from_pretrained``.
    The simplest correct implementation; unblocks multi-model
    workers today.

:class:`TieredModelPool`
    Two-tier: VRAM + pinned host RAM. When a VRAM slot evicts, its
    weights are demoted to pinned CPU memory instead of destroyed.
    Reload from host is ``.to(device, non_blocking=True)`` — a
    single DMA at PCIe speed, skipping the HF loader entirely.
    The disk tier is left to HF's own cache — its content-addressed
    ``blobs/`` + ``snapshots/`` layout already deduplicates weights
    across revisions, so there is nothing to gain by reimplementing
    it.

Switching between pools is one env var (``HATCHERY_MODEL_POOL``) with
no code changes.

LoRA heterogeneity
------------------
Only the *frozen base model* is pooled. LoRA adapters are per-session
and attached to whichever pool slot holds that session's base. A
session with rank=8 and another with rank=64 can live on the same
pool slot simultaneously — PEFT's ``add_adapter`` handles different
ranks independently. On eviction, all adapters hosted by the evicted
slot are torn down with the slot and reloaded on the next access.

See also
--------
:class:`hatchery.core.trainer.VanillaTrainer` — currently loads one
hard-coded base model at construction. The pool is the abstraction
that replaces that single model with a collection.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from hatchery.core.quantization import (
    QuantConfig,
    prepare_onebit_loader_kwargs,
    resolve_quant_scheme,
)

# ─── VLM capability detection ──────────────────────────────────────────
#
# These live with the pool rather than the worker because "is this a
# vision-language model?" and "which tokens need stripping from text-only
# inputs?" are facts about a freshly loaded base — the pool owns base
# loading, so it owns these derived facts too. worker.py re-exports the
# helpers to preserve existing import paths.

_VLM_CLASS_NAMES = {
    "Qwen2VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "InternVLChatModel",
}


def _is_vlm_model(model: Any) -> bool:
    """Check if a model is a vision-language model by class name."""
    cls_name = type(model).__name__
    return cls_name in _VLM_CLASS_NAMES


def _get_vision_token_ids(tokenizer: Any) -> set[int]:
    """Collect all special vision/image token IDs from the tokenizer.

    VLM tokenizers define placeholder tokens like ``<image>``,
    ``<|vision_start|>``, ``<|image_pad|>``, ``<|vision_end|>`` that
    must only appear when corresponding ``pixel_values`` are present.
    If text-only input accidentally contains these tokens (e.g., from
    a chat template that always inserts them), the model's vision
    encoder crashes with missing embeddings.
    """
    vision_prefixes = ("image", "vision", "img", "visual", "pic")
    ids: set[int] = set()
    vocab = getattr(tokenizer, "get_vocab", None)
    if vocab is None:
        return ids
    for tok_str, tok_id in vocab().items():
        tok_lower = tok_str.lower().strip("<>|")
        if any(tok_lower.startswith(p) or p in tok_lower for p in vision_prefixes):
            ids.add(tok_id)
    return ids


@dataclass
class PoolSlot:
    """One base model resident on the worker.

    The pool owns load-time concerns: the raw base weights, the
    tokenizer, an optional VLM processor, and derived capability flags.
    Adapter/PEFT wiring is managed by the worker.

    ``peft_model`` is optional because the pool can hold a raw base
    model before PEFT has wrapped it. Once the first session attaches
    an adapter, ``peft_model`` is set and all subsequent adapters use
    the same wrapper.

    ``host_state_dict`` is populated by tiered pools when a slot is
    demoted from VRAM to pinned host RAM. In the VRAM tier it stays
    ``None``; the live weights are on ``raw_base``.
    """

    base_model_name: str
    raw_base: Any
    peft_model: Any = None
    adapters: set[str] = field(default_factory=set)
    last_touched: float = field(default_factory=time.time)
    load_time_s: Optional[float] = None
    parallel_applied: bool = False
    precision_applied: bool = False
    # Load-time concerns (populated by the pool).
    tokenizer: Any = None
    processor: Any = None
    is_vlm: bool = False
    vision_token_ids: set[int] = field(default_factory=set)
    # Tier state (populated only by tiered pools).
    host_state_dict: Optional[dict] = None
    # Quantization scheme detected / forced at load time. ``"none"``
    # for ordinary full-precision checkpoints; ``"onebit"`` for
    # BitNet-family ternary master-weight bases. Mirrors
    # :class:`hatchery.core.quantization.QuantConfig.scheme`.
    quant_scheme: str = "none"
    # Mixed-mode (LoRA + full-param) bookkeeping. ``pristine_sd`` is the
    # CPU snapshot of the base weights captured the first time a
    # full-param session attaches to this slot — every LoRA session
    # expects the base to be at this snapshot, so it's the "ground
    # truth" we restore to when switching modes. ``active_session_id``
    # records which session's weights are currently materialized into
    # ``raw_base`` so the worker can skip a redundant swap.
    pristine_sd: Optional[dict] = None
    active_session_id: Optional[str] = None


class ModelPool(Protocol):
    """Contract for per-worker base-model pools.

    Implementations manage the mapping from ``base_model_name`` to a
    live :class:`PoolSlot`. ``get_or_load`` is the one hot-path
    method; ``loaded_models`` is used by the queue to prefer jobs for
    models already resident.
    """

    def loaded_models(self) -> list[str]: ...
    def get(self, base_model_name: str) -> Optional[PoolSlot]: ...
    def get_or_load(self, base_model_name: str) -> PoolSlot: ...
    def evict(self, base_model_name: str) -> None: ...
    def evict_lru(self) -> Optional[str]: ...
    def size(self) -> int: ...


# ─── Rewrap implementation ──────────────────────────────────────────────


class RewrapModelPool:
    """Freeze-and-rewrap pool. LRU eviction. No caching.

    Parameters
    ----------
    max_slots:
        Maximum number of base models resident in VRAM at once.
        Typical: 1-2 for large models, up to 8-10 for <1B models.
    device:
        Torch device string passed to ``.to(device)`` after loading.
    dtype:
        Torch dtype for the loaded base model. ``None`` = use the
        model's native dtype from the HF config.
    attn_implementation:
        Transformers attention backend.
    loader:
        Optional hook for tests — ``(name) -> raw_base``. When
        supplied, we don't touch transformers / HF cache at all.
    """

    def __init__(
        self,
        *,
        max_slots: int = 1,
        device: str = "cuda:0",
        dtype: Any = None,
        attn_implementation: str = "sdpa",
        loader: Optional[Any] = None,
        tokenizer_loader: Optional[Any] = None,
        parallel: Any = None,
        quant_config: Optional[QuantConfig] = None,
    ) -> None:
        if max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {max_slots}")
        self.max_slots = max_slots
        self.device = device
        self.dtype = dtype
        # Sequence packing v1 requires HF's flash-attn-2 path — SDPA
        # silently ignores ``position_ids`` resets so attention would
        # leak across document boundaries. Upgrade here at pool
        # construction; if flash-attn-2 can't resolve at load time,
        # HF raises and the worker surfaces a clear error.
        if parallel is not None and getattr(parallel, "sequence_packing", False):
            attn_implementation = "flash_attention_2"
        self.attn_implementation = attn_implementation
        # When ``loader`` is injected (tests), ``tokenizer_loader`` is
        # the matching hook for tokenizer/processor loading. Tests that
        # don't care about tokenizers can leave it ``None``, in which
        # case the slot's tokenizer/processor/is_vlm stay at defaults.
        # In the real (production) path, both are None and the pool
        # calls into ``transformers`` directly.
        self._loader = loader
        self._tokenizer_loader = tokenizer_loader
        # Caller-supplied quantization intent. ``None`` means "use the
        # default ``QuantConfig()`` (auto-detect, no force)". We keep
        # it as-is so callers can tell whether a config was explicit.
        self._quant_config = quant_config
        self._slots: OrderedDict[str, PoolSlot] = OrderedDict()

    def size(self) -> int:
        return len(self._slots)

    def loaded_models(self) -> list[str]:
        return list(self._slots.keys())

    def get(self, base_model_name: str) -> Optional[PoolSlot]:
        slot = self._slots.get(base_model_name)
        if slot is not None:
            slot.last_touched = time.time()
            self._slots.move_to_end(base_model_name)
        return slot

    def get_or_load(self, base_model_name: str) -> PoolSlot:
        slot = self.get(base_model_name)
        if slot is not None:
            return slot

        # Evict if at capacity — do this BEFORE loading the new base
        # so we don't transiently OOM while two models are resident.
        while len(self._slots) >= self.max_slots:
            evicted = self.evict_lru()
            if evicted is None:
                break

        t0 = time.time()
        raw, scheme = self._load_base_with_scheme(base_model_name)
        load_time = time.time() - t0

        tokenizer, processor, is_vlm, vision_ids = self._load_tokenizer_and_processor(
            base_model_name, raw
        )

        slot = PoolSlot(
            base_model_name=base_model_name,
            raw_base=raw,
            load_time_s=load_time,
            tokenizer=tokenizer,
            processor=processor,
            is_vlm=is_vlm,
            vision_token_ids=vision_ids,
            quant_scheme=scheme,
        )
        self._slots[base_model_name] = slot
        return slot

    def _load_base_with_scheme(self, base_model_name: str) -> tuple[Any, str]:
        """Dispatch to :meth:`_load_base` plus detection of the quant scheme.

        Returns ``(raw_model, scheme)`` where ``scheme`` is one of
        ``"none"`` / ``"onebit"``. The scheme string is persisted on
        the slot so the worker (and later layers like PEFT wrapping)
        can key off it without re-running detection.
        """
        raw = self._load_base(base_model_name)
        scheme = _detect_scheme_on_model(raw, base_model_name, self._quant_config)
        return raw, scheme

    def _load_base(self, base_model_name: str) -> Any:
        if self._loader is not None:
            return self._loader(base_model_name)
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        kwargs = _maybe_adjust_kwargs_for_quant(base_model_name, kwargs, self._quant_config)
        # HF's fine-grained FP8 (FP8Linear) kernels are inference-only — no
        # backward pass — so LoRA can't train through them (autograd needs
        # grad-w.r.t.-input even with the base frozen). Dequantize the on-disk
        # FP8 weights back to dtype at load so LoRA attaches over plain bf16
        # linears. To keep the base in FP8 *and* train LoRA, use the
        # autograd-compatible TorchAO path instead (scheme="fp8_torchao"), or
        # opt into re-quantizing this dequantized base to TorchAO FP8 via
        # HATCHERY_FP8_REQUANT_TORCHAO (see _maybe_requant_finegrained_fp8_to_torchao).
        was_finegrained_fp8 = False
        try:
            from transformers import AutoConfig

            _cfg = AutoConfig.from_pretrained(base_model_name)
            qc = getattr(_cfg, "quantization_config", None)
            if isinstance(qc, dict) and qc.get("quant_method") == "fp8":
                was_finegrained_fp8 = True
                if not qc.get("dequantize"):
                    _cfg.quantization_config = {**qc, "dequantize": True}
                    kwargs["config"] = _cfg
        except Exception:
            pass

        try:
            raw = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        except ValueError:
            raw = _load_image_text_to_text_as_causal_lm(base_model_name, kwargs)
        if was_finegrained_fp8 and self.dtype is not None:
            # FineGrainedFP8 dequantize restores the affected layers in fp32
            # regardless of torch_dtype, leaving the base mixed-precision (fp32
            # quantized layers vs dtype everywhere else) and breaking matmuls.
            # Cast back to the compute dtype so the base is consistent (and so an
            # optional requant below quantizes from dtype weights, not fp32).
            raw = raw.to(self.dtype)
        if self.device:
            raw = raw.to(self.device)
        raw = _maybe_requant_finegrained_fp8_to_torchao(raw, was_finegrained_fp8)
        raw.gradient_checkpointing_enable()
        raw.enable_input_require_grads()
        raw.eval()
        for p in raw.parameters():
            p.requires_grad = False
        return raw

    def _load_tokenizer_and_processor(
        self, base_model_name: str, raw_base: Any
    ) -> tuple[Any, Any, bool, set[int]]:
        """Return ``(tokenizer, processor, is_vlm, vision_token_ids)``.

        Tests that inject a ``loader`` but no ``tokenizer_loader`` get
        all-defaults, since they usually don't care. Production path
        (no loader hooks) goes through ``transformers``.
        """
        if self._tokenizer_loader is not None:
            return self._tokenizer_loader(base_model_name, raw_base)
        if self._loader is not None:
            # Test path with a model loader but no tokenizer loader —
            # leave tokenizer/processor unset rather than hit HF.
            return None, None, False, set()

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        is_vlm = _is_vlm_model(raw_base)
        processor: Any = None
        vision_ids: set[int] = set()
        if is_vlm:
            try:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(base_model_name)
                vision_ids = _get_vision_token_ids(tokenizer)
            except Exception:  # noqa: BLE001
                # Best-effort — if the processor won't load, fall back
                # to text-only treatment rather than failing the load.
                is_vlm = False
                processor = None
                vision_ids = set()
        return tokenizer, processor, is_vlm, vision_ids

    def evict(self, base_model_name: str) -> None:
        slot = self._slots.pop(base_model_name, None)
        if slot is None:
            return
        self._teardown_slot(slot)

    def evict_lru(self) -> Optional[str]:
        if not self._slots:
            return None
        name, slot = self._slots.popitem(last=False)
        self._teardown_slot(slot)
        return name

    def _teardown_slot(self, slot: PoolSlot) -> None:
        # Drop references so Python GC can reclaim. The caller is
        # responsible for tearing down any attached PEFT adapters on
        # ``slot.peft_model`` before calling evict.
        slot.raw_base = None
        slot.peft_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class TieredModelPool:
    """Two-tier base-model pool: VRAM + pinned host RAM.

    A slot can be in one of two tiers:

    * **VRAM** — ``raw_base`` lives on ``self.device`` and is usable
      immediately. Up to ``max_vram_slots`` resident at once.
    * **Host** — ``raw_base`` lives on pinned CPU memory. Up to
      ``max_host_slots`` resident; use the pool as a promote-on-demand
      cache to avoid re-paying the HF loader cost. With
      ``max_host_slots=0``, the pool behaves identically to
      :class:`RewrapModelPool` — evicted VRAM slots are torn down.

    Residency contract
    ------------------
    :meth:`get` and :meth:`loaded_models` report only the VRAM tier —
    that's the "hot" signal for schedulers/queues. :meth:`get_or_load`
    promotes a host-resident slot back to VRAM transparently.

    Eviction cascade
    ----------------
    Need to admit a new model to VRAM but VRAM is full:
        pop LRU VRAM slot. If host has room, demote it; otherwise
        tear it down.
    Need to admit a demoted slot to host but host is full:
        pop LRU host slot. Tear it down.

    Hooks (tests / custom transfer impls)
    -------------------------------------
    ``demote_hook(slot)`` is called when a slot transitions VRAM→host.
    Default implementation copies each parameter to pinned CPU memory
    via ``torch.Tensor.pin_memory()`` and swaps ``.data`` in place.
    ``promote_hook(slot)`` handles host→VRAM. Default uses
    ``raw_base.to(device, non_blocking=True)``.
    Tests inject no-op hooks to avoid touching torch.
    """

    def __init__(
        self,
        *,
        max_vram_slots: int = 1,
        max_host_slots: int = 0,
        device: str = "cuda:0",
        dtype: Any = None,
        attn_implementation: str = "sdpa",
        loader: Optional[Any] = None,
        tokenizer_loader: Optional[Any] = None,
        demote_hook: Optional[Any] = None,
        promote_hook: Optional[Any] = None,
        parallel: Any = None,
        quant_config: Optional[QuantConfig] = None,
    ) -> None:
        if max_vram_slots < 1:
            raise ValueError(f"max_vram_slots must be >= 1, got {max_vram_slots}")
        if max_host_slots < 0:
            raise ValueError(f"max_host_slots must be >= 0, got {max_host_slots}")
        self.max_vram_slots = max_vram_slots
        self.max_host_slots = max_host_slots
        self.device = device
        self.dtype = dtype
        # See RewrapModelPool.__init__ for rationale.
        if parallel is not None and getattr(parallel, "sequence_packing", False):
            attn_implementation = "flash_attention_2"
        self.attn_implementation = attn_implementation
        self._loader = loader
        self._tokenizer_loader = tokenizer_loader
        self._demote_hook = demote_hook
        self._promote_hook = promote_hook
        self._quant_config = quant_config
        self._vram: OrderedDict[str, PoolSlot] = OrderedDict()
        self._host: OrderedDict[str, PoolSlot] = OrderedDict()

    # ── ModelPool protocol ────────────────────────────────────────

    def size(self) -> int:
        """Number of VRAM-resident models (the 'hot' count)."""
        return len(self._vram)

    def loaded_models(self) -> list[str]:
        """VRAM-resident names only (the scheduler routing signal)."""
        return list(self._vram.keys())

    def get(self, base_model_name: str) -> Optional[PoolSlot]:
        slot = self._vram.get(base_model_name)
        if slot is not None:
            slot.last_touched = time.time()
            self._vram.move_to_end(base_model_name)
        return slot

    def get_or_load(self, base_model_name: str) -> PoolSlot:
        # VRAM hit.
        slot = self.get(base_model_name)
        if slot is not None:
            return slot

        # Host hit — promote back to VRAM.
        host_slot = self._host.pop(base_model_name, None)
        if host_slot is not None:
            self._make_vram_room(reserve=1)
            self._promote(host_slot)
            host_slot.last_touched = time.time()
            self._vram[base_model_name] = host_slot
            return host_slot

        # Cold — load fresh.
        self._make_vram_room(reserve=1)
        t0 = time.time()
        raw = self._load_base(base_model_name)
        scheme = _detect_scheme_on_model(raw, base_model_name, self._quant_config)
        load_time = time.time() - t0

        tokenizer, processor, is_vlm, vision_ids = self._load_tokenizer_and_processor(
            base_model_name, raw
        )

        slot = PoolSlot(
            base_model_name=base_model_name,
            raw_base=raw,
            load_time_s=load_time,
            tokenizer=tokenizer,
            processor=processor,
            is_vlm=is_vlm,
            vision_token_ids=vision_ids,
            quant_scheme=scheme,
        )
        self._vram[base_model_name] = slot
        return slot

    def evict(self, base_model_name: str) -> None:
        """Remove a model from both tiers, tearing it down."""
        vram_slot = self._vram.pop(base_model_name, None)
        if vram_slot is not None:
            self._teardown_slot(vram_slot)
        host_slot = self._host.pop(base_model_name, None)
        if host_slot is not None:
            self._teardown_slot(host_slot)

    def evict_lru(self) -> Optional[str]:
        """Evict the LRU VRAM slot. If host has room, demote; else teardown.

        Used by callers that want to free GPU memory without caring
        about preserving host-cached copies of other models.
        """
        if not self._vram:
            return None
        name, slot = self._vram.popitem(last=False)
        self._demote_or_destroy(name, slot)
        return name

    # ── Introspection (not part of ModelPool protocol) ────────────

    def host_resident_models(self) -> list[str]:
        return list(self._host.keys())

    def host_size(self) -> int:
        return len(self._host)

    # ── Internals ─────────────────────────────────────────────────

    def _make_vram_room(self, reserve: int) -> None:
        """Ensure the VRAM tier has room for ``reserve`` more slots."""
        while len(self._vram) + reserve > self.max_vram_slots:
            name, slot = self._vram.popitem(last=False)
            self._demote_or_destroy(name, slot)

    def _demote_or_destroy(self, name: str, slot: PoolSlot) -> None:
        if self.max_host_slots <= 0:
            self._teardown_slot(slot)
            return
        # Make room in host tier first.
        while len(self._host) >= self.max_host_slots:
            _ename, evicted = self._host.popitem(last=False)
            self._teardown_slot(evicted)
        self._demote(slot)
        self._host[name] = slot

    def _demote(self, slot: PoolSlot) -> None:
        if self._demote_hook is not None:
            self._demote_hook(slot)
            return
        self._default_demote(slot)

    def _promote(self, slot: PoolSlot) -> None:
        if self._promote_hook is not None:
            self._promote_hook(slot)
            return
        self._default_promote(slot)

    def _default_demote(self, slot: PoolSlot) -> None:
        """Move weights from GPU to pinned CPU in place."""
        raw = slot.raw_base
        if raw is None:
            return
        try:
            import torch
        except ImportError:
            return
        # Swap each parameter's and buffer's underlying storage to a
        # pinned CPU tensor. Keeping the module object alive avoids
        # paying the HF constructor cost on promotion.
        with torch.no_grad():
            for p in raw.parameters():
                if p.device.type == "cpu":
                    continue
                cpu = p.data.to("cpu", non_blocking=False)
                try:
                    cpu = cpu.pin_memory()
                except (RuntimeError, AssertionError):
                    # Pinned memory can fail in some environments
                    # (e.g. no CUDA, container ulimits); fall back to
                    # unpinned CPU tensors which still work, just with
                    # a slower promote.
                    pass
                p.data = cpu
            for b in raw.buffers():
                if b.device.type == "cpu":
                    continue
                cpu = b.data.to("cpu", non_blocking=False)
                try:
                    cpu = cpu.pin_memory()
                except (RuntimeError, AssertionError):
                    pass
                b.data = cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _default_promote(self, slot: PoolSlot) -> None:
        raw = slot.raw_base
        if raw is None:
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            return
        slot.raw_base = raw.to(self.device, non_blocking=True)

    def _load_base(self, base_model_name: str) -> Any:
        # Same as RewrapModelPool — kept local so TieredModelPool
        # doesn't inherit, since the two classes diverge beyond this.
        if self._loader is not None:
            return self._loader(base_model_name)
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        kwargs = _maybe_adjust_kwargs_for_quant(base_model_name, kwargs, self._quant_config)
        raw = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        if self.device:
            raw = raw.to(self.device)
        raw.gradient_checkpointing_enable()
        raw.enable_input_require_grads()
        raw.eval()
        for p in raw.parameters():
            p.requires_grad = False
        return raw

    def _load_tokenizer_and_processor(
        self, base_model_name: str, raw_base: Any
    ) -> tuple[Any, Any, bool, set[int]]:
        if self._tokenizer_loader is not None:
            return self._tokenizer_loader(base_model_name, raw_base)
        if self._loader is not None:
            return None, None, False, set()

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        is_vlm = _is_vlm_model(raw_base)
        processor: Any = None
        vision_ids: set[int] = set()
        if is_vlm:
            try:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(base_model_name)
                vision_ids = _get_vision_token_ids(tokenizer)
            except Exception:  # noqa: BLE001
                is_vlm = False
                processor = None
                vision_ids = set()
        return tokenizer, processor, is_vlm, vision_ids

    def _teardown_slot(self, slot: PoolSlot) -> None:
        slot.raw_base = None
        slot.peft_model = None
        slot.host_state_dict = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _detect_scheme_on_model(
    raw_model: Any,
    base_model_name: str,
    quant_config: Optional[QuantConfig],
) -> str:
    """Pick a quantization scheme string for a freshly loaded model.

    Reads the model's ``.config`` (duck-typed — works for HF
    ``PretrainedConfig`` as well as the trivial ``_FakeBase`` stand-in
    used in unit tests) and combines it with the caller's
    :class:`QuantConfig` via
    :func:`hatchery.core.quantization.resolve_quant_scheme`. When the
    model has no ``.config`` (test fakes), we fall back to the
    requested scheme or ``"none"``.
    """
    hf_config = getattr(raw_model, "config", None)
    if hf_config is None:
        if quant_config is None:
            return "none"
        return quant_config.scheme
    return resolve_quant_scheme(hf_config, model_name=base_model_name, requested=quant_config)


def _maybe_adjust_kwargs_for_quant(
    base_model_name: str,
    kwargs: dict[str, Any],
    quant_config: Optional[QuantConfig],
) -> dict[str, Any]:
    """Peek at the HF config before loading to adjust ``from_pretrained`` kwargs.

    We pay the ``AutoConfig.from_pretrained`` round-trip only when
    quant-aware handling could matter — either the caller asked for
    it explicitly (``quant_config`` is set) or the model name
    *itself* hints at a 1-bit checkpoint. For the common full-
    precision path we return ``kwargs`` unchanged, avoiding an extra
    HTTP / cache read per load.

    FP8 TorchAO (``quant_config.is_fp8_torchao``) is handled first and
    bypasses the AutoConfig round-trip — we inject ``TorchAoConfig``
    directly from the caller's explicit request.

    Note: checkpoints already saved with a TorchAO FP8 ``quantization_config``
    in ``config.json`` are handled automatically by HuggingFace
    Transformers during ``from_pretrained`` without any injection here.
    This FP8 branch covers the *apply-at-load-time* case only (explicit
    ``scheme="fp8_torchao"``).
    """
    from hatchery.core.quantization import is_onebit_by_name, prepare_fp8_torchao_loader_kwargs

    # FP8 is always an explicit caller request — no need to consult AutoConfig.
    if quant_config is not None and quant_config.is_fp8_torchao:
        return prepare_fp8_torchao_loader_kwargs(kwargs, fp8_mode=quant_config.fp8_mode)

    want_check = quant_config is not None and (quant_config.force or quant_config.is_onebit)
    if not want_check and not is_onebit_by_name(base_model_name):
        return kwargs

    try:
        from transformers import AutoConfig
    except ImportError:  # pragma: no cover
        return kwargs
    try:
        hf_cfg = AutoConfig.from_pretrained(base_model_name)
    except Exception:  # noqa: BLE001
        return kwargs

    scheme = resolve_quant_scheme(hf_cfg, model_name=base_model_name, requested=quant_config)
    if scheme == "onebit":
        return prepare_onebit_loader_kwargs(kwargs)
    return kwargs


def _maybe_requant_finegrained_fp8_to_torchao(raw: Any, was_finegrained_fp8: bool) -> Any:
    """Opt-in: re-quantize a dequantized fine-grained-FP8 base to TorchAO FP8.

    HF fine-grained FP8 checkpoints can't be LoRA-trained in place (``FP8Linear``
    is inference-only), so :meth:`RewrapModelPool._load_base` dequantizes them to
    bf16 — which gives up the FP8 memory footprint. When
    ``HATCHERY_FP8_REQUANT_TORCHAO`` is truthy, re-quantize the live bf16 module to
    TorchAO float8 (autograd-compatible, so LoRA still trains) to claw that back.

    Opt-in, not automatic, because of the trade-offs:

    * Numerics diverge — this is a *fresh* tensorwise float8 quantization of the
      dequantized bf16 weights, not the checkpoint's original blockwise fine-grained
      FP8.
    * No load-peak savings — the full bf16 base is materialized before requant;
      only steady-state memory drops (~half).
    * Requires FP8-capable hardware (Ada/Hopper/Blackwell, CUDA capability >= 8.9).

    The output embedding (``lm_head``) is deliberately left unquantized: fp8 on the
    vocab projection wrecks training (loss diverges), and it must also stay a plain
    ``nn.Linear`` so the bf16 logits path / fused loss works. LoRA over the float8
    layers must be kept in ``dtype`` (not fp32) by the caller; see
    :meth:`hatchery.core.worker.GPUWorker._attach_adapter`.

    Best-effort: any failure (missing torchao, wrong hardware, kernel error) falls
    back to the already-loaded bf16 module rather than erroring the load.
    """
    if not was_finegrained_fp8:
        return raw
    if os.environ.get("HATCHERY_FP8_REQUANT_TORCHAO", "").lower() not in {"1", "true", "yes", "on"}:
        return raw
    try:
        import torch

        if not torch.cuda.is_available():
            return raw
        try:
            dev = next(raw.parameters()).device
        except StopIteration:
            return raw
        if dev.type != "cuda" or torch.cuda.get_device_capability(dev) < (8, 9):
            return raw
        from torchao.quantization import Float8WeightOnlyConfig, quantize_

        # Quantize Linear layers only, excluding the output embedding (lm_head).
        out_emb = None
        try:
            out_emb = raw.get_output_embeddings()
        except Exception:
            out_emb = None

        def _is_quantizable_linear(module: Any, _fqn: str) -> bool:
            return isinstance(module, torch.nn.Linear) and module is not out_emb

        quantize_(raw, Float8WeightOnlyConfig(), filter_fn=_is_quantizable_linear)
    except Exception:
        # Keep the bf16 base; requant is a best-effort memory optimization.
        return raw
    return raw


def _load_image_text_to_text_as_causal_lm(base_model_name: str, kwargs: dict[str, Any]) -> Any:
    """Fallback for text-only use of multimodal wrapper configs.

    Some VLM configs are not registered in ``AutoModelForCausalLM`` even though their
    conditional-generation class can serve text-only calls when no image inputs are
    provided.
    """
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        raise
    return AutoModelForImageTextToText.from_pretrained(base_model_name, **kwargs)


def build_default_model_pool(
    *,
    max_slots: Optional[int] = None,
    device: str = "cuda:0",
    quant_config: Optional[QuantConfig] = None,
    **kwargs: Any,
) -> ModelPool:
    """Factory that picks a pool implementation from the environment.

    Env vars:

    * ``HATCHERY_MODEL_POOL`` — ``"rewrap"`` (default) or ``"tiered"``.
    * ``HATCHERY_MODEL_POOL_MAX_SLOTS`` — VRAM slots, default 1.
    * ``HATCHERY_MODEL_POOL_MAX_HOST_SLOTS`` — host-tier slots (only
      honored when ``HATCHERY_MODEL_POOL=tiered``), default 0.

    ``max_slots`` (if passed) overrides the env var for VRAM slots.
    ``quant_config`` (if passed) is forwarded to the pool constructor
    so load-time quantization routing (e.g. BitNet 1.58-bit) applies
    uniformly across both pool implementations.
    """
    slots = (
        max_slots
        if max_slots is not None
        else int(os.environ.get("HATCHERY_MODEL_POOL_MAX_SLOTS", "1"))
    )
    kind = os.environ.get("HATCHERY_MODEL_POOL", "rewrap").lower()
    if kind == "tiered":
        host_slots = int(os.environ.get("HATCHERY_MODEL_POOL_MAX_HOST_SLOTS", "0"))
        return TieredModelPool(
            max_vram_slots=slots,
            max_host_slots=host_slots,
            device=device,
            quant_config=quant_config,
            **kwargs,
        )
    return RewrapModelPool(max_slots=slots, device=device, quant_config=quant_config, **kwargs)
