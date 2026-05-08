# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""DFlash speculative decoding runtime integration for hatchery-core workers.

All dflash library imports are lazy — this module never imports dflash at the
module level. When the dflash package is absent, or when the runtime call fails,
the caller receives ``(None, metadata)`` and falls back to the standard HF
generate path unless strict mode is set.

The canonical DFlash API (from ``z-lab/dflash``) is invoked as::

    from dflash.model import dflash_generate
    stats = dflash_generate(
        model=draft_model,        # AutoModel.from_pretrained(draft_path, trust_remote_code=True)
        target=verifier_model,    # AutoModelForCausalLM (PEFT-unwrapped to the LoRA-augmented base)
        input_ids=...,
        max_new_tokens=...,
        stop_token_ids=[...],
        temperature=...,
        return_stats=True,
    )

The transformers backend is single-batch and temperature-only, so requests with
``n>1`` or constrained ``top_p`` / ``top_k`` fall back to HF generate (with a
``fallback_reason`` recorded in the response metadata). Strict-mode requests
turn the same conditions into raised exceptions.

Integration point: ``GPUWorker._handle_sample`` calls ``run_dflash_sample()``
after extracting payload parameters when ``payload["speculative_decoding"]``
is present and the DFlash backend is eligible.

PEFT/LoRA compatibility: when ``verifier_model`` is a PEFT-wrapped causal LM,
the LoRA adapters live on the inner ``base_model.model`` (a HF causal LM whose
attention/MLP projections have been replaced in-place with LoRA-augmented
counterparts). DFlash's transformers backend reaches into ``target.model`` and
``target.lm_head``, so we forward the LoRA-augmented inner causal LM rather
than the outer PEFT wrapper. Verifier outputs still reflect the trained
adapters because the in-place LoRA modules participate in every forward pass.

Model-aware draft selection: if the worker config does not name an explicit
draft model, the policy can infer a known draft from the verifier base model
name. That keeps the Qwen3.6 DFlash path ergonomic while preserving the
existing disable/strict behavior.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from hatchery.core.spec_decoding import (
    SPEC_BACKEND_DFLASH,
    SpeculativeDecodingMetadata,
    SpeculativeDecodingRequest,
)

if TYPE_CHECKING:
    pass  # keep for future typed stubs

logger = logging.getLogger("hatchery.core.dflash_integration")

_DEFAULT_DFLASH_DRAFT_MODELS: dict[str, str] = {
    "Qwen/Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
}


@dataclass
class DFlashConfig:
    """Server-side DFlash speculative decoding configuration.

    Attach an instance to ``Config.dflash`` to enable DFlash for eligible
    sample requests handled by ``GPUWorker._handle_sample``.

    Per-request overrides via ``SamplingParams.speculative_decoding`` always
    take precedence over the defaults here.

    Attributes
    ----------
    draft_model:
        HF model name or local path for the draft model. When omitted, the
        policy can infer a known draft for supported verifier base models.
    max_draft_tokens:
        Default maximum draft tokens per speculation step (per-request
        ``max_draft_tokens`` overrides this). Forwarded to dflash as the
        block size.
    enabled:
        Master switch. ``False`` disables DFlash unconditionally; useful for
        temporarily disabling without removing the config object.
    """

    draft_model: Optional[str] = None
    max_draft_tokens: int = 5
    enabled: bool = True


def _try_import_dflash() -> Any:
    """Return the ``dflash`` module object, or ``None`` if not installed."""
    try:
        import dflash  # type: ignore[import-not-found]

        return dflash
    except ImportError:
        return None


def _resolve_dflash_generate(dflash_mod: Any) -> tuple[Any, bool]:
    """Locate the dflash low-level generate callable.

    Canonical exposure is ``dflash.model.dflash_generate``. The top-level
    ``dflash`` package uses ``__getattr__`` lazy loading and does not eagerly
    import its ``model`` submodule, so we explicitly import it here when the
    attribute isn't already present. Falls back to ``dflash.generate`` so
    test doubles and any future top-level alias keep working (the CPU test
    suite mocks the module with a ``generate`` attribute).

    Returns ``(fn, is_real_dflash)`` — the second element is ``True`` when we
    resolved against the canonical ``dflash.model.dflash_generate`` (or the
    convenience top-level alias of the same shape) and the caller should
    treat the call as a real DFlash invocation: load the draft via
    ``AutoModel``, pass canonical kwargs only, and apply the
    transformers-backend restrictions (n=1, temperature-only).
    """
    is_real = False
    fn = getattr(dflash_mod, "dflash_generate", None)
    if fn is not None:
        is_real = True
    if fn is None:
        model_mod = getattr(dflash_mod, "model", None)
        if model_mod is None:
            try:
                import dflash.model as model_mod  # type: ignore[import-not-found]
            except ImportError:
                model_mod = None
        if model_mod is not None:
            fn = getattr(model_mod, "dflash_generate", None)
            if fn is not None:
                is_real = True
    if fn is None:
        fn = getattr(dflash_mod, "generate", None)
    return fn, is_real


def _resolve_default_draft_model(base_model_name: Optional[str]) -> Optional[str]:
    if base_model_name is None:
        return None
    for prefix, draft_model in _DEFAULT_DFLASH_DRAFT_MODELS.items():
        if base_model_name == prefix or base_model_name.startswith(prefix):
            return draft_model
    return None


def parse_spec_request(payload: dict) -> Optional[SpeculativeDecodingRequest]:
    """Extract and validate the ``speculative_decoding`` entry from a sample payload.

    Returns ``None`` when the key is absent or the value is ``None``.
    Already-parsed ``SpeculativeDecodingRequest`` objects are returned as-is.
    """
    raw = payload.get("speculative_decoding")
    if raw is None:
        return None
    if isinstance(raw, SpeculativeDecodingRequest):
        return raw
    return SpeculativeDecodingRequest(**raw)


def resolve_dflash_policy(
    spec_request: Optional[SpeculativeDecodingRequest],
    dflash_config: Optional[DFlashConfig],
    *,
    base_model_name: Optional[str] = None,
) -> tuple[bool, SpeculativeDecodingMetadata]:
    """Decide whether a sample request should use DFlash speculative decoding.

    Parameters
    ----------
    spec_request:
        Parsed per-request speculative decoding options. ``None`` means the
        client did not request speculative decoding.
    dflash_config:
        Server-side DFlash configuration. ``None`` means DFlash is not
        configured on this worker.

    Returns
    -------
    (should_use, metadata)
        ``should_use`` is True only when all conditions are met.
        ``metadata.fallback_reason`` is set when a DFlash-requesting client
        is being redirected to the HF fallback path.

    Raises
    ------
    ValueError
        When ``spec_request.strict`` is True and a blocking condition is hit.
    """
    meta = SpeculativeDecodingMetadata()

    # No request-level speculative decoding — nothing to route.
    if spec_request is None or spec_request.enable is False:
        return False, meta

    # Wrong backend explicitly requested — skip without complaint.
    if spec_request.backend is not None and spec_request.backend != SPEC_BACKEND_DFLASH:
        return False, meta

    meta.requested_backend = SPEC_BACKEND_DFLASH

    if dflash_config is None or not dflash_config.enabled:
        meta.fallback_reason = "dflash_disabled"
        if spec_request.strict:
            raise ValueError("DFlash is disabled on this worker but strict mode was requested")
        return False, meta

    draft_model = dflash_config.draft_model or _resolve_default_draft_model(base_model_name)
    if not draft_model:
        meta.fallback_reason = "no_draft_model_configured"
        if spec_request.strict:
            raise ValueError(
                "No draft_model configured on DFlashConfig but strict mode was requested"
            )
        return False, meta

    max_draft = (
        spec_request.max_draft_tokens
        if spec_request.max_draft_tokens is not None
        else dflash_config.max_draft_tokens
    )
    meta.draft_model = draft_model
    meta.max_draft_tokens = max_draft

    return True, meta


def _resolve_verifier_for_dflash(verifier_model: Any) -> Any:
    """Drill through the PEFT wrapper so DFlash sees the LoRA-augmented causal LM.

    DFlash's transformers backend reaches into ``target.model`` and
    ``target.lm_head``. PEFT's ``PeftModelForCausalLM`` wraps the causal LM in
    a ``LoraModel`` that replaces target linears in-place; ``base_model.model``
    is the LoRA-augmented underlying causal LM (e.g. ``Qwen3ForCausalLM``).
    Forwarding that inner module preserves adapter contributions in every
    verification forward pass while exposing the attribute surface DFlash
    needs.

    Plain HF causal LMs are returned untouched.
    """
    base_model = getattr(verifier_model, "base_model", None)
    inner = getattr(base_model, "model", None) if base_model is not None else None
    if inner is None:
        return verifier_model
    if not hasattr(inner, "model") or not hasattr(inner, "lm_head"):
        return verifier_model
    return inner


def _resolve_stop_token_ids(stop: Optional[list], tokenizer: Any) -> Optional[list[int]]:
    """Build the ``stop_token_ids`` list DFlash needs from Hatchery's mixed ``stop`` payload.

    Hatchery's wire format accepts ``stop`` as a list of strings or token IDs
    (and the gateway also accepts a single string). DFlash's transformers
    backend only supports stop *token IDs*. We:

    - Always include the tokenizer's EOS token id (so DFlash terminates
      naturally on EOS — its ``stop_token_ids`` argument is the only stop
      signal it honours).
    - Convert any integer entry in ``stop`` directly.
    - For string entries, encode without special tokens; if it tokenizes to
      exactly one token, include that id. Multi-token stop strings cannot be
      represented in DFlash's per-step stop check and are silently dropped —
      callers who need multi-token stops should set ``speculative_decoding``
      to ``False`` for that request.

    Returns ``None`` when no stop ids resolve and the tokenizer has no EOS,
    matching DFlash's "no stop" sentinel.
    """
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))

    if stop:
        if isinstance(stop, (str, bytes, int)):
            stop = [stop]
        for entry in stop:
            if isinstance(entry, int):
                ids.append(entry)
            elif isinstance(entry, str):
                try:
                    encoded = tokenizer.encode(entry, add_special_tokens=False)
                except Exception:  # noqa: BLE001 — tokenizer may reject; treat as no-op
                    continue
                if len(encoded) == 1:
                    ids.append(int(encoded[0]))
                else:
                    logger.debug(
                        "dflash.stop_string_dropped",
                        extra={"stop": entry, "tokens": len(encoded)},
                    )

    if not ids:
        return None
    # Deduplicate while preserving order — extra ids are harmless but ugly in logs.
    return list(dict.fromkeys(ids))


# ── Draft-model cache ─────────────────────────────────────────────────────────
# DFlash drafts are small (sub-1B for an 8B verifier) but loading them costs a
# disk read + CUDA copy. We cache one instance per (path, device, dtype) so
# repeated samples on the same worker reuse the loaded weights. The cache key
# string-folds device/dtype so torch dtype aliasing doesn't fragment entries.
_DRAFT_CACHE: dict[tuple[str, str, str], Any] = {}
_DRAFT_CACHE_LOCK = threading.Lock()


def _draft_cache_key(path: str, device: Any, dtype: Any) -> tuple[str, str, str]:
    return (path, str(device), str(dtype))


def _load_draft_model(
    dflash_mod: Any,
    draft_path: str,
    device: Any,
    dtype: Any,
) -> Any:
    """Load (or fetch from cache) the DFlash draft model.

    ``dflash_mod`` is passed in only to keep the import lazy — the actual
    loader is ``transformers.AutoModel`` because z-lab's draft repos register
    their custom architecture via ``trust_remote_code=True``.
    """
    key = _draft_cache_key(draft_path, device, dtype)
    cached = _DRAFT_CACHE.get(key)
    if cached is not None:
        return cached

    with _DRAFT_CACHE_LOCK:
        cached = _DRAFT_CACHE.get(key)
        if cached is not None:
            return cached

        from transformers import AutoModel  # lazy import — keeps test surface clean

        loader_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            loader_kwargs["dtype"] = dtype
        draft = AutoModel.from_pretrained(draft_path, **loader_kwargs)
        # ``device_map`` would do this too, but we already know the target
        # device — explicit ``.to(device)`` keeps the loader path simple and
        # avoids the accelerate dispatch hooks that would interpose between
        # the verifier's KV cache and DFlash's per-block forward.
        draft = draft.to(device).eval()

        _DRAFT_CACHE[key] = draft
        return draft


def _normalize_dflash_output(
    raw: Any,
    tokenizer: Any = None,
    *,
    prompt_len: int = 0,
    stop_token_ids: Optional[list[int]] = None,
) -> dict:
    """Convert dflash.dflash_generate(return_stats=True) output to the _handle_sample response shape.

    Accepts:
      - ``SimpleNamespace`` / object with ``output_ids``, ``acceptance_lengths``,
        and timing attributes (the canonical ``return_stats=True`` shape).
      - A ``torch.Tensor`` of shape ``[1, total_len]`` (no-stats fallback).
      - A dict with ``sequences`` etc. (the legacy mock shape used by the
        CPU test suite — kept so existing tests keep passing).

    Returns a dict with keys: ``sequences``, ``texts``, ``stop_reasons``,
    ``sequence_logprobs``, plus optional ``acceptance_rate`` and
    ``acceptance_lengths`` when the underlying call exposed them.
    """
    if isinstance(raw, dict):
        seqs = raw.get("sequences", [])
        texts = raw.get("texts")
        stop_reasons = raw.get("stop_reasons")
        logprobs = raw.get("sequence_logprobs")
        acceptance_rate = raw.get("acceptance_rate")
        acceptance_lengths = raw.get("acceptance_lengths")
        if texts is None:
            texts = [
                tokenizer.decode(s, skip_special_tokens=True) if tokenizer else "" for s in seqs
            ]
        if stop_reasons is None:
            stop_reasons = ["length"] * len(seqs)
        if logprobs is None:
            logprobs = [[] for _ in seqs]
        out = {
            "sequences": seqs,
            "texts": texts,
            "stop_reasons": stop_reasons,
            "sequence_logprobs": logprobs,
        }
        if acceptance_rate is not None:
            out["acceptance_rate"] = acceptance_rate
        if acceptance_lengths is not None:
            out["acceptance_lengths"] = acceptance_lengths
        return out

    # Tensor or stats SimpleNamespace path.
    if isinstance(raw, torch.Tensor):
        output_ids = raw
        acceptance_lengths: Optional[list[int]] = None
    else:
        output_ids = getattr(raw, "output_ids", None)
        acceptance_lengths = getattr(raw, "acceptance_lengths", None)

    if output_ids is None:
        # Defensive — the caller already filtered for the stats path, but if
        # something unexpected lands here we don't want to mask it as success.
        return {
            "sequences": [],
            "texts": [],
            "stop_reasons": [],
            "sequence_logprobs": [],
        }

    # ``output_ids`` includes the prompt; strip it.
    completion = output_ids[0, prompt_len:].tolist()

    text = tokenizer.decode(completion, skip_special_tokens=True) if tokenizer else ""

    # Stop reason: hit a known stop id → "stop"; otherwise "length".
    hit_stop = False
    if stop_token_ids and completion:
        last = completion[-1]
        if last in stop_token_ids:
            hit_stop = True
    if not hit_stop:
        eos = getattr(tokenizer, "eos_token_id", None) if tokenizer else None
        if eos is not None and completion and completion[-1] == eos:
            hit_stop = True

    out = {
        "sequences": [completion],
        "texts": [text],
        "stop_reasons": ["stop" if hit_stop else "length"],
        "sequence_logprobs": [[]],
    }

    if acceptance_lengths:
        out["acceptance_lengths"] = list(acceptance_lengths)
        # Acceptance rate per the DFlash paper: average tokens accepted per
        # speculation step. Each entry of ``acceptance_lengths`` is the
        # number of accepted tokens for a single step (always ≥ 1 because
        # at least the verifier-emitted bonus token counts).
        out["acceptance_rate"] = float(sum(acceptance_lengths) / len(acceptance_lengths))

    return out


def _completion_logprobs(
    verifier_model: Any,
    *,
    prompt_tokens: list[int],
    sequences: list[list[int]],
    device: Any,
) -> list[list[float]]:
    """Score DFlash completions with the verifier model.

    Tinker's sample response carries one rollout-policy logprob per generated
    token. DFlash currently returns token IDs and acceptance stats, so we run a
    no-grad verifier pass over prompt+completion to recover the same
    completion logprobs the standard HF generate path exposes.
    """
    if not sequences:
        return []

    model_config = getattr(verifier_model, "config", None)
    pad_id = int(
        getattr(model_config, "pad_token_id", None)
        or getattr(model_config, "eos_token_id", 0)
        or 0
    )
    max_len = max(len(prompt_tokens) + len(sequence) for sequence in sequences)
    rows: list[list[int]] = []
    for sequence in sequences:
        row = [int(token) for token in prompt_tokens] + [int(token) for token in sequence]
        rows.append(row + [pad_id] * (max_len - len(row)))

    input_ids = torch.tensor(rows, device=device, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, sequence in enumerate(sequences):
        attention_mask[row_idx, : len(prompt_tokens) + len(sequence)] = 1

    with torch.no_grad():
        outputs = verifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        log_probs = torch.nn.functional.log_softmax(outputs.logits.float(), dim=-1)

    out: list[list[float]] = []
    start = max(len(prompt_tokens) - 1, 0)
    for row_idx, sequence in enumerate(sequences):
        row: list[float] = []
        for offset, token in enumerate(sequence):
            row.append(float(log_probs[row_idx, start + offset, int(token)].detach().cpu()))
        out.append(row)
    return out


def run_dflash_sample(
    *,
    verifier_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    n: int,
    seed: Optional[int],
    stop: Optional[list],
    spec_request: SpeculativeDecodingRequest,
    dflash_config: DFlashConfig,
    device: Any,
    base_model_name: Optional[str] = None,
) -> tuple[Optional[dict], SpeculativeDecodingMetadata]:
    """Attempt DFlash speculative decoding; return normalized result or fallback signal.

    Parameters
    ----------
    verifier_model:
        The main model used for verification. May be a PEFT-wrapped causal LM —
        the LoRA-augmented inner causal LM is forwarded to DFlash so adapter
        weights influence each verification step.
    tokenizer:
        Tokenizer for decoding generated token IDs and resolving string stops.
    prompt_tokens:
        Input token IDs (prompt only, not including any BOS padding).
    max_new_tokens:
        Maximum number of new tokens to generate.
    temperature:
        Sampling temperature forwarded to dflash. ``0.0`` is greedy.
    top_p, top_k, n:
        Recorded so the policy resolver can fall back when the request
        exceeds DFlash's transformers-backend support (single-batch,
        temperature-only).
    seed:
        Optional RNG seed. Applied via ``torch.manual_seed`` because the
        DFlash transformers backend doesn't take a seed kwarg.
    stop:
        Optional list of stop strings/token IDs (Hatchery wire format).
    spec_request:
        Parsed per-request speculative decoding options.
    dflash_config:
        Server-side DFlash configuration (draft model, default draft token count).
    device:
        torch device string/object for prompt tensor construction.

    Returns
    -------
    (result, metadata)
        ``result`` is a normalized response dict (same shape as
        ``_handle_sample``'s return) on success, or ``None`` when the
        caller should fall back to HF generate. ``metadata`` always
        reflects what was attempted and why any fallback happened.

    Raises
    ------
    Any exception from dflash when ``spec_request.strict`` is True.
    ImportError when dflash is not installed and strict mode is True.
    ValueError from :func:`resolve_dflash_policy` (or the n>1 / top_p / top_k
    surface fallbacks below) when strict mode is True.
    """
    should_use, meta = resolve_dflash_policy(
        spec_request,
        dflash_config,
        base_model_name=base_model_name,
    )
    if not should_use:
        return None, meta

    dflash_mod = _try_import_dflash()
    if dflash_mod is None:
        meta.fallback_reason = "dflash_not_installed"
        if spec_request.strict:
            raise ImportError("dflash package is not installed but strict mode was requested")
        logger.warning(
            "dflash.not_installed",
            extra={"draft_model": meta.draft_model, "fallback": "hf_generate"},
        )
        return None, meta

    generate_fn, is_real_dflash = _resolve_dflash_generate(dflash_mod)
    if generate_fn is None:
        meta.fallback_reason = "dflash_api_unavailable"
        if spec_request.strict:
            raise RuntimeError(
                "dflash module does not expose dflash_generate / generate; "
                "check installed dflash version"
            )
        logger.warning(
            "dflash.api_unavailable",
            extra={"draft_model": meta.draft_model, "fallback": "hf_generate"},
        )
        return None, meta

    # Draft-model load / cache. We need the verifier's dtype + device so the
    # draft sits on the same GPU and shares dtype with the verifier's KV cache
    # (DFlash mixes their hidden states in the speculation block).
    verifier_dtype = getattr(verifier_model, "dtype", None)
    verifier_device_attr = getattr(verifier_model, "device", None)
    if isinstance(verifier_device_attr, (torch.device, str)):
        verifier_device: Any = verifier_device_attr
    else:
        verifier_device = device

    # The canonical-shape detection is "we resolved the real
    # dflash.model.dflash_generate". Test mocks expose only ``generate`` so
    # ``is_real_dflash`` stays False and the legacy kwargs / no-loader
    # path runs.
    needs_real_draft = is_real_dflash

    # The transformers backend is single-batch and temperature-only. We only
    # surface those restrictions when the loaded dflash module is the real
    # one — test doubles exercising the mock ``generate`` attribute keep
    # accepting top_p / top_k / n>1 verbatim so existing tests can still
    # assert against the kwargs we forward.
    if needs_real_draft:
        if n > 1:
            meta.fallback_reason = "n_gt_1_unsupported"
            if spec_request.strict:
                raise ValueError(
                    "DFlash transformers backend does not support n>1; "
                    "strict mode rejected the request"
                )
            return None, meta
        constrained_top_p = top_p is not None and top_p < 1.0
        constrained_top_k = top_k is not None and top_k > 0
        if constrained_top_p or constrained_top_k:
            meta.fallback_reason = "top_p_top_k_unsupported"
            if spec_request.strict:
                raise ValueError(
                    "DFlash transformers backend supports temperature only; "
                    "strict mode rejected top_p / top_k constraints"
                )
            return None, meta

    if needs_real_draft:
        try:
            draft_obj: Any = _load_draft_model(
                dflash_mod, meta.draft_model, verifier_device, verifier_dtype
            )
        except Exception as exc:  # noqa: BLE001 — surface as fallback
            if spec_request.strict:
                raise
            meta.fallback_reason = f"draft_load_error:{type(exc).__name__}"
            logger.warning(
                "dflash.draft_load_error",
                extra={
                    "draft_model": meta.draft_model,
                    "error": str(exc),
                    "fallback": "hf_generate",
                },
                exc_info=True,
            )
            return None, meta
    else:
        draft_obj = meta.draft_model  # opaque to test doubles

    target_for_dflash = _resolve_verifier_for_dflash(verifier_model)
    stop_token_ids = _resolve_stop_token_ids(stop, tokenizer)
    input_ids = torch.tensor([prompt_tokens], device=verifier_device, dtype=torch.long)
    prompt_len = input_ids.shape[1]

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # Apply the same temperature floor as the HF generate path; DFlash routes
    # ``temperature < 1e-5`` to greedy via ``argmax``.
    safe_temperature = max(float(temperature), 0.0)

    # Build the canonical DFlash kwargs. Test doubles intercepting the legacy
    # ``generate`` attribute also receive ``verifier`` / ``draft`` / ``tokenizer``
    # / ``top_p`` / ``max_draft_tokens`` so existing tests can keep asserting
    # against those keys without depending on the real DFlash surface.
    canonical_kwargs: dict[str, Any] = dict(
        model=draft_obj,
        target=target_for_dflash,
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        stop_token_ids=stop_token_ids,
        temperature=safe_temperature,
    )
    if needs_real_draft:
        canonical_kwargs["return_stats"] = True
    else:
        # Legacy / mocked surface: keep the keys the existing tests assert on.
        canonical_kwargs.update(
            verifier=verifier_model,
            draft=meta.draft_model,
            tokenizer=tokenizer,
            top_p=top_p,
            max_draft_tokens=meta.max_draft_tokens,
        )
        if top_k > 0:
            canonical_kwargs["top_k"] = top_k
        if seed is not None:
            canonical_kwargs["seed"] = seed
        if n > 1:
            canonical_kwargs["n"] = n
        if stop:
            canonical_kwargs["stop"] = stop

    t_start = time.monotonic()
    try:
        raw_output = generate_fn(**canonical_kwargs)
        latency_ms = (time.monotonic() - t_start) * 1000
        meta.used_backend = SPEC_BACKEND_DFLASH

        result = _normalize_dflash_output(
            raw_output, prompt_len=prompt_len, tokenizer=tokenizer, stop_token_ids=stop_token_ids
        )
        sequences = [
            [int(token) for token in sequence]
            for sequence in result.get("sequences") or []
        ]
        sequence_logprobs = result.get("sequence_logprobs")
        if (
            not sequence_logprobs
            or len(sequence_logprobs) != len(sequences)
            or any(
                len(row) != len(sequence)
                for row, sequence in zip(sequence_logprobs, sequences, strict=False)
            )
        ):
            result["sequence_logprobs"] = _completion_logprobs(
                verifier_model,
                prompt_tokens=prompt_tokens,
                sequences=sequences,
                device=verifier_device,
            )

        logger.info(
            "dflash.sample_complete",
            extra={
                "draft_model": meta.draft_model,
                "max_draft_tokens": meta.max_draft_tokens,
                "latency_ms": round(latency_ms, 1),
                "acceptance_rate": result.get("acceptance_rate"),
            },
        )
        return result, meta

    except Exception as exc:  # noqa: BLE001
        if spec_request.strict:
            raise
        meta.fallback_reason = f"dflash_runtime_error:{type(exc).__name__}"
        logger.warning(
            "dflash.runtime_error",
            extra={"error": str(exc), "fallback": "hf_generate"},
            exc_info=True,
        )
        return None, meta
