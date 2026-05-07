# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""DFlash speculative decoding runtime integration for hatchery-core workers.

All dflash library imports are lazy — this module never imports dflash at the
module level. When the dflash package is absent, or when the runtime call fails,
the caller receives ``(None, metadata)`` and falls back to the standard HF
generate path unless strict mode is set.

Integration point: ``GPUWorker._handle_sample`` calls ``run_dflash_sample()``
after extracting payload parameters when ``payload["speculative_decoding"]``
is present and the DFlash backend is eligible.

LoRA/PEFT compatibility: the verifier_model passed to ``run_dflash_sample``
may be a PEFT-wrapped model. It is forwarded to ``dflash.generate()`` without
unwrapping — DFlash is expected to handle PEFT-wrapped verifiers.
"""

from __future__ import annotations

import logging
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
        HF model name or local path for the draft model. *Required* — without
        it, DFlash is skipped regardless of per-request opts.
    max_draft_tokens:
        Default maximum draft tokens per speculation step (per-request
        ``max_draft_tokens`` overrides this).
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
            raise ValueError(
                "DFlash is disabled on this worker but strict mode was requested"
            )
        return False, meta

    if not dflash_config.draft_model:
        meta.fallback_reason = "no_draft_model_configured"
        if spec_request.strict:
            raise ValueError(
                "No draft_model configured on DFlashConfig but strict mode was requested"
            )
        return False, meta

    max_draft = spec_request.max_draft_tokens or dflash_config.max_draft_tokens
    meta.draft_model = dflash_config.draft_model
    meta.max_draft_tokens = max_draft

    return True, meta


def _normalize_dflash_output(raw: Any, tokenizer: Any) -> dict:
    """Convert dflash.generate() output to the _handle_sample response shape.

    Accepts either a dict (the expected DFlash wire format) or an object
    with attribute access. Returns a dict with keys:
      ``sequences``, ``texts``, ``stop_reasons``, ``sequence_logprobs``.

    dflash.generate() is expected to return a dict containing:
      - ``sequences``: list[list[int]] — generated token IDs per sequence
      - ``texts``: list[str] — decoded texts (optional; re-decoded if absent)
      - ``stop_reasons``: list[str] — "stop" or "length" per sequence (optional)
      - ``sequence_logprobs``: list[list[float]] — token logprobs (optional)
      - ``acceptance_rate``: float — speculative acceptance rate (optional)
    """
    if isinstance(raw, dict):
        seqs = raw.get("sequences", [])
        texts = raw.get("texts")
        stop_reasons = raw.get("stop_reasons")
        logprobs = raw.get("sequence_logprobs")
    else:
        seqs = getattr(raw, "sequences", [])
        texts = getattr(raw, "texts", None)
        stop_reasons = getattr(raw, "stop_reasons", None)
        logprobs = getattr(raw, "sequence_logprobs", None)

    if texts is None:
        texts = [
            tokenizer.decode(s, skip_special_tokens=True) if tokenizer else ""
            for s in seqs
        ]
    if stop_reasons is None:
        stop_reasons = ["length"] * len(seqs)
    if logprobs is None:
        logprobs = [[] for _ in seqs]

    return {
        "sequences": seqs,
        "texts": texts,
        "stop_reasons": stop_reasons,
        "sequence_logprobs": logprobs,
    }


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
) -> tuple[Optional[dict], SpeculativeDecodingMetadata]:
    """Attempt DFlash speculative decoding; return normalized result or fallback signal.

    Parameters
    ----------
    verifier_model:
        The main model used for verification. May be a PEFT-wrapped model —
        passed through to ``dflash.generate()`` without unwrapping.
    tokenizer:
        Tokenizer for decoding generated token IDs.
    prompt_tokens:
        Input token IDs (prompt only, not including any BOS padding).
    max_new_tokens:
        Maximum number of new tokens to generate.
    temperature, top_p, top_k:
        Sampling parameters forwarded to dflash.generate().
    n:
        Number of sequences to generate.
    seed:
        Optional RNG seed for reproducibility.
    stop:
        Optional list of stop strings/token IDs.
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
    Any exception from dflash.generate() when ``spec_request.strict`` is True.
    ImportError when dflash is not installed and strict mode is True.
    ValueError from :func:`resolve_dflash_policy` when strict mode is True.
    """
    should_use, meta = resolve_dflash_policy(spec_request, dflash_config)
    if not should_use:
        return None, meta

    dflash_mod = _try_import_dflash()
    if dflash_mod is None:
        meta.fallback_reason = "dflash_not_installed"
        if spec_request.strict:
            raise ImportError(
                "dflash package is not installed but strict mode was requested"
            )
        logger.warning(
            "dflash.not_installed",
            extra={"draft_model": meta.draft_model, "fallback": "hf_generate"},
        )
        return None, meta

    input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    t_start = time.monotonic()
    try:

        generate_kwargs: dict[str, Any] = dict(
            verifier=verifier_model,
            draft=meta.draft_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_draft_tokens=meta.max_draft_tokens,
        )
        if top_k > 0:
            generate_kwargs["top_k"] = top_k
        if seed is not None:
            generate_kwargs["seed"] = seed
        if n > 1:
            generate_kwargs["n"] = n
        if stop:
            generate_kwargs["stop"] = stop

        raw_output = dflash_mod.generate(**generate_kwargs)
        latency_ms = (time.monotonic() - t_start) * 1000
        meta.used_backend = SPEC_BACKEND_DFLASH

        acceptance_rate: Optional[float] = None
        if isinstance(raw_output, dict):
            acceptance_rate = raw_output.get("acceptance_rate")
        logger.info(
            "dflash.sample_complete",
            extra={
                "draft_model": meta.draft_model,
                "max_draft_tokens": meta.max_draft_tokens,
                "latency_ms": round(latency_ms, 1),
                "acceptance_rate": acceptance_rate,
            },
        )

        result = _normalize_dflash_output(raw_output, tokenizer)
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
