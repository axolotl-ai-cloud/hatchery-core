# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tinker/Fireworks-compatible API surface.

Mounts a FastAPI router at ``/api/v1`` mirroring the public Tinker REST
surface (see ``tinker`` on PyPI). Clients of the official ``tinker``
Python SDK can point at our gateway by setting ``TINKER_API_URL`` to our
base URL and using any valid API key registered with our auth provider.

Design notes:
* Tinker's API is futures-based: each training op returns an
  ``UntypedAPIFuture`` immediately and the client polls
  ``/api/v1/retrieve_future`` until completion. We preserve that
  contract even though our internal queue already delivers futures.
* ``model_id`` maps 1:1 to our ``session_id``.
* ``seq_id`` is recorded but, because the platform already enforces
  per-session ordering via the job queue, isn't used for re-ordering.
* We cover the operations required by the standard tinker training
  loop: ``create_session``, ``create_model``, ``forward_backward``,
  ``optim_step``, ``asample``, ``save_weights``, checkpoint listing,
  and ``retrieve_future``. Out-of-scope: custom loss functions beyond
  ``cross_entropy``, multi-modal chunks, ``save_state``/``load_state``
  with optimizer weights.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Optional, Union

import msgpack
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from hatchery.core.config import Config
from hatchery.core.gateway import (
    _enqueue_job,
    _resolve_preferred_worker,
    get_config,
    get_current_user,
)
from hatchery.core.lora_target_modules import target_modules_for
from hatchery.core.plugins import run_post_op_hooks, run_pre_op_hooks
from hatchery.core.protocols import (
    AuthenticatedUser,
    JobStatus,
    SessionRecord,
    SessionStatus,
)
from hatchery.core.spec_decoding import SpeculativeDecodingRequest

router = APIRouter(prefix="/api/v1", tags=["tinker-compat"])


# ─── Wire types (pydantic mirrors of tinker.types) ────────────────────────


class EncodedTextChunk(BaseModel):
    type: str = "encoded_text"
    tokens: list[int]


class EncodedImageChunk(BaseModel):
    """Base64-encoded image for VLM fine-tuning.

    Matches the Fireworks convention of embedding images as
    ``data:<mime_type>;base64,<data>`` in message content. The
    ``data`` field should contain the base64-encoded image bytes
    (without the ``data:`` prefix) or the full data URI.

    Also used as the normalized internal representation after
    resolving ``ImageAssetPointerChunk`` at gateway ingress: the
    Tinker SDK's wire schema sends raw bytes (base64-encoded) in
    the ``data`` field too, so after fetching we just base64 it.
    """

    type: str = "image"
    data: Any = None  # base64-encoded image (str) or raw bytes; workers get bytes
    mime_type: str = "image/jpeg"
    # Image format hint propagated from SDK ``ImageChunk``/``ImageAssetPointerChunk``.
    # When set, takes precedence over ``mime_type`` for PIL decoding.
    format: Optional[str] = None


class ImageAssetPointerChunk(BaseModel):
    """Reference to an image by URL/path instead of inline bytes.

    Mirrors the Tinker SDK ``ImageAssetPointerChunk`` wire type.
    The gateway resolves ``location`` to raw bytes at request
    ingress (HTTPS fetch only for MVP) and replaces the chunk with
    an ``EncodedImageChunk`` before the payload reaches workers.
    """

    type: str = "image_asset_pointer"
    format: str = "png"  # Literal["png", "jpeg"] on the SDK side.
    location: str
    expected_tokens: Optional[int] = None


class ModelInput(BaseModel):
    """Composite input supporting both alternate and SDK-unified wire shapes.

    Alternate shape (also accepted):
      ``{"chunks": [encoded_text, ...], "image_chunks": [image, ...]}``

    Tinker SDK shape (``tinker.types.ModelInput``) sends a single
    discriminated union under ``chunks``:
      ``{"chunks": [
            {"type": "encoded_text", "tokens": [...]},
            {"type": "image", "data": "<base64>", "format": "png"},
            {"type": "image_asset_pointer", "location": "...", "format": "jpeg"},
         ]}``

    A ``@model_validator(mode="before")`` splits the SDK-style list into
    our internal ``chunks`` (text) / ``image_chunks`` (image + pointer)
    buckets so the rest of the pipeline is unchanged.
    """

    chunks: list[EncodedTextChunk] = Field(default_factory=list)
    image_chunks: list[Union[EncodedImageChunk, ImageAssetPointerChunk]] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def _split_unified_chunks(cls, data: Any) -> Any:
        """If ``chunks`` contains image entries (SDK-unified format),
        pull them into ``image_chunks``. Idempotent for alternate-format callers."""
        if not isinstance(data, dict):
            return data
        raw_chunks = data.get("chunks")
        if not isinstance(raw_chunks, list):
            return data
        text_chunks: list[Any] = []
        image_chunks: list[Any] = list(data.get("image_chunks") or [])
        for c in raw_chunks:
            if isinstance(c, dict):
                ctype = c.get("type", "encoded_text")
                if ctype == "encoded_text":
                    text_chunks.append(c)
                elif ctype in ("image", "image_asset_pointer"):
                    image_chunks.append(c)
                else:
                    # Unknown chunk type — pass through as text to surface
                    # a validation error downstream instead of silently dropping.
                    text_chunks.append(c)
            else:
                text_chunks.append(c)
        new_data = dict(data)
        new_data["chunks"] = text_chunks
        new_data["image_chunks"] = image_chunks
        return new_data


class TensorData(BaseModel):
    """Port of tinker's TensorData with sparse CSR support.

    Dense format: ``{dtype, shape, data}`` — a flat list of values.
    Sparse CSR format: ``{shape, sparse_col_indices, sparse_values}``
    where only non-zero entries are transmitted. Useful for long-context
    ``token_weights`` where most positions are zero (prompt tokens).

    The Tinker SDK also supports ``sparse_crow_indices`` for full CSR,
    but the common case is a 1-D sparse vector where ``sparse_col_indices``
    + ``sparse_values`` suffice.
    """

    dtype: Optional[str] = None
    shape: Optional[list[int]] = None
    data: list[Any] = Field(default_factory=list)
    # Sparse CSR encoding (col indices + values).
    sparse_col_indices: Optional[list[int]] = None
    sparse_values: Optional[list[Any]] = None
    # Full CSR row pointers (for 2-D sparse tensors).
    sparse_crow_indices: Optional[list[int]] = None


class Datum(BaseModel):
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData] = Field(default_factory=dict)


class LoraConfigPayload(BaseModel):
    rank: int = Field(32, ge=1, le=256)
    seed: Optional[int] = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True
    use_rslora: bool = False
    init_lora_weights: str = "default"
    lora_dropout: float = Field(0.0, ge=0.0, le=1.0)


class CreateSessionRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, str] = Field(default_factory=dict)
    sdk_version: Optional[str] = None
    project_id: Optional[str] = None


class SessionHeartbeatRequest(BaseModel):
    session_id: str


class CreateModelRequest(BaseModel):
    session_id: str
    model_seq_id: int = 0
    base_model: str
    # Fireworks-compatible signalling: ``lora_config`` present ⇒ LoRA
    # fine-tuning at the given rank. ``None`` (or absent on the wire)
    # ⇒ full-parameter fine-tuning. The default — a populated
    # ``LoraConfigPayload`` — preserves rank=32 LoRA behavior for any
    # client that sends ``lora_config: {}``, matching what the SDK
    # has done historically; new clients targeting FFT must send
    # ``lora_config: null`` (or omit the field) explicitly.
    lora_config: Optional[LoraConfigPayload] = Field(default_factory=LoraConfigPayload)
    user_metadata: dict[str, str] = Field(default_factory=dict)


class AdamParams(BaseModel):
    learning_rate: float = Field(1e-4, gt=0, le=10.0)
    beta1: float = Field(0.9, ge=0.0, le=1.0)
    beta2: float = Field(0.95, ge=0.0, le=1.0)
    eps: float = Field(1e-12, gt=0, le=1.0)
    weight_decay: float = Field(0.0, ge=0.0, le=10.0)
    grad_clip_norm: float = Field(0.0, ge=0.0)


class ForwardBackwardRequest(BaseModel):
    model_id: str
    seq_id: int = 0
    forward_backward_input: dict
    type: str = "forward_backward"


class ForwardRequest(BaseModel):
    """Wire schema for the tinker SDK's ``/forward`` endpoint.

    The SDK serializes forward-only calls with the key ``forward_input``
    (see ``tinker/types/forward_request.py``) — distinct from the
    ``forward_backward_input`` key used by ``/forward_backward``. Server
    must accept the SDK key verbatim or clients get 422s.
    """

    model_id: str
    seq_id: int = 0
    forward_input: dict
    type: str = "forward"


class ForwardOnlyRequest(BaseModel):
    """Wire schema for ``TrainingClient.forward_only``.

    Matches the ``forward_backward`` envelope shape but uses the
    ``forward_only_input`` key to make the distinct op unambiguous on
    the wire. Contents mirror ``forward_backward_input`` (``data``,
    ``loss_fn``, ``loss_fn_config``) but the server routes to a
    no-grad handler that does not mutate training state.
    """

    model_id: str
    seq_id: int = 0
    forward_only_input: dict
    type: str = "forward_only"


class OptimStepRequest(BaseModel):
    model_id: str
    seq_id: int = 0
    adam_params: AdamParams = Field(default_factory=AdamParams)
    grad_accumulation_normalization: Optional[str] = (
        None  # "num_loss_tokens" | "num_sequences" | None
    )
    type: str = "optim_step"


class SamplingParams(BaseModel):
    max_tokens: Optional[int] = Field(256, ge=1, le=32768)
    seed: Optional[int] = None
    stop: Optional[Any] = None
    temperature: float = Field(1.0, ge=0.0, le=100.0)
    top_k: int = Field(-1, ge=-1, le=1000)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    speculative_decoding: Optional[SpeculativeDecodingRequest] = None
    enable_thinking: Optional[bool] = None


class SampleRequest(BaseModel):
    prompt: ModelInput
    num_samples: int = Field(1, ge=1, le=64)
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    base_model: Optional[str] = None
    model_path: Optional[str] = None
    sampling_session_id: Optional[str] = None
    model_id: Optional[str] = None
    seq_id: Optional[int] = None
    prompt_logprobs: bool = False
    topk_prompt_logprobs: int = 0
    type: str = "sample"


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    seq_id: Optional[int] = None
    ttl_seconds: Optional[int] = None
    type: str = "save_weights"


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    sampling_session_seq_id: Optional[int] = None
    seq_id: Optional[int] = None
    ttl_seconds: Optional[int] = None
    type: str = "save_weights_for_sampler"


class SaveStateRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    seq_id: Optional[int] = None
    ttl_seconds: Optional[int] = None
    type: str = "save_state"


class LoadWeightsRequest(BaseModel):
    model_id: str
    path: str
    optimizer: bool = False
    seq_id: Optional[int] = None
    weights_access_token: Optional[str] = None
    type: str = "load_weights"


class RetrieveFutureRequest(BaseModel):
    # Accept either our historical ``future_id`` or the tinker SDK's
    # ``request_id`` (both are Optional so Pydantic lets us pick one).
    future_id: Optional[str] = None
    request_id: Optional[str] = None
    allow_metadata_only: bool = False

    def resolve(self) -> str:
        rid = self.request_id or self.future_id
        if not rid:
            raise HTTPException(400, "request_id (or future_id) is required")
        return rid


# ─── Future registry ──────────────────────────────────────────────────────


_REGISTRY_MAX_ENTRIES = 100_000
_REGISTRY_MAX_AGE_S = 3600.0


class _FutureEntry:
    __slots__ = (
        "job_id",
        "user_id",
        "operation",
        "inline_result",
        "pre_op_contexts",
        "session",
        "created_at",
    )

    def __init__(
        self,
        job_id: str,
        user_id: str,
        operation: str,
        inline_result: Optional[dict] = None,
        pre_op_contexts: Optional[list] = None,
        session: Optional[SessionRecord] = None,
    ) -> None:
        self.job_id = job_id
        self.user_id = user_id
        self.operation = operation
        self.inline_result = inline_result
        self.pre_op_contexts = pre_op_contexts
        self.session = session
        self.created_at = time.time()


class _FutureRegistry:
    """Maps tinker ``request_id`` to our internal ``job_id``.

    Entries are evicted after ``_REGISTRY_MAX_AGE_S`` (1 hour) or when
    the registry exceeds ``_REGISTRY_MAX_ENTRIES`` (100k).
    """

    def __init__(self) -> None:
        self._map: dict[str, _FutureEntry] = {}
        self._reg_count = 0

    def _sweep(self) -> None:
        cutoff = time.time() - _REGISTRY_MAX_AGE_S
        stale = [k for k, v in self._map.items() if v.created_at < cutoff]
        for k in stale:
            del self._map[k]
        if len(self._map) > _REGISTRY_MAX_ENTRIES:
            by_age = sorted(self._map, key=lambda k: self._map[k].created_at)
            for k in by_age[: len(self._map) - _REGISTRY_MAX_ENTRIES]:
                del self._map[k]

    def register(
        self,
        job_id: str,
        user_id: str,
        operation: str,
        inline_result: Optional[dict] = None,
        pre_op_contexts: Optional[list] = None,
        session: Optional[SessionRecord] = None,
    ) -> str:
        self._reg_count += 1
        if self._reg_count % 1000 == 0:
            self._sweep()
        future_id = f"fut_{uuid.uuid4().hex}"
        self._map[future_id] = _FutureEntry(
            job_id=job_id,
            user_id=user_id,
            operation=operation,
            inline_result=inline_result,
            pre_op_contexts=pre_op_contexts,
            session=session,
        )
        return future_id

    def lookup(self, future_id: str) -> Optional[_FutureEntry]:
        return self._map.get(future_id)


_futures = _FutureRegistry()


class _SeqIdTracker:
    """Per-session monotonic ``seq_id`` tracker for idempotency.

    Entries are evicted after ``_REGISTRY_MAX_AGE_S`` to prevent
    unbounded growth from abandoned sessions.
    """

    def __init__(self) -> None:
        self._seen: dict[str, dict[int, dict]] = {}
        self._ts: dict[str, float] = {}
        self._check_count = 0

    def check(self, session_id: str, seq_id: Optional[int]) -> Optional[dict]:
        """Return the cached future response if this ``seq_id`` was
        already processed for this session. Return ``None`` if the
        request is new (or ``seq_id`` is ``None`` / 0 / not provided).

        ``seq_id=0`` is the Pydantic default and indicates the client
        didn't explicitly set a sequence number — treat as non-idempotent.
        """
        self._check_count += 1
        if self._check_count % 1000 == 0:
            self._sweep()
        if not seq_id:
            return None
        session_map = self._seen.get(session_id)
        if session_map is None:
            return None
        return session_map.get(seq_id)

    def _sweep(self) -> None:
        cutoff = time.time() - _REGISTRY_MAX_AGE_S
        stale = [sid for sid, ts in self._ts.items() if ts < cutoff]
        for sid in stale:
            self._seen.pop(sid, None)
            self._ts.pop(sid, None)

    def record(self, session_id: str, seq_id: Optional[int], response: dict) -> None:
        if not seq_id:
            return
        if session_id not in self._seen:
            self._seen[session_id] = {}
        self._seen[session_id][seq_id] = response
        self._ts[session_id] = time.time()


_seq_tracker = _SeqIdTracker()


def _future_response(
    job_id: str,
    user_id: str,
    operation: str,
    *,
    model_id: Optional[str] = None,
    inline_result: Optional[dict] = None,
    pre_op_contexts: Optional[list] = None,
    session: Optional[SessionRecord] = None,
) -> dict:
    """Return the SDK-shaped ``UntypedAPIFuture`` envelope.

    The tinker SDK POSTs an op and expects ``{request_id, model_id}``
    back (schema: ``UntypedAPIFuture``). Subsequent calls to
    ``/api/v1/retrieve_future`` deliver the actual typed response.

    ``inline_result`` lets a caller register a synchronously-completed
    response (e.g. ``create_model`` returns the finished model_id
    directly). Extra fields are kept for callers that read
    ``future_id`` / ``operation`` / ``status``.
    """
    future_id = _futures.register(
        job_id,
        user_id,
        operation,
        inline_result,
        pre_op_contexts=pre_op_contexts,
        session=session,
    )
    return {
        "request_id": future_id,
        "model_id": model_id,
        # Extra fields for internal tests / non-SDK clients.
        "future_id": future_id,
        "operation": operation,
        "status": "pending",
    }


def _idempotent_future_response(
    session_id: str,
    seq_id: Optional[int],
    job_id: str,
    user_id: str,
    operation: str,
    *,
    pre_op_contexts: Optional[list] = None,
    session: Optional[SessionRecord] = None,
) -> dict:
    """Wrap ``_future_response`` with ``seq_id`` deduplication.

    Key the idempotency cache by ``(operation, session_id)`` not just
    ``session_id`` — the SDK keeps independent ``seq_id`` counters for
    training ops and sampling, and without the operation prefix a
    ``sample(seq_id=0)`` would return the cached ``forward_backward(seq_id=0)``
    response (wrong shape).
    """
    scoped = f"{operation}::{session_id}"
    cached = _seq_tracker.check(scoped, seq_id)
    if cached is not None:
        return cached
    resp = _future_response(
        job_id,
        user_id,
        operation,
        pre_op_contexts=pre_op_contexts,
        session=session,
    )
    _seq_tracker.record(scoped, seq_id, resp)
    return resp


# ─── Helpers: wire format conversion ──────────────────────────────────────


def _model_input_tokens(mi: ModelInput) -> list[int]:
    out: list[int] = []
    for chunk in mi.chunks:
        out.extend(chunk.tokens)
    return out


def _model_input_has_images(mi: ModelInput) -> bool:
    return bool(mi.image_chunks)


def _decode_image_chunks(mi: ModelInput) -> list[bytes]:
    """Decode image chunks to raw bytes.

    Expects that any ``ImageAssetPointerChunk`` entries have already
    been resolved to ``EncodedImageChunk`` by
    ``_resolve_image_asset_pointers`` before the request is enqueued.
    Encountering an unresolved pointer here is a programmer error
    (ingress forgot to await the resolver).

    ``EncodedImageChunk.data`` may be either a base64 string (wire
    format from internal clients) or raw bytes (SDK ``ImageChunk``
    after Pydantic's validator). Both are normalized to ``bytes``.
    """
    import base64

    images: list[bytes] = []
    for ic in mi.image_chunks:
        if isinstance(ic, ImageAssetPointerChunk):
            raise HTTPException(500, "ImageAssetPointerChunk not resolved before worker dispatch")
        data = ic.data
        if isinstance(data, bytes):
            images.append(data)
            continue
        if isinstance(data, str):
            # Strip data URI prefix if present.
            if data.startswith("data:"):
                data = data.split(",", 1)[-1]
            images.append(base64.b64decode(data))
            continue
        raise HTTPException(
            400, f"EncodedImageChunk.data must be str or bytes, got {type(data).__name__}"
        )
    return images


_CHECKPOINT_NAME_RE = re.compile(r"^[a-zA-Z0-9_.\-]{1,128}$")
_MAX_IMAGES_PER_REQUEST = 16
_MAX_AGGREGATE_IMAGE_BYTES = 100 * 1024 * 1024

# Max response body for an HTTP(S) image fetch. 50 MB is generous for
# photos yet blocks obvious misuse (accidental video link, HTML page).
_IMAGE_FETCH_MAX_BYTES = 50 * 1024 * 1024
_IMAGE_FETCH_TIMEOUT_S = 30.0


def _is_private_ip(host: str) -> bool:
    """Return True if the host resolves to a private/loopback/link-local IP."""
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return True
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


async def _fetch_image_asset(location: str) -> bytes:
    """Fetch image bytes for an ``ImageAssetPointerChunk.location``.

    Only HTTPS URLs are accepted by default. HTTP is allowed only when
    ``HATCHERY_ALLOW_HTTP_IMAGE_FETCH=1`` is set (for local dev). Private,
    loopback, link-local, and metadata endpoint IPs are always rejected.
    """
    import os
    from urllib.parse import urlparse

    import httpx

    parsed = urlparse(location)
    allow_http = os.environ.get("HATCHERY_ALLOW_HTTP_IMAGE_FETCH", "0") == "1"
    if parsed.scheme == "http" and not allow_http:
        raise HTTPException(400, "Only HTTPS image URLs are accepted")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            400,
            f"ImageAssetPointerChunk.location scheme not supported: {parsed.scheme}",
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise HTTPException(400, "ImageAssetPointerChunk.location has no hostname")
    if _is_private_ip(hostname):
        raise HTTPException(400, "Image URL resolves to a private/internal address")

    try:
        async with httpx.AsyncClient(
            timeout=_IMAGE_FETCH_TIMEOUT_S, follow_redirects=False
        ) as client:
            resp = await client.get(location)
    except httpx.HTTPError as exc:
        raise HTTPException(400, f"failed to fetch image asset: {exc}") from exc
    if resp.status_code != 200:
        raise HTTPException(
            400,
            f"image asset fetch returned HTTP {resp.status_code}",
        )
    body = resp.content
    if len(body) > _IMAGE_FETCH_MAX_BYTES:
        raise HTTPException(
            400,
            f"image asset exceeds {_IMAGE_FETCH_MAX_BYTES} byte limit",
        )
    return body


async def _resolve_image_asset_pointers(mi: ModelInput) -> None:
    """Replace every ``ImageAssetPointerChunk`` in ``mi.image_chunks``
    with an equivalent ``EncodedImageChunk`` holding the fetched bytes.

    Mutates ``mi`` in place. No-op if the request has no pointers.
    """
    if not mi.image_chunks:
        return
    if len(mi.image_chunks) > _MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            400, f"Too many images ({len(mi.image_chunks)} > {_MAX_IMAGES_PER_REQUEST})"
        )
    total_bytes = 0
    resolved: list[Union[EncodedImageChunk, ImageAssetPointerChunk]] = []
    for ic in mi.image_chunks:
        if isinstance(ic, ImageAssetPointerChunk):
            body = await _fetch_image_asset(ic.location)
            total_bytes += len(body)
            if total_bytes > _MAX_AGGREGATE_IMAGE_BYTES:
                raise HTTPException(400, "Aggregate image size exceeds limit")
            fmt = ic.format
            mime = f"image/{'jpeg' if fmt == 'jpeg' else fmt}"
            resolved.append(EncodedImageChunk(data=body, mime_type=mime, format=fmt))
        else:
            if ic.data:
                total_bytes += len(ic.data) if isinstance(ic.data, (bytes, str)) else 0
            resolved.append(ic)
    mi.image_chunks = resolved


async def _resolve_datum_assets(datum: Datum) -> None:
    await _resolve_image_asset_pointers(datum.model_input)


async def _build_data_items(raw_data: list[dict]) -> list[dict]:
    """Parse SDK ``data`` array into worker-facing training items.

    Performs asset-pointer resolution (HTTP fetch) before translation
    so the worker-side payload always carries raw image bytes.
    """
    items: list[dict] = []
    total_image_bytes = 0
    total_images = 0
    for d in raw_data:
        datum = Datum(**d)
        total_images += len(
            [ic for ic in datum.model_input.image_chunks if isinstance(ic, ImageAssetPointerChunk)]
        )
        if total_images > _MAX_IMAGES_PER_REQUEST:
            raise HTTPException(
                400, f"Too many images across batch ({total_images} > {_MAX_IMAGES_PER_REQUEST})"
            )
        await _resolve_datum_assets(datum)
        for ic in datum.model_input.image_chunks:
            if isinstance(ic, EncodedImageChunk) and ic.data:
                total_image_bytes += len(ic.data) if isinstance(ic.data, (bytes, str)) else 0
        if total_image_bytes > _MAX_AGGREGATE_IMAGE_BYTES:
            raise HTTPException(400, "Aggregate image size across batch exceeds limit")
        items.append(_datum_to_training_item(datum))
    return items


def _reshape_tensor_data(td: TensorData) -> Any:
    """Reshape a ``TensorData`` into a Python list, expanding sparse if needed.

    Dense format: flat ``data`` list reshaped according to ``shape``.
    Sparse format: ``sparse_col_indices`` + ``sparse_values`` expanded
    into a dense list of length ``shape[0]`` (1-D) or ``shape[0]*shape[1]``
    (2-D with ``sparse_crow_indices``).
    """
    # ── Sparse expansion ──
    if td.sparse_col_indices is not None and td.sparse_values is not None:
        return _expand_sparse_tensor(td)

    # ── Dense path ──
    raw = list(td.data)
    if not td.shape or len(td.shape) <= 1:
        return raw
    if len(td.shape) == 2:
        _t, k = td.shape
        return [raw[i : i + k] for i in range(0, len(raw), k)]
    raise ValueError(f"TensorData rank {len(td.shape)} not supported (max 2-D)")


def _expand_sparse_tensor(td: TensorData) -> Any:
    """Expand sparse CSR/COO tensor data into a dense Python list.

    For 1-D (most common — e.g., ``token_weights``):
        ``sparse_col_indices`` = positions of non-zero values
        ``sparse_values`` = the non-zero values
        Result: dense list of length ``shape[0]``, zeros elsewhere.

    For 2-D with ``sparse_crow_indices``:
        Full CSR format expanded row by row.
    """
    if not td.shape:
        raise ValueError("Sparse TensorData requires shape")

    indices = td.sparse_col_indices or []
    values = td.sparse_values or []
    if len(indices) != len(values):
        raise ValueError(
            f"sparse_col_indices length ({len(indices)}) != sparse_values length ({len(values)})"
        )

    if len(td.shape) == 1 or (len(td.shape) == 2 and td.sparse_crow_indices is None):
        # 1-D sparse vector.
        total = td.shape[0]
        dense = [0.0] * total
        for idx, val in zip(indices, values, strict=True):
            if 0 <= idx < total:
                dense[idx] = val
        return dense

    if len(td.shape) == 2 and td.sparse_crow_indices is not None:
        # Full 2-D CSR.
        rows, cols = td.shape
        crow = td.sparse_crow_indices
        dense = [[0.0] * cols for _ in range(rows)]
        for row_idx in range(rows):
            start, end = crow[row_idx], crow[row_idx + 1]
            for k in range(start, end):
                col = indices[k]
                dense[row_idx][col] = values[k]
        return dense

    raise ValueError(f"Sparse TensorData with shape {td.shape} not supported")


def _datum_to_training_item(datum: Datum) -> dict:
    """Translate a tinker Datum into our worker-facing batch item.

    Extracts every loss_fn_input we know how to handle:
    * ``target_tokens`` (1-D or 2-D) → ``labels``
    * ``weights`` (1-D or 2-D, matching target_tokens rank) → ``weights``
    * ``logprobs`` → ``logprobs`` (old-policy logprobs for RL)
    * ``advantages`` → ``advantages``

    Anything else in ``loss_fn_inputs`` is silently dropped — same
    behavior Tinker's client warns about when it doesn't know a field.
    """
    input_ids = _model_input_tokens(datum.model_input)
    item: dict[str, Any] = {"input_ids": input_ids}

    # Include raw image bytes for VLM models.
    if _model_input_has_images(datum.model_input):
        item["images"] = _decode_image_chunks(datum.model_input)

    target_tokens_td = datum.loss_fn_inputs.get("target_tokens")
    weights_td = datum.loss_fn_inputs.get("weights")
    logprobs_td = datum.loss_fn_inputs.get("logprobs")
    advantages_td = datum.loss_fn_inputs.get("advantages")

    if target_tokens_td is None:
        # Default to LM self-prediction.
        item["labels"] = list(input_ids)
    else:
        reshaped = _reshape_tensor_data(target_tokens_td)
        if reshaped and isinstance(reshaped[0], list):
            # 2-D case (SDFT). Cast every inner entry to int.
            item["labels"] = [[int(x) for x in row] for row in reshaped]
        else:
            item["labels"] = [int(x) for x in reshaped]

    if weights_td is not None:
        reshaped_w = _reshape_tensor_data(weights_td)
        if reshaped_w and isinstance(reshaped_w[0], list):
            item["weights"] = [[float(x) for x in row] for row in reshaped_w]
        else:
            item["weights"] = [float(x) for x in reshaped_w]

    if logprobs_td is not None:
        reshaped_lp = _reshape_tensor_data(logprobs_td)
        # old logprobs are always 1-D in Tinker's RL wire format
        if reshaped_lp and isinstance(reshaped_lp[0], list):
            raise ValueError("logprobs tensor must be 1-D")
        item["logprobs"] = [float(x) for x in reshaped_lp]

    if advantages_td is not None:
        reshaped_adv = _reshape_tensor_data(advantages_td)
        if reshaped_adv and isinstance(reshaped_adv[0], list):
            raise ValueError("advantages tensor must be 1-D")
        item["advantages"] = [float(x) for x in reshaped_adv]

    return item


async def _owned_session_for_model(config: Config, model_id: str, user_id: str) -> SessionRecord:
    record = await config.metadata.get_session(model_id)
    if record is None:
        raise HTTPException(404, "model_id not found")
    if record.user_id != user_id:
        raise HTTPException(403, "Not your model")
    if record.status not in (SessionStatus.ACTIVE, SessionStatus.SUSPENDED):
        raise HTTPException(410, f"Session is {record.status.value}")
    return record


# ─── Service endpoints ────────────────────────────────────────────────────


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/client/config")
@router.post("/client/config")
async def client_config():
    """Return static SDK flow-control knobs.

    These are deploy-time constants — no DB lookup, no auth required.
    Updated by redeploying the gateway, not per-request. The SDK uses
    these to configure concurrency limits and backpressure behavior.
    """
    return {
        "sample_dispatch_bytes_semaphore_size": 128 * 1024 * 1024,
        "inflight_response_bytes_semaphore_size": 256 * 1024 * 1024,
        "max_pipelined_requests": 64,
        "default_poll_interval_ms": 100,
    }


class TelemetryEvent(BaseModel):
    # Alternate shape (internal tooling).
    event_type: Optional[str] = None
    # Tinker SDK 0.18+ shape.
    event: Optional[str] = None
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    event_session_index: Optional[int] = None
    severity: Optional[str] = None
    timestamp: Optional[Any] = None  # str (ISO 8601) or float
    duration: Optional[str] = None
    event_data: dict = Field(default_factory=dict)
    # Common to both shapes.
    session_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)


class TelemetryBatch(BaseModel):
    """SDK 0.18 sends telemetry as a batch envelope, not a bare list."""

    events: list[TelemetryEvent]
    platform: Optional[str] = None
    sdk_version: Optional[str] = None
    session_id: Optional[str] = None


@router.post("/telemetry")
async def telemetry(
    batch: TelemetryBatch,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    events = batch.events
    """Ingest client-side telemetry events.

    SDK reports heartbeats, exceptions, and latency measurements.
    Events are logged via structlog. Extension metrics backends may
    forward events to analytics services.
    """
    import structlog

    telem_logger = structlog.get_logger("hatchery.core.telemetry")
    for evt in events[:100]:
        event_type = evt.event_type or evt.event or evt.event_name or "unknown"
        session_id = evt.session_id or batch.session_id
        props = {**evt.payload, **evt.event_data}
        telem_logger.info(
            "sdk.telemetry",
            user_id=user.user_id,
            event_type=event_type,
            session_id=session_id,
            severity=evt.severity,
            **props,
        )
        config.metrics.increment_counter(
            "sdk_telemetry",
            {"event_type": event_type, "user_id": user.user_id},
        )
    return {"status": "accepted"}


@router.get("/get_server_capabilities")
@router.post("/get_server_capabilities")  # Accept both GET and POST for compat.
async def get_server_capabilities():
    from hatchery.core.fused_losses import (
        _try_import_cce,
        _try_import_liger,
        list_known_architectures,
    )
    from hatchery.core.losses import DECLARED_LOSS_FNS, SUPPORTED_LOSS_FNS
    from hatchery.core.model_registry import (
        _DEFAULT_CONTEXT,
    )

    fused_kernels: list[str] = []
    if _try_import_cce() is not None:
        fused_kernels.append("cce")
    if _try_import_liger() is not None:
        fused_kernels.append("liger")
    fused_kernels.append("chunked")  # always available

    # Build per-model capabilities list matching Tinker's shape.
    supported_models = [
        {
            "model_id": model_prefix,
            "max_context_length": ctx,
            "tokenizer_id": model_prefix,
        }
        for model_prefix, ctx in _DEFAULT_CONTEXT.items()
    ]

    return {
        "supported_loss_fns": list(SUPPORTED_LOSS_FNS),
        "declared_loss_fns": list(DECLARED_LOSS_FNS),
        "supports_forward_backward_custom": True,
        "supports_topk_prompt_logprobs": True,
        "supports_2d_target_tokens": True,
        "supported_models": supported_models,
        "max_rank": 256,
        "futures_api": True,
        "sdk_compat_version": "tinker-0.x",
        "fused_ce_kernels_available": fused_kernels,
        "fused_ce_known_architectures": list_known_architectures(),
    }


@router.post("/create_session")
async def create_session(
    req: CreateSessionRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    return {
        "session_id": f"tinker-sess-{uuid.uuid4().hex}",
        "info_message": "created",
        "warning_message": None,
        "error_message": None,
    }


@router.post("/session_heartbeat")
async def session_heartbeat(
    req: SessionHeartbeatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    return {"session_id": req.session_id, "alive": True}


@router.post("/create_model")
async def create_model(
    req: CreateModelRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    # ``lora_config is None`` (or absent) signals full-parameter
    # fine-tuning, matching Fireworks' implicit pattern.
    is_fp = req.lora_config is None
    if not is_fp and req.lora_config.rank > user.max_rank:
        raise HTTPException(403, f"Max rank for your tier: {user.max_rank}")
    if user.allowed_models is not None and req.base_model not in user.allowed_models:
        raise HTTPException(403, f"Model not allowed: {req.base_model}")
    existing = await config.metadata.list_sessions(
        user_id=user.user_id, status=SessionStatus.ACTIVE
    )
    if len(existing) >= user.max_concurrent_sessions:
        raise HTTPException(429, f"Max {user.max_concurrent_sessions} concurrent sessions")

    model_id = f"mdl_{uuid.uuid4().hex}"
    if is_fp:
        # Full-param ignores target_modules; fill with an empty list to
        # keep the SessionRecord schema satisfied.
        target_modules: list[str] = []
        rank_for_session: Optional[int] = None
        alpha_for_session = 0
        rslora = False
        init_lw = "default"
        lora_dropout = 0.0
    else:
        target_modules = target_modules_for(
            req.base_model,
            train_attn=req.lora_config.train_attn,
            train_mlp=req.lora_config.train_mlp,
            train_unembed=req.lora_config.train_unembed,
        )
        rank_for_session = req.lora_config.rank
        alpha_for_session = req.lora_config.rank * 2
        rslora = req.lora_config.use_rslora
        init_lw = req.lora_config.init_lora_weights
        lora_dropout = req.lora_config.lora_dropout

    record = SessionRecord(
        session_id=model_id,
        user_id=user.user_id,
        base_model=req.base_model,
        lora_rank=rank_for_session,
        lora_alpha=alpha_for_session,
        target_modules=target_modules,
        total_steps=0,
        accum_steps=0,
        created_at=time.time(),
        last_accessed=time.time(),
        status=SessionStatus.ACTIVE,
        state_prefix=f"{config.sessions_prefix}/{model_id}/live_state",
    )
    await config.metadata.create_session(record)
    config.metrics.record_session_event(model_id, "created")

    init_payload = {
        "base_model": req.base_model,
        "rank": rank_for_session,
        "lora_alpha": alpha_for_session,
        "target_modules": target_modules,
        "use_rslora": rslora,
        "init_lora_weights": init_lw,
        "lora_dropout": lora_dropout,
    }
    pre_op_ctx = await run_pre_op_hooks(config, record, user, "init_session", init_payload)
    job = await _enqueue_job(
        config=config,
        session_id=model_id,
        user_id=user.user_id,
        operation="init_session",
        payload=init_payload,
        priority=10,
        required_model=req.base_model,
    )
    result = await config.queue.wait_for_result(job.job_id, timeout=120.0)
    await run_post_op_hooks(config, pre_op_ctx, record, user, result)
    if result.status != JobStatus.COMPLETED:
        await config.metadata.update_session(model_id, status=SessionStatus.FAILED)
        raise HTTPException(500, f"Model init failed: {result.error}")

    # Wrap as an UntypedAPIFuture so the official tinker SDK can
    # chain it through retrieve_future → CreateModelResponse.
    inline = {"type": "create_model", "model_id": model_id}
    resp = _future_response(
        job.job_id, user.user_id, "create_model", model_id=model_id, inline_result=inline
    )
    # Pass through the extra identity fields non-SDK callers rely on.
    resp.update(
        base_model=req.base_model,
        lora_config=req.lora_config.model_dump() if req.lora_config is not None else None,
        session_id=req.session_id,
        model_seq_id=req.model_seq_id,
    )
    return resp


@router.post("/forward_backward")
async def forward_backward(
    req: ForwardBackwardRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    from hatchery.core.losses import DECLARED_LOSS_FNS, SUPPORTED_LOSS_FNS

    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    raw_data = req.forward_backward_input.get("data", [])
    if not raw_data:
        raise HTTPException(400, "data must not be empty")
    loss_fn = req.forward_backward_input.get("loss_fn", "cross_entropy")
    loss_fn_config = req.forward_backward_input.get("loss_fn_config")
    if loss_fn not in SUPPORTED_LOSS_FNS:
        if loss_fn in DECLARED_LOSS_FNS:
            raise HTTPException(
                501,
                f"loss_fn {loss_fn!r} is declared by Tinker but not yet "
                f"implemented server-side. Supported: {list(SUPPORTED_LOSS_FNS)}",
            )
        raise HTTPException(
            400,
            f"unknown loss_fn {loss_fn!r}. Supported: {list(SUPPORTED_LOSS_FNS)}",
        )
    data_items = await _build_data_items(raw_data)

    fb_payload = {
        "data": data_items,
        "loss_fn": loss_fn,
        "loss_fn_config": loss_fn_config,
        "return_per_datum_logprobs": True,
    }
    pre_op_ctx = await run_pre_op_hooks(config, session, user, "forward_backward", fb_payload)
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="forward_backward",
        payload=fb_payload,
        preferred_worker=preferred,
        required_model=session.base_model,
        estimated_duration_ms=session.avg_step_duration_ms,
    )
    return _idempotent_future_response(
        req.model_id,
        req.seq_id,
        job.job_id,
        user.user_id,
        "forward_backward",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.post("/forward_only")
async def forward_only(
    req: ForwardOnlyRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """No-grad forward pass with a custom loss.

    Mirrors ``/forward_backward`` but dispatches to the ``forward_only``
    worker handler — runs under ``torch.no_grad()``, does not accumulate
    grads, does not bump ``accum_steps``, does not persist session state.
    Token counts are still emitted so metering is uniform with
    forward_backward (matches Tinker's server-side metering).
    """
    from hatchery.core.losses import DECLARED_LOSS_FNS, SUPPORTED_LOSS_FNS

    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    raw_data = req.forward_only_input.get("data", [])
    if not raw_data:
        raise HTTPException(400, "data must not be empty")
    loss_fn = req.forward_only_input.get("loss_fn", "cross_entropy")
    loss_fn_config = req.forward_only_input.get("loss_fn_config")
    if loss_fn not in SUPPORTED_LOSS_FNS:
        if loss_fn in DECLARED_LOSS_FNS:
            raise HTTPException(
                501,
                f"loss_fn {loss_fn!r} is declared by Tinker but not yet "
                f"implemented server-side. Supported: {list(SUPPORTED_LOSS_FNS)}",
            )
        raise HTTPException(
            400,
            f"unknown loss_fn {loss_fn!r}. Supported: {list(SUPPORTED_LOSS_FNS)}",
        )
    data_items = await _build_data_items(raw_data)

    fo_payload = {
        "data": data_items,
        "loss_fn": loss_fn,
        "loss_fn_config": loss_fn_config,
    }
    pre_op_ctx = await run_pre_op_hooks(config, session, user, "forward_only", fo_payload)
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="forward_only",
        payload=fo_payload,
        preferred_worker=preferred,
        required_model=session.base_model,
        estimated_duration_ms=session.avg_step_duration_ms,
    )
    return _idempotent_future_response(
        req.model_id,
        req.seq_id,
        job.job_id,
        user.user_id,
        "forward_only",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.post("/forward_backward_custom_step1")
async def forward_backward_custom_step1(
    req: ForwardBackwardRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """First leg of the ``forward_backward_custom`` round trip.

    Returns per-position log-probabilities at the caller-supplied
    targets. The client computes its custom loss on those values,
    back-propagates through them, and sends the resulting
    ``grad_logprobs`` tensor to step2.

    ``custom_id`` must be present in ``forward_backward_input`` — it
    keys the worker-side cache so step2 can replay the same forward.
    Step1 and step2 MUST hit the same worker (we rely on
    ``preferred_worker`` pinning).
    """
    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    raw_data = req.forward_backward_input.get("data", [])
    if not raw_data:
        raise HTTPException(400, "data must not be empty")
    custom_id = req.forward_backward_input.get("custom_id")
    if not custom_id:
        raise HTTPException(400, "forward_backward_input.custom_id required")
    data_items = await _build_data_items(raw_data)
    step1_payload = {"data": data_items, "custom_id": custom_id}
    pre_op_ctx = await run_pre_op_hooks(
        config, session, user, "forward_custom_step1", step1_payload
    )
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="forward_custom_step1",
        payload=step1_payload,
        preferred_worker=preferred,
        required_model=session.base_model,
        estimated_duration_ms=session.avg_step_duration_ms,
    )
    return _future_response(
        job.job_id,
        user.user_id,
        "forward_custom_step1",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.post("/forward_backward_custom_step2")
async def forward_backward_custom_step2(
    req: ForwardBackwardRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """Second leg of the ``forward_backward_custom`` round trip.

    Takes the ``grad_logprobs`` the client produced from step1's
    logprobs and runs the surrogate backward on the server. The
    parameter gradient this writes is identical to what the user's
    custom loss would have produced — see
    :func:`hatchery.core.losses.surrogate_loss_from_grad`.
    """
    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    custom_id = req.forward_backward_input.get("custom_id")
    grad_logprobs = req.forward_backward_input.get("grad_logprobs")
    if not custom_id:
        raise HTTPException(400, "forward_backward_input.custom_id required")
    if grad_logprobs is None:
        raise HTTPException(400, "forward_backward_input.grad_logprobs required")
    step2_payload = {"custom_id": custom_id, "grad_logprobs": grad_logprobs}
    pre_op_ctx = await run_pre_op_hooks(
        config, session, user, "forward_custom_step2", step2_payload
    )
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="forward_custom_step2",
        payload=step2_payload,
        preferred_worker=preferred,
        required_model=session.base_model,
        estimated_duration_ms=session.avg_step_duration_ms,
    )
    return _future_response(
        job.job_id,
        user.user_id,
        "forward_custom_step2",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.post("/forward")
async def forward(
    req: ForwardRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    # Forward-only: SDK ``forward_backward_custom`` calls this first to
    # get per-position logprobs at ``target_tokens``, then sends
    # ``weights = -dC/dlogprobs`` on ``/forward_backward``. We must
    # respect ``target_tokens`` from ``loss_fn_inputs`` (not default to
    # self-prediction) so clients that pre-shift targets get correctly
    # aligned logprobs.
    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    raw_data = req.forward_input.get("data", [])
    if not raw_data:
        raise HTTPException(400, "data must not be empty")
    data_items = await _build_data_items(raw_data)
    fwd_payload = {"data": data_items}
    pre_op_ctx = await run_pre_op_hooks(config, session, user, "forward_logprobs", fwd_payload)
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="forward_logprobs",
        payload=fwd_payload,
        preferred_worker=preferred,
        required_model=session.base_model,
    )
    return _future_response(
        job.job_id,
        user.user_id,
        "forward",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.post("/optim_step")
async def optim_step(
    req: OptimStepRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    opt_payload = {
        "learning_rate": req.adam_params.learning_rate,
        "beta1": req.adam_params.beta1,
        "beta2": req.adam_params.beta2,
        "eps": req.adam_params.eps,
        "weight_decay": req.adam_params.weight_decay,
        "grad_clip_norm": req.adam_params.grad_clip_norm,
    }
    if req.grad_accumulation_normalization is not None:
        opt_payload["grad_accumulation_normalization"] = req.grad_accumulation_normalization
    pre_op_ctx = await run_pre_op_hooks(config, session, user, "optim_step", opt_payload)
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="optim_step",
        payload=opt_payload,
        priority=5,
        preferred_worker=preferred,
        required_model=session.base_model,
    )
    return _idempotent_future_response(
        req.model_id,
        req.seq_id,
        job.job_id,
        user.user_id,
        "optim_step",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.post("/asample")
async def asample(
    req: SampleRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    # The SDK's SamplingClient routes requests through one of:
    #   * ``model_id``
    #   * ``model_path`` — ``tinker://<model_id>/...`` or similar
    #   * ``sampling_session_id`` — returned by
    #     ``save_weights_and_get_sampling_client``; we encode the model
    #     id inside it as ``samp-<model_id>-<seq>-<hash>``.
    #   * ``base_model`` — for untrained base-model sampling (not yet wired)
    model_id = req.model_id
    if model_id is None and req.model_path:
        mp = req.model_path
        if mp.startswith("tinker://"):
            mp = mp.split("://", 1)[1]
        model_id = mp.split("/", 1)[0]
    if model_id is None and req.sampling_session_id:
        sid = req.sampling_session_id
        if sid.startswith("samp-"):
            parts = sid.split("-")
            if len(parts) >= 4:
                model_id = "-".join(parts[1:-2])
        else:
            model_id = sid
    if model_id is None:
        raise HTTPException(400, "model_id, model_path, or sampling_session_id required")
    session = await _owned_session_for_model(config, model_id, user.user_id)
    await _resolve_image_asset_pointers(req.prompt)
    prompt_tokens = _model_input_tokens(req.prompt)
    sample_payload: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "max_tokens": req.sampling_params.max_tokens or 256,
        "temperature": req.sampling_params.temperature,
        "top_p": req.sampling_params.top_p,
        "top_k": req.sampling_params.top_k,
        "n": req.num_samples,
        "seed": req.sampling_params.seed,
        "stop": req.sampling_params.stop,
        "include_prompt_logprobs": bool(req.prompt_logprobs),
        "topk_prompt_logprobs": int(req.topk_prompt_logprobs or 0),
    }
    if req.sampling_params.speculative_decoding is not None:
        sample_payload["speculative_decoding"] = (
            req.sampling_params.speculative_decoding.model_dump(exclude_none=True)
        )
    if req.sampling_params.enable_thinking is not None:
        sample_payload["enable_thinking"] = req.sampling_params.enable_thinking
    if _model_input_has_images(req.prompt):
        sample_payload["images"] = _decode_image_chunks(req.prompt)
    pre_op_ctx = await run_pre_op_hooks(config, session, user, "sample", sample_payload)
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=model_id,
        user_id=user.user_id,
        operation="sample",
        payload=sample_payload,
        preferred_worker=preferred,
        required_model=session.base_model,
    )
    # Sampling is inherently stochastic — never return a cached response.
    # The seq_id idempotency tracker is designed for training ops (where a
    # retry should not double-update). For sampling, every call must
    # produce a fresh result even if the prompt and params are identical.
    return _future_response(
        job.job_id,
        user.user_id,
        "sample",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.post("/save_weights")
async def save_weights_route(
    req: SaveWeightsRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    name = req.path or f"ckpt-{int(time.time())}"
    if name.startswith("tinker://"):
        name = name.split("/")[-1]
    if not _CHECKPOINT_NAME_RE.match(name):
        raise HTTPException(400, f"Invalid checkpoint name: {name!r}")

    payload = {"name": name}
    pre_op_ctx = await run_pre_op_hooks(config, session, user, "save_weights", payload)
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="save_weights",
        payload=payload,
        preferred_worker=preferred,
        required_model=session.base_model,
    )
    result = await config.queue.wait_for_result(job.job_id, timeout=60.0)
    await run_post_op_hooks(config, pre_op_ctx, session, user, result)
    if result.status != JobStatus.COMPLETED:
        raise HTTPException(500, result.error or "save_weights failed")

    from hatchery.core.protocols import CheckpointRecord

    ckpt_id = f"ckpt-{uuid.uuid4().hex[:12]}"
    expires_at = None
    if req.ttl_seconds is not None:
        expires_at = time.time() + req.ttl_seconds
    ckpt_prefix = f"{config.sessions_prefix}/{req.model_id}/checkpoints/{name}"
    await config.metadata.create_checkpoint(
        CheckpointRecord(
            checkpoint_id=ckpt_id,
            session_id=req.model_id,
            user_id=user.user_id,
            name=name,
            checkpoint_type="training",
            created_at=time.time(),
            expires_at=expires_at,
            object_key=ckpt_prefix,
        )
    )
    tinker_path = f"tinker://{req.model_id}/checkpoints/{name}"
    inline = {
        "type": "save_weights",
        "path": tinker_path,
        "checkpoint_id": ckpt_id,
        "checkpoint_type": "training",
        "expires_at": expires_at,
    }
    return _future_response(
        job_id=f"inline-{ckpt_id}",
        user_id=user.user_id,
        operation="save_weights",
        model_id=req.model_id,
        inline_result=inline,
    )


@router.post("/save_weights_for_sampler")
async def save_weights_for_sampler_route(
    req: SaveWeightsForSamplerRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """Make the current session's weights available for sampling.

    When no external sampling backend (vLLM) is configured, the
    adapter is already loaded on the worker — skip the expensive
    checkpoint materialization and just return the model_id as the
    sampling session. The worker's ``/asample`` handler routes
    through the same in-process model.

    When a sampling backend IS configured (e.g. vLLM pool), the
    full checkpoint is materialized and published to the pool.
    """
    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    name = req.path or f"sampler_weights/{uuid.uuid4().hex[:8]}"
    if name.startswith("tinker://"):
        name = name.split("/")[-1]

    has_external_sampler = config.sampling_backend is not None and hasattr(
        config.sampling_backend, "publish_adapter"
    )

    if has_external_sampler:
        payload = {"name": name}
        pre_op_ctx = await run_pre_op_hooks(config, session, user, "save_weights", payload)
        preferred = await _resolve_preferred_worker(config, session)
        job = await _enqueue_job(
            config=config,
            session_id=req.model_id,
            user_id=user.user_id,
            operation="save_weights",
            payload=payload,
            preferred_worker=preferred,
            required_model=session.base_model,
        )
        result = await config.queue.wait_for_result(job.job_id, timeout=120.0)
        await run_post_op_hooks(config, pre_op_ctx, session, user, result)
        if result.status != JobStatus.COMPLETED:
            raise HTTPException(500, result.error or "save_weights_for_sampler failed")

        dst_prefix = f"{config.sessions_prefix}/{req.model_id}/checkpoints/{name}"
        adapter_path = f"{dst_prefix}/lora_weights.pt"
        try:
            await config.sampling_backend.publish_adapter(req.model_id, adapter_path)
        except Exception:  # noqa: BLE001
            pass

    from hatchery.core.protocols import CheckpointRecord

    ckpt_id = f"ckpt-{uuid.uuid4().hex[:12]}"
    ttl = req.ttl_seconds if req.ttl_seconds is not None else 3600
    expires_at = time.time() + ttl
    ckpt_prefix = f"{config.sessions_prefix}/{req.model_id}/checkpoints/{name}"
    await config.metadata.create_checkpoint(
        CheckpointRecord(
            checkpoint_id=ckpt_id,
            session_id=req.model_id,
            user_id=user.user_id,
            name=name,
            checkpoint_type="sampler",
            created_at=time.time(),
            expires_at=expires_at,
            object_key=ckpt_prefix,
        )
    )

    tinker_path = f"tinker://{req.model_id}/sampler_weights/{name}"
    # SDK's ``save_weights_and_get_sampling_client`` sends no ``path`` but
    # does set ``sampling_session_seq_id``; in that mode the response must
    # carry ``sampling_session_id`` (and ``path`` must be None). When the
    # caller provides a ``path`` / ``name`` the response is the opposite:
    # ``path`` populated, no session id.
    if req.path is None and req.sampling_session_seq_id is not None:
        sampling_session_id = (
            f"samp-{req.model_id}-{req.sampling_session_seq_id}-{uuid.uuid4().hex[:8]}"
        )
        inline = {
            "type": "save_weights_for_sampler",
            "path": None,
            "sampling_session_id": sampling_session_id,
            "checkpoint_id": ckpt_id,
            "checkpoint_type": "sampler",
            "expires_at": expires_at,
        }
    else:
        inline = {
            "type": "save_weights_for_sampler",
            "path": tinker_path,
            "sampling_session_id": None,
            "checkpoint_id": ckpt_id,
            "checkpoint_type": "sampler",
            "expires_at": expires_at,
        }
    return _future_response(
        job_id=f"inline-{ckpt_id}",
        user_id=user.user_id,
        operation="save_weights_for_sampler",
        model_id=req.model_id,
        inline_result=inline,
    )


@router.post("/save_state")
async def save_state_route(
    req: SaveStateRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """Save a full training state checkpoint (weights + optimizer + grad_accum).

    Unlike ``save_weights`` (which only copies the LoRA adapter state),
    ``save_state`` copies the entire training state so training can be
    resumed exactly where it left off — including accumulated gradients
    from a partial accumulation cycle and the Adam momentum/variance.
    """
    await _owned_session_for_model(config, req.model_id, user.user_id)
    name = req.path or f"state-{int(time.time())}"
    if name.startswith("tinker://"):
        name = name.split("/")[-1]

    src_prefix = f"{config.sessions_prefix}/{req.model_id}/live_state"
    dst_prefix = f"{config.sessions_prefix}/{req.model_id}/checkpoints/{name}"

    # Materialize LoRA weights into a self-contained snapshot at the
    # checkpoint prefix. The persister picks the appropriate path for
    # whatever state format it writes.
    src = f"{src_prefix}/lora_weights.pt"
    if not await config.objects.exists(src):
        raise HTTPException(409, "No training state to save — run at least one forward_backward.")
    await config.lora_state.materialize(config.objects, src_prefix, dst_prefix)

    # Copy optimizer state (if exists).
    opt_key = f"{src_prefix}/optimizer_state.pt"
    if await config.objects.exists(opt_key):
        await config.objects.copy(opt_key, f"{dst_prefix}/optimizer_state.pt")

    # Copy accumulated gradients (if exists).
    grad_key = f"{src_prefix}/grad_accum.pt"
    if await config.objects.exists(grad_key):
        await config.objects.copy(grad_key, f"{dst_prefix}/grad_accum.pt")

    # Copy session meta.
    meta_key = f"{src_prefix}/session_meta.json"
    if await config.objects.exists(meta_key):
        await config.objects.copy(meta_key, f"{dst_prefix}/session_meta.json")

    # Create checkpoint metadata record.
    from hatchery.core.protocols import CheckpointRecord

    ckpt_id = f"ckpt-{uuid.uuid4().hex[:12]}"
    expires_at = None
    if req.ttl_seconds is not None:
        expires_at = time.time() + req.ttl_seconds
    await config.metadata.create_checkpoint(
        CheckpointRecord(
            checkpoint_id=ckpt_id,
            session_id=req.model_id,
            user_id=user.user_id,
            name=name,
            checkpoint_type="training_state",
            created_at=time.time(),
            expires_at=expires_at,
            object_key=dst_prefix,
        )
    )

    tinker_path = f"tinker://{req.model_id}/checkpoints/{name}"
    inline = {
        "type": "save_state",
        "path": tinker_path,
        "checkpoint_id": ckpt_id,
        "checkpoint_type": "training_state",
        "includes_optimizer": await config.objects.exists(f"{dst_prefix}/optimizer_state.pt"),
        "includes_grad_accum": await config.objects.exists(f"{dst_prefix}/grad_accum.pt"),
        "expires_at": expires_at,
    }
    return _future_response(
        job_id=f"inline-{ckpt_id}",
        user_id=user.user_id,
        operation="save_state",
        model_id=req.model_id,
        inline_result=inline,
    )


@router.post("/load_weights")
async def load_weights_route(
    req: LoadWeightsRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """Resume training from a previously saved checkpoint.

    ``path`` is a ``tinker://`` URI (e.g.,
    ``tinker://<model_id>/checkpoints/<name>``). The worker loads the
    LoRA state dict from the checkpoint and optionally restores the
    optimizer state (if ``optimizer=True`` and the checkpoint includes
    one).
    """
    session = await _owned_session_for_model(config, req.model_id, user.user_id)

    # Resolve the tinker:// path to an object-store prefix. Supports
    # both ``tinker://<model_id>/checkpoints/<name>`` and bare
    # ``<model_id>/checkpoints/<name>`` forms.
    path = req.path
    if path.startswith("tinker://"):
        path = path.split("://", 1)[1]

    # Verify the caller owns the SOURCE checkpoint's session, not just
    # the destination. Without this, a user who knows another user's
    # model_id can load their checkpoint weights.
    source_model_id = path.split("/", 1)[0] if "/" in path else path
    if source_model_id != req.model_id:
        source_session = await config.metadata.get_session(source_model_id)
        if source_session is None:
            raise HTTPException(404, f"Source session {source_model_id!r} not found")
        if source_session.user_id != user.user_id:
            is_public = getattr(source_session, "public", False)
            if not is_public:
                raise HTTPException(403, "Cannot load checkpoint from another user's session")

    ckpt_prefix = f"{config.sessions_prefix}/{path}"
    ckpt_key = f"{ckpt_prefix}/lora_weights.pt"
    if not await config.objects.exists(ckpt_key):
        raise HTTPException(404, f"Checkpoint not found at {req.path!r}")

    lw_payload = {
        "checkpoint_prefix": ckpt_prefix,
        "restore_optimizer": req.optimizer,
    }
    pre_op_ctx = await run_pre_op_hooks(config, session, user, "load_weights", lw_payload)
    preferred = await _resolve_preferred_worker(config, session)
    job = await _enqueue_job(
        config=config,
        session_id=req.model_id,
        user_id=user.user_id,
        operation="load_weights",
        payload=lw_payload,
        preferred_worker=preferred,
        required_model=session.base_model,
    )
    return _future_response(
        job.job_id,
        user.user_id,
        "load_weights",
        pre_op_contexts=pre_op_ctx,
        session=session,
    )


@router.get("/training_runs/{model_id}/checkpoints")
async def list_checkpoints(
    model_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    await _owned_session_for_model(config, model_id, user.user_id)
    prefix = f"{config.sessions_prefix}/{model_id}/checkpoints/"
    keys = await config.objects.list_keys(prefix)
    names = sorted({k[len(prefix) :].split("/", 1)[0] for k in keys if "/" in k[len(prefix) :]})
    checkpoints = []
    for n in names:
        rec = await config.metadata.get_checkpoint(model_id, n)
        ckpt: dict = {
            "checkpoint_id": n,
            "checkpoint_type": rec.checkpoint_type if rec else "training",
            "tinker_path": f"tinker://{model_id}/checkpoints/{n}",
            "path": f"tinker://{model_id}/checkpoints/{n}",
            "time": rec.created_at if rec else None,
            "public": getattr(rec, "public", False) if rec else False,
            "expires_at": rec.expires_at if rec else None,
        }
        checkpoints.append(ckpt)
    return {"model_id": model_id, "checkpoints": checkpoints, "cursor": None}


@router.delete("/training_runs/{model_id}/checkpoints/{checkpoint_id}")
async def delete_checkpoint(
    model_id: str,
    checkpoint_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    await _owned_session_for_model(config, model_id, user.user_id)
    prefix = f"{config.sessions_prefix}/{model_id}/checkpoints/{checkpoint_id}"
    for suffix in (
        "/lora_weights.pt",
        "/optimizer_state.pt",
        "/grad_accum.pt",
        "/session_meta.json",
    ):
        try:
            await config.objects.delete(f"{prefix}{suffix}")
        except Exception:  # noqa: BLE001
            pass
    return {"deleted": True}


@router.post("/unload_model")
async def unload_model(
    model_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """Explicitly evict a model (session) from its worker.

    Terminates the session and cleans up live state. Equivalent to
    ``DELETE /v1/sessions/{session_id}`` on the native API.
    """
    session = await _owned_session_for_model(config, model_id, user.user_id)
    await config.metadata.update_session(model_id, status=SessionStatus.TERMINATED)

    # Best-effort cleanup of live state blobs.
    for suffix in ("lora_weights.pt", "optimizer_state.pt", "grad_accum.pt", "session_meta.json"):
        try:
            await config.objects.delete(f"{session.state_prefix}/{suffix}")
        except Exception:  # noqa: BLE001
            pass

    return {"model_id": model_id, "status": "unloaded"}


class GetInfoRequest(BaseModel):
    model_id: str
    type: Optional[str] = "get_info"


@router.post("/get_info")
async def get_info(
    req: GetInfoRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """Return architecture + tokenizer info for a training-client model.

    The tinker SDK calls this to populate ``get_tokenizer()`` on the
    TrainingClient. Response shape matches ``GetInfoResponse`` in the
    SDK types package: model_id, model_data{arch, model_name,
    tokenizer_id}, is_lora, lora_rank.
    """
    session = await _owned_session_for_model(config, req.model_id, user.user_id)
    # Without a loaded model registry, we report the HF repo id as the
    # tokenizer id — the SDK downloads the tokenizer itself from HF.
    return {
        "type": "get_info",
        "model_id": req.model_id,
        "model_data": {
            "arch": session.base_model,
            "model_name": session.base_model,
            "tokenizer_id": session.base_model,
        },
        "model_name": session.base_model,
        "is_lora": True,
        "lora_rank": session.lora_rank,
    }


class WeightsInfoRequest(BaseModel):
    # The tinker SDK (>=0.18) sends ``tinker_path``; older Hatchery callers
    # and internal scripts use ``path``. Accept either.
    path: Optional[str] = None
    tinker_path: Optional[str] = None

    def resolve_path(self) -> str:
        p = self.tinker_path or self.path
        if not p:
            raise HTTPException(400, "tinker_path (or path) is required")
        return p


@router.post("/weights_info")
async def weights_info(
    req: WeightsInfoRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    """Introspect a checkpoint before loading it.

    Returns the base model, LoRA rank, and which module groups were
    trained. Matches Tinker's ``weights_info`` response shape.

    Unlike the session-scoped endpoints, this one does *not* require the
    underlying session to be ACTIVE/SUSPENDED — the tinker SDK's
    ``create_training_client_from_state*`` flow calls weights_info to
    reconstruct adapter config before creating a fresh session, which
    happens precisely when the original session is long gone.
    """
    tinker_path = req.resolve_path()
    path = tinker_path
    if path.startswith("tinker://"):
        path = path[len("tinker://") :]
    parts = path.split("/", 1)
    if len(parts) < 2:
        raise HTTPException(400, f"Invalid weights path: {tinker_path}")
    model_id = parts[0]

    session = await config.metadata.get_session(model_id)
    if session is None:
        raise HTTPException(404, "model_id not found")
    if session.user_id != user.user_id:
        raise HTTPException(403, "Not your model")

    # Check if the checkpoint exists in the object store.
    ckpt_key = f"{config.sessions_prefix}/{path}/lora_weights.pt"
    if not await config.objects.exists(ckpt_key):
        raise HTTPException(404, f"Weights not found at {tinker_path}")

    # Determine which module groups were trained from target_modules.
    # FP sessions carry an empty target_modules list and lora_rank=None.
    targets = set(session.target_modules)
    is_lora = session.lora_rank is not None
    train_attn = bool(targets & {"q_proj", "k_proj", "v_proj", "o_proj"})
    train_mlp = bool(targets & {"gate_proj", "up_proj", "down_proj"})
    train_unembed = bool(targets & {"lm_head", "embed_tokens"})

    return {
        "base_model": session.base_model,
        "is_lora": is_lora,
        "lora_rank": session.lora_rank,
        "lora_alpha": session.lora_alpha,
        "target_modules": session.target_modules,
        "train_attn": train_attn,
        "train_mlp": train_mlp,
        "train_unembed": train_unembed,
    }


@router.post("/retrieve_future")
async def retrieve_future(
    req: RetrieveFutureRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    config: Config = Depends(get_config),
):
    future_id = req.resolve()
    entry = _futures.lookup(future_id)
    if entry is None:
        raise HTTPException(404, "request_id not found")
    if entry.user_id != user.user_id:
        raise HTTPException(403, "Not your future")

    # Synchronous operations (e.g. create_model) pre-register their
    # response — return it directly without waiting on the queue.
    if entry.inline_result is not None:
        return entry.inline_result

    result = await config.queue.wait_for_result(
        entry.job_id, timeout=config.max_job_timeout_seconds
    )

    # Post-op hooks run after the job completes. Pre-op hooks ran
    # before enqueue; the contexts were stashed in the future entry
    # so post-op processing happens here when the result resolves.
    if entry.pre_op_contexts and entry.session:
        await run_post_op_hooks(config, entry.pre_op_contexts, entry.session, user, result)

    if result.status == JobStatus.FAILED:
        return {
            "type": "request_failed",
            "error": result.error or "unknown error",
            "category": "user_error",
        }
    if result.status == JobStatus.TIMED_OUT:
        queue_depth = await config.queue.get_queue_depth()
        queue_state = "paused_capacity" if queue_depth > 100 else "active"
        return {
            "type": "try_again",
            "request_id": future_id,
            "queue_state": queue_state,
        }

    payload = msgpack.unpackb(result.result, raw=False) if result.result else {}
    wrapped = _wrap_future_result(entry.operation, payload)
    # Attach per-call metrics if the worker reported any. SDK response
    # types carry ``metrics`` as a Dict[str, float]; only merge scalar
    # numeric fields (worker metrics include dicts like ``cost_dimensions``
    # for internal metrics — those must not leak into the SDK shape).
    if result.metrics:
        # SDK's chunked-fwdbwd reducer requires keys in ``name:reduction``
        # form. Worker-emitted metrics without a colon get a default
        # ":mean" suffix, and non-scalar entries (e.g. ``cost_dimensions``
        # dict used for internal metrics) are dropped.
        scalar_metrics: dict[str, float] = {}
        for k, v in result.metrics.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            key = k if ":" in k else f"{k}:mean"
            scalar_metrics[key] = float(v)
        if scalar_metrics:
            wrapped.setdefault("metrics", {}).update(scalar_metrics)
    return wrapped


def _wrap_future_result(operation: str, payload: dict) -> dict:
    """Coerce our internal result dict into the tinker response type."""
    if operation == "forward_custom_step1":
        return {
            "logprobs": payload.get("logprobs"),
            "shape": payload.get("shape"),
            "num_tokens": payload.get("num_tokens", 0),
        }
    if operation == "forward_custom_step2":
        return {
            "surrogate": payload.get("surrogate", 0.0),
            "num_tokens": payload.get("num_tokens", 0),
            "accum_steps": payload.get("accum_steps", 0),
            "metrics": {
                "surrogate": payload.get("surrogate", 0.0),
                "num_tokens": payload.get("num_tokens", 0),
            },
        }
    if operation == "forward_only":
        loss = float(payload.get("loss", 0.0))
        num_tokens = int(payload.get("num_tokens", 0))
        # Response shape mirrors forward_backward's loss_fn_outputs
        # envelope, minus per-datum logprobs (forward_only doesn't
        # emit them) and minus the accum_steps-derived fields.
        return {
            "loss_fn_output_type": "cross_entropy",
            "loss_fn_outputs": [
                {
                    "logprobs": {
                        "data": [],
                        "dtype": "float32",
                        "shape": [0],
                    }
                }
            ],
            "metrics": {
                "loss:mean": loss,
                "loss:sum": float(loss) * float(num_tokens),
                "num_tokens:sum": float(num_tokens),
            },
        }
    if operation in ("forward_backward", "forward"):
        loss = float(payload.get("loss", 0.0))
        num_tokens = int(payload.get("num_tokens", 0))
        # Per-datum logprobs (list of per-token lists). If the worker
        # didn't emit them, fall back to a single empty entry so the
        # loss_fn_outputs shape is still list-of-dicts.
        per_datum = payload.get("per_datum_logprobs") or payload.get("logprobs")
        loss_fn_outputs: list[dict] = []
        if (
            per_datum
            and isinstance(per_datum, list)
            and per_datum
            and isinstance(per_datum[0], list)
        ):
            # List-of-lists: one row per datum.
            for row in per_datum:
                loss_fn_outputs.append(
                    {
                        "logprobs": {
                            "data": [float(x) for x in row],
                            "dtype": "float32",
                            "shape": [len(row)],
                        },
                    }
                )
        else:
            # No per-datum logprobs available — emit a single empty tensor
            # so the SDK schema is still satisfied. Cookbook mean-NLL will
            # be meaningless here but the API validates.
            loss_fn_outputs.append(
                {
                    "logprobs": {
                        "data": [],
                        "dtype": "float32",
                        "shape": [0],
                    },
                }
            )
        return {
            "loss_fn_output_type": "cross_entropy",
            "loss_fn_outputs": loss_fn_outputs,
            # SDK chunked-fwdbwd reducer requires ``name:reduction`` keys
            # (e.g. "loss:mean"). See tinker.lib.chunked_fwdbwd_helpers.
            # fw-ai's sft_loop reads ``loss:sum`` directly (not mean*count),
            # so emit both reductions for compatibility.
            "metrics": {
                "loss:mean": loss,
                "loss:sum": float(loss) * float(num_tokens),
                "num_tokens:sum": float(num_tokens),
                "accum_steps:mean": float(payload.get("accum_steps", 0)),
            },
        }
    if operation == "optim_step":
        return {
            "type": "optim_step",
            "metrics": {
                "step": float(payload.get("step", 0)),
                "learning_rate": float(payload.get("learning_rate", 0.0)),
            },
        }
    if operation == "load_weights":
        return {
            "path": payload.get("path"),
            "type": "load_weights",
        }
    if operation == "save_weights":
        return {
            "type": "save_weights",
            "path": payload.get("path", ""),
        }
    if operation == "save_weights_for_sampler":
        return {
            "type": "save_weights_for_sampler",
            "path": payload.get("path", ""),
        }
    if operation == "save_state":
        return {
            "type": "save_state",
            "path": payload.get("path", ""),
        }
    if operation == "unload_model":
        return {
            "type": "unload_model",
            "model_id": payload.get("model_id", ""),
        }
    if operation == "sample":
        raw_seqs = payload.get("sequences", [])
        stop_reasons = payload.get("stop_reasons") or []
        seq_logprobs = payload.get("sequence_logprobs") or []
        sequences = []
        for i, seq in enumerate(raw_seqs):
            stop_reason = (
                stop_reasons[i]
                if i < len(stop_reasons) and stop_reasons[i] in ("length", "stop")
                else "length"
            )
            lp = seq_logprobs[i] if i < len(seq_logprobs) else None
            sequences.append(
                {
                    "tokens": list(seq),
                    "stop_reason": stop_reason,
                    # Per-token logprobs from the rollout policy. The cookbook
                    # RL recipes use these as ``old_logprobs`` in IS/PPO.
                    "logprobs": [float(x) for x in lp] if lp is not None else None,
                }
            )
        return {
            "sequences": sequences,
            "sequence_logprobs": seq_logprobs,
            "prompt_logprobs": payload.get("prompt_logprobs"),
            "topk_prompt_logprobs": payload.get("topk_prompt_logprobs"),
            "spec_decoding_metadata": payload.get("spec_decoding_metadata"),
        }
    return payload
