# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""FastAPI gateway — stateless control plane for the platform."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import msgpack
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from hatchery.core.config import Config, build_core_config
from hatchery.core.plugins import (
    GATEWAY_PLUGINS,
    run_post_op_hooks,
    run_pre_op_hooks,
    sign_payload,
)
from hatchery.core.protocols import (
    AuthenticatedUser,
    JobRecord,
    JobStatus,
    QueuedJob,
    SessionRecord,
    SessionStatus,
)

logger = structlog.get_logger("hatchery.core.gateway")


# ─── Shared state ─────────────────────────────────────────────────────────

_config: Optional[Config] = None


def set_config(config: Config) -> None:
    """Install a ``Config`` (or an extension subclass thereof) for the running app.

    Tests and the unified launcher use this to inject pre-wired backends
    instead of the env-driven factory.
    """
    global _config
    _config = config


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("Platform not initialized. Call set_config() first.")
    return _config


async def get_current_user(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    config: Config = Depends(get_config),
) -> AuthenticatedUser:
    # Accept either `Authorization: Bearer <tok>` or `X-API-Key: <tok>`.
    # The tinker SDK sends X-API-Key; our own tooling uses Bearer.
    if x_api_key:
        token = x_api_key
    elif authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() not in ("bearer", "token"):
            raise HTTPException(401, "Invalid auth scheme")
    else:
        raise HTTPException(401, "Missing auth credentials")

    # Token-auth plugins (e.g. an extension-provided JWT verifier) run
    # first. Each may return an AuthenticatedUser or None to fall through.
    for hook in GATEWAY_PLUGINS.token_auth:
        user = await hook(token, config)
        if user is not None:
            return user

    # API key auth (DB/hash lookup).
    user = await config.auth.authenticate(token)
    if user is None:
        raise HTTPException(401, "Invalid or expired token")
    return user


# ─── Lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config
    if _config is None:
        _config = build_core_config()
    await _config.metadata.initialize()
    await _config.queue.initialize()
    for hook in GATEWAY_PLUGINS.lifespan_startup:
        await hook(_config)
    try:
        yield
    finally:
        for hook in GATEWAY_PLUGINS.lifespan_shutdown:
            try:
                await hook(_config)
            except Exception:  # noqa: BLE001
                logger.warning("lifespan.shutdown.hook_failed", exc_info=True)
        await _config.queue.close()
        await _config.metadata.close()


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create a FastAPI app. Pass ``config`` to inject backends directly."""
    if config is not None:
        set_config(config)

    app = FastAPI(
        title="Tinker-Compatible Training Platform",
        version="0.3.0",
        lifespan=lifespan,
    )
    # Order matters: inner-most is added first. Request flow on the wire
    # is (compressed) → decompress → (msgpack) → unpack → app. Response
    # flow is app → (json) → pack msgpack → compress → wire. So msgpack
    # must be added BEFORE compression.
    _install_msgpack_middleware(app)
    _install_compression_middleware(app)
    _register_routes(app)

    # Mount the tinker-compat router so clients of the official tinker SDK
    # can hit /api/v1/* without code changes.
    from hatchery.core.tinker_compat import router as tinker_router

    app.include_router(tinker_router)

    # Plugin-registered routers (e.g. extension internal worker routes,
    # billing endpoints, JWT /auth/token).
    for router in GATEWAY_PLUGINS.routers:
        app.include_router(router)
    return app


# ─── Request body size limits ────────────────────────────────────────────

_MAX_REQUEST_BODY = int(os.environ.get("HATCHERY_MAX_REQUEST_BODY_BYTES", str(50 * 1024 * 1024)))
_MAX_DECOMPRESSED_BODY = int(
    os.environ.get("HATCHERY_MAX_DECOMPRESSED_BODY_BYTES", str(200 * 1024 * 1024))
)

# ─── Msgpack transport middleware ─────────────────────────────────────────


def _install_msgpack_middleware(app: FastAPI) -> None:
    """Negotiate msgpack request/response bodies transparently.

    Clients opt in with headers:
      - ``Content-Type: application/msgpack`` — the request body is
        msgpack; the middleware unpacks it to a Python object and
        re-encodes as JSON before handing off to the FastAPI routes.
      - ``Accept: application/msgpack`` — the middleware buffers the
        JSON response body and repacks it as msgpack before returning.

    Pure JSON clients (including the official tinker SDK) are untouched
    — both of the above negotiations are no-ops when neither header is
    present. Non-JSON response bodies (binary downloads, text) are also
    passed through.

    Why this lives in the gateway: forward_backward and compute_logprobs
    payloads carry per-datum tensors as base64-encoded strings inside
    JSON. Msgpack's native ``bin`` type is ~33% smaller and decodes
    faster. Stacked with response compression (which sees the already
    smaller msgpack bytes) the total wire savings are compounding.

    Configuration: set ``HATCHERY_MSGPACK_ENABLED=0`` to disable the
    middleware entirely (e.g. if an upstream proxy owns content
    negotiation, or for debugging with wire captures).
    """
    import json
    import os

    import msgpack as _mp
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    if os.environ.get("HATCHERY_MSGPACK_ENABLED", "1") == "0":
        return

    CT_JSON = b"application/json"
    CT_MSGPACK = b"application/msgpack"

    def _header(headers: list[tuple[bytes, bytes]], name: bytes) -> Optional[bytes]:
        for k, v in headers:
            if k.lower() == name:
                return v
        return None

    async def _reject(send: Send, status: int, detail: str) -> None:
        body = b'{"detail":"' + detail.encode("utf-8") + b'"}'
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    class MsgpackTransportMiddleware:
        def __init__(self, inner: ASGIApp) -> None:
            self.inner = inner

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self.inner(scope, receive, send)
                return

            headers = scope.get("headers", [])
            req_ct = (_header(headers, b"content-type") or b"").lower()
            req_accept = (_header(headers, b"accept") or b"").lower()
            req_is_mp = req_ct.split(b";", 1)[0].strip() == CT_MSGPACK
            resp_wants_mp = CT_MSGPACK in req_accept

            if not req_is_mp and not resp_wants_mp:
                await self.inner(scope, receive, send)
                return

            eff_scope: Scope = scope
            eff_receive: Receive = receive

            # ── Request side ──────────────────────────────────────
            if req_is_mp:
                body = bytearray()
                more = True
                while more:
                    m = await receive()
                    if m["type"] != "http.request":
                        await self.inner(scope, receive, send)
                        return
                    body.extend(m.get("body", b""))
                    if len(body) > _MAX_REQUEST_BODY:
                        await _reject(send, 413, "request body too large")
                        return
                    more = bool(m.get("more_body", False))
                try:
                    obj = _mp.unpackb(bytes(body), raw=False)
                except Exception as exc:  # noqa: BLE001
                    await _reject(send, 400, f"invalid msgpack body: {exc}")
                    return
                try:
                    json_body = json.dumps(obj).encode("utf-8")
                except Exception as exc:  # noqa: BLE001
                    await _reject(send, 400, f"msgpack body not JSON-serializable: {exc}")
                    return
                new_headers: list[tuple[bytes, bytes]] = []
                for k, v in headers:
                    kl = k.lower()
                    if kl in (b"content-type", b"content-length"):
                        continue
                    new_headers.append((k, v))
                new_headers.append((b"content-type", CT_JSON))
                new_headers.append((b"content-length", str(len(json_body)).encode("ascii")))
                eff_scope = dict(scope)
                eff_scope["headers"] = new_headers

                replayed = False

                async def _replay() -> Message:
                    nonlocal replayed
                    if replayed:
                        return {"type": "http.disconnect"}
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": json_body,
                        "more_body": False,
                    }

                eff_receive = _replay

            # ── Response side ─────────────────────────────────────
            if not resp_wants_mp:
                await self.inner(eff_scope, eff_receive, send)
                return

            start_msg: Optional[Message] = None
            buf = bytearray()

            async def wrapped_send(message: Message) -> None:
                nonlocal start_msg
                mt = message["type"]
                if mt == "http.response.start":
                    start_msg = message
                    return
                if mt != "http.response.body":
                    await send(message)
                    return
                buf.extend(message.get("body", b""))
                if message.get("more_body", False):
                    return
                assert start_msg is not None
                resp_headers = list(start_msg.get("headers", []))
                resp_ct = (_header(resp_headers, b"content-type") or b"").lower()
                resp_ct_main = resp_ct.split(b";", 1)[0].strip()
                body_bytes = bytes(buf)
                # Only repack JSON bodies. Non-JSON (or empty) passes
                # through untouched.
                if resp_ct_main != CT_JSON or not body_bytes:
                    await send(start_msg)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": body_bytes,
                            "more_body": False,
                        }
                    )
                    return
                try:
                    obj = json.loads(body_bytes)
                    new_body = _mp.packb(obj, use_bin_type=True)
                except Exception:  # noqa: BLE001
                    logger.warning("msgpack_middleware.repack_failed", exc_info=True)
                    await send(start_msg)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": body_bytes,
                            "more_body": False,
                        }
                    )
                    return
                new_headers2: list[tuple[bytes, bytes]] = []
                for k, v in resp_headers:
                    kl = k.lower()
                    if kl in (b"content-type", b"content-length"):
                        continue
                    new_headers2.append((k, v))
                new_headers2.append((b"content-type", CT_MSGPACK))
                new_headers2.append((b"content-length", str(len(new_body)).encode("ascii")))
                new_start = dict(start_msg)
                new_start["headers"] = new_headers2
                await send(new_start)
                await send(
                    {
                        "type": "http.response.body",
                        "body": new_body,
                        "more_body": False,
                    }
                )

            await self.inner(eff_scope, eff_receive, wrapped_send)

    app.add_middleware(MsgpackTransportMiddleware)


# ─── Compression middleware ───────────────────────────────────────────────


def _install_compression_middleware(app: FastAPI) -> None:
    """Wire optional response compression + request decompression.

    Supports three encodings with content negotiation driven by the
    client's ``Accept-Encoding`` header: ``zstd`` (preferred for binary
    msgpack / TensorData blobs — level-3 is a better size/CPU trade-off
    than gzip on float32 arrays), ``br`` (brotli, good on JSON text),
    and ``gzip`` (universal fallback). Selection order is
    ``zstd > br > gzip > identity``; availability is gated on whether
    the optional ``zstandard`` / ``brotli`` packages are importable.

    Configuration via env vars (read at app-construction time):

    - ``HATCHERY_COMPRESS_RESPONSES`` (alias ``HATCHERY_GZIP_RESPONSES``,
      default ``1``) — compress responses when the client advertises a
      supported encoding and the body exceeds
      ``HATCHERY_COMPRESS_MIN_SIZE`` (alias ``HATCHERY_GZIP_MIN_SIZE``,
      default 1024 bytes).
    - ``HATCHERY_DECOMPRESS_REQUESTS`` (alias ``HATCHERY_GZIP_REQUESTS``,
      default ``1``) — transparently decode incoming request bodies
      that carry ``Content-Encoding: gzip | br | zstd``.
    - ``HATCHERY_COMPRESS_ZSTD_LEVEL`` (default ``3``) — zstd compression
      level. 3 is the sweet spot for float32 logprob arrays.
    - ``HATCHERY_COMPRESS_BROTLI_QUALITY`` (default ``4``) — brotli quality
      (0-11). 4 balances JSON ratio against CPU time.

    Both defaults are enabled because the cost is marginal (~100µs for
    a small JSON response) and the wins at RL scale are real: a
    forward_backward payload with per-datum logprobs from a
    group-size-16 rollout compresses ~10× with zstd.

    Deployment note: if a reverse proxy (nginx, envoy, an API gateway)
    is already handling compression, set both env vars to ``0`` to skip
    redundant work.
    """
    import gzip
    import os

    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    try:
        import brotli as _brotli  # type: ignore
    except ImportError:  # pragma: no cover — optional dep
        _brotli = None
    try:
        import zstandard as _zstd  # type: ignore
    except ImportError:  # pragma: no cover — optional dep
        _zstd = None

    def _env(new_name: str, old_name: str, default: str) -> str:
        """Look up an env var, falling back to an alias."""
        v = os.environ.get(new_name)
        if v is not None:
            return v
        return os.environ.get(old_name, default)

    responses_on = _env("HATCHERY_COMPRESS_RESPONSES", "HATCHERY_GZIP_RESPONSES", "1") != "0"
    requests_on = _env("HATCHERY_DECOMPRESS_REQUESTS", "HATCHERY_GZIP_REQUESTS", "1") != "0"
    min_size = int(_env("HATCHERY_COMPRESS_MIN_SIZE", "HATCHERY_GZIP_MIN_SIZE", "1024"))
    zstd_level = int(os.environ.get("HATCHERY_COMPRESS_ZSTD_LEVEL", "3"))
    brotli_quality = int(os.environ.get("HATCHERY_COMPRESS_BROTLI_QUALITY", "4"))

    def _parse_accept_encoding(raw: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            name, *params = part.split(";")
            name = name.strip().lower()
            q = 1.0
            for p in params:
                p = p.strip()
                if p.startswith("q="):
                    try:
                        q = float(p[2:])
                    except ValueError:
                        pass
            out[name] = q
        return out

    def _negotiate(accept_enc: str) -> Optional[str]:
        """Pick the best encoding. None = send identity."""
        if not accept_enc:
            return None
        enc = _parse_accept_encoding(accept_enc)
        wildcard_q = enc.get("*", 0.0)
        for name, available in (
            ("zstd", _zstd is not None),
            ("br", _brotli is not None),
            ("gzip", True),
        ):
            if not available:
                continue
            q = enc.get(name, wildcard_q)
            if q > 0.0:
                return name
        return None

    def _compress(encoding: str, body: bytes) -> bytes:
        if encoding == "gzip":
            return gzip.compress(body)
        if encoding == "br":
            assert _brotli is not None
            return _brotli.compress(body, quality=brotli_quality)
        if encoding == "zstd":
            assert _zstd is not None
            return _zstd.ZstdCompressor(level=zstd_level).compress(body)
        raise ValueError(f"unsupported encoding: {encoding}")

    def _decompress(encoding: str, body: bytes) -> bytes:
        if encoding == "gzip":
            return gzip.decompress(body)
        if encoding == "br":
            if _brotli is None:
                raise ValueError("brotli not installed")
            return _brotli.decompress(body)
        if encoding == "zstd":
            if _zstd is None:
                raise ValueError("zstandard not installed")
            return _zstd.ZstdDecompressor().decompress(body)
        raise ValueError(f"unsupported encoding: {encoding}")

    # ── Response compression ──────────────────────────────────────────
    if responses_on:

        class ResponseCompressionMiddleware:
            def __init__(self, inner: ASGIApp) -> None:
                self.inner = inner

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                if scope["type"] != "http":
                    await self.inner(scope, receive, send)
                    return

                accept_enc = ""
                for k, v in scope.get("headers", []):
                    if k.lower() == b"accept-encoding":
                        accept_enc = v.decode("latin-1")
                        break

                chosen = _negotiate(accept_enc)
                if chosen is None:
                    await self.inner(scope, receive, send)
                    return

                buffered_start: Optional[Message] = None
                buffered_body = bytearray()

                async def wrapped_send(message: Message) -> None:
                    nonlocal buffered_start
                    mtype = message["type"]
                    if mtype == "http.response.start":
                        buffered_start = message
                        return
                    if mtype != "http.response.body":
                        await send(message)
                        return
                    buffered_body.extend(message.get("body", b""))
                    if message.get("more_body", False):
                        return
                    # Final chunk — decide and flush.
                    assert buffered_start is not None
                    orig_headers = list(buffered_start.get("headers", []))
                    existing_enc: Optional[bytes] = None
                    for k, v in orig_headers:
                        if k.lower() == b"content-encoding":
                            existing_enc = v
                            break
                    body_bytes = bytes(buffered_body)
                    if existing_enc is not None or len(body_bytes) < min_size:
                        # Pass-through: already encoded by the app, or
                        # too small to bother.
                        await send(buffered_start)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": body_bytes,
                                "more_body": False,
                            }
                        )
                        return

                    try:
                        compressed = _compress(chosen, body_bytes)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "compression_middleware.compress_failed",
                            encoding=chosen,
                            exc_info=True,
                        )
                        await send(buffered_start)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": body_bytes,
                                "more_body": False,
                            }
                        )
                        return

                    new_headers: list[tuple[bytes, bytes]] = []
                    vary_seen = False
                    for k, v in orig_headers:
                        kl = k.lower()
                        if kl == b"content-length":
                            continue
                        if kl == b"vary":
                            # Append Accept-Encoding to existing Vary header.
                            existing = v.decode("latin-1")
                            tokens = {t.strip().lower() for t in existing.split(",")}
                            if "accept-encoding" not in tokens:
                                existing = existing + ", Accept-Encoding"
                            new_headers.append((b"vary", existing.encode("latin-1")))
                            vary_seen = True
                            continue
                        new_headers.append((k, v))
                    new_headers.append((b"content-encoding", chosen.encode("latin-1")))
                    new_headers.append((b"content-length", str(len(compressed)).encode("ascii")))
                    if not vary_seen:
                        new_headers.append((b"vary", b"Accept-Encoding"))
                    new_start = dict(buffered_start)
                    new_start["headers"] = new_headers
                    await send(new_start)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": compressed,
                            "more_body": False,
                        }
                    )

                await self.inner(scope, receive, wrapped_send)

        app.add_middleware(ResponseCompressionMiddleware)

    # ── Request decompression ─────────────────────────────────────────
    if requests_on:

        class RequestDecompressionMiddleware:
            def __init__(self, inner: ASGIApp) -> None:
                self.inner = inner

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                if scope["type"] != "http":
                    await self.inner(scope, receive, send)
                    return
                request = Request(scope)
                encoding = request.headers.get("content-encoding", "").lower().strip()
                if encoding in ("", "identity"):
                    await self.inner(scope, receive, send)
                    return
                if encoding not in ("gzip", "br", "zstd"):
                    # Unknown encoding — let the app decide (likely 415).
                    await self.inner(scope, receive, send)
                    return
                if encoding == "br" and _brotli is None:
                    await self._reject(send, 415, "brotli not supported on this server")
                    return
                if encoding == "zstd" and _zstd is None:
                    await self._reject(send, 415, "zstd not supported on this server")
                    return

                # Drain the full body.
                body = bytearray()
                more_body = True
                while more_body:
                    message = await receive()
                    if message["type"] != "http.request":
                        await self.inner(scope, receive, send)
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > _MAX_REQUEST_BODY:
                        await self._reject(send, 413, "compressed request body too large")
                        return
                    more_body = bool(message.get("more_body", False))
                try:
                    decompressed = _decompress(encoding, bytes(body))
                except Exception as exc:  # noqa: BLE001
                    await self._reject(send, 400, f"invalid {encoding} request body: {exc}")
                    return
                if len(decompressed) > _MAX_DECOMPRESSED_BODY:
                    await self._reject(send, 413, "decompressed request body too large")
                    return

                new_headers: list[tuple[bytes, bytes]] = []
                for k, v in scope.get("headers", []):
                    kl = k.lower()
                    if kl == b"content-encoding" or kl == b"content-length":
                        continue
                    new_headers.append((k, v))
                new_headers.append((b"content-length", str(len(decompressed)).encode("ascii")))
                new_scope = dict(scope)
                new_scope["headers"] = new_headers

                sent = False

                async def replay_receive() -> Message:
                    nonlocal sent
                    if sent:
                        return {"type": "http.disconnect"}
                    sent = True
                    return {
                        "type": "http.request",
                        "body": decompressed,
                        "more_body": False,
                    }

                await self.inner(new_scope, replay_receive, send)

            async def _reject(self, send: Send, status: int, detail: str) -> None:
                body = b'{"detail":"' + detail.encode("utf-8") + b'"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False,
                    }
                )

        app.add_middleware(RequestDecompressionMiddleware)


# ─── Request / response models ────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    base_model: str
    rank: Optional[int] = Field(32, ge=1, le=256)
    lora_alpha: Optional[int] = None
    target_modules: Optional[list[str]] = None
    use_rslora: bool = False
    init_lora_weights: str = "default"
    lora_dropout: float = Field(0.0, ge=0.0, le=1.0)


class CreateSessionResponse(BaseModel):
    session_id: str
    base_model: str
    status: str


class ResumeSessionResponse(BaseModel):
    session_id: str
    base_model: str
    rank: Optional[int] = None
    total_steps: int
    accum_steps: int
    created_at: float
    avg_step_duration_ms: Optional[float] = None


class CheckpointRequest(BaseModel):
    session_id: str
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_.\-]{1,128}$")


# ─── Route registration ───────────────────────────────────────────────────


def _register_routes(app: FastAPI) -> None:
    @app.get("/v1/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/sessions", response_model=CreateSessionResponse)
    async def create_session(
        req: CreateSessionRequest,
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
    ):
        # Enforce quotas.
        existing = await config.metadata.list_sessions(
            user_id=user.user_id, status=SessionStatus.ACTIVE
        )
        if len(existing) >= user.max_concurrent_sessions:
            raise HTTPException(429, f"Max {user.max_concurrent_sessions} concurrent sessions")
        # ``rank is None`` ⇒ full-parameter fine-tuning (Fireworks-style
        # implicit signal). Worker dispatches on ``payload["rank"] is
        # None`` and skips PEFT wrapping entirely.
        if req.rank is not None and req.rank > user.max_rank:
            raise HTTPException(403, f"Max rank for your tier: {user.max_rank}")
        if user.allowed_models is not None and req.base_model not in user.allowed_models:
            raise HTTPException(403, f"Model not allowed for your tier: {req.base_model}")

        # Resolve model ID (handles `:peft:<context>` long-context variants).
        from hatchery.core.model_registry import resolve_model_id

        resolved = resolve_model_id(req.base_model)

        session_id = str(uuid.uuid4())
        if req.target_modules is not None:
            target_modules = req.target_modules
        else:
            from hatchery.core.lora_target_modules import target_modules_for

            target_modules = target_modules_for(resolved.base_model)
        # For full-param, the worker ignores lora_alpha; default to 0
        # so the gateway record carries a defined integer rather than
        # leaving the field None and tripping downstream consumers.
        lora_alpha = req.lora_alpha or (req.rank if req.rank is not None else 0)

        record = SessionRecord(
            session_id=session_id,
            user_id=user.user_id,
            base_model=resolved.base_model,
            lora_rank=req.rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            total_steps=0,
            accum_steps=0,
            created_at=time.time(),
            last_accessed=time.time(),
            status=SessionStatus.ACTIVE,
            state_prefix=f"{config.sessions_prefix}/{session_id}/live_state",
        )
        await config.metadata.create_session(record)
        config.metrics.record_session_event(session_id, "created")

        # Fast-fail when workers ARE registered but none serve the
        # requested model. Without this, the gateway enqueues into
        # Redis and silently waits 120s for a worker that may have
        # crashed or never booted — the caller sees a mysterious
        # timeout instead of a clear error.
        #
        # When list_workers() is empty (dev/test, or no worker has
        # ever registered), skip the check — the queue is the only
        # routing mechanism and we rely on wait_for_result timeout.
        workers = await config.compute.list_workers()
        if workers:
            model_ready = any(
                w.status not in ("offline", "draining") and resolved.base_model in w.loaded_models
                for w in workers
            )
            if not model_ready:
                await config.metadata.update_session(session_id, status=SessionStatus.FAILED)
                available = sorted({m for w in workers for m in w.loaded_models})
                raise HTTPException(
                    503,
                    f"No worker available for model {resolved.base_model!r}. "
                    f"Workers registered: {len(workers)}, "
                    f"available models: {available}. "
                    f"The worker may still be booting — retry in 30-60s.",
                )

        job = await _enqueue_job(
            config=config,
            session_id=session_id,
            user_id=user.user_id,
            operation="init_session",
            payload={
                "base_model": resolved.base_model,
                "rank": req.rank,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
                "use_rslora": req.use_rslora,
                "init_lora_weights": req.init_lora_weights,
                "lora_dropout": req.lora_dropout,
            },
            priority=10,
            required_model=resolved.base_model,
            required_cp_degree=resolved.required_cp_degree,
        )

        result = await config.queue.wait_for_result(job.job_id, timeout=120.0)
        await config.metadata.update_job(
            job.job_id,
            status=result.status,
            completed_at=time.time(),
            error_message=result.error,
        )
        if result.status != JobStatus.COMPLETED:
            await config.metadata.update_session(session_id, status=SessionStatus.FAILED)
            raise HTTPException(500, f"Session init failed: {result.error}")

        return CreateSessionResponse(
            session_id=session_id,
            base_model=req.base_model,
            status="active",
        )

    @app.get("/v1/sessions/{session_id}")
    async def get_session(
        session_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
    ):
        record = await _get_owned_session(config, session_id, user.user_id)
        return {
            "session_id": record.session_id,
            "base_model": record.base_model,
            "rank": record.lora_rank,
            "lora_alpha": record.lora_alpha,
            "target_modules": record.target_modules,
            "total_steps": record.total_steps,
            "accum_steps": record.accum_steps,
            "created_at": record.created_at,
            "last_accessed": record.last_accessed,
            "status": record.status.value,
            "avg_step_duration_ms": record.avg_step_duration_ms,
            "total_tokens_processed": record.total_tokens_processed,
        }

    @app.get("/v1/sessions")
    async def list_sessions(
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
        limit: int = 100,
        offset: int = 0,
    ):
        sessions = await config.metadata.list_sessions(
            user_id=user.user_id, limit=limit, offset=offset
        )
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "base_model": s.base_model,
                    "rank": s.lora_rank,
                    "total_steps": s.total_steps,
                    "status": s.status.value,
                    "created_at": s.created_at,
                    "last_accessed": s.last_accessed,
                }
                for s in sessions
            ]
        }

    @app.post("/v1/sessions/{session_id}/resume", response_model=ResumeSessionResponse)
    async def resume_session(
        session_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
    ):
        record = await config.metadata.get_session(session_id)
        if record is None:
            raise HTTPException(404, "Session not found")
        if record.user_id != user.user_id:
            raise HTTPException(403, "Not your session")
        if record.status == SessionStatus.TERMINATED:
            raise HTTPException(410, "Session was terminated")

        await config.metadata.update_session(
            session_id, status=SessionStatus.ACTIVE, last_accessed=time.time()
        )
        config.metrics.record_session_event(session_id, "resumed")

        return ResumeSessionResponse(
            session_id=record.session_id,
            base_model=record.base_model,
            rank=record.lora_rank,
            total_steps=record.total_steps,
            accum_steps=record.accum_steps,
            created_at=record.created_at,
            avg_step_duration_ms=record.avg_step_duration_ms,
        )

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(
        session_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
    ):
        await _get_owned_session(config, session_id, user.user_id, accept_suspended=True)
        await config.metadata.update_session(session_id, status=SessionStatus.TERMINATED)
        config.metrics.record_session_event(session_id, "terminated")

        # Optional best-effort cleanup of live state.
        for suffix in (
            "lora_weights.pt",
            "optimizer_state.pt",
            "grad_accum.pt",
            "session_meta.json",
        ):
            try:
                await config.objects.delete(
                    f"{config.sessions_prefix}/{session_id}/live_state/{suffix}"
                )
            except Exception:  # noqa: BLE001
                pass

        return {"session_id": session_id, "status": "terminated"}

    @app.post("/v1/save_weights")
    async def save_weights(
        req: CheckpointRequest,
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
    ):
        await _get_owned_session(config, req.session_id, user.user_id)
        src = f"{config.sessions_prefix}/{req.session_id}/live_state/lora_weights.pt"
        dst = f"{config.sessions_prefix}/{req.session_id}/checkpoints/{req.name}/lora_weights.pt"
        if not await config.objects.exists(src):
            raise HTTPException(
                409,
                "No live weights to checkpoint — run a forward_backward+optim_step first.",
            )
        await config.objects.copy(src, dst)
        return {"path": f"tinker://{req.session_id}/checkpoints/{req.name}"}

    @app.get("/v1/sessions/{session_id}/checkpoints")
    async def list_checkpoints(
        session_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
    ):
        await _get_owned_session(config, session_id, user.user_id, accept_suspended=True)
        prefix = f"{config.sessions_prefix}/{session_id}/checkpoints/"
        keys = await config.objects.list_keys(prefix)
        names = sorted({k[len(prefix) :].split("/", 1)[0] for k in keys if "/" in k[len(prefix) :]})
        return {"checkpoints": names}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(
        job_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
        config: Config = Depends(get_config),
    ):
        record = await config.metadata.get_job(job_id)
        if record is None:
            raise HTTPException(404, "Job not found")
        if record.user_id != user.user_id:
            raise HTTPException(403, "Not your job")
        return {
            "job_id": record.job_id,
            "session_id": record.session_id,
            "operation": record.operation,
            "status": record.status.value,
            "created_at": record.created_at,
            "completed_at": record.completed_at,
            "gpu_time_ms": record.gpu_time_ms,
            "error": record.error_message,
        }


# ─── Internal helpers ─────────────────────────────────────────────────────


async def _get_owned_session(
    config: Config,
    session_id: str,
    user_id: str,
    *,
    accept_suspended: bool = True,
) -> SessionRecord:
    record = await config.metadata.get_session(session_id)
    if record is None:
        raise HTTPException(404, "Session not found")
    if record.user_id != user_id:
        raise HTTPException(403, "Not your session")
    allowed = {SessionStatus.ACTIVE}
    if accept_suspended:
        allowed.add(SessionStatus.SUSPENDED)
    if record.status not in allowed:
        raise HTTPException(410, f"Session is {record.status.value}")
    await config.metadata.update_session(
        session_id,
        last_accessed=time.time(),
        status=SessionStatus.ACTIVE,
    )
    return record


async def _resolve_preferred_worker(config: Config, session: SessionRecord) -> Optional[str]:
    """Prefer the session_registry (cross-replica) over metadata fallback.

    ``session_registry`` is an optional extension field; core configs
    won't have it, so we use ``getattr``.
    """
    preferred = session.last_worker_id
    registry = getattr(config, "session_registry", None)
    if registry is not None:
        try:
            registry_worker = await registry.get(session.session_id)
            if registry_worker is not None:
                preferred = registry_worker
        except Exception:  # noqa: BLE001
            pass
    return preferred


async def _run_op(
    config: Config,
    user: AuthenticatedUser,
    session_id: str,
    operation: str,
    payload: dict,
    *,
    priority: int = 0,
    timeout: Optional[float] = None,
) -> dict:
    session = await _get_owned_session(config, session_id, user.user_id)

    # Pre-op plugins (e.g. balance hold). Each returns an opaque context
    # that's threaded back into the post-op hook for settlement.
    pre_op_contexts = await run_pre_op_hooks(config, session, user, operation, payload)

    preferred = await _resolve_preferred_worker(config, session)

    job = await _enqueue_job(
        config=config,
        session_id=session_id,
        user_id=user.user_id,
        operation=operation,
        payload=payload,
        priority=priority,
        preferred_worker=preferred,
        required_model=session.base_model,
        estimated_duration_ms=session.avg_step_duration_ms,
    )

    wait = timeout if timeout is not None else config.max_job_timeout_seconds
    result = await config.queue.wait_for_result(job.job_id, timeout=wait)

    await config.metadata.update_job(
        job.job_id,
        status=result.status,
        completed_at=time.time(),
        error_message=result.error,
        gpu_time_ms=(result.metrics or {}).get("duration_ms"),
        tokens_processed=(result.metrics or {}).get("tokens"),
    )

    # Post-op plugins (e.g. balance settle / release).
    await run_post_op_hooks(config, pre_op_contexts, session, user, result)

    if result.status == JobStatus.FAILED:
        raise HTTPException(500, result.error or "job failed")
    if result.status == JobStatus.TIMED_OUT:
        raise HTTPException(504, result.error or "job timed out")

    return msgpack.unpackb(result.result, raw=False) if result.result else {}


async def _enqueue_job(
    config: Config,
    session_id: str,
    user_id: str,
    operation: str,
    payload: dict,
    priority: int = 0,
    preferred_worker: Optional[str] = None,
    required_model: Optional[str] = None,
    estimated_duration_ms: Optional[float] = None,
    required_cp_degree: int = 1,
) -> JobRecord:
    job_id = str(uuid.uuid4())

    # Payload signers may attach scoped auth tokens (e.g. an
    # extension's internal-auth scheme).
    payload = sign_payload(config, session_id, dict(payload))

    payload_bytes = msgpack.packb(payload, use_bin_type=True)

    inline = len(payload_bytes) <= config.inline_payload_threshold
    if inline:
        payload_key = None
        payload_inline = payload_bytes
    else:
        payload_key = f"{config.jobs_prefix}/{job_id}/payload.msgpack"
        await config.objects.put(payload_key, payload_bytes)
        payload_inline = None

    record = JobRecord(
        job_id=job_id,
        session_id=session_id,
        user_id=user_id,
        operation=operation,
        status=JobStatus.QUEUED,
        created_at=time.time(),
        payload_key=payload_key,
        payload_inline=payload_inline,
    )
    await config.metadata.create_job(record)

    transport = payload_inline if inline else (payload_key or "").encode()
    await config.queue.enqueue(
        QueuedJob(
            job_id=job_id,
            session_id=session_id,
            operation=operation,
            payload=transport,
            priority=priority,
            preferred_worker=preferred_worker,
            required_model=required_model,
            estimated_duration_ms=estimated_duration_ms,
            user_id=user_id,
            enqueued_at=time.time(),
            required_cp_degree=required_cp_degree,
        )
    )

    config.metrics.increment_counter(
        "jobs_enqueued",
        {"operation": operation, "model": required_model or "unknown"},
    )
    return record
