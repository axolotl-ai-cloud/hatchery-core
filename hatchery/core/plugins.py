# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Gateway plugin hooks.

The core gateway is self-contained and runs on core backends alone.
Extension packages layer in additional features (richer auth schemes,
billing, checkpoint sweeping, internal worker routes, scoped-token
payload signing, etc.) by registering callbacks on ``GATEWAY_PLUGINS``.

This module is imported only by core and has no extension dependency.
Extensions import it to register; core calls the hooks if present.

When core is extracted on its own, the plugin registry remains but no
plugins are registered — the gateway runs in pure-core mode.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter

if TYPE_CHECKING:
    from hatchery.core.config import Config
    from hatchery.core.protocols import (
        AuthenticatedUser,
        JobResult,
        QueuedJob,
        SessionRecord,
    )


# ─── Hook signatures ──────────────────────────────────────────────────────

# Token-based auth interceptor: inspect the bearer token, return an
# authenticated user on success, or None to fall through to the next
# interceptor / the default config.auth.authenticate path. Raises
# HTTPException to reject outright.
TokenAuthHook = Callable[[str, "Config"], Awaitable[Optional["AuthenticatedUser"]]]

# Lifespan hooks run on app startup / shutdown. They receive the Config.
LifespanHook = Callable[["Config"], Awaitable[None]]

# Pre-op hook: runs before a session-bound op is enqueued. Returns an
# opaque "context" dict that's passed to the matching post-op hook on
# the same request (for hold → settle reconciliation, etc.). Raise
# HTTPException to reject the op.
PreOpHook = Callable[
    ["Config", "SessionRecord", "AuthenticatedUser", str, dict],
    Awaitable[dict],
]

# Post-op hook: runs after the job result is received. Receives the
# context returned by the paired pre-op hook so it can reconcile state.
PostOpHook = Callable[
    ["Config", dict, "SessionRecord", "AuthenticatedUser", "JobResult"],
    Awaitable[None],
]

# Payload signer: called inside _enqueue_job. May mutate and return the
# payload dict, e.g. to attach a scoped auth token.
PayloadSigner = Callable[["Config", str, dict], dict]

# Payload verifier: called inside the worker's _process_one before the
# handler runs. Raises to reject the job.
PayloadVerifier = Callable[["QueuedJob", dict], None]


@dataclass
class GatewayPlugins:
    """Registry of extension callbacks. Populated by extension packages on import."""

    token_auth: list[TokenAuthHook] = field(default_factory=list)
    lifespan_startup: list[LifespanHook] = field(default_factory=list)
    lifespan_shutdown: list[LifespanHook] = field(default_factory=list)
    routers: list[APIRouter] = field(default_factory=list)
    pre_op: list[PreOpHook] = field(default_factory=list)
    post_op: list[PostOpHook] = field(default_factory=list)
    payload_signers: list[PayloadSigner] = field(default_factory=list)
    payload_verifiers: list[PayloadVerifier] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all registered hooks. Intended for tests."""
        self.token_auth.clear()
        self.lifespan_startup.clear()
        self.lifespan_shutdown.clear()
        self.routers.clear()
        self.pre_op.clear()
        self.post_op.clear()
        self.payload_signers.clear()
        self.payload_verifiers.clear()


GATEWAY_PLUGINS = GatewayPlugins()


def get_plugins() -> GatewayPlugins:
    return GATEWAY_PLUGINS


async def run_pre_op_hooks(
    config: Config,
    session: SessionRecord,
    user: AuthenticatedUser,
    operation: str,
    payload: dict,
) -> list[tuple[PreOpHook, dict]]:
    """Run pre-op hooks in order. Returns pairs for matching post-op hooks."""
    contexts: list[tuple[PreOpHook, dict]] = []
    for hook in GATEWAY_PLUGINS.pre_op:
        ctx = await hook(config, session, user, operation, payload)
        contexts.append((hook, ctx))
    return contexts


async def run_post_op_hooks(
    config: Config,
    contexts: list[tuple[Any, dict]],
    session: SessionRecord,
    user: AuthenticatedUser,
    result: JobResult,
) -> None:
    """Run post-op hooks paired with their pre-op counterpart.

    Each pre-op hook at index *i* is paired with the post-op hook at
    index *i*. If there are fewer post-op hooks than pre-op hooks,
    the extra pre-op contexts are silently skipped (no post-op to
    call). Runs in reverse order for LIFO settlement.
    """
    post_hooks = GATEWAY_PLUGINS.post_op
    for i in range(min(len(contexts), len(post_hooks)) - 1, -1, -1):
        _hook, ctx = contexts[i]
        await post_hooks[i](config, ctx, session, user, result)


def sign_payload(config: Config, session_id: str, payload: dict) -> dict:
    for signer in GATEWAY_PLUGINS.payload_signers:
        payload = signer(config, session_id, payload)
    return payload


def verify_payload(job: QueuedJob, payload: dict) -> None:
    for verifier in GATEWAY_PLUGINS.payload_verifiers:
        verifier(job, payload)
