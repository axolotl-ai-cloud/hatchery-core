# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Simple API-key auth provider.

Keys are held in a dict keyed by token. Suitable for solo-dev and tests.
Production deployments swap in an SSO-backed auth provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional

from hatchery.core.protocols import AuthenticatedUser


@dataclass
class _APIKeyRecord:
    user: AuthenticatedUser
    token_hash: str
    revoked: bool = False


class APIKeyAuthProvider:
    """In-memory API key store.

    Tokens are stored hashed so a memory dump doesn't leak credentials.
    """

    def __init__(self) -> None:
        self._records: dict[str, _APIKeyRecord] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def add_key(
        self,
        token: str,
        user_id: str,
        *,
        email: Optional[str] = None,
        org_id: Optional[str] = None,
        roles: Optional[list[str]] = None,
        tier: str = "free",
        max_concurrent_sessions: int = 5,
        max_rank: int = 64,
        allowed_models: Optional[list[str]] = None,
    ) -> None:
        th = self._hash(token)
        user = AuthenticatedUser(
            user_id=user_id,
            email=email,
            org_id=org_id,
            roles=roles or [],
            tier=tier,
            max_concurrent_sessions=max_concurrent_sessions,
            max_rank=max_rank,
            allowed_models=allowed_models,
        )
        self._records[th] = _APIKeyRecord(user=user, token_hash=th)

    def generate_key(self, user_id: str, **kwargs) -> str:
        """Generate a fresh key, register it, and return the raw token."""
        token = secrets.token_urlsafe(32)
        self.add_key(token, user_id, **kwargs)
        return token

    def revoke(self, token: str) -> None:
        th = self._hash(token)
        rec = self._records.get(th)
        if rec:
            rec.revoked = True

    async def authenticate(self, token: str) -> Optional[AuthenticatedUser]:
        if not token:
            return None
        th = self._hash(token)
        async with self._lock:
            rec = self._records.get(th)
            if rec is None or rec.revoked:
                return None
            return rec.user

    async def authorize(self, user: AuthenticatedUser, action: str, resource: str) -> bool:
        # Allow by default; model allow-list is enforced at the gateway.
        if action == "train" and user.allowed_models is not None:
            return resource in user.allowed_models
        return True
