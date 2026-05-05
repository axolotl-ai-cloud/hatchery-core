# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the API key auth provider."""

from __future__ import annotations

from hatchery.core.backends.auth.api_key import APIKeyAuthProvider


async def test_authenticate_valid_token():
    auth = APIKeyAuthProvider()
    auth.add_key("t1", user_id="alice", tier="pro")
    user = await auth.authenticate("t1")
    assert user is not None
    assert user.user_id == "alice"
    assert user.tier == "pro"


async def test_authenticate_rejects_invalid_token():
    auth = APIKeyAuthProvider()
    assert await auth.authenticate("bogus") is None
    assert await auth.authenticate("") is None


async def test_revoke_token():
    auth = APIKeyAuthProvider()
    auth.add_key("t1", user_id="alice")
    assert await auth.authenticate("t1") is not None
    auth.revoke("t1")
    assert await auth.authenticate("t1") is None


async def test_generate_key_roundtrip():
    auth = APIKeyAuthProvider()
    token = auth.generate_key("bob", tier="free")
    assert token
    user = await auth.authenticate(token)
    assert user.user_id == "bob"


async def test_authorize_model_allowlist():
    auth = APIKeyAuthProvider()
    auth.add_key("t1", user_id="alice", allowed_models=["Qwen/Qwen2-0.5B"])
    user = await auth.authenticate("t1")
    assert await auth.authorize(user, "train", "Qwen/Qwen2-0.5B")
    assert not await auth.authorize(user, "train", "meta-llama/Llama-3-70B")


async def test_tokens_stored_hashed():
    auth = APIKeyAuthProvider()
    auth.add_key("raw-secret", user_id="alice")
    # The raw token should not appear in the records dict values.
    raw_values = [getattr(r, "token_hash", "") for r in auth._records.values()]
    assert "raw-secret" not in raw_values
