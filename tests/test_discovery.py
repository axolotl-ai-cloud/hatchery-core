# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the GatewayDiscovery abstraction."""

from __future__ import annotations

import hatchery.core.discovery as discovery_module
from hatchery.core.discovery import (
    EnvGatewayDiscovery,
    GatewayEndpoint,
    StaticListDiscovery,
    build_default_discovery,
    register_discovery_factory,
)


async def test_env_discovery_reads_url(monkeypatch):
    monkeypatch.setenv("HATCHERY_GATEWAY_URL", "https://gateway.example.com")
    d = EnvGatewayDiscovery()
    endpoints = await d.resolve()
    assert len(endpoints) == 1
    assert endpoints[0].url == "https://gateway.example.com"


async def test_env_discovery_empty_when_unset(monkeypatch):
    monkeypatch.delenv("HATCHERY_GATEWAY_URL", raising=False)
    d = EnvGatewayDiscovery()
    assert await d.resolve() == []
    assert await d.choose_one() is None


async def test_env_discovery_with_explicit_default(monkeypatch):
    monkeypatch.delenv("HATCHERY_GATEWAY_URL", raising=False)
    d = EnvGatewayDiscovery(default="http://localhost:8420")
    endpoint = await d.choose_one()
    assert endpoint is not None
    assert endpoint.url == "http://localhost:8420"


async def test_env_discovery_custom_env_var(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_GATEWAY", "http://internal.svc:9000")
    monkeypatch.delenv("HATCHERY_GATEWAY_URL", raising=False)
    d = EnvGatewayDiscovery(env_var="MY_CUSTOM_GATEWAY")
    endpoint = await d.choose_one()
    assert endpoint is not None
    assert endpoint.url == "http://internal.svc:9000"


async def test_static_list_cycles():
    d = StaticListDiscovery(
        [
            GatewayEndpoint(url="http://a", region="us-east"),
            GatewayEndpoint(url="http://b", region="us-west"),
        ]
    )
    resolved = await d.resolve()
    assert len(resolved) == 2
    assert {e.url for e in resolved} == {"http://a", "http://b"}
    # choose_one returns one of them.
    chosen = await d.choose_one()
    assert chosen is not None and chosen.url in {"http://a", "http://b"}


async def test_static_list_empty():
    d = StaticListDiscovery([])
    assert await d.resolve() == []
    assert await d.choose_one() is None


def test_build_default_returns_env_impl(monkeypatch):
    monkeypatch.delenv("HATCHERY_GATEWAY_DISCOVERY", raising=False)
    assert isinstance(build_default_discovery(), EnvGatewayDiscovery)


def test_build_default_dispatches_on_env(monkeypatch):
    """HATCHERY_GATEWAY_DISCOVERY=<name> selects the registered factory."""
    # Save + restore the registry so other tests aren't affected.
    saved = dict(discovery_module._DISCOVERY_FACTORIES)
    sentinel = StaticListDiscovery([GatewayEndpoint(url="http://sentinel")])
    try:
        register_discovery_factory("fake", lambda: sentinel)
        monkeypatch.setenv("HATCHERY_GATEWAY_DISCOVERY", "fake")
        got = build_default_discovery()
        assert got is sentinel
    finally:
        discovery_module._DISCOVERY_FACTORIES.clear()
        discovery_module._DISCOVERY_FACTORIES.update(saved)


def test_build_default_unknown_name_falls_back(monkeypatch):
    """An unregistered name falls back to EnvGatewayDiscovery rather than raising."""
    monkeypatch.setenv("HATCHERY_GATEWAY_DISCOVERY", "does-not-exist")
    assert isinstance(build_default_discovery(), EnvGatewayDiscovery)
