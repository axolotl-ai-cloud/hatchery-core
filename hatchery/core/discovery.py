# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Gateway discovery for workers and the autoscaler.

A worker needs to know which gateway(s) to call for internal routes
(heartbeats, registration, future metrics reporting). Right now
``HATCHERY_GATEWAY_URL`` in the env is enough: one gateway service
behind a stable DNS name (Railway / Fly / k8s Service), workers
point at it directly.

But the moment you have more than one gateway replica, or workers
need to pick between gateways by region / load / version, you need
something real. This module defines the :class:`GatewayDiscovery`
protocol so we can swap in a fancier implementation later without
touching worker code.

Implementations currently shipped:

* :class:`EnvGatewayDiscovery` — single URL from ``HATCHERY_GATEWAY_URL``.
  Default for everyone.
* :class:`StaticListDiscovery` — a hardcoded list for smoke tests or
  hand-rolled deployments.

Extension packages may register additional implementations (e.g., a
Redis-backed registry for multi-replica deployments) via
:func:`register_discovery_factory` — see below.

Future implementations (not implemented):

* ``DNSSrvDiscovery`` — SRV records drive selection (Consul,
  Kubernetes headless services).
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class GatewayEndpoint:
    url: str
    region: Optional[str] = None
    version: Optional[str] = None


class GatewayDiscovery(Protocol):
    """Return one or more gateway endpoints reachable by a caller.

    Implementations should not raise on "no gateways" — they return
    an empty list and let the caller decide whether to retry, fall
    back to a default, or fail loudly.
    """

    async def resolve(self) -> list[GatewayEndpoint]: ...

    async def choose_one(self) -> Optional[GatewayEndpoint]: ...


class EnvGatewayDiscovery:
    """Single-URL discovery driven by ``HATCHERY_GATEWAY_URL``.

    This covers the common deployment shape: one gateway service
    behind a stable hostname provided by your hosting platform.
    """

    def __init__(
        self,
        *,
        env_var: str = "HATCHERY_GATEWAY_URL",
        default: Optional[str] = None,
    ) -> None:
        self.env_var = env_var
        self.default = default

    def _current(self) -> Optional[str]:
        return os.environ.get(self.env_var, self.default)

    async def resolve(self) -> list[GatewayEndpoint]:
        url = self._current()
        if not url:
            return []
        return [GatewayEndpoint(url=url)]

    async def choose_one(self) -> Optional[GatewayEndpoint]:
        endpoints = await self.resolve()
        return endpoints[0] if endpoints else None


class StaticListDiscovery:
    """Hand-rolled list of endpoints. Used by tests and hand-tuned
    deployments where the operator knows every gateway's URL.
    """

    def __init__(self, endpoints: list[GatewayEndpoint]) -> None:
        self._endpoints = list(endpoints)

    async def resolve(self) -> list[GatewayEndpoint]:
        return list(self._endpoints)

    async def choose_one(self) -> Optional[GatewayEndpoint]:
        if not self._endpoints:
            return None
        return random.choice(self._endpoints)


# ─── Pluggable factory registry ───────────────────────────────────────────

# Extension packages can register additional discovery implementations
# via :func:`register_discovery_factory`. Keeping this registry in core
# avoids a core → extension import while still letting
# ``build_default_discovery`` dispatch on ``HATCHERY_GATEWAY_DISCOVERY``.

_DISCOVERY_FACTORIES: dict[str, Callable[[], GatewayDiscovery]] = {}


def register_discovery_factory(name: str, factory: Callable[[], GatewayDiscovery]) -> None:
    """Register a discovery factory keyed by env-var value.

    Example — an extension package may register ``"redis"`` so that
    setting ``HATCHERY_GATEWAY_DISCOVERY=redis`` selects a Redis-backed
    registry implementation.
    """
    _DISCOVERY_FACTORIES[name] = factory


def build_default_discovery() -> GatewayDiscovery:
    """Return the discovery impl the worker should use by default.

    Dispatch table:

    * ``HATCHERY_GATEWAY_DISCOVERY`` unset / ``"env"`` → :class:`EnvGatewayDiscovery`
    * anything registered via :func:`register_discovery_factory` wins when
      its name matches ``HATCHERY_GATEWAY_DISCOVERY``

    Core ships the ``env`` path. Extensions may register additional
    kinds (e.g. ``"redis"``) on import.
    """
    kind = os.environ.get("HATCHERY_GATEWAY_DISCOVERY", "env")
    factory = _DISCOVERY_FACTORIES.get(kind)
    if factory is not None:
        return factory()
    return EnvGatewayDiscovery()
