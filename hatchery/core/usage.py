# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Per-customer usage tracking protocol and neutral dataclasses.

Core defines the protocol and summary shapes. Pricing, invoice lines,
and concrete implementations live in extension packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional, Protocol


class UsageStore(Protocol):
    """Persists per-customer usage records."""

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...

    async def record_tokens(
        self,
        *,
        user_id: str,
        model: str,
        operation: str,
        tokens: int,
        cost_usd: float,
        recorded_date: date,
        cost_dimensions: Optional[dict[str, Any]] = None,
    ) -> None: ...

    async def record_storage(
        self,
        *,
        user_id: str,
        checkpoint_type: str,
        gb_days: float,
        recorded_date: date,
    ) -> None: ...

    async def record_credit(
        self,
        *,
        user_id: str,
        amount_usd: float,
        reason: str,
        created_by: str,
        recorded_date: date,
    ) -> None: ...

    async def record_adjustment(
        self,
        *,
        user_id: str,
        model: str,
        operation: str,
        tokens: int,
        cost_usd: float,
        reason: str,
        created_by: str,
        recorded_date: date,
    ) -> None: ...

    async def get_usage_summary(self, user_id: str, year: int, month: int) -> UsageSummary: ...

    async def get_invoice_lines(self, user_id: str, year: int, month: int) -> list: ...


@dataclass
class UsageSummary:
    """Monthly rollup for the customer-facing dashboard."""

    user_id: str
    year: int
    month: int
    total_spend_usd: float = 0.0
    total_tokens: int = 0
    checkpoint_storage_gb_days: float = 0.0
    credits_usd: float = 0.0
    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    by_operation: dict[str, int] = field(default_factory=dict)
    daily_spend: list[DailySpend] = field(default_factory=list)


@dataclass
class ModelUsage:
    model: str
    tokens: int = 0
    spend_usd: float = 0.0


@dataclass
class DailySpend:
    day: date
    spend_usd: float = 0.0
    tokens: int = 0
