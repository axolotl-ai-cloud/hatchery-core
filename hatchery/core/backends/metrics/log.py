# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Logging-based metrics collector.

Emits structured log events. Also accumulates counters/gauges/histograms
in memory for tests and basic diagnostics. A Prometheus/Datadog collector
would subclass or replace this at deployment time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog


class LogMetrics:
    def __init__(self) -> None:
        self._logger = structlog.get_logger("hatchery.core.metrics")
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self.gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _tag_key(tags: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(tags.items()))

    def record_job_duration(
        self,
        session_id: str,
        user_id: str,
        operation: str,
        duration_ms: float,
        tokens: int,
        worker_id: str,
        gpu_type: str,
        cost_dimensions: dict | None = None,
    ) -> None:
        self.histograms[f"job_duration_ms:{operation}"].append(duration_ms)
        self.counters[("tokens_processed_total", self._tag_key({"operation": operation}))] += int(
            tokens
        )
        # Per-user token accumulator for usage-dashboard-style queries.
        if user_id:
            self.counters[
                ("user_tokens_total", self._tag_key({"user_id": user_id, "operation": operation}))
            ] += int(tokens)
        cd = cost_dimensions or {}
        self._logger.info(
            "job.duration",
            session_id=session_id,
            user_id=user_id,
            operation=operation,
            duration_ms=duration_ms,
            tokens=tokens,
            worker_id=worker_id,
            gpu_type=gpu_type,
            **{f"cd_{k}": v for k, v in cd.items()},
        )
        # Append to a structured list so the cost-analysis pipeline
        # (or tests) can query the full row after the fact.
        self.events.append(
            {
                "type": "job_duration",
                "session_id": session_id,
                "user_id": user_id,
                "operation": operation,
                "duration_ms": duration_ms,
                "tokens": tokens,
                "worker_id": worker_id,
                "gpu_type": gpu_type,
                **(cd or {}),
            }
        )

    def record_queue_depth(self, model: str, depth: int) -> None:
        self.gauges[("queue_depth", self._tag_key({"model": model}))] = depth

    def record_worker_utilization(
        self, worker_id: str, gpu_util_pct: float, vram_used_mb: int
    ) -> None:
        self.gauges[("worker_gpu_utilization_pct", self._tag_key({"worker": worker_id}))] = (
            gpu_util_pct
        )
        self.gauges[("worker_vram_used_mb", self._tag_key({"worker": worker_id}))] = vram_used_mb

    def record_lora_swap_time(
        self,
        session_id: str,
        swap_direction: str,
        duration_ms: float,
        state_size_bytes: int,
    ) -> None:
        self.histograms[f"lora_swap_ms:{swap_direction}"].append(duration_ms)
        self._logger.info(
            "lora.swap",
            session_id=session_id,
            direction=swap_direction,
            duration_ms=duration_ms,
            state_size_bytes=state_size_bytes,
        )

    def record_object_store_io(
        self,
        operation: str,
        key: str,
        size_bytes: int,
        duration_ms: float,
    ) -> None:
        self.histograms[f"object_store_{operation}_ms"].append(duration_ms)

    def record_session_event(self, session_id: str, event: str) -> None:
        self.events.append({"session_id": session_id, "event": event})
        # structlog reserves ``event`` as the log message key.
        self._logger.info("session.event", session_id=session_id, event_type=event)

    def increment_counter(self, name: str, tags: dict[str, str]) -> None:
        self.counters[(name, self._tag_key(tags))] += 1

    def set_gauge(self, name: str, value: float, tags: dict[str, str]) -> None:
        self.gauges[(name, self._tag_key(tags))] = value

    # ── Test helpers ────────────────────────────────────────

    def get_counter(self, name: str, tags: dict[str, str] | None = None) -> int:
        return self.counters.get((name, self._tag_key(tags or {})), 0)

    def get_gauge(self, name: str, tags: dict[str, str] | None = None) -> float | None:
        return self.gauges.get((name, self._tag_key(tags or {})))
