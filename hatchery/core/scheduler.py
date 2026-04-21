# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Scheduling primitives for the GPU worker.

``StepCadenceTracker`` records per-session request timing for cache
eviction decisions and worker affinity. ``SmartLoRACache`` uses the
cadence data to evict the least-urgent adapter when VRAM fills up.

These are core because every worker needs them regardless of deployment
topology. A worker autoscaler that uses these primitives to make
provisioning decisions lives outside core, alongside whichever compute
backend it drives.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any, Optional


class StepCadenceTracker:
    """Tracks per-session request timing for cache and affinity decisions."""

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self._history: dict[str, deque] = {}

    def record(self, session_id: str, operation: str, duration_ms: float) -> None:
        if session_id not in self._history:
            self._history[session_id] = deque(maxlen=self.window_size)
        self._history[session_id].append((time.time(), operation, duration_ms))

    def last_seen(self, session_id: str) -> Optional[float]:
        hist = self._history.get(session_id)
        if not hist:
            return None
        return hist[-1][0]

    def avg_inter_request_interval(self, session_id: str) -> Optional[float]:
        hist = self._history.get(session_id)
        if hist is None or len(hist) < 2:
            return None
        timestamps = [t for t, _, _ in hist]
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        return sum(intervals) / len(intervals)

    def predicted_next_request(self, session_id: str) -> Optional[float]:
        hist = self._history.get(session_id)
        if hist is None or len(hist) < 2:
            return None
        avg = self.avg_inter_request_interval(session_id)
        if avg is None:
            return None
        return hist[-1][0] + avg

    def request_rate_hz(self, session_id: str) -> Optional[float]:
        interval = self.avg_inter_request_interval(session_id)
        if interval is None or interval <= 0:
            return None
        return 1.0 / interval

    def classify_workload(self, session_id: str) -> str:
        rate = self.request_rate_hz(session_id)
        if rate is None:
            return "unknown"
        if rate > 1.0:
            return "burst"
        if rate > 0.01:
            return "interactive"
        return "background"

    def avg_duration_ms(self, session_id: str) -> Optional[float]:
        hist = self._history.get(session_id)
        if not hist:
            return None
        durations = [d for _, _, d in hist if d is not None]
        if not durations:
            return None
        return sum(durations) / len(durations)


class SmartLoRACache:
    """Worker-local LoRA cache with workload-aware eviction.

    Eviction priority considers recency, request rate, and the predicted
    time until the next request.
    """

    def __init__(
        self,
        max_slots: int = 20,
        cadence_tracker: Optional[StepCadenceTracker] = None,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        self.max_slots = max_slots
        self.cadence = cadence_tracker or StepCadenceTracker()
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._now = now
        self.on_evict: Optional[Callable[[str, Any], None]] = None
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._cache

    def keys(self) -> list[str]:
        return list(self._cache.keys())

    def get(self, session_id: str) -> Optional[Any]:
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            self.hits += 1
            return self._cache[session_id]
        self.misses += 1
        return None

    def put(self, session_id: str, state: Any) -> None:
        if session_id in self._cache:
            self._cache[session_id] = state
            self._cache.move_to_end(session_id)
            return
        self._cache[session_id] = state
        while len(self._cache) > self.max_slots:
            self._evict_one()

    def pop(self, session_id: str) -> Optional[Any]:
        return self._cache.pop(session_id, None)

    def evict(self, session_id: str) -> None:
        state = self._cache.pop(session_id, None)
        if state is not None and self.on_evict is not None:
            self.on_evict(session_id, state)

    def clear(self) -> None:
        if self.on_evict is not None:
            for sid, state in list(self._cache.items()):
                self.on_evict(sid, state)
        self._cache.clear()

    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def _evict_one(self) -> None:
        now = self._now()
        best_victim: Optional[str] = None
        best_score = float("inf")
        for sid in self._cache:
            score = self._eviction_score(sid, now)
            if score < best_score:
                best_score = score
                best_victim = sid
        if best_victim is None:
            best_victim = next(iter(self._cache))
        state = self._cache.pop(best_victim)
        self.evictions += 1
        if self.on_evict is not None:
            self.on_evict(best_victim, state)

    def _eviction_score(self, session_id: str, now: float) -> float:
        rate = self.cadence.request_rate_hz(session_id)
        if rate is None:
            return float("inf")
        predicted = self.cadence.predicted_next_request(session_id)
        if predicted is None:
            return rate
        time_until = max(predicted - now, 0.01)
        urgency = 1.0 / time_until
        return rate * urgency
