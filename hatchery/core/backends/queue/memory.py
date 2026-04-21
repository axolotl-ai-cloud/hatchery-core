# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""In-process asyncio job queue.

Design notes
------------
The queue enforces two invariants that pipelined training depends on:

1. **Per-session FIFO.** Jobs submitted for the same ``session_id``
   execute in submission order regardless of per-op priority. Without
   this, ``optim_step`` (high priority) could jump ahead of a
   ``forward_backward`` (normal priority) submitted earlier on the
   same session and apply grads to the wrong weights.

2. **Single in-flight per session.** At most one job per session is
   held by any worker at a time. Different sessions run concurrently
   on different workers, but operations on the same session are
   strictly serialized so object-store reads and writes compose
   correctly.

Implementation: each session owns a FIFO ``deque`` of waiting jobs.
A dequeue picks the "head of each session" (only eligible when the
session has nothing in flight) and orders those heads across sessions
by priority, then by enqueue order. Priority still wins *across*
sessions — e.g., a brand-new ``init_session`` at priority 10 gets
picked up before a low-priority ``forward_backward`` for a different
session that was enqueued earlier.

Visibility timeout: a per-session in-flight job not ``ack``-ed within
the timeout is released back to its session's deque (at the head) and
the session's in-flight lock is cleared so a replacement worker can
pick it up.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from typing import Optional, Union

from hatchery.core.protocols import JobResult, JobStatus, QueuedJob

# How long a job with preferred_worker is exclusively available to
# that worker before falling back to any worker. Short enough to avoid
# blocking on a dead/busy preferred worker, long enough for the
# preferred worker to finish its current job and poll again.
_AFFINITY_WINDOW_S = 2.0


def _normalize_filter(
    model_filter: Optional[Union[str, list[str]]],
) -> frozenset[str]:
    """Normalize a str / list-of-str / None filter to a frozenset.

    An empty frozenset means "no filter" — callers should treat it the
    same as ``None``.
    """
    if model_filter is None:
        return frozenset()
    if isinstance(model_filter, str):
        return frozenset({model_filter})
    return frozenset(model_filter)


class InMemoryJobQueue:
    def __init__(self, *, clock=None) -> None:
        self._clock = clock or time.time
        # session_id -> deque of waiting jobs (head = next eligible)
        self._session_queues: dict[str, deque[QueuedJob]] = {}
        # session_id -> job_id currently in flight (single-in-flight invariant)
        self._inflight_by_session: dict[str, str] = {}
        # job_id -> (session_id, worker_id, deadline) for visibility timeout
        self._inflight: dict[str, tuple[str, str, float]] = {}

        self._counter = itertools.count()
        # Per-job enqueue sequence, used as a global tiebreaker so older
        # jobs beat newer ones when priorities are equal.
        self._seq: dict[str, int] = {}

        self._lock = asyncio.Lock()

        # Result delivery
        self._results: dict[str, JobResult] = {}
        self._result_events: dict[str, asyncio.Event] = {}

        # Retry / dead-letter bookkeeping
        self._attempts: dict[str, int] = {}
        self._max_attempts = 3
        self._dead_letter: dict[str, tuple[QueuedJob, str]] = {}

        # Hard-pin: session_id -> (owner_worker_id, expires_at). Only the
        # owner may dequeue pinned sessions — used while grad_accum state
        # lives only on the owner's local disk.
        self._accum_pins: dict[str, tuple[str, float]] = {}

    # ── Lifecycle ────────────────────────────────────────────

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    # ── Core queue ops ───────────────────────────────────────

    async def enqueue(self, job: QueuedJob) -> None:
        if job.enqueued_at is None:
            job.enqueued_at = self._clock()
        async with self._lock:
            dq = self._session_queues.setdefault(job.session_id, deque())
            dq.append(job)
            self._seq[job.job_id] = next(self._counter)
            self._result_events.setdefault(job.job_id, asyncio.Event())

    async def _requeue_expired(self) -> None:
        """Release any in-flight jobs whose visibility timeout has lapsed."""
        now = self._clock()
        async with self._lock:
            expired = [
                job_id
                for job_id, (_sess, _worker, deadline) in self._inflight.items()
                if deadline <= now
            ]
            for job_id in expired:
                session_id, _worker, _deadline = self._inflight.pop(job_id)
                self._inflight_by_session.pop(session_id, None)
                # The job itself remains at the head of its session deque,
                # since _put_inflight only moves the in-flight marker.
                # If we removed it from the deque, put it back at the head.
                dq = self._session_queues.get(session_id)
                if dq is None:
                    dq = deque()
                    self._session_queues[session_id] = dq
                # The expired job isn't currently in the deque — it was
                # removed when we dequeued it. Put it back at the head
                # so the next dequeue retries the same job in order.
                # We look it up via the dead-letter retry mechanism.
                # Simpler: we stashed the full QueuedJob in _dead_letter_pending.
                job = self._pending_expired.pop(job_id, None)
                if job is not None:
                    dq.appendleft(job)

    # A side table for "job was in-flight, we need to remember the full
    # QueuedJob in case the visibility timeout releases it". Populated
    # on dequeue, cleared on ack/nack/visibility-timeout handling.
    _pending_expired: dict[str, QueuedJob]

    def __init_pending_expired(self) -> None:
        if not hasattr(self, "_pending_expired"):
            self._pending_expired = {}

    async def dequeue(
        self,
        worker_id: str,
        model_filter: Optional[Union[str, list[str]]] = None,
        visibility_timeout: int = 300,
    ) -> Optional[QueuedJob]:
        self.__init_pending_expired()
        await self._requeue_expired()

        # Normalize filter to a set. None or empty list/set = no filter.
        filter_set = _normalize_filter(model_filter)

        async with self._lock:
            now_pin = self._clock()
            # Build the list of eligible heads: one per session that has
            # a waiting job AND no in-flight job.
            candidates: list[QueuedJob] = []
            for session_id, dq in self._session_queues.items():
                if not dq:
                    continue
                if session_id in self._inflight_by_session:
                    continue
                head = dq[0]
                if (
                    filter_set
                    and head.required_model is not None
                    and head.required_model not in filter_set
                ):
                    continue
                # Hard-pin: if another worker owns the session's grad_accum
                # state, skip it entirely (the affinity window is
                # time-bounded; the pin is not).
                pin = self._accum_pins.get(session_id)
                if pin is not None:
                    owner, expires = pin
                    if expires <= now_pin:
                        self._accum_pins.pop(session_id, None)
                    elif owner != worker_id:
                        continue
                candidates.append(head)

            if not candidates:
                return None

            # ── Sticky affinity window ──
            # Jobs with a preferred_worker are exclusively available to
            # that worker for the first ``_AFFINITY_WINDOW_S`` seconds
            # after enqueue. After the window, any worker can take them.
            # This prevents unnecessary LoRA state downloads when the
            # preferred worker is just momentarily busy.
            now = self._clock()
            visible: list[QueuedJob] = []
            for job in candidates:
                if (
                    job.preferred_worker
                    and job.preferred_worker != worker_id
                    and job.enqueued_at is not None
                    and (now - job.enqueued_at) < _AFFINITY_WINDOW_S
                ):
                    continue  # Still in exclusive window for another worker.
                visible.append(job)

            if not visible:
                return None

            # Sort: priority first, then affinity bonus, then enqueue order.
            def _sort_key(job: QueuedJob) -> tuple:
                affinity_bonus = 0 if job.preferred_worker == worker_id else 1
                return (
                    -job.priority,
                    affinity_bonus,
                    self._seq.get(job.job_id, 0),
                )

            visible.sort(key=_sort_key)
            selected = visible[0]

            # Pop it from the session deque and mark in-flight.
            dq = self._session_queues[selected.session_id]
            dq.popleft()
            self._inflight[selected.job_id] = (
                selected.session_id,
                worker_id,
                self._clock() + visibility_timeout,
            )
            self._inflight_by_session[selected.session_id] = selected.job_id
            self._pending_expired[selected.job_id] = selected
            return selected

    async def ack(self, job_id: str, result: JobResult) -> None:
        self.__init_pending_expired()
        async with self._lock:
            entry = self._inflight.pop(job_id, None)
            if entry is not None:
                session_id, _worker, _deadline = entry
                self._inflight_by_session.pop(session_id, None)
                self._pending_expired.pop(job_id, None)
            self._attempts.pop(job_id, None)
            self._results[job_id] = result
            event = self._result_events.setdefault(job_id, asyncio.Event())
            event.set()

    async def nack(self, job_id: str, error: str) -> None:
        self.__init_pending_expired()
        async with self._lock:
            entry = self._inflight.pop(job_id, None)
            if entry is None:
                return
            session_id, _worker, _deadline = entry
            self._inflight_by_session.pop(session_id, None)
            job = self._pending_expired.pop(job_id, None)

            attempts = self._attempts.get(job_id, 0) + 1
            self._attempts[job_id] = attempts

            if attempts >= self._max_attempts or job is None:
                if job is not None:
                    self._dead_letter[job_id] = (job, error)
                self._results[job_id] = JobResult(
                    job_id=job_id, status=JobStatus.FAILED, error=error
                )
                self._result_events.setdefault(job_id, asyncio.Event()).set()
                return

            # Retry: push back to the head of the session's deque so it
            # runs before anything newer.
            dq = self._session_queues.setdefault(session_id, deque())
            dq.appendleft(job)

    async def wait_for_result(self, job_id: str, timeout: float = 120.0) -> JobResult:
        async with self._lock:
            if job_id in self._results:
                return self._results[job_id]
            event = self._result_events.setdefault(job_id, asyncio.Event())

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            return JobResult(
                job_id=job_id,
                status=JobStatus.TIMED_OUT,
                error=f"Timed out after {timeout}s waiting for result",
            )

        async with self._lock:
            return self._results[job_id]

    async def get_queue_depth(self, model_filter: Optional[Union[str, list[str]]] = None) -> int:
        filter_set = _normalize_filter(model_filter)
        async with self._lock:
            depth = 0
            for dq in self._session_queues.values():
                for job in dq:
                    if (
                        not filter_set
                        or job.required_model is None
                        or job.required_model in filter_set
                    ):
                        depth += 1
            for job_id, (_sess, _w, _d) in self._inflight.items():
                job = (
                    self._pending_expired.get(job_id) if hasattr(self, "_pending_expired") else None
                )
                if job is None:
                    # Fall back to counting — we don't have the required_model.
                    depth += 1 if not filter_set else 0
                elif (
                    not filter_set or job.required_model is None or job.required_model in filter_set
                ):
                    depth += 1
            return depth

    # ── Accumulation hard-pin ────────────────────────────────

    async def set_accum_pin(self, session_id: str, worker_id: str, *, ttl_s: float = 600.0) -> None:
        async with self._lock:
            self._accum_pins[session_id] = (worker_id, self._clock() + ttl_s)

    async def clear_accum_pin(self, session_id: str) -> None:
        async with self._lock:
            self._accum_pins.pop(session_id, None)

    # ── Debug helpers ────────────────────────────────────────

    async def dead_letter_jobs(self) -> list[tuple[QueuedJob, str]]:
        async with self._lock:
            return list(self._dead_letter.values())
