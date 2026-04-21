# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Verify that /asample never returns cached responses.

Sampling is inherently stochastic — the seq_id idempotency tracker must
NOT be used for sample requests.  These tests verify the tracker itself
still works (for training ops) and that the /asample handler bypasses it.
"""

from __future__ import annotations

import pytest

from hatchery.core.tinker_compat import _SeqIdTracker


class TestSeqIdTrackerBasics:
    """The tracker itself works correctly for training ops."""

    def test_same_session_same_seqid_returns_cached(self):
        tracker = _SeqIdTracker()
        resp = {"future_id": "job-aaa"}
        tracker.record("fb::sess1", 1, resp)

        assert tracker.check("fb::sess1", 1) is resp

    def test_same_session_different_seqid_no_cache(self):
        tracker = _SeqIdTracker()
        tracker.record("fb::sess1", 1, {"future_id": "job-aaa"})

        assert tracker.check("fb::sess1", 2) is None

    def test_different_sessions_same_seqid_no_collision(self):
        tracker = _SeqIdTracker()
        tracker.record("fb::sess1", 1, {"future_id": "job-aaa"})

        assert tracker.check("fb::sess2", 1) is None

    def test_seqid_zero_bypasses_cache(self):
        tracker = _SeqIdTracker()
        tracker.record("fb::sess1", 0, {"future_id": "should-not-cache"})

        assert tracker.check("fb::sess1", 0) is None

    def test_seqid_none_bypasses_cache(self):
        tracker = _SeqIdTracker()
        tracker.record("fb::sess1", None, {"future_id": "should-not-cache"})

        assert tracker.check("fb::sess1", None) is None


class TestAsampleBypassesIdempotency:
    """The /asample handler must call _future_response directly, not
    _idempotent_future_response.  Verify by checking the source."""

    def test_asample_does_not_use_idempotent_wrapper(self):
        """Grep the asample handler to confirm it calls _future_response
        directly rather than _idempotent_future_response."""
        import ast
        import inspect

        from hatchery.core import tinker_compat

        source = inspect.getsource(tinker_compat.asample)
        tree = ast.parse(source)

        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_names.add(node.func.id)

        assert "_future_response" in called_names, (
            "asample should call _future_response directly"
        )
        assert "_idempotent_future_response" not in called_names, (
            "asample must NOT use _idempotent_future_response — "
            "sampling is stochastic and must never return cached results"
        )
