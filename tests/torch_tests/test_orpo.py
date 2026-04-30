# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for the server-side ORPO loss.

Mirrors :mod:`hatchery.core.losses._orpo`. The reference math is from
Hong, Lee, Thorne — *ORPO: Monolithic Preference Optimization without
Reference Model* (arXiv:2403.07691).

The tests pin:

* the **shape contract** (even batch size required),
* **gradient flow** through the logits,
* **lambda sensitivity** at the two ends of the [0, 1] range,
* **numerical stability** in the ``log(1 - exp(x))`` underflow regime,
* **length-mask correctness** (zero-weight rows don't NaN; longer
  responses with the same per-token logprob produce a smaller
  length-normalized average — well, actually equal, because the
  normalization makes them identical; we use the property to anchor
  the math),
* the **metrics dict shape** the worker plumbs into
  ``JobResult.metrics``.

The "equivalence with client-side reference" test in the spec is
covered here by recomputing the spec's closed-form math (sft + λ * OR)
on the same per-sequence average logprobs and comparing to ``_orpo``'s
output — there's no separate ``compute_orpo_loss`` import in this
worktree, so the equivalence check uses the spec's formula directly,
which is the same math the client-side reference implements.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from hatchery.core.losses import (  # noqa: E402
    LossInputs,
    SUPPORTED_LOSS_FNS,
    _log1mexp,
    _new_logprobs_at_targets,
    _orpo,
    compute,
)


# ─── Helpers ────────────────────────────────────────────────────────────


def _make_inputs(
    B: int = 4,
    T: int = 16,
    V: int = 128,
    *,
    seed: int = 0,
    response_len: int | None = None,
    weights_override: torch.Tensor | None = None,
    loss_fn_config: dict | None = None,
) -> LossInputs:
    """Construct a deterministic LossInputs for ORPO tests.

    ``response_len`` masks the first ``T - response_len`` positions
    (treated as prompt). ``None`` keeps the full sequence as response.

    Uses :func:`torch.random.fork_rng` so the per-test seed doesn't
    leak into the global RNG state — other suites (notably
    ``test_worker_cpu.test_overfit_single_batch_reduces_loss``)
    initialize a LoRA *before* setting their own seed and rely on
    the prior state being whatever the conftest left behind.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        logits = torch.randn(B, T, V, requires_grad=True)
        target_tokens = torch.randint(0, V, (B, T))
    if weights_override is not None:
        weights = weights_override
    else:
        weights = torch.zeros(B, T)
        if response_len is None:
            weights.fill_(1.0)
        else:
            weights[:, T - response_len :] = 1.0
    return LossInputs(
        logits=logits,
        target_tokens=target_tokens,
        weights=weights,
        loss_fn_config=loss_fn_config,
    )


def _avg_logp(inputs: LossInputs) -> torch.Tensor:
    """Recompute the same length-normalized per-sequence logprob the
    loss uses internally. Used to drive the equivalence check from the
    outside without depending on private intermediates.
    """
    logprobs = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    weights = inputs.weights.to(logprobs.dtype)
    masked = logprobs * weights
    denom = weights.sum(dim=-1).clamp_min(1.0)
    return masked.sum(dim=-1) / denom


def _reference_orpo(avg_logp: torch.Tensor, orpo_lambda: float) -> torch.Tensor:
    """Closed-form ORPO loss given pre-computed length-normalized
    per-sequence logprobs. This is the canonical math from the paper
    and the client-side reference; the server-side ``_orpo`` should
    agree to within float tolerance.
    """
    chosen = avg_logp[0::2]
    rejected = avg_logp[1::2]
    sft = -chosen.mean()
    eps = 1e-7
    chosen_clamped = chosen.clamp(max=-eps)
    rejected_clamped = rejected.clamp(max=-eps)
    log_odds = (chosen - rejected) - (
        _log1mexp(chosen_clamped) - _log1mexp(rejected_clamped)
    )
    or_loss = -F.logsigmoid(log_odds).mean()
    return sft + orpo_lambda * or_loss


# ─── Test 1: shape contract ────────────────────────────────────────────


def test_orpo_in_supported_loss_fns():
    assert "orpo" in SUPPORTED_LOSS_FNS


def test_orpo_dispatch_via_compute_returns_tuple():
    """``compute("orpo", ...)`` returns ``(loss, metrics)`` so the
    worker can route diagnostics to ``JobResult.metrics``."""
    inputs = _make_inputs(B=4, T=8, V=16, seed=0)
    out = compute("orpo", inputs)
    assert isinstance(out, tuple)
    loss, metrics = out
    assert torch.is_tensor(loss)
    assert isinstance(metrics, dict)


def test_orpo_rejects_odd_batch_size():
    """Pair-interleaving by index requires an even batch."""
    inputs = _make_inputs(B=3, T=8, V=16, seed=1)
    with pytest.raises(ValueError, match="even batch size"):
        _orpo(inputs)


# ─── Test 2: equivalence with the closed-form spec math ───────────────


def test_orpo_matches_closed_form_reference():
    """The server-side ``_orpo`` and the spec's closed-form
    expression must agree to within float tolerance when fed the
    same per-sequence average logprobs.

    We extract avg logprobs via :func:`_avg_logp` (same code path as
    the loss internally), then run the closed-form math and compare
    to the loss output. The two should be exactly equal modulo the
    detach-on-clamp boundary; use a tight ``1e-5`` tolerance.
    """
    inputs = _make_inputs(B=4, T=16, V=128, seed=42)
    avg_logp = _avg_logp(inputs)
    orpo_lambda = 0.1
    ref = _reference_orpo(avg_logp, orpo_lambda)
    loss, _ = _orpo(
        LossInputs(
            logits=inputs.logits,
            target_tokens=inputs.target_tokens,
            weights=inputs.weights,
            loss_fn_config={"orpo_lambda": orpo_lambda},
        )
    )
    assert torch.allclose(loss, ref, atol=1e-5, rtol=0.0)


# ─── Test 3: gradient flow ─────────────────────────────────────────────


def test_orpo_backward_populates_logits_grad():
    inputs = _make_inputs(B=4, T=16, V=64, seed=7)
    loss, _ = _orpo(inputs)
    assert loss.requires_grad
    loss.backward()
    assert inputs.logits.grad is not None
    assert torch.isfinite(inputs.logits.grad).all()
    # Gradient should be non-trivial — at least some non-zero entries
    # at response-mask positions.
    assert inputs.logits.grad.abs().sum().item() > 0


# ─── Test 4: lambda sensitivity ────────────────────────────────────────


def test_orpo_lambda_zero_equals_sft_loss():
    """With ``orpo_lambda=0`` the OR term drops out and the loss
    collapses to the length-normalized NLL on the chosen response.
    """
    inputs = _make_inputs(B=4, T=16, V=64, seed=11)
    loss, metrics = _orpo(
        LossInputs(
            logits=inputs.logits,
            target_tokens=inputs.target_tokens,
            weights=inputs.weights,
            loss_fn_config={"orpo_lambda": 0.0},
        )
    )
    sft_loss = metrics["orpo/sft_loss"]
    assert math.isclose(float(loss.item()), sft_loss, abs_tol=1e-6)


def test_orpo_lambda_one_or_term_contributes():
    """At ``orpo_lambda=1.0`` the loss must differ from the sft-only
    case by exactly ``or_loss`` (within float tolerance)."""
    base = _make_inputs(B=4, T=16, V=64, seed=12)
    sft_only_loss, _ = _orpo(
        LossInputs(
            logits=base.logits,
            target_tokens=base.target_tokens,
            weights=base.weights,
            loss_fn_config={"orpo_lambda": 0.0},
        )
    )
    full_loss, full_metrics = _orpo(
        LossInputs(
            logits=base.logits,
            target_tokens=base.target_tokens,
            weights=base.weights,
            loss_fn_config={"orpo_lambda": 1.0},
        )
    )
    expected = float(sft_only_loss.item()) + full_metrics["orpo/or_loss"]
    assert math.isclose(float(full_loss.item()), expected, abs_tol=1e-5)


# ─── Test 5: numerical stability in the underflow regime ───────────────


def test_log1mexp_handles_near_zero_input():
    """``log1mexp(-1e-8)`` is ≈ ``log(1e-8) ≈ -18.4``; verify finite."""
    x = torch.tensor([-1e-8, -1e-3, -1.0, -100.0])
    out = _log1mexp(x)
    assert torch.isfinite(out).all()
    # Reference: log(1 - exp(-1e-8)) ≈ log(1e-8) = -18.42 (actually closer
    # to -18.42 because expm1(-1e-8) ≈ -1e-8 so log(-expm1) ≈ log(1e-8)).
    assert out[0].item() < 0.0  # it's a log of something tiny


def test_log1mexp_grad_is_finite_at_underflow_boundary():
    """The grad of ``log1mexp`` at very-near-zero inputs must not NaN
    even though the unselected branch involves ``log1p(-exp(0))``.
    The implementation uses a safe-fill so the dead branch stays
    well-defined.
    """
    x = torch.tensor([-1e-8, -1e-7], requires_grad=True)
    y = _log1mexp(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_orpo_stable_when_chosen_logp_very_close_to_zero():
    """Hand-craft inputs that drive the chosen avg logprob up to
    near 0 (i.e. P(y|x) → 1) — the regime where ``log(1 - p)``
    underflows. The Maechler-2012 stable formulation in
    ``_log1mexp`` plus the ``-eps`` clamp must keep loss + grad
    finite.
    """
    B, T, V = 2, 4, 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        targets = torch.randint(0, V, (B, T))
    # Logits heavily favoring the target tokens → chosen logprob ≈ 0.
    logits = torch.full((B, T, V), -10.0, requires_grad=True)
    with torch.no_grad():
        for b in range(B):
            for t in range(T):
                logits.data[b, t, targets[b, t]] = 50.0
    weights = torch.ones(B, T)
    inputs = LossInputs(
        logits=logits,
        target_tokens=targets,
        weights=weights,
        loss_fn_config={"orpo_lambda": 0.1},
    )
    loss, metrics = _orpo(inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert all(math.isfinite(v) for v in metrics.values())


# ─── Test 6: length-mask correctness ───────────────────────────────────


def test_orpo_fully_masked_sequence_no_nan():
    """A sequence with all-zero weights yields ``avg_logp = 0`` (the
    ``clamp_min(1.0)`` denominator floor). The loss must remain finite
    in that degenerate case so a fully-padded slot doesn't poison the
    batch.
    """
    B, T, V = 4, 8, 16
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        logits = torch.randn(B, T, V, requires_grad=True)
        targets = torch.randint(0, V, (B, T))
    weights = torch.ones(B, T)
    # Mask out the second pair entirely.
    weights[2:, :] = 0.0
    inputs = LossInputs(
        logits=logits,
        target_tokens=targets,
        weights=weights,
        loss_fn_config={"orpo_lambda": 0.1},
    )
    loss, metrics = _orpo(inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert all(math.isfinite(v) for v in metrics.values())


def test_orpo_response_mask_normalization_is_per_token():
    """Length normalization means a response that's twice as long with
    the same per-token logprob produces the *same* length-normalized
    average (the ratio of sums equals the per-token logprob).

    Construct two batches whose response masks differ only in length;
    with constant per-token logprob the loss should be identical.
    Sanity check on the divisor.
    """
    B, T, V = 2, 16, 8
    # Constant logits so logprob is uniform across positions; no RNG used.
    logits = torch.zeros(B, T, V, requires_grad=False)
    targets = torch.zeros(B, T, dtype=torch.long)
    short_w = torch.zeros(B, T)
    short_w[:, -4:] = 1.0
    long_w = torch.zeros(B, T)
    long_w[:, -8:] = 1.0
    short_loss, _ = _orpo(
        LossInputs(
            logits=logits.clone(),
            target_tokens=targets,
            weights=short_w,
            loss_fn_config={"orpo_lambda": 0.1},
        )
    )
    long_loss, _ = _orpo(
        LossInputs(
            logits=logits.clone(),
            target_tokens=targets,
            weights=long_w,
            loss_fn_config={"orpo_lambda": 0.1},
        )
    )
    # Same per-token logprob → same length-normalized average →
    # identical loss regardless of response length.
    assert torch.allclose(short_loss, long_loss, atol=1e-6)


# ─── Test 7: metrics dict shape ────────────────────────────────────────


REQUIRED_METRIC_KEYS = {
    "loss",
    "orpo/sft_loss",
    "orpo/or_loss",
    "orpo/log_odds_ratio",
    "orpo/accuracy",
    "orpo/margin",
    "orpo/chosen_logp",
    "orpo/rejected_logp",
    "orpo/chosen_reward",
    "orpo/lambda",
}


def test_orpo_metrics_carries_required_keys():
    inputs = _make_inputs(B=4, T=16, V=64, seed=303)
    _, metrics = _orpo(inputs)
    assert set(metrics.keys()) == REQUIRED_METRIC_KEYS
    assert len(REQUIRED_METRIC_KEYS) == 10
    for k, v in metrics.items():
        assert isinstance(v, float), f"{k!r} is not float: {type(v).__name__}"
        assert math.isfinite(v), f"{k!r} is not finite: {v}"


def test_orpo_lambda_metric_echoes_input():
    inputs = _make_inputs(
        B=4, T=8, V=16, seed=404, loss_fn_config={"orpo_lambda": 0.42}
    )
    _, metrics = _orpo(inputs)
    assert metrics["orpo/lambda"] == pytest.approx(0.42)


def test_orpo_default_lambda_is_paper_recommended():
    """Default ``orpo_lambda`` is 0.1 per arXiv:2403.07691."""
    inputs = _make_inputs(B=4, T=8, V=16, seed=505)
    _, metrics = _orpo(inputs)
    assert metrics["orpo/lambda"] == pytest.approx(0.1)
