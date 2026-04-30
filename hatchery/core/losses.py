# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Server-side loss functions.

Tinker's public API declares five built-in ``loss_fn`` values
(``cross_entropy``, ``importance_sampling``, ``ppo``, ``cispo``,
``dro``) plus a client-side ``forward_backward_custom`` path. This
module owns the actual math for the four losses we can ship without
guessing (CE, IS, PPO, CISPO) and defines the protocol that lets the
worker/trainer dispatch to them.

All loss functions here share the same contract:

    loss = compute(
        logits: Tensor,          # [B, T, V] — model output
        target_tokens: Tensor,   # [B, T]    — action / target per position
        weights: Tensor,         # [B, T]    — per-position weight (0 = ignore)
        old_logprobs: Tensor|None,  # [B, T]  — from the rollout policy (RL only)
        advantages: Tensor|None, # [B, T]    — per-position advantage (RL only)
        loss_fn_config: dict,    # clip thresholds, KL coeffs, etc.
    ) -> Tensor                  # scalar

The caller is responsible for producing ``target_tokens`` and
``weights`` (via the collate path), extracting old logprobs and
advantages from the incoming ``loss_fn_inputs``, and shifting
shapes for the causal LM (predict token ``t+1`` from position ``t``).

Math
----

``cross_entropy``
    ``L = -Σ_t w_t * log π(a_t | x_<t) / Σ_t w_t``
    Standard SFT loss with optional per-token weighting.

``importance_sampling``
    ``L = -Σ_t w_t * (π_new(a_t|x_<t) / π_old(a_t|x_<t)) * A_t / Σ_t w_t``
    Unclipped policy gradient with importance weighting. Used when
    the rollout and training policy are close enough that no clipping
    is needed (e.g., REINFORCE on on-policy data).

``ppo``
    ``r_t = exp(log π_new(a_t|x_<t) - log π_old(a_t|x_<t))``
    ``L = -Σ_t w_t * min(r_t * A_t, clip(r_t, lo, hi) * A_t) / Σ_t w_t``
    Schulman et al. 2017 clipped surrogate objective. ``lo`` and
    ``hi`` come from ``loss_fn_config`` (``clip_low_threshold``,
    ``clip_high_threshold``, defaults ``1 - 0.2`` and ``1 + 0.2``).

``cispo``
    ``r_t = exp(log π_new(a_t|x_<t) - log π_old(a_t|x_<t))``
    ``L = -Σ_t w_t * stop_grad(clip(r_t, 0, r_max)) * log π_new(a_t|x_<t) * A_t / Σ_t w_t``
    Clipped Importance Sampling Policy Optimization (MiniMax-M1).
    The clipping is applied on the ``stop_grad`` side so the policy
    gradient flows purely through ``log π_new``, not through ``r_t``.
    ``r_max`` defaults to ``1 + 0.2`` via ``clip_high_threshold``.

``grpo``
    ``L = -Σ_t w_t * min(r_t * A_t, clip(r_t, 0.8, 1.2) * A_t) / Σ_t w_t
         + β * KL(π_new || π_old)``
    Group Relative Policy Optimization. Symmetric clipping with
    optional KL penalty (default β = 0.001). KL estimated via
    Schulman's k3 estimator.

``dapo``
    Same clip-surrogate as PPO but with asymmetric defaults
    (lo=0.8, hi=1.28) and no KL penalty. Token-mean aggregation.

``gspo``
    Group Stable Policy Optimization (token-level). Very tight
    clip (lo=1-3e-4, hi=1+4e-4). Aggregation: per-sequence token
    mean, then batch mean, so long sequences don't dominate.

``orpo``
    ``L = L_SFT(y_w | x) + λ * (-log σ(log_odds_ratio))``
    where ``log_odds_ratio = log(odds(y_w|x) / odds(y_l|x))`` and
    ``odds(y|x) = P(y|x) / (1 - P(y|x))``. ``P(y|x)`` is the
    *length-normalized* per-token probability of the response.
    Reference-free preference optimization (arXiv:2403.07691).
    Batch convention: chosen at even indices, rejected at odd —
    each adjacent (chosen, rejected) pair shares the same prompt.
    ``λ`` from ``loss_fn_config["orpo_lambda"]``, default 0.1.

``dro``
    Not implemented — the exact formulation is not in the public
    Tinker docs. Raises :class:`NotImplementedError` with a clear
    message. Contact the Tinker team for their specific DRO spec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

try:  # pragma: no cover
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore


@dataclass
class LossInputs:
    """Collected tensors + config for one loss computation.

    Attributes
    ----------
    logits:
        Model output, shape ``[B, T, V]``. Must support autograd.
    target_tokens:
        Per-position target indices, shape ``[B, T]``. For the causal
        LM shift convention this is ``input_ids[:, 1:]`` padded with
        ``-100`` in ignored positions.
    weights:
        Per-position float weight, shape ``[B, T]``. Zero means
        "ignore". For ``cross_entropy``-style SFT this is 1 on real
        tokens and 0 on padding/prompt. For RL it's typically the
        action mask.
    old_logprobs:
        Per-position log π_old(target), shape ``[B, T]``. Required
        for ``importance_sampling`` / ``ppo`` / ``cispo``; unused for
        ``cross_entropy``.
    advantages:
        Per-position advantage, shape ``[B, T]``. Required for all
        RL losses.
    loss_fn_config:
        Scalar config from the request (clip thresholds, KL weight).
    """

    logits: Any
    target_tokens: Any
    weights: Any
    old_logprobs: Optional[Any] = None
    advantages: Optional[Any] = None
    loss_fn_config: Optional[dict] = None


class LossNotImplementedError(NotImplementedError):
    """Raised when the requested loss isn't implemented server-side."""


# ─── Shared helpers ──────────────────────────────────────────────────────


def _new_logprobs_at_targets(logits, target_tokens):
    """Return ``log π_new(target_t | x_<t)``.

    Supports both 1-D targets (``[B, T]`` → returns ``[B, T]``) and
    2-D soft/top-K targets (``[B, T, K]`` → returns ``[B, T, K]``).
    The 2-D form is Tinker's wire format for top-K distillation —
    SDFT uses it to pass the teacher's top-K token IDs per position
    with per-token teacher probabilities as ``weights``.

    Positions where ``target_tokens == -100`` contribute a zero
    logprob so the weighted-mean reducer can ignore them without
    leaking non-finite values into the backward pass.
    """
    valid = target_tokens.ne(-100)
    safe_targets = target_tokens.masked_fill(~valid, 0)
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, T, V]

    if safe_targets.dim() == log_probs.dim() - 1:
        # 1-D: one target index per position.
        gathered = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    elif safe_targets.dim() == log_probs.dim():
        # 2-D: K target indices per position. log_probs is [B, T, V];
        # safe_targets is [B, T, K]; gather along V gives [B, T, K].
        gathered = log_probs.gather(-1, safe_targets)
    else:
        raise ValueError(
            f"target_tokens shape {tuple(target_tokens.shape)} incompatible "
            f"with logits shape {tuple(logits.shape)}"
        )
    return gathered * valid.to(gathered.dtype)


def _weighted_mean(per_token: Any, weights: Any) -> Any:
    """Return ``Σ (per_token * weights) / Σ weights``, with an
    epsilon floor to avoid divide-by-zero on fully-masked batches.
    """
    w = weights.to(per_token.dtype)
    denom = w.sum().clamp_min(1.0)
    return (per_token * w).sum() / denom


def _effective_weights(weights, valid_mask):
    """Combine the user-provided weights with the ``target != -100``
    mask. If ``weights`` is ``None``, fall back to pure masking.
    """
    if weights is None:
        return valid_mask.to(torch.float32)
    return weights.to(torch.float32) * valid_mask.to(torch.float32)


def _require_rl_inputs(loss_fn: str, old_logprobs: Any, advantages: Any) -> None:
    if old_logprobs is None:
        raise ValueError(f"{loss_fn} requires 'logprobs' in loss_fn_inputs (old policy logprobs)")
    if advantages is None:
        raise ValueError(f"{loss_fn} requires 'advantages' in loss_fn_inputs")


# ─── Dispatch ────────────────────────────────────────────────────────────


SUPPORTED_LOSS_FNS = (
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
    "grpo",
    "dapo",
    "gspo",
    "orpo",
)

# ``dro`` is a declared Tinker type but the exact formulation isn't
# public. We list it so calling code can discover it at runtime but
# raise when actually invoked.
DECLARED_LOSS_FNS = SUPPORTED_LOSS_FNS + ("dro",)


def compute(loss_fn: str, inputs: LossInputs) -> Any:
    """Dispatch to the named loss function.

    Raises
    ------
    ValueError
        If the loss name is unknown.
    LossNotImplementedError
        For ``dro`` (declared but not implemented).
    """
    if loss_fn == "cross_entropy":
        return _cross_entropy(inputs)
    if loss_fn == "importance_sampling":
        return _importance_sampling(inputs)
    if loss_fn == "ppo":
        return _ppo(inputs)
    if loss_fn == "cispo":
        return _cispo(inputs)
    if loss_fn == "grpo":
        return _grpo(inputs)
    if loss_fn == "dapo":
        return _dapo(inputs)
    if loss_fn == "gspo":
        return _gspo(inputs)
    if loss_fn == "orpo":
        return _orpo(inputs)
    if loss_fn == "dro":
        raise LossNotImplementedError(
            "dro is declared by the Tinker API but the exact "
            "formulation is not in the public docs. Contact "
            "tinker@thinkingmachines.ai for their DRO spec, or "
            "use forward_backward_custom to implement it client-side."
        )
    raise ValueError(f"unknown loss_fn: {loss_fn!r}")


# ─── Concrete losses ─────────────────────────────────────────────────────


def _cross_entropy(inputs: LossInputs) -> Any:

    # 1-D fast path with no custom weights: use the stock reduction.
    if inputs.weights is None and inputs.target_tokens.dim() == inputs.logits.dim() - 1:
        return F.cross_entropy(
            inputs.logits.view(-1, inputs.logits.size(-1)).float(),
            inputs.target_tokens.view(-1),
            ignore_index=-100,
        )

    # General path — handles both 1-D ([B, T]) and 2-D ([B, T, K]) targets,
    # and respects arbitrary per-token (or per-(token,k)) weights.
    # ``_new_logprobs_at_targets`` already zeros out ``-100`` positions,
    # so the weighted-mean reducer sees a clean signal.
    new_lp = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    per_token = -new_lp
    valid = inputs.target_tokens.ne(-100)
    effective = _effective_weights(inputs.weights, valid)
    return _weighted_mean(per_token, effective)


def _importance_sampling(inputs: LossInputs) -> Any:
    _require_rl_inputs("importance_sampling", inputs.old_logprobs, inputs.advantages)

    new_lp = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    ratio = (new_lp - inputs.old_logprobs).exp()
    per_token = -(ratio * inputs.advantages)
    valid = inputs.target_tokens.ne(-100)
    effective = _effective_weights(inputs.weights, valid)
    return _weighted_mean(per_token, effective)


def _ppo(inputs: LossInputs) -> Any:
    _require_rl_inputs("ppo", inputs.old_logprobs, inputs.advantages)
    cfg = inputs.loss_fn_config or {}
    lo = float(cfg.get("clip_low_threshold", 0.8))
    hi = float(cfg.get("clip_high_threshold", 1.2))
    if lo >= hi:
        raise ValueError(
            f"ppo clip_low_threshold ({lo}) must be strictly less than clip_high_threshold ({hi})"
        )

    new_lp = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    ratio = (new_lp - inputs.old_logprobs).exp()
    unclipped = ratio * inputs.advantages
    clipped = ratio.clamp(lo, hi) * inputs.advantages
    # PPO takes the MIN of unclipped and clipped (pessimistic bound).
    per_token = -torch.minimum(unclipped, clipped)
    valid = inputs.target_tokens.ne(-100)
    effective = _effective_weights(inputs.weights, valid)
    return _weighted_mean(per_token, effective)


def _cispo(inputs: LossInputs) -> Any:
    _require_rl_inputs("cispo", inputs.old_logprobs, inputs.advantages)
    cfg = inputs.loss_fn_config or {}
    r_max = float(cfg.get("clip_high_threshold", 1.2))
    r_min = float(cfg.get("clip_low_threshold", 0.0))
    if r_min < 0 or r_max <= r_min:
        raise ValueError(f"cispo thresholds invalid: clip_low={r_min}, clip_high={r_max}")

    new_lp = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    ratio = (new_lp - inputs.old_logprobs).exp()
    # CISPO: detach the clipped ratio so the gradient flows through
    # new_lp only. The ratio becomes a scalar multiplier, not a
    # differentiable term.
    weighting = ratio.clamp(r_min, r_max).detach() * inputs.advantages
    per_token = -(weighting * new_lp)
    valid = inputs.target_tokens.ne(-100)
    effective = _effective_weights(inputs.weights, valid)
    return _weighted_mean(per_token, effective)


def _grpo(inputs: LossInputs) -> Any:
    """GRPO — Group Relative Policy Optimization.

    NOTE: For production throughput, prefer the fused Liger kernel path
    (``LigerFusedLinearGRPOLoss``) which is integrated at the trainer
    level (operates on hidden states + lm_head weight, avoiding the
    full logits materialization). This pure-PyTorch implementation is
    the correctness reference and fallback when Liger is unavailable.

    Symmetric clipping (default [0.8, 1.2]) with optional KL penalty.

    ``L = -Σ_t w_t * min(r_t * A_t, clip(r_t, lo, hi) * A_t) / Σ_t w_t
         + β * KL(π_new || π_old)``

    The KL term is estimated per-token as:
    ``KL_t ≈ exp(log π_old - log π_new) - (log π_old - log π_new) - 1``
    (Schulman's unbiased KL estimator, k3).
    """
    _require_rl_inputs("grpo", inputs.old_logprobs, inputs.advantages)
    cfg = inputs.loss_fn_config or {}
    lo = float(cfg.get("clip_low_threshold", 0.8))
    hi = float(cfg.get("clip_high_threshold", 1.2))
    kl_beta = float(cfg.get("kl_beta", 0.001))

    new_lp = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    ratio = (new_lp - inputs.old_logprobs).exp()
    unclipped = ratio * inputs.advantages
    clipped = ratio.clamp(lo, hi) * inputs.advantages
    policy_loss = -torch.minimum(unclipped, clipped)

    valid = inputs.target_tokens.ne(-100)
    effective = _effective_weights(inputs.weights, valid)

    if kl_beta > 0:
        # Schulman k3 estimator: exp(old - new) - (old - new) - 1
        log_ratio = inputs.old_logprobs - new_lp
        kl = log_ratio.exp() - log_ratio - 1.0
        per_token = policy_loss + kl_beta * kl
    else:
        per_token = policy_loss

    return _weighted_mean(per_token, effective)


def _dapo(inputs: LossInputs) -> Any:
    """DAPO — Decoupled Asymmetric Policy Optimization.

    Asymmetric clipping (default [0.8, 1.28]) with no KL penalty and
    token-level mean aggregation. Key difference from PPO: the lower
    clip ratio is typically wider than the upper, and there's no KL
    regularization — the asymmetric clip alone constrains the update.
    """
    _require_rl_inputs("dapo", inputs.old_logprobs, inputs.advantages)
    cfg = inputs.loss_fn_config or {}
    lo = float(cfg.get("clip_low_threshold", 0.8))
    hi = float(cfg.get("clip_high_threshold", 1.28))

    new_lp = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    ratio = (new_lp - inputs.old_logprobs).exp()
    unclipped = ratio * inputs.advantages
    clipped = ratio.clamp(lo, hi) * inputs.advantages
    per_token = -torch.minimum(unclipped, clipped)
    valid = inputs.target_tokens.ne(-100)
    effective = _effective_weights(inputs.weights, valid)
    return _weighted_mean(per_token, effective)


def _gspo(inputs: LossInputs) -> Any:
    """GSPO-token — Group Stable Policy Optimization (token-level).

    Very tight symmetric clipping (default [1 - 3e-4, 1 + 4e-4]) with
    sequence-mean-token-mean aggregation. The tight clip range means
    the policy barely moves per step, relying on many steps to converge.

    Aggregation: first average per-token losses within each sequence,
    then average across sequences. This prevents long sequences from
    dominating the batch.
    """
    _require_rl_inputs("gspo", inputs.old_logprobs, inputs.advantages)
    cfg = inputs.loss_fn_config or {}
    lo = float(cfg.get("clip_low_threshold", 1 - 3e-4))
    hi = float(cfg.get("clip_high_threshold", 1 + 4e-4))

    new_lp = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    ratio = (new_lp - inputs.old_logprobs).exp()
    unclipped = ratio * inputs.advantages
    clipped = ratio.clamp(lo, hi) * inputs.advantages
    per_token = -torch.minimum(unclipped, clipped)

    valid = inputs.target_tokens.ne(-100)
    effective = _effective_weights(inputs.weights, valid)

    # Sequence-mean-token-mean: average per-token within each sequence,
    # then average across sequences.
    masked = per_token * effective
    seq_sums = masked.sum(dim=-1)  # [B]
    seq_counts = effective.sum(dim=-1).clamp_min(1.0)  # [B]
    seq_means = seq_sums / seq_counts  # [B]
    return seq_means.mean()


def _log1mexp(x: Any) -> Any:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0`` (Maechler 2012).

    Switches between ``log1p(-exp(x))`` (better for ``x << 0``) and
    ``log(-expm1(x))`` (better for ``x`` near 0); the threshold
    ``x = -ln(2)`` is where the two formulations have equal absolute
    error. The unselected branch is fed a safe constant so ``torch.where``
    doesn't propagate NaN gradients from the ``log`` of a non-positive
    argument.
    """
    threshold = -math.log(2.0)
    safe_far = torch.where(x < threshold, x, torch.full_like(x, threshold - 1.0))
    safe_near = torch.where(x >= threshold, x, torch.full_like(x, threshold + 1.0))
    return torch.where(
        x < threshold,
        torch.log1p(-torch.exp(safe_far)),
        torch.log(-torch.expm1(safe_near)),
    )


def _orpo(inputs: LossInputs) -> Any:
    """ORPO — Odds Ratio Preference Optimization (reference-free).

    Reference: Hong, Lee, Thorne, *ORPO: Monolithic Preference
    Optimization without Reference Model* (arXiv:2403.07691).

    Wire format conventions
    -----------------------
    * **Pair interleaving by index.** Even-index sequences in the
      batch are chosen (``y_w``); odd-index are rejected (``y_l``).
      Adjacent rows ``2i`` and ``2i + 1`` share the same prompt.
      Chosen here over a per-datum ``role`` tag because it matches
      the canonical client-side reference and the cookbook DPO
      convention; the only obligation on the gateway is to preserve
      the row order the client submitted (it does — the Datum list
      is forwarded unmodified). Validated by an even-batch-size
      check below.
    * **λ via ``loss_fn_config["orpo_lambda"]``.** The same channel
      ``cispo``/``dapo`` use for clip thresholds. Defaults to ``0.1``
      per the paper's recommendation.

    Math
    ----
    With ``P(y|x) = exp((1/m) Σ_t log π(y_t|x, y_<t))`` (length-
    normalized over the response mask in ``weights``):

        odds(y|x)        = P(y|x) / (1 - P(y|x))
        log_odds_ratio   = log( odds(y_w|x) / odds(y_l|x) )
                         = (log p_w - log p_l)
                           - ( log(1 - p_w) - log(1 - p_l) )
        L_OR             = -log σ(log_odds_ratio)
        L_SFT            = -mean(log p_w)        # length-normalized NLL
        L                = L_SFT + λ * L_OR

    Length normalization is mandatory: without it ``exp(Σ log p)``
    underflows for non-trivial response lengths and the OR term
    collapses. ``log(1 - exp(...))`` uses the Maechler 2012 stable
    formulation in :func:`_log1mexp`.

    Returns ``(loss, metrics)`` rather than a bare scalar — the metrics
    dict carries the ten ``orpo/*`` diagnostics the SDK surfaces in
    ``result.metrics``. The worker's ``_compute_loss`` unpacks the
    tuple and routes the metrics into the JobResult.
    """
    cfg = inputs.loss_fn_config or {}
    orpo_lambda = float(cfg.get("orpo_lambda", 0.1))

    # 1. Per-position logprobs at targets. [B, T]
    logprobs = _new_logprobs_at_targets(inputs.logits, inputs.target_tokens)
    weights = inputs.weights
    if weights is None:
        # Fall back to the target validity mask so a caller that
        # didn't supply explicit response weights still gets a sane
        # length-normalized average over real tokens.
        valid = inputs.target_tokens.ne(-100)
        weights = valid.to(logprobs.dtype)
    else:
        weights = weights.to(logprobs.dtype)

    if logprobs.shape[0] % 2 != 0:
        raise ValueError(
            f"orpo requires an even batch size (chosen at even indices, "
            f"rejected at odd); got batch size {logprobs.shape[0]}"
        )

    # 2. Length-normalized response logprob per sequence. [B]
    masked = logprobs * weights
    denom = weights.sum(dim=-1).clamp_min(1.0)
    avg_logp = masked.sum(dim=-1) / denom

    chosen = avg_logp[0::2]   # even indices → y_w
    rejected = avg_logp[1::2]  # odd indices  → y_l

    # 3. SFT term — NLL on the chosen response, length-normalized so
    #    it sits on the same scale as L_OR (both are per-token log
    #    probabilities, not full-sequence sums).
    sft_loss = -chosen.mean()

    # 4. Odds-ratio term in log space. Clamp the inputs to log(1-p)
    #    away from 0 so a near-deterministic policy doesn't overflow
    #    log1p(-exp(0)) = log(0). The clamp is *forward-only*; the
    #    gradient still flows through the unclamped ``chosen`` /
    #    ``rejected`` because those drive the SFT term and the
    #    ``(chosen - rejected)`` part of the log-odds.
    eps = 1e-7
    chosen_clamped = chosen.clamp(max=-eps)
    rejected_clamped = rejected.clamp(max=-eps)
    log_odds = (chosen - rejected) - (
        _log1mexp(chosen_clamped) - _log1mexp(rejected_clamped)
    )
    or_loss = -F.logsigmoid(log_odds).mean()

    loss = sft_loss + orpo_lambda * or_loss

    # 5. Diagnostics — mirror the client-side reference implementation.
    with torch.no_grad():
        log_sigmoid_lor = F.logsigmoid(log_odds)
        metrics = {
            "loss": float(loss.detach().item()),
            "orpo/sft_loss": float(sft_loss.detach().item()),
            "orpo/or_loss": float(or_loss.detach().item()),
            "orpo/log_odds_ratio": float(log_odds.mean().item()),
            "orpo/accuracy": float((chosen > rejected).float().mean().item()),
            "orpo/margin": float((chosen - rejected).mean().item()),
            "orpo/chosen_logp": float(chosen.mean().item()),
            "orpo/rejected_logp": float(rejected.mean().item()),
            "orpo/chosen_reward": float(
                (orpo_lambda * log_sigmoid_lor).mean().item()
            ),
            "orpo/lambda": orpo_lambda,
        }
    return loss, metrics


# TODO(perf): no fused-kernel path yet — `fused_losses.py` targets the
# CE/IS/PPO families that fit the fused-CE shape. ORPO needs full
# logits materialization for length-normalized response logprobs;
# revisit if the per-step wall-time becomes a bottleneck.


# ─── Helpers for forward_backward_custom ────────────────────────────────


def compute_target_logprobs(logits: Any, target_tokens: Any) -> Any:
    """Return ``log π(target_t | x_<t)`` per position, shape ``[B, T]``.

    Follows the Tinker convention: inputs are pre-aligned by the client.
    At position ``i``, ``logits[i]`` scores against ``target_tokens[i]``.
    The output has the same length as the input — no shifting, no zero
    padding.
    """
    return _new_logprobs_at_targets(logits, target_tokens)


def surrogate_loss_from_grad(logprobs: Any, grad_logprobs: Any) -> Any:
    """Build the server-side surrogate loss for custom-function backward.

    We detach ``grad_logprobs`` so autograd treats it as a constant
    coefficient, then multiply by ``logprobs`` (which requires_grad).
    The chain rule gives
    ``∂surrogate/∂θ = Σ grad_logprobs * ∂logprobs/∂θ``,
    exactly the parameter gradient of the user's custom loss even
    though the custom loss itself never runs here.
    """
    if logprobs.shape != grad_logprobs.shape:
        raise ValueError(
            f"logprobs shape {tuple(logprobs.shape)} != "
            f"grad_logprobs shape {tuple(grad_logprobs.shape)}"
        )
    return (logprobs * grad_logprobs.detach()).sum()
