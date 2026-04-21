# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""GPU worker tests that run on CPU with a tiny synthetic model.

These exercise the full worker pipeline — init_session → forward_backward
→ optim_step → sample → save_weights — without requiring a GPU or a
Huggingface download. A 2-layer GPT-2-style LM is built in-memory.
"""

from __future__ import annotations

import io
import os

import pytest
import pytest_asyncio

torch = pytest.importorskip("torch")

# Force CPU for these tests regardless of the ambient CUDA state.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

pytest.importorskip("peft")
pytest.importorskip("transformers")


@pytest_asyncio.fixture
async def cpu_worker(platform_config, tmp_path, monkeypatch):
    """A GPUWorker wired to a tiny from-config transformers model on CPU."""
    from transformers import GPT2Config, GPT2LMHeadModel

    VOCAB = 64
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    class FakeTokenizer:
        """Minimal stand-in for HF tokenizer used by GPUWorker."""

        def __init__(self) -> None:
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.pad_token = "<pad>"

        def decode(self, ids, skip_special_tokens: bool = False) -> str:
            return " ".join(str(int(i)) for i in ids)

    tok = FakeTokenizer()

    from hatchery.core.worker import GPUWorker

    worker = GPUWorker(
        worker_id="cpu-worker",
        base_model_name="gpt2-tiny",
        config=platform_config,
        device="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
        load_model=False,
    )
    worker._raw_base = model
    worker.tokenizer = tok
    yield worker


async def test_init_forward_backward_optim_step(cpu_worker, platform_config):
    worker = cpu_worker
    session_id = "cpu-sess-1"

    # init_session
    await worker._handle_init_session(
        session_id,
        {
            "rank": 4,
            "lora_alpha": 8,
            "target_modules": ["c_attn"],
        },
    )

    # Sanity: state is now on the worker's local disk. Under Ship 3,
    # the worker writes session state locally and only mirrors to the
    # remote object store via MirroredSessionStateStore (extension). Core
    # tests use LocalSessionStateStore where flush is a no-op, so
    # verification must target ``worker._state.local`` directly.
    weights_bytes = await worker._state.local.get(
        f"sessions/{session_id}/live_state/lora_weights.pt"
    )
    assert len(weights_bytes) > 0

    # forward_backward
    data = [{"input_ids": [1, 2, 3, 4, 5]}, {"input_ids": [6, 7, 8]}]
    fb_result, fb_metrics = await worker._handle_forward_backward(
        session_id, {"data": data, "loss_fn": "cross_entropy"}
    )
    assert "loss" in fb_result
    assert fb_result["num_tokens"] > 0
    assert fb_result["accum_steps"] == 1

    # Cost dimensions should be present for internal pricing analysis.
    cd = fb_metrics.get("cost_dimensions")
    assert cd is not None, "cost_dimensions missing from forward_backward metrics"
    assert cd["model_name"] == "gpt2-tiny"
    assert cd["batch_size"] == 2
    assert cd["max_seq_len"] == 5  # padded to longest sequence
    assert cd["lora_rank"] == 4
    assert cd["loss_fn"] == "cross_entropy"
    assert isinstance(cd["fused_path"], bool)
    assert cd["dp_degree"] >= 1

    # optim_step
    step_result, _ = await worker._handle_optim_step(
        session_id,
        {"learning_rate": 1e-3},
    )
    assert step_result["step"] == 1


async def test_grad_accum_skipped_after_optim_step(cpu_worker, platform_config):
    """After optim_step, grad_accum.pt must serialize as an empty dict.

    Between the first forward_backward and optim_step, grad_accum holds
    real gradients and the saved blob is proportional to LoRA param count.
    Right after optim_step the runtime resets grad_accum to zero tensors —
    saving those would cost MBs per step on remote sync for no benefit. The
    save path detects accum_steps == 0 and writes ``{}`` instead.
    """
    worker = cpu_worker
    sid = "cpu-grad-empty"
    await worker._handle_init_session(
        sid, {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}
    )

    prefix = f"sessions/{sid}/live_state"
    data = [{"input_ids": [1, 2, 3, 4, 5]}]

    # After fwd_bwd, grad_accum has real values; the saved blob is
    # non-trivial (exact size depends on LoRA param count, but > a few
    # hundred bytes even for the 2-layer tiny GPT-2 used here).
    await worker._handle_forward_backward(sid, {"data": data, "loss_fn": "cross_entropy"})
    mid_bytes = await worker._state.local.get(f"{prefix}/grad_accum.pt")
    mid_loaded = torch.load(io.BytesIO(mid_bytes), map_location="cpu", weights_only=False)
    assert len(mid_loaded) > 0
    assert any(t.any().item() for t in mid_loaded.values())

    # After optim_step, grad_accum is reset; the saved blob collapses to
    # an empty dict pickle (< 100 bytes for torch.save({})).
    await worker._handle_optim_step(sid, {"learning_rate": 1e-3})
    post_bytes = await worker._state.local.get(f"{prefix}/grad_accum.pt")
    assert len(post_bytes) < len(mid_bytes)
    post_loaded = torch.load(io.BytesIO(post_bytes), map_location="cpu", weights_only=False)
    assert post_loaded == {}


async def test_sample_returns_tokens(cpu_worker):
    worker = cpu_worker
    session_id = "cpu-sess-2"
    await worker._handle_init_session(
        session_id,
        {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]},
    )
    result, _ = await worker._handle_sample(
        session_id,
        {
            "prompt_tokens": [10, 20, 30],
            "max_tokens": 4,
            "temperature": 0.0,
            "n": 1,
        },
    )
    assert len(result["sequences"]) == 1
    assert len(result["sequences"][0]) <= 4


async def test_compute_logprobs_shape(cpu_worker):
    worker = cpu_worker
    session_id = "cpu-sess-3"
    await worker._handle_init_session(
        session_id,
        {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]},
    )
    result, _ = await worker._handle_compute_logprobs(
        session_id,
        {"input_tokens": [[1, 2, 3, 4, 5]]},
    )
    # N-1 logprobs for N tokens (no prediction for the first).
    assert len(result["logprobs"]) == 1
    assert len(result["logprobs"][0]) == 4


async def test_state_persists_across_load(cpu_worker, platform_config):
    """Tear down the session runtime and ensure it reloads from object store."""
    worker = cpu_worker
    sid = "cpu-sess-4"
    await worker._handle_init_session(
        sid, {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}
    )
    data = [{"input_ids": [1, 2, 3, 4]}]
    await worker._handle_forward_backward(sid, {"data": data})

    # Evict from cache.
    worker._cache.pop(sid)
    adapter = worker._adapter_name(sid)
    worker._peft.delete_adapter(adapter)

    # Another forward_backward should reload state transparently.
    await worker._handle_forward_backward(sid, {"data": data})
    runtime = worker._cache.get(sid)
    assert runtime is not None
    assert runtime.meta["accum_steps"] == 2


async def test_forward_backward_custom_matches_direct_ce(cpu_worker):
    """Exercise the two-legged custom-backward protocol.

    Running ``forward_custom_step1`` + a client-side cross_entropy
    gradient + ``forward_custom_step2`` should land the same parameter
    gradient as a plain ``forward_backward`` with ``cross_entropy``.
    This pins the correctness of the surrogate-loss trick.

    We run both paths on a single session, snapshotting (and zeroing)
    ``grad_accum`` between runs so the two gradient sets start from
    identical LoRA weights.
    """
    worker = cpu_worker
    sid = "custom-sess"
    await worker._handle_init_session(
        sid,
        {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]},
    )

    # Pre-shifted data (Tinker convention — client pre-aligns).
    shifted_data = [
        {"input_ids": [2, 3, 5, 7], "labels": [3, 5, 7, 11]},
        {"input_ids": [1, 4], "labels": [4, 9]},
    ]

    # GPT-2 has attention dropout even in train mode, so three separate
    # forward passes would draw three different masks. Force eval mode
    # for this test so the three forward passes are deterministic —
    # the worker handlers call ``.train()`` internally, so we neuter
    # that for the test duration. In production the custom-backward
    # protocol has to arrange for matched RNG state (either by seeding
    # or by caching activations) — that's out of scope here; we're
    # verifying the surrogate loss math.
    original_train = worker._peft.train
    worker._peft.train = lambda *_args, **_kwargs: worker._peft
    worker._peft.eval()
    try:
        # Path A: stock cross_entropy forward_backward (pre-shifted per Tinker convention).
        await worker._handle_forward_backward(
            sid, {"data": shifted_data, "loss_fn": "cross_entropy"}
        )
        runtime = worker._cache.get(sid)
        assert runtime is not None
        direct_grads = {k: v.clone() for k, v in runtime.grad_accum.items()}

        # Reset the accumulator so path B starts clean.
        runtime.grad_accum.clear()
        runtime.meta["accum_steps"] = 0

        # Path B: two-leg custom protocol. Step1 → client-side CE loss →
        # backprop through the returned logprobs → step2 with grad_logprobs.
        step1, _ = await worker._handle_forward_custom_step1(
            sid, {"data": shifted_data, "custom_id": "run-1"}
        )
        # Step1 returns per-item logprob lists aligned with the
        # pre-shifted input (Tinker convention — client pre-shifts).
        assert isinstance(step1["logprobs"], list)
        assert isinstance(step1["logprobs"][0], list)
        assert len(step1["logprobs"]) == 2
        assert len(step1["logprobs"][0]) == 4  # [2,3,5,7] → T=4
        assert len(step1["logprobs"][1]) == 2  # [1,4]     → T=2

        # Client-side custom loss: compute per-item loss and back-prop.
        logprobs_list = [
            torch.tensor(lp, dtype=torch.float32, requires_grad=True) for lp in step1["logprobs"]
        ]
        all_lp = torch.cat(logprobs_list)
        client_loss = -all_lp.sum() / max(all_lp.numel(), 1)
        client_loss.backward()
        # Return ragged grad lists matching the step1 shape.
        grad_logprobs = [lp.grad.tolist() for lp in logprobs_list]

        step2, _ = await worker._handle_forward_custom_step2(
            sid,
            {"custom_id": "run-1", "grad_logprobs": grad_logprobs},
        )
        assert step2["accum_steps"] == 1
        # Cache must be evicted after step2.
        assert "run-1" not in runtime.custom_cache

        assert set(runtime.grad_accum) == set(direct_grads)
        for name, direct_g in direct_grads.items():
            surrogate_g = runtime.grad_accum[name]
            assert torch.allclose(direct_g, surrogate_g, atol=1e-4), (
                f"grad mismatch on {name}: max diff {(direct_g - surrogate_g).abs().max().item()}"
            )
    finally:
        worker._peft.train = original_train


async def test_custom_step1_returns_t_length_ragged_logprobs(cpu_worker):
    """Pin the step1 return shape contract: per-item logprob lists match
    each Datum's input_ids length (pre-shifted per Tinker convention).
    """
    worker = cpu_worker
    sid = "shape-test"
    await worker._handle_init_session(
        sid, {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}
    )
    # Pre-shifted data (Tinker convention).
    data = [
        {"input_ids": [1, 2], "labels": [2, 3]},  # T=2
        {"input_ids": [4, 5, 6, 7], "labels": [5, 6, 7, 8]},  # T=4
        {"input_ids": [9], "labels": [10]},  # T=1
    ]
    step1, _ = await worker._handle_forward_custom_step1(
        sid, {"data": data, "custom_id": "shape-check"}
    )
    lps = step1["logprobs"]
    shapes = step1["shapes"]

    assert len(lps) == 3
    assert len(lps[0]) == 2, f"item 0 should be T=2, got {len(lps[0])}"
    assert len(lps[1]) == 4, f"item 1 should be T=4, got {len(lps[1])}"
    assert len(lps[2]) == 1, f"item 2 should be T=1, got {len(lps[2])}"

    # Shapes match the logprob list lengths.
    assert shapes[0] == [2]
    assert shapes[1] == [4]
    assert shapes[2] == [1]


async def test_forward_custom_step2_without_step1_raises(cpu_worker):
    """Step2 must refuse to run if step1 wasn't called for that custom_id."""
    worker = cpu_worker
    await worker._handle_init_session(
        "custom-nostep1",
        {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]},
    )
    with pytest.raises(ValueError, match="no cached forward"):
        await worker._handle_forward_custom_step2(
            "custom-nostep1",
            {"custom_id": "missing", "grad_logprobs": [[0.0]]},
        )


async def test_sdft_topk_end_to_end(cpu_worker):
    """Full SDFT round trip: wire-format Datum with 2-D target_tokens +
    2-D weights (teacher top-K distribution) → ``_datum_to_training_item``
    → worker collate → ``cross_entropy`` loss.

    The whole point of SDFT is that a single ``forward_backward`` call
    with ``loss_fn="cross_entropy"`` can consume a teacher's top-K
    distribution per position. This test pins the end-to-end plumbing:
    wire reshape, 2-D label detection in ``_collate``, and the
    ``_new_logprobs_at_targets`` 2-D gather path in ``losses.py``.
    """
    from hatchery.core.tinker_compat import (
        Datum,
        ModelInput,
        TensorData,
        _datum_to_training_item,
    )

    worker = cpu_worker
    sid = "sdft-sess"
    await worker._handle_init_session(
        sid,
        {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]},
    )

    # Build a SDFT-shaped Datum the way the cookbook does: 4 tokens of
    # prompt, top-3 teacher distribution at every position. The inner
    # lists carry the teacher's chosen token IDs and the teacher's
    # (already-normalized) probability mass over those tokens.
    T, K = 4, 3
    mi = ModelInput(chunks=[{"type": "encoded_tokens", "tokens": [10, 20, 30, 40]}])
    target_tokens = TensorData(
        dtype="int32",
        shape=[T, K],
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # T*K flat
    )
    weights = TensorData(
        dtype="float32",
        shape=[T, K],
        data=[1 / K] * (T * K),  # uniform teacher
    )
    datum = Datum(
        model_input=mi,
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )

    # Step 1: the wire → training item translation.
    item = _datum_to_training_item(datum)
    assert item["input_ids"] == [10, 20, 30, 40]
    assert item["labels"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert item["weights"] == [
        [1 / K, 1 / K, 1 / K],
        [1 / K, 1 / K, 1 / K],
        [1 / K, 1 / K, 1 / K],
        [1 / K, 1 / K, 1 / K],
    ]

    # Step 2: full forward_backward through the worker. Succeeding here
    # proves the 2-D labels survive collate + loss.
    result, _ = await worker._handle_forward_backward(
        sid,
        {"data": [item], "loss_fn": "cross_entropy"},
    )
    assert result["loss"] > 0  # CE is always > 0 for non-degenerate targets
    assert result["num_tokens"] > 0
    assert result["accum_steps"] == 1

    # And the LoRA gradients should have been accumulated.
    runtime = worker._cache.get(sid)
    assert runtime is not None
    assert len(runtime.grad_accum) > 0


async def test_fused_ce_matches_direct_ce(cpu_worker, monkeypatch):
    """Fused CE path must produce the same loss + LoRA gradients as
    the direct (materialize-logits) path.

    We run the same forward_backward twice on the same session-state
    once via the fused path (the default, since GPT-2 is fused-
    eligible) and once with fused eligibility forced off. Grad_accum
    before and after each run is snapshotted and compared.
    """
    import hatchery.core.fused_losses as fused_mod

    worker = cpu_worker
    sid = "fused-parity"
    await worker._handle_init_session(
        sid,
        {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]},
    )
    data = [{"input_ids": [2, 3, 5, 7, 11]}, {"input_ids": [1, 4, 9]}]

    # Dropout makes three forward passes non-deterministic — force
    # eval mode for both runs so we can compare numerically. Same
    # trick as ``test_forward_backward_custom_matches_direct_ce``.
    original_train = worker._peft.train
    worker._peft.train = lambda *_a, **_kw: worker._peft
    worker._peft.eval()
    try:
        # Path A: fused (default).
        fb_a, _ = await worker._handle_forward_backward(
            sid, {"data": data, "loss_fn": "cross_entropy"}
        )
        runtime = worker._cache.get(sid)
        assert runtime is not None
        fused_loss = fb_a["loss"]
        fused_grads = {k: v.clone() for k, v in runtime.grad_accum.items()}

        runtime.grad_accum.clear()
        runtime.meta["accum_steps"] = 0

        # Path B: direct. Force the eligibility check to return False
        # so the worker drops into the logits-materialization branch.
        monkeypatch.setattr(
            fused_mod,
            "is_fused_eligible",
            lambda **kwargs: False,
        )
        fb_b, _ = await worker._handle_forward_backward(
            sid, {"data": data, "loss_fn": "cross_entropy"}
        )
        direct_loss = fb_b["loss"]
        direct_grads = runtime.grad_accum

        assert abs(fused_loss - direct_loss) < 1e-4, (
            f"loss mismatch: fused={fused_loss}, direct={direct_loss}"
        )
        assert set(fused_grads) == set(direct_grads)
        for name, fused_g in fused_grads.items():
            direct_g = direct_grads[name]
            assert torch.allclose(fused_g, direct_g, atol=1e-4), (
                f"grad mismatch on {name}: max diff {(fused_g - direct_g).abs().max().item()}"
            )
    finally:
        worker._peft.train = original_train


async def test_load_weights_resumes_from_checkpoint(cpu_worker, platform_config):
    """Save a checkpoint, mutate the adapter, load the checkpoint back,
    and verify the adapter state was restored.
    """
    worker = cpu_worker
    sid = "load-test"
    await worker._handle_init_session(
        sid, {"rank": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}
    )

    # Train a bit so the adapter has non-initial weights.
    data = [{"input_ids": [1, 2, 3, 4, 5]}]
    await worker._handle_forward_backward(sid, {"data": data})
    await worker._handle_optim_step(sid, {"learning_rate": 1e-2})

    # Save the current state as a checkpoint. Under Ship 3 the worker
    # writes live state to local disk; _handle_load_weights reads the
    # checkpoint from config.objects, so we manually flush the single
    # snapshot file from local to remote under the checkpoint prefix.
    live_prefix = f"sessions/{sid}/live_state"
    ckpt_prefix = f"sessions/{sid}/checkpoints/ckpt-resume"
    snapshot = await worker._state.local.get(f"{live_prefix}/lora_weights.pt")
    await platform_config.objects.put(f"{ckpt_prefix}/lora_weights.pt", snapshot)

    # Capture the adapter weights at checkpoint time.
    from peft import get_peft_model_state_dict

    adapter = worker._adapter_name(sid)
    ckpt_state = {
        k: v.clone()
        for k, v in get_peft_model_state_dict(worker._peft, adapter_name=adapter).items()
    }

    # Train more — the adapter drifts away from the checkpoint.
    for _ in range(3):
        await worker._handle_forward_backward(sid, {"data": data})
        await worker._handle_optim_step(sid, {"learning_rate": 1e-2})

    # Verify the adapter has actually changed.
    current_state = get_peft_model_state_dict(worker._peft, adapter_name=adapter)
    drifted = any(
        not torch.allclose(current_state[k], ckpt_state[k], atol=1e-6) for k in ckpt_state
    )
    assert drifted, "adapter didn't change after extra training"

    # Load the checkpoint — should restore the adapter to checkpoint state.
    result, _ = await worker._handle_load_weights(sid, {"checkpoint_prefix": ckpt_prefix})
    assert result["type"] == "load_weights"

    restored_state = get_peft_model_state_dict(worker._peft, adapter_name=adapter)
    for k in ckpt_state:
        assert torch.allclose(restored_state[k], ckpt_state[k], atol=1e-2), (
            f"{k} not restored: max diff {(restored_state[k] - ckpt_state[k]).abs().max().item()}"
        )


async def test_overfit_single_batch_reduces_loss(cpu_worker):
    """Classic sanity check: LoRA should be able to memorize one batch."""
    worker = cpu_worker
    sid = "cpu-overfit"
    await worker._handle_init_session(
        sid,
        {
            "rank": 8,
            "lora_alpha": 16,
            "target_modules": ["c_attn"],
        },
    )
    torch.manual_seed(0)
    data = [{"input_ids": [5, 7, 11, 13, 17, 19]}]

    first_loss = None
    last_loss = None
    for _step in range(6):
        result, _ = await worker._handle_forward_backward(
            sid, {"data": data, "loss_fn": "cross_entropy"}
        )
        if first_loss is None:
            first_loss = result["loss"]
        last_loss = result["loss"]
        await worker._handle_optim_step(sid, {"learning_rate": 3e-2})

    assert last_loss < first_loss, f"loss did not decrease: {first_loss} -> {last_loss}"


# ── FP capacity gate ─────────────────────────────────────────────


def test_fft_capacity_gate_rejects_second_fp_session(cpu_worker):
    """One FP session per worker — the gate fires before any state is
    materialized so callers see a clean rejection.

    ``_fp_base_state`` is the worker's authoritative live-FP tracker;
    seeding an entry plus a runtime with ``total_steps > 0`` simulates
    an actively-training FP session (zombies with ``total_steps == 0``
    get evicted instead — that path is covered by the dedicated
    ``test_worker_capacity_gate`` suite).
    """
    from hatchery.core.worker import _SessionRuntime

    worker = cpu_worker

    # No FP yet — gate is a no-op.
    worker._enforce_fft_capacity("fp-new")

    # Simulate an actively-training FP session.
    worker._fp_base_state["fp-existing"] = {}
    worker._cache.put(
        "fp-existing",
        _SessionRuntime(
            session_id="fp-existing",
            training_mode="full_param",
            meta={"total_steps": 1, "accum_steps": 0},
        ),
    )
    with pytest.raises(RuntimeError, match="single-FFT-per-worker"):
        worker._enforce_fft_capacity("fp-new")

    # Re-init of the *same* FP session is allowed (idempotent).
    worker._enforce_fft_capacity("fp-existing")


# ── FP reload after eviction ─────────────────────────────────────


def _attach_test_slot(worker) -> None:
    """Wire a minimal :class:`PoolSlot` so FP sessions can initialize.

    The ``cpu_worker`` fixture sets ``_raw_base`` directly (bypassing
    the model pool) which is sufficient for LoRA paths but not for FP:
    :func:`_attach_full_param_session` reads from ``_slot.pristine_sd``.
    """
    from hatchery.core.model_pool import PoolSlot

    worker._slot = PoolSlot(
        base_model_name=worker.base_model_name,
        raw_base=worker._raw_base,
        tokenizer=worker.tokenizer,
    )


async def test_fp_reload_after_optim_step_and_eviction(cpu_worker):
    """An FP session that has trained at least one step must survive an
    eviction + reload cycle without KeyError.

    Failure mode (observed on staging 2026-04-18): ``_save_fp_init_marker``
    writes ``session_meta.json`` with ``fp_pristine_init=True`` but no
    ``lora_weights.pt``. After the first optim_step, ``_save_session_to_store``
    rewrites the meta *without* that flag, and also writes the real
    ``lora_weights.pt``. So in the single-worker / local-disk case, reload
    works. The bug surfaces when a different worker picks up the session:
    its local disk has nothing, it falls back to the remote store, the
    remote's stale meta may have ``fp_pristine_init=True`` while
    ``lora_weights.pt`` hasn't been mirrored yet — reload's fast-path
    branch then tries to read the missing blob.

    This test covers the single-worker path end-to-end. Cross-worker
    mirror-race is exercised in the extension MirroredSessionStateStore
    suite.
    """
    worker = cpu_worker
    _attach_test_slot(worker)
    sid = "cpu-fp-reload"

    # FP session: no rank in the payload.
    await worker._handle_init_session(sid, {})
    data = [{"input_ids": [1, 2, 3, 4, 5]}]
    await worker._handle_forward_backward(sid, {"data": data, "loss_fn": "cross_entropy"})
    await worker._handle_optim_step(sid, {"learning_rate": 1e-4})

    # Evict the runtime. ``_on_evict`` clears ``_fp_base_state`` and
    # restores pristine base into the slot, matching what a real
    # SmartLoRACache eviction would do under LRU pressure.
    worker._cache.pop(sid)
    # Simulate cache miss: force a reload on the next op.
    assert worker._cache.get(sid) is None

    # Second fwd_bwd must succeed — the reload path should not raise.
    await worker._handle_forward_backward(sid, {"data": data, "loss_fn": "cross_entropy"})
    runtime = worker._cache.get(sid)
    assert runtime is not None
    assert runtime.training_mode == "full_param"
    assert runtime.meta["accum_steps"] == 1
    assert runtime.meta["total_steps"] == 1


async def test_fp_reload_when_remote_meta_has_stale_init_marker(cpu_worker):
    """Cross-worker race variant A: remote meta still has
    ``fp_pristine_init=True`` but ``lora_weights.pt`` is missing because
    the outgoing worker's async mirror hadn't propagated the full payload
    yet.

    The fast-path gated on ``fp_pristine_init`` runs without attempting
    the missing file, so reload should succeed even if the source has a
    stale marker + no weights blob.
    """
    import json

    worker = cpu_worker
    _attach_test_slot(worker)
    sid = "cpu-fp-stale-marker"

    prefix = f"sessions/{sid}/live_state"
    await worker._state.local.put(
        f"{prefix}/session_meta.json",
        json.dumps(
            {
                "training_mode": "full_param",
                "accum_steps": 0,
                "total_steps": 0,
                "fp_pristine_init": True,
                "snapshot_version": 0,
                "delta_count": 0,
                "optim_snapshot_version": 0,
                "optim_delta_count": 0,
            }
        ).encode("utf-8"),
    )

    runtime = await worker._load_session_from_store(sid)
    assert runtime.training_mode == "full_param"
    assert runtime.meta.get("fp_pristine_init") is True


async def test_fp_reload_from_remote_missing_weights_blob(cpu_worker):
    """Cross-worker race variant B: remote has the post-optim_step meta
    (``fp_pristine_init`` absent, ``total_steps=1``) but the weights
    blob hasn't mirrored yet.

    The outgoing worker's ``_mirror_session`` uploads all local keys in
    parallel via ``asyncio.gather`` — ordering is non-deterministic, so
    remote can end up with the newer meta but without the matching
    ``lora_weights.pt``. Load must not blow up with ``KeyError`` on the
    missing blob; it should fall back to the slot's pristine snapshot
    (the base weights this session started from, which are the only
    thing it's diverged from by <= 1 optim step of magnitude).

    This directly models the failure observed on staging 2026-04-18,
    where Qwen FFT smoke crashed at step 3 with
    ``KeyError: lora_weights.pt``.
    """
    import json

    worker = cpu_worker
    _attach_test_slot(worker)
    sid = "cpu-fp-no-weights"

    # Force reload to go through the remote (config.objects) path by
    # leaving worker._state.local empty and writing only to the
    # platform object store.
    prefix = f"sessions/{sid}/live_state"
    remote = worker.config.objects
    await remote.put(
        f"{prefix}/session_meta.json",
        json.dumps(
            {
                "training_mode": "full_param",
                "accum_steps": 0,
                "total_steps": 1,
                # No fp_pristine_init — it was cleared by the first
                # _save_session_to_store after optim_step.
                "snapshot_version": 1,
                "delta_count": 0,
                "optim_snapshot_version": 0,
                "optim_delta_count": 0,
            }
        ).encode("utf-8"),
    )
    # Deliberately no lora_weights.pt on remote.

    # Expected: reload succeeds by falling back to pristine. Today this
    # raises KeyError on ``lora_weights.pt``.
    runtime = await worker._load_session_from_store(sid)
    assert runtime.training_mode == "full_param"
    assert runtime.meta["total_steps"] == 1
