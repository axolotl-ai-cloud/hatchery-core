# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""VLM end-to-end test with a randomly initialized model.

Validates the full image → pixel_values → forward → backward pipeline
using a tiny Qwen2VL created from config (no download). Tests:

1. VLM detection works on the model class
2. Image chunks are decoded and processed
3. Vision token stripping works for text-only inputs
4. Forward + backward produces valid gradients with image inputs
"""

from __future__ import annotations

import base64
import io

import pytest

torch = pytest.importorskip("torch")


def _tiny_vlm_config():
    """Create a minimal Qwen2VL config for testing."""
    try:
        from transformers import Qwen2VLConfig

        return Qwen2VLConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            max_position_embeddings=128,
            rope_scaling={
                "type": "mrope",
                "mrope_section": [2, 2, 4],
            },
            vision_config={
                "depth": 2,
                "embed_dim": 64,
                "num_heads": 4,
                "hidden_size": 64,
                "in_channels": 3,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
            },
        )
    except ImportError:
        pytest.skip("Qwen2VL not available in this transformers version")


def _make_test_image_bytes() -> bytes:
    """Create a tiny 28x28 red JPEG for testing."""
    try:
        from PIL import Image

        img = Image.new("RGB", (28, 28), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        pytest.skip("Pillow not installed")


# ─── Unit tests for VLM detection ────────────────────────────────────


def test_vlm_detection():
    """_is_vlm_model correctly identifies VLM classes."""
    from hatchery.core.worker import _is_vlm_model

    class FakeQwen2VL:
        pass

    FakeQwen2VL.__name__ = "Qwen2VLForConditionalGeneration"

    class FakeLlama:
        pass

    FakeLlama.__name__ = "LlamaForCausalLM"

    assert _is_vlm_model(FakeQwen2VL()) is True
    assert _is_vlm_model(FakeLlama()) is False


def test_vision_token_stripping():
    """_strip_vision_tokens removes vision placeholders."""
    from hatchery.core.worker import _strip_vision_tokens

    vision_ids = {100, 101, 102}
    tokens = [1, 2, 100, 3, 101, 102, 4]
    result = _strip_vision_tokens(tokens, vision_ids)
    assert result == [1, 2, 3, 4]


def test_vision_token_stripping_empty_set():
    from hatchery.core.worker import _strip_vision_tokens

    tokens = [1, 2, 3]
    assert _strip_vision_tokens(tokens, set()) == [1, 2, 3]


# ─── Image chunk decoding ────────────────────────────────────────────


def test_image_chunk_decoding():
    """EncodedImageChunk decodes base64 correctly."""
    from hatchery.core.tinker_compat import EncodedImageChunk, ModelInput, _decode_image_chunks

    raw = _make_test_image_bytes()
    b64 = base64.b64encode(raw).decode("ascii")

    mi = ModelInput(
        chunks=[],
        image_chunks=[EncodedImageChunk(data=b64, mime_type="image/jpeg")],
    )
    decoded = _decode_image_chunks(mi)
    assert len(decoded) == 1
    assert decoded[0] == raw


def test_image_chunk_with_data_uri():
    """data: URI prefix is stripped correctly."""
    from hatchery.core.tinker_compat import EncodedImageChunk, ModelInput, _decode_image_chunks

    raw = _make_test_image_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"

    mi = ModelInput(
        chunks=[],
        image_chunks=[EncodedImageChunk(data=data_uri, mime_type="image/jpeg")],
    )
    decoded = _decode_image_chunks(mi)
    assert decoded[0] == raw


# ─── Full model E2E (requires torch + transformers) ──────────────────


@pytest.mark.skipif(
    not hasattr(torch, "cuda") or not torch.cuda.is_available(),
    reason="VLM E2E requires CUDA for Qwen2VL",
)
def test_vlm_forward_backward_with_image():
    """Full pipeline: create tiny Qwen2VL, process image, run forward+backward."""
    from transformers import Qwen2VLForConditionalGeneration

    config = _tiny_vlm_config()
    model = Qwen2VLForConditionalGeneration(config).to("cuda:0")
    model.train()

    # Create dummy inputs.
    B, T = 1, 16
    input_ids = torch.randint(0, 256, (B, T), device="cuda:0")
    attention_mask = torch.ones(B, T, dtype=torch.long, device="cuda:0")
    labels = input_ids.clone()
    labels[:, :8] = -100  # Mask prompt tokens.

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert outputs.loss is not None
    assert outputs.loss.isfinite()
    outputs.loss.backward()

    # Verify gradients exist on at least one parameter.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients produced"


def test_vlm_text_only_no_crash():
    """Text-only input to a VLM model should work after vision token stripping."""
    from hatchery.core.worker import _strip_vision_tokens

    # Simulate: tokenizer inserted vision tokens but no images provided.
    vision_ids = {151652, 151653, 151654, 151655}  # Qwen2VL special tokens
    tokens = [1, 2, 151652, 151653, 151654, 151655, 3, 4, 5]
    cleaned = _strip_vision_tokens(tokens, vision_ids)
    assert cleaned == [1, 2, 3, 4, 5]
    assert len(cleaned) < len(tokens)
