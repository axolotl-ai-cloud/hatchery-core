# Hatchery Core

Open-source fine-tuning platform compatible with the [Tinker SDK](https://pypi.org/project/tinker/). Train LoRA adapters and run RLHF on any Hugging Face model — from a single GPU to a multi-worker fleet.

## What it does

Hatchery runs a FastAPI gateway and GPU worker that together expose the Tinker training API. You send tokenized data, Hatchery trains the model and returns loss metrics. The official Tinker SDK, or Hatchery's built-in `HatcheryClient`, both work as clients.

**Supported training:**
- **SFT** — supervised fine-tuning with cross-entropy loss and prompt masking
- **RLHF** — GRPO, PPO, CISPO, DAPO, GSPO with server-side loss computation
- **Custom losses** — two-step `forward_backward_custom` for client-defined objectives
- **LoRA and full-parameter** — rank 1–256 LoRA, or full-param fine-tuning for smaller models

**Key features:**
- Tinker SDK compatible — drop-in replacement for `tinker.ServiceClient`
- Pipelined async API — submit batches without waiting, drain results later
- Chat-template aware — uses the model's native chat format for training data
- Gradient checkpointing enabled by default
- Pluggable backends — swap storage, queues, and auth without changing training code

## Quick start

```bash
# Install
uv pip install -e '.[gpu,test]'

# Start the local dev server (auto-detects GPU)
python -m hatchery.core.local_dev

# Or with a specific model:
HATCHERY_DEV_BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct python -m hatchery.core.local_dev
```

In another terminal:

```python
from hatchery.core.client import HatcheryClient

client = HatcheryClient(base_url="http://127.0.0.1:8420", token="dev")
tc = client.create_lora_training_client("Qwen/Qwen2-0.5B-Instruct", rank=32)

# Train
datum = {
    "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_ids}]},
    "loss_fn_inputs": {"target_tokens": {"data": labels, "shape": [len(labels)]}},
}
tc.forward_backward([datum]).result()
tc.optim_step(learning_rate=1e-4).result()

# Sample
result = tc.sample(prompt_ids, max_tokens=64).result()
print(result["sequences"])
```

Or use the official Tinker SDK:

```python
import tinker
client = tinker.ServiceClient(api_key="dev", base_url="http://127.0.0.1:8420")
tc = client.create_lora_training_client(base_model="Qwen/Qwen2-0.5B-Instruct")
```

## Full example: SFT + GRPO on pig-latin

The included example trains a model to translate English to pig-latin using WikiText-2 sentences, then refines it with GRPO:

```bash
python -m hatchery.core.local_dev
python hatchery/core/examples/train_sft.py --steps 100 --rl-steps 25
```

Phase 1 trains with SFT (cross-entropy, chat template, prompt masking). Phase 2 samples 16 completions per prompt, scores them with a word-level reward, and runs GRPO optimization. Loss and reward are logged to stdout (or W&B with `--wandb`).

## Data format

Hatchery follows the Tinker convention — the client pre-shifts inputs:

```python
tokens = tokenizer.encode(text)
datum = {
    "model_input": {
        "chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]
    },
    "loss_fn_inputs": {
        "target_tokens": {"data": tokens[1:], "shape": [len(tokens) - 1]},
        "weights": {"data": [0.0] * prompt_len + [1.0] * completion_len,
                    "shape": [len(tokens) - 1]},
    },
}
```

At position `i`, `tokens[i]` produces logits and `target_tokens[i]` is what should be predicted. Weights of `0.0` mask the prompt; `1.0` marks the completion. The `-100` ignore index also works for masking.

## Architecture

```
Client (HatcheryClient / tinker SDK)
  │
  ▼
Gateway (FastAPI, /api/v1/*)
  │
  ▼
Job Queue (in-memory / SQLite)
  │
  ▼
GPU Worker (model + LoRA + optimizer)
  │
  ▼
Object Store (local disk)
```

The gateway is stateless. The worker holds the model in VRAM and processes jobs sequentially. Multiple workers can share a queue for horizontal scaling.

All backends are pluggable via protocols — swap in Redis, Postgres, S3, or any custom implementation without changing training code.

## Included backends

| Component | Options |
|-----------|---------|
| Queue | In-memory, SQLite |
| Metadata | In-memory, SQLite |
| Object Store | Local disk, in-memory |
| Auth | API key |
| Metrics | Structured logging |

## Running tests

```bash
# All tests (CPU)
python -m pytest tests/ -q

# GPU tests
python -m pytest tests/torch_tests/ -q

# Lint
ruff check hatchery/ tests/
```

## Configuration

The local dev server works with zero configuration. For production, set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HATCHERY_DEV_PORT` | `8420` | HTTP port |
| `HATCHERY_DEV_API_KEY` | `dev` | Bearer token |
| `HATCHERY_DEV_BASE_MODEL` | `Qwen/Qwen2-0.5B-Instruct` | Model to load |
| `HATCHERY_METADATA_STORE` | `memory` | `memory` or `sqlite` |
| `HATCHERY_JOB_QUEUE` | `memory` | `memory` or `sqlite` |
| `HATCHERY_OBJECT_STORE` | `local` | `local` or `memory` |

## License

Apache 2.0 — see [LICENSE](LICENSE).
