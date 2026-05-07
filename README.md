# Hatchery Core

Hatchery Core is an open-source, Tinker-compatible runtime for fine-tuning and post-training language models on infrastructure you control. It provides a local or self-hosted gateway, GPU worker, LoRA trainer, checkpoint store, and Python client surface for supervised fine-tuning, RL-style objectives, custom losses, and sampling.

Use Hatchery when you want to:

- Run Tinker-style training recipes locally or inside your own deployment.
- Fine-tune Hugging Face models with LoRA without adopting a hosted control plane.
- Experiment with model families and continued-fine-tuning workflows beyond the hosted base-model catalog.
- Keep sensitive datasets and domain-specific training workflows inside infrastructure you control.
- Prototype SFT, GRPO, PPO-style, and custom-loss workflows against a real API.
- Keep the training runtime extensible: storage, queues, auth, and workers are pluggable.

Hatchery is not a managed training service by itself. The open-core package provides the runtime and extension points; production fleet orchestration, hosted auth, billing, and managed infrastructure can be layered around it.

## What You Get

- **Tinker-compatible API**: use the official `tinker.ServiceClient` or Hatchery's built-in client.
- **Local-first development**: start a gateway and worker with one command.
- **Real GPU training**: train LoRA adapters against Hugging Face causal language models.
- **Post-training primitives**: SFT, GRPO, PPO, CISPO, DAPO, GSPO, custom forward/backward losses, sampling, and checkpoint save/load.
- **Base-model friendly recipes**: train pretrained base checkpoints while borrowing tokenizer/chat templates from matching instruction models when useful.
- **Pluggable backends**: in-memory and local backends ship in core; external packages can provide shared queues, object stores, auth, and deployment integrations.

## Requirements

- Python 3.12+
- Linux or macOS for local development
- `uv` recommended, `pip` also works
- CUDA-capable GPU for real model training
- PyTorch-compatible GPU drivers when using CUDA
- Hugging Face access for any model or dataset you load

The scripted worker can run without a GPU and is useful for API smoke tests. Real fine-tuning requires a GPU and a model that fits your device.

## Install From Source

```bash
git clone https://github.com/axolotl-ai-cloud/hatchery-core.git
cd hatchery-core

uv venv
source .venv/bin/activate

# Runtime plus test/dev tools.
uv pip install -e '.[test,dev]'

# Add cookbook dataset dependencies when running examples.
uv pip install -e '.[examples]'
```

For GPU machines, PyTorch wheel selection still comes from your package manager or index configuration. The `gpu` extra is intentionally a compatibility hook:

```bash
uv pip install -e '.[gpu,test,examples]'
```

### Optional DFlash Support

Hatchery can use [DFlash](https://github.com/z-lab/dflash) for speculative decoding on supported model/draft-adapter pairs. DFlash is not declared as a `hatchery-core` package extra because public PyPI rejects packages whose published metadata contains direct Git dependencies. Install it explicitly in worker images or local environments that enable DFlash:

```bash
uv pip install -e '.[gpu,test,examples]'
uv pip install 'dflash[transformers] @ git+https://github.com/z-lab/dflash.git@4febcb4b32824a39fc683c9b74d193f885d9fe19'
```

Without DFlash installed, non-strict speculative-decoding requests fall back to the normal Hugging Face generation path. Strict requests raise so deployment smoke tests can catch a missing dependency.

## Quick Start

Start the local dev server:

```bash
python -m hatchery.core.local_dev
```

By default this starts a gateway at `http://127.0.0.1:8420` with bearer token `dev`. If a CUDA device is available, the launcher uses a GPU worker; otherwise it falls back to the scripted worker.

Use a specific pretrained base model:

```bash
HATCHERY_DEV_DEVICE=cuda:0 \
HATCHERY_DEV_BASE_MODEL=Qwen/Qwen2-0.5B \
python -m hatchery.core.local_dev
```

Check the server:

```bash
curl http://127.0.0.1:8420/v1/health
```

Expected response:

```json
{"status":"ok"}
```

## First Training Run

In another terminal, run the Pig Latin SFT/GRPO cookbook:

```bash
python -m hatchery.core.examples.train_sft \
  --base-url http://127.0.0.1:8420 \
  --token dev \
  --base-model Qwen/Qwen2-0.5B \
  --steps 100 \
  --rl-steps 0 \
  --batch-size 2
```

This trains the pretrained `Qwen/Qwen2-0.5B` base model. By default the example borrows the chat template from `Qwen/Qwen2-0.5B-Instruct` for formatting, but the trained model remains the base checkpoint.

For a simple style-transfer smoke test:

```bash
python -m hatchery.core.examples.train_pirate_style \
  --base-url http://127.0.0.1:8420 \
  --token dev \
  --base-model Qwen/Qwen2-0.5B \
  --steps 100 \
  --batch-size 2
```

The examples print loss, save a checkpoint, and sample from the trained adapter.

## Python Client

Hatchery follows the Tinker convention: clients pre-shift causal language-model training data. If `tokens` is the full token sequence, send `tokens[:-1]` as the model input and `tokens[1:]` as `target_tokens`.

```python
from hatchery.core.client import HatcheryClient

client = HatcheryClient(base_url="http://127.0.0.1:8420", token="dev")
training = client.create_lora_training_client("Qwen/Qwen2-0.5B", rank=8)

tokens = [10, 20, 30, 40, 50]
datum = {
    "model_input": {
        "chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}],
    },
    "loss_fn_inputs": {
        "target_tokens": {
            "dtype": "int64",
            "data": tokens[1:],
            "shape": [len(tokens) - 1],
        },
        "weights": {
            "dtype": "float32",
            "data": [1.0] * (len(tokens) - 1),
            "shape": [len(tokens) - 1],
        },
    },
}

fb = training.forward_backward([datum]).result(timeout=120)
training.optim_step(learning_rate=1e-4).result(timeout=60)

print(fb)
print(training.sample(tokens[:-1], max_tokens=32).result(timeout=30))

client.close()
```

For prompt/completion training, set weights to `0.0` on prompt positions and `1.0` on completion positions after the shift. The examples in `hatchery/core/examples` show this pattern with chat templates and prompt masking.

You can also use the official Tinker SDK:

```python
import tinker

client = tinker.ServiceClient(api_key="dev", base_url="http://127.0.0.1:8420")
training = client.create_lora_training_client(base_model="Qwen/Qwen2-0.5B")
```

## Architecture

```
Client (HatcheryClient or tinker SDK)
  |
  v
Gateway (FastAPI, /api/v1 and /v1)
  |
  v
Job queue (in-memory, SQLite, or extension backend)
  |
  v
GPU worker (model, LoRA adapter, optimizer)
  |
  v
Object store (local, in-memory, or extension backend)
```

The gateway is stateless. Workers own model loading, adapter state, training operations, sampling, and checkpoint materialization. Multiple workers can share queue and storage backends supplied by extensions.

## Configuration

Common local-dev variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HATCHERY_DEV_PORT` | `8420` | HTTP port |
| `HATCHERY_DEV_API_KEY` | `dev` | Bearer token |
| `HATCHERY_DEV_BASE_MODEL` | `Qwen/Qwen2-0.5B` | Hugging Face model id |
| `HATCHERY_DEV_DEVICE` | auto-detected | Torch device, for example `cuda:0` |
| `HATCHERY_DEV_NO_GPU` | `0` | Set `1` to force the scripted worker |

Core runtime variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HATCHERY_ADMIN_API_KEY` | generated | Admin API key for the core gateway |
| `HATCHERY_OBJECT_STORE` | `local` | `local` or `memory` |
| `HATCHERY_LOCAL_STORE_PATH` | `/tmp/hatchery_data` | Local object-store root |
| `HATCHERY_BASE_MODEL` | `Qwen/Qwen2-0.5B` | Standalone worker model |
| `HATCHERY_WORKER_DEVICE` | `cuda:0` | Standalone worker device |
| `HATCHERY_CONFIG_FACTORY` | unset | `module:callable` config factory |

## Documentation

- [Quickstart](content/docs/getting-started/quickstart.mdx): local gateway, worker, and client workflow.
- [Self-hosting](content/docs/getting-started/self-hosting.mdx): running core outside the local dev launcher.
- [Architecture](content/docs/learn/architecture.mdx): control plane, worker, queue, and storage flow.
- [API endpoints](content/docs/reference/endpoints.mdx): native and Tinker-compatible routes.
- [Testing](content/docs/development/testing.mdx): CPU, GPU, lint, and packaging checks.
- [Extending Hatchery Core](content/docs/development/extending.mdx): backend and extension contracts.

For enterprise or self-hosted deployment questions, or if you need help adapting Hatchery to a sensitive workflow, contact `contact@axolotl.ai`.

## Development

```bash
ruff check hatchery/ tests/
ruff format --check hatchery/ tests/
python -m pytest tests/ -q
```

GPU tests live under `tests/torch_tests` and `tests/torch_distributed`.
They require CUDA plus compatible model/cache access:

```bash
python -m pytest tests/torch_tests/ tests/torch_distributed/ -q
```

## Project Status

Hatchery Core is alpha software. The public API is intended to be useful for experimentation and extension work, but interfaces may change while the project moves toward a stable release.

## Known Limitations

- The default local launcher is designed for development and smoke testing, not production scheduling.
- Multi-worker deployments require shared queue, metadata, and object-store backends supplied through configuration or extension packages.
- GPU compatibility depends on the installed PyTorch build, CUDA drivers, model architecture, and available VRAM.
- The included cookbooks are validation and experimentation launchpads, not fully tuned training recipes for every model family.
- Multi-modal model paths exist in core, but release validation is still focused primarily on text-only workflows.
- Public APIs, config contracts, and packaging metadata may still change before a stable release.

## Roadmap

- Incorporate optimized Axolotl training patches and kernels where they improve throughput, memory use, or model-family coverage.
- Expand multi-modal validation for VLM training, sampling, and checkpoint flows.
- Harden distributed and multi-worker self-hosting patterns with shared queue and object-store backends.
- Add more end-to-end cookbooks for common SFT, RL, style-transfer, and custom-loss workflows.
- Improve observability around loss, token throughput, queue latency, GPU memory, and checkpoint operations.
- Stabilize the public API and configuration surface for a non-alpha release.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [RELEASE.md](RELEASE.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
