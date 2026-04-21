#!/usr/bin/env bash
# One-command local dev environment: gateway + worker in a single
# Python process using in-memory (or local filesystem) backends.
# No Postgres, Redis, R2, or WorkOS needed.
#
# Usage:
#   ./scripts/run_local_dev.sh                         # in-memory, no GPU
#   GPU=1 ./scripts/run_local_dev.sh                    # local GPU, Qwen2-0.5B
#   GPU=1 BASE_MODEL=Qwen/Qwen3-4B ./scripts/run_local_dev.sh
#   PORT=9000 API_KEY=dev-key ./scripts/run_local_dev.sh
#
# When the server comes up it prints the base URL and a ready-to-use
# bearer token. Example client invocation:
#
#   curl http://localhost:8420/v1/health
#   curl -H "Authorization: Bearer dev-key" \
#        -H "Content-Type: application/json" \
#        -d '{"base_model":"scripted","rank":8}' \
#        http://localhost:8420/v1/sessions
set -euo pipefail
cd "$(dirname "$0")/../../.."

PORT="${PORT:-8420}"
API_KEY="${API_KEY:-dev-key}"
GPU="${GPU:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-0.5B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"

VENV_PY="${VENV_PY:-python}"

export TINKER_DEV_API_KEY="${API_KEY}"
export TINKER_DEV_PORT="${PORT}"
export TINKER_DEV_BASE_MODEL="${BASE_MODEL}"
export TINKER_DEV_DEVICE="${DEVICE}"
export TINKER_DEV_USE_GPU="${GPU}"

exec "${VENV_PY}" -m hatchery.core.local_dev
