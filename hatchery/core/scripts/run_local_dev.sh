#!/usr/bin/env bash
# One-command local dev environment: gateway + worker in a single
# Python process using in-memory (or local filesystem) backends.
# No Postgres, Redis, R2, or WorkOS needed.
#
# Usage:
#   ./hatchery/core/scripts/run_local_dev.sh
#   GPU=1 ./hatchery/core/scripts/run_local_dev.sh
#   GPU=1 BASE_MODEL=Qwen/Qwen3-4B ./hatchery/core/scripts/run_local_dev.sh
#   PORT=9000 API_KEY=dev ./hatchery/core/scripts/run_local_dev.sh
#
# When the server comes up it prints the base URL and a ready-to-use
# bearer token. Example client invocation:
#
#   curl http://localhost:8420/v1/health
#   curl -H "Authorization: Bearer dev" \
#        -H "Content-Type: application/json" \
#        -d '{"base_model":"scripted","rank":8}' \
#        http://localhost:8420/v1/sessions
set -euo pipefail
cd "$(dirname "$0")/../../.."

PORT="${PORT:-8420}"
API_KEY="${API_KEY:-dev}"
GPU="${GPU:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-0.5B}"
DEVICE="${DEVICE:-cuda:0}"

VENV_PY="${VENV_PY:-python}"

export HATCHERY_DEV_API_KEY="${API_KEY}"
export HATCHERY_DEV_PORT="${PORT}"
export HATCHERY_DEV_BASE_MODEL="${BASE_MODEL}"
export HATCHERY_DEV_DEVICE="${DEVICE}"
if [ "${GPU}" != "1" ]; then
  export HATCHERY_DEV_NO_GPU=1
fi

exec "${VENV_PY}" -m hatchery.core.local_dev
