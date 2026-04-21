#!/usr/bin/env bash
# Launch a GPU worker on a single node with optional multi-GPU parallelism.
#
# Usage:
#   ./scripts/run_worker.sh                       # single GPU, cuda:0
#   NPROC=2 ./scripts/run_worker.sh               # 2-GPU FSDP2
#   NPROC=8 TP_DEGREE=2 CP_DEGREE=2 ./scripts/run_worker.sh  # 8-GPU 2D
#
# Environment variables you'll typically want to set:
#   HATCHERY_BASE_MODEL=Qwen/Qwen2-0.5B-Instruct
#   NPROC=2
#   HATCHERY_DP_DEGREE=2
#   HATCHERY_TP_DEGREE=1
#   HATCHERY_CP_DEGREE=1
#   HATCHERY_CPU_OFFLOAD_PARAMS=1
#   HATCHERY_ACTIVATION_CKPT=1
#   HATCHERY_OBJECT_STORE, HATCHERY_METADATA_STORE, HATCHERY_JOB_QUEUE, and the corresponding creds.
set -euo pipefail

NPROC="${NPROC:-1}"
HATCHERY_BASE_MODEL="${HATCHERY_BASE_MODEL:-Qwen/Qwen2-0.5B-Instruct}"

# Export the parallelism knobs that ParallelConfig.from_env() reads.
export HATCHERY_DP_DEGREE="${HATCHERY_DP_DEGREE:-${NPROC}}"
export HATCHERY_TP_DEGREE="${HATCHERY_TP_DEGREE:-1}"
export HATCHERY_CP_DEGREE="${HATCHERY_CP_DEGREE:-1}"
export HATCHERY_SP="${HATCHERY_SP:-0}"
export HATCHERY_CPU_OFFLOAD_PARAMS="${HATCHERY_CPU_OFFLOAD_PARAMS:-0}"
export HATCHERY_CPU_OFFLOAD_OPTIMIZER="${HATCHERY_CPU_OFFLOAD_OPTIMIZER:-0}"
export HATCHERY_ACTIVATION_CKPT="${HATCHERY_ACTIVATION_CKPT:-0}"

# Sanity: world size must equal DP × TP × CP.
expected=$((HATCHERY_DP_DEGREE * HATCHERY_TP_DEGREE * HATCHERY_CP_DEGREE))
if [ "${expected}" -ne "${NPROC}" ]; then
  echo "error: NPROC=${NPROC} != DP*TP*CP=${expected}" >&2
  exit 2
fi

export HATCHERY_BASE_MODEL

if [ "${NPROC}" -eq "1" ]; then
  # Single-GPU path — no torchrun needed.
  exec python -m hatchery.core.worker
else
  exec torchrun \
    --standalone \
    --nproc-per-node="${NPROC}" \
    -m hatchery.core.worker
fi
