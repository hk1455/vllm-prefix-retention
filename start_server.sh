#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
POLICY="${1:-lru}"
BLOCKS="${2:-2048}"
case "$POLICY" in lru|slru) ;; *) echo 'Usage: bash start_server.sh lru|slru BLOCKS' >&2; exit 2;; esac
[[ "$BLOCKS" =~ ^[1-9][0-9]*$ ]] || { echo 'BLOCKS must be positive' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/models/Qwen3-4B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
export VLLM_USE_V1=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_PREFIX_RETENTION_POLICY="$POLICY"
export VLLM_PREFIX_RETENTION_PROTECTED_RATIO="${VLLM_PREFIX_RETENTION_PROTECTED_RATIO:-0.5}"
"$PYTHON_BIN" manage_patch.py check
# Require the patch even for the LRU comparison so a policy typo cannot silently
# run an unpatched server. Import in the same interpreter as the server.
"$PYTHON_BIN" -c 'from vllm.v1.core.prefix_retention import SegmentedFreeBlockQueue'
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="logs/${POLICY}_b${BLOCKS}_${STAMP}.log"
CMD=( "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
  --model "$MODEL" --served-model-name qwen3-4b
  --host "$HOST" --port "$PORT" --dtype bfloat16
  --max-model-len 16384 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.60}"
  --max-num-seqs 8 --max-num-batched-tokens 16384
  --block-size 16 --num-gpu-blocks-override "$BLOCKS"
  --enable-prefix-caching --enable-prompt-tokens-details --enforce-eager
  --disable-log-requests )
{
  echo "policy=$POLICY blocks=$BLOCKS protected_ratio=$VLLM_PREFIX_RETENTION_PROTECTED_RATIO"
  printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'
} | tee "${LOG%.log}.launch.txt"
"${CMD[@]}" 2>&1 | tee "$LOG"
