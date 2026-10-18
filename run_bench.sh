#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
POLICY="${1:?Usage: bash run_bench.sh lru|slru BLOCKS [SEED]}"
BLOCKS="${2:?Missing block count}"
SEED="${3:-42}"
case "$POLICY" in lru|slru) ;; *) echo 'Policy must be lru or slru' >&2; exit 2;; esac
BENCH="${BENCH:-$PWD/zipf_prefix_bench.py}"
[[ -f "$BENCH" ]] || { echo "Benchmark not found: $BENCH; set BENCH to your working script" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
OUT="results/${POLICY}_b${BLOCKS}_s${SEED}_${RUN_ID}"
echo "Label only: verify the SERVER log says policy=$POLICY and blocks=$BLOCKS."
"$PYTHON_BIN" "$BENCH" \
  --base-url "${BASE_URL:-http://127.0.0.1:8000}" --model qwen3-4b \
  --prefixes 100 --prefix-tokens 3072 --alpha 1.0 --seed "$SEED" \
  --warmup "${WARMUP:-20}" --requests "${REQUESTS:-100}" \
  --concurrency "${CONCURRENCY:-1}" --gpu-blocks "$BLOCKS" --out "$OUT"
echo "Results: $OUT"
