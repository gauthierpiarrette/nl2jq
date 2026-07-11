#!/usr/bin/env bash
set -euo pipefail
cd /root/nl2jq
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "==================== MERGE $(date) ===================="
python3 -m train.merge_qwen --base /root/qwen3-06b \
  --adapter artifacts/qwen06b-v5 --out artifacts/nl2jq-qwen3-0.6b
echo "==================== BENCH EVAL ===================="
python3 -m bench.eval_qwen --model artifacts/nl2jq-qwen3-0.6b
echo "==================== COPY-SKILL ===================="
python3 -m bench.cli_smoke --backend qwen --model artifacts/nl2jq-qwen3-0.6b
echo "==================== EVAL_DONE $(date) ===================="
