#!/usr/bin/env bash
set -uo pipefail
cd /root/nl2jq
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "==================== BENCH EVAL $(date) ===================="
python3 -m bench.eval_qwen --model artifacts/nl2jq-qwen3-0.6b
echo "==================== COPY-SKILL ===================="
python3 -m bench.cli_smoke --backend qwen --model artifacts/nl2jq-qwen3-0.6b
echo "==================== EVAL2_DONE $(date) ===================="
