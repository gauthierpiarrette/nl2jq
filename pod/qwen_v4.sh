#!/usr/bin/env bash
# Fine-tune the Qwen3-0.6B baseline on v4 data (LoRA). Run AFTER the 40m train frees the GPU.
set -euo pipefail
cd /workspace/nl2jq
export PYTHONUNBUFFERED=1
echo "==================== QWEN v4 FT $(date) ===================="
python3 -m train.finetune_qwen --data data/v4 --out artifacts/qwen06b-v4 \
  --limit 150000 --lora --epochs 2 --bs 16 --lr 2e-4 --max_len 768
echo "QWEN_V4_DONE $(date)"
