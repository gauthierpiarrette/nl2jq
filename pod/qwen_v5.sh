#!/usr/bin/env bash
set -euo pipefail
cd /root/nl2jq
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "==================== QWEN v5 FT $(date) ===================="
# Gradient checkpointing caps activation memory, so a bs=16 micro-batch fits (~14GB of
# 32GB): the ~152k-vocab loss-logits are the only real peak. 1 epoch over 150k
# execution-verified rows is enough for a LoRA baseline and avoids overfitting the
# synthetic distribution. best-effort ~1.5h.
python3 -m train.finetune_qwen --data /dev/shm/v5 --model /root/qwen3-06b \
  --out artifacts/qwen06b-v5 --limit 150000 --lora --epochs 1 \
  --bs 16 --grad_accum 1 --lr 2e-4 --max_len 768
echo "QWEN_V5_DONE $(date)"
cp -r artifacts/qwen06b-v5 /workspace/nl2jq/artifacts/ 2>/dev/null || true
echo "QWEN_V5_PERSIST_DONE $(date)"
