#!/usr/bin/env bash
# Qwen3.5-2B LoRA on v7 data — the "useful tier" bet. bs8/ga2 keeps the ~150k-vocab
# loss logits + 2B weights inside 31GB with gradient checkpointing.
set -euo pipefail
cd /root/nl2jq
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "==================== QWEN3.5-2B v7 FT $(date) ===================="
python3 -m train.finetune_qwen --data /dev/shm/v7 --model /root/qwen35-2b \
  --out artifacts/qwen35-2b-v7 --limit 150000 --lora --lora_r 32 --lora_alpha 64 \
  --epochs 1 --bs 8 --grad_accum 2 --lr 1e-4 --max_len 768
echo "QWEN35_V7_DONE $(date)"
python3 -m train.merge_qwen --base /root/qwen35-2b \
  --adapter artifacts/qwen35-2b-v7 --out artifacts/nl2jq-qwen3.5-2b-v7
echo "QWEN35_V7_MERGED $(date)"
cp -r artifacts/qwen35-2b-v7 /workspace/nl2jq/artifacts/ 2>/dev/null || true
echo "QWEN35_V7_PERSIST_DONE $(date)"
