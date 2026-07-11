#!/usr/bin/env bash
# Qwen3-0.6B v6 LoRA — GENTLE recipe (v5's r=64/alpha=128 @ lr 2e-4 was aggressive;
# the base has little to preserve at 0.6B, but lower lr + rank reduces distribution
# lock-in on the synthetic data). Waits for the GPU if checkpoint eval is still running.
set -euo pipefail
cd /root/nl2jq
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "bench.eval_ckpts" >/dev/null 2>&1; do sleep 30; done
echo "==================== QWEN v6 FT $(date) ===================="
python3 -m train.finetune_qwen --data /dev/shm/v6 --model /root/qwen3-06b \
  --out artifacts/qwen06b-v6 --limit 150000 --lora --lora_r 32 --lora_alpha 64 \
  --epochs 1 --bs 16 --grad_accum 1 --lr 1e-4 --max_len 768
echo "QWEN_V6_DONE $(date)"
python3 -m train.merge_qwen --base /root/qwen3-06b \
  --adapter artifacts/qwen06b-v6 --out artifacts/nl2jq-qwen3-0.6b-v6
echo "QWEN_V6_MERGED $(date)"
cp -r artifacts/qwen06b-v6 /workspace/nl2jq/artifacts/ 2>/dev/null || true
echo "QWEN_V6_PERSIST_DONE $(date)"
