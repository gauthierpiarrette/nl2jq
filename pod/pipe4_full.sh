#!/usr/bin/env bash
# v4 full pipeline on the pod: regenerate broadened data -> tokenizer -> train 40m.
# Data is the only changed variable vs v3 (same arch, vocab 12288, batch 64).
set -euo pipefail
cd /workspace/nl2jq
export PYTHONUNBUFFERED=1

echo "==================== V4 GENERATE $(date) ===================="
python3 -m pipeline.build_parallel --n 2000000 --workers 96 --seed 4000 --out data/v4
echo "V4_GEN_DONE"

echo "==================== V4 TOKENIZER $(date) ===================="
python3 -m train.tokenizer --data data/v4 --vocab 12288 --out artifacts/tok4
echo "V4_TOK_DONE"

echo "==================== V4 TRAIN $(date) ===================="
python3 -m train.train --config 40m --data data/v4 --tok artifacts/tok4 \
  --steps 50000 --batch 64 --lr 1.5e-3 --warmup 500 \
  --eval_every 2500 --eval_n 300 --keep_ckpts 1 \
  --out artifacts/nl2jq-40m-v4
echo "V4_TRAIN_DONE $(date)"
