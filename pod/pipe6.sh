#!/usr/bin/env bash
# v6 pipeline: unique-field-name data (copy-forcing) + T5-construct grammar.
# Split into stages so datagen (CPU) can run while the GPU is busy with other jobs:
#   pipe6.sh gen    — generate 2M examples to /dev/shm/v6 (CPU-only, ~25 min)
#   pipe6.sh train  — tokenizer + 40m training (GPU, ~4-5h)
set -euo pipefail
cd /root/nl2jq
export PYTHONUNBUFFERED=1
DATA=/dev/shm/v6
STAGE="${1:-gen}"

if [ "$STAGE" = "gen" ]; then
  echo "==================== V6 GENERATE $(date) ===================="
  rm -rf "$DATA"
  python3 -m pipeline.build_parallel --n 2000000 --workers 128 --seed 6000 --out "$DATA"
  echo "V6_GEN_DONE $(wc -l < $DATA/train.jsonl) train lines"
fi

if [ "$STAGE" = "train" ]; then
  echo "==================== V6 TOKENIZER $(date) ===================="
  python3 -m train.tokenizer --data "$DATA" --vocab 12288 --out artifacts/tok6
  echo "V6_TOK_DONE"

  echo "==================== V6 TRAIN $(date) ===================="
  python3 -m train.train --config 40m --data "$DATA" --tok artifacts/tok6 \
    --steps 50000 --batch 64 --lr 1.5e-3 --warmup 500 \
    --eval_every 2500 --eval_n 300 --keep_ckpts 1 \
    --out artifacts/nl2jq-40m-v6
  echo "V6_TRAIN_DONE $(date)"

  echo "==================== PERSIST -> /workspace $(date) ===================="
  mkdir -p /workspace/nl2jq/artifacts
  cp -r artifacts/tok6 /workspace/nl2jq/artifacts/ 2>/dev/null || true
  cp -r artifacts/nl2jq-40m-v6 /workspace/nl2jq/artifacts/ 2>/dev/null || true
  head -n 5000 "$DATA/train.jsonl" > /workspace/nl2jq/artifacts/v6_train_sample.jsonl 2>/dev/null || true
  cp "$DATA/manifest.json" /workspace/nl2jq/artifacts/v6_manifest.json 2>/dev/null || true
  echo "V6_PERSIST_DONE $(date)"
fi
