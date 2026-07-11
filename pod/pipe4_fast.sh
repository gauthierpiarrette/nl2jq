#!/usr/bin/env bash
# v4 pipeline on FAST storage: /workspace is a slow network FS (MooseFS) that stalls the
# generator's many small writes. Run code from a local copy (/root/nl2jq), stage data in
# /dev/shm (RAM), keep artifacts local, then persist the trained model back to /workspace.
set -euo pipefail
cd /root/nl2jq
export PYTHONUNBUFFERED=1
DATA=/dev/shm/v4

echo "==================== V4 GENERATE $(date) ===================="
rm -rf "$DATA"
python3 -m pipeline.build_parallel --n 2000000 --workers 128 --seed 4000 --out "$DATA"
echo "V4_GEN_DONE $(wc -l < $DATA/train.jsonl) train lines"

echo "==================== V4 TOKENIZER $(date) ===================="
python3 -m train.tokenizer --data "$DATA" --vocab 12288 --out artifacts/tok4
echo "V4_TOK_DONE"

echo "==================== V4 TRAIN $(date) ===================="
python3 -m train.train --config 40m --data "$DATA" --tok artifacts/tok4 \
  --steps 50000 --batch 64 --lr 1.5e-3 --warmup 500 \
  --eval_every 2500 --eval_n 300 --keep_ckpts 1 \
  --out artifacts/nl2jq-40m-v4
echo "V4_TRAIN_DONE $(date)"

echo "==================== PERSIST -> /workspace $(date) ===================="
mkdir -p /workspace/nl2jq/artifacts
cp -r artifacts/tok4 /workspace/nl2jq/artifacts/ 2>/dev/null || true
cp -r artifacts/nl2jq-40m-v4 /workspace/nl2jq/artifacts/ 2>/dev/null || true
# keep a small data sample for the dataset card / provenance
head -n 5000 "$DATA/train.jsonl" > /workspace/nl2jq/artifacts/v4_train_sample.jsonl 2>/dev/null || true
cp "$DATA/manifest.json" /workspace/nl2jq/artifacts/v4_manifest.json 2>/dev/null || true
echo "V4_PERSIST_DONE $(date)"
