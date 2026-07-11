#!/usr/bin/env bash
# After v7 40m training: ckpt selection -> oracle@16 -> Qwen3.5-2B LoRA.
set -uo pipefail
cd /root/nl2jq
echo "chain8: waiting for v7 train..."
until grep -q "V7_PERSIST_DONE" chain7.log 2>/dev/null; do sleep 60; done
echo "chain8: v7 ckpt selection $(date)"
rm -rf bench/__pycache__
python3 -m bench.eval_ckpts --dir artifacts/nl2jq-40m-v7 --tok artifacts/tok7 \
  --items devnovel-v1.jsonl > ckpt7.log 2>&1
echo "chain8: oracle@16 gen (qwen-v6) $(date)"
python3 gen_oracle.py --model artifacts/nl2jq-qwen3-0.6b-v6 --items devnovel-v1.jsonl \
  --out oracle_qwenv6_k16.jsonl --k 16 > oracle.log 2>&1
echo "chain8: waiting for qwen3.5-2b weights..."
until [ -f /root/qwen35-2b/.rsync_done ]; do sleep 60; done
echo "chain8: 2B LoRA $(date)"
bash pod/qwen35_v7.sh
echo "CHAIN8_ALL_DONE $(date)"
