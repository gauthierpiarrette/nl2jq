#!/usr/bin/env bash
# Wait for the GPU to free (base-qwen prediction gen), then run v6 tokenizer+train.
set -uo pipefail
cd /root/nl2jq
echo "chain6: waiting for gen_preds to finish and v6 data..."
while pgrep -f "gen_preds.py" >/dev/null 2>&1; do sleep 30; done
until grep -q "V6_GEN_DONE" pipe6gen.log 2>/dev/null; do sleep 30; done
echo "chain6: GPU free + data ready, starting train stage $(date)"
bash pod/pipe6.sh train
