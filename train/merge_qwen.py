"""Merge a LoRA adapter into its Qwen base to produce a standalone HF model.

A merged model loads with a plain `AutoModelForCausalLM.from_pretrained` — no peft at
inference — which is what the CLI, the bench eval, and the Hub upload all want.

    python -m train.merge_qwen --base /root/qwen3-06b \
        --adapter artifacts/qwen06b-v5 --out artifacts/nl2jq-qwen3-0.6b
"""
import argparse
from pathlib import Path


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model dir (Qwen3-0.6B)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir from finetune_qwen")
    ap.add_argument("--out", required=True, help="output dir for the merged model")
    a = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    # ship the tokenizer from the adapter dir (finetune saved it there) so the merged
    # model is fully self-contained
    tok = AutoTokenizer.from_pretrained(a.adapter)
    tok.save_pretrained(str(out))
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
