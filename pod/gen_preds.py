"""Standalone GPU prediction generator for nl2jq-bench items (runs on the pod).

Loads any HF causal-LM with a chat template, generates one jq program per item, writes
{id, program} jsonl. Scoring happens elsewhere (locally, under the pinned jq 1.7.1) via
`python -m bench.eval_frozen --backend preds`.

    python3 gen_preds.py --model /root/qwen3-06b --items items.jsonl --out preds.jsonl \
        --max-new 1024
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
          "jq program (jq 1.7). Output ONLY the jq program on one line — no explanation, "
          "no markdown fences, no `jq` prefix.")


def raw_prefix(doc, limit=800):
    s = json.dumps(doc, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--no-thinking", action="store_true",
                    help="pass enable_thinking=False to the chat template")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    items = [json.loads(l) for l in open(a.items)]

    with open(a.out, "w") as f:
        for i, it in enumerate(items, 1):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Request: {it['request']}\n"
                                                f"JSON sample: {raw_prefix(it['input'])}"}]
            kw = {"add_generation_prompt": True, "return_tensors": "pt"}
            if a.no_thinking:
                kw["enable_thinking"] = False
            enc = tok.apply_chat_template(msgs, **kw)
            ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to("cuda")
            # stop at end-of-turn: without an explicit eos, some templates keep generating
            # fabricated user/assistant turns (observed with Qwen3.5)
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            eos = [t for t in {im_end, tok.eos_token_id} if isinstance(t, int) and t >= 0]
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                                     eos_token_id=eos, pad_token_id=eos[0])
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            text = re.sub(r"(?s)^\s*<think>.*?</think>\s*", "", text).strip()
            # belt-and-braces: cut anything after a fabricated turn marker
            text = re.split(r"\n(?:user|assistant)\b", text)[0].strip()
            f.write(json.dumps({"id": it["id"], "program": text}) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(items)}", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
