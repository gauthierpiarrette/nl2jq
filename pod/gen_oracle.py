"""Oracle-headroom sampler (pod GPU): k candidates per item -> {id, programs} jsonl.

Measures whether reranking/RL has room: oracle@k >> pass@1 means the right program is
already in the sample set and better selection pays; oracle@k ~= pass@1 means the model
lacks the capability and selection can't create it. Scoring happens locally under the
pinned jq (bench/score_oracle.py).
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
          "jq program. Output only the jq program, nothing else.")


def raw_prefix(doc, limit=800):
    s = json.dumps(doc, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=128)
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
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt")
            ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to("cuda")
            progs = []
            for j in range(a.k):
                kw = ({"do_sample": False} if j == 0
                      else {"do_sample": True, "temperature": a.temperature})
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=a.max_new,
                                         pad_token_id=tok.eos_token_id, **kw)
                t = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                progs.append(re.sub(r"(?s)^\s*<think>.*?</think>\s*", "", t).strip())
            f.write(json.dumps({"id": it["id"], "programs": progs}) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(items)}", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
