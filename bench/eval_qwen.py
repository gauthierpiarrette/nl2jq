"""Score the Qwen3-0.6B fine-tune on nl2jq-bench (execution accuracy), locally.

Mirrors bench.baselines but generates with the local fine-tune instead of a frontier API.
Loads either a merged model or a base+adapter pair.

    python -m bench.eval_qwen --model artifacts/nl2jq-qwen3-0.6b
    python -m bench.eval_qwen --model artifacts/qwen06b-v5 --base /root/qwen3-06b
"""
import argparse
import json
import re
from pathlib import Path

from bench.harness import score_items
from cli.jqgen import build_context

BENCH = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"

SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
          "jq program. Output only the jq program, nothing else.")


def _clean(text: str) -> str:
    text = re.sub(r"(?s)^\s*<think>.*?</think>\s*", "", text)  # Qwen3 reasoning block
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    if text.startswith("jq "):
        text = text[3:].strip()
    return text.strip().strip("`").strip()


def load(model_dir, base, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    is_adapter = (Path(model_dir) / "adapter_config.json").exists()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if is_adapter:
        from peft import PeftModel
        assert base, "--base is required when --model is a LoRA adapter dir"
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
        m = PeftModel.from_pretrained(m, model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir)
    else:
        m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
        tok = AutoTokenizer.from_pretrained(model_dir)
    return m.to(device).eval(), tok


def make_gen(model, tok, device, k, temperature):
    import torch

    def gen(item):
        context = build_context(item["input"], "auto")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Request: {item['request']}\nJSON sample: {context}"}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        # newer transformers return a BatchEncoding here, not a bare tensor
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
        outs = []
        for j in range(k):
            do_sample = temperature > 0 and j > 0
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=128, do_sample=do_sample,
                                     temperature=temperature if do_sample else None,
                                     pad_token_id=tok.eos_token_id)
            outs.append(_clean(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
        return outs

    return gen


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir or LoRA adapter dir")
    ap.add_argument("--base", default=None, help="base model (required for an adapter dir)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(a.model, a.base, device)
    items = [json.loads(l) for l in BENCH.open()]
    gen = make_gen(model, tok, device, a.k, a.temperature)
    res = score_items(items, gen, k=a.k)
    summary = {kk: round(v, 3) for kk, v in res.items() if kk != "details"}
    print(f"nl2jq-qwen3-0.6b on nl2jq-bench (k={a.k}, T={a.temperature})")
    print(json.dumps(summary, indent=2))
    wrong = [d["id"] for d in res["details"] if not (d["results"] and d["results"][0]["correct"])]
    print(f"missed ({len(wrong)}):", wrong)


if __name__ == "__main__":
    main()
