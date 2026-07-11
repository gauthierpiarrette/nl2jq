"""Score a trained nl2jq model against nl2jq-bench.

    python -m bench.run --model artifacts/sanity --k 1
    python -m bench.run --model artifacts/sanity --k 5 --temperature 0.8
"""
import argparse
import json
from pathlib import Path

import torch

from bench.harness import score_items
from cli.jqgen import build_context
from pipeline.common import format_example
from train.model import ModelConfig, NL2JQModel
from train.tokenizer import load_tokenizer

BENCH = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"


def load(model_dir: Path, device, tok_dir=None):
    ckpt = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    model = NL2JQModel(ModelConfig(**ckpt["cfg"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = load_tokenizer(tok_dir or model_dir.parent / "tok")
    return model, tok


def make_generate_fn(model, tok, device, k, temperature, mode="auto"):
    eos = tok.token_to_id("<|end|>")

    def gen(item):
        context = build_context(item["input"], mode)
        prompt = format_example(item["request"], context, None)
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        progs = []
        n = 1 if temperature <= 0 else k
        for _ in range(n):
            out = model.generate(x, 128, eos, temperature=temperature)[0].tolist()[len(ids):]
            if eos in out:
                out = out[: out.index(eos)]
            progs.append(tok.decode(out).strip())
        return progs

    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/sanity")
    ap.add_argument("--tok", default=None, help="tokenizer dir (default: <model>/../tok)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--mode", default="auto")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    items = [json.loads(l) for l in BENCH.open()]
    model, tok = load(Path(a.model), device, Path(a.tok) if a.tok else None)
    gen = make_generate_fn(model, tok, device, a.k, a.temperature, a.mode)
    res = score_items(items, gen, k=a.k)
    summary = {kk: round(v, 3) for kk, v in res.items() if kk != "details"}
    print(json.dumps(summary, indent=2))
    wrong = [d["id"] for d in res["details"] if not (d["results"] and d["results"][0]["correct"])]
    print(f"missed ({len(wrong)}):", wrong)


if __name__ == "__main__":
    main()
