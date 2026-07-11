"""Print request / gold / prediction for every MISSED bench item, tagged by failure mode
(invalid = didn't parse/run; wrong = ran but mismatched), so we can see whether the
remaining gap is phrasing, field-derivation, output-key naming, or uncovered shapes.

    CUDA_VISIBLE_DEVICES="" python -m bench.diagnose --model <dir> --tok <tok>
"""
import argparse
import json
from pathlib import Path

import torch

from bench.harness import score_program
from bench.run import load, make_generate_fn

BENCH = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--mode", default="auto")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = [json.loads(l) for l in BENCH.open()]
    model, tok = load(Path(a.model), device, Path(a.tok))
    gen = make_generate_fn(model, tok, device, k=1, temperature=0.0, mode=a.mode)
    inval, wrong = [], []
    npass = nvalid = 0
    for it in items:
        prog = gen(it)[0]
        sc = score_program(prog, it)
        nvalid += sc["valid"]
        npass += sc["correct"]
        if sc["correct"]:
            continue
        row = (it["id"], it["request"], it.get("reference_program", ""), prog)
        (wrong if sc["valid"] else inval).append(row)
    n = len(items)
    print(json.dumps({"n": n, "pass@1": round(npass / n, 3), "valid@1": round(nvalid / n, 3)}))
    for label, rows in [("INVALID (did not parse/run)", inval), ("WRONG (ran, mismatch)", wrong)]:
        print(f"\n===== {label}: {len(rows)} =====")
        for id_, req, gold, pred in rows:
            print(f"[{id_}] {req}")
            print(f"   gold: {gold}")
            print(f"   pred: {pred}")


if __name__ == "__main__":
    main()
