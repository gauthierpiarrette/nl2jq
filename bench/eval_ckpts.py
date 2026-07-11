"""Score several checkpoints (model_*.pt + model.pt) against the 100-item DEV split and
report pass@1 / valid@1 for each, so we keep the best-on-devset step rather than the last.

DEV SPLIT ONLY. Checkpoint selection is development — it must never see the frozen
release benchmark (bench/frozen/*), or its scores stop being held-out. This script is
hard-wired to the devset and refuses any frozen path (FROZEN_BENCH_DESIGN.md §e step 1).

    python -m bench.eval_ckpts --dir artifacts/nl2jq-40m-v4 --tok artifacts/tok4
"""
import argparse
import json
from pathlib import Path

import torch

from bench.harness import score_items
from bench.run import load, make_generate_fn

# the 100-item development split (formerly "nl2jq-bench"; demoted to devset-v0)
BENCH = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"
_FROZEN = Path(__file__).resolve().parent / "frozen"


def _steps(dir_path: Path):
    """model_*.pt (numbered) then model.pt (final), ascending by step."""
    numbered = sorted(dir_path.glob("model_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))
    final = dir_path / "model.pt"
    out = list(numbered)
    if final.exists():
        out.append(final)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--items", default=str(BENCH),
                    help="DEV items to select on (devset or dev-novel; NEVER bench/frozen/*)")
    a = ap.parse_args()
    if "frozen" in a.items:
        raise SystemExit("refusing: checkpoint selection on the frozen benchmark is "
                         "test-set contamination (FROZEN_BENCH_DESIGN.md §e).")
    if _FROZEN.exists() and any(_FROZEN.glob("nl2jq-bench-*.jsonl")):
        print("NOTE: frozen benchmark exists; this tool scores DEV splits only. "
              "Official frozen scores: python -m bench.eval_frozen (one-shot, no selection).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = [json.loads(l) for l in open(a.items)]
    ckpts = _steps(Path(a.dir))
    print(f"evaluating {len(ckpts)} checkpoints on {len(items)} bench items (device={device})")
    results = []
    for ck in ckpts:
        blob = torch.load(ck, map_location=device, weights_only=False)
        model, tok = load(Path(a.dir), device, Path(a.tok))  # rebuild + load this ckpt's weights
        model.load_state_dict(blob["model"])
        gen = make_generate_fn(model, tok, device, k=1, temperature=0.0, mode=a.mode)
        res = score_items(items, gen, k=1)
        row = {"ckpt": ck.name, "step": blob.get("step"),
               "pass@1": round(res["pass@1"], 3), "valid@1": round(res.get("valid@1", 0), 3)}
        results.append(row)
        print("  ", json.dumps(row))
    results.sort(key=lambda r: r["pass@1"], reverse=True)
    print("\nBEST:", json.dumps(results[0]))
    print("RANKED:", json.dumps(results))


if __name__ == "__main__":
    main()
