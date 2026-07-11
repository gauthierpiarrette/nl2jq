"""Score an oracle dump ({id, programs:[k]} jsonl) against a dev items file.

Reports pass@1 (greedy = first program), oracle@4/8/k (any candidate correct), and
verifier-rerank@k (pick the first candidate whose execution is informative — runs and
yields non-empty, non-all-null output; approximates what the CLI's execution filter
would select without knowing the answer). The pass@1 <-> oracle gap is the headroom
that reranking/RL could capture; verifier-rerank shows how much of it FREE selection
already gets.

    python -m bench.score_oracle oracle_qwenv6_k16.jsonl bench/devnovel/devnovel-v1.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.harness import score_program, run_program  # noqa: E402


def _informative(prog, doc_json):
    ok, outputs = run_program(prog, doc_json)
    if not ok or outputs is None or not outputs:
        return False
    return not all(v is None for v in outputs)


def main():
    dump_path, items_path = sys.argv[1], sys.argv[2]
    dumps = {json.loads(l)["id"]: json.loads(l)["programs"] for l in open(dump_path)}
    items = [json.loads(l) for l in open(items_path)]
    ks = (1, 4, 8, 16)
    oracle = {k: 0 for k in ks}
    rerank = 0
    n = 0
    for it in items:
        progs = dumps.get(it["id"])
        if not progs:
            continue
        n += 1
        correct = [score_program(p, it)["correct"] for p in progs]
        for k in ks:
            if any(correct[:k]):
                oracle[k] += 1
        # verifier-rerank: first informative candidate wins
        doc_json = json.dumps(it["input"])
        pick = next((i for i, p in enumerate(progs) if _informative(p, doc_json)), 0)
        rerank += correct[pick]
    print(f"n={n}")
    for k in ks:
        print(f"oracle@{k}: {oracle[k] / n:.3f}" + ("  (= pass@1, greedy)" if k == 1 else ""))
    print(f"verifier-rerank@16: {rerank / n:.3f}")
    gap = oracle[16] / n - oracle[1] / n
    print(f"headroom (oracle@16 - pass@1): {gap:+.3f} -> "
          + ("rerank/RL worth pursuing" if gap >= 0.12 else "little to gain from selection"))


if __name__ == "__main__":
    main()
