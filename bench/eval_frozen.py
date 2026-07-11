"""One-shot evaluation on the FROZEN benchmark (FROZEN_BENCH_DESIGN.md §e step 7).

Discipline: the frozen bench is scored ONCE per model release. No checkpoint selection,
no per-item diagnosis feeding back into training data. Results append to
bench/frozen/RESULTS.md together with the bench content hash.

    python -m bench.eval_frozen --backend flagship --model artifacts/nl2jq-40m-hf
    python -m bench.eval_frozen --backend qwen --model artifacts/nl2jq-qwen3-0.6b
    python -m bench.eval_frozen --backend anthropic --model claude-opus-4-8
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.harness import score_program  # noqa: E402
from cli.jqgen import build_context  # noqa: E402

FROZEN = ROOT / "bench" / "frozen" / "nl2jq-bench-1.0.0.jsonl"
RESULTS = ROOT / "bench" / "frozen" / "RESULTS.md"

SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
          "jq program (jq 1.7). Output ONLY the jq program on one line — no explanation, "
          "no markdown fences, no `jq` prefix.")


def _clean(text):
    text = re.sub(r"(?s)^\s*<think>.*?</think>\s*", "", text).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    if text.startswith("jq "):
        text = text[3:].strip()
    return text.strip().strip("`").strip("'").strip()


def gen_local(backend, model_dir, grounded=False, k=4, temperature=0.7):
    import torch
    from cli.jqgen import load_model, generate_flagship, generate_qwen, resolve_jq
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(model_dir, device)
    g = generate_qwen if backend == "qwen" else generate_flagship

    if not grounded:
        def fn(item):
            return g(model, tok, item["request"],
                     build_context(item["input"], "auto"), device)
        return fn

    from cli.decoding import grounded_pick
    jq_bin = resolve_jq()

    def fn(item):
        ctx = build_context(item["input"], "auto")
        cands = [g(model, tok, item["request"], ctx, device)]
        for _ in range(k - 1):
            cands.append(g(model, tok, item["request"], ctx, device,
                           temperature=temperature))
        prog, _meta = grounded_pick(cands, item["input"], item["request"], jq_bin)
        return prog
    return fn


def gen_anthropic(model):
    import anthropic
    client = anthropic.Anthropic()

    def fn(item):
        user = (f"Request: {item['request']}\n"
                f"JSON sample: {build_context(item['input'], 'auto')}")
        resp = client.messages.create(model=model, max_tokens=512, system=SYSTEM,
                                      messages=[{"role": "user", "content": user}])
        if resp.stop_reason == "refusal":
            return ""
        return _clean(next((b.text for b in resp.content if b.type == "text"), ""))
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["flagship", "qwen", "anthropic", "preds"],
                    required=True,
                    help="preds = score a predictions file produced elsewhere (e.g. GPU "
                         "generation on a pod), so scoring always runs under the pinned jq")
    ap.add_argument("--model", required=True,
                    help="local dir / repo id / API model name / predictions .jsonl "
                         "({id, program} lines) when --backend preds")
    ap.add_argument("--items", default=str(FROZEN))
    ap.add_argument("--label", default=None, help="name for the RESULTS.md row")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke test: print summary, do NOT append to RESULTS.md")
    ap.add_argument("--grounded", action="store_true",
                    help="input-grounded decoding: k candidates, field repair against "
                         "the input's actual keys, execution filtering (cli/decoding.py)")
    ap.add_argument("--k", type=int, default=4, help="candidates for --grounded")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.items)]
    bench_sha = hashlib.sha256(Path(a.items).read_bytes()).hexdigest()
    if a.backend == "preds":
        preds = {json.loads(l)["id"]: json.loads(l)["program"] for l in open(a.model)}
        missing = [it["id"] for it in items if it["id"] not in preds]
        if missing:
            print(f"WARNING: {len(missing)} items missing from predictions "
                  f"(scored invalid): {missing[:5]}", file=sys.stderr)
        gen = lambda item: _clean(preds.get(item["id"], ""))  # noqa: E731
    else:
        gen = (gen_anthropic(a.model) if a.backend == "anthropic"
               else gen_local(a.backend, a.model, grounded=a.grounded, k=a.k))

    per_tier = defaultdict(lambda: [0, 0, 0])  # correct, valid, n
    for i, it in enumerate(items, 1):
        try:
            prog = gen(it)
        except Exception as e:  # count API/generation failures as invalid, keep going
            print(f"  [{it['id']}] generation error: {e}", file=sys.stderr)
            prog = ""
        sc = score_program(prog, it) if prog else {"valid": False, "correct": False}
        t = it["tier"]
        per_tier[t][0] += sc["correct"]
        per_tier[t][1] += sc["valid"]
        per_tier[t][2] += 1
        if i % 50 == 0:
            done = sum(v[2] for v in per_tier.values())
            corr = sum(v[0] for v in per_tier.values())
            print(f"  {done}/{len(items)}  running pass@1={corr / done:.3f}")

    n = sum(v[2] for v in per_tier.values())
    correct = sum(v[0] for v in per_tier.values())
    valid = sum(v[1] for v in per_tier.values())
    label = a.label or f"{a.backend}:{a.model}"
    tier_cells = []
    for t in sorted(per_tier):
        c, _v, tn = per_tier[t]
        tier_cells.append(f"{c / tn:.2f}")
    t5 = per_tier.get(5, [0, 0, 1])
    summary = {
        "label": label, "pass@1": round(correct / n, 3), "valid@1": round(valid / n, 3),
        "per_tier": {t: round(per_tier[t][0] / per_tier[t][2], 3) for t in sorted(per_tier)},
        "T5_generalization": round(t5[0] / t5[2], 3), "n": n, "bench_sha256": bench_sha,
    }
    print(json.dumps(summary, indent=2))

    if a.dry_run:
        print("dry run — not appended to RESULTS.md")
        return
    row = (f"| {date.today().isoformat()} | {label} | {correct / n:.2f} | {valid / n:.2f} | "
           + " | ".join(tier_cells) + f" | {t5[0] / t5[2]:.2f} | `{bench_sha[:12]}` |\n")
    if not RESULTS.exists():
        RESULTS.write_text(
            "# nl2jq-bench frozen results (append-only)\n\n"
            "One row per official run. No checkpoint selection against this file.\n\n"
            "| date | model | pass@1 | valid@1 | T1 | T2 | T3 | T4 | T5 | T5-gen | bench sha |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(RESULTS, "a") as f:
        f.write(row)
    print(f"appended -> {RESULTS}")


if __name__ == "__main__":
    main()
