"""Diagnostic: show what a frontier model actually emits on specific bench items,
so we can tell extraction failures from legitimate stylistic output-structure mismatch."""
import json
import sys
from pathlib import Path

from bench.baselines import anthropic_generate, _clean
from bench.harness import score_program
from cli.jqgen import build_context
from pipeline.execute import run_program

BENCH = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"


def main():
    ids = set(sys.argv[1:]) or {"seed-001", "seed-005", "seed-019", "seed-024", "seed-027"}
    items = {j["id"]: j for j in (json.loads(l) for l in BENCH.open())}
    import anthropic
    client = anthropic.Anthropic()
    from bench.baselines import SYSTEM
    for tid in sorted(ids):
        item = items[tid]
        context = build_context(item["input"], "auto")
        user = f"Request: {item['request']}\nJSON sample: {context}"
        resp = client.messages.create(model="claude-opus-4-8", max_tokens=512,
                                       system=SYSTEM, messages=[{"role": "user", "content": user}])
        raw = next((b.text for b in resp.content if b.type == "text"), "")
        prog = _clean(raw)
        ok, produced = run_program(prog, json.dumps(item["input"]))
        sc = score_program(prog, item)
        print(f"\n=== {tid}: {item['request']}")
        print(f"  ref_program : {item['reference_program']}")
        print(f"  expected    : {item['expected_output']}")
        print(f"  opus RAW    : {raw!r}")
        print(f"  opus prog   : {prog!r}")
        print(f"  produced    : ok={ok} {produced}")
        print(f"  score       : {sc}")


if __name__ == "__main__":
    main()
