"""Execute each seed task's reference_program with pinned jq to fill expected_output,
then write bench/nl2jq-bench.jsonl. Fails loudly if any reference program errors —
a broken reference must never enter the benchmark.
"""
import json
from pathlib import Path

from bench.seed_tasks import SEED
from bench.seed_tasks_ext import SEED_EXT
from bench.seed_tasks_hard import SEED_HARD
from pipeline.execute import run_program

OUT = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"

ALL_TASKS = SEED + SEED_EXT + SEED_HARD


def main():
    items, bad = [], []
    seen_ids = set()
    for t in ALL_TASKS:
        if t["id"] in seen_ids:
            raise SystemExit(f"duplicate task id: {t['id']}")
        seen_ids.add(t["id"])
        ok, out = run_program(t["reference_program"], json.dumps(t["input"]))
        if not ok or not out:
            bad.append((t["id"], t["reference_program"]))
            continue
        item = dict(t)
        item["expected_output"] = out
        item.setdefault("order_insensitive", False)
        item.setdefault("license", "CC-BY-4.0")
        item.setdefault("source", "original")
        items.append(item)
    if bad:
        print("BROKEN REFERENCE PROGRAMS:")
        for i, p in bad:
            print(f"  {i}: {p}")
        raise SystemExit(1)
    with OUT.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} verified items -> {OUT}")
    from collections import Counter
    tags = Counter(t for it in items for t in it["tags"])
    print("tag coverage:", dict(tags))


if __name__ == "__main__":
    main()
