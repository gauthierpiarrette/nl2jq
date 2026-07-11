"""Curation tool for real-world nl2jq-bench items (SPEC §4).

Real items are collected as records with a source URL + author for attribution, then
verified: the reference_program must execute cleanly and produce non-degenerate output.
Only verified items are appended to nl2jq-bench.jsonl.

Sources and licensing:
  - Stack Overflow `[jq]` accepted answers  -> license "CC-BY-SA-4.0", record question URL + answer author
  - tldr pages / jq manual                  -> reworded, license "CC-BY-4.0"
  - hand-authored originals                  -> license "CC-BY-4.0"

Input: a JSON/JSONL file of candidate items, e.g.
  {"request": "...", "input": <json>, "reference_program": "...", "tags": [...],
   "difficulty": 2, "order_insensitive": false,
   "source_url": "https://stackoverflow.com/q/...", "author": "user123",
   "license": "CC-BY-SA-4.0"}

    python -m bench.curate --add candidates.jsonl
    python -m bench.curate --stats
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.execute import run_program

BENCH = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"
REQUIRED = ("request", "input", "reference_program", "tags")


def load_bench():
    return [json.loads(l) for l in BENCH.open()] if BENCH.exists() else []


def verify(item):
    for k in REQUIRED:
        if k not in item:
            return None, f"missing field {k}"
    ok, out = run_program(item["reference_program"], json.dumps(item["input"]))
    if not ok:
        return None, "reference program errored"
    if not out or json.dumps(out) in ("[]", "[null]", "[{}]"):
        return None, "degenerate output"
    if item.get("license", "").startswith("CC-BY-SA") and not item.get("source_url"):
        return None, "SO-derived item needs source_url for attribution"
    item = dict(item)
    item["expected_output"] = out
    item.setdefault("order_insensitive", False)
    item.setdefault("license", "CC-BY-4.0")
    item.setdefault("source", "curated")
    return item, None


def add(path):
    existing = load_bench()
    seen = {(e["request"], json.dumps(e["input"])) for e in existing}
    ids = {e["id"] for e in existing}
    raw = Path(path).read_text().strip()
    cands = ([json.loads(raw)] if raw.startswith("{") and "\n" not in raw
             else [json.loads(l) for l in raw.splitlines() if l.strip()])
    added, rejected = [], []
    next_n = len([e for e in existing if e["id"].startswith("cur-")]) + 1
    for c in cands:
        if (c["request"], json.dumps(c.get("input"))) in seen:
            rejected.append((c.get("request"), "duplicate"))
            continue
        item, err = verify(c)
        if err:
            rejected.append((c.get("request"), err))
            continue
        if "id" not in item or item["id"] in ids:
            item["id"] = f"cur-{next_n:03d}"
            next_n += 1
        ids.add(item["id"])
        added.append(item)
    with BENCH.open("a") as f:
        for it in added:
            f.write(json.dumps(it) + "\n")
    print(f"added {len(added)}, rejected {len(rejected)}")
    for r, why in rejected:
        print(f"  REJECT [{why}] {r!r}")


def stats():
    items = load_bench()
    print(f"total items: {len(items)}")
    print("by license:", dict(Counter(i.get("license", "?") for i in items)))
    print("by difficulty:", dict(Counter(i.get("difficulty", "?") for i in items)))
    print("by source:", dict(Counter(i.get("source", "?") for i in items)))
    print("tag coverage:", dict(Counter(t for i in items for t in i["tags"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", help="JSON/JSONL file of candidate items")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.add:
        add(a.add)
    if a.stats or not a.add:
        stats()


if __name__ == "__main__":
    main()
