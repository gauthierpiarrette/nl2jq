"""Build pipeline for the frozen benchmark (FROZEN_BENCH_DESIGN.md §e).

Two subcommands:

  gate    candidates_raw.jsonl -> candidates_gated.jsonl
          Re-executes every reference program under the pinned jq (expected_output is
          NEVER trusted from an author), then applies: degeneracy checks, within-set and
          vs-devset dedup, the novelty gate, and the T5 coverage gate. Prints a report;
          rejected items go to candidates_rejected.jsonl with reasons.

  freeze  candidates_final.jsonl -> frozen/nl2jq-bench-1.0.0.jsonl (400 public)
                                    + sealed/nl2jq-bench-sealed-v1.jsonl (100, unpublished)
          Enforces tier quotas, does a deterministic stratified 400/100 split (seed 42),
          assigns ids + canary ids, canonicalizes, hashes, writes FREEZE.txt.

    python -m bench.build_frozen gate bench/frozen/candidates_raw.jsonl
    python -m bench.build_frozen freeze bench/frozen/candidates_final.jsonl
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.harness import run_program, _resolve_jq  # noqa: E402
from bench.validate_novelty import build_vtrain, check_item  # noqa: E402
from bench.audit_coverage import check_grammar_clean, item_markers  # noqa: E402

VERSION = "1.0.0"
# final-set tier quotas: 500 total = 400 public + 100 sealed, 15/25/25/20/15%
QUOTA_TOTAL = {1: 75, 2: 125, 3: 125, 4: 100, 5: 75}
QUOTA_SEALED = {1: 15, 2: 25, 3: 25, 4: 20, 5: 15}
DEVSET = ROOT / "bench" / "nl2jq-bench.jsonl"
FROZEN_DIR = ROOT / "bench" / "frozen"
SEALED_DIR = ROOT / "bench" / "sealed"


def _norm_prog(p):
    return re.sub(r"\s+", " ", p.strip())


def _degenerate(outputs, request):
    if outputs is None:
        return "no output"
    flat = outputs[0] if len(outputs) == 1 and isinstance(outputs[0], list) else outputs
    req = request.lower()
    counting = any(w in req for w in ("how many", "count", "number of", "any ", "are there",
                                      "is there", "empty", "at all", "does ", "do any"))
    if isinstance(flat, list) and not flat and not counting:
        return "empty output"
    vals = flat if isinstance(flat, list) else [flat]
    if vals and all(v is None for v in vals):
        return "all-null output"
    return None


def cmd_gate(path):
    items = [json.loads(l) for l in open(path)]
    print(f"gating {len(items)} candidates")

    dirty = check_grammar_clean()
    if dirty:
        print("FATAL: grammar sources contain T5 markers:", dirty)
        sys.exit(1)

    vtrain, enum_values = build_vtrain()
    dev_progs = {_norm_prog(json.loads(l)["reference_program"]) for l in open(DEVSET)}

    kept, rejected, seen = [], [], set()
    for it in items:
        reasons = []
        prog = it["reference_program"]
        # 1. execute under pinned jq; pin expected_output ourselves
        ok, outputs = run_program(prog, json.dumps(it["input"]))
        if not ok:
            reasons.append("program failed under pinned jq")
        else:
            d = _degenerate(outputs, it["request"])
            if d:
                reasons.append(f"degenerate: {d}")
            it["expected_output"] = outputs
        # 2. dedup: within set (program shape + field set) and vs devset
        fields = set()
        from bench.validate_novelty import _input_fields
        _input_fields(it["input"], fields)
        key = (_norm_prog(prog), tuple(sorted(fields)))
        if key in seen:
            reasons.append("duplicate (program+fields) within candidate set")
        if _norm_prog(prog) in dev_progs:
            reasons.append("program identical to a devset item")
        # 3. novelty gate
        nv = check_item(it, vtrain, enum_values)
        if nv:
            reasons.append("novelty: " + "; ".join(nv[:3]))
        # 4. coverage gate (rule B)
        marks = item_markers(it)
        if it["tier"] == 5 and not marks:
            reasons.append("tier5 without any T5 construct")
        if it["tier"] != 5 and marks:
            reasons.append(f"tier{it['tier']} uses T5 construct {marks}")

        if reasons:
            rejected.append({**it, "_reject": reasons})
        else:
            seen.add(key)
            kept.append(it)

    out = Path(path).parent / "candidates_gated.jsonl"
    rej = Path(path).parent / "candidates_rejected.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
    rej.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rejected))
    print(f"kept {len(kept)}  rejected {len(rejected)}  -> {out.name}, {rej.name}")
    print("kept per tier:", dict(sorted(Counter(x['tier'] for x in kept).items())))
    n_abs = sum(bool(x.get("abstract")) for x in kept)
    print(f"abstract: {n_abs} ({100 * n_abs / max(len(kept), 1):.1f}%)")
    reasons_hist = Counter(r.split(":")[0] for x in rejected for r in x["_reject"])
    print("reject reasons:", dict(reasons_hist))


def cmd_freeze(path):
    items = [json.loads(l) for l in open(path)]
    by_tier = {t: [x for x in items if x["tier"] == t] for t in QUOTA_TOTAL}
    for t, q in QUOTA_TOTAL.items():
        if len(by_tier[t]) < q:
            print(f"FATAL: tier {t} has {len(by_tier[t])} < quota {q}")
            sys.exit(1)

    rng = random.Random(42)
    public, sealed = [], []
    for t, q in QUOTA_TOTAL.items():
        pool = sorted(by_tier[t], key=lambda x: _norm_prog(x["reference_program"]))
        rng.shuffle(pool)
        chosen = pool[:q]
        sealed_t = chosen[:QUOTA_SEALED[t]]
        public_t = chosen[QUOTA_SEALED[t]:]
        sealed.extend(sealed_t)
        public.extend(public_t)

    def finalize(rows, prefix):
        rng2 = random.Random(7)
        rng2.shuffle(rows)
        out = []
        for i, it in enumerate(rows, 1):
            iid = f"{prefix}-{i:03d}"
            canary = hashlib.sha256(f"njb1:{iid}:{it['request']}".encode()).hexdigest()[:12]
            out.append({
                "id": iid,
                "request": it["request"],
                "input": it["input"],
                "reference_program": it["reference_program"],
                "expected_output": it["expected_output"],
                "tags": it["tags"],
                "tier": it["tier"],
                "difficulty": it["tier"],
                "order_insensitive": bool(it.get("order_insensitive")),
                "abstract": bool(it.get("abstract")),
                "domain": it["domain"],
                "novel_fields": it.get("novel_fields", []),
                "grammar_covered": it["tier"] != 5,
                "provenance": "hand",
                "jq_version": "1.7.1",
                "frozen_in": VERSION,
                "canary_id": canary,
                "license": "CC-BY-4.0",
                "source": "original",
            })
            if "acceptable_outputs" in it:
                out[-1]["acceptable_outputs"] = it["acceptable_outputs"]
        return out

    pub_rows = finalize(public, "v1")
    sea_rows = finalize(sealed, "s1")

    FROZEN_DIR.mkdir(exist_ok=True)
    SEALED_DIR.mkdir(exist_ok=True)
    pub_path = FROZEN_DIR / f"nl2jq-bench-{VERSION}.jsonl"
    sea_path = SEALED_DIR / "nl2jq-bench-sealed-v1.jsonl"

    def canonical(rows):
        return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")) + "\n" for r in rows)

    pub_txt, sea_txt = canonical(pub_rows), canonical(sea_rows)
    pub_path.write_text(pub_txt)
    sea_path.write_text(sea_txt)

    jq_bin = _resolve_jq()
    jq_sha = hashlib.sha256(Path(jq_bin).read_bytes()).hexdigest()
    jq_ver = subprocess.run([jq_bin, "--version"], capture_output=True,
                            text=True).stdout.strip()
    hist = dict(sorted(Counter(r["tier"] for r in pub_rows).items()))
    freeze = (
        f"nl2jq-bench v{VERSION} — FREEZE RECORD\n"
        f"frozen: {date.today().isoformat()}\n"
        f"public file: {pub_path.name}\n"
        f"public sha256: {hashlib.sha256(pub_txt.encode()).hexdigest()}\n"
        f"public items: {len(pub_rows)}  tier histogram: {hist}\n"
        f"sealed sha256: {hashlib.sha256(sea_txt.encode()).hexdigest()}\n"
        f"sealed items: {len(sea_rows)} (unpublished canary set)\n"
        f"jq: {jq_ver}  binary sha256: {jq_sha}\n"
        f"identity: the public sha256 above IS the benchmark's identity. Any change to\n"
        f"the items is a new version with a new hash; this file is append-only.\n"
    )
    (FROZEN_DIR / "FREEZE.txt").write_text(freeze)
    print(freeze)
    print(f"wrote {pub_path}, {sea_path}, FREEZE.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "freeze"])
    ap.add_argument("path")
    a = ap.parse_args()
    (cmd_gate if a.cmd == "gate" else cmd_freeze)(a.path)


if __name__ == "__main__":
    main()
