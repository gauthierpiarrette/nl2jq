"""Apply adversarial-review verdicts to the gated candidate set.

Takes candidates_gated.jsonl (order = _k index) + a verdicts JSON and produces
candidates_final.jsonl:

  - drop    -> item removed
  - fix     -> fixed_request / fixed_program / fixed_tier / order_insensitive applied
  - alt_programs (any verdict) -> each alt is EXECUTED under the pinned jq; outputs that
    differ from the primary expected_output become acceptable_outputs entries (the claim
    "this alternative is also correct" is never trusted textually — only its executed
    output is).

Every surviving item is then re-verified end to end: program re-executed (expected_output
refreshed), degeneracy, novelty, and tier/coverage checks re-applied. Violators are
dropped loudly rather than shipped.

    python -m bench.apply_review bench/frozen/candidates_gated.jsonl verdicts.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.harness import run_program  # noqa: E402
from bench.validate_novelty import build_vtrain, check_item  # noqa: E402
from bench.audit_coverage import item_markers  # noqa: E402
from bench.build_frozen import _degenerate  # noqa: E402


def main():
    gated_path, verdicts_path = sys.argv[1], sys.argv[2]
    items = [json.loads(l) for l in open(gated_path)]
    verdicts = {v["k"]: v for v in json.load(open(verdicts_path))["verdicts"]}
    vtrain, enum_values = build_vtrain()

    out, dropped, fixed_n, alts_n = [], [], 0, 0
    for k, it in enumerate(items):
        v = verdicts.get(k, {"verdict": "keep"})
        if v["verdict"] == "drop":
            dropped.append((k, v.get("reason", "")))
            continue
        if v["verdict"] == "fix":
            fixed_n += 1
            if v.get("fixed_request"):
                it["request"] = v["fixed_request"]
            if v.get("fixed_program"):
                it["reference_program"] = v["fixed_program"]
            if v.get("fixed_tier"):
                # Tier 5 is CONSTRUCT-defined (beyond-grammar), not difficulty-defined.
                # A reviewer may not move an item across the T5 boundary against its
                # constructs — that fights the coverage gate and drops a valid item.
                has_t5 = bool(item_markers(it))
                if (has_t5 and v["fixed_tier"] != 5) or (not has_t5 and v["fixed_tier"] == 5):
                    print(f"  ignore fixed_tier={v['fixed_tier']} on _k={k} "
                          f"(T5 constructs present={has_t5}; tier is construct-defined)")
                else:
                    it["tier"] = v["fixed_tier"]
        if "order_insensitive" in v:
            it["order_insensitive"] = v["order_insensitive"]

        # re-execute the (possibly fixed) reference; never trust stale outputs
        ok, outputs = run_program(it["reference_program"], json.dumps(it["input"]))
        if not ok:
            dropped.append((k, "post-fix program failed under jq"))
            continue
        deg = _degenerate(outputs, it["request"])
        if deg:
            dropped.append((k, f"post-fix degenerate: {deg}"))
            continue
        it["expected_output"] = outputs

        # alt programs -> executed acceptable_outputs
        accept = [outputs]
        for alt in v.get("alt_programs", []):
            aok, aout = run_program(alt, json.dumps(it["input"]))
            if aok and aout is not None and aout not in accept:
                accept.append(aout)
                alts_n += 1
        if len(accept) > 1:
            it["acceptable_outputs"] = accept

        # re-gate: novelty + tier/coverage
        nv = check_item(it, vtrain, enum_values)
        if nv:
            dropped.append((k, "post-fix novelty: " + nv[0]))
            continue
        marks = item_markers(it)
        if it["tier"] == 5 and not marks:
            dropped.append((k, "post-fix tier5 without T5 construct"))
            continue
        if it["tier"] != 5 and marks:
            dropped.append((k, f"post-fix tier{it['tier']} uses T5 {marks}"))
            continue
        it.pop("_k", None)
        out.append(it)

    final = Path(gated_path).parent / "candidates_final.jsonl"
    final.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"kept {len(out)}  fixed {fixed_n}  alt-outputs added {alts_n}  "
          f"dropped {len(dropped)}")
    for k, r in dropped:
        print(f"  drop _k={k}: {r}")
    print("final per tier:", dict(sorted(Counter(x['tier'] for x in out).items())))
    print(f"-> {final}")


if __name__ == "__main__":
    main()
