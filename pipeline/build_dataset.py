"""Orchestrate the M0 pipeline: schemas -> docs -> programs -> execute+filter -> dedup
-> template NL -> context modes -> schema-family split -> JSONL.

Usage:
    python -m pipeline.build_dataset --n 10000 --out data/v0 --seed 0
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from .common import DATA_DIR, canon_outputs, format_example
from .documents import sample_documents
from .execute import run_program
from .grammar import sample_task
from .schemas import sample_schema
from .shape import raw_prefix, shape_sketch

VAL_FRACTION = 0.08


def build(n_target: int, out_dir: Path, seed: int):
    rng = random.Random(seed)
    manifest = Counter()
    behavioral = {}          # dedup key -> canonical example
    val_families = set()     # schema families reserved for validation
    t0 = time.time()
    attempts = 0

    while len(behavioral) < n_target and attempts < n_target * 40:
        attempts += 1
        manifest["attempts"] += 1
        schema_info = sample_schema(rng)
        docs = sample_documents(schema_info, rng, n_docs=4)

        task = sample_task(schema_info, docs, rng)
        if task is None:
            manifest["no_task"] += 1
            continue
        manifest["programs_sampled"] += 1

        # execute on all docs; require success + non-degenerate behavior
        all_outputs, ok_all = [], True
        for d in docs:
            ok, out = run_program(task["program"], json.dumps(d))
            if not ok:
                ok_all = False
                break
            all_outputs.append(out)
        if not ok_all:
            manifest["exec_failed"] += 1
            continue

        # the SHOWN document's output (doc0) must itself be non-degenerate
        shown = canon_outputs(all_outputs[0])
        if shown in ("", "null", "[]", "{}", "0", '""'):
            manifest["degenerate_shown"] += 1
            continue
        flat = canon_outputs([v for o in all_outputs for v in o])
        if flat in ("", "null", "[]", "{}"):
            manifest["degenerate_output"] += 1
            continue
        # reject constant-across-docs outputs (except intentional length/keys tasks)
        per_doc = [canon_outputs(o) for o in all_outputs]
        if len(set(per_doc)) == 1 and task["tags"][0] not in ("keys",):
            manifest["constant_output"] += 1
            continue

        # behavioral dedup: (family, tuple of outputs) -> keep shortest program
        beh_key = (schema_info["family"], tuple(per_doc))
        prev = behavioral.get(beh_key)
        if prev and len(prev["program"]) <= len(task["program"]):
            manifest["behavioral_dup"] += 1
            continue

        # assign split by schema family (hold out whole families)
        fam = schema_info["family"]
        if fam not in val_families and prev is None:
            if rng.random() < VAL_FRACTION:
                val_families.add(fam)
        split = "val" if fam in val_families else "train"

        # build the example against the FIRST doc (the one shown to the model)
        doc0 = docs[0]
        use_shape = rng.random() < 0.5
        context = shape_sketch(doc0) if use_shape else raw_prefix(doc0)
        request = rng.choice(task["nl"])

        example = {
            "request": request,
            "context": context,
            "context_mode": "shape" if use_shape else "raw",
            "program": task["program"],
            "expected_output": all_outputs[0],
            "text": format_example(request, context, task["program"]),
            "tier": task["tier"],
            "tags": task["tags"],
            "domain": schema_info["domain"],
            "family": fam,
            "split": split,
            # keep the true first doc so eval can re-execute exactly
            "input_doc": doc0,
        }
        behavioral[beh_key] = example
        manifest[f"kept_{split}"] += 1
        if len(behavioral) % 1000 == 0:
            print(f"  {len(behavioral)}/{n_target} kept "
                  f"({attempts} attempts, {time.time() - t0:.0f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_f = (out_dir / "train.jsonl").open("w")
    val_f = (out_dir / "val.jsonl").open("w")
    tag_counter, tier_counter = Counter(), Counter()
    for ex in behavioral.values():
        (val_f if ex["split"] == "val" else train_f).write(json.dumps(ex) + "\n")
        tag_counter[ex["tags"][0]] += 1
        tier_counter[ex["tier"]] += 1
    train_f.close()
    val_f.close()

    manifest_out = {
        "n_kept": len(behavioral),
        "n_train": manifest["kept_train"],
        "n_val": manifest["kept_val"],
        "val_families": len(val_families),
        "seconds": round(time.time() - t0, 1),
        "yield": round(len(behavioral) / max(1, manifest["programs_sampled"]), 3),
        "stage_counts": dict(manifest),
        "by_primary_tag": dict(tag_counter),
        "by_tier": {str(k): v for k, v in tier_counter.items()},
        "seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(json.dumps(manifest_out, indent=2))
    return manifest_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", type=str, default=str(DATA_DIR / "v0"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.n, Path(args.out), args.seed)


if __name__ == "__main__":
    main()
