"""Sharded, multiprocess dataset generation for v1/v2 scale.

Each worker generates a stream of verified records (seeded distinctly) and reports them
with their behavioral-dedup key; the parent does global dedup, schema-family splitting,
and writes train/val JSONL + manifest.

    python -m pipeline.build_parallel --n 500000 --out data/v1 --workers 10
"""
import argparse
import hashlib
import json
import random
import time
from collections import Counter
from multiprocessing import Process, Queue
from pathlib import Path

from .common import DATA_DIR
from .generate import generate_one

VAL_FRACTION = 0.05
BATCH = 200


def worker(seed: int, out_q: Queue, stop_flag, target_per_worker: int):
    rng = random.Random(seed)
    produced = 0
    local = []
    reasons = Counter()
    while produced < target_per_worker:
        rec, key, fam = generate_one(rng)
        if rec is None:
            reasons[key] += 1
            continue
        local.append((rec, key, fam))
        produced += 1
        if len(local) >= BATCH:
            out_q.put(("data", local, dict(reasons)))
            local, reasons = [], Counter()
    if local:
        out_q.put(("data", local, dict(reasons)))
    out_q.put(("done", seed, None))


def _hash_key(fam, key):
    return hashlib.blake2b(repr((fam, key)).encode(), digest_size=12).digest()


def build(n_target, out_dir, workers, base_seed):
    out_q = Queue(maxsize=workers * 4)
    per = n_target // workers + BATCH
    procs = [Process(target=worker, args=(base_seed + i, out_q, None, per))
             for i in range(workers)]
    for p in procs:
        p.start()

    # Stream records straight to disk; hold only a compact seen-key set + family splits
    # in RAM (keep-first on behavioral collision), so memory stays flat at 2M+ scale.
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = (out_dir / "train.jsonl").open("w")
    vf = (out_dir / "val.jsonl").open("w")
    seen = set()
    val_families = set()
    seen_families = set()
    reasons = Counter()
    tags, tiers = Counter(), Counter()
    n_kept = n_train = n_val = 0
    rng = random.Random(base_seed ^ 0x5151)
    t0 = time.time()
    done = 0
    while done < workers and n_kept < n_target:
        kind, payload, r = out_q.get()
        if kind == "done":
            done += 1
            continue
        for name, c in (r or {}).items():
            reasons[name] += c
        for rec, key, fam in payload:
            hk = _hash_key(fam, key)
            if hk in seen:
                reasons["behavioral_dup"] += 1
                continue
            seen.add(hk)
            if fam not in seen_families:
                seen_families.add(fam)
                if rng.random() < VAL_FRACTION:
                    val_families.add(fam)
            is_val = fam in val_families
            rec["split"] = "val" if is_val else "train"
            (vf if is_val else tf).write(json.dumps(rec) + "\n")
            tags[rec["tags"][0]] += 1
            tiers[rec["tier"]] += 1
            n_kept += 1
            n_val += is_val
            n_train += not is_val
            if n_kept % 50000 == 0:
                print(f"  {n_kept}/{n_target} kept ({time.time()-t0:.0f}s, "
                      f"{n_kept/(time.time()-t0):.0f}/s)")

    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=2)
    tf.close()
    vf.close()
    manifest = {"n_kept": n_kept, "n_train": n_train, "n_val": n_val,
                "val_families": len(val_families), "seen_families": len(seen_families),
                "seconds": round(time.time() - t0, 1),
                "reject_reasons": dict(reasons), "by_primary_tag": dict(tags),
                "by_tier": {str(k): v for k, v in tiers.items()},
                "workers": workers, "base_seed": base_seed}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500000)
    ap.add_argument("--out", default=str(DATA_DIR / "v1"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    build(a.n, Path(a.out), a.workers, a.seed)


if __name__ == "__main__":
    main()
