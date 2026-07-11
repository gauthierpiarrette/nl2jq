"""Paraphrase track + round-trip consistency filter (SPEC §3.6–3.7).

Rewrites template-generated requests into natural, varied human phrasings using an
OpenAI-compatible endpoint (vLLM-served open-weights model for license cleanliness, or
any API). Then re-solves each paraphrase independently and keeps it only if the resulting
program's output matches the original — killing mislabeled pairs.

    python -m pipeline.paraphrase --data data/v2 --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3-32B --share 0.4

Requires `pip install openai`. Designed to be resumable and to LOG the rejection rate
(if >25%, fix the paraphraser prompt, not the filter).
"""
import argparse
import json
import random
from pathlib import Path

from .execute import outputs_match, run_program

STYLES = [
    "terse, like a shell one-liner comment",
    "a full polite question",
    "casual and slightly sloppy, how a tired engineer types",
    "precise and technical",
    "using a synonym for the field names where natural",
]

PARAPHRASE_SYS = (
    "You rewrite a request for a JSON data transformation into a different natural "
    "phrasing with the SAME meaning. Keep every constraint (fields, filters, numbers). "
    "Do not add or drop conditions. Reply with ONLY the rewritten request.")

SOLVE_SYS = (
    "Given a request and a JSON sample, output ONLY a single jq program (jq 1.7) that "
    "fulfills the request. No explanation, no code fences.")


def _client(endpoint, key):
    from openai import OpenAI
    return OpenAI(base_url=endpoint, api_key=key or "EMPTY")


def paraphrase_one(client, model, request, style):
    r = client.chat.completions.create(
        model=model, temperature=0.9, max_tokens=80,
        messages=[{"role": "system", "content": PARAPHRASE_SYS},
                  {"role": "user", "content": f"Style: {style}\nRequest: {request}"}])
    return r.choices[0].message.content.strip().strip('"')


def solve(client, model, request, context):
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=160,
        messages=[{"role": "system", "content": SOLVE_SYS},
                  {"role": "user", "content": f"Request: {request}\nJSON sample: {context}"}])
    return r.choices[0].message.content.strip().strip("`").removeprefix("jq").strip()


def consistency_ok(client, model, paraphrase, row):
    """Re-solve the paraphrase; keep only if its program matches the gold output."""
    prog = solve(client, model, paraphrase, row["context"])
    ok, produced = run_program(prog, json.dumps(row["input_doc"]))
    return ok and outputs_match(produced, row["expected_output"])


def run(data_dir, endpoint, model, key, share, seed):
    rng = random.Random(seed)
    client = _client(endpoint, key)
    src = Path(data_dir) / "train.jsonl"
    out = Path(data_dir) / "train_paraphrased.jsonl"
    rows = [json.loads(l) for l in src.open()]
    kept = rejected = 0
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")  # keep the template original
            if rng.random() > share:
                continue
            style = rng.choice(STYLES)
            try:
                para = paraphrase_one(client, model, row["request"], style)
                if not consistency_ok(client, model, para, row):
                    rejected += 1
                    continue
            except Exception as e:  # noqa: BLE001 - never let one call kill the run
                rejected += 1
                continue
            new = dict(row)
            new["request"] = para
            new["context_mode"] += "+paraphrase"
            from .common import format_example
            new["text"] = format_example(para, row["context"], row["program"])
            f.write(json.dumps(new) + "\n")
            kept += 1
    total = kept + rejected
    rate = rejected / total if total else 0
    print(f"paraphrases kept {kept}, rejected {rejected} "
          f"(rejection rate {rate:.1%}{'  <-- >25%, fix the prompt' if rate > 0.25 else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--key", default=None)
    ap.add_argument("--share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run(a.data, a.endpoint, a.model, a.key, a.share, a.seed)


if __name__ == "__main__":
    main()
