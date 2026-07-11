"""Qualitative CLI-usability check: run a model on natural, OFF-BENCH phrasings over
realistic JSON and show request -> program -> jq result. This is the "is it actually
usable when a human types at it" gate, complementary to the exact-match bench.

Uses the exact same inference path as the shipped `jqgen` CLI, so it measures what a
user would actually get. Works for either backend:

    python -m bench.cli_smoke --backend flagship --model artifacts/nl2jq-40m-hf
    python -m bench.cli_smoke --backend qwen --model artifacts/nl2jq-qwen3-0.6b

Prints a free-form validity rate plus an auto-scored copy-skill rate over novel field
names. None of these appear in nl2jq-bench; phrasings are deliberately casual/varied.
"""
import argparse
import json
import subprocess

import torch

from cli.jqgen import (build_context, generate_flagship, generate_qwen, load_model,
                       resolve_jq, FLAGSHIP_REPO, QWEN_REPO)
from bench.harness import score_program

# (JSON input, natural-language request) — casual phrasings, varied shapes, off-bench.
CASES = [
    ([{"name": "Ada", "dept": "eng", "salary": 120}, {"name": "Béla", "dept": "eng", "salary": 95},
      {"name": "Cy", "dept": "sales", "salary": 80}],
     "who earns the most?"),
    ([{"name": "Ada", "dept": "eng", "salary": 120}, {"name": "Béla", "dept": "eng", "salary": 95},
      {"name": "Cy", "dept": "sales", "salary": 80}],
     "average pay for each department"),
    ([{"sku": "A1", "qty": 3, "price": 10.0}, {"sku": "B2", "qty": 1, "price": 40.0},
      {"sku": "C3", "qty": 5, "price": 2.0}],
     "grand total of qty times price across everything"),
    ([{"user": "sam", "tags": ["urgent", "new"]}, {"user": "kai", "tags": ["new"]},
      {"user": "mo", "tags": ["urgent", "backlog"]}],
     "which users have the urgent tag"),
    ([{"level": "info", "msg": "ok"}, {"level": "error", "msg": "boom"},
      {"level": "error", "msg": "kaput"}, {"level": "warn", "msg": "hmm"}],
     "count the log lines per level"),
    ([{"id": 1, "email": "a@x.com"}, {"id": 2, "email": "b@y.org"}],
     "pull out just the email domains"),
    ([9, 4, 7, 1, 12, 3], "sort these and give me the biggest three"),
    (["2024-01-05", "2023-11-20", "2024-06-30"], "how many dates are here"),
    ({"width": 800, "height": 600, "fps": 30}, "which config value is largest"),
    ({"dark_mode": True, "wifi": False, "sync": True}, "list the settings that are turned on"),
    ([{"repo": "core", "stars": 1200}, {"repo": "cli", "stars": 340}, {"repo": "docs", "stars": 90}],
     "top 2 repos by stars, just the names"),
    ([{"price": 19.99}, {"price": 5.5}, {"price": 120.0}], "add 10% tax to every price"),
    ([{"name": "widget", "stock": 0}, {"name": "gadget", "stock": 12}],
     "only the products that are in stock"),
    (["/usr/local/bin", "/etc/hosts"], "split each path on the slashes"),
    ([{"first": "Grace", "last": "Hopper"}, {"first": "Alan", "last": "Turing"}],
     "full names please, first then last"),
    ([{"team": "red", "pts": 3}, {"team": "red", "pts": 5}, {"team": "blue", "pts": 4}],
     "total points by team"),
    ([{"n": "x", "ok": True}, {"n": "y", "ok": False}, {"n": "z", "ok": True}],
     "give me the first one that isn't ok"),
    ([{"title": "hello world foo"}, {"title": "one two"}],
     "how many words in each title"),
]

# DIRECT phrasings over NOVEL/compound field names, with computable expected outputs — this
# isolates the field-name COPY skill (the request names the field; the model must use that
# exact field from the input, not a memorized one). None of these fields are bench fields.
DIRECT = [
    ([{"vendor": "acme", "unit_price": 10, "units": 3},
      {"vendor": "globex", "unit_price": 40, "units": 1}],
     "the unit_price of each", [10, 40]),
    ([{"vendor": "acme", "unit_price": 10, "units": 3},
      {"vendor": "globex", "unit_price": 40, "units": 1}],
     "total units", [4]),
    ([{"vendor": "acme", "unit_price": 10}, {"vendor": "globex", "unit_price": 40}],
     "vendors sorted by unit_price descending", [["globex", "acme"]]),
    ([{"region": "us-east", "latency": 12}, {"region": "eu-central", "latency": 5}],
     "the region with the lowest latency", [{"region": "eu-central", "latency": 5}]),
    ([{"handle": "a", "streak": 5}, {"handle": "b", "streak": 9}, {"handle": "c", "streak": 2}],
     "handles where streak is over 3", [["a", "b"]]),
    ([{"sku": "x", "daily_stock": 0}, {"sku": "y", "daily_stock": 7}],
     "skus with daily_stock above 0", [["y"]]),
    ([{"assignee": "lea", "priority": "high"}, {"assignee": "sam", "priority": "low"}],
     "assignees whose priority is high", [["lea"]]),
    ([{"city": "Oslo", "avg_temperature": 4}, {"city": "Cairo", "avg_temperature": 30}],
     "the highest avg_temperature", [30]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["flagship", "qwen"], default="flagship")
    ap.add_argument("--model", default=None, help="repo id or local dir (defaults per backend)")
    ap.add_argument("--mode", default="auto")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = a.model or (QWEN_REPO if a.backend == "qwen" else FLAGSHIP_REPO)
    model, tok = load_model(repo, device)
    gen = generate_qwen if a.backend == "qwen" else generate_flagship
    jq = resolve_jq()

    def program(doc, req):
        return gen(model, tok, req, build_context(doc, a.mode), device)

    print(f"===== {a.backend} :: FREE-FORM (validity + eyeball) =====")
    valid = 0
    for doc, req in CASES:
        prog = program(doc, req)
        proc = subprocess.run([jq, prog], input=json.dumps(doc), capture_output=True, text=True)
        ok = proc.returncode == 0
        valid += ok
        out = (proc.stdout.strip() if ok else proc.stderr.strip()).replace("\n", " ")
        print(f"Q: {req}\n   jq: {prog}\n   -> {'OK ' if ok else 'ERR'} {out[:110]}")

    print(f"\n===== {a.backend} :: DIRECT novel-field (copy skill, auto-scored) =====")
    correct = 0
    for doc, req, expected in DIRECT:
        prog = program(doc, req)
        sc = score_program(prog, {"input": doc, "expected_output": expected})
        correct += sc["correct"]
        tag = "PASS" if sc["correct"] else ("wrong" if sc["valid"] else "INVALID")
        print(f"Q: {req}\n   jq: {prog}  -> {tag}")
    print(f"\nfree-form valid: {valid}/{len(CASES)} = {valid / len(CASES):.0%}")
    print(f"direct copy-skill: {correct}/{len(DIRECT)} = {correct / len(DIRECT):.0%}")


if __name__ == "__main__":
    main()
