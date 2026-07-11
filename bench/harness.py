"""nl2jq-bench scoring harness. Standalone: needs only Python 3.10+ and a jq binary.

Item schema (SPEC §4):
  {"id", "request", "input", "expected_output" | "acceptable_outputs",
   "reference_program", "tags", "difficulty", "order_insensitive", "source", "license"}

Scoring: execution match after normalization. Key order always ignored; array order
ignored only when order_insensitive; floats compared with tolerance.

Array/stream equivalence: `map(.name)` (one array output) and `.[].name` (a stream of
N outputs) are both idiomatic jq for "list the names", so we treat a single top-level
array and the corresponding value stream as equal. This is applied symmetrically to the
reference and the candidate, so it neither favors the from-scratch model (trained toward
the array style) nor the frontier baselines (which often emit the stream style).
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

FLOAT_TOL = 1e-9
TIMEOUT_S = 1.0
MAX_OUTPUT_BYTES = 256 * 1024


def _resolve_jq() -> str:
    """JQ_BIN env var > the repo's pinned bin/jq (when run from a checkout) > PATH."""
    if os.environ.get("JQ_BIN"):
        return os.environ["JQ_BIN"]
    pinned = Path(__file__).resolve().parent.parent / "bin" / "jq"
    if pinned.exists():
        return str(pinned)
    found = shutil.which("jq")
    if found:
        return found
    raise FileNotFoundError("jq not found: install jq 1.7.1 or set JQ_BIN")


def run_program(program: str, doc_json: str):
    """Execute one jq program on one JSON document.

    Returns (ok, outputs) where outputs is the list of streamed values.
    """
    jq = _resolve_jq()  # a missing jq is a setup error — fail loudly, don't score invalid
    try:
        proc = subprocess.run(
            [jq, "-c", program],
            input=doc_json, capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, None
    if proc.returncode != 0 or len(proc.stdout) > MAX_OUTPUT_BYTES:
        return False, None
    outputs = []
    for line in proc.stdout.splitlines():
        try:
            outputs.append(json.loads(line))
        except json.JSONDecodeError:
            return False, None
    return True, outputs


def _as_collection(stream):
    """Normalize array/stream packaging: a single output that is a list becomes the
    stream of its elements; everything else is returned unchanged. One level only."""
    if len(stream) == 1 and isinstance(stream[0], list):
        return stream[0]
    return stream


def _norm(v, order_insensitive):
    if isinstance(v, dict):
        return {k: _norm(v[k], order_insensitive) for k in sorted(v)}
    if isinstance(v, list):
        items = [_norm(x, order_insensitive) for x in v]
        if order_insensitive:
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
        return items
    if isinstance(v, float):
        return round(v, 9)
    return v


def outputs_equal(a, b, order_insensitive=False):
    na = [_norm(x, order_insensitive) for x in a]
    nb = [_norm(x, order_insensitive) for x in b]
    if order_insensitive:
        na = sorted(na, key=lambda x: json.dumps(x, sort_keys=True))
        nb = sorted(nb, key=lambda x: json.dumps(x, sort_keys=True))
    if len(na) != len(nb):
        return False
    for x, y in zip(na, nb):
        if isinstance(x, float) and isinstance(y, float):
            if abs(x - y) > FLOAT_TOL:
                return False
        elif x != y:
            return False
    return True


def acceptable(item):
    """Return list of acceptable output-streams for an item."""
    if "acceptable_outputs" in item:
        return item["acceptable_outputs"]
    return [item["expected_output"]]


def _matches(produced, exp, oi):
    if outputs_equal(produced, exp, oi):
        return True
    # accept array<->stream repackaging of the same collection
    return outputs_equal(_as_collection(produced), _as_collection(exp), oi)


def score_program(program, item):
    ok, produced = run_program(program, json.dumps(item["input"]))
    if not ok:
        return {"valid": False, "correct": False}
    oi = item.get("order_insensitive", False)
    correct = any(_matches(produced, exp, oi) for exp in acceptable(item))
    return {"valid": True, "correct": correct}


def score_items(items, generate_fn, k=1):
    """generate_fn(item) -> list of up to k candidate programs. Reports pass@1 & pass@k."""
    pass1 = passk = valid1 = 0
    details = []
    for item in items:
        cands = generate_fn(item)
        results = [score_program(c, item) for c in cands[:k]]
        if results:
            if results[0]["valid"]:
                valid1 += 1
            if results[0]["correct"]:
                pass1 += 1
            if any(r["correct"] for r in results):
                passk += 1
        details.append({"id": item.get("id"), "results": results})
    n = len(items)
    return {"n": n, "pass@1": pass1 / n, f"pass@{k}": passk / n,
            "valid@1": valid1 / n, "details": details}
