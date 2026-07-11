"""Core single-example generator, shared by build_dataset (serial) and build_parallel.

Produces one fully-verified record or a rejection reason, given an rng. Executes the
sampled program on all sampled docs so behavioral dedup / degeneracy checks can run.
"""
import json
import random

from .common import canon_outputs, format_example
from .documents import sample_documents
from .execute import run_program
from .grammar import sample_task
from .schemas import sample_schema
from .shape import raw_prefix, shape_sketch


def generate_one(rng: random.Random):
    """Returns (record, per_doc_outputs, family) on success, or (None, reason, None)."""
    schema_info = sample_schema(rng)
    docs = sample_documents(schema_info, rng, n_docs=4)
    task = sample_task(schema_info, docs, rng)
    if task is None:
        return None, "no_task", None

    all_outputs = []
    for d in docs:
        ok, out = run_program(task["program"], json.dumps(d))
        if not ok:
            return None, "exec_failed", None
        all_outputs.append(out)

    shown = canon_outputs(all_outputs[0])
    if shown in ("", "null", "[]", "{}", "0", '""'):
        return None, "degenerate_shown", None
    flat = canon_outputs([v for o in all_outputs for v in o])
    if flat in ("", "null", "[]", "{}"):
        return None, "degenerate_output", None
    per_doc = [canon_outputs(o) for o in all_outputs]
    if (len(set(per_doc)) == 1 and not task.get("constant_ok")
            and task["tags"][0] not in ("keys",)):
        return None, "constant_output", None

    doc0 = docs[0]
    # primitive shapes (scalar/nested arrays, bare strings) are small and their programs
    # depend on the actual values/types, so always show them raw (matches inference, and
    # the shape sketch is meaningless for a bare array of numbers).
    use_shape = schema_info["domain"] != "primitive" and rng.random() < 0.5
    context = shape_sketch(doc0) if use_shape else raw_prefix(doc0)
    request = rng.choice(task["nl"])
    record = {
        "request": request,
        "context": context,
        "context_mode": "shape" if use_shape else "raw",
        "program": task["program"],
        "expected_output": all_outputs[0],
        "text": format_example(request, context, task["program"]),
        "tier": task["tier"],
        "tags": task["tags"],
        "domain": schema_info["domain"],
        "family": schema_info["family"],
        "input_doc": doc0,
    }
    return record, tuple(per_doc), schema_info["family"]
