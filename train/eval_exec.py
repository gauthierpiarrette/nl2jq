"""Execution-based evaluation: generate a program, run it, compare outputs.

This is the metric that matters — not loss, not exact-match. Reused in-loop during
training and standalone for the M0 gate.
"""
import json

import torch

from pipeline.common import format_example
from pipeline.execute import outputs_match, run_program


@torch.no_grad()
def eval_exec(model, tokenizer, rows, device, max_new=96, temperature=0.0, limit=None):
    model.eval()
    prog_open = "<|program|>"
    eos_id = tokenizer.token_to_id("<|end|>")
    rows = rows[:limit] if limit else rows
    n_ok = n_valid = 0
    fails = []
    for row in rows:
        prompt = format_example(row["request"], row["context"], None)  # ends at <|program|>
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new, eos_id, temperature=temperature)[0].tolist()
        gen_ids = out[len(ids):]
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        program = tokenizer.decode(gen_ids).strip()
        ok, produced = run_program(program, json.dumps(row["input_doc"]))
        if ok:
            n_valid += 1
            if outputs_match(produced, row["expected_output"]):
                n_ok += 1
                continue
        if len(fails) < 8:
            fails.append({"request": row["request"], "gold": row["program"],
                          "pred": program, "valid": ok})
    model.train()
    n = len(rows)
    return {"n": n, "exec_acc": n_ok / n, "valid_rate": n_valid / n, "fails": fails}
