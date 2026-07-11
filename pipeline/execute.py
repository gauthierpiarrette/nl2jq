"""Run jq programs against documents: the ground-truth verifier."""
import json
import subprocess

from .common import JQ_BIN, canon_outputs

TIMEOUT_S = 1.0
MAX_OUTPUT_BYTES = 256 * 1024


def run_program(program: str, doc_json: str):
    """Execute one jq program on one JSON document.

    Returns (ok, outputs) where outputs is the list of streamed values.
    """
    try:
        proc = subprocess.run(
            [JQ_BIN, "-c", program],
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


def outputs_match(a: list, b: list) -> bool:
    return canon_outputs(a) == canon_outputs(b)
